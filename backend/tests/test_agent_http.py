from __future__ import annotations

import json
from typing import Any

from httpx import AsyncClient
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.llm import qa_agent

# The chat agent finishes by emitting a structured `Answer` (markdown + citations)
# as its final output tool — the answer is NOT streamed token-by-token. So a
# FunctionModel drives the loop as: read_pages -> Answer (with citations).

_QUOTE = "the rent is GBP 1.75 million"


def _parse_sse(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _has_tool_return(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in getattr(message, "parts", [])
    )


def _returns(messages: list[ModelMessage]) -> list[str]:
    """Tool names that have already returned (incl. any replayed history)."""
    return [
        part.tool_name
        for message in messages
        for part in getattr(message, "parts", [])
        if isinstance(part, ToolReturnPart)
    ]


def _read_call(document: Document) -> dict[int, DeltaToolCall]:
    return {
        0: DeltaToolCall(
            name="read_pages",
            json_args=json.dumps({"document_id": document.id, "start_page": 1}),
        )
    }


def _answer_call(info: AgentInfo, document: Document) -> dict[int, DeltaToolCall]:
    """The final structured-Answer output tool, citing page 1's rent line."""
    return {
        0: DeltaToolCall(
            name=info.output_tools[0].name,
            json_args=json.dumps(
                {
                    "markdown": "The rent is GBP 1.75 million [1].",
                    "citations": [
                        {
                            "document_id": document.id,
                            "document_name": "lease.pdf",
                            "page": 1,
                            "quote": _QUOTE,
                        }
                    ],
                }
            ),
        )
    }


async def _seed_doc(db_session: AsyncSession) -> tuple[Conversation, Document]:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.flush()
    document = Document(
        conversation_id=conversation.id,
        filename="lease.pdf",
        file_path="/tmp/lease.pdf",
        page_count=2,
        extracted_text="blob",
    )
    document.pages = [
        Page(page_number=1, text="SECRET_PAGE_ONE the rent is GBP 1.75 million"),
        Page(page_number=2, text="SECRET_PAGE_TWO the break clause"),
    ]
    db_session.add(document)
    await db_session.commit()
    return conversation, document


async def _first_turn(
    client: AsyncClient, conversation: Conversation, document: Document
) -> None:
    """Drive one read -> answer turn so the conversation gains rich history."""

    async def fn(messages: list[ModelMessage], info: AgentInfo):
        if "read_pages" not in _returns(messages):
            yield _read_call(document)
        else:
            yield _answer_call(info, document)

    with qa_agent.override(model=FunctionModel(stream_function=fn)):
        response = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "What is the rent?"},
        )
    assert response.status_code == 200


async def test_agent_answers_by_reading_pages(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conversation, document = await _seed_doc(db_session)
    seen_requests: list[list[ModelMessage]] = []

    async def stream_function(messages: list[ModelMessage], info: AgentInfo):
        seen_requests.append(messages)
        if "read_pages" not in _returns(messages):
            yield _read_call(document)
        else:
            yield _answer_call(info, document)

    with qa_agent.override(model=FunctionModel(stream_function=stream_function)):
        response = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "What is the rent?"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)

    # The agent actually read a page, and no document text was placed in the prompt
    # before the tool ran.
    assert len(seen_requests) >= 2
    assert "SECRET_PAGE_ONE" not in repr(seen_requests[0])

    # The assistant message persisted, carries the answer, and is grounded.
    final = [e for e in events if e.get("type") == "message"]
    assert final, "expected a final message event"
    assert final[0]["message"]["role"] == "assistant"
    assert "GBP 1.75 million" in final[0]["message"]["content"]
    assert final[0]["message"]["sources_cited"] >= 1


# --------------------------------------------------------------------------- #
# Rich ModelMessage history -> reuse on repeat + compaction (ticket 08)
# --------------------------------------------------------------------------- #


async def test_rich_history_persisted_with_tool_returns(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    # After a turn that read a page, the conversation stores the FULL ModelMessage
    # snapshot — not just plain text — including the tool return (the page text the
    # agent read). That earned context is what lets a repeat answer without re-reading.
    conversation, document = await _seed_doc(db_session)
    await _first_turn(client, conversation, document)

    await db_session.refresh(conversation)
    assert conversation.model_history, "expected model_history to be persisted"
    history = ModelMessagesTypeAdapter.validate_python(conversation.model_history)
    assert _has_tool_return(history), "snapshot must carry the prior tool return"


async def test_history_replayed_into_next_turn(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    # The stored snapshot is replayed: the model's very first request on turn 2
    # already carries the prior turn's tool returns (rich history, not plain text).
    conversation, document = await _seed_doc(db_session)
    await _first_turn(client, conversation, document)

    seen: list[list[ModelMessage]] = []

    async def fn(messages: list[ModelMessage], info: AgentInfo):
        seen.append(messages)
        yield _answer_call(info, document)

    with qa_agent.override(model=FunctionModel(stream_function=fn)):
        response = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "What is the rent?"},
        )

    assert response.status_code == 200
    assert seen and _has_tool_return(seen[0]), "turn 2 must receive replayed reads"


async def test_repeat_question_reuses_context_without_rereading(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    # With the prior reads replayed into context, a repeated question is answered
    # directly — no new read/search step — yet still cites a quote that is
    # re-verified against the live page (grounding preserved on reuse, ADR-0002).
    conversation, document = await _seed_doc(db_session)
    await _first_turn(client, conversation, document)

    async def fn(messages: list[ModelMessage], info: AgentInfo):
        yield _answer_call(info, document)  # answer straight from replayed context

    with qa_agent.override(model=FunctionModel(stream_function=fn)):
        response = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "What is the rent?"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)

    steps = [e for e in events if e.get("type") == "step"]
    assert not any(s.get("kind") in {"read", "search"} for s in steps), (
        "a repeated question must not re-read pages already in context"
    )
    final = next(e for e in events if e.get("type") == "message")["message"]
    assert final["sources_cited"] >= 1, "reused answer must still be grounded"


def test_compaction_capability_is_configured() -> None:
    # Anthropic server-side compaction is wired so long chats stay within the
    # context window (ticket 08); it only round-trips because we persist the full
    # ModelMessage history (tested above).
    from pydantic_ai.models.anthropic import AnthropicCompaction

    found: list[object] = []
    qa_agent._root_capability.apply(found.append)  # noqa: SLF001
    compactions = [c for c in found if isinstance(c, AnthropicCompaction)]
    assert len(compactions) == 1
    assert compactions[0].token_threshold == 300_000
