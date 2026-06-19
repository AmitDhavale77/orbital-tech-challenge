from __future__ import annotations

import json
from typing import Any

from httpx import AsyncClient
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.document import grep_pages
from takehome.services.llm import qa_agent


def _parse_sse(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def _tool_returns(messages: list[ModelMessage]) -> int:
    return sum(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in getattr(message, "parts", [])
    )


async def _seed_bundle(db: AsyncSession) -> tuple[str, str, str]:
    conversation = Conversation()
    db.add(conversation)
    await db.flush()
    lease = Document(
        conversation_id=conversation.id, filename="lease.pdf", file_path="/tmp/lease.pdf", page_count=2
    )
    lease.pages = [
        Page(page_number=1, text="The Initial Rent is GBP 850,000 per annum."),
        Page(page_number=2, text="The Tenant may break on 6 months notice."),
    ]
    deed = Document(
        conversation_id=conversation.id, filename="deed.pdf", file_path="/tmp/deed.pdf", page_count=1
    )
    deed.pages = [Page(page_number=1, text="Rent reduced to a peppercorn.")]
    db.add_all([lease, deed])
    await db.commit()
    return conversation.id, lease.id, deed.id


# --- grep_pages service --- #


async def test_grep_finds_a_term_across_the_bundle(db_session: AsyncSession) -> None:
    conversation_id, _, deed_id = await _seed_bundle(db_session)
    hits = await grep_pages(db_session, conversation_id, "peppercorn")
    assert len(hits) == 1
    assert hits[0]["document_id"] == deed_id and hits[0]["page"] == 1
    assert "peppercorn" in str(hits[0]["line"]).lower()


async def test_grep_is_case_insensitive_and_spans_documents(
    db_session: AsyncSession,
) -> None:
    conversation_id, lease_id, deed_id = await _seed_bundle(db_session)
    hits = await grep_pages(db_session, conversation_id, "rent")  # "Rent" and "Rent"
    docs = {h["document_id"] for h in hits}
    assert lease_id in docs and deed_id in docs


async def test_grep_can_scope_to_one_document(db_session: AsyncSession) -> None:
    conversation_id, lease_id, deed_id = await _seed_bundle(db_session)
    hits = await grep_pages(db_session, conversation_id, "rent", document_id=lease_id)
    assert {h["document_id"] for h in hits} == {lease_id}


async def test_grep_supports_regex_whitespace_within_a_line(
    db_session: AsyncSession,
) -> None:
    conversation_id, lease_id, _ = await _seed_bundle(db_session)
    hits = await grep_pages(db_session, conversation_id, r"Initial\s+Rent")
    assert any(h["document_id"] == lease_id and h["page"] == 1 for h in hits)


async def test_grep_returns_empty_when_nothing_matches(
    db_session: AsyncSession,
) -> None:
    conversation_id, _, _ = await _seed_bundle(db_session)
    assert await grep_pages(db_session, conversation_id, "nonexistent-term-xyz") == []


# --- grep -> read_pages -> cite (HTTP seam) --- #


async def test_agent_greps_then_reads_then_cites(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conversation_id, lease_id, _ = await _seed_bundle(db_session)
    calls: list[str] = []

    async def stream_function(messages: list[ModelMessage], info: AgentInfo):
        seen = _tool_returns(messages)
        if seen == 0:
            calls.append("grep")
            yield {0: DeltaToolCall(name="grep", json_args=json.dumps({"pattern": "rent"}))}
        elif seen == 1:
            calls.append("read_pages")
            yield {0: DeltaToolCall(name="read_pages", json_args=json.dumps({"document_id": lease_id, "start_page": 1}))}
        else:
            yield {
                0: DeltaToolCall(
                    name=info.output_tools[0].name,
                    json_args=json.dumps(
                        {
                            "markdown": "The rent is GBP 850,000.[1]",
                            "citations": [{"document_id": lease_id, "document_name": "lease.pdf", "page": 1, "quote": "The Initial Rent is GBP 850,000 per annum."}],
                        }
                    ),
                )
            }

    with qa_agent.override(model=FunctionModel(stream_function=stream_function)):
        response = await client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "What is the rent?"},
        )

    assert response.status_code == 200
    assert calls == ["grep", "read_pages"]
    events = _parse_sse(response.text)
    # grep surfaces as a search step.
    steps = [e for e in events if e.get("type") == "step"]
    assert steps[0]["kind"] == "search"
    assert "rent" in steps[0]["label"]
    message = next(e["message"] for e in events if e.get("type") == "message")
    assert message["citations"][0]["page"] == 1
