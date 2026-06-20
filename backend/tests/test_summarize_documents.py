from __future__ import annotations

import json
from typing import Any

import pytest
from httpx import AsyncClient
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from takehome.db.models import Conversation, Document, Page
from takehome.services.llm import qa_agent
from takehome.services.portfolio import DocFinding, map_agent, map_documents
from tests.helpers import last_user_text, page1, parse_sse, tool_returns

LEASE_P1 = "The lease grants parking rights to the tenant."
DEED_P1 = "The deed grants parking rights and a right of way."


def _map_returns_its_page1(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A fake per-document map: relevant, quoting this document's verbatim page 1."""
    quote = page1(last_user_text(messages))
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=info.output_tools[0].name,
                args={"relevant": True, "summary": f"Found: {quote}", "quotes": [{"page": 1, "quote": quote}]},
            )
        ]
    )


async def _seed_two_docs(db: AsyncSession) -> tuple[str, str, str]:
    conversation = Conversation()
    db.add(conversation)
    await db.flush()
    lease = Document(
        conversation_id=conversation.id, filename="lease.pdf", file_path="/tmp/lease.pdf", page_count=1
    )
    lease.pages = [Page(page_number=1, text=LEASE_P1)]
    deed = Document(
        conversation_id=conversation.id, filename="deed.pdf", file_path="/tmp/deed.pdf", page_count=1
    )
    deed.pages = [Page(page_number=1, text=DEED_P1)]
    db.add_all([lease, deed])
    await db.commit()
    return conversation.id, lease.id, deed.id


async def test_breadth_question_fans_out_via_summarize_documents_and_cites(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conv_id, lease_id, deed_id = await _seed_two_docs(db_session)
    tool_calls: list[str] = []

    async def chat_fn(messages: list[ModelMessage], info: AgentInfo):
        if tool_returns(messages) == 0:
            tool_calls.append("summarize_documents")
            yield {
                0: DeltaToolCall(
                    name="summarize_documents",
                    json_args=json.dumps(
                        {
                            "question": "Which documents grant parking rights?",
                            "document_ids": [lease_id, deed_id],
                        }
                    ),
                )
            }
        else:
            # The chat agent IS the reduce: it cites the verbatim quotes the
            # fan-out returned (copied character-for-character).
            yield {
                0: DeltaToolCall(
                    name=info.output_tools[0].name,
                    json_args=json.dumps(
                        {
                            "markdown": "Both documents grant parking rights[1][2].",
                            "citations": [
                                {"document_id": lease_id, "document_name": "lease.pdf", "page": 1, "quote": LEASE_P1},
                                {"document_id": deed_id, "document_name": "deed.pdf", "page": 1, "quote": DEED_P1},
                            ],
                        }
                    ),
                )
            }

    with (
        qa_agent.override(model=FunctionModel(stream_function=chat_fn)),
        map_agent.override(model=FunctionModel(_map_returns_its_page1)),
    ):
        response = await client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": "Summarise which documents grant parking rights."},
        )

    assert response.status_code == 200, response.text
    # Routing: the breadth question went through the new breadth tool.
    assert tool_calls == ["summarize_documents"]
    events = parse_sse(response.text)
    # Grounding survives the fan-out → reduce hop: both per-doc quotes verify.
    message = next(e["message"] for e in events if e.get("type") == "message")
    assert {c["document_id"] for c in message["citations"]} == {lease_id, deed_id}
    assert message["citations"][0]["page"] == 1


@pytest.mark.parametrize("bad_ids", [[], ["nope-1", "nope-2"]], ids=["empty", "all-bogus"])
async def test_summarize_documents_steers_when_no_valid_ids(
    bad_ids: list[str], client: AsyncClient, db_session: AsyncSession
) -> None:
    conv_id, lease_id, _ = await _seed_two_docs(db_session)
    saw_retry: list[bool] = []
    calls: list[int] = []

    async def chat_fn(messages: list[ModelMessage], info: AgentInfo):
        calls.append(1)
        if any(
            isinstance(part, RetryPromptPart)
            for message in messages
            for part in getattr(message, "parts", [])
        ):
            saw_retry.append(True)
        attempt = len(calls)
        if attempt == 1:
            # No VALID document_ids (empty or all-bogus) — steer, not silently no-op.
            yield {
                0: DeltaToolCall(
                    name="summarize_documents",
                    json_args=json.dumps({"question": "?", "document_ids": bad_ids}),
                )
            }
        elif attempt == 2:
            # Corrected after the steer.
            yield {
                0: DeltaToolCall(
                    name="summarize_documents",
                    json_args=json.dumps({"question": "Does it grant parking?", "document_ids": [lease_id]}),
                )
            }
        else:
            yield {
                0: DeltaToolCall(
                    name=info.output_tools[0].name,
                    json_args=json.dumps(
                        {
                            "markdown": "The lease grants parking[1].",
                            "citations": [
                                {"document_id": lease_id, "document_name": "lease.pdf", "page": 1, "quote": LEASE_P1}
                            ],
                        }
                    ),
                )
            }

    with (
        qa_agent.override(model=FunctionModel(stream_function=chat_fn)),
        map_agent.override(model=FunctionModel(_map_returns_its_page1)),
    ):
        response = await client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": "Summarise the bundle."},
        )

    assert response.status_code == 200, response.text
    # The empty-ids call bounced back a ModelRetry (not a silent empty result),
    # so the agent saw a retry prompt before correcting and answering.
    assert saw_retry, "empty document_ids should be steered with a ModelRetry"
    events = parse_sse(response.text)
    message = next(e["message"] for e in events if e.get("type") == "message")
    assert message["citations"][0]["document_id"] == lease_id


async def test_summarize_documents_proceeds_with_partially_valid_ids(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conv_id, lease_id, _ = await _seed_two_docs(db_session)
    saw_retry: list[bool] = []

    async def chat_fn(messages: list[ModelMessage], info: AgentInfo):
        if any(
            isinstance(part, RetryPromptPart)
            for message in messages
            for part in getattr(message, "parts", [])
        ):
            saw_retry.append(True)
        if tool_returns(messages) == 0:
            # One real id + one bogus id — the bogus id is skipped, the call proceeds.
            yield {
                0: DeltaToolCall(
                    name="summarize_documents",
                    json_args=json.dumps(
                        {"question": "parking?", "document_ids": [lease_id, "bogus-id"]}
                    ),
                )
            }
        else:
            yield {
                0: DeltaToolCall(
                    name=info.output_tools[0].name,
                    json_args=json.dumps(
                        {
                            "markdown": "The lease grants parking[1].",
                            "citations": [
                                {"document_id": lease_id, "document_name": "lease.pdf", "page": 1, "quote": LEASE_P1}
                            ],
                        }
                    ),
                )
            }

    with (
        qa_agent.override(model=FunctionModel(stream_function=chat_fn)),
        map_agent.override(model=FunctionModel(_map_returns_its_page1)),
    ):
        response = await client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": "Does the lease grant parking?"},
        )

    assert response.status_code == 200, response.text
    # A list with >=1 valid id proceeds (the bogus id is silently skipped) — no steer.
    assert not saw_retry
    events = parse_sse(response.text)
    message = next(e["message"] for e in events if e.get("type") == "message")
    assert message["citations"][0]["document_id"] == lease_id


def _map_relevant_if_parking(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    """A fake map that is relevant only if this document's page 1 mentions parking."""
    page_text = page1(last_user_text(messages))
    relevant = "parking" in page_text.lower()
    return ModelResponse(
        parts=[
            ToolCallPart(
                tool_name=info.output_tools[0].name,
                args={
                    "relevant": relevant,
                    "summary": "Grants parking." if relevant else "",
                    "quotes": [{"page": 1, "quote": page_text}] if relevant else [],
                },
            )
        ]
    )


async def test_irrelevant_document_is_excluded_from_breadth_answer(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.flush()
    lease = Document(
        conversation_id=conversation.id, filename="lease.pdf", file_path="/tmp/lease.pdf", page_count=1
    )
    lease.pages = [Page(page_number=1, text=LEASE_P1)]
    memo = Document(
        conversation_id=conversation.id, filename="memo.pdf", file_path="/tmp/memo.pdf", page_count=1
    )
    memo.pages = [Page(page_number=1, text="Internal memo about the office coffee machine.")]
    db_session.add_all([lease, memo])
    await db_session.commit()
    conv_id, lease_id, memo_id = conversation.id, lease.id, memo.id

    summarize_payloads: list[Any] = []

    async def chat_fn(messages: list[ModelMessage], info: AgentInfo):
        for message in messages:
            for part in getattr(message, "parts", []):
                if isinstance(part, ToolReturnPart) and part.tool_name == "summarize_documents":
                    content = part.content
                    summarize_payloads.append(
                        json.loads(content) if isinstance(content, str) else content
                    )
        if tool_returns(messages) == 0:
            yield {
                0: DeltaToolCall(
                    name="summarize_documents",
                    json_args=json.dumps(
                        {"question": "Which grant parking?", "document_ids": [lease_id, memo_id]}
                    ),
                )
            }
        else:
            # The chat agent excludes the irrelevant memo and cites only the lease.
            yield {
                0: DeltaToolCall(
                    name=info.output_tools[0].name,
                    json_args=json.dumps(
                        {
                            "markdown": "Only the lease grants parking[1].",
                            "citations": [
                                {"document_id": lease_id, "document_name": "lease.pdf", "page": 1, "quote": LEASE_P1}
                            ],
                        }
                    ),
                )
            }

    with (
        qa_agent.override(model=FunctionModel(stream_function=chat_fn)),
        map_agent.override(model=FunctionModel(_map_relevant_if_parking)),
    ):
        response = await client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": "Which documents grant parking rights?"},
        )

    assert response.status_code == 200, response.text
    # The tool surfaces the memo as relevant=false with no quotes...
    findings = {f["document_id"]: f for f in summarize_payloads[0]["findings"]}
    assert findings[lease_id]["relevant"] is True and findings[lease_id]["quotes"]
    assert findings[memo_id]["relevant"] is False and findings[memo_id]["quotes"] == []
    # ...and the irrelevant memo never reaches the final answer's citations.
    msg = next(e["message"] for e in parse_sse(response.text) if e.get("type") == "message")
    assert {c["document_id"] for c in msg["citations"]} == {lease_id}


async def test_map_documents_keeps_bracketed_summary_text_verbatim(
    sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the map summary is plain prose; its incidental bracketed numbers
    (e.g. a year or a schedule ref) must NOT be rewritten by the citation-marker pass."""
    import takehome.db.session as db_session_module

    monkeypatch.setattr(db_session_module, "async_session", sessionmaker)
    async with sessionmaker() as session:
        conversation = Conversation()
        session.add(conversation)
        await session.flush()
        lease = Document(
            conversation_id=conversation.id, filename="lease.pdf", file_path="/tmp/lease.pdf", page_count=1
        )
        lease.pages = [Page(page_number=1, text=LEASE_P1)]
        session.add(lease)
        await session.commit()
        conv_id, lease_id = conversation.id, lease.id

    def map_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        quote = page1(last_user_text(messages))
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name,
                    args={
                        "relevant": True,
                        "summary": "Rent reviewed in [2024]; see Schedule [4].",
                        "quotes": [{"page": 1, "quote": quote}],
                    },
                )
            ]
        )

    with map_agent.override(model=FunctionModel(map_fn)):
        findings = await map_documents(conv_id, "rent?", [lease_id])

    assert len(findings) == 1
    assert findings[0].summary == "Rent reviewed in [2024]; see Schedule [4]."
    assert [c.quote for c in findings[0].citations] == [LEASE_P1]


async def test_map_documents_reports_each_document_via_on_doc(
    sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The breadth path streams per-document progress: `on_doc` fires once per
    mapped document with its (id, name), backing the live chat summarize steps."""
    import takehome.db.session as db_session_module

    monkeypatch.setattr(db_session_module, "async_session", sessionmaker)
    async with sessionmaker() as session:
        conv_id, lease_id, deed_id = await _seed_two_docs(session)

    seen: list[tuple[str, str]] = []
    with map_agent.override(model=FunctionModel(_map_returns_its_page1)):
        await map_documents(
            conv_id,
            "Which documents grant parking rights?",
            [lease_id, deed_id],
            on_doc=lambda doc_id, name: seen.append((doc_id, name)),
        )

    assert sorted(seen) == sorted(
        [(lease_id, "lease.pdf"), (deed_id, "deed.pdf")]
    )


async def test_map_documents_reports_verdict_via_on_done(
    sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`on_done` fires once per document with its completed verdict: a relevant doc
    carries its verified quotes, an irrelevant one carries none. This backs the
    per-document verdict step the chat tool streams after each map."""
    import takehome.db.session as db_session_module

    monkeypatch.setattr(db_session_module, "async_session", sessionmaker)
    async with sessionmaker() as session:
        conversation = Conversation()
        session.add(conversation)
        await session.flush()
        lease = Document(
            conversation_id=conversation.id, filename="lease.pdf", file_path="/tmp/lease.pdf", page_count=1
        )
        lease.pages = [Page(page_number=1, text=LEASE_P1)]
        memo = Document(
            conversation_id=conversation.id, filename="memo.pdf", file_path="/tmp/memo.pdf", page_count=1
        )
        memo.pages = [Page(page_number=1, text="Internal memo about the office coffee machine.")]
        session.add_all([lease, memo])
        await session.commit()
        conv_id, lease_id, memo_id = conversation.id, lease.id, memo.id

    done: list[DocFinding] = []
    with map_agent.override(model=FunctionModel(_map_relevant_if_parking)):
        await map_documents(
            conv_id, "Which grant parking?", [lease_id, memo_id], on_done=done.append
        )

    by_id = {f.document_id: f for f in done}
    assert set(by_id) == {lease_id, memo_id}
    assert by_id[lease_id].relevant is True and by_id[lease_id].citations
    assert by_id[memo_id].relevant is False and not by_id[memo_id].citations


async def test_breadth_path_streams_per_document_verdict_steps(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """The breadth path streams a verdict step per document (not just a single
    umbrella call): a "relevant · N quote(s)" / "not relevant" label plus the map
    summary as `detail`, so the agent's per-document reasoning is visible live."""
    conv_id, lease_id, deed_id = await _seed_two_docs(db_session)

    async def chat_fn(messages: list[ModelMessage], info: AgentInfo):
        if tool_returns(messages) == 0:
            yield {
                0: DeltaToolCall(
                    name="summarize_documents",
                    json_args=json.dumps(
                        {"question": "parking?", "document_ids": [lease_id, deed_id]}
                    ),
                )
            }
        else:
            yield {
                0: DeltaToolCall(
                    name=info.output_tools[0].name,
                    json_args=json.dumps(
                        {
                            "markdown": "Both grant parking[1][2].",
                            "citations": [
                                {"document_id": lease_id, "document_name": "lease.pdf", "page": 1, "quote": LEASE_P1},
                                {"document_id": deed_id, "document_name": "deed.pdf", "page": 1, "quote": DEED_P1},
                            ],
                        }
                    ),
                )
            }

    with (
        qa_agent.override(model=FunctionModel(stream_function=chat_fn)),
        map_agent.override(model=FunctionModel(_map_returns_its_page1)),
    ):
        response = await client.post(
            f"/api/conversations/{conv_id}/messages",
            json={"content": "Which documents grant parking rights?"},
        )

    assert response.status_code == 200, response.text
    events = parse_sse(response.text)
    summarize = [
        e for e in events if e.get("type") == "step" and e.get("kind") == "summarize"
    ]
    verdicts = [e for e in summarize if "relevant" in e["label"]]
    # One verdict step per document, each carrying its single verified quote count
    # and the map summary as detail.
    assert len(verdicts) == 2
    assert {e["document_id"] for e in verdicts} == {lease_id, deed_id}
    assert all("relevant · 1 quote" in e["label"] for e in verdicts)
    assert all(e.get("detail") for e in verdicts)
