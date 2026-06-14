from __future__ import annotations

import json
from typing import Any

from httpx import AsyncClient
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.document import search_pages
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

    # A long lease that mentions "rent" on every one of its 5 pages.
    lease = Document(
        conversation_id=conversation.id,
        filename="lease.pdf",
        file_path="/tmp/lease.pdf",
        page_count=5,
    )
    lease.pages = [Page(page_number=i + 1, text="rent " * 50) for i in range(5)]

    # A short deed that mentions "rent" once.
    deed = Document(
        conversation_id=conversation.id,
        filename="deed.pdf",
        file_path="/tmp/deed.pdf",
        page_count=1,
    )
    deed.pages = [Page(page_number=1, text="the rent was varied to a peppercorn")]
    db.add_all([lease, deed])
    await db.commit()
    return conversation.id, lease.id, deed.id


async def test_search_ranks_and_diversifies_across_documents(
    db_session: AsyncSession,
) -> None:
    conversation_id, lease_id, deed_id = await _seed_bundle(db_session)

    results = await search_pages(
        db_session, conversation_id, "rent", per_document=2, limit=8
    )

    assert results, "expected hits"
    assert set(results[0]) >= {"document_id", "document_name", "page", "preview"}
    # Highest-ranked hit is from the lease (more frequent term).
    assert results[0]["document_id"] == lease_id
    # Per-document cap: the long lease can't take more than 2 slots...
    assert sum(r["document_id"] == lease_id for r in results) <= 2
    # ...so the short deed still surfaces despite ranking lower.
    assert any(r["document_id"] == deed_id for r in results)


async def test_preview_contains_the_matched_keyword(db_session: AsyncSession) -> None:
    conversation_id, _, _ = await _seed_bundle(db_session)
    results = await search_pages(db_session, conversation_id, "peppercorn")
    hit = next(r for r in results if "peppercorn" in str(r["preview"]).lower())
    assert "peppercorn" in str(hit["preview"]).lower()


async def test_search_is_scoped_to_the_conversation(db_session: AsyncSession) -> None:
    conversation_id, _, _ = await _seed_bundle(db_session)
    _, other_lease_id, _ = await _seed_bundle(db_session)

    results = await search_pages(db_session, conversation_id, "rent")
    assert other_lease_id not in {r["document_id"] for r in results}


async def test_agent_searches_then_reads(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conversation_id, lease_id, _ = await _seed_bundle(db_session)
    calls: list[str] = []

    async def stream_function(messages: list[ModelMessage], info: AgentInfo):
        seen = _tool_returns(messages)
        if seen == 0:
            calls.append("search")
            yield {0: DeltaToolCall(name="search", json_args=json.dumps({"query": "rent"}))}
        elif seen == 1:
            calls.append("read_page")
            yield {0: DeltaToolCall(name="read_page", json_args=json.dumps({"document_id": lease_id, "page": 1}))}
        else:
            payload = {
                "markdown": "The lease discusses rent.[1]",
                "citations": [
                    {"document_id": lease_id, "document_name": "lease.pdf", "page": 1, "quote": "rent rent rent"}
                ],
            }
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args=json.dumps(payload))}

    with qa_agent.override(model=FunctionModel(stream_function=stream_function)):
        response = await client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "What does the lease say about rent?"},
        )

    assert response.status_code == 200
    assert calls[:2] == ["search", "read_page"]  # search narrows, then read confirms
    events = _parse_sse(response.text)
    message = next(e["message"] for e in events if e.get("type") == "message")
    assert message["citations"][0]["document_id"] == lease_id
