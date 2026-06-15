from __future__ import annotations

from typing import Any

from httpx import AsyncClient
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.portfolio import map_agent, reduce_agent


def _last_user_text(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        for part in getattr(message, "parts", []):
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                return part.content
    return ""


def _page1(prompt: str) -> str:
    """Pull page 1's verbatim text out of a map prompt (so the fake map agent can
    return a quote that will actually verify against the page)."""
    rest = prompt.split("--- Page 1 ---\n", 1)[1]
    for stop in ("\n\n--- ", "\n--- "):
        idx = rest.find(stop)
        if idx != -1:
            rest = rest[:idx]
    return rest.strip()


def _output(info: AgentInfo, payload: dict[str, Any]) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=payload)])


async def _seed_two_docs(
    db: AsyncSession,
) -> tuple[str, tuple[str, str], tuple[str, str]]:
    conversation = Conversation()
    db.add(conversation)
    await db.flush()
    lease = Document(
        conversation_id=conversation.id,
        filename="lease.pdf",
        file_path="/tmp/lease.pdf",
        page_count=1,
    )
    lease.pages = [Page(page_number=1, text="The lease grants parking rights to the tenant.")]
    deed = Document(
        conversation_id=conversation.id,
        filename="deed.pdf",
        file_path="/tmp/deed.pdf",
        page_count=1,
    )
    deed.pages = [Page(page_number=1, text="The deed grants parking rights and a right of way.")]
    db.add_all([lease, deed])
    await db.commit()
    return conversation.id, (lease.id, "lease.pdf"), (deed.id, "deed.pdf")


async def test_portfolio_fans_out_per_document_and_synthesizes_with_citations(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conv_id, (a_id, a_name), (b_id, b_name) = await _seed_two_docs(db_session)

    def map_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        quote = _page1(_last_user_text(messages))
        return _output(info, {"relevant": True, "summary": f"Found: {quote}", "quotes": [{"page": 1, "quote": quote}]})

    def reduce_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _output(
            info,
            {
                "markdown": "Both documents grant parking rights[1][2].",
                "citations": [
                    {"document_id": a_id, "document_name": a_name, "page": 1, "quote": "The lease grants parking rights to the tenant."},
                    {"document_id": b_id, "document_name": b_name, "page": 1, "quote": "The deed grants parking rights and a right of way."},
                ],
            },
        )

    with (
        map_agent.override(model=FunctionModel(map_fn)),
        reduce_agent.override(model=FunctionModel(reduce_fn)),
    ):
        resp = await client.post(
            f"/api/conversations/{conv_id}/portfolio",
            json={"question": "Which documents grant parking rights?"},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()

    # Fan-out: one row per document (the map ran for every doc).
    rows = data["rows"]
    assert {r["document_id"] for r in rows} == {a_id, b_id}
    assert all(r["relevant"] for r in rows)
    # Map grounding: each row carries its own verified citation.
    assert all(len(r["citations"]) >= 1 for r in rows)
    # Reduce grounding: the synthesised answer cites both, markers intact.
    assert len(data["citations"]) == 2
    assert "[1]" in data["markdown"] and "[2]" in data["markdown"]


async def test_portfolio_excludes_irrelevant_documents(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.flush()
    lease = Document(
        conversation_id=conversation.id,
        filename="lease.pdf",
        file_path="/tmp/lease.pdf",
        page_count=1,
    )
    lease.pages = [Page(page_number=1, text="The lease grants parking rights to the tenant.")]
    memo = Document(
        conversation_id=conversation.id,
        filename="memo.pdf",
        file_path="/tmp/memo.pdf",
        page_count=1,
    )
    memo.pages = [Page(page_number=1, text="Internal memo regarding the office coffee machine.")]
    db_session.add_all([lease, memo])
    await db_session.commit()
    conv_id, lease_id, memo_id = conversation.id, lease.id, memo.id

    def map_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        page1 = _page1(_last_user_text(messages))  # this document's page-1 text
        relevant = "parking" in page1.lower()
        quotes = [{"page": 1, "quote": page1}] if relevant else []
        return _output(info, {"relevant": relevant, "summary": "Grants parking." if relevant else "", "quotes": quotes})

    def reduce_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return _output(
            info,
            {
                "markdown": "Only the lease grants parking rights[1].",
                "citations": [
                    {"document_id": lease_id, "document_name": "lease.pdf", "page": 1, "quote": "The lease grants parking rights to the tenant."}
                ],
            },
        )

    with (
        map_agent.override(model=FunctionModel(map_fn)),
        reduce_agent.override(model=FunctionModel(reduce_fn)),
    ):
        resp = await client.post(
            f"/api/conversations/{conv_id}/portfolio",
            json={"question": "Which documents grant parking rights?"},
        )

    assert resp.status_code == 200, resp.text
    data = resp.json()
    by_id = {r["document_id"]: r for r in data["rows"]}
    # Both documents are reported as rows, with their relevance flags...
    assert by_id[lease_id]["relevant"] is True
    assert by_id[memo_id]["relevant"] is False
    assert by_id[memo_id]["citations"] == []
    # ...but only the relevant document reaches the synthesised answer.
    assert len(data["citations"]) == 1
    assert data["citations"][0]["document_id"] == lease_id


async def test_portfolio_404_for_unknown_conversation(client: AsyncClient) -> None:
    resp = await client.post(
        "/api/conversations/does-not-exist/portfolio",
        json={"question": "anything"},
    )
    assert resp.status_code == 404
