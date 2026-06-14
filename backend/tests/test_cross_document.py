from __future__ import annotations

import io
import json
from typing import Any

import pymupdf
from httpx import AsyncClient
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.llm import qa_agent


def _pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    return doc.tobytes()


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


async def test_conversation_accepts_multiple_uploads(client: AsyncClient) -> None:
    conv = (await client.post("/api/conversations")).json()
    cid = conv["id"]

    for name in ("lease.pdf", "deed.pdf", "title.pdf"):
        resp = await client.post(
            f"/api/conversations/{cid}/documents",
            files={"file": (name, _pdf(f"contents of {name}"), "application/pdf")},
        )
        assert resp.status_code == 201, resp.text  # no 409 "already has a document"

    detail = (await client.get(f"/api/conversations/{cid}")).json()
    assert len(detail["documents"]) == 3
    assert {d["filename"] for d in detail["documents"]} == {
        "lease.pdf",
        "deed.pdf",
        "title.pdf",
    }


async def test_answer_cites_across_multiple_documents(
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
    lease.pages = [Page(page_number=1, text="The rent is GBP 1.75 million per annum.")]
    deed = Document(
        conversation_id=conversation.id,
        filename="deed-of-variation.pdf",
        file_path="/tmp/deed.pdf",
        page_count=1,
    )
    deed.pages = [Page(page_number=1, text="The rent was varied to a peppercorn.")]
    db_session.add_all([lease, deed])
    await db_session.commit()

    async def stream_function(messages: list[ModelMessage], info: AgentInfo):
        seen = _tool_returns(messages)
        if seen == 0:
            yield {0: DeltaToolCall(name="read_page", json_args=json.dumps({"document_id": lease.id, "page": 1}))}
        elif seen == 1:
            yield {0: DeltaToolCall(name="read_page", json_args=json.dumps({"document_id": deed.id, "page": 1}))}
        else:
            payload = {
                "markdown": "Rent is GBP 1.75m[1], previously a peppercorn[2].",
                "citations": [
                    {"document_id": lease.id, "document_name": "lease.pdf", "page": 1, "quote": "The rent is GBP 1.75 million"},
                    {"document_id": deed.id, "document_name": "deed-of-variation.pdf", "page": 1, "quote": "varied to a peppercorn"},
                ],
            }
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args=json.dumps(payload))}

    with qa_agent.override(model=FunctionModel(stream_function=stream_function)):
        response = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "What is the rent now and what was it before?"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    message = next(e["message"] for e in events if e.get("type") == "message")
    cited_docs = {c["document_id"] for c in message["citations"]}
    assert cited_docs == {lease.id, deed.id}
    assert len(message["citations"]) == 2
