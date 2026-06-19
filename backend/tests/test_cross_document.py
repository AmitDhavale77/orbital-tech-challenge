from __future__ import annotations

import json

from httpx import AsyncClient
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.llm import qa_agent
from tests.helpers import make_pdf, parse_sse, tool_returns


async def test_conversation_accepts_multiple_uploads(client: AsyncClient) -> None:
    conv = (await client.post("/api/conversations")).json()
    cid = conv["id"]

    for name in ("lease.pdf", "deed.pdf", "title.pdf"):
        resp = await client.post(
            f"/api/conversations/{cid}/documents",
            files={"file": (name, make_pdf(f"contents of {name}"), "application/pdf")},
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
        seen = tool_returns(messages)
        if seen == 0:
            yield {0: DeltaToolCall(name="read_pages", json_args=json.dumps({"document_id": lease.id, "start_page": 1}))}
        elif seen == 1:
            yield {0: DeltaToolCall(name="read_pages", json_args=json.dumps({"document_id": deed.id, "start_page": 1}))}
        else:
            yield {
                0: DeltaToolCall(
                    name=info.output_tools[0].name,
                    json_args=json.dumps(
                        {
                            "markdown": "Rent is GBP 1.75m[1], previously a peppercorn[2].",
                            "citations": [
                                {"document_id": lease.id, "document_name": "lease.pdf", "page": 1, "quote": "The rent is GBP 1.75 million"},
                                {"document_id": deed.id, "document_name": "deed-of-variation.pdf", "page": 1, "quote": "varied to a peppercorn"},
                            ],
                        }
                    ),
                )
            }

    with qa_agent.override(model=FunctionModel(stream_function=stream_function)):
        response = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "What is the rent now and what was it before?"},
        )

    assert response.status_code == 200
    events = parse_sse(response.text)
    message = next(e["message"] for e in events if e.get("type") == "message")
    cited_docs = {c["document_id"] for c in message["citations"]}
    assert cited_docs == {lease.id, deed.id}
    assert len(message["citations"]) == 2


async def test_bundle_refreshes_when_a_document_is_added_mid_conversation(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    # A document uploaded mid-conversation must become visible on the next turn.
    # The agent's instructions list the CURRENT bundle fresh every turn, so a
    # replayed prior `list_documents` result (frozen at 1 doc) can't hide the new
    # document.
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.flush()
    lease = Document(
        conversation_id=conversation.id,
        filename="lease.pdf",
        file_path="/tmp/lease.pdf",
        page_count=1,
    )
    lease.pages = [Page(page_number=1, text="The rent is GBP 850,000 per annum.")]
    db_session.add(lease)
    await db_session.commit()

    seen_instructions: list[str] = []

    async def fn(messages: list[ModelMessage], info: AgentInfo):
        seen_instructions.append(info.instructions or "")
        yield {
            0: DeltaToolCall(
                name=info.output_tools[0].name,
                json_args=json.dumps({"markdown": "ok", "citations": []}),
            )
        }

    with qa_agent.override(model=FunctionModel(stream_function=fn)):
        r1 = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "hi"},
        )
    assert r1.status_code == 200
    # Turn 1: only the lease exists, so only it is listed.
    assert "lease.pdf" in seen_instructions[-1]
    assert "environmental.pdf" not in seen_instructions[-1]

    # Upload a second document AFTER the first turn (mid-conversation).
    env = Document(
        conversation_id=conversation.id,
        filename="environmental.pdf",
        file_path="/tmp/env.pdf",
        page_count=1,
    )
    env.pages = [Page(page_number=1, text="Phase I environmental assessment.")]
    db_session.add(env)
    await db_session.commit()

    with qa_agent.override(model=FunctionModel(stream_function=fn)):
        r2 = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "compare the dates of both documents"},
        )
    assert r2.status_code == 200
    # Turn 2: the instructions reflect BOTH documents — fresh, not the stale view.
    assert "lease.pdf" in seen_instructions[-1]
    assert "environmental.pdf" in seen_instructions[-1]
