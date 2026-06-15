from __future__ import annotations

import json
from typing import Any

from httpx import AsyncClient
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.document import get_document_text
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


async def _seed_doc(db: AsyncSession) -> tuple[str, str]:
    conversation = Conversation()
    db.add(conversation)
    await db.flush()
    lease = Document(
        conversation_id=conversation.id,
        filename="lease.pdf",
        file_path="/tmp/lease.pdf",
        page_count=3,
    )
    lease.pages = [
        Page(page_number=1, text="The rent is GBP 1.75 million per annum."),
        Page(page_number=2, text="The Permitted Use is offices."),
        Page(page_number=3, text="The term is 25 years."),
    ]
    db.add(lease)
    await db.commit()
    return conversation.id, lease.id


async def test_get_document_text_joins_pages_with_page_markers(
    db_session: AsyncSession,
) -> None:
    conversation_id, lease_id = await _seed_doc(db_session)

    text = await get_document_text(db_session, conversation_id, lease_id)

    assert text is not None
    # Every page is present, in order, with its page-anchored marker so the agent
    # can read the whole document yet still cite the exact page.
    assert "--- Page 1 ---" in text
    assert "--- Page 2 ---" in text
    assert "--- Page 3 ---" in text
    assert text.index("--- Page 1 ---") < text.index("--- Page 2 ---")
    assert "The rent is GBP 1.75 million per annum." in text
    assert "The Permitted Use is offices." in text
    assert "The term is 25 years." in text


async def test_get_document_text_is_scoped_to_the_conversation(
    db_session: AsyncSession,
) -> None:
    conversation_id, lease_id = await _seed_doc(db_session)
    other_conversation_id, _ = await _seed_doc(db_session)

    # The lease belongs to the first conversation; reading it as the other
    # conversation must not leak it.
    assert await get_document_text(db_session, conversation_id, lease_id) is not None
    assert await get_document_text(db_session, other_conversation_id, lease_id) is None


async def test_agent_reads_whole_document_then_cites_a_page(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conversation_id, lease_id = await _seed_doc(db_session)
    calls: list[str] = []

    async def stream_function(messages: list[ModelMessage], info: AgentInfo):
        seen = _tool_returns(messages)
        if seen == 0:
            calls.append("read_document")
            yield {0: DeltaToolCall(name="read_document", json_args=json.dumps({"document_id": lease_id}))}
        else:
            payload = {
                "markdown": "The rent is GBP 1.75 million per annum.[1]",
                "citations": [
                    {
                        "document_id": lease_id,
                        "document_name": "lease.pdf",
                        "page": 1,
                        "quote": "The rent is GBP 1.75 million per annum.",
                    }
                ],
            }
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args=json.dumps(payload))}

    with qa_agent.override(model=FunctionModel(stream_function=stream_function)):
        response = await client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "What is the rent?"},
        )

    assert response.status_code == 200
    assert calls == ["read_document"]
    events = _parse_sse(response.text)
    message = next(e["message"] for e in events if e.get("type") == "message")
    # The page-anchored citation survives verification (quote is on page 1).
    assert message["citations"][0]["document_id"] == lease_id
    assert message["citations"][0]["page"] == 1


async def test_read_document_step_streams_live_and_persists(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conversation_id, lease_id = await _seed_doc(db_session)

    async def stream_function(messages: list[ModelMessage], info: AgentInfo):
        seen = _tool_returns(messages)
        if seen == 0:
            yield {0: DeltaToolCall(name="read_document", json_args=json.dumps({"document_id": lease_id}))}
        else:
            payload = {
                "markdown": "Offices.[1]",
                "citations": [
                    {
                        "document_id": lease_id,
                        "document_name": "lease.pdf",
                        "page": 2,
                        "quote": "The Permitted Use is offices.",
                    }
                ],
            }
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args=json.dumps(payload))}

    with qa_agent.override(model=FunctionModel(stream_function=stream_function)):
        response = await client.post(
            f"/api/conversations/{conversation_id}/messages",
            json={"content": "What is the permitted use?"},
        )

    events = _parse_sse(response.text)

    # Live: a "read" step for the whole-document read, labelled with the filename,
    # streamed before the answer.
    live_steps = [e for e in events if e.get("type") == "step"]
    assert [s["kind"] for s in live_steps] == ["read"]
    assert live_steps[0]["document_id"] == lease_id
    assert "lease.pdf" in live_steps[0]["label"]
    assert live_steps[0]["page"] is None  # whole-document read, not a single page
    types = [e.get("type") for e in events]
    assert types.index("step") < types.index("message")

    # Persisted: a reload returns the same steps.
    listed = await client.get(f"/api/conversations/{conversation_id}/messages")
    assistant = next(m for m in listed.json() if m["role"] == "assistant")
    assert [s["kind"] for s in assistant["steps"]] == ["read"]
