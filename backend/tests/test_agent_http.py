from __future__ import annotations

import json
from typing import Any

from httpx import AsyncClient
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.llm import qa_agent


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


async def test_agent_answers_by_reading_pages(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    # Seed a single-document conversation with two pages.
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

    seen_requests: list[list[ModelMessage]] = []

    async def stream_function(messages: list[ModelMessage], info: AgentInfo):
        seen_requests.append(messages)
        if not _has_tool_return(messages):
            # First turn: drive a read_page tool call.
            yield {
                0: DeltaToolCall(
                    name="read_page",
                    json_args=json.dumps(
                        {"document_id": document.id, "page": 1}
                    ),
                )
            }
        else:
            # Second turn: emit the structured answer via the output tool.
            yield {
                0: DeltaToolCall(
                    name=info.output_tools[0].name,
                    json_args=json.dumps(
                        {"markdown": "The rent is GBP 1.75 million.", "citations": []}
                    ),
                )
            }

    with qa_agent.override(model=FunctionModel(stream_function=stream_function)):
        response = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "What is the rent?"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)

    streamed = "".join(
        e["content"] for e in events if e.get("type") == "content"
    )
    assert "GBP 1.75 million." in streamed

    # The agent actually read a page (tool was driven), and no document text was
    # placed in the prompt before the tool ran.
    assert len(seen_requests) >= 2
    assert "SECRET_PAGE_ONE" not in repr(seen_requests[0])

    # The assistant message persisted and carries the answer.
    final = [e for e in events if e.get("type") == "message"]
    assert final, "expected a final message event"
    assert final[0]["message"]["role"] == "assistant"
    assert "GBP 1.75 million." in final[0]["message"]["content"]
