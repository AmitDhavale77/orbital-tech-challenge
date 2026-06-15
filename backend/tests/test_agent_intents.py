from __future__ import annotations

import json
from typing import Any

from httpx import AsyncClient
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation
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


def _last_tool_return(messages: list[ModelMessage]) -> object:
    for message in reversed(messages):
        for part in getattr(message, "parts", []):
            if isinstance(part, ToolReturnPart):
                return part.content
    return None


async def test_empty_bundle_lets_the_agent_decline_via_the_envelope(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    # A conversation with NO documents. list_documents() must hand the agent a
    # legible empty-state (count 0 + guidance) so it answers plainly instead of
    # looping to the tool-call limit.
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.commit()

    seen: dict[str, Any] = {"list_calls": 0, "envelope": None}

    async def stream_function(messages: list[ModelMessage], info: AgentInfo):
        if not _has_tool_return(messages):
            seen["list_calls"] += 1
            yield {0: DeltaToolCall(name="list_documents", json_args="{}")}
        else:
            seen["envelope"] = _last_tool_return(messages)
            yield {
                0: DeltaToolCall(
                    name=info.output_tools[0].name,
                    json_args=json.dumps(
                        {
                            "markdown": "No documents have been uploaded to this bundle yet. Please upload documents and ask again.",
                            "citations": [],
                        }
                    ),
                )
            }

    with qa_agent.override(model=FunctionModel(stream_function=stream_function)):
        response = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "Summarise the documents"},
        )

    assert response.status_code == 200
    # The tool told the agent the bundle is empty (count 0 + guidance)...
    envelope = str(seen["envelope"])
    assert "document_count" in envelope
    assert "empty" in envelope.lower()
    # ...so the agent did not loop on list_documents.
    assert seen["list_calls"] == 1

    events = _parse_sse(response.text)
    message = next(e["message"] for e in events if e.get("type") == "message")
    assert message["citations"] == []
    assert "document" in message["content"].lower()
