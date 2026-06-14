from __future__ import annotations

import json
from typing import Any

from httpx import AsyncClient
from pydantic_ai.messages import ModelMessage
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


async def test_only_verified_citations_are_returned_and_persisted(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.flush()
    document = Document(
        conversation_id=conversation.id,
        filename="lease.pdf",
        file_path="/tmp/lease.pdf",
        page_count=2,
    )
    document.pages = [
        Page(page_number=1, text="Clause 3.  The rent is GBP 1.75 million per annum."),
        Page(page_number=2, text="Clause 8.  Break clause provisions."),
    ]
    db_session.add(document)
    await db_session.commit()

    async def stream_function(messages: list[ModelMessage], info: AgentInfo):
        name = info.output_tools[0].name
        payload = {
            "markdown": "The rent is GBP 1.75 million.[1]",
            "citations": [
                {  # valid: quote present on page 1 (whitespace-tolerant)
                    "document_id": document.id,
                    "document_name": "lease.pdf",
                    "page": 1,
                    "quote": "The rent is GBP 1.75 million",
                },
                {  # hallucinated: quote not present anywhere -> must be dropped
                    "document_id": document.id,
                    "document_name": "lease.pdf",
                    "page": 2,
                    "quote": "the tenant must give 24 months notice",
                },
            ],
        }
        yield {0: DeltaToolCall(name=name, json_args=json.dumps(payload))}

    with qa_agent.override(model=FunctionModel(stream_function=stream_function)):
        response = await client.post(
            f"/api/conversations/{conversation.id}/messages",
            json={"content": "What is the rent?"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)

    # Streamed answer text arrived.
    streamed = "".join(e["content"] for e in events if e.get("type") == "content")
    assert "GBP 1.75 million" in streamed

    # Final message event carries exactly the one verified citation.
    final = [e for e in events if e.get("type") == "message"]
    assert final, "expected a final message event"
    citations = final[0]["message"]["citations"]
    assert len(citations) == 1
    assert citations[0]["page"] == 1
    assert citations[0]["document_id"] == document.id
    assert final[0]["message"]["sources_cited"] == 1

    done = [e for e in events if e.get("type") == "done"]
    assert done and done[0]["sources_cited"] == 1

    # Persisted: GET /messages returns the verified citation, bogus dropped.
    listed = await client.get(f"/api/conversations/{conversation.id}/messages")
    assert listed.status_code == 200
    assistant = [m for m in listed.json() if m["role"] == "assistant"]
    assert assistant and len(assistant[0]["citations"]) == 1
    assert assistant[0]["citations"][0]["quote"] == "The rent is GBP 1.75 million"
