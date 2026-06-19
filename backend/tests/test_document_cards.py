from __future__ import annotations

import io
from typing import cast

from fastapi import UploadFile
from pydantic_ai import ModelResponse
from pydantic_ai.messages import ModelMessage, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document
from takehome.services.cards import card_agent
from takehome.services.document import document_summaries, upload_document
from tests.helpers import make_pdf


async def test_card_is_generated_and_stored_on_upload(db_session: AsyncSession) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.commit()

    def card_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "kind": "Lease",
                        "summary": "A commercial lease between Landlord X and Tenant Y.",
                    },
                )
            ]
        )

    upload = UploadFile(
        filename="lease.pdf",
        file=io.BytesIO(make_pdf("LEASE between Landlord X and Tenant Y. Rent £100.")),
    )
    with card_agent.override(model=FunctionModel(card_function)):
        document = await upload_document(db_session, conversation.id, upload)

    assert document.card is not None
    assert document.card["kind"] == "Lease"
    assert document.card["summary"] == "A commercial lease between Landlord X and Tenant Y."


async def test_list_documents_includes_the_card() -> None:
    document = Document(
        conversation_id="c1",
        filename="title-report.pdf",
        file_path="/tmp/t.pdf",
        page_count=3,
        card={
            "kind": "Official Title Report",
            "summary": "A title report for Lot 7.",
        },
    )

    summaries = document_summaries([document])

    assert {"document_id", "document_name", "page_count", "card"} <= set(summaries[0])
    card = cast("dict[str, object]", summaries[0]["card"])
    assert card["kind"] == "Official Title Report"
