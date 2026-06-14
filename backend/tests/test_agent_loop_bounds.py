from __future__ import annotations

import json

import pytest
from pydantic_ai import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.llm import answer_question, qa_agent


async def test_agent_loop_is_bounded_by_usage_limits(db_session: AsyncSession) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.flush()
    document = Document(
        conversation_id=conversation.id,
        filename="lease.pdf",
        file_path="/tmp/lease.pdf",
        page_count=1,
    )
    document.pages = [Page(page_number=1, text="some page text")]
    db_session.add(document)
    await db_session.commit()

    # A model that never stops calling a tool — only the usage limits can end it.
    async def runaway(messages: list[ModelMessage], info: AgentInfo):
        yield {
            0: DeltaToolCall(
                name="read_page",
                json_args=json.dumps({"document_id": document.id, "page": 1}),
            )
        }

    with qa_agent.override(model=FunctionModel(stream_function=runaway)):
        with pytest.raises(UsageLimitExceeded):
            async for _ in answer_question(
                db=db_session,
                conversation_id=conversation.id,
                question="loop forever",
                history=[],
            ):
                pass
