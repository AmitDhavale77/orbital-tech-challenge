from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import ModelMessage
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from takehome.db.models import Message
from takehome.db.session import get_session
from takehome.services.citations import GroundedAnswer, VerifiedCitation
from takehome.services.conversation import get_conversation, update_conversation
from takehome.services.llm import (
    Step,
    answer_question,
    generate_title,
    to_model_history,
)

logger = structlog.get_logger()

router = APIRouter(tags=["messages"])


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    sources_cited: int
    citations: list[VerifiedCitation] = []
    steps: list[Step] = []
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageCreate(BaseModel):
    content: str


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.get(
    "/api/conversations/{conversation_id}/messages",
    response_model=list[MessageOut],
)
async def list_messages(
    conversation_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[MessageOut]:
    """List all messages in a conversation, ordered by creation time."""
    # Verify the conversation exists
    conversation = await get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    result = await session.execute(stmt)
    messages = list(result.scalars().all())

    return [
        MessageOut(
            id=m.id,
            conversation_id=m.conversation_id,
            role=m.role,
            content=m.content,
            sources_cited=m.sources_cited,
            citations=[VerifiedCitation.model_validate(c) for c in (m.citations or [])],
            steps=[Step.model_validate(s) for s in (m.steps or [])],
            created_at=m.created_at,
        )
        for m in messages
    ]


@router.post("/api/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: str,
    body: MessageCreate,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Send a user message and stream back the AI response via SSE."""
    # Verify the conversation exists
    conversation = await get_conversation(session, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Save the user message
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=body.content,
    )
    session.add(user_message)
    await session.commit()
    await session.refresh(user_message)

    logger.info("User message saved", conversation_id=conversation_id, message_id=user_message.id)

    # Resolve the agent-replay history. Prefer the rich ModelMessage snapshot
    # (preserves prior tool calls/returns so a repeated question needn't re-read,
    # and round-trips compaction); fall back to seeding from the plain-text
    # `messages` table for conversations created before this feature (docs §6).
    if conversation.model_history:
        message_history: list[ModelMessage] = ModelMessagesTypeAdapter.validate_python(
            conversation.model_history
        )
    else:
        stmt = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .where(Message.id != user_message.id)
            .order_by(Message.created_at.asc())
        )
        result = await session.execute(stmt)
        message_history = to_model_history(
            {"role": m.role, "content": m.content} for m in result.scalars().all()
        )

    # First user message? (drives title generation) — counted independently of the
    # history branch above so it stays correct when replaying the rich snapshot.
    count_stmt = (
        select(func.count())
        .select_from(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.id != user_message.id)
        .where(Message.role == "user")
    )
    is_first_message = (await session.execute(count_stmt)).scalar_one() == 0

    async def event_stream() -> AsyncIterator[str]:
        """Generate SSE events with the streamed agent response.

        Opens a single session for the run: the agent's tools read pages on
        demand through it, and the assistant message is persisted with it. A
        fresh session is needed because the request-scoped one closes once the
        StreamingResponse is returned.
        """
        from takehome.db.session import async_session as session_factory

        async with session_factory() as run_session:
            full_response = ""
            final_answer: GroundedAnswer | None = None
            steps: list[Step] = []

            try:
                async for item in answer_question(
                    db=run_session,
                    conversation_id=conversation_id,
                    question=body.content,
                    message_history=message_history,
                ):
                    if isinstance(item, str):
                        full_response += item
                        event_data = json.dumps({"type": "content", "content": item})
                        yield f"data: {event_data}\n\n"
                    elif isinstance(item, Step):
                        steps.append(item)
                        event_data = json.dumps({"type": "step", **item.model_dump()})
                        yield f"data: {event_data}\n\n"
                    else:
                        final_answer = item

            except Exception:
                logger.exception(
                    "Error during agent run",
                    conversation_id=conversation_id,
                )
                error_msg = "I'm sorry, an error occurred while generating a response. Please try again."
                full_response = error_msg
                event_data = json.dumps({"type": "content", "content": error_msg})
                yield f"data: {event_data}\n\n"

            # Verified citations are the source of truth; sources_cited derives from them.
            content = final_answer.markdown if final_answer else full_response
            verified = final_answer.citations if final_answer else []
            citations_json = [c.model_dump() for c in verified]
            steps_json = [s.model_dump() for s in steps]

            assistant_message = Message(
                conversation_id=conversation_id,
                role="assistant",
                content=content,
                sources_cited=len(verified),
                citations=citations_json,
                steps=steps_json,
            )
            run_session.add(assistant_message)

            # Persist the full ModelMessage snapshot for replay/compaction, in the
            # same transaction as the display message so the two can't diverge. The
            # degrade path leaves model_history None — we keep the last good snapshot.
            if final_answer is not None and final_answer.model_history is not None:
                conv = await get_conversation(run_session, conversation_id)
                if conv is not None:
                    conv.model_history = final_answer.model_history

            await run_session.commit()
            await run_session.refresh(assistant_message)

            # Auto-generate title from first user message
            if is_first_message:
                try:
                    title = await generate_title(body.content)
                    await update_conversation(run_session, conversation_id, title)
                    logger.info(
                        "Auto-generated conversation title",
                        conversation_id=conversation_id,
                        title=title,
                    )
                except Exception:
                    logger.exception(
                        "Failed to generate title",
                        conversation_id=conversation_id,
                    )

            # Send the final message event with the complete assistant message
            message_data = json.dumps(
                {
                    "type": "message",
                    "message": {
                        "id": assistant_message.id,
                        "conversation_id": assistant_message.conversation_id,
                        "role": assistant_message.role,
                        "content": assistant_message.content,
                        "sources_cited": assistant_message.sources_cited,
                        "citations": citations_json,
                        "steps": steps_json,
                        "created_at": assistant_message.created_at.isoformat(),
                    },
                }
            )
            yield f"data: {message_data}\n\n"

            # Send the done signal
            done_data = json.dumps(
                {
                    "type": "done",
                    "sources_cited": len(verified),
                    "message_id": assistant_message.id,
                }
            )
            yield f"data: {done_data}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
