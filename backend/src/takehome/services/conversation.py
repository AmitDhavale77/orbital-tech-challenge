from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from takehome.db.models import Conversation


def title_from_message(text: str, limit: int = 40) -> str:
    """A cheap, deterministic conversation title: the first ~`limit` chars of the
    first user message (no LLM call). Falls back to "New Conversation" if empty."""
    collapsed = " ".join(text.split())
    if not collapsed:
        return "New Conversation"
    return collapsed[:limit] + "…" if len(collapsed) > limit else collapsed


async def create_conversation(session: AsyncSession) -> Conversation:
    """Create a new conversation with default title."""
    conversation = Conversation()
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def list_conversations(session: AsyncSession) -> list[Conversation]:
    """List all conversations ordered by most recently updated."""
    stmt = (
        select(Conversation)
        .options(selectinload(Conversation.documents))
        .order_by(Conversation.updated_at.desc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_conversation(session: AsyncSession, conversation_id: str) -> Conversation | None:
    """Get a single conversation with its documents eagerly loaded."""
    stmt = (
        select(Conversation)
        .options(selectinload(Conversation.documents))
        .where(Conversation.id == conversation_id)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def update_conversation(
    session: AsyncSession, conversation_id: str, title: str
) -> Conversation | None:
    """Update the title of a conversation."""
    conversation = await get_conversation(session, conversation_id)
    if conversation is None:
        return None
    conversation.title = title
    await session.commit()
    await session.refresh(conversation)
    return conversation


async def delete_conversation(session: AsyncSession, conversation_id: str) -> bool:
    """Delete a conversation. Returns True if it existed and was deleted."""
    stmt = select(Conversation).where(Conversation.id == conversation_id)
    result = await session.execute(stmt)
    conversation = result.scalar_one_or_none()
    if conversation is None:
        return False
    await session.delete(conversation)
    await session.commit()
    return True
