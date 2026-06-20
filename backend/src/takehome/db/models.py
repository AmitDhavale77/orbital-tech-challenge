from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: uuid.uuid4().hex[:16]
    )
    title: Mapped[str] = mapped_column(String, default="New Conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )
    # Full PydanticAI ModelMessage history (to_jsonable_python(all_messages())),
    # overwritten each turn. The agent replays this so it can reuse pages/quotes it
    # already read instead of re-running its tools, and so Anthropic CompactionParts
    # round-trip (ticket 08). The `messages` table stays the display source of truth;
    # this is the agent-replay source of truth. Null until a conversation's first
    # turn under this feature (back-compat: seed from plain text, then persist this).
    model_history: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSONB, nullable=True
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )
    documents: Mapped[list[Document]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: uuid.uuid4().hex[:16]
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    role: Mapped[str] = mapped_column(String)  # "user", "assistant", "system"
    content: Mapped[str] = mapped_column(Text)
    sources_cited: Mapped[int] = mapped_column(Integer, default=0)  # verified citation count
    # Verified citations as JSON: [{document_id, document_name, page, quote}, ...]
    citations: Mapped[list[dict[str, str | int]] | None] = mapped_column(
        JSON, nullable=True
    )
    # The agent's steps for this answer as JSON: [{kind, label, document_id, page}, ...]
    steps: Mapped[list[dict[str, str | int | None]] | None] = mapped_column(
        JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: uuid.uuid4().hex[:16]
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    filename: Mapped[str] = mapped_column(String)
    file_path: Mapped[str] = mapped_column(String)
    # sha256 of the uploaded bytes — dedup key within a conversation's bundle
    # (the same PDF cannot be uploaded twice). Nullable for pre-dedup rows.
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    # Routing card as JSON: {kind, summary}.
    # A hint for the agent only — never a citable Source (ADR-0002).
    card: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    conversation: Mapped[Conversation] = relationship(back_populates="documents")
    pages: Mapped[list[Page]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="Page.page_number",
    )


class Page(Base):
    """One page of a Document — the unit a Citation anchors to (ADR-0002)."""

    __tablename__ = "pages"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: uuid.uuid4().hex[:16]
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE")
    )
    page_number: Mapped[int] = mapped_column(Integer)  # 1-based
    text: Mapped[str] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="pages")
