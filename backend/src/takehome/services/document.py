from __future__ import annotations

import os
import uuid
from typing import cast

import pymupdf  # PyMuPDF (typed; the legacy `fitz` alias ships no py.typed)
import structlog
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.config import settings
from takehome.db.models import Document, Page

logger = structlog.get_logger()


async def upload_document(
    session: AsyncSession, conversation_id: str, file: UploadFile
) -> Document:
    """Upload and process a PDF document for a conversation.

    Validates the file is a PDF, saves it to disk, extracts text using PyMuPDF,
    and stores metadata in the database.

    Raises ValueError if the conversation already has a document or the file is not a PDF.
    """
    # Check if conversation already has a document
    existing = await get_document_for_conversation(session, conversation_id)
    if existing is not None:
        raise ValueError("Conversation already has a document. Only one document per conversation is allowed.")

    # Validate file type
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        filename = file.filename or ""
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

    # Read file content
    content = await file.read()

    # Validate file size
    if len(content) > settings.max_upload_size:
        raise ValueError(
            f"File too large. Maximum size is {settings.max_upload_size // (1024 * 1024)}MB."
        )

    # Generate a unique filename to avoid collisions
    original_filename = file.filename or "document.pdf"
    unique_name = f"{uuid.uuid4().hex}_{original_filename}"
    file_path = os.path.join(settings.upload_dir, unique_name)

    # Ensure upload directory exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Save the file to disk
    with open(file_path, "wb") as f:
        f.write(content)

    logger.info("Saved uploaded PDF", filename=original_filename, path=file_path, size=len(content))

    # Extract text per page using PyMuPDF. Each page becomes a Page row — the
    # unit the agent reads on demand and a Citation later anchors to (ADR-0002).
    # `extracted_text` is kept populated for backward compatibility, but it is
    # no longer the path the agent reads from.
    page_texts: list[str] = []
    page_count = 0
    try:
        doc = pymupdf.open(file_path)
        page_count = len(doc)
        for page_num in range(page_count):
            page = doc[page_num]
            # Default option ("text") returns a str; the stub widens the return.
            page_texts.append(cast(str, page.get_text()))  # pyright: ignore[reportUnknownMemberType]
        doc.close()
    except Exception:
        logger.exception("Failed to extract text from PDF", filename=original_filename)
        page_texts = []
        page_count = 0

    blob = "\n\n".join(
        f"--- Page {i + 1} ---\n{t}" for i, t in enumerate(page_texts) if t.strip()
    )

    logger.info(
        "Extracted text from PDF",
        filename=original_filename,
        page_count=page_count,
        text_length=len(blob),
    )

    # Create the document record with one Page per PDF page.
    document = Document(
        conversation_id=conversation_id,
        filename=original_filename,
        file_path=file_path,
        extracted_text=blob if blob else None,
        page_count=page_count,
        pages=[
            Page(page_number=i + 1, text=text)
            for i, text in enumerate(page_texts)
        ],
    )
    session.add(document)
    await session.commit()
    await session.refresh(document)
    return document


async def get_document(session: AsyncSession, document_id: str) -> Document | None:
    """Get a document by its ID."""
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def get_document_for_conversation(
    session: AsyncSession, conversation_id: str
) -> Document | None:
    """Get the document for a conversation, if one exists."""
    stmt = select(Document).where(Document.conversation_id == conversation_id)
    result = await session.execute(stmt)
    return result.scalars().first()


async def list_documents_for_conversation(
    session: AsyncSession, conversation_id: str
) -> list[Document]:
    """Return every Document in a conversation's bundle, oldest first."""
    stmt = (
        select(Document)
        .where(Document.conversation_id == conversation_id)
        .order_by(Document.uploaded_at.asc())
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_page_text(
    session: AsyncSession,
    conversation_id: str,
    document_id: str,
    page_number: int,
) -> str | None:
    """Return the text of one page, scoped to the conversation.

    Returns None when the document is not in this conversation or the page does
    not exist — the agent's `read_page` tool turns that into a ModelRetry.
    """
    stmt = (
        select(Page.text)
        .join(Document, Page.document_id == Document.id)
        .where(Document.conversation_id == conversation_id)
        .where(Page.document_id == document_id)
        .where(Page.page_number == page_number)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()
