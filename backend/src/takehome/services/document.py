from __future__ import annotations

import os
import uuid
from typing import cast

import pymupdf  # PyMuPDF (typed; the legacy `fitz` alias ships no py.typed)
import structlog
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.config import settings
from takehome.db.models import Document, Page

logger = structlog.get_logger()

# ts_headline options: no markup (so previews read cleanly) and a single short
# fragment around the match.
_HEADLINE_OPTIONS = "StartSel=,StopSel=,MaxFragments=1,MaxWords=40,MinWords=15"


async def upload_document(
    session: AsyncSession, conversation_id: str, file: UploadFile
) -> Document:
    """Upload and process a PDF document for a conversation.

    Validates the file is a PDF, saves it to disk, extracts text using PyMuPDF,
    and stores metadata in the database. A conversation owns a Document Bundle,
    so repeated uploads are accepted (ADR-0001 / ticket 03).

    Raises ValueError if the file is not a PDF.
    """
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


async def search_pages(
    session: AsyncSession,
    conversation_id: str,
    query: str,
    *,
    per_document: int = 2,
    limit: int = 8,
) -> list[dict[str, str | int]]:
    """Keyword search over the bundle: rank pages, return diversified hits.

    Ranks pages by Postgres full-text relevance and returns the top hits as
    `(document_id, document_name, page, preview)`, where `preview` is the
    keyword-in-context fragment (`ts_headline`). A per-document cap stops a long
    document from burying a short one (ADR-0002). The previews are routing hints:
    the agent `read_page`s a hit to read the full page and quote it.
    """
    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank(Page.tsv, tsquery)
    preview = func.ts_headline("english", Page.text, tsquery, _HEADLINE_OPTIONS)
    stmt = (
        select(
            Document.id,
            Document.filename,
            Page.page_number,
            preview.label("preview"),
        )
        .join(Document, Page.document_id == Document.id)
        .where(Document.conversation_id == conversation_id)
        .where(Page.tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(100)  # candidate pool, narrowed by the per-document cap below
    )
    rows = (await session.execute(stmt)).all()

    # Per-document cap + global limit (rows are already ranked best-first).
    seen: dict[str, int] = {}
    results: list[dict[str, str | int]] = []
    for document_id, filename, page_number, preview_text in rows:
        if seen.get(document_id, 0) >= per_document:
            continue
        seen[document_id] = seen.get(document_id, 0) + 1
        results.append(
            {
                "document_id": document_id,
                "document_name": filename,
                "page": page_number,
                "preview": " ".join((preview_text or "").split()),
            }
        )
        if len(results) >= limit:
            break
    return results
