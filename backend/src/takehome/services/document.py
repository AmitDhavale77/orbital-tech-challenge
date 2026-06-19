from __future__ import annotations

import os
import re
import uuid
from typing import Any, cast

import pymupdf  # PyMuPDF (typed; the legacy `fitz` alias ships no py.typed)
import structlog
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.config import settings
from takehome.db.models import Document, Page
from takehome.services import cards

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

    # Generate a routing card (cheap model). Best-effort: a failure must not
    # block the upload — the agent can still search/read without a card.
    card: dict[str, object] | None = None
    if blob:
        try:
            card = (await cards.generate_card(blob)).model_dump()
        except Exception:
            logger.exception("Failed to generate document card", filename=original_filename)

    # Create the document record with one Page per PDF page.
    document = Document(
        conversation_id=conversation_id,
        filename=original_filename,
        file_path=file_path,
        extracted_text=blob if blob else None,
        page_count=page_count,
        card=card,
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


def document_summaries(documents: list[Document]) -> list[dict[str, object]]:
    """Routing summaries for `list_documents`: id, name, page count, and card."""
    return [
        {
            "document_id": d.id,
            "document_name": d.filename,
            "page_count": d.page_count,
            "card": d.card,
        }
        for d in documents
    ]


async def get_document_paths(
    session: AsyncSession, conversation_id: str
) -> dict[str, str]:
    """Map document_id → file_path for a conversation (for on-PDF highlighting)."""
    stmt = select(Document.id, Document.file_path).where(
        Document.conversation_id == conversation_id
    )
    result = await session.execute(stmt)
    return {str(doc_id): str(path) for doc_id, path in result.all()}


def compute_quote_rects(
    file_path: str, page_number: int, quote: str
) -> tuple[list[list[float]], float, float]:
    """Locate `quote` on a page and return its bounding boxes + page dimensions.

    Returns `(rects, width, height)` where each rect is `[x0, y0, x1, y1]` in PDF
    points and width/height are the page size in points (so the frontend can
    scale boxes to the rendered width). A quote that can't be located yields an
    empty rect list — the viewer then lands on the page without a highlight
    (ADR-0002 graceful degrade). Never raises.
    """
    rects: list[list[float]] = []
    width = height = 0.0
    try:
        doc = pymupdf.open(file_path)
        try:
            page: Any = doc[page_number - 1]  # pymupdf Page is only loosely typed
            width, height = float(page.rect.width), float(page.rect.height)
            found = page.search_for(quote)
            if not found:
                # Fall back to the first sentence — long multi-line quotes are
                # harder for search_for to match end-to-end.
                head = quote.split(". ")[0].strip()
                if head and head != quote:
                    found = page.search_for(head)
            rects = [[float(r.x0), float(r.y0), float(r.x1), float(r.y1)] for r in found]
        finally:
            doc.close()
    except Exception:
        logger.exception("Failed to compute quote rects", file_path=file_path)
    return rects, width, height


async def get_document_text(
    session: AsyncSession,
    conversation_id: str,
    document_id: str,
) -> str | None:
    """Return the full text of one document, scoped to the conversation.

    Pages are joined with `--- Page N ---` markers (the same format as the ingest
    blob) so the agent can read a whole document in a single call yet still cite
    the exact page a passage came from. Blank pages are skipped but real page
    numbers are preserved, so the markers stay accurate.

    Returns None when the document is not in this conversation or has no readable
    text — the agent's `read_document` tool turns that into a ModelRetry.
    """
    stmt = (
        select(Page.page_number, Page.text)
        .join(Document, Page.document_id == Document.id)
        .where(Document.conversation_id == conversation_id)
        .where(Page.document_id == document_id)
        .order_by(Page.page_number.asc())
    )
    rows = (await session.execute(stmt)).all()
    parts = [
        f"--- Page {page_number} ---\n{text}"
        for page_number, text in rows
        if text and text.strip()
    ]
    return "\n\n".join(parts) if parts else None


async def get_pages_text(
    session: AsyncSession,
    conversation_id: str,
    document_id: str,
    start_page: int,
    end_page: int,
) -> str | None:
    """Return the text of pages [start_page, end_page] joined with `--- Page N ---`
    markers, scoped to the conversation. Blank pages are skipped, real page numbers
    preserved (markers stay accurate). Returns None when the range has no readable
    text — the agent's `read_pages` tool turns that into a ModelRetry.
    """
    lo, hi = (start_page, end_page) if start_page <= end_page else (end_page, start_page)
    stmt = (
        select(Page.page_number, Page.text)
        .join(Document, Page.document_id == Document.id)
        .where(Document.conversation_id == conversation_id)
        .where(Page.document_id == document_id)
        .where(Page.page_number >= lo)
        .where(Page.page_number <= hi)
        .order_by(Page.page_number.asc())
    )
    rows = (await session.execute(stmt)).all()
    parts = [f"--- Page {n} ---\n{t}" for n, t in rows if t and t.strip()]
    return "\n\n".join(parts) if parts else None


async def grep_pages(
    session: AsyncSession,
    conversation_id: str,
    pattern: str,
    *,
    document_id: str | None = None,
) -> list[dict[str, str | int]]:
    """Search the bundle with a regular expression, like grep.

    Matches `pattern` (case-insensitive) against page text across the whole
    conversation, or one document when `document_id` is given, and returns every
    matching line with its document and page. No ranking, no caps — the agent
    controls breadth via the pattern it writes.
    """
    stmt = (
        select(Document.id, Document.filename, Page.page_number, Page.text)
        .join(Document, Page.document_id == Document.id)
        .where(Document.conversation_id == conversation_id)
        .where(Page.text.op("~*")(pattern))  # case-insensitive POSIX regex
        .order_by(Document.filename.asc(), Page.page_number.asc())
    )
    if document_id is not None:
        stmt = stmt.where(Page.document_id == document_id)
    rows = (await session.execute(stmt)).all()

    regex = re.compile(pattern, re.IGNORECASE)
    return [
        {
            "document_id": doc_id,
            "document_name": filename,
            "page": page_number,
            "line": line.strip(),
        }
        for doc_id, filename, page_number, text in rows
        for line in text.splitlines()
        if regex.search(line)
    ]


async def get_document_pages(
    session: AsyncSession,
    conversation_id: str,
    document_id: str,
) -> list[tuple[int, str]]:
    """Return `(page_number, text)` for every page of a document, in order.

    Used by citation verification to locate a quote anywhere in the document when
    the model attributed it to the wrong page. Scoped to the conversation.
    """
    stmt = (
        select(Page.page_number, Page.text)
        .join(Document, Page.document_id == Document.id)
        .where(Document.conversation_id == conversation_id)
        .where(Page.document_id == document_id)
        .order_by(Page.page_number.asc())
    )
    rows = (await session.execute(stmt)).all()
    return [(int(page_number), text) for page_number, text in rows]


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
