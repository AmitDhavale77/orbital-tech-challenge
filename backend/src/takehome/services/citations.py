from __future__ import annotations

import re

import structlog
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.services import document as document_service

logger = structlog.get_logger()


class Citation(BaseModel):
    """A verifiable reference from an answer back to a (document, page).

    `quote` is verbatim text the model copied from the page; it is string-matched
    back into that page before the citation is persisted (ADR-0002).
    """

    document_id: str
    document_name: str
    page: int
    quote: str


class Answer(BaseModel):
    """The agent's typed output: answer markdown plus its citations."""

    markdown: str
    citations: list[Citation]


class VerifiedCitation(Citation):
    """A Citation enriched with server-computed fields for on-PDF highlighting.

    `rects` are the quote's bounding boxes on the page and `page_width/height`
    the page size, both in PDF points (ADR-0002). These are computed during
    verification — never provided by the model.
    """

    rects: list[list[float]] = []
    page_width: float | None = None
    page_height: float | None = None


class GroundedAnswer(BaseModel):
    """The server-side result after verification: markdown + verified citations."""

    markdown: str
    citations: list[VerifiedCitation]


_WHITESPACE = re.compile(r"\s+")
_MARKER = re.compile(r"\[(\d+)\]")
# Smart punctuation a model may reproduce differently from the PDF's text layer.
_PUNCT = str.maketrans(
    {
        "“": '"',
        "”": '"',  # curly double quotes
        "‘": "'",
        "’": "'",  # curly single quotes / apostrophe
        "–": "-",
        "—": "-",  # en / em dash
        " ": " ",  # non-breaking space
    }
)


def _normalize(text: str) -> str:
    """Collapse whitespace and normalise smart punctuation, then strip.

    PDF extraction wraps lines and doubles spaces, and models reproduce curly
    quotes / dashes inconsistently, so a verbatim quote rarely matches
    byte-for-byte. This keeps the match robust while staying faithful to the
    exact tokens (case preserved).
    """
    return _WHITESPACE.sub(" ", text.translate(_PUNCT)).strip()


def verify_quote(quote: str, page_text: str) -> bool:
    """True if `quote` appears in `page_text`, tolerant of whitespace only."""
    needle = _normalize(quote)
    if not needle:
        return False
    return needle in _normalize(page_text)


def locate_quote(quote: str, pages: list[tuple[int, str]], preferred: int) -> int | None:
    """Return the page number where `quote` appears, or None if it's nowhere.

    Reading a whole document, the model often cites the right quote on the wrong
    page. Rather than drop a genuinely-present quote, we accept it on whichever
    page actually contains it — preferring the cited page when it matches, so a
    quote that legitimately recurs stays anchored where the model put it.
    """
    needle = _normalize(quote)
    if not needle:
        return None
    pages_by_text = [(n, _normalize(t)) for n, t in pages]
    for page_number, text in pages_by_text:
        if page_number == preferred and needle in text:
            return preferred
    for page_number, text in pages_by_text:
        if needle in text:
            return page_number
    return None


async def verify_and_renumber(
    db: AsyncSession,
    conversation_id: str,
    markdown: str,
    citations: list[Citation],
) -> tuple[str, list[VerifiedCitation]]:
    """Verify each citation against its page and reconcile the answer's markers.

    A citation is kept if its quote appears verbatim ANYWHERE in the cited
    document; when the model attributed it to the wrong page (common when it reads
    a whole document at once), the page is corrected to where the quote actually
    is. Only a quote that is nowhere in the document (or an unknown document) is
    dropped as hallucinated (ADR-0002). Survivors are enriched with on-PDF
    highlight rects (PyMuPDF `search_for`), then the `[n]` markers in `markdown`
    are rewritten so the surviving citations are numbered contiguously and every
    rendered marker resolves — a dropped citation's marker is removed.

    Returns the rewritten markdown and the verified, renumbered citations.
    """
    paths = await document_service.get_document_paths(db, conversation_id)
    pages_cache: dict[str, list[tuple[int, str]]] = {}
    verified: list[VerifiedCitation] = []
    old_to_new: dict[int, int] = {}
    for old_number, citation in enumerate(citations, start=1):
        pages = pages_cache.get(citation.document_id)
        if pages is None:
            pages = await document_service.get_document_pages(
                db, conversation_id, citation.document_id
            )
            pages_cache[citation.document_id] = pages
        page = locate_quote(citation.quote, pages, citation.page)
        if page is None:
            logger.warning(
                "Dropped unverifiable citation",
                document_id=citation.document_id,
                page=citation.page,
                reason="document not found" if not pages else "quote not in document",
                quote=citation.quote[:160],
            )
            continue
        rects: list[list[float]] = []
        page_width = page_height = None
        path = paths.get(citation.document_id)
        if path:
            rects, page_width, page_height = document_service.compute_quote_rects(
                path, page, citation.quote
            )
        data = citation.model_dump()
        data["page"] = page  # corrected to where the quote actually appears
        verified.append(
            VerifiedCitation(
                **data,
                rects=rects,
                page_width=page_width,
                page_height=page_height,
            )
        )
        old_to_new[old_number] = len(verified)

    def _rewrite(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return f"[{old_to_new[number]}]" if number in old_to_new else ""

    return _MARKER.sub(_rewrite, markdown), verified
