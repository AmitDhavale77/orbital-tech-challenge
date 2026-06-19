from __future__ import annotations

import re
import unicodedata
from typing import Any

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
    """A typed answer: markdown plus the citations that support it.

    The final structured output of both the interactive chat agent (`llm.py`) and
    the map-reduce reduce step (`portfolio.py`). Citations ride along with the
    answer rather than via a separate `cite` tool, and it is returned whole, not
    streamed token-by-token (Anthropic buffers tool-call JSON) — only the tool
    steps stream live.
    """

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
    """The server-side result after verification: markdown + verified citations.

    `model_history` is the serialized PydanticAI `all_messages()` snapshot for the
    run (`to_jsonable_python(...)`), carried back so the router can persist it for
    replay/compaction. It is `None` on the degrade path so a failed turn never
    overwrites the last good snapshot.
    """

    markdown: str
    citations: list[VerifiedCitation]
    model_history: list[dict[str, Any]] | None = None


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


# Zero-width / formatting characters PDFs and models inject that carry no meaning.
_ZERO_WIDTH = str.maketrans(
    {
        "​": "",  # zero-width space
        "‌": "",  # zero-width non-joiner
        "‍": "",  # zero-width joiner
        "﻿": "",  # zero-width no-break space / BOM
        "­": "",  # soft hyphen
    }
)
# A hyphen immediately before a line break = a word wrapped across lines in the
# PDF text layer ("inter-\nest" -> "interest"). En/em dashes are already ascii "-"
# by the time this runs (see _normalize order), so we only handle "-".
_DEHYPHENATE = re.compile(r"-\s*\n\s*")
# "…" or "..." — an author-inserted elision the model may use to bridge two spans.
_ELLIPSIS = re.compile(r"…|\.\.\.+")
# Minimum length for an ellipsis fragment to count, so a stray "..." can't
# manufacture a trivially-true match out of a few characters.
_MIN_FRAGMENT = 12


def _normalize(text: str) -> str:
    """Canonicalise text so a faithful quote matches the page despite formatting.

    PDF extraction wraps lines, doubles spaces, emits ligatures and soft hyphens;
    models reproduce curly quotes / dashes / case inconsistently. We fold all of
    that to a canonical form — applied identically to quote and page — so only a
    genuine difference in *words* (a wrong number, a changed term) can fail the
    match. Order matters: NFKC first (so ligatures / full-width expand and soft- /
    zero-width chars survive to be stripped), then punctuation, then de-hyphenation
    (needs the real newline), then whitespace collapse, then casefold.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH)
    text = text.translate(_PUNCT)
    text = _DEHYPHENATE.sub("", text)
    text = _WHITESPACE.sub(" ", text)
    return text.casefold().strip()


def _fragments(needle: str) -> list[str]:
    """Split an already-normalised needle on ellipsis into substantial fragments.

    No ellipsis → `[needle]`, so a short legitimate quote ("£50,000") keeps exact
    substring behaviour. With an ellipsis, only fragments >= _MIN_FRAGMENT survive.
    """
    parts = [p.strip() for p in _ELLIPSIS.split(needle)]
    if len(parts) == 1:
        return parts
    return [p for p in parts if len(p) >= _MIN_FRAGMENT]


def _contains(haystack: str, needle: str) -> bool:
    """True if `needle` is found in `haystack`, tolerant of formatting only.

    A single-fragment needle must be an exact normalised substring. An
    ellipsis-split needle requires every substantial fragment to appear in
    `haystack` IN ORDER — each still an exact substring, so no fabricated word or
    number slips through, and reordered or cross-text stitching is rejected.
    """
    hay = _normalize(haystack)
    fragments = _fragments(_normalize(needle))
    if not fragments or not all(fragments):
        return False
    if len(fragments) == 1:
        return fragments[0] in hay
    pos = 0
    for fragment in fragments:
        index = hay.find(fragment, pos)
        if index == -1:
            return False
        pos = index + len(fragment)
    return True


def verify_quote(quote: str, page_text: str) -> bool:
    """True if `quote` appears in `page_text`, tolerant of formatting differences
    (whitespace, case, smart punctuation, ligatures, line-break hyphenation) and of
    an author-inserted ellipsis bridging two verbatim spans on the page."""
    if not _normalize(quote):
        return False
    return _contains(page_text, quote)


def locate_quote(quote: str, pages: list[tuple[int, str]], preferred: int) -> int | None:
    """Return the page number where `quote` appears, or None if it's nowhere.

    Reading a whole document, the model often cites the right quote on the wrong
    page. Rather than drop a genuinely-present quote, we accept it on whichever
    page actually contains it — preferring the cited page when it matches, so a
    quote that legitimately recurs stays anchored where the model put it. An
    ellipsis quote must be satisfied entirely within a single page.
    """
    if not _normalize(quote):
        return None
    for page_number, text in pages:
        if page_number == preferred and _contains(text, quote):
            return preferred
    for page_number, text in pages:
        if _contains(text, quote):
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
