from __future__ import annotations

import re

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.services import document as document_service


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


_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Collapse whitespace runs to single spaces and strip.

    PDF extraction wraps lines and doubles spaces, so a verbatim quote rarely
    matches byte-for-byte. Normalising whitespace makes the match robust while
    staying faithful to the exact tokens (case preserved).
    """
    return _WHITESPACE.sub(" ", text).strip()


def verify_quote(quote: str, page_text: str) -> bool:
    """True if `quote` appears in `page_text`, tolerant of whitespace only."""
    needle = _normalize(quote)
    if not needle:
        return False
    return needle in _normalize(page_text)


async def verify_citations(
    db: AsyncSession,
    conversation_id: str,
    citations: list[Citation],
) -> list[Citation]:
    """Keep only citations whose quote is present on their cited page.

    The page text is fetched independently (conversation-scoped); a citation to
    an unknown document/page, or whose quote does not match, is dropped as
    hallucinated (ADR-0002).
    """
    verified: list[Citation] = []
    for citation in citations:
        page_text = await document_service.get_page_text(
            db, conversation_id, citation.document_id, citation.page
        )
        if page_text is not None and verify_quote(citation.quote, page_text):
            verified.append(citation)
    return verified
