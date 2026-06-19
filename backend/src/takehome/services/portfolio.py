"""Bundle-wide document analysis — the breadth path.

For questions that must look across many documents at once ("summarise the
bundle", "which leases grant parking rights?"), the sequential chat loop can't
scale: one read per doc, capped tool calls. So the `summarize_documents` chat
tool calls in here to fan out a cheap per-document pass IN PARALLEL — each
document is read whole and summarised in its own isolated LLM context (Haiku),
returning a summary plus verbatim, page-verified quotes.

There is no reduce step here: the chat agent in `llm.py` *is* the reduce. It
receives these findings and synthesises + cites the answer itself. Keeping each
document's full text out of the chat agent's context is the whole point — only
the distilled summary + verified quotes ever reach it.

Grounding is preserved: every quote is verified against its page
(`verify_and_renumber`), so a quote the model paraphrased or stitched together is
dropped before it can become a citation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import structlog
from pydantic import BaseModel
from pydantic_ai import Agent

from takehome.config import settings
from takehome.services import document as document_service
from takehome.services.citations import (
    Citation,
    VerifiedCitation,
    verify_and_renumber,
)

logger = structlog.get_logger()

MAP_INSTRUCTIONS = (
    "You analyse ONE document from a commercial real estate Document Bundle.\n"
    "Your job is to decide whether this document contains evidence that helps answer "
    "the user's bundle-wide question.\n\n"
    "You are given the document's full text with `--- Page N ---` markers.\n\n"
    "Rules:\n"
    "- Use only this document's text; never use outside knowledge.\n"
    "- If the document directly answers the question, set relevant=true and write a "
    "concise summary of what it says about the question.\n"
    "- If it only mentions related words but does not help answer the question, set "
    "relevant=false.\n"
    "- When relevant=true, include up to 5 short supporting quotes.\n"
    "- Each quote MUST be a single contiguous span copied character-for-character "
    "from one place in the text. Do NOT join text across lines or fields, do NOT "
    "insert '...', and do NOT trim interior words — a quote that is not an exact "
    "substring of a page is discarded.\n"
    "- Prefer one short span (<= 20 words) over a long passage.\n"
    "- Tag each quote with the page number from the nearest preceding "
    "`--- Page N ---` marker.\n"
    "- Do not quote a heading alone unless the heading itself answers the question.\n"
    "- If you cannot supply at least one exact supporting quote, set relevant=false."
)


class PageQuote(BaseModel):
    """A verbatim quote and the page it came from (the map pass's evidence)."""

    page: int
    quote: str


class MapResult(BaseModel):
    """One document's 'map' output, before citation verification."""

    relevant: bool
    summary: str = ""
    quotes: list[PageQuote] = []


class DocFinding(BaseModel):
    """A verified per-document finding the chat agent reduces over."""

    document_id: str
    document_name: str
    relevant: bool
    summary: str = ""
    citations: list[VerifiedCitation] = []


# Haiku for the per-doc map: it's a one-shot structured call (no tools, no loop)
# run N× in parallel, so the cheap model is the right choice (CLAUDE.md policy).
# It's an `Agent` only for the typed output + `agent.override(TestModel)` test
# seam — not a tool-using sub-agent.
map_agent = Agent(
    settings.map_model, output_type=MapResult, instructions=MAP_INSTRUCTIONS
)


def map_prompt(question: str, document_name: str, document_text: str) -> str:
    return (
        f"Question about the bundle: {question}\n\n"
        f"Document: {document_name}\n\n"
        f"--- BEGIN DOCUMENT ---\n{document_text}\n--- END DOCUMENT ---"
    )


async def map_one(
    conversation_id: str,
    document_id: str,
    document_name: str,
    question: str,
    on_doc: Callable[[str, str], None] | None = None,
) -> DocFinding:
    """Map one document → a verified finding. Opens its own session so parallel
    map tasks never share a session (AsyncSession is not concurrency-safe).

    `on_doc(document_id, document_name)` (if given) fires as this document's map
    begins, so a caller can stream live per-document progress.
    """
    from takehome.db.session import async_session

    if on_doc is not None:
        on_doc(document_id, document_name)

    async with async_session() as session:
        text = await document_service.get_document_text(
            session, conversation_id, document_id
        )
        if not text:
            return DocFinding(
                document_id=document_id, document_name=document_name, relevant=False
            )
        result = await map_agent.run(map_prompt(question, document_name, text))
        mapped = result.output
        if not mapped.relevant:
            return DocFinding(
                document_id=document_id, document_name=document_name, relevant=False
            )
        citations = [
            Citation(
                document_id=document_id,
                document_name=document_name,
                page=q.page,
                quote=q.quote,
            )
            for q in mapped.quotes
        ]

        # Verify only the quotes. The map summary is not an answer body, so we avoid
        # running citation-marker cleanup on prose that may contain harmless bracketed text.
        _, verified = await verify_and_renumber(session, conversation_id, "", citations)
        if citations and not verified:
            # Relevant doc, but every quote failed the verbatim check (Haiku tends to
            # elide / stitch quotes). The finding still carries the summary, but the
            # chat agent now has no citable evidence for it — make that visible.
            logger.warning(
                "map_one: relevant document lost all quotes to verification",
                document_id=document_id,
                document_name=document_name,
                quotes=len(citations),
            )
        return DocFinding(
            document_id=document_id,
            document_name=document_name,
            relevant=True,
            summary=mapped.summary,
            citations=verified,
        )


async def gather_maps(
    conversation_id: str,
    question: str,
    meta: list[tuple[str, str]],
    concurrency: int,
    on_doc: Callable[[str, str], None] | None = None,
) -> list[DocFinding]:
    """Run `map_one` over (document_id, document_name) pairs in parallel, capped
    by a semaphore so a large bundle doesn't open one LLM request (and session)
    per document at once.

    One document's failure (LLM/API error, oversized doc) must not sink the whole
    breadth answer, so a failed map degrades to a non-relevant finding rather than
    propagating out of the tool.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(document_id: str, document_name: str) -> DocFinding:
        async with semaphore:
            return await map_one(
                conversation_id, document_id, document_name, question, on_doc
            )

    results = await asyncio.gather(
        *(guarded(d, n) for d, n in meta), return_exceptions=True
    )
    findings: list[DocFinding] = []
    for (document_id, document_name), result in zip(meta, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning(
                "map_one failed; treating document as non-relevant",
                document_id=document_id,
                document_name=document_name,
                exc_info=result,
            )
            findings.append(
                DocFinding(
                    document_id=document_id,
                    document_name=document_name,
                    relevant=False,
                )
            )
        else:
            findings.append(result)
    return findings


async def map_documents(
    conversation_id: str,
    question: str,
    document_ids: list[str],
    *,
    concurrency: int = settings.map_concurrency,
    on_doc: Callable[[str, str], None] | None = None,
) -> list[DocFinding]:
    """Map the named documents in PARALLEL → one finding each (no reduce).

    Backs the `summarize_documents` chat tool: each requested document is read and
    summarised in its own isolated context (Haiku), returning a summary plus
    verbatim, page-verified quotes. The chat agent reduces these into the answer,
    so the work stops at the map. Ids not in the bundle are skipped. `on_doc` (if
    given) fires per document as its map begins, for live progress.
    """
    from takehome.db.session import async_session

    async with async_session() as session:
        documents = await document_service.list_documents_for_conversation(
            session, conversation_id
        )
    wanted = set(document_ids)
    meta = [(d.id, d.filename) for d in documents if d.id in wanted]
    return await gather_maps(conversation_id, question, meta, concurrency, on_doc)
