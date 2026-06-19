"""Fan-out (map-reduce) portfolio analysis over a whole Document Bundle.

This is the **breadth/aggregation** path, separate from the interactive chat loop
(ADR-0001 reserves fan-out for batch, not chat). For questions that must touch
*every* document — "summarise the bundle", "across all leases, which grant
parking rights?" — the sequential chat loop can't scale (one read per doc, capped
tool calls). Instead we:

  map:    run a cheap per-document agent IN PARALLEL (one doc each, Haiku),
          returning a structured per-doc finding + verbatim supporting quotes;
  reduce: synthesise the relevant findings into one grounded answer (Sonnet).

Grounding is preserved end-to-end: per-doc quotes are verified against their page
(`verify_and_renumber`), and the reduce answer is verified again — so every
citation in the final answer is backed by real page text, exactly like chat.

See docs/research/architectures.md (map-reduce / scatter-gather) and the ADR-0002
amendment.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from pydantic import BaseModel
from pydantic_ai import Agent

from takehome.config import settings
from takehome.services import document as document_service
from takehome.services.citations import (
    Answer,
    Citation,
    VerifiedCitation,
    verify_and_renumber,
)

MAP_INSTRUCTIONS = (
    "You analyse ONE document from a commercial real estate Document Bundle to help "
    "answer a lawyer's question about the whole bundle. You are given that document's "
    "full text with `--- Page N ---` markers.\n"
    "Decide whether this document bears on the question (`relevant`). If it does, give "
    "a concise `summary` of what it says about the question, and list `quotes`: short "
    "snippets copied VERBATIM from the text, each with the page number from the nearest "
    "`--- Page N ---` marker above it. If the document does not bear on the question, "
    "return relevant=false with an empty summary and no quotes. Never use outside "
    "knowledge; quote only text that appears in THIS document."
)

REDUCE_INSTRUCTIONS = (
    "You synthesise per-document findings into a single answer to a lawyer's question "
    "across a commercial real estate Document Bundle. You are given, for each relevant "
    "document, its name, its document_id, a summary, and supporting verbatim quotes with "
    "page numbers.\n"
    "Write a clear, concise answer — use a markdown table when the question compares "
    "documents. Support each claim with a citation: copy the `quote` VERBATIM from the "
    "supplied quotes, with the matching document_id, document_name and page, and add "
    "`[1]`, `[2]` markers in order. Do not invent quotes or facts beyond the supplied "
    'findings. If none of the findings answer the question, reply "Not specified".'
)


class PageQuote(BaseModel):
    """A verbatim quote and the page it came from (the map agent's evidence)."""

    page: int
    quote: str


class MapResult(BaseModel):
    """One document's 'map' output, before citation verification."""

    relevant: bool
    summary: str = ""
    quotes: list[PageQuote] = []


class DocFinding(BaseModel):
    """A verified per-document finding — one row of the portfolio answer table."""

    document_id: str
    document_name: str
    relevant: bool
    summary: str = ""
    citations: list[VerifiedCitation] = []


class PortfolioAnswer(BaseModel):
    """The reduced answer plus the per-document rows that produced it."""

    markdown: str
    citations: list[VerifiedCitation] = []
    rows: list[DocFinding] = []


# Haiku for the per-doc map (cheap, runs N× in parallel); Sonnet for the reduce
# synthesis (CLAUDE.md model policy). Models live in config.py.
map_agent = Agent(
    settings.map_model, output_type=MapResult, instructions=MAP_INSTRUCTIONS
)
reduce_agent = Agent(
    settings.reduce_model, output_type=Answer, instructions=REDUCE_INSTRUCTIONS
)


def map_prompt(question: str, document_name: str, document_text: str) -> str:
    return (
        f"Question about the bundle: {question}\n\n"
        f"Document: {document_name}\n\n"
        f"--- BEGIN DOCUMENT ---\n{document_text}\n--- END DOCUMENT ---"
    )


def reduce_context(findings: list[DocFinding]) -> str:
    blocks: list[str] = []
    for finding in findings:
        lines = [f"Document: {finding.document_name} (document_id: {finding.document_id})"]
        if finding.summary:
            lines.append(f"Summary: {finding.summary}")
        if finding.citations:
            lines.append("Supporting quotes:")
            lines.extend(f'- [page {c.page}] "{c.quote}"' for c in finding.citations)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


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
        # Verify/relocate the quotes only — pass empty markdown so the summary's
        # prose is NOT run through citation-marker rewriting, which would delete or
        # renumber any incidental bracketed number in it (e.g. a year "[2024]" or a
        # schedule reference "[4]"). The map summary carries no [n] markers of its
        # own; only a genuine answer body should ever have its markers rewritten.
        _, verified = await verify_and_renumber(session, conversation_id, "", citations)
        return DocFinding(
            document_id=document_id,
            document_name=document_name,
            relevant=True,
            summary=mapped.summary,
            citations=verified,
        )


async def reduce(
    conversation_id: str, question: str, findings: list[DocFinding]
) -> tuple[str, list[VerifiedCitation]]:
    """Reduce relevant findings → one grounded answer (citations re-verified)."""
    from takehome.db.session import async_session

    result = await reduce_agent.run(
        f"Question: {question}\n\n"
        f"Findings from the relevant documents:\n\n{reduce_context(findings)}"
    )
    answer = result.output
    async with async_session() as session:
        return await verify_and_renumber(
            session, conversation_id, answer.markdown, answer.citations
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
    per document at once."""
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(document_id: str, document_name: str) -> DocFinding:
        async with semaphore:
            return await map_one(
                conversation_id, document_id, document_name, question, on_doc
            )

    return list(await asyncio.gather(*(guarded(d, n) for d, n in meta)))


async def map_documents(
    conversation_id: str,
    question: str,
    document_ids: list[str],
    *,
    concurrency: int = settings.map_concurrency,
    on_doc: Callable[[str, str], None] | None = None,
) -> list[DocFinding]:
    """Map the named documents in PARALLEL → one finding each, no reduce.

    The breadth path for the interactive chat loop (the `summarize_documents`
    tool): each requested document is read and summarised in its own isolated
    subagent context (Haiku), returning a summary plus verbatim, page-verified
    quotes. The chat agent itself is the reduce — it synthesises and cites these
    findings — so this stops at the map. Ids not in the bundle are skipped.
    `on_doc` (if given) fires per document as its map begins, for live progress.
    """
    from takehome.db.session import async_session

    async with async_session() as session:
        documents = await document_service.list_documents_for_conversation(
            session, conversation_id
        )
    wanted = set(document_ids)
    meta = [(d.id, d.filename) for d in documents if d.id in wanted]
    return await gather_maps(conversation_id, question, meta, concurrency, on_doc)


async def analyze_portfolio(
    conversation_id: str, question: str, *, concurrency: int = settings.map_concurrency
) -> PortfolioAnswer:
    """Run map-reduce over the conversation's whole bundle and return the answer
    plus one finding row per document."""
    from takehome.db.session import async_session

    async with async_session() as session:
        documents = await document_service.list_documents_for_conversation(
            session, conversation_id
        )
    meta = [(d.id, d.filename) for d in documents]
    findings = await gather_maps(conversation_id, question, meta, concurrency)
    relevant = [f for f in findings if f.relevant]
    if not relevant:
        return PortfolioAnswer(
            markdown="Not specified — no document in the bundle addresses this question.",
            rows=findings,
        )
    markdown, citations = await reduce(conversation_id, question, relevant)
    return PortfolioAnswer(markdown=markdown, citations=citations, rows=findings)
