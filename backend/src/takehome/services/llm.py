from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    ModelRetry,
    RunContext,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UsageLimits,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.anthropic import (
    AnthropicCompaction,
    AnthropicModel,
    AnthropicModelSettings,
)
from pydantic_core import to_jsonable_python
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.services import (
    document as document_service,  # imports config → exports ANTHROPIC_API_KEY
)
from takehome.services.citations import Answer, GroundedAnswer, verify_and_renumber

# Capable model for the reasoning/tool loop; Haiku is reserved for cheap aux
# calls (conversation titles). See CLAUDE.md and docs/pydantic-ai.md.
QA_MODEL = "claude-sonnet-4-6"
TITLE_MODEL = "anthropic:claude-haiku-4-5-20251001"

# Bounds the worst-case latency/cost of the agentic loop (docs/pydantic-ai.md §9).
# Sized for card-routed reading: list_documents() then one read_document per
# relevant doc (the whole doc in a single request). Breadth/aggregation across a
# large 12–50-doc bundle is handled by the parallel fan-out batch path
# (services/portfolio.py), not this sequential chat loop.
CHAT_USAGE_LIMITS = UsageLimits(
    request_limit=20,
    tool_calls_limit=25,
    total_tokens_limit=400_000,
)

INSTRUCTIONS = (
    "You are a precise assistant for commercial real estate lawyers reviewing a "
    "Document Bundle during due diligence. For any factual claim you rely ONLY on "
    "the documents in this bundle — never on outside knowledge.\n\n"
    "Classify each message into one of three types, then follow that branch. "
    "Decide the type BEFORE calling any tool.\n\n"
    "Type 1 — Conversational: a greeting, small talk, thanks, or a question about "
    "you or what you can do. Reply briefly and naturally, with no tool calls and no "
    "citations. This applies whether or not documents are uploaded; never tell "
    "someone to upload documents in response to a greeting or a capability "
    "question. You may add one sentence on what you do: answer questions grounded "
    "in the documents in their bundle.\n\n"
    "Type 2 — A question about the documents: call `list_documents()` once, then\n"
    "   - if document_count is 0, the bundle is empty: tell the user no documents "
    "have been uploaded yet and that you can help as soon as they add some, and "
    "call no other tool;\n"
    "   - otherwise locate the relevant pages with `grep`/`outline`, read and quote "
    "them with `read_pages`, and check every document that could bear on the "
    "answer;\n"
    "   - if an earlier turn in this conversation already read the pages this "
    "question needs and the bundle is unchanged, answer from that text already in "
    "your context and reuse those citations — do NOT re-run "
    "`list_documents`/`grep`/`read_pages` to re-read pages you have already read. "
    "You must still emit a citation with a verbatim `quote` for every factual "
    "claim (quotes are re-checked against the source each turn); only read again "
    "for pages you have not yet seen;\n"
    "   - if the user names a document or detail not in the bundle, say it isn't in "
    "the bundle and mention what the bundle does contain;\n"
    "   - if you read the relevant pages but the answer isn't there, say "
    '"Not specified" (no citation) rather than speculating.\n\n'
    "Type 3 — A question NOT about this bundle (general law, outside knowledge, "
    "current events): politely decline, explain you only answer from the documents "
    "in this bundle, and suggest a grounded next step (e.g. checking how a term is "
    "used in a specific document). Do not answer from outside knowledge; do not "
    "cite.\n\n"
    "Classification examples:\n"
    "- 'hi' / 'how are you?' → Type 1\n"
    "- 'what can you do?' / 'how does this work?' → Type 1\n"
    "- 'thanks, that's helpful' → Type 1\n"
    "- 'what does the lease say about the rent review?' → Type 2\n"
    "- 'is there a break clause?' / 'summarise the bundle' → Type 2\n"
    "- 'is a break clause enforceable under English law?' → Type 3\n"
    "- 'what's the latest case law on dilapidations?' → Type 3\n\n"
    "Tools (you cannot see any document text until you read it):\n"
    "- `list_documents()` — the bundle's document_count, its documents (each with a "
    "routing `card`: type, parties, date, topics, one-line), and guidance. Cards "
    "are hints only: never quote or cite a card.\n"
    "- `grep(pattern, document_id?)` — case-insensitive regex search across the "
    "bundle, or one document if document_id is given; returns matching lines with "
    "their document and page. Page text can wrap mid-phrase, so use `\\s+`/`.*` "
    "between the words of a phrase.\n"
    "- `outline(document_id)` — a per-page map of one document, to decide which "
    "pages to read.\n"
    "- `read_pages(document_id, start_page, end_page?)` — read a page range "
    "(`--- Page N ---` markers); this is the ONLY text you may quote and cite.\n\n"
    "Rules:\n"
    "- The current bundle is listed in your instructions and is always up to date. "
    "Call `list_documents()` when you need the cards; call it again only if that "
    "list shows a document you have not yet seen a card for (e.g. one uploaded "
    "mid-conversation). Otherwise never repeat a tool call hoping for a different "
    "result.\n"
    "- Base every factual statement on text you have actually read; never guess or "
    "rely on prior knowledge of the documents.\n"
    "- Cite each factual claim with the document, the page (from the nearest "
    "`--- Page N ---` marker above the passage), and a `quote` copied VERBATIM "
    "from that page (the quote is checked against the page, so it must match "
    "exactly). Add `[1]`, `[2]` markers in the markdown in citation order; never "
    "invent a citation.\n"
    "- Be concise and precise. Lawyers value accuracy over verbosity."
)


class Step(BaseModel):
    """One action the agent took to reach its answer (a streamed/persisted trace)."""

    kind: str  # "search" | "read" | "list" | "tool"
    label: str  # human-readable, e.g. 'Reading lease.pdf · p.4'
    document_id: str | None = None
    page: int | None = None


@dataclass
class AppDeps:
    """Per-run dependencies injected into the agent's tools."""

    db: AsyncSession
    conversation_id: str


# Cache only the stable prefix — system prompt + tool definitions — never a
# preloaded document (docs/pydantic-ai.md §8, CLAUDE.md context-management rule).
_QA_SETTINGS = AnthropicModelSettings(
    anthropic_cache_instructions=True,
    anthropic_cache_tool_definitions=True,
)

# Anthropic server-side compaction: once a conversation's replayed history exceeds
# the threshold, the model summarises older turns server-side and surfaces a
# CompactionPart we round-trip via the persisted ModelMessage history (docs §7).
# 150k sits ~2.6x below CHAT_USAGE_LIMITS' 400k hard cap, so compaction (graceful)
# fires well before UsageLimitExceeded (degrade). Inert until a long, page-heavy
# chat actually grows; needs a compaction-capable model (Sonnet 4.6 ✓, not Haiku).
qa_agent = Agent(
    AnthropicModel(QA_MODEL),
    deps_type=AppDeps,
    output_type=Answer,  # structured output; str kept out of the union (forces it)
    instructions=INSTRUCTIONS,
    model_settings=_QA_SETTINGS,
    capabilities=[AnthropicCompaction(token_threshold=150_000)],
    retries=2,
)


@qa_agent.instructions
def todays_date(ctx: RunContext[AppDeps]) -> str:
    """Inject today's date so the agent can resolve 'as at today' questions."""
    return f"Today's date is {datetime.now(UTC):%d %B %Y}."


@qa_agent.instructions
async def current_bundle(ctx: RunContext[AppDeps]) -> str:
    """Inject the CURRENT document bundle every turn.

    Instructions are re-sent each turn and are NOT part of the replayed message
    history, so this list is always live — even when a prior turn's
    `list_documents()` result (frozen in the replayed history) is stale because a
    document was added or removed since. This is what keeps a document uploaded
    mid-conversation from being invisible.
    """
    docs = await document_service.list_documents_for_conversation(
        ctx.deps.db, ctx.deps.conversation_id
    )
    if not docs:
        return "The document bundle is currently EMPTY (0 documents)."
    listing = "\n".join(
        f"- {d.filename} (document_id: {d.id}, {d.page_count} pages)" for d in docs
    )
    return (
        f"The bundle currently contains {len(docs)} document(s), as of right now:\n"
        f"{listing}\n"
        "This list is always current and authoritative. If it differs from a "
        "`list_documents()` result earlier in the conversation, documents were "
        "added or removed since — trust THIS list, and call `list_documents()` "
        "again to get any new document's card."
    )


# Self-describing tool returns: the empty state must instruct the next action so
# the agent stops instead of re-calling list_documents() until it hits the limit.
_EMPTY_BUNDLE_GUIDANCE = (
    "The bundle is EMPTY — no documents have been uploaded. Tell the user there is "
    "nothing to analyse yet and to upload documents. Do not call any more tools."
)
_BUNDLE_GUIDANCE = (
    "Pick the relevant documents from the cards, then use grep to find pages, "
    "outline to see a document's pages, and read_pages to read and quote them."
)


def _documents_payload(summaries: list[dict[str, object]]) -> dict[str, object]:
    """Wrap document summaries in a self-describing envelope (count + guidance).

    The agent reads `guidance` to decide what to do next — most importantly, the
    empty-bundle case tells it to stop and inform the user rather than loop.
    """
    return {
        "document_count": len(summaries),
        "documents": summaries,
        "guidance": _BUNDLE_GUIDANCE if summaries else _EMPTY_BUNDLE_GUIDANCE,
    }


@qa_agent.tool
async def list_documents(ctx: RunContext[AppDeps]) -> dict[str, object]:
    """List the documents in this conversation's bundle.

    Call this first (once). Returns `document_count`, the `documents` (each with
    `document_id`, `document_name`, `page_count`, and a routing `card`: type,
    parties, date/range, key topics, one-line), and `guidance` for what to do
    next. If `document_count` is 0 the bundle is empty — tell the user and stop.
    Cards are hints only, never a source.
    """
    docs = await document_service.list_documents_for_conversation(
        ctx.deps.db, ctx.deps.conversation_id
    )
    return _documents_payload(document_service.document_summaries(docs))


@qa_agent.tool
async def outline(
    ctx: RunContext[AppDeps], document_id: str
) -> list[dict[str, str | int]]:
    """Show a per-page map of one document — its table of contents.

    Returns each page with the start of its text, so you can see what's on each
    page and decide which pages to read. Then call `read_pages` on the pages you
    need.

    Args:
        document_id: id from list_documents().
    """
    pages = await document_service.get_document_outline(
        ctx.deps.db, ctx.deps.conversation_id, document_id
    )
    if pages is None:
        raise ModelRetry(
            f"No document {document_id} in this bundle. "
            "Call list_documents() for valid document ids."
        )
    return pages


@qa_agent.tool
async def grep(
    ctx: RunContext[AppDeps], pattern: str, document_id: str | None = None
) -> list[dict[str, str | int]]:
    r"""Search the bundle with a regular expression, like grep.

    Returns every matching line with its `document_name`, `document_id`, and
    `page`. Searches ALL documents unless you pass `document_id`. Page text can
    wrap mid-phrase, so use `\s+` or `.*` between the words of a phrase. You choose
    the pattern and how broad or specific to make it; an empty result means no
    line matched — try a simpler or different pattern. Use grep to find which
    pages to read, then `read_pages` to read and quote them.

    Args:
        pattern: a case-insensitive regular expression.
        document_id: optional — restrict the search to a single document.
    """
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ModelRetry(
            f"Invalid regular expression: {exc}. Fix the pattern and retry."
        ) from exc
    return await document_service.grep_pages(
        ctx.deps.db, ctx.deps.conversation_id, pattern, document_id=document_id
    )


@qa_agent.tool
async def read_pages(
    ctx: RunContext[AppDeps],
    document_id: str,
    start_page: int,
    end_page: int | None = None,
) -> str:
    """Read a range of pages — the text you quote and cite.

    Returns pages `start_page`..`end_page` (inclusive) with `--- Page N ---`
    markers, so quotes stay page-anchored: quote from a page and cite that page's
    number. Omit `end_page` to read a single page.

    Args:
        document_id: id from list_documents().
        start_page: 1-based first page.
        end_page: 1-based last page (defaults to start_page).
    """
    end = start_page if end_page is None else end_page
    text = await document_service.get_pages_text(
        ctx.deps.db, ctx.deps.conversation_id, document_id, start_page, end
    )
    if text is None:
        raise ModelRetry(
            f"No readable pages {start_page}-{end} in document {document_id}. "
            "Call list_documents() / outline() to see valid documents and page counts."
        )
    return text


def to_model_history(history: Iterable[dict[str, str]]) -> list[ModelMessage]:
    """Convert stored plain role/content messages into PydanticAI history.

    The back-compat seed for conversations predating the rich-history snapshot:
    instructions are re-sent each turn by the agent, so history carries only the
    prior turns' text (docs/pydantic-ai.md §6). Once a turn persists the full
    `all_messages()` snapshot, that snapshot is replayed instead.
    """
    messages: list[ModelMessage] = []
    for entry in history:
        role, content = entry.get("role"), entry.get("content")
        if not content:
            continue
        if role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role == "assistant":
            messages.append(ModelResponse(parts=[TextPart(content=content)]))
    return messages


def _tool_step(part: ToolCallPart, names: dict[str, str]) -> Step:
    """Build a human-readable Step from a tool call (no DB access — `names` is a
    pre-fetched document-id → filename map, so this is safe to call mid-run)."""
    try:
        args = part.args_as_dict()
    except Exception:
        args = {}
    if part.tool_name == "grep":
        pattern = str(args.get("pattern", "")).strip()
        doc_id = str(args.get("document_id", "")) or None
        where = names.get(doc_id or "", "the bundle") if doc_id else "the bundle"
        label = f"Searching {where} for “{pattern}”" if pattern else f"Searching {where}"
        return Step(kind="search", label=label, document_id=doc_id)
    if part.tool_name == "outline":
        document_id = str(args.get("document_id", "")) or None
        name = names.get(document_id or "", "a document")
        return Step(kind="list", label=f"Mapping {name}", document_id=document_id)
    if part.tool_name == "read_pages":
        document_id = str(args.get("document_id", "")) or None
        start = args.get("start_page")
        end = args.get("end_page", start)
        name = names.get(document_id or "", "a document")
        span = f"p.{start}" if (end is None or end == start) else f"p.{start}-{end}"
        return Step(
            kind="read",
            label=f"Reading {name} · {span}",
            document_id=document_id,
            page=int(start) if isinstance(start, int) else None,
        )
    if part.tool_name == "list_documents":
        return Step(kind="list", label="Scanning the bundle")
    return Step(kind="tool", label=f"Running {part.tool_name}")


async def answer_question(
    db: AsyncSession,
    conversation_id: str,
    question: str,
    message_history: list[ModelMessage] | None = None,
) -> AsyncIterator[str | Step | GroundedAnswer]:
    """Stream the agent's answer over the conversation's bundle.

    Yields, in order of occurrence: `Step`s (the agent's tool actions, as they
    happen), markdown deltas (`str`), and finally a `GroundedAnswer` whose
    citations are verified against their pages and whose `model_history` is the
    serialized full `all_messages()` snapshot for replay/compaction. The question
    is the only new prompt content — the agent reads pages on demand, so no
    document text is placed in the prompt; `message_history` carries prior turns
    (including pages already read) so a repeat needn't re-read (docs §6).
    """
    deps = AppDeps(db=db, conversation_id=conversation_id)
    # Pre-fetch filenames so the event handler can label `read_page` steps without
    # touching the DB session concurrently with the running agent.
    docs = await document_service.list_documents_for_conversation(db, conversation_id)
    names = {d.id: d.filename for d in docs}

    # The agent run (tools + streaming + verification) runs as a task and pushes
    # items onto a queue; we yield them in arrival order so steps appear live.
    queue: asyncio.Queue[str | Step | GroundedAnswer | None] = asyncio.Queue()

    async def on_event(
        ctx: RunContext[AppDeps], event_stream: AsyncIterable[AgentStreamEvent]
    ) -> None:
        async for event in event_stream:
            if isinstance(event, FunctionToolCallEvent):
                queue.put_nowait(_tool_step(event.part, names))

    async def run() -> None:
        try:
            async with qa_agent.run_stream(
                question,
                deps=deps,
                message_history=message_history,
                usage_limits=CHAT_USAGE_LIMITS,
                event_stream_handler=on_event,
            ) as result:
                streamed = ""
                async for partial in result.stream_output():
                    markdown = partial.markdown or ""
                    if markdown != streamed:
                        queue.put_nowait(markdown[len(streamed) :])
                        streamed = markdown
                answer = await result.get_output()
                # get_output() finalises the run, so all_messages() now includes
                # this turn's tool calls/returns and the final structured response
                # (the stream_text(delta=True) gotcha in §5 doesn't apply here).
                serialized_history = to_jsonable_python(result.all_messages())
            markdown, verified = await verify_and_renumber(
                db, conversation_id, answer.markdown, answer.citations
            )
            queue.put_nowait(
                GroundedAnswer(
                    markdown=markdown,
                    citations=verified,
                    model_history=serialized_history,
                )
            )
        except (UsageLimitExceeded, UnexpectedModelBehavior):
            # The agent hit its step budget (e.g. it kept looping). Degrade to a
            # graceful, source-free answer instead of surfacing a raw error.
            message = (
                "I couldn't complete that within my step budget. Try a more "
                "specific question, or ask about a particular document."
            )
            queue.put_nowait(message)
            queue.put_nowait(GroundedAnswer(markdown=message, citations=[]))
        finally:
            queue.put_nowait(None)  # sentinel: run finished (success or error)

    task = asyncio.create_task(run())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
        await task  # re-raise any error from the run so the caller can handle it
    finally:
        if not task.done():
            task.cancel()


title_agent = Agent(TITLE_MODEL)


async def generate_title(user_message: str) -> str:
    """Generate a 3-5 word conversation title from the first user message."""
    result = await title_agent.run(
        f"Generate a concise 3-5 word title for a conversation that starts with: "
        f"'{user_message}'. Return only the title, nothing else."
    )
    title = str(result.output).strip().strip('"').strip("'")
    if len(title) > 100:
        title = title[:97] + "..."
    return title
