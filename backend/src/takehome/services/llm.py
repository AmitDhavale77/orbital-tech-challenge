from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
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
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
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
# Sized to read a small bundle page-by-page (each read_page is one request); the
# keyword search tool (ticket 04) is what keeps this from growing with doc count.
CHAT_USAGE_LIMITS = UsageLimits(
    request_limit=20,
    tool_calls_limit=25,
    total_tokens_limit=400_000,
)

INSTRUCTIONS = (
    "You are a precise assistant for commercial real estate lawyers reviewing a "
    "Document Bundle during due diligence.\n\n"
    "You cannot see any document text until you read it. Use your tools:\n"
    "- `list_documents()` to see the documents in the bundle, each with a routing "
    "`card` (type, parties, date, topics, one-line). Cards are hints for choosing "
    "what to open — treat them as guidance only: never quote or cite a card, and "
    "never conclude a fact is absent from a card alone; search/read to confirm.\n"
    "- `search(query)` to find candidate pages across the bundle by keyword; it "
    "returns a preview fragment per hit to help you choose.\n"
    "- `read_page(document_id, page)` to read one page's full text on demand.\n\n"
    "Workflow: for anything but the smallest bundle, `search(query)` first to find "
    "candidate pages, then `read_page` the ones that look relevant to read the full "
    "page and quote from it — never quote from a search preview alone. Re-run "
    "`search` with different keywords if needed. For a tiny bundle you may read pages "
    "directly. Read more pages if the answer is not yet clear.\n\n"
    "Rules:\n"
    "- Base every statement on text you have actually read with `read_page`. "
    "Never guess or rely on prior knowledge of the document.\n"
    "- Support each factual claim with a citation: the document, the page, and a "
    "`quote` copied VERBATIM from that page's text (the quote is checked against "
    "the page, so it must match exactly). Add `[1]`, `[2]` markers in the markdown "
    "matching the citation order.\n"
    "- If the bundle does not contain the answer, say \"Not specified\" with no "
    "citation rather than speculating.\n"
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

qa_agent = Agent(
    AnthropicModel(QA_MODEL),
    deps_type=AppDeps,
    output_type=Answer,  # structured output; str kept out of the union (forces it)
    instructions=INSTRUCTIONS,
    model_settings=_QA_SETTINGS,
    retries=2,
)


@qa_agent.instructions
def todays_date(ctx: RunContext[AppDeps]) -> str:
    """Inject today's date so the agent can resolve 'as at today' questions."""
    return f"Today's date is {datetime.now(UTC):%d %B %Y}."


@qa_agent.tool
async def list_documents(ctx: RunContext[AppDeps]) -> list[dict[str, object]]:
    """List the documents in this conversation's bundle.

    Call this first. Returns one entry per document with its `document_id`,
    `document_name`, `page_count`, and a `card` (a routing summary: type,
    parties, date/range, key topics, one-line). Use the cards to decide which
    documents to `search`/`read_page` — they are hints only, never a source.
    """
    docs = await document_service.list_documents_for_conversation(
        ctx.deps.db, ctx.deps.conversation_id
    )
    return document_service.document_summaries(docs)


@qa_agent.tool
async def search(ctx: RunContext[AppDeps], query: str) -> list[dict[str, str | int]]:
    """Find the most relevant pages across the bundle by keyword.

    Returns page-level hits — `document_id`, `document_name`, `page`, and a
    `preview` (the keyword-in-context fragment) — diversified across documents so
    a long document can't crowd out a short one. The preview is only a hint for
    deciding what to open: call `read_page` on a hit to read the full page and
    quote it. Re-run with different keywords if nothing relevant comes back.

    Args:
        query: keywords or a natural-language phrase (exact terms like dates,
            clause numbers, and defined terms work best).
    """
    return await document_service.search_pages(
        ctx.deps.db, ctx.deps.conversation_id, query
    )


@qa_agent.tool
async def read_page(ctx: RunContext[AppDeps], document_id: str, page: int) -> str:
    """Return the full text of one page — the ONLY source you may quote.

    Args:
        document_id: id from list_documents().
        page: 1-based page number.
    """
    text = await document_service.get_page_text(
        ctx.deps.db, ctx.deps.conversation_id, document_id, page
    )
    if text is None:
        raise ModelRetry(
            f"No page {page} found in document {document_id}. "
            "Call list_documents() to see valid documents and their page counts."
        )
    return text


def _to_model_history(history: Iterable[dict[str, str]]) -> list[ModelMessage]:
    """Convert stored plain role/content messages into PydanticAI history.

    Instructions are re-sent each turn by the agent, so history carries only the
    prior turns (docs/pydantic-ai.md §6).
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
    if part.tool_name == "search":
        query = str(args.get("query", "")).strip()
        label = f"Searching the bundle for “{query}”" if query else "Searching the bundle"
        return Step(kind="search", label=label)
    if part.tool_name == "read_page":
        document_id = str(args.get("document_id", "")) or None
        page = args.get("page")
        name = names.get(document_id or "", "a document")
        return Step(
            kind="read",
            label=f"Reading {name} · p.{page}",
            document_id=document_id,
            page=int(page) if isinstance(page, int) else None,
        )
    if part.tool_name == "list_documents":
        return Step(kind="list", label="Scanning the bundle")
    return Step(kind="tool", label=f"Running {part.tool_name}")


async def answer_question(
    db: AsyncSession,
    conversation_id: str,
    question: str,
    history: Iterable[dict[str, str]],
) -> AsyncIterator[str | Step | GroundedAnswer]:
    """Stream the agent's answer over the conversation's bundle.

    Yields, in order of occurrence: `Step`s (the agent's tool actions, as they
    happen), markdown deltas (`str`), and finally an `Answer` whose citations are
    verified against their pages. The question is the only prompt content — the
    agent reads pages on demand, so no document text is placed in the prompt.
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
                message_history=_to_model_history(history),
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
            markdown, verified = await verify_and_renumber(
                db, conversation_id, answer.markdown, answer.citations
            )
            queue.put_nowait(GroundedAnswer(markdown=markdown, citations=verified))
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
