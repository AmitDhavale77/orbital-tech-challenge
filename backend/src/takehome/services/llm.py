from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.services import (
    document as document_service,  # imports config → exports ANTHROPIC_API_KEY
)
from takehome.services.citations import Answer, verify_citations

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
    "- `list_documents()` to see the documents in the bundle and their page counts.\n"
    "- `read_page(document_id, page)` to read one page's text on demand.\n\n"
    "Workflow: call `list_documents()` first, then read the pages you need to "
    "answer the question. Read more pages if the answer is not yet clear.\n\n"
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


@qa_agent.tool
async def list_documents(ctx: RunContext[AppDeps]) -> list[dict[str, str | int]]:
    """List the documents in this conversation's bundle.

    Call this first. Returns one entry per document with its `document_id`,
    `document_name`, and `page_count`. Use the `document_id` values with
    `read_page`.
    """
    docs = await document_service.list_documents_for_conversation(
        ctx.deps.db, ctx.deps.conversation_id
    )
    return [
        {
            "document_id": d.id,
            "document_name": d.filename,
            "page_count": d.page_count,
        }
        for d in docs
    ]


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


async def answer_question(
    db: AsyncSession,
    conversation_id: str,
    question: str,
    history: Iterable[dict[str, str]],
) -> AsyncIterator[str | Answer]:
    """Stream the agent's answer over the conversation's bundle.

    Yields markdown deltas (str) as they stream, then a final `Answer` whose
    citations have been verified against their cited pages. The question is the
    only prompt content — the agent reads pages on demand via its tools, so no
    document text is ever placed in the prompt.
    """
    deps = AppDeps(db=db, conversation_id=conversation_id)
    async with qa_agent.run_stream(
        question,
        deps=deps,
        message_history=_to_model_history(history),
        usage_limits=CHAT_USAGE_LIMITS,
    ) as result:
        streamed = ""
        async for partial in result.stream_output():
            markdown = partial.markdown or ""
            if markdown != streamed:
                yield markdown[len(streamed) :]
                streamed = markdown
        answer = await result.get_output()

    verified = await verify_citations(db, conversation_id, answer.citations)
    yield Answer(markdown=answer.markdown, citations=verified)


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
