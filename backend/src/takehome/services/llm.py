from __future__ import annotations

import re
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

# Capable model for the reasoning/tool loop; Haiku is reserved for cheap aux
# calls (conversation titles). See CLAUDE.md and docs/pydantic-ai.md.
QA_MODEL = "claude-sonnet-4-6"
TITLE_MODEL = "anthropic:claude-haiku-4-5-20251001"

# Bounds the worst-case latency/cost of the agentic loop (docs/pydantic-ai.md §9).
CHAT_USAGE_LIMITS = UsageLimits(
    request_limit=6,
    tool_calls_limit=12,
    total_tokens_limit=200_000,
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
    "- If the bundle does not contain the answer, say \"Not specified\" rather "
    "than speculating.\n"
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
) -> AsyncIterator[str]:
    """Stream the agent's answer to a question over the conversation's bundle.

    The question is the only prompt content — the agent reads pages on demand via
    its tools, so no document text is ever placed in the prompt.
    """
    deps = AppDeps(db=db, conversation_id=conversation_id)
    async with qa_agent.run_stream(
        question,
        deps=deps,
        message_history=_to_model_history(history),
        usage_limits=CHAT_USAGE_LIMITS,
    ) as result:
        async for delta in result.stream_text(delta=True):
            yield delta


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


def count_sources_cited(response: str) -> int:
    """Count references to document sections, clauses, pages, etc.

    Interim heuristic retained from the baseline; replaced by a real verified
    citation count in ticket 02.
    """
    patterns = [
        r"section\s+\d+",
        r"clause\s+\d+",
        r"page\s+\d+",
        r"paragraph\s+\d+",
    ]
    return sum(len(re.findall(p, response, re.IGNORECASE)) for p in patterns)
