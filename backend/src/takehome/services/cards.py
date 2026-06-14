"""Per-document routing cards (ticket 06).

A card is a cheap, structured summary generated at ingest (Haiku) so the agent
can route across the bundle. It is a **routing hint only — never a Source**: the
agent must never cite a card; only `read_page` text is citable (ADR-0002).

Kept in a leaf module (no imports from `document`/`llm`) to avoid an import cycle.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent

CARD_MODEL = "anthropic:claude-haiku-4-5-20251001"
CARD_SAMPLE_CHARS = 6000  # a sample is enough for a routing summary; keeps it cheap


class DocumentCard(BaseModel):
    """A structured, routing-only summary of a document."""

    type: str  # e.g. "Lease", "Official Title Report", "Environmental Assessment"
    parties: list[str] = []
    date_or_range: str | None = None
    key_topics: list[str] = []
    one_line: str


_INSTRUCTIONS = (
    "You summarise a single legal/real-estate document for ROUTING ONLY — to help "
    "an assistant decide which document to open. You are NOT producing a citable "
    "source. From the provided text, extract: the document type; the parties; the "
    "date or date range it concerns; a few key topics; and a one-line summary. Be "
    "concise and factual; if a field is unclear, leave it empty or null."
)

card_agent = Agent(CARD_MODEL, output_type=DocumentCard, instructions=_INSTRUCTIONS)


async def generate_card(text: str) -> DocumentCard:
    """Generate a routing card from a sample of the document's text."""
    result = await card_agent.run(
        "Summarise this document for routing:\n\n" + text[:CARD_SAMPLE_CHARS]
    )
    return result.output
