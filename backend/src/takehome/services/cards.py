"""Per-document routing cards (ticket 06).

A card is a cheap, structured summary generated at ingest (Haiku) so the agent
can route across the bundle. It is a **routing hint only — never a Source**: the
agent must never cite a card; only `read_page` text is citable (ADR-0002).

Kept in a leaf module (no imports from `document`/`llm`) to avoid an import cycle.
"""

from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import Agent

from takehome.config import settings


class DocumentCard(BaseModel):
    """Routing-only summary of a document. Never citable."""

    kind: str
    summary: str


CARD_INSTRUCTIONS = (
    "You create a ROUTING-ONLY card for one legal/real-estate document. "
    "The card helps an assistant decide whether this document is worth opening "
    "to answer a user's question. It is NOT a source and must never be cited. "
    "\n\n"
    "Return only:\n"
    "1. kind: the kind of document, e.g. Lease, Title Report, Environmental Report, "
    "Planning Document, Deed of Variation, Valuation Report.\n"
    "2. summary: a concise factual summary of what this document appears to cover. "
    "Include only details useful for routing, such as property, parties, dates, "
    "subject matter, and notable legal/commercial topics if visible.\n\n"
    "Do not infer. If unsure, say so briefly in the summary."
)

card_agent = Agent(
    settings.card_model, output_type=DocumentCard, instructions=CARD_INSTRUCTIONS
)


async def generate_card(text: str) -> DocumentCard:
    """Generate a routing card from a sample of the document's text."""
    result = await card_agent.run(
        "Summarise this document for routing:\n\n" + text[: settings.card_sample_chars]
    )
    return result.output
