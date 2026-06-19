"""Shared test helpers, imported across the test modules."""

from __future__ import annotations

import json
from typing import Any

import pymupdf
from pydantic_ai.messages import ModelMessage, ToolReturnPart, UserPromptPart


def parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse an SSE response body into its list of `data:` JSON events."""
    return [
        json.loads(line[len("data: ") :])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


def tool_returns(messages: list[ModelMessage]) -> int:
    """Count the ToolReturnParts across a message history."""
    return sum(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in getattr(message, "parts", [])
    )


def has_tool_return(messages: list[ModelMessage]) -> bool:
    """Whether any message in the history carries a ToolReturnPart."""
    return any(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in getattr(message, "parts", [])
    )


def last_user_text(messages: list[ModelMessage]) -> str:
    """The text of the most recent user prompt in the history (or "")."""
    for message in reversed(messages):
        for part in getattr(message, "parts", []):
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                return part.content
    return ""


def page1(prompt: str) -> str:
    """Pull page 1's verbatim text out of a map prompt, so a fake map agent can
    return a quote that actually verifies against the seeded page."""
    rest = prompt.split("--- Page 1 ---\n", 1)[1]
    for stop in ("\n\n--- ", "\n--- "):
        idx = rest.find(stop)
        if idx != -1:
            rest = rest[:idx]
    return rest.strip()


def make_pdf(*pages: str) -> bytes:
    """Build an in-memory PDF, one page per text argument."""
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 72), body)  # pyright: ignore[reportUnknownMemberType]
    return doc.tobytes()  # pyright: ignore[reportUnknownMemberType]
