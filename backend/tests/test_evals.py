"""Opt-in, real-LLM evaluation suite (ticket 07).

Exercises the full pipeline — ingest → agent (search/read) → verified citations —
against the **real** model on the brief's three cross-document questions plus a
red-line "not specified" case.

Run it explicitly (it is excluded from the default test run and costs API calls):

    docker compose exec backend uv run pytest -m slow

NOTE on the bundle: the brief's answers (£1.75m rent, peppercorn, granted rights)
live in `real-docs/`, but those PDFs are **scanned with no text layer** (the
Lease is 140 pages at ~66 chars/page — a Land Registry stamp), so reading them
needs the OCR extension we have not built. To still validate the brief's exact
questions and answers reliably, this suite reproduces that scenario in small
text-bearing documents. Swap in `real-docs/` once OCR lands.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pymupdf
import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.slow

# A controlled stand-in for the real property bundle: filename -> page texts.
BUNDLE: dict[str, list[str]] = {
    "lease.pdf": [
        "LEASE dated 6 June 2008 relating to 8th Floor, Building 5, New Street "
        "Square, London.\nThe initial yearly rent is £100,000.",
        "Clause 11. Rights granted. The Landlord grants the Tenant the right of "
        "access over the Common Parts and the right to use the service media "
        "serving the Premises, together with the right of support and shelter.",
    ],
    "deed-of-variation.pdf": [
        "DEED OF VARIATION dated 15 August 2016.\nWith effect from 15 August 2016 "
        "the yearly rent reserved by the Lease is varied to a peppercorn (if "
        "demanded).\nThe Tenant is additionally granted the right to park three "
        "vehicles in the basement car park.",
    ],
    "rent-review-memorandum.pdf": [
        "RENT REVIEW MEMORANDUM dated 1 January 2024.\nFollowing the rent review on "
        "1 January 2024, the yearly rent payable under the Lease is £1.75 million "
        "per annum with effect from 1 January 2024, replacing the previous "
        "peppercorn rent.",
    ],
}


def _pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 72), body)
    return doc.tobytes()


async def _make_bundle(client: AsyncClient) -> str:
    conversation = (await client.post("/api/conversations")).json()
    cid = conversation["id"]
    for name, pages in BUNDLE.items():
        resp = await client.post(
            f"/api/conversations/{cid}/documents",
            files={"file": (name, _pdf(pages), "application/pdf")},
        )
        assert resp.status_code == 201, resp.text
    return cid


async def _ask(client: AsyncClient, cid: str, question: str) -> dict[str, Any]:
    resp = await client.post(
        f"/api/conversations/{cid}/messages", json={"content": question}
    )
    assert resp.status_code == 200, resp.text
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            event = json.loads(line[len("data: ") :])
            if event.get("type") == "message":
                return event["message"]
    raise AssertionError("no final message event in the stream")


def _cited_documents(message: dict[str, Any]) -> set[str]:
    return {c["document_name"] for c in message["citations"]}


async def test_rent_as_at_today(client: AsyncClient) -> None:
    cid = await _make_bundle(client)
    message = await _ask(client, cid, "What is the rent as at today's date?")
    content = message["content"].lower()
    assert "1.75" in content or "1,750,000" in content
    assert "rent-review-memorandum.pdf" in _cited_documents(message)


async def test_rent_on_a_past_date(client: AsyncClient) -> None:
    cid = await _make_bundle(client)
    message = await _ask(client, cid, "What was the rent on 15/08/2016?")
    assert "peppercorn" in message["content"].lower()
    assert "deed-of-variation.pdf" in _cited_documents(message)


async def test_rights_granted_to_tenant(client: AsyncClient) -> None:
    cid = await _make_bundle(client)
    message = await _ask(client, cid, "What rights are granted to the tenant?")
    content = message["content"].lower()
    assert "access" in content or "service media" in content
    # Rights are summarised across the lease and the deed's additional rights.
    assert "lease.pdf" in _cited_documents(message)


async def test_red_line_not_specified(client: AsyncClient) -> None:
    cid = await _make_bundle(client)
    message = await _ask(
        client, cid, "Does the lease grant the tenant a right to install solar panels?"
    )
    content = message["content"].lower()
    # Honest absence, and crucially no fabricated solar-panel citation.
    assert any(
        phrase in content
        for phrase in ("not specified", "does not", "no provision", "not mention", "silent")
    )
    assert not any("solar" in c["quote"].lower() for c in message["citations"])
