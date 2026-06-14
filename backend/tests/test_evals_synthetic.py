"""Comprehensive opt-in eval suite over the real `synthetic-docs/` bundle (#07).

Drives the real agent over the three actual synthetic documents and checks both
the answer content and the citing document(s) — covering single-document facts,
cross-document questions (citations must span multiple documents), and the
grounding red-line ("not specified", no fabricated citation).

Run with `just eval` (or `pytest -m slow`); excluded from the default run.
The documents are mounted at /app/synthetic-docs (docker-compose).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.slow

SYNTHETIC_DIR = Path("/app/synthetic-docs")
LEASE = "commercial-lease-100-bishopsgate.pdf"
TITLE = "title-report-lot-7.pdf"
ENV = "environmental-assessment-manchester.pdf"

_ABSENCE = (
    "not specified",
    "does not",
    "not stated",
    "not mention",
    "not contain",
    "not provide",
    "no purchase price",
    "no rent",
    "not state",
    "unable to",
    "silent",
    "not included",
    "not available",
)


@dataclass(frozen=True)
class Scenario:
    id: str
    question: str
    answer_any: tuple[str, ...] = ()  # at least one substring present (lowercased)
    answer_all: tuple[str, ...] = ()  # all substrings present
    cite_all: tuple[str, ...] = ()  # citations must include all of these documents
    min_cited_docs: int = 0  # citations span at least N distinct documents
    not_specified: bool = False  # red-line: answer signals absence
    forbid: tuple[str, ...] = field(default=())  # substrings that must NOT appear


SCENARIOS: list[Scenario] = [
    # ── Single-document facts (answer + citation to the right document) ──
    Scenario("lease-rent", "What is the annual rent payable under the lease?",
             answer_any=("850,000", "850000", "eight hundred and fifty thousand"),
             cite_all=(LEASE,)),
    Scenario("lease-permitted-use", "What is the permitted use of the premises under the lease?",
             answer_any=("e(g)(i)", "class e", "office"), cite_all=(LEASE,)),
    Scenario("lease-break-dates", "On which dates may the tenant break the lease?",
             answer_all=("2029", "2034"), cite_all=(LEASE,)),
    Scenario("lease-parties", "Who are the landlord and the tenant under the lease?",
             answer_all=("bishopsgate property holdings", "meridian consulting"),
             cite_all=(LEASE,)),
    Scenario("lease-sublet-part", "Is the tenant permitted to sublet part of the premises?",
             answer_any=("whole", "cannot", "may not", "not permitted", "prohibit", "not allowed"),
             cite_all=(LEASE,)),
    Scenario("title-number", "What is the title number for the Lot 7 property?",
             answer_any=("ln782451",), cite_all=(TITLE,)),
    Scenario("title-covenant-date", "When was the restrictive covenant affecting Lot 7 created?",
             answer_any=("1 june 1952", "1952"), cite_all=(TITLE,)),
    Scenario("title-tenure", "What is the tenure of the Lot 7 property?",
             answer_any=("freehold",), cite_all=(TITLE,)),
    Scenario("env-tank", "What is the capacity of the underground storage tank at the Manchester site?",
             answer_any=("5,000", "5000"), cite_all=(ENV,)),
    Scenario("env-phase2-cost", "What is the estimated cost of the Phase II environmental investigation?",
             answer_any=("15,000", "25,000"), cite_all=(ENV,)),
    # ── Cross-document (citations must span multiple documents) ──
    Scenario("xdoc-areas",
             "What is the approximate site area of each property described in this bundle? Cite each document.",
             answer_all=("0.34", "0.28"), cite_all=(TITLE, ENV), min_cited_docs=2),
    Scenario("xdoc-tenure",
             "Which property is held freehold and which is held under a lease? Cite the documents.",
             answer_all=("freehold", "lease"), cite_all=(TITLE, LEASE), min_cited_docs=2),
    Scenario("xdoc-inventory",
             "List each document in this bundle and the property or site it concerns, citing each.",
             cite_all=(LEASE, TITLE, ENV), min_cited_docs=3),
    # The title report *does* state a price paid — the agent must surface it.
    Scenario("title-purchase-price", "What was the purchase price paid for the Lot 7 property?",
             answer_any=("4,250,000", "four million two hundred and fifty thousand"),
             cite_all=(TITLE,)),
    # ── Red-line: bundle is silent → "not specified", no fabrication ──
    Scenario("redline-solar", "Does the lease grant the tenant a right to install solar panels on the roof?",
             not_specified=True),
    Scenario("redline-manchester-rent", "What is the annual rent for the Manchester Deansgate property?",
             not_specified=True),
]


async def _make_bundle(client: AsyncClient) -> str:
    conversation = (await client.post("/api/conversations")).json()
    cid = conversation["id"]
    for name in (LEASE, TITLE, ENV):
        data = (SYNTHETIC_DIR / name).read_bytes()
        resp = await client.post(
            f"/api/conversations/{cid}/documents",
            files={"file": (name, data, "application/pdf")},
        )
        assert resp.status_code == 201, resp.text
    return cid


async def _ask(client: AsyncClient, cid: str, question: str) -> dict[str, Any]:
    resp = await client.post(f"/api/conversations/{cid}/messages", json={"content": question})
    assert resp.status_code == 200, resp.text
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            event = json.loads(line[len("data: ") :])
            if event.get("type") == "message":
                return event["message"]
    raise AssertionError("no final message event in the stream")


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.id for s in SCENARIOS])
async def test_synthetic_bundle(client: AsyncClient, scenario: Scenario) -> None:
    cid = await _make_bundle(client)
    message = await _ask(client, cid, scenario.question)
    content = message["content"].lower()
    cited = {c["document_name"] for c in message["citations"]}

    if scenario.answer_any:
        assert any(s in content for s in scenario.answer_any), (
            f"answer missing any of {scenario.answer_any}: {content[:300]}"
        )
    for needle in scenario.answer_all:
        assert needle in content, f"answer missing {needle!r}: {content[:300]}"
    for needle in scenario.forbid:
        assert needle not in content, f"answer should not contain {needle!r}"
    for doc in scenario.cite_all:
        assert doc in cited, f"expected a citation to {doc}; cited {cited}"
    if scenario.min_cited_docs:
        assert len(cited) >= scenario.min_cited_docs, f"expected ≥{scenario.min_cited_docs} docs; cited {cited}"
    if scenario.not_specified:
        assert any(p in content for p in _ABSENCE), f"expected an absence answer: {content[:300]}"
