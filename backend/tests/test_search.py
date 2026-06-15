from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.document import search_pages

# NOTE: the keyword `search` *tool* is disabled (routing goes via cards →
# read_document — see ADR-0002 / docs/research/architectures.md), but the
# `search_pages` *service* and the `pages.tsv` index are kept so a future
# reranked/hybrid search can be re-enabled. These tests guard that service.


async def _seed_bundle(db: AsyncSession) -> tuple[str, str, str]:
    conversation = Conversation()
    db.add(conversation)
    await db.flush()

    # A long lease that mentions "rent" on every one of its 5 pages.
    lease = Document(
        conversation_id=conversation.id,
        filename="lease.pdf",
        file_path="/tmp/lease.pdf",
        page_count=5,
    )
    lease.pages = [Page(page_number=i + 1, text="rent " * 50) for i in range(5)]

    # A short deed that mentions "rent" once.
    deed = Document(
        conversation_id=conversation.id,
        filename="deed.pdf",
        file_path="/tmp/deed.pdf",
        page_count=1,
    )
    deed.pages = [Page(page_number=1, text="the rent was varied to a peppercorn")]
    db.add_all([lease, deed])
    await db.commit()
    return conversation.id, lease.id, deed.id


async def test_search_ranks_and_diversifies_across_documents(
    db_session: AsyncSession,
) -> None:
    conversation_id, lease_id, deed_id = await _seed_bundle(db_session)

    results = await search_pages(
        db_session, conversation_id, "rent", per_document=2, limit=8
    )

    assert results, "expected hits"
    assert set(results[0]) >= {"document_id", "document_name", "page", "preview"}
    # Highest-ranked hit is from the lease (more frequent term).
    assert results[0]["document_id"] == lease_id
    # Per-document cap: the long lease can't take more than 2 slots...
    assert sum(r["document_id"] == lease_id for r in results) <= 2
    # ...so the short deed still surfaces despite ranking lower.
    assert any(r["document_id"] == deed_id for r in results)


async def test_preview_contains_the_matched_keyword(db_session: AsyncSession) -> None:
    conversation_id, _, _ = await _seed_bundle(db_session)
    results = await search_pages(db_session, conversation_id, "peppercorn")
    hit = next(r for r in results if "peppercorn" in str(r["preview"]).lower())
    assert "peppercorn" in str(hit["preview"]).lower()


async def test_search_is_scoped_to_the_conversation(db_session: AsyncSession) -> None:
    conversation_id, _, _ = await _seed_bundle(db_session)
    _, other_lease_id, _ = await _seed_bundle(db_session)

    results = await search_pages(db_session, conversation_id, "rent")
    assert other_lease_id not in {r["document_id"] for r in results}
