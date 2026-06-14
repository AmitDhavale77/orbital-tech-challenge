from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.citations import Citation, verify_and_renumber, verify_quote


def test_exact_substring_matches() -> None:
    page = "Clause 3.1  The rent is £1.75 million per annum."
    assert verify_quote("The rent is £1.75 million", page)


def test_whitespace_and_newlines_are_tolerated() -> None:
    # PDF extraction often wraps lines and doubles spaces.
    page = "The rent is\n   £1.75    million\nper annum"
    assert verify_quote("The rent is £1.75 million per annum", page)


def test_mismatched_quote_is_rejected() -> None:
    page = "The rent is £1.75 million per annum."
    assert not verify_quote("The rent is £2 million", page)


def test_empty_quote_is_rejected() -> None:
    assert not verify_quote("", "any page text")
    assert not verify_quote("   ", "any page text")


def test_smart_punctuation_is_tolerated() -> None:
    # Page uses straight quotes / hyphen; model quotes with curly quotes / en-dash.
    page = 'the "Initial Rent" of £850,000 - payable quarterly'
    quote = "the “Initial Rent” of £850,000 – payable quarterly"
    assert verify_quote(quote, page)


async def test_verify_and_renumber_drops_orphan_markers_and_renumbers(
    db_session: AsyncSession,
) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.flush()
    document = Document(
        conversation_id=conversation.id,
        filename="lease.pdf",
        file_path="/tmp/lease.pdf",
        page_count=2,
    )
    document.pages = [
        Page(page_number=1, text="Alpha clause about rent."),
        Page(page_number=2, text="Gamma clause about use."),
    ]
    db_session.add(document)
    await db_session.commit()

    markdown = "Rent is X.[1] Beta point.[2] Use is Y.[3]"
    citations = [
        Citation(document_id=document.id, document_name="lease.pdf", page=1, quote="Alpha clause"),
        Citation(document_id=document.id, document_name="lease.pdf", page=2, quote="BETA NOT ON PAGE"),
        Citation(document_id=document.id, document_name="lease.pdf", page=2, quote="Gamma clause"),
    ]

    new_markdown, verified = await verify_and_renumber(
        db_session, conversation.id, markdown, citations
    )

    # Only the two verifiable citations survive, renumbered 1..2 in order.
    assert [c.quote for c in verified] == ["Alpha clause", "Gamma clause"]
    # The dropped citation's marker is removed; survivors are contiguous [1], [2].
    assert new_markdown == "Rent is X.[1] Beta point. Use is Y.[2]"
