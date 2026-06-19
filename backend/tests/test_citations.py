from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Document, Page
from takehome.services.citations import (
    Citation,
    locate_quote,
    verify_and_renumber,
    verify_quote,
)


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


# --- Tier A: formatting-tolerant matching (still exact on the actual words) --- #


def test_ligature_and_fullwidth_are_tolerated() -> None:
    # PDF text layer may carry the "ﬁ" ligature and full-width digits; the model
    # reproduces them as plain ascii. NFKC folds both before matching.
    page = "The ofﬁce ﬁxtures cost １００ pounds"  # office fixtures, 100
    assert verify_quote("the office fixtures cost 100 pounds", page)


def test_case_is_tolerated() -> None:
    page = "The Tenant shall indemnify the Landlord against all losses."
    assert verify_quote("the tenant shall indemnify the landlord", page)


def test_hyphenation_across_a_line_break_is_joined() -> None:
    # PDF wraps "interest" as "inter-\nest"; the model quotes the whole word.
    page = "the inter-\nest accrues daily"
    assert verify_quote("the interest accrues daily", page)


def test_soft_hyphen_and_zero_width_are_stripped() -> None:
    page = "the inter­est​ accrues"  # soft hyphen + zero-width space
    assert verify_quote("the interest accrues", page)


def test_ellipsis_bridges_two_spans_on_one_page() -> None:
    page = "The Tenant shall pay the Rent on the usual Quarter Days in equal instalments."
    assert verify_quote("The Tenant shall pay the Rent ... in equal instalments", page)


def test_ellipsis_fragments_out_of_order_are_rejected() -> None:
    page = "The Tenant shall pay the Rent on the usual Quarter Days in equal instalments."
    assert not verify_quote("in equal instalments ... The Tenant shall pay the Rent", page)


def test_ellipsis_fragments_split_across_pages_are_rejected() -> None:
    # Each fragment exists, but on different pages — an ellipsis quote must be
    # satisfied entirely within one page, so this is not located.
    pages = [
        (1, "The Tenant shall pay the Rent on the Quarter Days."),
        (2, "The amounts are payable in equal instalments each year."),
    ]
    quote = "The Tenant shall pay the Rent ... payable in equal instalments"
    assert locate_quote(quote, pages, preferred=1) is None


def test_short_single_fragment_stays_exact() -> None:
    # A short quote with no ellipsis keeps exact-substring behaviour: a present
    # value matches, a one-token-different value does not.
    page = "at a rent of £50,000 per annum payable quarterly"
    assert verify_quote("£50,000", page)
    assert not verify_quote("£60,000", page)


def test_ellipsis_with_only_tiny_fragments_is_rejected() -> None:
    # Fragments below the substantiality floor can't manufacture a match.
    page = "Section A then Section B then Section C across the lease."
    assert not verify_quote("A ... B ... C", page)


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


async def test_verify_corrects_a_misattributed_page(db_session: AsyncSession) -> None:
    # Reading a whole document, the model often cites the wrong page for an
    # otherwise-verbatim quote. The quote is genuinely in the document, just on a
    # different page — so it must be kept (with its page corrected), not dropped.
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
        Page(page_number=1, text="Rent and review provisions."),
        Page(page_number=2, text="The Tenant shall indemnify the Landlord against all losses."),
    ]
    db_session.add(document)
    await db_session.commit()

    markdown = "The tenant indemnifies the landlord.[1]"
    citations = [
        # Verbatim quote, but the model attributed it to page 1 (it is on page 2).
        Citation(
            document_id=document.id,
            document_name="lease.pdf",
            page=1,
            quote="The Tenant shall indemnify the Landlord against all losses.",
        )
    ]

    new_markdown, verified = await verify_and_renumber(
        db_session, conversation.id, markdown, citations
    )

    assert len(verified) == 1, "the quote is in the document and must not be dropped"
    assert verified[0].page == 2, "the cited page is corrected to where the quote is"
    assert new_markdown == "The tenant indemnifies the landlord.[1]"
