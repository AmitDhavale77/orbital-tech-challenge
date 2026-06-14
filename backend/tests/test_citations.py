from __future__ import annotations

from takehome.services.citations import verify_quote


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
