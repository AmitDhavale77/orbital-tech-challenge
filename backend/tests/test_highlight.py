from __future__ import annotations

from pathlib import Path

import pymupdf

from takehome.services.document import compute_quote_rects


def _write_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)  # pyright: ignore[reportUnknownMemberType]
    doc.save(str(path))  # pyright: ignore[reportUnknownMemberType]


def test_rects_found_for_a_known_quote(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, "Clause 3.1 The annual rent is GBP 850,000 per annum.")

    rects, width, height = compute_quote_rects(str(pdf), 1, "annual rent")

    assert rects, "expected at least one bounding box"
    assert len(rects[0]) == 4  # [x0, y0, x1, y1]
    x0, y0, x1, y1 = rects[0]
    assert x1 > x0 and y1 > y0  # a real, positive-area box
    assert width > 0 and height > 0  # page dimensions in PDF points


def test_absent_quote_yields_no_rects_but_keeps_page_dimensions(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    _write_pdf(pdf, "Some unrelated text on the page.")

    rects, width, height = compute_quote_rects(str(pdf), 1, "a phrase not present")

    assert rects == []
    assert width > 0 and height > 0
