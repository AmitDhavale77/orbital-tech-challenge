from __future__ import annotations

import io

import pymupdf
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Page
from takehome.services.document import search_pages, upload_document


def _make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        page.insert_text((72, 72), body)
    return doc.tobytes()


async def test_upload_creates_one_page_per_pdf_page(db_session: AsyncSession) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.commit()

    content = _make_pdf(["Hello page one rent", "Page two break clause"])
    upload = UploadFile(filename="lease.pdf", file=io.BytesIO(content))

    document = await upload_document(db_session, conversation.id, upload)

    rows = (
        await db_session.execute(
            select(Page)
            .where(Page.document_id == document.id)
            .order_by(Page.page_number)
        )
    ).scalars().all()

    assert [p.page_number for p in rows] == [1, 2]
    assert "rent" in rows[0].text
    assert "break clause" in rows[1].text
    assert document.page_count == 2


async def test_uploaded_pages_are_searchable(db_session: AsyncSession) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.commit()

    content = _make_pdf(
        ["The tenant shall pay the rent quarterly.", "Break clause notice period."]
    )
    upload = UploadFile(filename="lease.pdf", file=io.BytesIO(content))
    document = await upload_document(db_session, conversation.id, upload)

    # The pages' full-text vectors are populated, so the document is searchable.
    results = await search_pages(db_session, conversation.id, "rent")
    assert any(r["document_id"] == document.id for r in results)
