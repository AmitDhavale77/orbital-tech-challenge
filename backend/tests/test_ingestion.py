from __future__ import annotations

import io

import pytest
from fastapi import UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from takehome.db.models import Conversation, Page
from takehome.services.document import (
    DuplicateDocumentError,
    grep_pages,
    upload_document,
)
from tests.helpers import make_pdf


async def test_upload_creates_one_page_per_pdf_page(db_session: AsyncSession) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.commit()

    content = make_pdf("Hello page one rent", "Page two break clause")
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


async def test_uploaded_pages_are_greppable(db_session: AsyncSession) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.commit()

    content = make_pdf(
        "The tenant shall pay the rent quarterly.", "Break clause notice period."
    )
    upload = UploadFile(filename="lease.pdf", file=io.BytesIO(content))
    document = await upload_document(db_session, conversation.id, upload)

    # The pages are stored and grep-able by the agent's keyword tool.
    results = await grep_pages(db_session, conversation.id, "rent")
    assert any(r["document_id"] == document.id for r in results)


async def test_same_pdf_cannot_be_uploaded_twice(db_session: AsyncSession) -> None:
    conversation = Conversation()
    db_session.add(conversation)
    await db_session.commit()

    content = make_pdf("Identical bytes.")
    first = UploadFile(filename="dup.pdf", file=io.BytesIO(content))
    await upload_document(db_session, conversation.id, first)

    # Re-uploading the byte-for-byte identical PDF into the same bundle is rejected.
    second = UploadFile(filename="dup-renamed.pdf", file=io.BytesIO(content))
    with pytest.raises(DuplicateDocumentError):
        await upload_document(db_session, conversation.id, second)
