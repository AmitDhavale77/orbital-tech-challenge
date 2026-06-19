"""Add content_hash column to documents (PDF dedup)

Revision ID: 008_document_content_hash
Revises: 007_conversation_model_history
Create Date: 2026-06-19 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "008_document_content_hash"
down_revision: str | None = "007_conversation_model_history"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("content_hash", sa.String(), nullable=True))
    op.create_index(
        "ix_documents_content_hash", "documents", ["content_hash"]
    )


def downgrade() -> None:
    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.drop_column("documents", "content_hash")
