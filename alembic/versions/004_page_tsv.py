"""Add full-text vector to pages for keyword search (ticket 04)

Revision ID: 004_page_tsv
Revises: 003_message_citations
Create Date: 2026-06-14 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "004_page_tsv"
down_revision: str | None = "003_message_citations"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "pages",
        sa.Column(
            "tsv",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', text)", persisted=True),
            nullable=True,
        ),
    )
    op.create_index("ix_pages_tsv", "pages", ["tsv"], postgresql_using="gin")


def downgrade() -> None:
    op.drop_index("ix_pages_tsv", table_name="pages")
    op.drop_column("pages", "tsv")
