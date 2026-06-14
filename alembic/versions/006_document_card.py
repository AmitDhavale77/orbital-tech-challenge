"""Add routing card JSON column to documents (ticket 06)

Revision ID: 006_document_card
Revises: 005_message_steps
Create Date: 2026-06-14 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "006_document_card"
down_revision: str | None = "005_message_steps"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("card", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "card")
