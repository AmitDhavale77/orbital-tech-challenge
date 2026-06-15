"""Add ModelMessage history JSON column to conversations (ticket 08)

Revision ID: 007_conversation_model_history
Revises: 006_document_card
Create Date: 2026-06-15 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "007_conversation_model_history"
down_revision: str | None = "006_document_card"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column("model_history", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversations", "model_history")
