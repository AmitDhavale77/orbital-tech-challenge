"""Add steps JSON column to messages (agent trace)

Revision ID: 005_message_steps
Revises: 004_page_tsv
Create Date: 2026-06-14 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "005_message_steps"
down_revision: str | None = "004_page_tsv"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("messages", sa.Column("steps", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("messages", "steps")
