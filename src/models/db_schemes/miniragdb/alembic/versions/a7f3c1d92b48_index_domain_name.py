"""Index domains.name

Revision ID: a7f3c1d92b48
Revises: 92609297a596
Create Date: 2026-08-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a7f3c1d92b48'
down_revision: Union[str, Sequence[str], None] = '92609297a596'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # replace the implicit unique constraint with an explicit unique index
    op.drop_constraint('domains_name_key', 'domains', type_='unique')
    op.create_index('ix_domains_name', 'domains', ['name'], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_domains_name', table_name='domains')
    op.create_unique_constraint('domains_name_key', 'domains', ['name'])
