"""init_agent_teams

Revision ID: d64bdbad1e35
Revises: 
Create Date: 2025-07-30 15:01:41.358743

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd64bdbad1e35'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.
    
    Creates the agent_teams table with the following columns:
    - id: UUID primary key
    - tenant_id: UUID for tenant identification
    - config_json: JSONB to store team configuration
    - updated_at: Timestamp with timezone for tracking updates
    - created_at: Timestamp with timezone for creation time
    """
    op.create_table(
        'agent_teams',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('tenant_id', sa.UUID(), nullable=False),
        sa.Column('config_json', sa.JSON(), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    # Create an index on tenant_id for faster lookups
    op.create_index('ix_agent_teams_tenant_id', 'agent_teams', ['tenant_id'])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop the index first
    op.drop_index('ix_agent_teams_tenant_id', table_name='agent_teams')
    # Then drop the table
    op.drop_table('agent_teams')
