"""add_soft_delete_and_versioning

Revision ID: 7a5c1e9f32d8
Revises: d64bdbad1e35
Create Date: 2025-07-30 15:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '7a5c1e9f32d8'
down_revision: Union[str, Sequence[str], None] = 'd64bdbad1e35'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add soft-delete fields to agent_teams table and create versioning table.
    
    Changes:
    1. Add is_deleted and deleted_at columns to agent_teams
    2. Create agent_team_config_versions table to track version history
    """
    # Add soft-delete columns to agent_teams
    op.add_column('agent_teams', sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('agent_teams', sa.Column('deleted_at', sa.TIMESTAMP(timezone=True), nullable=True))
    
    # Create new table for storing config versions
    op.create_table(
        'agent_team_config_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('team_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('config_json', postgresql.JSONB(), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['team_id'], ['agent_teams.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('ix_agent_team_config_versions_team_id', 'agent_team_config_versions', ['team_id'])
    op.create_index('ix_agent_team_config_versions_team_id_version', 'agent_team_config_versions', 
                   ['team_id', 'version'], unique=True)


def downgrade() -> None:
    """
    Revert changes: remove version table and soft-delete columns.
    """
    # Drop versioning table and indexes
    op.drop_index('ix_agent_team_config_versions_team_id_version', table_name='agent_team_config_versions')
    op.drop_index('ix_agent_team_config_versions_team_id', table_name='agent_team_config_versions')
    op.drop_table('agent_team_config_versions')
    
    # Remove soft-delete columns
    op.drop_column('agent_teams', 'deleted_at')
    op.drop_column('agent_teams', 'is_deleted')
