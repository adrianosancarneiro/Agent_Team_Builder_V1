from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, UniqueConstraint, Index, Text
import uuid
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy import func, Integer

from .base import Base

class TenantAppConfigFile(Base):
    """Versioned configuration file for an Agent Team (stores JSON config snapshots per version)."""
    __tablename__ = 'tenant_app_config_file'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="Primary key (UUID) for the config file record")
    agent_team_id = Column(UUID(as_uuid=True), ForeignKey('agent_team.id', ondelete="CASCADE"), nullable=False,
                           comment="FK to AgentTeam that this config version belongs to")
    version = Column(Integer, default=1, nullable=False, comment="Version number of this configuration")
    config_json = Column(JSON, nullable=False, comment="The configuration content (JSON) at this version")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False,
                        comment="Timestamp when this config version was created")
    __table_args__ = (
        UniqueConstraint('agent_team_id', 'version', name='uq_config_file_team_version'),
        Index('ix_config_file_team_id', 'agent_team_id'),
        {"extend_existing": True},
    )

    agent_team = relationship("AgentTeam", back_populates="versions", foreign_keys=[agent_team_id])

class AgentTeam(Base):
    """AgentTeam represents a saved multi-agent team configuration (current version and metadata)."""
    __tablename__ = 'agent_team'
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="Primary key (UUID) for the agent team")
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenant.id', ondelete="CASCADE"), nullable=False,
                       comment="Tenant that owns this agent team (FK to tenant.id)")
    app_id = Column(UUID(as_uuid=True), ForeignKey('app.id', ondelete="CASCADE"), nullable=False,
                   comment="App context for this agent team (FK to app.id)")
    main_goal = Column(Text, nullable=False, comment="Primary goal or problem statement for the agent team")
    config_jsonb = Column(JSON, nullable=False, comment="Latest agent team configuration stored as JSON")
    current_config_file_id = Column(UUID(as_uuid=True), ForeignKey('tenant_app_config_file.id', use_alter=True, name='fk_agent_team_config_file_id'), nullable=True,
                                   comment="FK to TenantAppConfigFile record representing the latest config version")
    is_deleted = Column(Boolean, default=False, nullable=False, comment="Soft-delete flag for the team (True if deleted)")
    deleted_at = Column(DateTime(timezone=True), nullable=True, comment="Timestamp when the team was soft-deleted")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False,
                        comment="Timestamp when the team was created")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False,
                        comment="Timestamp of last update to the team or its config")
    __table_args__ = (
        Index('ix_agent_team_tenant_id', 'tenant_id'),
        Index('ix_agent_team_app_id', 'app_id'),
        {"extend_existing": True},
    )

    tenant = relationship("Tenant", back_populates="agent_teams")
    app = relationship("App", back_populates="agent_teams")
    versions = relationship(
        "TenantAppConfigFile",
        back_populates="agent_team",
        cascade="all, delete-orphan",
        foreign_keys="TenantAppConfigFile.agent_team_id",
    )
    current_config_file = relationship("TenantAppConfigFile", foreign_keys=[current_config_file_id], post_update=True)

