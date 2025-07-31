"""
Database models for the Agent Team Builder Service.

This module defines the SQLAlchemy ORM models for:
- AgentTeam: Stores the team configuration and metadata
- AgentTeamConfigVersion: Tracks each version of a team's config
"""

from sqlalchemy import Column, Integer, Boolean, DateTime, Text, ForeignKey, String, Index, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()


class AgentTeam(Base):
    """Model representing a stored agent team configuration with metadata."""
    __tablename__ = "agent_teams"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    tenant_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    config_json = Column(JSON, nullable=False)           # Store full JSON of AgentTeamConfig
    is_deleted = Column(Boolean, default=False)           # Soft-delete flag
    deleted_at = Column(DateTime(timezone=True), nullable=True)    # Timestamp of deletion
    # Use a function default for SQLite compatibility
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)


class AgentTeamConfigVersion(Base):
    """Model representing a specific version of an agent team's configuration."""
    __tablename__ = "agent_team_config_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    team_id = Column(UUID(as_uuid=True), ForeignKey("agent_teams.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    config_json = Column(JSON, nullable=False)           # JSON of the config for this version
    # Use a function default for SQLite compatibility
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    # Create an index on team_id and version
    __table_args__ = (
        Index("ix_agent_team_config_versions_team_id_version", "team_id", "version", unique=True),
    )
