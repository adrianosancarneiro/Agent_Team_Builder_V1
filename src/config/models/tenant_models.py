from sqlalchemy import Column, String, Text, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from .base import Base

class Tenant(Base):
    """Tenant (organization or client) that owns applications and agent teams."""
    __tablename__ = 'tenant'
    id = Column(UUID(as_uuid=True), primary_key=True, comment="Primary key (UUID) for the tenant")
    name = Column(String(100), nullable=False, unique=True, comment="Unique tenant name or identifier")
    __table_args__ = (
        UniqueConstraint('name', name='uq_tenant_name'),
    )

    apps = relationship("App", back_populates="tenant", cascade="all, delete-orphan")
    agent_teams = relationship("AgentTeam", back_populates="tenant", cascade="all, delete-orphan")

class App(Base):
    """Application context under a tenant (for grouping agent teams, features, documents)."""
    __tablename__ = 'app'
    id = Column(UUID(as_uuid=True), primary_key=True, comment="Primary key (UUID) for the app")
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenant.id', ondelete="CASCADE"), nullable=False,
                       comment="Tenant owning this app (FK to tenant.id)")
    name = Column(String(100), nullable=False, comment="Name of the application (unique per tenant)")
    description = Column(Text, nullable=True, comment="Optional description of the app")
    __table_args__ = (
        UniqueConstraint('tenant_id', 'name', name='uq_app_tenant_id_name'),
        Index('ix_app_tenant_id', 'tenant_id'),
    )

    tenant = relationship("Tenant", back_populates="apps")
    features = relationship("Feature", back_populates="app", cascade="all, delete-orphan")
    agent_teams = relationship("AgentTeam", back_populates="app", cascade="all, delete-orphan")

