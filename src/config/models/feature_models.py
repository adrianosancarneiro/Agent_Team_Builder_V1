from sqlalchemy import Column, String, Text, ForeignKey, CheckConstraint, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID

from .base import Base

class Feature(Base):
    """Feature represents a functional area or capability, organized hierarchically within an App."""
    __tablename__ = 'feature'
    id = Column(UUID(as_uuid=True), primary_key=True, comment="Primary key (UUID) for the feature")
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenant.id', ondelete="CASCADE"), nullable=False,
                       comment="Tenant owning this feature (redundant to app.tenant_id for isolation)")
    app_id = Column(UUID(as_uuid=True), ForeignKey('app.id', ondelete="CASCADE"), nullable=False,
                   comment="Application to which this feature belongs (FK to app.id)")
    name = Column(String(100), nullable=False, comment="Name of the feature (unique within an app)")
    description = Column(Text, nullable=True, comment="Detailed description of the feature or subfeature")
    parent_id = Column(UUID(as_uuid=True), ForeignKey('feature.id', ondelete="CASCADE"), nullable=True,
                      comment="Optional self-referential FK to a parent feature (for subfeature hierarchy)")
    __table_args__ = (
        UniqueConstraint('app_id', 'name', name='uq_feature_app_id_name'),
        Index('ix_feature_app_id', 'app_id'),
        Index('ix_feature_parent_id', 'parent_id'),
    )

    parent = relationship('Feature', remote_side=[id], back_populates='children')
    children = relationship('Feature', back_populates='parent', cascade='all, delete-orphan')

    app = relationship('App', back_populates='features')
    documents = relationship('DocumentMetadata', back_populates='feature')

class RelatedFeature(Base):
    """Association table for many-to-many relationships between features."""
    __tablename__ = 'related_feature'
    feature_id = Column(UUID(as_uuid=True), ForeignKey('feature.id', ondelete="CASCADE"), primary_key=True)
    related_feature_id = Column(UUID(as_uuid=True), ForeignKey('feature.id', ondelete="CASCADE"), primary_key=True)
    __table_args__ = (
        CheckConstraint('feature_id <> related_feature_id', name='ck_related_feature_no_self'),
        CheckConstraint('feature_id < related_feature_id', name='ck_related_feature_order'),
        Index('ix_related_feature_feature_id', 'feature_id'),
        Index('ix_related_feature_related_feature_id', 'related_feature_id'),
    )

