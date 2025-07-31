from sqlalchemy import Column, String, Float, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import func

from .base import Base

class DocumentMetadata(Base):
    """Metadata for a knowledge document or transcript, including classification and provenance."""
    __tablename__ = 'document_metadata'
    id = Column(UUID(as_uuid=True), primary_key=True, comment="Primary key (UUID) for the document metadata record")
    tenant_id = Column(UUID(as_uuid=True), ForeignKey('tenant.id', ondelete="CASCADE"), nullable=False,
                       comment="Tenant owning this document (FK to tenant.id)")
    app_id = Column(UUID(as_uuid=True), ForeignKey('app.id', ondelete="CASCADE"), nullable=False,
                   comment="App within tenant context for this document (FK to app.id)")
    feature_id = Column(UUID(as_uuid=True), ForeignKey('feature.id'), nullable=True,
                       comment="Main feature associated with this document (FK to feature.id)")
    source_type_id = Column(ForeignKey('source_type.id'), nullable=False, comment="FK to SourceType (origin of document)")
    staleness_type_id = Column(ForeignKey('staleness_type.id'), nullable=False, comment="FK to StalenessType (content freshness category)")
    staleness_score = Column(Float, nullable=True, comment="Numeric freshness score (0.0 very stale to 1.0 very fresh)")
    content_type_id = Column(ForeignKey('content_type.id'), nullable=False, comment="FK to ContentType (nature of content)")
    document_title = Column(String(255), nullable=False, comment="Title or name of the document")
    created_by = Column(String(100), nullable=True, comment="Name of the person who created the document")
    created_date = Column(DateTime(timezone=True), server_default=func.now(), nullable=True,
                         comment="Timestamp when the document was created")
    updated_by = Column(String(100), nullable=True, comment="Name of the last person who updated the document")
    updated_date = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=True,
                         comment="Timestamp when the document was last updated")
    __table_args__ = (
        Index('ix_document_metadata_tenant_id', 'tenant_id'),
        Index('ix_document_metadata_app_id', 'app_id'),
        Index('ix_document_metadata_feature_id', 'feature_id'),
    )

    feature = relationship('Feature', back_populates='documents')
    source_type = relationship('SourceType', back_populates='documents')
    staleness_type = relationship('StalenessType', back_populates='documents')
    content_type = relationship('ContentType', back_populates='documents')

