from sqlalchemy import Column, String, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import INTEGER

from .base import Base

class SourceType(Base):
    """Enumeration of document source/origin types (e.g., content source)."""
    __tablename__ = 'source_type'
    id = Column(INTEGER, primary_key=True, autoincrement=True, comment="Primary key (int) for the source type")
    name = Column(String(50), nullable=False, comment="Name of the source type (e.g. ConfluencePage, MeetingTranscript)")
    __table_args__ = (
        UniqueConstraint('name', name='uq_source_type_name'),
    )

    documents = relationship("DocumentMetadata", back_populates="source_type", cascade="all, delete-orphan")

class StalenessType(Base):
    """Enumeration of document staleness categories (how current or outdated content is)."""
    __tablename__ = 'staleness_type'
    id = Column(INTEGER, primary_key=True, autoincrement=True, comment="Primary key (int) for the staleness category")
    name = Column(String(50), nullable=False, comment="Name of staleness category (e.g. Deprecated, Current, etc.)")
    __table_args__ = (
        UniqueConstraint('name', name='uq_staleness_type_name'),
    )

    documents = relationship("DocumentMetadata", back_populates="staleness_type", cascade="all, delete-orphan")

class ContentType(Base):
    """Enumeration of document content types (general vs feature-specific, technical vs business)."""
    __tablename__ = 'content_type'
    id = Column(INTEGER, primary_key=True, autoincrement=True, comment="Primary key (int) for the content type")
    name = Column(String(50), nullable=False, comment="Name of content type (e.g. FeatureSpecificTechnical, GeneralBusiness)")
    __table_args__ = (
        UniqueConstraint('name', name='uq_content_type_name'),
    )

    documents = relationship("DocumentMetadata", back_populates="content_type", cascade="all, delete-orphan")

