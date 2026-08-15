from .minirag_base import SQLAlchemyBase

from sqlalchemy import (
    Column,
    String,
    DateTime,
    ForeignKey,
    Index,
    func,
)

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

import uuid


class Document(SQLAlchemyBase):

    __tablename__ = "documents"

    document_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    domain_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "domains.domain_id",
            ondelete="CASCADE"
        ),
        nullable=False
    )

    sub_domain_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "sub_domains.sub_domain_id",
            ondelete="SET NULL"
        ),
        nullable=True
    )

    title = Column(
        String(500),
        nullable=False
    )

    source_name = Column(
        String(500),
        nullable=False
    )

    source_type = Column(
        String(100),
        nullable=False
    )

    language = Column(
        String(20),
        default="ar-en",
        nullable=False
    )

    content_hash = Column(
        String(64),
        unique=True,
        nullable=False
    )

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    updated_at = Column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True
    )

    domain = relationship(
        "Domain",
        back_populates="documents"
    )

    sub_domain = relationship(
        "SubDomain",
        back_populates="documents"
    )

    chunks = relationship(
        "DataChunk",
        back_populates="document"
    )

    __table_args__ = (
        Index(
            "ix_documents_domain_sub_domain",
            "domain_id",
            "sub_domain_id"
        ),
    )