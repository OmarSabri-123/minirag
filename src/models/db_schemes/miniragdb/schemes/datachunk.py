from .minirag_base import SQLAlchemyBase
from pydantic import BaseModel

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    func,
)

from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

import uuid


class DataChunk(SQLAlchemyBase):

    __tablename__ = "data_chunks"

    chunk_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "documents.document_id",
            ondelete="CASCADE"
        ),
        nullable=False
    )

    chunk_index = Column(
        Integer,
        nullable=False
    )

    content = Column(
        Text,
        nullable=False
    )

    token_count = Column(
        Integer,
        nullable=True
    )

    chunk_metadata = Column(
        "metadata",
        JSONB,
        nullable=True
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

    document = relationship(
        "Document",
        back_populates="chunks"
    )

    embeddings = relationship(
        "ChunkEmbedding",
        back_populates="chunk"
    )

    __table_args__ = (

        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunk_index"
        ),

        Index(
            "ix_data_chunks_document_id",
            "document_id"
        ),
    )
    

class RetrievedDocument(BaseModel):
    text: str
    score: float