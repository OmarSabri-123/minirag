from .minirag_base import SQLAlchemyBase

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    func,
)

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from pgvector.sqlalchemy import VECTOR

import uuid


EMBEDDING_DIMENSION = 1024


class ChunkEmbedding(SQLAlchemyBase):

    __tablename__ = "chunk_embeddings"

    embedding_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    chunk_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "data_chunks.chunk_id",
            ondelete="CASCADE"
        ),
        nullable=False
    )

    embedding_model = Column(
        String(200),
        nullable=False
    )

    embedding_dimension = Column(
        Integer,
        default=EMBEDDING_DIMENSION,
        nullable=False
    )

    embedding = Column(
        VECTOR(EMBEDDING_DIMENSION),
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

    chunk = relationship(
        "DataChunk",
        back_populates="embeddings"
    )

    __table_args__ = (

        UniqueConstraint(
            "chunk_id",
            "embedding_model",
            name="uq_chunk_embedding_model"
        ),

        Index(
            "ix_chunk_embeddings_vector_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={
                "embedding": "vector_cosine_ops"
            }
        ),
    )