from .minirag_base import SQLAlchemyBase

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    DateTime,
    func,
)

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

import uuid


class Domain(SQLAlchemyBase):

    __tablename__ = "domains"

    domain_id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    name = Column(
        String(200),
        unique=True,
        index=True,
        nullable=False
    )

    description = Column(
        Text,
        nullable=True
    )

    is_active = Column(
        Boolean,
        default=True,
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

    sub_domains = relationship(
        "SubDomain",
        back_populates="domain"
    )

    documents = relationship(
        "Document",
        back_populates="domain"
    )