from .minirag_base import SQLAlchemyBase

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    func,
)

from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

import uuid


class SubDomain(SQLAlchemyBase):

    __tablename__ = "sub_domains"

    sub_domain_id = Column(
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

    name = Column(
        String(200),
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

    domain = relationship(
        "Domain",
        back_populates="sub_domains"
    )

    documents = relationship(
        "Document",
        back_populates="sub_domain"
    )

    __table_args__ = (
        UniqueConstraint(
            "domain_id",
            "name",
            name="uq_sub_domain_name_per_domain"
        ),
    )