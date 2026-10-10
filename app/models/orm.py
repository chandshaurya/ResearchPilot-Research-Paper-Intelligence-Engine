"""SQLAlchemy tables: papers and chunks (with dense vector + full-text vector)."""
from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core import config


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(primary_key=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    categories: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    published: Mapped[str | None] = mapped_column(String(32))
    pdf_url: Mapped[str | None] = mapped_column(Text)
    num_pages: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    paper_id: Mapped[int] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    section: Mapped[str | None] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(16), default="text")  # "text" | "caption"
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)

    embedding: Mapped[list[float]] = mapped_column(Vector(config.EMBED_DIM))
    # Generated column: kept in sync by Postgres, powers the sparse (keyword) retriever.
    tsv = mapped_column(TSVECTOR, Computed("to_tsvector('english', content)", persisted=True))

    paper: Mapped[Paper] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("paper_id", "chunk_index", name="uq_chunk_paper_index"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )