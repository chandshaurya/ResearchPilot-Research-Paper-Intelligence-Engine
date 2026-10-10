"""Hybrid retrieval: dense (pgvector) + sparse (Postgres full-text) -> RRF fusion -> cross-encoder rerank.

Note on "BM25": Postgres `ts_rank_cd` is a BM25-style lexical ranker (term frequency + cover
density) but not textbook BM25. If you need exact BM25 scoring, swap `sparse_search` for
ParadeDB `pg_search` or an in-memory `rank_bm25` index; the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core import config
from app.models.orm import Chunk, Paper

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ----------------------------------------------------------------- models
@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL, max_length=512)


def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    return get_embedder().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def embed_query(query: str) -> list[float]:
    prefix = _BGE_QUERY_PREFIX if "bge" in config.EMBED_MODEL.lower() else ""
    return get_embedder().encode(prefix + query, normalize_embeddings=True).tolist()


# ------------------------------------------------------------------- hits
@dataclass
class Hit:
    chunk_id: int
    paper_id: int
    arxiv_id: Optional[str]
    title: str
    section: Optional[str]
    page_start: int
    page_end: int
    content: str
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None


def load_hits(db: Session, chunk_ids: list[int]) -> dict[int, Hit]:
    if not chunk_ids:
        return {}
    rows = db.execute(
        select(
            Chunk.id, Chunk.paper_id, Paper.arxiv_id, Paper.title, Chunk.section,
            Chunk.page_start, Chunk.page_end, Chunk.content,
        )
        .join(Paper, Chunk.paper_id == Paper.id)
        .where(Chunk.id.in_(chunk_ids))
    ).all()
    return {r[0]: Hit(*r) for r in rows}


def paper_ids_for(db: Session, arxiv_ids: Optional[list[str]]) -> Optional[list[int]]:
    if not arxiv_ids:
        return None
    return list(db.scalars(select(Paper.id).where(Paper.arxiv_id.in_(arxiv_ids))))


# -------------------------------------------------------------- retrievers
def dense_search(db: Session, query: str, k: int, paper_ids: Optional[list[int]] = None) -> list[int]:
    dist = Chunk.embedding.cosine_distance(embed_query(query))
    stmt = select(Chunk.id).order_by(dist).limit(k)
    if paper_ids is not None:
        stmt = stmt.where(Chunk.paper_id.in_(paper_ids))
    return list(db.scalars(stmt))


def sparse_search(db: Session, query: str, k: int, paper_ids: Optional[list[int]] = None) -> list[int]:
    terms = re.findall(r"[A-Za-z0-9]+", query.lower())
    if not terms:
        return []
    # OR-semantics: natural-language questions rarely match every term (AND would return nothing).
    tsq = func.to_tsquery("english", " | ".join(dict.fromkeys(terms)))
    rank = func.ts_rank_cd(Chunk.tsv, tsq)
    stmt = select(Chunk.id).where(Chunk.tsv.op("@@")(tsq)).order_by(rank.desc()).limit(k)
    if paper_ids is not None:
        stmt = stmt.where(Chunk.paper_id.in_(paper_ids))
    return list(db.scalars(stmt))


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = config.RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def rerank(query: str, hits: list[Hit], top_k: int) -> list[Hit]:
    if not hits:
        return []
    scores = get_reranker().predict([(query, h.content) for h in hits])
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)
    return sorted(hits, key=lambda h: h.rerank_score, reverse=True)[:top_k]


# ------------------------------------------------------------------ search
def search(
    db: Session,
    query: str,
    top_k: int = 10,
    mode: str = "hybrid",  # "dense" | "sparse" | "hybrid"
    use_rerank: bool = True,
    paper_ids: Optional[list[int]] = None,
) -> list[Hit]:
    n = config.CANDIDATES_PER_RETRIEVER
    dense_ids = dense_search(db, query, n, paper_ids) if mode in ("dense", "hybrid") else []
    sparse_ids = sparse_search(db, query, n, paper_ids) if mode in ("sparse", "hybrid") else []
    dense_rank = {cid: r for r, cid in enumerate(dense_ids, 1)}
    sparse_rank = {cid: r for r, cid in enumerate(sparse_ids, 1)}

    fused = reciprocal_rank_fusion([r for r in (dense_ids, sparse_ids) if r])
    ordered = sorted(fused, key=fused.get, reverse=True)
    pool_size = max(top_k, config.RERANK_POOL) if use_rerank else top_k
    pool_ids = ordered[:pool_size]

    loaded = load_hits(db, pool_ids)
    hits = []
    for cid in pool_ids:
        h = loaded.get(cid)
        if h is None:
            continue
        h.dense_rank, h.sparse_rank, h.rrf_score = dense_rank.get(cid), sparse_rank.get(cid), fused[cid]
        hits.append(h)
    return rerank(query, hits, top_k) if use_rerank else hits[:top_k]