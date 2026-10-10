"""Hybrid search & rerank endpoints."""
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.schemas import HitOut, SearchRequest, SearchResponse
from app.services import retriever

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
def search(req: SearchRequest, db: Session = Depends(get_db)):
    paper_ids = retriever.paper_ids_for(db, req.arxiv_ids)
    if req.arxiv_ids and not paper_ids:
        raise HTTPException(404, "None of the requested arxiv_ids are ingested")
    t0 = time.perf_counter()
    hits = retriever.search(db, req.query, req.top_k, req.mode, req.rerank, paper_ids)
    return SearchResponse(
        query=req.query, mode=req.mode, reranked=req.rerank,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        hits=[HitOut(**h.__dict__) for h in hits],
    )


@router.get("/chunk/{chunk_id}", response_model=HitOut)
def get_chunk(chunk_id: int, db: Session = Depends(get_db)):
    """Resolve a citation back to its passage and pages."""
    hit = retriever.load_hits(db, [chunk_id]).get(chunk_id)
    if hit is None:
        raise HTTPException(404, "chunk not found")
    return HitOut(**hit.__dict__)