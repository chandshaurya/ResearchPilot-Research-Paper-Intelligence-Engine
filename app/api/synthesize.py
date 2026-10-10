"""Literature review, claim verification, methodology comparison and conflict detection endpoints."""
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core import config
from app.core.database import get_db
from app.models.orm import Paper
from app.models.schemas import (
    CompareRequest, CompareResponse, ConflictsRequest, ConflictsResponse,
    ReviewRequest, ReviewResponse, VerificationOut, VerifyRequest, VerifyResponse,
)
from app.services import retriever, synthesizer
from app.services.synthesizer import LLMClient, hit_to_evidence
from app.services.verifier import ClaimVerifier

router = APIRouter(prefix="/synthesize", tags=["synthesize"])


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    if not config.GROQ_API_KEY:
        raise HTTPException(503, "GROQ_API_KEY is not set")
    return LLMClient()


@lru_cache(maxsize=1)
def get_verifier() -> ClaimVerifier:
    # The LLM judge for borderline cases is optional; NLI alone works without an API key.
    return ClaimVerifier(llm=LLMClient() if config.GROQ_API_KEY else None)


def _paper_ids(db: Session, arxiv_ids):
    ids = retriever.paper_ids_for(db, arxiv_ids)
    if arxiv_ids and not ids:
        raise HTTPException(404, "None of the requested arxiv_ids are ingested")
    return ids


@router.post("/review", response_model=ReviewResponse)
def literature_review(req: ReviewRequest, db: Session = Depends(get_db)):
    """Citation-grounded literature review; each claim is verified against the chunks it cites."""
    return synthesizer.generate_review(
        db, req.question, get_llm(), get_verifier() if req.verify else None,
        top_k=req.top_k, verify=req.verify, multi_query=req.multi_query,
        paper_ids=_paper_ids(db, req.arxiv_ids),
    )


@router.post("/verify", response_model=VerifyResponse)
def verify_claim(req: VerifyRequest, db: Session = Depends(get_db)):
    """Is this claim supported by its cited paper? Works without an LLM key (NLI only)."""
    if req.chunk_ids:
        hits = list(retriever.load_hits(db, req.chunk_ids).values())
    elif req.arxiv_id:
        paper_id = db.scalar(select(Paper.id).where(Paper.arxiv_id == req.arxiv_id))
        if paper_id is None:
            raise HTTPException(404, f"arXiv:{req.arxiv_id} is not ingested")
        hits = retriever.search(db, req.claim, top_k=req.top_k, paper_ids=[paper_id])
    else:
        raise HTTPException(422, "provide chunk_ids or arxiv_id")
    if not hits:
        raise HTTPException(404, "no evidence chunks found")
    result = get_verifier().verify(req.claim, [hit_to_evidence(h) for h in hits])
    return VerifyResponse(claim=req.claim, result=VerificationOut(**result.to_dict()))


@router.post("/compare", response_model=CompareResponse)
def compare_methods(req: CompareRequest, db: Session = Depends(get_db)):
    """Side-by-side methodology table (task / method / data / metrics / results / limitations)."""
    return synthesizer.compare_methodologies(
        db, req.question, get_llm(), req.top_k, _paper_ids(db, req.arxiv_ids)
    )


@router.post("/conflicts", response_model=ConflictsResponse)
def detect_conflicts(req: ConflictsRequest, db: Session = Depends(get_db)):
    """Candidate contradictions between findings of different papers."""
    return synthesizer.find_conflicts(
        db, req.question, get_llm(), get_verifier(), req.top_k, _paper_ids(db, req.arxiv_ids)
    )