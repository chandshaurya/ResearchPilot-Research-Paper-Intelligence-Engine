"""Pydantic request/response schemas."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# ------------------------------------------------------------------ search


class HitOut(BaseModel):
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


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(10, ge=1, le=50)
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    rerank: bool = True
    arxiv_ids: Optional[list[str]] = None  # restrict search to these papers


class SearchResponse(BaseModel):
    query: str
    mode: str
    reranked: bool
    latency_ms: float
    hits: list[HitOut]


# ----------------------------------------------------------- verification


class VerificationOut(BaseModel):
    verdict: Literal["supported", "partially_supported", "unsupported", "contradicted", "uncited"]
    entailment: float = 0.0
    contradiction: float = 0.0
    evidence: Optional[str] = None  # best-matching passage from the cited paper
    chunk_id: Optional[int] = None
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    method: str = "nli"  # "nli" | "llm"
    note: str = ""


class VerifyRequest(BaseModel):
    claim: str = Field(..., min_length=5)
    chunk_ids: Optional[list[int]] = None  # verify against these chunks...
    arxiv_id: Optional[str] = None  # ...or against the best chunks of this paper
    top_k: int = Field(6, ge=1, le=20)


class VerifyResponse(BaseModel):
    claim: str
    result: VerificationOut


# ---------------------------------------------------------------- synthesis


class CitationOut(BaseModel):
    tag: str
    chunk_id: int
    arxiv_id: Optional[str]
    title: str
    section: Optional[str]
    page_start: int
    page_end: int


class ClaimOut(VerificationOut):
    text: str
    cite: list[str] = []


class SectionOut(BaseModel):
    heading: str
    claims: list[ClaimOut]


class ReviewRequest(BaseModel):
    question: str = Field(..., min_length=5)
    top_k: int = Field(12, ge=2, le=30)
    verify: bool = True
    multi_query: bool = False  # +1 LLM call: expands the question into sub-queries
    arxiv_ids: Optional[list[str]] = None


class ReviewResponse(BaseModel):
    question: str
    markdown: str
    sections: list[SectionOut]
    citations: list[CitationOut]
    stats: dict


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=5)
    top_k: int = Field(12, ge=2, le=30)
    arxiv_ids: Optional[list[str]] = None


class MethodRow(BaseModel):
    paper: str
    arxiv_id: Optional[str]
    task: str = ""
    method: str = ""
    data: str = ""
    metrics: str = ""
    results: str = ""
    limitations: str = ""
    citations: list[CitationOut] = []


class CompareResponse(BaseModel):
    question: str
    rows: list[MethodRow]


class ConflictsRequest(BaseModel):
    question: str = Field(..., min_length=5)
    top_k: int = Field(15, ge=4, le=40)
    arxiv_ids: Optional[list[str]] = None


class FindingOut(BaseModel):
    text: str
    arxiv_id: Optional[str]
    title: str
    page_start: int
    page_end: int
    chunk_id: int


class ConflictOut(BaseModel):
    a: FindingOut
    b: FindingOut
    contradiction: float
    similarity: float


class ConflictsResponse(BaseModel):
    question: str
    findings_checked: int
    pairs_checked: int
    conflicts: list[ConflictOut]
    caveat: str