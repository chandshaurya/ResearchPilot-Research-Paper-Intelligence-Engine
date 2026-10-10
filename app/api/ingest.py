"""Ingestion router: arXiv (round-robin over topics) and manual PDF upload -> parse -> chunk -> embed -> pgvector."""
import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core import config
from app.core.database import SessionLocal, get_db
from app.models.orm import Chunk, Paper
from app.services.arxiv_client import ArxivClient, ArxivPaper, Topic
from app.services.chunker import chunk_paper
from app.services.parser import parse_pdf
from app.services.retriever import embed_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

# One arXiv ingest at a time: two overlapping runs share the same cursor file and race to insert
# the same papers. (With several uvicorn workers this lock is per-process; the unique-key
# handling below still keeps the database consistent.)
_ingest_lock = threading.Lock()


# ------------------------------------------------------------------ pipeline
def _no_nul(s):
    return s.replace("\x00", "") if isinstance(s, str) else s


def ingest_pdf(db: Session, pdf_path: Path, arxiv: Optional[ArxivPaper] = None) -> Paper:
    """parse -> chunk -> embed -> store. Raises ValueError for PDFs with no extractable text."""
    parsed = parse_pdf(pdf_path)
    title = _no_nul(arxiv.title if arxiv else parsed.title)
    embed_fn = embed_documents if config.SEMANTIC_CHUNKING else None
    chunks = chunk_paper(parsed, embed_fn=embed_fn)
    for c in chunks:  # last line of defence: PostgreSQL text columns cannot hold NUL bytes
        c.content, c.section = _no_nul(c.content), _no_nul(c.section)
    if not chunks:
        raise ValueError("No text extracted (scanned PDF? run OCR first).")

    # Prefix each chunk with title + section: gives the embedder context a bare paragraph lacks.
    vectors = embed_documents([f"{title} | {c.section}\n{c.content}" for c in chunks])

    paper = Paper(
        arxiv_id=arxiv.arxiv_id if arxiv else None,
        title=title,
        abstract=arxiv.abstract if arxiv else None,
        authors=arxiv.authors if arxiv else [],
        categories=arxiv.categories if arxiv else [],
        published=arxiv.published[:10] if arxiv and arxiv.published else None,
        pdf_url=arxiv.pdf_url if arxiv else None,
        num_pages=parsed.num_pages,
    )
    paper.chunks = [
        Chunk(chunk_index=c.index, content=c.content, section=c.section, kind=c.kind,
              page_start=c.page_start, page_end=c.page_end, embedding=v)
        for c, v in zip(chunks, vectors)
    ]
    db.add(paper)
    db.commit()
    return paper


# -------------------------------------------------------------------- arXiv
class TopicIn(BaseModel):
    name: str
    query: str = Field(..., examples=['cat:cs.CL AND all:"retrieval augmented generation"'])


class ArxivIngestRequest(BaseModel):
    topics: list[TopicIn]
    page_size: int = Field(config.ARXIV_PAGE_SIZE, ge=1, le=100)
    max_requests: int = Field(config.ARXIV_MAX_REQUESTS, ge=1, le=200)
    max_papers: Optional[int] = Field(None, ge=1)


def _run_arxiv_ingest(req: ArxivIngestRequest) -> None:
    try:
        _do_arxiv_ingest(req)
    finally:
        _ingest_lock.release()  # acquired by the endpoint


def _do_arxiv_ingest(req: ArxivIngestRequest) -> None:
    db = SessionLocal()
    client = ArxivClient(
        cache_dir=config.ARXIV_CACHE_DIR,
        min_interval=config.ARXIV_MIN_INTERVAL,
        cache_ttl_hours=config.ARXIV_CACHE_TTL_HOURS,
    )
    stats = {"ingested": 0, "already_in_db": 0, "failed": 0}
    try:
        topics = [Topic(t.name, t.query) for t in req.topics]
        for paper in client.round_robin(topics, req.page_size, req.max_requests, req.max_papers):
            if db.scalar(select(Paper.id).where(Paper.arxiv_id == paper.arxiv_id)):
                stats["already_in_db"] += 1  # skip the PDF download entirely
                continue
            try:
                ingest_pdf(db, client.download_pdf(paper), arxiv=paper)
                stats["ingested"] += 1
            except IntegrityError:  # someone else inserted this paper first: not a failure
                db.rollback()
                stats["already_in_db"] += 1
            except Exception:
                db.rollback()
                client.state.seen_ids.discard(paper.arxiv_id)  # let a later run retry it
                stats["failed"] += 1
                logger.exception("failed to ingest arXiv:%s", paper.arxiv_id)
    finally:
        client.close()
        db.close()
        logger.info("arXiv ingest done: %s | api_requests=%d cache_hits=%d",
                    stats, client.requests_made, client.cache_hits)


@router.post("/arxiv", status_code=202)
def ingest_arxiv(req: ArxivIngestRequest, background: BackgroundTasks):
    """Starts a background run (arXiv's 3 s/request limit makes this slow by design). Watch the logs.

    Open /docs and use POST; opening this URL in a browser sends GET, which gives 405."""
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "An arXiv ingest is already running; wait for 'arXiv ingest done' in the logs.")
    background.add_task(_run_arxiv_ingest, req)
    return {"status": "started", "topics": [t.name for t in req.topics], "max_requests": req.max_requests}


# --------------------------------------------------------------- manual PDF
@router.post("/pdf")
def ingest_uploaded_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a .pdf file")
    dest_dir = Path(config.ARXIV_CACHE_DIR) / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}.pdf"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        paper = ingest_pdf(db, dest)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"paper_id": paper.id, "title": paper.title, "pages": paper.num_pages, "chunks": len(paper.chunks)}


@router.get("/stats")
def corpus_stats(db: Session = Depends(get_db)):
    return {"papers": db.scalar(select(func.count(Paper.id))), "chunks": db.scalar(select(func.count(Chunk.id)))}