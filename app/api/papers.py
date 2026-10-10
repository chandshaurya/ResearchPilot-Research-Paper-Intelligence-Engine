"""Paper discovery and selection for the UI.

  GET  /papers/arxiv-search   live topic search on arXiv (marks papers already in your library)
  POST /papers/ingest         ingest the papers you selected (background job, one at a time)
  GET  /papers/jobs/{id}      per-paper progress for that job
  GET  /papers                your library;  DELETE /papers/{id} removes a paper
"""
import logging
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Literal, Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.ingest import ingest_pdf
from app.core import config
from app.core.database import SessionLocal, get_db
from app.models.orm import Chunk, Paper
from app.services.arxiv_client import ArxivClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/papers", tags=["papers"])

# One shared client + lock: every arXiv call in the UI flow goes through the same 3-second
# rate limiter, so a search and a download can never hit arXiv at the same moment.
arxiv_lock = threading.Lock()


@lru_cache(maxsize=1)
def shared_client() -> ArxivClient:
    return ArxivClient(
        cache_dir=config.ARXIV_CACHE_DIR,
        min_interval=config.ARXIV_MIN_INTERVAL,
        cache_ttl_hours=config.ARXIV_CACHE_TTL_HOURS,
    )


# ------------------------------------------------------------------- search
_FIELD_PREFIX = re.compile(r"\b(all|ti|abs|au|cat|co|jr|rn|id):", re.I)
_CATEGORY = re.compile(r"^[A-Za-z\-]+(\.[A-Za-z\-]+)?$")
_VERSION = re.compile(r"v\d+$")


def build_query(q: str, category: Optional[str] = None) -> str:
    """Plain words become `all:a AND all:b`; arXiv syntax like `cat:cs.CL AND ti:rag` passes through."""
    q = q.strip()
    if _FIELD_PREFIX.search(q):
        base = q
    else:
        base = " AND ".join(f"all:{w}" for w in re.findall(r"[A-Za-z0-9]+", q))
    if not base:
        raise HTTPException(422, "Enter a topic with at least one word.")
    if category:
        if not _CATEGORY.match(category):
            raise HTTPException(422, "Unknown category format.")
        return f"({base}) AND cat:{category}"
    return base


@router.get("/arxiv-search")
def arxiv_search(
    q: str = Query(..., min_length=2),
    start: int = Query(0, ge=0),
    limit: int = Query(15, ge=1, le=50),
    sort: Literal["relevance", "submittedDate"] = "relevance",
    category: Optional[str] = None,
    db: Session = Depends(get_db),
):
    query = build_query(q, category)
    try:
        with arxiv_lock:
            papers, total = shared_client().search(query, start, limit, sort)
    except (httpx.HTTPError, RuntimeError) as e:
        logger.warning("arXiv search failed: %s", e)
        raise HTTPException(502, "arXiv did not respond. Wait a few seconds and search again.")
    in_library = set(db.scalars(select(Paper.arxiv_id).where(Paper.arxiv_id.in_([p.arxiv_id for p in papers]))))
    return {
        "query": query,
        "total": total,
        "start": start,
        "results": [
            {
                "arxiv_id": p.arxiv_id, "title": p.title, "abstract": p.abstract, "authors": p.authors,
                "published": p.published[:10], "categories": p.categories, "pdf_url": p.pdf_url,
                "in_library": p.arxiv_id in in_library,
            }
            for p in papers
        ],
    }


# ------------------------------------------------------------ ingest selected
JOBS: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")  # jobs run one after another


class IngestSelectedRequest(BaseModel):
    arxiv_ids: list[str] = Field(..., min_length=1, max_length=30)


def _short(e: Exception) -> str:
    return (str(e).strip().splitlines() or [type(e).__name__])[0][:200]


def _run_job(job_id: str) -> None:
    job = JOBS[job_id]
    papers = job["papers"]
    job["status"] = "running"
    db = SessionLocal()
    try:
        have = {
            a: title
            for a, title in db.execute(select(Paper.arxiv_id, Paper.title).where(Paper.arxiv_id.in_(list(papers))))
        }
        for a, title in have.items():
            papers[a].update(status="ready", title=title, message="already in your library")

        todo = [a for a in papers if a not in have]
        if not todo:
            return
        try:
            with arxiv_lock:
                metas = {m.arxiv_id: m for m in shared_client().fetch_by_ids(todo)}  # a single API request
        except Exception as e:
            logger.warning("metadata fetch failed: %s", e)
            for a in todo:
                papers[a].update(status="failed", message="could not reach arXiv; try again")
            return

        for a in todo:
            p, meta = papers[a], metas.get(a)
            if meta is None:
                p.update(status="failed", message="not found on arXiv")
                continue
            p.update(title=meta.title, status="downloading")
            try:
                with arxiv_lock:
                    pdf = shared_client().download_pdf(meta)
                p["status"] = "processing"
                paper = ingest_pdf(db, pdf, arxiv=meta)
                p.update(status="ready", chunks=len(paper.chunks))
            except IntegrityError:  # inserted by another run in the meantime
                db.rollback()
                p.update(status="ready", message="already in your library")
            except Exception as e:
                db.rollback()
                logger.exception("failed to ingest arXiv:%s", a)
                p.update(status="failed", message=_short(e))
    finally:
        db.close()
        job["status"] = "done"


@router.post("/ingest", status_code=202)
def ingest_selected(req: IngestSelectedRequest):
    ids = list(dict.fromkeys(_VERSION.sub("", i.strip()) for i in req.arxiv_ids if i.strip()))
    if not ids:
        raise HTTPException(422, "No arXiv ids given.")
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "id": job_id, "status": "queued", "created": time.time(),
        "papers": {i: {"status": "queued", "title": None, "message": "", "chunks": None} for i in ids},
    }
    for old in sorted(JOBS, key=lambda j: JOBS[j]["created"])[:-50]:  # keep the last 50 jobs
        JOBS.pop(old, None)
    _executor.submit(_run_job, job_id)
    return {"job_id": job_id, "count": len(ids)}


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job. The server may have restarted; pick the papers from your library.")
    return job


# ------------------------------------------------------------------ library
@router.get("")
def library(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Paper, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.paper_id == Paper.id)
        .group_by(Paper.id)
        .order_by(Paper.created_at.desc())
    ).all()
    return {
        "papers": [
            {"id": p.id, "arxiv_id": p.arxiv_id, "title": p.title, "authors": p.authors,
             "published": p.published, "num_pages": p.num_pages, "chunks": n}
            for p, n in rows
        ]
    }


@router.delete("/{paper_id}")
def delete_paper(paper_id: int, db: Session = Depends(get_db)):
    paper = db.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(404, "Paper not found")
    db.delete(paper)
    db.commit()
    return {"deleted": paper_id}