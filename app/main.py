"""FastAPI entrypoint.  Run:  uvicorn app.main:app --reload"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import ingest, papers, search, synthesize
from app.core.database import init_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="ResearchPilot",
    description="Research-paper intelligence engine: hybrid retrieval, citation-grounded synthesis, claim verification.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(papers.router)
app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(synthesize.router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def web_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}