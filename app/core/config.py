"""Central settings for ResearchPilot (override via environment variables)."""
import os


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


# ---------------------------------------------------------------- database
DATABASE_URL = os.getenv("DATABASE_URL") or "postgresql+psycopg://postgres:postgres@localhost:5432/researchpilot"
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://", "postgresql+psycopg://", 1
    )

# ------------------------------------------------------------------ models
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = _int("EMBED_DIM", 384)  # must match EMBED_MODEL (bge-small = 384, bge-base = 768)
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
SEMANTIC_CHUNKING = os.getenv("SEMANTIC_CHUNKING", "0") == "1"  # extra embedding pass at ingest

# --------------------------------------------------------------------- LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")  # Groq retired llama-3.3-70b-versatile on 2026-08-16
LLM_MAX_RETRIES = _int("LLM_MAX_RETRIES", 4)  # backoff on Groq 429 rate limits

# --------------------------------------------------------------- retrieval
CANDIDATES_PER_RETRIEVER = _int("CANDIDATES_PER_RETRIEVER", 50)
RERANK_POOL = _int("RERANK_POOL", 30)  # how many fused candidates the cross-encoder scores
RRF_K = _int("RRF_K", 60)

# ------------------------------------------------------------ verification
SUPPORT_THRESHOLD = _float("SUPPORT_THRESHOLD", 0.70)  # entailment prob => supported
CONTRADICTION_THRESHOLD = _float("CONTRADICTION_THRESHOLD", 0.70)
BORDERLINE_THRESHOLD = _float("BORDERLINE_THRESHOLD", 0.30)  # below this => plainly unsupported
CONFLICT_THRESHOLD = _float("CONFLICT_THRESHOLD", 0.80)  # cross-paper contradiction prob

# ------------------------------------------------------------------- arXiv
ARXIV_CACHE_DIR = os.getenv("ARXIV_CACHE_DIR", ".arxiv_cache")
ARXIV_MIN_INTERVAL = _float("ARXIV_MIN_INTERVAL", 3.0)  # arXiv guideline: 1 req / 3 s
ARXIV_CACHE_TTL_HOURS = _float("ARXIV_CACHE_TTL_HOURS", 24)
ARXIV_PAGE_SIZE = _int("ARXIV_PAGE_SIZE", 50)
ARXIV_MAX_REQUESTS = _int("ARXIV_MAX_REQUESTS", 20)