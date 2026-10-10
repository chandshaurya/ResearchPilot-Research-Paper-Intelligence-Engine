# ResearchPilot — Research Paper Intelligence Engine

Reads research papers, retrieves evidence, compares methodologies, flags conflicting results and writes
literature reviews with page-level citations. **Standout feature:** every generated claim is checked against the
paper it cites, and unsupported or contradicted claims are flagged.

```
arXiv API ─(round-robin, cached)─► PDFs ─► parser ─► chunker ─► embeddings ─► PostgreSQL + pgvector
                                                                                   │
question ─► dense (pgvector) ┐                                                     │
            sparse (FTS)     ├─► RRF fusion ─► cross-encoder rerank ─► top-k chunks ◄┘
                             ┘                         │
                                      LLM writes claims tagged [C1]..[Cn]
                                                       │
                          verifier: NLI (PyTorch) per claim vs ONLY its cited chunks
                                                       ▼
                       review with ⚠️ flags + page-level references  (+ methods table, conflicts)
```

## Deploy online without terminal commands

1. Install [GitHub Desktop](https://desktop.github.com/) and sign in. Add this project folder
   as a local repository, create the repository if prompted, then choose **Publish repository**
   and keep it private. The `.gitignore` file excludes your virtual environment, arXiv cache,
   and local `.env` secrets.
2. Open the [Render Dashboard](https://dashboard.render.com/), choose **New +** >
   **Blueprint**, and connect the GitHub repository you just published. Render will read
   `render.yaml` and build the app from the included Dockerfile.
3. When prompted, set `DATABASE_URL` to the **newly rotated** Neon connection string
   (choose the pooled connection string and keep its SSL query parameters). Do not put the
   URL in source files. The app initializes its pgvector extension and tables at startup.
4. Wait for the Render deployment to finish. Its dashboard will show your public app URL;
   open that URL for the web UI, or append `/docs` for the API documentation.
5. To enable review/compare/conflict synthesis, add `GROQ_API_KEY` under the service's
   **Environment** settings in Render and redeploy. This is optional for search and verify.

The public URL is assigned by Render after deployment; it cannot be known in advance.
Rotate any database password or API key that has been pasted into chat or committed to a
file before deploying.

## Quick start (local terminal)

```bash
# 1. Postgres with pgvector
docker run -d --name rp-db -p 5432:5432 -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=researchpilot pgvector/pgvector:pg16


$env:GROQ_API_KEY = "..."  # Set this locally; do not store API keys in source.


# 2. Python env
python -m venv .venv
# PowerShell:
. .\.venv\Scripts\Activate.ps1
# bash/zsh:
# source .venv/bin/activate

pip install -r requirements.txt

# 3. Config (all optional except the key for synthesis endpoints)
# PowerShell:
$env:GROQ_API_KEY = "..."                 # free key from console.groq.com; review / compare / conflicts need it, /verify and /search don't
$env:LLM_MODEL = "llama-3.3-70b-versatile"  # optional: any Groq chat model with JSON mode
$env:DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/researchpilot"
# bash/zsh:
# export GROQ_API_KEY=...
# export LLM_MODEL=llama-3.3-70b-versatile
# export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/researchpilot

# 4. Run (use python -m so it works even when uvicorn isn't on PATH)

python -m uvicorn app.main:app --reload    # docs at http://localhost:8000/docs
```

For Docker with Neon on PowerShell, set the URL in the shell and pass it at runtime
(replace the placeholders; do not put credentials in the Dockerfile):

```powershell
$env:DATABASE_URL = "postgresql://<user>:<password>@<host>/<database>?sslmode=require&channel_binding=require"
docker build -t researchpilot .
docker run -p 8000:8000 --env DATABASE_URL researchpilot
```

## Typical session

```bash
# Ingest from arXiv: one page per topic per turn (round-robin), at most 20 API requests
curl -X POST localhost:8000/ingest/arxiv -H 'content-type: application/json' -d '{
  "topics": [
    {"name": "rag",   "query": "cat:cs.CL AND all:\"retrieval augmented generation\""},
    {"name": "halluc","query": "cat:cs.CL AND all:hallucination AND all:evaluation"}
  ], "max_requests": 10, "max_papers": 40 }'

curl localhost:8000/ingest/stats

# Hybrid search
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "how is faithfulness of RAG answers measured?", "top_k": 5}'

# Literature review with verification
curl -X POST localhost:8000/synthesize/review -H 'content-type: application/json' \
  -d '{"question": "How do recent papers reduce hallucination in RAG systems?"}'

# THE STANDOUT FEATURE: does this paper really say that?
curl -X POST localhost:8000/synthesize/verify -H 'content-type: application/json' \
  -d '{"claim": "Method X improves F1 by 12.5% over the baseline.", "arxiv_id": "2401.01234"}'
```

Verdicts: `supported` · `partially_supported` (e.g. numbers absent from the cited text) · `unsupported` ·
`contradicted` · `uncited`. Each result includes the best-matching passage and its page range.

## How the arXiv ingestion keeps API usage low

Round-robin over topics (one page each per turn), 3 s rate limit, 24 h response cache, persistent cursors and
seen-IDs (reruns fetch only new papers), early stop when a page is fully seen, papers already in the DB skip the PDF
download, and a hard `max_requests` budget per run. State lives in `.arxiv_cache/`.

## Layout

| Path | Role |
|---|---|
| `app/main.py` | FastAPI app; creates the pgvector extension and tables on startup |
| `app/api/ingest.py` | arXiv ingestion (background) and PDF upload |
| `app/api/search.py` | hybrid search, citation lookup |
| `app/api/synthesize.py` | review, verify, compare, conflicts |
| `app/core/` | `config.py` (env settings), `database.py` (engine/session) |
| `app/models/` | `orm.py` (papers, chunks + HNSW/GIN indexes), `schemas.py` (Pydantic) |
| `app/services/arxiv_client.py` | round-robin arXiv fetcher with cache and rate limit |
| `app/services/parser.py` | PyMuPDF extraction, section tracking, captions |
| `app/services/chunker.py` | section-aware paragraph packing with page ranges |
| `app/services/retriever.py` | dense + sparse + RRF + cross-encoder rerank |
| `app/services/synthesizer.py` | LLM review, methodology table, conflict detection |
| `app/services/verifier.py` | claim verification (NLI + numeric guard + optional LLM judge) |
| `evaluation/evaluate_retrieval.py` | Recall / Hit / MRR / nDCG across retrieval configs |

## Evaluation

```bash
python -m evaluation.evaluate_retrieval --synthetic 50 --eval-file eval_synthetic.jsonl   # smoke test only
python -m evaluation.evaluate_retrieval --eval-file my_eval.jsonl --k 10 --json-out report.json
```

Hand-label real questions (`{"query": ..., "relevant_arxiv_ids": [...]}`) for numbers you can trust.

## Known limitations

- The sparse retriever uses Postgres `ts_rank_cd`, which is BM25-style but not textbook BM25 (see `retriever.py`).
- Citations carry the chunk's page *range*; a chunk spanning a page break can't say which of the two pages.
- NLI can't compare setups: a flagged conflict may just be two different datasets. The numeric guard catches
  missing figures, not wrong units. Treat verdicts as a strong signal for review, not a final judgement.
- Groq's free tier is rate-limited: calls retry with backoff on 429s; if you hit a token-limit error, lower `top_k`.
  Model names on Groq change over time, so check the console for current ones if `LLM_MODEL` is rejected.
- Scanned PDFs yield no text and are rejected; run OCR first.
- Tables are extracted as plain text blocks, not structured cells.
