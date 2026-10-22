<p align="center">
  <img src="assets/researchpilot-banner.svg" alt="ResearchPilot — Research Paper Intelligence Engine" width="100%" />
</p>

<h1 align="center">ResearchPilot — Research Paper Intelligence Engine</h1>

<p align="center">Discover papers · Retrieve evidence · Compare methods · Verify claims</p>

<p align="center">
  <img alt="Python 3.11" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white" />
  <img alt="PostgreSQL + pgvector" src="https://img.shields.io/badge/Database-PostgreSQL%20%2B%20pgvector-4169E1?logo=postgresql&logoColor=white" />
  <img alt="License MIT" src="https://img.shields.io/badge/License-MIT-green.svg" />
</p>

ResearchPilot is a research-paper intelligence application for building a searchable local paper library, retrieving relevant passages, drafting evidence-grounded literature reviews, comparing research methodologies, and checking whether claims are supported by paper excerpts.

The project combines document ingestion, section-aware PDF parsing, hybrid information retrieval, neural reranking, LLM-assisted synthesis, and natural-language-inference (NLI)-based claim verification in one workflow.

> **Research responsibly:** ResearchPilot is a research aid, not an authority. Generated reviews, comparisons, conflict candidates, retrieval scores, and verification verdicts can be wrong or incomplete. Always inspect the cited passages, paper metadata, and page ranges before relying on a result.

## Table of contents

- [Why ResearchPilot?](#why-researchpilot)
- [Capabilities](#capabilities)
- [System architecture](#system-architecture)
- [Technology stack](#technology-stack)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Using the application](#using-the-application)
- [API reference](#api-reference)
- [Retrieval evaluation](#retrieval-evaluation)
- [Docker deployment](#docker-deployment)
- [Project structure](#project-structure)
- [How retrieval and verification work](#how-retrieval-and-verification-work)
- [Troubleshooting](#troubleshooting)
- [Security and data handling](#security-and-data-handling)
- [Limitations](#limitations)
- [Development and contribution notes](#development-and-contribution-notes)

## Why ResearchPilot?

Research work often requires moving repeatedly between papers, searching for a passage, checking how a method differs from another, and validating whether a written claim is actually supported by its source. ResearchPilot brings these tasks into one workflow while keeping the underlying evidence visible.

The application is designed around three principles:

1. **Evidence before prose:** retrieve paper passages before asking an LLM to synthesize an answer.
2. **Traceability:** retain paper identity and page-range information with extracted chunks so users can inspect the source.
3. **Verification as a second step:** use an NLI model to assess claim–evidence relationships rather than assuming that fluent generated text is correct.

## Capabilities

| Capability | Description |
|---|---|
| arXiv discovery | Search arXiv and select papers for ingestion. |
| PDF ingestion | Ingest selected arXiv papers or upload a text-based PDF manually. |
| Structure-aware parsing | Extract PDF text while retaining page information, detected sections, and available captions. |
| Chunking | Divide extracted text into section-aware passages with page ranges. |
| Dense retrieval | Search using sentence-transformer embeddings stored in PostgreSQL with `pgvector`. |
| Lexical retrieval | Search using PostgreSQL full-text search and `ts_rank_cd`. |
| Hybrid retrieval | Combine dense and lexical rankings using reciprocal-rank fusion (RRF). |
| Reranking | Reorder candidate passages with a cross-encoder model. |
| Literature-review drafting | Use the Groq API to generate structured, citation-grounded review content from retrieved passages. |
| Methodology comparison | Produce a structured, side-by-side comparison of research methods. |
| Candidate conflict detection | Surface potential cross-paper conflicts for manual inspection. These are candidates, not confirmed scientific contradictions. |
| Claim verification | Compare a claim with evidence from a paper or selected chunks using an NLI model, with an optional LLM judge for borderline cases. |
| Retrieval evaluation | Measure retrieval quality using Recall@k, Hit@k, MRR@k, nDCG@k, and average latency. |

## System architecture

```text
                 ┌───────────────────────────┐
                 │  arXiv search / PDF upload │
                 └─────────────┬─────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │ PDF text + page extraction│
                 └─────────────┬─────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │ Section-aware chunking    │
                 │ + page-range metadata     │
                 └─────────────┬─────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │ Sentence-transformer       │
                 │ embeddings                 │
                 └─────────────┬─────────────┘
                               ▼
       ┌────────────────────────────────────────────┐
       │ PostgreSQL + pgvector                      │
       │ Paper metadata, chunks, vectors, indexes   │
       └──────────────────────┬─────────────────────┘
                              │
       User question ─────────┼──────────────────────────┐
              │               │                          │
              ▼               ▼                          │
       Dense vector search  PostgreSQL full-text search  │
              │               │                          │
              └───────────────┬──────────────────────────┘
                              ▼
                 ┌───────────────────────────┐
                 │ Reciprocal-rank fusion    │
                 └─────────────┬─────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │ Cross-encoder reranking   │
                 └─────────────┬─────────────┘
                               ▼
                 ┌───────────────────────────┐
                 │ Evidence passages with    │
                 │ paper/page references     │
                 └─────────────┬─────────────┘
                               ▼
           ┌───────────────────┴───────────────────┐
           ▼                                       ▼
┌────────────────────────┐              ┌────────────────────────┐
│ Groq-backed synthesis   │              │ NLI claim verification │
│ review / compare /      │              │ + optional LLM judge   │
│ conflict candidates     │              └────────────────────────┘
└────────────────────────┘
```

### Retrieval notes

- **Dense search** is useful for semantically related passages even when the wording differs from the query.
- **Sparse search** uses PostgreSQL full-text search and `ts_rank_cd`. It is lexical ranking with BM25-like intent, **not a textbook BM25 implementation**.
- **Hybrid search** combines ranked lists with reciprocal-rank fusion.
- **Reranking** applies a cross-encoder to reorder candidate passages based on query–passage relevance.
- Retrieved chunks preserve paper identifiers and page ranges so that generated content can be checked against the source.

## Technology stack

| Layer | Technology | Role |
|---|---|---|
| API and application server | FastAPI, Uvicorn | HTTP endpoints and web interface serving |
| Database | PostgreSQL 16 | Persistent paper and chunk metadata |
| Vector search | `pgvector` | Store and query dense embeddings |
| Lexical search | PostgreSQL full-text search | Keyword-oriented retrieval |
| Embeddings | Sentence Transformers | Encode chunks and queries as vectors |
| Reranking | Cross-encoder | Reorder retrieved passages |
| Synthesis | Groq API | Generate literature reviews, comparisons, and conflict candidates |
| Claim verification | NLI cross-encoder | Estimate the relationship between a claim and its evidence |
| PDF processing | Project parser service | Extract text, sections, captions, and page ranges |
| Runtime packaging | Docker | Package the API and evaluation code |

## Requirements

- Python 3.11
- Docker Desktop or another Docker-compatible runtime
- PostgreSQL 16 with the `pgvector` extension (the supplied Docker image provides it)
- Internet access for package installation and first-time Hugging Face model downloads
- A Groq API key for LLM-backed review, comparison, and conflict-detection features
- Enough disk space for the database, cached PDFs, and downloaded model files
- Optional CUDA-capable GPU; CPU inference is supported but may be slower

The application initializes the vector extension, tables, and indexes during startup. The first retrieval may download the embedding and reranking models; the first claim-verification run may download the NLI model.

## Quick start

The commands below use **Windows PowerShell** by default. macOS/Linux alternatives are included where needed.

### 1. Clone the repository

```powershell
git clone https://github.com/chandshaurya/ResearchPilot-Research-Paper-Intelligence-Engine.git
Set-Location ResearchPilot-Research-Paper-Intelligence-Engine
```

If you already have a local checkout, open a terminal in the repository root instead.

### 2. Start PostgreSQL with pgvector

Create a persistent Docker volume and start PostgreSQL:

```powershell
docker run -d --name researchpilot-db `
  -p 5432:5432 `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=postgres `
  -e POSTGRES_DB=researchpilot `
  -v researchpilot-postgres-data:/var/lib/postgresql/data `
  pgvector/pgvector:pg16
```

Check that the database is ready:

```powershell
docker exec researchpilot-db pg_isready -U postgres -d researchpilot
```

If the container already exists, start it rather than creating another one:

```powershell
docker start researchpilot-db
```

To stop the database without deleting its stored data:

```powershell
docker stop researchpilot-db
```

> The `postgres` username/password above are convenient local-development defaults only. Do not reuse them for a public or production deployment. If port `5432` is already occupied, change the host-side port mapping and update `DATABASE_URL` accordingly.

### 3. Create a virtual environment and install dependencies

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current PowerShell process and then activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create your local environment file from the example:

```powershell
Copy-Item .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

Edit `.env` and set `GROQ_API_KEY` if you want to use LLM-backed synthesis. The database URL should match the local PostgreSQL container created in step 2.

### 5. Start the API

With the virtual environment active and the database running:

```powershell
python -m uvicorn app.main:app --reload
```

Open these local URLs in your browser:

| Resource | URL |
|---|---|
| Web interface | http://localhost:8000/ |
| Interactive API documentation | http://localhost:8000/docs |
| Health check | http://localhost:8000/health |

The browser interface is served by FastAPI. A static-only deployment of `app/static/index.html` (for example, through GitHub Pages) does not run the Python API, PostgreSQL database, or ML models; therefore, it cannot provide the full application by itself.

## Configuration

The application loads settings from `.env`. Keep personal environment files out of version control and commit only a safe `.env.example`.

| Variable | Required? | Purpose / default |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL connection string. Local example: `postgresql+psycopg://postgres:postgres@localhost:5432/researchpilot`. |
| `GROQ_API_KEY` | For LLM synthesis | Groq API key. Reviews, comparisons, and conflict detection require a key. Direct NLI verification can run without it. |
| `LLM_MODEL` | No | Groq chat model; default `openai/gpt-oss-120b`. Availability depends on the provider/account. |
| `EMBED_MODEL` | No | Embedding model; default `BAAI/bge-small-en-v1.5`. Its vector dimension must match the configured schema. |
| `EMBED_DIM` | No | Embedding vector dimension; default `384` for the default embedding model. |
| `RERANK_MODEL` | No | Cross-encoder reranker; default `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| `NLI_MODEL` | No | Claim-verification model; default `cross-encoder/nli-deberta-v3-base`. |
| `SEMANTIC_CHUNKING` | No | Set to `1` to enable an additional embedding pass for semantic chunk boundaries; default `0`. |
| `ARXIV_MIN_INTERVAL` | No | Minimum delay between arXiv requests in seconds; default `3`. |
| `ARXIV_CACHE_DIR` | No | Local cache for arXiv metadata/PDFs and uploaded PDFs; default `.arxiv_cache`. |

**Important configuration notes**

- If you change the embedding model, ensure `EMBED_DIM` and the database vector column are compatible. Existing vectors created with a different model should not be treated as interchangeable.
- Keep API keys, passwords, and production connection strings out of source code, screenshots, issues, and committed files.
- If a real credential has been exposed, revoke it and create a replacement. Removing it from the latest commit does not remove it from Git history.

## Using the application

A typical workflow is:

1. **Discover papers:** search arXiv from the web UI or use the API search endpoint.
2. **Ingest papers:** select arXiv records or upload a text-based PDF. Ingestion extracts text and creates searchable chunks.
3. **Check ingestion status:** use the job-status endpoint or inspect corpus statistics.
4. **Search the corpus:** ask a natural-language question and choose dense, sparse, or hybrid retrieval. Reranking is enabled by default.
5. **Draft a literature review:** provide a research question and allow the system to retrieve evidence before generating structured Markdown and claims.
6. **Compare methods:** request a side-by-side methodology comparison across relevant papers.
7. **Check a claim:** verify a claim against one paper or a set of retrieved chunk IDs, then inspect the returned evidence and page ranges.
8. **Evaluate retrieval:** run the evaluation script against a populated corpus and, ideally, a hand-labelled evaluation set.

### Example: ingest a small arXiv sample

Run this in a second PowerShell terminal after the API has started:

```powershell
$ingestBody = @{
  topics = @(
    @{
      name = "rag"
      query = 'cat:cs.CL AND all:"retrieval augmented generation"'
    }
  )
  max_requests = 2
  max_papers = 2
} | ConvertTo-Json -Depth 5

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/ingest/arxiv" `
  -ContentType "application/json" `
  -Body $ingestBody
```

The endpoint returns `202 Accepted` because ingestion runs in the background. Check progress in the API terminal and inspect the corpus totals:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/ingest/stats"
```

The first ingestion may take longer because the embedding model needs to be downloaded and initialized. To ingest a local PDF, use `POST /ingest/pdf` through the interactive API docs at `/docs`.

The web UI also supports searching arXiv through `GET /papers/arxiv-search`, selecting up to 30 arXiv IDs, and ingesting them through `POST /papers/ingest`. Follow the returned job status to track progress.

### Example: search ingested papers

```powershell
$searchBody = @{
  query = "How is faithfulness of RAG answers measured?"
  top_k = 5
  mode = "hybrid"
  rerank = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/search" `
  -ContentType "application/json" `
  -Body $searchBody
```

`mode` accepts `dense`, `sparse`, or `hybrid`. Results include the passage, paper reference, page range, and retrieval scores. The first search may take longer while the embedding and cross-encoder models are downloaded.

### Example: generate a citation-grounded literature review

Set `GROQ_API_KEY` in `.env` and restart the API before using this endpoint:

```powershell
$reviewBody = @{
  question = "How do recent papers reduce hallucination in RAG systems?"
  top_k = 12
  verify = $true
  multi_query = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/synthesize/review" `
  -ContentType "application/json" `
  -Body $reviewBody
```

The response includes rendered Markdown, structured sections and claims, citations, and run statistics. Set `multi_query` to `$true` to expand the question into additional searches; this requires an extra LLM call. You can optionally pass an `arxiv_ids` array to restrict retrieval to papers already in the library.

### Example: verify a claim

Replace the example arXiv ID with one returned by your own search or library:

```powershell
$verifyBody = @{
  claim = "The method improves answer faithfulness over the baseline."
  arxiv_id = "2401.01234"
  top_k = 6
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri "http://localhost:8000/synthesize/verify" `
  -ContentType "application/json" `
  -Body $verifyBody
```

The endpoint can also accept `chunk_ids` instead of `arxiv_id`. Possible verdicts are `supported`, `partially_supported`, `unsupported`, and `contradicted`; generated-review claims without citations may be labelled `uncited`. Treat these as model assessments and read the evidence before drawing a conclusion.

## API reference

The interactive documentation at [http://localhost:8000/docs](http://localhost:8000/docs) lists the request and response schemas and allows endpoints to be tested directly.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check. |
| `GET` | `/papers` | List papers in the local library. |
| `GET` | `/papers/arxiv-search?q=...` | Search arXiv and mark papers already in the library. |
| `POST` | `/papers/ingest` | Ingest selected arXiv IDs as a background job. |
| `GET` | `/papers/jobs/{job_id}` | Check the status of a selected-paper ingestion job. |
| `DELETE` | `/papers/{paper_id}` | Delete a paper and its chunks. |
| `POST` | `/ingest/arxiv` | Start a bounded, round-robin topic ingestion job. |
| `POST` | `/ingest/pdf` | Upload and ingest a text-based PDF. |
| `GET` | `/ingest/stats` | Return paper and chunk counts. |
| `POST` | `/search` | Search the corpus and retrieve passages. |
| `GET` | `/search/chunk/{chunk_id}` | Resolve a result/citation to its passage. |
| `POST` | `/synthesize/review` | Generate a citation-grounded literature review. |
| `POST` | `/synthesize/verify` | Verify a claim against a paper or chunk IDs. |
| `POST` | `/synthesize/compare` | Build a side-by-side methodology comparison. |
| `POST` | `/synthesize/conflicts` | Identify candidate conflicts between paper findings. |

LLM-backed synthesis endpoints require a valid Groq key. Direct NLI claim verification does not require a Groq key, but the NLI model still needs to be downloaded and run locally. Without a key, the optional LLM judge for borderline cases is unavailable.

## Retrieval evaluation

Evaluation compares dense search, sparse search, hybrid reciprocal-rank fusion, and hybrid search with reranking. Run it from the repository root after ingesting papers into the database.

### Quick smoke test

```powershell
python -m evaluation.evaluate_retrieval --synthetic 50 --eval-file eval_synthetic.jsonl
```

The synthetic title-to-paper examples are intended as a smoke test only. They are lexically easy and should **not** be presented as a rigorous retrieval benchmark.

### Build a labelled evaluation set

Create a UTF-8 JSONL file with one example per line. Use paper or chunk IDs that actually exist in your local database:

```jsonl
{"query":"How do papers measure faithfulness in RAG?","relevant_arxiv_ids":["2401.01234"]}
{"query":"What dataset is used to evaluate retrieval?","relevant_chunk_ids":[12,40]}
```

Then run the evaluator:

```powershell
python -m evaluation.evaluate_retrieval --eval-file my_eval.jsonl --k 10 --json-out report.json
```

The report includes:

- **Recall@k:** fraction of relevant items retrieved within the first *k* results.
- **Hit@k:** whether at least one relevant item appears in the first *k* results.
- **MRR@k:** reciprocal rank of the first relevant result, averaged across queries.
- **nDCG@k:** ranking quality with relevance discounted by position.
- **Average latency:** mean query time for each retrieval configuration.

For credible comparisons, use representative questions, label relevance carefully, document the corpus and configuration, and compare all retrieval modes on the same queries. Reranking may download its model on the first run.

## Docker deployment

The included `Dockerfile` packages the API and evaluation module using Python 3.11 and CPU-only PyTorch.

Build the image from the repository root:

```powershell
docker build -t researchpilot .
```

The API container needs a `DATABASE_URL` reachable **from inside the container**, and a Groq key for LLM-backed features. `localhost` inside a container refers to that container, not automatically to the host machine.

For Docker Desktop, a host database is commonly reachable through `host.docker.internal`. For container-to-container communication, place the API and database on a shared Docker network and use the database service/container name.

```powershell
docker run --rm -p 8000:8000 `
  --env-file .env `
  -e DATABASE_URL="postgresql+psycopg://postgres:postgres@host.docker.internal:5432/researchpilot" `
  researchpilot
```

This example assumes the local PostgreSQL container is reachable through Docker Desktop's host gateway. Database migrations are not included; this project initializes its schema through idempotent table creation at startup. For production, configure secrets, network access, persistent storage, logging, and database access controls appropriately rather than relying on the local-development defaults.

## Project structure

```text
.
├── app/
│   ├── main.py                  # FastAPI app, lifespan initialization, UI route
│   ├── static/
│   │   └── index.html           # Browser UI served by FastAPI
│   ├── api/                     # Paper library, ingestion, search, synthesis routes
│   ├── core/
│   │   ├── config.py            # Environment-based settings
│   │   └── database.py          # Database engine, sessions, schema initialization
│   ├── models/
│   │   ├── orm.py               # Paper/chunk tables, vector column, indexes
│   │   └── schemas.py           # API request/response schemas
│   └── services/
│       ├── parser.py            # PDF text, page tracking, sections, captions
│       ├── chunker.py           # Section-aware chunks and page ranges
│       ├── arxiv_client.py      # arXiv client, cache, rate limits, topic rotation
│       ├── retriever.py         # Dense/sparse search, fusion, reranking
│       ├── synthesizer.py       # Reviews, comparisons, conflict candidates
│       └── verifier.py          # NLI checks, numeric guard, optional LLM judge
├── evaluation/
│   └── evaluate_retrieval.py    # Retrieval metrics and CLI
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

The directory listing describes the main paths; additional files may exist in the repository.

## How retrieval and verification work

### 1. Ingestion and chunking

The parser extracts text from a PDF and attempts to retain page and section context. The chunker creates passages that are more meaningful than arbitrary character slices and attaches page-range information. These chunks become the units returned by retrieval and referenced by downstream synthesis.

Text extraction is not equivalent to a full semantic understanding of a PDF. Complex layouts, scanned pages, and tables can reduce extraction quality.

### 2. Embedding and indexing

The configured sentence-transformer converts chunks into dense vectors. PostgreSQL stores these vectors using `pgvector`, along with paper and chunk metadata. The embedding dimension must match the database vector schema.

### 3. Hybrid retrieval and reranking

The search service can use dense vector retrieval, PostgreSQL lexical retrieval, or both. Hybrid mode combines ranked results with reciprocal-rank fusion, after which a cross-encoder can rerank candidates for query–passage relevance. Retrieval scores are ranking signals, not probabilities that a passage is scientifically correct.

### 4. Synthesis

For review, comparison, and conflict detection, retrieved passages are supplied to the configured Groq model. The response is structured for inspection and includes citations or source references tied to retrieved chunks. The quality of synthesis depends on the retrieved evidence, the prompt, the selected model, and provider availability.

### 5. Claim verification

The NLI model estimates whether a claim is supported by, contradicted by, or otherwise related to a passage. An optional LLM judge may be used for borderline cases when a Groq key is configured. A verdict is not a proof: NLI systems can be uncertain about multi-part claims, numerical statements, domain-specific terminology, or evidence spread across multiple passages.

## arXiv ingestion and local data

Ingestion is deliberately conservative. The arXiv client rotates among topics, enforces a request interval, caches responses, stores cursor/seen-ID state, skips papers already in the database, and respects a per-run request budget.

- Cached arXiv metadata, PDFs, and uploaded PDFs are stored under `.arxiv_cache/` by default.
- `.arxiv_cache/` is local runtime data and should remain out of Git.
- The PostgreSQL volume stores indexed paper and chunk data.
- Text-based PDFs work best. Image-only/scanned PDFs need OCR before ingestion.

## Troubleshooting

| Problem | Things to check |
|---|---|
| Database connection refused | Confirm Docker is running; run `docker exec researchpilot-db pg_isready -U postgres -d researchpilot`; verify host, port, database, and credentials in `DATABASE_URL`. |
| `vector` extension or schema initialization fails | Use `pgvector/pgvector:pg16` and ensure the database user can create extensions. The API initializes the extension and schema at startup. |
| Port `5432` or `8000` is already in use | Stop the conflicting process or change the host-side port mapping. If PostgreSQL's host port changes, update `DATABASE_URL`. |
| Review, compare, or conflicts reports a missing key | Set `GROQ_API_KEY` in `.env` and restart Uvicorn. Avoid putting secrets in commands that may be saved to shell history. |
| Model download is slow or fails | Check internet connectivity, available disk space, proxy/firewall settings, and access to Hugging Face. Retry after resolving connectivity. |
| Groq model or rate-limit error | Check that the configured model is available to your account. Reduce `top_k` to limit prompt size and retry after a rate limit clears. |
| PDF ingestion extracts no text | The PDF may be scanned or image-only. Run OCR first and upload a searchable PDF. |
| A citation shows two pages | A chunk can cross a page break; the citation reflects the chunk's page range rather than always identifying one exact page. |
| Dockerized API cannot connect to database | Do not use `localhost` for a database in another container. Use `host.docker.internal` for a host service on Docker Desktop or a service/container name on a shared Docker network. |
| Embedding dimension mismatch | Ensure `EMBED_MODEL`, `EMBED_DIM`, and the database vector column are consistent. Do not mix vectors from incompatible embedding models. |

## Security and data handling

Before publishing or deploying:

1. Keep `.env`, `.arxiv_cache/`, uploaded PDFs, and local runtime data out of version control unless there is a deliberate reason to include a specific file.
2. Commit `.env.example` only with safe local defaults and blank secret values.
3. Review staged files and Git history for API keys, passwords, database URLs, private papers, or other confidential data.
4. Rotate any credential that has been exposed; deleting it from the latest version does not erase prior Git history.
5. Do not expose a development database with default credentials to a public network.
6. Treat uploaded PDFs and generated reports according to the applicable copyright, confidentiality, and institutional policies.
7. Remember that LLM-backed synthesis sends prompts and retrieved context to the configured provider. Check provider terms and data-handling requirements before using sensitive or unpublished material.

## Limitations

- PostgreSQL `ts_rank_cd` is not textbook BM25.
- Citation references identify chunk page ranges; a chunk spanning a page break may refer to multiple pages.
- NLI may flag studies that use different datasets, baselines, or experimental setups as conflicting. Inspect the studies before concluding that findings contradict.
- Numeric checks can detect some numbers absent from cited text, but do not validate units or guarantee that a number has been interpreted correctly.
- Scanned PDFs require OCR; tables are extracted as text rather than reconstructed as structured cells.
- Retrieval depends on the papers already ingested and on the quality of the extracted text. A missing result does not prove that no relevant paper exists.
- LLM outputs may omit context, overgeneralize, or phrase claims more strongly than the source supports. Verify important claims manually.
- Groq model availability and rate limits depend on the account and selected model. Reduce `top_k` if prompts exceed model limits.
- The project initializes tables at startup and does not include a formal database-migration workflow.
- The included synthetic retrieval examples are smoke tests, not a substitute for a representative labelled benchmark.

## Development and contribution notes

Contributions that improve reproducibility, evidence traceability, ingestion robustness, evaluation quality, or error reporting are especially useful.

Before opening a pull request:

- Keep changes focused and explain the problem they solve.
- Add or update tests when changing parsing, retrieval, synthesis, or verification behaviour.
- For retrieval changes, evaluate all relevant configurations against the same labelled queries and report the corpus, model configuration, and metric definitions.
- Avoid unsupported performance claims. If you report metrics, include the evaluation set and procedure so others can reproduce them.
- Never commit secrets, private documents, or local cache/database files.
- Update this README when commands, endpoints, environment variables, or architectural behaviour change.

## License and attribution

ResearchPilot's source code is licensed under the **MIT License**. See the [`LICENSE`](LICENSE) file for the complete license text and required copyright notice. You may use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the conditions stated in that file.

### Third-party resources

The MIT license for this repository does **not** automatically apply to third-party resources used with ResearchPilot. In particular:

- **Research papers and PDFs:** arXiv papers and user-uploaded documents remain subject to their own copyright, licenses, and access terms. Only ingest or share documents you are authorized to use.
- **Machine-learning models:** embedding, reranking, and NLI models are distributed under their respective model licenses and terms. Review those terms before commercial use or redistribution.
- **External services and APIs:** arXiv and Groq have their own usage policies, rate limits, and terms. Keep API credentials private and comply with the relevant policies.
- **Dependencies:** third-party Python packages retain their own licenses; consult their respective project and distribution metadata.

Please preserve the copyright and license notices required by the MIT License when redistributing the software. If you add code, assets, or datasets from another source, include the attribution or license notices required by that source.

---

**ResearchPilot** — retrieve evidence, compare methods, and draft research summaries with source passages kept in view.
