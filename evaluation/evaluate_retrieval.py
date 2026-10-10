"""Retrieval evaluation: dense vs sparse vs hybrid vs hybrid+rerank.

Eval file (JSONL), one object per line, ground truth at paper level or chunk level:
    {"query": "...", "relevant_arxiv_ids": ["2401.01234", ...]}
    {"query": "...", "relevant_chunk_ids": [12, 40]}

Usage (from the project root):
    python -m evaluation.evaluate_retrieval --synthetic 50 --eval-file eval_synthetic.jsonl   # weak sanity set
    python -m evaluation.evaluate_retrieval --eval-file my_eval.jsonl --k 10 --json-out report.json

The --synthetic set uses each paper's title as the query and the paper itself as ground truth. It is
a smoke test (lexically easy), NOT a benchmark: hand-label real questions for numbers you can trust.
For answer-level faithfulness / relevance you can additionally run Ragas on the review endpoint's output.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from sqlalchemy import select

from app.core.database import SessionLocal
from app.models.orm import Paper
from app.services import retriever

CONFIGS = {
    "dense": dict(mode="dense", use_rerank=False),
    "sparse": dict(mode="sparse", use_rerank=False),
    "hybrid (RRF)": dict(mode="hybrid", use_rerank=False),
    "hybrid + rerank": dict(mode="hybrid", use_rerank=True),
}


def metrics_at_k(ranked: list, relevant: set, k: int) -> dict[str, float]:
    top = ranked[:k]
    hits = [1 if x in relevant else 0 for x in top]
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    first = next((i for i, h in enumerate(hits) if h), None)
    return {
        f"recall@{k}": sum(hits) / len(relevant) if relevant else 0.0,
        f"hit@{k}": float(any(hits)),
        f"mrr@{k}": 1 / (first + 1) if first is not None else 0.0,
        f"ndcg@{k}": dcg / ideal if ideal else 0.0,
    }


def build_synthetic(db, n: int, path: Path, seed: int = 0) -> None:
    rows = db.execute(select(Paper.arxiv_id, Paper.title).where(Paper.arxiv_id.is_not(None))).all()
    random.Random(seed).shuffle(rows)
    with path.open("w") as f:
        for arxiv_id, title in rows[:n]:
            f.write(json.dumps({"query": title, "relevant_arxiv_ids": [arxiv_id]}) + "\n")
    print(f"wrote {min(n, len(rows))} synthetic queries to {path}")


def evaluate(db, examples: list[dict], k: int) -> dict:
    report = {}
    for name, cfg in CONFIGS.items():
        agg: dict[str, float] = {}
        latency = 0.0
        for ex in examples:
            t0 = time.perf_counter()
            hits = retriever.search(db, ex["query"], top_k=k * 3, **cfg)  # over-fetch, then dedupe to k papers
            latency += time.perf_counter() - t0
            if "relevant_arxiv_ids" in ex:
                ranked = list(dict.fromkeys(h.arxiv_id for h in hits))  # paper-level, keeps order
                relevant = set(ex["relevant_arxiv_ids"])
            else:
                ranked = [h.chunk_id for h in hits]
                relevant = set(ex["relevant_chunk_ids"])
            for key, val in metrics_at_k(ranked, relevant, k).items():
                agg[key] = agg.get(key, 0.0) + val
        n = len(examples)
        report[name] = {key: round(v / n, 4) for key, v in agg.items()} | {"avg_latency_ms": round(latency / n * 1000, 1)}
    return report


def print_table(report: dict, k: int) -> None:
    cols = [f"recall@{k}", f"hit@{k}", f"mrr@{k}", f"ndcg@{k}", "avg_latency_ms"]
    print(f"{'config':<18}" + "".join(f"{c:>14}" for c in cols))
    for name, row in report.items():
        print(f"{name:<18}" + "".join(f"{row[c]:>14}" for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-file", type=Path, default=Path("eval.jsonl"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--synthetic", type=int, default=0, help="generate N title->paper queries into --eval-file first")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    db = SessionLocal()
    try:
        if args.synthetic:
            build_synthetic(db, args.synthetic, args.eval_file)
        examples = [json.loads(line) for line in args.eval_file.read_text().splitlines() if line.strip()]
        if not examples:
            raise SystemExit("eval file is empty")
        report = evaluate(db, examples, args.k)
        print_table(report, args.k)
        if args.json_out:
            args.json_out.write_text(json.dumps(report, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()