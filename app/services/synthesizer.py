"""Multi-document synthesis with citation grounding.

Flow for a literature review:
  retrieve (hybrid + rerank) -> LLM writes claims, each tagged with excerpt ids [C1]..[Cn]
  -> every claim is verified against ONLY the chunks it cites (verifier.py) -> flagged in the output.
Also: methodology comparison table, and cross-paper conflict detection (LLM finding extraction + NLI).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core import config
from app.services import retriever
from app.services.retriever import Hit
from app.services.verifier import ClaimVerifier, EvidenceChunk, Verification


# --------------------------------------------------------------------- LLM
class LLMClient:
    """Thin wrapper around the Groq chat-completions API (OpenAI-compatible).

    Groq's free tier is rate-limited (requests and tokens per minute), so calls retry with
    backoff on 429s. If you hit a 413/token-limit error, lower `top_k` in the request.
    """

    def __init__(self, model: str = config.LLM_MODEL, api_key: str = config.GROQ_API_KEY):
        from groq import Groq

        self.model = model
        self.client = Groq(api_key=api_key)
        self.calls = 0

    # Used only if the configured model is unavailable (Groq retires models regularly).
    PREFERRED = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b")
    _NOT_CHAT = ("whisper", "orpheus", "guard", "safeguard", "tts", "compound", "embed", "playai")

    def _pick_available_model(self) -> str:
        ids = [m.id for m in self.client.models.list().data]
        for want in self.PREFERRED:
            if want in ids:
                return want
        chat = [i for i in ids if not any(x in i.lower() for x in self._NOT_CHAT)]
        if chat:
            return chat[0]
        raise RuntimeError("This Groq API key can see no chat models. Check the key at console.groq.com.")

    def complete(self, system: str, user: str, max_tokens: int = 2500, json_mode: bool = False) -> str:
        import logging
        import time

        import groq

        def build() -> dict[str, Any]:
            kw: dict[str, Any] = dict(
                model=self.model, max_tokens=max_tokens, temperature=0.2,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            if self.model.startswith("openai/gpt-oss"):
                kw["max_tokens"] = max_tokens + 1500  # reasoning tokens count against the budget
                kw["extra_body"] = {"reasoning_effort": "low"}  # short thinking: faster, fits free-tier limits
            if json_mode:  # Groq JSON mode: requires the prompt to mention "JSON" (ours always does)
                kw["response_format"] = {"type": "json_object"}
            return kw

        kwargs, switched = build(), False
        for attempt in range(config.LLM_MAX_RETRIES + 2):
            self.calls += 1
            try:
                return self.client.chat.completions.create(**kwargs).choices[0].message.content or ""
            except groq.NotFoundError:
                if switched:
                    raise
                old, self.model, switched = self.model, self._pick_available_model(), True
                logging.getLogger(__name__).warning("Groq model %s is unavailable; using %s instead", old, self.model)
                kwargs = build()
            except groq.BadRequestError:
                optional = [k for k in ("response_format", "extra_body") if k in kwargs]
                if not optional:
                    raise
                for k in optional:  # the model rejected an optional setting: retry plainly
                    kwargs.pop(k)
            except (groq.RateLimitError, groq.APIConnectionError, groq.InternalServerError) as e:
                if attempt >= config.LLM_MAX_RETRIES:
                    raise
                headers = getattr(getattr(e, "response", None), "headers", None) or {}
                try:
                    delay = float(headers.get("retry-after", 0))
                except ValueError:
                    delay = 0.0
                time.sleep(delay or 2 ** (attempt + 1))
        return ""

    def complete_json(self, system: str, user: str, max_tokens: int = 2500) -> Any:
        raw = self.complete(system, user, max_tokens, json_mode=True)
        raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
        start = min((i for i in (raw.find("{"), raw.find("[")) if i != -1), default=-1)
        if start == -1:
            raise ValueError(f"LLM did not return JSON: {raw[:200]}")
        try:
            return json.loads(raw[start:])
        except json.JSONDecodeError:
            # Models sometimes add a sentence after the JSON: parse just the first JSON value.
            return json.JSONDecoder().raw_decode(raw[start:])[0]


# ----------------------------------------------------------------- helpers
def hit_to_evidence(h: Hit) -> EvidenceChunk:
    return EvidenceChunk(h.chunk_id, h.content, h.page_start, h.page_end, h.title, h.arxiv_id)


def _pages(h: Hit) -> str:
    return f"p.{h.page_start}" if h.page_start == h.page_end else f"pp.{h.page_start}-{h.page_end}"


def build_context(hits: list[Hit]) -> tuple[str, dict[str, Hit]]:
    """Number the excerpts C1..Cn so the LLM can cite them and we can map tags back to pages."""
    tag_map = {f"C{i}": h for i, h in enumerate(hits, start=1)}
    parts = [
        f"[{tag}] {h.title} (arXiv:{h.arxiv_id or 'n/a'}), {_pages(h)}, section: {h.section}\n{h.content}"
        for tag, h in tag_map.items()
    ]
    return "\n\n".join(parts), tag_map


def citation_dict(tag: str, h: Hit) -> dict:
    return dict(tag=tag, chunk_id=h.chunk_id, arxiv_id=h.arxiv_id, title=h.title,
                section=h.section, page_start=h.page_start, page_end=h.page_end)


def expand_queries(llm: LLMClient, question: str, n: int = 3) -> list[str]:
    data = llm.complete_json(
        "Rewrite a research question into distinct search queries covering different aspects "
        '(methods, datasets, results, limitations). Return JSON: {"queries": ["..."]}',
        f"Question: {question}\nReturn {n} queries.", max_tokens=300,
    )
    return [q for q in data.get("queries", []) if isinstance(q, str)][:n]


def retrieve_for(db: Session, question: str, top_k: int, paper_ids: Optional[list[int]],
                 llm: Optional[LLMClient] = None, multi_query: bool = False) -> list[Hit]:
    if not (multi_query and llm):
        return retriever.search(db, question, top_k=top_k, paper_ids=paper_ids)
    pool: dict[int, Hit] = {}
    for q in [question, *expand_queries(llm, question)]:
        for h in retriever.search(db, q, top_k=top_k, use_rerank=False, paper_ids=paper_ids):
            pool.setdefault(h.chunk_id, h)
    return retriever.rerank(question, list(pool.values()), top_k)  # final order judged against the real question


# ------------------------------------------------------------ literature review
REVIEW_SYSTEM = """You are a careful scientific writer. Write a literature review answering the question \
using ONLY the numbered excerpts provided.
Rules:
- Split the review into 2-5 sections with short headings.
- Each claim is ONE self-contained sentence that states what a specific paper found, proposed or measured.
- Each claim lists the excerpt ids that directly support it, e.g. ["C1","C4"]. Never cite an excerpt that does not say it.
- No outside knowledge. Include numbers only if they appear in the cited excerpts.
- Where excerpts disagree, write each side as its own claim and add a claim that names the disagreement, citing both.
- If the excerpts cannot answer part of the question, say so in a claim with an empty cite list.
Return ONLY JSON: {"sections":[{"heading":"...","claims":[{"text":"...","cite":["C1"]}]}]}"""

_FLAG = {"unsupported": " ⚠️ *not supported by cited paper*", "contradicted": " ❌ *contradicted by cited paper*",
         "partially_supported": " ⚠️ *partially supported*", "uncited": " ⚠️ *no citation*"}


def generate_review(db: Session, question: str, llm: LLMClient, verifier: Optional[ClaimVerifier],
                    top_k: int = 12, verify: bool = True, multi_query: bool = False,
                    paper_ids: Optional[list[int]] = None) -> dict:
    hits = retrieve_for(db, question, top_k, paper_ids, llm, multi_query)
    if not hits:
        return dict(question=question, markdown="No relevant passages found.", sections=[], citations=[], stats={})
    context, tag_map = build_context(hits)
    data = llm.complete_json(REVIEW_SYSTEM, f"QUESTION: {question}\n\nEXCERPTS:\n{context}", max_tokens=3500)

    sections, used, counts = [], set(), {}
    for sec in data.get("sections", []):
        claims = []
        for c in sec.get("claims", []):
            text = str(c.get("text", "")).strip()
            tags = [t.strip().upper() for t in c.get("cite", []) if t.strip().upper() in tag_map]
            if not text:
                continue
            used.update(tags)
            if not tags:
                ver = Verification("uncited", note="claim has no valid citation")
            elif verify and verifier:
                ver = verifier.verify(text, [hit_to_evidence(tag_map[t]) for t in tags])
            else:
                ver = Verification("supported", method="skipped", note="verification disabled")
            counts[ver.verdict] = counts.get(ver.verdict, 0) + 1
            claims.append({"text": text, "cite": tags, **ver.to_dict()})
        sections.append({"heading": str(sec.get("heading", "")).strip() or "Findings", "claims": claims})

    md = [f"# Literature review: {question}\n"]
    for s in sections:
        md.append(f"## {s['heading']}\n")
        md.append(" ".join(c["text"] + (f" [{', '.join(c['cite'])}]" if c["cite"] else "") + _FLAG.get(c["verdict"], "")
                           for c in s["claims"]) + "\n")
    cites = [citation_dict(t, tag_map[t]) for t in sorted(used, key=lambda x: int(x[1:]))]
    md.append("## References\n")
    md += [f"- **[{c['tag']}]** {c['title']} (arXiv:{c['arxiv_id']}), {_pages(tag_map[c['tag']])}" for c in cites]

    n = sum(counts.values()) or 1
    stats = {"claims": sum(counts.values()), "verdicts": counts, "llm_calls": llm.calls,
             "supported_rate": round(counts.get("supported", 0) / n, 3)}
    return dict(question=question, markdown="\n".join(md), sections=sections, citations=cites, stats=stats)


# ----------------------------------------------------------- methodology table
COMPARE_SYSTEM = """You compare research methodologies using ONLY the numbered excerpts.
Produce ONE row per distinct paper. Use "" for any field the excerpts do not state - never guess.
Return ONLY JSON: {"rows":[{"paper":"<title>","task":"","method":"","data":"","metrics":"","results":"","limitations":"","cite":["C1"]}]}"""


def compare_methodologies(db: Session, question: str, llm: LLMClient, top_k: int = 12,
                          paper_ids: Optional[list[int]] = None) -> dict:
    hits = retriever.search(db, question, top_k=top_k, paper_ids=paper_ids)
    context, tag_map = build_context(hits)
    data = llm.complete_json(COMPARE_SYSTEM, f"FOCUS: {question}\n\nEXCERPTS:\n{context}", max_tokens=3000)
    rows = []
    for r in data.get("rows", []):
        tags = [t.upper() for t in r.get("cite", []) if t.upper() in tag_map]
        first = tag_map[tags[0]] if tags else None
        rows.append({
            "paper": r.get("paper") or (first.title if first else ""),
            "arxiv_id": first.arxiv_id if first else None,
            **{k: str(r.get(k, "")) for k in ("task", "method", "data", "metrics", "results", "limitations")},
            "citations": [citation_dict(t, tag_map[t]) for t in tags],
        })
    return {"question": question, "rows": rows}


# ------------------------------------------------------------- conflict detection
FINDINGS_SYSTEM = """Extract atomic empirical findings from numbered excerpts: one self-contained statement each, \
keeping the dataset, setting and numbers it depends on. At most 3 per excerpt; skip excerpts with no findings.
Return ONLY JSON: {"findings":[{"cite":"C1","finding":"..."}]}"""

CONFLICT_CAVEAT = ("Candidates only: NLI flags statements that read as contradictory, but papers often differ in "
                   "dataset, model size or setup. Check the cited pages before treating these as true conflicts.")


def find_conflicts(db: Session, question: str, llm: LLMClient, verifier: ClaimVerifier, top_k: int = 15,
                   paper_ids: Optional[list[int]] = None, min_similarity: float = 0.55, max_pairs: int = 150) -> dict:
    hits = retriever.search(db, question, top_k=top_k, paper_ids=paper_ids)
    context, tag_map = build_context(hits)
    data = llm.complete_json(FINDINGS_SYSTEM, f"EXCERPTS:\n{context}", max_tokens=3000)
    findings = [(f["finding"].strip(), tag_map[f["cite"].upper()]) for f in data.get("findings", [])
                if isinstance(f, dict) and f.get("finding") and str(f.get("cite", "")).upper() in tag_map]
    out = {"question": question, "findings_checked": len(findings), "pairs_checked": 0,
           "conflicts": [], "caveat": CONFLICT_CAVEAT}
    if len(findings) < 2:
        return out

    vecs = retriever.embed_documents([f for f, _ in findings])
    pairs = []
    for i in range(len(findings)):
        for j in range(i + 1, len(findings)):
            if findings[i][1].paper_id == findings[j][1].paper_id:
                continue  # conflicts are only interesting ACROSS papers
            sim = sum(a * b for a, b in zip(vecs[i], vecs[j]))
            if sim >= min_similarity:
                pairs.append((sim, i, j))
    pairs = sorted(pairs, reverse=True)[:max_pairs]
    out["pairs_checked"] = len(pairs)
    if not pairs:
        return out

    ab = verifier.nli([(findings[i][0], findings[j][0]) for _, i, j in pairs])["contradiction"]
    ba = verifier.nli([(findings[j][0], findings[i][0]) for _, i, j in pairs])["contradiction"]

    def fo(k: int) -> dict:
        text, h = findings[k]
        return dict(text=text, arxiv_id=h.arxiv_id, title=h.title, page_start=h.page_start,
                    page_end=h.page_end, chunk_id=h.chunk_id)

    for (sim, i, j), c1, c2 in zip(pairs, ab, ba):
        score = (c1 + c2) / 2  # both directions must agree, which cuts false positives
        if score >= config.CONFLICT_THRESHOLD:
            out["conflicts"].append({"a": fo(i), "b": fo(j), "contradiction": round(score, 4), "similarity": round(sim, 4)})
    out["conflicts"].sort(key=lambda c: c["contradiction"], reverse=True)
    return out