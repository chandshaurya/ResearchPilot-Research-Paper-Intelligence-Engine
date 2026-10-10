"""THE STANDOUT FEATURE: detect when a generated claim isn't supported by its cited paper.

Pipeline for each (claim, cited chunks):
  1. Slice the cited text into sentence windows and keep the ones most lexically similar to the claim.
  2. Score every (window -> claim) pair with an NLI cross-encoder (PyTorch / transformers):
     P(entailment), P(contradiction), P(neutral).
  3. Verdict:
       entailment >= SUPPORT_THRESHOLD ........................ supported
       contradiction >= CONTRADICTION_THRESHOLD and > entail .. contradicted
       borderline (either prob >= BORDERLINE_THRESHOLD) ....... ask the LLM judge if available
       otherwise ............................................... unsupported
  4. Numeric guard: NLI models are weak at numbers, so a "supported" claim whose figures
     (e.g. 87.3, 12%) don't appear in the cited chunk is downgraded to partially_supported.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Optional

from app.core import config
from app.services.chunker import split_sentences

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are", "was", "were", "with",
    "that", "this", "by", "as", "at", "from", "it", "be", "their", "than", "which", "can", "has", "have",
}


@dataclass
class EvidenceChunk:
    chunk_id: int
    content: str
    page_start: int
    page_end: int
    title: str = ""
    arxiv_id: Optional[str] = None


@dataclass
class Verification:
    verdict: str  # supported | partially_supported | unsupported | contradicted | uncited
    entailment: float = 0.0
    contradiction: float = 0.0
    evidence: Optional[str] = None
    chunk_id: Optional[int] = None
    arxiv_id: Optional[str] = None
    title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    method: str = "nli"
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- helpers
def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOP}


def candidate_windows(chunk: EvidenceChunk, claim: str, size: int = 2, keep: int = 12) -> list[str]:
    """Sentence windows of `size`, pruned to the `keep` most lexically similar to the claim."""
    sents = split_sentences(chunk.content) or [chunk.content]
    windows = [" ".join(sents[i : i + size]) for i in range(max(1, len(sents) - size + 1))]
    ct = _tokens(claim)
    scored = sorted(windows, key=lambda w: len(ct & _tokens(w)) / (len(ct) or 1), reverse=True)
    return scored[:keep]


def numbers_missing(claim: str, evidence_text: str) -> list[str]:
    """Salient numbers in the claim (decimals, percentages, 2+ digit ints) absent from the evidence."""
    missing = []
    for m in re.findall(r"\d+(?:\.\d+)?%?", claim):
        bare = m.rstrip("%")
        if "." not in bare and "%" not in m and len(bare) < 2:
            continue  # ignore bare single digits ("3 datasets")
        if not re.search(rf"(?<![\d.]){re.escape(bare)}(?!\d)", evidence_text):
            missing.append(m)
    return missing


# --------------------------------------------------------------- verifier
class ClaimVerifier:
    def __init__(self, model_name: str = config.NLI_MODEL, llm: Any = None, device: Optional[str] = None):
        """`llm` is any object with `.complete_json(system, user)`; used only for borderline cases."""
        self.model_name = model_name
        self.llm = llm
        self.device = device
        self._tok = self._model = None

    # -- NLI -------------------------------------------------------------
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self._model.config.id2label.items()}
        missing = {"entailment", "contradiction", "neutral"} - set(labels)
        if missing:
            raise ValueError(f"{self.model_name} lacks NLI labels {missing}; got {sorted(labels)}")
        self._idx = labels

    def nli(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> dict[str, list[float]]:
        """pairs = [(premise, hypothesis)] -> class probabilities per pair."""
        import torch

        self._load()
        out: dict[str, list[float]] = {"entailment": [], "contradiction": [], "neutral": []}
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            enc = self._tok(
                [p for p, _ in batch], [h for _, h in batch],
                truncation=True, max_length=512, padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self._model(**enc).logits, dim=-1).cpu()
            for name in out:
                out[name].extend(probs[:, self._idx[name]].tolist())
        return out

    # -- main entry ------------------------------------------------------
    def verify(self, claim: str, evidence: list[EvidenceChunk]) -> Verification:
        cands: list[tuple[str, EvidenceChunk]] = []
        for ch in evidence:
            for w in candidate_windows(ch, claim):
                cands.append((w, ch))
        if not cands:
            return Verification("unsupported", note="no evidence text was provided")

        p = self.nli([(w, claim) for w, _ in cands])
        ent, con = p["entailment"], p["contradiction"]
        bi = max(range(len(cands)), key=ent.__getitem__)
        bc = max(range(len(cands)), key=con.__getitem__)
        best_e, best_c = ent[bi], con[bc]

        def make(verdict: str, idx: int, method: str = "nli", note: str = "") -> Verification:
            w, ch = cands[idx]
            return Verification(
                verdict, round(best_e, 4), round(best_c, 4), w, ch.chunk_id, ch.arxiv_id,
                ch.title, ch.page_start, ch.page_end, method, note,
            )

        if best_e >= config.SUPPORT_THRESHOLD:
            v = make("supported", bi)
            all_text = " ".join(c.content for c in evidence)
            gone = numbers_missing(claim, all_text)
            if gone:
                v.verdict = "partially_supported"
                v.note = f"numbers not found in cited text: {', '.join(gone)}"
            return v
        if best_c >= config.CONTRADICTION_THRESHOLD and best_c > best_e:
            return make("contradicted", bc)
        if max(best_e, best_c) >= config.BORDERLINE_THRESHOLD and self.llm is not None:
            judged = self._llm_judge(claim, [cands[i] for i in self._top(ent, 3)])
            if judged:
                verdict, reason = judged
                return make(verdict, bi, method="llm", note=reason)
        return make("unsupported", bi, note="no passage in the cited text entails this claim")

    @staticmethod
    def _top(scores: list[float], n: int) -> list[int]:
        return sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:n]

    def _llm_judge(self, claim: str, passages: list[tuple[str, EvidenceChunk]]) -> Optional[tuple[str, str]]:
        system = (
            "You verify whether a claim is supported by passages from a cited paper. "
            "Judge ONLY from the passages. Return JSON: "
            '{"verdict": "supported|partially_supported|unsupported|contradicted", "reason": "<one sentence>"}'
        )
        body = "\n".join(f"[{i + 1}] {w}" for i, (w, _) in enumerate(passages))
        try:
            data = self.llm.complete_json(system, f"CLAIM: {claim}\n\nPASSAGES:\n{body}", max_tokens=300)
            verdict = str(data.get("verdict", "")).lower()
            if verdict in {"supported", "partially_supported", "unsupported", "contradicted"}:
                return verdict, str(data.get("reason", ""))[:300]
        except Exception:  # judge is best-effort; fall back to the NLI verdict
            return None
        return None