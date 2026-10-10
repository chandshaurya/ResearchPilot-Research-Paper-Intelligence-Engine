"""Academic PDF parser: text per page, section tracking, figure/table captions.

Uses PyMuPDF. LaTeX-generated PDFs store text blocks in reading order (column by column),
so we keep the stream order instead of sorting by y/x, which would interleave two-column text.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SECTION_NAMES = {
    "abstract", "introduction", "related work", "background", "preliminaries", "problem formulation",
    "method", "methods", "methodology", "approach", "model", "framework", "experiments",
    "experimental setup", "experimental results", "results", "evaluation", "analysis", "discussion",
    "limitations", "conclusion", "conclusions", "future work", "acknowledgments", "acknowledgements",
    "references", "bibliography", "appendix",
}
_NUM_PREFIX = re.compile(r"^(\d+(\.\d+){0,3}|[A-Z])\.?\s+")
_NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+){0,2})\.?\s+[A-Z][A-Za-z0-9 ,:\-–&/()']{2,90}$")
_CAPTION = re.compile(r"^(Figure|Fig\.|Table)\s*\d+\s*[.:|]", re.I)
_INLINE_ABSTRACT = re.compile(r"^abstract[\s.:—–-]+(.{200,})$", re.I | re.S)


@dataclass
class Block:
    text: str
    page: int  # 1-based
    section: str
    kind: str  # "body" | "heading" | "caption"
    font_size: float = 0.0


@dataclass
class ParsedPaper:
    title: str
    num_pages: int
    blocks: list[Block]

    @property
    def captions(self) -> list[Block]:
        return [b for b in self.blocks if b.kind == "caption"]

    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.kind != "heading")


def _clean(text: str) -> str:
    text = text.replace("\x00", "")  # some PDFs embed NUL bytes; PostgreSQL text columns reject them
    text = unicodedata.normalize("NFKC", text)  # expands ligatures (ﬁ -> fi)
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line breaks
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _heading_name(text: str) -> str:
    """'3.1 Experimental Setup' -> 'experimental setup'."""
    return _NUM_PREFIX.sub("", text.strip()).strip(" .:").lower()


def _is_known_heading(text: str) -> bool:
    return len(text) < 60 and _heading_name(text) in SECTION_NAMES


def _is_heading(text: str, size: float, bold_ratio: float, body_size: float) -> bool:
    if len(text) > 100 or len(text.split()) > 12 or text.endswith((",", ";")):
        return False
    if _is_known_heading(text):
        return True
    styled = bold_ratio > 0.8 or size >= body_size * 1.12
    if _NUMBERED_HEADING.match(text) and styled:
        return True
    return styled and not text.endswith(".") and len(text.split()) <= 8 and text[:1].isupper()


def parse_pdf(path: str | Path) -> ParsedPaper:
    import fitz  # PyMuPDF (imported lazily so the rest of the package imports without it)

    doc = fitz.open(str(path))
    raw: list[dict] = []
    size_counts: Counter = Counter()

    for pno, page in enumerate(doc, start=1):
        height = page.rect.height
        for b in page.get_text("dict")["blocks"]:
            if b.get("type") != 0:
                continue
            lines, max_size, bold, chars, rotated = [], 0.0, 0, 0, False
            for ln in b["lines"]:
                if abs(ln["dir"][0]) < 0.9:  # vertical text, e.g. the arXiv margin stamp
                    rotated = True
                    break
                parts = []
                for sp in ln["spans"]:
                    t = sp["text"]
                    parts.append(t)
                    n = len(t.strip())
                    if n:
                        chars += n
                        max_size = max(max_size, sp["size"])
                        size_counts[round(sp["size"] * 2) / 2] += n
                        if sp["flags"] & 16 or "bold" in sp["font"].lower():
                            bold += n
                lines.append("".join(parts))
            if rotated or not chars:
                continue
            text = _clean("\n".join(lines))
            if not text:
                continue
            y0, y1 = b["bbox"][1], b["bbox"][3]
            running = (y0 < 0.06 * height or y1 > 0.94 * height) and len(text) < 100
            raw.append(
                dict(page=pno, text=text, size=max_size, bold=bold / chars, running=running)
            )

    body_size = size_counts.most_common(1)[0][0] if size_counts else 10.0

    # Drop running headers/footers: short blocks repeating (modulo digits) on many pages.
    n_pages = max(len(doc), 1)
    repeat = Counter()
    for r in raw:
        if len(r["text"]) < 80:
            repeat[(re.sub(r"\d+", "#", r["text"]).lower())] += 1
    min_repeat = max(3, int(0.3 * n_pages))
    raw = [
        r
        for r in raw
        if not (
            r["running"]
            or (len(r["text"]) < 80 and repeat[re.sub(r"\d+", "#", r["text"]).lower()] >= min_repeat)
        )
    ]

    # Title: PDF metadata if sensible, else the largest-font text on page 1.
    title = (doc.metadata or {}).get("title", "") or ""
    if len(title) < 8 or title.lower().startswith(("untitled", "microsoft word")):
        first = [r for r in raw if r["page"] == 1]
        if first:
            top = max(r["size"] for r in first)
            title = " ".join(r["text"] for r in first if r["size"] >= top - 0.5)[:300]
    title = title.strip() or Path(path).stem

    has_known = any(_is_known_heading(r["text"]) for r in raw)
    started = not has_known  # without recognisable headings, treat the whole doc as body
    section = "Body" if not has_known else "Front matter"
    blocks: list[Block] = []

    for r in raw:
        text = r["text"]
        inline = _INLINE_ABSTRACT.match(text)
        if inline and not started:
            started, section = True, "Abstract"
            blocks.append(Block("Abstract", r["page"], section, "heading", r["size"]))
            blocks.append(Block(inline.group(1).strip(), r["page"], section, "body", r["size"]))
            continue
        if _CAPTION.match(text):
            blocks.append(Block(text, r["page"], section, "caption", r["size"]))
            continue
        known = _is_known_heading(text)
        if (started or known) and _is_heading(text, r["size"], r["bold"], body_size):
            started = True
            section = _NUM_PREFIX.sub("", text).strip(" .:").title() if known else text.strip(" .:")
            blocks.append(Block(text, r["page"], section, "heading", r["size"]))
            continue
        blocks.append(Block(text, r["page"], section, "body", r["size"]))

    return ParsedPaper(title=title, num_pages=len(doc), blocks=blocks)