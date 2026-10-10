"""Section-aware chunking: paragraph packing, optional semantic breaks, page tracking.

Every chunk records page_start/page_end so downstream citations can point at pages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from app.services.parser import ParsedPaper

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")
SKIP_SECTIONS = {"references", "bibliography"}


@dataclass
class ChunkData:
    index: int
    content: str
    section: str
    page_start: int
    page_end: int
    kind: str = "text"  # "text" | "caption"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _words(s: str) -> int:
    return len(s.split())


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def chunk_paper(
    parsed: ParsedPaper,
    max_words: int = 220,
    min_words: int = 60,
    overlap: bool = True,
    embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
    sim_threshold: float = 0.55,
) -> list[ChunkData]:
    """Pack paragraphs into ~max_words chunks inside each section.

    * Oversized paragraphs are split on sentence boundaries.
    * With `embed_fn` (returns L2-normalised vectors) a chunk also breaks where adjacent
      paragraphs' cosine similarity drops below `sim_threshold` (semantic chunking).
    * On size-based breaks the last sentence is carried into the next chunk (overlap).
    * Figure/table captions become their own chunks. References are skipped.
    """
    # Phase 1: flatten into units (text, page, section) and caption markers.
    units: list[tuple[str, int, str, str]] = []  # text, page, section, kind
    for b in parsed.blocks:
        if b.kind == "heading" or b.section.lower() in SKIP_SECTIONS:
            continue
        if b.kind == "caption":
            if _words(b.text) >= 8:
                units.append((b.text, b.page, b.section, "caption"))
        elif _words(b.text) <= max_words:
            units.append((b.text, b.page, b.section, "body"))
        else:
            for s in split_sentences(b.text):
                units.append((s, b.page, b.section, "body"))

    vecs: list[Optional[list[float]]] = [None] * len(units)
    if embed_fn:
        idx = [i for i, u in enumerate(units) if u[3] == "body"]
        if idx:
            for i, v in zip(idx, embed_fn([units[i][0] for i in idx])):
                vecs[i] = v

    # Phase 2: pack.
    chunks: list[ChunkData] = []
    cur: list[tuple[str, int]] = []
    cur_section = ""
    prev_vec: Optional[list[float]] = None

    carried_only = False  # cur holds just an overlap sentence: never emit it as a chunk

    def flush(carry: bool = False) -> None:
        nonlocal cur, carried_only
        if not cur or carried_only:
            cur, carried_only = [], False
            return
        text = " ".join(t for t, _ in cur)
        pages = [p for _, p in cur]
        chunks.append(ChunkData(len(chunks), text, cur_section, min(pages), max(pages)))
        last = split_sentences(text)[-1] if carry else ""
        if carry and 0 < _words(last) <= 40:
            cur, carried_only = [(last, pages[-1])], True
        else:
            cur = []

    for i, (text, page, section, kind) in enumerate(units):
        if kind == "caption":
            flush()
            chunks.append(ChunkData(len(chunks), text, section, page, page, kind="caption"))
            prev_vec = None
            continue
        if section != cur_section:
            flush()
            cur_section, prev_vec = section, None
        cur_words = sum(_words(t) for t, _ in cur)
        v = vecs[i]
        if cur and cur_words >= min_words:
            if cur_words + _words(text) > max_words:
                flush(carry=overlap)
            elif v is not None and prev_vec is not None and _dot(v, prev_vec) < sim_threshold:
                flush()
        cur.append((text, page))
        carried_only = False
        prev_vec = v
    flush()

    # Merge a tiny trailing text chunk into its predecessor (same section).
    merged: list[ChunkData] = []
    for c in chunks:
        if (
            merged
            and c.kind == "text"
            and merged[-1].kind == "text"
            and merged[-1].section == c.section
            and _words(c.content) < min_words // 2
        ):
            p = merged[-1]
            p.content += " " + c.content
            p.page_end = max(p.page_end, c.page_end)
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c.index = i
    return merged