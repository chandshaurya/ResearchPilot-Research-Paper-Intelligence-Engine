"""arXiv ingestion client for ResearchPilot.

Cost-saving design (arXiv's API is free but rate-limited and slow, so "cost" = requests + time):
  1. Round-robin across topics: one page per topic per turn, so every topic makes progress
     and no single broad query burns the whole request budget.
  2. Global rate limit: one request per >= 3 s (arXiv's published guideline), single connection.
  3. On-disk response cache: identical page requests are never re-fetched within the TTL.
  4. Incremental stop: a topic whose page contains only already-seen papers is retired early.
  5. Persistent state: per-topic cursors + seen IDs survive restarts, so reruns only fetch new papers.
  6. Hard request budget per run (max_requests).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterator, Optional

import httpx

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
_VERSION_RE = re.compile(r"v\d+$")


@dataclass
class Topic:
    name: str
    query: str  # arXiv search syntax, e.g. 'cat:cs.CL AND all:"retrieval augmented generation"'


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    updated: str
    categories: list[str]
    pdf_url: str
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    topic: str = ""


@dataclass
class _TopicState:
    start: int = 0
    total: Optional[int] = None
    done: bool = False


@dataclass
class _State:
    topics: dict[str, _TopicState] = field(default_factory=dict)
    seen_ids: set[str] = field(default_factory=set)


class ArxivClient:
    def __init__(
        self,
        cache_dir: str | Path = ".arxiv_cache",
        min_interval: float = 3.0,
        cache_ttl_hours: float = 24.0,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
        http: Optional[httpx.Client] = None,
    ):
        self.cache_dir = Path(cache_dir)
        (self.cache_dir / "pages").mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "pdfs").mkdir(parents=True, exist_ok=True)
        self.state_path = self.cache_dir / "state.json"
        self.min_interval = min_interval
        self.cache_ttl = cache_ttl_hours * 3600
        self._sleep = sleep
        self._http = http or httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "ResearchPilot/0.1 (academic research tool)"},
        )
        self._last_request = 0.0
        self.requests_made = 0
        self.cache_hits = 0
        self.state = self._load_state()

    # ------------------------------------------------------------------ state
    def _load_state(self) -> _State:
        if not self.state_path.exists():
            return _State()
        raw = json.loads(self.state_path.read_text())
        return _State(
            topics={k: _TopicState(**v) for k, v in raw.get("topics", {}).items()},
            seen_ids=set(raw.get("seen_ids", [])),
        )

    def save_state(self) -> None:
        payload = {
            "topics": {k: asdict(v) for k, v in self.state.topics.items()},
            "seen_ids": sorted(self.state.seen_ids),
        }
        self.state_path.write_text(json.dumps(payload))

    # ------------------------------------------------------------ rate limit
    # Shared by every client instance in the process, so UI searches, selected-paper ingests and
    # round-robin runs together still respect arXiv's one-request-per-3-seconds guideline.
    _rate_lock = threading.Lock()
    _last_request_ts = 0.0

    def _throttle(self) -> None:
        cls = type(self)
        with cls._rate_lock:
            wait = self.min_interval - (time.monotonic() - cls._last_request_ts)
            if wait > 0:
                self._sleep(wait)
            cls._last_request_ts = time.monotonic()

    def _get(self, url: str, params: Optional[dict] = None) -> httpx.Response:
        for attempt in range(4):
            self._throttle()
            self.requests_made += 1
            resp = self._http.get(url, params=params)
            if resp.status_code in (429, 503):  # back off, honour Retry-After if given
                delay = float(resp.headers.get("Retry-After", 5 * (2 ** attempt)))
                self._sleep(delay)
                continue
            resp.raise_for_status()
            return resp
        raise RuntimeError(f"arXiv kept rate-limiting us: {url}")

    # ----------------------------------------------------------------- fetch
    def _cached_get(self, cache_key: str, params: dict) -> str:
        """One API call, served from the on-disk cache when a fresh copy exists."""
        key = hashlib.sha1(cache_key.encode()).hexdigest()
        cache_file = self.cache_dir / "pages" / f"{key}.xml"
        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < self.cache_ttl:
            self.cache_hits += 1
            return cache_file.read_text(encoding="utf-8")
        resp = self._get(ARXIV_API, params=params)
        cache_file.write_text(resp.text, encoding="utf-8")
        return resp.text

    def _fetch_page(self, query: str, start: int, page_size: int, sort_by: str = "submittedDate") -> str:
        return self._cached_get(
            f"{query}|{start}|{page_size}|{sort_by}",
            {"search_query": query, "start": start, "max_results": page_size,
             "sortBy": sort_by, "sortOrder": "descending"},
        )

    # ------------------------------------------------- interactive (UI) helpers
    def search(self, query: str, start: int = 0, max_results: int = 15,
               sort_by: str = "relevance") -> tuple[list[ArxivPaper], int]:
        """One page of results for a topic search -> (papers, total matches)."""
        return self._parse(self._fetch_page(query, start, max_results, sort_by), topic="")

    def fetch_by_ids(self, ids: list[str]) -> list[ArxivPaper]:
        """Metadata for many papers in ONE request (instead of one request per paper)."""
        ids = sorted(set(ids))
        xml_text = self._cached_get(
            "ids|" + ",".join(ids), {"id_list": ",".join(ids), "max_results": len(ids)}
        )
        return self._parse(xml_text, topic="")[0]

    @staticmethod
    def _parse(xml_text: str, topic: str) -> tuple[list[ArxivPaper], int]:
        root = ET.fromstring(xml_text)
        total = int(root.findtext(f"{OPENSEARCH}totalResults") or 0)
        papers: list[ArxivPaper] = []
        for e in root.findall(f"{ATOM}entry"):
            full_id = e.findtext(f"{ATOM}id") or ""
            if "/abs/" not in full_id:  # arXiv reports query errors as a fake <entry>
                continue
            raw_id = full_id.rsplit("/abs/", 1)[-1]
            aid = _VERSION_RE.sub("", raw_id)
            pdf = next(
                (l.get("href") for l in e.findall(f"{ATOM}link") if l.get("title") == "pdf"),
                f"https://arxiv.org/pdf/{aid}",
            )
            papers.append(
                ArxivPaper(
                    arxiv_id=aid,
                    title=" ".join((e.findtext(f"{ATOM}title") or "").split()),
                    abstract=" ".join((e.findtext(f"{ATOM}summary") or "").split()),
                    authors=[a.findtext(f"{ATOM}name") or "" for a in e.findall(f"{ATOM}author")],
                    published=e.findtext(f"{ATOM}published") or "",
                    updated=e.findtext(f"{ATOM}updated") or "",
                    categories=[c.get("term", "") for c in e.findall(f"{ATOM}category")],
                    pdf_url=pdf,
                    doi=e.findtext(f"{ARXIV}doi"),
                    journal_ref=e.findtext(f"{ARXIV}journal_ref"),
                    topic=topic,
                )
            )
        return papers, total

    # ------------------------------------------------------------ round robin
    def round_robin(
        self,
        topics: list[Topic],
        page_size: int = 50,
        max_requests: int = 20,
        max_papers: Optional[int] = None,
    ) -> Iterator[ArxivPaper]:
        """Yield NEW papers, cycling topics one page at a time until budget or topics run out."""
        queue: deque[Topic] = deque()
        for t in topics:
            st = self.state.topics.setdefault(t.name, _TopicState())
            if not st.done:
                queue.append(t)

        start_requests, yielded = self.requests_made, 0
        try:
            while queue and (self.requests_made - start_requests) < max_requests:
                topic = queue.popleft()
                st = self.state.topics[topic.name]
                xml_text = self._fetch_page(topic.query, st.start, page_size)
                papers, total = self._parse(xml_text, topic.name)
                st.total = total
                new = [p for p in papers if p.arxiv_id not in self.state.seen_ids]

                for p in new:
                    self.state.seen_ids.add(p.arxiv_id)
                    yield p
                    yielded += 1
                    if max_papers and yielded >= max_papers:
                        return

                st.start += len(papers)
                exhausted = not papers or st.start >= total
                caught_up = bool(papers) and not new  # page fully seen -> older pages are too
                if exhausted or caught_up:
                    st.done = exhausted  # caught_up topics resume from start=0 next run
                    if caught_up:
                        st.start = 0
                else:
                    queue.append(topic)  # back of the line: round robin
        finally:
            self.save_state()

    # ------------------------------------------------------------------ pdfs
    def download_pdf(self, paper: ArxivPaper) -> Path:
        path = self.cache_dir / "pdfs" / f"{paper.arxiv_id.replace('/', '_')}.pdf"
        if path.exists() and path.stat().st_size > 0:
            return path
        path.write_bytes(self._get(paper.pdf_url).content)
        return path

    def close(self, save: bool = True) -> None:
        if save:  # interactive search/selection never touches the round-robin cursors
            self.save_state()
        self._http.close()