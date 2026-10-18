"""
ResearchPilot — Streamlit frontend
----------------------------------
This app ports the main flows from the supplied index.html:
- Search arXiv papers
- Select papers and ingest/index them through the existing API
- Browse the indexed library
- Generate summaries, literature reviews, method comparisons, conflict checks,
  claim verification, and passage searches
- Download Markdown and BibTeX exports

Run:
    pip install streamlit requests
    streamlit run streamlit_app.py

Configure the FastAPI backend with:
    RESEARCHPILOT_API_URL=http://localhost:8000

The backend must expose the same endpoints used by the original HTML frontend.
"""
from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

st.set_page_config(page_title="ResearchPilot", page_icon="📚", layout="wide")

API_BASE = os.getenv("RESEARCHPILOT_API_URL", "http://localhost:8000").rstrip("/")
PAGE_SIZE = 15

CATEGORIES = {
    "Any field": "",
    "Computation and language": "cs.CL",
    "Machine learning": "cs.LG",
    "Artificial intelligence": "cs.AI",
    "Computer vision": "cs.CV",
    "Information retrieval": "cs.IR",
    "Statistics: machine learning": "stat.ML",
    "Biomolecules": "q-bio.BM",
}
EXAMPLES = [
    "retrieval augmented generation",
    "protein solubility prediction",
    "vision language model hallucination",
    "graph neural networks",
]
STATUS_LABELS = {
    "queued": "Waiting",
    "downloading": "Downloading PDF",
    "processing": "Reading and indexing",
    "ready": "Ready",
    "failed": "Failed",
}


def api(path: str, method: str = "GET", body: dict | None = None, params: dict | None = None) -> Any:
    """Call the ResearchPilot API and raise readable errors."""
    url = f"{API_BASE}{path}"
    try:
        response = requests.request(
            method,
            url,
            json=body,
            params=params,
            timeout=180,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Cannot reach the ResearchPilot API at {API_BASE}. "
            "Start the backend or set RESEARCHPILOT_API_URL. "
            f"Details: {exc}"
        ) from exc

    if not response.ok:
        try:
            payload = response.json()
            detail = payload.get("detail", payload)
        except ValueError:
            detail = response.text or response.reason
        raise RuntimeError(f"API error ({response.status_code}): {detail}")

    if not response.content:
        return {}
    try:
        return response.json()
    except ValueError:
        return response.text


def init_state() -> None:
    defaults = {
        "search_results": [],
        "search_total": 0,
        "search_start": 0,
        "search_query": "",
        "search_category": "",
        "search_sort": "relevance",
        "selected": {},
        "library": [],
        "library_selected": set(),
        "workspace": [],
        "job_id": None,
        "active_tab": "Literature review",
        "outputs": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def paper_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/abs/{arxiv_id}"


def render_paper(paper: dict, key_prefix: str, selected: bool = False) -> bool:
    """Render one paper and return its current checkbox state."""
    arxiv_id = paper.get("arxiv_id", "")
    title = paper.get("title", "Untitled paper")
    authors = paper.get("authors") or []
    author_line = ", ".join(authors[:3]) + (" and others" if len(authors) > 3 else "")
    with st.container(border=True):
        checked = st.checkbox(
            title,
            value=selected,
            key=f"{key_prefix}_{arxiv_id}",
            disabled=not bool(arxiv_id),
        )
        metadata = [x for x in [author_line, paper.get("published"), ", ".join((paper.get("categories") or [])[:3])] if x]
        if metadata:
            st.caption(" · ".join(metadata))
        abstract = paper.get("abstract", "")
        if abstract:
            st.write(abstract)
        cols = st.columns([1, 1, 4])
        if arxiv_id:
            cols[0].link_button("Open on arXiv", paper_url(arxiv_id), use_container_width=True)
        if paper.get("pdf_url"):
            cols[1].link_button("PDF", paper["pdf_url"], use_container_width=True)
    return checked


def selected_workspace_ids() -> list[str]:
    return [p["arxiv_id"] for p in st.session_state.workspace if p.get("arxiv_id") and p.get("status") == "ready"]


def refresh_job_status() -> None:
    job_id = st.session_state.job_id
    if not job_id:
        return
    job = api(f"/papers/jobs/{job_id}")
    job_papers = job.get("papers", {})
    for item in st.session_state.workspace:
        info = job_papers.get(item.get("arxiv_id"), {})
        if info:
            item["status"] = info.get("status", item.get("status", "queued"))
            item["message"] = info.get("message", "")
            item["title"] = info.get("title") or item.get("title")
    if job.get("status") == "done":
        st.session_state.job_id = None


def start_ingestion(papers: list[dict]) -> None:
    if not papers:
        st.warning("Select at least one paper first.")
        return
    result = api("/papers/ingest", method="POST", body={"arxiv_ids": [p["arxiv_id"] for p in papers]})
    st.session_state.job_id = result["job_id"]
    st.session_state.workspace = [
        {
            "arxiv_id": p["arxiv_id"],
            "title": p.get("title", p["arxiv_id"]),
            "status": "queued",
            "message": "",
        }
        for p in papers
    ]
    st.session_state.selected = {}
    st.session_state.outputs = {}
    st.success("Ingestion started. Use **Refresh job status** below to check progress.")


def render_json_result(data: Any) -> None:
    """Display API output while keeping all backend fields available."""
    if isinstance(data, dict):
        markdown = data.get("markdown")
        if markdown:
            st.markdown(markdown)
        sections = data.get("sections")
        if sections:
            for section in sections:
                st.subheader(section.get("heading", "Section"))
                for claim in section.get("claims", []):
                    verdict = claim.get("verdict", "unknown").replace("_", " ").title()
                    st.markdown(f"**{verdict}:** {claim.get('text', '')}")
                    if claim.get("cite"):
                        st.caption("Citations: " + ", ".join(claim["cite"]))
                    if claim.get("evidence"):
                        with st.expander("Evidence"):
                            st.write(claim["evidence"])
            if data.get("citations"):
                with st.expander("References"):
                    for citation in data["citations"]:
                        arxiv_id = citation.get("arxiv_id", "")
                        st.markdown(
                            f"- **{citation.get('tag', '')}** {citation.get('title', '')} "
                            f"[arXiv:{arxiv_id}]({paper_url(arxiv_id)})"
                        )
            return

        if "rows" in data and isinstance(data["rows"], list):
            if data["rows"]:
                st.dataframe(data["rows"], use_container_width=True, hide_index=True)
            else:
                st.info("No comparison rows were returned.")
            return

        if "hits" in data and isinstance(data["hits"], list):
            if not data["hits"]:
                st.info("No passages matched.")
            for hit in data["hits"]:
                st.markdown(f"**{hit.get('title', 'Passage')}**")
                st.caption(
                    " · ".join(
                        str(x) for x in [hit.get("section"), hit.get("arxiv_id"), hit.get("page_start")]
                        if x is not None and x != ""
                    )
                )
                st.write(hit.get("content", ""))
                st.divider()
            return

        if "result" in data and isinstance(data["result"], dict):
            result = data["result"]
            st.markdown(f"### Verdict: {result.get('verdict', 'unknown').replace('_', ' ').title()}")
            c1, c2 = st.columns(2)
            c1.metric("Entailment", f"{float(result.get('entailment', 0)):.1%}")
            c2.metric("Contradiction", f"{float(result.get('contradiction', 0)):.1%}")
            if result.get("note"):
                st.info(result["note"])
            if result.get("evidence"):
                st.markdown("**Evidence**")
                st.write(result["evidence"])
            return

        if "conflicts" in data and isinstance(data["conflicts"], list):
            st.write(
                f"Compared {data.get('findings_checked', 0)} findings across "
                f"{data.get('pairs_checked', 0)} related pairs."
            )
            if not data["conflicts"]:
                st.success("No contradictions found among the retrieved findings.")
            for conflict in data["conflicts"]:
                st.warning(f"Potential conflict: {float(conflict.get('contradiction', 0)):.1%}")
                left, right = st.columns(2)
                for col, side in [(left, conflict.get("a", {})), (right, conflict.get("b", {}))]:
                    with col:
                        st.write(side.get("text", ""))
                        st.caption(side.get("title", ""))
            if data.get("caveat"):
                st.caption(data["caveat"])
            return

    st.json(data)


def markdown_export(tab_label: str, output: Any) -> str:
    titles = "\n".join(
        f"- {p.get('title', p.get('arxiv_id', 'Paper'))} (arXiv:{p.get('arxiv_id', '')})"
        for p in st.session_state.workspace
    ) or "- No papers"
    if isinstance(output, str):
        body = output
    elif isinstance(output, dict):
        body = output.get("markdown") or "```json\n" + __import__("json").dumps(output, indent=2, ensure_ascii=False) + "\n```"
    else:
        body = str(output) if output else "_No output generated yet._"
    return f"# ResearchPilot — {tab_label}\n\n## Workspace papers\n{titles}\n\n## Output\n\n{body}\n"


def bibtex_export() -> str:
    entries = []
    for i, paper in enumerate(st.session_state.workspace, 1):
        arxiv_id = str(paper.get("arxiv_id", ""))
        key = "arxiv" + "".join(ch for ch in (arxiv_id or str(i)) if ch.isalnum())
        title = str(paper.get("title", "Untitled")).replace("{", "").replace("}", "").replace("\n", " ")
        entries.append(
            f"@misc{{{key},\n  title = {{{title}}},\n  eprint = {{{arxiv_id}}},\n"
            f"  archivePrefix = {{arXiv}},\n  url = {{{paper_url(arxiv_id)}}}\n}}"
        )
    return "\n\n".join(entries)


init_state()

st.title("📚 ResearchPilot")
st.caption("Find papers, build a research library, and synthesize evidence from indexed PDFs.")
with st.sidebar:
    st.subheader("Backend")
    st.code(API_BASE, language=None)
    st.caption("Set RESEARCHPILOT_API_URL if your API runs somewhere else.")
    if st.button("Check API connection", use_container_width=True):
        try:
            api("/papers")
            st.success("API is reachable.")
        except RuntimeError as exc:
            st.error(str(exc))

search_tab, library_tab, workspace_tab = st.tabs(["Find papers", "Library", "Workspace"])

with search_tab:
    st.header("What do you want to read about?")
    with st.form("arxiv_search_form"):
        query = st.text_input("Topic", value=st.session_state.search_query, placeholder="retrieval augmented generation")
        category_label = st.selectbox(
            "Field",
            list(CATEGORIES),
            index=list(CATEGORIES.values()).index(st.session_state.search_category),
        )
        sort_label = st.radio(
            "Sort by",
            ["Best match", "Newest"],
            horizontal=True,
            index=0 if st.session_state.search_sort == "relevance" else 1,
        )
        submitted = st.form_submit_button("Search arXiv", type="primary")
    st.caption("Try one of these topics: " + " · ".join(EXAMPLES))
    chosen_example = st.selectbox("Example searches", ["(choose an example)"] + EXAMPLES, label_visibility="collapsed")
    if chosen_example != "(choose an example)":
        st.session_state.example_query = chosen_example
        st.info(f"Selected example: {chosen_example}. Copy it into the Topic field and search.")

    if submitted:
        if len(query.strip()) < 2:
            st.warning("Type a topic first.")
        else:
            st.session_state.search_query = query.strip()
            st.session_state.search_category = CATEGORIES[category_label]
            st.session_state.search_sort = "relevance" if sort_label == "Best match" else "submittedDate"
            st.session_state.search_results = []
            st.session_state.search_start = 0
            try:
                with st.spinner("Searching arXiv…"):
                    result = api(
                        "/papers/arxiv-search",
                        params={
                            "q": st.session_state.search_query,
                            "start": 0,
                            "limit": PAGE_SIZE,
                            "sort": st.session_state.search_sort,
                            **({"category": st.session_state.search_category} if st.session_state.search_category else {}),
                        },
                    )
                st.session_state.search_results = result.get("results", [])
                st.session_state.search_total = result.get("total", len(st.session_state.search_results))
                st.session_state.search_start = len(st.session_state.search_results)
            except RuntimeError as exc:
                st.error(str(exc))

    if st.session_state.search_results:
        st.caption(
            f"{st.session_state.search_total:,} papers match. "
            f"Showing {len(st.session_state.search_results)}."
        )
        for paper in st.session_state.search_results:
            paper_id = paper.get("arxiv_id")
            currently_selected = paper_id in st.session_state.selected
            checked = render_paper(paper, "search_pick", currently_selected)
            if checked:
                st.session_state.selected[paper_id] = {
                    "arxiv_id": paper_id,
                    "title": paper.get("title", paper_id),
                }
            else:
                st.session_state.selected.pop(paper_id, None)

        col1, col2, col3 = st.columns([1, 1, 3])
        col1.metric("Selected", len(st.session_state.selected))
        if col2.button("Clear selection"):
            st.session_state.selected = {}
            st.rerun()
        if col3.button("Start work with selected papers", type="primary", disabled=not st.session_state.selected):
            try:
                start_ingestion(list(st.session_state.selected.values()))
            except (RuntimeError, KeyError) as exc:
                st.error(str(exc))

        if len(st.session_state.search_results) < st.session_state.search_total:
            if st.button("Show more papers"):
                try:
                    with st.spinner("Loading more papers…"):
                        result = api(
                            "/papers/arxiv-search",
                            params={
                                "q": st.session_state.search_query,
                                "start": st.session_state.search_start,
                                "limit": PAGE_SIZE,
                                "sort": st.session_state.search_sort,
                                **({"category": st.session_state.search_category} if st.session_state.search_category else {}),
                            },
                        )
                    existing = {p.get("arxiv_id") for p in st.session_state.search_results}
                    st.session_state.search_results.extend(
                        p for p in result.get("results", []) if p.get("arxiv_id") not in existing
                    )
                    st.session_state.search_start = len(st.session_state.search_results)
                    st.session_state.search_total = result.get("total", st.session_state.search_total)
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
    elif st.session_state.search_query and not submitted:
        st.info("No results are currently stored. Submit a new search to refresh them.")

with library_tab:
    st.header("Your library")
    if st.button("Refresh library"):
        try:
            st.session_state.library = api("/papers").get("papers", [])
        except RuntimeError as exc:
            st.error(str(exc))
    if not st.session_state.library:
        try:
            st.session_state.library = api("/papers").get("papers", [])
        except RuntimeError as exc:
            st.warning(str(exc))
    if not st.session_state.library:
        st.info("Your library is empty. Search for papers and start work to index them.")
    else:
        st.caption("Select papers already indexed to use them in the workspace.")
        for paper in st.session_state.library:
            paper_id = paper.get("arxiv_id")
            with st.container(border=True):
                st.markdown(f"**{paper.get('title', 'Untitled paper')}**")
                st.caption(
                    " · ".join(
                        str(x) for x in [
                            ", ".join((paper.get("authors") or [])[:3]) or "Uploaded PDF",
                            paper.get("published"),
                            f"{paper.get('chunks', 0)} passages",
                            f"{paper.get('num_pages')} pages" if paper.get("num_pages") else None,
                        ] if x
                    )
                )
                if paper_id:
                    checked = st.checkbox(
                        "Select for workspace",
                        value=paper_id in st.session_state.library_selected,
                        key=f"library_{paper_id}",
                    )
                    if checked:
                        st.session_state.library_selected.add(paper_id)
                    else:
                        st.session_state.library_selected.discard(paper_id)
                remove_col, link_col = st.columns([1, 4])
                if remove_col.button("Remove", key=f"remove_{paper.get('id')}"):
                    try:
                        api(f"/papers/{paper.get('id')}", method="DELETE")
                        st.session_state.library = [
                            p for p in st.session_state.library if p.get("id") != paper.get("id")
                        ]
                        st.session_state.library_selected.discard(paper_id)
                        st.success("Paper removed.")
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(str(exc))
                if paper_id:
                    link_col.markdown(f"[Open on arXiv]({paper_url(paper_id)})")

        if st.button("Work with selected library papers", type="primary", disabled=not st.session_state.library_selected):
            picks = [p for p in st.session_state.library if p.get("arxiv_id") in st.session_state.library_selected]
            st.session_state.workspace = [
                {"arxiv_id": p["arxiv_id"], "title": p.get("title", p["arxiv_id"]), "status": "ready", "message": ""}
                for p in picks
            ]
            st.session_state.outputs = {}
            st.success("Workspace updated. Open the Workspace tab.")

with workspace_tab:
    st.header("Research workspace")
    if not st.session_state.workspace:
        st.info("No papers in the workspace yet. Search for papers or choose items from your library.")
    else:
        if st.session_state.job_id:
            if st.button("Refresh job status", type="primary"):
                try:
                    refresh_job_status()
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
        ready_count = sum(1 for p in st.session_state.workspace if p.get("status") == "ready")
        st.caption(f"{ready_count} of {len(st.session_state.workspace)} papers ready for analysis.")
        for i, paper in enumerate(st.session_state.workspace):
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"**{paper.get('title', paper.get('arxiv_id', 'Paper'))}**")
                status = STATUS_LABELS.get(paper.get("status", "queued"), paper.get("status", "Unknown"))
                c1.caption(status + (f": {paper['message']}" if paper.get("message") else ""))
                if c2.button("Remove", key=f"workspace_remove_{paper.get('arxiv_id')}_{i}"):
                    st.session_state.workspace.pop(i)
                    st.rerun()

        ready_ids = selected_workspace_ids()
        if not ready_ids:
            st.warning("Wait for ingestion to finish, then refresh job status. Only ready papers can be analyzed.")
        else:
            st.divider()
            st.subheader("Analysis tools")
            tool = st.selectbox(
                "Choose a task",
                ["Summary", "Literature review", "Compare methods", "Find conflicts", "Check a claim", "Search passages"],
                key="workspace_tool",
            )
            question = st.text_area(
                "Question or topic",
                placeholder={
                    "Summary": "A detailed structured summary is generated for one paper.",
                    "Literature review": "How do recent papers reduce hallucination?",
                    "Compare methods": "Compare training data and evaluation metrics",
                    "Find conflicts": "Effect of model size on accuracy",
                    "Check a claim": "Method X improves F1 by 12.5% over the baseline.",
                    "Search passages": "How is faithfulness measured?",
                }[tool],
                height=100,
            )
            selected_arxiv = st.selectbox(
                "Paper",
                ready_ids,
                format_func=lambda arxiv_id: next(
                    (p.get("title", arxiv_id) for p in st.session_state.workspace if p.get("arxiv_id") == arxiv_id),
                    arxiv_id,
                ),
                key="analysis_paper",
            ) if tool in ("Summary", "Check a claim") else None

            if tool == "Find conflicts" and len(ready_ids) < 2:
                st.warning("Conflict analysis requires at least two ready papers.")

            if st.button("Run analysis", type="primary", disabled=(tool == "Find conflicts" and len(ready_ids) < 2)):
                if tool != "Summary" and len((question or "").strip()) < 5:
                    st.warning("Enter a question or topic of at least five characters.")
                else:
                    try:
                        top_k = min(30, max(6, len(ready_ids) * 4))
                        with st.spinner("Analyzing papers. This may take a minute…"):
                            if tool == "Summary":
                                payload = api(
                                    "/synthesize/review",
                                    method="POST",
                                    body={
                                        "question": (
                                            "Create a detailed, structured summary of this single paper. Cover the "
                                            "research problem and motivation, objectives and contributions, proposed "
                                            "method and technical details, datasets and experimental setup including "
                                            "baselines, main quantitative and qualitative results, limitations and "
                                            "caveats acknowledged by the authors, and a concise takeaway. Ground "
                                            "claims in excerpts; if evidence is unavailable, say it is not described."
                                        ),
                                        "top_k": 10,
                                        "verify": True,
                                        "multi_query": False,
                                        "arxiv_ids": [selected_arxiv],
                                    },
                                )
                            elif tool == "Literature review":
                                payload = api(
                                    "/synthesize/review", method="POST",
                                    body={"question": question, "top_k": top_k, "verify": True, "arxiv_ids": ready_ids},
                                )
                            elif tool == "Compare methods":
                                payload = api(
                                    "/synthesize/compare", method="POST",
                                    body={"question": question, "top_k": top_k, "arxiv_ids": ready_ids},
                                )
                            elif tool == "Find conflicts":
                                payload = api(
                                    "/synthesize/conflicts", method="POST",
                                    body={"question": question, "top_k": max(8, min(40, len(ready_ids) * 5)), "arxiv_ids": ready_ids},
                                )
                            elif tool == "Check a claim":
                                payload = api(
                                    "/synthesize/verify", method="POST",
                                    body={"claim": question, "arxiv_id": selected_arxiv, "top_k": 6},
                                )
                            else:
                                payload = api(
                                    "/search", method="POST",
                                    body={"query": question, "top_k": 8, "mode": "hybrid", "rerank": True, "arxiv_ids": ready_ids},
                                )
                        st.session_state.outputs[tool] = payload
                    except RuntimeError as exc:
                        st.error(str(exc))

            if tool in st.session_state.outputs:
                st.subheader("Result")
                render_json_result(st.session_state.outputs[tool])
                export_text = markdown_export(tool, st.session_state.outputs[tool])
                col_a, col_b = st.columns(2)
                col_a.download_button(
                    "Download Markdown",
                    data=export_text,
                    file_name=f"researchpilot-{tool.lower().replace(' ', '-')}.md",
                    mime="text/markdown",
                )
                if st.session_state.workspace:
                    col_b.download_button(
                        "Export BibTeX",
                        data=bibtex_export(),
                        file_name="researchpilot-references.bib",
                        mime="application/x-bibtex",
                    )

st.divider()
st.caption("ResearchPilot Streamlit frontend · API-backed paper search and evidence-grounded synthesis")
