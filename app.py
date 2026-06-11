"""
Streamlit UI for Financial RAG.

Day 12 Stage 6 + B3: table chunks parsed and rendered as DataFrames
(falls back to monospace code block if parsing fails).

Run from project root:
    streamlit run app.py
"""

import json
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.generate import Generator


load_dotenv()


EXAMPLE_QUESTIONS = [
    "What was Apple's iPhone net sales in fiscal year 2024?",
    "What was the data center revenue for NVIDIA and Microsoft?",
    "What does deferred revenue mean in Apple's 10-K?",
    "How did Tesla's research and development expense change year over year?",
]


# Separator pattern: at least 2 cells of dashes joined by pipes, e.g. |--|--|--|
_SEP_PATTERN = re.compile(r"(?:\|\s*-{2,}\s*){2,}\|")


def parse_pipe_table(text: str) -> list[list[str]] | None:
    """
    Parse a single-line pipe-delimited table into rows of cells.

    Returns a list-of-lists (rows of strings) where row 0 is the header,
    or None if the structure can't be confidently determined.
    """
    sep_match = _SEP_PATTERN.search(text)
    if not sep_match:
        return None

    sep_str = sep_match.group(0)
    # Each "|---" segment is one cell; subtract one for the trailing |
    n_cols = sep_str.count("|") - 1
    if n_cols < 2:
        return None

    pre = text[: sep_match.start()].strip().strip("|").strip()
    post = text[sep_match.end():].strip().strip("|").strip()

    if not pre or not post:
        return None

    header_cells = [c.strip() for c in pre.split("|")]
    data_cells = [c.strip() for c in post.split("|")]

    # Header: trim to n_cols (in case prefix like "[in millions]" added a cell)
    header = header_cells[-n_cols:] if len(header_cells) >= n_cols else header_cells
    if len(header) < n_cols:
        header += [""] * (n_cols - len(header))

    # Group data cells into rows of n_cols
    rows: list[list[str]] = []
    for i in range(0, len(data_cells), n_cols):
        row = data_cells[i:i + n_cols]
        if len(row) < n_cols:
            row += [""] * (n_cols - len(row))
        # Skip rows that are all empty or all dashes (separator noise)
        if all(c in ("", "--", "---") for c in row):
            continue
        rows.append(row)

    if not rows:
        return None

    return [header] + rows


def render_table_chunk(text: str) -> None:
    """Try to parse pipe-text as a DataFrame; fall back to monospace code block."""
    rows = parse_pipe_table(text)
    if rows and len(rows) >= 2:
        try:
            df = pd.DataFrame(rows[1:], columns=rows[0])
            st.dataframe(df, use_container_width=True, hide_index=True)
            return
        except Exception:
            pass

    # Fallback: monospace box so pipes at least line up visually
    st.code(text, language=None)


# ---------- Filter options derived from corpus ----------
@st.cache_data
def get_filter_options() -> tuple[list[str], list[int]]:
    """Read chunks.jsonl once and extract unique tickers + fiscal years."""
    tickers: set[str] = set()
    years: set[int] = set()
    with open("data/chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line).get("metadata") or {}
            if meta.get("ticker"):
                tickers.add(meta["ticker"])
            if meta.get("fiscal_year") is not None:
                years.add(int(meta["fiscal_year"]))
    return sorted(tickers), sorted(years)


# ---------- Chunks indexed by ID (= line number in chunks.jsonl) ----------
@st.cache_data
def get_chunks_by_id() -> dict[int, dict]:
    """Load all chunks into a dict keyed by line position (= chunk ID)."""
    result: dict[int, dict] = {}
    with open("data/chunks.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                result[i] = json.loads(line)
    return result


def render_chunk_card(
    cid: int,
    chunks_by_id: dict,
    citations: list[int],
    preview_chars: int = 300,
) -> None:
    """Render one chunk: metadata header + text/table-aware preview."""
    chunk = chunks_by_id.get(cid)
    if not chunk:
        st.warning(f"chunk_{cid} not found in chunks.jsonl")
        return

    meta = chunk.get("metadata") or {}
    ticker = meta.get("ticker", "?")
    fy = meta.get("fiscal_year", "?")
    section = meta.get("section", "?")
    subsection = meta.get("subsection") or ""
    chunk_type = (meta.get("chunk_type") or "").lower()

    parts = [f"**chunk_{cid}**", ticker, f"FY{fy}", section]
    if subsection:
        parts.append(subsection)
    header = " — ".join(parts)
    if chunk_type == "table":
        header += "  `[TABLE]`"
    if cid in citations:
        header += "  **[CITED]**"

    st.markdown(header)

    text = chunk.get("text", "")
    if chunk_type == "table":
        render_table_chunk(text)
    else:
        preview = text[:preview_chars] + ("..." if len(text) > preview_chars else "")
        st.caption(preview)


def use_example(example_text: str) -> None:
    """Callback: set the question input to the clicked example."""
    st.session_state.question_input = example_text


# ---------- Page config (must be the first Streamlit call) ----------
st.set_page_config(
    page_title="Financial RAG",
    layout="wide",
)


# ---------- Cached singleton: models load ONCE per Streamlit process ----------
@st.cache_resource(
    show_spinner="Loading retrieval + embedding models (first time only, ~30-60s)..."
)
def get_generator() -> Generator:
    """Generator holds BGE-large (~1.34GB), Qdrant client, BM25 index."""
    return Generator()


# ---------- Sidebar: ticker + fiscal-year filters ----------
with st.sidebar:
    st.header("Filters")

    available_tickers, available_years = get_filter_options()

    ticker_filter = st.selectbox(
        "Ticker",
        options=["All"] + available_tickers,
        index=0,
        help="Restrict retrieval to a single company.",
    )

    year_filter = st.selectbox(
        "Fiscal year",
        options=["All"] + [str(y) for y in available_years],
        index=0,
        help="Restrict retrieval to a single fiscal year.",
    )

    st.caption(
        "Filters narrow which chunks the retriever considers. "
        "Cross-company questions work best with no ticker filter."
    )


# ---------- Main area ----------
st.title("Financial RAG")

chunks_by_id = get_chunks_by_id()
st.caption(
    f"Hybrid retrieval over {len(chunks_by_id):,} chunks from SEC 10-K filings "
    f"for Apple, Microsoft, NVIDIA, Tesla, and Meta. "
    f"Cross-company questions decompose into per-entity retrieval."
)

st.markdown("**Try an example:**")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLE_QUESTIONS):
    with cols[i % 2]:
        st.button(
            ex,
            key=f"example_{i}",
            on_click=use_example,
            args=(ex,),
            use_container_width=True,
        )

question = st.text_area(
    "Your question",
    placeholder="e.g., What was Apple's iPhone revenue in FY2024?",
    height=80,
    key="question_input",
)

submit = st.button("Submit", type="primary")


# ---------- Submit handler ----------
if submit:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        gen = get_generator()

        filters: dict = {}
        if ticker_filter != "All":
            filters["ticker"] = ticker_filter
        if year_filter != "All":
            filters["fiscal_year"] = int(year_filter)

        with st.spinner("Retrieving + generating..."):
            try:
                result = gen.answer(question, filters=filters or None)
            except Exception as e:
                st.error(f"Generation failed: {type(e).__name__}: {e}")
                st.stop()

        st.divider()

        if result.get("refused"):
            st.warning("Refused to answer (out of corpus or insufficient context).")
            if result.get("answer"):
                st.write(result["answer"])
        else:
            st.subheader("Answer")
            st.markdown(result.get("answer", "(no answer returned)"))

        if result.get("decomposed"):
            entities = result.get("entities", [])
            st.info(
                f"Cross-company query — decomposed into "
                f"{len(entities)} entities: {', '.join(entities)}"
            )

        citations = result.get("citations", []) or []
        if citations:
            st.caption(f"Cited chunks: {citations}")

        retrieved_ids = result.get("retrieved_chunk_ids", []) or []
        entity_map = result.get("entity_chunk_map") or {}

        if result.get("decomposed") and entity_map:
            st.markdown("##### Retrieved chunks (by entity)")
            for entity, eids in entity_map.items():
                with st.expander(f"{entity} — {len(eids)} chunks"):
                    for cid in eids:
                        render_chunk_card(cid, chunks_by_id, citations)
                        st.divider()
        else:
            with st.expander(f"Retrieved chunks ({len(retrieved_ids)})"):
                for cid in retrieved_ids:
                    render_chunk_card(cid, chunks_by_id, citations)
                    st.divider()

        usage = result.get("usage") or {}
        prompt_tok = usage.get("prompt_tokens", 0)
        comp_tok = usage.get("completion_tokens", 0)
        total_tok = usage.get("total_tokens", 0)
        st.caption(
            f"Tokens: {prompt_tok} prompt + {comp_tok} completion = {total_tok} total"
        )


st.divider()
st.caption(
    "Hybrid retrieval: BGE-large-en-v1.5 + BM25 + RRF (k=60). "
    "Generation: Llama 3.3 70B via Groq. "
    "Vector store: Qdrant. "
    "Query decomposition: rule-based entity detection."
)