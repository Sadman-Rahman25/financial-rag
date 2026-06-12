"""
Streamlit UI for Financial RAG.

Day 14 redesign: financial-slate theme, hero band, provenance-forward
chunk cards. All Stage 6 logic preserved — this is a presentation-layer swap.

Run from project root:
    streamlit run app.py
"""

import html
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


# ===========================================================
# THEME
# ===========================================================

def inject_theme() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ink: #0B1929;
    --panel: #0F2942;
    --panel-2: #13314f;
    --line: #1E3A5C;
    --text: #E8EDF2;
    --text-dim: #8FA8C4;
    --blue: #5B8DEF;
    --teal: #3FB68B;
    --amber: #E4B363;
}

.stApp {
    background:
        radial-gradient(1100px 520px at 82% -12%, rgba(91,141,239,0.08), transparent),
        var(--ink);
}
#MainMenu, footer { visibility: hidden; }
/* Keep header transparent but functional so the sidebar toggle survives */
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }
[data-testid="collapsedControl"] { visibility: visible !important; }

html, body, [class*="css"], .stMarkdown, p, span, div, label {
    font-family: 'Inter', sans-serif;
    color: var(--text);
}

/* Main container width */
.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* ---- Hero ---- */
.hero-wrap {
    border: 1px solid var(--line);
    border-radius: 16px;
    background: linear-gradient(160deg, var(--panel) 0%, var(--ink) 100%);
    padding: 30px 34px; margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: ""; position: absolute; inset: 0;
    background-image: repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(91,141,239,0.04) 39px, rgba(91,141,239,0.04) 40px);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.22em; text-transform: uppercase;
    color: var(--blue); margin-bottom: 12px;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700;
    font-size: 44px; line-height: 1.04; margin: 0 0 12px 0;
    color: var(--text); letter-spacing: -0.02em;
}
.hero-title .accent { color: var(--blue); }
.hero-sub { font-size: 15px; color: var(--text-dim); max-width: 700px; line-height: 1.55; }
.hero-stats { display: flex; gap: 30px; margin-top: 20px; flex-wrap: wrap; }
.stat { display: flex; flex-direction: column; }
.stat .num { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 23px; color: var(--text); }
.stat .lbl { font-family: 'JetBrains Mono', monospace; font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--text-dim); margin-top: 2px; }
.stat .num .teal { color: var(--teal); }

/* ---- Answer / refusal cards ---- */
.answer-card {
    border: 1px solid var(--line); border-left: 3px solid var(--blue);
    border-radius: 12px; background: var(--panel);
    padding: 22px 26px; margin: 4px 0 16px 0;
}
.answer-label, .refusal-label {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 10px;
}
.answer-label { color: var(--blue); }
.answer-body { font-size: 16px; line-height: 1.6; color: var(--text); }
.cite {
    font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--teal);
    background: rgba(63,182,139,0.1); padding: 1px 6px; border-radius: 4px; white-space: nowrap;
}
.refusal-card {
    border: 1px solid rgba(228,179,99,0.4); border-left: 3px solid var(--amber);
    border-radius: 12px; background: rgba(228,179,99,0.06);
    padding: 22px 26px; margin: 4px 0 16px 0;
}
.refusal-label { color: var(--amber); }
.refusal-body { font-size: 15px; line-height: 1.6; color: var(--text); }

/* ---- Decomposition banner ---- */
.decomp {
    display: flex; align-items: center; gap: 12px;
    border: 1px solid var(--line); border-radius: 10px;
    background: var(--panel-2); padding: 12px 18px; margin: 4px 0 16px 0;
}
.decomp .tag {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--blue);
    border: 1px solid var(--blue); padding: 3px 9px; border-radius: 20px; white-space: nowrap;
}
.decomp .ents { font-size: 14px; color: var(--text-dim); }
.decomp .ents b { color: var(--text); font-weight: 600; }

/* ---- Chunk cards ---- */
.chunk {
    border: 1px solid var(--line); border-radius: 10px;
    background: var(--panel); padding: 14px 16px; margin-bottom: 10px;
}
.chunk.cited {
    border-left: 3px solid var(--teal);
    background: linear-gradient(90deg, rgba(63,182,139,0.08), var(--panel) 42%);
}
.chunk-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
.chunk-id { font-family: 'JetBrains Mono', monospace; font-size: 12px; font-weight: 500; color: var(--text); }
.pill {
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    letter-spacing: 0.08em; padding: 2px 7px; border-radius: 20px; text-transform: uppercase;
}
.pill.ticker { background: rgba(91,141,239,0.15); color: var(--blue); }
.pill.year { background: rgba(143,168,196,0.12); color: var(--text-dim); }
.pill.cited { background: var(--teal); color: var(--ink); font-weight: 600; }
.pill.table { background: rgba(228,179,99,0.15); color: var(--amber); }
.chunk-sect { font-size: 12px; color: var(--text-dim); margin-bottom: 6px; }
.chunk-text { font-size: 13px; color: var(--text-dim); line-height: 1.5; }

/* ---- Section eyebrow + token meter ---- */
.eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-dim);
    margin: 18px 0 12px 0;
}
.tokmeter {
    display: inline-flex; gap: 16px; font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--text-dim); margin-top: 10px;
}
.tokmeter b { color: var(--text); }

/* ---- Streamlit widget restyling ---- */
.stButton button {
    background: var(--panel-2) !important; border: 1px solid var(--line) !important;
    color: var(--text) !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-size: 13px !important;
    text-align: left !important; transition: all 0.15s ease !important;
    min-height: 52px !important;
}
.stButton button:hover { border-color: var(--blue) !important; background: var(--panel) !important; }
.stButton button[kind="primary"] {
    background: var(--blue) !important; border-color: var(--blue) !important;
    color: var(--ink) !important; font-weight: 600 !important; text-align: center !important;
}
.stButton button[kind="primary"]:hover { background: #6f9bf2 !important; }

.stTextArea textarea {
    background: var(--panel) !important; border: 1px solid var(--line) !important;
    color: var(--text) !important; border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus { border-color: var(--blue) !important; box-shadow: 0 0 0 1px var(--blue) !important; }

section[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--panel-2) !important; border-color: var(--line) !important;
    cursor: pointer !important;
}
.stSelectbox div[data-baseweb="select"] * { cursor: pointer !important; }

[data-testid="stExpander"] {
    border: 1px solid var(--line) !important; border-radius: 10px !important;
    background: transparent !important;
}
[data-testid="stExpander"] summary { font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; color: var(--text-dim) !important; }

div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ===========================================================
# DATA LOADING (cached)
# ===========================================================

@st.cache_data
def get_filter_options() -> tuple[list[str], list[int]]:
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


@st.cache_data
def get_chunks_by_id() -> dict[int, dict]:
    result: dict[int, dict] = {}
    with open("data/chunks.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                result[i] = json.loads(line)
    return result


@st.cache_resource(
    show_spinner="Loading retrieval + embedding models (first time only, ~30-60s)..."
)
def get_generator() -> Generator:
    return Generator()


# ===========================================================
# TABLE PARSING (unchanged from Day 12)
# ===========================================================

_SEP_PATTERN = re.compile(r"(?:\|\s*-{2,}\s*){2,}\|")


def parse_pipe_table(text: str) -> list[list[str]] | None:
    sep_match = _SEP_PATTERN.search(text)
    if not sep_match:
        return None
    sep_str = sep_match.group(0)
    n_cols = sep_str.count("|") - 1
    if n_cols < 2:
        return None
    pre = text[: sep_match.start()].strip().strip("|").strip()
    post = text[sep_match.end():].strip().strip("|").strip()
    if not pre or not post:
        return None
    header_cells = [c.strip() for c in pre.split("|")]
    data_cells = [c.strip() for c in post.split("|")]
    header = header_cells[-n_cols:] if len(header_cells) >= n_cols else header_cells
    if len(header) < n_cols:
        header += [""] * (n_cols - len(header))
    rows: list[list[str]] = []
    for i in range(0, len(data_cells), n_cols):
        row = data_cells[i:i + n_cols]
        if len(row) < n_cols:
            row += [""] * (n_cols - len(row))
        if all(c in ("", "--", "---") for c in row):
            continue
        rows.append(row)
    if not rows:
        return None
    return [header] + rows


# ===========================================================
# HTML RENDERERS
# ===========================================================

def _esc(s: str) -> str:
    return html.escape(str(s))


def linkify_citations(answer_text: str) -> str:
    """Turn [chunk_64] references in answer text into styled citation pills."""
    escaped = _esc(answer_text)
    return re.sub(
        r"\[chunk_(\d+)\]",
        r'<span class="cite">chunk_\1</span>',
        escaped,
    )


def chunk_card_html(cid: int, chunks_by_id: dict, citations: list[int]) -> str:
    """Build the HTML for one chunk card (no table — tables render separately)."""
    chunk = chunks_by_id.get(cid)
    if not chunk:
        return f'<div class="chunk"><div class="chunk-id">chunk_{cid} not found</div></div>'

    meta = chunk.get("metadata") or {}
    ticker = meta.get("ticker", "?")
    fy = meta.get("fiscal_year", "?")
    section = meta.get("section", "?")
    subsection = meta.get("subsection") or ""
    chunk_type = (meta.get("chunk_type") or "").lower()
    is_cited = cid in citations

    pills = [
        f'<span class="pill ticker">{_esc(ticker)}</span>',
        f'<span class="pill year">FY{_esc(fy)}</span>',
    ]
    if chunk_type == "table":
        pills.append('<span class="pill table">table</span>')
    if is_cited:
        pills.append('<span class="pill cited">cited</span>')

    sect_line = _esc(section)
    if subsection:
        sect_line += f" — {_esc(subsection)}"

    text = chunk.get("text", "")
    cls = "chunk cited" if is_cited else "chunk"

    if chunk_type == "table":
        body = '<div class="chunk-text">Table — rendered below.</div>'
    else:
        preview = text[:300] + ("..." if len(text) > 300 else "")
        body = f'<div class="chunk-text">{_esc(preview)}</div>'

    return f"""<div class="{cls}">
  <div class="chunk-head"><span class="chunk-id">chunk_{cid}</span>{''.join(pills)}</div>
  <div class="chunk-sect">{sect_line}</div>
  {body}
</div>"""


def render_chunk(cid: int, chunks_by_id: dict, citations: list[int]) -> None:
    """Render a chunk card; if it's a table, attempt a DataFrame below it."""
    st.markdown(chunk_card_html(cid, chunks_by_id, citations), unsafe_allow_html=True)

    chunk = chunks_by_id.get(cid)
    if not chunk:
        return
    meta = chunk.get("metadata") or {}
    if (meta.get("chunk_type") or "").lower() == "table":
        rows = parse_pipe_table(chunk.get("text", ""))
        if rows and len(rows) >= 2:
            try:
                df = pd.DataFrame(rows[1:], columns=rows[0])
                st.dataframe(df, use_container_width=True, hide_index=True)
                return
            except Exception:
                pass
        st.code(chunk.get("text", ""), language=None)


def use_example(example_text: str) -> None:
    st.session_state.question_input = example_text


# ===========================================================
# PAGE
# ===========================================================

st.set_page_config(page_title="Financial RAG", layout="wide", page_icon="◆")
inject_theme()

chunks_by_id = get_chunks_by_id()
available_tickers, available_years = get_filter_options()

# ---- Sidebar ----
with st.sidebar:
    st.markdown('<div class="eyebrow" style="margin-top:0">Filters</div>', unsafe_allow_html=True)
    ticker_filter = st.selectbox(
        "Ticker", options=["All"] + available_tickers, index=0,
        help="Restrict retrieval to a single company.",
    )
    year_filter = st.selectbox(
        "Fiscal year", options=["All"] + [str(y) for y in available_years], index=0,
        help="Restrict retrieval to a single fiscal year.",
    )
    st.caption(
        "Filters narrow which chunks the retriever considers. "
        "Cross-company questions work best with no ticker filter."
    )

# ---- Hero ----
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-eyebrow">SEC 10-K · Retrieval-Augmented QA</div>
  <div class="hero-title">Financial <span class="accent">RAG</span></div>
  <div class="hero-sub">Ask questions about SEC 10-K filings for Apple, Microsoft, NVIDIA, Tesla, and Meta. Hybrid retrieval with per-company decomposition and grounded, cited answers.</div>
  <div class="hero-stats">
    <div class="stat"><span class="num">{len(chunks_by_id):,}</span><span class="lbl">Indexed chunks</span></div>
    <div class="stat"><span class="num">{len(available_tickers)}</span><span class="lbl">Companies</span></div>
    <div class="stat"><span class="num"><span class="teal">28/28</span></span><span class="lbl">Eval coverage</span></div>
    <div class="stat"><span class="num">$0</span><span class="lbl">Infra cost</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---- Examples ----
st.markdown('<div class="eyebrow">Try an example</div>', unsafe_allow_html=True)
cols = st.columns(2)
for i, ex in enumerate(EXAMPLE_QUESTIONS):
    with cols[i % 2]:
        st.button(ex, key=f"example_{i}", on_click=use_example, args=(ex,), use_container_width=True)

# ---- Input ----
question = st.text_area(
    "Your question",
    placeholder="e.g., What was Apple's iPhone revenue in FY2024?",
    height=80, key="question_input", label_visibility="collapsed",
)
submit = st.button("Ask", type="primary")

# ---- Submit handler ----
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

        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        citations = result.get("citations", []) or []

        # Answer or refusal
        if result.get("refused"):
            body = _esc(result.get("answer", "")) or "No answer — out of corpus or insufficient context."
            st.markdown(f"""
<div class="refusal-card">
  <div class="refusal-label">Refused</div>
  <div class="refusal-body">{body}</div>
</div>""", unsafe_allow_html=True)
        else:
            answer_html = linkify_citations(result.get("answer", "(no answer returned)"))
            st.markdown(f"""
<div class="answer-card">
  <div class="answer-label">Answer</div>
  <div class="answer-body">{answer_html}</div>
</div>""", unsafe_allow_html=True)

        # Decomposition banner
        if result.get("decomposed"):
            entities = result.get("entities", [])
            ent_str = ", ".join(f"<b>{_esc(e)}</b>" for e in entities)
            st.markdown(f"""
<div class="decomp">
  <span class="tag">Decomposed</span>
  <span class="ents">Query split into {len(entities)} entities: {ent_str} — balanced retrieval per company</span>
</div>""", unsafe_allow_html=True)

        # Retrieved evidence
        retrieved_ids = result.get("retrieved_chunk_ids", []) or []
        entity_map = result.get("entity_chunk_map") or {}

        if result.get("decomposed") and entity_map:
            st.markdown('<div class="eyebrow">Retrieved evidence · by entity</div>', unsafe_allow_html=True)
            for entity, eids in entity_map.items():
                with st.expander(f"{entity} — {len(eids)} chunks"):
                    for cid in eids:
                        render_chunk(cid, chunks_by_id, citations)
        else:
            st.markdown(f'<div class="eyebrow">Retrieved evidence · {len(retrieved_ids)} chunks</div>', unsafe_allow_html=True)
            with st.expander("Show retrieved chunks"):
                for cid in retrieved_ids:
                    render_chunk(cid, chunks_by_id, citations)

        # Token meter
        usage = result.get("usage") or {}
        p = usage.get("prompt_tokens", 0)
        c = usage.get("completion_tokens", 0)
        t = usage.get("total_tokens", 0)
        st.markdown(
            f'<div class="tokmeter">Tokens · <b>{p:,}</b> prompt · <b>{c:,}</b> completion · <b>{t:,}</b> total</div>',
            unsafe_allow_html=True,
        )