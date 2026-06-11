# Day 12 — Streamlit UI

**Date:** 2026-06-11
**Status:** Shipped. UI complete. Day 13 next.

## Decisions taken at start of day

1. **Deployment target: HuggingFace Spaces (free, 16GB RAM).**
2. **Defer q18 retrieval fix** — graceful-degradation story is stronger than a re-engineered answer.
3. **Minimum scope Streamlit** — no clickable citations, no token meter overlays, no side-by-side decomposed view.

## What shipped

- `app.py` at project root (~230 lines) — imports existing `Generator` directly, no logic duplication.
- `requirements.txt` — full `pip freeze` snapshot for HF Spaces reproducibility.

### UI features (Stages 1–6)

- Question input + Submit
- Sidebar filters: ticker (5 companies + All) and fiscal year (All + dynamically-detected years from `chunks.jsonl`).
- Generator loaded as singleton via `@st.cache_resource` — models load ONCE per Streamlit process. ~30-60s cold start on first query, fast thereafter.
- Answer area with refused vs answered visual differentiation.
- Cross-company decomposition indicator: blue info box showing detected entities ("NVDA, MSFT").
- Retrieved chunks panel:
  - Single-entity: flat list of 10 chunks in one expander.
  - Cross-company: grouped by entity (one expander per company).
  - Each chunk card: ticker, fiscal year, section, subsection, `[CITED]` marker if used, `[TABLE]` marker for table chunks.
- 4 example question buttons (2-column layout) — populate input via `st.session_state` callback.
- Token cost caption (prompt + completion + total) — reads from nested `result["usage"]` dict.
- Tech-stack footer line.

## Architectural choices that matter for Day 14 deploy

1. **`@st.cache_resource` over per-query model load.** Single biggest deployment-readiness decision. Without it, every Streamlit rerun reloads the 1.34GB BGE-large model.
2. **`app.py` at project root.** Clean `from src.generate import Generator` imports, matches HF Spaces convention.
3. **Filter options derived from corpus.** Adding a company or fiscal year doesn't require UI code changes.
4. **Single-select filters, not multi-select.** Matches the Generator's `filters` param contract.
5. **No reranker in UI.** Day 8 showed it hurts MRR on average.

## The interesting story: table rendering → chunker bug → deferred fix

**Observation:** table chunks rendered as ugly pipe-spaghetti.

Tried three approaches:

1. **Markdown render (`st.markdown(text)`)** — failed because chunks are stored as single-line pipe-text with no row newlines.
2. **Manual parser → `st.dataframe`** — parser couldn't reliably reconstruct rows because em-dash fillers are indistinguishable from in-row empty cells.
3. **Chunker fix at the source** — found the bug in `src/chunk.py` `fix_empty_table_cells`: regex `\|\s+\|` matches across newlines, consuming `|\n|` boundaries between table rows.

### The fix that wasn't

One-character change: `\s+` → `[^\S\n]+`. Tested on NVDA_2025: tables now multi-line, working as intended.

**But** chunk count drifted: 299 → 304 in NVDA_2025 alone. The old chunker wasn't just flattening individual tables — it was also collapsing blank lines BETWEEN adjacent tables, merging them into single chunks. The fix correctly separates them. Chunk IDs would drift, breaking:

- `data/eval/questions.jsonl` reference_chunk_ids (especially q18, the carefully-corrected one)
- `results/day8_retrieval_eval.json` Hit/MRR/NDCG metrics
- `results/day9_generation.json` citations
- The q18 portfolio narrative

Estimated re-validation cost: 3-5 hours + ~50k Groq tokens. Not worth it for a UI cosmetic improvement.

### Decision: abort

Reverted via `git checkout pre-chunker-fix -- src/chunk.py`. Backups retained but gitignored.

UI now renders table chunks with `[TABLE]` tag + monospace fallback (`st.code`). README documents the limitation as a v2 enhancement.

## Tokens used today

~50,000 Groq tokens across ~10–12 test queries. Within budget (100k TPD).

## New portfolio stories from today

**Story 4: `@st.cache_resource` as the deployment-readiness decision.** The difference between viable HF Spaces deployment and 30s-per-query latency is recognizing that singleton model loading is the load-bearing UI decision.

**Story 5: Diagnosed-but-deferred chunker bug.** Found a one-character regex bug in `chunk.py` that's been silently merging adjacent tables since Day 3. The fix is trivial. Deploying it correctly is not. Choosing to ship Day 12 and defer the fix is the right scoping call — over-eager refactoring would have blown up validated results for a UI polish.

**Story 6: ENTITY MATCH discipline held under retrieval bias.** Tesla revenue question with all filters off pulled 6 NVDA chunks and 4 TSLA chunks (NVDA's "Automotive And AI" subsection scored high). The LLM cited only the TSLA chunk and produced the correct answer ($97.69B). Day 9's ENTITY MATCH rule prevented hallucination despite 60% noise.

## What's next

**Day 13 — README + architecture diagram (3-4 hours).** The portfolio front page. Sections: TL;DR, problem, pitch, demo link, architecture diagram, tech stack, key results table, the six portfolio stories, how to run locally, limitations & v2 work.

**Day 14 — HF Spaces deploy + demo video (4-6 hours).** Test cold-start, manage Groq key at the Space level, record 2-3 min demo covering single-entity, cross-company, refusal.

## Carry-forward items for v2 (post-portfolio launch)

1. **Chunker fix:** preserve table boundaries and newlines; regenerate `chunks.jsonl`; re-embed; re-validate Day 8/9/10 metrics on the corrected corpus.
2. **q18 retrieval fix:** merge intro+table+footnote chunks for table sections so chunk 2550 becomes retrievable.
3. **LLM-augmented eval expansion 28 → 100+** for stronger metrics.