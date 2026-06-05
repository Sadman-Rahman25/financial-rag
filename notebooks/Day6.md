# Day 6 — BGE Reranker + Metadata Pre-Filtering

## Decisions
- **Reranker model**: BAAI/bge-reranker-base (1.11 GB), NOT bge-reranker-v2-m3 (2.27 GB).
  Swapped due to 8 GB RAM constraint. Documented quality tradeoff for revisit if needed.
- **Loading**: fp16 via `model_kwargs={"torch_dtype": torch.float16}`. Halves model memory.
- **Two-phase loading**: rerank/compare modes explicitly free Retriever (releasing embedder)
  before loading reranker. Solves OOM cliff on 8 GB machines.
- **Reranker candidates**: top 20 from hybrid → cross-encoder rescore → final top-`limit`.
- **Pre-filtering**: regex/keyword based, NOT LLM. LLM query parsing deferred to Day 9-10.
- **Pre-filter activation**: opt-in via `--auto-filter` flag. Default preserves Day 5 behavior.

## New modules
- `src/rerank.py` — Reranker class wrapping bge-reranker-base in fp16
- `src/retrieve.py` — extended with `extract_filters()`, `--auto-filter` CLI flag,
  `--mode rerank`, `--mode compare` (now 4 modes), two-phase CLI orchestration

## Build stats
- Reranker first download: 1.11 GB at ~10 MB/s ≈ 2 min
- Reranker cached load (fp16): ~8-14s
- Rerank inference: ~3-15s for 20 candidates on CPU fp16

## Issues hit + solutions

### Silent OOM on reranker load (encountered 3 times)
- Trigger: embedder (1.5 GB) + reranker (1.1 GB) > 1.9 GB available RAM on 8 GB machine
- Tried fp16 via deprecated `automodel_args` → never engaged, OOMed
- Tried fp16 via current `model_kwargs` → engaged but still OOMed (embedder still resident)
- **Final fix: two-phase loading.** CLI explicitly does `del retriever` + `gc.collect()`
  before loading reranker. Peak memory is now one model at a time.
- Restructured: removed `hybrid_rerank_search` from Retriever class; CLI orchestrates the phases.

### Model size correction
- Initially documented bge-reranker-base as ~278 MB; actual disk size is 1.11 GB.

## Validation queries

### Q1: `"data center business" --mode compare --limit 10` (headline test)
- Ticker distribution: Dense 8/2/0/0/0 (NVDA/MSFT/META/AAPL/TSLA),
  BM25 10/0/0/0/0, Hybrid 8/1/1/0/0, Rerank 8/2/0/0/0
- **Reranker did NOT solve diversity.** Still 8/10 NVDA in top 10.
- Reranker dramatically reordered (chunks from hybrid #5-7 became rerank #1-3)
  but kept ticker mix similar.
- Failure mode: reranker preferred risk-factor boilerplate (Item 1A) over direct business
  descriptions (Item 1 Business). Two near-identical FY25/FY26 risk factor chunks
  both scored 0.989 — duplicates not deduplicated.
- Score saturation: top 9 chunks scored 0.80-0.99, indicating model uncertainty rather
  than clear winners.

### Q2: `"What was Apple's iPhone revenue in 2024?" --mode rerank --auto-filter`
- Auto-detected: ticker=AAPL, fiscal_year=2024
- All 5 hits are AAPL FY2024 (filter worked)
- Top hits: Note 3 EPS, Note 2 Revenue table (literally has "iPhone"),
  Products & Services Performance, Item 7 iPhone sales prose
- Score range 0.94 to 0.44 — clear cliff
- **Pre-filter cleanly solved the META false-positive problem** observed earlier in
  unfiltered rerank ("Apple iPhone revenue" returned a META Segment table at rank #4).

### Q3: `"Meta's data center infrastructure" --mode rerank --auto-filter`
- Auto-detected: ticker=META
- All 5 hits are META (filter worked)
- **But top hits are about advertising, Meta AI, Reality Labs — not data centers.**
- **Honest finding: META's 10-K doesn't densely document data center infrastructure.**
  Filtering narrows the pool; it can't surface content that isn't there.
- This CONFIRMS the corpus-density diagnosis from Q1: NVDA-dominant results on broad
  tech queries reflect real corpus density, not retrieval bias.

## extract_filters() rules
- Ticker keywords: apple/aapl, microsoft/msft, nvidia/nvda, tesla/tsla, meta/facebook
- Year regex: `(?:^|\W)(?:fy)?(202\d)\b` matches 2020-2029, optional FY prefix
- Conservative: multiple tickers detected → no auto-filter (likely comparison query)
- Explicit `--ticker`/`--fiscal-year` flags always override auto-detect

## Accepted limitations + carried forward
1. **NVDA-dominance on broad tech queries is a CORPUS DENSITY fact**, not a retrieval bug.
   Rerank can re-order but can't manufacture diversity. Fixes for later: query
   reformulation (Day 10) or MMR-style diversification reranking (Day 12 candidate).
2. **Reranker systematically prefers prose over tables** — observed on "Apple iPhone revenue"
   (prose chunks at 0.99+, tables at 0.20-0.27). Important for Day 7 eval design;
   hybrid+BM25 keeps tables in the candidate pool.
3. **Near-duplicate boilerplate text** (FY24 vs FY25 risk factors) both rank high — no
   deduplication. Candidate fix: Day 12.
4. **Pre-filter only handles ticker + fiscal_year** — no section, no chunk_type, no complex
   date logic. Day 9-10 LLM-based query parsing is the cleaner long-term solution.
5. **CrossEncoder fp16 on CPU**: quality preserved (scores differ by <0.002 from fp32),
   inference slightly slower (no hardware fp16 acceleration on most CPUs). Acceptable.

## Next: Day 7 — Build 25-30 question eval set
Eval questions should test all the failure modes above:
- Table-heavy retrieval (e.g. "iPhone revenue breakdown by year")
- Boilerplate near-duplicates (multi-year risk factor topics)
- Cross-company comparisons (multi-ticker, no auto-filter)
- Specific entity queries (auto-filter friendly)
- Paraphrase queries (BM25-resistant)
- Numerical queries (challenging for dense + BM25 both)