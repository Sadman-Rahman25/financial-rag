# Day 4 — BGE Embedding + Qdrant Indexing

## Decisions
- **Embedding model**: BAAI/bge-large-en-v1.5 (1024-dim, normalized)
- **Vector DB**: Qdrant local file mode at `./qdrant_data/`
- **Distance metric**: Cosine
- **Point IDs**: sequential integers (0..3171), reproducible per chunks.jsonl
- **Payload**: full chunk metadata + full chunk text
- **Big tables**: BGE auto-truncates inputs at 512 tokens for embedding; full untruncated text preserved in payload
- **Batch sizes**: EMBED_BATCH=1, UPSERT_BATCH=32 (tuned for 8 GB RAM machine)

## Build stats
- Total chunks indexed: 3,172
- Wall time: 4288.4 s (~71 min)
- Final rate: ~0.7 chunks/sec
- Hardware: 8 GB RAM, Intel Iris Xe iGPU, CPU at 1.55–1.68 GHz

## Issues hit
- **OOM on first build attempt** at EMBED_BATCH=64. 8 GB RAM is tight; only ~1.9 GB available after closing apps. Resolved by dropping internal encode batch to 1 and capping OMP/MKL/torch threads at 4.
- **Payload indexes are no-ops in Qdrant local mode** (warning printed during --setup). Confirmed via Query 2 that filters still work correctly — Qdrant falls back to full scan, which is fast enough at 3,172 points. Documented for revisit if scale grows.

## Validation queries

### Q1: "Apple iPhone revenue" (no filter)
- All 5 top results are AAPL
- Scores 0.7266–0.7435
- Hits include Note 2 Revenue table (iPhone breakdown), Products/Services Performance tables, Note 3 EPS revenue recognition prose
- Verdict: strong

### Q2: "data center" with ticker=NVDA, fiscal_year=2026
- All 5 results are NVDA FY2026 — filter works correctly
- Scores 0.6039–0.6384 (lower because smaller candidate pool)
- Hits cover NVDA's Item 1 Business (data center infrastructure), Item 2 Properties (data center sites), Item 7 MD&A (energy capacity)
- Verdict: filter functional, content relevant

### Q3: "data center business" cross-company, limit=10
- Distribution: NVDA 8, MSFT 2, META 0, AAPL 0, TSLA 0
- NVDA dominance is corpus-driven (their 10-K is dense with data-center language), not a retrieval bug
- Will revisit if Day 8 evals flag diversity as an issue
- Day 5 hybrid (BM25) and Day 6 reranking are the natural fixes

## Accepted limitations (carried into Day 5+)
1. NVDA-dominant retrieval on broad tech queries (corpus density issue, not retrieval issue)
2. Occasional MSFT chunks with empty `section` field (Day 3 accepted; MSFT structure is irregular)
3. Subsection labels sometimes mis-attributed across section boundaries (Day 3 accepted; LlamaParse Item-prefix quirks)
4. Big-table truncation at 512 tokens for embedding; full text in payload (Day 3 accepted; revisit Day 12)

## Next: Day 5 — BM25 + hybrid retrieval with RRF