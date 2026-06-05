# Day 5 — BM25 + Hybrid Retrieval with RRF

## Decisions
- **Sparse retriever**: rank-bm25 (BM25Okapi)
- **Tokenizer**: lowercase + split on non-alphanumeric + drop tokens < 2 chars. No stemming, no stopwords.
- **BM25 storage**: in-memory only; rebuilt from chunks.jsonl on script start (build takes ~0.5s)
- **RRF constant**: k = 60 (Cormack et al., 2009 default)
- **Candidate sizes before fusion**: dense top-20, BM25 top-20
- **Metadata filters**: Qdrant-native for dense; post-hoc Python filter for BM25
- **New module**: `src/retrieve.py` with `Retriever` class

## Build stats
- BM25 index build: ~0.5s (3,172 chunks, 0.81s tokenize + 0.47s index on cold cache)
- Retriever full init (BM25 + embedder + Qdrant): ~9-12s cold

## Validation queries

### Q1: "Apple iPhone revenue" --mode dense
- Bit-perfect regression match against Day 4 (same IDs, same scores)
- Refactor into Retriever.dense_search() is clean

### Q2: "data center business" --mode compare --limit 10
- Dense: NVDA 8, MSFT 2, META 0 (same as Day 4)
- BM25: NVDA 10 (even less diverse than dense)
- Hybrid: NVDA 8, MSFT 1, META 1 (META id=717 surfaces at rank #8)
- **Honest finding**: hybrid added META but didn't dramatically diversify. NVDA-dominance is a corpus-density issue both retrievers agree on. Reranker (Day 6) is the natural fix.

### Q3: "deferred revenue" --mode compare
- Textbook RRF demonstration: AAPL id=97 was dense#5 and bm25#1; hybrid promoted it to #1 due to cross-retriever agreement
- TSLA dominates because Apple/Tesla use the phrase "deferred revenue" verbatim; MSFT uses "unearned revenue" so doesn't surface

### Q4: "company's main smartphone product" --mode compare
- Predicted dense advantage didn't materialize; both retrievers converged on AAPL chunks
- "Smartphone" is essentially an Apple-only term in this corpus, so BM25 wins by exact-match alone without needing semantics
- Important lesson for eval design (Day 7): queries that test dense advantage must use terms that exist across multiple companies

### Q5: "operating income" --ticker MSFT --fiscal-year 2024 --mode hybrid
- Filter works correctly through hybrid path
- 4/5 hits are MSFT Cash Flows Statements chunks (whitespace-aligned tables from Day 3 — accepted limitation)
- Hit #4 (id=1437) was `[from: bm25#1]` only — short 120-tok chunk literally containing "Operating income increased..."
- Demonstrates hybrid catching short, exact-term chunks that dense undervalues

## New observations / accepted limitations
1. **Hybrid's diversification gains are corpus-dependent.** When both retrievers strongly agree (corpus density skew), RRF amplifies the consensus rather than diversifies. Document for Day 6 reranker priority.
2. **MSFT uses "unearned revenue" vs others' "deferred revenue".** Vocabulary asymmetry affects BM25; dense partially compensates. Note for Day 7 eval question design.
3. **Qdrant local mode `__del__` traceback** still appears on every script exit. Cosmetic only — fires after all output. Carrying from Day 4.

## Next: Day 6 — BGE reranker + metadata pre-filtering