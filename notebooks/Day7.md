# Day 7 — Build the Eval Set

## Goal
Build a 28-question evaluation set with reference answers and ground-truth
chunk IDs — the artifact that lets Day 8 produce real numbers for the README's
ablation table.

## Decisions
- 28 questions across 7 categories: 8 single-entity factual, 4 accounting term,
  4 cross-company comparison, 4 numerical/table, 4 paraphrase, 2 multi-step,
  2 out-of-corpus.
- Schema: `id`, `category`, `question`, `reference_answer`, `reference_chunk_ids`,
  `expected_filters`, `difficulty`, `notes`. JSONL at `data/eval/questions.jsonl`.
- All questions hand-written. Reference chunks chosen by reading chunk text
  across 4 retrieval modes (dense, BM25, hybrid, rerank) — not by accepting
  top-1 retrieval (avoids circular eval).
- Out-of-corpus questions (q27 Google, q28 Amazon) have `reference_chunk_ids: []`
  and refusal-style answers. They are excluded from retrieval Hit@K evaluation
  and will be tested on refusal-quality in Day 9.

## Tooling
New module `src/eval.py` with:
- `find_candidates(qid)` — 4-mode aggregation across dense/BM25/hybrid/rerank
- `view_chunk(id)` — full chunk text by ID
- `update_question(qid, refs=..., answer=...)` — in-place jsonl update
- `dry_run(mode)` — Hit@10 evaluation across all testable questions
- `status()` — progress check

Two-phase loading in `find_candidates` to fit on 8 GB RAM (free Retriever
before loading Reranker — pattern from Day 6).

## Dry-run results
26/28 questions tested (q27/q28 excluded; empty refs by design).

| Mode | Hit@10 | Misses |
|---|---|---|
| Hybrid | 26/26 = 100.0% | none |
| Dense  | 25/26 = 96.2%  | q24 (paraphrase) |
| BM25   | 22/26 = 84.6%  | q08, q15, q22, q25 |

## Observations and corpus realities surfaced

### Auto-filter regex picks first year in query
`extract_filters()` in `src/retrieve.py` takes the first 4-digit year regex
match. On "from 2023 to 2024", it filtered for fiscal_year=2023, which doesn't
exist in our corpus (AAPL is FY2024/FY2025). Result: q25's original query
returned zero results across all four modes. Workaround: rephrase to single
year. Real fix is a Day 12 polish item — detect year ranges and pick the
latest, or extract all years.

### Reranker false positives on broad subsection names
Three documented cases where the cross-encoder ranked clearly-wrong chunks #1
with scores > 0.7:
- q10 (Tesla regulatory credits) → chunk 2615 (product liability self-insurance)
- q23 (Tesla self-driving) → chunk 2874 (energy generation Technology)
- q26 (Tesla 2024 challenges) → chunk 2816 (litigation re: going-private
  transaction)
Pattern: reranker over-weights generic subsection names ("Technology",
"Risk Factors") even when chunk content is off-topic. Hybrid retrieval still
recovered these via BM25 contribution, so Hit@10 wasn't affected — but
reranker ordering was. Worth re-evaluating reranker model choice at Day 12
or accepting the limitation.

### Corpus density bias on cross-company queries
NVDA-dominated retrieval on "data center" queries (q13) because NVDA's entire
10-K is data center, while MSFT frames the same business as "Intelligent
Cloud." Similarly, q14 (Apple vs Meta advertising) returned zero Apple chunks
in top 10 because Apple's ad business is small and embedded in Services.
Forced inclusion of MSFT chunk 1438 in q13 references reveals retrieval
weakness; q14 references skew Meta-heavy by necessity.

### Numerical-table queries miss the actual table
q19 (MSFT segment revenue) marked chunk 1438 (Segment Results of Operations
table) as canonical, but it wasn't retrieved in any mode's top 10. Retrieval
finds prose discussion of segments instead. Confirms the table-bias finding
from Day 6: reranker prefers prose, dense embeddings underweight structured
table chunks. This shows up in BM25 misses on q15 too.

### NVDA geographic table not in retrieved chunks
q18 (NVDA geo revenue) — the actual geographic dollar table was not surfaced
by any mode. Chunk 2551 contains the methodology notes (percentages, customer
concentration) but not the country-by-country dollar table. The table likely
exists in a separate chunk that wasn't surfaced by the query "revenue by
geographic region." A real corpus coverage gap revealed by the eval.

### Apple supplier concentration: lexical match wins
q22 marked chunk 306 (subsection literally named "Concentrations In The
Available Sources Of Supply Of Materials And Product") as canonical. BM25
retrieval missed it but hybrid found it via dense contribution. Counterpoint
to the table-bias finding: when SEC section headers align with question
phrasing, BM25 alone isn't sufficient — dense semantic match adds the
specific subsection that matters.

## Accepted limitations
- Hand-written eval is small (28 questions). At Day 12 we can consider
  LLM-augmented expansion to 100+ questions.
- Reference chunk IDs are based on author judgment; could be more or fewer
  per question. Single-annotator bias not corrected.
- Some categories (cross-company) have inherently more chunks that partially
  answer, so reference sets are larger.
- q14 and q18 reference sets are limited by genuine corpus coverage gaps
  (Apple's advertising disclosure is sparse; NVDA's geographic dollar table
  wasn't surfaced).
- PowerShell quoting issues required avoiding internal double quotes in
  reference answers — minor style limitation, not a content limitation.

## Next: Day 8 — Wire up RAGAS, compute MRR/NDCG across all 4 retrieval modes