# Day 8 — Retrieval Evaluation

## Goal
Turn Day 7's hand-curated eval set into hard numbers. Compute industry-standard
retrieval metrics across all four retrieval modes (dense, BM25, hybrid, rerank)
so the README's ablation table is backed by real measurement, not vibes.

## Decision: roll-our-own metrics, not RAGAS

The original plan said "wire up RAGAS." On reflection, RAGAS is built for
*generation* metrics (faithfulness, answer relevance) using LLM-as-judge —
that's Day 9's job. For retrieval metrics, the formulas are short, well-defined,
and don't need an LLM. Implementing them from scratch in `src/metrics.py`:
- Removes external API dependency (deterministic, no rate limits, free)
- Demonstrates depth (~80 lines of math, readable in two minutes)
- Easier to extend with custom metrics later

RAGAS earns its place on Day 9 where LLM-as-judge is genuinely useful.

## Tooling

### `src/metrics.py`
Five metrics, binary relevance, stdlib only. Each a pure function:
- `hit_at_k` — 1 if any reference in top K
- `recall_at_k` — fraction of refs in top K
- `precision_at_k` — fraction of top K that are refs
- `mrr_at_k` — reciprocal of first-hit rank (1-indexed)
- `ndcg_at_k` — DCG / IDCG with log_2 discount

Includes a self-test (`python -m src.metrics`) covering perfect retrieval,
no-hit cases, empty-refs handling, and NDCG order-sensitivity.

### `src/run_eval.py`
The driver. Two-phase loading for 8 GB RAM:
- **Phase 1**: load Retriever (BGE embedder + BM25 index). For each
  question, fetch dense top-10, BM25 top-10, hybrid top-20.
  Slice `[:10]` from the hybrid result for evaluation; pass the full 20
  to the reranker (one hybrid call serves both purposes).
- **Free retriever**, load Reranker.
- **Phase 2**: rerank each question's cached hybrid pool → top 10.
- Compute 5 metrics × 4 K values × 4 modes per question.
- Save per-question and aggregate results to
  `results/day8_retrieval_eval.json`; print summary + by-category tables.

26 questions tested. q27 and q28 (out-of-corpus, empty refs by design) are
excluded from retrieval evaluation; their refusal quality is Day 9's work.

## Aggregate results
Metric             dense      bm25    hybrid    rerank
hit@1              0.577     0.192     0.500     0.423
hit@3              0.769     0.308     0.731     0.731
hit@5              0.885     0.615     0.923     0.846
hit@10             0.962     0.846     1.000     1.000
mrr@10             0.686     0.341     0.668     0.592
ndcg@10            0.651     0.352     0.582     0.568
recall@10          0.821     0.532     0.750     0.763
precision@10       0.227     0.142     0.208     0.212

Two distinct stories live in this table.

### Story 1: Coverage (Hit@10) — Hybrid wins, as expected
Hybrid achieves **100% Hit@10**, confirming Day 7's dry-run. RRF fusion of
dense and BM25 catches every reference chunk in top 10 across all 26
testable questions. Rerank ties at 100% because it just reorders hybrid's
top-20 candidates.

### Story 2: Ranking quality (MRR, NDCG, Hit@1) — Dense wins, against the standard narrative
Dense puts the correct chunk first **57.7%** of the time (Hit@1), more
than Hybrid (50.0%) and significantly more than Rerank (42.3%). Dense
leads on MRR@10 (0.686 vs 0.668), NDCG@10 (0.651 vs 0.582), and Hit@1.

This is unusual. The textbook RAG narrative is "hybrid + rerank > hybrid
> dense > BM25." Our data says: **hybrid > dense > rerank > BM25 on
coverage, but dense > hybrid > rerank > BM25 on ranking quality.** The
reranker is dragging MRR and NDCG **down** on average.

## By-category breakdown

### Hit@10 by category
Category                     N    dense    bm25   hybrid   rerank
accounting_term              4    1.000   1.000    1.000    1.000
cross_company_comparison     4    1.000   0.750    1.000    1.000
multi_step                   2    1.000   0.500    1.000    1.000
numerical_table              4    1.000   1.000    1.000    1.000
paraphrase                   4    0.750   0.750    1.000    1.000
single_entity_factual        8    1.000   0.875    1.000    1.000
Hybrid is the only mode at 100% across every category. Dense matches except
on paraphrase. BM25 stumbles on cross-company comparison and multi-step —
categories needing semantic generalization beyond exact match.

### MRR@10 by category
Category                     N    dense    bm25   hybrid   rerank
accounting_term              4    0.781   1.000    0.800    0.590
cross_company_comparison     4    0.708   0.169    0.750    0.494
multi_step                   2    1.000   0.125    0.600    1.000
numerical_table              4    0.833   0.379    0.750    0.833
paraphrase                   4    0.128   0.182    0.192    0.323
single_entity_factual        8    0.754   0.214    0.775    0.553
This is where the architecture-level finding lives:

- **BM25 wins MRR on accounting_term (1.000)** — exact phrases like
  "stock-based compensation," "unearned revenue," and "regulatory credits"
  match SEC subsection headers letter-for-letter. For terminology-heavy
  queries, lexical retrieval is the right tool and semantic similarity is
  wasted effort.
- **Reranker has exactly one strong win: paraphrase (0.323 vs hybrid
  0.192)** — the only category where cross-encoder scoring genuinely helps.
  NDCG@10 nearly doubles (0.465 vs 0.251).
- **Reranker hurts on every other category**:
  accounting_term (0.590 vs 0.800), cross_company (0.494 vs 0.750),
  single_entity (0.553 vs 0.775). The reranker false positives we
  documented in Day 7 (q10, q23, q26) are systematic, not isolated.
- **Multi-step (n=2)** ties dense and rerank at 1.000 MRR, but n=2 is
  too small for confident inference.

### NDCG@10 by category
Category                     N    dense    bm25   hybrid   rerank
accounting_term              4    0.730   0.837    0.789    0.591
cross_company_comparison     4    0.624   0.250    0.553    0.461
multi_step                   2    0.825   0.101    0.628    0.718
numerical_table              4    0.691   0.391    0.628    0.672
paraphrase                   4    0.205   0.228    0.251    0.465
single_entity_factual        8    0.783   0.266    0.623    0.571
Same patterns as MRR. Dense dominates most categories; reranker wins
decisively on paraphrase; BM25 wins on accounting_term.

## Case study: q24, the reranker doing its actual job

Question 24 ("What does Microsoft say about how its AI infrastructure
investments are paying off?") was a Day 7 dense miss. Day 8 per-question
metrics show the reranker rescuing it cleanly:

| Mode | Hit@10 | Hit@3 | MRR@10 | NDCG@10 | Recall@10 |
|---|---|---|---|---|---|
| Dense  | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| BM25   | 1.000 | 0.000 | 0.250 | 0.384 | 0.667 |
| Hybrid | 1.000 | 0.000 | 0.167 | 0.308 | 0.667 |
| **Rerank** | **1.000** | **1.000** | **0.500** | **0.679** | **1.000** |

Dense missed all three references. Hybrid retrieved two of three but
buried them at positions 6+. The reranker promoted a reference to
position 2 AND surfaced the third missing one (recall went from 0.667 →
1.000). NDCG@10 jumped from 0.308 to 0.679 — more than 2x.

This is the cross-encoder doing exactly what it should on a paraphrase
query: re-scoring based on semantic relevance to the question, not lexical
or vector similarity to the chunk text.

The lesson: **the reranker is not broken. It is highly category-specific.**
A selective architecture (apply reranker only on paraphrase-style queries)
would give us the best of both worlds. Day 12 polish.

## Architecture recommendations

The eval data supports three concrete production choices:

1. **Default pipeline: hybrid only, no reranker.** Hybrid hits 100% Hit@10,
   0.668 MRR@10, 0.582 NDCG@10. Reranker reduces MRR and NDCG on average;
   it costs latency and complicates ops without an average gain.
2. **Selective reranker for paraphrase queries.** When the query lacks
   vocabulary overlap with the corpus, reranker yields ~2x NDCG@10
   improvement. A query intent router would unlock this. (Day 12.)
3. **Dense-only is a defensible degradation path.** Best MRR (0.686), best
   NDCG (0.651), best Hit@1 (0.577). Within 4% of hybrid on Hit@10.
   Suitable when two-index ops complexity is prohibitive.

This is the opposite of the textbook "hybrid + rerank = SOTA" narrative —
and the data does not support that on this corpus.

## Honest limitations

- **Reranker is unacceptably slow.** Phase 2 took 49 minutes (~110s per
  question to rerank 20 candidates). Expected: ~5s per question with proper
  batching. `src/rerank.py` is almost certainly calling the cross-encoder
  one (query, doc) pair at a time in a Python loop instead of one batched
  forward pass per question. This affects latency, not quality — the
  metrics here are valid. Day 12 polish: fix batching.
- **n=26 testable questions is small.** Aggregate confidence intervals are
  wide. Category-level numbers (n=2 to n=8) are directionally indicative,
  not statistically definitive.
- **Single-annotator reference chunks.** All refs chosen by author
  judgment. No inter-annotator agreement check. Some choices were
  defensible-but-debatable — e.g. q19's chunk 1438 marked as canonical
  but never retrieved by any mode, suggesting either author bias or a
  corpus chunk-boundary problem.
- **Macro-averaging treats all questions equally.** A category-weighted
  scheme was considered but rejected for simplicity.
- **Reference set sizes vary** (1 to 4 chunks per question), affecting
  Recall@K interpretation. Hit@K and MRR are more robust to this.

## Reproducibility

Anyone with the repo can reproduce these numbers end-to-end:
python -m src.metrics      # self-test the math
python -m src.run_eval     # ~5min phase 1, ~50min phase 2
Outputs:
- `results/day8_retrieval_eval.json` — full per-question and aggregate
  metrics; ~50 KB.
- Console summary tables identical to those above.

## Next: Day 9 — generation with citations

Wire up Groq + Llama 3.3 70B to take retrieved chunks and produce answers
with inline citations. Open questions for Day 9:
- Faithfulness: does the answer use only information from retrieved chunks?
- Citation precision: do inline citations point to chunks that actually
  support the claim?
- Refusal quality on q27 (Google) and q28 (Amazon AWS): does the LLM
  refuse, or does it hallucinate from irrelevant retrieved MSFT/AAPL/NVDA
  chunks?

RAGAS earns its keep here, where LLM-as-judge is appropriate.
