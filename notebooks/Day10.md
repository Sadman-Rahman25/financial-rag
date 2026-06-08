# Day 10 — Query decomposition for multi-entity questions

The system can now answer cross-company comparison questions that it refused on Day 9.

## The problem we solved

On Day 9, two of four cross-company questions refused unexpectedly. Investigation showed why: hybrid retrieval was returning chunks dominated by one entity. For q13 ("Compare NVIDIA's and Microsoft's data center business focus"), all six top retrieved chunks were NVIDIA. The LLM had zero Microsoft chunks to compare against, so it correctly refused — but the user expected an answer.

The root cause is retrieval bias, not a generation failure. "Data center" is NVIDIA's home turf vocabulary; semantic and lexical search both lean toward NVDA chunks. q15 ("Tesla vs NVIDIA R&D") had the same shape: 4 NVIDIA chunks, 1 Tesla chunk in top-6. q16 ("Microsoft and NVIDIA AI strategy") worked because both companies discuss AI in similar language, so retrieval naturally balanced — that's why it answered when q13 and q15 didn't.

## What we built

1. **`src/decompose.py`** (new) — `detect_entities(question)` returns a list of corpus tickers found in the question, in order of first appearance. Pure regex over a fixed dictionary of company names. Zero LLM calls, deterministic. 14 unit tests cover positive cases, case sensitivity, deduplication, out-of-corpus, and the old "Facebook" name.

2. **`src/generate.py`** (rewrite) — `Generator.answer()` is now a thin router:
   - If `decompose=True` (default) and 2+ entities detected → route to new `_answer_decomposed()` which retrieves 5 chunks per entity (with ticker filter) and runs one synthesis LLM call
   - Otherwise → route to `_answer_single()` which is Day 9 behavior unchanged
   - Added `SYNTHESIS_SYSTEM_PROMPT` with per-company entity match rule and required structure (Company A: ... Company B: ... Comparison: ...)
   - New CLI flags: `--no-decompose`, `--top-k-per-entity`

3. **`src/run_generation.py`** (patched) — added three flags: `--category`, `--no-decompose`, `--top-k-per-entity`. Passes new flags through to `Generator.answer()`.

## The design decisions

**Why auto-decompose by default with a flag for ablation.** Cleanest UX (system "just works" for compare questions) AND ablation capability. The `decompose=False` flag is the receipt that the change was measured before shipping.

**Why 5 chunks per entity (10 total) instead of more.** Keeps total prompt tokens flat with the non-decomposed baseline. If decomposed used 20 chunks and non-decomposed used 10, you couldn't tell whether the improvement came from decomposition or just more context. Holding chunks constant isolates the variable.

**Why rule-based entity detection, not an LLM call.** The corpus is fixed at 5 companies. A 30-line regex over a dictionary is faster, free, and deterministic. Escalate to an LLM call only if paraphrased entity references become a problem.

## Smoke tests (3 LLM calls, ~14k tokens)

- q13 → decomposed, answered with 3 NVDA + 3 MSFT citations
- q15 → decomposed, answered with 2 TSLA + 2 NVDA citations
- q01 (regression) → NOT decomposed (single entity), returned $201,183M with citations [64, 96] identical to Day 9

The regression check is the important one: single-entity questions route to `_answer_single()` which is bit-identical to Day 9's `answer()`.

## Ablation: cross-company subset (4 LLM calls, ~21k tokens)

Ran the 4 cross_company_comparison questions through decomposed mode, saved to `results/day10_decomposed.json`:

|     | Day 9 (no decompose) | Day 10 (decomposed) |
|-----|----------------------|---------------------|
| q13 | REFUSED              | 3 citations         |
| q14 | answered             | 4 citations         |
| q15 | REFUSED              | 4 citations         |
| q16 | answered             | 3 citations         |
| **Cross-company answer rate** | **2 / 4 = 50%** | **4 / 4 = 100%** |

Full 28-question re-eval was deliberately not run. Single-entity questions (24 of 28) route to `_answer_single()`, which is bit-identical to Day 9 — same SYSTEM_PROMPT, same retrieval, same parsing. Running them would burn ~108k tokens (over the Groq daily limit) with zero new information.

## Spot-check findings

**q13:** balanced citations across both companies, per-company entity match clean, zero cross-attribution. Weakness: qualitative answer with no specific dollar figures — the MSFT financial chunks weren't in the top-5 per entity. Same retrieval-precision pattern as q04 in Day 9.

**q15:** perfectly balanced citations (2 TSLA + 2 NVDA), per-company entity match clean, grounded synthesis. Reasonable contrast between Tesla (automotive/energy) and NVIDIA (tech infrastructure). Less sharp than the reference framing but no errors.

The safety property from Day 9 held: zero cross-entity attribution in either spot-check. The PER-COMPANY ENTITY MATCH rule in `SYNTHESIS_SYSTEM_PROMPT` is doing its job. The q27 hallucination class is not reintroduced.

## What's deferred to Day 12 polish

- **q18 (NVIDIA revenue by geographic region):** investigation showed the reference chunks point to footnote text, not the actual revenue table. Eval-set fix needed.
- **q04 partial retrieval:** table chunk (2553) wasn't retrieved alongside the narrative chunk (2413). Chunking change needed (keep table+narrative together) or retrieval change (always pull adjacent chunks).
- **q13 light MSFT citations:** the synthesis prompt could explicitly require at least N citations per company. Marginal improvement.
- **Reranker batching** from Day 8 (49 min → ~2 min)
- **Auto-filter year-range regex** from Day 7 (q25 bug)

## Token economics

Day 10 used ~50,900 tokens total — well under the 100k daily TPD. Strategic scope discipline mattered: full re-eval would have been ~126k, wouldn't have fit, and would have produced zero new information for the 24 single-entity questions.

## The interview story

Three things worth telling:

1. **Diagnosed retrieval bias as the root cause, not generation.** Showed it concretely: for q13, six of six top-retrieved chunks were NVIDIA. The LLM was being asked to compare two companies with chunks from only one. That's a retrieval problem, not a generation problem — and you don't fix retrieval problems by tuning the prompt.

2. **Built the fix as a router with an ablation flag.** Auto-decompose with `decompose=False` for measurement. The ablation table (50% → 100% on cross-company) is the receipt that the change was validated before being made default.

3. **Per-company ENTITY MATCH preserved the Day 9 safety property.** Decomposition uses a new SYNTHESIS_SYSTEM_PROMPT that requires every cited chunk to match the company being described at that point. q27's hallucination class is not reintroduced — verified by spot-checks on q13 and q15, both clean.

## Day 10 wrap

10 of 14 days done. Days 11-14 remaining: Streamlit UI, polish (reranker batching, year-range filter, eval-set fixes), README + architecture diagram, deploy + demo video.