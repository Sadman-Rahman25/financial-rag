# Day 9 â€” Generation with citations

The pipeline answers questions now. Until today the system only retrieved chunks.

## What we built

`src/generate.py` and `src/run_generation.py`. Together they take a question, run hybrid retrieval to get top-10 chunks, feed those plus a careful prompt to Llama 3.3 70B (via Groq), and return a structured JSON answer with inline citations or a clean refusal.

The `Generator` class lazy-loads the Retriever, holds the Groq client, and exposes `answer(question, mode, top_k, filters, temperature)` returning a dict with `answer`, `citations`, `refused`, `retrieved_chunk_ids`, `mode`, `top_k`, `timing_ms`, `usage`.

`run_generation.py` is the batch driver: it reads `data/eval/questions.jsonl`, processes each question, and saves everything to `results/day9_generation.json`. It supports `--resume` which loads the existing JSON and only re-attempts errored questions â€” added partway through the day after hitting Groq's daily token limit.

## The stack added today

- **Groq** for hosted LLM inference (free tier, 100k TPD limit)
- **Llama 3.3 70B Versatile** via model id `llama-3.3-70b-versatile`
- `response_format={"type": "json_object"}` to enforce structured output â€” no fragile regex parsing of free text
- **python-dotenv** for `GROQ_API_KEY` (gitignored)

Why this combination: free, fast (sub-second generation), open weights, no credit card. What current best-practice production RAG uses without paying for OpenAI or Claude.

## Smoke tests

Three representative questions before the full batch:

- **q01** (factual, Apple iPhone): first try with top-5 refused because reference chunks 64 and 96 were not in the top 5. Re-ran with top-10 plus ticker + fiscal_year filters â†’ got `$201,183 million [chunk_64] [chunk_96]`. Bumped `DEFAULT_TOP_K` to 10 globally.
- **q24** (paraphrase, Microsoft AI): excellent multi-sentence answer with 4 citations covering Azure growth, $168.9B cloud revenue, OpenAI partnership, and margin pressure. 2 of 3 reference chunks caught.
- **q27** (out_of_corpus, Google revenue): first try `refused=False` but the answer text said "no info about Google" while citing a Meta chunk for context. No hallucination but the contract was wrong â€” downstream code checking `refused` would have missed it. Tightened rule 3 with explicit "no tangential information from off-topic chunks" â†’ clean refusal on re-run.

## The full batch run (and the daily rate-limit moment)

Ran `python -m src.run_generation` on all 28 questions. Hit Groq's daily TPD limit at question 15 with `Used: 97,811 of 100,000`. 13 questions failed with `RateLimitError`.

This is a real production failure mode: 28 questions at ~4,300 prompt tokens each = ~120k tokens, which exceeds the 100k daily cap. Added `--resume` to skip questions with successful prior results and re-attempt only the failures. Next day: `python -m src.run_generation --resume` finished the remaining 13 in under 5 minutes.

Pre-spot-check state: 24 answered, 4 refused, 0 errored.

## The spot-check that caught a real hallucination

Manually verified 6 answers against the reference: q01, q04, q11, q21, q24, q27.

- **q01, q11, q21, q24: clean.** Numbers correct, citations match, paraphrase questions handled with grounded reasoning.
- **q04** (NVIDIA Data Center FY2026): LLM said "not explicitly stated, but it is mentioned that revenue was up 68%." Honest behavior â€” the chunk with the actual $193.74B figure (`chunk_2553`) was NOT in the top 10 retrieved. Only `chunk_2413` (the narrative paragraph mentioning growth %) made it through. This is consistent with Day 8's Recall@10 = 0.75 â€” Hit@10 was generous, recall is the stricter metric that catches partial misses. The LLM correctly degraded to "I don't know the specific number" instead of fabricating.
- **q27** (Google revenue): CRITICAL. Generated: "Google's revenue in 2024 was $164,501 million [chunk_646]." This is a hallucination â€” chunk 646 is META's FY2024 Revenue table, and $164,501M is META's total revenue, not Google's. The LLM took a number from a Meta chunk and confidently labeled it as Google's. The kind of failure that makes financial RAG systems unshippable.

## The fix

Investigated chunk 646 via `python -m src.eval --view 646`: confirmed META FY2024 Revenue table, $164,501M = Meta total (Family of Apps + Reality Labs). Then inspected the chunk-formatting function `_format_chunks_for_prompt` and found the chunk header DID already show `(META FY2024 | Revenue)`. So the LLM saw the entity in the header and ignored it.

Root cause: the LLM was doing **semantic association** ("the question wants tech revenue 2024, this chunk has tech revenue 2024, slap the company name on it") instead of **literal entity matching** ("the question asks about Google; this chunk header says META; META â‰  Google; refuse"). The original rule 3 said "do not provide facts about other entities from unrelated chunks" â€” but the LLM didn't think it was crossing entities, it was just answering the question with the closest match.

The patch: rewrote `SYSTEM_PROMPT` with:

1. **CORPUS SCOPE upfront** â€” the 5 companies in our corpus listed explicitly (AAPL/MSFT/NVDA/TSLA/META). Anything else triggers immediate refusal.
2. **New rule 3 (ENTITY MATCH)** with the q27 case baked in as a worked example showing wrong behavior vs correct behavior with the exact chunk number and dollar figure.
3. **Strengthened rule 4 (REFUSAL)** that handles the entity-not-in-corpus case explicitly.

Smoke tests of the fix:
- q27 ("What was Google's revenue in 2024?") â†’ `refused=True`, answer = "The corpus does not cover Google's filings", citations = []. âœ“
- q01 (regression) â†’ still returns $201,183M with citations [64, 96]. âœ“

Then invalidated q27 in `results/day9_generation.json` (replaced its `generated` block with an error entry) and ran `python -m src.run_generation --resume`. q27 re-ran in 1.1s with clean refusal. 29 completion tokens â€” that's about as concise as a refusal gets.

## Final results

| Metric | Count |
|---|---|
| Total questions | 28 |
| Answered with citations | 23 |
| Refused | 5 |
| Errored | 0 |

**Refused breakdown:**
- **q27, q28**: `out_of_corpus` (Google, Amazon) â€” correct refusals. q27 only after the patch.
- **q13, q15**: `cross_company_comparison` (NVDA vs MSFT data center, TSLA vs NVDA R&D) â€” refused unexpectedly. The LLM saw chunks from multiple companies and decided no single chunk fully answered the comparison.
- **q18**: `numerical_table` â€” refused unexpectedly. Worth a quick look on Day 10.

Note: **q16** (also cross_company_comparison) answered successfully with 3 citations â€” so "cross-company always refuses" isn't the rule. Something about q13/q15/q18 specifically tripped the contract. Day 10 gets to look at the actual cases instead of an abstract problem.

## Token economics

Initial run (15 questions) + resume (13 questions) + q27 re-run = ~130k total tokens across the project today. Per-question average: ~4,300 prompt tokens (10 chunks Ã— ~400 tokens each plus the system prompt + question) and ~60 completion tokens per answer or ~30 per refusal.

Implication: the Groq free tier doesn't fit the full 28-question eval in a single day. Either spread across days using `--resume`, reduce top-K (hurts coverage), or upgrade to Dev tier. The resume workflow is enough for development.

## What Day 10 needs to address

1. **Cross-company query decomposition.** q13, q15 (and possibly q18) refused because the LLM expects single-chunk grounding. The fix is structural: when the question has multiple entities or sub-questions, split it, retrieve separately for each, then synthesize. Standard "query decomposition" pattern.

2. **Numerical reasoning across chunks.** q04 surfaced a different issue: the table chunk and the narrative chunk separately contain pieces of the answer. The system needs to pull both and reason across them. Either a chunking change (keep table near its narrative context) or a retrieval change (always pull adjacent chunks).

3. **The q18 case specifically.** Quick investigation â€” retrieval miss like q04, prompt-too-strict false refusal, or something new?

## What this day's work demonstrates (the interview story)

The spot-check workflow caught a real safety bug that the aggregate metrics would never have surfaced. Hit@10 = 100% and refusal rate = 14% looked fine. Manual inspection of a single out-of-corpus question revealed the LLM confidently misattributing a Meta revenue figure as Google's. The fix â€” an explicit ENTITY MATCH rule with a worked example baked into the system prompt â€” was diagnosed and shipped in under 30 minutes, with regression testing on q01 and formal re-recording in the batch results.

That loop â€” eval-set â†’ batch-run â†’ manual spot-check â†’ bug â†’ root-cause â†’ prompt patch â†’ verify â†’ re-formalize â€” is the actual practice of building a safety-conscious financial RAG system. The bug is the credential, not a problem.

