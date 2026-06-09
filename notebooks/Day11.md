# Day 11 — Polish: fixed two known bugs, exposed a third for Day 12

A short polish day. Most of the work was code/data changes verified without LLM calls. Total LLM token spend: 4,360 (one verification run of q18). Two known bugs fixed, one previously-masked retrieval bug now exposed.

## What we fixed

### 1. Year-range regex in `extract_filters` (src/retrieve.py)

The auto-filter regex used `re.search`, which returns the first year match. For queries like "How did Apple's services revenue grow from 2023 to 2024?", it picked 2023 — a year Apple's filings in our corpus don't cover — and retrieval returned zero results.

**The fix is one line:** replace `re.search` with `re.findall`, then take the maximum.

```python
# Before
year_match = re.search(r"(?:^|\W)(?:fy)?(202\d)\b", query, re.IGNORECASE)
if year_match:
    detected["fiscal_year"] = int(year_match.group(1))

# After
all_years = re.findall(r"(?:^|\W)(?:fy)?(202\d)\b", query, re.IGNORECASE)
if all_years:
    detected["fiscal_year"] = max(int(y) for y in all_years)
```

Same regex pattern — just collect all matches and take the latest year. Handles every range form uniformly: "from X to Y", "X to Y", "X vs Y", "X-Y", "X through Y", "between X and Y", "X and Y". The structural rationale: in a multi-year query, the **later year's 10-K typically contains the earlier year as a comparative**, so filtering to the later year retrieves chunks that cover both.

**Validation:** 8/8 unit tests pass.

| Question | Expected | Got |
|---|---|---|
| "How did Apple's services revenue grow from 2023 to 2024?" | 2024 | 2024 ✓ |
| "What was Apple's FY2024 revenue?" | 2024 | 2024 ✓ |
| "Microsoft 2024 vs 2023 cloud growth" | 2024 | 2024 ✓ |
| "Apple revenue between 2023 and 2024" | 2024 | 2024 ✓ |
| "Tesla revenue from 2022 to 2025" | 2025 | 2025 ✓ |
| "Tesla revenue" | None | None ✓ |
| "What was the revenue?" | None | None ✓ |
| "NVIDIA FY2026 results" | 2026 | 2026 ✓ |

**Scope note:** this helps CLI and Streamlit usage where users type natural-language queries and rely on auto-filter. It does NOT change the stored Day 9 eval results, because the eval driver uses explicit `expected_filters` from `questions.jsonl`, not auto-extracted values. Token cost: 0.

### 2. q18 eval-set fix (data/eval/questions.jsonl)

q18 asks "What was NVIDIA's revenue by geographic region in fiscal year 2026?" — the eval set's reference chunks were `[2551, 2516]`:

- Chunk 2551 *is* relevant — it's the methodology footnote with the "31% outside US" data point.
- Chunk 2516 is the **Foreign Currency Derivatives** section. Completely unrelated to geographic revenue. Got into the eval set by accident during Day 7 hand-curation, probably confused with an adjacent NVDA chunk.

Inspecting the window around 2551 revealed the actual structure:

| id | type | content |
|---|---|---|
| 2549 | prose | "Revenue by geographic area is based upon the location of the customers' headquarters..." (intro for the table) |
| **2550** | **table** | **The actual geographic revenue table:** US $149,617M, Taiwan $42,345M, China (incl. HK) $19,677M, Other $4,299M, total $215,938M |
| 2551 | prose | Methodology footnote (Q3 FY2026 change to customer-headquarters basis) + the "31% outside US" data point |
| 2553 | table | **Revenue by End Market** table (Data Center / Gaming / Auto) — sits adjacent but wrong table for this question |

**The fix:** updated q18's reference chunks to `[2549, 2550, 2551]` (intro + table + footnote) and rewrote `reference_answer` to reflect the actual country-level breakdown.

## The bigger finding: the eval set was masking a real retrieval bug

This is the story worth telling.

With the old refs `[2551, 2516]`, q18's retrieval Hit@10 was 1.0 because chunk 2551 was retrieved at rank 1 — the eval looked clean. But the LLM still refused, because retrieval found the *footnote* without the *table* the footnote annotates. Retrieval was technically "hitting" the references; it just wasn't surfacing the chunk with the actual answer.

After fixing the refs to `[2549, 2550, 2551]`:
- Hit@10 still = 1.0 (2551 still retrieves)
- Recall@10 drops from 1.0 (2/2) to 0.33 (1/3) — **honest measurement of the bug**

The eval is now diagnostic instead of comforting. Recall@10 < 1.0 for q18 keeps flagging the retrieval issue until it's actually fixed.

## The q18 re-run: graceful degradation in practice

After patching the refs, we invalidated q18 and re-ran via `python -m src.run_generation --resume`. One LLM call, 4,360 tokens.

Outcome: **the system gracefully degraded from refusal to partial answer.**


Refused: False
Citations: [2551]
Answer: "Revenue from sales to customers headquartered outside of the United
States accounted for 31% of total revenue for fiscal year 2026
[chunk_2551]. Additionally, it is mentioned that 76% of Data Center
revenue from Taiwan-headquartered customers was attributed to end
customers based in the United States and Europe [chunk_2551]."


What the LLM did:
- Read chunk 2551 (the only relevant chunk retrieved)
- Extracted the two concrete geographic data points it contained
- Did NOT fabricate the country-level breakdown (US $149,617M etc.) — those numbers live in chunk 2550, which wasn't retrieved
- Cited only [2551], an NVDA chunk — no cross-entity attribution

This is the correct behavior. The system honestly works with what's retrieved. A user asking this question still gets a useful partial answer instead of a hard refusal — and crucially, doesn't get fabricated table figures.

## New state of `results/day9_generation.json`

|     | Day 9 final | Day 11 final |
|-----|-------------|--------------|
| Answered | 23 | **24** |
| Refused  | 5  | **4**  |
| Errored  | 0  | 0      |

The 4 remaining refusals:
- **q27** (Google revenue) — correctly refused, out of corpus
- **q28** (Amazon revenue) — correctly refused, out of corpus
- **q13** (NVDA vs MSFT data center) — refused in non-decomposed mode; answers in decomposed mode per `results/day10_decomposed.json`
- **q15** (TSLA vs NVDA R&D) — same as q13

Of 28 total: 24 answered single-entity + 2 correctly-refused-out-of-corpus + 2 answered-in-Day10-decomposed-mode = **28/28 covered**.

## Token economics

| Stage | Tokens |
|---|---|
| 1. Year-range regex fix + 8 unit tests | 0 |
| 2. q18 eval-set inspection + patch | 0 |
| 3. q18 re-run via `--resume` | 4,360 |
| 4. Spot-check | 0 |
| **Total Day 11** | **4,360** |

A polish day shouldn't burn tokens. This one barely did.

## What's deferred to Day 12

- **q18 retrieval issue (now exposed):** chunk 2550 (geographic table) isn't retrieved for "by region" queries because table chunks have sparse prose and the word "region" doesn't appear in the table header ("Geographic Revenue based upon Customer Headquarters Location"). Two paths — chunk-level: merge table with intro and footnote in the same chunk (2549 + 2550 + 2551 together). Retrieval-level: query rewriting to expand "region" → "geographic area" etc.
- **q04 partial retrieval:** chunk 2553 (Data Center end-market table) not retrieved alongside narrative 2413. Same table-chunking pattern.
- **Reranker:** documented as suboptimal per Day 8 (hurts MRR on average). Speed optimization and quality work both deferred.
- **q13 light MSFT citations from Day 10:** synthesis prompt could require N citations per company. Marginal improvement.

## The interview story

Two parts that land cleanly:

1. **Found a bug masking another bug.** The eval set for q18 looked clean (Hit@10 = 1.0) because the wrong reference chunks happened to be retrieved at rank 1. Fixing the references to point at the *right* chunks dropped Recall@10 to 0.33 — the metric now correctly diagnoses the underlying retrieval issue. This is the kind of bug that survives in a system until you sit down and actually read what the references are.

2. **The system gracefully degraded.** When retrieval missed the table chunk for q18, the LLM didn't fabricate country-level figures — it extracted the partial info from the footnote it had, cited it, and stopped. The Day 9 ENTITY MATCH rule and the synthesis prompt's strict citation discipline preserved the no-hallucination property even in a partial-retrieval failure mode. That's the system working as intended.

## Day 11 wrap

11 of 14 days done. Three remaining: Day 12 polish + Streamlit UI, Day 13 README + architecture diagram, Day 14 deploy + demo. The Day 12 retrieval-fix work has clear scope and a clear test — q18 should produce the country-level breakdown after the fix.


