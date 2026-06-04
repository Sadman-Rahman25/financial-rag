# Day 3 — Chunking

## Goal
Convert 10 parsed Markdown filings into retrieval-ready chunks with rich metadata, ready for embedding on Day 4.

## Outcome
- **3,172 chunks** written to `data/chunks.jsonl` (~7 MB)
- All chunks tagged with: `ticker`, `company_name`, `fiscal_year`, `filing_date`, `period_end`, `source_file`, `section`, `subsection`, `chunk_type`, `n_tokens`
- Two chunk types: `prose` (~2,521 chunks, ~80%) and `table` (~651 chunks, ~20%)
- Token sizing: 256-512 token target, ~20% overlap, tables atomic (no splitting)

## Per-file breakdown

| File           | Total | Prose | Table |
|----------------|-------|-------|-------|
| AAPL_2024.md   | 169   | 122   | 47    |
| AAPL_2025.md   | 165   | 122   | 43    |
| META_2024.md   | 486   | 418   | 68    |
| META_2025.md   | 494   | 425   | 69    |
| MSFT_2024.md   | 387   | 303   | 84    |
| MSFT_2025.md   | 287   | 211   | 76    |
| NVDA_2025.md   | 299   | 239   | 60    |
| NVDA_2026.md   | 281   | 224   | 57    |
| TSLA_2024.md   | 295   | 218   | 77    |
| TSLA_2025.md   | 309   | 239   | 70    |

## Pipeline (src/chunk.py)

The chunker runs five sequential stages:

1. **Preprocessing** — `html.unescape()`, strip page footers (regex), strip `NO_CONTENT_HERE` LlamaParse sentinels, fix empty table cells, collapse blank lines.
2. **Metadata extraction** — Read `FILED AS OF DATE` and `CONFORMED PERIOD OF REPORT` from each filing's `full-submission.txt` header. Filing date is the date SEC accepted the document; period_end is the fiscal year-end.
3. **Structural parsing** — Walk the cleaned Markdown line by line. Classify each line as heading, table row, blank, or prose. Group consecutive same-type lines into "blocks." Apply real-SEC-section detection (`Item N.` regex) regardless of Markdown heading level. Normalize section names (ASCII punctuation, title case) to collapse case/apostrophe duplicates.
4. **Chunking** — Tables are atomic (one table = one chunk). Prose is packed into ~384-token chunks with ~75-token overlap, snapping to sentence boundaries. Section boundaries are hard breaks. Subsection headings are prepended to following prose for retrievability.
5. **Output** — Write each chunk as one JSON object per line to `data/chunks.jsonl`.

## Key decisions

**Decided to fix the Day 2 file-naming bug.** META_2024/2025 and TSLA_2024/2025 were originally named with the wrong fiscal years (parser used accession year instead of fiscal period end). Renamed files; also fixed `parse.py`'s `get_accession_year` to read from the SEC header for future re-runs.

**Decided NOT to fix LlamaParse mislabels.** Apple's `Item 4. Exhibits and Financial Statement Schedules` (LlamaParse rendered the wrong Item number on a real heading) is preserved as-is. The body text is correct, the Item number is wrong, but fixing it requires hardcoded knowledge per-filer that doesn't generalize. Section text is what matters for retrieval, not Item numbers.

**Decided NOT to fix whitespace-aligned tables.** Some MSFT financial statements (cash flow, income statement) are rendered by LlamaParse as spatial text rather than Markdown pipes. They end up classified as `prose` chunks. The numbers are still present and BM25-searchable, so we accept this for now and revisit on Day 8 if retrieval evals show problems.

**Tables can exceed 512 tokens.** Some financial statement and exhibit tables are 1,000-1,700 tokens. This exceeds BGE-large's 512-token max context. Day 4 will handle this — likely truncate to first 512 tokens for embedding while preserving full text for display/citation.

## Token statistics (AAPL_2025 sample)

- Min: 2 tokens (sections like "Item 6. [Reserved]" with body "None.")
- Max: 1,695 tokens (large exhibit tables, kept atomic)
- Avg: ~321 tokens
- Distribution: ~55% in 256-512 sweet spot, ~25% in 100-256, ~16% under 100 (mostly small tables and reserved/empty sections), ~6% over 512 (all tables)

## What's queued for later

- **Day 4:** Decide how to handle 512+ token tables for BGE embedding (truncate vs. summarize vs. row-split)
- **Day 8:** Measure baseline retrieval quality; revisit whitespace-aligned table problem if metrics show it hurts
- **Day 12 (stretch):** Anthropic Contextual Retrieval — generate per-chunk context summaries to boost retrieval recall by 35-67%