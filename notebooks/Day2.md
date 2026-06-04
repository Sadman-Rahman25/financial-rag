# Day 2 Observations: Parsing 10 SEC Filings with LlamaParse

## What I did
- Updated `src/ingest.py` to also download the primary HTML document (`download_details=True`)
- Wrote `src/parse.py` using LlamaParse with a custom parsing instruction for SEC filings
- Tested on AAPL_2025 first, verified table quality, then batch-processed all 10 filings
- Total runtime: ~15 minutes for 10 filings

## Files produced
10 parsed Markdown files in `data/parsed/`:
- AAPL_2024.md, AAPL_2025.md
- MSFT_2024.md, MSFT_2025.md
- NVDA_2025.md, NVDA_2026.md
- TSLA_2025.md, TSLA_2026.md
- META_2025.md, META_2026.md

File sizes range from ~280 KB (Apple) to ~595 KB (Meta).

## Parse quality validation

### Apple FY2025 — checked manually
- Segment Operating Performance table: all numbers match SEC original
- Consolidated Statements of Operations: clean pipe-delimited Markdown, totals bolded
- Consolidated Balance Sheets: 25+ data points preserved correctly
- Section headings (Item 1, Item 1A, Item 7, Item 8) preserved as Markdown H2/H3

### NVIDIA FY2026 — checked manually
- Revenue by Reportable Segments: clean parse (Compute & Networking $193,479M, +67% YoY)
- Operating Income by Reportable Segments: clean
- Consolidated Statements of Shareholders' Equity: multi-column matrix preserved
- Liquidity tables: clean

Quality is consistent across companies despite different table styles.

## Issues observed (to address in Day 3 chunking)

1. **HTML entities not decoded** — found `&#x26;` instead of `&` in some places. Need an HTML unescape pass before chunking.
2. **Page footers from the original document** (e.g., "Apple Inc. | 2025 Form 10-K | 8") may still appear in the parsed output. Need to detect and strip.
3. **Empty table cells** sometimes show as `|  |` — fine for Markdown, but chunker should not treat empty cells as missing data.

## Key insights for chunking strategy (Day 3)

1. **Section structure is preserved** — `## Item 1A. Risk Factors` is detectable. Can use this as a chunk boundary signal.
2. **Tables are intact** — chunker must treat tables as atomic units, never splitting them mid-row.
3. **Narrative follows tables** — keep narrative + adjacent table together when possible.
4. **Bold formatting marks totals** — `**Total net sales**` and `**$416,161**` indicate summary rows. Useful metadata.

## Decision: stick with LlamaParse
Considered alternative parsers (Unstructured.io, Marker, pdfplumber). Quality is high enough with LlamaParse that switching tools isn't justified. LlamaParse handled:
- 5 different companies' filing styles
- Income statements, balance sheets, cash flow statements
- Segment tables, R&D breakdowns, share repurchase tables
- Multi-column equity statements

## Cost
- LlamaParse free tier: 1,000 pages/day
- 10 filings × ~80 pages = ~800 pages used today
- Well within free tier; would also be cheap on paid tier