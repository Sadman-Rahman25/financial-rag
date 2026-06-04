# Day 1 Observations: Apple 10-K (Fiscal Year 2025)

## Document basics
- Filing date: 2025-10-31
- Fiscal year end: September 27, 2025
- Total length: roughly 80 pages

## Sections I inspected

### Item 1A — Risk Factors
- Organized into 4 subsections: Business Risks, Legal and Regulatory Compliance Risks, Financial Risks, General Risks
- Each individual risk has a bold heading + 2-5 paragraphs of detail
- Risks are independent units — natural chunk boundaries
- Apple discloses specific real risks: China manufacturing concentration, Google antitrust case, EU DMA compliance, AI-related risks

### Item 7 — MD&A
- Contains "Segment Operating Performance" table: geographic revenue by region (Americas, Europe, Greater China, Japan, Rest of Asia Pacific) for 3 fiscal years
- Total net sales FY2025: $416,161M (+6% YoY)
- Greater China declined 4% in 2025 — interesting story point
- Narrative paragraphs surround the table explaining what drove the changes
- Critical chunking observation: narrative and table must stay together

### Item 8 — Consolidated Statements of Operations (Income Statement)
- 3 columns of fiscal years (2025, 2024, 2023)
- Rows: Products revenue, Services revenue, Total net sales, Cost of sales (split), Gross margin, R&D, SG&A, Operating income, Net income, EPS
- All numbers in millions
- Apple FY2025: Total revenue $416,161M, Net income $112,010M, Diluted EPS $7.46
- Services revenue grew every year (consistent narrative for queries)

### Item 8 — Consolidated Balance Sheets
- Snapshot at 2 points in time (Sept 27 2025 vs Sept 28 2024)
- Three sections: Assets, Liabilities, Shareholders' equity
- Apple FY2025 totals: Total assets $359,241M, Total liabilities $285,508M, Equity $73,733M
- 25+ distinct data points in one table

## Patterns I noticed

### Page footers (need to strip)
Every page has a footer like "Apple Inc. | 2025 Form 10-K | 29". Repeats on every page. Will pollute embeddings if not stripped during parsing.

### Risk Factor categories as metadata
The 4 risk subsections should be captured as metadata so users can filter ("show me only financial risks").

### Tables are dense and meaningful
The income statement has 3 columns × ~15 rows of dollar amounts. The balance sheet has 2 columns × 25+ rows. Flattening these would destroy meaning.

## Sample queries my RAG should answer

Easy (single chunk):
1. What was Apple's total revenue in fiscal 2025?
2. What was Apple's R&D expense in 2025?
3. How much cash did Apple have at end of fiscal 2025?

Medium (requires Item 7 narrative or income statement):
4. How did Services revenue change in fiscal 2025?
5. What was Apple's gross margin percentage in 2025?
6. Did Apple's revenue in Greater China grow or decline in 2025?

Multi-hop / comparison:
7. How did iPhone revenue change between fiscal 2024 and 2025?
8. What's the trend in R&D spending over the last 3 years?
9. By how much did Apple's net income grow from 2024 to 2025?

Risk-related:
10. What macroeconomic risks did Apple disclose in 2025?
11. What does Apple say about AI-related risks?
12. What antitrust investigations is Apple currently subject to?

## Implications for the pipeline

1. **Chunking:** Use section-aware splitting. Each risk in Item 1A = one chunk. Each major MD&A subsection = one chunk (keeping narrative + adjacent table together).
2. **Metadata per chunk:** ticker, fiscal_year, filing_date, section_name (Item 1, 1A, 7, 8), subsection_name (e.g., "Business Risks"), chunk_type (prose vs table).
3. **Preprocessing must strip:** page footers, repeated boilerplate.
4. **Tables:** Extract as standalone single chunks. Convert to markdown to preserve structure.
5. **Time-aware retrieval:** Many queries are comparisons across fiscal years — metadata filtering by fiscal_year is essential.

## Surprises
- The Risk Factors section is far more substantive than expected — Apple discusses specific risks (China concentration, Google antitrust, EU DMA) in concrete language, not generic legal disclaimers.
- The Segment Operating Performance table in MD&A is more useful than the income statement for many queries because it adds geographic breakdown.
- Page footers absolutely repeat on every page — preprocessing must strip them.