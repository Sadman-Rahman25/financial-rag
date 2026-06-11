"""
src/chunk.py

Chunks parsed SEC 10-K Markdown files into retrieval-ready units with metadata.

Pipeline:
  1. Preprocess (clean HTML entities, footers, empty cells)
  2. Extract per-file metadata (ticker, fiscal_year, filing_date)
  3. Structural parse (sections, subsections, tables, prose)
  4. Chunk with size-aware splitting (256-512 tokens, tables atomic)
  5. Write to data/chunks.jsonl

Run: python -m src.chunk --one AAPL_2025   OR   python -m src.chunk --all
"""

import argparse
import html
import re
from pathlib import Path

import tiktoken

# ---------- Paths ----------
PARSED_DIR = Path("data/parsed")
CHUNKS_PATH = Path("data/chunks.jsonl")

# ---------- Tokenizer (used for sizing only) ----------
ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base. Not the exact tokenizer BGE uses,
    but close enough for sizing decisions and an industry-standard proxy."""
    return len(ENCODER.encode(text))


# ===========================================================
# STAGE 1: PREPROCESSING
# ===========================================================

# Page footers look like: "Apple Inc. | 2025 Form 10-K | 8"
# or:                    "Table of Contents"
# or:                    "NVIDIA Corporation and Subsidiaries"
# We strip lines that look like footers/headers but are not real content.
FOOTER_PATTERNS = [
    re.compile(r"^.+\|\s*\d{4}\s+Form\s+10-K\s*\|\s*\d+\s*$", re.IGNORECASE),
    re.compile(r"^Table\s+of\s+Contents\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),  # standalone page numbers
]


def fix_html_entities(text: str) -> str:
    """Decode HTML entities like &#x26; -> & and &amp; -> &."""
    return html.unescape(text)


def strip_footers(text: str) -> str:
    """Remove repeating page footers/headers line by line."""
    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if any(pat.match(stripped) for pat in FOOTER_PATTERNS):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def strip_llamaparse_sentinels(text: str) -> str:
    """LlamaParse occasionally inserts 'NO_CONTENT_HERE' as a layout sentinel.
    These add noise to embeddings without conveying information — strip them."""
    # Remove standalone NO_CONTENT_HERE markers and any blank lines around them.
    return re.sub(r"NO_CONTENT_HERE\s*", "", text)


def fix_empty_table_cells(text: str) -> str:
    """Replace empty table cells | | with | — | so the cell still parses."""
    # Repeat the substitution because regex doesn't catch overlapping matches in one pass.
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\|[^\S\n]+\|", "| — |", text)
    return text


def collapse_blank_lines(text: str) -> str:
    """Collapse 3+ consecutive blank lines into 2."""
    return re.sub(r"\n{3,}", "\n\n", text)


def preprocess(text: str) -> str:
    """Run all cleanup steps in order."""
    text = fix_html_entities(text)
    text = strip_footers(text)
    text = strip_llamaparse_sentinels(text)
    text = fix_empty_table_cells(text)
    text = collapse_blank_lines(text)
    return text

# ===========================================================
# STAGE 2: METADATA EXTRACTION
# ===========================================================

RAW_DIR = Path("data/raw/sec-edgar-filings")

COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.",
    "META": "Meta Platforms, Inc.",
}

# SEC submission header lines look like:
#   FILED AS OF DATE:               20241101
#   CONFORMED PERIOD OF REPORT:     20240928
FILED_DATE_RE = re.compile(r"^FILED AS OF DATE:\s+(\d{8})", re.MULTILINE)
PERIOD_END_RE = re.compile(r"^CONFORMED PERIOD OF REPORT:\s+(\d{8})", re.MULTILINE)


def _yyyymmdd_to_iso(s: str) -> str:
    """20241101 -> 2024-11-01"""
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def parse_filename(filename: str) -> tuple[str, int]:
    """AAPL_2025.md -> ('AAPL', 2025). Raises if filename is malformed."""
    stem = Path(filename).stem  # 'AAPL_2025'
    ticker, year_str = stem.split("_")
    return ticker, int(year_str)


def find_filing_dates(ticker: str, fiscal_year: int) -> dict:
    """
    Walk data/raw/sec-edgar-filings/<TICKER>/10-K/<accession>/ folders,
    open each full-submission.txt header, and find the one whose
    CONFORMED PERIOD OF REPORT year matches our fiscal_year.

    Returns {'filing_date': 'YYYY-MM-DD', 'period_end': 'YYYY-MM-DD'} or {} if not found.
    """
    ticker_dir = RAW_DIR / ticker / "10-K"
    if not ticker_dir.exists():
        return {}

    for accession_folder in ticker_dir.iterdir():
        if not accession_folder.is_dir():
            continue
        submission_file = accession_folder / "full-submission.txt"
        if not submission_file.exists():
            continue

        # Only need the header — first ~50 lines is plenty
        with submission_file.open("r", encoding="utf-8", errors="ignore") as f:
            header = "".join(f.readline() for _ in range(50))

        filed_match = FILED_DATE_RE.search(header)
        period_match = PERIOD_END_RE.search(header)
        if not (filed_match and period_match):
            continue

        period_end = _yyyymmdd_to_iso(period_match.group(1))
        # Fiscal year in our filenames is the year of the period end.
        # E.g., AAPL_2025.md -> period_end starts with '2025'
        if not period_end.startswith(str(fiscal_year)):
            continue

        return {
            "filing_date": _yyyymmdd_to_iso(filed_match.group(1)),
            "period_end": period_end,
        }

    return {}


def build_file_metadata(filename: str) -> dict:
    """Assemble the per-file metadata dict that we'll attach to every chunk."""
    ticker, fiscal_year = parse_filename(filename)
    dates = find_filing_dates(ticker, fiscal_year)
    return {
        "ticker": ticker,
        "company_name": COMPANIES.get(ticker, ticker),
        "fiscal_year": fiscal_year,
        "filing_date": dates.get("filing_date"),
        "period_end": dates.get("period_end"),
        "source_file": filename,
    }



# ===========================================================
# STAGE 3: STRUCTURAL PARSING
# ===========================================================

# Heading detection: Markdown ATX style. We capture level and text.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Table row: starts with optional whitespace, then |
TABLE_ROW_RE = re.compile(r"^\s*\|")

# Table separator: |---|---| (the row that follows the header in a Markdown table)
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _classify_line(line: str) -> str:
    """Return one of: 'heading', 'table', 'blank', 'prose'."""
    stripped = line.strip()
    if not stripped:
        return "blank"
    if HEADING_RE.match(stripped):
        return "heading"
    if TABLE_ROW_RE.match(line):
        return "table"
    return "prose"


def parse_heading(line: str) -> tuple[int, str]:
    """'## Item 1A. Risk Factors' -> (2, 'Item 1A. Risk Factors')"""
    m = HEADING_RE.match(line.strip())
    if not m:
        return (0, "")
    return (len(m.group(1)), m.group(2).strip())


# Real SEC 10-K section pattern: "Item N", "Item NA", "Item NB", etc.,
# followed by a period and at least one word of body text. This filters out
# bare "Item 1" page-header fragments and non-SEC headings like "Apple Inc."
SEC_SECTION_RE = re.compile(
    r"^Item\s+\d+[A-Z]?\.\s+\S+",
    re.IGNORECASE,
)


def _is_real_section(heading_text: str) -> bool:
    """True if heading looks like a real SEC 10-K Item heading with body text."""
    return bool(SEC_SECTION_RE.match(heading_text))


# Map "smart" Unicode punctuation to ASCII equivalents so semantically
# identical headings compare equal across documents.
PUNCT_NORMALIZE = str.maketrans({
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote (curly apostrophe)
    "\u201C": '"',   # left double quote
    "\u201D": '"',   # right double quote
    "\u2013": "-",   # en-dash
    "\u2014": "-",   # em-dash
})


def _normalize_section(heading_text: str) -> str:
    """Normalize section text so semantically identical headings compare equal:
    - Apply ASCII punctuation (curly apostrophes -> straight, em-dash -> -)
    - Collapse internal whitespace
    - Title-case
    """
    text = heading_text.translate(PUNCT_NORMALIZE)
    text = " ".join(text.split())  # collapse multiple spaces
    return text.strip().title()


def split_into_blocks(text: str) -> list[dict]:
    """
    Walk the cleaned Markdown and produce a list of blocks. Each block is one of:
      {'type': 'heading', 'level': int, 'text': str, 'section': str, 'subsection': str}
      {'type': 'table',   'text': str,  'section': str, 'subsection': str}
      {'type': 'prose',   'text': str,  'section': str, 'subsection': str}

    Section context rules:
      - If a heading's TEXT matches the SEC section pattern (e.g.,
        'Item 1A. Risk Factors'), it sets `section` regardless of Markdown
        heading level. Different filers use different levels (MSFT uses H3,
        AAPL uses H2).
      - Other headings set `subsection`.
      - TOC entries (heading immediately followed by another heading) are
        demoted to prose so they don't update state.
      - Repeated identical headings don't reset state.
    """
    blocks: list[dict] = []
    lines = text.split("\n")

    current_section = ""
    current_subsection = ""

    def next_nonblank_kind(start_idx: int) -> str:
        j = start_idx
        while j < len(lines):
            k = _classify_line(lines[j])
            if k != "blank":
                return k
            j += 1
        return "blank"

    i = 0
    while i < len(lines):
        kind = _classify_line(lines[i])

        if kind == "blank":
            i += 1
            continue

        if kind == "heading":
            level, text_h = parse_heading(lines[i])
            following = next_nonblank_kind(i + 1)
            is_toc_entry = (following == "heading")

            if _is_real_section(text_h) and not is_toc_entry:
                # Real SEC section heading — irrespective of level
                normalized = _normalize_section(text_h)
                if normalized != current_section:
                    current_section = normalized
                    current_subsection = ""
                blocks.append({
                    "type": "heading",
                    "level": level,
                    "text": text_h,
                    "section": current_section,
                    "subsection": current_subsection,
                })
            elif not is_toc_entry:
                # Not a SEC section, but a real subsection heading
                normalized = _normalize_section(text_h)
                if normalized != current_subsection:
                    current_subsection = normalized
                blocks.append({
                    "type": "heading",
                    "level": level,
                    "text": text_h,
                    "section": current_section,
                    "subsection": current_subsection,
                })
            else:
                # TOC entry — demote to prose so it doesn't update state
                blocks.append({
                    "type": "prose",
                    "text": text_h,
                    "section": current_section,
                    "subsection": current_subsection,
                })

            i += 1
            continue

        if kind == "table":
            start = i
            while i < len(lines) and _classify_line(lines[i]) == "table":
                i += 1
            table_text = "\n".join(lines[start:i])
            blocks.append({
                "type": "table",
                "text": table_text,
                "section": current_section,
                "subsection": current_subsection,
            })
            continue

        # prose
        start = i
        while i < len(lines) and _classify_line(lines[i]) == "prose":
            i += 1
        prose_text = "\n".join(lines[start:i]).strip()
        if prose_text:
            blocks.append({
                "type": "prose",
                "text": prose_text,
                "section": current_section,
                "subsection": current_subsection,
            })

    return blocks



# ===========================================================
# STAGE 4: CHUNKING
# ===========================================================

TARGET_TOKENS = 384      # aim for chunks in the 256-512 range
MAX_TOKENS = 512         # hard ceiling for prose chunks
MIN_TOKENS = 80          # below this, try to merge forward
OVERLAP_TOKENS = 75      # ~20% overlap between adjacent prose chunks


def _split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter. Good enough for SEC prose.
    Splits on '. ', '! ', '? ', and newlines, preserving the delimiters."""
    # First split on paragraph breaks, then on sentence punctuation.
    sentences: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        # Split on sentence enders followed by space + capital letter.
        # Keep it simple — over-split is fine, under-split is not.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", para)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def _take_overlap(text: str, overlap_tokens: int) -> str:
    """Return the last `overlap_tokens` tokens of text, snapped to a
    sentence boundary if possible."""
    if not text:
        return ""
    sentences = _split_into_sentences(text)
    if not sentences:
        return ""
    # Walk backward, accumulating sentences until we hit the overlap budget.
    accum: list[str] = []
    accum_tokens = 0
    for s in reversed(sentences):
        s_tokens = count_tokens(s)
        if accum_tokens + s_tokens > overlap_tokens and accum:
            break
        accum.insert(0, s)
        accum_tokens += s_tokens
    return " ".join(accum)


def _pack_prose_chunks(prose_blocks: list[dict]) -> list[dict]:
    """Take a list of consecutive prose blocks (same section) and pack them
    into chunks targeting TARGET_TOKENS, with overlap between adjacent chunks.

    Each prose block may already include a subsection-heading prefix.
    """
    chunks: list[dict] = []
    if not prose_blocks:
        return chunks

    # First, flatten the prose blocks into a single ordered sentence list,
    # remembering each sentence's source block (so we know its subsection).
    flat: list[tuple[str, dict]] = []
    for b in prose_blocks:
        for s in _split_into_sentences(b["text"]):
            flat.append((s, b))

    if not flat:
        return chunks

    current_sentences: list[str] = []
    current_tokens = 0
    current_meta_block = flat[0][1]  # remember which block contributed most

    def close_chunk(carry_overlap: bool):
        nonlocal current_sentences, current_tokens, current_meta_block
        if not current_sentences:
            return
        text = " ".join(current_sentences).strip()
        if count_tokens(text) >= MIN_TOKENS or not chunks:
            # Always emit if it's the first chunk, even if small.
            chunks.append({
                "text": text,
                "section": current_meta_block["section"],
                "subsection": current_meta_block["subsection"],
                "chunk_type": "prose",
            })
        # Reset, optionally carrying forward overlap text
        if carry_overlap and text:
            overlap = _take_overlap(text, OVERLAP_TOKENS)
            current_sentences = [overlap] if overlap else []
            current_tokens = count_tokens(overlap) if overlap else 0
        else:
            current_sentences = []
            current_tokens = 0

    for sentence, src_block in flat:
        s_tokens = count_tokens(sentence)
        # If a single sentence is freakishly large (>MAX_TOKENS), emit it alone.
        if s_tokens > MAX_TOKENS:
            close_chunk(carry_overlap=False)
            chunks.append({
                "text": sentence,
                "section": src_block["section"],
                "subsection": src_block["subsection"],
                "chunk_type": "prose",
            })
            continue

        # Would adding this sentence push us over the ceiling?
        if current_tokens + s_tokens > MAX_TOKENS:
            close_chunk(carry_overlap=True)

        current_sentences.append(sentence)
        current_tokens += s_tokens
        current_meta_block = src_block  # latest block wins for metadata

        # Once we've crossed the target size, close the chunk so the next
        # one can start fresh (with overlap).
        if current_tokens >= TARGET_TOKENS:
            close_chunk(carry_overlap=True)

    # Flush the tail
    close_chunk(carry_overlap=False)
    return chunks


def chunks_from_blocks(blocks: list[dict], file_metadata: dict) -> list[dict]:
    """Convert block stream into final chunks with file-level metadata attached.

    - Tables: each becomes a single atomic chunk.
    - Prose runs (consecutive prose blocks within a section): packed into
      ~TARGET_TOKENS chunks with overlap.
    - Pending subsection headings are prepended to the next prose run, so a
      chunk for the 'Goodwill' subsection starts with 'Goodwill\n\n...'.
    - Section boundaries always close any open prose run.
    """
    final_chunks: list[dict] = []
    prose_run: list[dict] = []
    current_section = None

    def flush_prose():
        nonlocal prose_run
        if not prose_run:
            return
        for c in _pack_prose_chunks(prose_run):
            final_chunks.append(c)
        prose_run = []

    pending_subsection_header = None  # subsection heading awaiting prose

    for b in blocks:
        # Detect section boundary
        if b.get("section", "") != current_section:
            flush_prose()
            current_section = b.get("section", "")
            pending_subsection_header = None

        btype = b["type"]
        if btype == "table":
            flush_prose()
            final_chunks.append({
                "text": b["text"],
                "section": b["section"],
                "subsection": b["subsection"],
                "chunk_type": "table",
            })
            pending_subsection_header = None

        elif btype == "heading":
            # Section headings already handled above. Subsection headings get
            # held to prepend to the next prose block.
            if not _is_real_section(b["text"]):
                pending_subsection_header = b["text"]

        elif btype == "prose":
            text = b["text"]
            if pending_subsection_header:
                text = f"{pending_subsection_header}\n\n{text}"
                pending_subsection_header = None
            prose_run.append({
                **b,
                "text": text,
            })

    flush_prose()

    # Attach file-level metadata to every chunk
    for c in final_chunks:
        c["metadata"] = {
            **file_metadata,
            "section": c.pop("section"),
            "subsection": c.pop("subsection"),
            "chunk_type": c.pop("chunk_type"),
            "n_tokens": count_tokens(c["text"]),
        }

    return final_chunks


def chunk_file(filename: str) -> list[dict]:
    """End-to-end: load, preprocess, parse, chunk, attach metadata."""
    path = PARSED_DIR / filename
    raw = path.read_text(encoding="utf-8")
    cleaned = preprocess(raw)
    blocks = split_into_blocks(cleaned)
    file_meta = build_file_metadata(filename)
    return chunks_from_blocks(blocks, file_meta)



# ===========================================================
# STAGE 5: WRITE ALL CHUNKS TO chunks.jsonl
# ===========================================================

import json


def chunk_all_files() -> None:
    """Process all 10 parsed Markdown files and write to data/chunks.jsonl."""
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    per_file_stats: list[tuple[str, int, dict]] = []

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for md_file in sorted(PARSED_DIR.glob("*.md")):
            chunks = chunk_file(md_file.name)
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

            # Per-file breakdown
            by_type: dict[str, int] = {}
            for c in chunks:
                t = c["metadata"]["chunk_type"]
                by_type[t] = by_type.get(t, 0) + 1
            per_file_stats.append((md_file.name, len(chunks), by_type))
            total_chunks += len(chunks)
            print(f"  {md_file.name:25s}  {len(chunks):>4d} chunks  "
                  f"(prose {by_type.get('prose', 0)}, table {by_type.get('table', 0)})")

    print(f"\n  TOTAL: {total_chunks:,} chunks written to {CHUNKS_PATH}")



# ===========================================================
# Test: chunk one file and print stats
# ===========================================================

def _test_chunk(filename: str) -> None:
    chunks = chunk_file(filename)

    print(f"=== Chunking test: {filename} ===")
    print(f"  Total chunks: {len(chunks):,}")

    # Type breakdown
    by_type: dict[str, int] = {}
    for c in chunks:
        t = c["metadata"]["chunk_type"]
        by_type[t] = by_type.get(t, 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"    {t:6s}  {n:>5,}")

    # Token statistics
    token_counts = [c["metadata"]["n_tokens"] for c in chunks]
    print(f"  Token stats:")
    print(f"    min:    {min(token_counts):>5,}")
    print(f"    max:    {max(token_counts):>5,}")
    print(f"    avg:    {sum(token_counts) / len(token_counts):>7.1f}")

    # Distribution buckets
    buckets = {"<100": 0, "100-256": 0, "256-512": 0, "512-1024": 0, ">1024": 0}
    for t in token_counts:
        if t < 100: buckets["<100"] += 1
        elif t < 256: buckets["100-256"] += 1
        elif t < 512: buckets["256-512"] += 1
        elif t < 1024: buckets["512-1024"] += 1
        else: buckets[">1024"] += 1
    print(f"  Size buckets:")
    for k, v in buckets.items():
        print(f"    {k:10s}  {v:>5,}")

    # Sections covered
    sections_seen = sorted({c["metadata"]["section"] for c in chunks if c["metadata"]["section"]})
    print(f"  Sections with chunks: {len(sections_seen)}")

    # Sample 2 prose and 2 tables
    proses = [c for c in chunks if c["metadata"]["chunk_type"] == "prose"]
    tables = [c for c in chunks if c["metadata"]["chunk_type"] == "table"]

    if proses:
        sample_p = proses[len(proses) // 2]
        print(f"\n  Sample prose chunk:")
        print(f"    section:    {sample_p['metadata']['section']}")
        print(f"    subsection: {sample_p['metadata']['subsection']}")
        print(f"    n_tokens:   {sample_p['metadata']['n_tokens']}")
        print(f"    text (first 200 chars):")
        print(f"      {sample_p['text'][:200]}")

    if tables:
        sample_t = tables[len(tables) // 2]
        print(f"\n  Sample table chunk:")
        print(f"    section:    {sample_t['metadata']['section']}")
        print(f"    n_tokens:   {sample_t['metadata']['n_tokens']}")
        print(f"    text (first 200 chars):")
        print(f"      {sample_t['text'][:200]}")

    # Show extremes — the smallest and largest chunks
    sorted_by_size = sorted(chunks, key=lambda c: c["metadata"]["n_tokens"])
    print(f"\n  Smallest 5 chunks:")
    for c in sorted_by_size[:5]:
        n = c["metadata"]["n_tokens"]
        t = c["metadata"]["chunk_type"]
        sect = c["metadata"]["section"][:40]
        preview = c["text"][:120].replace("\n", " ")
        print(f"    [{n:>4d} tok | {t:5s} | {sect}] {preview}")

    print(f"\n  Largest 5 chunks:")
    for c in sorted_by_size[-5:]:
        n = c["metadata"]["n_tokens"]
        t = c["metadata"]["chunk_type"]
        sect = c["metadata"]["section"][:40]
        preview = c["text"][:120].replace("\n", " ")
        print(f"    [{n:>4d} tok | {t:5s} | {sect}] {preview}")

# ===========================================================
# TEMPORARY: test structural parsing on one file
# ===========================================================

def _test_structure(filename: str) -> None:
    """Load one file, preprocess, parse into blocks, print summary stats."""
    path = PARSED_DIR / filename
    if not path.exists():
        print(f"  File not found: {path}")
        return

    raw = path.read_text(encoding="utf-8")
    cleaned = preprocess(raw)
    blocks = split_into_blocks(cleaned)

    # Aggregate
    type_counts: dict[str, int] = {}
    for b in blocks:
        type_counts[b["type"]] = type_counts.get(b["type"], 0) + 1

    print(f"=== Structural parse: {filename} ===")
    print(f"  Total blocks:  {len(blocks):,}")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:8s}  {c:>5,}")

    # Show all H2 headings (Item 1, 1A, 7, 8 etc.) — gives us a feel for section coverage
    # Show DISTINCT real section names (not page-header repeats)
    print(f"\n  Distinct SEC sections found:")
    distinct_sections: list[str] = []
    for b in blocks:
        s = b.get("section", "")
        if s and s not in distinct_sections:
            distinct_sections.append(s)
    for s in distinct_sections:
        print(f"    SECTION: {s[:80]}")
    print(f"  Total distinct sections: {len(distinct_sections)}")

    # Show distinct subsections
    print(f"\n  Distinct subsections found:")
    distinct_subs: list[str] = []
    for b in blocks:
        s = b.get("subsection", "")
        if s and s not in distinct_subs:
            distinct_subs.append(s)
    for s in distinct_subs[:15]:
        print(f"    SUB: {s[:80]}")
    if len(distinct_subs) > 15:
        print(f"    ... and {len(distinct_subs) - 15} more")

    # Show 2 sample tables and 2 sample prose blocks
    print(f"\n  Sample table block (truncated):")
    tables = [b for b in blocks if b["type"] == "table"]
    if tables:
        sample = tables[len(tables) // 2]  # pick one from the middle
        print(f"    section: {sample['section']}")
        print(f"    text (first 300 chars):")
        for line in sample["text"][:300].split("\n"):
            print(f"      {line}")

    print(f"\n  Sample prose block (truncated):")
    proses = [b for b in blocks if b["type"] == "prose"]
    if proses:
        sample = proses[len(proses) // 2]
        print(f"    section: {sample['section']}")
        print(f"    subsection: {sample['subsection']}")
        print(f"    text (first 300 chars):")
        for line in sample["text"][:300].split("\n"):
            print(f"      {line}")




# ===========================================================
# TEMPORARY: test preprocessing on one file
# ===========================================================

def _test_preprocess(filename: str) -> None:
    """Quick smoke test: load one file, preprocess, print before/after stats."""
    path = PARSED_DIR / filename
    if not path.exists():
        print(f"  File not found: {path}")
        return

    raw = path.read_text(encoding="utf-8")
    cleaned = preprocess(raw)

    print(f"=== Preprocessing test: {filename} ===")
    print(f"  Raw chars:        {len(raw):,}")
    print(f"  Cleaned chars:    {len(cleaned):,}")
    print(f"  Raw tokens:       {count_tokens(raw):,}")
    print(f"  Cleaned tokens:   {count_tokens(cleaned):,}")
    print(f"  Diff chars:       {len(raw) - len(cleaned):,} removed")

    # Show a quick sanity check: any HTML entities left?
    leftover_entities = re.findall(r"&#?\w+;", cleaned)
    print(f"  HTML entities remaining: {len(leftover_entities)}")
    if leftover_entities[:5]:
        print(f"    examples: {leftover_entities[:5]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-preprocess", type=str)
    ap.add_argument("--test-metadata", action="store_true")
    ap.add_argument("--test-structure", type=str)
    ap.add_argument("--test-chunk", type=str)
    ap.add_argument("--all", action="store_true",
                    help="Chunk all parsed files and write data/chunks.jsonl")
    args = ap.parse_args()

    if args.test_preprocess:
        _test_preprocess(args.test_preprocess)
    elif args.test_metadata:
        for md_file in sorted(PARSED_DIR.glob("*.md")):
            print(build_file_metadata(md_file.name))
    elif args.test_structure:
        _test_structure(args.test_structure)
    elif args.test_chunk:
        _test_chunk(args.test_chunk)
    elif args.all:
        chunk_all_files()
    else:
        print("Usage: --test-preprocess F | --test-metadata | --test-structure F | --test-chunk F | --all")