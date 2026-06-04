"""
src/parse.py

Parses SEC 10-K HTML filings into clean Markdown using LlamaParse.
LlamaParse is purpose-built for complex documents with tables.

Two modes:
  - parse_one(ticker): parse the most recent filing for one company (for testing)
  - parse_all(): parse all filings for all companies (production run)

Run from project root:
  python -m src.parse --one AAPL     # parse only the most recent AAPL filing
  python -m src.parse --all          # parse everything
"""
import argparse
import os
import re
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

# === Configuration ===
RAW_DATA_DIR = Path("data/raw/sec-edgar-filings")
PARSED_DATA_DIR = Path("data/parsed")
PARSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "META"]


def get_accession_year(accession_folder: Path) -> int:
    """
    Extract the fiscal year from the SEC submission header.
    Uses CONFORMED PERIOD OF REPORT (period end date), which is the
    correct fiscal year regardless of when the filing was submitted.

    Fallback: parse the accession number year if header is missing.
    """
    submission_file = accession_folder / "full-submission.txt"
    if submission_file.exists():
        with submission_file.open("r", encoding="utf-8", errors="ignore") as f:
            header = "".join(f.readline() for _ in range(50))
        match = re.search(r"^CONFORMED PERIOD OF REPORT:\s+(\d{4})\d{4}",
                          header, re.MULTILINE)
        if match:
            return int(match.group(1))

    # Fallback: accession number year (less reliable)
    match = re.match(r"\d+-(\d{2})-\d+", accession_folder.name)
    if match:
        yy = int(match.group(1))
        return 2000 + yy if yy < 50 else 1900 + yy
    return 0


def find_primary_html(filing_folder: Path) -> Path | None:
    """Locate the primary-document.html inside an accession folder."""
    candidates = list(filing_folder.glob("*.html"))
    if candidates:
        return candidates[0]
    return None


def parse_filing(html_path: Path, ticker: str, filing_year: int) -> Path:
    """Parse one HTML filing into a Markdown file. Returns output path."""
    output_path = PARSED_DATA_DIR / f"{ticker}_{filing_year}.md"

    if output_path.exists():
        print(f"  Skipping {ticker}_{filing_year} — already parsed")
        return output_path

    print(f"  Parsing {ticker}_{filing_year} from {html_path.name}...")
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=False,
        # Hint for SEC filings — improves table extraction
        parsing_instruction=(
            "This is a SEC 10-K annual report. Preserve all tables exactly as they appear, "
            "including column headers and row labels. Keep section headings (Item 1, Item 1A, etc) "
            "as Markdown H2 headings. Preserve numerical precision exactly."
        ),
    )

    documents = parser.load_data(str(html_path))
    markdown_content = "\n\n".join(doc.text for doc in documents)

    output_path.write_text(markdown_content, encoding="utf-8")
    print(f"  Saved {output_path} ({len(markdown_content):,} characters)")
    return output_path


def find_filings_for_ticker(ticker: str):
    """Find all filing folders for a ticker, sorted newest first."""
    ticker_dir = RAW_DATA_DIR / ticker / "10-K"
    if not ticker_dir.exists():
        print(f"  No filings found for {ticker}")
        return []

    filings = []
    for folder in ticker_dir.iterdir():
        if folder.is_dir():
            year = get_accession_year(folder.name)
            html_path = find_primary_html(folder)
            if html_path and year:
                filings.append((year, html_path))

    # Sort by year, newest first
    filings.sort(key=lambda x: x[0], reverse=True)
    return filings


def parse_one(ticker: str):
    """Parse only the most recent filing for a single ticker (for testing)."""
    print(f"\nParsing most recent filing for {ticker}...")
    filings = find_filings_for_ticker(ticker)
    if not filings:
        print(f"  No filings found for {ticker}")
        return

    year, html_path = filings[0]
    parse_filing(html_path, ticker, year)


def parse_all():
    """Parse all filings for all tickers."""
    for ticker in TICKERS:
        print(f"\n=== {ticker} ===")
        filings = find_filings_for_ticker(ticker)
        for year, html_path in filings:
            parse_filing(html_path, ticker, year)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", type=str, help="Parse most recent filing for this ticker only (e.g., AAPL)")
    ap.add_argument("--all", action="store_true", help="Parse all filings for all tickers")
    args = ap.parse_args()

    if args.one:
        parse_one(args.one.upper())
    elif args.all:
        parse_all()
    else:
        print("Usage: python -m src.parse --one AAPL   OR   python -m src.parse --all")