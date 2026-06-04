"""
src/ingest.py

Downloads 10-K filings from SEC EDGAR for our target companies.
Uses sec-edgar-downloader which wraps the official SEC EDGAR API.

Run this from the project root: python -m src.ingest
"""
from pathlib import Path
from sec_edgar_downloader import Downloader

# === Configuration ===
COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla, Inc.",
    "META": "Meta Platforms, Inc.",
}

# How many of the most recent 10-Ks to download per company
NUM_FILINGS_PER_COMPANY = 2

# Where to save them
RAW_DATA_DIR = Path("data/raw")

# SEC EDGAR requires you to identify yourself. Any real-looking email works.
USER_AGENT_NAME = "Financial RAG Project"
USER_AGENT_EMAIL = "shafin.tasfi@gmail.com.com"  # change to your actual email


def download_filings():
    """Download the most recent N 10-K filings for each company."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dl = Downloader(USER_AGENT_NAME, USER_AGENT_EMAIL, RAW_DATA_DIR)

    for ticker, company_name in COMPANIES.items():
        print(f"\nDownloading {NUM_FILINGS_PER_COMPANY} most recent 10-Ks for {ticker} ({company_name})...")
        try:
            num_downloaded = dl.get(
                "10-K",
                ticker,
                limit=NUM_FILINGS_PER_COMPANY,
                download_details=False,  # we just want the primary filing document
            )
            print(f"  Downloaded {num_downloaded} filings for {ticker}")
        except Exception as e:
            print(f"  ERROR for {ticker}: {e}")

    print("\nAll downloads complete. Files saved under data/raw/sec-edgar-filings/")


if __name__ == "__main__":
    download_filings()