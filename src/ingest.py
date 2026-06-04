"""
src/ingest.py

Downloads 10-K filings from SEC EDGAR for our target companies.
Downloads BOTH the full submission and the primary 10-K HTML document,
which is what we'll actually parse.

Run from project root: python -m src.ingest
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

NUM_FILINGS_PER_COMPANY = 2
RAW_DATA_DIR = Path("data/raw")
USER_AGENT_NAME = "Financial RAG Project"
USER_AGENT_EMAIL = "your-email@example.com"  # change to your real email


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
                download_details=True,  # also downloads the primary .htm document
            )
            print(f"  Downloaded {num_downloaded} filings for {ticker}")
        except Exception as e:
            print(f"  ERROR for {ticker}: {e}")

    print("\nAll downloads complete.")


if __name__ == "__main__":
    download_filings()