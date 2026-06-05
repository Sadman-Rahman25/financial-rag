"""
src/retrieve.py — Day 5+6: BM25 + dense + RRF hybrid + cross-encoder rerank
                          + metadata pre-filtering (Day 6 Stage 4).

For low-RAM (8 GB) machines, rerank/compare modes use a two-phase strategy:
  Phase 1: retrieve candidates via dense/BM25/hybrid (uses embedder)
  Phase 2: free Retriever (releases embedder), then load + run reranker

Usage:
    python -m src.retrieve --build-bm25
    python -m src.retrieve --query "..." --mode dense   [filters]
    python -m src.retrieve --query "..." --mode bm25    [filters]
    python -m src.retrieve --query "..." --mode hybrid  [filters]
    python -m src.retrieve --query "..." --mode rerank  [filters]
    python -m src.retrieve --query "..." --mode compare [filters]

Filters: --ticker AAPL, --fiscal-year 2024, --chunk-type prose|table, --auto-filter
"""
import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.index import (
    CHUNKS_PATH,
    COLLECTION_NAME,
    get_qdrant_client,
    get_embedder,
)


RRF_K = 60
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
RERANK_CANDIDATES = 20


# =============== Day 6 Stage 4: metadata pre-filtering =============== #

_TICKER_KEYWORDS = {
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT",
    "nvidia": "NVDA", "nvda": "NVDA",
    "tesla": "TSLA", "tsla": "TSLA",
    "meta": "META", "facebook": "META",
}

def extract_filters(query: str) -> dict:
    """Pull ticker/fiscal_year hints from a natural-language query.

    Conservative rules:
      - Ticker is set only if exactly ONE company keyword is detected.
        Multiple tickers (e.g. "Compare Apple and Microsoft") → no ticker filter.
      - Year matches 2020-2029, optionally prefixed with "FY".

    Returns dict with optional 'ticker' and 'fiscal_year' keys (empty if none).
    """
    detected: dict = {}

    q_lower = query.lower()
    matches = set()
    for kw, ticker in _TICKER_KEYWORDS.items():
        if re.search(rf"\b{kw}\b", q_lower):
            matches.add(ticker)
    if len(matches) == 1:
        detected["ticker"] = matches.pop()

    # Match "2024", "FY2024", "fiscal year 2024" → 2024
    year_match = re.search(r"(?:^|\W)(?:fy)?(202\d)\b", query, re.IGNORECASE)
    if year_match:
        detected["fiscal_year"] = int(year_match.group(1))

    return detected


# =============== tokenizer + BM25 helpers =============== #

_SPLIT_RE = re.compile(r"\W+")

def tokenize(text: str) -> list[str]:
    return [t for t in _SPLIT_RE.split(text.lower()) if len(t) >= 2]


def _load_chunks() -> list[dict]:
    chunks = []
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _smoke_test_bm25() -> None:
    print("Loading chunks...", flush=True)
    chunks = _load_chunks()
    print(f"  Loaded {len(chunks)} chunks.", flush=True)
    sample = chunks[0]["text"][:200]
    print(f"\nSample text (first 200 chars):\n  {sample!r}")
    print(f"\nTokenized (first 20 tokens):\n  {tokenize(sample)[:20]}")
    print(f"\nTokenizing {len(chunks)} chunks...", flush=True)
    t = time.time()
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    print(f"  Tokenized in {time.time()-t:.2f}s", flush=True)
    print(f"Building BM25 index...", flush=True)
    t = time.time()
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  Built in {time.time()-t:.2f}s", flush=True)
    query = "iPhone revenue"
    print(f"\nSmoke-test query: {query!r}")
    query_tokens = tokenize(query)
    print(f"  Query tokens: {query_tokens}")
    scores = bm25.get_scores(query_tokens)
    top5_idx = np.argsort(-scores)[:5]
    print(f"\nTop 5 BM25 hits:")
    for rank, idx in enumerate(top5_idx, 1):
        meta = chunks[idx]["metadata"]
        text = chunks[idx]["text"][:150].replace("\n", " ").strip()
        print(f"  [{rank}] score={scores[idx]:.3f} | {meta.get('ticker')} "
              f"FY{meta.get('fiscal_year')} | {meta.get('section', '')[:55]}")
        print(f"      {text}...")
    print("\nOK: BM25 index builds and produces plausible results.")


# =============== RRF =============== #

def rrf_fuse(ranked_lists, list_names, k=RRF_K):
    assert len(ranked_lists) == len(list_names)
    fused: dict[int, dict] = {}
    for retriever_name, results in zip(list_names, ranked_lists):
        for rank, r in enumerate(results):
            doc_id = r["id"]
            if doc_id not in fused:
                fused[doc_id] = {
                    "id": doc_id,
                    "score": 0.0,
                    "payload": r["payload"],
                    "sources": [],
                }
            fused[doc_id]["score"] += 1.0 / (k + rank + 1)
            fused[doc_id]["sources"].append(f"{retriever_name}#{rank+1}")
    return sorted(fused.values(), key=lambda x: -x["score"])


# =============== Retriever =============== #

class Retriever:
    def __init__(self) -> None:
        print("Initializing retriever...", flush=True)
        t0 = time.time()
        self.chunks = _load_chunks()
        print(f"  Loaded {len(self.chunks)} chunks", flush=True)
        t = time.time()
        tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"  Built BM25 index in {time.time()-t:.2f}s", flush=True)
        self.client = get_qdrant_client()
        print(f"  Qdrant client opened", flush=True)
        self.embedder = get_embedder()
        print(f"Retriever ready ({time.time()-t0:.1f}s total)\n", flush=True)

    def _build_qdrant_filter(self, ticker, fiscal_year, chunk_type) -> Optional[Filter]:
        conditions = []
        if ticker:
            conditions.append(FieldCondition(key="ticker", match=MatchValue(value=ticker)))
        if fiscal_year is not None:
            conditions.append(FieldCondition(key="fiscal_year", match=MatchValue(value=fiscal_year)))
        if chunk_type:
            conditions.append(FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type)))
        return Filter(must=conditions) if conditions else None

    def _passes_filter(self, meta, ticker, fiscal_year, chunk_type):
        if ticker and meta.get("ticker") != ticker:
            return False
        if fiscal_year is not None and meta.get("fiscal_year") != fiscal_year:
            return False
        if chunk_type and meta.get("chunk_type") != chunk_type:
            return False
        return True

    def dense_search(self, query, limit=5, ticker=None, fiscal_year=None, chunk_type=None):
        vec = self.embedder.encode(query, normalize_embeddings=True)
        qfilter = self._build_qdrant_filter(ticker, fiscal_year, chunk_type)
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec.tolist(),
            limit=limit,
            query_filter=qfilter,
        )
        return [
            {"id": int(p.id), "score": float(p.score), "payload": p.payload}
            for p in response.points
        ]

    def bm25_search(self, query, limit=5, ticker=None, fiscal_year=None, chunk_type=None):
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked_idx = np.argsort(-scores)
        results = []
        for idx in ranked_idx:
            chunk = self.chunks[idx]
            meta = chunk["metadata"]
            if not self._passes_filter(meta, ticker, fiscal_year, chunk_type):
                continue
            payload = dict(meta)
            payload["text"] = chunk["text"]
            results.append({
                "id": int(idx),
                "score": float(scores[idx]),
                "payload": payload,
            })
            if len(results) >= limit:
                break
        return results

    def hybrid_search(self, query, limit=5, ticker=None, fiscal_year=None, chunk_type=None):
        dense = self.dense_search(
            query, limit=DENSE_TOP_K,
            ticker=ticker, fiscal_year=fiscal_year, chunk_type=chunk_type,
        )
        sparse = self.bm25_search(
            query, limit=SPARSE_TOP_K,
            ticker=ticker, fiscal_year=fiscal_year, chunk_type=chunk_type,
        )
        fused = rrf_fuse([dense, sparse], ["dense", "bm25"], k=RRF_K)
        return fused[:limit]


# =============== pretty-print =============== #

def print_results(results, mode):
    print(f"\nTop {len(results)} results ({mode}):")
    print("-" * 100)
    for i, r in enumerate(results, 1):
        meta = r["payload"]
        text = meta.get("text", "")
        preview = text[:240].replace("\n", " ").strip()
        section = meta.get("section", "")[:90]
        if "rerank_score" in r:
            score_str = f"rerank={r['rerank_score']:.4f} rrf={r['score']:.4f}"
        else:
            score_str = f"score={r['score']:.4f}"
        sources_str = ""
        if r.get("sources"):
            sources_str = f"  [from: {', '.join(r['sources'])}]"
        print(f"[{i}] id={r['id']} {score_str}{sources_str} | "
              f"{meta.get('ticker')} FY{meta.get('fiscal_year')} | "
              f"{meta.get('chunk_type')} | {meta.get('n_tokens')} tok")
        print(f"    Section:    {section}")
        if meta.get("subsection"):
            print(f"    Subsection: {meta['subsection']}")
        print(f"    Text:       {preview}...")
        print()


def print_compare(by_mode):
    for mode in ("dense", "bm25", "hybrid", "rerank"):
        if mode not in by_mode:
            continue
        print("\n" + "=" * 100)
        print(f"=== MODE: {mode.upper()} ===")
        print("=" * 100)
        print_results(by_mode[mode], mode=mode)


# =============== CLI =============== #

def _resolve_filters(args) -> dict:
    """Merge explicit CLI filters with --auto-filter detection. Explicit wins."""
    filters = dict(
        ticker=args.ticker,
        fiscal_year=args.fiscal_year,
        chunk_type=args.chunk_type,
    )
    if args.auto_filter and args.query:
        auto = extract_filters(args.query)
        if auto:
            print(f"Auto-detected from query: {auto}", flush=True)
            for k, v in auto.items():
                if filters.get(k) is None:
                    filters[k] = v
            print(f"Final filters: { {k: v for k, v in filters.items() if v is not None} }\n",
                  flush=True)
        else:
            print(f"Auto-filter: nothing detected from query.\n", flush=True)
    return filters


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-bm25", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--mode", type=str, default="dense",
                        choices=["dense", "bm25", "hybrid", "rerank", "compare"])
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--fiscal-year", type=int, default=None)
    parser.add_argument("--chunk-type", type=str, default=None, choices=[None, "prose", "table"])
    parser.add_argument("--auto-filter", action="store_true",
                        help="Auto-detect ticker/fiscal_year from query text")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    if args.build_bm25:
        _smoke_test_bm25()
    elif args.query is None:
        parser.print_help()
    else:
        filter_kwargs = _resolve_filters(args)

        if args.mode in ("dense", "bm25", "hybrid"):
            retriever = Retriever()
            full_kwargs = dict(limit=args.limit, **filter_kwargs)
            if args.mode == "dense":
                print_results(retriever.dense_search(args.query, **full_kwargs), mode="dense")
            elif args.mode == "bm25":
                print_results(retriever.bm25_search(args.query, **full_kwargs), mode="bm25")
            else:
                print_results(retriever.hybrid_search(args.query, **full_kwargs), mode="hybrid")

        elif args.mode == "rerank":
            retriever = Retriever()
            candidates = retriever.hybrid_search(
                args.query, limit=RERANK_CANDIDATES, **filter_kwargs
            )
            print("Freeing retriever (embedder + BM25) to make room for reranker...", flush=True)
            del retriever
            gc.collect()
            from src.rerank import Reranker
            reranker = Reranker()
            results = reranker.rerank(args.query, candidates, top_k=args.limit)
            print_results(results, mode="rerank")

        elif args.mode == "compare":
            retriever = Retriever()
            full_kwargs = dict(limit=args.limit, **filter_kwargs)
            by_mode = {
                "dense":  retriever.dense_search(args.query, **full_kwargs),
                "bm25":   retriever.bm25_search(args.query, **full_kwargs),
                "hybrid": retriever.hybrid_search(args.query, **full_kwargs),
            }
            rerank_candidates = retriever.hybrid_search(
                args.query, limit=RERANK_CANDIDATES, **filter_kwargs
            )
            print("\nFreeing retriever (embedder + BM25) to make room for reranker...", flush=True)
            del retriever
            gc.collect()
            from src.rerank import Reranker
            reranker = Reranker()
            by_mode["rerank"] = reranker.rerank(args.query, rerank_candidates, top_k=args.limit)
            print_compare(by_mode)