"""
src/retrieve.py — Day 5: BM25 + dense + RRF hybrid retrieval.

Usage:
    python -m src.retrieve --build-bm25                              # Stage 1 smoke test
    python -m src.retrieve --query "..." --mode dense  [filters]
    python -m src.retrieve --query "..." --mode bm25   [filters]
    python -m src.retrieve --query "..." --mode hybrid [filters]
    python -m src.retrieve --query "..." --mode compare [filters]    # side-by-side

Filters: --ticker AAPL, --fiscal-year 2024, --chunk-type prose|table, --limit N
"""
import argparse
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


# Locked-in hyperparameters for Day 5
RRF_K = 60          # Cormack et al. 2009 default
DENSE_TOP_K = 20    # candidates pulled from dense before fusion
SPARSE_TOP_K = 20   # candidates pulled from BM25 before fusion


# =============== Stage 1: tokenizer + BM25 helpers =============== #

_SPLIT_RE = re.compile(r"\W+")

def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, drop tokens shorter than 2 chars.

    Preserves: 'iPhone' -> 'iphone', 'GAAP' -> 'gaap', 'FY2024' -> 'fy2024'.
    Drops: 'U.S.' (becomes 'u','s' both <2 char), '$7.7' (becomes '7','7' both <2 char).
    No stemming, no stopwords.
    """
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
    """Stage 1 verification: build BM25, run a tiny test query."""
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


# =============== Stage 3: RRF fusion =============== #

def rrf_fuse(
    ranked_lists: list[list[dict]],
    list_names: list[str],
    k: int = RRF_K,
) -> list[dict]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    For each ranked list, walk by rank r (0-indexed). For each doc at rank r,
    add 1 / (k + r + 1) to its accumulated score. Sort docs by accumulated
    score descending. Docs appearing in MULTIPLE lists get boosted.

    Args:
        ranked_lists: list of result lists; each item is {id, score, payload}
        list_names:   parallel list of human names (e.g. ["dense", "bm25"])
        k:            RRF damping constant (60 is the de facto standard)

    Returns:
        Fused list, each item: {id, score (RRF), payload, sources (debug)}
    """
    assert len(ranked_lists) == len(list_names), "names must match lists"
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


# =============== Stage 2 + 3: Retriever class =============== #

class Retriever:
    """Holds the embedder, BM25 index, and Qdrant client. Build once, reuse."""

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

    def _passes_filter(self, meta: dict, ticker, fiscal_year, chunk_type) -> bool:
        if ticker and meta.get("ticker") != ticker:
            return False
        if fiscal_year is not None and meta.get("fiscal_year") != fiscal_year:
            return False
        if chunk_type and meta.get("chunk_type") != chunk_type:
            return False
        return True

    def dense_search(self, query, limit=5, ticker=None, fiscal_year=None, chunk_type=None):
        """Semantic search via BGE-large + Qdrant."""
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
        """Keyword search via BM25. Filters applied post-hoc in Python."""
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
        """Run dense + BM25 (each pulling top-K candidates), fuse with RRF, return top-`limit`."""
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

    def compare(self, query, limit=5, ticker=None, fiscal_year=None, chunk_type=None):
        """Run all three modes; useful for spot-checking on Day 5."""
        return {
            "dense":  self.dense_search(query, limit, ticker, fiscal_year, chunk_type),
            "bm25":   self.bm25_search(query, limit, ticker, fiscal_year, chunk_type),
            "hybrid": self.hybrid_search(query, limit, ticker, fiscal_year, chunk_type),
        }


# =============== pretty-print + CLI =============== #

def print_results(results: list[dict], mode: str) -> None:
    print(f"\nTop {len(results)} results ({mode}):")
    print("-" * 100)
    for i, r in enumerate(results, 1):
        meta = r["payload"]
        text = meta.get("text", "")
        preview = text[:240].replace("\n", " ").strip()
        section = meta.get("section", "")[:90]
        sources_str = ""
        if r.get("sources"):
            sources_str = f"  [from: {', '.join(r['sources'])}]"
        print(f"[{i}] id={r['id']} score={r['score']:.4f}{sources_str} | "
              f"{meta.get('ticker')} FY{meta.get('fiscal_year')} | "
              f"{meta.get('chunk_type')} | {meta.get('n_tokens')} tok")
        print(f"    Section:    {section}")
        if meta.get("subsection"):
            print(f"    Subsection: {meta['subsection']}")
        print(f"    Text:       {preview}...")
        print()


def print_compare(by_mode: dict[str, list[dict]]) -> None:
    """Print results from all three modes back-to-back. Easier than three windows."""
    for mode in ("dense", "bm25", "hybrid"):
        print("\n" + "=" * 100)
        print(f"=== MODE: {mode.upper()} ===")
        print("=" * 100)
        print_results(by_mode[mode], mode=mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-bm25", action="store_true",
                        help="Stage 1: smoke-test BM25 build (no Qdrant, no embedder)")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--mode", type=str, default="dense",
                        choices=["dense", "bm25", "hybrid", "compare"])
    parser.add_argument("--ticker", type=str, default=None)
    parser.add_argument("--fiscal-year", type=int, default=None)
    parser.add_argument("--chunk-type", type=str, default=None, choices=[None, "prose", "table"])
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    if args.build_bm25:
        _smoke_test_bm25()
    elif args.query is not None:
        retriever = Retriever()
        kwargs = dict(
            limit=args.limit,
            ticker=args.ticker,
            fiscal_year=args.fiscal_year,
            chunk_type=args.chunk_type,
        )
        if args.mode == "dense":
            print_results(retriever.dense_search(args.query, **kwargs), mode="dense")
        elif args.mode == "bm25":
            print_results(retriever.bm25_search(args.query, **kwargs), mode="bm25")
        elif args.mode == "hybrid":
            print_results(retriever.hybrid_search(args.query, **kwargs), mode="hybrid")
        elif args.mode == "compare":
            print_compare(retriever.compare(args.query, **kwargs))
    else:
        parser.print_help()