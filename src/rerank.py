"""
src/rerank.py — Day 6: BGE cross-encoder reranker.

Usage:
    python -m src.rerank --smoke-test    # Stage 1: verify reranker loads and scores plausibly
    python -m src.rerank --test-rerank   # Stage 2: test rerank() method on synthetic candidates
"""
import argparse
import json
import time
from pathlib import Path

from sentence_transformers import CrossEncoder
import torch

from src.index import CHUNKS_PATH

RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_BATCH = 4  # conservative for 8 GB RAM


class Reranker:
    """Cross-encoder reranker. Loaded once, reused across queries."""

    def __init__(self) -> None:
        print(f"Loading reranker {RERANKER_MODEL} (fp16)...", flush=True)
        t = time.time()
        import torch
        self.model = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,
            model_kwargs={"torch_dtype": torch.float16},
        )
        print(f"  Loaded in {time.time()-t:.1f}s", flush=True)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """Re-score (query, chunk) pairs with the cross-encoder, sort by rerank_score desc.

        Args:
            query:      original query text
            candidates: list from upstream retrieval, each dict has 'id', 'score', 'payload'
                        (payload must contain 'text')
            top_k:      how many top results to return

        Returns:
            Re-ordered list (length <= top_k). Each candidate gets a new 'rerank_score' field.
            Original 'score' is preserved so downstream code can show both.
        """
        if not candidates:
            return []
        pairs = [(query, c["payload"]["text"]) for c in candidates]
        scores = self.model.predict(pairs, batch_size=RERANK_BATCH, show_progress_bar=False)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: -c["rerank_score"])[:top_k]


def _smoke_test_reranker() -> None:
    """Stage 1 verification: load reranker, score known good and bad pairs."""
    target_ids = {97, 235, 2712}
    print("Loading test chunks from chunks.jsonl...", flush=True)
    chunks_by_id: dict[int, dict] = {}
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line and i in target_ids:
                chunks_by_id[i] = json.loads(line)
            if len(chunks_by_id) == len(target_ids):
                break
    print(f"  Loaded {len(chunks_by_id)} test chunks", flush=True)

    reranker = Reranker()

    query = "Apple iPhone revenue"
    test_pairs = [
        (query, chunks_by_id[97]["text"],   "POSITIVE  (AAPL Note 3 EPS — discusses iPhone/services revenue)"),
        (query, chunks_by_id[235]["text"],  "POSITIVE  (AAPL Products & Services Performance table)"),
        (query, chunks_by_id[2712]["text"], "NEGATIVE  (TSLA deferred revenue — wrong company)"),
    ]

    print(f"\nQuery: {query!r}")
    print(f"Scoring {len(test_pairs)} (query, chunk) pairs...\n", flush=True)
    pairs_only = [(q, c) for q, c, _ in test_pairs]
    scores = reranker.model.predict(pairs_only, batch_size=RERANK_BATCH, show_progress_bar=False)

    print("Results:")
    print("-" * 90)
    for (q, c, label), score in zip(test_pairs, scores):
        preview = c[:120].replace("\n", " ").strip()
        print(f"  score = {float(score):+8.4f}   | {label}")
        print(f"  preview:                   {preview}...")
        print()

    pos_avg = (float(scores[0]) + float(scores[1])) / 2
    neg = float(scores[2])
    print(f"Positives avg: {pos_avg:+.4f}   Negative: {neg:+.4f}   Delta: {pos_avg - neg:+.4f}")
    if pos_avg > neg:
        print("OK: positives scored higher than negative. Reranker works as expected.")
    else:
        print("WARNING: negative scored as high or higher than positives. Investigate.")


def _test_rerank() -> None:
    """Stage 2 verification: build synthetic candidates, rerank them, observe reordering."""
    target_ids = {97, 235, 64, 2712, 2718}  # mix of AAPL prose/tables + TSLA chunks
    chunks_by_id: dict[int, dict] = {}
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line and i in target_ids:
                chunks_by_id[i] = json.loads(line)
            if len(chunks_by_id) == len(target_ids):
                break

    # Build fake candidates (as if they came from hybrid_search)
    candidates = [
        {
            "id": i,
            "score": 0.5,  # fake upstream score
            "payload": {**chunks_by_id[i]["metadata"], "text": chunks_by_id[i]["text"]},
        }
        for i in sorted(target_ids)
    ]

    print(f"\nORIGINAL order (by id, as if from upstream retrieval):")
    for j, c in enumerate(candidates, 1):
        meta = c["payload"]
        preview = meta["text"][:70].replace("\n", " ").strip()
        print(f"  [{j}] id={c['id']} {meta['ticker']} FY{meta['fiscal_year']} | {preview}...")

    reranker = Reranker()
    query = "Apple iPhone revenue"
    print(f"\nReranking with query: {query!r}\n", flush=True)
    reranked = reranker.rerank(query, candidates, top_k=len(candidates))

    print(f"RERANKED order:")
    for j, c in enumerate(reranked, 1):
        meta = c["payload"]
        preview = meta["text"][:70].replace("\n", " ").strip()
        print(f"  [{j}] id={c['id']} rerank={c['rerank_score']:.4f} | "
              f"{meta['ticker']} FY{meta['fiscal_year']} | {preview}...")
    print("\nOK: rerank() method works — AAPL chunks should be at the top, TSLA at the bottom.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Stage 1: verify reranker loads and scores plausibly")
    parser.add_argument("--test-rerank", action="store_true",
                        help="Stage 2: test rerank() method on synthetic candidates")
    args = parser.parse_args()

    if args.smoke_test:
        _smoke_test_reranker()
    elif args.test_rerank:
        _test_rerank()
    else:
        parser.print_help()