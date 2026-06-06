"""src/metrics.py — Day 8: retrieval evaluation metrics from scratch.

All metrics use binary relevance: a retrieved chunk is either in the reference
set (relevant=1) or not (relevant=0). No graded relevance, no LLM judge.

Conventions:
  - Ranks are 1-indexed (top-1 is the first result, not the zeroth).
  - NDCG discount factor: 1 / log_2(rank + 1).
  - All metrics return 0.0 when the reference set is empty — the caller is
    responsible for filtering out questions with no refs (e.g. out-of-corpus
    questions q27/q28).
  - Precision@K denominator is K even if fewer than K results were retrieved
    (the standard convention; otherwise short result lists get an unfair bump).

Self-test: `python -m src.metrics`
"""
from __future__ import annotations
from typing import Iterable, Sequence
import math


def _prepare(retrieved: Sequence[int], references: Iterable[int], k: int):
    refs = set(references)
    top_k = list(retrieved)[: max(k, 0)]
    return refs, top_k


def hit_at_k(retrieved: Sequence[int], references: Iterable[int], k: int) -> float:
    """1.0 if any reference chunk is in top K, else 0.0."""
    refs, top_k = _prepare(retrieved, references, k)
    if not refs:
        return 0.0
    return 1.0 if any(chunk_id in refs for chunk_id in top_k) else 0.0


def recall_at_k(retrieved: Sequence[int], references: Iterable[int], k: int) -> float:
    """Fraction of reference chunks present in top K."""
    refs, top_k = _prepare(retrieved, references, k)
    if not refs:
        return 0.0
    hits = sum(1 for chunk_id in top_k if chunk_id in refs)
    return hits / len(refs)


def precision_at_k(retrieved: Sequence[int], references: Iterable[int], k: int) -> float:
    """Fraction of top K that are reference chunks."""
    refs, top_k = _prepare(retrieved, references, k)
    if k <= 0 or not refs:
        return 0.0
    hits = sum(1 for chunk_id in top_k if chunk_id in refs)
    return hits / k


def mrr_at_k(retrieved: Sequence[int], references: Iterable[int], k: int) -> float:
    """Reciprocal of the rank (1-indexed) of the first reference chunk in top K.

    Returns 0.0 if no reference chunk appears in top K.
    """
    refs, top_k = _prepare(retrieved, references, k)
    if not refs:
        return 0.0
    for rank, chunk_id in enumerate(top_k, start=1):
        if chunk_id in refs:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[int], references: Iterable[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K with binary relevance.

    DCG  = sum_{i=1..K} rel_i / log_2(i + 1)
    IDCG = sum_{i=1..min(K, |refs|)} 1 / log_2(i + 1)   # best possible
    NDCG = DCG / IDCG
    """
    refs, top_k = _prepare(retrieved, references, k)
    if not refs or k <= 0:
        return 0.0

    dcg = 0.0
    for rank, chunk_id in enumerate(top_k, start=1):
        if chunk_id in refs:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(k, len(refs))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ---------- batch helper ----------

ALL_METRICS = {
    "hit": hit_at_k,
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}


def compute_all(
    retrieved: Sequence[int],
    references: Iterable[int],
    k_values: Sequence[int],
) -> dict:
    """Compute all 5 metrics at every K value for one (retrieved, refs) pair.

    Returns a flat dict like {"hit@10": 1.0, "mrr@10": 0.33, ...}.
    """
    refs = list(references)
    out: dict = {}
    for metric_name, fn in ALL_METRICS.items():
        for k in k_values:
            out[f"{metric_name}@{k}"] = fn(retrieved, refs, k)
    return out


# ---------- self-test ----------

if __name__ == "__main__":
    print("=== src/metrics.py self-test ===\n")

    # Test 1: perfect retrieval — all refs in top positions
    retrieved = [64, 96, 65, 99, 100]
    refs = [64, 96, 65]
    assert hit_at_k(retrieved, refs, 10) == 1.0
    assert recall_at_k(retrieved, refs, 10) == 1.0
    assert mrr_at_k(retrieved, refs, 10) == 1.0
    assert ndcg_at_k(retrieved, refs, 10) == 1.0
    print("Test 1 (perfect retrieval, all refs at top): OK")

    # Test 2: one hit at rank 3
    retrieved = [101, 102, 64, 103, 104]
    refs = [64, 96, 65]
    assert hit_at_k(retrieved, refs, 5) == 1.0
    assert hit_at_k(retrieved, refs, 2) == 0.0   # 64 is at rank 3, not in top 2
    assert abs(mrr_at_k(retrieved, refs, 5) - 1 / 3) < 1e-9
    assert abs(recall_at_k(retrieved, refs, 5) - 1 / 3) < 1e-9
    assert abs(precision_at_k(retrieved, refs, 5) - 1 / 5) < 1e-9
    print("Test 2 (single hit at rank 3): OK")

    # Test 3: no hits in top K
    retrieved = [101, 102, 103]
    refs = [64, 96, 65]
    assert hit_at_k(retrieved, refs, 10) == 0.0
    assert mrr_at_k(retrieved, refs, 10) == 0.0
    assert ndcg_at_k(retrieved, refs, 10) == 0.0
    print("Test 3 (no hits in top K): OK")

    # Test 4: NDCG penalizes lower-ranked hits
    refs = [64]
    ndcg_rank1 = ndcg_at_k([64, 99, 98, 97, 96], refs, 5)  # ref at rank 1
    ndcg_rank5 = ndcg_at_k([99, 98, 97, 96, 64], refs, 5)  # ref at rank 5
    assert ndcg_rank1 == 1.0
    assert 0 < ndcg_rank5 < 1.0
    assert ndcg_rank1 > ndcg_rank5
    print(f"Test 4 (NDCG order matters): rank1={ndcg_rank1:.3f}, rank5={ndcg_rank5:.3f}: OK")

    # Test 5: empty refs -> 0.0 across the board
    for fn in ALL_METRICS.values():
        assert fn([1, 2, 3], [], 10) == 0.0
    print("Test 5 (empty refs -> 0 everywhere): OK")

    # Test 6: compute_all returns expected keys
    result = compute_all([64, 96, 65, 99, 100], [64, 96, 65], [1, 3, 5, 10])
    assert result["hit@10"] == 1.0
    assert result["recall@10"] == 1.0
    assert result["mrr@1"] == 1.0
    assert "ndcg@5" in result
    assert "precision@3" in result
    expected_n = len(ALL_METRICS) * 4  # 5 metrics × 4 K values
    assert len(result) == expected_n
    print(f"Test 6 (compute_all): {len(result)} keys, hit@10={result['hit@10']}, "
          f"mrr@1={result['mrr@1']}: OK")

    print("\nAll tests passed.")