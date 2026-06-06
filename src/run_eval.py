"""src/run_eval.py — Day 8: full retrieval evaluation across all 4 modes.

Runs the eval set through dense / BM25 / hybrid / rerank and computes
Hit/Recall/Precision/MRR/NDCG @ K={1,3,5,10}.

Two-phase loading for 8 GB RAM:
  Phase 1: load Retriever (embedder + BM25), run dense/BM25/hybrid for all
           testable questions. Cache hybrid pool (top-{rerank_pool}) for reranking.
  Phase 2: free Retriever, load Reranker, rerank hybrid pool -> top-{limit}.

Out-of-corpus questions (empty reference_chunk_ids) are excluded from
retrieval metrics by design — they test refusal, which is Day 9's job.

Usage:
    python -m src.run_eval
    python -m src.run_eval --output results/day8_retrieval_eval.json
    python -m src.run_eval --rerank-pool 30
"""
from __future__ import annotations
import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

from src.metrics import compute_all
from src.retrieve import Retriever, RERANK_CANDIDATES

EVAL_PATH = Path("data/eval/questions.jsonl")
DEFAULT_OUTPUT = Path("results/day8_retrieval_eval.json")
DEFAULT_K_VALUES = [1, 3, 5, 10]
DEFAULT_LIMIT = 10
MODES = ["dense", "bm25", "hybrid", "rerank"]


# ============================================================ utilities

def load_questions(path: Path = EVAL_PATH) -> list[dict]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            questions.append(json.loads(line))
    return questions


def is_testable(q: dict) -> bool:
    return bool(q.get("reference_chunk_ids"))


def ids_only(results: list[dict]) -> list[int]:
    return [r["id"] for r in results]


def expand_filters(filters_dict: dict | None) -> dict:
    """Convert expected_filters dict -> kwargs accepted by Retriever methods.

    Retriever takes individual kwargs (ticker, fiscal_year, chunk_type), not
    a dict. This helper unpacks the eval-set's expected_filters into the right
    shape and drops null/unknown keys.
    """
    if not filters_dict:
        return {}
    out = {}
    for key in ("ticker", "fiscal_year", "chunk_type"):
        v = filters_dict.get(key)
        if v is not None:
            out[key] = v
    return out


# ============================================================ retrieval phases

def run_phase1(
    questions: list[dict],
    retriever: Retriever,
    limit: int,
    rerank_pool: int,
) -> dict:
    """For each question: dense/BM25 top-{limit}, hybrid top-{rerank_pool}.

    The top-{limit} slice of the hybrid pool serves as the hybrid evaluation
    result; the full pool is fed to the reranker in Phase 2.
    """
    assert rerank_pool >= limit, "rerank_pool must be >= limit"
    out: dict[str, dict] = {}
    n = len(questions)
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        query = q["question"]
        kwargs = expand_filters(q.get("expected_filters"))
        t = time.time()
        dense_res = retriever.dense_search(query, limit=limit, **kwargs)
        bm25_res = retriever.bm25_search(query, limit=limit, **kwargs)
        hybrid_pool = retriever.hybrid_search(query, limit=rerank_pool, **kwargs)
        out[qid] = {
            "dense": dense_res,
            "bm25": bm25_res,
            "hybrid": hybrid_pool[:limit],
            "rerank_pool": hybrid_pool,
        }
        print(f"  [{i:>2}/{n}] {qid}: dense={len(dense_res)}  "
              f"bm25={len(bm25_res)}  hybrid_pool={len(hybrid_pool)}  "
              f"({time.time() - t:.1f}s)",
              flush=True)
    return out


def run_phase2(questions: list[dict], phase1: dict, limit: int) -> dict:
    """Rerank each question's cached hybrid pool -> top-{limit}."""
    from src.rerank import Reranker
    print("Loading reranker (BGE-reranker-base fp16)...", flush=True)
    reranker = Reranker()
    out: dict[str, list] = {}
    n = len(questions)
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        query = q["question"]
        pool = phase1[qid]["rerank_pool"]
        t = time.time()
        reranked = reranker.rerank(query, pool, top_k=limit)
        out[qid] = reranked
        print(f"  [{i:>2}/{n}] {qid}: rerank({len(pool)} -> {len(reranked)})  "
              f"({time.time() - t:.1f}s)",
              flush=True)
    return out


# ============================================================ metric aggregation

def compute_per_question(
    questions: list[dict],
    phase1: dict,
    rerank_results: dict,
    k_values: list[int],
) -> dict:
    per_q = {}
    for q in questions:
        qid = q["id"]
        refs = q["reference_chunk_ids"]
        retrieved_by_mode = {
            "dense": ids_only(phase1[qid]["dense"]),
            "bm25": ids_only(phase1[qid]["bm25"]),
            "hybrid": ids_only(phase1[qid]["hybrid"]),
            "rerank": ids_only(rerank_results[qid]),
        }
        per_q[qid] = {
            "category": q.get("category"),
            "question": q["question"],
            "reference_chunk_ids": refs,
            "retrieved": retrieved_by_mode,
            "metrics": {
                mode: compute_all(retrieved_by_mode[mode], refs, k_values)
                for mode in MODES
            },
        }
    return per_q


def aggregate_metrics(per_q: dict) -> dict:
    """Macro-average each metric across all questions, per mode."""
    if not per_q:
        return {mode: {} for mode in MODES}
    metric_keys = list(next(iter(per_q.values()))["metrics"]["dense"].keys())
    agg = {}
    for mode in MODES:
        agg[mode] = {}
        for mk in metric_keys:
            vals = [pq["metrics"][mode][mk] for pq in per_q.values()]
            agg[mode][mk] = sum(vals) / len(vals) if vals else 0.0
    return agg


def aggregate_by_category(per_q: dict) -> dict:
    """Macro-average per (category, mode) for category-level analysis."""
    by_cat: dict[str, list[str]] = defaultdict(list)
    for qid, info in per_q.items():
        by_cat[info["category"]].append(qid)
    if not per_q:
        return {}
    metric_keys = list(next(iter(per_q.values()))["metrics"]["dense"].keys())
    out = {}
    for cat, qids in by_cat.items():
        out[cat] = {"n": len(qids), "metrics": {}}
        for mode in MODES:
            out[cat]["metrics"][mode] = {}
            for mk in metric_keys:
                vals = [per_q[qid]["metrics"][mode][mk] for qid in qids]
                out[cat]["metrics"][mode][mk] = sum(vals) / len(vals) if vals else 0.0
    return out


# ============================================================ pretty-print

def print_summary_table(agg: dict) -> None:
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS (macro-averaged across testable questions)")
    print("=" * 80)
    headline = [
        "hit@1", "hit@3", "hit@5", "hit@10",
        "mrr@10", "ndcg@10", "recall@10", "precision@10",
    ]
    w_m, w_v = 14, 10
    print(f"{'Metric':<{w_m}}" + "".join(f"{m:>{w_v}}" for m in MODES))
    print("-" * (w_m + len(MODES) * w_v))
    for metric in headline:
        line = f"{metric:<{w_m}}"
        for mode in MODES:
            line += f"{agg[mode][metric]:>{w_v}.3f}"
        print(line)


def print_category_table(by_cat: dict, metric_key: str) -> None:
    print(f"\n{metric_key.upper()} BY CATEGORY")
    print("-" * 80)
    w_c, w_n, w_v = 30, 5, 10
    header = f"{'Category':<{w_c}}{'N':>{w_n}}" + "".join(f"{m:>{w_v}}" for m in MODES)
    print(header)
    print("-" * (w_c + w_n + len(MODES) * w_v))
    for cat in sorted(by_cat.keys()):
        line = f"{cat:<{w_c}}{by_cat[cat]['n']:>{w_n}}"
        for mode in MODES:
            line += f"{by_cat[cat]['metrics'][mode][metric_key]:>{w_v}.3f}"
        print(line)


# ============================================================ main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="Top-K returned per mode (default: 10)")
    parser.add_argument("--rerank-pool", type=int, default=RERANK_CANDIDATES,
                        help=f"Hybrid candidates fed to reranker "
                             f"(default: {RERANK_CANDIDATES})")
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
                        help="K values for metric computation (default: 1 3 5 10)")
    args = parser.parse_args()

    print(f"Eval set:    {EVAL_PATH}")
    print(f"Output:      {args.output}")
    print(f"Limit:       top-{args.limit} per mode")
    print(f"Rerank pool: top-{args.rerank_pool} from hybrid")
    print(f"K values:    {args.k_values}")

    # Load eval set
    all_q = load_questions()
    testable = [q for q in all_q if is_testable(q)]
    skipped = [q for q in all_q if not is_testable(q)]
    skipped_ids = [q["id"] for q in skipped]
    print(f"\nLoaded {len(all_q)} questions; testable: {len(testable)}; "
          f"skipped (empty refs): {len(skipped)} {skipped_ids}")

    # Phase 1
    print("\n" + "=" * 80)
    print("PHASE 1: Retriever (dense + BM25 + hybrid)")
    print("=" * 80)
    t_p1 = time.time()
    retriever = Retriever()
    phase1 = run_phase1(testable, retriever, limit=args.limit,
                        rerank_pool=args.rerank_pool)
    p1_seconds = time.time() - t_p1
    print(f"\nPhase 1 done in {p1_seconds:.1f}s")

    # Free retriever
    print("\nFreeing retriever (embedder + BM25) to make room for reranker...",
          flush=True)
    del retriever
    gc.collect()

    # Phase 2
    print("\n" + "=" * 80)
    print("PHASE 2: Reranker (BGE-reranker-base fp16)")
    print("=" * 80)
    t_p2 = time.time()
    rerank_results = run_phase2(testable, phase1, limit=args.limit)
    p2_seconds = time.time() - t_p2
    print(f"\nPhase 2 done in {p2_seconds:.1f}s")
    gc.collect()

    # Compute metrics
    print("\n" + "=" * 80)
    print("Computing metrics ...")
    print("=" * 80)
    per_q = compute_per_question(testable, phase1, rerank_results, args.k_values)
    agg = aggregate_metrics(per_q)
    by_cat = aggregate_by_category(per_q)

    # Print tables
    print_summary_table(agg)
    print_category_table(by_cat, "hit@10")
    print_category_table(by_cat, "mrr@10")
    print_category_table(by_cat, "ndcg@10")

    # Save full results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "n_questions_total": len(all_q),
        "n_testable": len(testable),
        "n_skipped": len(skipped),
        "skipped_qids": skipped_ids,
        "k_values": args.k_values,
        "limit": args.limit,
        "rerank_pool": args.rerank_pool,
        "timing_seconds": {
            "phase1": round(p1_seconds, 2),
            "phase2": round(p2_seconds, 2),
            "total": round(p1_seconds + p2_seconds, 2),
        },
        "aggregate": agg,
        "by_category": by_cat,
        "per_question": per_q,
    }
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()