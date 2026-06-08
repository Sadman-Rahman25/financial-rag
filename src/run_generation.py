"""src/run_generation.py - Day 9: batch generation across the full eval set.

For each question in data/eval/questions.jsonl, runs retrieval + Llama 3.3 70B
generation with citations. Saves all results to results/day9_generation.json.

Supports --resume to skip questions that already have successful results, so
partial runs after a rate-limit can be completed without re-doing finished work.

Usage:
    python -m src.run_generation                       # full 28-question batch
    python -m src.run_generation --limit 3             # smoke test first 3
    python -m src.run_generation --resume              # only retry errored/missing
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from src.generate import Generator, DEFAULT_MODE, DEFAULT_TOP_K, DEFAULT_TEMPERATURE

EVAL_PATH = Path("data/eval/questions.jsonl")
DEFAULT_OUTPUT = Path("results/day9_generation.json")


def load_questions(path: Path = EVAL_PATH) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE,
                        choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None,
                        help="If set, only process the first N questions")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to sleep between requests (rate-limit safety)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip questions with successful prior results in --output")
    parser.add_argument("--category", type=str, default=None,
                        help="If set, only process questions matching this category")
    parser.add_argument("--no-decompose", action="store_true",
                        help="Disable auto-decomposition (forces single-question path)")
    parser.add_argument("--top-k-per-entity", type=int, default=5,
                        help="Per-entity top-K when decomposition triggers (default 5)")
    args = parser.parse_args()

    print(f"Eval set:    {EVAL_PATH}")
    print(f"Output:      {args.output}")
    print(f"Mode:        {args.mode}")
    print(f"Top-K:       {args.top_k}")
    print(f"Temperature: {args.temperature}")
    print(f"Delay:       {args.delay}s between requests")
    print(f"Resume:      {args.resume}")

    all_questions = load_questions()
    if args.category:
        n_before = len(all_questions)
        all_questions = [q for q in all_questions if q.get("category") == args.category]
        print(f"** Filtered to category={args.category}: {len(all_questions)} of {n_before} questions **")
    if args.limit:
        all_questions = all_questions[: args.limit]
        print(f"\n** Limited to first {args.limit} questions **")

    # Resume: load prior results, filter to questions that need (re)processing
    existing: dict = {}
    if args.resume and args.output.exists():
        try:
            existing = json.loads(args.output.read_text()).get("results", {})
            print(f"\nResume: loaded {len(existing)} prior results from {args.output}")
        except Exception as e:
            print(f"\nResume: couldn't load existing results ({e}); starting fresh")
            existing = {}

        questions = []
        for q in all_questions:
            prior = existing.get(q["id"])
            has_success = prior and "error" not in (prior.get("generated") or {})
            if not has_success:
                questions.append(q)
        skipped = len(all_questions) - len(questions)
        print(f"Resume: skipping {skipped} successful; processing {len(questions)} remaining")

        if not questions:
            print("\nNothing to do - all questions already have successful results. Exiting.")
            return
    else:
        questions = all_questions
        print(f"\nProcessing {len(questions)} questions.")

    # Build Generator and force-load Retriever so init time is paid up-front
    print("\nInitializing Generator...")
    t_init = time.time()
    gen = Generator()
    gen._get_retriever()
    init_seconds = time.time() - t_init
    print(f"  Ready in {init_seconds:.1f}s\n")

    results: dict = {}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    t_loop = time.time()
    n = len(questions)

    for i, q in enumerate(questions, 1):
        qid = q["id"]
        category = q.get("category", "")
        filters = q.get("expected_filters") or None

        t = time.time()
        try:
            result = gen.answer(
                q["question"],
                mode=args.mode,
                top_k=args.top_k,
                decompose=not args.no_decompose,
                top_k_per_entity=args.top_k_per_entity,
                filters=filters,
                temperature=args.temperature,
            )
            elapsed = time.time() - t

            if result.get("usage"):
                for k in total_usage:
                    total_usage[k] += result["usage"][k]

            results[qid] = {
                "question": q["question"],
                "category": category,
                "expected_filters": filters,
                "reference_answer": q.get("reference_answer", ""),
                "reference_chunk_ids": q.get("reference_chunk_ids", []),
                "generated": result,
            }

            refused_marker = "[REFUSE]" if result["refused"] else "        "
            n_cites = len(result.get("citations") or [])
            n_retrieved = len(result.get("retrieved_chunk_ids") or [])
            print(f"  [{i:>2}/{n}] {qid} ({category[:22]:22}) {refused_marker}  "
                  f"retr={n_retrieved:>2}  cites={n_cites:>2}  "
                  f"({elapsed:.1f}s)",
                  flush=True)
        except Exception as e:
            elapsed = time.time() - t
            print(f"  [{i:>2}/{n}] {qid} FAILED: {type(e).__name__}: {e}  "
                  f"({elapsed:.1f}s)",
                  flush=True)
            results[qid] = {
                "question": q["question"],
                "category": category,
                "expected_filters": filters,
                "reference_answer": q.get("reference_answer", ""),
                "reference_chunk_ids": q.get("reference_chunk_ids", []),
                "generated": {
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            }

        if i < n and args.delay > 0:
            time.sleep(args.delay)

    loop_seconds = time.time() - t_loop

    # Merge with prior results (resume mode)
    if existing:
        final_results = dict(existing)
        final_results.update(results)  # new results overwrite old (errored) ones
    else:
        final_results = results

    # Recompute summary across the FULL set
    n_refused = sum(1 for r in final_results.values()
                    if r.get("generated", {}).get("refused") is True)
    n_errors = sum(1 for r in final_results.values()
                   if "error" in r.get("generated", {}))
    n_answered = len(final_results) - n_refused - n_errors

    print(f"\n{'=' * 80}")
    print(f"BATCH COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total questions:    {len(final_results)}")
    print(f"  answered:         {n_answered}")
    print(f"  refused:          {n_refused}")
    print(f"  errored:          {n_errors}")
    print(f"\nThis run's token usage:")
    print(f"  prompt:           {total_usage['prompt_tokens']:>8,}")
    print(f"  completion:       {total_usage['completion_tokens']:>8,}")
    print(f"  total:            {total_usage['total_tokens']:>8,}")
    print(f"\nTiming (this run):")
    print(f"  Init:             {init_seconds:>6.1f}s")
    print(f"  Loop:             {loop_seconds:>6.1f}s")
    print(f"  Total:            {init_seconds + loop_seconds:>6.1f}s")
    if results:
        print(f"  Avg per question: {loop_seconds / len(results):>6.1f}s")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "n_questions": len(final_results),
        "n_answered": n_answered,
        "n_refused": n_refused,
        "n_errors": n_errors,
        "mode": args.mode,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "delay_between_requests": args.delay,
        "this_run_usage": total_usage,
        "this_run_timing_seconds": {
            "init": round(init_seconds, 2),
            "loop": round(loop_seconds, 2),
            "total": round(init_seconds + loop_seconds, 2),
        },
        "results": final_results,
    }
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
