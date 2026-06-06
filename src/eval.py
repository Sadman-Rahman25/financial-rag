"""
src/eval.py — Day 7: eval set tooling for building and validating questions.jsonl.

Usage:
    python -m src.eval --status                                # progress check
    python -m src.eval --find q01                              # find candidate chunks for a question
    python -m src.eval --view <chunk_id>                       # print full text of a chunk
    python -m src.eval --update q01 --refs 96,64,97,65         # write refs to jsonl
    python -m src.eval --update q01 --answer "..."             # write reference answer
    python -m src.eval --dry-run --mode hybrid                 # run all questions, show hit@10
"""
import argparse
import gc
import json
from pathlib import Path
from typing import Optional

EVAL_PATH = Path("data/eval/questions.jsonl")
CHUNKS_PATH = Path("data/chunks.jsonl")


def load_eval_set() -> list[dict]:
    questions = []
    with EVAL_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def save_eval_set(questions: list[dict]) -> None:
    """Write back, one JSON object per line."""
    with EVAL_PATH.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")


def status() -> None:
    """Progress summary by category."""
    questions = load_eval_set()
    n = len(questions)
    with_refs = sum(1 for q in questions
                    if q.get("reference_chunk_ids") is not None
                    and (q.get("reference_chunk_ids") or q.get("category") == "out_of_corpus"))
    with_ans = sum(1 for q in questions if q.get("reference_answer"))

    print(f"Eval set: {n} questions total")
    print(f"  With reference_chunk_ids: {with_refs}/{n} ({100*with_refs/n:.0f}%)")
    print(f"  With reference_answer:    {with_ans}/{n} ({100*with_ans/n:.0f}%)")
    print()
    print("Per-category breakdown:")
    cats: dict = {}
    for q in questions:
        cat = q.get("category", "unknown")
        cats.setdefault(cat, {"total": 0, "with_refs": 0, "with_ans": 0})
        cats[cat]["total"] += 1
        if q.get("reference_chunk_ids") or q.get("category") == "out_of_corpus":
            cats[cat]["with_refs"] += 1
        if q.get("reference_answer"):
            cats[cat]["with_ans"] += 1
    for cat, c in sorted(cats.items()):
        print(f"  {cat:30s}  refs={c['with_refs']}/{c['total']}  answers={c['with_ans']}/{c['total']}")


def find_candidates(question_id: str, top_per_mode: int = 10) -> None:
    """Run a question through all 4 modes, aggregate unique candidates, show previews."""
    from src.retrieve import Retriever, extract_filters, RERANK_CANDIDATES

    questions = load_eval_set()
    q = next((qi for qi in questions if qi["id"] == question_id), None)
    if q is None:
        print(f"Question {question_id!r} not found in eval set.")
        return

    print(f"\nQuestion: {q['question']}")
    print(f"Category: {q['category']}")

    filter_kwargs = dict(q.get("expected_filters") or {})
    if not filter_kwargs:
        auto = extract_filters(q["question"])
        if auto:
            print(f"Auto-detected filters: {auto}")
            filter_kwargs = auto
    if filter_kwargs:
        print(f"Applying filters: {filter_kwargs}")
    else:
        print("No filters applied (cross-company / out-of-corpus query).")
    print()

    retriever = Retriever()
    K = top_per_mode
    dense = retriever.dense_search(q["question"], limit=K, **filter_kwargs)
    bm25 = retriever.bm25_search(q["question"], limit=K, **filter_kwargs)
    hybrid = retriever.hybrid_search(q["question"], limit=K, **filter_kwargs)
    rerank_candidates = retriever.hybrid_search(
        q["question"], limit=RERANK_CANDIDATES, **filter_kwargs
    )

    print("Freeing retriever to make room for reranker...", flush=True)
    del retriever
    gc.collect()

    from src.rerank import Reranker
    reranker = Reranker()
    reranked = reranker.rerank(q["question"], rerank_candidates, top_k=K)

    aggregated: dict = {}
    for mode_name, results in [
        ("dense", dense), ("bm25", bm25),
        ("hybrid", hybrid), ("rerank", reranked),
    ]:
        for rank, r in enumerate(results, 1):
            cid = r["id"]
            if cid not in aggregated:
                aggregated[cid] = {"payload": r["payload"], "appearances": []}
            score_key = "rerank_score" if "rerank_score" in r else "score"
            aggregated[cid]["appearances"].append(f"{mode_name}#{rank}={r[score_key]:.4f}")

    sorted_chunks = sorted(
        aggregated.items(),
        key=lambda x: (-len(x[1]["appearances"]), x[0]),
    )

    print(f"\nAggregated candidates ({len(sorted_chunks)} unique chunks):")
    print("=" * 100)
    for cid, info in sorted_chunks:
        meta = info["payload"]
        text = meta.get("text", "")
        preview = text[:200].replace("\n", " ").strip()
        section = meta.get("section", "")[:80]
        appearances_str = ", ".join(info["appearances"])
        print(f"\nid={cid}  ({len(info['appearances'])} modes)  [{appearances_str}]")
        print(f"  {meta.get('ticker')} FY{meta.get('fiscal_year')} | "
              f"{meta.get('chunk_type')} | {meta.get('n_tokens')} tok")
        print(f"  Section:    {section}")
        if meta.get("subsection"):
            print(f"  Subsection: {meta['subsection']}")
        print(f"  Text: {preview}...")

    print("\n" + "=" * 100)
    print(f"\nTo mark reference chunks for {question_id}:")
    print(f"  python -m src.eval --update {question_id} --refs <comma,separated,ids>")


def view_chunk(chunk_id: int) -> None:
    """Print the full text of a chunk by line index in chunks.jsonl."""
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == chunk_id:
                line = line.strip()
                if line:
                    chunk = json.loads(line)
                    meta = chunk["metadata"]
                    print(f"\n=== Chunk id={chunk_id} ===")
                    print(f"Ticker:       {meta.get('ticker')}")
                    print(f"Fiscal year:  {meta.get('fiscal_year')}")
                    print(f"Section:      {meta.get('section')}")
                    if meta.get("subsection"):
                        print(f"Subsection:   {meta['subsection']}")
                    print(f"Chunk type:   {meta.get('chunk_type')}")
                    print(f"Tokens:       {meta.get('n_tokens')}")
                    print(f"\n--- Full text ---")
                    print(chunk["text"])
                    print("--- End ---\n")
                return
    print(f"Chunk id={chunk_id} not found.")


def update_question(question_id: str,
                    refs: Optional[list[int]] = None,
                    answer: Optional[str] = None,
                    notes: Optional[str] = None) -> None:
    """Update reference_chunk_ids and/or reference_answer for one question."""
    questions = load_eval_set()
    updated = False
    for q in questions:
        if q["id"] == question_id:
            if refs is not None:
                q["reference_chunk_ids"] = refs
                print(f"Updated {question_id} reference_chunk_ids -> {refs}")
            if answer is not None:
                q["reference_answer"] = answer
                print(f"Updated {question_id} reference_answer ({len(answer)} chars)")
            if notes is not None:
                q["notes"] = notes
                print(f"Updated {question_id} notes")
            updated = True
            break
    if not updated:
        print(f"Question {question_id!r} not found.")
        return
    save_eval_set(questions)


def dry_run(mode: str = "hybrid", limit: int = 10, use_filters: bool = True) -> None:
    """Run every testable question, show Hit@K. Skips out-of-corpus (refs=[])."""
    from src.retrieve import Retriever
    questions = load_eval_set()
    testable = [q for q in questions if q.get("reference_chunk_ids")]
    print(f"Dry-run: {len(testable)}/{len(questions)} questions have refs to test "
          f"(mode={mode}, limit={limit}, filters={'on' if use_filters else 'off'})\n")

    retriever = Retriever()
    hits = 0
    misses = []
    for q in testable:
        filter_kwargs = q.get("expected_filters") or {} if use_filters else {}
        if mode == "dense":
            results = retriever.dense_search(q["question"], limit=limit, **filter_kwargs)
        elif mode == "bm25":
            results = retriever.bm25_search(q["question"], limit=limit, **filter_kwargs)
        else:
            results = retriever.hybrid_search(q["question"], limit=limit, **filter_kwargs)
        retrieved = [r["id"] for r in results]
        ref = set(q["reference_chunk_ids"])
        hit = bool(set(retrieved) & ref)
        hits += int(hit)
        marker = "✓" if hit else "✗"
        print(f"{marker} {q['id']} ({q['category']:30s}): {q['question'][:55]}")
        if not hit:
            misses.append((q, retrieved))

    print(f"\nHit@{limit} ({mode}): {hits}/{len(testable)} = {hits/len(testable):.1%}")

    if misses:
        print(f"\n--- {len(misses)} misses ---")
        for q, retrieved in misses:
            print(f"\n{q['id']}: {q['question']}")
            print(f"  Reference: {q['reference_chunk_ids']}")
            print(f"  Retrieved: {retrieved}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--status", action="store_true")
    p.add_argument("--find", type=str, default=None, metavar="QID")
    p.add_argument("--view", type=int, default=None, metavar="CHUNK_ID")
    p.add_argument("--update", type=str, default=None, metavar="QID")
    p.add_argument("--refs", type=str, default=None, metavar="IDS",
                   help="Comma-separated chunk IDs (with --update)")
    p.add_argument("--answer", type=str, default=None, help="Reference answer (with --update)")
    p.add_argument("--notes", type=str, default=None, help="Notes (with --update)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mode", type=str, default="hybrid", choices=["dense", "bm25", "hybrid"])
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--no-filters", action="store_true")
    args = p.parse_args()

    if args.status:
        status()
    elif args.find:
        find_candidates(args.find)
    elif args.view is not None:
        view_chunk(args.view)
    elif args.update:
        refs = [int(x.strip()) for x in args.refs.split(",")] if args.refs else None
        update_question(args.update, refs=refs, answer=args.answer, notes=args.notes)
    elif args.dry_run:
        dry_run(mode=args.mode, limit=args.limit, use_filters=not args.no_filters)
    else:
        p.print_help()