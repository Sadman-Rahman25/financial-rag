"""src/generate.py — Day 9: question answering with cited generation.

Retrieves chunks for a question, prompts Llama 3.3 70B (via Groq) to answer
using only those chunks with inline [chunk_<id>] citations, and returns a
structured dict.

Forces explicit refusal when retrieved chunks don't contain the answer — this
is what prevents hallucination on out-of-corpus questions (e.g. asking about
Google when the corpus only covers AAPL/MSFT/NVDA/TSLA/META).

Usage:
    python -m src.generate --question "What was Apple's iPhone revenue in fiscal 2024?"
    python -m src.generate --question "..." --mode hybrid --top-k 5
    python -m src.generate --question "..." --ticker AAPL --fiscal-year 2024
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

# Lazy-imported inside Generator to avoid loading BGE model when only the
# client is needed (e.g. running this module's helpers in isolation).


DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MODE = "hybrid"
DEFAULT_TOP_K = 10
DEFAULT_TEMPERATURE = 0.2


SYSTEM_PROMPT = """
﻿You are a financial analyst assistant answering questions about SEC 10-K annual report filings.

CORPUS SCOPE: This corpus contains 10-K filings for ONLY these 5 companies:
  * Apple (AAPL)
  * Microsoft (MSFT)
  * NVIDIA (NVDA)
  * Tesla (TSLA)
  * Meta (META)

If the user asks about any OTHER company (Google/Alphabet, Amazon, Oracle, IBM, etc.), NO chunk can answer the question regardless of what numbers appear in the retrieved chunks. You MUST refuse per rule 4.

You will receive:
1. A user QUESTION
2. A set of CHUNKS extracted from 10-K filings. Each chunk has a header showing its source company and fiscal year, e.g. [chunk_64] (AAPL FY2024 | Item 7. MD&A | Revenue).

Your job: answer the question using ONLY information from chunks whose source company in the header matches the entity named in the question.

RULES:

1. SOURCE: Use only facts from the chunks. Do NOT use your training data, even if you know the answer from somewhere else.

2. CITATION: Cite every factual claim with [chunk_<id>] inline, placed immediately after the fact it supports.

3. ENTITY MATCH (CRITICAL): The company shown in each cited chunk''s header MUST match the entity named in the user''s question.

   WORKED EXAMPLE of the mistake to NEVER make:
   - Question: "What was Google''s revenue in 2024?"
   - Retrieved chunk_646 header: (META FY2024 | Revenue) showing total revenue $164,501M
   - WRONG ANSWER: "Google''s revenue was $164,501M [chunk_646]"
     Why wrong: META is NOT Google. The $164,501M is META''s revenue, not Google''s.
   - CORRECT BEHAVIOR: refuse per rule 4 because no retrieved chunk has Google/GOOGL/Alphabet in its header.

   A number, growth rate, or fact from Company X''s 10-K is NEVER a fact about Company Y, no matter how plausible the numerical match might appear.

4. REFUSAL: You MUST refuse if EITHER (a) no retrieved chunk has a header matching the entity in the question, OR (b) the chunks otherwise lack the specific information needed:
   - Set "refused" to true
   - In "answer", state in ONE sentence what is missing (e.g. "The corpus does not cover Google''s filings" or "The retrieved chunks do not contain the specific dollar figure")
   - Leave "citations" as an empty list
   - Do NOT provide partial answers, comparisons, or numbers from other companies'' chunks.

5. CONCISENESS: Be concise. Match the question''s level of detail. Prefer specific numbers from chunks over vague descriptions.

6. GROUNDING: Every number, dollar amount, percentage, named entity, and specific claim must come from a cited chunk whose header matches the question''s entity.

OUTPUT FORMAT: You MUST respond with a JSON object matching exactly this schema:
{
  "answer": "Your answer text with [chunk_<id>] citations inline. If refused, briefly explain what is missing.",
  "citations": [list of chunk_id integers you actually cited],
  "refused": true or false
}

Do not include any text outside the JSON object.
"""


# ============================================================ helpers

def _format_chunks_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks as a single string for the user prompt."""
    parts = []
    for c in chunks:
        cid = c["id"]
        meta = c.get("payload", {}) or {}
        ticker = meta.get("ticker", "")
        fy = meta.get("fiscal_year", "")
        section = (meta.get("section") or "")[:80]
        subsection = (meta.get("subsection") or "")[:60]

        header = f"[chunk_{cid}] ({ticker} FY{fy}"
        if section:
            header += f" | {section}"
        if subsection:
            header += f" | {subsection}"
        header += ")"

        text = meta.get("text", "").strip()
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts)


def _build_user_prompt(question: str, chunks: list[dict]) -> str:
    return (
        f"USER QUESTION:\n{question}\n\n"
        f"CHUNKS:\n\n{_format_chunks_for_prompt(chunks)}\n\n"
        f"---\n\nAnswer the question above using only the chunks provided. "
        f"Output JSON."
    )


def _extract_filter_kwargs(filters: Optional[dict]) -> dict:
    """Pull ticker/fiscal_year/chunk_type from a filters dict (None-safe)."""
    if not filters:
        return {}
    out = {}
    for key in ("ticker", "fiscal_year", "chunk_type"):
        v = filters.get(key)
        if v is not None:
            out[key] = v
    return out


# ============================================================ Generator

class Generator:
    """Holds heavy state (Retriever, Groq client) so multiple calls share it."""

    def __init__(self, retriever=None, model: str = DEFAULT_MODEL):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise SystemExit(
                "GROQ_API_KEY not found. Create .env with: GROQ_API_KEY=gsk_..."
            )
        self.client = Groq(api_key=api_key)
        self.model = model
        self.retriever = retriever  # may be None; lazy-loaded on first answer()

    def _get_retriever(self):
        if self.retriever is None:
            from src.retrieve import Retriever
            self.retriever = Retriever()
        return self.retriever

    def _retrieve(self, question, mode, top_k, filters):
        retriever = self._get_retriever()
        kwargs = _extract_filter_kwargs(filters)
        if mode == "dense":
            return retriever.dense_search(question, limit=top_k, **kwargs)
        elif mode == "bm25":
            return retriever.bm25_search(question, limit=top_k, **kwargs)
        elif mode == "hybrid":
            return retriever.hybrid_search(question, limit=top_k, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use dense/bm25/hybrid.")

    def answer(
        self,
        question: str,
        mode: str = DEFAULT_MODE,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[dict] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict:
        """Retrieve, prompt, parse — return a structured answer dict."""
        # Phase 1: retrieve
        t0 = time.time()
        chunks = self._retrieve(question, mode, top_k, filters)
        retrieve_ms = (time.time() - t0) * 1000

        if not chunks:
            return {
                "answer": "No chunks retrieved (filters may be too restrictive).",
                "citations": [],
                "refused": True,
                "retrieved_chunk_ids": [],
                "mode": mode,
                "top_k": top_k,
                "timing_ms": {"retrieve": round(retrieve_ms, 1), "generate": 0.0},
                "usage": None,
            }

        retrieved_ids = [c["id"] for c in chunks]

        # Phase 2: prompt + call LLM
        user_prompt = _build_user_prompt(question, chunks)
        t1 = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        generate_ms = (time.time() - t1) * 1000

        raw_content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        # Phase 3: parse JSON
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as e:
            return {
                "answer": f"[JSON parse error: {e}] Raw output: {raw_content}",
                "citations": [],
                "refused": True,
                "retrieved_chunk_ids": retrieved_ids,
                "mode": mode,
                "top_k": top_k,
                "timing_ms": {
                    "retrieve": round(retrieve_ms, 1),
                    "generate": round(generate_ms, 1),
                },
                "usage": usage,
                "_parse_error": True,
                "_raw": raw_content,
            }

        # Coerce types defensively
        answer_text = str(parsed.get("answer", ""))
        citations = parsed.get("citations", []) or []
        if not isinstance(citations, list):
            citations = []
        citations = [int(c) for c in citations if str(c).lstrip("-").isdigit()]
        refused = bool(parsed.get("refused", False))

        return {
            "answer": answer_text,
            "citations": citations,
            "refused": refused,
            "retrieved_chunk_ids": retrieved_ids,
            "mode": mode,
            "top_k": top_k,
            "timing_ms": {
                "retrieve": round(retrieve_ms, 1),
                "generate": round(generate_ms, 1),
            },
            "usage": usage,
        }


# ============================================================ CLI

def _print_result(question: str, result: dict) -> None:
    print("\n" + "=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    print(f"Mode: {result['mode']}   Top-K: {result['top_k']}   "
          f"Refused: {result['refused']}")
    print(f"Retrieved chunks: {result['retrieved_chunk_ids']}")
    if result.get("citations"):
        print(f"Citations from LLM: {result['citations']}")

    print("\n--- ANSWER ---")
    print(result["answer"])
    print("--- END ANSWER ---")

    if result.get("usage"):
        u = result["usage"]
        print(f"\nUsage:  prompt={u['prompt_tokens']}  "
              f"completion={u['completion_tokens']}  total={u['total_tokens']}")
    t = result.get("timing_ms", {})
    if t:
        print(f"Timing: retrieve={t.get('retrieve', 0):.0f}ms  "
              f"generate={t.get('generate', 0):.0f}ms  "
              f"total={(t.get('retrieve', 0) + t.get('generate', 0)):.0f}ms")


def main():
    parser = argparse.ArgumentParser(description="Day 9: cited answer generation")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--mode", type=str, default=DEFAULT_MODE,
                        choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--ticker", type=str, default=None,
                        help="Filter chunks by ticker (AAPL/MSFT/NVDA/TSLA/META)")
    parser.add_argument("--fiscal-year", type=int, default=None,
                        help="Filter chunks by fiscal year")
    parser.add_argument("--chunk-type", type=str, default=None,
                        choices=[None, "prose", "table"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker
    if args.fiscal_year is not None:
        filters["fiscal_year"] = args.fiscal_year
    if args.chunk_type:
        filters["chunk_type"] = args.chunk_type

    gen = Generator(model=args.model)
    result = gen.answer(
        args.question,
        mode=args.mode,
        top_k=args.top_k,
        filters=filters or None,
        temperature=args.temperature,
    )
    _print_result(args.question, result)


if __name__ == "__main__":
    main()