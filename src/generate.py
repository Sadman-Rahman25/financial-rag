"""Generate cited answers using the Retriever + Groq Llama 3.3 70B.

Public API: Generator().answer(question, decompose=True, top_k_per_entity=5, ...)

Auto-detects 2+ entities in the question and routes to per-entity retrieval +
synthesis. Single-entity (or out-of-corpus) questions route to the original
single-prompt path. The decompose=False kwarg forces the single-prompt path for
ablation purposes.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

from src.decompose import detect_entities

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_MODE = "hybrid"
DEFAULT_TOP_K = 10
DEFAULT_TOP_K_PER_ENTITY = 5
DEFAULT_TEMPERATURE = 0.2

SYSTEM_PROMPT = """
You are a financial analyst assistant answering questions about SEC 10-K annual report filings.

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

3. ENTITY MATCH (CRITICAL): The company shown in each cited chunk's header MUST match the entity named in the user's question.

   WORKED EXAMPLE of the mistake to NEVER make:
   - Question: "What was Google's revenue in 2024?"
   - Retrieved chunk_646 header: (META FY2024 | Revenue) showing total revenue $164,501M
   - WRONG ANSWER: "Google's revenue was $164,501M [chunk_646]"
     Why wrong: META is NOT Google. The $164,501M is META's revenue, not Google's.
   - CORRECT BEHAVIOR: refuse per rule 4 because no retrieved chunk has Google/GOOGL/Alphabet in its header.

   A number, growth rate, or fact from Company X's 10-K is NEVER a fact about Company Y, no matter how plausible the numerical match might appear.

4. REFUSAL: You MUST refuse if EITHER (a) no retrieved chunk has a header matching the entity in the question, OR (b) the chunks otherwise lack the specific information needed:
   - Set "refused" to true
   - In "answer", state in ONE sentence what is missing (e.g. "The corpus does not cover Google's filings" or "The retrieved chunks do not contain the specific dollar figure")
   - Leave "citations" as an empty list
   - Do NOT provide partial answers, comparisons, or numbers from other companies' chunks.

5. CONCISENESS: Be concise. Match the question's level of detail. Prefer specific numbers from chunks over vague descriptions.

6. GROUNDING: Every number, dollar amount, percentage, named entity, and specific claim must come from a cited chunk whose header matches the question's entity.

OUTPUT FORMAT: You MUST respond with a JSON object matching exactly this schema:
{
  "answer": "Your answer text with [chunk_<id>] citations inline. If refused, briefly explain what is missing.",
  "citations": [list of chunk_id integers you actually cited],
  "refused": true or false
}

Do not include any text outside the JSON object.
"""

SYNTHESIS_SYSTEM_PROMPT = """
You are a financial analyst assistant answering COMPARISON questions about SEC 10-K filings.

CORPUS SCOPE: Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA), Tesla (TSLA), Meta (META).

You will receive:
1. A user QUESTION that compares two or more companies in our corpus
2. A set of CHUNKS spanning all the named companies. Each chunk has a header showing its source company and fiscal year, e.g. [chunk_64] (NVDA FY2026 | Item 7. MD&A | Data Center).

Your job: produce a structured comparison. For EACH company named in the question, describe its position using ONLY chunks whose header matches that company. Then explicitly compare them.

RULES:

1. SOURCE: Use only facts from the chunks. No training data.

2. CITATION: Cite every factual claim with [chunk_<id>] inline, placed immediately after the fact it supports.

3. PER-COMPANY ENTITY MATCH (CRITICAL): Each cited chunk's company in its header MUST match the company being described at that point in your answer.
   - When describing NVIDIA's position, cite ONLY chunks with (NVDA ...) in the header.
   - When describing Microsoft's position, cite ONLY chunks with (MSFT ...) in the header.
   - NEVER attribute a fact from Company X's chunk to Company Y, even briefly.

4. STRUCTURE: Organize the answer in two parts:
   - Part 1: One short paragraph per company, with that company's specific facts and citations. Lead each paragraph with the company name.
   - Part 2: A final short comparison paragraph that draws contrasts or similarities across the companies.

5. PARTIAL HANDLING: If chunks for ONE of the named companies are missing or insufficient (e.g. you have NVIDIA chunks but only weak Microsoft chunks), say so explicitly for that company in its paragraph rather than refusing the whole answer. Still produce the parts you can ground.

6. FULL REFUSAL: Only refuse the entire answer if ALL companies in the question lack supporting chunks:
   - Set "refused" to true
   - In "answer", state in one sentence what is missing
   - Leave "citations" as an empty list

7. CONCISENESS: Be focused, not exhaustive. Aim for ~2-4 sentences per company plus a short comparison.

8. GROUNDING: Every number, percentage, and named entity must come from a cited chunk whose header matches the company being discussed at that point.

OUTPUT FORMAT: You MUST respond with a JSON object matching exactly this schema:
{
  "answer": "Your structured comparison with [chunk_<id>] citations inline.",
  "citations": [list of chunk_id integers you actually cited],
  "refused": true or false
}

Do not include any text outside the JSON object.
"""


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


def _build_synthesis_user_prompt(
    question: str, chunks: list[dict], entities: list[str]
) -> str:
    entity_label = ", ".join(entities)
    return (
        f"USER QUESTION (compares companies: {entity_label}):\n{question}\n\n"
        f"CHUNKS spanning these companies:\n\n"
        f"{_format_chunks_for_prompt(chunks)}\n\n"
        f"---\n\nAnswer the question by describing each company's position "
        f"separately using only chunks from that company, then comparing. "
        f"Output JSON."
    )


def _extract_filter_kwargs(filters):
    """Translate a filters dict into Retriever search() kwargs."""
    if not filters:
        return {}
    out = {}
    if "ticker" in filters:
        out["ticker"] = filters["ticker"]
    if "fiscal_year" in filters:
        out["fiscal_year"] = filters["fiscal_year"]
    if "chunk_type" in filters:
        out["chunk_type"] = filters["chunk_type"]
    return out


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
        decompose: bool = True,
        top_k_per_entity: int = DEFAULT_TOP_K_PER_ENTITY,
        filters: Optional[dict] = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> dict:
        """Top-level: detect multi-entity questions and route accordingly.

        With decompose=True (default), uses entity detection to route between
        the single-prompt path (1 entity or 0 = out-of-corpus) and the
        decomposed path (2+ entities). With decompose=False, always uses the
        single-prompt path - useful for producing an ablation baseline.
        """
        entities = detect_entities(question) if decompose else []
        if len(entities) >= 2:
            return self._answer_decomposed(
                question=question,
                entities=entities,
                mode=mode,
                top_k_per_entity=top_k_per_entity,
                filters=filters,
                temperature=temperature,
            )
        return self._answer_single(
            question=question,
            mode=mode,
            top_k=top_k,
            filters=filters,
            temperature=temperature,
        )

    def _answer_single(
        self,
        question: str,
        mode: str,
        top_k: int,
        filters: Optional[dict],
        temperature: float,
    ) -> dict:
        """Single-question path - identical behavior to Day 9 answer()."""
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
                "decomposed": False,
                "entities": [],
                "timing_ms": {"retrieve": round(retrieve_ms, 1), "generate": 0.0},
                "usage": None,
            }

        retrieved_ids = [c["id"] for c in chunks]
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

        return self._parse_llm_response(
            response,
            retrieved_ids=retrieved_ids,
            mode=mode,
            top_k=top_k,
            decomposed=False,
            entities=[],
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
        )

    def _answer_decomposed(
        self,
        question: str,
        entities: list[str],
        mode: str,
        top_k_per_entity: int,
        filters: Optional[dict],
        temperature: float,
    ) -> dict:
        """Multi-entity path: retrieve per-entity, then one synthesis LLM call.

        Each entity gets top-K=top_k_per_entity chunks retrieved with that
        ticker filter applied. Results are concatenated (preserving entity
        ordering from detect_entities) and deduplicated. A single LLM call
        with SYNTHESIS_SYSTEM_PROMPT produces the comparison answer.
        """
        t0 = time.time()
        all_chunks: list[dict] = []
        per_entity_ids: dict[str, list[int]] = {}
        seen_ids: set[int] = set()

        for ticker in entities:
            entity_filters = dict(filters or {})
            entity_filters["ticker"] = ticker
            chunks = self._retrieve(question, mode, top_k_per_entity, entity_filters)
            per_entity_ids[ticker] = []
            for c in chunks:
                cid = c["id"]
                if cid not in seen_ids:
                    all_chunks.append(c)
                    seen_ids.add(cid)
                    per_entity_ids[ticker].append(cid)

        retrieve_ms = (time.time() - t0) * 1000

        if not all_chunks:
            return {
                "answer": "No chunks retrieved for any of the named entities (filters may be too restrictive).",
                "citations": [],
                "refused": True,
                "retrieved_chunk_ids": [],
                "mode": mode,
                "top_k": top_k_per_entity,
                "decomposed": True,
                "entities": entities,
                "entity_chunk_map": per_entity_ids,
                "timing_ms": {"retrieve": round(retrieve_ms, 1), "generate": 0.0},
                "usage": None,
            }

        retrieved_ids = [c["id"] for c in all_chunks]
        user_prompt = _build_synthesis_user_prompt(question, all_chunks, entities)

        t1 = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        generate_ms = (time.time() - t1) * 1000

        result = self._parse_llm_response(
            response,
            retrieved_ids=retrieved_ids,
            mode=mode,
            top_k=top_k_per_entity,
            decomposed=True,
            entities=entities,
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
        )
        result["entity_chunk_map"] = per_entity_ids
        return result

    def _parse_llm_response(
        self,
        response,
        retrieved_ids: list[int],
        mode: str,
        top_k: int,
        decomposed: bool,
        entities: list[str],
        retrieve_ms: float,
        generate_ms: float,
    ) -> dict:
        """Shared response-parsing logic for both single and decomposed paths."""
        raw_content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

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
                "decomposed": decomposed,
                "entities": entities,
                "timing_ms": {
                    "retrieve": round(retrieve_ms, 1),
                    "generate": round(generate_ms, 1),
                },
                "usage": usage,
                "_parse_error": True,
                "_raw": raw_content,
            }

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
            "decomposed": decomposed,
            "entities": entities,
            "timing_ms": {
                "retrieve": round(retrieve_ms, 1),
                "generate": round(generate_ms, 1),
            },
            "usage": usage,
        }


# ============================================================ CLI

def _parse_cli_args():
    p = argparse.ArgumentParser(
        description="Generate cited answers using hybrid retrieval + Llama 3.3 70B."
    )
    p.add_argument("--question", "-q", required=True, help="The question to answer.")
    p.add_argument("--mode", default=DEFAULT_MODE,
                   choices=["dense", "bm25", "hybrid"])
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--no-decompose", action="store_true",
                   help="Disable auto-decomposition (forces single-question path)")
    p.add_argument("--top-k-per-entity", type=int, default=DEFAULT_TOP_K_PER_ENTITY,
                   help="Per-entity top-K when decomposition triggers (default 5)")
    p.add_argument("--ticker", default=None)
    p.add_argument("--fiscal-year", type=int, default=None)
    p.add_argument("--chunk-type", default=None, choices=[None, "prose", "table"])
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    return p.parse_args()


def _main():
    args = _parse_cli_args()
    filters = {}
    if args.ticker:
        filters["ticker"] = args.ticker
    if args.fiscal_year:
        filters["fiscal_year"] = args.fiscal_year
    if args.chunk_type:
        filters["chunk_type"] = args.chunk_type

    gen = Generator()
    result = gen.answer(
        question=args.question,
        mode=args.mode,
        top_k=args.top_k,
        decompose=not args.no_decompose,
        top_k_per_entity=args.top_k_per_entity,
        filters=filters or None,
        temperature=args.temperature,
    )

    print("=" * 80)
    print(f"QUESTION: {args.question}")
    print("=" * 80)
    print(f"Mode: {result['mode']}   Top-K: {result['top_k']}   "
          f"Refused: {result['refused']}")
    if result.get("decomposed"):
        print(f"Decomposed: True   Entities: {result.get('entities')}")
        ecm = result.get("entity_chunk_map", {})
        for ent, ids in ecm.items():
            print(f"  {ent} chunks: {ids}")
    print(f"Retrieved chunks: {result['retrieved_chunk_ids']}")
    if result["citations"]:
        print(f"Citations from LLM: {result['citations']}")
    print("--- ANSWER ---")
    print(result["answer"])
    print("--- END ANSWER ---")
    if result.get("usage"):
        u = result["usage"]
        print(f"Usage:  prompt={u['prompt_tokens']}  "
              f"completion={u['completion_tokens']}  total={u['total_tokens']}")
    t = result["timing_ms"]
    total_ms = t.get("retrieve", 0) + t.get("generate", 0)
    print(f"Timing: retrieve={t.get('retrieve', 0):.0f}ms  "
          f"generate={t.get('generate', 0):.0f}ms  total={total_ms:.0f}ms")


if __name__ == "__main__":
    _main()