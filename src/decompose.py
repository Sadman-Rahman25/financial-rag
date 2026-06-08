"""Entity detection for query decomposition.

Detects which corpus companies are named in a user question, by case-insensitive
regex match on company names and tickers. Returns tickers in order of first
appearance, deduplicated.

The corpus covers AAPL, MSFT, NVDA, TSLA, META. Common name variants and the
old "Facebook" name for Meta are all recognized.

Usage:
    from src.decompose import detect_entities
    entities = detect_entities("Compare NVIDIA and Microsoft data center focus")
    # -> ["NVDA", "MSFT"]
"""
from __future__ import annotations
import re
from typing import List

# Each ticker maps to a list of name variants that should detect it.
# All variants are matched case-insensitively with word boundaries.
ENTITY_VARIANTS: dict[str, list[str]] = {
    "AAPL": ["Apple", "AAPL"],
    "MSFT": ["Microsoft", "MSFT"],
    "NVDA": ["NVIDIA", "Nvidia", "NVDA"],
    "TSLA": ["Tesla", "TSLA"],
    "META": ["Meta", "Facebook", "META"],
}

# Flat lookup: lowercase variant -> ticker
_VARIANT_TO_TICKER: dict[str, str] = {
    variant.lower(): ticker
    for ticker, variants in ENTITY_VARIANTS.items()
    for variant in variants
}

# Single compiled regex matching any variant with word boundaries.
# Sort by length descending so longer variants match first (defensive).
_ALL_VARIANTS = sorted(_VARIANT_TO_TICKER.keys(), key=len, reverse=True)
_ENTITY_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in _ALL_VARIANTS) + r")\b",
    flags=re.IGNORECASE,
)


def detect_entities(question: str) -> List[str]:
    """Detect corpus entities mentioned in the question.

    Returns tickers in order of first appearance, deduplicated. Returns an
    empty list if no recognized entities are found (e.g., a question about
    Google or Amazon, neither of which is in the corpus).
    """
    seen: list[str] = []
    for match in _ENTITY_PATTERN.finditer(question):
        ticker = _VARIANT_TO_TICKER[match.group(0).lower()]
        if ticker not in seen:
            seen.append(ticker)
    return seen


if __name__ == "__main__":
    # Self-test - run with: python -m src.decompose
    cases: list[tuple[str, list[str]]] = [
        # Single-entity (should route to single-question path)
        ("What was Apple's iPhone net sales in fiscal year 2024?", ["AAPL"]),
        ("What does AAPL's 10-K say about supply chain?", ["AAPL"]),
        ("What is Meta's Reality Labs revenue?", ["META"]),

        # Cross-company (should trigger decomposition)
        ("Compare NVIDIA's and Microsoft's data center business focus.", ["NVDA", "MSFT"]),
        ("Compare Tesla's and NVIDIA's R&D spending priorities.", ["TSLA", "NVDA"]),
        ("How do Microsoft and NVIDIA describe their AI strategy?", ["MSFT", "NVDA"]),
        ("How did MSFT, NVDA, and TSLA spend on R&D?", ["MSFT", "NVDA", "TSLA"]),

        # Out-of-corpus (return empty -> caller refuses via ENTITY MATCH rule)
        ("What was Google's revenue in 2024?", []),
        ("How did Amazon perform in 2024?", []),

        # Case sensitivity (lowercase, uppercase)
        ("apple and microsoft", ["AAPL", "MSFT"]),
        ("APPLE AND MICROSOFT", ["AAPL", "MSFT"]),

        # Deduplication (same entity mentioned multiple times)
        ("Apple Apple Apple", ["AAPL"]),

        # All five companies in one question, mixed names
        ("Tesla and Meta and NVIDIA and Microsoft and Apple",
         ["TSLA", "META", "NVDA", "MSFT", "AAPL"]),

        # Old name handling (Facebook -> META)
        ("Facebook's pivot to Reality Labs", ["META"]),
    ]

    print(f"Running {len(cases)} entity-detection test cases...\n")
    n_pass, n_fail = 0, 0
    for i, (q, expected) in enumerate(cases, 1):
        got = detect_entities(q)
        ok = got == expected
        status = "OK  " if ok else "FAIL"
        n_pass += int(ok)
        n_fail += int(not ok)
        truncated_q = q if len(q) <= 55 else q[:52] + "..."
        print(f"  [{i:>2}] {status}  {truncated_q!r}")
        if not ok:
            print(f"           expected: {expected}")
            print(f"           got:      {got}")

    print()
    print(f"Results: {n_pass}/{len(cases)} passed, {n_fail} failed.")
    if n_fail == 0:
        print("All tests passed. Stage 1 complete.")
    else:
        print("Some tests failed. Review before continuing to Stage 2.")
        raise SystemExit(1)

