# Day 13 — README + architecture diagram

**Date:** 2026-06-11
**Status:** Shipped. README complete with TL;DR, problem statement, architecture diagram (Mermaid), tech stack, key results with ablation tables and screenshots, six portfolio stories, how-to-run, limitations & next steps, acknowledgments. Ready for Day 14 (HF Spaces deploy + demo video).

## Decisions taken at start of day

1. **Architecture diagram format: Mermaid inline.** Renders natively on GitHub, version-controlled, no binary export step.
2. **README style: punchy + collapsible.** Scannable summaries on top of each section; `<details>` blocks for deep dives.
3. **Three screenshots:** hero (single-entity), decomposition (cross-company), refusal (out-of-corpus).
4. **Demo + video link placeholders for Day 14:** clearly marked TODO at the top of the README.

## What shipped

- `README.md` (~10 KB) — full project documentation, structured for the recruiter-first scan
- `docs/screenshots/Hero_01.png`, `Decomposition_02.png`, `Refusal_03.png` — three Streamlit captures
- Inline Mermaid architecture diagram showing indexing pipeline, stores, and query pipeline
- Six portfolio stories with `<details>` blocks for deep dives
- Limitations & next steps section honestly documenting the chunker bug, q18 retrieval bug, and other known issues

### README structure

1. TL;DR (5 bullets + demo/video placeholders)
2. The problem (the 4 challenges of SEC 10-K RAG)
3. What it does (single-entity vs cross-company routing in one paragraph)
4. Architecture (Mermaid diagram + 3-bullet explanation)
5. Tech stack (12-row table)
6. Key results (coverage + retrieval ablation + decomposition ablation + 2 screenshots)
7. The interesting decisions (6 portfolio stories with `<details>`)
8. How to run locally (prerequisites + setup + run + rebuild + utility commands)
9. Limitations & next steps (6 limitations, 6 next steps ordered by ROI)
10. Acknowledgments + Contact

## Tokens used today

**Zero Groq tokens.** Day 13 was pure documentation. No LLM calls.

## What's next

### Day 14 — HF Spaces deploy + demo video (4-6 hours)

1. **HF Spaces deploy:**
   - Create Space at `huggingface.co/spaces/Sadman-Rahman25/financial-rag`
   - Push repo as the Space's source
   - Configure Space secrets: `GROQ_API_KEY`
   - Decide on `qdrant_data/` distribution — either (a) commit to the Space repo, (b) attach as HF Dataset, or (c) auto-rebuild on first launch
   - Verify cold-start completes in <3 min
   - Run all 3 example queries end-to-end on the deployed app
   - Replace `DEMO_LINK` placeholder in README.md with the Space URL

2. **Demo video (2-3 min):**
   - 0:00-0:15 — what this is (RAG over SEC 10-Ks)
   - 0:15-0:45 — three example queries (single-entity, cross-company, refusal)
   - 0:45-1:30 — show retrieval/decomposition mechanics (open the chunks panel, point at [CITED] markers)
   - 1:30-2:00 — architecture diagram and key results
   - 2:00-2:30 — link to GitHub + closing
   - Recording tool: Loom or OBS. Don't over-produce.
   - Upload (Loom direct or YouTube), replace `VIDEO_LINK` placeholder in README.md

3. **Final commit and tag:**
   - `git tag v1.0.0 -m "Project complete: 14-day Financial RAG build"`
   - `git push --tags`
   - Project complete. Move to portfolio + applications.

## Notes for Day 14

- HF Spaces' free-tier 16GB RAM should comfortably fit BGE-large (1.34GB) + Qdrant + Streamlit + Groq client
- Cold start will be ~2-3 min on first user after idle — that's the persistent disadvantage of free-tier
- If `qdrant_data/` is too large to commit, the easiest fallback is option (c) above: bake a "rebuild index from chunks.jsonl on first run" path into `Generator.__init__` that checks if Qdrant is empty and runs the embedding pipeline if so. Adds ~71 min to first cold start but the Space becomes self-contained.