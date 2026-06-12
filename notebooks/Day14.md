# Day 14 — Deploy + UI redesign

**Status:** Shipped. v1.0.0 tagged. Live at https://huggingface.co/spaces/25Sad/financial-rag

This was the longest day of the 14 by a wide margin. Not because the work was hard, but because Hugging Face Spaces has a lot of edges and we hit most of them. The actual code changes were minimal: a UI restyle, a Dockerfile, README frontmatter. Everything else was deploy mechanics.

## What shipped

- **Streamlit UI redesign** — financial-slate palette, hero band with eyebrow + stat readouts, citation pills inline in answers, chunk cards grouped by entity for decomposed queries, decomposed banner instead of `st.info`. Stage 6 logic preserved end-to-end.
- **Public HF Space** with Docker SDK, CPU Basic hardware, GROQ_API_KEY as secret. Auto-rebuilds on push to `space` remote.
- **LFS-tracked binaries** — `qdrant_data/collection/financial_rag/storage.sqlite` (37 MB) and 4 PNGs (~3 MB total). `.gitattributes` tracks `*.sqlite` and `*.png`.
- **v1.0.0 tag** on both GitHub and HF, marking the release.
- **README live demo link** active. Video link still placeholder (recording deferred).

## Stages

### 0. UI redesign (before deploy)
Replaced `app.py` end-to-end. Theme: Space Grotesk (display), Inter (body), JetBrains Mono (monospace). Ink `#0B1929`, blue `#5B8DEF`, teal `#3FB68B` for `[CITED]` markers, amber `#E4B363` for refusals. `app.py.stage6_backup` retained locally.

Minor regressions caught and fixed: sidebar collapse button disappeared (header over-hidden), selectbox cursor was I-beam (added `cursor: pointer`).

### A. Pre-deploy cleanup
Caught broken image references: `docs/` was untracked despite Day 13's README referencing screenshots there. Live README on GitHub was showing broken images. Removed `qdrant_data/` from `.gitignore`, committed everything in one combined commit (15.55 MiB).

### B. HF Space creation
HF deprecated standalone Streamlit SDK in April 2025. Correct path now: Docker template → select Streamlit sub-template. Created `25Sad/financial-rag`, CPU Basic hardware (free), public. Default template demo rendered cleanly on first load — confirmed Docker + port 8501 work.

### C. Push gauntlet
Five rejected pushes before one landed:

1. **Auth fail** — `huggingface-cli login` is deprecated. Renamed to `hf auth login`. Credentials saved to Windows credential manager after re-login.
2. **File size rejection** — `storage.sqlite` exceeded HF's 10 MiB per-file limit. Fix: `git lfs migrate import --include="*.sqlite" --everything`. Rewrote 21 commits.
3. **Binary file rejection** — HF requires LFS/Xet for ALL binary files regardless of size, including PNGs. Fix: second migration with `--include="*.png" --everything`. Rewrote 42 commits cumulatively.
4. **YAML validation #1** — `emoji: ◆` failed Extended_Pictographic check. Geometric shapes aren't emojis to HF. Switched to `🔷`.
5. **YAML validation #2** — `short_description` was 65 chars, max is 60. Shortened to "Hybrid RAG over SEC 10-K filings with decomposition" (50 chars).

Sixth push landed. `+ c60e7eb...6fc2efc main -> main (forced update)` — that's the line we worked toward.

### D. Docker build
First build failed at `pip install` on `pywin32==311`. Windows-only, no Linux wheel. Removed from `requirements.txt`, pushed.

Second build succeeded. ~12 minutes total: apt-get build-essential (~8s), pip install ~160 packages (~10 min), copy + start (~30s).

### E. Smoke tests
All three test cases passed on the live deploy:

1. **Single-entity** — "What was Apple's iPhone net sales in fiscal year 2024?" → `$201,183 million` cited from chunk_44 + chunk_96. 3,806 tokens.
2. **Cross-company decomposition** — "Data center revenue for NVIDIA and Microsoft?" → decomposed banner showed `Query split into 2 entities: NVDA, MSFT`. Cited chunks from both companies (chunk_2134, chunk_2423 for NVDA; chunk_1785 for MSFT). 4,233 tokens.
3. **Refusal** — "How did Google's R&D change year over year?" → `The corpus does not cover Google's filings`. Grounding works.

## Key decisions and trade-offs

- **`requirements.txt` is bloated** (~160 packages, much of it inherited from the shared venv's LangChain/LlamaIndex/Bangla-PDF history). v1.0 ships as-is. Slim down to ~12 direct deps in v2 — should cut cold-start time meaningfully.
- **Chunker table-cell bug** (Day 12: `\s+` matches newlines and merges adjacent tables across blank lines) remains deferred to v2. One-character fix exists but invalidates eval set chunk IDs, Day 8 metrics, Day 10 ablation, and Day 11 q18 fix narrative. Not worth re-running the full eval set on deploy day.
- **Video recording deferred.** VIDEO_LINK placeholder reads "coming soon". Plan: 2-3 minute Loom, covering hero → three queries → architecture overview → GitHub link.

## What I learned about HF Spaces deployment

- The path to deployment is well-documented but failure modes aren't surfaced in advance. Each rejection had a clear error message, but knowing the full set up front would have saved 90 minutes.
- HF's LFS/Xet enforcement on binary files is stricter than GitHub's. PNG screenshots that GitHub accepts without LFS get rejected by HF.
- `git lfs migrate import` is destructive (rewrites history) but safe for solo repos. Force pushes to both remotes are required after.
- The `hf` CLI is what to use now; tutorials still referencing `huggingface-cli` are out of date.
- Windows-frozen `requirements.txt` will fail on Linux containers. `pywin32`, `pywin32-ctypes`, `pywinpty`, `pyreadline3`, and `winrt-*` are the usual suspects.

## What's next

- **Record + ship demo video** (1-2 hours). Tag v1.0.1 with VIDEO_LINK updated.
- **v2 backlog:** chunker table-cell fix, requirements.txt slim-down, possibly async sub-query retrieval for decomposition.
- **Next project:** multi-agent operations assistant for Bangladesh e-commerce SMEs (per long-term roadmap). Financial RAG goes on the portfolio as a complete, deployed, citable artifact.

## Tokens used today

Effectively zero Groq tokens on deploy mechanics. Smoke tests on the deployed app consumed ~11k (3,806 + 4,233 + 2,852). Days 12-14 combined: ~60k of 300k available across three days. Comfortable margins throughout.