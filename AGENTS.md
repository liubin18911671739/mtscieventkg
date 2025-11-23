# Repository Guidelines

## Project Structure & Module Organization
- `config.yaml` holds concepts, time ranges, languages, model choices, and training knobs—document edits to keep runs reproducible.
- `src/00_download_openalex.py` → `src/06_frontier_detect.py` are ordered CLI steps; each consumes `data/` outputs from the prior stage.
- `src/utils/` hosts shared helpers (prompt templates, text I/O, embeddings, KG ops, metrics, runtime stats); reuse instead of cloning logic.
- `data/` is workspace output only (`raw/`, `extracted/`, `kg/`, `splits/`); avoid committing real datasets. Dummy data auto-generates when downloads fail.

## Build, Test, and Development Commands
- Isolate env and install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Dummy-data smoke: `python src/00_download_openalex.py --force-dummy && python src/01_build_corpus.py && python src/02_llm_event_extract.py --provider local` to populate `data/` without API keys.
- Real run: export `ZHIPUAI_API_KEY` or `OPENAI_API_KEY`, then execute steps 00→06; each prints `[stats]` with wall time and RSS.
- Use `--help` on any script for overrides.

## Coding Style & Naming Conventions
- Python 3.10+, PEP8, 4-space indents; `snake_case` for functions/vars, caps for constants, and short docstrings where behavior is non-obvious.
- Prefer `pathlib.Path`, `argparse`, and type hints; keep CLI entrypoints in `main()` and guard with `if __name__ == "__main__":`.
- Seed randomness (`seed_all`) for reproducibility, and reuse `track_stats()` for timing/memory instead of ad-hoc prints. Comment only when logic is not self-evident.

## Testing Guidelines
- No formal suite yet; smoke-test on dummy data and confirm key artifacts (`data/raw/openalex_*.jsonl`, `data/extracted/events_*.jsonl`, `data/kg/triples.tsv`, `data/kg/align_pairs.tsv`) are non-empty and parseable.
- For new utilities, add small `pytest` cases under `tests/` (create as needed) using synthetic inputs and fixed seeds.
- When touching training/alignment, record before/after metrics (loss, `dfa_metrics`, frontier outputs) in the PR to aid manual regression checks.

## Commit & Pull Request Guidelines
- History uses short imperative messages (e.g., "first"); keep commits scoped and present-tense.
- PRs should note purpose, commands run, notable stats, whether dummy or real data was used, and any config changes; avoid committing generated `data/` outputs.
- Reference related issues and flag external requirements (API keys, large models) needed to reproduce the run.

## Security & Configuration Tips
- Do not commit API keys or real datasets; rely on env vars (`ZHIPUAI_API_KEY`, `OPENAI_API_KEY`) and keep secrets out of `config.yaml`.
- Scripts default to CPU unless `train.device` requests CUDA; keep CPU fallbacks intact when adding GPU-centric changes.
- When adjusting prompts or schemas, update `src/utils/prompt_templates.py` and downstream parsing together to prevent runtime mismatches.
