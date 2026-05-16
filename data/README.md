# Data Directory

This directory stores local inputs and generated artifacts for the SEC workflow.

- `sample_universe.csv`: 30 public tickers for a 200+ filing crawl when using `--limit-per-company 8`.
- `sample_queries.jsonl`: offline benchmark prompts for memo completion and schema validation.

Large filing downloads, vector indexes, and evaluation artifacts should be written to `artifacts/` or `.cache/`, not committed.
