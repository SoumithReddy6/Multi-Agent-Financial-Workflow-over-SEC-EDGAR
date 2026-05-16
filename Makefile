.PHONY: install test api demo ingest evaluate

install:
	pip install -r requirements.txt

test:
	pytest

api:
	uvicorn sec_memo_agents.api.app:app --reload

demo:
	python3 scripts/generate_memo.py --ticker AAPL --template deal_screening --offline

ingest:
	python3 scripts/ingest_sec_filings.py --universe data/sample_universe.csv --forms 10-K 10-Q --limit-per-company 8 --target-filings 200 --output artifacts/sec_index

evaluate:
	python3 scripts/evaluate_workflow.py --offline --queries data/sample_queries.jsonl --output artifacts/evaluation_metrics.json
