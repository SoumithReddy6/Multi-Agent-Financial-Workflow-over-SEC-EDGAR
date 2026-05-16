#!/usr/bin/env python3
"""Evaluate memo completion, schema validity, and latency on benchmark queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sec_memo_agents.agents.workflow import FinancialWorkflow
from sec_memo_agents.evaluation.metrics import run_queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", default="data/sample_queries.jsonl", help="JSONL benchmark queries.")
    parser.add_argument("--offline", action="store_true", help="Use bundled fixtures instead of live SEC calls.")
    parser.add_argument("--output", default="artifacts/evaluation_metrics.json", help="Metrics JSON output path.")
    return parser.parse_args()


def load_queries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    workflow = FinancialWorkflow()
    metrics = run_queries(load_queries(Path(args.queries)), workflow.run, offline=args.offline)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(metrics, indent=2)
    output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
