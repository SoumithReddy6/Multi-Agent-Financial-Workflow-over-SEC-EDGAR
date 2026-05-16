#!/usr/bin/env python3
"""Generate a schema-valid investment memo from SEC filings or offline fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sec_memo_agents.agents.workflow import FinancialWorkflow
from sec_memo_agents.schemas import MemoRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol or SEC CIK.")
    parser.add_argument("--template", default="deal_screening", help="Workflow template name.")
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q"], help="Filing forms to consider.")
    parser.add_argument("--filing-limit", type=int, default=2, help="Number of filings to ingest.")
    parser.add_argument("--top-k", type=int, default=5, help="Comparable snippets to retrieve.")
    parser.add_argument("--offline", action="store_true", help="Use bundled fixtures instead of live SEC calls.")
    parser.add_argument("--question", default=None, help="Optional analyst focus question.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = MemoRequest(
        ticker=args.ticker,
        template_name=args.template,
        forms=args.forms,
        filing_limit=args.filing_limit,
        top_k_comparables=args.top_k,
        offline=args.offline,
        analyst_question=args.question,
    )
    memo = FinancialWorkflow().run(request)
    payload = memo.model_dump(mode="json")
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
