#!/usr/bin/env python3
"""Ingest SEC filings for a ticker universe and build a local comparable index."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sec_memo_agents.agents.parser import ParserAgent
from sec_memo_agents.agents.retrieval import RetrievalAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="data/sample_universe.csv", help="CSV with a ticker column.")
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q"], help="SEC form types to ingest.")
    parser.add_argument("--limit-per-company", type=int, default=8, help="Filings per ticker.")
    parser.add_argument("--target-filings", type=int, default=200, help="Stop after this many filings.")
    parser.add_argument("--output", default="artifacts/sec_index", help="Output directory for index and manifest.")
    return parser.parse_args()


def read_tickers(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row["ticker"].strip().upper() for row in reader if row.get("ticker")]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = read_tickers(Path(args.universe))
    parser = ParserAgent()
    filings = parser.ingest_universe(
        tickers,
        forms=args.forms,
        limit_per_company=args.limit_per_company,
        target_filings=args.target_filings,
    )

    retrieval = RetrievalAgent()
    chunks = retrieval.index_filings(filings)
    retrieval.vector_store.save(output_dir / "vector_store.json")

    manifest = {
        "tickers_requested": len(tickers),
        "filings_processed": len(filings),
        "chunks_indexed": chunks,
        "forms": args.forms,
        "vector_backend": retrieval.vector_store.backend,
        "filings": [filing.metadata.model_dump() for filing in filings],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
