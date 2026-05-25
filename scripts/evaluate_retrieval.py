#!/usr/bin/env python3
"""Benchmark comparable-company retrieval quality.

Measures whether the retrieval layer surfaces same-sector peers. Each document
in the labeled corpus is used as a query; we retrieve the top-k most similar
*other* documents and score how many share the query's sector
(precision@k / recall@k against same-sector peers, plus mean reciprocal rank).

This benchmark needs no API key. It exercises the embedding + vector-store path
directly and can compare backends, so the value of semantic embeddings over the
lexical hash fallback is measurable rather than asserted.

Examples:
    # Compare semantic vs hash on the bundled labeled corpus
    python3 scripts/evaluate_retrieval.py --compare

    # Score a single backend and write metrics JSON
    python3 scripts/evaluate_retrieval.py --backend semantic \
        --output artifacts/retrieval_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sec_memo_agents.core.embeddings import build_embedder
from sec_memo_agents.core.vector_store import VectorStore


def load_corpus(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_backend(corpus: list[dict], backend: str, top_k: int) -> dict:
    sector_by_ticker = {doc["ticker"]: doc["sector"] for doc in corpus}
    sector_counts: dict[str, int] = {}
    for doc in corpus:
        sector_counts[doc["sector"]] = sector_counts.get(doc["sector"], 0) + 1

    embedder = build_embedder(backend=backend)
    store = VectorStore(embedder=embedder)
    store.add_texts(
        [doc["text"] for doc in corpus],
        [{"company_name": doc["company_name"], "ticker": doc["ticker"]} for doc in corpus],
    )

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    per_query: list[dict] = []

    for doc in corpus:
        query_sector = doc["sector"]
        relevant_available = sector_counts[query_sector] - 1  # exclude the query itself

        # Retrieve a little extra, then drop the query document if it returns itself.
        results = store.search(doc["text"], top_k=top_k + 1)
        results = [r for r in results if r.ticker != doc["ticker"]][:top_k]

        hits = [1 if sector_by_ticker.get(r.ticker) == query_sector else 0 for r in results]
        precision = sum(hits) / len(hits) if hits else 0.0
        recall = (sum(hits) / relevant_available) if relevant_available else 0.0
        reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(hits, start=1) if hit), 0.0)

        precision_scores.append(precision)
        recall_scores.append(recall)
        mrr_scores.append(reciprocal_rank)
        per_query.append(
            {
                "query": doc["company_name"],
                "sector": query_sector,
                "retrieved": [f"{r.company_name} [{sector_by_ticker.get(r.ticker)}]" for r in results],
                "precision_at_k": round(precision, 3),
            }
        )

    n = len(corpus)
    return {
        "backend": store.backend,
        "semantic": getattr(embedder, "semantic", False),
        "top_k": top_k,
        "queries": n,
        "precision_at_k": round(sum(precision_scores) / n, 3),
        "recall_at_k": round(sum(recall_scores) / n, 3),
        "mrr": round(sum(mrr_scores) / n, 3),
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/retrieval_benchmark.jsonl", help="Labeled corpus JSONL.")
    parser.add_argument("--backend", default="auto", choices=["auto", "semantic", "hash"], help="Embedding backend.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of comparables retrieved per query.")
    parser.add_argument("--compare", action="store_true", help="Run both semantic and hash backends side by side.")
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corpus = load_corpus(Path(args.corpus))

    if args.compare:
        semantic = evaluate_backend(corpus, "semantic", args.top_k)
        lexical = evaluate_backend(corpus, "hash", args.top_k)
        report = {
            "corpus_size": len(corpus),
            "top_k": args.top_k,
            "semantic": semantic,
            "hash": lexical,
            "semantic_precision_gain": round(semantic["precision_at_k"] - lexical["precision_at_k"], 3),
        }
    else:
        report = evaluate_backend(corpus, args.backend, args.top_k)

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
