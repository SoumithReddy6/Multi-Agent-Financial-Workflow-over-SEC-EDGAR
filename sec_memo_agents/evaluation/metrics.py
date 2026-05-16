"""Benchmark metrics for memo completion, schema validity, and latency."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import ceil
from typing import Callable

from pydantic import ValidationError

from sec_memo_agents.schemas import InvestmentMemo, MemoRequest


@dataclass
class QueryResult:
    query_id: str
    schema_valid: bool
    completed: bool
    latency_seconds: float
    error: str | None = None


def score_memo(memo: InvestmentMemo) -> bool:
    checks = [
        bool(memo.executive_summary),
        bool(memo.business_overview),
        bool(memo.key_financials),
        bool(memo.risks),
        bool(memo.source_citations),
        0 <= memo.confidence <= 1,
    ]
    return sum(checks) >= 5


def run_queries(
    queries: list[dict],
    runner: Callable[[MemoRequest], InvestmentMemo],
    offline: bool = True,
) -> dict:
    results: list[QueryResult] = []
    for query in queries:
        started = time.perf_counter()
        try:
            request = MemoRequest(
                ticker=query["ticker"],
                template_name=query.get("template_name", "deal_screening"),
                analyst_question=query.get("question"),
                offline=offline,
            )
            memo = runner(request)
            InvestmentMemo.model_validate(memo.model_dump())
            completed = score_memo(memo)
            results.append(
                QueryResult(
                    query_id=query["id"],
                    schema_valid=True,
                    completed=completed,
                    latency_seconds=round(time.perf_counter() - started, 4),
                )
            )
        except (ValidationError, Exception) as exc:
            results.append(
                QueryResult(
                    query_id=query.get("id", "unknown"),
                    schema_valid=False,
                    completed=False,
                    latency_seconds=round(time.perf_counter() - started, 4),
                    error=str(exc),
                )
            )

    total = len(results) or 1
    completed = sum(1 for item in results if item.completed)
    schema_valid = sum(1 for item in results if item.schema_valid)
    latency_values = [item.latency_seconds for item in results]
    p95_index = max(0, min(total - 1, ceil(0.95 * total) - 1))
    return {
        "queries": total,
        "completed": completed,
        "completion_rate": round(completed / total, 4),
        "schema_valid": schema_valid,
        "schema_valid_rate": round(schema_valid / total, 4),
        "avg_latency_seconds": round(sum(latency_values) / total, 4),
        "p95_latency_seconds": round(sorted(latency_values)[p95_index], 4),
        "results": [item.__dict__ for item in results],
    }
