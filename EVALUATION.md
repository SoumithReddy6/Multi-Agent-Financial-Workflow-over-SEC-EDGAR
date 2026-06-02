# Agent Evaluation

This project evaluates the SEC-EDGAR workflow as a non-deterministic agent system, not just a schema generator. The harness runs 50 analyst-style financial queries through the parser, retrieval, and synthesis workflow, then scores task completion, answer coverage, citation quality, hallucination risk, latency, and failure modes.

## Reproduce

```bash
EMBEDDING_BACKEND=hash python3 scripts/evaluate_agent_quality.py \
  --offline \
  --output artifacts/agent_evaluation_metrics.json
```

The offline mode uses bundled SEC-style fixtures and a deterministic hash embedder so the benchmark is repeatable without API keys or network access. For live SEC runs, remove `--offline` and set a valid `SEC_USER_AGENT`.

## Current Results

Run date: May 30, 2026. This harness runs on the **offline deterministic path**
(template synthesis + bundled fixtures), chosen so the benchmark is reproducible
with no API keys or network. The primary signals are **answer accuracy** and the
**failure-mode count**; the schema/citation/hallucination figures are guaranteed
by construction in offline mode (the synthesis fallback always emits a valid,
templated memo), so they reflect the pipeline's correctness, not LLM reasoning
quality. A live-LLM run over real SEC filings is the next step.

| Metric | Result | Interpretation |
| --- | ---: | --- |
| Test queries | 50 | Analyst-style questions across 6 memo templates |
| **Answer accuracy** | **85.6%** | Primary signal — concept coverage vs expected answers |
| **Failure modes detected** | **16** | Primary signal — 13 wrong-section extraction, 3 incomplete; surfaces real retrieval weaknesses |
| Task completion | 100.0% | Guaranteed in offline mode (synthesis always returns a schema-valid memo) |
| Citation correctness | 100.0% | Guaranteed in offline mode (citations templated from filing metadata) |
| Hallucination rate | 0.0% | Guaranteed in offline mode (deterministic template, no generation) |
| Average latency | 0.0038s | Offline path (no LLM call; live LLM latency is seconds) |

## What Is Measured

| Dimension | Measurement Method |
| --- | --- |
| Task completion | Memo validates against the Pydantic `InvestmentMemo` schema and includes required core fields |
| Answer accuracy | Fraction of expected financial concepts present in the generated memo JSON |
| Citation correctness | Presence of expected SEC citation terms in `source_citations` |
| Hallucination rate | Flags forbidden terms, missing citations, or key financial metrics without sources |
| Failure modes | Classifies wrong section extraction, hallucinated metric, missing comparable, incomplete answer, API failure, and schema failure |

## Failure-Mode Findings

| Failure Mode | Count | Interpretation | Mitigation |
| --- | ---: | --- | --- |
| `wrong_section_extracted` | 13 | Mostly MD&A and earnings-quality questions where the offline fixture has limited section labels, so the answer is valid but section coverage is incomplete. | Add finer MD&A/risk/cash-flow section extraction rules and use SEC item heading normalization. |
| `answer_incomplete` | 3 | The memo completed but missed enough expected terms to fall below the keyword coverage threshold. | Add query-specific retrieval prompts and template-level required evidence checks before synthesis. |
| `hallucinated_metric_or_claim` | 0 | No unsupported metrics or missing citations were detected. | Keep source-required metric validation in the synthesis schema. |
| `citation_missing_or_incorrect` | 0 | SEC citations were present in all 50 cases. | Keep citation validation as a release gate. |

## Evaluation Files

- `data/financial_eval_queries.jsonl`: 50 test queries with expected concepts and failure constraints.
- `sec_memo_agents/evaluation/agent_quality.py`: scoring and failure-mode classification logic.
- `scripts/evaluate_agent_quality.py`: command-line runner.
- `artifacts/agent_evaluation_metrics.json`: generated run output, ignored by Git because artifacts are reproducible.

## Next Improvements

- Expand the query set from 50 AAPL fixture cases to 200 live multi-company filings.
- Add LLM-as-judge scoring for reasoning quality while keeping deterministic citation and schema checks.
- Add regression gates in CI once API-backed evaluation credentials are available.
