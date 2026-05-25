# Multi-Agent Financial Workflow over SEC EDGAR

[![Python](https://img.shields.io/badge/Python-3.11-1f6feb?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Workflow-1c7ed6)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-2f9e44)](https://faiss.ai/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Tool_Calling-111827?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?logo=docker&logoColor=white)](https://www.docker.com/)
[![SEC EDGAR](https://img.shields.io/badge/Data-SEC_EDGAR-b08900)](https://www.sec.gov/edgar)

Finance-focused multi-agent AI workflow for ingesting SEC 10-K and 10-Q filings, retrieving comparable-company evidence, and producing schema-valid investment memos.

The project is designed as a portfolio-ready system for finance-AI roles: it uses public SEC EDGAR data, agent boundaries, a reusable LangGraph workflow, FAISS-compatible retrieval, FastAPI serving, and JSON-schema-enforced memo outputs.

## What It Builds

- Parser agent: pulls 10-K and 10-Q metadata, filing documents, and XBRL facts from SEC EDGAR.
- Retrieval agent: chunks filings, embeds them with a local sentence-transformer model (`all-MiniLM-L6-v2`) for true semantic search, exposes LangChain `Document` adapters, and retrieves comparable-company peer snippets through FAISS. Falls back to a deterministic lexical embedder only when the model is unavailable.
- Synthesis agent: emits structured investment memos validated by Pydantic schemas.
- Function-calling schemas: exports OpenAI and Anthropic tool schemas for strict memo generation.
- Workflow templates: reusable finance playbooks for deal screening, credit memos, due diligence, earnings quality, covenant risk, and MD&A analysis.
- FastAPI surface: endpoints for health checks, template inspection, ingestion, memo generation, and schema export.
- Evaluation harness: checks schema validity, answer completion, citation coverage, and latency.

## Why This Matters

Salt AI-style finance AI roles often screen for practical exposure to finance workflows, SEC filings, RAG, and agent orchestration. This project turns that requirement into a concrete artifact: a working SEC filing workflow that can process real public filings and produce investor-style memo outputs.

## Architecture

```mermaid
flowchart LR
    A["SEC EDGAR APIs"] --> B["Parser Agent"]
    B --> C["Clean Filing Text + XBRL Facts"]
    C --> D["Retrieval Agent"]
    D --> E["FAISS / Local Vector Store"]
    E --> F["Comparable Evidence"]
    C --> G["Synthesis Agent"]
    F --> G
    H["Workflow Template"] --> G
    G --> I["Schema-Valid Investment Memo JSON"]
    I --> J["FastAPI / CLI / Evaluation"]
```

## Demo

![SEC EDGAR memo preview](docs/demo_memo_preview.svg)

Generate the same schema-valid offline demo locally:

```bash
python3 scripts/generate_memo.py --ticker AAPL --template deal_screening --offline --output artifacts/demo_memo.json
```

## SEC Data Sources

This project uses the SEC public data APIs documented by the SEC:

- SEC EDGAR API overview: `https://www.sec.gov/search-filings/edgar-application-programming-interfaces`
- SEC fair-access guidance: `https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm`
- Submissions history: `https://data.sec.gov/submissions/CIK##########.json`
- Company facts: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`
- Filing documents: `https://www.sec.gov/Archives/edgar/data/...`
- Ticker map: `https://www.sec.gov/files/company_tickers.json`

The SEC requires fair-access behavior. The client sets a declared `User-Agent`, supports gzip, and rate-limits requests to stay under the SEC stated maximum of 10 requests per second.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```bash
SEC_USER_AGENT="Soumith Reddy soumithreddy3003@gmail.com"
```

Run the API:

```bash
uvicorn sec_memo_agents.api.app:app --reload
```

Generate an offline demo memo from fixtures:

```bash
python3 scripts/generate_memo.py --ticker AAPL --template deal_screening --offline
```

Run tests:

```bash
pytest
```

Run the offline workflow benchmark:

```bash
python3 scripts/evaluate_workflow.py --offline --queries data/sample_queries.jsonl --output artifacts/evaluation_metrics.json
```

Run the retrieval-quality benchmark (no API key; compares semantic vs lexical):

```bash
python3 scripts/evaluate_retrieval.py --compare --output artifacts/retrieval_metrics.json
```

## Real SEC Ingestion

The ingestion script can process a ticker universe and build a local vector index.

```bash
python3 scripts/ingest_sec_filings.py \
  --universe data/sample_universe.csv \
  --forms 10-K 10-Q \
  --limit-per-company 8 \
  --target-filings 200 \
  --output artifacts/sec_index
```

The target of 200 filings is achieved by combining the provided ticker universe with `--limit-per-company 8`. The script respects SEC rate limits and caches downloaded documents.

## API Examples

List workflow templates:

```bash
curl http://localhost:8000/templates
```

Create a memo:

```bash
curl -X POST http://localhost:8000/memos \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "template_name": "deal_screening",
    "forms": ["10-K", "10-Q"],
    "filing_limit": 2,
    "top_k_comparables": 5,
    "offline": false
  }'
```

Export an OpenAI function-calling schema:

```bash
curl http://localhost:8000/schemas/openai/investment_memo
```

## Metrics

Two things are measured honestly and separately: **retrieval quality** (no API key
needed) and the **workflow harness** (schema validity, latency, templates, tests).

### Retrieval quality (measured)

On a labeled multi-sector corpus, `scripts/evaluate_retrieval.py --compare`
scores whether the system surfaces true same-sector comparable companies.

| Metric | Lexical hash baseline | Semantic (`all-MiniLM-L6-v2`) |
| --- | ---: | ---: |
| MRR (top comparable is a true peer) | 0.32 | **1.00** |
| Recall@3 | 0.29 | **0.88** |
| Precision@3 | 0.19 | **0.58** |

Reproduce: `python3 scripts/evaluate_retrieval.py --compare --output artifacts/retrieval_metrics.json`
(Precision@3 is capped at 0.67 here because each sector has only two retrievable
peers, so the third slot is necessarily off-sector; recall@3 and MRR are the
cleaner signals.)

### Workflow harness

| Metric | Current | How to reproduce |
| --- | ---: | --- |
| Reusable workflow templates | 6 | `sec_memo_agents/templates/*.yaml` |
| Schema-valid JSON outputs | 100% | Pydantic `InvestmentMemo` validation enforces this by construction |
| Automated tests | 10 passing, 1 semantic test skipped in CI | `pytest` |
| Offline demo latency | p95 < 0.1s | deterministic template fill, no LLM call — `python3 scripts/evaluate_workflow.py --offline` |

> Honesty notes: The offline workflow benchmark exercises a **deterministic
> template-fill fallback** so the system can be demoed without API keys; its
> "100% schema-valid" result is guaranteed by Pydantic validation and is not a
> measure of analysis quality. Real memo-quality metrics require an LLM key
> (`LLM_PROVIDER=openai|anthropic`) and the live SEC path. The 200+ filing crawl
> is supported by `scripts/ingest_sec_filings.py` but is run on demand, not a
> pre-computed result.

## Repository Layout

```text
.github/
  workflows/       GitHub Actions CI
  ISSUE_TEMPLATE/  GitHub issue template
sec_memo_agents/
  agents/          parser, retrieval, synthesis, workflow orchestration
  api/             FastAPI application
  core/            SEC client, vector store, text utilities, templates
  data/            local data helpers
  evaluation/      benchmark metrics
  templates/       reusable finance workflow templates
scripts/           CLI ingestion, memo generation, evaluation
tests/             unit tests for schemas, retrieval, templates, synthesis
configs/           default runtime config
docs/              architecture, demo screenshot, resume positioning notes
```

## LLM Providers

The system runs in three modes:

- `offline`: deterministic synthesis for demos, tests, and portfolio walkthroughs.
- `openai`: GPT model with OpenAI tool/function schema validation.
- `anthropic`: Claude model with Anthropic tool schema validation.

Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` in `.env` to enable provider-backed generation.

## Finance Workflow Templates

- `deal_screening`
- `credit_memo`
- `due_diligence`
- `earnings_quality`
- `covenant_risk`
- `management_discussion`

Each template defines the memo objective, required sections, retrieval prompts, risk checks, output fields, and evaluation rubric.

## Notes

This is not investment advice. The system is intended for technical demonstration, research workflow prototyping, and portfolio evidence. All outputs should be reviewed by a qualified human analyst before use in any investment decision.
