"""FastAPI surface for SEC memo agents."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from sec_memo_agents import __version__
from sec_memo_agents.agents.parser import ParserAgent
from sec_memo_agents.agents.retrieval import RetrievalAgent
from sec_memo_agents.agents.workflow import FinancialWorkflow
from sec_memo_agents.core.templates import list_templates
from sec_memo_agents.schemas import (
    IngestRequest,
    InvestmentMemo,
    MemoRequest,
    anthropic_tool_schema,
    openai_tool_schema,
)


app = FastAPI(
    title="SEC EDGAR Multi-Agent Financial Workflow",
    version=__version__,
    description="Parser, retrieval, and synthesis agents for SEC filing investment memos.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/templates")
def templates() -> list[dict]:
    return [template.model_dump() for template in list_templates()]


@app.get("/schemas/openai/investment_memo")
def openai_schema() -> dict:
    return openai_tool_schema(InvestmentMemo, "emit_investment_memo", "Emit a schema-valid investment memo.")


@app.get("/schemas/anthropic/investment_memo")
def anthropic_schema() -> dict:
    return anthropic_tool_schema(InvestmentMemo, "emit_investment_memo", "Emit a schema-valid investment memo.")


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    try:
        parser = ParserAgent()
        filings = parser.ingest_universe(
            request.tickers,
            forms=request.forms,
            limit_per_company=request.filing_limit,
            target_filings=request.target_filings,
        )
        retrieval = RetrievalAgent()
        chunks = retrieval.index_filings(filings)
        return {
            "filings_processed": len(filings),
            "chunks_indexed": chunks,
            "vector_backend": retrieval.vector_store.backend,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/memos", response_model=InvestmentMemo)
def create_memo(request: MemoRequest) -> InvestmentMemo:
    try:
        return FinancialWorkflow().run(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
