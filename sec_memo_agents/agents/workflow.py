"""Workflow orchestration for SEC filing memo generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from sec_memo_agents.agents.parser import ParserAgent
from sec_memo_agents.agents.retrieval import RetrievalAgent
from sec_memo_agents.agents.synthesis import SynthesisAgent
from sec_memo_agents.core.templates import load_template
from sec_memo_agents.schemas import FilingDocument, InvestmentMemo, MemoRequest, RetrievedEvidence


class WorkflowState(TypedDict, total=False):
    request: MemoRequest
    filings: list[FilingDocument]
    target_filing: FilingDocument
    evidence: list[RetrievedEvidence]
    memo: InvestmentMemo


class FinancialWorkflow:
    """Coordinates parser, retrieval, and synthesis agents."""

    def __init__(
        self,
        parser: ParserAgent | None = None,
        retrieval: RetrievalAgent | None = None,
        synthesis: SynthesisAgent | None = None,
    ) -> None:
        self.parser = parser or ParserAgent()
        self.retrieval = retrieval or RetrievalAgent()
        self.synthesis = synthesis or SynthesisAgent()

    def run(self, request: MemoRequest) -> InvestmentMemo:
        template = load_template(request.template_name)
        filings = self._load_offline_filings(request.ticker) if request.offline else self.parser.ingest_company(
            request.ticker,
            forms=request.forms,
            limit=request.filing_limit,
            include_facts=True,
        )
        if not filings:
            raise RuntimeError(f"No filings found for {request.ticker}")
        target_filing = filings[0]
        self.retrieval.index_filings(filings)
        evidence = self.retrieval.retrieve_comparables(target_filing, template, top_k=request.top_k_comparables)
        return self.synthesis.generate(
            target_filing,
            evidence,
            template,
            analyst_question=request.analyst_question,
            provider="offline" if request.offline else None,
        )

    def run_with_filings(
        self,
        filings: list[FilingDocument],
        template_name: str = "deal_screening",
        top_k: int = 5,
        analyst_question: str | None = None,
    ) -> InvestmentMemo:
        if not filings:
            raise ValueError("At least one filing is required")
        template = load_template(template_name)
        self.retrieval.index_filings(filings)
        target_filing = filings[0]
        evidence = self.retrieval.retrieve_comparables(target_filing, template, top_k=top_k)
        return self.synthesis.generate(target_filing, evidence, template, analyst_question=analyst_question, provider="offline")

    def build_langgraph(self):
        """Build a LangGraph DAG when langgraph is installed."""

        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        graph = StateGraph(WorkflowState)

        def parse_node(state: WorkflowState) -> WorkflowState:
            request = state["request"]
            state["filings"] = self.parser.ingest_company(request.ticker, forms=request.forms, limit=request.filing_limit)
            state["target_filing"] = state["filings"][0]
            return state

        def retrieve_node(state: WorkflowState) -> WorkflowState:
            request = state["request"]
            template = load_template(request.template_name)
            self.retrieval.index_filings(state["filings"])
            state["evidence"] = self.retrieval.retrieve_comparables(state["target_filing"], template, top_k=request.top_k_comparables)
            return state

        def synthesize_node(state: WorkflowState) -> WorkflowState:
            request = state["request"]
            template = load_template(request.template_name)
            state["memo"] = self.synthesis.generate(
                state["target_filing"],
                state["evidence"],
                template,
                analyst_question=request.analyst_question,
            )
            return state

        graph.add_node("parser_agent", parse_node)
        graph.add_node("retrieval_agent", retrieve_node)
        graph.add_node("synthesis_agent", synthesize_node)
        graph.set_entry_point("parser_agent")
        graph.add_edge("parser_agent", "retrieval_agent")
        graph.add_edge("retrieval_agent", "synthesis_agent")
        graph.add_edge("synthesis_agent", END)
        return graph.compile()

    @staticmethod
    def _load_offline_filings(ticker: str) -> list[FilingDocument]:
        fixture_path = Path(__file__).resolve().parents[1] / "data" / "offline_filings.json"
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        records = payload.get(ticker.upper()) or payload.get("AAPL")
        return [FilingDocument.model_validate(item) for item in records]
