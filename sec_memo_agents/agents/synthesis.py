"""Synthesis agent that emits schema-valid investment memos."""

from __future__ import annotations

import json
import re
from typing import Any

from sec_memo_agents.core.text import summarize_snippet
from sec_memo_agents.schemas import (
    FinancialMetric,
    InvestmentMemo,
    MemoRisk,
    Recommendation,
    RetrievedEvidence,
    WorkflowTemplate,
    anthropic_tool_schema,
    openai_tool_schema,
)
from sec_memo_agents.settings import Settings, get_settings


FACT_TAGS = {
    "Revenue": ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"],
    "Operating income": ["OperatingIncomeLoss"],
    "Net income": ["NetIncomeLoss"],
    "Assets": ["Assets"],
    "Liabilities": ["Liabilities"],
    "Cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "Long-term debt": ["LongTermDebt", "LongTermDebtAndFinanceLeaseObligationsCurrentAndNoncurrent"],
}


class SynthesisAgent:
    """Generates investment memos through LLM tool calls or deterministic fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @staticmethod
    def openai_schema() -> dict[str, Any]:
        return openai_tool_schema(
            InvestmentMemo,
            "emit_investment_memo",
            "Emit a schema-valid investment memo from SEC filing evidence.",
        )

    @staticmethod
    def anthropic_schema() -> dict[str, Any]:
        return anthropic_tool_schema(
            InvestmentMemo,
            "emit_investment_memo",
            "Emit a schema-valid investment memo from SEC filing evidence.",
        )

    def generate(
        self,
        filing,
        comparable_evidence: list[RetrievedEvidence],
        template: WorkflowTemplate,
        analyst_question: str | None = None,
        provider: str | None = None,
    ) -> InvestmentMemo:
        selected_provider = (provider or self.settings.llm_provider).lower()
        if selected_provider == "openai" and self.settings.openai_api_key:
            return self._generate_openai(filing, comparable_evidence, template, analyst_question)
        if selected_provider == "anthropic" and self.settings.anthropic_api_key:
            return self._generate_anthropic(filing, comparable_evidence, template, analyst_question)
        return self._generate_offline(filing, comparable_evidence, template, analyst_question)

    def _prompt_payload(
        self,
        filing,
        comparable_evidence: list[RetrievedEvidence],
        template: WorkflowTemplate,
        analyst_question: str | None,
    ) -> dict[str, Any]:
        return {
            "template": template.model_dump(),
            "filing_metadata": filing.metadata.model_dump(),
            "sections": {key: summarize_snippet(value, 1800) for key, value in filing.sections.items()},
            "xbrl_summary": [metric.model_dump() for metric in self._extract_metrics(filing.xbrl_facts)],
            "comparable_evidence": [item.model_dump() for item in comparable_evidence],
            "analyst_question": analyst_question,
            "instruction": "Return only the investment memo tool call with balanced finance analysis and source citations.",
        }

    def _generate_openai(
        self,
        filing,
        comparable_evidence: list[RetrievedEvidence],
        template: WorkflowTemplate,
        analyst_question: str | None,
    ) -> InvestmentMemo:
        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a careful buy-side investment analyst."},
                {"role": "user", "content": json.dumps(self._prompt_payload(filing, comparable_evidence, template, analyst_question))},
            ],
            tools=[self.openai_schema()],
            tool_choice={"type": "function", "function": {"name": "emit_investment_memo"}},
        )
        tool_call = response.choices[0].message.tool_calls[0]
        return InvestmentMemo.model_validate_json(tool_call.function.arguments)

    def _generate_anthropic(
        self,
        filing,
        comparable_evidence: list[RetrievedEvidence],
        template: WorkflowTemplate,
        analyst_question: str | None,
    ) -> InvestmentMemo:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.settings.anthropic_api_key)
        response = client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=3000,
            temperature=0,
            system="You are a careful buy-side investment analyst.",
            tools=[self.anthropic_schema()],
            tool_choice={"type": "tool", "name": "emit_investment_memo"},
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(self._prompt_payload(filing, comparable_evidence, template, analyst_question)),
                }
            ],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return InvestmentMemo.model_validate(block.input)
        raise RuntimeError("Anthropic response did not include the expected tool call")

    def _generate_offline(
        self,
        filing,
        comparable_evidence: list[RetrievedEvidence],
        template: WorkflowTemplate,
        analyst_question: str | None,
    ) -> InvestmentMemo:
        metadata = filing.metadata
        metrics = self._extract_metrics(filing.xbrl_facts)
        if not metrics:
            metrics = [
                FinancialMetric(
                    name="Filing text coverage",
                    value=f"{len(filing.text):,} characters",
                    period=metadata.filing_date,
                    interpretation="The parser captured enough filing text for qualitative analysis.",
                    source=metadata.document_url,
                )
            ]

        risk_text = filing.sections.get("risk_factors") or filing.text
        risks = self._extract_risks(risk_text, comparable_evidence)
        overview_source = filing.sections.get("business") or filing.sections.get("mda") or filing.text
        comparable_insights = [
            f"{item.company_name}: {item.snippet}" for item in comparable_evidence[:5]
        ] or ["Comparable index is empty; rerun ingestion with a broader ticker universe for peer evidence."]

        citations = [metadata.document_url]
        citations.extend(item.source_url for item in comparable_evidence if item.source_url)
        unique_citations = list(dict.fromkeys(citations))

        recommendation = Recommendation.watchlist
        confidence = 0.58 + min(len(metrics), 4) * 0.05 + min(len(comparable_evidence), 5) * 0.02
        confidence = min(confidence, 0.86)

        summary = (
            f"{metadata.company_name} was analyzed from its {metadata.form_type} filed on {metadata.filing_date}. "
            f"The memo uses the {template.title} workflow, combines filing evidence with peer snippets, "
            f"and keeps the name on a watchlist pending deeper human review."
        )
        if analyst_question:
            summary += f" Analyst focus: {analyst_question}"

        memo = InvestmentMemo(
            company_name=metadata.company_name,
            ticker=metadata.ticker,
            cik=metadata.cik,
            form_type=metadata.form_type,
            filing_date=metadata.filing_date,
            workflow_template=template.name,
            recommendation=recommendation,
            confidence=round(confidence, 2),
            executive_summary=summary,
            business_overview=summarize_snippet(overview_source, 950),
            key_financials=metrics[:7],
            comparable_insights=comparable_insights,
            risks=risks,
            catalysts=[
                "Upcoming earnings calls and guidance updates could clarify demand durability.",
                "Margin trajectory and cash conversion are the highest-signal follow-up indicators.",
                "Peer multiple dispersion should be reviewed before moving from screening to valuation.",
            ],
            diligence_questions=[
                "Which revenue lines are most exposed to cyclical or customer concentration risk?",
                "Are reported margins supported by sustainable operating drivers or one-time items?",
                "How do balance-sheet risks compare with the closest peer group?",
                "What evidence would change the recommendation from watchlist to buy, hold, or avoid?",
            ],
            source_citations=unique_citations,
        )
        return InvestmentMemo.model_validate(memo.model_dump())

    def _extract_metrics(self, facts: dict[str, Any]) -> list[FinancialMetric]:
        us_gaap = (facts or {}).get("facts", {}).get("us-gaap", {})
        metrics: list[FinancialMetric] = []
        for label, tags in FACT_TAGS.items():
            fact = None
            for tag in tags:
                if tag in us_gaap:
                    fact = us_gaap[tag]
                    break
            if not fact:
                continue
            unit_payload = fact.get("units", {})
            unit_values = unit_payload.get("USD") or next(iter(unit_payload.values()), [])
            latest = self._latest_fact(unit_values)
            if not latest:
                continue
            value = latest.get("val")
            period = latest.get("fy") or latest.get("end") or latest.get("filed")
            metrics.append(
                FinancialMetric(
                    name=label,
                    value=self._format_number(value),
                    period=str(period) if period else None,
                    interpretation=f"{label} is a core filing metric for screening operating scale and financial risk.",
                    source=latest.get("accn"),
                )
            )
        return metrics

    @staticmethod
    def _latest_fact(values: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not values:
            return None
        filtered = [item for item in values if item.get("form") in {"10-K", "10-Q"}] or values
        return sorted(filtered, key=lambda item: (str(item.get("filed", "")), str(item.get("end", ""))), reverse=True)[0]

    @staticmethod
    def _format_number(value: Any) -> str:
        if isinstance(value, (int, float)):
            magnitude = abs(float(value))
            if magnitude >= 1_000_000_000:
                return f"${value / 1_000_000_000:,.2f}B"
            if magnitude >= 1_000_000:
                return f"${value / 1_000_000:,.2f}M"
            return f"${value:,.0f}"
        return str(value)

    def _extract_risks(self, risk_text: str, comparable_evidence: list[RetrievedEvidence]) -> list[MemoRisk]:
        sentences = re.split(r"(?<=[.!?])\s+", risk_text)
        risk_sentences = [
            sentence.strip()
            for sentence in sentences
            if any(keyword in sentence.lower() for keyword in ["risk", "competition", "liquidity", "debt", "regulation", "supply"])
        ]
        selected = risk_sentences[:3]
        while len(selected) < 3 and comparable_evidence:
            selected.append(comparable_evidence[min(len(selected), len(comparable_evidence) - 1)].snippet)
        if not selected:
            selected = ["Filing text did not expose a clean risk section; analyst review is required before investment use."]

        categories = ["Business model", "Financial profile", "Market and regulatory"]
        risks: list[MemoRisk] = []
        for index, sentence in enumerate(selected[:3]):
            risks.append(
                MemoRisk(
                    category=categories[index % len(categories)],
                    severity="medium" if index < 2 else "low",
                    description=summarize_snippet(sentence, 280),
                    evidence=summarize_snippet(sentence, 220),
                )
            )
        return risks
