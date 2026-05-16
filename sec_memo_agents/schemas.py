"""Pydantic schemas used for strict JSON memo outputs."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Recommendation(str, Enum):
    buy = "buy"
    hold = "hold"
    avoid = "avoid"
    watchlist = "watchlist"
    insufficient_data = "insufficient_data"


class FilingMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cik: str = Field(..., description="10-digit SEC CIK with leading zeros.")
    ticker: str | None = None
    company_name: str
    form_type: Literal["10-K", "10-Q"]
    accession_number: str
    filing_date: str
    report_date: str | None = None
    primary_document: str
    filing_url: str
    document_url: str

    @field_validator("cik")
    @classmethod
    def cik_is_numeric(cls, value: str) -> str:
        digits = value.strip().zfill(10)
        if not digits.isdigit() or len(digits) != 10:
            raise ValueError("CIK must be numeric and at most 10 digits")
        return digits


class FilingDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: FilingMetadata
    text: str = Field(..., min_length=20)
    xbrl_facts: dict[str, Any] = Field(default_factory=dict)
    sections: dict[str, str] = Field(default_factory=dict)


class RetrievedEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company_name: str
    ticker: str | None = None
    cik: str | None = None
    form_type: str | None = None
    filing_date: str | None = None
    section: str | None = None
    score: float
    snippet: str
    source_url: str | None = None


class FinancialMetric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: str
    period: str | None = None
    interpretation: str
    source: str | None = None


class MemoRisk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    severity: Literal["low", "medium", "high"]
    description: str
    evidence: str


class InvestmentMemo(BaseModel):
    """Validated output contract for the synthesis agent."""

    model_config = ConfigDict(extra="forbid")

    memo_id: str = Field(default_factory=lambda: str(uuid4()))
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    company_name: str
    ticker: str | None = None
    cik: str
    form_type: str
    filing_date: str
    workflow_template: str
    recommendation: Recommendation
    confidence: float = Field(..., ge=0.0, le=1.0)
    executive_summary: str
    business_overview: str
    key_financials: list[FinancialMetric]
    comparable_insights: list[str]
    risks: list[MemoRisk]
    catalysts: list[str]
    diligence_questions: list[str]
    source_citations: list[str]

    @field_validator("source_citations")
    @classmethod
    def citations_required(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("At least one source citation is required")
        return value


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickers: list[str] = Field(..., min_length=1)
    forms: list[Literal["10-K", "10-Q"]] = Field(default_factory=lambda: ["10-K", "10-Q"])
    filing_limit: int = Field(3, ge=1, le=20)
    target_filings: int | None = Field(None, ge=1)


class MemoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    template_name: str = "deal_screening"
    forms: list[Literal["10-K", "10-Q"]] = Field(default_factory=lambda: ["10-K", "10-Q"])
    filing_limit: int = Field(2, ge=1, le=10)
    top_k_comparables: int = Field(5, ge=1, le=12)
    offline: bool = False
    analyst_question: str | None = None


class WorkflowTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    title: str
    objective: str
    target_user: str
    retrieval_queries: list[str]
    required_sections: list[str]
    risk_checks: list[str]
    output_fields: list[str]
    evaluation_rubric: list[str]


def openai_tool_schema(model: type[BaseModel], name: str, description: str) -> dict[str, Any]:
    """Return an OpenAI-compatible tool definition for a Pydantic model."""

    schema = model.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
            "strict": True,
        },
    }


def anthropic_tool_schema(model: type[BaseModel], name: str, description: str) -> dict[str, Any]:
    """Return an Anthropic-compatible tool definition for a Pydantic model."""

    return {
        "name": name,
        "description": description,
        "input_schema": model.model_json_schema(),
    }
