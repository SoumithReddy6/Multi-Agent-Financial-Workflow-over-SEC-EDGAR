import pytest
from pydantic import ValidationError

from sec_memo_agents.schemas import InvestmentMemo, anthropic_tool_schema, openai_tool_schema


def valid_memo_payload():
    return {
        "company_name": "Apple Inc.",
        "ticker": "AAPL",
        "cik": "0000320193",
        "form_type": "10-K",
        "filing_date": "2025-10-31",
        "workflow_template": "deal_screening",
        "recommendation": "watchlist",
        "confidence": 0.75,
        "executive_summary": "Schema-valid memo.",
        "business_overview": "Business overview.",
        "key_financials": [
            {
                "name": "Revenue",
                "value": "$391.04B",
                "period": "2025",
                "interpretation": "Large operating scale.",
                "source": "filing",
            }
        ],
        "comparable_insights": ["Peer evidence."],
        "risks": [
            {
                "category": "Competition",
                "severity": "medium",
                "description": "Competitive intensity remains meaningful.",
                "evidence": "Risk factor section.",
            }
        ],
        "catalysts": ["Product cycle."],
        "diligence_questions": ["How durable are margins?"],
        "source_citations": ["https://www.sec.gov/example"],
    }


def test_investment_memo_schema_validates_payload():
    memo = InvestmentMemo.model_validate(valid_memo_payload())

    assert memo.recommendation == "watchlist"
    assert memo.cik == "0000320193"


def test_investment_memo_rejects_missing_citations():
    payload = valid_memo_payload()
    payload["source_citations"] = []

    with pytest.raises(ValidationError):
        InvestmentMemo.model_validate(payload)


def test_tool_schemas_export_function_calling_contracts():
    openai_schema = openai_tool_schema(InvestmentMemo, "emit_investment_memo", "Emit memo.")
    anthropic_schema = anthropic_tool_schema(InvestmentMemo, "emit_investment_memo", "Emit memo.")

    assert openai_schema["type"] == "function"
    assert openai_schema["function"]["strict"] is True
    assert anthropic_schema["input_schema"]["title"] == "InvestmentMemo"
