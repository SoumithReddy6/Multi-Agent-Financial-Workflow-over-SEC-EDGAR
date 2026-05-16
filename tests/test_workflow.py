from sec_memo_agents.agents.workflow import FinancialWorkflow
from sec_memo_agents.schemas import InvestmentMemo, MemoRequest


def test_offline_workflow_generates_schema_valid_memo():
    request = MemoRequest(ticker="AAPL", template_name="deal_screening", offline=True)

    memo = FinancialWorkflow().run(request)

    assert isinstance(memo, InvestmentMemo)
    assert memo.company_name == "Apple Inc."
    assert memo.workflow_template == "deal_screening"
    assert memo.source_citations
    assert memo.key_financials
