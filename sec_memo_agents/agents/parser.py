"""Parser agent for SEC 10-K and 10-Q filings."""

from __future__ import annotations

from collections.abc import Iterable

from sec_memo_agents.core.sec_client import SECClient
from sec_memo_agents.schemas import FilingDocument


class ParserAgent:
    """Ingests filing metadata, filing text, and company facts from SEC EDGAR."""

    def __init__(self, sec_client: SECClient | None = None) -> None:
        self.sec_client = sec_client or SECClient()

    def ingest_company(
        self,
        ticker_or_cik: str,
        forms: Iterable[str] = ("10-K", "10-Q"),
        limit: int = 3,
        include_facts: bool = True,
    ) -> list[FilingDocument]:
        filings = self.sec_client.list_filings(ticker_or_cik, forms=forms, limit=limit)
        return [self.sec_client.fetch_filing_document(metadata, include_facts=include_facts) for metadata in filings]

    def ingest_universe(
        self,
        tickers: Iterable[str],
        forms: Iterable[str] = ("10-K", "10-Q"),
        limit_per_company: int = 5,
        target_filings: int | None = None,
    ) -> list[FilingDocument]:
        documents: list[FilingDocument] = []
        for ticker in tickers:
            documents.extend(self.ingest_company(ticker, forms=forms, limit=limit_per_company))
            if target_filings and len(documents) >= target_filings:
                return documents[:target_filings]
        return documents
