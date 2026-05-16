"""SEC EDGAR client with fair-access headers, rate limiting, and local caching."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import requests

from sec_memo_agents.core.text import clean_filing_text, extract_sections
from sec_memo_agents.schemas import FilingDocument, FilingMetadata
from sec_memo_agents.settings import Settings, get_settings


DATA_SEC_BASE = "https://data.sec.gov"
WWW_SEC_BASE = "https://www.sec.gov"


class RateLimiter:
    """Small sliding-window limiter for SEC fair-access requests."""

    def __init__(self, max_per_second: float) -> None:
        self.max_per_second = max(1.0, min(max_per_second, 10.0))
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        now = time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 1.0:
            self._timestamps.popleft()
        if len(self._timestamps) >= self.max_per_second:
            sleep_for = 1.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


class SECClient:
    """Client for public SEC EDGAR APIs and filing archives."""

    def __init__(self, settings: Settings | None = None, session: requests.Session | None = None) -> None:
        self.settings = settings or get_settings()
        self.session = session or requests.Session()
        self.rate_limiter = RateLimiter(self.settings.sec_max_requests_per_second)
        self.cache_dir = self.settings.sec_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_cik(cik: str | int) -> str:
        digits = "".join(ch for ch in str(cik) if ch.isdigit())
        if not digits:
            raise ValueError("CIK must contain digits")
        return digits.zfill(10)

    @staticmethod
    def archive_url(cik: str, accession_number: str, primary_document: str) -> str:
        cik_int = str(int(SECClient.normalize_cik(cik)))
        accession_no_dash = accession_number.replace("-", "")
        return f"{WWW_SEC_BASE}/Archives/edgar/data/{cik_int}/{accession_no_dash}/{primary_document}"

    @staticmethod
    def filing_index_url(cik: str, accession_number: str) -> str:
        cik_int = str(int(SECClient.normalize_cik(cik)))
        accession_no_dash = accession_number.replace("-", "")
        return f"{WWW_SEC_BASE}/Archives/edgar/data/{cik_int}/{accession_no_dash}/{accession_number}-index.html"

    def _cache_path(self, url: str, suffix: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.{suffix}"

    def _get_json(self, url: str, use_cache: bool = True) -> dict[str, Any]:
        cache_path = self._cache_path(url, "json")
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        self.rate_limiter.wait()
        headers = dict(self.settings.sec_headers)
        if "data.sec.gov" in url:
            headers["Host"] = "data.sec.gov"
        response = self.session.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _get_text(self, url: str, use_cache: bool = True) -> str:
        cache_path = self._cache_path(url, "txt")
        if use_cache and cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="ignore")

        self.rate_limiter.wait()
        response = self.session.get(url, headers=self.settings.sec_headers, timeout=45)
        response.raise_for_status()
        text = response.text
        cache_path.write_text(text, encoding="utf-8", errors="ignore")
        return text

    def ticker_map(self) -> dict[str, dict[str, Any]]:
        url = f"{WWW_SEC_BASE}/files/company_tickers.json"
        raw = self._get_json(url)
        return {entry["ticker"].upper(): entry for entry in raw.values()}

    def resolve_ticker(self, ticker_or_cik: str) -> tuple[str, str | None, str | None]:
        token = ticker_or_cik.strip().upper()
        if token.isdigit():
            return self.normalize_cik(token), None, None
        mapping = self.ticker_map()
        if token not in mapping:
            raise KeyError(f"Ticker {token} was not found in SEC ticker map")
        entry = mapping[token]
        return self.normalize_cik(entry["cik_str"]), entry["ticker"].upper(), entry["title"]

    def get_submissions(self, ticker_or_cik: str) -> dict[str, Any]:
        cik, _, _ = self.resolve_ticker(ticker_or_cik)
        url = f"{DATA_SEC_BASE}/submissions/CIK{cik}.json"
        return self._get_json(url)

    def list_filings(
        self,
        ticker_or_cik: str,
        forms: Iterable[str] = ("10-K", "10-Q"),
        limit: int = 5,
    ) -> list[FilingMetadata]:
        submissions = self.get_submissions(ticker_or_cik)
        cik = self.normalize_cik(submissions["cik"])
        ticker = (submissions.get("tickers") or [None])[0]
        company_name = submissions.get("name") or submissions.get("entityName") or "Unknown company"
        recent = submissions["filings"]["recent"]
        allowed = set(forms)
        filings: list[FilingMetadata] = []

        for index, form_type in enumerate(recent.get("form", [])):
            if form_type not in allowed:
                continue
            accession_number = recent["accessionNumber"][index]
            primary_document = recent["primaryDocument"][index]
            report_dates = recent.get("reportDate") or []
            report_date = report_dates[index] if index < len(report_dates) else None
            document_url = self.archive_url(cik, accession_number, primary_document)
            filings.append(
                FilingMetadata(
                    cik=cik,
                    ticker=ticker,
                    company_name=company_name,
                    form_type=form_type,
                    accession_number=accession_number,
                    filing_date=recent["filingDate"][index],
                    report_date=report_date,
                    primary_document=primary_document,
                    filing_url=self.filing_index_url(cik, accession_number),
                    document_url=document_url,
                )
            )
            if len(filings) >= limit:
                break
        return filings

    def get_company_facts(self, ticker_or_cik: str) -> dict[str, Any]:
        cik, _, _ = self.resolve_ticker(ticker_or_cik)
        url = f"{DATA_SEC_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
        return self._get_json(url)

    def fetch_filing_document(self, metadata: FilingMetadata, include_facts: bool = True) -> FilingDocument:
        raw = self._get_text(metadata.document_url)
        text = clean_filing_text(raw)
        facts: dict[str, Any] = {}
        if include_facts:
            try:
                facts = self.get_company_facts(metadata.cik)
            except Exception:
                facts = {}
        return FilingDocument(
            metadata=metadata,
            text=text,
            xbrl_facts=facts,
            sections=extract_sections(text),
        )
