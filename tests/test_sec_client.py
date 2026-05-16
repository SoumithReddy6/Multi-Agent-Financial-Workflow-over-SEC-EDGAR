from sec_memo_agents.core.sec_client import SECClient


def test_sec_archive_url_uses_cik_and_accession_without_dashes():
    url = SECClient.archive_url("320193", "0000320193-25-000079", "aapl-20250927.htm")

    assert url == "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"


def test_normalize_cik_pads_to_ten_digits():
    assert SECClient.normalize_cik("320193") == "0000320193"
