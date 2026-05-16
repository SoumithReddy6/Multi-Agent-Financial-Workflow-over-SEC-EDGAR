from sec_memo_agents.core.vector_store import VectorStore


def test_vector_store_returns_relevant_comparable():
    store = VectorStore(dimensions=64)
    store.add_texts(
        [
            "Cloud revenue grew because Azure demand and enterprise software renewals improved.",
            "Retail traffic declined due to weaker store demand and inventory pressure.",
        ],
        [
            {"company_name": "Microsoft", "ticker": "MSFT", "source_url": "https://example.com/msft"},
            {"company_name": "RetailCo", "ticker": "RTL", "source_url": "https://example.com/rtl"},
        ],
    )

    results = store.search("cloud software Azure revenue", top_k=1)

    assert results[0].company_name == "Microsoft"
    assert results[0].source_url == "https://example.com/msft"
