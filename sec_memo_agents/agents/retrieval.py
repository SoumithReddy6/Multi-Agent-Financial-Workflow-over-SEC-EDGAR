"""Retrieval agent for comparable-company evidence."""

from __future__ import annotations

from sec_memo_agents.core.text import chunk_text
from sec_memo_agents.core.vector_store import VectorStore
from sec_memo_agents.schemas import FilingDocument, RetrievedEvidence, WorkflowTemplate
from sec_memo_agents.settings import Settings, get_settings


class RetrievalAgent:
    """Indexes filing chunks and retrieves relevant comparable evidence."""

    def __init__(self, vector_store: VectorStore | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or VectorStore(dimensions=self.settings.embedding_dimensions)

    def index_filings(self, filings: list[FilingDocument]) -> int:
        texts: list[str] = []
        metadatas: list[dict[str, str | None]] = []
        for filing in filings:
            metadata = filing.metadata
            section_items = filing.sections.items() or [("full_filing", filing.text)]
            for section, section_text in section_items:
                for chunk_index, chunk in enumerate(
                    chunk_text(section_text, chunk_size=self.settings.chunk_size, overlap=self.settings.chunk_overlap)
                ):
                    texts.append(chunk)
                    metadatas.append(
                        {
                            "company_name": metadata.company_name,
                            "ticker": metadata.ticker,
                            "cik": metadata.cik,
                            "form_type": metadata.form_type,
                            "filing_date": metadata.filing_date,
                            "section": section,
                            "chunk_index": str(chunk_index),
                            "source_url": metadata.document_url,
                        }
                    )
        if texts:
            self.vector_store.add_texts(texts, metadatas)
        return len(texts)

    def to_langchain_documents(self, filings: list[FilingDocument]):
        """Convert filings to LangChain Documents for teams that prefer LCEL/RAG chains."""

        try:
            from langchain_core.documents import Document
        except Exception as exc:
            raise RuntimeError("Install langchain to use LangChain document adapters") from exc

        documents = []
        for filing in filings:
            metadata = filing.metadata
            section_items = filing.sections.items() or [("full_filing", filing.text)]
            for section, section_text in section_items:
                for chunk_index, chunk in enumerate(
                    chunk_text(section_text, chunk_size=self.settings.chunk_size, overlap=self.settings.chunk_overlap)
                ):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "company_name": metadata.company_name,
                                "ticker": metadata.ticker,
                                "cik": metadata.cik,
                                "form_type": metadata.form_type,
                                "filing_date": metadata.filing_date,
                                "section": section,
                                "chunk_index": chunk_index,
                                "source_url": metadata.document_url,
                            },
                        )
                    )
        return documents

    def retrieve_comparables(
        self,
        filing: FilingDocument,
        template: WorkflowTemplate,
        top_k: int = 5,
    ) -> list[RetrievedEvidence]:
        query_parts = [
            filing.metadata.company_name,
            filing.metadata.form_type,
            template.objective,
            " ".join(template.retrieval_queries),
        ]
        if filing.sections:
            query_parts.extend(list(filing.sections.keys()))
        query = " ".join(query_parts)
        return self.vector_store.search(query, top_k=top_k, exclude_ticker=filing.metadata.ticker)
