"""Vector store with a pluggable embedder and optional FAISS search.

The store is embedder-agnostic. By default it uses the dependency-free
``HashEmbedder`` so it can run anywhere; the ``RetrievalAgent`` wires in a real
``SentenceTransformerEmbedder`` for production semantic retrieval. FAISS
(``IndexFlatIP``) is used for similarity search when ``faiss-cpu`` is installed,
with a pure-Python cosine fallback otherwise. Because all embeddings are
L2-normalized, inner product equals cosine similarity in both paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sec_memo_agents.core.embeddings import Embedder, HashEmbedder, build_embedder
from sec_memo_agents.core.text import summarize_snippet
from sec_memo_agents.schemas import RetrievedEvidence


@dataclass
class VectorRecord:
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


class VectorStore:
    """Vector store over an injected embedder.

    Pass an ``embedder`` to control how text is vectorized. When omitted, a
    deterministic ``HashEmbedder`` of ``dimensions`` is used so the store has no
    heavy dependencies by default.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        dimensions: int = 384,
    ) -> None:
        self.embedder: Embedder = embedder or HashEmbedder(dimensions)
        self.dimensions = self.embedder.dimensions
        self.records: list[VectorRecord] = []

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must be the same length")
        if not texts:
            return
        embeddings = self.embedder.embed(texts)
        for text, metadata, embedding in zip(texts, metadatas, embeddings):
            self.records.append(VectorRecord(text=text, metadata=metadata, embedding=embedding))

    def search(self, query: str, top_k: int = 5, exclude_ticker: str | None = None) -> list[RetrievedEvidence]:
        query_vector = self.embedder.embed([query])[0]
        eligible: list[VectorRecord] = []
        for record in self.records:
            ticker = (record.metadata.get("ticker") or "").upper()
            if exclude_ticker and ticker == exclude_ticker.upper():
                continue
            eligible.append(record)

        scored = self._score_with_faiss(query_vector, eligible, top_k)
        if scored is None:
            scored = self._score_with_python(query_vector, eligible, top_k)

        evidence: list[RetrievedEvidence] = []
        for score, record in scored:
            metadata = record.metadata
            evidence.append(
                RetrievedEvidence(
                    company_name=metadata.get("company_name", "Unknown company"),
                    ticker=metadata.get("ticker"),
                    cik=metadata.get("cik"),
                    form_type=metadata.get("form_type"),
                    filing_date=metadata.get("filing_date"),
                    section=metadata.get("section"),
                    score=round(float(score), 4),
                    snippet=summarize_snippet(record.text),
                    source_url=metadata.get("source_url"),
                )
            )
        return evidence

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def _score_with_python(
        self,
        query_vector: list[float],
        records: list[VectorRecord],
        top_k: int,
    ) -> list[tuple[float, VectorRecord]]:
        scored = [(self._cosine(query_vector, record.embedding), record) for record in records]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def _score_with_faiss(
        self,
        query_vector: list[float],
        records: list[VectorRecord],
        top_k: int,
    ) -> list[tuple[float, VectorRecord]] | None:
        try:
            import faiss
            import numpy as np
        except Exception:
            return None
        if not records:
            return []

        matrix = np.array([record.embedding for record in records], dtype="float32")
        query = np.array([query_vector], dtype="float32")
        index = faiss.IndexFlatIP(self.dimensions)
        index.add(matrix)
        scores, indices = index.search(query, min(top_k, len(records)))
        return [
            (float(score), records[int(index_value)])
            for score, index_value in zip(scores[0], indices[0])
            if index_value >= 0
        ]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimensions": self.dimensions,
            "embedder": self.embedder.name,
            "semantic": getattr(self.embedder, "semantic", False),
            "records": [
                {"text": record.text, "metadata": record.metadata, "embedding": record.embedding}
                for record in self.records
            ],
        }
        target.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> "VectorStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        dimensions = payload["dimensions"]
        if embedder is None:
            # Rebuild a matching embedder so query embedding stays consistent
            # with the stored record vectors.
            saved_name = payload.get("embedder", "")
            if payload.get("semantic") and saved_name:
                embedder = build_embedder(backend="auto", model_name=saved_name, hash_dimensions=dimensions)
            else:
                embedder = HashEmbedder(dimensions)
        store = cls(embedder=embedder)
        store.records = [
            VectorRecord(
                text=item["text"],
                metadata=item["metadata"],
                embedding=[float(value) for value in item["embedding"]],
            )
            for item in payload["records"]
        ]
        return store

    @property
    def backend(self) -> str:
        try:
            import faiss  # noqa: F401

            search = "faiss"
        except Exception:
            search = "python-cosine"
        kind = "semantic" if getattr(self.embedder, "semantic", False) else "lexical"
        return f"{self.embedder.name} ({kind}) + {search}"
