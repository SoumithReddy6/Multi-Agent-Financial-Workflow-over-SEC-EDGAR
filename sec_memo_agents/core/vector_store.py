"""FAISS-compatible retrieval layer with a pure-Python fallback."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sec_memo_agents.core.text import summarize_snippet
from sec_memo_agents.schemas import RetrievedEvidence


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]+")


def _hash_embedding(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    tokens = TOKEN_RE.findall(text.lower())
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        signed_digest = hashlib.blake2b(("salt:" + token).encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "big") % dimensions
        sign = -1.0 if int.from_bytes(signed_digest, "big") % 2 else 1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


@dataclass
class VectorRecord:
    text: str
    metadata: dict[str, Any]
    embedding: list[float]


@dataclass
class VectorStore:
    """Simple vector store that uses deterministic embeddings and optional FAISS search."""

    dimensions: int = 384
    records: list[VectorRecord] = field(default_factory=list)

    def add_texts(self, texts: list[str], metadatas: list[dict[str, Any]]) -> None:
        if len(texts) != len(metadatas):
            raise ValueError("texts and metadatas must be the same length")
        for text, metadata in zip(texts, metadatas):
            self.records.append(VectorRecord(text=text, metadata=metadata, embedding=_hash_embedding(text, self.dimensions)))

    def search(self, query: str, top_k: int = 5, exclude_ticker: str | None = None) -> list[RetrievedEvidence]:
        query_vector = _hash_embedding(query, self.dimensions)
        eligible = []
        for record in self.records:
            ticker = (record.metadata.get("ticker") or "").upper()
            if exclude_ticker and ticker == exclude_ticker.upper():
                continue
            eligible.append(record)

        scored = self._score_with_faiss(query_vector, eligible, top_k) or self._score_with_python(query_vector, eligible, top_k)

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

    def _score_with_python(
        self,
        query_vector: list[float],
        records: list[VectorRecord],
        top_k: int,
    ) -> list[tuple[float, VectorRecord]]:
        scored = [(_cosine(query_vector, record.embedding), record) for record in records]
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
        return [(float(score), records[int(index_value)]) for score, index_value in zip(scores[0], indices[0]) if index_value >= 0]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dimensions": self.dimensions,
            "records": [
                {"text": record.text, "metadata": record.metadata, "embedding": record.embedding}
                for record in self.records
            ],
        }
        target.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(dimensions=payload["dimensions"])
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

            return "faiss-compatible"
        except Exception:
            return "python"
