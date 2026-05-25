"""Embedding backends for filing retrieval.

This module provides two embedders behind a common interface:

- ``SentenceTransformerEmbedder``: real semantic embeddings produced by a local
  sentence-transformers model (default ``all-MiniLM-L6-v2``, 384 dimensions).
  Runs on CPU, requires no API key, and captures meaning rather than exact
  token overlap. This is the production path.
- ``HashEmbedder``: a dependency-free, deterministic hashing embedder. It is a
  lexical fingerprint (token-overlap), NOT semantic. It exists only as an
  offline fallback for environments where sentence-transformers is unavailable
  (e.g. fast CI) so the rest of the system stays runnable.

``build_embedder`` selects between them. The honest distinction matters: only
the sentence-transformer path provides true semantic retrieval.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]+")

DEFAULT_SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class Embedder(Protocol):
    """Common interface for embedding backends."""

    dimensions: int
    name: str
    semantic: bool

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbedder:
    """Deterministic hashing embedder.

    Lexical, not semantic: two texts are 'similar' only when they share exact
    tokens. Kept as a zero-dependency fallback, never the preferred backend.
    """

    semantic = False

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions
        self.name = f"hash-{dimensions}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            signed_digest = hashlib.blake2b(("salt:" + token).encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self.dimensions
            sign = -1.0 if int.from_bytes(signed_digest, "big") % 2 else 1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class SentenceTransformerEmbedder:
    """Real semantic embeddings via a local sentence-transformers model.

    Embeddings are L2-normalized so inner product equals cosine similarity,
    which keeps the FAISS ``IndexFlatIP`` search and the Python cosine fallback
    consistent.
    """

    semantic = True

    def __init__(self, model_name: str = DEFAULT_SEMANTIC_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimensions = int(self._model.get_sentence_embedding_dimension())
        self.name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]


def build_embedder(
    backend: str = "auto",
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    hash_dimensions: int = 384,
) -> Embedder:
    """Build an embedder.

    backend:
        - "semantic": require the sentence-transformer model (raises if missing).
        - "hash": force the deterministic lexical fallback.
        - "auto" (default): use the semantic model if installed, else fall back
          to hash without failing.
    """

    backend = (backend or "auto").lower()
    if backend == "hash":
        return HashEmbedder(hash_dimensions)
    if backend == "semantic":
        return SentenceTransformerEmbedder(model_name)
    # auto
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception:
        return HashEmbedder(hash_dimensions)
