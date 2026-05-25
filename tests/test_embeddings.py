import math
import os

import pytest

from sec_memo_agents.core.embeddings import HashEmbedder, build_embedder


def test_hash_embedder_is_deterministic_and_normalized():
    embedder = HashEmbedder(dimensions=128)
    text = "cloud revenue grew on enterprise software demand"

    first = embedder.embed([text])[0]
    second = embedder.embed([text])[0]

    assert first == second  # deterministic
    assert len(first) == 128
    assert embedder.semantic is False
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0, rel_tol=1e-6)


def test_build_embedder_hash_backend():
    embedder = build_embedder(backend="hash", hash_dimensions=64)
    assert isinstance(embedder, HashEmbedder)
    assert embedder.dimensions == 64
    assert embedder.semantic is False


@pytest.mark.skipif(
    os.getenv("EMBEDDING_BACKEND") == "hash",
    reason="semantic backend disabled (EMBEDDING_BACKEND=hash); skips model download in CI",
)
def test_semantic_embedder_beats_hash_on_paraphrase():
    """Semantic embeddings should rank a paraphrase above a lexical decoy.

    The query shares almost no tokens with the paraphrase but is topically
    identical; it shares surface tokens with the decoy but is topically
    unrelated. A real embedder ranks the paraphrase higher; the lexical hash
    embedder does not.
    """
    pytest.importorskip("sentence_transformers")

    query = "data center compute capacity expanded for cloud customers"
    paraphrase = "we grew server infrastructure to host enterprise SaaS workloads"
    decoy = "the data on customers showed expanded demand for grocery capacity"

    semantic = build_embedder(backend="semantic")

    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    q, p, d = semantic.embed([query, paraphrase, decoy])
    assert cosine(q, p) > cosine(q, d)
