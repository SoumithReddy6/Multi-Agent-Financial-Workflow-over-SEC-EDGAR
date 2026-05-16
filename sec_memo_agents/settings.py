"""Runtime settings for SEC memo agents."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    sec_user_agent: str
    sec_max_requests_per_second: float
    sec_cache_dir: Path
    vector_index_dir: Path
    llm_provider: str
    openai_api_key: str
    anthropic_api_key: str
    openai_model: str
    anthropic_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int

    @property
    def sec_headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        sec_user_agent=os.getenv(
            "SEC_USER_AGENT",
            "SEC-Memo-Agents/0.1 portfolio-demo contact@example.com",
        ),
        sec_max_requests_per_second=min(_float_env("SEC_MAX_REQUESTS_PER_SECOND", 8.0), 10.0),
        sec_cache_dir=Path(os.getenv("SEC_CACHE_DIR", ".cache/sec")),
        vector_index_dir=Path(os.getenv("VECTOR_INDEX_DIR", "artifacts/vector_index")),
        llm_provider=os.getenv("LLM_PROVIDER", "offline").lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        embedding_dimensions=_int_env("EMBEDDING_DIMENSIONS", 384),
        chunk_size=_int_env("CHUNK_SIZE", 1800),
        chunk_overlap=_int_env("CHUNK_OVERLAP", 220),
    )
