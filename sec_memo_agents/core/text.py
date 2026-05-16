"""Text cleaning, filing section extraction, and chunking helpers."""

from __future__ import annotations

import html
import re


TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style).*?>.*?</\1>", re.IGNORECASE | re.DOTALL)
SPACE_RE = re.compile(r"\s+")


SECTION_PATTERNS = {
    "business": re.compile(r"item\s+1[.\s]+business", re.IGNORECASE),
    "risk_factors": re.compile(r"item\s+1a[.\s]+risk\s+factors", re.IGNORECASE),
    "mda": re.compile(r"item\s+7[.\s]+management'?s\s+discussion", re.IGNORECASE),
    "financial_statements": re.compile(r"item\s+8[.\s]+financial\s+statements", re.IGNORECASE),
    "controls": re.compile(r"item\s+9a[.\s]+controls", re.IGNORECASE),
}


def clean_filing_text(raw: str) -> str:
    """Convert SEC HTML/SGML filing text into compact analyst-readable text."""

    without_scripts = SCRIPT_STYLE_RE.sub(" ", raw)
    without_tags = TAG_RE.sub(" ", without_scripts)
    unescaped = html.unescape(without_tags)
    cleaned = SPACE_RE.sub(" ", unescaped)
    return cleaned.strip()


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 220) -> list[str]:
    """Split text into overlapping chunks without cutting every sentence midstream."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            sentence_boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
            if sentence_boundary > start + int(chunk_size * 0.55):
                end = sentence_boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def extract_sections(text: str, max_chars: int = 12000) -> dict[str, str]:
    """Extract coarse 10-K sections by item heading heuristics."""

    matches: list[tuple[str, int]] = []
    for section, pattern in SECTION_PATTERNS.items():
        match = pattern.search(text)
        if match:
            matches.append((section, match.start()))

    matches.sort(key=lambda item: item[1])
    sections: dict[str, str] = {}
    for index, (name, start) in enumerate(matches):
        end = matches[index + 1][1] if index + 1 < len(matches) else min(len(text), start + max_chars)
        sections[name] = text[start:end][:max_chars].strip()
    return sections


def summarize_snippet(text: str, max_chars: int = 550) -> str:
    compact = SPACE_RE.sub(" ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3].rsplit(" ", 1)[0] + "..."
