"""Runtime settings resolved from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Paths, LLM endpoint and guardrail limits."""

    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("T2S_DB_PATH", ROOT / "data" / "warehouse.duckdb"))
    )
    semantic_path: Path = field(
        default_factory=lambda: Path(os.getenv("T2S_SEMANTIC_PATH", ROOT / "semantic" / "retail.yaml"))
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("T2S_LLM_BASE_URL", "http://localhost:11434/v1")
    )
    llm_model: str = field(default_factory=lambda: os.getenv("T2S_LLM_MODEL", "qwen2.5-coder:3b"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("T2S_LLM_API_KEY", "not-needed"))
    llm_timeout_s: float = field(default_factory=lambda: float(os.getenv("T2S_LLM_TIMEOUT_S", "300")))
    max_rows: int = 1000
    max_retries: int = 2
    query_timeout_s: float = 20.0
