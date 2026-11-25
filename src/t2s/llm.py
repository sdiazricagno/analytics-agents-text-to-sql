"""Pluggable chat-completion clients for any OpenAI-compatible endpoint."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from openai import OpenAI

from t2s.config import Settings


class LLM(Protocol):
    """Minimal chat interface used by the agents."""

    name: str

    def complete(self, system: str, user: str) -> str:
        """Return the assistant reply for one system and user message."""
        ...


class OpenAICompatibleLLM:
    """Client for Ollama, llama.cpp server, vLLM or any hosted OpenAI-style API."""

    def __init__(self, settings: Settings | None = None, cache_dir: Path | None = None) -> None:
        s = settings or Settings()
        self.name = s.llm_model
        self.client = OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=s.llm_timeout_s)
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def complete(self, system: str, user: str) -> str:
        """Call the endpoint deterministically, with an optional on-disk cache."""
        key = hashlib.sha256(json.dumps([self.name, system, user]).encode()).hexdigest()[:24]
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            return json.loads(path.read_text())["content"]
        resp = self.client.chat.completions.create(
            model=self.name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            seed=0,
            max_tokens=400,
            stop=["\n```"],
        )
        content = resp.choices[0].message.content or ""
        if path:
            path.write_text(json.dumps({"model": self.name, "content": content}))
        return content


class ScriptedLLM:
    """Test double that answers from a callable."""

    def __init__(self, fn: Callable[[str, str], str], name: str = "scripted") -> None:
        self.fn = fn
        self.name = name
        self.calls: list[str] = []

    def complete(self, system: str, user: str) -> str:
        """Return the scripted reply."""
        self.calls.append(user)
        return self.fn(system, user)
