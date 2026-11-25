"""Lexical schema retrieval and few-shot example selection."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from t2s.schema import Catalog

WORD = re.compile(r"[a-z0-9]+")
STOP = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "in",
    "on",
    "for",
    "by",
    "to",
    "what",
    "which",
    "how",
    "is",
    "are",
    "was",
    "were",
    "per",
    "each",
    "with",
    "that",
    "who",
    "from",
    "at",
    "as",
    "than",
    "more",
    "most",
    "do",
    "did",
    "does",
    "show",
    "list",
    "me",
    "give",
    "all",
    "their",
    "its",
    "has",
    "have",
    "many",
    "much",
    "there",
    "be",
    "any",
}

ANALYTIC_WORDS = {
    "average",
    "mean",
    "median",
    "total",
    "sum",
    "count",
    "number",
    "top",
    "bottom",
    "highest",
    "lowest",
    "rank",
    "share",
    "fraction",
    "percent",
    "percentage",
    "ratio",
    "rate",
    "cumulative",
    "running",
    "change",
    "growth",
    "previous",
    "month",
    "year",
    "quarter",
    "day",
    "week",
    "daily",
    "distinct",
    "unique",
    "never",
    "both",
    "only",
    "above",
    "below",
    "between",
    "least",
    "most",
    "first",
    "last",
    "trend",
    "compare",
    "versus",
    "ordered",
    "placed",
    "bought",
    "sold",
    "returned",
    "paid",
    "spent",
    "each",
    "every",
    "per",
    "counting",
    "id",
    "ids",
}


def tokens(text: str) -> list[str]:
    """Lowercase content tokens with naive singularisation."""
    out = []
    for w in WORD.findall(text.lower().replace("_", " ")):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return out


@dataclass
class SchemaRetriever:
    """Scores tables by question overlap and expands them into a joinable subset."""

    catalog: Catalog
    top_k: int = 4

    def __post_init__(self) -> None:
        self.docs: dict[str, Counter[str]] = {}
        for name, t in self.catalog.tables.items():
            text = " ".join([name, t.description, *t.synonyms, *t.columns, *t.descriptions.values()])
            self.docs[name] = Counter(tokens(text))
        df: Counter[str] = Counter()
        for doc in self.docs.values():
            df.update(set(doc))
        n = len(self.docs)
        semantic = self.catalog.raw.get("metrics", {}) | self.catalog.raw.get("dimensions", {})
        extra = " ".join(" ".join(v.get("synonyms", [])) + " " + k for k, v in semantic.items())
        self.vocab = set().union(*self.docs.values()) | set(tokens(extra)) | ANALYTIC_WORDS
        self.idf = {w: math.log(1 + n / c) for w, c in df.items()}

    def score(self, question: str) -> dict[str, float]:
        """Return a relevance score per table."""
        q = tokens(question)
        scores = {}
        for name, doc in self.docs.items():
            s = sum(self.idf.get(w, 0.0) * (1 + math.log(doc[w])) for w in q if doc[w])
            name_hit = any(
                w in tokens(name)
                or w in [tokens(x)[0] for x in self.catalog.tables[name].synonyms if tokens(x)]
                for w in q
            )
            scores[name] = s + (2.0 if name_hit else 0.0)
        return scores

    def grounding(self, question: str) -> float:
        """Fraction of non-numeric content tokens found in the schema and semantic vocabulary."""
        q = [w for w in tokens(question) if not w.isdigit()]
        if not q:
            return 0.0
        return sum(w in self.vocab for w in q) / len(q)

    def retrieve(self, question: str) -> list[str]:
        """Select the top tables and add bridge tables required to join them."""
        scores = self.score(question)
        ranked = [t for t, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s > 0][: self.top_k]
        if not ranked:
            return []
        return self.catalog.connect(ranked)


@dataclass
class Example:
    """A question and verified SQL used as a few-shot demonstration."""

    question: str
    sql: str


class FewShotStore:
    """Selects the most similar curated examples by token overlap."""

    def __init__(self, path: Path) -> None:
        self.examples = [Example(**e) for e in yaml.safe_load(path.read_text())]

    def select(self, question: str, k: int = 3) -> list[Example]:
        """Return the k examples with the highest Jaccard overlap."""
        q = set(tokens(question))

        def sim(e: Example) -> float:
            t = set(tokens(e.question))
            return len(q & t) / max(1, len(q | t))

        return sorted(self.examples, key=sim, reverse=True)[:k]
