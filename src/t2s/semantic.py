"""Governed semantic layer: metric and dimension matching plus deterministic SQL compilation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import duckdb

from t2s.schema import Catalog, Edge

GROUP_WORDS = {"by", "per", "each", "across", "every", "which", "what", "top", "bottom"}
FOLLOW_UP = re.compile(r"^(and|what about|how about|now|same|only|break (it|that) down|split|instead|just)\b")
ALLOWED = {
    "what",
    "whats",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "total",
    "overall",
    "by",
    "per",
    "in",
    "for",
    "each",
    "every",
    "top",
    "bottom",
    "show",
    "me",
    "give",
    "get",
    "how",
    "much",
    "many",
    "which",
    "has",
    "had",
    "have",
    "highest",
    "lowest",
    "most",
    "least",
    "best",
    "worst",
    "largest",
    "smallest",
    "biggest",
    "of",
    "and",
    "all",
    "across",
    "broken",
    "down",
    "breakdown",
    "split",
    "it",
    "that",
    "this",
    "to",
    "from",
    "with",
    "our",
    "we",
    "did",
    "do",
    "does",
    "generated",
    "made",
    "make",
    "sum",
    "value",
    "during",
    "on",
    "about",
    "now",
    "same",
    "only",
    "instead",
    "just",
    "please",
    "tell",
    "number",
    "count",
    "brought",
    "bring",
    "be",
    "there",
    "then",
    "us",
    "i",
    "can",
    "you",
    "see",
    "group",
    "grouped",
    "order",
    "ordered",
    "sorted",
    "year",
    "years",
    "month",
    "months",
    "quarter",
    "quarters",
    "q1",
    "q2",
    "q3",
    "q4",
    "achieved",
    "level",
    "trend",
    "over",
    "time",
    "monthly",
    "yearly",
    "quarterly",
    "annual",
}
MONTHS = {
    m: i + 1
    for i, m in enumerate(
        [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
    )
}
TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")


@dataclass(frozen=True)
class Metric:
    """A governed metric definition."""

    name: str
    synonyms: list[str]
    base: str
    expr: str
    filters: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Dimension:
    """A governed dimension definition."""

    name: str
    synonyms: list[str]
    column: str
    via: list[str] = field(default_factory=list)
    grain: str | None = None

    @property
    def table(self) -> str:
        """Table that owns the dimension column."""
        return self.column.split(".")[0]


@dataclass(frozen=True)
class SemanticQuery:
    """A structured request against governed metrics."""

    metric: str
    dimensions: tuple[str, ...] = ()
    filters: tuple[tuple[str, str], ...] = ()
    year: int | None = None
    quarter: int | None = None
    month: int | None = None
    order: str | None = None
    limit: int | None = None


@dataclass
class RouteDecision:
    """Outcome of semantic routing."""

    query: SemanticQuery | None
    reason: str


def _forms(phrase: str) -> list[str]:
    """Return a phrase with its simple plural."""
    if phrase.endswith("y") and not phrase.endswith("ay"):
        return [phrase, phrase[:-1] + "ies"]
    if phrase.endswith(("s", "ly", "al")):
        return [phrase]
    return [phrase, phrase + "s"]


class SemanticLayer:
    """Matches questions to governed metrics and compiles them to SQL."""

    def __init__(self, catalog: Catalog, con: duckdb.DuckDBPyConnection | None = None) -> None:
        self.catalog = catalog
        raw = catalog.raw
        self.metrics = {k: Metric(name=k, **v) for k, v in raw["metrics"].items()}
        self.dimensions = {k: Dimension(name=k, **v) for k, v in raw["dimensions"].items()}
        self.values: dict[str, tuple[str, str]] = {}
        if con is not None:
            self._index_values(con)
        self.phrases = self._build_phrases()

    def _index_values(self, con: duckdb.DuckDBPyConnection) -> None:
        for dim in self.dimensions.values():
            if dim.grain or dim.name in {"product", "customer_region"}:
                continue
            table, column = dim.column.split(".")
            for (value,) in con.execute(f"SELECT DISTINCT {column} FROM {table}").fetchall():
                key = str(value).lower()
                self.values.setdefault(key, (dim.name, str(value)))

    def _build_phrases(self) -> list[tuple[str, str, str]]:
        phrases = []
        for m in self.metrics.values():
            phrases.extend((s.lower(), "metric", m.name) for s in m.synonyms)
        for d in self.dimensions.values():
            for syn in d.synonyms:
                phrases.extend((form, "dim", d.name) for form in _forms(syn.lower()))
        phrases.extend((v, "value", v) for v in self.values)
        return sorted(phrases, key=lambda p: -len(p[0]))

    def _spans(self, text: str) -> list[tuple[int, int, str, str]]:
        taken = [False] * len(text)
        found = []
        for phrase, kind, ref in self.phrases:
            for m in re.finditer(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text):
                if any(taken[m.start() : m.end()]):
                    continue
                for i in range(m.start(), m.end()):
                    taken[i] = True
                found.append((m.start(), m.end(), kind, ref))
        return sorted(found)

    def parse(self, question: str, previous: SemanticQuery | None = None) -> RouteDecision:
        """Map a question to a SemanticQuery or explain why it cannot be governed."""
        text = question.lower().strip().rstrip("?.!")
        text = re.sub(r"\bq([1-4])\b", r"quarter \1", text)
        spans = self._spans(text)
        metrics = [ref for _, _, kind, ref in spans if kind == "metric"]
        weak = {"order_count", "return_count", "amount_paid"}
        strong = [m for m in metrics if m not in weak]
        if strong:
            metrics = strong
        distinct_metrics = list(dict.fromkeys(metrics))
        is_follow_up = bool(FOLLOW_UP.match(text)) and previous is not None
        if len(distinct_metrics) > 1:
            return RouteDecision(None, f"multiple metrics requested: {distinct_metrics}")
        if not distinct_metrics and not is_follow_up:
            return RouteDecision(None, "no governed metric matched")

        dims: list[str] = []
        filters: dict[str, str] = {}
        rest = text
        for start, end, _, _ in reversed(spans):
            rest = rest[:start] + " " + rest[end:]
        for _, _, kind, ref in spans:
            if kind == "value":
                dim_name, value = self.values[ref]
                filters[dim_name] = value
        prev_end, prev_kind = -10, ""
        for start, end, kind, ref in spans:
            adjacent_to_value = prev_kind == "value" and not text[prev_end:start].strip()
            prev_end, prev_kind = end, kind
            if kind != "dim" or adjacent_to_value:
                continue
            before = TOKEN.findall(text[:start])[-3:]
            grouped = any(w in GROUP_WORDS for w in before) or bool(
                re.search(r"top \d+|bottom \d+", text[:start])
            )
            if ref in filters and not grouped:
                continue
            if ref not in dims:
                dims.append(ref)

        year = quarter = month = None
        for y in re.findall(r"\b(20\d\d)\b", rest):
            year = int(y)
        rest = re.sub(r"\b20\d\d\b", " ", rest)
        qm = re.search(r"quarter (\d)", text)
        if qm:
            quarter = int(qm.group(1))
            rest = re.sub(r"\b[1-4]\b", " ", rest, count=1)
        for name, idx in MONTHS.items():
            if re.search(rf"\b{name}\b", rest):
                month = idx
                rest = re.sub(rf"\b{name}\b", " ", rest)
        if quarter and "quarter" in dims and not re.search(r"(by|per|each) quarter", text):
            dims.remove("quarter")
        order = limit = None
        superlative = re.search(
            r"\b(highest|most|best|largest|biggest|lowest|least|worst|smallest|top|bottom)\b", text
        )
        if superlative and re.search(r"\b(each|every|per)\b", text):
            return RouteDecision(None, "top-N within groups needs window functions")
        top = re.search(r"\b(top|bottom) (\d+)\b", text)
        if top:
            order = "desc" if top.group(1) == "top" else "asc"
            limit = int(top.group(2))
            rest = re.sub(r"\b\d+\b", " ", rest, count=1)
        elif re.search(r"\b(highest|most|best|largest|biggest)\b", text) and dims:
            order, limit = "desc", 1
        elif re.search(r"\b(lowest|least|worst|smallest)\b", text) and dims:
            order, limit = "asc", 1

        leftovers = [w for w in TOKEN.findall(rest) if w not in ALLOWED and not w.isdigit()]
        if leftovers:
            return RouteDecision(None, f"unrecognised terms: {leftovers}")

        if is_follow_up and previous is not None:
            merged = dict(previous.filters)
            merged.update(filters)
            new_dims = tuple(dict.fromkeys(dims)) if dims else previous.dimensions
            query = replace(
                previous,
                metric=distinct_metrics[0] if distinct_metrics else previous.metric,
                dimensions=new_dims,
                filters=tuple(merged.items()),
                year=year or previous.year,
                quarter=quarter or previous.quarter,
                month=month or previous.month,
                order=order or previous.order,
                limit=limit or previous.limit,
            )
            return RouteDecision(query, "follow-up merged with previous governed query")
        if "region" in dims and "customer_region" in dims:
            return RouteDecision(None, "store and customer region in one query need table aliases")
        query = SemanticQuery(
            metric=distinct_metrics[0],
            dimensions=tuple(dims),
            filters=tuple(filters.items()),
            year=year,
            quarter=quarter,
            month=month,
            order=order,
            limit=limit,
        )
        return RouteDecision(query, "matched governed metric")

    def _join_edges(self, base: str, needed: list[tuple[str, list[str]]]) -> list[Edge]:
        edges: list[Edge] = []
        joined = {base}
        for table, via in needed:
            hops = [*via, table]
            cur = base
            for hop in hops:
                if hop == cur:
                    continue
                for e in self.catalog.shortest_path(
                    cur, hop, avoid=frozenset({"customers", "stores"} - set(hops))
                ):
                    if e not in edges:
                        edges.append(e)
                        joined.update({e.left_table, e.right_table})
                cur = hop
        return edges

    def compile(self, q: SemanticQuery) -> str:
        """Compile a SemanticQuery into DuckDB SQL using declared join paths."""
        metric = self.metrics[q.metric]
        dims = [self.dimensions[d] for d in q.dimensions]
        filter_dims = [(self.dimensions[d], v) for d, v in q.filters]
        needed: list[tuple[str, list[str]]] = []
        for d in dims + [fd for fd, _ in filter_dims]:
            needed.append((d.table, d.via))
        for t in metric.requires:
            needed.append((t, []))
        for f in metric.filters:
            needed.append((f.split(".")[0], []))
        if q.year or q.quarter or q.month:
            needed.append(("orders", []))
        edges = self._join_edges(metric.base, needed)

        select, group = [], []
        for d in dims:
            if d.grain == "year":
                expr = f"EXTRACT(year FROM {d.column})"
            elif d.grain:
                expr = f"DATE_TRUNC('{d.grain}', {d.column})"
            else:
                expr = d.column
            select.append(f"{expr} AS {d.name}")
            group.append(expr)
        select.append(f"{metric.expr} AS {metric.name}")

        where = list(metric.filters)
        where += [f"{fd.column} = '{v.replace(chr(39), chr(39) * 2)}'" for fd, v in filter_dims]
        if q.year:
            where.append(f"EXTRACT(year FROM orders.order_date) = {q.year}")
        if q.quarter:
            where.append(f"EXTRACT(quarter FROM orders.order_date) = {q.quarter}")
        if q.month:
            where.append(f"EXTRACT(month FROM orders.order_date) = {q.month}")

        sql = [f"SELECT {', '.join(select)}", f"FROM {metric.base}"]
        joined = {metric.base}
        for e in edges:
            new = e.right_table if e.left_table in joined else e.left_table
            sql.append(f"JOIN {new} ON {e.condition()}")
            joined.add(new)
        if where:
            sql.append("WHERE " + " AND ".join(where))
        if group:
            sql.append("GROUP BY " + ", ".join(group))
        if q.order:
            sql.append(f"ORDER BY {metric.name} {q.order.upper()}")
        elif group:
            sql.append("ORDER BY " + ", ".join(group))
        if q.limit:
            sql.append(f"LIMIT {q.limit}")
        return "\n".join(sql)

    def definitions(self) -> str:
        """Business definitions rendered for LLM prompts."""
        lines = []
        for m in self.metrics.values():
            cond = f" WHERE {' AND '.join(m.filters)}" if m.filters else ""
            lines.append(f"-- {m.name}: {m.expr}{cond}")
        return "\n".join(lines)
