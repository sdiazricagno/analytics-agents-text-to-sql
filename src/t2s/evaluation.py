"""Execution-accuracy comparison and difficulty labelling for the golden benchmark."""

from __future__ import annotations

import datetime as dt
from collections import Counter
from decimal import Decimal

import sqlglot
from sqlglot import exp


def normalise(v: object) -> object:
    """Map a DuckDB value to a comparable canonical form."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float, Decimal)):
        return float(f"{float(v):.8g}")
    if isinstance(v, dt.datetime):
        return v.date().isoformat() if v.time() == dt.time() else v.isoformat()
    if isinstance(v, dt.date):
        return v.isoformat()
    return str(v).strip()


def _column(rows: list[tuple], j: int) -> list[object]:
    return [normalise(r[j]) for r in rows]


def results_match(gold: list[tuple], pred: list[tuple]) -> bool:
    """True when every gold column maps to a distinct predicted column with equal row multisets."""
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    n_gold, n_pred = len(gold[0]), len(pred[0])
    if n_pred < n_gold:
        return False
    gold_cols = [_column(gold, j) for j in range(n_gold)]
    pred_cols = [_column(pred, k) for k in range(n_pred)]
    key = lambda x: (x is None, str(x))  # noqa: E731
    candidates = [
        [k for k in range(n_pred) if sorted(pred_cols[k], key=key) == sorted(gold_cols[j], key=key)]
        for j in range(n_gold)
    ]
    target = Counter(zip(*gold_cols, strict=True))

    def search(j: int, used: list[int]) -> bool:
        if j == n_gold:
            return Counter(zip(*(pred_cols[k] for k in used), strict=True)) == target
        return any(search(j + 1, [*used, k]) for k in candidates[j] if k not in used)

    return search(0, [])


def difficulty(sql: str) -> tuple[str, int]:
    """Label gold SQL as easy, medium, hard or extra and count the tables it touches."""
    tree = sqlglot.parse_one(sql, read="duckdb")
    ctes = {c.alias_or_name for c in tree.find_all(exp.CTE)}
    tables = {t.name for t in tree.find_all(exp.Table) if t.name not in ctes}
    nested = any(
        isinstance(n, (exp.Subquery, exp.CTE, exp.Window, exp.SetOperation))
        or (n is not tree and isinstance(n, exp.Select))
        for n in tree.walk()
    )
    if nested:
        return "extra", len(tables)
    return {1: "easy", 2: "medium"}.get(len(tables), "hard"), len(tables)
