"""Execution-match logic used by the benchmark."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from t2s.evaluation import difficulty, results_match


def test_ignores_row_and_column_order() -> None:
    gold = [("a", 1), ("b", 2)]
    assert results_match(gold, [(2, "b"), (1, "a")])


def test_allows_extra_predicted_columns() -> None:
    assert results_match([("Store 01",)], [(7, "Store 01", 123.4)])


def test_rejects_missing_rows_or_values() -> None:
    assert not results_match([("a", 1), ("b", 2)], [("a", 1)])
    assert not results_match([("a", 1)], [("a", 2)])
    assert not results_match([("a", 1)], [(1,)])


def test_numeric_and_date_normalisation() -> None:
    assert results_match([(Decimal("10.50"), dt.date(2024, 1, 1))], [(10.5, dt.datetime(2024, 1, 1))])


def test_keeps_rows_aligned() -> None:
    assert not results_match([("a", 1), ("b", 2)], [("a", 2), ("b", 1)])


def test_difficulty_labels() -> None:
    assert difficulty("SELECT COUNT(*) FROM orders") == ("easy", 1)
    assert difficulty("SELECT 1 FROM orders o JOIN stores s ON o.store_id = s.store_id") == ("medium", 2)
    assert (
        difficulty(
            "SELECT customer_id FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM orders)"
        )[0]
        == "extra"
    )
