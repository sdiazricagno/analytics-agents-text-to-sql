"""Semantic layer routing, compilation and follow-up handling."""

from __future__ import annotations

import pytest

from t2s.guardrails import SQLGuard
from t2s.semantic import SemanticLayer


@pytest.mark.parametrize(
    ("question", "metric", "dims"),
    [
        ("What was total revenue in 2024?", "revenue", ()),
        ("Revenue by region", "revenue", ("region",)),
        ("Top 5 categories by gross margin", "gross_margin", ("category",)),
        ("How many orders per channel?", "order_count", ("channel",)),
        ("Refund amount by department", "refund_amount", ("department",)),
        ("Revenue by customer region and segment", "revenue", ("customer_region", "segment")),
    ],
)
def test_routes_governed_questions(semantic: SemanticLayer, question: str, metric: str, dims: tuple) -> None:
    decision = semantic.parse(question)
    assert decision.query is not None, decision.reason
    assert decision.query.metric == metric and decision.query.dimensions == dims


@pytest.mark.parametrize(
    "question",
    [
        "Which customers never placed an order?",
        "What is the cumulative revenue by month in 2024?",
        "Rank store regions by 2025 revenue",
        "Which product had the most units sold in each department?",
        "Revenue and refunds by region",
    ],
)
def test_defers_what_it_cannot_govern(semantic: SemanticLayer, question: str) -> None:
    assert semantic.parse(question).query is None


def test_value_followed_by_noun_is_a_filter(semantic: SemanticLayer) -> None:
    q = semantic.parse("Revenue by product category for Flagship stores").query
    assert q is not None and q.dimensions == ("category",) and dict(q.filters) == {"store_type": "Flagship"}


def test_top_n_and_time_filters(semantic: SemanticLayer) -> None:
    q = semantic.parse("Top 3 stores by revenue in Q4 2024").query
    assert q is not None and (q.order, q.limit, q.year, q.quarter) == ("desc", 3, 2024, 4)


def test_follow_up_merges_previous_query(semantic: SemanticLayer) -> None:
    first = semantic.parse("Revenue by region in 2024").query
    follow = semantic.parse("what about by channel?", first).query
    assert follow is not None and follow.metric == "revenue" and follow.dimensions == ("channel",)
    assert follow.year == 2024
    narrowed = semantic.parse("only Online", follow).query
    assert narrowed is not None and dict(narrowed.filters) == {"channel": "Online"}


def test_store_region_uses_store_join_path(semantic: SemanticLayer) -> None:
    sql = semantic.compile(semantic.parse("Revenue by region").query)
    assert "JOIN stores" in sql and "JOIN customers" not in sql
    sql = semantic.compile(semantic.parse("Revenue by customer region").query)
    assert "JOIN customers" in sql and "JOIN stores" not in sql


def test_compiled_sql_passes_guardrails(semantic: SemanticLayer, guard: SQLGuard) -> None:
    for question in [
        "Revenue by department in 2025",
        "Units sold by supplier",
        "Number of returns by store type",
    ]:
        report = guard.check(semantic.compile(semantic.parse(question).query))
        assert report.ok, (question, report.errors)


def test_revenue_matches_hand_written_sql(semantic: SemanticLayer, con) -> None:  # noqa: ANN001
    governed = con.execute(semantic.compile(semantic.parse("total revenue in 2024").query)).fetchone()[0]
    manual = con.execute(
        "SELECT SUM(oi.quantity * oi.unit_price * (1 - oi.discount)) FROM order_items oi "
        "JOIN orders o USING (order_id) WHERE o.status <> 'Cancelled' AND year(o.order_date) = 2024"
    ).fetchone()[0]
    assert governed == manual
