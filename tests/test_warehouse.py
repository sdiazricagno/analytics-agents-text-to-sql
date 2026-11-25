"""Determinism of the synthetic warehouse."""

from __future__ import annotations

from t2s.warehouse import generate


def test_seeded_generation_is_reproducible() -> None:
    a, b = generate(seed=3, n_customers=200, n_orders=500), generate(seed=3, n_customers=200, n_orders=500)
    assert all(a[k].equals(b[k]) for k in a)
    assert len(a) == 10


def test_referential_integrity(con) -> None:  # noqa: ANN001
    orphans = con.execute(
        "SELECT COUNT(*) FROM order_items oi LEFT JOIN orders o USING (order_id) WHERE o.order_id IS NULL"
    ).fetchone()[0]
    assert orphans == 0
