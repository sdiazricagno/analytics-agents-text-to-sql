"""Guardrail behaviour on safe, unsafe and hallucinated SQL."""

from __future__ import annotations

import duckdb
import pytest

from t2s.guardrails import SQLGuard, extract_sql


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "DROP TABLE customers",
        "UPDATE products SET list_price = 0",
        "INSERT INTO regions VALUES (9, 'X', 'Y')",
        "CREATE TABLE t AS SELECT 1",
        "ATTACH 'other.db'",
        "PRAGMA database_list",
        "COPY orders TO 'out.csv'",
    ],
)
def test_rejects_non_select(guard: SQLGuard, sql: str) -> None:
    assert not guard.check(sql).ok


def test_rejects_stacked_statements(guard: SQLGuard) -> None:
    report = guard.check("SELECT 1; DROP TABLE orders")
    assert not report.ok and "exactly one statement" in report.errors[0]


def test_rejects_unparseable(guard: SQLGuard) -> None:
    assert not guard.check("SELEC order_id FRM orders").ok


def test_flags_unknown_table(guard: SQLGuard) -> None:
    report = guard.check("SELECT * FROM employees")
    assert not report.ok and report.unknown_tables == ["employees"] and report.hallucinated


def test_flags_unknown_column(guard: SQLGuard) -> None:
    report = guard.check("SELECT o.revenue FROM orders o")
    assert not report.ok and report.unknown_columns == ["o.revenue"]


def test_flags_column_on_wrong_table(guard: SQLGuard) -> None:
    report = guard.check("SELECT c.channel FROM customers c")
    assert report.unknown_columns == ["c.channel"]


@pytest.mark.parametrize(
    "sql",
    ["SELECT * FROM read_csv('/etc/passwd')", "SELECT getenv('HOME')", "SELECT * FROM glob('*')"],
)
def test_rejects_file_and_env_access(guard: SQLGuard, sql: str) -> None:
    assert not guard.check(sql).ok


def test_rejects_undeclared_key_join(guard: SQLGuard) -> None:
    report = guard.check("SELECT 1 FROM orders o JOIN payments p ON o.order_id = p.payment_id")
    assert not report.ok and "not a declared relationship" in report.errors[0]


def test_accepts_sibling_foreign_keys(guard: SQLGuard) -> None:
    sql = "SELECT COUNT(*) FROM order_items oi JOIN payments p ON p.order_id = oi.order_id"
    assert guard.check(sql).ok


def test_injects_and_clamps_limit(guard: SQLGuard) -> None:
    assert guard.check("SELECT order_id FROM orders").sql.endswith("LIMIT 1000")
    assert guard.check("SELECT order_id FROM orders LIMIT 99999").sql.endswith("LIMIT 1000")
    assert guard.check("SELECT order_id FROM orders LIMIT 5").sql.endswith("LIMIT 5")


def test_dry_run_catches_binder_errors(guard: SQLGuard) -> None:
    report = guard.check("SELECT region_id FROM stores s JOIN regions r ON s.region_id = r.region_id")
    assert not report.ok and "dry-run failed" in report.errors[0]


def test_accepts_ctes_subqueries_and_aliases(guard: SQLGuard) -> None:
    sql = """
    WITH per_customer AS (SELECT customer_id, COUNT(*) AS n FROM orders GROUP BY customer_id)
    SELECT AVG(n) AS avg_orders FROM per_customer WHERE n > (SELECT MIN(n) FROM per_customer)
    """
    report = guard.check(sql)
    assert report.ok, report.errors


def test_read_only_connection_blocks_writes(con) -> None:  # noqa: ANN001
    with pytest.raises(duckdb.Error):
        con.execute("CREATE TABLE hack AS SELECT 1")


def test_extract_sql_from_markdown() -> None:
    assert extract_sql("Here you go:\n```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("```sql\nSELECT 2") == "SELECT 2"
