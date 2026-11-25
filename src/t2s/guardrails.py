"""SQL guardrails: parse, read-only, grounding, join-path, row-limit and dry-run checks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import duckdb
import sqlglot
from sqlglot import exp

from t2s.schema import Catalog

FORBIDDEN_NODES = tuple(
    getattr(exp, n)
    for n in (
        "Insert",
        "Update",
        "Delete",
        "Drop",
        "Create",
        "Alter",
        "Command",
        "Merge",
        "Copy",
        "Pragma",
        "Set",
        "Attach",
        "Detach",
        "Use",
        "Transaction",
        "Commit",
        "Rollback",
        "TruncateTable",
        "Grant",
        "Install",
        "LoadData",
        "Export",
    )
    if hasattr(exp, n)
)
FORBIDDEN_FUNCS = re.compile(
    r"^(read_|glob$|getenv$|current_setting$|query$|query_table$|sniff_csv|parquet_|duckdb_|pragma_)"
)
FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


@dataclass
class GuardrailReport:
    """Result of validating one SQL candidate."""

    ok: bool
    sql: str | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unknown_tables: list[str] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        """True when the SQL references identifiers absent from the schema."""
        return bool(self.unknown_tables or self.unknown_columns)

    def feedback(self) -> str:
        """Render errors as correction feedback for the generator."""
        return "\n".join(f"- {e}" for e in self.errors)


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of model output."""
    m = FENCE.search(text)
    body = m.group(1) if m else text
    body = body.strip()
    start = re.search(r"\b(with|select)\b", body, re.I)
    if start:
        body = body[start.start() :]
    return body.strip().rstrip(";").strip()


def connect_read_only(db_path: str) -> duckdb.DuckDBPyConnection:
    """Open the warehouse read-only with external file and network access disabled."""
    return duckdb.connect(db_path, read_only=True, config={"enable_external_access": False})


class SQLGuard:
    """Validates and normalises generated SQL before execution."""

    def __init__(self, catalog: Catalog, con: duckdb.DuckDBPyConnection, max_rows: int = 1000) -> None:
        self.catalog = catalog
        self.con = con
        self.max_rows = max_rows

    def check(self, sql: str) -> GuardrailReport:
        """Run every guardrail and return the possibly rewritten SQL."""
        report = GuardrailReport(ok=False, sql=None)
        try:
            statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
        except sqlglot.errors.ParseError as e:
            report.errors.append(f"SQL does not parse: {str(e).splitlines()[0]}")
            return report
        if len(statements) != 1:
            report.errors.append(f"expected exactly one statement, got {len(statements)}")
            return report
        root = statements[0]
        if not isinstance(root, (exp.Select, exp.SetOperation)):
            report.errors.append(f"only SELECT queries are allowed, got {type(root).__name__}")
            return report
        bad = [type(n).__name__ for n in root.walk() if isinstance(n, FORBIDDEN_NODES)]
        if bad:
            report.errors.append(f"write or session statements are not allowed: {sorted(set(bad))}")
            return report
        for fn in root.find_all(exp.Func):
            name = (fn.sql_name() if not isinstance(fn, exp.Anonymous) else fn.name).lower()
            if FORBIDDEN_FUNCS.match(name):
                report.errors.append(f"function {name} is not allowed")
        self._check_identifiers(root, report)
        self._check_joins(root, report)
        if report.errors:
            return report
        self._enforce_limit(root)
        final = root.sql(dialect="duckdb")
        try:
            self.con.execute(f"EXPLAIN {final}")
        except duckdb.Error as e:
            report.errors.append(f"dry-run failed: {str(e).splitlines()[0]}")
            return report
        report.ok = True
        report.sql = final
        return report

    def _check_identifiers(self, root: exp.Expression, report: GuardrailReport) -> None:
        ctes = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
        aliases: dict[str, str | None] = {}
        for t in root.find_all(exp.Table):
            if isinstance(t.this, exp.Func):
                report.errors.append("table functions are not allowed")
                continue
            name = t.name.lower()
            if t.args.get("db") or t.args.get("catalog"):
                report.errors.append(f"schema-qualified table {t.sql()} is not allowed")
            if name not in self.catalog.tables and name not in ctes:
                report.unknown_tables.append(name)
            aliases[t.alias_or_name.lower()] = name if name in self.catalog.tables else None
        for sub in root.find_all(exp.Subquery):
            if sub.alias:
                aliases[sub.alias.lower()] = None
        known = set(self.catalog.all_columns)
        known |= {a.alias.lower() for a in root.find_all(exp.Alias) if a.alias}
        for node in root.find_all(exp.TableAlias):
            known |= {c.name.lower() for c in node.columns}
        for col in root.find_all(exp.Column):
            if isinstance(col.this, exp.Star):
                continue
            name, qual = col.name.lower(), col.table.lower()
            if qual and qual in aliases and aliases[qual] is not None:
                if name not in self.catalog.tables[aliases[qual]].columns:
                    report.unknown_columns.append(f"{qual}.{name}")
            elif qual and qual not in aliases and qual not in ctes:
                report.unknown_columns.append(f"{qual}.{name}")
            elif name not in known:
                report.unknown_columns.append(name)
        report.unknown_tables = sorted(set(report.unknown_tables))
        report.unknown_columns = sorted(set(report.unknown_columns))
        if report.unknown_tables:
            report.errors.append(f"unknown tables: {report.unknown_tables}; use only tables in the schema")
        if report.unknown_columns:
            report.errors.append(f"unknown columns: {report.unknown_columns}; use only columns in the schema")
        self._aliases = aliases

    def _check_joins(self, root: exp.Expression, report: GuardrailReport) -> None:
        aliases = getattr(self, "_aliases", {})
        for join in root.find_all(exp.Join):
            on, using = join.args.get("on"), join.args.get("using")
            kind = (join.args.get("kind") or "").upper()
            if on is None and not using and kind != "CROSS" and not join.args.get("method"):
                if isinstance(join.this, (exp.Table, exp.Subquery)):
                    report.warnings.append(f"join without condition: {join.this.sql()}")
                continue
            if on is None:
                continue
            for eq in on.find_all(exp.EQ):
                left, right = eq.left, eq.right
                if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
                    continue
                lt, rt = aliases.get(left.table.lower()), aliases.get(right.table.lower())
                if not lt or not rt or lt == rt:
                    continue
                a, b = (lt, left.name.lower()), (rt, right.name.lower())
                if self.catalog.is_fk_pair(a, b):
                    continue
                if a[1].endswith("_id") or b[1].endswith("_id"):
                    report.errors.append(
                        f"join {lt}.{a[1]} = {rt}.{b[1]} is not a declared relationship; "
                        "use the valid join keys"
                    )
                else:
                    report.warnings.append(f"non-key join {lt}.{a[1]} = {rt}.{b[1]}")

    def _enforce_limit(self, root: exp.Expression) -> None:
        limit = root.args.get("limit")
        if limit is None:
            root.set("limit", exp.Limit(expression=exp.Literal.number(self.max_rows)))
            return
        value = limit.expression
        if isinstance(value, exp.Literal) and value.is_int and int(value.this) > self.max_rows:
            limit.set("expression", exp.Literal.number(self.max_rows))
