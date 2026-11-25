"""Schema catalog, foreign-key join graph and join-path planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import duckdb
import yaml


@dataclass(frozen=True)
class Edge:
    """A declared foreign-key relationship between two columns."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str

    def other(self, table: str) -> str:
        """Return the table on the opposite side of the edge."""
        return self.right_table if table == self.left_table else self.left_table

    def condition(self) -> str:
        """Return the SQL equality for this edge."""
        return f"{self.left_table}.{self.left_column} = {self.right_table}.{self.right_column}"


@dataclass
class Table:
    """A warehouse table with typed, described columns."""

    name: str
    description: str
    synonyms: list[str]
    columns: dict[str, str]
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class Catalog:
    """Tables, columns and relationships known to the system."""

    tables: dict[str, Table]
    edges: list[Edge]
    raw: dict

    @classmethod
    def load(cls, semantic_path: Path, con: duckdb.DuckDBPyConnection) -> Catalog:
        """Load descriptions from YAML and column types from the warehouse."""
        raw = yaml.safe_load(semantic_path.read_text())
        rows = con.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
        ).fetchall()
        types: dict[str, dict[str, str]] = {}
        for table, column, dtype in rows:
            types.setdefault(table, {})[column] = dtype
        tables = {}
        for name, cols in types.items():
            meta = raw["tables"].get(name, {})
            tables[name] = Table(
                name=name,
                description=meta.get("description", ""),
                synonyms=meta.get("synonyms", []),
                columns=cols,
                descriptions=meta.get("columns", {}),
            )
        edges = []
        for left, right in raw["relationships"]:
            lt, lc = left.split(".")
            rt, rc = right.split(".")
            edges.append(Edge(lt, lc, rt, rc))
        return cls(tables=tables, edges=edges, raw=raw)

    @cached_property
    def adjacency(self) -> dict[str, list[Edge]]:
        """Map each table to its incident edges."""
        adj: dict[str, list[Edge]] = {t: [] for t in self.tables}
        for e in self.edges:
            adj[e.left_table].append(e)
            adj[e.right_table].append(e)
        return adj

    @cached_property
    def all_columns(self) -> set[str]:
        """Every column name in the warehouse."""
        return {c for t in self.tables.values() for c in t.columns}

    def sqlglot_schema(self) -> dict[str, dict[str, str]]:
        """Return the schema in sqlglot's mapping format."""
        return {name: dict(t.columns) for name, t in self.tables.items()}

    def canonical_key(self, col: tuple[str, str]) -> tuple[str, str]:
        """Resolve a foreign-key column to the key it references."""
        for e in self.edges:
            if (e.left_table, e.left_column) == col:
                return e.right_table, e.right_column
        return col

    def is_fk_pair(self, a: tuple[str, str], b: tuple[str, str]) -> bool:
        """True when two columns are a declared relationship or reference the same key."""
        return self.canonical_key(a) == self.canonical_key(b)

    def shortest_path(self, start: str, goal: str, avoid: frozenset[str] = frozenset()) -> list[Edge]:
        """Breadth-first shortest join path between two tables."""
        if start == goal:
            return []
        prev: dict[str, tuple[str, Edge]] = {}
        seen = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for e in self.adjacency[node]:
                nxt = e.other(node)
                if nxt in seen or (nxt in avoid and nxt != goal):
                    continue
                seen.add(nxt)
                prev[nxt] = (node, e)
                if nxt == goal:
                    path = []
                    cur = goal
                    while cur != start:
                        cur, edge = prev[cur]
                        path.append(edge)
                    return path[::-1]
                queue.append(nxt)
        raise ValueError(f"no join path between {start} and {goal}")

    def connect(self, tables: list[str]) -> list[str]:
        """Expand a table set with the bridge tables needed to join it."""
        if not tables:
            return []
        keep = [tables[0]]
        for t in tables[1:]:
            best: list[Edge] | None = None
            for k in keep:
                path = self.shortest_path(k, t)
                if best is None or len(path) < len(best):
                    best = path
            for e in best or []:
                for side in (e.left_table, e.right_table):
                    if side not in keep:
                        keep.append(side)
            if t not in keep:
                keep.append(t)
        return keep

    def describe(self, tables: list[str]) -> str:
        """Render compact DDL with comments and join hints for prompting."""
        lines = []
        for name in tables:
            t = self.tables[name]
            lines.append(f"-- {t.description}")
            lines.append(f"CREATE TABLE {name} (")
            cols = []
            for col, dtype in t.columns.items():
                note = t.descriptions.get(col, "")
                cols.append(f"  {col} {dtype}" + (f"  -- {note}" if note else ""))
            lines.append(",\n".join(cols))
            lines.append(");")
        joins = [e.condition() for e in self.edges if e.left_table in tables and e.right_table in tables]
        if joins:
            lines.append("-- Valid join keys:")
            lines.extend(f"--   {j}" for j in joins)
        return "\n".join(lines)
