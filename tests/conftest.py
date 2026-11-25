"""Shared fixtures: a freshly generated warehouse and wired components."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from t2s.config import ROOT, Settings
from t2s.guardrails import SQLGuard, connect_read_only
from t2s.schema import Catalog
from t2s.semantic import SemanticLayer
from t2s.warehouse import build


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the seeded warehouse once per test session."""
    return build(tmp_path_factory.mktemp("wh") / "warehouse.duckdb")


@pytest.fixture(scope="session")
def settings(db_path: Path) -> Settings:
    """Settings pointing at the test warehouse."""
    return Settings(db_path=db_path, semantic_path=ROOT / "semantic" / "retail.yaml")


@pytest.fixture(scope="session")
def con(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Read-only connection to the test warehouse."""
    return connect_read_only(str(settings.db_path))


@pytest.fixture(scope="session")
def catalog(settings: Settings, con: duckdb.DuckDBPyConnection) -> Catalog:
    """Schema catalog for the test warehouse."""
    return Catalog.load(settings.semantic_path, con)


@pytest.fixture(scope="session")
def guard(catalog: Catalog, con: duckdb.DuckDBPyConnection) -> SQLGuard:
    """SQL guard with the default row limit."""
    return SQLGuard(catalog, con, max_rows=1000)


@pytest.fixture(scope="session")
def semantic(catalog: Catalog, con: duckdb.DuckDBPyConnection) -> SemanticLayer:
    """Semantic layer with indexed dimension values."""
    return SemanticLayer(catalog, con)
