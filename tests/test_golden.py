"""Integrity of the golden dataset and few-shot pool."""

from __future__ import annotations

import yaml

from t2s.config import ROOT
from t2s.guardrails import SQLGuard
from t2s.retrieval import FewShotStore

GOLDEN = yaml.safe_load((ROOT / "benchmark" / "golden.yaml").read_text())


def test_gold_sql_is_valid_and_non_empty(guard: SQLGuard, con) -> None:  # noqa: ANN001
    for item in GOLDEN:
        if item.get("answerable", True):
            report = guard.check(item["sql"])
            assert report.ok, (item["id"], report.errors)
            assert con.execute(report.sql).fetchall(), item["id"]


def test_ids_unique_and_few_shot_disjoint() -> None:
    assert len({i["id"] for i in GOLDEN}) == len(GOLDEN)
    shots = {e.question.lower() for e in FewShotStore(ROOT / "semantic" / "examples.yaml").examples}
    assert not shots & {i["question"].lower() for i in GOLDEN}
