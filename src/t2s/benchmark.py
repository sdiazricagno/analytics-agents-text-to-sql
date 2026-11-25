"""Golden-dataset benchmark harness for the analytics agents."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import yaml

from t2s.config import ROOT, Settings
from t2s.evaluation import difficulty, results_match
from t2s.graph import AnalyticsAgents, Options
from t2s.llm import OpenAICompatibleLLM

MODES = {
    "semantic_only": Options(use_router=True, use_llm=False),
    "llm_raw": Options(use_router=False, use_guard=False),
    "llm_guarded": Options(use_router=False),
    "full": Options(),
}
LEVELS = ["easy", "medium", "hard", "extra"]


def run(mode: str, golden: Path, out_dir: Path, limit: int | None = None) -> dict:
    """Run one configuration over the golden set and write per-item and summary JSON."""
    settings = Settings()
    options = MODES[mode]
    llm = OpenAICompatibleLLM(settings, cache_dir=out_dir / "cache") if options.use_llm else None
    agents = AnalyticsAgents(llm, settings, options)
    items = yaml.safe_load(golden.read_text())[:limit]
    records = []
    for i, item in enumerate(items):
        answerable = item.get("answerable", True)
        level, n_tables = difficulty(item["sql"]) if answerable else ("unanswerable", 0)
        start = time.perf_counter()
        state = agents.ask(item["question"], thread_id=f"{mode}-{item['id']}")
        latency = time.perf_counter() - start
        refused = bool(state.get("refused"))
        correct = False
        if answerable and not refused:
            gold = agents.con.execute(item["sql"]).fetchall()
            correct = results_match(gold, state["rows"])
        elif not answerable:
            correct = refused
        first = state.get("first_report") or {}
        rec = {
            "id": item["id"],
            "question": item["question"],
            "difficulty": level,
            "tables": n_tables,
            "answerable": answerable,
            "route": state.get("route"),
            "route_reason": state.get("route_reason"),
            "refused": refused,
            "correct": correct,
            "attempts": state.get("attempts", 0),
            "draft_hallucinated": bool(first.get("unknown_tables") or first.get("unknown_columns")),
            "draft_unknown": (first.get("unknown_tables") or []) + (first.get("unknown_columns") or []),
            "draft_valid": first.get("ok"),
            "sql": state.get("sql"),
            "latency_s": round(latency, 2),
        }
        records.append(rec)
        print(
            f"[{mode}] {i + 1}/{len(items)} {item['id']} {level:12s} {rec['route']:12s} correct={correct}",
            flush=True,
        )
    summary = summarise(records)
    summary.update({"mode": mode, "model": llm.name if llm else None, "n_items": len(records)})
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{mode}__{(llm.name if llm else 'none').replace(':', '-').replace('/', '-')}"
    (out_dir / f"{tag}.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, default=str)
    )
    return summary


def _rate(xs: list[bool]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def summarise(records: list[dict]) -> dict:
    """Aggregate accuracy, validity, hallucination and refusal metrics."""
    ans = [r for r in records if r["answerable"]]
    unans = [r for r in records if not r["answerable"]]
    by_level = {lvl: _rate([r["correct"] for r in ans if r["difficulty"] == lvl]) for lvl in LEVELS}
    counts = {lvl: sum(r["difficulty"] == lvl for r in ans) for lvl in LEVELS}
    by_route: dict[str, list[bool]] = defaultdict(list)
    for r in ans:
        if not r["refused"]:
            by_route[r["route"]].append(r["correct"])
    drafts = [r for r in records if r["route"] == "text_to_sql" and r["draft_valid"] is not None]
    multi = [r["correct"] for r in ans if r["tables"] >= 3]
    return {
        "execution_accuracy": _rate([r["correct"] for r in ans]),
        "accuracy_by_difficulty": by_level,
        "items_by_difficulty": counts,
        "multi_join_accuracy": _rate(multi),
        "multi_join_items": len(multi),
        "accuracy_when_answered_by_route": {k: _rate(v) for k, v in by_route.items()},
        "answered_by_route": {k: len(v) for k, v in by_route.items()},
        "answered_rate": _rate([not r["refused"] for r in ans]),
        "precision_when_answered": _rate([r["correct"] for r in ans if not r["refused"]]),
        "wrong_answer_rate": _rate([(not r["refused"]) and not r["correct"] for r in ans]),
        "refusal_accuracy_unanswerable": _rate([r["refused"] for r in unans]),
        "draft_hallucination_rate": _rate([r["draft_hallucinated"] for r in drafts]),
        "draft_validity_rate": _rate([bool(r["draft_valid"]) for r in drafts]),
        "mean_latency_s": round(sum(r["latency_s"] for r in records) / max(1, len(records)), 2),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), default="full")
    parser.add_argument("--golden", type=Path, default=ROOT / "benchmark" / "golden.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "benchmark" / "results")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.mode, args.golden, args.out, args.limit), indent=2))


if __name__ == "__main__":
    main()
