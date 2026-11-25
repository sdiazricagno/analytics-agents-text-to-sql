"""Render benchmark JSON into a markdown table and charts under docs/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from t2s.config import ROOT  # noqa: E402

LABELS = {
    "semantic_only": "Semantic layer only",
    "llm_raw": "LLM, no guardrails",
    "llm_guarded": "LLM + guardrails",
    "full": "Router + LLM + guardrails",
}
ORDER = ["semantic_only", "llm_raw", "llm_guarded", "full"]
LEVELS = ["easy", "medium", "hard", "extra"]


def load(results_dir: Path) -> list[dict]:
    """Load summaries sorted by model and configuration."""
    out = [json.loads(p.read_text())["summary"] for p in sorted(results_dir.glob("*.json"))]
    return sorted(out, key=lambda s: (s["model"] or "", ORDER.index(s["mode"])))


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def table(summaries: list[dict]) -> str:
    """Markdown results table."""
    head = (
        "| Configuration | Model | Exec. accuracy | Easy | Medium | Hard | Extra | Multi-join (3+ tables) "
        "| Wrong answers returned | Refused unanswerable | Drafts with hallucinated identifiers |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for s in summaries:
        lv = s["accuracy_by_difficulty"]
        rows.append(
            f"| {LABELS[s['mode']]} | {s['model'] or 'none'} | **{_pct(s['execution_accuracy'])}** | "
            + " | ".join(_pct(lv[k]) for k in LEVELS)
            + f" | {_pct(s['multi_join_accuracy'])} | {_pct(s['wrong_answer_rate'])} | "
            f"{_pct(s['refusal_accuracy_unanswerable'])} | {_pct(s['draft_hallucination_rate'])} |"
        )
    return "\n".join([head, *rows])


def charts(summaries: list[dict], docs: Path) -> list[Path]:
    """Accuracy-by-difficulty and safety charts."""
    docs.mkdir(parents=True, exist_ok=True)
    names = [f"{LABELS[s['mode']]}\n({s['model'] or 'no LLM'})" for s in summaries]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(summaries)
    for i, (s, name) in enumerate(zip(summaries, names, strict=True)):
        vals = [100 * (s["accuracy_by_difficulty"][k] or 0) for k in LEVELS] + [100 * s["execution_accuracy"]]
        xs = [j + i * width for j in range(len(vals))]
        ax.bar(xs, vals, width, label=name.replace("\n", " "))
    counts = summaries[0]["items_by_difficulty"]
    ticks = [f"{k}\n(n={counts[k]})" for k in LEVELS] + [f"overall\n(n={sum(counts.values())})"]
    ax.set_xticks([j + width * (len(summaries) - 1) / 2 for j in range(len(ticks))], ticks)
    ax.set_ylabel("Execution accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Golden benchmark: execution accuracy by difficulty")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    acc = docs / "benchmark_accuracy.png"
    fig.savefig(acc, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    metrics = [
        ("execution_accuracy", "Correct answers"),
        ("wrong_answer_rate", "Wrong answers returned"),
        ("refusal_accuracy_unanswerable", "Unanswerable refused"),
    ]
    width = 0.8 / len(summaries)
    for i, (s, name) in enumerate(zip(summaries, names, strict=True)):
        vals = [100 * (s[m] or 0) for m, _ in metrics]
        ax.bar([j + i * width for j in range(len(metrics))], vals, width, label=name.replace("\n", " "))
    ax.set_xticks(
        [j + width * (len(summaries) - 1) / 2 for j in range(len(metrics))], [m[1] for m in metrics]
    )
    ax.set_ylabel("% of questions")
    ax.set_ylim(0, 105)
    ax.set_title("Answer, abstain or fail: what each configuration does with the golden set")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    safety = docs / "benchmark_safety.png"
    fig.savefig(safety, dpi=150)
    plt.close(fig)
    return [acc, safety]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "benchmark" / "results")
    parser.add_argument("--docs", type=Path, default=ROOT / "docs")
    args = parser.parse_args()
    summaries = load(args.results)
    md = table(summaries)
    (args.docs / "benchmark_results.md").write_text(md + "\n")
    print(md)
    for p in charts(summaries, args.docs):
        print(p)


if __name__ == "__main__":
    main()
