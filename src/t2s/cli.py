"""Terminal chat over the warehouse."""

from __future__ import annotations

import argparse
import uuid

from t2s.graph import AnalyticsAgents, Options
from t2s.llm import OpenAICompatibleLLM


def main() -> None:
    """Run an interactive session; type 'sql' to show the last query and 'exit' to quit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-only", action="store_true", help="answer only through governed metrics")
    args = parser.parse_args()
    llm = None if args.semantic_only else OpenAICompatibleLLM()
    agents = AnalyticsAgents(llm, options=Options(use_llm=not args.semantic_only))
    thread = uuid.uuid4().hex
    last_sql = None
    print("Ask about revenue, orders, customers, products, payments or returns. 'exit' quits.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "sql":
            print(last_sql or "no query yet")
            continue
        state = agents.ask(q, thread)
        last_sql = state.get("sql")
        print(state["answer"])
        if state.get("rows"):
            print(" | ".join(state["columns"]))
            for row in state["rows"][:15]:
                print(" | ".join(str(v) for v in row))


if __name__ == "__main__":
    main()
