"""LangGraph multi-agent workflow: intent, semantic router, Text-to-SQL, guardrails, execution, answer."""

from __future__ import annotations

import operator
import re
import threading
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypedDict

import duckdb
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from t2s.config import Settings
from t2s.guardrails import SQLGuard, connect_read_only, extract_sql
from t2s.llm import LLM
from t2s.prompts import SYSTEM, build_user_prompt
from t2s.retrieval import FewShotStore, SchemaRetriever
from t2s.schema import Catalog
from t2s.semantic import FOLLOW_UP, SemanticLayer, SemanticQuery

GREETING = re.compile(r"^\s*(hi|hello|hey|help|thanks|thank you)\b", re.I)
WRITE_INTENT = re.compile(r"\b(delete|drop|truncate|insert|update|alter|create table|grant)\b", re.I)


class AgentState(TypedDict, total=False):
    """Shared state passed between agents."""

    question: str
    history: Annotated[list[dict], operator.add]
    route: str
    route_reason: str
    semantic_query: dict | None
    tables: list[str]
    sql: str | None
    raw_sql: str | None
    first_report: dict | None
    attempts: int
    feedback: str | None
    columns: list[str]
    rows: list[tuple]
    answer: str
    chart: dict | None
    refused: bool
    error: str | None


@dataclass
class Options:
    """Feature switches used for ablations."""

    use_router: bool = True
    use_llm: bool = True
    use_guard: bool = True


class AnalyticsAgents:
    """Builds and runs the conversational analytics graph."""

    def __init__(
        self,
        llm: LLM | None,
        settings: Settings | None = None,
        options: Options | None = None,
        checkpointer: Any | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.options = options or Options()
        self.llm = llm
        self.con = connect_read_only(str(self.settings.db_path))
        self.catalog = Catalog.load(self.settings.semantic_path, self.con)
        self.semantic = SemanticLayer(self.catalog, self.con)
        self.retriever = SchemaRetriever(self.catalog)
        self.examples = FewShotStore(self.settings.semantic_path.parent / "examples.yaml")
        self.guard = SQLGuard(self.catalog, self.con, self.settings.max_rows)
        self.graph = self._build().compile(checkpointer=checkpointer or InMemorySaver())

    def _build(self) -> StateGraph:
        g = StateGraph(AgentState)
        g.add_node("intent", self.intent_agent)
        g.add_node("semantic_router", self.semantic_router)
        g.add_node("text_to_sql", self.text_to_sql_agent)
        g.add_node("guardrails", self.guardrail_agent)
        g.add_node("execute", self.executor)
        g.add_node("answer", self.answer_agent)
        g.add_node("refuse", self.refuse)
        g.add_edge(START, "intent")
        g.add_conditional_edges(
            "intent", lambda s: s["route"], {"refuse": "refuse", "route": "semantic_router"}
        )
        g.add_conditional_edges(
            "semantic_router",
            lambda s: s["route"],
            {"semantic": "guardrails", "text_to_sql": "text_to_sql", "refuse": "refuse"},
        )
        g.add_conditional_edges(
            "text_to_sql",
            lambda s: "refuse" if s.get("refused") else "guardrails",
            {"refuse": "refuse", "guardrails": "guardrails"},
        )
        g.add_conditional_edges(
            "guardrails",
            self._after_guard,
            {"execute": "execute", "retry": "text_to_sql", "refuse": "refuse"},
        )
        g.add_conditional_edges(
            "execute",
            lambda s: "refuse" if s.get("error") else "answer",
            {"refuse": "refuse", "answer": "answer"},
        )
        g.add_edge("answer", END)
        g.add_edge("refuse", END)
        return g

    def intent_agent(self, state: AgentState) -> dict:
        """Classify the request and refuse writes or questions with no schema grounding."""
        q = state["question"]
        if WRITE_INTENT.search(q):
            return {
                "route": "refuse",
                "route_reason": "write operations are not permitted; this assistant is read-only",
            }
        if GREETING.match(q) and len(q.split()) < 6:
            return {
                "route": "refuse",
                "route_reason": "ask a question about orders, revenue, customers, products or returns",
            }
        follow_up = bool(state.get("history")) and bool(FOLLOW_UP.match(q.lower().strip()))
        if self.retriever.grounding(q) < 0.5 and not follow_up:
            return {"route": "refuse", "route_reason": "the question does not map to data in this warehouse"}
        return {"route": "route"}

    def semantic_router(self, state: AgentState) -> dict:
        """Answer through governed metrics when the whole question is understood, else defer."""
        previous = None
        history = state.get("history") or []
        if history and history[-1].get("semantic_query"):
            previous = SemanticQuery(
                **{
                    k: tuple(map(tuple, v)) if k == "filters" else (tuple(v) if k == "dimensions" else v)
                    for k, v in history[-1]["semantic_query"].items()
                }
            )
        if self.options.use_router:
            decision = self.semantic.parse(state["question"], previous)
            if decision.query is not None:
                return {
                    "route": "semantic",
                    "route_reason": decision.reason,
                    "semantic_query": asdict(decision.query),
                    "sql": self.semantic.compile(decision.query),
                }
            reason = decision.reason
        else:
            reason = "router disabled"
        if not self.options.use_llm or self.llm is None:
            return {"route": "refuse", "route_reason": f"no governed metric covers this question ({reason})"}
        return {"route": "text_to_sql", "route_reason": reason, "attempts": 0, "semantic_query": None}

    def text_to_sql_agent(self, state: AgentState) -> dict:
        """Generate schema-grounded SQL from retrieved tables, definitions and examples."""
        assert self.llm is not None
        q = state["question"]
        context = [t for t in (state.get("history") or []) if t.get("sql")]
        retrieval_q = q if not context else f"{context[-1]['question']} {q}"
        tables = self.retriever.retrieve(retrieval_q) or list(self.catalog.tables)
        prompt = build_user_prompt(
            q,
            self.catalog.describe(tables),
            self.semantic.definitions(),
            self.examples.select(q),
            context,
            feedback=state.get("feedback"),
            previous_sql=state.get("sql"),
        )
        reply = self.llm.complete(SYSTEM, prompt)
        attempts = state.get("attempts", 0) + 1
        if "CANNOT_ANSWER" in reply.upper() and "SELECT" not in reply.upper():
            reason = reply.split(":", 1)[-1].strip() or "not answerable from the schema"
            return {"refused": True, "route_reason": f"model declined: {reason}", "attempts": attempts}
        sql = extract_sql(reply)
        update: dict = {"sql": sql, "tables": tables, "attempts": attempts}
        if attempts == 1:
            update["raw_sql"] = sql
        return update

    def guardrail_agent(self, state: AgentState) -> dict:
        """Validate the candidate SQL and prepare correction feedback."""
        sql = state.get("sql") or ""
        report = self.guard.check(sql)
        update: dict = {}
        if state.get("route") == "text_to_sql" and state.get("attempts", 0) == 1:
            update["first_report"] = asdict(report)
        if not self.options.use_guard:
            update["feedback"] = None
            return update
        if report.ok:
            update.update({"sql": report.sql, "feedback": None})
        else:
            update["feedback"] = report.feedback()
        return update

    def _after_guard(self, state: AgentState) -> str:
        if not self.options.use_guard or not state.get("feedback"):
            return "execute"
        if state.get("route") == "text_to_sql" and state.get("attempts", 0) <= self.settings.max_retries:
            return "retry"
        return "refuse"

    def executor(self, state: AgentState) -> dict:
        """Run the SQL on a read-only connection with a timeout."""
        cur = self.con.cursor()
        timer = threading.Timer(self.settings.query_timeout_s, cur.interrupt)
        timer.start()
        try:
            rel = cur.execute(state["sql"] or "")
            rows = rel.fetchmany(self.settings.max_rows)
            columns = [d[0] for d in rel.description]
        except duckdb.Error as e:
            return {"error": f"execution failed: {str(e).splitlines()[0]}"}
        finally:
            timer.cancel()
            cur.close()
        return {"columns": columns, "rows": rows, "error": None}

    def answer_agent(self, state: AgentState) -> dict:
        """Summarise the result and propose a chart."""
        cols, rows = state["columns"], state["rows"]
        if not rows:
            text = "The query returned no rows."
        elif len(rows) == 1 and len(cols) == 1:
            text = f"{cols[0]}: {_fmt(rows[0][0])}"
        else:
            head = ", ".join(f"{c}={_fmt(v)}" for c, v in zip(cols, rows[0], strict=True))
            text = f"{len(rows)} rows. First row: {head}."
        chart = None
        if (
            2 <= len(rows) <= 60
            and len(cols) == 2
            and isinstance(rows[0][1], (int, float))
            or (rows and len(cols) == 2 and hasattr(rows[0][1], "as_integer_ratio"))
        ):
            kind = "line" if cols[0] in {"month", "quarter", "year"} else "bar"
            chart = {"type": kind, "x": cols[0], "y": cols[1]}
        source = "governed semantic layer" if state.get("route") == "semantic" else "Text-to-SQL"
        turn = {
            "question": state["question"],
            "sql": state.get("sql"),
            "semantic_query": state.get("semantic_query"),
        }
        return {"answer": f"{text} (via {source})", "chart": chart, "refused": False, "history": [turn]}

    def refuse(self, state: AgentState) -> dict:
        """Decline instead of guessing."""
        reason = state.get("error") or state.get("route_reason") or "could not produce a grounded query"
        if state.get("feedback"):
            reason = f"no valid SQL after {state.get('attempts', 0)} attempts: {state['feedback']}"
        return {
            "refused": True,
            "answer": f"I can't answer that reliably: {reason}",
            "history": [{"question": state["question"]}],
        }

    def ask(self, question: str, thread_id: str = "default") -> AgentState:
        """Run one conversational turn."""
        reset: AgentState = {
            "question": question,
            "route": "",
            "route_reason": "",
            "semantic_query": None,
            "tables": [],
            "sql": None,
            "raw_sql": None,
            "first_report": None,
            "attempts": 0,
            "feedback": None,
            "columns": [],
            "rows": [],
            "answer": "",
            "chart": None,
            "refused": False,
            "error": None,
        }
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
        return self.graph.invoke(reset, config)


def _fmt(v: object) -> str:
    if isinstance(v, float) or hasattr(v, "as_integer_ratio") and not isinstance(v, int):
        return f"{float(v):,.2f}"
    return str(v)
