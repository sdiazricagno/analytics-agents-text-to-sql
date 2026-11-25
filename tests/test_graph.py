"""End-to-end agent workflow with a scripted model."""

from __future__ import annotations

from t2s.config import Settings
from t2s.graph import AnalyticsAgents
from t2s.llm import ScriptedLLM


def _agents(settings: Settings, replies: list[str]) -> tuple[AnalyticsAgents, ScriptedLLM]:
    queue = list(replies)
    llm = ScriptedLLM(lambda _s, _u: queue.pop(0) if len(queue) > 1 else queue[0])
    return AnalyticsAgents(llm, settings), llm


def test_semantic_route_skips_the_model(settings: Settings) -> None:
    agents, llm = _agents(settings, ["SELECT 1"])
    state = agents.ask("Revenue by channel")
    assert state["route"] == "semantic" and not llm.calls and len(state["rows"]) == 3


def test_self_correction_loop(settings: Settings) -> None:
    bad = "```sql\nSELECT c.email FROM customers c\n```"
    good = "```sql\nSELECT COUNT(*) FROM customers WHERE loyalty_member\n```"
    agents, llm = _agents(settings, [bad, good])
    state = agents.ask("How many loyalty members do we have?")
    assert state["attempts"] == 2 and not state["refused"]
    assert "unknown columns" in llm.calls[1] and state["first_report"]["unknown_columns"] == ["c.email"]


def test_refuses_after_retry_budget(settings: Settings) -> None:
    agents, llm = _agents(settings, ["SELECT salary FROM employees"])
    state = agents.ask("What is the average salary of store employees?")
    assert state["refused"] and len(llm.calls) == settings.max_retries + 1 and not state["rows"]


def test_model_can_decline(settings: Settings) -> None:
    agents, _ = _agents(settings, ["CANNOT_ANSWER: no marketing data"])
    state = agents.ask("How much did the store spend on marketing per customer?")
    assert state["refused"] and "marketing" in state["answer"]


def test_write_requests_never_reach_the_model(settings: Settings) -> None:
    agents, llm = _agents(settings, ["DELETE FROM orders"])
    state = agents.ask("Delete all cancelled orders")
    assert state["refused"] and not llm.calls


def test_off_topic_questions_are_refused(settings: Settings) -> None:
    agents, llm = _agents(settings, ["SELECT 1"])
    assert agents.ask("What is the weather in Paris?")["refused"] and not llm.calls


def test_conversation_memory_for_follow_ups(settings: Settings) -> None:
    agents, _ = _agents(settings, ["SELECT 1"])
    agents.ask("Revenue by region in 2024", thread_id="t")
    state = agents.ask("what about by channel?", thread_id="t")
    assert state["route"] == "semantic" and state["columns"] == ["channel", "revenue"]
    assert "2024" in state["sql"] and len(state["history"]) == 2
