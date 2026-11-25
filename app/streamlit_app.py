"""Streamlit chat front end for the analytics agents."""

from __future__ import annotations

import uuid

import pandas as pd
import streamlit as st

from t2s.graph import AnalyticsAgents, Options
from t2s.llm import OpenAICompatibleLLM

st.set_page_config(page_title="Analytics Agents", layout="wide")
st.title("Conversational analytics: semantic layer + guarded Text-to-SQL")

semantic_only = st.sidebar.toggle("Governed metrics only (no LLM)", value=False)


@st.cache_resource
def load_agents(semantic_only: bool) -> AnalyticsAgents:
    """Build the agent graph once per configuration."""
    llm = None if semantic_only else OpenAICompatibleLLM()
    return AnalyticsAgents(llm, options=Options(use_llm=not semantic_only))


agents = load_agents(semantic_only)
st.session_state.setdefault("thread", uuid.uuid4().hex)
st.session_state.setdefault("messages", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("e.g. Revenue by region in 2024, then: what about by channel?"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    state = agents.ask(question, st.session_state.thread)
    with st.chat_message("assistant"):
        st.markdown(state["answer"])
        st.caption(f"route: {state['route']} | {state['route_reason']}")
        if state.get("sql"):
            st.code(state["sql"], language="sql")
        if state.get("rows"):
            df = pd.DataFrame(state["rows"], columns=state["columns"])
            st.dataframe(df, use_container_width=True)
            chart = state.get("chart")
            if chart:
                data = df.set_index(chart["x"])[chart["y"]].astype(float)
                (st.line_chart if chart["type"] == "line" else st.bar_chart)(data)
    st.session_state.messages.append({"role": "assistant", "content": state["answer"]})
