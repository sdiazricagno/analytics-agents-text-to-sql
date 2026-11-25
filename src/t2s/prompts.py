"""Prompt templates for the Text-to-SQL agent."""

from __future__ import annotations

from t2s.retrieval import Example

SYSTEM = """You are a careful analytics engineer who writes DuckDB SQL.
Rules:
- Use only the tables and columns in the provided schema. Never invent tables, columns or values.
- Join tables only on the listed valid join keys.
- Follow the business definitions exactly when the question uses those metrics.
- Write a single read-only SELECT statement (CTEs allowed).
- If the schema cannot answer the question, reply exactly: CANNOT_ANSWER: <short reason>
Return only the SQL inside a ```sql code block."""


def build_user_prompt(
    question: str,
    schema: str,
    definitions: str,
    examples: list[Example],
    context: list[dict],
    feedback: str | None = None,
    previous_sql: str | None = None,
) -> str:
    """Assemble the grounded generation prompt."""
    parts = ["### Schema", schema, "", "### Business definitions", definitions, ""]
    if examples:
        parts.append("### Examples")
        for e in examples:
            parts += [f"Question: {e.question}", f"```sql\n{e.sql.strip()}\n```", ""]
    if context:
        parts.append("### Conversation so far")
        for turn in context[-2:]:
            parts.append(f"Question: {turn['question']}")
            if turn.get("sql"):
                parts.append(f"```sql\n{turn['sql']}\n```")
        parts.append("")
    parts += ["### Question", question]
    if feedback:
        parts += [
            "",
            "### Your previous SQL was rejected",
            f"```sql\n{previous_sql}\n```",
            feedback,
            "Fix it.",
        ]
    return "\n".join(parts)
