"""LLM-as-judge grading for golden-set answer evaluation (Sprint J)."""
from __future__ import annotations

import json
import re

JUDGE_PROMPT = """You grade RAG answers for a technical documentation assistant.

Question:
{question}

Answer:
{answer}

Expected source documents (if any):
{expected_sources}

Reply with JSON only:
{{"pass": true|false, "score": 0.0-1.0, "reason": "one sentence"}}

Pass when the answer is grounded, relevant, and not a refusal unless the question is truly unanswerable from the sources."""


def parse_judge_response(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"pass": False, "score": 0.0, "reason": "Unparseable judge output"}
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return {"pass": False, "score": 0.0, "reason": "Invalid judge JSON"}
    return {
        "pass": bool(payload.get("pass", False)),
        "score": float(payload.get("score", 0.0)),
        "reason": str(payload.get("reason", "")),
    }


def grade_answer(
    llm,
    *,
    question: str,
    answer: str,
    expected_sources: list[str] | None = None,
) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        expected_sources=", ".join(expected_sources or []) or "(none listed)",
    )
    raw = llm.generate(prompt)
    result = parse_judge_response(raw)
    result["raw"] = raw
    return result
