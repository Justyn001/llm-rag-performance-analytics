"""Tests for the JSON-based LLM-as-judge."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.evaluation.judge import (
    JudgeVerdict,
    LLMJudge,
    build_judge_prompt,
    parse_judge_response,
)


@dataclass
class FakeMessage:
    content: str


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return FakeMessage(self.responses.pop(0))


def valid_json() -> str:
    return (
        '{"correct": 1, "faithful": 1, "context_relevant": 1, '
        '"refusal_appropriate": 1, "reason": "supported"}'
    )


def test_parse_valid_json_response():
    verdict = parse_judge_response(valid_json())

    assert verdict.is_valid
    assert verdict.correct == 1
    assert verdict.faithful == 1
    assert verdict.context_relevant == 1
    assert verdict.refusal_appropriate == 1
    assert verdict.reason == "supported"
    assert verdict.error is None


def test_parse_fenced_json_and_think_markers():
    text = f"<think>reasoning</think>\n```json\n{valid_json()}\n```"
    verdict = parse_judge_response(text)

    assert verdict.is_valid


def test_parse_boolean_and_text_flags():
    verdict = parse_judge_response(
        '{"correct": true, "faithful": "yes", "context_relevant": 1, '
        '"refusal_appropriate": "1", "reason": "ok"}'
    )

    assert verdict.is_valid
    assert verdict.correct == 1
    assert verdict.faithful == 1


def test_parse_invalid_json_returns_error():
    verdict = parse_judge_response("I think the answer is fine.")

    assert not verdict.is_valid
    assert "JSON" in (verdict.error or "")


def test_parse_missing_metric_returns_error():
    verdict = parse_judge_response('{"correct": 1, "faithful": 1}')

    assert not verdict.is_valid
    assert "missing" in (verdict.error or "")


def test_build_judge_prompt_includes_control_status_and_contexts():
    prompt = build_judge_prompt(
        question="Who is the king of France?",
        answer="I don't have enough information.",
        contexts=["France is a republic."],
        answerable=False,
        references=[],
        gold_context=None,
    )

    assert "NOT answerable" in prompt
    assert "France is a republic." in prompt
    assert "I don't have enough information." in prompt


def test_judge_evaluate_returns_verdict():
    fake = FakeLLM([valid_json()])
    judge = LLMJudge(llm=fake)
    verdict = judge.evaluate(
        question="What is the capital of France?",
        answer="Paris",
        contexts=["Paris is the capital of France."],
        answerable=True,
        references=["Paris"],
        gold_context="Paris is the capital of France.",
    )

    assert verdict.is_valid
    assert fake.calls == 1


def test_judge_retries_once_on_invalid_json():
    fake = FakeLLM(["not json", valid_json()])
    judge = LLMJudge(llm=fake, max_attempts=2)
    verdict = judge.evaluate(
        question="What is the capital of France?",
        answer="Paris",
        contexts=["Paris is the capital of France."],
        answerable=True,
        references=["Paris"],
    )

    assert verdict.is_valid
    assert fake.calls == 2


def test_judge_returns_last_error_after_retries():
    fake = FakeLLM(["not json", "still not json"])
    judge = LLMJudge(llm=fake, max_attempts=2)
    verdict = judge.evaluate(
        question="What is the capital of France?",
        answer="Paris",
        contexts=[],
        answerable=True,
        references=["Paris"],
    )

    assert not verdict.is_valid
    assert fake.calls == 2
    assert isinstance(verdict, JudgeVerdict)


def test_judge_requires_model_or_injected_llm():
    with pytest.raises(ValueError, match="requires"):
        LLMJudge()
