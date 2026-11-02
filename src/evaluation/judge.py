"""
LLM-as-judge for RAG answers.

The judge adds semantic dimensions that exact match cannot capture:

- correctness against the reference answer(s),
- faithfulness to the retrieved contexts,
- context relevance,
- refusal appropriateness for unanswerable controls.

The implementation deliberately does not import RAGAS.  RAGAS 0.4.x currently
pulls in an optional LangChain VertexAI module that is absent in this
environment; a small, explicit judge keeps the evaluation runnable and makes
the prompt/JSON contract auditable.  The judge is injectable, so tests and
offline experiments can use a fake model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config.models import ModelConfig
from src.rag.llm_provider import get_llm

JUDGE_SYSTEM_PROMPT = """You are a strict, careful evaluator of a \
retrieval-augmented question-answering system.

Evaluate the model answer using ONLY the information in this prompt. Do not use
outside knowledge to excuse an unsupported answer.

Scoring rules:
- correct: 1 if the answer is semantically equivalent to a reference answer for
  an answerable question. For an unanswerable question, 1 only if the model
  explicitly refuses to answer. Otherwise 0.
- faithful: 1 if every factual claim in the model answer is supported by the
  retrieved contexts. A clear refusal with no invented facts is faithful.
  Otherwise 0.
- context_relevant: 1 if at least one retrieved context contains information
  that is relevant to the question. Otherwise 0.
- refusal_appropriate: 1 if the model refuses exactly when the question is
  unanswerable. For an answerable question, refusing is a 0. For an
  unanswerable question, answering is a 0.

Return ONLY one valid JSON object with exactly these keys:
{"correct": 0 or 1, "faithful": 0 or 1, "context_relevant": 0 or 1, \
"refusal_appropriate": 0 or 1, "reason": "one short sentence"}

Do not wrap the JSON in markdown. Do not add any text before or after it.
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"</?think>", re.IGNORECASE)


@dataclass(frozen=True)
class JudgeVerdict:
    """Parsed verdict from one judge call."""

    correct: int | None = None
    faithful: int | None = None
    context_relevant: int | None = None
    refusal_appropriate: int | None = None
    reason: str = ""
    error: str | None = None

    @property
    def is_valid(self) -> bool:
        """True when all four metrics were parsed successfully."""
        return self.error is None and all(
            value in {0, 1}
            for value in (
                self.correct,
                self.faithful,
                self.context_relevant,
                self.refusal_appropriate,
            )
        )


def _strip_think(text: str) -> str:
    """Remove model control markers without trying to interpret reasoning."""
    return _THINK_RE.sub("", text).strip()


def _extract_json(text: str) -> str | None:
    """Return the most likely JSON object embedded in a model response."""
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return None


def _coerce_flag(value: Any) -> int | None:
    """Convert common LLM encodings of a binary judgement to 0/1."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value) if value in {0, 1} else None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "tak"}:
            return 1
        if normalized in {"0", "false", "no", "n", "nie"}:
            return 0
        try:
            numeric = float(normalized)
        except ValueError:
            return None
        return int(numeric) if numeric in {0, 1} else None
    return None


def parse_judge_response(text: str) -> JudgeVerdict:
    """Parse a judge response, returning a descriptive error on failure."""
    cleaned = _strip_think(text)
    candidate = _extract_json(cleaned)
    if candidate is None:
        return JudgeVerdict(error="judge response did not contain a JSON object")

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return JudgeVerdict(error=f"invalid judge JSON: {exc.msg}")

    if not isinstance(payload, dict):
        return JudgeVerdict(error="judge JSON must be an object")

    verdict = JudgeVerdict(
        correct=_coerce_flag(payload.get("correct")),
        faithful=_coerce_flag(payload.get("faithful")),
        context_relevant=_coerce_flag(payload.get("context_relevant")),
        refusal_appropriate=_coerce_flag(payload.get("refusal_appropriate")),
        reason=str(payload.get("reason", "")).strip()[:500],
    )
    if not verdict.is_valid:
        return JudgeVerdict(
            correct=verdict.correct,
            faithful=verdict.faithful,
            context_relevant=verdict.context_relevant,
            refusal_appropriate=verdict.refusal_appropriate,
            reason=verdict.reason,
            error="judge JSON is missing one or more 0/1 metrics",
        )
    return verdict


def _truncate(text: str | None, max_chars: int) -> str:
    if not text:
        return "(none)"
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}…"


def _format_contexts(contexts: list[str], max_chars_per_context: int = 1200) -> str:
    if not contexts:
        return "(no contexts retrieved)"
    return "\n\n".join(
        f"[Context {index}]\n{_truncate(context, max_chars_per_context)}"
        for index, context in enumerate(contexts, start=1)
    )


def build_judge_prompt(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    answerable: bool,
    references: list[str] | None = None,
    gold_context: str | None = None,
) -> str:
    """Build the deterministic human-readable part of the judge prompt."""
    if answerable:
        reference_text = (
            "\n".join(f"- {reference}" for reference in references or [])
            or "(reference answer unavailable)"
        )
        answerability = "ANSWERABLE from the corpus."
    else:
        reference_text = "(not applicable — this question is NOT answerable from the corpus)"
        answerability = "NOT answerable from the corpus. A refusal is correct."

    return f"""Question:
{question}

Question status: {answerability}

Reference answer(s):
{reference_text}

Gold evidence excerpt:
{_truncate(gold_context, 1200)}

Retrieved contexts:
{_format_contexts(contexts)}

Model answer:
{_truncate(answer, 1500)}

Return the JSON verdict now."""


class LLMJudge:
    """Small wrapper around a chat model that returns a structured verdict."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        *,
        llm: BaseChatModel | None = None,
        max_attempts: int = 2,
    ):
        if llm is None and model_config is None:
            msg = "LLMJudge requires either model_config or an injected llm"
            raise ValueError(msg)
        self.model_config = model_config
        self.llm = llm or get_llm(model_config)  # type: ignore[arg-type]
        self.max_attempts = max(1, max_attempts)

    def evaluate(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        answerable: bool,
        references: list[str] | None = None,
        gold_context: str | None = None,
    ) -> JudgeVerdict:
        """Ask the judge model for a verdict, retrying once on invalid JSON."""
        prompt = build_judge_prompt(
            question=question,
            answer=answer,
            contexts=contexts,
            answerable=answerable,
            references=references,
            gold_context=gold_context,
        )
        messages: list[Any] = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        last_verdict = JudgeVerdict(error="judge was not called")
        for attempt in range(self.max_attempts):
            response = self.llm.invoke(messages)
            text = _message_text(response)
            last_verdict = parse_judge_response(text)
            if last_verdict.is_valid:
                return last_verdict

            if attempt + 1 < self.max_attempts:
                messages = [
                    SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                    AIMessage(content=text),
                    HumanMessage(
                        content=(
                            "Your previous response was not valid JSON with all four "
                            "0/1 metrics. Return ONLY the required JSON object now."
                        )
                    ),
                ]

        return last_verdict


def _message_text(response: Any) -> str:
    """Extract text from a LangChain message or a plain string response."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def make_judge(model_config: ModelConfig, *, max_attempts: int = 2) -> LLMJudge:
    """Convenience factory used by the CLI and the runner."""
    return LLMJudge(model_config, max_attempts=max_attempts)
