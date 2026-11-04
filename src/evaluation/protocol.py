"""
Measurement protocol for reproducible RAG performance runs (P0.5).

The protocol fixes every knob that could otherwise differ between models:

- a number of discarded warm-up requests (model load, caches, TLS, GPU clocks),
- a number of measured repetitions,
- one ``max_tokens`` value for every provider,
- one ``num_ctx`` value for local Ollama models,
- a confidence level used by the summary statistics.

All requests are executed sequentially.  The protocol object is serialised
into the run metadata so a stored result can always be traced back to the
settings that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_WARMUP_RUNS = 2
DEFAULT_REPETITIONS = 3
DEFAULT_MAX_TOKENS = 1024
DEFAULT_NUM_CTX = 8192
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True)
class MeasurementProtocol:
    """Fixed settings shared by every model in a measurement campaign."""

    warmup_runs: int = DEFAULT_WARMUP_RUNS
    repetitions: int = DEFAULT_REPETITIONS
    max_tokens: int = DEFAULT_MAX_TOKENS
    num_ctx: int = DEFAULT_NUM_CTX
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL

    def __post_init__(self) -> None:
        if self.warmup_runs < 0:
            msg = "warmup_runs must be at least 0"
            raise ValueError(msg)
        if self.repetitions < 1:
            msg = "repetitions must be at least 1"
            raise ValueError(msg)
        if self.max_tokens < 1:
            msg = "max_tokens must be at least 1"
            raise ValueError(msg)
        if self.num_ctx < 1:
            msg = "num_ctx must be at least 1"
            raise ValueError(msg)
        if not 0 < self.confidence_level < 1:
            msg = "confidence_level must be between 0 and 1"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description for run metadata."""
        data = asdict(self)
        data["execution_mode"] = "sequential"
        return data


DEFAULT_PROTOCOL = MeasurementProtocol()
