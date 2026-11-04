"""Tests for the P0.5 measurement protocol and descriptive statistics."""

from __future__ import annotations

import pytest

from src.evaluation.protocol import MeasurementProtocol
from src.evaluation.stats import describe, flatten_performance, performance_summary


def test_protocol_defaults_are_fixed_and_serialisable():
    protocol = MeasurementProtocol()
    assert protocol.warmup_runs == 2
    assert protocol.repetitions == 3
    assert protocol.max_tokens == 1024
    assert protocol.num_ctx == 8192
    data = protocol.to_dict()
    assert data["execution_mode"] == "sequential"
    assert data["confidence_level"] == pytest.approx(0.95)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"warmup_runs": -1}, "warmup_runs"),
        ({"repetitions": 0}, "repetitions"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"num_ctx": 0}, "num_ctx"),
        ({"confidence_level": 1.0}, "confidence_level"),
    ],
)
def test_protocol_rejects_invalid_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MeasurementProtocol(**kwargs)


def test_describe_handles_missing_and_single_values():
    empty = describe([None, None])
    assert empty["n"] == 0
    assert empty["mean"] is None
    assert empty["ci_low"] is None

    single = describe([10.0])
    assert single["n"] == 1
    assert single["mean"] == 10.0
    assert single["median"] == 10.0
    assert single["stdev"] is None
    assert single["ci_low"] is None


def test_describe_computes_mean_median_stdev_and_ci():
    result = describe([10.0, 12.0, 14.0])
    assert result["n"] == 3
    assert result["mean"] == 12.0
    assert result["median"] == 12.0
    assert result["stdev"] == pytest.approx(2.0)
    assert result["ci_low"] == pytest.approx(9.73683, abs=1e-4)
    assert result["ci_high"] == pytest.approx(14.26317, abs=1e-4)
    assert result["min"] == 10.0
    assert result["max"] == 14.0


def test_performance_summary_skips_errors_and_flattens():
    records = [
        {"generation_time_ms": 10.0, "retrieval_time_ms": 1.0, "error": None},
        {"generation_time_ms": 20.0, "retrieval_time_ms": 1.0, "error": None},
        {"generation_time_ms": 999.0, "retrieval_time_ms": 1.0, "error": "boom"},
    ]
    performance = performance_summary(records)
    assert performance["generation_time_ms"]["n"] == 2
    assert performance["generation_time_ms"]["mean"] == 15.0
    assert performance["retrieval_time_ms"]["mean"] == 1.0

    flat = flatten_performance(performance)
    assert flat["mean_generation_time_ms"] == 15.0
    assert flat["median_generation_time_ms"] == 15.0
    assert flat["stdev_generation_time_ms"] == pytest.approx(7.0711, abs=1e-4)
    assert "mean_total_time_ms" in flat
