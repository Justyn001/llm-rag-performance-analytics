"""
Descriptive statistics for repeated performance measurements.

The functions are deliberately dependency-free (Python standard library only)
and tolerant of missing values.  Each measurement is one question x repetition;
``None`` values (for example a failed request) are skipped rather than treated
as zero.

Confidence intervals use the normal approximation ``mean +/- z * s / sqrt(n)``.
With the small number of repetitions used in practice this is an
approximation, but it is explicit and reproducible.  The interval is not
reported for ``n < 2``.
"""

from __future__ import annotations

import math
from statistics import NormalDist, mean, median, stdev
from typing import Any

PERFORMANCE_FIELDS = (
    "retrieval_time_ms",
    "generation_time_ms",
    "total_time_ms",
    "tokens_out",
    "tokens_per_second",
    "retry_backoff_time_ms",
)


def describe(
    values: list[float | int | None],
    *,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Return count, mean, median, sample stdev, min/max and a normal-approx CI."""
    present = [float(value) for value in values if value is not None]
    result: dict[str, Any] = {
        "n": len(present),
        "mean": None,
        "median": None,
        "stdev": None,
        "ci_low": None,
        "ci_high": None,
        "min": None,
        "max": None,
    }
    if not present:
        return result

    result["mean"] = mean(present)
    result["median"] = median(present)
    result["min"] = min(present)
    result["max"] = max(present)

    if len(present) < 2:
        return result

    sample_stdev = stdev(present)
    result["stdev"] = sample_stdev
    z_value = NormalDist().inv_cdf(1 - (1 - confidence_level) / 2)
    half_width = z_value * sample_stdev / math.sqrt(len(present))
    result["ci_low"] = result["mean"] - half_width
    result["ci_high"] = result["mean"] + half_width
    return result


def performance_summary(
    records: list[dict[str, Any]],
    *,
    confidence_level: float = 0.95,
) -> dict[str, dict[str, Any]]:
    """Describe every performance field across completed records."""
    completed = [record for record in records if not record.get("error")]
    return {
        field: describe(
            [record.get(field) for record in completed],
            confidence_level=confidence_level,
        )
        for field in PERFORMANCE_FIELDS
    }


def flatten_performance(
    performance: dict[str, dict[str, Any]],
) -> dict[str, float | int | None]:
    """Flatten nested performance statistics into summary-friendly keys."""
    flat: dict[str, float | int | None] = {}
    for field, values in performance.items():
        for statistic in ("mean", "median", "stdev", "ci_low", "ci_high"):
            flat[f"{statistic}_{field}"] = values.get(statistic)
    return flat
