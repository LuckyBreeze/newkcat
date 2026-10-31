"""Regression metrics in log10(kcat / s^-1), with explicit undefined values."""

from __future__ import annotations

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Rank finite values with average ranks for ties; no scipy dependency."""
    order = np.argsort(values, kind="stable")
    ranked = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranked[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranked


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2:
        return None
    ac, bc = a - a.mean(), b - b.mean()
    denominator = float(np.linalg.norm(ac) * np.linalg.norm(bc))
    if denominator == 0:
        return None
    return float(np.clip(np.dot(ac, bc) / denominator, -1.0, 1.0))


def regression_metrics(target, prediction) -> dict[str, float | int | None]:
    """Refuse implicit broadcasting, NaNs, and empty result arrays.

    Undefined R2/correlations are JSON-compatible ``None`` instead of misleading
    zeros. Predictions and targets must both be one-dimensional.
    """
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape:
        raise ValueError(f"Expected equal 1D shapes, got target={y.shape}, prediction={p.shape}")
    if y.size == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require finite targets and predictions")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            residual = p - y
            mse = float(np.mean(residual**2))
            total = float(np.sum((y - y.mean()) ** 2))
            return {
                "n": int(y.size),
                "mse": mse,
                "rmse": float(np.sqrt(mse)),
                "mae": float(np.mean(np.abs(residual))),
                "r2": float(1 - np.sum(residual**2) / total) if total > 0 else None,
                "pearson": _correlation(y, p),
                "spearman": _correlation(_average_ranks(y), _average_ranks(p)),
            }
    except FloatingPointError as error:
        raise ValueError("Regression metrics overflowed; check target and prediction scale") from error
