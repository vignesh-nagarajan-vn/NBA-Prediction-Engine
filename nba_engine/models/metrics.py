"""Probability scoring and calibration metrics."""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def accuracy(y, p) -> float:
    return float(np.mean((np.asarray(p) > 0.5) == (np.asarray(y) > 0.5)))


def fit_logistic(X, y, l2: float = 1e-6, iters: int = 50) -> np.ndarray:
    """Logistic regression by Newton's method. X should include an intercept column."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def calibration(y, p) -> tuple[float, float]:
    """(intercept, slope) of y ~ a + b * logit(p). Perfect calibration = (0, 1)."""
    lp = logit(p)
    a, b = fit_logistic(np.column_stack([np.ones_like(lp), lp]), y)
    return float(a), float(b)


def reliability(y, p, bins: int = 10) -> list[tuple[float, float, int]]:
    """(mean predicted, observed rate, n) per probability bin."""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            out.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return out


def summarize(y, p) -> dict:
    a, b = calibration(y, p)
    return {"n": int(len(y)), "log_loss": log_loss(y, p), "brier": brier(y, p),
            "accuracy": accuracy(y, p), "calib_intercept": a, "calib_slope": b}
