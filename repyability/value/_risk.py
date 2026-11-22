"""Risk measures on priced outcomes: how bad a bad run is.

An outcome here is a loss per run (a downtime in hours or its cost, a
cost), larger being worse. ``quantile`` is the value-at-risk at a level
(the smallest outcome with at least that share of the runs at or below
it), ``tail_mean`` the mean of the worst share beyond it (the CVaR), and
each has an upper confidence bound, so that a design is said to keep a
bad run under a limit only when the bound is under it:

- the quantile's bound is an order statistic, by the binomial method
  (distribution-free and exact for a continuous outcome): the ``k``-th
  smallest of ``n`` runs is above the ``level`` quantile with chance
  ``P(Binomial(n, level) <= k - 1)``, and ``k`` is the smallest making that
  at least ``confidence``;
- the tail mean's bound is the normal one, from the standard error of
  the tail mean (the variance of ``max(X - q, 0) / (1 - level)``, which
  is the tail mean's influence beyond the quantile), so it is
  approximate, and says so.
"""

import math
from dataclasses import dataclass

import numpy as np

#: The risk measures a target takes.
MEASURES = ("var", "cvar")


def quantile(ordered: np.ndarray, level: float) -> float:
    """The ``level`` quantile of the sorted outcomes ``ordered``: the
    smallest with at least a share ``level`` of them at or below it (the
    ``ceil(level n)``-th, at least the first)."""
    k = max(1, math.ceil(round(level * ordered.size, 9)))
    return float(ordered[k - 1])


def quantile_bound(ordered: np.ndarray, level: float, confidence: float):
    """An upper confidence bound on the ``level`` quantile of the
    outcomes, at ``confidence``: an order statistic (the binomial method),
    ``inf`` when the runs are too few to give one."""
    from scipy.stats import binom

    n = ordered.size
    k = int(binom.ppf(confidence, n, level)) + 1
    while k > 1 and binom.cdf(k - 2, n, level) >= confidence:
        k -= 1
    while k <= n and binom.cdf(k - 1, n, level) < confidence:
        k += 1
    return float(ordered[k - 1]) if k <= n else math.inf


def tail_mean(ordered: np.ndarray, level: float) -> float:
    """The mean of the worst share ``1 - level`` of the outcomes (at
    least one run): the conditional value-at-risk."""
    worst = max(1, math.ceil(round((1.0 - level) * ordered.size, 9)))
    return float(ordered[-worst:].mean())


def tail_mean_bound(ordered: np.ndarray, level: float, confidence: float):
    """An approximate upper confidence bound on the tail mean: the
    estimate plus the normal quantile at ``confidence`` times its
    standard error (see the module's notes)."""
    from scipy.stats import norm

    n = ordered.size
    q = quantile(ordered, level)
    excess = np.maximum(ordered - q, 0.0) / (1.0 - level)
    se = float(excess.std(ddof=1)) / math.sqrt(n) if n > 1 else math.inf
    return tail_mean(ordered, level) + float(norm.ppf(confidence)) * se


@dataclass(frozen=True)
class Risk:
    """A risk measure of some outcomes: the ``estimate`` (the quantile,
    or the tail mean, at ``level``), its upper ``bound`` at ``confidence``
    and how the bound is found (``bound_method``)."""

    measure: str
    level: float
    estimate: float
    bound: float
    confidence: float
    bound_method: str


def risk(outcomes, level: float, measure: str, confidence: float) -> Risk:
    """``measure`` (``"var"`` or ``"cvar"``) of ``outcomes`` at
    ``level``, with its upper bound at ``confidence``."""
    ordered = np.sort(np.asarray(outcomes, float))
    if measure == "var":
        return Risk(
            measure,
            level,
            quantile(ordered, level),
            quantile_bound(ordered, level, confidence),
            confidence,
            "order statistic (binomial method, exact)",
        )
    return Risk(
        measure,
        level,
        tail_mean(ordered, level),
        tail_mean_bound(ordered, level, confidence),
        confidence,
        "normal (approximate)",
    )
