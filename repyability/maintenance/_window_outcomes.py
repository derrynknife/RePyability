"""A part's cost in each of a run of consecutive windows from its state
now, and how often each outcome happens in each (UC-9 of this package's
``README.md``).

The windows are ``[0, e_1), [e_1, e_2), ...``, from the part's state now.
They are not independent: a failure in the first renews the part, and
changes what can happen in the next. So each window's distribution comes
from the process through every earlier window, not from a part started
afresh at the window's start.

**A window that starts later** (at ``e``). Under run to failure and age
replacement, what happens in ``[e, e + w)`` depends on the past only
through the first event at or after ``e``: once it has happened, a new
part is in and the rest is the cycles of new parts, as from a new part
in ``cost_distribution``'s formulas. The first event is made by the part
in service at ``e``, put in at the last renewal before it:

- the part in now, never renewed (while ``e <= tau``): it fails with
  ``S(e) - S(min(e + x, tau))``, or is replaced at ``tau``;
- the last of the planned chain from now, ``tau + k T`` (each planned
  replacement in turn, no failure yet), a chance ``p_1 p^k``: it fails
  with ``R(u) - R(min(u + x, T))``, ``u`` its age at ``e``, or is replaced
  at ``T - u``;
- a renewal after at least one failure, at ``s``: by the renewal measure
  ``M_c`` (the expected number of such renewals by ``s``, the same sums of
  the cycles' totals, by order, as the counts), the same with ``u = e -
  s``. Its failures' distribution is a correlation of ``M_c``'s cells
  with ``R`` (by FFT); its planned replacements', ``R(T) (M_c(e - T + x) -
  M_c(e - T))``.

``cost_distribution``'s formulas then give the window's counts, the
first cycle's distribution being this first event's. A window from 0 is
``cost_distribution``'s window. Under block replacement the planned
replacements are on fixed times, so each window's are known; its
failures are the stretch run to failure from the last planned
replacement before the window (or from now), seen from ``e`` on as
above, then the stretches after each planned replacement in it, from
new, independent.

The windows' means add up to the whole span's (expectations add), but
their spreads do not: ``total`` is the whole span's own distribution.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from repyability.maintenance._window_cost import (
    LEVELS,
    CostDistribution,
    _budget,
    distributions,
)
from repyability.rbd.results import _ResultMapping

#: The outcome classes, in the order of ``outcome_probabilities``.
OUTCOMES = ("nothing", "planned only", "one failure", "several failures")


@dataclass
class WindowCosts(_ResultMapping):
    """What a part's maintenance costs in each of a run of consecutive
    windows from its state now, and how often each outcome happens in each:
    returned by
    [`NonRepairable.cost_by_window`][repyability.NonRepairable.cost_by_window].

    Each window's numbers are its own (not a running total), worked out
    through every earlier window: a failure in one renews the part for the
    next. The outcomes are four classes, by what a planner must act on:
    ``"nothing"`` (no event), ``"planned only"`` (one or more planned
    replacements and no failure: known work, known cost), ``"one
    failure"`` and ``"several failures"`` (two or more), each whatever
    the planned work beside it; ``windows[k].outcomes()`` has every count.
    The means add up over the windows; the spreads do not, so ``total``
    is the whole span's own distribution.

    Attributes
    ----------
    starts : numpy.ndarray
        Each window's start, from now.
    ends : numpy.ndarray
        Each window's end.
    policy : str
        ``"run to failure"``, ``"age replacement"``, ``"block
        replacement"`` or ``"calendar"``.
    method : str
        ``"numerical"`` or ``"simulated"``.
    windows : list of CostDistribution
        Each window's distribution of counts and cost.
    mean : numpy.ndarray
        Each window's expected cost.
    standard_deviation : numpy.ndarray
        Each window's standard deviation of cost.
    levels : tuple of float
        The percentiles of ``percentiles``.
    percentiles : numpy.ndarray
        Each window's cost at each of ``levels``, shape ``(windows,
        len(levels))``.
    outcome_classes : tuple of str
        The outcome classes, in the order of ``outcome_probabilities``.
    outcome_probabilities : numpy.ndarray
        Each window's chance of each outcome class, shape ``(windows,
        4)``; each row adds up to 1.
    budget : float or None
        The budget given, if any.
    probability_above_budget : numpy.ndarray or None
        Each window's chance of costing more than ``budget``.
    total : CostDistribution
        The whole span's, from now to the last end.
    precision : float or None
        The largest ``precision`` of the windows and the total (the
        numerical route), or None.

    Examples
    --------
    See
    [`NonRepairable.cost_by_window`][repyability.NonRepairable.cost_by_window].
    """

    starts: np.ndarray
    ends: np.ndarray
    policy: str
    method: str
    windows: List[CostDistribution]
    mean: np.ndarray
    standard_deviation: np.ndarray
    levels: Tuple[float, ...]
    percentiles: np.ndarray
    outcome_classes: Tuple[str, ...]
    outcome_probabilities: np.ndarray
    budget: Optional[float]
    probability_above_budget: Optional[np.ndarray]
    total: CostDistribution
    precision: Optional[float]

    def probability_above(self, budget: float) -> np.ndarray:
        """Each window's chance of costing more than ``budget``.

        Parameters
        ----------
        budget : float
            The cost to exceed.
        """
        budget = _budget(budget)
        return np.array([w.probability_above(budget) for w in self.windows])

    def outcomes(self, window: int) -> list:
        """Every count of failures and planned replacements in a window
        (counted from 0) with a chance of at least 1e-12, most likely
        first, as ``CostDistribution.outcomes``.

        Parameters
        ----------
        window : int
            The window, counted from 0.
        """
        return self.windows[window].outcomes()


def _classes(P: np.ndarray) -> np.ndarray:
    """The chance of each outcome class (``OUTCOMES``) from ``P[i, j]``."""
    failures = P.sum(axis=1)
    return np.array(
        [
            P[0, 0],
            P[0, 1:].sum(),
            failures[1] if len(failures) > 1 else 0.0,
            failures[2:].sum(),
        ]
    )


def cost_by_window(
    part,
    ends=None,
    every=None,
    *,
    width=None,
    count=None,
    resets=None,
    age=0.0,
    since_renewal=None,
    calendar=None,
    budget=None,
    levels=LEVELS,
    method: str = "exact",
    mc_samples=None,
    seed=None,
) -> WindowCosts:
    """The work of ``NonRepairable.cost_by_window`` (see there)."""
    from repyability.rbd._window_report import window_ends

    stops = window_ends(ends, width, count)
    edges = np.concatenate([[0.0], stops])
    found = distributions(
        part,
        edges,
        every,
        resets,
        age,
        since_renewal,
        budget,
        levels,
        method,
        mc_samples,
        seed,
        whole=True,
        calendar=calendar,
    )
    windows, total = found[:-1], found[-1]
    precisions = [w.precision for w in found if w.precision is not None]
    return WindowCosts(
        edges[:-1],
        edges[1:],
        total.policy,
        total.method,
        windows,
        np.array([w.mean for w in windows]),
        np.array([w.standard_deviation for w in windows]),
        total.levels,
        np.array([w.percentiles for w in windows]),
        OUTCOMES,
        np.array([_classes(w.probabilities) for w in windows]),
        total.budget,
        (
            None
            if total.budget is None
            else np.array([w.probability_above_budget for w in windows])
        ),
        total,
        max(precisions) if precisions else None,
    )
