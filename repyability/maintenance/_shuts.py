"""The best way to replace a part when replacing it at a shut is cheapest,
a planned replacement between shuts dearer, and one at a failure dearest
(#319, UC-7 of this package's ``README.md``).

The shuts come every ``shut_every`` (in the life's time unit); a failed
part is replaced at once, at ``failure_cost``. Four families of policy,
each at its best:

- **run to failure:** ``failure_cost`` over the mean life;
- **between shuts:** replaced at age ``d`` (the plan resets at every
  replacement), at ``planned_cost``: the age replacement of
  [`NonRepairable.optimal_replacement_policy`][repyability.NonRepairable.optimal_replacement_policy];
- **every shut:** replaced at every ``z``-th shut whatever its age, at
  ``shut_cost``;
- **at a shut if older:** at every ``x``-th shut, replaced only if at least
  ``y`` old, at ``shut_cost``.

The last two are replacement on condition at inspections every ``z`` (or
``x``) shuts, the rule an age (``older_than``, 0 for every shut): its
regeneration cycle (``_condition_replacement.condition_cycle``) gives the
mean cycle length, failures and replacements, exactly (on its grid), and
the long-run cost a unit of time is their costs over the length. The
search tries every number of shuts, and for each the age on a grid of
``_PER_INTERVAL`` points a shut interval, refined between the best one's
neighbours, up to the age a unit outlives with a chance of only ``_FAR``:
a rule that waits longer is all but running to failure. A planned policy
is chosen over running to failure only if it saves more than
``_ACCURACY`` of its cost.
"""

import math
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import minimize_scalar
from surpyval import ExactEventTime

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks

#: The chance of a unit outliving the oldest age the search tries.
_FAR = 1e-3
#: The ages a shut interval the search tries for the age rule, before it
#: refines the best.
_PER_INTERVAL = 8
#: The share of running to failure's cost a planned policy must save to be
#: chosen over it: above the cycle calculation's accuracy where the rule
#: seldom replaces (its cycles run to many thousands of failures, and their
#: cost a unit of time is good to a few parts in 10,000), where a rule
#: that is all but running to failure would otherwise win by its error.
_ACCURACY = 1e-3


@dataclass
class ReplacementPolicy(_ResultMapping):
    """One replacement policy and its long-run cost: an entry of
    [`ReplacementPolicies`][repyability.ReplacementPolicies].

    Attributes
    ----------
    policy : str
        ``"run to failure"``, ``"between shuts"`` (at an age, off the
        shuts), ``"every shut"`` (at every ``shuts``-th shut, whatever the
        age) or ``"at a shut if older"`` (at every ``shuts``-th shut, if at
        least ``older_than`` old).
    shuts : int or None
        Every how many shuts it may replace the part.
    older_than : float or None
        The age at or past which a shut replaces it.
    age : float or None
        Between shuts: the age it replaces the part at (``inf``: never, a
        planned replacement never pays).
    cost_rate : float
        The long-run expected cost a unit of time.
    """

    policy: str
    shuts: Optional[int]
    older_than: Optional[float]
    age: Optional[float]
    cost_rate: float


@dataclass
class ReplacementPolicies(_ResultMapping):
    """The best replacement policy of each family, and the best of all:
    returned by [`replacement_policies`][repyability.replacement_policies].

    Attributes
    ----------
    best : ReplacementPolicy
        The cheapest of the four (running to failure, unless a planned
        policy saves more than 0.1% against it).
    run_to_failure : ReplacementPolicy
        Running to failure.
    between_shuts : ReplacementPolicy
        The best age replacement between shuts.
    every_shut : ReplacementPolicy
        The best of replacing at every *z*-th shut, whatever the age.
    at_a_shut_if_older : ReplacementPolicy
        The best of replacing at every *x*-th shut if at least *y* old.
    saving : float
        What the best saves a unit of time against running to failure.
    """

    best: ReplacementPolicy
    run_to_failure: ReplacementPolicy
    between_shuts: ReplacementPolicy
    every_shut: ReplacementPolicy
    at_a_shut_if_older: ReplacementPolicy
    saving: float


def _amount(value, name: str) -> float:
    """``value`` as a cost: a finite number, at least 0."""
    if not (checks.is_number(value) and 0.0 <= float(value) < math.inf):
        raise ValueError(
            f"{name} must be a number, at least 0; got {value!r}."
        )
    return float(value)


def _far_age(life, scale: float) -> float:
    """The age a unit outlives with a chance of ``_FAR``: doubled from
    ``scale`` past it, then halved down to it."""
    low, high = 0.0, scale
    while float(life.sf(high)) > _FAR:
        low, high = high, 2.0 * high
    for _ in range(40):
        middle = 0.5 * (low + high)
        if float(life.sf(middle)) > _FAR:
            low = middle
        else:
            high = middle
    return high


def replacement_policies(
    life, *, shut_every, shut_cost, planned_cost, failure_cost
) -> ReplacementPolicies:
    """The cheapest way to replace a part, in the long run, when replacing
    it at a shut costs least, a planned replacement between shuts more,
    and one at a failure most.

    Four families, each at its best (see the guide's maintenance page and
    this package's use case UC-7): run to failure; replacement between
    shuts at an age; at every *z*-th shut, whatever the age; and at every
    *x*-th shut, only if at least *y* old. A failed part is replaced at
    once; replacements take no time.

    Parameters
    ----------
    life : surpyval model
        A part's life, fitted or built with ``from_params``, in the time
        unit of ``shut_every`` (days, running hours).
    shut_every : float
        The time from one shut to the next, more than 0.
    shut_cost, planned_cost, failure_cost : float
        What a replacement costs at a shut, planned between shuts, and at
        a failure, each at least 0.

    Returns
    -------
    ReplacementPolicies
        Each family's best policy and its long-run cost a unit of time,
        and the best of all. A planned policy is the best only if it saves
        more than 0.1% against running to failure: the costs of rules that
        seldom replace are good to a few parts in 10,000, and such a rule
        is all but running to failure.

    Examples
    --------
    A bearing with a Weibull life (scale 365 days, shape 2), shuts every
    180 days, $100 at a shut, $150 between shuts, $400 at a failure:

    >>> import surpyval as surv
    >>> from repyability import replacement_policies
    >>> found = replacement_policies(
    ...     surv.Weibull.from_params([365.0, 2.0]),
    ...     shut_every=180.0,
    ...     shut_cost=100.0,
    ...     planned_cost=150.0,
    ...     failure_cost=400.0,
    ... )
    >>> found.best.policy, found.best.shuts
    ('at a shut if older', 1)
    >>> round(found.best.cost_rate, 3)
    0.99
    >>> round(found.run_to_failure.cost_rate, 4)
    1.2366
    """
    from repyability.non_repairable import NonRepairable
    from repyability.rbd._condition_replacement import condition_cycle
    from repyability.rbd._model_utils import model_mean

    checks.no_distribution(life, "life")
    if not callable(getattr(life, "sf", None)):
        raise TypeError(
            "life must be a lifetime model with a survival function (sf), "
            f"such as a fitted surpyval model; got {life!r}."
        )
    if not (
        checks.is_number(shut_every) and 0.0 < float(shut_every) < math.inf
    ):
        raise ValueError(
            f"shut_every must be a number, more than 0; got {shut_every!r}."
        )
    every = float(shut_every)
    at_shut = _amount(shut_cost, "shut_cost")
    planned = _amount(planned_cost, "planned_cost")
    failed = _amount(failure_cost, "failure_cost")
    mean = model_mean(life)
    if not (0.0 < mean < math.inf):
        raise ValueError(
            "The life has no finite mean, so running it to failure has no "
            "long-run cost to compare with."
        )
    run_to_failure = ReplacementPolicy(
        "run to failure", None, None, None, failed / mean
    )

    if planned < failed:
        part = NonRepairable(life, planned_cost=planned, unplanned_cost=failed)
        found = part.optimal_replacement_policy()
        between = ReplacementPolicy(
            "between shuts",
            None,
            None,
            float(found.interval),
            float(found.cost_rate),
        )
    else:
        between = ReplacementPolicy(
            "between shuts", None, None, math.inf, run_to_failure.cost_rate
        )

    instant = ExactEventTime.from_params([0.0])

    def cost_rate(shuts: int, older_than: float) -> float:
        cycle = condition_cycle(
            life, instant, None, shuts * every, None, older_than=older_than
        )
        cost = at_shut * cycle.replaced + failed * cycle.failures
        return float(cost / cycle.length)

    far = _far_age(life, mean)
    most = max(1, math.ceil(far / every))
    every_shut = min(
        (
            ReplacementPolicy("every shut", z, 0.0, None, cost_rate(z, 0.0))
            for z in range(1, most + 1)
        ),
        key=lambda found: found.cost_rate,
    )

    candidates = []
    for shuts in range(1, most + 1):
        step = shuts * every / _PER_INTERVAL
        ages = [step * k for k in range(math.floor(far / step) + 1)]
        rates = [cost_rate(shuts, age) for age in ages]
        k = min(range(len(ages)), key=rates.__getitem__)
        low, high = ages[max(k - 1, 0)], ages[min(k + 1, len(ages) - 1)]
        refined = minimize_scalar(
            lambda age: cost_rate(shuts, age),
            bounds=(low, min(high, far)),
            method="bounded",
            options={"xatol": step / 1000.0},
        )
        if refined.success and refined.fun < rates[k]:
            age, rate = float(refined.x), float(refined.fun)
        else:
            age, rate = ages[k], rates[k]
        candidates.append(
            ReplacementPolicy("at a shut if older", shuts, age, None, rate)
        )
    at_a_shut_if_older = min(candidates, key=lambda found: found.cost_rate)

    families = (run_to_failure, between, every_shut, at_a_shut_if_older)
    best = min(families, key=lambda found: found.cost_rate)
    if best.cost_rate > (1.0 - _ACCURACY) * run_to_failure.cost_rate:
        best = run_to_failure
    return ReplacementPolicies(
        best,
        run_to_failure,
        between,
        every_shut,
        at_a_shut_if_older,
        run_to_failure.cost_rate - best.cost_rate,
    )
