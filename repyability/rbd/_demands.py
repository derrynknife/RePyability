"""Demand plans: a unit that fails on demand (its ``"reliability"`` a
``surpyval.FixedEventProbability`` of ``p``) called on at set times
(``"demand_times"``) or at a rate that changes over time
(``"demand_plan"``), as a spec gives them (UC-73 of the maintenance
README).

At set times ``t_1 <= t_2 <= ...`` each demand that meets the unit working
fails it with probability ``p``, so its life can end only at them: a unit
put into service at ``t0`` survives a geometric number of the demands
after ``t0`` and fails at the next (``DemandTimesLife``). The plan is on
the calendar, not restarted by a repair, so it is no renewal process,
and only the simulations take it as a life; its values over time are
exact (``DemandCurve``): a unit is up for demand ``k`` unless an earlier
failure is still being repaired, and at most one is.

A rate ``λ(t)`` that changes over time is the duty plans' machinery: the
life ``Exponential(p)`` counted in demands, put on the diagram's clock by
the expected demands ``Λ(t)`` (a ``schedule.DutySchedule`` of rates of
``"demands"``, see ``_duty.PlannedLife``).

A plan is on the diagram's clock unless the spec's ``"demand_basis"`` is
``"operating"``: then it is in the block's operating time, which its
``"duty"`` puts on the calendar (``on_demand``).
"""

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from repyability.utils.checks import is_number, one_of, real_array

from ._covariate_schedule import ClockLife
from ._model_utils import distribution_name

#: What a demand plan's times are counted in (its spec's
#: ``"demand_basis"``): the diagram's clock (the calendar, the default),
#: or the block's operating time.
BASES = ("calendar", "operating")

#: The spec keys that say when demands come, one at most.
DEMAND_KEYS = ("demand_rate", "demand_times", "demand_plan")

#: The spec key of a demand mode beside a life that wears out: a dict of
#: the ``"probability"`` that a demand fails the unit and the demands'
#: ``"rate"`` (see ``LifeWithDemands``).
LIFE_KEY = "demand"

#: The keys of a spec's ``"demand"``.
LIFE_DEMAND_KEYS = ("probability", "rate")

#: What a block called on at set times or on a plan that changes does not
#: take, and what each is.
_REFUSED_KEYS = (
    ("standby", "a standby group"),
    ("load_sharing", "a load-sharing group"),
    ("identical", "a block of identical units"),
    ("covariates", "a covariate schedule"),
    ("upgrade", "an upgrade"),
)

#: The unit of use of a demand plan's rates (``schedule.DutySchedule``).
USE = "demands"


def demand_basis(node, spec: dict) -> str:
    """The spec's ``"demand_basis"``, checked: ``"calendar"`` (the
    default) or ``"operating"``, given only with a demand key."""
    basis = spec.get("demand_basis")
    if basis is None:
        return "calendar"
    one_of(f"Component {node!r}: the demand_basis", basis, BASES)
    if not any(spec.get(key) is not None for key in DEMAND_KEYS + (LIFE_KEY,)):
        raise ValueError(
            f"Component {node!r}: a demand_basis says what a demand plan "
            "is counted in, and it has none: give a demand_rate, "
            "demand_times, demand_plan or demand with it, or drop it."
        )
    return str(basis)


def demand_times_value(node, value) -> np.ndarray:
    """A spec's ``"demand_times"``, checked: finite times above 0 (the
    diagram starts at 0; a recurrence rule's, ``rule_times``, are), in
    order; the same time twice is two demands."""
    if isinstance(value, (str, bytes, dict)):
        raise TypeError(
            f"Component {node!r}: demand_times must be the times the unit "
            f"is called on, a list of numbers; got {value!r}."
        )
    times = np.ravel(
        real_array(value, f"Component {node!r}: demand_times")
    ).astype(float)
    if not np.all(np.isfinite(times)) or np.any(times <= 0.0):
        raise ValueError(
            f"Component {node!r}: demand_times must be finite times after "
            "the start (above 0), as CalendarContext.rule_times gives a "
            "rule's; a demand at 0 is before the unit is in service."
        )
    return np.sort(times)


def demand_plan_value(node, value) -> Any:
    """A spec's ``"demand_plan"``, checked: the expected demands per unit
    of the diagram's time from each of its times (a
    ``schedule.DutySchedule``, its ``shares`` the rates, or a dict of
    ``"times"`` and ``"rates"``), as a ``DutySchedule`` of rates of
    ``USE``; or the rate itself, for a plan of one rate."""
    from repyability.schedule import DutySchedule

    from ._duty import duty_value

    if isinstance(value, dict):
        if not {"times", "rates"} <= set(value):
            raise ValueError(
                f"Component {node!r}: a demand_plan is the demands per "
                "unit of time from each of its times, a dict of 'times' "
                "and 'rates' (or a schedule.DutySchedule); got the keys "
                f"{sorted(map(str, value))}."
            )
        unit = str(value.get("unit", "hours"))
        times, rates = value["times"], value["rates"]
    elif isinstance(value, DutySchedule):
        unit, times, rates = value.unit, value.times, value.shares
    else:
        raise TypeError(
            f"Component {node!r}: a demand_plan is the demands per unit "
            "of time from each of its times, a dict of 'times' and "
            f"'rates' (or a schedule.DutySchedule); got {value!r}. For "
            "one rate throughout, give a demand_rate."
        )
    plan = DutySchedule(
        unit,
        tuple(
            float(t)
            for t in np.ravel(real_array(times, f"{node!r}: plan times"))
        ),
        tuple(
            float(r)
            for r in np.ravel(real_array(rates, f"{node!r}: plan rates"))
        ),
        USE,
    )
    try:
        return duty_value(node, plan)
    except ValueError as refused:
        raise ValueError(
            str(refused).replace("duty plan", "demand plan", 1)
        ) from None


def _clock(duty) -> Any:
    """The operating time ``D`` of a block's duty (a share, or a plan
    that changes over time): a ``_duty.PlannedLife`` with no model,
    whose ``operating``, ``inverses`` and plan ``D`` are."""
    from repyability.schedule import DutySchedule

    from ._duty import PlannedLife

    if isinstance(duty, float):
        duty = DutySchedule("hours", (0.0,), (duty,))
    return PlannedLife(None, duty)


def compose(duty, plan) -> Any:
    """The demands on the diagram's clock of a demand ``plan`` in
    operating time (a ``DutySchedule`` of rates, or one rate) on a block
    of ``duty`` (a share, or a plan that changes): ``Λ(D(t))``, piecewise
    linear, its knots the duty's times and those at which ``D`` reaches
    the plan's; a ``DutySchedule`` of rates of ``USE`` (or one rate)."""
    from repyability.schedule import DutySchedule

    from ._duty import duty_value

    if isinstance(plan, float) and isinstance(duty, float):
        return plan * duty
    clock = _clock(duty)
    if isinstance(plan, float):
        plan_times, rates = np.zeros(1), np.array([plan])
    else:
        plan_times = np.asarray(plan.times, float)
        rates = np.asarray(plan.shares, float)
    knots = np.unique(
        np.concatenate((clock._times, clock.inverses(plan_times)))
    )
    knots = knots[np.isfinite(knots) & (knots >= 0.0)]
    # Each piece's rate: the duty's share there times the plan's rate at
    # the operating time reached within it (its middle, clear of the
    # knots' rounding).
    ends = np.append(knots[1:], knots[-1] + 1.0)
    middle = 0.5 * (knots + ends)
    reached = np.asarray(clock.operating(middle), float)
    which = np.clip(
        np.searchsorted(plan_times, reached, side="right") - 1,
        0,
        len(rates) - 1,
    )
    slopes = clock.share_after(knots) * rates[which]
    unit = getattr(plan, "unit", None) or "hours"
    return duty_value(
        None,
        DutySchedule(
            unit, tuple(map(float, knots)), tuple(map(float, slopes)), USE
        ),
    )


def on_demand(node, component: dict, built: dict) -> dict:
    """``component``, a spec with a demand key (see ``DEMAND_KEYS``) on a
    probability ``p`` of failing per demand, given the life its demands
    make: at a ``"demand_rate"`` ``d``, ``Exponential(d p)``; at set times,
    a ``DemandTimesLife``; on a plan that changes, ``Exponential(p)`` in
    demands with the plan as its ``"duty"`` (``_duty.PlannedLife``). The
    demand keys are taken off; a plan in operating time
    (``"demand_basis"``) is put on the diagram's clock through the
    block's duty, which a plan on the calendar refuses. ``built`` keeps
    one life for each probability and plan, so that components sharing
    them share it. A probability of 0 never fails: the spec drops its
    demands, and the node is a junction."""
    import surpyval as surv

    from ._duty import duty_value

    if component.get(LIFE_KEY) is not None:
        return _with_demands(node, component, built)
    p = demand_probability(node, component)
    if p is None:
        return component
    basis = demand_basis(node, component)
    out = {
        key: value
        for key, value in component.items()
        if key not in DEMAND_KEYS + ("demand_basis",)
    }
    if p == 0.0:
        return out
    life = component["reliability"]
    rate = component.get("demand_rate")
    duty = component.get("duty")
    if rate is not None:
        # In operating time, the duty puts it on the calendar as it does
        # any life in operating time.
        key: Any = ("rate", id(life), float(rate))
        if key not in built:
            built[key] = surv.Exponential.from_params([float(rate) * p])
        out["reliability"] = built[key]
        return out
    _refuse_with_plans(node, component)
    d = None if duty is None or basis == "calendar" else duty_value(node, duty)
    out.pop("duty", None)
    if component.get("demand_times") is not None:
        times = demand_times_value(node, component["demand_times"])
        if d is not None:
            times = np.sort(np.asarray(_clock(d).inverses(times), float))
            times = times[np.isfinite(times)]
        key = ("times", id(life), times.tobytes())
        if key not in built:
            built[key] = DemandTimesLife(p, times, node)
        out["reliability"] = built[key]
        return out
    plan = demand_plan_value(node, component["demand_plan"])
    if d is not None:
        plan = compose(d, plan)
    if isinstance(plan, float):
        key = ("rate", id(life), plan)
        if key not in built:
            built[key] = surv.Exponential.from_params([plan * p])
        out["reliability"] = built[key]
        return out
    key = ("plan", id(life))
    if key not in built:
        built[key] = surv.Exponential.from_params([p])
    out["reliability"] = built[key]
    out["duty"] = plan
    return out


def life_demand(node, spec: dict) -> Optional[Tuple[float, float]]:
    """A spec's ``"demand"`` beside its life, checked: ``(p, d)``, the
    probability that a demand fails the unit (in [0, 1]) and the demands'
    rate (a finite number above 0); None for a spec with none. Its
    reliability must be a life (a probability per demand takes a
    ``"demand_rate"``), with no other demand key, and no group of units,
    covariate schedule or upgrade beside it (``_REFUSED_KEYS``)."""
    from ._model_utils import is_fixed_probability

    demand = spec.get(LIFE_KEY)
    if demand is None:
        return None
    if not isinstance(demand, dict) or not set(demand) <= set(
        LIFE_DEMAND_KEYS
    ):
        raise ValueError(
            f"Component {node!r}: its demand is the demand mode beside its "
            "life, a dict of the 'probability' that a demand fails the "
            "unit and the demands' 'rate', e.g. {'probability': 0.02, "
            f"'rate': 0.5}}; got {demand!r}."
        )
    p: Any = demand.get("probability")
    d: Any = demand.get("rate")
    if not is_number(p) or not 0.0 <= float(p) <= 1.0:
        raise ValueError(
            f"Component {node!r}: its demand's probability must be the "
            "chance that a demand fails the unit, a number in [0, 1]; got "
            f"{p!r}."
        )
    if not is_number(d) or not (0.0 < float(d) < math.inf):
        raise ValueError(
            f"Component {node!r}: its demand's rate must be a finite "
            f"number above 0, the demands per unit of time; got {d!r}."
        )
    given = [key for key in DEMAND_KEYS if spec.get(key) is not None]
    if given:
        raise ValueError(
            f"Component {node!r} is given a demand and {given[0]}: a "
            "demand is the demand mode beside a life that wears out, and "
            f"a {given[0]} goes with a probability of failing per demand "
            "as the life; give one."
        )
    life = spec.get("reliability")
    if is_fixed_probability(life):
        raise ValueError(
            f"Component {node!r}: its reliability is a probability of "
            "failing per demand, which takes the demands' rate as "
            "'demand_rate'; a 'demand' goes beside a life that wears out."
        )
    for key, what in _REFUSED_KEYS:
        if spec.get(key) is not None:
            raise ValueError(
                f"Component {node!r} is given a demand mode beside its "
                f"life, and {what} ({key!r}), which are not taken "
                "together: draw the demand mode as a component of its "
                "own in series, with the probability as a "
                "surpyval.FixedEventProbability and its demand_rate."
            )
    return float(p), float(d)


def _with_demands(node, component: dict, built: dict) -> dict:
    """``component`` with a demand mode beside its life (see
    ``life_demand``) given the life of the two: a ``LifeWithDemands``,
    or for a plain exponential life of rate ``λ`` (with no duty, or the
    demands counted in operating time) ``Exponential(λ + d p)``, the
    competing exponentials' first. The demands are on the diagram's
    clock unless ``"demand_basis"`` is ``"operating"``; on the calendar,
    a duty moves the life alone, and must be a share (a duty plan that
    changes moves a life whose demands come at its pace). A probability
    of 0 leaves the life as it is."""
    import surpyval as surv

    from ._duty import duty_value
    from ._model_utils import is_exponential

    given = life_demand(node, component)
    assert given is not None
    p, d = given
    basis = demand_basis(node, component)
    out = {
        key: value
        for key, value in component.items()
        if key not in (LIFE_KEY, "demand_basis")
    }
    if p == 0.0:
        return out
    life: Any = component.get("reliability")
    duty = component.get("duty")
    operating = basis == "operating"
    if duty is not None and not operating:
        if not isinstance(duty_value(node, duty), float):
            raise ValueError(
                f"Component {node!r}: its demands are on the diagram's "
                "clock (the calendar), and its duty plan changes over "
                "time, which moves its life alone at a pace that changes: "
                "give the demands in its operating time with "
                "'demand_basis': 'operating', or a duty of one share."
            )
    elif operating and duty is None:
        raise ValueError(
            f"Component {node!r}: its demand_basis says the demands are "
            "counted in its operating time, and it has no duty to put "
            "that on the calendar: give a duty, or drop the demand_basis."
        )
    key: Any = ("life", id(life), d, p, operating)
    if key not in built:
        if is_exponential(life) and (duty is None or operating):
            rate = float(np.ravel(life.params)[0])
            built[key] = surv.Exponential.from_params([rate + d * p])
        else:
            built[key] = LifeWithDemands(life, d, p, operating)
    out["reliability"] = built[key]
    return out


def _refuse_with_plans(node, spec: dict) -> None:
    """Refuse what demands at set times or on a plan that changes cannot
    go with: a group of units, a covariate schedule, an upgrade (see
    ``_REFUSED_KEYS``), a task counted in operating time (the block's
    clock is its demands'), and a duty with a plan on the calendar."""
    which = next(k for k in DEMAND_KEYS if spec.get(k) is not None)
    for key, what in _REFUSED_KEYS:
        if spec.get(key) is not None:
            raise ValueError(
                f"Component {node!r} is called on by its {which}, and is "
                f"given {what} ({key!r}), which are not taken together: "
                "give it one or the other."
            )
    for key in ("preventive", "inspection"):
        task = spec.get(key)
        if isinstance(task, dict) and task.get("basis") == "operating":
            raise ValueError(
                f"Component {node!r} is called on by its {which}, and its "
                f"{key} is counted in operating time, which its demands "
                "do not keep: give the task's times on the diagram's "
                "clock (drop its basis)."
            )


def demand_probability(node, spec: dict) -> Optional[float]:
    """The probability ``p`` that a demand fails a component whose spec
    says when demands come (a ``"demand_rate"``, ``"demand_times"`` or
    ``"demand_plan"``), checked with them: its reliability a
    ``surpyval.FixedEventProbability``, one demand key, a rate a finite
    number above 0, and a duty only with a plan in operating time. None
    for a spec with no demand key."""
    from ._model_utils import is_fixed_probability

    given = [key for key in DEMAND_KEYS if spec.get(key) is not None]
    if not given:
        demand_basis(node, spec)
        return None
    if len(given) > 1:
        raise ValueError(
            f"Component {node!r} is given {' and '.join(given)}: its "
            "demands come at one rate (demand_rate), at set times "
            "(demand_times) or at a rate that changes (demand_plan); "
            "give one."
        )
    which = given[0]
    rate = spec.get("demand_rate")
    if rate is not None and (
        not is_number(rate) or not (0.0 < float(rate) < math.inf)
    ):
        raise ValueError(
            f"Component {node!r}: demand_rate must be a finite number "
            f"above 0, the demands per unit of time; got {rate!r}."
        )
    life: Any = spec.get("reliability")
    if distribution_name(life) != "FixedEventProbability":
        if is_fixed_probability(life):
            raise ValueError(
                f"Component {node!r}: a {which} takes the probability "
                "that a demand fails the unit as "
                "surpyval.FixedEventProbability.from_params(p); its "
                f"reliability is a {distribution_name(life)}."
            )
        raise ValueError(
            f"Component {node!r}: a {which} goes with a probability "
            "of failing per demand (surpyval.FixedEventProbability), and "
            "its reliability is a life. For a unit that wears out and may "
            "also fail on a demand, give the demand mode beside the life "
            "as 'demand': {'probability': p, 'rate': d}; or, for demands "
            "at set times or on a plan, draw the two modes as two "
            f"components in series: one with the life, one with the "
            f"probability and the {which}."
        )
    if spec.get("duty") is not None and demand_basis(node, spec) == "calendar":
        raise ValueError(
            f"Component {node!r}: its {which} is on the diagram's clock "
            "(the calendar), which a duty would stretch as though the "
            "demands came only while it runs: give the demands on the "
            "calendar and no duty, or say they are counted in its "
            "operating time with 'demand_basis': 'operating'."
        )
    return float(np.ravel(life.ff(1.0))[0])


class DemandTimesLife(ClockLife):
    """The life of a unit that fails with probability ``p`` at each
    demand that meets it working, the demands at the set ``times`` of the
    diagram's clock (in order). A unit put into service at ``t0`` meets
    the demands after ``t0``; it survives the first ``K - 1`` and fails
    at the ``K``-th, ``P(K > k) = (1 - p)^k``, so its life is from ``t0``
    to that demand (endless past the last). The demands are the
    calendar's, not restarted by a repair, so it is no renewal process,
    and only the simulations take it as a life; ``DemandCurve`` gives its
    values over time exactly. It does not age: its age is ignored."""

    lead = "is called on at set times (its demand_times)"

    def __init__(self, p: float, times, node: Any = None):
        self.p = float(p)
        self.times = np.asarray(times, dtype=float)
        self.node = node

    def demands_survived(self, u: float) -> int:
        """``K - 1`` from the uniform ``u``: the demands a unit survives,
        the least ``k`` with ``1 - (1 - p)^(k + 1) >= u``."""
        if self.p >= 1.0 or u <= 0.0:
            return 0
        k = math.ceil(math.log1p(-u) / math.log1p(-self.p)) - 1
        return max(k, 0)

    def life_left(self, t0: float, age: float, u: float) -> float:
        """The time from ``t0`` to the demand that fails a unit put into
        service then, from the uniform ``u`` (see ``demands_survived``);
        ``inf`` if it survives the last."""
        first = int(np.searchsorted(self.times, t0, side="right"))
        k = first + self.demands_survived(u)
        if k >= self.times.size:
            return math.inf
        return float(self.times[k]) - t0

    def _met(self, x) -> np.ndarray:
        """The demands by each time ``x`` (at ``x`` included)."""
        x = np.asarray(x, dtype=float)
        return np.searchsorted(self.times, x, side="right")

    def sf(self, x):
        """``(1 - p)^N(x)``: a unit new at 0 is up at ``x``, never
        repaired."""
        return np.power(1.0 - self.p, self._met(x)).astype(float)

    def ff(self, x):
        return -np.expm1(self._met(x) * math.log1p(-self.p))

    def Hf(self, x):
        return -self._met(x) * math.log1p(-self.p)

    def time_scale(self) -> float:
        """A typical failure time from new: the mean demand at which a
        unit new at 0, never repaired, fails, given that it fails within
        the plan; NaN for a plan that never fails it."""
        k = np.arange(self.times.size)
        chances = self.p * np.power(1.0 - self.p, k)
        total = float(chances.sum())
        if not total > 0.0:
            return float("nan")
        return float(np.dot(chances, self.times) / total)

    def _no_reliability(self, what: str = "reliability") -> Exception:
        """The refusal of the analyses that need a life of its own (its
        long run, mean life, reliability), one message whatever ``what``
        they need."""
        return NotImplementedError(
            f"Component {self.node!r} {self.lead} on the diagram's clock, "
            "so it has no reliability, mean life or long run of its own: "
            "each unit's life depends on when it is put into service, and "
            "the demands end. Its values over a window are exact "
            "(point_availability, mission_availability, expected_failures, "
            "expected_cost), and the simulations take it (availability, "
            "cost, simulate_timelines; the Python engine)."
        )

    def mean(self):
        raise self._no_reliability("mean life")

    def random(self, size):
        raise self._no_reliability("draws without a start time")

    def __repr__(self) -> str:
        return f"DemandTimesLife(p={self.p:g}, {self.times.size} demands)"


def demand_times_life(rbd, node) -> Optional[DemandTimesLife]:
    """The ``DemandTimesLife`` of a ``RepairableRBD``'s component called
    on at set times, or None."""
    life = getattr(rbd.components.get(node), "reliability", None)
    return life if isinstance(life, DemandTimesLife) else None


def over_time_blocker(rbd, node) -> Optional[str]:
    """Why the values over time of a component called on at set times
    are not ``DemandCurve``'s, in words; None when they are: it is
    repaired as new (or imperfectly, which a unit that does not age
    makes no different) and has no task, tests, maintenance group or
    shutdown."""
    from . import _shutdowns

    if node in rbd._preventive:
        return "it has a preventive task"
    if node in rbd._inspection:
        return "it has tests of hidden failures"
    if node in rbd._member_group:
        return "it is in a maintenance group"
    if node in rbd._restoring:
        return "its maintenance restores it partly"
    if _shutdowns.holding(rbd, node):
        return "a shutdown holds it off line"
    return None


def _repair_under_way(model, gaps: np.ndarray) -> np.ndarray:
    """``P(t_j + R >= t_k)`` for each gap ``t_k - t_j >= 0``, ``R`` the
    repair time (``model``): whether a repair begun at a demand is still
    under way at a later one, at which a unit repaired at its instant is
    not called on (the simulation's ``DemandTimesLife.life_left``, which
    takes the demands after it). A fixed time compares the sum, as the
    simulation's clock adds it; otherwise ``sf``, ``R`` having no atom."""
    gaps = np.asarray(gaps, dtype=float)
    if distribution_name(model) == "ExactEventTime":
        r = float(np.ravel(model.params)[0])
        return (gaps <= r).astype(float)
    with np.errstate(all="ignore"):
        out = np.asarray(model.sf(gaps), dtype=float).reshape(gaps.shape)
    return np.where(gaps <= 0.0, 1.0, out)


def failure_chances(life: DemandTimesLife, repair) -> np.ndarray:
    """``f_k``, the chance that demand ``k`` fails a unit new at 0
    (repaired as new in a time with model ``repair``): ``p`` times the
    chance it is not under repair then, ``1 - sum_{j<k} f_j P(t_j + R >=
    t_k)``, the repairs under way at one time being exclusive (a unit
    under repair is not called on)."""
    times, p = life.times, life.p
    n = times.size
    f = np.zeros(n)
    for k in range(n):
        if k:
            gaps = times[k] - times[:k]
            if distribution_name(repair) == "ExactEventTime":
                # As the simulation's clock adds it: t_j + r >= t_k.
                r = float(np.ravel(repair.params)[0])
                under = (times[:k] + r >= times[k]).astype(float)
            else:
                under = _repair_under_way(repair, gaps)
            busy = math.fsum(f[:k] * under)
        else:
            busy = 0.0
        f[k] = p * (1.0 - busy)
    return f


class DemandCurve:
    """A unit called on at set times (``DemandTimesLife``), new (or up) at
    0, repaired as new in a time with model ``repair``: down at ``x``
    with chance ``sum_{t_k <= x} f_k P(R > x - t_k)`` (``failure_chances``;
    at most one repair is under way), its failures ``f_k`` at the
    demands. Exact at every time; its knots the demands and the repairs'
    quantiles after each."""

    settle = np.inf
    period = None

    def __init__(self, life: DemandTimesLife, repair):
        from ._point_availability import knots

        self.life = life
        self.repair = repair
        self.times = life.times
        self.chances = failure_chances(life, repair)
        self.splits = np.append(0.0, knots(repair))

    def _repair_sf(self, s: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            out = np.asarray(self.repair.sf(s), dtype=float)
        return out.reshape(np.shape(s))

    def down_at(self, x) -> np.ndarray:
        """The chance it is down (under repair) at each time ``x``."""
        x = np.asarray(x, dtype=float)
        out = np.zeros(x.shape)
        for t, f in zip(self.times, self.chances):
            after = x >= t
            if f and after.any():
                out[after] += f * self._repair_sf(x[after] - t)
        return np.clip(out, 0.0, 1.0)

    def at(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - self.down_at(x)

    def failed_at(self, x: np.ndarray) -> np.ndarray:
        return self.down_at(x)

    def derivative(self, x: np.ndarray, scale: float) -> np.ndarray:
        """Its rate of change: less each repair's density after its
        demand."""
        from ._point_availability import _density

        x = np.asarray(x, dtype=float)
        out = np.zeros(x.shape)
        for t, f in zip(self.times, self.chances):
            after = x >= t
            if f and after.any():
                out[after] += f * _density(
                    self._repair_sf, x[after] - t, self.splits
                )
        return out

    def knots(self, start: float, stop: float) -> np.ndarray:
        """The demands in ``[start, stop]``, and the repairs' quantiles
        after each, between which the curve is smooth."""
        times = (self.times[:, None] + self.splits[None, :]).ravel()
        times = np.unique(np.concatenate(([start, stop], times)))
        return times[(times >= start) & (times <= stop)]

    def breaks(self, start: float, stop: float) -> np.ndarray:
        return self.knots(start, stop)

    def grids(self) -> list:
        """None: its knots are all breaks."""
        return []

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The expected failures before each time ``x`` (in ``[0, x)``),
        each a repair."""
        from ._point_availability import _events

        x = np.asarray(x, dtype=float)
        running = np.concatenate(([0.0], np.cumsum(self.chances)))
        failures = running[np.searchsorted(self.times, x, side="left")]
        return _events(np.asarray(failures, dtype=float).reshape(x.shape))

    def atoms(self, stop: float):
        """Its failures, at the demands before ``stop``, each taking it
        down with the chance its repair takes time."""
        from ._point_availability import Atoms

        keep = self.times < stop
        times, chances = self.times[keep], self.chances[keep]
        zero = np.zeros(times.size)
        takes_time = float(self._repair_sf(np.zeros(1))[0])
        return Atoms(times, chances, zero, zero, chances * takes_time)


def demand_curve(rbd, node, start=None) -> DemandCurve:
    """The curve of a component called on at set times (see
    ``DemandCurve``), new or up at 0 (a unit that does not age is the
    same up at any age); refuse one down at 0 or in its long-run state,
    or one whose values over time are not its (``over_time_blocker``)."""
    life = demand_times_life(rbd, node)
    assert life is not None
    why = over_time_blocker(rbd, node)
    if why is not None:
        raise NotImplementedError(blocked(life, why))
    if start is not None and not start.new:
        if start.stationary:
            raise NotImplementedError(
                f"Component {node!r} {life.lead}, which end, so it has no "
                "long-run state to start from: start it new or up."
            )
        if not start.alive:
            raise NotImplementedError(
                f"Component {node!r} {life.lead}: its values over time "
                "are worked out from a unit up at 0. Start it up, or "
                "simulate it: availability(state=...)."
            )
    return DemandCurve(life, rbd.components[node].time_to_replace)


def blocked(life: DemandTimesLife, why: str) -> str:
    """The refusal of a component called on at set times whose values
    over time are not ``DemandCurve``'s (see ``over_time_blocker``)."""
    return (
        f"{life._no_reliability()} Its values over time are worked out "
        "exactly only when it is repaired with no task, tests, "
        f"maintenance group or shutdown, and {why}."
    )


#: How its values over time are found (an analysis route's phrase).
HOW = (
    "a sum over its demands: each fails it with its probability unless "
    "an earlier failure is still under repair"
)


#: The most halvings (or Newton steps) a ``LifeWithDemands`` takes to
#: invert its cumulative hazard: enough to close a bracket of 1e300 to
#: the last bit.
_INVERSE_STEPS = 1100


class LifeWithDemands:
    """The life of a unit that wears out while it runs (``life``) and may
    also fail when it is called on: demands a Poisson process of rate
    ``d``, each failing a working unit with probability ``p``. The two are
    competing risks, renewed together at a repair, so the unit's hazard
    is the life's plus ``c = d p`` and its survival ``R(t) exp(-c t)``.

    Its functions are the life's with the demands' hazard added: ``Hf``
    the sum, ``ff`` its ``-expm1``, so that a small probability keeps its
    digits, and ``qf`` (and the draw at a virtual age, ``aged``) the
    inverse of ``Hf``, found by Newton's steps kept within a bracket.
    ``operating`` says the demands are counted in the unit's operating
    time, which a duty (``rescale_time``) then moves to the calendar with
    the life; otherwise they are on the calendar and stay as they are.
    """

    #: Every unit fails, at the latest at a demand.
    never_fails = 0.0

    def __init__(self, life, d: float, p: float, operating: bool = False):
        self.life = life
        self.d = float(d)
        self.p = float(p)
        self.c = self.d * self.p
        self.operating = bool(operating)
        self._mean: Optional[float] = None

    def _life(self, name: str, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            out = np.asarray(getattr(self.life, name)(x), dtype=float)
        return out.reshape(x.shape)

    def Hf(self, x):
        x = np.asarray(x, dtype=float)
        return self._life("Hf", x) + self.c * x

    def sf(self, x):
        x = np.asarray(x, dtype=float)
        return self._life("sf", x) * np.exp(-self.c * x)

    def ff(self, x):
        return -np.expm1(-self.Hf(x))

    def hf(self, x):
        return self._life("hf", x) + self.c

    def df(self, x):
        return self.hf(x) * self.sf(x)

    def _inverse(self, target, lo, hi) -> np.ndarray:
        """The least ``x`` in ``[lo, hi]`` with ``Hf(x) >= target``
        (``Hf(hi) >= target``), by Newton's steps from ``hi`` kept within
        the bracket, which halves where a step would leave it."""
        target, lo, hi = np.broadcast_arrays(
            np.asarray(target, float), np.asarray(lo, float), hi
        )
        lo, hi, target = lo.copy(), np.array(hi, float), target.copy()
        x = hi.copy()
        open_ = hi > lo
        for _ in range(_INVERSE_STEPS):
            if not open_.any():
                break
            at = x[open_]
            with np.errstate(all="ignore"):
                gap = self.Hf(at) - target[open_]
                step = at - gap / self.hf(at)
            above = gap >= 0.0
            low, h = lo[open_], hi[open_]
            h = np.where(above, at, h)
            low = np.where(above, low, at)
            inside = np.isfinite(step) & (step > low) & (step < h)
            nxt = np.where(inside, step, 0.5 * (low + h))
            lo[open_], hi[open_] = low, h
            x[open_] = nxt
            done = (h - low <= 4.0 * np.spacing(h)) | (gap == 0.0)
            idx = np.flatnonzero(open_)
            open_[idx[done]] = False
            x[idx[done]] = np.where(gap[done] == 0.0, at[done], h[done])
        return x

    def qf(self, q):
        """The time by which a share ``q`` of units has failed: the least
        ``x`` with ``Hf(x) >= -log(1 - q)``."""
        q = np.asarray(q, dtype=float)
        with np.errstate(all="ignore"):
            target = -np.log1p(-q)
            own = self._life("qf", q)
        # The sum's hazard is at least each part's, so its quantile is at
        # most each part's.
        hi = np.fmin(np.where(np.isnan(own), np.inf, own), target / self.c)
        out = np.where(q > 0.0, np.inf, 0.0)
        finite = (q > 0.0) & np.isfinite(hi)
        if finite.any():
            out[finite] = self._inverse(
                target[finite], np.zeros(finite.sum()), hi[finite]
            )
        return out

    def aged(self, age: float, u: float) -> float:
        """The life left to a unit at virtual age ``age``, from the
        uniform ``u``: the ``x`` with ``Hf(age + x) = Hf(age) - log(u)``,
        as ``_event_loop._aged_life`` draws any life."""
        exposure = 0.0 - math.log(u) if u > 0.0 else math.inf
        if not math.isfinite(exposure):
            return math.inf
        start = float(self.Hf(np.array([age]))[0])
        hi = age + exposure / self.c
        x = self._inverse(
            np.array([start + exposure]), np.array([age]), np.array([hi])
        )
        return max(float(x[0]) - age, 0.0)

    def mean(self) -> float:
        """The mean life, the integral of ``sf`` (see
        ``_mean_lifetime``)."""
        if self._mean is None:
            from ._mean_lifetime import mean_lifetime, model_kinks, model_knots

            self._mean = mean_lifetime(
                self.sf, model_knots(self), model_kinks(self)
            )
        return self._mean

    def random(self, size) -> np.ndarray:
        return self.qf(np.random.uniform(size=size))

    def rescale_time(self, factor: float, *, unit: Any = None):
        """The life on the calendar of a unit that runs a share
        ``factor`` of the time: the life moved (surpyval's
        ``rescale_time``), and the demands with it when they are counted
        in operating time."""
        from ._duty import on_calendar

        life = on_calendar(None, self.life, float(factor))
        d = self.d * float(factor) if self.operating else self.d
        return LifeWithDemands(life, d, self.p, self.operating)

    def to_dict(self) -> dict:
        from .serialisation import serialise_model

        return {
            "life": serialise_model(self.life),
            "demand_rate": self.d,
            "probability": self.p,
            "operating": self.operating,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LifeWithDemands":
        from .serialisation import deserialise_model

        return cls(
            deserialise_model(d["life"]),
            d["demand_rate"],
            d["probability"],
            bool(d.get("operating", False)),
        )

    def __repr__(self) -> str:
        return (
            f"LifeWithDemands({self.life!r}, demand_rate={self.d:g}, "
            f"p={self.p:g})"
        )
