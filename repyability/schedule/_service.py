"""What a service is worth (UC-67 of this package's ``README.md``).

A service (lubricate, adjust, clean, overhaul) neither replaces the unit
nor tests it, and what it does to the failure rate is not known from the
task. ``Services(plan, effect=)`` says it: by default nothing, the
service an accepted cost, whose value is minus the present value of its
own cost, exactly. A benefit comes only from an effect the caller states,
a task effect on the plan:

- a ``maintenance.Restoration`` (a Kijima factor, as new at factor 0
  under Kijima II, a virtual age to restore to, or a fitted process),
  which the service's dated task carries (``BlockPolicy(times=,
  restoration=)``);
- a ``Condition``: the plan's life a regression model fitted with a
  "serviced or overdue" covariate, which the dates turn into a covariate
  schedule on the diagram's clock.

With an effect, the plan's one-node diagram (``UnitPlan.diagram``) runs
twice from one seed, with the service and without it; the node draws the
same random numbers in both (as ``compare_npv`` has it), each run's
history is priced by ``_npv_runs.priced_chunk`` with the plan's own
prices (the money module's ``Ledger``) by calendar period, and the value
is the mean difference with its interval. The service's own cost is the
exact one, at its instants, in both cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from repyability.utils.checks import is_number, real_array

#: The runs a service's value takes by default.
MC_SAMPLES = 4000


@dataclass(frozen=True)
class Condition:
    """What a service does, as a fitted condition: the unit's ``life`` is
    a surpyval regression model fitted with a covariate that says whether
    the unit is serviced, ``fresh`` for ``lasts`` (in the plan's unit of
    time) after each service and ``stale`` otherwise (and throughout
    without the service). Plain data for
    [`Services`][repyability.schedule.Services].

    Parameters
    ----------
    life : surpyval regression model
        Proportional or additive hazards, or accelerated failure time,
        fitted with the covariate (as a block's ``"covariates"`` takes).
    fresh : sequence of float
        The covariates in force while the unit is serviced.
    stale : sequence of float
        The covariates in force while it is overdue (and throughout
        without the service), as many as ``fresh``.
    lasts : float
        How long a service keeps the unit ``fresh``, more than 0.
    """

    life: Any
    fresh: Tuple[float, ...] = field(kw_only=True)
    stale: Tuple[float, ...] = field(kw_only=True)
    lasts: float = field(kw_only=True)

    def __post_init__(self) -> None:
        for name in ("fresh", "stale"):
            row = real_array(getattr(self, name), name).ravel()
            if not np.all(np.isfinite(row)):
                raise ValueError(f"{name} must be finite covariates.")
            object.__setattr__(self, name, tuple(float(z) for z in row))
        if len(self.fresh) != len(self.stale):
            raise ValueError(
                f"fresh has {len(self.fresh)} covariates and stale "
                f"{len(self.stale)}: give the same covariates for both."
            )
        if not (is_number(self.lasts) and 0.0 < float(self.lasts) < np.inf):
            raise ValueError(
                "lasts must be how long a service keeps the unit fresh, a "
                f"number more than 0; got {self.lasts!r}."
            )

    def schedule(self, times: np.ndarray):
        """The covariate schedule the services at ``times`` give.

        Parameters
        ----------
        times : array_like
            The times of the services, on the diagram's clock (the plan's
            unit of time, from 0).
        """
        from surpyval.univariate.regression import StepSchedule

        # Each service keeps the unit fresh for ``lasts``; a service before
        # the last ran out keeps it fresh from then.
        stretches: list = []
        for t in np.unique(np.asarray(times, float)):
            end = float(t) + float(self.lasts)
            if stretches and t <= stretches[-1][1]:
                stretches[-1][1] = end
            else:
                stretches.append([float(t), end])
        points, values = [0.0], [self.stale]
        for start, end in stretches:
            if start <= points[-1]:
                values[-1] = self.fresh
            else:
                points.append(start)
                values.append(self.fresh)
            points.append(end)
            values.append(self.stale)
        return StepSchedule.from_changepoints(points, values)

    def describe(self) -> str:
        """What the condition is, in words."""
        return (
            f"a fitted condition: covariates {list(self.fresh)} for "
            f"{float(self.lasts):g} after each service, "
            f"{list(self.stale)} otherwise"
        )


@dataclass(frozen=True)
class Services:
    """What a task does: it services the unit of ``plan`` (a
    ``maintenance.UnitPlan``) without replacing it, with the stated
    ``effect`` on its life. Plain data for
    [`task_value`][repyability.schedule.task_value].

    By default (``effect=None``) a service has **no effect** on the life:
    it is an accepted cost, valued at minus its own cost's present value,
    exactly, and the plan is not needed. Otherwise:

    - a ``maintenance.Restoration``: each service restores the unit as
      the restoration says (Kijima I or II factor ``q`` from 0, 0 as new
      under Kijima II, 1 no effect, above 1 a service that does harm; a
      virtual age to restore to; a fitted process);
    - a [`Condition`][repyability.schedule.Condition]: the life a
      regression model fitted with a serviced-or-overdue covariate.

    Parameters
    ----------
    plan : UnitPlan, optional
        The unit's plan: its life, repairs and prices, with no preventive
        task of its own. Needed with an effect.
    effect : Restoration or Condition, optional
        What a service does to the life; by default nothing.
    """

    plan: Any = None
    effect: Any = None

    def __post_init__(self) -> None:
        from repyability.maintenance import Restoration, UnitPlan

        if self.effect is not None and not isinstance(
            self.effect, (Restoration, Condition)
        ):
            raise TypeError(
                "effect must say what a service does to the life: a "
                "maintenance.Restoration (a Kijima factor, as new at q=0, "
                "a virtual age, or a fitted process) or a "
                "schedule.Condition (a fitted serviced-or-overdue "
                "covariate); leave it out for no effect (the service an "
                f"accepted cost). Got {self.effect!r}."
            )
        if self.plan is not None and not isinstance(self.plan, UnitPlan):
            raise TypeError(
                "plan must be the unit's maintenance.UnitPlan (its life, "
                f"repairs and prices); got {self.plan!r}."
            )
        if self.no_effect:
            return
        if self.plan is None:
            raise ValueError(
                "A service with an effect needs the unit's plan: give "
                "Services(plan, effect=...), plan a UnitPlan with its "
                "life, repairs and prices."
            )
        held = self.plan.task("preventive")
        if held is not None:
            raise ValueError(
                f"The plan already holds a preventive task ({held!r}), and "
                "holds one at most, which the service would be: value the "
                "service on the plan without it."
            )

    @property
    def no_effect(self) -> bool:
        """Whether the service leaves the life as it is: no effect given,
        or a Kijima factor of exactly 1 (the unit as old as it was)."""
        effect = self.effect
        if effect is None:
            return True
        return (
            not isinstance(effect, Condition)
            and effect.q is not None
            and effect.model in ("kijima1", "kijima2")
            and float(effect.q) == 1.0
            and effect.replace_every is None
        )

    def describe(self) -> str:
        """What the task does, in words."""
        effect = self.effect
        if self.no_effect:
            return "services the unit, with no effect on its life"
        if isinstance(effect, Condition):
            return f"services the unit, with {effect.describe()}"
        if effect.to_age is not None:
            what = f"restoring it to a virtual age of {effect.to_age:g}"
        elif effect.process is not None:
            what = "restoring it by its fitted process"
        else:
            what = f"restoring it by a {effect.model} factor of {effect.q:g}"
        return f"services the unit, {what}"


#: The fields of a restoration an effect's plain data may give.
_RESTORATION_KEYS = (
    "q",
    "model",
    "to_age",
    "replace_every",
    "replacement_cost",
)


def effect_to_dict(effect) -> Optional[Dict[str, Any]]:
    """A service's ``effect`` as plain data, for JSON: None for no
    effect, ``{"kind": "restoration", ...}`` with the restoration's
    fields, or ``{"kind": "condition", "life": ..., "fresh": ...,
    "stale": ..., "lasts": ...}`` with the life as a diagram saves a
    node's model. ``effect_from_dict`` loads it.

    Raises
    ------
    NotImplementedError
        A restoration by a fitted process, which has no plain form:
        give its Kijima model and factor (``q``) instead.
    """
    from repyability.maintenance import Restoration
    from repyability.rbd.serialisation import serialise_model

    if effect is None:
        return None
    if isinstance(effect, Condition):
        return {
            "kind": "condition",
            "life": serialise_model(effect.life),
            "fresh": list(effect.fresh),
            "stale": list(effect.stale),
            "lasts": float(effect.lasts),
        }
    if isinstance(effect, Restoration):
        if effect.process is not None:
            raise NotImplementedError(
                "A restoration by a fitted process has no plain form: give "
                "the process's Kijima model and factor as "
                "Restoration(q=..., model=...)."
            )
        out: Dict[str, Any] = {"kind": "restoration"}
        for name in _RESTORATION_KEYS:
            value = getattr(effect, name)
            if value is not None:
                out[name] = value
        return out
    raise TypeError(
        "effect must be a maintenance.Restoration, a schedule.Condition "
        f"or None; got {effect!r}."
    )


def effect_from_dict(data):
    """The effect ``effect_to_dict`` saved (None for no effect).

    Parameters
    ----------
    data : dict or None
        ``{"kind": "restoration", "q": ..., "model": ..., "to_age": ...,
        "replace_every": ..., "replacement_cost": ...}`` (the fields
        given), ``{"kind": "condition", "life": ..., "fresh": ...,
        "stale": ..., "lasts": ...}``, or None (or ``{"kind": "none"}``)
        for no effect.
    """
    from repyability.maintenance import Restoration
    from repyability.rbd.serialisation import deserialise_model

    if data is None:
        return None
    kind = data.get("kind") if isinstance(data, dict) else None
    takes = {
        "none": (),
        "restoration": _RESTORATION_KEYS,
        "condition": ("life", "fresh", "stale", "lasts"),
    }
    if kind not in takes:
        raise ValueError(
            "an effect is None (no effect), or an object whose kind is "
            "'none', 'restoration' (a Kijima factor q with its model, or a "
            "virtual age to_age) or 'condition' (a fitted life with fresh, "
            f"stale and lasts); got {data!r}."
        )
    unknown = sorted(set(data) - {"kind", *takes[kind]})
    if unknown:
        raise ValueError(
            f"a {kind!r} effect takes no key(s) {unknown}; it takes "
            f"{list(takes[kind])}."
        )
    if kind == "none":
        return None
    if kind == "restoration":
        return Restoration(
            **{name: data[name] for name in _RESTORATION_KEYS if name in data}
        )
    missing = [name for name in takes[kind] if name not in data]
    if missing:
        raise ValueError(f"a 'condition' effect needs {missing}.")
    life = data["life"]
    if not isinstance(life, dict) or "kind" not in life:
        raise ValueError(
            "a condition's life is a model as a diagram saves a node's "
            "(serialise_model's {'kind': 'surpyval', 'model': ...})."
        )
    return Condition(
        deserialise_model(life),
        fresh=data["fresh"],
        stale=data["stale"],
        lasts=data["lasts"],
    )


@dataclass(frozen=True)
class Sides:
    """A service's simulated value: by period, the two sides' mean
    present values and undiscounted costs (the unit's own, the service's
    cost apart), failures, and the value's interval, and the run's
    settings."""

    with_money: np.ndarray
    without_money: np.ndarray
    with_cost: np.ndarray
    without_cost: np.ndarray
    failures_with: np.ndarray
    failures_without: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    total_lower: float
    total_upper: float
    n: int
    seed: int
    engine: str


def _diagrams(
    does: Services, usage, unit: str, times: np.ndarray, age, use=None
):
    """The plan's one-node diagrams with the service (at ``times`` on the
    diagram's clock, in ``unit``) and without it, on the usage's
    calendar, and the start state (``age`` in the life's unit: ``use``,
    a unit of use read from the plan's meter, or ``unit``)."""
    from repyability.maintenance import BlockPolicy, UnitPlan
    from repyability.rbd.node_state import NodeState

    from ._calendar import CalendarContext

    plan: UnitPlan = does.plan
    effect = does.effect
    later = times[times > 0.0]
    at_start = bool(np.any(times <= 0.0))
    if isinstance(effect, Condition):
        with_plan = replace(
            plan, life=effect.life, covariates=effect.schedule(times)
        )
        without_plan = replace(
            plan,
            life=effect.life,
            covariates=effect.schedule(np.array([])),
        )
    else:
        task = BlockPolicy(
            times=tuple(float(t) for t in later), restoration=effect
        )
        with_plan = replace(plan, tasks=plan.tasks + (task,))
        without_plan = plan
    uses = {} if use is None else {plan.NODE: use}
    context = CalendarContext(usage, unit=unit)
    duty = CalendarContext(
        usage, {plan.NODE: usage}, unit=unit, use=uses
    ).duty_plan(plan.NODE)
    if use is not None or duty != 1.0:
        context = CalendarContext(
            usage, {plan.NODE: usage}, unit=unit, use=uses
        )
    # A state's age on a block of one share or rate is on the diagram's
    # clock (the operating age over the duty); on a plan that changes it
    # is the operating age itself.
    if use is not None:
        from repyability.rbd._duty import duty_value

        duty = duty_value(plan.NODE, duty)
    per = float(duty) if isinstance(duty, float) else 1.0
    state_with = state_without = None
    if age:
        state_without = {plan.NODE: NodeState(age=float(age) / per)}
        start_age = float(age)
        if at_start and not isinstance(effect, Condition):
            # A service at the start: the unit taken to have had no action
            # since new, so a factor leaves q times its age.
            if effect.to_age is not None:
                start_age = float(effect.to_age)
            elif effect.process is not None:
                raise NotImplementedError(
                    "A service due at the forecast's start restores the "
                    "unit by its fitted process, whose state at the start "
                    "is not given: start the forecast after it, or state "
                    "the effect as a Kijima factor."
                )
            else:
                start_age = float(effect.q) * float(age)
        state_with = {plan.NODE: NodeState(age=start_age / per)}
    return (
        with_plan.diagram(context),
        without_plan.diagram(context),
        state_with,
        state_without,
    )


def simulated(
    does: Services,
    usage,
    unit: str,
    age: float,
    times: np.ndarray,
    ends: np.ndarray,
    rate: float,
    mc_samples: Optional[int],
    seed,
    confidence: float,
    engine: str = "auto",
    use=None,
) -> Sides:
    """The service's value by simulation: the plan's diagram with and
    without it from one seed (``times`` and the periods' ``ends`` on the
    diagram's clock, ``rate`` per unit of it), each run priced by period;
    the service's own cost is not in either side."""
    from repyability.rbd import _montecarlo as montecarlo
    from repyability.rbd import _npv, _npv_runs, _pricing
    from repyability.utils import checks

    n = checks.whole_number(
        MC_SAMPLES if mc_samples is None else mc_samples, "mc_samples"
    )
    montecarlo.check_confidence(confidence)
    seed = _npv_runs._seed(seed)
    with_d, without_d, state_with, state_without = _diagrams(
        does, usage, unit, times, age, use
    )
    horizon = float(ends[-1])
    node = does.plan.NODE
    periods = ends.size
    money: Dict[str, np.ndarray] = {}
    plain: Dict[str, np.ndarray] = {}
    fails: Dict[str, np.ndarray] = {}
    used = engine
    for name, diagram, state in (
        ("with", with_d, state_with),
        ("without", without_d, state_without),
    ):
        _npv_runs._require_priceable(diagram)
        priced = _npv.Money(diagram, rate, 0.0)
        # The plan's own retirement ends its runs there, and its own
        # salvage is credited then (as npv_distribution's runs), in the
        # period it falls in; the periods after hold nothing.
        stop, salvage, nominal_salvage, _ = _npv.own_end(
            diagram, horizon, priced
        )
        own_ends = np.append(ends[ends < stop], stop)
        last = own_ends.size - 1
        _npv_runs._require_matched(
            diagram, f"the plan {name} the service", stop, state
        )
        pv = np.zeros((n, periods))
        nominal = np.zeros((n, periods))
        count = np.zeros((n, periods))
        pv[:, last] -= salvage
        nominal[:, last] -= nominal_salvage
        for first in range(0, n, _npv_runs._CHUNK):
            size = min(_npv_runs._CHUNK, n - first)
            ledger, runs = _npv_runs.priced_chunk(
                diagram,
                stop,
                own_ends,
                priced,
                size,
                seed,
                engine,
                first,
                state,
            )
            used = runs.engine
            for (k, _category, _node), (a, b) in ledger.cells.items():
                pv[first : first + size, k] += a
                nominal[first : first + size, k] += b
            history, at, down, planned = _pricing.changes(
                runs.components[node]._data
            )
            failed = down & ~planned & (at <= stop)
            k = np.minimum(
                np.searchsorted(own_ends, at[failed], side="right"), last
            )
            np.add.at(count, (first + history[failed], k), 1.0)
        money[name], plain[name], fails[name] = pv, nominal, count
    differences = money["without"] - money["with"]
    z = montecarlo.z_value(confidence)
    mean = differences.mean(axis=0)
    spread = differences.std(axis=0, ddof=1) / math.sqrt(n)
    whole = differences.sum(axis=1)
    whole_se = float(whole.std(ddof=1) / math.sqrt(n))
    return Sides(
        with_money=money["with"].mean(axis=0),
        without_money=money["without"].mean(axis=0),
        with_cost=plain["with"].mean(axis=0),
        without_cost=plain["without"].mean(axis=0),
        failures_with=fails["with"].mean(axis=0),
        failures_without=fails["without"].mean(axis=0),
        lower=mean - z * spread,
        upper=mean + z * spread,
        total_lower=float(whole.mean()) - z * whole_se,
        total_upper=float(whole.mean()) + z * whole_se,
        n=n,
        seed=seed,
        engine=used,
    )
