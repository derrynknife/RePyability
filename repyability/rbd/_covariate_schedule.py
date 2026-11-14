"""Covariate schedules on a ``RepairableRBD``'s blocks: a block whose life
is a surpyval regression model, run under covariates that change over the
diagram's time (a load stepping between levels).

In plain words: the schedule is on the system's clock, not the unit's. A
pump run at 60% load for the first 2,000 hours and at 85% after that is at
85% from hour 2,000 whatever unit is fitted then. A unit put in at hour
2,500 is new, but runs at 85% from the start; a repair or replacement
resets its age as its repair says (a replacement makes it new, imperfect
repair leaves it a virtual age), while the schedule keeps running.

How a unit's life is drawn depends on the kind of regression:

- **Proportional and additive hazards** (``HAZARD_BASED``): the hazard at
  any moment is the model's hazard at the unit's age and the covariates in
  force then, ``h(a, Z(t))`` (``h0(a) exp(beta Z(t))`` for proportional
  hazards). Over a step of the schedule, where ``Z`` is constant, the
  cumulative hazard a unit takes on from age ``a`` to ``b`` is
  ``Hf(b, Z) - Hf(a, Z)``, exactly.
- **Accelerated failure time and accelerated life** (``TIME_SCALING``): the
  covariates speed the unit's clock up or slow it down (the cumulative
  exposure model). A unit carries its cumulative hazard into each step, at
  the age that has that cumulative hazard at the step's covariates, and
  ages on from there. Imperfect repair, partial restoration and a start
  at a given age, which would need a virtual age on a clock that changes
  speed, are refused.
- **Other kinds** (proportional odds, distributional) have no hazard that
  follows the covariates in force alone, so they are refused.

A unit's life is drawn by inverting its cumulative hazard along the steps
from the time it is put into service: from one uniform ``u``, the time at
which the cumulative hazard reaches ``-log(1 - u)``, found within its
step to the last bits. Only the simulations take such a block (its life
depends on when it starts, so it is no renewal process); the exact and
numerical methods refuse it, saying so.
"""

import math
from typing import Any, Optional

import numpy as np

#: The most steps ``_solve`` takes (it ends long before, in practice in
#: two or three).
_MAX_STEPS = 2000

#: The components' spec keys a covariate schedule cannot go with, and why.
_REFUSED_KEYS = {
    "standby": (
        "a standby group, whose spares age at their dormant rate on a "
        "clock of their own"
    ),
    "duty": (
        "operated part of the time (a duty), which rescales a "
        "distribution's time: give the covariates the block runs at on "
        "the calendar instead (a step at no load while it stands)"
    ),
}


def is_scheduled(model) -> bool:
    """Whether ``model`` is a life under a covariate schedule."""
    return isinstance(model, ScheduledLife)


def scheduled(component) -> Optional["ScheduledLife"]:
    """The ``ScheduledLife`` of a ``RepairableRBD`` component (a
    ``NonRepairable``), or None for one with no covariate schedule."""
    life = getattr(component, "reliability", None)
    return life if isinstance(life, ScheduledLife) else None


class ScheduledLife:
    """A unit's life under a covariate schedule on the diagram's clock
    (see the module's notes). It is a life only simulations take: its
    ``sf``, ``ff``, ``mean`` and ``random`` refuse (a unit's life depends
    on when it is put into service), and ``life_left`` draws one.

    Parameters
    ----------
    model : surpyval regression model
        A fitted parametric regression model of proportional or additive
        hazards, accelerated failure time or accelerated life.
    schedule : surpyval StepSchedule
        The covariates in force over the diagram's time, from 0 (its
        start at or before 0), open-ended or cyclic.
    node : hashable
        The block's name, for the messages.
    """

    #: Only simulations take it (as a simulated ``StandbyModel``).
    is_simulated = True

    def __init__(self, model: Any, schedule: Any, node: Any = None):
        from surpyval.univariate.regression import StepSchedule
        from surpyval.univariate.regression.kinds import (
            HAZARD_BASED,
            TIME_SCALING,
        )

        from .regression_node import _fitted_width, _is_semiparametric

        lead = f"Component {node!r}: its covariates"
        if not isinstance(schedule, StepSchedule):
            raise TypeError(
                f"{lead} must be a surpyval StepSchedule (the covariates "
                "in force over the diagram's time, e.g. "
                "StepSchedule.from_changepoints([0, 2000], [[0.6], "
                f"[0.85]])), got {type(schedule).__name__}."
            )
        kind = getattr(model, "kind", None)
        if kind is None or not callable(getattr(model, "Hf", None)):
            raise ValueError(
                f"{lead} need a fitted surpyval regression model as its "
                f"reliability (e.g. surv.WeibullPH.fit(x, Z=...)), got a "
                f"{type(model).__name__}."
            )
        if _is_semiparametric(model):
            raise ValueError(
                f"{lead}: its reliability is a semiparametric model (a "
                "Cox model), whose baseline ends at the data, so lives "
                "past it cannot be drawn. Fit a parametric regression "
                "(e.g. surv.WeibullPH) in surpyval."
            )
        if kind in HAZARD_BASED:
            self.clock = False
        elif kind in TIME_SCALING:
            self.clock = True
        else:
            raise ValueError(
                f"{lead}: its reliability is a {kind} model, whose hazard "
                "at a moment does not follow from the covariates in force "
                "then, so a schedule on the diagram's clock is not "
                "defined for it. Fit a proportional hazards, additive "
                "hazards, accelerated failure time or accelerated life "
                "model in surpyval."
            )
        edges = np.asarray(schedule.edges, dtype=float)
        if not schedule.start <= 0.0:
            raise ValueError(
                f"{lead} start at {schedule.start:g}, and must say the "
                "covariates from 0, the start of the diagram's time."
            )
        if schedule.period is None and np.isfinite(edges[-1]):
            raise ValueError(
                f"{lead} end at {edges[-1]:g}, and must say the "
                "covariates at every time after 0: build the schedule "
                "with StepSchedule.from_changepoints (its last value "
                "held on) or StepSchedule.cyclic."
            )
        fitted = _fitted_width(model)
        if fitted is not None and schedule.p != fitted:
            raise ValueError(
                f"{lead} have {schedule.p} column(s), and the model was "
                f"fitted with {fitted}."
            )
        self.model = model
        self.schedule = schedule
        self.node = node
        self._edges = edges
        self._rows = [
            np.asarray(row, dtype=float)[np.newaxis, :]
            for row in np.asarray(schedule.Z, dtype=float)
        ]
        self._period = schedule.period
        # Each step's end, as its start, within one period (or once).
        self._ends = edges[1:].copy()
        if self._period is not None:
            self._ends[-1] = edges[0] + self._period
        # A typical life at each step's covariates: where a search for a
        # life in an open-ended step starts (its result does not depend on
        # it, only how many doublings it takes).
        try:
            self._scales = [
                float(np.ravel(model.qf(np.array([0.5]), row))[0])
                for row in self._rows
            ]
            probe = [
                float(np.ravel(model.Hf(np.array([s]), row))[0])
                for s, row in zip(self._scales, self._rows)
            ]
        except (ArithmeticError, LookupError, TypeError, ValueError) as e:
            raise ValueError(
                f"{lead}: the model cannot be evaluated at them "
                f"({type(e).__name__}: {e})."
            ) from e
        if not all(
            0.0 < s < math.inf and math.isfinite(h)
            for s, h in zip(self._scales, probe)
        ):
            raise ValueError(
                f"{lead}: the model has no proper life at some of them "
                "(its median or cumulative hazard there is not finite)."
            )

    # -- What it refuses ---------------------------------------------------

    def _no_reliability(self, what: str = "reliability") -> Exception:
        """The refusal of the exact and numerical methods (see
        ``is_simulated``), one message whatever ``what`` they need."""
        return NotImplementedError(
            f"Component {self.node!r} runs on a covariate schedule on the "
            "diagram's clock, so it has no reliability or mean life of its "
            "own: each unit's life depends on when it is put into service. "
            "Only the "
            "simulations take it: availability, cost and "
            "simulate_timelines (the Python engine)."
        )

    def sf(self, x):
        raise self._no_reliability()

    def ff(self, x):
        raise self._no_reliability()

    def Hf(self, x):
        raise self._no_reliability()

    def mean(self):
        raise self._no_reliability("mean life")

    def random(self, size):
        raise self._no_reliability("draws without a start time")

    # -- Its draws ---------------------------------------------------------

    def _curves(self, k: int):
        """The model's cumulative hazard and hazard over ages at step
        ``k``'s covariates, and its quantile there (a starting guess)."""
        row = self._rows[k]
        model = self.model

        def H(x: float) -> float:
            return float(np.ravel(model.Hf(np.array([x]), row))[0])

        def h(x: float) -> float:
            return float(np.ravel(model.hf(np.array([x]), row))[0])

        def guess(target: float) -> float:
            p = -math.expm1(-target)
            if not p < 1.0:
                return math.nan
            return float(np.ravel(model.qf(np.array([p]), row))[0])

        return H, h, guess

    def _step_at(self, t: float):
        """The step in force at time ``t`` (at least the schedule's start):
        its index and its period (0 for a schedule that is not cyclic)."""
        edges = self._edges
        last = len(self._rows) - 1
        n = 0
        if self._period is not None:
            n = int(math.floor((t - edges[0]) / self._period))
            t = t - n * self._period
        k = int(np.searchsorted(edges, t, side="right")) - 1
        return min(max(k, 0), last), n

    def _bounds(self, k: int, n: int):
        """When step ``k`` of period ``n`` starts and ends."""
        shift = 0.0 if self._period is None else n * self._period
        return self._edges[k] + shift, self._ends[k] + shift

    def _next(self, k: int, n: int):
        """The step after step ``k`` of period ``n``."""
        if k + 1 < len(self._rows):
            return k + 1, n
        return 0, n + 1

    def life_left(self, t0: float, age: float, u: float) -> float:
        """The operating time to failure of a unit put into (or back in)
        service at time ``t0`` of the diagram at ``age`` (its virtual age;
        0 for a new unit), from the uniform ``u``: the time at which its
        cumulative hazard along the schedule from ``t0`` reaches ``-log(1
        - u)``. ``inf`` if it never does."""
        exposure = -math.log1p(-u) if u < 1.0 else math.inf
        if not exposure > 0.0:
            return 0.0
        if not math.isfinite(exposure):
            return math.inf
        k, n = self._step_at(t0)
        t = t0
        a = age  # the age at t (on this step's clock, for TIME_SCALING)
        taken = 0.0  # the cumulative hazard taken on since t0
        while True:
            _, end = self._bounds(k, n)
            H, h, guess = self._curves(k)
            scale = self._scales[k]
            if self.clock and t > t0:
                # At the age with the cumulative hazard carried in, at
                # this step's covariates.
                a = _solve(H, h, taken, 0.0, math.inf, guess(taken), scale)
            at_a = H(a)
            # What it must reach from a, on this step's curve.
            target = at_a + (exposure - taken)
            length = max(end - t, 0.0)
            if math.isfinite(length):
                at_end = H(a + length)
                if at_end < target:
                    taken = at_end if self.clock else taken + (at_end - at_a)
                    a += length
                    t = max(end, t)
                    k, n = self._next(k, n)
                    continue
                x = _solve(H, h, target, a, a + length, guess(target), scale)
            else:
                x = _solve(H, h, target, a, math.inf, guess(target), scale)
            return (t - t0) + (x - a)

    def __repr__(self) -> str:
        return (
            f"ScheduledLife({type(self.model).__name__}, "
            f"schedule={self.schedule!r})"
        )


def _solve(H, h, target, lo, hi, x, scale) -> float:
    """The age ``x`` in ``[lo, hi]`` at which the rising cumulative hazard
    ``H`` (hazard ``h``) reaches ``target``, ``H(lo) <= target`` and, for
    a finite ``hi``, ``H(hi) >= target``: Newton's steps from the guess
    ``x`` kept inside a bracket that each step narrows, halving it (or,
    with no upper end yet, doubling past ``lo`` by ``scale``) where a
    step would leave it; to where ``H(x)`` is ``target`` to the last bits,
    or the bracket's ends are neighbouring numbers. ``inf`` if ``H``
    never reaches it."""
    if not lo <= x <= hi:
        x = 0.5 * (lo + hi) if math.isfinite(hi) else lo + scale
    for _ in range(_MAX_STEPS):
        if not math.isfinite(x):
            return math.inf
        value = H(x)
        if value >= target:
            hi = x
        else:
            lo = x
        if abs(value - target) <= 4.0 * np.spacing(target):
            return x
        if math.isfinite(hi) and math.nextafter(lo, math.inf) >= hi:
            return hi
        slope = h(x)
        step = x - (value - target) / slope if slope > 0.0 else math.nan
        if lo < step < hi:
            x = step
        elif math.isfinite(hi):
            x = 0.5 * (lo + hi)
        else:
            x = lo + 2.0 * max(lo - x, 0.0) + max(x, scale)
    return hi


def on_schedules(components: dict) -> dict:
    """The components, each spec with ``"covariates"`` given its life
    under them (a ``ScheduledLife``) as its ``"reliability"``, and the key
    taken off; refuse a spec whose other keys a schedule cannot go with
    (see ``_REFUSED_KEYS``, and for accelerated failure time or life,
    imperfect repair and partial restoration)."""
    out = {}
    for name, component in components.items():
        if isinstance(component, dict) and "covariates" in component:
            schedule = component["covariates"]
            component = {
                key: value
                for key, value in component.items()
                if key != "covariates"
            }
            if schedule is not None:
                for key, why in _REFUSED_KEYS.items():
                    if component.get(key) is not None:
                        raise ValueError(
                            f"Component {name!r} runs on a covariate "
                            f"schedule, and is {why}: the two together "
                            "are not taken."
                        )
                life = ScheduledLife(
                    component.get("reliability"), schedule, name
                )
                repair = component.get("repair")
                if isinstance(repair, dict) and repair.get("model") == (
                    "process"
                ):
                    raise ValueError(
                        f"Component {name!r} runs on a covariate "
                        "schedule, and is repaired by a fitted process's "
                        "own rule: give a Kijima repair (kijima1 or "
                        "kijima2) or none."
                    )
                preventive = component.get("preventive")
                restoring = (
                    isinstance(preventive, dict)
                    and preventive.get("restoration") is not None
                )
                if life.clock and (repair is not None or restoring):
                    raise ValueError(
                        f"Component {name!r} runs on a covariate schedule "
                        "with an accelerated failure time (or life) "
                        "model, whose clock changes speed with the "
                        "covariates, so a virtual age (imperfect repair, "
                        "or partial restoration) has no one meaning on "
                        "it: give a proportional or additive hazards "
                        "model, or repairs and tasks that renew."
                    )
                if (
                    isinstance(preventive, dict)
                    and preventive.get("policy") == "condition"
                ):
                    raise ValueError(
                        f"Component {name!r} runs on a covariate "
                        "schedule, and is replaced on condition, whose "
                        "threshold judges the chance of failing before "
                        "the next inspection at the unit's age alone: "
                        "give it age or block replacement, or tests."
                    )
                component = {**component, "reliability": life}
        out[name] = component
    return out


def check(rbd) -> None:
    """Refuse what a diagram ties a block on a covariate schedule into
    that the simulation does not take: a common-cause group, whose
    members fail at their constant rates."""
    for group in rbd.ccf_groups:
        for member in group.members:
            if scheduled(rbd.components.get(member)) is not None:
                raise ValueError(
                    f"Component {member!r} runs on a covariate schedule, "
                    "and is in a common-cause group, whose members fail "
                    "at constant rates: leave it out of the group."
                )


def check_state(rbd, node, state, inspected: bool) -> None:
    """Refuse a start from ``state`` the simulation cannot take for block
    ``node`` on a covariate schedule: one in its long-run state (it has
    none), with a life of its own, at an age on a clock that changes
    speed (accelerated failure time or life), or at an age since a test
    (its life would start before 0); and one that cannot be up at its
    age at the covariates in force at 0."""
    life = scheduled(rbd.components[node])
    assert life is not None
    lead = f"Component {node!r} runs on a covariate schedule"
    if state.stationary:
        raise NotImplementedError(
            f"{lead}, so it has no long-run state: give its state now, or "
            "leave it out (new)."
        )
    if state.own_life:
        raise NotImplementedError(
            f"{lead}: starting it with a life of its own (a measured "
            "level, or its own model) is not taken. Give its age alone, "
            "or leave it out (new)."
        )
    if not state.alive:
        return
    age = (state.age or 0.0) + (state.virtual_age or 0.0)
    if not age:
        return
    if life.clock:
        raise NotImplementedError(
            f"{lead} with an accelerated failure time (or life) model, "
            "whose clock changes speed with the covariates, so an age "
            "has no one meaning on it: leave it out (new)."
        )
    if inspected:
        raise NotImplementedError(
            f"{lead} and its failures are hidden: its life would be "
            "drawn from its last test, before the diagram's time starts. "
            "Leave it out (new)."
        )
    k, _ = life._step_at(0.0)
    if not life._curves(k)[0](age) < math.inf:
        raise ValueError(
            f"Component {node!r} cannot be up at age {age:g}: its "
            "reliability there, at the covariates in force at 0, is 0."
        )


def schedule_to_dict(schedule) -> dict:
    """A covariate schedule in a JSON-friendly form: its steps' edges (an
    open end as ``None``), its rows of covariates and its period (None
    for one that is not cyclic)."""
    return {
        "edges": [
            float(e) if math.isfinite(e) else None for e in schedule.edges
        ],
        "values": np.asarray(schedule.Z, dtype=float).tolist(),
        "period": None if schedule.period is None else float(schedule.period),
    }


def schedule_from_dict(d: dict):
    """The covariate schedule ``schedule_to_dict`` wrote."""
    from surpyval.univariate.regression import StepSchedule

    edges = list(d["edges"])
    if edges[0] is None:
        edges[0] = -math.inf
    if edges[-1] is None:
        edges[-1] = math.inf
    return StepSchedule(edges, d["values"], period=d.get("period"))
