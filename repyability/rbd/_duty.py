"""Duty cycles: a node whose life is in operating time, run part of the
time.

A node that operates a fraction ``d`` of the calendar ages only then, so
its life in calendar time is its operating life over ``d``: ``R_c(t) =
R(d t)``. A surpyval model of a time stays one of its kind under that
change of scale, with its times moved: the model moves itself
(``rescale_time``, from surpyval 0.29), so the node keeps its model's kind
and every exact method and simulation takes it as it would the model
fitted in calendar time.
"""

import math
from typing import Any, Union

import numpy as np

from repyability.utils.checks import is_number

from ._covariate_schedule import ClockLife
from ._model_utils import distribution_name, is_fixed_probability


def duty_fraction(node, value) -> float:
    """A node's duty, checked: the fraction of the time it operates, in
    ``(0, 1]``."""
    from repyability.utils.checks import number_or_nan

    d = number_or_nan(value)
    if not 0.0 < d <= 1.0:
        raise ValueError(
            f"The duty of node {node!r} must be the fraction of the time it "
            f"operates, in (0, 1], got {value!r}."
        )
    return d


def planned_refusals(node, spec: dict) -> None:
    """Refuse what a block on a duty plan that changes over time cannot
    go with (a virtual age, Kijima's or a fitted process's own, is
    counted in operating time, ``PlannedLife.operated_between``):
    replacement on condition by a measured level (by a threshold on the
    chance of failing, the risk is judged through the operating time to
    the next inspection, ``_event_loop._replaced_at``); and a standby
    group, whose spares age on a clock of their own."""
    lead = f"Component {node!r} runs on a duty plan that changes over time"
    preventive = spec.get("preventive")
    if (
        isinstance(preventive, dict)
        and preventive.get("policy") == "condition"
        and preventive.get("level") is not None
    ):
        raise NotImplementedError(
            f"{lead}, and is replaced on condition by its measured level, "
            "whose degradation is not followed on a plan that changes "
            "yet: give a threshold on the chance of failing, or the duty "
            "as one share."
        )
    if spec.get("standby") is not None:
        raise NotImplementedError(
            f"{lead}, and is a standby group, whose spares age on a clock "
            "of their own: give the duty as one share."
        )


def duty_value(node, value) -> Union[float, Any]:
    """A node's duty, checked: a share of the time in ``(0, 1]``, or a
    duty plan that changes over time (a ``schedule.DutySchedule``, or the
    dict it saves as: from ``times[k]``, the share ``shares[k]``). A plan
    of one share is that share. A plan of a unit of use (its ``use``,
    such as ``"km"``) gives rates of that use per unit of the diagram's
    clock (400 km a day), any number from 0, for a life counted in it: a
    plan of one rate is that rate."""
    from repyability.schedule import DutySchedule

    if isinstance(value, dict) and {"times", "shares"} <= set(value):
        value = DutySchedule(
            str(value.get("unit", "hours")),
            tuple(float(t) for t in value["times"]),
            tuple(float(x) for x in value["shares"]),
            value.get("use"),
        )
    if not isinstance(value, DutySchedule):
        return duty_fraction(node, value)
    lead = f"The duty plan of node {node!r}"
    times = np.asarray(value.times, dtype=float)
    shares = np.asarray(value.shares, dtype=float)
    if times.size == 0 or times.size != shares.size:
        raise ValueError(
            f"{lead} must have a share for each of its times, and at least "
            f"one; got {times.size} times and {shares.size} shares."
        )
    if times[0] != 0.0 or not np.all(np.diff(times) > 0.0):
        raise ValueError(
            f"{lead} must start at 0, the diagram's start, its times "
            f"increasing (UsageForecast.schedule() gives one); got times "
            f"from {times[0]:g}."
        )
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(shares))):
        raise ValueError(f"{lead} must have finite times and shares.")
    if value.use is not None:
        if np.any(shares < 0.0) or not np.any(shares):
            raise ValueError(
                f"{lead} must have rates of {value.use} per unit of time "
                "of at least 0, at least one above 0."
            )
        if np.all(shares == shares[0]):
            return float(shares[0])
        return value
    if np.any(shares < 0.0) or np.any(shares > 1.0) or not np.any(shares):
        raise ValueError(
            f"{lead} must have shares of the time in [0, 1], at least one "
            "above 0."
        )
    if np.all(shares == shares[0]):
        return duty_fraction(node, float(shares[0]))
    return value


class PlannedLife(ClockLife):
    """A unit's life in operating time on a duty plan that changes over
    time: from ``times[k]`` of the diagram's clock it operates the share
    ``shares[k]`` of the time (the last held on), so its operating time is
    ``D(t)``, piecewise linear from ``D(0) = 0``. A unit put into service
    at ``t0`` fails when ``D`` has run its operating life on from
    ``D(t0)``: at ``D⁻¹(D(t0) + L)``, ``L`` drawn from the model's own
    quantile function. From new at 0 its survival is ``R(D(t))`` (exact);
    a unit put in later has another, so it is no renewal process on the
    diagram's clock, and the simulations (the Python engine) take it.
    Where its repairs and tasks take no time it is its operating-time
    process seen through ``D`` (``time_change_blocker``), which the
    diagram's values over time take (``TimeChangedCurve``).

    Parameters
    ----------
    model
        The life in operating time: any model with ``qf`` and ``sf``.
    plan : schedule.DutySchedule
        The shares over the diagram's clock (``duty_value`` checks it).
    node : hashable
        The block's name, for the messages.
    """

    lead = "runs on a duty plan that changes over time"
    aged = "whose operating time runs at a pace that changes"

    def __init__(self, model: Any, plan: Any, node: Any = None):
        self.model = model
        self.plan = plan
        self.node = node
        self._times = np.asarray(plan.times, dtype=float)
        self._shares = np.asarray(plan.shares, dtype=float)
        hours = np.concatenate(
            ([0.0], np.cumsum(self._shares[:-1] * np.diff(self._times)))
        )
        #: D at each time of the plan.
        self._hours = hours

    def operating(self, t):
        """``D(t)``: the operating time from the diagram's 0 to ``t``.
        Before 0 it is ``t`` itself: an age a state gives at 0 is in
        operating time, so a unit put into service ``a`` operating hours
        before 0 is put in at ``-a``."""
        x = np.asarray(t, dtype=float)
        k = np.clip(
            np.searchsorted(self._times, x, side="right") - 1,
            0,
            self._times.size - 1,
        )
        share = self._shares[k]
        with np.errstate(invalid="ignore"):
            # At no duty for ever after, an endless time runs nothing.
            ran = np.where(share > 0.0, share * (x - self._times[k]), 0.0)
        out = np.where(x <= 0.0, x, self._hours[k] + ran)
        return float(out) if out.ndim == 0 else out

    def inverse(self, y: float) -> float:
        """``D⁻¹(y)``: the first time the operating time reaches ``y``
        (``inf`` if it never does; ``y`` itself before 0, see
        ``operating``)."""
        y = float(y)
        if not y > 0.0:
            return y
        if math.isinf(y):
            return math.inf
        k = int(np.searchsorted(self._hours, y, side="left")) - 1
        share = float(self._shares[k])
        if share == 0.0:
            return math.inf  # the last share is 0: it never gets there
        return float(self._times[k] + (y - self._hours[k]) / share)

    def inverses(self, y) -> np.ndarray:
        """``D⁻¹`` at each operating time ``y`` (see ``inverse``)."""
        y = np.asarray(y, dtype=float)
        k = np.clip(
            np.searchsorted(self._hours, y, side="left") - 1,
            0,
            self._times.size - 1,
        )
        share = self._shares[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = self._times[k] + (y - self._hours[k]) / share
        out = np.where(share > 0.0, out, np.inf)
        out = np.where(np.isinf(y), np.inf, out)
        return np.where(y > 0.0, out, y)

    def share_after(self, t) -> np.ndarray:
        """The share of the time run just after each time ``t``: ``D``'s
        slope from then on."""
        k = np.clip(
            np.searchsorted(self._times, np.asarray(t, dtype=float), "right")
            - 1,
            0,
            self._times.size - 1,
        )
        return self._shares[k]

    def life_left(self, t0: float, age: float, u: float) -> float:
        """The time on the diagram's clock to the failure of a unit put
        into (or back in) service at ``t0`` at virtual age ``age``, in
        operating time (0 for a new unit), from the uniform ``u``: the
        operating life left at that age (``_event_loop._aged_life``; as
        new, the model's quantile), run on from ``D(t0)``."""
        if age > 0.0:
            from ._event_loop import _aged_life

            life = _aged_life(self.model, age, u)
        else:
            life = float(np.ravel(self.model.qf(np.array([u])))[0])
        return self.run_on(t0, life)

    def run_on(self, t0: float, life: float) -> float:
        """The time on the diagram's clock from ``t0`` until the plan has
        run ``life`` of operating time on from ``D(t0)``."""
        return self.inverse(self.operating(t0) + life) - t0

    def operated_between(self, t0: float, t1: float) -> float:
        """``D(t1) - D(t0)``: the operating time run from ``t0`` to
        ``t1``, which a virtual age counts."""
        return float(self.operating(t1)) - float(self.operating(t0))

    def operated_for(self, t0: float, x: float) -> float:
        """``D(t0 + x) - D(t0)``: the operating time over the ``x`` of
        the diagram's clock from ``t0``."""
        return self.operated_between(t0, t0 + x)

    def back_by(self, t: float, x: float) -> float:
        """The time from which the plan has run ``x`` of operating time
        by ``t``: ``D⁻¹(D(t) - x)``, the first such (0 for ``x`` at least
        ``D(t)``)."""
        return self.inverse(float(self.operating(t)) - x)

    def sf(self, x):
        """``R(D(x))``: a unit new at 0 is up at ``x``."""
        return self.model.sf(self._from_zero(x))

    def ff(self, x):
        return self.model.ff(self._from_zero(x))

    def Hf(self, x):
        return self.model.Hf(self._from_zero(x))

    def _from_zero(self, x):
        return np.maximum(self.operating(x), 0.0)

    def mean(self):
        raise self._no_reliability("mean life")

    def random(self, size):
        raise self._no_reliability("draws without a start time")

    def __repr__(self) -> str:
        return (
            f"PlannedLife({type(self.model).__name__}, "
            f"{self._times.size} steps)"
        )


def on_calendar(node, model, d: float):
    """``model``, a life in operating time, as the life on the calendar of
    a node that operates the fraction ``d`` of the time: the model's own
    ``rescale_time(d)``; raise for a model that has none or refuses."""
    if d == 1.0:
        return model
    rescale = getattr(model, "rescale_time", None)
    reason = ""
    if callable(rescale) and not is_fixed_probability(model):
        try:
            return rescale(d)
        except ValueError as refused:
            reason = f" ({refused})"
    name = distribution_name(model) or type(model).__name__
    raise ValueError(
        f"Node {node!r} is given a duty, which moves a life in operating "
        f"time onto the calendar, but its life is {name}, which cannot be "
        f"moved{reason}: give the life it has on the calendar instead."
    )


def _clocked_task(node, key: str, task: dict) -> None:
    """Refuse a task counted in operating time on a duty plan that
    changes over time that the simulation does not place on it: an age
    from which a shutdown renews the unit (a task at a shutdown is on the
    calendar, so it is refused there too)."""
    lead = (
        f"Component {node!r} runs on a duty plan that changes over time, "
        f"and its {key} is counted in operating time"
    )
    for name in ("older_than",):
        if task.get(name) is not None:
            raise NotImplementedError(
                f"{lead}, with an age for the stops ({name!r}), which is "
                "not placed on a plan that changes yet: drop it, or give "
                "the duty as one share."
            )


#: What a task's times (``"basis"``) are counted in: the diagram's clock,
#: the calendar (the default), or the block's operating time.
BASES = ("calendar", "operating")

#: The times of a task counted in operating time, which go onto the
#: calendar through the block's duty.
_TASK_TIMES = {
    "preventive": ("interval", "opportunity", "older_than"),
    "inspection": ("interval", "offset", "full_test"),
}


def tasks_on_calendar(node, spec: dict, d):
    """``spec`` with each task (a ``"preventive"`` or ``"inspection"``)
    on the calendar: one counted in operating time (``"basis":
    "operating"``) has its times divided by ``d``, the block's duty (None
    for none), and the basis is taken off either way. On a duty plan that
    changes over time (``d`` a ``PlannedLife``) a preventive task keeps
    its times in operating time, for its ``_Preventive``'s ``clock`` to
    place, and an inspection for its ``_Inspection``'s; with no duty,
    the diagram's clock is the block's operating time. Return the spec
    and whether a task was counted in operating time; refuse a basis
    other than ``BASES``, and one in operating time at a shutdown or
    under a ``"calendar"`` policy."""
    from repyability.utils.checks import one_of

    out = spec
    operating = False
    for key, times in _TASK_TIMES.items():
        task = spec.get(key)
        if not isinstance(task, dict) or "basis" not in task:
            continue
        basis = task["basis"]
        task = {name: value for name, value in task.items() if name != "basis"}
        if basis is not None:
            one_of(f"Component {node!r}: the {key} basis", basis, BASES)
        if basis == "operating":
            if task.get("shutdown") is not None:
                raise ValueError(
                    f"Component {node!r}: a {key} task at a shutdown is "
                    "on the calendar, as the shut is, so it cannot be "
                    "counted in operating time: drop its basis."
                )
            if key == "preventive" and task.get("policy") == "calendar":
                raise ValueError(
                    f"Component {node!r}: a 'calendar' policy's "
                    "thresholds are its own times, so it cannot be "
                    "counted in operating time: drop its basis."
                )
            if d is None:
                # No duty here: the diagram's clock is the block's
                # operating time, until an analysis's calendar gives the
                # block its duty plan (schedule.CalendarContext).
                d = 1.0
            operating = True
            if isinstance(d, PlannedLife):
                _clocked_task(node, key, task)
                out = {**out, key: task}
                continue
            for name in times:
                value = task.get(name)
                if value is not None and is_number(value):
                    # Not refused here if not a number: the task's own
                    # checks say what it must be.
                    task[name] = float(value) / d
            plan = task.get("replace_at") if key == "preventive" else None
            if plan is not None:
                try:
                    task["replace_at"] = [float(t) / d for t in plan]
                except (TypeError, ValueError):
                    pass  # not numbers: the task's own check says so
        out = {**out, key: task}
    return out, operating


def planned_life(rbd, node) -> Any:
    """The ``PlannedLife`` of a ``RepairableRBD``'s component on a duty
    plan that changes over time, or None."""
    life = getattr(rbd.components.get(node), "reliability", None)
    return life if isinstance(life, PlannedLife) else None


def time_change_blocker(rbd, node) -> Any:
    """Why a component on a duty plan that changes over time is not its
    operating-time process seen through its plan ``D`` (see
    ``time_changed_unit``), in words; None when it is: its repairs take
    no time, and its preventive task, if any, is counted in operating
    time and takes no time, with no tests and no maintenance group.
    Every event then falls at an operating time, as from new on a plan
    of one share, put on the calendar at ``D⁻¹`` of it."""
    from repyability.rbd._requirements import _repairs_take_time

    plan = planned_life(rbd, node)
    if plan is None:
        return "it is not on a duty plan that changes over time"
    if _repairs_take_time(rbd.components[node]):
        return "its repairs take time"
    if node in rbd._inspection:
        return "it has tests of hidden failures"
    if node in rbd._member_group:
        return "it is in a maintenance group"
    schedule = rbd._preventive.get(node)
    if schedule is not None:
        if schedule.clock is not plan:
            return (
                "its preventive task is counted in calendar time, not in "
                "its operating time"
            )
        if schedule.duration is not None:
            return "its preventive task takes time"
    return None


def time_changed_unit(rbd, node):
    """The one-node ``RepairableRBD`` of a component that a duty plan
    that changes over time only time-changes (see
    ``time_change_blocker``): its spec without its duty, so that its
    clock is the unit's operating time. Kept on ``rbd``."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    kept = rbd.__dict__.setdefault("_time_changed_units", {})
    if node not in kept:
        spec = rbd._init_args["components"][node]
        as_spec = getattr(spec, "spec", None)
        if callable(as_spec) and not isinstance(spec, dict):
            spec = as_spec()
        spec = {key: value for key, value in spec.items() if key != "duty"}
        kept[node] = RepairableRBD(
            [(("in", node), node), (node, ("out", node))], {node: spec}
        )
    return kept[node]
