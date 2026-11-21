"""Duty cycles: a node whose life is in operating time, run part of the
time.

A node that operates a fraction ``d`` of the calendar ages only then, so
its life in calendar time is its operating life over ``d``: ``R_c(t) =
R(d t)``. A surpyval model of a time stays one of its kind under that
change of scale, with its times moved: the model moves itself where it can
(``rescale_time``, from the surpyval release after 0.28), so the node keeps
its model's kind and every exact method and simulation takes it as it would
the model fitted in calendar time. With surpyval 0.28, which has no
``rescale_time``, ``_SCALED`` moves the parameters of each distribution
(the workaround goes once the minimum surpyval rescales).
"""

import math
from typing import Any, Callable, Dict, List, Union

import numpy as np

from repyability.utils.checks import is_number

from ._covariate_schedule import ClockLife
from ._model_utils import distribution_name, is_fixed_probability


def _scale_first(p: List[float], d: float) -> List[float]:
    """A scale parameter first (a characteristic life), shapes after."""
    return [p[0] / d, *p[1:]]


def _location_scale(p: List[float], d: float) -> List[float]:
    """A location and a scale, both times."""
    return [p[0] / d, p[1] / d]


#: How each surpyval distribution's parameters move when its time is
#: divided by ``d`` (``R_c(t) = R(d t)``), for a surpyval whose models do
#: not rescale themselves (0.28).
_SCALED: Dict[str, Callable[[List[float], float], List[float]]] = {
    "Exponential": lambda p, d: [p[0] * d],
    "Weibull": _scale_first,
    "ExpoWeibull": _scale_first,
    "LogLogistic": _scale_first,
    "Rayleigh": _scale_first,
    # The log of a time.
    "LogNormal": lambda p, d: [p[0] - math.log(d), p[1]],
    "Galton": lambda p, d: [p[0] - math.log(d), p[1]],
    # A shape and a rate.
    "Gamma": lambda p, d: [p[0], p[1] * d],
    "Normal": _location_scale,
    "Gauss": _location_scale,
    "Gumbel": _location_scale,
    "GumbelLEV": _location_scale,
    "Logistic": _location_scale,
    "Uniform": _location_scale,
}


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
    go with: a virtual age (imperfect repair, partial restoration), whose
    operating time would run at a pace that changes; replacement on
    condition, which judges the next inspection's risk at the age alone;
    and a standby group, whose spares age on a clock of their own."""
    lead = f"Component {node!r} runs on a duty plan that changes over time"
    preventive = spec.get("preventive")
    restoring = (
        isinstance(preventive, dict)
        and preventive.get("restoration") is not None
    )
    if spec.get("repair") is not None or restoring:
        raise NotImplementedError(
            f"{lead}, and keeps a virtual age (imperfect repair or partial "
            "restoration), which is not followed on a plan that changes "
            "yet: give repairs and tasks that renew, or the duty as one "
            "share."
        )
    if isinstance(preventive, dict) and preventive.get("policy") == (
        "condition"
    ):
        raise NotImplementedError(
            f"{lead}, and is replaced on condition, whose threshold judges "
            "the chance of failing before the next inspection at the "
            "unit's age alone: give it age or block replacement, or the "
            "duty as one share."
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
    of one share is that share."""
    from repyability.schedule import DutySchedule

    if isinstance(value, dict) and {"times", "shares"} <= set(value):
        value = DutySchedule(
            str(value.get("unit", "hours")),
            tuple(float(t) for t in value["times"]),
            tuple(float(x) for x in value["shares"]),
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
    a unit put in later has another, so the diagram's exact methods do not
    take it, and only the simulations (the Python engine) do.

    Parameters
    ----------
    model
        The life in operating time: any model with ``qf`` and ``sf``.
    plan : schedule.DutySchedule
        The shares over the diagram's clock (``duty_value`` checks it).
    node : hashable
        The block's name, for the messages.
    """

    clock = True
    new_only = True
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
        """``D(t)``: the operating time from the diagram's 0 to ``t``."""
        x = np.asarray(t, dtype=float)
        k = np.clip(
            np.searchsorted(self._times, x, side="right") - 1,
            0,
            self._times.size - 1,
        )
        out = self._hours[k] + self._shares[k] * (x - self._times[k])
        out = np.where(x <= 0.0, 0.0, out)
        return float(out) if out.ndim == 0 else out

    def inverse(self, y: float) -> float:
        """``D⁻¹(y)``: the first time the operating time reaches ``y``
        (``inf`` if it never does)."""
        y = float(y)
        if not y > 0.0:
            return 0.0
        if math.isinf(y):
            return math.inf
        k = int(np.searchsorted(self._hours, y, side="left")) - 1
        share = float(self._shares[k])
        if share == 0.0:
            return math.inf  # the last share is 0: it never gets there
        return float(self._times[k] + (y - self._hours[k]) / share)

    def life_left(self, t0: float, age: float, u: float) -> float:
        """The time on the diagram's clock to the failure of a unit put
        into service new at ``t0``, from the uniform ``u`` (no ``age``: a
        state other than new is refused)."""
        life = float(np.ravel(self.model.qf(np.array([u])))[0])
        return self.inverse(self.operating(t0) + life) - t0

    def sf(self, x):
        """``R(D(x))``: a unit new at 0 is up at ``x``."""
        return self.model.sf(self.operating(x))

    def ff(self, x):
        return self.model.ff(self.operating(x))

    def Hf(self, x):
        return self.model.Hf(self.operating(x))

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
    ``rescale_time(d)`` where it has one, else (surpyval 0.28) a surpyval
    parametric distribution with its parameters moved (``_SCALED``);
    raise for any other model."""
    if d == 1.0:
        return model
    rescale = getattr(model, "rescale_time", None)
    if callable(rescale) and not is_fixed_probability(model):
        return rescale(d)
    name = distribution_name(model)
    move = _SCALED.get(name) if name is not None else None
    if move is None or is_fixed_probability(model):
        raise ValueError(
            f"Node {node!r} is given a duty, which moves a life in "
            "operating time onto the calendar, but its life is "
            f"{name or type(model).__name__}, which cannot be moved: give "
            "the life it has on the calendar instead."
        )
    import surpyval

    params = [float(p) for p in model.params]
    extras = dict(getattr(model, "extras", {}))
    if extras.get("gamma"):
        # An offset is a time too.
        extras["gamma"] = float(extras["gamma"]) / d
    assert name is not None  # a distribution of time (see _SCALED)
    return getattr(surpyval, name).from_params(move(params, d), **extras)


def _clocked_task(node, key: str, task: dict) -> None:
    """Refuse a task counted in operating time on a duty plan that
    changes over time that the simulation does not place on it."""
    lead = (
        f"Component {node!r} runs on a duty plan that changes over time, "
        f"and its {key} is counted in operating time"
    )
    if key == "inspection":
        raise NotImplementedError(
            f"{lead}: tests counted in operating time are not placed on a "
            "plan that changes yet. Give the test interval in calendar "
            "time, or the duty as one share."
        )
    for name in ("opportunity", "older_than"):
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
    place; with no duty, the diagram's clock is the block's operating
    time. Return the spec and whether a task was counted in operating
    time; refuse a basis other than ``BASES``, and one in operating time
    at a shutdown or under a ``"calendar"`` policy (and, on
    a plan that changes, an inspection, or an age at which a stop renews
    the unit)."""
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
