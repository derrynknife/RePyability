"""A ``RepairableRBD``'s shutdowns: outages planned at given times, each
lasting a fixed time, that hold the whole system or a named set of its
blocks off line, with the work done in them given on the blocks.

A shutdown starts every ``period`` from its ``offset`` (or at given
``times``) and lasts its ``duration``; during it the blocks in its scope
are off line (the whole system, by default, is down), planned, whatever
their own states. A block's own process goes on: idle, it keeps ageing
and can fail (``idle_ageing``, the default), or it stands still until
the shut ends, a shift of its own process by the shut's length.

The work done at a shut is given on the block (``"preventive":
{"shutdown": name}``, ``"inspection": {"shutdown": name}``), and *is*
today's schedule aligned with the shut: renewed at every shut, block
replacement with the shut's period and offset and a fixed duration; tested
at every shut, an inspection likewise. ``_spec`` translates such a task
here (``bound_preventive``, ``bound_inspection``) into the ``_Preventive``
or ``_Inspection`` it is, which every method then takes as it is. What
the shut adds is a deterministic mask: the blocks in its scope held down
inside its windows (``MaskedCurve`` over time; ``mask`` in the long run),
its starts planned outages at exact times, and its cost at each start.
The simulation follows the windows as events of their own
(``_event_loop``).
"""

import math
from typing import (
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Tuple,
)

import numpy as np

from repyability.rbd._common import _common_period, _matched
from repyability.rbd._events import _Shutdown
from repyability.rbd._point_availability import (
    Atoms,
    curve_breaks,
    curve_grids,
)
from repyability.utils.checks import number_or_nan, real_array

#: The keys of a shutdown's spec.
SHUTDOWN_KEYS = (
    "period",
    "offset",
    "times",
    "duration",
    "scope",
    "cost",
    "idle_ageing",
)

#: What a task bound to a shutdown takes from it, and so may not give.
_FROM_THE_SHUT = {
    "preventive": ("interval", "duration", "opportunity", "calendar", "level"),
    "inspection": ("interval", "offset", "duration"),
}

#: The share of a period below which an offset is taken as 0 (as an
#: inspection's is, #237).
_TINY_OFFSET = 1e-9


# -- validation ---------------------------------------------------------------


def validate(given) -> Dict[Hashable, _Shutdown]:
    """A diagram's ``shutdowns`` (a dict of a name and a spec each),
    checked: each spec's keys, its calendar, duration, cost and option,
    and that no two shuts' windows overlap. The scopes are checked
    against the diagram's nodes once it is built (``check``)."""
    if given is None:
        return {}
    if not isinstance(given, dict):
        raise ValueError(
            "shutdowns must be a dict of a name and a spec for each "
            f"shutdown, got {given!r}."
        )
    out = {name: _validate_one(name, spec) for name, spec in given.items()}
    _refuse_overlaps(list(out.values()))
    return out


def _validate_one(name, spec) -> _Shutdown:
    if not isinstance(spec, dict):
        raise ValueError(
            f"Shutdown {name!r} must be a dict with "
            f"{', '.join(SHUTDOWN_KEYS)}, got {spec!r}."
        )
    unknown = set(spec) - set(SHUTDOWN_KEYS)
    if unknown:
        raise ValueError(
            f"Shutdown {name!r} has unknown key(s) "
            f"{sorted(map(str, unknown))}. A shutdown takes "
            f"{', '.join(SHUTDOWN_KEYS)}."
        )
    if "duration" not in spec:
        raise ValueError(
            f"Shutdown {name!r} needs a duration: the time it lasts (0 for "
            "one that takes no time)."
        )
    duration = number_or_nan(spec["duration"])
    if not (duration >= 0.0 and math.isfinite(duration)):
        raise ValueError(
            f"Shutdown {name!r}: the duration must be a number, 0 or more, "
            f"got {spec['duration']!r}."
        )
    period, offset, times = _calendar(name, spec, duration)
    cost = spec.get("cost")
    if cost is None:
        cost = 0.0
    else:
        number = number_or_nan(cost)
        if not (number >= 0.0 and math.isfinite(number)):
            raise ValueError(
                f"Shutdown {name!r}: the cost must be a number, 0 or more "
                f"(charged at each shut), got {cost!r}."
            )
        cost = number
    idle = spec.get("idle_ageing", True)
    if not isinstance(idle, (bool, np.bool_)):
        raise ValueError(
            f"Shutdown {name!r}: idle_ageing must be True (the blocks not "
            "worked on keep ageing) or False (they stand still), got "
            f"{idle!r}."
        )
    scope = spec.get("scope")
    if scope is not None:
        if isinstance(scope, (str, bytes)) or not hasattr(scope, "__iter__"):
            raise ValueError(
                f"Shutdown {name!r}: the scope must be a list of the "
                "diagram's nodes, or None for the whole system, got "
                f"{scope!r}."
            )
        scope = tuple(dict.fromkeys(scope))
        if not scope:
            raise ValueError(
                f"Shutdown {name!r}: the scope names no node; give the "
                "nodes it holds off line, or None for the whole system."
            )
    return _Shutdown(
        name, period, offset, times, duration, scope, cost, bool(idle)
    )


def _calendar(name, spec: dict, duration: float):
    """A shutdown's ``period`` and ``offset``, or its ``times``,
    checked: the shuts must not overlap one another."""
    if ("period" in spec) == ("times" in spec):
        raise ValueError(
            f"Shutdown {name!r} needs a period (with an offset) or its "
            "times, one or the other."
        )
    if "times" in spec:
        if "offset" in spec:
            raise ValueError(
                f"Shutdown {name!r}: an offset places shuts on a period; "
                "given times, give no offset."
            )
        try:
            times = np.atleast_1d(real_array(spec["times"], "times"))
        except TypeError:
            times = np.array([math.nan])
        if (
            times.ndim != 1
            or not times.size
            or not np.all(np.isfinite(times))
            or times[0] < 0.0
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                f"Shutdown {name!r}: the times must be the shuts' start "
                "times, increasing, from 0 on, got "
                f"{spec['times']!r}."
            )
        gaps = np.diff(times)
        if gaps.size and not duration < float(gaps.min()):
            raise ValueError(
                f"Shutdown {name!r} lasts {duration:g}, as long as the gap "
                f"between two of its times ({float(gaps.min()):g}) or "
                "longer: one shut would start before the last ended."
            )
        return None, 0.0, tuple(float(t) for t in times)
    period = number_or_nan(spec["period"])
    if not (period > 0.0 and math.isfinite(period)):
        raise ValueError(
            f"Shutdown {name!r}: the period must be a positive, finite "
            f"number, got {spec['period']!r}."
        )
    offset = number_or_nan(spec.get("offset", 0.0))
    if not 0.0 <= offset < period:
        raise ValueError(
            f"Shutdown {name!r}: the offset, the start of its first shut, "
            f"must be at least 0 and less than its period, {period:g}, "
            f"got {spec.get('offset')!r}."
        )
    if offset <= _TINY_OFFSET * period:
        # An offset of 0 puts the first shut at the period.
        offset = 0.0
    if not duration < period:
        raise ValueError(
            f"Shutdown {name!r} lasts {duration:g}, its period ({period:g}) "
            "or longer: one shut would start before the last ended."
        )
    return period, offset, None


def _refuse_overlaps(shuts: List[_Shutdown]) -> None:
    """Refuse two shutdowns whose windows overlap (not yet taken)."""
    timed = [shut for shut in shuts if shut.duration > 0.0]
    for i, a in enumerate(timed):
        for b in timed[i + 1 :]:  # noqa: E203
            if _overlap(a, b):
                raise ValueError(
                    f"Shutdowns {a.name!r} and {b.name!r} overlap: one "
                    "starts before the other has ended, which is not "
                    "taken yet. Move one (its offset or times), or make "
                    "them one shutdown."
                )


def _overlap(a: _Shutdown, b: _Shutdown) -> bool:
    """Whether some window of ``a`` overlaps one of ``b``."""
    if a.periodic and b.periodic:
        try:
            period = _common_period([a.period, b.period])
        except NotImplementedError:
            # Periods with no common one bring the shuts as close as one
            # likes, in time.
            return True
        stop = period + max(a.offset, b.offset) + a.duration + b.duration
    else:
        given = [s for s in (a, b) if not s.periodic]
        stop = max(float(s.times[-1]) for s in given) + max(  # type: ignore
            a.duration, b.duration
        )
        stop = max(stop, *(s.offset + 1.0 for s in (a, b)))
    a_start, a_end = a.windows(stop)
    b_start, b_end = b.windows(stop)
    if not a_start.size or not b_start.size:
        return False
    return bool(
        np.any(
            (a_start[:, None] < b_end[None, :])
            & (b_start[None, :] < a_end[:, None])
        )
    )


def bound_preventive(node, spec: dict, shutdowns: dict) -> Tuple[dict, dict]:
    """A ``"preventive"`` spec bound to a shutdown (its ``"shutdown"``)
    as the schedule it is: the spec with the shut's interval, the policy
    and its fixed duration (for ``_spec._validate_preventive``), and the
    ``_Preventive``'s fields the shut gives (its offset or times, the
    shut's name and an ``"older_than"``)."""
    shut = _named(node, spec["shutdown"], shutdowns, "preventive")
    _refuse_given(node, spec, "preventive", shut)
    policy = spec.get("policy", "block")
    if policy not in ("block", "condition"):
        raise ValueError(
            f"Component {node!r}: a task at shutdown {shut.name!r} renews "
            "the unit at every shut ('block', the default) or inspects it "
            "there ('condition', with a threshold), so its policy cannot "
            f"be {policy!r}."
        )
    older = spec.get("older_than")
    view = {
        key: value
        for key, value in spec.items()
        if key not in ("shutdown", "older_than")
    }
    if older is not None:
        if policy == "condition" or spec.get("threshold") is not None:
            raise ValueError(
                f"Component {node!r}: at a shut, the unit is replaced if "
                "it is older than older_than, or on its condition (a "
                "threshold): give one or the other."
            )
        if spec.get("restoration") is not None:
            raise ValueError(
                f"Component {node!r}: older_than replaces the unit at a "
                "shut by its age since new, which a restoration does not "
                "give: give one or the other."
            )
        age = number_or_nan(older)
        if not (age > 0.0 and math.isfinite(age)):
            raise ValueError(
                f"Component {node!r}: older_than must be a positive, "
                f"finite age, got {older!r}."
            )
        older = age
        # Judged by its age at each shut: replacement on condition with
        # the decision on the age (see _condition_replacement).
        policy, view["threshold"] = "condition", 0.0
    view["policy"] = policy
    view["interval"] = _interval(shut)
    view["duration"] = _fixed(shut.duration)
    return view, {
        "offset": shut.offset,
        "older_than": older,
        "times": shut.times,
        "shutdown": shut.name,
    }


def bound_inspection(node, spec: dict, shutdowns: dict) -> Tuple[dict, dict]:
    """An ``"inspection"`` spec bound to a shutdown, as
    ``bound_preventive`` gives a preventive one: the tests take the shut."""
    shut = _named(node, spec["shutdown"], shutdowns, "inspection")
    _refuse_given(node, spec, "inspection", shut)
    view = {key: value for key, value in spec.items() if key != "shutdown"}
    if not shut.periodic and (
        spec.get("full_test") is not None
        or number_or_nan(spec.get("coverage", 1.0)) < 1.0
    ):
        raise ValueError(
            f"Component {node!r}: tests that can miss a failure are full "
            "every so many tests on a period, and shutdown "
            f"{shut.name!r} is at given times: test it fully (a coverage "
            "of 1), or put the shutdown on a period."
        )
    view["interval"] = _interval(shut)
    view["duration"] = _fixed(shut.duration)
    view["offset"] = shut.offset
    return view, {"times": shut.times, "shutdown": shut.name}


def _named(node, name, shutdowns: dict, key: str) -> _Shutdown:
    """The shutdown a component's task names, or a ValueError."""
    try:
        return shutdowns[name]
    except (KeyError, TypeError):
        known = list(shutdowns) or "none"
        raise ValueError(
            f"Component {node!r}: its {key} names shutdown {name!r}, which "
            f"the diagram's shutdowns do not have (they are {known})."
        ) from None


def _refuse_given(node, spec: dict, key: str, shut: _Shutdown) -> None:
    for option in _FROM_THE_SHUT[key]:
        if spec.get(option) is not None:
            raise ValueError(
                f"Component {node!r}: its {key} is done at shutdown "
                f"{shut.name!r}, whose calendar and duration apply, so it "
                f"takes no {option}."
            )


def _interval(shut: _Shutdown) -> float:
    """The interval a task bound to ``shut`` is on: the shut's period, or
    at given times the shortest gap between them (which only sizes the
    simulation's streams; the times themselves place the tasks)."""
    if shut.periodic:
        return float(shut.period)  # type: ignore[arg-type]
    times = np.asarray(shut.times, dtype=float)
    gaps = np.diff(np.concatenate(([0.0], times)))
    gaps = gaps[gaps > 0.0]
    return float(gaps.min()) if gaps.size else max(float(times[-1]), 1.0)


def _fixed(duration: float):
    """A task's time at a shut, the shut's: fixed, or none."""
    if not duration > 0.0:
        return "instant"
    from surpyval import ExactEventTime

    return ExactEventTime.from_params([duration])


def check(rbd) -> None:
    """Check a built diagram's shutdowns against its nodes: each scope's
    nodes are components of it, each task bound to a shut is on a block
    in its scope, and the blocks a shut pauses can be paused (see
    ``_pause_refusal``)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    for node, component in rbd.components.items():
        if isinstance(component, RepairableRBD) and component._shutdowns:
            raise ValueError(
                f"Node {node!r} is an RBD with shutdowns of its own, which "
                "a diagram it is nested in does not take yet: give the "
                "shutdowns to this diagram, with the nested RBD in their "
                "scope."
            )
    for shut in rbd._shutdowns.values():
        for node in shut.scope or ():
            if node not in rbd.components:
                what = (
                    "a junction (it never fails)"
                    if node in rbd._junctions()
                    else "not a component of the diagram"
                )
                raise ValueError(
                    f"Shutdown {shut.name!r}: its scope names {node!r}, "
                    f"which is {what}."
                )
    for node in rbd.components:
        for name in bound_to(rbd, node):
            shut = rbd._shutdowns[name]
            if shut.scope is not None and node not in shut.scope:
                raise ValueError(
                    f"Component {node!r} is worked on at shutdown "
                    f"{name!r}, so it must be in its scope "
                    f"{list(shut.scope)}."
                )
    for shut in rbd._shutdowns.values():
        if shut.idle_ageing or not shut.duration > 0.0:
            continue
        for node in scope(rbd, shut):
            message = _pause_refusal(rbd, node, shut)
            if message:
                raise ValueError(message)


def _pause_refusal(rbd, node, shut: _Shutdown) -> Optional[str]:
    """Why a block in the scope of a shut that pauses its idle blocks
    (``idle_ageing`` False) cannot be paused, or None: a block worked on
    there (renewed, restored or tested) is not idle; one with a calendar
    of its own, or tied to others, is not taken yet."""
    from repyability.rbd import _crews
    from repyability.rbd.repairable_rbd import RepairableRBD

    lead = (
        f"Shutdown {shut.name!r} stands its idle blocks still "
        f"(idle_ageing False), and {node!r}"
    )
    after = (
        " is not taken yet: let its idle blocks age (idle_ageing True), "
        "or leave it out of the scope."
    )
    schedule = rbd._preventive.get(node)
    inspection = rbd._inspection.get(node)
    if schedule is not None and schedule.shutdown == shut.name:
        if schedule.policy == "condition":
            return (
                f"{lead} is inspected there and kept unless it is due "
                "(its condition, or older_than): standing such a block "
                "still" + after
            )
        return None
    if inspection is not None and inspection.shutdown == shut.name:
        return None
    if schedule is not None or inspection is not None:
        return (
            f"{lead} has a calendar of its own (its maintenance or tests): "
            "standing it still" + after
        )
    component = rbd.components[node]
    if isinstance(component, RepairableRBD):
        return f"{lead} is a nested RBD: standing it still" + after
    if node in rbd._standby:
        return f"{lead} is a standby group: standing it still" + after
    if any(node in group.members for group in rbd.ccf_groups):
        return f"{lead} is in a common-cause group: standing it still" + after
    if node in rbd._member_group:
        return f"{lead} is in a maintenance group: standing it still" + after
    if _crews._crews_limited(rbd):
        return (
            f"{lead} waits for the repair crews with the others: standing "
            "it still" + after
        )
    return None


# -- what a diagram's shutdowns do to its nodes -------------------------------


def bound_to(rbd, node) -> List[Hashable]:
    """The shutdowns a component is worked on at (its maintenance or
    tests bound to them)."""
    out = []
    for schedule in (rbd._preventive.get(node), rbd._inspection.get(node)):
        name = getattr(schedule, "shutdown", None)
        if name is not None:
            out.append(name)
    return out


def scope(rbd, shut: _Shutdown) -> Tuple[Hashable, ...]:
    """The components a shutdown holds off line: its scope, or every
    component of the diagram."""
    if shut.scope is None:
        return tuple(rbd.components)
    return tuple(shut.scope)


def holding(rbd, node) -> List[Tuple[_Shutdown, bool]]:
    """The shutdowns that take a time and hold ``node`` off line, each
    with whether it stands the node still (it is idle there and the shut
    pauses its idle blocks)."""
    out = []
    for shut in rbd._shutdowns.values():
        if shut.duration > 0.0 and node in scope(rbd, shut):
            paused = not shut.idle_ageing and shut.name not in bound_to(
                rbd, node
            )
            out.append((shut, paused))
    return out


def held_nodes(rbd) -> Dict[Hashable, List[Tuple[_Shutdown, bool]]]:
    """Every component a shutdown holds off line (see ``holding``)."""
    out = {}
    for node in rbd.components:
        shuts = holding(rbd, node)
        if shuts:
            out[node] = shuts
    return out


def timed(rbd) -> List[_Shutdown]:
    """The shutdowns that take a time (a window that holds nodes)."""
    return [s for s in rbd._shutdowns.values() if s.duration > 0.0]


def started_before(rbd, ends: np.ndarray) -> np.ndarray:
    """How many shuts start before each of ``ends`` (``[0, end)``), of
    every shutdown: they are planned, so the count is exact."""
    ends = np.asarray(ends, dtype=float)
    total = np.zeros(ends.shape)
    if not rbd._shutdowns or not ends.size:
        return total
    horizon = float(ends.max())
    for shut in rbd._shutdowns.values():
        starts = shut.starts(horizon)
        total = total + np.searchsorted(starts, ends, side="left")
    return total


def cost_before(rbd, ends: np.ndarray, rate: float = 0.0) -> np.ndarray:
    """The shuts' costs before each of ``ends``, each at its start,
    discounted at ``rate``."""
    ends = np.asarray(ends, dtype=float)
    total = np.zeros(ends.shape)
    if not ends.size:
        return total
    horizon = float(ends.max())
    for shut in rbd._shutdowns.values():
        if not shut.cost:
            continue
        starts = shut.starts(horizon)
        paid = shut.cost * (np.exp(-rate * starts) if rate else 1.0)
        running = np.concatenate(
            ([0.0], np.cumsum(paid * np.ones_like(starts)))
        )
        total = total + running[np.searchsorted(starts, ends, side="left")]
    return total


def cost_rate(rbd) -> float:
    """The shuts' cost per unit time in the long run."""
    return float(
        sum(
            shut.cost / shut.period
            for shut in rbd._shutdowns.values()
            if shut.cost and shut.periodic
        )
    )


def running_share(rbd, node) -> float:
    """The share of the time a component's own process runs in the long
    run: less the shuts that stand it still."""
    share = 1.0
    for shut, still in holding(rbd, node):
        if still:
            share -= shut.duration / shut.period
    return share


# -- the windows, at any time -------------------------------------------------


def _first_k(shut: _Shutdown) -> float:
    return 0.0 if shut.offset else 1.0


def _containing(shut: _Shutdown, x: np.ndarray):
    """For each time ``x``, the start of the last shut of ``shut`` at or
    before it (NaN if none), worked out as ``_Shutdown.starts`` works
    them out."""
    x = np.asarray(x, dtype=float)
    if shut.times is not None:
        times = np.asarray(shut.times, dtype=float)
        k = np.searchsorted(times, x, side="right") - 1
        out = np.full(x.shape, np.nan)
        ok = k >= 0
        out[ok] = times[k[ok]]
        return out
    k = np.floor((x - shut.offset) / shut.period)
    start = shut.offset + k * float(shut.period)
    # Rounding can put k one off the start at or before x.
    later = start > x
    k = np.where(later, k - 1.0, k)
    following = shut.offset + (k + 1.0) * float(shut.period)
    k = np.where(following <= x, k + 1.0, k)
    start = shut.offset + k * float(shut.period)
    return np.where(k >= _first_k(shut), start, np.nan)


def inside(shut: _Shutdown, x: np.ndarray) -> np.ndarray:
    """Whether each time ``x`` falls in a window of ``shut``,
    ``[start, start + duration)``."""
    start = _containing(shut, x)
    with np.errstate(invalid="ignore"):
        return ~np.isnan(start) & (np.asarray(x) < start + shut.duration)


def inside_long_run(shut: _Shutdown, x: np.ndarray) -> np.ndarray:
    """``inside``, in the long run: on the shut's calendar repeating from
    long before 0, so that a window at the period's start counts."""
    phase = np.asarray(x, dtype=float) - shut.offset
    phase = phase - shut.period * np.floor(phase / shut.period)
    return phase < shut.duration


def _elapsed(shut: _Shutdown, x: np.ndarray) -> np.ndarray:
    """The time of ``shut``'s windows before each time ``x``."""
    x = np.asarray(x, dtype=float)
    start = _containing(shut, x)
    if shut.times is not None:
        times = np.asarray(shut.times, dtype=float)
        whole = np.searchsorted(times, x, side="right") - 1.0
    else:
        k = np.round((start - shut.offset) / shut.period)
        k = np.where(np.isnan(k), 0.0, k)
        whole = k - _first_k(shut)
    whole = np.maximum(whole, 0.0)
    # The windows before the last one, whole, and the last as far as x.
    known = ~np.isnan(start)
    within = np.where(
        known, np.minimum(x - np.where(known, start, 0.0), shut.duration), 0.0
    )
    return np.where(known, whole * shut.duration + within, 0.0)


# -- over time ----------------------------------------------------------------


class MaskedCurve:
    """A component's point availability with the shutdowns that hold it
    off line: its own ``curve``, but 0 inside their windows; a shut that
    stands it still (``paused``) shifts its own process by its windows'
    time, so that it is ``curve`` at its own time, ``tau(x)``, the time
    less the windows that stood it still before ``x``.

    Its events are its own (failures while it is idle, ageing, are
    failures), at its own time; a shut's start adds a planned outage, the
    probability that it is up just before (after its own events then), as
    an event at an exact time (see ``atoms``). Its own events inside a
    window take nothing from the system, which it is held out of, and are
    not given as events at exact times. ``held`` says where it is held,
    for the system's Birnbaum importances (see ``held_importance``)."""

    def __init__(self, curve, shuts: List[Tuple[_Shutdown, bool]]):
        self.curve = curve
        self.shuts = shuts
        self.still = [shut for shut, paused in shuts if paused]
        periods = {shut.period for shut, _ in shuts}
        if curve.period is not None:
            periods.add(curve.period)
        try:
            self.period = (
                _common_period(periods) if None not in periods else None
            )
        except NotImplementedError:
            self.period = None
        self.settle = float(self._at_own(float(curve.settle)))
        if self.period is None and curve.period is not None:
            self.settle = math.inf
        if None in periods:
            # At given times: settles after the last.
            last = max(
                float(shut.times[-1]) + shut.duration  # type: ignore
                for shut, _ in shuts
                if shut.times is not None
            )
            self.settle = max(self.settle, last)
            self.period = curve.period

    # The time map of a paused node.

    def tau(self, x) -> np.ndarray:
        """Its own time at each time ``x``."""
        x = np.asarray(x, dtype=float)
        out = x.copy()
        for shut in self.still:
            out = out - _elapsed(shut, x)
        return out

    def _at_own(self, u: float) -> float:
        """The first time at which its own time reaches ``u``."""
        if not self.still or not math.isfinite(u):
            return u
        x = u
        for _ in range(64):
            x_next = u + sum(
                float(_elapsed(s, np.array([x]))[0]) for s in self.still
            )
            if x_next <= x:
                break
            x = x_next
        return x

    def _from_own(self, u: np.ndarray, stop: float) -> np.ndarray:
        """The times at which its own time is each of ``u`` (each at most
        its own time at ``stop``): a time its own clock stands still at
        is the start of that window."""
        u = np.asarray(u, dtype=float)
        if not self.still or not u.size:
            return u
        starts, ends = self._windows(stop, still=True)
        if not starts.size:
            return u
        before = np.concatenate(([0.0], np.cumsum(ends - starts)))[:-1]
        # Each window's start in its own time.
        own_starts = starts - before
        k = np.searchsorted(own_starts, u, side="right")
        shift = np.concatenate(([0.0], np.cumsum(ends - starts)))[k]
        at_start = (k > 0) & (u == own_starts[np.maximum(k - 1, 0)])
        out = u + shift
        out[at_start] = starts[k[at_start] - 1]
        return out

    def _windows(
        self, stop: float, still: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Its windows that start before ``stop`` (only those that stand it
        still, with ``still``), in order."""
        starts, ends = [], []
        for shut, paused in self.shuts:
            if still and not paused:
                continue
            a, b = shut.windows(stop)
            starts.append(a)
            ends.append(b)
        if not starts:
            return np.empty(0), np.empty(0)
        a, b = np.concatenate(starts), np.concatenate(ends)
        order = np.argsort(a, kind="stable")
        return a[order], b[order]

    def held(self, x) -> np.ndarray:
        """Whether it is held off line at each time ``x``."""
        x = np.asarray(x, dtype=float)
        out = np.zeros(x.shape, dtype=bool)
        for shut, _ in self.shuts:
            out |= inside(shut, x)
        return out

    def at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        values = np.asarray(self.curve.at(self.tau(x)), dtype=float)
        return np.where(self.held(x), 0.0, values)

    def down_at(self, x: np.ndarray, own: Optional[Callable] = None):
        """Its probability of being down at each time ``x``: 1 inside a
        window, else its own (``own(tau)`` where it has a closed form,
        else one less its curve)."""
        x = np.asarray(x, dtype=float)
        u = self.tau(x)
        down = None if own is None else own(u)
        if down is None:
            down = 1.0 - np.asarray(self.curve.at(u), dtype=float)
        return np.where(self.held(x), 1.0, down)

    def failed_at(self, x: np.ndarray) -> np.ndarray:
        """Its probability of having failed at each time ``x`` (see
        ``_unplanned``): its own, at its own time, whether or not a shut
        holds it off line (held, but not failed, it is down for planned
        work)."""
        from repyability.rbd._point_availability import failed_at

        x = np.asarray(x, dtype=float)
        return failed_at(self.curve, self.tau(x))

    def derivative(self, x: np.ndarray, scale: float) -> np.ndarray:
        """Its rate of change at each ``x``: its own at its own time, 0
        inside a window (the jumps at the windows' edges are not
        rates)."""
        from repyability.rbd._rates import derivative

        x = np.asarray(x, dtype=float)
        rate = derivative(self.curve, self.tau(x), scale)
        return np.where(self.held(x), 0.0, rate)

    def _starts(self, stop: float) -> np.ndarray:
        parts = [shut.windows(stop)[0] for shut, _ in self.shuts]
        return np.sort(np.concatenate(parts)) if parts else np.empty(0)

    def _own_atoms(self, stop: float):
        """Its own events at exact times before ``stop``, at their times
        (a paused node's from its own time)."""
        own = self.curve.atoms(float(self.tau(np.array([stop]))[0]))
        return own, self._from_own(own.times, stop)

    def _extras(self, stop: float) -> Tuple[np.ndarray, np.ndarray]:
        """Its shuts' starts before ``stop``, and at each the probability
        that it is up there after its own events: what the shut takes
        down. That is its value just before (a hair before, its own curve
        being continuous but at its events) less its own events' fall
        there, worked out so, not as its value at the start, which a
        start a rounding off its own calendar's time could place on the
        wrong side of its own events."""
        starts = self._starts(stop)
        if not starts.size:
            return starts, np.empty(0)
        hair = 1e-6 * min(shut.duration for shut, _ in self.shuts)
        before = np.asarray(
            self.curve.at(self.tau(starts) - hair), dtype=float
        )
        own, times = self._own_atoms(np.nextafter(starts[-1], np.inf))
        fall = _matched(starts, times, own.drop)
        return starts, np.clip(before - fall, 0.0, 1.0)

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Its expected events before each time ``x``: its own (at its own
        time), and as planned outages the shuts that take it down."""
        x = np.asarray(x, dtype=float)
        out = dict(self.curve.events(self.tau(x)))
        stop = float(x.max()) if x.size else 0.0
        starts, extra = self._extras(stop)
        if starts.size:
            running = np.concatenate(([0.0], np.cumsum(extra)))
            before = running[np.searchsorted(starts, x, side="left")]
            out["planned"] = np.asarray(out["planned"], dtype=float) + before
        return out

    def atoms(self, stop: float) -> Atoms:
        own, times = self._own_atoms(stop)
        starts, extra = self._extras(stop)
        # Its own events inside a window take nothing from the system,
        # but those at a window's start (a task bound to the shut) do.
        keep = (times < stop) & (~self.held(times) | np.isin(times, starts))
        zero = np.zeros(len(starts))
        return Atoms(
            np.concatenate([times[keep], starts]),
            np.concatenate([own.failure[keep], zero]),
            np.concatenate([own.planned[keep], extra]),
            np.concatenate([own.preventive[keep], zero]),
            np.concatenate([own.drop[keep], extra]),
        )

    def _edges(self, start: float, stop: float) -> np.ndarray:
        a, b = self._windows(stop + 1.0)
        edges = np.concatenate([a, b])
        return edges[(edges >= start) & (edges <= stop)]

    def _own_times(self, times: np.ndarray, start: float, stop: float):
        mapped = self._from_own(times, stop)
        return mapped[(mapped >= start) & (mapped <= stop)]

    def knots(self, start: float, stop: float) -> np.ndarray:
        a, b = self.tau(np.array([start, stop]))
        own = self._own_times(
            self.curve.knots(float(a), float(b)), start, stop
        )
        return np.concatenate([own, self._edges(start, stop)])

    def breaks(self, start: float, stop: float) -> np.ndarray:
        a, b = self.tau(np.array([start, stop]))
        own = self._own_times(
            curve_breaks(self.curve, float(a), float(b)), start, stop
        )
        return np.concatenate([own, self._edges(start, stop)])

    def grids(self) -> list:
        return [
            (step, self._at_own(until))
            for step, until in curve_grids(self.curve)
        ]


def masked_curves(rbd, curves: dict) -> dict:
    """``curves`` with each component a shutdown holds off line (and
    takes a time) a ``MaskedCurve``."""
    if not rbd._shutdowns:
        return curves
    out = dict(curves)
    for node, shuts in held_nodes(rbd).items():
        if node in out:
            out[node] = MaskedCurve(out[node], shuts)
    return out


def held_importance(curves: dict, importance: dict, x: np.ndarray) -> dict:
    """The nodes' Birnbaum ``importance`` at the times ``x``, 0 for a node
    held off line by a shut there: its own changes then change nothing."""
    out = importance
    for node, curve in curves.items():
        if isinstance(curve, MaskedCurve) and node in importance:
            held = curve.held(x)
            if held.any():
                if out is importance:
                    out = dict(importance)
                values = np.asarray(importance[node], dtype=float)
                out[node] = np.where(held, 0.0, values)
    return out


# -- in the long run ----------------------------------------------------------


def held_long_run(rbd, node, times: np.ndarray) -> np.ndarray:
    """Whether a shut holds ``node`` off line at each of ``times``, in the
    long run (on the calendars repeating from long before 0)."""
    times = np.asarray(times, dtype=float)
    out = np.zeros(times.shape, dtype=bool)
    for shut, _ in holding(rbd, node):
        out |= inside_long_run(shut, times)
    return out


def starting_long_run(rbd, node, times: np.ndarray, period: float):
    """Whether a shut holding ``node`` starts at each of ``times`` (on
    one common ``period``, to rounding)."""
    times = np.asarray(times, dtype=float)
    out = np.zeros(times.shape, dtype=bool)
    for shut, _ in holding(rbd, node):
        phase = (times - shut.offset) / shut.period
        out |= np.abs(phase - np.round(phase)) * shut.period <= 1e-9 * period
    return out


def mask(rbd, values: dict, times: np.ndarray, down: bool = False) -> dict:
    """Every node's long-run availability (with ``down``, unavailability)
    at each of ``times``, held at 0 (1) where a shut holds it."""
    if not rbd._shutdowns:
        return values
    out = dict(values)
    for node in held_nodes(rbd):
        if node in out:
            held = held_long_run(rbd, node, times)
            out[node] = np.where(held, 1.0 if down else 0.0, out[node])
    return out


def long_run_edges(rbd, period: float) -> list:
    """The edges of the shuts' windows over one common ``period`` of the
    long run's calendars."""
    parts = []
    for shut in timed(rbd):
        count = int(round(period / shut.period))
        starts = shut.offset + shut.period * np.arange(count)
        ends = starts + shut.duration
        for edges in (starts, ends):
            parts.append(edges - period * np.floor(edges / period))
    return parts


def long_run_starts(rbd, period: float) -> np.ndarray:
    """The shuts' starts (that take a time) over one common ``period``."""
    parts = [np.empty(0)]
    for shut in timed(rbd):
        count = int(round(period / shut.period))
        starts = shut.offset + shut.period * np.arange(count)
        parts.append(starts - period * np.floor(starts / period))
    return np.concatenate(parts)


# -- what the methods take ----------------------------------------------------


def _names(rbd) -> str:
    return ", ".join(repr(name) for name in rbd._shutdowns)


def require_none(rbd, what: str) -> None:
    """Raise if the diagram has shutdowns, which ``what`` does not take
    in yet."""
    if rbd._shutdowns:
        raise NotImplementedError(
            f"This diagram has shutdowns ({_names(rbd)}), which {what} "
            "does not take in yet. Work it out on the diagram without "
            "them, or estimate what the shutdowns do by simulation, with "
            "availability() or cost()."
        )


#: The public methods that do not take shutdowns in yet, which refuse a
#: diagram with any (see ``refuse``): the importance and sensitivity
#: measures over the components' values held, the capacities, the
#: first failure, the spares, the allocations, the parameter
#: uncertainty, the interval searches, the plan against the actual,
#: the repair crews' chain and stepping a simulation.
NOT_TAKEN = (
    "birnbaum_importance",
    "improvement_potential",
    "risk_achievement_worth",
    "risk_reduction_worth",
    "criticality_importance",
    "fussell_vesely",
    "differential_importance",
    "joint_importance",
    "barlow_proschan_importance",
    "availability_rate",
    "uncertainty_importance",
    "capacity_distribution",
    "point_capacity",
    "mission_capacity",
    "first_failure_sf",
    "mean_time_to_first_failure",
    "spares_demand",
    "spares_stock",
    "allocate_redundancy",
    "availability_allocation",
    "mttf_mttr_allocation",
    "mean_availability_uncertainty",
    "point_availability_uncertainty",
    "mission_availability_uncertainty",
    "expected_cost_rate_uncertainty",
    "optimal_replacement_intervals",
    "optimal_inspection_intervals",
    "plan_vs_actual",
    "crew_states",
)

#: What the exact methods say of a diagram's shutdowns (see the module
#: docstring).
HELD = (
    " The shutdowns' windows hold the blocks in their scope off line (a "
    "block that stands still in them is taken at its own time), and their "
    "starts are planned outages at exact times."
)

#: What stepping a simulation by hand is called in a refusal.
STEPPING = "stepping a simulation event by event"


def refuse(rbd, method: str) -> None:
    """Raise if the diagram has shutdowns, which the public ``method``
    (one of ``NOT_TAKEN``) does not take in yet."""
    require_none(rbd, f"{method}()")


def require_exact(rbd) -> None:
    """Raise if the exact methods cannot take the diagram's shutdowns:
    at given times (they take shuts on a period), or with what ties the
    blocks together (repair crews, common-cause groups, maintenance
    groups), or a nested RBD with a calendar of its own beside them."""
    if not rbd._shutdowns:
        return
    from repyability.rbd import _crews, _long_run
    from repyability.rbd.repairable_rbd import RepairableRBD

    given = [s.name for s in rbd._shutdowns.values() if not s.periodic]
    if given:
        raise NotImplementedError(
            f"Shutdown {given[0]!r} is at given times; the exact methods "
            "take shutdowns on a period. Estimate its values by "
            "simulation, with availability() or cost()."
        )
    tie = None
    if rbd._crews_couple() or _crews._crews_limited(rbd):
        tie = "repair crews"
    elif rbd.ccf_groups:
        tie = "common-cause groups"
    elif rbd._maintenance:
        tie = "maintenance groups"
    if tie is not None:
        raise NotImplementedError(
            f"This diagram has shutdowns ({_names(rbd)}) and {tie}, which "
            "tie its blocks together: the exact methods take shutdowns "
            "over blocks that fail and are repaired independently. "
            "Estimate its values by simulation, with availability() or "
            "cost()."
        )
    nested = [
        node
        for node, c in rbd.components.items()
        if isinstance(c, RepairableRBD) and _long_run._has_calendar(c)
    ]
    if nested:
        raise NotImplementedError(
            f"Node {nested[0]!r} is an RBD with block replacement or "
            "hidden failures, whose calendar the shutdowns here are not "
            "averaged with exactly: estimate the values by simulation, "
            "with availability() or cost()."
        )


def check_state(rbd, node, state) -> None:
    """Refuse a state that places the calendar of a task bound to a
    shutdown: the shut's calendar places it (given relative to now)."""
    if bound_to(rbd, node) and state.since_scheduled is not None:
        raise ValueError(
            f"Component {node!r} is worked on at shutdown "
            f"{bound_to(rbd, node)[0]!r}, whose calendar places its tasks "
            "(its offset or times, from now): leave its since_scheduled "
            "out."
        )
