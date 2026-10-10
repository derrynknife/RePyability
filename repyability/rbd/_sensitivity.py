"""Parameter sensitivity of a repairable diagram (#192): how its
availability (in the long run, at times or over a window) and its cost
rate move with each lever.

The levers are each component's life and repair models' parameters, its
scheduled replacement and its tests (their intervals, durations, coverage,
...), its standby group's, its imperfect repair's, a common-cause group's
model (its members' together, under the tuple of their names, and its
beta, ...), and the repair crews. A continuous lever's derivative is a
central difference of the system's own value with the lever moved either
way by ``rel_step`` of its value, the diagram rebuilt from its
constructor's arguments with the changed spec (see ``Lever``): one-sided
where one side is not a valid value (a probability leaving ``[0, 1]``),
NaN where neither is. A discrete lever (one more standby unit, one more
repair crew) reports the change one more makes.

Over time with independent components, a component's lever moves its own
curve alone, and the system's availability is linear in each component's
(the structure function is multilinear): its derivative at a time is the
component's Birnbaum importance then times the difference of its own point
availability, worked out on a diagram of the component alone; over a
window, the same summed on the window's quadrature points (see
``_importance_time.window_points``). That is exact and a fraction of the
cost of rebuilding the system, except for a lever that moves the curve's
breaks (a replacement or test interval, offset or threshold), which over
a window is differenced on the system, whose quadrature follows them.

The levers are public (#244): ``levers()`` lists them as ``Lever``
results, each with its value, its range and whether it is discrete or
moves a shared calendar, and ``with_levers`` builds the diagram with them
at other values, as the sensitivities rebuild it. A non-repairable
diagram's levers, its nodes' models' parameters and its common-cause
groups', are listed and set here too.
"""

import math
from collections.abc import Mapping
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np

from repyability.rbd import _curves, _long_run
from repyability.utils.checks import is_number, is_whole, number_or_nan

from . import _importance_time
from ._model_utils import CovariateSpec, lever_spec

#: What the sensitivities can be of.
QUANTITIES = ("availability", "cost_rate")
#: The maintenance options a lever moves, by spec key; those that move a
#: curve's breaks are differenced on the system over a window.
_PREVENTIVE = ("interval", "threshold", "opportunity")
_INSPECTION = ("interval", "coverage", "offset")
_MOVES_BREAKS = {
    "preventive.interval",
    "preventive.threshold",
    "preventive.opportunity",
    "inspection.interval",
    "inspection.offset",
}


#: The range of a probability, a fraction or a share.
_UNIT = (0.0, 1.0)


class _Lever(NamedTuple):
    """One lever: whose (a node, a common-cause group's members, or None
    for the system), its name, its value, whether it is discrete (one
    more), the constructor's arguments with it at a value (``build``),
    the node whose curve alone it moves (None if it moves more), the
    range of its values (see ``Lever``), and what a proportional change
    moves it by, per unit of the change (see ``_proportional_scale``)."""

    key: Hashable
    name: str
    value: float
    discrete: bool
    build: Callable[[float], dict]
    node: Optional[Hashable]
    bounds: Tuple[float, float]
    scale: Optional[float]


def _model_bounds(given) -> List[Tuple[float, float]]:
    """The range of each of a surpyval model's parameters, as its
    distribution bounds them (``from_params`` refuses a value outside),
    ``-inf`` and ``inf`` where it does not."""
    return [
        (
            -math.inf if low is None else float(low),
            math.inf if high is None else float(high),
        )
        for low, high in given
    ]


def _model_levers(prefix: str, model, rebuild) -> List[tuple]:
    """``(name, value, set, bounds, scale)`` for each parameter of a
    parametric ``model``, named ``prefix.<parameter>`` (or the parameter's
    own name, without a prefix), ``set(v)`` giving ``rebuild`` of the model
    with it at ``v``: a parametric model's parameters, or a regression
    node's covariates (#272); ``scale`` as ``_proportional_scale``."""
    spec = lever_spec(model)
    if spec is None:
        return []
    ranges = _model_bounds(spec.bounds)
    out = []
    for j, name in enumerate(spec.names):

        def moved(v, j=j):
            trial = list(spec.params)
            trial[j] = v
            return rebuild(spec.build(trial))

        named = f"{prefix}.{name}" if prefix else name
        scale = _proportional_scale(spec, name, spec.params[j])
        out.append((named, spec.params[j], moved, ranges[j], scale))
    return out


#: Parameters on a log scale, by distribution: the log of a time (a
#: LogNormal's ``mu``, Galton its other name) or of a rate (a Cox-Lewis
#: process's ``alpha``). Changing the unit of time adds to them, so they
#: have no zero to be a fraction of.
_LOG_SCALE = {"LogNormal": {"mu"}, "Galton": {"mu"}, "Cox-Lewis": {"alpha"}}


def _proportional_scale(spec, name: str, value: float) -> Optional[float]:
    """What a proportional change of a model's parameter moves it by, per
    unit of the change, so that it does not depend on the units of time
    or of a covariate: its value (a scale, a rate, a shape, a time); 1 for
    a parameter on a log scale (``_LOG_SCALE``), whose change moves the
    time it is the log of by that fraction; None for a regression node's
    covariate, whose zero is its unit's (0 degrees C is not 0 K), so that a
    proportional change of it means nothing."""

    if isinstance(spec, CovariateSpec):
        return None
    if name in _LOG_SCALE.get(getattr(spec.cls, "name", ""), ()):
        return 1.0
    return float(value)


def _option_bounds(key: str, option: str, options: dict) -> tuple:
    """The range of a maintenance option (as the constructor checks it):
    an interval past its offset or opportunity, an offset or opportunity
    within the interval, a coverage or threshold a probability."""
    if option == "interval":
        floor = number_or_nan(
            options.get("offset" if key == "inspection" else "opportunity")
        )
        return (0.0 if math.isnan(floor) else floor, math.inf)
    if option in ("offset", "opportunity"):
        return (0.0, float(options["interval"]))
    return _UNIT


def _spec_levers(spec: dict) -> List[tuple]:
    """``(name, value, set, discrete, bounds)`` for each lever of a
    component's spec dict, ``set(v)`` giving the spec with it at ``v``."""
    out: List[tuple] = []
    for key in ("reliability", "repairability"):
        model = spec.get(key)
        if model is None or isinstance(model, str):
            continue
        for name, value, moved, bounds, scale in _model_levers(
            key, model, lambda m, key=key: {**spec, key: m}
        ):
            out.append((name, value, moved, False, bounds, scale))
    for key, numeric in (
        ("preventive", _PREVENTIVE),
        ("inspection", _INSPECTION),
    ):
        options = spec.get(key)
        if not isinstance(options, dict):
            continue
        for option in numeric:
            value = options.get(option)
            if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)
            ):
                continue

            def moved(v, key=key, option=option, options=options):
                changed = {**options, option: v}
                full = options.get("full_test")
                if option == "interval" and full is not None:
                    # A full test every so many tests, as before.
                    changed["full_test"] = full * (v / options["interval"])
                return {**spec, key: changed}

            out.append(
                (
                    f"{key}.{option}",
                    float(value),
                    moved,
                    False,
                    _option_bounds(key, option, options),
                    float(value),
                )
            )
        duration = options.get("duration")
        if duration is not None and not isinstance(duration, str):
            for name, value, moved, bounds, scale in _model_levers(
                f"{key}.duration",
                duration,
                lambda m, key=key, options=options: {
                    **spec,
                    key: {**options, "duration": m},
                },
            ):
                out.append((name, value, moved, False, bounds, scale))
    standby = spec.get("standby")
    if isinstance(standby, dict):
        for option in ("dormancy_factor", "switching_probability"):
            if option in standby:

                def moved(v, option=option):
                    return {**spec, "standby": {**standby, option: v}}

                out.append(
                    (
                        f"standby.{option}",
                        float(standby[option]),
                        moved,
                        False,
                        _UNIT,
                        float(standby[option]),
                    )
                )

        def more(v):
            return {**spec, "standby": {**standby, "units": int(v)}}

        # A group needs a spare: more units than must work.
        spare = (float(standby.get("k", 1)) + 1.0, math.inf)
        out.append(
            (
                "standby.units",
                float(standby.get("units", 2)),
                more,
                True,
                spare,
                float(standby.get("units", 2)),
            )
        )
    repair = spec.get("repair")
    if isinstance(repair, dict) and "q" in repair:

        def moved_q(v):
            return {**spec, "repair": {**repair, "q": v}}

        q = float(repair["q"])
        out.append(("repair.q", q, moved_q, False, _UNIT, q))
    return out


def _as_spec(component) -> Optional[dict]:
    """A component as a spec dict, or None if it has no levers here (a
    nested RBD, a junction)."""
    if isinstance(component, dict):
        return component
    reliability = getattr(component, "reliability", None)
    repair = getattr(component, "time_to_replace", None)
    if reliability is None or repair is None:
        return None
    return {"reliability": reliability, "repairability": repair}


def _levers(rbd) -> List[_Lever]:
    """Every lever of a repairable ``rbd`` (see ``_Lever``), in the order
    of its components, then its common-cause groups and its repair
    crews."""
    from .ccf import CCFGroup
    from .ccf import parameters as ccf_parameters
    from .ccf import with_parameters as with_ccf_parameters

    args = rbd._init_args
    components = dict(args["components"])
    groups = list(args.get("ccf_groups") or [])
    members = {m for group in groups for m in group.members}
    out: List[_Lever] = []
    for node, component in components.items():
        if node in members:
            continue
        spec = _as_spec(component)
        if spec is None:
            continue
        for name, value, moved, discrete, bounds, scale in _spec_levers(spec):

            def build(v, node=node, moved=moved):
                return {**args, "components": {**components, node: moved(v)}}

            out.append(
                _Lever(node, name, value, discrete, build, node, bounds, scale)
            )
    for g, group in enumerate(groups):
        names = tuple(group.members)
        spec = _as_spec(components[names[0]])
        if spec is not None:
            for name, value, moved, discrete, bounds, scale in _spec_levers(
                spec
            ):

                def build_group(v, moved=moved, names=names):
                    changed = dict(components)
                    for member in names:
                        changed[member] = moved(v)
                    return {**args, "components": changed}

                out.append(
                    _Lever(
                        names,
                        name,
                        value,
                        discrete,
                        build_group,
                        None,
                        bounds,
                        scale,
                    )
                )
        for letter, value in ccf_parameters(group.model).items():

            def build_ccf(v, g=g, letter=letter, group=group):
                model = with_ccf_parameters(group.model, {letter: v})
                changed = list(groups)
                changed[g] = CCFGroup(group.members, model)
                return {**args, "ccf_groups": changed}

            out.append(
                _Lever(
                    names,
                    f"ccf_{letter}",
                    value,
                    False,
                    build_ccf,
                    None,
                    _UNIT,
                    value,
                )
            )
    crews = args.get("repair_crews")
    if crews is not None:

        def more_crews(v):
            return {**args, "repair_crews": int(v)}

        out.append(
            _Lever(
                None,
                "repair_crews",
                float(crews),
                True,
                more_crews,
                None,
                (1.0, math.inf),
                float(crews),
            )
        )
    return out


def _nonrepairable_levers(rbd) -> List[_Lever]:
    """Every lever of a non-repairable ``rbd``, as its
    ``parameter_sensitivity`` reports them: each node's model's parameters,
    by their own names, and a common-cause group's (its members' one
    model's, then its own, ``ccf_beta``, ...) under the tuple of its
    members, at its first member's place, in the order of the nodes."""
    from .ccf import CCFGroup
    from .ccf import parameters as ccf_parameters
    from .ccf import with_parameters as with_ccf_parameters

    args = rbd._init_args
    models = dict(args["reliabilities"])
    groups = list(rbd.ccf_groups)
    group_of = {m: g for g, group in enumerate(groups) for m in group.members}
    out: List[_Lever] = []
    for node, model in rbd.reliabilities.items():
        g = group_of.get(node)
        if g is not None and node != groups[g].members[0]:
            continue
        members = (node,) if g is None else tuple(groups[g].members)
        key = node if g is None else members
        for name, value, moved, bounds, scale in _model_levers(
            "", model, lambda m: m
        ):

            def build(v, moved=moved, members=members):
                changed = moved(v)
                return {
                    **args,
                    "reliabilities": {
                        **models,
                        **{member: changed for member in members},
                    },
                }

            out.append(
                _Lever(
                    key,
                    name,
                    value,
                    False,
                    build,
                    node if g is None else None,
                    bounds,
                    scale,
                )
            )
        if g is None:
            continue
        group = groups[g]
        for letter, value in ccf_parameters(group.model).items():

            def build_ccf(v, g=g, letter=letter, group=group):
                model = with_ccf_parameters(group.model, {letter: v})
                changed = list(groups)
                changed[g] = CCFGroup(group.members, model)
                return {**args, "ccf_groups": changed}

            out.append(
                _Lever(
                    key,
                    f"ccf_{letter}",
                    float(value),
                    False,
                    build_ccf,
                    None,
                    _UNIT,
                    float(value),
                )
            )
    return out


def _levers_of(rbd) -> List[_Lever]:
    """``rbd``'s levers, repairable or not."""
    from .non_repairable_rbd import NonRepairableRBD

    if isinstance(rbd, NonRepairableRBD):
        return _nonrepairable_levers(rbd)
    return _levers(rbd)


def public_levers(rbd) -> list:
    """``rbd``'s levers as the ``Lever`` results its ``levers()`` gives."""
    from .results import Lever

    return [
        Lever(
            key=lever.key,
            name=lever.name,
            value=lever.value,
            discrete=lever.discrete,
            bounds=lever.bounds,
            calendar=_calendar_lever(rbd, lever),
        )
        for lever in _levers_of(rbd)
    ]


class _Asked(NamedTuple):
    """What is asked: the times ``x`` or the ``window`` (or neither: the
    long run), from ``state``, with the nodes held."""

    x: Any
    window: Optional[float]
    state: Any
    working: set
    broken: set


def _value(rbd, quantity: str, asked: _Asked) -> np.ndarray:
    """``rbd``'s value of ``quantity``: its unavailability (whose
    derivative is the availability's, negated; in the long run worked out
    in its own right, so a small one keeps its precision), or its cost per
    unit time."""
    working, broken = asked.working, asked.broken
    if quantity == "availability":
        if asked.x is not None:
            value = 1.0 - np.asarray(
                rbd.point_availability(
                    asked.x, working, broken, state=asked.state
                ),
                dtype=float,
            )
        elif asked.window is not None:
            value = 1.0 - np.asarray(
                rbd.mission_availability(
                    asked.window, working, broken, state=asked.state
                ),
                dtype=float,
            )
        else:
            value = rbd.mean_unavailability(working, broken)
    elif asked.window is not None:
        cost = rbd.expected_cost(
            asked.window, working, broken, state=asked.state
        )
        value = np.asarray(cost.mean, dtype=float) / asked.window
    else:
        value = rbd.expected_cost_rate(working, broken)
    return np.atleast_1d(np.asarray(value, dtype=float))


def _difference(at: Callable, lever: _Lever, rel_step: float, base):
    """The lever's derivative (or, discrete, its change for one more) of
    ``at(value)`` (None where the value is not valid), from ``base``."""
    theta = lever.value
    if lever.discrete:
        more = at(theta + 1.0)
        return np.full_like(base, np.nan) if more is None else more - base
    h = rel_step * abs(theta) if theta != 0.0 else rel_step
    up, down = at(theta + h), at(theta - h)
    if up is not None and down is not None:
        return (up - down) / (2.0 * h)
    if up is not None:
        return (up - base) / h
    if down is not None:
        return (base - down) / h
    return np.full_like(base, np.nan)


def _rebuilt(rbd, lever: _Lever, value: float):
    """The diagram with the lever at ``value``, or None if that is not a
    valid value."""
    try:
        return type(rbd)(**lever.build(value))
    except (ValueError, TypeError):
        return None


def _system_at(rbd, lever: _Lever, quantities, asked: _Asked):
    """``at(value)`` for each quantity: the rebuilt system's value."""
    cache: Dict[float, Optional[dict]] = {}

    def values(v):
        if v not in cache:
            changed = _rebuilt(rbd, lever, v)
            if changed is None:
                cache[v] = None
            else:
                try:
                    cache[v] = {
                        q: _value(changed, q, asked) for q in quantities
                    }
                except (ValueError, NotImplementedError):
                    cache[v] = None
        return cache[v]

    def at(quantity):
        def one(v):
            found = values(v)
            return None if found is None else found[quantity]

        return one

    return at


def _calendar_lever(rbd, lever: _Lever) -> bool:
    """Whether the lever moves the period of its component's calendar (a
    block replacement's or a test's interval) that others share: the
    long-run values jump as it leaves their common calendar, and are
    averaged over a common period that it would make too long to follow
    (see ``_common._common_period``)."""
    if lever.name not in ("preventive.interval", "inspection.interval"):
        return False
    members = set(lever.key) if isinstance(lever.key, tuple) else {lever.key}
    calendar = set(_long_run._block_nodes(rbd)) | set(rbd._inspection)
    return bool(members & calendar) and bool(calendar - members)


def _apart_at(rbd, lever: _Lever, quantity: str, importance: float):
    """In the long run, ``at(value)`` for a calendar lever (see
    ``_calendar_lever``) with the component's schedule taken apart from
    the others': its long-run Birnbaum importance (``importance``) times
    its own long-run unavailability, the part of the system's that moves
    with it (the structure function is linear in each component's, and
    the others' schedules average out against it), from a diagram of the
    component alone; for the cost rate, its own cost rate (its repairs,
    replacements, tests and down time) plus the system's downtime cost of
    that. None for a value that is not valid, or a component in a
    maintenance group (whose set-ups it shares) for the cost."""
    node = lever.node
    rate = float(rbd.downtime_cost_rate or 0.0)

    def one(v):
        args = lever.build(v)
        spec = args["components"][node]
        if quantity == "cost_rate" and isinstance(spec, dict):
            if spec.get("group") is not None:
                return None
        try:
            alone = type(rbd)(
                [(rbd.input_node, node), (node, rbd.output_node)],
                {node: spec},
                input_node=rbd.input_node,
                output_node=rbd.output_node,
                repair_crews=args.get("repair_crews"),
            )
            part = importance * float(alone.mean_unavailability())
            if quantity == "cost_rate":
                part = float(alone.expected_cost_rate()) + rate * part
        except (ValueError, TypeError, NotImplementedError):
            return None
        return np.atleast_1d(part)

    return one


class _Alone:
    """Over time with independent components, a component's own point
    availability with a lever at a value, from a diagram of it alone (with
    the system's repair crews, which a standby group's units share), at
    the times asked or the window's quadrature points; and each
    component's Birnbaum importance there."""

    def __init__(self, rbd, asked: _Asked):
        self.rbd = rbd
        self.asked = asked
        if asked.x is not None:
            self.times = np.atleast_1d(np.asarray(asked.x, dtype=float))
            self.weights = None
            shape = self.times.shape
            self.times = self.times.ravel()
        else:
            self.times, self.weights = self._window_points()
            shape = (1,)
        self.shape = shape
        self.birnbaum = rbd.birnbaum_importance(
            asked.working,
            asked.broken,
            x=self.times,
            state=asked.state,
        )

    def _window_points(self):
        from ._curves import _MISSION_POINTS

        rbd, asked = self.rbd, self.asked
        assert asked.window is not None
        forced = asked.working | asked.broken
        states = _curves._states(rbd, asked.state, forced)
        curves = _curves._availability_curves(
            rbd, asked.window, forced, state=states
        )

        def integrands(x):
            p, q = _importance_time._independent(
                rbd, curves, x, asked.working, asked.broken
            )
            importance, works, _, _, _ = rbd._importances(p, q)
            return {
                "up": works,
                **{("b", n): v for n, v in importance.items()},
            }

        return _importance_time.window_points(
            list(curves.values()), asked.window, integrands, _MISSION_POINTS
        )

    def at(self, lever: _Lever) -> Callable:
        rbd, node = self.rbd, lever.node
        given = self.asked.state
        if isinstance(given, dict):
            given = {node: given[node]} if node in given else None
        importance = np.asarray(self.birnbaum[node], dtype=float).ravel()

        def one(v):
            args = lever.build(v)
            try:
                alone = type(rbd)(
                    [(rbd.input_node, node), (node, rbd.output_node)],
                    {node: args["components"][node]},
                    input_node=rbd.input_node,
                    output_node=rbd.output_node,
                    repair_crews=args.get("repair_crews"),
                )
                curve = np.asarray(
                    alone.point_availability(self.times, state=given),
                    dtype=float,
                ).ravel()
            except (ValueError, TypeError, NotImplementedError):
                return None
            # The system's unavailability moves by minus its Birnbaum
            # importance times the node's availability's move.
            moved = -importance * curve
            if self.weights is None:
                return moved.reshape(self.shape)
            assert self.asked.window is not None
            return np.atleast_1d(
                (self.weights @ moved) / float(self.asked.window)
            )

        return one


def sensitivity(
    rbd,
    *,
    x,
    window,
    state,
    working_nodes,
    broken_nodes,
    rel_step: Optional[float],
    of,
    unit_costs: Optional[dict],
) -> dict:
    """``RepairableRBD.parameter_sensitivity``: see there."""
    from repyability.utils.checks import nonnegative_times

    quantities = (of,) if isinstance(of, str) else tuple(of)
    if not quantities or any(q not in QUANTITIES for q in quantities):
        raise ValueError(
            f"of must be 'availability', 'cost_rate' or a tuple of them, "
            f"got {of!r}."
        )
    if x is not None and window is not None:
        raise ValueError(
            "Give the times x or a window's length, not both: x evaluates "
            "the sensitivity at each time, window over [0, window)."
        )
    if x is not None and "cost_rate" in quantities:
        raise ValueError(
            "The cost rate's sensitivity is of its long-run value, or of the "
            "mean cost per unit time over a window: give window, not x."
        )
    if state is not None and x is None and window is None:
        raise ValueError(
            "state is where the components start from, for a sensitivity at "
            "times (x) or over a window (window): give one of them. The "
            "long-run sensitivity does not depend on it."
        )
    timed = x is not None or window is not None
    if rel_step is None:
        # The curves over time are numerical: a small step would difference
        # their error. The long-run values are exact, or numerical to
        # about 1e-10.
        rel_step = 1e-2 if timed else 1e-5
    if not (
        isinstance(rel_step, (int, float, np.integer, np.floating))
        and not isinstance(rel_step, bool)
        and math.isfinite(rel_step)
        and 0.0 < rel_step < 0.5
    ):
        raise ValueError(
            f"rel_step must be a number in (0, 0.5), got {rel_step!r}."
        )
    if x is not None:
        nonnegative_times(x)
    if window is not None:
        if (
            isinstance(window, bool)
            or not isinstance(window, (int, float, np.integer, np.floating))
            or not (math.isfinite(window) and window > 0.0)
        ):
            raise ValueError(
                f"window must be a positive, finite length, got {window!r}."
            )
        window = float(window)
    working = set(working_nodes or ())
    broken = set(broken_nodes or ())
    rbd._validate_node_overrides(working, broken)
    held = working | broken
    asked = _Asked(x, window, state, working, broken)
    found = _levers(rbd)
    if unit_costs is not None:
        known = {(lever.key, lever.name) for lever in found}
        unknown = [key for key in unit_costs if key not in known]
        if unknown:
            raise ValueError(
                f"unit_costs names {unknown!r}, which are not levers of "
                "this diagram: give them as (node, lever), the node as "
                "parameter_sensitivity reports it (None for repair_crews)."
            )
    base = {q: _value(rbd, q, asked) for q in quantities}
    alone = None
    if (
        timed
        and "availability" in quantities
        and not rbd._crews_couple()
        and not rbd.ccf_groups
    ):
        alone = _Alone(rbd, asked)
    scalar = x is None or np.ndim(x) == 0
    birnbaum: Dict[Any, float] = {}
    out: Dict[str, Dict[Any, Dict[str, Any]]] = {q: {} for q in quantities}
    for lever in found:
        if (
            unit_costs is not None
            and (lever.key, lever.name) not in unit_costs
        ):
            continue
        pinned = lever.key in held or (
            isinstance(lever.key, tuple) and bool(held & set(lever.key))
        )
        system = None
        for quantity in quantities:
            if pinned:
                change = np.zeros_like(base[quantity])
            elif not timed and _calendar_lever(rbd, lever):
                if lever.node is None:
                    # A common-cause group's: the groups' chains follow
                    # their members' schedules together with the others'.
                    change = np.full_like(base[quantity], np.nan)
                else:
                    if not birnbaum:
                        birnbaum.update(
                            rbd.birnbaum_importance(working, broken)
                        )
                    at = _apart_at(
                        rbd, lever, quantity, float(birnbaum[lever.node])
                    )
                    change = _difference(at, lever, rel_step, at(lever.value))
            elif (
                quantity == "availability"
                and alone is not None
                and lever.node is not None
                and (x is not None or lever.name not in _MOVES_BREAKS)
            ):
                at = alone.at(lever)
                change = _difference(at, lever, rel_step, at(lever.value))
            else:
                if system is None:
                    system = _system_at(rbd, lever, quantities, asked)
                change = _difference(
                    system(quantity), lever, rel_step, base[quantity]
                )
            if quantity == "availability":
                change = -change
            if unit_costs is not None:
                change = change / float(unit_costs[(lever.key, lever.name)])
            value = (
                float(np.ravel(change)[0])
                if scalar
                else np.asarray(change, dtype=float).reshape(np.shape(x))
            )
            out[quantity].setdefault(lever.key, {})[lever.name] = value
    if isinstance(of, str):
        return out[of]
    return out


def _named(lever) -> Tuple[Hashable, str]:
    """A lever given to ``with_levers``, by its ``Lever`` or its ``(key,
    name)``, as ``(key, name)``."""
    from . import results

    if isinstance(lever, results.Lever):
        return lever.key, lever.name
    if isinstance(lever, tuple) and len(lever) == 2:
        key, name = lever
        if isinstance(name, str):
            return key, name
    raise TypeError(
        "with_levers takes each lever by its Lever (from levers()) or as "
        f"(key, name), as parameter_sensitivity reports it, got {lever!r}."
    )


def _no_lever(key, name: str, found: dict) -> ValueError:
    """The error for a lever the diagram does not have, with the closest
    name if one is close (#232): of the levers of ``key``, or else of the
    keys."""
    from .rbd import _close_name

    names = [n for k, n in found if k == key]
    if names:
        close = _close_name(name, names)
        hint = "" if close is None else f" Did you mean {close!r}?"
        return ValueError(
            f"{key!r} has no lever {name!r}.{hint} Its levers are: {names}."
        )
    keys = list(dict.fromkeys(k for k, _ in found))
    close = _close_name(key, keys)
    hint = "" if close is None else f" Did you mean {close!r}?"
    return ValueError(
        f"{key!r} has no levers: it is no node with a model, common-cause "
        f"group or the repair crews (None).{hint} The keys with levers are: "
        f"{keys}."
    )


def with_levers(rbd, values) -> Any:
    """``RepairableRBD.with_levers`` and ``NonRepairableRBD.with_levers``:
    see there."""
    if not isinstance(values, Mapping):
        raise TypeError(
            "with_levers takes a dict of new values, by Lever (from "
            "levers()) or by (key, name) as parameter_sensitivity reports "
            f"them, got {type(values).__name__}."
        )
    changes = [(_named(lever), value) for lever, value in values.items()]
    changed = rbd
    for (key, name), value in changes:
        found = {
            (lever.key, lever.name): lever for lever in _levers_of(changed)
        }
        if (key, name) not in found:
            raise _no_lever(key, name, found)
        lever = found[(key, name)]
        if not is_number(value):
            raise TypeError(
                f"The lever {name!r} of {key!r} must be given a number, got "
                f"{value!r}."
            )
        if lever.discrete and not is_whole(value):
            raise ValueError(
                f"The lever {name!r} of {key!r} counts, so its value is a "
                f"whole number, got {value!r}."
            )
        try:
            changed = type(rbd)(**lever.build(float(value)))
        except (ValueError, TypeError) as error:
            low, high = lever.bounds
            raise ValueError(
                f"The lever {name!r} of {key!r} cannot be {value!r} (its "
                f"values lie between {low:g} and {high:g}): {error}"
            ) from error
    if changed is rbd:
        # A copy, as with no levers changed.
        changed = type(rbd)(**rbd._init_args)
    return changed


#: The names callers used before the levers were public (#244), kept as
#: they were for 0.13; from 0.14 they may change or go. Use ``levers()``,
#: ``Lever`` and ``with_levers``.
Lever = _Lever
levers = _levers
