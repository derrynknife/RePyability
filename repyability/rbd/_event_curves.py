"""Events that are repaired or tested, for a structure that is not drawn as
a block diagram: a fault tree's (and a network's, which can take them the
same way).

Each event is a component spec, as a ``RepairableRBD`` takes one: its
``"reliability"``, its ``"repairability"`` and, for a hidden failure found
by periodic tests, its ``"inspection"``. Its curves come from the
repairable machinery with no renewal code of its own: the events are the
components of one ``RepairableRBD`` (``Structured``) whose structure is
the owner's decomposition (a tree's decision diagram, repeated events and
VOTE gates included), so that the diagram's values over time
(``point_unavailability``, ``mission_unavailability``,
``expected_failures``) and in the long run (``mean_unavailability``,
``system_failure_frequency``) are the owner's. Without common-cause
groups the events are independent (no crews couple them), so the
system's failure frequency at a time is ``sum_i I_B^i(t) w_i(t)``,
``w_i`` the event's failure intensity (``failing_at``).

A tree may mix them with events that are not repaired (#831): a fixed
probability ``q`` (down with it at every time, never failing over time:
``w = 0``), which the diagram adds to the structure's inputs wherever it
is evaluated (``_node_pairs``), so that every value over time and in the
long run takes it; and a life that is not repaired, down from its failure
on (``q(t) = F(t)``, ``w(t)`` its density), which follows a curve of its
own (``LifeCurve``) beside the components' and has no long run. With
common-cause groups over its repaired events (the diagram's own
``ccf_groups``), the diagram's values are its groups' chains' (see
``_ccf_groups``), conditioned on module by module over the structure;
so is the failure frequency at a time (``grouped_failure_frequency``),
and the owner's measures at a time condition on the groups' joint
states then (``grouped_at``).
"""

from typing import Any, Dict, Hashable, Mapping, Optional, Tuple

import numpy as np

from repyability.rbd import _curves, _rates, _windows
from repyability.rbd._common import _constant_rate
from repyability.rbd._model_utils import failure_time_scale
from repyability.rbd._point_availability import (
    Atoms,
    curve_breaks,
    curve_grids,
    knots,
)
from repyability.rbd.repairable_rbd import RepairableRBD

#: The keys an event's spec may have: its life, its repair, and its tests.
EVENT_SPEC_KEYS = ("reliability", "repairability", "inspection")


def checked_spec(name: Hashable, spec: Mapping, convert: str) -> dict:
    """An event's ``spec``, its keys checked (the values are checked as a
    diagram's component's are, when ``Structured`` is built). Any other
    key refuses, naming the diagram (``convert``) that takes it."""
    unknown = set(spec) - set(EVENT_SPEC_KEYS)
    if unknown:
        raise ValueError(
            f"Event {name!r} has key(s) {sorted(map(str, unknown))}: an "
            f"event takes {', '.join(EVENT_SPEC_KEYS)} only. For costs, "
            "preventive maintenance, standby units, spares and the rest, "
            f"convert the structure with {convert} and give its "
            "components the full spec."
        )
    return dict(spec)


class LifeCurve:
    """An event that is not repaired, as a curve (see
    ``_point_availability``): up until its life ends and down from then
    on, ``sf(t)``, its failures by ``t`` ``F(t)`` and its failure
    intensity its density. It bends between its life's quantiles
    (``knots``), which are its breaks, and is never constant (it has no
    long run)."""

    period = None
    settle = np.inf
    #: Its probability of being down is its own (``down_at``), so a small
    #: one keeps its digits (see ``_curves._nodes_down_at``).
    has_down = True

    def __init__(self, model):
        self.model = model
        self._knots = knots(model)

    def _values(self, f, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            values = np.asarray(f(x), dtype=float)
        return np.broadcast_to(values, x.shape).copy()

    def at(self, x: np.ndarray) -> np.ndarray:
        return self._values(self.model.sf, x)

    def down_at(self, x: np.ndarray) -> np.ndarray:
        return self._values(self.model.ff, x)

    failed_at = down_at

    def intensity(self, x: np.ndarray) -> np.ndarray:
        """Its failures per unit time at ``x``: its density."""
        return self._values(self.model.df, x)

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Its expected events before each time ``x``: one failure, by
        ``F(x)``, and nothing else (it is not repaired)."""
        zero = np.zeros(np.shape(x))
        return {
            "failures": self.down_at(x),
            "planned": zero,
            "corrective": zero,
            "preventive": zero,
            "inspections": zero,
        }

    def atoms(self, stop: float) -> Atoms:
        return Atoms.none()

    def breaks(self, start: float, stop: float) -> np.ndarray:
        """Its life's quantiles in ``[start, stop]``, between which it is
        smooth (see ``curve_breaks``)."""
        k = self._knots
        return k[(k >= start) & (k <= stop)]

    knots = breaks

    def grids(self) -> list:
        return []


class Structured(RepairableRBD):
    """A ``RepairableRBD`` of the events (in series, as drawn), evaluated
    over ``structure``: a decomposition over the events working, which
    stands for the diagram's own (see ``RBD._decomposition``). The
    structure's other inputs are the events that are not repaired:
    ``fixed``, each a probability of being down at every time, and
    ``lives``, each a life that is not repaired (a ``LifeCurve``). With
    ``ccf_groups``, over the repaired events, as a diagram takes them."""

    def __init__(
        self,
        specs: Mapping[Hashable, dict],
        structure: Any,
        fixed: Optional[Mapping[Hashable, float]] = None,
        lives: Optional[Mapping[Hashable, Any]] = None,
        ccf_groups: Optional[list] = None,
    ):
        self._fixed = dict(fixed or {})
        self._lives = {name: LifeCurve(m) for name, m in (lives or {}).items()}
        names = list(specs)
        taken = set(names)
        ends = []
        for base in ("input", "output"):
            end = base
            while end in taken:
                end = f"{end}'"
            taken.add(end)
            ends.append(end)
        nodes = [ends[0], *names, ends[1]]
        super().__init__(
            list(zip(nodes[:-1], nodes[1:])),
            dict(specs),
            input_node=ends[0],
            output_node=ends[1],
            ccf_groups=ccf_groups,
        )
        # Built as a series diagram (every event relevant, each checked
        # as a component), and evaluated over the structure from now on.
        self._structure = structure

    def _decomposition(self):
        structure = self.__dict__.get("_structure")
        return super()._decomposition() if structure is None else structure

    def _evaluated_nodes(self) -> list:
        """The components, and the lives that are not repaired, whose
        probabilities come with the components' (from their curves)."""
        return [*self.nodes, *self.__dict__.get("_lives", ())]

    def _unrepaired_curves(self) -> dict:
        """The curves of the lives that are not repaired (see
        ``_curves._availability_curves``)."""
        return dict(self._lives)

    def _node_arrays(self, node_probabilities: Dict) -> tuple:
        """``RBD._node_arrays``, with the fixed events' probabilities of
        not being down."""
        arrays, size = super()._node_arrays(node_probabilities)
        for event, q in self.__dict__.get("_fixed", {}).items():
            arrays[event] = np.full(size, 1.0 - q)
        return arrays, size

    def _node_pairs(
        self, node_probabilities: Dict, node_failures: Optional[Dict] = None
    ) -> tuple:
        """``RBD._node_pairs``, with each fixed event down with its
        probability exactly."""
        p, q, size = super()._node_pairs(node_probabilities, node_failures)
        for event, value in self.__dict__.get("_fixed", {}).items():
            q[event] = np.full(size, value)
        return p, q, size


def _curves_to(
    rbd: Structured, times: np.ndarray, counts: bool = False, past=1.0
):
    """Each event's curve over ``[0, past * max(times)]`` (counting its
    events, with ``counts``), and the states they start from (new)."""
    states = _curves._states(rbd, None, frozenset())
    horizon = past * float(times.max()) if times.size else 0.0
    curves = _curves._availability_curves(
        rbd, horizon, frozenset(), counts=counts, state=states
    )
    return curves, states


def down_at(rbd: Structured, times: np.ndarray) -> Dict[Hashable, Any]:
    """Each event's probability of being down at the ``times``, worked out
    in its own right (see ``_curves._nodes_down_at``)."""
    curves, states = _curves_to(rbd, times)
    return _curves._nodes_down_at(rbd, curves, times, states)


class _Failures:
    """A curve's expected failures since 0 as a curve of its own, on its
    grids and breaks, for ``_rates.differences``."""

    def __init__(self, curve):
        self.curve = curve

    def at(self, x: np.ndarray) -> np.ndarray:
        return self.curve.events(np.asarray(x, dtype=float))["failures"]

    def grids(self) -> list:
        return curve_grids(self.curve)

    def breaks(self, start: float, stop: float) -> np.ndarray:
        return curve_breaks(self.curve, start, stop)


def _typical_life(rbd: Structured) -> float:
    """The shortest typical life of the events (see
    ``failure_time_scale``), 1.0 if none has one."""
    scales = [
        failure_time_scale(component.reliability)
        for component in rbd.components.values()
    ] + [failure_time_scale(curve.model) for curve in rbd._lives.values()]
    scales = [x for x in scales if np.isfinite(x) and x > 0.0]
    return min(scales) if scales else 1.0


def failing_at(rbd: Structured, times: np.ndarray):
    """Each event's probability of being down and its failure intensity
    (its expected failures per unit time) at the ``times``. An event of
    constant failure rate ``lambda`` fails at ``lambda`` while it works
    (not under repair nor in a test), exactly ``lambda (1 - q(t))``; any
    other's intensity is the rate of change of its expected failures,
    by differences on its curve's grid (``_rates.differences``)."""
    # A little past the last time, for the differences there.
    curves, states = _curves_to(rbd, times, counts=True, past=1.01)
    down = _curves._nodes_down_at(rbd, curves, times, states)
    return down, _intensities(rbd, curves, down, times)


def _intensities(
    rbd: Structured, curves: dict, down: dict, times: np.ndarray
) -> Dict[Hashable, np.ndarray]:
    """The failure intensity at the ``times`` of each event of ``curves``
    (see ``failing_at``), ``down`` its probability of being down then."""
    horizon = float(times.max()) if times.size else 0.0
    scale = horizon if horizon > 0.0 else _typical_life(rbd)
    intensity = {}
    for node, curve in curves.items():
        if isinstance(curve, LifeCurve):
            # Not repaired: its density, exactly.
            intensity[node] = curve.intensity(times)
            continue
        rate = _constant_rate(rbd.components[node].reliability)
        if rate is not None:
            intensity[node] = rate * (1.0 - down[node])
        else:
            intensity[node] = _rates.differences(
                _Failures(curve), times, scale
            )
    return intensity


def _grouped(rbd: Structured, times: np.ndarray, counts=False, past=1.0):
    """With common-cause groups: the curves of the events outside them
    (to ``past * max(times)``, counting their events with ``counts``),
    their states (new), and the system over the groups' chains (see
    ``_ccf_chain.GroupsSystem``)."""
    from repyability.rbd import _ccf_groups

    horizon = past * float(times.max()) if times.size else 0.0
    grouped = _ccf_groups._groups_curve(rbd, horizon, counts=counts)
    states = _curves._states(rbd, None, frozenset())
    return grouped.curves, states, grouped.system


def grouped_at(rbd: Structured, times: np.ndarray) -> Tuple[dict, list]:
    """With common-cause groups, at the ``times``: each event's
    probability of being down (one outside the groups in its own right,
    see ``down_at``; a member's from its group's chain, summed over its
    combinations), and each group's members' joint states then (its
    ``_ccf_chain.GroupStates``), which ``_ccf_groups._ccf_tabled``
    conditions on."""
    from repyability.rbd._ccf_chain import GroupStates

    curves, states, system = _grouped(rbd, times)
    down = _curves._nodes_down_at(rbd, curves, times, states)
    tables = []
    for group in system.groups:
        chances = np.asarray(group.probabilities(times), dtype=float)
        tables.append(GroupStates(tuple(group.members), group.down, chances))
        for k, member in enumerate(group.members):
            down[member] = chances @ np.where(group.down[:, k], 1.0, 0.0)
    return down, tables


def failure_frequency(rbd: Structured, times: np.ndarray) -> np.ndarray:
    """The system's failure frequency at the ``times``: each event's
    failure intensity where it is critical, ``sum_i I_B^i(t) w_i(t)``,
    the Birnbaum importances at the events' availabilities then, from one
    pass of the structure (see ``RBD._importances``). With common-cause
    groups, see ``grouped_failure_frequency``."""
    if rbd.ccf_groups:
        return grouped_failure_frequency(rbd, times)
    down, intensity = failing_at(rbd, times)
    size = len(times)
    none: frozenset = frozenset()
    probabilities = _windows._filled(
        rbd, {node: 1.0 - q for node, q in down.items()}, size, none, none
    )
    failures = dict(down)
    for node in rbd.in_or_out:
        failures[node] = np.zeros(size)
    importance = rbd._importances(probabilities, failures)[0]
    total = np.zeros(size)
    for node, rate in intensity.items():
        total += importance[node] * rate
    return total


def grouped_failure_frequency(
    rbd: Structured, times: np.ndarray
) -> np.ndarray:
    """``failure_frequency`` with common-cause groups over the repaired
    events, as ``_windows._window_counts`` counts the diagram's failures
    through the groups' system (``_ccf_chain.GroupsSystem.evaluate``), at
    the ``times`` rather than over a window: each event outside the
    groups' failure intensity times its Birnbaum importance then, over
    the groups' joint states, and each of the groups' causes' rate times
    the rise in the system's unavailability its strike makes then (a
    member's own failures are its own cause's). Its integral is
    ``expected_failures``, less the failures at exact times."""
    curves, states, system = _grouped(rbd, times, counts=True, past=1.01)
    down = _curves._nodes_down_at(rbd, curves, times, states)
    intensity = _intensities(rbd, curves, down, times)
    values = {node: 1.0 - q for node, q in down.items()}
    importance, _, _, rate, _ = system.evaluate(values, times)
    total = np.array(np.broadcast_to(rate, times.shape), dtype=float)
    for node, value in intensity.items():
        total += importance[node] * value
    return total


def down_routes(rbd: Structured) -> Dict[Hashable, Tuple[str, str]]:
    """How each event's probability of being down at a time is worked
    out (see ``routes.with_nodes``): in closed form for an exponential
    life and repair, else from its curve, numerically."""
    from repyability.rbd import routes as r

    closed = _curves._closed_form_rates(rbd, None)
    out = {
        node: (
            (r.EXACT, "an exponential life and repair, in closed form")
            if node in closed
            else (
                r.NUMERICAL,
                "its renewal (or test) cycle's curve, on a grid",
            )
        )
        for node in rbd.components
    }
    for node in rbd._fixed:
        out[node] = (r.EXACT, "a fixed probability")
    for node in rbd._lives:
        out[node] = (r.EXACT, "not repaired: its life's F(t)")
    return out


def intensity_routes(
    rbd: Structured, name: str
) -> Dict[Hashable, Tuple[str, str]]:
    """How each event's failure intensity is worked out for the analysis
    ``name`` (see ``failing_at``): at a time, exactly for a constant
    failure rate whose probability of being down is exact, else by
    differences; in the long run, exactly; over a window, numerically."""
    from repyability.rbd import routes as r

    # A fixed probability never fails over time; a life that is not
    # repaired fails once, by F(t), at its density.
    lives = {
        node: (
            (r.EXACT, "not repaired: its life's F(t)")
            if name == "expected_failures"
            else (r.EXACT, "not repaired: its life's density")
        )
        for node in rbd._lives
    }
    if name == "system_failure_frequency":
        return {
            node: (r.EXACT, "its long-run failure frequency")
            for node in rbd.components
        }
    if name == "expected_failures":
        return {
            **{
                node: (r.NUMERICAL, "its renewal equation, on a grid")
                for node in rbd.components
            },
            **lives,
        }
    closed = _curves._closed_form_rates(rbd, None)
    return {
        **{
            node: (
                (r.EXACT, "its constant rate while it works")
                if node in closed
                else (
                    r.NUMERICAL,
                    "differences of its expected failures on its curve's "
                    "grid",
                )
            )
            for node in rbd.components
        },
        **lives,
    }
