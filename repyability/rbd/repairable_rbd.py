"""Repairable reliability block diagrams: the ``RepairableRBD`` class.

Besides the class, the module holds the helpers of its availability
simulation: the ``Event`` records of its event queue, the timeline
arithmetic that defines the up/down criticality measures
(``combined_timeline``, ``intersection``, ``union``, ``time_at_status``;
the simulation adds the same times up as it goes) and the failure and
restoration criticality index ratios. The simulation's random streams are
in ``_streams``, and its compiled engine in ``_compiled`` and
``_kernel``.
"""

import dataclasses
import hashlib
import heapq
import itertools
import json
import math
import pickle
import warnings
from collections import Counter, defaultdict, deque
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field
from fractions import Fraction
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    Iterable,
    List,
    NamedTuple,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy.optimize import OptimizeResult, brentq, minimize
from scipy.special import expit, logit, logsumexp, softmax
from surpyval import ExactEventTime

from repyability._version import __version__
from repyability.non_repairable import NonRepairable
from repyability.rbd import (
    _ccf_chain,
    _chain_transient,
    _crew_chain,
)
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd import (
    _quadrature,
    _spares,
    _standby_chain,
    _streams,
    _timeline_runs,
)
from repyability.rbd import capacity as _capacity
from repyability.rbd._block_replacement import (
    BlockHead,
    block_availability,
    block_cycle,
)
from repyability.rbd._condition_replacement import condition_cycle
from repyability.rbd._exact import ExactSum, add_columns
from repyability.rbd._hidden_life import (
    TestedLife,
    TestedLifeCurve,
    TestedLifeSteady,
)
from repyability.rbd._model_utils import (
    failure_time_scale,
    is_fixed_probability,
    model_mean,
)
from repyability.rbd._point_availability import (
    Atoms,
    BlockCurve,
    First,
    InspectionCurve,
    PartialTestCurve,
    ShiftedCurve,
    StartedBlockCurve,
    SteadyCurve,
    SystemCurve,
)
from repyability.rbd._point_availability import knots as point_knots
from repyability.rbd._point_availability import unit_curve
from repyability.rbd._sampling import inverse_sampler
from repyability.rbd.degrading_node import DegradingNode
from repyability.rbd.helper_classes import PerfectReliability
from repyability.rbd.node_state import NodeState
from repyability.rbd.rbd import RBD, _check_on_infeasible_rbd
from repyability.rbd.redundancy_allocation import (
    lowest_total_cost,
    redundancy_caps,
)
from repyability.rbd.results import (
    AvailabilityAllocation,
    AvailabilityResult,
    CapacityDistribution,
    ConfidenceInterval,
    ControlVariate,
    CostResult,
    Criticalities,
    ExpectedCost,
    ExpectedEvents,
    FailureCriticalityIndex,
    MaintenancePlan,
    RestorationCriticalityIndex,
    SparesDemand,
    SparesStock,
    TimelineSimulation,
    TotalCostAllocation,
    UpDownImportance,
)

if TYPE_CHECKING:
    from repyability.rbd.chunks import SimulationChunk

from repyability.rbd.routes import AnalysisRoute
from repyability.utils.deprecation import (
    REMOVAL,
    nonparametric_nodes,
    renamed,
    warn_nonparametric,
)


class _StreamedRBD:
    """Stands in for a nested :class:`RepairableRBD` component during
    ``availability()``: runs the nested RBD's own simulation, with its
    components drawing from their own streams through their stand-ins (see
    ``RepairableRBD._streamed_components``)."""

    def __init__(self, rbd: "RepairableRBD", sources: dict):
        self._rbd = rbd
        self._sources = sources

    def initialize_event_queue(self, t_simulation, state=None):
        """Start the nested RBD's simulation, its components from
        ``state`` (checked; see ``RepairableRBD._simulation_states``)."""
        self._rbd._start_queue(
            t_simulation, set(), set(), "p", self._sources, state or {}
        )

    def next_event(self):
        return self._rbd.next_event(sources=self._sources)

    @property
    def system_state(self) -> bool:
        return self._rbd.system_state

    @property
    def last_change_planned(self) -> bool:
        return self._rbd.last_change_planned


def _planned(source, status: bool) -> bool:
    """Whether a nested RBD's change to ``status``, just drawn from
    ``source``, starts a planned outage."""
    return not status and getattr(source, "last_change_planned", False)


class _EventQueue:
    """The simulation's event queue.

    ``queue.PriorityQueue`` is this same heap (``heappush``/``heappop``) plus
    thread locking on every operation, which a single-threaded simulation
    only pays for; events come out in exactly the same order.

    The heap holds ``(time, event)`` pairs, so that times compare as floats
    rather than through the events' own ordering. Two events at the same time
    compare equal either way, so ties come out as they would with the events
    alone. (A ``(time, counter, event)`` entry would release ties in the
    order they were queued instead, which changes the results.)
    """

    __slots__ = ("_heap",)

    def __init__(self):
        self._heap: list = []

    def put(self, event: "Event") -> None:
        heapq.heappush(self._heap, (event.time, event))

    def get(self) -> "Event":
        return heapq.heappop(self._heap)[1]

    def empty(self) -> bool:
        return not self._heap

    def qsize(self) -> int:
        return len(self._heap)


class _StreamedComponent:
    """Stands in for a :class:`NonRepairable` component during
    ``availability()``: alternates failure and repair draws exactly as
    ``NonRepairable.next_event`` does, but takes them from the component's
    own streams (see ``_streams``). A maintained or inspected component
    draws its maintenance or test times from a stream of its own too, or,
    when they cannot be streamed, from ``model.random``."""

    __slots__ = (
        "_failure",
        "_repair",
        "_fails_next",
        "_duration",
        "_model",
        "_start",
        "_test",
    )

    def __init__(
        self,
        failure,
        repair,
        duration=None,
        model=None,
        start=None,
        test=None,
    ):
        self._failure = failure
        self._repair = repair
        self._fails_next = True
        self._duration = duration
        self._model = model
        self._start = start
        self._test = test

    def reset(self):
        self._fails_next = True

    def maintenance_time(self) -> float:
        if self._duration is not None:
            return self._duration.draw()
        return self._model.random(1).item()

    def start_uniform(self) -> float:
        """The uniform a start from a state draws what is left of the
        unit's life, repair or maintenance from (see
        ``RepairableRBD._started``): from its ``START`` stream, or numpy's
        global RNG."""
        if self._start is not None:
            return self._start.draw()
        return float(np.random.random())

    def test_uniform(self) -> float:
        """The uniform that decides whether a test that can miss a failure
        finds it: from its ``TEST`` stream, or numpy's global RNG."""
        if self._test is not None:
            return self._test.draw()
        return float(np.random.random())

    def life_drawn(self) -> None:
        """The unit's life has been drawn (given its age, at a start from a
        state): its next draw is its repair."""
        self._fails_next = False

    def next_event(self):
        if self._fails_next:
            self._fails_next = False
            return self._failure.draw(), False
        self._fails_next = True
        return self._repair.draw(), True


class _ModelDraws:
    """A model's draws, one at a time, from numpy's global RNG: as a
    stream gives them, and as ``NonRepairable.next_event`` draws them (one
    ``random(1)`` call each)."""

    __slots__ = ("_model",)

    def __init__(self, model):
        self._model = model

    def draw(self) -> float:
        return self._model.random(1).item()


def _aged_life(model, age: float, u: float) -> float:
    """The life left to a unit of lifetime ``model`` at virtual age
    ``age``, from the uniform ``u``: the ``x`` with ``H(age + x) = H(age) -
    log(u)`` (``H`` the cumulative hazard), as surpyval's virtual-age
    renewal models draw it (``conditional_gaps``). A plain Exponential or
    Weibull is worked in closed form, any other model by surpyval."""
    dist = getattr(model, "dist", None)
    name = getattr(dist, "name", None)
    if (
        name in ("Exponential", "Weibull")
        and getattr(model, "p", None) == 1
        and getattr(model, "f0", None) == 0
        and not getattr(model, "gamma", 0)
    ):
        exposure = 0.0 - math.log(u) if u > 0.0 else math.inf
        if name == "Exponential":
            return exposure / float(model.params[0])
        alpha, beta = (float(p) for p in model.params)
        hazard = (age / alpha) ** beta if age > 0.0 else 0.0
        if hazard == 0.0:
            # As new (or too young for its hazard to register).
            return max(alpha * exposure ** (1.0 / beta) - age, 0.0)
        # alpha * (H(age) + exposure) ** (1 / beta) - age, without the
        # cancellation.
        return age * math.expm1(math.log1p(exposure / hazard) / beta)
    from surpyval.recurrent.renewal.renewal_model import conditional_gaps

    return float(conditional_gaps(model, np.array([age]), np.array([u]))[0])


class _ImperfectComponent(_StreamedComponent):
    """Stands in for a component repaired imperfectly (a spec's
    ``"repair"``; see ``RepairableRBD``) during a simulation, keeping the
    unit's virtual age (Kijima): a repair after an operating time ``x``
    takes it from ``v`` to ``v + q * x`` (Kijima I) or ``q * (v + x)``
    (Kijima II), and its next life is drawn given it (``_aged_life``),
    from a stream of its own. A unit as new draws from its life stream, as
    any component does. At the ``replace_after``-th failure since it was
    renewed it is replaced instead, as new, and a scheduled replacement
    renews it too (``reset``). ``operated`` is its operating time since it
    was renewed, which age replacement counts. Without streams, it draws
    from its models and numpy's global RNG."""

    __slots__ = (
        "_unit",
        "_aged",
        "_q",
        "_second",
        "_limit",
        "_x",
        "age",
        "operated",
        "count",
    )

    def __init__(
        self,
        unit,
        imperfect: "_Imperfect",
        failure=None,
        repair=None,
        aged=None,
        duration=None,
        model=None,
        test=None,
    ):
        super().__init__(failure, repair, duration, model, test=test)
        self._unit = unit
        self._aged = aged
        self._q = imperfect.q
        self._second = imperfect.kijima == "kijima2"
        self._limit = imperfect.replace_after
        self._x = 0.0
        self.age = self.operated = 0.0
        self.count = 0

    def reset(self):
        self._fails_next = True
        self.age = self.operated = 0.0
        self.count = 0

    @property
    def replacing(self) -> bool:
        """Whether the unit's failure (the last drawn) is the
        ``replace_after``-th since it was renewed, which replaces it."""
        return self._limit is not None and self.count + 1 >= self._limit

    def next_event(self):
        if self._fails_next:
            self._fails_next = False
            if self.age > 0.0:
                u = (
                    self._aged.draw()
                    if self._aged is not None
                    else float(np.random.random())
                )
                x = _aged_life(self._unit.reliability, self.age, u)
            elif self._failure is not None:
                x = self._failure.draw()
            else:
                x = self._unit.reliability.random(1).item()
            self._x = x
            return x, False
        self._fails_next = True
        if self.replacing:
            self.age = self.operated = 0.0
            self.count = 0
        else:
            x = self._x
            self.count += 1
            self.operated += x
            self.age = (
                self._q * (self.age + x)
                if self._second
                else self.age + self._q * x
            )
        if self._repair is not None:
            return self._repair.draw(), True
        return self._unit.time_to_replace.random(1).item(), True


def _stand_in(
    component,
    run,
    made: dict,
    path: tuple,
    duration=None,
    imperfect: Optional["_Imperfect"] = None,
):
    """What a component draws its events from during ``availability()``
    (see ``RepairableRBD._streamed_components``): a stand-in taking its
    draws from its own streams, or the component itself when they cannot be
    streamed (a model other than a surpyval parametric one, or a subclass,
    which may draw its events its own way), which then draws from numpy's
    global RNG. ``path`` is the node's place, nested RBDs included, which
    names its streams; ``duration`` its maintenance or test time model, and
    ``imperfect`` its imperfect repair (see ``_ImperfectComponent``)."""
    if type(component) is RepairableRBD:
        return _StreamedRBD(
            component, component._streamed_components(run, made, path)
        )
    if type(component) is not NonRepairable:
        return component
    failure = run.stream(path, _streams.FAILURE)
    repair = run.stream(path, _streams.REPAIR)
    if imperfect is not None:
        if failure is None or repair is None:
            return _ImperfectComponent(component, imperfect, model=duration)
        return _ImperfectComponent(
            component,
            imperfect,
            failure,
            repair,
            run.stream(path, _streams.AGED),
            run.stream(path, _streams.DURATION),
            duration,
            run.stream(path, _streams.TEST),
        )
    if failure is None or repair is None:
        return component
    return _StreamedComponent(
        failure,
        repair,
        run.stream(path, _streams.DURATION),
        duration,
        run.stream(path, _streams.START),
        run.stream(path, _streams.TEST),
    )


@dataclass(frozen=True)
class _Unit:
    """A unit of a standby group: its repair is a job for the repair crews
    under this key (a dataclass, so equal to no node name)."""

    node: Any
    unit: int


class _StandbyDraws:
    """What a standby group's units draw during a simulation: each unit's
    lives and repair times from streams of its own, and the group's switches
    from one (see ``_streams``); or, when they cannot be streamed, from the
    unit models' ``random`` and numpy's global RNG."""

    __slots__ = ("_models", "_lives", "_repairs", "_switches")

    def __init__(self, models, lives=None, repairs=None, switches=None):
        self._models = models
        self._lives = lives
        self._repairs = repairs
        self._switches = switches

    def life(self, unit: int) -> float:
        if self._lives is None:
            return self._models.reliability.random(1).item()
        return self._lives[unit].draw()

    def repair(self, unit: int) -> float:
        if self._repairs is None:
            return self._models.time_to_replace.random(1).item()
        return self._repairs[unit].draw()

    def switch(self) -> float:
        if self._switches is None:
            return float(np.random.random())
        return self._switches.draw()


def _uniforms(u: np.ndarray) -> np.ndarray:
    """The draws of a stream of uniforms: the uniforms themselves."""
    return u


def _state_key(states: dict) -> Optional[str]:
    """The components' ``states`` at the start of a run (checked; see
    ``RepairableRBD._simulation_states``) as JSON, the same in every
    process, for a chunk's settings (see ``simulate_chunk``); None for
    every component new."""

    def plain(states: dict) -> list:
        out = []
        for node, start in states.items():
            if isinstance(start, NodeState):
                if start.new:
                    continue
                value: Any = dataclasses.asdict(start)
            else:
                value = plain(start)
                if not value:
                    continue
            out.append([repr(node), value])
        return sorted(out, key=lambda item: item[0])

    entries = plain(states)
    return json.dumps(entries, sort_keys=True) if entries else None


def _hot_units(spec: dict) -> "RepairableRBD":
    """A standby group's twin (see ``RepairableRBD._twin``): its units
    operating together, each failing and repaired on its own, ``k`` of them
    needed, as a nested RBD whose components are named ``0, 1, ...``, as
    the group's units are, so that their streams are the units'."""
    standby = spec["standby"]
    units, k = int(standby.get("units", 2)), int(standby.get("k", 1))
    keys = ("reliability", "repairability") + RepairableRBD.COST_KEYS
    unit = {key: spec[key] for key in keys if key in spec}
    return RepairableRBD(
        [("s", u) for u in range(units)] + [(u, "t") for u in range(units)],
        {u: dict(unit) for u in range(units)},
        k={"t": k} if k > 1 else None,
    )


def _check_shard_map(shard_map, shard_size, n_jobs) -> None:
    """Check ``availability``'s ``shard_map`` and ``shard_size``."""
    if shard_map is None:
        if shard_size is not None:
            raise ValueError(
                "shard_size is the size of the shards shard_map runs: give "
                "shard_map too."
            )
        return
    if not callable(shard_map):
        raise ValueError(
            "shard_map must be a map: called as shard_map(run_shard, "
            f"shards), like map; got {shard_map!r}."
        )
    if n_jobs is not None:
        raise ValueError(
            "shard_map runs the shards wherever it sends them: give its "
            "workers the CPUs, and leave out n_jobs."
        )


def _shard_bytes(template: dict, start: int, stop: int) -> bytes:
    """A shard (see ``RepairableRBD.shards``): the run's ``template`` with
    the range of its simulations, as JSON."""
    return json.dumps({**template, "start": start, "stop": stop}).encode()


def _states_from_key(rbd, key: Optional[str]) -> dict:
    """The components' states ``_state_key`` saved, for ``rbd`` (a copy of
    the system the key was made for, as a shard rebuilds it): each node by
    its ``repr``, a nested RBD's own states in turn."""

    def states(rbd, entries: list) -> dict:
        nodes = {repr(node): node for node in rbd.components}
        out: dict = {}
        for name, value in entries:
            node = nodes[name]
            if isinstance(value, dict):
                out[node] = NodeState(**value)
            else:
                out[node] = states(rbd.components[node], value)
        return out

    return {} if key is None else states(rbd, json.loads(key))


def _draws_start(start) -> bool:
    """Whether a component started from ``start`` (a ``NodeState``, or
    None for new) draws what is left of its life, repair or maintenance:
    when it is down, or up at an age. Up at age 0, it is new, whatever its
    phase."""
    return isinstance(start, NodeState) and (not start.alive or start.age > 0)


def _standby_draws(component, run, path: tuple, arrangement: "_Standby"):
    """A standby group's draws in a run (see ``_StandbyDraws``): from
    streams if every unit's lives and repairs have one."""
    units = range(arrangement.units)
    lives = [run.stream(path + (u,), _streams.FAILURE) for u in units]
    repairs = [run.stream(path + (u,), _streams.REPAIR) for u in units]
    if any(stream is None for stream in lives + repairs):
        return _StandbyDraws(component)
    return _StandbyDraws(
        component, lives, repairs, run.stream(path, _streams.SWITCH)
    )


# The kinds of a standby group's own events.
_UNIT_FAILS, _SPARE_FAILS, _UNIT_REPAIRED = 0, 1, 2


class _StandbyGroup:
    """A standby group's units during one simulation (a component spec's
    ``"standby"``; see ``RepairableRBD``).

    Each unit put into service as new draws a life, used up at rate 1 while
    it operates and at ``dormancy_factor`` while it waits as a spare (the
    cumulative-exposure model of ``StandbyModel``): a spare that uses it up
    fails in standby, and is found at once. A failed unit's repair is a job
    for the RBD's repair crews, under the key ``_Unit(node, unit)`` (see
    ``_Crews``), and its repair time is drawn when it fails. When an
    operating unit fails and a spare is waiting, the one that has waited
    longest is switched in, which works with ``switching_probability``; a
    failed switch leaves the position empty until a repaired unit fills it,
    and the spare waits on. A repaired unit fills an empty position, or
    joins the spares. The group is up while ``k`` units operate.

    The group keeps its units' events in a queue of its own and has one
    event in the RBD's queue, ``entry``: its next. The RBD passes that
    event to ``advance``, which returns the group's state after it.
    """

    __slots__ = (
        "_rbd",
        "_node",
        "_k",
        "_dormancy",
        "_switching",
        "_draws",
        "_end",
        "_operating",
        "_spares",
        "_life",
        "_since",
        "_version",
        "_events",
        "_order",
        "entry",
        "up",
    )

    def __init__(
        self,
        rbd: "RepairableRBD",
        node,
        arrangement: "_Standby",
        draws,
        t_simulation: float,
    ):
        self._rbd = rbd
        self._node = node
        self._k = arrangement.k
        self._dormancy = arrangement.dormancy_factor
        self._switching = arrangement.switching_probability
        self._draws = (
            draws if isinstance(draws, _StandbyDraws) else _StandbyDraws(draws)
        )
        self._end = t_simulation
        self._operating: set = set()
        self._spares: deque = deque()
        # Each unit's life left, as of ``_since``; and a count that a
        # queued event of the unit's must match to be current.
        self._life = [0.0] * arrangement.units
        self._since = [0.0] * arrangement.units
        self._version = [0] * arrangement.units
        self._events: list = []
        self._order = 0
        self.entry: Optional["Event"] = None
        self.up = True
        for unit in range(arrangement.units):
            self._life[unit] = self._draws.life(unit)
            if unit < self._k:
                self._operate(unit, 0.0)
            else:
                self._wait(unit, 0.0)
        self._arm()

    def _queue(self, time: float, kind: int, unit: int) -> None:
        heapq.heappush(
            self._events,
            (time, self._order, kind, unit, self._version[unit]),
        )
        self._order += 1

    def _operate(self, unit: int, t: float) -> None:
        self._operating.add(unit)
        self._since[unit] = t
        self._queue(t + self._life[unit], _UNIT_FAILS, unit)

    def _wait(self, unit: int, t: float) -> None:
        self._spares.append(unit)
        self._since[unit] = t
        if self._dormancy > 0.0:
            self._queue(
                t + self._life[unit] / self._dormancy, _SPARE_FAILS, unit
            )

    def _repair(self, unit: int, t: float) -> None:
        """Unit ``unit`` has failed at ``t``: its repair, drawn now, is a
        job for a crew."""
        self._version[unit] += 1
        ends = t + self._draws.repair(unit)
        crews = self._rbd._crews
        if crews is None or crews.request(
            _Unit(self._node, unit), t, Event(ends, None, True)
        ):
            self._queue(ends, _UNIT_REPAIRED, unit)

    def crew_started(self, unit: int, ends: float) -> None:
        """A crew has started unit ``unit``'s repair late: it ends at
        ``ends``."""
        self._queue(ends, _UNIT_REPAIRED, unit)
        if self.entry is None or ends < self.entry.time:
            self._arm()

    def _arm(self) -> None:
        """Queue the group's next event in the RBD's queue (an earlier one
        already there is left, and ignored)."""
        events, version = self._events, self._version
        while events and events[0][4] != version[events[0][3]]:
            heapq.heappop(events)
        if not events or events[0][0] >= self._end:
            self.entry = None
            return
        self.entry = Event(events[0][0], self._node, self.up)
        self._rbd._event_queue.put(self.entry)

    def holds(self, event: "Event") -> bool:
        """Whether ``event``, from the RBD's queue, is the group's current
        one (not superseded by an earlier)."""
        return event is self.entry

    def advance(self, t: float) -> Tuple[bool, int]:
        """Take the group's event at ``t`` (its ``entry``): whether it is up
        after it, and how many of its units failed (each is repaired)."""
        self.entry = None
        events, version = self._events, self._version
        while events[0][4] != version[events[0][3]]:
            heapq.heappop(events)
        _, _, kind, unit, _ = heapq.heappop(events)
        failures = 0
        if kind == _UNIT_REPAIRED:
            crews = self._rbd._crews
            if crews is not None:
                started = crews.release(_Unit(self._node, unit), t)
                if started is not None:
                    self._rbd._crew_started(started)
            self._life[unit] = self._draws.life(unit)
            if len(self._operating) < self._k:
                self._operate(unit, t)
            else:
                self._wait(unit, t)
        elif kind == _SPARE_FAILS:
            self._spares.remove(unit)
            failures = 1
            self._repair(unit, t)
        else:
            self._operating.discard(unit)
            failures = 1
            self._repair(unit, t)
            if self._spares:
                spare = self._spares.popleft()
                switching = self._switching
                if switching >= 1.0 or (
                    switching > 0.0 and self._draws.switch() < switching
                ):
                    # It has used up its life at the dormant rate so far (all
                    # of it, at most: a spare due to fail in standby now).
                    self._version[spare] += 1
                    used = self._dormancy * (t - self._since[spare])
                    self._life[spare] = max(self._life[spare] - used, 0.0)
                    self._operate(spare, t)
                else:
                    self._spares.appendleft(spare)
        self.up = len(self._operating) == self._k
        self._arm()
        return self.up, failures


class _Crews:
    """The repair crews of one simulation (``RepairableRBD``'s
    ``repair_crews``): how many are free, the components they are working
    on, and the jobs waiting for one.

    A job is the work that brings a component back up: a repair or
    replacement, maintenance that takes time, or a test that takes time.
    The component is down from when the job fell due. A job that finds no
    crew free waits, and the next crew to finish starts the waiting job of
    the highest priority, and of those the one that fell due first. The
    job's length was drawn when it fell due, so waiting shifts the
    component's restoration without changing its draws.
    """

    __slots__ = ("free", "served", "holding", "waiting", "_order", "_rank")

    def __init__(self, crews: int, served, priority: dict):
        self.free = crews
        self.served = frozenset(served)
        self.holding: set = set()
        self.waiting: list = []
        self._order = 0
        # Higher priority first; a standby group's units have the group's.
        self._rank = {
            key: -priority.get(
                key.node if isinstance(key, _Unit) else key, 0.0
            )
            for key in served
        }

    def request(self, node, due: float, done: "Event") -> Optional["Event"]:
        """``node``'s job, due at ``due`` and ending with ``done`` if a crew
        starts it at once: ``done``, or None while the job waits."""
        if self.free:
            self.free -= 1
            self.holding.add(node)
            return done
        entry = (self._rank[node], due, self._order, node, done)
        heapq.heappush(self.waiting, entry)
        self._order += 1
        return None

    def release(self, node, t: float) -> Optional[Tuple[Any, float, "Event"]]:
        """``node``'s job is done at ``t``: its crew starts the next waiting
        job, if any. That job's component, how long it waited, and the event
        that now ends it; or None."""
        self.holding.discard(node)
        if not self.waiting:
            self.free += 1
            return None
        _, due, _, waiting, done = heapq.heappop(self.waiting)
        self.holding.add(waiting)
        wait = t - due
        ends = Event(
            done.time + wait,
            waiting,
            done.status,
            done.preventive,
            done.inspection,
        )
        return waiting, wait, ends


class _Fixed:
    """A cost charged at the same amount every time."""

    __slots__ = ("amount",)

    def __init__(self, amount: float):
        self.amount = amount

    def draw(self) -> float:
        return self.amount


#: Simulations per task of a parallel run of the Python engine (see
#: ``RepairableRBD``'s ``availability``).
PARALLEL_BLOCK = 250

#: The cost categories, in the order ``CostResult.by_category`` lists them.
_CATEGORIES = (
    "repair",
    "replace",
    "preventive",
    "inspection",
    "component_downtime",
    "system_downtime",
    "setup",
)
# The category of a repair's own cost (all an imperfect repair is charged).
_REPAIR = _CATEGORIES.index("repair")


class _Replication:
    """One simulation's results, which a :class:`_Tally` adds to its totals
    in the order of the simulations.

    ``node_up``, ``both_up`` and ``both_down`` hold each component's up
    time, and the time it and the system were both up and both down;
    ``counts`` each component's failures, the system failures they caused,
    its restorations and the system restorations they caused; and
    ``changes`` the times the system changed state, with ``deltas`` +1 for
    a restoration and -1 for a failure or planned outage. ``cost`` is the
    simulation's total cost (None when nothing is priced), ``by_category``
    and ``by_node`` its breakdown, and the ``capacity`` fields what the
    capacity trace recorded (see ``_CapacityTrace``). ``replacements``
    counts each component's replacements (its failures, found ones for a
    component with hidden failures, its preventive replacements, and a
    standby group's units' failures): the spares it used.
    """

    __slots__ = (
        "uptime",
        "node_up",
        "both_up",
        "both_down",
        "counts",
        "failures",
        "restorations",
        "planned",
        "changes",
        "deltas",
        "cost",
        "by_category",
        "by_node",
        "capacity_changes",
        "capacity_time",
        "delivered",
        "replacements",
        "opportunistic",
        "history",
    )
    uptime: float
    node_up: List[float]
    both_up: List[float]
    both_down: List[float]
    counts: Tuple[List[int], List[int], List[int], List[int]]
    failures: int
    restorations: int
    planned: int
    changes: List[float]
    deltas: List[int]
    cost: Optional[float]
    by_category: List[float]
    by_node: Dict[Any, float]
    capacity_changes: Optional[list]
    capacity_time: Optional[list]
    delivered: Optional[float]
    replacements: List[int]
    opportunistic: List[int]
    history: Optional[tuple]


def _working_over_time(
    changed_at: np.ndarray, deltas: np.ndarray, t_end: float, start: int
) -> Tuple[np.ndarray, np.ndarray]:
    """How many simulated systems work after each time at which one changed
    state (and at 0 and ``t_end`` whether or not any did): the times, in
    order, and the counts, ``start`` working at 0. Changes given in order of
    time (as an engine may keep them) are added up as they come; otherwise
    they are sorted first. Either way the counts are whole numbers added
    exactly, so the order cannot change them."""
    in_order = changed_at.size == 0 or (
        changed_at[0] >= 0.0
        and changed_at[-1] < t_end
        and bool(np.all(changed_at[1:] >= changed_at[:-1]))
    )
    if not in_order:
        time, inverse = np.unique(
            np.concatenate(([0.0, t_end], changed_at)), return_inverse=True
        )
        working = np.bincount(
            inverse.ravel(),
            weights=np.concatenate(([start, 0], deltas)),
            minlength=time.size,
        )
        return time, working.cumsum()
    new = np.empty(changed_at.size, bool)
    if changed_at.size:
        new[0] = True
        np.not_equal(changed_at[1:], changed_at[:-1], out=new[1:])
    firsts = np.flatnonzero(new)
    times = changed_at[firsts]
    sums = (
        np.add.reduceat(deltas, firsts).astype(float)
        if changed_at.size
        else np.zeros(0)
    )
    at_zero = bool(times.size) and times[0] == 0.0
    if at_zero:
        sums[0] += start
        time = np.concatenate((times, [t_end]))
        weights = np.concatenate((sums, [0.0]))
    else:
        time = np.concatenate(([0.0], times, [t_end]))
        weights = np.concatenate(([float(start)], sums, [0.0]))
    return time, weights.cumsum()


def _add_at(totals: dict, key, value) -> None:
    """Add ``value`` (a float or an ``ExactSum``) to ``totals[key]``,
    exactly: a float while only one value has come, an ``ExactSum`` once
    another does."""
    held = totals.get(key)
    if held is None:
        totals[key] = ExactSum(value) if isinstance(value, ExactSum) else value
    elif isinstance(held, ExactSum):
        held.add(value)
    else:
        totals[key] = ExactSum((held,)).add(value)


def _partials(value) -> list:
    """The floats whose exact sum is ``value`` (see ``_add_at``)."""
    return value.partials if isinstance(value, ExactSum) else [value]


def _by_time(
    times: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``values`` grouped by their ``times``: the distinct times, in order;
    the values, in the order of their times; and where each time's values
    start."""
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    new = np.ones(times.size, dtype=bool)
    new[1:] = times[1:] != times[:-1]
    starts = np.flatnonzero(new)
    return times[starts], values, starts


def _group_stops(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Where each group of ``values`` that starts at one of ``starts``
    stops."""
    stops = np.empty_like(starts)
    stops[:-1] = starts[1:]
    stops[-1:] = values.size
    return stops


def _group_totals(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """The exact sum of each group of ``values`` (from each of ``starts``
    to the next), rounded once: a group of one value, the value."""
    totals = values[starts]
    stops = _group_stops(values, starts)
    for i in np.flatnonzero(stops - starts > 1).tolist():
        totals[i] = float(ExactSum(values[starts[i] : stops[i]]))  # noqa: E203
    return totals


def _group_partials(values: np.ndarray, starts: np.ndarray) -> list:
    """For each group of ``values`` (from each of ``starts`` to the next),
    the floats whose exact sum is the group's (``ExactSum.partials``: none
    for a sum of 0)."""
    stops = _group_stops(values, starts)
    out = []
    for a, b, first in zip(
        starts.tolist(), stops.tolist(), values[starts].tolist()
    ):
        if b - a == 1:
            out.append([first] if first else [])
        else:
            out.append(ExactSum(values[a:b]).partials)
    return out


def _exact_total(data) -> ExactSum:
    """An exact total from ``_Tally.to_dict``'s data: the floats whose
    exact sum it is (or a number)."""
    if isinstance(data, (int, float)):
        return ExactSum(float(data))
    return ExactSum([float(v) for v in data])


class _Tally:
    """The running totals of an availability simulation.

    Every total that is a sum of floats is kept exactly, as an ``ExactSum``
    (see ``_exact``), and rounded once when the result is built (#151): so
    the totals are the same to the last bit however the simulations ran,
    one after another in this process, in blocks in other processes,
    compiled, or in chunks merged later. Per-node totals are lists in the
    order of ``nodes``; the per-simulation values (``uptimes``,
    ``cost_samples``, ``delivered``) are kept in the simulations' order.
    """

    #: Simulations whose per-node values wait to be folded into the
    #: per-node totals (see ``_fold``).
    _ROWS = 256

    def __init__(
        self,
        nodes: list,
        costs: dict,
        t_simulation: float,
        curve_points: Optional[int] = None,
    ):
        n = len(nodes)
        self.nodes = nodes
        self.t_simulation = t_simulation
        self.n = 0
        # The times the simulated systems changed state, and +1 or -1; and
        # the same, as arrays, from the compiled engine.
        self.changes: list = []
        self.deltas: list = []
        self.change_arrays: list = []
        # With curve_points (#153), the changes are counted on a grid of
        # times instead of kept: the net change in how many systems work
        # in each bin up to each grid time (from just after the one
        # before), the grid's first time 0.
        self.curve_points = curve_points
        self.edges: Optional[np.ndarray] = None
        self.binned: Optional[np.ndarray] = None
        if curve_points is not None:
            steps = np.arange(curve_points + 1)
            self.edges = t_simulation * steps / curve_points
            self.binned = np.zeros(curve_points + 1, dtype=np.int64)
        # Per node: its failures, the system failures they caused, its
        # restorations and the system restorations they caused.
        self.counts = [[0] * n for _ in range(4)]
        self.system_restorations = 0
        self.system_failures = 0
        self.system_planned_outages = 0
        self.system_uptime = ExactSum()
        self.system_downtime = ExactSum()
        self.node_uptime = [ExactSum() for _ in range(n)]
        self.node_downtime = [ExactSum() for _ in range(n)]
        self.intersection_uptime = [ExactSum() for _ in range(n)]
        self.intersection_downtime = [ExactSum() for _ in range(n)]
        self.union_uptime = [ExactSum() for _ in range(n)]
        self.union_downtime = [ExactSum() for _ in range(n)]
        # Each simulation's up time, its nodes' up times and times up and
        # down with the system, and its costs, until they are folded into
        # the totals (see _fold).
        self._rows: list = []
        self._cost_rows: list = []
        # One system uptime (and, when priced, one total cost) per
        # replication, in order.
        self.uptimes: List[float] = []
        self.cost_samples: List[float] = []
        self.cost_by_category = {key: ExactSum() for key in _CATEGORIES}
        self.cost_by_component = {node: ExactSum() for node in costs}
        # With capacities (see _CapacityRecorder): the times a simulated
        # system's expected capacity changed and by how much, and the times
        # whether it can carry an unlimited amount did (+1 or -1), as lists
        # from the Python engine and arrays from the compiled one (see
        # capacity_records); the time spent at each capacity (exactly, see
        # _add_at); and each replication's delivered fraction of the
        # demand, in order.
        self.capacity_times: list = []
        self.capacity_steps: list = []
        self.unlimited_times: list = []
        self.unlimited_steps: list = []
        self.capacity_arrays: list = []
        self.capacity_time: dict = {}
        self.delivered: List[float] = []
        # Each replication's replacements of each node, when asked for
        # (see RepairableRBD.spares_demand).
        self.replacements: Optional[List[List[int]]] = None
        # Per node: its early renewals at stops of its maintenance group.
        self.opportunistic = [0] * n
        # The simulations' histories, when asked for (see
        # RepairableRBD.simulate_timelines): kept, in order, in place of
        # every total.
        self.histories: Optional[_timeline_runs.Records] = None

    def add(self, rec: _Replication) -> None:
        """Add the next simulation's results."""
        if self.histories is not None:
            assert rec.history is not None  # recorded (see _replicate)
            self.histories.add(rec.history)
            self.n += 1
            return
        self.n += 1
        self.uptimes.append(rec.uptime)
        self.system_failures += rec.failures
        self.system_restorations += rec.restorations
        self.system_planned_outages += rec.planned
        self.changes.extend(rec.changes)
        self.deltas.extend(rec.deltas)
        self._rows.append(
            [rec.uptime, *rec.node_up, *rec.both_up, *rec.both_down]
        )
        for totals, counts in zip(self.counts, rec.counts):
            for c, count in enumerate(counts):
                if count:
                    totals[c] += count
        if rec.cost is not None:
            self.cost_samples.append(rec.cost)
            self._cost_rows.append([*rec.by_category, *rec.by_node.values()])
        if len(self._rows) >= self._ROWS:
            self._fold()
        if rec.capacity_changes is not None:
            for time, mean, limitless in rec.capacity_changes:
                if mean:
                    self.capacity_times.append(time)
                    self.capacity_steps.append(mean)
                if limitless:
                    self.unlimited_times.append(time)
                    self.unlimited_steps.append(limitless)
            spent = self.capacity_time
            for level, time in rec.capacity_time or ():
                _add_at(spent, level, time)
            if rec.delivered is not None:
                self.delivered.append(rec.delivered)
        if self.replacements is not None:
            self.replacements.append(rec.replacements)
        for c, count in enumerate(rec.opportunistic):
            if count:
                self.opportunistic[c] += count

    def add_changes(self, times: np.ndarray, deltas: np.ndarray) -> None:
        """Add simulations' changes of state (times, and +1 or -1), as an
        engine gives them: kept, or counted on the grid."""
        self.change_arrays.append((times, deltas))
        if self.binned is not None:
            self._fold_changes()

    def _fold_changes(self) -> None:
        """Count the changes kept so far in the grid's bins (each time in
        the bin of the first grid time at or after it), and drop them."""
        assert self.binned is not None and self.edges is not None
        if not (self.changes or self.change_arrays):
            return
        times, deltas = self.state_changes()
        self.changes, self.deltas, self.change_arrays = [], [], []
        bins = np.minimum(
            np.searchsorted(self.edges, times, side="left"),
            len(self.edges) - 1,
        )
        counted = np.bincount(bins, weights=deltas, minlength=len(self.edges))
        self.binned += counted.astype(np.int64)

    def _fold(self) -> None:
        """Fold the values of the simulations added since the last fold
        into the totals, all at once (see ``fold_columns``), and the
        changes into the grid's bins, with ``curve_points``."""
        if self.binned is not None:
            self._fold_changes()
        if self._rows:
            table = np.asarray(self._rows, dtype=float)
            self._rows = []
            n = len(self.nodes)
            self.fold_columns(
                table[:, 0],
                table[:, 1 : 1 + n],  # noqa: E203
                table[:, 1 + n : 1 + 2 * n],  # noqa: E203
                table[:, 1 + 2 * n :],  # noqa: E203
            )
        if self._cost_rows:
            self.fold_costs(np.asarray(self._cost_rows, dtype=float))
            self._cost_rows = []
        if self.capacity_times or self.unlimited_times:
            self.capacity_arrays.append(
                (
                    np.asarray(self.capacity_times, dtype=float),
                    np.asarray(self.capacity_steps, dtype=float),
                    np.asarray(self.unlimited_times, dtype=float),
                    np.asarray(self.unlimited_steps, dtype=np.int64),
                )
            )
            self.capacity_times, self.capacity_steps = [], []
            self.unlimited_times, self.unlimited_steps = [], []

    def compact(self) -> "_Tally":
        """This tally with its simulations' values folded into the totals
        and its changes of state (and of capacity) in arrays: small to send
        from a worker process to the parent (see ``_simulate_block``)."""
        if self.histories is not None:
            self.histories.compact()
            return self
        self._fold()
        if self.changes:
            times, deltas = self.state_changes()
            self.changes, self.deltas = [], []
            self.change_arrays = [(times, deltas)]
        return self

    def fold_columns(self, uptime, up, both_up, both_down) -> None:
        """Add simulations' values to the totals, exactly (one row per
        simulation): the system's up time (and so its down time), each
        node's up time (and down time), and its times up and down with the
        system (and so either up or either down)."""
        t_end = self.t_simulation
        uptime = np.asarray(uptime, dtype=float).reshape(-1, 1)
        add_columns(
            [
                self.system_uptime,
                self.system_downtime,
                *self.node_uptime,
                *self.node_downtime,
                *self.intersection_uptime,
                *self.intersection_downtime,
                *self.union_uptime,
                *self.union_downtime,
            ],
            np.hstack(
                [
                    uptime,
                    t_end - uptime,
                    up,
                    t_end - up,
                    both_up,
                    both_down,
                    t_end - both_down,
                    t_end - both_up,
                ]
            ),
        )

    def fold_costs(self, table) -> None:
        """Add simulations' costs to the totals, exactly: one row per
        simulation, its costs by category and then by component (in the
        order of ``cost_by_component``)."""
        add_columns(
            [
                *self.cost_by_category.values(),
                *self.cost_by_component.values(),
            ],
            table,
        )

    def state_changes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Every time a simulated system changed state (``-0.0`` made
        ``0.0``), and +1 for a restoration or -1 for a failure or planned
        outage."""
        times = [np.asarray(self.changes, dtype=float)]
        deltas = [np.asarray(self.deltas, dtype=np.int64)]
        for chunk_times, chunk_deltas in self.change_arrays:
            times.append(chunk_times)
            deltas.append(chunk_deltas)
        return np.concatenate(times) + 0.0, np.concatenate(deltas)

    def capacity_records(self) -> Tuple[np.ndarray, ...]:
        """Every change of a simulated system's expected capacity, as
        times (``-0.0`` made ``0.0``) and changes, whose exact sum at a
        time is the total change then; and every change of whether it can
        carry an unlimited amount, as times and +1 or -1 (see
        ``capacity_changes``)."""
        times = [np.asarray(self.capacity_times, dtype=float)]
        steps = [np.asarray(self.capacity_steps, dtype=float)]
        free_times = [np.asarray(self.unlimited_times, dtype=float)]
        free_steps = [np.asarray(self.unlimited_steps, dtype=np.int64)]
        for arrays in self.capacity_arrays:
            for kept, array in zip(
                (times, steps, free_times, free_steps), arrays
            ):
                kept.append(array)
        return (
            np.concatenate(times) + 0.0,
            np.concatenate(steps),
            np.concatenate(free_times) + 0.0,
            np.concatenate(free_steps),
        )

    def capacity_changes(self) -> Tuple[np.ndarray, ...]:
        """The times a simulated system's expected capacity changed, in
        order, the changes then, in the order of their times, and where
        each time's changes start (their exact sum is the total change
        then, see ``_group_totals``); and the times whether it can carry an
        unlimited amount changed, in order, and the net change then in how
        many can."""
        times, steps, free_times, free_steps = self.capacity_records()
        at, steps, starts = _by_time(times, steps)
        free_at, free_steps, free_starts = _by_time(free_times, free_steps)
        counts = (
            np.add.reduceat(free_steps, free_starts)
            if free_steps.size
            else free_steps
        )
        return at, steps, starts, free_at, counts

    # The totals that are lists of floats, one per node, and the counts.
    _NODE_SUMS = (
        "node_uptime",
        "node_downtime",
        "intersection_uptime",
        "intersection_downtime",
        "union_uptime",
        "union_downtime",
    )
    _COUNTS = (
        "system_restorations",
        "system_failures",
        "system_planned_outages",
    )

    def merge(self, other: "_Tally") -> None:
        """Add ``other``'s simulations, which come after this tally's (see
        ``SimulationChunk``). The per-simulation values follow on in order,
        and the totals, kept exactly, are those of one run of them all, to
        the last bit."""
        if self.histories is not None:
            assert other.histories is not None  # blocks of one run
            self.histories.merge(other.histories)
            self.n += other.n
            return
        self._fold()
        other._fold()
        self.n += other.n
        self.system_uptime.add(other.system_uptime)
        self.system_downtime.add(other.system_downtime)
        for name in self._COUNTS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in self._NODE_SUMS:
            for mine, theirs in zip(getattr(self, name), getattr(other, name)):
                mine.add(theirs)
        for totals, counts in zip(self.counts, other.counts):
            for c, count in enumerate(counts):
                totals[c] += count
        for c, count in enumerate(other.opportunistic):
            self.opportunistic[c] += count
        if self.binned is not None:
            assert other.binned is not None  # chunks of one run
            self.binned += other.binned
        else:
            times, deltas = other.state_changes()
            self.change_arrays.append((times, deltas))
        self.uptimes.extend(other.uptimes)
        self.cost_samples.extend(other.cost_samples)
        for key, amount in other.cost_by_category.items():
            self.cost_by_category[key].add(amount)
        for node, amount in other.cost_by_component.items():
            self.cost_by_component[node].add(amount)
        self.capacity_arrays.append(other.capacity_records())
        for level, time in other.capacity_time.items():
            _add_at(self.capacity_time, level, time)
        self.delivered.extend(other.delivered)
        if self.replacements is not None:
            assert other.replacements is not None  # blocks of one run
            self.replacements.extend(other.replacements)

    def to_dict(self) -> dict:
        """The totals, as JSON data (see ``SimulationChunk.to_dict``): each
        exact total as the floats whose exact sum it is."""
        self._fold()
        times, deltas = self.state_changes()
        at, steps, starts, free_at, counts = self.capacity_changes()
        return {
            "nodes": list(self.nodes),
            "t_simulation": self.t_simulation,
            "n": self.n,
            "changes": times.tolist(),
            "deltas": deltas.tolist(),
            "curve_points": self.curve_points,
            "binned": None if self.binned is None else self.binned.tolist(),
            "counts": [list(counts) for counts in self.counts],
            **{name: getattr(self, name) for name in self._COUNTS},
            "system_uptime": self.system_uptime.partials,
            "system_downtime": self.system_downtime.partials,
            **{
                name: [total.partials for total in getattr(self, name)]
                for name in self._NODE_SUMS
            },
            "uptimes": list(self.uptimes),
            "cost_samples": list(self.cost_samples),
            "cost_by_category": {
                key: amount.partials
                for key, amount in self.cost_by_category.items()
            },
            "cost_by_component": [
                [node, amount.partials]
                for node, amount in self.cost_by_component.items()
            ],
            "capacity_changes": list(
                zip(at.tolist(), _group_partials(steps, starts))
            ),
            "unlimited_changes": list(zip(free_at.tolist(), counts.tolist())),
            "capacity_time": sorted(
                (level, _partials(time))
                for level, time in self.capacity_time.items()
            ),
            "delivered": list(self.delivered),
            "opportunistic": list(self.opportunistic),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "_Tally":
        """The totals ``to_dict`` gave (node names as JSON gives them
        back)."""
        from repyability.rbd.serialisation import _node_name

        nodes = [_node_name(node) for node in data["nodes"]]
        components = [
            _node_name(node) for node, _ in data["cost_by_component"]
        ]
        tally = cls(
            nodes,
            dict.fromkeys(components),
            data["t_simulation"],
            data.get("curve_points"),
        )
        tally.n = int(data["n"])
        tally.changes = [float(t) for t in data["changes"]]
        tally.deltas = [int(d) for d in data["deltas"]]
        if data.get("binned") is not None:
            tally.binned = np.array(data["binned"], dtype=np.int64)
        tally.counts = [[int(c) for c in counts] for counts in data["counts"]]
        for name in cls._COUNTS:
            setattr(tally, name, int(data[name]))
        tally.system_uptime = _exact_total(data["system_uptime"])
        tally.system_downtime = _exact_total(data["system_downtime"])
        for name in cls._NODE_SUMS:
            setattr(tally, name, [_exact_total(v) for v in data[name]])
        tally.uptimes = [float(v) for v in data["uptimes"]]
        tally.cost_samples = [float(v) for v in data["cost_samples"]]
        tally.cost_by_category.update(
            {
                key: _exact_total(v)
                for key, v in data["cost_by_category"].items()
            }
        )
        tally.cost_by_component = {
            _node_name(node): _exact_total(amount)
            for node, amount in data["cost_by_component"]
        }
        for level, value in data["capacity_time"]:
            tally.capacity_time[float(level)] = _exact_total(value)
        for time, value in data["capacity_changes"]:
            # A total change of 0 still marks its time (as one 0.0).
            for part in _exact_total(value).partials or [0.0]:
                tally.capacity_times.append(float(time))
                tally.capacity_steps.append(part)
        for time, change in data["unlimited_changes"]:
            tally.unlimited_times.append(float(time))
            tally.unlimited_steps.append(int(change))
        tally.delivered = [float(v) for v in data["delivered"]]
        tally.opportunistic = [int(c) for c in data["opportunistic"]]
        return tally

    def criticality_counts(self) -> Tuple[dict, dict]:
        """The failure and restoration counts behind the criticality
        indices: node -> Counter, for each node that failed (FCI) or was
        restored (RCI), in the order of the nodes."""
        failed, caused_down, restored, caused_up = self.counts
        FCI = {
            node: Counter(
                component_failures=failed[c], system_failures=caused_down[c]
            )
            for c, node in enumerate(self.nodes)
            if failed[c]
        }
        RCI = {
            node: Counter(
                component_restorations=restored[c],
                system_restorations=caused_up[c],
            )
            for c, node in enumerate(self.nodes)
            if restored[c]
        }
        return FCI, RCI


class _CapacityState(NamedTuple):
    """The system's capacity in one state of the simulation (see
    ``_CapacityRecorder``)."""

    levels: tuple
    probabilities: tuple
    # The expected capacity when it is finite (0 otherwise), and whether it
    # can be unlimited.
    finite_mean: float
    unlimited: int
    # The expected fraction of the demand met, if there is a demand.
    delivered: float


class _CapacityRecorder:
    """The system's capacity in each state of an availability simulation.

    Given which components are down, the capacity has an exact
    distribution: one level, unless some nodes work at several (a node's
    capacity levels, given as a dict), which the simulation, drawing only
    whether each component is up, then weighs by their probabilities. It is
    worked out once per state and kept."""

    def __init__(self, rbd: "RepairableRBD", demand: Optional[float]):
        rbd._require_capacities_given()
        self._rbd = rbd
        self._nodes = list(rbd.nodes)
        self._states: Dict[frozenset, _CapacityState] = {}
        # Whether each node works at one level: a state's capacity is then
        # one level for sure (see evaluate).
        self._single = all(
            not isinstance(levels, dict) or len(levels) == 1
            for levels in rbd.capacity.values()
        )
        if demand is None:
            # Everything up at its highest level: the design capacity.
            top = float(self._distribution(frozenset())[0][-1])
            self.demand: Optional[float] = top if np.isfinite(top) else None
        else:
            demand = float(demand)
            if not (np.isfinite(demand) and demand > 0.0):
                raise ValueError(
                    f"demand must be a positive, finite number, got "
                    f"{demand!r}."
                )
            self.demand = demand

    def _distribution(self, down: frozenset) -> Tuple[np.ndarray, np.ndarray]:
        arrays = {
            node: np.zeros(1) if node in down else np.ones(1)
            for node in self._nodes
        }
        levels, rows = self._rbd._capacity_arrays(arrays, 1)
        return levels, rows[:, 0]

    def __call__(self, down: frozenset) -> _CapacityState:
        state = self._states.get(down)
        if state is None:
            state = self._states[down] = self._state(*self._distribution(down))
        return state

    def _state(
        self, levels: np.ndarray, probabilities: np.ndarray
    ) -> _CapacityState:
        """The state whose capacity has ``levels`` with ``probabilities``
        (those that can be reached)."""
        finite = np.isfinite(levels)
        unlimited = int(np.any(probabilities[~finite] > 0.0))
        delivered = 0.0
        if self.demand is not None:
            delivered = (
                float(np.minimum(levels, self.demand) @ probabilities)
                / self.demand
            )
        return _CapacityState(
            tuple(levels.tolist()),
            tuple(probabilities.tolist()),
            (
                0.0
                if unlimited
                else float(levels[finite] @ probabilities[finite])
            ),
            unlimited,
            delivered,
        )

    def evaluate(self, components: list, up: np.ndarray) -> list:
        """The states with ``components`` up as each row of ``up`` (1 or 0
        for each) says, the others up. When each node works at one level,
        they are worked out together, one column of probabilities each:
        each probability of each is then a product of 0s and 1s, or a sum
        with one 1 at most, so exact, and the same as in the state's own
        distribution (``__call__``), whatever else is worked out with it. A
        state that does not come out as one level for sure is worked out
        on its own all the same."""
        states: list = [None] * len(up)
        if self._single and len(up) > 1:
            ones = np.ones(len(up))
            given = dict(zip(components, np.asarray(up, dtype=float).T))
            levels, rows = self._rbd._capacity_arrays(
                {node: given.get(node, ones) for node in self._nodes},
                len(up),
            )
            top = rows.argmax(axis=0)
            sure = (np.count_nonzero(rows, axis=0) == 1) & (
                rows[top, np.arange(len(up))] == 1.0
            )
            # A state of one level for sure is its level's.
            alone: dict = {}
            for i in np.flatnonzero(sure).tolist():
                level = int(top[i])
                state = alone.get(level)
                if state is None:
                    state = alone[level] = self._state(
                        levels[level : level + 1], np.ones(1)  # noqa: E203
                    )
                states[i] = state
        for i, state in enumerate(states):
            if state is None:
                states[i] = self(
                    frozenset(
                        c for c, works in zip(components, up[i]) if not works
                    )
                )
        return states

    def states(self, downs: list) -> list:
        """The state of each of ``downs`` (the components down), those not
        met yet worked out together (see ``evaluate``)."""
        known = self._states
        new = [down for down in dict.fromkeys(downs) if down not in known]
        if len(new) > 1 and self._single:
            up = [[node not in down for node in self._nodes] for down in new]
            for down, state in zip(
                new, self.evaluate(self._nodes, np.array(up))
            ):
                known[down] = state
        return [self(down) for down in downs]

    def trace(self, status: dict) -> "_CapacityTrace":
        """The capacity of one simulation, which starts with ``status``."""
        return _CapacityTrace(self, status)


class _CapacityTrace:
    """One simulation's capacity over time (for the tally to add up in
    order): each change of the expected capacity, and the time spent at
    each level and the fraction of the demand delivered. The components
    down after each change are recorded as it goes, and the states they
    are in worked out at the end, those met first together (see
    ``_CapacityRecorder.states``)."""

    def __init__(self, recorder: _CapacityRecorder, status):
        self.recorder = recorder
        self.down = {node for node, up in status.items() if not up}
        # The time of each change and the components down after it, from
        # the start.
        self.records: list = [(0.0, frozenset(self.down))]

    def change(self, time: float, node, up: bool) -> None:
        """Component ``node`` went up (or down) at ``time``."""
        if up:
            self.down.discard(node)
        else:
            self.down.add(node)
        self.records.append((time, frozenset(self.down)))

    def finish(self, t_simulation: float, rec: _Replication) -> None:
        """The trace, closed at the end of the window, into ``rec``: (time,
        change of the expected capacity, change of whether it can be
        unlimited), from the start's, and (level, time spent at it), in
        order."""
        records = self.records + [(t_simulation, None)]
        states = self.recorder.states([down for _, down in self.records])
        state = states[0]
        since = delivered = 0.0
        changes = [(0.0, state.finite_mean, state.unlimited)]
        spent = []
        for (time, _), new in zip(records[1:], states[1:] + [None]):
            span = time - since
            if span > 0.0:
                for level, p in zip(state.levels, state.probabilities):
                    spent.append((level, p * span))
                delivered += state.delivered * span
            since = time
            if new is not None:
                mean = new.finite_mean - state.finite_mean
                unlimited = new.unlimited - state.unlimited
                if mean or unlimited:
                    changes.append((time, mean, unlimited))
                state = new
        rec.capacity_changes = changes
        rec.capacity_time = spent
        rec.delivered = (
            None if self.recorder.demand is None else delivered / t_simulation
        )


def _stopping_rule(
    N: int,
    tolerance: Optional[float],
    confidence: float,
    max_N: Optional[int],
    antithetic: bool,
    target: str,
    t_simulation: float,
) -> Optional[Callable[..., int]]:
    """``None`` for a fixed number of replications; otherwise a function
    of the tally so far giving how many more replications to run (0 once
    the confidence interval of the mean availability over the window, or of
    the mean cost, is at most ``tolerance`` either side, or ``max_N`` have
    run: then with a warning). Given the simulations' ``values`` too (a
    controlled run's, see ``_controlled_run``), it judges those."""
    montecarlo.check_confidence(confidence)
    limit = montecarlo.sample_limit(
        N, tolerance, max_N, antithetic, ("mc_samples", "max_samples")
    )
    if limit is None:
        return None

    def stop(tally: "_Tally", values=None) -> int:
        if values is not None:
            # The controlled values of a run with an exact twin.
            values = np.asarray(values, dtype=float)
        elif target == "cost":
            values = np.asarray(tally.cost_samples, dtype=float)
        else:
            values = np.asarray(tally.uptimes, dtype=float) / t_simulation
        return montecarlo.more_samples(
            values,
            N,
            tolerance,  # type: ignore[arg-type]
            confidence,
            limit,
            antithetic,
            target,
            "max_samples",
        )

    return stop


#: A parallel run's worker process's run (see ``_start_worker``).
_WORKER: Dict[str, Any] = {}


def _start_worker(run: bytes) -> None:
    """Start a worker process of a parallel run (see ``_PythonRunner``):
    unpickle the run, the system and what its simulations share, once, for
    every block the worker runs."""
    rbd, args, curve_points, replacements, histories = pickle.loads(run)
    _WORKER["run"] = (
        rbd,
        rbd._context(*args),
        curve_points,
        replacements,
        histories,
    )


def _simulate_block(span: Tuple[int, int]) -> "_Tally":
    """Simulations ``start`` to ``stop - 1`` of a parallel run, in a worker
    process (see ``_start_worker``): their totals, compact."""
    rbd, context, curve_points, replacements, histories = _WORKER["run"]
    tally = _Tally(
        list(rbd.components),
        rbd.costs,
        context.t_simulation,
        curve_points,
    )
    if replacements:
        tally.replacements = []
    if histories:
        tally.histories = _timeline_runs.Records(len(rbd.components))
    for replication in range(*span):
        tally.add(rbd._replicate(context, replication))
    return tally.compact()


class _Context(NamedTuple):
    """What a run's simulations share (see ``RepairableRBD._context``)."""

    t_simulation: float
    working: set
    broken: set
    method: str
    capacity: Optional[_CapacityRecorder]
    run: _streams.Run
    sources: dict
    charges: tuple
    downtime_cost_rates: dict
    has_costs: bool
    position: dict
    plain: set
    works: Callable
    #: The components' states at the start (see ``_simulation_states``),
    #: and whether the system is up at the start with every component up
    #: but those held broken.
    states: dict
    initial_up: bool
    #: Whether each simulation records its components' changes (see
    #: ``simulate_timelines``).
    history: bool = False


class _PythonRunner:
    """Runs a run's simulations in Python: one after another in this
    process, or, with ``jobs`` above 1, in blocks of ``PARALLEL_BLOCK`` in
    that many processes. Each worker process is given the run once, when it
    starts (see ``_start_worker``), and sends back each block's totals,
    compact (#152), which are merged into the tally in order: the same
    totals, kept exactly, as adding the simulations one by one. ``args``
    are ``RepairableRBD._context``'s."""

    def __init__(self, rbd, tally, progress, args: tuple, jobs):
        self._rbd = rbd
        self._tally = tally
        self._progress = progress
        self._context: Optional[_Context] = None
        self._executor: Any = None
        if jobs is not None and jobs > 1:
            run = montecarlo.dumps(
                (
                    rbd,
                    args,
                    tally.curve_points,
                    tally.replacements is not None,
                    tally.histories is not None,
                )
            )
            self._executor = montecarlo.process_pool(
                jobs, initializer=_start_worker, initargs=(run,)
            )
        else:
            self._context = rbd._context(*args)

    def __call__(self, start: int, stop: int) -> None:
        """Simulations ``start`` to ``stop - 1``."""
        if self._executor is None:
            for replication in range(start, stop):
                self._tally.add(
                    self._rbd._replicate(self._context, replication)
                )
                self._progress.update()
            return
        spans = [
            (first, min(first + PARALLEL_BLOCK, stop))
            for first in range(start, stop, PARALLEL_BLOCK)
        ]
        for block in self._executor.map(_simulate_block, spans):
            self._tally.merge(block)
            self._progress.update(block.n)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()


class _ShardRunner:
    """Runs a run's simulations as shards (see ``repyability.rbd.shards``):
    each range of simulations asked for is cut into shards at the multiples
    of ``step``, which ``shard_map(run_shard, shards)`` runs wherever it
    sends them, and the partials it gives back, checked to be those
    shards', are merged into the tally in order. ``template`` is what
    every shard of the run holds (see ``RepairableRBD._shard_template``)."""

    def __init__(self, rbd, tally, progress, shard_map, template, step):
        from repyability.rbd.chunks import _run_key

        self._tally = tally
        self._progress = progress
        self._map = shard_map
        self._template = template
        self._step = step
        self._key = _run_key({**template, "fingerprint": rbd._fingerprint()})

    def __call__(self, start: int, stop: int) -> None:
        """Simulations ``start`` to ``stop - 1``."""
        from repyability.rbd.chunks import SimulationChunk, _run_key
        from repyability.rbd.shards import run_shard

        if stop <= start:
            return
        step = self._step
        cuts = [start, *range((start // step + 1) * step, stop, step), stop]
        ranges = list(zip(cuts, cuts[1:]))
        shards = [_shard_bytes(self._template, a, b) for a, b in ranges]
        chunks = []
        for saved in self._map(run_shard, shards):
            chunk = SimulationChunk._load(saved)
            chunks.append(chunk)
            self._progress.update(chunk.n_simulations)
        chunks.sort(key=lambda chunk: chunk.ranges[0])
        if [chunk.ranges for chunk in chunks] != [[r] for r in ranges] or any(
            _run_key(chunk.settings) != self._key for chunk in chunks
        ):
            raise ValueError(
                "shard_map gave back other partials than those of the "
                "shards it was given: it must give back run_shard's result "
                "for each shard, as map(run_shard, shards) does."
            )
        for chunk in chunks:
            self._tally.merge(chunk._tally)

    def close(self) -> None:
        pass


#: The per-action costs: charged at each preventive action or inspection.
_ACTION_COST_KEYS = ("preventive_cost", "inspection_cost")

#: Why a simulation does not start in the long-run state.
_STATIONARY_SIMULATION = (
    "A simulation starts from given states, not from the long-run one: give "
    "each component's NodeState (its age, or how long it has been down). "
    "The exact methods (point_availability, mission_availability, "
    "expected_failures, expected_events, expected_cost, point_capacity and "
    "mission_capacity) take state='stationary'."
)


def _validate_crews(crews) -> Optional[int]:
    """``repair_crews``, validated: a whole number of crews, at least 1, or
    None for as many as are needed."""
    if crews is None:
        return None
    if isinstance(crews, bool) or int(crews) != crews or crews < 1:
        raise ValueError(
            f"repair_crews must be a whole number, 1 or more, or None; got "
            f"{crews!r}."
        )
    return int(crews)


def _validate_priority(node, priority) -> float:
    """A component's ``"priority"`` for a crew, validated: a finite
    number."""
    if isinstance(priority, bool) or not isinstance(
        priority, (int, float, np.integer, np.floating)
    ):
        raise ValueError(
            f"Component {node!r}: priority must be a number, got "
            f"{priority!r}."
        )
    if not np.isfinite(priority):
        raise ValueError(
            f"Component {node!r}: priority must be finite, got {priority!r}."
        )
    return float(priority)


def _safe_mean(model) -> float:
    """A model's mean (see ``model_mean``), or NaN if it has none."""
    try:
        return model_mean(model)
    except Exception:
        return float("nan")


@dataclass(order=True, slots=True)
class Event:
    """A component's change of state in the availability simulation.

    Events compare by ``time`` alone (the other fields take no part in
    comparisons), so the simulation's event queue releases them in time
    order.

    Attributes
    ----------
    time : float
        When the change happens, on the simulation clock.
    component : Hashable
        The node whose state changes.
    status : bool
        The node's state after the event: False for a failure, True for a
        restoration.
    preventive : bool
        True for scheduled preventive maintenance, by default False. With
        ``status`` False it starts a planned outage (maintenance that takes
        time, or a nested RBD's planned outage); with ``status`` True it
        ends one or, if the node is up, renews it in place (maintenance in
        zero time).
    inspection : bool
        True for an inspection of a node whose failures are hidden, by
        default False. With ``status`` True it is a test in zero time of a
        working node, or the end of a test that took time; with ``status``
        False it starts such a test (a planned outage) or, if the node has
        failed, finds the failure, and the repair starts.
    missed : bool
        True for an inspection of a failed node that does not find the
        failure (a test whose coverage is below 1), by default False: the
        node stays down until a full test finds it.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import Event
    >>> Event(12.0, "pump", True) > Event(10.0, "valve", False)
    True
    """

    time: float
    component: Hashable = field(compare=False)
    status: bool = field(compare=False)
    preventive: bool = field(default=False, compare=False)
    inspection: bool = field(default=False, compare=False)
    missed: bool = field(default=False, compare=False)


class _Preventive(NamedTuple):
    """A node's scheduled preventive maintenance: a replacement every
    ``interval`` under the ``"age"`` or ``"block"`` policy, or under
    ``"condition"`` an inspection at every multiple of ``interval`` that
    replaces the unit if it is more likely than ``threshold`` to fail
    before the next; a replacement takes a time drawn from ``duration``
    (None: no time)."""

    interval: float
    policy: str
    duration: Any
    threshold: float = 0.0
    #: Under ``"age"``, the age from which the unit is renewed early at a
    #: stop of its maintenance group (#108); ``inf`` for never.
    opportunity: float = math.inf

    def due(self, renewed: float) -> float:
        """When the next preventive action (under ``"condition"``, the next
        inspection) falls, for a unit put into service as new at
        ``renewed``, or inspected then."""
        if self.policy == "age":
            return renewed + self.interval
        # Block, or inspections: the next multiple of the interval after
        # ``renewed`` (a unit renewed on the schedule is not renewed again
        # there).
        due = float(np.floor(renewed / self.interval) + 1.0) * self.interval
        return due if due > renewed else due + self.interval


class _Imperfect(NamedTuple):
    """A component's imperfect repair (#109): ``kijima`` (``"kijima1"`` or
    ``"kijima2"``) and its restoration factor ``q`` (0 renews, 1 is minimal
    repair), and the failure at which it is replaced instead, counting
    from its last renewal (None for never)."""

    kijima: str
    q: float
    replace_after: Optional[int]


class _MaintenanceGroup(NamedTuple):
    """A maintenance group (#108): its members, the set-up cost charged
    once at each of its stops, and whether every system outage is a stop
    too."""

    members: Tuple[Hashable, ...]
    setup_cost: float
    system_down: bool


def _failures_between(model, ages: np.ndarray) -> np.ndarray:
    """The probability that a unit of ``model`` that has survived to each
    of ``ages`` but the last fails before the next, ``1 - sf(next) /
    sf(age)``: from its cumulative hazard where the model gives one, which
    keeps a small probability precise."""
    ages = np.asarray(ages, dtype=float)
    out = np.full(len(ages) - 1, np.nan)
    Hf = getattr(model, "Hf", None)
    if Hf is not None:
        with np.errstate(all="ignore"):
            hazard = np.asarray(Hf(ages), dtype=float)
            out = -np.expm1(-np.maximum(np.diff(hazard), 0.0))
    missing = np.flatnonzero(np.isnan(out))
    if missing.size:
        with np.errstate(all="ignore"):
            survival = np.asarray(model.sf(ages), dtype=float)
            now, later = survival[missing], survival[missing + 1]
            fails = np.where(now > 0.0, 1.0 - later / now, 1.0)
        out[missing] = np.clip(np.nan_to_num(fails, nan=1.0), 0.0, 1.0)
    return out


class _Inspection(NamedTuple):
    """A node's periodic inspection, which is what finds its (hidden)
    failures: at ``offset`` and every ``interval`` after it (at every
    multiple of ``interval`` with no offset), taking a time drawn from
    ``duration`` (None: no time). A test finds a failure with probability
    ``coverage``; the tests at ``offset`` and every ``full_test`` after it
    (a whole multiple of ``interval``) find every failure, and with a
    ``coverage`` of 1 every test does."""

    interval: float
    duration: Any
    offset: float = 0.0
    coverage: float = 1.0
    full_test: Optional[float] = None

    @property
    def partial(self) -> bool:
        """Whether its tests can miss a failure (a coverage below 1)."""
        return self.coverage < 1.0

    @property
    def period(self) -> float:
        """The time after which its tests repeat: its full tests' interval
        when its tests can miss a failure, else its interval."""
        if self.partial and self.full_test is not None:
            return self.full_test
        return self.interval

    @property
    def per_full_test(self) -> int:
        """Its tests from one full test to the next."""
        return int(round(self.period / self.interval))

    def due(self, t: float) -> float:
        """The first inspection after ``t``."""
        if not self.offset:
            k = np.floor(t / self.interval) + 1.0
            due = float(k * self.interval)
            return due if due > t else float((k + 1.0) * self.interval)
        k = np.floor((t - self.offset) / self.interval) + 1.0
        due = float(self.offset + k * self.interval)
        if due > t:
            return due
        return float(self.offset + (k + 1.0) * self.interval)

    def finds(self, t: float) -> float:
        """The inspection that finds a failure at ``t``: the first at or
        after it."""
        if not self.offset:
            k = np.ceil(t / self.interval)
            due = float(k * self.interval)
            return due if due >= t else float((k + 1.0) * self.interval)
        k = np.ceil((t - self.offset) / self.interval)
        due = float(self.offset + k * self.interval)
        if due >= t:
            return due
        return float(self.offset + (k + 1.0) * self.interval)

    def is_full(self, t: float) -> bool:
        """Whether the test at ``t`` (one of its tests) finds every
        failure."""
        if not self.partial:
            return True
        k = int(round((t - self.offset) / self.interval))
        return k % self.per_full_test == 0

    def without_offset(self) -> "_Inspection":
        """The schedule on a calendar from 0 (a start from a state, whose
        phase places it)."""
        return self._replace(offset=0.0) if self.offset else self


class _Standby(NamedTuple):
    """A standby group (a component spec's ``"standby"``): ``units``
    identical units, ``k`` of which must operate, the rest waiting as
    spares that age at ``dormancy_factor`` of the operating rate, switched
    in with ``switching_probability``."""

    units: int
    k: int
    dormancy_factor: float
    switching_probability: float


def _constant_rate(model) -> Optional[float]:
    """The failure rate of a model whose rate is constant (an exponential
    life, or a Weibull of shape 1), or None."""
    try:
        t = model_mean(model) * np.array([0.01, 0.5, 1.0, 3.0])
        hazard = np.asarray(model.hf(t), dtype=float).ravel()
        survival = np.asarray(model.sf(t), dtype=float).ravel()
    except Exception:
        return None
    rate = float(hazard[0])
    if not (np.isfinite(rate) and rate > 0.0):
        return None
    constant = np.allclose(hazard, rate, rtol=1e-9, atol=0.0)
    exponential = np.allclose(survival, np.exp(-rate * t), rtol=1e-9)
    return rate if constant and exponential else None


def _repair_rate(model) -> Optional[float]:
    """The rate of an exponential repair time, ``inf`` for an instant one
    (in no time), or None."""
    if _safe_mean(model) == 0.0:
        return math.inf
    return _constant_rate(model)


def _cdf(model) -> Callable[[np.ndarray], np.ndarray]:
    """A model's CDF, for arrays of times (0 where it gives none)."""

    def cdf(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        with np.errstate(all="ignore"):
            values = np.asarray(model.ff(x), dtype=float).reshape(x.shape)
        return np.nan_to_num(values, nan=0.0)

    return cdf


def _fleet(probabilities: np.ndarray, fleet: int) -> np.ndarray:
    """The distribution of the sum of ``fleet`` independent counts, each
    with ``probabilities`` for ``0, 1, 2, ...``."""
    from scipy.signal import fftconvolve

    def product(a, b):
        out = np.maximum(fftconvolve(a, b), 0.0)
        keep = np.nonzero(out > 1e-16)[0]
        return out[: keep[-1] + 1] if keep.size else np.ones(1)

    total, power = np.ones(1), np.asarray(probabilities, dtype=float)
    while fleet:
        if fleet & 1:
            total = product(total, power)
        fleet >>= 1
        if fleet:
            power = product(power, power)
    return total / total.sum()


def _whole(name: str, value, least: int = 1) -> int:
    """``value`` as a whole number of at least ``least``, or a
    ValueError naming it."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or int(value) != value
        or value < least
    ):
        raise ValueError(
            f"{name} must be a whole number, {least} or more; got {value!r}."
        )
    return int(value)


#: What the spares counts say when a component can wait for a repair crew.
_SPARES_CREWS = (
    "the spares counts assume",
    "Count them by simulation: spares_demand(method='simulate').",
)

#: The most nested RBDs the crews' chain is followed over time with: it
#: is solved for each pattern of them up and down.
_MAX_CREW_NESTED = 10
_ALLOCATION_CREWS = (
    "the allocations assume",
    "Compare designs by their long-run values, or simulate them with "
    "compare().",
)


def _curve_points(value) -> Optional[int]:
    """``curve_points`` checked: None, or a whole number of at least 1."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"curve_points must be a whole number of grid steps, got "
            f"{value!r}."
        )
    if value < 1:
        raise ValueError(f"curve_points must be at least 1, got {value!r}.")
    return int(value)


def _is_junction(name, component) -> bool:
    """Whether ``component`` makes node ``name`` a junction, which never
    fails: ``PerfectReliability`` itself, or a spec whose life it is (#175,
    #182). Such a spec may give a repair model, never used, but nothing
    that only a part that fails or is maintained has."""
    if component is PerfectReliability:
        return True
    if not (
        isinstance(component, dict)
        and component.get("reliability") is PerfectReliability
    ):
        return False
    other = sorted(
        str(key)
        for key, value in component.items()
        if key not in ("reliability", "repairability") and value is not None
    )
    if other:
        raise ValueError(
            f"Component {name!r} never fails (its reliability is "
            f"PerfectReliability), so it takes no {', '.join(other)}: give "
            "it as PerfectReliability alone, a junction that always works."
        )
    return True


def _horizon(horizon) -> float:
    """``horizon`` as a finite, non-negative float, or a ValueError."""
    try:
        value = float(horizon)
    except (TypeError, ValueError):
        value = float("nan")
    if not (np.isfinite(value) and value >= 0.0):
        raise ValueError(
            f"horizon must be a finite, non-negative number, got {horizon!r}."
        )
    return value


def _objective(cost: float, availability: float, max_cost_rate) -> float:
    """What an interval choice minimises: the cost rate, or the
    unavailability when the cost rate is capped."""
    return 1.0 - availability if max_cost_rate is not None else cost


def _meets(cost, availability, min_availability, max_cost_rate) -> bool:
    """Whether long-run values meet an interval choice's target."""
    if min_availability is not None and availability < min_availability:
        return False
    if max_cost_rate is not None and cost > max_cost_rate:
        return False
    return True


def _allocation_target(target) -> float:
    """An availability allocation's target, checked to be in [0, 1]."""
    value = _as_float(target)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"target must be a number in [0, 1], got {target!r}.")
    return value


def _as_float(value) -> float:
    """``value`` as a float, or NaN if it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _choose_intervals(
    nodes: list,
    evaluate,
    starts: list,
    bounds: Tuple[np.ndarray, np.ndarray],
    min_availability: Optional[float],
    max_cost_rate: Optional[float],
) -> dict:
    """The intervals (by node) that minimise the cost rate, or, with a cost
    cap, the unavailability, subject to the target: SLSQP over the
    logarithms of the intervals from each of ``starts``, keeping the best
    that meets the target. ``evaluate(intervals)`` gives their exact
    ``(cost rate, availability)``. A ValueError if no intervals in
    ``bounds`` meet the target, giving the best they can do."""
    low, high = bounds
    box = list(zip(low, high))
    options = {"ftol": 1e-10, "maxiter": 200, "eps": 1e-6}
    cache: dict = {}

    def point(x) -> tuple:
        """``x`` rounded: the intervals evaluated, and returned."""
        return tuple(np.round(np.asarray(x, dtype=float), 12))

    def values(x) -> Tuple[float, float]:
        key = point(x)
        if key not in cache:
            cache[key] = evaluate(
                {node: math.exp(xi) for node, xi in zip(nodes, key)}
            )
        return cache[key]

    # The objective and the constraint, scaled to be about 1.
    cost0, availability0 = values(high)
    down0 = max(1.0 - availability0, 1e-300)
    rate0 = max(abs(cost0), 1e-300)

    def unavailable(x):
        return (1.0 - values(x)[1]) / down0

    def cost(x):
        return values(x)[0] / rate0

    if max_cost_rate is not None:
        objective = unavailable

        def slack(x):
            return (max_cost_rate * (1.0 - 1e-12) - values(x)[0]) / (
                max_cost_rate
            )

    else:
        objective = cost

        def slack(x):
            return (values(x)[1] - min_availability - 1e-12) / (
                1.0 - min_availability
            )

    constrained = min_availability is not None or max_cost_rate is not None
    if constrained and not any(
        _meets(*values(start), min_availability, max_cost_rate)
        for start in starts
    ):
        # Can the target be met at all? The most available intervals, or
        # the cheapest, from the starts nearest to it first.
        extreme = unavailable if min_availability is not None else cost
        found = []
        for start in sorted(starts, key=extreme):
            reach = minimize(
                extreme, start, method="SLSQP", bounds=box, options=options
            ).x
            found.append(reach)
            if _meets(*values(reach), min_availability, max_cost_rate):
                break
        else:
            best_cost, best_availability = values(
                min(found + list(starts), key=extreme)
            )
            if min_availability is not None:
                raise ValueError(
                    f"min_availability {min_availability} cannot be met: the "
                    f"most any intervals give is {best_availability:.6g}."
                )
            raise ValueError(
                f"max_cost_rate {max_cost_rate} cannot be met: the least any "
                f"intervals cost is {best_cost:.6g} per unit time."
            )
        starts = [reach] + list(starts)
    best, best_value = None, math.inf
    for start in starts:
        found = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=box,
            constraints=(
                [{"type": "ineq", "fun": slack}] if constrained else []
            ),
            options=options,
        )
        for x in (found.x, start):
            rate, availability = values(x)
            if not _meets(rate, availability, min_availability, max_cost_rate):
                continue
            value = _objective(rate, availability, max_cost_rate)
            if value < best_value:
                best, best_value = np.asarray(x, dtype=float), value
    assert best is not None  # the start that meets the target is kept
    return {node: math.exp(xi) for node, xi in zip(nodes, point(best))}


#: Allowed intervals are chosen by trying every combination, up to this many;
#: by a local search beyond.
_MAX_COMBINATIONS = 2000


def _choose_from(
    nodes: list,
    options: dict,
    evaluate,
    min_availability: Optional[float],
    max_cost_rate: Optional[float],
) -> dict:
    """The intervals, each from its node's ``options``, that minimise the
    cost rate, or with a cost cap the unavailability, subject to the target:
    every combination when there are at most 2000, a local search (one
    interval changed at a time, from several starts) otherwise. A
    ValueError if none meets the target, giving the best any does."""

    def merit(intervals: dict) -> Tuple[float, float]:
        """(How far from the target, the objective): lower is better."""
        cost, availability = evaluate(intervals)
        if min_availability is not None:
            short = max(0.0, min_availability - availability)
        elif max_cost_rate is not None:
            short = max(0.0, cost - max_cost_rate)
        else:
            short = 0.0
        return short, _objective(cost, availability, max_cost_rate)

    count = math.prod(len(options[node]) for node in nodes)
    if count <= _MAX_COMBINATIONS:
        candidates = (
            dict(zip(nodes, combination))
            for combination in itertools.product(
                *(options[node] for node in nodes)
            )
        )
        best = min(candidates, key=merit)
    else:
        starts = [
            {node: options[node][0] for node in nodes},
            {node: options[node][-1] for node in nodes},
            {node: options[node][len(options[node]) // 2] for node in nodes},
        ]
        best = None
        for current in starts:
            value = merit(current)
            improved = True
            while improved:
                improved = False
                for node in nodes:
                    for option in options[node]:
                        candidate = {**current, node: option}
                        candidate_value = merit(candidate)
                        if candidate_value < value:
                            current, value = candidate, candidate_value
                            improved = True
            if best is None or value < merit(best):
                best = current
    assert best is not None
    cost, availability = evaluate(best)
    if not _meets(cost, availability, min_availability, max_cost_rate):
        if min_availability is not None:
            raise ValueError(
                f"min_availability {min_availability} cannot be met: the "
                f"most the allowed intervals give is {availability:.6g}."
            )
        raise ValueError(
            f"max_cost_rate {max_cost_rate} cannot be met: the least the "
            f"allowed intervals cost is {cost:.6g} per unit time."
        )
    return best


#: Why antithetic pairs and common random numbers (``compare``) are refused
#: when some component's draws do not come from a stream.
_UNSTREAMED = (
    "Antithetic and common random numbers need every component's draws to "
    "be replayable (surpyval parametric distributions and the composite "
    "models built from them)."
)


def _failed_by(component: NonRepairable, age: float, survives: float) -> float:
    """The probability that a component's unit fails before ``age``: from
    its model's own ``ff``, which keeps a small one's precision, or one less
    ``survives`` (its survival there) for a non-parametric model, whose
    survival the component interpolates."""
    if component.model_parameterization == "non-parametric":
        return 1.0 - survives
    return float(np.ravel(component.reliability.ff(age))[0])


def _log_kept(rate: float, inspection: "_Inspection") -> float:
    """``log(rho)``, ``rho = 1 - (1 - c) (1 - exp(-rate * interval))``: the
    chance that a unit with hidden failures, up after a test, is up after
    the next one or had its failure found by it (a coverage ``c``)."""
    missed = (1.0 - inspection.coverage) * -math.expm1(
        -rate * inspection.interval
    )
    return math.log1p(-missed)


def _tests_in(inspection: "_Inspection", period: float) -> list:
    """The times of an inspection's tests in ``[0, period)`` that are not
    multiples of its interval (those of a schedule with an offset), on a
    calendar that repeats every ``period``."""
    if not inspection.offset:
        return []
    count = int(round(period / inspection.interval))
    times = inspection.offset + inspection.interval * np.arange(count)
    return list(np.where(times >= period, times - period, times))


def _test_finds(source, coverage: float) -> bool:
    """Whether a test that finds a failure with probability ``coverage``
    finds one: from the component's detection stream, or numpy's global
    RNG."""
    if isinstance(source, _StreamedComponent):
        return source.test_uniform() < coverage
    return float(np.random.random()) < coverage


def _number_or_nan(value) -> float:
    """``value`` as a float, or nan if it is not a number."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _common_period(intervals: Iterable[float]) -> float:
    """The least common multiple of the inspection intervals: the period
    after which the schedules repeat together."""
    intervals = sorted(set(intervals))
    if len(intervals) == 1:
        return intervals[0]
    fractions = [Fraction(x).limit_denominator(10**6) for x in intervals]
    if any(
        abs(float(f) - x) > 1e-12 * x for f, x in zip(fractions, intervals)
    ):
        raise NotImplementedError(
            f"The inspection intervals {intervals} have no common period, "
            "so the long-run values cannot be averaged over one: estimate "
            "them by simulation, with availability() or cost()."
        )
    numerator = math.lcm(*(f.numerator for f in fractions))
    denominator = math.gcd(*(f.denominator for f in fractions))
    return numerator / denominator


def combined_timeline(
    timeline_1: List[Tuple[float, int]], timeline_2: List[Tuple[float, int]]
):
    """Merge two up/down timelines and count how many are up over time.

    Each timeline is a list of ``(time, change)`` pairs, as the
    availability simulation records them for a node and for the system: a
    first entry ``(0.0, 1)`` if it starts up or ``(0.0, 0)`` if it starts
    down, then ``(t, 1)`` for each restoration and ``(t, -1)`` for each
    failure, and a last entry ``(t_end, 0)`` closing the window. The
    changes of the two timelines are summed at equal times and accumulated
    in time order.

    Parameters
    ----------
    timeline_1 : list[tuple[float, int]]
        The first timeline, e.g. a node's.
    timeline_2 : list[tuple[float, int]]
        The second timeline, e.g. the system's.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The distinct times in increasing order, and at each time the number
        of the two (0, 1 or 2) that are up from that time until the next.

    Examples
    --------
    A node down on ``[2, 3)`` and a system down on ``[2, 5)``, over a
    window of length 10:

    >>> from repyability.rbd.repairable_rbd import combined_timeline
    >>> node = [(0.0, 1), (2.0, -1), (3.0, 1), (10.0, 0)]
    >>> system = [(0.0, 1), (2.0, -1), (5.0, 1), (10.0, 0)]
    >>> times, n_up = combined_timeline(node, system)
    >>> times.tolist(), n_up.tolist()
    ([0.0, 2.0, 3.0, 5.0, 10.0], [2, 0, 1, 2, 2])
    """
    joint_timeline: defaultdict = defaultdict(lambda: 0)
    for t, e in timeline_1 + timeline_2:
        joint_timeline[t] += e
    events: np.ndarray = np.fromiter(joint_timeline.values(), dtype=np.int8)
    timeline: np.ndarray = np.fromiter(joint_timeline.keys(), dtype=np.float64)
    idx = np.argsort(timeline)
    timeline = timeline[idx]
    events = events[idx]
    events = events.cumsum()
    return timeline, events


def intersection(timeline, event_cumsum):
    """Total time during which both of two timelines are up.

    Sums the lengths of the intervals ``[timeline[j], timeline[j + 1])``
    on which ``event_cumsum[j] == 2``. Pass ``2 - event_cumsum`` instead to
    get the total time during which both are down.

    Parameters
    ----------
    timeline : numpy.ndarray
        Increasing times, as returned by ``combined_timeline``.
    event_cumsum : numpy.ndarray
        How many of the two are up from each time, as returned by
        ``combined_timeline``.

    Returns
    -------
    float
        The total length of those intervals.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     combined_timeline,
    ...     intersection,
    ... )
    >>> node = [(0.0, 1), (2.0, -1), (3.0, 1), (10.0, 0)]
    >>> system = [(0.0, 1), (2.0, -1), (5.0, 1), (10.0, 0)]
    >>> times, n_up = combined_timeline(node, system)
    >>> float(intersection(times, n_up))  # both up
    7.0
    >>> float(intersection(times, 2 - n_up))  # both down
    1.0
    """
    from_idx = np.where(event_cumsum[:-1] == 2)[0]
    to_idx = from_idx + 1
    intersection = timeline[to_idx] - timeline[from_idx]
    return intersection.sum()


def union(timeline, event_cumsum):
    """Total time during which at least one of two timelines is up.

    Sums the lengths of the intervals ``[timeline[j], timeline[j + 1])``
    on which ``event_cumsum[j] > 0``. Pass ``2 - event_cumsum`` instead to
    get the total time during which at least one is down.

    Parameters
    ----------
    timeline : numpy.ndarray
        Increasing times, as returned by ``combined_timeline``.
    event_cumsum : numpy.ndarray
        How many of the two are up from each time, as returned by
        ``combined_timeline``.

    Returns
    -------
    float
        The total length of those intervals.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import combined_timeline, union
    >>> node = [(0.0, 1), (2.0, -1), (3.0, 1), (10.0, 0)]
    >>> system = [(0.0, 1), (2.0, -1), (5.0, 1), (10.0, 0)]
    >>> times, n_up = combined_timeline(node, system)
    >>> float(union(times, n_up))  # either up
    9.0
    >>> float(union(times, 2 - n_up))  # either down
    3.0
    """
    from_idx = np.where(event_cumsum[:-1] > 0)[0]
    to_idx = from_idx + 1
    union = timeline[to_idx] - timeline[from_idx]
    return union.sum()


def intersection_over_union(
    node_timeline: List[Tuple[float, int]],
    system_timeline: List[Tuple[float, int]],
):
    """Intersection over union of a node's and the system's up time.

    The time both are up divided by the time at least one is up, for one
    pair of timelines. (``RepairableRBD.availability`` computes its ``iou``
    measure from totals over all its simulations instead.) If neither is
    ever up the result is nan, with a numpy warning.

    Parameters
    ----------
    node_timeline : list[tuple[float, int]]
        The node's timeline, in the format described in
        ``combined_timeline``.
    system_timeline : list[tuple[float, int]]
        The system's timeline, in the same format.

    Returns
    -------
    float
        The ratio, in ``[0, 1]``: 1 when the two are up at exactly the same
        times.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import intersection_over_union
    >>> node = [(0.0, 1), (2.0, -1), (3.0, 1), (10.0, 0)]
    >>> system = [(0.0, 1), (2.0, -1), (5.0, 1), (10.0, 0)]
    >>> round(float(intersection_over_union(node, system)), 4)  # 7 / 9
    0.7778
    """
    timeline, event_cumsum = combined_timeline(node_timeline, system_timeline)
    return intersection(timeline, event_cumsum) / union(timeline, event_cumsum)


def time_at_status(timeline, status):
    """Sum the gaps that follow a timeline's entries of a given value.

    Sums ``t[j + 1] - t[j]`` over every entry ``(t[j], value)`` of
    ``timeline`` whose value equals ``status``. For a single node's or the
    system's timeline, in the format described in ``combined_timeline``,
    ``status=1`` gives its total up time: every up period starts at the
    initial ``(0.0, 1)`` entry or at a restoration. (``status=-1`` would
    miss a down period at the start, whose entry is ``(0.0, 0)``; the
    simulation only uses ``status=1``.)

    Parameters
    ----------
    timeline : list[tuple[float, int]]
        One node's or the system's timeline.
    status : int
        The entry value whose following gaps are summed.

    Returns
    -------
    float
        The total time.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import time_at_status
    >>> node = [(0.0, 1), (2.0, -1), (3.0, 1), (10.0, 0)]
    >>> float(time_at_status(node, 1))
    9.0
    """
    t = np.array([a for a, _ in timeline])
    events = np.array([b for _, b in timeline])
    from_idx = np.where(events[:-1] == status)[0]
    to_idx = from_idx + 1
    union = t[to_idx] - t[from_idx]
    return union.sum()


#: Grid steps over a component's typical up time, for its point
#: availability (see ``_point_availability``): the error falls as the
#: square of the step, to about 1e-8 here.
_POINT_STEPS = 2000
#: The most grid points for one component's point availability.
_POINT_MAX = 2**22
#: The most pieces ``mission_availability`` integrates over.
_MISSION_POINTS = 5_000_000


def _check_window(t_simulation) -> float:
    """A simulation's window, ``t_simulation``, as a float: a positive and
    finite time."""
    if (
        isinstance(t_simulation, bool)
        or not isinstance(t_simulation, (int, float, np.integer, np.floating))
        or not (math.isfinite(t_simulation) and t_simulation > 0.0)
    ):
        raise ValueError(
            f"t_simulation must be a positive, finite time, got "
            f"{t_simulation!r}."
        )
    return float(t_simulation)


def _check_times(x) -> np.ndarray:
    """``x`` as a 1-d float array of times, all finite and non-negative."""
    times = np.atleast_1d(np.asarray(x, dtype=float))
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError(f"Times must be finite and non-negative, got {x!r}.")
    return times


def _sf_values(sf, x: np.ndarray) -> np.ndarray:
    """A survival function's values at ``x``, as floats in ``x``'s shape."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(sf(x), dtype=float).reshape(np.shape(x))


def _up_scale(model, age: Optional[float]) -> float:
    """A typical up time for a unit with lifetime ``model`` (replaced at
    ``age``, if given), to size its point-availability grid by: the smaller
    of its median and the width of its middle 80%; else its typical failure
    time (NaN if it has none, e.g. it never fails)."""
    scale = float("nan")
    qf = getattr(model, "qf", None)
    if qf is not None:
        try:
            with np.errstate(all="ignore"):
                q10, q50, q90 = np.ravel(
                    np.asarray(qf(np.array([0.1, 0.5, 0.9])), dtype=float)
                )
            widths = [w for w in (q50, q90 - q10) if np.isfinite(w) and w > 0]
            if widths:
                scale = min(widths)
        except Exception:  # a model whose qf cannot take these
            pass
    if not np.isfinite(scale):
        try:
            scale = failure_time_scale(model)
        except Exception:  # no mean either (e.g. never fails)
            scale = float("nan")
    if age is not None:
        scale = age if not np.isfinite(scale) else min(scale, age)
    return float(scale)


def _settling(curves) -> Tuple[float, Optional[float]]:
    """When the point availabilities ``curves`` (see
    ``_point_availability``) all settle: the time after which they are
    constant (a period of None) or repeat together with the period returned
    (a time of inf if they never do, as far as is known)."""
    curves = list(curves)
    settle = max((float(curve.settle) for curve in curves), default=0.0)
    periods = {curve.period for curve in curves if curve.period is not None}
    if not periods:
        return settle, None
    try:
        return settle, _common_period(periods)
    except NotImplementedError:  # no common period
        return np.inf, None


class _Jumps(NamedTuple):
    """What the system does at the times its nodes fail or are taken down
    at exact times (see ``RepairableRBD._atom_groups``): the times, the
    probability at each that the system fails there, and that a planned
    outage takes it down there, and how much its point availability there
    falls with them (``drop``)."""

    times: np.ndarray
    failures: np.ndarray
    planned: np.ndarray
    drop: np.ndarray


def _matched(times: np.ndarray, at: np.ndarray, values: np.ndarray):
    """``values``, given at the times ``at``, summed at each of ``times``
    (sorted) that they fall on exactly; 0 at the others."""
    index = np.searchsorted(times, at)
    hit = index < len(times)
    hit[hit] = times[index[hit]] == at[hit]
    return np.bincount(index[hit], values[hit], len(times))


def _shaped(values: np.ndarray, t, times: np.ndarray):
    """``values``, one per time, as a float for a scalar ``t``, else in the
    shape of ``times``."""
    values = np.asarray(values, dtype=float)
    if np.ndim(t) == 0:
        return float(values[0])
    return values.reshape(times.shape)


def _squeeze_values(d: dict) -> dict:
    """Convert a dict of 1-element arrays (as produced by the base RBD
    importance helpers) to a dict of plain floats. Steady-state availability
    has no time dimension, so the repairable importances return floats."""
    return {k: float(np.asarray(v).reshape(-1)[0]) for k, v in d.items()}


def _safe_ratio(numerator, denominator):
    """Ratio that is 0 when the denominator is 0.

    Several criticality/importance measures divide by a total (system uptime,
    downtime, failures, restorations, or a union of intervals) that can
    legitimately be 0 -- e.g. when a redundant component is forced working so
    the system never fails, or a component is forced broken. With nothing to
    attribute, the measure is 0 rather than a NaN/inf (or a crash).
    """
    return numerator / denominator if denominator else 0.0


def _mean_cost(cost) -> float:
    """A declared cost's expected value: the number itself, or the mean of a
    cost distribution."""
    if isinstance(cost, float):
        return cost
    return float(np.ravel(cost.mean())[0])


def failure_criticality_index_per_system_failures(FCI, system_failures):
    """Share of all system failures that each node caused.

    A node causes a system failure when its own failure takes the system
    from up to down. For each node this is the number of system failures
    it caused divided by ``system_failures``; every node gets 0 when there
    were no system failures.

    Parameters
    ----------
    FCI : dict
        Node -> counts, with the number of system failures the node caused
        under ``"system_failures"`` (as accumulated by
        ``RepairableRBD.availability``).
    system_failures : int
        The total number of system failures.

    Returns
    -------
    dict
        Node -> share of the system failures, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     failure_criticality_index_per_system_failures as per_system,
    ... )
    >>> counts = {"a": {"system_failures": 3}, "b": {"system_failures": 1}}
    >>> per_system(counts, 4)
    {'a': 0.75, 'b': 0.25}
    """
    fci = {}
    for node in FCI.keys():
        if system_failures == 0:
            # No system failures occurred (e.g. a redundant node was forced
            # working, or the system was highly reliable over the simulated
            # window), so no node can be credited with causing one.
            fci[node] = 0
        else:
            fci[node] = FCI[node]["system_failures"] / system_failures
    return fci


def failure_criticality_index_per_component_failures(FCI):
    """Share of each node's own failures that caused a system failure.

    For each node: the number of system failures it caused (its failures
    that took the system from up to down) divided by its number of
    failures, or 0 if it never failed.

    Parameters
    ----------
    FCI : dict
        Node -> counts, with the system failures the node caused under
        ``"system_failures"`` and its own failures under
        ``"component_failures"`` (as accumulated by
        ``RepairableRBD.availability``).

    Returns
    -------
    dict
        Node -> share of the node's failures, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     failure_criticality_index_per_component_failures as per_component,
    ... )
    >>> per_component({"a": {"system_failures": 3, "component_failures": 4}})
    {'a': 0.75}
    """
    fci = {}
    for node in FCI.keys():
        try:
            fci[node] = (
                FCI[node]["system_failures"] / FCI[node]["component_failures"]
            )
        except ZeroDivisionError:
            # If there were no component failures then there were no
            # system failures caused by that node
            fci[node] = 0
    return fci


def restoration_criticality_index_by_system(RCI, system_restorations):
    """Share of all system restorations that each node caused.

    A node causes a system restoration when its own restoration takes the
    system from down to up. For each node this is the number of system
    restorations it caused divided by ``system_restorations``; every node
    gets 0 when there were no system restorations.

    Parameters
    ----------
    RCI : dict
        Node -> counts, with the number of system restorations the node
        caused under ``"system_restorations"`` (as accumulated by
        ``RepairableRBD.availability``).
    system_restorations : int
        The total number of system restorations.

    Returns
    -------
    dict
        Node -> share of the system restorations, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     restoration_criticality_index_by_system as by_system,
    ... )
    >>> by_system({"a": {"system_restorations": 1}}, 2)
    {'a': 0.5}
    """
    rci = {}
    for node in RCI.keys():
        if system_restorations == 0:
            # No system restorations occurred, so no node can be credited with
            # causing one.
            rci[node] = 0
        else:
            rci[node] = RCI[node]["system_restorations"] / system_restorations
    return rci


def restoration_criticality_index_by_component(RCI):
    """Share of each node's own restorations that restored the system.

    For each node: the number of system restorations it caused (its
    restorations that took the system from down to up) divided by its
    number of restorations, or 0 if it was never restored.

    Parameters
    ----------
    RCI : dict
        Node -> counts, with the system restorations the node caused under
        ``"system_restorations"`` and its own restorations under
        ``"component_restorations"`` (as accumulated by
        ``RepairableRBD.availability``).

    Returns
    -------
    dict
        Node -> share of the node's restorations, in ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.repairable_rbd import (
    ...     restoration_criticality_index_by_component as by_component,
    ... )
    >>> counts = {"system_restorations": 1, "component_restorations": 4}
    >>> by_component({"a": counts})
    {'a': 0.25}
    """
    rci = {}
    for node in RCI.keys():
        try:
            rci[node] = (
                RCI[node]["system_restorations"]
                / RCI[node]["component_restorations"]
            )
        except ZeroDivisionError:
            # If there were no component restorations then there were no
            # times the system was restored by restoring this node.
            rci[node] = 0
    return rci


class RepairableRBD(RBD):
    """A reliability block diagram of repairable components.

    Each component alternates between working and failed: it fails after a
    time drawn from its reliability model, and is then repaired, as good
    as new, after a time drawn from its repairability model. Components fail
    and are repaired independently of one another and of the system state
    (unless ``repair_crews`` makes them wait for a crew), and a component
    keeps running (and can fail) while the system is down. The system is up
    whenever its working components connect the input node to the output
    node.

    Two kinds of analysis are offered:

    - Exact long-run (steady-state) metrics, built from each component's
      availability ``MTTF / (MTTF + MTTR)`` and failure frequency
      ``1 / (MTTF + MTTR)``: ``mean_availability``, ``node_availability``,
      ``system_failure_frequency``, ``mean_up_time``, ``mean_down_time``,
      ``mean_time_between_failures``, ``expected_cost_rate``,
      ``capacity_distribution`` (with node capacities) and the importance
      measures. A component under age or block replacement enters through
      its renewal cycle instead (see ``node_availability``). With fewer
      ``repair_crews`` than components, the long-run values come from the
      Markov chain of the components and the repair queue, for exponential
      components (see ``mean_availability``).
    - Monte-Carlo simulation of a finite window ``[0, t_simulation]`` that
      starts with every component working: ``availability`` (availability
      over time, criticality measures and, with node capacities, the
      capacity over time and the fraction of a demand delivered) and
      ``cost`` (the distribution of the window's cost).

    Times are in the time unit of the component models, and costs in the
    currency they are given in.

    Parameters
    ----------
    edges : Iterable[tuple[Hashable, Hashable]]
        The directed edges of the diagram, e.g.
        ``[("s", "a"), ("a", "b"), ("b", "t")]`` for ``"a"`` and ``"b"`` in
        series between the input node ``"s"`` and the output node ``"t"``.
    components : dict
        One entry per node other than the input and output nodes, keyed by
        node name. Each is one of:

        - A spec dict with ``"reliability"`` (a time-to-failure model, such
          as a fitted surpyval distribution) and ``"repairability"`` (a
          time-to-repair model, or ``"instant"`` for repair in zero time:
          the component still fails, and any repair or replace cost is
          charged, but it is never down), plus optional costs.
          ``"repair_cost"`` and ``"replace_cost"`` are charged at every
          failure of the component: each is a number, or a distribution of
          the cost (anything with ``qf`` and ``mean``, such as a fitted
          surpyval model) drawn afresh at each failure. ``"downtime_cost"``
          is a number charged per unit time the component is down, whether
          or not the system is. A cost left out, None or 0 prices nothing.
          ``"preventive"`` schedules preventive maintenance: a dict with an
          ``"interval"`` and optional ``"policy"``, ``"duration"`` and
          ``"cost"``. Under ``"policy": "age"`` (the default) the unit is
          replaced ``interval`` after it was last put into service as new
          (at time 0, or at the end of a repair or of the last preventive
          action), unless it fails first; under ``"block"`` at every
          multiple of ``interval``, whatever its age, unless it is down
          then. Under ``"condition"`` it is inspected at every multiple of
          ``interval`` while it is up, in no time, and replaced if it is
          then more likely than ``"threshold"`` (a probability, required)
          to fail before the next inspection, given its age ``a``:
          ``1 - R(a + interval) / R(a)``. Each inspection is charged
          ``"inspection_cost"``, a number or a distribution drawn afresh
          each time. The replacement takes a time drawn from ``"duration"``, a
          time-to-maintain model, during which the unit is down (a planned
          outage); ``"instant"`` (the default) takes no time, renewing the
          unit in place. Each replacement is charged ``"cost"``, a number
          or a distribution drawn afresh each time. The unit comes back as
          new: a failure it had yet to reach never happens. An
          ``interval`` of ``inf`` never maintains. Under ``"age"``, an
          ``"opportunity"`` (an age from 0 to ``interval``) renews the
          unit early, from that age, at a stop of its maintenance group
          (see ``"group"``), as its scheduled replacement would.
          ``"inspection"`` makes the component's failures *hidden*: a
          failure takes it down, but nobody knows until an inspection (a
          proof test) finds it, and only then does its repair start. It
          is a dict with an ``"interval"`` and optional ``"duration"``,
          ``"cost"``, ``"offset"``, ``"coverage"`` and ``"full_test"``: the
          component is inspected at every multiple of the (positive,
          finite) interval, from time 0, or from its ``"offset"`` (the time
          of its first test, at least 0 and less than the interval: tests of
          redundant components staggered). A test finds a failure with
          probability ``"coverage"`` (by default 1); a failure it misses
          stays hidden until a full test, every ``"full_test"`` (required
          with a coverage below 1, a whole multiple of the interval, from
          the offset), which finds every failure. The test takes a time
          drawn from ``"duration"``, a time-to-test model, during which the
          component is off-line (a planned outage) and does not age;
          ``"instant"`` (the default) takes no time. A failure found by a
          test is repaired once the test is done, and the repair and
          replace costs are charged when it is found. An inspection due
          while the component is being repaired is skipped. Each
          inspection is charged ``"cost"``, a number or a distribution
          drawn afresh each time. A component cannot have both a
          ``"preventive"`` schedule and an ``"inspection"``.
          ``"acquisition_cost"`` is the one-off cost of buying the unit, a
          number: it is not a running cost, so it is left out of
          ``expected_cost_rate`` and the simulated costs, and counted by
          ``total_cost`` and ``allocate_redundancy``. ``"priority"`` is the
          component's place in the queue for a repair crew (see
          ``repair_crews``): a number, higher first, by default 0.
          ``"group"`` names the component's maintenance group (see
          ``maintenance_groups``), any hashable value.
          ``"repair"`` makes its repairs imperfect, by Kijima's
          virtual-age models: a dict of a ``"model"``, ``"kijima1"`` or
          ``"kijima2"``, and a restoration factor ``"q"`` in [0, 1]. A
          repair after the unit has operated ``x`` since the last takes
          its virtual age from ``v`` to ``v + q * x`` (Kijima I) or
          ``q * (v + x)`` (Kijima II), and each life is drawn given it:
          ``H(v + X) = H(v) + E``, ``E`` exponential, ``H`` the cumulative
          hazard. ``q = 0`` renews the unit at every repair (the default)
          and ``q = 1`` is minimal repair. ``"replace_after"``, a whole
          number ``N`` (with a ``"repair"`` of ``q`` above 0), replaces the
          unit, as new, at the ``N``-th failure since it was renewed. A
          repair is then charged only its ``"repair_cost"``; a replacement
          its ``"repair_cost"`` and ``"replace_cost"``, and it uses a spare.
          A preventive replacement renews the unit, and age replacement
          counts its operating time since it was renewed. The exact
          methods refuse such a component; the simulations follow it, in
          Python.
          ``"standby"`` makes the node a standby group of identical units,
          each failing and repaired as ``"reliability"`` and
          ``"repairability"`` say: a dict of ``"units"`` (by default 2),
          ``"k"`` (how many must operate, by default 1, fewer than
          ``"units"``), ``"dormancy_factor"`` (how fast a spare ages, as a
          fraction of an operating unit: 0, the default, for cold standby,
          1 for hot) and ``"switching_probability"`` (that switching a spare
          in works, by default 1). The group is up while ``k`` units
          operate. When one fails, the spare that has waited longest is
          switched in; a failed switch leaves the position empty until a
          repaired unit fills it. A spare that fails in standby is found at
          once. Each failed unit is repaired on its own, a job for the
          repair crews at the group's ``"priority"``, and then fills an
          empty position or waits as a spare. ``"repair_cost"`` and
          ``"replace_cost"`` are charged at each unit's failure, and
          ``"downtime_cost"`` while the group is down. A group takes no
          ``"preventive"`` or ``"inspection"`` schedule, and no
          ``"instant"`` repair. Its long-run values are exact from its
          Markov chain when its units' lives and repair times are
          exponential; it is simulated in Python.
        - A [`NonRepairable`][repyability.NonRepairable], pairing a
          reliability model with a time-to-replace model. Each node gets
          its own copy, so one object can be given for several identical
          nodes.
        - A nested ``RepairableRBD``, which acts as one node that is up
          while its own system is up. In a simulation it runs its own
          simulation, on the same clock. It is not copied, so give each
          node its own object: nodes sharing one would share its
          simulation state, and ``availability`` then typically raises a
          ValueError. Its own costs are not counted in this RBD's costs.
        - [`PerfectReliability`][repyability.PerfectReliability] itself
          (or a spec whose ``"reliability"`` it is, with no costs or
          maintenance), for a junction: a node that never fails, such as a
          k-out-of-n vote point (#182). It is no component: the analyses
          leave it out, the simulations draw nothing for it, and it passes
          whatever reaches it, up to a ``capacity`` if it is given one.
    k : dict[Hashable, int], optional
        k-out-of-n nodes, as ``{node: k}``: the node passes only if at
        least ``k`` of the branches entering it are working (and the node
        itself is working). By default None, meaning every k is 1.
    input_node : Hashable, optional
        The input (source) node. By default None: the only node with no
        incoming edge. If given, it must be that node, or a ValueError is
        raised.
    output_node : Hashable, optional
        The output (sink) node. By default None: the only node with no
        outgoing edge. If given, it must be that node, or a ValueError is
        raised.
    on_infeasible_rbd : str, optional
        ``"raise"`` (the default), ``"warn"`` or ``"ignore"``: what to do if
        the diagram is invalid (it has a cycle, a node other than the input
        or output node with no incoming or no outgoing edge, an unusable
        ``k``, or a node with no entry in ``components``): raise a
        ValueError, warn and build the RBD anyway, or build it silently.
        The problems found are recorded in ``structure_check``.
    downtime_cost_rate : float, optional
        Cost per unit time the whole system is down (e.g. lost
        production), by default 0.0 (not priced).
    capacity : dict[Hashable, float or dict], optional
        Each node's capacity, keyed by node name, by default None: the
        throughput it passes while it is up, a positive number in any unit
        (the same for every node), or a dict ``{level: probability}`` of the
        levels it works at and the probability of each while it is up; for
        [`capacity_distribution`][repyability.RepairableRBD.capacity_distribution].
        A node that is down passes nothing. A node with no capacity given
        limits nothing (``inf``), unless its model has capacities of its
        own: a reliability that is a
        [`DegradingNode`][repyability.DegradingNode], whose stages give its
        levels, or a nested ``RepairableRBD`` with capacities, whose
        distribution it then has.
    repair_crews : int, optional
        How many repair crews work on the components, by default None: as
        many as are needed, so no work waits. At most this many jobs
        proceed at once: a job is what brings a component back up (a repair
        or replacement, preventive maintenance that takes time, or a test
        that takes time), and the component is down from when the job falls
        due. A job that finds every crew busy waits, the component down,
        until a crew is free, which then takes the waiting job of the
        highest ``"priority"``, and of those the one that fell due first;
        a crew stays with a job until it is done. Maintenance or a test in
        no time needs no crew. A nested ``RepairableRBD``'s components are
        worked on by its own crews. With fewer crews than components,
        components wait for each other: the simulations (``availability``,
        ``cost``, ``compare``) follow the queue, in Python. The exact
        long-run values then come from a Markov chain of the components'
        states and the queue when their lives and repairs are exponential
        (see ``mean_availability``); the other exact methods, which assume
        independent components, raise ``NotImplementedError``. With at
        least as many, no job waits, and every result is as without crews.
    maintenance_groups : dict, optional
        Options for the maintenance groups the components' ``"group"``
        keys form, by group name, by default None: for each, a dict of
        ``"setup_cost"`` (a number, by default 0) and ``"system_down"``
        (by default False). A member's failure, or its scheduled
        replacement, opens a *stop* of its group, at which every other
        member that is working, has an ``"opportunity"`` and is at least
        that old, is renewed too (opportunistic maintenance), taking its
        own maintenance time; a member whose own failure or replacement is
        due at that instant keeps it. With ``"system_down"``, any outage
        of the system is a stop of the group as well. The set-up cost is
        charged once per stop: once for all that starts at one instant.
        The simulations count each component's early renewals
        (``AvailabilityResult.opportunistic_renewals``) and charge the
        set-ups under ``"setup"``. Group members cannot have hidden
        failures or be standby groups. The exact methods refuse a
        component that can be renewed early; with none, a group's set-up
        is charged at each failure and each preventive replacement of a
        member in ``expected_cost_rate``. Simulated in Python.
    ccf_groups : list of CCFGroup, optional
        Common-cause groups (see [`CCFGroup`][repyability.CCFGroup]), by
        default none: identical components (the same life and repair
        models) that fail together. A group's model (a ``BetaFactor`` or
        ``MGL``) splits the members' failure rate between causes, each
        member's own and shared ones, and each cause fails the members it
        names that are up, at once (whatever the model's ``basis``: a
        repairable component's failures are a rate). Each member alone
        still fails at its rate, so its own values are as without the
        group; the system's long-run values (``mean_availability``,
        ``mean_unavailability``, ``system_failure_frequency``, MTBF, MUT,
        MDT, the cost rate and the interval choices built on them) take
        the group in exactly, from a Markov chain of which members are
        down together. The members need exponential lives, and either
        hidden failures (an ``"inspection"``, with instant tests and
        repairs and one coverage for the group; a test finds a cause's
        failures alike, and each member is found by its own tests) or
        revealed ones with exponential repairs. The importance measures
        take the groups in too, a member's conditioned on its state at
        each time; the allocations, the values over time from new and the
        simulations refuse a diagram with groups, as yet.

    Attributes
    ----------
    components : dict
        Node name -> the model the node is simulated with: a
        ``NonRepairable`` (built from the spec dict, or a copy of the one
        given) or a nested ``RepairableRBD``.
    repairability : dict
        Node name -> its time-to-repair model (an ``ExactEventTime`` at 0
        for ``"instant"``), or None for a nested ``RepairableRBD``.
    costs : dict
        Node name -> ``{cost key: number or distribution}``, for the nodes
        that declare at least one non-zero cost; a preventive-maintenance
        cost is under ``"preventive_cost"`` and an inspection cost under
        ``"inspection_cost"``.
    downtime_cost_rate : float
        The system downtime cost rate.
    capacity : dict
        The capacities given, keyed by node name, as floats.
    repair_crews : int or None
        The number of repair crews (None: as many as are needed).
    acquisition_costs : dict
        Node name -> the one-off cost of buying the unit, for the nodes that
        declare a non-zero ``"acquisition_cost"``.
    input_node : Hashable
        The input node.
    output_node : Hashable
        The output node.
    nodes : list
        The names of the nodes other than the input and output nodes.
    structure_check : dict
        The results of the structural checks made at construction.
    COST_KEYS : tuple[str, ...]
        The optional cost keys of a spec dict: ``"repair_cost"``,
        ``"replace_cost"`` and ``"downtime_cost"``.
    PER_FAILURE_COST_KEYS : tuple[str, ...]
        The cost keys charged per failure, which may be distributions:
        ``"repair_cost"`` and ``"replace_cost"``.
    COMPONENT_SPEC_KEYS : tuple[str, ...]
        Every key a spec dict may carry.
    PREVENTIVE_KEYS : tuple[str, ...]
        The keys of a ``"preventive"`` spec: ``"interval"``, ``"policy"``,
        ``"duration"`` and ``"cost"``, a ``"condition"`` policy's
        ``"threshold"`` and ``"inspection_cost"``, and an ``"age"``
        policy's ``"opportunity"``.
    GROUP_KEYS : tuple[str, ...]
        The keys of a maintenance group's options: ``"setup_cost"`` and
        ``"system_down"``.
    REPAIR_MODELS : tuple[str, ...]
        The imperfect repair models a ``"repair"`` can name:
        ``"kijima1"`` and ``"kijima2"``.
    INSPECTION_KEYS : tuple[str, ...]
        The keys of an ``"inspection"`` spec: ``"interval"``, ``"duration"``,
        ``"cost"``, ``"offset"``, ``"coverage"`` and ``"full_test"``.
    STANDBY_KEYS : tuple[str, ...]
        The keys of a ``"standby"`` spec: ``"units"``, ``"k"``,
        ``"dormancy_factor"`` and ``"switching_probability"``.

    Raises
    ------
    ValueError
        If a spec dict has an unknown key, or a ``"repairability"`` string
        other than ``"instant"``; if a cost is not a finite, non-negative
        number, a ``"downtime_cost"`` or ``downtime_cost_rate`` is a
        distribution, or a cost distribution has no finite mean or puts
        appreciable probability on a negative cost (its 1e-12 quantile is
        below 0); if a ``"preventive"`` spec is not a dict of its keys with
        a positive ``interval``, a ``policy`` of ``"age"``, ``"block"`` or
        ``"condition"`` and a ``duration`` that is a model or
        ``"instant"``; if an ``"inspection"`` spec is not a dict of its
        keys with a positive, finite ``interval``, a ``duration`` that
        is a model or ``"instant"``, an ``offset`` from 0 to less than the
        interval, a ``coverage`` from 0 to 1 and, with a coverage below 1,
        a ``full_test`` that is a whole multiple of the interval, or a
        component has both; if a
        ``"standby"`` spec is not a dict of its keys with whole numbers
        ``units`` above ``k`` of at least 1, and a ``dormancy_factor`` and
        ``switching_probability`` in [0, 1], or its component has a
        schedule or instant repair; if a reliability model is not a
        surpyval parametric or non-parametric model or a ``StandbyModel``;
        if ``input_node`` or ``output_node`` is not in the diagram, or is
        not its source or sink;
        if ``on_infeasible_rbd`` is not ``"raise"``, ``"warn"`` or
        ``"ignore"``; if the diagram is invalid and ``on_infeasible_rbd``
        is ``"raise"``; if a capacity is not a positive number or is for
        the input or output node or a node not in the diagram; if
        ``repair_crews`` is not a whole number of at least 1, or a
        ``"priority"`` is not a finite number; or if an ``"opportunity"``
        is not an age from 0 to an age policy's interval, is given to a
        component in no group, or a group's member has hidden failures or
        is a standby group, or ``maintenance_groups`` names a group no
        component is in, or has options other than a non-negative
        ``"setup_cost"`` and a boolean ``"system_down"``; or if a
        ``"repair"`` is not a dict of a known ``"model"`` and a ``"q"`` in
        [0, 1], a ``"replace_after"`` is not a whole number of at least 1
        given with a ``q`` above 0, or an imperfectly repaired component
        is a standby group, is replaced on condition, or has a lifetime
        model with no ``Hf`` and ``qf``; or if ``ccf_groups`` holds
        anything but ``CCFGroup`` instances, a member that is not a
        component (or is a nested RBD or a standby group), a component in
        two groups, or a group of components with different life or
        repair models.
    TypeError
        If a component is not a spec dict, a ``NonRepairable`` or a
        ``RepairableRBD`` (a ``Repairable``, which models imperfect repair,
        cannot be a node).
    KeyError
        If a spec dict has no ``"reliability"`` or no ``"repairability"``.

    Examples
    --------
    Two pumps in parallel, each failing on average every 10 hours and
    taking 1 hour on average to repair:

    >>> import surpyval as surv
    >>> from repyability import RepairableRBD
    >>> pump = {
    ...     "reliability": surv.Exponential.from_params([0.1]),
    ...     "repairability": surv.Exponential.from_params([1.0]),
    ... }
    >>> pumps = RepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {"a": pump, "b": pump},
    ... )
    >>> round(pumps.mean_availability(), 4)  # 1 - (1 / 11) ** 2
    0.9917
    >>> result = pumps.availability(t_simulation=50, mc_samples=200, seed=0)
    >>> window = result.n_simulations * result.time_simulated_to
    >>> round(float(result.system_uptime) / window, 4)  # simulated
    0.9925

    The pair as one node of a larger system, in series with a valve that
    is replaced instantly at a cost of 250 per failure, while lost
    production costs 1000 per hour of system downtime:

    >>> plant = RepairableRBD(
    ...     [("s", "pumps"), ("pumps", "valve"), ("valve", "t")],
    ...     {
    ...         "pumps": pumps,
    ...         "valve": {
    ...             "reliability": surv.Weibull.from_params([500.0, 2.0]),
    ...             "repairability": "instant",
    ...             "replace_cost": 250.0,
    ...         },
    ...     },
    ...     downtime_cost_rate=1000.0,
    ... )
    >>> round(plant.mean_availability(), 4)  # the valve is never down
    0.9917
    >>> round(plant.expected_cost_rate(), 2)  # cost per hour, long run
    8.83

    A ``NonRepairable`` can stand in for a spec dict, and one object can
    serve several nodes, since each node gets its own copy:

    >>> from repyability import NonRepairable
    >>> unit = NonRepairable(
    ...     surv.Exponential.from_params([0.1]),
    ...     surv.Exponential.from_params([1.0]),
    ... )
    >>> same = RepairableRBD(
    ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
    ...     {"a": unit, "b": unit},
    ... )
    >>> round(same.mean_availability(), 4)
    0.9917
    >>> same.components["a"] is same.components["b"]
    False
    """

    #: Optional per-component cost fields accepted in a component spec dict.
    COST_KEYS = ("repair_cost", "replace_cost", "downtime_cost")
    #: The costs charged per failure, which may also be given as a
    #: distribution of the cost (drawn afresh at each failure).
    PER_FAILURE_COST_KEYS = ("repair_cost", "replace_cost")
    #: Every key a component spec dict may carry.
    COMPONENT_SPEC_KEYS = (
        ("reliability", "repairability")
        + COST_KEYS
        + (
            "preventive",
            "inspection",
            "acquisition_cost",
            "priority",
            "standby",
            "group",
            "repair",
            "replace_after",
        )
    )
    #: The models of imperfect repair a spec's ``"repair"`` can name.
    REPAIR_MODELS = ("kijima1", "kijima2")
    #: The keys of a component's ``"preventive"`` spec.
    PREVENTIVE_KEYS = (
        "interval",
        "policy",
        "duration",
        "cost",
        "threshold",
        "inspection_cost",
        "opportunity",
    )
    #: The keys of a maintenance group's options.
    GROUP_KEYS = ("setup_cost", "system_down")
    #: The keys of a component's ``"inspection"`` spec.
    INSPECTION_KEYS = (
        "interval",
        "duration",
        "cost",
        "offset",
        "coverage",
        "full_test",
    )
    #: The keys of a component's ``"standby"`` spec.
    STANDBY_KEYS = ("units", "k", "dormancy_factor", "switching_probability")
    #: The costs charged per action (per failure, per preventive
    #: replacement or per inspection), which may be distributions.
    _PER_ACTION_COST_KEYS = PER_FAILURE_COST_KEYS + (
        "preventive_cost",
        "inspection_cost",
    )

    def __init__(
        self,
        edges: Iterable[tuple[Hashable, Hashable]],
        components: dict[Any, Any],
        k: Optional[dict[Any, int]] = None,
        input_node: Optional[Any] = None,
        output_node: Optional[Any] = None,
        on_infeasible_rbd: str = "raise",
        downtime_cost_rate: float = 0.0,
        capacity: Optional[dict[Any, float]] = None,
        repair_crews: Optional[int] = None,
        maintenance_groups: Optional[dict[Any, dict]] = None,
        ccf_groups: Optional[Sequence[Any]] = None,
    ):
        _check_on_infeasible_rbd(on_infeasible_rbd)
        # Capture the constructor inputs verbatim (before any mutation) so the
        # RBD can be faithfully serialised via to_dict()/to_json().
        edges = list(edges)
        self._init_args = {
            "edges": [tuple(e) for e in edges],
            "components": dict(components),
            "k": dict(k) if k else None,
            "input_node": input_node,
            "output_node": output_node,
            "on_infeasible_rbd": on_infeasible_rbd,
            "downtime_cost_rate": downtime_cost_rate,
            "capacity": dict(capacity) if capacity else None,
            "repair_crews": repair_crews,
            "maintenance_groups": (
                dict(maintenance_groups) if maintenance_groups else None
            ),
            "ccf_groups": list(ccf_groups) if ccf_groups else None,
        }
        self.repair_crews = _validate_crews(repair_crews)
        # Each component's place in the queue for a crew (higher first).
        self._priority: dict[Any, float] = {}
        self.downtime_cost_rate = self._validate_cost(
            "<system>", "downtime_cost_rate", downtime_cost_rate
        )
        # Per-node cost fields, pulled out of the component specs: a number,
        # or (per-failure costs only) a distribution. Only nodes that declare
        # at least one non-zero cost appear here.
        self.costs: dict[Any, dict[str, Any]] = {}
        # Scheduled preventive maintenance, by node (only schedules that
        # maintain: an infinite interval never does).
        self._preventive: dict[Any, _Preventive] = {}
        # Periodic inspection, by node: the nodes whose failures are hidden.
        self._inspection: dict[Any, _Inspection] = {}
        # Standby groups, by node: the nodes that are groups of units.
        self._standby: dict[Any, _Standby] = {}
        # One-off purchase costs, by node (only non-zero ones).
        self.acquisition_costs: dict[Any, float] = {}
        # Each component's maintenance group, by node (see _maintenance).
        self._member_group: dict[Any, Hashable] = {}
        # Components repaired imperfectly, by node (see _ImperfectComponent).
        self._imperfect: dict[Any, _Imperfect] = {}
        # The base class checks the names given components against the
        # edges; a component for a name in no edge is not part of the
        # diagram (the structure check reports it).
        self._models_given = list(components)
        in_edges = {node for edge in edges for node in edge}
        components = {
            name: component
            for name, component in components.items()
            if name in in_edges
        }
        # Junctions: nodes that never fail, such as a k-out-of-n vote point
        # (#182). They are no components: the structure is folded with them
        # working (see RBD._decomposition).
        self._junction_nodes = frozenset(
            name
            for name, component in components.items()
            if _is_junction(name, component)
        )
        components = {
            name: component
            for name, component in components.items()
            if name not in self._junction_nodes
        }
        reliability = {}
        repairability = {}
        for name, component in components.items():
            if isinstance(component, dict):
                self._validate_component_spec(name, component)
                node_costs = {}
                for key in self.COST_KEYS:
                    if component.get(key) is None:
                        continue
                    cost = self._validate_component_cost(
                        name, key, component[key]
                    )
                    # A cost of 0 prices nothing, so it is left out.
                    if not (isinstance(cost, float) and cost == 0.0):
                        node_costs[key] = cost
                if component.get("preventive") is not None:
                    schedule, cost, inspecting = self._validate_preventive(
                        name, component["preventive"]
                    )
                    if cost is not None:
                        node_costs["preventive_cost"] = cost
                    if np.isfinite(schedule.interval):
                        self._preventive[name] = schedule
                        if inspecting is not None:
                            node_costs["inspection_cost"] = inspecting
                if component.get("inspection") is not None:
                    if name in self._preventive:
                        raise ValueError(
                            f"Component {name!r} has both a preventive "
                            "schedule and an inspection; give it one or the "
                            "other."
                        )
                    inspection, cost = self._validate_inspection(
                        name, component["inspection"]
                    )
                    if cost is not None:
                        node_costs["inspection_cost"] = cost
                    self._inspection[name] = inspection
                if component.get("priority") is not None:
                    self._priority[name] = _validate_priority(
                        name, component["priority"]
                    )
                if component.get("standby") is not None:
                    self._standby[name] = self._validate_standby(
                        name, component
                    )
                if component.get("group") is not None:
                    self._member_group[name] = component["group"]
                imperfect = self._validate_imperfect(name, component)
                if imperfect is not None:
                    self._imperfect[name] = imperfect
                if component.get("acquisition_cost") is not None:
                    acquisition = self._validate_cost(
                        name, "acquisition_cost", component["acquisition_cost"]
                    )
                    if acquisition:
                        self.acquisition_costs[name] = acquisition
                if node_costs:
                    self.costs[name] = node_costs
                repair_model = component["repairability"]
                if isinstance(repair_model, str):
                    # "instant": repaired in zero time. The component still
                    # *fails* (failure events fire, and any repair/replace
                    # cost is charged), but each outage has zero length, so
                    # it contributes no downtime and its availability is 1.
                    # The convenient assumption when repairs are much faster
                    # than the timescale being studied, or when no
                    # repair-time data exists.
                    if repair_model != "instant":
                        raise ValueError(
                            f"Component {name!r}: unknown repairability "
                            f"{repair_model!r}. Pass a fitted time-to-repair "
                            "model, or the string 'instant' for repair in "
                            "zero time."
                        )
                    repair_model = ExactEventTime.from_params(0)
                components[name] = NonRepairable(
                    component["reliability"], repair_model
                )
                reliability[name] = component["reliability"]
                repairability[name] = repair_model
            elif isinstance(component, RepairableRBD):
                reliability[name] = component
                repairability[name] = None
            elif isinstance(component, NonRepairable):
                # A NonRepairable carries its simulation state (whether it
                # fails or is repaired next), so each node gets its own copy:
                # one object can then be given for several identical parts,
                # here or in a nested RBD.
                components[name] = copy(component)
                reliability[name] = component.reliability
                repairability[name] = component.time_to_replace
            else:
                raise TypeError(self._unknown_component(name, component))
        warn_nonparametric(
            nonparametric_nodes(
                {
                    name: spec
                    for name, spec in self._init_args["components"].items()
                    if not isinstance(spec, RepairableRBD)
                }
            )
        )

        super().__init__(
            edges,
            None,
            k,
            input_node,
            output_node,
            on_infeasible_rbd,
            capacity=capacity,
        )

        # Every intermediate graph node needs a component definition (the
        # input/output nodes do not): the base class has checked, and
        # reported the missing ones, rather than a KeyError mid-simulation.
        missing = list(self.structure_check["nodes_with_no_model"])
        self.structure_check["is_missing_components"] = bool(missing)
        self.structure_check["nodes_with_no_component"] = missing

        self.components = components
        self.repairability = copy(repairability)
        self._maintenance = self._validate_groups(maintenance_groups)
        self.ccf_groups = self._validate_ccf_groups(ccf_groups)

    #: The junctions are folded out of the structure (see ``RBD``).
    _FOLDS_JUNCTIONS = True
    _SIMULATE_INSTEAD = (
        "availability, cost and simulate_timelines simulate it (in Python)."
    )

    def _junctions(self) -> frozenset:
        """The junctions: nodes given ``PerfectReliability``, which never
        fail (see ``RBD._junctions``)."""
        return getattr(self, "_junction_nodes", frozenset())

    def _repr_details(self) -> List[str]:
        """What shapes the repairable diagram, for ``repr``: its
        maintenance, tests, standby groups, nested RBDs and repair crews."""
        counts = [
            (len(getattr(self, "_preventive", {})), "maintained"),
            (len(getattr(self, "_inspection", {})), "tested"),
            (len(getattr(self, "_standby", {})), "standby group(s)"),
            (len(getattr(self, "_imperfect", {})), "repaired imperfectly"),
            (
                sum(
                    isinstance(c, RepairableRBD)
                    for c in getattr(self, "components", {}).values()
                ),
                "nested RBD(s)",
            ),
        ]
        counts.append(
            (len(getattr(self, "ccf_groups", ())), "common-cause group(s)")
        )
        out = [f"{count} {what}" for count, what in counts if count]
        if getattr(self, "repair_crews", None) is not None:
            out.append(f"{self.repair_crews} repair crew(s)")
        junctions = self._junctions()
        if junctions:
            out.append(
                "junction(s) "
                + ", ".join(repr(n) for n in sorted(junctions, key=str))
            )
        return out

    def _validate_ccf_groups(self, ccf_groups) -> list:
        """The common-cause groups, checked: each a ``CCFGroup`` of
        components of this RBD (not a nested RBD or a standby group), each
        component in one group at most, the members of a group identical
        (the same life and repair models, compared through their saved
        form)."""
        if not ccf_groups:
            return []
        from repyability.rbd.ccf import CCFGroup
        from repyability.rbd.serialisation import serialise_model

        seen: set = set()
        for group in ccf_groups:
            if not isinstance(group, CCFGroup):
                raise ValueError(
                    "ccf_groups must contain CCFGroup instances, got "
                    f"{type(group).__name__}."
                )
            for member in group.members:
                if member not in self.components:
                    raise ValueError(
                        f"CCF group member {member!r} is not a component of "
                        "the RBD."
                    )
                if (
                    isinstance(self.components[member], RepairableRBD)
                    or member in self._standby
                ):
                    raise ValueError(
                        f"CCF group member {member!r} is a nested RBD or a "
                        "standby group: a group's members are single "
                        "components."
                    )
                if member in seen:
                    raise ValueError(
                        f"Node {member!r} appears in more than one CCF group."
                    )
                seen.add(member)
            try:
                specs = [
                    (
                        serialise_model(self.components[m].reliability),
                        serialise_model(self.components[m].time_to_replace),
                    )
                    for m in group.members
                ]
            except Exception:  # models that cannot be saved are not compared
                continue
            if any(spec != specs[0] for spec in specs[1:]):
                raise ValueError(
                    f"The members of a CCF group, {list(group.members)}, "
                    "must be identical components, with the same life and "
                    "repair models."
                )
        return list(ccf_groups)

    def _has_ccf(self) -> bool:
        """Whether this RBD, or one nested in it, has common-cause
        groups."""
        return bool(getattr(self, "ccf_groups", ())) or any(
            isinstance(c, RepairableRBD) and c._has_ccf()
            for c in self.components.values()
        )

    def _require_no_ccf(self, what: str, nested: bool = False) -> None:
        """Raise if this RBD (or, with ``nested``, one nested in it) has
        common-cause groups, which ``what`` does not take in."""
        if self.ccf_groups or (nested and self._has_ccf()):
            raise NotImplementedError(
                f"The RBD has common-cause groups, whose members fail "
                f"together, which {what} does not take in, as yet: their "
                "long-run values (mean_availability, "
                "system_failure_frequency and those built on them) do."
            )

    def _ccf_rates(self, group) -> Tuple[float, Optional[float]]:
        """A common-cause group's members' failure rate, and their repair
        rate (None for hidden failures, found by tests), for its chain
        (see ``_ccf_chain``); raise if the chain does not cover them."""
        where = f"Common-cause group {list(group.members)}"
        if self._crews_couple():
            raise NotImplementedError(
                f"{where}: with limited repair crews, the crews' Markov "
                "chain does not take common causes in, as yet."
            )
        rates = set()
        for member in group.members:
            for kinds, what in (
                (self._preventive, "scheduled maintenance"),
                (self._imperfect, "imperfect repair"),
                (self._member_group, "a maintenance group"),
            ):
                if member in kinds:
                    raise NotImplementedError(
                        f"{where}: member {member!r} has {what}, which the "
                        "group's chain does not take in."
                    )
            life = _constant_rate(self.components[member].reliability)
            if life is None:
                raise NotImplementedError(
                    f"{where}: its chain needs exponential lives (a constant "
                    f"failure rate), and member {member!r}'s is not."
                )
            rates.add(life)
        tested = [member in self._inspection for member in group.members]
        if any(tested) and not all(tested):
            raise NotImplementedError(
                f"{where}: some members' failures are hidden and others' "
                "revealed; the group's chain takes one kind."
            )
        life = rates.pop()
        if all(tested):
            for member in group.members:
                self._inspected_rate(member)
            if len({self._inspection[m].coverage for m in group.members}) > 1:
                raise NotImplementedError(
                    f"{where}: its members' tests have different coverages; "
                    "the group's chain takes one, which a cause's failures "
                    "are found by alike."
                )
            return life, None
        component = self.components[group.members[0]]
        repair = _constant_rate(component.time_to_replace)
        if repair is None:
            raise NotImplementedError(
                f"{where}: its chain needs exponential repairs, and its "
                "members' are not."
            )
        return life, repair

    def _require_ccf_long_run(self) -> None:
        """Raise if a common-cause group's chain does not cover it."""
        for group in self.ccf_groups:
            self._ccf_rates(group)

    def _calendar_period(self) -> float:
        """The period of the schedules the long-run values average over
        (see ``_long_run_grid``): of the tests and block replacements."""
        intervals = {
            self._preventive[node].interval for node in self._block_nodes()
        }
        for schedule in self._inspection.values():
            intervals |= {schedule.interval, schedule.period}
        return _common_period(intervals) if intervals else 1.0

    def _group_states(self, group, times: np.ndarray):
        """A common-cause group's members' joint states, at each of
        ``times`` (of ``_long_run_grid``), in the long run (see
        ``_ccf_chain``)."""
        life, repair = self._ccf_rates(group)
        if repair is not None:
            states = _ccf_chain.revealed(
                group.model, group.members, life, repair
            )
            return states._replace(
                probabilities=np.repeat(
                    states.probabilities, len(times), axis=0
                )
            )
        period = self._calendar_period()
        tests = []
        for position, member in enumerate(group.members):
            schedule = self._inspection[member]
            count = int(round(period / schedule.interval))
            first = 0 if schedule.offset else 1
            for k in range(first, first + count):
                time = schedule.offset + k * schedule.interval
                tests.append(
                    _ccf_chain.ProofTest(
                        time, position, schedule.is_full(time)
                    )
                )
        coverage = self._inspection[group.members[0]].coverage
        return _ccf_chain.hidden(
            group.model, group.members, life, coverage, tests, period, times
        )

    def _require_free_members(self, working_nodes, broken_nodes) -> None:
        """Raise if a common-cause group's member is held working or
        broken: the cause it shares would still strike the others."""
        held = set(working_nodes or ()) | set(broken_nodes or ())
        for group in self.ccf_groups:
            caught = held & set(group.members)
            if caught:
                raise NotImplementedError(
                    f"Node(s) {sorted(caught, key=str)} are in a "
                    "common-cause group, whose shared causes would still "
                    "strike the others: they cannot be held working or "
                    "broken."
                )

    def _with_ccf_groups(
        self,
        times: np.ndarray,
        probabilities: dict,
        failures: Optional[dict],
        weights: np.ndarray,
    ) -> Tuple[dict, Optional[dict], np.ndarray, np.ndarray]:
        """The long-run points (the times of ``_long_run_grid``, and their
        weights) split by the common-cause groups' joint states: each
        time into one point for each combination of each group's members
        up or down, weighted by its probability then, with those members
        up or down for certain and the other nodes as at the time. The
        long-run values are averages over these points as over the times.
        Also each point's time's position in ``times``."""
        index = np.arange(len(times))
        for group in self.ccf_groups:
            states = self._group_states(group, times)
            combinations = len(states.down)
            mass = (weights[:, None] * states.probabilities[index, :]).ravel()
            keep = mass > 0.0
            weights = mass[keep]

            def spread(values, combinations=combinations, keep=keep):
                points = len(keep) // combinations
                values = np.broadcast_to(
                    np.asarray(values, dtype=float), (points,)
                )
                return np.repeat(values, combinations)[keep]

            probabilities = {n: spread(v) for n, v in probabilities.items()}
            if failures is not None:
                failures = {n: spread(v) for n, v in failures.items()}
            for k, member in enumerate(states.members):
                down = np.tile(states.down[:, k], len(index))[keep]
                probabilities[member] = np.where(down, 0.0, 1.0)
                if failures is not None:
                    failures[member] = np.where(down, 1.0, 0.0)
            index = np.repeat(index, combinations)[keep]
        return probabilities, failures, weights, index

    def _require_ccf_frequencies(self) -> None:
        """Raise if the failure frequency with common-cause groups is not
        worked out: with block replacements that take time (planned
        outages at block times, which the groups' states would change)."""
        if not self.ccf_groups:
            return
        self._require_ccf_long_run()
        timed = [
            node
            for node in self._block_nodes()
            if self._preventive[node].duration is not None
        ]
        if timed:
            raise NotImplementedError(
                "The system's planned outages at block replacements that "
                f"take time (of {sorted(timed, key=str)}) are not worked "
                "out with common-cause groups, as yet."
            )

    def _ccf_outage_frequencies(
        self, working_nodes, broken_nodes
    ) -> Tuple[float, float]:
        """``_outage_frequencies`` with common-cause groups, over the points
        of ``_with_ccf_groups``: a node outside the groups fails at its
        rate and takes the system down where it is critical (its Birnbaum
        importance at the point); each cause strikes at its rate and takes
        the system down by failing the members it names that are up (the
        rise in the system's unavailability with them down)."""
        self._require_ccf_frequencies()
        self._require_free_members(working_nodes, broken_nodes)
        times, weights = self._long_run_grid()
        availability = self._probabilities_with_overrides(
            self._availabilities_at(times), working_nodes, broken_nodes
        )
        unavailability = self._failures_with_overrides(
            self._unavailabilities_at(times), working_nodes, broken_nodes
        )
        availability, grouped, weights, index = self._with_ccf_groups(
            times, availability, unavailability, weights
        )
        assert grouped is not None
        unavailability = grouped
        forced = set(working_nodes or ()) | set(broken_nodes or ())
        members = {m for group in self.ccf_groups for m in group.members}
        birnbaum = super()._birnbaum_importance(
            availability, node_failures=unavailability
        )
        failures = planned = 0.0
        blocks = set(self._block_nodes())
        for node in self.components:
            if node in forced or node in members:
                continue
            importance = np.asarray(birnbaum[node])
            if node in self._inspection:
                node_failures: Any = self._tested_intensity(
                    node, times[index], availability[node]
                )
                node_planned: Any = 0.0
            elif node in blocks:
                node_failures = self._block_profile(
                    node, times[index], rates=True
                )
                node_planned = 0.0
            else:
                node_failures, _, node_planned = self._node_frequencies(node)
            failures += float(weights @ (importance * node_failures))
            planned += float(weights @ (importance * node_planned))
        base = self._system_unreliability(availability, unavailability)
        for group in self.ccf_groups:
            life, _ = self._ccf_rates(group)
            for struck, rate in _ccf_chain.causes(
                group.model, group.members, life
            ):
                up, down = dict(availability), dict(unavailability)
                for position in struck:
                    member = group.members[position]
                    up[member] = np.zeros_like(up[member])
                    down[member] = np.ones_like(down[member])
                rise = self._system_unreliability(up, down) - base
                failures += rate * float(weights @ rise)
        return failures, planned

    def _validate_imperfect(self, node, spec: dict) -> Optional[_Imperfect]:
        """A component spec's imperfect repair (``"repair"`` and
        ``"replace_after"``), validated; None if it is renewed by every
        repair, as by default (a restoration factor of 0)."""
        repair, limit = spec.get("repair"), spec.get("replace_after")
        if repair is None and limit is None:
            return None
        q, model = 0.0, ""
        if repair is not None:
            if (
                not isinstance(repair, dict)
                or set(repair) != {"model", "q"}
                or repair["model"] not in self.REPAIR_MODELS
            ):
                raise ValueError(
                    f"Component {node!r}: repair must be a dict of a "
                    f"'model' ({' or '.join(map(repr, self.REPAIR_MODELS))}) "
                    f"and its restoration factor 'q', got {repair!r}."
                )
            q = repair["q"]
            if (
                isinstance(q, bool)
                or not isinstance(q, (int, float, np.number))
                or not 0.0 <= float(q) <= 1.0
            ):
                raise ValueError(
                    f"Component {node!r}: the restoration factor q must be "
                    "a number in [0, 1] (0 renews the unit, 1 is minimal "
                    f"repair), got {q!r}."
                )
            q, model = float(q), repair["model"]
        if limit is not None:
            if (
                isinstance(limit, bool)
                or not isinstance(limit, (int, np.integer))
                or limit < 1
            ):
                raise ValueError(
                    f"Component {node!r}: replace_after must be a whole "
                    f"number of failures of at least 1, got {limit!r}."
                )
            if q == 0.0:
                raise ValueError(
                    f"Component {node!r}: replace_after needs imperfect "
                    "repair (a 'repair' with q above 0): renewed by every "
                    "repair, the unit is as new after each failure anyway."
                )
            limit = int(limit)
        if q == 0.0:
            return None
        if spec.get("standby") is not None:
            raise ValueError(
                f"Component {node!r} is a standby group, whose units are "
                "renewed by their repairs, so it cannot be repaired "
                "imperfectly."
            )
        preventive = spec.get("preventive")
        if isinstance(preventive, dict) and preventive.get("policy") == (
            "condition"
        ):
            raise ValueError(
                f"Component {node!r} is replaced on condition, judged by its "
                "age as new, so it cannot be repaired imperfectly."
            )
        life = spec["reliability"]
        if not (hasattr(life, "Hf") and hasattr(life, "qf")):
            raise ValueError(
                f"Component {node!r}: an imperfectly repaired unit's lives "
                "are drawn given its virtual age, which needs a lifetime "
                "model with a cumulative hazard (Hf) and quantiles (qf), "
                "such as a surpyval distribution."
            )
        return _Imperfect(model, q, limit)

    def _validate_groups(
        self, options: Optional[dict]
    ) -> Dict[Hashable, _MaintenanceGroup]:
        """The maintenance groups (#108), from the components' ``"group"``
        and the options given for each, checked; and the members that can
        be renewed early (``_early_members``)."""
        members: Dict[Hashable, list] = {}
        for node, group in self._member_group.items():
            try:
                hash(group)
            except TypeError:
                raise ValueError(
                    f"Component {node!r}: a group is named by a hashable "
                    f"value, got {group!r}."
                ) from None
            for spec_key, what in (
                ("_inspection", "hidden failures"),
                ("_standby", "a standby group"),
            ):
                if node in getattr(self, spec_key):
                    raise ValueError(
                        f"Component {node!r} has {what}, so it cannot be in "
                        "a maintenance group."
                    )
            members.setdefault(group, []).append(node)
        self._early_members = {
            node
            for node, schedule in self._preventive.items()
            if schedule.opportunity < schedule.interval
        }
        for node in self._early_members:
            if node not in self._member_group:
                raise ValueError(
                    f"Component {node!r} has an opportunity, but no "
                    "maintenance group to give it one: add a 'group'."
                )
        options = dict(options or {})
        unknown = set(options) - set(members)
        if unknown:
            raise ValueError(
                f"maintenance_groups names group(s) "
                f"{sorted(map(str, unknown))} that no component is in."
            )
        out: Dict[Hashable, _MaintenanceGroup] = {}
        for group, nodes in members.items():
            given = options.get(group) or {}
            if not isinstance(given, dict) or set(given) - set(
                self.GROUP_KEYS
            ):
                raise ValueError(
                    f"Maintenance group {group!r}: its options must be a "
                    f"dict of {', '.join(self.GROUP_KEYS)}, got {given!r}."
                )
            setup = given.get("setup_cost")
            setup = (
                0.0
                if setup is None
                else self._validate_cost(group, "setup_cost", setup)
            )
            system_down = given.get("system_down", False)
            if not isinstance(system_down, bool):
                raise ValueError(
                    f"Maintenance group {group!r}: system_down must be True "
                    f"or False, got {system_down!r}."
                )
            out[group] = _MaintenanceGroup(tuple(nodes), setup, system_down)
        return out

    @staticmethod
    def _unknown_component(node, component) -> str:
        """Why ``component`` cannot be node ``node``: it is none of the
        kinds a node can be."""
        from repyability.repairable import Repairable

        kinds = (
            "Give a spec dict with 'reliability' and 'repairability', a "
            "NonRepairable, or a nested RepairableRBD (or, for a junction "
            "that never fails, such as a k-out-of-n vote point, "
            "PerfectReliability)."
        )
        if isinstance(component, Repairable):
            return (
                f"Component {node!r} is a Repairable, which models the "
                "minimal or imperfect repair of a single unit: it cannot be "
                "a node, whose repairs renew it as new. " + kinds
            )
        return f"Component {node!r} is a {type(component).__name__}. " + kinds

    @classmethod
    def _validate_component_spec(cls, node, spec: dict) -> None:
        """Reject unknown keys in a component spec.

        A mistyped cost key (``repair_costs``) would otherwise be silently
        ignored and priced at zero, which is a quiet way to get the money
        wrong; surface it at construction instead.
        """
        unknown = set(spec) - set(cls.COMPONENT_SPEC_KEYS)
        if unknown:
            raise ValueError(
                f"Component {node!r} has unknown key(s) "
                f"{sorted(map(str, unknown))}. A component spec takes "
                f"{', '.join(cls.COMPONENT_SPEC_KEYS)}."
            )
        for key, what in (
            ("reliability", "its lives (time to failure)"),
            ("repairability", "its repair times"),
        ):
            if key not in spec:
                raise ValueError(
                    f"Component {node!r} needs a {key!r}: the "
                    f"distribution of {what}."
                )
            value = spec[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mean = "MTTF" if key == "reliability" else "MTTR"
                instant = (
                    ", or 'instant' for a repair in zero time"
                    if key == "repairability"
                    else ""
                )
                raise TypeError(
                    f"Component {node!r}: {key} is the number {value!r}, "
                    f"not a distribution of {what}. For an {mean} of "
                    f"{value:g}, give e.g. "
                    f"surpyval.Exponential.from_params([1 / {value:g}]); "
                    "for a fixed time, "
                    f"surpyval.ExactEventTime.from_params([{value:g}])"
                    f"{instant}."
                )

    @classmethod
    def _validate_preventive(cls, node, spec) -> Tuple[_Preventive, Any, Any]:
        """A component's ``"preventive"`` spec, validated: its schedule,
        and the cost of each replacement and of each inspection (None if
        it prices nothing)."""
        if not isinstance(spec, dict):
            raise ValueError(
                f"Component {node!r}: preventive must be a dict with "
                f"{', '.join(cls.PREVENTIVE_KEYS)}, got {spec!r}."
            )
        unknown = set(spec) - set(cls.PREVENTIVE_KEYS)
        if unknown:
            raise ValueError(
                f"Component {node!r} has unknown preventive key(s) "
                f"{sorted(map(str, unknown))}. A preventive spec takes "
                f"{', '.join(cls.PREVENTIVE_KEYS)}."
            )
        try:
            interval = float(spec["interval"])
        except KeyError:
            raise ValueError(
                f"Component {node!r}: a preventive spec needs an interval."
            ) from None
        except (TypeError, ValueError):
            interval = float("nan")
        if not interval > 0.0:
            raise ValueError(
                f"Component {node!r}: the preventive interval must be a "
                f"positive number (inf for none), got {spec['interval']!r}."
            )
        policy = spec.get("policy", "age")
        if policy not in ("age", "block", "condition"):
            raise ValueError(
                f"Component {node!r}: the preventive policy must be 'age', "
                f"'block' or 'condition', got {policy!r}."
            )
        threshold = 0.0
        if policy == "condition":
            if spec.get("threshold") is None:
                raise ValueError(
                    f"Component {node!r}: the 'condition' policy needs a "
                    "threshold: the probability of failing before the next "
                    "inspection above which the unit is replaced."
                )
            threshold = spec["threshold"]
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float, np.number))
                or not 0.0 <= float(threshold) <= 1.0
            ):
                raise ValueError(
                    f"Component {node!r}: the threshold must be a "
                    f"probability, in [0, 1], got {threshold!r}."
                )
            threshold = float(threshold)
        else:
            for key in ("threshold", "inspection_cost"):
                if spec.get(key) is not None:
                    raise ValueError(
                        f"Component {node!r}: {key} applies only to the "
                        "'condition' policy, which inspects the unit."
                    )
        opportunity = math.inf
        if spec.get("opportunity") is not None:
            if policy != "age":
                raise ValueError(
                    f"Component {node!r}: an opportunity applies only to "
                    "the 'age' policy (a unit renewed early at a stop of "
                    "its maintenance group)."
                )
            opportunity = spec["opportunity"]
            if (
                isinstance(opportunity, bool)
                or not isinstance(opportunity, (int, float, np.number))
                or not 0.0 <= float(opportunity) <= interval
            ):
                raise ValueError(
                    f"Component {node!r}: the opportunity must be an age "
                    f"from 0 to the interval ({interval:g}), got "
                    f"{opportunity!r}."
                )
            opportunity = float(opportunity)
        duration = spec.get("duration", "instant")
        if isinstance(duration, str) and duration == "instant":
            duration = None
        elif not hasattr(duration, "random"):
            raise ValueError(
                f"Component {node!r}: the preventive duration must be a "
                "time-to-maintain model (such as a fitted surpyval "
                f"distribution) or 'instant', got {duration!r}."
            )
        costs = []
        for key, name in (
            ("cost", "preventive_cost"),
            ("inspection_cost", "inspection_cost"),
        ):
            cost = spec.get(key)
            if cost is not None:
                cost = cls._validate_component_cost(node, name, cost)
                if isinstance(cost, float) and cost == 0.0:
                    cost = None
            costs.append(cost)
        schedule = _Preventive(
            interval, policy, duration, threshold, opportunity
        )
        return schedule, costs[0], costs[1]

    @classmethod
    def _validate_standby(cls, node, spec: dict) -> _Standby:
        """A component spec's ``"standby"``, validated: the group of
        units it makes the component (see ``_StandbyGroup``)."""
        standby = spec["standby"]
        if not isinstance(standby, dict):
            raise ValueError(
                f"Component {node!r}: standby must be a dict of "
                f"{', '.join(cls.STANDBY_KEYS)}, got {standby!r}."
            )
        unknown = set(standby) - set(cls.STANDBY_KEYS)
        if unknown:
            raise ValueError(
                f"Component {node!r}: unknown standby key(s) "
                f"{sorted(map(str, unknown))}; it takes "
                f"{', '.join(cls.STANDBY_KEYS)}."
            )
        for key in ("preventive", "inspection"):
            if spec.get(key) is not None:
                raise ValueError(
                    f"Component {node!r} is a standby group, which takes no "
                    f"{key!r} schedule."
                )
        if isinstance(spec["repairability"], str):
            raise ValueError(
                f"Component {node!r} is a standby group, whose units need a "
                "repair time: repaired instantly, a spare would never be "
                "needed."
            )
        counts = {"units": standby.get("units", 2), "k": standby.get("k", 1)}
        for key, value in counts.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or int(value) != value
                or value < 1
            ):
                raise ValueError(
                    f"Component {node!r}: standby {key} must be a whole "
                    f"number, 1 or more; got {value!r}."
                )
        units, k = int(counts["units"]), int(counts["k"])
        if k >= units:
            raise ValueError(
                f"Component {node!r}: a standby group needs a spare, so k "
                f"must be less than units; got k={k} and units={units}."
            )
        fractions = {
            "dormancy_factor": standby.get("dormancy_factor", 0.0),
            "switching_probability": standby.get("switching_probability", 1.0),
        }
        for key, value in fractions.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(
                    f"Component {node!r}: standby {key} must be a number in "
                    f"[0, 1]; got {value!r}."
                )
        return _Standby(
            units,
            k,
            float(fractions["dormancy_factor"]),
            float(fractions["switching_probability"]),
        )

    @classmethod
    def _validate_inspection(cls, node, spec) -> Tuple[_Inspection, Any]:
        """A component's ``"inspection"`` spec, validated: its schedule, and
        its cost (None if it prices nothing)."""
        if not isinstance(spec, dict):
            raise ValueError(
                f"Component {node!r}: inspection must be a dict with "
                f"{', '.join(cls.INSPECTION_KEYS)}, got {spec!r}."
            )
        unknown = set(spec) - set(cls.INSPECTION_KEYS)
        if unknown:
            raise ValueError(
                f"Component {node!r} has unknown inspection key(s) "
                f"{sorted(map(str, unknown))}. An inspection spec takes "
                f"{', '.join(cls.INSPECTION_KEYS)}."
            )
        try:
            interval = float(spec["interval"])
        except KeyError:
            raise ValueError(
                f"Component {node!r}: an inspection spec needs an interval."
            ) from None
        except (TypeError, ValueError):
            interval = float("nan")
        if not (interval > 0.0 and np.isfinite(interval)):
            raise ValueError(
                f"Component {node!r}: the inspection interval must be a "
                f"positive, finite number, got {spec['interval']!r}. (A "
                "hidden failure is found only by an inspection.)"
            )
        duration = spec.get("duration", "instant")
        if isinstance(duration, str) and duration == "instant":
            duration = None
        elif not hasattr(duration, "random"):
            raise ValueError(
                f"Component {node!r}: the inspection duration must be a "
                "time-to-test model (such as a fitted surpyval distribution) "
                f"or 'instant', got {duration!r}."
            )
        cost = spec.get("cost")
        if cost is not None:
            cost = cls._validate_component_cost(node, "inspection_cost", cost)
            if isinstance(cost, float) and cost == 0.0:
                cost = None
        offset = _number_or_nan(spec.get("offset", 0.0))
        if not 0.0 <= offset < interval:
            raise ValueError(
                f"Component {node!r}: the inspection offset, the time of its "
                "first test, must be at least 0 and less than its interval, "
                f"{interval:g}, got {spec['offset']!r}."
            )
        coverage = _number_or_nan(spec.get("coverage", 1.0))
        if not 0.0 <= coverage <= 1.0:
            raise ValueError(
                f"Component {node!r}: the inspection coverage, the chance "
                "that a test finds a failure, must be from 0 to 1, got "
                f"{spec['coverage']!r}."
            )
        full_test = spec.get("full_test")
        if full_test is not None:
            given = full_test
            full_test = _number_or_nan(full_test)
            count = full_test / interval
            if not (
                np.isfinite(count)
                and round(count) >= 1
                and abs(count - round(count)) <= 1e-9 * count
            ):
                raise ValueError(
                    f"Component {node!r}: the full_test interval must be a "
                    f"whole multiple of its inspection interval, {interval:g} "
                    f"(every so many tests is a full one), got {given!r}."
                )
            full_test = round(count) * interval
        elif coverage < 1.0:
            raise ValueError(
                f"Component {node!r}: with a coverage below 1, the failures "
                "its tests miss are found only by a full test: give its "
                "full_test interval, a whole multiple of its inspection "
                "interval (the mission time, say, after which it is "
                "renewed)."
            )
        schedule = _Inspection(interval, duration, offset, coverage, full_test)
        return schedule, cost

    @classmethod
    def _validate_cost(cls, node, key: str, value) -> float:
        """Coerce a cost to a finite, non-negative float."""
        if hasattr(value, "qf"):
            raise ValueError(
                f"{node!r}: {key} must be a number, not a distribution. Only "
                "the costs charged per action (repair_cost, replace_cost and "
                "the preventive and inspection costs) may be distributions."
            )
        try:
            cost = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{node!r}: {key} must be a number, got {value!r}."
            ) from None
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError(
                f"{node!r}: {key} must be finite and non-negative, got "
                f"{value!r}."
            )
        return cost

    @classmethod
    def _validate_component_cost(cls, node, key: str, value):
        """Validate a per-component cost: a number (coerced to float) or, for
        the per-action costs, a distribution of the cost -- anything with
        ``qf`` and ``mean``, such as a fitted surpyval model.

        A distribution must have a finite mean and put no more than 1e-12
        probability on a negative cost.
        """
        if key not in cls._PER_ACTION_COST_KEYS or not hasattr(value, "qf"):
            return cls._validate_cost(node, key, value)
        mean = _mean_cost(value)
        if not np.isfinite(mean):
            raise ValueError(
                f"{node!r}: the {key} distribution must have a finite mean, "
                f"got {mean:g}."
            )
        lowest = float(np.ravel(value.qf(np.array([1e-12])))[0])
        if not lowest >= 0.0:
            raise ValueError(
                f"{node!r}: the {key} distribution puts appreciable "
                f"probability on a negative cost (its 1e-12 quantile is "
                f"{lowest:g}). Use one on [0, inf), such as a LogNormal, "
                "Gamma or Weibull."
            )
        return value

    @property
    def has_costs(self) -> bool:
        """Whether any cost has been declared (per-component or system-wide).

        True if some component declares a non-zero ``"repair_cost"``,
        ``"replace_cost"``, ``"downtime_cost"``, or preventive-maintenance
        or inspection ``"cost"`` (a cost distribution always counts), a
        maintenance group has a set-up cost, or ``downtime_cost_rate`` is
        non-zero: whether running the system costs anything. Costs of 0
        price nothing, costs declared inside a nested ``RepairableRBD`` do
        not count, and neither does an
        ``"acquisition_cost"`` (a one-off cost, see ``total_cost``). When
        nothing is priced there is no cost model to evaluate, so the cost
        methods short-circuit rather than doing the work:
        ``expected_cost_rate`` returns 0.0, ``cost`` returns None, and
        ``availability`` skips the cost accounting (its result's ``cost`` is
        None).

        Returns
        -------
        bool
            True if anything is priced.
        """
        return (
            bool(self.costs)
            or bool(self.downtime_cost_rate)
            or any(
                group.setup_cost
                for group in getattr(self, "_maintenance", {}).values()
            )
        )

    def _stream_specs(
        self,
        t_simulation: float,
        prefix: tuple = (),
        specs: Optional[dict] = None,
        states: Optional[dict] = None,
    ) -> Tuple[dict, bool]:
        """The streams this RBD's simulations draw from (see ``_streams``),
        by name, nested RBDs' included; and whether every draw comes from
        one. A component whose draws cannot be streamed (a model other than
        a surpyval parametric one, or a subclass of the component classes)
        has none, nor has a maintenance or test time that cannot: they draw
        from numpy's global RNG. ``prefix`` is this RBD's place in the RBD
        simulated: only that RBD's own costs are drawn. A component started
        from a state in ``states`` (see ``_simulation_states``) that draws
        what is left of its life or repair has a stream for that too."""
        specs = {} if specs is None else specs
        states = {} if states is None else states
        complete = True

        def add(path: tuple, kind: int, sampler, expected: float) -> None:
            rows = _streams.first_rows(expected)
            specs[(path, kind)] = _streams.Spec(
                path, kind, sampler, rows, _streams.block_width(rows)
            )

        for name, component in self.components.items():
            path = prefix + (name,)
            if type(component) is RepairableRBD:
                nested = component._stream_specs(
                    t_simulation, path, specs, states.get(name)
                )
                complete = complete and nested[1]
                continue
            failure = repair = None
            if type(component) is NonRepairable:
                failure = inverse_sampler(component.reliability)
                repair = inverse_sampler(component.time_to_replace)
            if failure is None or repair is None:
                complete = False
                continue
            arrangement = self._standby.get(name)
            if arrangement is not None:
                # Each unit's lives and repairs, and the group's switches.
                lives, switches = self._standby_draw_counts(name, t_simulation)
                for unit in range(arrangement.units):
                    add(path + (unit,), _streams.FAILURE, failure, lives)
                    add(path + (unit,), _streams.REPAIR, repair, lives)
                if 0.0 < arrangement.switching_probability < 1.0:
                    add(path, _streams.SWITCH, _uniforms, switches)
                continue
            expected = self._expected_draws(name, t_simulation)
            add(path, _streams.FAILURE, failure, expected["failure"])
            add(path, _streams.REPAIR, repair, expected["repair"])
            if _draws_start(states.get(name)):
                add(path, _streams.START, _uniforms, 1.0)
            if name in self._imperfect:
                # A life after each repair, given the unit's virtual age.
                add(path, _streams.AGED, _uniforms, expected["repair"])
            inspection = self._inspection.get(name)
            if inspection is not None and inspection.partial:
                # Whether a test finds each failure: one for each failure
                # a test can miss.
                add(path, _streams.TEST, _uniforms, expected["failure"])
            duration = self._duration_model(name)
            if duration is not None:
                sampler = inverse_sampler(duration)
                if sampler is None:
                    complete = False
                else:
                    add(path, _streams.DURATION, sampler, expected["action"])
        if not prefix:
            for node, node_costs in self.costs.items():
                expected = self._expected_draws(node, t_simulation)
                for key, kind in _streams.COST_KINDS.items():
                    cost = node_costs.get(key)
                    if cost is None or isinstance(cost, float):
                        continue
                    count = expected[
                        "action" if key in _ACTION_COST_KEYS else "repair"
                    ]
                    add((node,), kind, _streams.cost_sampler(cost), count)
        return specs, complete

    def _duration_model(self, node):
        """The model of ``node``'s maintenance or test time, or None."""
        schedule = self._preventive.get(node) or self._inspection.get(node)
        return None if schedule is None else schedule.duration

    def _standby_draw_counts(
        self, node, t_simulation: float
    ) -> Tuple[float, float]:
        """Roughly how many lives (and repairs) a simulation draws for each
        unit of a standby group, and how many switches: its operating units
        fail at their rate and its spares at the dormant one (see
        ``_expected_draws``)."""
        arrangement = self._standby[node]
        life = _safe_mean(self.components[node].reliability)
        if not life > 0.0:
            return float("nan"), float("nan")
        active = (
            arrangement.k
            + (arrangement.units - arrangement.k) * arrangement.dormancy_factor
        )
        failures = t_simulation * active / life
        return failures / arrangement.units + 1.0, failures + 1.0

    def _expected_draws(self, node, t_simulation: float) -> Dict[str, float]:
        """Roughly how many times a simulation draws ``node``'s time to
        failure (``"failure"``: one per unit put into service), its repair
        time and per-failure costs (``"repair"``), and its maintenance or
        test time and their costs (``"action"``). NaN where it is not known.
        Only how the draws are computed depends on these (see
        ``_streams``), never what they are."""
        component = self.components[node]
        if node in self._standby:
            # Its units' failures, each a repair charged for.
            _, failures = self._standby_draw_counts(node, t_simulation)
            return {"failure": failures, "repair": failures, "action": 0.0}
        life = repair = float("nan")
        if isinstance(component, NonRepairable):
            life = _safe_mean(component.reliability)
            repair = _safe_mean(component.time_to_replace)
        if life == np.inf:
            failures = 0.0
        elif life + repair > 0.0:
            failures = t_simulation / (life + repair)
        else:
            failures = float("nan")
        # A member renewed early at its group's stops (#108) has more lives
        # and actions than these: its streams then take more chunks, which
        # changes only how its draws are computed.
        lives, actions = failures + 1.0, 0.0
        schedule = self._preventive.get(node)
        inspection = self._inspection.get(node)
        if schedule is not None:
            actions = t_simulation / schedule.interval + 1.0
            if schedule.policy == "age":
                lives = t_simulation / np.fmin(life, schedule.interval) + 1.0
            else:
                lives = failures + actions
        elif inspection is not None:
            actions = t_simulation / inspection.interval + 1.0
        return {
            "failure": float(lives),
            "repair": failures + 1.0,
            "action": actions,
        }

    def _stream_plan(
        self,
        t_simulation: float,
        entropy,
        antithetic: bool,
        widths: Optional[dict] = None,
        states: Optional[dict] = None,
    ) -> Tuple[_streams.Plan, bool]:
        """A run's streams (``widths`` overriding some of their widths, as
        ``compare`` does to line up two systems' streams), and whether every
        draw comes from one; the components start from ``states``."""
        specs, complete = self._stream_specs(t_simulation, states=states)
        if widths:
            specs = {
                name: (
                    dataclasses.replace(spec, width=widths[name])
                    if name in widths
                    else spec
                )
                for name, spec in specs.items()
            }
        return _streams.Plan(entropy, antithetic, specs), complete

    def _streamed_components(
        self,
        run: _streams.Run,
        made: Optional[dict] = None,
        prefix: tuple = (),
    ) -> dict[Any, Any]:
        """What each component draws its events from in a run: a stand-in
        replaying its draws from its own streams, nested RBDs' components
        included, or the component itself when they cannot be streamed
        (see ``_stand_in``). The stand-ins are called at exactly the points
        the components would be.

        A component object used for several nodes has one simulation state,
        so it gets one stand-in (``made``, by object). Since each node has
        its own copy of a ``NonRepairable``, only a nested RBD object can be.
        """
        made = {} if made is None else made
        streamed: dict[Any, Any] = {}
        for name, component in self.components.items():
            if id(component) not in made:
                arrangement = self._standby.get(name)
                made[id(component)] = (
                    _stand_in(
                        component,
                        run,
                        made,
                        prefix + (name,),
                        self._duration_model(name),
                        self._imperfect.get(name),
                    )
                    if arrangement is None
                    else _standby_draws(
                        component, run, prefix + (name,), arrangement
                    )
                )
            streamed[name] = made[id(component)]
        return streamed

    def _keyed_charges(self, run: _streams.Run) -> Tuple[
        dict[Any, list[tuple[int, Any]]],
        dict[Any, Any],
        dict[Any, Any],
    ]:
        """For each costed node, its ``(category, charges)`` pairs: what is
        charged at each of its failures (the categories are positions in
        ``_CATEGORIES``); and, for each node with a preventive-maintenance
        or an inspection cost, what is charged at each preventive action or
        inspection. ``charges.draw()`` gives the next amount: a fixed cost,
        or the next draw from the cost's own stream (see ``_streams``)."""

        def charges(node, key: str):
            cost = self.costs[node][key]
            if isinstance(cost, float):
                return _Fixed(cost)
            return run.stream((node,), _streams.COST_KINDS[key])

        failures = {
            node: [
                (
                    _CATEGORIES.index(key.removesuffix("_cost")),
                    charges(node, key),
                )
                for key in self.PER_FAILURE_COST_KEYS
                if key in node_costs
            ]
            for node, node_costs in self.costs.items()
        }
        preventive = {
            node: charges(node, "preventive_cost")
            for node, node_costs in self.costs.items()
            if "preventive_cost" in node_costs
        }
        inspection = {
            node: charges(node, "inspection_cost")
            for node, node_costs in self.costs.items()
            if "inspection_cost" in node_costs
        }
        return failures, preventive, inspection

    def _context(
        self,
        t_simulation: float,
        working_nodes,
        broken_nodes,
        method: str,
        capacity: Optional[_CapacityRecorder],
        entropy,
        antithetic: bool,
        widths: Optional[dict] = None,
        states: Optional[dict] = None,
        history: bool = False,
    ) -> "_Context":
        """Everything a run's simulations share: the streams, what each
        component draws from, the charges, the structure function, and the
        components' ``states`` at the start (checked, as
        ``_simulation_states`` gives them; None for new). With
        ``history``, each simulation records its histories (see
        ``_replicate``)."""
        states = {} if states is None else states
        plan, complete = self._stream_plan(
            t_simulation, entropy, antithetic, widths, states
        )
        run = _streams.Run(plan, reseed=not complete)
        has_costs = self.has_costs
        nodes = list(self.components)
        return _Context(
            t_simulation=t_simulation,
            working=set(working_nodes),
            broken=set(broken_nodes),
            method=method,
            capacity=capacity,
            run=run,
            sources=self._streamed_components(run),
            charges=self._keyed_charges(run) if has_costs else ({}, {}, {}),
            downtime_cost_rates={
                node: c["downtime_cost"]
                for node, c in self.costs.items()
                if "downtime_cost" in c
            },
            has_costs=has_costs,
            position={node: i for i, node in enumerate(nodes)},
            # Components with no maintenance or inspection that are not
            # nested RBDs: their next event is simply their next draw.
            plain={
                node
                for node, component in self.components.items()
                if node not in self._preventive
                and node not in self._inspection
                and node not in self._standby
                and not isinstance(component, RepairableRBD)
            },
            works=self._decomposition().structure_function(method),
            states=states,
            initial_up=bool(
                self.is_system_working(
                    {c: c not in broken_nodes for c in self.components}, method
                )
            ),
            history=history,
        )

    def expected_cost_rate(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the long-run expected cost per unit time.

        Exact (no simulation), from the steady-state quantities:

        ```text
        rate = downtime_cost_rate * (1 - A_sys)
               + sum_i omega_i * (repair_cost_i + replace_cost_i)
               + sum_i nu_i * preventive_cost_i
               + sum_i inspection_cost_i / tau_i
               + sum_i (1 - A_i) * downtime_cost_i
        ```

        where ``A_sys`` is the system's long-run availability
        (``mean_availability``), ``A_i`` is node i's (see
        ``node_availability``), ``omega_i`` is node i's long-run failure
        frequency, ``1 / (MTTF_i + MTTR_i)``, and ``nu_i`` its frequency of
        preventive replacements. So ``repair_cost`` and ``replace_cost``
        are charged per corrective action (failure), the preventive cost
        per preventive action, and the downtime rates per unit time down.
        A cost given as a distribution enters through its mean. The result
        is in cost per unit of the component models' time.

        Under age replacement at ``T`` a component renews at a failure or
        at a preventive replacement, whichever comes first, in cycles of
        mean length ``C = integral_0^T R + F(T) * MTTR + R(T) * MTTP``
        (``MTTP`` the mean maintenance time), so ``omega_i = F(T) / C`` and
        ``nu_i = R(T) / C``: the renewal-reward rate. With instant repair
        and maintenance the component's own cost rate is
        ``(c_p * R(T) + c_u * F(T)) / integral_0^T R``, as
        ``NonRepairable.cost_rate`` computes.

        Under block replacement at ``T`` the renewals are the block times at
        which the component is up (see ``node_availability``): with ``L``
        the mean time between them and ``N`` the mean number of failures in
        between, ``omega_i = N / L`` and ``nu_i = 1 / L``. With instant
        repair and replacement that is ``(c_p + c_u * M(T)) / T``, ``M`` the
        renewal function of the lives.

        A component with hidden failures is inspected every ``tau_i`` and,
        with a constant failure rate ``lambda``, fails ``(1 - exp(-lambda *
        tau_i)) / tau_i`` times per unit time (at most once per interval).
        With inspection cost ``c_i`` and downtime cost rate ``c_d`` it
        costs ``c_i / tau + c_d * U(tau)``, ``U`` its unavailability (see
        ``node_availability``): frequent tests cost more, and rare ones
        leave failures hidden for longer. The rate is least near ``tau =
        sqrt(2 * c_i / (lambda * c_d))``.

        With limited ``repair_crews``, ``A_sys``, ``A_i`` and ``omega_i``
        come from the crews' Markov chain (see ``mean_availability``): a
        component waiting for a crew is down, and pays its downtime cost,
        and it fails at its constant rate while it is up.

        Every cost is optional and defaults to 0, so any subset can be
        priced; with nothing priced (see ``has_costs``) this is 0.0. Costs
        are undiscounted, and only this RBD's own costs count: costs
        declared inside a nested ``RepairableRBD`` are left out. The
        simulated counterpart is ``cost``, whose ``cost_rate`` converges to
        this value as the window grows.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working: they never fail, so
            they incur no corrective or downtime cost, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed: they incur no corrective
            cost (they never change state) but are down for all time, by
            default None.

        Returns
        -------
        float
            Expected cost per unit time.

        Raises
        ------
        ValueError
            If something is priced and a working/broken node is unknown, is
            the input or output node, or is in both sets, or a component
            has a non-parametric reliability model. (With nothing priced
            this returns 0.0 without any checks.)
        NotImplementedError
            If something is priced and a component is under block
            replacement with models its exact values do not cover (see
            ``node_availability``), or has hidden failures whose tests or
            repairs take time, or whose tests can miss a failure of a life
            that is not exponential; or, while a component can wait for a
            repair crew, as for ``mean_availability``: simulate it with
            ``cost``.

        Examples
        --------
        One component with MTTF 10 and MTTR 1 is up 10/11 of the time and
        fails 1/11 times per unit time, so at 100 per repair and 50 per unit
        time of outage the cost rate is ``(100 + 50) / 11``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...             "repair_cost": 100.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=50.0,
        ... )
        >>> round(rbd.expected_cost_rate(), 4)
        13.6364
        """
        if not self.has_costs:
            return 0.0

        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        forced = working_nodes | broken_nodes
        setups = [
            group for group in self._maintenance.values() if group.setup_cost
        ]
        if setups:
            self._require_separate_setups()

        rate = 0.0

        # Production lost while the *system* is down.
        if self.downtime_cost_rate:
            unavailability = self.mean_unavailability(
                working_nodes, broken_nodes
            )
            rate += self.downtime_cost_rate * unavailability

        if not self.costs and not setups:
            return rate

        if self._crews_couple():
            # Held working or broken, a node needs no crew, and the others
            # have more of them: from the chain without it.
            probabilities, weights = self._chain_probabilities(
                working_nodes, broken_nodes
            )
            node_availability = {
                node: float(weights @ probabilities[node])
                for node in self.nodes
            }
        else:
            node_availability = _squeeze_values(
                self._probabilities_with_overrides(
                    self.node_availability(), working_nodes, broken_nodes
                )
            )
        for node in self.costs:
            rate += self._node_cost_rate(
                node, node_availability[node], node in forced
            )
        # A maintenance group's set-up, at each failure and each preventive
        # replacement of a member (none is renewed early: that would have
        # been refused), a forced member making none.
        for group in setups:
            rate += group.setup_cost * sum(
                sum(self._node_actions(node, node_availability[node]))
                for node in group.members
                if node not in forced
            )
        return rate

    def _node_cost_rate(
        self, node, availability: float, forced: bool = False
    ) -> float:
        """A component's own running cost per unit time, in the long run:
        its corrective, preventive and inspection actions, and its own
        downtime (``availability`` its long-run availability). A forced node
        never changes state, so it incurs no actions."""
        node_costs = self.costs.get(node, {})
        rate = 0.0
        # Corrective actions, charged per failure, and preventive ones.
        per_action = sum(
            _mean_cost(node_costs[key])
            for key in self.PER_FAILURE_COST_KEYS
            if key in node_costs
        )
        preventive = node_costs.get("preventive_cost")
        if (per_action or preventive is not None) and not forced:
            failures, maintained = self._node_actions(node, availability)
            rate += per_action * failures
            if preventive is not None:
                rate += _mean_cost(preventive) * maintained
        inspection = node_costs.get("inspection_cost")
        if inspection is not None and not forced:
            schedule = self._preventive.get(node)
            if schedule is not None and schedule.policy == "condition":
                # Inspected at each multiple of the interval at which it is
                # up (one in a repair or replacement is not).
                up = self._block_cycle(node).before
                rate += _mean_cost(inspection) * up / schedule.interval
            else:
                # One test per interval (none is skipped: repairs are
                # instant, as the exact values require).
                self._require_tested_exact(node)
                rate += (
                    _mean_cost(inspection) / self._inspection[node].interval
                )
        # Optional cost of *this component* being down, whether or not the
        # system as a whole is.
        downtime_cost = node_costs.get("downtime_cost", 0.0)
        if downtime_cost:
            rate += downtime_cost * (1.0 - availability)
        return rate

    def _node_actions(self, node, availability: float) -> Tuple[float, float]:
        """A component's corrective and preventive actions per unit time,
        in the long run (``availability`` its long-run availability)."""
        if self._crews_couple():
            # Waiting for a crew, as while repaired, it cannot fail: it
            # fails at its constant rate while it is up.
            life = self._crew_chain_rates()[node][0]
            return life * availability, 0.0
        if node in self._standby:
            # Each of its units' failures is a repair.
            return self._standby_long_run(node).unit_failure_frequency, 0.0
        failures, maintained, _ = self._node_frequencies(node)
        return failures, maintained

    @property
    def acquisition_cost(self) -> float:
        """The one-off cost of buying the components: the sum of their
        ``"acquisition_cost"`` (0.0 if none is given). Only this RBD's own
        components count, not those inside a nested ``RepairableRBD``."""
        return float(sum(self.acquisition_costs.values()))

    def total_cost(
        self,
        horizon: float,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the total cost of owning the system for ``horizon``.

        The life-cycle cost, undiscounted: buying the components, then
        running the system for ``horizon`` at the long-run cost rate,

        ```text
        total = acquisition_cost + expected_cost_rate() * horizon
        ```

        with ``acquisition_cost`` the sum of the components'
        ``"acquisition_cost"``. The running cost is the long-run rate,
        exact over a horizon long compared with the components' cycles;
        ``expected_cost(horizon).total`` is the exact expected cost of
        owning the system from new, and ``cost`` simulates a window from new
        (whose ``CostResult`` gives the same ``acquisition_cost``
        separately).

        Parameters
        ----------
        horizon : float
            How long the system is owned, in the time unit of the component
            models: finite and non-negative.
        working_nodes : Collection[Hashable], optional
            As for ``expected_cost_rate``, by default None.
        broken_nodes : Collection[Hashable], optional
            As for ``expected_cost_rate``, by default None.

        Returns
        -------
        float
            The total cost over the horizon.

        Raises
        ------
        ValueError
            If ``horizon`` is not finite and non-negative, or as for
            ``expected_cost_rate``.
        NotImplementedError
            As for ``expected_cost_rate``.

        Examples
        --------
        A pump bought for 20,000, failing on average every 1000 hours,
        repaired in 10 at 500 per repair, with lost production at 100 per
        hour, owned for ten years:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "p"), ("p", "t")],
        ...     {
        ...         "p": {
        ...             "reliability": surv.Exponential.from_params([1e-3]),
        ...             "repairability": surv.Exponential.from_params([0.1]),
        ...             "repair_cost": 500.0,
        ...             "acquisition_cost": 20000.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=100.0,
        ... )
        >>> round(rbd.expected_cost_rate(), 4)  # (500 + 100 * 10) / 1010
        1.4851
        >>> round(rbd.total_cost(87600.0))
        150099
        """
        horizon = _horizon(horizon)
        return self.acquisition_cost + horizon * self.expected_cost_rate(
            working_nodes, broken_nodes
        )

    def _spares_nodes(self, nodes) -> list:
        """The components whose spares are counted: ``nodes`` checked, or
        every component that is not a nested RBD."""
        if nodes is None:
            return [
                node
                for node, component in self.components.items()
                if not isinstance(component, RepairableRBD)
            ]
        chosen = list(dict.fromkeys(nodes))
        for node in chosen:
            if node not in self.components:
                raise ValueError(
                    f"Node {node!r} in nodes is not a component of this RBD."
                )
            if isinstance(self.components[node], RepairableRBD):
                raise ValueError(
                    f"Node {node!r} is a nested RBD: its spares are its "
                    "components', which its own spares_demand counts."
                )
        return chosen

    def _replacements(self, node, long_run: bool = False):
        """What a component's replacements follow, for counting them
        exactly (see ``_spares``): a renewal process, a block schedule's
        (``_spares.Block``) or a test lattice's (``_spares.Tested``); raise
        if they are not counted exactly. ``long_run``: for the counts in a
        lead time, which need a demand that goes on."""
        simulate = (
            "count its spares by simulation: spares_demand(method='simulate')."
        )
        if node in self._standby:
            raise NotImplementedError(
                f"Component {node!r} is a standby group, whose units' "
                f"failures depend on each other: {simulate}"
            )
        if node in self._inspection:
            return self._tested_replacements(node, simulate)
        schedule = self._preventive.get(node)
        if schedule is not None and schedule.policy == "block":
            if long_run:
                raise NotImplementedError(
                    f"Component {node!r} is replaced on a block schedule: "
                    "its demand in a lead time depends on where the lead "
                    "time falls in the block interval, which the stock "
                    "levels do not take yet (#160). spares_demand counts "
                    "its spares over a horizon."
                )
            component = self.components[node]
            duration = schedule.duration
            return _spares.Block(
                _cdf(component.reliability),
                _cdf(component.time_to_replace),
                None if duration is None else _cdf(duration),
                float(schedule.interval),
            )
        if schedule is not None and schedule.policy == "condition":
            raise NotImplementedError(
                f"Component {node!r} is replaced on condition, at "
                f"inspections on a calendar that do not renew it: {simulate}"
            )
        if node in self._early_members:
            raise NotImplementedError(
                f"Component {node!r} is renewed early at the stops of its "
                f"maintenance group, which depend on the other members: "
                f"{simulate}"
            )
        if node in self._imperfect:
            raise NotImplementedError(
                f"Component {node!r} is repaired imperfectly "
                f"({self._imperfect_phrase(node)}), so its replacements are "
                f"not a renewal process of lives as new: {simulate}"
            )
        component = self.components[node]
        life, repair = component.reliability, component.time_to_replace
        age = math.inf if schedule is None else float(schedule.interval)
        maintenance = None if schedule is None else schedule.duration
        if math.isinf(age):
            up = _safe_mean(life)
            fails = 1.0
        else:
            up = float(component.avg_replacement_time(age))
            fails = float(np.ravel(_cdf(life)(np.array([age])))[0])
        down = fails * _safe_mean(repair) + (1.0 - fails) * (
            0.0 if maintenance is None else _safe_mean(maintenance)
        )
        mean_cycle = up + down
        if long_run and not (0.0 < mean_cycle < math.inf):
            raise NotImplementedError(
                f"Component {node!r}'s life may never end, or has no mean, "
                "so its demand has no long run: it has no stock level."
            )
        return _spares.Replacements(
            _cdf(life),
            _cdf(repair),
            None if maintenance is None else _cdf(maintenance),
            age,
            mean_cycle,
        )

    def _tested_replacements(self, node, simulate: str) -> "_spares.Tested":
        """A component with hidden failures: its replacements fall on its
        tests, as tested and repaired in no time with tests that find every
        failure (see ``_spares.Tested``); raise otherwise."""
        inspection = self._inspection[node]
        component = self.components[node]
        if (
            inspection.duration is not None
            or model_mean(component.time_to_replace) != 0.0
            or inspection.partial
        ):
            raise NotImplementedError(
                f"Component {node!r}'s failures are found by tests: its "
                "spares are counted exactly only when its tests and repairs "
                "take no time and its tests find every failure (#159): "
                f"{simulate}"
            )
        life = self._tested_life(node) or TestedLife(
            component.reliability, inspection.interval
        )
        cycle = np.where(
            life.down[:-1] <= 0.5,
            np.diff(life.down),
            life.up[:-1] - life.up[1:],
        )
        model = component.reliability
        first = float(inspection.offset or inspection.interval)
        interval = float(inspection.interval)

        def found(tests: int) -> np.ndarray:
            # The unit new at 0 fails before the first test, or between two.
            times = first + interval * np.arange(tests)
            failed = np.clip(_cdf(model)(times), 0.0, 1.0)
            survive = np.clip(_sf_values(model.sf, times), 0.0, 1.0)
            before = np.concatenate([[0.0], failed[:-1]])
            alive = np.concatenate([[1.0], survive[:-1]])
            return np.maximum(
                np.where(before <= 0.5, failed - before, alive - survive),
                0.0,
            )

        return _spares.Tested(first, interval, found, np.maximum(cycle, 0.0))

    def spares_demand(
        self,
        horizon: float,
        *,
        nodes: Optional[Collection[Hashable]] = None,
        fleet: int = 1,
        method: str = "exact",
        mc_samples: Optional[int] = None,
        seed=None,
    ) -> Dict[Hashable, SparesDemand]:
        """How many spares each component uses over ``[0, horizon)``, from
        new: the distribution of its replacements, for one system or a
        fleet of ``fleet``.

        A component uses a spare at each failure and at each preventive
        replacement; a standby group at each of its units' failures. Its
        replacements are a renewal process: each up time (its life, or the
        replacement age if that comes first) ends in one, then a repair or
        maintenance time, and it starts again as new. So the ``n``-th
        replacement comes after ``n - 1`` whole cycles and an up time, and
        the probability of ``n`` or more by the horizon is that of their
        sum falling before it (a replacement at the horizon itself falls
        after it, as in the simulation). ``method="exact"`` works that out
        on a grid, to about 1e-6, for components with corrective repair
        alone or under age replacement. Under block replacement, a unit's
        next replacement is at its failure or at the next block time,
        whichever comes first (a unit down then skips it), counted block
        interval by block interval on the grid. With hidden failures
        tested and repaired in no time, the replacements fall on the tests,
        and are counted there exactly. A fleet's count is the sum of its
        systems', independent and each from new. ``method="simulate"``
        counts them in ``mc_samples`` simulations of the whole system
        instead, which also covers standby groups, repair crews and tests
        that take time. With constant failure rates and instant repair,
        the counts are Poisson.

        Parameters
        ----------
        horizon : float
            The time counted over, from new.
        nodes : Collection[Hashable], optional
            The components to count for, by default every component but a
            nested RBD (whose own ``spares_demand`` counts its components').
        fleet : int, optional
            How many systems, by default 1.
        method : str, optional
            ``"exact"`` (the default) or ``"simulate"``.
        mc_samples : int, optional
            The number of simulations for ``method="simulate"``, by default
            10_000.
        seed : int, optional
            Seeds the simulations, by default None.

        Returns
        -------
        dict[Hashable, SparesDemand]
            Per component, the distribution of the spares it uses, with its
            ``mean()``, ``std()``, and ``stock(probability)``: the fewest
            spares that cover the horizon with that probability.

        Raises
        ------
        ValueError
            If ``horizon`` is negative or not finite, ``fleet`` is not a
            whole number of at least 1, ``method`` is unknown, or ``nodes``
            names a node that is not a component, or a nested RBD.
        NotImplementedError
            With ``method="exact"``, for a standby group, a component with
            hidden failures whose tests or repairs take time or whose tests
            can miss a failure, or while a component can wait for a repair
            crew (see ``repair_crews``), or if more than 2,000 replacements
            are likely: count them by simulation.

        Examples
        --------
        A pump with a constant failure rate of 0.01 an hour, replaced in no
        time, for a fleet of 5 over 1,000 hours: Poisson with mean 50.

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "pump"), ("pump", "t")],
        ...     {
        ...         "pump": {
        ...             "reliability": surv.Exponential.from_params([0.01]),
        ...             "repairability": "instant",
        ...         }
        ...     },
        ... )
        >>> demand = rbd.spares_demand(1000.0, fleet=5)["pump"]
        >>> round(demand.mean(), 4)
        50.0
        >>> demand.stock(0.95)
        62
        """
        end = _horizon(horizon)
        fleet = _whole("fleet", fleet)
        if method not in ("exact", "simulate"):
            raise ValueError(
                f"method must be 'exact' or 'simulate', got {method!r}."
            )
        chosen = self._spares_nodes(nodes)
        if method == "simulate":
            counts = self._simulated_replacements(
                end, chosen, 10_000 if mc_samples is None else mc_samples, seed
            )
        else:
            self._require_unlimited_crews(*_SPARES_CREWS)
            models = {node: self._replacements(node) for node in chosen}
            counts = {
                node: _spares.count(model, end, "new")
                for node, model in models.items()
            }
        return {
            node: SparesDemand(_fleet(count, fleet), end, fleet, method)
            for node, count in counts.items()
        }

    def _simulated_replacements(
        self, horizon: float, nodes: list, mc_samples: int, seed
    ) -> Dict[Any, np.ndarray]:
        """The fractions of ``mc_samples`` simulations of ``[0, horizon)``
        in which each of ``nodes`` was replaced ``0, 1, 2, ...`` times."""
        montecarlo.check_count(mc_samples, False, "mc_samples")
        if horizon == 0.0:
            # Nothing is replaced in no time (as the exact count says).
            return {node: np.ones(1) for node in nodes}
        tally = self._run(
            horizon,
            set(),
            set(),
            "p",
            mc_samples,
            False,
            seed,
            replacements=True,
        )
        counts = np.array(tally.replacements, dtype=np.int64)
        column = {node: c for c, node in enumerate(self.components)}
        return {
            node: np.bincount(counts[:, column[node]]) / len(counts)
            for node in nodes
        }

    def spares_stock(
        self,
        lead_time: float,
        *,
        fill_rate: Optional[float] = None,
        stockout_probability: Optional[float] = None,
        nodes: Optional[Collection[Hashable]] = None,
        fleet: int = 1,
    ) -> Dict[Hashable, SparesStock]:
        """The fewest spares of each component to hold for a fill rate, or
        to run out no more than a fraction of the time, when each spare
        used is reordered at once and arrives ``lead_time`` later
        (one-for-one, or ``(S - 1, S)``, replenishment), in the long run.

        With a stock of ``S`` a spare is on the shelf while fewer than
        ``S`` are on order: those used in the last lead time. The demand is
        the component's replacements (see ``spares_demand``), in the long
        run: from a random time the first comes at the end of what is left
        of an up time, or after what is left of a repair or maintenance and
        an up time, and the next after whole cycles. So the stock-out
        probability is that of ``S`` or more replacements in a lead time
        from a random time, and the fill rate that of fewer than ``S`` in
        the lead time before a replacement. For a fleet, the systems'
        demands add up. Worked out on a grid, to about 1e-6, for components
        with corrective repair alone or under age replacement; and exactly,
        on its tests, for a component with hidden failures tested and
        repaired in no time (from a random time, the next replacement is
        ``j`` tests on with probability ``R((j - 1) T) / S``, ``S`` the
        mean cycle in tests).

        Parameters
        ----------
        lead_time : float
            The time a spare takes to arrive once ordered.
        fill_rate : float, optional
            The fraction of demands to meet from the shelf, in ``(0, 1)``.
        stockout_probability : float, optional
            The most fraction of time to be out of stock, in ``(0, 1)``.
            At least one of the two targets must be given; the stock meets
            both.
        nodes : Collection[Hashable], optional
            The components, by default every component but a nested RBD.
        fleet : int, optional
            How many systems share the stock, by default 1.

        Returns
        -------
        dict[Hashable, SparesStock]
            Per component, the stock and what it achieves, with the
            distributions behind them.

        Raises
        ------
        ValueError
            If ``lead_time`` is negative or not finite, neither target is
            given or one is not in ``(0, 1)``, ``fleet`` is not a whole
            number of at least 1, or ``nodes`` names a node that is not a
            component, or a nested RBD.
        NotImplementedError
            For a component under block replacement (#160), with hidden
            failures whose tests or repairs take time or whose tests can
            miss a failure, a standby group or a life that may never end,
            or while a component can wait for a repair crew, or if more
            than 2,000 replacements are likely in a lead time.

        Examples
        --------
        A pump with a constant failure rate of 0.01 an hour, replaced in no
        time, and 300 hours to get a spare: 3 are on order on average, and
        7 on the shelf meet 96.6% of demands.

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "pump"), ("pump", "t")],
        ...     {
        ...         "pump": {
        ...             "reliability": surv.Exponential.from_params([0.01]),
        ...             "repairability": "instant",
        ...         }
        ...     },
        ... )
        >>> stock = rbd.spares_stock(300.0, fill_rate=0.95)["pump"]
        >>> stock.stock
        7
        >>> round(stock.fill_rate, 4)
        0.9665
        """
        tau = _horizon(lead_time)
        fleet = _whole("fleet", fleet)
        if fill_rate is None and stockout_probability is None:
            raise ValueError(
                "Give a fill_rate, a stockout_probability, or both."
            )
        for name, target in (
            ("fill_rate", fill_rate),
            ("stockout_probability", stockout_probability),
        ):
            if target is not None and not 0.0 < target < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {target!r}.")
        chosen = self._spares_nodes(nodes)
        self._require_unlimited_crews(*_SPARES_CREWS)
        models = {node: self._replacements(node, True) for node in chosen}
        out: Dict[Hashable, SparesStock] = {}
        for node, model in models.items():
            random = _spares.count(model, tau, "random")
            arrival = _spares.count(model, tau, "arrival")
            on_order = _fleet(random, fleet)
            at_demand = (
                arrival
                if fleet == 1
                else _fleet(np.convolve(arrival, _fleet(random, fleet - 1)), 1)
            )
            result = SparesStock(0, 0.0, 1.0, tau, fleet, on_order, at_demand)
            stock = 0
            while not (
                (
                    fill_rate is None
                    or result.fill_rate_for(stock) >= fill_rate - 1e-12
                )
                and (
                    stockout_probability is None
                    or result.stockout_probability_for(stock)
                    <= stockout_probability + 1e-12
                )
            ):
                stock += 1
            out[node] = dataclasses.replace(
                result,
                stock=stock,
                fill_rate=result.fill_rate_for(stock),
                stockout_probability=result.stockout_probability_for(stock),
            )
        return out

    def allocate_redundancy(
        self,
        horizon: float,
        *,
        nodes: Optional[Collection[Hashable]] = None,
        min_availability: Optional[float] = None,
        max_units: Union[int, Dict[Hashable, int], None] = None,
        method: str = "exact",
    ) -> TotalCostAllocation:
        """Choose the redundancy with the lowest total cost of ownership.

        How many identical copies of each node to fit in active parallel so
        that owning the system for ``horizon`` costs least: each copy costs
        its ``"acquisition_cost"`` to buy and its running costs (repairs,
        replacements, preventive maintenance, inspections, its own downtime
        cost) to keep, and together the copies save the cost of the system
        being down (``downtime_cost_rate``). That is ``total_cost`` for the
        system with the copies drawn out:

        ```text
        total = sum_i n_i * (a_i + horizon * r_i)
                + horizon * downtime_cost_rate * (1 - A_sys)
                + the cost of the nodes not considered
        ```

        with ``n_i`` copies of node ``i``, each bought for ``a_i`` and
        running at ``r_i`` per unit time, and ``A_sys`` the system's
        long-run availability. The copies fail and are repaired
        independently, so ``n`` copies of a node of long-run availability
        ``A`` are all down a fraction ``(1 - A) ** n`` of the time, and each
        design is scored exactly, with no simulation. (Copies of a
        component with hidden failures are inspected together, and are
        scored over the inspection period, as ``mean_availability`` does.)

        More copies cost more and save less and less downtime, so the total
        is not monotone in them. ``method="exact"`` finds a proven optimum
        from the greedy solution: a design no worse than that spends no more
        on copies than its total, which caps every node's copies, and
        (without ``min_availability``) no node takes a copy that could not
        pay for itself, since the ``k + 1``-th copy of a node of
        unavailability ``U`` can save at most ``horizon *
        downtime_cost_rate * U ** k * (1 - U)``. When every node considered
        is in series with the rest of the system (it lies on every path)
        and has no hidden failures, the system's availability is theirs
        times the rest's, and a dynamic program over the nodes finds the
        optimum however many there are; otherwise a branch and bound over
        the designs does, which suits a handful of nodes.
        ``method="greedy"`` adds or removes one copy at a time while that
        lowers the total: fast, but not guaranteed optimal. Discounting is
        not modelled.

        Parameters
        ----------
        horizon : float
            How long the system is owned, in the time unit of the component
            models: finite and non-negative.
        nodes : Collection[Hashable], optional
            The components that may be given copies, by default every one
            with an ``"acquisition_cost"``. The others stay as they are, and
            their costs are counted once. Nested ``RepairableRBD`` nodes
            cannot be given copies.
        min_availability : float, optional
            Only consider designs whose long-run availability is at least
            this, in (0, 1), by default no limit.
        max_units : int or dict, optional
            The most copies (at least 1) of every node considered (an int)
            or of particular ones (a dict; nodes it leaves out are
            unlimited), by default unlimited. A node whose copies cost
            nothing over the horizon needs one.
        method : str, optional
            ``"exact"`` (the default) or ``"greedy"``. The exact search
            gives up with an explanatory error after examining 500,000
            designs (the dynamic program, after holding 2,000,000 partial
            ones).

        Returns
        -------
        TotalCostAllocation
            The chosen ``units`` per node, with the design's
            ``total_cost``, ``acquisition_cost``, ``cost_rate`` and
            ``availability`` (see
            [`TotalCostAllocation`][repyability.TotalCostAllocation]).

        Raises
        ------
        ValueError
            If ``horizon``, ``nodes``, ``min_availability``, ``max_units``
            or ``method`` is invalid; if no component has an acquisition
            cost and ``nodes`` is not given; if a node's copies cost nothing
            and are not capped; if ``min_availability`` cannot be reached;
            if a component has a non-parametric reliability model; or if
            the exact search examines more than 500,000 designs.
        NotImplementedError
            If a component is under block replacement with models its exact
            values do not cover (see ``node_availability``), or has hidden
            failures whose tests or repairs take time, or whose tests can
            miss a failure of a life that is not exponential: its long-run
            values are not known exactly. Or if a component can wait for a
            repair crew (see ``repair_crews``): the search assumes that none
            does.

        Examples
        --------
        A pump that fails on average every 1000 hours and takes 10 to
        repair, bought for 20,000 and repaired for 500, when an hour without
        pumping costs 100. Over ten years (87,600 hours) a second pump pays
        for itself; a third would not:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "pump"), ("pump", "t")],
        ...     {
        ...         "pump": {
        ...             "reliability": surv.Exponential.from_params([1e-3]),
        ...             "repairability": surv.Exponential.from_params([0.1]),
        ...             "repair_cost": 500.0,
        ...             "acquisition_cost": 20000.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=100.0,
        ... )
        >>> round(rbd.total_cost(87600.0))  # one pump
        150099
        >>> best = rbd.allocate_redundancy(87600.0)
        >>> best.units, round(best.total_cost)
        ({'pump': 2}, 127591)

        Over one year the second pump does not pay for itself:

        >>> rbd.allocate_redundancy(8760.0).units
        {'pump': 1}
        """
        horizon = _horizon(horizon)
        if method not in ("exact", "greedy"):
            raise ValueError(
                f"method must be 'exact' or 'greedy', got {method!r}."
            )
        if nodes is None:
            chosen = [
                n for n in self.components if n in self.acquisition_costs
            ]
            if not chosen:
                raise ValueError(
                    "No component has an acquisition_cost, so there is "
                    "nothing to buy copies of: give the components one, or "
                    "name the nodes that may be given copies with `nodes`."
                )
        else:
            chosen = list(dict.fromkeys(nodes))
            if not chosen:
                raise ValueError("nodes must name at least one component.")
            for node in chosen:
                if node not in self.components:
                    raise ValueError(
                        f"Node {node!r} in nodes is not a component of this "
                        "RBD."
                    )
                if isinstance(self.components[node], RepairableRBD):
                    raise ValueError(
                        f"Node {node!r} is a nested RepairableRBD, which "
                        "cannot be given copies here: its costs are not "
                        "this RBD's."
                    )
        caps = redundancy_caps(chosen, max_units, named="nodes")
        max_unavailability = None
        if min_availability is not None:
            try:
                target = float(min_availability)
            except (TypeError, ValueError):
                target = float("nan")
            if not 0.0 < target < 1.0:
                raise ValueError(
                    "min_availability must be a number in (0, 1), got "
                    f"{min_availability!r}."
                )
            max_unavailability = 1.0 - target

        # Adding copies adds jobs for the crews: the search assumes none waits.
        self._require_unlimited_crews(*_ALLOCATION_CREWS)
        self._require_no_ccf("the allocations")
        # Every node's availability (and unavailability) over the times the
        # long-run values average over, and each component's own cost.
        times, weights = self._long_run_grid()
        up = {
            node: np.atleast_1d(np.asarray(a, dtype=float))
            for node, a in self._availabilities_at(times).items()
        }
        down = {
            node: np.atleast_1d(np.asarray(u, dtype=float))
            for node, u in self._unavailabilities_at(times).items()
        }
        copy_cost = {}
        for node in self.components:
            rate = self._node_cost_rate(node, self._node_availability(node))
            copy_cost[node] = (
                self.acquisition_costs.get(node, 0.0),
                rate,
                self.acquisition_costs.get(node, 0.0) + horizon * rate,
            )
        for node, cap in zip(chosen, caps):
            if copy_cost[node][2] <= 0.0 and cap == math.inf:
                raise ValueError(
                    f"A copy of {node!r} costs nothing over the horizon (it "
                    "has no acquisition or running cost), so copies could be "
                    "added without end: give it an acquisition_cost or a "
                    "max_units."
                )
        decomposition = self._decomposition()
        size = len(times)

        def unavailability(counts) -> float:
            p, q = dict(up), dict(down)
            for node, n in zip(chosen, counts):
                if n != 1:
                    q[node] = down[node] ** n
                    p[node] = 1.0 - q[node]
            _, fails = decomposition.probabilities(
                p, q, shape=size, works=False, fails=True
            )
            fails = np.broadcast_to(np.asarray(fails, dtype=float), (size,))
            return float(weights @ fails)

        def gain(node):
            # The most the k + 1-th copy of the node can lower the system
            # unavailability: the node's own fall in unavailability.
            return lambda k: float(weights @ (down[node] ** k * up[node]))

        def in_series(node) -> bool:
            # Down alone, it takes the system down: it lies on every path.
            alone = {n: np.ones(1) for n in self.nodes}
            alone[node] = np.zeros(1)
            return float(self.system_probability(alone)[0]) == 0.0

        series = None
        varying = set(self._inspection) | set(self._block_nodes())
        if all(node not in varying and in_series(node) for node in chosen):
            # Each such node's availability is constant, and the system's is
            # theirs times the rest's: the exact search is then a dynamic
            # program over the nodes.
            series = [
                (lambda n, q=float(down[node][0]): q**n) for node in chosen
            ]
        counts, _, u = lowest_total_cost(
            unavailability,
            [copy_cost[node][2] for node in chosen],
            horizon * self.downtime_cost_rate,
            [gain(node) for node in chosen],
            caps,
            max_unavailability,
            method,
            series,
        )
        units = {node: int(n) for node, n in zip(chosen, counts)}
        acquisition = math.fsum(
            units.get(node, 1) * copy_cost[node][0] for node in self.components
        )
        cost_rate = (
            math.fsum(
                units.get(node, 1) * copy_cost[node][1]
                for node in self.components
            )
            + self.downtime_cost_rate * u
        )
        return TotalCostAllocation(
            units=units,
            total_cost=acquisition + horizon * cost_rate,
            acquisition_cost=acquisition,
            cost_rate=cost_rate,
            availability=1.0 - u,
            horizon=horizon,
            method=method,
        )

    def availability_allocation(
        self,
        target: float,
        method: str = "cost_based",
        *,
        fixed: Optional[Collection[Hashable]] = None,
        weights: Optional[Dict] = None,
        max_availabilities: Optional[Dict] = None,
        feasibility: Optional[Dict] = None,
    ) -> AvailabilityAllocation:
        """What availability each component needs for the system to meet a
        target, and the MTTF or MTTR that gives it.

        Availability allocation: one of the reliability allocation methods
        (see ``cost_based_allocation``), run on the components' long-run
        availabilities (``node_availability``), with the system scored as
        ``mean_availability`` scores it. A component's availability is
        ``MTTF / (MTTF + MTTR)``, so the availability ``A`` allocated to it
        is met by an MTTF of ``MTTR * A / (1 - A)`` at its current MTTR, or
        by an MTTR of ``MTTF * (1 - A) / A`` at its current MTTF (or by any
        pair in the same ratio): both are reported. To choose the cheapest
        pair instead, use ``mttf_mttr_allocation``.

        Only components with corrective repair alone are allocated an
        availability: those that fail and take time to repair, with no
        ``"preventive"`` or ``"inspection"`` schedule. The others keep
        theirs: components with a schedule (whose intervals
        ``optimal_replacement_intervals`` and
        ``optimal_inspection_intervals`` choose), components repaired
        instantly (never down), nested ``RepairableRBD`` nodes, components
        with units that never fail or repairs that may never end, and the
        components in ``fixed``. A held component that is inspected or
        block-replaced is up with a probability that varies over its
        schedule, together with the others on the same calendar; it enters
        with that variation, as it does in ``mean_availability``, so that
        the allocation meets the target exactly.

        Parameters
        ----------
        target : float
            The system's long-run availability to reach, in [0, 1].
        method : str, optional
            How to allocate it, from the current availabilities:
            ``"cost_based"`` (the default) for Mettas's cheapest allocation
            (see ``cost_based_allocation``); ``"improvement"`` to scale
            every unavailability by one factor (``improvement_allocation``);
            ``"minimum_effort"``, for a series system, to raise the least
            available components to one level
            (``minimum_effort_allocation``); or ``"equal"`` to give every
            component allocated an availability the same one.
        fixed : Collection[Hashable], optional
            Components that keep their availability.
        weights : dict, optional
            With ``"improvement"`` only: a weight for each component
            allocated an availability (see ``improvement_allocation``).
        max_availabilities : dict, optional
            With ``"cost_based"`` only: the most a component's availability
            can reach. A component without one can approach 1.
        feasibility : dict, optional
            With ``"cost_based"`` only: a component's feasibility, in
            [0, 1), 0.5 without one: the higher, the easier to improve.

        Returns
        -------
        AvailabilityAllocation
            The ``availability`` of every component; for each one allocated
            an availability, the ``mttf`` that gives it at the current MTTR
            and the ``mttr`` that gives it at the current MTTF; and the
            ``system_availability`` with them. The solver's result is
            stored on the RBD as ``res``, as the method run stores it.

        Raises
        ------
        ValueError
            If ``target`` is not in [0, 1] or cannot be reached (the message
            gives how far the system can go); if ``method`` is unknown, or
            an option is given to a method that does not take it; if
            ``fixed`` or an option names a node that is not a component, or
            an option names a component that keeps its availability; if no
            component can be allocated an availability; if
            ``"minimum_effort"`` is asked of a system that is not a series;
            or if a component has a non-parametric reliability model.
        KeyError
            If ``weights`` has no entry for a component allocated an
            availability.
        NotImplementedError
            As for ``mean_availability``, or if a component can wait for a
            repair crew (see ``repair_crews``): the allocation assumes
            independent components.

        Examples
        --------
        Two pumps in parallel, each with an MTTF of 10 h and an MTTR of
        1 h, in series with a valve with an MTTF of 50 h and an MTTR of
        2 h, are up 95.4% of the time. For 98%, the cheapest allocation
        asks most of the valve, which every path goes through:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def unit(mttf, mttr):
        ...     return {
        ...         "reliability": surv.Exponential.from_params([1 / mttf]),
        ...         "repairability": surv.Exponential.from_params([1 / mttr]),
        ...     }
        >>> plant = RepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {"p1": unit(10, 1), "p2": unit(10, 1), "v": unit(50, 2)},
        ... )
        >>> round(plant.mean_availability(), 4)
        0.9536
        >>> need = plant.availability_allocation(0.98)
        >>> {node: round(a, 4) for node, a in need.availability.items()}
        {'p1': 0.9318, 'p2': 0.9318, 'v': 0.9846}

        The valve needs an MTTF of 128 h at its 2 h repairs, or repairs of
        0.78 h at its 50 h MTTF:

        >>> round(need.mttf["v"]), round(need.mttr["v"], 2)
        (128, 0.78)
        """
        methods = ("cost_based", "improvement", "minimum_effort", "equal")
        if method not in methods:
            raise ValueError(
                "method must be one of "
                f"{', '.join(repr(m) for m in methods)}, got {method!r}."
            )
        options = {
            "weights": (weights, "improvement"),
            "max_availabilities": (max_availabilities, "cost_based"),
            "feasibility": (feasibility, "cost_based"),
        }
        for name, (value, owner) in options.items():
            if value is not None and method != owner:
                raise ValueError(f"{name} applies to method={owner!r} only.")
        target = _allocation_target(target)
        current, free, held = self._allocatable(fixed)
        for name, (value, _) in options.items():
            self._allocation_option(name, value, held)
        view = self._allocation_view(held)
        start = {node: current[node] for node in self.nodes}
        if method == "cost_based":
            maxima = {node: current[node] for node in held}
            maxima.update(max_availabilities or {})
            allocated = view.cost_based_allocation(
                target,
                start,
                max_probabilities=maxima,
                feasibility=feasibility,
            )
        elif method == "improvement":
            allocated = view.improvement_allocation(
                target, start, fixed=list(held), weights=weights
            )
        elif method == "equal":
            # From 0.5 for every component allocated, scaled alike, so they
            # all end up the same.
            allocated = view.improvement_allocation(
                target,
                {
                    node: current[node] if node in held else 0.5
                    for node in self.nodes
                },
                fixed=list(held),
            )
        else:
            allocated = view._minimum_effort_held(target, current, held)
        if view is not self and method != "minimum_effort":
            self.res = view.res
        mttf, mttr = {}, {}
        for node, (up, down) in free.items():
            a = allocated[node]
            mttf[node] = down * a / (1.0 - a) if a < 1.0 else math.inf
            mttr[node] = up * (1.0 - a) / a if a > 0.0 else math.inf
        return self._allocation_result(view, allocated, mttf, mttr)

    def mttf_mttr_allocation(
        self,
        target: float,
        *,
        levers: str = "both",
        fixed: Optional[Collection[Hashable]] = None,
        max_mttf: Optional[Dict] = None,
        min_mttr: Optional[Dict] = None,
        mttf_feasibility: Optional[Dict] = None,
        mttr_feasibility: Optional[Dict] = None,
    ) -> AvailabilityAllocation:
        """The cheapest MTTFs and MTTRs that meet a system availability
        target.

        A component's availability, ``MTTF / (MTTF + MTTR)``, can be raised
        with a longer MTTF (reliability) or a shorter MTTR
        (maintainability), which cost different amounts. This is Mettas's
        cost-based allocation (see ``cost_based_allocation``) with both
        levers: it finds each component's MTTF and MTTR that meet the target
        at the least total cost, where lowering its failure rate
        ``lambda = 1 / MTTF`` from ``lambda_0`` towards its least,
        ``lambda_min = 1 / max_mttf``, costs

            exp((1 - f) * (lambda_0 - lambda) / (lambda - lambda_min)),

        and cutting its MTTR from ``MTTR_0`` towards its least,
        ``min_mttr``, costs

            exp((1 - f) * (MTTR_0 - MTTR) / (MTTR - min_mttr)),

        each with its own feasibility ``f`` in [0, 1): the lower it is, the
        faster the cost rises. Each cost is 1 while its lever is unused and
        grows without bound towards its limit. Without limits, raising the
        MTTF by a factor ``k`` costs ``exp((1 - f) * (k - 1))``, and so
        does cutting the MTTR by one: the availability depends on the ratio
        of the two alone, so at equal feasibilities both are used alike.
        The cheaper lever is used first, and the dearer one only once the
        cheaper one costs as much at the margin.

        With ``levers="mttr"`` the failure behaviour is held and only the
        repairs are allocated (maintainability allocation): a shorter
        repair is worth most on a component that is often down, so, other
        things equal, the components that fail most often get the shortest
        repairs. With ``levers="mttf"`` only the MTTFs change.

        The components allocated, and those that keep their availability,
        are as for ``availability_allocation``, and the system is scored as
        ``mean_availability`` scores it. It is solved with
        ``scipy.optimize.minimize`` (SLSQP) with exact gradients, from the
        point where every lever has closed the same fraction of its gap to
        its limit, and the design returned meets the target. The solver's
        result is stored on the RBD as ``res``, replacing any earlier one;
        its ``fun`` is the log of the total cost of the levers that may
        change.

        Parameters
        ----------
        target : float
            The system's long-run availability to reach, in [0, 1].
        levers : str, optional
            ``"both"`` (the default), ``"mttr"`` to change only the MTTRs, or
            ``"mttf"`` to change only the MTTFs.
        fixed : Collection[Hashable], optional
            Components that keep their MTTF and MTTR.
        max_mttf : dict, optional
            The most a component's MTTF can reach, at least its current
            MTTF (which holds it). A component without one has no limit.
        min_mttr : dict, optional
            The least a component's MTTR can reach, from 0 (the default for
            a component without one) to its current MTTR (which holds it).
        mttf_feasibility : dict, optional
            The feasibility of raising a component's MTTF, in [0, 1), 0.5
            without one.
        mttr_feasibility : dict, optional
            The feasibility of cutting a component's MTTR, in [0, 1), 0.5
            without one.

        Returns
        -------
        AvailabilityAllocation
            Every component's ``availability``; for each one allocated, its
            ``mttf`` and ``mttr`` in the cheapest design (together); and the
            ``system_availability``. A target the system already meets
            returns the current values.

        Raises
        ------
        ValueError
            If ``target`` is not in [0, 1] or cannot be reached (the message
            gives the most the limits allow: they are only approached, at an
            ever-growing cost); if ``levers`` is unknown; if ``fixed`` or an
            option names a node that is not a component, or an option names
            one that keeps its availability; if a limit is on the wrong side
            of the current value, or a feasibility is outside [0, 1); if no
            lever can change; or if a component has a non-parametric
            reliability model.
        NotImplementedError
            As for ``mean_availability``, or if a component can wait for a
            repair crew (see ``repair_crews``): the allocation assumes
            independent components.

        Warns
        -----
        UserWarning
            If the solver stops before converging; the design returned
            still meets the target, but may not be the cheapest.

        Examples
        --------
        The pumps and valve of ``availability_allocation``, to 98%: at
        equal feasibilities each component's MTTF rises by the factor its
        MTTR falls by, and the valve's the most:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def unit(mttf, mttr):
        ...     return {
        ...         "reliability": surv.Exponential.from_params([1 / mttf]),
        ...         "repairability": surv.Exponential.from_params([1 / mttr]),
        ...     }
        >>> plant = RepairableRBD(
        ...     [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"),
        ...      ("v", "t")],
        ...     {"p1": unit(10, 1), "p2": unit(10, 1), "v": unit(50, 2)},
        ... )
        >>> design = plant.mttf_mttr_allocation(0.98)
        >>> {node: round(t, 1) for node, t in design.mttf.items()}
        {'p1': 10.6, 'p2': 10.6, 'v': 85.5}
        >>> {node: round(t, 2) for node, t in design.mttr.items()}
        {'p1': 0.94, 'p2': 0.94, 'v': 1.17}
        >>> round(design.system_availability, 4)
        0.98

        Holding the failure behaviour, only the repairs change:

        >>> repairs = plant.mttf_mttr_allocation(0.98, levers="mttr")
        >>> {node: round(t, 2) for node, t in repairs.mttr.items()}
        {'p1': 0.74, 'p2': 0.74, 'v': 0.78}
        """
        if levers not in ("both", "mttf", "mttr"):
            raise ValueError(
                f"levers must be 'both', 'mttf' or 'mttr', got {levers!r}."
            )
        target = _allocation_target(target)
        current, free, held = self._allocatable(fixed)
        most = self._allocation_option("max_mttf", max_mttf, held)
        least = self._allocation_option("min_mttr", min_mttr, held)
        ease = (
            self._allocation_option(
                "mttf_feasibility", mttf_feasibility, held
            ),
            self._allocation_option(
                "mttr_feasibility", mttr_feasibility, held
            ),
        )
        names = ("mttf_feasibility", "mttr_feasibility")
        # Each component's failure rate and repair time, as (start, floor):
        # each falls from its start towards its floor as its lever is used,
        # to floor + (start - floor) * exp(-v) for the lever's v >= 0.
        span: Dict[Hashable, Tuple[Tuple[float, float], ...]] = {}
        variables: List[Tuple[Hashable, int, float]] = []
        for node, (mttf, mttr) in free.items():
            top = _as_float(most.get(node, math.inf))
            if not top >= mttf * (1.0 - 1e-12):
                raise ValueError(
                    f"max_mttf[{node!r}] must be at least the component's "
                    f"MTTF, {mttf:.6g}, got {most[node]!r}."
                )
            bottom = _as_float(least.get(node, 0.0))
            if not 0.0 <= bottom <= mttr * (1.0 + 1e-12):
                raise ValueError(
                    f"min_mttr[{node!r}] must be between 0 and the "
                    f"component's MTTR, {mttr:.6g}, got {least[node]!r}."
                )
            span[node] = (
                (1.0 / mttf, 1.0 / max(top, mttf)),
                (mttr, min(bottom, mttr)),
            )
            for lever, name in enumerate(names):
                f = _as_float(ease[lever].get(node, 0.5))
                if not 0.0 <= f < 1.0:
                    raise ValueError(
                        f"{name}[{node!r}] must be in [0, 1), got "
                        f"{ease[lever][node]!r}."
                    )
                start, floor = span[node][lever]
                if levers in ("both", ("mttf", "mttr")[lever]) and (
                    start > floor
                ):
                    variables.append((node, lever, 1.0 - f))
        if not variables:
            raise ValueError(
                "No MTTF or MTTR can change: every lever is held (by levers, "
                "max_mttf or min_mttr)."
            )
        steepness = np.array([s for _, _, s in variables])
        size = len(variables)

        def design(v) -> Dict[Hashable, List[float]]:
            """Each component's failure rate and repair time."""
            values = {
                node: [ends[0][0], ends[1][0]] for node, ends in span.items()
            }
            for (node, lever, _), x in zip(variables, v):
                if x > 0.0:
                    start, floor = span[node][lever]
                    gap = (start - floor) * math.exp(-x)
                    values[node][lever] = floor + gap
            return values

        def probabilities(v) -> Tuple[Dict, Dict]:
            p = {node: current[node] for node in self.nodes}
            q = {node: 1.0 - current[node] for node in self.nodes}
            for node, (rate, repair) in design(v).items():
                odds = rate * repair  # of being down
                p[node] = 1.0 / (1.0 + odds)
                q[node] = odds / (1.0 + odds)
            return p, q

        def result(v) -> AvailabilityAllocation:
            values = design(v)
            mttf = {}
            for node, (rate, _) in values.items():
                if rate == span[node][0][0]:
                    mttf[node] = free[node][0]  # unchanged, exactly
                else:
                    mttf[node] = 1.0 / rate if rate > 0.0 else math.inf
            return self._allocation_result(
                view,
                probabilities(v)[0],
                mttf,
                {node: repair for node, (_, repair) in values.items()},
            )

        view = self._allocation_view(held)
        goal = float(logit(target))
        now = view._log_odds(*probabilities(np.zeros(size)))[0]
        if goal <= now:
            self.res = OptimizeResult(
                x=np.zeros(size),
                fun=float(np.log(size)),
                success=True,
                message="The target is already met.",
            )
            return result(np.zeros(size))
        best = view._log_odds(*probabilities(np.full(size, np.inf)))[0]
        # The limits are only approached (at an ever-growing cost), so a
        # target at them, to within rounding, cannot be met either.
        if goal >= best - 1e-9:
            raise ValueError(
                f"target {target} cannot be reached: from the current "
                f"availability ({expit(now):.6g}) the system can only "
                f"approach {expit(best):.6g}, with every MTTF and MTTR that "
                "may change at its limit."
            )

        def shortfall(v: np.ndarray) -> float:
            return view._log_odds(*probabilities(v))[0] - goal

        def shortfall_gradient(v: np.ndarray) -> np.ndarray:
            p, q = probabilities(v)
            derivative = view._log_odds(p, q)[1]
            values = design(v)
            gradient = np.empty(size)
            for i, ((node, lever, _), x) in enumerate(zip(variables, v)):
                start, floor = span[node][lever]
                # The lever lowers the odds of being down, rate * repair, at
                # the other quantity times its own fall; p = 1 / (1 + odds).
                other = values[node][1 - lever]
                fall = other * (start - floor) * math.exp(-x)
                gradient[i] = derivative[node] * p[node] ** 2 * fall
            return gradient

        def log_total_cost(v: np.ndarray) -> Tuple[float, np.ndarray]:
            costs = steepness * np.expm1(v)
            return float(logsumexp(costs)), softmax(
                costs
            ) * steepness * np.exp(v)

        def common_shift(start: np.ndarray, along: np.ndarray) -> np.ndarray:
            """The start moved up by the least common v, on the levers
            ``along`` marks, meeting the target (it exists, as the target is
            below what the limits allow)."""
            high = 1.0
            while shortfall(start + high * along) < 0.0 and high < 1e6:
                high *= 2.0
            return start + along * brentq(
                lambda d: shortfall(start + d * along), 0.0, high, xtol=1e-14
            )

        every = np.ones(size)
        res = minimize(
            log_total_cost,
            common_shift(np.zeros(size), every),
            jac=True,
            method="SLSQP",
            bounds=[(0.0, None)] * size,
            constraints=[
                {"type": "ineq", "fun": shortfall, "jac": shortfall_gradient}
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        # A lever the solver left within a hair of its bound is unused.
        v = np.where(res.x < 1e-9, 0.0, res.x)
        if shortfall(v) < 0.0:
            # Close the solver's last sliver of constraint tolerance, with
            # the levers in use.
            used = (v > 0.0).astype(float)
            v = common_shift(v, used if used.any() else every)
        res.x = v
        res.fun = log_total_cost(v)[0]
        self.res = res
        if not res.success:
            warnings.warn(
                "the cost minimisation stopped before converging "
                f"({res.message}); the design meets the target but may not "
                "be the cheapest.",
                stacklevel=2,
            )
        return result(v)

    def _allocatable(
        self, fixed
    ) -> Tuple[
        Dict[Hashable, float], Dict[Hashable, Tuple[float, float]], set
    ]:
        """For an availability allocation: every component's long-run
        availability; the MTTF and MTTR of those allocated one (with
        corrective repair alone: they fail and take time to repair, with no
        preventive or inspection schedule, and are not in ``fixed``); and
        the set of the others, which keep theirs."""
        self._require_unlimited_crews(*_ALLOCATION_CREWS)
        self._require_no_ccf("the allocations")
        held = set()
        if fixed is not None:
            held = set(fixed)
            unknown = [node for node in held if node not in self.components]
            if unknown:
                raise ValueError(
                    f"fixed names {sorted(unknown, key=str)}, which are not "
                    "components of this RBD."
                )
        current: Dict[Hashable, float] = {}
        free: Dict[Hashable, Tuple[float, float]] = {}
        for node, component in self.components.items():
            current[node] = self._node_availability(node)
            if (
                node in held
                or isinstance(component, RepairableRBD)
                or node in self._preventive
                or node in self._inspection
                or node in self._standby
            ):
                held.add(node)
                continue
            mttf = model_mean(component.reliability)
            mttr = model_mean(component.time_to_replace)
            if not (0.0 < mttf < math.inf and 0.0 < mttr < math.inf):
                # Never down, down for good in the end, or up for good.
                held.add(node)
                continue
            free[node] = (mttf, mttr)
            current[node] = mttf / (mttf + mttr)
        if not free:
            raise ValueError(
                "No component can be allocated an availability: only "
                "components with corrective repair alone (that fail and take "
                "time to repair, with no preventive or inspection schedule), "
                "not in fixed, can."
            )
        return current, free, held

    def _allocation_option(self, name: str, mapping, held: set) -> dict:
        """A per-component option of an availability allocation, checked to
        name only components allocated an availability."""
        values = self._node_overrides(name, mapping)
        kept = [node for node in values if node in held]
        if kept:
            raise ValueError(
                f"{name} names {sorted(kept, key=str)}, which keep their "
                "availability: only components with corrective repair alone, "
                "not in fixed, are allocated one."
            )
        return values

    def _allocation_view(self, held: set) -> "RepairableRBD":
        """This RBD as an availability allocation scores it. The components
        allocated an availability have a constant one, so when every held
        component does too the system's availability is the structure
        function of theirs, and the RBD itself will do. A held component
        that is inspected or block-replaced is up with a probability that
        varies over its schedule, together with the others on the calendar:
        the system's availability is then averaged over the long-run grid,
        as ``mean_availability`` averages it, by a view of the RBD whose
        ``_allocation_probability`` and ``_log_odds`` do so."""
        times, weights = self._long_run_grid()
        if len(times) == 1:
            return self
        profiles = self._availabilities_at(times)
        view = copy(self)
        view.__dict__["_allocation_calendar"] = (
            weights,
            {node: profiles[node] for node in held},
            [node for node in self.nodes if node not in held],
        )
        return view

    def _calendar_arrays(
        self, p: Dict, q: Optional[Dict]
    ) -> Tuple[Dict, Dict]:
        """Node probabilities ``p`` and their complements ``q`` (by default
        ``1 - p``) over the allocation calendar's grid, with the held
        components' own availabilities there."""
        weights, profiles, _ = self.__dict__["_allocation_calendar"]
        size = len(weights)
        works = {node: np.full(size, float(p[node])) for node in p}
        fails = {
            node: np.full(
                size, 1.0 - float(p[node]) if q is None else float(q[node])
            )
            for node in p
        }
        for node, profile in profiles.items():
            works[node] = profile
            fails[node] = 1.0 - profile
        for node in self.in_or_out:
            works[node] = np.ones(size)
            fails[node] = np.zeros(size)
        return works, fails

    def _calendar_means(self, works: Dict, fails: Dict) -> Tuple[float, float]:
        """The system's probabilities of working and of failing over the
        allocation calendar's grid, averaged over it."""
        weights = self.__dict__["_allocation_calendar"][0]
        size = len(weights)
        up, down = self._decomposition().probabilities(
            works, fails, shape=size
        )
        return (
            float(weights @ np.broadcast_to(up, (size,))),
            float(weights @ np.broadcast_to(down, (size,))),
        )

    def _allocation_probability(
        self, probabilities: Dict[Any, float]
    ) -> float:
        if "_allocation_calendar" not in self.__dict__:
            return super()._allocation_probability(probabilities)
        return self._calendar_means(
            *self._calendar_arrays(probabilities, None)
        )[0]

    def _log_odds(
        self, p: Dict[Any, float], q: Dict[Any, float]
    ) -> Tuple[float, Dict[Any, float]]:
        if "_allocation_calendar" not in self.__dict__:
            return super()._log_odds(p, q)
        weights, _, free = self.__dict__["_allocation_calendar"]
        works, fails = self._calendar_arrays(p, q)
        up, down = self._calendar_means(works, fails)
        with np.errstate(divide="ignore"):
            log_odds = float(np.log(up) - np.log(down))
        scale = up * down
        gradient = dict.fromkeys(self.nodes, 0.0)
        if scale:
            ones, zeros = np.ones(len(weights)), np.zeros(len(weights))
            for node in free:
                # The system's availability is linear in the node's, with
                # slope P(down | node down) - P(down | node up).
                if_up = self._calendar_means(
                    {**works, node: ones}, {**fails, node: zeros}
                )[1]
                if_down = self._calendar_means(
                    {**works, node: zeros}, {**fails, node: ones}
                )[1]
                gradient[node] = (if_down - if_up) / scale
        return log_odds, gradient

    def _minimum_effort_held(self, target: float, current: Dict, held: set):
        """Albert's minimum-effort allocation to the components not
        ``held``: in series, the held components' availabilities (averaged
        over their calendars) are a factor of the system's, and the others
        make up the rest."""
        self._require_series()
        limit = self._allocation_probability(
            {
                node: current[node] if node in held else 1.0
                for node in self.nodes
            }
        )
        if target > limit:
            raise ValueError(
                f"target {target} cannot be reached: the components that "
                "keep their availability hold the system to at most "
                f"{limit:.6g}."
            )
        raised = self.minimum_effort_allocation(
            target / limit if limit > 0.0 else 0.0,
            {
                node: 1.0 if node in held else current[node]
                for node in self.nodes
            },
        )
        return {
            node: current[node] if node in held else raised[node]
            for node in self.nodes
        }

    def _allocation_result(
        self, view: "RepairableRBD", allocated: Dict, mttf: Dict, mttr: Dict
    ) -> AvailabilityAllocation:
        """An availability allocation's result, scored as ``view`` scores
        it."""
        return AvailabilityAllocation(
            availability={
                node: float(allocated[node]) for node in self.components
            },
            mttf=mttf,
            mttr=mttr,
            system_availability=view._allocation_probability(
                {node: allocated[node] for node in self.nodes}
            ),
        )

    def _with_intervals(
        self, preventive=None, inspection=None
    ) -> "RepairableRBD":
        """This RBD with the intervals of some of its preventive or
        inspection schedules changed, for the exact long-run values: a
        shallow copy, sharing everything else, including the renewal cycles
        already worked out (kept by interval)."""
        self.__dict__.setdefault("_age_cycles", {})
        self.__dict__.setdefault("_block_cycles", {})
        plan = copy(self)
        if preventive:
            plan._preventive = dict(self._preventive)
            for node, interval in preventive.items():
                plan._preventive[node] = self._preventive[node]._replace(
                    interval=float(interval)
                )
        if inspection:
            plan._inspection = dict(self._inspection)
            for node, interval in inspection.items():
                schedule = self._inspection[node]
                # An offset keeps its share of the interval (a pair tested
                # half an interval apart stays so).
                share = schedule.offset / schedule.interval
                plan._inspection[node] = schedule._replace(
                    interval=float(interval), offset=share * float(interval)
                )
        return plan

    def _interval_targets(self, min_availability, max_cost_rate):
        """The checked targets of an interval choice."""
        if min_availability is not None and max_cost_rate is not None:
            raise ValueError(
                "Give at most one of min_availability and max_cost_rate."
            )
        if not self.has_costs:
            raise ValueError(
                "Nothing is priced, so no interval costs more than another: "
                "give the components costs (or a downtime_cost_rate)."
            )
        if min_availability is not None:
            if not 0.0 < float(min_availability) < 1.0:
                raise ValueError(
                    "min_availability must be a number in (0, 1), got "
                    f"{min_availability!r}."
                )
            return float(min_availability), None
        if max_cost_rate is not None:
            if not 0.0 < float(max_cost_rate) < math.inf:
                raise ValueError(
                    "max_cost_rate must be a positive number, got "
                    f"{max_cost_rate!r}."
                )
            return None, float(max_cost_rate)
        return None, None

    def optimal_replacement_intervals(
        self,
        nodes: Optional[Collection[Hashable]] = None,
        *,
        min_availability: Optional[float] = None,
        max_cost_rate: Optional[float] = None,
    ) -> MaintenancePlan:
        """Choose the age-replacement intervals for the system as a whole.

        ``NonRepairable.find_optimal_replacement`` chooses one unit's
        replacement age on its own. In a system the components should be
        chosen together: a unit whose failure stops the system is worth
        replacing sooner than one with a standby, and replacing a unit takes
        the system down if its replacement takes time and nothing covers
        for it. This chooses the age-replacement interval of each component
        in ``nodes`` for:

        - the lowest long-run cost rate (``expected_cost_rate``), by
          default;
        - the lowest cost rate that keeps the system's long-run availability
          (``mean_availability``) at least ``min_availability``;
        - or the highest availability within a cost rate of
          ``max_cost_rate``.

        The long-run values are exact (see ``expected_cost_rate``), so the
        choice is too, to the precision of the search: a gradient search
        (SLSQP) over the logarithms of the intervals, from several starting
        points, keeping the best. Each interval ranges from a thousandth to a
        thousand times the component's mean life; one found at the top of
        that range is compared with never replacing the component (an
        interval of ``inf``), which is taken if no worse. The cost rate is
        usually flat near its minimum, so intervals some way from the ones
        found cost almost the same.

        Parameters
        ----------
        nodes : Collection[Hashable], optional
            The components whose intervals to choose, each under age
            replacement (a ``"preventive"`` schedule with ``"policy":
            "age"``; its interval is one of the starting points). By default
            every component under age replacement. The others keep their
            schedules.
        min_availability : float, optional
            The least long-run system availability allowed, in (0, 1).
        max_cost_rate : float, optional
            The highest long-run cost rate allowed: the intervals then give
            the highest availability within it.

        Returns
        -------
        MaintenancePlan
            The interval of each component in ``nodes`` (``inf`` for never),
            and the system's cost rate and availability with them.

        Raises
        ------
        ValueError
            If a node is not a component under age replacement; if nothing
            is priced; if both targets are given, or one is out of range;
            or if no intervals meet the target (the message gives the best
            they can do).
        NotImplementedError
            If the long-run values are not known exactly (see
            ``expected_cost_rate``).

        Examples
        --------
        A pump that wears out, in series with a pair of them in parallel;
        repairs take about 23 hours and replacements about 7, and the plant
        loses 500 an hour while it is down:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def pump():
        ...     return {
        ...         "reliability": surv.Weibull.from_params([1000, 2.5]),
        ...         "repairability": surv.LogNormal.from_params([3.0, 0.5]),
        ...         "replace_cost": 5000.0,
        ...         "preventive": {
        ...             "interval": 1000.0,
        ...             "duration": surv.Weibull.from_params([8, 3]),
        ...             "cost": 1000.0,
        ...         },
        ...     }
        >>> plant = RepairableRBD(
        ...     [("s", "a"), ("a", "b1"), ("a", "b2"), ("b1", "t"),
        ...      ("b2", "t")],
        ...     {"a": pump(), "b1": pump(), "b2": pump()},
        ...     downtime_cost_rate=500.0,
        ... )
        >>> plan = plant.optimal_replacement_intervals()
        >>> {node: round(t) for node, t in plan.intervals.items()}
        {'a': 590, 'b1': 497, 'b2': 497}
        >>> round(plan.cost_rate, 2), round(plan.availability, 4)
        (20.12, 0.9803)

        The pump alone in the line is replaced later than those with a
        standby: its replacements stop the plant too.
        """
        chosen = self._maintained(nodes)
        min_availability, max_cost_rate = self._interval_targets(
            min_availability, max_cost_rate
        )
        scale = {}
        for node in chosen:
            life = failure_time_scale(self.components[node].reliability)
            scale[node] = life if np.isfinite(life) and life > 0.0 else 1.0
        low = np.array([math.log(1e-3 * scale[n]) for n in chosen])
        high = np.array([math.log(1e3 * scale[n]) for n in chosen])
        current = [
            math.log(min(self._preventive[n].interval, 1e3 * scale[n]))
            for n in chosen
        ]
        starts = [
            np.clip(current, low, high),
            np.array([math.log(scale[n]) for n in chosen]),
            np.array([math.log(0.3 * scale[n]) for n in chosen]),
            high.copy(),
        ]

        def evaluate(intervals: dict) -> Tuple[float, float]:
            plan = self._with_intervals(preventive=intervals)
            return plan.expected_cost_rate(), plan.mean_availability()

        best = _choose_intervals(
            chosen,
            evaluate,
            starts,
            (low, high),
            min_availability,
            max_cost_rate,
        )
        # An interval at the top of its range may be better never used.
        for i, node in enumerate(chosen):
            if best[node] >= 0.99 * math.exp(high[i]):
                never = {**best, node: math.inf}
                cost, availability = evaluate(never)
                if _meets(
                    cost, availability, min_availability, max_cost_rate
                ) and _objective(
                    cost, availability, max_cost_rate
                ) <= _objective(
                    *evaluate(best), max_cost_rate
                ):
                    best = never
        cost, availability = evaluate(best)
        return MaintenancePlan(
            {node: float(best[node]) for node in chosen}, cost, availability
        )

    def optimal_inspection_intervals(
        self,
        nodes: Optional[Collection[Hashable]] = None,
        *,
        allowed=None,
        min_availability: Optional[float] = None,
        max_cost_rate: Optional[float] = None,
    ) -> MaintenancePlan:
        """Choose the proof-test intervals of components with hidden
        failures, for the system as a whole.

        Testing a component with hidden failures more often costs more
        tests, and testing it less often leaves its failures hidden for
        longer (see ``node_availability``). This chooses the inspection
        interval of each component in ``nodes`` for:

        - the lowest long-run cost rate (``expected_cost_rate``: tests,
          repairs and downtime), by default;
        - the lowest cost rate that keeps the system's long-run availability
          at least ``min_availability``: for a safety function, whose
          unavailability is its average probability of failure on demand,
          a PFDavg of at most ``1 - min_availability``;
        - or the highest availability within a cost rate of
          ``max_cost_rate``.

        Tests are usually made on a calendar (monthly, quarterly, yearly),
        and components tested at the same times are down together, so the
        long-run values depend on how the schedules line up. So the
        intervals are chosen from ``allowed``: every combination is tried
        when there are at most 2000 of them, which gives the optimum, and a
        local search (changing one interval at a time, from several
        starting points) is made otherwise. With ``allowed`` left out, the
        RBD must have one component with hidden failures, whose interval
        is then searched continuously, as by
        ``optimal_replacement_intervals``.

        The exact long-run values need a constant failure rate, instant
        tests and instant repair (see ``node_availability``).

        Parameters
        ----------
        nodes : Collection[Hashable], optional
            The components whose intervals to choose, each with hidden
            failures (an ``"inspection"`` schedule). By default every
            component with one. The others keep their intervals.
        allowed : sequence of float or dict, optional
            The intervals to choose from: one sequence for every node, or a
            dict of a sequence per node. Left out, the one component with
            hidden failures has its interval chosen from all.
        min_availability : float, optional
            The least long-run system availability allowed, in (0, 1).
        max_cost_rate : float, optional
            The highest long-run cost rate allowed: the intervals then give
            the highest availability within it.

        Returns
        -------
        MaintenancePlan
            The interval of each component in ``nodes``, and the system's
            cost rate and availability (1 - PFDavg) with them.

        Raises
        ------
        ValueError
            If a node has no hidden failures; if ``allowed`` is left out
            with more than one component with hidden failures, or holds
            something other than positive, finite intervals; if nothing is
            priced; if both targets are given, or one is out of range; or if
            no intervals meet the target (the message gives the best they
            can do).
        NotImplementedError
            If a component's hidden failures have no exact long-run values
            (see ``node_availability``), or the intervals repeat together
            only after too many tests to average over.

        Examples
        --------
        A shutdown valve whose dangerous failures are hidden, at ``2e-6``
        per hour, each proof test costing 500: the cheapest monthly,
        quarterly, half-yearly, yearly or two-yearly tests (in hours) that
        keep the PFDavg at most ``1e-3``, for one valve and for two in
        parallel (1oo2):

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def valve():
        ...     return {
        ...         "reliability": surv.Exponential.from_params([2e-6]),
        ...         "repairability": "instant",
        ...         "inspection": {"interval": 8760.0, "cost": 500.0},
        ...     }
        >>> calendar = [730.0, 2190.0, 4380.0, 8760.0, 17520.0]
        >>> one = RepairableRBD([("s", "v"), ("v", "t")], {"v": valve()})
        >>> one.optimal_inspection_intervals(
        ...     allowed=calendar, min_availability=1 - 1e-3
        ... ).intervals
        {'v': 730.0}
        >>> pair = RepairableRBD(
        ...     [("s", "v1"), ("s", "v2"), ("v1", "t"), ("v2", "t")],
        ...     {"v1": valve(), "v2": valve()},
        ... )
        >>> plan = pair.optimal_inspection_intervals(
        ...     allowed=calendar, min_availability=1 - 1e-3
        ... )
        >>> plan.intervals
        {'v1': 17520.0, 'v2': 17520.0}
        >>> round(1 - plan.availability, 6)
        0.000399

        The redundant pair meets the target with tests every two years; the
        single valve needs them monthly.
        """
        chosen = self._inspected(nodes)
        min_availability, max_cost_rate = self._interval_targets(
            min_availability, max_cost_rate
        )
        rates = {node: self._tested_scale(node) for node in chosen}

        def evaluate(intervals: dict) -> Tuple[float, float]:
            plan = self._with_intervals(inspection=intervals)
            return plan.expected_cost_rate(), plan.mean_availability()

        if allowed is None:
            self._require_one_inspected()
            (node,) = chosen
            rate = rates[node]
            low = np.array([math.log(1e-4 / rate)])
            high = np.array([math.log(10.0 / rate)])
            current = math.log(self._inspection[node].interval)
            starts = [
                np.clip([current], low, high),
                np.array([math.log(0.01 / rate)]),
                np.array([math.log(0.1 / rate)]),
                np.array([math.log(1.0 / rate)]),
            ]
            best = _choose_intervals(
                chosen,
                evaluate,
                starts,
                (low, high),
                min_availability,
                max_cost_rate,
            )
        else:
            best = _choose_from(
                chosen,
                self._allowed_intervals(allowed, chosen),
                evaluate,
                min_availability,
                max_cost_rate,
            )
        cost, availability = evaluate(best)
        return MaintenancePlan(
            {node: float(best[node]) for node in chosen}, cost, availability
        )

    def _inspected(self, nodes) -> list:
        """The components with hidden failures named by ``nodes`` (all of
        them by default), checked."""
        if nodes is None:
            if not self._inspection:
                raise ValueError(
                    "No component has hidden failures: give the components "
                    "an 'inspection' schedule to have its interval chosen."
                )
            chosen = list(self._inspection)
        else:
            chosen = list(nodes)
            if not chosen:
                raise ValueError("nodes is empty.")
            for node in chosen:
                if node not in self._inspection:
                    raise ValueError(
                        f"Node {node!r} has no hidden failures: give it an "
                        "'inspection' schedule to have its interval chosen."
                    )
        for node in chosen:
            if self._inspection[node].partial:
                raise NotImplementedError(
                    f"Component {node!r}'s tests can miss a failure (a "
                    "coverage below 1), and its interval must divide its "
                    "full tests' interval, so it is not chosen here: "
                    "compare the intervals that do with mean_availability "
                    "and expected_cost_rate."
                )
        return chosen

    @staticmethod
    def _allowed_intervals(allowed, chosen: list) -> dict:
        """``allowed`` as a sorted tuple of intervals per chosen node."""
        per_node = (
            allowed
            if isinstance(allowed, dict)
            else {node: allowed for node in chosen}
        )
        options = {}
        for node in chosen:
            if node not in per_node:
                raise ValueError(
                    f"allowed gives no intervals for node {node!r}."
                )
            values = []
            for value in per_node[node]:
                try:
                    interval = float(value)
                except (TypeError, ValueError):
                    interval = float("nan")
                if not 0.0 < interval < math.inf:
                    raise ValueError(
                        "allowed intervals must be positive, finite "
                        f"numbers, got {value!r} for node {node!r}."
                    )
                values.append(interval)
            if not values:
                raise ValueError(
                    f"allowed gives no intervals for node {node!r}."
                )
            options[node] = tuple(sorted(set(values)))
        return options

    def _maintained(self, nodes) -> list:
        """The components under age replacement named by ``nodes`` (all of
        them by default), checked."""
        if nodes is None:
            chosen = [
                node
                for node, schedule in self._preventive.items()
                if schedule.policy == "age"
            ]
            if not chosen:
                raise ValueError(
                    "No component is under age replacement: give the "
                    "components a 'preventive' schedule to have its "
                    "interval chosen."
                )
            return chosen
        chosen = list(nodes)
        if not chosen:
            raise ValueError("nodes is empty.")
        for node in chosen:
            schedule = self._preventive.get(node)
            if schedule is None or schedule.policy != "age":
                raise ValueError(
                    f"Node {node!r} is not a component under age "
                    "replacement: give it a 'preventive' schedule with "
                    "'policy': 'age' to have its interval chosen."
                )
        return chosen

    def initialize_event_queue(
        self,
        t_simulation,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        sources: Optional[dict] = None,
        state=None,
    ):
        """Start one simulation of the system over ``[0, t_simulation)``.

        Advanced, event-stepping API: ``availability`` runs its simulations
        through it, and a parent RBD uses it to drive a nested
        ``RepairableRBD``, which therefore has the same event API as a
        ``NonRepairable`` component. Every component starts working and new
        (or from its ``state``) except the ``broken_nodes``, which stay down
        for the whole window; the ``working_nodes`` never fail. The first
        failure of each other component is queued (a nested RBD starts its
        own simulation and queues its first state change), and events at or
        after ``t_simulation`` are dropped. Then call ``next_event``
        repeatedly to step through the system's state changes; see it for
        an example.

        Unlike ``availability``, this takes no seed and does not check the
        working/broken nodes: the draws come from numpy's global RNG, so
        seed that (``np.random.seed``) for a reproducible run. The state is
        kept on the RBD itself (``system_state``, ``component_status``,
        ``t_simulation``, ``last_change_planned``), and ``availability``
        uses and then deletes the same state, so do not interleave the two.

        Parameters
        ----------
        t_simulation : float
            The end of the simulated window.
        working_nodes : Collection[Hashable], optional
            Nodes held working for the whole window, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed for the whole window, by default None.
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``,
            the default) or the minimal cut sets (``"c"``); both give the
            same state.
        sources : dict, optional
            Internal: node name -> the object each component's events are
            drawn from, which ``availability`` passes so that the
            components draw from random streams of their own (see
            ``_streams``). By default None: the components themselves.
        state : dict, optional
            Start the components it names from their states rather than new,
            as for ``availability``: ``{node: NodeState}``, and a nested
            RBD's own such dict for a nested RBD. By default None: every
            component new.

        Raises
        ------
        ValueError
            If ``method`` is not ``"p"`` or ``"c"``, or ``state`` is not
            one the components can be in.
        NotImplementedError
            If ``state`` is ``"stationary"``, or gives a state to a
            component whose state is not taken (see ``availability``).
        """
        self._require_no_ccf("the simulation", nested=True)
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        states = self._simulation_states(state, working_nodes | broken_nodes)
        self._start_queue(
            t_simulation, working_nodes, broken_nodes, method, sources, states
        )

    def _start_queue(
        self,
        t_simulation,
        working_nodes: set,
        broken_nodes: set,
        method: str,
        sources: Optional[dict],
        states: dict,
    ) -> None:
        """``initialize_event_queue``, with ``states`` checked (see
        ``_simulation_states``)."""
        # What the components draw their events from: themselves (an
        # imperfectly repaired one through a stand-in keeping its virtual
        # age, and one started from a state through one whose next draw can
        # be set; kept for next_event), or stand-ins drawing from their own
        # streams (see _streamed_components), which the caller passes to
        # next_event too.
        if sources is None:
            sources = self._step_sources = self._own_sources(states)
        else:
            self.__dict__.pop("_step_sources", None)
        # The window's end, which the first events already look to (see
        # _replaced_at).
        self.t_simulation = t_simulation
        # Each calendar a state's phase shifts: by node, the time since its
        # last scheduled replacement or test (see _due).
        self._phases: dict = {
            node: start.phase
            for node, start in states.items()
            if isinstance(start, NodeState)
            and (
                start.phase
                or (
                    # On a calendar from 0, not at its offset.
                    start.phase is not None
                    and node in self._inspection
                    and self._inspection[node].offset
                )
            )
        }

        # Keep record of component status', initially they're all working
        component_status: dict[Any, bool] = {
            component: True for component in self.components.keys()
        }

        for component in broken_nodes:
            component_status[component] = False

        # The queue supplies failure/repair events in chronological order
        event_queue = _EventQueue()
        # When each working node with hidden failures is due to fail (None
        # once it has failed).
        self._pending_failure: dict[Any, Optional[float]] = {}
        # When each working node replaced on condition is due to fail, and
        # to be replaced (see _replaced_at).
        self._in_service: dict[Any, Tuple[float, float]] = {}
        # Opportunistic maintenance (#108): when each member that can be
        # renewed early was put into service as new, and its pending event;
        # the pending events early renewals have cancelled, and the early
        # renewals queued, by identity, and their members (see _stop).
        self._renewed_at: dict[Any, float] = {}
        self._pending_event: dict[Any, Event] = {}
        self._cancelled: dict[int, Event] = {}
        self._early: dict[int, Event] = {}
        self._renewing: set = set()
        # The components down at the start in a repair or maintenance going
        # on, which holds a repair crew.
        in_hand: list = []

        # For each component add in the initial failure
        for component_id in self.components.keys():
            component = self.components[component_id]
            if component_id in working_nodes:
                continue
            elif component_id in broken_nodes:
                continue
            elif component_id in self._standby:
                continue  # started below, once the queue and crews exist
            source = sources[component_id]
            start = states.get(component_id)
            if isinstance(component, RepairableRBD):
                source.initialize_event_queue(t_simulation, state=start)
                component_status[component_id] = source.system_state
                t_event, event = source.next_event()
                first = Event(
                    t_event, component_id, event, _planned(source, event)
                )
            elif start is not None and not start.new:
                up, first = self._started(component_id, source, start)
                component_status[component_id] = up
                if not up and first.status:
                    in_hand.append(component_id)
            elif component_id in self._preventive:
                # Put into service as new at 0: it fails, or is maintained.
                source.reset()
                first = self._renewal(
                    component_id, 0.0, source, self._preventive[component_id]
                )
            elif component_id in self._inspection:
                # Put into service as new at 0: it fails, or is inspected.
                source.reset()
                first = self._inspected_renewal(
                    component_id, 0.0, source, self._inspection[component_id]
                )
            else:
                source.reset()
                t_event, event = source.next_event()
                first = Event(t_event, component_id, event)

            # Only consider it if it occurs within the simulation window
            if first.time < t_simulation:
                event_queue.put(first)
        self._event_queue = event_queue
        # The repair crews, when there are fewer than the components that
        # may need one (with enough, no job ever waits).
        crews = self.repair_crews
        self._crews = (
            _Crews(crews, self._crew_served(), self._priority)
            if crews is not None and self._crews_limited()
            else None
        )
        if self._crews is not None:
            # A repair or maintenance going on at the start holds a crew
            # (_simulation_states has checked that there are enough).
            for node in in_hand:
                if node in self._crews.served:
                    self._crews.free -= 1
                    self._crews.holding.add(node)
        # Each standby group's units, which queue the group's first event.
        self._groups: dict[Any, _StandbyGroup] = {
            node: _StandbyGroup(
                self, node, arrangement, sources[node], t_simulation
            )
            for node, arrangement in self._standby.items()
            if node not in working_nodes and node not in broken_nodes
        }
        self.last_change_planned = False
        # The initial system state must reflect any forced-broken components
        # (e.g. a broken component in series starts the system down), and
        # any component that starts down, rather than assuming everything is
        # up.
        self.system_state = self.is_system_working(component_status, method)
        self.component_status = component_status

    def _started(self, node, source, start: NodeState) -> Tuple[bool, Event]:
        """Whether component ``node`` is up at 0, started from ``start``
        (not new), and its first event.

        One uniform, from the component's own stream (see
        ``_streamed_components``), draws what is left of its repair or
        maintenance given how long it has taken so far, or of its life
        given its age (``_aged_life``: ``S(a + x) / S(a)`` is the chance
        of more than ``x`` left). Up at age 0 it is new, and draws its life
        as a new unit does. Its calendar is shifted by the state's phase
        (see ``_due``); under age replacement, its replacement is due when
        it reaches the age, at once if it has. A unit with hidden failures
        is known to have been up only at its last test (or when put into
        service, if since): its life left is drawn from then, and if it
        has run out it failed unseen, and the next test finds it. After
        the first, its units are new, as from new.
        """
        component = self.components[node]
        schedule = self._preventive.get(node)
        inspection = self._inspection.get(node)
        source.reset()
        if not start.alive:
            # Down: up again, as new, once its repair or maintenance is
            # done.
            model = (
                schedule.duration  # type: ignore[union-attr]
                if start.maintenance
                else component.time_to_replace
            )
            left = _aged_life(model, start.down_for, source.start_uniform())
            if inspection is not None:
                self._pending_failure[node] = None
            return False, Event(left, node, True, start.maintenance)
        if not start.age:
            # New at 0, on a calendar at its phase.
            if schedule is not None:
                return True, self._renewal(node, 0.0, source, schedule)
            if inspection is not None:
                return True, self._inspected_renewal(
                    node, 0.0, source, inspection
                )
            delay, status = source.next_event()
            return True, Event(delay, node, status)
        since = 0.0
        if inspection is not None:
            # Last known up at its last test, or when put into service.
            since = min(start.age, start.phase or 0.0)
        life = (
            _aged_life(
                component.reliability,
                start.age - since,
                source.start_uniform(),
            )
            - since
        )
        source.life_drawn()
        if inspection is not None:
            if life < 0.0:
                # Failed, unseen, since it was last known up.
                self._pending_failure[node] = None
                found = self._finds(node, inspection, life)
                return False, Event(found, node, False, inspection=True)
            self._pending_failure[node] = life
            return True, self._inspected_next(node, 0.0, inspection)
        if schedule is None:
            return True, Event(life, node, False)
        return True, self._renewal(
            node, 0.0, source, schedule, life=life, start=-start.age
        )

    def _due(self, node, schedule, t: float) -> float:
        """The next scheduled action of ``schedule`` (a ``_Preventive`` or
        ``_Inspection``) after ``t``, on ``node``'s calendar: shifted, from
        a state, by its phase, so that the last action fell that long
        before 0."""
        phase = self._phases.get(node) if self._phases else None
        if phase is None:
            return schedule.due(t)
        if isinstance(schedule, _Inspection):
            # The phase places the calendar: no offset.
            schedule = schedule.without_offset()
        due = schedule.due(t + phase) - phase
        return due if due > t else due + schedule.interval

    def _finds(self, node, inspection: _Inspection, t: float) -> float:
        """The test that finds a failure at ``t``, on ``node``'s calendar
        (see ``_due``): the first at or after it, and not before 0."""
        phase = self._phases.get(node) if self._phases else None
        if phase is None:
            return inspection.finds(t)
        inspection = inspection.without_offset()
        found = inspection.finds(t + phase) - phase
        if found < t or found < 0.0:
            found += inspection.interval
        return found

    def mean_unavailability(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> float:
        """Returns the system's long-run (steady-state) unavailability.

        ``1 - mean_availability()``: the long-run fraction of time the
        system is down, exact as
        [`mean_availability`][repyability.RepairableRBD.mean_availability]
        is, and over the same cases. It is worked out in its own right,
        from each component's unavailability (``MTTR / (MTTF + MTTR)``,
        say, or ``1 - exp(-lambda * u)`` at a time ``u`` since a test), as
        a sum of products over the structure, so that a small one keeps its
        full relative precision: a PFDavg of ``1e-12`` gets ``1e-12``,
        where one less the availability would keep only about four digits
        of it.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working (unavailability 0), by
            default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed (unavailability 1), by
            default None.
        method : str, optional
            ``"p"`` or ``"c"``, as for ``mean_availability``; both give the
            same result, computed the same way.

        Returns
        -------
        float
            Long-run unavailability of the system, in ``[0, 1]``.

        Raises
        ------
        ValueError
            As for ``mean_availability``.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        Two components in parallel, each down a millionth of the time, are
        down together a millionth of a millionth of it:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> unit = {
        ...     "reliability": surv.Exponential.from_params([1e-6]),
        ...     "repairability": surv.ExactEventTime.from_params([1.0]),
        ... }
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {"a": unit, "b": unit},
        ... )
        >>> f"{rbd.mean_unavailability():.6e}"
        '9.999980e-13'
        """
        if method not in ("p", "c"):
            raise ValueError("`method` must be either 'p' or 'c'")
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        availability, unavailability, weights = (
            self._long_run_unavailabilities(working_nodes, broken_nodes)
        )
        system = self._system_unreliability(availability, unavailability)
        return float(weights @ system)

    def analysis_routes(self) -> Dict[str, "AnalysisRoute"]:
        """How each analysis of this RBD is computed, found without running
        it: exactly, numerically, by simulation, or not at all.

        The route follows from the method, then from the components: the
        long-run values are exact from each component's long-run
        availability, which is a closed form, or numerical for a component
        under preventive maintenance, or refused for a case they do not
        cover (hidden failures at a rate that is not constant, say); the
        availability over time solves each component's renewal equation on
        a grid; ``availability``, ``cost`` and ``compare`` always simulate.
        A component whose life is simulated (a ``StandbyModel`` fitted to
        simulated lifetimes) makes the exact values it enters simulated
        too. A refusal is found by the check the method itself runs, and its
        reason is the message it would raise; limits that only computing
        shows (a grid grown too large) are not foreseen.

        A method that takes nodes, targets or intervals is described for
        its defaults: ``optimal_replacement_intervals`` choosing for every
        component under age replacement, say.

        For a simulation, ``engine`` is the engine ``engine="auto"`` runs a
        long simulation on: when the compiled engine simulates the system,
        ``"numba"`` if numba is installed (or the name of an engine another
        package adds, see ``repyability.rbd.engines``), else ``"python"``,
        with the reason in ``engine_reason``.

        Returns
        -------
        dict[str, AnalysisRoute]
            For each public analysis, by method name: its ``route``
            (``"exact"``, ``"numerical"``, ``"simulated"`` or
            ``"refused"``), the ``reason``, and the ``nodes`` that decide it
            (see [`AnalysisRoute`][repyability.AnalysisRoute]).

        Examples
        --------
        A pump found failed only by monthly tests that take it off-line
        for about an hour: the exact values need tests in no time, so they
        are refused, and the simulation is the way:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "pump"), ("pump", "t")],
        ...     {
        ...         "pump": {
        ...             "reliability": surv.Weibull.from_params([500, 1.5]),
        ...             "repairability": "instant",
        ...             "inspection": {
        ...                 "interval": 720,
        ...                 "duration": surv.Exponential.from_params([1.0]),
        ...             },
        ...         }
        ...     },
        ... )
        >>> routes = rbd.analysis_routes()
        >>> routes["mean_availability"].route
        'refused'
        >>> routes["mean_availability"].nodes
        ('pump',)
        >>> routes["availability"].route
        'simulated'
        """
        from repyability.rbd import routes as r

        out: Dict[str, r.AnalysisRoute] = {}

        def give(names, route) -> None:
            for name in names:
                out[name] = route

        long_run = self._long_run_route()

        long_run_nodes = self._long_run_nodes()

        def from_long_run(route, reason):
            """A method of ``route``, explained by ``reason``, built on the
            long-run values: refused as they are, and numerical or simulated
            as the components' are."""
            if long_run.route == r.REFUSED:
                return long_run
            return r.with_nodes(
                route, reason, long_run_nodes, "long-run values"
            )

        give(("mean_availability", "mean_unavailability"), long_run)
        out["node_availability"] = (
            self._long_run_route(groups=False) if self.ccf_groups else long_run
        )
        frequencies = (
            None
            if long_run.route == r.REFUSED
            else r.refusal(self._require_ccf_frequencies)
        )
        give(
            (
                "system_failure_frequency",
                "mean_time_between_failures",
                "mean_up_time",
                "mean_down_time",
            ),
            r.refused(frequencies) if frequencies else long_run,
        )
        # The allocations assume independent components, which limited
        # repair crews make them not, and common-cause groups too.
        allocation = r.refusal(
            partial(self._require_unlimited_crews, *_ALLOCATION_CREWS)
        ) or r.refusal(partial(self._require_no_ccf, "the allocations"))
        conditioned = long_run
        if self.ccf_groups and long_run.route != r.REFUSED:
            conditioned = dataclasses.replace(
                long_run,
                reason=long_run.reason
                + " A common-cause group's member is conditioned on its "
                "state at each time, as the shared causes tie the other "
                "members' to it.",
            )
        if self._crews_couple() and long_run.route != r.REFUSED:
            conditioned = dataclasses.replace(
                long_run,
                reason=long_run.reason
                + " With the crews, Birnbaum's measure, the improvement "
                "potential, RAW and RRW hold each node working and failed "
                "in the chain, solved without it; the criticality and "
                "Fussell-Vesely measures are probabilities over its "
                "states.",
            )
        give(
            (
                "birnbaum_importance",
                "improvement_potential",
                "risk_achievement_worth",
                "risk_reduction_worth",
                "criticality_importance",
            ),
            conditioned,
        )
        give(
            ("fussell_vesely", "fussel_vesely"),
            from_long_run(
                r.EXACT,
                "The exact probability that a minimal cut set containing "
                "each node is down, over the system's unavailability, from "
                "the exact long-run values (method='rare_event' sums the cut "
                "sets' probabilities instead).",
            ),
        )
        if self.has_costs:
            setups = r.refusal(self._require_separate_setups)
            give(
                ("expected_cost_rate", "total_cost"),
                (
                    r.refused(setups)
                    if setups
                    else from_long_run(
                        r.EXACT,
                        "The long-run cost rate, from the exact long-run "
                        "values (total_cost multiplies it by the time).",
                    )
                ),
            )
        else:
            give(
                ("expected_cost_rate", "total_cost"),
                r.AnalysisRoute(
                    r.EXACT,
                    "No running cost is priced, so the cost rate is 0 "
                    "(total_cost adds any acquisition costs).",
                ),
            )
        capacity = (
            None if long_run.route == r.REFUSED else self._capacity_refusal()
        )
        out["capacity_distribution"] = (
            r.refused(*capacity)
            if capacity
            else from_long_run(
                r.EXACT,
                "The exact long-run distribution of the system's capacity, "
                + (
                    "averaged over the states of the repair crews' Markov "
                    "chain."
                    if self._crews_couple()
                    else "from the components' long-run availabilities."
                ),
            )
        )
        no_capacity = r.refusal(self._require_capacity)
        out["system_capacity"] = (
            r.refused(no_capacity)
            if no_capacity
            else r.AnalysisRoute(
                r.EXACT,
                "The exact distribution of the system's capacity from the "
                "node probabilities given.",
            )
        )
        give(("point_availability", "mission_availability"), self._over_time())
        over_time_capacity = self._capacity_refusal()
        give(
            ("point_capacity", "mission_capacity"),
            (
                r.refused(*over_time_capacity)
                if over_time_capacity
                else self._over_time(
                    "Each component's renewal equation solved on a grid (to "
                    "about 1e-7), a degrading component's stages through its "
                    "renewals too, and the system's capacity distribution "
                    "exactly at its components' at each time.",
                    kind="capacity",
                )
            ),
        )
        window = self._over_time(
            "Each component's expected events from its renewal equation, "
            "solved on a grid (to about 1e-7), and the system's failures by "
            "the time-dependent Birnbaum/Vesely formula.",
            kind="window",
        )
        give(("expected_failures", "expected_events"), window)
        out["expected_cost"] = (
            window
            if self.has_costs
            else r.AnalysisRoute(
                r.EXACT,
                "No running cost is priced, so the expected cost is 0 (with "
                "any acquisition costs beside it).",
            )
        )
        give(
            ("allocate_redundancy", "availability_allocation"),
            (
                r.refused(allocation)
                if allocation
                else from_long_run(
                    r.EXACT,
                    "A search over the exact long-run values of the "
                    "candidates.",
                )
            ),
        )
        out["mttf_mttr_allocation"] = (
            r.refused(allocation)
            if allocation
            else from_long_run(
                r.NUMERICAL,
                "A solver for the components' MTTF and MTTR targets, over the "
                "exact long-run values.",
            )
        )
        # The optimisers, called for every node they can choose for.
        targets = r.refusal(partial(self._interval_targets, None, None))
        refusal = r.refusal(partial(self._maintained, None)) or targets
        out["optimal_replacement_intervals"] = (
            r.refused(refusal)
            if refusal
            else from_long_run(
                r.NUMERICAL,
                "An optimiser over the age-replacement intervals, with the "
                "exact long-run values at each.",
            )
        )
        refusal = (
            r.refusal(partial(self._inspected, None))
            or targets
            or next(
                (
                    message
                    for message in (
                        r.refusal(partial(self._require_tested_exact, node))
                        for node in self._inspection
                    )
                    if message
                ),
                None,
            )
            or r.refusal(self._require_one_inspected)
        )
        out["optimal_inspection_intervals"] = (
            r.refused(refusal)
            if refusal
            else from_long_run(
                r.NUMERICAL,
                "An optimiser over the inspection interval, with the exact "
                "long-run values at each.",
            )
        )
        # Spares: each component's replacements, counted as a renewal
        # process, in the order the methods check them.
        crews = r.refusal(
            partial(self._require_unlimited_crews, *_SPARES_CREWS)
        )
        for name, long_run_count in (
            ("spares_demand", False),
            ("spares_stock", True),
        ):
            refusal = crews
            which: Tuple[Hashable, ...] = ()
            for node in self._spares_nodes(None):
                if refusal:
                    break
                refusal = r.refusal(
                    partial(self._replacements, node, long_run_count)
                )
                which = (node,)
            out[name] = (
                r.refused(refusal, which)
                if refusal
                else r.AnalysisRoute(
                    r.NUMERICAL,
                    "Each component's replacements, a renewal process, "
                    "counted on a grid (to about 1e-6): under block "
                    "replacement, from one block interval to the next; with "
                    "hidden failures, exactly, on its tests.",
                )
            )
        plan, streamed = self._stream_plan(1.0, 0, False)
        paired = (
            ""
            if streamed
            else " Antithetic pairs are refused: some components' draws "
            "do not come from a stream."
        )
        groups = r.refusal(
            partial(self._require_no_ccf, "the simulation", nested=True)
        )
        given = r.refusal(self._require_capacities_given)
        engine, why = self._engine_choice(capacity=self._has_capacity())
        out["availability"] = (
            r.refused(groups)
            if groups
            else (
                r.refused(given, tuple(self._capacity_models()))
                if given
                else r.AnalysisRoute(
                    r.SIMULATED,
                    "A discrete-event simulation of the components' failures, "
                    "repairs and maintenance." + paired,
                    engine=engine,
                    engine_reason=why,
                )
            )
        )
        out["simulate_chunk"] = out["availability"]
        unsaved = r.refusal(self._shard_system)
        out["shards"] = (
            out["availability"]
            if out["availability"].route == r.REFUSED
            else (
                r.refused(unsaved)
                if unsaved
                else r.AnalysisRoute(
                    r.SIMULATED,
                    "The run's simulations as shards, to simulate anywhere "
                    "with run_shard (see shards); availability_from_chunks "
                    "merges their partials into its result.",
                )
            )
        )
        out["availability_from_chunks"] = (
            out["availability"]
            if given or groups
            else r.AnalysisRoute(
                r.SIMULATED,
                "Simulated chunks of a run (see simulate_chunk), merged into "
                "its result.",
            )
        )
        engine, why = _timeline_runs.engine_choice(self, plan)
        out["simulate_timelines"] = (
            r.refused(groups)
            if groups
            else r.AnalysisRoute(
                r.SIMULATED,
                "The simulations availability runs, their histories kept "
                "as timelines: recorded by the event loop as it runs"
                + (
                    "; on the Python engine, each component's drawn "
                    "straight from its streams, and the system's merged "
                    "from theirs."
                    if _timeline_runs.independent(self, plan)
                    else "."
                )
                + paired,
                engine=engine,
                engine_reason=why,
            )
        )
        engine, why = self._engine_choice(capacity=False)
        out["cost"] = (
            r.refused(groups)
            if groups and self.has_costs
            else r.AnalysisRoute(
                r.SIMULATED,
                "A discrete-event simulation of the components' failures, "
                "repairs and maintenance, and what they cost." + paired,
                engine=engine,
                engine_reason=why,
            )
        )
        twin = self._twin_report()
        for name in ("availability", "cost"):
            if out[name].route == r.SIMULATED:
                out[name] = dataclasses.replace(out[name], twin=twin)
        out["compare"] = (
            r.refused(groups)
            if groups
            else (
                r.AnalysisRoute(
                    r.SIMULATED,
                    "The two systems simulated with common random numbers.",
                    engine=engine,
                    engine_reason=why,
                )
                if streamed
                else r.refused(_UNSTREAMED)
            )
        )
        give(
            ("initialize_event_queue", "next_event"),
            (
                r.refused(groups)
                if groups
                else r.AnalysisRoute(
                    r.SIMULATED,
                    "One simulation, stepped through event by event, drawing "
                    "from numpy's global RNG.",
                )
            ),
        )
        out["structural_importance"] = r.AnalysisRoute(
            r.EXACT, "From the structure alone."
        )
        out["system_probability"] = r.AnalysisRoute(
            r.EXACT,
            "The structure function over the node probabilities given.",
        )
        out["path_set_probabilities"] = r.AnalysisRoute(
            r.EXACT, "From the node probabilities given."
        )
        give(
            (
                "improvement_allocation",
                "equal_allocation",
                "simple_allocation",
                "minimum_effort_allocation",
                "cost_based_allocation",
            ),
            r.AnalysisRoute(
                r.NUMERICAL,
                "A solver over the exact system probability, from the node "
                "probabilities given (not the component models).",
            ),
        )
        # Each component's own values, and the long-run costs but the
        # system's downtime, need no structure; nor the expected cost when
        # nothing is priced.
        free = {"node_availability", "spares_demand", "spares_stock"}
        if not self.downtime_cost_rate:
            free |= {"expected_cost_rate", "total_cost"}
        if not self.has_costs:
            free.add("expected_cost")
        meshed = self._too_meshed()
        if meshed is not None and self._has_capacity():
            # The simulations that follow the system's capacity do so
            # through its reduced diagram, which a structure too meshed has
            # not.
            for name in (
                "availability",
                "simulate_chunk",
                "availability_from_chunks",
                "shards",
            ):
                out[name] = r.refused(meshed)
        return self._meshed_routes(out, free)

    def _capacity_refusal(self) -> Optional[Tuple[str, tuple]]:
        """What ``capacity_distribution`` refuses beyond the long-run
        values, in the order it checks: a degrading component on a
        schedule, here or in a nested RBD, then no capacity at all. The
        message and the nodes, or None."""
        from repyability.rbd import routes as r

        for node, model in self._capacity_models().items():
            if isinstance(model, RepairableRBD):
                inner = model._capacity_refusal()
                if inner:
                    return inner[0], (node,)
                continue
            message = r.refusal(
                partial(self._require_unscheduled_stages, node)
            )
            if message:
                return message, (node,)
        message = r.refusal(self._require_capacity)
        return (message, ()) if message else None

    def _node_long_run(self, node) -> Tuple[str, str]:
        """How a component's long-run values are found: the route, and a
        phrase saying how (the message it raises, if refused)."""
        from repyability.rbd import routes as r

        component = self.components[node]
        if isinstance(component, RepairableRBD):
            inner = component._long_run_route()
            if inner.route == r.REFUSED:
                return r.REFUSED, inner.reason
            return (
                inner.route,
                f"a nested RBD's long-run values, {inner.route}",
            )
        message = r.refusal(partial(self._require_perfect_repair, node))
        if message:
            return r.REFUSED, message
        if node in self._standby:
            message = r.refusal(partial(self._standby_rates, node))
            if message:
                return r.REFUSED, message
            return r.EXACT, "a standby group's Markov chain"
        if node in self._inspection:
            message = r.refusal(partial(self._require_tested_exact, node))
            if message:
                return r.REFUSED, message
            if _constant_rate(component.reliability) is None:
                return (
                    r.NUMERICAL,
                    "hidden failures, renewed at the tests, summed over the "
                    "test intervals",
                )
            return r.EXACT, "hidden failures at a constant rate"
        schedule = self._preventive.get(node)
        if schedule is None:
            message = r.refusal(component.mean_availability)
            if message:
                return r.REFUSED, message
            life, how = r.mean_route(component.reliability)
            if life == r.EXACT:
                return r.EXACT, "its mean life and mean repair time"
            return life, f"its mean life, {how}"
        message = r.refusal(partial(self._require_no_opportunities, node))
        if message:
            return r.REFUSED, message
        life, how = r.model_route(component.reliability)
        if schedule.policy in ("block", "condition"):
            message = r.refusal(partial(self._require_block_models, node))
            if message:
                return r.REFUSED, message
            if schedule.policy == "condition":
                return (
                    r.NUMERICAL,
                    "replacement on condition: its renewal cycle, followed "
                    "from one inspection to the next on a grid",
                )
            return (
                r.NUMERICAL,
                "block replacement: its renewal cycle, solved on a grid",
            )
        if life == r.SIMULATED:
            return life, f"age replacement of a life from {how}"
        return r.NUMERICAL, "age replacement: its mean up time, by quadrature"

    def _long_run_nodes(self) -> Dict[Any, Tuple[str, str]]:
        """Each component's long-run route (see ``_node_long_run``)."""
        return {node: self._node_long_run(node) for node in self.components}

    def _long_run_route(self, groups: bool = True) -> "AnalysisRoute":
        """How the exact long-run values are found (see
        ``analysis_routes``); without ``groups``, the components' own, as
        ``node_availability`` gives them (the common-cause groups change
        only which are down together)."""
        from repyability.rbd import routes as r

        if self._crews_couple():
            return self._crew_chain_route()
        calendars = r.refusal(self._require_calendars)
        if calendars:
            return r.refused(calendars)
        nodes = self._long_run_nodes()
        refusals = {
            n: how for n, (route, how) in nodes.items() if route == r.REFUSED
        }
        if refusals:
            return r.refused(next(iter(refusals.values())), tuple(refusals))
        chains = r.refusal(self._require_ccf_long_run) if groups else None
        if chains:
            return r.refused(chains)
        return r.with_nodes(
            r.EXACT,
            "The structure function over the components' long-run "
            "availabilities, exactly"
            + (
                ", and over each common-cause group's members' joint states "
                "(a Markov chain)."
                if self.ccf_groups and groups
                else "."
            ),
            nodes,
            "long-run values",
        )

    def _crew_chain_route(self) -> "AnalysisRoute":
        """How the exact long-run values are found with limited repair
        crews: from their Markov chain, and nested RBDs' own values."""
        from repyability.rbd import routes as r

        chain = r.refusal(self._require_crew_chain)
        if chain:
            return r.refused(chain)
        nested = {
            node: self._node_long_run(node)
            for node, component in self.components.items()
            if isinstance(component, RepairableRBD)
        }
        refusals = {
            n: how for n, (route, how) in nested.items() if route == r.REFUSED
        }
        if refusals:
            return r.refused(next(iter(refusals.values())), tuple(refusals))
        return r.with_nodes(
            r.EXACT,
            "The long-run distribution of the Markov chain of the "
            "components' states and the repair queue, solved exactly, and "
            "the system's values averaged over its states.",
            nested,
            "long-run values",
        )

    def _node_over_time(
        self, node, kind: str = "availability"
    ) -> Tuple[str, str]:
        """How a component's availability over time is found, for ``kind``
        (see ``_over_time``): the route, and a phrase saying how (the
        message it raises, if refused)."""
        from repyability.rbd import routes as r

        component = self.components[node]
        if isinstance(component, RepairableRBD):
            if (
                kind == "capacity"
                and component._crews_couple()
                and component._has_capacity()
            ):
                return r.REFUSED, str(
                    r.refusal(
                        partial(
                            component._no_crew_window, None, "the capacities"
                        )
                    )
                )
            inner = component._over_time(kind=kind)
            if inner.route == r.REFUSED:
                return r.REFUSED, inner.reason
            return inner.route, f"a nested RBD's availability, {inner.route}"
        if node in self._standby:
            message = r.refusal(partial(self._standby_rates, node))
            if message:
                return r.REFUSED, message
            return (
                r.NUMERICAL,
                "its units' Markov chain, followed over time by "
                "uniformization",
            )
        message = r.refusal(partial(self._require_time_models, node))
        if not message:
            message = r.refusal(partial(self._require_no_condition, node))
        if not message:
            message = r.refusal(partial(self._require_perfect_repair, node))
        if not message:
            message = r.refusal(partial(self._require_no_opportunities, node))
        if not message and node in self._inspection:
            message = r.refusal(partial(self._require_tested_exact, node))
        schedule = self._preventive.get(node)
        if not message and schedule is not None and schedule.policy == "block":
            message = r.refusal(partial(self._require_block_models, node))
        if message:
            return r.REFUSED, message
        life, how = r.model_route(component.reliability)
        if life == r.SIMULATED:
            return life, f"a life from {how}"
        return r.NUMERICAL, "its renewal equation, solved on a grid"

    def _over_time(
        self,
        reason: str = (
            "Each component's renewal equation solved on a grid (to about "
            "1e-7), and the system exactly at its components' availabilities "
            "at each time."
        ),
        kind: str = "availability",
    ) -> "AnalysisRoute":
        """How the availability over time from new is found (see
        ``analysis_routes``), and what is built on it: the method's
        ``reason``. ``kind`` is what is asked: the ``"availability"``, the
        expected events over a ``"window"``, or the ``"capacity"``, which
        with limited repair crews come from their chain (see
        ``_crew_over_time``)."""
        from repyability.rbd import routes as r

        if self._crews_couple():
            return self._crew_over_time(kind)
        ccf = r.refusal(partial(self._require_no_ccf, "the values over time"))
        if ccf:
            return r.refused(ccf)
        nodes = {
            node: self._node_over_time(node, kind) for node in self.components
        }
        refusals = {
            n: how for n, (route, how) in nodes.items() if route == r.REFUSED
        }
        if refusals:
            return r.refused(next(iter(refusals.values())), tuple(refusals))
        return r.with_nodes(r.NUMERICAL, reason, nodes, "availabilities")

    def _crew_over_time(self, kind: str) -> "AnalysisRoute":
        """How ``kind`` over time (see ``_over_time``) is found with limited
        repair crews: from their chain, followed over time by
        uniformization (see ``_crew_curve``), unless it refuses."""
        from repyability.rbd import routes as r

        refusal = r.refusal(
            partial(self._require_crew_over_time, frozenset(), kind)
        )
        if refusal:
            return r.refused(refusal)
        chain = (
            "The Markov chain of the components' states and the repair "
            "queue, followed over time from its state at 0 by "
            "uniformization (to about 1e-13): "
        )
        nested = self._crew_nested(frozenset())
        if kind == "window":
            reason = chain + (
                "the system's expected failures, and each component's "
                "failures and down time, are the integrals of their rates "
                "over its states."
            )
        elif kind == "capacity":
            reason = chain + (
                "the capacity distribution is the probability of the states "
                "at each level, and over a mission its integral."
            )
        elif nested:
            reason = chain + (
                "the system's availability is the probability of the states "
                "it is up in, worked out for each pattern of the nested RBDs "
                "up and down with their own availabilities, and the mission "
                "availability its integral, by quadrature."
            )
        else:
            reason = chain + (
                "the system's availability is the probability of the states "
                "it is up in, and the mission availability its integral, "
                "exactly."
            )
        nodes = {node: self._node_over_time(node, kind) for node in nested}
        refusals = {
            n: how for n, (route, how) in nodes.items() if route == r.REFUSED
        }
        if refusals:
            return r.refused(next(iter(refusals.values())), tuple(refusals))
        return r.with_nodes(r.NUMERICAL, reason, nodes, "availabilities")

    def _require_crew_over_time(
        self, forced=frozenset(), kind: str = "availability"
    ) -> list:
        """Raise what the crews' chain over time refuses, in the order the
        methods check it: common-cause groups, what the chain does not
        cover (see ``_require_crew_chain``), too many nested RBDs, and,
        for the expected events and the capacity, any nested RBD (#162).
        Return the nested RBDs, less those in ``forced``."""
        self._require_no_ccf("the values over time")
        self._require_crew_chain()
        nested = self._crew_nested(forced)
        if nested and kind != "availability":
            self._no_crew_window(
                nested[0],
                (
                    "the expected events"
                    if kind == "window"
                    else "the capacities"
                ),
            )
        return nested

    def _engine_choice(self, capacity: bool) -> Tuple[str, str]:
        """The engine ``engine="auto"`` runs a long simulation on, and why
        (see ``_simulation_engine``)."""
        from repyability.rbd import _compiled

        plan = self._stream_plan(1.0, 0, False)[0]
        engine, reason = _compiled.choice(
            self, plan, object() if capacity else None
        )
        if reason is not None:
            return "python", f"the compiled engine does not simulate {reason}"
        if engine is None:
            return (
                "python",
                "numba is not installed; pip install 'repyability[fast]' "
                "for the compiled engine",
            )
        return (
            engine,
            "compiled for a run long enough to repay loading it; a short "
            "one runs in Python",
        )

    def mean_availability(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
    ) -> float:
        """Returns the system's long-run (steady-state) availability.

        Exact, with no simulation. Each component's long-run availability
        is ``MTTF / (MTTF + MTTR)`` (1 if it is repaired instantly; a nested
        ``RepairableRBD``'s is its own ``mean_availability``; under age
        replacement, see ``node_availability``). Since the components fail
        and are repaired independently, the system's is
        the RBD's structure function evaluated exactly at those
        availabilities. It is the long-run fraction of time the system is
        up; for availability over time, from a start with everything
        working, use ``availability``.

        A component with hidden failures is up with probability
        ``exp(-lambda * u)`` at a time ``u`` since its last inspection, so
        components inspected at the same times are down together more
        often than independent ones would be. The system's availability is
        then averaged over time, over one period of the inspection
        schedules (the least common multiple of their intervals): for two
        such components in parallel, inspected together every ``tau``, the
        unavailability is ``(1 / tau) * integral_0^tau (1 -
        exp(-lambda * t)) ** 2 dt``, about ``(lambda * tau) ** 2 / 3``: the
        average probability of failure on demand (PFDavg) of a 1oo2 safety
        function. This is exact for a constant failure rate, and numerical
        (summed over the test intervals) for any other life, with instant
        tests and instant repair (see ``node_availability``).

        With fewer ``repair_crews`` than components, a failed component can
        wait for a crew, and the components no longer fail and recover
        independently. When the components the crews work on have
        exponential lives and exponential (or instant) repairs, with no
        scheduled maintenance or inspection, the system is a Markov chain:
        its state is which components are under repair and which are
        waiting, in the order the crews will take them. Its long-run
        distribution is solved exactly (for up to 15,000 states), and the
        availability is the system's, state by state, averaged over it. A
        node held working never fails and one held broken is never
        repaired, so neither needs a crew: the chain is of the others. A
        nested ``RepairableRBD`` has crews of its own, and enters through
        its own long-run availability.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working (availability 1), by
            default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed (availability 0), by
            default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"``, the default) or the minimal cut sets (``"c"``); both
            give the same exact result.

        Returns
        -------
        float
            Long-run availability of the system, in ``[0, 1]``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model, whose MTTF this cannot
            compute.
        NotImplementedError
            If a component is under block replacement with models its exact
            values do not cover (see ``node_availability``), or has hidden
            failures whose tests or repairs take time, or whose tests can
            miss a failure of a life that is not exponential; or, while a
            component can wait for a repair crew, if the Markov chain does
            not cover the components (a life
            or repair that is not exponential, scheduled maintenance or an
            inspection) or would have more than 15,000 states: simulate it
            with ``availability``.

        Examples
        --------
        A single component with mean time to failure 10 and mean time to
        repair 1 has long-run availability ``10 / (10 + 1)``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...         }
        ...     },
        ... )
        >>> round(rbd.mean_availability(), 4)
        0.9091
        """
        # Good reference on the Availability of a system
        # https://www.diva-portal.org/smash/get/diva2:986067/FULLTEXT01.pdf
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)

        # Over one period of the inspection schedules, if any (see
        # _long_run_grid), or over the states of the repair crews' Markov
        # chain; otherwise at one point, as the node availabilities are
        # constant.
        availability, weights = self._long_run_probabilities(
            working_nodes, broken_nodes
        )
        system = self.system_probability(availability, method=method)
        return float(weights @ system)

    def point_availability(
        self,
        x,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ):
        """The probability that the system is up at each time ``x``, with
        every component new at 0: exact, with no simulation.

        Each component's point availability ``A(t)`` follows from the
        distributions of its up and down times by the renewal equation,
        solved numerically (see ``repyability/rbd/_point_availability.py``):
        its error is about 1e-7 (up to 1e-6 soon after the start). Since
        the components fail and are repaired independently, the system's is
        the structure function evaluated exactly at theirs, at each time.
        It starts at 1 (less any components dead on arrival) and settles at
        [`mean_availability`][repyability.RepairableRBD.mean_availability]
        (or, with components replaced or inspected on a calendar, repeats
        with the calendar about it);
        [`availability`][repyability.RepairableRBD.availability] estimates
        the same curve by simulation. At a time something happens -- a
        block replacement, say -- it is the availability just after.

        Components under age or block replacement are covered, as in
        ``mean_availability``, and nested RBDs through their own point
        availability. A component with hidden failures is up with the
        probability that it has not failed since its last test, ``u``
        before (``exp(-lambda * u)`` for a constant failure rate): as in
        ``mean_availability``, with instant tests and instant repair.

        With fewer ``repair_crews`` than components, a component can wait
        for a crew and the components are not independent: the system's
        point availability is then that of the crews' Markov chain (see
        ``mean_availability``), followed over time from its state at 0 by
        uniformization, to about 1e-13; a nested RBD, which has crews of
        its own, enters through its own point availability. A standby
        group's comes from its units' chain in the same way.

        Each component's curve is computed on a grid of 2,000 steps over its
        typical up time. Near a time at which its units start or stop on a
        schedule -- at 0, at its scheduled replacements, and at the failures
        of a lifetime known exactly -- what happens faster than a step, such
        as a short repair, is smoothed over the step: a point value within
        a step of such a time can be off by up to about the probability
        that the component is under repair then. ``mission_availability``
        is not affected.

        Parameters
        ----------
        x : float or array-like
            Times, from 0 (every component new).
        working_nodes : Collection[Hashable], optional
            Nodes that always work (availability 1), by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed (availability 0), by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"``, the default) or the cut sets (``"c"``); both give the
            same result.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        float or numpy.ndarray
            The system's point availability at each time, in ``x``'s shape.

        Raises
        ------
        ValueError
            If a time is negative or not finite, or a working/broken node is
            invalid (see ``mean_availability``).
        NotImplementedError
            If a component has hidden failures whose tests or repairs take
            time, or whose tests can miss a failure of a life that is not
            exponential, a model that block replacement's exact values do
            not cover (see
            ``mean_availability``), or time scales too far apart for the
            grid (years of running, seconds of repair, over centuries); or,
            while a component can wait for a repair crew (see
            ``repair_crews``), if the crews' Markov chain does not cover the
            components (see ``mean_availability``) or its rates are too far
            apart to follow it over time: simulate it with
            ``availability``.

        Examples
        --------
        One component with failure rate 0.1 and repair rate 1 is up at time
        ``t`` with probability ``1 / 1.1 + (0.1 / 1.1) * exp(-1.1 t)``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...         }
        ...     },
        ... )
        >>> rbd.point_availability([0.0, 1.0, 100.0]).round(4).tolist()
        [1.0, 0.9394, 0.9091]
        """
        times = _check_times(x)
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        horizon = float(times.max()) if times.size else 0.0
        forced = working_nodes | broken_nodes
        states = self._states(state, forced)
        if self._crews_couple():
            values = self._crew_curve(
                horizon, working_nodes, broken_nodes, method, states
            ).at(times.ravel())
        else:
            curves = self._availability_curves(horizon, forced, state=states)
            values = self._curves_at(
                curves, times.ravel(), working_nodes, broken_nodes, method
            )
        if np.ndim(x) == 0:
            return float(values[0])
        return values.reshape(times.shape)

    def mission_availability(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ):
        """The expected fraction of ``[0, t]`` the system is up, with every
        component new at 0: exact, with no simulation.

        It is the mean of
        [`point_availability`][repyability.RepairableRBD.point_availability]
        over the window, integrated by 4-point Gauss-Legendre quadrature on
        pieces that end where the components' curves bend (a scheduled
        replacement, a test, a down time from a known instant) and are at
        most a few steps of the finest grid the curves are worked out on,
        each halved until the quadrature on it agrees with that on its
        halves, to about 1e-8: it takes little longer than the point
        availability. That runs up to the time when the curves have all
        settled at their long-run values (or into repeating with their
        inspections and block replacements); past it, the integral is
        extended exactly, so a mission of decades costs no more than one of
        a few years. It is what
        [`availability`][repyability.RepairableRBD.availability] estimates
        by simulation as the mean of each simulation's uptime divided by
        ``t_simulation``. As ``t`` grows it approaches ``mean_availability``,
        from which it differs by about ``b / t``, ``b`` a constant of the
        components' up and down times: positive unless their lives vary
        more than exponential ones do, and largest for components that wear
        out (Weibull shape above 1, say), which from new fail less early
        on.

        Parameters
        ----------
        t : float or array-like
            The windows' lengths (one mission average for each).
        working_nodes : Collection[Hashable], optional
            Nodes that always work (availability 1), by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed (availability 0), by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"``, the default) or the cut sets (``"c"``); both give the
            same result.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        float or numpy.ndarray
            The mission availability for each window, in ``t``'s shape (at
            ``t = 0``, the point availability at 0).

        Raises
        ------
        ValueError, NotImplementedError
            As for ``point_availability``.

        Examples
        --------
        One component with failure rate 0.1 and repair rate 1, over 10 time
        units: ``1 / 1.1 + 0.1 / 1.1 ** 2 * (1 - exp(-11)) / 10``.

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...         }
        ...     },
        ... )
        >>> round(rbd.mission_availability(10.0), 4)
        0.9174
        """
        windows = _check_times(t)
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        ends = windows.ravel()
        horizon = float(ends.max()) if ends.size else 0.0
        forced = working_nodes | broken_nodes
        states = self._states(state, forced)
        if self._crews_couple():
            crew = self._crew_curve(
                horizon, working_nodes, broken_nodes, method, states
            )
            system_at, pieces = crew.at, [crew]
        else:
            curves = self._availability_curves(horizon, forced, state=states)

            def system_at(x):
                return self._curves_at(
                    curves, x, working_nodes, broken_nodes, method
                )

            pieces = list(curves.values())
        if self._crews_couple() and not crew.nested:
            # The crews' chain integrates it exactly.
            totals = crew.integral(ends)
        else:
            totals = self._integrated(system_at, pieces, ends, horizon)
        averages = np.empty(len(ends))
        positive = ends > 0.0
        averages[positive] = totals[positive] / ends[positive]
        if not positive.all():
            averages[~positive] = system_at(np.zeros(1))[0]
        if np.ndim(t) == 0:
            return float(averages[0])
        return averages.reshape(windows.shape)

    def _integrated(
        self, system_at, curves: list, ends: np.ndarray, horizon: float
    ) -> np.ndarray:
        """The integral from 0 to each of ``ends`` of the system's point
        availability, ``system_at``, from its nodes' ``curves`` (over the
        pieces of ``_quadrature``, and settling into a constant or a period
        it is extended over: see ``mission_availability``)."""
        # After ``settle`` the system's availability is constant, or repeats
        # with ``period``: it is integrated up to ``reach``, and extended.
        settle, period = _settling(curves)
        reach = min(horizon, settle if period is None else settle + period)
        beyond = ends > reach
        fixed = [ends[~beyond]]
        if period is not None and beyond.any():
            cycles = np.floor((ends[beyond] - settle) / period)
            rest = ends[beyond] - settle - cycles * period
            rest = np.clip(rest, 0.0, period)
            fixed += [np.array([settle]), settle + rest]

        def estimate(a, b):
            x, half = _quadrature.points(a, b)
            return {"up": _quadrature.summed(system_at(x), half)}

        try:
            edges, finest = _quadrature.pieces(
                curves, np.concatenate(fixed), reach, _MISSION_POINTS
            )
            edges, integrals = _quadrature.refined(
                estimate, edges, finest, _MISSION_POINTS
            )
        except _quadrature.TooMany as error:
            raise NotImplementedError(
                f"Integrating the availability over [0, {reach}] takes "
                f"{error.count} pieces (the components' curves bend that "
                "often), more than the limit: estimate it by simulation, "
                "with availability()."
            ) from None
        running = _quadrature.running(integrals.get("up"), len(edges) - 1)

        def uptime(x):
            return running[np.searchsorted(edges, x)]

        totals = np.empty(len(ends))
        totals[~beyond] = uptime(ends[~beyond])
        if beyond.any():
            if period is None:
                level = system_at(np.array([horizon]))[0]
                totals[beyond] = uptime(reach) + (ends[beyond] - reach) * level
            else:
                base = uptime(settle)
                totals[beyond] = (
                    base
                    + cycles * (uptime(reach) - base)
                    + (uptime(settle + rest) - base)
                )
        return totals

    def _window(
        self,
        t,
        working_nodes,
        broken_nodes,
        method: str,
        nodes: bool = False,
        setups: bool = False,
        state=None,
    ):
        """The windows ``t`` checked, flat, and the nodes' curves and the
        system's expected events over them (see ``_window_counts``): the
        forced nodes have no curves. With ``setups``, the stops of each
        maintenance group with a set-up cost are counted too."""
        if method not in ("p", "c"):
            raise ValueError("`method` must be either 'p' or 'c'")
        windows = _check_times(t)
        working = set() if working_nodes is None else set(working_nodes)
        broken = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working, broken)
        forced = working | broken
        ends = windows.ravel()
        horizon = float(ends.max()) if ends.size else 0.0
        states = self._states(state, forced)
        groups = None
        if setups:
            groups = {
                name: [node for node in spec.members if node not in forced]
                for name, spec in self._maintenance.items()
                if spec.setup_cost
            }
        if self._crews_couple():
            curves, counts = self._crew_window(
                ends, working, broken, method, states, nodes, groups
            )
            return windows, ends, curves, counts, working, broken
        curves = self._availability_curves(
            horizon, forced, counts=True, state=states
        )
        counts = self._window_counts(
            curves, ends, working, broken, method, nodes=nodes, groups=groups
        )
        return windows, ends, curves, counts, working, broken

    def expected_failures(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ):
        """The expected number of system failures in ``[0, t)``, with every
        component new at 0: exact, with no simulation.

        A component's failure takes the system down if the component is
        critical then: if the system is up with it and down without it.
        The components fail and are repaired independently, so at a time
        ``s`` that is so with probability ``I_B^i(s)``, the component's
        Birnbaum importance at the components' point availabilities then,
        and the system's expected failures are

        ```text
        integral from 0 to t of  sum_i I_B^i(s) dM_i(s)
        ```

        ``M_i(s)`` the component's expected failures by ``s``: the
        time-dependent form of the Birnbaum/Vesely formula, whose long-run
        rate is ``system_failure_frequency``. Each ``M_i`` follows from the
        component's renewal equation, solved on the same grid as its point
        availability (see ``point_availability``), to about 1e-7; the
        integral is summed over the pieces ``mission_availability``
        integrates over, on each with the importance taken as the cubic
        through its values at the quadrature points, and extended exactly
        past the time the curves have settled. Failures at exact
        times (units dead on arrival, an exact lifetime) are taken together
        when several fall at once. It is what ``availability`` estimates as
        ``system_failures / n_simulations``: every system failure counts,
        including the zero-length outages an instantly repaired component
        causes; planned outages do not (see ``expected_events``).

        With fewer ``repair_crews`` than components, the components are not
        independent: the rate of system failures is then that of the crews'
        Markov chain in each of its states (a component up failing at its
        rate where it is critical), integrated over the chain's states over
        time by uniformization, to about 1e-13 (see ``point_availability``).
        The crews' chain takes no nested RBD's events in, as yet (#162).

        Parameters
        ----------
        t : float or array-like
            The windows' lengths (one count for each).
        working_nodes : Collection[Hashable], optional
            Nodes that always work, and never fail, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed, by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"``, the default) or the cut sets (``"c"``); both give the
            same result.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        float or numpy.ndarray
            The expected number of system failures in each window, in
            ``t``'s shape.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``point_availability``.

        Examples
        --------
        One component with failure rate 0.1 and repair rate 1 fails at the
        rate 0.1 while it is up, ``0.1 * (10 / 1.1 + 0.1 / 1.1 ** 2 * (1 -
        exp(-11)))`` times in 10 time units:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...         }
        ...     },
        ... )
        >>> round(rbd.expected_failures(10.0), 4)
        0.9174
        """
        windows, _, _, counts, _, _ = self._window(
            t, working_nodes, broken_nodes, method, state=state
        )
        return _shaped(counts["failures"], t, windows)

    def expected_events(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ) -> ExpectedEvents:
        """What the system and each component are expected to do in
        ``[0, t)``, with every component new at 0: exact, with no
        simulation.

        The system's failures are those of ``expected_failures``, and its
        planned outages are counted the same way, from its components'
        preventive maintenance that takes time; its down time is ``t * (1
        - mission_availability(t))``. Each component's failures, corrective
        and preventive actions and tests follow from its renewal equation,
        solved on the grid of its point availability (see
        ``point_availability``), to about 1e-7, and its down time is the
        integral of its unavailability. The maintenance due at exact times
        (at a replacement age, at block times) is counted exactly, and
        maintenance of several components due at the same time takes the
        system down at most once. It is the mean of what ``availability``
        counts in each simulation of the window, and events at ``t`` itself
        fall outside the window, as there.

        A component's corrective actions, at each of which its
        ``repair_cost`` and ``replace_cost`` are charged, are its failures,
        except for hidden failures: those found by a test in the window. A
        nested RBD's failures are its system failures, and its own
        maintenance is not this RBD's to count. A forced node does nothing:
        held working it is never down, held broken it is down throughout.

        Parameters
        ----------
        t : float or array-like
            The windows' lengths.
        working_nodes : Collection[Hashable], optional
            Nodes that always work, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed, by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"``, the default) or the cut sets (``"c"``); both give the
            same result.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        ExpectedEvents
            The system's expected failures, planned outages and down time,
            and each component's expected failures, corrective and
            preventive actions, tests and down time: floats for one window,
            arrays in ``t``'s shape for several.

        Raises
        ------
        ValueError, NotImplementedError
            As for ``point_availability``.

        Examples
        --------
        A pump that wears out, replaced at 600 hours in about 7 (a planned
        outage), over a year:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "p"), ("p", "t")],
        ...     {
        ...         "p": {
        ...             "reliability": surv.Weibull.from_params([1000, 2.5]),
        ...             "repairability": surv.LogNormal.from_params([3, 0.5]),
        ...             "preventive": {
        ...                 "interval": 600.0,
        ...                 "duration": surv.Weibull.from_params([8, 3]),
        ...             },
        ...         }
        ...     },
        ... )
        >>> year = rbd.expected_events(8760.0)
        >>> round(year.system_failures, 3), round(year.node_preventive["p"], 3)
        (3.706, 11.275)
        >>> round(year.system_planned_outages, 3)  # each replacement
        11.275
        """
        windows, ends, curves, counts, working, broken = self._window(
            t, working_nodes, broken_nodes, method, nodes=True, state=state
        )
        node_events = {
            node: curve.events(ends) for node, curve in curves.items()
        }
        zero = np.zeros(len(ends))

        def shaped(values) -> Any:
            return _shaped(values, t, windows)

        def per_node(key: str) -> dict:
            return {
                node: shaped(
                    node_events[node][key] if node in curves else zero
                )
                for node in self.components
            }

        downtime = {
            node: shaped(
                counts["downtime"][node]
                if node in curves
                else (ends if node in broken else zero)
            )
            for node in self.components
        }
        return ExpectedEvents(
            window=shaped(ends),
            system_failures=shaped(counts["failures"]),
            system_planned_outages=shaped(counts["planned"]),
            system_downtime=shaped(np.maximum(ends - counts["uptime"], 0.0)),
            node_failures=per_node("failures"),
            node_corrective=per_node("corrective"),
            node_preventive=per_node("preventive"),
            node_inspections=per_node("inspections"),
            node_downtime=downtime,
        )

    def expected_cost(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        state=None,
    ) -> ExpectedCost:
        """The expected cost of running the system for ``t``, from new:
        exact, with no simulation.

        Each category is its events' expected number over ``[0, t)`` (see
        ``expected_events``) times its mean cost, as ``cost`` charges them:
        ``repair_cost`` and ``replace_cost`` at each corrective action, the
        preventive cost at each preventive replacement, the inspection cost
        at each test, ``downtime_cost`` over each component's expected down
        time, ``downtime_cost_rate`` over the system's, and a maintenance
        group's set-up cost once per stop (each failure or replacement of a
        member, those at one instant one stop). It is the mean that
        ``cost`` estimates by simulation, with its distribution;
        ``total_cost`` is the long-run approximation, exact only over a
        window long compared with the components' cycles. As ``t`` grows
        this approaches ``expected_cost_rate() * t`` plus a constant, of the
        early years from new.

        Every cost is optional, and a cost given as a distribution enters
        through its mean. Only this RBD's own costs count, not those inside
        a nested ``RepairableRBD``. With nothing priced (see
        ``has_costs``), every category is 0, without any checks.

        Parameters
        ----------
        t : float or array-like
            The windows' lengths.
        working_nodes : Collection[Hashable], optional
            Nodes that always work: they incur no corrective, preventive,
            inspection or downtime cost, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always failed: they incur no corrective cost, but
            their own downtime cost throughout, by default None.
        method : str, optional
            Evaluate the structure function from the minimal path sets
            (``"p"``, the default) or the cut sets (``"c"``); both give the
            same result.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        ExpectedCost
            The expected cost of each window, by category and component,
            with the acquisition cost beside it.

        Raises
        ------
        ValueError, NotImplementedError
            When something is priced, as for ``point_availability``.

        Examples
        --------
        A pump bought for 20,000, failing on average every 1000 hours,
        repaired in 10 at 500 per repair, with lost production at 100 per
        hour, owned for a year:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "p"), ("p", "t")],
        ...     {
        ...         "p": {
        ...             "reliability": surv.Exponential.from_params([1e-3]),
        ...             "repairability": surv.Exponential.from_params([0.1]),
        ...             "repair_cost": 500.0,
        ...             "acquisition_cost": 20000.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=100.0,
        ... )
        >>> year = rbd.expected_cost(8760.0)
        >>> round(year.mean), round(year.total)
        (13000, 33000)
        >>> round(rbd.total_cost(8760.0))  # at the long-run rate
        33010
        """
        if not self.has_costs:
            windows = _check_times(t)
            zero = _shaped(np.zeros(windows.size), t, windows)
            return ExpectedCost(
                window=_shaped(windows.ravel(), t, windows),
                mean=zero,
                by_category={category: zero for category in _CATEGORIES},
                by_component={},
                acquisition_cost=self.acquisition_cost,
            )
        windows, ends, curves, counts, working, broken = self._window(
            t,
            working_nodes,
            broken_nodes,
            method,
            nodes=True,
            setups=True,
            state=state,
        )
        node_events = {
            node: curve.events(ends) for node, curve in curves.items()
        }
        categories = {
            category: np.zeros(len(ends)) for category in _CATEGORIES
        }
        by_component: Dict[Hashable, Any] = {}
        for node, node_costs in self.costs.items():
            own = np.zeros(len(ends))
            charges = []
            events = node_events.get(node)
            if events is not None:
                for key, category, kind in (
                    ("repair_cost", "repair", "corrective"),
                    ("replace_cost", "replace", "corrective"),
                    ("preventive_cost", "preventive", "preventive"),
                    ("inspection_cost", "inspection", "inspections"),
                ):
                    if key in node_costs:
                        charges.append(
                            (
                                category,
                                _mean_cost(node_costs[key]) * events[kind],
                            )
                        )
            rate = node_costs.get("downtime_cost", 0.0)
            if rate:
                down = (
                    counts["downtime"][node]
                    if node in curves
                    else (ends if node in broken else np.zeros(len(ends)))
                )
                charges.append(("component_downtime", rate * down))
            for category, charge in charges:
                categories[category] = categories[category] + charge
                own = own + charge
            by_component[node] = _shaped(own, t, windows)
        if self.downtime_cost_rate:
            categories["system_downtime"] = self.downtime_cost_rate * (
                np.maximum(ends - counts["uptime"], 0.0)
            )
        for name, spec in self._maintenance.items():
            if not spec.setup_cost:
                continue
            # Each failure or replacement of a member is a stop, but those
            # at one instant are one.
            stops = -counts["overlaps"].get(name, np.zeros(len(ends)))
            for node in spec.members:
                if node in node_events:
                    stops = stops + (
                        node_events[node]["failures"]
                        + node_events[node]["preventive"]
                    )
            categories["setup"] = categories["setup"] + spec.setup_cost * stops
        mean = np.zeros(len(ends))
        for values in categories.values():
            mean = mean + values
        return ExpectedCost(
            window=_shaped(ends, t, windows),
            mean=_shaped(mean, t, windows),
            by_category={
                category: _shaped(values, t, windows)
                for category, values in categories.items()
            },
            by_component=by_component,
            acquisition_cost=self.acquisition_cost,
        )

    def _filled(
        self, probabilities: dict, size: int, working_nodes, broken_nodes
    ) -> dict:
        """The nodes' ``probabilities`` (arrays of ``size``), with the input
        and output nodes, and the forced nodes held at 1 or 0, added."""
        out = dict(probabilities)
        for node in list(self.in_or_out) + list(working_nodes | broken_nodes):
            out[node] = np.ones(size)
        return self._probabilities_with_overrides(
            out, working_nodes, broken_nodes
        )

    def _atom_groups(
        self, curves: dict, atoms: dict, working_nodes, broken_nodes
    ) -> _Jumps:
        """The system at the times its nodes fail or are taken down at
        exact times (their ``atoms``): a unit dead on arrival at 0, an exact
        lifetime, a replacement at its age or at a block time.

        Nodes due at one time go down together, so the system fails there
        if it is up just before and down just after: since they only go
        down, that is the fall in its availability, worked out as the rise
        in its unavailability (a sum of products, which keeps a small one's
        precision). A node's value just before is its point availability
        there with its own events' fall (``drop``) added back; just after,
        less the probabilities of its failures, then of its planned outages
        (the failures are taken to come first). The others are at their
        point availability there: tests and restorations due then are
        done. One node alone fails the system with the probability of its
        failure times its Birnbaum importance there."""
        down = [
            a.times[(a.failure > 0.0) | (a.planned > 0.0)]
            for a in atoms.values()
        ]
        times = np.unique(np.concatenate(down)) if down else np.empty(0)
        if not times.size:
            empty = np.empty(0)
            return _Jumps(empty, empty, empty, empty)
        at, before, failed, planned = {}, {}, {}, {}
        for node, curve in curves.items():
            value = np.asarray(curve.at(times.copy()), dtype=float)
            a = atoms[node]
            drop = _matched(times, a.times, a.drop)
            failing = _matched(times, a.times, a.failure)
            stopping = _matched(times, a.times, a.planned)
            at[node] = value
            before[node] = np.clip(value + drop, 0.0, 1.0)
            failed[node] = np.clip(before[node] - failing, 0.0, 1.0)
            planned[node] = np.clip(failed[node] - stopping, 0.0, 1.0)

        def unreliability(values: dict) -> np.ndarray:
            return self._system_unreliability(
                self._filled(values, len(times), working_nodes, broken_nodes)
            )

        now, first = unreliability(at), unreliability(before)
        after, last = unreliability(failed), unreliability(planned)
        return _Jumps(
            times,
            np.maximum(after - first, 0.0),
            np.maximum(last - after, 0.0),
            np.maximum(now - first, 0.0),
        )

    def _system_atoms(self, curves: dict, stop: float) -> Atoms:
        """This RBD's own failures and planned outages at exact times before
        ``stop``, from its nodes' ``curves``, as a node of another (see
        ``Atoms``)."""
        atoms = {node: curve.atoms(stop) for node, curve in curves.items()}
        jumps = self._atom_groups(curves, atoms, set(), set())
        return Atoms(
            jumps.times,
            jumps.failures,
            jumps.planned,
            np.zeros(len(jumps.times)),
            jumps.drop,
        )

    def _window_counts(
        self,
        curves: dict,
        ends,
        working_nodes,
        broken_nodes,
        method: str,
        nodes: bool = False,
        groups: Optional[dict] = None,
    ) -> dict:
        """The system's expected up time, failures and planned outages in
        ``[0, end)`` for each of ``ends``, from its nodes' ``curves`` (which
        count their events); with ``nodes``, each node's expected down time
        too; and for each of ``groups`` (a name and members), the expected
        number of its members' failures and replacements that fall at one
        instant with another's (each stop of a maintenance group is one).

        The up time integrates the system's point availability as
        ``mission_availability`` does. A node's failure takes the system
        down if the node is critical then, which, the nodes being
        independent, it is at ``t`` with probability ``I_B(t)``, its
        Birnbaum importance at the nodes' availabilities then: the system's
        failures are ``integral of sum_i I_B^i(t) dM_i(t)``, ``M_i(t)`` the
        node's expected failures by ``t``, and its planned outages likewise.
        It is summed over the pieces of ``_quadrature``, on each with the
        importance taken as the cubic through its values at the points the
        up time is integrated at (see ``_quadrature.stieltjes``); the events
        at exact times are kept apart and taken together (see
        ``_atom_groups``), and start pieces. The ends do not: a total at
        an end within a piece adds the integral from the piece's start,
        by the same quadrature, so that counting at many times (a nested
        RBD's, for the RBD it is in) costs no more pieces. Past the time the
        nodes have all settled, every count is extended exactly: at its
        long-run rate, or a period at a time.
        """
        ends = np.asarray(ends, dtype=float).ravel()
        horizon = float(ends.max()) if ends.size else 0.0
        settle, period = _settling(curves.values())
        reach = min(horizon, settle if period is None else settle + period)
        beyond = ends > reach
        atoms = {node: curve.atoms(reach) for node, curve in curves.items()}
        jumps = self._atom_groups(curves, atoms, working_nodes, broken_nodes)
        # The events at exact times start pieces; the ends need not (see
        # ``totals_at``), so that a nested RBD counted at many times costs
        # no more pieces.
        fixed = [jumps.times] + [own.times for own in atoms.values()]
        cycles = rest = np.empty(0)
        if period is not None and beyond.any():
            cycles = np.floor((ends[beyond] - settle) / period)
            rest = np.clip(
                ends[beyond] - settle - cycles * period, 0.0, period
            )
            fixed += [np.array([settle]), settle + rest]

        # Each node's events at exact times, kept apart (taken together,
        # below): their running totals, by time.
        exact = {}
        for node, own in atoms.items():
            order = np.argsort(own.times, kind="stable")
            exact[node] = (
                own.times[order],
                _quadrature.running(own.failure[order], len(order)),
                _quadrature.running(own.planned[order], len(order)),
            )

        def spread(node, x: np.ndarray):
            """The node's expected failures and planned outages before
            each time ``x``, less those at exact times."""
            events = curves[node].events(x)
            times, failures, planned = exact[node]
            before = np.searchsorted(times, x, side="left")
            return (
                events["failures"] - failures[before],
                events["planned"] - planned[before],
            )

        def estimate(a: np.ndarray, b: np.ndarray) -> dict:
            x, half = _quadrature.points(a, b)
            at_points = {node: curve.at(x) for node, curve in curves.items()}
            # The system's availability and every node's importance at each
            # point, in one pass.
            importance, works, fails, _, _ = self._importances(
                self._filled(at_points, len(x), working_nodes, broken_nodes)
            )
            up = works if method == "p" else 1.0 - fails
            out: Dict[Hashable, np.ndarray] = {
                "uptime": _quadrature.summed(up, half)
            }
            if nodes:
                for node, values in at_points.items():
                    out[("downtime", node)] = _quadrature.summed(
                        1.0 - values, half
                    )
            n = len(a)
            failures, planned = np.zeros(n), np.zeros(n)
            for node in curves:
                counted = spread(node, np.concatenate([a, b, x]))
                for total, values in zip((failures, planned), counted):
                    total += _quadrature.stieltjes(
                        importance[node],
                        values[:n],
                        values[n : 2 * n],  # noqa: E203
                        values[2 * n :],  # noqa: E203
                    )
            out["failures"], out["planned"] = failures, planned
            return out

        try:
            edges, finest = _quadrature.pieces(
                curves.values(), np.concatenate(fixed), reach, _MISSION_POINTS
            )
            edges, integrals = _quadrature.refined(
                estimate,
                edges,
                finest,
                _MISSION_POINTS,
                relative={"failures", "planned"},
            )
        except _quadrature.TooMany as error:
            raise NotImplementedError(
                f"Counting the expected events over [0, {reach}] takes "
                f"{error.count} pieces (the components' curves bend that "
                "often), more than the limit: estimate them by simulation, "
                "with availability() or cost()."
            ) from None
        pieces = len(edges) - 1
        # The events at exact times, at the start of the piece they fall
        # in (each is at an edge).
        starting: Dict[Any, np.ndarray] = {
            "failures": np.zeros(pieces),
            "planned": np.zeros(pieces),
        }
        if jumps.times.size:
            piece = np.searchsorted(edges, jumps.times, side="right") - 1
            np.add.at(starting["failures"], piece, jumps.failures)
            np.add.at(starting["planned"], piece, jumps.planned)
        for name, members in (groups or {}).items():
            starting[("overlaps", name)] = self._overlaps(
                atoms, members, edges
            )
        keys: List[Any] = ["uptime", "failures", "planned"]
        if nodes:
            keys += [("downtime", node) for node in curves]
        keys += [("overlaps", name) for name in groups or {}]
        series: Dict[Any, np.ndarray] = {
            key: _quadrature.running(
                integrals.get(key, np.zeros(pieces))
                + starting.get(key, np.zeros(pieces)),
                pieces,
            )
            for key in keys
        }

        def totals_at(x: np.ndarray) -> Dict[Any, np.ndarray]:
            """Each series before each time ``x`` (within ``[0, reach]``):
            at an edge, its running total; within a piece, the running
            total at its start, the events there, and its integral from
            there, by the quadrature of the piece's start to ``x``."""
            at = np.minimum(np.searchsorted(edges, x), len(edges) - 1)
            on_edge = edges[at] == x
            out = {key: series[key][at] for key in keys}
            if pieces and not on_edge.all():
                inside = ~on_edge
                k = np.searchsorted(edges, x[inside], side="right") - 1
                k = np.clip(k, 0, pieces - 1)
                partial = _quadrature.in_blocks(estimate, edges[k], x[inside])
                for key in keys:
                    out[key][inside] = (
                        series[key][k]
                        + starting.get(key, np.zeros(pieces))[k]
                        + partial.get(key, 0.0)
                    )
            return out

        # Past ``reach``: constant rates, or a period at a time.
        rates: dict = {}
        if period is None and beyond.any():
            rates = self._settled_rates(
                curves, reach, horizon, working_nodes, broken_nodes, method
            )
        out: dict = {"downtime": {}, "overlaps": {}}
        inside = totals_at(ends[~beyond])
        for key, values in series.items():
            totals = np.empty(len(ends))
            totals[~beyond] = inside[key]
            if beyond.any():
                if period is None:
                    rate = rates.get(key, 0.0)
                    totals[beyond] = values[-1] + (ends[beyond] - reach) * rate
                else:
                    base = values[np.searchsorted(edges, settle)]
                    partial = values[np.searchsorted(edges, settle + rest)]
                    totals[beyond] = (
                        base + cycles * (values[-1] - base) + (partial - base)
                    )
            if isinstance(key, tuple):
                out[key[0]][key[1]] = totals
            else:
                out[key] = totals
        return out

    def _settled_rates(
        self, curves: dict, reach, horizon, working_nodes, broken_nodes, method
    ) -> dict:
        """What ``_window_counts`` counts, per unit time, once every node's
        curve is constant: the system's availability, its failures and
        planned outages (each node's rates, from its counts' slope then,
        times its Birnbaum importance), and each node's unavailability."""
        x = np.array([horizon])
        values = {node: curve.at(x) for node, curve in curves.items()}
        filled = self._filled(values, 1, working_nodes, broken_nodes)
        rates: dict = {
            "uptime": float(
                np.ravel(self.system_probability(filled, method=method))[0]
            )
        }
        importance = self._importances(filled)[0]
        span = np.array([reach, reach + max(reach, 1.0)])
        failures = planned = 0.0
        for node, curve in curves.items():
            events = curve.events(span)
            width = span[1] - span[0]
            weight = float(importance[node][0])
            failures += weight * np.diff(events["failures"])[0] / width
            planned += weight * np.diff(events["planned"])[0] / width
            rates[("downtime", node)] = 1.0 - float(values[node][0])
        rates["failures"], rates["planned"] = failures, planned
        return rates

    @staticmethod
    def _overlaps(atoms: dict, members, edges: np.ndarray) -> np.ndarray:
        """In each piece between ``edges``, the expected number of the
        ``members``' failures and replacements at exact times that fall at
        one instant with another member's: at each such instant the
        members' expected number, less the probability of at least one
        (they are independent). A maintenance group charges one set-up per
        instant."""
        pieces = len(edges) - 1
        own = [atoms[m] for m in members if m in atoms]
        acting = [a.times[(a.failure + a.preventive) > 0.0] for a in own]
        if len(own) < 2 or not any(t.size for t in acting):
            return np.zeros(pieces)
        times = np.unique(np.concatenate(acting))
        expected = np.zeros(len(times))
        none = np.ones(len(times))
        for a in own:
            chance = np.clip(
                _matched(times, a.times, a.failure + a.preventive), 0.0, 1.0
            )
            expected += chance
            none *= 1.0 - chance
        overlap = np.maximum(expected - (1.0 - none), 0.0)
        piece = np.searchsorted(edges, times, side="right") - 1
        keep = (piece >= 0) & (piece < pieces)
        return np.bincount(piece[keep], overlap[keep], pieces)

    def _availability_curves(
        self,
        horizon: float,
        skip,
        counts: bool = False,
        stages: bool = False,
        state: Optional[dict] = None,
    ) -> dict:
        """Each node's point availability over ``[0, horizon]``, from new,
        as a curve (see ``_point_availability``), except the nodes in
        ``skip`` (held working or failed). With ``counts``, the curves
        count the nodes' expected events too, and follow them until they
        have settled as well; with ``stages``, a degrading component's
        curve follows its stages (see ``_capacity_models``). ``state``
        (checked by ``_states``) starts the nodes it names from their
        states rather than new. With limited repair crews the nodes are not
        independent: their values over time come from the crews' chain
        instead (see ``_crew_curve``)."""
        assert not self._crews_couple(), "the crews' chain, not curves"
        self._require_no_ccf("the values over time")
        state = state or {}
        curves: dict = {}
        degrading = self._capacity_models() if stages else {}
        for node, component in self.components.items():
            if node in skip:
                continue
            start = state.get(node)
            if isinstance(component, RepairableRBD):
                curves[node] = component._nested_curve(
                    horizon, counts, stages, start
                )
            elif node in self._standby:
                curves[node] = self._standby_curve(node, start)
            elif node in self._inspection:
                life = self._tested_life(node)
                if life is not None:
                    curves[node] = self._tested_life_curve(node, life, start)
                    continue
                rate, interval = self._inspected_rate(node)
                inspection = self._inspection[node]
                if inspection.partial:
                    # From new (its state is not taken: see _check_state).
                    curves[node] = PartialTestCurve(
                        rate,
                        interval,
                        inspection.offset,
                        inspection.coverage,
                        inspection.per_full_test,
                    )
                    continue
                if start is None and inspection.offset:
                    # New at 0, first tested at the offset: on a calendar
                    # whose last test was interval - offset before 0, last
                    # known up at 0.
                    curves[node] = InspectionCurve(
                        rate, interval, interval - inspection.offset, 0.0
                    )
                    continue
                if start is not None and not start.alive:
                    raise NotImplementedError(
                        f"Component {node!r} has hidden failures, repaired "
                        "at once when a test finds them, as the exact values "
                        "need: it cannot be down. Give the time since its "
                        "last test as its phase (it was found working), or "
                        "simulate it: availability(state=...)."
                    )
                phase = 0.0 if start is None else start.phase or 0.0
                # Last known up at its last test, or, put into service
                # since, when it was.
                since = (
                    phase
                    if start is None or start.stationary
                    else min(start.age, phase)
                )
                curves[node] = InspectionCurve(rate, interval, phase, since)
            else:
                curves[node] = self._unit_curve(
                    node, horizon, counts, node in degrading, start
                )
        return curves

    def _tested_life_curve(self, node, life: TestedLife, start):
        """A tested component's point availability and events from 0, for
        a life other than exponential (see ``_hidden_life``): new at 0
        (first tested at its offset, or after an interval), in its long-run
        state, or known up at its last test (or since put into service) at
        the age its state gives."""
        inspection = self._inspection[node]
        interval = inspection.interval
        if start is None:
            return TestedLifeCurve(life, inspection.offset or interval)
        # Up: its repairs take no time (see _check_state).
        phase = start.phase or 0.0
        if start.stationary:
            return TestedLifeSteady(life, phase)
        model = life.model
        # Last known up at its last test (or when put into service since),
        # at age ``known_age``; at 0 it is ``start.age`` old.
        since = min(start.age, phase)
        known_age = np.array([start.age - since])
        known_up = float(_sf_values(model.sf, known_age)[0])
        known_down = float(np.atleast_1d(model.ff(known_age))[0])

        def initial(x):
            later = start.age + np.asarray(x, dtype=float)
            return _sf_values(model.sf, later) / known_up

        def initial_down(x):
            later = start.age + np.asarray(x, dtype=float)
            if known_down <= 0.5:
                failed = np.asarray(model.ff(later), dtype=float) - known_down
            else:
                failed = known_up - _sf_values(model.sf, later)
            return np.clip(failed / known_up, 0.0, 1.0)

        first = interval - phase if phase else interval
        return TestedLifeCurve(life, first, initial, initial_down)

    def _states(self, state, forced=frozenset()) -> dict:
        """``state`` checked (see ``point_availability``): each component it
        names, its ``NodeState``, and each nested RBD, its components'
        states, in the same form; for ``"stationary"``, every component in
        its long-run state. Nodes in ``forced`` (held working or broken)
        take none."""
        if state is None:
            return {}
        if isinstance(state, str):
            if state != "stationary":
                raise ValueError(
                    "state must be a dict of NodeStates by node, or "
                    f"'stationary', got {state!r}."
                )
            return {
                node: (
                    component._states("stationary")
                    if isinstance(component, RepairableRBD)
                    else NodeState(stationary=True)
                )
                for node, component in self.components.items()
                if node not in forced
            }
        if not isinstance(state, Mapping):
            raise TypeError(
                "state must be a dict of NodeStates by node, or "
                f"'stationary', got {state!r}."
            )
        out: dict = {}
        for node, value in state.items():
            if node not in self.components:
                raise ValueError(
                    f"state names {node!r}, which is not a component of "
                    "this RBD."
                )
            if node in forced:
                raise ValueError(
                    f"Node {node!r} is held working or broken: give it no "
                    "state."
                )
            component = self.components[node]
            if isinstance(component, RepairableRBD):
                if isinstance(value, NodeState):
                    if not value.stationary:
                        raise ValueError(
                            f"Node {node!r} is a nested RBD: give its "
                            "components' states, as a dict of NodeStates "
                            "(or 'stationary')."
                        )
                    value = "stationary"
                out[node] = component._states(value)
                continue
            if value == "stationary":
                value = NodeState(stationary=True)
            if not isinstance(value, NodeState):
                raise TypeError(
                    f"The state of component {node!r} must be a NodeState, "
                    f"got {value!r}."
                )
            self._check_state(node, value)
            out[node] = value
        return out

    def _check_state(self, node, state: NodeState) -> None:
        """Raise if a component cannot be in ``state``: a phase off a
        calendar or past its interval, maintenance it does not have, an age
        it cannot be up at or a repair or maintenance that is always over
        sooner, or a state of a kind of component whose state is not
        taken."""
        schedule = self._preventive.get(node)
        inspection = self._inspection.get(node)
        if (
            inspection is not None
            and inspection.partial
            and (state.phase is not None or not state.new)
        ):
            raise NotImplementedError(
                f"Component {node!r}'s tests can miss a failure (a coverage "
                "below 1): its state (how long since its last full test, "
                "and whether a failure its tests missed is waiting) is not "
                "taken: leave it out (new)."
            )
        interval = None
        if inspection is not None:
            interval = inspection.interval
        elif schedule is not None and schedule.policy != "age":
            interval = schedule.interval
        if state.phase is not None:
            if interval is None:
                raise ValueError(
                    f"Component {node!r} is not on a calendar (block "
                    "replacement, replacement on condition, or tests): give "
                    "it no phase."
                )
            if state.phase >= interval:
                raise ValueError(
                    f"Component {node!r}: the phase, {state.phase:g}, is "
                    "the time since its last scheduled replacement or test, "
                    f"less than its interval, {interval:g}."
                )
        if state.maintenance and (
            schedule is None or schedule.duration is None
        ):
            raise ValueError(
                f"Component {node!r} has no preventive maintenance that "
                "takes time: it cannot be down for one."
            )
        if state.new or (state.stationary and node in self._standby):
            # A standby group in its long-run state starts its units'
            # chain from its long-run distribution.
            return
        for kinds, what in (
            (self._standby, "a standby group, whose units' states"),
            (self._imperfect, "repaired imperfectly: its virtual age"),
        ):
            if node in kinds:
                raise NotImplementedError(
                    f"Component {node!r} is {what} are not taken as a "
                    "state: leave it out (new)."
                )
        if state.stationary or (state.alive and not state.age):
            return
        component = self.components[node]
        for what, model in [
            ("reliability", component.reliability),
            ("repairability", component.time_to_replace),
        ]:
            if is_fixed_probability(model):
                raise NotImplementedError(
                    f"Component {node!r}: its {what} model is a probability, "
                    "not a distribution of times, so it cannot start part "
                    "way through a life or repair: leave it out (new)."
                )
        if state.alive:
            # A unit with hidden failures was known to be up only at its
            # last test, or when put into service since.
            since = 0.0
            if inspection is not None:
                since = min(state.age, state.phase or 0.0)
            age = state.age - since
            survive = _sf_values(component.reliability.sf, np.array([age]))
            if not survive[0] > 0.0:
                raise ValueError(
                    f"Component {node!r} cannot be up at age {age:g}: its "
                    "reliability there is 0."
                )
            return
        model = (
            schedule.duration  # type: ignore[union-attr]
            if state.maintenance
            else component.time_to_replace
        )
        left = _sf_values(model.sf, np.array([state.down_for]))
        if not left[0] > 0.0:
            raise ValueError(
                f"Component {node!r} cannot have been down for "
                f"{state.down_for:g}: its "
                f"{'maintenance' if state.maintenance else 'repair'} is "
                "always over by then."
            )

    def _simulation_states(self, state, forced=frozenset()) -> dict:
        """``state`` checked for a simulation (see ``availability``): as
        ``_states`` checks it, none stationary (a simulation starts from
        given states), and see ``_check_simulated``."""
        if state is None:
            return {}
        states = self._states(state, forced)
        self._check_simulated(states)
        return states

    def _check_simulated(self, states: dict) -> None:
        """Raise if a simulation cannot start from ``states`` (checked by
        ``_states``): a component in its long-run state, which only the
        exact methods take; one started from a state whose draws are not
        streamed, being its own subclass or having a model other than a
        surpyval parametric one; or more components down in a repair or
        maintenance going on than repair crews to work on them."""
        in_hand = 0
        served = set(self._crew_served()) if self._crews_limited() else set()
        for node, start in states.items():
            component = self.components[node]
            if isinstance(component, RepairableRBD):
                component._check_simulated(start)
                continue
            if start.stationary:
                raise NotImplementedError(_STATIONARY_SIMULATION)
            if start.new:
                continue
            if (
                type(component) is not NonRepairable
                or inverse_sampler(component.reliability) is None
                or inverse_sampler(component.time_to_replace) is None
            ):
                raise NotImplementedError(
                    f"Component {node!r} draws its events its own way (it is "
                    f"a {type(component).__name__}, or its models are not "
                    "surpyval parametric ones), so it cannot start part way "
                    "through a life or repair: leave it out (new)."
                )
            if not start.alive and node in served:
                in_hand += 1
        if served and in_hand > self.repair_crews:  # type: ignore[operator]
            raise ValueError(
                f"{in_hand} components are down at the start, in repairs "
                "or maintenance going on, which takes more than the "
                f"{self.repair_crews} repair crew(s): a state does not say "
                "which jobs wait."
            )

    def _curves_at(
        self, curves: dict, x: np.ndarray, working_nodes, broken_nodes, method
    ) -> np.ndarray:
        """The system's point availability at the times ``x``, from its
        nodes' curves (the forced nodes held at 1 or 0)."""
        probabilities = {node: curve.at(x) for node, curve in curves.items()}
        for node in list(self.in_or_out) + list(working_nodes | broken_nodes):
            probabilities[node] = np.ones(len(x))
        probabilities = self._probabilities_with_overrides(
            probabilities, working_nodes, broken_nodes
        )
        return np.asarray(
            self.system_probability(probabilities, method=method), dtype=float
        )

    @staticmethod
    def _uniformized(what: str, generator, start, steady, vectors):
        """A Markov chain followed over time from ``start`` (see
        ``_chain_transient.Uniformized``), or a NotImplementedError saying
        whose (``what``) cannot be."""
        try:
            return _chain_transient.Uniformized(
                generator, start, steady, vectors
            )
        except NotImplementedError as error:
            raise NotImplementedError(
                f"{what} Markov chain cannot be followed over time here: "
                f"{error}. Simulate it with availability()."
            ) from None

    def _standby_curve(self, node, start: Optional[NodeState] = None):
        """A standby group over time (see ``_chain_transient.ChainCurve``):
        its units' Markov chain from every unit ready, or, for a
        ``start`` that is stationary, from its long-run distribution."""
        life, repair = self._standby_rates(node)
        arrangement = self._standby[node]
        group = _standby_chain.chain(
            arrangement.units,
            arrangement.k,
            life,
            repair,
            arrangement.dormancy_factor,
            arrangement.switching_probability,
            self.repair_crews if self._crews_limited() else None,
        )
        if start is not None and start.stationary:
            initial = group.probabilities
        else:
            initial = np.zeros(len(group.states))
            initial[0] = 1.0
        chain = self._uniformized(
            f"Component {node!r} is a standby group, whose",
            group.generator,
            initial,
            group.probabilities,
            np.column_stack([group.up, group.failures, group.unit_failures]),
        )
        return _chain_transient.ChainCurve(chain)

    def _nested_curve(
        self,
        horizon: float,
        counts: bool = False,
        stages: bool = False,
        start=None,
    ):
        """This RBD as a node of another, over time: the structure function
        at its own nodes' curves (a ``SystemCurve``), or, with limited
        repair crews, its crews' chain (see ``_crew_curve``). ``counts``,
        ``stages`` and ``start`` are as for ``_availability_curves``."""
        if not self._crews_couple():
            inner = self._availability_curves(
                horizon, set(), counts, stages, start
            )
            return SystemCurve(self, inner, *_settling(inner.values()))
        if stages and self._has_capacity():
            self._no_crew_window(None, "the capacities")
        curve = self._crew_curve(horizon, states=start or {})
        if counts and curve.nested:
            self._no_crew_window(next(iter(curve.nested)))
        return curve

    def _crew_nested(self, forced) -> list:
        """The nested RBDs the crews' chain is followed over time with:
        those not held working or broken. Raise if there are too many."""
        nested = [
            node
            for node, component in self.components.items()
            if isinstance(component, RepairableRBD) and node not in forced
        ]
        if len(nested) > _MAX_CREW_NESTED:
            raise NotImplementedError(
                f"With limited repair crews, the availability over time is "
                "worked out for each pattern of the nested RBDs up and down: "
                f"{len(nested)} nested RBDs are more than the "
                f"{_MAX_CREW_NESTED} it takes. Simulate it with "
                "availability()."
            )
        return nested

    def _crew_vectors(
        self, chain, values: dict, working_nodes, broken_nodes, method: str
    ) -> Tuple[np.ndarray, dict]:
        """Over the states of the crews' ``chain``, with the other nodes'
        probabilities ``values`` (arrays, one entry per state): whether the
        system is up, and each node's Birnbaum importance."""
        size = len(chain.probabilities)
        own = {
            node: chain.up[:, k].astype(float)
            for k, node in enumerate(chain.nodes)
        }
        own.update(values)
        importance, works, fails, _, _ = self._importances(
            self._filled(own, size, working_nodes, broken_nodes)
        )
        up = works if method == "p" else 1.0 - fails
        return np.asarray(up, dtype=float), importance

    def _crew_failing(self, chain, importance: dict) -> np.ndarray:
        """The rate of the system's failures in each state of the crews'
        ``chain``: a component that is up fails at its rate, and takes the
        system down where it is critical (its Birnbaum ``importance``
        there, 1 or 0), as ``system_failure_frequency`` has it."""
        rates = self._crew_chain_rates()
        failing = np.zeros(len(chain.probabilities))
        for k, node in enumerate(chain.nodes):
            failing += importance[node] * chain.up[:, k] * rates[node][0]
        return failing

    def _crew_start(self, chain, states: dict) -> np.ndarray:
        """The probabilities of the crews' chain's states at 0, from the
        components' ``states`` (checked by ``_states``): every component up
        (a component's age does not matter, its life being exponential),
        each in its long-run state (all of them, ``"stationary"``), or as
        they say: a component down is in a repair (as its repair is
        exponential, however long it has taken), and none waits, so there
        can be no more down than crews."""
        named = {node: states[node] for node in chain.nodes if node in states}
        stationary = [node for node, s in named.items() if s.stationary]
        if stationary:
            if len(stationary) < len(chain.nodes):
                raise NotImplementedError(
                    "With limited repair crews, the components' long-run "
                    "states are tied together by the repair queue: start "
                    "every component in its long-run state "
                    "(state='stationary'), or give each its state."
                )
            return np.array(chain.probabilities, dtype=float)
        down = tuple(
            k
            for k, node in enumerate(chain.nodes)
            if node in named and not named[node].alive
        )
        assert self.repair_crews is not None  # limited crews only
        if len(down) > self.repair_crews:
            raise ValueError(
                f"{len(down)} components are down at the start, in repairs "
                "going on, which takes more than the "
                f"{self.repair_crews} repair crew(s): a state does not say "
                "which jobs wait."
            )
        start = np.zeros(len(chain.probabilities))
        start[chain.states.index((down, ()))] = 1.0
        return start

    def _crew_curve(
        self,
        horizon: float,
        working_nodes=frozenset(),
        broken_nodes=frozenset(),
        method: str = "p",
        states: Optional[dict] = None,
    ) -> "_chain_transient.CrewCurve":
        """The system over time with limited repair crews (see
        ``_chain_transient.CrewCurve``): the crews' chain, without the
        nodes held working or broken, from the components' ``states``
        (see ``_crew_start``), with the nested RBDs' own curves (each from
        its own state in ``states``) to ``horizon``."""
        states = states or {}
        working, broken = set(working_nodes), set(broken_nodes)
        forced = working | broken
        nested = self._require_crew_over_time(frozenset(forced))
        chain = self._crew_chain(frozenset(forced))
        size = len(chain.probabilities)
        vectors = []
        for pattern in range(2 ** len(nested)):
            up, importance = self._crew_vectors(
                chain,
                {
                    node: np.full(size, float((pattern >> j) & 1))
                    for j, node in enumerate(nested)
                },
                working,
                broken,
                method,
            )
            vectors.append(up)
        if not nested:
            # With no nested RBDs, the only pattern's importance gives the
            # rate of the system's failures, which the chain then counts.
            vectors.append(self._crew_failing(chain, importance))
        uniformized = self._uniformized(
            "The repair crews'",
            chain.generator,
            self._crew_start(chain, states),
            chain.probabilities,
            np.column_stack(vectors),
        )
        curves = {
            node: self.components[node]._nested_curve(
                horizon, start=states.get(node)
            )
            for node in nested
        }
        return _chain_transient.CrewCurve(self, uniformized, curves)

    def _no_crew_window(
        self, node=None, what: str = "the expected events"
    ) -> NoReturn:
        """Raise that with limited repair crews, ``what`` over time (from
        their chain) cannot take in a nested RBD (``node``), or (no
        ``node``) be given to another RBD that has this one nested, as
        yet."""
        where = (
            "which has no place, as yet, for those of a nested RBD (#162): "
            f"node {node!r} is one"
            if node is not None
            else "which gives none to an RBD this one is nested in, as yet "
            "(#162)"
        )
        raise NotImplementedError(
            f"With {self.repair_crews} repair crew(s) for "
            f"{len(self._crew_served())} components, {what} over time come "
            f"from the crews' Markov chain, {where}. Simulate them with "
            "availability() or cost(); the availability over time and the "
            "long-run values are exact."
        )

    def _crew_window(
        self,
        ends: np.ndarray,
        working_nodes,
        broken_nodes,
        method: str,
        states: dict,
        nodes: bool,
        groups: Optional[dict],
    ) -> Tuple[dict, dict]:
        """``_window``'s curves and counts with limited repair crews, from
        their chain over time: the system's up time and failures (the
        integral of the rate of its failures in each state, as
        ``system_failure_frequency`` has it), and each component's down
        time and failures (at its rate while it is up), each a repair. No
        component has scheduled maintenance or tests, and none fails at the
        same instant as another."""
        working, broken = set(working_nodes), set(broken_nodes)
        forced = working | broken
        self._require_crew_over_time(frozenset(forced), "window")
        chain = self._crew_chain(frozenset(forced))
        rates = self._crew_chain_rates()
        up, importance = self._crew_vectors(chain, {}, working, broken, method)
        failing = self._crew_failing(chain, importance)
        uniformized = self._uniformized(
            "The repair crews'",
            chain.generator,
            self._crew_start(chain, states),
            chain.probabilities,
            np.column_stack([up, failing, chain.up.astype(float)]),
        )
        totals = uniformized.integrals(ends)
        zero = np.zeros(len(ends))
        counts: dict = {
            "uptime": np.clip(totals[:, 0], 0.0, ends),
            "failures": np.maximum(totals[:, 1], 0.0),
            "planned": zero,
            "downtime": {},
            "overlaps": {name: zero for name in groups or {}},
        }
        curves = {}
        for k, node in enumerate(chain.nodes):
            curves[node] = _chain_transient.CrewNodeEvents(
                uniformized, 2 + k, rates[node][0]
            )
            if nodes:
                counts["downtime"][node] = np.clip(
                    ends - totals[:, 2 + k], 0.0, ends
                )
        return curves, counts

    def _crew_capacity(
        self, working_nodes, broken_nodes, states: dict
    ) -> Tuple[np.ndarray, "_chain_transient.Uniformized"]:
        """The capacity over time with limited repair crews: its levels,
        and the crews' chain followed over time with, as its vectors,
        whether the system is at each level in each state (as
        ``capacity_distribution`` averages them in the long run)."""
        working, broken = set(working_nodes), set(broken_nodes)
        forced = working | broken
        self._require_crew_over_time(frozenset(forced), "capacity")
        chain = self._crew_chain(frozenset(forced))
        size = len(chain.probabilities)
        own = {
            node: chain.up[:, k].astype(float)
            for k, node in enumerate(chain.nodes)
        }
        arrays, _ = self._node_arrays(self._filled(own, size, working, broken))
        levels, rows = self._capacity_arrays(arrays, size, {})
        uniformized = self._uniformized(
            "The repair crews'",
            chain.generator,
            self._crew_start(chain, states),
            chain.probabilities,
            rows.T,
        )
        return levels, uniformized

    def _unit_curve(
        self,
        node,
        horizon: float,
        counts: bool = False,
        stages: bool = False,
        start: Optional[NodeState] = None,
        unscheduled: bool = False,
    ):
        """A component's point availability from new over ``[0, horizon]``
        (see ``_point_availability.unit_curve``): on a grid of
        ``_POINT_STEPS`` steps over its typical up time, with its
        replacement age on the grid, and only as long as it takes to settle
        at its long-run availability (to 1e-10), after which the curve holds
        that value. With ``counts`` it counts the component's expected
        events too, until they grow at their long-run rates (to 1e-7), and
        at those rates after. With ``stages``, a degrading component's
        curve follows its stages too, until they have settled at their
        long-run shares of its up time (to 1e-8). Under block replacement,
        see ``_block_curve``. ``start`` starts it from a state rather than
        new (see ``point_availability``). ``unscheduled`` leaves out its
        preventive maintenance, and follows it to the horizon: the head of a
        block-replacement curve, up to its first block time."""
        self._require_time_models(node)
        self._require_no_condition(node)
        self._require_no_opportunities(node)
        self._require_perfect_repair(node)
        component = self.components[node]
        schedule = None if unscheduled else self._preventive.get(node)
        if schedule is not None and schedule.policy == "block":
            return self._block_curve(node, horizon, start)
        age = None if schedule is None else float(schedule.interval)
        duration = None if schedule is None else schedule.duration
        life = component.reliability
        if component.model_parameterization == "non-parametric":
            up_sf = component.reliability_function
            up_splits = np.asarray(component._knots, dtype=float)
        else:
            up_sf = life.sf
            up_splits = point_knots(life)
        repair = component.time_to_replace

        def up_cdf(x):
            return 1.0 - _sf_values(up_sf, x)

        def repair_sf(s):
            return _sf_values(repair.sf, s)

        def duration_sf(s):
            return _sf_values(duration.sf, s)

        maintenance_sf = None if duration is None else duration_sf
        try:
            if unscheduled:
                raise ValueError("followed to the horizon")
            long_run: Optional[float] = self._node_availability(node)
            rates: Optional[Tuple[float, float, float]] = (
                self._node_frequencies(node) if counts else None
            )
        except (ValueError, NotImplementedError):
            long_run, rates = None, None
        stage_model = None
        if stages:
            self._require_unscheduled_stages(node)
            stage_model = life
        if start is not None and start.stationary:
            # Long in service: in its long-run state throughout.
            if long_run is None:
                self._node_availability(node)  # raises why it has none
            failures, maintained, _ = self._node_frequencies(node)
            return SteadyCurve(
                float(long_run),  # type: ignore[arg-type]
                failures,
                maintained,
                takedown=duration is not None,
                fractions=(
                    None
                    if stage_model is None
                    else stage_model.stage_fractions()
                ),
            )
        first = None
        if _draws_start(start):
            first = self._first_unit(
                node,
                start,  # type: ignore[arg-type]
                up_sf,
                up_splits,
                age,
                stage_model,
            )
        scale = _up_scale(life, age)
        if schedule is not None:
            up, cycle, _, _ = self._maintenance_cycle(node, schedule)
        else:
            up = model_mean(life)
            cycle = up + model_mean(repair)
        horizon = max(horizon, 0.0)
        step = horizon / 1024.0 if horizon > 0.0 else np.inf
        if np.isfinite(scale):
            step = min(step, scale / _POINT_STEPS)
        if not np.isfinite(step):
            step = 1.0
        if age is not None and step < age:
            step = age / np.ceil(age / step)
        # Units that each reach their age are maintained at nearly fixed
        # times, which the curve follows until they have all but ended.
        survive = 0.0
        if age is not None and duration is not None:
            survive = float(_sf_values(up_sf, np.array([age]))[0])
        end = max(horizon, 16.0 * step)
        settles = age is None or survive ** (horizon // age) < 1e-10
        if (
            long_run is not None
            and settles
            and np.isfinite(cycle)
            and cycle > 0.0
        ):
            end = min(end, max(8.0 * cycle, 16.0 * step))
        while True:
            n = int(np.ceil(end / step))
            if n > _POINT_MAX:
                step = end / _POINT_MAX
                if age is not None and step < age:
                    step = age / np.floor(age / step)
                if np.isfinite(scale) and step > scale / 250.0:
                    raise NotImplementedError(
                        f"Component {node!r} would need more than "
                        f"{_POINT_MAX} grid points to follow its "
                        f"availability to {end} (its typical up time is "
                        f"about {scale:.3g}): estimate it by simulation, "
                        "with availability()."
                    )
                n = min(int(np.ceil(end / step)), _POINT_MAX)
            try:
                with np.errstate(divide="ignore", invalid="ignore"):
                    curve = unit_curve(
                        up_cdf,
                        up_splits,
                        repair_sf,
                        point_knots(repair),
                        step,
                        n,
                        age=age,
                        maintenance_sf=maintenance_sf,
                        maintenance_splits=(
                            () if duration is None else point_knots(duration)
                        ),
                        counts=counts,
                        stages=(
                            None
                            if stage_model is None
                            else stage_model.stage_probabilities
                        ),
                        first=first,
                    )
            except NotImplementedError as error:
                raise NotImplementedError(
                    f"Component {node!r}: {error}. Estimate its availability "
                    "by simulation, with availability()."
                ) from None
            events = curve.unit_events
            if events is not None and rates is not None:
                events.failure_rate, events.preventive_rate = rates[:2]
            if (
                curve.stages is not None
                and stage_model is not None
                and long_run is not None
            ):
                curve.stages.fractions = stage_model.stage_fractions()
            if end >= horizon:
                break
            tail = curve.times[len(curve.times) - len(curve.times) // 4 :]
            if (
                long_run is not None
                and np.all(np.abs(curve.at(tail) - long_run) < 1e-10)
                and (age is None or survive ** (end // age) < 1e-10)
                and (events is None or events.settled())
                and (curve.stages is None or curve.stages.settled())
            ):
                break
            end = min(horizon, 4.0 * end)
        curve.long_run = long_run
        if curve.unit_events is not None and long_run is None:
            # Followed to the horizon: no rates past it.
            curve.unit_events.failure_rate = None
            curve.unit_events.preventive_rate = None
        return curve

    def _block_curve(
        self, node, horizon: float, start: Optional[NodeState] = None
    ):
        """A component's point availability from new under block
        replacement, over ``[0, horizon]`` (see
        ``_block_replacement.block_availability``); in its long-run state
        (a stationary ``start``), its settled cycle from its phase on; from
        another state, its own curve up to its first block time, and the
        block replacement's from there (see ``StartedBlockCurve``)."""
        component = self.components[node]
        schedule = self._preventive[node]
        duration = schedule.duration
        stationary = start is not None and start.stationary
        knots = np.empty(0) if duration is None else point_knots(duration)
        if start is not None and not start.new and not stationary:
            length = schedule.interval - (start.phase or 0.0)
            head = self._unit_curve(
                node, length, counts=True, start=start, unscheduled=True
            )
            down_sf = None
            if not start.alive:
                down_sf = self._first_unit(
                    node, start, None, (), None, None
                ).down_sf

            def failures(u):
                return head.events(np.asarray(u, dtype=float))["failures"]

            result = block_availability(
                component.reliability,
                component.time_to_replace,
                duration,
                schedule.interval,
                max(horizon - length, 0.0),
                node,
                BlockHead(
                    length,
                    float(head.at(np.array([length]))[0]),
                    failures,
                    down_sf,
                ),
            )
            return StartedBlockCurve(head, BlockCurve(result, knots), length)
        result = block_availability(
            component.reliability,
            component.time_to_replace,
            duration,
            schedule.interval,
            np.inf if stationary else max(horizon, 0.0),
            node,
        )
        curve = BlockCurve(result, knots)
        if not stationary:
            return curve
        assert start is not None
        return ShiftedCurve(curve, curve.settle + (start.phase or 0.0))

    def _first_unit(
        self, node, start: NodeState, up_sf, up_splits, age, stage_model
    ) -> First:
        """A component's unit at 0 in state ``start`` (see
        ``_point_availability.First``): up at its age, with what is left of
        its life (the survival function ``R(a + s) / R(a)``) and its
        replacement due ``age - a`` from now; or down, with what is left of
        its repair or maintenance (``G(r + s) / G(r)``, after ``r`` so
        far). ``_check_state`` has checked that it can be in the state."""
        if start.alive:
            a = float(start.age)
            survive = float(_sf_values(up_sf, np.array([a]))[0])

            def cdf(s):
                return np.clip(
                    1.0 - _sf_values(up_sf, a + s) / survive, 0.0, 1.0
                )

            splits = np.asarray(up_splits, dtype=float) - a

            def stages(s):
                return stage_model.stage_probabilities(a + s) / survive

            return First(
                cdf,
                splits[splits > 0.0],
                None if age is None else age - a,
                stages=None if stage_model is None else stages,
            )
        if start.maintenance:
            model = self._preventive[node].duration
        else:
            model = self.components[node].time_to_replace
        r = float(start.down_for)
        left = float(_sf_values(model.sf, np.array([r]))[0])

        def down_sf(s):
            return _sf_values(model.sf, r + s) / left

        splits = point_knots(model) - r
        return First(down_sf=down_sf, down_splits=splits[splits > 0.0])

    def capacity_distribution(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> CapacityDistribution:
        """The exact long-run distribution of the system's capacity.

        Each node carries its ``capacity`` (given when the RBD was built)
        while it is up, at one level or several, and nothing while it is
        down; the system's capacity is the most that can flow through the
        nodes that are up from the input to the output. A node given no
        capacity limits nothing, unless its model has capacities of its own:
        a degrading component spends its up time in its stages in
        proportion to their mean times (by the renewal-reward theorem), and
        a nested RBD with capacities brings its own long-run
        distribution. The probability of each level is the long-run fraction of
        time the system spends at it; the capacity is positive exactly when
        the system is up, so the fraction of time it is positive is
        [`mean_availability`][repyability.RepairableRBD.mean_availability].
        With three pumps of half the demand each, one down costs nothing
        and two cost half the output.

        Exact, with no simulation: worked out from each node's long-run
        availability as ``mean_availability`` is (see
        [`RBD.system_capacity`][repyability.RBD.system_capacity]), and
        averaged over the schedules of components inspected or replaced on
        a calendar, which are down together more often than independent
        ones would be. With limited ``repair_crews``, it is averaged over
        the states of the crews' Markov chain instead (see
        ``mean_availability``).

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always being up, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being down, by default None.

        Returns
        -------
        CapacityDistribution
            The capacities the system can have, and the long-run fraction of
            time at each. Its ``meets(demand)`` is the fraction of time the
            capacity meets a demand, ``mean()`` the average capacity, and
            ``delivered_fraction(demand)`` the fraction of the demand
            delivered: the production availability.

        Raises
        ------
        ValueError
            If no node has a capacity, or as for ``mean_availability``.
        NotImplementedError
            As for ``mean_availability``, or if a degrading component is
            maintained or inspected on a schedule.

        Examples
        --------
        Three pumps of 50 each, each up 10 / 11 of the time, against a
        demand of 100:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> pump = {
        ...     "reliability": surv.Exponential.from_params([0.1]),
        ...     "repairability": surv.Exponential.from_params([1.0]),
        ... }
        >>> plant = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("s", "c"),
        ...      ("a", "t"), ("b", "t"), ("c", "t")],
        ...     {"a": pump, "b": pump, "c": pump},
        ...     capacity={"a": 50, "b": 50, "c": 50},
        ... )
        >>> capacity = plant.capacity_distribution()
        >>> capacity.levels.tolist()
        [0.0, 50.0, 100.0, 150.0]
        >>> round(capacity.meets(100), 4)  # at least two up
        0.9767
        >>> round(capacity.delivered_fraction(100), 4)
        0.988
        >>> round(capacity.mean(), 2)  # 150 * 10 / 11
        136.36
        """
        probabilities, weights = self._long_run_probabilities(
            working_nodes, broken_nodes
        )
        working_nodes = set(working_nodes or ())
        broken_nodes = set(broken_nodes or ())
        arrays, size = self._node_arrays(probabilities)
        own = {}
        for node in self._capacity_models():
            levels, shares = self._long_run_capacity(node)
            own[node] = self._forced(
                (levels, np.repeat(shares[:, None], size, axis=1)),
                node,
                working_nodes,
                broken_nodes,
            )
        levels, rows = self._capacity_arrays(arrays, size, own)
        return CapacityDistribution(levels, rows @ weights)

    def _require_capacity_models(self) -> None:
        """Raise what ``_capacity_refusal`` reports, in the same order: a
        degrading component on a schedule, here or in a nested RBD, then no
        capacity at all."""
        for node, model in self._capacity_models().items():
            if isinstance(model, RepairableRBD):
                model._require_capacity_models()
            else:
                self._require_unscheduled_stages(node)
        self._require_capacity()

    def _capacity_curves(
        self, horizon: float, working_nodes, broken_nodes, state=None
    ) -> dict:
        """The nodes' curves for the capacity over time (see
        ``point_capacity``): a degrading component's following its stages,
        and a node that takes its capacity from its model kept even when
        held working (its levels are then those it is up at)."""
        held = set(working_nodes) - set(self._capacity_models())
        return self._availability_curves(
            horizon,
            held | set(broken_nodes),
            stages=True,
            state=self._states(state, set(working_nodes) | set(broken_nodes)),
        )

    def _capacity_rows(
        self, curves: dict, x: np.ndarray, working_nodes, broken_nodes
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The distribution of the system's capacity at each of the times
        ``x``, from its nodes' curves (see ``_capacity_curves``): its
        levels, and one row of probabilities per level, one column per
        time. A node with a capacity carries it with its point availability
        there; a degrading component is in each stage with the probability
        its curve gives, and a nested RBD with capacities brings its own
        distribution."""
        size = len(x)
        values = {node: curve.at(x) for node, curve in curves.items()}
        arrays, _ = self._node_arrays(
            self._filled(values, size, working_nodes, broken_nodes)
        )
        own = {}
        for node, model in self._capacity_models().items():
            if node in broken_nodes:
                own[node] = (np.zeros(1), np.ones((1, size)))
                continue
            curve = curves[node]
            if isinstance(model, RepairableRBD):
                distribution = model._capacity_rows(
                    curve.curves, x, set(), set()
                )
            else:
                distribution = _capacity.merged(
                    np.array((0.0,) + model.capacities),
                    np.vstack([1.0 - values[node], curve.stages_at(x)]),
                )
            own[node] = self._forced(
                distribution, node, working_nodes, broken_nodes
            )
        return self._capacity_arrays(arrays, size, own)

    def point_capacity(
        self,
        x,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        state=None,
    ) -> CapacityDistribution:
        """The distribution of the system's capacity at each time ``x``,
        with every component new at 0: exact, with no simulation.

        It is ``capacity_distribution`` at a time rather than in the long
        run: the components fail and are repaired independently, so each
        is up at ``t`` with its point availability (see
        ``point_availability``), and the system's capacity distribution is
        the same exact computation at those (see
        [`RBD.system_capacity`][repyability.RBD.system_capacity]). A node
        with levels while it works is at each with the level's share of its
        availability. A degrading component (a ``DegradingNode`` as its
        reliability) is in each stage with the probability that its first
        unit is, or a unit put into service later is, by its renewals,
        solved on the grid of its availability; a nested RBD with
        capacities brings its own distribution over time. The probability
        of a capacity above 0 is ``point_availability(x)``, and the
        distribution settles at ``capacity_distribution()``;
        ``availability(demand=...)`` estimates its mean by simulation.

        With fewer ``repair_crews`` than components the components are not
        independent: the probability of each capacity is then that of the
        states of the crews' Markov chain at it, followed over time by
        uniformization (see ``point_availability``). The chain takes no
        nested RBD's capacities in, as yet (#162).

        Parameters
        ----------
        x : float or array-like
            Times, from 0 (every component new).
        working_nodes : Collection[Hashable], optional
            Nodes that are always up, by default None. One that takes its
            capacity from its model (a degrading component, a nested RBD)
            is at the levels it is up at, in proportion.
        broken_nodes : Collection[Hashable], optional
            Nodes that are always down, by default None.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        CapacityDistribution
            The capacities the system can have and their probabilities:
            one per level for a scalar ``x``, else one row per level and one
            column per time (``x`` flattened). Its ``meets(demand)``,
            ``mean()`` and ``delivered_fraction(demand)`` are per time.

        Raises
        ------
        ValueError
            If no node has a capacity, or as for ``point_availability``.
        NotImplementedError
            If a degrading component is maintained or inspected on a
            schedule, or as for ``point_availability``.

        Examples
        --------
        Three pumps of 50 each, new at 0, with failure rate 0.1 and repair
        rate 1, each up at ``t`` with probability ``1 / 1.1 + 0.1 / 1.1 *
        exp(-1.1 t)``, against a demand of 100:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> pump = {
        ...     "reliability": surv.Exponential.from_params([0.1]),
        ...     "repairability": surv.Exponential.from_params([1.0]),
        ... }
        >>> plant = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("s", "c"),
        ...      ("a", "t"), ("b", "t"), ("c", "t")],
        ...     {"a": pump, "b": pump, "c": pump},
        ...     capacity={"a": 50, "b": 50, "c": 50},
        ... )
        >>> capacity = plant.point_capacity([0.0, 1.0, 100.0])
        >>> capacity.levels.tolist()
        [0.0, 50.0, 100.0, 150.0]
        >>> capacity.meets(100).round(4).tolist()  # at least two up
        [1.0, 0.9894, 0.9767]
        >>> round(plant.capacity_distribution().meets(100), 4)  # long run
        0.9767
        """
        times = _check_times(x)
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        self._require_capacity_models()
        ends = times.ravel()
        horizon = float(ends.max()) if ends.size else 0.0
        if self._crews_couple():
            levels, chain = self._crew_capacity(
                working_nodes,
                broken_nodes,
                self._states(state, working_nodes | broken_nodes),
            )
            rows = np.clip(chain.values(ends).T, 0.0, 1.0)
        else:
            curves = self._capacity_curves(
                horizon, working_nodes, broken_nodes, state
            )
            levels, rows = self._capacity_rows(
                curves, ends, working_nodes, broken_nodes
            )
        return CapacityDistribution(
            levels, rows[:, 0] if np.ndim(x) == 0 else rows
        )

    def mission_capacity(
        self,
        t,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        state=None,
    ) -> CapacityDistribution:
        """The distribution of the system's capacity over ``[0, t]``, with
        every component new at 0: the expected fraction of the window it
        spends at each level. Exact, with no simulation.

        It is the mean of ``point_capacity`` over the window, integrated as
        ``mission_availability`` integrates the availability: by
        Gauss-Legendre quadrature on pieces cut where the components' curves
        bend, and extended exactly past the time they have settled.
        Its ``delivered_fraction(demand)`` is the expected fraction of the
        demand delivered over the window, the production availability of a
        contract period from new, which ``availability(demand=...)``
        estimates by simulation as ``delivered_fraction``; ``meets(demand)``
        is the expected fraction of the window the capacity meets the
        demand, and ``mean()`` the average capacity. As ``t`` grows it
        approaches ``capacity_distribution()``.

        Parameters
        ----------
        t : float or array-like
            The windows' lengths (one distribution for each).
        working_nodes : Collection[Hashable], optional
            Nodes that are always up, by default None (see
            ``point_capacity``).
        broken_nodes : Collection[Hashable], optional
            Nodes that are always down, by default None.
        state : dict or str, optional
            Start from the components' current states rather than new:
            ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), a nested RBD's own such
            dict for a nested RBD, or ``"stationary"`` for every component
            in its long-run state. A component left out starts new. By
            default None: every component new at 0.

        Returns
        -------
        CapacityDistribution
            The capacities and the expected fraction of each window spent
            at each: one per level for a scalar ``t``, else one row per
            level and one column per window (at a window of 0, the
            distribution at 0).

        Raises
        ------
        ValueError, NotImplementedError
            As for ``point_capacity``.

        Examples
        --------
        The three pumps above over 10 time units, against a demand of 100:
        more than in the long run, as every pump starts up.

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> pump = {
        ...     "reliability": surv.Exponential.from_params([0.1]),
        ...     "repairability": surv.Exponential.from_params([1.0]),
        ... }
        >>> plant = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("s", "c"),
        ...      ("a", "t"), ("b", "t"), ("c", "t")],
        ...     {"a": pump, "b": pump, "c": pump},
        ...     capacity={"a": 50, "b": 50, "c": 50},
        ... )
        >>> round(plant.mission_capacity(10.0).delivered_fraction(100), 4)
        0.9896
        >>> round(plant.capacity_distribution().delivered_fraction(100), 4)
        0.988
        """
        windows = _check_times(t)
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        self._require_capacity_models()
        ends = windows.ravel()
        horizon = float(ends.max()) if ends.size else 0.0
        if self._crews_couple():
            levels, chain = self._crew_capacity(
                working_nodes,
                broken_nodes,
                self._states(state, working_nodes | broken_nodes),
            )
            positive = ends > 0.0
            rows = chain.values(ends).T
            rows[:, positive] = (
                chain.integrals(ends[positive]).T / ends[positive]
            )
            rows = np.clip(rows, 0.0, 1.0)
            keep = np.any(rows != 0.0, axis=1)
            levels, rows = levels[keep], rows[keep]
            return CapacityDistribution(
                levels, rows[:, 0] if np.ndim(t) == 0 else rows
            )
        curves = self._capacity_curves(
            horizon, working_nodes, broken_nodes, state
        )
        settle, period = _settling(curves.values())
        reach = min(horizon, settle if period is None else settle + period)
        beyond = ends > reach
        fixed = [ends[~beyond]]
        cycles = rest = np.empty(0)
        if period is not None and beyond.any():
            cycles = np.floor((ends[beyond] - settle) / period)
            rest = np.clip(
                ends[beyond] - settle - cycles * period, 0.0, period
            )
            fixed += [np.array([settle]), settle + rest]

        def estimate(a: np.ndarray, b: np.ndarray) -> dict:
            # The probability of each capacity level, integrated.
            x, half = _quadrature.points(a, b)
            levels, rows = self._capacity_rows(
                curves, x, working_nodes, broken_nodes
            )
            return {
                float(level): _quadrature.summed(row, half)
                for level, row in zip(levels, rows)
            }

        try:
            edges, finest = _quadrature.pieces(
                curves.values(), np.concatenate(fixed), reach, _MISSION_POINTS
            )
            edges, integrals = _quadrature.refined(
                estimate, edges, finest, _MISSION_POINTS
            )
        except _quadrature.TooMany as error:
            raise NotImplementedError(
                f"Integrating the capacity over [0, {reach}] takes "
                f"{error.count} pieces (the components' curves bend that "
                "often), more than the limit: estimate it by simulation, "
                "with availability(demand=...)."
            ) from None
        # The levels, as ``estimate`` keyed them.
        running: Dict[float, np.ndarray] = {
            float(level): _quadrature.running(values, len(edges) - 1)
            for level, values in integrals.items()
            if isinstance(level, float)
        }
        settled: Dict[float, float] = {}
        if period is None and beyond.any():
            levels, rows = self._capacity_rows(
                curves, np.array([horizon]), working_nodes, broken_nodes
            )
            settled = {float(v): float(r) for v, r in zip(levels, rows[:, 0])}
        inside = np.searchsorted(edges, ends[~beyond])
        positive = ends > 0.0
        at_zero: Dict[float, float] = {}
        if not positive.all():
            levels, rows = self._capacity_rows(
                curves, np.zeros(1), working_nodes, broken_nodes
            )
            at_zero = {float(v): float(r) for v, r in zip(levels, rows[:, 0])}
        averages = {}
        for level in sorted(set(running) | set(settled) | set(at_zero)):
            values = running.get(level, np.zeros(len(edges)))
            totals = np.empty(len(ends))
            totals[~beyond] = values[inside]
            if beyond.any():
                if period is None:
                    totals[beyond] = values[-1] + (
                        ends[beyond] - reach
                    ) * settled.get(level, 0.0)
                else:
                    base = values[np.searchsorted(edges, settle)]
                    partial = values[np.searchsorted(edges, settle + rest)]
                    totals[beyond] = (
                        base + cycles * (values[-1] - base) + (partial - base)
                    )
            average = np.empty(len(ends))
            average[positive] = totals[positive] / ends[positive]
            average[~positive] = at_zero.get(level, 0.0)
            averages[level] = average
        levels = np.array(sorted(averages), dtype=float)
        rows = np.array([averages[level] for level in levels]).reshape(
            len(levels), len(ends)
        )
        keep = np.any(rows != 0.0, axis=1)
        levels, rows = levels[keep], rows[keep]
        return CapacityDistribution(
            levels, rows[:, 0] if np.ndim(t) == 0 else rows
        )

    def _capacity_models(self) -> dict:
        """``{node: model}`` for the nodes with no capacity entry whose model
        has a capacity distribution of its own: a nested ``RepairableRBD``
        with capacities, or a component whose reliability is a
        ``DegradingNode``."""
        out: Dict[Any, Any] = {}
        for node, component in self.components.items():
            if node in self.capacity:
                continue
            if isinstance(component, RepairableRBD):
                if component._has_capacity():
                    out[node] = component
            elif isinstance(component.reliability, DegradingNode):
                out[node] = component.reliability
        return out

    def _long_run_capacity(self, node) -> Tuple[np.ndarray, np.ndarray]:
        """The long-run distribution of the capacity of a node whose model
        has one (see ``_capacity_models``): a nested RBD's own; for a
        degrading component, down a fraction ``1 - A`` of the time and in
        each stage the rest of it in proportion to the stage's share of its
        working time."""
        component = self.components[node]
        if isinstance(component, RepairableRBD):
            distribution = component.capacity_distribution()
            return distribution.levels, distribution.probabilities
        self._require_unscheduled_stages(node)
        stages = component.reliability
        up = self._node_availability(node)
        shares = np.concatenate([[1.0 - up], up * stages.stage_fractions()])
        levels, rows = _capacity.merged(
            np.array((0.0,) + stages.capacities), shares[:, None]
        )
        return levels, rows[:, 0]

    def _follow_up(self, event: Event, source) -> Event:
        """The next event of ``event``'s component, drawn from ``source``.

        A component's ``next_event()`` gives the time *to* its next event,
        measured from ``event``. A nested RBD's gives the time *of* its next
        state change: its simulation runs on the same clock, from 0.
        """
        node = event.component
        schedule = self._preventive.get(node)
        if schedule is not None:
            return self._maintained_follow_up(event, source, schedule)
        inspection = self._inspection.get(node)
        if inspection is not None:
            return self._inspected_follow_up(event, source, inspection)
        t, status = source.next_event()
        if isinstance(self.components[node], RepairableRBD):
            return Event(t, node, status, _planned(source, status))
        return Event(event.time + t, node, status)

    def _crew_follow_up(
        self, event: Event, next_event: Event
    ) -> Optional[Event]:
        """``next_event`` as the repair crews allow (see ``_Crews``): a job
        that falls due when ``event`` takes a component down waits for a
        crew (None while it does), and a crew that finishes one at
        ``event`` starts the next waiting job, whose end is queued here."""
        crews, node = self._crews, event.component
        if crews is None or node not in crews.served:
            return next_event
        if event.status:
            if node in crews.holding:
                started = crews.release(node, event.time)
                if started is not None:
                    self._crew_started(started)
            return next_event
        if next_event.status:
            return crews.request(node, event.time, next_event)
        return next_event

    def _crew_started(self, started: Tuple[Any, float, Event]) -> None:
        """A crew has started a waiting job, ``started`` as
        ``_Crews.release`` gives it: queue the end of a component's, or
        tell its standby group of a unit's."""
        key, wait, ends = started
        if isinstance(key, _Unit):
            self._groups[key.node].crew_started(key.unit, ends.time)
            return
        ends = self._started_late(key, wait, ends)
        if ends.time < self.t_simulation:
            self._event_queue.put(ends)

    def _started_late(self, node, wait: float, ends: Event) -> Event:
        """A job of ``node`` that a crew starts ``wait`` after it fell due,
        and so ends with ``ends``: a component off-line for a test does not
        age while it waits, so its hidden failure is ``wait`` later."""
        pending = self._pending_failure.get(node)
        if pending is not None:
            self._pending_failure[node] = pending + wait
        return ends

    def _crew_served(self) -> list:
        """The jobs the repair crews work on, by key: this RBD's own
        components, not a nested RBD's (which has crews of its own), and
        each unit of a standby group (``_Unit(node, unit)``)."""
        served: list = []
        for node, component in self.components.items():
            if isinstance(component, RepairableRBD):
                continue
            arrangement = self._standby.get(node)
            if arrangement is None:
                served.append(node)
            else:
                served.extend(
                    _Unit(node, unit) for unit in range(arrangement.units)
                )
        return served

    def _crews_limited(self) -> bool:
        """Whether there are fewer repair crews than components that may
        need one, so that a job may wait."""
        return self.repair_crews is not None and self.repair_crews < len(
            self._crew_served()
        )

    def _crews_couple(self) -> bool:
        """Whether a job waiting for a repair crew can tie different nodes
        together: the crews are limited, and work on more than one node's
        jobs. With a single standby group's units the only jobs, the waiting
        stays inside the group, whose own Markov chain counts the crews (see
        ``_standby_long_run``), and the nodes stay independent."""
        if not self._crews_limited():
            return False
        owners = {
            key.node if isinstance(key, _Unit) else key
            for key in self._crew_served()
        }
        return len(owners) > 1

    def _require_unlimited_crews(
        self,
        assumes: str = "these values assume",
        advice: str = "Simulate the system with availability() or cost().",
    ) -> None:
        """Raise if a job may wait for a repair crew: components then no
        longer fail and recover independently, which ``assumes`` (what
        assumes it, and ``assume``): do ``advice`` instead."""
        if self._crews_couple():
            raise NotImplementedError(
                f"With {self.repair_crews} repair crew(s) for "
                f"{len(self._crew_served())} components, a component can "
                "wait for a crew, so the components no longer fail and "
                f"recover independently, which {assumes}. {advice}"
            )

    def _no_crew_chain(self, why: str) -> NoReturn:
        """Raise that the repair crews' Markov chain does not cover this RBD,
        because ``why``."""
        raise NotImplementedError(
            f"With {self.repair_crews} repair crew(s) for "
            f"{len(self._crew_served())} components, a component can wait "
            "for a crew, and the exact long-run values come from a Markov "
            "chain of the components' states and the repair queue: "
            f"{why}. Simulate the system with availability() or cost()."
        )

    def _crew_chain_rates(self) -> Dict[Any, Tuple[float, float]]:
        """The failure and repair rates (``inf`` for an instant repair) of
        the components the repair crews work on, for their Markov chain
        (see ``_crew_chain.py``). Raise if the chain does not cover them:
        scheduled maintenance or inspection, a life or repair time that is
        not exponential, or more states than it is solved for."""
        assert self.repair_crews is not None  # limited crews only
        for node in self._standby:
            self._no_crew_chain(
                "it has no place for a standby group, which component "
                f"{node!r} is"
            )
        rates: Dict[Any, Tuple[float, float]] = {}
        for node in self._crew_served():
            component = self.components[node]
            if node in self._preventive:
                self._no_crew_chain(
                    "it has no place for scheduled maintenance, which "
                    f"component {node!r} has"
                )
            if node in self._inspection:
                self._no_crew_chain(
                    "it has no place for inspections, which component "
                    f"{node!r} has"
                )
            if node in self._imperfect:
                self._no_crew_chain(
                    "it has no place for imperfect repair, which component "
                    f"{node!r} has"
                )
            life = _constant_rate(component.reliability)
            if life is None:
                self._no_crew_chain(
                    "it needs exponential lives (a constant failure rate), "
                    f"and component {node!r}'s is not"
                )
            repair = _repair_rate(component.time_to_replace)
            if repair is None:
                self._no_crew_chain(
                    "it needs exponential repair times (or instant repair), "
                    f"and component {node!r}'s are not"
                )
            rates[node] = (life, repair)
        count = _crew_chain.state_count(
            [self._priority.get(node, 0.0) for node in rates],
            [math.isinf(repair) for _, repair in rates.values()],
            self.repair_crews,
        )
        if count > _crew_chain.MAX_STATES:
            self._no_crew_chain(
                f"here it has {count:,} states, more than the "
                f"{_crew_chain.MAX_STATES:,} it is solved for"
            )
        return rates

    def _require_crew_chain(self) -> None:
        """Raise if the exact long-run values with limited repair crews
        cannot be computed: the Markov chain does not cover the components
        (see ``_crew_chain_rates``), or nested RBDs' calendars fall together
        (see ``_require_calendars``)."""
        self._crew_chain_rates()
        self._require_calendars()

    def _crew_chain(
        self, forced: frozenset = frozenset()
    ) -> "_crew_chain.CrewChain":
        """The Markov chain of the components the repair crews work on, and
        its long-run distribution (see ``_crew_chain.py``), leaving out the
        nodes in ``forced``: held working a node never fails, and held
        broken it is never repaired, so neither needs a crew. Solved once
        for each set of rates and kept."""
        self._require_crew_chain()
        rates = self._crew_chain_rates()
        assert self.repair_crews is not None  # limited crews only
        nodes = [node for node in rates if node not in forced]
        priorities = [self._priority.get(node, 0.0) for node in nodes]
        key = (
            self.repair_crews,
            tuple(zip(nodes, (rates[node] for node in nodes), priorities)),
        )
        cache = self.__dict__.setdefault("_crew_chains", {})
        if key not in cache:
            cache[key] = _crew_chain.solve(
                nodes,
                [rates[node][0] for node in nodes],
                [rates[node][1] for node in nodes],
                priorities,
                self.repair_crews,
            )
        return cache[key]

    def _chain_probabilities(
        self, working_nodes, broken_nodes
    ) -> Tuple[dict, np.ndarray]:
        """With limited repair crews, every node's availability in each
        state of their Markov chain (1 or 0 for the components the crews
        work on and those held working or broken, and a nested RBD's own
        long-run availability, as it has crews of its own), and the states'
        long-run probabilities: the long-run values are then averages over
        the states, as over the times of ``_long_run_grid``."""
        working_nodes = set(working_nodes or ())
        broken_nodes = set(broken_nodes or ())
        self._validate_node_overrides(working_nodes, broken_nodes)
        chain = self._crew_chain(frozenset(working_nodes | broken_nodes))
        size = len(chain.probabilities)
        out: dict = {
            node: chain.up[:, k].astype(float)
            for k, node in enumerate(chain.nodes)
        }
        for node, component in self.components.items():
            if node in working_nodes:
                out[node] = np.ones(size)
            elif node in broken_nodes:
                out[node] = np.zeros(size)
            elif node not in out:
                out[node] = np.full(size, float(component.mean_availability()))
        for node in self.in_or_out:
            out[node] = np.ones(size)
        return out, chain.probabilities

    def _chain_outage_frequencies(
        self, working_nodes, broken_nodes
    ) -> Tuple[float, float]:
        """``_outage_frequencies`` with limited repair crews, over the
        states of their Markov chain: in each, a component that is up fails
        at its constant rate, and takes the system down if it is critical
        there. A nested RBD enters through its own frequencies, as it has
        crews of its own."""
        availability, unavailability, weights = (
            self._long_run_unavailabilities(working_nodes, broken_nodes)
        )
        forced = set(working_nodes or ()) | set(broken_nodes or ())
        rates = self._crew_chain_rates()
        birnbaum = super()._birnbaum_importance(
            availability, node_failures=unavailability
        )
        failures = planned = 0.0
        for node in self.components:
            if node in forced:
                continue
            importance = np.asarray(birnbaum[node])
            if node in rates:
                life = rates[node][0]
                node_failures, node_planned = life * availability[node], 0.0
            else:
                node_failures, _, node_planned = self._node_frequencies(node)
            failures += float(weights @ (importance * node_failures))
            planned += float(weights @ (importance * node_planned))
        return failures, planned

    def _maintained_follow_up(
        self, event: Event, source, schedule: _Preventive
    ) -> Event:
        """The next event of a component under preventive maintenance."""
        node, t = event.component, event.time
        if event.inspection:
            # Inspected, and kept (replacement on condition): its failure is
            # still ahead of it.
            return self._condition_next(node, t, schedule)
        if event.preventive:
            # Replaced: the new unit has its own life ahead of it (the
            # failure drawn for the old one never happens).
            source.reset()
            if not event.status:
                # Down for maintenance: back, as new, once it is done.
                if isinstance(source, _StreamedComponent):
                    duration = source.maintenance_time()
                else:
                    duration = schedule.duration.random(1).item()
                return Event(t + duration, node, True, True)
        elif not event.status:
            # Failed: back, as new, once it is repaired.
            repair, status = source.next_event()
            return Event(t + repair, node, status)
        # Working, as new, from t.
        return self._renewal(node, t, source, schedule)

    def _renewal(
        self,
        node,
        t: float,
        source,
        schedule: _Preventive,
        life: Optional[float] = None,
        start: Optional[float] = None,
    ) -> Event:
        """The first event of a unit put into service as new at ``t``: its
        failure, or its preventive replacement if that is due first (a
        failure at the same time comes first); replaced on condition, its
        failure or its first inspection (see ``_condition_next``). Started
        from a state (see ``_started``), the unit was put into service at
        ``start``, before ``t``, and has ``life`` left."""
        if life is None:
            life, _ = source.next_event()
        if schedule.policy == "condition":
            failure = t + life
            self._in_service[node] = (
                failure,
                self._replaced_at(
                    node, t if start is None else start, failure, schedule, t
                ),
            )
            return self._condition_next(node, t, schedule)
        if start is None:
            # Back from an imperfect repair, the unit is as old as its
            # operating time since it was renewed, which age replacement
            # counts.
            start = t - source.operated if node in self._imperfect else t
        due = self._due(
            node, schedule, start if schedule.policy == "age" else t
        )
        if due < t:
            due = t  # past its age at the start: replaced at once
        if t + life <= due:
            event = Event(t + life, node, False)
        else:
            # Maintenance that takes time is a planned outage; in zero time
            # the unit is renewed in place, and stays up.
            event = Event(due, node, schedule.duration is None, True)
        if node in self._early_members:
            # Kept, should a stop of its group renew it early (see _stop).
            self._renewed_at[node] = start
            self._pending_event[node] = event
        return event

    def _own_sources(self, states: Optional[dict] = None) -> dict:
        """What the components draw their events from when no streams are
        given: themselves, but for an imperfectly repaired one, a stand-in
        keeping its virtual age (see ``_ImperfectComponent``), and for one
        started from a state in ``states`` that draws what is left of its
        life or repair, a stand-in whose next draw can be set (see
        ``_started``): drawing from their models and numpy's global
        RNG."""
        starting = [
            node
            for node, start in (states or {}).items()
            if _draws_start(start)
        ]
        if not self._imperfect and not starting:
            return self.components
        sources = dict(self.components)
        for node, imperfect in self._imperfect.items():
            sources[node] = _ImperfectComponent(
                self.components[node],
                imperfect,
                model=self._duration_model(node),
            )
        for node in starting:
            component = self.components[node]
            sources[node] = _StreamedComponent(
                _ModelDraws(component.reliability),
                _ModelDraws(component.time_to_replace),
                model=self._duration_model(node),
            )
        return sources

    def _stop(self, group, t: float, trigger=None) -> List[Event]:
        """A stop of maintenance ``group`` at ``t``, opened by ``trigger``
        (a member taken down for a failure or its scheduled replacement) or
        by the system going down: each other member that is working, and at
        least its opportunity age, is renewed now, as its scheduled
        replacement would renew it, and the failure or replacement it had
        ahead of it is cancelled. A member whose own failure or replacement
        is due at ``t`` keeps it: that joins the stop. Returns the renewals
        to queue."""
        out: List[Event] = []
        for member in self._maintenance[group].members:
            if (
                member == trigger
                or member not in self._early_members
                or member not in self._renewed_at
                or member in self._renewing
                or not self.component_status[member]
            ):
                continue
            schedule = self._preventive[member]
            if t - self._renewed_at[member] < schedule.opportunity:
                continue
            pending = self._pending_event.get(member)
            if pending is not None and pending.time <= t:
                continue  # due now, as part of this stop
            self._pending_event.pop(member, None)
            if pending is not None and pending.time < self.t_simulation:
                self._cancelled[id(pending)] = pending
            renewal = Event(t, member, schedule.duration is None, True)
            self._early[id(renewal)] = renewal
            self._renewing.add(member)
            out.append(renewal)
        return out

    def _replaced_at(
        self,
        node,
        renewed: float,
        failure: float,
        schedule: _Preventive,
        after: Optional[float] = None,
    ) -> float:
        """When a unit put into service as new at ``renewed``, and due to
        fail at ``failure``, is replaced on condition: at the first
        inspection before its failure (and the window's end), and after
        ``after`` (by default ``renewed``), at which it is more likely than
        the threshold to fail before the next, given its age (the time
        since ``renewed``); ``inf`` if at none."""
        end = min(failure, self.t_simulation)
        dues = []
        due = self._due(node, schedule, renewed if after is None else after)
        while due < end:
            dues.append(due)
            due = self._due(node, schedule, due)
        if not dues:
            return math.inf
        # Its ages at those inspections, and at the one after the last.
        ages = np.array(dues + [due]) - renewed
        likely = _failures_between(self.components[node].reliability, ages)
        above = np.flatnonzero(likely > schedule.threshold)
        return dues[above[0]] if above.size else math.inf

    def _condition_next(self, node, t: float, schedule: _Preventive) -> Event:
        """The next event, from ``t``, of a working unit replaced on
        condition: its failure, or the next inspection, which replaces the
        unit (see ``_replaced_at``) or only checks it. A failure at an
        inspection's time comes first."""
        failure, replaced = self._in_service[node]
        due = self._due(node, schedule, t)
        if failure <= due:
            return Event(failure, node, False)
        if due >= replaced:
            # Replaced, as under block replacement.
            return Event(due, node, schedule.duration is None, True)
        return Event(due, node, True, inspection=True)

    def _inspected_follow_up(
        self, event: Event, source, inspection: _Inspection
    ) -> Event:
        """The next event of a component whose failures are hidden."""
        node, t = event.component, event.time
        if not event.inspection:
            if event.status:
                # Repaired: as new, from t.
                return self._inspected_renewal(node, t, source, inspection)
            # Failed, unseen: found by the first inspection at or after t,
            # unless that one can miss it and does.
            self._pending_failure[node] = None
            found = self._finds(node, inspection, t)
            if not inspection.is_full(found) and not _test_finds(
                source, inspection.coverage
            ):
                return Event(found, node, False, inspection=True, missed=True)
            return Event(found, node, False, inspection=True)
        if event.missed:
            # A test that missed the failure: the next test misses it too,
            # unless it is a full test.
            due = self._due(node, inspection, t)
            return Event(
                due,
                node,
                False,
                inspection=True,
                missed=not inspection.is_full(due),
            )
        if event.status:
            # Tested in zero time, or back from a test: working, with its
            # failure still ahead of it.
            return self._inspected_next(node, t, inspection)
        # A test that takes time, of a failed unit (repaired once the test is
        # done) or of a working one (off-line, and not ageing, until then).
        duration = 0.0
        if inspection.duration is not None:
            if isinstance(source, _StreamedComponent):
                duration = source.maintenance_time()
            else:
                duration = inspection.duration.random(1).item()
        failure = self._pending_failure[node]
        if failure is None:
            repair, status = source.next_event()
            return Event(t + duration + repair, node, status)
        self._pending_failure[node] = failure + duration
        return Event(t + duration, node, True, inspection=True)

    def _inspected_renewal(
        self, node, t: float, source, inspection: _Inspection
    ) -> Event:
        """The first event of a unit with hidden failures put into service
        as new at ``t``."""
        life, _ = source.next_event()
        self._pending_failure[node] = t + life
        return self._inspected_next(node, t, inspection)

    def _inspected_next(
        self, node, t: float, inspection: _Inspection
    ) -> Event:
        """The next event of a working unit with hidden failures, at ``t``:
        its failure, or the next inspection if that comes first (a failure
        at the same time comes first, and is found by it). A test that takes
        time takes the unit off-line; one in zero time leaves it up."""
        failure = self._pending_failure[node]
        due = self._due(node, inspection, t)
        if failure <= due:  # type: ignore[operator]
            return Event(failure, node, False)  # type: ignore[arg-type]
        return Event(due, node, inspection.duration is None, inspection=True)

    def next_event(self, method="p", sources: Optional[dict] = None):
        """Advance the current simulation to the system's next state change.

        Advanced, event-stepping API, after ``initialize_event_queue``. It
        takes the queued component events in time order, updating each
        component's state and queueing its next event (if before
        ``t_simulation``), until the system changes state, and returns
        that change. It gives a ``RepairableRBD`` the same event API as a
        ``NonRepairable``, so it can be a node of another RBD. Unlike
        ``NonRepairable.next_event``, which returns the time *to* the
        component's next event, this returns the time *of* the change, on
        the simulation's clock.

        When no further change happens before ``t_simulation``, it returns
        ``(t_simulation, current state)`` and discards the queue, so the
        next call raises a ValueError until ``initialize_event_queue`` is
        called again. After each call ``last_change_planned`` says whether
        the change was the start of a planned outage (preventive
        maintenance that takes the system down).

        Parameters
        ----------
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``,
            the default) or the minimal cut sets (``"c"``); both give the
            same state.
        sources : dict, optional
            Internal: the same ``sources`` given to
            ``initialize_event_queue``. By default None: the components
            themselves.

        Returns
        -------
        tuple[float, bool]
            The time of the change, and the system's new state (True for
            working, False for failed).

        Raises
        ------
        ValueError
            If the event queue has not been initialised, or was used up by
            an earlier call; or if ``method`` is not ``"p"`` or ``"c"``.

        Examples
        --------
        A component that fails exactly 10 hours after each repair, and
        takes exactly 2 hours to repair:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.ExactEventTime.from_params(10),
        ...             "repairability": surv.ExactEventTime.from_params(2),
        ...         }
        ...     },
        ... )
        >>> rbd.initialize_event_queue(30.0)
        >>> changes = [rbd.next_event()]
        >>> while changes[-1][0] < 30.0:
        ...     changes.append(rbd.next_event())
        >>> changes[:4]
        [(10.0, False), (12.0, True), (22.0, False), (24.0, True)]

        The next failure, at 34, is outside the window, so the last call
        returns the end of the window and the state the system is left in:

        >>> changes[4]
        (30.0, True)
        """
        self._require_no_ccf("the simulation", nested=True)
        if not hasattr(self, "_event_queue"):
            raise ValueError("Need to initialize the event queue")
        # The components' draws come from the same sources the queue was
        # initialised with (see initialize_event_queue).
        if sources is None:
            sources = getattr(self, "_step_sources", self.components)
        new_system_state = copy(self.system_state)

        # Use a while loop to find the next time/event at which the system
        # status changes.
        while new_system_state == self.system_state:
            if self._event_queue.qsize() == 0:
                del self._event_queue
                self.last_change_planned = False
                return self.t_simulation, self.system_state

            event = self._event_queue.get()
            if self._cancelled and self._cancelled.get(id(event)) is event:
                del self._cancelled[id(event)]
                continue  # superseded by an early renewal (see _stop)
            renewing_early = False
            if self._early and self._early.get(id(event)) is event:
                del self._early[id(event)]
                self._renewing.discard(event.component)
                renewing_early = True
            group = self._groups.get(event.component)
            if group is not None:
                # A standby group's own event (see _StandbyGroup).
                if not group.holds(event):
                    continue  # superseded by an earlier one
                now, _ = group.advance(event.time)
                if now == self.component_status[event.component]:
                    continue
                self.component_status[event.component] = now
                if now != self.system_state:
                    new_system_state = self.is_system_working(
                        self.component_status, method
                    )
                continue  # the group has queued its next event
            was = self.component_status[event.component]
            self.component_status[event.component] = event.status
            node = event.component
            if node in self._member_group and not renewing_early:
                # A failure, or a scheduled replacement starting (or done in
                # place), opens a stop of its maintenance group.
                if (not event.status and not event.inspection) or (
                    event.preventive and event.status and was
                ):
                    for renewal in self._stop(
                        self._member_group[node], event.time, node
                    ):
                        self._event_queue.put(renewal)
            # Only a change against the system's state can change it (the
            # structure is coherent; see _replicate).
            if event.status != self.system_state:
                new_system_state = self.is_system_working(
                    self.component_status, method
                )
                if self.system_state and not new_system_state:
                    for name, spec in self._maintenance.items():
                        if spec.system_down:
                            for renewal in self._stop(name, event.time):
                                self._event_queue.put(renewal)

            follow = self._follow_up(event, sources[event.component])
            next_event: Optional[Event] = (
                follow
                if self._crews is None
                else self._crew_follow_up(event, follow)
            )
            # But only queue up the event if it occurs before the end
            # of the simulation
            if next_event is not None and next_event.time < self.t_simulation:
                self._event_queue.put(next_event)

        self.system_state = new_system_state
        # A system taken down by maintenance or a test is a planned outage.
        self.last_change_planned = (
            event.preventive or event.inspection
        ) and not new_system_state

        return event.time, self.system_state

    def availability(
        self,
        t_simulation: float,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        mc_samples: Optional[int] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
        *,
        tolerance: Optional[float] = None,
        confidence: float = 0.95,
        max_samples: Optional[int] = None,
        antithetic: bool = False,
        n_jobs: Optional[int] = None,
        demand: Optional[float] = None,
        engine: str = "auto",
        state=None,
        curve_points: Optional[int] = None,
        shard_map: Optional[Callable] = None,
        shard_size: Optional[int] = None,
        control_variate: bool = False,
        N: Optional[int] = None,
        max_N: Optional[int] = None,
    ) -> AvailabilityResult:
        """Simulate the system's availability over ``[0, t_simulation]``.

        Runs ``mc_samples`` independent Monte-Carlo (discrete-event)
        simulations of the system from time 0, each starting with every
        component working and new (except ``broken_nodes``), or from the
        components' current ``state``. Each component alternates failure and
        repair independently of the others and of the system state: it
        fails after a time drawn from its reliability model and is
        restored, as good as new, after a time drawn from its repairability
        model. A component under preventive maintenance is also replaced on
        its schedule (see the class docstring), down for the maintenance
        time if it takes any: a planned outage, which counts in every up
        and down time but is not a failure. A nested ``RepairableRBD`` runs
        its own simulation on the same clock, and is down while its system
        is down. The system state is re-evaluated at every component event.

        The result holds the estimated availability over time, the system's
        and every node's total up and down time, the number of system
        failures and restorations, and criticality measures that attribute
        the system's up and down time, failures and restorations to the
        nodes (see [`Criticalities`][repyability.Criticalities]). When
        something is priced (see ``has_costs``), the same simulations also
        accumulate costs, and the result's ``cost`` is a
        [`CostResult`][repyability.CostResult]; otherwise it is None. For
        exact long-run values, use ``mean_availability`` and the other
        steady-state methods.

        When nodes have capacities (see the ``capacity`` argument of the
        class), the same simulations also follow what the system can
        deliver: its capacity after every component event (a failure that
        leaves the system up can still take some of its capacity), the time
        it spends at each capacity, and the fraction of ``demand`` it
        delivers (see
        [`AvailabilityResult`][repyability.AvailabilityResult]). The
        simulation draws only whether each component is up, so a node
        working at several levels counts at each in proportion to its
        probability. Nodes taking their capacity from their models (a
        ``DegradingNode``'s stages, or a nested RBD's capacities) are not
        followed: give them a capacity, or use ``capacity_distribution`` for
        the long run.

        Parameters
        ----------
        t_simulation : float
            Length of the window each simulation covers, in the time unit
            of the component models. Events at or after it are not
            simulated.
        working_nodes : Collection[Hashable], optional
            Nodes held working for the whole window: they never fail. By
            default None.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed for the whole window: they are down from time
            0 and never repaired. By default None.
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``,
            the default) or the minimal cut sets (``"c"``); the results are
            identical.
        mc_samples : int, optional
            Number of simulations, by default 10_000.
        verbose : bool, optional
            If True, displays a progress bar of the simulations, by default
            False.
        seed : int, optional
            Seed for a reproducible run, by default None. Each random
            quantity -- a component's times to failure, its repair times,
            its maintenance or test times, a cost given as a distribution
            -- is drawn from a stream of its own, named by the component's
            place and the quantity, and seeded from ``seed``: simulation
            ``r`` draws the same numbers however many simulations the run
            has, in however many processes (``n_jobs``), and whichever
            ``engine`` runs it, and one component's draws never depend on
            another's. With None, the seed is a number drawn from numpy's
            global RNG, so seeding that with ``np.random.seed(s)`` makes
            the run the one ``seed=s`` gives; the global RNG is otherwise
            left as it was. Models whose draws cannot be streamed (other
            than surpyval's parametric ones, and the component classes'
            subclasses) draw from the global RNG, seeded afresh for each
            simulation; surpyval's non-parametric models are not
            reproducible either way.
        tolerance : float, optional
            Simulate until the mean availability over the window (the
            fraction of it the system is up) is known to within
            ``tolerance`` either side, at ``confidence``: after the first
            ``mc_samples`` simulations, and each further ``mc_samples``, the
            run stops once the half-width of the confidence interval of
            ``result.mean_availability_interval()`` is at most
            ``tolerance``, or ``max_samples`` simulations have run (then
            with a RuntimeWarning). By default None: exactly
            ``mc_samples``. A run that stops after ``n`` simulations gives
            the result of a run of ``mc_samples=n``.
        confidence : float, optional
            The confidence level ``tolerance`` is judged at, by default
            0.95.
        max_samples : int, optional
            The most simulations a run to ``tolerance`` makes, by default
            100 times ``mc_samples``.
        antithetic : bool, optional
            Run the simulations in antithetic pairs, by default False: in
            the second simulation of a pair, each stream (see ``seed``)
            draws ``1 - u`` for every uniform ``u`` it drew in the first, in
            the same order, so a failure or repair that was early in one is
            late in the other. Each simulation is still a correct one, but
            the pair's results are negatively correlated, so their mean
            varies less than two independent simulations': a narrower
            interval for the same ``mc_samples``. The pairs, not the
            simulations, are independent, and the result's intervals are
            worked out from the pairs' means. ``mc_samples`` (and
            ``max_samples``) must be even, and every draw must come from a
            stream (surpyval parametric models), else
            ``NotImplementedError``.
        n_jobs : int, optional
            Run the simulations in parallel on ``n_jobs`` CPUs (-1: all of
            them), by default None: one. The compiled engine runs on that
            many threads; the Python one sends blocks of ``PARALLEL_BLOCK``
            (250) simulations to that many processes, each with a copy of
            the RBD, which pays off for long simulations. The result is the
            same to the last bit for any ``n_jobs``, and without it.
        engine : str, optional
            What runs the simulations: ``"python"``, ``"numba"`` (compiled,
            with numba, an optional dependency: ``pip install
            "repyability[fast]"``) or ``"auto"`` (the default), which
            compiles when numba is installed, the system is one the
            compiled engine simulates, and the run is long enough to pay
            for loading it (about a third of a second from numba's cache;
            a minute or two the first time ever, while numba compiles it).
            The engines give the same
            results to the last bit. The compiled engine simulates plain
            components (surpyval parametric models) in any structure, with
            nodes held working or broken, costs, antithetic pairs and
            tolerances, under age and block replacement, with hidden
            failures found by inspections, with repair crews, with standby
            groups, with nested RBDs of up to 20 components, and with
            capacities on systems of up to 63 components; replacement on
            condition, maintenance groups, imperfect repair, other models
            and runs from a ``state`` run in Python. Another package can
            add a compiled engine of its own, which ``engine`` then takes
            by name and ``"auto"`` may prefer (see
            ``repyability.rbd.engines``). By default ``"auto"``.
        demand : float, optional
            The demand the delivered fraction is measured against, in the
            capacities' units, when nodes have capacities. By default the
            system's capacity with every component up (at its highest
            level): its design capacity. If that is unlimited, no delivered
            fraction is worked out unless a demand is given.
        state : dict, optional
            Start each simulation from the components' current states
            rather than new: ``{node: NodeState}`` (see
            [`NodeState`][repyability.NodeState]), and a nested RBD's own
            such dict for a nested RBD. A component up at an age draws what
            is left of its life given the age, one down draws what is left
            of its repair (or maintenance) given how long it has taken so
            far, from a stream of its own, and its calendar (block
            replacement, inspections) is shifted by its phase; a component
            left out starts new. Such a run is simulated in Python. By
            default None: every component new at 0.
        curve_points : int, optional
            Keep the availability over time on a grid of ``curve_points``
            steps, ``t_simulation * k / curve_points`` for ``k`` from 0,
            rather than at every time a simulated system changed state:
            the simulations count their changes in the grid's steps, so
            the curve costs ``curve_points`` counts however many run, and
            its values at the grid's times are those the full curve takes
            there, exactly (and only there). For a large run, whose full
            curve has a point at every change of every simulation. By
            default None: the full curve. Everything else in the result is
            the same either way.
        shard_map : callable, optional
            Run the simulations as shards (see ``shards``) through this
            map, wherever it sends them: ``shard_map(run_shard, shards)``
            must give back ``run_shard``'s result for each shard, in any
            order, as ``map`` does. The built-in ``map`` runs them here;
            ``concurrent.futures.ProcessPoolExecutor(...).map`` in other
            processes; Ray's, Dask's or a batch system's on other machines
            (see [Shards](guide/simulation.md#shards)). The result is the
            same to the last bit; a run to a ``tolerance`` maps a round of
            shards at a time. Each shard is simulated by ``engine`` where it
            runs, and carries the system as JSON, so the system must save
            (``to_dict``). Not with ``n_jobs``: give the map's workers the
            CPUs. By default None.
        shard_size : int, optional
            The simulations to a shard, with ``shard_map``: as ``shards``'
            ``size``, by default as many as make 1024 or more.
        control_variate : bool, optional
            Control the estimate of the mean availability by the system's
            exact twin, by default False. The twin has the same diagram,
            components and models, failing and repaired independently:
            without a limit on repair crews or maintenance groups and,
            component by component, without what the exact methods over
            time do not take (a standby group's switching, its units then
            operating together; imperfect repair; replacement on
            condition; inspections they do not take). Its mean
            availability over the window is exact (``mission_availability``,
            and ``expected_cost`` its cost), and it is simulated alongside
            the system with common random numbers, as ``compare`` does, so
            its error against its exact value shows how far the system's
            own mean is off. ``mean_availability_interval`` (and the
            cost's ``mean_interval``) then give ``mean(x) - b * (mean(twin)
            - exact)``, with the coefficient ``b`` that leaves the least
            variance: ``1 - corr**2`` times the plain mean's. A
            ``tolerance`` is judged on it, so the run stops sooner. The
            result's ``control_variate`` holds the twin's values, its exact
            value and ``b`` (see
            [`ControlVariate`][repyability.ControlVariate]); everything else
            is the simulations' own. Every draw must come from a stream
            (surpyval parametric models), the streams are laid out as
            ``compare`` lays them, so a seeded run's simulations can differ
            from those of a run without it, and it does not run with
            ``shard_map``. See
            [An exact twin](guide/simulation.md#an-exact-twin).

        N : int, optional
            Deprecated: the old name of ``mc_samples``.
        max_N : int, optional
            Deprecated: the old name of ``max_samples``.
        Returns
        -------
        AvailabilityResult
            The availability over time (``timeline``, ``availability``; on
            the grid, with ``curve_points``), the up and down totals summed
            over the simulations, the system
            failure, planned outage and restoration counts, the
            ``criticalities``, the ``cost`` and each simulation's up time
            (``uptimes``); with capacities, also the capacity over time,
            the time at each capacity and the delivered fraction.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets, if ``method`` is not ``"p"`` or
            ``"c"``, or if ``mc_samples``, ``tolerance``, ``confidence``,
            ``max_samples``, ``n_jobs`` or ``engine`` is invalid
            (``mc_samples`` odd with ``antithetic``, ``max_samples`` without
            a tolerance or below ``mc_samples``, ...); or if a ``demand`` is
            not a positive, finite number, or is
            given for an RBD with no capacities; or if ``shard_map`` is not
            callable or comes with ``n_jobs`` or ``control_variate``, or
            ``shard_size`` is not a whole number at least 1 or comes without
            ``shard_map``.
        NotImplementedError
            With ``antithetic``, if a component's draws cannot be replayed;
            with capacities, if a node takes its capacity from its model;
            with ``engine="numba"``, if the compiled engine does not
            simulate the system; with ``shard_map``, if the system cannot
            be saved as JSON; with ``control_variate``, if the system has
            no exact twin (a component's model is a probability, or its
            life is simulated) or a draw cannot come from a stream.
        ImportError
            With ``engine="numba"``, if numba is not installed.

        Examples
        --------
        Two components in series, each with MTTF 10 and MTTR 1:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> unit = {
        ...     "reliability": surv.Exponential.from_params([0.1]),
        ...     "repairability": surv.Exponential.from_params([1.0]),
        ... }
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")], {"a": unit, "b": unit}
        ... )
        >>> result = rbd.availability(t_simulation=50, mc_samples=200, seed=0)
        >>> float(result.timeline[0]), float(result.availability[0])
        (0.0, 1.0)
        >>> float(result.timeline[-1])
        50.0

        The fraction of the window the system was up is close to the
        long-run availability, ``(10 / 11) ** 2`` (over a short window it is
        a little above it on average, as every simulation starts up):

        >>> window = result.n_simulations * result.time_simulated_to
        >>> round(float(result.system_uptime) / window, 4)
        0.8303
        >>> round(rbd.mean_availability(), 4)
        0.8264

        In series the system is up only while every node is up:

        >>> oci = result.criticalities.operational_criticality_index
        >>> {node: round(float(v), 4) for node, v in oci.up.items()}
        {'a': 1.0, 'b': 1.0}
        """
        mc_samples = renamed("mc_samples", mc_samples, "N", N)
        N = 10_000 if mc_samples is None else mc_samples
        max_N = renamed("max_samples", max_samples, "max_N", max_N)

        return self._simulated(
            t_simulation,
            working_nodes,
            broken_nodes,
            method,
            N,
            verbose,
            seed,
            tolerance=tolerance,
            confidence=confidence,
            max_N=max_N,
            antithetic=antithetic,
            n_jobs=n_jobs,
            target="availability",
            demand=demand,
            engine=engine,
            state=state,
            curve_points=_curve_points(curve_points),
            shard_map=shard_map,
            shard_size=shard_size,
            control_variate=control_variate,
        )

    def simulate_timelines(
        self,
        t_simulation: float,
        mc_samples: Optional[int] = None,
        seed: Optional[int] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        antithetic: bool = False,
        engine: str = "auto",
        n_jobs: Optional[int] = None,
        start: int = 0,
    ) -> "TimelineSimulation":
        """Simulate the system over ``[0, t_simulation]`` and keep each
        simulation's up/down histories as timelines: every component's and
        the system's, with the component that caused each of the system's
        changes.

        These are the simulations
        [`availability`][repyability.RepairableRBD.availability] runs with
        the same ``seed`` (and options), kept whole rather than added up:
        so any measure of a history can be read off them (the time to the
        first system failure, the longest outage, the outages per year),
        and they merge with other timelines (see
        ``repyability.timelines``). Their histories are the event loop's,
        whichever engine makes them: each simulation's system uptime is
        ``availability``'s to the last bit.

        The event loop records them as it runs, compiled with numba where
        it can (``pip install "repyability[fast]"``), as ``availability``
        runs. On the Python engine, independent components (plain units
        with streamed models, and nested RBDs of them) have their histories
        drawn straight from their streams instead, a batch of simulations
        at once, and the system's merged from theirs (see
        [`RBD.system_timeline`][repyability.RBD.system_timeline]): a
        simulation in which components change at the same instant is run
        in the event loop, which orders them as it does.

        Parameters
        ----------
        t_simulation : float
            The window's end.
        mc_samples : int, optional
            The number of simulations. By default 1,000.
        seed : int, optional
            Seeds the simulations, as for ``availability``. By default
            unseeded.
        working_nodes : collection, optional
            Nodes held working throughout.
        broken_nodes : collection, optional
            Nodes held broken throughout.
        antithetic : bool, optional
            Simulate in antithetic pairs (``mc_samples`` even), as for
            ``availability``. By default False.
        engine : str, optional
            ``"auto"`` (the default), ``"python"`` or ``"numba"``, as for
            ``availability``; ``"auto"`` compiles a run long enough to
            repay loading numba, when it is installed and simulates the
            system. Another package's engine records no histories.
        n_jobs : int, optional
            Run on this many processes (the Python event loop) or threads
            (numba, and the streams), ``-1`` for one per CPU. The histories
            are the same however many. By default one.
        start : int, optional
            The run's first simulation to make: simulations ``start`` to
            ``start + mc_samples - 1`` of the run ``seed`` seeds (so parts
            of one run, made apart, join with ``TimelineSimulation.join``).
            Even with ``antithetic``. By default 0.

        Returns
        -------
        TimelineSimulation
            ``system`` and ``components`` as
            [`Timelines`][repyability.Timelines], one history per
            simulation, and how they were made (``engine``, ``method``).

        Raises
        ------
        ValueError
            If ``t_simulation`` is not a positive, finite time,
            ``mc_samples`` is not a positive integer (even with
            ``antithetic``), a working or broken node is unknown, the input
            or output node, or in both, ``engine`` is not one of those,
            ``start`` is negative (or odd with ``antithetic``), or ``start``
            is given without a ``seed``.
        NotImplementedError
            If the RBD has common-cause groups, which the simulation does
            not take in, as yet; with ``antithetic``, if a component's draws
            cannot be replayed; or, with ``engine="numba"``, if the compiled
            engine does not simulate the system.
        ImportError
            With ``engine="numba"``, if numba is not installed.

        Examples
        --------
        Two pumps in parallel: how long until the plant first fails, and
        which pump's failure did it:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> unit = {
        ...     "reliability": surv.Weibull.from_params([100, 1.5]),
        ...     "repairability": surv.Exponential.from_params([0.5]),
        ... }
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {"a": unit, "b": unit},
        ... )
        >>> runs = rbd.simulate_timelines(1000.0, mc_samples=2000, seed=1)
        >>> bool(runs.system.first_failure.min() > 0)
        True
        >>> sorted(runs.system.failures_by_cause())
        ['a', 'b']

        The same simulations ``availability`` runs:

        >>> result = rbd.availability(1000.0, mc_samples=2000, seed=1)
        >>> bool(np.array_equal(runs.system.uptime, result.uptimes))
        True
        """
        return _timeline_runs.simulate(
            self,
            t_simulation,
            1000 if mc_samples is None else mc_samples,
            seed,
            working_nodes,
            broken_nodes,
            antithetic,
            engine,
            n_jobs,
            start,
        )

    def simulate_chunk(
        self,
        t_simulation: float,
        start: int,
        stop: int,
        *,
        seed: int,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        antithetic: bool = False,
        demand: Optional[float] = None,
        engine: str = "auto",
        n_jobs: Optional[int] = None,
        verbose: bool = False,
        state=None,
        curve_points: Optional[int] = None,
    ) -> "SimulationChunk":
        """Run simulations ``start`` to ``stop - 1`` of the run
        ``availability(t_simulation, mc_samples=N, seed=seed, ...)`` makes,
        for any ``N`` of at least ``stop``, and return them as a chunk to
        merge with the run's others.

        Each simulation draws from streams of its own, seeded from ``seed``
        and its position in the run (see
        [Random streams](guide/simulation.md#random-streams)), so it
        comes out the same wherever and whenever it runs, by either engine,
        in any company. A run can so be split across processes, machines or
        preemptible workers: each runs its chunk, saves it
        (``SimulationChunk.to_json``) and sends it back, and
        ``availability_from_chunks`` merges the chunks into the run's
        [`AvailabilityResult`][repyability.AvailabilityResult]. Chunks of
        simulations ``0`` to ``N - 1`` give the result of
        ``availability(..., mc_samples=N)``, to the last bit: the same
        simulations, so the same per-simulation values and timeline, and
        the same totals, which are kept exactly however the run is cut
        (#151).

        Parameters
        ----------
        t_simulation : float
            The window each simulation covers, from 0.
        start : int
            The first simulation to run, by its position in the run (from
            0; even with ``antithetic``, so that a chunk holds whole pairs).
        stop : int
            The simulation after the last to run (above ``start``; even with
            ``antithetic``).
        seed : int
            The run's seed, required: chunks of a run share it.
        working_nodes : Collection[Hashable], optional
            Nodes held working, as for ``availability``.
        broken_nodes : Collection[Hashable], optional
            Nodes held broken, as for ``availability``.
        method : str, optional
            ``"p"`` or ``"c"``, as for ``availability``, by default ``"p"``.
        antithetic : bool, optional
            Simulate in antithetic pairs, as for ``availability``, by
            default False.
        demand : float, optional
            A demand on the system's capacity, as for ``availability``.
        engine : str, optional
            ``"auto"``, ``"python"`` or ``"numba"``, as for
            ``availability``: the chunk is the same either way.
        n_jobs : int, optional
            Run the chunk on several CPUs, as for ``availability``.
        verbose : bool, optional
            Show a progress bar, by default False.
        state : dict, optional
            The components' states at the start, as for ``availability``
            (the chunks of a run share it); by default None: new.
        curve_points : int, optional
            Count the curve on a grid, as for ``availability`` (the chunks
            of a run share it): the chunk then holds the grid's counts
            rather than every change. By default None: every change.

        Returns
        -------
        SimulationChunk
            The simulations' totals, with the run's settings.

        Raises
        ------
        ValueError
            If ``seed`` is None, ``start`` and ``stop`` are not whole
            numbers with ``0 <= start < stop`` (even, with ``antithetic``),
            or the other arguments are invalid, as for ``availability``.
        NotImplementedError
            As for ``availability``.

        See Also
        --------
        shards : The run's simulations as plain data, to run anywhere.

        Examples
        --------
        Two chunks of a run, merged, are the run:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "t")],
        ...     {"a": {"reliability": E([0.1]), "repairability": E([1.0])}},
        ... )
        >>> first = rbd.simulate_chunk(100.0, 0, 300, seed=1)
        >>> rest = rbd.simulate_chunk(100.0, 300, 500, seed=1)
        >>> merged = rbd.availability_from_chunks([first, rest])
        >>> whole = rbd.availability(100.0, mc_samples=500, seed=1)
        >>> bool((merged.uptimes == whole.uptimes).all())
        True
        """
        if seed is None:
            raise ValueError(
                "A chunk needs the run's seed: the chunks of a run share it."
            )
        for name, value in (("start", start), ("stop", stop)):
            if isinstance(value, bool) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(
                    f"{name} must be a whole number, got {value!r}."
                )
        if not 0 <= start < stop:
            raise ValueError(
                f"The chunk must be simulations start to stop - 1, with 0 <= "
                f"start < stop; got start={start!r}, stop={stop!r}."
            )
        if antithetic and (start % 2 or stop % 2):
            raise ValueError(
                "With antithetic pairs, start and stop must be even, so that "
                "the chunk holds whole pairs."
            )
        working = set() if working_nodes is None else set(working_nodes)
        broken = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working, broken)
        states = self._simulation_states(state, working | broken)
        return self._chunk(
            t_simulation,
            int(start),
            int(stop),
            _streams.entropy_of(seed),
            working,
            broken,
            method,
            antithetic,
            demand,
            engine,
            None if n_jobs is None else montecarlo.jobs(n_jobs),
            verbose,
            states,
            _curve_points(curve_points),
        )

    def _chunk(
        self,
        t_simulation: float,
        start: int,
        stop: int,
        entropy: int,
        working: set,
        broken: set,
        method: str,
        antithetic: bool,
        demand: Optional[float],
        engine: str,
        jobs: Optional[int],
        verbose: bool,
        states: dict,
        curve_points: Optional[int],
    ) -> "SimulationChunk":
        """Simulations ``start`` to ``stop - 1`` of the run of ``entropy``
        (see ``simulate_chunk``, whose arguments, checked, these are), as
        a chunk with the run's settings."""
        from repyability.rbd.chunks import SimulationChunk

        capacity = self._chunk_capacity(broken, method, demand)
        tally = self._run(
            t_simulation,
            working,
            broken,
            method,
            stop - start,
            verbose,
            None,
            antithetic,
            capacity=capacity,
            jobs=jobs,
            engine=engine,
            entropy=entropy,
            first=start,
            states=states,
            curve_points=curve_points,
        )
        settings = {
            "t_simulation": float(t_simulation),
            "entropy": entropy,
            "method": method,
            "working_nodes": sorted(working, key=repr),
            "broken_nodes": sorted(broken, key=repr),
            "antithetic": bool(antithetic),
            "demand": None if demand is None else float(demand),
            "fingerprint": self._fingerprint(),
            "state": _state_key(states),
            "curve_points": tally.curve_points,
        }
        return SimulationChunk([(start, stop)], settings, tally)

    def _chunk_capacity(
        self, broken: set, method: str, demand: Optional[float]
    ) -> Optional[_CapacityRecorder]:
        """Check a chunk's (or shard's) ``method`` and ``demand``, and give
        the recorder of the capacities it follows, if the system has
        them."""
        self.is_system_working(
            {c: c not in broken for c in self.components}, method
        )
        if self._has_capacity():
            return _CapacityRecorder(self, demand)
        if demand is not None:
            raise ValueError(
                "A demand is measured against capacities, and no node has "
                "one: give them with capacity={node: capacity}."
            )
        return None

    def shards(
        self,
        t_simulation: float,
        mc_samples: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        size: Optional[int] = None,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        antithetic: bool = False,
        demand: Optional[float] = None,
        engine: str = "auto",
        state=None,
        curve_points: Optional[int] = None,
    ) -> List[bytes]:
        """The run ``availability(t_simulation, mc_samples, seed=seed,
        ...)`` makes, as shards: ranges of its simulations as plain data,
        to run anywhere (see ``repyability.rbd.shards``).

        Each shard is JSON: this system as ``to_dict`` gives it, the run's
        entropy (drawn once, here, from the seed) and settings, and its
        range. ``repyability.run_shard`` runs one in any process, on any
        machine and by any executor (``concurrent.futures``, Ray, Dask, a
        batch system with ``python -m repyability.rbd.shards < shard.json
        > partial.npz``), and gives back its partial, the simulations'
        totals as ``.npz`` bytes; ``availability_from_chunks(partials,
        mc_samples)`` puts them together, in any order, into the run's
        result, to the last bit. ``availability(..., shard_map=...)`` does
        it all through a ``map`` of your choosing.

        Parameters
        ----------
        t_simulation : float
            The window each simulation covers, from 0.
        mc_samples : int, optional
            The run's simulations, by default 10_000.
        seed : int, optional
            The run's seed, by default None: drawn from numpy's global RNG,
            as ``availability`` does.
        size : int, optional
            Simulations to a shard, rounded up to a whole number of the
            run's widest block of draws (see
            [Random streams](guide/simulation.md#random-streams)), so that
            no two shards draw the same block; by default as many as make
            1024 or more. A shard should run for some seconds, to repay a
            worker's start.
        working_nodes : Collection[Hashable], optional
            Nodes held working, as for ``availability``.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed, as for ``availability``.
        method : str, optional
            ``"p"`` or ``"c"``, as for ``availability``.
        antithetic : bool, optional
            Run the simulations in antithetic pairs, as for
            ``availability``: ``mc_samples`` must then be even.
        demand : float, optional
            The demand the delivered fraction is measured against, as for
            ``availability``.
        state : dict, optional
            The components' states at the start, as for ``availability``.
        engine : str, optional
            What runs each shard: as for ``availability``, in the worker,
            by default ``"auto"``.
        curve_points : int, optional
            Count the curve on a grid, as for ``availability``, so that a
            partial carries the grid's counts rather than every change. By
            default None.

        Returns
        -------
        list of bytes
            The shards, in order of their simulations, as JSON.

        Raises
        ------
        ValueError
            If an argument is invalid, as for ``availability`` (with
            ``antithetic``, ``mc_samples`` must be even).
        NotImplementedError
            If this system cannot be saved as JSON (a model that is not a
            surpyval parametric one), as a shard carries it: run it with
            ``availability(..., n_jobs=...)`` instead.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "t")],
        ...     {"a": {"reliability": E([0.1]), "repairability": E([1.0])}},
        ... )
        >>> len(rbd.shards(100.0, 10_000, seed=1, size=4096))
        3
        """
        N = 10_000 if mc_samples is None else mc_samples
        montecarlo.check_count(N, antithetic, "mc_samples")
        working = set() if working_nodes is None else set(working_nodes)
        broken = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working, broken)
        states = self._simulation_states(state, working | broken)
        capacity = self._chunk_capacity(broken, method, demand)
        self._require_no_ccf("the simulation", nested=True)
        t_simulation = _check_window(t_simulation)
        entropy = _streams.entropy_of(seed)
        template = self._shard_template(
            t_simulation,
            entropy,
            working,
            broken,
            method,
            antithetic,
            demand,
            engine,
            states,
            _curve_points(curve_points),
        )
        # What a run checks before it simulates (see _run).
        plan, complete = self._stream_plan(
            t_simulation, entropy, antithetic, None, states
        )
        if antithetic and not complete:
            raise NotImplementedError(_UNSTREAMED)
        self._simulation_engine(
            engine, plan, capacity, N, here=False, states=states
        )
        step = self._shard_size(plan, size)
        return [
            _shard_bytes(template, first, min(first + step, N))
            for first in range(0, N, step)
        ]

    def _shard_template(
        self,
        t_simulation: float,
        entropy: int,
        working: set,
        broken: set,
        method: str,
        antithetic: bool,
        demand: Optional[float],
        engine: str,
        states: dict,
        curve_points: Optional[int],
    ) -> dict:
        """What every shard of a run holds (see ``shards``): all but its
        range."""
        from repyability.rbd.shards import KIND

        return {
            "kind": KIND,
            "repyability_version": __version__,
            "system": self._shard_system(),
            "t_simulation": float(t_simulation),
            "entropy": int(entropy),
            "method": method,
            "working_nodes": sorted(working, key=repr),
            "broken_nodes": sorted(broken, key=repr),
            "antithetic": bool(antithetic),
            "demand": None if demand is None else float(demand),
            "engine": engine,
            "state": _state_key(states),
            "curve_points": curve_points,
        }

    def _shard_system(self) -> dict:
        """This system as a shard carries it: ``to_dict``'s JSON data, with
        every model of a class that loads back as itself (see
        ``serialisation.exactly``), so that a worker simulates this
        system."""
        from repyability.rbd.serialisation import exactly

        try:
            with exactly():
                system = self.to_dict()
            json.dumps(system)
            if type(self) is not RepairableRBD:
                raise NotImplementedError(
                    f"it is a {type(self).__name__}, which loads as a "
                    "RepairableRBD"
                )
        except Exception as error:
            raise NotImplementedError(
                "A shard carries the system as JSON, and this one cannot be "
                f"saved: {str(error).rstrip('.')}. Run it with "
                "availability(n_jobs=...) instead."
            ) from None
        return system

    @staticmethod
    def _shard_size(plan: _streams.Plan, size: Optional[int]) -> int:
        """Simulations to a shard of the run of ``plan``: ``size`` (by
        default 1024), rounded up to a whole number of its widest block of
        draws, so that no two shards draw the same block."""
        wanted = 1024 if size is None else size
        if (
            isinstance(wanted, bool)
            or not isinstance(wanted, (int, np.integer))
            or wanted < 1
        ):
            raise ValueError(
                "The size of a shard must be a whole number of simulations, "
                f"at least 1; got {size!r}."
            )
        widest = max(
            [plan.columns(spec) for spec in plan.specs.values()]
            + [2 if plan.antithetic else 1]
        )
        return int(-(-int(wanted) // widest) * widest)

    def availability_from_chunks(
        self,
        chunks,
        mc_samples: Optional[int] = None,
        *,
        allow_gaps: bool = False,
    ) -> AvailabilityResult:
        """The result of the simulations of ``chunks`` (see
        ``simulate_chunk``, and the partials ``run_shard`` gives back from
        ``shards``), as ``availability`` gives it: merged, in order of
        their positions in the run, whatever order they come in.

        Chunks of simulations ``0`` to ``N - 1`` give the result of
        ``availability(t_simulation, mc_samples=N, seed=seed, ...)``: the
        same per-simulation values (``uptimes``, the costs' ``samples``) and
        timeline, and the same totals, to the last bit: every total is kept
        exactly and rounded once, so it does not depend on how the run was
        cut (#151).
        The chunks must hold simulations ``0`` to ``N - 1`` with none
        missing, so a lost chunk (a shard that never came back) is not
        taken for a smaller run (#176); with ``allow_gaps=True`` they give
        the result of whichever simulations they hold. With costs, the
        result's ``cost`` is the simulated cost distribution, as ``cost``
        gives it.

        Parameters
        ----------
        chunks : SimulationChunk, or an iterable of them
            The chunks, their ``to_dict`` data, or their ``to_npz`` bytes
            (a shard's partial): of one run of this system, holding
            different simulations.
        mc_samples : int, optional
            The run's number of simulations: the chunks must then hold
            simulations ``0`` to ``mc_samples - 1``, all of them, which
            also refuses a missing last chunk. By default None: simulations
            ``0`` to however many they hold.
        allow_gaps : bool, optional
            Take chunks with simulations missing between or before them,
            for the result of those they hold. By default False.

        Returns
        -------
        AvailabilityResult
            The result of their simulations.

        Raises
        ------
        ValueError
            If the chunks are of different runs, of another system (or of
            it saved by another RePyability version), or their simulations
            overlap or interleave; or, unless ``allow_gaps``, some
            simulations are missing between or before them; or, with
            ``mc_samples``, they do not hold simulations ``0`` to
            ``mc_samples - 1``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD, SimulationChunk
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "t")],
        ...     {"a": {"reliability": E([0.1]), "repairability": E([1.0])}},
        ... )
        >>> saved = rbd.simulate_chunk(100.0, 0, 200, seed=3).to_json()
        >>> chunk = SimulationChunk.from_json(saved)
        >>> rbd.availability_from_chunks(chunk).n_simulations
        200
        """
        from repyability.rbd.chunks import SimulationChunk

        if isinstance(chunks, (SimulationChunk, dict, bytes, bytearray)):
            chunks = [chunks]
        chunk = SimulationChunk.merge(chunks)
        if mc_samples is not None and chunk.ranges != [(0, int(mc_samples))]:
            raise ValueError(
                f"The chunks hold simulations {chunk.ranges} (as "
                "(start, stop) ranges), not all of simulations 0 to "
                f"{int(mc_samples) - 1}: some are missing."
            )
        if not allow_gaps and chunk.ranges != [(0, chunk.n_simulations)]:
            held = ", ".join(f"{a} to {b - 1}" for a, b in chunk.ranges)
            raise ValueError(
                f"The chunks hold simulations {held}: those between or "
                "before them are missing (a chunk that never came back?). "
                "Give them all, or pass allow_gaps=True for the result of "
                "the simulations held."
            )
        settings = chunk.settings
        fingerprint = self._fingerprint()
        if (
            settings["fingerprint"] is not None
            and fingerprint is not None
            and settings["fingerprint"] != fingerprint
        ):
            raise ValueError(
                "The chunks were simulated with another system (or with "
                "this one saved by another RePyability version)."
            )
        broken = set(settings["broken_nodes"])
        capacity = None
        if self._has_capacity():
            capacity = _CapacityRecorder(self, settings["demand"])
        initial_up = bool(
            self.is_system_working(
                {c: c not in broken for c in self.components},
                settings["method"],
            )
        )
        return self._availability_result(
            chunk._tally,
            settings["t_simulation"],
            initial_up,
            settings["antithetic"],
            capacity,
        )

    def _fingerprint(self) -> Optional[str]:
        """A hash of this system saved as JSON (with the RePyability
        version), which chunks of its runs carry; None if it cannot be
        saved."""
        try:
            text = json.dumps(self.to_dict(), sort_keys=True)
        except Exception:
            return None
        return hashlib.sha256(text.encode()).hexdigest()

    def compare(
        self,
        other: "RepairableRBD",
        t_simulation: float,
        mc_samples: Optional[int] = None,
        seed: Optional[int] = None,
        *,
        quantity: str = "availability",
        confidence: float = 0.95,
        n_jobs: Optional[int] = None,
        engine: str = "auto",
        state=None,
        N: Optional[int] = None,
    ) -> ConfidenceInterval:
        """How much better (or worse) this system is than ``other``, by
        simulation with common random numbers.

        Both systems are simulated ``mc_samples`` times over
        ``[0, t_simulation]`` (every component working and new at the start,
        or from ``state``), and in each simulation a
        component in the same place in both (the same node name, and the
        same names down through nested RBDs) draws the same random numbers
        in both: the same failures and repairs where it is modelled the
        same way, and matching ones (the same quantiles of its own models)
        where it is not. The differences between the two systems' results
        then come from how the systems differ rather than from chance, so
        their mean is a more precise estimate of the difference than the
        difference of two independent simulations of the same size: much
        more, when the systems differ in a component's models and share
        the rest.

        Parameters
        ----------
        other : RepairableRBD
            The system to compare with.
        t_simulation : float
            The window each simulation covers.
        mc_samples : int, optional
            The number of simulations of each system, by default 10_000.
        seed : int, optional
            Seed for a reproducible comparison, by default None: a number
            drawn from numpy's global RNG (see ``availability``).
        quantity : str, optional
            ``"availability"`` (the default): the fraction of the window the
            system is up. ``"cost"``: its total cost over the window (both
            systems must be priced; a cost given as a distribution draws
            the same numbers in both too, from a stream of its own).
        confidence : float, optional
            The confidence level of the interval, by default 0.95.
        n_jobs : int, optional
            Run the simulations in parallel on ``n_jobs`` CPUs (see
            ``availability``), by default None: one.
        engine : str, optional
            What runs the simulations: ``"python"``, ``"numba"`` or
            ``"auto"`` (the default), as in ``availability``.
        state : dict, optional
            The components' states at the start, as for ``availability``,
            in both systems: each must have the components it names. By
            default None: new.

        N : int, optional
            Deprecated: the old name of ``mc_samples``.
        Returns
        -------
        ConfidenceInterval
            The mean difference (this system's quantity minus ``other``'s)
            over the simulations, with its standard error and a normal
            confidence interval (not clipped: the difference may be
            negative).

        Raises
        ------
        ValueError
            If ``quantity``, ``mc_samples``, ``confidence``, ``n_jobs`` or
            ``engine`` is invalid, or a system to compare by cost has no
            costs.
        NotImplementedError
            If a component's draws cannot be replayed from a stream of its
            own (a non-parametric model, for example), or, with
            ``engine="numba"``, the compiled engine does not simulate a
            system.
        ImportError
            With ``engine="numba"``, if numba is not installed.

        Examples
        --------
        Two pumps in parallel, each failing about every 10 hours: how much
        more of a 100-hour window is the system up if a repair takes 1 hour
        on average instead of 2?

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> def pumps(mttr):
        ...     pump = {
        ...         "reliability": surv.Exponential.from_params([0.1]),
        ...         "repairability": surv.Exponential.from_params([1 / mttr]),
        ...     }
        ...     return RepairableRBD(
        ...         [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")],
        ...         {"p1": pump, "p2": pump},
        ...     )
        >>> gain = pumps(1.0).compare(
        ...     pumps(2.0), 100.0, mc_samples=2000, seed=0
        ... )
        >>> round(gain.estimate, 4), round(gain.standard_error, 5)
        (0.0196, 0.00042)

        The exact difference is 0.0189 (each pump is up
        ``mu / (lambda + mu) + lambda / (lambda + mu) * exp(-(lambda + mu) t)``
        of the time at ``t``, averaged over the window). Two independent
        runs of 2000 simulations would estimate it with a standard error of
        about 0.00058.
        """
        mc_samples = renamed("mc_samples", mc_samples, "N", N)
        N = 10_000 if mc_samples is None else mc_samples
        t_simulation = _check_window(t_simulation)

        if quantity not in ("availability", "cost"):
            raise ValueError(
                "quantity must be 'availability' or 'cost', got "
                f"{quantity!r}."
            )
        montecarlo.check_confidence(confidence)
        montecarlo.check_count(N, False, "mc_samples")
        if quantity == "cost":
            for rbd in (self, other):
                if not rbd.has_costs:
                    raise ValueError(
                        "Both systems must be priced to compare their costs."
                    )
        jobs = None if n_jobs is None else montecarlo.jobs(n_jobs)
        states = [rbd._simulation_states(state) for rbd in (self, other)]
        entropy = _streams.entropy_of(seed)
        widths = self._common_widths(other, t_simulation, *states)
        values = []
        for rbd, start in zip((self, other), states):
            tally = rbd._run(
                t_simulation,
                set(),
                set(),
                "p",
                N,
                False,
                None,
                jobs=jobs,
                engine=engine,
                entropy=entropy,
                widths=widths,
                common=True,
                states=start,
            )
            if quantity == "cost":
                values.append(np.asarray(tally.cost_samples, dtype=float))
            else:
                values.append(np.asarray(tally.uptimes) / t_simulation)
        differences = values[0] - values[1]
        estimate = float(np.mean(differences))
        standard_error = montecarlo.standard_error(differences, False)
        z = montecarlo.z_value(confidence)
        return ConfidenceInterval(
            estimate=estimate,
            lower=estimate - z * standard_error,
            upper=estimate + z * standard_error,
            confidence=confidence,
            standard_error=standard_error,
            n_samples=N,
        )

    def _common_widths(
        self,
        other: "RepairableRBD",
        t_simulation: float,
        states: Optional[dict] = None,
        other_states: Optional[dict] = None,
    ) -> dict:
        """The block widths two systems' streams take in ``compare``: for
        each stream both have, the narrower of the two, so that each
        simulation of either draws the same uniforms from it. The systems
        start from ``states`` and ``other_states``."""
        mine, _ = self._stream_specs(t_simulation, states=states)
        theirs, _ = other._stream_specs(t_simulation, states=other_states)
        return {
            name: min(spec.width, theirs[name].width)
            for name, spec in mine.items()
            if name in theirs
        }

    def _twin(self) -> Tuple["RepairableRBD", List[str]]:
        """This system's exact twin (#154), and what it changes, in words.

        The twin has this system's diagram and components, with their
        models, failing and repaired independently: without the limit on
        repair crews or the maintenance groups, which tie components
        together, and, component by component, without what the exact
        methods over time do not take (see ``_twin_simpler``). Its expected
        values over a window are then exact (``mission_availability``,
        ``expected_cost``), and each of its streams is named as this
        system's is (a standby group's units as the twin's nested units),
        so that, simulated with common random numbers, it follows this
        system closely: see ``availability``'s ``control_variate``.

        Raises
        ------
        NotImplementedError
            If no such twin has exact values over time: a component's model
            is a probability, or a life is simulated.
        """
        from repyability.rbd import routes as r

        args = self._init_args
        changes = []
        if self._crews_limited():
            changes.append("the limit on repair crews")
        if self._maintenance:
            changes.append("the maintenance groups")
        specs: Dict[Any, Any] = {}
        for node, value in args["components"].items():
            if isinstance(value, RepairableRBD):
                nested, inner = value._twin()
                specs[node] = nested
                changes.extend(f"{what} in {node!r}" for what in inner)
            elif isinstance(value, dict):
                spec = {
                    key: item
                    for key, item in value.items()
                    if key not in ("priority", "group")
                }
                preventive = value.get("preventive")
                if preventive and "opportunity" in preventive:
                    # Opportunities come from a group's stops.
                    spec["preventive"] = {
                        key: item
                        for key, item in preventive.items()
                        if key != "opportunity"
                    }
                specs[node] = spec
            else:
                specs[node] = value
        twin = self._twin_from(specs)
        simpler = {}
        for node, spec in specs.items():
            if isinstance(spec, dict):
                simpler[node], change = twin._twin_simpler(node, spec)
                if change:
                    changes.append(change)
        if any(simpler[node] is not specs[node] for node in simpler):
            twin = self._twin_from({**specs, **simpler})
        route = twin._over_time()
        if route.route == r.REFUSED:
            raise NotImplementedError(
                "This system has no exact twin to control its simulation "
                f"by: {route.reason}"
            )
        if route.route == r.SIMULATED:
            raise NotImplementedError(
                "This system has no exact twin to control its simulation "
                "by: its components' values over time are simulated "
                f"({route.reason})"
            )
        return twin, changes

    def _twin_report(self) -> str:
        """What ``analysis_routes`` says of the exact twin a run with
        ``control_variate`` is controlled by (see ``AnalysisRoute.twin``):
        what it leaves out of this system, or why there is none."""
        try:
            _, changes = self._twin()
        except NotImplementedError as error:
            return f"none. {error}"
        _, complete = self._stream_specs(1.0)
        if not complete:
            return f"none. {_UNSTREAMED}"
        if not changes:
            return (
                "the system itself, whose values over the window are exact "
                "(mission_availability, expected_cost)."
            )
        return f"the system without {', '.join(changes)}."

    def _twin_from(self, components: dict) -> "RepairableRBD":
        """A system of this one's diagram with ``components`` (see
        ``_twin``)."""
        args = self._init_args
        return RepairableRBD(
            args["edges"],
            components,
            k=args["k"],
            input_node=args["input_node"],
            output_node=args["output_node"],
            on_infeasible_rbd=args["on_infeasible_rbd"],
            downtime_cost_rate=args["downtime_cost_rate"],
        )

    def _twin_simpler(self, node, spec: dict) -> Tuple[Any, str]:
        """A component's spec in an exact twin (see ``_twin``), this system
        being the twin so far, and what it changes (or ""): a standby group
        whose own chain cannot follow it over time becomes its units
        operating together; imperfect repair becomes perfect; replacement
        on condition, inspections and block replacement that the exact
        methods do not take go."""
        from repyability.rbd import routes as r

        if node in self._standby:
            if r.refusal(partial(self._standby_rates, node)):
                return (
                    _hot_units(spec),
                    f"the switching of standby group {node!r} (its units "
                    "operate together)",
                )
            return spec, ""
        drop: Dict[str, str] = {}
        if r.refusal(partial(self._require_perfect_repair, node)):
            drop["repair"] = drop["replace_after"] = "imperfect repair"
        if r.refusal(partial(self._require_no_condition, node)):
            drop["preventive"] = "replacement on condition"
        if node in self._inspection and r.refusal(
            partial(self._require_tested_exact, node)
        ):
            drop["inspection"] = "inspections"
        schedule = self._preventive.get(node)
        if (
            schedule is not None
            and schedule.policy == "block"
            and r.refusal(partial(self._require_block_models, node))
        ):
            drop["preventive"] = "block replacement"
        if not drop:
            return spec, ""
        what = list(dict.fromkeys(drop[key] for key in drop if key in spec))
        note = " (its failures are revealed)" if "inspection" in drop else ""
        return (
            {key: value for key, value in spec.items() if key not in drop},
            f"the {' and '.join(what)} of {node!r}{note}",
        )

    def _simulated(
        self,
        t_simulation: float,
        working_nodes,
        broken_nodes,
        method: str,
        N: int,
        verbose: bool,
        seed,
        *,
        tolerance: Optional[float],
        confidence: float,
        max_N: Optional[int],
        antithetic: bool,
        n_jobs: Optional[int],
        target: str,
        demand: Optional[float] = None,
        engine: str = "auto",
        state=None,
        curve_points: Optional[int] = None,
        shard_map: Optional[Callable] = None,
        shard_size: Optional[int] = None,
        control_variate: bool = False,
    ) -> AvailabilityResult:
        """``availability`` (and ``cost``): validate, run the replications
        (serially, in parallel, or as shards through ``shard_map``, until
        converged if asked; with ``control_variate``, alongside the system's
        exact twin) and build the result, its curve on a grid of
        ``curve_points`` steps if given."""
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        states = self._simulation_states(state, working_nodes | broken_nodes)
        # The initial system state with every component up but those forced
        # broken, which can make the system start down; a simulation that
        # starts down from the components' states records the change at 0
        # (see _replicate).
        initial_status = {c: c not in broken_nodes for c in self.components}
        initial_up = bool(self.is_system_working(initial_status, method))
        montecarlo.check_count(N, antithetic, "mc_samples")
        stop = _stopping_rule(
            N, tolerance, confidence, max_N, antithetic, target, t_simulation
        )
        _check_shard_map(shard_map, shard_size, n_jobs)
        if not isinstance(control_variate, (bool, np.bool_)):
            raise ValueError(
                f"control_variate must be True or False, got "
                f"{control_variate!r}."
            )
        twin = None
        if control_variate:
            if shard_map is not None:
                raise ValueError(
                    "A run with control_variate simulates the system's twin "
                    "alongside it, here: leave out shard_map."
                )
            twin, _ = self._twin()
        capacity = None
        # Shards follow the capacities whatever the target, as chunks do.
        if (
            target == "availability" or shard_map is not None
        ) and self._has_capacity():
            capacity = _CapacityRecorder(self, demand)
        elif demand is not None:
            raise ValueError(
                "A demand is measured against capacities, and no node has "
                "one: give them with capacity={node: capacity}."
            )
        entropy = sharded = None
        if shard_map is not None:
            self._require_no_ccf("the simulation", nested=True)
            entropy = _streams.entropy_of(seed)
            template = self._shard_template(
                t_simulation,
                entropy,
                working_nodes,
                broken_nodes,
                method,
                antithetic,
                demand,
                engine,
                states,
                curve_points,
            )
            plan, _ = self._stream_plan(
                t_simulation, entropy, antithetic, None, states
            )
            step = self._shard_size(plan, shard_size)
            sharded = (shard_map, template, step)
        jobs = None if n_jobs is None else montecarlo.jobs(n_jobs)
        controls: Tuple[Optional[ControlVariate], ...] = (None, None)
        if twin is not None:
            tally, controls = self._controlled_run(
                twin,
                t_simulation,
                working_nodes,
                broken_nodes,
                method,
                N,
                verbose,
                seed,
                antithetic,
                stop,
                capacity=capacity,
                jobs=jobs,
                engine=engine,
                state=state,
                states=states,
                curve_points=curve_points,
                target=target,
            )
        else:
            tally = self._run(
                t_simulation,
                working_nodes,
                broken_nodes,
                method,
                N,
                verbose,
                seed,
                antithetic,
                stop,
                capacity=capacity,
                jobs=jobs,
                engine=engine,
                entropy=entropy,
                states=states,
                curve_points=curve_points,
                sharded=sharded,
            )
        return self._availability_result(
            tally, t_simulation, initial_up, antithetic, capacity, controls
        )

    def _controlled_run(
        self,
        twin: "RepairableRBD",
        t_simulation: float,
        working: set,
        broken: set,
        method: str,
        N: int,
        verbose: bool,
        seed,
        antithetic: bool,
        stop: Optional[Callable[..., int]],
        *,
        capacity: Optional[_CapacityRecorder],
        jobs: Optional[int],
        engine: str,
        state,
        states: dict,
        curve_points: Optional[int],
        target: str,
    ) -> Tuple["_Tally", Tuple[Optional[ControlVariate], ...]]:
        """Run this system and its exact ``twin`` (see ``_twin``) with
        common random numbers, as ``compare`` does, in rounds while
        ``stop`` asks for more, judged by the controlled values; return
        this system's totals and the controls of its fractions up and of
        its costs (None without costs) by the twin's (see
        ``ControlVariate``)."""
        twin_states = twin._simulation_states(state, working | broken)
        exact = float(
            np.ravel(
                twin.mission_availability(
                    t_simulation, working, broken, method, state=state
                )
            )[0]
        )
        priced = self.has_costs and twin.has_costs
        exact_cost = (
            float(
                np.ravel(
                    twin.expected_cost(
                        t_simulation, working, broken, method, state=state
                    ).mean
                )[0]
            )
            if priced
            else None
        )
        entropy = _streams.entropy_of(seed)
        widths = self._common_widths(twin, t_simulation, states, twin_states)
        tally: Optional[_Tally] = None
        twin_tally: Optional[_Tally] = None
        first, count = 0, N
        while True:
            part = self._run(
                t_simulation,
                working,
                broken,
                method,
                count,
                verbose,
                None,
                antithetic,
                capacity=capacity,
                jobs=jobs,
                engine=engine,
                entropy=entropy,
                widths=widths,
                common=True,
                first=first,
                states=states,
                curve_points=curve_points,
            )
            twin_part = twin._run(
                t_simulation,
                working,
                broken,
                method,
                count,
                False,
                None,
                antithetic,
                jobs=jobs,
                engine=engine,
                entropy=entropy,
                widths=widths,
                common=True,
                first=first,
                states=twin_states,
                curve_points=1,
            )
            if tally is None or twin_tally is None:
                tally, twin_tally = part, twin_part
            else:
                tally.merge(part)
                twin_tally.merge(twin_part)
            fractions = np.asarray(tally.uptimes, dtype=float) / t_simulation
            control = ControlVariate.of(
                fractions,
                np.asarray(twin_tally.uptimes, dtype=float) / t_simulation,
                exact,
                antithetic,
            )
            cost_control = None
            if exact_cost is not None:
                cost_control = ControlVariate.of(
                    tally.cost_samples,
                    twin_tally.cost_samples,
                    exact_cost,
                    antithetic,
                )
            if stop is None:
                break
            if target == "cost":
                values = (
                    tally.cost_samples
                    if cost_control is None
                    else cost_control.controlled(tally.cost_samples)
                )
            else:
                values = control.controlled(fractions)
            more = stop(tally, values)
            if not more:
                break
            first, count = first + count, more
        return tally, (control, cost_control)

    def _run(
        self,
        t_simulation: float,
        working_nodes,
        broken_nodes,
        method: str,
        N: int,
        verbose: bool,
        seed,
        antithetic: bool = False,
        stop: Optional[Callable[["_Tally"], int]] = None,
        capacity: Optional[_CapacityRecorder] = None,
        jobs: Optional[int] = None,
        engine: str = "auto",
        entropy: Any = None,
        widths: Optional[dict] = None,
        common: bool = False,
        replacements: bool = False,
        first: int = 0,
        states: Optional[dict] = None,
        curve_points: Optional[int] = None,
        sharded: Optional[tuple] = None,
        histories: bool = False,
    ) -> "_Tally":
        """Run ``N`` simulations, from simulation ``first`` (then more,
        while ``stop`` asks for them), and return their totals: each from
        the components' ``states`` (checked; see ``_simulation_states``),
        new by default.

        Every draw comes from a stream of its own (see ``_streams``), seeded
        from the run's ``entropy``: the ``seed``, or without one a number
        drawn from numpy's global RNG. Draws that cannot be streamed come
        from the global RNG, seeded afresh for each simulation. So each
        simulation is the same whichever ``engine`` runs it, in however many
        processes or threads (``jobs``), and in a run to a tolerance as in a
        run of its final size. The global RNG is left as it was, except for
        the number drawn for the entropy. ``compare`` passes both systems
        the same ``entropy`` and ``widths``, and ``common``: that, like
        ``antithetic``, needs every draw to come from a stream. With
        ``capacity``, each simulation also follows the system's capacity;
        with ``replacements``, the tally keeps each simulation's
        replacements of each component, which only the Python engine
        counts; with ``curve_points``, it counts the changes of state on a
        grid (see ``availability``). With ``sharded``, ``(shard_map,
        template, step)``, the simulations run as shards (see
        ``_ShardRunner``), on the engine each shard's worker chooses. With
        ``histories``, the tally keeps each simulation's histories in place
        of every total (see ``simulate_timelines``), on the Python engine
        or numba's.
        """
        self._require_no_ccf("the simulation", nested=True)
        t_simulation = _check_window(t_simulation)
        from tqdm import tqdm

        state = np.random.get_state()
        after = state
        tally = _Tally(
            list(self.components), self.costs, t_simulation, curve_points
        )
        if replacements:
            # Only the Python engine counts them.
            tally.replacements = []
            engine = "python"
        if histories:
            tally.histories = _timeline_runs.Records(len(self.components))
        runner: Any = None
        try:
            if entropy is None:
                entropy = _streams.entropy_of(seed)
                if seed is None:
                    after = np.random.get_state()
            plan, complete = self._stream_plan(
                t_simulation, entropy, antithetic, widths, states
            )
            if (antithetic or common) and not complete:
                raise NotImplementedError(_UNSTREAMED)
            engine = self._simulation_engine(
                engine, plan, capacity, N, here=sharded is None, states=states
            )
            progress = tqdm(
                total=N, disable=not verbose, desc="Running simulations"
            )
            if sharded is not None:
                runner = _ShardRunner(self, tally, progress, *sharded)
            elif engine == "numba":
                from repyability.rbd import _compiled

                runner = _compiled.Runner(
                    self,
                    plan,
                    tally,
                    progress,
                    working_nodes,
                    broken_nodes,
                    method,
                    jobs,
                    capacity=capacity,
                )
            elif engine != "python":
                from repyability.rbd import engines

                runner = engines.registered()[engine].runner(
                    self,
                    plan,
                    tally,
                    progress,
                    working_nodes,
                    broken_nodes,
                    method,
                    jobs,
                )
            else:
                runner = _PythonRunner(
                    self,
                    tally,
                    progress,
                    (
                        t_simulation,
                        working_nodes,
                        broken_nodes,
                        method,
                        capacity,
                        entropy,
                        antithetic,
                        widths,
                        states,
                        histories,
                    ),
                    jobs,
                )
            start, goal = first, first + N
            while True:
                runner(start, goal)
                more = 0 if stop is None else stop(tally)
                if not more:
                    break
                start, goal = goal, goal + more
                progress.total = goal
                progress.refresh()
            progress.close()
        finally:
            if runner is not None:
                runner.close()
            np.random.set_state(after)
            self._forget_run()
        return tally

    #: The interim variables of a simulation (see _forget_run).
    _RUN_STATE = (
        "_event_queue",
        "system_state",
        "t_simulation",
        "component_status",
        "last_change_planned",
        "_pending_failure",
        "_in_service",
        "_renewed_at",
        "_pending_event",
        "_cancelled",
        "_early",
        "_renewing",
        "_crews",
        "_groups",
        "_step_sources",
        "_phases",
    )

    def _forget_run(self) -> None:
        """Clean up the interim variables of a simulation, this RBD's and
        its nested RBDs': left behind, a nested standby group's draws (whose
        samplers cannot be pickled) would keep the RBD from going to the
        processes of a later parallel run."""
        for name in self._RUN_STATE:
            self.__dict__.pop(name, None)
        for component in self.components.values():
            if isinstance(component, RepairableRBD):
                component._forget_run()

    def _simulation_engine(
        self,
        engine: str,
        plan: _streams.Plan,
        capacity: Optional[_CapacityRecorder],
        N: int,
        here: bool = True,
        states: Optional[dict] = None,
    ) -> str:
        """The engine that runs a simulation: ``"python"``, ``"numba"`` or
        one another package adds (see ``availability``'s ``engine``), for a
        run from the components' ``states``. Not ``here`` (the simulations
        run as shards elsewhere), only whether it can simulate the system
        is checked, and ``engine`` is kept, for each shard's worker to
        choose by."""
        from repyability.rbd import _compiled, engines

        added = engines.registered()
        if engine not in ("auto", "python", "numba") and engine not in added:
            names = ", ".join(
                repr(name) for name in ["auto", "python", "numba", *added]
            )
            raise ValueError(f"engine must be one of {names}, got {engine!r}.")
        if engine == "python":
            return engine
        if engine != "auto":
            reason = _compiled.unsupported(
                self, plan, capacity, numba=engine == "numba", states=states
            )
            if here and engine == "numba":
                _compiled.require()
            elif here and not added[engine].available():
                raise ImportError(
                    f"The {engine!r} simulation engine cannot run here."
                )
            if reason is not None:
                raise NotImplementedError(
                    f"The compiled engine does not simulate {reason}: use "
                    "engine='python', or 'auto', which chooses the engine "
                    "that can."
                )
            return _compiled.ready(engine, auto=False) if here else engine
        if not here:
            return engine
        name, _ = _compiled.choice(self, plan, capacity, states)
        if name is not None and _compiled.worthwhile(plan, N, name):
            return _compiled.ready(name, auto=True)
        return "python"

    def _replicate(self, ctx: "_Context", replication: int) -> _Replication:
        """Simulation ``replication`` of ``[0, t_simulation]``.

        Besides the counts, it follows each component's up time and its
        overlaps with the system's as it goes: when a component changes
        state, the time since its last change is added to its up time (if
        it was up), and the system's up (or down) time over the same span
        to the time both were up (or down), as the difference of the
        system's up (or down) time so far at either end.
        """
        ctx.run.begin(replication)
        t_simulation = ctx.t_simulation
        sources = ctx.sources
        failure_charges, preventive_charges, inspection_charges = ctx.charges
        self._start_queue(
            t_simulation,
            ctx.working,
            ctx.broken,
            ctx.method,
            sources,
            ctx.states,
        )
        status = self.component_status
        index, plain, works = ctx.position, ctx.plain, ctx.works
        inspected = self._inspection
        crews = self._crews
        groups = self._groups
        n = len(index)
        # Each component's failures, the system failures they caused, its
        # restorations and the system restorations they caused.
        failed, caused_down = [0] * n, [0] * n
        restored, caused_up = [0] * n, [0] * n
        # Each component's replacements: the spares it used.
        replaced = [0] * n
        # Imperfect repair (#109): a failure repaired, not replaced, uses no
        # spare and is charged only its repair.
        imperfect = self._imperfect
        # Opportunistic maintenance (#108): each member's early renewals.
        maintenance, member_group = self._maintenance, self._member_group
        cancelled, early = self._cancelled, self._early
        opportunistic = [0] * n
        failures = restorations = planned = 0
        changes: list = []
        deltas: list = []
        if ctx.initial_up and not self.system_state:
            # Down at the start, from the components' states: the curve of
            # the availability over time counts it as a change at 0.
            changes.append(0.0)
            deltas.append(-1)
        by_category = [0.0] * len(_CATEGORIES)
        by_node = dict.fromkeys(self.costs, 0.0)
        rep_cost = 0.0

        def pay(node, category: int, charges) -> float:
            """The next charge of ``charges`` (nothing without any)."""
            if charges is None:
                return 0.0
            charge = charges.draw()
            by_category[category] += charge
            by_node[node] += charge
            return charge

        trace = None if ctx.capacity is None else ctx.capacity.trace(status)
        # With ctx.history (see simulate_timelines): each component's state
        # at 0 and its changes (the component, the time, and whether it is
        # a planned change down), how many it has had, and each change of
        # the system's (the time, the component whose change made it, which
        # of that one's changes it was, and whether it was planned).
        record: Optional[list] = [] if ctx.history else None
        system_record: Optional[list] = [] if ctx.history else None
        seen = [0] * n
        at_start = [status[node] for node in index] if ctx.history else None
        # The system's state, and its up and down time until ``since``, its
        # last change; each component's last change, and the system's up and
        # down time until then.
        up = self.system_state
        started_up = up
        system_up = system_down = since = 0.0
        last, up_at, down_at = [0.0] * n, [0.0] * n, [0.0] * n
        node_up, both_up, both_down = [0.0] * n, [0.0] * n, [0.0] * n

        # The queue's heap, worked directly (see _EventQueue).
        heap = self._event_queue._heap
        push, pop = heapq.heappush, heapq.heappop

        # When each group last stopped: the work started at one instant is
        # one stop, with one set-up.
        stopped: Dict[Hashable, float] = {}

        def corrective(node, c: int) -> float:
            """Charge a failure's corrective action, and count the spare a
            replacement uses: an imperfect repair (see _ImperfectComponent)
            uses none, and is charged only as a repair."""
            charge = 0.0
            renewed = not imperfect or node not in imperfect
            renewed = renewed or sources[node].replacing
            if renewed:
                replaced[c] += 1
            for category, charges in failure_charges.get(node, ()):
                if renewed or category == _REPAIR:
                    charge += pay(node, category, charges)
            return charge

        def stop(group, t: float, trigger=None) -> float:
            """Open a stop of maintenance ``group`` (see ``_stop``): queue
            its early renewals, and return its set-up cost, charged once
            per stop, if a member opened it or it renews any."""
            renewals = self._stop(group, t, trigger)
            for renewal in renewals:
                push(heap, (renewal.time, renewal))
            if (trigger is None and not renewals) or stopped.get(group) == t:
                return 0.0
            stopped[group] = t
            setup = maintenance[group].setup_cost
            if setup:
                by_category[6] += setup
            return setup

        # No event at or after the end of the window is queued, so the
        # simulation runs until the queue is empty.
        while heap:
            event = pop(heap)[1]
            if cancelled and cancelled.get(id(event)) is event:
                del cancelled[id(event)]
                continue  # superseded by an early renewal
            node = event.component
            renewing_early = False
            if early and early.get(id(event)) is event:
                del early[id(event)]
                self._renewing.discard(node)
                opportunistic[index[node]] += 1
                renewing_early = True
            grouped = False
            if groups and node in groups:
                # A standby group's own event (see _StandbyGroup): a unit
                # failure, charged as a repair, or a repair's end.
                group = groups[node]
                if not group.holds(event):
                    continue  # superseded by an earlier one
                now, broken = group.advance(event.time)
                if broken:
                    replaced[index[node]] += 1
                    for category, charges in failure_charges.get(node, ()):
                        rep_cost += pay(node, category, charges)
                if now == status[node]:
                    continue
                event, grouped = Event(event.time, node, now), True
            if (event.preventive or event.inspection) and (
                event.status == status[node]
            ):
                # No change of state: maintenance or a test in zero time of a
                # working unit (renewed in place, or found working), or a
                # test that finds a hidden failure, whose repair starts now.
                if event.preventive:
                    replaced[index[node]] += 1
                    rep_cost += pay(node, 2, preventive_charges.get(node))
                    # Replaced on condition: at an inspection, charged too.
                    rep_cost += pay(node, 3, inspection_charges.get(node))
                    if node in member_group and not renewing_early:
                        # Renewed in place on schedule: a stop of its group.
                        rep_cost += stop(member_group[node], event.time, node)
                else:
                    rep_cost += pay(node, 3, inspection_charges.get(node))
                    if not event.status and not event.missed:
                        rep_cost += corrective(node, index[node])
                follow = self._follow_up(event, sources[node])
                next_event: Optional[Event] = (
                    follow
                    if crews is None
                    else self._crew_follow_up(event, follow)
                )
                if next_event is not None and next_event.time < t_simulation:
                    push(heap, (next_event.time, next_event))
                continue
            t = event.time
            c = index[node]
            if up:
                system_up_t, system_down_t = (
                    system_up + (t - since),
                    system_down,
                )
            else:
                system_up_t, system_down_t = system_up, system_down + (
                    t - since
                )
            if status[node]:
                node_up[c] += t - last[c]
                both_up[c] += system_up_t - up_at[c]
            else:
                both_down[c] += system_down_t - down_at[c]
            last[c], up_at[c], down_at[c] = t, system_up_t, system_down_t
            status[node] = event.status
            if trace is not None:
                trace.change(t, node, event.status)
            if record is not None:
                # A planned change down: maintenance or a test off line.
                record.append(
                    (
                        c,
                        t,
                        not event.status
                        and bool(event.preventive or event.inspection),
                    )
                )
                seen[c] += 1
            if event.status:
                restored[c] += 1
            elif event.preventive:
                # A planned outage: charged the preventive cost, and the
                # inspection's if replaced on condition (a nested RBD's own
                # costs are not counted).
                replaced[c] += 1
                rep_cost += pay(node, 2, preventive_charges.get(node))
                rep_cost += pay(node, 3, inspection_charges.get(node))
                if node in member_group and not renewing_early:
                    rep_cost += stop(member_group[node], t, node)
            elif event.inspection:
                # A test that takes a working unit off-line.
                rep_cost += pay(node, 3, inspection_charges.get(node))
            else:
                failed[c] += 1
                # Repair and replace are charged per corrective action, at
                # the failure that triggers it (for a hidden failure, when an
                # inspection finds it; for a standby group, at each unit's).
                if node not in inspected and not grouped:
                    if imperfect and node in imperfect:
                        rep_cost += corrective(node, c)
                    else:
                        replaced[c] += 1
                        for category, charges in failure_charges.get(node, ()):
                            rep_cost += pay(node, category, charges)
                if node in member_group:
                    rep_cost += stop(member_group[node], t, node)

            # The structure is coherent: a restoration can't take the
            # system down, nor a failure bring it up, so only a change
            # against the system's state needs the structure function.
            if event.status != up and works(status) != up:
                system_up, system_down, since = system_up_t, system_down_t, t
                up = not up
                changes.append(t)
                if system_record is not None:
                    system_record.append(
                        (
                            t,
                            c,
                            seen[c] - 1,
                            not up
                            and bool(event.preventive or event.inspection),
                        )
                    )
                if up:
                    deltas.append(1)
                    restorations += 1
                    caused_up[c] += 1
                else:
                    deltas.append(-1)
                    if event.preventive or event.inspection:
                        planned += 1
                    else:
                        failures += 1
                        caused_down[c] += 1
                    for name, spec in maintenance.items():
                        if spec.system_down:
                            # Any system outage is a stop of this group.
                            rep_cost += stop(name, t)

            if grouped:
                continue  # the group has queued its next event
            # The component's next event: its next failure if it has just
            # been restored, or its restoration if it has just gone down;
            # queued only if it falls inside the window.
            if node in plain:
                delay, next_status = sources[node].next_event()
                next_event = Event(t + delay, node, next_status)
            else:
                next_event = self._follow_up(event, sources[node])
            if crews is not None:
                # A job waits for a repair crew (see _Crews).
                next_event = self._crew_follow_up(event, next_event)
            if next_event is not None and next_event.time < t_simulation:
                push(heap, (next_event.time, next_event))
        self.system_state = up

        # Close the system's time, and each component's, at the end of the
        # window.
        if up:
            system_up += t_simulation - since
        else:
            system_down += t_simulation - since
        for c, node in enumerate(index):
            if status[node]:
                node_up[c] += t_simulation - last[c]
                both_up[c] += system_up - up_at[c]
            else:
                both_down[c] += system_down - down_at[c]

        rates = ctx.downtime_cost_rates
        if rates:
            for c, node in enumerate(index):
                rate = rates.get(node)
                if rate is not None:
                    charge = rate * (t_simulation - node_up[c])
                    rep_cost += charge
                    by_category[4] += charge
                    by_node[node] += charge
        if ctx.has_costs:
            charge = self.downtime_cost_rate * (t_simulation - system_up)
            rep_cost += charge
            by_category[5] += charge

        rec = _Replication()
        rec.uptime = system_up
        rec.node_up, rec.both_up, rec.both_down = node_up, both_up, both_down
        rec.counts = (failed, caused_down, restored, caused_up)
        rec.failures, rec.restorations = failures, restorations
        rec.planned = planned
        rec.changes, rec.deltas = changes, deltas
        rec.cost = rep_cost if ctx.has_costs else None
        rec.by_category, rec.by_node = by_category, by_node
        rec.capacity_changes = rec.capacity_time = rec.delivered = None
        rec.replacements = replaced
        rec.opportunistic = opportunistic
        rec.history = (
            None
            if record is None
            else (at_start, record, started_up, system_record)
        )
        if trace is not None:
            trace.finish(t_simulation, rec)
        return rec

    def _availability_result(
        self,
        tally: "_Tally",
        t_simulation: float,
        initial_up: bool,
        antithetic: bool,
        capacity: Optional[_CapacityRecorder] = None,
        controls: Tuple[Optional[ControlVariate], ...] = (None, None),
    ) -> AvailabilityResult:
        """The ``AvailabilityResult`` of the replications in ``tally``: its
        exact totals rounded, once; with ``controls``, the controls of its
        fractions up and its costs by an exact twin (see
        ``_controlled_run``)."""
        N = tally.n
        nodes = tally.nodes
        tally._fold()

        def rounded(totals) -> list:
            return [float(total) for total in totals]

        system_uptime = float(tally.system_uptime)
        system_downtime = float(tally.system_downtime)
        both_up = rounded(tally.intersection_uptime)
        both_down = rounded(tally.intersection_downtime)
        # Collect Importance/Criticality measures from the simulation
        # reference: https://www.weibull.com/pubs/2004rm_05B_02.pdf
        # Operational Criticality Index
        oci_down = {
            k: _safe_ratio(v, system_downtime)
            for k, v in zip(nodes, both_down)
        }
        oci_up = {
            k: _safe_ratio(v, system_uptime) for k, v in zip(nodes, both_up)
        }
        # Intersection Over Union Importance
        iou_up = {
            k: _safe_ratio(both, either)
            for k, both, either in zip(
                nodes, both_up, rounded(tally.union_uptime)
            )
        }
        iou_down = {
            k: _safe_ratio(both, either)
            for k, both, either in zip(
                nodes, both_down, rounded(tally.union_downtime)
            )
        }
        FCI, RCI = tally.criticality_counts()
        # Failure Criticality Index Importance
        fci_sys = failure_criticality_index_per_system_failures(
            FCI, tally.system_failures
        )
        fci_comp = failure_criticality_index_per_component_failures(FCI)
        # Restoration Criticality Index Importance
        rci_sys = restoration_criticality_index_by_system(
            RCI, tally.system_restorations
        )
        rci_comp = restoration_criticality_index_by_component(RCI)
        criticalities = Criticalities(
            operational_criticality_index=UpDownImportance(
                up=oci_up, down=oci_down
            ),
            iou=UpDownImportance(up=iou_up, down=iou_down),
            failure_criticality_index=FailureCriticalityIndex(
                per_system_failure=fci_sys, per_component_failure=fci_comp
            ),
            restoration_criticality_index=RestorationCriticalityIndex(
                by_system=rci_sys, by_component=rci_comp
            ),
        )

        # The availability from t=0..t_simulation: how many of the N
        # simulated systems work after each time at which one changed state
        # (and at 0 and t_simulation whether or not any did), over N.
        if tally.binned is not None:
            # On the grid (#153): how many work at each of its times.
            assert tally.edges is not None
            time = tally.edges.copy()
            working = (N if initial_up else 0) + np.cumsum(tally.binned)
        else:
            changed_at, deltas = tally.state_changes()
            time, working = _working_over_time(
                changed_at, deltas, t_simulation, N if initial_up else 0
            )
        system_availability = working / N

        cost_result = None
        if self.has_costs:
            # Per-component means cover each node's repair, replace,
            # preventive and own downtime cost. (System downtime is a
            # system-level quantity and is not attributed to components.)
            cost_result = CostResult(
                samples=np.asarray(tally.cost_samples, dtype=float),
                t_simulation=t_simulation,
                n_simulations=N,
                acquisition_cost=self.acquisition_cost,
                by_category={
                    k: float(v) / N for k, v in tally.cost_by_category.items()
                },
                by_component={
                    k: float(v) / N for k, v in tally.cost_by_component.items()
                },
                antithetic=antithetic,
                control_variate=controls[1],
            )

        capacity_fields: dict = {}
        if capacity is not None:
            # The mean capacity from t=0..t_simulation: the simulated
            # systems' total expected capacity after each time at which one
            # changed (unlimited while one can carry an unlimited amount).
            at, steps, starts, free_at, counts = tally.capacity_changes()
            times = np.unique(np.r_[at, free_at, 0.0, t_simulation])
            change = np.zeros(times.size)
            change[np.searchsorted(times, at)] = _group_totals(steps, starts)
            freed = np.zeros(times.size, dtype=np.int64)
            freed[np.searchsorted(times, free_at)] = counts
            totals, unlimited = np.cumsum(change), np.cumsum(freed)
            curve = np.where(
                unlimited > 0, np.inf, np.maximum(totals / N, 0.0)
            )
            capacity_fields = dict(
                capacity_timeline=times,
                capacity=curve,
                capacity_time={
                    level: float(tally.capacity_time[level])
                    for level in sorted(tally.capacity_time)
                },
                demand=capacity.demand,
                delivered=(
                    None
                    if capacity.demand is None
                    else np.asarray(tally.delivered, dtype=float)
                ),
            )

        return AvailabilityResult(
            timeline=time,
            availability=system_availability,
            system_uptime=system_uptime,
            time_simulated_to=t_simulation,
            criticalities=criticalities,
            node_uptime=dict(zip(nodes, rounded(tally.node_uptime))),
            node_downtime=dict(zip(nodes, rounded(tally.node_downtime))),
            system_downtime=system_downtime,
            system_failures=tally.system_failures,
            system_restorations=tally.system_restorations,
            n_simulations=N,
            cost=cost_result,
            system_planned_outages=tally.system_planned_outages,
            uptimes=np.asarray(tally.uptimes, dtype=float),
            antithetic=antithetic,
            opportunistic_renewals=(
                dict(zip(nodes, tally.opportunistic))
                if self._maintenance
                else None
            ),
            control_variate=controls[0],
            **capacity_fields,
        )

    def cost(
        self,
        t_simulation: float,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        mc_samples: Optional[int] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
        *,
        tolerance: Optional[float] = None,
        confidence: float = 0.95,
        max_samples: Optional[int] = None,
        antithetic: bool = False,
        n_jobs: Optional[int] = None,
        engine: str = "auto",
        state=None,
        shard_map: Optional[Callable] = None,
        shard_size: Optional[int] = None,
        control_variate: bool = False,
        N: Optional[int] = None,
        max_N: Optional[int] = None,
    ) -> Optional[CostResult]:
        """Simulate the cost of running the system for ``t_simulation``.

        Runs the ``availability`` simulation with cost accumulation and
        returns its [`CostResult`][repyability.CostResult]: the distribution
        of the window's total cost (``samples``, ``mean``,
        ``percentile(q)``), a confidence interval for the mean
        (``mean_interval()``), and per-category and per-component
        breakdowns. In each simulation ``repair_cost`` and ``replace_cost``
        are charged at every failure of their component (a fresh draw for a
        cost distribution), a preventive-maintenance cost at every
        preventive replacement, ``downtime_cost`` per unit time its
        component is down (planned outages included), and
        ``downtime_cost_rate`` per unit time the system is down.
        Only this RBD's own costs count: costs declared inside a nested
        ``RepairableRBD`` are left out. ``result.cost_rate`` converges to
        the exact ``expected_cost_rate`` as the window grows, which is the
        cross-check to reach for.

        Returns None when nothing is priced (see ``has_costs``): there is
        no cost model to evaluate. (The same result is available as
        ``availability(...).cost`` if you also want the availability outputs
        from the same simulations.)

        Parameters
        ----------
        t_simulation : float
            Length of the window each simulation covers, in the time unit
            of the component models.
        working_nodes : Collection[Hashable], optional
            Nodes held working for the whole window, by default None.
        broken_nodes : Collection[Hashable], optional
            Nodes held failed for the whole window, by default None.
        method : str, optional
            Evaluate the system state from the minimal path sets (``"p"``,
            the default) or the minimal cut sets (``"c"``); the results are
            identical.
        mc_samples : int, optional
            Number of simulations, each giving one sample of the window's
            total cost, by default 10_000.
        verbose : bool, optional
            If True, displays a progress bar of the simulations, by default
            False.
        seed : int, optional
            Seed for a reproducible run, by default None (see
            ``availability``). A cost given as a distribution draws from a
            stream of its own, so pricing never changes the simulated
            failures and repairs.
        tolerance : float, optional
            Simulate until the mean cost of a window is known to within
            ``tolerance`` (in the costs' currency) either side, at
            ``confidence`` (the half-width of ``mean_interval()``): as in
            ``availability``, checked after the first ``mc_samples``
            simulations and each further ``mc_samples``, up to
            ``max_samples``. By default None: exactly ``mc_samples``.
        confidence : float, optional
            The confidence level ``tolerance`` is judged at, by default
            0.95.
        max_samples : int, optional
            The most simulations a run to ``tolerance`` makes, by default
            100 times ``mc_samples``.
        antithetic : bool, optional
            Run the simulations in antithetic pairs (see ``availability``),
            by default False. The draws of costs given as distributions are
            paired too.
        n_jobs : int, optional
            Run the simulations in parallel on ``n_jobs`` CPUs (see
            ``availability``), by default None.
        engine : str, optional
            What runs the simulations: ``"python"``, ``"numba"`` or
            ``"auto"`` (the default), as in ``availability``.
        state : dict, optional
            Start each simulation from the components' current states
            rather than new, as for ``availability``: ``{node:
            NodeState}``. Costs incurred before 0 (a repair or maintenance
            going on at 0 was charged when it started) are not counted. By
            default None: every component new at 0.
        shard_map : callable, optional
            Run the simulations as shards through this map, wherever it
            sends them, as for ``availability``. By default None.
        shard_size : int, optional
            The simulations to a shard, with ``shard_map``, as for
            ``availability``.
        control_variate : bool, optional
            Control the estimate of the mean cost by the system's exact
            twin, as for ``availability``: ``mean_interval`` then gives the
            controlled estimate, from the twin's exact ``expected_cost``,
            and a ``tolerance`` is judged on it. By default False.

        N : int, optional
            Deprecated: the old name of ``mc_samples``.
        max_N : int, optional
            Deprecated: the old name of ``max_samples``.
        Returns
        -------
        CostResult or None
            The simulated costs, or None if nothing is priced.

        Raises
        ------
        ValueError
            As for ``availability``, when something is priced. (With nothing
            priced this returns None without any checks.)
        NotImplementedError
            As for ``availability``.
        ImportError
            As for ``availability``.

        Examples
        --------
        One component with MTTF 10 and MTTR 1, at 100 per repair, and 50 per
        hour of system downtime:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> rbd = RepairableRBD(
        ...     [("s", "c"), ("c", "t")],
        ...     {
        ...         "c": {
        ...             "reliability": surv.Exponential.from_params([0.1]),
        ...             "repairability": surv.Exponential.from_params([1.0]),
        ...             "repair_cost": 100.0,
        ...         }
        ...     },
        ...     downtime_cost_rate=50.0,
        ... )
        >>> result = rbd.cost(t_simulation=100.0, mc_samples=200, seed=0)
        >>> round(result.mean, 2)  # mean cost of a 100-hour window
        1354.77
        >>> round(result.percentile(90), 2)  # 9 windows in 10 cost less
        1961.22
        >>> round(result.cost_rate, 2), round(rbd.expected_cost_rate(), 2)
        (13.55, 13.64)
        """
        mc_samples = renamed("mc_samples", mc_samples, "N", N)
        N = 10_000 if mc_samples is None else mc_samples
        max_N = renamed("max_samples", max_samples, "max_N", max_N)
        t_simulation = _check_window(t_simulation)

        if not self.has_costs:
            return None
        return self._simulated(
            t_simulation,
            working_nodes,
            broken_nodes,
            method,
            N,
            verbose,
            seed,
            tolerance=tolerance,
            confidence=confidence,
            max_N=max_N,
            antithetic=antithetic,
            n_jobs=n_jobs,
            target="cost",
            engine=engine,
            state=state,
            # The cost result has no curve: count the changes, not keep them.
            curve_points=1,
            shard_map=shard_map,
            shard_size=shard_size,
            control_variate=control_variate,
        ).cost

    def node_availability(self) -> dict[Hashable, float]:
        """Returns each node's long-run (steady-state) availability.

        Exact: ``MTTF / (MTTF + MTTR)`` for each component (1.0 if it is
        repaired instantly), a nested ``RepairableRBD``'s own
        ``mean_availability``, and 1.0 for the input and output nodes. These
        are the availabilities the importance measures are evaluated at.

        A component under age replacement at ``T`` renews at a failure or
        at a preventive replacement, whichever comes first. By the
        renewal-reward theorem its availability is its mean up time per
        cycle over the mean cycle length,
        ``integral_0^T R / (integral_0^T R + F(T) * MTTR + R(T) * MTTP)``,
        with ``MTTP`` the mean maintenance time (0 for ``"instant"``).

        Under block replacement every ``T`` a unit is replaced at each
        multiple of ``T`` at which it is up, and in between it fails and is
        repaired as usual; a replacement due while it is down is skipped.
        The block times at which it is up are its renewals, so its
        availability is its mean up time between two of them over their
        mean distance apart. That needs the expected up time and number of
        failures of an alternating renewal process of lives and repairs
        within an interval, and what a repair still going on at a block time
        carries into the next: they are computed numerically, with an error
        falling as the square of the grid step (about 1e-6 or less). With
        instant repair and replacement the unit is always up, and it fails
        ``M(T)`` times per interval, ``M`` the renewal function of its
        lives.

        A component with hidden failures, a constant failure rate
        ``lambda``, inspected every ``tau`` with instant tests and instant
        repair, is up a fraction ``(1 - exp(-lambda * tau)) / (lambda *
        tau)`` of the time, about ``1 - lambda * tau / 2``: it is down, from
        a failure until the next inspection, half an interval on average.
        Other hidden failures have no exact long-run values here: simulate
        them.

        With limited ``repair_crews``, a component's is its long-run
        probability of being up in the crews' Markov chain (see
        ``mean_availability``): waiting for a crew, it is down.

        Returns
        -------
        dict[Hashable, float]
            Node name -> long-run availability, in ``[0, 1]``.

        Raises
        ------
        ValueError
            If a component has a non-parametric reliability model, whose
            MTTF this cannot compute.
        NotImplementedError
            If a component is under block replacement with models its exact
            values do not cover (a lifetime that is not a surpyval
            parametric model with a density, dead-on-arrival units, repairs
            that may never end, or repairs or maintenance far longer than
            the interval), or has hidden failures whose tests or repairs
            take time, or whose tests can miss a failure of a life that is
            not exponential; or, while a component can wait for a repair
            crew, as for ``mean_availability``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": "instant"},
        ...     },
        ... )
        >>> availability = rbd.node_availability()
        >>> {node: round(a, 4) for node, a in availability.items()}
        {'a': 0.8333, 'b': 1.0, 's': 1.0, 't': 1.0}
        """
        node_av: dict[Hashable, float] = {}
        for node_name in self.components:
            node_av[node_name] = self._node_availability(node_name)

        for node_name in self.in_or_out:
            node_av[node_name] = 1.0

        return node_av

    def _node_availability(self, node) -> float:
        """A component's long-run availability (see ``node_availability``):
        with limited repair crews, its long-run probability of being up in
        their Markov chain, or a nested RBD's own, as it has crews of its
        own."""
        self._require_perfect_repair(node)
        if self._crews_couple():
            chain = self._crew_chain()
            if node in chain.nodes:
                return chain.availability(node)
            return float(self.components[node].mean_availability())
        if node in self._standby:
            return self._standby_long_run(node).availability
        if node in self._inspection:
            life = self._tested_life(node)
            if life is not None:
                return life.availability
            rate, interval = self._inspected_rate(node)
            inspection = self._inspection[node]
            if inspection.partial:
                # Up with probability rho ** k * exp(-rate * u) (see
                # _tested_profile): averaged over a full test's cycle,
                # (1 - rho ** m) / ((1 - coverage) * rate * full_test).
                kept = _log_kept(rate, inspection)
                return float(
                    -np.expm1(inspection.per_full_test * kept)
                    / ((1.0 - inspection.coverage) * rate * inspection.period)
                )
            return float(-np.expm1(-rate * interval) / (rate * interval))
        schedule = self._preventive.get(node)
        if schedule is None:
            component = self.components[node]
            return float(np.atleast_1d(component.mean_availability())[0])
        up, cycle, _, _ = self._maintenance_cycle(node, schedule)
        return min(1.0, up / cycle)

    def _node_unavailability(self, node) -> float:
        """A component's long-run unavailability, as ``_node_availability``
        gives its availability, but worked out in its own right (from a
        standby group's down states, or the mean down time over a cycle),
        so that a small one keeps its precision. For a component whose
        value is constant over the long-run grid, and while the crews do
        not couple the components (see ``_long_run_unavailabilities``)."""
        self._require_perfect_repair(node)
        if node in self._standby:
            return self._standby_long_run(node).unavailability
        component = self.components[node]
        schedule = self._preventive.get(node)
        if schedule is None:
            return float(np.atleast_1d(component.mean_unavailability())[0])
        up, cycle, fails, survives = self._maintenance_cycle(node, schedule)
        if schedule.policy == "block":
            return max(0.0, (cycle - up) / cycle)
        # Down for a repair after a failure before the age (its probability
        # from the model's own ff), else for the maintenance.
        maintenance = (
            0.0 if schedule.duration is None else model_mean(schedule.duration)
        )
        down = (
            fails * model_mean(component.time_to_replace)
            + survives * maintenance
        )
        return min(1.0, down / cycle)

    def _require_one_inspected(self) -> None:
        """Raise if more than one component has hidden failures, when no
        intervals are given to choose from (``optimal_inspection_intervals``
        with ``allowed=None``)."""
        if len(self._inspection) > 1:
            raise ValueError(
                "More than one component has hidden failures: give the "
                "intervals to choose from in allowed (tests are made on "
                "a calendar, and the long-run values depend on how the "
                "schedules line up)."
            )

    def _require_capacities_given(self) -> None:
        """Raise if a node takes its capacity from its model (a
        ``DegradingNode``'s stages, or a nested RBD's capacities), which
        the availability simulation does not follow."""
        own = self._capacity_models()
        if own:
            raise NotImplementedError(
                f"Node(s) {sorted(own, key=str)} take their capacity from "
                "their models (a DegradingNode's stages, or a nested RBD's "
                "capacities), which the simulation does not follow. Give "
                "them a capacity, or use capacity_distribution() for the "
                "long run."
            )

    def _require_time_models(self, node) -> None:
        """Raise if a component's life or repair model is a probability,
        not a distribution of times: its availability over time then has
        no exact value."""
        component = self.components[node]
        for what, model in [
            ("reliability", component.reliability),
            ("repairability", component.time_to_replace),
        ]:
            if is_fixed_probability(model):
                raise NotImplementedError(
                    f"Component {node!r}: its {what} model is a probability, "
                    "not a distribution of times, so its availability over "
                    "time has no exact value. Estimate it by simulation, "
                    "with availability()."
                )

    def _require_unscheduled_stages(self, node) -> None:
        """Raise if a degrading component is maintained or inspected on a
        schedule: its long-run time in each stage has no exact value."""
        if node in self._preventive or node in self._inspection:
            raise NotImplementedError(
                f"Component {node!r} degrades through stages and is "
                "maintained or inspected on a schedule: its long-run time in "
                "each stage has no exact value here. Give it a capacity "
                "instead."
            )

    def _require_no_condition(self, node) -> None:
        """Raise if a component is replaced on condition (a ``"preventive"``
        schedule with ``"policy": "condition"``): its availability over time
        is known only by simulation (its long-run values are numerical:
        see ``_condition_replacement``)."""
        schedule = self._preventive.get(node)
        if schedule is not None and schedule.policy == "condition":
            raise NotImplementedError(
                f"Component {node!r} is replaced on condition (at an "
                "inspection, if it is then likely enough to fail before the "
                "next), so its availability over time has no exact value "
                "here, as yet (#161): its long-run values do. Estimate the "
                "values over time by simulation, with availability() or "
                "cost()."
            )

    def _imperfect_phrase(self, node) -> str:
        """How a component is repaired imperfectly, in words."""
        imperfect = self._imperfect[node]
        kind = "I" if imperfect.kijima == "kijima1" else "II"
        phrase = f"Kijima {kind}, q = {imperfect.q:g}"
        if imperfect.replace_after is not None:
            phrase += (
                f", replaced at failure {imperfect.replace_after} since it "
                "was renewed"
            )
        return phrase

    def _require_perfect_repair(self, node) -> None:
        """Raise if a component is repaired imperfectly (a spec's
        ``"repair"``): a repair does not renew it, so its long-run values
        and its availability over time are known only by simulation."""
        if node in self._imperfect:
            raise NotImplementedError(
                f"Component {node!r} is repaired imperfectly "
                f"({self._imperfect_phrase(node)}), so a repair does not "
                "renew it: its long-run values and its availability over "
                "time have no exact value here. Estimate them by "
                "simulation, with availability() or cost()."
            )

    def _require_no_opportunities(self, node) -> None:
        """Raise if a component can be renewed early at the stops of its
        maintenance group (an ``"opportunity"`` below its interval): when
        depends on the other members, so its long-run values and its
        availability over time are known only by simulation."""
        if node in self._early_members:
            raise NotImplementedError(
                f"Component {node!r} is renewed early at the stops of its "
                f"maintenance group {self._member_group[node]!r}, which "
                "depend on the other members, so its long-run values and "
                "its availability over time have no exact value here. "
                "Estimate them by simulation, with availability() or cost()."
            )

    def _require_separate_setups(self) -> None:
        """Raise if two members of a maintenance group with a set-up cost
        can be replaced at the same instants, again and again: on block
        schedules, or never failing before an age replacement in zero time.
        Such replacements share one stop, and its set-up, which the exact
        cost rate, charging a set-up for each member's failures and
        replacements, does not count."""
        for group, spec in self._maintenance.items():
            if not spec.setup_cost:
                continue
            clocked = [node for node in spec.members if self._on_a_clock(node)]
            if len(clocked) > 1:
                raise NotImplementedError(
                    f"Components {clocked} of maintenance group {group!r} "
                    "are replaced on a clock (on block schedules, or never "
                    "failing before an age replacement in zero time), so "
                    "their replacements can fall at the same instants and "
                    "share a set-up, which the exact cost rate does not "
                    "count: estimate it by simulation, with cost()."
                )

    def _on_a_clock(self, node) -> bool:
        """Whether a component's replacements fall on a fixed lattice of
        times: under block replacement, or under age replacement in zero
        time without a failure before it."""
        schedule = self._preventive.get(node)
        if schedule is None or not math.isfinite(schedule.interval):
            return False
        if schedule.policy == "block":
            return True
        if schedule.policy != "age" or schedule.duration is not None:
            return False
        survives = self.components[node].reliability_function(
            schedule.interval
        )
        return bool(np.ravel(survives)[0] >= 1.0)

    def _require_block_models(self, node) -> None:
        """Raise unless the exact block-replacement values cover the
        component's models (the checks ``block_cycle`` makes first)."""
        from repyability.rbd._block_replacement import _check_life, _Duration

        component = self.components[node]
        _check_life(component.reliability, node)
        _Duration(component.time_to_replace, "repair", node)
        _Duration(self._preventive[node].duration, "replacement", node)

    def _require_calendars(self) -> None:
        """Raise if the exact long-run values cannot average over the
        components' calendars (block replacement and inspection): a nested
        RBD's calendar with another here, intervals with no common period,
        or one repeating too often in it (before any computation)."""
        nested = [
            node
            for node, c in self.components.items()
            if isinstance(c, RepairableRBD) and c._has_calendar()
        ]
        blocks = self._block_nodes()
        if nested and len(nested) + len(self._inspection) + len(blocks) > 1:
            raise NotImplementedError(
                f"Node(s) {sorted(nested, key=str)} are RBDs with hidden "
                "failures or block replacement, and other nodes' inspections "
                "or block replacements here fall at the same times: "
                "estimate the long-run values by simulation, with "
                "availability() or cost()."
            )
        intervals = {self._preventive[node].interval for node in blocks}
        intervals |= {
            self._inspection[node].interval for node in self._inspection
        }
        if not intervals:
            return
        period = _common_period(intervals)
        if any(round(period / interval) > 100_000 for interval in intervals):
            if blocks:
                raise NotImplementedError(
                    f"The block-replacement and inspection intervals "
                    f"{sorted(intervals)} repeat together only after too many "
                    "intervals to average over: estimate the long-run values "
                    "by simulation, with availability() or cost()."
                )
            raise NotImplementedError(
                f"The inspection intervals {sorted(intervals)} repeat "
                "together only after too many inspections to average "
                "over: estimate the long-run values by simulation, with "
                "availability() or cost()."
            )

    def _require_tested_exact(self, node) -> None:
        """Raise unless a component with hidden failures has exact values
        (long-run, or from new over time): with instant tests and instant
        repair, for any life (#144), and, for tests that can miss a
        failure, a constant failure rate."""
        component = self.components[node]
        inspection = self._inspection[node]
        if (
            inspection.duration is not None
            or model_mean(component.time_to_replace) != 0.0
        ):
            raise NotImplementedError(
                f"Component {node!r} has hidden failures: its exact values "
                "(long-run, or from new over time) are known only with "
                "instant tests and instant repair. "
                "Estimate them by simulation, with availability() or cost()."
            )
        if (
            inspection.partial
            and _constant_rate(component.reliability) is None
        ):
            raise NotImplementedError(
                f"Component {node!r} has hidden failures, found by tests "
                "that can miss them: its exact values are known only with a "
                "constant failure rate (an exponential life). "
                "Estimate them by simulation, with availability() or cost()."
            )

    def _inspected_rate(self, node) -> Tuple[float, float]:
        """The constant failure rate and the inspection interval of a
        component with hidden failures, for the closed forms of an
        exponential life (see ``_tested_life`` for any other)."""
        self._require_tested_exact(node)
        component = self.components[node]
        rate = _constant_rate(component.reliability)
        if rate is None:
            raise NotImplementedError(
                f"Component {node!r} has hidden failures and a life that is "
                "not exponential, which this does not take: estimate it by "
                "simulation, with availability() or cost()."
            )
        return rate, self._inspection[node].interval

    def _tested_life(self, node) -> Optional[TestedLife]:
        """For a component with hidden failures and a life other than
        exponential, its long run under its tests (see ``_hidden_life``),
        worked out once; None for an exponential life, whose values have
        closed forms."""
        self._require_tested_exact(node)
        component = self.components[node]
        if _constant_rate(component.reliability) is not None:
            return None
        interval = float(self._inspection[node].interval)
        cache = self.__dict__.setdefault("_tested_lives", {})
        key = (node, id(component.reliability), interval)
        if key not in cache:
            cache[key] = TestedLife(component.reliability, interval)
        return cache[key]

    def _tested_rate(self, node) -> float:
        """How fast a tested component's long-run availability falls
        between tests: its failure rate, or for any other life the rate of
        its profile's mean decay (see ``TestedLife.rate``), for the long-run
        grid's spacing."""
        life = self._tested_life(node)
        return self._inspected_rate(node)[0] if life is None else life.rate

    def _tested_scale(self, node) -> float:
        """The rate that sets the scale of a tested component's interval:
        its failure rate, or one over the mean of any other life."""
        life = self._tested_life(node)
        if life is None:
            return self._inspected_rate(node)[0]
        return 1.0 / float(model_mean(life.model))

    def _tested_phase(self, node, times: np.ndarray) -> np.ndarray:
        """The time since a tested component's last test at each of
        ``times`` on its calendar (its tests at its offset and every
        interval from it)."""
        inspection = self._inspection[node]
        position = times - inspection.offset if inspection.offset else times
        interval = inspection.interval
        return position - interval * np.floor(position / interval)

    def _tested_intensity(
        self, node, times: np.ndarray, availability
    ) -> np.ndarray:
        """A tested component's long-run rate of failing at each of
        ``times`` (``availability`` its availability there): its constant
        rate while it is up, or for any other life the rate of its profile
        (see ``TestedLife.intensity``)."""
        life = self._tested_life(node)
        if life is None:
            rate, _ = self._inspected_rate(node)
            return rate * availability
        return life.intensity(self._tested_phase(node, times))

    def _block_nodes(self) -> list:
        """The components renewed on a calendar, whose long-run values vary
        with it: under block replacement, or replaced on condition at
        inspections (#145)."""
        return [
            node
            for node, schedule in self._preventive.items()
            if schedule.policy in ("block", "condition")
        ]

    def _block_cycle(self, node):
        """The renewal cycle of a component under block replacement (see
        ``_block_replacement``) or replaced on condition (see
        ``_condition_replacement``), computed once and kept."""
        component = self.components[node]
        schedule = self._preventive[node]
        cache = self.__dict__.setdefault("_block_cycles", {})
        key = (
            node,
            id(component.reliability),
            id(component.time_to_replace),
            float(schedule.interval),
            id(schedule.duration),
            schedule.policy,
            schedule.threshold,
        )
        if key not in cache:
            if schedule.policy == "condition":
                cache[key] = condition_cycle(
                    component.reliability,
                    component.time_to_replace,
                    schedule.duration,
                    schedule.interval,
                    float(schedule.threshold),
                    node,
                )
            else:
                cache[key] = block_cycle(
                    component.reliability,
                    component.time_to_replace,
                    schedule.duration,
                    schedule.interval,
                    node,
                )
        return cache[key]

    def _has_calendar(self) -> bool:
        """Whether a component here, or in a nested RBD, is inspected or
        replaced on a calendar: its long-run availability then varies with
        the time of the schedule."""
        return (
            bool(self._inspection)
            or bool(self._block_nodes())
            or any(
                isinstance(c, RepairableRBD) and c._has_calendar()
                for c in self.components.values()
            )
        )

    def _calendar_grid(self, blocks: list) -> Tuple[np.ndarray, np.ndarray]:
        """``_long_run_grid`` with components under block replacement: the
        middles of cells over one common period of the block and inspection
        intervals. The cells' edges are those of every block-replaced
        component's profile (its long-run availability over its interval,
        cell by cell), the block and inspection times, and enough points in
        between for an inspected component's availability to vary little
        across a cell; so each cell lies in one cell of every profile, and
        the mean over the cells is as exact as the profiles."""
        intervals = {self._preventive[node].interval for node in blocks}
        intervals |= {
            self._inspection[node].interval for node in self._inspection
        }
        schedules = list(self._inspection.values())
        period = _common_period(intervals | {s.period for s in schedules})
        pieces = [np.array([0.0, period])]
        for interval in intervals:
            pieces.append(interval * np.arange(int(round(period / interval))))
        for schedule in schedules:
            pieces.append(np.array(sorted(_tests_in(schedule, period))))
        for node in blocks:
            phase = self._block_cycle(node).phase
            repeats = int(round(period / phase[-1]))
            pieces.append(
                (
                    phase[-1] * np.arange(repeats)[:, None] + phase[None, :-1]
                ).ravel()
            )
        for node in self._inspection:
            rate = self._tested_rate(node)
            interval = self._inspection[node].interval
            per_interval = int(np.ceil(256.0 * rate * interval))
            pieces.append(
                np.linspace(
                    0.0,
                    period,
                    1 + per_interval * int(round(period / interval)),
                )
            )
        edges = np.concatenate(pieces)
        if len(edges) > 4_000_000:
            raise NotImplementedError(
                f"The block-replacement and inspection intervals "
                f"{sorted(intervals)} repeat together only after too long a "
                "time to average over finely enough: estimate the long-run "
                "values by simulation, with availability() or cost()."
            )
        # Edges closer than rounding are one.
        edges = np.unique(np.round(edges / period, 12)) * period
        return 0.5 * (edges[1:] + edges[:-1]), np.diff(edges) / period

    def _block_profile(self, node, times: np.ndarray, rates: bool = False):
        """A block-replaced component's long-run availability (or failure
        intensity) at each of ``times``: the value of its profile's cell
        (over a block interval) that the time falls in."""
        cycle = self._block_cycle(node)
        interval = float(cycle.phase[-1])
        values = cycle.failure_rate if rates else cycle.availability
        phase = times - interval * np.floor(times / interval)
        cell = np.searchsorted(cycle.phase, phase, side="right") - 1
        return values[np.clip(cell, 0, len(values) - 1)]

    def _block_outages(self, working_nodes, broken_nodes) -> float:
        """The system's planned outages per unit time, in the long run,
        from the replacements at block times that take time: at each block
        time, the probability that the system is up just before the
        replacements due then start and down just after, which (as they
        only take units down) is the fall in the system availability. Units
        due at the same time go down together. An inspection due at the
        same time comes first."""
        blocks = [
            node
            for node in self._block_nodes()
            if self._preventive[node].duration is not None
        ]
        if not blocks:
            return 0.0
        intervals = {self._preventive[node].interval for node in blocks}
        intervals |= {
            self._inspection[node].period for node in self._inspection
        }
        period = _common_period(intervals)
        due: dict = {}
        for node in blocks:
            interval = self._preventive[node].interval
            for k in range(int(round(period / interval))):
                instant = round(k * interval / period, 12)
                due.setdefault(instant, []).append(node)
        instants = np.array(sorted(due)) * period
        # Just after each block time, before its replacements start (an
        # inspection due then is done).
        just_after = instants + 1e-9 * period
        up, down = self._availabilities_at(just_after), (
            self._unavailabilities_at(just_after)
        )
        before, after = dict(up), dict(up)
        before_down, after_down = dict(down), dict(down)
        for column, key in enumerate(sorted(due)):
            for node in due[key]:
                cycle = self._block_cycle(node)
                for works, fails, value in (
                    (before, before_down, cycle.before),
                    (after, after_down, cycle.after),
                ):
                    works[node] = np.array(works[node], dtype=float)
                    fails[node] = np.array(fails[node], dtype=float)
                    works[node][column] = value
                    fails[node][column] = 1.0 - value
        # The fall in the system availability, as the rise in its
        # unavailability: a difference of small values in a reliable
        # system, not of values near 1.
        rise = self._system_unreliability(
            self._probabilities_with_overrides(
                after, working_nodes, broken_nodes
            ),
            self._failures_with_overrides(
                after_down, working_nodes, broken_nodes
            ),
        ) - self._system_unreliability(
            self._probabilities_with_overrides(
                before, working_nodes, broken_nodes
            ),
            self._failures_with_overrides(
                before_down, working_nodes, broken_nodes
            ),
        )
        return float(np.sum(rise)) / period

    def _long_run_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """The times, and their weights (which sum to 1), that the exact
        long-run values average over.

        Components with hidden failures are up with a probability that falls
        between inspections and is restored at each one, and components
        inspected at the same times are down together, so the system's
        long-run values are averages over one period of the inspection
        schedules: by Gauss-Legendre quadrature between consecutive
        inspections, on pieces short enough (a failure rate times their
        length at most 1) for it to be exact to rounding. With no such
        components they are constant: one time, of weight 1. A nested RBD
        with hidden failures enters through its own long-run values, which
        is exact only if nothing else varies with the inspections.
        """
        self._require_unlimited_crews()
        self._require_calendars()
        blocks = self._block_nodes()
        if blocks:
            return self._calendar_grid(blocks)
        if not self._inspection:
            return np.zeros(1), np.ones(1)
        rates = {
            node: (self._tested_rate(node), self._inspection[node].interval)
            for node in self._inspection
        }
        intervals = {interval for _, interval in rates.values()}
        schedules = list(self._inspection.values())
        period = _common_period(intervals | {s.period for s in schedules})
        breaks = {0.0, period}
        for interval in intervals:
            count = int(round(period / interval))
            breaks.update(k * interval for k in range(1, count))
        for schedule in schedules:
            breaks.update(_tests_in(schedule, period))
        fastest = max(rate for rate, _ in rates.values())
        # The tests of components whose lives are not exponential: a unit
        # renewed there may not age smoothly from it (a Weibull life's
        # survival, say), so the pieces after it close in on it.
        graded: set = set()
        for node, schedule in self._inspection.items():
            life = self._tested_life(node)
            if life is None:
                continue
            count = int(round(period / schedule.interval))
            starts = schedule.offset + schedule.interval * np.arange(count)
            starts = np.where(starts >= period, starts - period, starts)
            graded.update(round(float(s) / period, 12) for s in starts)
            # Where its life bends within an interval (a threshold, say).
            bends = (starts[:, None] + life.bends[None, :]).ravel()
            breaks.update(float(b) for b in np.mod(bends, period))
        edges = np.array(sorted(breaks))
        points, weights = np.polynomial.legendre.leggauss(16)
        times, masses = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            pieces = np.linspace(
                a, b, max(1, math.ceil((b - a) * fastest)) + 1
            )
            if round(float(a) / period, 12) in graded:
                width = pieces[1] - pieces[0]
                pieces = np.concatenate(
                    [
                        pieces[:1],
                        a + width * 2.0 ** -np.arange(48.0, 0.0, -1.0),
                        pieces[1:],
                    ]
                )
            for lo, hi in zip(pieces[:-1], pieces[1:]):
                half = 0.5 * (hi - lo)
                times.append(lo + half * (points + 1.0))
                masses.append(half * weights)
        return np.concatenate(times), np.concatenate(masses) / period

    def _tested_profile(
        self, node, times: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """A component with hidden failures: its probabilities of being up
        and down at each of ``times``, in the long run.

        Tested at ``offset`` and every ``interval`` after it, it is up with
        probability ``exp(-rate * u)``, ``u`` the time since its last test.
        A test that can miss a failure (a ``coverage`` ``c`` below 1) finds
        what it can: up just after the ``k``-th test since a full one with
        probability ``rho ** k``, where ``rho = 1 - (1 - c) * (1 -
        exp(-rate * interval))`` keeps the units that were up or whose
        failure it found; so up with probability ``rho ** k * exp(-rate *
        u)``. Down is worked out in its own right, by ``expm1``. Any other
        life's profile is its ``TestedLife``'s (#144)."""
        life = self._tested_life(node)
        if life is not None:
            return life.profile(self._tested_phase(node, times))
        rate, interval = self._inspected_rate(node)
        inspection = self._inspection[node]
        position = times - inspection.offset if inspection.offset else times
        if not inspection.partial:
            since = position - interval * np.floor(position / interval)
            return np.exp(-rate * since), -np.expm1(-rate * since)
        cycle = inspection.period
        within = position - cycle * np.floor(position / cycle)
        tests = np.clip(
            np.floor(within / interval), 0, inspection.per_full_test - 1
        )
        exponent = tests * _log_kept(rate, inspection) - rate * (
            within - tests * interval
        )
        return np.exp(exponent), -np.expm1(exponent)

    def _availabilities_at(self, times: np.ndarray) -> dict:
        """Every node's availability at each of ``times`` (see
        ``_long_run_grid``): ``exp(-lambda * u)``, ``u`` the time since the
        last inspection, for a component with hidden failures; its constant
        long-run availability for any other."""
        out: dict = {}
        blocks = set(self._block_nodes())
        for node in self.components:
            if node in self._inspection:
                out[node] = self._tested_profile(node, times)[0]
            elif node in blocks:
                out[node] = self._block_profile(node, times)
            else:
                out[node] = np.full(len(times), self._node_availability(node))
        for node in self.in_or_out:
            out[node] = np.ones(len(times))
        return out

    def _unavailabilities_at(self, times: np.ndarray) -> dict:
        """Every node's unavailability at each of ``times``, as
        ``_availabilities_at`` gives their availabilities, each worked out
        in its own right so that a small one keeps its precision:
        ``1 - exp(-lambda * u)`` by ``expm1`` at a time ``u`` since a test,
        for a component with hidden failures, and ``_node_unavailability``
        for one whose value is constant.
        (A block-replaced component's profile is numerical, to about 1e-7,
        so one less it loses nothing.)"""
        out: dict = {}
        blocks = set(self._block_nodes())
        for node in self.components:
            if node in self._inspection:
                out[node] = self._tested_profile(node, times)[1]
            elif node in blocks:
                out[node] = 1.0 - self._block_profile(node, times)
            else:
                out[node] = np.full(
                    len(times), self._node_unavailability(node)
                )
        for node in self.in_or_out:
            out[node] = np.zeros(len(times))
        return out

    def _long_run_unavailabilities(
        self, working_nodes, broken_nodes
    ) -> Tuple[dict, dict, np.ndarray]:
        """``_long_run_probabilities``, with every node's unavailability
        too (at each time of ``_long_run_grid``, or in each state of the
        repair crews' Markov chain), each worked out in its own right, so
        that the system's unavailability keeps a small one's precision."""
        probabilities, failures, weights, _ = self._long_run_points(
            working_nodes, broken_nodes
        )
        return probabilities, failures, weights

    def _long_run_points(
        self, working_nodes, broken_nodes
    ) -> Tuple[dict, dict, np.ndarray, Optional[np.ndarray]]:
        """``_long_run_unavailabilities``, and each point's time's position
        in ``_long_run_grid`` (with common-cause groups, several points
        share a time: see ``_with_ccf_groups``), or None in the states of
        the repair crews' chain."""
        if self._crews_couple():
            probabilities, weights = self._chain_probabilities(
                working_nodes, broken_nodes
            )
            forced = set(working_nodes or ()) | set(broken_nodes or ())
            chain = self._crew_chain(frozenset(forced))
            # In each state a node in the chain, or held working or broken,
            # is up or down for certain: one less it is exact.
            failures = {
                node: 1.0 - value for node, value in probabilities.items()
            }
            for node, component in self.components.items():
                if node not in forced and node not in chain.nodes:
                    failures[node] = np.full(
                        len(weights), float(component.mean_unavailability())
                    )
            return probabilities, failures, weights, None
        times, weights = self._long_run_grid()
        probabilities = self._probabilities_with_overrides(
            self._availabilities_at(times), working_nodes, broken_nodes
        )
        failures = self._failures_with_overrides(
            self._unavailabilities_at(times), working_nodes, broken_nodes
        )
        index = np.arange(len(times))
        if self.ccf_groups:
            self._require_free_members(working_nodes, broken_nodes)
            probabilities, grouped, weights, index = self._with_ccf_groups(
                times, probabilities, failures, weights
            )
            assert grouped is not None
            failures = grouped
        return probabilities, failures, weights, index

    def _long_run_probabilities(
        self, working_nodes, broken_nodes
    ) -> Tuple[dict, np.ndarray]:
        """The node availabilities the long-run values are evaluated at
        (with the forced nodes held at 1 or 0), over the times of
        ``_long_run_grid``, and those times' weights; or, with limited
        repair crews, over the states of their Markov chain (see
        ``_chain_probabilities``). Each value is then a ratio of averages
        of system quantities."""
        if self._crews_couple():
            return self._chain_probabilities(working_nodes, broken_nodes)
        times, weights = self._long_run_grid()
        probabilities = self._probabilities_with_overrides(
            self._availabilities_at(times), working_nodes, broken_nodes
        )
        if self.ccf_groups:
            self._require_free_members(working_nodes, broken_nodes)
            probabilities, _, weights, _ = self._with_ccf_groups(
                times, probabilities, None, weights
            )
        return probabilities, weights

    def _importance_probabilities(
        self, working_nodes, broken_nodes
    ) -> Tuple[dict, dict, np.ndarray, Optional[np.ndarray]]:
        """``_long_run_points`` for the importance measures. With
        common-cause groups, the long-run points are split by the groups'
        joint states (see ``_with_ccf_groups``), under which the nodes are
        independent; with limited repair crews they are the states of the
        crews' Markov chain, in each of which every node the crews work on
        is up or down for certain (#146)."""
        return self._long_run_points(working_nodes, broken_nodes)

    def _crew_held_importance(
        self, measure: str, working_nodes, broken_nodes
    ) -> dict:
        """With limited repair crews (#146), an importance measure of each
        node from the system's long-run unavailability with the node held
        working, ``U(1_i)``, and held failed, ``U(0_i)``: the crews' Markov
        chain solved without it, as held it needs no crew. ``"birnbaum"``
        is ``U(0_i) - U(1_i)``, ``"improvement"`` ``U - U(1_i)``, ``"raw"``
        ``U(0_i) / U`` and ``"rrw"`` ``U / U(1_i)``. A node already held
        is held the other way for its own measure."""
        working = set(working_nodes or ())
        broken = set(broken_nodes or ())
        self._validate_node_overrides(working, broken)
        base = np.float64(self.mean_unavailability(working, broken))
        out: dict = {}
        for node in self.nodes:
            up = np.float64(
                self.mean_unavailability(working | {node}, broken - {node})
            )
            down = np.float64(
                self.mean_unavailability(working - {node}, broken | {node})
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                out[node] = {
                    "birnbaum": down - up,
                    "improvement": base - up,
                    "raw": down / base,
                    "rrw": base / up,
                }[measure]
        return _squeeze_values(out)

    def _ccf_member_importance(
        self,
        measure: str,
        probabilities: dict,
        failures: dict,
        weights: np.ndarray,
        index: Optional[np.ndarray],
        kind: str = "failure",
    ) -> dict:
        """An importance measure of each common-cause group member, over
        the long-run points of ``_importance_probabilities``, in each of
        which it is up or down for certain. The measures of a node outside
        the groups average, over the long-run times, its measure at each
        time, with it held up and down (the base measures); a member's are
        those of the same averages, with ``A(1_i)`` and ``A(0_i)`` at each
        time the system's availability given the member up and given it
        down then: conditioned on its state, as the shared causes tie the
        other members' to it. So they are the base measures when the
        members are independent (``beta = 0``)."""
        if not self.ccf_groups:
            return {}
        assert index is not None
        p, q, _ = self._node_pairs(probabilities, failures)
        up, down = self._system_probabilities(p, q)
        assert up is not None
        times = int(index.max()) + 1

        def per_time(values):
            # The weighted sum over each time's points.
            return np.bincount(
                index, weights=weights * values, minlength=times
            )

        R, Q = float(weights @ up), float(weights @ down)
        out = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            # The system's at each time, and its probability that it works.
            R_t, Q_t = per_time(up), per_time(down)
            for group in self.ccf_groups:
                for member in group.members:
                    on, off = p[member], q[member]
                    works, fails = per_time(on), per_time(off)
                    r1, q1 = (
                        per_time(on * up) / works,
                        per_time(on * down) / works,
                    )
                    r0, q0 = (
                        per_time(off * up) / fails,
                        per_time(off * down) / fails,
                    )
                    # At each time from whichever end keeps its precision.
                    change = np.where(Q_t <= R_t, q0 - q1, r1 - r0)
                    # Each time's weight: its points', the member's works
                    # and fails with it.
                    weight = works + fails
                    if measure == "birnbaum":
                        value = weight @ change
                    elif measure == "improvement":
                        value = fails @ change
                    elif measure == "raw":
                        value = (weight @ q0) / Q
                    elif measure == "rrw":
                        value = Q / (weight @ q1)
                    elif kind == "failure":
                        value = (fails @ change) / Q
                    else:
                        value = (works @ change) / R
                    out[member] = np.atleast_1d(value)
        return out

    def _standby_rates(self, node) -> Tuple[float, float]:
        """A standby group's units' failure and repair rates, for its exact
        long-run values; raise if their lives and repair times are not
        exponential."""
        component = self.components[node]
        life = _constant_rate(component.reliability)
        repair = _constant_rate(component.time_to_replace)
        if life is None or repair is None:
            raise NotImplementedError(
                f"Component {node!r} is a standby group, whose exact "
                "long-run values come from a Markov chain of its units, "
                "which needs their lives and repair times exponential: "
                "simulate it with availability() or cost()."
            )
        return life, repair

    def _standby_long_run(self, node) -> "_standby_chain.StandbyLongRun":
        """A standby group's long-run values, from its Markov chain (see
        ``_standby_chain``). The crews do not tie it to other nodes here (see
        ``_crews_couple``): its units' repairs wait only for each other,
        with ``repair_crews`` crews when they are the crews' only jobs."""
        life, repair = self._standby_rates(node)
        arrangement = self._standby[node]
        return _standby_chain.long_run(
            arrangement.units,
            arrangement.k,
            life,
            repair,
            arrangement.dormancy_factor,
            arrangement.switching_probability,
            self.repair_crews if self._crews_limited() else None,
        )

    def _node_frequencies(self, node) -> Tuple[float, float, float]:
        """A component's long-run failures, preventive replacements and
        planned outages, per unit time: recursive for a nested
        RepairableRBD (whose own preventive replacements are not this RBD's
        to count), ``1 / (MTTF + MTTR)`` failures for a NonRepairable, and
        from its renewal cycle for one under age replacement."""
        self._require_perfect_repair(node)
        component = self.components[node]
        if isinstance(component, RepairableRBD):
            failures, planned = component._outage_frequencies()
            return failures, 0.0, planned
        if node in self._standby:
            return self._standby_long_run(node).failure_frequency, 0.0, 0.0
        if node in self._inspection:
            # At most one failure per inspection interval: the unit, down
            # from its failure, is renewed at the inspection that finds it.
            life = self._tested_life(node)
            if life is not None:
                return life.failures, 0.0, 0.0
            rate, interval = self._inspected_rate(node)
            if self._inspection[node].partial:
                # It fails at its constant rate while it is up.
                return rate * self._node_availability(node), 0.0, 0.0
            return float(-np.expm1(-rate * interval) / interval), 0.0, 0.0
        schedule = self._preventive.get(node)
        if schedule is None:
            return component.failure_frequency(), 0.0, 0.0
        _, cycle, failures, maintenances = self._maintenance_cycle(
            node, schedule
        )
        maintained = maintenances / cycle
        planned = 0.0 if schedule.duration is None else maintained
        return failures / cycle, maintained, planned

    def _maintenance_cycle(
        self, node, schedule: _Preventive
    ) -> Tuple[float, float, float, float]:
        """A component's renewal cycle under preventive maintenance: its mean
        up time, mean length, mean number of failures and mean number of
        preventive replacements.

        Under age replacement a cycle ends at a failure or at a preventive
        replacement, whichever comes first. Under block replacement it runs
        from one block time at which the unit is up (and replaced) to the
        next, with any failures and repairs in between (see
        ``_block_replacement``); replaced on condition, from one inspection
        that replaces the unit to the next (see ``_condition_replacement``).
        It is computed once and kept."""
        self._require_no_opportunities(node)
        self._require_perfect_repair(node)
        component = self.components[node]
        if schedule.policy == "block":
            block = self._block_cycle(node)
            return block.up, block.length, block.failures, 1.0
        if schedule.policy == "condition":
            kept = self._block_cycle(node)
            return kept.up, kept.length, kept.failures, kept.replaced
        # Kept by interval, as a search over intervals revisits them.
        cache = self.__dict__.setdefault("_age_cycles", {})
        key = (
            node,
            id(component.reliability),
            id(component.time_to_replace),
            float(schedule.interval),
            id(schedule.duration),
        )
        if key in cache:
            return cache[key]
        up = float(component.avg_replacement_time(schedule.interval))
        survives = float(
            np.ravel(component.reliability_function(schedule.interval))[0]
        )
        fails = _failed_by(component, schedule.interval, survives)
        maintenance = (
            0.0 if schedule.duration is None else model_mean(schedule.duration)
        )
        cycle = (
            up
            + fails * model_mean(component.time_to_replace)
            + survives * maintenance
        )
        cache[key] = (up, cycle, fails, survives)
        return cache[key]

    def system_failure_frequency(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run failure frequency (failures per unit
        time), by the Birnbaum/Vesely formula.

        In steady state the system failure frequency is

        ```text
        omega_sys = sum_i I_B(i) * omega_i
        ```

        where ``I_B(i)`` is node i's Birnbaum importance evaluated at the
        nodes' long-run availabilities (see ``birnbaum_importance``) and
        ``omega_i`` is node i's long-run failure frequency: ``1 / (MTTF_i +
        MTTR_i)`` for a component, ``F(T) / C`` for one under age
        replacement at ``T`` (``C`` its mean renewal cycle, see
        ``expected_cost_rate``), its mean failures per renewal cycle over
        the cycle's mean length for one under block replacement, and a
        nested ``RepairableRBD``'s own
        ``system_failure_frequency``. Exact for independent repairable
        nodes, with no simulation. Each Birnbaum importance is a sum of
        products of the node availabilities and unavailabilities, so a small
        frequency (a reliable redundant system's, say) keeps its full
        relative precision. Every system failure counts, including
        the zero-length outages an instantly repaired component causes;
        planned outages (preventive maintenance) are not failures.

        With limited ``repair_crews`` the formula is summed over the states
        of the crews' Markov chain (see ``mean_availability``): in each, a
        component that is up fails at its constant rate, and takes the
        system down if it is critical there.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working (they contribute no
            failures), by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed (they contribute no
            failures), by default None.

        Returns
        -------
        float
            Expected number of system failures per unit time, in the long
            run.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.
        NotImplementedError
            If a component is under block replacement with models its exact
            values do not cover (see ``node_availability``); or, while a
            component can wait for a repair crew, as for
            ``mean_availability``.

        Examples
        --------
        Two components in series, with MTTFs 5 and 2 and MTTRs of 1:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.system_failure_frequency(), 4)  # 2/3 * 1/6 + 5/6 * 1/3
        0.3889
        >>> round(rbd.system_failure_frequency(working_nodes=["a"]), 4)
        0.3333
        """
        return self._outage_frequencies(working_nodes, broken_nodes)[0]

    def _outage_frequencies(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> Tuple[float, float]:
        """The system's long-run failures and planned outages per unit time:
        ``sum_i I_B(i) * omega_i`` over the nodes' failures, and the same
        over their planned outages (a node's outage takes the system down
        when the node is critical, which it is with probability
        ``I_B(i)``). With limited repair crews, see
        ``_chain_outage_frequencies``."""
        if self._crews_couple():
            return self._chain_outage_frequencies(working_nodes, broken_nodes)
        if self.ccf_groups:
            return self._ccf_outage_frequencies(working_nodes, broken_nodes)
        times, weights = self._long_run_grid()
        availability = self._probabilities_with_overrides(
            self._availabilities_at(times), working_nodes, broken_nodes
        )
        unavailability = self._failures_with_overrides(
            self._unavailabilities_at(times), working_nodes, broken_nodes
        )
        forced = (set() if working_nodes is None else set(working_nodes)) | (
            set() if broken_nodes is None else set(broken_nodes)
        )
        # Each node's Birnbaum importance, from the nodes' unavailabilities
        # as well as their availabilities, so that a small one (a node
        # backed up by reliable redundancy) keeps its precision.
        birnbaum = super()._birnbaum_importance(
            availability, node_failures=unavailability
        )
        failures = planned = 0.0
        blocks = set(self._block_nodes())
        for node in self.components:
            if node in forced:
                # A forced node never changes state, so it contributes no
                # system failures.
                continue
            importance = np.asarray(birnbaum[node])
            if node in self._inspection:
                # It fails at its constant rate whenever it is up (any other
                # life: at its profile's rate).
                node_failures: Any = self._tested_intensity(
                    node, times, availability[node]
                )
                node_planned: Any = 0.0
            elif node in blocks:
                # Its failure intensity varies over its block interval; its
                # replacements fall at block times (counted below).
                node_failures = self._block_profile(node, times, rates=True)
                node_planned = 0.0
            else:
                node_failures, _, node_planned = self._node_frequencies(node)
            failures += float(weights @ (importance * node_failures))
            planned += float(weights @ (importance * node_planned))
        if blocks:
            planned += self._block_outages(working_nodes, broken_nodes)
        return failures, planned

    def mean_time_between_failures(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run Mean Time Between Failures.

        Exact, to full precision however rare the failures:
        ``MTBF = 1 / system_failure_frequency`` — the mean length of
        one full up-down cycle, i.e. ``MTBF = MUT + MDT`` (when there are
        no planned outages: preventive maintenance that takes time also
        ends up periods, so MTBF is then longer). Infinite if the system
        never fails (e.g. a node held working keeps it up, or one held
        broken keeps it down).

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        float
            The mean time between system failures, possibly ``inf``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        Two components in series, with MTTFs 5 and 2 and MTTRs of 1:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.mean_up_time(), 4)  # 1 / (0.2 + 0.5)
        1.4286
        >>> round(rbd.mean_down_time(), 4)
        1.1429
        >>> round(rbd.mean_time_between_failures(), 4)  # MUT + MDT
        2.5714
        """
        omega = self.system_failure_frequency(working_nodes, broken_nodes)
        return 1.0 / omega if omega > 0.0 else float("inf")

    def mean_up_time(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run Mean Up Time.

        Exact: the mean duration of an uninterrupted working period of the
        system (sometimes called the repairable-system MTTF),
        ``MUT = mean_availability / system_failure_frequency``, to full
        precision however rare the failures. A planned
        outage (preventive maintenance that takes time) also ends a working
        period, so its frequency is added to the failure frequency. If the
        system never goes down it is infinite when the system is up and
        0.0 when it is always down.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        float
            The mean up time, possibly ``inf``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        For exponential failures in series, MUT is one over the sum of the
        failure rates:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.mean_up_time(), 4)  # 1 / (0.2 + 0.5)
        1.4286
        >>> round(rbd.mean_up_time(working_nodes=["a"]), 4)  # b alone
        2.0
        """
        availability = self.mean_availability(working_nodes, broken_nodes)
        omega = sum(self._outage_frequencies(working_nodes, broken_nodes))
        if omega > 0.0:
            return float(availability) / omega
        return float("inf") if availability > 0.0 else 0.0

    def mean_down_time(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> float:
        """Returns the system's long-run Mean Down Time.

        Exact: the mean duration of a system outage (the repairable-system
        MTTR), ``MDT = mean_unavailability / system_failure_frequency``,
        both worked out in their own right, so a reliable system's keeps
        its precision. Planned outages (preventive maintenance that takes
        time) count as
        outages, so their frequency is added to the failure frequency. If
        the system never goes down it is 0.0 when the system is always up
        and infinite when it is down.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes always working, by default None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        float
            The mean down time, possibly ``inf``.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        A parallel pair is down only while both are, and the outage ends
        when either is repaired:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> round(rbd.mean_down_time(), 4)  # 1 / (1.0 + 1.0)
        0.5
        >>> rbd.mean_down_time(working_nodes=["a"])  # never down
        0.0
        """
        unavailability = self.mean_unavailability(working_nodes, broken_nodes)
        omega = sum(self._outage_frequencies(working_nodes, broken_nodes))
        if omega > 0.0:
            return unavailability / omega
        return 0.0 if unavailability <= 0.0 else float("inf")

    def birnbaum_importance(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Returns the Birnbaum measure of importance for all nodes,
        evaluated at the nodes' long-run availabilities.

        Exact, with no simulation: ``I_B(i) = A_sys(A_i = 1) -
        A_sys(A_i = 0)``, the system's long-run availability with node i
        always working minus that with node i always failed. It is the
        probability that node i is critical (the system works if and only
        if node i does), and how much the system availability changes per
        unit change in node i's availability. The node availabilities are
        those of ``node_availability``, with ``working_nodes`` and
        ``broken_nodes`` held at 1 and 0. It is worked out for every node
        at once, as the derivative of the system availability: a sum of
        products of the node availabilities and unavailabilities (each
        worked out in its own right, see ``mean_unavailability``), so a
        small one keeps its full relative precision, where the difference
        above would cancel in a reliable system.

        Note: Birnbaum's measure of importance assumes all nodes are
        independent. With common-cause groups, a member's values with it
        working and failed are the system's availability *given* its state
        at each long-run time (the groups' Markov chains give the members'
        joint states), averaged over the times as every node's are; a node
        outside the groups is held up and down, as without them. The other
        measures are built on the same values, and ``beta = 0`` gives them
        as without the group.

        With limited ``repair_crews``, under which a component can wait for
        a crew, the components are not independent: node i is then held
        working and failed in the crews' Markov chain, which is solved
        without it (held either way, it needs no crew), and ``I_B(i)`` is
        the difference of the system's long-run availabilities so held. The
        improvement potential, RAW and RRW are built on the same values;
        the criticality and Fussell-Vesely measures are probabilities over
        the chain's states. With a crew for each component they are the
        independent ones (#146).

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Birnbaum importances as
            values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        In series, a node is critical exactly when the others work:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> importance = rbd.birnbaum_importance()
        >>> {node: round(i, 4) for node, i in importance.items()}
        {'a': 0.6667, 'b': 0.8333}
        >>> round(rbd.birnbaum_importance(working_nodes=["a"])["b"], 4)
        1.0
        """
        if self._crews_couple():
            return self._crew_held_importance(
                "birnbaum", working_nodes, broken_nodes
            )
        node_probabilities, node_failures, weights, index = (
            self._importance_probabilities(working_nodes, broken_nodes)
        )
        return _squeeze_values(
            {
                **super()._birnbaum_importance(
                    node_probabilities, weights, node_failures
                ),
                **self._ccf_member_importance(
                    "birnbaum",
                    node_probabilities,
                    node_failures,
                    weights,
                    index,
                ),
            }
        )

    def improvement_potential(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Returns the improvement potential of all nodes, evaluated at the
        nodes' long-run availabilities.

        Exact, with no simulation: ``A_sys(A_i = 1) - A_sys``, the gain in
        the system's long-run availability if node i never failed, worked
        out as ``I_B(i) * (1 - A_i)`` (see ``birnbaum_importance``) so that
        a small one keeps its precision. The node availabilities are those
        of ``node_availability``, with ``working_nodes`` and
        ``broken_nodes`` held at 1 and 0.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and improvement potentials as
            values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> potential = rbd.improvement_potential()
        >>> {node: round(p, 4) for node, p in potential.items()}
        {'a': 0.1111, 'b': 0.2778}
        """
        if self._crews_couple():
            return self._crew_held_importance(
                "improvement", working_nodes, broken_nodes
            )
        node_probabilities, node_failures, weights, index = (
            self._importance_probabilities(working_nodes, broken_nodes)
        )
        return _squeeze_values(
            {
                **super()._improvement_potential(
                    node_probabilities, weights, node_failures
                ),
                **self._ccf_member_importance(
                    "improvement",
                    node_probabilities,
                    node_failures,
                    weights,
                    index,
                ),
            }
        )

    def risk_achievement_worth(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Returns the RAW importance per Modarres & Kaminskiy, evaluated at
        the nodes' long-run availabilities. That is RAW_i =
        (unavailability of system given i failed) /
        (nominal system unavailability).

        Exact, with no simulation: ``RAW_i = U_sys(A_i = 0) / U_sys``,
        where ``U = 1 - A`` is long-run unavailability: the factor by which
        the system's unavailability grows if node i is always failed. The
        node availabilities are those of ``node_availability``, with
        ``working_nodes`` and ``broken_nodes`` held at 1 and 0. Where the
        nominal unavailability is 0 (e.g. every component is repaired
        instantly), the ratio is ``inf`` or ``nan``, with a numpy warning.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RAW importances as
            values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        In series the system is down whenever a node is, so failing any
        node multiplies the unavailability by ``1 / U_sys``:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> raw = rbd.risk_achievement_worth()
        >>> {node: round(r, 4) for node, r in raw.items()}
        {'a': 2.25, 'b': 2.25}
        """
        if self._crews_couple():
            return self._crew_held_importance(
                "raw", working_nodes, broken_nodes
            )
        node_probabilities, node_failures, weights, index = (
            self._importance_probabilities(working_nodes, broken_nodes)
        )
        return _squeeze_values(
            {
                **super()._risk_achievement_worth(
                    node_probabilities, weights, node_failures
                ),
                **self._ccf_member_importance(
                    "raw",
                    node_probabilities,
                    node_failures,
                    weights,
                    index,
                ),
            }
        )

    def risk_reduction_worth(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Returns the RRW importance per Modarres & Kaminskiy, evaluated at
        the nodes' long-run availabilities. That is RRW_i =
        (nominal unavailability of system) /
        (unavailability of system given i is working).

        Exact, with no simulation: ``RRW_i = U_sys / U_sys(A_i = 1)``,
        where ``U = 1 - A`` is long-run unavailability: the factor by which
        the system's unavailability would fall if node i never failed. The
        node availabilities are those of ``node_availability``, with
        ``working_nodes`` and ``broken_nodes`` held at 1 and 0. It is
        ``inf`` (with a numpy warning) where making node i perfect would
        make the system perfect, as for either node of a parallel pair, and
        ``nan`` if the system is already never down.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and RRW importances as
            values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; or if a component has a
            non-parametric reliability model.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> rrw = rbd.risk_reduction_worth()
        >>> {node: round(r, 4) for node, r in rrw.items()}
        {'a': 1.3333, 'b': 2.6667}
        """
        if self._crews_couple():
            return self._crew_held_importance(
                "rrw", working_nodes, broken_nodes
            )
        node_probabilities, node_failures, weights, index = (
            self._importance_probabilities(working_nodes, broken_nodes)
        )
        return _squeeze_values(
            {
                **super()._risk_reduction_worth(
                    node_probabilities, weights, node_failures
                ),
                **self._ccf_member_importance(
                    "rrw",
                    node_probabilities,
                    node_failures,
                    weights,
                    index,
                ),
            }
        )

    def criticality_importance(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        kind: str = "failure",
    ) -> dict[Any, float]:
        """Returns the criticality importance of all nodes, evaluated at the
        nodes' long-run availabilities.

        Exact, with no simulation. With ``I_B(i)`` the Birnbaum importance,
        ``A_i`` node i's availability and ``A_sys`` the system's:

        - ``kind="failure"`` (the default) gives the failure-oriented form
          (Rausand & Høyland), ``I_B(i) * (1 - A_i) / (1 - A_sys)``: the
          probability that node i is down and critical, given that the
          system is down -- the share of the system's downtime that node i
          accounts for. It ranks nodes in series by their unavailability.
          It is computed from the node unavailabilities, through the
          minimal cut sets, so the system unavailability is not lost to
          cancellation in ``1 - A_sys`` however available the system is.
          It is ``nan`` if the system is never down.
        - ``kind="success"`` gives the success-oriented form,
          ``I_B(i) * A_i / A_sys``: the probability that node i is working
          and critical, given that the system is working. It is 1 for every
          node in series with the rest of the system, so it cannot rank
          them. It is ``nan`` if the system is never up.

        The node availabilities are those of ``node_availability``, with
        ``working_nodes`` and ``broken_nodes`` held at 1 and 0.

        Parameters
        ----------
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.
        kind : str, optional
            ``"failure"`` (the default) or ``"success"``.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and criticality importances
            as values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets; if a component has a non-parametric
            reliability model; or if ``kind`` is neither ``"failure"`` nor
            ``"success"``.
        NotImplementedError
            As for ``mean_availability``.

        References
        ----------
        M. Rausand and A. Høyland, System Reliability Theory: Models,
        Statistical Methods, and Applications, 2nd edition, Wiley, 2004.

        Examples
        --------
        Two nodes in series, ``b`` down twice as often as ``a``: ``b``
        accounts for more of the system's downtime (the shares add to less
        than 1, as neither is critical while both are down):

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> criticality = rbd.criticality_importance()
        >>> {node: round(c, 4) for node, c in criticality.items()}
        {'a': 0.25, 'b': 0.625}

        The success-oriented form cannot tell them apart:

        >>> criticality = rbd.criticality_importance(kind="success")
        >>> {node: round(c, 4) for node, c in criticality.items()}
        {'a': 1.0, 'b': 1.0}
        """
        node_probabilities, node_failures, weights, index = (
            self._importance_probabilities(working_nodes, broken_nodes)
        )
        return _squeeze_values(
            {
                **super()._criticality_importance(
                    node_probabilities,
                    kind,
                    weights=weights,
                    node_failures=node_failures,
                ),
                **self._ccf_member_importance(
                    "criticality",
                    node_probabilities,
                    node_failures,
                    weights,
                    index,
                    kind=kind,
                ),
            }
        )

    def fussell_vesely(
        self,
        fv_type: str = "c",
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "exact",
    ) -> dict[Any, float]:
        """Calculate Fussell-Vesely importance of all nodes, evaluated at the
        nodes' long-run availabilities.

        The Fussell-Vesely importance of node i is the probability that
        some minimal cut set containing node i has failed (all its nodes
        down), over the probability that the system has failed: the share
        of the system's unavailability that involves node i. Here a node's
        probability of having failed is its long-run unavailability,
        ``1 - A_i``, from ``node_availability`` (with ``working_nodes`` and
        ``broken_nodes`` held at availability 1 and 0), and the system's is
        ``1 - A_sys``; both are exact, with no simulation, and worked out
        in their own right (see ``mean_unavailability``), so small ones
        keep their precision. ``method="exact"`` (the default) works out
        the probability that some cut set containing node i has failed
        exactly, from the exact engine, so the measure is between 0 and 1.
        ``method="rare_event"`` sums the cut sets' probabilities instead,
        the usual rare-event approximation, which with large
        unavailabilities can exceed 1. If the system never fails, the ratio
        is ``nan`` or ``inf``, with a numpy warning.

        Typically this measure is implemented using cut-sets as mentioned
        above, although it can be implemented using path-sets. Both are
        implemented here, selected by ``fv_type``: ``"c"`` takes the
        minimal cut sets containing node i, ``"p"`` the minimal path sets
        containing it. Either way a set has failed when all of its nodes
        have, and the probability that one of them has (or, with
        ``"rare_event"``, the sum of their probabilities) is divided by the
        system's unavailability.

        Parameters
        ----------
        fv_type : str, optional
            Dictates the method of calculation, "c" = cut-set and
            "p" = path-set, by default "c".
        working_nodes : Collection[Hashable], optional
            Condition on these nodes being always available, by default
            None.
        broken_nodes : Collection[Hashable], optional
            Condition on these nodes being failed, by default None.
        method : str, optional
            ``"exact"`` (the default) or ``"rare_event"``.

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If ``fv_type`` is not 'c' (cut-set) or 'p' (path-set), or
            ``method`` not 'exact' or 'rare_event'; if a working/broken node
            is unknown, is the input or output node, or is in both sets; or
            if a component has a non-parametric reliability model.
        NotImplementedError
            As for ``mean_availability``.

        Examples
        --------
        In series each node is a cut set on its own, so its importance is
        its unavailability over the system's:

        >>> import surpyval as surv
        >>> from repyability import RepairableRBD
        >>> E = surv.Exponential.from_params
        >>> rbd = RepairableRBD(
        ...     [("s", "a"), ("a", "b"), ("b", "t")],
        ...     {
        ...         "a": {"reliability": E([0.2]), "repairability": E([1.0])},
        ...         "b": {"reliability": E([0.5]), "repairability": E([1.0])},
        ...     },
        ... )
        >>> fv = rbd.fussell_vesely()
        >>> {node: round(v, 4) for node, v in fv.items()}
        {'a': 0.375, 'b': 0.75}
        """
        node_probabilities, node_failures, weights, index = (
            self._importance_probabilities(working_nodes, broken_nodes)
        )
        return _squeeze_values(
            super()._fussell_vesely(
                node_probabilities,
                fv_type,
                method,
                weights=weights,
                node_failures=node_failures,
            )
        )

    def fussel_vesely(self, fv_type: str = "c") -> dict[Any, float]:
        """Deprecated alias for ``fussell_vesely`` (corrected spelling).

        Deprecated: use ``fussell_vesely`` instead; this alias will be
        removed in 0.12. It returns ``fussell_vesely(fv_type)``
        and, unlike it, takes no ``working_nodes`` or ``broken_nodes``.

        Parameters
        ----------
        fv_type : str, optional
            "c" = cut-set and "p" = path-set, by default "c".

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values.

        Raises
        ------
        ValueError
            As for ``fussell_vesely``.

        Warns
        -----
        FutureWarning
            On every call.
        """
        warnings.warn(
            "fussel_vesely() is deprecated; use fussell_vesely() "
            f"(Fussell-Vesely). This alias will be removed in {REMOVAL}.",
            FutureWarning,
            stacklevel=2,
        )
        return self.fussell_vesely(fv_type)
