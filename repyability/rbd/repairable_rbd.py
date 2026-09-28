"""Repairable reliability block diagrams: the ``RepairableRBD`` class.

Besides the class, the module holds the helpers its availability
simulation uses: the ``Event`` records of its event queue, the timeline
arithmetic behind the up/down criticality measures (``combined_timeline``,
``intersection``, ``union``, ``time_at_status``) and the failure and
restoration criticality index ratios.
"""

import heapq
import itertools
import math
import pprint
import warnings
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from scipy.optimize import OptimizeResult, brentq, minimize
from scipy.special import expit, logit, logsumexp, softmax
from surpyval import ExactEventTime
from tqdm import tqdm

from repyability.non_repairable import NonRepairable
from repyability.rbd import _montecarlo as montecarlo
from repyability.rbd._block_replacement import BlockCycle, block_cycle
from repyability.rbd._model_utils import failure_time_scale, model_mean
from repyability.rbd._sampling import UniformStream, draw, inverse_sampler
from repyability.rbd.rbd import RBD, _check_on_infeasible_rbd
from repyability.rbd.redundancy_allocation import (
    lowest_total_cost,
    redundancy_caps,
)
from repyability.rbd.results import (
    AvailabilityAllocation,
    AvailabilityResult,
    ConfidenceInterval,
    CostResult,
    Criticalities,
    FailureCriticalityIndex,
    MaintenancePlan,
    RestorationCriticalityIndex,
    TotalCostAllocation,
    UpDownImportance,
)


class _StreamedRBD:
    """Stands in for a nested :class:`RepairableRBD` component during
    ``availability()``: runs the nested RBD's own simulation, with its
    components' draws replayed from the shared :class:`UniformStream` by
    their own stand-ins (see ``RepairableRBD._streamed_components``)."""

    def __init__(self, rbd: "RepairableRBD", sources: dict):
        self._rbd = rbd
        self._sources = sources

    def initialize_event_queue(self, t_simulation):
        self._rbd.initialize_event_queue(t_simulation, sources=self._sources)

    def next_event(self):
        return self._rbd.next_event(sources=self._sources)

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
    """

    __slots__ = ("_heap",)

    def __init__(self):
        self._heap: list = []

    def put(self, event) -> None:
        heapq.heappush(self._heap, event)

    def get(self):
        return heapq.heappop(self._heap)

    def empty(self) -> bool:
        return not self._heap

    def qsize(self) -> int:
        return len(self._heap)


class _StreamedComponent:
    """Stands in for a :class:`NonRepairable` component during
    ``availability()``: alternates failure and repair draws exactly as
    ``NonRepairable.next_event`` does, but takes them from a shared
    :class:`UniformStream`, which reproduces the same numbers."""

    def __init__(self, failure, repair, stream: UniformStream):
        self._failure = failure
        self._repair = repair
        self._stream = stream
        self._fails_next = True
        # The time-to-maintain sampler of a node under preventive
        # maintenance, or the time-to-test sampler of an inspected node (set
        # by RepairableRBD._streamed_components).
        self.maintenance: Any = None

    def reset(self):
        self._fails_next = True

    def maintenance_time(self) -> float:
        return self._stream.draw(self.maintenance)

    def next_event(self):
        if self._fails_next:
            self._fails_next = False
            return self._stream.draw(self._failure), False
        self._fails_next = True
        return self._stream.draw(self._repair), True


def _stand_in(component, stream, made: dict, path: tuple = ()):
    """A component's stand-in for ``RepairableRBD._streamed_components``, or
    ``None`` if its draws cannot be streamed. (A subclass may draw its events
    its own way, so only the classes themselves are streamed.) ``path`` is
    the node's place, nested RBDs included, which names its own stream under
    keyed streams."""
    if type(component) is RepairableRBD:
        nested = component._streamed_components(stream, made, path)
        return None if nested is None else _StreamedRBD(component, nested)
    if type(component) is not NonRepairable:
        return None
    failure = inverse_sampler(component.reliability)
    repair = inverse_sampler(component.time_to_replace)
    if failure is None or repair is None:
        return None
    return _StreamedComponent(failure, repair, _stream_for(stream, path))


def _stream_for(stream, path: tuple):
    """The stream a component at ``path`` draws from: the shared one, or,
    under keyed streams, its own."""
    if isinstance(stream, _KeyedStreams):
        return stream.for_node(path)
    return stream


#: Replications per block of a parallel run (see ``RepairableRBD``'s
#: ``availability``): each block is seeded by its position, so a parallel
#: run's results do not depend on the number of processes.
PARALLEL_BLOCK = 250


class _Tally:
    """The running totals of an availability simulation: added to one
    replication at a time, and merged across batches and processes."""

    def __init__(self, costs: dict):
        self.n = 0
        # Time -> net change in the number of simulated systems working.
        self.changes: dict = defaultdict(int)
        self.RCI: dict = defaultdict(Counter)
        self.FCI: dict = defaultdict(Counter)
        self.system_restorations = 0
        self.system_failures = 0
        self.system_planned_outages = 0
        self.system_uptime: float = 0
        self.system_downtime: float = 0
        self.node_uptime: dict = defaultdict(int)
        self.node_downtime: dict = defaultdict(int)
        self.intersection_uptime: dict = defaultdict(int)
        self.intersection_downtime: dict = defaultdict(int)
        self.union_uptime: dict = defaultdict(int)
        self.union_downtime: dict = defaultdict(int)
        # One system uptime (and, when priced, one total cost) per
        # replication, in order.
        self.uptimes: List[float] = []
        self.cost_samples: List[float] = []
        self.cost_by_category = {
            "repair": 0.0,
            "replace": 0.0,
            "preventive": 0.0,
            "inspection": 0.0,
            "component_downtime": 0.0,
            "system_downtime": 0.0,
        }
        self.cost_by_component = {node: 0.0 for node in costs}

    def pay(self, node, category: str, charges) -> float:
        """The next amount of a stream of charges, booked to its node and
        category (a node or category with no stream costs nothing)."""
        if charges is None:
            return 0.0
        charge = next(charges)
        self.cost_by_category[category] += charge
        self.cost_by_component[node] += charge
        return charge

    def merge(self, other: "_Tally") -> None:
        """Add ``other``'s replications, which follow this tally's."""
        self.n += other.n
        for t, change in other.changes.items():
            self.changes[t] += change
        for mine, theirs in ((self.RCI, other.RCI), (self.FCI, other.FCI)):
            for node, counts in theirs.items():
                mine[node].update(counts)
        for name in (
            "system_restorations",
            "system_failures",
            "system_planned_outages",
            "system_uptime",
            "system_downtime",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for name in (
            "node_uptime",
            "node_downtime",
            "intersection_uptime",
            "intersection_downtime",
            "union_uptime",
            "union_downtime",
        ):
            mine = getattr(self, name)
            for node, value in getattr(other, name).items():
                mine[node] += value
        self.uptimes.extend(other.uptimes)
        self.cost_samples.extend(other.cost_samples)
        for key, value in other.cost_by_category.items():
            self.cost_by_category[key] += value
        for key, value in other.cost_by_component.items():
            self.cost_by_component[key] += value


class _NodeStream:
    """One component's uniforms under keyed streams: a fresh generator for
    each replication, keyed by the seed, the component's place and the
    replication, so that another RBD with a component in the same place
    draws the same numbers there (common random numbers).

    For antithetic pairs, the two replications of a pair share the
    generator, and the second takes ``1 - u`` for each of its uniforms
    ``u``: the component's ``k``-th draw in the second is ``1 - u`` for its
    ``k``-th in the first, whatever the other components do."""

    def __init__(self, keyed: "_KeyedStreams", key: int):
        self._keyed = keyed
        self._key = key
        self._replication = -1
        self._rng: Any = None
        self._flip = False
        self._block = np.empty(0)
        self._pos = 0
        self._values: dict = {}

    def draw(self, sampler) -> float:
        if self._replication != self._keyed.replication:
            self._replication = replication = self._keyed.replication
            self._flip = self._keyed.antithetic and replication % 2 == 1
            if self._keyed.antithetic:
                replication //= 2
            self._rng = np.random.default_rng(
                [self._keyed.entropy, self._key, replication]
            )
            self._block = np.empty(0)
            self._pos = 0
        if self._pos == len(self._block):
            block = self._rng.random(64)
            self._block = 1.0 - block if self._flip else block
            self._pos = 0
            self._values = {}
        values = self._values.get(sampler)
        if values is None:
            values = self._values[sampler] = np.asarray(
                sampler(self._block), dtype=float
            ).tolist()
        value = values[self._pos]
        self._pos += 1
        return value


class _KeyedStreams:
    """Streams of uniforms keyed by component (see :class:`_NodeStream`),
    for common random numbers or, with ``antithetic``, antithetic pairs of
    replications."""

    def __init__(self, entropy: int, antithetic: bool = False):
        self.entropy = entropy
        self.antithetic = antithetic
        self.replication = 0
        self._streams: dict = {}

    def for_node(self, path: tuple) -> _NodeStream:
        key = zlib.crc32(repr(path).encode())
        if key not in self._streams:
            self._streams[key] = _NodeStream(self, key)
        return self._streams[key]

    def begin(self, replication: int) -> None:
        self.replication = replication

    def close(self) -> None:
        pass


def _stopping_rule(
    N: int,
    tolerance: Optional[float],
    confidence: float,
    max_N: Optional[int],
    antithetic: bool,
    target: str,
    t_simulation: float,
) -> Optional[Callable[["_Tally"], int]]:
    """``None`` for a fixed number of replications; otherwise a function
    of the tally so far giving how many more replications to run (0 once
    the confidence interval of the mean availability over the window, or of
    the mean cost, is at most ``tolerance`` either side, or ``max_N`` have
    run: then with a warning)."""
    montecarlo.check_confidence(confidence)
    limit = montecarlo.sample_limit(
        N, tolerance, max_N, antithetic, ("N", "max_N")
    )
    if limit is None:
        return None

    def stop(tally: "_Tally") -> int:
        if target == "cost":
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
            "max_N",
        )

    return stop


def _simulate_block(task) -> "_Tally":
    """One block of a parallel run (in its own process)."""
    rbd, t, working, broken, method, n, seed, antithetic = task
    return rbd._run(t, working, broken, method, n, False, seed, antithetic)


@dataclass(order=True)
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


class _Preventive(NamedTuple):
    """A node's scheduled preventive maintenance: every ``interval``
    under the ``"age"`` or ``"block"`` policy, taking a time drawn from
    ``duration`` (None: no time)."""

    interval: float
    policy: str
    duration: Any

    def due(self, renewed: float) -> float:
        """When the next preventive action falls, for a unit put into
        service as new at ``renewed``."""
        if self.policy == "age":
            return renewed + self.interval
        # Block: the next multiple of the interval after ``renewed`` (a unit
        # renewed on the schedule is not renewed again there).
        due = float(np.floor(renewed / self.interval) + 1.0) * self.interval
        return due if due > renewed else due + self.interval


class _Inspection(NamedTuple):
    """A node's periodic inspection, which is what finds its (hidden)
    failures: at every multiple of ``interval``, taking a time drawn from
    ``duration`` (None: no time)."""

    interval: float
    duration: Any

    def due(self, t: float) -> float:
        """The first inspection after ``t``."""
        k = np.floor(t / self.interval) + 1.0
        due = float(k * self.interval)
        return due if due > t else float((k + 1.0) * self.interval)

    def finds(self, t: float) -> float:
        """The inspection that finds a failure at ``t``: the first at or
        after it."""
        k = np.ceil(t / self.interval)
        due = float(k * self.interval)
        return due if due >= t else float((k + 1.0) * self.interval)


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


def _charges(cost, rng, batch: int = 1024) -> Iterator[float]:
    """The endless stream of amounts charged for one per-failure cost: the
    number itself every time, or a fresh draw from its distribution at each
    failure (drawn ``batch`` at a time, since ``qf`` is vectorised). Draws are
    clipped at 0; validation has already made a negative one negligible."""
    while True:
        if isinstance(cost, float):
            yield cost
        else:
            draws = np.ravel(cost.qf(rng.random(batch)))
            yield from np.maximum(draws, 0.0).tolist()


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
    and are repaired independently of one another and of the system state:
    there is no shared repair crew, and a component keeps running (and can
    fail) while the system is down. The system is up whenever its working
    components connect the input node to the output node.

    Two kinds of analysis are offered:

    - Exact long-run (steady-state) metrics, built from each component's
      availability ``MTTF / (MTTF + MTTR)`` and failure frequency
      ``1 / (MTTF + MTTR)``: ``mean_availability``, ``node_availability``,
      ``system_failure_frequency``, ``mean_up_time``, ``mean_down_time``,
      ``mean_time_between_failures``, ``expected_cost_rate`` and the
      importance measures. A component under age or block replacement
      enters through its renewal cycle instead (see
      ``node_availability``).
    - Monte-Carlo simulation of a finite window ``[0, t_simulation]`` that
      starts with every component working: ``availability`` (availability
      over time, and criticality measures) and ``cost`` (the distribution
      of the window's cost).

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
          then. The replacement takes a time drawn from ``"duration"``, a
          time-to-maintain model, during which the unit is down (a planned
          outage); ``"instant"`` (the default) takes no time, renewing the
          unit in place. Each replacement is charged ``"cost"``, a number
          or a distribution drawn afresh each time. The unit comes back as
          new: a failure it had yet to reach never happens. An
          ``interval`` of ``inf`` never maintains.
          ``"inspection"`` makes the component's failures *hidden*: a
          failure takes it down, but nobody knows until an inspection (a
          proof test) finds it, and only then does its repair start. It
          is a dict with an ``"interval"`` and optional ``"duration"`` and
          ``"cost"``: the component is inspected at every multiple of the
          (positive, finite) interval, from time 0. The test takes a time
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
          ``total_cost`` and ``allocate_redundancy``.
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
        ``"duration"`` and ``"cost"``.
    INSPECTION_KEYS : tuple[str, ...]
        The keys of an ``"inspection"`` spec: ``"interval"``, ``"duration"``
        and ``"cost"``.

    Raises
    ------
    ValueError
        If a spec dict has an unknown key, or a ``"repairability"`` string
        other than ``"instant"``; if a cost is not a finite, non-negative
        number, a ``"downtime_cost"`` or ``downtime_cost_rate`` is a
        distribution, or a cost distribution has no finite mean or puts
        appreciable probability on a negative cost (its 1e-12 quantile is
        below 0); if a ``"preventive"`` spec is not a dict of its keys with
        a positive ``interval``, a ``policy`` of ``"age"`` or ``"block"``
        and a ``duration`` that is a model or ``"instant"``; if an
        ``"inspection"`` spec is not a dict of its keys with a positive,
        finite ``interval`` and a ``duration`` that is a model or
        ``"instant"``, or a component has both; if a
        reliability model is not a surpyval parametric or
        non-parametric model or a ``StandbyModel``; if ``input_node`` or
        ``output_node`` is not in the diagram, or is not its source or sink;
        if ``on_infeasible_rbd`` is not ``"raise"``, ``"warn"`` or
        ``"ignore"``; or if the diagram is invalid and ``on_infeasible_rbd``
        is ``"raise"``.
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
    >>> result = pumps.availability(t_simulation=50, N=200, seed=0)
    >>> window = result.n_simulations * result.time_simulated_to
    >>> round(float(result.system_uptime) / window, 4)  # simulated
    0.9927

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
        + ("preventive", "inspection", "acquisition_cost")
    )
    #: The keys of a component's ``"preventive"`` spec.
    PREVENTIVE_KEYS = ("interval", "policy", "duration", "cost")
    #: The keys of a component's ``"inspection"`` spec.
    INSPECTION_KEYS = ("interval", "duration", "cost")
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
        }
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
        # One-off purchase costs, by node (only non-zero ones).
        self.acquisition_costs: dict[Any, float] = {}
        components = copy(components)
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
                    schedule, cost = self._validate_preventive(
                        name, component["preventive"]
                    )
                    if cost is not None:
                        node_costs["preventive_cost"] = cost
                    if np.isfinite(schedule.interval):
                        self._preventive[name] = schedule
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

        super().__init__(
            edges,
            set(components.keys()),
            k,
            input_node,
            output_node,
            on_infeasible_rbd,
        )

        # Every intermediate graph node needs a component definition (the
        # input/output nodes do not). Surface missing ones now with a clear
        # error rather than a KeyError mid-simulation.
        missing = [
            n
            for n in self.G.nodes
            if n not in components and n not in self.in_or_out
        ]
        self.structure_check["is_missing_components"] = bool(missing)
        self.structure_check["nodes_with_no_component"] = missing
        if missing:
            self.structure_check["is_valid"] = False
            if on_infeasible_rbd == "raise":
                raise ValueError(
                    f"Node(s) {sorted(missing, key=str)} have no entry in "
                    "the components dict."
                )
            elif on_infeasible_rbd == "warn":
                warnings.warn(
                    "Nodes with no component definition: "
                    + pprint.pformat(missing),
                    stacklevel=2,
                )

        self.components = components
        self.repairability = copy(repairability)

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

    @classmethod
    def _validate_preventive(cls, node, spec) -> Tuple[_Preventive, Any]:
        """A component's ``"preventive"`` spec, validated: its schedule,
        and its cost (None if it prices nothing)."""
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
        if policy not in ("age", "block"):
            raise ValueError(
                f"Component {node!r}: the preventive policy must be 'age' or "
                f"'block', got {policy!r}."
            )
        duration = spec.get("duration", "instant")
        if isinstance(duration, str) and duration == "instant":
            duration = None
        elif not hasattr(duration, "random"):
            raise ValueError(
                f"Component {node!r}: the preventive duration must be a "
                "time-to-maintain model (such as a fitted surpyval "
                f"distribution) or 'instant', got {duration!r}."
            )
        cost = spec.get("cost")
        if cost is not None:
            cost = cls._validate_component_cost(node, "preventive_cost", cost)
            if isinstance(cost, float) and cost == 0.0:
                cost = None
        return _Preventive(interval, policy, duration), cost

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
        return _Inspection(interval, duration), cost

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
        or inspection ``"cost"`` (a cost distribution always counts), or
        ``downtime_cost_rate`` is non-zero: whether running the system costs
        anything. Costs of 0 price nothing, costs declared inside a nested
        ``RepairableRBD`` do not count, and neither does an
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
        return bool(self.costs) or bool(self.downtime_cost_rate)

    def _streamed_components(
        self, stream: Any, made: Optional[dict] = None, prefix: tuple = ()
    ) -> Optional[dict[Any, Any]]:
        """Stand-ins that replay every component's failure/repair draws from
        ``stream``, nested RBDs' components included, or ``None`` if any
        component's draws cannot be reproduced exactly that way (a
        non-parametric model, ...), in which case the simulation draws from
        the components themselves. The stand-ins are called at exactly the
        points the components would be, so the draws come in the same order.

        A component object used for several nodes has one simulation state,
        so it gets one stand-in (``made``, by object). Since each node has
        its own copy of a ``NonRepairable``, only a nested RBD object can be.
        """
        made = {} if made is None else made
        streamed: dict[Any, Any] = {}
        for name, component in self.components.items():
            if id(component) not in made:
                made[id(component)] = _stand_in(
                    component, stream, made, prefix + (name,)
                )
            if made[id(component)] is None:
                return None
            streamed[name] = made[id(component)]
        # A maintained or inspected component draws its maintenance or test
        # times from the same stream, at the same points.
        durations: dict = {n: p.duration for n, p in self._preventive.items()}
        durations.update({n: i.duration for n, i in self._inspection.items()})
        for name, duration in durations.items():
            if duration is not None:
                sampler = inverse_sampler(duration)
                if sampler is None:
                    return None
                streamed[name].maintenance = sampler
        return streamed

    def _action_charges(
        self,
    ) -> Tuple[
        dict[Any, list[tuple[str, Iterator[float]]]],
        dict[Any, Iterator[float]],
        dict[Any, Iterator[float]],
    ]:
        """For each costed node, its ``("repair" | "replace", charges)``
        pairs: the stream of amounts charged at the node's successive
        failures in a simulation (see :func:`_charges`); and, for each node
        with a preventive-maintenance or an inspection cost, the stream
        charged at its successive preventive actions or inspections.

        A cost distribution draws from its own generator, seeded from -- but
        not consuming -- numpy's global RNG: a seeded run stays reproducible,
        and the failure/repair draws (so every availability output) are the
        same whether or not any cost is random.
        """
        state = np.random.get_state()
        rng = np.random.default_rng(np.random.randint(2**31 - 1))
        np.random.set_state(state)
        failures = {
            node: [
                (key.removesuffix("_cost"), _charges(node_costs[key], rng))
                for key in self.PER_FAILURE_COST_KEYS
                if key in node_costs
            ]
            for node, node_costs in self.costs.items()
        }
        preventive = {
            node: _charges(node_costs["preventive_cost"], rng)
            for node, node_costs in self.costs.items()
            if "preventive_cost" in node_costs
        }
        inspection = {
            node: _charges(node_costs["inspection_cost"], rng)
            for node, node_costs in self.costs.items()
            if "inspection_cost" in node_costs
        }
        return failures, preventive, inspection

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
            ``node_availability``), or has hidden failures other than with
            a constant failure rate,
            instant tests and instant repair: simulate it with ``cost``.

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

        rate = 0.0

        # Production lost while the *system* is down.
        if self.downtime_cost_rate:
            unavailability = 1.0 - self.mean_availability(
                working_nodes, broken_nodes
            )
            rate += self.downtime_cost_rate * unavailability

        if not self.costs:
            return rate

        node_availability = _squeeze_values(
            self._probabilities_with_overrides(
                self.node_availability(), working_nodes, broken_nodes
            )
        )
        for node in self.costs:
            rate += self._node_cost_rate(
                node, node_availability[node], node in forced
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
            failures, maintained, _ = self._node_frequencies(node)
            rate += per_action * failures
            if preventive is not None:
                rate += _mean_cost(preventive) * maintained
        inspection = node_costs.get("inspection_cost")
        if inspection is not None and not forced:
            # One inspection per interval (none is skipped: repairs are
            # instant, as the exact values require).
            _, interval = self._inspected_rate(node)
            rate += _mean_cost(inspection) / interval
        # Optional cost of *this component* being down, whether or not the
        # system as a whole is.
        downtime_cost = node_costs.get("downtime_cost", 0.0)
        if downtime_cost:
            rate += downtime_cost * (1.0 - availability)
        return rate

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
        ``cost`` simulates a finite window from new (whose ``CostResult``
        gives the same ``acquisition_cost`` separately).

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
            failures other than with a constant failure rate, instant tests
            and instant repair: its long-run values are not known exactly.

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

        # Every node's availability (and unavailability) over the times the
        # long-run values average over, and each component's own cost.
        times, weights = self._long_run_grid()
        up = {
            node: np.atleast_1d(np.asarray(a, dtype=float))
            for node, a in self._availabilities_at(times).items()
        }
        down = {node: 1.0 - a for node, a in up.items()}
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
            As for ``mean_availability``.

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
            As for ``mean_availability``.

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
                plan._inspection[node] = self._inspection[node]._replace(
                    interval=float(interval)
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
        rates = {node: self._inspected_rate(node)[0] for node in chosen}

        def evaluate(intervals: dict) -> Tuple[float, float]:
            plan = self._with_intervals(inspection=intervals)
            return plan.expected_cost_rate(), plan.mean_availability()

        if allowed is None:
            if len(self._inspection) > 1:
                raise ValueError(
                    "More than one component has hidden failures: give the "
                    "intervals to choose from in allowed (tests are made on "
                    "a calendar, and the long-run values depend on how the "
                    "schedules line up)."
                )
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
            return list(self._inspection)
        chosen = list(nodes)
        if not chosen:
            raise ValueError("nodes is empty.")
        for node in chosen:
            if node not in self._inspection:
                raise ValueError(
                    f"Node {node!r} has no hidden failures: give it an "
                    "'inspection' schedule to have its interval chosen."
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
    ):
        """Start one simulation of the system over ``[0, t_simulation)``.

        Advanced, event-stepping API: ``availability`` runs its simulations
        through it, and a parent RBD uses it to drive a nested
        ``RepairableRBD``, which therefore has the same event API as a
        ``NonRepairable`` component. Every component starts working except
        the ``broken_nodes``, which stay down for the whole window; the
        ``working_nodes`` never fail. The first failure of each other
        component is queued (a nested RBD starts its own simulation and
        queues its first state change), and events at or after
        ``t_simulation`` are dropped. Then call ``next_event`` repeatedly to
        step through the system's state changes; see it for an example.

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
            drawn from, which ``availability`` passes to replay the draws
            in batches. By default None: the components themselves.

        Raises
        ------
        ValueError
            If ``method`` is not ``"p"`` or ``"c"``.
        """
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        # What the components draw their events from: themselves, or
        # stand-ins replaying the same draws (see _streamed_components).
        sources = self.components if sources is None else sources

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

        # For each component add in the initial failure
        for component_id in self.components.keys():
            component = self.components[component_id]
            if component_id in working_nodes:
                continue
            elif component_id in broken_nodes:
                continue
            source = sources[component_id]
            if isinstance(component, RepairableRBD):
                source.initialize_event_queue(t_simulation)
                t_event, event = source.next_event()
                first = Event(
                    t_event, component_id, event, _planned(source, event)
                )
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
        self.last_change_planned = False
        # The initial system state must reflect any forced-broken components
        # (e.g. a broken component in series starts the system down), rather
        # than assuming everything is up.
        self.system_state = self.is_system_working(component_status, method)
        self.t_simulation = t_simulation
        self.component_status = component_status

    def mean_unavailability(self, *args, **kwargs) -> float:
        """Returns the system's long-run (steady-state) unavailability.

        Exactly ``1 - mean_availability(*args, **kwargs)``: the long-run
        fraction of time the system is down.

        Parameters
        ----------
        *args
            Positional arguments of ``mean_availability``
            (``working_nodes``, ``broken_nodes``, ``method``).
        **kwargs
            Keyword arguments of ``mean_availability``.

        Returns
        -------
        float
            Long-run unavailability of the system, in ``[0, 1]``.

        Raises
        ------
        ValueError
            As for ``mean_availability``.
        """
        return 1 - self.mean_availability(*args, **kwargs)

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
        function. This is exact for a constant failure rate, instant tests
        and instant repair (see ``node_availability``).

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
            failures other than with a constant failure rate, instant tests
            and instant repair:
            simulate it with ``availability``.

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
        # _long_run_grid); otherwise at one point, as the node availabilities
        # are constant.
        times, weights = self._long_run_grid()
        availability = self._probabilities_with_overrides(
            self._availabilities_at(times), working_nodes, broken_nodes
        )
        system = self.system_probability(availability, method=method)
        return float(weights @ system)

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

    def _maintained_follow_up(
        self, event: Event, source, schedule: _Preventive
    ) -> Event:
        """The next event of a component under preventive maintenance."""
        node, t = event.component, event.time
        if event.preventive:
            # Replaced: the new unit has its own life ahead of it (the
            # failure drawn for the old one never happens).
            source.reset()
            if not event.status:
                # Down for maintenance: back, as new, once it is done.
                if isinstance(source, _StreamedComponent):
                    duration = source.maintenance_time()
                else:
                    duration = draw(schedule.duration, 1).item()
                return Event(t + duration, node, True, True)
        elif not event.status:
            # Failed: back, as new, once it is repaired.
            repair, status = source.next_event()
            return Event(t + repair, node, status)
        # Working, as new, from t.
        return self._renewal(node, t, source, schedule)

    @staticmethod
    def _renewal(node, t: float, source, schedule: _Preventive) -> Event:
        """The first event of a unit put into service as new at ``t``: its
        failure, or its preventive replacement if that is due first (a
        failure at the same time comes first)."""
        life, _ = source.next_event()
        due = schedule.due(t)
        if t + life <= due:
            return Event(t + life, node, False)
        # Maintenance that takes time is a planned outage; in zero time
        # the unit is renewed in place, and stays up.
        return Event(due, node, schedule.duration is None, True)

    def _inspected_follow_up(
        self, event: Event, source, inspection: _Inspection
    ) -> Event:
        """The next event of a component whose failures are hidden."""
        node, t = event.component, event.time
        if not event.inspection:
            if event.status:
                # Repaired: as new, from t.
                return self._inspected_renewal(node, t, source, inspection)
            # Failed, unseen: found by the first inspection at or after t.
            self._pending_failure[node] = None
            return Event(inspection.finds(t), node, False, inspection=True)
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
                duration = draw(inspection.duration, 1).item()
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
        due = inspection.due(t)
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
        if not hasattr(self, "_event_queue"):
            raise ValueError("Need to initialize the event queue")
        # The components' draws come from the same sources the queue was
        # initialised with (see initialize_event_queue).
        sources = self.components if sources is None else sources
        new_system_state = copy(self.system_state)

        # Use a while loop to find the next time/event at which the system
        # status changes.
        while new_system_state == self.system_state:
            if self._event_queue.qsize() == 0:
                del self._event_queue
                self.last_change_planned = False
                return self.t_simulation, self.system_state

            event = self._event_queue.get()
            self.component_status[event.component] = event.status
            new_system_state = self.is_system_working(
                self.component_status, method
            )

            next_event = self._follow_up(event, sources[event.component])
            # But only queue up the event if it occurs before the end
            # of the simulation
            if next_event.time < self.t_simulation:
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
        N: int = 10_000,
        verbose: bool = False,
        seed: Optional[int] = None,
        *,
        tolerance: Optional[float] = None,
        confidence: float = 0.95,
        max_N: Optional[int] = None,
        antithetic: bool = False,
        n_jobs: Optional[int] = None,
    ) -> AvailabilityResult:
        """Simulate the system's availability over ``[0, t_simulation]``.

        Runs ``N`` independent Monte-Carlo (discrete-event) simulations of
        the system from time 0, each starting with every component working
        (except ``broken_nodes``). Each component alternates failure and
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
        N : int, optional
            Number of simulations, by default 10_000.
        verbose : bool, optional
            If True, displays a progress bar of the simulations, by default
            False.
        seed : int, optional
            Seed for a reproducible run, by default None. The simulations
            draw from numpy's global RNG (as surpyval's ``random`` does).
            With a seed, that RNG is seeded for the run and the caller's
            RNG state is restored afterwards. With None, the run draws from
            (and advances) the global RNG as it stands, so seeding it with
            ``np.random.seed`` beforehand also makes the run reproducible.
            Models that do not draw from the global RNG, such as surpyval's
            non-parametric ones, are not reproducible either way. Costs
            given as distributions draw from a generator of their own,
            seeded from the global RNG without consuming it, so pricing
            never changes the simulated failures and repairs.
        tolerance : float, optional
            Simulate until the mean availability over the window (the
            fraction of it the system is up) is known to within
            ``tolerance`` either side, at ``confidence``: after the first
            ``N`` simulations, and each further ``N``, the run stops once
            the half-width of the confidence interval of
            ``result.mean_availability_interval()`` is at most
            ``tolerance``, or ``max_N`` simulations have run (then with a
            RuntimeWarning). By default None: exactly ``N``. Without
            ``n_jobs``, a run that stops after ``n`` simulations gives the
            result of a run of ``N=n``.
        confidence : float, optional
            The confidence level ``tolerance`` is judged at, by default
            0.95.
        max_N : int, optional
            The most simulations a run to ``tolerance`` makes, by default
            100 times ``N``.
        antithetic : bool, optional
            Run the simulations in antithetic pairs, by default False: each
            component draws its random numbers from a stream of its own,
            and in the second simulation of a pair it draws ``1 - u`` for
            every uniform ``u`` it drew in the first, in the same order, so
            a failure or repair that was early in one is late in the other.
            Each simulation is still a correct one, but the pair's results
            are negatively correlated, so their mean varies less than two
            independent simulations': a narrower interval for the same
            ``N``. The pairs, not the simulations, are independent, and the
            result's intervals are worked out from the pairs' means. ``N``
            (and ``max_N``) must be even, and every component's draws
            replayable (surpyval parametric distributions and the composite
            models built from them), else ``NotImplementedError``. The
            streams are keyed from the seed (from the global RNG without
            one), so a pair's numbers differ from a run without pairs.
        n_jobs : int, optional
            Run the simulations in parallel over ``n_jobs`` processes (-1:
            one per CPU), in blocks of ``PARALLEL_BLOCK`` (250) simulations,
            each seeded in turn from ``seed``. The blocks are the same
            however many processes run them, so the result does not depend
            on ``n_jobs`` (any number, 1 included), but it differs from a
            run without it. By default None: every simulation in this
            process, in turn. Each process is sent a copy of the RBD, so
            this pays off for long simulations; ``verbose`` has no effect.

        Returns
        -------
        AvailabilityResult
            The availability over time (``timeline``, ``availability``),
            the up and down totals summed over the simulations, the system
            failure, planned outage and restoration counts, the
            ``criticalities``, the ``cost`` and each simulation's up time
            (``uptimes``).

        Raises
        ------
        ValueError
            If a working/broken node is unknown, is the input or output
            node, or is in both sets, if ``method`` is not ``"p"`` or
            ``"c"``, or if ``N``, ``tolerance``, ``confidence``, ``max_N``
            or ``n_jobs`` is invalid (``N`` odd with ``antithetic``,
            ``max_N`` without a tolerance or below ``N``, ...).
        NotImplementedError
            With ``antithetic``, if a component's draws cannot be replayed.

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
        >>> result = rbd.availability(t_simulation=50, N=200, seed=0)
        >>> float(result.timeline[0]), float(result.availability[0])
        (0.0, 1.0)
        >>> float(result.timeline[-1])
        50.0

        The fraction of the window the system was up is close to the
        long-run availability, ``(10 / 11) ** 2`` (over a short window it is
        a little above it on average, as every simulation starts up):

        >>> window = result.n_simulations * result.time_simulated_to
        >>> round(float(result.system_uptime) / window, 4)
        0.8272
        >>> round(rbd.mean_availability(), 4)
        0.8264

        In series the system is up only while every node is up:

        >>> oci = result.criticalities.operational_criticality_index
        >>> {node: round(float(v), 4) for node, v in oci.up.items()}
        {'a': 1.0, 'b': 1.0}
        """
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
        )

    def compare(
        self,
        other: "RepairableRBD",
        t_simulation: float,
        N: int = 10_000,
        seed: Optional[int] = None,
        *,
        quantity: str = "availability",
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """How much better (or worse) this system is than ``other``, by
        simulation with common random numbers.

        Both systems are simulated ``N`` times over ``[0, t_simulation]``
        (every component working at the start), and in each simulation a
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
        N : int, optional
            The number of simulations of each system, by default 10_000.
        seed : int, optional
            Seed for a reproducible comparison, by default None.
        quantity : str, optional
            ``"availability"`` (the default): the fraction of the window the
            system is up. ``"cost"``: its total cost over the window (both
            systems must be priced; costs given as distributions draw from
            the generator ``availability`` uses, not from the components'
            streams).
        confidence : float, optional
            The confidence level of the interval, by default 0.95.

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
            If ``quantity``, ``N`` or ``confidence`` is invalid, or a system
            to compare by cost has no costs.
        NotImplementedError
            If a component's draws cannot be replayed from a stream of its
            own (a non-parametric model, for example).

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
        >>> gain = pumps(1.0).compare(pumps(2.0), 100.0, N=2000, seed=0)
        >>> round(gain.estimate, 4), round(gain.standard_error, 5)
        (0.0187, 0.00041)

        The exact difference is 0.0189 (each pump is up
        ``mu / (lambda + mu) + lambda / (lambda + mu) * exp(-(lambda + mu) t)``
        of the time at ``t``, averaged over the window). Two independent
        runs of 2000 simulations would estimate it with a standard error of
        about 0.00058.
        """
        if quantity not in ("availability", "cost"):
            raise ValueError(
                "quantity must be 'availability' or 'cost', got "
                f"{quantity!r}."
            )
        montecarlo.check_confidence(confidence)
        montecarlo.check_count(N, False, "N")
        if quantity == "cost":
            for rbd in (self, other):
                if not rbd.has_costs:
                    raise ValueError(
                        "Both systems must be priced to compare their costs."
                    )
        key = int(np.random.SeedSequence(seed).generate_state(1, np.uint64)[0])
        values = []
        for rbd in (self, other):
            tally = rbd._run(
                t_simulation,
                set(),
                set(),
                "p",
                N,
                False,
                key % 2**32,
                streams=_KeyedStreams(key),
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
    ) -> AvailabilityResult:
        """``availability`` (and ``cost``): validate, run the replications
        (serially or in parallel, until converged if asked) and build the
        result."""
        working_nodes = set() if working_nodes is None else set(working_nodes)
        broken_nodes = set() if broken_nodes is None else set(broken_nodes)
        self._validate_node_overrides(working_nodes, broken_nodes)
        # The initial system state is the same for every simulation (the
        # forced working/broken sets are fixed): all components start working
        # except those forced broken, which can make the system start down.
        initial_status = {c: c not in broken_nodes for c in self.components}
        initial_up = bool(self.is_system_working(initial_status, method))
        montecarlo.check_count(N, antithetic, "N")
        stop = _stopping_rule(
            N, tolerance, confidence, max_N, antithetic, target, t_simulation
        )
        if n_jobs is None:
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
            )
        else:
            tally = self._run_parallel(
                t_simulation,
                working_nodes,
                broken_nodes,
                method,
                N,
                seed,
                antithetic,
                stop,
                montecarlo.jobs(n_jobs),
            )
        return self._availability_result(
            tally, t_simulation, initial_up, antithetic
        )

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
        streams: Optional["_KeyedStreams"] = None,
    ) -> "_Tally":
        """Run ``N`` replications (then more, while ``stop`` asks for
        them) and return their totals.

        surpyval's ``.random`` draws from numpy's global RNG, so it is
        seeded here (if a seed is given) to make the run reproducible, and
        the caller's RNG state is restored once the simulations have
        finished. With ``streams`` (common random numbers, see ``compare``)
        or ``antithetic``, each component draws from a stream of its own
        instead, keyed (for antithetic pairs) from the global RNG.
        """
        tally = _Tally(self.costs)
        has_costs = self.has_costs
        downtime_cost_rates = {
            node: c["downtime_cost"]
            for node, c in self.costs.items()
            if "downtime_cost" in c
        }
        _rng_state = None
        if seed is not None:
            _rng_state = np.random.get_state()
            np.random.seed(seed)
        charges = self._action_charges() if has_costs else ({}, {}, {})
        if streams is None and antithetic:
            # Each component's pairs come from a stream of its own, keyed
            # from the global RNG (so seeded with it).
            key = int(np.random.randint(0, 2**62, dtype=np.int64))
            streams = _KeyedStreams(key, antithetic=True)
        stream: Any = UniformStream() if streams is None else streams
        streamed = self._streamed_components(stream)
        if streamed is None and (antithetic or streams is not None):
            raise NotImplementedError(
                "Antithetic and common random numbers need every component's "
                "draws to be replayable (surpyval parametric distributions "
                "and the composite models built from them)."
            )
        sources = streamed or self.components

        progress = tqdm(
            total=N, disable=not verbose, desc="Running simulations"
        )
        goal = N
        while True:
            while tally.n < goal:
                if streams is not None:
                    streams.begin(tally.n)
                self._replicate(
                    tally,
                    t_simulation,
                    working_nodes,
                    broken_nodes,
                    method,
                    sources,
                    charges,
                    downtime_cost_rates,
                    has_costs,
                )
                progress.update()
            more = 0 if stop is None else stop(tally)
            if not more:
                break
            goal += more
            progress.total = goal
            progress.refresh()
        progress.close()

        # Randomised simulations are done. Leave the global RNG where the
        # draw-at-a-time simulation would have, then restore the caller's
        # state if a seed was given. (If a simulation raises, neither
        # happens: the RNG is left up to a block of uniforms further on.)
        stream.close()
        if _rng_state is not None:
            np.random.set_state(_rng_state)
        # Clean up the interim variables of the simulation.
        for name in (
            "_event_queue",
            "system_state",
            "t_simulation",
            "component_status",
            "last_change_planned",
            "_pending_failure",
        ):
            self.__dict__.pop(name, None)
        return tally

    def _replicate(
        self,
        tally: "_Tally",
        t_simulation: float,
        working_nodes,
        broken_nodes,
        method: str,
        sources: dict,
        charges: tuple,
        downtime_cost_rates: dict,
        has_costs: bool,
    ) -> None:
        """One simulation of ``[0, t_simulation]``, added to ``tally``."""
        failure_charges, preventive_charges, inspection_charges = charges
        pay = tally.pay
        RCI, FCI = tally.RCI, tally.FCI
        # Initialize the event queue and the system/component statuses
        self.initialize_event_queue(
            t_simulation,
            working_nodes,
            broken_nodes,
            method,
            sources,
        )

        # Seed each timeline from the actual initial status so that
        # forced-broken components (and a system they start down) are
        # accounted as down from t=0, not assumed up.
        component_timelines: dict = {
            comp: [(0.0, 1 if self.component_status[comp] else 0)]
            for comp in self.components
        }
        system_timeline = [(0.0, 1 if self.system_state else 0)]
        rep_cost = 0.0

        # Implemented ensure that no events that occur after the end-time of
        # the simulation are added to the queue; so we just need to keep
        # going through the queue until it's empty
        while not self._event_queue.empty():
            # Get the next event and update the component's status

            event = self._event_queue.get()
            node = event.component
            if (event.preventive or event.inspection) and (
                event.status == self.component_status[node]
            ):
                # No change of state: maintenance or a test in zero time of a
                # working unit (renewed in place, or found working), or a
                # test that finds a hidden failure, whose repair starts now.
                if event.preventive:
                    rep_cost += pay(
                        node, "preventive", preventive_charges.get(node)
                    )
                else:
                    rep_cost += pay(
                        node, "inspection", inspection_charges.get(node)
                    )
                    if not event.status:
                        for category, stream in failure_charges.get(node, ()):
                            rep_cost += pay(node, category, stream)
                next_event = self._follow_up(event, sources[node])
                if next_event.time < t_simulation:
                    self._event_queue.put(next_event)
                continue
            # Update the component's status
            self.component_status[node] = event.status
            if event.status:
                RCI[node]["component_restorations"] += 1
            elif event.preventive:
                # A planned outage: charged the preventive cost (a nested
                # RBD's own costs are not counted).
                rep_cost += pay(
                    node, "preventive", preventive_charges.get(node)
                )
            elif event.inspection:
                # A test that takes a working unit off-line.
                rep_cost += pay(
                    node, "inspection", inspection_charges.get(node)
                )
            else:
                FCI[node]["component_failures"] += 1
                # Repair and replace are charged per corrective action, at
                # the failure that triggers it (for a hidden failure, when an
                # inspection finds it).
                if node not in self._inspection:
                    for category, stream in failure_charges.get(node, ()):
                        rep_cost += pay(node, category, stream)

            status = 1 if event.status else -1
            component_timelines[node].append((event.time, status))

            # Record new system state, it could still be the same as
            # system_state in which case we don't bother changing the
            # aggregate timeline, but if it is different, we need to +/-1
            # to it if the system has gone on/off-line
            new_system_state = self.is_system_working(
                self.component_status, method
            )
            if new_system_state != self.system_state:
                status = 1 if new_system_state else -1
                system_timeline.append((event.time, status))
                if new_system_state:
                    # System restored
                    tally.changes[event.time] += 1
                    tally.system_restorations += 1
                    RCI[node]["system_restorations"] += 1
                else:
                    tally.changes[event.time] -= 1
                    if event.preventive or event.inspection:
                        tally.system_planned_outages += 1
                    else:
                        tally.system_failures += 1
                        FCI[node]["system_failures"] += 1

                # Set the system_state to the new state
                self.system_state = new_system_state

            # Now we need to get the component's next event: its next failure
            # if it just got repaired, or its repair if it just broke; but
            # only queue it up if it occurs before the end of the simulation.
            next_event = self._follow_up(event, sources[node])
            if next_event.time < t_simulation:
                self._event_queue.put(next_event)

        system_timeline.append((t_simulation, 0))

        for component in self.components.keys():
            component_timelines[component].append((t_simulation, 0))
            # This simulation's uptime for the component; the downtime is the
            # remainder of the window.
            component_ut = time_at_status(component_timelines[component], 1)
            tally.node_uptime[component] += component_ut
            tally.node_downtime[component] += t_simulation - component_ut
            if component in downtime_cost_rates:
                charge = downtime_cost_rates[component] * (
                    t_simulation - component_ut
                )
                rep_cost += charge
                tally.cost_by_category["component_downtime"] += charge
                tally.cost_by_component[component] += charge
            joint_t, joint_events = combined_timeline(
                component_timelines[component], system_timeline
            )
            tally.intersection_uptime[component] += intersection(
                joint_t, joint_events
            )
            tally.intersection_downtime[component] += intersection(
                joint_t, 2 - joint_events
            )
            tally.union_uptime[component] += union(joint_t, joint_events)
            tally.union_downtime[component] += union(joint_t, 2 - joint_events)

        simulation_system_ut = time_at_status(system_timeline, 1)
        tally.system_uptime += simulation_system_ut
        tally.system_downtime += t_simulation - simulation_system_ut
        tally.uptimes.append(simulation_system_ut)

        if has_costs:
            charge = self.downtime_cost_rate * (
                t_simulation - simulation_system_ut
            )
            rep_cost += charge
            tally.cost_by_category["system_downtime"] += charge
            tally.cost_samples.append(rep_cost)
        tally.n += 1

    def _availability_result(
        self,
        tally: "_Tally",
        t_simulation: float,
        initial_up: bool,
        antithetic: bool,
    ) -> AvailabilityResult:
        """The ``AvailabilityResult`` of the replications in ``tally``."""
        N = tally.n
        # Collect Importance/Criticality measures from the simulation
        # reference: https://www.weibull.com/pubs/2004rm_05B_02.pdf
        # Operational Criticality Index
        oci_down = {
            k: _safe_ratio(v, tally.system_downtime)
            for k, v in dict(tally.intersection_downtime).items()
        }
        oci_up = {
            k: _safe_ratio(v, tally.system_uptime)
            for k, v in dict(tally.intersection_uptime).items()
        }
        # Intersection Over Union Importance
        iou_up = {
            k: _safe_ratio(tally.intersection_uptime[k], tally.union_uptime[k])
            for k in dict(tally.intersection_uptime).keys()
        }
        iou_down = {
            k: _safe_ratio(
                tally.intersection_downtime[k], tally.union_downtime[k]
            )
            for k in dict(tally.intersection_downtime).keys()
        }
        # Failure Criticality Index Importance
        fci_sys = failure_criticality_index_per_system_failures(
            tally.FCI, tally.system_failures
        )
        fci_comp = failure_criticality_index_per_component_failures(tally.FCI)
        # Restoration Criticality Index Importance
        rci_sys = restoration_criticality_index_by_system(
            tally.RCI, tally.system_restorations
        )
        rci_comp = restoration_criticality_index_by_component(tally.RCI)
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
        aggregate_timeline: dict = defaultdict(int)
        aggregate_timeline[t_simulation] = 0
        aggregate_timeline[0] = N if initial_up else 0
        for time_of_change, change in tally.changes.items():
            aggregate_timeline[time_of_change] += change
        timeline_arr: np.ndarray = np.array(list(aggregate_timeline.items()))
        timeline_arr = timeline_arr[timeline_arr[:, 0].argsort()]
        time = timeline_arr[:, 0]
        system_availability = timeline_arr[:, 1].cumsum() / N

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
            )

        return AvailabilityResult(
            timeline=time,
            availability=system_availability,
            system_uptime=tally.system_uptime,
            time_simulated_to=t_simulation,
            criticalities=criticalities,
            node_uptime=dict(tally.node_uptime),
            node_downtime=dict(tally.node_downtime),
            system_downtime=tally.system_downtime,
            system_failures=tally.system_failures,
            system_restorations=tally.system_restorations,
            n_simulations=N,
            cost=cost_result,
            system_planned_outages=tally.system_planned_outages,
            uptimes=np.asarray(tally.uptimes, dtype=float),
            antithetic=antithetic,
        )

    def _run_parallel(
        self,
        t_simulation: float,
        working_nodes,
        broken_nodes,
        method: str,
        N: int,
        seed,
        antithetic: bool,
        stop,
        jobs: int,
    ) -> "_Tally":
        """The replications in blocks of ``PARALLEL_BLOCK``, each seeded
        from ``seed`` by its position, run ``jobs`` at a time in separate
        processes and totalled in order. With a stopping rule, it is
        checked after each batch of ``N`` replications, as in a serial run,
        so the result does not depend on ``jobs``."""
        seeds = np.random.SeedSequence(seed)
        tally = _Tally(self.costs)
        executor = ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
        try:
            batch = N
            while batch:
                tasks = [
                    (
                        self,
                        t_simulation,
                        working_nodes,
                        broken_nodes,
                        method,
                        size,
                        montecarlo.block_seed(seeds),
                        antithetic,
                    )
                    for size in montecarlo.blocks(batch, PARALLEL_BLOCK)
                ]
                if executor is None:
                    results = [_simulate_block(task) for task in tasks]
                else:
                    results = list(executor.map(_simulate_block, tasks))
                for block in results:
                    tally.merge(block)
                batch = 0 if stop is None else stop(tally)
        finally:
            if executor is not None:
                executor.shutdown()
        return tally

    def cost(
        self,
        t_simulation: float,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
        method: str = "p",
        N: int = 10_000,
        verbose: bool = False,
        seed: Optional[int] = None,
        *,
        tolerance: Optional[float] = None,
        confidence: float = 0.95,
        max_N: Optional[int] = None,
        antithetic: bool = False,
        n_jobs: Optional[int] = None,
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
        N : int, optional
            Number of simulations, each giving one sample of the window's
            total cost, by default 10_000.
        verbose : bool, optional
            If True, displays a progress bar of the simulations, by default
            False.
        seed : int, optional
            Seed for a reproducible run, by default None. As in
            ``availability``: the global RNG is seeded for the run and the
            caller's state restored afterwards.
        tolerance : float, optional
            Simulate until the mean cost of a window is known to within
            ``tolerance`` (in the costs' currency) either side, at
            ``confidence`` (the half-width of ``mean_interval()``): as in
            ``availability``, checked after the first ``N`` simulations
            and each further ``N``, up to ``max_N``. By default None:
            exactly ``N``.
        confidence : float, optional
            The confidence level ``tolerance`` is judged at, by default
            0.95.
        max_N : int, optional
            The most simulations a run to ``tolerance`` makes, by default
            100 times ``N``.
        antithetic : bool, optional
            Run the simulations in antithetic pairs (see ``availability``),
            by default False. The draws of costs given as distributions are
            not paired.
        n_jobs : int, optional
            Run the simulations in parallel over ``n_jobs`` processes (see
            ``availability``), by default None.

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
        >>> result = rbd.cost(t_simulation=100.0, N=200, seed=0)
        >>> round(result.mean, 2)  # mean cost of a 100-hour window
        1351.33
        >>> round(result.percentile(90), 2)  # 9 windows in 10 cost less
        1915.24
        >>> round(result.cost_rate, 2), round(rbd.expected_cost_rate(), 2)
        (13.51, 13.64)
        """
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
            the interval), or has hidden failures other than with a
            constant failure rate, instant tests and instant repair.

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
        """A component's long-run availability (see ``node_availability``)."""
        if node in self._inspection:
            rate, interval = self._inspected_rate(node)
            return float(-np.expm1(-rate * interval) / (rate * interval))
        schedule = self._preventive.get(node)
        if schedule is None:
            component = self.components[node]
            return float(np.atleast_1d(component.mean_availability())[0])
        up, cycle, _, _ = self._maintenance_cycle(node, schedule)
        return min(1.0, up / cycle)

    def _inspected_rate(self, node) -> Tuple[float, float]:
        """The constant failure rate and the inspection interval of a
        component with hidden failures, for the exact long-run values,
        which cover only a constant failure rate, instant tests and instant
        repair."""
        component = self.components[node]
        rate = _constant_rate(component.reliability)
        inspection = self._inspection[node]
        if (
            rate is None
            or inspection.duration is not None
            or model_mean(component.time_to_replace) != 0.0
        ):
            raise NotImplementedError(
                f"Component {node!r} has hidden failures: its exact long-run "
                "values are known only with a constant failure rate (an "
                "exponential life), instant tests and instant repair. "
                "Estimate them by simulation, with availability() or cost()."
            )
        return rate, inspection.interval

    def _block_nodes(self) -> list:
        """The components under block replacement."""
        return [
            node
            for node, schedule in self._preventive.items()
            if schedule.policy == "block"
        ]

    def _block_cycle(self, node) -> BlockCycle:
        """The renewal cycle of a component under block replacement (see
        ``_block_replacement``), computed once and kept."""
        component = self.components[node]
        schedule = self._preventive[node]
        cache = self.__dict__.setdefault("_block_cycles", {})
        key = (
            node,
            id(component.reliability),
            id(component.time_to_replace),
            float(schedule.interval),
            id(schedule.duration),
        )
        if key not in cache:
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
        period = _common_period(intervals)
        pieces = [np.array([0.0, period])]
        for interval in intervals:
            count = int(round(period / interval))
            if count > 100_000:
                raise NotImplementedError(
                    f"The block-replacement and inspection intervals "
                    f"{sorted(intervals)} repeat together only after too many "
                    "intervals to average over: estimate the long-run values "
                    "by simulation, with availability() or cost()."
                )
            pieces.append(interval * np.arange(count))
        for node in blocks:
            phase = self._block_cycle(node).phase
            repeats = int(round(period / phase[-1]))
            pieces.append(
                (
                    phase[-1] * np.arange(repeats)[:, None] + phase[None, :-1]
                ).ravel()
            )
        for node in self._inspection:
            rate, interval = self._inspected_rate(node)
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
            self._inspection[node].interval for node in self._inspection
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
        base = self._availabilities_at(instants + 1e-9 * period)
        before, after = dict(base), dict(base)
        for column, key in enumerate(sorted(due)):
            for node in due[key]:
                cycle = self._block_cycle(node)
                before[node] = np.array(before[node], dtype=float)
                after[node] = np.array(after[node], dtype=float)
                before[node][column] = cycle.before
                after[node][column] = cycle.after
        before = self._probabilities_with_overrides(
            before, working_nodes, broken_nodes
        )
        after = self._probabilities_with_overrides(
            after, working_nodes, broken_nodes
        )
        fall = np.asarray(self.system_probability(before)) - np.asarray(
            self.system_probability(after)
        )
        return float(np.sum(fall)) / period

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
        if blocks:
            return self._calendar_grid(blocks)
        if not self._inspection:
            return np.zeros(1), np.ones(1)
        rates = {node: self._inspected_rate(node) for node in self._inspection}
        intervals = {interval for _, interval in rates.values()}
        period = _common_period(intervals)
        breaks = {0.0, period}
        for interval in intervals:
            count = int(round(period / interval))
            if count > 100_000:
                raise NotImplementedError(
                    f"The inspection intervals {sorted(intervals)} repeat "
                    "together only after too many inspections to average "
                    "over: estimate the long-run values by simulation, with "
                    "availability() or cost()."
                )
            breaks.update(k * interval for k in range(1, count))
        edges = np.array(sorted(breaks))
        fastest = max(rate for rate, _ in rates.values())
        points, weights = np.polynomial.legendre.leggauss(16)
        times, masses = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            pieces = np.linspace(
                a, b, max(1, math.ceil((b - a) * fastest)) + 1
            )
            for lo, hi in zip(pieces[:-1], pieces[1:]):
                half = 0.5 * (hi - lo)
                times.append(lo + half * (points + 1.0))
                masses.append(half * weights)
        return np.concatenate(times), np.concatenate(masses) / period

    def _availabilities_at(self, times: np.ndarray) -> dict:
        """Every node's availability at each of ``times`` (see
        ``_long_run_grid``): ``exp(-lambda * u)``, ``u`` the time since the
        last inspection, for a component with hidden failures; its constant
        long-run availability for any other."""
        out: dict = {}
        blocks = set(self._block_nodes())
        for node in self.components:
            if node in self._inspection:
                rate, interval = self._inspected_rate(node)
                since = times - interval * np.floor(times / interval)
                out[node] = np.exp(-rate * since)
            elif node in blocks:
                out[node] = self._block_profile(node, times)
            else:
                out[node] = np.full(len(times), self._node_availability(node))
        for node in self.in_or_out:
            out[node] = np.ones(len(times))
        return out

    def _long_run_probabilities(
        self, working_nodes, broken_nodes
    ) -> Tuple[dict, np.ndarray]:
        """The node availabilities the long-run importance measures are
        evaluated at (with the forced nodes held at 1 or 0), over the times
        of ``_long_run_grid``, and those times' weights: each measure is
        then a ratio of time-averaged system quantities."""
        times, weights = self._long_run_grid()
        probabilities = self._probabilities_with_overrides(
            self._availabilities_at(times), working_nodes, broken_nodes
        )
        return probabilities, weights

    def _node_frequencies(self, node) -> Tuple[float, float, float]:
        """A component's long-run failures, preventive replacements and
        planned outages, per unit time: recursive for a nested
        RepairableRBD (whose own preventive replacements are not this RBD's
        to count), ``1 / (MTTF + MTTR)`` failures for a NonRepairable, and
        from its renewal cycle for one under age replacement."""
        component = self.components[node]
        if isinstance(component, RepairableRBD):
            failures, planned = component._outage_frequencies()
            return failures, 0.0, planned
        if node in self._inspection:
            # At most one failure per inspection interval: the unit, down
            # from its failure, is renewed at the inspection that finds it.
            rate, interval = self._inspected_rate(node)
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
        ``_block_replacement``); it is computed once and kept."""
        component = self.components[node]
        if schedule.policy == "block":
            block = self._block_cycle(node)
            return block.up, block.length, block.failures, 1.0
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
        maintenance = (
            0.0 if schedule.duration is None else model_mean(schedule.duration)
        )
        cycle = (
            up
            + (1.0 - survives) * model_mean(component.time_to_replace)
            + survives * maintenance
        )
        cache[key] = (up, cycle, 1.0 - survives, survives)
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
        nodes, with no simulation. Every system failure counts, including
        the zero-length outages an instantly repaired component causes;
        planned outages (preventive maintenance) are not failures.

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
            values do not cover (see ``node_availability``).

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
        ``I_B(i)``)."""
        times, weights = self._long_run_grid()
        availability = self._probabilities_with_overrides(
            self._availabilities_at(times), working_nodes, broken_nodes
        )
        forced = (set() if working_nodes is None else set(working_nodes)) | (
            set() if broken_nodes is None else set(broken_nodes)
        )
        birnbaum = super()._birnbaum_importance(availability)
        failures = planned = 0.0
        blocks = set(self._block_nodes())
        for node in self.components:
            if node in forced:
                # A forced node never changes state, so it contributes no
                # system failures.
                continue
            importance = np.asarray(birnbaum[node])
            if node in self._inspection:
                # It fails at its constant rate whenever it is up.
                rate, _ = self._inspected_rate(node)
                node_failures: Any = rate * availability[node]
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

        Exact: ``MTBF = 1 / system_failure_frequency`` — the mean length of
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
        ``MUT = mean_availability / system_failure_frequency``. A planned
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
        MTTR), ``MDT = mean_unavailability / system_failure_frequency``.
        Planned outages (preventive maintenance that takes time) count as
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
        availability = self.mean_availability(working_nodes, broken_nodes)
        omega = sum(self._outage_frequencies(working_nodes, broken_nodes))
        if omega > 0.0:
            return (1.0 - float(availability)) / omega
        return 0.0 if availability >= 1.0 else float("inf")

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
        ``broken_nodes`` held at 1 and 0.

        Note: Birnbaum's measure of importance assumes all nodes are
        independent.

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
        node_probabilities, weights = self._long_run_probabilities(
            working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._birnbaum_importance(node_probabilities, weights)
        )

    def improvement_potential(
        self,
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Returns the improvement potential of all nodes, evaluated at the
        nodes' long-run availabilities.

        Exact, with no simulation: ``A_sys(A_i = 1) - A_sys``, the gain in
        the system's long-run availability if node i never failed. The node
        availabilities are those of ``node_availability``, with
        ``working_nodes`` and ``broken_nodes`` held at 1 and 0.

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
        node_probabilities, weights = self._long_run_probabilities(
            working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._improvement_potential(node_probabilities, weights)
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
        node_probabilities, weights = self._long_run_probabilities(
            working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._risk_achievement_worth(node_probabilities, weights)
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
        node_probabilities, weights = self._long_run_probabilities(
            working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._risk_reduction_worth(node_probabilities, weights)
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
        node_probabilities, weights = self._long_run_probabilities(
            working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._criticality_importance(
                node_probabilities, kind, weights=weights
            )
        )

    def fussell_vesely(
        self,
        fv_type: str = "c",
        working_nodes: Optional[Collection[Hashable]] = None,
        broken_nodes: Optional[Collection[Hashable]] = None,
    ) -> dict[Any, float]:
        """Calculate Fussell-Vesely importance of all nodes, evaluated at the
        nodes' long-run availabilities.

        Briefly, the Fussell-Vesely importance measure for node i =
        (sum of probabilities of cut-sets including node i occurring, i.e.
        all their nodes failed) / (the probability of the system failing).
        Here a node's probability of having failed is its long-run
        unavailability, ``1 - A_i``, from ``node_availability`` (with
        ``working_nodes`` and ``broken_nodes`` held at availability 1 and
        0), and the system's is ``1 - A_sys``; both are exact, with no
        simulation. The sum over cut sets is the usual rare-event
        approximation of the probability that some cut set containing node
        i has occurred, so with large unavailabilities the measure can
        exceed 1. If the system never fails, the ratio is ``nan`` or
        ``inf``, with a numpy warning.

        Typically this measure is implemented using cut-sets as mentioned
        above, although it can be implemented using path-sets. Both are
        implemented here, selected by ``fv_type``: ``"c"`` sums over the
        minimal cut sets containing node i, ``"p"`` over the minimal path
        sets containing it. Either way each set contributes the probability
        that all of its nodes are failed, and the sum is divided by the
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

        Returns
        -------
        dict[Any, float]
            Dictionary with node names as keys and Fussell-Vesely importances
            as values, for every node except the input and output nodes.

        Raises
        ------
        ValueError
            If ``fv_type`` is not 'c' (cut-set) or 'p' (path-set); if a
            working/broken node is unknown, is the input or output node, or
            is in both sets; or if a component has a non-parametric
            reliability model.

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
        node_probabilities, weights = self._long_run_probabilities(
            working_nodes, broken_nodes
        )
        return _squeeze_values(
            super()._fussell_vesely(
                node_probabilities, fv_type, weights=weights
            )
        )

    def fussel_vesely(self, fv_type: str = "c") -> dict[Any, float]:
        """Deprecated alias for ``fussell_vesely`` (corrected spelling).

        Deprecated: use ``fussell_vesely`` instead; this alias will be
        removed in a future release. It returns ``fussell_vesely(fv_type)``
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
        DeprecationWarning
            On every call.
        """
        warnings.warn(
            "fussel_vesely() is deprecated; use fussell_vesely() "
            "(Fussell-Vesely). This alias will be removed in a future "
            "release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.fussell_vesely(fv_type)
