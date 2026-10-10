"""The records and state of a ``RepairableRBD`` simulation's event loop.

The events on its queue (``Event``, ``_EventQueue``), a component's
maintenance and inspection schedules (``_Preventive``, ``_Inspection``,
``_MaintenanceGroup``), the repair crews' queue (``_Crews``), a standby
group's units (``_StandbyGroup``) and a common-cause group's causes
(``_Cause``). The loop itself is ``RepairableRBD._replicate``; the crews'
and standby groups' Markov chains (``_crew_chain``, ``_standby_chain``)
copy these rules, and the compiled engine (``_kernel``) the loop's.
"""

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Hashable,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np

if TYPE_CHECKING:
    from repyability.rbd.repairable_rbd import RepairableRBD


# The kinds of a standby group's own events.
_UNIT_FAILS, _SPARE_FAILS, _UNIT_REPAIRED = 0, 1, 2


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


@dataclass(frozen=True)
class _Unit:
    """A unit of a standby group: its repair is a job for the repair crews
    under this key (a dataclass, so equal to no node name)."""

    node: Any
    unit: int


@dataclass(frozen=True)
class _Cause:
    """A shared cause of a common-cause group, as the simulation strikes
    it (#158; a dataclass, so equal to no node name): the members it fails
    (``struck``, those up when it strikes), its rate, and whether a test
    that can miss its failures decides for all of them at once
    (``coins``)."""

    struck: tuple
    rate: float
    coins: bool


class _CauseDraws:
    """What a shared common cause draws in a simulation (#158): the times
    between its strikes, exponential at its rate, and at each strike the
    uniform that decides whether the tests that can miss its failures find
    them (found, or missed, alike); from its own streams, or numpy's global
    RNG."""

    __slots__ = ("_rate", "_gaps", "_coins")

    def __init__(self, rate: float, gaps=None, coins=None):
        self._rate = rate
        self._gaps = gaps
        self._coins = coins

    def gap(self) -> float:
        if self._gaps is not None:
            return self._gaps.draw()
        return float(np.random.exponential(1.0 / self._rate))

    def coin(self) -> float:
        if self._coins is not None:
            return self._coins.draw()
        return float(np.random.random())


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
    before the next (or, with a ``level``, if its measured degradation
    level is at or past it); a replacement takes a time drawn from
    ``duration`` (None: no time)."""

    interval: float
    policy: str
    duration: Any
    threshold: float = 0.0
    #: Under ``"age"``, the age from which the unit is renewed early at a
    #: stop of its maintenance group (#108); ``inf`` for never.
    opportunity: float = math.inf
    #: Under ``"condition"`` with a degradation process for a life (#271):
    #: the measured level at or past which an inspection replaces the
    #: unit, in place of the age-based ``threshold``; None for none.
    level: Optional[float] = None

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
