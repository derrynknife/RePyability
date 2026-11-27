"""The records and state of a ``RepairableRBD`` simulation's event loop.

The events on its queue (``Event``, ``_EventQueue``), a component's
maintenance and inspection schedules (``_Preventive``, ``_Inspection``,
``_MaintenanceGroup``), the repair crews' queue (``_Crews``), a standby
group's units (``_StandbyGroup``) and a common-cause group's causes
(``_Cause``). The loop itself is ``RepairableRBD._replicate``; the crews'
and standby groups' Markov chains (``_crew_chain``, ``_standby_chain``)
copy these rules, and the compiled engine (``_kernel``) the loop's.
"""

import bisect
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Tuple,
)

import numpy as np

if TYPE_CHECKING:
    from repyability.rbd.repairable_rbd import RepairableRBD


# The kinds of a standby group's own events.
_UNIT_FAILS, _SPARE_FAILS, _UNIT_REPAIRED = 0, 1, 2
#: A failed unit's spare is in, with repair crews: its job now queues for
#: a crew (see _shelves).
_SPARE_IN = 3


#: The classes of event at one instant, in the order they are taken (see
#: ``_EventQueue``): a failure, a planned stop (maintenance or a test
#: taking a unit down, a shutdown starting) and a restoration (any event
#: that leaves its node up).
FAILURE, STOP, RESTORATION = 0, 1, 2

#: How far a tie's class is shifted above a place in its rank, and where
#: the nodes' places start, after the causes' and shutdowns' (a diagram has
#: fewer of either than this).
_CLASS_SHIFT = 32
_NODES_FROM = 1 << 31

#: What a component's event counts as, as bits, where the loops record a
#: run's histories (see ``_timeline_runs.Block.events``): a failure, a
#: corrective action (a failure's, or a hidden failure's when a test finds
#: it), a preventive task (a renewal in place or an outage) and a test.
#: Each loop records every event of these at the system's level as it
#: takes it, whether or not it changes the component's state (a task or
#: test in zero time changes none), so a history counts each one.
#: ``TALLY_PLANNED`` and ``TALLY_KEPT`` are the compiled loop's own, which
#: keeps its changes and these in one record: a planned change down, and
#: an event that changes nothing.
TALLY_PLANNED, TALLY_KEPT = 1, 2
TALLY_FAILURE, TALLY_CORRECTIVE, TALLY_PREVENTIVE, TALLY_TEST = 4, 8, 16, 32
#: The bits that count an event.
TALLY_COUNTED = (
    TALLY_FAILURE | TALLY_CORRECTIVE | TALLY_PREVENTIVE | TALLY_TEST
)


def tie_class(event: "Event") -> int:
    """The class of ``event`` among events at one instant (``FAILURE``,
    ``STOP`` or ``RESTORATION``; see ``_EventQueue``)."""
    if event.status:
        return RESTORATION
    if event.preventive or event.inspection:
        return STOP
    return FAILURE


class _EventQueue:
    """The simulation's event queue: a heap (``heapq``) of ``(time, rank,
    event)`` entries.

    **Events at one instant are taken in a stated order**: by their
    class, failures first, then planned stops (maintenance or a test that
    takes a unit down, a shutdown's start), then restorations (every event
    that leaves its node up: a repair's end, maintenance in zero time, a
    test that finds a unit working, a shutdown's end); and within a class
    the diagram's shared common causes and shutdowns first, in the order
    they were first queued, then the nodes, by their place in the diagram
    (their order in ``components``). A standby or load-sharing group's
    event is its next unit's: a unit failing, a failure, and a repair done,
    a restoration. So the members a common cause strikes at one instant
    fail in the diagram's order, a failure at the instant of another node's
    repair is taken before it, as the conditional runs' paths take them
    (``_conditional.paths``), and a shutdown that ends as the work done in
    it does lets its nodes go before their own restorations. A node has one
    event in the queue that is taken (an entry superseded is skipped,
    wherever it falls), so the rank orders every tie that matters. The
    compiled engine's heap (``_kernel._push``, ``_pop``) orders by the same
    key.

    ``rank`` is ``class << 32 | place`` (a node's place from ``1 << 31``),
    one integer, so that entries compare as a float and an integer rather
    than through the events.
    """

    __slots__ = ("_heap", "_places", "_others")

    def __init__(self, nodes=()):
        self._heap: list = []
        self._places = {node: _NODES_FROM + k for k, node in enumerate(nodes)}
        # The causes' and shutdowns' places, by identity (a cause is a
        # dataclass, which two causes could be equal as).
        self._others: Dict[int, int] = {}

    def rank(self, event: "Event", tie: Optional[int] = None) -> int:
        """``event``'s rank among events at its time: its class (``tie``
        when given) above its node's place."""
        node = event.component
        place = self._places.get(node)
        if place is None:
            others = self._others
            place = others.get(id(node))
            if place is None:
                place = others[id(node)] = len(others)
        if tie is None:
            tie = tie_class(event)
        return (tie << _CLASS_SHIFT) | place

    def entry(self, event: "Event", tie: Optional[int] = None) -> tuple:
        """``event``'s entry in the heap."""
        return (event.time, self.rank(event, tie), event)

    def put(self, event: "Event", tie: Optional[int] = None) -> None:
        heapq.heappush(self._heap, self.entry(event, tie))

    def get(self) -> "Event":
        return heapq.heappop(self._heap)[2]

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
    them (found, or missed, alike), and for members started from states,
    when it last struck before 0; from its own streams, or numpy's global
    RNG."""

    __slots__ = ("_rate", "_gaps", "_coins", "_start")

    def __init__(self, rate: float, gaps=None, coins=None, start=None):
        self._rate = rate
        self._gaps = gaps
        self._coins = coins
        self._start = start

    def gap(self) -> float:
        if self._gaps is not None:
            return self._gaps.draw()
        return float(np.random.exponential(1.0 / self._rate))

    def coin(self) -> float:
        if self._coins is not None:
            return self._coins.draw()
        return float(np.random.random())

    def start_uniform(self) -> float:
        """The uniform that draws how long before 0 it last struck, for
        members started from states (#106)."""
        if self._start is not None:
            return self._start.draw()
        return float(np.random.random())


class _StandbyDraws:
    """What a standby group's units draw during a simulation: each unit's
    lives and repair times from streams of its own, and the group's switches
    from one (see ``_streams``); or, when they cannot be streamed, from the
    unit models' ``random`` and numpy's global RNG."""

    __slots__ = ("_models", "_life", "_lives", "_repairs", "_switches")

    def __init__(self, models, lives=None, repairs=None, switches=None):
        self._models = models
        # A unit's life in its own (operating) time: on a duty plan that
        # changes over time, the plan's model (see _StandbyGroup).
        life = models.reliability
        self._life = getattr(life, "model", life) if _planned(life) else life
        self._lives = lives
        self._repairs = repairs
        self._switches = switches

    def life(self, unit: int) -> float:
        if self._lives is None:
            return self._life.random(1).item()
        return self._lives[unit].draw()

    def repair(self, unit: int) -> float:
        if self._repairs is None:
            return self._models.time_to_replace.random(1).item()
        return self._repairs[unit].draw()

    def switch(self) -> float:
        if self._switches is None:
            return float(np.random.random())
        return self._switches.draw()


def _planned(life) -> bool:
    """Whether ``life`` is on a duty plan that changes over time."""
    from ._duty import PlannedLife

    return isinstance(life, PlannedLife)


class _StandbyGroup:
    """A standby group's units during one simulation (a component spec's
    ``"standby"``; see ``RepairableRBD``).

    On a duty plan that changes over time (the group's life a
    ``_duty.PlannedLife``), every unit's life is in operating time and
    is used up on the plan's clock ``D``: at rate 1 of ``D`` while it
    operates, and at ``dormancy_factor`` of ``D`` while it waits (a
    warm spare ages at its share of the duty, a cold one not at all), so
    a unit fails when ``D`` has run its life on from now (``_after``).

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

    A group with ``rates`` (load-sharing, or identical units: see
    ``_Standby``) has no spares: every unit up operates, a repaired unit
    operating at once, and the group is up while at least ``k`` do. While
    ``s`` operate each uses up its life at ``rates[s - 1]``, so whenever
    one fails or comes back, each other operating unit's life left is
    counted down at the rate it had and its failure put at the new rate
    (the cumulative-exposure model of ``LoadSharingModel``).

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
        "_rates",
        "_rate",
        "_draws",
        "_end",
        "_operating",
        "_spares",
        "_life",
        "_drawn",
        "_since",
        "_version",
        "_events",
        "_order",
        "_clock",
        "_ends",
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
        self._rates = arrangement.rates
        self._rate = 1.0
        self._draws = (
            draws if isinstance(draws, _StandbyDraws) else _StandbyDraws(draws)
        )
        self._end = t_simulation
        life = rbd.components[node].reliability
        self._clock = life if _planned(life) else None
        self._operating: set = set()
        self._spares: deque = deque()
        # Each unit's life left, as of ``_since``; and a count that a
        # queued event of the unit's must match to be current.
        self._life = [0.0] * arrangement.units
        # Each unit's life when it was drawn (put into service as new),
        # for its used life at the end (``ages``).
        self._drawn = [0.0] * arrangement.units
        self._since = [0.0] * arrangement.units
        self._version = [0] * arrangement.units
        self._events: list = []
        # When each unit's repair waiting for its spare ends, if a crew
        # takes it at once (see _repair).
        self._ends = [0.0] * arrangement.units
        self._order = 0
        self.entry: Optional["Event"] = None
        self.up = True
        if self._rates is not None:
            # Every unit operates, at the rate they share.
            self._rate = self._rates[arrangement.units - 1]
        for unit in range(arrangement.units):
            self._life[unit] = self._drawn[unit] = self._draws.life(unit)
            if self._rates is not None or unit < self._k:
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
        self._queue(
            self._after(t, self._life[unit] / self._rate), _UNIT_FAILS, unit
        )

    def _after(self, t: float, x: float) -> float:
        """When a unit's clock has run ``x`` on from ``t``: ``t + x``, or
        on a duty plan that changes, ``D⁻¹(D(t) + x)``."""
        if self._clock is None:
            return t + x
        return t + self._clock.run_on(t, x)

    def _ran(self, t0: float, t1: float) -> float:
        """How much a unit's clock has run from ``t0`` to ``t1``: ``t1 -
        t0``, or on a duty plan that changes, ``D(t1) - D(t0)``."""
        if self._clock is None:
            return t1 - t0
        return self._clock.operated_between(t0, t1)

    def _share(self, t: float) -> None:
        """The units operating at ``t`` have changed (a group with
        ``rates``): each other's life left is counted down at the rate it
        had, and its failure put at the rate they now share."""
        rate, rates = self._rate, self._rates
        assert rates is not None  # a group with rates alone
        if self._operating:
            self._rate = rates[len(self._operating) - 1]
        for unit in self._operating:
            self._life[unit] = max(
                self._life[unit] - rate * self._ran(self._since[unit], t),
                0.0,
            )
            self._version[unit] += 1
            self._operate(unit, t)

    def _advance_shared(self, t: float, kind: int, unit: int) -> int:
        """``advance`` for a group with ``rates``: how many units failed."""
        if kind == _UNIT_REPAIRED:
            self._life[unit] = self._drawn[unit] = self._draws.life(unit)
            self._since[unit] = t
            self._operating.add(unit)
            self._share(t)
            return 0
        self._operating.discard(unit)
        self._repair(unit, t)
        self._share(t)
        return 1

    def _wait(self, unit: int, t: float) -> None:
        self._spares.append(unit)
        self._since[unit] = t
        if self._dormancy > 0.0:
            self._queue(
                self._after(t, self._life[unit] / self._dormancy),
                _SPARE_FAILS,
                unit,
            )

    def _repair(self, unit: int, t: float) -> None:
        """Unit ``unit`` has failed at ``t``: its repair, drawn now, is a
        job for a crew."""
        self._version[unit] += 1
        ready = t
        shelf = self._rbd._run_state._shelf
        if shelf is not None and self._node in shelf.of:
            # Its replacement takes a spare, and waits for one if the
            # shelf is empty (see _shelves).
            ready = shelf.of[self._node][0].take(t)
        ends = ready + self._draws.repair(unit)
        crews = self._rbd._run_state._crews
        if crews is not None and ready > t:
            # Its job falls due when the spare is in, and only then
            # queues for a crew (the crew is not held while it waits).
            self._ends[unit] = ends
            self._queue(ready, _SPARE_IN, unit)
            return
        self._request(unit, t, ends)

    def _request(self, unit: int, t: float, ends: float) -> None:
        """Unit ``unit``'s repair, ending at ``ends`` if started at once,
        falls due at ``t``: a job for the crews."""
        crews = self._rbd._run_state._crews
        if crews is None or crews.request(
            _Unit(self._node, unit), t, Event(ends, None, True)
        ):
            self._queue(ends, _UNIT_REPAIRED, unit)
        elif crews.call_when:
            from . import _event_loop

            started = crews.call(t)
            if started is not None:
                _event_loop._crew_started(self._rbd, started)

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
        # Ranked at its instant as its unit's event (see _EventQueue).
        tie = RESTORATION if events[0][2] == _UNIT_REPAIRED else FAILURE
        self._rbd._run_state._event_queue.put(self.entry, tie)

    def holds(self, event: "Event") -> bool:
        """Whether ``event``, from the RBD's queue, is the group's current
        one (not superseded by an earlier)."""
        return event is self.entry

    def _released(self, unit: int, t: float) -> None:
        """Unit ``unit``'s repair is done at ``t``: its crew is free."""
        from . import _event_loop

        crews = self._rbd._run_state._crews
        if crews is not None:
            started = crews.release(_Unit(self._node, unit), t)
            if started is not None:
                _event_loop._crew_started(self._rbd, started)

    def ages(self, t: float) -> List[float]:
        """Each unit's age at ``t`` (the end of the run), NaN for one in
        its repair: the life it has used up since it was put into service
        as new, at its rate while it operates and at ``dormancy_factor``
        while it waits as a spare (a salvage by age credits each unit's
        worth at it)."""
        out = []
        for unit, drawn in enumerate(self._drawn):
            if unit in self._operating:
                rate = self._rate
            elif unit in self._spares:
                rate = self._dormancy
            else:
                out.append(math.nan)
                continue
            used = rate * self._ran(self._since[unit], t) if rate else 0.0
            out.append(drawn - max(self._life[unit] - used, 0.0))
        return out

    def advance(self, t: float) -> Tuple[bool, int]:
        """Take the group's event at ``t`` (its ``entry``): whether it is up
        after it, and how many of its units failed (each is repaired)."""
        self.entry = None
        events, version = self._events, self._version
        while events[0][4] != version[events[0][3]]:
            heapq.heappop(events)
        _, _, kind, unit, _ = heapq.heappop(events)
        failures = 0
        if kind == _SPARE_IN:
            self._request(unit, t, self._ends[unit])
            self._arm()
            return self.up, 0
        if self._rates is not None:
            if kind == _UNIT_REPAIRED:
                self._released(unit, t)
            failures = self._advance_shared(t, kind, unit)
            self.up = len(self._operating) >= self._k
            self._arm()
            return self.up, failures
        if kind == _UNIT_REPAIRED:
            self._released(unit, t)
            self._life[unit] = self._drawn[unit] = self._draws.life(unit)
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
                    used = self._dormancy * self._ran(self._since[spare], t)
                    self._life[spare] = max(self._life[spare] - used, 0.0)
                    self._operate(spare, t)
                else:
                    self._spares.appendleft(spare)
        self.up = len(self._operating) == self._k
        self._arm()
        return self.up, failures


class _MarkovDraws:
    """What a Markov component draws during a simulation: the uniforms
    of its times in each state, of its next states and of its state at
    0, from streams of its own (see ``_streams``), or from numpy's
    global RNG."""

    __slots__ = ("_sojourns", "_moves", "_start")

    def __init__(self, sojourns=None, moves=None, start=None):
        self._sojourns = sojourns
        self._moves = moves
        self._start = start

    @staticmethod
    def _uniform(stream) -> float:
        if stream is None:
            return float(np.random.random())
        return float(stream.draw())

    def sojourn(self) -> float:
        """A time in a state left at rate 1: an exponential draw."""
        return -math.log1p(-self._uniform(self._sojourns))

    def move(self) -> float:
        return self._uniform(self._moves)

    def start(self) -> float:
        return self._uniform(self._start)


class _MarkovUnit:
    """A Markov component during one simulation (``MarkovComponent``;
    see ``RepairableRBD``).

    It starts in a state drawn from its start, stays in each state for
    an exponential time at the state's total rate out, and moves to a
    next state chosen by the rates' shares, each from a uniform of its
    own. It is up in its working states, and a move from a working
    state to a failed one is a failure.

    As a standby group does (``_StandbyGroup``), it has one event in the
    RBD's queue, ``entry``: its next move. The RBD passes that event to
    ``advance``, which returns its state after it.
    """

    __slots__ = (
        "_rbd",
        "_node",
        "_draws",
        "_end",
        "_out",
        "_targets",
        "_shares",
        "_works",
        "_next",
        "state",
        "entry",
        "up",
    )

    def __init__(self, rbd, node, component, draws, t_simulation: float):
        self._rbd = rbd
        self._node = node
        self._draws = (
            draws if isinstance(draws, _MarkovDraws) else _MarkovDraws()
        )
        self._end = t_simulation
        generator = component._generator
        size = len(generator)
        self._out = [-float(generator[i, i]) for i in range(size)]
        # Each state's next states, and their rates' shares summed up.
        self._targets = [
            [j for j in range(size) if j != i and generator[i, j] > 0.0]
            for i in range(size)
        ]
        self._shares = [
            _shares([float(generator[i, j]) for j in self._targets[i]])
            for i in range(size)
        ]
        self._works = [bool(up) for up in component._up]
        started = [i for i in range(size) if component._start[i] > 0.0]
        if len(started) == 1:
            self.state = started[0]
        else:
            shares = _shares([float(component._start[i]) for i in started])
            self.state = started[_chosen(shares, self._draws.start())]
        self._next = self.state
        self.up = self._works[self.state]
        self.entry: Optional["Event"] = None
        self._arm(0.0)

    def _arm(self, t: float) -> None:
        """Draw the time in the state entered at ``t`` and the state
        after it, and queue that move in the RBD's queue."""
        rate = self._out[self.state]
        self.entry = None
        if rate == 0.0:
            return  # a state it never leaves
        time = t + self._draws.sojourn() / rate
        chosen = _chosen(self._shares[self.state], self._draws.move())
        self._next = self._targets[self.state][chosen]
        if time >= self._end:
            return
        self.entry = Event(time, self._node, self.up)
        # Ranked at its instant as the change it makes (see _EventQueue).
        tie = RESTORATION if self._works[self._next] else FAILURE
        self._rbd._run_state._event_queue.put(self.entry, tie)

    def holds(self, event: "Event") -> bool:
        """Whether ``event``, from the RBD's queue, is its current one."""
        return event is self.entry

    def ages(self, t: float) -> List[float]:
        """Its age at ``t``: none (NaN), as its states carry no age."""
        return [math.nan]

    def advance(self, t: float) -> Tuple[bool, int]:
        """Take its move at ``t`` (its ``entry``): whether it is up
        after it, and how many failures it made (1 for a move from a
        working state to a failed one)."""
        was = self.up
        self.state = self._next
        self.up = self._works[self.state]
        self._arm(t)
        return self.up, int(was and not self.up)


def _shares(weights: List[float]) -> List[float]:
    """The weights' shares of their sum, summed up: the last is 1."""
    total = math.fsum(weights)
    out, running = [], 0.0
    for weight in weights:
        running += weight
        out.append(running / total)
    if out:
        out[-1] = 1.0
    return out


def _chosen(shares: List[float], u: float) -> int:
    """Which of the choices with these summed shares the uniform ``u``
    takes."""
    return min(bisect.bisect_right(shares, u), len(shares) - 1)


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

    With a call-out crew (``call_when`` waiting jobs call it, a
    ``RepairableRBD``'s ``call_out``): a job that makes that many wait
    brings it, and it starts the first of them at once (``call``). While
    it is engaged there is a crew more; it leaves at the first end of a
    job that leaves a crew with no waiting job to take (``release``): the
    crews are alike, so whichever job it has then is finished by the crew
    that came free. ``calls`` counts its call-outs and ``engaged_time``
    its time engaged, which ``close`` ends at the end of the window. The
    crews' Markov chain copies this (``_crew_chain._enumerate``).
    """

    __slots__ = (
        "free",
        "served",
        "holding",
        "waiting",
        "_order",
        "_rank",
        "call_when",
        "engaged",
        "calls",
        "engaged_time",
        "_since",
    )

    def __init__(self, crews: int, served, priority: dict, call_when: int = 0):
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
        self.call_when = call_when
        self.engaged = False
        self.calls = 0
        self.engaged_time = 0.0
        self._since = 0.0

    def request(self, node, due: float, done: "Event") -> Optional["Event"]:
        """``node``'s job, due at ``due`` and ending with ``done`` if a crew
        starts it at once: ``done``, or None while the job waits (with a
        call-out crew, ``call`` then says whether it comes)."""
        if self.free:
            self.free -= 1
            self.holding.add(node)
            return done
        entry = (self._rank[node], due, self._order, node, done)
        heapq.heappush(self.waiting, entry)
        self._order += 1
        return None

    def call(self, t: float) -> Optional[Tuple[Any, float, "Event"]]:
        """A job has had to wait at ``t``: if that makes ``call_when`` of
        them and the call-out crew is not engaged, it comes and starts
        the first (as ``release`` gives it); or None."""
        if (
            not self.call_when
            or self.engaged
            or len(self.waiting) < self.call_when
        ):
            return None
        self.engaged = True
        self.calls += 1
        self._since = t
        return self._start(t)

    def release(self, node, t: float) -> Optional[Tuple[Any, float, "Event"]]:
        """``node``'s job is done at ``t``: its crew starts the next waiting
        job, if any. That job's component, how long it waited, and the event
        that now ends it; or None, and the call-out crew, if engaged,
        leaves in its place."""
        self.holding.discard(node)
        if self.waiting:
            return self._start(t)
        if self.engaged:
            self.engaged = False
            self.engaged_time += t - self._since
        else:
            self.free += 1
        return None

    def _start(self, t: float) -> Tuple[Any, float, "Event"]:
        """A crew starts the first waiting job at ``t``."""
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

    def close(self, t: float) -> Tuple[int, float]:
        """The call-out crew's call-outs and its time engaged by ``t``,
        the end of the window."""
        if self.engaged:
            return self.calls, self.engaged_time + (t - self._since)
        return self.calls, self.engaged_time


class _Fixed:
    """A cost charged at the same amount every time."""

    __slots__ = ("amount",)

    def __init__(self, amount: float):
        self.amount = amount

    def draw(self) -> float:
        return self.amount


class _Run:
    """A simulation in progress on a diagram: what ``initialize_event_queue``
    starts and ``next_event`` advances. The diagram holds it as
    ``_run_state`` while it runs (its public ``system_state``,
    ``component_status``, ``t_simulation`` and ``last_change_planned``
    read it), a run starts a new one, and its end forgets it
    (``_event_loop._forget_run``), so that nothing of a run is left on the
    diagram to be shared, pickled or read by the next."""

    __slots__ = (
        "_event_queue",
        "system_state",
        "t_simulation",
        "component_status",
        "last_change_planned",
        "_pending_failure",
        "_in_service",
        "_levels",
        "_renewed_at",
        "_pending_event",
        "_cancelled",
        "_early",
        "_renewing",
        "_crews",
        "_groups",
        "_step_sources",
        "_phases",
        "_ccf_members",
        "_ccf_pending",
        "_ccf_coins",
        "_ccf_struck",
        "_calendar_price",
        "_held",
        "_paused_pending",
        "_sound",
        "_shelf",
    )

    #: The events to come, in time order.
    _event_queue: "_EventQueue"
    #: Whether the system works, and each component.
    system_state: bool
    component_status: Dict[Hashable, bool]
    #: The window's end.
    t_simulation: float
    #: Whether the last change of the system's state was planned.
    last_change_planned: bool
    #: When each working node with hidden failures is due to fail (None
    #: once it has failed).
    _pending_failure: Dict[Hashable, Optional[float]]
    #: When each working node replaced on condition is due to fail, and to
    #: be replaced (see ``_event_loop._replaced_at``).
    _in_service: dict
    #: Each working node replaced on condition by its measured level: when
    #: its level was last known, and the level then.
    _levels: dict
    #: Opportunistic maintenance: when each member that can be renewed
    #: early was put into service as new, and its pending event; the
    #: pending events early renewals have cancelled, and the early
    #: renewals queued, by identity, and their members (see
    #: ``_event_loop._stop``).
    _renewed_at: dict
    _pending_event: dict
    _cancelled: dict
    _early: dict
    _renewing: set
    #: The repair crews, when there are fewer than the components that
    #: may need one (None otherwise).
    _crews: Optional["_Crews"]
    #: Each standby group's units.
    _groups: dict
    #: What the components draw their events from, when the diagram
    #: started the run itself (None when the caller passed them).
    _step_sources: Optional[dict]
    #: Each calendar a state shifts: by node, the time since its
    #: last scheduled replacement or test.
    _phases: dict
    #: Common-cause groups: their members, each member's next event,
    #: and the uniform a strike has decided its tests by.
    _ccf_members: frozenset
    _ccf_pending: dict
    _ccf_coins: dict
    #: Common-cause groups started from states (#106): each member a
    #: shared cause struck since it was last known up, and when (< 0).
    _ccf_struck: dict
    #: Each component on a calendar policy: the price of its next planned
    #: replacement (the opportunity's), charged when it falls; of its
    #: early renewal at a stop of its maintenance group, once queued
    #: (``inf``: the calendar allows none then, see
    #: ``_event_loop._CalendarCharge``).
    _calendar_price: dict
    #: Shutdowns (see ``_shutdowns``): the nodes a shut holds off line
    #: now, and each node a shut stands still with its pending event (None
    #: if it has none in the window).
    _held: set
    _paused_pending: dict
    #: Which components have failed, against down for planned work, and
    #: whether the system is down unplanned (see ``_unplanned``); None
    #: where that is not followed.
    _sound: Any
    #: The shelves of spares the components draw on (see ``_shelves``),
    #: or None.
    _shelf: Any

    def __init__(self) -> None:
        self._step_sources = None
        self._sound = None
        self._shelf = None


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
    >>> from repyability.rbd._events import Event
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
    level is at or past it), or under ``"calendar"`` at the first time
    its ``calendar`` policy's threshold is reached (``interval`` is then
    ``inf``); a replacement takes a time drawn from ``duration`` (None:
    no time)."""

    interval: float
    policy: str
    duration: Any
    threshold: float = 0.0
    #: Under ``"age"`` or ``"calendar"``, the age from which the unit is
    #: renewed early at a stop of its maintenance group (#108, #349);
    #: ``inf`` for never.
    opportunity: float = math.inf
    #: Under ``"condition"`` with a degradation process for a life (#271):
    #: the measured level at or past which an inspection replaces the
    #: unit, in place of the age-based ``threshold``; None for none.
    level: Optional[float] = None
    #: Under ``"calendar"`` (#319): the ``CalendarPolicy`` whose threshold
    #: at each time replaces the unit, at its calendar's price there.
    calendar: Any = None
    #: Under ``"block"`` or ``"condition"``: the time of the first action
    #: (then every ``interval``), from a shutdown's calendar; 0 puts the
    #: first at ``interval``, as an inspection's offset does.
    offset: float = 0.0
    #: Under ``"condition"`` at a shutdown's times: the age at or past
    #: which the unit is replaced there, in place of the ``threshold``;
    #: None for none.
    older_than: Optional[float] = None
    #: At given times (sorted, from 0) rather than every ``interval``: a
    #: shutdown's, or the component's own plan (``"replace_at"``); None
    #: for a period.
    times: Optional[Tuple[float, ...]] = None
    #: The name of the shutdown the actions are done at; None for none.
    #: At several shutdowns, their names (a tuple), with ``shuts``.
    shutdown: Any = None
    #: At several shutdowns (block replacement at each of their starts):
    #: each ``_Shutdown`` with the model of the time its renewal takes
    #: there (its own duration; None for none). ``interval`` is then the
    #: shortest of theirs and ``duration`` the longest, which only size
    #: the simulation's streams; None at one shutdown or none.
    shuts: Optional[Tuple[Tuple[Any, Any], ...]] = None
    #: Under ``"block"`` or ``"age"`` on its own interval, for an
    #: upgraded component (see ``_Upgrade``): the upgrade's time, at
    #: which the unit up is replaced as at a block time or its age (at
    #: its cost and in its time); None for none (or a unit kept).
    upgrade_at: Optional[float] = None

    #: Counted in operating time on a duty plan that changes over time
    #: (a ``_duty.PlannedLife``, the block's life): each action falls when
    #: the plan's operating time reaches the time it is due at in
    #: operating time; None on the diagram's clock.
    clock: Any = None

    @property
    def plan(self) -> Optional[Tuple[float, ...]]:
        """The component's own list of replacement times
        (``"replace_at"``), after the last of which it is replaced no
        more; None on a period, or at a shutdown's times."""
        return self.times if self.shutdown is None else None

    def due(self, renewed: float) -> float:
        """When the next preventive action (under ``"condition"``, the next
        inspection) falls, for a unit put into service as new at
        ``renewed``, or inspected then; an upgrade's time (``upgrade_at``)
        is one such action."""
        if self.clock is not None:
            # In operating time, and back on the diagram's clock.
            clock = self.clock
            due = clock.inverse(self._due(clock.operating(renewed)))
        else:
            due = self._due(renewed)
        if self.upgrade_at is not None and renewed < self.upgrade_at:
            return min(due, self.upgrade_at)
        return due

    def _due(self, renewed: float) -> float:
        if self.policy == "age":
            return renewed + self.interval
        if self.policy == "calendar":
            return self.calendar._next_planned(renewed, renewed)[0]
        if self.shuts is not None:
            # The first start of any of its shutdowns after ``renewed``.
            return min(shut.due(renewed) for shut, _ in self.shuts)
        if self.times is not None:
            return _next_time(self.times, renewed)
        if self.offset or self.shutdown is not None:
            # From a shutdown's first time on (as _Inspection.due, with or
            # without an offset: its times are offset + k * interval).
            k = np.floor((renewed - self.offset) / self.interval) + 1.0
            due = float(self.offset + k * self.interval)
            if due > renewed:
                return due
            return float(self.offset + (k + 1.0) * self.interval)
        # Block, or inspections: the next multiple of the interval after
        # ``renewed`` (a unit renewed on the schedule is not renewed again
        # there).
        due = float(np.floor(renewed / self.interval) + 1.0) * self.interval
        return due if due > renewed else due + self.interval

    def duration_at(self, t: float):
        """The model of the time the preventive action due at ``t`` takes
        (None for none): at several shutdowns, that of the one starting
        at ``t``; else ``duration``."""
        if self.shuts is None:
            return self.duration
        before = float(np.nextafter(t, -np.inf))
        for shut, model in self.shuts:
            if shut.due(before) == t:
                return model
        return self.duration


class _Upgrade(NamedTuple):
    """A component's upgrade (its spec's ``"upgrade"``): at the time
    ``at`` the unit in service is replaced by one of the life ``life``,
    as is every unit put into service after it, for the capital cost
    ``cost``, charged then. The replacement is in no time, or as its
    block or age replacement's (at its cost and in its time); a unit
    then in a repair (or a test) is replaced by the new life once it is
    over. With ``kept`` (its spec's ``"replace_in_service": False``),
    the unit in service at ``at`` is not replaced then: it stays until
    its next failure or due replacement, and the unit put into service
    then has the new life."""

    at: float
    life: Any
    cost: float
    kept: bool = False


class _Imperfect(NamedTuple):
    """A component's imperfect repair (#109): ``kijima`` (``"kijima1"`` or
    ``"kijima2"``) and its restoration factor ``q`` (0 renews, 1 is minimal
    repair), and the failure at which it is replaced instead, counting
    from its last renewal (None for never). A unit repaired by a fitted
    surpyval renewal process's own rule (G1, ARA, ARI; see
    ``_processes``) has ``kijima`` ``"process"``, ``q`` 1 (it is not
    renewed), and the fitted model as ``process``."""

    kijima: str
    q: float
    replace_after: Optional[int]
    process: Any = None


class _Restoration(NamedTuple):
    """A component's preventive maintenance that restores it only partly
    (a ``"preventive"`` spec's ``"restoration"``): ``kijima``
    (``"kijima1"`` or ``"kijima2"``) and its restoration factor ``q`` (0
    as new, 1 as bad as old, above 1 older than it was), or
    ``"to_age"`` and the virtual age ``q`` every task leaves the unit at
    (``"restore_to"``); and the task,
    counting from the unit's last renewal, that replaces it as new
    instead (``"replace_every"``; None for never)."""

    kijima: str
    q: float
    replace_every: Optional[int]


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
    #: At a shutdown's given times (sorted, from 0) rather than every
    #: ``interval``: those times; None for a period.
    times: Optional[Tuple[float, ...]] = None
    #: The name of the shutdown the tests are done at; None for none.
    shutdown: Any = None
    #: Counted in operating time on a duty plan that changes over time
    #: (a ``_duty.PlannedLife``, the block's life): each test falls when
    #: the plan's operating time reaches the time it is due at in
    #: operating time; None on the diagram's clock.
    clock: Any = None
    #: For an upgraded component (see ``_Upgrade``): the upgrade's time,
    #: at which the unit in service (working, or failed unseen) is
    #: replaced in no time by one of the new life (a test due then is
    #: not done); None for none (or a unit kept in service).
    upgrade_at: Optional[float] = None

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
        if self.clock is not None:
            # In operating time, and back on the diagram's clock.
            clock = self.clock
            plain = self._replace(clock=None)
            return clock.inverse(plain.due(float(clock.operating(t))))
        if self.times is not None:
            return _next_time(self.times, t)
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
        if self.clock is not None:
            # In operating time: the first test at or after the operating
            # time at t that falls at or after t itself (a test reached
            # before a stretch at no duty is not one after it).
            clock = self.clock
            plain = self._replace(clock=None)
            at = plain.finds(float(clock.operating(t)))
            found = clock.inverse(at)
            return found if found >= t else clock.inverse(plain.due(at))
        if self.times is not None:
            return _next_time(self.times, t, at=True)
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
        if self.clock is not None:
            t = float(self.clock.operating(t))
        k = int(round((t - self.offset) / self.interval))
        return k % self.per_full_test == 0

    def without_offset(self) -> "_Inspection":
        """The schedule on a calendar from 0 (a start from a state, whose
        ``since_scheduled`` places it)."""
        return self._replace(offset=0.0) if self.offset else self


def _next_time(times: Tuple[float, ...], t: float, at: bool = False) -> float:
    """The first of the sorted ``times`` after ``t`` (with ``at``, at or
    after it); ``inf`` if none is."""
    k = int(np.searchsorted(times, t, side="left" if at else "right"))
    return float(times[k]) if k < len(times) else math.inf


class _Shutdown(NamedTuple):
    """A shutdown (a diagram's ``shutdowns``): the times it starts, every
    ``period`` from ``offset`` (0 puts the first at ``period``, as an
    inspection's offset does) or at the sorted ``times``; the time it
    lasts, ``duration``; the nodes it holds off line (``scope``, None for
    the whole system); its ``cost``, charged at each; and whether the
    blocks in its scope not worked on keep ageing through it
    (``idle_ageing``) or stand still until it ends; and the components
    it renews if they are old enough (``opportunistic``)."""

    name: Any
    #: A float, or None at given times.
    period: Any
    offset: float
    times: Optional[Tuple[float, ...]]
    duration: float
    scope: Optional[Tuple[Any, ...]]
    cost: float
    idle_ageing: bool
    #: The components renewed at each start if old enough: ``{node:
    #: age}``, a number for every component in the scope under age
    #: replacement, or None (resolved by ``_shutdowns.check``).
    opportunistic: Any = None

    @property
    def periodic(self) -> bool:
        """Whether it starts every ``period`` (else at given times)."""
        return self.times is None

    def due(self, t: float) -> float:
        """The first start after ``t`` (as a task bound to it falls due:
        see ``_Preventive.due``)."""
        if self.times is not None:
            return _next_time(self.times, t)
        return _Preventive(
            float(self.period), "block", None, offset=self.offset, shutdown=0
        ).due(t)

    def starts(self, stop: float) -> np.ndarray:
        """Its starts in ``[0, stop)``: ``offset + k * period`` (from the
        first ``k`` that is more than 0 with no offset), as ``due`` and a
        task bound to it work them out, to the last bit."""
        if self.times is not None:
            times = np.asarray(self.times, dtype=float)
            return times[times < stop]
        if not stop > 0.0:
            return np.empty(0)
        first = 0.0 if self.offset else 1.0
        last = np.floor((stop - self.offset) / self.period) + 1.0
        k = np.arange(first, max(first, last) + 1.0)
        times = self.offset + k * float(self.period)
        return times[times < stop]

    def windows(self, stop: float) -> Tuple[np.ndarray, np.ndarray]:
        """Its windows that start before ``stop``: their starts and ends
        (none for a shutdown in no time)."""
        if not self.duration > 0.0:
            return np.empty(0), np.empty(0)
        starts = self.starts(stop)
        return starts, starts + self.duration


class _Standby(NamedTuple):
    """A group of identical units, each failing and repaired on its own,
    as one node (see ``_StandbyGroup``).

    A standby group (a component spec's ``"standby"``, ``rates`` None):
    ``units`` units, ``k`` of which must operate, the rest waiting as
    spares that age at ``dormancy_factor`` of the operating rate, switched
    in with ``switching_probability``.

    With ``rates`` (a spec's ``"load_sharing"`` or ``"identical"``), every
    unit that is up operates, none waits, and the group is up while at
    least ``k`` operate. While ``s`` operate, each uses up its life at
    ``rates[s - 1]``: a load-sharing group's units' ageing under the load
    ``s`` of them share, and 1 throughout for independent identical
    units. Such a group has a dormancy of 0 and switching of 1, which it
    never uses."""

    units: int
    k: int
    dormancy_factor: float
    switching_probability: float
    rates: Optional[Tuple[float, ...]] = None
    #: The spec's key that made the group: ``"standby"``,
    #: ``"load_sharing"`` or ``"identical"``.
    key: str = "standby"

    @property
    def what(self) -> str:
        """The group in words, for a message."""
        return GROUP_KINDS[self.key]


#: Each kind of group of units in words, by the spec's key.
GROUP_KINDS = {
    "standby": "a standby group",
    "load_sharing": "a load-sharing group",
    "identical": "a block of identical units",
    "markov": "a Markov component",
}

#: A Markov component (``MarkovComponent``) as a group: one unit, whose
#: states and moves are its chain's (see ``_MarkovUnit``).
MARKOV = _Standby(1, 1, 0.0, 1.0, None, "markov")
