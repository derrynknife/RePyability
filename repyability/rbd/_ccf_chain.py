"""Common-cause groups in a ``RepairableRBD`` (#136): which of a group's
members are down together, in the long run.

A group's members are identical units with a constant failure rate. Its
model (a ``BetaFactor`` or ``MGL``, see ``ccf.py``) splits that rate
between causes: each member's own, and shared ones, each a set of members
that one cause fails at once, with a share of the rate (``_causes``). Each
cause strikes as a Poisson process, and fails the members it names that
are up (those already down stay down). So each member on its own still
fails at its rate while it is up, and its own long-run values are as
without the group: what the group changes is which members are down
together.

- Members whose failures are revealed, each repaired at a constant rate,
  form a continuous-time Markov chain over which of them are down. Its
  long-run distribution is found by the GTH algorithm (Grassmann, Taksar
  and Heyman), which only adds, multiplies and divides positive numbers,
  so a small probability keeps its precision.
- Members whose failures are hidden are found by their tests, at their
  own offsets and intervals: a test finds a failure with the group's
  coverage (all the failures one cause makes are found alike, or missed
  alike), and a full test finds every one. A member is up, down until
  its next test, or down until its next full test (missed). Between tests
  the members only go down, and at a test the member tested is repaired
  if its failure is found; the state at any time is worked out by
  uniformization, a sum of positive terms again. Once every member has
  had a full test, the state no longer depends on where it started (each
  member is up after its full test, and from then on its state depends
  only on the causes and its own tests), so one period of the tests from
  all up reaches the long run, and a second gives the state at each time.
  A member whose tests or repairs take time (#220) has more levels: off
  line for a test while working (where it neither ages nor is struck),
  under a test while failed, and under repair. A test or repair of a fixed
  length ends at a fixed time after its test, as a test does; one of an
  exponential length at a rate, as a cause strikes. A test that falls in
  a member's own test or repair is not done. With those, two periods from
  all up reach the long run (a repair can run past a period's end), and a
  third gives the state at each time.

In the redundancy allocation (#158) a member may have copies, which join
its group (a ``BetaFactor`` model's ``beta`` holding at any size: its
shared cause fails every copy of every member). A member's copies are
alike, with the same rates, causes and tests, so the chain need only count
how many of each member's copies are at each level (``_Counted``), and the
member's node is down while all of them are. With one copy each, the
counts are the members' own levels, in the same order, and the chain is
the group's own.
"""

import math
from itertools import product
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

#: The most states a group's chain may have.
MAX_STATES = 20_000

UP, FOUND, MISSED = 0, 1, 2

#: A member's levels when its tests or repairs take time (#220): off line
#: for a test while working, under a test while failed, under repair.
OFF, TESTING, REPAIR = 3, 4, 5

#: What happens at a ``ProofTest``'s time: the member's test, or the end of
#: one of its tests (``TESTED``) or repairs (``REPAIRED``) of a fixed length.
TEST, TESTED, REPAIRED = 0, 1, 2


class GroupStates(NamedTuple):
    """Which of a group's ``members`` are down in each combination
    (``down``, one row per combination, every combination of the members
    up or down), and each combination's probability at each time
    (``probabilities``, one row per time, adding up to 1)."""

    members: Tuple[Hashable, ...]
    down: np.ndarray
    probabilities: np.ndarray


def causes(model, members: Sequence[Hashable], rate: float) -> List[tuple]:
    """The group's causes: ``(member positions, rate)``, each member's own
    first, then the shared ones (see ``ccf._Model._causes``)."""
    own, shared = model._causes(tuple(members))
    position = {member: i for i, member in enumerate(members)}
    out: List[Tuple[Tuple[int, ...], float]] = [
        ((i,), own * rate) for i in range(len(members))
    ]
    out += [
        (tuple(sorted(position[m] for m in struck)), fraction * rate)
        for struck, fraction in shared
        if fraction > 0.0
    ]
    return out


def _states(n: int, levels: int) -> np.ndarray:
    """Every state of ``n`` members with ``levels`` levels each, one row
    per state (its index read in base ``levels``, the first member
    last)."""
    return np.array(list(product(range(levels), repeat=n)), dtype=np.int8)[
        :, ::-1
    ]


def _index(states: np.ndarray, levels: int) -> np.ndarray:
    """Each state's index (see ``_states``)."""
    return states.astype(np.int64) @ (levels ** np.arange(states.shape[1]))


class _Counted:
    """The states of a group's members, ``counts[i]`` alike copies of
    member ``i`` (see the module's notes), each copy up or down at one of
    the levels below ``levels`` (``FOUND``, and with three ``MISSED``): a
    state counts how many of each member's copies are ``found`` and
    ``missed`` (one row per state, one column per member), the others
    ``up``. Each member's counts are a digit of the state's index, the
    first member's the lowest, its (found, missed) counts in the order
    (0, 0), (1, 0), ... (counts, 0), (0, 1), (1, 1), ...: with one copy
    each, the states are those of ``_states(n, levels)``, in its order."""

    def __init__(self, members, counts, levels: int):
        self.counts = counts = np.asarray(counts, dtype=np.int64)
        options = [
            [
                (f, m)
                for m in (range(c + 1) if levels == 3 else (0,))
                for f in range(c + 1 - m)
            ]
            for c in counts.tolist()
        ]
        self.size = size = math.prod(len(o) for o in options)
        _check_size(size, members)
        radices = np.array([len(o) for o in options], dtype=np.int64)
        self.strides = np.concatenate([[1], np.cumprod(radices)[:-1]]).astype(
            np.int64
        )
        state = np.arange(size, dtype=np.int64)
        n = len(counts)
        self.found = np.empty((size, n), dtype=np.int64)
        self.missed = np.empty((size, n), dtype=np.int64)
        self.place = []
        for i, (c, pairs) in enumerate(zip(counts.tolist(), options)):
            place = np.full((c + 1, c + 1), -1, dtype=np.int64)
            for k, (f, m) in enumerate(pairs):
                place[f, m] = k
            self.place.append(place)
            digit = (state // self.strides[i]) % radices[i]
            self.found[:, i] = np.array([f for f, _ in pairs])[digit]
            self.missed[:, i] = np.array([m for _, m in pairs])[digit]
        self.up = counts[None, :] - self.found - self.missed

    def index(self, found: np.ndarray, missed: np.ndarray) -> np.ndarray:
        """The index of the states with these counts."""
        out = np.zeros(len(found), dtype=np.int64)
        for i, place in enumerate(self.place):
            out += place[found[:, i], missed[:, i]] * self.strides[i]
        return out

    def strike(
        self, struck: Tuple[int, ...], level: int, own: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """The state each state goes to when a cause strikes the members
        ``struck``, to ``level``: a member's ``own`` cause one of its up
        copies (each at the cause's rate, so the rate is as many times the
        cause's as it has up), a shared one every up copy of each. Also
        that multiple of the cause's rate (1 for a shared cause)."""
        found, missed = self.found.copy(), self.missed.copy()
        hit = found if level == FOUND else missed
        times = np.ones(self.size)
        for i in struck:
            if own:
                hit[:, i] += np.minimum(self.up[:, i], 1)
                times = self.up[:, i].astype(float)
            else:
                hit[:, i] += self.up[:, i]
        return self.index(found, missed), times

    def combinations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Every combination of the members up or down (``down``, in the
        order of ``_states(n, 2)``), a member down while every copy of it
        is, and the combination of each state."""
        n = len(self.counts)
        down = _states(n, 2).astype(bool)
        return down, _index((self.up == 0).astype(np.int8), 2)


#: A move of a group's chain: whose it is (a member's position, for its
#: own cause and its repairs, or ``SHARED`` for a cause that strikes more
#: than one), and its sources, targets and rates.
Move = Tuple[int, np.ndarray, np.ndarray, np.ndarray]
SHARED = -1


def _revealed_moves(
    model,
    members: Sequence[Hashable],
    rate: float,
    repair: float,
    counts: Optional[Sequence[int]] = None,
) -> Tuple[int, List[Move]]:
    """The moves between the states of a group whose members' failures
    are revealed, each repaired at rate ``repair`` (independently: as many
    repairs at once as are needed), its states those of ``_Counted``
    (with ``counts`` copies of each member, by default one: each state
    then its own combination of the members up or down), and how many
    states it has. Each move is labelled with whose it is (see
    ``Move``)."""
    n = len(members)
    space = _Counted(members, [1] * n if counts is None else counts, 2)
    source = np.arange(space.size)
    out: List[Move] = []
    for number, (struck, cause) in enumerate(causes(model, members, rate)):
        target, times = space.strike(struck, FOUND, own=number < n)
        moves = target != source
        out.append(
            (
                number if number < n else SHARED,
                source[moves],
                target[moves],
                cause * times[moves],
            )
        )
    for i in range(n):
        down = space.found[:, i] > 0
        fixed = space.found.copy()
        fixed[:, i] -= down.astype(np.int64)
        out.append(
            (
                i,
                source[down],
                space.index(fixed, space.missed)[down],
                repair * space.found[down, i].astype(float),
            )
        )
    return space.size, out


def _revealed_flow(
    model,
    members: Sequence[Hashable],
    rate: float,
    repair: float,
    counts: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """The rates between the states of a group whose members' failures
    are revealed (see ``_revealed_moves``)."""
    size, moves = _revealed_moves(model, members, rate, repair, counts)
    flow = np.zeros((size, size))
    for _, begin, end, value in moves:
        np.add.at(flow, (begin, end), value)
    return flow


def revealed(
    model,
    members: Sequence[Hashable],
    rate: float,
    repair: float,
    counts: Optional[Sequence[int]] = None,
) -> GroupStates:
    """The long-run states of a group whose members' failures are
    revealed, each repaired at rate ``repair`` (independently: as many
    repairs at once as are needed), with ``counts`` copies of each member
    (see ``_Counted``)."""
    n = len(members)
    stationary = gth(_revealed_flow(model, members, rate, repair, counts))
    space = _Counted(members, [1] * n if counts is None else counts, 2)
    combinations, combination = space.combinations()
    grouped = np.bincount(
        combination, weights=stationary, minlength=len(combinations)
    )
    return GroupStates(tuple(members), combinations, grouped[None, :])


def gth(flow: np.ndarray) -> np.ndarray:
    """The long-run distribution of the continuous-time Markov chain with
    the rates ``flow[i, j]`` from state ``i`` to state ``j`` (its diagonal
    ignored), by the GTH algorithm: states are folded into those before
    them, one at a time from the last, and the distribution unfolded."""
    rates = np.array(flow, dtype=float)
    np.fill_diagonal(rates, 0.0)
    size = len(rates)
    for last in range(size - 1, 0, -1):
        out = rates[last, :last].sum()
        if out <= 0.0:
            raise ValueError("The chain does not leave one of its states.")
        rates[:last, last] /= out
        rates[:last, :last] += np.outer(rates[:last, last], rates[last, :last])
    weights = np.zeros(size)
    weights[0] = 1.0
    for j in range(1, size):
        weights[j] = weights[:j] @ rates[:j, j]
    return weights / weights.sum()


class ProofTest(NamedTuple):
    """A member's test: when, which member, and whether it is a full test
    (one that finds a failure it would otherwise miss). With ``kind``, the
    end of one of its tests or repairs of a fixed length instead."""

    time: float
    member: int
    full: bool
    kind: int = TEST


class Duration(NamedTuple):
    """How long a member's tests, or its repairs, take: a fixed
    time (``fixed``, 0 for none), or an exponential one (at ``rate``)."""

    fixed: float = 0.0
    rate: Optional[float] = None

    @property
    def takes_time(self) -> bool:
        return self.fixed > 0.0 or self.rate is not None


class Timing(NamedTuple):
    """How long a member's tests and repairs take (see ``Duration``)."""

    test: Duration = Duration()
    repair: Duration = Duration()

    @property
    def takes_time(self) -> bool:
        return self.test.takes_time or self.repair.takes_time


#: The most memory a chain's transitions over the steps it has taken more
#: than once may hold (see ``_Hidden.evolve``).
_TRANSITION_BYTES = 1 << 26


class _Hidden:
    """The chain of a group whose members' failures are hidden, found by
    their tests (see ``hidden``): its states, a step of the uniformized
    chain between tests, and where each member's test takes each state."""

    def __init__(
        self,
        model,
        members,
        rate: float,
        coverage: float,
        counts: Optional[Sequence[int]] = None,
    ):
        n = len(members)
        self.partial = partial = coverage < 1.0
        self.levels = levels = 3 if partial else 2
        space = _Counted(
            members, [1] * n if counts is None else counts, levels
        )
        self.size = size = space.size
        # The uniformized chain: a step of it is the chance of each move in
        # a time 1 / pace.
        source = np.arange(size)
        #: The causes' moves, labelled (see ``Move``).
        self.moves: List[Move] = []
        for number, (struck, cause) in enumerate(causes(model, members, rate)):
            for level, share in ((FOUND, coverage), (MISSED, 1.0 - coverage)):
                if share <= 0.0 or (level == MISSED and not partial):
                    continue
                target, times = space.strike(struck, level, own=number < n)
                moved = target != source
                self.moves.append(
                    (
                        number if number < n else SHARED,
                        source[moved],
                        target[moved],
                        cause * share * times[moved],
                    )
                )
        outflow = np.zeros(size)
        for _, begin, _, value in self.moves:
            np.add.at(outflow, begin, value)
        self.pace = pace = float(outflow.max()) if self.moves else 0.0
        step = np.zeros((size, size))
        if pace > 0.0:
            for _, begin, end, value in self.moves:
                np.add.at(step, (begin, end), value / pace)
        np.fill_diagonal(step, 1.0 - outflow / pace if pace > 0.0 else 1.0)
        self.step = step
        # Each member's tests (of all its copies): where each state goes.
        self.found = []
        for i in range(n):
            after = {}
            for full in (False, True):
                found, missed = space.found.copy(), space.missed.copy()
                found[:, i] = 0
                if full:
                    missed[:, i] = 0
                after[full] = space.index(found, missed)
            self.found.append(after)
        self.combinations, self.combination = space.combinations()

    def evolve(self, v: np.ndarray, dt: float) -> np.ndarray:
        """``v exp(G dt)``, by uniformization: ``exp(G dt)`` worked out as
        a matrix and kept once ``dt`` comes again (#229), as the steps
        between tests and times repeat, period after period."""
        if not (dt > 0.0 and self.pace > 0.0):
            return v
        kept = self.__dict__.setdefault("_transitions", {})
        found = kept.get(dt)
        if found is None:
            # The first time, the vector alone.
            if len(kept) * 8 * self.size**2 >= _TRANSITION_BYTES:
                kept.clear()
            kept[dt] = False
            return self._uniformized(v, dt)
        if found is False:
            found = kept[dt] = self._uniformized(np.eye(self.size), dt)
        return v @ found

    def _uniformized(self, v: np.ndarray, dt: float) -> np.ndarray:
        """``v exp(G dt)`` (``v`` a vector, or a matrix's rows), by
        uniformization, in steps short enough for ``exp(-x)`` not to
        underflow."""
        pace, step = self.pace, self.step
        while dt > 0.0 and pace > 0.0:
            piece = min(dt, 8.0 / pace)
            dt -= piece
            x = pace * piece
            weight = np.exp(-x)
            term = v
            out = weight * term
            k = 0
            while True:
                k += 1
                term = term @ step
                weight *= x / k
                out = out + weight * term
                if k > x and weight < 1e-34:
                    break
            v = out
        return v

    def tested_at(self, v: np.ndarray, test: "ProofTest") -> np.ndarray:
        """``v`` after the member's test ``test``."""
        target = self.found[test.member][test.full or not self.partial]
        return np.bincount(target, weights=v, minlength=self.size)

    def start(self) -> np.ndarray:
        """Every member up."""
        v = np.zeros(self.size)
        v[0] = 1.0
        return v

    def follow(self, tests, times: np.ndarray, v=None, now: float = 0.0):
        """The state at each of ``times`` (rows), from ``v`` (every member
        up) at ``now``, through the ``tests`` in order (tests at a time come
        before it)."""
        order = sorted(tests)
        v = self.start() if v is None else v
        out = np.empty((len(times), self.size))
        upcoming = 0
        for row in np.argsort(times, kind="stable"):
            t = float(times[row])
            while upcoming < len(order) and order[upcoming].time <= t:
                test = order[upcoming]
                if test.time >= now:
                    v = self.tested_at(self.evolve(v, test.time - now), test)
                    now = test.time
                upcoming += 1
            v = self.evolve(v, t - now)
            now = t
            out[row] = v
        return out

    def grouped(self, states: np.ndarray) -> np.ndarray:
        """Each row of ``states`` summed over each combination of the
        members up or down."""
        out = np.zeros((len(states), len(self.combinations)))
        for j in range(len(self.combinations)):
            out[:, j] = states[:, self.combination == j].sum(axis=1)
        return out


class _Timed(_Hidden):
    """``_Hidden`` for members whose tests or repairs take time (#220,
    see the module's notes), one copy of each member: each member's level
    is one of ``levels`` (``UP``, ``FOUND``, ``MISSED`` where tests can
    miss a failure, ``OFF`` and ``TESTING`` where tests take time,
    ``REPAIR`` where repairs do), a state one level for each member (see
    ``_states``). Causes strike the members that are up; a test or repair
    of an exponential length ends at its rate; one of a fixed length ends
    at a ``ProofTest`` of its ``kind``, a fixed time after the test."""

    def __init__(self, model, members, rate: float, coverage: float, timings):
        n = len(members)
        self.partial = partial = coverage < 1.0
        tests_take = any(t.test.takes_time for t in timings)
        repairs_take = any(t.repair.takes_time for t in timings)
        ladder = [UP, FOUND] + ([MISSED] if partial else [])
        ladder += [OFF, TESTING] if tests_take else []
        ladder += [REPAIR] if repairs_take else []
        #: The levels a member may be at.
        self.ladder = ladder
        self.levels = count = len(ladder)
        code = {level: k for k, level in enumerate(ladder)}
        states = _states(n, count)
        self.size = size = len(states)
        _check_size(size, members)
        source = np.arange(size)

        def index(target: np.ndarray) -> np.ndarray:
            return _index(target, count)

        def moving(member: int, rules) -> np.ndarray:
            """The state each state goes to with ``member`` moved by
            ``rules`` (``{level: level}``), the others as they are."""
            target = states.copy()
            column = states[:, member]
            for old, new in rules.items():
                if old in code and new in code:
                    target[column == code[old], member] = code[new]
            return index(target)

        #: The continuous moves, labelled (see ``Move``).
        self.moves: List[Move] = []
        for number, (struck, cause) in enumerate(causes(model, members, rate)):
            for level, share in ((FOUND, coverage), (MISSED, 1.0 - coverage)):
                if share <= 0.0 or (level == MISSED and not partial):
                    continue
                target = states.copy()
                for i in struck:
                    target[:, i] = np.where(
                        states[:, i] == code[UP], code[level], states[:, i]
                    )
                struck_to = index(target)
                moved = struck_to != source
                self.moves.append(
                    (
                        number if number < n else SHARED,
                        source[moved],
                        struck_to[moved],
                        np.full(int(moved.sum()), cause * share),
                    )
                )
        self.jumps: Dict[Tuple[int, int, bool], np.ndarray] = {}
        for i, timing in enumerate(timings):
            after = REPAIR if timing.repair.takes_time else UP
            ends = {
                "test": {OFF: UP, TESTING: after},
                "repair": {REPAIR: UP},
            }
            for which, duration in (
                ("test", timing.test),
                ("repair", timing.repair),
            ):
                if duration.rate is not None:
                    for old, new in ends[which].items():
                        target = moving(i, {old: new})
                        hit = target != source
                        self.moves.append(
                            (
                                i,
                                source[hit],
                                target[hit],
                                np.full(int(hit.sum()), duration.rate),
                            )
                        )
                elif duration.fixed > 0.0:
                    kind = TESTED if which == "test" else REPAIRED
                    target = moving(i, ends[which])
                    self.jumps[(i, kind, False)] = target
                    self.jumps[(i, kind, True)] = target
            # Its test: off line if it takes time, a failure found into its
            # test or repair; one in its own test or repair is not done.
            found = (
                TESTING
                if timing.test.takes_time
                else (REPAIR if timing.repair.takes_time else UP)
            )
            for full in (False, True):
                rules = {FOUND: found}
                if timing.test.takes_time:
                    rules[UP] = OFF
                if partial and full:
                    rules[MISSED] = found
                self.jumps[(i, TEST, full)] = moving(i, rules)
        outflow = np.zeros(size)
        for _, sources, _, value in self.moves:
            np.add.at(outflow, sources, value)
        self.pace = pace = float(outflow.max()) if self.moves else 0.0
        step = np.zeros((size, size))
        if pace > 0.0:
            for _, sources, targets, value in self.moves:
                np.add.at(step, (sources, targets), value / pace)
        np.fill_diagonal(step, 1.0 - outflow / pace if pace > 0.0 else 1.0)
        self.step = step
        self.combinations = _states(n, 2).astype(bool)
        self.combination = _index((states != code[UP]).astype(np.int8), 2)

    def tested_at(self, v: np.ndarray, test: "ProofTest") -> np.ndarray:
        """``v`` after the member's test, or the end of its test or repair,
        ``test``."""
        full = test.full or not self.partial or test.kind != TEST
        target = self.jumps.get((test.member, test.kind, full))
        if target is None:
            return v
        return np.bincount(target, weights=v, minlength=self.size)


def timed_events(tests, timings, period: float) -> List[ProofTest]:
    """``tests`` (each member's, in ``(0, period]``) with the ends of the
    tests and repairs of a fixed length after each, each in
    ``(0, period]`` as the schedule repeats: an end past the period is the
    previous period's test's, at the start of this one (where, from all up
    at 0, it ends nothing)."""
    out = [test._replace(kind=TEST) for test in tests]
    for test in tests:
        timing = timings[test.member]
        ends = []
        if timing.test.rate is None and timing.test.fixed > 0.0:
            ends.append((timing.test.fixed, TESTED))
        if timing.repair.rate is None and timing.repair.fixed > 0.0:
            ends.append((timing.test.fixed + timing.repair.fixed, REPAIRED))
        for length, kind in ends:
            time = test.time + length
            time = time - period * math.floor((time - 1e-12 * period) / period)
            out.append(ProofTest(time, test.member, True, kind))
    return sorted(out)


def _timed(timings) -> bool:
    return timings is not None and any(t.takes_time for t in timings)


def _shifted(tests, by: float) -> List[ProofTest]:
    return [test._replace(time=test.time + by) for test in tests]


def hidden(
    model,
    members: Sequence[Hashable],
    rate: float,
    coverage: float,
    tests: Sequence[ProofTest],
    period: float,
    times: np.ndarray,
    counts: Optional[Sequence[int]] = None,
    timings: Optional[Sequence[Timing]] = None,
) -> GroupStates:
    """The long-run states, at each of ``times`` (in ``[0, period)``), of a
    group whose members' failures are hidden, found by ``tests`` (each
    member's, in ``(0, period]``, the schedule repeating every
    ``period``), with ``coverage`` the chance that a test finds a cause's
    failures (a full test finds all), and ``counts`` copies of each member
    (see ``_Counted``), tested together; or, with ``timings`` that take
    time, one copy of each, its tests and repairs taking them (see
    ``_Timed``)."""
    if _timed(timings):
        timed = _Timed(model, members, rate, coverage, timings)
        events = timed_events(tests, timings, period)
        v = timed.follow(events, np.array([period]))[0]
        v = timed.follow(
            _shifted(events, period), np.array([2.0 * period]), v, period
        )[0]
        out = timed.follow(
            _shifted(events, 2.0 * period),
            2.0 * period + np.asarray(times),
            v,
            2.0 * period,
        )
        return GroupStates(
            tuple(members), timed.combinations, timed.grouped(out)
        )
    chain = _Hidden(model, members, rate, coverage, counts)
    # From all up, one period reaches the long run (every member has had a
    # full test by its end); a second gives the state at each time.
    v = chain.follow(tests, np.array([period]))[0]
    out = chain.follow(
        _shifted(tests, period), period + np.asarray(times), v, period
    )
    return GroupStates(tuple(members), chain.combinations, chain.grouped(out))


class OverTime:
    """A group's members' joint states over time from every member up at
    0 (see ``probabilities``): from its chain followed by uniformization
    (``revealed`` failures), or (``hidden``) through its tests' first
    period, by the end of which every member has had a full test and the
    states repeat the long run's every period."""

    def __init__(self, members, combinations, settle: float, period):
        self.members = tuple(members)
        #: Which members are down in each combination (see GroupStates).
        self.down = combinations
        self.settle = settle
        self.period = period
        self._chain: Any = None
        self._tests: Optional[list] = None
        self._settled: Any = None
        #: The chain's moves, labelled (see ``Move``), and the combination
        #: of the members up or down of each of its states (#199).
        self.moves: List[Move] = []
        self.combination = np.arange(len(combinations))

    @classmethod
    def revealed(
        cls, model, members, rate: float, repair: float, uniformized
    ) -> "OverTime":
        """A group whose failures are revealed (see ``revealed``), its
        chain followed by ``uniformized(generator, start, steady,
        vectors)`` (a ``_chain_transient.Uniformized``)."""
        size, moves = _revealed_moves(model, members, rate, repair)
        flow = np.zeros((size, size))
        for _, begin, end, value in moves:
            np.add.at(flow, (begin, end), value)
        steady = gth(flow)
        generator = flow.copy()
        np.fill_diagonal(generator, 0.0)
        np.fill_diagonal(generator, -generator.sum(axis=1))
        start = np.zeros(size)
        start[0] = 1.0
        chain = uniformized(generator, start, steady, np.identity(size))
        out = cls(
            members, _states(len(members), 2).astype(bool), chain.settle, None
        )
        out._chain = chain
        out._tests = None
        out.moves = moves
        return out

    @classmethod
    def hidden(
        cls,
        model,
        members,
        rate: float,
        coverage: float,
        tests,
        period,
        timings: Optional[Sequence[Timing]] = None,
    ) -> "OverTime":
        """A group whose failures are hidden (see ``hidden``): with
        ``timings`` that take time, its tests' and repairs' ends
        among its tests, and settling after two periods."""
        chain: _Hidden
        if _timed(timings):
            chain = _Timed(model, members, rate, coverage, timings)
            tests = timed_events(tests, timings, period)
            settle = 2.0 * float(period)
        else:
            chain = _Hidden(model, members, rate, coverage)
            settle = float(period)
        out = cls(members, chain.combinations, settle, float(period))
        out._chain = chain
        out._tests = sorted(tests)
        # The states at the start of the period that repeats.
        out._settled = chain.follow(out._early(settle), np.array([settle]))[0]
        out.moves = chain.moves
        out.combination = chain.combination
        return out

    def _early(self, stop: float) -> List[ProofTest]:
        """The tests (and their ends) from 0 to ``stop``, in order."""
        assert self._tests is not None
        cycles = int(np.ceil(stop / self.period))
        return sorted(
            test
            for k in range(max(cycles, 1))
            for test in _shifted(self._tests, k * self.period)
        )

    def raw(self, x) -> np.ndarray:
        """Each of the chain's states' probability (columns) at each time
        ``x`` (rows): at a test's time, after it."""
        x = np.asarray(x, dtype=float).ravel()
        if self._tests is None:
            values = self._chain.values(x)
        else:
            chain, period, settle = self._chain, self.period, self.settle
            values = np.empty((len(x), chain.size))
            early = x <= settle
            if early.any():
                values[early] = chain.follow(self._early(settle), x[early])
            if (~early).any():
                phase = np.mod(x[~early] - settle, period)
                values[~early] = chain.follow(
                    _shifted(self._tests, settle),
                    settle + phase,
                    self._settled,
                    settle,
                )
        values = np.maximum(values, 0.0)
        total = values.sum(axis=1, keepdims=True)
        return values / np.where(total > 0.0, total, 1.0)

    def probabilities(self, x) -> np.ndarray:
        """Each combination's probability (columns) at each time ``x``
        (rows): at a test's time, after it."""
        values = self.raw(x)
        if self._tests is not None:
            values = self._chain.grouped(values)
        return values

    def grouped(self, values: np.ndarray) -> np.ndarray:
        """The chain's states' probabilities (rows of ``values``) summed
        over each combination of the members up or down."""
        if self._tests is None:
            return values
        return self._chain.grouped(values)

    def tests(self, stop: float) -> dict:
        """The tests in ``(0, stop]``: each time's, as a list of
        ``ProofTest`` (with their times in the first period's, where the
        schedule repeats from)."""
        out: dict = {}
        if self._tests is None:
            return out
        cycles = int(np.floor(stop / self.period)) + 1
        for cycle in range(cycles):
            for test in self._tests:
                time = test.time + cycle * self.period
                if 0.0 < time <= stop:
                    out.setdefault(time, []).append(test)
        return out

    def tested(self, values: np.ndarray, tests) -> np.ndarray:
        """The chain's states' probabilities (rows of ``values``) after
        ``tests``."""
        out = np.array(values, dtype=float)
        for test in tests:
            out = np.vstack([self._chain.tested_at(row, test) for row in out])
        return out

    def knots(self, start: float, stop: float) -> np.ndarray:
        """Where its states bend or jump: the tests (every period), or the
        chain's times."""
        if self._tests is None:
            return self._chain.knots(start, stop)
        times = np.array([test.time for test in self._tests])
        first = int(np.floor(start / self.period))
        last = int(np.ceil(stop / self.period))
        every = (
            times[None, :] + self.period * np.arange(first, last + 1)[:, None]
        ).ravel()
        return np.concatenate(
            [[start], every[(every >= start) & (every <= stop)]]
        )

    breaks = knots


#: The most values (a probability for each node, or two, at each point)
#: the points of every combination of every group's states may take,
#: where an analysis splits each time by them (#218).
SPLIT_VALUES = 1 << 27


def check_split(times: int, counts, values: int, what: str, groups) -> None:
    """Refuse, before anything is built, to split each of ``times`` times
    by every combination of the groups' states (``counts``, each group's
    number of combinations) where the points would take more than
    ``SPLIT_VALUES`` values (``values`` at each point): ``what`` is the
    analysis that would."""
    points = times * math.prod(counts)
    if points * max(values, 1) <= SPLIT_VALUES:
        return
    raise NotImplementedError(
        f"{what} takes every combination of the common-cause groups' "
        f"members' states at once: {math.prod(counts):,} of them for "
        f"{[list(group.members) for group in groups]}, at each of "
        f"{times:,} times, too many to work out exactly (more than "
        f"{SPLIT_VALUES:,} values). The long-run values, the importance "
        "measures and the values over time condition on each group within "
        "its own module instead. Simulate the system with availability() "
        "or cost(), which take the groups in."
    )


def _check_size(size: int, members) -> None:
    if size > MAX_STATES:
        raise NotImplementedError(
            f"The common-cause group of {list(members)} has {size} joint "
            f"states, more than the {MAX_STATES} worked out exactly: split "
            "it, or leave it out."
        )


class _GroupsCurve:
    """The groups' states over time, as the analyses over a window follow
    them: when they settle into a constant or a period, and where they
    bend."""

    def __init__(self, groups: Sequence[OverTime]):
        from ._curves import _settling

        self.groups = list(groups)
        self.settle, self.period = _settling(self.groups)

    def knots(self, start: float, stop: float) -> np.ndarray:
        parts = [group.knots(start, stop) for group in self.groups]
        return np.concatenate(parts) if parts else np.array([start])

    breaks = knots


class GroupsSystem:
    """A system with common-cause groups over time, as the analyses
    over a window evaluate it at given times (the hook
    ``RepairableRBD._window_counts`` takes as ``crew``, as for repair
    crews' ``_chain_transient.CrewSystem``). Its nodes outside the groups
    are independent of the groups and of each other, and given by their
    availabilities at each time; each group's members' joint states come
    from its chain over time (``groups``, see ``OverTime``). At a time,
    each combination of each group's members up or down has its
    probability, and the system is worked out with those members up or
    down for certain and the other nodes at their availabilities then: its
    availability, and each other node's Birnbaum importance, are the
    averages over the combinations, as in the long run (see
    ``RepairableRBD._with_ccf_groups``). Each cause strikes at its rate
    (``causes``, by group, see ``causes``) and fails the members it names
    that are up: the system's failures by the groups' members are each
    cause's rate times the rise in its unavailability that the cause makes
    then."""

    def __init__(self, rbd, groups, causes, working, broken, method: str):
        self.rbd = rbd
        self.groups = list(groups)
        self.causes = list(causes)
        self.working, self.broken = set(working), set(broken)
        self.method = method
        self.served = [m for group in self.groups for m in group.members]
        #: What the system's failures by the groups' causes are counted
        #: under (see ``caused``).
        self.cause_keys = list(
            dict.fromkeys(
                (
                    group.members[struck[0]]
                    if len(struck) == 1
                    else tuple(group.members)
                )
                for group, by in zip(self.groups, self.causes)
                for struck, _ in by
            )
        )
        #: For the pieces of the integrals: the groups' knots and settling.
        self.curve = _GroupsCurve(self.groups)

    def _split(self, values: dict, x: np.ndarray, chances=None):
        """The points at the times ``x`` split by the groups' combinations:
        each node's probability at each point, each point's probability,
        its time's position in ``x``, and each group's combination there.
        ``chances`` gives a group's combinations' probabilities at the
        times (rows), where not None, in place of its own. For the
        capacity over time, which takes the nodes' joint states: it
        refuses where the points would be too many (see
        ``check_split``)."""
        from . import _windows

        size = len(x)
        probabilities = _windows._filled(
            self.rbd, values, size, self.working, self.broken
        )
        check_split(
            size,
            [len(group.down) for group in self.groups],
            len(probabilities),
            "The capacity over time",
            self.groups,
        )
        index, weights = np.arange(size), np.ones(size)
        combinations: List[np.ndarray] = []
        for number, group in enumerate(self.groups):
            given = None if chances is None else chances[number]
            table = group.probabilities(x) if given is None else given
            count = table.shape[1]
            mass = (weights[:, None] * table[index, :]).ravel()
            keep = mass > 0.0
            weights = mass[keep]
            points = len(index)
            probabilities = {
                node: np.repeat(
                    np.broadcast_to(np.asarray(v, dtype=float), (points,)),
                    count,
                )[keep]
                for node, v in probabilities.items()
            }
            combinations = [np.repeat(c, count)[keep] for c in combinations]
            combinations.append(np.tile(np.arange(count), points)[keep])
            for k, member in enumerate(group.members):
                down = np.tile(group.down[:, k], points)[keep]
                probabilities[member] = np.where(down, 0.0, 1.0)
            index = np.repeat(index, count)[keep]
        return probabilities, weights, index, combinations

    def _evaluation(self, values: dict, x: np.ndarray, chances=None):
        """The system at the times ``x`` (see
        ``RepairableRBD._ccf_tabled``): each node outside the groups up
        with the probability ``values[node]`` then, and each group's
        members in each of their combinations with its probability then
        (``chances[number]``, where given, in place of its chain's)."""
        from . import _ccf_groups, _windows

        size = len(x)
        p = {
            node: np.broadcast_to(np.asarray(v, dtype=float), (size,))
            for node, v in values.items()
        }
        p = _windows._filled(self.rbd, p, size, self.working, self.broken)
        tables = []
        for number, group in enumerate(self.groups):
            given = None if chances is None else chances[number]
            table = np.asarray(
                group.probabilities(x) if given is None else given,
                dtype=float,
            )
            tables.append(GroupStates(group.members, group.down, table))
            for k, member in enumerate(group.members):
                p[member] = table @ np.where(group.down[:, k], 0.0, 1.0)
        q = {node: 1.0 - value for node, value in p.items()}
        return _ccf_groups._ccf_tabled(self.rbd, p, q, tables)

    def probabilities(self, values: dict, x: np.ndarray):
        """The system's availability and unavailability at the times
        ``x``, each node outside the groups up with the probability
        ``values[node]`` (see ``evaluate``)."""
        x = np.asarray(x, dtype=float).ravel()
        up, down = self._evaluation(values, x).system()
        return self._clipped(up, down)

    def _clipped(self, up, down) -> Tuple[np.ndarray, np.ndarray]:
        up = up if self.method == "p" else 1.0 - down
        return np.clip(up, 0.0, 1.0), np.clip(down, 0.0, 1.0)

    def importance(self, values: dict, x: np.ndarray) -> dict:
        """Each node outside the groups' Birnbaum importance at the times
        ``x`` (see ``evaluate``)."""
        x = np.asarray(x, dtype=float).ravel()
        evaluation = self._evaluation(values, x)
        return self._importance(evaluation, values, *evaluation.system())

    @staticmethod
    def _importance(evaluation, values: dict, up, down) -> dict:
        """Each node of ``values``' Birnbaum importance: the difference
        the node held working or failed makes to the system, from
        whichever end keeps its precision (its unavailability, where the
        system is more often up)."""
        out = {}
        for node in values:
            up_1, down_1 = evaluation.system(hold={node: True})
            up_0, down_0 = evaluation.system(hold={node: False})
            out[node] = np.where(down <= up, down_0 - down_1, up_1 - up_0)
        return out

    def availability(self, values: dict, chances: list) -> np.ndarray:
        """The system's availability at each row, with each node outside
        the groups up with the probability ``values[node]`` and each
        group's members in each combination with the probabilities
        ``chances[number]`` (one row each)."""
        rows = len(chances[0]) if chances else 1
        evaluation = self._evaluation(values, np.zeros(rows), chances)
        return 1.0 - evaluation.system()[1]

    def given(self, values: dict, x: np.ndarray, number: int) -> np.ndarray:
        """The system's unavailability at the times ``x`` (rows) with the
        group ``number``'s members in each of their combinations (columns),
        the other groups' as at the times, and each node outside the groups
        up with the probability ``values[node]``."""
        x = np.asarray(x, dtype=float).ravel()
        evaluation = self._evaluation(values, x)
        return evaluation.combinations(number, self.groups[number].down)[1]

    def _causes(self, evaluation, down) -> dict:
        """The rate of the system's failures by each cause (see
        ``caused``), not clipped at 0."""
        out: dict = {}
        for group, causes in zip(self.groups, self.causes):
            for struck, cause in causes:
                hit = {group.members[position]: False for position in struck}
                rise = evaluation.system(hold=hit)[1] - down
                key = (
                    group.members[struck[0]]
                    if len(struck) == 1
                    else tuple(group.members)
                )
                out[key] = out.get(key, 0.0) + cause * rise
        return out

    def caused(self, values: dict, x: np.ndarray) -> dict:
        """The rate of the system's failures by each cause at the times
        ``x`` (see ``evaluate``): a member's own causes under its name, and
        each group's causes that strike more than one member under the
        tuple of its members."""
        x = np.asarray(x, dtype=float).ravel()
        evaluation = self._evaluation(values, x)
        down = evaluation.system()[1]
        return {
            key: np.maximum(v, 0.0)
            for key, v in self._causes(evaluation, down).items()
        }

    def evaluate(self, values: dict, x: np.ndarray):
        """At the times ``x``, with each node outside the groups up with the
        probability ``values[node]`` (an array over ``x``): each such
        node's Birnbaum importance (a dict), the system's availability and
        unavailability, the rate of its failures by the groups' causes,
        and each member's availability (a dict). Each group is conditioned
        on within the smallest module holding its members (see
        ``RepairableRBD._ccf_tabled``, #218)."""
        x = np.asarray(x, dtype=float).ravel()
        evaluation = self._evaluation(values, x)
        up, down = evaluation.system()
        importance = self._importance(evaluation, values, up, down)
        rate: Any = np.zeros(len(x))
        for value in self._causes(evaluation, down).values():
            rate = rate + value
        own = {member: evaluation.marginal(member) for member in self.served}
        up, down = self._clipped(up, down)
        return importance, up, down, np.maximum(rate, 0.0), own


class GroupsCurve:
    """A system with common-cause groups over time, from its nodes
    outside the groups (``curves``, as another RBD's) and its groups'
    states (``system``, a ``GroupsSystem``): its point availability, and
    its events counted as another RBD counts its nodes', the groups'
    members' failures by their causes (see ``GroupsSystem``)."""

    def __init__(self, rbd, curves: dict, system: GroupsSystem):
        from ._curves import _settling

        self.rbd = rbd
        self.curves = curves
        self.system = system
        #: As a node with capacities of another RBD: refused, as yet.
        self.capacity = None
        self.settle, self.period = _settling([*curves.values(), system.curve])

    def at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        values = {node: curve.at(flat) for node, curve in self.curves.items()}
        return self.system.probabilities(values, flat)[0].reshape(x.shape)

    def down_at(self, x: np.ndarray) -> np.ndarray:
        """The system's point unavailability at the times ``x``, worked
        out in its own right."""
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        values = {node: curve.at(flat) for node, curve in self.curves.items()}
        return self.system.probabilities(values, flat)[1].reshape(x.shape)

    def events(self, x: np.ndarray):
        """The system's expected failures and planned outages before each
        time ``x`` (see ``RepairableRBD._window_counts``)."""
        from . import _windows
        from ._point_availability import _events

        x = np.asarray(x, dtype=float)
        counts = _windows._window_counts(
            self.rbd,
            self.curves,
            x.ravel(),
            set(),
            set(),
            "p",
            crew=self.system,
        )
        return _events(
            counts["failures"].reshape(x.shape),
            counts["planned"].reshape(x.shape),
        )

    def atoms(self, stop: float):
        """The system's failures and planned outages at exact times before
        ``stop``: its nodes outside the groups' (the groups' members fail
        only at random times)."""
        from . import _windows

        return _windows._system_atoms(
            self.rbd, self.curves, stop, crew=self.system
        )

    def knots(self, start: float, stop: float) -> np.ndarray:
        parts = [self.system.curve.knots(start, stop)]
        parts += [curve.knots(start, stop) for curve in self.curves.values()]
        return np.concatenate(parts)

    def breaks(self, start: float, stop: float) -> np.ndarray:
        from ._point_availability import curve_breaks

        parts = [self.system.curve.breaks(start, stop)]
        parts += [curve_breaks(c, start, stop) for c in self.curves.values()]
        return np.concatenate(parts)

    def grids(self) -> list:
        from ._point_availability import curve_grids

        return [g for c in self.curves.values() for g in curve_grids(c)]
