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
from typing import Any, Hashable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

#: The most states a group's chain may have.
MAX_STATES = 20_000

UP, FOUND, MISSED = 0, 1, 2


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


def _revealed_flow(
    model,
    members: Sequence[Hashable],
    rate: float,
    repair: float,
    counts: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """The rates between the states of a group whose members' failures
    are revealed, each repaired at rate ``repair`` (independently: as many
    repairs at once as are needed), its states those of ``_Counted``
    (with ``counts`` copies of each member, by default one: each state
    then its own combination of the members up or down)."""
    n = len(members)
    space = _Counted(members, [1] * n if counts is None else counts, 2)
    size = space.size
    flow = np.zeros((size, size))
    source = np.arange(size)
    for number, (struck, cause) in enumerate(causes(model, members, rate)):
        target, times = space.strike(struck, FOUND, own=number < n)
        moves = target != source
        np.add.at(flow, (source[moves], target[moves]), cause * times[moves])
    for i in range(n):
        down = space.found[:, i] > 0
        fixed = space.found.copy()
        fixed[:, i] -= down.astype(np.int64)
        np.add.at(
            flow,
            (source[down], space.index(fixed, space.missed)[down]),
            repair * space.found[down, i],
        )
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
    (one that finds a failure it would otherwise miss)."""

    time: float
    member: int
    full: bool


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
        moves: List[Tuple[np.ndarray, np.ndarray, Any]] = []
        for number, (struck, cause) in enumerate(causes(model, members, rate)):
            for level, share in ((FOUND, coverage), (MISSED, 1.0 - coverage)):
                if share <= 0.0 or (level == MISSED and not partial):
                    continue
                target, times = space.strike(struck, level, own=number < n)
                moved = target != source
                moves.append(
                    (
                        source[moved],
                        target[moved],
                        cause * share * times[moved],
                    )
                )
        outflow = np.zeros(size)
        for begin, _, value in moves:
            np.add.at(outflow, begin, value)
        self.pace = pace = float(outflow.max()) if len(moves) else 0.0
        step = np.zeros((size, size))
        if pace > 0.0:
            for begin, end, value in moves:
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
        """``v exp(G dt)``, by uniformization, in steps short enough for
        ``exp(-x)`` not to underflow."""
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
) -> GroupStates:
    """The long-run states, at each of ``times`` (in ``[0, period)``), of a
    group whose members' failures are hidden, found by ``tests`` (each
    member's, in ``(0, period]``, the schedule repeating every
    ``period``), with ``coverage`` the chance that a test finds a cause's
    failures (a full test finds all), and ``counts`` copies of each member
    (see ``_Counted``), tested together."""
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

    @classmethod
    def revealed(
        cls, model, members, rate: float, repair: float, uniformized
    ) -> "OverTime":
        """A group whose failures are revealed (see ``revealed``), its
        chain followed by ``uniformized(generator, start, steady,
        vectors)`` (a ``_chain_transient.Uniformized``)."""
        flow = _revealed_flow(model, members, rate, repair)
        steady = gth(flow)
        generator = flow.copy()
        np.fill_diagonal(generator, 0.0)
        np.fill_diagonal(generator, -generator.sum(axis=1))
        size = len(flow)
        start = np.zeros(size)
        start[0] = 1.0
        chain = uniformized(generator, start, steady, np.identity(size))
        out = cls(
            members, _states(len(members), 2).astype(bool), chain.settle, None
        )
        out._chain = chain
        out._tests = None
        return out

    @classmethod
    def hidden(
        cls, model, members, rate: float, coverage: float, tests, period
    ) -> "OverTime":
        """A group whose failures are hidden (see ``hidden``)."""
        chain = _Hidden(model, members, rate, coverage)
        out = cls(members, chain.combinations, float(period), float(period))
        out._chain = chain
        out._tests = sorted(tests)
        # The states at the start of the second period, which repeats.
        out._settled = chain.follow(tests, np.array([float(period)]))[0]
        return out

    def probabilities(self, x) -> np.ndarray:
        """Each combination's probability (columns) at each time ``x``
        (rows): at a test's time, after it."""
        x = np.asarray(x, dtype=float).ravel()
        if self._tests is None:
            values = np.maximum(self._chain.values(x), 0.0)
        else:
            chain, period = self._chain, self.period
            values = np.empty((len(x), chain.size))
            early = x <= period
            if early.any():
                values[early] = chain.follow(self._tests, x[early])
            if (~early).any():
                phase = np.mod(x[~early] - period, period)
                values[~early] = chain.follow(
                    _shifted(self._tests, period),
                    period + phase,
                    self._settled,
                    period,
                )
            values = chain.grouped(np.maximum(values, 0.0))
        total = values.sum(axis=1, keepdims=True)
        return values / np.where(total > 0.0, total, 1.0)

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
        from .repairable_rbd import _settling

        self.groups = list(groups)
        self.settle, self.period = _settling(self.groups)

    def knots(self, start: float, stop: float) -> np.ndarray:
        parts = [group.knots(start, stop) for group in self.groups]
        return np.concatenate(parts) if parts else np.array([start])

    breaks = knots


class GroupsSystem:
    """A system with common-cause groups over time (#158), as the analyses
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
        #: For the pieces of the integrals: the groups' knots and settling.
        self.curve = _GroupsCurve(self.groups)

    def _split(self, values: dict, x: np.ndarray):
        """The points at the times ``x`` split by the groups' combinations:
        each node's probability at each point, each point's probability,
        and its time's position in ``x``."""
        size = len(x)
        probabilities = self.rbd._filled(
            values, size, self.working, self.broken
        )
        index, weights = np.arange(size), np.ones(size)
        for group in self.groups:
            chances = group.probabilities(x)
            count = chances.shape[1]
            mass = (weights[:, None] * chances[index, :]).ravel()
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
            for k, member in enumerate(group.members):
                down = np.tile(group.down[:, k], points)[keep]
                probabilities[member] = np.where(down, 0.0, 1.0)
            index = np.repeat(index, count)[keep]
        return probabilities, weights, index

    def evaluate(self, values: dict, x: np.ndarray):
        """At the times ``x``, with each node outside the groups up with the
        probability ``values[node]`` (an array over ``x``): each such
        node's Birnbaum importance (a dict), the system's availability and
        unavailability, the rate of its failures by the groups' causes,
        and each member's availability (a dict)."""
        x = np.asarray(x, dtype=float).ravel()
        size = len(x)
        probabilities, weights, index = self._split(values, x)

        def total(v) -> np.ndarray:
            v = np.broadcast_to(np.asarray(v, dtype=float), weights.shape)
            return np.bincount(index, weights * v, size)

        importance, works, fails, _, _ = self.rbd._importances(probabilities)
        down = total(fails)
        up = total(works) if self.method == "p" else 1.0 - down
        rate = np.zeros(size)
        if self.causes:
            base = self.rbd._system_unreliability(probabilities)
            for group, causes in zip(self.groups, self.causes):
                for struck, cause in causes:
                    hit = dict(probabilities)
                    for position in struck:
                        hit[group.members[position]] = np.zeros_like(weights)
                    rise = self.rbd._system_unreliability(hit) - base
                    rate += cause * total(rise)
        node_importance = {node: total(importance[node]) for node in values}
        own = {member: total(probabilities[member]) for member in self.served}
        return (
            node_importance,
            np.clip(up, 0.0, 1.0),
            np.clip(down, 0.0, 1.0),
            np.maximum(rate, 0.0),
            own,
        )


class GroupsCurve:
    """A system with common-cause groups over time (#158), from its nodes
    outside the groups (``curves``, as another RBD's) and its groups'
    states (``system``, a ``GroupsSystem``): its point availability, and
    its events counted as another RBD counts its nodes', the groups'
    members' failures by their causes (see ``GroupsSystem``)."""

    def __init__(self, rbd, curves: dict, system: GroupsSystem):
        from .repairable_rbd import _settling

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
        return self.system.evaluate(values, flat)[1].reshape(x.shape)

    def events(self, x: np.ndarray):
        """The system's expected failures and planned outages before each
        time ``x`` (see ``RepairableRBD._window_counts``)."""
        from ._point_availability import _events

        x = np.asarray(x, dtype=float)
        counts = self.rbd._window_counts(
            self.curves, x.ravel(), set(), set(), "p", crew=self.system
        )
        return _events(
            counts["failures"].reshape(x.shape),
            counts["planned"].reshape(x.shape),
        )

    def atoms(self, stop: float):
        """The system's failures and planned outages at exact times before
        ``stop``: its nodes outside the groups' (the groups' members fail
        only at random times)."""
        return self.rbd._system_atoms(self.curves, stop, crew=self.system)

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
