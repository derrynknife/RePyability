"""Common-cause groups in a ``RepairableRBD`` (#136): which of a group's
members are down together, in the long run.

A group's members are identical units with a constant failure rate. Its
model (a ``BetaFactor``, ``MGL`` or ``AlphaFactor``, see ``ccf.py``)
splits that rate
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

import bisect
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

#: A member's level when it starts in a repair under way (#106) whose end
#: is not at a rate: the repair it is in at the start, on its own clock.
LEFT = 6

#: What happens at a ``ProofTest``'s time: the member's test, or the end of
#: one of its tests (``TESTED``) or repairs (``REPAIRED``) of a fixed length,
#: or of the repair of a fixed length under way at the start (``ENDED``).
TEST, TESTED, REPAIRED, ENDED = 0, 1, 2, 3


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
    time (``fixed``, 0 for none), an exponential one (at ``rate``), or
    one of any other length (``model``, a surpyval model, #106)."""

    fixed: float = 0.0
    rate: Optional[float] = None
    model: Any = None

    @property
    def takes_time(self) -> bool:
        return self.fixed > 0.0 or self.rate is not None or self.general

    @property
    def general(self) -> bool:
        """Whether it is of a length neither fixed nor exponential."""
        return self.model is not None

    @property
    def random(self) -> bool:
        """Whether its length is drawn (exponential, or general)."""
        return self.rate is not None or self.general

    def sf(self, s) -> np.ndarray:
        """``P(length > s)``, 1 before 0."""
        s = np.asarray(s, dtype=float)
        if self.general:
            with np.errstate(divide="ignore", invalid="ignore"):
                values = np.asarray(
                    self.model.sf(np.maximum(s, 0.0).ravel()), dtype=float
                ).reshape(s.shape)
            return np.where(s < 0.0, 1.0, np.clip(values, 0.0, 1.0))
        if self.rate is not None:
            return np.exp(-self.rate * np.maximum(s, 0.0))
        return (s < self.fixed).astype(float)

    def quantiles(self, p) -> np.ndarray:
        """The lengths below which it is with the probabilities ``p``."""
        p = np.asarray(p, dtype=float)
        if self.general:
            return np.asarray(self.model.qf(p.ravel()), dtype=float).reshape(
                p.shape
            )
        if self.rate is not None:
            return -np.log1p(-p) / self.rate
        return np.full(p.shape, self.fixed)

    def tail(self) -> float:
        """A length it is all but surely over by (but for a chance of
        ``TAIL``)."""
        return float(self.quantiles(np.array([1.0 - TAIL]))[0])


#: The chance a test or repair of a random length may still run, past
#: its ``tail``: what the chain leaves out where it takes their ends from
#: the time since the test that started them (#106).
TAIL = 1e-12


class Timing(NamedTuple):
    """How long a member's tests and repairs take (see ``Duration``)."""

    test: Duration = Duration()
    repair: Duration = Duration()

    @property
    def takes_time(self) -> bool:
        return self.test.takes_time or self.repair.takes_time

    @property
    def repair_jumps(self) -> bool:
        """Whether its repair ends a fixed time after its test: a repair
        of a fixed length after a test of none or a fixed one."""
        return self.repair.fixed > 0.0 and not self.test.random

    @property
    def varies(self) -> Tuple[bool, bool]:
        """Whether its test, and its repair, end at a rate that varies
        with the time since the test (#106): a test of a general length;
        a repair of a general length, or of a fixed one after a test of a
        random length (whose end is then no fixed time after the test)."""
        repair = self.repair.general or (
            self.repair.fixed > 0.0 and self.test.random
        )
        return self.test.general, repair


#: The chances at which a member's test and repair of a general length are
#: cut into pieces of a constant rate (#106), and the evenly spaced pieces
#: laid over them.
_PIECE_CHANCES = np.unique(
    np.concatenate(
        [
            10.0 ** -np.arange(12.0, 1.0, -1.0),
            np.linspace(0.0, 1.0, 25)[1:-1],
            1.0 - 10.0 ** -np.arange(2.0, 13.0),
        ]
    )
)
_EVEN_PIECES = 16
#: The cells of equal chance a test of a random length is split into, to
#: add a repair of a general length to it.
_SUM_CELLS = 4096


def _pieces(knots: np.ndarray, end: float) -> np.ndarray:
    """``knots`` from 0 to ``end``, with ``_EVEN_PIECES`` even pieces
    laid over them."""
    knots = np.concatenate(
        [[0.0, end], knots, np.linspace(0.0, end, _EVEN_PIECES + 1)]
    )
    knots = knots[np.isfinite(knots) & (knots >= 0.0) & (knots <= end)]
    knots = np.unique(knots)
    # Pieces too short for their rates to keep their precision (a
    # difference of logarithms over them) are merged into the next.
    keep = np.concatenate([[True], np.diff(knots) > 1e-7 * end])
    keep[-1] = True
    knots = knots[keep]
    if len(knots) > 2 and knots[-1] - knots[-2] <= 1e-7 * end:
        knots = np.delete(knots, -2)
    return knots


#: How far a length's chance of running may be from a rate held constant
#: on a piece, within it (see ``_refined``), and the most knots.
_PIECE_ERROR = 1e-5
_MAX_KNOTS = 2000


def _refined(knots: np.ndarray, sf, end: float) -> np.ndarray:
    """``knots`` with each piece halved, again and again, until a rate
    held constant on it (see ``_hazard_rates``) follows a length of
    survival function ``sf`` within it to ``_PIECE_ERROR``: its halves'
    rates differ by so little that, over the piece, the chance of
    running they give at its middle is off by less (``S(a)`` times a
    quarter of the piece's width times their difference). The values at
    times between the knots come from the rates held constant."""
    for _ in range(30):
        a, b = knots[:-1], knots[1:]
        m = 0.5 * (a + b)
        with np.errstate(divide="ignore", invalid="ignore"):
            Sa, Sm, Sb = (np.clip(sf(x), 0.0, 1.0) for x in (a, m, b))
            Ha, Hm, Hb = (-np.log(x) for x in (Sa, Sm, Sb))
            left, right = (Hm - Ha) / (m - a), (Hb - Hm) / (b - m)
            off = Sa * np.abs(left - right) * (b - a) / 4.0
        split = (off > _PIECE_ERROR) & np.isfinite(off) & (b - a > 1e-7 * end)
        if not split.any() or len(knots) + split.sum() > _MAX_KNOTS:
            break
        knots = np.sort(np.concatenate([knots, m[split]]))
    return knots


def _hazard_rates(sf, knots: np.ndarray) -> np.ndarray:
    """The constant rate on each piece between ``knots`` that ends a
    length of survival function ``sf`` on its own: ``(H(b) - H(a)) / (b
    - a)``, ``H = -log sf``, so that its chance of running is exact at the
    knots. Where ``sf`` has reached 0, the last finite rate before."""
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.log(np.clip(sf(knots), 0.0, 1.0))
        rates = np.diff(H) / np.diff(knots)
    last = 0.0
    out = np.empty_like(rates)
    for k, rate in enumerate(rates):
        if np.isfinite(rate):
            last = max(float(rate), 0.0)
        out[k] = last
    return out


def _repair_rate(testing, repair, r1, dt, target) -> float:
    """The constant rate on a piece of length ``dt`` at which a repair
    ends so that a member alone, in its test with the chance ``testing``
    (which ends at the rate ``r1`` into the repair) and in its repair with
    the chance ``repair`` at the piece's start, is in its repair with the
    chance ``target`` at its end."""
    if repair <= 0.0 and testing * r1 <= 0.0:
        return 0.0
    if target <= 0.0:
        return 50.0 / dt
    if testing * r1 <= 0.0:
        return max(math.log(repair / target) / dt, 0.0)
    e1 = math.exp(-r1 * dt)

    def left(r2: float) -> float:
        e2 = math.exp(-r2 * dt)
        gap = (r1 - r2) * dt
        if abs(gap) > 1e-6:
            moved = (e2 - e1) / (r1 - r2)
        else:
            moved = dt * math.exp(-0.5 * (r1 + r2) * dt)
        return repair * e2 + testing * r1 * moved - target

    if left(0.0) <= 0.0:
        return 0.0
    hi = 1.0 / dt
    while left(hi) > 0.0:
        hi *= 4.0
        if hi * dt > 50.0:
            return hi
    from scipy.optimize import brentq

    return float(brentq(left, 0.0, hi, xtol=1e-300, rtol=1e-13))


class _Episode:
    """A member's test and repair after its test, where one ends at a
    rate that varies with the time ``s`` since the test (#106, see
    ``Timing.varies``): followed on pieces of ``s`` (``knots``), each with
    a constant rate for the test's end (``test``) and the repair's
    (``repair``). The member is apart from the causes then, so its own
    path through them is that of the member alone: each test rate keeps
    the chance that its test is running exact at the knots, and each
    repair rate the chance that a member found failed is in its repair
    (the test and repair's sum taken on a grid where both are random).
    Past the last knot (where either runs with a chance below ``TAIL``),
    the last piece's rates."""

    def __init__(self, timing: Timing):
        test, repair = timing.test, timing.repair
        test_varies, repair_varies = timing.varies
        chances = _PIECE_CHANCES
        parts = [np.array([test.fixed])]
        shift = test.fixed
        if test.random:
            parts.append(test.quantiles(chances))
            shift = float(test.quantiles(np.array([0.5]))[0])
        if repair.random:
            parts.append(shift + repair.quantiles(chances))
        elif repair.fixed > 0.0 and test.random:
            parts.append(repair.fixed + test.quantiles(chances))
        end = test.tail() + (repair.tail() if repair_varies else 0.0)
        knots = _pieces(np.concatenate(parts), end)
        if test.random:
            knots = _refined(knots, test.sf, end)
        if repair_varies:
            knots = _refined(
                knots, lambda s: self._running(test, repair, s), end
            )
        self.knots = knots
        widths = np.diff(knots)
        pieces = len(widths)
        self.test = np.zeros(pieces)
        if test_varies:
            self.test = _hazard_rates(test.sf, knots)
        self.repair = np.zeros(pieces)
        if repair_varies:
            testing = test.sf(knots)
            running = np.clip(self._running(test, repair, knots), testing, 1)
            in_repair = running - testing
            r1 = self.test if test_varies else np.full(pieces, test.rate or 0)
            self.repair = np.array(
                [
                    _repair_rate(
                        float(testing[k]),
                        float(in_repair[k]),
                        float(r1[k]),
                        float(widths[k]),
                        float(in_repair[k + 1]),
                    )
                    for k in range(pieces)
                ]
            )

    @staticmethod
    def _running(test: Duration, repair: Duration, s: np.ndarray):
        """``P(test + repair > s)``: exact where either is fixed (or
        none); otherwise ``1 - E[F_R(s - T)]`` by the midpoint rule on
        ``_SUM_CELLS`` cells of equal chance of the test."""
        if not test.random:
            return repair.sf(s - test.fixed)
        if not repair.random:
            return test.sf(s - repair.fixed)
        edges = np.linspace(0.0, 1.0, _SUM_CELLS + 1)
        middles = test.quantiles(0.5 * (edges[1:] + edges[:-1]))
        out = np.empty(len(s))
        for start in range(0, len(s), 64):
            chunk = s[start : start + 64]
            done = 1.0 - repair.sf(chunk[:, None] - middles[None, :])
            out[start : start + 64] = 1.0 - done.mean(axis=1)
        return np.clip(out, 0.0, 1.0)

    def rates(self, s: float) -> Tuple[float, float]:
        """The test's and the repair's rate at ``s`` after the test."""
        k = int(np.searchsorted(self.knots, s, side="right")) - 1
        k = min(max(k, 0), len(self.test) - 1)
        return float(self.test[k]), float(self.repair[k])


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
        self.space = space

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

    def _placed(self, member: int, level: int) -> np.ndarray:
        """The state each state goes to with ``member`` put at
        ``level``."""
        space = self.space
        found, missed = space.found.copy(), space.missed.copy()
        found[:, member] = int(level == FOUND)
        missed[:, member] = int(level == MISSED)
        return space.index(found, missed)

    def placed(self, v: np.ndarray, member: int, level: int) -> np.ndarray:
        """``v`` with ``member`` put at ``level``, whatever its level
        was."""
        target = self._placed(member, level)
        return np.bincount(target, weights=v, minlength=self.size)

    def started(self, known: Sequence[float], levels: dict) -> np.ndarray:
        """The states at 0 from the members' states (#106): each member
        ``i`` known to be up ``known[i]`` before 0 (when it was last
        tested and found working, or put into service), and the members
        in ``levels`` at their level at 0. Each cause strikes the members
        up since they were known up, alike, as from 0 on: from every
        member up at the earliest, the members are put up as they come to
        be known up, the causes followed between (what a member's level
        was before it was known up does not change the others': a cause
        strikes each up member whatever the others' levels are)."""
        v = self.start()
        now = -max(list(known) + [0.0])
        for member in sorted(range(len(known)), key=lambda i: -known[i]):
            v = self._uniformized(v, -known[member] - now)
            now = -known[member]
            v = self.placed(v, member, UP)
        v = self._uniformized(v, -now)
        for member, level in levels.items():
            v = self.placed(v, member, level)
        return v

    def _advance(self, v: np.ndarray, start: float, stop: float):
        """``v`` followed from ``start`` to ``stop``, with no test
        between."""
        return self.evolve(v, stop - start)

    def _advance_many(self, v: np.ndarray, start: float, times: np.ndarray):
        """``v`` followed from ``start`` to each of ``times`` (rows;
        increasing, none before ``start``), with no test between."""
        out = np.empty((len(times), self.size))
        now = start
        for row, t in enumerate(times):
            v = self._advance(v, now, float(t))
            now = float(t)
            out[row] = v
        return out

    def follow(self, tests, times: np.ndarray, v=None, now: float = 0.0):
        """The state at each of ``times`` (rows), from ``v`` (every member
        up) at ``now``, through the ``tests`` in order (tests at a time come
        before it)."""
        order = sorted(tests)
        v = self.start() if v is None else v
        times = np.asarray(times, dtype=float)
        out = np.empty((len(times), self.size))
        rows = np.argsort(times, kind="stable")
        upcoming, k = 0, 0
        while k < len(rows):
            t = float(times[rows[k]])
            while upcoming < len(order) and order[upcoming].time <= t:
                test = order[upcoming]
                if test.time >= now:
                    v = self._advance(v, now, test.time)
                    v = self.tested_at(v, test)
                    now = test.time
                upcoming += 1
            # The times before the next test, followed together.
            stop = order[upcoming].time if upcoming < len(order) else np.inf
            end = k + int(np.searchsorted(times[rows[k:]], stop, side="left"))
            batch = rows[k:end]
            values = self._advance_many(v, now, times[batch])
            out[batch] = values
            v, now, k = values[-1], float(times[batch[-1]]), end
        return out

    def grouped(self, states: np.ndarray) -> np.ndarray:
        """Each row of ``states`` summed over each combination of the
        members up or down."""
        out = np.zeros((len(states), len(self.combinations)))
        for j in range(len(self.combinations)):
            out[:, j] = states[:, self.combination == j].sum(axis=1)
        return out


class Left(NamedTuple):
    """A repair under way at the start (#106): how long the member's
    repairs take (``repair``), and how long this one has run
    (``down_for``)."""

    repair: Duration
    down_for: float

    def sf(self, s) -> np.ndarray:
        """``P(what is left of it > s)``."""
        s = np.asarray(s, dtype=float)
        known = float(self.repair.sf(np.array([self.down_for]))[0])
        return np.clip(self.repair.sf(self.down_for + s) / known, 0.0, 1.0)

    @property
    def level(self) -> int:
        """The member's level at the start: under repair, where it ends
        at a rate (``REPAIR``); otherwise in the repair under way
        (``LEFT``)."""
        return REPAIR if self.repair.rate is not None else LEFT

    @property
    def end(self) -> Optional[float]:
        """When a repair of a fixed length ends; None for another."""
        if self.repair.random:
            return None
        return self.repair.fixed - self.down_for

    def pieces(self) -> Tuple[np.ndarray, np.ndarray]:
        """For a repair of a general length: the knots of the time from
        the start, and the rate on each piece between them, that keep its
        chance of running exact at the knots (see ``_hazard_rates``)."""
        known = float(self.repair.sf(np.array([self.down_for]))[0])
        chances = 1.0 - (1.0 - _PIECE_CHANCES) * known
        knots = self.repair.quantiles(chances) - self.down_for
        end = float(
            self.repair.quantiles(np.array([1.0 - TAIL * known]))[0]
            - self.down_for
        )
        knots = _pieces(knots, end)
        return knots, _hazard_rates(self.sf, knots)


class _Timed(_Hidden):
    """``_Hidden`` for members whose tests or repairs take time (#220,
    see the module's notes), one copy of each member: each member's level
    is one of ``levels`` (``UP``, ``FOUND``, ``MISSED`` where tests can
    miss a failure, ``OFF`` and ``TESTING`` where tests take time,
    ``REPAIR`` where repairs do, ``LEFT`` for a repair under way at the
    start), a state one level for each member (see ``_states``). Causes
    strike the members that are up; a test or repair of an exponential
    length ends at its rate; one of a fixed length ends at a
    ``ProofTest`` of its ``kind``, a fixed time after the test; and one
    of any other length (#106) at a rate that varies with the time since
    the member's last test (``_Episode``): the chain is then followed
    piece by piece (``_advance``), each piece's rates constant. A
    member in ``lefts`` may start in a repair under way (``Left``): one
    of a fixed length ends at an ``ENDED`` event, one of a general length
    at a rate that varies with the time from 0."""

    def __init__(
        self,
        model,
        members,
        rate: float,
        coverage: float,
        timings,
        lefts: Optional[Dict[int, "Left"]] = None,
    ):
        n = len(members)
        lefts = {} if lefts is None else dict(lefts)
        self.partial = partial = coverage < 1.0
        tests_take = any(t.test.takes_time for t in timings)
        repairs_take = any(t.repair.takes_time for t in timings)
        ladder = [UP, FOUND] + ([MISSED] if partial else [])
        ladder += [OFF, TESTING] if tests_take else []
        ladder += [REPAIR] if repairs_take else []
        ladder += (
            [LEFT]
            if any(left.level == LEFT for left in lefts.values())
            else []
        )
        #: The levels a member may be at.
        self.ladder = ladder
        self.levels = count = len(ladder)
        self.code = code = {level: k for k, level in enumerate(ladder)}
        self.states = states = _states(n, count)
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

        #: The continuous moves at constant rates, labelled (see ``Move``).
        self.moves: List[Move] = []
        #: The moves at rates that vary (#106): whose, which end (of the
        #: ``"test"``, the ``"repair"``, or the repair under way at the
        #: start, ``"left"``), and their sources and targets.
        self.varying: List[Tuple[int, str, np.ndarray, np.ndarray]] = []
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
        #: Each member's test and repair where one varies (see
        #: ``_Episode``), and the pieces of a general repair under way at
        #: the start.
        self.episodes: Dict[int, _Episode] = {}
        self.left_pieces: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for i, timing in enumerate(timings):
            after = REPAIR if timing.repair.takes_time else UP
            ends = {
                "test": {OFF: UP, TESTING: after},
                "repair": {REPAIR: UP},
            }
            varies = dict(zip(("test", "repair"), timing.varies))
            if any(varies.values()):
                self.episodes[i] = _Episode(timing)
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
                elif varies[which]:
                    for old, new in ends[which].items():
                        target = moving(i, {old: new})
                        hit = target != source
                        if hit.any():
                            self.varying.append(
                                (i, which, source[hit], target[hit])
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
            left = lefts.get(i)
            if left is not None and left.level == LEFT:
                target = moving(i, {LEFT: UP})
                if left.end is not None:
                    self.jumps[(i, ENDED, False)] = target
                    self.jumps[(i, ENDED, True)] = target
                else:
                    self.left_pieces[i] = left.pieces()
                    hit = target != source
                    self.varying.append((i, "left", source[hit], target[hit]))
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
        self._outflow = outflow
        self.combinations = _states(n, 2).astype(bool)
        self.combination = _index((states != code[UP]).astype(np.int8), 2)
        #: Each member's tests' times in a period of the calendar, and the
        #: period (see ``calendar``): where the time since a member's last
        #: test comes from.
        self.calendar: Optional[Tuple[List[np.ndarray], float]] = None

    def tested_at(self, v: np.ndarray, test: "ProofTest") -> np.ndarray:
        """``v`` after the member's test, or the end of its test or repair,
        ``test``."""
        full = test.full or not self.partial or test.kind != TEST
        target = self.jumps.get((test.member, test.kind, full))
        if target is None:
            return v
        return np.bincount(target, weights=v, minlength=self.size)

    def _placed(self, member: int, level: int) -> np.ndarray:
        target = self.states.copy()
        target[:, member] = self.code[level]
        return _index(target, self.levels)

    def set_calendar(self, tests, period: float) -> None:
        """Take each member's tests' times from ``tests`` (``ProofTest``,
        one period of the calendar, repeating every ``period``)."""
        members = len(self.states[0]) if self.size else 0
        times: List[list] = [[] for _ in range(members)]
        for test in tests:
            if test.kind == TEST:
                times[test.member].append(float(test.time))
        self.calendar = ([np.array(sorted(t)) for t in times], float(period))

    def _last_tests(self, t: float) -> np.ndarray:
        """Each member's last test at or before ``t`` (-inf for none)."""
        assert self.calendar is not None, "the chain needs its calendar"
        times, period = self.calendar
        out = np.full(len(times), -np.inf)
        at = t + 1e-9 * period
        cycle = math.floor(at / period)
        within = at - cycle * period
        for i, mine in enumerate(times):
            if len(mine):
                k = bisect.bisect_right(mine.tolist(), within)
                if k:
                    out[i] = float(mine[k - 1]) + cycle * period
                else:
                    out[i] = float(mine[-1]) + (cycle - 1) * period
        return out

    def _advance(self, v: np.ndarray, start: float, stop: float):
        """``v`` followed from ``start`` to ``stop``, with no test
        between: at once where every rate is constant (see ``evolve``);
        otherwise piece by piece where a rate varies (#106), at the
        knots of each member's test and repair after its last test, and
        of a repair under way from 0, each piece by uniformization with
        its rates."""
        if not self.varying:
            return self.evolve(v, stop - start)
        if not stop > start:
            return v
        last, edges = self._edges(start, stop)
        for a, b in zip(edges[:-1], edges[1:]):
            v = self._piece(v, float(b - a), self._rates(last, a, b))
        return v

    def _edges(self, start: float, stop: float):
        """Each member's last test at ``start``, and the edges of the
        pieces from ``start`` to ``stop``: the knots of each member's
        test and repair after its last test, and of a repair under way
        from 0."""
        last = self._last_tests(start)
        cuts = [np.array([start, stop])]
        for i, episode in self.episodes.items():
            if np.isfinite(last[i]):
                cuts.append(last[i] + episode.knots)
        for knots, _ in self.left_pieces.values():
            cuts.append(knots)
        edges = np.unique(np.concatenate(cuts))
        return last, edges[(edges >= start) & (edges <= stop)]

    def _rates(self, last: np.ndarray, a: float, b: float) -> List[float]:
        """The varying moves' rates on the piece from ``a`` to ``b``."""
        middle = 0.5 * (a + b)
        rates = []
        for i, which, _, _ in self.varying:
            if which == "left":
                knots, values = self.left_pieces[i]
                k = int(np.searchsorted(knots, middle, side="right")) - 1
                rates.append(float(values[min(max(k, 0), len(values) - 1)]))
            elif np.isfinite(last[i]):
                test, repair = self.episodes[i].rates(middle - last[i])
                rates.append(test if which == "test" else repair)
            else:
                rates.append(0.0)
        return rates

    def _advance_many(self, v: np.ndarray, start: float, times: np.ndarray):
        """``v`` followed from ``start`` to each of ``times`` (see
        ``_Hidden._advance_many``): where rates vary, piece by piece, the
        times within a piece all from its start by uniformization, whose
        terms (the start's state times the step's powers) they share, each
        with its own Poisson weights."""
        if not self.varying or not len(times):
            return super()._advance_many(v, start, times)
        out = np.empty((len(times), self.size))
        last, edges = self._edges(start, float(times[-1]))
        now, row = start, 0
        for b in list(edges[1:]):
            inside = int(np.searchsorted(times, b, side="right"))
            rates = self._rates(last, now, b)
            if inside > row:
                out[row:inside] = self._spread(
                    v, times[row:inside] - now, rates
                )
                row = inside
            v = self._piece(v, float(b - now), rates)
            now = b
        while row < len(times):  # at ``start`` itself
            out[row] = v
            row += 1
        return out

    def _spread(self, v: np.ndarray, gaps: np.ndarray, rates) -> np.ndarray:
        """``v exp(G gap)`` for each of ``gaps`` (rows; ``G`` as for
        ``_piece``), by uniformization sharing its terms: where the
        longest gap is short next to the uniformization's rate, else one
        by one."""
        step, pace = self._step_with(rates)
        if not pace > 0.0:
            return np.repeat(v[None, :], len(gaps), axis=0)
        x = pace * gaps
        if float(x.max()) > 8.0:
            return np.array([self._piece(v, float(g), rates) for g in gaps])
        top = float(x.max())
        weight = np.exp(-x)
        out = weight[:, None] * v[None, :]
        term, k, peak = v, 0, math.exp(-top)
        while True:
            k += 1
            term = term @ step
            weight = weight * x / k
            peak *= top / k
            out += weight[:, None] * term[None, :]
            if k > top and peak < 1e-34:
                break
        return out

    def _piece(self, v: np.ndarray, dt: float, rates: List[float]):
        """``v exp(G dt)``, ``G`` the constant moves' rates and the
        varying moves' at ``rates``, by uniformization: worked out as a
        matrix and kept once the same piece comes again (as pieces
        repeat, period after period, and past each test and repair's
        knots, where the rates no longer change), as ``evolve`` keeps
        its steps."""
        if not dt > 0.0:
            return v
        key = (float(dt), tuple(rates))
        kept = self.__dict__.setdefault("_pieces", {})
        found = kept.get(key)
        if found is None:
            if len(kept) * 8 * self.size**2 >= _TRANSITION_BYTES:
                kept.clear()
            kept[key] = False
            return self._uniformized_with(v, dt, rates)
        if found is False:
            found = kept[key] = self._uniformized_with(
                np.eye(self.size), dt, rates
            )
        return v @ found

    def _step_with(self, rates) -> Tuple[np.ndarray, float]:
        """A step of the chain uniformized at its fastest outflow, ``G``
        the constant moves' rates and the varying moves' at ``rates``:
        the step, and that rate."""
        key = tuple(rates)
        kept = self.__dict__.setdefault("_steps", {})
        found = kept.get(key)
        if found is not None:
            return found
        outflow = self._outflow.copy()
        for (_, _, sources, _), value in zip(self.varying, rates):
            if value > 0.0:
                np.add.at(outflow, sources, value)
        pace = float(outflow.max()) if self.size else 0.0
        step = self.step * (self.pace / pace) if pace > 0.0 else self.step
        if pace > 0.0:
            step = step.copy()
            for (_, _, sources, targets), value in zip(self.varying, rates):
                if value > 0.0:
                    np.add.at(step, (sources, targets), value / pace)
            np.fill_diagonal(step, 1.0 - outflow / pace)
        if len(kept) * 8 * self.size**2 >= _TRANSITION_BYTES:
            kept.clear()
        kept[key] = (step, pace)
        return step, pace

    def _uniformized_with(self, v: np.ndarray, dt: float, rates):
        """``v exp(G dt)`` (see ``_piece``; ``v`` a vector, or a
        matrix's rows): the whole steps of ``8 / pace`` (short enough for
        ``exp(-x)`` not to underflow) through their matrices for 1, 2,
        4, ... steps, kept by rates (each the square of the one before:
        sums of positive terms), and what is left by uniformization's
        series."""
        step, pace = self._step_with(rates)
        if not pace > 0.0:
            return v
        whole = 8.0 / pace
        count = int(dt // whole)
        powers = None
        if count:
            kept = self.__dict__.setdefault("_powers", {})
            held = sum(len(p) for p in kept.values())
            if held * 8 * self.size**2 >= _TRANSITION_BYTES:
                kept.clear()
            powers = kept.setdefault(tuple(rates), [])
            if not powers:
                powers.append(_series(np.eye(self.size), step, 8.0))
        k = 0
        while count:
            assert powers is not None
            if k == len(powers):
                powers.append(powers[-1] @ powers[-1])
            if count & 1:
                v = v @ powers[k]
            count >>= 1
            k += 1
        left = dt - whole * int(dt // whole)
        return _series(v, step, pace * left) if left > 0.0 else v


def _series(v: np.ndarray, step: np.ndarray, x: float) -> np.ndarray:
    """``v`` times uniformization's series for a time of ``x`` steps of
    the uniformized chain ``step`` (``x`` at most 8, so that ``exp(-x)``
    does not underflow), to a term below 1e-34."""
    weight = math.exp(-x)
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
    return out


def timed_events(
    tests, timings, period: float, lefts: Optional[Dict[int, Left]] = None
) -> List[ProofTest]:
    """``tests`` (each member's, in ``(0, period]``) with the ends of the
    tests and repairs of a fixed length after each (see ``Timing``), each
    in ``(0, period]`` as the schedule repeats: an end past the period is
    the previous period's test's, at the start of this one (where, from
    all up at 0, it ends nothing)."""
    out = [test._replace(kind=TEST) for test in tests]
    for test in tests:
        timing = timings[test.member]
        ends = []
        if not timing.test.random and timing.test.fixed > 0.0:
            ends.append((timing.test.fixed, TESTED))
        if timing.repair_jumps:
            ends.append((timing.test.fixed + timing.repair.fixed, REPAIRED))
        for length, kind in ends:
            time = test.time + length
            time = time - period * math.floor((time - 1e-12 * period) / period)
            out.append(ProofTest(time, test.member, True, kind))
    return sorted(out)


def started_ends(lefts: Optional[Dict[int, Left]]) -> List[ProofTest]:
    """The ends of the repairs of a fixed length under way at the start
    (#106): once each, not every period."""
    return sorted(
        ProofTest(left.end, member, True, ENDED)
        for member, left in (lefts or {}).items()
        if left.level == LEFT and left.end is not None
    )


def _scale(duration: Duration) -> float:
    """The time scale of a test or repair of a random length: its mean
    if exponential, its median otherwise (#106)."""
    if duration.rate is not None:
        return 1.0 / duration.rate
    return float(duration.quantiles(np.array([0.5]))[0])


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
        timed.set_calendar(events, period)
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


class Start(NamedTuple):
    """Where a group's members start (#106), for its chain over time:
    in its long run (``stationary``), or each member ``i`` known to be up
    ``known[i]`` before 0 (when last tested and found working, or put into
    service: it may have failed since, unseen) but those in ``down``,
    each down at 0: in a repair under way (a ``Left``) where its failures
    are hidden, and (``True``) in a repair at a rate where they are
    revealed."""

    known: Tuple[float, ...] = ()
    down: Dict[int, Any] = {}
    stationary: bool = False


class OverTime:
    """A group's members' joint states over time from every member up at
    0, or from their states (``Start``, #106) (see ``probabilities``):
    from its chain followed by uniformization (``revealed`` failures), or
    (``hidden``) through its tests' first periods, by the end of which
    every member has had a full test and the states repeat the long run's
    every period."""

    def __init__(self, members, combinations, settle: float, period):
        self.members = tuple(members)
        #: Which members are down in each combination (see GroupStates).
        self.down = combinations
        self.settle = settle
        self.period = period
        self._chain: Any = None
        self._tests: Optional[list] = None
        self._timings: Optional[Sequence[Timing]] = None
        self._settled: Any = None
        #: The states at 0, and the events that happen once (the ends of
        #: repairs of a fixed length under way at 0).
        self._start: Any = None
        self._once: List[ProofTest] = []
        #: The chain's moves, labelled (see ``Move``), and the combination
        #: of the members up or down of each of its states (#199).
        self.moves: List[Move] = []
        self.combination = np.arange(len(combinations))

    @classmethod
    def revealed(
        cls,
        model,
        members,
        rate: float,
        repair: float,
        uniformized,
        start: Optional[Start] = None,
    ) -> "OverTime":
        """A group whose failures are revealed (see ``revealed``), its
        chain followed by ``uniformized(generator, start, steady,
        vectors)`` (a ``_chain_transient.Uniformized``): from every
        member up, or from ``start`` (its long run, or the members in
        ``start.down`` in a repair)."""
        size, moves = _revealed_moves(model, members, rate, repair)
        flow = np.zeros((size, size))
        for _, begin, end, value in moves:
            np.add.at(flow, (begin, end), value)
        steady = gth(flow)
        generator = flow.copy()
        np.fill_diagonal(generator, 0.0)
        np.fill_diagonal(generator, -generator.sum(axis=1))
        if start is not None and start.stationary:
            initial = steady
        else:
            initial = np.zeros(size)
            down = () if start is None else start.down
            initial[sum(2**i for i in down)] = 1.0
        chain = uniformized(generator, initial, steady, np.identity(size))
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
        start: Optional[Start] = None,
    ) -> "OverTime":
        """A group whose failures are hidden (see ``hidden``): with
        ``timings`` that take time, its tests' and repairs' ends
        among its tests, and settling after two periods. From every
        member up at 0, or from ``start``: its long run at 0 (on the
        calendar of ``tests``), or the states its members' give (see
        ``_Hidden.started``), settling once the repairs under way at 0
        are over too."""
        chain: _Hidden
        lefts = {} if start is None else start.down
        if _timed(timings) or lefts:
            chain = _Timed(model, members, rate, coverage, timings, lefts)
            tests = timed_events(tests, timings, period)
            chain.set_calendar(tests, period)
            settle = 2.0 * float(period)
        else:
            chain = _Hidden(model, members, rate, coverage)
            settle = float(period)
            timings = None
        out = cls(members, chain.combinations, settle, float(period))
        out._chain = chain
        out._tests = sorted(tests)
        out._timings = timings
        out._start = chain.start()
        if start is not None and start.stationary:
            # The long run at 0: where the states repeat from every
            # member up, a whole number of periods on.
            out._start = chain.follow(out._early(settle), np.array([settle]))[
                0
            ]
            out.settle = 0.0
        elif start is not None:
            out._start = chain.started(
                start.known,
                {member: left.level for member, left in lefts.items()},
            )
            out._once = started_ends(lefts)
            longest = max(
                [0.0]
                + [
                    (
                        left.end
                        if left.end is not None
                        else float(left.pieces()[0][-1])
                    )
                    for left in lefts.values()
                ]
            )
            out.settle = settle + float(period) * math.ceil(
                longest / float(period)
            )
        # The states at the start of the period that repeats.
        out._settled = out._follow_early(np.array([out.settle]))[0]
        out.moves = chain.moves
        out.combination = chain.combination
        return out

    def _follow_early(self, x: np.ndarray) -> np.ndarray:
        """The chain's states at the times ``x`` up to the settling time,
        from its states at 0."""
        marks = self.__dict__.get("_early_marks")
        if marks is None:
            events = sorted(self._early(self.settle) + self._once)
            events = [e for e in events if e.time <= self.settle]
            marks = self._early_marks = self._marked(events, self._start, 0.0)
        return self._from_marks(x, *marks)

    def _follow_late(self, x: np.ndarray) -> np.ndarray:
        """The chain's states at the times ``x`` in the period after the
        settling time, from its states then."""
        marks = self.__dict__.get("_late_marks")
        if marks is None:
            stop = self.settle + self.period
            events = [
                e for e in _shifted(self._tests, self.settle) if e.time <= stop
            ]
            marks = self._late_marks = self._marked(
                sorted(events), self._settled, self.settle
            )
        return self._from_marks(x, *marks)

    def _marked(self, events, v, now: float):
        """The chain's states just after each time ``events`` happen at,
        from ``v`` at ``now``: its times and states, ``now``'s first. The
        states at any time come from the last mark before it, as there is
        no event between (see ``_from_marks``), so that each call need
        not follow the chain from the start again."""
        chain = self._chain
        times, vectors = [now], [v]
        k = 0
        while k < len(events):
            at = events[k].time
            v = chain._advance(v, now, at)
            while k < len(events) and events[k].time == at:
                v = chain.tested_at(v, events[k])
                k += 1
            now = at
            times.append(at)
            vectors.append(v)
        return np.array(times), vectors

    def _from_marks(self, x: np.ndarray, times: np.ndarray, vectors):
        """The chain's states at the times ``x``, each followed from the
        last mark at or before it (see ``_marked``), or from the time
        before it, where that is later."""
        chain = self._chain
        out = np.empty((len(x), chain.size))
        order = np.argsort(x, kind="stable")
        ordered = x[order]
        marks = np.maximum(
            np.searchsorted(times, ordered, side="right") - 1, 0
        )
        for mark in np.unique(marks):
            rows = order[marks == mark]
            out[rows] = chain._advance_many(
                vectors[mark], float(times[mark]), x[rows]
            )
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
                values[early] = self._follow_early(x[early])
            if (~early).any():
                phase = np.mod(x[~early] - settle, period)
                values[~early] = self._follow_late(settle + phase)
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

    def jump_times(self, stop: float) -> np.ndarray:
        """The times in ``(0, stop)`` at which a member's test that takes
        time starts (#220), taking it off line while it works: where the
        states jump down, as a node's planned outage does."""
        if self._timings is None:
            return np.empty(0)
        timings = self._timings
        return np.array(
            sorted(
                time
                for time, tests in self.tests(stop).items()
                if time < stop
                and any(
                    test.kind == TEST and timings[test.member].test.takes_time
                    for test in tests
                )
            )
        )

    def _bends(self) -> np.ndarray:
        """Where its states bend or jump within a period (times in ``(0,
        period]``): the tests and the ends of their fixed lengths, and
        after a test of a member whose test or repair takes an
        exponential time (#220), the transient it starts, at geometric
        offsets at the length's scale (the test's, and the test's and
        repair's together), closing in on the test as a tested unit's
        breaks close in on 0."""
        assert self._tests is not None
        period = self.period
        times = [np.array([test.time for test in self._tests])]
        for test in self._tests:
            if test.kind != TEST or self._timings is None:
                continue
            timing = self._timings[test.member]
            scales = []
            if timing.test.random:
                scales.append(_scale(timing.test))
            if timing.repair.random:
                scales.append(sum(scales) + _scale(timing.repair))
            for scale in scales:
                stop = math.ceil(math.log2(period / scale))
                offsets = scale * 2.0 ** np.arange(-3.0, stop + 1.0)
                offsets = timing.test.fixed + offsets[offsets < period]
                at = test.time + offsets
                at -= period * np.floor((at - 1e-12 * period) / period)
                times.append(at)
        return np.unique(np.concatenate(times))

    def knots(self, start: float, stop: float) -> np.ndarray:
        """Where its states bend or jump: the tests and the transients
        they start (every period, see ``_bends``), or the chain's
        times."""
        if self._tests is None:
            return self._chain.knots(start, stop)
        times = self._bends()
        first = int(np.floor(start / self.period))
        last = int(np.ceil(stop / self.period))
        every = (
            times[None, :] + self.period * np.arange(first, last + 1)[:, None]
        ).ravel()
        # The ends of the repairs under way at 0, and the transient of
        # one of a general length (#106), at geometric times on its scale.
        once = [test.time for test in self._once]
        for knots, _ in getattr(self._chain, "left_pieces", {}).values():
            scale = float(np.median(knots[1:])) if len(knots) > 1 else 0.0
            if scale > 0.0:
                once += list(scale * 2.0 ** np.arange(-3.0, 8.0))
        every = np.concatenate([every, once])
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

    def jump_times(self, stop: float) -> np.ndarray:
        """The times in ``(0, stop)`` at which a group's states jump
        down (see ``OverTime.jump_times``)."""
        parts = [group.jump_times(stop) for group in self.groups]
        return np.unique(np.concatenate(parts)) if parts else np.empty(0)


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

    def jump_times(self, stop: float) -> np.ndarray:
        """The times in ``(0, stop)`` at which the groups' states jump
        down, a member's test that takes time starting (#220): the
        system's planned outages there are counted from its values
        just before and at them, as a node's at its exact times (see
        ``_windows._atom_groups``)."""
        return self.curve.jump_times(stop)

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
        ``stop``: its nodes outside the groups', and the planned outages
        where a group's members' tests that take time start (the groups'
        members fail only at random times)."""
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
