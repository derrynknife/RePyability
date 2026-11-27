"""A component given as a Markov chain: its states, the rates between
them, and the states it works in (or each state's capacity).

Some components are not a life and a repair: a unit with a degraded
state it can be restored from, a failure that is detected or not, a pump
that stands by and runs. Their states and the rates between them are a
continuous-time Markov chain (IEC 61165). Its probabilities over time
come from the uniformized chain every other chain here is followed by
(``_chain_transient``), and its long-run distribution from one linear
solve.
"""

import math
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np

from repyability.utils.checks import is_number, real_array
from repyability.utils.checks import seed as check_seed
from repyability.utils.deprecation import refuse_removed_names
from repyability.utils.wrappers import numpy_seed

from . import _chain_transient


def _closure(moves: np.ndarray) -> np.ndarray:
    """Whether each state (row) reaches each other (column) in any number
    of moves, itself included."""
    size = len(moves)
    reach = moves | np.eye(size, dtype=bool)
    for middle in range(size):
        reach |= reach[:, [middle]] & reach[[middle], :]
    return reach


def _names(states) -> str:
    return ", ".join(repr(state) for state in states)


class _Life:
    """A lifetime's ``ff`` and ``sf`` and its median, as the inversion
    of a diagram's life takes them (``qf``)."""

    def __init__(self, component, invert):
        self.ff, self.sf = component.ff, component.sf
        self._invert = invert

    def time_to_reliability(self, reliability: float) -> float:
        return float(self._invert(lambda t: float(self.sf(t)), reliability))


@refuse_removed_names
class MarkovComponent:
    """A component whose states and the rates between them are known: a
    continuous-time Markov chain.

    The component is in one of its ``states`` at a time, and moves from
    one to another at the constant rates of ``rates``. It works in the
    states of ``up`` (or, with ``capacity``, in those with a capacity
    above 0) and is failed in the others. A unit that degrades, is
    restored from its degraded state or fails from it, and is repaired
    from failure, has three states.

    A component with a way back from its failed states is repairable: it
    is a component of a
    [`RepairableRBD`][repyability.RepairableRBD], given as it is (not in
    a spec dict: its restorations are its own transitions), independent
    of the diagram's other components. One with no way back has a
    lifetime, the time until it first fails: it is a node of a
    [`NonRepairableRBD`][repyability.NonRepairableRBD], or the
    ``"reliability"`` of a repairable component's spec, as any model of
    a life is.

    Parameters
    ----------
    states : sequence of hashable
        The states' names, each once.
    rates : dict
        The rate of each transition, ``{(from, to): rate}``: a positive
        number, in events per unit of the diagram's time. A transition
        not given never happens.
    up : collection of hashable, optional
        The states the component works in. Give this or ``capacity``.
    capacity : dict, optional
        Each state's capacity, ``{state: level}``, at least 0 (``inf``
        for no limit): the component works in the states with a capacity
        above 0, and a diagram's capacity analyses take each state's
        level.
    start : hashable or dict, optional
        The state at time 0, or its distribution ``{state: probability}``
        (summing to 1). By default the first state.

    Attributes
    ----------
    states : tuple
        The states, in the order given.
    rates : dict
        The transitions' rates, as floats.
    up : tuple
        The states the component works in, in the states' order.
    capacity : dict or None
        Each state's capacity, if given.
    start : dict
        The distribution of the state at 0, over the states it may be.
    repairable : bool
        Whether a failed state has a way back to a working one.

    Raises
    ------
    TypeError
        If a rate, a capacity or a probability is not a number.
    ValueError
        If a state is named twice, a transition, ``up``, ``capacity`` or
        ``start`` names a state not in ``states``, a rate is not positive
        and finite, a state is never reached from the start, a state
        never reaches a failed one, or the chain's long run depends on
        where it goes (it has more than one set of states it cannot
        leave).

    Examples
    --------
    A unit that degrades at 0.01 a time unit, is restored from its
    degraded state at 0.2 or fails from it at 0.05, and is repaired from
    failure at 0.5:

    >>> from repyability import MarkovComponent
    >>> unit = MarkovComponent(
    ...     ["working", "degraded", "failed"],
    ...     {
    ...         ("working", "degraded"): 0.01,
    ...         ("degraded", "working"): 0.2,
    ...         ("degraded", "failed"): 0.05,
    ...         ("failed", "working"): 0.5,
    ...     },
    ...     up=["working", "degraded"],
    ... )
    >>> round(unit.mean_availability(), 6)
    0.996169
    >>> round(unit.failure_frequency(), 6)
    0.001916
    """

    def __init__(
        self,
        states,
        rates,
        up=None,
        capacity=None,
        start=None,
    ):
        if isinstance(states, (str, bytes)) or not hasattr(states, "__iter__"):
            raise TypeError(
                "states must be a sequence of the states' names, got "
                f"{states!r}."
            )
        self.states: Tuple[Hashable, ...] = tuple(states)
        if len(self.states) < 2:
            raise ValueError(
                "A Markov component needs at least two states (one it "
                f"works in and one it is failed in), got {states!r}."
            )
        twice = sorted(
            {s for s in self.states if self.states.count(s) > 1}, key=str
        )
        if twice:
            raise ValueError(
                f"states names {_names(twice)} more than once: give each "
                "state once."
            )
        index = {state: i for i, state in enumerate(self.states)}
        self._index = index
        size = len(self.states)

        def known(state, what: str):
            try:
                found = state in index
            except TypeError:
                found = False
            if not found:
                raise ValueError(
                    f"{what} names the state {state!r}, which is not in "
                    f"states ({_names(self.states)})."
                )
            return state

        if not isinstance(rates, dict) or not rates:
            raise TypeError(
                "rates must be a dict of each transition's rate, "
                f"{{(from, to): rate}}, got {rates!r}."
            )
        self.rates: Dict[Tuple[Hashable, Hashable], float] = {}
        generator = np.zeros((size, size))
        for move, rate in rates.items():
            if not isinstance(move, tuple) or len(move) != 2:
                raise TypeError(
                    "rates takes each transition as a (from, to) pair of "
                    f"states, got {move!r}."
                )
            source = known(move[0], f"The transition {move!r}")
            target = known(move[1], f"The transition {move!r}")
            if source == target:
                raise ValueError(
                    f"The transition {move!r} is from a state to itself: "
                    "leave it out."
                )
            if not is_number(rate):
                raise TypeError(
                    f"The rate of the transition {move!r} must be a "
                    f"number, got {rate!r}."
                )
            rate = float(rate)
            if not (math.isfinite(rate) and rate > 0.0):
                raise ValueError(
                    f"The rate of the transition {move!r} must be "
                    f"positive and finite, got {rate!r}: leave out a "
                    "transition that never happens."
                )
            self.rates[(source, target)] = rate
            generator[index[source], index[target]] = rate
        generator -= np.diag(generator.sum(axis=1))

        if (up is None) == (capacity is None):
            raise ValueError(
                "Give up, the states the component works in, or "
                "capacity, each state's capacity (it works in those "
                "above 0): one of them."
            )
        self.capacity: Optional[Dict[Hashable, float]] = None
        if capacity is not None:
            if not isinstance(capacity, dict):
                raise TypeError(
                    "capacity must be a dict of each state's capacity, "
                    f"{{state: level}}, got {capacity!r}."
                )
            levels = {}
            for state, level in capacity.items():
                known(state, "capacity")
                if not is_number(level):
                    raise TypeError(
                        f"The capacity of state {state!r} must be a "
                        f"number, got {level!r}."
                    )
                if not float(level) >= 0.0:
                    raise ValueError(
                        f"The capacity of state {state!r} must be at "
                        f"least 0, got {level!r}."
                    )
                levels[state] = float(level)
            missing = [s for s in self.states if s not in levels]
            if missing:
                raise ValueError(
                    f"capacity gives no level for {_names(missing)}: "
                    "give every state's (0 for a failed state)."
                )
            self.capacity = {s: levels[s] for s in self.states}
            working = {s for s in self.states if levels[s] > 0.0}
        else:
            if isinstance(up, (str, bytes)) or not hasattr(up, "__iter__"):
                raise TypeError(
                    "up must be a collection of the states the component "
                    f"works in, got {up!r}."
                )
            working = {known(state, "up") for state in up}
        self.up: Tuple[Hashable, ...] = tuple(
            s for s in self.states if s in working
        )
        if not self.up or len(self.up) == size:
            raise ValueError(
                "A Markov component needs a state it works in and a "
                "state it is failed in; it works in "
                f"{_names(self.up) or 'none'} of {_names(self.states)}."
            )

        if start is None:
            start = self.states[0]
        if isinstance(start, dict):
            given = {}
            for state, p in start.items():
                known(state, "start")
                if not is_number(p):
                    raise TypeError(
                        f"The probability of starting in {state!r} must "
                        f"be a number, got {p!r}."
                    )
                if not 0.0 <= float(p) <= 1.0:
                    raise ValueError(
                        f"The probability of starting in {state!r} must "
                        f"be from 0 to 1, got {p!r}."
                    )
                given[state] = float(p)
            total = math.fsum(given.values())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    "start's probabilities must sum to 1, got a sum of "
                    f"{total!r}."
                )
            self.start = {
                s: given[s] / total
                for s in self.states
                if given.get(s, 0.0) > 0.0
            }
        else:
            self.start = {known(start, "start"): 1.0}

        self._generator = generator
        self._up = np.array([s in working for s in self.states], dtype=float)
        self._start = np.array([self.start.get(s, 0.0) for s in self.states])
        reach = _closure(generator > 0.0)
        reached = np.asarray(reach[self._start > 0.0].any(axis=0))
        if not reached.all():
            lost = [s for s, r in zip(self.states, reached) if not r]
            raise ValueError(
                f"The state(s) {_names(lost)} are never reached from the "
                f"start ({_names(self.start)}): give the transitions "
                "into them, or leave them out."
            )
        down = self._up == 0.0
        fails = np.asarray(reach[:, down].any(axis=1))
        if not fails.all():
            safe = [s for s, f in zip(self.states, fails) if not f]
            raise ValueError(
                f"From the state(s) {_names(safe)} the component never "
                "reaches a failed state: give the transitions by which "
                "it fails."
            )
        #: Whether a failed state has a way back to a working one.
        self.repairable = bool((generator[down][:, ~down] > 0.0).any())
        if self.repairable:
            # The states it cannot leave together: one set, or its long
            # run would depend on which it reaches.
            closed = [
                i
                for i in range(size)
                if (reach[i] <= reach[:, i]).all()  # all it reaches reach it
            ]
            classes = {
                tuple(int(j) for j in np.flatnonzero(reach[i] & reach[:, i]))
                for i in closed
            }
            if len(classes) > 1:
                sets = "; ".join(
                    _names(self.states[i] for i in members)
                    for members in sorted(classes)
                )
                raise ValueError(
                    "The chain has more than one set of states it cannot "
                    f"leave ({sets}), so its long-run values would depend "
                    "on which it reaches: give the transitions between "
                    "them, or model each as a component."
                )
            (members,) = classes
            self._steady = self._stationary(generator, list(members))
            self._chain = generator
        else:
            # A lifetime: the failed states are where it ends.
            ends = generator.copy()
            ends[down] = 0.0
            self._chain = ends
            self._steady = self._absorbed(ends, down)
        self._followed: Optional[_chain_transient.Uniformized] = None

    @staticmethod
    def _stationary(generator: np.ndarray, members: List[int]) -> np.ndarray:
        """The long-run distribution of a chain whose states it cannot
        leave are ``members``: their balance equations, for their
        probabilities relative to the first's (which keeps small ones
        precise, see ``_crew_chain``)."""
        steady = np.zeros(len(generator))
        balance = generator[np.ix_(members, members)].T
        relative = (
            np.linalg.solve(balance[1:, 1:], -balance[1:, 0])
            if len(members) > 1
            else np.empty(0)
        )
        inside = np.append(1.0, np.maximum(relative, 0.0))
        steady[members] = inside / inside.sum()
        return steady

    def _absorbed(self, ends: np.ndarray, down: np.ndarray) -> np.ndarray:
        """Where a lifetime's chain ends: the probability of each failed
        state being the one it fails into."""
        steady = np.where(down, self._start, 0.0)
        inside = ends[np.ix_(~down, ~down)]
        # The expected time in each working state, from the start.
        time = np.linalg.solve(-inside.T, self._start[~down])
        steady[down] += time @ ends[np.ix_(~down, down)]
        return steady

    def __repr__(self) -> str:
        kind = "repairable" if self.repairable else "a lifetime"
        return (
            f"MarkovComponent({len(self.states)} states, "
            f"{len(self.rates)} transitions, up in {_names(self.up)}; "
            f"{kind})"
        )

    #: As a RepairableRBD's component: it is its own model, and its
    #: restorations are its own transitions.
    time_to_replace = None

    @property
    def reliability(self) -> "MarkovComponent":
        """The component itself: as a ``RepairableRBD``'s component it
        is its own model."""
        return self

    # The chain's vectors, each a value per state.

    @property
    def _down(self) -> np.ndarray:
        return 1.0 - self._up

    @property
    def _failures(self) -> np.ndarray:
        """In each state, the rate of the component's failures: of its
        moves from a working state to a failed one."""
        off = self._generator - np.diag(np.diag(self._generator))
        return self._up * (off @ self._down)

    @property
    def _restorations(self) -> np.ndarray:
        """In each state, the rate of its moves from a failed state to a
        working one."""
        off = self._generator - np.diag(np.diag(self._generator))
        return self._down * (off @ self._up)

    @property
    def _levels(self) -> np.ndarray:
        """Each state's capacity (1 and 0 without ``capacity``)."""
        if self.capacity is None:
            return self._up.copy()
        return np.array([self.capacity[s] for s in self.states])

    @property
    def capacities(self) -> Tuple[float, ...]:
        """The capacity of each state it works in, in the states'
        order (as a ``DegradingNode``'s stages')."""
        levels = self._levels
        return tuple(float(levels[i]) for i in np.flatnonzero(self._up))

    def stage_fractions(self) -> np.ndarray:
        """The share of a repairable component's long-run working time
        in each state it works in."""
        self._require_repairable("long-run time in each state")
        working = self._steady[self._up > 0.0]
        return working / working.sum()

    def capacity_distribution(self, x):
        """The distribution of the component's capacity at the times
        ``x``: the capacity of the state it is in (1 in a working state
        and 0 in a failed one, without ``capacity``).

        Parameters
        ----------
        x : float or array_like
            The time(s), at least 0.

        Returns
        -------
        CapacityDistribution
            The capacities it can have (equal ones merged) and their
            probabilities: one per level for a scalar ``x``, and one row
            per level and one column per time for an array.
        """
        from .capacity import merged
        from .results import CapacityDistribution

        scalar = np.ndim(x) == 0
        levels, probabilities = merged(
            self._levels, self._probabilities(np.atleast_1d(x)).T
        )
        return CapacityDistribution(
            levels, probabilities[:, 0] if scalar else probabilities
        )

    def _follow(self) -> "_chain_transient.Uniformized":
        """The chain followed over time from its start: every state's
        probability (``Uniformized``'s vectors the identity)."""
        if self._followed is None:
            try:
                self._followed = _chain_transient.Uniformized(
                    self._chain,
                    self._start,
                    self._steady,
                    np.identity(len(self.states)),
                )
            except NotImplementedError as error:
                raise NotImplementedError(
                    "The Markov component's chain cannot be followed "
                    f"over time: {error}."
                ) from None
        return self._followed

    def _probabilities(self, x) -> np.ndarray:
        """Each state's probability (columns) at each time (rows)."""
        x = real_array(x, "x").ravel()
        if np.any(x < 0.0) or np.any(np.isnan(x)):
            raise ValueError("x must be times of at least 0.")
        # Past the time the chain has settled by, its long run exactly.
        followed = self._follow()
        before = x <= followed.settle
        out = np.tile(self._steady, (len(x), 1))
        if before.any():
            out[before] = followed.values(x[before])
        return np.clip(out, 0.0, 1.0)

    def state_probabilities(self, x) -> Dict[Hashable, np.ndarray]:
        """The probability of each state at the times ``x``.

        Parameters
        ----------
        x : array_like
            Times, at least 0.

        Returns
        -------
        dict
            Each state's probability at each time, an array of ``x``'s
            shape. A lifetime's failed states are where it ends: it
            stays in the one it fails into.
        """
        shape = np.shape(x)
        values = self._probabilities(x)
        return {
            state: values[:, i].reshape(shape)
            for i, state in enumerate(self.states)
        }

    def steady_state(self) -> Dict[Hashable, float]:
        """The long-run probability of each state: the fraction of time
        a repairable component spends in it; for a lifetime, the
        probability of each failed state being the one it fails into."""
        return {s: float(p) for s, p in zip(self.states, self._steady)}

    def _require_repairable(self, what: str) -> None:
        if not self.repairable:
            raise NotImplementedError(
                f"This Markov component has no {what}: no failed state "
                "has a way back to a working one, so it has a lifetime "
                "(sf, ff, mean, random). Give the transitions that "
                "restore it, or use it as a life (a NonRepairableRBD's "
                "node, or the 'reliability' of a repairable component's "
                "spec)."
            )

    def _require_lifetime(self, what: str) -> None:
        if self.repairable:
            back = [
                move
                for move in self.rates
                if move[0] not in self.up and move[1] in self.up
            ]
            raise NotImplementedError(
                f"This Markov component has no {what}: it is restored "
                f"from its failed states (the transition(s) {_names(back)}"
                "), so it fails and works again and has no one lifetime. "
                "Give it as a component of a RepairableRBD (and ask for "
                "point_availability or mean_availability), or leave out "
                "the transitions back for the time to its first failure."
            )

    def point_availability(self, x) -> np.ndarray:
        """The probability that the component works at the times ``x``
        (for a lifetime, its reliability).
        Parameters
        ----------
        x : array_like
            Times, at least 0.
        """
        shape = np.shape(x)
        return np.clip(self._probabilities(x) @ self._up, 0.0, 1.0).reshape(
            shape
        )

    def point_unavailability(self, x) -> np.ndarray:
        """The probability that the component is failed at the times
        ``x``, summed over its failed states (so that a small one keeps
        its precision).
        Parameters
        ----------
        x : array_like
            Times, at least 0.
        """
        shape = np.shape(x)
        return np.clip(self._probabilities(x) @ self._down, 0.0, 1.0).reshape(
            shape
        )

    def mean_availability(self) -> float:
        """The long-run fraction of time a repairable component works."""
        self._require_repairable("long-run availability")
        return float(self._steady @ self._up)

    def mean_unavailability(self) -> float:
        """The long-run fraction of time a repairable component is
        failed, summed over its failed states."""
        self._require_repairable("long-run unavailability")
        return float(self._steady @ self._down)

    def failure_frequency(self) -> float:
        """A repairable component's long-run failures per unit time: its
        moves from a working state to a failed one."""
        self._require_repairable("long-run failure frequency")
        return float(self._steady @ self._failures)

    # A lifetime, as a model of a life.

    def sf(self, x, *args, **kwargs):
        """The reliability at the times ``x``: the probability that the
        component has not yet failed. A lifetime only (no way back from
        the failed states).
        Parameters
        ----------
        x : array_like
            Times, at least 0.
        """
        self._require_lifetime("reliability (sf)")
        return self.point_availability(x)

    def ff(self, x, *args, **kwargs):
        """The probability of having failed by the times ``x``, summed
        over the failed states.
        Parameters
        ----------
        x : array_like
            Times, at least 0.
        """
        self._require_lifetime("unreliability (ff)")
        return self.point_unavailability(x)

    def df(self, x, *args, **kwargs):
        """The density of the lifetime at the times ``x``: the rate of
        failures from each working state, by its probability.
        Parameters
        ----------
        x : array_like
            Times, at least 0.
        """
        self._require_lifetime("density (df)")
        shape = np.shape(x)
        return np.maximum(
            self._probabilities(x) @ self._failures, 0.0
        ).reshape(shape)

    def hf(self, x, *args, **kwargs):
        """The hazard rate of the lifetime at the times ``x``.
        Parameters
        ----------
        x : array_like
            Times, at least 0.
        """
        self._require_lifetime("hazard rate (hf)")
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.df(x) / self.sf(x)

    def Hf(self, x, *args, **kwargs):
        """The cumulative hazard of the lifetime at the times ``x``.
        Parameters
        ----------
        x : array_like
            Times, at least 0.
        """
        self._require_lifetime("cumulative hazard (Hf)")
        with np.errstate(divide="ignore"):
            return -np.log(self.sf(x))

    def mean(self, *args, **kwargs) -> float:
        """The mean lifetime: the expected time in the working states,
        from one linear solve."""
        self._require_lifetime("mean lifetime")
        working = self._up > 0.0
        inside = self._generator[np.ix_(working, working)]
        time = np.linalg.solve(-inside.T, self._start[working])
        return float(time.sum())

    def qf(self, p, *args, **kwargs):
        """The time by which a share ``p`` of units have failed: the
        inverse of ``ff``, for each ``p`` in [0, 1], found as a
        diagram's is
        ([`NonRepairableRBD.qf`][repyability.NonRepairableRBD.qf]). The
        chain's probabilities are accurate to about 1e-13, so a share
        within that of 0 or 1 is not resolved.
        Parameters
        ----------
        p : float or array_like
            The shares failed, each in [0, 1].
        """
        from ._measures import invert_reliability
        from .non_repairable_rbd import _invert_life

        self._require_lifetime("quantiles (qf)")
        given = np.asarray(p, dtype=float)
        shares = np.atleast_1d(given).ravel()
        if not np.all((shares >= 0.0) & (shares <= 1.0)):
            raise ValueError(
                "qf takes shares failed in [0, 1] (a probability of "
                f"failing by the time); got {p!r}."
            )
        out = np.empty_like(shares)
        at_start = float(self.ff(0.0))
        out[shares <= at_start] = 0.0
        out[shares == 1.0] = np.inf
        todo = (shares > at_start) & (shares < 1.0)
        if np.any(todo):
            out[todo] = _invert_life(
                _Life(self, invert_reliability), shares[todo], {}
            )
        return float(out[0]) if given.ndim == 0 else out.reshape(given.shape)

    def random(self, size, seed=None) -> np.ndarray:
        """Draw lifetimes: each a walk through the states from the
        start, an exponential time in each, until a failed one.

        Parameters
        ----------
        size : int
            The number of lifetimes to draw.
        seed : int or None, optional
            If given, numpy's global RNG is seeded for the draw and its
            previous state restored afterwards. By default None.

        Returns
        -------
        numpy.ndarray
            The lifetimes, of shape ``size``.
        """
        self._require_lifetime("lifetimes to draw (random)")
        check_seed(seed)
        shape = (size,) if np.ndim(size) == 0 else tuple(size)
        count = int(np.prod(shape))
        out = -np.diag(self._generator)
        with np.errstate(divide="ignore", invalid="ignore"):
            jumps = np.cumsum(
                (self._generator - np.diag(np.diag(self._generator)))
                / out[:, None],
                axis=1,
            )
        with numpy_seed(seed):
            state = np.searchsorted(
                np.cumsum(self._start), np.random.random(count), side="right"
            ).clip(max=len(self.states) - 1)
            time = np.zeros(count)
            alive = np.flatnonzero(self._up[state] > 0.0)
            while alive.size:
                here = state[alive]
                time[alive] += np.random.exponential(1.0 / out[here])
                u = np.random.random(alive.size)
                state[alive] = ((u[:, None] >= jumps[here]).sum(axis=1)).clip(
                    max=len(self.states) - 1
                )
                alive = alive[self._up[state[alive]] > 0.0]
        return time.reshape(shape)

    # Its rates as levers, and saving.

    def rate_names(self) -> List[str]:
        """The name of each transition's rate as a lever and an
        uncertain parameter, in the order of ``rates``:
        ``"rate:<from>-><to>"``."""
        return [f"rate:{source}->{target}" for source, target in self.rates]

    def with_rates(self, rates: dict) -> "MarkovComponent":
        """The component with some transitions' rates changed: a copy,
        ``rates`` (``{(from, to): rate}``) replacing or adding to its
        own.
        Parameters
        ----------
        rates : dict
            The rates to change or add, ``{(from, to): rate}``.
        """
        if not isinstance(rates, dict):
            raise TypeError(
                "rates must be a dict of transitions' rates, "
                f"{{(from, to): rate}}, got {rates!r}."
            )
        return MarkovComponent(
            self.states,
            {**self.rates, **rates},
            up=None if self.capacity is not None else self.up,
            capacity=self.capacity,
            start=self.start,
        )

    def to_dict(self) -> Dict[str, Any]:
        """The component as plain data (ready for ``json.dumps`` when its
        states' names are text or numbers), which ``from_dict`` reads."""
        out: Dict[str, Any] = {
            "states": list(self.states),
            "rates": [
                [source, target, rate]
                for (source, target), rate in self.rates.items()
            ],
            "start": [[state, p] for state, p in self.start.items()],
        }
        if self.capacity is not None:
            out["capacity"] = [
                [state, level] for state, level in self.capacity.items()
            ]
        else:
            out["up"] = list(self.up)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "MarkovComponent":
        """The component ``to_dict`` wrote.
        Parameters
        ----------
        data : dict
            What ``to_dict`` returned (as read back from JSON, too).
        """

        def name(state):
            return tuple(state) if isinstance(state, list) else state

        capacity = data.get("capacity")
        return cls(
            [name(s) for s in data["states"]],
            {(name(a), name(b)): rate for a, b, rate in data["rates"]},
            up=(
                None if capacity is not None else [name(s) for s in data["up"]]
            ),
            capacity=(
                None
                if capacity is None
                else {name(s): level for s, level in capacity}
            ),
            start={name(s): p for s, p in data["start"]},
        )
