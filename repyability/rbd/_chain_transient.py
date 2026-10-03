"""Markov chains over time (#146): a system whose components wait for repair
crews, and a standby group, from their state at 0.

Both have exact long-run values from a continuous-time Markov chain of the
components' (or the units') states (``_crew_chain``, ``_standby_chain``).
Started in a known distribution ``p(0)``, the chain is in each state at
``t`` with probability ``p(t) = p(0) exp(Q t)``, ``Q`` its generator. What
the analyses ask of it is ``p(t) v``, for a few vectors ``v`` over the
states (1 in the states the system is up in, say, or the rate of its
failures in each), and its integral from 0. Both come by uniformization:
with ``L`` a little more than the fastest rate out of any state, ``P = I +
Q / L`` is a discrete chain that steps at the events of a Poisson process
of rate ``L``, so that

    p(t) v = E[s_N],    s_k = p(0) P^k v,    N ~ Poisson(L t),

and, as the time that process spends with ``k`` events from 0 to ``t`` has
mean ``P(N > k) / L``,

    integral from 0 to t of p(u) v du = E[S_N] / L,
    S_j = s_0 + ... + s_(j-1).

The steps are taken until ``p(0) P^k`` is within ``SETTLED`` (in total
variation) of the long-run distribution, after which it stays as close (a
stochastic matrix contracts it): from there each ``s_k`` is taken as its
long-run value. (Rounding in the steps of a chain whose rates are far
apart can hold it a little further away, up to ``FLOOR``: it is then taken
to have settled once it no longer comes closer.) Each time's Poisson sum
runs over the steps within 12 standard deviations and 20 of its mean,
which leaves out less than 1e-30;
past the time at which no step before the last one is left, the values are
their long-run ones and the integrals grow at those rates, exactly. Every
``s_k`` is an average over the states, so the values are accurate to about
1e-13 whatever the rates.

The cost is the number of steps, about 30 times ``L`` over the rate at
which the chain settles: the ratio of the fastest repairs, say, to the
slowest. A small chain takes them many at a time, from its matrix's
powers. A chain whose rates are too far apart for its steps to be taken
within ``MAX_WORK`` is refused.
"""

import math
from typing import Dict, List, Sequence

import numpy as np

from ._point_availability import Atoms, _events, curve_breaks, curve_grids

#: How close (in total variation) the chain must come to its long run to be
#: taken to have reached it; or, if rounding in its steps keeps it further
#: away, how close it must have come when it no longer comes closer.
SETTLED = 1e-13
FLOOR = 1e-10
#: Each time's Poisson sum runs over the steps within this many standard
#: deviations of its mean, and this many more.
_SPREAD, _MARGIN = 12.0, 20.0
#: The most work taken to follow a chain until it settles: its steps times
#: the transitions and states each step touches.
MAX_WORK = 5e9
#: A chain with at most this many states steps from its matrix's powers,
#: as many steps at a time as keep the powers within ``_BLOCK_VALUES``.
_DENSE_STATES = 64
_BLOCK_VALUES = 2**18


def _poisson(j: np.ndarray, mean: float) -> np.ndarray:
    """The Poisson probabilities of ``j`` events with this mean."""
    if mean == 0.0:
        return (j == 0).astype(float)
    from scipy.special import gammaln

    return np.exp(j * math.log(mean) - mean - gammaln(j + 1.0))


class Uniformized:
    """``p(t) v`` and its integral from 0 for the columns ``v`` of
    ``vectors``, for a chain with generator ``generator`` (``Q[i, j]`` the
    rate from state ``i`` to ``j``, dense or sparse) started in ``start``,
    whose long-run distribution is ``steady`` (see the module docstring).

    Raises
    ------
    NotImplementedError
        If its rates are too far apart for its steps to be taken until it
        settles (see ``MAX_WORK``).
    """

    def __init__(self, generator, start, steady, vectors):
        from scipy import sparse

        matrix = sparse.csr_matrix(generator, dtype=float)
        size = matrix.shape[0]
        columns = np.asarray(vectors, dtype=float).reshape(size, -1)
        self.steady = np.asarray(steady, dtype=float)
        #: The long-run value of each ``p(t) v``.
        self.limits = self.steady @ columns
        top = float(np.max(-matrix.diagonal())) if size else 0.0
        #: The uniformizing rate: a step at each event of a Poisson
        #: process of this rate, and every state keeps some probability at
        #: each, so the steps settle steadily.
        self.rate = 1.02 * top if top > 0.0 else 1.0
        if top > 0.0:
            jump = sparse.identity(size, format="csr") + matrix / self.rate
            deviations = self._follow(
                jump, np.asarray(start, dtype=float), columns
            )
        else:
            deviations = np.empty((0, columns.shape[1]))
        #: ``s_k`` less its long-run value, for each step before the chain
        #: settles; and their running sums ``D_j`` (``D_0 = 0``), and all
        #: of them, ``D_K``.
        self.deviations = deviations
        steps = len(deviations)
        sums = np.cumsum(deviations, axis=0)
        self.total = sums[-1] if steps else np.zeros(columns.shape[1])
        self.running = np.vstack([np.zeros_like(self.total)[None], sums])[
            :steps
        ]
        if steps == 0:
            self.settle = 0.0
        else:
            # The time from which no step before the last one is left in
            # the Poisson sums: mean - 12 sd - 20 = steps.
            root = 0.5 * (
                _SPREAD + math.sqrt(_SPREAD**2 + 4.0 * (steps + _MARGIN))
            )
            self.settle = root * root / self.rate

    def _follow(self, jump, start: np.ndarray, columns: np.ndarray):
        """``s_k - s_inf`` for each step ``k`` until ``start P^k`` has
        settled at the long run (see the module docstring)."""
        size = jump.shape[0]
        limits = self.limits
        budget = MAX_WORK / (jump.nnz + 3 * size + size * columns.shape[1])
        rows: List[np.ndarray] = []
        v = start.copy()
        steps = 0
        history = [(0, float(np.abs(v - self.steady).sum()))]
        if size <= _DENSE_STATES:
            dense = jump.toarray()
            block = int(max(1, min(256, _BLOCK_VALUES // (size * size))))
            powers = [np.identity(size)]
            for _ in range(block):
                powers.append(powers[-1] @ dense)
            # v times each power from the 0th to the (block - 1)th, and the
            # block-th for the next block.
            stacked = np.hstack(powers)
        else:
            stacked = None
            transposed = jump.T.tocsr()
            block = 64
        while True:
            if stacked is not None:
                ahead = (v @ stacked).reshape(block + 1, size)
                chunk, v = ahead[:-1], ahead[-1]
            else:
                chunk = np.empty((block, size))
                for j in range(block):
                    chunk[j] = v
                    v = transposed @ v
            distance = np.abs(chunk - self.steady).sum(axis=1)
            settled = np.flatnonzero(distance <= SETTLED)
            stop = int(settled[0]) if settled.size else block
            rows.append(chunk[:stop] @ columns - limits)
            steps += stop
            if settled.size:
                break
            # Rounding in the steps can leave the chain a little further
            # from the long run than SETTLED (the more, the further apart
            # its rates): once it is within FLOOR and no longer coming
            # closer, by a tenth over the last hundredth of its steps
            # (which, settling steadily, it would), it has settled.
            history.append((steps, float(distance[-1])))
            back = steps - max(block, steps // 100)
            earlier = next(d for s, d in reversed(history) if s <= back)
            if distance[-1] <= FLOOR and distance[-1] > 0.9 * earlier:
                break
            if steps > budget:
                raise NotImplementedError(
                    "its rates are too far apart: following it from its "
                    f"state at 0 until it settles takes more than {steps:,} "
                    f"steps of its uniformized chain, at a rate of "
                    f"{self.rate:.3g}"
                )
            # Keep it a distribution, against rounding over many steps.
            v = np.maximum(v, 0.0)
            v /= v.sum()
        return np.vstack(rows)

    def _sums(self, x: np.ndarray, terms: np.ndarray, out: np.ndarray):
        """Add to ``out`` each time's Poisson sum of ``terms`` (one row per
        step) over the steps before the chain settles."""
        steps = len(terms)
        if not steps:
            return out
        mean = self.rate * x
        spread = _SPREAD * np.sqrt(mean) + _MARGIN
        low = np.floor(np.maximum(mean - spread, 0.0)).astype(np.int64)
        high = np.minimum(np.ceil(mean + spread).astype(np.int64) + 1, steps)
        for i in np.flatnonzero(low < high):
            j = np.arange(low[i], high[i])
            span = terms[low[i] : high[i]]  # noqa: E203
            out[i] += _poisson(j, float(mean[i])) @ span
        return out

    def values(self, x) -> np.ndarray:
        """``p(t) v`` at each time of ``x`` (rows) for each vector
        (columns)."""
        x = np.asarray(x, dtype=float).ravel()
        out = np.tile(self.limits, (len(x), 1))
        return self._sums(x, self.deviations, out)

    def integrals(self, x) -> np.ndarray:
        """The integral of ``p(t) v`` from 0 to each time of ``x`` (rows)
        for each vector (columns)."""
        x = np.asarray(x, dtype=float).ravel()
        # E[S_N] = L t s_inf + E[D_N], and D_N = D_K for N >= K.
        out = np.outer(x, self.limits) + self.total / self.rate
        return self._sums(x, (self.running - self.total) / self.rate, out)

    def knots(self, start: float, stop: float) -> np.ndarray:
        """Times from 0 to ``settle`` between which the values are smooth
        enough for quadrature: from a twentieth of the fastest time scale,
        ``1 / L``, each a fifth (in the log) further on."""
        if self.settle == 0.0:
            return np.empty(0)
        first = 0.05 / self.rate
        count = int(math.ceil(math.log(max(self.settle / first, 1.0)) / 0.2))
        times = np.concatenate(
            [[0.0], first * np.exp(0.2 * np.arange(count)), [self.settle]]
        )
        times = times[times <= self.settle]
        return times[(times >= start) & (times <= stop)]


class ChainCurve:
    """A standby group over time (see ``_point_availability`` for what a
    curve gives): its point availability, and its expected failures and
    the repairs it pays for (its units' failures) before each time, from
    a ``Uniformized`` chain whose vectors are, in each state, whether the
    group is up, the rate of its failures and that of its units'."""

    period = None

    def __init__(self, chain: Uniformized):
        self.chain = chain
        self.settle = chain.settle

    def at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        up = self.chain.values(x)[:, 0]
        return np.clip(up, 0.0, 1.0).reshape(x.shape)

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The group's expected failures before each time ``x``, and the
        repairs it pays for (its corrective actions)."""
        x = np.asarray(x, dtype=float)
        totals = np.maximum(self.chain.integrals(x), 0.0)
        return _events(
            totals[:, 1].reshape(x.shape),
            corrective=totals[:, 2].reshape(x.shape),
        )

    def atoms(self, stop: float) -> Atoms:
        """None: the chain moves only at random times."""
        return Atoms.none()

    def knots(self, start: float, stop: float) -> np.ndarray:
        return self.chain.knots(start, stop)


def pattern_weights(availabilities: Sequence[np.ndarray]) -> np.ndarray:
    """For independent nodes up with these probabilities at each time,
    the probability of each pattern of them up (bit ``j`` of the pattern's
    number: node ``j`` up), one row per time."""
    count = len(availabilities)
    patterns = np.arange(2**count)
    size = len(availabilities[0]) if count else 1
    weights = np.ones((size, len(patterns)))
    for j, up in enumerate(availabilities):
        up = np.asarray(up, dtype=float)[:, None]
        weights *= np.where((patterns >> j) & 1, up, 1.0 - up)
    return weights


class CrewCurve:
    """A system whose components wait for repair crews, over time, from the
    crews' ``Uniformized`` chain: its point availability is ``sum_m p(t)
    u_m w_m(t)``, ``u_m`` (the chain's ``m``-th vector) whether it is up in
    each state with its nested RBDs in pattern ``m`` (see
    ``pattern_weights``), and ``w_m(t)`` the probability of that pattern at
    ``t``, from their own curves (``nested``): they have crews of their
    own, so they are independent of the chain. It is constant after
    ``settle``, or repeats with ``period`` with the nested RBDs'.

    With no nested RBDs the chain's second vector is the rate of the
    system's failures in each state, and its integrals give the system's
    uptime and failures from 0 exactly (``integral``, ``events``)."""

    def __init__(self, rbd, chain: Uniformized, nested: dict):
        from .repairable_rbd import _settling

        self.rbd = rbd
        self.chain = chain
        self.nested = nested
        self.patterns = 2 ** len(nested)
        self.settle, self.period = _settling(
            [ChainCurve(chain), *nested.values()]
        )

    def at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        flat = x.ravel()
        values = self.chain.values(flat)[:, : self.patterns]
        if self.nested:
            values = values * pattern_weights(
                [curve.at(flat) for curve in self.nested.values()]
            )
        return np.clip(values.sum(axis=1), 0.0, 1.0).reshape(x.shape)

    def integral(self, x: np.ndarray) -> np.ndarray:
        """The integral of the point availability from 0 to each time
        ``x``: with no nested RBDs, the chain's."""
        assert not self.nested, "the nested RBDs' curves need quadrature"
        return np.maximum(self.chain.integrals(x)[:, 0], 0.0)

    def _require_alone(self) -> None:
        """Raise unless the chain counts the system's events: there are
        no nested RBDs."""
        if self.nested:
            self.rbd._no_crew_window(next(iter(self.nested)))

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The system's expected failures before each time ``x``, and its
        planned outages: none, as the crews' components have no scheduled
        maintenance."""
        self._require_alone()
        x = np.asarray(x, dtype=float)
        failures = np.maximum(self.chain.integrals(x.ravel())[:, 1], 0.0)
        failures = failures.reshape(x.shape)
        return _events(failures, np.zeros_like(failures))

    def atoms(self, stop: float) -> Atoms:
        """None: the chain moves only at random times."""
        self._require_alone()
        return Atoms.none()

    def knots(self, start: float, stop: float) -> np.ndarray:
        parts = [self.chain.knots(start, stop)]
        parts += [curve.knots(start, stop) for curve in self.nested.values()]
        return np.concatenate(parts)

    def breaks(self, start: float, stop: float) -> np.ndarray:
        parts = [self.chain.knots(start, stop)]
        parts += [curve_breaks(c, start, stop) for c in self.nested.values()]
        return np.concatenate(parts)

    def grids(self) -> list:
        return [
            g for curve in self.nested.values() for g in curve_grids(curve)
        ]


class CrewNodeEvents:
    """A component that waits for repair crews, as the expected events over
    a window count it: it fails at its constant ``rate`` while it is up, and
    each failure is a repair, from the integral of the ``column``-th
    vector of a ``Uniformized`` chain (whether it is up, in each
    state)."""

    def __init__(self, chain: Uniformized, column: int, rate: float):
        self.chain = chain
        self.column = column
        self.rate = rate

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        x = np.asarray(x, dtype=float)
        up = self.chain.integrals(x)[:, self.column].reshape(x.shape)
        return _events(self.rate * np.maximum(up, 0.0))
