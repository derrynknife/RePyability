"""The exact lifetime distribution of a standby arrangement of exponential
units (#339): any ``k``, any dormancy (cold, warm or hot), units of
different rates, and spares that may fail to switch in or to start.

An exponential unit's chance of failing in the next moment does not depend
on its age, so the arrangement's future depends only on which units are
left and which of them operate: a Markov chain. ``StandbyModel``'s rules
make the states:

- the first ``k`` units left (in list order) operate, each failing at its
  rate; the others wait as spares, failing at ``dormancy`` times theirs
  (found at once, and passed over when their turn comes);
- when an operating unit fails, the next spare left is switched in: the
  switch fails with ``1 - s`` (and the arrangement fails then), else the
  spare fails to start with ``1 - p`` (it is lost, and the next spare is
  tried at once), else it operates;
- the arrangement fails when fewer than ``k`` units can operate.

A state is the rates of the units operating (as a sorted tuple, as which
of two units with the same rate operates does not matter) and the spares
left, in order, each with its rate and probabilities; identical units
then share states, so the chain stays small (``n - k + 2`` states for
identical units). Units only ever leave, so the chain never returns to a
state.

The chain is followed by uniformization with squaring: with ``L`` the
largest rate out of any state and ``P = I + Q / L`` (no negative entry),
``exp(Q / L)`` is the Poisson(1) mixture of the powers of ``P``, and
``exp(Q t)`` the product of that matrix squared once for each bit of
``floor(L t)`` and the Poisson(``L r``) mixture for the remainder ``r``.
Every term is a sum of products of numbers that are not negative, so the
probabilities of surviving and of having failed are each worked out to
full relative precision, however small, and at any time, however far
apart the rates are. The mean life is the expected time to absorption,
from one linear solve.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

#: The most states a chain may have; an arrangement whose chain would have
#: more (many spares of different lives, warm) is simulated.
MAX_STATES = 200
#: The Poisson terms of a step of at most one mean holding time: the
#: weight left out is below 1e-30.
_TERMS = 32

#: A spare: its rate, and the probabilities its switch works and it
#: starts.
Spare = Tuple[float, float, float]
#: A state: the operating units' rates, sorted, and the spares left.
State = Tuple[Tuple[float, ...], Tuple[Spare, ...]]


class TooManyStates(ValueError):
    """The arrangement's chain would have more than ``MAX_STATES``
    states."""


def _chain(
    rates: Sequence[float],
    k: int,
    dormancy: float,
    switching: Sequence[float],
    starting: Sequence[float],
) -> Tuple[List[State], Dict[Tuple[int, int], float]]:
    """The transient states, the first being the start, and the rates
    between them; the target ``-1`` is the arrangement failed."""
    spares = tuple(
        (float(r), float(s), float(p))
        for r, s, p in zip(rates[k:], switching, starting)
    )
    start: State = (tuple(sorted(float(r) for r in rates[:k])), spares)
    states = [start]
    index = {start: 0}
    moves: Dict[Tuple[int, int], float] = {}

    def add(source: int, target: State, rate: float) -> None:
        if rate <= 0.0:
            return
        if target not in index:
            if len(states) >= MAX_STATES:
                raise TooManyStates(
                    f"its Markov chain would have more than {MAX_STATES} "
                    "states"
                )
            index[target] = len(states)
            states.append(target)
        key = (source, index[target])
        moves[key] = moves.get(key, 0.0) + rate

    def fail(source: int, rate: float) -> None:
        if rate > 0.0:
            key = (source, -1)
            moves[key] = moves.get(key, 0.0) + rate

    i = 0
    while i < len(states):
        operating, waiting = states[i]
        # An operating unit fails: the spares are tried in order.
        for j, rate in enumerate(operating):
            if j and rate == operating[j - 1]:
                continue
            count = operating.count(rate)
            rest = list(operating)
            rest.remove(rate)
            chance = 1.0
            for m, (spare, s, p) in enumerate(waiting):
                fail(i, count * rate * chance * (1.0 - s))
                add(
                    i,
                    (tuple(sorted(rest + [spare])), waiting[m + 1 :]),
                    count * rate * chance * s * p,
                )
                chance *= s * (1.0 - p)
            # No spare left to take over.
            fail(i, count * rate * chance)
        # A spare fails while it waits.
        if dormancy > 0.0:
            for m, (spare_rate, _, _) in enumerate(waiting):
                add(
                    i,
                    (operating, waiting[:m] + waiting[m + 1 :]),
                    dormancy * spare_rate,
                )
        i += 1
    return states, moves


class MarkovStandbySurvival:
    """The exact lifetime distribution of a standby arrangement of
    exponential units (see the module docstring).

    Parameters
    ----------
    rates : sequence of float
        The units' failure rates while operating, in list order.
    k : int
        How many must operate.
    dormancy : float
        A waiting spare's failure rate as a fraction of its operating rate.
    switching, starting : sequence of float
        For each spare (``len(rates) - k``), the probability that its
        switch works and that it starts.

    Raises
    ------
    TooManyStates
        If the chain would have more than ``MAX_STATES`` states.
    """

    route = "exact"
    how = "a Markov chain of its exponential units"
    never_fails = 0.0

    def __init__(
        self,
        rates: Sequence[float],
        k: int,
        dormancy: float,
        switching: Sequence[float],
        starting: Sequence[float],
    ):
        states, moves = _chain(rates, k, dormancy, switching, starting)
        m = len(states)
        # The generator on the transient states and the failed one (last).
        Q = np.zeros((m + 1, m + 1))
        for (source, target), rate in moves.items():
            Q[source, target] += rate
        Q[np.arange(m), np.arange(m)] = -Q[:m].sum(axis=1)
        self._Q = Q
        self._states = m
        self._L = float(np.max(-np.diag(Q))) if m else 0.0
        self._steps: List[np.ndarray] = []
        if self._L > 0.0:
            P = np.eye(m + 1) + Q / self._L
            # The start's row of each power of P, for the remainders.
            row = np.zeros(m + 1)
            row[0] = 1.0
            rows = [row]
            powers = [np.eye(m + 1)]
            for _ in range(1, _TERMS):
                rows.append(rows[-1] @ P)
                powers.append(powers[-1] @ P)
            self._rows = np.array(rows)
            # exp(Q / L): the Poisson(1) mixture of the powers.
            weights = self._poisson(np.ones(1))[0]
            self._steps.append(np.einsum("n,nij->ij", weights, powers))

    @staticmethod
    def _poisson(means: np.ndarray) -> np.ndarray:
        """The Poisson probabilities of 0 to ``_TERMS - 1`` for each mean
        (each at most 1), one row a mean."""
        from scipy.special import gammaln

        n = np.arange(_TERMS)
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = n * np.log(means[:, None]) - means[:, None] - gammaln(n + 1)
        out = np.exp(logs)
        out[means == 0.0] = (n == 0).astype(float)
        return out

    def _step(self, bit: int) -> np.ndarray:
        """``exp(Q 2**bit / L)``, by squaring."""
        while len(self._steps) <= bit:
            last = self._steps[-1]
            self._steps.append(last @ last)
        return self._steps[bit]

    def _distribution(self, t: np.ndarray) -> np.ndarray:
        """The probability of each state (the failed one last) at each of
        the finite times ``t`` (at least 0), one row a time."""
        if self._L == 0.0:
            out = np.zeros((t.size, self._states + 1))
            out[:, 0] = 1.0
            return out
        scaled = t * self._L
        whole = np.floor(scaled)
        out = self._poisson(scaled - whole) @ self._rows
        bit = 0
        while np.any(whole > 0.0):
            odd = np.mod(whole, 2.0) == 1.0
            if odd.any():
                out[odd] = out[odd] @ self._step(bit)
            whole = np.floor(whole / 2.0)
            bit += 1
        return out

    def _both(self, x) -> Tuple[np.ndarray, np.ndarray]:
        """The survival and failure probabilities at ``x``."""
        t = np.asarray(x, dtype=float)
        flat = t.ravel()
        sf = np.ones(flat.shape)
        ff = np.zeros(flat.shape)
        sf[np.isnan(flat)] = ff[np.isnan(flat)] = np.nan
        sf[flat == np.inf], ff[flat == np.inf] = 0.0, 1.0
        inside = np.isfinite(flat) & (flat > 0.0)
        if inside.any():
            p = self._distribution(flat[inside])
            sf[inside] = np.clip(p[:, :-1].sum(axis=1), 0.0, 1.0)
            ff[inside] = np.clip(p[:, -1], 0.0, 1.0)
        return sf.reshape(t.shape), ff.reshape(t.shape)

    def sf(self, x, *args, **kwargs):
        return self._both(x)[0]

    def ff(self, x, *args, **kwargs):
        return self._both(x)[1]

    def mean(self, *args, **kwargs) -> float:
        """The expected time to absorption from the start."""
        m = self._states
        if m == 0:
            return 0.0
        times = np.linalg.solve(-self._Q[:m, :m], np.ones(m))
        return float(times[0])
