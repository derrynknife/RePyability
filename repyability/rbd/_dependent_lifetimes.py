"""Lifetime distributions of standby and load-sharing arrangements, worked
out without simulation (#135, #138, #139).

``StandbyModel`` and ``LoadSharingModel`` use these where no closed form
applies (until 0.12 a Kaplan-Meier fit to simulated lifetimes stood in):

- :class:`KOutOfNSurvival`, exact: hot standby is active k-out-of-n
  redundancy, so the arrangement works while at least ``k`` units survive,
  each independently. The number of failed units has a Poisson-binomial
  distribution, worked out at each time from the units' own ``sf`` and
  ``ff`` (sums of products, so both tails keep their precision).
- :class:`WarmStandbySurvival`, numerical: one unit operating and warm
  spares that age at ``dormancy_factor`` times the operating rate while
  they wait. A recursion over the switch-ins on a time grid, in cumulative
  probabilities.
- :class:`RenewalStandbySurvival`, numerical: cold standby of identical
  units with ``k`` operating. Each operating position is a renewal process
  of the units' lives, so the number of failures by ``t`` is a sum of
  ``k`` independent renewal counts, whose distributions come from the
  partial sums' convolution (``ConvolvedSurvival``).
- :class:`LoadSharingSurvival`, numerical: identical accelerated-failure-
  time units sharing a load. They all age alike, so they fail in the order
  of their exposures to failure; a recursion over the failures on a grid
  of exposure and time.

Each has ``sf``, ``ff`` and ``mean`` and says how it is computed
(``route``, ``how``) for ``analysis_routes``.
"""

from typing import List, Sequence

import numpy as np

from ._model_utils import never_fails
from .numerical_convolution import ConvolvedSurvival, _cdf, _upper_time

#: Points of the time grid of a warm-standby arrangement: the error is
#: about 1e-5 or less of a probability (smaller for smooth lives).
WARM_POINTS = 2001
#: Points of each axis of a load-sharing arrangement's grid of exposure and
#: time.
SHARING_POINTS = 1001


def _grid(upper: float, n_points: int) -> np.ndarray:
    """A time grid from 0 to ``upper`` of about ``n_points`` points: half
    spread evenly, half geometrically from ``upper * 1e-8``, so that the
    early times, where lives with a steep start change fastest, are as well
    resolved as the body."""
    even = np.linspace(0.0, upper, n_points // 2 + 1)
    geometric = np.geomspace(upper * 1e-8, upper, n_points - n_points // 2)
    return np.unique(np.concatenate((even, geometric)))


def _sf(model, x: np.ndarray) -> np.ndarray:
    """The model's survival function at ``x``, as a flat float array."""
    with np.errstate(divide="ignore"):
        values = np.asarray(model.sf(x), dtype=float)
    return np.clip(np.nan_to_num(np.ravel(values), nan=1.0), 0.0, 1.0)


class _Gridded:
    """A survival function kept on a time grid: ``sf`` and ``ff``
    interpolate it linearly, and ``mean`` is its area."""

    route = "numerical"
    how = ""
    _t: np.ndarray
    _sf: np.ndarray
    _ff: np.ndarray
    #: The probability that the arrangement never fails.
    never_fails: float = 0.0

    def sf(self, x, *args, **kwargs):
        return np.interp(
            x, self._t, self._sf, left=1.0, right=self.never_fails
        )

    def ff(self, x, *args, **kwargs):
        return np.interp(
            x, self._t, self._ff, left=0.0, right=1.0 - self.never_fails
        )

    def mean(self, *args, **kwargs) -> float:
        from scipy.integrate import trapezoid

        if self.never_fails > 0.0:
            return float("inf")
        return float(trapezoid(self._sf, self._t))


class KOutOfNSurvival:
    """Exact lifetime distribution of an active k-out-of-n group: it works
    while at least ``k`` of its independent units do (hot standby).

    At each time the number of failed units has a Poisson-binomial
    distribution over the units' own probabilities of failing and
    surviving; ``sf`` sums its probabilities of at most ``n - k`` failed,
    and ``ff`` of more, each a sum of products, so a small one keeps its
    precision.
    """

    route = "exact"
    how = "k-out-of-n of its units' lives"

    def __init__(self, models: Sequence, k: int):
        self.models = list(models)
        self.k = int(k)

    def _counts(self, x) -> np.ndarray:
        """The probabilities of 0, 1, ..., n units failed at each of the
        times ``x`` (one row per count)."""
        times = np.atleast_1d(np.asarray(x, dtype=float))
        counts = np.zeros((len(self.models) + 1, times.size))
        counts[0] = 1.0
        for model in self.models:
            q = np.broadcast_to(_cdf(model, times), times.shape)
            p = np.broadcast_to(_sf(model, times), times.shape)
            shifted = np.zeros_like(counts)
            shifted[1:] = counts[:-1] * q
            counts = counts * p + shifted
        return counts

    def _shaped(self, values: np.ndarray, x):
        return values if np.ndim(x) else values.reshape(-1)[0]

    def sf(self, x, *args, **kwargs):
        allowed = len(self.models) - self.k
        return self._shaped(self._counts(x)[: allowed + 1].sum(axis=0), x)

    def ff(self, x, *args, **kwargs):
        allowed = len(self.models) - self.k
        return self._shaped(self._counts(x)[allowed + 1 :].sum(axis=0), x)

    def mean(self, *args, **kwargs) -> float:
        from ._mean_lifetime import mean_lifetime, model_knots

        knots = np.concatenate(
            [np.empty(0)] + [model_knots(m) for m in self.models]
        )
        return mean_lifetime(lambda t: self.sf(t), knots)


class WarmStandbySurvival(_Gridded):
    """Numerical lifetime distribution of one operating unit with warm
    spares, which age at ``dormancy_factor`` (``kappa``) times the operating
    rate while dormant and are switched in, in list order, when the
    operating unit fails; a spare that has failed while dormant is passed
    over. The arrangement fails when a unit fails and no later spare is
    left working.

    A spare switched in at ``tau`` has dormant age ``kappa * tau``, and ends
    at ``tau + X - kappa * tau`` for its life ``X`` (if ``X`` is more than
    its dormant age). So, on a time grid of cells ``(t_j, t_j+1]``, each
    unit's switch-in and end times are worked out in turn, as cumulative
    probabilities: the probability that unit ``m`` is switched in during a
    cell is that of an earlier unit ending there with the units between
    dead at the cell's midpoint ``s``, and it then ends by ``t`` with
    probability ``F_m(t - (1 - kappa) s) - F_m(kappa s)``. The arrangement
    is working at ``t`` while some unit runs, ``sum over m and s of
    P(switch at s) S_m(t - (1 - kappa) s)``, and has failed once a unit
    ends with every later spare dead: the two are worked out separately,
    each a sum of products, and add to 1. With two units this is
    ``S_1(t) + integral of f_1(u) S_2(t - (1 - kappa) u)`` from 0 to t.
    The grid is half even and half geometric (see ``_grid``), and the
    error falls as the square of its step; units' CDFs are interpolated
    from a grid 100 times as fine.
    """

    how = "a numerical recursion over its spares' switch-ins"

    def __init__(
        self,
        models: Sequence,
        dormancy_factor: float,
        n_points: int = WARM_POINTS,
        eps: float = 1e-10,
    ):
        models = list(models)
        kappa = float(dormancy_factor)
        c = 1.0 - kappa
        upper = sum(_upper_time(model, eps) for model in models)
        t = _grid(upper, n_points)
        n_points = t.size
        mid = 0.5 * (t[:-1] + t[1:])
        fine = _grid(upper, 100 * n_points)
        cdfs = [_cdf(model, fine) for model in models]
        sfs = [_sf(model, fine) for model in models]

        def F(i: int, x):
            return np.interp(x, fine, cdfs[i], right=cdfs[i][-1])

        def S(i: int, x):
            return np.interp(x, fine, sfs[i], right=sfs[i][-1])

        # Each spare's probability of having died dormant by each cell's
        # midpoint, and by 0.
        dead = [F(i, kappa * mid) for i in range(len(models))]
        dead_at_0 = [float(F(i, 0.0)) for i in range(len(models))]

        def after(last: int, upto: int):
            """The probability that the spares after ``last`` and before
            ``upto`` are all dead, at each cell's midpoint and at 0."""
            cells, zero = np.ones(n_points - 1), 1.0
            for i in range(last + 1, upto):
                cells = cells * dead[i]
                zero *= dead_at_0[i]
            return cells, zero

        # Unit 1 runs from 0: its end's probability in each cell, and at 0
        # (dead on arrival).
        ends: List[np.ndarray] = [np.diff(F(0, t))]
        ends_at_0 = [dead_at_0[0]]
        running = S(0, t).copy()
        for m in range(1, len(models)):
            switch = np.zeros(n_points - 1)
            switch_at_0 = 0.0
            for last in range(m):
                cells, zero = after(last, m)
                switch += ends[last] * cells
                switch_at_0 += ends_at_0[last] * zero
            ended = switch_at_0 * (F(m, t) - dead_at_0[m])
            runs = switch_at_0 * S(m, t)
            for j in np.flatnonzero(switch):
                later = t[j + 1 :] - c * mid[j]  # noqa: E203
                ended[j + 1 :] += switch[j] * (  # noqa: E203
                    F(m, later) - dead[m][j]
                )
                runs[j + 1 :] += switch[j] * S(m, later)  # noqa: E203
            ends.append(np.diff(ended))
            ends_at_0.append(0.0)
            running += runs

        failed = np.zeros(n_points)
        for last in range(len(models)):
            cells, zero = after(last, len(models))
            failed[1:] += np.cumsum(ends[last] * cells)
            failed += ends_at_0[last] * zero

        self._t = t
        self._sf = np.clip(running, 0.0, 1.0)
        self._ff = np.clip(failed, 0.0, 1.0)
        self.never_fails = (
            float(self._sf[-1])
            if any(never_fails(model) > 0.0 for model in models)
            else 0.0
        )


class RenewalStandbySurvival(_Gridded):
    """Numerical lifetime distribution of cold standby of ``n`` identical
    units with ``k`` operating.

    A spare goes into the position of the unit that failed and starts new,
    so each of the ``k`` positions runs a renewal process of the units'
    lives, independently of the others, and the arrangement fails at the
    ``n - k + 1``-th failure in all. One position has failed at least ``a``
    times by ``t`` when the sum of ``a`` lives is at most ``t``: the
    convolution of the partial sums (``ConvolvedSurvival``) gives each
    position's count of failures, and the ``k`` counts add up by
    convolution. Under imperfect switching the ``j``-th switch succeeds with
    probability ``p_j`` (``switch_probs``, one per spare), and a failed
    switch ends the arrangement's life then: the arrangement works at ``t``
    with ``m`` failures so far with probability
    ``P(m failures) * p_1 * ... * p_m``.
    """

    how = (
        "a numerical convolution of its units' lives, as renewal counts of "
        "its operating positions"
    )

    def __init__(
        self,
        model,
        n: int,
        k: int,
        switch_probs: Sequence[float] = (),
        n_points: int = 100_001,
        eps: float = 1e-10,
    ):
        spares = n - k
        switch_probs = list(switch_probs) or [1.0] * spares
        sums = ConvolvedSurvival(
            [model] * (spares + 1), n_points=n_points, eps=eps, partials=True
        )
        t = sums._t
        # One position's count of failures by each time: P(N >= a) is the
        # probability that the sum of a lives has ended.
        at_least = [np.ones(t.size)] + [
            1.0 - sums.partial_sf(a, t) for a in range(1, spares + 2)
        ]
        one = np.array(
            [
                np.clip(at_least[a] - at_least[a + 1], 0.0, 1.0)
                for a in range(spares + 1)
            ]
        )
        total = one.copy()
        for _ in range(k - 1):
            added = np.zeros_like(total)
            for a in range(spares + 1):
                added[a:] += total[: spares + 1 - a] * one[a]
            total = added
        ok = np.cumprod([1.0] + switch_probs[:spares])
        works = (total * ok[:, None]).sum(axis=0)
        self._t = t
        self._sf = np.clip(works, 0.0, 1.0)
        self._ff = np.clip(1.0 - works, 0.0, 1.0)
        # Units that may never fail can leave positions running for ever:
        # past the grid the arrangement works with the probability it has
        # at the grid's end.
        self.never_fails = (
            float(self._sf[-1]) if never_fails(model) > 0.0 else 0.0
        )


class LoadSharingSurvival(_Gridded):
    """Numerical lifetime distribution of ``n`` identical accelerated-
    failure-time units sharing a load, ``k`` of which must survive.

    With ``s`` survivors each carries ``L / s`` and its exposure grows at
    ``phi_s``, the same for every survivor, so all the survivors have one
    exposure ``v`` and the units fail in the order of their exposures to
    failure (their baseline lives, ``X_(1) < X_(2) < ...``). The ``i``-th
    failure comes at time ``sum over j <= i of (X_(j) - X_(j-1)) / phi``
    with ``n - j + 1`` survivors. Given ``X_(i) = v``, the others are
    beyond ``v`` independently, so the next failure's exposure ``v'`` has
    density ``(n - i) f(v') S(v')^(n-i-1) / S(v)^(n-i)``.

    The recursion runs on the joint distribution of exposure and time after
    each failure, leaving out the survivors' factor ``S(v)^(n-i)`` (so that
    a step is a cumulative sum, not a ratio): exposure in cells of equal
    baseline probability, and time cumulative on an even grid. During a
    stage, ``t - v / phi`` does not change, so in that coordinate a step
    adds up the cells below each exposure; the arrangement has failed by
    ``t`` with the ``n - k + 1``-th failure, weighted by its survivors'
    ``S(v)^(k-1)``.
    """

    how = "a numerical recursion over its units' failures"

    def __init__(
        self,
        baseline,
        phis: Sequence[float],
        n: int,
        k: int,
        n_points: int = SHARING_POINTS,
        eps: float = 1e-10,
    ):
        # phis[j]: the exposure rate in stage j, with n - j survivors.
        phis = [float(p) for p in phis]
        failures = n - k + 1
        top = _upper_time(baseline, eps)
        reach = float(_cdf(baseline, np.array([top]))[0])
        floor = float(_cdf(baseline, np.array([0.0]))[0])
        # Exposure cells: edges of equal baseline probability, for the body,
        # and on the even-and-geometric grid, for the tails and the start.
        even_probability = np.asarray(
            baseline.qf(np.linspace(floor, reach, n_points // 2)[1:-1]),
            dtype=float,
        ).ravel()
        edges = np.unique(
            np.concatenate(
                (
                    _grid(top, n_points - n_points // 2),
                    even_probability[np.isfinite(even_probability)],
                )
            )
        )
        edges = edges[(edges >= 0.0) & (edges <= top)]
        probability = _cdf(baseline, edges)
        probability[0] = floor
        mass = np.diff(probability)
        # Each cell's middle, by probability (its median).
        middle = np.asarray(
            baseline.qf(
                np.clip(0.5 * (probability[:-1] + probability[1:]), 0.0, 1.0)
            ),
            dtype=float,
        ).ravel()
        middle = np.clip(
            np.where(np.isfinite(middle), middle, edges[1:]),
            edges[:-1],
            edges[1:],
        )
        slowest = min(phis[:failures])
        t = _grid(top / slowest, n_points)

        # After the first failure: n f(v), at time v / phi_1, cumulative in
        # time over each exposure cell (exact within the cell).
        first = phis[0]
        reached = np.clip(first * t, 0.0, None)
        upto = np.minimum(edges[1:, None], reached[None, :])
        inside = upto > edges[:-1, None]
        lower = _cdf(baseline, edges[:-1])
        cdf_upto = _cdf(baseline, upto.ravel()).reshape(upto.shape)
        cumulative = np.where(
            inside, n * np.clip(cdf_upto - lower[:, None], 0.0, None), 0.0
        )
        # A unit dead on arrival fails at once.
        if floor > 0.0:
            cumulative[0] += n * floor
        for stage in range(1, failures):
            phi = phis[stage]
            survivors = n - stage
            # w = t - v / phi is fixed during the stage: each cell's
            # cumulative distribution in time, as a function of w, whose
            # changes come at about its exposure times a constant, so on a
            # grid dense near 0 both ways.
            w = np.concatenate(
                (-_grid(top / phi, n_points)[:0:-1], _grid(t[-1], n_points))
            )
            sheared = np.array(
                [
                    np.interp(w + v / phi, t, row, left=0.0)
                    for v, row in zip(middle, cumulative)
                ]
            )
            # The cells below each exposure, and half its own.
            below = np.cumsum(sheared, axis=0) - 0.5 * sheared
            cumulative = (
                survivors
                * mass[:, None]
                * np.array(
                    [
                        np.interp(t - v / phi, w, row, left=0.0)
                        for v, row in zip(middle, below)
                    ]
                )
            )
        weight = _sf(baseline, middle) ** (k - 1)
        failed = np.clip(weight @ cumulative, 0.0, 1.0)
        self._t = t
        self._ff = failed
        self._sf = np.clip(1.0 - failed, 0.0, 1.0)
        self.never_fails = (
            float(self._sf[-1]) if never_fails(baseline) > 0.0 else 0.0
        )


class ColdPairSurvival(_Gridded):
    """Numerical lifetime distribution of cold standby with two units
    operating, of any units, switched in in list order.

    Units 1 and 2 start at 0; at the ``j``-th failure unit ``j + 2`` is
    switched in (with probability ``p_j``; a failed switch, or the
    ``n - 1``-th failure, ends the arrangement). After a failure the state
    is its time, which unit is the other one operating, and when that one
    started: the newcomer starts new. On a time grid of cells, each state's
    probability is kept without the other unit's survival since it started
    (so that the steps are convolutions), for each other unit:

    - the newcomer fails first: the state's time moves on by its life (a
      convolution along the time of the failure);
    - the other fails first: the newcomer becomes the other, started at the
      failure before, and the time moves on by the other's remaining life
      (for each failure time, a convolution along the other's start).

    The probability of each failure is the states' probability with the
    other's survival put back; the arrangement has failed by ``t`` with
    the ``n - 1``-th failure, or a failed switch, by then.
    """

    how = "a numerical recursion over its spares' switch-ins"

    def __init__(
        self,
        models: Sequence,
        switch_probs: Sequence[float] = (),
        n_points: int = 1001,
        eps: float = 1e-10,
    ):
        from scipy.signal import fftconvolve

        models = list(models)
        n = len(models)
        probs = list(switch_probs) or [1.0] * (n - 2)
        # The arrangement ends at the earlier of its two positions' ends, so
        # by half the sum of the units' lives: half the time by which that
        # sum has passed with probability ``1 - eps`` (from a coarse
        # convolution), or of its upper bound if some never end.
        total = ConvolvedSurvival(models, n_points=20_001, eps=eps)
        ended = np.flatnonzero(total._sf <= eps)
        upper = 0.5 * float(total._t[ended[0]] if ended.size else total._t[-1])
        t = np.linspace(0.0, upper, n_points + 1)
        dt = t[1] - t[0]
        cells = n_points
        offsets = np.arange(cells)
        mid = (offsets + 0.5) * dt
        cdf = [
            lambda x, m=model: _cdf(m, np.asarray(x, float))
            for model in models
        ]
        surv = [
            lambda x, m=model: _sf(m, np.asarray(x, float)) for model in models
        ]

        # Each unit's chance of failing in each cell after starting at a
        # cell's midpoint (offset m cells on), in the cell it starts in
        # after it (from its midpoint), and from 0.
        step = [
            np.diff(np.concatenate(([0.0], F((offsets + 0.5) * dt))))
            for F in cdf
        ]
        rest = [F((offsets + 0.5) * dt) - F(offsets * dt) for F in cdf]
        from_zero = [np.diff(np.concatenate(([0.0], F(t[1:])))) for F in cdf]
        rest_of_cell = [F(t[1:]) - F(mid) for F in cdf]
        # Survival over m cells between midpoints, and from 0.
        lasted = [S(offsets * dt) for S in surv]
        lasted_from_zero = [S(mid) for S in surv]
        gap = np.subtract.outer(offsets, offsets)  # failure cell - start cell
        later = gap >= 0

        def failures(zero, started):
            """Each failure's probability by cell: the states' probability
            with the other unit's survival put back."""
            out = np.zeros(cells)
            for u, mass in zero.items():
                out += mass * lasted_from_zero[u]
            for u, mass in started.items():
                out += (
                    mass * np.where(later, lasted[u][np.abs(gap)], 0.0)
                ).sum(axis=1)
            return out

        # After the first failure: the other unit is 1 or 0, started at 0.
        zero = {1: from_zero[0].copy(), 0: from_zero[1].copy()}
        started: dict = {}
        failed = np.zeros(cells)
        for j in range(1, n - 1):
            mass = failures(zero, started)
            p = probs[j - 1]
            failed += (1.0 - p) * mass
            new = j + 1  # the spare switched in at the j-th failure
            zero = {u: p * m for u, m in zero.items()}
            started = {u: p * m for u, m in started.items()}
            # The newcomer fails first: the time moves on by its life.
            next_zero = {
                u: fftconvolve(m, step[new])[:cells] for u, m in zero.items()
            }
            next_started = {
                u: fftconvolve(m, step[new][:, None], axes=0)[:cells]
                for u, m in started.items()
            }
            # The other fails first: the newcomer becomes the other, started
            # at the j-th failure's cell (the column), and the failure comes
            # in a later cell (the row), or later in the same one.
            handed = np.zeros((cells, cells))
            for u, m in zero.items():
                handed += np.where(
                    gap > 0, m[None, :] * from_zero[u][:, None], 0.0
                )
                handed[offsets, offsets] += m * rest_of_cell[u]
            for u, m in started.items():
                # For each j-th failure cell a (a row of m), over the start
                # cells b: the chance of failing in cell a' is step[a' - b].
                spread = fftconvolve(m, step[u][None, :], axes=1)[:, :cells]
                handed += np.where(gap > 0, spread.T, 0.0)
                same = (m * np.where(later, rest[u][np.abs(gap)], 0.0)).sum(
                    axis=1
                )
                handed[offsets, offsets] += same
            next_started[new] = next_started.get(new, 0.0) + handed
            zero, started = next_zero, next_started
        failed += failures(zero, started)
        ff = np.concatenate(([0.0], np.cumsum(failed)))
        self._t = t
        self._ff = np.clip(ff, 0.0, 1.0)
        self._sf = np.clip(1.0 - ff, 0.0, 1.0)
        self.never_fails = (
            float(self._sf[-1])
            if any(never_fails(model) > 0.0 for model in models)
            else 0.0
        )
