from queue import PriorityQueue

import numpy as np
from scipy.stats import gamma as _gamma
from surpyval import Hypoexponential, KaplanMeier

from repyability.utils.wrappers import numpy_seed

from ._model_utils import is_exponential
from ._sampling import RowSampler, column, draw_rows, inverse_sampler
from .numerical_convolution import (
    ConvolvedSurvival,
    is_perfect_switching,
    switch_success_probs,
)


def _identical_exponential_rate(models):
    """If every model is an Exponential with the same rate, return that rate;
    otherwise return None. The rate is taken as 1 / mean."""
    rate = None
    for model in models:
        if not is_exponential(model):
            return None
        mean = float(model.mean())
        if mean <= 0.0:
            return None
        this_rate = 1.0 / mean
        if rate is None:
            rate = this_rate
        elif not np.isclose(this_rate, rate):
            return None
    return rate


class _ExponentialStandbySurvival:
    """Exact survival function of an identical-exponential k-out-of-n cold
    standby arrangement.

    With N identical units of rate ``rate``, k operating at a time, the system
    fails at the (N-k+1)-th failure. While k units operate the failure rate is
    ``k*rate``, and by the memorylessness of the exponential the inter-failure
    times are i.i.d. Exponential(k*rate). Hence the lifetime is exactly
    Erlang(N-k+1, k*rate) -- deterministic, for any k.
    """

    def __init__(self, rate, n_units, k):
        self.shape = n_units - k + 1  # number of inter-failure gaps
        self.rate = k * rate  # failure rate while k units operate

    def sf(self, x):
        return _gamma.sf(x, a=self.shape, scale=1.0 / self.rate)

    def ff(self, x):
        return _gamma.cdf(x, a=self.shape, scale=1.0 / self.rate)

    def mean(self, *args, **kwargs):
        return self.shape / self.rate


class StandbyModel:
    """A k-out-of-n standby arrangement, from cold through warm to hot.

    ``n = len(reliabilities)`` units with ``k`` operating at a time: the
    first ``k`` units in the list start operating, spares are promoted in
    list order as operating units fail, and the arrangement fails when
    fewer than ``k`` units survive. ``dormancy_factor`` sets how fast a
    *dormant* spare ages relative to an operating unit (the
    cumulative-exposure model, the same virtual-age machinery as
    [`LoadSharingModel`][repyability.LoadSharingModel]):

    - ``0`` — **cold** standby (the default): spares do not age while
      dormant. With ``k = 1`` the lifetime is the sum of the units'
      lifetimes.
    - ``0 < dormancy_factor < 1`` — **warm** standby: a dormant spare ages
      at that fraction of the operating rate, so it can fail *latent*
      (dead before it is ever switched in; it is skipped at promotion).
    - ``1`` — **hot** standby: spares age as fast as operating units,
      which is exactly an ordinary k-out-of-n parallel arrangement.

    The survival function is set up once, at construction, by the first
    of these methods that applies:

    1. **Exact closed form**, for identical Exponential units (one common
       rate) with perfect switching. Cold, the lifetime is
       ``Erlang(n - k + 1, k * rate)`` for any ``k``; warm or hot, it is
       hypoexponential with stage rates
       ``rate * (k + (j - k) * dormancy_factor)`` for ``j = n, ..., k``
       units alive (simulated instead if the stage rates are too close
       for surpyval to separate, as with a tiny ``dormancy_factor``).
    2. **Numerical convolution**, for cold standby with ``k = 1`` and any
       lifetime models. The lifetime is the sum of the units' lifetimes
       (under imperfect switching, a mixture of partial sums), and its
       survival function is computed deterministically by convolving the
       units' densities on a time grid.
    3. **Simulation** otherwise (cold with ``k >= 2``, or warm/hot, with
       units that are not identical Exponentials): ``n_sims`` lifetimes
       are drawn with ``random`` and ``sf`` is a Kaplan-Meier fit to them,
       a step function that is reproducible only with ``seed``. The fit
       is kept in ``model``.

    As an RBD node, ``sf``/``ff`` give its reliability, ``random`` its
    lifetimes for Monte-Carlo system simulation and ``mean`` its MTTF.

    Parameters
    ----------
    reliabilities : sequence of lifetime models
        The units' lifetime distributions (e.g. surpyval distributions),
        in promotion order. The same model object may be listed several
        times; each entry is an independent unit.
    k : int, optional
        The number of units that must operate for the arrangement to work,
        from 1 to ``len(reliabilities)``, by default 1.
    n_sims : int, optional
        The number of simulated lifetimes behind the Kaplan-Meier fit, by
        default 10_000. Used only when the arrangement is simulated.
    lower : float, optional
        The ``set_lower_limit`` of that Kaplan-Meier fit: a point with
        survival 1 is added there, so ``sf`` is 1 from ``lower`` up to the
        first simulated failure. By default -inf. Used only when simulated.
    switching_probability : float or sequence of float, optional
        The probability, in ``[0, 1]``, that switching onto the next spare
        succeeds: a scalar for every switch, or one value per switch
        (length ``len(reliabilities) - 1``). A failed switch ends the
        arrangement's life when the unit it should replace fails. By
        default 1.0 (perfect switching). Imperfect switching is supported
        for cold standby with ``k = 1`` only.
    seed : int or None, optional
        Seed for the simulation, making the fit reproducible: numpy's
        global RNG is seeded for the draw and restored afterwards. By
        default None (not reproducible). Used only when simulated.
    dormancy_factor : float, optional
        The dormant-to-operating aging ratio, in ``[0, 1]``: 0 is cold, 1
        is hot and anything between is warm. By default 0.0.

    Attributes
    ----------
    N : int
        The number of units, ``len(reliabilities)``.
    model : surpyval NonParametric or None
        The Kaplan-Meier fit to the simulated lifetimes when the
        arrangement is simulated; None when ``sf`` is an exact closed form
        or a numerical convolution.

    Raises
    ------
    ValueError
        If ``k`` exceeds ``len(reliabilities)``, ``dormancy_factor`` is
        outside ``[0, 1]``, or (cold, ``k = 1``) a switching probability is
        outside ``[0, 1]`` or a sequence of them has the wrong length.
    NotImplementedError
        If ``switching_probability`` is not 1 while ``k >= 2`` or
        ``dormancy_factor > 0``.

    Examples
    --------
    One operating unit and one cold spare, both Exponential with mean life
    100: the lifetimes add, and the exact Erlang form applies.

    >>> import surpyval as surv
    >>> from repyability import StandbyModel
    >>> unit = surv.Exponential.from_params([0.01])
    >>> cold = StandbyModel([unit, unit])
    >>> round(float(cold.mean()), 4)
    200.0
    >>> round(float(cold.sf(100.0)), 4)  # exp(-1) * (1 + 1)
    0.7358

    A warm spare aging at half the operating rate, and a hot spare (an
    ordinary parallel pair):

    >>> warm = StandbyModel([unit, unit], dormancy_factor=0.5)
    >>> round(float(warm.mean()), 4)  # 1 / (1.5 * 0.01) + 1 / 0.01
    166.6667
    >>> hot = StandbyModel([unit, unit], dormancy_factor=1.0)
    >>> round(float(hot.mean()), 4)  # 1 / (2 * 0.01) + 1 / 0.01
    150.0

    Two of three Weibull units must operate, so the arrangement is
    simulated; ``seed`` makes the fit reproducible and a small ``n_sims``
    keeps it quick:

    >>> w = surv.Weibull.from_params([100.0, 2.0])
    >>> sim = StandbyModel([w, w, w], k=2, n_sims=2000, seed=1)
    >>> sim.model is not None
    True
    >>> round(float(sim.sf([50.0])[0]), 2)
    0.94

    As a node of an RBD:

    >>> from repyability import NonRepairableRBD
    >>> rbd = NonRepairableRBD(
    ...     [("s", "pumps"), ("pumps", "t")], {"pumps": cold}
    ... )
    >>> round(rbd.sf(100.0), 4)
    0.7358
    """

    def __init__(
        self,
        reliabilities,
        k=1,
        n_sims=10_000,
        lower=-np.inf,
        switching_probability=1.0,
        seed=None,
        dormancy_factor=0.0,
    ):
        if k > len(reliabilities):
            raise ValueError(
                "Must be more nodes in the standby arrangement"
                + " than are required (k)"
            )
        if not (0.0 <= float(dormancy_factor) <= 1.0):
            raise ValueError(
                "dormancy_factor is the dormant-to-operating aging ratio, in "
                f"[0, 1] (0 = cold, 1 = hot); got {dormancy_factor!r}."
            )
        self.reliabilities = reliabilities
        self.k = k
        self.N = len(reliabilities)
        self.n_sims = n_sims
        self.switching_probability = switching_probability
        self.dormancy_factor = float(dormancy_factor)

        rate = _identical_exponential_rate(reliabilities)
        if self.dormancy_factor > 0.0:
            # Warm/hot standby: dormant aging couples the spares' clocks to
            # elapsed time, so the cold-only machinery (lifetime sums /
            # convolution) does not apply.
            if not is_perfect_switching(switching_probability):
                raise NotImplementedError(
                    "switching_probability is only supported for cold "
                    "standby (dormancy_factor == 0) with k == 1."
                )
            closed_form = None
            if rate is not None:
                # Identical Exponential units: with j units alive, k operate
                # at rate `rate` and j - k sit dormant at `dormancy_factor *
                # rate`, and by memorylessness the stage durations are
                # independent Exponentials. The lifetime is hypoexponential
                # with those stage rates; dormancy_factor == 1 gives j*rate,
                # the ordinary k-out-of-n parallel order statistic.
                stage_rates = [
                    rate * (self.k + (j - self.k) * self.dormancy_factor)
                    for j in range(self.N, self.k - 1, -1)
                ]
                try:
                    closed_form = Hypoexponential.from_params(stage_rates)
                except ValueError:
                    # A tiny dormancy factor leaves the stage rates too close
                    # for surpyval to separate; simulate instead.
                    closed_form = None
            if closed_form is not None:
                self._sf_model = closed_form
                self.model = None
            else:
                x_random = self.random(n_sims, seed=seed)
                self.model = KaplanMeier.fit(x_random, set_lower_limit=lower)
                self._sf_model = None
        elif rate is not None and is_perfect_switching(switching_probability):
            # Identical exponential units: the cold standby lifetime is exactly
            # Erlang(N-k+1, k*rate) for any k, by the memorylessness of the
            # exponential. Use that closed form directly.
            self._sf_model = _ExponentialStandbySurvival(rate, self.N, k)
            self.model = None
        elif k == 1:
            # Cold standby (k=1): the lifetime is the sum of the components'
            # lifetimes (or a mixture of partial sums under imperfect
            # switching), whose survival function is computed deterministically
            # by numerical convolution rather than from Monte-Carlo samples.
            self._sf_model = ConvolvedSurvival(
                reliabilities, switching_probability=switching_probability
            )
            self.model = None
        else:
            # For k >= 2 with general (non-exponential) lifetimes the lifetime
            # is not a simple sum (it depends on the order in which components
            # fail), so fall back to the Monte-Carlo + Kaplan-Meier
            # approximation. Imperfect switching is not modelled in that case
            # yet.
            if not is_perfect_switching(switching_probability):
                raise NotImplementedError(
                    "switching_probability is only supported for k=1 cold"
                    " standby; for k>=2 leave it at 1.0 (perfect switching)."
                )
            x_random = self.random(n_sims, seed=seed)
            self.model = KaplanMeier.fit(x_random, set_lower_limit=lower)
            self._sf_model = None

    def _random_warm(self, size):
        """Warm-standby lifetimes by cumulative exposure (virtual age).

        Each unit draws the operating age at which it would fail. Operating
        units age at rate 1 and dormant spares at ``dormancy_factor``; spares
        are promoted in list order, and a spare whose dormant aging exhausts
        its budget dies latent and is skipped. The system lifetime is the
        instant the number of surviving units drops below ``k``.
        """
        budgets = np.column_stack(
            [
                np.asarray(model.random(size), dtype=float)
                for model in self.reliabilities
            ]
        )
        return self._warm_from_budgets(budgets)

    def _warm_from_budgets(self, budgets):
        """Warm-standby lifetimes from the units' budgets (one row per
        sample)."""
        if np.all(np.isfinite(budgets)):
            return self._warm_lifetimes(budgets)
        return self._warm_lifetimes_by_sample(budgets)

    def _warm_lifetimes(self, budgets):
        """The virtual-age loop for every sample at once.

        Each step fails exactly one unit in every sample (after ``N - k + 1``
        steps fewer than ``k`` remain), so all samples stay in step and one
        array operation per step does what the per-sample loop does, with the
        same arithmetic and the same tie-break (the lowest-indexed healthy
        unit). Requires finite budgets, so that the failing unit always has a
        finite remaining time.
        """
        size, n = budgets.shape
        k, kappa = self.k, self.dormancy_factor
        rows = np.arange(size)
        age = np.zeros((size, n))
        alive = np.ones((size, n), dtype=bool)
        t = np.zeros(size)
        for _ in range(n - k + 1):
            # The first k healthy units (in list order) operate; the rest of
            # the healthy ones are dormant.
            rank = np.cumsum(alive, axis=1)
            operating = alive & (rank <= k)
            dormant = alive & (rank > k)
            remaining = np.full((size, n), np.inf)
            remaining[operating] = budgets[operating] - age[operating]
            remaining[dormant] = (budgets[dormant] - age[dormant]) / kappa
            idx = np.argmin(remaining, axis=1)
            dt = remaining[rows, idx]
            t += dt
            age[operating] += np.broadcast_to(dt[:, None], age.shape)[
                operating
            ]
            age[dormant] += np.broadcast_to((kappa * dt)[:, None], age.shape)[
                dormant
            ]
            alive[rows, idx] = False
        return t

    def _warm_lifetimes_by_sample(self, budgets):
        """The virtual-age loop, one sample at a time (any budgets)."""
        size = budgets.shape[0]
        kappa = self.dormancy_factor
        out = np.empty(size, dtype=float)
        for s in range(size):
            X = budgets[s]
            age = np.zeros(self.N)
            alive = np.ones(self.N, dtype=bool)
            t = 0.0
            while True:
                healthy = np.flatnonzero(alive)
                if healthy.size < self.k:
                    break
                operating = healthy[: self.k]
                dormant = healthy[self.k :]  # noqa: E203
                remaining = np.concatenate(
                    [
                        X[operating] - age[operating],
                        (X[dormant] - age[dormant]) / kappa,
                    ]
                )
                idx = int(np.argmin(remaining))
                dt = float(remaining[idx])
                t += dt
                age[operating] += dt
                age[dormant] += kappa * dt
                alive[int(np.concatenate([operating, dormant])[idx])] = False
            out[s] = t
        return out

    def random(self, size, seed=None):
        """Draw random lifetimes of the arrangement.

        Each draw samples every unit's lifetime and plays the arrangement
        out. Cold with ``k = 1``, the lifetimes are summed; under imperfect
        switching a spare adds its lifetime only if every switch up to and
        including its own succeeds (one uniform draw per switch). Cold with
        ``k >= 2``, each spare in turn extends whichever of the ``k``
        operating positions ends first, and the arrangement fails at the
        first end after the spares run out. Warm or hot, a virtual-age loop
        runs: operating units age at rate 1 and dormant spares at
        ``dormancy_factor``, and a spare whose dormant aging reaches its
        failure age dies latent and is skipped. This is the sampler behind
        the simulated ``sf`` and an RBD's Monte-Carlo ``random``/``mean``.

        Parameters
        ----------
        size : int
            The number of lifetimes to draw.
        seed : int or None, optional
            If given, numpy's global RNG is seeded for the draw and its
            previous state restored afterwards, so the result is
            reproducible. By default None (draw from the current state).

        Returns
        -------
        numpy.ndarray
            The ``size`` lifetimes, shape ``(size,)``.

        Examples
        --------
        >>> import numpy as np
        >>> import surpyval as surv
        >>> from repyability import StandbyModel
        >>> unit = surv.Exponential.from_params([0.01])
        >>> standby = StandbyModel([unit, unit])
        >>> standby.random(5, seed=1).shape
        (5,)
        >>> a = standby.random(5, seed=1)
        >>> bool(np.array_equal(a, standby.random(5, seed=1)))
        True
        """
        with numpy_seed(seed):
            if self.dormancy_factor > 0.0:
                return self._random_warm(size)
            if self.k == 1:
                # If k is only one for the standby node the reliability can be
                # estimated from the sum of each of the components in the node,
                # i.e. it will fail after all of them fail.
                x_random = np.asarray(
                    self.reliabilities[0].random(size), dtype=float
                )
                if is_perfect_switching(self.switching_probability):
                    for model in self.reliabilities[1:]:
                        x_random = x_random + model.random(size)
                else:
                    # Under imperfect switching a spare only contributes if
                    # every switch up to and including its own has succeeded.
                    probs = switch_success_probs(
                        self.switching_probability, self.N
                    )
                    running = np.ones(size, dtype=bool)
                    for model, p in zip(self.reliabilities[1:], probs):
                        running = running & (np.random.random(size) < p)
                        x_random = x_random + np.where(
                            running, model.random(size), 0.0
                        )

            else:
                # If k are required to continue then a random draw needs
                # a little more complexity. An individual run instance
                # can be simulated by getting failures for the first k
                # components. The simulation then tracks the k active
                # components. This is done by adding the next random
                # instance to the lowest of the k active components.
                # This is because the next standby component will start
                # working once the next of the k fails. This means the
                # lowest of the k active components will be the next
                # to fail. By simply adding the standby failures to the
                # lowest of the k active, at the end of the simulation
                # the lowest value in the queue will be the standby nodes
                # failure time. This simulation is repeated size
                # number of times and then the model is approximated
                # with a non parametric estimate.
                x_random = self._random_cold_k(size)
        return x_random

    def _random_cold_k(self, size):
        """Cold standby with k >= 2: every sample at once when each unit's
        draws can be reproduced exactly in one block (see ``_sampling``),
        otherwise sample by sample."""
        samplers = [inverse_sampler(m) for m in self.reliabilities]
        if all(sampler is not None for sampler in samplers):
            state = np.random.get_state()
            # One draw per unit per sample, in list order: the order the
            # sample-by-sample loop draws them in.
            lifetimes = draw_rows(samplers, size)
            if not any(np.isnan(x).any() for x in lifetimes):
                return self._cold_k_lifetimes(lifetimes)
            # NaN times order differently in the queue; rewind and loop.
            np.random.set_state(state)

        x_random = np.zeros(size)
        for i in range(size):
            pq: PriorityQueue = PriorityQueue()
            # start k streams:
            for node in self.reliabilities[: self.k]:
                pq.put(node.random(1).item())

            # Add the next event time to the lowest value in the queue
            for node in self.reliabilities[self.k :]:  # noqa: E203
                next_t = node.random(1).item()
                current_lowest = pq.get()
                pq.put(current_lowest + next_t)

            x_random[i] = pq.get()
        return x_random

    def _cold_k_lifetimes(self, lifetimes):
        """Cold standby with k >= 2 for every sample at once, from each
        unit's lifetimes (a list of arrays, one per unit): the queue loop's
        stream arithmetic, one array operation per spare."""
        streams = np.column_stack(lifetimes[: self.k])
        rows = np.arange(len(streams))
        for spare in lifetimes[self.k :]:  # noqa: E203
            # The spare extends whichever stream ends first (a tie between
            # equal ends gives the same result either way).
            lowest = np.argmin(streams, axis=1)
            streams[rows, lowest] = streams[rows, lowest] + spare
        return streams.min(axis=1)

    def _row_sampler(self):
        """``random(1)`` as a :class:`~._sampling.RowSampler`, so an RBD with
        this node batches its draws; ``None`` unless every unit's draws can
        be replayed. Columns follow the order ``random(1)`` draws in: one per
        unit, and under imperfect k=1 switching a switch draw before each
        spare's."""
        units = [inverse_sampler(m) for m in self.reliabilities]
        if any(unit is None for unit in units):
            return None

        if self.dormancy_factor > 0.0:

            def draw(u):
                budgets = np.column_stack(
                    [column(u, j, unit) for j, unit in enumerate(units)]
                )
                return self._warm_from_budgets(budgets)

            return RowSampler(self.N, draw)

        if self.k == 1 and is_perfect_switching(self.switching_probability):

            def draw(u):
                x = column(u, 0, units[0])
                for j in range(1, self.N):
                    x = x + column(u, j, units[j])
                return x

            return RowSampler(self.N, draw)

        if self.k == 1:
            probs = switch_success_probs(self.switching_probability, self.N)
            spares = list(zip(units[1:], probs))

            def draw(u):
                x = column(u, 0, units[0])
                running = np.ones(len(u), dtype=bool)
                for i, (unit, p) in enumerate(spares):
                    running = running & (u[:, 1 + 2 * i] < p)
                    x = x + np.where(running, column(u, 2 + 2 * i, unit), 0.0)
                return x

            return RowSampler(1 + 2 * len(spares), draw)

        def draw(u):
            lifetimes = [column(u, j, unit) for j, unit in enumerate(units)]
            out = self._cold_k_lifetimes(lifetimes)
            # The queue loop orders NaN times its own way; mark the sample
            # so the caller falls back to it.
            for lifetime in lifetimes:
                out[np.isnan(lifetime)] = np.nan
            return out

        return RowSampler(self.N, draw)

    def mean(self, N=10_000, seed=None):
        """Mean lifetime (MTTF) of the arrangement.

        Exact for the Erlang and hypoexponential closed forms, and
        deterministic for the numerical convolution (the integral of its
        survival function). When the arrangement is simulated it is a
        Monte-Carlo estimate: the mean of ``N`` fresh draws of ``random``,
        not the mean of the Kaplan-Meier fit behind ``sf``.

        Parameters
        ----------
        N : int, optional
            The number of draws for the Monte-Carlo estimate, by default
            10_000. Ignored unless the arrangement is simulated.
        seed : int or None, optional
            Seed for those draws (see ``random``), by default None.
            Ignored unless the arrangement is simulated.

        Returns
        -------
        float
            The mean lifetime.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import StandbyModel
        >>> unit = surv.Exponential.from_params([0.01])
        >>> round(float(StandbyModel([unit] * 3).mean()), 4)
        300.0
        >>> w = surv.Weibull.from_params([100.0, 2.0])
        >>> sim = StandbyModel([w, w, w], k=2, n_sims=2000, seed=1)
        >>> round(float(sim.mean(N=2000, seed=1)), 1)
        105.2
        """
        # Use the exact/deterministic mean when an analytic survival model is
        # available (exponential closed form or convolution); otherwise fall
        # back to the Monte-Carlo estimate.
        if self._sf_model is not None:
            return float(np.ravel(self._sf_model.mean())[0])
        return self.random(N, seed=seed).mean()

    def sf(self, *args, **kwargs):
        """Survival function (reliability) of the arrangement.

        Evaluates the survival function set up at construction: the exact
        closed form, the numerical convolution (linearly interpolated on
        its grid) or the Kaplan-Meier fit to simulated lifetimes (a step
        function). An RBD calls this for the node's reliability.

        Parameters
        ----------
        *args : array_like
            The time(s) ``x``, as in ``sf(x)``.
        **kwargs
            Passed on, with ``x``, to the underlying survival model.

        Returns
        -------
        float or numpy.ndarray
            The probability of surviving beyond ``x``: an array for an
            array ``x``; for a scalar ``x`` a numpy float, except when the
            arrangement is simulated, which gives a 1-element array.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import StandbyModel
        >>> unit = surv.Exponential.from_params([0.01])
        >>> standby = StandbyModel([unit, unit])
        >>> [round(float(r), 4) for r in standby.sf([50.0, 100.0])]
        [0.9098, 0.7358]
        """
        if self._sf_model is not None:
            return self._sf_model.sf(*args, **kwargs)
        return self.model.sf(*args, **kwargs)

    def ff(self, *args, **kwargs):
        """Cumulative failure probability, ``1 - sf(x)``.

        Evaluated from the same survival function as ``sf``.

        Parameters
        ----------
        *args : array_like
            The time(s) ``x``, as in ``ff(x)``.
        **kwargs
            Passed on, with ``x``, to the underlying survival model.

        Returns
        -------
        float or numpy.ndarray
            The probability of failing by ``x``, shaped as for ``sf``.
        """
        if self._sf_model is not None:
            return self._sf_model.ff(*args, **kwargs)
        return self.model.ff(*args, **kwargs)

    def cs(self, x, X):
        """Conditional survival ``R(x | X) = sf(X + x) / sf(X)``.

        The probability of surviving a further ``x`` given the arrangement
        has already survived to age ``X``, computed from ``sf``.

        Parameters
        ----------
        x : float or array_like
            The further time(s) to survive.
        X : float or array_like
            The age(s) already survived.

        Returns
        -------
        float or numpy.ndarray
            The conditional survival, clipped to ``[0, 1]`` and 0 where
            ``sf(X)`` is 0: a float if ``x`` and ``X`` are both scalars,
            otherwise an array.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import StandbyModel
        >>> unit = surv.Exponential.from_params([0.01])
        >>> round(StandbyModel([unit, unit]).cs(50.0, 100.0), 4)
        0.7582
        """
        from repyability.utils.wrappers import conditional_survival

        return conditional_survival(self, x, X)
