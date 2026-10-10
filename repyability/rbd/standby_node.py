from queue import PriorityQueue

import numpy as np
from surpyval import Hypoexponential

from repyability.utils.checks import simulation_options, whole_number
from repyability.utils.deprecation import ignored, refuse_removed_names
from repyability.utils.wrappers import conditional_survival, numpy_seed

from ._dependent_lifetimes import (
    ColdPairSurvival,
    KOutOfNSurvival,
    RenewalStandbySurvival,
    WarmStandbySurvival,
)
from ._model_utils import is_exponential
from ._sampling import RowSampler, column, inverse_sampler
from .numerical_convolution import (
    ConvolvedSurvival,
    is_perfect_switching,
    switch_success_probs,
)


def drawn_mean(model, name: str, exact, mc_samples, seed, method) -> float:
    """A node model's ``mean(mc_samples, seed, method=...)`` (#233):
    ``exact()``, its exact or numerical mean (a NotImplementedError where it
    has none), by default and with ``method="exact"``; with
    ``method="simulate"``, or by default where it has no exact mean and
    the draws' options are given, the mean of ``mc_samples`` draws of
    ``random`` (10_000 by default). Options given to an exact mean by
    default are ignored, with a ``FutureWarning``: 0.14 refuses them."""
    if method not in (None, "exact", "simulate"):
        raise ValueError(
            f"method must be 'exact' or 'simulate' (or None), got {method!r}."
        )
    options = {"mc_samples": mc_samples, "seed": seed}
    count = 10_000 if mc_samples is None else mc_samples
    if method == "simulate":
        return float(model.random(count, seed=seed).mean())
    try:
        value = exact()
    except NotImplementedError:
        if method == "exact" or (mc_samples is None and seed is None):
            raise
        return float(model.random(count, seed=seed).mean())
    if method == "exact":
        simulation_options(f"{name}.mean(method='exact')", options)
    else:
        ignored(
            f"{name}.mean()",
            "the mean is worked out exactly (or numerically), so nothing is "
            "drawn; method='simulate' estimates it from draws instead.",
            options,
        )
    return value


def _same_unit(a, b) -> bool:
    """Whether two units' lifetime models are the same: one object, or the
    same distribution with the same parameters."""
    from .non_repairable_rbd import NonRepairableRBD

    return NonRepairableRBD._same_model(a, b)


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
        from scipy.stats import gamma

        return gamma.sf(x, a=self.shape, scale=1.0 / self.rate)

    def ff(self, x):
        from scipy.stats import gamma

        return gamma.cdf(x, a=self.shape, scale=1.0 / self.rate)

    def mean(self, *args, **kwargs):
        return self.shape / self.rate


@refuse_removed_names
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
       units alive (unless the stage rates are too close for surpyval to
       separate, as with a tiny ``dormancy_factor``).
    2. **Exact k-out-of-n**, for hot standby of any units: the
       arrangement works while at least ``k`` of its units do, each
       independently.
    3. **Numerical**, deterministic and accurate to about ``1e-5`` of a
       probability or better:

       - cold standby with ``k = 1``: the lifetime is the sum of the
         units' lifetimes (under imperfect switching, a mixture of partial
         sums), convolved on a time grid;
       - cold standby of identical units with ``k >= 2``: each operating
         position runs a renewal process of the units' lives, and the
         arrangement fails at the ``n - k + 1``-th failure in all;
       - cold standby of different units with ``k = 2``: a recursion over
         the switch-ins on the time and the other operating unit's start
         (about ``1e-4``, ``1e-3`` for lives with a steep start, such as a
         Weibull of shape below 1);
       - warm standby with ``k = 1``: a recursion over the spares'
         switch-ins on a time grid (a spare switched in at ``tau`` has
         aged ``dormancy_factor * tau``).
    4. **None** otherwise (cold standby of different units with
       ``k >= 3``, and warm standby with ``k >= 2``): ``sf``, ``ff``,
       ``cs`` and ``mean()`` refuse, and ``is_simulated`` is True. Its
       lifetimes are still drawn by ``random``, so a diagram with it is
       simulated: ``random``, ``mean(method="simulate")`` and
       ``unreliability_interval`` of a ``NonRepairableRBD``, and
       ``availability`` and ``cost`` of a ``RepairableRBD``.
       ``mean(method="simulate", mc_samples=..., seed=...)`` estimates its
       own mean from new draws. (Until 0.12 its ``sf`` was a Kaplan-Meier
       fit to simulated lifetimes.)

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
    switching_probability : float or sequence of float, optional
        The probability, in ``[0, 1]``, that switching onto the next spare
        succeeds: a scalar for every switch, or one value per spare
        (length ``len(reliabilities) - k``). A failed switch ends the
        arrangement's life when the unit it should replace fails. By
        default 1.0 (perfect switching). Imperfect switching is supported
        for cold standby only. Given by name, as ``dormancy_factor`` is.
    dormancy_factor : float, optional
        The dormant-to-operating aging ratio, in ``[0, 1]``: 0 is cold, 1
        is hot and anything between is warm. By default 0.0.

    Attributes
    ----------
    N : int
        The number of units, ``len(reliabilities)``.

    Raises
    ------
    ValueError
        If ``k`` exceeds ``len(reliabilities)``, ``dormancy_factor`` is
        outside ``[0, 1]``, or (cold, ``k = 1``) a switching probability is
        outside ``[0, 1]`` or a sequence of them has the wrong length.
    NotImplementedError
        If ``switching_probability`` is not 1 while ``dormancy_factor >
        0``.

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

    Two of three identical Weibull units must operate: each operating
    position renews its unit from the spares, and the reliability is
    numerical, with no simulation:

    >>> w = surv.Weibull.from_params([100.0, 2.0])
    >>> two = StandbyModel([w, w, w], k=2)
    >>> two.is_simulated
    False
    >>> round(float(two.sf(50.0)), 4)
    0.9364

    A warm spare, aging at a third of the operating rate:

    >>> warm = StandbyModel([w, w], dormancy_factor=0.3)
    >>> round(float(warm.sf(150.0)), 4)
    0.4815

    As a node of an RBD:

    >>> from repyability import NonRepairableRBD
    >>> rbd = NonRepairableRBD(
    ...     [("s", "pumps"), ("pumps", "t")], {"pumps": cold}
    ... )
    >>> round(rbd.sf(100.0), 4)
    0.7358
    """

    # Whether a convolved survival function keeps every partial sum's too
    # (a DegradingNode needs them for the stage it is in).
    _partial_sums = False

    # The arrangement, when no exact or numerical method gives its
    # reliability ("warm standby with 2 units operating"), else None.
    _case = None

    def __init__(
        self,
        reliabilities,
        k=1,
        *,
        switching_probability=1.0,
        dormancy_factor=0.0,
    ):
        k = whole_number(k, "k (how many units must operate)")
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
        self.switching_probability = switching_probability
        self.dormancy_factor = float(dormancy_factor)

        rate = _identical_exponential_rate(reliabilities)
        identical = all(
            _same_unit(model, reliabilities[0]) for model in reliabilities
        )
        if self.dormancy_factor > 0.0:
            # Warm/hot standby: dormant aging couples the spares' clocks to
            # elapsed time, so the cold-only machinery (lifetime sums /
            # convolution) does not apply.
            if not is_perfect_switching(switching_probability):
                raise NotImplementedError(
                    "switching_probability is only supported for cold "
                    "standby (dormancy_factor == 0)."
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
            elif self.dormancy_factor == 1.0:
                # Hot standby is active k-out-of-n redundancy: exact.
                self._sf_model = KOutOfNSurvival(reliabilities, k)
            elif k == 1:
                # One unit operating and warm spares: a recursion over the
                # switch-ins (see WarmStandbySurvival).
                self._sf_model = WarmStandbySurvival(
                    reliabilities, self.dormancy_factor
                )
            else:
                self._case = f"warm standby with {k} units operating"
                self._sf_model = None
        elif rate is not None and is_perfect_switching(switching_probability):
            # Identical exponential units: the cold standby lifetime is exactly
            # Erlang(N-k+1, k*rate) for any k, by the memorylessness of the
            # exponential. Use that closed form directly.
            self._sf_model = _ExponentialStandbySurvival(rate, self.N, k)
        elif k == 1:
            # Cold standby (k=1): the lifetime is the sum of the components'
            # lifetimes (or a mixture of partial sums under imperfect
            # switching), whose survival function is computed deterministically
            # by numerical convolution rather than from Monte-Carlo samples.
            self._sf_model = ConvolvedSurvival(
                reliabilities,
                switching_probability=switching_probability,
                partials=self._partial_sums,
            )
        elif identical:
            # Identical units with k operating: each operating position
            # runs a renewal process of the units' lives (see
            # RenewalStandbySurvival).
            self._sf_model = RenewalStandbySurvival(
                reliabilities[0], self.N, k, self._switch_probs()
            )
        elif k == 2:
            # Two different units operating: a recursion over the switch-ins
            # on the time and the other operating unit's start (see
            # ColdPairSurvival).
            self._sf_model = ColdPairSurvival(
                reliabilities, self._switch_probs()
            )
        else:
            # For k >= 3 different units the lifetime is not a simple sum
            # (which spare goes where depends on the order in which the
            # operating units fail, with more ages to track): only
            # simulations take it.
            self._switch_probs()
            self._case = f"cold standby with {k} different units operating"
            self._sf_model = None

    def _no_reliability(self, what: str = "reliability") -> Exception:
        """The refusal of an arrangement with no exact or numerical
        ``what`` (see ``is_simulated``)."""
        own = (
            " Estimate it with mean(method='simulate', mc_samples=..., "
            "seed=...)."
            if what == "mean life"
            else ""
        )
        return NotImplementedError(
            f"This StandbyModel ({self._case}) has no exact or numerical "
            f"{what}.{own} A diagram with it is simulated: random, "
            "mean(method='simulate') and unreliability_interval of a "
            "NonRepairableRBD, or availability and cost of a RepairableRBD."
        )

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
        sample): all at once where every budget is finite, sample by sample
        elsewhere (a unit that never fails has an infinite budget)."""
        finite = np.all(np.isfinite(budgets), axis=1)
        if finite.all():
            return self._warm_lifetimes(budgets)
        out = np.empty(budgets.shape[0], dtype=float)
        if finite.any():
            out[finite] = self._warm_lifetimes(budgets[finite])
        out[~finite] = self._warm_lifetimes_by_sample(budgets[~finite])
        return out

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
                if dt == np.inf:
                    # Every unit left never fails: nor does the arrangement.
                    t = np.inf
                    break
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

    def _switch_probs(self) -> list:
        """The probability that each spare's switch-in succeeds, in list
        order: ``len(reliabilities) - k`` of them."""
        return switch_success_probs(
            self.switching_probability, self.N - self.k + 1
        )

    def _random_cold_k(self, size):
        """Cold standby with k >= 2: every sample at once when each unit's
        draws can be reproduced exactly in one block (see ``_sampling``),
        otherwise sample by sample. Either way a sample draws the operating
        units' lifetimes, then, for each spare, the uniform that decides its
        switch (under imperfect switching) and its lifetime."""
        sampler = self._row_sampler()
        if sampler is not None:
            state = np.random.get_state()
            out = sampler.draw(np.random.random_sample((size, sampler.width)))
            if not np.isnan(out).any():
                return out
            # NaN times order differently in the queue; rewind and loop.
            np.random.set_state(state)

        perfect = is_perfect_switching(self.switching_probability)
        probs = self._switch_probs()
        x_random = np.zeros(size)
        for i in range(size):
            pq: PriorityQueue = PriorityQueue()
            # start k streams:
            for node in self.reliabilities[: self.k]:
                pq.put(node.random(1).item())

            # Add the next event time to the lowest value in the queue; a
            # failed switch ends the arrangement then.
            ended = None
            spares = self.reliabilities[self.k :]  # noqa: E203
            for node, p in zip(spares, probs):
                switched = perfect or np.random.random() < p
                next_t = node.random(1).item()
                current_lowest = pq.get()
                if ended is None and not switched:
                    ended = current_lowest
                pq.put(current_lowest + next_t)

            x_random[i] = pq.get() if ended is None else ended
        return x_random

    def _cold_k_lifetimes(self, lifetimes, switched=None):
        """Cold standby with k >= 2 for every sample at once, from each
        unit's lifetimes (a list of arrays, one per unit) and, under
        imperfect switching, whether each spare's switch succeeds (a list
        of boolean arrays): the queue loop's stream arithmetic, one array
        operation per spare."""
        streams = np.column_stack(lifetimes[: self.k])
        rows = np.arange(len(streams))
        done = np.zeros(len(streams), dtype=bool)
        ended = np.zeros(len(streams))
        for j, spare in enumerate(lifetimes[self.k :]):  # noqa: E203
            # The spare extends whichever stream ends first (a tie between
            # equal ends gives the same result either way).
            lowest = np.argmin(streams, axis=1)
            current = streams[rows, lowest]
            if switched is not None:
                failed = ~done & ~switched[j]
                ended = np.where(failed, current, ended)
                done |= failed
            streams[rows, lowest] = current + spare
        return np.where(done, ended, streams.min(axis=1))

    def _row_sampler(self):
        """``random(1)`` as a :class:`~._sampling.RowSampler`, so an RBD with
        this node batches its draws; ``None`` unless every unit's draws can
        be replayed. Columns follow the order ``random(1)`` draws in: one per
        unit, and under imperfect k=1 switching a switch draw before each
        spare's."""
        found = [inverse_sampler(m) for m in self.reliabilities]
        units = [unit for unit in found if unit is not None]
        if len(units) < len(found):
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

        if is_perfect_switching(self.switching_probability):

            def draw(u):
                lifetimes = [
                    column(u, j, unit) for j, unit in enumerate(units)
                ]
                out = self._cold_k_lifetimes(lifetimes)
                # The queue loop orders NaN times its own way; mark the
                # sample so the caller falls back to it.
                for lifetime in lifetimes:
                    out[np.isnan(lifetime)] = np.nan
                return out

            return RowSampler(self.N, draw)

        # Under imperfect switching: the operating units, then each spare's
        # switch and lifetime.
        k, probs = self.k, self._switch_probs()

        def switched_draw(u):
            lifetimes = [column(u, j, units[j]) for j in range(k)] + [
                column(u, k + 2 * j + 1, units[k + j])
                for j in range(len(probs))
            ]
            switched = [u[:, k + 2 * j] < p for j, p in enumerate(probs)]
            out = self._cold_k_lifetimes(lifetimes, switched)
            for lifetime in lifetimes:
                out[np.isnan(lifetime)] = np.nan
            return out

        return RowSampler(k + 2 * len(probs), switched_draw)

    def mean(self, mc_samples=None, seed=None, *, method=None):
        """Mean lifetime (MTTF) of the arrangement.

        Exact for the Erlang and hypoexponential closed forms, and
        deterministic for the numerical methods (the integral of the
        survival function). An arrangement with neither (see
        ``is_simulated``) has no exact mean, and refuses, unless asked for
        an estimate: the mean of ``mc_samples`` new draws of ``random``.
        ``method="simulate"`` estimates it so whatever the arrangement, to
        check the exact mean against draws, say (#233).

        Parameters
        ----------
        mc_samples : int, optional
            The number of draws to estimate the mean from, 10_000 by
            default: with ``method="simulate"``, or for an arrangement with
            no exact mean. An exact mean draws nothing, and ignores it with
            a ``FutureWarning`` (0.14 will refuse it).
        seed : int or None, optional
            Seed for those draws (see ``random``), by default None.
            Ignored as ``mc_samples`` is.
        method : {None, "exact", "simulate"}, optional
            None (the default): the exact or numerical mean, or for an
            arrangement with neither, the draws' estimate when
            ``mc_samples`` or ``seed`` is given. ``"exact"``: the exact or
            numerical mean, refused where there is none. ``"simulate"``:
            the draws' estimate, whatever.

        Returns
        -------
        float
            The mean lifetime, or its estimate.

        Raises
        ------
        NotImplementedError
            If the arrangement has no exact mean and neither ``mc_samples``
            nor ``seed`` is given (or ``method="exact"``).
        ValueError
            If ``method`` is not one of these.
        TypeError
            If ``method="exact"`` is given with ``mc_samples`` or ``seed``.

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import StandbyModel
        >>> unit = surv.Exponential.from_params([0.01])
        >>> round(float(StandbyModel([unit] * 3).mean()), 4)
        300.0
        >>> w = surv.Weibull.from_params([100.0, 2.0])
        >>> round(float(StandbyModel([w, w, w], k=2).mean()), 2)
        104.44

        Warm standby with two operating has no exact mean, which draws
        estimate:

        >>> sim = StandbyModel([w, w, w], k=2, dormancy_factor=0.5)
        >>> sim.is_simulated
        True
        >>> round(sim.mean(method="simulate", mc_samples=20_000, seed=1), 1)
        95.3

        The exact mean, checked against draws:

        >>> exact = StandbyModel([w, w, w], k=2)
        >>> drawn = exact.mean(method="simulate", mc_samples=20_000, seed=1)
        >>> round(drawn, 1)
        104.5
        """

        def exact() -> float:
            if self._sf_model is None:
                raise self._no_reliability("mean life")
            return float(np.ravel(self._sf_model.mean())[0])

        return drawn_mean(
            self, "StandbyModel", exact, mc_samples, seed, method
        )

    @property
    def is_simulated(self) -> bool:
        """Whether only simulations take the arrangement: no exact or
        numerical method gives its reliability, so ``sf``, ``ff``, ``cs``
        and ``mean()`` refuse, and a diagram with it is simulated."""
        return self._case is not None

    def sf(self, x, *args, **kwargs):
        """Survival function (reliability) of the arrangement.

        Evaluates the survival function set up at construction: the exact
        closed form or the numerical method (interpolated on its grid). An
        RBD calls this for the node's reliability.

        Parameters
        ----------
        x : array_like
            The time(s).
        *args, **kwargs
            Passed on, with ``x``, to the underlying survival model.

        Returns
        -------
        float or numpy.ndarray
            The probability of surviving beyond ``x``: an array for an
            array ``x``, and a numpy float for a scalar ``x``.

        Raises
        ------
        NotImplementedError
            If no exact or numerical method gives it (see
            ``is_simulated``).

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import StandbyModel
        >>> unit = surv.Exponential.from_params([0.01])
        >>> standby = StandbyModel([unit, unit])
        >>> [round(float(r), 4) for r in standby.sf([50.0, 100.0])]
        [0.9098, 0.7358]
        """
        if self._sf_model is None:
            raise self._no_reliability()
        return self._sf_model.sf(x, *args, **kwargs)

    def ff(self, x, *args, **kwargs):
        """Cumulative failure probability, ``1 - sf(x)``.

        Evaluated from the same survival function as ``sf``.

        Parameters
        ----------
        x : array_like
            The time(s).
        *args, **kwargs
            Passed on, with ``x``, to the underlying survival model.

        Returns
        -------
        float or numpy.ndarray
            The probability of failing by ``x``, shaped as for ``sf``.

        Raises
        ------
        NotImplementedError
            As for ``sf``.
        """
        if self._sf_model is None:
            raise self._no_reliability()
        return self._sf_model.ff(x, *args, **kwargs)

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

        return conditional_survival(self, x, X)
