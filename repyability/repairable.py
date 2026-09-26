"""Repairable-component economics across the repair-effectiveness spectrum.

A repaired unit is restored to service, but *how much* a repair rejuvenates it
varies. ``Repairable`` prices the classic repair-vs-renew trade-off for a unit
whose repairs are anything from worthless to partial:

- **Minimal repair** ("as bad as old") — a repair returns the unit to service
  but removes none of its accumulated age, so failures recur with rising
  frequency. This is a recurrent-event process with cumulative intensity
  ``Lambda(t) = E[N(t)]`` (the expected number of failures by age ``t``),
  available analytically from a surpyval recurrence model as ``model.cif(t)``
  (e.g. Crow-AMSAA / NHPP, Duane, HPP).
- **Imperfect repair** (generalized renewal / Kijima virtual-age) — each repair
  rejuvenates the unit *partially* (a restoration factor ``0 < q < 1``), so
  failures still accelerate, but more slowly than under minimal repair.
  ``E[N(t)]`` has no closed form here and is estimated by (seeded) simulation
  from a fitted surpyval ``GeneralizedRenewal`` model as ``model.mcf(t)``.

Minimal repair is the ``q = 1`` edge of imperfect repair; perfect repair
(``q = 0``, a full renewal each time) is the ``NonRepairable`` boundary.
Whichever the unit is, each repair costs ``cr`` while an *overhaul /
replacement* costs ``co`` (> ``cr``) and renews the unit to as-new. Renewing
every ``t`` time units makes each cycle a renewal cycle costing
``cr * E[N(t)] + co``, so the long-run cost rate is::

    g(t) = (cr * E[N(t)] + co) / t

Minimising ``g`` gives the optimal overhaul/replacement interval (the
Barlow-Hunter policy). A finite optimum exists only when the unit wears out
(``E[N(t)]`` growing super-linearly); otherwise repairs never become frequent
enough for renewal to pay and the optimal interval is infinite (reported as
``inf`` for an analytic model; the simulated search is bounded by a horizon
and returns the horizon instead).

``Repairable`` also prices a *failure-limit* policy: repair the first
``n - 1`` failures and replace the unit at the ``n``-th, with long-run cost
rate ``(cr * (n - 1) + co) / E[T_n]``, where ``E[T_n]``, the expected time to
the ``n``-th failure, is simulated from an imperfect-repair model (for
power-law minimal repair, ``minimal_repair_time_to_nth_failure`` gives it in
closed form).

Contrast with ``NonRepairable``, which models renewal *by replacement* ("as
good as new") and the age-replacement policy. RBD components are assumed to
renew on repair, so a repairable unit modelled here is not a valid RBD node
model — ``Repairable`` is a standalone component-level tool.
"""

import warnings
from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from repyability.maintenance import FailureLimitPolicy, MaintenancePolicy
from repyability.utils.wrappers import numpy_seed

# Simulation draws used to estimate E[N(t)] for a simulation-backed
# (imperfect-repair) model. Ignored for analytic (``cif``) models.
_DEFAULT_N_SIMULATIONS = 1000

# Default largest failure count searched by the replace-at-N-th-failure policy.
_DEFAULT_MAX_FAILURES = 30


def minimal_repair_time_to_nth_failure(
    alpha: float, beta: float, n: int
) -> float:
    """Expected time to the ``n``-th failure under power-law minimal repair.

    For a minimal-repair process with expected number of failures
    ``E[N(t)] = (t / alpha) ** beta`` (a Weibull(``alpha``, ``beta``) unit
    repaired "as bad as old": surpyval's
    ``CrowAMSAA.from_params([alpha, beta])``, or a ``GeneralizedRenewal``
    with that Weibull baseline and ``q = 1``), the ``n``-th failure time
    is ``alpha * S ** (1 / beta)`` with ``S ~ Gamma(n, 1)``, so
    ``E[T_n] = alpha * Gamma(n + 1 / beta) / Gamma(n)``.

    The closed form is exact and cheap: use it for minimal repair in place
    of the simulated
    [`Repairable.expected_time_to_nth_failure`]
    [repyability.Repairable.expected_time_to_nth_failure] (which needs an
    imperfect-repair model), or to check that estimate at ``q = 1``.
    ``n = 1`` gives the Weibull mean, ``alpha * Gamma(1 + 1 / beta)``.

    Parameters
    ----------
    alpha : float
        Scale of the power law (the Weibull baseline's scale), in time
        units.
    beta : float
        Shape of the power law (the Weibull baseline's shape).
    n : int
        The failure number, 1 for the first failure.

    Returns
    -------
    float
        The expected time to the ``n``-th failure, in the units of
        ``alpha``.

    Raises
    ------
    ValueError
        If ``n < 1``.

    Examples
    --------
    >>> from repyability import minimal_repair_time_to_nth_failure
    >>> round(minimal_repair_time_to_nth_failure(100.0, 2.0, 1), 2)
    88.62
    >>> round(minimal_repair_time_to_nth_failure(100.0, 2.0, 3), 2)
    166.17
    """
    if n < 1:
        raise ValueError("n must be a positive integer.")
    return float(alpha * np.exp(gammaln(n + 1.0 / beta) - gammaln(n)))


# Default search horizon for the simulated optimum, as a multiple of the
# baseline mean-time-to-first-failure. Imperfect repair pushes the optimal
# renewal interval to several times that mean; simulating much farther is both
# slow and numerically unstable (the virtual-age process asymptotes), so the
# horizon is bounded and exposed as ``max_interval``.
_DEFAULT_HORIZON_MULTIPLE = 15.0


class Repairable:
    """Repairable component with optimal overhaul and failure-limit policies.

    Prices the repair-versus-renew trade-off for a unit whose repairs are
    minimal ("as bad as old") or imperfect (partial rejuvenation). Each
    failure is repaired at cost ``cr``; an overhaul/replacement costs
    ``co > cr`` and renews the unit to as good as new. Two policies are
    optimised:

    - periodic overhaul: renew every ``t`` time units, with long-run cost
      rate ``g(t) = (cr * E[N(t)] + co) / t`` (the Barlow-Hunter policy),
      where ``E[N(t)]`` is the expected number of failures by age ``t``;
      see ``optimal_overhaul_policy()``;
    - failure limit: repair the first ``n - 1`` failures and replace at
      the ``n``-th, with cost rate ``(cr * (n - 1) + co) / E[T_n]``, where
      ``E[T_n]`` is the expected time to the ``n``-th failure; see
      ``optimal_failure_limit_policy()`` (simulation-backed models only).

    Repairs and overhauls take no time and costs are per event, so cost
    rates are per unit of the model's time. Set the costs with
    ``set_repair_and_overhaul_costs()`` before any cost calculation.

    This is a standalone component-level tool, not an RBD node model: RBD
    components are assumed to renew on repair, as a
    [`NonRepairable`][repyability.NonRepairable] does.

    Parameters
    ----------
    model : object
        A recurrent-event model exposing ``E[N(t)]`` as either:

        - ``cif(t)``: an analytic cumulative intensity (minimal repair),
          e.g. a surpyval ``CrowAMSAA`` or ``Duane`` model (fitted, or
          built with ``from_params``) or a fitted ``HPP``; or
        - ``mcf(t, items=..., seed=...)``: a simulation-estimated mean
          cumulative function (imperfect repair), e.g. a surpyval
          ``GeneralizedRenewal`` (Kijima I/II) model, fitted or built with
          ``fit_from_parameters``.

        If both are present ``cif`` is used (analytic, exact). The
        failure-count methods (``expected_time_to_nth_failure()`` and the
        failure-limit policy) also call the model's
        ``count_terminated_simulation``, which surpyval's
        simulation-backed models provide.

    Attributes
    ----------
    model : object
        The recurrent-event model given.

    Raises
    ------
    ValueError
        If ``model`` exposes neither ``cif`` nor ``mcf``.

    Examples
    --------
    Minimal repair with a power-law (Crow-AMSAA) intensity, where
    ``E[N(t)] = (t / 100) ** 1.5`` is exact:

    >>> from surpyval.recurrent import CrowAMSAA
    >>> from repyability import Repairable
    >>> unit = Repairable(CrowAMSAA.from_params([100.0, 1.5]))
    >>> unit.set_repair_and_overhaul_costs(cr=10.0, co=1000.0)
    >>> policy = unit.optimal_overhaul_policy()
    >>> round(policy.interval, 1), round(policy.cost_rate, 4)
    (3420.0, 0.8772)

    Imperfect repair (Kijima I with restoration factor ``q = 0.5``) has no
    closed-form ``E[N(t)]``, so it is simulated; pass a ``seed`` for a
    reproducible result:

    >>> import surpyval as surv
    >>> from surpyval.recurrent import GeneralizedRenewal
    >>> grp = GeneralizedRenewal.fit_from_parameters(
    ...     [100.0, 2.0], 0.5, kijima="i", dist=surv.Weibull
    ... )
    >>> unit = Repairable(grp)
    >>> unit.set_repair_and_overhaul_costs(cr=1.0, co=5.0)
    >>> policy = unit.optimal_overhaul_policy(
    ...     seed=1, n_simulations=100, max_interval=600.0
    ... )
    >>> round(policy.interval), round(policy.cost_rate, 3)
    (369, 0.035)
    """

    def __init__(self, model):
        self._analytic = hasattr(model, "cif")
        if not (self._analytic or hasattr(model, "mcf")):
            raise ValueError(
                "model must expose cif() (analytic cumulative intensity, e.g. "
                "a surpyval recurrence model such as CrowAMSAA) or mcf() (a "
                "simulation-estimated mean cumulative function, e.g. a fitted "
                "GeneralizedRenewal)"
            )
        self.model = model

    @property
    def is_simulated(self) -> bool:
        """Whether ``E[N(t)]`` is estimated by simulation.

        ``True`` for an ``mcf``-only (imperfect-repair) model, whose
        methods honour ``seed``, ``n_simulations`` and ``max_interval``;
        ``False`` for an analytic ``cif`` model, whose methods ignore them.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> from repyability import Repairable
        >>> Repairable(CrowAMSAA.from_params([100.0, 1.5])).is_simulated
        False
        """
        return not self._analytic

    def set_repair_and_overhaul_costs(self, cr: float, co: float) -> None:
        """Set the repair cost ``cr`` and overhaul/replacement cost ``co``.

        Required before any cost or policy calculation. The costs are
        stored as the ``cr`` and ``co`` attributes.

        Parameters
        ----------
        cr : float
            Cost of each (minimal or imperfect) repair. Must be positive.
        co : float
            Cost of an overhaul/replacement, which renews the unit. Must
            exceed ``cr``: otherwise one would simply renew at every
            failure.

        Raises
        ------
        ValueError
            If ``cr <= 0`` or ``cr >= co``.
        """
        if cr <= 0:
            raise ValueError("repair cost, cr, must be positive.")
        if cr >= co:
            raise ValueError(
                "repair cost, cr, must be less than overhaul cost, co."
            )
        self.cr = cr
        self.co = co

    def _require_costs(self) -> None:
        if not hasattr(self, "cr"):
            raise ValueError(
                "costs not set: call set_repair_and_overhaul_costs(cr, co) "
                "first"
            )

    def _expected_failures(
        self, t, seed: Optional[int], n_simulations: int
    ) -> np.ndarray:
        """E[N(t)], the expected number of failures by age ``t``.

        Analytic via ``cif`` for intensity models; a seeded simulation
        estimate via ``mcf`` for imperfect-repair (renewal) models.
        """
        if self._analytic:
            return np.asarray(self.model.cif(t), dtype=float)
        return np.asarray(
            self.model.mcf(t, items=n_simulations, seed=seed), dtype=float
        )

    def _expected_failures_scalar(
        self, t: float, seed: Optional[int], n_simulations: int
    ) -> float:
        return float(
            np.asarray(
                self._expected_failures(t, seed, n_simulations), dtype=float
            ).reshape(-1)[0]
        )

    def cost(
        self,
        t,
        seed: Optional[int] = None,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
    ) -> Union[float, np.ndarray]:
        """Expected cost of one overhaul/replacement cycle of length ``t``.

        ``cr * E[N(t)] + co``: the repairs expected before the overhaul at
        age ``t``, plus the overhaul itself.

        Parameters
        ----------
        t : float or array_like
            The cycle length(s), in the model's time units.
        seed : int, optional
            Seed for the simulation of a simulation-backed model (ignored
            for an analytic one). ``None`` (the default) draws from numpy's
            global RNG.
        n_simulations : int, optional
            Number of simulated histories used to estimate ``E[N(t)]``
            (default 1000; ignored for an analytic model).

        Returns
        -------
        float or numpy.ndarray
            The expected cycle cost: a float for a scalar ``t``, an array
            for an array ``t``.

        Raises
        ------
        ValueError
            If the costs have not been set.

        Examples
        --------
        With ``E[N(250)] = (250 / 100) ** 1.5 = 3.95`` expected repairs:

        >>> from surpyval.recurrent import CrowAMSAA
        >>> from repyability import Repairable
        >>> unit = Repairable(CrowAMSAA.from_params([100.0, 1.5]))
        >>> unit.set_repair_and_overhaul_costs(cr=10.0, co=1000.0)
        >>> round(unit.cost(250.0), 2)
        1039.53
        """
        self._require_costs()
        scalar_in = np.ndim(t) == 0
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        lam = self._expected_failures(tt, seed, n_simulations)
        out = self.cr * lam + self.co
        return out.item() if scalar_in else out

    def cost_rate(
        self,
        t,
        seed: Optional[int] = None,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
    ) -> Union[float, np.ndarray]:
        """Long-run cost per unit time when overhauling every ``t``.

        ``g(t) = (cr * E[N(t)] + co) / t``, the cost of one cycle over its
        length (renewal-reward); ``t = 0`` gives ``inf``. This is the cost
        rate the overhaul policy minimises.

        For a simulation-backed model a single simulation, run to the
        largest ``t``, serves every age in the call, so pass all the ages
        to be compared in one call: with a fixed ``seed``, the estimate at
        an age depends on the largest age requested.

        Parameters
        ----------
        t : float or array_like
            The overhaul interval(s), in the model's time units.
        seed : int, optional
            Seed for the simulation of a simulation-backed model (ignored
            for an analytic one). ``None`` (the default) draws from numpy's
            global RNG.
        n_simulations : int, optional
            Number of simulated histories used to estimate ``E[N(t)]``
            (default 1000; ignored for an analytic model).

        Returns
        -------
        float or numpy.ndarray
            The cost rate: a float for a scalar ``t``, an array for an
            array ``t``.

        Raises
        ------
        ValueError
            If the costs have not been set.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> from repyability import Repairable
        >>> unit = Repairable(CrowAMSAA.from_params([100.0, 1.5]))
        >>> unit.set_repair_and_overhaul_costs(cr=10.0, co=1000.0)
        >>> round(unit.cost_rate(250.0), 4)
        4.1581
        >>> [round(float(g), 4) for g in unit.cost_rate([1000.0, 3420.0])]
        [1.3162, 0.8772]
        """
        self._require_costs()
        scalar_in = np.ndim(t) == 0
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        lam = self._expected_failures(tt, seed, n_simulations)
        with np.errstate(divide="ignore"):
            out = (self.cr * lam + self.co) / tt
        return out.item() if scalar_in else out

    def _g(self, t: float, seed: Optional[int], n_simulations: int) -> float:
        """Cost rate at a scalar ``t > 0`` (float in, float out)."""
        return (
            self.cr * self._expected_failures_scalar(t, seed, n_simulations)
            + self.co
        ) / t

    def _optimise_analytic(self) -> tuple[float, float]:
        """Analytic optimum for an intensity (``cif``) model; the interval is
        ``inf`` (with the limiting repair-only cost rate) when overhauls never
        pay."""

        def enf(t: float) -> float:
            return self._expected_failures_scalar(t, None, 0)

        def g(t: float) -> float:
            return self._g(t, None, 0)

        # At a finite optimum the cumulative repair spend is comparable to the
        # overhaul cost (for a power law, Lambda(t*) = co/(cr*(b-1))), so first
        # bracket the age where Lambda(t) reaches co/cr and centre the search
        # grid on that scale.
        target = self.co / self.cr
        t_scale = 1.0
        if enf(t_scale) < target:
            for _ in range(300):
                t_scale *= 2.0
                if enf(t_scale) >= target:
                    break
        else:
            for _ in range(300):
                if enf(t_scale / 2.0) < target:
                    break
                t_scale /= 2.0

        for _ in range(6):
            grid = t_scale * np.logspace(-4.0, 4.0, 1601)
            gr = np.asarray(self.cost_rate(grid))
            i = int(np.argmin(gr))
            if i == len(grid) - 1:
                # Minimum at the right edge: overhauling later keeps getting
                # cheaper. If Lambda is not growing super-linearly out here, it
                # never stops getting cheaper (no wear-out) — never overhaul.
                big_t = float(grid[-1])
                lam_t = enf(big_t)
                lam_2t = enf(2.0 * big_t)
                if lam_t <= 0.0 or (
                    np.log(lam_2t / lam_t) / np.log(2.0) <= 1.0 + 1e-9
                ):
                    return float(np.inf), g(big_t)
                t_scale *= 1e4  # genuine wear-out; optimum is further out
                continue
            if i == 0:
                t_scale *= 1e-4
                continue
            res = minimize_scalar(
                g,
                bounds=(float(grid[i - 1]), float(grid[i + 1])),
                method="bounded",
            )
            t_star = float(res.x)
            return t_star, g(t_star)
        # Search saturated (pathological model): best point found.
        t_star = float(grid[i])
        return t_star, g(t_star)

    def _timescale(self) -> float:
        """A characteristic time for the failure process, used to centre the
        search grid without repeated simulation.

        The baseline mean-time-to-first-failure sets the scale (the optimal
        renewal interval is a small multiple of it). Falls back to 1.0 if the
        model does not expose a baseline mean.
        """
        baseline = getattr(self.model, "model", None)
        if baseline is not None and hasattr(baseline, "mean"):
            try:
                m = float(np.atleast_1d(baseline.mean())[0])
                if np.isfinite(m) and m > 0.0:
                    return m
            except Exception:
                pass
        return 1.0

    def _optimise_simulated(
        self,
        seed: Optional[int],
        n_simulations: int,
        max_interval: Optional[float],
    ) -> tuple[float, float]:
        """Grid optimum for a simulation-backed (imperfect-repair) model.

        The cost of a seeded ``mcf`` call scales with the sample size and the
        horizon, not the number of grid points, and a single seeded ``mcf``
        call over a grid is self-consistent (monotone), so this uses **one**
        ``mcf`` evaluation over a log grid up to ``max_interval`` and takes the
        minimum of the (unimodal) cost rate ``(cr*E[N(t)] + co)/t``.

        ``max_interval`` bounds the search: imperfect repair pushes the
        optimum to several times the baseline mean, and simulating much farther
        is slow and numerically unstable, so effective-repair cases whose
        optimum lies beyond the horizon return the horizon (raise
        ``max_interval`` to search further).
        """
        with numpy_seed(seed):
            if max_interval is None:
                max_interval = _DEFAULT_HORIZON_MULTIPLE * self._timescale()

            grid = max_interval * np.logspace(-2.5, 0.0, 250)
            gr = np.asarray(
                self.cost_rate(grid, seed=seed, n_simulations=n_simulations)
            )
            i = int(np.nanargmin(gr))
            return float(grid[i]), float(gr[i])

    def _optimise(
        self,
        seed: Optional[int],
        n_simulations: int,
        max_interval: Optional[float],
    ) -> tuple[float, float]:
        self._require_costs()
        if self._analytic:
            return self._optimise_analytic()
        return self._optimise_simulated(seed, n_simulations, max_interval)

    def find_optimal_overhaul_interval(
        self,
        seed: Optional[int] = None,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        max_interval: Optional[float] = None,
    ) -> float:
        """The overhaul/replacement interval minimising the long-run cost rate.

        Minimises ``g(t) = (cr * E[N(t)] + co) / t`` (see ``cost_rate``).

        For an analytic (``cif``) model the search is deterministic: it
        brackets, to within a factor of 2, the age at which
        ``cr * E[N(t)]`` reaches ``co``, scans a 1601-point log-spaced grid
        from 1e-4 to 1e4 times that age (moving the grid when the minimum
        falls on an edge), and refines the best grid point with a bounded
        scalar minimiser. It returns ``inf`` when renewal never pays: the
        unit does not wear out (``E[N(t)]`` grows at most linearly, e.g. an
        HPP, or Crow-AMSAA with ``beta <= 1``), so the cost rate keeps
        falling as the interval grows. ``seed``, ``n_simulations`` and
        ``max_interval`` are ignored.

        For a simulation-backed (imperfect-repair) model, ``E[N(t)]`` is
        estimated by a single simulation over 250 log-spaced ages from
        ``max_interval / 10**2.5`` to ``max_interval``, and the cheapest of
        those ages is returned (no refinement). The result is never
        ``inf``: if the cost rate is still falling at ``max_interval`` (the
        optimum lies beyond it, or there is none), ``max_interval`` itself
        is returned, so a result equal to it calls for a longer horizon.
        The horizon must also stay within what surpyval's simulator can
        resolve: once the baseline survival function at the unit's virtual
        age falls below about 1e-16 (double precision; surpyval then warns
        that sequences stalled), the simulated ``E[N(t)]`` stops growing,
        which can drag the optimum to the horizon. Near-minimal repair
        (``q`` close to 1) can reach that point within the default
        horizon; lower ``max_interval`` if it does.

        Parameters
        ----------
        seed : int, optional
            Seed for a reproducible simulation (simulation-backed models
            only). ``None`` (the default) draws from numpy's global RNG.
        n_simulations : int, optional
            Number of simulated histories used to estimate ``E[N(t)]``
            (default 1000; simulation-backed models only).
        max_interval : float, optional
            The search horizon (simulation-backed models only). By default
            15 times the mean of the model's baseline lifetime distribution
            (``model.model.mean()``), or 15 if there is none.

        Returns
        -------
        float
            The optimal interval, in the model's time units; ``inf`` (for
            an analytic model only) when overhauls never pay.

        Raises
        ------
        ValueError
            If the costs have not been set.

        Examples
        --------
        A unit with a constant failure intensity (no wear-out) is never
        worth overhauling:

        >>> from surpyval.recurrent import CrowAMSAA
        >>> from repyability import Repairable
        >>> unit = Repairable(CrowAMSAA.from_params([500.0, 1.0]))
        >>> unit.set_repair_and_overhaul_costs(cr=10.0, co=1000.0)
        >>> unit.find_optimal_overhaul_interval()
        inf

        Simulated minimal repair (``q = 1``) lands near the closed-form
        optimum ``alpha * (co / (cr * (beta - 1))) ** (1 / beta)``, here
        ``100 * 5 ** 0.5 = 223.6``:

        >>> import surpyval as surv
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> grp = GeneralizedRenewal.fit_from_parameters(
        ...     [100.0, 2.0], 1.0, kijima="i", dist=surv.Weibull
        ... )
        >>> unit = Repairable(grp)
        >>> unit.set_repair_and_overhaul_costs(cr=1.0, co=5.0)
        >>> t = unit.find_optimal_overhaul_interval(
        ...     seed=1, n_simulations=100, max_interval=500.0
        ... )
        >>> round(t)
        239
        """
        return self._optimise(seed, n_simulations, max_interval)[0]

    def optimal_overhaul_policy(
        self,
        seed: Optional[int] = None,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        max_interval: Optional[float] = None,
    ) -> MaintenancePolicy:
        """The optimal overhaul/replacement policy as a typed result.

        Runs the search of ``find_optimal_overhaul_interval()`` (see there
        for the method, its limits and the parameters) and also reports the
        long-run cost rate at the optimum.

        Parameters
        ----------
        seed : int, optional
            Seed for a reproducible simulation (simulation-backed models
            only). ``None`` (the default) draws from numpy's global RNG.
        n_simulations : int, optional
            Number of simulated histories used to estimate ``E[N(t)]``
            (default 1000; simulation-backed models only).
        max_interval : float, optional
            The search horizon (simulation-backed models only); see
            ``find_optimal_overhaul_interval()``.

        Returns
        -------
        MaintenancePolicy
            ``interval`` is the optimal overhaul interval and ``cost_rate``
            the long-run cost per unit time under it. When the interval is
            ``inf`` (never overhaul), ``cost_rate`` is the cost rate at the
            far end of the search grid, which approximates the limiting
            rate of repairs alone, ``cr * E[N(t)] / t`` as ``t`` grows. For
            a simulation-backed model, ``cost_rate`` is the simulated
            estimate at the chosen interval.

        Raises
        ------
        ValueError
            If the costs have not been set.

        Examples
        --------
        >>> from surpyval.recurrent import CrowAMSAA
        >>> from repyability import Repairable
        >>> unit = Repairable(CrowAMSAA.from_params([100.0, 1.5]))
        >>> unit.set_repair_and_overhaul_costs(cr=10.0, co=1000.0)
        >>> policy = unit.optimal_overhaul_policy()
        >>> round(policy.interval, 1), round(policy.cost_rate, 4)
        (3420.0, 0.8772)

        Without wear-out the unit is never overhauled and only repairs are
        paid for, here ``cr / alpha = 10 / 500`` per unit time:

        >>> unit = Repairable(CrowAMSAA.from_params([500.0, 1.0]))
        >>> unit.set_repair_and_overhaul_costs(cr=10.0, co=1000.0)
        >>> policy = unit.optimal_overhaul_policy()
        >>> policy.interval, round(policy.cost_rate, 3)
        (inf, 0.02)
        """
        interval, rate = self._optimise(seed, n_simulations, max_interval)
        return MaintenancePolicy(interval=interval, cost_rate=rate)

    # -- Replace-at-N-th-failure policy ------------------------------------

    def _require_simulation(self) -> None:
        if self._analytic:
            raise ValueError(
                "the N-th-failure policy needs an imperfect-repair "
                "(simulation-backed) model; for a power-law minimal-repair "
                "process use minimal_repair_time_to_nth_failure() directly."
            )

    def _expected_times_to_failures(
        self, max_failures: int, seed: Optional[int], n_simulations: int
    ) -> np.ndarray:
        """``[E[T_1], ..., E[T_k]]`` from a single seeded count-terminated
        simulation, where ``k <= max_failures``.

        Simulating to the ``max_failures``-th failure records every earlier
        failure time too, so one simulation yields the whole curve, and for
        each item the ``n``-th failure time is its ``n``-th ordered event. The
        surpyval simulator can become unstable (NaN interarrival times) at high
        failure counts for near-minimal repair; if it does, the horizon is
        halved until it succeeds and the (shorter) achievable curve is
        returned with a warning.
        """
        self._require_simulation()
        with numpy_seed(seed):
            achievable = max_failures
            result = None
            while achievable >= 1:
                try:
                    result = self.model.count_terminated_simulation(
                        achievable, items=n_simulations, seed=seed
                    )
                    break
                except (ValueError, FloatingPointError):
                    achievable //= 2
            if result is None:
                raise ValueError(
                    "could not simulate the failure process; the model may be "
                    "degenerate."
                )
        if achievable < max_failures:
            warnings.warn(
                "the failure simulator became unstable beyond "
                f"{achievable} failures; the search was truncated from "
                f"{max_failures}. For minimal repair use "
                "minimal_repair_time_to_nth_failure().",
                stacklevel=3,
            )
        data = result.data
        x = np.asarray(data.x, dtype=float)
        item = np.asarray(data.i)

        # Order by (item, time), then read the n-th event of each item block.
        order = np.lexsort((x, item))
        xs, items = x[order], item[order]
        ids = np.arange(1, n_simulations + 1)
        starts = np.searchsorted(items, ids, side="left")
        ends = np.searchsorted(items, ids, side="right")

        expected = np.empty(achievable, dtype=float)
        for n in range(1, achievable + 1):
            idx = starts + (n - 1)
            valid = idx < ends
            expected[n - 1] = np.nanmean(xs[idx[valid]])
        return expected

    def expected_time_to_nth_failure(
        self,
        n: int,
        seed: Optional[int] = None,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
    ) -> float:
        """Expected time to the ``n``-th failure, ``E[T_n]``.

        Estimated from a single count-terminated simulation of
        ``n_simulations`` failure histories (the model's
        ``count_terminated_simulation``), as the mean of their ``n``-th
        failure times. It needs a simulation-backed (imperfect-repair)
        model; for a power-law minimal-repair process the closed form
        [`minimal_repair_time_to_nth_failure`]
        [repyability.minimal_repair_time_to_nth_failure] is exact and
        cheaper.

        If the simulation fails at a high failure count (surpyval's
        simulator can, for near-minimal repair), it is retried with the
        count halved until it succeeds, with a warning; ``ValueError`` is
        then raised, as ``n`` failures were not reached.

        Parameters
        ----------
        n : int
            The failure number, 1 for the first failure.
        seed : int, optional
            Seed for a reproducible simulation. ``None`` (the default)
            draws from numpy's global RNG.
        n_simulations : int, optional
            Number of simulated histories (default 1000).

        Returns
        -------
        float
            The expected time to the ``n``-th failure, in the model's time
            units.

        Raises
        ------
        ValueError
            If ``n < 1``, if the model is analytic (``cif``), or if the
            simulation cannot reach ``n`` failures.

        Examples
        --------
        At ``q = 1`` (minimal repair) the estimate approaches the closed
        form:

        >>> import surpyval as surv
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> from repyability import Repairable
        >>> from repyability import minimal_repair_time_to_nth_failure
        >>> grp = GeneralizedRenewal.fit_from_parameters(
        ...     [100.0, 2.0], 1.0, kijima="i", dist=surv.Weibull
        ... )
        >>> unit = Repairable(grp)
        >>> t3 = unit.expected_time_to_nth_failure(
        ...     3, seed=1, n_simulations=500
        ... )
        >>> round(t3, 1)
        165.4
        >>> round(minimal_repair_time_to_nth_failure(100.0, 2.0, 3), 1)
        166.2
        """
        if n < 1:
            raise ValueError("n must be a positive integer.")
        curve = self._expected_times_to_failures(n, seed, n_simulations)
        if len(curve) < n:
            raise ValueError(
                f"the simulator could not reach {n} failures for this model "
                "(near-minimal repair); use "
                "minimal_repair_time_to_nth_failure() instead."
            )
        return float(curve[n - 1])

    def _optimise_failure_limit(
        self, seed: Optional[int], n_simulations: int, max_failures: int
    ) -> tuple[int, float]:
        self._require_costs()
        expected = self._expected_times_to_failures(
            max_failures, seed, n_simulations
        )
        n = np.arange(1, len(expected) + 1)
        # Replace at the n-th failure: n-1 repairs (cr each) then a replacement
        # (co), over an expected cycle length E[T_n].
        cost_rate = (self.cr * (n - 1) + self.co) / expected
        i = int(np.nanargmin(cost_rate))
        return int(n[i]), float(cost_rate[i])

    def find_optimal_replacement_failure_count(
        self,
        seed: Optional[int] = None,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        max_failures: int = _DEFAULT_MAX_FAILURES,
    ) -> int:
        """The failure count at which to replace, minimising the cost rate.

        Under a replace-at-``n``-th-failure policy the unit is repaired
        (cost ``cr``) at each of its first ``n - 1`` failures and replaced
        (cost ``co``, renewing it) at the ``n``-th, so the long-run cost
        rate is ``(cr * (n - 1) + co) / E[T_n]``. ``E[T_1], ...,
        E[T_max_failures]`` come from a single count-terminated
        simulation (see ``expected_time_to_nth_failure()``), and the
        cheapest ``n`` is returned. It needs a simulation-backed
        (imperfect-repair) model.

        ``n = 1`` means replace at every failure. If the simulation fails
        at a high failure count (as surpyval's simulator can for
        near-minimal repair), the count is halved until it succeeds and
        the search is truncated there, with a warning. A result equal to
        the largest count searched may mean the optimum lies beyond it:
        raise ``max_failures`` if the search was not truncated. The cost
        rate is often flat near its minimum, so the chosen count can shift
        with the seed, ``n_simulations`` and ``max_failures``.

        Parameters
        ----------
        seed : int, optional
            Seed for a reproducible simulation. ``None`` (the default)
            draws from numpy's global RNG.
        n_simulations : int, optional
            Number of simulated histories (default 1000).
        max_failures : int, optional
            The largest failure count searched (default 30).

        Returns
        -------
        int
            The optimal number of failures per replacement cycle.

        Raises
        ------
        ValueError
            If the costs have not been set, if the model is analytic
            (``cif``), or if the failure process cannot be simulated at
            all.

        Examples
        --------
        >>> import surpyval as surv
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> from repyability import Repairable
        >>> grp = GeneralizedRenewal.fit_from_parameters(
        ...     [100.0, 2.0], 0.4, kijima="i", dist=surv.Weibull
        ... )
        >>> unit = Repairable(grp)
        >>> unit.set_repair_and_overhaul_costs(cr=1.0, co=5.0)
        >>> unit.find_optimal_replacement_failure_count(
        ...     seed=1, n_simulations=200, max_failures=15
        ... )
        7
        """
        return self._optimise_failure_limit(seed, n_simulations, max_failures)[
            0
        ]

    def optimal_failure_limit_policy(
        self,
        seed: Optional[int] = None,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        max_failures: int = _DEFAULT_MAX_FAILURES,
    ) -> FailureLimitPolicy:
        """The optimal replace-at-N-th-failure policy as a typed result.

        Repair on each failure (cost ``cr``) and replace on the
        ``failure_count``-th (cost ``co``); the long-run cost rate
        ``(cr * (n - 1) + co) / E[T_n]`` is minimised over ``n`` by the
        search of ``find_optimal_replacement_failure_count()`` (see there
        for the method and its limits).

        Parameters
        ----------
        seed : int, optional
            Seed for a reproducible simulation. ``None`` (the default)
            draws from numpy's global RNG.
        n_simulations : int, optional
            Number of simulated histories (default 1000).
        max_failures : int, optional
            The largest failure count searched (default 30).

        Returns
        -------
        FailureLimitPolicy
            ``failure_count`` is the optimal number of failures per
            replacement cycle and ``cost_rate`` the long-run cost per unit
            time under it.

        Raises
        ------
        ValueError
            If the costs have not been set, if the model is analytic
            (``cif``), or if the failure process cannot be simulated at
            all.

        Examples
        --------
        >>> import surpyval as surv
        >>> from surpyval.recurrent import GeneralizedRenewal
        >>> from repyability import Repairable
        >>> grp = GeneralizedRenewal.fit_from_parameters(
        ...     [100.0, 2.0], 0.4, kijima="i", dist=surv.Weibull
        ... )
        >>> unit = Repairable(grp)
        >>> unit.set_repair_and_overhaul_costs(cr=1.0, co=5.0)
        >>> policy = unit.optimal_failure_limit_policy(
        ...     seed=1, n_simulations=200, max_failures=15
        ... )
        >>> policy.failure_count, round(policy.cost_rate, 3)
        (7, 0.031)
        """
        count, rate = self._optimise_failure_limit(
            seed, n_simulations, max_failures
        )
        return FailureLimitPolicy(failure_count=count, cost_rate=rate)
