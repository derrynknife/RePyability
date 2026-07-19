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
Whichever the unit is, ``Repairable`` prices the same policy: each repair costs
``cr`` while an *overhaul / replacement* costs ``co`` (> ``cr``) and renews the
unit to as-new. Renewing every ``t`` time units makes each cycle a renewal
cycle costing ``cr * E[N(t)] + co``, so the long-run cost rate is::

    g(t) = (cr * E[N(t)] + co) / t

Minimising ``g`` gives the optimal overhaul/replacement interval (the
Barlow-Hunter policy). A finite optimum exists only when the unit wears out
(``E[N(t)]`` growing super-linearly); otherwise repairs never become frequent
enough for renewal to pay and the optimal interval is infinite.

Contrast with ``NonRepairable``, which models renewal *by replacement* ("as
good as new") and the age-replacement policy. RBD components are assumed to
renew on repair, so a repairable unit modelled here is not a valid RBD node
model — ``Repairable`` is a standalone component-level tool.
"""

from typing import Optional, Union

import numpy as np
from scipy.optimize import minimize_scalar

from repyability.maintenance import MaintenancePolicy
from repyability.utils.wrappers import numpy_seed

# Simulation draws used to estimate E[N(t)] for a simulation-backed
# (imperfect-repair) model. Ignored for analytic (``cif``) models.
_DEFAULT_N_SIMULATIONS = 1000

# Default search horizon for the simulated optimum, as a multiple of the
# baseline mean-time-to-first-failure. Imperfect repair pushes the optimal
# renewal interval to several times that mean; simulating much farther is both
# slow and numerically unstable (the virtual-age process asymptotes), so the
# horizon is bounded and exposed as ``max_interval``.
_DEFAULT_HORIZON_MULTIPLE = 15.0


class Repairable:
    """Repairable component with an optimal overhaul/replacement policy.

    Parameters
    ----------
    model : object
        A recurrent-event model exposing the expected number of failures by
        age ``t``, ``E[N(t)]``, as either:

        - ``cif(t)`` — an analytic cumulative intensity (minimal repair), e.g.
          a surpyval ``CrowAMSAA``/``Duane``/``HPP`` (fitted or from params);
          or
        - ``mcf(t, items=..., seed=...)`` — a simulation-estimated mean
          cumulative function (imperfect repair), e.g. a fitted surpyval
          ``GeneralizedRenewal`` (Kijima I/II).

        If both are present ``cif`` is used (analytic, exact).
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
        """True if ``E[N(t)]`` is estimated by simulation (imperfect repair),
        so the policy methods honour ``seed``/``n_simulations``."""
        return not self._analytic

    def set_repair_and_overhaul_costs(self, cr: float, co: float) -> None:
        """Set the repair cost ``cr`` and overhaul/replacement cost ``co``.

        Requires ``0 < cr < co``: an overhaul/replacement (a full renewal) must
        cost more than a repair, otherwise one would simply renew at every
        failure.
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
        """Expected cost of one overhaul/replacement cycle of length ``t``:
        ``cr * E[N(t)] + co``.

        Scalar ``t`` returns a float; array ``t`` returns an array.
        ``seed``/``n_simulations`` apply only to a simulation-backed
        (imperfect-repair) model.
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
        """Long-run cost per unit time when renewing every ``t``:
        ``(cr * E[N(t)] + co) / t``.

        Scalar ``t`` returns a float; array ``t`` returns an array.
        ``seed``/``n_simulations`` apply only to a simulation-backed
        (imperfect-repair) model.
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

        For an analytic (minimal-repair) model, returns ``inf`` when renewal
        never pays: the unit does not wear out (``E[N(t)]`` grows at most
        linearly — e.g. HPP, or Crow-AMSAA with ``beta <= 1``), so the cost
        rate keeps falling as the interval grows.

        For a simulation-backed (imperfect-repair) model, pass a ``seed`` for a
        reproducible result; ``n_simulations`` sets the Monte-Carlo sample size
        and ``max_interval`` the search horizon (default ~15x the baseline
        mean-time-to-first-failure).
        """
        return self._optimise(seed, n_simulations, max_interval)[0]

    def optimal_overhaul_policy(
        self,
        seed: Optional[int] = None,
        n_simulations: int = _DEFAULT_N_SIMULATIONS,
        max_interval: Optional[float] = None,
    ) -> MaintenancePolicy:
        """The optimal overhaul/replacement policy as a typed result.

        Returns a ``MaintenancePolicy`` carrying the optimal interval and the
        long-run cost rate under it. For an analytic model an ``inf`` interval
        (never renew) reports the limiting rate of repairs alone.

        For a simulation-backed (imperfect-repair) model, pass a ``seed`` for a
        reproducible result; ``n_simulations`` sets the Monte-Carlo sample size
        and ``max_interval`` the search horizon.
        """
        interval, rate = self._optimise(seed, n_simulations, max_interval)
        return MaintenancePolicy(interval=interval, cost_rate=rate)
