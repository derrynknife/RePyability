"""Minimal-repair ("as bad as old") component economics.

A unit restored by *minimal repair* keeps its accumulated age through every
fix: repairs return it to service but do not rejuvenate it, so failures
recur with rising frequency. That failure history is a recurrent-event
process with cumulative intensity ``Lambda(t)`` — the expected number of
failures by age ``t`` — available from a surpyval recurrence model as
``model.cif(t)`` (e.g. Crow-AMSAA / NHPP, Duane, HPP).

``Repairable`` prices the classic trade-off for such a unit: each minimal
repair costs ``cr``, while an *overhaul* costs ``co`` (> ``cr``) and renews
the unit to as-new. Overhauling every ``t`` time units makes each overhaul
cycle a renewal cycle costing ``cr * Lambda(t) + co``, so the long-run cost
rate is::

    g(t) = (cr * Lambda(t) + co) / t

Minimising ``g`` gives the optimal overhaul interval (the Barlow-Hunter
policy). A finite optimum exists only for wear-out models (``Lambda``
growing super-linearly, e.g. Crow-AMSAA with ``beta > 1``); otherwise
repairs never become frequent enough for overhauls to pay and the optimal
interval is infinite.

Contrast with ``NonRepairable``, which models renewal *by replacement*
("as good as new") and the age-replacement policy. RBD components are
assumed to renew on repair, so a minimal-repair unit is not a valid RBD
node model — ``Repairable`` is a standalone component-level tool.
"""

from typing import Union

import numpy as np
from scipy.optimize import minimize_scalar

from repyability.maintenance import MaintenancePolicy


class Repairable:
    """Minimal-repair component with an optimal-overhaul-interval policy.

    Parameters
    ----------
    model : object
        A recurrent-event model exposing ``cif(t)``, the cumulative
        intensity (expected number of failures by age ``t``) — e.g. a
        surpyval recurrence model (``CrowAMSAA``, ``Duane``, ``HPP``),
        fitted or built ``from_params``.
    """

    def __init__(self, model):
        if not hasattr(model, "cif"):
            raise ValueError(
                "model must expose cif() (a cumulative intensity "
                "function), e.g. a surpyval recurrence model such as "
                "CrowAMSAA"
            )
        self.model = model

    def set_repair_and_overhaul_costs(self, cr: float, co: float) -> None:
        """Set the minimal-repair cost ``cr`` and overhaul cost ``co``.

        Requires ``0 < cr < co``: an overhaul (a full renewal) must cost
        more than a minimal repair, otherwise one would simply overhaul at
        every failure.
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

    def _cif_scalar(self, t: float) -> float:
        return float(np.asarray(self.model.cif(t), dtype=float).item())

    def _g(self, t: float) -> float:
        """Cost rate at a scalar ``t > 0`` (float in, float out)."""
        return (self.cr * self._cif_scalar(t) + self.co) / t

    def cost(self, t) -> Union[float, np.ndarray]:
        """Expected cost of one overhaul cycle of length ``t``:
        ``cr * Lambda(t) + co``.

        Scalar ``t`` returns a float; array ``t`` returns an array.
        """
        self._require_costs()
        scalar_in = np.ndim(t) == 0
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        lam = np.asarray(self.model.cif(tt), dtype=float)
        out = self.cr * lam + self.co
        return out.item() if scalar_in else out

    def cost_rate(self, t) -> Union[float, np.ndarray]:
        """Long-run cost per unit time when overhauling every ``t``:
        ``(cr * Lambda(t) + co) / t``.

        Scalar ``t`` returns a float; array ``t`` returns an array.
        """
        self._require_costs()
        scalar_in = np.ndim(t) == 0
        tt = np.atleast_1d(np.asarray(t, dtype=float))
        lam = np.asarray(self.model.cif(tt), dtype=float)
        with np.errstate(divide="ignore"):
            out = (self.cr * lam + self.co) / tt
        return out.item() if scalar_in else out

    def _optimise(self) -> tuple[float, float]:
        """Return ``(optimal interval, cost rate at it)``; the interval is
        ``inf`` (with the limiting repair-only cost rate) when overhauls
        never pay."""
        self._require_costs()

        # At a finite optimum the cumulative repair spend is comparable to
        # the overhaul cost (for a power law, Lambda(t*) = co/(cr*(b-1))),
        # so first bracket the age where Lambda(t) reaches co/cr and centre
        # the search grid on that scale.
        target = self.co / self.cr
        t_scale = 1.0
        if self._cif_scalar(t_scale) < target:
            for _ in range(300):
                t_scale *= 2.0
                if self._cif_scalar(t_scale) >= target:
                    break
        else:
            for _ in range(300):
                if self._cif_scalar(t_scale / 2.0) < target:
                    break
                t_scale /= 2.0

        for _ in range(6):
            grid = t_scale * np.logspace(-4.0, 4.0, 1601)
            g = np.asarray(self.cost_rate(grid))
            i = int(np.argmin(g))
            if i == len(grid) - 1:
                # Minimum at the right edge: overhauling later keeps
                # getting cheaper. If Lambda is not growing super-linearly
                # out here, it never stops getting cheaper (no wear-out) —
                # never overhaul.
                big_t = float(grid[-1])
                lam_t = self._cif_scalar(big_t)
                lam_2t = self._cif_scalar(2.0 * big_t)
                if lam_t <= 0.0 or (
                    np.log(lam_2t / lam_t) / np.log(2.0) <= 1.0 + 1e-9
                ):
                    return float(np.inf), self._g(big_t)
                t_scale *= 1e4  # genuine wear-out; optimum is further out
                continue
            if i == 0:
                t_scale *= 1e-4
                continue
            res = minimize_scalar(
                self._g,
                bounds=(float(grid[i - 1]), float(grid[i + 1])),
                method="bounded",
            )
            t_star = float(res.x)
            return t_star, self._g(t_star)
        # Search saturated (pathological model): best point found.
        t_star = float(grid[i])
        return t_star, self._g(t_star)

    def find_optimal_overhaul_interval(self) -> float:
        """The overhaul interval minimising the long-run cost rate.

        Returns ``inf`` when overhauls never pay: the model does not wear
        out (``Lambda(t)`` grows at most linearly — e.g. HPP, or Crow-AMSAA
        with ``beta <= 1``), so the cost rate keeps falling as the
        interval grows.
        """
        return self._optimise()[0]

    def optimal_overhaul_policy(self) -> MaintenancePolicy:
        """The optimal overhaul policy as a typed result.

        Returns a ``MaintenancePolicy`` carrying the optimal interval and
        the long-run cost rate under it. When the interval is ``inf``
        (never overhaul), the cost rate is the limiting rate of minimal
        repairs alone.
        """
        interval, rate = self._optimise()
        return MaintenancePolicy(interval=interval, cost_rate=rate)
