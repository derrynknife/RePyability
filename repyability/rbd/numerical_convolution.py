"""
Numerical convolution of independent lifetimes.

A cold-standby arrangement (with perfect switching) fails after the *sum* of
its components' lifetimes, so the survival function of the arrangement is the
convolution of the components' distributions. This module computes that
survival function numerically -- deterministically and quickly -- as a robust
alternative to estimating it from Monte-Carlo samples with a Kaplan-Meier fit.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.signal import fftconvolve


def _scalar(value) -> float:
    """Return a plain float from a surpyval scalar/array result."""
    return float(np.atleast_1d(value)[0])


def _upper_time(model, eps: float = 1e-10) -> float:
    """A time by which ``model``'s survival has effectively reached zero.

    Found by doubling from the mean until sf <= eps, so it is robust for any
    distribution exposing sf() and mean() (it does not rely on a quantile
    function).
    """
    t = max(_scalar(model.mean()), 1.0)
    for _ in range(200):
        if _scalar(model.sf(t)) <= eps:
            break
        t *= 2.0
    return t


def _density_on_grid(model, t: np.ndarray) -> np.ndarray:
    """The model's density on the grid, with any non-finite values (e.g. an
    infinite density at t=0 for some shapes) replaced by zero. The negligible
    mass lost is restored by the later CDF normalisation."""
    pdf = np.asarray(model.df(t), dtype=float)
    return np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)


class ConvolvedSurvival:
    """Survival function of a sum of independent lifetimes.

    Computes the survival function of ``X_1 + X_2 + ... + X_n`` (the lifetime
    of a cold-standby arrangement with perfect switching) by numerically
    convolving the component densities on a fine time grid. sf()/ff() then
    interpolate the pre-computed grid, so they are fast and deterministic.

    Parameters
    ----------
    models : sequence
        The component lifetime distributions. Each must expose df() (density),
        sf() (survival) and mean().
    n_points : int, optional
        Number of grid points, by default 100_001. More points give more
        accuracy at the cost of construction time.
    eps : float, optional
        Survival threshold used to bound the time grid, by default 1e-10.
    """

    def __init__(self, models, n_points: int = 100_001, eps: float = 1e-10):
        models = list(models)
        if len(models) == 0:
            raise ValueError("Need at least one model to convolve.")

        # Bound the grid by the sum of the components' effective upper times
        # (the support of the sum is contained in [0, sum of supports]).
        upper = sum(_upper_time(model, eps) for model in models)
        t = np.linspace(0.0, upper, n_points)
        dt = t[1] - t[0]

        # Convolve the component densities to get the density of the sum.
        pdf = _density_on_grid(models[0], t)
        for model in models[1:]:
            pdf = fftconvolve(pdf, _density_on_grid(model, t))[:n_points] * dt

        # Integrate to the CDF, normalising away discretisation drift so the
        # total probability is exactly one, then form the survival function.
        cdf = cumulative_trapezoid(pdf, t, initial=0.0)
        cdf = cdf / cdf[-1]
        self._t = t
        self._sf = np.clip(1.0 - cdf, 0.0, 1.0)

    def sf(self, x):
        """Survival function at x (1 at/below 0, ~0 beyond the grid)."""
        return np.interp(x, self._t, self._sf, left=1.0, right=0.0)

    def ff(self, x):
        """Cumulative failure probability (CDF) at x."""
        return 1.0 - self.sf(x)

    def mean(self, *args, **kwargs) -> float:
        """Mean lifetime, E[T] = integral of the survival function."""
        return float(trapezoid(self._sf, self._t))
