"""
Numerical convolution of independent lifetimes.

A cold-standby arrangement fails after the *sum* of its components' lifetimes
(when switching is perfect), so the survival function of the arrangement is
the convolution of the components' distributions. With imperfect switching it
is a mixture of partial sums, weighted by how many switches succeed. This
module computes that survival function numerically -- deterministically and
quickly -- as a robust alternative to estimating it from Monte-Carlo samples
with a Kaplan-Meier fit.
"""

from typing import cast

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


def switch_success_probs(switching_probability, n) -> list:
    """Normalise a switching_probability into a list of (n-1) per-switch
    success probabilities (the switches into components 2..n).

    switching_probability is either a scalar (same probability for every
    switch) or a sequence of length n-1 (one success probability per switch).
    """
    if n <= 1:
        return []
    if np.isscalar(switching_probability):
        probs = [float(cast(float, switching_probability))] * (n - 1)
    else:
        probs = [float(p) for p in switching_probability]
        if len(probs) != n - 1:
            raise ValueError(
                "switching_probability sequence must have length "
                f"{n - 1} (one per switch), got {len(probs)}"
            )
    for p in probs:
        if not (0.0 <= p <= 1.0):
            raise ValueError("switching probabilities must be in [0, 1]")
    return probs


def is_perfect_switching(switching_probability) -> bool:
    """True if every switch succeeds with probability 1."""
    if np.isscalar(switching_probability):
        return float(cast(float, switching_probability)) == 1.0
    return all(float(p) == 1.0 for p in switching_probability)


def _switching_weights(switching_probability, n) -> list:
    """Mixture weights for a cold-standby chain with imperfect switching.

    Returns a list ``w`` of length n where ``w[k]`` is the probability that
    exactly the first (k+1) components run: switches 1..k succeeded and switch
    (k+1) failed (or, for the last, every switch succeeded).
    """
    if n == 1:
        return [1.0]
    probs = switch_success_probs(switching_probability, n)
    weights = []
    prefix = 1.0
    for p in probs:
        # This switch fails (after all earlier ones succeeded): stop here.
        weights.append(prefix * (1.0 - p))
        prefix *= p
    # Every switch succeeded: all n components run.
    weights.append(prefix)
    return weights


def _sf_from_pdf(pdf: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Survival function on the grid from a (possibly un-normalised) density,
    normalising away discretisation drift."""
    cdf = cumulative_trapezoid(pdf, t, initial=0.0)
    total = cdf[-1]
    if total <= 0.0:
        return np.ones_like(t)
    return np.clip(1.0 - cdf / total, 0.0, 1.0)


class ConvolvedSurvival:
    """Survival function of a cold-standby arrangement (sum of lifetimes).

    With perfect switching the arrangement fails after the sum of its
    components' lifetimes, whose survival function is the convolution of the
    component distributions. With imperfect switching, each switch onto the
    next spare succeeds only with some probability, so the lifetime is a
    mixture of partial sums (run the first component; if its switch works, run
    the second too; and so on). This computes that mixture by numerically
    convolving the component densities on a fine time grid. sf()/ff() then
    interpolate the pre-computed grid, so they are fast and deterministic.

    Parameters
    ----------
    models : sequence
        The component lifetime distributions, in standby order (primary
        first). Each must expose df() (density), sf() (survival) and mean().
    switching_probability : float or sequence, optional
        Probability that a switch onto the next spare succeeds. A scalar
        applies to every switch; a sequence gives one probability per switch
        (length len(models) - 1). By default 1.0 (perfect switching), which
        reduces to the plain convolution.
    n_points : int, optional
        Number of grid points, by default 100_001. More points give more
        accuracy at the cost of construction time.
    eps : float, optional
        Survival threshold used to bound the time grid, by default 1e-10.
    """

    def __init__(
        self,
        models,
        switching_probability=1.0,
        n_points: int = 100_001,
        eps: float = 1e-10,
    ):
        models = list(models)
        n = len(models)
        if n == 0:
            raise ValueError("Need at least one model to convolve.")

        weights = _switching_weights(switching_probability, n)
        self.switching_weights = weights

        # Bound the grid by the sum of the components' effective upper times
        # (the support of the full sum is contained in [0, sum of supports]).
        upper = sum(_upper_time(model, eps) for model in models)
        t = np.linspace(0.0, upper, n_points)
        dt = t[1] - t[0]

        # Incrementally convolve to get the density of each partial sum
        # T_1 + ... + T_k (k = 1..n).
        partial_pdfs = []
        pdf = _density_on_grid(models[0], t)
        partial_pdfs.append(pdf)
        for model in models[1:]:
            pdf = fftconvolve(pdf, _density_on_grid(model, t))[:n_points] * dt
            partial_pdfs.append(pdf)

        # Survival function is the weighted mixture of the partial-sum survival
        # functions (zero-weight partials are skipped, so perfect switching
        # only evaluates the full convolution).
        sf = np.zeros(n_points)
        for weight, partial_pdf in zip(weights, partial_pdfs):
            if weight == 0.0:
                continue
            sf += weight * _sf_from_pdf(partial_pdf, t)

        self._t = t
        self._sf = np.clip(sf, 0.0, 1.0)

    def sf(self, x):
        """Survival function at x (1 at/below 0, ~0 beyond the grid)."""
        return np.interp(x, self._t, self._sf, left=1.0, right=0.0)

    def ff(self, x):
        """Cumulative failure probability (CDF) at x."""
        return 1.0 - self.sf(x)

    def mean(self, *args, **kwargs) -> float:
        """Mean lifetime, E[T] = integral of the survival function."""
        return float(trapezoid(self._sf, self._t))
