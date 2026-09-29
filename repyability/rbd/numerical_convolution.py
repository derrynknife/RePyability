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

from ._model_utils import distribution_name, failure_time_scale, never_fails


def _scalar(value) -> float:
    """Return a plain float from a surpyval scalar/array result."""
    return float(np.atleast_1d(value)[0])


def _upper_time(model, eps: float = 1e-10) -> float:
    """A time by which ``model``'s survival has effectively reached its
    floor: zero, or the fraction of units that never fail.

    Found by doubling from a typical failure time (the mean lifetime, or
    that of the units that fail when some never do) until sf <= eps above
    that floor, so it is robust for any distribution exposing sf() and
    mean() (it does not rely on a quantile function).
    """
    floor = never_fails(model)
    scale = failure_time_scale(model)
    t = max(scale, 1.0) if np.isfinite(scale) else 1.0
    for _ in range(200):
        if _scalar(model.sf(t)) - floor <= eps:
            break
        t *= 2.0
    return t


def _dead_on_arrival(model) -> float:
    """The fraction of units dead on arrival (failed at time 0): a
    zero-inflated surpyval model's ``f0``, and 0 for any other."""
    if distribution_name(model) is None:
        return 0.0
    return float(getattr(model, "f0", 0.0) or 0.0)


def _density_on_grid(model, t: np.ndarray) -> np.ndarray:
    """The density of the model's continuous part on the grid, with any
    non-finite values (e.g. an infinite density at t=0 for some shapes)
    replaced by zero. The negligible mass lost is restored by the later CDF
    normalisation. For a zero-inflated model that leaves out the units dead
    on arrival, whose mass its ``df`` otherwise gives at exactly 0."""
    if _dead_on_arrival(model):
        pdf = model.df(t, continuous=True)
    else:
        pdf = model.df(t)
    pdf = np.asarray(pdf, dtype=float)
    return np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)


def switch_success_probs(switching_probability, n) -> list:
    """Normalise a switching probability into per-switch probabilities.

    Returns the ``n - 1`` success probabilities of the switches into
    components 2, ..., n of a standby chain.

    Parameters
    ----------
    switching_probability : float or sequence of float
        A scalar (the same probability for every switch) or a sequence of
        length ``n - 1`` (one success probability per switch), each in
        ``[0, 1]``.
    n : int
        The number of components in the chain.

    Returns
    -------
    list of float
        The per-switch success probabilities; empty when ``n <= 1``, in
        which case ``switching_probability`` is not checked.

    Raises
    ------
    ValueError
        If a sequence does not have length ``n - 1``, or a probability is
        outside ``[0, 1]``.

    Examples
    --------
    >>> from repyability.rbd.numerical_convolution import switch_success_probs
    >>> switch_success_probs(0.9, 3)
    [0.9, 0.9]
    >>> switch_success_probs([1.0, 0.5], 3)
    [1.0, 0.5]
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
    """True if every switch succeeds with probability 1.

    Parameters
    ----------
    switching_probability : float or sequence of float
        A scalar, or one success probability per switch.

    Returns
    -------
    bool
        Whether the scalar, or every element of the sequence, equals 1.0
        (an empty sequence counts as perfect). Values are not
        range-checked.

    Examples
    --------
    >>> from repyability.rbd.numerical_convolution import is_perfect_switching
    >>> is_perfect_switching(1.0), is_perfect_switching([1.0, 0.9])
    (True, False)
    """
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


def _sf_from_pdf(
    pdf: np.ndarray, t: np.ndarray, at_zero: float = 0.0, finite: float = 1.0
) -> np.ndarray:
    """Survival function on the grid of a lifetime that is 0 with
    probability ``at_zero``, finite with probability ``finite`` (the rest
    never fail) and otherwise has the (possibly un-normalised) density
    ``pdf``, scaled to its mass ``finite - at_zero`` to normalise away
    discretisation drift."""
    cdf = cumulative_trapezoid(pdf, t, initial=0.0)
    total = cdf[-1]
    mass = finite - at_zero
    if total <= 0.0 or mass <= 0.0:
        return np.full_like(t, 1.0 - at_zero)
    return np.clip(1.0 - at_zero - cdf / total * mass, 0.0, 1.0)


class ConvolvedSurvival:
    """Survival function of a cold-standby arrangement (sum of lifetimes).

    With perfect switching the arrangement fails after the sum of its
    components' lifetimes, whose survival function is the convolution of the
    component distributions. With imperfect switching, each switch onto the
    next spare succeeds only with some probability, so the lifetime is a
    mixture of partial sums (run the first component; if its switch works, run
    the second too; and so on). This computes that mixture by numerically
    convolving the component densities on a fine time grid. ``sf``/``ff``
    then interpolate the pre-computed grid, so they are fast and
    deterministic.

    The grid runs from 0 to the sum of the components' upper times (for
    each, the smallest ``max(mean, 1) * 2 ** j`` at which its survival is
    at most ``eps``). The densities are evaluated on it (non-finite values
    set to 0), convolved by FFT, and each partial sum's distribution is
    normalised to total probability 1 before the mixture is formed.

    A component may be a limited-failure-population or zero-inflated
    surpyval model: a fraction ``1 - p`` of its units never fail, and a
    fraction ``f0`` fail at 0. A sum with a unit that never fails never
    fails, and one of units all dead on arrival is 0, so a partial sum is
    finite with probability ``prod(p)``, 0 with probability ``prod(f0)``,
    and continuous otherwise: the convolution carries the three parts, and
    ``sf`` levels off at the probability that the arrangement never fails
    (``mean`` is then infinite).

    Parameters
    ----------
    models : sequence
        The component lifetime distributions, in standby order (primary
        first). Each must expose ``df`` (density), ``sf`` (survival) and
        ``mean``.
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

    Attributes
    ----------
    switching_weights : list of float
        The mixture weights: ``switching_weights[i]`` is the probability
        that exactly the first ``i + 1`` components run (the switches
        before succeed and the next one fails, or, for the last, every
        switch succeeds). They sum to 1.

    Raises
    ------
    ValueError
        If ``models`` is empty, a switching probability is outside
        ``[0, 1]``, or a sequence of them has the wrong length.

    Examples
    --------
    The sum of two Exponential(1) lifetimes is Erlang(2, 1), whose
    survival function is ``exp(-t) * (1 + t)``; the numerical result is
    close to it:

    >>> import surpyval as surv
    >>> from repyability.rbd.numerical_convolution import ConvolvedSurvival
    >>> unit = surv.Exponential.from_params([1.0])
    >>> conv = ConvolvedSurvival([unit, unit])
    >>> round(float(conv.sf(2.0)), 3)  # exact: exp(-2) * 3 = 0.406
    0.406
    >>> round(conv.mean(), 2)  # exact: 2
    2.0
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
        # T_1 + ... + T_k (k = 1..n), with the probabilities that it is 0
        # (every unit dead on arrival) and finite (none never fails): the
        # continuous part of T + T_k is the continuous parts convolved, plus
        # each one's continuous part while the other is 0.
        partials = []
        at_zero = _dead_on_arrival(models[0])
        finite = 1.0 - never_fails(models[0])
        pdf = _density_on_grid(models[0], t)
        partials.append((at_zero, finite, pdf))
        for model in models[1:]:
            density = _density_on_grid(model, t)
            zero = _dead_on_arrival(model)
            summed = fftconvolve(pdf, density)[:n_points] * dt
            if at_zero:
                summed = summed + at_zero * density
            if zero:
                summed = summed + zero * pdf
            pdf = summed
            at_zero *= zero
            finite *= 1.0 - never_fails(model)
            partials.append((at_zero, finite, pdf))

        # Survival function is the weighted mixture of the partial-sum survival
        # functions (zero-weight partials are skipped, so perfect switching
        # only evaluates the full convolution).
        sf = np.zeros(n_points)
        never = 0.0
        for weight, (at_zero, finite, partial_pdf) in zip(weights, partials):
            if weight == 0.0:
                continue
            sf += weight * _sf_from_pdf(partial_pdf, t, at_zero, finite)
            never += weight * (1.0 - finite)

        self._t = t
        self._sf = np.clip(sf, 0.0, 1.0)
        #: The probability that the arrangement never fails.
        self.never_fails = never

    def sf(self, x):
        """Survival function at x (1 below 0, ~0 beyond the grid).

        Linear interpolation of the pre-computed curve: 1 for ``x`` below
        the grid (``x < 0``) and, beyond its end, the probability that the
        arrangement never fails (0 unless a component has units that never
        fail).

        Parameters
        ----------
        x : float or array_like
            The time(s) at which to evaluate.

        Returns
        -------
        float or numpy.ndarray
            The survival probability: a numpy float for a scalar ``x``, an
            array of the same shape for an array ``x``.
        """
        return np.interp(
            x, self._t, self._sf, left=1.0, right=self.never_fails
        )

    def ff(self, x):
        """Cumulative failure probability (CDF) at x.

        Parameters
        ----------
        x : float or array_like
            The time(s) at which to evaluate.

        Returns
        -------
        float or numpy.ndarray
            ``1 - sf(x)``, shaped as for ``sf``.
        """
        return 1.0 - self.sf(x)

    def mean(self, *args, **kwargs) -> float:
        """Mean lifetime, E[T] = integral of the survival function.

        Integrated over the grid with the trapezoidal rule; infinite when
        the arrangement may never fail.

        Parameters
        ----------
        *args, **kwargs
            Ignored; accepted so ``mean`` can be called like other models'.

        Returns
        -------
        float
            The mean lifetime.
        """
        if self.never_fails > 0.0:
            return float("inf")
        return float(trapezoid(self._sf, self._t))
