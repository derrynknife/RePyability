"""
Numerical convolution of independent lifetimes.

A cold-standby arrangement fails after the *sum* of its components' lifetimes
(when switching is perfect), so the survival function of the arrangement is
the convolution of the components' distributions. With imperfect switching it
is a mixture of partial sums, weighted by how many switches succeed. This
module computes that survival function numerically -- deterministically and
quickly, from the components' cumulative distribution functions, rather
than estimating it from Monte-Carlo samples.
"""

from typing import cast

import numpy as np

from ._model_utils import failure_time_scale, never_fails


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


def _cdf(model, x: np.ndarray) -> np.ndarray:
    """The model's CDF at the times ``x``: its ``ff``, or ``1 - sf`` for a
    model with no ``ff``, as a flat float array. For a surpyval model it
    includes the units dead on arrival (``F(0) = f0``) and levels off below
    1 when some units never fail (``F(inf) = p``)."""
    ff = getattr(model, "ff", None)
    # At 0 some models take log(0) = -inf on the way to a CDF of 0.
    with np.errstate(divide="ignore"):
        if callable(ff):
            values = np.asarray(ff(x), dtype=float)
        else:
            values = 1.0 - np.asarray(model.sf(x), dtype=float)
    return np.nan_to_num(np.ravel(values), nan=0.0)


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


class ConvolvedSurvival:
    """Survival function of a cold-standby arrangement (sum of lifetimes).

    With perfect switching the arrangement fails after the sum of its
    components' lifetimes, whose survival function is the convolution of the
    component distributions. With imperfect switching, each switch onto the
    next spare succeeds only with some probability, so the lifetime is a
    mixture of partial sums (run the first component; if its switch works, run
    the second too; and so on). This computes that mixture numerically on a
    fine time grid. ``sf``/``ff`` then interpolate the pre-computed grid, so
    they are fast and deterministic.

    The grid runs from 0 to the sum of the components' upper times (for
    each, the smallest ``max(mean, 1) * 2 ** j`` at which its survival is
    at most ``eps``). Each partial sum's CDF is kept on it: the next
    component is added by convolving (by FFT) the probability that the sum
    so far ends in each cell of the grid with that component's CDF half a
    cell back. Cumulative probabilities stay finite where densities do not
    (at 0, for a Weibull or gamma with shape below 1), so each cell's
    probability is exact. The result is accurate to about ``1e-6`` or
    better, the error falling as the square of the grid step, except for
    the steepest early-life densities: about ``1e-5`` for gamma units of
    shape 0.5 or less, whose error falls more slowly (raise ``n_points``
    for more).

    A component may be a limited-failure-population or zero-inflated
    surpyval model: a fraction ``1 - p`` of its units never fail, and a
    fraction ``f0`` fail at 0. A sum with a unit that never fails never
    fails, and one of units all dead on arrival is 0, so a partial sum is
    finite with probability ``prod(p)`` and 0 with probability
    ``prod(f0)``. Both come in through the components' CDFs
    (``F(0) = f0``, ``F(inf) = p``), and ``sf`` levels off at the
    probability that the arrangement never fails (``mean`` is then
    infinite).

    Parameters
    ----------
    models : sequence
        The component lifetime distributions, in standby order (primary
        first). Each must expose ``sf`` (survival) and ``mean``; its ``ff``
        (CDF) is used when it has one.
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
    partials : bool, optional
        Whether to keep the survival function of every partial sum
        ``T_1 + ... + T_j`` too (see ``partial_sf``), by default False.

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
        partials: bool = False,
    ):
        from scipy.signal import fftconvolve

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

        # The CDF on the grid of each partial sum T_1 + ... + T_k. The first
        # is T_1's own; each next adds T to the sum S so far by
        # conditioning on the cell S ends in:
        #   P(S + T <= t_i) = P(S = 0) F_T(t_i)
        #                   + sum_j P(t_j < S <= t_j+1) F_T(t_i - t_j - dt/2),
        # a convolution of S's cell probabilities with T's CDF half a cell
        # back (nothing from the cell after t_i: T is never below 0). A
        # cumulative probability stays finite where a density does
        # not (at 0 for a Weibull with shape below 1), so each cell's
        # probability is exact; a unit dead on arrival or never failing
        # comes in through its CDF too (F(0) = f0, F(inf) = p).
        back = t[1:] - 0.5 * dt
        cdf = _cdf(models[0], t)
        finite = 1.0 - never_fails(models[0])
        sums = [(cdf, finite)]
        # Each model's CDF half a cell back, once for a model given twice.
        shifted: dict = {}
        for model in models[1:]:
            later = shifted.get(id(model))
            if later is None:
                later = np.concatenate(([0.0], _cdf(model, back)))
                shifted[id(model)] = later
            summed = fftconvolve(np.diff(cdf), later)[:n_points]
            if cdf[0] > 0.0:
                # The sum so far is 0 (every unit dead on arrival) with
                # probability cdf[0], and the new sum is then T.
                summed += cdf[0] * _cdf(model, t)
            cdf = summed
            finite *= 1.0 - never_fails(model)
            sums.append((cdf, finite))

        # Survival function is the weighted mixture of the partial-sum survival
        # functions (zero-weight partials are skipped, so perfect switching
        # only evaluates the full convolution, unless every partial sum's is
        # to be kept).
        sf = np.zeros(n_points)
        never = 0.0
        self._partials: list = []
        for weight, (partial_cdf, finite) in zip(weights, sums):
            if weight == 0.0 and not partials:
                continue
            partial = np.clip(1.0 - partial_cdf, 0.0, 1.0)
            if partials:
                self._partials.append((partial, 1.0 - finite))
            sf += weight * partial
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

    def partial_sf(self, j: int, x):
        """Survival function of the partial sum ``T_1 + ... + T_j`` at x,
        for ``j`` from 1 to the number of models, with perfect switching
        (kept only if the object was built with ``partials=True``).

        Parameters
        ----------
        j : int
            How many of the lifetimes to add, from 1.
        x : float or array_like
            The time(s) at which to evaluate.

        Returns
        -------
        float or numpy.ndarray
            The survival probability, shaped as for ``sf``.
        """
        curve, never = self._partials[j - 1]
        return np.interp(x, self._t, curve, left=1.0, right=never)

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
        from scipy.integrate import trapezoid

        if self.never_fails > 0.0:
            return float("inf")
        return float(trapezoid(self._sf, self._t))
