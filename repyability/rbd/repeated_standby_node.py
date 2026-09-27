import numpy as np

from repyability.utils.wrappers import numpy_seed

from ._sampling import RowSampler, column, inverse_sampler
from .numerical_convolution import (
    ConvolvedSurvival,
    is_perfect_switching,
    switch_success_probs,
)


class RepeatedStandbyNode:
    """Cold standby of identical copies of one component, as one node.

    ``repeats`` independent copies of ``model`` in cold standby with one
    operating at a time: the first copy runs and, each time the running
    copy fails, the next is switched in. Spares do not age while dormant,
    so with perfect switching the lifetime is the sum of ``repeats``
    independent lifetimes. With imperfect switching it is a mixture of
    partial sums, since a failed switch ends the node's life when the copy
    it should replace fails. The survival function is computed at
    construction by numerical convolution of the copies' densities on a
    time grid: deterministic (no Monte-Carlo sampling), and linearly
    interpolated when evaluated.

    For different units, ``k >= 2`` or warm and hot spares, use
    [`StandbyModel`][repyability.StandbyModel]. Unlike this node, which
    always convolves numerically, ``StandbyModel([model] * repeats)`` uses
    the exact Erlang form for identical Exponential units with perfect
    switching.

    As an RBD node, ``sf``/``ff`` give its reliability, ``random`` its
    lifetimes for Monte-Carlo system simulation and ``mean`` its MTTF.

    Parameters
    ----------
    model : lifetime model
        The copies' lifetime distribution, e.g. a surpyval distribution. It
        must expose ``df``, ``sf`` and ``mean`` (for the convolution) and
        ``random`` (for sampling).
    repeats : int
        The number of copies, at least 1.
    N : int, optional
        Unused; kept for backwards compatibility (a Kaplan-Meier fit to
        ``N`` simulated lifetimes used to be made here). By default 10_000.
    lower : float, optional
        Unused; kept for backwards compatibility. By default -inf.
    switching_probability : float or sequence of float, optional
        The probability, in ``[0, 1]``, that switching onto the next copy
        succeeds: a scalar for every switch, or one value per switch
        (length ``repeats - 1``). By default 1.0 (perfect switching).

    Raises
    ------
    ValueError
        If ``repeats < 1``, a switching probability is outside ``[0, 1]``,
        or a sequence of them does not have length ``repeats - 1``.

    Examples
    --------
    Three copies of a Weibull unit with mean life 88.62; the lifetimes
    add, so the mean is close to ``3 * 88.62 = 265.87``:

    >>> import surpyval as surv
    >>> from repyability import RepeatedStandbyNode
    >>> unit = surv.Weibull.from_params([100.0, 2.0])
    >>> node = RepeatedStandbyNode(unit, 3)
    >>> round(float(node.mean()), 1)
    265.9
    >>> round(float(node.sf(200.0)), 4)
    0.786

    If each switch works with probability 0.9, the mean falls to about
    ``88.62 * (1 + 0.9 + 0.81)``:

    >>> flaky = RepeatedStandbyNode(unit, 3, switching_probability=0.9)
    >>> round(float(flaky.mean()), 1)
    240.2
    """

    def __init__(
        self,
        model,
        repeats,
        N=10_000,
        lower=-np.inf,
        switching_probability=1.0,
    ):
        # N and lower are kept for backwards compatibility (a Kaplan-Meier fit
        # was previously made here); they are no longer used.
        self.model = model
        self.repeats = repeats
        self.switching_probability = switching_probability

        # Repeated cold standby: the lifetime is the sum of `repeats`
        # independent copies of `model` (a mixture of partial sums under
        # imperfect switching). Its survival function is computed
        # deterministically by numerical convolution.
        self._sf_model = ConvolvedSurvival(
            [model] * repeats, switching_probability=switching_probability
        )

    def random(self, size, seed=None):
        """Draw random lifetimes of the node.

        Each lifetime is the sum of ``repeats`` independent draws of
        ``model``. Under imperfect switching a spare adds its lifetime only
        if every switch up to and including its own succeeds (one uniform
        draw per switch).

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
        """
        # Sum of `repeats` independent draws from the base model. Under
        # imperfect switching a spare only contributes if every switch up to
        # and including its own has succeeded.
        with numpy_seed(seed):
            x_random = np.asarray(self.model.random(size), dtype=float)
            if is_perfect_switching(self.switching_probability):
                for _ in range(self.repeats - 1):
                    x_random = x_random + self.model.random(size)
            else:
                probs = switch_success_probs(
                    self.switching_probability, self.repeats
                )
                running = np.ones(size, dtype=bool)
                for p in probs:
                    running = running & (np.random.random(size) < p)
                    x_random = x_random + np.where(
                        running, self.model.random(size), 0.0
                    )
        return x_random

    def _row_sampler(self):
        """``random(1)`` as a :class:`~._sampling.RowSampler`, so an RBD with
        this node batches its draws; ``None`` unless the model's draws can be
        replayed. Columns follow the order ``random(1)`` draws in: one per
        copy, and under imperfect switching a switch draw before each
        spare's."""
        unit = inverse_sampler(self.model)
        if unit is None:
            return None

        if is_perfect_switching(self.switching_probability):

            def draw(u):
                x = column(u, 0, unit)
                for j in range(1, self.repeats):
                    x = x + column(u, j, unit)
                return x

            return RowSampler(self.repeats, draw)

        probs = switch_success_probs(self.switching_probability, self.repeats)

        def draw(u):
            x = column(u, 0, unit)
            running = np.ones(len(u), dtype=bool)
            for i, p in enumerate(probs):
                running = running & (u[:, 1 + 2 * i] < p)
                x = x + np.where(running, column(u, 2 + 2 * i, unit), 0.0)
            return x

        return RowSampler(1 + 2 * len(probs), draw)

    def mean(self, *args, **kwargs):
        """Mean lifetime (MTTF) of the node, from the convolution.

        ``E[T]``, the integral of the survival function over its grid
        (trapezoidal rule): deterministic, with no sampling.

        Parameters
        ----------
        *args, **kwargs
            Ignored; accepted so ``mean`` can be called like the other
            node models' (e.g. with a sample count).

        Returns
        -------
        float
            The mean lifetime.
        """
        # Exact, deterministic mean from the convolution (E[T] = integral of
        # the survival function).
        return self._sf_model.mean()

    def sf(self, *args, **kwargs):
        """Survival function (reliability) of the node.

        The numerically convolved survival function, linearly interpolated
        on its grid: 1 for ``x`` below 0 and 0 beyond the grid's end
        (placed where the survival is negligible). An RBD calls this for
        the node's reliability.

        Parameters
        ----------
        *args : array_like
            The time(s) ``x``, as in ``sf(x)``.
        **kwargs
            Passed on with ``x``; the only keyword accepted is ``x``
            itself, as in ``sf(x=t)``.

        Returns
        -------
        float or numpy.ndarray
            The probability of surviving beyond ``x``: a numpy float for a
            scalar ``x``, an array for an array ``x``.
        """
        return self._sf_model.sf(*args, **kwargs)

    def ff(self, *args, **kwargs):
        """Cumulative failure probability of the node, ``1 - sf(x)``.

        Parameters
        ----------
        *args : array_like
            The time(s) ``x``, as in ``ff(x)``.
        **kwargs
            Passed on with ``x``; the only keyword accepted is ``x``
            itself, as in ``ff(x=t)``.

        Returns
        -------
        float or numpy.ndarray
            The probability of failing by ``x``: a numpy float for a
            scalar ``x``, an array for an array ``x``.
        """
        return self._sf_model.ff(*args, **kwargs)

    def cs(self, x, X):
        """Conditional survival ``R(x | X) = sf(X + x) / sf(X)``.

        The probability that the node survives a further ``x`` given it has
        already survived to age ``X``, computed from ``sf``.

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
        """
        from repyability.utils.wrappers import conditional_survival

        return conditional_survival(self, x, X)
