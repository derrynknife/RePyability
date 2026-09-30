import numpy as np

from repyability.utils.deprecation import ignored, renamed
from repyability.utils.wrappers import numpy_seed

from ._mean_lifetime import mean_lifetime, model_knots
from ._sampling import RowSampler, inverse_sampler

REPEATED_NODE_TYPES = {"parallel", "series"}
PARALLEL = 1
SERIES = 0


class RepeatedNode:
    """Independent identical copies of one component, in series or parallel.

    One RBD node standing for ``repeats`` independent copies of ``model``,
    arranged in series (the node fails at the first copy failure) or in
    parallel (at the last). Its survival function is exact:
    ``R(x) ** repeats`` in series and ``1 - F(x) ** repeats`` in parallel,
    with ``R`` and ``F`` the model's ``sf`` and ``ff``, and so is its
    ``mean``, the area under ``sf``.

    As an RBD node, ``sf``/``ff`` give its reliability, ``random`` its
    lifetimes for Monte-Carlo system simulation and ``mean`` its MTTF.
    The copies fail independently. This differs from a repeated node of an
    RBD (a ``reliabilities`` entry that names another node), which puts
    one component in several places of the diagram, failing in all of them
    at once.

    Parameters
    ----------
    model : lifetime model
        The component's lifetime distribution, e.g. a surpyval
        distribution. Sampling calls its ``random`` with a
        ``(size, repeats)`` shape, which surpyval distributions accept.
    repeats : int
        The number of copies, at least 1.
    kind : {'series', 'parallel'}
        How the copies are arranged.

    Raises
    ------
    ValueError
        If ``kind`` is neither ``'series'`` nor ``'parallel'``.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import RepeatedNode
    >>> unit = surv.Weibull.from_params([100.0, 2.0])
    >>> series = RepeatedNode(unit, 3, "series")
    >>> round(float(series.sf(50.0)), 4)  # unit.sf(50) ** 3
    0.4724
    >>> parallel = RepeatedNode(unit, 3, "parallel")
    >>> round(float(parallel.sf(50.0)), 4)  # 1 - unit.ff(50) ** 3
    0.9892
    """

    def __init__(self, model, repeats, kind):
        if kind not in REPEATED_NODE_TYPES:
            raise ValueError("'kind' must be either 'parallel' or 'series'")

        self.model = model
        if kind == "parallel":
            self.kind = PARALLEL
        else:
            self.kind = SERIES
        self.repeats = repeats

    def random(self, size, seed=None):
        """Draw random lifetimes of the node.

        Draws ``repeats`` independent lifetimes of ``model`` per sample and
        takes their minimum (series) or maximum (parallel).

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
        with numpy_seed(seed):
            randoms = np.asarray(self.model.random((size, self.repeats)))
        if self.kind == SERIES:
            # If repetition is in series, then a random event will be the
            # smallest of all the events in series. i.e. when the first item
            # fails.
            return randoms.min(axis=1)
        else:
            # If repetition is in parallel, then a random event will be the
            # largest of all the events in series. i.e. when the last item
            # fails.
            return randoms.max(axis=1)

    def _row_sampler(self):
        """``random(1)`` as a :class:`~._sampling.RowSampler` (``repeats``
        draws of the model, in order), so an RBD with this node batches its
        draws; ``None`` unless the model's draws can be replayed."""
        sampler = inverse_sampler(self.model)
        if sampler is None:
            return None
        reduce = np.min if self.kind == SERIES else np.max

        def draw(u):
            draws = sampler(np.ascontiguousarray(u))
            return reduce(np.asarray(draws, dtype=float), axis=1)

        return RowSampler(self.repeats, draw)

    def mean(self, mc_samples=None, seed=None, *, method="exact", N=None):
        """Mean lifetime (MTTF) of the node.

        Exact by default: the area under ``sf``, integrated by adaptive
        Gauss-Legendre quadrature to about ``1e-10``, relative, and ``inf``
        if the node may never fail (its model has units that never fail,
        and the node needs only them). ``method="simulate"`` estimates it
        instead, as the mean of ``mc_samples`` draws of ``random``.

        Parameters
        ----------
        mc_samples : int, optional
            With ``method="simulate"``: the number of draws, by default
            1_000_000.
        seed : int or None, optional
            With ``method="simulate"``: the seed for the draws (see
            ``random``), by default None.
        method : {"exact", "simulate"}, optional
            How to find the mean, by default ``"exact"``.
        N : int, optional
            Deprecated: the old name of ``mc_samples``.

        Returns
        -------
        float
            The mean lifetime, or with ``method="simulate"`` its estimate.

        Raises
        ------
        ValueError
            If ``method`` is neither ``"exact"`` nor ``"simulate"``.

        Warns
        -----
        DeprecationWarning
            If ``N`` is given, or a simulation option without
            ``method="simulate"`` (it is ignored).

        Examples
        --------
        >>> import surpyval as surv
        >>> from repyability import RepeatedNode
        >>> unit = surv.Weibull.from_params([100.0, 2.0])
        >>> node = RepeatedNode(unit, 3, "series")
        >>> round(node.mean(), 4)  # 100 * Gamma(1.5) / sqrt(3)
        51.1663
        >>> round(node.mean(method="simulate", mc_samples=10_000, seed=1), 1)
        51.1
        """
        mc_samples = renamed("mc_samples", mc_samples, "N", N)
        if method == "exact":
            ignored(
                "RepeatedNode.mean()",
                "the mean is exact unless method='simulate'.",
                {"mc_samples": mc_samples, "seed": seed},
            )
            return mean_lifetime(self.sf, model_knots(self.model))
        if method != "simulate":
            raise ValueError(
                f"method must be 'exact' or 'simulate', got {method!r}."
            )
        count = 1_000_000 if mc_samples is None else mc_samples
        return float(self.random(count, seed=seed).mean())

    def sf(self, x):
        """Survival function (reliability) of the node, exactly.

        ``model.sf(x) ** repeats`` in series and
        ``1 - model.ff(x) ** repeats`` in parallel. An RBD calls this for
        the node's reliability.

        Parameters
        ----------
        x : float or array_like
            The time(s) at which to evaluate.

        Returns
        -------
        float or numpy.ndarray
            The probability of surviving beyond ``x``, as a scalar or an
            array following ``model.sf``.
        """
        if self.kind == SERIES:
            sf = self.model.sf(x) ** self.repeats
        else:
            sf = 1 - (self.model.ff(x) ** self.repeats)
        return sf

    def ff(self, x):
        """Cumulative failure probability of the node, ``1 - sf(x)``.

        ``1 - model.sf(x) ** repeats`` in series and
        ``model.ff(x) ** repeats`` in parallel.

        Parameters
        ----------
        x : float or array_like
            The time(s) at which to evaluate.

        Returns
        -------
        float or numpy.ndarray
            The probability of failing by ``x``, as a scalar or an array
            following ``model.ff``.
        """
        if self.kind == SERIES:
            ff = 1 - (self.model.sf(x) ** self.repeats)
        else:
            ff = self.model.ff(x) ** self.repeats
        return ff

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
