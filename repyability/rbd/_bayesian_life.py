"""A surpyval Bayesian fit as a life: its posterior predictive.

surpyval's ``fit_bayes`` gives a ``BayesianParametric``: the posterior of a
distribution's parameters, as points with weights (a grid's, a sampler's
draws, or the nodes of a conjugate posterior). The life of a new unit, with
the uncertainty in the parameters averaged in, is the posterior predictive
distribution, ``S(t) = sum_i w_i S(t | theta_i)``: a mixture of the
distribution at the posterior's points, weighted by their probabilities
(equal for draws). ``PosteriorPredictive`` is that mixture as a node model,
each function to its own precision (its ``ff``, ``Hf`` and ``qf`` are the
fit's own, which keep their digits in both tails from surpyval 0.27), and
its draws a point and a life from one uniform. Where the fit has the predictive
in closed form (an Exponential with a Gamma posterior on its rate: a Lomax;
a Normal mean with the spread known), that is used, as surpyval does. A fit
with an offset, a limited failure population or zero inflation is a mixture
of those models, the posterior's points each holding its own offset, share
that ever fails and share dead on arrival: every function is then the
fit's own, and a draw inverts the point's model, as for the plain ones.

A diagram, a ``NonRepairable`` and a ``RepairableRBD``'s component take a
Bayesian fit as it is, and wrap it in this (``as_life``): one fit object is
one predictive object wherever it is given, so blocks given the same fit
hold the same model, as blocks given one ordinary fit do.

The parameter-uncertainty methods draw a Bayesian block's credible band
from its posterior (``_posterior_models``): each draw a posterior point
picked by its probability (from the exact family of a conjugate posterior),
and the block in that draw the plain distribution at it.
"""

import weakref
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ._model_utils import is_bayesian_fit
from ._sampling import RowSampler

#: The most values of the points' functions worked out at once (16 MB):
#: a query is taken in rows of at most this many over the points.
_BLOCK = 1 << 21
#: The largest double below 1, a draw's share of a point's life at most.
_BELOW_ONE = float(np.nextafter(1.0, 0.0))
#: The extras a surpyval parametric model may have after its
#: distribution's parameters (an offset, the share that ever fails, the
#: share dead on arrival), each with the value that means it has none.
_EXTRAS = (("gamma", 0.0), ("lfp_p", 1.0), ("f0", 0.0))


#: The predictive of each Bayesian fit given as a life, by the fit's id
#: (the entry goes with the last model holding it).
_PREDICTIVES: "weakref.WeakValueDictionary[int, PosteriorPredictive]" = (
    weakref.WeakValueDictionary()
)


def as_life(model: Any) -> Any:
    """``model``, but a surpyval Bayesian fit as its posterior predictive
    (a ``PosteriorPredictive``): the same one for the same fit object, so
    that blocks given one fit share one model (and, under parameter
    uncertainty, one draw of its posterior), as blocks given one ordinary
    fit do."""
    if not is_bayesian_fit(model):
        return model
    life = _PREDICTIVES.get(id(model))
    if life is None or life.fit is not model:
        life = PosteriorPredictive(model)
        _PREDICTIVES[id(model)] = life
    return life


class PosteriorPredictive:
    """A surpyval Bayesian fit's posterior predictive, as a life.

    A Bayesian fit (``surpyval.Weibull.fit_bayes(...)``, a
    ``BayesianParametric``) holds the posterior of its distribution's
    parameters. As a life it is the life of a new unit with the
    uncertainty in the parameters averaged in: the survival function at
    each posterior point, averaged with the points' probabilities,
    ``S(t) = sum_i w_i S(t | theta_i)``. It is a little more spread than the
    distribution at the best single estimate. A block of a
    [`NonRepairableRBD`][repyability.NonRepairableRBD], a
    [`NonRepairable`][repyability.NonRepairable] and a component of a
    [`RepairableRBD`][repyability.RepairableRBD] take the fit itself and
    wrap it in this; it can be built by hand too.

    Each function is the predictive's, to its own precision: ``sf`` and
    ``df`` are the points' averaged, ``hf`` follows from them, ``ff``,
    ``Hf`` and ``qf`` are the fit's own (surpyval's, which keep their
    digits in both tails), and ``mean`` is the points' means averaged
    (``inf`` if any is). Where the fit has the predictive
    in closed form (an Exponential with a Gamma posterior on its rate,
    whose predictive is a Lomax; a Normal mean with the spread known),
    that is used, as surpyval does. A life is drawn from one uniform: it
    picks a posterior point by the points' probabilities, and the share of
    that point's probability the uniform falls at is the point's life's
    quantile (the closed forms are inverted instead).

    A diagram's value is worked out with the predictive (its point value).
    How sure the fit is of that value, its credible band, is what the
    parameter-uncertainty methods (``sf_uncertainty`` and the like) carry
    to the system: each draw takes a point of the posterior by its
    probability (from the exact family of a conjugate posterior) and the
    plain distribution at it (see the guide's "A Bayesian fit as a
    life").

    A fit with an offset, a limited failure population or zero inflation
    (``fit_bayes(offset=True)``, ``lfp=True``, ``zi=True``, or draws with
    ``gamma``, ``lfp_p`` or ``f0`` columns) is the mixture of those
    models: each posterior point has its own offset and shares, its
    functions are the fit's own (surpyval's predictive), a draw inverts
    the point's model (0 for a unit dead on arrival, ``inf`` for one that
    never fails), and the credible band draws each point with them. A fit
    of a discrete distribution is refused: it is of a count, not a life.

    Parameters
    ----------
    fit : surpyval BayesianParametric
        The Bayesian fit (from ``fit_bayes`` or
        ``BayesianParametric.from_draws``).

    Attributes
    ----------
    fit : surpyval BayesianParametric
        The fit.
    points : numpy.ndarray
        The posterior's points, one row each, a column for each of the
        distribution's parameters (in its ``parameter_names`` order).
    extras : dict
        Each point's offset (``gamma``), share that ever fails (``lfp_p``)
        and share dead on arrival (``f0``), as arrays over the points, for
        those the fit has (empty for a plain fit).
    weights : numpy.ndarray
        Their probabilities, summing to 1.

    Raises
    ------
    TypeError
        If ``fit`` is not a surpyval Bayesian fit.
    ValueError
        If its distribution is discrete.

    Examples
    --------
    An exponential life whose rate has a Gamma(5, 1250) posterior (from
    another sampler's draws, or a conjugate fit) has the Lomax predictive
    ``S(t) = (1250 / (1250 + t)) ** 5``:

    >>> import numpy as np
    >>> import surpyval as surv
    >>> from repyability import NonRepairableRBD, PosteriorPredictive
    >>> rate = np.random.default_rng(1).gamma(5.0, 1 / 1250.0, 4000)
    >>> fit = surv.bayes.BayesianParametric.from_draws(
    ...     surv.Exponential, rate[:, None])
    >>> pump = PosteriorPredictive(fit)
    >>> bool(np.isclose(pump.sf(100.0), np.mean(np.exp(-rate * 100.0))))
    True
    >>> rbd = NonRepairableRBD([("s", "pump"), ("pump", "t")], {"pump": fit})
    >>> bool(rbd.sf(100.0) == pump.sf(100.0))
    True
    """

    def __init__(self, fit: Any):
        if isinstance(fit, PosteriorPredictive):
            fit = fit.fit
        if not is_bayesian_fit(fit):
            raise TypeError(
                "A PosteriorPredictive takes a surpyval Bayesian fit "
                "(Distribution.fit_bayes(...), or "
                "BayesianParametric.from_draws(...)); got a "
                f"{type(fit).__name__}."
            )
        family = fit.dist
        if getattr(family, "discrete", False):
            raise ValueError(
                f"A Bayesian fit of the {family.name} distribution is of a "
                "count, not a life, so it is no lifetime: give a fit of a "
                "distribution of time."
            )
        posterior = fit.posterior
        columns = [
            posterior.parameter_names.index(name)
            for name in family.parameter_names
        ]
        self.fit: Any = fit
        self.points = np.ascontiguousarray(posterior.points[:, columns])
        # Each point's offset and shares, where the fit has them (a column
        # of the posterior, or fixed at a value the fit was given).
        names = list(posterior.parameter_names)
        fixed = dict(getattr(fit, "fixed", None) or {})
        self.extras: dict = {}
        for name, none in _EXTRAS:
            if name in names:
                values = posterior.points[:, names.index(name)]
            elif float(fixed.get(name, none)) != none:
                values = np.full(len(self.points), float(fixed[name]))
            else:
                continue
            self.extras[name] = np.ascontiguousarray(values, dtype=float)
        self.weights = np.asarray(posterior.weights, dtype=float)
        self._family = family
        self._columns = [
            self.points[:, [j]].T for j in range(self.points.shape[1])
        ]
        ends = np.cumsum(self.weights)
        ends[-1] = 1.0
        self._ends = ends
        self._starts = np.r_[0.0, ends[:-1]]
        exact = posterior.exact or {}
        self._lomax = None
        self._closed = None
        if exact.get("family") == "gamma":
            a, b = exact["params"]
            self._lomax = (float(a), float(b))
        elif exact.get("family") == "normal":
            m, s = exact["params"]
            spread = float(np.hypot(fit.fixed["sigma"], s))
            self._closed = family.from_params([float(m), spread])

    # -- The points' functions ---------------------------------------------

    def _values(self, name: str, x: np.ndarray) -> np.ndarray:
        """The function ``name`` of every point at the 1-d ``x``: one row
        a value of ``x``, one column a point."""
        with np.errstate(all="ignore"):
            out = getattr(self._family, name)(x[:, None], *self._columns)
        return np.asarray(out, dtype=float)

    def _rows(self) -> int:
        """How many values of a query are worked out at once."""
        return max(1, _BLOCK // len(self.weights))

    def _mixed(self, name: str, x: np.ndarray) -> np.ndarray:
        """The points' function ``name`` averaged with their probabilities,
        at the 1-d ``x``, in rows of ``_rows()`` values at a time."""
        step = self._rows()
        parts = [
            np.sum(self._values(name, x[i : i + step]) * self.weights, axis=1)
            for i in range(0, len(x), step)
        ]
        return np.concatenate(parts) if parts else np.empty(0)

    def _lomax_values(self, name: str, x: np.ndarray) -> np.ndarray:
        """The Lomax predictive of an Exponential with a Gamma(a, b)
        posterior on its rate, by its cumulative hazard ``H(x) = a log(1 +
        x / b)`` (which keeps both tails)."""
        assert self._lomax is not None
        a, b = self._lomax
        after = np.maximum(x, 0.0)
        with np.errstate(all="ignore"):
            H = a * np.log1p(after / b)
            h = np.where(x >= 0.0, a / (b + after), 0.0)
            return {
                "sf": lambda: np.exp(-H),
                "df": lambda: h * np.exp(-H),
                "hf": lambda: h,
            }[name]()

    def _function(self, name: str, x: ArrayLike) -> Any:
        """The predictive's function ``name`` (``sf``, ``df`` or ``hf``)
        at ``x``, with its shape."""
        t = np.asarray(x, dtype=float)
        flat = t.ravel()
        if self._lomax is not None:
            out = self._lomax_values(name, flat)
        elif self._closed is not None:
            out = np.asarray(getattr(self._closed, name)(flat), dtype=float)
        elif self.extras:
            # The points' offsets and shares: the fit's own predictive.
            out = self._fits(name, flat)
        elif name == "sf":
            # (a probability, which the weights' rounding can take a
            # last bit past 1)
            out = np.clip(self._mixed(name, flat), 0.0, 1.0)
        elif name == "df":
            out = self._mixed(name, flat)
        else:
            with np.errstate(all="ignore"):
                out = self._mixed("df", flat) / self._mixed("sf", flat)
        return np.asarray(out, dtype=float).reshape(t.shape)[()]

    def _fits(self, name: str, x: ArrayLike) -> Any:
        """The fit's own function ``name`` at ``x`` (surpyval's
        ``BayesianParametric``'s), as floats with the shape of ``x``."""
        t = np.asarray(x, dtype=float)
        out = np.asarray(getattr(self.fit, name)(t.ravel()), dtype=float)
        return out.reshape(t.shape)[()]

    # -- The life ------------------------------------------------------------

    def sf(self, x: ArrayLike) -> Any:
        """The probability that a new unit survives to ``x``: the points'
        survival functions averaged, ``sum_i w_i S(x | theta_i)``.

        Parameters
        ----------
        x : array_like
            The time(s).

        Returns
        -------
        float or numpy.ndarray
            With the shape of ``x``.
        """
        return self._function("sf", x)

    def ff(self, x: ArrayLike) -> Any:
        """The probability that a new unit fails by ``x``: the fit's own
        ``ff``, the points' ``ff`` averaged (not ``1 - sf``, so it keeps
        its precision when small).

        Parameters
        ----------
        x : array_like
            The time(s).
        """
        return self._fits("ff", x)

    def df(self, x: ArrayLike) -> Any:
        """The density of a new unit's life: the points' densities
        averaged.

        Parameters
        ----------
        x : array_like
            The time(s).
        """
        return self._function("df", x)

    def hf(self, x: ArrayLike) -> Any:
        """The hazard of the predictive, ``df / sf`` (not the points'
        hazards averaged).

        Parameters
        ----------
        x : array_like
            The time(s).
        """
        return self._function("hf", x)

    def Hf(self, x: ArrayLike) -> Any:
        """The cumulative hazard of the predictive, ``-log sf``: the fit's
        own, which keeps its precision in both tails.

        Parameters
        ----------
        x : array_like
            The time(s).
        """
        return self._fits("Hf", x)

    def cs(self, x: ArrayLike, given: ArrayLike) -> Any:
        """The probability that a unit that has survived to ``given``
        survives a further ``x``: ``sf(given + x) / sf(given)``.

        Parameters
        ----------
        x : array_like
            The further time(s).
        given : array_like
            The age(s) survived to.
        """
        x = np.asarray(x, dtype=float)
        given = np.asarray(given, dtype=float)
        with np.errstate(all="ignore"):
            out = np.asarray(self.sf(given + x)) / np.asarray(self.sf(given))
        return np.asarray(out, dtype=float)[()]

    def qf(self, p: ArrayLike) -> Any:
        """The age by which a fraction ``p`` of new units have failed: the
        predictive's quantile, the fit's own (surpyval's, which converges
        in the upper tail).

        Parameters
        ----------
        p : array_like
            The probabilities of failing.
        """
        if self.extras:
            return self._extras_qf(p)
        return self._fits("qf", p)

    def _extras_qf(self, p: ArrayLike) -> Any:
        """The quantile of a fit with an offset or shares, by inverting
        its ``ff``: 0 up to the share dead on arrival (``ff(0)``), ``inf``
        from the share that ever fails, and between them the root of
        ``ff(x) = p`` (surpyval 0.27's own fails to bracket some of these
        roots)."""
        from scipy.optimize import brentq

        q = np.asarray(p, dtype=float)
        flat = q.ravel()
        out = np.full(flat.shape, np.nan)
        dead = float(self._fits("ff", np.array([0.0]))[0])
        ever = 1.0 - self.never_fails
        out[flat <= dead] = 0.0
        out[flat >= ever] = np.inf
        gamma = self.extras.get("gamma")
        low = float(np.min(gamma)) if gamma is not None else 0.0
        middle = np.flatnonzero((flat > dead) & (flat < ever))
        if middle.size:

            def ff(x: float) -> float:
                return float(self._fits("ff", np.array([x]))[0])

            high = max(low, 1.0) * 2.0
            for i in middle[np.argsort(flat[middle])]:
                while ff(high) < flat[i]:
                    high *= 2.0
                out[i] = brentq(
                    lambda x: ff(x) - flat[i],
                    low,
                    high,
                    xtol=1e-300,
                    rtol=4 * np.finfo(float).eps,
                )
        return out.reshape(q.shape)[()]

    def mean(self) -> float:
        """The mean life of a new unit: the points' means averaged with
        their probabilities, ``inf`` if any point's is (surpyval's).

        Returns
        -------
        float
            The mean life.
        """
        if self.never_fails > 0.0:
            # (some units never fail)
            return float("inf")
        return float(self.fit.mean())

    def _draw(self, u: ArrayLike) -> np.ndarray:
        """The lives the uniforms ``u`` give, one uniform a life: the
        posterior point whose share of ``[0, 1)`` (its probability, the
        points one after another) ``u`` falls in, and that point's quantile
        at where in its share ``u`` falls. Lives so drawn have the
        predictive distribution; a closed form is inverted instead."""
        u = np.asarray(u, dtype=float)
        if self._lomax is not None or self._closed is not None:
            return np.asarray(self.qf(u), dtype=float)
        flat = u.ravel()
        last = len(self.weights) - 1
        point = np.minimum(np.searchsorted(self._ends, flat, "right"), last)
        weight = self.weights[point]
        with np.errstate(all="ignore"):
            share = (flat - self._starts[point]) / weight
        share = np.clip(np.where(weight > 0.0, share, 0.5), 0.0, _BELOW_ONE)
        params = [self.points[point, j] for j in range(self.points.shape[1])]
        if not self.extras:
            with np.errstate(all="ignore"):
                lives = self._family.qf(share, *params)
            return np.asarray(lives, dtype=float).reshape(u.shape)
        # The point's model, F = f0 + (p - f0) F0(x - gamma), inverted:
        # dead on arrival below f0, never failing from p, and otherwise
        # the base distribution's quantile past the offset.
        gamma, p, f0 = (
            self.extras.get(name, np.full(len(self.weights), none))[point]
            for name, none in _EXTRAS
        )
        with np.errstate(all="ignore"):
            base = np.clip((share - f0) / (p - f0), 0.0, _BELOW_ONE)
            lives = np.where(
                share < f0,
                0.0,
                np.where(
                    share >= p, np.inf, gamma + self._family.qf(base, *params)
                ),
            )
        return np.asarray(lives, dtype=float).reshape(u.shape)

    def random(self, size: int) -> np.ndarray:
        """Draw ``size`` lives of new units, from uniforms from numpy's
        global RNG (see the class's description): seed it
        (``np.random.seed``) to reproduce them; an RBD's seeded methods do.

        Parameters
        ----------
        size : int
            The number of draws.

        Returns
        -------
        numpy.ndarray
            The ``size`` lives, shape ``(size,)``.
        """
        return self._draw(np.random.uniform(size=size))

    def _row_sampler(self) -> RowSampler:
        """``random(1)`` as a :class:`~._sampling.RowSampler` (one uniform
        a draw), so an RBD with this node batches its draws."""
        return RowSampler(
            1, lambda u: self._draw(np.ascontiguousarray(u[:, 0]))
        )

    def _posterior_models(self, u: ArrayLike) -> list:
        """The plain distributions at the posterior points the uniforms
        ``u`` pick, one a draw: the point whose share of ``[0, 1)`` (its
        probability, the points one after another) ``u`` falls in, so
        points are picked by their probabilities, unequal on a grid, none
        left out; a conjugate posterior's parameter is its exact family's
        quantile at ``u`` instead (surpyval's ``posterior.quantile``),
        the other parameters (a fixed spread) as its points hold them."""
        u = np.asarray(u, dtype=float).ravel()
        posterior = self.fit.posterior
        exact = posterior.exact
        if exact:
            rows = np.repeat(self.points[:1], len(u), axis=0)
            column = self._family.parameter_names.index(exact["name"])
            rows[:, column] = posterior.quantile(exact["name"], u)
        else:
            last = len(self.weights) - 1
            picked = np.minimum(np.searchsorted(self._ends, u, "right"), last)
            rows = self.points[picked]
            if self.extras:
                # Each point with its own offset and shares.
                return [
                    self._family.from_params(
                        list(row),
                        **{
                            name: float(values[i])
                            for name, values in self.extras.items()
                        },
                    )
                    for row, i in zip(rows, picked)
                ]
        return [self._family.from_params(list(row)) for row in rows]

    @property
    def never_fails(self) -> float:
        """The share of new units that never fail: the points' ``1 -
        lfp_p`` averaged with their probabilities (0 for a fit without a
        limited failure population)."""
        p = self.extras.get("lfp_p")
        if p is None:
            return 0.0
        return float(max(0.0, np.sum(self.weights * (1.0 - p))))

    # -- Saving --------------------------------------------------------------

    def to_dict(self) -> dict:
        """The life as a JSON-friendly dict: the fit's own ``to_dict()``,
        its posterior whole. ``from_dict`` rebuilds it, and an RBD's
        ``to_dict``/``to_json`` save it so.

        Returns
        -------
        dict
            ``{"fit": ...}``.
        """
        return {"fit": self.fit.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "PosteriorPredictive":
        """Rebuild the life from :meth:`to_dict`'s dict.

        Parameters
        ----------
        d : dict
            A dict ``to_dict`` made, e.g. after a JSON round trip.

        Returns
        -------
        PosteriorPredictive
            The life, its fit loaded by surpyval.
        """
        import surpyval

        return cls(surpyval.from_dict(d["fit"]))

    def __repr__(self) -> str:
        how = (
            "in closed form"
            if self._lomax is not None or self._closed is not None
            else f"over {len(self.weights)} posterior points"
        )
        return (
            f"PosteriorPredictive({self._family.name} Bayesian fit, "
            f"its posterior predictive {how})"
        )
