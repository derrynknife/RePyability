"""Regression node: an RBD node whose reliability depends on stored covariates.

A ``RegressionNode`` wraps a fitted surpyval **regression** model (an
accelerated-failure-time, proportional-hazards, proportional-odds, ... model)
together with the covariate history of *this* component in the system, and
exposes its survival as an ordinary univariate node. The covariates can be:

* a **fixed vector** ``Z`` (constant operating conditions) -- reliability is
  ``R(x) = model.sf(x, Z)``; or
* a **time-varying schedule** ``Z(t)`` (a surpyval ``StepSchedule``: the load
  the component runs under changes over its life) -- reliability is
  ``R(x) = model.sf_tvc(x, schedule)``, the exact survival along that
  covariate path (accelerated-failure-time, proportional- and
  additive-hazards, and proportional-odds models). This is the
  load-dependent-aging / digital-twin node of issue #37: as-new survival
  integrates the whole load path, and conditioning on ``age`` gives the
  go-forward reliability from the component's current life, since
  ``R(x | age) = sf_tvc(age + x) / sf_tvc(age)`` is exactly surpyval's
  ``sf_tvc(..., given=age)``.

Either way it is an ordinary univariate node -- it takes part in system
reliability, importance, MTTF and the condition-based (``age``) layer with no
special handling. RePyability *consumes* the fitted model; do the regression
fit in surpyval.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._sampling import RowSampler


def _is_semiparametric(model) -> bool:
    """Whether ``model`` is a surpyval semiparametric regression model (a
    Cox model): its baseline is an estimate on the observed range only,
    with no tail beyond it. (Recognised by its type, not by its survival
    curve.)"""
    from surpyval.univariate.regression import (
        semi_parametric_regression_model as semiparametric,
    )

    return isinstance(model, semiparametric.SemiParametricRegressionModel)


def _inverse_hazard(hazard, target: np.ndarray) -> np.ndarray:
    """The times ``t`` with ``hazard(t) = target``, for a cumulative hazard
    rising from 0: bracketed by doubling, then bisected until each bracket
    is as narrow as floating point allows. ``inf`` where ``target`` is
    (``u = 1``) or the hazard never reaches it; 0 where ``target`` is 0."""
    target = np.asarray(target, dtype=float)
    out = np.zeros(target.shape)
    live = np.flatnonzero(target > 0.0)
    out[target == np.inf] = np.inf
    live = live[np.isfinite(target[live])]
    if not live.size:
        return out
    goal = target[live]
    lo = np.zeros(live.size)
    hi = np.ones(live.size)
    short = hazard(hi) < goal
    while short.any():
        lo[short] = hi[short]
        hi[short] *= 2.0
        too_far = hi > 1e300
        if too_far.any():
            hi[too_far & short] = np.inf
            short &= ~too_far
        rising = np.flatnonzero(short)
        if rising.size:
            short[rising] = hazard(hi[rising]) < goal[rising]
    open_ = np.flatnonzero(np.isfinite(hi))
    while open_.size:
        a, b = lo[open_], hi[open_]
        mid = 0.5 * (a + b)
        settled = (mid <= a) | (mid >= b)
        below = hazard(mid) < goal[open_]
        lo[open_] = np.where(below & ~settled, mid, a)
        hi[open_] = np.where(~below & ~settled, mid, b)
        open_ = open_[~settled]
    out[live] = hi
    return out


class RegressionNode:
    """An RBD node backed by a fitted surpyval regression model.

    Pairs a fitted regression model (accelerated failure time,
    proportional hazards, proportional odds, ...) with this component's
    covariates, giving an ordinary univariate node. Provide exactly one of:

    - ``covariates``, a fixed operating point ``Z``: the reliability is
      ``R(x) = model.sf(x, Z)``, for any regression family; or
    - ``schedule``, a time-varying covariate path ``Z(t)``: the reliability
      is ``R(x) = model.sf_tvc(x, schedule)``, the survival along that
      path (accelerated-failure-time, proportional- and additive-hazards,
      and proportional-odds models).

    ``sf`` and ``ff`` evaluate that curve directly, so the node takes part
    in system reliability, importance measures and the condition-based
    (``age``) methods with no special handling. ``mean`` integrates the
    curve to a relative 1e-10, and ``random`` inverts it exactly (the
    model's quantile at fixed covariates; the cumulative hazard along a
    schedule), which needs a proper parametric curve: a semiparametric
    model such as ``surpyval.CoxPH`` supports ``sf`` but not ``mean`` or
    ``random``. Do the regression fit
    in surpyval and pass the fitted model in.

    Parameters
    ----------
    model : surpyval regression model
        A fitted regression model, e.g. ``surpyval.WeibullAFT.fit(...)`` or
        ``surpyval.CoxPH.fit(...)``. Fixed covariates use its ``sf(x, Z)``; a
        schedule uses its ``sf_tvc(x, schedule)``.
    covariates : array_like, optional
        The component's fixed covariate vector ``Z`` (its operating
        conditions), matching the covariates the model was fitted with.
    schedule : surpyval StepSchedule, optional
        A piecewise-constant covariate path ``Z(t)`` -- the load the component
        runs under over its life (build with
        ``surpyval.StepSchedule.from_changepoints`` / ``from_intervals``).

    Raises
    ------
    ValueError
        If not exactly one of ``covariates`` and ``schedule`` is given, or
        if a trial evaluation of the survival at ``x = 1`` fails or is not
        finite: e.g. ``model`` is not a fitted regression model,
        ``covariates`` has the wrong width, or in schedule mode the model's
        ``sf_tvc`` cannot evaluate it.

    Examples
    --------
    Fit an Exponential AFT model with load as its covariate in surpyval.
    With these data the life at load 1 is 100 and at load 2 is 25:

    >>> import numpy as np
    >>> import surpyval as surv
    >>> from repyability import RegressionNode
    >>> x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
    >>> load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
    >>> model = surv.ExponentialAFT.fit(x, Z=load)

    A component always run at load 1:

    >>> node = RegressionNode(model, covariates=[1.0])
    >>> round(float(node.sf(100.0)), 4)  # exp(-100 / 100)
    0.3679
    >>> round(node.mean(), 1)
    100.0

    One run at load 1 for 50 time units and at load 2 after that:

    >>> schedule = surv.StepSchedule.from_changepoints(
    ...     [0, 50], [[1.0], [2.0]]
    ... )
    >>> ramped = RegressionNode(model, schedule=schedule)
    >>> round(float(ramped.sf(100.0)), 4)  # exp(-50 / 100 - 50 / 25)
    0.0821
    """

    def __init__(
        self,
        model: Any,
        covariates: Optional[ArrayLike] = None,
        schedule: Any = None,
    ):
        if (covariates is None) == (schedule is None):
            raise ValueError(
                "RegressionNode requires exactly one of `covariates` (a fixed "
                "covariate vector) or `schedule` (a time-varying "
                "StepSchedule)."
            )
        self.model = model
        self.covariates = (
            None
            if covariates is None
            else np.atleast_1d(np.asarray(covariates, dtype=float))
        )
        self.schedule = schedule
        # A covariate vector of another width than the model was fitted
        # with: say so directly.
        fitted = getattr(model, "phi_param_map", None)
        if self.covariates is not None and isinstance(fitted, dict):
            if len(self.covariates) != len(fitted):
                raise ValueError(
                    f"The model was fitted with {len(fitted)} covariate(s); "
                    f"covariates has {len(self.covariates)}."
                )
        # Probe the survival interface so a misuse fails clearly at
        # construction (wrong covariate width, or a surpyval whose sf_tvc
        # cannot evaluate the model's family in schedule mode).
        try:
            probe = self._sf_at(np.array([1.0]))
            if not np.all(np.isfinite(probe)):
                raise ValueError("survival returned non-finite values")
        except Exception as e:
            raise ValueError(
                "RegressionNode requires a fitted surpyval regression model. "
                "In fixed-covariate mode its sf(x, Z) must accept a covariate "
                "matrix of the fitted width; in schedule mode the model must "
                "support sf_tvc(x, schedule) (accelerated-failure-time, "
                "proportional- and additive-hazards, and proportional-odds "
                "models). Probing "
                f"survival failed: {type(e).__name__}: {e}."
            ) from e

    def _Z(self, n: int) -> np.ndarray:
        """Fixed covariate vector broadcast to ``n`` rows for ``sf(x, Z)``."""
        assert self.covariates is not None  # fixed-covariate mode only
        return np.repeat(self.covariates[np.newaxis, :], n, axis=0)

    def _sf_at(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if self.schedule is not None:
            return np.asarray(self.model.sf_tvc(x, self.schedule), dtype=float)
        return np.asarray(self.model.sf(x, self._Z(len(x))), dtype=float)

    # -- Node reliability interface ---------------------------------------

    def sf(self, x: ArrayLike) -> np.ndarray:
        """Reliability at the stored covariates or along the schedule.

        ``model.sf(x, Z)`` at the fixed covariates ``Z``, or
        ``model.sf_tvc(x, schedule)`` along the covariate schedule. An RBD
        calls this for the node's reliability.

        Parameters
        ----------
        x : array_like
            The time(s) at which to evaluate.

        Returns
        -------
        float or numpy.ndarray
            The probability of surviving beyond each ``x``: a float for a
            scalar ``x``, as a surpyval model gives it, and an array for an
            array.
        """
        out = self._sf_at(np.atleast_1d(np.asarray(x, dtype=float)))
        return out[0] if np.ndim(x) == 0 else out

    def ff(self, x: ArrayLike) -> np.ndarray:
        """Unreliability: ``1 - sf(x)``.

        ``model.ff(x, Z)`` at the fixed covariates, or
        ``-expm1(-model.Hf_tvc(x, schedule))`` along the schedule: worked
        out in its own right, so that a small one keeps its precision (for
        a proportional-odds model along a schedule too, from surpyval 0.22:
        surpyval #528).

        Parameters
        ----------
        x : array_like
            The time(s) at which to evaluate.

        Returns
        -------
        float or numpy.ndarray
            The probability of failing by each ``x``: a float for a scalar
            ``x``, as for ``sf``.
        """
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        if self.schedule is not None:
            H = np.asarray(self.model.Hf_tvc(x, self.schedule), dtype=float)
            out = -np.expm1(-H)
        else:
            out = np.asarray(self.model.ff(x, self._Z(len(x))), dtype=float)
        return out[0] if scalar else out

    def _require_proper(self) -> None:
        """Refuse a lifetime with no proper survival curve (starting at ~1):
        a semiparametric baseline (e.g. surpyval's Cox) is defined only on
        the observed range, with no tail, so its mean and draws are
        undefined there, a clear error rather than a wrong number."""
        if _is_semiparametric(self.model) or (
            float(self._sf_at(np.array([1e-9]))[0]) <= 0.99
        ):
            raise ValueError(
                "mean()/random() need a proper parametric survival curve "
                "(sf(0+) ~ 1, decaying to 0), but this model's survival "
                "is improper -- e.g. a semiparametric Cox baseline, "
                "defined only on the observed range. Its MTTF is "
                "undefined; use the sf-based reliability / remaining-life "
                "methods instead."
            )

    def _knots(self) -> np.ndarray:
        """Times marking the curve's scale, for its integral: at fixed
        covariates the model's quantiles, along a schedule none (the
        integral finds them from the curve)."""
        if self.schedule is not None:
            return np.empty(0)
        from ._point_availability import _KNOT_PROBABILITIES

        with np.errstate(all="ignore"):
            q = self._quantiles(_KNOT_PROBABILITIES)
        return q[np.isfinite(q)]

    def _kinks(self) -> np.ndarray:
        """Times at which the curve may bend sharply: a step schedule's
        change points, where the covariates jump. (A ``CovariatePath``
        moves between its points linearly, so its hazard only bends, which
        the integral follows without them.)"""
        edges = getattr(self.schedule, "edges", None)
        if edges is None:
            return np.empty(0)
        edges = np.asarray(edges, dtype=float)
        return edges[np.isfinite(edges) & (edges > 0.0)]

    def _quantiles(self, p: np.ndarray) -> np.ndarray:
        """The model's quantiles ``F^-1(p)`` at the fixed covariates."""
        assert self.covariates is not None  # fixed-covariate mode only
        return np.asarray(
            self.model.qf(p, self.covariates[np.newaxis, :]), dtype=float
        )

    def _draw(self, u: np.ndarray) -> np.ndarray:
        """The lifetimes ``F^-1(u)`` of the uniforms ``u``: the model's
        quantiles at fixed covariates, and along a schedule the time at
        which the cumulative hazard reaches ``-log(1 - u)``."""
        self._require_proper()
        u = np.asarray(u, dtype=float)
        if self.schedule is None:
            return self._quantiles(u)
        with np.errstate(divide="ignore"):
            target = -np.log1p(-u)
        return _inverse_hazard(
            lambda t: np.asarray(
                self.model.Hf_tvc(t, self.schedule), dtype=float
            ),
            target,
        )

    def mean(self) -> float:
        """Mean time to failure at the stored covariates / along the schedule.

        For a non-negative lifetime ``E[T] = integral of R(t)``, integrated
        to a relative accuracy of 1e-10 on pieces split at the model's
        quantiles (fixed covariates) or the schedule's change points, as
        an RBD's MTTF is: ``inf`` if some lifetimes never end.

        Returns
        -------
        float
            The mean time to failure.

        Raises
        ------
        ValueError
            If the model is semiparametric (a Cox model: its baseline is
            defined on the observed range only), or if the survival curve
            is improper (``R(1e-9) <= 0.99``).
        """
        from ._mean_lifetime import mean_lifetime

        self._require_proper()
        return float(mean_lifetime(self._sf_at, self._knots(), self._kinks()))

    def random(self, size: int) -> np.ndarray:
        """Draw ``size`` failure times at the stored covariates / schedule.

        Inverse-transform sampling, exact: each uniform ``u`` maps to the
        time at which the distribution reaches ``u``, the model's
        quantile ``qf(u, Z)`` at fixed covariates, and along a schedule the
        time at which the cumulative hazard reaches ``-log(1 - u)`` (found
        by bisection, to the last bits). There is no ``seed`` argument:
        this uses numpy's global RNG, so seed it (``np.random.seed``) or
        wrap the call in ``repyability.utils.wrappers.numpy_seed`` to
        reproduce the draws. An RBD's seeded ``random`` and ``mean`` do
        this for you.

        Parameters
        ----------
        size : int
            The number of failure times to draw.

        Returns
        -------
        numpy.ndarray
            The ``size`` failure times, shape ``(size,)``.

        Raises
        ------
        ValueError
            If the survival curve is improper, as for ``mean``.
        """
        return self._draw(np.random.uniform(size=size))

    def _row_sampler(self) -> RowSampler:
        """``random(1)`` as a :class:`~._sampling.RowSampler` (one uniform
        per draw), so an RBD with this node batches its draws."""
        return RowSampler(
            1, lambda u: self._draw(np.ascontiguousarray(u[:, 0]))
        )

    # -- Serialisation ----------------------------------------------------

    @staticmethod
    def _schedule_to_dict(schedule: Any) -> dict:
        if getattr(schedule, "edges", None) is None:
            raise NotImplementedError(
                f"Serialising a {type(schedule).__name__} is not supported "
                "yet; use a StepSchedule."
            )
        if getattr(schedule, "period", None) is not None:
            raise NotImplementedError(
                "Serialising a cyclic StepSchedule is not supported yet; use "
                "a change-point / interval schedule."
            )
        times = [float(e) for e in schedule.edges if np.isfinite(e)]
        return {"times": times, "values": np.asarray(schedule.Z).tolist()}

    def to_dict(self) -> dict:
        """Serialise the node to a JSON-friendly dict.

        The fitted model is stored with its own ``to_dict()``, plus either
        ``"covariates"`` (a list of floats) or ``"schedule"`` (the
        schedule's finite segment edges as ``"times"`` and its covariate
        rows as ``"values"``). ``from_dict`` rebuilds the node, and an
        RBD's ``to_dict``/``to_json`` use this for a regression node.

        A schedule is rebuilt with ``StepSchedule.from_changepoints``, so
        only one whose last segment is open-ended round-trips (e.g. from
        ``from_changepoints``). One whose last edge is finite (e.g. from
        ``from_intervals``) is written, but ``from_dict`` then raises
        ``ValueError``.

        Returns
        -------
        dict
            ``{"model": ..., "covariates": [...]}`` or
            ``{"model": ..., "schedule": {"times": [...], "values": [...]}}``.

        Raises
        ------
        NotImplementedError
            If the schedule is cyclic (has a ``period``).
        """
        out: dict = {"model": self.model.to_dict()}
        if self.schedule is not None:
            out["schedule"] = self._schedule_to_dict(self.schedule)
        else:
            assert self.covariates is not None
            out["covariates"] = [float(v) for v in self.covariates]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "RegressionNode":
        """Reconstruct a node from the output of ``to_dict``.

        The fitted model round-trips through ``surpyval.from_dict``, and a
        schedule is rebuilt with ``StepSchedule.from_changepoints``.

        Parameters
        ----------
        d : dict
            A dict produced by ``to_dict``, e.g. after a JSON round trip.

        Returns
        -------
        RegressionNode
            The reconstructed node.

        Raises
        ------
        KeyError
            If ``d`` has no ``"model"``, or neither ``"schedule"`` nor
            ``"covariates"``.
        ValueError
            If the schedule cannot be rebuilt, or the node fails the
            constructor's checks.

        Examples
        --------
        >>> import json
        >>> import numpy as np
        >>> import surpyval as surv
        >>> from repyability import RegressionNode
        >>> x = np.array([50.0, 100.0, 150.0, 12.5, 25.0, 37.5])
        >>> load = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [2.0]])
        >>> model = surv.ExponentialAFT.fit(x, Z=load)
        >>> node = RegressionNode(model, covariates=[1.0])
        >>> text = json.dumps(node.to_dict())
        >>> restored = RegressionNode.from_dict(json.loads(text))
        >>> round(float(restored.sf(100.0)), 4)
        0.3679
        """
        import surpyval

        model = surpyval.from_dict(d["model"])
        if "schedule" in d:
            from surpyval.univariate.regression import StepSchedule

            sd = d["schedule"]
            schedule = StepSchedule.from_changepoints(
                sd["times"], sd["values"]
            )
            return cls(model, schedule=schedule)
        return cls(model, covariates=d["covariates"])

    def __repr__(self) -> str:
        if self.schedule is not None:
            which = f"schedule={self.schedule!r}"
        else:
            cov = list(self.covariates)  # type: ignore[arg-type]
            which = f"covariates={cov}"
        return f"RegressionNode({type(self.model).__name__}, {which})"
