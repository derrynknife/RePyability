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
  covariate path (accelerated-failure-time / proportional- / additive-hazards;
  not proportional-odds). This is the load-dependent-aging / digital-twin node
  of issue #37: as-new survival integrates the whole load path, and
  conditioning on ``age`` gives the go-forward reliability from the component's
  current life, since ``R(x | age) = sf_tvc(age + x) / sf_tvc(age)`` is exactly
  surpyval's ``sf_tvc(..., given=age)``.

Either way it is an ordinary univariate node -- it takes part in system
reliability, importance, MTTF and the condition-based (``age``) layer with no
special handling. RePyability *consumes* the fitted model; do the regression
fit in surpyval.
"""

from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from ._sampling import RowSampler


class RegressionNode:
    """An RBD node backed by a fitted surpyval regression model.

    Pairs a fitted regression model (accelerated failure time,
    proportional hazards, proportional odds, ...) with this component's
    covariates, giving an ordinary univariate node. Provide exactly one of:

    - ``covariates``, a fixed operating point ``Z``: the reliability is
      ``R(x) = model.sf(x, Z)``, for any regression family; or
    - ``schedule``, a time-varying covariate path ``Z(t)``: the reliability
      is ``R(x) = model.sf_tvc(x, schedule)``, the survival along that
      path (accelerated-failure-time and proportional- or additive-hazards
      models; not proportional odds).

    ``sf`` and ``ff`` evaluate that curve directly, so the node takes part
    in system reliability, importance measures and the condition-based
    (``age``) methods with no special handling. ``mean`` and ``random``
    work from the curve tabulated on a grid, which needs a proper
    parametric curve: a semiparametric model such as ``surpyval.CoxPH``
    supports ``sf`` but not ``mean`` or ``random``. Do the regression fit
    in surpyval and pass the fitted model in.

    Parameters
    ----------
    model : surpyval regression model
        A fitted regression model, e.g. ``surpyval.WeibullAFT.fit(...)`` or
        ``surpyval.CoxPH.fit(...)``. Fixed covariates use its ``sf(x, Z)``; a
        schedule uses its ``sf_tvc(x, schedule)`` (needs a surpyval that
        provides it, and a family other than proportional-odds).
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
        ``covariates`` has the wrong width, or in schedule mode the model
        has no ``sf_tvc`` or is proportional odds.

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
    >>> round(float(node.sf(100.0)[0]), 4)  # exp(-100 / 100)
    0.3679
    >>> round(node.mean(), 1)
    100.0

    One run at load 1 for 50 time units and at load 2 after that:

    >>> schedule = surv.StepSchedule.from_changepoints(
    ...     [0, 50], [[1.0], [2.0]]
    ... )
    >>> ramped = RegressionNode(model, schedule=schedule)
    >>> round(float(ramped.sf(100.0)[0]), 4)  # exp(-50 / 100 - 50 / 25)
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
        # Probe the survival interface so a misuse fails clearly at
        # construction (wrong covariate width, an sf_tvc-less surpyval, or a
        # proportional-odds model in schedule mode).
        try:
            probe = self._sf_at(np.array([1.0]))
            if not np.all(np.isfinite(probe)):
                raise ValueError("survival returned non-finite values")
        except Exception as e:
            raise ValueError(
                "RegressionNode requires a fitted surpyval regression model. "
                "In fixed-covariate mode its sf(x, Z) must accept a covariate "
                "matrix of the fitted width; in schedule mode the model must "
                "support sf_tvc(x, schedule) (accelerated-failure-time / "
                "proportional- or additive-hazards on a recent surpyval, not "
                f"proportional-odds). Probing survival failed: "
                f"{type(e).__name__}: {e}."
            ) from e
        # Cached (t, sf(t)) grid for mean()/random() (built lazily).
        self._grid: Any = None

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
        numpy.ndarray
            The probability of surviving beyond each ``x``. Always an
            array: a scalar ``x`` gives a 1-element array.
        """
        return self._sf_at(np.atleast_1d(np.asarray(x, dtype=float)))

    def ff(self, x: ArrayLike) -> np.ndarray:
        """Unreliability: ``1 - sf(x)``.

        Parameters
        ----------
        x : array_like
            The time(s) at which to evaluate.

        Returns
        -------
        numpy.ndarray
            The probability of failing by each ``x``; always an array, as
            for ``sf``.
        """
        return 1.0 - self.sf(x)

    def _survival_grid(self):
        """A cached ``(t, sf(t))`` grid spanning the bulk of the lifetime.

        surpyval's regression models expose no working ``random``, so ``mean``
        and ``random`` are obtained from the survival curve directly (a numeric
        integral, and inverse-transform sampling). This needs a *proper*
        lifetime curve (starting at ~1 and decaying to 0); a semiparametric
        baseline (e.g. surpyval's Cox) is defined only on the observed range
        and has no proper tail, so MTTF/simulation is undefined there and is
        reported as a clear error rather than a wrong number.
        """
        if self._grid is None:
            if float(self._sf_at(np.array([1e-9]))[0]) <= 0.99:
                raise ValueError(
                    "mean()/random() need a proper parametric survival curve "
                    "(sf(0+) ~ 1, decaying to 0), but this model's survival "
                    "is improper -- e.g. a semiparametric Cox baseline, "
                    "defined only on the observed range. Its MTTF is "
                    "undefined; use the sf-based reliability / remaining-life "
                    "methods instead."
                )
            hi = 1.0
            while self._sf_at(np.array([hi]))[0] > 1e-4:
                hi *= 2.0
                if hi > 1e15:
                    raise ValueError(
                        "mean()/random(): the survival curve does not decay "
                        "to 0 (no finite MTTF). Use the sf-based methods."
                    )
            t = np.linspace(0.0, hi, 4096)
            self._grid = (t, self._sf_at(t))
        return self._grid

    def mean(self) -> float:
        """Mean time to failure at the stored covariates / along the schedule.

        For a non-negative lifetime ``E[T] = integral of R(t)``, integrated
        numerically over the survival curve: the trapezoidal rule on 4096
        points from 0 to the first ``t`` in 1, 2, 4, 8, ... at which
        ``R(t) <= 1e-4``. The tail beyond is left out, so the result is
        slightly low (by that tail's integral). The grid is built on first
        use and cached.

        Returns
        -------
        float
            The mean time to failure.

        Raises
        ------
        ValueError
            If the survival curve is improper (``R(1e-9) <= 0.99``, as with
            a semiparametric Cox baseline) or does not fall to 1e-4 by
            ``t = 1e15``.
        """
        t, s = self._survival_grid()
        return float(np.trapezoid(s, t))

    def random(self, size: int) -> np.ndarray:
        """Draw ``size`` failure times at the stored covariates / schedule.

        Inverse-transform sampling on the survival curve tabulated for
        ``mean``: each uniform ``u`` maps to the time at which ``R = u``
        (linear interpolation), so draws never exceed the grid's end,
        where ``R <= 1e-4``. There is no ``seed`` argument: this uses
        numpy's global RNG, so seed it (``np.random.seed``) or wrap the
        call in ``repyability.utils.wrappers.numpy_seed`` to reproduce the
        draws. An RBD's seeded ``random`` and ``mean`` do this for you.

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
            If the survival curve is improper or does not decay, as for
            ``mean``.
        """
        t, s = self._survival_grid()
        u = np.random.uniform(size=size)
        # s decreases in t; np.interp needs an increasing sample-point array.
        return np.interp(u, s[::-1], t[::-1])

    def _row_sampler(self) -> RowSampler:
        """``random(1)`` as a :class:`~._sampling.RowSampler` (one uniform
        per draw), so an RBD with this node batches its draws."""

        def draw(u):
            t, s = self._survival_grid()
            return np.interp(np.ascontiguousarray(u[:, 0]), s[::-1], t[::-1])

        return RowSampler(1, draw)

    # -- Serialisation ----------------------------------------------------

    @staticmethod
    def _schedule_to_dict(schedule: Any) -> dict:
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
        >>> round(float(restored.sf(100.0)[0]), 4)
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
