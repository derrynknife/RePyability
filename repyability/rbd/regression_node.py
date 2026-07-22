"""Regression node: an RBD node whose reliability depends on stored covariates.

A :class:`RegressionNode` wraps a fitted surpyval **regression** model (an
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


class RegressionNode:
    """An RBD node backed by a fitted surpyval regression model.

    Provide exactly one of ``covariates`` (a fixed operating point) or
    ``schedule`` (a time-varying covariate path).

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

    Examples
    --------
    >>> import numpy as np
    >>> import surpyval as surv
    >>> from repyability.rbd.regression_node import RegressionNode
    >>> rng = np.random.default_rng(0)
    >>> Z = rng.normal(size=(300, 1))
    >>> x = rng.weibull(2.0, size=300) * 100 + 1e-3
    >>> model = surv.WeibullAFT.fit(x, Z=Z)
    >>> node = RegressionNode(model, covariates=[0.5])
    >>> bool(0.0 < node.sf(np.array([50.0]))[0] < 1.0)
    True
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
        """Reliability at the stored covariates (``model.sf(x, Z)``) or along
        the covariate schedule (``model.sf_tvc(x, schedule)``)."""
        return self._sf_at(np.atleast_1d(np.asarray(x, dtype=float)))

    def ff(self, x: ArrayLike) -> np.ndarray:
        """Unreliability: ``1 - sf(x)``."""
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
        numerically over the survival curve.
        """
        t, s = self._survival_grid()
        return float(np.trapezoid(s, t))

    def random(self, size: int) -> np.ndarray:
        """Draw ``size`` failure times at the stored covariates / schedule.

        Inverse-transform sampling on the survival curve. Uses numpy's global
        RNG, so wrap the call in
        :func:`~repyability.utils.wrappers.numpy_seed` to reproduce.
        """
        t, s = self._survival_grid()
        u = np.random.uniform(size=size)
        # s decreases in t; np.interp needs an increasing sample-point array.
        return np.interp(u, s[::-1], t[::-1])

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
        """Serialise to a JSON-friendly dict (the fitted model + covariates or
        schedule). See :meth:`from_dict`."""
        out: dict = {"model": self.model.to_dict()}
        if self.schedule is not None:
            out["schedule"] = self._schedule_to_dict(self.schedule)
        else:
            assert self.covariates is not None
            out["covariates"] = [float(v) for v in self.covariates]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "RegressionNode":
        """Reconstruct from :meth:`to_dict` (the fitted model round-trips
        through ``surpyval.from_dict``)."""
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
