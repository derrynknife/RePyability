"""Regression node: an RBD node whose reliability depends on stored covariates.

A :class:`RegressionNode` wraps a fitted surpyval **regression** model (an
accelerated-failure-time, proportional-hazards, proportional-odds, ... model)
together with a fixed **covariate vector** ``Z`` -- the operating conditions of
this component in the system. Its reliability is simply the model's survival at
those covariates::

    R(x) = model.sf(x, Z)

so a regression node is an ordinary univariate node: it takes part in system
reliability, importance, MTTF and the condition-based (``age``) layer exactly
like any other node, with no special handling. Conditioning on age keeps its
usual meaning -- operating time survived -- because at a fixed covariate the
component has a plain univariate lifetime and
``R(x | age) = sf(age + x, Z) / sf(age, Z)`` holds for every regression family.

RePyability *consumes* the fitted model; do the regression fit in surpyval.
See issue #37.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class RegressionNode:
    """An RBD node backed by a fitted surpyval regression model at fixed
    covariates.

    Parameters
    ----------
    model : surpyval regression model
        A fitted regression model whose ``sf(x, Z)`` gives survival at a
        covariate matrix ``Z`` (e.g. ``surpyval.WeibullAFT.fit(...)``,
        ``surpyval.CoxPH.fit(...)``).
    covariates : array_like
        The component's covariate vector ``Z`` (the operating conditions),
        matching the covariates the model was fitted with.

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

    def __init__(self, model: Any, covariates: ArrayLike):
        self.model = model
        self.covariates = np.atleast_1d(np.asarray(covariates, dtype=float))
        # Probe the regression interface so a misuse (a non-regression model,
        # or covariates of the wrong width) fails clearly at construction.
        try:
            probe = self._sf_at(np.array([1.0]))
            if not np.all(np.isfinite(probe)):
                raise ValueError("sf(x, Z) returned non-finite values")
        except Exception as e:
            raise ValueError(
                "RegressionNode requires a fitted surpyval regression model "
                "whose sf(x, Z) accepts a covariate matrix, and covariates "
                f"matching the model's fitted width. Probing sf failed: "
                f"{type(e).__name__}: {e}."
            ) from e
        # Cached (t, sf(t)) grid for mean()/random() (built lazily).
        self._grid: Any = None

    def _Z(self, n: int) -> np.ndarray:
        """The covariate vector broadcast to ``n`` rows for ``sf(x, Z)``."""
        return np.repeat(self.covariates[np.newaxis, :], n, axis=0)

    def _sf_at(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return np.asarray(self.model.sf(x, self._Z(len(x))), dtype=float)

    # -- Node reliability interface ---------------------------------------

    def sf(self, x: ArrayLike) -> np.ndarray:
        """Reliability at the stored covariates: ``model.sf(x, Z)``."""
        return self._sf_at(np.atleast_1d(np.asarray(x, dtype=float)))

    def ff(self, x: ArrayLike) -> np.ndarray:
        """Unreliability at the stored covariates: ``1 - sf(x)``."""
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
        """Mean time to failure at the stored covariates.

        For a non-negative lifetime ``E[T] = integral of R(t)``, integrated
        numerically over the survival curve.
        """
        t, s = self._survival_grid()
        return float(np.trapezoid(s, t))

    def random(self, size: int) -> np.ndarray:
        """Draw ``size`` failure times at the stored covariates.

        Inverse-transform sampling on the survival curve. Uses numpy's global
        RNG, so wrap the call in
        :func:`~repyability.utils.wrappers.numpy_seed` to reproduce.
        """
        t, s = self._survival_grid()
        u = np.random.uniform(size=size)
        # s decreases in t; np.interp needs an increasing sample-point array.
        return np.interp(u, s[::-1], t[::-1])

    # -- Serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict (the fitted model + covariates).
        See :meth:`from_dict`."""
        return {
            "model": self.model.to_dict(),
            "covariates": [float(v) for v in self.covariates],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RegressionNode":
        """Reconstruct from :meth:`to_dict` (the fitted model round-trips
        through ``surpyval.from_dict``)."""
        import surpyval

        return cls(surpyval.from_dict(d["model"]), d["covariates"])

    def __repr__(self) -> str:
        return (
            f"RegressionNode({type(self.model).__name__}, "
            f"covariates={list(self.covariates)})"
        )
