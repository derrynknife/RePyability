"""Load-sharing dynamic node: coupled units that share a load and age faster
as siblings fail (dependent failure).

*n* active units share a total load ``L``. While ``s`` of them survive, each
carries load ``L / s`` and -- through its accelerated-failure-time (AFT)
response -- ages at rate ``phi(L / s)`` on its baseline reliability clock. When
a unit fails the survivors pick up its share, so they age *faster*; the group
works while at least ``k`` units survive. Because the units are coupled (each
unit's aging depends on how many siblings are alive), the group is a single RBD
node -- it cannot be modelled as ``n`` independent nodes.

This is the "self-loading" sibling of :class:`~repyability.StandbyModel`: the
condition-based layer streams a component's load from sensors, whereas here the
load is computed from the group's own survivors (``L / s``). See issue #38.

Engine (per Monte-Carlo replicate) -- a cumulative-exposure event loop::

    draw each unit's baseline exposure-to-failure threshold  tau_i
    e_i = 0 (accumulated exposure);  t = 0
    while >= k units survive:
        phi_i = M_i.phi(L / survivors)      # aging rate under the load
        dt    = min_i (tau_i - e_i) / phi_i  # real time to next failure
        t    += dt;  e_i += phi_i * dt       # advance; accrue exposure
        drop the failing unit
    group lifetime = t

For identical units with an Exponential baseline the group lifetime has an
exact closed form (a hypoexponential distribution); otherwise the survival
function is a Kaplan-Meier fit to simulated lifetimes (like ``StandbyModel``).
"""

import numpy as np
from surpyval import KaplanMeier

from repyability.utils.wrappers import conditional_survival, numpy_seed

from ._model_utils import is_exponential

_AFT_KIND = "Accelerated Failure Time"


def _baseline(model):
    """The AFT model's univariate baseline distribution ``R0`` (phi = 1)."""
    return model.distribution.from_params(
        [float(p) for p in np.atleast_1d(model.dist_params)]
    )


class _HypoexponentialSurvival:
    """Survival function of a sum of independent Exponentials with *distinct*
    rates (a hypoexponential / generalised-Erlang distribution).

    For rates ``r_1, ..., r_m`` the survival is the partial-fraction sum
    ``sf(t) = sum_j C_j exp(-r_j t)`` with ``C_j = prod_{l != j} r_l / (r_l -
    r_j)`` (and ``sum_j C_j = 1``, so ``sf(0) = 1``). The mean is
    ``sum_j 1 / r_j``.
    """

    def __init__(self, rates):
        r = np.asarray(rates, dtype=float)
        m = len(r)
        coef = np.ones(m)
        for j in range(m):
            for ell in range(m):
                if ell != j:
                    coef[j] *= r[ell] / (r[ell] - r[j])
        self.rates = r
        self.coef = coef

    def sf(self, x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        terms = self.coef[None, :] * np.exp(-self.rates[None, :] * x[:, None])
        return np.clip(terms.sum(axis=1), 0.0, 1.0)

    def ff(self, x):
        return 1.0 - self.sf(x)

    def mean(self, *args, **kwargs):
        return float(np.sum(1.0 / self.rates))


def _identical_exponential_stage_rates(models, load, k, baselines, phi_table):
    """Stage failure rates for identical Exponential-baseline units, or None.

    When every unit is the same AFT model with an Exponential baseline of rate
    ``lambda``, the group passes through stages ``s = N, N-1, ..., k`` alive.
    In stage ``s`` each survivor's residual exposure is Exponential
    (memoryless) and accrues at rate ``phi(L / s)``, so its residual *time* is
    ``Exponential(lambda * phi(L / s))`` and the ``s`` survivors' next failure
    is ``Exponential(s * lambda * phi(L / s))``. The group lifetime sums these
    independent stage times -- a hypoexponential with those rates. With
    ``phi == 1`` the rates are ``s * lambda`` (``s = N..k``): exactly the
    ``(N-k+1)``-th order statistic of ``N`` i.i.d. Exponentials, i.e. the
    ordinary k-out-of-n parallel result.
    """
    n = len(models)
    lam = None
    for base in baselines:
        if not is_exponential(base):
            return None
        mean = float(np.atleast_1d(base.mean())[0])
        if mean <= 0.0:
            return None
        rate = 1.0 / mean
        if lam is None:
            lam = rate
        elif not np.isclose(rate, lam):
            return None
    # phi must be identical across units at every stage for the closed form.
    rates = []
    for s in range(n, k - 1, -1):
        phi_s = phi_table[0, s - 1]
        if not np.allclose(phi_table[:, s - 1], phi_s):
            return None
        rates.append(s * lam * phi_s)
    return np.asarray(rates, dtype=float)


class LoadSharingModel:
    """A load-sharing arrangement of coupled AFT units as one RBD node.

    Parameters
    ----------
    models : sequence of fitted surpyval AFT models
        The units. Each must be an accelerated-failure-time model fitted with
        the (scalar) load as its covariate, exposing ``phi(load)`` and an AFT
        baseline distribution.
    load : float
        The total load ``L`` shared by the active units. Each of ``s``
        survivors carries ``L / s``.
    k : int, optional
        The minimum number of surviving units for the group to work, by default
        1. The group fails at the ``(N - k + 1)``-th unit failure.
    n_sims : int, optional
        Monte-Carlo replicates for the Kaplan-Meier fit when no closed form
        applies, by default 10_000.
    lower : float, optional
        ``set_lower_limit`` passed to the Kaplan-Meier fit, by default -inf.
    seed : int or None, optional
        Seed for the Monte-Carlo fit (reproducible), by default None.

    Notes
    -----
    ``phi`` with no load effect (``phi == 1``) reproduces the ordinary
    k-out-of-n parallel result exactly, since the units then neither share
    stress nor age faster as siblings fail.
    """

    def __init__(
        self,
        models,
        load,
        k=1,
        n_sims=10_000,
        lower=-np.inf,
        seed=None,
    ):
        models = list(models)
        if len(models) == 0:
            raise ValueError("LoadSharingModel needs at least one unit.")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k!r}.")
        if k > len(models):
            raise ValueError(
                f"k ({k}) cannot exceed the number of units ({len(models)})."
            )
        for m in models:
            if getattr(m, "kind", None) != _AFT_KIND:
                raise ValueError(
                    "LoadSharingModel units must be fitted accelerated-"
                    "failure-time (AFT) models exposing phi(load); got "
                    f"{type(m).__name__} with "
                    f"kind={getattr(m, 'kind', None)!r}."
                )
        self.models = models
        self.load = float(load)
        self.k = int(k)
        self.N = len(models)
        self.n_sims = n_sims

        self._baselines = [_baseline(m) for m in models]
        # phi_table[i, s-1] = unit i's aging rate when s units share the load.
        self._phi_table = np.array(
            [
                [
                    float(np.ravel(m.phi(np.atleast_2d(self.load / s)))[0])
                    for s in range(1, self.N + 1)
                ]
                for m in models
            ]
        )

        rates = _identical_exponential_stage_rates(
            models, self.load, self.k, self._baselines, self._phi_table
        )
        # The partial-fraction hypoexponential needs distinct rates; if a load
        # effect collides two stage rates, fall back to the simulation path.
        if rates is not None and _all_distinct(rates):
            self._sf_model: object = _HypoexponentialSurvival(rates)
            self.model = None
        else:
            x_random = self.random(n_sims, seed=seed)
            self.model = KaplanMeier.fit(x_random, set_lower_limit=lower)
            self._sf_model = None

    def random(self, size, seed=None):
        """Monte-Carlo simulate ``size`` group lifetimes via the cumulative-
        exposure event loop."""
        n, k, phi_tab = self.N, self.k, self._phi_table
        with numpy_seed(seed):
            # Baseline exposure-to-failure thresholds: (N, size).
            tau = np.empty((n, size))
            for i, base in enumerate(self._baselines):
                tau[i] = np.asarray(base.random(size), dtype=float).reshape(-1)

            out = np.empty(size)
            for j in range(size):
                thr = tau[:, j]
                e = np.zeros(n)
                alive = np.ones(n, dtype=bool)
                t = 0.0
                s = n
                while s >= k:
                    idx = np.flatnonzero(alive)
                    phi = phi_tab[idx, s - 1]
                    remaining = (thr[idx] - e[idx]) / phi
                    w = int(np.argmin(remaining))
                    dt = remaining[w]
                    t += dt
                    e[idx] += phi * dt
                    alive[idx[w]] = False
                    s -= 1
                out[j] = t
        return out

    def mean(self, N=10_000, seed=None):
        if self._sf_model is not None:
            return self._sf_model.mean()
        return float(self.random(N, seed=seed).mean())

    def sf(self, *args, **kwargs):
        if self._sf_model is not None:
            return self._sf_model.sf(*args, **kwargs)
        return self.model.sf(*args, **kwargs)

    def ff(self, *args, **kwargs):
        if self._sf_model is not None:
            return self._sf_model.ff(*args, **kwargs)
        return self.model.ff(*args, **kwargs)

    def cs(self, x, X):
        """Conditional survival ``R(x | X) = sf(X + x) / sf(X)``."""
        return conditional_survival(self, x, X)

    @property
    def is_simulated(self) -> bool:
        """True if the survival function is a Monte-Carlo (Kaplan-Meier) fit
        rather than the exact hypoexponential closed form."""
        return self._sf_model is None


def _all_distinct(rates, rtol=1e-6):
    r = np.sort(np.asarray(rates, dtype=float))
    if len(r) < 2:
        return True
    gaps = np.diff(r)
    scale = np.maximum(np.abs(r[:-1]), np.abs(r[1:]))
    return bool(np.all(gaps > rtol * np.maximum(scale, 1e-12)))
