"""The long-run values of a cold standby pair with general lives and
repairs (#629, step 1): ``units=2``, ``k=1``, a spare that does not age
(``dormancy_factor`` 0) and switches in surely, and one repair at a time
(``repair_crews`` 1 among the pair's units; both repaired at once is
``_standby_both``).

**The cycle.** The pair regenerates each time a unit fails while the
other is ready: the ready one is switched in, as new (a cold spare does
not age), and the failed one starts its repair, at once. From there,
with ``X`` the operating unit's life (``F``) and ``R`` the other's
repair (``G``):

- ``R < X``: the repaired unit waits; at ``X`` the operating one fails
  with it ready, and the cycle ends after ``X``, up throughout;
- ``R >= X``: the pair is down from ``X`` while the failed unit waits
  for the one crew; at ``R`` the repaired unit operates, as new, and the
  crew starts the other's repair, the cycle's end after ``R``, down
  ``R - X``.

So a cycle lasts ``max(X, R)``, ``E = int (1 - F G)``, with ``E[(R -
X)+] = int F (1 - G)`` of it down and one system failure if ``R >= X``,
``P = int (1 - G) dF``; and one unit's repair in each. By the renewal
reward theorem the long-run unavailability is ``E[(R - X)+] / E``, the
failure frequency ``P / E`` and the units' repairs ``1 / E`` per unit
time. With exponential lives and repairs this is the group's Markov
chain with one crew (``_standby_chain``), which checks it.

The integrals are worked out by adaptive quadrature (``scipy``'s
``quad``), on pieces cut at the models' quantiles, to a relative
``RTOL``; the down time is integrated as itself, not as one less the
up time, so a small unavailability keeps its digits.
"""

import math
from typing import Optional

import numpy as np

from repyability.rbd._standby_chain import StandbyLongRun

#: The relative accuracy of each integral.
RTOL = 1e-10
#: The quantiles the integrals' pieces are cut at, of each model.
_CUTS = (0.01, 0.1, 0.5, 0.9, 0.99, 0.9999, 1 - 1e-8, 1 - 1e-12)


def why_not(arrangement) -> Optional[str]:
    """Why a standby group's long-run values are worked out neither by
    its cycle (see the module's notes) nor, with both units repaired at
    once, by ``_standby_both``; or None."""
    if arrangement.rates is not None:
        return "it is not a standby group (its units operate together)"
    if arrangement.units != 2 or arrangement.k != 1:
        return (
            f"it has {arrangement.units} units, {arrangement.k} needed, and "
            "they are worked out for a pair, one needed, alone"
        )
    if arrangement.dormancy_factor != 0.0:
        return "its spare ages while it waits (a warm spare)"
    if arrangement.switching_probability != 1.0:
        return "switching onto its spare may fail"
    return None


def _values(model, x, name: str) -> np.ndarray:
    return np.ravel(getattr(model, name)(np.asarray(x, dtype=float)))


def _integral(integrand, cuts) -> float:
    """``int_0^inf integrand`` by adaptive quadrature on the pieces
    between ``cuts``, and on to infinity past the last."""
    from scipy import integrate

    edges = [0.0, *sorted(c for c in cuts if c > 0.0 and math.isfinite(c))]
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            total += integrate.quad(
                integrand, lo, hi, epsabs=0.0, epsrel=RTOL, limit=200
            )[0]
    total += integrate.quad(
        integrand, edges[-1], math.inf, epsabs=0.0, epsrel=RTOL, limit=200
    )[0]
    return float(total)


def _cuts(*models) -> list:
    out = []
    for model in models:
        q = getattr(model, "qf", None)
        if q is None:
            continue
        with np.errstate(all="ignore"):
            values = _values(model, np.array(_CUTS), "qf")
        out += [float(v) for v in values if np.isfinite(v) and v > 0.0]
    return out


def long_run(life, repair) -> StandbyLongRun:
    """The long-run values of a cold standby pair with one repair at a
    time, lives ``life`` and repairs ``repair`` (any models with ``ff``,
    ``sf`` and a density ``df``)."""

    def F(t: float) -> float:
        return float(_values(life, [t], "ff")[0])

    def S(t: float) -> float:
        return float(_values(life, [t], "sf")[0])

    def G(t: float) -> float:
        return float(_values(repair, [t], "ff")[0])

    def Gbar(t: float) -> float:
        return float(_values(repair, [t], "sf")[0])

    def f(t: float) -> float:
        return float(_values(life, [t], "df")[0])

    cuts = _cuts(life, repair)
    # 1 - F G = S + F Gbar (each term kept, for their digits).
    cycle = _integral(lambda t: S(t) + F(t) * Gbar(t), cuts)
    down = _integral(lambda t: F(t) * Gbar(t), cuts)
    fails = _integral(lambda t: f(t) * Gbar(t), cuts)
    unavailability = down / cycle
    return StandbyLongRun(
        availability=1.0 - unavailability,
        failure_frequency=fails / cycle,
        unit_failure_frequency=1.0 / cycle,
        unavailability=unavailability,
    )
