"""The replacement cycle of a minimally repaired unit (#402).

A unit repaired minimally ("as bad as old": Kijima with ``q = 1``, or a
fitted Poisson process such as Crow-AMSAA, Duane or Cox-Lewis) fails as a
non-homogeneous Poisson process in its operating age, whose cumulative
intensity ``H`` is its life's cumulative hazard. Replaced as new at its
``n``-th failure since it was renewed (a spec's ``"replace_after"``), or
at an operating age ``T`` (age replacement), whichever comes first, it
starts afresh: each replacement ends a renewal cycle, and its long-run
values are renewal-reward ratios over one cycle.

In plain words: the cycle lasts until the ``n``-th failure or the age
``T``. ``N(t)``, the failures by the age ``t``, is Poisson with mean
``H(t)``, so the ``n``-th failure comes after ``t`` with probability
``Q(n, H(t))`` (``Q`` the regularised upper incomplete gamma function:
the chance of fewer than ``n`` failures). Hence:

- the cycle's operating time, ``E[min(T, T_n)] = integral from 0 to T of
  Q(n, H(t)) dt``;
- its failures, ``E[min(N(T), n)] = H(T) Q(n - 1, H(T)) + n P(n, H(T))``
  (``P = 1 - Q``; the first term 0 for ``n = 1``);
- whether it ends at a failure, with probability ``P(n, H(T))``, or at
  the age, ``Q(n, H(T))``.

For a power law, ``H(t) = (t / alpha) ** beta`` (Crow-AMSAA, Duane, a
Weibull life), the operating time is in closed form,
``T Q(n, H) + alpha Gamma(n + 1 / beta) / Gamma(n) P(n + 1 / beta, H)``
with ``H = H(T)`` (by parts, as ``t = alpha u ** (1 / beta)``), and so
``E[T_n] = alpha Gamma(n + 1 / beta) / Gamma(n)`` without the age; for
any other ``H`` (Cox-Lewis's, say) it is the integral, by adaptive
quadrature. Without ``n`` (age replacement alone) the cycle lasts ``T``
and has ``H(T)`` failures, exactly, whatever ``H``.

The diagram adds the down time to the operating time (each failure's
repair, the ``n``-th's too, which replaces the unit, and the
replacement's time at the age); ``Repairable`` takes ``E[T_n]`` for its
failure-limit policy.
"""

import math
from typing import Callable, NamedTuple, Optional, Tuple

import numpy as np
from scipy import integrate, optimize
from scipy.special import gammainc, gammaincc, gammaln

#: The share of the operating time the quadrature may leave out past the
#: last piece it adds up (see ``_integral``).
_TAIL = 1e-13


class Cycle(NamedTuple):
    """One replacement cycle of a minimally repaired unit, in its
    operating time: ``up`` its mean operating time, ``failures`` its mean
    number of failures, ``replaced`` the probability that it ends at the
    ``n``-th failure, ``maintained`` that it ends at the age (they sum to
    1), and ``route`` ``"exact"`` or ``"numerical"`` (by quadrature)."""

    up: float
    failures: float
    replaced: float
    maintained: float
    route: str


def power_law(model) -> Optional[Tuple[float, float]]:
    """``(alpha, beta)`` of a life whose cumulative hazard is ``(t /
    alpha) ** beta``: a surpyval Weibull (an Exponential, ``beta = 1``)
    with no offset, limited failure population or zero inflation; None for
    any other."""
    from repyability.rbd._model_utils import lfp_p

    name = getattr(getattr(model, "dist", None), "name", None)
    if (
        name not in ("Exponential", "Weibull")
        or lfp_p(model) != 1
        or getattr(model, "f0", None) != 0
        or getattr(model, "gamma", 0)
    ):
        return None
    if name == "Exponential":
        return 1.0 / float(model.params[0]), 1.0
    alpha, beta = (float(p) for p in model.params)
    return alpha, beta


def _upper(n: float, h: float) -> float:
    """``Q(n, h)``: the chance of fewer than ``n`` failures when ``h``
    are expected (0 for ``n = 0``)."""
    return float(gammaincc(n, h)) if n > 0 else 0.0


def cycle(
    hazard: Callable[[float], float],
    n: Optional[int],
    age: float,
    power: Optional[Tuple[float, float]] = None,
) -> Cycle:
    """The replacement cycle (see the module's notes) of a unit whose
    failures by the operating age ``t`` number ``hazard(t)`` on average,
    replaced at its ``n``-th failure (None: never) or at the operating
    age ``age`` (``inf``: never), one of them finite; ``power`` the power
    law ``hazard`` is, if it is one (``power_law``). Raises
    ``NotImplementedError`` if the cycle may never end: with no age, a
    ``hazard`` that levels off below ``n`` failures in expectation (a
    falling intensity)."""
    finite = math.isfinite(age)
    if n is None:
        assert finite, "a cycle ends at a failure or at an age"
        h = float(hazard(age))
        return Cycle(age, h, 0.0, 1.0, "exact")
    h = float(hazard(age)) if finite else math.inf
    replaced = 1.0 if not finite else float(gammainc(n, h))
    maintained = 0.0 if not finite else float(gammaincc(n, h))
    if finite:
        failures = h * _upper(n - 1, h) + n * replaced
    else:
        failures = float(n)
    if power is not None:
        alpha, beta = power
        # alpha * Gamma(n + 1 / beta) / Gamma(n), E[T_n].
        nth = alpha * math.exp(gammaln(n + 1.0 / beta) - gammaln(n))
        if not finite:
            return Cycle(nth, failures, 1.0, 0.0, "exact")
        up = age * maintained + nth * float(gammainc(n + 1.0 / beta, h))
        return Cycle(min(up, age), failures, replaced, maintained, "exact")
    up = _integral(hazard, n, age)
    return Cycle(up, failures, replaced, maintained, "numerical")


def _time_at(hazard: Callable[[float], float], level: float) -> float:
    """The age at which ``hazard`` reaches ``level``, or ``inf`` if it
    never does (it levels off below it)."""
    hi = 1.0
    with np.errstate(all="ignore"):
        if float(hazard(hi)) < level:
            while float(hazard(hi)) < level:
                hi *= 2.0
                if hi > 1e300:
                    return math.inf
            lo = hi / 2.0
        else:
            lo = hi
            while float(hazard(lo)) >= level and lo > 1e-300:
                lo /= 2.0
        return float(
            optimize.brentq(
                lambda t: float(hazard(t)) - level,
                lo,
                hi,
                xtol=1e-300,
                rtol=1e-12,
            )
        )


def _integral(hazard: Callable[[float], float], n: int, age: float) -> float:
    """``integral from 0 to age of Q(n, hazard(t)) dt`` (``age`` may be
    ``inf``), by adaptive quadrature on pieces that double in length from
    an eighth of the age at which ``hazard`` reaches ``n`` (where ``Q``
    falls through one half), up to ``age``; with no age, until a piece
    adds less than ``_TAIL`` of the total."""

    def survival(t: float) -> float:
        with np.errstate(all="ignore"):
            return float(gammaincc(n, float(hazard(t))))

    middle = _time_at(hazard, float(n))
    if not math.isfinite(middle):
        if not math.isfinite(age):
            # Fewer than n failures expected ever: T_n may be infinite.
            _never_ends(n)
        middle = age
    lo, total = 0.0, 0.0
    hi = min(middle / 8.0, age)
    while True:
        piece = integrate.quad(
            survival, lo, hi, epsabs=0.0, epsrel=1e-12, limit=200
        )[0]
        total += piece
        if hi >= age or (lo > middle and piece <= _TAIL * total):
            return total
        if hi > 1e300:
            _never_ends(n)
        lo, hi = hi, min(2.0 * hi, age)


def _never_ends(n: int):
    raise NotImplementedError(
        "its failure intensity falls so that it may never reach its "
        f"failure {n}, and the time to that failure has no finite mean"
    )


def nth_failure_mean(
    hazard: Callable[[float], float],
    n: int,
    power: Optional[Tuple[float, float]] = None,
) -> float:
    """``E[T_n]``, the mean operating age at the ``n``-th failure of a
    unit repaired minimally (see ``cycle``)."""
    return cycle(hazard, n, math.inf, power).up
