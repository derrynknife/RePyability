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

**A switch that can fail** (``switching_probability`` ``p`` below 1).
When the operating unit fails with the other ready, the switch onto it
works with probability ``p``; if it does not, the pair is down while the
failed unit is repaired, which then operates again, as new, with the
spare still waiting. So the pair has a second state it starts afresh
in, a unit starting to operate with the other *ready*: from it, after
``X``, the switch works (the cycle above begins) or the pair is down
``R`` and is in that state again. A cycle that ends with the other unit
ready (``R < X``) ends likewise with a switch. The long-run values are
the two states' sums weighted by how often each is entered (``p`` to
``(1 - p) P(R < X)``); with ``p = 0`` the spare is never used, and the
pair is one unit. ``switched`` gives what a failed switch adds until
the pair is back at the cycle's start, which ``_standby_both`` adds to
its own equation.

**Over time, from new** (#629, step 2; ``over_time``, a ``PairCurve``).
Both units are ready at 0: the first operates, and the pair first
regenerates when it fails, at ``X`` (``F``). From a regeneration at 0 a
cycle ends before ``s`` with probability ``F(s) G(s)``, and within it
the pair is down at ``s`` if ``X <= s < R``, ``K(s) = F(s) (1 - G(s))``;
it fails (at ``X``, if ``R > X``) by ``s`` with the measure ``(1 - G)
dF``, and a unit fails at ``X``. So with ``u`` the regenerations'
renewal measure from new, ``u = F * (delta_0 + FG + FG * FG + ...)``,
the pair is down at ``t`` with probability ``(u * K)(t)``, has failed
``(u * (1 - G) dF)(t)`` times by then, and its units ``F(t) + (u *
F)(t)`` times. These are worked out on a grid as a unit's are
(``_point_availability``): each distribution a measure on the grid's
points that keeps every cell's mass and mean, the renewals by inverting
the power series ``1 - c`` of the cycle's, and the down periods spread
back over hat functions and integrated against ``K`` exactly, so a
repair far shorter than a step costs nothing. The error falls as the
square of the step, so the values on two grids, one of half the other's
step, are extrapolated at the coarser one's points (to about 1e-9 at
them; the curve is linear between them).
"""

import math
from typing import Dict, Optional

import numpy as np

from repyability.rbd import _point_availability as _grid
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


def mean_time(model) -> float:
    """A model's mean time, the integral of its survival function."""

    def S(t: float) -> float:
        return float(_values(model, [t], "sf")[0])

    return _integral(S, _cuts(model))


def unswitched(life, repair) -> StandbyLongRun:
    """The long-run values of a pair whose switch never works: one unit,
    up for a life and down for a repair in turn (its spare is never
    used)."""
    up, down = mean_time(life), mean_time(repair)
    return StandbyLongRun(
        availability=up / (up + down),
        failure_frequency=1.0 / (up + down),
        unit_failure_frequency=1.0 / (up + down),
        unavailability=down / (up + down),
    )


def switched(repair, switching: float) -> np.ndarray:
    """What a failed switch adds, on average, until a switch next works
    (``switching`` above 0): the units' failures, the down time and the
    system's failures. The pair is down for a repair, and then each life
    of the unit ends with a switch that works, or with another repair:
    ``1 / switching`` lives, each a unit's failure, and as many failures
    of the system and repairs down, this one included."""
    return np.array([1.0, mean_time(repair), 1.0]) / switching


def long_run(life, repair, switching: float = 1.0) -> StandbyLongRun:
    """The long-run values of a cold standby pair with one repair at a
    time, lives ``life`` and repairs ``repair`` (any models with ``ff``,
    ``sf`` and a density ``df``), whose switch works with probability
    ``switching``."""
    if switching <= 0.0:
        return unswitched(life, repair)

    def F(t: float) -> float:
        return float(_values(life, [t], "ff")[0])

    def S(t: float) -> float:
        return float(_values(life, [t], "sf")[0])

    def Gbar(t: float) -> float:
        return float(_values(repair, [t], "sf")[0])

    def f(t: float) -> float:
        return float(_values(life, [t], "df")[0])

    cuts = _cuts(life, repair)
    # 1 - F G = S + F Gbar (each term kept, for their digits).
    cycle = _integral(lambda t: S(t) + F(t) * Gbar(t), cuts)
    down = _integral(lambda t: F(t) * Gbar(t), cuts)
    fails = _integral(lambda t: f(t) * Gbar(t), cuts)
    units = 1.0
    if switching < 1.0:
        # A cycle that ends with the other unit ready, and a switch that
        # fails: what follows until a switch works. The lives in it add
        # their own length to the cycle.
        ready = _integral(lambda t: f(t) * _values(repair, [t], "ff")[0], cuts)
        more_units, more_down, more_fails = (
            (1.0 - switching) * ready * switched(repair, switching)
        )
        cycle += mean_time(life) * more_units + more_down
        down += more_down
        fails += more_fails
        units += more_units
    unavailability = down / cycle
    return StandbyLongRun(
        availability=1.0 - unavailability,
        failure_frequency=fails / cycle,
        unit_failure_frequency=units / cycle,
        unavailability=unavailability,
    )


class PairCurve:
    """A cold standby pair with one repair at a time over time, from new
    (see the module's notes, and ``_point_availability`` for what a curve
    gives): on the grid ``times``, the probability that it is ``down``,
    and its expected ``failures`` and its ``units``' failures (the
    repairs it pays for) before each point; linear between them. Past
    the grid's end the curve holds its long-run values ``long_run`` if
    given (it has settled there by then): down as often, and the counts
    growing at their rates."""

    period = None
    #: Its unavailability is worked out as itself (``down_at``, #237).
    has_down = True

    def __init__(
        self,
        step: float,
        down: np.ndarray,
        failures: np.ndarray,
        units: np.ndarray,
        long_run: Optional[StandbyLongRun] = None,
    ):
        self.times = step * np.arange(len(down))
        self.down = down
        self.failures = failures
        self.units = units
        self.long_run = long_run

    @property
    def settle(self) -> float:
        """The time after which the curve is constant."""
        return np.inf if self.long_run is None else float(self.times[-1])

    def down_at(self, x: np.ndarray) -> np.ndarray:
        """The probability of being down at each time ``x``."""
        x = np.asarray(x, dtype=float)
        out = np.interp(x, self.times, self.down)
        if self.long_run is not None:
            out = np.where(
                x > self.times[-1], self.long_run.unavailability, out
            )
        return np.clip(out, 0.0, 1.0)

    def at(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - self.down_at(x)

    def _counted(self, x: np.ndarray, counts: np.ndarray, rate: float):
        out = np.interp(x, self.times, counts)
        if self.long_run is not None:
            past = np.maximum(x - self.times[-1], 0.0)
            out = out + rate * past
        return np.maximum(out, 0.0)

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The pair's expected failures before each time ``x``, and the
        repairs it pays for (its corrective actions)."""
        x = np.asarray(x, dtype=float)
        rates = (0.0, 0.0)
        if self.long_run is not None:
            rates = (
                self.long_run.failure_frequency,
                self.long_run.unit_failure_frequency,
            )
        return _grid._events(
            self._counted(x, self.failures, rates[0]),
            corrective=self._counted(x, self.units, rates[1]),
        )

    def atoms(self, stop: float) -> "_grid.Atoms":
        """None: the pair's units fail at random times."""
        return _grid.Atoms.none()

    def knots(self, start: float, stop: float) -> np.ndarray:
        return self.times[(self.times >= start) & (self.times <= stop)]

    def breaks(self, start: float, stop: float) -> np.ndarray:
        """The grid's ends (see ``_point_availability.curve_breaks``)."""
        ends = self.times[[0, -1]]
        return ends[(ends >= start) & (ends <= stop)]

    def grids(self) -> list:
        """Its grid, to its end (see ``curve_grids``)."""
        if len(self.times) < 2:
            return []
        return [(float(self.times[1]), float(self.times[-1]))]

    def settled(self, values: StandbyLongRun) -> bool:
        """Whether the curve has reached the long-run ``values`` over the
        last quarter of its grid: its unavailability to 1e-10, and its
        counts' rates to 1e-7 of them."""
        n = len(self.times) - 1
        tail = slice(n - n // 4, None)
        if len(self.times[tail]) < 3:
            return False
        if np.any(np.abs(self.down[tail] - values.unavailability) >= 1e-10):
            return False
        step = self.times[1] - self.times[0]
        for counts, rate in (
            (self.failures, values.failure_frequency),
            (self.units, values.unit_failure_frequency),
        ):
            slope = np.diff(counts[tail]) / step
            if np.any(np.abs(slope - rate) > 1e-7 * rate):
                return False
        return True


def steady(values: StandbyLongRun) -> PairCurve:
    """A pair long in service: in its long-run state from 0."""
    return PairCurve(
        1.0,
        np.array([values.unavailability]),
        np.zeros(1),
        np.zeros(1),
        values,
    )


def _on_grid(life, repair, splits, step: float, n: int):
    """The pair's probability of being down at each point of the grid of
    ``n`` steps of ``step``, and its expected failures and its units'
    before each (see the module's notes)."""
    t = step * np.arange(n + 1)
    convolve = _grid._Spectra(n + 1).convolve

    def F(x):
        return _values(life, x, "ff")

    def survives(x):
        return _values(repair, x, "sf")

    def down(x):
        return F(x) * survives(x)

    def density(x):
        return F(x) * _values(repair, x, "df")

    with np.errstate(divide="ignore", invalid="ignore"):
        cycle = _grid._lattice(
            *_grid._cells(lambda x: F(x) * _values(repair, x, "ff"), t, splits)
        )
        first = _grid._lattice(*_grid._cells(F, t, splits))
        hat = _grid._survival_weights(down, t, splits)[2]
        # The system's failures in a cycle, the measure (1 - G) dF: each
        # cell's mass and mean, by parts (no density of the life).
        edge = down(t)
        through = _grid._cell_integrals(density, t, splits)
        mass = np.diff(edge) + through
        mean = (
            np.diff(t * edge)
            - _grid._cell_integrals(down, t, splits)
            + _grid._cell_integrals(lambda x: x * density(x), t, splits)
        )
    position = np.full(n, 0.5)
    has = mass > 0.0
    position[has] = (mean[has] / mass[has] - t[:-1][has]) / step
    failing = _grid._lattice(0.0, mass, np.clip(position, 0.0, 1.0))
    one_minus = -cycle
    one_minus[0] += 1.0
    starts = convolve(first, _grid._series_inverse(one_minus, n + 1))
    return (
        convolve(starts, hat),
        _grid._running(convolve(starts, failing)),
        _grid._running(first + convolve(starts, first)),
    )


def over_time(life, repair, step: float, n: int) -> PairCurve:
    """A cold standby pair with one repair at a time, from new, on a grid
    of ``n`` steps of ``step`` (see the module's notes): the values on it
    and on one of half its step, extrapolated."""
    splits = np.union1d(_grid.knots(life), _grid.knots(repair))
    coarse = _on_grid(life, repair, splits, step, n)
    fine = _on_grid(life, repair, splits, 0.5 * step, 2 * n)
    down, failures, units = (
        np.maximum((4.0 * b[::2] - a) / 3.0, 0.0) for a, b in zip(coarse, fine)
    )
    return PairCurve(step, down, failures, units)
