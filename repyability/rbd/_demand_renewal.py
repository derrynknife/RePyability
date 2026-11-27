"""Values over time of a unit that wears out with demands at set times, or
on a plan of rates that changes, beside its life
(``_demands.LifeWithPlannedDemands``), without simulating: a renewal
equation on the diagram's clock.

A unit put into service at ``u`` survives ``s`` more with chance ``R(s)
E(u, u + s)``, ``R`` its life's survival and ``E(u, v)`` the demands'
between: ``(1 - p)`` for each demand in ``(u, v]`` at set times, ``exp(-p
(Λ(v) - Λ(u)))`` on a plan. It is repaired as new in a time ``R_r``, and
the demands stay on the calendar, so the kernel depends on when each unit
is put in, and the renewals are not a convolution: the equation is solved
forward in time, cell by cell.

The grid's points are a uniform step's, with the demands, the plan's
times and the times a repair of fixed length after the start or a demand
ends added, so that no cell holds a demand or a change of the plan: a
unit put in within a cell meets the demands at and after its end, and
none at its start. In each cell the renewals (the repairs ending) and the
failures of the life (on a plan, of both modes) are kept as their mass
and mean, and each is taken, for what follows, as its share at the
cell's two ends (keeping its mass and mean, as ``unit_curve``'s lattice
does), counted with the demands of its cell. A cell's failures from the
units put in before it, and its renewals from the failures before it,
are the life's and the repair's distributions integrated exactly over the
cell (4-point Gauss-Legendre, split at their quantiles); the failures and
renewals within one cell, which feed each other, are found by a few
rounds of substitution (each round's change is the life's and the
repair's chances within a cell, multiplied).

At set times each demand fails the unit with probability ``p`` if it is
up then: ``p`` less ``p`` times the chance that it is in a repair (one
that ends at the demand is not called on), from the failures before it,
which are the earlier demands' (exactly, as ``_demands.failure_chances``
counts them) and the life's in the cells before it (their linear
densities against the repair's survival). Those failures are atoms at
the demands, kept off the grid (a ``GridCurve``'s dips): their repairs
are exact at any time, however short. With a life that never fails,
nothing is on the grid, and the values are ``DemandCurve``'s.

The point availability is one less the chance of being in a repair: the
atoms' repairs, and the life's failures in each cell (a linear density
with the cell's mass and mean) against the repair time's survival, on
the grid, linear between its points.

Accuracy: each share at a cell's ends keeps the mass and mean of what it
stands for, so what is made of it is wrong by about the square of the
step against the scale on which it bends, ``(h / s)^2 / 8`` of the chance
moved for each pass (a failure, a renewal) it takes; the step is at most
``1 / STEPS_PER_SCALE`` of the life's typical up time and ``1 /
MIN_STEPS`` of the horizon, so about 5e-7 on a value over a window with
the defaults, a few times that pointwise next to a bend of the curve
(the end of a fixed repair) between the grid's points.
"""

import math
from typing import Dict

import numpy as np

from ._common import _sf_values
from ._point_availability import (
    Atoms,
    _atoms,
    _cell_integrals,
    _density,
    _events,
    knots,
)

#: Grid steps over the life's typical up time (``_curves._up_scale``), and
#: at least over the horizon; at most this many points in all, and no
#: coarser than ``COARSEST`` steps over the typical up time (else the
#: values are refused, and the simulations estimate them).
STEPS_PER_SCALE = 400
MIN_STEPS = 1024
MAX_STEPS = 4096
COARSEST = 50

#: Rounds of substitution for the failures and renewals within one cell.
_ROUNDS = 4


def _position(mass: float, moment: float, width: float) -> float:
    """Where in its cell (0 to 1) a mass with ``moment`` about the cell's
    start has its mean (the middle for no mass)."""
    if not mass > 0.0 or not width > 0.0:
        return 0.5
    return min(max(moment / (mass * width), 0.0), 1.0)


class PlannedDemandsCurve:
    """The point availability and events of a unit that wears out with
    demands at set times or on a plan beside its life, repaired as new in
    a time with model ``repair``, from new (or up at ``age``) at 0 to
    ``horizon``, on a grid of step at most ``step`` (see the module's
    docstring). ``life`` is the ``LifeWithPlannedDemands``."""

    settle = np.inf
    period = None

    def __init__(
        self, life, repair, horizon: float, step: float, age: float = 0.0
    ):
        self.life = life
        self.repair = repair
        self.p = life.p
        self.age = float(age)
        model = life.life
        self._life_knots = np.unique(knots(model))
        self._repair_knots = np.unique(knots(repair))
        finite = self._repair_knots[np.isfinite(self._repair_knots)]
        self._reach = float(finite.max()) if finite.size else np.inf
        tail = float(self._repair_sf(np.array([self._reach]))[0])
        if not tail < 1e-15:
            self._reach = np.inf
        self.demands = life.times
        self.t = self._grid(float(horizon), float(step))
        self._solve()

    # -- the models -------------------------------------------------------

    def _repair_sf(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        out = _sf_values(self.repair.sf, np.maximum(s, 0.0))
        return np.where(s < 0.0, 1.0, out)

    def _life_sf(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        return _sf_values(self.life.life.sf, np.maximum(s, 0.0))

    def _first_sf(self, s: np.ndarray) -> np.ndarray:
        """The first unit's survival from 0, up at ``age``."""
        if not self.age:
            return self._life_sf(s)
        s = np.asarray(s, dtype=float)
        lived = float(self._life_sf(np.array([self.age]))[0])
        return self._life_sf(self.age + s) / lived

    def _expected(self, s: np.ndarray) -> np.ndarray:
        """The plan's expected demands ``Λ`` from 0 to each time."""
        return np.asarray(
            self.life._expected.operating(np.asarray(s, dtype=float)),
            dtype=float,
        )

    # -- the grid -------------------------------------------------------------

    def _grid(self, horizon: float, step: float) -> np.ndarray:
        """The uniform step's points to ``horizon``, with the demands (or
        the plan's times) and the ends of repairs of fixed length from 0
        and from each demand in it, and the regular points too close to
        them left out."""
        n = max(int(math.ceil(horizon / step - 1e-9)), 1)
        regular = np.linspace(0.0, horizon, n + 1)
        if self.demands is not None:
            starts = self.demands[self.demands < horizon]
        else:
            starts = self.life._knots[self.life._knots < horizon]
        exact = _atoms(
            lambda s: 1.0 - self._repair_sf(s), self._repair_knots, horizon
        )
        ends = [
            begin + at
            for at, _ in exact
            for begin in np.append(
                0.0, starts if self.demands is not None else []
            )
        ]
        special = np.unique(
            np.concatenate([starts, np.asarray(ends, dtype=float)])
        )
        special = special[(special > 0.0) & (special < horizon)]
        if special.size:
            gap = horizon / n
            near = np.searchsorted(special, regular)
            close = np.zeros(regular.shape, dtype=bool)
            for k in (near - 1, near):
                ok = (k >= 0) & (k < special.size)
                close[ok] |= np.abs(special[k[ok]] - regular[ok]) < 1e-6 * gap
            close[[0, -1]] = False
            regular = regular[~close]
        return np.unique(np.concatenate([regular, special]))

    # -- the renewal equation, cell by cell -----------------------------------

    def _push(self, x, edges, sf, splits, scaled=None):
        """The mass and moment (about each cell's start) in each cell
        between ``edges`` of a time from ``x`` with survival ``sf`` (of
        the time since ``x``), times ``scaled`` (a function of the time,
        the demands' survival on a plan): ``S(a) - S(b)`` and ``∫ S - (b -
        a) S(b)`` over each cell ``(a, b]``."""
        if scaled is None:

            def f(s):
                return sf(s - x)

        else:

            def f(s):
                return sf(s - x) * scaled(s)

        values = f(edges)
        integral = _cell_integrals(f, edges, x + splits)
        widths = np.diff(edges)
        return values[:-1] - values[1:], integral - widths * values[1:]

    def _plan_factor(self, x: float):
        """On a plan, the demands' survival from ``x`` to each time."""
        if self.demands is not None:
            return None
        p, at = self.p, float(self._expected(np.array([x]))[0])

        def scaled(s):
            return np.exp(-p * (self._expected(s) - at))

        return scaled

    def _life_from(self, j: int, first=False):
        """The life's failures in the cells after point ``j`` of a unit
        put in at it (the first unit, from 0, for ``first``),
        counted with the demands from it: a unit put in at point ``j``
        meets those after it."""
        t = self.t
        edges = t[j:]
        sf = self._first_sf if first else self._life_sf
        splits = self._life_knots
        if first and self.age:
            splits = splits[splits > self.age] - self.age
        mass, moment = self._push(
            t[j], edges, sf, splits, self._plan_factor(t[j])
        )
        if self.demands is not None and self.p > 0.0:
            met = self._met[j : j + mass.size] - self._met[j]
            factor = np.power(1.0 - self.p, met)
            mass, moment = mass * factor, moment * factor
        return mass, moment

    def _repairs_from(self, j: int) -> tuple:
        """The repairs ending in the cells after point ``j`` of a failure
        at it (but those that take no time)."""
        t = self.t
        return self._push(t[j], t[j:], self._repair_sf, self._repair_knots)

    def _down_at(self, i: int) -> float:
        """The chance of being in the repair of a failure of the life (not
        a demand's) at point ``i``: each cell's linear density with its
        mass and mean, against the repair time's survival."""
        t = self.t
        top = t[i]
        lo = 1
        if np.isfinite(self._reach):
            lo = max(int(np.searchsorted(t, top - self._reach, "right")), 1)
        if lo > i:
            return 0.0
        mass = self._fm[lo : i + 1]
        if not np.any(mass != 0.0):
            return 0.0
        edges = t[lo - 1 : i + 1]
        widths = np.diff(edges)
        moment = self._fx[lo : i + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            position = np.where(mass > 0.0, moment / (mass * widths), 0.5)
        position = np.clip(position, 0.0, 1.0)
        level = mass / widths
        tilt = 12.0 * (position - 0.5)
        sf = self._repair_sf

        def f(s):
            k = np.clip(
                np.searchsorted(edges, s, side="right") - 1, 0, len(widths) - 1
            )
            u = (s - edges[k]) / widths[k]
            return level[k] * (1.0 + tilt[k] * (u - 0.5)) * sf(top - s)

        return float(
            np.sum(_cell_integrals(f, edges, top - self._repair_knots))
        )

    def _demand_chances(self, i: int) -> None:
        """The failures at the demands at point ``i``: each fails the unit
        with ``p`` unless it is in a repair then (see
        ``_demands.failure_chances``)."""
        from ._demands import _under_way

        times, p = self.demands, self.p
        assert times is not None
        first = int(np.searchsorted(times, self.t[i], side="left"))
        last = int(np.searchsorted(times, self.t[i], side="right"))
        if first == last:
            return
        life = self._down_at(i)
        for k in range(first, last):
            if k:
                under = _under_way(self.repair, times[:k], times[k])
                busy = math.fsum(self.chances[:k] * under)
            else:
                busy = 0.0
            self.chances[k] = p * (1.0 - (busy + life))
        self._atom[i] = float(np.sum(self.chances[first:last]))

    def _solve(self) -> None:
        t = self.t
        n = t.size - 1
        widths = np.diff(t)
        p = self.p
        if self.demands is not None:
            self._met = np.searchsorted(self.demands, t, side="right")
            self.chances = np.zeros(self.demands.size)
        # Each cell's failures of the life and renewals: mass and moment
        # about its start (index c for the cell (t[c-1], t[c]]).
        fm, fx = np.zeros(n + 1), np.zeros(n + 1)
        vm, vx = np.zeros(n + 1), np.zeros(n + 1)
        self._fm, self._fx = fm, fx
        self._atom = np.zeros(n + 1)
        # Their shares at each cell's ends.
        f_right, v_right = np.zeros(n + 1), np.zeros(n + 1)
        f_left, v_left = np.zeros(n + 1), np.zeros(n + 1)
        # The first unit, from 0.
        mass, moment = self._life_from(0, first=True)
        fm[1:] += mass
        fx[1:] += moment
        instant = 1.0 - float(self._repair_sf(np.zeros(1))[0])
        for c in range(1, n + 1):
            j = c - 1
            w = widths[j]
            # A unit put in at point j, and a failure there: their parts in
            # cell c (the first), and in the cells after it.
            life_m, life_x = self._life_from(j)
            repair_m, repair_m_x = self._repairs_from(j)
            reach = repair_m[0] + instant
            repair_x = repair_m_x[0]
            # What point j holds already: the renewals in cell j at its
            # end (which met the demand there), and the failures at it.
            held_life = v_right[j]
            if self.demands is not None and j:
                held_life *= (1.0 - p) ** (self._met[j] - self._met[j - 1])
            held_repair = f_right[j] + self._atom[j]
            f0m, f0x = fm[c], fx[c]
            v0m = vm[c] + held_repair * reach - f_right[j] * instant
            v0x = vx[c] + held_repair * repair_x
            cm = f0m + held_life * life_m[0]
            cx = f0x + held_life * life_x[0]
            for _ in range(_ROUNDS):
                at = _position(cm, cx, w)
                left, right = cm * (1.0 - at), cm * at
                um = v0m + left * reach + right * instant
                ux = v0x + left * repair_x + right * instant * w
                back = um * (1.0 - _position(um, ux, w))
                cm = f0m + (held_life + back) * life_m[0]
                cx = f0x + (held_life + back) * life_x[0]
            fm[c], fx[c], vm[c], vx[c] = cm, cx, um, ux
            at = _position(cm, cx, w)
            f_left[c], f_right[c] = cm * (1.0 - at), cm * at
            at = _position(um, ux, w)
            v_left[c], v_right[c] = um * (1.0 - at), um * at
            if self.demands is not None:
                self._demand_chances(c)
            if c < n:
                # Point j's units and failures, into the cells after c.
                source = held_life + v_left[c]
                fm[c + 1 :] += source * life_m[1:]
                fx[c + 1 :] += source * life_x[1:]
                source = held_repair + f_left[c]
                vm[c + 1 :] += source * repair_m[1:]
                vx[c + 1 :] += source * repair_m_x[1:]
        # The point availability on the grid, without the demands'
        # failures (kept as dips).
        self.smooth = np.ones(n + 1)
        for i in range(1, n + 1):
            self.smooth[i] = 1.0 - self._down_at(i)
        self._running = np.concatenate([[0.0], np.cumsum(fm[1:])])

    # -- the curve ----------------------------------------------------------

    def _dips(self, x: np.ndarray) -> np.ndarray:
        """The chance of being in the repair of a demand's failure."""
        out = np.zeros(x.shape)
        if self.demands is None:
            return out
        for at, chance in zip(self.demands, self.chances):
            after = x >= at
            if chance and after.any():
                out[after] += chance * self._repair_sf(x[after] - at)
        return out

    def down_at(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        smooth = np.interp(x, self.t, self.smooth)
        return np.clip(1.0 - smooth + self._dips(x), 0.0, 1.0)

    def at(self, x: np.ndarray) -> np.ndarray:
        return 1.0 - self.down_at(x)

    def failed_at(self, x: np.ndarray) -> np.ndarray:
        return self.down_at(x)

    def derivative(self, x: np.ndarray, scale: float) -> np.ndarray:
        """Its rate of change: the grid's slope after ``x``, and each
        demand's repairs' density after it."""
        x = np.asarray(x, dtype=float)
        t = self.t
        k = np.clip(np.searchsorted(t, x, side="right") - 1, 0, t.size - 2)
        out = (self.smooth[k + 1] - self.smooth[k]) / (t[k + 1] - t[k])
        out = np.where(x >= t[-1], 0.0, out)
        if self.demands is not None:
            for at, chance in zip(self.demands, self.chances):
                after = x >= at
                if chance and after.any():
                    out[after] += chance * _density(
                        self._repair_sf, x[after] - at, self._repair_knots
                    )
        return out

    def knots(self, start: float, stop: float) -> np.ndarray:
        """The grid's points in ``[start, stop]``, and each demand's
        repairs' quantiles after it."""
        parts: list = [self.t, np.array([start, stop], dtype=float)]
        if self.demands is not None:
            parts.append(
                (
                    self.demands[:, None]
                    + np.append(0.0, self._repair_knots)[None, :]
                ).ravel()
            )
        out = np.unique(np.concatenate(parts))
        return out[(out >= start) & (out <= stop)]

    def breaks(self, start: float, stop: float) -> np.ndarray:
        return self.knots(start, stop)

    def grids(self) -> list:
        """None: its grid is not uniform (its knots are all breaks)."""
        return []

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The expected failures before each time ``x`` (in ``[0, x)``),
        each a repair: the life's, from each cell's linear density, and
        the demands' before ``x``."""
        x = np.asarray(x, dtype=float)
        shape = x.shape
        x = x.ravel()
        t = self.t
        c = np.clip(np.searchsorted(t, x, side="left"), 1, t.size - 1)
        a, w = t[c - 1], t[c] - t[c - 1]
        u = np.clip((x - a) / w, 0.0, 1.0)
        mass, moment = self._fm[c], self._fx[c]
        with np.errstate(divide="ignore", invalid="ignore"):
            at = np.where(mass > 0.0, moment / (mass * w), 0.5)
        tilt = 12.0 * (np.clip(at, 0.0, 1.0) - 0.5)
        out = self._running[c - 1] + mass * (u + 0.5 * tilt * (u * u - u))
        out = np.where(x <= 0.0, 0.0, out)
        out = np.where(x > t[-1], self._running[-1], out)
        if self.demands is not None:
            running = np.concatenate(([0.0], np.cumsum(self.chances)))
            out = out + running[np.searchsorted(self.demands, x, "left")]
        return _events(out.reshape(shape))

    def atoms(self, stop: float) -> Atoms:
        """The demands' failures before ``stop``, each taking it down with
        the chance its repair takes time."""
        if self.demands is None:
            return Atoms.none()
        keep = self.demands < stop
        times, chances = self.demands[keep], self.chances[keep]
        zero = np.zeros(times.size)
        takes_time = float(self._repair_sf(np.zeros(1))[0])
        return Atoms(times, chances, zero, zero, chances * takes_time)


def curve(
    life, repair, horizon: float, age: float = 0.0
) -> PlannedDemandsCurve:
    """The ``PlannedDemandsCurve`` of a unit with ``life`` (a
    ``LifeWithPlannedDemands``) repaired as new in a time with model
    ``repair``, new (or up at ``age``) at 0, to ``horizon``: on a grid of
    ``STEPS_PER_SCALE`` steps over the life's typical up time (on a plan,
    or the mean time to a demand's failure at its highest rate, if less),
    and at least ``MIN_STEPS`` over the horizon, up to ``MAX_STEPS``;
    refused when that is coarser than ``COARSEST`` steps over the typical
    up time."""
    from ._curves import _up_scale

    scale = _up_scale(life.life, None)
    if life.times is None and life.p > 0.0:
        fastest = float(np.max(life._expected._shares)) * life.p
        if fastest > 0.0:
            scale = (
                min(scale, 1.0 / fastest)
                if np.isfinite(scale)
                else (1.0 / fastest)
            )
    end = float(horizon)
    if not end > 0.0:
        end = scale if np.isfinite(scale) and scale > 0.0 else 1.0
    step = end / MIN_STEPS
    if np.isfinite(scale) and scale > 0.0:
        step = min(step, scale / STEPS_PER_SCALE)
    if end / step > MAX_STEPS:
        step = end / MAX_STEPS
        if step > scale / COARSEST:
            raise NotImplementedError(
                f"Component {life.node!r} {life.lead}: its values over a "
                f"window to {end:g} would need more than {MAX_STEPS} grid "
                f"points (its typical up time is about {scale:.3g}): "
                "estimate them by simulation, with availability(), or ask "
                "for a shorter window."
            )
    with np.errstate(divide="ignore", invalid="ignore"):
        return PlannedDemandsCurve(life, repair, end, step, age)
