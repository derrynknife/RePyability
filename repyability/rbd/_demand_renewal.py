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
forward in time, cell by cell. (A plain unit's ``unit_curve`` inverts a
power series, which needs the kernel to be one; this one is not.)

The grid's points are a uniform step's (the *lattice*), with the demands,
the plan's times and the times a repair of fixed length after the start
or a demand ends added, so that no cell holds a demand or a change of the
plan: a unit put in within a cell meets the demands at and after its end,
and none at its start. The step is chosen, where it can be, so that all
of those times are lattice points (a whole number of steps into a common
step of theirs, as weekly demands are); a time within a millionth of a
step of a lattice point is taken as on it. In each cell the renewals (the
repairs ending) and the failures of the life (on a plan, of both modes)
are kept as their mass and mean, and each is taken, for what follows, as
its share at the cell's two ends (keeping its mass and mean, as
``unit_curve``'s lattice does), counted with the demands of its cell. A
cell's failures from the units put in before it, and its renewals from
the failures before it, are the life's and the repair's distributions
integrated exactly over the cell (4-point Gauss-Legendre, split at their
quantiles); the failures and renewals within one cell, which feed each
other, are found by a few rounds of substitution (each round's change is
the life's and the repair's chances within a cell, multiplied). A share
at a point stands for its hat (the mass on either side of the point,
linearly down to none a step away), and a fixed repair time carries it
so (``_share_sf``): put wholly at one point, every share's renewals
would fall before or after a demand there, an error first order in the
step.

The demands' survival between a unit's start ``x`` and a cell ``(a, b]``
is a ratio ``G(a) / G(x)`` (``G(t) = (1 - p)^N(t)``, or ``exp(-p Λ(t))``),
times, on a plan, the demands within the cell (exact for the mass, linear
across the cell for the mean). So from a lattice point to a cell of the
lattice, what the life and the repair carry depends on the number of
steps between alone: one table each, and the sums over the units put in
before are convolutions, by FFT in blocks of cells (each block from the
points complete before it; within the block, each point as it is
complete), with every ratio taken against one reference in the block,
so that none overflows. What reaches or leaves a point off the lattice is
integrated for it, as is the first unit. The tables end where the life
(on a plan, with its least demand rate) and the repair have less than
``_TAIL`` left. The cost is ``O(n log n)`` for ``n`` lattice cells, and
the work of each point off the lattice over the reaches.

At set times each demand fails the unit with probability ``p`` if it is
up then: ``p`` less ``p`` times the chance that it is in a repair (one
that ends at the demand is not called on), from the failures before it,
which are the earlier demands' (exactly, as ``_demands.failure_chances``
counts them) and the life's in the cells before it (their linear
densities against the repair's survival). Those failures are atoms at
the demands, kept off the grid (a ``GridCurve``'s dips): their repairs
are exact at any time, however short, and are pushed into the cells
after them exactly, a repair that ends at a point in service after the
demands there (``_push_atom``). With a life that never fails,
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
(the end of a fixed repair) between the grid's points. A window that
would take more than ``MAX_STEPS`` steps takes that many, down to
``COARSEST`` over the typical up time (about ``(1 / COARSEST)^2 / 8``,
2e-5 per pass); past that it is refused.
"""

import math
from typing import Callable, Dict, Tuple

import numpy as np

from ..utils.vectors import dot
from ._common import _sf_values
from ._point_availability import Atoms, _atoms, _density, _events, knots

#: Grid steps over the life's typical up time (``_curves._up_scale``), and
#: at least over the horizon; at most this many points in all, and no
#: coarser than ``COARSEST`` steps over the typical up time (else the
#: values are refused, and the simulations estimate them).
STEPS_PER_SCALE = 400
MIN_STEPS = 1024
MAX_STEPS = 2**17
COARSEST = 50

#: Rounds of substitution for the failures and renewals within one cell.
_ROUNDS = 4
#: What the life and the repair may have left past the end of their tables.
_TAIL = 1e-17
#: A time this share of a step from a lattice point is on it.
_SNAP = 1e-6

_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)


def _position(mass: float, moment: float, width: float) -> float:
    """Where in its cell (0 to 1) a mass with ``moment`` about the cell's
    start has its mean (the middle for no mass)."""
    if not mass > 0.0 or not width > 0.0:
        return 0.5
    return min(max(moment / (mass * width), 0.0), 1.0)


def _pieces(sf: Callable, lo: np.ndarray, hi: np.ndarray, splits):
    """For each interval ``(lo, hi]`` of the time with survival ``sf``:
    ``S(lo)``, ``S(hi)``, ``∫ S`` and ``∫ (v - lo) S(v) dv`` over it
    (4-point Gauss-Legendre, split at ``splits``, sorted)."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    n = lo.size
    if not n:
        empty = np.empty(0)
        return empty, empty, empty, empty
    first = np.searchsorted(splits, lo, side="right")
    count = np.maximum(np.searchsorted(splits, hi, side="left") - first, 0)
    sizes = count + 1
    owner = np.repeat(np.arange(n), sizes)
    within = np.arange(owner.size) - np.repeat(np.cumsum(sizes) - sizes, sizes)
    left, right = lo[owner], hi[owner]
    if splits.size:
        k = first[owner] + within
        top = splits.size - 1
        left = np.where(within > 0, splits[np.clip(k - 1, 0, top)], left)
        right = np.where(
            within < count[owner], splits[np.clip(k, 0, top)], right
        )
    middle, half = 0.5 * (left + right), 0.5 * (right - left)
    x = middle[:, None] + half[:, None] * _GL_X
    f = sf(x.ravel()).reshape(x.shape)
    whole = (f @ _GL_W) * half
    tilted = ((f * (x - lo[owner][:, None])) @ _GL_W) * half
    return (
        sf(lo),
        sf(hi),
        np.bincount(owner, weights=whole, minlength=n),
        np.bincount(owner, weights=tilted, minlength=n),
    )


def _kernel(pieces, lo, hi) -> Tuple[np.ndarray, ...]:
    """A distribution's parts in each cell ``(lo, hi]`` after its start
    from its ``_pieces``: the mass ``dS``, ``S(hi)``, the moment about the
    cell's start ``M``, and ``N`` (the moment's part that a factor linear
    across the cell takes; see ``_carried``)."""
    s_lo, s_hi, whole, tilted = pieces
    width = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(width > 0.0, tilted / width, 0.0)
    return s_lo - s_hi, s_hi, whole - width * s_hi, per - width * s_hi


def _carried(parts, keep, rest) -> Tuple[np.ndarray, np.ndarray]:
    """The mass and moment of what a distribution carries into each cell
    with ``parts`` (``_kernel``'s), times ``keep``, the demands' survival
    from its start to the cell's start, and with ``rest`` (one less the
    demands' survival across the cell; 0 at set times) failing by them
    too: ``keep (dS + rest S(hi))`` and ``keep (M - rest N)``."""
    drop, end, moment, tilt = parts
    return keep * (drop + rest * end), keep * (moment - rest * tilt)


def _block_size(reach: int) -> int:
    """Lattice cells to a block (see ``PlannedDemandsCurve._start_block``)
    for tables of ``reach`` steps: about the square root of what a block's
    FFT costs, so that the FFTs (one a block, of twice the reach) and the
    pushes within the blocks (one a point, of a block) cost alike. It
    changes how the sums are taken, not what they are."""
    return max(64, min(4096, int(2.0 * math.sqrt(max(reach, 1)))))


def _common_step(times: np.ndarray, tolerance: float) -> float:
    """The greatest step of which every one of ``times`` is a whole
    multiple to ``tolerance`` (nearest-integer Euclid), or 0 for none."""
    common = 0.0
    for t in times:
        a, b = max(common, float(t)), min(common, float(t))
        while b > tolerance:
            a, b = b, abs(a - round(a / b) * b)
        common = a
    return common


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
        self.demands = life.times
        self.t = self._grid(float(horizon), float(step))
        # A share at a point stands for its hat (see ``_share_sf``).
        self._hats = [(at, w) for at, w in self._exact if at >= self.h]
        hat = [at + d * self.h for at, _ in self._hats for d in (-1, 0, 1)]
        self._share_knots = np.unique(
            np.concatenate([self._repair_knots, np.asarray(hat, float)])
        )
        self._clocks()
        self._tables()
        self._solve()

    # -- the models -------------------------------------------------------

    def _repair_sf(self, s: np.ndarray) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        out = _sf_values(self.repair.sf, np.maximum(s, 0.0))
        return np.where(s < 0.0, 1.0, out)

    def _share_sf(self, s: np.ndarray) -> np.ndarray:
        """The repair's survival as a share at a point carries it: a
        share stands for its hat (the mass on either side, down to none a
        step away), so each fixed time ``r`` of the repair (at least a step
        long) is spread as the hat about it, ``1 - (u + h)^2 / 2h^2`` from
        ``u = -h`` to 0 and ``(h - u)^2 / 2h^2`` from 0 to ``h``, ``u = s -
        r``. Without it, every share's renewals would fall on one point,
        before or after a demand there, where its mass is on both sides."""
        s = np.asarray(s, dtype=float)
        out = self._repair_sf(s)
        h = self.h
        for at, w in self._hats:
            u = np.clip((s - at) / h, -1.0, 1.0)
            tri = np.where(
                u < 0.0, 1.0 - 0.5 * (u + 1.0) ** 2, 0.5 * (1.0 - u) ** 2
            )
            out = out + w * (tri - (s < at))
        return out

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

    def _kept(self, gap) -> np.ndarray:
        """The demands' survival across ``gap`` of the clock (``C``)."""
        gap = np.asarray(gap, dtype=float)
        if self.demands is not None:
            return np.power(1.0 - self.p, gap)
        return np.exp(-self.p * gap)

    # -- the grid -------------------------------------------------------------

    def _grid(self, horizon: float, step: float) -> np.ndarray:
        """The lattice's points to ``horizon``, with the demands (or the
        plan's times) and the ends of repairs of fixed length from 0 and
        from each demand in it. The lattice's step is the greatest up to
        ``step`` of which they are all whole multiples, where that is at
        least half of ``step``; else ``horizon`` over a whole number."""
        if self.demands is not None:
            starts = self.demands[self.demands < horizon]
        else:
            starts = self.life._knots[self.life._knots < horizon]
        exact = _atoms(
            lambda s: 1.0 - self._repair_sf(s), self._repair_knots, horizon
        )
        self._exact = exact
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
        common = _common_step(special, _SNAP * step)
        if common >= 0.5 * step:
            h = common / max(math.ceil(common / step - 1e-9), 1)
        else:
            h = horizon / max(math.ceil(horizon / step - 1e-9), 1)
        self.h = h
        top = int(math.floor(horizon / h + _SNAP))
        lattice = np.arange(top + 1) * h
        on = abs(lattice[-1] - horizon) <= _SNAP * h
        times = np.append(special, horizon) if not on else special
        index = np.rint(times / h)
        near = np.abs(times - index * h) <= _SNAP * h
        off = times[~near]
        if on:
            lattice[-1] = horizon
        t = np.concatenate([lattice, off])
        order = np.argsort(t, kind="stable")
        t = t[order]
        lat = np.concatenate([np.arange(top + 1), np.full(off.size, -1)])
        self._lat = lat[order]
        self._top = top
        return t

    def _clocks(self) -> None:
        """Each point's lattice index (-1 off it), the lattice cell each
        cell is in, whether it is one (regular), each point's clock ``C``
        (after its demands) and ``C^-`` (before them), and the demands'
        points."""
        t, lat = self.t, self._lat
        n = t.size - 1
        last = np.maximum.accumulate(np.where(lat >= 0, lat, 0))
        self._cellk = np.zeros(n + 1, dtype=int)
        self._cellk[1:] = last[:-1] + 1
        self._regular = np.zeros(n + 1, dtype=bool)
        self._regular[1:] = (
            (lat[:-1] >= 0) & (lat[1:] >= 0) & (lat[1:] == lat[:-1] + 1)
        )
        self._point = np.full(self._top + 1, -1)
        self._point[lat[lat >= 0]] = np.flatnonzero(lat >= 0)
        self._cell_of = np.full(self._top + 2, -1)
        regular = np.flatnonzero(self._regular)
        self._cell_of[self._cellk[regular]] = regular
        if self.demands is not None:
            # Each demand on the point it is at (or within _SNAP of).
            k = np.searchsorted(t, self.demands, side="left")
            k = np.clip(k, 0, n)
            below = np.clip(k - 1, 0, n)
            nearer = np.abs(t[below] - self.demands) < np.abs(
                t[k] - self.demands
            )
            k = np.where(nearer, below, k)
            far = np.abs(t[k] - self.demands) > _SNAP * self.h
            self._dpoint = np.where(far | (self.demands > t[-1]), n + 1, k)
            self.C = np.searchsorted(
                self._dpoint, np.arange(n + 1), side="right"
            ).astype(float)
            self.C_before = np.searchsorted(
                self._dpoint, np.arange(n + 1), side="left"
            ).astype(float)
            self._rest = np.zeros(n + 1)
            self._at_demand = np.zeros(n + 2, dtype=bool)
            self._at_demand[self._dpoint] = True
        else:
            self._at_demand = np.zeros(n + 2, dtype=bool)
            self.C = self._expected(t)
            self.C_before = self.C
            self._rest = np.zeros(n + 1)
            self._rest[1:] = 1.0 - self._kept(np.diff(self.C))

    # -- the tables -----------------------------------------------------------

    def _tables(self) -> None:
        """What the life and the repair carry from a lattice point into
        each lattice cell after it, by the steps between, to where less
        than ``_TAIL`` is left (``m_life``, ``m_repair`` steps)."""
        h, top = self.h, self._top
        o = np.arange(top + 2, dtype=float)
        lo, hi = (o[:-1]) * h, (o[1:]) * h
        parts = _kernel(
            _pieces(self._life_sf, lo, hi, self._life_knots), lo, hi
        )
        s_lo = parts[1] + parts[0]
        if self.demands is not None:
            alive = s_lo > _TAIL
            moving = (parts[0] != 0.0) | (parts[2] != 0.0)
            reach = alive & moving
        else:
            slowest = float(np.min(self.life._rates))
            reach = s_lo * np.exp(-slowest * lo) > _TAIL
        self.m_life = int(np.flatnonzero(reach)[-1]) + 1 if reach.any() else 0
        # Index o: the cell ((o - 1) h, o h] after the point.
        self._life_table = [
            np.concatenate([[0.0], x[: self.m_life]]) for x in parts
        ]
        rparts = _kernel(
            _pieces(self._share_sf, lo, hi, self._share_knots), lo, hi
        )
        r_lo = rparts[1] + rparts[0]
        reach = r_lo > _TAIL
        self.m_repair = (
            int(np.flatnonzero(reach)[-1]) + 1 if reach.any() else 0
        )
        self._repair_table = [
            np.concatenate([[0.0], x[: self.m_repair]]) for x in rparts[::2]
        ]
        # The down weights: a cell o steps before a point (a linear
        # density in it against the repair's survival).
        pieces = _pieces(self._repair_sf, lo, hi, self._repair_knots)
        self._down_whole = pieces[2][: self.m_repair + 1]
        self._down_tilted = pieces[3][: self.m_repair + 1] / h
        self.block = _block_size(max(self.m_life, self.m_repair))
        self._spectra: Dict[Tuple[int, int], np.ndarray] = {}

    def _convolve(self, which: int, table: np.ndarray, x: np.ndarray, n):
        """The first ``n`` terms of ``x`` convolved with ``table`` (one of
        the tables, its spectrum kept), by FFT."""
        from scipy import fft

        length = fft.next_fast_len(2 * table.size + 1, real=True)
        if x.size + table.size - 1 > length:
            length = fft.next_fast_len(x.size + table.size - 1, real=True)
        key = (which, length)
        spectrum = self._spectra.get(key)
        if spectrum is None:
            spectrum = self._spectra[key] = fft.rfft(table, length)
        out = fft.irfft(fft.rfft(x, length) * spectrum, length)[:n]
        if out.size < n:
            out = np.concatenate([out, np.zeros(n - out.size)])
        return out

    # -- what is carried, off the lattice ---------------------------------

    def _life_cells(self, x: float, cx: float, cells: np.ndarray, sf=None):
        """The life's failures in each of ``cells`` of a unit put in at
        ``x`` (clock ``cx``), counted with the demands from it."""
        t = self.t
        lo, hi = t[cells - 1] - x, t[cells] - x
        splits = self._life_knots
        if sf is not None and self.age:
            splits = splits[splits > self.age] - self.age
        parts = _kernel(_pieces(sf or self._life_sf, lo, hi, splits), lo, hi)
        keep = self._kept(self.C[cells - 1] - cx)
        return _carried(parts, keep, self._rest[cells])

    def _repair_cells(self, x: float, cells: np.ndarray):
        """The repairs ending in each of ``cells`` of a share of failures
        at ``x``."""
        t = self.t
        lo, hi = t[cells - 1] - x, t[cells] - x
        drop, _, moment, _ = _kernel(
            _pieces(self._share_sf, lo, hi, self._share_knots), lo, hi
        )
        return drop, moment

    def _push_atom(self, i: int) -> None:
        """The repairs of the demands' failures at point ``i`` (exact
        times), ending in each cell after it: a repair that ends at a point
        is in service after the demands there (as one that ends at a
        demand is not called on), so each cell is taken ``_SNAP`` of a
        step early, and the repairs in no time are in the next cell's
        start."""
        t, n = self.t, self.t.size - 1
        if i >= n or not self._atom[i]:
            return
        end = int(np.searchsorted(t, t[i] + (self.m_repair + 1) * self.h))
        cells = np.arange(i + 1, min(end, n) + 1)
        early = _SNAP * self.h
        lo, hi = t[cells - 1] - t[i] - early, t[cells] - t[i] - early
        drop, _, moment, _ = _kernel(
            _pieces(self._repair_sf, lo, hi, self._repair_knots), lo, hi
        )
        self._av_m[cells] += self._atom[i] * drop
        self._av_x[cells] += self._atom[i] * (moment - early * drop)

    def _cells_within(self, c: int, reach: int) -> np.ndarray:
        """Cell ``c`` and those after it starting within ``reach`` steps
        of its start."""
        t = self.t
        end = int(np.searchsorted(t, t[c - 1] + reach * self.h, "left"))
        return np.arange(c, max(min(end, t.size - 1), c) + 1)

    def _lattice_sources(self, c: int, reach: int) -> np.ndarray:
        """The lattice points complete before cell ``c`` (off the
        lattice), within ``reach`` steps of its start."""
        a = self.t[c - 1]
        k = self._cellk[c] - 1  # the lattice point at or before a
        last = k - 1 if self._lat[c - 1] >= 0 else k
        first = max(int(math.ceil(a / self.h - 1e-9)) - reach, 0)
        return np.arange(first, last + 1)

    def _from_lattice(self, c: int) -> Tuple[float, float, float, float]:
        """The failures and renewals in cell ``c`` (off the lattice) from
        the complete lattice points before it."""
        t, h = self.t, self.h
        out = [0.0, 0.0, 0.0, 0.0]
        js = self._lattice_sources(c, self.m_life)
        js = js[self._s_lat[js] != 0.0] if js.size else js
        if js.size:
            x = js * h
            lo, hi = t[c - 1] - x, t[c] - x
            parts = _kernel(
                _pieces(self._life_sf, lo, hi, self._life_knots), lo, hi
            )
            keep = self._s_lat[js] * self._kept(
                self.C[c - 1] - self._c_lat[js]
            )
            mass, moment = _carried(parts, keep, self._rest[c])
            out[0], out[1] = float(np.sum(mass)), float(np.sum(moment))
        js = self._lattice_sources(c, self.m_repair)
        js = js[self._q_lat[js] != 0.0] if js.size else js
        if js.size:
            x = js * h
            lo, hi = t[c - 1] - x, t[c] - x
            drop, _, moment, _ = _kernel(
                _pieces(self._share_sf, lo, hi, self._share_knots), lo, hi
            )
            out[2] = dot(self._q_lat[js], drop)
            out[3] = dot(self._q_lat[js], moment)
        return out[0], out[1], out[2], out[3]

    # -- the chance of being in a life failure's repair --------------------

    def _density_parts(self, c) -> Tuple[np.ndarray, np.ndarray]:
        """Cell ``c``'s failures of the life as a linear density: their
        level with half its tilt, and the level times the tilt (``A`` and
        ``B``; a cell ``(a, b]`` adds ``A ∫ S_r - B ∫ (v - t + b) S_r / w``
        to the chance of being in a repair at ``t``)."""
        c = np.atleast_1d(c)
        mass, moment = self._fm[c], self._fx[c]
        width = self.t[c] - self.t[c - 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            at = np.where(mass > 0.0, moment / (mass * width), 0.5)
            level = np.where(width > 0.0, mass / width, 0.0)
        tilt = 12.0 * (np.clip(at, 0.0, 1.0) - 0.5)
        return level * (1.0 + 0.5 * tilt), level * tilt

    def _down_off(self, c: int) -> None:
        """Cell ``c`` (off the lattice): its failures' repairs under way
        at each point from its end within the repair's reach."""
        t = self.t
        if not self._fm[c] != 0.0:
            return
        a, b = t[c - 1], t[c]
        end = int(np.searchsorted(t, b + self.m_repair * self.h, "left"))
        points = np.arange(c, min(end, t.size - 1) + 1)
        lo, hi = t[points] - b, t[points] - a
        _, _, whole, tilted = _pieces(
            self._repair_sf, lo, hi, self._repair_knots
        )
        A, B = self._density_parts(c)
        self._down_acc[points] += A[0] * whole - B[0] * tilted / (b - a)

    def _down_lattice(self, i: int) -> float:
        """At point ``i``, the repairs under way of the life's failures in
        the lattice's cells (those off it are in ``_down_acc``)."""
        t, h = self.t, self.h
        if self.m_repair == 0:
            return 0.0
        k = self._lat[i]
        if k >= 0:
            lo = max(k - self.m_repair, 1)
            if lo > k:
                return 0.0
            ks = np.arange(lo, k + 1)
            A, B = self._lattice_density(ks)
            o = k - ks
            return dot(A, self._down_whole[o]) - dot(B, self._down_tilted[o])
        top = self._cellk[i] - 1  # the last lattice cell before t[i]
        lo = max(int(math.floor((t[i] - self.m_repair * h) / h)), 1)
        if lo > top:
            return 0.0
        ks = np.arange(lo, top + 1)
        A, B = self._lattice_density(ks)
        keep = (A != 0.0) | (B != 0.0)
        ks, A, B = ks[keep], A[keep], B[keep]
        if not ks.size:
            return 0.0
        lo_v, hi_v = t[i] - ks * h, t[i] - (ks - 1) * h
        _, _, whole, tilted = _pieces(
            self._repair_sf, lo_v, hi_v, self._repair_knots
        )
        return dot(A, whole) - dot(B, tilted) / h

    def _lattice_density(self, ks: np.ndarray):
        """``_density_parts`` of the lattice cells ``ks`` (0 for those
        split by a point off the lattice, which ``_down_off`` takes)."""
        cells = self._cell_of[ks]
        A, B = np.zeros(ks.size), np.zeros(ks.size)
        whole = cells >= 0
        if whole.any():
            A[whole], B[whole] = self._density_parts(cells[whole])
        return A, B

    def _demand_chances(self, i: int) -> None:
        """The failures at the demands at point ``i``: each fails the unit
        with ``p`` unless it is in a repair then (see
        ``_demands.failure_chances``)."""
        from ._demands import _under_way

        times, p = self.demands, self.p
        assert times is not None
        first = int(np.searchsorted(self._dpoint, i, side="left"))
        last = int(np.searchsorted(self._dpoint, i, side="right"))
        if first == last:
            return
        life = self._down_acc[i] + self._down_lattice(i)
        reach = self.m_repair * self.h
        for k in range(first, last):
            if k:
                earliest = int(
                    np.searchsorted(times[:k], times[k] - reach, "left")
                )
                under = _under_way(self.repair, times[earliest:k], times[k])
                busy = math.fsum(self.chances[earliest:k] * under)
            else:
                busy = 0.0
            self.chances[k] = p * (1.0 - (busy + life))
        self._atom[i] = float(np.sum(self.chances[first:last]))
        self._push_atom(i)

    # -- the renewal equation, cell by cell -----------------------------------

    def _start_block(self, k0: int) -> None:
        """The lattice cells from ``k0`` (a block): what the lattice points
        complete before it (to ``k0 - 2``) carry into them, by FFT."""
        top = self._top
        k1 = min(k0 + self.block, top + 1)
        self._block_end = k1
        jhi = k0 - 2
        if jhi < 0:
            return
        ks = np.arange(k0, k1)
        n = k1 - k0
        if self.m_life:
            jlo = max(k0 - self.m_life, 0)
            ref = self._c_lat[jhi]
            weights = self._s_lat[jlo : jhi + 1] * self._kept(
                ref - self._c_lat[jlo : jhi + 1]
            )
            if np.any(weights != 0.0):
                keep = self._kept(self._c_lat[ks - 1] - ref)
                rest = self._rest_lat[ks]
                # At set times no demand is within a cell (no ``rest``).
                used = (0, 2) if self.demands is not None else (0, 1, 2, 3)
                got = [
                    (
                        self._convolve(w, table, weights, k1 - jlo)[k0 - jlo :]
                        if w in used
                        else 0.0
                    )
                    for w, table in enumerate(self._life_table)
                ]
                mass, moment = _carried(got, keep, rest)
                self._lf_m[k0:k1] += mass[:n]
                self._lf_x[k0:k1] += moment[:n]
        if self.m_repair:
            jlo = max(k0 - self.m_repair, 0)
            weights = self._q_lat[jlo : jhi + 1]
            if np.any(weights != 0.0):
                for w, (table, out) in enumerate(
                    zip(self._repair_table, (self._lv_m, self._lv_x))
                ):
                    got = self._convolve(4 + w, table, weights, k1 - jlo)
                    out[k0:k1] += got[k0 - jlo :][:n]

    def _push_lattice(self, j: int) -> None:
        """Lattice point ``j``, complete: what it carries into the cells of
        the block from two steps after it."""
        k1 = self._block_end
        s, q = self._s_lat[j], self._q_lat[j]
        lo = j + 2
        if s != 0.0 and self.m_life:
            hi = min(k1, j + self.m_life + 1)
            if lo < hi:
                o = slice(lo - j, hi - j)
                parts = [table[o] for table in self._life_table]
                keep = s * self._kept(
                    self._c_lat[lo - 1 : hi - 1] - self._c_lat[j]
                )
                mass, moment = _carried(parts, keep, self._rest_lat[lo:hi])
                self._lf_m[lo:hi] += mass
                self._lf_x[lo:hi] += moment
        if q != 0.0 and self.m_repair:
            hi = min(k1, j + self.m_repair + 1)
            if lo < hi:
                o = slice(lo - j, hi - j)
                self._lv_m[lo:hi] += q * self._repair_table[0][o]
                self._lv_x[lo:hi] += q * self._repair_table[1][o]

    def _solve(self) -> None:
        t = self.t
        n = t.size - 1
        top = self._top
        widths = np.diff(t)
        p = self.p
        if self.demands is not None:
            self.chances = np.zeros(self.demands.size)
        # Each cell's failures of the life and renewals: mass and moment
        # about its start (index c for the cell (t[c-1], t[c]]).
        fm, fx = np.zeros(n + 1), np.zeros(n + 1)
        vm, vx = np.zeros(n + 1), np.zeros(n + 1)
        self._fm, self._fx = fm, fx
        self._atom = np.zeros(n + 1)
        self._down_acc = np.zeros(n + 1)
        # What the points off the lattice (and the first unit) carry into
        # each cell; what the lattice's points carry into its cells.
        af_m, af_x = np.zeros(n + 1), np.zeros(n + 1)
        av_m, av_x = np.zeros(n + 1), np.zeros(n + 1)
        self._av_m, self._av_x = av_m, av_x
        self._lf_m, self._lf_x = np.zeros(top + 2), np.zeros(top + 2)
        self._lv_m, self._lv_x = np.zeros(top + 2), np.zeros(top + 2)
        # Each lattice point's units put in (``s``, after its demands) and
        # failures (``q``); its clock; the down parts of its cells.
        self._s_lat, self._q_lat = np.zeros(top + 1), np.zeros(top + 1)
        self._c_lat = self.C[self._point]
        self._rest_lat = np.zeros(top + 2)
        cells = self._cell_of >= 0
        self._rest_lat[cells] = self._rest[self._cell_of[cells]]
        # Their shares at each cell's ends.
        f_right, v_right = np.zeros(n + 1), np.zeros(n + 1)
        f_left, v_left = np.zeros(n + 1), np.zeros(n + 1)
        # The first unit, from 0.
        everything = np.arange(1, n + 1)
        mass, moment = self._life_cells(
            0.0, self.C[0], everything, sf=self._first_sf
        )
        af_m[1:] += mass
        af_x[1:] += moment
        instant = 1.0 - float(self._repair_sf(np.zeros(1))[0])
        self._block_end = 0
        life_table, repair_table = self._life_table, self._repair_table
        for c in range(1, n + 1):
            j = c - 1
            w = widths[j]
            k = int(self._cellk[c])
            if k >= self._block_end:
                self._start_block(k)
            regular = bool(self._regular[c])
            off = self._lat[j] < 0
            # A unit put in at point j, and a failure there: their parts in
            # cell c (the first), and, off the lattice, in the cells after.
            if off:
                after = self._cells_within(c, max(self.m_life, 1))
                life_m, life_x = self._life_cells(t[j], self.C[j], after)
                rafter = self._cells_within(c, max(self.m_repair, 1))
                repair_m, repair_m_x = self._repair_cells(t[j], rafter)
                life0, lifex0 = float(life_m[0]), float(life_x[0])
                rep0, repx0 = float(repair_m[0]), float(repair_m_x[0])
            elif regular and self.m_life:
                rest = self._rest[c]
                life0 = float(life_table[0][1] + rest * life_table[1][1])
                lifex0 = float(life_table[2][1] - rest * life_table[3][1])
            else:
                life_m, life_x = self._life_cells(
                    t[j], self.C[j], np.array([c])
                )
                life0, lifex0 = float(life_m[0]), float(life_x[0])
            if not off:
                if regular and self.m_repair:
                    rep0 = float(repair_table[0][1])
                    repx0 = float(repair_table[1][1])
                elif regular:
                    rep0 = repx0 = 0.0
                else:
                    rm, rx = self._repair_cells(t[j], np.array([c]))
                    rep0, repx0 = float(rm[0]), float(rx[0])
            reach = rep0 + instant
            repair_x = repx0
            # What point j holds already: the renewals in cell j at its
            # end (which met the demands there), and the share of failures
            # at it (the demands' are pushed, exactly, in _push_atom).
            held_life = v_right[j]
            if self.demands is not None and j:
                held_life *= (1.0 - p) ** (self.C[j] - self.C_before[j])
            held_repair = f_right[j]
            # What the points complete before it carry into cell c.
            if regular:
                f0m = af_m[c] + self._lf_m[k]
                f0x = af_x[c] + self._lf_x[k]
                g0m = av_m[c] + self._lv_m[k]
                g0x = av_x[c] + self._lv_x[k]
            else:
                lm, lx, rm_, rx_ = self._from_lattice(c)
                f0m, f0x = af_m[c] + lm, af_x[c] + lx
                g0m, g0x = av_m[c] + rm_, av_x[c] + rx_
            v0m = g0m + held_repair * (reach - instant)
            v0x = g0x + held_repair * repair_x
            cm = f0m + held_life * life0
            cx = f0x + held_life * lifex0
            for _ in range(_ROUNDS):
                at = _position(cm, cx, w)
                left, right = cm * (1.0 - at), cm * at
                um = v0m + left * reach + right * instant
                ux = v0x + left * repair_x + right * instant * w
                back = um * (1.0 - _position(um, ux, w))
                cm = f0m + (held_life + back) * life0
                cx = f0x + (held_life + back) * lifex0
            fm[c], fx[c], vm[c], vx[c] = cm, cx, um, ux
            at = _position(cm, cx, w)
            f_left[c], f_right[c] = cm * (1.0 - at), cm * at
            at = _position(um, ux, w)
            v_left[c], v_right[c] = um * (1.0 - at), um * at
            # Its failures' repairs under way after it.
            if not regular:
                self._down_off(c)
            if self._at_demand[c]:
                self._demand_chances(c)
            # Point j, complete: into the cells after c.
            source = held_life + v_left[c]
            failed = held_repair + f_left[c]
            if off:
                rest_cells = after[1:]
                af_m[rest_cells] += source * life_m[1:]
                af_x[rest_cells] += source * life_x[1:]
                rest_cells = rafter[1:]
                av_m[rest_cells] += failed * repair_m[1:]
                av_x[rest_cells] += failed * repair_m_x[1:]
            else:
                lj = int(self._lat[j])
                self._s_lat[lj], self._q_lat[lj] = source, failed
                self._push_lattice(lj)
        # The point availability on the grid, without the demands'
        # failures (kept as dips).
        down = self._down_acc.copy()
        on = np.flatnonzero(self._lat >= 0)
        if self.m_repair and on.size:
            m = top + 1
            A, B = self._lattice_density(np.arange(m))
            whole = self._convolve(6, self._down_whole, A, m)
            tilted = self._convolve(7, self._down_tilted, B, m)
            down[on] += (whole - tilted)[self._lat[on]]
        for i in np.flatnonzero(self._lat < 0):
            down[i] += self._down_lattice(int(i))
        self.smooth = 1.0 - down
        self.smooth[0] = 1.0
        self._running = np.concatenate([[0.0], np.cumsum(fm[1:])])

    # -- the curve ----------------------------------------------------------

    def _dips(self, x: np.ndarray) -> np.ndarray:
        """The chance of being in the repair of a demand's failure (each
        to where less than ``_TAIL`` of it is left)."""
        out = np.zeros(x.shape)
        if self.demands is None:
            return out
        flat = x.ravel()
        order = np.argsort(flat, kind="stable")
        ordered = flat[order]
        found = np.zeros(flat.size)
        reach = self.m_repair * self.h
        for at, chance in zip(self.demands, self.chances):
            if not chance:
                continue
            lo = int(np.searchsorted(ordered, at, "left"))
            hi = int(np.searchsorted(ordered, at + reach, "right"))
            if lo < hi:
                found[lo:hi] += chance * self._repair_sf(ordered[lo:hi] - at)
        out.ravel()[order] = found
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
