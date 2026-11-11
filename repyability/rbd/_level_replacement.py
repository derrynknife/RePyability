"""Numerical values of a component replaced on condition by its measured
degradation level (#271).

A unit whose life is a degradation process (``_degradation``) is inspected
at every multiple of the interval ``T`` while it works, and replaced, as
new, when the level the inspection measures is at or past the replacement
level. Its failures (the level reaching the process's failure threshold)
are revealed and repaired as usual, and a new unit starts at the process's
starting level ``y0``. These are the simulation's rules
(``_event_loop._condition_next``): an inspection that falls while the unit
is down is not done, and a replacement that takes time takes the unit down
for it.

Between inspections the level is a Markov process: all that the past says
about a working unit's future is its level now. So the unit is followed
from one inspection to the next as replacement on condition by age
(``_condition_replacement``) follows it, with the level in place of the
age. What it carries from one inspection to the next is the distribution
of the level of a unit still working, on a grid of levels below the
replacement level, and the repairs and replacements going on. Within an
interval:

- a unit kept at level ``y`` is up a time ``s`` later with the process's
  first-passage survival from ``y``; its failure starts a repair, and a
  new unit after it;
- the units put into service in the interval (after the replacement at
  its start, or after a repair) are an alternating renewal process of
  lives from ``y0`` and repairs, as under block replacement
  (``_block_replacement``);
- at the inspection that ends it, a kept unit's level has moved by the
  process's law given that it has not failed (``_degradation.level_below``,
  the distribution the simulation draws from); a unit put into service
  ``a`` before it is at a level drawn from ``y0`` over ``a``. Those at or
  past the replacement level are replaced (the next interval starts with
  their replacement), the rest carry on.

Over time, the map is followed from new over the intervals, giving the
unit's availability, failures, replacements and inspections interval by
interval (a ``BlockAvailability``, as replacement on condition by age gives
it), until it repeats from one interval to the next. That repeating
interval is the long run: the unit's state at the inspections has reached
the map's stationary distribution, and its values per interval are its
long-run values (renewal-reward over one interval of the stationary
process).

The level grid: cells of equal width from a lowest level up to the
replacement level, the mass at or past it kept as one more cell (it is
replaced). A gamma process never goes below ``y0``; a Wiener process with
drift ``mu`` and spread ``sigma`` goes no lower than ``y0 - (k sigma)^2 /
(4 mu)`` but with a chance below that of a standard normal beyond ``k``
(the deepest point of ``mu t - k sigma sqrt(t)``), and what goes lower is
kept in the lowest cell. The cells are sized from the model: a fixed
number per spread of the level's change over one interval, never in
absolute units. Within a cell the level is taken as spread evenly, so each
transfer is the process's law averaged over the cell (Gauss-Legendre
points), and the mass of each cell it lands in is exact (differences of
``level_below``). The error falls as the square of the cells' width:
at 32 cells a spread, about 1e-6 in the availability and 1e-5 (relative)
in the rates of failure and replacement, and up to about 1e-4 in the
availability just after an inspection when the replacement level is close
to the failure threshold (the units kept just short of it fail at once,
within the cell). The grid over an interval adds its own error, as for
replacement on condition by age (``_block_replacement._grid``).

From new only: the first interval starts with the unit put into service
at ``y0``. Started from a measured level, the first interval would start
instead with that level's mass in ``kept`` (and the time to the first
inspection as a shorter first interval), the rest as it is.
"""

import math
from typing import NamedTuple, Optional, Tuple

import numpy as np

from ._block_replacement import (
    _MAX_VALUES,
    _SETTLED,
    _UNSUPPORTED,
    BlockAvailability,
    _carry,
    _Duration,
    _grid,
    _Grid,
    _masses,
    _steps,
)
from ._condition_replacement import ConditionCycle
from ._degradation import _kinds, level_below

#: Cells of the level grid per spread (standard deviation) of the level's
#: change over one inspection interval; at least, and at most, so many.
_PER_SPREAD = 32
_MIN_CELLS, _MAX_CELLS = 64, 2048
#: How far below its start a Wiener level is followed: the standard
#: normal's quantile whose tail is left in the lowest cell.
_DEPTH = 8.0
#: Gauss-Legendre points a cell's level is averaged over.
_POINTS = 2
#: The most values the survival from each cell over the interval's grid
#: may take (cells times grid points).
_MAX_SURVIVAL = 2**23

_WHY = (
    "Component {node!r} is replaced on condition by its measured "
    "degradation level, "
)


def _spread(life, interval: float) -> Tuple[float, float]:
    """The spread (standard deviation) of the level's change over one
    inspection interval, and the lowest level the grid follows."""
    wiener, _ = _kinds()
    a, b = (float(p) for p in life.params)
    y0 = float(life.y0)
    if isinstance(life, wiener):
        # The deepest point of mu t - k sigma sqrt(t), at any t.
        return b * math.sqrt(interval), y0 - (_DEPTH * b) ** 2 / (4.0 * a)
    return math.sqrt(a * interval) / b, y0


def _edges(life, level: float, interval: float, node) -> np.ndarray:
    """The edges of the level grid's cells below the replacement level,
    the last the level itself (no cell if no working unit can be below
    it at an inspection), of equal width, ``y0`` at the middle of its
    cell: a unit put into service shortly before an inspection is at
    ``y0`` or near it, and a cell's level is taken as spread evenly
    about its middle. Raises if more than ``_MAX_CELLS`` would be
    needed."""
    spread, lowest = _spread(life, interval)
    y0 = float(life.y0)
    if level <= lowest:
        return np.array([level])
    width = min(spread / _PER_SPREAD, (level - lowest) / _MIN_CELLS)
    if y0 < level:
        # Whole cells and a half from y0 up to the level.
        width = (level - y0) / (math.ceil((level - y0) / width - 0.5) + 0.5)
        lowest = min(lowest, y0 - 0.5 * width)
    cells = math.ceil((level - lowest) / width - 1e-9)
    if cells > _MAX_CELLS:
        raise NotImplementedError(
            _WHY.format(node=node)
            + f"and its level changes so little over an inspection "
            f"interval, against the levels it passes through, that the "
            f"grid of levels would need {cells} cells (at most "
            f"{_MAX_CELLS}). {_UNSUPPORTED}"
        )
    return level - width * np.arange(cells, -1, -1)


def check_models(life, repair, duration, level: float, interval, node):
    """Raise unless the numerical values cover the component (before any
    computation): a Wiener process needs a positive drift (else some
    units never fail, and the level wanders off without bound), the
    repairs and replacements must be models the grid over an interval
    takes (``_block_replacement._Duration``), and the level grid and the
    grid over an interval must be of a size to follow."""
    wiener, _ = _kinds()
    if isinstance(life, wiener) and not float(life.params[0]) > 0.0:
        raise NotImplementedError(
            _WHY.format(node=node)
            + "and its Wiener process does not drift towards the "
            "threshold (mu <= 0), so some of its units never fail and "
            f"their levels have no long-run spread to follow. {_UNSUPPORTED}"
        )
    _Duration(repair, "repair", node)
    _Duration(duration, "replacement", node)
    cells = len(_edges(life, level, float(interval), node)) - 1
    steps = _steps(life, float(interval))
    if cells * (steps + 1) > _MAX_SURVIVAL:
        raise NotImplementedError(
            _WHY.format(node=node)
            + "and its grid of levels times its grid over an inspection "
            f"interval ({cells} by {steps + 1}) is too large to follow. "
            f"{_UNSUPPORTED}"
        )


class _Levels(NamedTuple):
    """A unit's levels on the grid (see the module docstring): ``edges``,
    the ``n + 1`` edges of the cells below the replacement level (the
    last the level itself); ``kernel`` (``n`` by ``n + 1``), the
    probability that a unit in each cell is unfailed an interval later
    and in each cell, or at or past the replacement level (the last
    column); ``survival`` (``n`` by the interval's grid), that a unit in
    each cell is unfailed by each time of the grid; and ``fresh`` (the
    grid by ``n + 1``), that a unit put into service at ``y0`` is
    unfailed after each time of the grid and in each cell, or at or past
    the level."""

    edges: np.ndarray
    kernel: np.ndarray
    survival: np.ndarray
    fresh: np.ndarray


def _levels(life, level: float, g: _Grid, node) -> _Levels:
    """The level grid of a unit replaced at ``level``, and its transfers
    (see ``_Levels``), from ``level_below``."""
    threshold, y0 = float(life.threshold), float(life.y0)
    T = g.interval
    edges = _edges(life, level, T, node)
    n = len(edges) - 1
    # Each cell's level, evenly spread: Gauss-Legendre points and weights.
    nodes, weights = np.polynomial.legendre.leggauss(_POINTS)
    middle, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * np.diff(edges)
    points = middle[:, None] + half[:, None] * nodes[None, :]
    weights = 0.5 * weights
    # The levels the transfers are counted below: the cells' edges past
    # the lowest (below which the lowest cell keeps the rest), and the
    # threshold (the survival).
    tops = np.concatenate([edges[1:], [threshold]])

    def counted(start, t) -> np.ndarray:
        """The probability of being unfailed at ``t`` and in each cell
        (and past the level), from the levels ``start`` (any shape)."""
        below = level_below(life, start[..., None], t, tops)
        return np.maximum(np.diff(below, axis=-1, prepend=0.0), 0.0)

    kernel = np.einsum("ijk,j->ik", counted(points, T), weights)
    # The survival from each cell over the grid, Gauss point by point
    # (``level_below`` at the threshold), and at 0 surely.
    survival = np.ones((n, g.steps + 1))
    if n:
        times = g.grid[1:]
        survival[:, 1:] = sum(
            w * level_below(life, points[:, j, None], times, threshold)
            for j, w in enumerate(weights)
        )
    # From y0, over each time of the grid: at 0, at y0 itself.
    fresh = np.zeros((g.steps + 1, n + 1))
    fresh[1:] = counted(np.full(g.steps, y0), g.grid[1:, None])
    if y0 >= level or n == 0:
        fresh[0, -1] = 1.0
    else:
        cell = int(np.searchsorted(edges, y0, side="right")) - 1
        fresh[0, min(cell, n - 1)] = 1.0
    return _Levels(edges, kernel, survival, fresh)


class LevelModel:
    """A component replaced on condition by its level, ready to follow:
    the interval's grid (``_block_replacement._grid``) and the level
    grid's transfers (``_Levels``), built once and kept with the diagram
    (see ``level_model``)."""

    def __init__(self, life, repair, duration, interval, level, node=None):
        self.node = node
        check_models(life, repair, duration, level, interval, node)
        self.g = _grid(
            life,
            repair,
            duration,
            interval,
            node,
            starts=True,
            life_checked=False,
        )
        self.levels = _levels(life, float(level), self.g, node)
        self._cycle: Optional[ConditionCycle] = None

    def availability(self, horizon: float) -> BlockAvailability:
        """The unit's point availability from new over ``[0, horizon]``,
        or until it repeats from one inspection interval to the next
        (``inf``: until it does), interval by interval (see the module
        docstring): a ``BlockAvailability`` with ``inspected``, the
        probability that the unit is up at the inspection that starts
        each interval, and ``replaced``, that it is replaced there.

        Raises
        ------
        NotImplementedError
            If the availability has not settled into repeating within the
            intervals the grid can keep.
        """
        from scipy.signal import fftconvolve

        g, lv, node = self.g, self.levels, self.node
        steps, h, T = g.steps, g.h, g.interval
        assert g.renewals is not None and g.done is not None
        most = max(3, _MAX_VALUES // (steps + 1))
        count = (
            int(np.floor(horizon / T)) + 1
            if np.isfinite(horizon)
            else most + 1
        )
        # The units put into service: the one put in, and one after each
        # repair (half of each cell's renewals at each of its ends).
        renewals = np.diff(g.renewals)
        put_in = np.zeros(steps + 1)
        put_in[0] = 1.0
        put_in[:-1] += 0.5 * renewals
        put_in[1:] += 0.5 * renewals
        ending = np.diff(g.done)
        start_P = g.replace.cdf(g.rho)
        start_C = g.rho - g.replace.in_progress(g.rho)
        down = 1.0 - g.A

        # The first interval: put into service new at 0, no inspection.
        P = np.ones(g.length + 1)
        C = g.rho.copy()
        other = np.zeros(g.length + 1)
        replaced, inspected = 1.0, 0.0
        kept = np.zeros(len(lv.edges) - 1)
        rows: list = []
        probabilities: list = []
        failing: list = []
        looked: list = []
        settled = False
        for k in range(count):
            # The units kept at the inspection, by level: up, and their
            # failures in each cell of the interval, whose repairs put new
            # units into service.
            kept_up = kept @ lv.survival
            kept_cells = np.maximum(kept_up[:-1] - kept_up[1:], 0.0)
            if kept_cells.any():
                started = np.zeros(g.length + 1)
                started[1:] = fftconvolve(kept_cells / h, ending)[: g.length]
                started = np.clip(started, 0.0, None)
                P = P + started
                other = other + started
                C = C + np.concatenate(
                    [[0.0], np.cumsum(0.5 * h * (started[1:] + started[:-1]))]
                )
            w = _masses(P, C, steps, h)
            row = (
                other[: steps + 1]
                - fftconvolve(w, down)[: steps + 1]
                + kept_up
            )
            density = np.diff(fftconvolve(w, g.N)[: steps + 1]) / h
            failures = np.concatenate(
                [[0.0], np.cumsum(density * h + kept_cells)]
            )
            if (
                k >= 2
                and abs(replaced - probabilities[-1]) < _SETTLED
                and abs(inspected - looked[-1]) < _SETTLED
                and np.max(np.abs(row - rows[-1])) < _SETTLED
                and np.max(np.abs(failures - failing[-1])) < _SETTLED
            ):
                settled = True
                break
            if len(rows) == most:
                raise NotImplementedError(
                    _WHY.format(node=node)
                    + f"and its availability has not settled after {most} "
                    "inspection intervals, as far as the grid can follow "
                    f"it. {_UNSUPPORTED}"
                )
            rows.append(row)
            probabilities.append(replaced)
            failing.append(failures)
            looked.append(inspected)
            # The units up at the inspection that ends the interval, by
            # level: those put into service in it, from y0 over the time
            # since (scaled to the probability that such a unit is up,
            # so that none is lost), and those kept, an interval on.
            ages = fftconvolve(w, put_in)[steps::-1].copy()  # by age
            since = ages @ lv.fresh
            total, wanted = float(since.sum()), float(w @ g.A[::-1])
            if total > 0.0:
                since *= wanted / total
            by_level = since + kept @ lv.kernel
            other, C, _ = _carry(P, C, density, g)
            # The inspection: those at or past the level are replaced.
            inspected = float(by_level.sum())
            replaced = float(by_level[-1])
            kept = by_level[:-1]
            P = other + replaced * start_P
            C = C + replaced * start_C
        return BlockAvailability(
            T,
            g.grid,
            np.array(rows),
            np.array(probabilities),
            g.replace,
            settled,
            np.array(failing),
            True,
            np.array(looked),
        )

    def cycle(self) -> ConditionCycle:
        """The unit's long-run values: those of the interval its
        availability from new settles into (see the module docstring),
        as a ``ConditionCycle`` over one inspection interval (its
        ``length``), with ``replaced`` the probability that the
        inspection replaces the unit, so that each mean over the
        interval's length is a long-run rate.

        Raises
        ------
        NotImplementedError
            As ``availability`` does.
        """
        if self._cycle is not None:
            return self._cycle
        result = self.availability(np.inf)
        g = self.g
        h, T = g.h, g.interval
        replaced = float(result.replaced[-1])
        assert result.inspected is not None  # inspected at each interval
        before = float(result.inspected[-1])
        # The replacement at the interval's start, back in service by each
        # time (its CDF), averaged over each cell; the rest is linear
        # on the grid.
        done = g.grid - g.replace.in_progress(g.grid)
        row = result.smooth[-1]
        availability = np.clip(
            0.5 * (row[1:] + row[:-1]) + replaced * np.diff(done) / h,
            0.0,
            1.0,
        )
        failures = result.failures[-1]
        out_for = 1.0 - float(g.replace.cdf(np.zeros(1))[0])
        self._cycle = ConditionCycle(
            float(h * availability.sum()),
            T,
            float(failures[-1]),
            g.grid,
            availability,
            np.maximum(np.diff(failures) / h, 0.0),
            before,
            max(0.0, before - replaced * out_for),
            replaced,
        )
        return self._cycle


def level_model(rbd, node) -> LevelModel:
    """The ``LevelModel`` of a component of ``rbd`` replaced on condition
    by its level, built once and kept (with the renewal cycles)."""
    component = rbd.components[node]
    schedule = rbd._preventive[node]
    cache = rbd._cache.kept("block_cycles")
    key = (
        "level",
        node,
        id(component.reliability),
        id(component.time_to_replace),
        float(schedule.interval),
        id(schedule.duration),
        float(schedule.level),
    )
    if key not in cache:
        cache[key] = LevelModel(
            component.reliability,
            component.time_to_replace,
            schedule.duration,
            schedule.interval,
            schedule.level,
            node,
        )
    return cache[key]
