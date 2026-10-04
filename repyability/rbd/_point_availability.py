"""Point availability from all new at time 0, by the renewal equation.

A repairable unit starts new at 0 and runs until it fails or, under age
replacement, reaches its replacement age. It is then repaired (or
maintained) for a random time, after which it is as good as new, and so on:
its up and down periods form an alternating renewal process. Its point
availability ``A(t)``, the probability that it is up at ``t``, follows from
the distributions of those periods alone. Units that fail and are repaired
independently of each other are up or down independently at every ``t``, so
a system's point availability is its structure function evaluated exactly
at its units' (see ``RepairableRBD.point_availability``).

``unit_curve`` computes ``A(t)`` on a grid ``t_k = k h``:

- Every distribution becomes a measure on the grid that keeps each cell's
  mass and mean: a cell's mass is split between its two ends in proportion
  to where its mean falls. Sums of independent times are then exact
  convolutions of such measures, and keep every cell's mass and mean.
- The unit's renewals (its starts as new) solve ``u = delta_0 + u * c``,
  ``c`` the measure of a cycle (an up period and the down period after it),
  by inverting the power series ``1 - c`` (Newton's iteration, with FFT
  products): ``O(n log n)``.
- The unit is down at ``t`` if a down period started at some ``s <= t`` and
  lasts longer than ``t - s``. For the first unit that start is taken from
  the continuous distribution of its up time (a linear density in each cell,
  with the cell's mass and mean), and for later units from the measure
  above, spread back by hat functions; either way it is integrated exactly
  against the down time's survival function. So down times far shorter
  than the grid's step (hours of repair between years of running) cost
  nothing in accuracy.
- The integrals over cells are Gauss-Legendre, on cells split at each
  distribution's quantiles: a distribution concentrated inside one cell (a
  short repair) is integrated as finely as a long one.
- The down periods that start at a known time -- the repair of a unit dead
  on arrival, at 0, the first unit's repair after a failure at an exact
  time (an exact lifetime), and its preventive maintenance, at its
  replacement age -- are kept out of the grid (a ``GridCurve``'s dips):
  their survival functions are exact at any time, however short they are.

The error falls as the square of the step. With the default of 1,000 steps
over a unit's typical up time it is about 4e-7 (up to 4e-6 soon after the
start, for a unit whose repairs last some dozens of steps); a mission
average over more than a few steps is exact to about 4e-8, and the
long-run value is reached to about 1e-12.

Under age replacement, the units that each reach their replacement age are
maintained at nearly fixed times too: the ``n``-th after the first at
``(n + 1) age`` plus ``n`` maintenance times. Those dips are kept out of
the grid as well (``ChainDips``), each on a fine grid of its own; the
maintenance that follows a failure falls at times spread over the lives,
which the grid holds.

A unit under block replacement is followed interval by interval instead
(``_block_replacement.block_availability``, a ``BlockCurve``), a unit
with hidden failures by its closed form (an ``InspectionCurve``), and a
unit minimally repaired in no time by its life's cumulative hazard (a
``MinimalRepairCurve``). Every
curve is constant (``period`` None) or repeats with ``period`` after
``settle``, which is what lets a mission average over many years integrate
one period and repeat it.

Asked to (``counts``), a curve also counts the unit's expected events from
new (``events``): its failures, planned outages, corrective and preventive
actions and inspections before each time. The first unit's failures come
from its life's distribution, exactly; the later units' from the renewals'
lattice against the life's exact CDF, at the grid's points, which is exact
to the square of the step (about 1e-8 for a repair some hundreds of steps
long), and linear between them. Events at
exact times are kept apart (``atoms``): a unit dead on arrival at 0, the
failures of an exact lifetime, and the preventive maintenance of the units
that each reach their replacement age, one after another, at ``(n + 1)
age`` plus ``n`` maintenance times (exactly for a fixed or no maintenance
time, from the sum's distribution on ``ChainDips``' fine grid otherwise).
Past the grid's end the counts grow at their long-run rates.

A unit that degrades through stages (``stages``) is in stage ``j`` at ``t``
if its first unit is, by its age, or a unit put into service at ``s`` is,
by its age ``t - s``: the renewals against the probability of being in the
stage at each age, as for the failures; past the grid's end, the long-run
shares of the up time.
"""

from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import numpy as np

_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)

#: Fine-grid steps per standard deviation of the time a preventive
#: maintenance takes, for a unit's later maintenance under age replacement
#: (see ``ChainDips``), and the most points of a sum of such times on that
#: grid before its step doubles.
_CHAIN_STEPS = 40
_CHAIN_POINTS = 4096
#: The most values the dips of one unit may keep (its fine grids).
_CHAIN_VALUES = 2**24
#: A lifetime that ends at up to this many exact times (an exact lifetime,
#: say) has the first unit's failures at them kept as dips.
_MAX_ATOMS = 16

#: The probabilities whose quantiles split the grid's cells for the
#: integrals: finely in both tails.
_KNOT_PROBABILITIES = np.concatenate(
    [
        10.0 ** -np.arange(15, 1, -1),
        np.linspace(0.01, 0.99, 99),
        1.0 - 10.0 ** -np.arange(2, 16),
    ]
)


def knots(model) -> np.ndarray:
    """Quantiles of ``model`` to split the grid's cells at (none if it has
    no quantile function)."""
    qf = getattr(model, "qf", None)
    if qf is None:
        return np.empty(0)
    try:
        with np.errstate(all="ignore"):
            q = np.asarray(qf(_KNOT_PROBABILITIES), dtype=float).ravel()
    except Exception:  # a model whose qf cannot take these
        return np.empty(0)
    return q[np.isfinite(q)]


def _cell_integrals(f: Callable, t: np.ndarray, splits) -> np.ndarray:
    """The integral of ``f`` over each cell ``(t[k-1], t[k]]``: 4-point
    Gauss-Legendre on the cells split at ``splits``."""
    splits = np.asarray(splits, dtype=float)
    inside = splits[(splits > t[0]) & (splits < t[-1])]
    edges = np.union1d(t, inside) if inside.size else t
    a, b = edges[:-1], edges[1:]
    middle, half = 0.5 * (a + b), 0.5 * (b - a)
    x = middle[:, None] + half[:, None] * _GL_X
    pieces = (f(x.ravel()).reshape(x.shape) @ _GL_W) * half
    if edges is t:
        return pieces
    cell = np.searchsorted(t, a, side="right") - 1
    return np.bincount(cell, weights=pieces, minlength=len(t) - 1)


def _cells(cdf: Callable, t: np.ndarray, splits):
    """A (sub-)distribution with cumulative distribution function ``cdf``
    on the grid: its atom at 0, each cell's mass, and where in the cell
    (from 0 to 1) the cell's mean falls."""
    h = t[1] - t[0]
    F = cdf(t)
    integral = _cell_integrals(cdf, t, splits)
    mass = np.diff(F)
    # E[X; X in (t[k-1], t[k]]], by parts.
    mean = t[1:] * F[1:] - t[:-1] * F[:-1] - integral
    position = np.full(mass.shape, 0.5)
    has = mass > 0.0
    position[has] = (mean[has] / mass[has] - t[:-1][has]) / h
    return float(F[0]), mass, np.clip(position, 0.0, 1.0)


def _atoms(cdf: Callable, splits, top: float) -> list:
    """The times in ``(0, top]`` among ``splits`` at which ``cdf`` jumps,
    and the jumps: ``(time, probability)`` for each."""
    s = np.unique(np.asarray(splits, dtype=float))
    s = s[(s > 0.0) & (s <= top)]
    if not s.size:
        return []
    near = s * 1e-12
    at = cdf(s)
    jump, half = at - cdf(s - near), at - cdf(s - 0.5 * near)
    atom = (jump > 1e-12) & (np.abs(jump - half) <= 1e-3 * jump)
    return list(zip(s[atom].tolist(), jump[atom].tolist()))


def _lattice(atom: float, mass: np.ndarray, position: np.ndarray):
    """The measure on the grid points that keeps each cell's mass and mean:
    the atom at 0, and each cell's mass split between its ends."""
    upper = mass * position
    c = np.zeros(len(mass) + 1)
    c[0] = atom
    c[:-1] += mass - upper
    c[1:] += upper
    return c


def _atom_lattice(mass: float, at: float, t: np.ndarray) -> np.ndarray:
    """An atom of ``mass`` at time ``at`` as a measure on the grid (split
    between the grid points either side, keeping its mean)."""
    c = np.zeros(len(t))
    h = t[1] - t[0]
    if mass <= 0.0 or at > t[-1]:
        return c
    k = int(np.floor(at / h))
    if k >= len(t) - 1:
        c[-1] = mass
        return c
    upper = at / h - k
    c[k] += mass * (1.0 - upper)
    c[k + 1] += mass * upper
    return c


def _survival_weights(sf: Callable, t: np.ndarray, splits):
    """The weights by which down periods starting in a cell, or spread over
    a hat function about a grid point, are still under way ``i`` steps
    later (``sf`` is the down time's survival function).

    ``flat[i]`` and ``slope[i]`` integrate ``sf`` over lags ``[i h, (i + 1)
    h)`` against a constant and a linear density in the starting cell;
    ``hat[i]`` against a hat of half-width ``h`` about lag ``i h`` (none of
    it at negative lags, before the start).
    """
    h = t[1] - t[0]

    def position_weighted(s):
        return sf(s) * ((s - np.floor(s / h) * h) / h)

    whole = _cell_integrals(sf, t, splits)
    rising = _cell_integrals(position_weighted, t, splits)
    flat = whole / h
    slope = (0.5 * whole - rising) / h
    hat = np.zeros(len(t))
    hat[0] = (whole[0] - rising[0]) / h
    hat[1:-1] = (rising[:-1] + whole[1:] - rising[1:]) / h
    hat[-1] = rising[-1] / h
    return flat, slope, hat


def _series_inverse(p: np.ndarray, n: int) -> np.ndarray:
    """The first ``n`` coefficients of the power series ``1 / p``: Newton's
    iteration, with FFT products."""
    q = np.array([1.0 / p[0]])
    m = 1
    while m < n:
        m = min(2 * m, n)
        e = -_convolve(p[:m], q, m)
        e[0] += 2.0
        q = _convolve(q, e, m)
    return q


def _convolve(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """The first ``n`` terms of the convolution of ``a`` and ``b``, by
    FFT."""
    from scipy import fft

    length = fft.next_fast_len(len(a) + len(b) - 1, real=True)
    return fft.irfft(fft.rfft(a, length) * fft.rfft(b, length), length)[:n]


class _Spectra:
    """Convolutions of series of at most ``size`` terms, by FFT at one
    length, each series transformed once however many convolutions it is
    in (a series is not changed once it has been in one)."""

    def __init__(self, size: int):
        from scipy import fft

        self._fft = fft
        self.size = size
        self.length = fft.next_fast_len(2 * size - 1, real=True)
        self._known: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _of(self, x: np.ndarray) -> np.ndarray:
        known = self._known.get(id(x))
        if known is None or known[0] is not x:
            known = self._known[id(x)] = (x, self._fft.rfft(x, self.length))
        return known[1]

    def convolve(
        self, a: np.ndarray, b: np.ndarray, n: Optional[int] = None
    ) -> np.ndarray:
        """The first ``n`` (by default ``size``) terms of the convolution
        of ``a`` and ``b``."""
        product = self._of(a) * self._of(b)
        out = self._fft.irfft(product, self.length)
        return out[: self.size if n is None else n]


def _running(lattice: np.ndarray) -> np.ndarray:
    """The expected number of a lattice's events before each grid point:
    each point's mass is spread over its hat function (see ``_lattice``),
    so half of a point's own mass falls before it, and the mass at 0 falls
    wholly before the next point (none before 0)."""
    out = np.zeros(len(lattice))
    out[1:] = np.cumsum(lattice)[:-1] + 0.5 * lattice[1:]
    return out


def multiples_before(x: np.ndarray, interval: float) -> np.ndarray:
    """How many of ``interval, 2 interval, ...`` (each ``k * interval``, as
    the simulation schedules them) fall strictly before each ``x``."""
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore"):
        k = np.maximum(np.ceil(x / interval) - 1.0, 0.0)
    # Rounding in the division: one more or one fewer.
    k = np.where((k + 1.0) * interval < x, k + 1.0, k)
    k = np.where((k >= 1.0) & (k * interval >= x), k - 1.0, k)
    return k


def _events(
    failures: np.ndarray,
    planned: Optional[np.ndarray] = None,
    corrective: Optional[np.ndarray] = None,
    preventive: Optional[np.ndarray] = None,
    inspections: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """A curve's expected events before each time (see
    ``GridCurve.events``), none of those not given."""
    zero = np.zeros_like(failures)
    return {
        "failures": failures,
        "planned": zero if planned is None else planned,
        "corrective": failures if corrective is None else corrective,
        "preventive": zero if preventive is None else preventive,
        "inspections": zero if inspections is None else inspections,
    }


class Atoms(NamedTuple):
    """A node's events at exact times (see ``GridCurve.atoms``): when, and
    at each the probability that it fails there, that it is taken down for
    planned maintenance there, and that it is maintained there (a
    preventive replacement, taking time or not); and ``drop``, how much of
    its point availability there (``at``) those events take off (none for
    an outage in no time)."""

    times: np.ndarray
    failure: np.ndarray
    planned: np.ndarray
    preventive: np.ndarray
    drop: np.ndarray

    @staticmethod
    def none() -> "Atoms":
        empty = np.empty(0)
        return Atoms(empty, empty, empty, empty, empty)


class First(NamedTuple):
    """The unit at 0 when it is not new (see ``unit_curve``): up, with
    ``cdf`` the distribution of its failure time from 0 (what is left of
    its life; ``splits`` its quantiles) and its preventive maintenance due
    at ``due`` (0 if it is due now, None if it has none); or down (``cdf``
    None), with ``down_sf`` the survival function of what is left of its
    repair or maintenance (``down_splits`` its quantiles). ``stages``, for
    a unit up that degrades through stages, gives the probability that it
    is in each of them at each time from 0."""

    cdf: Optional[Callable] = None
    splits: Any = ()
    due: Optional[float] = None
    down_sf: Optional[Callable] = None
    down_splits: Any = ()
    stages: Optional[Callable] = None


def _capped(cdf: Callable, limit: float, at_limit: float) -> Callable:
    """A failure time's CDF ``cdf`` with no failure from ``limit`` on: the
    unit is maintained then instead."""

    def capped(x):
        return np.where(x < limit, cdf(np.minimum(x, limit)), at_limit)

    return capped


def unit_curve(
    up_cdf: Callable,
    up_splits,
    repair_sf: Callable,
    repair_splits,
    step: float,
    n: int,
    age: Optional[float] = None,
    maintenance_sf: Optional[Callable] = None,
    maintenance_splits=(),
    counts: bool = False,
    stages: Optional[Callable] = None,
    first: Optional[First] = None,
) -> "GridCurve":
    """The point availability of a unit new at 0 over ``[0, n step]``, on
    the grid ``k step`` (see the module's docstring).

    ``up_cdf`` is its lifetime's cumulative distribution function (its
    value at 0 the fraction dead on arrival; below 1 at infinity if some
    units never fail) and ``repair_sf`` the survival function of its repair
    time. Under age replacement at ``age`` a unit still up at that age is
    maintained instead, for a time with survival function
    ``maintenance_sf`` (none: no time). The ``*_splits`` are times at which
    those distributions change quickly (their quantiles): the integrals
    split the cells there. With ``counts``, the curve counts the unit's
    expected events too (see ``UnitEvents``). ``stages``, for a unit that
    degrades through stages, gives the probability that a unit is in each
    of them at each age (one row per stage): the curve then follows the
    unit's stage over time too (see ``StageCurves``).

    ``first`` starts the unit from a state other than new (see ``First``):
    only its first unit (or what is left of its down time) differs, and
    the renewals after it are those of a new unit, convolved with when they
    start.
    """
    t = step * np.arange(n + 1)
    size = n + 1
    convolve = _Spectra(size).convolve
    # Units still up at ``limit`` are maintained (none within the grid if
    # there is no age replacement, or it comes after the grid's end).
    limit = np.inf if age is None else float(age)
    if limit <= t[-1]:
        F_limit = float(up_cdf(np.array([limit]))[0])
        fail_cdf = _capped(up_cdf, limit, F_limit)
        fail_splits = np.append(np.asarray(up_splits, dtype=float), limit)
        preventive = 1.0 - F_limit
    else:
        fail_cdf, fail_splits, preventive = up_cdf, up_splits, 0.0

    # The unit at 0: new; up, with what is left of its life, and its
    # maintenance due sooner; or down (``first_cdf`` None). ``first_due``
    # and ``first_survive``: when its maintenance is due, and the
    # probability that it reaches it.
    first_cdf: Optional[Callable] = fail_cdf
    first_splits = fail_splits
    first_due: Optional[float] = limit if preventive > 0.0 else None
    first_survive = preventive
    if first is not None:
        first_due, first_survive = None, 0.0
        first_cdf, first_splits = first.cdf, first.splits
        if first.cdf is not None and first.due is not None:
            due = max(float(first.due), 0.0)
            if due <= t[-1]:
                F_due = float(first.cdf(np.array([due]))[0]) if due else 0.0
                first_cdf = _capped(first.cdf, due, F_due)
                first_splits = np.append(np.asarray(first.splits), due)
                first_due, first_survive = due, 1.0 - F_due

    def repair_cdf(s):
        return 1.0 - repair_sf(s)

    atom, mass, position = _cells(fail_cdf, t, fail_splits)
    fails = _lattice(atom, mass, position)
    # The first unit's failure times: its lattice, and its cells with the
    # atoms of a lifetime that ends at a few exact times kept apart (dips),
    # the rest a density.
    atom1, mass1, position1 = atom, mass, position
    fails1 = fails
    exact: list = []
    if first_cdf is not None:
        if first is not None:
            atom1, mass1, position1 = _cells(first_cdf, t, first_splits)
            fails1 = _lattice(atom1, mass1, position1)
        exact = _atoms(first_cdf, first_splits, t[-1])
        if not 0 < len(exact) <= _MAX_ATOMS:
            exact = []
        if exact:
            cdf = first_cdf

            def spread_cdf(x):
                out = cdf(x)
                for at, jump in exact:
                    out = out - jump * (x >= at)
                return out

            _, mass1, position1 = _cells(spread_cdf, t, first_splits)
    repairs = _lattice(*_cells(repair_cdf, t, repair_splits))
    cycle = convolve(fails, repairs)
    flat, slope, hat = _survival_weights(repair_sf, t, repair_splits)

    maintains = np.zeros(size)
    done: Optional[np.ndarray] = None
    if preventive > 0.0:
        maintains = _atom_lattice(preventive, limit, t)
        if maintenance_sf is not None:
            done = _lattice(
                *_cells(
                    lambda s: 1.0 - maintenance_sf(s), t, maintenance_splits
                )
            )
            cycle = cycle + convolve(maintains, done)
        else:
            cycle = cycle + maintains
    elif first_survive > 0.0 and maintenance_sf is not None:
        done = _lattice(
            *_cells(lambda s: 1.0 - maintenance_sf(s), t, maintenance_splits)
        )
    maintains1 = maintains
    if first is not None and first_survive > 0.0:
        assert first_due is not None
        maintains1 = _atom_lattice(first_survive, first_due, t)

    def maintained(lattice: np.ndarray) -> np.ndarray:
        """The renewals after the maintenance of ``lattice``'s starts."""
        if done is None:
            return lattice
        return convolve(lattice, done)

    # Renewals: u = delta_0 + u * cycle. The later ones (after the first
    # unit's start at 0) are u less the atom at 0.
    one_minus = -cycle
    one_minus[0] += 1.0
    later = _series_inverse(one_minus, size)
    later[0] -= 1.0
    if first is not None:
        # The first unit's cycle (its failure and repair, or its
        # maintenance; or what is left of its down time), and new units'
        # renewals after it.
        if first_cdf is not None:
            cycle1 = convolve(fails1, repairs)
            if first_survive > 0.0:
                cycle1 = cycle1 + maintained(maintains1)
        else:
            cycle1 = _lattice(
                *_cells(
                    lambda s: 1.0 - first.down_sf(s),  # type: ignore
                    t,
                    first.down_splits,
                )
            )
        later = cycle1 + convolve(cycle1, later)

    # The first unit: its failures from its continuous distribution, against
    # the repair time's survival function; its preventive maintenance, if
    # any, at exactly ``first_due``.
    down = np.zeros(size)
    if first_cdf is not None:
        down[1:] = convolve(mass1, flat, n) + convolve(
            mass1 * 6.0 * (2.0 * position1 - 1.0), slope, n
        )
    # Later units, through the grid.
    later_failures = convolve(later, fails)
    down += convolve(later_failures, hat)
    chain = None
    chained = None
    # The units that each reach their age, one after another: the n-th
    # after the first is maintained at ``first_due`` plus n ages and n
    # maintenance times, for n up to ``count`` (past it, beyond the grid or
    # all but never).
    count = 0
    if first_survive > 0.0 and preventive > 0.0:
        assert first_due is not None
        if first is None:
            count = int(np.floor(t[-1] / limit)) - 1
            if preventive < 1.0:
                count = min(count, int(np.log(1e-15) / np.log(preventive)))
        else:
            count = int(np.floor((t[-1] - first_due) / limit))
            if preventive < 1.0:
                count = min(
                    count,
                    int(np.log(1e-15 / first_survive) / np.log(preventive)),
                )
    if maintenance_sf is not None and (first_survive > 0.0 or preventive):
        _, _, maintenance_hat = _survival_weights(
            maintenance_sf, t, maintenance_splits
        )
        # The units that each reach their age are maintained at nearly
        # fixed times (``ChainDips``): the grid keeps the maintenance of
        # the units started after a failure (from a unit down at 0, of
        # every unit).
        if first_survive > 0.0:
            one_minus = -maintained(maintains)
            one_minus[0] += 1.0
            chained = _series_inverse(one_minus, size)
            chained[0] -= 1.0
            if first is not None:
                started = maintained(maintains1)
                chained = started + convolve(started, chained)
        else:
            chained = np.zeros(size)
        down += convolve(convolve(later - chained, maintains), maintenance_hat)
        if count > 0:
            chain = ChainDips(
                limit,
                preventive,
                maintenance_sf,
                maintenance_splits,
                count,
                cdfs=counts,
                first_due=first_due,
                first_survive=first_survive,
            )
    smooth = 1.0 - down
    # At 0 exactly the unit is down only if dead on arrival: a dip.
    smooth[0] = 1.0
    # The first unit's repair if it is dead on arrival, and its preventive
    # maintenance at exactly ``first_due``: exact, however short they are.
    # Down at 0, what is left of its down time is exact too.
    dips = []
    if first_cdf is None:
        assert first is not None and first.down_sf is not None
        dips.append((1.0, 0.0, first.down_sf, first.down_splits))
    elif atom1 > 0.0:
        dips.append((atom1, 0.0, repair_sf, repair_splits))
    dips += [(jump, at, repair_sf, repair_splits) for at, jump in exact]
    if first_survive > 0.0 and maintenance_sf is not None:
        assert first_due is not None
        dips.append(
            (first_survive, first_due, maintenance_sf, maintenance_splits)
        )
    curve = GridCurve(step, smooth, dips, chain=chain)
    if stages is not None:
        # In each stage: the first unit by its own age, the later ones by
        # the renewals against the probability at each age (a unit renewed
        # at a grid point itself, spread over its hat, is renewed before
        # it half the time).
        occupied = np.atleast_2d(np.asarray(stages(t), dtype=float))
        occupied = occupied.copy()
        occupied[:, 0] *= 0.5
        first_stages = stages
        if first is not None:
            first_stages = first.stages or (
                lambda x: np.zeros((len(occupied), np.size(x)))
            )
        curve.stages = StageCurves(
            t,
            first_stages,
            np.vstack(
                [np.maximum(convolve(later, row), 0.0) for row in occupied]
            ),
        )
    if not counts:
        return curve
    others = None
    fixed: Optional[float] = None
    if first_survive > 0.0 or preventive > 0.0:
        if chained is None:
            # In no time: the units that each reach their age are renewed
            # at its multiples (after the first's, from a state).
            one_minus = -maintains
            one_minus[0] += 1.0
            chained = _series_inverse(one_minus, size)
            chained[0] -= 1.0
            if first is not None:
                chained = (
                    maintains1 + convolve(maintains1, chained)
                    if first_survive > 0.0
                    else np.zeros(size)
                )
        # The other units' maintenance, through the grid.
        others = _running(convolve(later - chained, maintains))
        if maintenance_sf is None:
            fixed = 0.0
        elif chain is not None:
            fixed = chain.fixed  # None: a time that varies
        else:
            fixed = 0.0  # no later unit reaches its age on the grid
    # The later units' failures before each grid point: the renewals
    # against the life's exact CDF there (a unit renewed at the point
    # itself, spread over its hat, is dead on arrival before it half the
    # time).
    lived = np.array(fail_cdf(t), dtype=float)
    lived[0] = 0.5 * atom
    first_atoms = list(exact)
    if first_cdf is not None and atom1 > 0.0:
        first_atoms.insert(0, (0.0, atom1))
    curve.unit_events = UnitEvents(
        t,
        first_cdf if first_cdf is not None else (lambda x: np.zeros(len(x))),
        first_atoms,
        float(repair_sf(np.zeros(1))[0]),
        np.maximum(convolve(later, lived), 0.0),
        age=limit if others is not None else None,
        survive=preventive,
        fixed=fixed,
        chain=chain,
        count=max(count, 0),
        others=others,
        takedown=maintenance_sf is not None,
        maintenance_drop=(
            0.0
            if maintenance_sf is None
            else float(maintenance_sf(np.zeros(1))[0])
        ),
        first_due=first_due,
        first_survive=first_survive,
    )
    return curve


def _moments(sf: Callable, t: np.ndarray, splits) -> Tuple[float, float]:
    """The mean and variance of a time with survival function ``sf``, all
    but none of which falls after ``t[-1]``."""
    mean = float(_cell_integrals(sf, t, splits).sum())
    second = 2.0 * float(_cell_integrals(lambda s: s * sf(s), t, splits).sum())
    return mean, max(second - mean**2, 0.0)


def _second_difference(values: np.ndarray):
    """``values`` padded with a zero at each end, and its second
    difference there."""
    padded = np.concatenate([[0.0, 0.0], values, [0.0, 0.0]])
    second = padded[2:] - 2.0 * padded[1:-1] + padded[:-2]
    return padded[1:-1], second


def _cubic(values: np.ndarray, position: np.ndarray) -> np.ndarray:
    """``values`` at fractional indices ``position``, by 4-point Lagrange
    interpolation (0 outside)."""
    padded = np.concatenate([[0.0, 0.0], values, [0.0, 0.0]])
    j = np.clip(np.floor(position).astype(int), -1, len(values) - 1) + 2
    u = position - (j - 2)
    return (
        -u * (u - 1.0) * (u - 2.0) / 6.0 * padded[j - 1]
        + (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0 * padded[j]
        - (u + 1.0) * u * (u - 2.0) / 2.0 * padded[j + 1]
        + (u + 1.0) * u * (u - 1.0) / 6.0 * padded[j + 2]
    )


def _cubic_slope(values: np.ndarray, position: np.ndarray) -> np.ndarray:
    """The rate of change of ``_cubic(values, position)`` in ``position``:
    the derivative of the same 4-point Lagrange interpolation."""
    padded = np.concatenate([[0.0, 0.0], values, [0.0, 0.0]])
    j = np.clip(np.floor(position).astype(int), -1, len(values) - 1) + 2
    u = position - (j - 2)
    return (
        -(3.0 * u**2 - 6.0 * u + 2.0) / 6.0 * padded[j - 1]
        + (3.0 * u**2 - 4.0 * u - 1.0) / 2.0 * padded[j]
        - (3.0 * u**2 - 2.0 * u - 2.0) / 2.0 * padded[j + 1]
        + (3.0 * u**2 - 1.0) / 6.0 * padded[j + 2]
    )


def _density(sf: Callable, u: np.ndarray, splits) -> np.ndarray:
    """The density at each ``u >= 0`` of a time with survival function
    ``sf`` (quantiles ``splits``): its differences over a millionth of the
    time's own scale (its median, or ``u``), central where they fit after
    0 and one-sided, second order, where they do not. A down time far
    shorter than a curve's grid step is differentiated as finely as a long
    one (see ``GridCurve.rate``)."""
    u = np.asarray(u, dtype=float)
    knots = np.asarray(splits, dtype=float)
    knots = knots[knots > 0.0]
    typical = float(np.median(knots)) if knots.size else 0.0
    h = 1e-6 * np.maximum(np.maximum(u, typical), 1e-300)
    central = u >= h
    out = np.empty(u.shape)
    if central.any():
        a, b = u[central] - h[central], u[central] + h[central]
        out[central] = (sf(a) - sf(b)) / (2.0 * h[central])
    if (~central).any():
        v, k = u[~central], h[~central]
        out[~central] = (3.0 * sf(v) - 4.0 * sf(v + k) + sf(v + 2.0 * k)) / (
            2.0 * k
        )
    return out


class _GridPart:
    """A curve's part on its grid, linear between its points, as
    ``_rates.differences`` takes a curve: its values (``at``), the times
    it bends other than on its grid (``breaks``) and its grids."""

    def __init__(self, at: Callable, breaks: Callable, grids: list):
        self.at = at
        self.breaks = breaks
        self._grids = grids

    def grids(self) -> list:
        return self._grids


def chain_due(first_due: float, age: float, n):
    """When the ``n``-th of the units that each reach their age after the
    first (whose maintenance is due at ``first_due``) is due, but for the
    maintenance times before it: ``first_due + n age``, or ``(n + 1) age``
    from new (as block times are multiples of their interval)."""
    n = np.asarray(n, dtype=float)
    if first_due == age:
        return (n + 1.0) * age
    return first_due + n * age


class ChainDips:
    """The preventive maintenance of a unit under age replacement whose
    units, one after another, each reach their replacement age ``age``
    (each with probability ``survive``): the ``n``-th after the first is
    maintained from ``(n + 1) age + T_n``, ``T_n`` the sum of ``n``
    maintenance times, for one more. Those times are nearly fixed, however
    long the unit runs, so the grid cannot hold them; the probability that
    the unit is down in the ``n``-th such maintenance,
    ``survive ** (n + 1) * G_n(s)`` with ``G_n(s) = P(T_n <= s < T_n + D)``
    and ``s`` the time since ``(n + 1) age``, is kept for ``n = 1..count``
    on a fine grid of its own.

    There the maintenance time is a lattice that keeps each cell's mass and
    mean (see ``_lattice``), corrected by its second difference to keep its
    variance as well; its ``n``-fold sums are FFT convolutions, and each is
    spread by hat functions against the time's survival function and
    sharpened by the hats' variance, which makes the error fall as the
    fourth power of the step (``_CHAIN_STEPS`` steps per standard deviation
    give about 1e-7). A sum wider than ``_CHAIN_POINTS`` points moves to a
    grid of twice the step. A maintenance of a fixed time ``d`` has
    ``G_n(s) = 1`` on ``[n d, (n + 1) d)``, exactly.

    With ``cdfs``, it also keeps the distribution of each ``T_n``, the
    lattice's running sums sharpened likewise (the Euler-Maclaurin
    correction), for counting the maintenance (``before``)."""

    def __init__(
        self,
        age: float,
        survive: float,
        sf,
        splits,
        count: int,
        cdfs: bool = False,
        first_due: Optional[float] = None,
        first_survive: Optional[float] = None,
    ):
        self.age = float(age)
        self.survive = float(survive)
        # The first unit's maintenance (a unit not new at 0: sooner, and
        # with its own probability); the later ones follow at ``age``.
        self.first_due = self.age if first_due is None else float(first_due)
        self.first_survive = (
            self.survive if first_survive is None else float(first_survive)
        )
        self.count = int(count)
        self.sf = sf
        splits = np.asarray(splits, dtype=float)
        self.splits = splits[np.isfinite(splits) & (splits >= 0.0)]
        self.top = float(self.splits.max()) if self.splits.size else 0.0
        self.windows: list = []  # (start, step, values) for n = 1..count
        # (start, step, values, mass) of T_n's CDF for n = 1..count.
        self.cdfs: list = []
        self.fixed: Optional[float] = None
        if self.top <= 0.0:
            self.fixed = 0.0  # instant: no dips
            return
        mean, variance = _moments(
            sf, np.linspace(0.0, self.top, 4097), self.splits
        )
        if variance <= (1e-9 * self.top) ** 2:
            self.fixed = mean
            return
        from scipy.signal import fftconvolve

        step = np.sqrt(variance) / _CHAIN_STEPS
        if self.top / step > _CHAIN_VALUES:
            raise NotImplementedError(self._too_many)
        kernel, hat = self._kernels(step)
        total, first = kernel, -1  # the sum's lattice, from index ``first``
        kept = 0
        for n in range(1, self.count + 1):
            # P(T_n <= s < T_n + D) at the fine points, sharpened by the
            # hats' variance, step^2 / 6.
            padded, second = _second_difference(fftconvolve(total, hat))
            values = padded - second / 12.0
            start = first - 1
            if start < 0:  # none before its due time
                values, start = values[-start:], 0
            keep = np.flatnonzero(np.abs(values) > 1e-17)
            if keep.size:
                values = values[keep[0] : keep[-1] + 1]  # noqa: E203
                start += int(keep[0])
            self.windows.append((start * step, step, values))
            kept += len(values)
            if cdfs:
                # P(T_n < s) at the fine points: the masses before each,
                # half of its own, and the Euler-Maclaurin correction, which
                # where the time's density jumps (at 0, for an exponential
                # one) can overshoot below 0 or above the mass: a count of
                # them would dip there (#164).
                mass = float(total.sum())
                running = np.cumsum(total) - 0.5 * total
                padded = np.concatenate([[0.0], running, [mass]])
                second = padded[2:] - 2.0 * padded[1:-1] + padded[:-2]
                cdf = np.clip(running - second / 12.0, 0.0, mass)
                self.cdfs.append((first * step, step, cdf, mass))
                kept += len(running)
            if kept > _CHAIN_VALUES:
                raise NotImplementedError(self._too_many)
            total = fftconvolve(total, kernel)
            first -= 1
            keep = np.flatnonzero(np.abs(total) > 1e-18)
            total = total[keep[0] : keep[-1] + 1]  # noqa: E203
            first += int(keep[0])
            if len(total) > _CHAIN_POINTS and n < self.count:
                total, first = self._coarsen(total, first, step)
                step *= 2.0
                kernel, hat = self._kernels(step)

    _too_many = (
        "its later age replacements, which fall at nearly fixed times, are "
        "too many, or their times too spread out, to follow exactly"
    )

    def _due(self, n: int) -> Tuple[float, float]:
        """When the ``n``-th maintenance after the first is due, before the
        ``n`` maintenance times, and the probability that it happens."""
        due = float(chain_due(self.first_due, self.age, n))
        return due, self.first_survive * self.survive**n

    def _kernels(self, step: float):
        """The maintenance time's lattice on the grid ``step * k`` from
        ``k = -1``, corrected to keep its variance (unless that would let
        its sums grow), and its hat weights (see ``_survival_weights``)."""
        t = step * np.arange(int(np.ceil(self.top / step)) + 2)
        lattice = _lattice(*_cells(lambda s: 1.0 - self.sf(s), t, self.splits))
        _, variance = _moments(self.sf, t, self.splits)
        mass = lattice.sum()
        mean = (lattice @ t) / mass
        excess = (lattice @ t**2) / mass - mean**2 - variance
        padded, second = _second_difference(lattice)
        kernel = padded - excess / (2.0 * step**2) * second
        size = 8 * len(kernel)
        if np.abs(np.fft.rfft(kernel, size)).max() > mass * (1.0 + 1e-9):
            kernel = padded
        _, _, hat = _survival_weights(self.sf, t, self.splits)
        return kernel, hat

    @staticmethod
    def _coarsen(total: np.ndarray, first: int, step: float):
        """The sum's lattice on the grid of twice the step: each point on
        it, or halved between the two either side, keeping its mass and
        mean; then corrected to keep its variance."""
        if first % 2:
            total, first = np.concatenate([[0.0], total]), first - 1
        if len(total) % 2 == 0:
            total = np.concatenate([total, [0.0]])
        even, odd = total[0::2], total[1::2]
        coarse = even.copy()
        coarse[:-1] += 0.5 * odd
        coarse[1:] += 0.5 * odd
        added = step**2 * odd.sum() / total.sum()
        padded, second = _second_difference(coarse)
        return padded - added / (8.0 * step**2) * second, first // 2 - 1

    def at(self, x: np.ndarray) -> np.ndarray:
        """The probability that the unit is down at ``x`` in one of these
        maintenances."""
        out = np.zeros(len(x))
        if self.fixed == 0.0:
            return out
        order = np.argsort(x, kind="stable")
        ordered = x[order]
        for n in range(1, self.count + 1):
            due, weight = self._due(n)
            if self.fixed is not None:
                lo, hi = due + n * self.fixed, due + (n + 1) * self.fixed
                a, b = np.searchsorted(ordered, [lo, hi], side="left")
                out[order[a:b]] += weight
                continue
            start, step, values = self.windows[n - 1]
            lo = due + start
            hi = lo + step * len(values)
            a, b = np.searchsorted(ordered, [lo - step, hi])
            if b > a:
                position = (ordered[a:b] - lo) / step
                out[order[a:b]] += weight * _cubic(values, position)
        return out

    def rate(self, x: np.ndarray) -> np.ndarray:
        """The rate at which ``at`` changes at each ``x``: the derivative of
        each maintenance's interpolation on its own fine grid (none for a
        fixed time, constant between its ends)."""
        out = np.zeros(len(x))
        if self.fixed is not None:
            return out
        order = np.argsort(x, kind="stable")
        ordered = x[order]
        for n in range(1, self.count + 1):
            due, weight = self._due(n)
            start, step, values = self.windows[n - 1]
            lo = due + start
            hi = lo + step * len(values)
            a, b = np.searchsorted(ordered, [lo - step, hi])
            if b > a:
                position = (ordered[a:b] - lo) / step
                out[order[a:b]] += (
                    weight * _cubic_slope(values, position) / step
                )
        return out

    def before(self, x: np.ndarray) -> np.ndarray:
        """The expected number of these maintenances, ``n = 1..count``,
        that start before each ``x``: ``survive ** (n + 1) * P((n + 1) age
        + T_n < x)``, from the CDFs kept (``cdfs``), for a maintenance
        whose time varies."""
        out = np.zeros(len(x))
        order = np.argsort(x, kind="stable")
        ordered = x[order]
        # Each sum's mass, for the times past its window.
        past = np.zeros(len(x) + 1)
        for n in range(1, len(self.cdfs) + 1):
            due, weight = self._due(n)
            start, step, values, mass = self.cdfs[n - 1]
            lo = due + start - step
            hi = due + start + step * len(values)
            a, b = np.searchsorted(ordered, [lo, hi], side="right")
            if b > a:
                position = (ordered[a:b] - due - start) / step
                extended = np.concatenate([values, [mass, mass]])
                position = np.minimum(position, len(extended) - 1.0)
                # Clipped: a cubic undershoots where the sum's CDF rises
                # off 0 (#164).
                cdf = np.clip(_cubic(extended, position), 0.0, mass)
                out[order[a:b]] += weight * cdf
            past[b] += weight * mass
        out[order] += np.cumsum(past)[:-1]
        return out

    def knots(self, start: float, stop: float) -> np.ndarray:
        """Times in ``[start, stop]`` between which the dips are smooth."""
        parts = []
        for n in range(1, self.count + 1):
            due, _ = self._due(n)
            if self.fixed is not None:
                parts.append(due + self.fixed * np.array([n, n + 1.0]))
                continue
            begin, step, values = self.windows[n - 1]
            stride = max(1, len(values) // 256)
            # The window's ends too: where it starts, the dip rises from
            # nothing (a kink, as the maintenance starts at once).
            index = np.concatenate(
                [
                    [-1, 0],
                    np.arange(-1, len(values) + 1, stride),
                    [len(values) - 1, len(values)],
                ]
            )
            parts.append(due + begin + step * np.unique(index))
        if not parts:
            return np.empty(0)
        times = np.concatenate(parts)
        return times[(times >= start) & (times <= stop)]


class UnitEvents:
    """What a unit new at 0 is expected to do by each time (see
    ``unit_curve``), on the grid ``times``.

    Its first unit fails at a time with CDF ``fail_cdf`` (none after its
    replacement age), with ``first_atoms``, ``(time, probability)``, at
    exact times; a failure's repair takes time with probability
    ``repair_drop``. ``later`` is the later units' expected failures before
    each grid point. Under age replacement at ``age``, the units that each
    reach it one after another (each with probability ``survive``) are
    maintained at ``(n + 1) age + T_n``: ``T_n = n * fixed`` for a
    maintenance of a fixed time (0 for none), or from ``chain``'s CDFs, for
    ``n`` up to ``count``; ``others`` is the other units' expected
    maintenance before each grid point. ``takedown``: the maintenance takes
    the unit down (it has a time model), and takes time with probability
    ``maintenance_drop``. Past the grid's end, the counts grow at
    ``failure_rate`` and ``preventive_rate``, the long-run rates, which the
    caller sets when the grid has settled."""

    def __init__(
        self,
        times: np.ndarray,
        fail_cdf: Callable,
        first_atoms: list,
        repair_drop: float,
        later: np.ndarray,
        age: Optional[float] = None,
        survive: float = 0.0,
        fixed: Optional[float] = None,
        chain: Optional[ChainDips] = None,
        count: int = 0,
        others: Optional[np.ndarray] = None,
        takedown: bool = False,
        maintenance_drop: float = 0.0,
        first_due: Optional[float] = None,
        first_survive: Optional[float] = None,
    ):
        self.times = times
        self.fail_cdf = fail_cdf
        self.first_atoms = list(first_atoms)
        self.repair_drop = repair_drop
        self.later = later
        self.age = age
        self.survive = survive
        self.fixed = fixed
        self.chain = chain
        self.count = count
        self.others = others
        self.takedown = takedown
        self.maintenance_drop = maintenance_drop
        # The first unit's maintenance: at the age from new, sooner (or
        # none) from a state.
        self.first_due = age if first_due is None else first_due
        self.first_survive = (
            survive if first_survive is None else first_survive
        )
        self.failure_rate: Optional[float] = None
        self.preventive_rate: Optional[float] = None

    def _chain_dues(self) -> Tuple[np.ndarray, np.ndarray]:
        """When the units that each reach their age are maintained, and
        the probabilities, for a maintenance of a fixed time (or none):
        ``n = 0..count``, within the grid (none if the first is never
        maintained)."""
        if self.first_due is None or not self.first_survive:
            return np.empty(0), np.empty(0)
        assert self.age is not None and self.fixed is not None
        n = np.arange(self.count + 1, dtype=float)
        dues = chain_due(self.first_due, self.age, n) + n * self.fixed
        weights = self.first_survive * self.survive**n
        keep = dues <= self.times[-1]
        return dues[keep], weights[keep]

    def _maintained(self, x: np.ndarray, inside: np.ndarray) -> np.ndarray:
        """The expected preventive maintenance before each ``x`` (``inside``
        it, held at the grid's end)."""
        assert self.others is not None, "the unit is not maintained"
        out = np.interp(inside, self.times, self.others)
        if self.fixed is not None:
            dues, weights = self._chain_dues()
            running = np.concatenate([[0.0], np.cumsum(weights)])
            return out + running[np.searchsorted(dues, x, side="left")]
        # The first at its due time exactly; the later ones' times vary.
        if self.first_due is not None and self.first_survive:
            out += self.first_survive * (x > self.first_due)
        if self.chain is not None:
            out += self.chain.before(inside)
        return out

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The expected failures, planned outages, corrective and
        preventive actions before each ``x`` (see ``GridCurve.events``)."""
        x = np.asarray(x, dtype=float)
        end = float(self.times[-1])
        inside = np.minimum(x, end)
        # The first unit: its life's continuous part to ``inside``, and its
        # atoms strictly before ``x``.
        first = np.array(self.fail_cdf(inside), dtype=float)
        for at, probability in self.first_atoms:
            first -= probability * (inside >= at)
            first += probability * (x > at)
        failures = first + np.interp(inside, self.times, self.later)
        past = np.maximum(x - end, 0.0)
        if self.failure_rate is not None:
            failures = failures + self.failure_rate * past
        # Nothing happens before 0 (the lattice's rounding aside).
        started = x > 0.0
        failures = np.where(started, failures, 0.0)
        if self.age is None:
            return _events(failures)
        preventive = self._maintained(x, inside)
        if self.preventive_rate is not None:
            preventive = preventive + self.preventive_rate * past
        preventive = np.where(started, preventive, 0.0)
        planned = preventive if self.takedown else None
        return _events(failures, planned, preventive=preventive)

    def atoms(self, stop: float) -> Atoms:
        """The unit's events at exact times before ``stop`` (see
        ``Atoms``): the first unit's failures, and its maintenance at its
        age and, for a fixed time (or none), the later ones'."""
        first = np.array(self.first_atoms, dtype=float).reshape(-1, 2)
        first = first[first[:, 0] < stop]
        zero = np.zeros(len(first))
        failing = Atoms(
            first[:, 0],
            first[:, 1],
            zero,
            zero,
            first[:, 1] * self.repair_drop,
        )
        if self.age is None:
            return failing
        if self.fixed is not None:
            dues, weights = self._chain_dues()
            # The first is a dip; a later one takes the unit down for the
            # fixed time, if any.
            drops = np.full(len(dues), 1.0 if self.fixed > 0.0 else 0.0)
        elif self.first_due is not None and self.first_survive:
            dues = np.array([self.first_due])
            weights = np.array([self.first_survive])
            drops = np.ones(1)
        else:
            dues = weights = drops = np.empty(0)
        if len(drops):
            drops[0] = self.maintenance_drop
        keep = dues < stop
        dues, weights, drops = dues[keep], weights[keep], drops[keep]
        takedown = 1.0 if self.takedown else 0.0
        maintained = Atoms(
            dues,
            np.zeros(len(dues)),
            weights * takedown,
            weights,
            weights * drops * takedown,
        )
        merged = Atoms(*(np.concatenate(c) for c in zip(failing, maintained)))
        order = np.argsort(merged.times, kind="stable")
        return Atoms(*(column[order] for column in merged))

    def settled(self, tolerance: float = 1e-7) -> bool:
        """Whether the counts have reached their long-run rates over the
        last quarter of the grid, to ``tolerance`` relative to the rate
        (or, for a rate of 0, to the mean rate so far)."""
        n = len(self.times) - 1
        tail = self.times[n - n // 4 :]  # noqa: E203
        if len(tail) < 3:
            return False
        events = self.events(tail)
        for key, rate in (
            ("failures", self.failure_rate),
            ("preventive", self.preventive_rate),
        ):
            if key == "preventive" and self.age is None:
                continue
            if rate is None:
                return False
            slope = np.diff(events[key]) / np.diff(tail)
            scale = max(rate, float(events[key][-1]) / float(tail[-1]))
            if np.any(np.abs(slope - rate) > tolerance * scale):
                return False
        return True


class StageCurves:
    """Which stage a unit that degrades through stages is in, from new (see
    ``unit_curve``): ``first(x)``, the probability that the first unit is
    in each stage at ``x`` (one row per stage), and ``later``, that a later
    one is, at each grid point of ``times``. Past the grid's end the unit's
    up time is shared among the stages as ``fractions`` (set by the caller
    when the grid has settled): the long-run share of its up time in
    each."""

    def __init__(self, times: np.ndarray, first: Callable, later):
        self.times = times
        self.first = first
        self.later = later
        self.fractions: Optional[np.ndarray] = None

    def shares(self, x: np.ndarray) -> np.ndarray:
        """The share of the probability that the unit is up at each ``x``
        that falls in each stage (one row per stage)."""
        x = np.asarray(x, dtype=float)
        inside = np.minimum(x, self.times[-1])
        occupied = np.atleast_2d(np.asarray(self.first(inside), dtype=float))
        occupied = occupied + np.vstack(
            [np.interp(inside, self.times, row) for row in self.later]
        )
        total = occupied.sum(axis=0)
        shares = np.zeros_like(occupied)
        shares[0] = 1.0  # where it cannot be up: a new unit's first stage
        np.divide(occupied, total, out=shares, where=total > 0.0)
        if self.fractions is not None:
            shares[:, x > self.times[-1]] = self.fractions[:, None]
        return shares

    def settled(self, tolerance: float = 1e-8) -> bool:
        """Whether the shares have reached ``fractions`` over the last
        quarter of the grid."""
        if self.fractions is None:
            return False
        n = len(self.times) - 1
        tail = self.times[n - n // 4 :]  # noqa: E203
        shares = self.shares(tail)
        return bool(
            np.all(np.abs(shares - self.fractions[:, None]) <= tolerance)
        )


def curve_breaks(curve, start: float, stop: float) -> np.ndarray:
    """The times in ``[start, stop]`` at which ``curve`` may bend or jump
    other than on its grids (see ``curve_grids``): all its knots, for a
    curve that has none."""
    breaks = getattr(curve, "breaks", None)
    return curve.knots(start, stop) if breaks is None else breaks(start, stop)


def curve_grids(curve) -> list:
    """The grids ``curve`` is linear on between its breaks (see
    ``curve_breaks``): ``(step, until)`` for each, ``until`` the time after
    which the curve no longer changes on it. None (an empty list), for a
    curve whose knots are all breaks."""
    grids = getattr(curve, "grids", None)
    return [] if grids is None else grids()


class GridCurve:
    """A unit's point availability on a grid from 0, linear between its
    points, less its dips: ``(probability, start, sf, splits)`` for a down
    period that starts at ``start`` with that probability and lasts a time
    with survival function ``sf`` (quantiles ``splits``), and those of
    ``chain`` (see ``ChainDips``). After the grid's end the curve holds
    ``long_run`` if given (it has settled there by then), and the grid then
    reaches the times it is needed at. ``unit_events``, if counted, are the
    unit's expected events (see ``UnitEvents``), and ``stages``, if
    followed, the stages of a unit that degrades (see ``StageCurves``)."""

    period = None

    def __init__(
        self,
        step: float,
        smooth: np.ndarray,
        dips=(),
        long_run: Optional[float] = None,
        chain: Optional[ChainDips] = None,
    ):
        self.times = step * np.arange(len(smooth))
        self.smooth = smooth
        self.dips = list(dips)
        self.long_run = long_run
        self.chain = chain
        self.unit_events: Optional[UnitEvents] = None
        self.stages: Optional[StageCurves] = None

    def stages_at(self, x: np.ndarray) -> np.ndarray:
        """The probability that the unit is up and in each of its stages
        at each time ``x`` (one row per stage): its point availability
        shared among the stages (see ``StageCurves``), so that the rows add
        up to it."""
        assert self.stages is not None, "stages were not asked for"
        return self.stages.shares(x) * self.at(np.asarray(x, dtype=float))

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The unit's expected events before each time ``x`` (in
        ``[0, x)``, as the simulation counts them): ``"failures"``,
        ``"planned"`` outages (preventive maintenance that takes time),
        ``"corrective"`` actions (repairs, at each failure),
        ``"preventive"`` actions (maintenance, taking time or not) and
        ``"inspections"`` (none)."""
        assert self.unit_events is not None, "counts were not asked for"
        return self.unit_events.events(x)

    def atoms(self, stop: float) -> Atoms:
        """The unit's events at exact times before ``stop``."""
        assert self.unit_events is not None, "counts were not asked for"
        return self.unit_events.atoms(stop)

    @property
    def settle(self) -> float:
        """The time after which the curve is constant."""
        return np.inf if self.long_run is None else float(self.times[-1])

    def at(self, x: np.ndarray) -> np.ndarray:
        out = np.interp(x, self.times, self.smooth)
        for probability, start, sf, _ in self.dips:
            after = x >= start
            out[after] -= probability * sf(x[after] - start)
        if self.chain is not None:
            out -= self.chain.at(x)
        out = np.clip(out, 0.0, 1.0)
        if self.long_run is not None:
            out[x > self.times[-1]] = self.long_run
        return out

    def knots(self, start: float, stop: float) -> np.ndarray:
        """The times in ``[start, stop]`` between which the curve is
        smooth."""
        times = np.concatenate([self.times, self.breaks(start, stop)])
        return times[(times >= start) & (times <= stop)]

    def breaks(self, start: float, stop: float) -> np.ndarray:
        """The times in ``[start, stop]`` at which the curve bends other
        than at its grid's points (see ``curve_breaks``): the grid's ends,
        and its dips'."""
        parts = [self.times[[0, -1]]]
        for _, begin, _, splits in self.dips:
            parts.append(begin + np.append(0.0, splits))
        if self.chain is not None:
            parts.append(self.chain.knots(start, stop))
        times = np.concatenate(parts)
        return times[(times >= start) & (times <= stop)]

    def grids(self) -> list:
        """Its grid, to its end (see ``curve_grids``)."""
        if len(self.times) < 2:
            return []
        return [(float(self.times[1]), float(self.times[-1]))]

    def _grid_at(self, x: np.ndarray) -> np.ndarray:
        """Its values on its grid alone, without its dips."""
        out = np.interp(x, self.times, self.smooth)
        if self.long_run is not None:
            out[x > self.times[-1]] = self.long_run
        return out

    def rate(self, x: np.ndarray, scale: float) -> np.ndarray:
        """Its rate of change at each ``x`` (see ``_rates.derivative``): its
        grid's by differences (see ``_rates.differences``), which are
        second order a step apart, less its dips' exactly, each from its
        down time's density (``_density``), and those of ``chain`` on
        their own fine grids. A dip far shorter than a step (a few hours'
        maintenance on a grid of days) would be smoothed over by the
        differences of the whole curve."""
        from repyability.rbd._rates import differences

        ends = self.times[[0, -1]]

        def breaks(start: float, stop: float) -> np.ndarray:
            return ends[(ends >= start) & (ends <= stop)]

        out = differences(
            _GridPart(self._grid_at, breaks, self.grids()), x, scale
        )
        for probability, start, sf, splits in self.dips:
            after = x >= start
            if after.any():
                out[after] += probability * _density(
                    sf, x[after] - start, splits
                )
        if self.chain is not None:
            out -= self.chain.rate(x)
        if self.long_run is not None:
            out[x > self.times[-1]] = 0.0
        return out


class BlockCurve:
    """A unit's point availability under block replacement, from new (see
    ``_block_replacement.BlockAvailability``), the last interval repeating
    once it has settled. ``duration_knots`` are quantiles of the time a
    replacement takes. After a head (not ``fresh``), the curve starts at a
    block time, with the replacement due there. Replaced on condition (see
    ``_condition_replacement.condition_availability``), the block times
    are its inspections, which it counts too."""

    def __init__(self, result, duration_knots):
        self.interval = float(result.interval)
        self.grid = result.grid
        self.step = float(result.grid[1] - result.grid[0])
        self.smooth = result.smooth
        self.replaced = result.replaced
        self.replace = result.replace
        self.failures = result.failures
        self.fresh = result.fresh
        self.inspected = result.inspected
        knots = np.asarray(duration_knots, dtype=float)
        self.duration_knots = knots[(knots > 0.0) & (knots < self.interval)]
        last = len(self.replaced) - 1
        self.settle = last * self.interval if result.settled else np.inf
        self.period = self.interval if result.settled else None

    def _whole(self, values: np.ndarray, k: np.ndarray) -> np.ndarray:
        """The sum of ``values`` (one per interval, the last repeating) over
        the intervals before the ``k``-th."""
        rows = len(values)
        running = np.concatenate([[0.0], np.cumsum(values)])
        inside = np.minimum(k, rows).astype(int)
        return running[inside] + np.maximum(k - rows, 0.0) * values[-1]

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The unit's expected events before each time ``x`` (see
        ``GridCurve.events``): its failures, interval by interval, and its
        replacements at the block times, which take it down if they take
        time (have a time model); replaced on condition, its inspections,
        one at each block time it is up at."""
        x = np.asarray(x, dtype=float)
        k = np.floor(x / self.interval)
        s = x - k * self.interval
        over, under = s >= self.interval, s < 0.0
        k[over] += 1.0
        s[over] -= self.interval
        k[under] -= 1.0
        s[under] += self.interval
        row = np.minimum(k, len(self.replaced) - 1).astype(int)
        position = s / self.step
        j = np.clip(np.floor(position).astype(int), 0, len(self.grid) - 2)
        fraction = position - j
        within = (
            self.failures[row, j] * (1.0 - fraction)
            + self.failures[row, j + 1] * fraction
        )
        failures = self._whole(self.failures[:, -1], k) + within
        # The replacements at the block times before x: the k-th is
        # replaced with probability ``replaced[k]`` (none at 0 from new).
        due = multiples_before(x, self.interval)
        replaced = self.replaced.copy()
        if self.fresh:
            replaced[0] = 0.0
        before = np.where(x > 0.0, due + 1.0, 0.0)
        preventive = self._whole(replaced, before)
        planned = preventive if self.replace.model is not None else None
        inspections = None
        if self.inspected is not None:
            inspections = self._whole(self.inspected, before)
        return _events(
            failures, planned, preventive=preventive, inspections=inspections
        )

    def atoms(self, stop: float) -> Atoms:
        """The replacements at the block times before ``stop`` (see
        ``Atoms``)."""
        count = int(multiples_before(np.array([stop]), self.interval)[0])
        first = 1 if self.fresh else 0
        if count < first or stop <= 0.0:
            return Atoms.none()
        k = np.arange(first, count + 1)
        count = len(k)
        times = k * self.interval
        weights = self.replaced[np.minimum(k, len(self.replaced) - 1)]
        if self.replace.model is None:
            planned = drop = np.zeros(count)
        else:
            planned = weights
            drop = weights * (1.0 - float(self.replace.cdf(np.zeros(1))[0]))
        return Atoms(times, np.zeros(count), planned, weights, drop)

    def _place(self, x: np.ndarray) -> tuple:
        """The interval each time falls in (``k``), the time since its
        start (``s``, in ``[0, interval)``: a block time starts its
        interval), and the curve's values on its grid there."""
        k = np.floor(x / self.interval)
        s = x - k * self.interval
        over, under = s >= self.interval, s < 0.0
        k[over] += 1.0
        s[over] -= self.interval
        k[under] -= 1.0
        s[under] += self.interval
        row = np.minimum(k, len(self.replaced) - 1).astype(int)
        position = s / self.step
        j = np.clip(np.floor(position).astype(int), 0, len(self.grid) - 2)
        fraction = position - j
        smooth = (
            self.smooth[row, j] * (1.0 - fraction)
            + self.smooth[row, j + 1] * fraction
        )
        return k, s, row, smooth

    def at(self, x: np.ndarray) -> np.ndarray:
        k, s, row, smooth = self._place(x)
        back = self.replace.cdf(s)
        if self.fresh:
            back = np.where(k == 0.0, 1.0, back)
        return np.clip(smooth + self.replaced[row] * back, 0.0, 1.0)

    def rate(self, x: np.ndarray, scale: float) -> np.ndarray:
        """Its rate of change at each ``x`` (see ``GridCurve.rate``): its
        grid's by differences, and the return of the units replaced at the
        block time before it exactly, from the replacement time's density
        (none from new, in the first interval)."""
        from repyability.rbd._rates import differences

        def blocks(start: float, stop: float) -> np.ndarray:
            first = int(np.floor(start / self.interval))
            last = int(np.floor(stop / self.interval))
            times = self.interval * np.arange(first, last + 1, dtype=float)
            return times[(times >= start) & (times <= stop)]

        out = differences(
            _GridPart(lambda t: self._place(t)[3], blocks, self.grids()),
            x,
            scale,
        )
        if self.replace.model is None:
            return out
        k, s, row, _ = self._place(x)
        back = _density(
            lambda u: 1.0 - self.replace.cdf(u), s, self.duration_knots
        )
        if self.fresh:
            back = np.where(k == 0.0, 0.0, back)
        return out + self.replaced[row] * back

    def knots(self, start: float, stop: float) -> np.ndarray:
        """The times in ``[start, stop]`` between which the curve is smooth:
        the grid in each interval, and where the replacement at its start
        is likely to end."""
        first = int(np.floor(start / self.interval))
        last = int(np.floor(stop / self.interval))
        offsets = np.concatenate([self.grid[:-1], self.duration_knots])
        times = (
            self.interval * np.arange(first, last + 1)[:, None]
            + offsets[None, :]
        ).ravel()
        return times[(times >= start) & (times <= stop)]

    def breaks(self, start: float, stop: float) -> np.ndarray:
        """The times in ``[start, stop]`` at which the curve bends other
        than at its grid's points (see ``curve_breaks``): the block times,
        and where the replacement at each is likely to end."""
        first = int(np.floor(start / self.interval))
        last = int(np.floor(stop / self.interval))
        offsets = np.append(0.0, self.duration_knots)
        times = (
            self.interval * np.arange(first, last + 1)[:, None]
            + offsets[None, :]
        ).ravel()
        return times[(times >= start) & (times <= stop)]

    def grids(self) -> list:
        """Its grid, in every interval (see ``curve_grids``)."""
        return [(self.step, np.inf)]


class InspectionCurve:
    """A unit with hidden failures, a constant failure ``rate``, and instant
    tests and repair every ``interval``, the last test ``phase`` before 0
    (0: tested at 0, as from new), and last known up ``since`` before 0 (at
    that test, by default, or since put into service after it): up with
    probability ``exp(-rate * u)``, ``u`` the time since it was last known
    up (at a test, after the first). It repeats from its first test on (from
    0, if it was last known up at its last test)."""

    def __init__(
        self,
        rate: float,
        interval: float,
        phase: float = 0.0,
        since: Optional[float] = None,
    ):
        self.rate = rate
        self.interval = interval
        self.period = interval
        self.phase = float(phase)
        self.since = self.phase if since is None else float(since)
        #: The first test after 0.
        self.first = interval - self.phase
        self.settle = 0.0 if self.since == self.phase else self.first

    def at(self, x: np.ndarray) -> np.ndarray:
        position = x + self.phase
        since = position - self.interval * np.floor(position / self.interval)
        if self.since != self.phase:
            since = np.where(x < self.first, x + self.since, since)
        return np.exp(-self.rate * since)

    def _failed(self, position: np.ndarray) -> np.ndarray:
        """The expected hidden failures from the calendar's start (a test)
        to each ``position`` on it."""
        whole = np.floor(position / self.interval)
        since = np.maximum(position - self.interval * whole, 0.0)
        per_test = -np.expm1(-self.rate * self.interval)
        return whole * per_test - np.expm1(-self.rate * since)

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The unit's expected events before each time ``x`` (see
        ``GridCurve.events``): its hidden failures, at most one between
        two tests, ``1 - exp(-rate * u)`` by a time ``u`` after it was last
        known up; the tests; and the failures they find (each repaired
        then, as a corrective action)."""
        x = np.asarray(x, dtype=float)
        tests = multiples_before(x + self.phase, self.interval)
        per_test = -np.expm1(-self.rate * self.interval)
        if self.since == self.phase:
            failures = self._failed(x + self.phase)
            if self.phase:
                failures = failures - self._failed(np.array(self.phase))
            corrective = tests * per_test
        else:
            # Up to the first test, from when it was last known up; from
            # then on, as from a test.
            known = np.exp(-self.rate * self.since)
            head = known - np.exp(
                -self.rate * (self.since + np.minimum(x, self.first))
            )
            later = np.maximum(x - self.first, 0.0)
            failures = head + np.where(tests > 0, self._failed(later), 0.0)
            found = -np.expm1(-self.rate * (self.since + self.first))
            corrective = np.where(
                tests > 0, found + (tests - 1.0) * per_test, 0.0
            )
        return _events(failures, corrective=corrective, inspections=tests)

    def atoms(self, stop: float) -> Atoms:
        """None: a test, in no time, takes nothing down."""
        return Atoms.none()

    def knots(self, start: float, stop: float) -> np.ndarray:
        """The tests in ``[start, stop]``, and times between them close
        enough (``rate`` times the gap at most 1/4) for quadrature to be
        exact to rounding."""
        pieces = max(1, int(np.ceil(4.0 * self.rate * self.interval)))
        first = int(np.floor((start + self.phase) / self.interval))
        last = int(np.floor((stop + self.phase) / self.interval))
        times = (
            self.interval
            * (
                np.arange(first, last + 1)[:, None]
                + np.arange(pieces)[None, :] / pieces
            )
        ).ravel()
        if self.phase:
            times = times - self.phase
        return times[(times >= start) & (times <= stop)]


class PartialTestCurve:
    """A unit with hidden failures, a constant failure ``rate``, and instant
    tests and repair, from new, whose tests can miss a failure: tested in
    full at ``offset`` (at 0, as new, with none) and at every ``per_full``
    tests after it, and every ``interval`` in between by a test that finds
    a failure with probability ``coverage`` (below 1). Up with probability
    ``exp(-rate * x)`` before its first test, then ``rho ** k * exp(-rate *
    u)``, ``k`` the tests since the last full one and ``u`` the time since
    the last, ``rho = 1 - (1 - coverage) * (1 - exp(-rate * interval))``
    (see ``RepairableRBD._tested_profile``). It repeats every full test's
    interval from its first test."""

    def __init__(
        self,
        rate: float,
        interval: float,
        offset: float,
        coverage: float,
        per_full: int,
    ):
        self.rate = rate
        self.interval = interval
        self.offset = float(offset)
        self.coverage = coverage
        self.per_full = per_full
        self.period = interval * per_full
        self.settle = self.offset
        missed = (1.0 - coverage) * -np.expm1(-rate * interval)
        #: log(rho).
        self.kept = float(np.log1p(-missed))
        #: The expected failures over a full test's cycle, all found by its
        #: end: (1 - rho ** per_full) / (1 - coverage).
        self.per_cycle = float(-np.expm1(per_full * self.kept)) / (
            1.0 - coverage
        )

    def _position(self, x: np.ndarray):
        """For times ``x`` from the first test on: the full tests' cycles
        since it, the tests since the last full one, and the time since
        the last test."""
        since = x - self.offset
        cycles = np.floor(since / self.period)
        within = since - cycles * self.period
        tests = np.clip(np.floor(within / self.interval), 0, self.per_full - 1)
        return cycles, tests, within - tests * self.interval

    def at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        _, tests, since = self._position(np.maximum(x, self.offset))
        later = np.exp(tests * self.kept - self.rate * since)
        return np.where(x < self.offset, np.exp(-self.rate * x), later)

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The unit's expected events before each time ``x`` (see
        ``GridCurve.events``): its hidden failures (before the first test,
        at most one, which it finds; then, between two tests, at most one
        for a unit up after the first, ``rho ** j (1 - exp(-rate * u))``);
        the tests; and the failures they find, each repaired then: a
        failure in a cycle is found by its next test with probability
        ``coverage``, else by the full test that ends the cycle, so the
        ``j``-th test of a cycle finds ``coverage * rho ** (j - 1) * (1 -
        exp(-rate * interval))`` and the full test the rest."""
        x = np.asarray(x, dtype=float)
        c = self.coverage
        after = x > self.offset
        head = -np.expm1(-self.rate * np.minimum(x, self.offset))
        cycles, tests, since = self._position(np.maximum(x, self.offset))
        # Up after the k-th test with probability rho ** k: the failures in
        # its first k intervals, sum_j rho ** j (1 - e), are
        # (1 - rho ** k) / (1 - c).
        within = -np.expm1(tests * self.kept) / (1.0 - c) + np.exp(
            tests * self.kept
        ) * -np.expm1(-self.rate * since)
        failures = head + np.where(
            after, cycles * self.per_cycle + within, 0.0
        )
        later = multiples_before(x - self.offset, self.interval)
        whole, rest = np.divmod(later, self.per_full)
        found = whole * self.per_cycle + c * -np.expm1(rest * self.kept) / (
            1.0 - c
        )
        corrective = np.where(after, head + found, 0.0)
        inspections = later + (after if self.offset else 0.0)
        return _events(
            failures, corrective=corrective, inspections=inspections
        )

    def atoms(self, stop: float) -> Atoms:
        """None: a test, in no time, takes nothing down."""
        return Atoms.none()

    def knots(self, start: float, stop: float) -> np.ndarray:
        """The tests in ``[start, stop]``, and times between them close
        enough (``rate`` times the gap at most 1/4) for quadrature to be
        exact to rounding (before the first test too)."""
        pieces = max(1, int(np.ceil(4.0 * self.rate * self.interval)))
        first = int(np.floor((start - self.offset) / self.interval))
        last = int(np.floor((stop - self.offset) / self.interval))
        times = (
            self.offset
            + self.interval
            * (
                np.arange(first, last + 1)[:, None]
                + np.arange(pieces)[None, :] / pieces
            )
        ).ravel()
        return times[(times >= start) & (times <= stop)]


class SteadyCurve:
    """A unit in its long-run state from 0 (a stationary start, for a unit
    long in service whose state is not known): up with its long-run
    ``availability`` throughout, its failures and preventive maintenance
    at their long-run rates (taking it down, with ``takedown``), and a
    degrading unit in each stage with its long-run share of its up time,
    ``fractions``."""

    settle = 0.0
    period = None

    def __init__(
        self,
        availability: float,
        failure_rate: float = 0.0,
        preventive_rate: float = 0.0,
        takedown: bool = False,
        fractions: Optional[np.ndarray] = None,
    ):
        self.availability = float(availability)
        self.failure_rate = float(failure_rate)
        self.preventive_rate = float(preventive_rate)
        self.takedown = takedown
        self.fractions = fractions

    def at(self, x: np.ndarray) -> np.ndarray:
        return np.full(np.shape(x), self.availability)

    def knots(self, start: float, stop: float) -> np.ndarray:
        return np.empty(0)

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The unit's expected events before each time ``x``: at their
        long-run rates."""
        x = np.asarray(x, dtype=float)
        preventive = self.preventive_rate * x
        return _events(
            self.failure_rate * x,
            preventive if self.takedown else None,
            preventive=preventive,
        )

    def atoms(self, stop: float) -> Atoms:
        return Atoms.none()

    def stages_at(self, x: np.ndarray) -> np.ndarray:
        assert self.fractions is not None, "the unit has no stages"
        return self.fractions[:, None] * self.at(np.atleast_1d(x))[None, :]


class MinimalRepairCurve:
    """A unit minimally repaired in no time (Kijima's models with ``q = 1``
    and an instant repair): up throughout, and its failures a
    non-homogeneous Poisson process whose intensity is its life's hazard at
    its age, so that it fails ``H(x)`` times before ``x`` on average, ``H``
    the life's cumulative hazard, exactly.

    Its counts never settle (with a wearing-out life they grow ever faster),
    so it is followed to the horizon; for the integrals of its counts it
    declares a grid of ``steps`` steps over its first ``scale`` past the
    life's ``offset``, whose step doubles each time that span does
    (``grids``): a piece then spans the same share of the time it starts at
    however far on it is, as many pieces for each doubling of the window.
    The offset, before which the unit cannot fail, is a break. ``counts``
    False, only its availability is asked for, which is 1 from the start.
    """

    period = None

    def __init__(self, life, scale: float, steps: int, counts: bool):
        self.life = life
        self.offset = max(float(getattr(life, "gamma", 0.0) or 0.0), 0.0)
        self.scale = float(scale)
        self.steps = int(steps)
        self.counts = counts
        self.settle = np.inf if counts else 0.0

    def at(self, x: np.ndarray) -> np.ndarray:
        return np.ones(np.shape(x))

    def knots(self, start: float, stop: float) -> np.ndarray:
        return self.breaks(start, stop)

    def breaks(self, start: float, stop: float) -> np.ndarray:
        times = np.array([self.offset]) if self.offset > 0.0 else np.empty(0)
        return times[(times >= start) & (times <= stop)]

    def grids(self) -> list:
        """Its grids (see ``curve_grids``): to the offset plus ``scale``
        doubled ``k`` times, a step of that time over ``steps``."""
        if not self.counts:
            return []
        ends = self.offset + self.scale * 2.0 ** np.arange(64)
        return [(float(end) / self.steps, float(end)) for end in ends]

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The unit's expected failures before each time ``x``: its life's
        cumulative hazard there (from its survival function where the
        model gives no hazard, or a NaN one), each a repair."""
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            hazard = np.asarray(self.life.Hf(x), dtype=float).reshape(x.shape)
            missing = np.isnan(hazard)
            if missing.any():
                survival = np.asarray(self.life.sf(x[missing]), dtype=float)
                hazard[missing] = -np.log(np.clip(survival, 0.0, 1.0))
        return _events(np.maximum(hazard, 0.0))

    def atoms(self, stop: float) -> Atoms:
        return Atoms.none()


class ShiftedCurve:
    """A unit on a calendar in its long-run state, ``shift`` into ``curve``
    (one settled into repeating with its ``period``): from new, then far
    enough on for its long-run cycle, and on to the phase of its
    calendar. A time within rounding (1e-9 of the period) of a scheduled
    time is that time, so that the events there fall where the curve
    says."""

    def __init__(self, curve, shift: float):
        self.curve = curve
        self.shift = float(shift)
        self.period = curve.period
        # It repeats from the start, unless the start is itself one of its
        # scheduled times: that time's events are past, so the first period
        # is not like the later ones, and it repeats from the second.
        self.settle = 0.0
        if self.period and self._past().times.size:
            self.settle = float(self.period)

    def _position(self, x) -> np.ndarray:
        position = np.asarray(x, dtype=float) + self.shift
        if self.period:
            k = np.round(position / self.period)
            near = np.abs(position - k * self.period) <= 1e-9 * self.period
            position = np.where(near, k * self.period, position)
        return position

    def at(self, x: np.ndarray) -> np.ndarray:
        return self.curve.at(self._position(x))

    def rate(self, x: np.ndarray, scale: float) -> np.ndarray:
        """Its curve's rate (see ``_rates.derivative``), ``shift`` on."""
        from repyability.rbd._rates import derivative

        return derivative(self.curve, self._position(x), scale)

    def knots(self, start: float, stop: float) -> np.ndarray:
        times = self.curve.knots(start + self.shift, stop + self.shift)
        times = times - self.shift
        return times[(times >= start) & (times <= stop)]

    def breaks(self, start: float, stop: float) -> np.ndarray:
        times = curve_breaks(self.curve, start + self.shift, stop + self.shift)
        times = times - self.shift
        return times[(times >= start) & (times <= stop)]

    def grids(self) -> list:
        return [
            (step, until - self.shift)
            for step, until in curve_grids(self.curve)
        ]

    def _past(self) -> Atoms:
        """The curve's events at the start itself, which are past (the
        replacement just done, at the phase's 0)."""
        start = float(self._position(0.0))
        atoms = self.curve.atoms(np.nextafter(start, np.inf))
        keep = atoms.times == start
        return Atoms(*(column[keep] for column in atoms))

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The unit's expected events before each time ``x``: the curve's
        from the shift on, less those at the start itself, which are past
        (none before a time that is the start, to rounding)."""
        position = self._position(np.asarray(x, dtype=float))
        begin = float(self._position(0.0))
        later = self.curve.events(position)
        start = self.curve.events(np.array([begin]))
        past = self._past()
        taken = {
            "failures": past.failure.sum(),
            "planned": past.planned.sum(),
            "corrective": past.failure.sum(),
            "preventive": past.preventive.sum(),
            "inspections": 0.0,
        }
        after = position > begin
        return {
            key: np.where(after, later[key] - start[key][0] - taken[key], 0.0)
            for key in later
        }

    def atoms(self, stop: float) -> Atoms:
        start = float(self._position(0.0))
        atoms = self.curve.atoms(float(self._position(stop)))
        keep = atoms.times > start
        kept = [column[keep] for column in atoms]
        kept[0] = kept[0] - self.shift
        return Atoms(*kept)


class StartedBlockCurve:
    """A unit under block replacement started from a state: ``head``, its
    own curve up to its first block time, ``length`` after the start, and
    then ``tail``, the block-replacement curve from there (a ``BlockCurve``
    started after the head; see ``_block_replacement.BlockHead``)."""

    def __init__(self, head, tail, length: float):
        self.head = head
        self.tail = tail
        self.length = float(length)
        self.settle = self.length + tail.settle
        self.period = tail.period

    def at(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        before = x < self.length
        out = np.empty(x.shape)
        if before.any():
            out[before] = self.head.at(x[before])
        if (~before).any():
            out[~before] = self.tail.at(x[~before] - self.length)
        return out

    def rate(self, x: np.ndarray, scale: float) -> np.ndarray:
        """The head's rate before its end and the tail's from it (see
        ``_rates.derivative``)."""
        from repyability.rbd._rates import derivative

        before = x < self.length
        out = np.empty(x.shape)
        if before.any():
            out[before] = derivative(self.head, x[before], scale)
        if (~before).any():
            out[~before] = derivative(
                self.tail, x[~before] - self.length, scale
            )
        return out

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The unit's expected events before each time ``x``: the head's,
        to its end, and the tail's after it."""
        x = np.asarray(x, dtype=float)
        head = self.head.events(np.minimum(x, self.length))
        after = x > self.length
        if not after.any():
            return head
        tail = self.tail.events(np.maximum(x - self.length, 0.0))
        return {
            key: head[key] + np.where(after, tail[key], 0.0) for key in head
        }

    def atoms(self, stop: float) -> Atoms:
        head = self.head.atoms(min(stop, self.length))
        if stop <= self.length:
            return head
        tail = self.tail.atoms(stop - self.length)
        return Atoms(
            np.concatenate([head.times, tail.times + self.length]),
            *(
                np.concatenate([mine, theirs])
                for mine, theirs in zip(head[1:], tail[1:])
            ),
        )

    def knots(self, start: float, stop: float) -> np.ndarray:
        return self._joined(start, stop, lambda curve, a, b: curve.knots(a, b))

    def breaks(self, start: float, stop: float) -> np.ndarray:
        return self._joined(start, stop, curve_breaks)

    def _joined(self, start: float, stop: float, times_of) -> np.ndarray:
        """``times_of(curve, start, stop)`` of the head to its end and of
        the tail after it, with the time between them."""
        parts = [times_of(self.head, start, min(stop, self.length))]
        if stop >= self.length:
            later = times_of(
                self.tail, max(start - self.length, 0.0), stop - self.length
            )
            parts += [np.array([self.length]), later + self.length]
        times = np.concatenate(parts)
        return times[(times >= start) & (times <= stop)]

    def grids(self) -> list:
        head = [
            (step, min(until, self.length))
            for step, until in curve_grids(self.head)
        ]
        tail = [
            (step, until + self.length)
            for step, until in curve_grids(self.tail)
        ]
        return head + tail


class SystemCurve:
    """A nested RBD's point availability: its structure function at its
    own nodes' point availabilities, which settle at ``settle`` into a
    constant (``period`` None) or repeating with ``period``."""

    def __init__(self, rbd, curves: dict, settle: float, period):
        self.rbd = rbd
        self.curves = curves
        self.settle = settle
        self.period = period

    def at(self, x: np.ndarray) -> np.ndarray:
        return self.rbd._curves_at(self.curves, x, set(), set(), "p")

    def rate(self, x: np.ndarray, scale: float) -> np.ndarray:
        """Its rate of change at each ``x``: each of its nodes' (see
        ``_rates.derivative``) times that node's Birnbaum importance there,
        the system being multilinear in its nodes' availabilities (see
        ``_rates``)."""
        from repyability.rbd._rates import derivative

        values = {node: c.at(x) for node, c in self.curves.items()}
        importance = self.rbd._importances(
            self.rbd._filled(values, len(x), set(), set())
        )[0]
        out = np.zeros(len(x))
        for node, curve in self.curves.items():
            out += importance[node] * derivative(curve, x, scale)
        return out

    def events(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """The nested RBD's expected failures and planned outages before
        each time ``x`` (see ``RepairableRBD._window_counts``): its own
        maintenance is not this RBD's to count."""
        counts = self.rbd._window_counts(
            self.curves, np.asarray(x, dtype=float), set(), set(), "p"
        )
        return _events(counts["failures"], counts["planned"])

    def atoms(self, stop: float) -> Atoms:
        """The nested RBD's failures and planned outages at exact times
        before ``stop`` (see ``RepairableRBD._atom_groups``)."""
        return self.rbd._system_atoms(self.curves, stop)

    def knots(self, start: float, stop: float) -> np.ndarray:
        parts = [curve.knots(start, stop) for curve in self.curves.values()]
        return np.concatenate(parts) if parts else np.array([start])

    def breaks(self, start: float, stop: float) -> np.ndarray:
        parts = [curve_breaks(c, start, stop) for c in self.curves.values()]
        return np.concatenate(parts) if parts else np.array([start])

    def grids(self) -> list:
        return [
            g for curve in self.curves.values() for g in curve_grids(curve)
        ]
