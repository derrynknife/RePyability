"""Small failure probabilities by simulation (#115).

A ``NonRepairableRBD``'s lifetime ``T`` is drawn from a row of uniforms by
its row sampler (``_sampling.RowSampler``): ``T = g(U)``, ``U`` uniform on
the unit cube of the sampler's width. Its probability of failing by ``x``,
``p = P(g(U) <= x)``, is so an integral over the cube, and any way of
sampling the cube estimates it. Plain Monte Carlo needs about
``z^2 (1 - p) / (p eps^2)`` samples for a relative precision ``eps``
(``z`` the normal quantile): some 4e10 for ``p = 1e-8`` and 10 %. The
methods here sample the cube better:

- ``"plain"``: independent uniform samples.
- ``"latin_hypercube"``: Latin hypercube samples, in independent
  replicates, the error from their spread.
- ``"sobol"``: scrambled Sobol points (randomised quasi-Monte Carlo), in
  independent replicates.
- ``"cross_entropy"``: importance sampling. In the standard normal space
  (``U = Phi(Z)``) the samples are drawn from a mixture of up to
  ``MIXTURE`` unit Gaussians shifted towards failure, fitted level by
  level by the cross-entropy method (weighted EM) to the samples that fail
  soonest, and weighted by the likelihood ratio. It follows the ways of
  failing it is fitted to, and misses others.
- ``"subset"``: subset simulation (Au and Beck, 2001): ``p`` as a product
  of conditional probabilities of failing by ever earlier times, each
  about ``P0``, the conditional samples drawn by Markov chains (adaptive
  conditional sampling, in the standard normal space). Runs are
  independent, the error from their spread.

Plain sampling and the randomised designs run until the estimate is known
to the relative precision asked; subset simulation and the cross-entropy
method, whose estimates are skewed, plan their sample sizes from a pilot
(see ``_planned``). All stop when the budget of lifetimes is spent.
``choose`` picks a method by rules measured on a benchmark suite (see the
simulation guide).
"""

import math
import warnings
from typing import Callable, NamedTuple, Tuple

import numpy as np
from scipy.special import ndtr, ndtri

from repyability.utils.checks import seed as check_seed
from repyability.utils.wrappers import outside_level

#: The conditional probability of each level of subset simulation.
P0 = 0.1
#: Samples per level of subset simulation: enough that a run's estimate is
#: not much skewed, so that the runs' spread gives an honest interval.
LEVEL_SAMPLES = 5_000
#: Samples per stage of the cross-entropy method.
STAGE_SAMPLES = 2_000
#: Replicates of a randomised design (Latin hypercube, Sobol) at least: the
#: error is estimated from their spread.
REPLICATES = 16
#: Points in a replicate of a randomised design (a power of 2, for Sobol).
DESIGN_POINTS = 4096
#: A pilot of plain samples ``choose`` takes first.
PILOT = 20_000
#: Below this many failures in the pilot, ``auto`` treats ``p`` as small.
PILOT_FAILURES = 50
#: About how many uniforms a block of samples holds.
BLOCK_UNIFORMS = 2**20
#: The most levels a subset simulation or cross-entropy fit takes.
MAX_LEVELS = 60
#: Runs of subset simulation a pilot takes, to plan how many to run.
PILOT_RUNS = 20
#: The most Gaussians the cross-entropy method's mixture has.
MIXTURE = 8

METHODS = ("plain", "latin_hypercube", "sobol", "cross_entropy", "subset")

Draw = Callable[[np.ndarray], np.ndarray]


class Estimate(NamedTuple):
    """An estimate of ``p``, its standard error, and how many lifetimes it
    took."""

    p: float
    standard_error: float
    lifetimes: int


def _planned(spread: float, pilots: int, tolerance: float, z: float) -> float:
    """How many more samples (runs, or points) of an estimator whose
    pilot of ``pilots`` had a coefficient of variation ``spread`` give a
    relative half-width of ``tolerance``. The spread is taken at an upper
    confidence bound (a sample spread of a skewed estimator is mostly too
    small), and the count is fixed in advance: stopping as soon as a
    skewed estimate looks precise would make it too low."""
    upper = spread * (1.0 + 1.0 / math.sqrt(2.0 * max(pilots - 1, 1)))
    return (z * upper / tolerance) ** 2


def _failures(draw: Draw, u: np.ndarray, x: float) -> np.ndarray:
    """Whether each row of uniforms fails by ``x``, in blocks."""
    out = np.empty(len(u), dtype=bool)
    rows = max(1, BLOCK_UNIFORMS // max(u.shape[1], 1))
    for first in range(0, len(u), rows):
        out[first : first + rows] = draw(u[first : first + rows]) <= x
    return out


def _lifetimes(draw: Draw, u: np.ndarray) -> np.ndarray:
    """The lifetimes of the rows of uniforms, in blocks."""
    out = np.empty(len(u))
    rows = max(1, BLOCK_UNIFORMS // max(u.shape[1], 1))
    for first in range(0, len(u), rows):
        out[first : first + rows] = draw(u[first : first + rows])
    return out


def _uniforms(z: np.ndarray) -> np.ndarray:
    """Standard normal points as uniforms, kept inside (0, 1)."""
    return np.clip(ndtr(z), 1e-300, 1.0 - 1e-16)


class _Sampler:
    """A method's state: the draw, the width, the time, and the generator."""

    def __init__(self, draw: Draw, width: int, x: float, rng):
        self.draw = draw
        self.width = width
        self.x = x
        self.rng = rng


def _plain_batch(s: _Sampler, n: int) -> Tuple[int, int]:
    """Failures among ``n`` plain samples."""
    hits = 0
    rows = max(1, BLOCK_UNIFORMS // max(s.width, 1))
    for first in range(0, n, rows):
        size = min(rows, n - first)
        u = s.rng.random((size, s.width))
        hits += int(np.count_nonzero(s.draw(u) <= s.x))
    return hits, n


def plain(s: _Sampler, tolerance: float, z: float, budget: int) -> Estimate:
    """Plain Monte Carlo, doubling the samples until the relative
    half-width ``z * se / p`` is at most ``tolerance``."""
    hits, n = _plain_batch(s, min(budget, PILOT))
    return _plain_from(s, hits, n, tolerance, z, budget)


def _replicated(
    s: _Sampler,
    design: Callable[[int], np.ndarray],
    size: int,
    tolerance: float,
    z: float,
    budget: int,
) -> Estimate:
    """A randomised design of ``size`` points, replicated until the
    replicates' mean is known to the tolerance (at least ``REPLICATES``)."""
    values = []
    while True:
        values.append(float(np.mean(_failures(s.draw, design(size), s.x))))
        r = len(values)
        if r < REPLICATES:
            continue
        p = float(np.mean(values))
        se = float(np.std(values, ddof=1) / math.sqrt(r))
        if (p > 0.0 and z * se <= tolerance * p) or (r + 1) * size > budget:
            return Estimate(p, se, r * size)


def latin_hypercube(
    s: _Sampler, tolerance: float, z: float, budget: int
) -> Estimate:
    """Latin hypercube samples, in replicates."""
    from scipy.stats import qmc

    engine = qmc.LatinHypercube(d=s.width, seed=s.rng)
    return _replicated(s, engine.random, DESIGN_POINTS, tolerance, z, budget)


def sobol(s: _Sampler, tolerance: float, z: float, budget: int) -> Estimate:
    """Scrambled Sobol points (a power of 2 of them), in replicates, each
    scrambled afresh."""
    from scipy.stats import qmc

    m = int(math.log2(DESIGN_POINTS))

    def design(size: int) -> np.ndarray:
        engine = qmc.Sobol(d=s.width, scramble=True, seed=s.rng)
        return engine.random_base2(m)

    return _replicated(s, design, 2**m, tolerance, z, budget)


def _log_ratio(z: np.ndarray, means: np.ndarray, weights: np.ndarray):
    """log(phi(z) / q(z)) for each row of ``z``: the log likelihood ratio
    of a standard normal point drawn from ``q``, the mixture of unit
    Gaussians centred at ``means`` with mixing ``weights``."""
    terms = (
        z @ means.T
        - 0.5 * np.einsum("kd,kd->k", means, means)
        + np.log(np.maximum(weights, 1e-300))
    )
    top = terms.max(axis=1, keepdims=True)
    return -(top[:, 0] + np.log(np.exp(terms - top).sum(axis=1)))


def _mixture_draw(s: _Sampler, n: int, means, weights) -> np.ndarray:
    """``n`` points from the mixture of unit Gaussians."""
    which = s.rng.choice(len(weights), size=n, p=weights)
    return means[which] + s.rng.standard_normal((n, s.width))


def _fit_mixture(points, weights, means):
    """The mixture of unit Gaussians that best fits the weighted
    ``points`` (weighted EM, from the components ``means``)."""
    k = len(means)
    mix = np.full(k, 1.0 / k)
    total = weights.sum()
    for _ in range(30):
        terms = points @ means.T - 0.5 * np.einsum("kd,kd->k", means, means)
        terms += np.log(np.maximum(mix, 1e-300))
        terms -= terms.max(axis=1, keepdims=True)
        share = np.exp(terms)
        share /= share.sum(axis=1, keepdims=True)
        weighted = share * weights[:, None]
        mass = weighted.sum(axis=0)
        alive = mass > 1e-12 * total
        if not alive.all():
            means, weighted, mass = (
                means[alive],
                weighted[:, alive],
                mass[alive],
            )
        new = (weighted.T @ points) / mass[:, None]
        mix = mass / mass.sum()
        if np.allclose(new, means, atol=1e-6):
            means = new
            break
        means = new
    return means, mix


def cross_entropy(
    s: _Sampler, tolerance: float, z: float, budget: int
) -> Estimate:
    """Importance sampling from a mixture of unit Gaussians in the
    standard normal space, fitted by the cross-entropy method: at each
    level the samples failing soonest (a fraction ``P0``, or all that fail
    by ``x``), weighted by their likelihood ratios, are fitted by up to
    ``MIXTURE`` Gaussians (weighted EM), until a level reaches ``x``; so
    several ways of failing are each sampled. Then batches from the fitted
    mixture, until the weighted mean is known to the tolerance."""
    n = STAGE_SAMPLES
    means = np.zeros((1, s.width))
    weights = np.ones(1)
    used = 0
    for _ in range(MAX_LEVELS):
        points = _mixture_draw(s, n, means, weights)
        life = _lifetimes(s.draw, _uniforms(points))
        used += n
        level = max(s.x, float(np.quantile(life, P0)))
        elite = life <= level
        log_w = _log_ratio(points[elite], means, weights)
        w = np.exp(log_w - log_w.max())
        chosen = points[elite]
        k = min(MIXTURE, max(1, len(chosen) // 25))
        start = chosen[s.rng.choice(len(chosen), size=k, replace=False)]
        means, weights = _fit_mixture(chosen, w, start)
        if level <= s.x or used >= budget // 2:
            break
    # The estimate, from the fitted mixture: a pilot plans the samples.
    total = total_sq = 0.0
    count = 0
    goal = 2 * n
    while count < goal and used + n <= budget:
        points = _mixture_draw(s, n, means, weights)
        fails = _failures(s.draw, _uniforms(points), s.x)
        values = np.zeros(n)
        if fails.any():
            values[fails] = np.exp(_log_ratio(points[fails], means, weights))
        total += float(values.sum())
        total_sq += float((values * values).sum())
        count += n
        used += n
        if count == 2 * n and total > 0.0:
            mean = total / count
            spread = math.sqrt(max(total_sq / count - mean * mean, 0.0)) / mean
            goal = max(goal, math.ceil(_planned(spread, count, tolerance, z)))
        elif total == 0.0:
            goal = count + n  # nothing seen yet: keep sampling
    p = total / count if count else 0.0
    var = max(total_sq / count - p * p, 0.0) if count else 0.0
    return Estimate(p, math.sqrt(var / count) if count else 0.0, used)


def _subset_run(s: _Sampler, n: int) -> Tuple[float, int]:
    """One subset simulation of ``n`` samples a level: the estimate of
    ``p``, and the lifetimes it drew.

    Each level's samples fail by its threshold, the ``P0``-quantile of the
    previous level's lifetimes. They are drawn by Markov chains from the
    previous level's samples below it, by adaptive conditional sampling
    (Papaioannou, Betz, Zwirglmaier and Straub, 2015): a chain moves from
    ``z`` to ``rho * z + sigma * xi`` (``xi`` standard normal, ``rho =
    sqrt(1 - sigma**2)``, which leaves the standard normal distribution as
    it is) if the move still fails by the threshold, with ``sigma`` adapted
    for an acceptance rate of 0.44. It is the same in every coordinate:
    the seeds' spread in a coordinate mixes the system's ways of failing,
    and scaling by it slowed the chains within each way enough to bias the
    estimate low (by about 12 % for 35 pairs in series at 1e-8)."""
    points = s.rng.standard_normal((n, s.width))
    life = _lifetimes(s.draw, _uniforms(points))
    used = n
    p = 1.0
    seeds = max(1, int(round(n * P0)))
    steps = n // seeds
    scale = 0.6
    for _ in range(MAX_LEVELS):
        order = np.argsort(life, kind="stable")
        threshold = float(life[order[seeds - 1]])
        if threshold <= s.x:
            return p * float(np.mean(life <= s.x)), used
        within = float(np.mean(life <= threshold))
        if within >= 1.0:
            # Every sample fails by the same time: no level below it.
            return p * float(np.mean(life <= s.x)), used
        p *= within
        current = points[order[:seeds]].copy()
        current_life = life[order[:seeds]].copy()
        chain_points = [current.copy()]
        chain_life = [current_life.copy()]
        for step in range(1, steps):
            sigma = min(1.0, scale)
            rho = math.sqrt(1.0 - sigma * sigma)
            candidate = rho * current + sigma * s.rng.standard_normal(
                current.shape
            )
            candidate_life = s.draw(_uniforms(candidate))
            used += seeds
            moved = candidate_life <= threshold
            current = np.where(moved[:, None], candidate, current)
            current_life = np.where(moved, candidate_life, current_life)
            chain_points.append(current.copy())
            chain_life.append(current_life.copy())
            # Towards an acceptance rate of 0.44.
            rate = float(np.mean(moved))
            scale = float(
                np.clip(
                    scale * math.exp((rate - 0.44) / math.sqrt(step)),
                    0.01,
                    100.0,
                )
            )
        points = np.concatenate(chain_points)
        life = np.concatenate(chain_life)
    return p * float(np.mean(life <= s.x)), used


def subset(s: _Sampler, tolerance: float, z: float, budget: int) -> Estimate:
    """Independent subset simulations: a pilot of ``PILOT_RUNS`` plans how
    many runs reach the tolerance (see ``_planned``), and that many run
    (within the budget)."""
    values: list = []
    used = 0
    goal = PILOT_RUNS
    while len(values) < goal:
        p_run, cost = _subset_run(s, LEVEL_SAMPLES)
        values.append(p_run)
        used += cost
        r = len(values)
        if used * (r + 1) / r > budget:
            break
        if r == PILOT_RUNS:
            mean = float(np.mean(values))
            if mean > 0.0:
                spread = float(np.std(values, ddof=1)) / mean
                goal = max(r, math.ceil(_planned(spread, r, tolerance, z)))
            else:
                goal = r + PILOT_RUNS  # nothing seen yet: run more
        elif r == goal and float(np.mean(values)) == 0.0:
            goal = r + PILOT_RUNS
    r = len(values)
    p = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(r)) if r > 1 else 0.0
    return Estimate(p, se, used)


ESTIMATORS = {
    "plain": plain,
    "latin_hypercube": latin_hypercube,
    "sobol": sobol,
    "cross_entropy": cross_entropy,
    "subset": subset,
}


def choose(
    s: _Sampler, few_modes: Callable[[], bool]
) -> Tuple[str, Tuple[int, int]]:
    """The method for this problem, by rules (see the simulation guide):

    - plain sampling when a pilot of ``PILOT`` plain samples sees at least
      ``PILOT_FAILURES`` failures (``p`` above about 2.5e-3, where plain
      sampling reaches 10 % in a few hundred thousand lifetimes);
    - else the cross-entropy method when ``few_modes()``: the system fails
      in at most ``MIXTURE`` ways (its minimal cut sets), each of plain
      components, which its mixture can each follow;
    - else subset simulation, which needs no such knowledge.

    Returns the method, and the pilot's failures and samples, which plain
    sampling goes on from."""
    hits, n = _plain_batch(s, PILOT)
    if hits >= PILOT_FAILURES:
        return "plain", (hits, n)
    if few_modes():
        return "cross_entropy", (hits, n)
    return "subset", (hits, n)


def estimate(
    draw: Draw,
    width: int,
    x: float,
    method: str,
    tolerance: float,
    confidence: float,
    budget: int,
    seed,
    few_modes: Callable[[], bool] = lambda: False,
) -> Tuple[Estimate, str]:
    """``p = P(g(U) <= x)`` by ``method`` (or the one ``choose`` picks for
    ``"auto"``, asking ``few_modes`` if it must), to a relative half-width
    of ``tolerance`` at ``confidence``, within ``budget`` lifetimes. Warns
    if the budget runs out first. Returns the estimate and the method
    used."""
    rng = np.random.default_rng(check_seed(seed))
    s = _Sampler(draw, width, float(x), rng)
    z = float(ndtri(0.5 + confidence / 2.0))
    if width == 0:
        # No uniforms: the lifetime is fixed.
        fails = bool(draw(np.empty((1, 0)))[0] <= s.x)
        return Estimate(float(fails), 0.0, 1), method
    if method == "auto":
        method, (hits, n) = choose(s, few_modes)
        if method == "plain":
            result = _plain_from(s, hits, n, tolerance, z, budget)
        else:
            result = ESTIMATORS[method](s, tolerance, z, budget - n)
            result = result._replace(lifetimes=result.lifetimes + n)
    else:
        result = ESTIMATORS[method](s, tolerance, z, budget)
    reached = z * result.standard_error <= tolerance * result.p
    if not (result.p > 0.0 and reached):
        warnings.warn(
            f"The estimate did not reach a relative precision of "
            f"{tolerance:g} within {budget:,} lifetimes (max_samples).",
            RuntimeWarning,
            stacklevel=outside_level(),
        )
    return result, method


def _plain_from(
    s: _Sampler, hits: int, n: int, tolerance: float, z: float, budget: int
) -> Estimate:
    """Plain sampling, going on from a pilot's ``hits`` in ``n``."""
    while True:
        p = hits / n
        se = math.sqrt(p * (1.0 - p) / n)
        if (p > 0.0 and z * se <= tolerance * p) or n >= budget:
            return Estimate(p, se, n)
        step = min(n, budget - n)
        more, _ = _plain_batch(s, step)
        hits, n = hits + more, n + step
