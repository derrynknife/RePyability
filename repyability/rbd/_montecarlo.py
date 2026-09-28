"""What the Monte-Carlo estimates of both kinds of RBD share: the checks on
the number of samples, the confidence interval of a mean (of antithetic
pairs' means, when the samples come in antithetic pairs), the stopping
rule of a run to a tolerance, and the blocks a parallel run is split into.

A parallel run splits its samples into blocks of a fixed size, seeded in
order from one ``numpy.random.SeedSequence``: which process runs a block
does not change what it draws, so the results do not depend on the number
of processes.
"""

import math
import os
import warnings
from typing import List, Optional

import numpy as np
from scipy.stats import norm


def check_count(n, antithetic: bool, name: str) -> None:
    """A ValueError unless ``n`` is a positive integer (and even, for
    antithetic pairs)."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"{name} must be a positive integer, got {n!r}.")
    if antithetic and n % 2:
        raise ValueError(
            f"Antithetic samples come in pairs: {name} must be even, got {n}."
        )


def check_confidence(confidence) -> None:
    if not (
        isinstance(confidence, (int, float, np.floating))
        and 0.0 < confidence < 1.0
    ):
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}.")


def means(values, antithetic: bool) -> np.ndarray:
    """The independent values whose mean is the estimate: the samples, or
    the means of the antithetic pairs (samples ``2i`` and ``2i + 1``)."""
    values = np.asarray(values, dtype=float)
    if antithetic:
        return 0.5 * (values[0::2] + values[1::2])
    return values


def standard_error(values, antithetic: bool) -> float:
    """The standard error of the mean of ``values`` (NaN for fewer than two
    independent values, or with an infinite or NaN one)."""
    independent = means(values, antithetic)
    if len(independent) < 2 or not np.all(np.isfinite(independent)):
        return math.nan
    return float(np.std(independent, ddof=1)) / math.sqrt(len(independent))


def z_value(confidence: float) -> float:
    """The two-sided normal quantile for ``confidence``."""
    return float(norm.ppf(0.5 + confidence / 2.0))


def half_width(values, confidence: float, antithetic: bool) -> float:
    """Half the width of the normal confidence interval for the mean of
    ``values`` (infinite until there are two independent values)."""
    se = standard_error(values, antithetic)
    if math.isnan(se):
        return math.inf
    return z_value(confidence) * se


def sample_limit(
    n: int,
    tolerance,
    max_n,
    antithetic: bool,
    names: tuple,
) -> Optional[int]:
    """The most samples a run to ``tolerance`` may take (``None`` without a
    tolerance: exactly ``n``). ``names`` are the argument names for the
    sample count and the limit, for the error messages."""
    count, limit_name = names
    if tolerance is None:
        if max_n is not None:
            raise ValueError(f"{limit_name} applies only with a tolerance.")
        return None
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float, np.integer, np.floating))
        or not 0.0 < tolerance < math.inf
    ):
        raise ValueError(
            f"tolerance must be a positive number, got {tolerance!r}."
        )
    limit = 100 * n if max_n is None else max_n
    if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)):
        raise ValueError(f"{limit_name} must be an integer, got {max_n!r}.")
    if limit < n or (antithetic and limit % 2):
        raise ValueError(
            f"{limit_name} must be at least {count} ({n})"
            + (" and even" if antithetic else "")
            + f", got {limit}."
        )
    return int(limit)


def more_samples(
    values,
    n: int,
    tolerance: float,
    confidence: float,
    limit: int,
    antithetic: bool,
    what: str,
    limit_name: str,
) -> int:
    """How many more samples a run to ``tolerance`` needs: 0 once the
    confidence interval of the mean of ``values`` is at most ``tolerance``
    either side, or ``limit`` samples have been taken (then with a
    RuntimeWarning); otherwise another ``n``, up to the limit. An
    infinite (or NaN) value stops the run at once: the mean is infinite
    (or undefined) however many more are taken."""
    if not np.all(np.isfinite(values)):
        return 0
    if half_width(values, confidence, antithetic) <= tolerance:
        return 0
    taken = len(values)
    if taken >= limit:
        warnings.warn(
            f"The {what} did not converge to within {tolerance} in {taken} "
            f"samples ({limit_name}); the result is from those.",
            RuntimeWarning,
            stacklevel=4,
        )
        return 0
    return min(n, limit - taken)


def jobs(n_jobs) -> int:
    """The number of processes ``n_jobs`` asks for (-1: one per CPU)."""
    if n_jobs == -1:
        return os.cpu_count() or 1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
        raise ValueError(f"n_jobs must be an integer, got {n_jobs!r}.")
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be at least 1 (or -1), got {n_jobs}.")
    return int(n_jobs)


def blocks(n: int, block: int) -> List[int]:
    """``n`` split into blocks of ``block`` (the last may be smaller)."""
    return [block] * (n // block) + ([n % block] if n % block else [])


def block_seed(seeds: np.random.SeedSequence) -> int:
    """The next block's seed, for numpy's global RNG."""
    (child,) = seeds.spawn(1)
    return int(child.generate_state(1, dtype=np.uint32)[0])
