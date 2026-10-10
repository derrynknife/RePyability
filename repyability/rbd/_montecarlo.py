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
import pickle
import warnings
from typing import List, Optional

import numpy as np
from scipy.special import ndtri

from repyability.utils.wrappers import outside_level


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
    return float(ndtri(0.5 + confidence / 2.0))


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
    unjudged: Optional[str] = None,
) -> int:
    """How many more samples a run to ``tolerance`` needs: 0 once the
    confidence interval of the mean of ``values`` is at most ``tolerance``
    either side, or ``limit`` samples have been taken (then with a
    RuntimeWarning); otherwise another ``n``, up to the limit. An
    infinite (or NaN) value stops the run at once: the mean is infinite
    (or undefined) however many more are taken. With ``unjudged``, why
    the values do not show the error yet (a run of modules that have not
    changed state): the run goes on, to the limit."""
    if not np.all(np.isfinite(values)):
        return 0
    if (
        unjudged is None
        and half_width(values, confidence, antithetic) <= tolerance
    ):
        return 0
    taken = len(values)
    if taken >= limit:
        warnings.warn(
            f"The {what} did not converge to within {tolerance} in {taken} "
            f"samples ({limit_name}); the result is from those."
            + ("" if unjudged is None else f" {unjudged}"),
            RuntimeWarning,
            stacklevel=outside_level(),
        )
        return 0
    return min(n, limit - taken)


def jobs(n_jobs) -> int:
    """The number of processes ``n_jobs`` asks for (-1: one per CPU this
    process may run on, which in a container can be fewer than the host
    has)."""
    if n_jobs == -1:
        return available_cpus()
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
        raise ValueError(f"n_jobs must be an integer, got {n_jobs!r}.")
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be at least 1 (or -1), got {n_jobs}.")
    return int(n_jobs)


def available_cpus() -> int:
    """The CPUs this process may run on: its affinity where the platform
    reports one, else all of them."""
    if hasattr(os, "process_cpu_count"):  # Python 3.13+
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def process_pool(jobs: int, initializer=None, initargs: tuple = ()):
    """A pool of ``jobs`` worker processes for a parallel run, each
    started by ``initializer(*initargs)`` if given.

    Under the forkserver start method (Linux's default from Python 3.14)
    the server, which every worker is forked from, first imports
    repyability, so a worker starts with it loaded instead of spending a
    second or more importing it (and scipy and surpyval) before its first
    block. The server keeps that for later pools too. It only takes effect
    before the server starts; the server's other preloaded modules are
    kept.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    context = multiprocessing.get_context()
    if context.get_start_method() == "forkserver":
        from multiprocessing import forkserver

        server = getattr(forkserver, "_forkserver", None)
        preload = list(getattr(server, "_preload_modules", ["__main__"]))
        if "repyability" not in preload:
            context.set_forkserver_preload(preload + ["repyability"])
    return ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
        initializer=initializer,
        initargs=initargs,
    )


def dumps(payload) -> bytes:
    """``payload`` (a system and what its simulations share) pickled for
    worker processes.

    Raises
    ------
    ValueError
        If it cannot be pickled, saying why and what to do instead.
    """
    try:
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        raise ValueError(
            "The system cannot be sent to worker processes for n_jobs "
            f"(pickle failed: {type(error).__name__}: {error}). Run it "
            "without n_jobs, or give shard_map, which sends it as JSON."
        ) from error


def blocks(n: int, block: int) -> List[int]:
    """``n`` split into blocks of ``block`` (the last may be smaller)."""
    return [block] * (n // block) + ([n % block] if n % block else [])


def block_seed(seeds: np.random.SeedSequence) -> int:
    """The next block's seed, for numpy's global RNG."""
    (child,) = seeds.spawn(1)
    return int(child.generate_state(1, dtype=np.uint32)[0])
