"""Reliability demonstration test planning (#129).

A demonstration test shows, at a confidence level, that a design meets a
reliability target: test ``n`` units for a time, allow ``failures`` of them,
and if no more fail, the design has demonstrated the target. These functions
plan such tests the usual way round, from the target to the test, and give
what a finished test demonstrated:

- **Attribute (binomial) tests.** Each of ``n`` units runs for the mission
  time, or ``test_multiple`` times it, and passes or fails.
  ``demonstration_sample_size`` gives the fewest units that demonstrate a
  reliability, ``demonstrated_reliability`` what a test demonstrated (the
  Clopper-Pearson lower bound, which with no failures is surpyval's
  ``success_run``), ``demonstration_test_multiple`` how long each of ``n``
  units must run, and ``demonstration_pass_probability`` the chance a design
  of a given true reliability passes (the test's operating characteristic).
- **Extended tests (Weibayes).** Testing each unit for ``k`` times the
  mission time trades units for test time when the Weibull shape ``beta`` of
  the lifetime is known: a unit that survives ``k`` missions survives one
  with reliability ``R`` exactly when ``R ** (k ** beta)`` is its reliability
  over the test. Give ``test_multiple=k`` and ``shape=beta``.
- **Constant failure rate (chi-squared) tests.** Units run for a total
  ``test_time``, failed ones replaced or repaired, and the test passes with at
  most ``failures`` failures. ``mtbf_test_time`` gives the total time that
  demonstrates an MTBF, ``demonstrated_mtbf`` what a test demonstrated, and
  ``mtbf_pass_probability`` the chance a design of a given true MTBF passes.

The functions plan tests from targets and results; they fit nothing (surpyval
does that).
"""

from typing import Optional

import numpy as np
from scipy import stats


def demonstration_sample_size(
    reliability: float,
    confidence: float = 0.95,
    failures: int = 0,
    *,
    test_multiple: float = 1.0,
    shape: Optional[float] = None,
) -> int:
    """The fewest units whose test demonstrates a reliability.

    The smallest ``n`` for which ``n`` units tested, with at most
    ``failures`` of them failing, demonstrate ``reliability`` over a mission
    at ``confidence``: the chance that so few fail, were the reliability
    only the target, is at most ``1 - confidence``. With no failures this is
    the success run, ``n = ln(1 - confidence) / ln(reliability)`` rounded
    up, the inverse of surpyval's ``success_run``. Testing each unit for
    ``test_multiple`` missions, with the lifetime's Weibull ``shape`` known,
    needs fewer units (the extended success run, or Weibayes).

    Parameters
    ----------
    reliability : float
        The reliability to demonstrate over one mission, in (0, 1).
    confidence : float, optional
        The confidence level, in (0, 1), by default 0.95.
    failures : int, optional
        The failures the test allows, by default 0.
    test_multiple : float, optional
        How many missions long each unit's test is, by default 1.0.
    shape : float, optional
        The Weibull shape of the lifetime, needed when ``test_multiple`` is
        not 1.

    Returns
    -------
    int
        The number of units to test.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``test_multiple`` is not 1 and
        no ``shape`` is given.

    Examples
    --------
    59 units with no failures demonstrate 95% reliability at 95%
    confidence, and allowing one failure takes 93:

    >>> from repyability import demonstration_sample_size
    >>> demonstration_sample_size(0.95, 0.95)
    59
    >>> demonstration_sample_size(0.95, 0.95, failures=1)
    93

    Testing each unit for two missions, for a wear-out shape of 2, needs a
    quarter of the units:

    >>> demonstration_sample_size(0.95, 0.95, test_multiple=2.0, shape=2.0)
    15
    """
    _check_probability(reliability, "reliability")
    _check_probability(confidence, "confidence")
    failures = _check_failures(failures)
    per_test = _test_reliability(reliability, test_multiple, shape)
    alpha = 1.0 - confidence

    def passes_too_often(n: int) -> bool:
        return stats.binom.cdf(failures, n, 1.0 - per_test) > alpha

    low, high = failures + 1, 2 * failures + 2
    while passes_too_often(high):
        low, high = high + 1, 2 * high
    while low < high:
        middle = (low + high) // 2
        if passes_too_often(middle):
            low = middle + 1
        else:
            high = middle
    return int(low)


def demonstrated_reliability(
    n: int,
    confidence: float = 0.95,
    failures: int = 0,
    *,
    test_multiple: float = 1.0,
    shape: Optional[float] = None,
) -> float:
    """The reliability a finished test demonstrated.

    The lower one-sided confidence bound on the reliability over a mission,
    from ``n`` units tested with ``failures`` of them failing: the
    Clopper-Pearson bound, the smallest reliability for which so few
    failures have a chance of at least ``1 - confidence``. With no failures
    it is ``(1 - confidence) ** (1 / n)``, surpyval's ``success_run``. A test
    of ``test_multiple`` missions per unit, with the lifetime's Weibull
    ``shape`` known, demonstrates that bound to the power
    ``1 / test_multiple ** shape`` over one mission.

    Parameters
    ----------
    n : int
        The number of units tested, at least 1.
    confidence : float, optional
        The confidence level, in (0, 1), by default 0.95.
    failures : int, optional
        How many of them failed, from 0 to ``n``, by default 0.
    test_multiple : float, optional
        How many missions long each unit's test was, by default 1.0.
    shape : float, optional
        The Weibull shape of the lifetime, needed when ``test_multiple`` is
        not 1.

    Returns
    -------
    float
        The demonstrated reliability over one mission.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``test_multiple`` is not 1 and
        no ``shape`` is given.

    Examples
    --------
    >>> from repyability import demonstrated_reliability
    >>> round(demonstrated_reliability(59), 4)
    0.9505
    >>> round(demonstrated_reliability(93, failures=1), 5)
    0.95001
    """
    _check_probability(confidence, "confidence")
    n = _check_count(n)
    failures = _check_failures(failures)
    if failures > n:
        raise ValueError(f"failures must be at most n ({n}), got {failures}.")
    exponent = _test_exponent(test_multiple, shape)
    if failures == n:
        return 0.0
    upper = stats.beta.ppf(confidence, failures + 1, n - failures)
    return float((1.0 - upper) ** (1.0 / exponent))


def demonstration_test_multiple(
    reliability: float,
    n: int,
    confidence: float = 0.95,
    failures: int = 0,
    *,
    shape: float,
) -> float:
    """How many missions long each unit's test must be to demonstrate a
    reliability with ``n`` units.

    The shortest test, as a multiple of the mission time, for which ``n``
    units with at most ``failures`` failing demonstrate ``reliability`` at
    ``confidence``, the lifetime's Weibull ``shape`` known (the extended
    success run, or Weibayes). Below 1, a test shorter than a mission is
    enough.

    Parameters
    ----------
    reliability : float
        The reliability to demonstrate over one mission, in (0, 1).
    n : int
        The number of units to test, more than ``failures``.
    confidence : float, optional
        The confidence level, in (0, 1), by default 0.95.
    failures : int, optional
        The failures the test allows, by default 0.
    shape : float
        The Weibull shape of the lifetime.

    Returns
    -------
    float
        The test length, in missions.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    Twenty units, for a wear-out shape of 2, each run about 1.7 missions to
    demonstrate 95% reliability at 95% confidence (with a test of one
    mission, 59 units would be needed):

    >>> from repyability import demonstration_test_multiple
    >>> round(demonstration_test_multiple(0.95, 20, shape=2.0), 3)
    1.709
    """
    _check_probability(reliability, "reliability")
    n = _check_count(n)
    failures = _check_failures(failures)
    if failures >= n:
        raise ValueError(
            f"n must be more than the failures allowed ({failures}), got {n}."
        )
    _check_shape(shape)
    per_test = demonstrated_reliability(n, confidence, failures)
    return float((np.log(per_test) / np.log(reliability)) ** (1.0 / shape))


def demonstration_pass_probability(
    reliability: float,
    n: int,
    failures: int = 0,
    *,
    test_multiple: float = 1.0,
    shape: Optional[float] = None,
) -> float:
    """The chance that a design of a given true reliability passes a test.

    The probability that at most ``failures`` of ``n`` units fail when each
    survives a mission with probability ``reliability`` (the test's
    operating characteristic). At the target reliability it is at most
    ``1 - confidence`` for a planned test, the consumer's risk; one minus
    it at a good design's reliability is the producer's risk, the chance of
    failing a design that should pass.

    Parameters
    ----------
    reliability : float
        The design's true reliability over one mission, in (0, 1).
    n : int
        The number of units tested, at least 1.
    failures : int, optional
        The failures the test allows, by default 0.
    test_multiple : float, optional
        How many missions long each unit's test is, by default 1.0.
    shape : float, optional
        The Weibull shape of the lifetime, needed when ``test_multiple`` is
        not 1.

    Returns
    -------
    float
        The probability of passing.

    Raises
    ------
    ValueError
        If an argument is out of range, or ``test_multiple`` is not 1 and
        no ``shape`` is given.

    Examples
    --------
    The 59-unit, no-failure test passes a design at the 95% target about
    one time in twenty, and one of 99% reliability five times in nine:

    >>> from repyability import demonstration_pass_probability
    >>> round(demonstration_pass_probability(0.95, 59), 3)
    0.048
    >>> round(demonstration_pass_probability(0.99, 59), 3)
    0.553
    """
    _check_probability(reliability, "reliability")
    n = _check_count(n)
    failures = _check_failures(failures)
    per_test = _test_reliability(reliability, test_multiple, shape)
    return float(stats.binom.cdf(failures, n, 1.0 - per_test))


def mtbf_test_time(
    mtbf: float, confidence: float = 0.95, failures: int = 0
) -> float:
    """The total test time that demonstrates an MTBF, at a constant
    failure rate.

    The unit time (summed over the units, failed ones repaired or replaced)
    a time-terminated test needs, with at most ``failures`` failures, to
    demonstrate ``mtbf`` at ``confidence``:
    ``mtbf * chi2.ppf(confidence, 2 * failures + 2) / 2``.

    Parameters
    ----------
    mtbf : float
        The mean time between failures to demonstrate, positive.
    confidence : float, optional
        The confidence level, in (0, 1), by default 0.95.
    failures : int, optional
        The failures the test allows, by default 0.

    Returns
    -------
    float
        The total test time, in the MTBF's units.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    With no failures, about 3 MTBFs of testing demonstrate the MTBF at 95%
    confidence:

    >>> from repyability import mtbf_test_time
    >>> round(mtbf_test_time(1000.0), 1)
    2995.7
    >>> round(mtbf_test_time(1000.0, failures=2), 1)
    6295.8
    """
    _check_positive(mtbf, "mtbf")
    _check_probability(confidence, "confidence")
    failures = _check_failures(failures)
    return float(mtbf * stats.chi2.ppf(confidence, 2 * failures + 2) / 2.0)


def demonstrated_mtbf(
    test_time: float, confidence: float = 0.95, failures: int = 0
) -> float:
    """The MTBF a finished time-terminated test demonstrated, at a constant
    failure rate.

    The lower one-sided confidence bound on the MTBF from a total test time
    with ``failures`` failures: ``2 * test_time / chi2.ppf(confidence,
    2 * failures + 2)``.

    Parameters
    ----------
    test_time : float
        The total unit time tested, positive.
    confidence : float, optional
        The confidence level, in (0, 1), by default 0.95.
    failures : int, optional
        The failures seen, by default 0.

    Returns
    -------
    float
        The demonstrated MTBF, in the test time's units.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    >>> from repyability import demonstrated_mtbf
    >>> round(demonstrated_mtbf(2995.7), 1)
    1000.0
    """
    _check_positive(test_time, "test_time")
    _check_probability(confidence, "confidence")
    failures = _check_failures(failures)
    return float(
        2.0 * test_time / stats.chi2.ppf(confidence, 2 * failures + 2)
    )


def mtbf_pass_probability(
    mtbf: float, test_time: float, failures: int = 0
) -> float:
    """The chance that a design of a given true MTBF passes a
    time-terminated test, at a constant failure rate.

    The probability of at most ``failures`` failures in ``test_time`` when
    failures come at rate ``1 / mtbf``: the Poisson distribution's, the
    test's operating characteristic.

    Parameters
    ----------
    mtbf : float
        The design's true mean time between failures, positive.
    test_time : float
        The total unit time tested, positive.
    failures : int, optional
        The failures the test allows, by default 0.

    Returns
    -------
    float
        The probability of passing.

    Raises
    ------
    ValueError
        If an argument is out of range.

    Examples
    --------
    A test planned to demonstrate an MTBF of 1000 passes a design at that
    MTBF one time in twenty, and one with three times the MTBF only a
    little more than one time in three:

    >>> from repyability import mtbf_pass_probability, mtbf_test_time
    >>> time = mtbf_test_time(1000.0)
    >>> round(mtbf_pass_probability(1000.0, time), 3)
    0.05
    >>> round(mtbf_pass_probability(3000.0, time), 3)
    0.368
    """
    _check_positive(mtbf, "mtbf")
    _check_positive(test_time, "test_time")
    failures = _check_failures(failures)
    return float(stats.poisson.cdf(failures, test_time / mtbf))


def _test_reliability(
    reliability: float, test_multiple: float, shape: Optional[float]
) -> float:
    """A unit's reliability over the whole test."""
    return float(reliability ** _test_exponent(test_multiple, shape))


def _test_exponent(test_multiple: float, shape: Optional[float]) -> float:
    """The power a mission's reliability is raised to over a test of
    ``test_multiple`` missions: ``test_multiple ** shape``."""
    if not (np.isfinite(test_multiple) and test_multiple > 0.0):
        raise ValueError(
            f"test_multiple must be positive, got {test_multiple!r}."
        )
    if test_multiple == 1.0:
        return 1.0
    if shape is None:
        raise ValueError(
            "A test longer or shorter than a mission needs the lifetime's "
            "Weibull shape: give shape."
        )
    _check_shape(shape)
    return float(test_multiple**shape)


def _check_probability(value: float, name: str) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value!r}.")


def _check_positive(value: float, name: str) -> None:
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"{name} must be positive, got {value!r}.")


def _check_shape(shape: float) -> None:
    if not (np.isfinite(shape) and shape > 0.0):
        raise ValueError(f"shape must be positive, got {shape!r}.")


def _check_count(n) -> int:
    if int(n) != n or n < 1:
        raise ValueError(f"n must be a positive whole number, got {n!r}.")
    return int(n)


def _check_failures(failures) -> int:
    if int(failures) != failures or failures < 0:
        raise ValueError(
            f"failures must be a whole number, 0 or more, got {failures!r}."
        )
    return int(failures)
