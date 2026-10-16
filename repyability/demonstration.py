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
  A test is time-terminated (it stops at its time) unless
  ``failure_terminated`` says it stops at its ``failures``-th failure.

A test has two risks: the consumer's, that a design no better than the
target passes (at most ``1 - confidence``), and the producer's, that a good
design fails. ``demonstration_plan`` and ``mtbf_demonstration_plan`` design
the smallest test that keeps both (#184).

The functions plan tests from targets and results; they fit nothing (surpyval
does that). Each but the two plans takes arrays for its numbers too, and
answers for each element, broadcast together (#179).
"""

import functools
import inspect
from typing import Callable, Optional, TypeVar

import numpy as np
from scipy import stats

from repyability.rbd.results import DemonstrationPlan
from repyability.utils.checks import is_whole

_Function = TypeVar("_Function", bound=Callable)


def _elementwise(function: _Function) -> _Function:
    """``function``, of numbers, answering for each element of the
    arguments given as arrays, broadcast together: an array of its answers,
    or its answer when every argument is a number."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arrays = {
            name: np.asarray(value)
            for name, value in bound.arguments.items()
            if value is not None and np.ndim(value) > 0
        }
        if not arrays:
            return function(*bound.args, **bound.kwargs)
        shaped = dict(zip(arrays, np.broadcast_arrays(*arrays.values())))
        shape = next(iter(shaped.values())).shape
        answers = [
            function(
                **{
                    **bound.arguments,
                    **{name: a[index].item() for name, a in shaped.items()},
                }
            )
            for index in np.ndindex(shape)
        ]
        return np.array(answers).reshape(shape)

    return wrapper  # type: ignore[return-value]


@_elementwise
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


@_elementwise
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


@_elementwise
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


@_elementwise
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


@_elementwise
def mtbf_test_time(
    mtbf: float,
    confidence: float = 0.95,
    failures: int = 0,
    *,
    failure_terminated: bool = False,
) -> float:
    """The total test time that demonstrates an MTBF, at a constant
    failure rate.

    The unit time (summed over the units, failed ones repaired or replaced)
    a time-terminated test needs, with at most ``failures`` failures, to
    demonstrate ``mtbf`` at ``confidence``:
    ``mtbf * chi2.ppf(confidence, 2 * failures + 2) / 2``. A
    failure-terminated test, which runs until its ``failures``-th failure,
    demonstrates ``mtbf`` if that failure comes after
    ``mtbf * chi2.ppf(confidence, 2 * failures) / 2``: the time of a
    time-terminated test that allows one failure fewer.

    Parameters
    ----------
    mtbf : float
        The mean time between failures to demonstrate, positive.
    confidence : float, optional
        The confidence level, in (0, 1), by default 0.95.
    failures : int, optional
        The failures the test allows, by default 0; with
        ``failure_terminated``, the failure it stops at, from 1.
    failure_terminated : bool, optional
        Whether the test stops at its ``failures``-th failure rather than
        at a time, by default False.

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
    >>> round(mtbf_test_time(1000.0, failures=3, failure_terminated=True), 1)
    6295.8
    """
    _check_positive(mtbf, "mtbf")
    _check_probability(confidence, "confidence")
    failures = _check_failures(failures)
    freedom = _degrees_of_freedom(failures, failure_terminated)
    return float(mtbf * stats.chi2.ppf(confidence, freedom) / 2.0)


@_elementwise
def demonstrated_mtbf(
    test_time: float,
    confidence: float = 0.95,
    failures: int = 0,
    *,
    failure_terminated: bool = False,
) -> float:
    """The MTBF a finished test demonstrated, at a constant failure rate.

    The lower one-sided confidence bound on the MTBF from a total test time
    with ``failures`` failures. A test stopped at its time
    (time-terminated) gives ``2 * test_time / chi2.ppf(confidence,
    2 * failures + 2)``; one stopped at its ``failures``-th failure
    (``failure_terminated``), with ``2 * failures`` degrees of freedom
    instead.

    Parameters
    ----------
    test_time : float
        The total unit time tested, positive: with ``failure_terminated``,
        up to the last failure.
    confidence : float, optional
        The confidence level, in (0, 1), by default 0.95.
    failures : int, optional
        The failures seen, by default 0 (at least 1 with
        ``failure_terminated``).
    failure_terminated : bool, optional
        Whether the test stopped at its last failure rather than at a
        time, by default False.

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

    Three failures, the test stopped at the third, 6,295.8 hours in all:

    >>> mtbf = demonstrated_mtbf(6295.8, failures=3, failure_terminated=True)
    >>> round(mtbf, 1)
    1000.0
    """
    _check_positive(test_time, "test_time")
    _check_probability(confidence, "confidence")
    failures = _check_failures(failures)
    freedom = _degrees_of_freedom(failures, failure_terminated)
    return float(2.0 * test_time / stats.chi2.ppf(confidence, freedom))


@_elementwise
def mtbf_pass_probability(
    mtbf: float,
    test_time: float,
    failures: int = 0,
    *,
    failure_terminated: bool = False,
) -> float:
    """The chance that a design of a given true MTBF passes a test, at a
    constant failure rate.

    The probability of at most ``failures`` failures in ``test_time`` when
    failures come at rate ``1 / mtbf``: the Poisson distribution's, the
    test's operating characteristic. A failure-terminated test passes if
    its ``failures``-th failure comes after ``test_time``: at most one
    fewer by then.

    Parameters
    ----------
    mtbf : float
        The design's true mean time between failures, positive.
    test_time : float
        The total unit time tested, positive: with ``failure_terminated``,
        the time the last failure must come after.
    failures : int, optional
        The failures the test allows, by default 0; with
        ``failure_terminated``, the failure it stops at, from 1.
    failure_terminated : bool, optional
        Whether the test stops at its ``failures``-th failure, by default
        False.

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
    allowed = _degrees_of_freedom(failures, failure_terminated) // 2 - 1
    return float(stats.poisson.cdf(allowed, test_time / mtbf))


def demonstration_plan(
    reliability: float,
    good_reliability: float,
    confidence: float = 0.95,
    producer_risk: float = 0.2,
    *,
    n: Optional[int] = None,
    test_multiple: float = 1.0,
    shape: Optional[float] = None,
    max_failures: int = 1000,
) -> DemonstrationPlan:
    """The smallest attribute test that demonstrates a reliability and that
    a good design passes.

    A plan's consumer's risk is the chance that a design of the target
    ``reliability`` passes it, at most ``1 - confidence``; its producer's
    risk is the chance that a design of ``good_reliability``, which should
    pass, fails it, at most ``producer_risk``. A test allowing no
    failures keeps the first with the fewest units, but often fails a good
    design: allowing failures, with more units, keeps both. For each number
    of failures allowed, from none, the fewest units that keep the
    consumer's risk (``demonstration_sample_size``) pass the good design
    most often, and the first of those plans that keeps the producer's risk
    tests the fewest units of all.

    Given ``n``, the test length is searched instead, with the lifetime's
    Weibull ``shape``: the shortest test of ``n`` units that keeps both
    risks (each failure allowed takes a longer test, as
    ``demonstration_test_multiple`` gives it).

    Parameters
    ----------
    reliability : float
        The reliability to demonstrate over one mission, in (0, 1).
    good_reliability : float
        The reliability of a design that should pass, more than
        ``reliability`` and less than 1.
    confidence : float, optional
        The confidence level, in (0, 1), by default 0.95: the consumer's
        risk is at most ``1 - confidence``.
    producer_risk : float, optional
        The most chance of failing a good design, in (0, 1), by default
        0.2.
    n : int, optional
        The number of units to test, to search the test length instead of
        the number of units (with ``shape``).
    test_multiple : float, optional
        Without ``n``: how many missions long each unit's test is, by
        default 1.0 (another length needs ``shape``).
    shape : float, optional
        The Weibull shape of the lifetime, needed for a test longer or
        shorter than a mission, and with ``n``.
    max_failures : int, optional
        The most failures a plan may allow, by default 1000.

    Returns
    -------
    DemonstrationPlan
        The plan: ``n``, ``test_multiple`` and ``failures``, and its risks.

    Raises
    ------
    ValueError
        If an argument is out of range, ``good_reliability`` is no more
        than ``reliability``, ``n`` is given without ``shape``, or no plan
        within ``max_failures`` (or, given ``n``, fewer failures than
        units) keeps both risks.

    Examples
    --------
    Demonstrating 90% reliability at 90% confidence, so that a design of
    95% passes at least 80% of the time: a test of 22 units with no
    failures would pass that design only a third of the time, and the plan
    tests 128, allowing 8 to fail:

    >>> from repyability import demonstration_plan
    >>> from repyability import demonstration_pass_probability
    >>> round(demonstration_pass_probability(0.95, 22), 2)
    0.32
    >>> plan = demonstration_plan(0.9, 0.95, confidence=0.9)
    >>> plan.n, plan.failures
    (128, 8)
    >>> round(plan.consumer_risk, 3), round(plan.producer_risk, 3)
    (0.097, 0.192)
    """
    _check_probability(reliability, "reliability")
    _check_probability(good_reliability, "good_reliability")
    _check_probability(confidence, "confidence")
    _check_probability(producer_risk, "producer_risk")
    if good_reliability <= reliability:
        raise ValueError(
            "good_reliability must be more than the reliability to "
            f"demonstrate ({reliability!r}), got {good_reliability!r}: no "
            "test can pass a design no better than the target more often "
            "than the target."
        )
    most = _check_failures(max_failures)
    if n is not None:
        n = _check_count(n)
        if shape is None:
            raise ValueError(
                "Given n, the test length is searched: give the lifetime's "
                "Weibull shape."
            )
        _check_shape(shape)
        for failures in range(min(most, n - 1) + 1):
            multiple = demonstration_test_multiple(
                reliability, n, confidence, failures, shape=shape
            )
            plan = _attribute_plan(
                reliability, good_reliability, n, failures, multiple, shape
            )
            if plan.producer_risk <= producer_risk:
                return plan
        raise ValueError(
            f"No test of {n} units keeps both risks: test more units, or "
            "allow a larger producer's risk."
        )
    _test_exponent(test_multiple, shape)
    for failures in range(most + 1):
        count = demonstration_sample_size(
            reliability,
            confidence,
            failures,
            test_multiple=test_multiple,
            shape=shape,
        )
        plan = _attribute_plan(
            reliability,
            good_reliability,
            count,
            failures,
            test_multiple,
            shape,
        )
        if plan.producer_risk <= producer_risk:
            return plan
    raise ValueError(
        f"No plan allowing at most {most} failures keeps both risks: the "
        "good reliability is too close to the target (raise max_failures, "
        "or allow larger risks)."
    )


def _attribute_plan(
    reliability, good_reliability, n, failures, test_multiple, shape
) -> DemonstrationPlan:
    """An attribute test's plan, with its risks."""
    return DemonstrationPlan(
        n=int(n),
        test_multiple=float(test_multiple),
        test_time=None,
        failures=int(failures),
        consumer_risk=demonstration_pass_probability(
            reliability, n, failures, test_multiple=test_multiple, shape=shape
        ),
        producer_risk=1.0
        - demonstration_pass_probability(
            good_reliability,
            n,
            failures,
            test_multiple=test_multiple,
            shape=shape,
        ),
    )


def mtbf_demonstration_plan(
    mtbf: float,
    good_mtbf: float,
    confidence: float = 0.95,
    producer_risk: float = 0.2,
    *,
    max_failures: int = 1000,
) -> DemonstrationPlan:
    """The shortest constant-failure-rate test that demonstrates an MTBF and
    that a good design passes.

    The total test time and the failures it allows, for a time-terminated
    test, such that a design of the target ``mtbf`` passes with a chance of
    at most ``1 - confidence`` (the consumer's risk) and one of
    ``good_mtbf`` fails with a chance of at most ``producer_risk``:
    the fixed-length test plans of MIL-HDBK-781, for a discrimination ratio
    ``good_mtbf / mtbf``. For each number of failures allowed, from none,
    the shortest test that keeps the consumer's risk (``mtbf_test_time``)
    passes the good design most often, and the first that keeps the
    producer's risk is the shortest of all. Both risks are kept, where
    some of the handbook's plans exceed one a little for a shorter test;
    those that keep both are these (plans XI-D, XV-D and XVII-D).

    Parameters
    ----------
    mtbf : float
        The MTBF to demonstrate, positive.
    good_mtbf : float
        The MTBF of a design that should pass, more than ``mtbf``.
    confidence : float, optional
        The confidence level, in (0, 1), by default 0.95.
    producer_risk : float, optional
        The most chance of failing a good design, in (0, 1), by default
        0.2.
    max_failures : int, optional
        The most failures a plan may allow, by default 1000.

    Returns
    -------
    DemonstrationPlan
        The plan: ``test_time`` and ``failures``, and its risks.

    Raises
    ------
    ValueError
        If an argument is out of range, ``good_mtbf`` is no more than
        ``mtbf``, or no plan within ``max_failures`` keeps both risks.

    Examples
    --------
    Demonstrating an MTBF of 1000 hours at 90% confidence, so that a design
    of 2000 passes at least 80% of the time (a discrimination ratio of 2):
    the test runs about 14,200 unit hours, and passes with at most 9
    failures:

    >>> from repyability import mtbf_demonstration_plan
    >>> plan = mtbf_demonstration_plan(1000.0, 2000.0, confidence=0.9)
    >>> round(plan.test_time), plan.failures
    (14206, 9)
    >>> round(plan.consumer_risk, 3), round(plan.producer_risk, 3)
    (0.1, 0.18)
    """
    _check_positive(mtbf, "mtbf")
    _check_positive(good_mtbf, "good_mtbf")
    _check_probability(confidence, "confidence")
    _check_probability(producer_risk, "producer_risk")
    if good_mtbf <= mtbf:
        raise ValueError(
            f"good_mtbf must be more than the MTBF to demonstrate ({mtbf!r}), "
            f"got {good_mtbf!r}."
        )
    most = _check_failures(max_failures)
    for failures in range(most + 1):
        time = mtbf_test_time(mtbf, confidence, failures)
        missed = 1.0 - mtbf_pass_probability(good_mtbf, time, failures)
        if missed <= producer_risk:
            return DemonstrationPlan(
                n=None,
                test_multiple=None,
                test_time=float(time),
                failures=failures,
                consumer_risk=mtbf_pass_probability(mtbf, time, failures),
                producer_risk=float(missed),
            )
    raise ValueError(
        f"No plan allowing at most {most} failures keeps both risks: the "
        "good MTBF is too close to the target (raise max_failures, or allow "
        "larger risks)."
    )


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
    if not is_whole(n) or n < 1:
        raise ValueError(f"n must be a positive whole number, got {n!r}.")
    return int(n)


def _check_failures(failures) -> int:
    if not is_whole(failures) or failures < 0:
        raise ValueError(
            f"failures must be a whole number, 0 or more, got {failures!r}."
        )
    return int(failures)


def _degrees_of_freedom(failures: int, failure_terminated) -> int:
    """The chi-squared degrees of freedom of a test with ``failures``
    failures: ``2 * failures + 2`` if it stopped at its time, ``2 *
    failures`` at its last failure."""
    if not isinstance(failure_terminated, (bool, np.bool_)):
        raise ValueError(
            "failure_terminated must be True or False, got "
            f"{failure_terminated!r}."
        )
    if not failure_terminated:
        return 2 * failures + 2
    if failures < 1:
        raise ValueError(
            "A failure-terminated test stops at a failure: failures must be "
            "at least 1."
        )
    return 2 * failures
