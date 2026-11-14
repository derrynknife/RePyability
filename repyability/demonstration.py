"""Reliability demonstration test planning (#129), deprecated: use
``surpyval.test_planning``.

A demonstration test shows, at a confidence level, that a design meets a
reliability target: test ``n`` units for a time, allow ``failures`` of them,
and if no more fail, the design has demonstrated the target. Planning such
a test is test planning, which surpyval owns: ``surpyval.test_planning``
has these nine functions under the same names (with the sequential and
group-sequential tests beside them), and each function here is now a call
to surpyval's that warns with a ``FutureWarning``. They are removed in 0.18.

surpyval's take ``alpha_ci``, the consumer's risk, where these take
``confidence``: ``alpha_ci = 1 - confidence``, given by keyword, after the
other numbers (``demonstration_sample_size(0.95, 0.9, 1)`` here is
``demonstration_sample_size(0.95, 1, alpha_ci=0.1)`` there). Their numbers
are these functions' to the last bit or two (the ``1 - confidence`` taken
and given back), and a plan is the same plan: these give it as
RePyability's ``DemonstrationPlan`` (a read-only mapping, as the other
results are), surpyval as its own (with ``to_dict``).

- **Attribute (binomial) tests.** ``demonstration_sample_size``,
  ``demonstrated_reliability``, ``demonstration_test_multiple`` and
  ``demonstration_pass_probability``; extended (Weibayes) tests with
  ``test_multiple`` and the Weibull ``shape``.
- **Constant failure rate (chi-squared) tests.** ``mtbf_test_time``,
  ``demonstrated_mtbf`` and ``mtbf_pass_probability``, time- or
  failure-terminated.
- **Plans that keep both risks** (#184). ``demonstration_plan`` and
  ``mtbf_demonstration_plan``.

Each but the two plans takes arrays for its numbers too, and answers for
each element, broadcast together (#179).
"""

import functools
import inspect
import warnings
from typing import Any, Callable, Optional, TypeVar

import numpy as np
from surpyval import test_planning

from repyability.rbd.results import DemonstrationPlan
from repyability.utils import deprecation

_Function = TypeVar("_Function", bound=Callable)


def _moved(function: _Function) -> _Function:
    """``function``, warning that it is deprecated for surpyval's of the
    same name in ``surpyval.test_planning``."""
    name = function.__name__
    why = "test planning is surpyval's"
    if "confidence" in inspect.signature(function).parameters:
        why += " (give alpha_ci=1 - confidence there, by keyword)"

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        deprecation.moved(
            f"repyability.{name}", f"surpyval.test_planning.{name}", why
        )
        return function(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _alpha_ci(confidence: Any) -> Any:
    """``1 - confidence``, surpyval's ``alpha_ci``, each confidence checked
    to be between 0 and 1 (as these functions said it)."""
    levels = np.asarray(confidence)
    for level in np.unique(levels):
        value = level.item()
        try:
            inside = not isinstance(value, bool) and 0.0 < value < 1.0
        except TypeError:
            inside = False
        if not inside:
            raise ValueError(
                f"confidence must be between 0 and 1, got {value!r}."
            )
    return 1.0 - levels if levels.ndim else 1.0 - float(confidence)


def _call(function: Callable, *args, alpha_ci: Any = None, **kwargs) -> Any:
    """surpyval's ``function`` with these arguments, without its warning of
    an ``alpha_ci`` above 0.5: a confidence below 0.5 was given as such."""
    if alpha_ci is None:
        return function(*args, **kwargs)
    if np.all(np.asarray(alpha_ci) <= 0.5):
        return function(*args, alpha_ci=alpha_ci, **kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="alpha_ci is the significance level",
            category=UserWarning,
        )
        return function(*args, alpha_ci=alpha_ci, **kwargs)


def _plan(plan: Any) -> DemonstrationPlan:
    """surpyval's ``DemonstrationPlan`` as RePyability's."""
    return DemonstrationPlan(
        n=plan.n,
        test_multiple=plan.test_multiple,
        test_time=plan.test_time,
        failures=plan.failures,
        consumer_risk=plan.consumer_risk,
        producer_risk=plan.producer_risk,
    )


@_moved
def demonstration_sample_size(
    reliability: float,
    confidence: float = 0.95,
    failures: int = 0,
    *,
    test_multiple: float = 1.0,
    shape: Optional[float] = None,
) -> int:
    """The fewest units whose test demonstrates a reliability.

    Deprecated: use ``surpyval.test_planning.demonstration_sample_size``
    (``alpha_ci=1 - confidence``); removed in 0.18.

    The smallest ``n`` for which ``n`` units tested, with at most
    ``failures`` of them failing, demonstrate ``reliability`` over a mission
    at ``confidence``. With no failures this is the success run. Testing
    each unit for ``test_multiple`` missions, with the lifetime's Weibull
    ``shape`` known, needs fewer units (Weibayes).

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
    surpyval's call: 59 units with no failures demonstrate 95% reliability
    at 95% confidence, and allowing one failure takes 93:

    >>> from surpyval.test_planning import demonstration_sample_size
    >>> demonstration_sample_size(0.95, alpha_ci=0.05)
    59
    >>> demonstration_sample_size(0.95, 1, alpha_ci=0.05)
    93
    """
    return _call(
        test_planning.demonstration_sample_size,
        reliability,
        failures,
        alpha_ci=_alpha_ci(confidence),
        test_multiple=test_multiple,
        shape=shape,
    )


@_moved
def demonstrated_reliability(
    n: int,
    confidence: float = 0.95,
    failures: int = 0,
    *,
    test_multiple: float = 1.0,
    shape: Optional[float] = None,
) -> float:
    """The reliability a finished test demonstrated.

    Deprecated: use ``surpyval.test_planning.demonstrated_reliability``
    (``alpha_ci=1 - confidence``); removed in 0.18.

    The Clopper-Pearson lower one-sided bound on the reliability over a
    mission, from ``n`` units tested with ``failures`` failing; with no
    failures, ``(1 - confidence) ** (1 / n)``, surpyval's ``success_run``.

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
    surpyval's call:

    >>> from surpyval.test_planning import demonstrated_reliability
    >>> round(demonstrated_reliability(59, alpha_ci=0.05), 4)
    0.9505
    """
    return _call(
        test_planning.demonstrated_reliability,
        n,
        failures,
        alpha_ci=_alpha_ci(confidence),
        test_multiple=test_multiple,
        shape=shape,
    )


@_moved
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

    Deprecated: use ``surpyval.test_planning.demonstration_test_multiple``
    (``alpha_ci=1 - confidence``); removed in 0.18.

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
    surpyval's call: twenty units, for a wear-out shape of 2, each run
    about 1.7 missions to demonstrate 95% reliability at 95% confidence:

    >>> from surpyval.test_planning import demonstration_test_multiple
    >>> round(demonstration_test_multiple(0.95, 20, shape=2.0), 3)
    1.709
    """
    return _call(
        test_planning.demonstration_test_multiple,
        reliability,
        n,
        failures,
        alpha_ci=_alpha_ci(confidence),
        shape=shape,
    )


@_moved
def demonstration_pass_probability(
    reliability: float,
    n: int,
    failures: int = 0,
    *,
    test_multiple: float = 1.0,
    shape: Optional[float] = None,
) -> float:
    """The chance that a design of a given true reliability passes a test.

    Deprecated: use
    ``surpyval.test_planning.demonstration_pass_probability`` (the same
    arguments); removed in 0.18.

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
    surpyval's call:

    >>> from surpyval.test_planning import demonstration_pass_probability
    >>> round(demonstration_pass_probability(0.95, 59), 3)
    0.048
    """
    return test_planning.demonstration_pass_probability(
        reliability, n, failures, test_multiple=test_multiple, shape=shape
    )


@_moved
def mtbf_test_time(
    mtbf: float,
    confidence: float = 0.95,
    failures: int = 0,
    *,
    failure_terminated: bool = False,
) -> float:
    """The total test time that demonstrates an MTBF, at a constant
    failure rate.

    Deprecated: use ``surpyval.test_planning.mtbf_test_time``
    (``alpha_ci=1 - confidence``); removed in 0.18.

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
    surpyval's call:

    >>> from surpyval.test_planning import mtbf_test_time
    >>> round(mtbf_test_time(1000.0), 1)
    2995.7
    """
    return _call(
        test_planning.mtbf_test_time,
        mtbf,
        failures,
        alpha_ci=_alpha_ci(confidence),
        failure_terminated=failure_terminated,
    )


@_moved
def demonstrated_mtbf(
    test_time: float,
    confidence: float = 0.95,
    failures: int = 0,
    *,
    failure_terminated: bool = False,
) -> float:
    """The MTBF a finished test demonstrated, at a constant failure rate.

    Deprecated: use ``surpyval.test_planning.demonstrated_mtbf``
    (``alpha_ci=1 - confidence``); removed in 0.18.

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
    surpyval's call:

    >>> from surpyval.test_planning import demonstrated_mtbf
    >>> round(demonstrated_mtbf(2995.7), 1)
    1000.0
    """
    return _call(
        test_planning.demonstrated_mtbf,
        test_time,
        failures,
        alpha_ci=_alpha_ci(confidence),
        failure_terminated=failure_terminated,
    )


@_moved
def mtbf_pass_probability(
    mtbf: float,
    test_time: float,
    failures: int = 0,
    *,
    failure_terminated: bool = False,
) -> float:
    """The chance that a design of a given true MTBF passes a test, at a
    constant failure rate.

    Deprecated: use ``surpyval.test_planning.mtbf_pass_probability`` (the
    same arguments); removed in 0.18.

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
    surpyval's call:

    >>> from surpyval.test_planning import mtbf_pass_probability
    >>> round(mtbf_pass_probability(1000.0, 2995.7), 3)
    0.05
    """
    return test_planning.mtbf_pass_probability(
        mtbf, test_time, failures, failure_terminated=failure_terminated
    )


@_moved
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

    Deprecated: use ``surpyval.test_planning.demonstration_plan``
    (``alpha_ci=1 - confidence``, and the rest by keyword); removed in
    0.18. surpyval's returns its own ``DemonstrationPlan``, with the same
    fields.

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
    surpyval's call: demonstrating 90% reliability at 90% confidence, so
    that a design of 95% passes at least 80% of the time, tests 128 units,
    allowing 8 to fail:

    >>> from surpyval.test_planning import demonstration_plan
    >>> plan = demonstration_plan(0.9, 0.95, alpha_ci=0.1)
    >>> plan.n, plan.failures
    (128, 8)
    """
    return _plan(
        _call(
            test_planning.demonstration_plan,
            reliability,
            good_reliability,
            alpha_ci=_alpha_ci(confidence),
            producer_risk=producer_risk,
            n=n,
            test_multiple=test_multiple,
            shape=shape,
            max_failures=max_failures,
        )
    )


@_moved
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

    Deprecated: use ``surpyval.test_planning.mtbf_demonstration_plan``
    (``alpha_ci=1 - confidence``, and the rest by keyword); removed in
    0.18. surpyval's returns its own ``DemonstrationPlan``, with the same
    fields.

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
    surpyval's call: an MTBF of 1000 hours at 90% confidence, so that a
    design of 2000 passes at least 80% of the time:

    >>> from surpyval.test_planning import mtbf_demonstration_plan
    >>> plan = mtbf_demonstration_plan(1000.0, 2000.0, alpha_ci=0.1)
    >>> round(plan.test_time), plan.failures
    (14206, 9)
    """
    return _plan(
        _call(
            test_planning.mtbf_demonstration_plan,
            mtbf,
            good_mtbf,
            alpha_ci=_alpha_ci(confidence),
            producer_risk=producer_risk,
            max_failures=max_failures,
        )
    )
