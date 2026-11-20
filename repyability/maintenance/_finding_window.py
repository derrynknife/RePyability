"""A hidden failure tested at given times, over a window from the device's
age (UC-53 of this package's ``README.md``).

``failure_finding`` gives the long run of one even interval. Here the
tests are at any times, the device starts working at an age, and the
answer is over a window: the chance each test finds the device failed,
the time it is failed, the demands that meet it failed, and what the
plan costs.

**The model.** The device is working, at an age since it was last made
new, or failed without anyone knowing. A test of a working device leaves
its age; one that finds it failed makes it new, in no time. Demands come
at a rate ``d`` (a Poisson process, independent of the device), and one
that meets the device failed costs the consequence; with
``demands_reveal`` it also shows the failure, which is repaired as new
then.

**How it is worked out.** ``A(t)``, the chance the device is working at
``t``, is the sum over the times it was made new of the chance of that
and of surviving since::

    A(t) = R(age + t) / R(age)
           + sum_{tests t_k < t} q_k R(t - t_k)
           + d integral_0^t U(s) R(t - s) ds,     U = 1 - A,

with ``q_k = U(t_k)`` just before the test (what it finds), and the last
term only where demands reveal. Without it ``A`` is exact at any time
(a sum over the tests so far). With it, this is a Volterra equation of
the second kind, solved on a grid of each stretch between tests by the
trapezoid rule. Either way the integrals of ``U`` are the trapezoid
rule's on that grid, the whole worked out at one grid and at twice as
fine, and every number taken from the two by Richardson's step (``fine +
(fine - coarse) / 3``, the trapezoid's error falling four-fold); the
difference of the two is the precision given, and the grid is halved
until that is within the tolerance.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import numpy as np

from repyability.rbd.results import _ResultMapping
from repyability.utils import checks
from repyability.utils.vectors import dot

from ._decisions import _check_protective, _number, _prices, _survival_at

#: The most tests in a window: the work grows with their square.
MAX_TESTS = 2000
#: How close the failed time and the failures found are worked out, as a
#: share of each, by default.
TOLERANCE = 1e-6
#: Steps across the window at the coarsest grid, and the least in a
#: stretch between two tests.
_STEPS = 32
_LEAST = 2
#: The most survival values a grid may take (its work); a finer grid is
#: not tried, and the precision says how close the last one is.
_WORK = 4.0e7


@dataclass(frozen=True)
class _Pass:
    """The device's chance of being failed on one grid: at the nodes of
    each stretch between tests (a test's time is the end of one stretch,
    before the test, and the start of the next, after it)."""

    left: np.ndarray  # each step's start and end, and the values there
    right: np.ndarray
    down_left: np.ndarray
    down_right: np.ndarray
    cum_left: np.ndarray  # the failed time to each step's start and end
    cum_right: np.ndarray
    found: np.ndarray  # the chance each test finds the device failed
    nodes: np.ndarray
    weights: np.ndarray
    down: np.ndarray

    def _step(self, times: np.ndarray, side: Literal["left", "right"]):
        k = np.searchsorted(self.left, times, side=side) - 1
        k = np.clip(k, 0, self.left.size - 1)
        width = self.right[k] - self.left[k]
        return k, width, np.clip((times - self.left[k]) / width, 0.0, 1.0)

    def failed_time(self, times: np.ndarray) -> np.ndarray:
        """The integral of the chance of being failed from 0 to each of
        ``times``: the cubic through each step's ends with their slopes
        (the chances there)."""
        k, h, s = self._step(times, "right")
        return (
            (2 * s**3 - 3 * s**2 + 1) * self.cum_left[k]
            + (s**3 - 2 * s**2 + s) * h * self.down_left[k]
            + (3 * s**2 - 2 * s**3) * self.cum_right[k]
            + (s**3 - s**2) * h * self.down_right[k]
        )

    def unavailability(self, times: np.ndarray) -> np.ndarray:
        """The chance of being failed at each of ``times``: at a test's
        time, just before it."""
        k, _, s = self._step(times, "left")
        return (1.0 - s) * self.down_left[k] + s * self.down_right[k]


def _solve(
    survival: Callable,
    age: float,
    tests: np.ndarray,
    window: float,
    reveal: float,
    counts: np.ndarray,
) -> _Pass:
    """One grid's pass: ``tests`` the distinct test times in ``(0,
    window]``, ``reveal`` the rate at which a failed device is shown and
    repaired (the demand rate, or 0), ``counts`` each stretch's steps."""
    inside = tests[tests < window]
    edges = np.concatenate([[0.0], inside, [window]])
    stretches = edges.size - 1
    nodes_: list = []
    weights_: list = []
    stretch_: list = []
    first_: list = []
    for m in range(stretches):
        n = int(counts[m])
        h = (edges[m + 1] - edges[m]) / n
        weights = np.full(n + 1, h)
        weights[0] = weights[-1] = 0.5 * h
        first_.append(sum(len(x) for x in nodes_))
        nodes_.append(np.linspace(edges[m], edges[m + 1], n + 1))
        weights_.append(weights)
        stretch_.append(np.full(n + 1, m))
    nodes = np.concatenate(nodes_)
    weights = np.concatenate(weights_)
    stretch = np.concatenate(stretch_)
    first = set(first_)
    up = survival(age + nodes) / survival(np.array([age]))[0]
    down = np.zeros(nodes.size)
    mass = np.zeros(inside.size)  # the chance of being made new at a test
    flow = np.zeros(nodes.size)  # weights x the rate of being made new
    for i in range(nodes.size):
        m = int(stretch[i])
        value = float(up[i])
        if m > 0:
            value += dot(mass[:m], survival(nodes[i] - inside[:m]))
        if reveal > 0.0 and i > 0:
            value += dot(flow[:i], survival(nodes[i] - nodes[:i]))
            if i not in first:
                # This node's own half step of the integral.
                half = reveal * weights[i]
                if i + 1 < nodes.size and stretch[i + 1] == m:
                    half *= 0.5
                value = (value + half) / (1.0 + half)
        down[i] = min(max(1.0 - value, 0.0), 1.0)
        if i in first:
            # Just after a test (or at the start) the device is working.
            down[i] = 0.0
        flow[i] = reveal * weights[i] * down[i]
        if i + 1 < nodes.size and stretch[i + 1] != m:
            mass[m] = down[i]
    inner = np.flatnonzero(stretch[:-1] == stretch[1:])
    areas = 0.5 * (nodes[inner + 1] - nodes[inner])
    areas = areas * (down[inner] + down[inner + 1])
    cum = np.concatenate([[0.0], np.cumsum(areas)])
    found = mass
    if tests.size > inside.size:  # a test at the window's end
        found = np.append(mass, down[-1])
    return _Pass(
        left=nodes[inner],
        right=nodes[inner + 1],
        down_left=down[inner],
        down_right=down[inner + 1],
        cum_left=cum[:-1],
        cum_right=cum[1:],
        found=found,
        nodes=nodes,
        weights=weights,
        down=down,
    )


@dataclass(frozen=True)
class _Solved:
    """A device's hidden failures over a window, from two grids."""

    window: float
    tests: np.ndarray
    fine: Optional[_Pass]
    coarse: Optional[_Pass]

    def _both(self, of: Callable[[_Pass], np.ndarray]) -> np.ndarray:
        assert self.fine is not None and self.coarse is not None
        fine = of(self.fine)
        return fine + (fine - of(self.coarse)) / 3.0

    @property
    def found(self) -> np.ndarray:
        """The chance each test finds the device failed."""
        if self.fine is None:
            return np.zeros(self.tests.size)
        return np.clip(self._both(lambda p: p.found), 0.0, 1.0)

    def failed_time(self, times) -> np.ndarray:
        """The expected time failed from 0 to each of ``times``."""
        times = np.clip(np.asarray(times, float), 0.0, self.window)
        if self.fine is None:
            return np.zeros(times.shape)
        flat = times.ravel()
        total = self._both(lambda p: p.failed_time(flat))
        return np.maximum(total, 0.0).reshape(times.shape)

    def unavailability(self, times) -> np.ndarray:
        """The chance the device is failed at each of ``times`` (just
        before a test at a test's time)."""
        times = np.clip(np.asarray(times, float), 0.0, self.window)
        if self.fine is None:
            return np.zeros(times.shape)
        flat = times.ravel()
        value = self._both(lambda p: p.unavailability(flat))
        return np.clip(value, 0.0, 1.0).reshape(times.shape)

    def discounted_failed_time(self, rate: float) -> float:
        """The integral of the chance of being failed, discounted at
        ``rate`` a unit of time."""
        if self.fine is None:
            return 0.0
        return float(
            self._both(
                lambda p: np.array(
                    dot(p.weights * np.exp(-rate * p.nodes), p.down)
                )
            )
        )

    @property
    def time_error(self) -> float:
        """How far the failed time over the window can be out."""
        if self.fine is None or self.coarse is None:
            return 0.0
        end = np.array([self.window])
        return float(
            abs(self.fine.failed_time(end) - self.coarse.failed_time(end))[0]
        )

    @property
    def found_error(self) -> float:
        """How far the failures found, in all, can be out."""
        if self.fine is None or self.coarse is None:
            return 0.0
        return float(np.abs(self.fine.found - self.coarse.found).sum())


def _hidden(
    model,
    age: float,
    tests: np.ndarray,
    window: float,
    reveal: float,
    tolerance: float = TOLERANCE,
) -> _Solved:
    """The device's hidden failures over ``window`` from ``age``, tested
    at ``tests`` (distinct, increasing, in ``(0, window]``), a failed
    device shown and repaired at the rate ``reveal`` (0: only tests find
    it)."""
    tests = np.asarray(tests, float)
    if not window > 0.0:
        return _Solved(float(window), tests, None, None)
    if tests.size > MAX_TESTS:
        raise NotImplementedError(
            f"{tests.size} tests in the window: at most {MAX_TESTS} are "
            "worked out (the work grows with their square). Value a "
            "shorter window, or, for one even interval in the long run, "
            "use failure_finding."
        )

    def survival(times) -> np.ndarray:
        return _survival_at(model, times)

    if not survival(np.array([age]))[0] > 0.0:
        raise ValueError(
            f"The device has no chance of reaching age={age:g}: give "
            "the age of a working device."
        )
    edges = np.concatenate([[0.0], tests[tests < window], [window]])
    lengths = np.diff(edges)
    counts = np.maximum(
        _LEAST, np.ceil(lengths / (window / _STEPS)).astype(int)
    )

    def work(steps: np.ndarray) -> float:
        nodes = float(steps.sum() + steps.size)
        return nodes * tests.size + (0.5 * nodes**2 if reveal > 0.0 else 0.0)

    coarse = _solve(survival, age, tests, window, reveal, counts)
    while True:
        counts = 2 * counts
        fine = _solve(survival, age, tests, window, reveal, counts)
        solved = _Solved(float(window), tests, fine, coarse)
        total = float(solved.failed_time(np.array([window]))[0])
        close = solved.time_error <= tolerance * total and (
            solved.found_error <= tolerance * float(solved.found.sum())
        )
        if close or work(2 * counts) > _WORK:
            return solved
        coarse = fine


# -- the public calculation --------------------------------------------------


@dataclass
class FailureFindingWindow(_ResultMapping):
    """A hidden failure tested at given times, over a window: returned by
    [`failure_finding_window`][repyability.failure_finding_window].

    Attributes
    ----------
    window : float
        The window's length, from now.
    age : float
        The device's age at the window's start (working then).
    test_times : numpy.ndarray
        The times of the tests, from the window's start.
    demands_reveal : bool
        Whether a demand that meets the device failed shows the failure,
        which is then repaired as new.
    found_at_tests : numpy.ndarray
        The chance each test finds the device failed (the device's
        unavailability just before it).
    failures_found : float
        The expected failures the tests find (the sum of
        ``found_at_tests``).
    failed_time : float
        The expected time the device is failed in the window.
    mean_unavailability : float
        ``failed_time / window``: the average chance the device is
        failed when a demand comes.
    demands_on_failed : float or None
        The expected demands that meet the device failed (the demand
        rate times ``failed_time``); None without a demand rate.
    times : numpy.ndarray
        The times the unavailability is given at.
    unavailability : numpy.ndarray
        The chance the device is failed at each of ``times`` (at a
        test's time, just before the test).
    tests_cost : float
        The expected cost of the tests, undiscounted.
    repairs_cost : float
        The expected cost of the repairs, of the failures the tests find
        and of those demands show, undiscounted.
    consequence_cost : float
        The expected cost of the demands that meet the device failed,
        undiscounted.
    cost : float
        The sum of the three.
    discount_rate : float
        The discount rate, per unit of time, continuously compounded.
    present_value : float
        ``cost`` discounted to the window's start: each test and the
        repair of what it finds from its time, the demands' cost as it
        accrues. ``cost`` itself at a rate of 0.
    precision : float
        How far ``present_value`` (and ``cost``) can be from the true
        one: the difference between the last two grids' answers.
    """

    window: float
    age: float
    test_times: np.ndarray
    demands_reveal: bool
    found_at_tests: np.ndarray
    failures_found: float
    failed_time: float
    mean_unavailability: float
    demands_on_failed: Optional[float]
    times: np.ndarray
    unavailability: np.ndarray
    tests_cost: float
    repairs_cost: float
    consequence_cost: float
    cost: float
    discount_rate: float
    present_value: float
    precision: float

    def __repr__(self) -> str:
        return (
            f"FailureFindingWindow({self.test_times.size} tests over "
            f"{self.window:g}: mean unavailability "
            f"{self.mean_unavailability:.6g}, failures found "
            f"{self.failures_found:.6g}, cost {self.cost:.6g}, present "
            f"value {self.present_value:.6g})"
        )


def _test_times(test_times, window: float) -> np.ndarray:
    try:
        times = np.atleast_1d(np.asarray(test_times, dtype=float))
    except (TypeError, ValueError):
        times = np.array([np.nan])
    if np.size(test_times) == 0:
        return np.array([], float)
    ok = (
        times.ndim == 1
        and np.all(np.isfinite(times))
        and np.all(times >= 0.0)
        and np.all(times <= window)
        and np.all(np.diff(times) > 0.0)
    )
    if not ok:
        raise ValueError(
            "test_times must be the tests' times from the window's "
            f"start, increasing, each from 0 to window={window:g} (leave "
            f"out the tests after it); got {test_times!r}."
        )
    return times


def failure_finding_window(
    protective,
    test_times,
    window,
    *,
    age=0.0,
    test_cost=0.0,
    repair_cost=0.0,
    demand_rate=None,
    consequence_cost=0.0,
    discount_rate=0.0,
    demands_reveal=True,
    times=None,
    tolerance=TOLERANCE,
) -> FailureFindingWindow:
    """A hidden failure tested at given times, over a window from the
    device's age: what each test finds, how long the device is failed,
    the demands that meet it failed, and what the plan costs and is
    worth.

    In plain terms: a protective device (a relief valve, a trip) fails
    without anyone knowing. It is working now, at ``age``, and is tested
    at ``test_times`` from now (any times, not one interval; none for
    "never tested"). A test finds and repairs a failed device as new, in
    no time; a working one carries on at its age. Demands come at
    ``demand_rate``, and one that meets the device failed costs
    ``consequence_cost``; with ``demands_reveal`` (the default) it also
    shows the failure, which is repaired as new then, at
    ``repair_cost``. Without it, only a test finds a failure, as
    [`failure_finding`][repyability.failure_finding] has it in the long
    run.

    Exact at the tests where demands do not reveal (or there are none),
    and numerical otherwise, to ``precision``.

    Parameters
    ----------
    protective : NonRepairable
        The device: its life (a failure is hidden) as a part. Its own
        costs are not used.
    test_times : array_like
        The tests' times from now, increasing, within the window; empty
        for a device never tested.
    window : float
        How far ahead, more than 0.
    age : float, optional
        The device's age now (it is working), by default 0.
    test_cost : float, optional
        What a test costs, by default 0.
    repair_cost : float, optional
        What repairing a failure costs, found by a test or shown by a
        demand, by default 0.
    demand_rate : float, optional
        How often the protected event comes, per unit time.
    consequence_cost : float, optional
        What a demand met by a failed device costs (needs
        ``demand_rate``), by default 0.
    discount_rate : float, optional
        The discount rate per unit of time, continuously compounded, by
        default 0.
    demands_reveal : bool, optional
        Whether a demand that meets the device failed shows the failure
        and has it repaired, by default True.
    times : array_like, optional
        The times to give the unavailability at; by default 101 evenly
        over the window.
    tolerance : float, optional
        How close the failed time and the failures found are worked out,
        as a share of each; by default 1e-6.

    Returns
    -------
    FailureFindingWindow

    Raises
    ------
    NotImplementedError
        The device's life has no survival function in closed form, or
        there are more than 2,000 tests in the window.
    ValueError
        A number out of its range; test times not increasing or outside
        the window; a consequence with no demand rate.

    Examples
    --------
    A trip failing hidden at 0.1 a year, tested at 6 and 12 months and
    then at 2 years, over 3 years; a demand comes once in 2 years:

    >>> import surpyval as surv
    >>> from repyability import NonRepairable, failure_finding_window
    >>> trip = NonRepairable(surv.Exponential.from_params([0.1]))
    >>> tested = failure_finding_window(
    ...     trip, [0.5, 1.0, 2.0], 3.0, demand_rate=0.5,
    ...     test_cost=200.0, repair_cost=1500.0, consequence_cost=50_000.0,
    ... )
    >>> [round(float(q), 4) for q in tested.found_at_tests]
    [0.0432, 0.0432, 0.0752]
    >>> round(tested.mean_unavailability, 4), round(tested.cost, 2)
    (0.0351, 3555.15)
    >>> never = failure_finding_window(
    ...     trip, [], 3.0, demand_rate=0.5, consequence_cost=50_000.0,
    ...     repair_cost=1500.0,
    ... )
    >>> round(never.mean_unavailability, 4), round(never.cost, 2)
    (0.0894, 6904.57)
    """
    _check_protective(protective)
    window = _number(window, "window", strict=True)
    age = _number(age, "age")
    test, repair, consequence, demand = _prices(
        test_cost, repair_cost, consequence_cost, demand_rate
    )
    rate = _number(discount_rate, "discount_rate")
    if not isinstance(demands_reveal, (bool, np.bool_)):
        raise ValueError(
            f"demands_reveal must be True or False; got {demands_reveal!r}."
        )
    if not (checks.is_number(tolerance) and 0.0 < float(tolerance) < 1.0):
        raise ValueError(
            "tolerance must be a share, more than 0 and less than 1 "
            f"(such as 1e-6); got {tolerance!r}."
        )
    tests = _test_times(test_times, window)
    if times is None:
        at = np.linspace(0.0, window, 101)
    else:
        at = checks.real_array(times, "times")
        if np.any(at < 0.0) or np.any(at > window):
            raise ValueError(
                f"times must be from 0 to window={window:g}; got {times!r}."
            )
    reveal = demand if demands_reveal and demand is not None else 0.0
    solved = _hidden(
        protective.reliability,
        age,
        tests[tests > 0.0],
        window,
        reveal,
        float(tolerance),
    )
    found = np.zeros(tests.size)
    found[tests > 0.0] = solved.found  # at time 0 the device is working
    failed = float(solved.failed_time(np.array([window]))[0])
    discounted = solved.discounted_failed_time(rate)
    met = None if demand is None else demand * failed
    each = consequence + (repair if reveal > 0.0 else 0.0)
    factors = np.exp(-rate * tests)
    tests_cost = test * tests.size
    repairs_cost = repair * float(found.sum())
    if reveal > 0.0:
        repairs_cost += repair * reveal * failed
    consequences = 0.0 if demand is None else consequence * demand * failed
    present = (
        test * float(factors.sum())
        + repair * dot(factors, found)
        + (0.0 if demand is None else each * demand * discounted)
    )
    cost = tests_cost + repairs_cost + consequences
    if rate == 0.0:
        present = cost  # the same sum, to the last bit
    precision = repair * solved.found_error
    if demand is not None:
        precision += each * demand * solved.time_error
    return FailureFindingWindow(
        window=window,
        age=age,
        test_times=tests,
        demands_reveal=bool(demands_reveal),
        found_at_tests=found,
        failures_found=float(found.sum()),
        failed_time=failed,
        mean_unavailability=failed / window,
        demands_on_failed=met,
        times=at,
        unavailability=solved.unavailability(at),
        tests_cost=tests_cost,
        repairs_cost=repairs_cost,
        consequence_cost=consequences,
        cost=cost,
        discount_rate=rate,
        present_value=present,
        precision=precision,
    )
