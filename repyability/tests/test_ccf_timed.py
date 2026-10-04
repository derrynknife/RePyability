"""Common-cause groups whose members' tests and repairs take time (#220).

A member's level in its group's chain may then be off line for a test
while working (where it neither ages nor is struck), under a test while
failed, or under repair: a test or repair of a fixed length ends a fixed
time after its test, one of an exponential length at a rate, and a test
that falls in a member's own test or repair is not done, as in the
simulation. Checked against the members' own model when no cause is shared
(beta = 0, where they are independent: each a renewal cycle followed test
by test, #159), IEC 61508-6's 1oo2 formula, and the simulation, which
strikes the causes itself.
"""

import numpy as np
import pytest
from surpyval import ExactEventTime, Exponential, Weibull

from repyability import BetaFactor, CCFGroup, RepairableRBD
from repyability.rbd import routes as r

E, X, W = (
    Exponential.from_params,
    ExactEventTime.from_params,
    Weibull.from_params,
)
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def inspected(rate, interval, repair, test=None, **more):
    inspection = {"interval": interval, **more}
    if test is not None:
        inspection["duration"] = test
    return {
        "reliability": E([rate]),
        "repairability": repair,
        "inspection": inspection,
    }


TIMINGS = {
    "fixed repair": {"repair": X([8.0])},
    "exponential repair": {"repair": E([1 / 8.0])},
    "fixed test and repair": {"repair": X([8.0]), "test": X([2.0])},
    "fixed test, instant repair": {"repair": "instant", "test": X([3.0])},
    "exponential test and repair": {"repair": E([1 / 8.0]), "test": E([0.5])},
    "fixed test, exponential repair": {
        "repair": E([1 / 8.0]),
        "test": X([2.0]),
    },
}


def pair(beta, timing, rate=3e-4, interval=500.0, **more):
    """Two members tested at different times, in parallel; a group of the
    two unless ``beta`` is None."""
    a = inspected(rate, interval, **timing, **more)
    b = inspected(rate, interval, offset=0.34 * interval, **timing, **more)
    groups = [] if beta is None else [CCFGroup(["a", "b"], BetaFactor(beta))]
    return RepairableRBD(PAIR, {"a": a, "b": b}, ccf_groups=groups)


@pytest.mark.parametrize("name", list(TIMINGS))
def test_no_shared_cause_is_the_independent_units(name):
    grouped, plain = pair(0.0, TIMINGS[name]), pair(None, TIMINGS[name])
    # To the members' own model's grid: the chain itself is exact.
    assert grouped.mean_unavailability() == pytest.approx(
        plain.mean_unavailability(), rel=3e-5
    )
    x = np.array([50.0, 400.0, 520.0, 3000.0])
    np.testing.assert_allclose(
        1.0 - grouped.point_availability(x),
        1.0 - plain.point_availability(x),
        rtol=2e-3,
    )


def test_no_shared_cause_with_tests_that_miss():
    timing = {"repair": X([8.0])}
    more = {"coverage": 0.7, "full_test": 1500.0}
    grouped = pair(0.0, timing, **more)
    plain = pair(None, timing, **more)
    assert grouped.mean_unavailability() == pytest.approx(
        plain.mean_unavailability(), rel=3e-5
    )


def test_the_issues_one_out_of_two():
    # IEC 61508-6 (B.3.2.2.2), for lambda T small: PFDavg =
    # 2 ((1 - beta) lambda)^2 t_CE t_GE + beta lambda (T / 2 + MRT), with
    # t_CE = T / 2 + MRT and t_GE = T / 3 + MRT.
    rate, interval, beta, mrt = 2e-6, 8760.0, 0.1, 8.0
    unit = inspected(rate, interval, X([mrt]))
    rbd = RepairableRBD(
        PAIR,
        {"a": unit, "b": unit},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )
    t_ce, t_ge = interval / 2 + mrt, interval / 3 + mrt
    iec = 2 * ((1 - beta) * rate) ** 2 * t_ce * t_ge
    iec += beta * rate * (interval / 2 + mrt)
    pfd = rbd.mean_unavailability()
    assert pfd == pytest.approx(iec, rel=0.005)
    instant = inspected(rate, interval, "instant")
    no_mrt = RepairableRBD(
        PAIR,
        {"a": instant, "b": instant},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    ).mean_unavailability()
    # The repair keeps the pair down a little longer after each test.
    assert no_mrt < pfd < no_mrt * 1.01
    routes = rbd.analysis_routes()
    for name in ("mean_unavailability", "point_availability"):
        assert routes[name].route != r.REFUSED, name


@pytest.mark.parametrize(
    "name",
    [
        "fixed repair",
        "fixed test and repair",
        "exponential test and repair",
        "fixed test, exponential repair",
    ],
)
def test_the_simulation_agrees(name):
    rbd = pair(0.3, TIMINGS[name], rate=2e-3, interval=100.0)
    T, n = 2000.0, 4000
    result = rbd.availability(
        T, mc_samples=n, seed=7, control_variate=False, conditional=False
    )
    interval = result.mean_availability_interval()
    exact = rbd.mission_availability(T)
    assert abs(interval.estimate - exact) < 4.0 * interval.standard_error


def test_the_failure_frequency_with_repairs_that_take_time():
    # A repair after a test keeps a member down without a failure of the
    # system's own: the frequency counts the causes' and the nodes' rates
    # where they take the system down, as with instant repairs.
    rbd = pair(0.2, TIMINGS["fixed repair"], rate=2e-3, interval=100.0)
    T, n = 4000.0, 3000
    result = rbd.availability(
        T, mc_samples=n, seed=11, control_variate=False, conditional=False
    )
    failures = rbd.expected_failures(T)
    assert result.system_failures / n == pytest.approx(
        failures, abs=4.0 * np.sqrt(failures / n)
    )
    assert rbd.mean_time_between_failures() > 0.0
    # Tests that take time are planned outages, not worked out with groups.
    timed = pair(0.2, TIMINGS["fixed test and repair"])
    with pytest.raises(NotImplementedError, match="tests that take time"):
        timed.mean_time_between_failures()


def refusal(**timing):
    with pytest.raises(NotImplementedError) as error:
        pair(0.1, timing).mean_unavailability()
    message = str(error.value)
    assert "''" not in message
    assert "availability() or cost()" in message
    return message


def test_what_the_chain_does_not_take_says_what_to_do():
    assert "take none of those" in refusal(repair=W([8.0, 2.0]))
    assert "repairs a fixed one" in refusal(repair=X([8.0]), test=E([0.5]))
    assert "as long as its test interval" in refusal(repair=X([600.0]))
    # The routes say the same.
    rbd = pair(0.1, {"repair": W([8.0, 2.0])})
    route = rbd.analysis_routes()["mean_unavailability"]
    assert route.route == r.REFUSED
    assert "take none of those" in route.reason


def test_copies_of_members_whose_repairs_take_time_are_refused():
    rbd = pair(0.1, TIMINGS["fixed repair"])
    group = rbd.ccf_groups[0]
    times, _ = rbd._long_run_grid()
    with pytest.raises(NotImplementedError, match="copies of them"):
        rbd._group_states(group, times, counts=[2, 1])


def test_the_ends_of_tests_and_repairs_are_jumps_over_time():
    # A test of a fixed length ends, and a repair of one, at fixed times:
    # the system's availability jumps there, each jump split among what
    # changes then, and with the rates they make up its change.
    spec = {
        n: {
            "reliability": E([0.01]),
            "repairability": X([2.0]),
            "inspection": {
                "interval": 10.0,
                "offset": offset,
                "duration": X([1.0]),
            },
        }
        for n, offset in zip("ab", (0.0, 5.0))
    }
    spec["c"] = {"reliability": E([0.002]), "repairability": E([0.5])}
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        spec,
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2))],
    )
    rate = rbd.availability_rate(23.0)
    # Tests at 0, 5, 10, ..., their ends 1 later and the repairs' 3 later.
    for end in (1.0, 3.0, 6.0, 8.0, 11.0, 13.0):
        assert end in rate.jump_times
    before = rbd.point_availability(np.nextafter(rate.jump_times, -np.inf))
    after = rbd.point_availability(rate.jump_times)
    np.testing.assert_allclose(rate.jumps, after - before, atol=1e-9)
    end = 12.0
    edges = np.concatenate([[0.0], rate.jump_times[rate.jump_times < end]])
    edges = np.append(edges, end)
    integral = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        x = np.linspace(lo, hi, 801)
        if hi < end:
            x[-1] = np.nextafter(hi, -np.inf)
        values = rbd.availability_rate(x).rate
        integral += np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(x))
    jumps = rate.jumps[rate.jump_times < end].sum()
    change = rbd.point_availability(end) - rbd.point_availability(0.0)
    assert integral + jumps == pytest.approx(change, abs=1e-7)
