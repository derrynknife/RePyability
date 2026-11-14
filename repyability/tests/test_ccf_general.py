"""Common-cause groups with tests and repairs of any length, and started
from the members' current states (#106).

A test or repair of a length neither fixed nor exponential ends at a rate
that varies with the time since the member's test: the group's chain
follows it on pieces of that time, each at constant rates that keep the
member's own chance of being in its test, or in its repair, exact at the
pieces' ends (``_ccf_chain._Episode``). A chain started from the members'
states starts from the distribution they give: each member known to be up
since it was last tested (or put into service), the causes striking the
members up since then, and a member down in a repair under way; or its
long run. Checked against the fixed and exponential lengths the chain
takes exactly, the members' own model where no cause is shared, the
simulation, and hand-worked values.
"""

import numpy as np
import pytest
from scipy.linalg import expm
from surpyval import ExactEventTime, Exponential, Gamma, LogNormal, Weibull

from repyability import BetaFactor, CCFGroup, NodeState, RepairableRBD
from repyability.rbd import _ccf_chain as chain
from repyability.rbd import routes as r

E, X, W, L = (
    Exponential.from_params,
    ExactEventTime.from_params,
    Weibull.from_params,
    LogNormal.from_params,
)
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]

GENERAL = {
    "Weibull repair": {"repair": W([8.0, 2.0])},
    "lognormal test, Weibull repair": {
        "repair": W([8.0, 2.0]),
        "test": L([0.5, 0.4]),
    },
    "exponential test, fixed repair": {"repair": X([8.0]), "test": E([0.5])},
}


def inspected(rate, interval, repair, test=None, **more):
    inspection = {"interval": interval, **more}
    if test is not None:
        inspection["duration"] = test
    return {
        "reliability": E([rate]),
        "repairability": repair,
        "inspection": inspection,
    }


def pair(beta, timing, rate=2e-3, interval=100.0):
    """Two members tested at different times, in parallel; a group of the
    two unless ``beta`` is None."""
    a = inspected(rate, interval, **timing)
    b = inspected(rate, interval, offset=0.34 * interval, **timing)
    groups = [] if beta is None else [CCFGroup(["a", "b"], BetaFactor(beta))]
    return RepairableRBD(PAIR, {"a": a, "b": b}, ccf_groups=groups)


# -- tests and repairs of any length -----------------------------------------


def long_run_states(test, repair, times):
    tests = [chain.ProofTest(100.0, 0, True), chain.ProofTest(34.0, 1, True)]
    timings = [chain.Timing(test, repair)] * 2
    return chain.hidden(
        BetaFactor(0.3),
        ["a", "b"],
        2e-3,
        1.0,
        tests,
        100.0,
        times,
        timings=timings,
    ).probabilities


def as_general(rate):
    return chain.Duration(model=E([rate]))


@pytest.mark.parametrize(
    "exact, general, tolerance",
    [
        (
            (chain.Duration(), chain.Duration(rate=0.5)),
            (chain.Duration(), as_general(0.5)),
            1e-13,
        ),
        (
            (chain.Duration(fixed=2.0), chain.Duration(rate=0.5)),
            (chain.Duration(fixed=2.0), as_general(0.5)),
            1e-13,
        ),
        (
            (chain.Duration(rate=2.0), chain.Duration(rate=0.5)),
            (as_general(2.0), as_general(0.5)),
            1e-5,
        ),
    ],
    ids=["exponential repair", "fixed test", "exponential test and repair"],
)
def test_exponential_lengths_through_the_general_path(
    exact, general, tolerance
):
    # An exponential length given as a general one ends at its hazard,
    # constant: the pieces' rates are the rate itself, and the chain is
    # _Timed's, exact (to the sum of the test and repair taken on a grid,
    # where both are random). Each test and repair is over by the next
    # test (but for a chance below 1e-12), as _ccf_timings requires: the
    # general path follows one from the test that started it.
    times = np.linspace(0.0, 99.9, 21)
    want = long_run_states(*exact, times)
    got = long_run_states(*general, times)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=tolerance)


def test_a_narrow_length_tends_to_the_fixed_one():
    # A gamma repair of mean 8 and sd 0.04, against the fixed 8, which
    # ends with a jump: the pieces follow it as closely.
    times = np.linspace(0.0, 99.9, 11)
    fixed = long_run_states(chain.Duration(), chain.Duration(fixed=8.0), times)
    narrow = long_run_states(
        chain.Duration(),
        chain.Duration(model=Gamma.from_params([40000.0, 5000.0])),
        times,
    )
    np.testing.assert_allclose(narrow, fixed, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("name", list(GENERAL))
def test_no_shared_cause_is_the_independent_units(name):
    # beta = 0: the members are independent, each the members' own model
    # (#159, its own grid), which takes tests and repairs of any length.
    grouped, plain = pair(0.0, GENERAL[name]), pair(None, GENERAL[name])
    assert grouped.mean_unavailability() == pytest.approx(
        plain.mean_unavailability(), rel=5e-4
    )
    x = np.array([50.0, 150.0, 520.0])
    np.testing.assert_allclose(
        1.0 - grouped.point_availability(x),
        1.0 - plain.point_availability(x),
        rtol=2e-3,
    )


@pytest.mark.parametrize("name", list(GENERAL))
def test_the_simulation_agrees(name):
    rbd = pair(0.3, GENERAL[name])
    T, n = 1000.0, 3000
    result = rbd.availability(
        T, mc_samples=n, seed=11, control_variate=False, conditional=False
    )
    interval = result.mean_availability_interval()
    exact = rbd.mission_availability(T)
    assert abs(interval.estimate - exact) < 4.0 * interval.standard_error


def test_the_routes_take_general_lengths():
    rbd = pair(0.3, GENERAL["Weibull repair"])
    routes = rbd.analysis_routes()
    for name in ("mean_unavailability", "point_availability", "availability"):
        assert routes[name].route != r.REFUSED, name
    assert "general length" in routes["mean_unavailability"].reason


# -- starting from a current state -------------------------------------------


def pair_tested(beta):
    unit = {
        "reliability": E([1e-2]),
        "repairability": "instant",
        "inspection": {"interval": 100.0},
    }
    return RepairableRBD(
        PAIR,
        {"a": unit, "b": dict(unit)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )


def test_known_up_members_by_hand():
    # a last known up 30 ago (tested then, put into service 50 ago), b 10
    # ago (put into service then): before their next tests, each is up
    # while no cause that strikes it has struck since; both while neither
    # own cause has, nor the shared one since the earlier of the two.
    lam, beta = 1e-2, 0.2
    rbd = pair_tested(beta)
    ka, kb = 30.0, 10.0
    state = {
        "a": NodeState(age=50.0, since_scheduled=ka),
        "b": NodeState(age=kb, since_scheduled=40.0),
    }
    x = np.array([0.0, 5.0, 20.0])
    own, shared = (1.0 - beta) * lam, beta * lam
    up_a, up_b = np.exp(-lam * (ka + x)), np.exp(-lam * (kb + x))
    both = np.exp(-own * (ka + x) - own * (kb + x) - shared * (ka + x))
    np.testing.assert_allclose(
        1.0 - rbd.point_availability(x, state=state),
        1.0 - up_a - up_b + both,
        rtol=1e-10,
    )


def test_a_member_in_a_repair_by_hand():
    # Revealed failures, exponential repairs: a's repair is under way, its
    # time so far forgotten; the group's four states by hand.
    lam, mu, beta = 1e-2, 0.1, 0.2
    member = {"reliability": E([lam]), "repairability": E([mu])}
    rbd = RepairableRBD(
        PAIR,
        {"a": member, "b": dict(member)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(beta))],
    )
    own, shared = (1.0 - beta) * lam, beta * lam
    # States: both up, a down, b down, both down.
    Q = np.array(
        [
            [0.0, own, own, shared],
            [mu, 0.0, 0.0, own + shared],
            [mu, 0.0, 0.0, own + shared],
            [0.0, mu, mu, 0.0],
        ]
    )
    np.fill_diagonal(Q, -Q.sum(axis=1))
    start = np.array([0.0, 1.0, 0.0, 0.0])
    x = np.array([0.0, 1.0, 10.0, 50.0])
    want = np.array([1.0 - (start @ expm(Q * t))[3] for t in x])
    state = {"a": NodeState(alive=False, down_for=3.0)}
    np.testing.assert_allclose(
        rbd.point_availability(x, state=state), want, rtol=1e-9
    )
    # Its long run is its start: constant from the stationary state.
    steady = rbd.point_availability(x, state="stationary")
    np.testing.assert_allclose(steady, rbd.mean_availability(), rtol=1e-12)


def timed_pair(repair, test=None):
    a = inspected(2e-3, 100.0, repair, test)
    b = inspected(2e-3, 100.0, repair, test, offset=34.0)
    return RepairableRBD(
        PAIR,
        {"a": a, "b": b},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.3))],
    )


STATE = {
    "a": NodeState(alive=False, down_for=3.0, since_scheduled=5.0),
    "b": NodeState(age=300.0, since_scheduled=60.0),
}


@pytest.mark.parametrize(
    "repair", [X([8.0]), W([8.0, 2.0])], ids=["fixed", "Weibull"]
)
def test_a_state_against_the_simulation(repair):
    # a in a repair under way for 3, its calendar 5 past its last test; b
    # known up 60 ago, at its last test: the simulation draws what is
    # left of a's repair, b's own cause's life from 60 ago, and when the
    # shared cause last struck.
    rbd = timed_pair(repair)
    T = 300.0
    exact = rbd.mission_availability(T, state=STATE)
    # At 0 only b can be up, if nothing has struck it since its test.
    assert rbd.point_availability(0.0, state=STATE) == pytest.approx(
        np.exp(-2e-3 * 60.0), rel=1e-12
    )
    result = rbd.availability(
        T,
        mc_samples=20000,
        seed=2,
        state=STATE,
        control_variate=False,
        conditional=False,
    )
    interval = result.mean_availability_interval()
    assert abs(interval.estimate - exact) < 4.0 * interval.standard_error


def test_runs_from_a_state_are_reproduced():
    rbd = timed_pair(X([8.0]))
    one = rbd.availability(300.0, mc_samples=200, seed=4, state=STATE)
    two = rbd.availability(300.0, mc_samples=200, seed=4, state=STATE)
    assert np.array_equal(one.uptimes, two.uptimes)
    # And the control variate is the exact value from the same state.
    controlled = rbd.availability(
        300.0, mc_samples=200, seed=4, state=STATE, control_variate=True
    )
    interval = controlled.mean_availability_interval()
    assert interval.estimate == pytest.approx(
        rbd.mission_availability(300.0, state=STATE), rel=1e-12
    )


@pytest.mark.parametrize(
    "timing",
    [{"repair": X([8.0])}, GENERAL["Weibull repair"]],
    ids=["fixed", "Weibull"],
)
def test_the_stationary_start_repeats_the_long_run(timing):
    # From the long run, every period is the long run's: its average over
    # a period, and over two, is the long-run availability.
    rbd = pair(0.3, timing)
    mean = rbd.mean_availability()
    for stop in (100.0, 200.0):
        assert rbd.mission_availability(
            stop, state="stationary"
        ) == pytest.approx(mean, rel=2e-6)
    # Its values a period apart are the same.
    x = np.array([10.0, 37.0, 80.0])
    np.testing.assert_allclose(
        rbd.point_availability(x, state="stationary"),
        rbd.point_availability(x + 100.0, state="stationary"),
        rtol=1e-10,
    )


def test_what_a_state_cannot_be():
    rbd = pair_tested(0.2)
    with pytest.raises(NotImplementedError, match="long-run state"):
        rbd.point_availability(1.0, state={"a": NodeState(stationary=True)})
    # A member whose found failures are repaired at once is never down.
    down = {"a": NodeState(alive=False, down_for=1.0)}
    with pytest.raises(ValueError, match="always over"):
        rbd.point_availability(1.0, state=down)
    own = {"a": NodeState(age=1.0, model=E([0.02]))}
    with pytest.raises(NotImplementedError, match="life of its own"):
        rbd.point_availability(1.0, state=own)
    with pytest.raises(NotImplementedError, match="life of its own"):
        rbd.availability(10.0, mc_samples=5, state=own)
