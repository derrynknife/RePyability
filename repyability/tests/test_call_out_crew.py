"""A call-out crew (``call_out``): one more repair crew that comes when
``when_waiting`` jobs wait, at a fee each time and a rate while engaged,
and leaves at the first end of a job that leaves a crew with no waiting
job to take.

The crews' Markov chain has it (``_crew_chain``) and the Python event
loop's queue follows the same rule (``_events._Crews``). Here: the rule
itself, the chain against states solved by hand and against the
simulation, the costs in the long run, over time and discounted (against
the transient solution worked out with a matrix exponential), and what
refuses."""

import json
import math
from fractions import Fraction

import numpy as np
import pytest
import surpyval as surv
from scipy.linalg import expm

from repyability import RepairableRBD
from repyability.rbd import _compiled, _crew_chain, _event_loop
from repyability.rbd import routes as r
from repyability.rbd._events import Event, _Crews

E = surv.Exponential.from_params
LAM, MU = 0.02, 0.1
FEE, RATE = 40.0, 6.0


def parallel(n):
    nodes = "abcdef"[:n]
    return [("s", x) for x in nodes] + [(x, "t") for x in nodes]


def three(call_out=None, crews=1, **options):
    """Three alike units in parallel, one repair crew."""
    return RepairableRBD(
        parallel(3),
        {
            node: {
                "reliability": E([LAM]),
                "repairability": E([MU]),
                "repair_cost": 3.0,
            }
            for node in "abc"
        },
        repair_crews=crews,
        call_out=call_out,
        downtime_cost_rate=5.0,
        **options,
    )


def contract(k=2, fee=FEE, rate=RATE):
    return {"when_waiting": k, "fee": fee, "rate": rate}


def by_hand():
    """``three(contract(2))`` solved by hand. Its states, by the jobs
    under repair and waiting: A (0, 0), B (1, 0), C (1, 1), D (2, 1: the
    third failure made two wait, the call-out crew took one), F (2, 0:
    still engaged, until a job ends with none waiting). Balance, with
    ``rho = lam / mu``: ``B = 3 rho A``, ``C = 2 rho B / (1 + rho)``,
    ``F = rho C / 2``, ``D = (rho + 2) F / 2``."""
    lam, mu = (
        Fraction(LAM).limit_denominator(),
        Fraction(MU).limit_denominator(),
    )
    rho = lam / mu
    a = Fraction(1)
    b = 3 * rho * a
    c = 2 * rho * b / (1 + rho)
    f = rho * c / 2
    d = (rho + 2) * f / 2
    total = a + b + c + d + f
    return {
        "availability": float(1 - d / total),
        "calls": float(lam * c / total),
        "engaged": float((d + f) / total),
    }


def generator():
    """The five states' generator, in the order A, B, C, D, F, the rate at
    which the call-out crew is called in each, and where it is engaged."""
    q = np.zeros((5, 5))
    q[0, 1] = 3 * LAM
    q[1, 0], q[1, 2] = MU, 2 * LAM
    q[2, 1], q[2, 3] = MU, LAM
    q[3, 4] = 2 * MU
    q[4, 3], q[4, 1] = LAM, 2 * MU
    q -= np.diag(q.sum(axis=1))
    calls = np.array([0.0, 0.0, LAM, 0.0, 0.0])
    engaged = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    return q, calls, engaged


# -- the rule ---------------------------------------------------------------


def test_the_queue_calls_the_crew_and_lets_it_go():
    crews = _Crews(1, "abcd", {"c": 1.0}, call_when=2)

    def due(node, t, length):
        return crews.request(node, t, Event(t + length, node, True))

    assert due("a", 0.0, 10.0).time == 10.0  # the crew's
    assert due("b", 1.0, 5.0) is None and crews.call(1.0) is None
    assert due("c", 2.0, 4.0) is None
    # Two wait: the call-out crew comes, for the job of the highest
    # priority, which never waited.
    node, wait, ends = crews.call(2.0)
    assert (node, wait, ends.time) == ("c", 0.0, 6.0)
    assert crews.engaged and crews.calls == 1
    # One crew only: more waiting jobs call no other.
    assert due("d", 3.0, 1.0) is None and crews.call(3.0) is None
    # A job ends with jobs waiting: its crew takes the next (first due).
    node, wait, ends = crews.release("c", 6.0)
    assert (node, wait, ends.time) == ("b", 5.0, 11.0)
    node, wait, ends = crews.release("a", 10.0)
    assert (node, wait, ends.time) == ("d", 7.0, 11.0)
    assert crews.engaged
    # The first end of a job with none waiting: the call-out crew goes,
    # and the crew that came free has the job left.
    assert crews.release("b", 11.0) is None
    assert not crews.engaged and crews.free == 0
    assert crews.close(20.0) == (1, 9.0)
    assert crews.release("d", 11.0) is None and crews.free == 1


def test_time_engaged_is_closed_at_the_end_of_the_window():
    crews = _Crews(1, "ab", {}, call_when=1)
    crews.request("a", 0.0, Event(50.0, "a", True))
    assert crews.request("b", 4.0, Event(60.0, "b", True)) is None
    assert crews.call(4.0)[0] == "b"
    assert crews.close(10.0) == (1, 6.0)


# -- the chain --------------------------------------------------------------


def test_the_chain_has_the_states_solved_by_hand():
    rbd = three(contract(2))
    states = rbd.crew_states()
    assert len(states.probabilities) == 1 + 3 + 6 + 3 + 3
    hand = by_hand()
    p = states.probabilities
    assert rbd.mean_availability() == pytest.approx(
        hand["availability"], rel=1e-13
    )
    assert float(p @ states.call_out_rate) == pytest.approx(
        hand["calls"], rel=1e-13
    )
    assert float(p @ states.call_out_engaged) == pytest.approx(
        hand["engaged"], rel=1e-13
    )
    assert rbd.expected_cost_rate() - three(
        contract(2, 0.0, 0.0)
    ).expected_cost_rate() == pytest.approx(
        FEE * hand["calls"] + RATE * hand["engaged"], rel=1e-12
    )


def test_two_units_called_at_the_first_wait_fail_independently():
    # Each failure has a crew at once, so each unit is down lam / (lam +
    # mu) of the time on its own: the call-out crew is engaged while both
    # are down, and called when one fails with the other down.
    la, lb, mu = 0.03, 0.01, 0.2
    rbd = RepairableRBD(
        parallel(2),
        {
            "a": {"reliability": E([la]), "repairability": E([mu])},
            "b": {"reliability": E([lb]), "repairability": E([mu])},
        },
        repair_crews=1,
        call_out={"when_waiting": 1, "fee": FEE, "rate": RATE},
    )
    qa, qb = la / (la + mu), lb / (lb + mu)
    assert rbd.mean_availability() == pytest.approx(1 - qa * qb, rel=1e-13)
    calls = qa * (1 - qb) * lb + qb * (1 - qa) * la
    assert rbd.expected_cost_rate() == pytest.approx(
        FEE * calls + RATE * qa * qb, rel=1e-13
    )


def test_a_crew_never_called_changes_nothing():
    # Three units, one under repair: two wait at most.
    plain, never = three(), three(contract(3))
    assert never.mean_availability() == plain.mean_availability()
    assert never.expected_cost_rate() == plain.expected_cost_rate()
    windows = np.array([5.0, 50.0, 500.0])
    for rate in (0.0, 0.01):
        np.testing.assert_array_equal(
            never.expected_cost(windows, discount_rate=rate).mean,
            plain.expected_cost(windows, discount_rate=rate).mean,
        )
    np.testing.assert_array_equal(
        never.point_availability(windows), plain.point_availability(windows)
    )
    a = never.cost(200.0, mc_samples=50, seed=3, control_variate=False)
    b = plain.cost(200.0, mc_samples=50, seed=3, control_variate=False)
    assert a.mean == b.mean and a.by_category == b.by_category


def test_called_at_the_first_wait_it_is_a_crew_more_to_the_units():
    one, two = three(contract(1, 0.0, 0.0)), three(crews=2)
    assert one.mean_availability() == pytest.approx(
        two.mean_availability(), rel=1e-14
    )
    t = np.array([3.0, 30.0, 300.0])
    np.testing.assert_allclose(
        one.point_availability(t), two.point_availability(t), rtol=1e-12
    )
    assert one.mean_time_to_first_failure() == pytest.approx(
        two.mean_time_to_first_failure(), rel=1e-12
    )
    a = one.availability(300.0, mc_samples=40, seed=5, control_variate=False)
    b = two.availability(300.0, mc_samples=40, seed=5, control_variate=False)
    assert a.sample_mean_availability == b.sample_mean_availability


def test_the_states_are_counted_as_they_are_built():
    priorities, instant = [1.0, 0.0, 0.0, 0.0, 2.0], [False] * 4 + [True]
    for crews in (1, 2):
        for when in (1, 2, 3):
            built = _crew_chain.solve(
                list("abcde"),
                [0.1] * 5,
                [1.0] * 4 + [math.inf],
                priorities,
                crews,
                when,
            )
            assert _crew_chain.state_count(
                priorities, instant, crews, when
            ) == len(built.states)
            assert built.probabilities.sum() == pytest.approx(1.0)
            # Engaged exactly with a job more than the crews under repair.
            assert [len(s[0]) > crews for s in built.states] == list(
                built.engaged
            )


def test_too_many_states_are_refused():
    rbd = RepairableRBD(
        parallel(6) + [("s", "g"), ("g", "t"), ("s", "h"), ("h", "t")],
        {
            node: {"reliability": E([LAM]), "repairability": E([MU])}
            for node in "abcdefgh"
        },
        repair_crews=1,
        call_out={"when_waiting": 7},
    )
    with pytest.raises(NotImplementedError, match="more than the 15,000"):
        rbd.mean_availability()


# -- over time, and discounted ------------------------------------------------


def test_the_costs_over_time_are_the_transient_solution_s():
    q, calls, engaged = generator()
    cost = FEE * calls + RATE * engaged
    start = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    paid = three(contract(2))
    free = three(contract(2, 0.0, 0.0))
    windows = np.array([4.0, 40.0, 400.0])
    for rate in (0.0, 0.03):
        # The integral of exp(-r t) p(t) cost from 0 to T, by the matrix
        # exponential of [[Q - r, cost], [0, 0]].
        block = np.zeros((6, 6))
        block[:5, :5] = q - rate * np.eye(5)
        block[:5, 5] = cost
        want = [float((start @ expm(block * T)[:5, 5])) for T in windows]
        got = paid.expected_cost(windows, discount_rate=rate)
        np.testing.assert_allclose(
            got.by_category["crews"], want, rtol=1e-6 if rate else 1e-10
        )
        none = free.expected_cost(windows, discount_rate=rate)
        np.testing.assert_allclose(none.by_category["crews"], 0.0, atol=0.0)
        np.testing.assert_allclose(
            got.mean - none.mean,
            want,
            rtol=1e-6 if rate else 1e-10,
        )


def test_salaries_and_the_call_out_crew_share_the_category():
    T, rate, salary = 100.0, 0.02, 0.5
    both = three(contract(2), crew_cost_rate=salary)
    called = three(contract(2))
    for r_ in (0.0, rate):
        salaries = salary * T if not r_ else salary * -math.expm1(-r_ * T) / r_
        assert both.expected_cost(T, discount_rate=r_).by_category[
            "crews"
        ] == pytest.approx(
            called.expected_cost(T, discount_rate=r_).by_category["crews"]
            + salaries,
            rel=1e-12,
        )
    assert both.expected_cost_rate() == pytest.approx(
        called.expected_cost_rate() + salary, rel=1e-13
    )


def test_the_contract_s_value_is_the_difference_of_two_npvs():
    T, rate = 400.0, 0.01
    without, with_it = three(), three(contract(2))
    a = without.expected_npv(T, discount_rate=rate, periods=4)
    b = with_it.expected_npv(T, discount_rate=rate, periods=4)
    crews = sum(
        row["present_value"]
        for row in b.cash_flows
        if row["category"] == "crews"
    )
    assert -crews == pytest.approx(
        with_it.expected_cost(T, discount_rate=rate).by_category["crews"],
        rel=1e-9,
    )
    # It costs its fees and rate, and saves down time: the system's is
    # shorter with it.
    down = [
        abs(result.cost_by_category["system_downtime"]) for result in (a, b)
    ]
    assert down[1] < down[0]
    assert b.npv - a.npv == pytest.approx(
        -(
            with_it.expected_cost(T, discount_rate=rate).mean
            - without.expected_cost(T, discount_rate=rate).mean
        ),
        rel=1e-9,
    )


# -- the simulation -----------------------------------------------------------


def mixed(call_out):
    """Four units with priorities and an instant repair, two crews."""
    return RepairableRBD(
        parallel(4),
        {
            "a": {
                "reliability": E([0.05]),
                "repairability": E([0.2]),
                "priority": 2,
                "repair_cost": 4.0,
            },
            "b": {"reliability": E([0.08]), "repairability": E([0.1])},
            "c": {
                "reliability": E([0.03]),
                "repairability": "instant",
                "priority": 1,
            },
            "d": {"reliability": E([0.06]), "repairability": E([0.15])},
        },
        k={"t": 2},
        repair_crews=2,
        call_out=call_out,
        downtime_cost_rate=7.0,
    )


@pytest.mark.parametrize(
    "build, call_out",
    [
        (three, contract(2)),
        (three, contract(1, FEE, 0.0)),
        (three, contract(2, 0.0, RATE)),
        (mixed, contract(1)),
        (mixed, contract(2)),
    ],
)
def test_the_simulation_agrees_with_the_chain(build, call_out):
    T, n = 300.0, 4000
    rbd = build(call_out)
    exact = rbd.expected_cost(T)
    run = rbd.availability(
        T, mc_samples=n, seed=20261120, control_variate=False
    )
    cost = run.cost
    assert cost.by_category["crews"] > 0.0
    assert abs(cost.sample_mean - exact.mean) < 3.5 * cost.mean_se
    # The crews' category alone: the call-outs (fee) and the time engaged
    # (rate), to within the same error (the whole cost's bounds it: the
    # other categories vary with it).
    assert cost.by_category["crews"] == pytest.approx(
        exact.by_category["crews"], abs=3.5 * cost.mean_se
    )
    # The system's time down, against each simulation's.
    down = T - np.asarray(run.uptimes)
    error = down.std(ddof=1) / math.sqrt(n)
    assert abs(down.mean() - rbd.expected_events(T).system_downtime) < (
        3.5 * error
    )


def test_a_run_s_default_means_have_the_call_out_crew():
    # The exact means (the control variates) and the conditional run's.
    rbd = three(contract(2))
    T = 200.0
    cost = rbd.cost(T, mc_samples=200, seed=4)
    assert cost.mean == pytest.approx(rbd.expected_cost(T).mean, rel=1e-9)
    assert cost.by_category["crews"] == pytest.approx(
        rbd.expected_cost(T).by_category["crews"], rel=1e-9
    )


def test_units_of_a_standby_group_call_the_crew_in_the_simulation_only():
    rbd = RepairableRBD(
        [("s", "g"), ("g", "t")],
        {
            "g": {
                "reliability": E([LAM]),
                "repairability": E([MU]),
                "standby": {"units": 3, "k": 1},
            }
        },
        repair_crews=1,
        call_out=contract(1),
    )
    with pytest.raises(NotImplementedError, match="standby group"):
        rbd.mean_availability()
    report = rbd.analysis_routes()
    assert report["mean_availability"].route == r.REFUSED
    assert report["availability"].route == r.SIMULATED
    run = rbd.availability(500.0, mc_samples=200, seed=2)
    assert run.cost.by_category["crews"] > 0.0
    # A crew at once for every unit but the third: as two crews, but for
    # the cost.
    two = RepairableRBD(
        [("s", "g"), ("g", "t")],
        {
            "g": {
                "reliability": E([LAM]),
                "repairability": E([MU]),
                "standby": {"units": 3, "k": 1},
            }
        },
        repair_crews=2,
    )
    assert (
        run.sample_mean_availability
        == two.availability(
            500.0, mc_samples=200, seed=2
        ).sample_mean_availability
    )


# -- what it is given, saved and refused --------------------------------------


@pytest.mark.parametrize(
    "call_out, match",
    [
        (3, "must be a dict"),
        ({"fee": 1.0}, "when_waiting"),
        ({"when_waiting": 1, "price": 1.0}, "when_waiting"),
        ({"when_waiting": 0}, "whole number, at least 1"),
        ({"when_waiting": 1.5}, "whole number"),
        ({"when_waiting": 1, "fee": -1.0}, "non-negative"),
        ({"when_waiting": 1, "rate": "2"}, "must be a number"),
    ],
)
def test_a_call_out_that_is_not_one_is_refused(call_out, match):
    with pytest.raises(ValueError, match=match):
        three(call_out)


def test_it_needs_crews_to_wait_for():
    with pytest.raises(ValueError, match="repair_crews is None"):
        three(contract(), crews=None)
    # With a crew for every job none waits: it is never called, and
    # costs nothing.
    enough = three(contract(1), crews=3)
    assert enough.expected_cost_rate() == three(crews=3).expected_cost_rate()
    assert not RepairableRBD(
        parallel(2),
        {
            node: {"reliability": E([LAM]), "repairability": E([MU])}
            for node in "ab"
        },
        repair_crews=2,
        call_out=contract(1),
    ).has_costs


def test_it_is_saved_and_loaded():
    rbd = three(contract(2, 12.5, 0.75), crew_cost_rate=0.25)
    saved = json.loads(json.dumps(rbd.to_dict()))
    assert saved["call_out"] == {"when_waiting": 2, "fee": 12.5, "rate": 0.75}
    loaded = RepairableRBD.from_dict(saved)
    assert loaded.call_out == rbd.call_out
    assert loaded.expected_cost_rate() == rbd.expected_cost_rate()
    assert "call_out" not in three().to_dict()


def test_the_compiled_engine_hands_it_to_python():
    rbd = three(contract(2))
    plan = _event_loop._stream_plan(rbd, 1.0, 0, False)[0]
    assert _compiled.unsupported(rbd, plan, None, numba=True) == (
        "a call-out crew"
    )
    assert _compiled.unsupported(three(), plan, None, numba=True) is None
    assert rbd.analysis_routes()["availability"].engine == "python"


def test_runs_are_not_priced_from_histories_that_do_not_record_it():
    rbd = three(contract(2))
    with pytest.raises(NotImplementedError, match="call-out crew"):
        rbd.npv_distribution(100.0, mc_samples=10, seed=1)
    assert rbd.analysis_routes()["npv_distribution"].route == r.REFUSED
    # Unpriced, it only changes the histories.
    three(contract(2, 0.0, 0.0)).npv_distribution(100.0, mc_samples=10, seed=1)
