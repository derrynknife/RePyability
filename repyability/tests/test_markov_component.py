"""A component given as a Markov chain (``MarkovComponent``): its own
values against closed forms and a matrix exponential, and in a diagram
against the same component built from the node types there were, and
against the simulation."""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy.linalg import expm

from repyability import (
    DegradingNode,
    MarkovComponent,
    NodeState,
    NonRepairableRBD,
    RepairableRBD,
    StandbyModel,
)
from repyability.rbd import _streams

E = surv.Exponential.from_params
EDGES = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
LAM, MU = 0.002, 0.5
TIMES = np.array([0.0, 0.5, 3.0, 40.0, 1000.0])


def unit():
    return {"reliability": E([LAM]), "repairability": E([MU])}


def two_states(**more):
    return MarkovComponent(
        ["up", "down"],
        {("up", "down"): LAM, ("down", "up"): MU},
        up=["up"],
        **more,
    )


RATES = {
    ("working", "degraded"): 0.01,
    ("degraded", "working"): 0.2,
    ("degraded", "failed"): 0.05,
    ("failed", "working"): 0.5,
}


def three_states(**more):
    """Working, degraded (restored, or failing) and failed (repaired)."""
    if "capacity" not in more:
        more["up"] = ["working", "degraded"]
    return MarkovComponent(["working", "degraded", "failed"], RATES, **more)


def generator():
    q = np.zeros((3, 3))
    order = ["working", "degraded", "failed"]
    for (a, b), rate in RATES.items():
        q[order.index(a), order.index(b)] = rate
    return q - np.diag(q.sum(axis=1))


def system(a, **options):
    return RepairableRBD(EDGES, {"a": a, "b": unit(), "c": unit()}, **options)


# -- the component's own values ---------------------------------------------


def test_two_states_are_an_exponential_life_and_repair_in_closed_form():
    component = two_states()
    total = LAM + MU
    assert component.mean_availability() == pytest.approx(
        MU / total, abs=1e-15
    )
    assert component.mean_unavailability() == pytest.approx(
        LAM / total, rel=1e-13
    )
    assert component.failure_frequency() == pytest.approx(
        LAM * MU / total, rel=1e-13
    )
    down = LAM / total * -np.expm1(-total * TIMES)
    np.testing.assert_allclose(
        component.point_unavailability(TIMES), down, rtol=0, atol=1e-13
    )
    np.testing.assert_allclose(
        component.point_availability(TIMES), 1.0 - down, rtol=0, atol=1e-13
    )


def test_three_states_have_their_closed_form_long_run():
    # Balance: w * 0.01 = d * 0.2 + f * 0.5; d * 0.25 = w * 0.01;
    # f * 0.5 = d * 0.05.
    d = 0.01 / 0.25
    f = d * 0.05 / 0.5
    total = 1.0 + d + f
    component = three_states()
    steady = component.steady_state()
    assert steady["working"] == pytest.approx(1.0 / total, rel=1e-13)
    assert steady["degraded"] == pytest.approx(d / total, rel=1e-13)
    assert steady["failed"] == pytest.approx(f / total, rel=1e-13)
    assert component.mean_unavailability() == pytest.approx(
        f / total, rel=1e-13
    )
    assert component.failure_frequency() == pytest.approx(
        d / total * 0.05, rel=1e-13
    )


@pytest.mark.parametrize(
    "start, initial",
    [
        (None, [1.0, 0.0, 0.0]),
        ("failed", [0.0, 0.0, 1.0]),
        ({"degraded": 0.25, "failed": 0.75}, [0.0, 0.25, 0.75]),
    ],
)
def test_the_transient_is_the_matrix_exponential(start, initial):
    component = three_states(start=start)
    q = generator()
    expected = np.array([np.array(initial) @ expm(q * t) for t in TIMES])
    got = component.state_probabilities(TIMES)
    for j, state in enumerate(component.states):
        np.testing.assert_allclose(
            got[state], expected[:, j], rtol=0, atol=1e-12
        )
    np.testing.assert_allclose(
        component.point_unavailability(TIMES),
        expected[:, 2],
        rtol=0,
        atol=1e-12,
    )


def test_an_erlang_life_as_a_chain_is_the_gamma_life():
    chain = MarkovComponent([0, 1, 2], {(0, 1): 0.5, (1, 2): 0.5}, up=[0, 1])
    gamma = surv.Gamma.from_params([2, 0.5])
    x = np.array([0.0, 0.1, 1.0, 5.0, 20.0, 60.0])
    assert not chain.repairable
    np.testing.assert_allclose(chain.sf(x), gamma.sf(x), rtol=0, atol=1e-12)
    np.testing.assert_allclose(chain.ff(x), gamma.ff(x), rtol=0, atol=1e-12)
    np.testing.assert_allclose(chain.df(x), gamma.df(x), rtol=0, atol=1e-12)
    assert chain.mean() == pytest.approx(4.0, rel=1e-13)
    shares = np.array([0.01, 0.5, 0.99])
    np.testing.assert_allclose(chain.qf(shares), gamma.qf(shares), rtol=1e-9)
    draws = chain.random(20_000, seed=11)
    error = draws.std(ddof=1) / math.sqrt(len(draws))
    assert abs(draws.mean() - 4.0) < 4 * error
    assert chain.random(5, seed=3).tolist() == chain.random(5, seed=3).tolist()


def test_a_standby_arrangement_given_as_states_and_rates():
    # Two exponential units, one operating, a warm spare (dormancy 0.5)
    # and a switch that works 9 times in 10: the chain the standby model
    # solves, written out.
    lam, dormancy, switch = 0.01, 0.5, 0.9
    standby = StandbyModel(
        [E([lam]), E([lam])],
        dormancy_factor=dormancy,
        switching_probability=switch,
    )
    chain = MarkovComponent(
        ["both", "one", "none"],
        {
            ("both", "one"): lam * switch + dormancy * lam,
            ("both", "none"): lam * (1.0 - switch),
            ("one", "none"): lam,
        },
        up=["both", "one"],
    )
    x = np.array([1.0, 50.0, 200.0, 600.0])
    np.testing.assert_allclose(chain.sf(x), standby.sf(x), rtol=0, atol=1e-12)
    assert chain.mean() == pytest.approx(standby.mean(), rel=1e-9)


def test_a_degrading_node_given_as_states_and_rates():
    stages = DegradingNode([(100.0, E([0.004])), (50.0, E([0.01]))])
    chain = MarkovComponent(
        ["full", "half", "failed"],
        {("full", "half"): 0.004, ("half", "failed"): 0.01},
        capacity={"full": 100.0, "half": 50.0, "failed": 0.0},
    )
    x = np.array([10.0, 100.0, 400.0])
    # The sum of two exponential stages, in closed form; the degrading
    # node convolves stages of different rates numerically (to 1e-6).
    closed = (0.01 * np.exp(-0.004 * x) - 0.004 * np.exp(-0.01 * x)) / 0.006
    np.testing.assert_allclose(chain.sf(x), closed, rtol=0, atol=1e-12)
    np.testing.assert_allclose(chain.sf(x), stages.sf(x), rtol=0, atol=1e-6)
    assert chain.mean() == pytest.approx(stages.mean(), rel=1e-12)
    assert chain.capacities == (100.0, 50.0)


# -- in a repairable diagram ------------------------------------------------

LONG_RUN = [
    "mean_availability",
    "mean_unavailability",
    "system_failure_frequency",
    "mean_time_between_failures",
    "mean_up_time",
    "mean_down_time",
    "node_availability",
    "birnbaum_importance",
    "criticality_importance",
    "improvement_potential",
    "risk_achievement_worth",
    "risk_reduction_worth",
    "fussell_vesely",
    "barlow_proschan_importance",
]


def _gap(a, b) -> float:
    if hasattr(a, "keys"):
        return max(_gap(a[key], b[key]) for key in a.keys())
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


@pytest.mark.parametrize("method", LONG_RUN)
def test_two_states_give_the_exponential_unit_s_long_run(method):
    chain, plain = system(two_states()), system(unit())
    a, b = getattr(chain, method)(), getattr(plain, method)()
    scale = max(
        1.0,
        _gap(
            b, 0.0 if not hasattr(b, "keys") else dict.fromkeys(b.keys(), 0.0)
        ),
    )
    assert _gap(a, b) <= 1e-12 * scale


def test_two_states_give_the_exponential_unit_s_values_over_time():
    chain, plain = system(two_states()), system(unit())
    # The unavailability of an exponential unit is a closed form, as the
    # chain's is worked out as itself.
    np.testing.assert_allclose(
        chain.point_unavailability(TIMES),
        plain.point_unavailability(TIMES),
        rtol=0,
        atol=1e-12,
    )
    assert chain.mission_unavailability(100.0) == pytest.approx(
        plain.mission_unavailability(100.0), abs=1e-12
    )
    # The plain unit's availability and expected events come from its
    # renewal equation on a grid, to about 1e-7; the chain's are exact.
    np.testing.assert_allclose(
        chain.point_availability(TIMES),
        plain.point_availability(TIMES),
        rtol=0,
        atol=1e-7,
    )
    assert chain.expected_failures(100.0) == pytest.approx(
        plain.expected_failures(100.0), abs=1e-7
    )


def test_two_states_alone_have_their_expected_failures_in_closed_form():
    alone = RepairableRBD([("s", "a"), ("a", "t")], {"a": two_states()})
    total = LAM + MU
    t = 100.0
    # The integral of the availability, times the failure rate.
    up = MU / total * t + LAM / total**2 * -math.expm1(-total * t)
    assert alone.expected_failures(t) == pytest.approx(LAM * up, rel=1e-10)
    assert alone.mission_availability(t) == pytest.approx(up / t, rel=1e-10)


def test_three_states_over_time_are_the_matrix_exponential_in_the_system():
    component = three_states(start={"degraded": 0.5, "failed": 0.5})
    rbd = system(component)
    q = generator()
    a_down = np.array(
        [(np.array([0.0, 0.5, 0.5]) @ expm(q * t))[2] for t in TIMES]
    )
    total = LAM + MU
    other_down = LAM / total * -np.expm1(-total * TIMES)
    # c in series with a and b in parallel.
    down = 1.0 - (1.0 - other_down) * (1.0 - a_down * other_down)
    np.testing.assert_allclose(
        rbd.point_unavailability(TIMES), down, rtol=0, atol=1e-12
    )


def test_two_alike_components_are_not_given_one_curve_from_two_starts():
    rbd = RepairableRBD(
        EDGES,
        {"a": two_states(), "b": two_states(start="down"), "c": unit()},
    )
    available = rbd.node_availability()
    assert available["a"] == available["b"]
    from repyability.rbd import _curves

    assert not _curves._plain_unit(rbd, "a")
    total = LAM + MU
    t = np.array([1.0])
    a_down = LAM / total * -np.expm1(-total * t)
    b_down = LAM / total + MU / total * np.exp(-total * t)
    down = 1.0 - (1.0 - a_down) * (1.0 - a_down * b_down)
    np.testing.assert_allclose(
        rbd.point_unavailability(t), down, rtol=0, atol=1e-12
    )


def test_the_costs_are_per_failure_and_per_unit_of_down_time():
    component = three_states()
    rbd = system(
        {"reliability": component, "repair_cost": 5.0, "downtime_cost": 2.0}
    )
    expected = (
        5.0 * component.failure_frequency()
        + 2.0 * component.mean_unavailability()
    )
    assert rbd.expected_cost_rate() == pytest.approx(expected, rel=1e-12)


def test_capacities_take_each_state_s_level():
    component = three_states(
        capacity={"working": 100.0, "degraded": 60.0, "failed": 0.0}
    )
    rbd = RepairableRBD([("s", "a"), ("a", "t")], {"a": component})
    steady = component.steady_state()
    long_run = rbd.capacity_distribution()
    assert long_run.levels.tolist() == [0.0, 60.0, 100.0]
    np.testing.assert_allclose(
        long_run.probabilities,
        [steady["failed"], steady["degraded"], steady["working"]],
        rtol=1e-12,
    )
    x = np.array([2.0, 30.0])
    q = generator()
    expected = np.array([np.array([1.0, 0, 0]) @ expm(q * t) for t in x])
    at = rbd.point_capacity(x)
    assert at.levels.tolist() == [0.0, 60.0, 100.0]
    np.testing.assert_allclose(
        at.probabilities, expected[:, ::-1].T, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize(
    "start", [None, "failed", {"degraded": 0.5, "failed": 0.5}]
)
def test_the_simulation_agrees_with_the_chain(start):
    rbd = system(three_states(start=start), downtime_cost_rate=3.0)
    window = 200.0
    run = rbd.availability(
        window, mc_samples=20_000, seed=814, control_variate=False
    )
    shares = np.asarray(run.uptimes, dtype=float) / window
    error = shares.std(ddof=1) / math.sqrt(len(shares))
    assert abs(shares.mean() - rbd.mission_availability(window)) < 4 * error
    histories = rbd.simulate_timelines(window, mc_samples=20_000, seed=814)
    assert histories.engine == "python"
    failures = np.asarray(histories.system.failures, dtype=float)
    assert failures.shape == (20_000,)
    error = failures.std(ddof=1) / math.sqrt(len(failures))
    assert abs(failures.mean() - rbd.expected_failures(window)) < 4 * error
    cost = rbd.cost(window, mc_samples=20_000, seed=814, control_variate=False)
    assert abs(cost.mean - rbd.expected_cost(window).mean) < 4 * cost.mean_se


def test_a_seeded_run_is_its_streams_and_leaves_the_others_alone():
    plain = system(unit())
    chain = RepairableRBD(
        EDGES + [("s", "m"), ("m", "c")],
        {"a": unit(), "b": unit(), "c": unit(), "m": three_states()},
    )
    kinds = (_streams.SOJOURN, _streams.MOVE, _streams.MARKOV_START)
    assert len(set(kinds)) == 3
    taken = {
        value
        for name, value in vars(_streams).items()
        if name.isupper()
        and isinstance(value, int)
        and name
        not in (
            "SOJOURN",
            "MOVE",
            "MARKOV_START",
            "BLOCK_DRAWS",
            "MAX_WIDTH",
            "MAX_ROWS",
        )
    } | set(_streams.COST_KINDS.values())
    assert not taken & set(kinds)
    a = chain.availability(50.0, mc_samples=200, seed=5)
    b = chain.availability(50.0, mc_samples=200, seed=5)
    assert a.sample_mean_availability == b.sample_mean_availability
    # The other components draw what they draw without it: their
    # failures are the plain diagram's.
    with_it = chain.simulate_timelines(50.0, mc_samples=50, seed=5)
    without = plain.simulate_timelines(50.0, mc_samples=50, seed=5)
    for node in "abc":
        for one, other in zip(
            with_it.components[node].failures,
            without.components[node].failures,
        ):
            assert np.array_equal(one, other)


def test_the_compiled_engines_leave_it_to_python():
    from repyability.rbd import _compiled, _event_loop

    rbd = system(three_states())
    specs, complete = _event_loop._stream_specs(rbd, 20.0)
    assert complete
    plan = _streams.Plan(1, False, specs)
    for numba in (False, True):
        assert (
            _compiled.unsupported(rbd, plan, None, numba=numba)
            == "Markov components"
        )
    run = rbd.simulate_timelines(20.0, mc_samples=20, seed=1)
    assert run.engine == "python"


# -- refusals ---------------------------------------------------------------


def test_what_it_is_given_is_checked_by_name():
    with pytest.raises(ValueError, match="'c'.*never reached"):
        MarkovComponent(
            ["a", "b", "c"], {("a", "b"): 1.0, ("b", "a"): 1.0}, up=["a"]
        )
    with pytest.raises(ValueError, match="names the state 'x'"):
        MarkovComponent(["a", "b"], {("a", "x"): 1.0}, up=["a"])
    with pytest.raises(ValueError, match="positive and finite"):
        MarkovComponent(["a", "b"], {("a", "b"): 0.0}, up=["a"])
    with pytest.raises(TypeError, match="must be a number"):
        MarkovComponent(["a", "b"], {("a", "b"): "1"}, up=["a"])
    with pytest.raises(ValueError, match="more than once"):
        MarkovComponent(["a", "a"], {("a", "a"): 1.0}, up=["a"])
    with pytest.raises(ValueError, match="one of them"):
        MarkovComponent(["a", "b"], {("a", "b"): 1.0})
    with pytest.raises(ValueError, match="never\n?.*reaches a failed"):
        MarkovComponent(
            ["a", "b", "c"],
            {("a", "b"): 1.0, ("a", "c"): 1.0, ("c", "a"): 1.0},
            up=["a", "b"],
        )
    with pytest.raises(ValueError, match="sum to 1"):
        MarkovComponent(
            ["a", "b"], {("a", "b"): 1.0}, up=["a"], start={"a": 0.5}
        )
    with pytest.raises(ValueError, match="more than one set"):
        MarkovComponent(
            ["a", "b", "c", "d"],
            {
                ("a", "b"): 1.0,
                ("a", "c"): 1.0,
                ("c", "d"): 1.0,
                ("d", "c"): 1.0,
            },
            up=["a", "d"],
        )


def test_a_lifetime_and_a_repairable_chain_each_say_where_they_go():
    life = MarkovComponent([0, 1], {(0, 1): 0.1}, up=[0])
    with pytest.raises(NotImplementedError, match="has a lifetime"):
        life.mean_availability()
    with pytest.raises(ValueError, match="'repairability'"):
        system(life)
    with pytest.raises(NotImplementedError, match="RepairableRBD"):
        two_states().sf(1.0)
    with pytest.raises(ValueError, match="RepairableRBD"):
        NonRepairableRBD([("s", "a"), ("a", "t")], {"a": two_states()})
    # A lifetime is a life: a node, or a spec's reliability.
    rbd = system({"reliability": life, "repairability": E([MU])})
    plain = system({"reliability": E([0.1]), "repairability": E([MU])})
    assert rbd.mean_availability() == pytest.approx(
        plain.mean_availability(), rel=1e-12
    )


def test_what_a_repairable_chain_does_not_take_is_refused_with_the_reason():
    component = three_states()
    for key, value in (
        ("repairability", E([1.0])),
        ("preventive", {"interval": 10.0}),
        ("inspection", {"interval": 10.0}),
        ("spares", {"stock": 1, "lead_time": 5.0}),
        ("group", "g"),
    ):
        with pytest.raises(ValueError, match="own transitions"):
            system({"reliability": component, key: value})
    from repyability import BetaFactor, CCFGroup

    with pytest.raises(ValueError, match="Markov component"):
        RepairableRBD(
            EDGES,
            {"a": component, "b": unit(), "c": unit()},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        )
    rbd = system(component)
    with pytest.raises(NotImplementedError, match="start="):
        rbd.point_availability([1.0], state={"a": NodeState(age=3.0)})
    routes = rbd.analysis_routes()
    for method in ("first_failure_sf", "mean_time_to_first_failure"):
        # Its chain goes into the first failure's, its failed states
        # absorbing (see test_first_failure_groups.py).
        assert routes[method].route == "numerical"
    for method in ("spares_demand", "spares_stock", "plan_vs_actual"):
        assert routes[method].route == "refused"
        assert "Markov component" in routes[method].reason
        assert "units" not in routes[method].reason
    assert routes["mean_availability"].route == "exact"
    assert routes["point_availability"].route == "numerical"
    assert routes["availability"].route == "simulated"


# -- around it --------------------------------------------------------------


def test_each_rate_is_a_lever():
    component = three_states()
    rbd = system(component)
    levers = {lever.name: lever for lever in rbd.levers() if lever.key == "a"}
    assert list(levers) == [
        "reliability.rate:working->degraded",
        "reliability.rate:degraded->working",
        "reliability.rate:degraded->failed",
        "reliability.rate:failed->working",
    ]
    lever = levers["reliability.rate:failed->working"]
    assert lever.value == 0.5
    assert tuple(lever.bounds) == (0.0, math.inf)
    moved = rbd.with_levers({("a", lever.name): 1.0})
    again = system(component.with_rates({("failed", "working"): 1.0}))
    assert moved.mean_availability() == again.mean_availability()
    with pytest.raises(ValueError):
        rbd.with_levers({("a", lever.name): -1.0})
    # The derivative of the long-run availability in the repair rate.
    step = 1e-6
    up, down = (
        system(
            component.with_rates({("failed", "working"): 0.5 + sign * step})
        ).mean_availability()
        for sign in (1, -1)
    )
    sensitivity = rbd.parameter_sensitivity()["a"][lever.name]
    assert sensitivity == pytest.approx((up - down) / (2 * step), rel=1e-5)


def test_rates_scale_inversely_with_the_unit_of_time():
    for scale in (1e-6, 1e6):
        scaled = MarkovComponent(
            ["working", "degraded", "failed"],
            {move: rate / scale for move, rate in RATES.items()},
            up=["working", "degraded"],
        )
        base = three_states()
        assert scaled.mean_availability() == pytest.approx(
            base.mean_availability(), rel=1e-12
        )
        assert scaled.failure_frequency() * scale == pytest.approx(
            base.failure_frequency(), rel=1e-12
        )
        np.testing.assert_allclose(
            scaled.point_unavailability(TIMES * scale),
            base.point_unavailability(TIMES),
            rtol=0,
            atol=1e-12,
        )
        rbd = RepairableRBD([("s", "a"), ("a", "t")], {"a": scaled})
        one = RepairableRBD([("s", "a"), ("a", "t")], {"a": base})
        assert rbd.expected_failures(100.0 * scale) == pytest.approx(
            one.expected_failures(100.0), rel=1e-9
        )


def test_uncertain_rates_are_drawn_by_name():
    rbd = system(three_states())
    spread = surv.LogNormal.from_params([math.log(0.5), 0.2])
    result = rbd.mean_availability_uncertainty(
        {"a": {"reliability": {"rate:failed->working": spread}}},
        n_draws=200,
        seed=1,
    )
    assert result.band == "given"
    samples = np.asarray(result.samples, dtype=float)
    assert samples.min() < result.nominal < samples.max()
    assert result.nominal == rbd.mean_availability()
    with pytest.raises(ValueError, match="Markov component"):
        rbd.mean_availability_uncertainty(
            {"a": {"reliability": {"rate:nowhere": spread}}},
            n_draws=10,
            seed=1,
        )


def test_it_is_saved_and_loaded():
    component = three_states(
        capacity={"working": 100.0, "degraded": 60.0, "failed": 0.0},
        start={"degraded": 0.5, "failed": 0.5},
    )
    saved = json.loads(json.dumps(component.to_dict()))
    assert MarkovComponent.from_dict(saved).to_dict() == component.to_dict()
    priced = system(
        {"reliability": three_states(), "repair_cost": 5.0},
        downtime_cost_rate=1.0,
    )
    for rbd in (system(component, capacity={"b": 50.0}), priced):
        loaded = RepairableRBD.from_json(rbd.to_json())
        assert loaded.to_json() == rbd.to_json()
        assert loaded.mean_availability() == rbd.mean_availability()
    assert (
        RepairableRBD.from_json(priced.to_json()).expected_cost_rate()
        == priced.expected_cost_rate()
    )
    life = MarkovComponent(
        [("ok", 1), ("ok", 2), "failed"],
        {(("ok", 1), ("ok", 2)): 0.5, (("ok", 2), "failed"): 0.5},
        up=[("ok", 1), ("ok", 2)],
    )
    diagram = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": life})
    loaded = NonRepairableRBD.from_json(diagram.to_json())
    assert loaded.sf(3.0) == diagram.sf(3.0)


def test_a_lifetime_s_capacities_are_a_degrading_node_s_in_a_diagram():
    chain = MarkovComponent(
        ["full", "half", "failed"],
        {("full", "half"): 0.004, ("half", "failed"): 0.004},
        capacity={"full": 100.0, "half": 50.0, "failed": 0.0},
    )
    stages = DegradingNode([(100.0, E([0.004])), (50.0, E([0.004]))])
    x = np.array([100.0, 300.0])
    got, expected = (
        NonRepairableRBD(
            [("s", "a"), ("a", "b"), ("b", "t")],
            {"a": model, "b": E([0.001])},
            capacity={"b": 80.0},
        ).capacity_distribution(x)
        for model in (chain, stages)
    )
    assert got.levels.tolist() == expected.levels.tolist()
    np.testing.assert_allclose(
        got.probabilities, expected.probabilities, rtol=0, atol=1e-12
    )
    # As a spec's life its states are not followed through the repairs.
    with pytest.raises(ValueError, match="capacity in each state"):
        system({"reliability": chain, "repairability": E([MU])})
    system(
        {"reliability": chain, "repairability": E([MU])},
        capacity={"a": 100.0},
    )


def test_one_in_a_nested_diagram_is_simulated_as_its_chain():
    inner = RepairableRBD(
        [("s", "m"), ("m", "x"), ("x", "t")],
        {"m": three_states(), "x": unit()},
    )
    outer = RepairableRBD(
        [("s", "n"), ("s", "b"), ("n", "t"), ("b", "t")],
        {
            "n": inner,
            "b": {"reliability": E([0.05]), "repairability": E([0.1])},
        },
    )
    window = 300.0
    run = outer.availability(
        window, mc_samples=20_000, seed=7, control_variate=False
    )
    shares = np.asarray(run.uptimes, dtype=float) / window
    error = shares.std(ddof=1) / math.sqrt(len(shares))
    assert abs(shares.mean() - outer.mission_availability(window)) < 4 * error
    histories = outer.simulate_timelines(window, mc_samples=20_000, seed=7)
    failures = np.asarray(histories.system.failures, dtype=float)
    error = failures.std(ddof=1) / math.sqrt(len(failures))
    assert abs(failures.mean() - outer.expected_failures(window)) < 4 * error


# -- capacities simulated by state, and crews beside it ----------------------

LEVELS = {"working": 100.0, "degraded": 60.0, "failed": 0.0}


def test_the_simulation_follows_the_capacity_of_each_state():
    # Alone in series, the system's capacity is the component's state's:
    # its time at 0 is the system's down time, and at 60 and 100 its up
    # time, simulation by simulation.
    alone = RepairableRBD(
        [("s", "a"), ("a", "t")], {"a": three_states(capacity=LEVELS)}
    )
    result = alone.availability(
        200.0, mc_samples=400, seed=3, control_variate=False
    )
    time = result.capacity_time
    assert set(time) <= {0.0, 60.0, 100.0}
    assert time[60.0] > 0.0
    assert time.get(0.0, 0.0) == pytest.approx(result.system_downtime)
    assert time[60.0] + time[100.0] == pytest.approx(result.system_uptime)
    # The expected capacity over the window, as the exact methods give
    # it: within four standard errors of the simulations'.
    exact = alone.mission_capacity(200.0)
    # Each simulation's mean capacity: against the design capacity, 100,
    # the fraction of it delivered.
    assert result.demand == 100.0
    each = 100.0 * np.asarray(result.delivered, dtype=float)
    error = each.std(ddof=1) / math.sqrt(len(each))
    assert abs(each.mean() - exact.mean) <= 4.0 * error


def test_a_capacity_by_state_beside_other_nodes_against_the_exact_values():
    rbd = system(
        three_states(capacity=LEVELS), capacity={"b": 100.0, "c": 150.0}
    )
    result = rbd.availability(
        300.0, mc_samples=2000, seed=11, control_variate=False
    )
    exact = rbd.mission_capacity(300.0)
    assert result.demand == 150.0
    each = 150.0 * np.asarray(result.delivered, dtype=float)
    error = each.std(ddof=1) / math.sqrt(len(each))
    assert abs(each.mean() - exact.mean) <= 4.0 * error
    shares = {
        level: spent / (result.n_simulations * 300.0)
        for level, spent in result.capacity_time.items()
    }
    for level, p in zip(exact.levels, exact.probabilities):
        assert shares.get(float(level), 0.0) == pytest.approx(p, abs=3e-3)


def test_crews_serve_the_other_components_beside_a_markov_component():
    # A Markov component takes no crew: with one crew for b and c, the
    # system is the component beside the crews' chain of b and c, as
    # with b and c in a nested diagram with the crew of their own.
    pair = [("s", "b"), ("s", "c"), ("b", "t"), ("c", "t")]
    parts = {
        "b": unit(),
        "c": {"reliability": E([0.02]), "repairability": E([0.2])},
    }
    flat = RepairableRBD(
        [("s", "m"), ("m", "b"), ("m", "c"), ("b", "t"), ("c", "t")],
        {"m": three_states(), **parts},
        repair_crews=1,
    )
    nested = RepairableRBD(
        [("s", "m"), ("m", "n"), ("n", "t")],
        {"m": three_states(), "n": RepairableRBD(pair, parts, repair_crews=1)},
    )
    assert [job.node for job in flat.crew_jobs()] == ["b", "c"]
    x = np.array([1.0, 20.0, 200.0])
    for value in (
        lambda r: r.mean_availability(),
        lambda r: r.system_failure_frequency(),
        lambda r: r.point_availability(x),
        lambda r: r.point_unavailability(x),
        lambda r: r.mission_availability(200.0),
        lambda r: r.expected_failures(x),
    ):
        np.testing.assert_allclose(value(flat), value(nested), rtol=1e-12)
    # The first failure: the component's beside the pair's with the crew.
    alone = RepairableRBD([("s", "m"), ("m", "t")], {"m": three_states()})
    crewed = RepairableRBD(pair, parts, repair_crews=1)
    np.testing.assert_allclose(
        flat.first_failure_sf(x),
        alone.first_failure_sf(x) * crewed.first_failure_sf(x),
        rtol=1e-12,
    )
    # The simulation, with the crew taking b's and c's jobs only.
    result = flat.availability(
        200.0, mc_samples=2000, seed=7, control_variate=False
    )
    each = np.asarray(result.uptimes, dtype=float) / 200.0
    error = each.std(ddof=1) / math.sqrt(len(each))
    assert abs(each.mean() - flat.mission_availability(200.0)) <= 4.0 * error
