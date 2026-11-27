"""A network of repaired links and nodes: the connection's availability
over time and in the long run, and how often it is lost.

The values are checked against the ``RepairableRBD`` of the same system
(a network that is a series-parallel diagram), the bridge's polynomial,
the Markov chain of the bridge's 32 states, a sum over every state of the
elements, and a simulation written from the definition.
"""

import itertools
import json

import numpy as np
import pytest
import surpyval as surv
from scipy.linalg import expm

from repyability import Network, RepairableRBD
from repyability.rbd.routes import Refused

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LN = surv.LogNormal.from_params

TIMES = np.array([0.0, 0.5, 3.0, 20.0, 150.0, 2000.0])

OVER_TIME = (
    "point_availability",
    "point_unavailability",
    "mission_availability",
    "mission_unavailability",
    "expected_failures",
)
LONG_RUN = (
    "mean_availability",
    "mean_unavailability",
    "system_failure_frequency",
)

BRIDGE = {
    "a": ("s", "x"),
    "b": ("s", "y"),
    "c": ("x", "y"),
    "d": ("x", "t"),
    "e": ("y", "t"),
}
BRIDGE_PATHS = [{"a", "d"}, {"b", "e"}, {"a", "c", "e"}, {"b", "c", "d"}]


def spec(life, repair):
    return {"reliability": life, "repairability": repair}


def general():
    """Five components of different lives and repairs."""
    return {
        "a": spec(E([0.01]), E([0.5])),
        "b": spec(W([80.0, 1.5]), E([0.4])),
        "c": spec(E([0.03]), LN([1.0, 0.5])),
        "d": spec(W([120.0, 0.8]), W([4.0, 2.0])),
        "e": spec(E([0.002]), E([0.05])),
    }


def exponential():
    """The bridge's links with exponential lives and repairs: each one's
    failure and repair rates."""
    return {
        "a": (0.010, 0.50),
        "b": (0.020, 0.40),
        "c": (0.030, 0.30),
        "d": (0.015, 0.25),
        "e": (0.025, 0.60),
    }


def exponential_bridge():
    rates = exponential()
    return Network(
        {
            name: (u, v, spec(E([rates[name][0]]), E([rates[name][1]])))
            for name, (u, v) in BRIDGE.items()
        },
        "s",
        "t",
    )


def bridge_works(up) -> bool:
    return any(path <= up for path in BRIDGE_PATHS)


# -- against the diagram of the same system ----


def series_parallel():
    """Networks that are series-parallel diagrams, each with the
    ``RepairableRBD`` of the same system."""
    c = general()
    cases = {}
    cases["series"] = (
        Network(
            {
                "a": ("s", 1, c["a"]),
                "b": (1, 2, c["b"]),
                "c": (2, "t", c["c"]),
            },
            "s",
            "t",
        ),
        RepairableRBD(
            [("in", "a"), ("a", "b"), ("b", "c"), ("c", "out")],
            {n: c[n] for n in "abc"},
            input_node="in",
            output_node="out",
        ),
    )
    cases["parallel"] = (
        Network({n: ("s", "t", c[n]) for n in "abd"}, "s", "t"),
        RepairableRBD(
            [("in", n) for n in "abd"] + [(n, "out") for n in "abd"],
            {n: c[n] for n in "abd"},
            input_node="in",
            output_node="out",
        ),
    )
    # a and b one after the other, beside c; then d; the middle node
    # (e) can fail too.
    cases["series-parallel with a node"] = (
        Network(
            {
                "a": ("s", "m", c["a"]),
                "b": ("m", "x", c["b"]),
                "c": ("s", "x", c["c"]),
                "d": ("x", "t", c["d"]),
            },
            "s",
            "t",
            nodes={"x": c["e"]},
        ),
        RepairableRBD(
            [
                ("in", "a"),
                ("a", "b"),
                ("in", "c"),
                ("b", "e"),
                ("c", "e"),
                ("e", "d"),
                ("d", "out"),
            ],
            c,
            input_node="in",
            output_node="out",
        ),
    )
    return cases


@pytest.mark.parametrize("name", sorted(series_parallel()))
def test_a_series_parallel_network_is_its_diagram(name):
    net, rbd = series_parallel()[name]
    if name.endswith("node"):
        # The network names its failing node, the diagram its block.
        net = Network(
            {k: (u, v, net.models[k]) for k, (u, v) in net.links.items()},
            "s",
            "t",
            nodes={"x": net.models["x"]},
        )
    for method in (
        "point_unavailability",
        "mission_unavailability",
        "expected_failures",
    ):
        ours = getattr(net, method)(TIMES)
        theirs = getattr(rbd, method)(TIMES)
        np.testing.assert_allclose(
            ours, theirs, rtol=1e-12, atol=0, err_msg=method
        )
    # The network's availability is one less its unavailability, each
    # element's from its closed form or its curve of being down; the
    # diagram's is from the curves of being up (on their grids, 3.2e-9
    # apart before the early grids, with their stated 4e-6 allowed).
    for method in ("point_availability", "mission_availability"):
        ours = getattr(net, method)(TIMES)
        np.testing.assert_allclose(
            ours,
            1.0 - getattr(net, method.replace("_a", "_una"))(TIMES),
            rtol=1e-15,
        )
        np.testing.assert_allclose(
            ours, getattr(rbd, method)(TIMES), rtol=0, atol=2e-9
        )
    for method in LONG_RUN:
        assert getattr(net, method)() == pytest.approx(
            getattr(rbd, method)(), rel=1e-12, abs=0
        ), method


@pytest.mark.parametrize("name", sorted(series_parallel()))
def test_the_failure_frequency_integrates_to_the_expected_failures(name):
    # The integral of w(t) over a window is the diagram's expected
    # failures in it (Simpson's rule on 2,000 steps).
    net, rbd = series_parallel()[name]
    t = np.linspace(3.0, 150.0, 2001)
    w = net.failure_frequency(t)
    integral = (
        (w[0] + w[-1] + 4 * w[1:-1:2].sum() + 2 * w[2:-1:2].sum())
        * (t[1] - t[0])
        / 3
    )
    count = rbd.expected_failures(150.0) - rbd.expected_failures(3.0)
    assert integral == pytest.approx(count, rel=1e-5)
    assert net.failure_frequency(1e6) == pytest.approx(
        rbd.system_failure_frequency(), rel=1e-6
    )


# -- the bridge ----


def polynomial(p):
    return 2 * p**2 + 2 * p**3 - 5 * p**4 + 2 * p**5


def test_the_bridge_is_its_polynomial_at_the_links_availabilities():
    lam, mu = 0.02, 0.3
    link = spec(E([lam]), E([mu]))
    net = Network({n: (u, v, link) for n, (u, v) in BRIDGE.items()}, "s", "t")
    a = mu / (lam + mu) + lam / (lam + mu) * np.exp(-(lam + mu) * TIMES)
    np.testing.assert_allclose(
        net.point_availability(TIMES), polynomial(a), rtol=1e-13
    )
    assert net.mean_availability() == pytest.approx(
        polynomial(mu / (lam + mu)), rel=1e-13
    )
    # 1 - polynomial, from q: 2q^2 + 2q^3 - 5q^4 + 2q^5 (the bridge is
    # its own dual).
    q = lam / (lam + mu) * -np.expm1(-(lam + mu) * TIMES)
    np.testing.assert_allclose(
        net.point_unavailability(TIMES), polynomial(q), rtol=1e-12
    )
    assert net.mean_unavailability() == pytest.approx(
        polynomial(lam / (lam + mu)), rel=1e-13
    )


def bridge_chain():
    """The Markov chain of the bridge's 32 states (which links are up):
    its generator, the state all up, whether each state joins the
    terminals, and each state's rate of losing the connection."""
    rates = exponential()
    names = list(rates)
    states = list(itertools.product([True, False], repeat=5))
    index = {s: i for i, s in enumerate(states)}
    generator = np.zeros((32, 32))
    works = np.array(
        [bridge_works({n for n, up in zip(names, s) if up}) for s in states]
    )
    loss = np.zeros(32)
    for s in states:
        for k, name in enumerate(names):
            lam, mu = rates[name]
            other = s[:k] + (not s[k],) + s[k + 1 :]
            rate = lam if s[k] else mu
            generator[index[s], index[other]] += rate
            generator[index[s], index[s]] -= rate
            if s[k] and works[index[s]] and not works[index[other]]:
                loss[index[s]] += lam
    return generator, index[(True,) * 5], works, loss


def test_the_bridge_against_its_markov_chain():
    net = exponential_bridge()
    generator, start, works, loss = bridge_chain()
    # The chain with two counters: the time spent parted, and the losses
    # of the connection.
    augmented = np.zeros((34, 34))
    augmented[:32, :32] = generator
    augmented[:32, 32] = ~works
    augmented[:32, 33] = loss
    for t in TIMES[1:]:
        row = expm(augmented * t)[start]
        state = row[:32]
        assert net.point_availability(t) == pytest.approx(
            state[works].sum(), rel=1e-12
        )
        assert net.point_unavailability(t) == pytest.approx(
            state[~works].sum(), rel=1e-10
        )
        assert net.failure_frequency(t) == pytest.approx(
            state @ loss, rel=1e-10
        )
        assert net.mission_unavailability(t) == pytest.approx(
            row[32] / t, rel=1e-6
        )
        assert net.mission_availability(t) == pytest.approx(
            1 - row[32] / t, rel=1e-8
        )
        assert net.expected_failures(t) == pytest.approx(row[33], rel=1e-6)
    # In the long run: the chain's stationary distribution.
    a = np.vstack([generator.T, np.ones(32)])
    steady = np.linalg.lstsq(a, np.r_[np.zeros(32), 1.0], rcond=None)[0]
    assert net.mean_availability() == pytest.approx(
        steady[works].sum(), rel=1e-12
    )
    assert net.mean_unavailability() == pytest.approx(
        steady[~works].sum(), rel=1e-10
    )
    assert net.system_failure_frequency() == pytest.approx(
        steady @ loss, rel=1e-10
    )


def test_the_frequency_is_a_sum_over_the_elements_states():
    # Weibull lives and general repairs, in the long run: each link is up
    # MTTF / (MTTF + MTTR) of the time and fails 1 / (MTTF + MTTR) times
    # per unit time; a failure parts the terminals in the states of the
    # others where the link is critical.
    c = general()
    net = Network({n: (u, v, c[n]) for n, (u, v) in BRIDGE.items()}, "s", "t")
    up, rate = {}, {}
    for name, s in c.items():
        mttf = float(s["reliability"].mean())
        mttr = float(s["repairability"].mean())
        up[name] = mttf / (mttf + mttr)
        rate[name] = 1.0 / (mttf + mttr)
    names = list(c)
    frequency = availability = 0.0
    for state in itertools.product([True, False], repeat=5):
        working = {n for n, s in zip(names, state) if s}
        probability = np.prod(
            [up[n] if n in working else 1 - up[n] for n in names]
        )
        availability += probability * bridge_works(working)
        for name in working:
            if bridge_works(working) and not bridge_works(working - {name}):
                # The link is up and critical in this state.
                frequency += probability / up[name] * rate[name]
    assert net.mean_availability() == pytest.approx(availability, rel=1e-12)
    assert net.system_failure_frequency() == pytest.approx(
        frequency, rel=1e-12
    )


def test_the_frequency_over_time_is_a_sum_over_the_elements_states():
    net = exponential_bridge()
    rates = exponential()
    names = list(rates)
    t = 7.0
    up = {
        n: mu / (lam + mu) + lam / (lam + mu) * np.exp(-(lam + mu) * t)
        for n, (lam, mu) in rates.items()
    }
    frequency = 0.0
    for state in itertools.product([True, False], repeat=5):
        working = {n for n, s in zip(names, state) if s}
        probability = np.prod(
            [up[n] if n in working else 1 - up[n] for n in names]
        )
        for name in working:
            if bridge_works(working) and not bridge_works(working - {name}):
                frequency += probability * rates[name][0]
    assert net.failure_frequency(t) == pytest.approx(frequency, rel=1e-12)


# -- several terminals, one-way links ----


@pytest.mark.parametrize(
    "options",
    [
        {"terminals": ["s", "t", "x"]},
        {"terminals": "all"},
        {"source": "s", "target": "t", "directed": True},
    ],
    ids=["three terminals", "all terminals", "directed"],
)
def test_other_connections_at_the_elements_availabilities(options):
    # The network of fixed probabilities at the links' availabilities at
    # a time is the repaired network then.
    c = general()
    net = Network({n: (u, v, c[n]) for n, (u, v) in BRIDGE.items()}, **options)
    for t in (2.0, 40.0):
        down = {
            n: float(
                RepairableRBD(
                    [("in", n), (n, "out")],
                    {n: c[n]},
                    input_node="in",
                    output_node="out",
                ).point_unavailability(t)
            )
            for n in c
        }
        fixed = Network(
            {n: (u, v, down[n]) for n, (u, v) in BRIDGE.items()}, **options
        )
        assert net.point_unavailability(t) == pytest.approx(
            fixed.ff(), rel=1e-12
        )
        assert net.point_availability(t) == pytest.approx(fixed.sf(), rel=1e-7)
        for measure in (
            "birnbaum_importance",
            "improvement_potential",
            "risk_achievement_worth",
            "risk_reduction_worth",
            "criticality_importance",
            "fussell_vesely",
            "joint_importance",
            "differential_importance",
        ):
            ours, theirs = getattr(net, measure)(t), getattr(fixed, measure)()
            assert ours.keys() == theirs.keys()
            for key in ours:
                assert ours[key] == pytest.approx(
                    theirs[key], rel=1e-9, abs=1e-300
                ), (measure, key)
    assert net.structural_importance() == fixed.structural_importance()
    assert net.minimal_cut_sets() == fixed.minimal_cut_sets()
    assert net.minimal_path_sets() == fixed.minimal_path_sets()
    table = net.importances(40.0)
    assert "barlow_proschan_importance" in table.refused
    assert "birnbaum_importance" in table.measures


# -- a small unavailability keeps its digits ----


def test_a_small_unavailability_keeps_its_digits():
    lam, mu = 1e-7, 2.0
    line = spec(E([lam]), E([mu]))
    net = Network({n: ("s", "t", line) for n in "abc"}, "s", "t")
    t = np.array([0.1, 1.0, 50.0])
    q = lam / (lam + mu) * -np.expm1(-(lam + mu) * t)
    np.testing.assert_allclose(net.point_unavailability(t), q**3, rtol=1e-12)
    assert net.mean_unavailability() == pytest.approx(
        (lam / (lam + mu)) ** 3, rel=1e-12
    )
    # One less the availability holds none of it.
    assert 1.0 - net.point_availability(50.0) == 0.0
    # Its average over a window, to its own size.
    for end in (1.0, 50.0):
        s = np.linspace(0.0, end, 200_001)
        f = (lam / (lam + mu) * -np.expm1(-(lam + mu) * s)) ** 3
        mean = np.sum((f[1:] + f[:-1]) / 2) * (s[1] - s[0]) / end
        assert net.mission_unavailability(end) == pytest.approx(mean, rel=1e-6)


# -- simulation ----


def test_against_a_simulation_from_the_definition():
    # Each link alternates lives and repairs of its own; the terminals
    # are joined while a path of working links joins them.
    c = general()
    links = {n: (u, v, c[n]) for n, (u, v) in BRIDGE.items()}
    net = Network(links, "s", "t")
    rng = np.random.default_rng(817)
    samples, horizon = 4000, 60.0
    checks = np.array([5.0, 25.0, 60.0])
    up_at = np.zeros((samples, len(checks)), dtype=bool)
    losses = np.zeros(samples)
    downtime = np.zeros(samples)
    names = list(c)
    for r in range(samples):
        changes = []
        for name in names:
            t, working = 0.0, True
            while t < horizon:
                key = "reliability" if working else "repairability"
                model = c[name][key]
                t += float(model.random(1, random_state=rng)[0])
                if t < horizon:
                    changes.append((t, name))
                working = not working
        changes.sort()
        working = set(names)
        joined, last, k = True, 0.0, 0
        for t, name in changes + [(horizon, None)]:
            while k < len(checks) and checks[k] <= t:
                up_at[r, k] = joined
                k += 1
            if not joined:
                downtime[r] += t - last
            last = t
            if name is None:
                break
            working ^= {name}
            now = bridge_works(working)
            losses[r] += joined and not now
            joined = now

    def within(estimates, value, error=None):
        if error is None:
            error = estimates.std(ddof=1) / np.sqrt(samples)
        assert abs(estimates.mean() - value) < 4 * error, (
            estimates.mean(),
            value,
            error,
        )

    for k, t in enumerate(checks):
        # A proportion's standard error, at the value checked: the
        # sample's own is 0 when no simulation is parted then.
        q = net.point_unavailability(t)
        within(1.0 - up_at[:, k], q, np.sqrt(q * (1 - q) / samples))
    within(losses, net.expected_failures(horizon))
    within(downtime / horizon, net.mission_unavailability(horizon))


# -- the unit of time ----


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1e3, 1e6])
def test_results_scale_with_the_unit_of_time(scale):
    def bridge(k):
        c = {
            "a": spec(E([0.01 / k]), E([0.5 / k])),
            "b": spec(W([80.0 * k, 1.5]), E([0.4 / k])),
            "c": spec(E([0.03 / k]), LN([1.0 + np.log(k), 0.5])),
            "d": spec(W([120.0 * k, 0.8]), W([4.0 * k, 2.0])),
            "e": spec(E([0.002 / k]), E([0.05 / k])),
        }
        return Network(
            {n: (u, v, c[n]) for n, (u, v) in BRIDGE.items()}, "s", "t"
        )

    base, scaled = bridge(1.0), bridge(scale)
    t = np.array([0.5, 3.0, 20.0, 150.0])
    for method in OVER_TIME:
        np.testing.assert_allclose(
            getattr(scaled, method)(t * scale),
            getattr(base, method)(t),
            rtol=1e-8,
            err_msg=method,
        )
    np.testing.assert_allclose(
        scaled.failure_frequency(t * scale) * scale,
        base.failure_frequency(t),
        rtol=1e-8,
    )
    assert scaled.mean_availability() == pytest.approx(
        base.mean_availability(), rel=1e-10
    )
    assert scaled.mean_unavailability() == pytest.approx(
        base.mean_unavailability(), rel=1e-10
    )
    assert scaled.system_failure_frequency() * scale == pytest.approx(
        base.system_failure_frequency(), rel=1e-10
    )


# -- inputs, shapes, saving, refusals ----


def test_a_number_gives_a_float_and_an_array_an_array():
    net = exponential_bridge()
    for method in OVER_TIME + ("failure_frequency",):
        one = getattr(net, method)(10.0)
        assert isinstance(one, float), method
        many = getattr(net, method)([[10.0, 20.0], [30.0, 40.0]])
        assert many.shape == (2, 2), method
        # Whatever else is asked for: the elements' curves are their
        # closed forms (on a grid to the last time asked for, the
        # integrals over a window were 1e-7 apart).
        assert many[0, 0] == pytest.approx(one, rel=1e-12, abs=0), method
    for method in LONG_RUN:
        assert isinstance(getattr(net, method)(), float)


def test_new_at_0_the_terminals_are_joined():
    net = exponential_bridge()
    assert net.point_availability(0.0) == 1.0
    assert net.point_unavailability(0.0) == 0.0
    assert net.expected_failures(0.0) == 0.0
    assert net.failure_frequency(0.0) == 0.0


def test_a_repaired_network_is_saved_and_rebuilt():
    c = general()
    net = Network(
        {n: (u, v, c[n]) for n, (u, v) in list(BRIDGE.items())[:4]},
        "s",
        "t",
        nodes={"x": c["e"]},
    )
    again = Network.from_dict(json.loads(json.dumps(net.to_dict())))
    assert again.is_repairable
    np.testing.assert_array_equal(
        again.point_unavailability(TIMES), net.point_unavailability(TIMES)
    )
    assert Network.from_json(net.to_json()).mean_availability() == (
        net.mean_availability()
    )


def test_a_negative_time_is_refused():
    net = exponential_bridge()
    for method in OVER_TIME + ("failure_frequency",):
        with pytest.raises(ValueError, match="must be 0 or more"):
            getattr(net, method)(-1.0)
    with pytest.raises(ValueError, match="must be 0 or more"):
        net.birnbaum_importance(-1.0)


def test_an_element_takes_no_costs_or_maintenance():
    # A life, a repair and tests only (mixed and tested elements are in
    # test_network_mixed.py).
    priced = {**spec(E([0.01]), E([0.5])), "cost": 10.0}
    with pytest.raises(ValueError, match="'inspection' only"):
        Network({"a": ("s", "t", priced)}, "s", "t")
    with pytest.raises(ValueError, match="RepairableRBD"):
        Network(
            {"a": ("s", "t", spec(E([0.01]), E([0.5])))},
            "s",
            "t",
            nodes={"s": priced},
        )


def test_a_network_of_lifetimes_has_no_availability():
    net = Network({"a": ("s", "t", W([100.0, 2.0]))}, "s", "t")
    assert not net.is_repairable
    for method in OVER_TIME + ("failure_frequency",):
        with pytest.raises(Refused, match="component specs"):
            getattr(net, method)(1.0)
    for method in LONG_RUN:
        with pytest.raises(Refused, match="component specs"):
            getattr(net, method)()


@pytest.mark.parametrize(
    "call",
    [
        lambda net: net.sf(1.0),
        lambda net: net.ff(1.0),
        lambda net: net.mean(),
        lambda net: net.random(3, seed=1),
        lambda net: net.bx_life(10),
        lambda net: net.barlow_proschan_importance(1.0),
    ],
)
def test_a_repaired_network_has_no_lifetime(call):
    with pytest.raises(NotImplementedError, match="point_availability"):
        call(exponential_bridge())


@pytest.mark.parametrize(
    "call",
    [
        lambda net: net.levers(),
        lambda net: net.with_levers({}),
        lambda net: net.parameter_sensitivity(1.0),
        lambda net: net.tornado(1.0),
        lambda net: net.to_rbd(),
    ],
)
def test_what_a_repaired_network_does_not_give_yet(call):
    with pytest.raises(NotImplementedError, match="not available .* yet"):
        call(exponential_bridge())


def test_every_public_method_is_given_or_refused():
    # A method added later refuses a repaired network until it is listed.
    from repyability import network

    public = {
        name
        for name, value in vars(Network).items()
        if callable(value) and not name.startswith("_")
    }
    assert network._REPAIRED_GIVES <= public
    assert network._NEEDS_A_LIFETIME <= public - network._REPAIRED_GIVES


def test_a_repaired_network_too_large_says_what_can_be_done(monkeypatch):
    from repyability import network

    monkeypatch.setattr(network, "MAX_STATES", 3)
    net = exponential_bridge()
    routes = net.analysis_routes()
    for method in OVER_TIME + ("failure_frequency",) + LONG_RUN:
        with pytest.raises(NotImplementedError) as error:
            getattr(net, method)(*(() if method in LONG_RUN else (1.0,)))
        assert "MAX_STATES, can be raised" in str(error.value)
        assert "method=" not in str(error.value)
        assert routes[method].route == "refused"
        assert routes[method].reason == str(error.value)
    assert routes["birnbaum_importance"].reason == str(error.value)


def test_the_routes_of_a_repaired_network():
    net = exponential_bridge()
    routes = net.analysis_routes()
    assert routes["point_unavailability"].route == "exact"
    assert routes["failure_frequency"].route == "exact"
    assert routes["mission_availability"].route == "numerical"
    assert routes["sf"].route == "refused"
    c = general()
    numerical = Network(
        {n: (u, v, c[n]) for n, (u, v) in BRIDGE.items()}, "s", "t"
    ).analysis_routes()
    assert numerical["point_availability"].route == "numerical"
    assert numerical["mean_availability"].route == "exact"
    assert set(numerical["point_availability"].nodes) == {"b", "c", "d"}
