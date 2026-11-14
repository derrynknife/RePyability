"""Where a safety function's PFDavg comes from (``pfd_breakdown``).

The shares add up to the PFDavg; with instant tests and repairs and
constant failure rates they are checked against the exact integrals over a
test interval of 1oo1, 1oo2 and 2oo3 voting, with and without a
beta-factor common cause (and IEC 61508-6's simplified equations, to their
first order), and against the simulation.
"""

import json

import numpy as np
import pytest
from scipy.integrate import quad
from surpyval import ExactEventTime, Exponential

from repyability import (
    MGL,
    BetaFactor,
    CCFGroup,
    PerfectReliability,
    RepairableRBD,
)
from repyability.rbd import _repairable_importance
from repyability.rbd import routes as r

E, X = Exponential.from_params, ExactEventTime.from_params
RATE, TAU = 2e-6, 8760.0


def unit_spec(rate=RATE, interval=TAU, repair="instant", **inspection):
    return {
        "reliability": E([rate]),
        "repairability": repair,
        "inspection": {"interval": interval, **inspection},
    }


def voting(n, k, beta=None, unit=None, model=None):
    """``k`` out of ``n`` tested units (k working needed), in one
    common-cause group of ``beta`` (or ``model``) if given."""
    names = [f"u{i}" for i in range(n)]
    edges = [("s", u) for u in names] + [(u, "k") for u in names]
    edges.append(("k", "t"))
    components = {u: dict(unit or unit_spec()) for u in names}
    components["k"] = PerfectReliability
    if model is None and beta is not None:
        model = BetaFactor(beta)
    groups = [CCFGroup(names, model)] if model is not None else []
    return (
        RepairableRBD(edges, components, k={"k": k}, ccf_groups=groups),
        names,
    )


def averaged(f, rate=RATE, interval=TAU):
    """The average over a test interval of ``f(q)``, ``q`` a unit's
    probability of being down a time ``t`` after its test, ``1 - exp(-rate
    t)``."""
    return (
        quad(lambda t: f(-np.expm1(-rate * t), t), 0.0, interval)[0] / interval
    )


def assert_adds_up(result):
    parts = sum(result.components.values()) + sum(result.groups.values())
    assert parts == pytest.approx(result.pfd_avg, rel=1e-12)
    assert result.independent + result.common_cause == pytest.approx(
        result.pfd_avg, rel=1e-12
    )
    assert sum(result.fractions.values()) == pytest.approx(1.0, rel=1e-12)


# Down probability of k-out-of-n (k needed) from each unit's own q.
FAILS = {
    (1, 1): lambda q: q,
    (2, 1): lambda q: q**2,
    (3, 2): lambda q: 3 * q**2 - 2 * q**3,
}


@pytest.mark.parametrize("n, k", list(FAILS))
def test_voting_without_common_cause_is_the_exact_integral(n, k):
    rbd, names = voting(n, k)
    result = rbd.pfd_breakdown()
    exact = averaged(lambda q, t: FAILS[(n, k)](q))
    assert result.pfd_avg == pytest.approx(exact, rel=1e-9)
    assert_adds_up(result)
    # Identical units take equal shares; no common cause.
    for name in names:
        assert result.components[name] == pytest.approx(exact / n, rel=1e-9)
    assert result.common_cause == 0.0
    assert result.groups == {}


@pytest.mark.parametrize("n, k", [(2, 1), (3, 2)])
@pytest.mark.parametrize("beta", [0.02, 0.1])
def test_voting_with_a_beta_factor_is_the_exact_integral(n, k, beta):
    rbd, names = voting(n, k, beta)
    result = rbd.pfd_breakdown()
    own, shared = (1 - beta) * RATE, beta * RATE

    def system(q_own, t):
        q_shared = -np.expm1(-shared * t)
        return q_shared + (1 - q_shared) * FAILS[(n, k)](q_own)

    def independent(q_own, t):
        return FAILS[(n, k)](q_own)

    exact = averaged(system, rate=own)
    exact_independent = averaged(independent, rate=own)
    assert result.pfd_avg == pytest.approx(exact, rel=1e-9)
    assert result.independent == pytest.approx(exact_independent, rel=1e-8)
    assert result.common_cause == pytest.approx(
        exact - exact_independent, rel=1e-8
    )
    assert result.groups[tuple(names)] == pytest.approx(
        result.common_cause, rel=1e-12
    )
    assert_adds_up(result)
    # IEC 61508-6's simplified equations (instant tests and repairs),
    # to their first order in lambda tau: the shared part beta lambda
    # tau / 2, the independent ((1 - beta) lambda tau)^2 / 3 for 1oo2 and
    # ((1 - beta) lambda tau)^2 for 2oo3.
    lt = own * TAU
    iec_independent = {(2, 1): lt**2 / 3, (3, 2): lt**2}[(n, k)]
    assert result.independent == pytest.approx(iec_independent, rel=0.03)
    assert result.common_cause == pytest.approx(shared * TAU / 2, rel=0.03)


def test_a_redundant_pair_shares_its_coincident_downtime_equally():
    # Each term q_a q_b goes half to each, whatever their rates: both
    # must be down. The series unit takes its own downtime, and the
    # overlap with the pair is shared among the three.
    edges = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")]
    rbd = RepairableRBD(
        edges,
        {
            "a": unit_spec(1e-5),
            "b": unit_spec(1e-6),
            "c": unit_spec(3e-7, offset=2000.0),
        },
    )
    result = rbd.pfd_breakdown()
    assert_adds_up(result)
    pair = averaged(
        lambda q, t: q * -np.expm1(-1e-6 * t), rate=1e-5
    )  # both down
    single = averaged(lambda q, t: q, rate=3e-7)
    overlap = single + pair - result.pfd_avg  # c down with the pair
    assert result.components["a"] == pytest.approx(
        pair / 2 - overlap / 3, rel=1e-8
    )
    assert result.components["b"] == pytest.approx(
        result.components["a"], rel=1e-12
    )
    assert result.components["c"] == pytest.approx(
        single - overlap / 3, rel=1e-8
    )


def test_two_groups_each_take_their_shapley_share():
    sensors = [f"s{i}" for i in range(3)]
    valves = ["v1", "v2"]
    edges = [("s", x) for x in sensors] + [(x, "vote") for x in sensors]
    edges += [("vote", "logic")]
    edges += [("logic", v) for v in valves] + [(v, "t") for v in valves]
    components = {x: unit_spec(1e-6) for x in sensors}
    components.update({v: unit_spec(3e-6, offset=4380.0) for v in valves})
    components["logic"] = unit_spec(1e-7)
    components["vote"] = PerfectReliability
    groups = [
        CCFGroup(sensors, BetaFactor(0.05)),
        CCFGroup(valves, BetaFactor(0.1)),
    ]
    rbd = RepairableRBD(edges, components, k={"vote": 2}, ccf_groups=groups)
    result = rbd.pfd_breakdown()
    assert_adds_up(result)
    assert set(result.groups) == {tuple(sensors), tuple(valves)}
    # Each group's share is about what it adds alone.
    alone = {}
    for j, group in enumerate(groups):
        only = _repairable_importance._with_shared_causes(rbd, [j])
        alone[tuple(group.members)] = (
            only.mean_unavailability() - result.independent
        )
    for key, share in result.groups.items():
        assert share == pytest.approx(alone[key], rel=0.01)
    # The valves' shared cause, found at staggered tests, is the
    # largest part.
    assert max(result.fractions, key=result.fractions.get) == tuple(valves)


def test_partial_common_causes():
    rbd, names = voting(3, 2, model=MGL(0.1, 0.3))
    result = rbd.pfd_breakdown()
    assert_adds_up(result)
    assert result.common_cause > 0.0
    without = _repairable_importance._with_shared_causes(rbd, [])
    assert result.independent == pytest.approx(
        without.mean_unavailability(), rel=1e-9
    )


def test_tests_and_repairs_that_take_time():
    unit = unit_spec(3e-4, 500.0, repair=X([8.0]), duration=X([2.0]))
    plain, names = voting(2, 1, unit=unit)
    grouped, _ = voting(2, 1, beta=0.0, unit=unit)
    shared, _ = voting(2, 1, beta=0.2, unit=unit)
    for rbd in (plain, grouped, shared):
        assert_adds_up(rbd.pfd_breakdown())
    # No shared cause: no common-cause share (to the chain's agreement
    # with the units' own model).
    zero = grouped.pfd_breakdown()
    assert abs(zero.common_cause) < 3e-5 * zero.pfd_avg
    assert shared.pfd_breakdown().common_cause > 0.0


def test_held_nodes():
    rbd, names = voting(2, 1)
    result = rbd.pfd_breakdown(broken_nodes=["u0"])
    assert result.pfd_avg == pytest.approx(averaged(lambda q, t: q))
    assert_adds_up(result)
    result = rbd.pfd_breakdown(working_nodes=["u0"])
    assert result.pfd_avg == 0.0
    assert result.components == {"u0": 0.0, "u1": 0.0}


@pytest.mark.parametrize("beta", [None, 0.2])
def test_the_simulation_agrees(beta):
    # Tested together, instantly, from new: each interval repeats the
    # long run, so the mission's unavailability is the PFDavg.
    rate, interval, T, n = 2e-3, 100.0, 2000.0, 4000
    unit = unit_spec(rate, interval)
    rbd, _ = voting(2, 1, beta=beta, unit=unit)
    result = rbd.pfd_breakdown()

    def simulated(diagram):
        run = diagram.availability(
            T, mc_samples=n, seed=11, control_variate=False, conditional=False
        )
        interval = run.mean_availability_interval()
        return 1.0 - interval.estimate, interval.standard_error

    total, se = simulated(rbd)
    assert abs(total - result.pfd_avg) < 4.0 * se
    if beta is not None:
        alone = _repairable_importance._with_shared_causes(rbd, [])
        independent, se_independent = simulated(alone)
        assert abs(independent - result.independent) < 4.0 * se_independent


def test_save_and_load():
    rbd, _ = voting(3, 2, beta=0.1)
    result = rbd.pfd_breakdown()
    loaded = RepairableRBD.from_json(rbd.to_json())
    again = loaded.pfd_breakdown()
    assert again.pfd_avg == result.pfd_avg
    assert again.components == result.components
    assert again.groups == result.groups
    plain = json.loads(json.dumps(result.to_dict()))
    assert plain["pfd_avg"] == result.pfd_avg
    assert plain["groups"] == {
        str(key): value for key, value in result.groups.items()
    }


def test_too_many_groups_are_refused(monkeypatch):
    monkeypatch.setattr(_repairable_importance, "MAX_PFD_GROUPS", 0)
    rbd, _ = voting(2, 1, beta=0.1)
    with pytest.raises(NotImplementedError, match="at most 0"):
        rbd.pfd_breakdown()
    assert rbd.analysis_routes()["pfd_breakdown"].route == r.REFUSED


def test_the_route_is_the_long_run_values():
    rbd, _ = voting(2, 1, beta=0.1)
    routes = rbd.analysis_routes()
    assert routes["pfd_breakdown"].route == routes["mean_unavailability"].route
    assert routes["pfd_breakdown"].route == r.EXACT
