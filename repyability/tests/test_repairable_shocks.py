"""Shocks at a rate of their own in a repairable diagram (`Shocks`): the
simulation against the rate-split groups it equals for exponential
members (the same strikes from the same streams), against their chains'
exact values, and, for members that wear out, against a simulation of
the member's own renewal process written here; and what refuses."""

from math import comb

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    AlphaFactor,
    BetaFactor,
    CCFGroup,
    NodeState,
    PerfectReliability,
    RepairableRBD,
    Shocks,
)
from repyability.rbd import _ccf_groups

W = surv.Weibull.from_params
E = surv.Exponential.from_params
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
TEST = {"interval": 100.0, "coverage": 0.8, "full_test": 300.0}


def pair(model, rate, repair=lambda: E([0.1]), inspection=None):
    spec = {"reliability": E([rate]), "repairability": repair()}
    if inspection:
        spec = {**spec, "repairability": "instant", "inspection": inspection}
    return RepairableRBD(
        PARALLEL,
        {"a": dict(spec), "b": dict(spec)},
        ccf_groups=[CCFGroup(["a", "b"], model)],
    )


def same_run(one, other, t=2e4, n=2000):
    a = one.availability(t, mc_samples=n, seed=1, control_variate=False)
    b = other.availability(t, mc_samples=n, seed=1, control_variate=False)
    return (
        a.mean_availability == b.mean_availability
        and a.system_failures == b.system_failures
    )


def test_lethal_shocks_on_exponential_members_are_the_beta_factor_run():
    """Own rate 2^-10 and lethal shocks at 2^-10 are the beta-factor pair
    of rate 2^-9 with beta 1/2 (rates the split gives back exactly): the
    same causes at the same rates, drawn from the same streams, so the
    same run to the last bit."""
    own = shock = 2.0**-10
    total, beta = own + shock, 0.5
    assert (1 - beta) * total == own and beta * total == shock
    assert same_run(pair(Shocks(shock), own), pair(BetaFactor(beta), total))


def test_tested_members_are_the_beta_factor_run_too():
    """Hidden failures: a strike's damage waits for each member's test,
    one coin for the failures a partial test may miss."""
    own = shock = 2.0**-10
    assert same_run(
        pair(Shocks(shock), own, inspection=TEST),
        pair(BetaFactor(0.5), own + shock, inspection=TEST),
    )


def test_partial_shocks_against_the_alpha_factor_chain():
    """Three exponential members, two needed; shocks fail each with
    probability 0.4. The alpha-factor group with the same causes has an
    exact mean availability (its chain), which the simulation's interval
    of three standard errors covers."""
    own, rate, p, m = 2e-3, 6e-4, 0.4, 3
    total = own + rate * p
    share = {j: rate * p**j * (1 - p) ** (m - j) for j in range(1, m + 1)}
    share[1] += own
    alphas = [comb(m - 1, j - 1) * share[j] / total for j in range(1, m + 1)]
    edges = [("s", n) for n in "abc"] + [(n, "j") for n in "abc"]
    edges += [("j", "t")]

    def build(model, life):
        return RepairableRBD(
            edges,
            {
                **{
                    n: {"reliability": E([life]), "repairability": E([0.05])}
                    for n in "abc"
                },
                "j": PerfectReliability,
            },
            k={"j": 2},
            ccf_groups=[CCFGroup(list("abc"), model)],
        )

    t = 5000.0
    group = build(AlphaFactor(alphas), total)
    route = group.analysis_routes()["mission_availability"].route
    assert route in ("exact", "numerical")
    exact = group.mission_availability(t)
    run = build(Shocks(rate, p), own).availability(t, mc_samples=4000, seed=0)
    assert run.control_variate is None  # a plain run: nothing exact
    error = run.mean_availability_interval().standard_error
    assert abs(run.mean_availability - exact) <= 3 * error


def member_availability(life, repair, shock, t, n, seed):
    """The mean availability over ``[0, t]`` of one unit, new at 0, that
    fails at the earlier of its own life and the next shock and is then
    repaired, by plain simulation: the mean and its standard error."""
    rng = np.random.default_rng(seed)
    clock, up = np.zeros(n), np.zeros(n)
    while np.any(clock < t):
        going = clock < t
        own = life.qf(rng.random(n))
        works = np.minimum(own, rng.exponential(1 / shock, n))
        up += np.where(going, np.minimum(works, t - clock), 0.0)
        clock += works + repair.qf(rng.random(n))
    share = up / t
    return share.mean(), share.std(ddof=1) / np.sqrt(n)


def test_a_member_that_wears_out_against_its_own_renewal_process():
    """Lethal shocks on Weibull members with Weibull repairs: a member is
    up from each repair to the earlier of its life and the next shock.
    The system here is up exactly while member ``a`` is."""
    life, repair, shock, t = W([300.0, 2.0]), W([20.0, 1.5]), 4e-3, 5000.0
    spec = {"reliability": life, "repairability": repair}
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t"), ("a", "j"), ("j", "t")],
        {"a": dict(spec), "b": dict(spec), "j": PerfectReliability},
        ccf_groups=[CCFGroup(["a", "b"], Shocks(shock))],
    )
    run = rbd.availability(t, mc_samples=4000, seed=0)
    mean, error = member_availability(life, repair, shock, t, 40000, seed=0)
    assert abs(run.mean_availability - mean) <= 3 * np.hypot(
        run.mean_availability_interval().standard_error, error
    )
    # Without the shocks the member is up more: the check tells them.
    plain, _ = member_availability(life, repair, 1e-12, t, 40000, seed=0)
    assert plain - mean > 20 * error


def test_a_strike_fails_its_members_in_the_group_s_order():
    """Each cause names its members in the group's order, which is the
    order a strike fails them in (`_event_loop._strike`)."""
    spec = {"reliability": W([300.0, 2.0]), "repairability": E([0.1])}
    rbd = RepairableRBD(
        [("s", n) for n in "abc"] + [(n, "t") for n in "abc"],
        {n: dict(spec) for n in "abc"},
        ccf_groups=[CCFGroup(["c", "a", "b"], Shocks(1e-3, 0.5))],
    )
    struck = [cause.struck for cause in _ccf_groups._shared_causes(rbd)]
    assert struck == [
        ("c",),
        ("a",),
        ("b",),
        ("c", "a"),
        ("c", "b"),
        ("a", "b"),
        ("c", "a", "b"),
    ]
    assert all(_ccf_groups._own_rate(rbd, n) is None for n in "abc")


def shocked(extra=None, **options):
    spec = {"reliability": W([300.0, 2.0]), "repairability": E([0.1])}
    return RepairableRBD(
        PARALLEL,
        {"a": {**spec, **(extra or {})}, "b": dict(spec)},
        ccf_groups=[CCFGroup(["a", "b"], Shocks(1e-3, 0.5))],
        **options,
    )


def test_the_exact_methods_refuse_naming_the_simulation():
    rbd = shocked()
    report = rbd.analysis_routes()
    assert report["availability"].route == "simulated"
    for name, call in (
        ("point_availability", lambda: rbd.point_availability(100.0)),
        ("mean_availability", lambda: rbd.mean_availability()),
        ("birnbaum_importance", lambda: rbd.birnbaum_importance()),
    ):
        assert report[name].route == "refused"
        assert "availability() or cost()" in report[name].reason
        with pytest.raises(NotImplementedError, match="rate of their own"):
            call()


def test_maintenance_on_a_member_is_refused_by_name():
    rbd = shocked({"preventive": {"interval": 150.0, "duration": "instant"}})
    with pytest.raises(
        NotImplementedError, match="scheduled maintenance.*left out"
    ):
        rbd.availability(1000.0, mc_samples=10, seed=0)


def test_a_tested_member_started_from_a_state_is_refused():
    spec = {
        "reliability": W([300.0, 2.0]),
        "repairability": "instant",
        "inspection": {"interval": 100.0},
    }
    rbd = RepairableRBD(
        PARALLEL,
        {"a": dict(spec), "b": dict(spec)},
        ccf_groups=[CCFGroup(["a", "b"], Shocks(1e-3))],
    )
    rbd.availability(1000.0, mc_samples=10, seed=0)
    with pytest.raises(NotImplementedError, match="Start it as new"):
        rbd.availability(
            1000.0, mc_samples=10, seed=0, state={"a": NodeState(age=50.0)}
        )


def test_a_revealed_member_starts_from_its_age():
    rbd = shocked()
    old = rbd.availability(
        200.0, mc_samples=2000, seed=0, state={"a": NodeState(age=250.0)}
    )
    new = rbd.availability(200.0, mc_samples=2000, seed=0)
    assert old.system_failures > new.system_failures


def test_saved_and_read_back_it_runs_the_same():
    import json

    rbd = shocked()
    again = RepairableRBD.from_dict(json.loads(json.dumps(rbd.to_dict())))
    assert same_run(rbd, again, t=2000.0, n=200)


def test_a_group_of_too_many_members_is_refused():
    names = [f"n{i}" for i in range(11)]
    spec = {"reliability": E([1e-3]), "repairability": E([0.1])}
    with pytest.raises(ValueError, match="up to 10"):
        RepairableRBD(
            [("s", n) for n in names] + [(n, "t") for n in names],
            {n: dict(spec) for n in names},
            ccf_groups=[CCFGroup(names, Shocks(1e-3, 0.5))],
        )


@pytest.mark.parametrize("life", [E([1e-3]), W([300.0, 2.0])])
@pytest.mark.parametrize("crews", [None, 1])
def test_a_member_s_own_values_refuse_and_say_what_to_run(life, crews):
    """A member's own long-run availability and its spares come from its
    own life, which the shocks add to, so they refuse (with crews, as
    the crews' chain does), as the routes say; the simulated count of
    spares runs."""
    spec = {"reliability": life, "repairability": E([0.1])}
    rbd = RepairableRBD(
        PARALLEL,
        {"a": dict(spec), "b": dict(spec)},
        ccf_groups=[CCFGroup(["a", "b"], Shocks(1e-3, 0.5))],
        repair_crews=crews,
    )
    report = rbd.analysis_routes()
    for name, call in (
        ("node_availability", rbd.node_availability),
        ("spares_demand", lambda: rbd.spares_demand(1000.0)),
        ("spares_stock", lambda: rbd.spares_stock(1000.0, fill_rate=0.9)),
    ):
        assert report[name].route == "refused"
        with pytest.raises(NotImplementedError):
            call()
    if crews is None:
        assert "shocks fail it too" in report["node_availability"].reason
        assert "simulat" in report["spares_demand"].reason
    demand = rbd.spares_demand(
        1000.0, method="simulate", mc_samples=50, seed=0
    )
    assert set(demand) == {"a", "b"}
