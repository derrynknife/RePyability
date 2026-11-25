"""Shocks at a rate of their own in a repairable diagram (`Shocks`): the
simulation against the rate-split groups it equals for exponential
members (the same strikes from the same streams); the exact values for
exponential members against those groups', a Markov chain written here
and the simulation; for members that wear out, are maintained, repaired
imperfectly or started from a state, the simulation against the member's
own process simulated here; and what refuses."""

import math
from math import comb

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad
from scipy.linalg import expm

from repyability import (
    AlphaFactor,
    BetaFactor,
    CCFGroup,
    NodeState,
    PerfectReliability,
    RepairableRBD,
    Shocks,
)
from repyability.rbd import _ccf_chain, _ccf_groups

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


def test_a_cause_names_its_members_in_the_group_s_order():
    """Each cause names its members in the group's order (a strike's
    failures are then taken in the diagram's order, as every failure at
    one instant is: `_events._EventQueue`)."""
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


# -- exact values for exponential members ------------------------------------


def close(a, b, rel=1e-12):
    return abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)


@pytest.mark.parametrize("inspection", [None, TEST])
def test_lethal_shocks_are_the_beta_factor_group_exactly(inspection):
    """Own failures at 2^-10 with lethal shocks at 2^-10 have the causes
    of the beta-factor pair of rate 2^-9 (beta 1/2), so every exact value
    is the same: the long run, over time, importance, each member's own
    availability and its spares."""
    own = shock = 2.0**-10
    shocks = pair(Shocks(shock), own, inspection=inspection)
    split = pair(BetaFactor(0.5), own + shock, inspection=inspection)
    for call in (
        lambda rbd: rbd.mean_availability(),
        lambda rbd: rbd.mean_unavailability(),
        lambda rbd: rbd.point_availability(150.0),
        lambda rbd: rbd.mission_availability(2000.0),
        lambda rbd: rbd.birnbaum_importance()["a"],
        lambda rbd: rbd.node_availability()["a"],
        lambda rbd: rbd.spares_demand(1000.0)["a"].mean,
    ):
        assert close(float(call(shocks)), float(call(split)))


def test_lethal_shocks_with_a_repair_crew_are_the_beta_factor_group():
    """With one repair crew, the crews' chain takes the shocks in as it
    does the beta-factor group's shared cause."""
    own = shock = 2.0**-10

    def crewed(model, rate):
        spec = {"reliability": E([rate]), "repairability": E([0.1])}
        return RepairableRBD(
            PARALLEL,
            {"a": dict(spec), "b": dict(spec)},
            ccf_groups=[CCFGroup(["a", "b"], model)],
            repair_crews=1,
        )

    shocks, split = crewed(Shocks(shock), own), crewed(
        BetaFactor(0.5), 2 * own
    )
    assert shocks.analysis_routes()["mean_availability"].route == "exact"
    for call in (
        lambda rbd: rbd.mean_availability(),
        lambda rbd: rbd.node_availability()["a"],
        lambda rbd: rbd.point_availability(150.0),
    ):
        assert close(float(call(shocks)), float(call(split)))


def three_of(model, life):
    """Three exponential members, two needed (repaired at 0.05)."""
    edges = [("s", n) for n in "abc"] + [(n, "j") for n in "abc"]
    return RepairableRBD(
        edges + [("j", "t")],
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


def test_partial_shocks_are_the_alpha_factor_group_exactly():
    """Shocks failing each of three members with probability 0.4 are the
    alpha-factor group with the same causes (its alphas worked out here),
    to rounding; and the simulation agrees."""
    own, rate, p, m = 2e-3, 6e-4, 0.4, 3
    total = own + rate * p
    share = {j: rate * p**j * (1 - p) ** (m - j) for j in range(1, m + 1)}
    share[1] += own
    alphas = [comb(m - 1, j - 1) * share[j] / total for j in range(1, m + 1)]
    shocks = three_of(Shocks(rate, p), own)
    split = three_of(AlphaFactor(alphas), total)
    t = 5000.0
    for call in (
        lambda rbd: rbd.mean_availability(),
        lambda rbd: rbd.mission_availability(t),
        lambda rbd: rbd.birnbaum_importance()["b"],
    ):
        assert close(float(call(shocks)), float(call(split)))
    run = shocks.availability(
        t, mc_samples=4000, seed=0, control_variate=False
    )
    error = run.mean_availability_interval().standard_error
    assert abs(run.mean_availability - shocks.mission_availability(t)) <= (
        4 * error
    )


#: Two exponential members of their own rates and repairs, in parallel.
LIVES, REPAIRS = (3e-3, 1e-3), (0.5, 0.2)
RATE, P, LETHAL = 2e-3, 0.6, 5e-4


def own_rates():
    return RepairableRBD(
        PARALLEL,
        {
            node: {"reliability": E([life]), "repairability": E([repair])}
            for node, life, repair in zip("ab", LIVES, REPAIRS)
        },
        ccf_groups=[CCFGroup(["a", "b"], Shocks(RATE, P, LETHAL))],
    )


def generator() -> np.ndarray:
    """The members' Markov chain, written out: states (a down, b down) as
    0 (both up), 1 (a down), 2 (b down), 3 (both down). Up, a member fails
    at its own rate and at every shock that fails it; a shock that fails
    both, both up, takes both down at once."""
    (la, lb), (ma, mb) = LIVES, REPAIRS
    alone = RATE * P * (1 - P)  # a shock that fails one member only
    both = RATE * P**2 + LETHAL
    struck = alone + both  # every shock that fails a member
    q = np.zeros((4, 4))
    q[0, 1], q[0, 2], q[0, 3] = la + alone, lb + alone, both
    q[1, 0], q[1, 3] = ma, lb + struck
    q[2, 0], q[2, 3] = mb, la + struck
    q[3, 1], q[3, 2] = mb, ma
    np.fill_diagonal(q, -q.sum(axis=1))
    return q


def test_members_of_their_own_rates_against_a_chain_written_here():
    """The long run and the availability over time of the parallel pair
    are those of the chain over which members are down."""
    rbd = own_rates()
    q = generator()
    system = np.vstack([q.T, np.ones(4)])
    pi = np.linalg.lstsq(system, np.r_[np.zeros(4), 1.0], rcond=None)[0]
    assert close(rbd.mean_unavailability(), pi[3], 1e-10)
    for t in (5.0, 50.0, 400.0):
        at = expm(q * t)[0]
        assert close(float(rbd.point_availability(t)), 1 - at[3], 1e-9)
    # Each member alone: up, it fails at its own rate and every shock's.
    struck = RATE * P + LETHAL
    alone = rbd.node_availability()
    for node, life, repair in zip("ab", LIVES, REPAIRS):
        assert close(alone[node], repair / (repair + life + struck))


def test_members_of_their_own_rates_against_the_simulation():
    rbd = own_rates()
    t = 2000.0
    run = rbd.availability(t, mc_samples=4000, seed=2, control_variate=False)
    error = run.mean_availability_interval().standard_error
    assert abs(run.mean_availability - rbd.mission_availability(t)) <= (
        4 * error
    )


def two_tested():
    """Two exponential members of their own rates, tested on intervals of
    their own, struck by shocks."""
    return RepairableRBD(
        PARALLEL,
        {
            node: {
                "reliability": E([life]),
                "repairability": "instant",
                "inspection": {"interval": interval},
            }
            for node, life, interval in (
                ("a", 2e-3, 100.0),
                ("b", 5e-4, 150.0),
            )
        },
        ccf_groups=[CCFGroup(["a", "b"], Shocks(3e-3, 0.6, 1e-3))],
    )


#: The pair's members' states: each last tested and found working.
STARTS = {
    "a": NodeState(age=500.0, since_scheduled=80.0),
    "b": NodeState(age=300.0, since_scheduled=20.0),
}


@pytest.mark.parametrize("state", [None, STARTS])
def test_tested_members_against_the_simulation(state):
    """From new and from the members' states: a tested member may have
    been failed, unseen, by a shock since its last test, which the
    simulation draws and the chain starts from."""
    rbd = two_tested()
    t = 400.0
    exact = rbd.mission_availability(t, state=state)
    run = rbd.availability(
        t, mc_samples=20000, seed=3, control_variate=False, state=state
    )
    error = run.mean_availability_interval().standard_error
    assert abs(run.mean_availability - exact) <= 4 * error


def test_a_tested_member_s_own_availability_is_its_closed_form():
    """A tested member, repaired at once, is up from each test with the
    chance exp(-L s), L its own rate and every shock's that fails it."""
    rbd = two_tested()
    struck = 3e-3 * 0.6 + 1e-3
    for node, life, interval in (("a", 2e-3, 100.0), ("b", 5e-4, 150.0)):
        total = (life + struck) * interval
        assert close(
            rbd.node_availability()[node], -math.expm1(-total) / total
        )


def test_a_member_s_spares_are_those_of_its_failures():
    """A member's replacements are a renewal process of its failures from
    every cause: those of a unit of the exponential life at that rate."""
    rbd = own_rates()
    struck = RATE * P + LETHAL
    for node, life, repair in zip("ab", LIVES, REPAIRS):
        alone = RepairableRBD(
            [("s", node), (node, "t")],
            {
                node: {
                    "reliability": E([life + struck]),
                    "repairability": E([repair]),
                }
            },
        )
        ours = rbd.spares_demand(3000.0)[node]
        theirs = alone.spares_demand(3000.0)[node]
        assert close(ours.mean, theirs.mean)
        assert np.allclose(ours.probabilities, theirs.probabilities, 1e-12)
        ours, theirs = (
            d.spares_stock(3000.0, fill_rate=0.9)[node] for d in (rbd, alone)
        )
        assert ours.stock == theirs.stock
        assert close(ours.fill_rate, theirs.fill_rate)


def test_the_pfd_breakdown_splits_the_shocks_out():
    """The shares add up to the PFDavg, the shocks' share being the group's
    (its members' own causes, the shocks' that fail one alone included, are
    theirs)."""
    rbd = two_tested()
    breakdown = rbd.pfd_breakdown()
    total = float(rbd.mean_unavailability())
    parts = sum(breakdown.components.values()) + sum(breakdown.groups.values())
    assert close(parts, total, 1e-9)
    assert breakdown.groups[("a", "b")] > 0.0


def test_the_chain_takes_each_member_s_rates():
    """The chain's causes: each member's own rate and the shocks that fail
    it alone, then the shocks that fail both."""
    causes = _ccf_chain.causes(Shocks(2.0, 0.5, 1.0), ["a", "b"], (3.0, 5.0))
    assert causes == [((0,), 3.5), ((1,), 5.5), ((0, 1), 1.5)]


# -- what is still refused, and simulated -------------------------------------


def test_members_that_wear_out_are_simulated_and_the_exact_methods_refuse():
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
        with pytest.raises(NotImplementedError, match="any lives"):
            call()


@pytest.mark.parametrize("crews", [None, 1])
def test_a_member_that_wears_out_refuses_its_own_values(crews):
    """A member's own long-run availability and spares, for a life that is
    not exponential, refuse saying what to run (with crews, as the crews'
    chain does); the simulated count of spares runs."""
    rbd = shocked(repair_crews=crews)
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


def test_a_tested_member_that_wears_out_started_from_a_state():
    """A Weibull member, tested, last found working 60 before 0 at age
    240; lethal shocks. Before its next test (at 40) nothing is repaired,
    so the system (up exactly while ``a`` is) is up at ``s`` with the
    chance R(240 + s) / R(180) exp(-L (60 + s)): its mean over [0, 30]
    against the simulation."""
    life, shock = W([300.0, 2.0]), 2e-3
    spec = {
        "reliability": life,
        "repairability": "instant",
        "inspection": {"interval": 100.0},
    }
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t"), ("a", "j"), ("j", "t")],
        {"a": dict(spec), "b": dict(spec), "j": PerfectReliability},
        ccf_groups=[CCFGroup(["a", "b"], Shocks(shock))],
    )
    t = 30.0
    exact = (
        quad(
            lambda s: life.sf(240.0 + s)
            / life.sf(180.0)
            * math.exp(-shock * (60.0 + s)),
            0.0,
            t,
        )[0]
        / t
    )
    state = {"a": NodeState(age=240.0, since_scheduled=60.0)}
    run = rbd.availability(
        t, mc_samples=20000, seed=4, control_variate=False, state=state
    )
    error = run.mean_availability_interval().standard_error
    assert abs(run.mean_availability - exact) <= 4 * error


# -- members maintained, repaired imperfectly or grouped ----------------------

LIFE, REPAIR = W([300.0, 2.0]), W([20.0, 1.5])
SERIES_A = [("s", "a"), ("a", "b"), ("b", "t"), ("a", "j"), ("j", "t")]


def member_a(extra, shock, **options):
    """Members ``a`` and ``b`` struck by lethal shocks; the system is up
    exactly while ``a`` is."""
    spec = {"reliability": LIFE, "repairability": REPAIR, **extra}
    groups = [] if shock is None else [CCFGroup(["a", "b"], Shocks(shock))]
    return RepairableRBD(
        SERIES_A,
        {"a": dict(spec), "b": dict(spec), "j": PerfectReliability},
        ccf_groups=groups,
        **options,
    )


def unit_written_here(t, n, seed, shock, age=math.inf, task=0.0, q=None):
    """The mean availability over ``[0, t]`` of one unit, new at 0, that
    fails at the earlier of its own life (given its virtual age, which a
    repair takes on by ``q`` times the time it ran, Kijima I; renewed with
    ``q`` None) and the next shock, or is replaced at ``age`` (taking
    ``task``): by plain simulation, the mean and its standard error."""
    alpha, beta = (float(v) for v in LIFE.params)
    rng = np.random.default_rng(seed)
    clock, up, virtual = np.zeros(n), np.zeros(n), np.zeros(n)
    while np.any(clock < t):
        going = clock < t
        exposure = -np.log(rng.random(n))
        own = alpha * ((virtual / alpha) ** beta + exposure) ** (1 / beta)
        own -= virtual
        fails = np.minimum(own, rng.exponential(1 / shock, n))
        works = np.minimum(fails, age)
        up += np.where(going, np.minimum(works, t - clock), 0.0)
        failed = fails < age
        if q is not None:
            virtual = virtual + q * works
        clock += works + np.where(failed, REPAIR.qf(rng.random(n)), task)
    share = up / t
    return share.mean(), share.std(ddof=1) / np.sqrt(n)


@pytest.mark.parametrize(
    "extra, reference",
    [
        (
            {
                "preventive": {
                    "interval": 250.0,
                    "duration": surv.ExactEventTime.from_params([5.0]),
                }
            },
            {"age": 250.0, "task": 5.0},
        ),
        ({"repair": {"model": "kijima1", "q": 0.5}}, {"q": 0.5}),
    ],
    ids=["age replacement", "imperfect repair"],
)
def test_a_maintained_member_against_its_own_process(extra, reference):
    """A strike fails the member before its own failure or task, and an
    imperfect repair then takes its virtual age on by the time it ran,
    not by the life drawn for it."""
    t, shock = 5000.0, 4e-3
    rbd = member_a(extra, shock)
    assert rbd.analysis_routes()["availability"].route == "simulated"
    assert rbd.analysis_routes()["mean_availability"].route == "refused"
    run = rbd.availability(t, mc_samples=4000, seed=0, control_variate=False)
    mean, error = unit_written_here(t, 40000, 0, shock, **reference)
    assert abs(run.mean_availability - mean) <= 4 * np.hypot(
        run.mean_availability_interval().standard_error, error
    )


@pytest.mark.parametrize(
    "extra, options",
    [
        ({"preventive": {"interval": 250.0}}, {}),
        ({"repair": {"model": "kijima1", "q": 0.5}}, {}),
        (
            {
                "preventive": {
                    "interval": 250.0,
                    "restoration": {"model": "kijima1", "q": 0.3},
                }
            },
            {},
        ),
        (
            {
                "preventive": {"interval": 250.0, "opportunity": 100.0},
                "group": "train",
                "replace_cost": 10.0,
            },
            {"maintenance_groups": {"train": {"setup_cost": 5.0}}},
        ),
    ],
    ids=["age", "kijima", "restoration", "maintenance group"],
)
def test_shocks_that_never_come_leave_the_member_s_run(extra, options):
    """Shocks too rare to come in the window: the members' maintenance,
    repairs and stops run as without the group, to the last bit."""
    with_shocks = member_a(extra, 1e-300, **options)
    without = member_a(extra, None, **options)
    plain = dict(mc_samples=200, seed=1, control_variate=False)
    a = with_shocks.availability(3000.0, **plain)
    b = without.availability(3000.0, **plain)
    assert a.mean_availability == b.mean_availability
    assert a.system_failures == b.system_failures
    if with_shocks.has_costs:
        assert (
            with_shocks.cost(3000.0, **plain).mean
            == without.cost(3000.0, **plain).mean
        )


def test_a_member_in_a_maintenance_group_is_struck_too():
    """Shocks add failures, and so stops, to members of a maintenance
    group: their cost rises with the shocks."""
    extra = {
        "preventive": {"interval": 250.0, "opportunity": 100.0},
        "group": "train",
        "replace_cost": 10.0,
    }
    options = {"maintenance_groups": {"train": {"setup_cost": 5.0}}}
    plain = dict(mc_samples=400, seed=1, control_variate=False)
    struck = member_a(extra, 4e-3, **options).cost(3000.0, **plain)
    calm = member_a(extra, None, **options).cost(3000.0, **plain)
    assert struck.mean > calm.mean + 4 * np.hypot(
        struck.mean_interval().standard_error,
        calm.mean_interval().standard_error,
    )
