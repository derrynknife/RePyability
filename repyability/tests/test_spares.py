"""Spares demand and stock (#95): each component's replacements counted as a
renewal process, over a horizon from new and in a lead time in the long
run, checked against Poisson closed forms, a Monte Carlo of the process and
the RBD's own simulation."""

import numpy as np
import pytest
import surpyval as surv
from scipy.stats import poisson

from repyability import RepairableRBD
from repyability.rbd import _spares
from repyability.rbd import routes as r

E, W, L = (
    surv.Exponential.from_params,
    surv.Weibull.from_params,
    surv.LogNormal.from_params,
)
X = surv.ExactEventTime.from_params


def single(spec, **options):
    return RepairableRBD([("s", "c"), ("c", "t")], {"c": spec}, **options)


PUMP = {"reliability": E([0.01]), "repairability": "instant"}


@pytest.mark.parametrize(
    "horizon, fleet", [(300.0, 1), (1000.0, 1), (1000.0, 4)]
)
def test_constant_rates_and_instant_repair_give_poisson_demand(horizon, fleet):
    demand = single(PUMP).spares_demand(horizon, fleet=fleet)["c"]
    mean = 0.01 * horizon * fleet
    counts = np.arange(len(demand.probabilities))
    np.testing.assert_allclose(
        demand.probabilities, poisson.pmf(counts, mean), atol=2e-6
    )
    assert demand.mean() == pytest.approx(mean, rel=1e-5)
    assert demand.std() == pytest.approx(np.sqrt(mean), rel=1e-4)
    for probability in (0.5, 0.9, 0.99):
        assert demand.stock(probability) == poisson.ppf(probability, mean)
    assert demand.covered(demand.stock(0.9)) >= 0.9
    assert demand.method == "exact" and demand.fleet == fleet


@pytest.mark.parametrize("fleet", [1, 3])
def test_poisson_demand_gives_the_textbook_stock(fleet):
    # Poisson demand: the spares on order are Poisson(rate * lead time *
    # fleet), and a demand finds them so too.
    stock = single(PUMP).spares_stock(
        300.0, fill_rate=0.95, stockout_probability=0.1, fleet=fleet
    )["c"]
    mean = 3.0 * fleet
    for s in range(12):
        assert stock.fill_rate_for(s) == pytest.approx(
            poisson.cdf(s - 1, mean), abs=2e-6
        )
        assert stock.stockout_probability_for(s) == pytest.approx(
            poisson.sf(s - 1, mean), abs=2e-6
        )
    fewest = next(s for s in range(50) if poisson.cdf(s - 1, mean) >= 0.95)
    assert stock.stock == fewest
    assert stock.fill_rate >= 0.95 and stock.stockout_probability <= 0.1


def renewal_demands(rng, life, repair, maintenance, age, until):
    """Replacement times of one component simulated directly."""
    t, times = 0.0, []
    while True:
        up = life.random(1).item()
        if up < age:
            t += up
            if t > until:
                return np.array(times)
            times.append(t)
            t += repair.random(1).item()
        else:
            t += age
            if t > until:
                return np.array(times)
            times.append(t)
            t += maintenance.random(1).item()


WORN = {
    "reliability": W([100.0, 2.5]),
    "repairability": L([1.0, 0.6]),
    "preventive": {"interval": 80.0, "duration": E([1 / 1.5])},
}


def test_age_replacement_counts_match_a_direct_simulation():
    np.random.seed(3)
    model = single(WORN)._replacements("c", True)
    args = (W([100.0, 2.5]), L([1.0, 0.6]), E([1 / 1.5]), 80.0)
    rng = np.random.default_rng(1)
    # From new over 500 hours.
    exact = _spares.count(model, 500.0, "new")
    counts = [len(renewal_demands(rng, *args, 500.0)) for _ in range(20_000)]
    simulated = np.bincount(counts, minlength=len(exact))[: len(exact)]
    np.testing.assert_allclose(simulated / len(counts), exact, atol=0.012)
    # In 150 hours in the long run, from a random time and before a
    # replacement.
    path = renewal_demands(rng, *args, 2_000_000.0)
    starts = rng.uniform(1_000.0, path[-1] - 200.0, 100_000)
    inside = np.searchsorted(path, starts + 150.0, side="right")
    inside -= np.searchsorted(path, starts, side="right")
    exact = _spares.count(model, 150.0, "random")
    found = np.bincount(inside, minlength=len(exact))[: len(exact)]
    np.testing.assert_allclose(found / len(starts), exact, atol=0.01)
    at = np.arange(1_000, len(path))
    before = at - np.searchsorted(path, path[at] - 150.0, side="left")
    exact = _spares.count(model, 150.0, "arrival")
    found = np.bincount(before, minlength=len(exact))[: len(exact)]
    np.testing.assert_allclose(found / len(at), exact, atol=0.01)


def test_the_exact_demand_agrees_with_the_rbd_s_simulation():
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": WORN,
            "b": {"reliability": E([0.02]), "repairability": E([0.5])},
        },
    )
    exact = rbd.spares_demand(500.0)
    simulated = rbd.spares_demand(
        500.0, method="simulate", mc_samples=10_000, seed=4
    )
    for node in "ab":
        e, s = exact[node], simulated[node]
        assert s.method == "simulate"
        assert abs(e.mean() - s.mean()) < 4 * e.std() / np.sqrt(10_000)
        size = min(len(e.probabilities), len(s.probabilities))
        np.testing.assert_allclose(
            s.probabilities[:size], e.probabilities[:size], atol=0.015
        )
    # Seeded, the simulation is reproducible.
    again = rbd.spares_demand(
        500.0, method="simulate", mc_samples=10_000, seed=4
    )
    assert np.array_equal(
        again["a"].probabilities, simulated["a"].probabilities
    )


def test_a_replacement_at_the_horizon_falls_after_it():
    # A unit that would last 1,000 hours, replaced every 100 in no time:
    # nine replacements before 1,000 hours, the tenth at exactly 1,000,
    # which falls after the window, as in the simulation (so that windows
    # one after another add up).
    spec = {
        "reliability": X(1_000.0),
        "repairability": "instant",
        "preventive": {"interval": 100.0},
    }
    rbd = single(spec)
    assert rbd.spares_demand(1_000.0)["c"].probabilities[9] == (
        pytest.approx(1.0)
    )
    assert rbd.spares_demand(1_000.5)["c"].probabilities[10] == (
        pytest.approx(1.0)
    )
    simulated = rbd.spares_demand(1_000.0, method="simulate", mc_samples=20)
    assert simulated["c"].probabilities[9] == 1.0
    assert rbd.expected_events(1_000.0).node_preventive["c"] == (
        pytest.approx(9.0)
    )


def test_components_the_renewal_count_does_not_cover_are_simulated():
    group = {
        "reliability": E([0.01]),
        "repairability": E([0.5]),
        "standby": {"units": 2},
    }
    tested = {
        "reliability": E([0.01]),
        "repairability": "instant",
        "inspection": {"interval": 50.0},
    }
    block = {
        "reliability": W([100.0, 2.5]),
        "repairability": E([1.0]),
        "preventive": {"interval": 80.0, "policy": "block"},
    }
    for spec, why in (
        (group, "standby group"),
        (tested, "inspection"),
        (block, "block"),
    ):
        rbd = single(spec)
        with pytest.raises(NotImplementedError, match=why) as error:
            rbd.spares_demand(500.0)
        assert "method='simulate'" in str(error.value)
        report = rbd.analysis_routes()
        assert report["spares_demand"].reason == str(error.value)
        assert report["spares_stock"].route == r.REFUSED
        simulated = rbd.spares_demand(
            500.0, method="simulate", mc_samples=2_000, seed=1
        )["c"]
        assert simulated.mean() > 0.0
    # A standby group's units fail at 0.01 an hour while operating: about
    # five spares in 500 hours (a little less while the group is down).
    simulated = single(group).spares_demand(
        500.0, method="simulate", mc_samples=4_000, seed=2
    )["c"]
    assert simulated.mean() == pytest.approx(5.0, abs=0.15)


def test_what_is_refused_and_checked():
    crewed = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": dict(PUMP, repairability=E([0.5])),
            "b": dict(PUMP, repairability=E([0.5])),
        },
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="repair crew"):
        crewed.spares_demand(100.0)
    assert crewed.analysis_routes()["spares_demand"].route == r.REFUSED
    crewed.spares_demand(100.0, method="simulate", mc_samples=500, seed=1)
    nested = RepairableRBD([("s", "x"), ("x", "t")], {"x": single(PUMP)})
    assert nested.spares_demand(100.0) == {}
    with pytest.raises(ValueError, match="nested RBD"):
        nested.spares_demand(100.0, nodes=["x"])
    never = single(
        {"reliability": W([100.0, 1.5], p=0.5), "repairability": "instant"}
    )
    never.spares_demand(100.0)  # from new, fine
    with pytest.raises(NotImplementedError, match="may never end"):
        never.spares_stock(50.0, fill_rate=0.9)
    rbd = single(PUMP)
    for bad in (
        lambda: rbd.spares_demand(-1.0),
        lambda: rbd.spares_demand(100.0, fleet=0),
        lambda: rbd.spares_demand(100.0, method="guess"),
        lambda: rbd.spares_demand(100.0, nodes=["nope"]),
        lambda: rbd.spares_stock(50.0),
        lambda: rbd.spares_stock(50.0, fill_rate=1.0),
        lambda: rbd.spares_stock(50.0, stockout_probability=0.0),
        lambda: rbd.spares_demand(100.0)["c"].stock(1.0),
    ):
        with pytest.raises(ValueError):
            bad()
    assert rbd.analysis_routes()["spares_stock"].route == r.NUMERICAL
