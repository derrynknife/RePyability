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
    assert demand.mean == pytest.approx(mean, rel=1e-5)
    assert demand.std == pytest.approx(np.sqrt(mean), rel=1e-4)
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
    model = _spares._replacements(single(WORN), "c", True)
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
        assert abs(e.mean - s.mean) < 4 * e.std / np.sqrt(10_000)
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
    # Block-replaced and tested components are counted too (#147, #159:
    # see test_spares_block_and_tested.py), but not tests that can last as
    # long as their interval.
    group = {
        "reliability": E([0.01]),
        "repairability": E([0.5]),
        "standby": {"units": 2},
    }
    tested = {
        "reliability": E([0.01]),
        "repairability": "instant",
        "inspection": {"interval": 50.0, "duration": E([0.02])},
    }
    for spec, why in (
        (group, "standby group"),
        (tested, "within its test interval"),
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
        assert simulated.mean > 0.0
    # A standby group's units fail at 0.01 an hour while operating: about
    # five spares in 500 hours (a little less while the group is down).
    simulated = single(group).spares_demand(
        500.0, method="simulate", mc_samples=4_000, seed=2
    )["c"]
    assert simulated.mean == pytest.approx(5.0, abs=0.15)


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
        {
            "reliability": W([100.0, 1.5], lfp_p=0.5),
            "repairability": "instant",
        }
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


# -- one shelf for interchangeable parts (#183) -----------------------------


SEAL = {"reliability": W([4000.0, 1.8]), "repairability": L([2.0, 0.5])}
TRAINS = [("s", f"seal{i}") for i in (1, 2, 3)] + [
    (f"seal{i}", "t") for i in (1, 2, 3)
]
PART = {"seal": ["seal1", "seal2", "seal3"]}


def station(**specs):
    return RepairableRBD(
        TRAINS,
        {f"seal{i}": dict(specs.get(f"seal{i}", SEAL)) for i in (1, 2, 3)},
        k={"t": 2},
    )


def padded(*arrays):
    size = max(len(a) for a in arrays)
    return [np.pad(a, (0, size - len(a))) for a in arrays]


def test_identical_positions_on_one_shelf_are_a_bigger_fleet():
    # Three identical seals for 13 stations draw as one seal for 39.
    plant = station()
    shelf = plant.spares_stock(1008.0, fill_rate=0.95, fleet=13, parts=PART)
    one = plant.spares_stock(1008.0, fill_rate=0.95, nodes=["seal1"], fleet=39)
    pooled, alone = shelf["seal"], one["seal1"]
    assert list(shelf) == ["seal"] and pooled.members == tuple(PART["seal"])
    assert (pooled.stock, alone.stock) == (17, 17)
    for a, b in (
        (pooled.on_order, alone.on_order),
        (pooled.on_order_at_demand, alone.on_order_at_demand),
    ):
        np.testing.assert_allclose(*padded(a, b), atol=1e-14)
    separate = plant.spares_stock(1008.0, fill_rate=0.95, fleet=13)
    assert sum(s.stock for s in separate.values()) == 21


def test_poisson_positions_pool_to_a_poisson_shelf():
    # Constant rates, replaced in no time: the shelf's demand is Poisson at
    # the summed rate, and a demand finds it as a random time does, however
    # the rates differ (so each position's share of the demands is right).
    rates = (0.01, 0.003, 0.02)
    plant = station(
        **{
            f"seal{i}": {"reliability": E([rate]), "repairability": "instant"}
            for i, rate in zip((1, 2, 3), rates)
        }
    )
    stock = plant.spares_stock(300.0, fill_rate=0.95, parts=PART)["seal"]
    mean = 300.0 * sum(rates)
    expected = poisson.pmf(np.arange(len(stock.on_order)), mean)
    np.testing.assert_allclose(stock.on_order, expected, atol=1e-6)
    np.testing.assert_allclose(
        *padded(stock.on_order_at_demand, stock.on_order), atol=1e-6
    )
    assert stock.stock == int(poisson.ppf(0.95, mean)) + 1 or (
        poisson.cdf(stock.stock - 1, mean) >= 0.95 - 1e-6
    )
    demand = plant.spares_demand(1000.0, parts=PART)["seal"]
    assert demand.mean == pytest.approx(1000.0 * sum(rates), rel=1e-5)


def test_a_shelf_of_different_positions_by_simulation():
    plant = station(seal1=dict(SEAL, preventive={"interval": 2000.0}))
    exact = plant.spares_demand(8760.0, parts=PART)["seal"]
    simulated = plant.spares_demand(
        8760.0, parts=PART, method="simulate", mc_samples=4000, seed=3
    )["seal"]
    assert simulated.members == exact.members == tuple(PART["seal"])
    assert abs(exact.mean - simulated.mean) < 4 * exact.std / np.sqrt(4000)
    # The pooled count is the members' counted together.
    each = plant.spares_demand(8760.0)
    total = np.ones(1)
    for node in PART["seal"]:
        total = np.convolve(total, each[node].probabilities)
    np.testing.assert_allclose(*padded(exact.probabilities, total), atol=1e-12)
    # The stock of a mixed shelf, a demand weighed by each position's rate.
    stock = plant.spares_stock(1008.0, fill_rate=0.95, fleet=13, parts=PART)
    assert stock["seal"].fill_rate >= 0.95
    assert stock["seal"].fill_rate_for(stock["seal"].stock - 1) < 0.95


def test_parts_and_nodes_together():
    plant = station()
    pair = {"seal": ["seal1", "seal2"]}
    both = plant.spares_demand(8760.0, nodes=["seal3"], parts=pair)
    assert list(both) == ["seal3", "seal"]
    assert both["seal3"].members is None
    assert both["seal"].members == ("seal1", "seal2")
    # A component's spares come from one shelf, its own or its part's
    # (#233).
    with pytest.raises(ValueError, match="'seal1' is in nodes and in part"):
        plant.spares_demand(8760.0, nodes=["seal1"], parts=PART)


@pytest.mark.parametrize(
    "parts, error, message",
    [
        ({"seal1": ["seal2", "seal3"]}, ValueError, "name of a component"),
        (
            {"a": ["seal1", "seal2"], "b": ["seal2"]},
            ValueError,
            "parts 'a' and 'b'",
        ),
        ({"seal": "seal1"}, ValueError, "as a list"),
        ({"seal": []}, ValueError, "no components"),
        ({"seal": ["pump"]}, ValueError, "not a component"),
        ([("seal", ["seal1"])], ValueError, "dict of part name"),
    ],
)
def test_parts_are_checked(parts, error, message):
    with pytest.raises(error, match=message):
        station().spares_demand(100.0, parts=parts)


def test_a_common_cause_group_is_not_pooled():
    from repyability import BetaFactor, CCFGroup

    exponential = {"reliability": E([0.01]), "repairability": "instant"}
    plant = RepairableRBD(
        TRAINS,
        {f"seal{i}": dict(exponential) for i in (1, 2, 3)},
        k={"t": 2},
        ccf_groups=[CCFGroup(["seal1", "seal2"], BetaFactor(0.1, "rate"))],
    )
    with pytest.raises(NotImplementedError, match="common-cause group"):
        plant.spares_stock(100.0, fill_rate=0.9, parts=PART)
