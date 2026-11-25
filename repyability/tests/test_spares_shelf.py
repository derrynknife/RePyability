"""A shelf of spares in the simulation (``repyability.rbd._shelves``):
a stock reordered one for one, each order arriving a lead time later, a
repair that finds the shelf empty waiting for a spare; the stock to hold
to be sure of no stock-out (``spares_stock(method="simulate")``); and
the spares lever of ``least_spend_for_downtime``."""

import dataclasses
import math

import numpy as np
import pytest
import surpyval as surv
from scipy import stats

from repyability import (
    BetaFactor,
    CCFGroup,
    RepairableRBD,
    SimulatedSparesStock,
)
from repyability.rbd import _event_loop, _shelves

E = surv.Exponential.from_params
ONE = [("s", "p"), ("p", "t")]
FAIL = 1e-3


def unit(repair="instant", **spec):
    return RepairableRBD(
        ONE, {"p": {"reliability": E([FAIL]), "repairability": repair, **spec}}
    )


def shelved(rbd, stock, lead, repairs=False, of=None):
    of = {"p": "p"} if of is None else of
    names = set(of.values())
    return _shelves.shelved(
        rbd,
        _shelves.Shelves(
            {name: _shelves.shelf(stock, lead, repairs) for name in names}, of
        ),
    )


def records(rbd, horizon, runs, seed, name="p"):
    tally = _event_loop._run(
        rbd, horizon, set(), set(), "p", runs, False, seed
    )
    return [run[name] for run in tally.shelves], tally


def test_no_stock_delays_every_repair_by_the_lead_time():
    # With no stock every demand waits for its own order: a repair as long
    # as the lead time and the repair, against the same unit with that
    # repair worked out exactly (an exponential repair shifted by L).
    lead, repair, horizon, runs = 100.0, 0.1, 10_000.0, 4000
    rbd = unit(E([repair]))
    rows, _ = records(shelved(rbd, 0, lead), horizon, 200, 1)
    for row in rows:
        assert row.stockouts == row.demands
        assert row.waited == pytest.approx(lead * row.demands, rel=1e-12)
        assert row.out_of_stock == horizon
    run = shelved(rbd, 0, lead).availability(
        horizon,
        mc_samples=runs,
        seed=2,
        control_variate=False,
        conditional=False,
    )
    up = np.asarray(run.uptimes) / horizon
    exact = unit(
        surv.Exponential.from_params([repair], gamma=lead)
    ).mission_availability(horizon)
    se = up.std(ddof=1) / math.sqrt(runs)
    assert abs(up.mean() - exact) < 4 * se
    # And in the long run, (L + 1/mu) / (1/lambda + L + 1/mu) down.
    down = (lead + 1 / repair) / (1 / FAIL + lead + 1 / repair)
    assert 1 - exact == pytest.approx(down, rel=0.02)


def kinds():
    """Diagrams of each kind a shelf takes: a plain unit, one repaired
    imperfectly and replaced after three failures, a standby pair, a
    block of identical units, members of a common-cause group, and a
    unit beside a sub-system."""
    W = surv.Weibull.from_params
    life = {"reliability": E([FAIL]), "repairability": E([0.05])}
    inner = RepairableRBD(ONE, {"p": dict(life)})
    yield "plain", unit(E([0.05])), {"p": "p"}
    yield "imperfect", RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": W([1000.0, 2.0]),
                "repairability": E([0.05]),
                "repair": {"model": "kijima1", "q": 0.5},
                "replace_after": 3,
            }
        },
    ), {"p": "p"}
    yield "standby", RepairableRBD(
        ONE, {"p": {**life, "standby": {"units": 2, "k": 1}}}
    ), {"p": "p"}
    yield "identical", RepairableRBD(
        ONE, {"p": {**life, "identical": {"units": 3, "k": 2}}}
    ), {"p": "p"}
    yield "common cause", RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": dict(life), "b": dict(life)},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
    ), {"a": "ab", "b": "ab"}
    yield "beside a sub-system", RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")], {"p": dict(life), "q": inner}
    ), {"p": "p"}


@pytest.mark.parametrize("kind", [k[0] for k in kinds()])
def test_a_stock_no_run_uses_up_is_no_shelf_bit_for_bit(kind):
    _, rbd, of = next(k for k in kinds() if k[0] == kind)
    horizon = 20_000.0
    plain = rbd.availability(
        horizon,
        mc_samples=300,
        seed=4,
        control_variate=False,
        conditional=False,
    )
    held = shelved(rbd, 1000, 500.0, of=of).availability(
        horizon,
        mc_samples=300,
        seed=4,
        control_variate=False,
        conditional=False,
    )
    assert np.array_equal(plain.uptimes, held.uptimes)
    name = next(iter(of.values()))
    rows, tally = records(
        shelved(rbd, 1000, 500.0, of=of), horizon, 300, 4, name
    )
    assert all(row.stockouts == 0 and row.out_of_stock == 0 for row in rows)
    # Its demands are the replacements spares_demand counts.
    counted = _event_loop._run(
        rbd, horizon, set(), set(), "p", 300, False, 4, replacements=True
    )
    column = [list(rbd.components).index(node) for node in of]
    used = np.asarray(counted.replacements)[:, column].sum(axis=1)
    assert [row.demands for row in rows] == list(used)
    # A small stock does change it.
    short = shelved(rbd, 0, 500.0, of=of).availability(
        horizon,
        mc_samples=300,
        seed=4,
        control_variate=False,
        conditional=False,
    )
    assert np.mean(short.uptimes) < np.mean(plain.uptimes)


def test_a_repair_takes_a_spare_only_when_its_shelf_says_so():
    # Minimal repair: no failure is a replacement, so it takes no spare
    # unless repairs=True, when each failure does.
    rbd = RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": surv.Weibull.from_params([1000.0, 2.0]),
                "repairability": E([0.05]),
                "repair": {"model": "kijima1", "q": 1.0},
            }
        },
    )
    rows, tally = records(shelved(rbd, 0, 50.0), 10_000.0, 100, 3)
    assert sum(row.demands for row in rows) == 0
    rows, tally = records(shelved(rbd, 0, 50.0, True), 10_000.0, 100, 3)
    assert [row.demands for row in rows] == [
        row.stockouts for row in rows
    ] and sum(row.demands for row in rows) > 0


def stationary_fill_rate(fail, lead, steps=4000):
    """The long-run share of demands met at once from a stock of one, with
    the shelf's feedback: a unit with exponential lives repaired in no
    time. Each demand's wait follows ``w' = max(0, L - w - U)``; its
    stationary chance of 0, on a grid of ``w`` (the trapezium rule)."""
    w = np.linspace(0.0, lead, steps + 1)
    h = w[1] - w[0]
    weights = np.full(steps + 1, h)
    weights[[0, -1]] = h / 2
    # From w, the chance of no wait, and the density of each w' > 0.
    none = np.exp(-fail * (lead - w))
    gap = lead - w[:, None] - w[None, :]
    density = np.where(gap >= 0.0, fail * np.exp(-fail * gap), 0.0)
    # The chain on (0, then the grid): the mass at 0 goes to each w'.
    p0, f = 1.0, np.zeros(steps + 1)
    for _ in range(2000):
        new_f = (
            p0 * fail * np.exp(-fail * (lead - w)) + (f * weights) @ density
        )
        new_p0 = p0 * math.exp(-fail * lead) + (f * weights) @ none
        total = new_p0 + (new_f * weights).sum()
        p0, f = new_p0 / total, new_f / total
    return p0


def ratio_se(waited, demands):
    rate = waited.sum() / demands.sum()
    return (
        rate,
        math.sqrt(((waited - rate * demands) ** 2).sum()) / demands.sum(),
    )


def test_the_simulated_fill_rate_matches_the_long_run_one():
    # Where stock-outs are rare, the long-run values and the simulated
    # shelf agree within four standard errors.
    lead, horizon, runs = 300.0, 200_000.0, 1000
    rbd = unit(E([1 / 24]))
    exact = rbd.spares_stock(lead, fill_rate=0.5)["p"]
    rows, _ = records(shelved(rbd, 3, lead), horizon, runs, 5)
    demands = np.array([r.demands for r in rows], float)
    waited = np.array([r.stockouts for r in rows], float)
    rate, se = ratio_se(waited, demands)
    assert abs((1 - rate) - exact.fill_rate_for(3)) < 4 * se
    out = np.array([r.out_of_stock for r in rows]) / horizon
    assert abs(out.mean() - exact.stockout_probability_for(3)) < 4 * (
        out.std(ddof=1) / math.sqrt(runs)
    )


def test_the_shelf_feedback_against_its_chain():
    # With one spare, a repair that waits keeps the unit down, so the next
    # demand comes later: more demands are met than with the demand going
    # on, as the chain of the waits has it, and the long-run value (the
    # finite source's) is the chain's.
    lead, horizon, runs = 300.0, 200_000.0, 1000
    rbd = unit()
    rows, _ = records(shelved(rbd, 1, lead), horizon, runs, 6)
    demands = np.array([r.demands for r in rows], float)
    waited = np.array([r.stockouts for r in rows], float)
    rate, se = ratio_se(waited, demands)
    chain = stationary_fill_rate(FAIL, lead)
    assert abs((1 - rate) - chain) < 4 * se
    stock = rbd.spares_stock(lead, fill_rate=0.5)["p"]
    assert stock.exact
    assert stock.fill_rate_for(1) == pytest.approx(1 / 1.3, abs=1e-12)
    assert stock.fill_rate_for(1) == pytest.approx(chain, abs=1e-5)
    going_on = dataclasses.replace(stock, sources=None).fill_rate_for(1)
    assert going_on == pytest.approx(math.exp(-FAIL * lead), rel=1e-12)
    assert chain - going_on > 10 * se


def no_stockout_case():
    # Nothing ordered arrives within the horizon: no stock-out with S on
    # the shelf is at most S failures, Poisson (repairs in no time).
    return unit(), 2000.0, 5000.0  # rbd, horizon, lead (past the horizon)


def test_no_stockout_with_no_reorders_is_the_exact_demand():
    rbd, horizon, lead = no_stockout_case()
    found = rbd.spares_stock(
        lead,
        method="simulate",
        horizon=horizon,
        no_stockout=0.95,
        mc_samples=20_000,
        seed=7,
    )["p"]
    demand = rbd.spares_demand(horizon)["p"]
    for stock, values in found.by_stock.items():
        exact = stats.poisson.cdf(stock, FAIL * horizon)
        assert float(np.sum(demand.probabilities[: stock + 1])) == (
            pytest.approx(exact, abs=1e-6)
        )
        se = math.sqrt(exact * (1 - exact) / 20_000) or 1e-12
        assert abs(values["no_stockout"] - exact) < 4 * se + 1e-12
        assert values["no_stockout_bound"] <= values["no_stockout"]
    # P(N <= 5) = 0.9834, P(N <= 4) = 0.9473, for a mean of 2.
    assert found.stock == 5
    assert isinstance(found, SimulatedSparesStock)
    assert found.to_dict()["stock"] == 5


def test_the_no_stockout_bound_covers():
    # Over many seeds, the lower bound is under the true chance at least
    # as often as its confidence, and the stock chosen reaches the target.
    rbd, horizon, lead = no_stockout_case()
    target, confidence, repeats = 0.9, 0.9, 200
    truth = {s: stats.poisson.cdf(s, FAIL * horizon) for s in range(12)}
    covered = reached = 0
    for seed in range(repeats):
        found = rbd.spares_stock(
            lead,
            method="simulate",
            horizon=horizon,
            no_stockout=target,
            confidence=confidence,
            mc_samples=400,
            seed=seed,
        )["p"]
        values = found.by_stock[3]
        covered += values["no_stockout_bound"] <= truth[3]
        reached += truth[found.stock] >= target
    # Binomial(200, 0.9) falls under 170 with chance about 1e-3.
    assert covered >= 170
    assert reached >= 170


def test_the_compiled_engine_refuses_a_shelf():
    with pytest.raises(NotImplementedError, match="a shelf of spares"):
        shelved(unit(), 1, 10.0).availability(
            100.0, mc_samples=10, seed=1, engine="numba"
        )


@pytest.mark.parametrize(
    "rbd, words",
    [
        (
            RepairableRBD(
                [("s", "p"), ("p", "q"), ("q", "t")],
                {
                    n: {"reliability": E([FAIL]), "repairability": E([0.1])}
                    for n in "pq"
                },
                repair_crews=1,
            ),
            "repair crews",
        ),
        (
            RepairableRBD(
                ONE,
                {
                    "p": {
                        "reliability": E([FAIL]),
                        "repairability": "instant",
                        "preventive": {"interval": 100.0},
                    }
                },
            ),
            "preventive task",
        ),
        (
            RepairableRBD(
                ONE,
                {
                    "p": {
                        "reliability": E([FAIL]),
                        "repairability": "instant",
                        "inspection": {"interval": 100.0},
                    }
                },
            ),
            "found by tests",
        ),
    ],
)
def test_what_a_shelf_does_not_take_is_refused(rbd, words):
    with pytest.raises(NotImplementedError, match=words):
        rbd.spares_stock(
            10.0,
            method="simulate",
            horizon=100.0,
            no_stockout=0.9,
            nodes=["p"],
        )
    with pytest.raises(NotImplementedError, match=words):
        rbd.least_spend_for_downtime(
            100.0, 1.0, spares={"p": {"price": 1.0, "lead_time": 10.0}}
        )


def test_the_simulated_stock_checks_its_arguments():
    rbd = unit()
    with pytest.raises(ValueError, match="horizon"):
        rbd.spares_stock(10.0, method="simulate", no_stockout=0.9)
    with pytest.raises(ValueError, match="no_stockout"):
        rbd.spares_stock(10.0, method="simulate", horizon=10.0)
    with pytest.raises(ValueError, match="method='simulate'"):
        rbd.spares_stock(10.0, fill_rate=0.9, horizon=100.0)
    with pytest.raises(NotImplementedError, match="fleet"):
        rbd.spares_stock(
            10.0, method="simulate", horizon=10.0, no_stockout=0.9, fleet=2
        )


def test_a_part_pools_its_members_shelf():
    life = {"reliability": E([FAIL]), "repairability": "instant"}
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": dict(life), "b": dict(life)},
    )
    kw = dict(method="simulate", horizon=2000.0, no_stockout=0.95, seed=3)
    pooled = rbd.spares_stock(5000.0, parts={"seal": ["a", "b"]}, **kw)
    alone = rbd.spares_stock(5000.0, **kw)
    # Two units: N(2000) is Poisson(4), P(N <= 8) = 0.979; alone, each 5.
    assert pooled["seal"].members == ("a", "b")
    assert pooled["seal"].stock == 8
    assert alone["a"].stock == alone["b"].stock == 5


# The spares lever.


def test_the_spares_lever_finds_the_hand_worked_stock():
    # No order arrives within the year, repairs in no time: the unit is
    # down from its (S + 1)-th failure on, so a bad year's downtime is
    # under Y with chance P(N(T - Y) <= S), Poisson(3) here. At 90%:
    # S = 4 gives 0.815, S = 5 gives 0.916.
    horizon, limit = 4000.0, 1000.0
    found = unit().least_spend_for_downtime(
        horizon,
        limit,
        level=0.9,
        spares={"p": {"price": 10.0, "lead_time": 1e9}},
        mc_samples=4000,
        seed=1,
    )
    assert found.met
    assert found.design.spares == {"p": 5}
    assert found.design.spend == pytest.approx(50.0)
    stocks = [d.spares["p"] for d in found.frontier]
    assert stocks == list(range(6))
    assert stats.poisson.cdf(4, FAIL * (horizon - limit)) < 0.9
    assert stats.poisson.cdf(5, FAIL * (horizon - limit)) > 0.9


def test_the_spares_lever_starts_from_the_stock_held_and_takes_its_max():
    found = unit().least_spend_for_downtime(
        4000.0,
        1000.0,
        level=0.9,
        spares={"p": {"price": 10.0, "lead_time": 1e9, "stock": 2, "max": 2}},
        mc_samples=2000,
        seed=1,
    )
    assert not found.met and found.floor == "caps"
    assert found.design.spares == {"p": 4}
    assert found.design.spend == pytest.approx(20.0)


def test_a_copy_draws_on_its_node_shelf():
    # A design with a copy and a spare runs the copy on its node's shelf:
    # the same runs as the diagram with the copy drawn out and one shelf.
    from repyability.rbd import _downtime_risk as risk

    life = {"reliability": E([FAIL]), "repairability": E([1 / 24])}
    rbd = RepairableRBD(ONE, {"p": dict(life)})
    levers = risk._levers(
        rbd,
        {"p": 5000.0},
        None,
        None,
        None,
        None,
        None,
        None,
        {"p": {"price": 100.0, "lead_time": 720.0}},
    )
    design = risk._shelved(
        rbd, risk._design(rbd, levers, (1, 1)), levers, (1, 1)
    )
    assert _shelves.of(design).of == {"p": "p", "p (copy 1)": "p"}
    assert _shelves.of(design).shelves["p"].stock == 1
    drawn = RepairableRBD(
        [("s", "p"), ("s", "p (copy 1)"), ("p", "t"), ("p (copy 1)", "t")],
        {"p": dict(life), "p (copy 1)": dict(life)},
    )
    kw = dict(mc_samples=300, seed=2, control_variate=False, conditional=False)
    ours = design.availability(8760.0, **kw)
    theirs = shelved(drawn, 1, 720.0, of={"p": "p", "p (copy 1)": "p"})
    assert np.array_equal(
        ours.uptimes, theirs.availability(8760.0, **kw).uptimes
    )
    # A design at step 0 still has the shelf as held (none).
    base = risk._shelved(
        rbd, risk._design(rbd, levers, (0, 0)), levers, (0, 0)
    )
    assert _shelves.of(base).shelves["p"].stock == 0


@pytest.mark.parametrize("group", ["standby", "identical"])
def test_the_spares_lever_takes_a_group(group):
    rbd = RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": E([FAIL]),
                "repairability": E([1 / 24]),
                group: {"units": 2, "k": 1},
            }
        },
    )
    found = rbd.least_spend_for_downtime(
        8760.0,
        5.0,
        spares={"p": {"price": 10.0, "lead_time": 2000.0}},
        mc_samples=500,
        seed=3,
    )
    estimates = [d.estimate for d in found.frontier]
    assert estimates == sorted(estimates, reverse=True)
    assert found.frontier[0].spares == {"p": 0}
    assert found.design.spares["p"] >= 1


def test_the_spares_lever_beside_a_sub_system():
    life = {"reliability": E([FAIL]), "repairability": E([1 / 24])}
    inner = RepairableRBD(ONE, {"p": dict(life)})
    rbd = RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")], {"p": dict(life), "q": inner}
    )
    found = rbd.least_spend_for_downtime(
        8760.0,
        300.0,
        spares={"p": {"price": 10.0, "lead_time": 720.0}},
        mc_samples=500,
        seed=3,
    )
    assert found.design.spares["p"] >= 1
    with pytest.raises(NotImplementedError, match="nested RBD"):
        rbd.least_spend_for_downtime(
            8760.0, 300.0, spares={"q": {"price": 1.0, "lead_time": 1.0}}
        )


def test_the_spares_lever_checks_its_spec():
    for bad in (
        {"price": 1.0},
        {"lead_time": 1.0},
        {"price": 0.0, "lead_time": 1.0},
        {"price": 1.0, "lead_time": -1.0},
        {"price": 1.0, "lead_time": 1.0, "stock": 1.5},
        {"price": 1.0, "lead_time": 1.0, "colour": "red"},
    ):
        with pytest.raises(ValueError):
            unit().least_spend_for_downtime(100.0, 1.0, spares={"p": bad})
    with pytest.raises(NotImplementedError, match="a shelf of spares"):
        unit().least_spend_for_downtime(
            100.0,
            1.0,
            spares={"p": {"price": 1.0, "lead_time": 1.0}},
            engine="numba",
            mc_samples=10,
        )
