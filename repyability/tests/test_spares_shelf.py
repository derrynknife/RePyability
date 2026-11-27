"""A shelf of spares in the simulation (``repyability.rbd._shelves``):
a stock reordered one for one, each order arriving a lead time later, a
repair that finds the shelf empty waiting for a spare; the stock to hold
to be sure of no stock-out (``spares_stock(method="simulate")``); and
the spares lever of ``least_spend_for_downtime``."""

import dataclasses
import math
import re

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
    yield "preventive task", RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": W([1000.0, 2.0]),
                "repairability": E([0.05]),
                "preventive": {"interval": 400.0},
            }
        },
    ), {"p": "p"}
    yield "block task taking time", RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": W([1000.0, 2.0]),
                "repairability": E([0.05]),
                "preventive": {
                    "interval": 400.0,
                    "policy": "block",
                    "duration": E([0.2]),
                },
            }
        },
    ), {"p": "p"}
    yield "tested", RepairableRBD(
        ONE,
        {
            "p": {
                **life,
                "inspection": {"interval": 100.0, "duration": E([0.5])},
            }
        },
    ), {"p": "p"}
    yield "repair crews", RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")],
        {"p": dict(life), "q": dict(life)},
        repair_crews=1,
    ), {"p": "p", "q": "p"}
    yield "standby with crews", RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")],
        {"p": {**life, "standby": {"units": 2, "k": 1}}, "q": dict(life)},
        repair_crews=1,
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


def refusing(kind, spares=None):
    """A diagram whose node ``p`` cannot draw on a shelf: in a
    maintenance group, renewed early at shutdowns, or paused by one
    with repair crews (with the spec's ``spares``, if given)."""
    life = {"reliability": E([FAIL]), "repairability": E([0.1])}
    p = dict(life, **({} if spares is None else {"spares": spares}))
    edges = [("s", "p"), ("p", "q"), ("q", "t")]
    if kind == "group":
        task = {"preventive": {"interval": 300.0, "opportunity": 200.0}}
        return RepairableRBD(
            edges,
            {
                "p": {**p, **task, "group": "g"},
                "q": {**life, **task, "group": "g"},
            },
            maintenance_groups={"g": {"setup_cost": 1.0}},
        )
    if kind == "opportunistic":
        shut = {
            "period": 500.0,
            "duration": 8.0,
            "scope": ["p", "q"],
            "opportunistic": {"p": 100.0},
        }
        task = {"preventive": {"interval": 300.0}}
        return RepairableRBD(
            edges,
            {"p": {**p, **task}, "q": dict(life)},
            shutdowns={"stop": shut},
        )
    assert kind == "paused"
    shut = {
        "period": 500.0,
        "duration": 8.0,
        "scope": ["p"],
        "idle_ageing": False,
    }
    return RepairableRBD(
        edges,
        {"p": p, "q": dict(life)},
        shutdowns={"stop": shut},
        repair_crews=1,
    )


def test_a_unit_paused_with_crews_is_refused_before_its_shelf():
    # A shut that stands an idle block still, with repair crews, is
    # refused by the shutdowns' own check, shelf or none: the shelf's
    # own refusal for it is never reached.
    for spares in (None, {"stock": 1, "lead_time": 10.0}):
        with pytest.raises(ValueError, match="waits for the repair crews"):
            refusing("paused", spares)


@pytest.mark.parametrize(
    "rbd, words",
    [
        (
            RepairableRBD(
                ONE,
                {
                    "p": {
                        "reliability": E([FAIL]),
                        "repairability": "instant",
                        "preventive": {
                            "interval": 100.0,
                            "policy": "condition",
                            "threshold": 0.1,
                        },
                    }
                },
            ),
            "replaced on condition",
        ),
        (
            RepairableRBD(
                [("s", "p"), ("p", "q"), ("q", "t")],
                {
                    n: {
                        "reliability": E([FAIL]),
                        "repairability": E([0.1]),
                        "inspection": {"interval": 100.0},
                    }
                    for n in "pq"
                },
                repair_crews=1,
            ),
            "found by tests, and with repair crews",
        ),
        (refusing("group"), "maintenance group"),
        (refusing("opportunistic"), "shutdowns' renewals"),
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


def long_run(rbd, horizon, runs, seed):
    run = rbd.availability(
        horizon,
        mc_samples=runs,
        seed=seed,
        control_variate=False,
        conditional=False,
    )
    up = np.asarray(run.uptimes) / horizon
    return run, up.mean(), up.std(ddof=1) / math.sqrt(runs)


def test_a_task_that_finds_the_shelf_empty_is_put_off():
    # Age replacement at T in no time, failures exponential and replaced
    # in no time, no stock: a task orders its spare and the unit runs on
    # until it is in, L later; a failure before then takes that spare and
    # waits for it. From each renewal, with X the life:
    #   X < T: down L, a cycle of X + L;
    #   X >= T: a cycle of T + L, down (T + L - X)+.
    lam, T, lead = 1e-3, 500.0, 300.0
    rbd = unit(preventive={"interval": T})
    q = math.exp(-lam * T)
    cycle = (1 - q) / lam - T * q + lead * (1 - q) + (T + lead) * q
    down = lead * (1 - q) + q * (lead - (1 - math.exp(-lam * lead)) / lam)
    _, mean, se = long_run(shelved(rbd, 0, lead), 200_000.0, 400, 1)
    assert abs(mean - (1 - down / cycle)) < 4 * se
    # Taken down to wait instead, it would be down L every cycle.
    assert 1 - down / cycle - (1 - lead / ((1 - q) / lam + lead)) > 0.2
    # Each demand is a cycle's: a task's or a failure's, never both.
    rows, _ = records(shelved(rbd, 0, lead), 200_000.0, 400, 1)
    demands = np.array([r.demands for r in rows], float)
    assert demands.mean() == pytest.approx(200_000.0 / cycle, rel=0.01)


def test_a_test_that_finds_a_failure_waits_for_its_spare():
    # Tests every tau in no time, repairs in no time, no stock and a lead
    # time of tau: from each renewal (on a test) the failure is found at
    # the first test after it, the unit back L later, so a cycle is
    # tau / (1 - exp(-lam tau)) + L long, up 1 / lam of it.
    lam, tau = 1e-3, 200.0
    rbd = unit(inspection={"interval": tau})
    cycle = tau / (1 - math.exp(-lam * tau)) + tau
    _, mean, se = long_run(shelved(rbd, 0, tau), 200_000.0, 400, 1)
    assert abs(mean - (1 / lam) / cycle) < 4 * se
    rows, _ = records(shelved(rbd, 0, tau), 200_000.0, 50, 1)
    assert all(r.demands == r.stockouts > 0 for r in rows)
    assert all(
        r.waited == pytest.approx(tau * r.demands, rel=1e-9) for r in rows
    )


def crewed(group=False):
    lam, mu = 1e-3, 0.05
    # A group's units need a repair time: a billionth of an hour.
    p = {
        "reliability": E([lam]),
        "repairability": E([1e9]) if group else "instant",
    }
    if group:
        p["standby"] = {"units": 2, "k": 1}
    q = {"reliability": E([2 * lam]), "repairability": E([mu])}
    edges = [("s", "p"), ("p", "q"), ("q", "t")]
    return (
        RepairableRBD(edges, {"p": p, "q": q}, repair_crews=1),
        RepairableRBD(edges, {"p": dict(p), "q": dict(q)}),
        lam,
        mu,
    )


@pytest.mark.parametrize("group", [False, True])
def test_a_crew_is_not_held_while_a_spare_is_on_its_way(group):
    # p's repairs take no time but wait L for their spare; q's take a crew,
    # the only one. Were the crew held while p waits, q would wait too:
    # it never does, so q is as with no crews to the bit.
    lead, horizon, runs = 200.0, 200_000.0, 400
    with_crews, without, lam, mu = crewed(group)
    a = shelved(with_crews, 0, lead).availability(
        horizon,
        mc_samples=runs,
        seed=2,
        control_variate=False,
        conditional=False,
    )
    b = shelved(without, 0, lead).availability(
        horizon,
        mc_samples=runs,
        seed=2,
        control_variate=False,
        conditional=False,
    )
    assert a.node_uptime["q"] == pytest.approx(b.node_uptime["q"], rel=1e-12)
    if group:
        return
    # Once its spare is in, p's job queues for the crew: it waits for q's
    # repair when q is under one (q's long-run share 2 lam / (2 lam + mu),
    # its repair left 1 / mu), q being independent of p. So p is down
    # L + 2 lam / (2 lam + mu) / mu per failure.
    exact = (1 / lam) / (1 / lam + lead + 2 * lam / (2 * lam + mu) / mu)
    # p's share up in batches of runs on seeds of their own.
    batches = [
        shelved(with_crews, 0, lead)
        .availability(
            horizon,
            mc_samples=100,
            seed=10 + k,
            control_variate=False,
            conditional=False,
        )
        .node_uptime["p"]
        / (100 * horizon)
        for k in range(10)
    ]
    se = np.std(batches, ddof=1) / math.sqrt(len(batches))
    assert abs(np.mean(batches) - exact) < 4 * se
    # Under no crews' 1 / (1 + lam L), by about 0.0005.
    assert np.mean(batches) < (1 / lam) / (1 / lam + lead)


def nested_unit(lead_past=False):
    inner = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": {"reliability": E([FAIL]), "repairability": "instant"},
            "b": {"reliability": E([2 * FAIL]), "repairability": "instant"},
        },
    )
    return RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")],
        {
            "p": {"reliability": E([FAIL]), "repairability": "instant"},
            "q": inner,
        },
    )


def test_a_sub_system_component_draws_on_a_shelf_in_its_own_loop():
    rbd = nested_unit()
    horizon, lead = 200_000.0, 300.0
    # A stock no run uses up changes nothing.
    of = {("q", "a"): "qa"}
    plain = long_run(rbd, 20_000.0, 200, 3)[0]
    held = long_run(shelved(rbd, 1000, lead, of=of), 20_000.0, 200, 3)[0]
    assert np.array_equal(plain.uptimes, held.uptimes)
    # No stock: every one of a's failures waits L, in the sub-system's
    # loop, so a is up 1 / (1 + lam L) of the time in the long run and
    # the system (p, a, b in series) as each alone.
    run = shelved(rbd, 0, lead, of=of).availability(
        horizon,
        mc_samples=200,
        seed=4,
        control_variate=False,
        conditional=False,
    )
    rows, _ = records(shelved(rbd, 0, lead, of=of), horizon, 50, 4, "qa")
    assert all(r.demands == r.stockouts > 0 for r in rows)
    assert all(
        r.waited == pytest.approx(lead * r.demands, rel=1e-9) for r in rows
    )
    up = np.asarray(run.uptimes) / horizon
    exact = 1 / (1 + FAIL * lead)
    se = up.std(ddof=1) / math.sqrt(len(up))
    assert abs(up.mean() - exact) < 4 * se


def test_the_simulated_stock_counts_a_sub_system_components():
    rbd = nested_unit()
    found = rbd.spares_stock(
        5000.0,
        method="simulate",
        horizon=2000.0,
        no_stockout=0.95,
        mc_samples=4000,
        seed=7,
    )
    # Nothing ordered arrives: the stock is the Poisson's, mean 2 for p
    # and a (5 spares for 95%), 4 for b (8).
    assert set(found) == {"p", ("q", "a"), ("q", "b")}
    assert found["p"].stock == found[("q", "a")].stock == 5
    assert found[("q", "b")].stock == 8


def test_a_shelf_across_levels_is_refused():
    rbd = nested_unit()
    with pytest.raises(NotImplementedError, match="two levels"):
        shelved(rbd, 1, 10.0, of={"p": "x", ("q", "a"): "x"})


def test_a_fleet_draws_on_one_shelf():
    rbd = unit()
    found = rbd.spares_stock(
        5000.0,
        method="simulate",
        horizon=2000.0,
        no_stockout=0.95,
        mc_samples=4000,
        seed=3,
        fleet=2,
    )["p"]
    # Two systems: N(2000) is Poisson(4), P(N <= 8) = 0.979 (7: 0.949).
    assert found.stock == 8
    assert found.by_stock[7]["no_stockout"] == pytest.approx(
        stats.poisson.cdf(7, 4.0), abs=4 * math.sqrt(0.949 * 0.051 / 4000)
    )
    crewed = RepairableRBD(
        ONE,
        {"p": {"reliability": E([FAIL]), "repairability": E([0.1])}},
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="repair_crews"):
        crewed.spares_stock(
            10.0, method="simulate", horizon=10.0, no_stockout=0.9, fleet=2
        )


# The shelf as a spec key (a component's "spares").


def with_spares(rbd, of, stock, lead, repairs=False):
    """``rbd`` built again with each node of ``of`` given its own shelf
    in its spec (``"spares"``)."""
    args = dict(rbd._init_args)
    components = dict(args["components"])
    for node in of:
        spares = {"stock": stock, "lead_time": lead}
        if repairs:
            spares["repairs"] = True
        components[node] = {**components[node], "spares": spares}
    args["components"] = components
    return RepairableRBD(**args)


#: The kinds whose nodes each have a shelf of their own (a spec's shelf
#: is its node's alone).
OWN = [k[0] for k in kinds() if len(set(k[2].values())) == len(k[2])]


@pytest.mark.parametrize("kind", OWN)
def test_a_shelf_in_the_spec_is_the_shelf_given_bit_for_bit(kind):
    _, rbd, of = next(k for k in kinds() if k[0] == kind)
    horizon = 20_000.0
    plain = dict(
        mc_samples=300, seed=4, control_variate=False, conditional=False
    )
    for stock, lead, repairs in ((0, 500.0, False), (1, 500.0, True)):
        ours = with_spares(rbd, of, stock, lead, repairs)
        theirs = shelved(rbd, stock, lead, repairs, of={n: n for n in of})
        a = ours.availability(horizon, **plain)
        b = theirs.availability(horizon, **plain)
        assert np.array_equal(a.uptimes, b.uptimes)
        assert a.node_uptime == b.node_uptime
        rows, _ = records(ours, horizon, 300, 4, _shelves.Own(next(iter(of))))
        given, _ = records(theirs, horizon, 300, 4, next(iter(of)))
        assert rows == given
    # Its default run (control variates and conditional runs where they
    # apply) is the plain run's: nothing exact is taken for the node.
    assert np.array_equal(
        ours.availability(horizon, mc_samples=300, seed=4).uptimes,
        a.uptimes,
    )
    # A route's shelves are not those of the specs the modules of a
    # conditional run are built from: its default run is the plain one.
    default = theirs.availability(horizon, mc_samples=300, seed=4)
    assert default.mean_availability == b.mean_availability


def test_a_sub_system_component_shelf_in_its_spec():
    # nested_unit's, with a's shelf in its spec.
    given = nested_unit()
    inner = with_spares(given.components["q"], ["a"], 0, 300.0)
    ours = RepairableRBD(
        given._init_args["edges"],
        {**given._init_args["components"], "q": inner},
    )
    assert _shelves.of(ours).of == {("q", "a"): _shelves.Own(("q", "a"))}
    theirs = shelved(nested_unit(), 0, 300.0, of={("q", "a"): "qa"})
    kw = dict(mc_samples=200, seed=4, control_variate=False, conditional=False)
    a, b = ours.availability(20_000.0, **kw), theirs.availability(
        20_000.0, **kw
    )
    assert np.array_equal(a.uptimes, b.uptimes)
    rows, _ = records(ours, 20_000.0, 50, 4, _shelves.Own(("q", "a")))
    assert all(r.demands == r.stockouts > 0 for r in rows)
    # The sub-system alone draws on it too, and its exact values refuse.
    assert _shelves.of(inner).of == {"a": _shelves.Own("a")}
    with pytest.raises(NotImplementedError, match="shelf of spares"):
        inner.mean_availability()
    with pytest.raises(NotImplementedError, match="shelf of spares"):
        ours.mission_availability(100.0)


#: The analyses of the node probabilities given, or of the structure
#: alone, which a shelf does not change.
MODEL_FREE = {
    "structural_importance",
    "system_probability",
    "path_set_probabilities",
    "improvement_allocation",
    "equal_allocation",
    "simple_allocation",
    "cost_based_allocation",
    "minimum_effort_allocation",
}


def spec_shelved():
    """The catalogue's diagrams with a shelf, and each kind a shelf takes
    with a shelf of its own on each node given one."""
    from repyability.tests.catalogue import repairable_kinds

    out = {
        name: rbd
        for name, rbd in repairable_kinds().items()
        if _shelves.of(rbd) is not None
    }
    assert len(out) == 2
    for name, rbd, of in kinds():
        out[name] = with_spares(rbd, of, 1, 500.0)
    return out


@pytest.mark.parametrize("name", sorted(spec_shelved()))
def test_no_exact_method_takes_a_shelf(name):
    # The only exact or numerical analyses are those of the node
    # probabilities given or of the structure alone: every other refuses,
    # saying to simulate, and each method does what the routes say.
    from repyability.rbd import routes
    from repyability.tests.test_analysis_routes import (
        REPAIRABLE_CALLS,
        behaves_as_reported,
        outside_groups,
    )

    rbd = spec_shelved()[name]
    report = rbd.analysis_routes()
    computed = {
        method
        for method, route in report.items()
        if route.route in (routes.EXACT, routes.NUMERICAL)
    }
    assert computed <= MODEL_FREE, computed - MODEL_FREE
    assert report["availability"].route == routes.SIMULATED
    assert report["availability"].engine == "python"
    if outside_groups(rbd) - set(rbd.in_or_out):
        # (The calls need a component outside the common-cause groups.)
        for method, call in REPAIRABLE_CALLS.items():
            behaves_as_reported(report[method], lambda: call(rbd))
    # The availability's mean takes nothing exact for the shelved nodes.
    run = rbd.availability(1000.0, mc_samples=50, seed=1)
    assert run.mean_availability_interval().method != "exact"


def test_the_compiled_engine_refuses_a_shelf_in_the_spec():
    rbd = unit(spares={"stock": 1, "lead_time": 10.0})
    with pytest.raises(NotImplementedError, match="a shelf of spares"):
        rbd.availability(100.0, mc_samples=10, seed=1, engine="numba")
    # "auto" runs Python.
    run = rbd.availability(100.0, mc_samples=10, seed=1, engine="auto")
    assert len(run.uptimes) == 10


def test_the_spares_spec_is_checked():
    for bad in (
        5,
        {"stock": 1},
        {"lead_time": 1.0},
        {"stock": -1, "lead_time": 1.0},
        {"stock": 1.5, "lead_time": 1.0},
        {"stock": 1, "lead_time": -1.0},
        {"stock": 1, "lead_time": 1.0, "repairs": "yes"},
        {"stock": 1, "lead_time": 1.0, "colour": "red"},
    ):
        with pytest.raises(ValueError, match="component 'p'|Component 'p'"):
            unit(spares=bad)


def sub_system(more=None, **options):
    """A system whose sub-system ``q``'s component ``x`` has a shelf in
    its spec (and ``more`` in it)."""
    x = {
        "reliability": E([FAIL]),
        "repairability": E([0.1]),
        "spares": {"stock": 1, "lead_time": 10.0},
        **(more or {}),
    }
    inner = RepairableRBD(
        [("s", "x"), ("x", "y"), ("y", "t")],
        {"x": x, "y": {"reliability": E([FAIL]), "repairability": E([0.1])}},
        **options,
    )
    return RepairableRBD(
        [("s", "p"), ("p", "q"), ("q", "t")],
        {
            "p": {"reliability": E([FAIL]), "repairability": "instant"},
            "q": inner,
        },
    )


@pytest.mark.parametrize(
    "build, words",
    [
        (
            lambda: unit(
                preventive={
                    "interval": 100.0,
                    "policy": "condition",
                    "threshold": 0.1,
                },
                spares={"stock": 1, "lead_time": 10.0},
            ),
            "'policy': 'age' or 'block'",
        ),
        (
            lambda: refusing("group", {"stock": 1, "lead_time": 10.0}),
            "take it out of its maintenance group",
        ),
        (
            lambda: refusing("opportunistic", {"stock": 1, "lead_time": 10.0}),
            "take it out of its maintenance group and the shutdowns'",
        ),
        (
            lambda: RepairableRBD(
                [("s", "p"), ("p", "q"), ("q", "t")],
                {
                    "p": {
                        "reliability": E([FAIL]),
                        "repairability": E([0.1]),
                        "inspection": {"interval": 100.0},
                        "spares": {"stock": 1, "lead_time": 10.0},
                    },
                    "q": {"reliability": E([FAIL]), "repairability": E([0.1])},
                },
                repair_crews=1,
            ),
            "give the diagram no repair_crews",
        ),
        (
            lambda: sub_system({"preventive": {"interval": 300.0}}),
            "simulated as a diagram of its own",
        ),
        (
            lambda: sub_system({"inspection": {"interval": 300.0}}),
            "simulated as a diagram of its own",
        ),
        (
            lambda: sub_system(repair_crews=1),
            "give the sub-system no repair_crews",
        ),
        (
            lambda: RepairableRBD(
                [("s", "r"), ("r", "t")],
                {"r": sub_system()},
            ),
            "two levels down",
        ),
    ],
)
def test_what_a_spec_shelf_does_not_take_is_refused_saying_what_to_do(
    build, words
):
    # Refused when the diagram is built, so no analysis can run it.
    with pytest.raises(NotImplementedError, match=re.escape(words)):
        build()


def test_the_spares_spec_is_saved_and_loaded():
    rbd = with_spares(
        RepairableRBD(
            [("s", "p"), ("p", "q"), ("q", "t")],
            {
                n: {"reliability": E([FAIL]), "repairability": E([0.05])}
                for n in "pq"
            },
        ),
        ["p"],
        2,
        300.0,
        repairs=True,
    )
    back = RepairableRBD.from_json(rbd.to_json())
    assert back._init_args["components"]["p"]["spares"] == {
        "stock": 2,
        "lead_time": 300.0,
        "repairs": True,
    }
    assert _shelves.of(back) == _shelves.of(rbd)
    kw = dict(mc_samples=100, seed=6, control_variate=False, conditional=False)
    assert np.array_equal(
        back.availability(10_000.0, **kw).uptimes,
        rbd.availability(10_000.0, **kw).uptimes,
    )
    # A unit's plan holds it too.
    from repyability.maintenance import UnitPlan

    spec = rbd._init_args["components"]["p"]
    plan = UnitPlan.from_spec(spec)
    assert plan.spares == spec["spares"]
    assert plan.spec()["spares"] == spec["spares"]


def test_the_routes_shelves_go_over_the_spec_s():
    life = {"reliability": E([FAIL]), "repairability": "instant"}
    rbd = with_spares(
        RepairableRBD(
            [("s", "p"), ("p", "q"), ("q", "t")],
            {"p": dict(life), "q": dict(life)},
        ),
        ["p", "q"],
        3,
        1e9,
    )
    given = _shelves.Shelves({"p": _shelves.shelf(0, 5.0)}, {"p": "p"})
    merged = _shelves.of(_shelves.shelved(rbd, given))
    assert merged.of == {"p": "p", "q": _shelves.Own("q")}
    assert merged.shelves == {
        "p": _shelves.shelf(0, 5.0),
        _shelves.Own("q"): _shelves.shelf(3, 1e9),
    }
    # The simulated stock tries its own on the nodes it sizes: the
    # spec's stock there changes nothing.
    kw = dict(
        method="simulate",
        horizon=2000.0,
        no_stockout=0.95,
        seed=3,
        nodes=["p"],
        mc_samples=2000,
    )
    found = rbd.spares_stock(5000.0, **kw)["p"]
    again = with_spares(rbd, ["p"], 0, 1e9).spares_stock(5000.0, **kw)["p"]
    assert found.by_stock == again.by_stock
    # Nothing arrives in the window: P(Poisson(2) <= 5) = 0.983 (4: 0.947).
    assert found.stock == 5


def test_the_spares_lever_starts_from_the_spec_s_shelf():
    kw = dict(level=0.9, mc_samples=2000, seed=1)
    ours = unit(
        spares={"stock": 2, "lead_time": 1e9}
    ).least_spend_for_downtime(
        4000.0, 1000.0, spares={"p": {"price": 10.0}}, **kw
    )
    theirs = unit().least_spend_for_downtime(
        4000.0,
        1000.0,
        spares={"p": {"price": 10.0, "lead_time": 1e9, "stock": 2}},
        **kw,
    )
    assert ours.design.spares == theirs.design.spares == {"p": 5}
    assert [d.estimate for d in ours.frontier] == [
        d.estimate for d in theirs.frontier
    ]
    # Without a shelf of its own, the lever needs a lead time.
    with pytest.raises(ValueError, match="lead_time"):
        unit().least_spend_for_downtime(
            4000.0, 1000.0, spares={"p": {"price": 10.0}}
        )


def test_a_copy_draws_on_its_node_spec_shelf():
    from repyability.rbd import _downtime_risk as risk

    life = {"reliability": E([FAIL]), "repairability": E([1 / 24])}
    rbd = RepairableRBD(
        ONE, {"p": {**life, "spares": {"stock": 1, "lead_time": 720.0}}}
    )
    levers = risk._levers(
        rbd, {"p": 5000.0}, None, None, None, None, None, None, None
    )
    design = risk._shelved(rbd, risk._design(rbd, levers, (1,)), levers, (1,))
    assert _shelves.of(design).of == {
        "p": _shelves.Own("p"),
        "p (copy 1)": _shelves.Own("p"),
    }
    drawn = RepairableRBD(
        [("s", "p"), ("s", "p (copy 1)"), ("p", "t"), ("p (copy 1)", "t")],
        {"p": dict(life), "p (copy 1)": dict(life)},
    )
    kw = dict(mc_samples=300, seed=2, control_variate=False, conditional=False)
    theirs = shelved(drawn, 1, 720.0, of={"p": "p", "p (copy 1)": "p"})
    assert np.array_equal(
        design.availability(8760.0, **kw).uptimes,
        theirs.availability(8760.0, **kw).uptimes,
    )
