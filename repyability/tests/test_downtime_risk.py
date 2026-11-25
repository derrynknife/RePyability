"""The least spend that keeps a bad run's downtime under a limit (UC-68 of
the maintenance README): ``RepairableRBD.least_spend_for_downtime`` and
the money module's risk measures (``repyability.value._risk``)."""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy import optimize, stats

from repyability import RepairableRBD
from repyability.maintenance import Prices, UnitPlan
from repyability.value import _risk

E = surv.Exponential.from_params
W = surv.Weibull.from_params
ONE = [("s", "p"), ("p", "t")]
YEAR = 8760.0
FAIL, REPAIR = 1e-3, 0.02  # a pump: 1,000 h between failures, 50 h repairs


def pump(**more):
    return RepairableRBD(
        ONE,
        {"p": {"reliability": E([FAIL]), "repairability": E([REPAIR])}},
        **more,
    )


def worn(**preventive):
    """A wearing unit, its 8 h task every 8,000 h, repairs of 72 h."""
    spec = {
        "reliability": W([4000.0, 3.0]),
        "repairability": E([1 / 72.0]),
        "repair_cost": 2000.0,
        "preventive": {
            "interval": 8000.0,
            "cost": 300.0,
            "duration": surv.ExactEventTime.from_params([8.0]),
            **preventive,
        },
    }
    return RepairableRBD(ONE, {"p": spec})


INTERVALS = {"p": [4000.0, 2000.0, 1000.0, 500.0]}


def downtime_cdf(x, fail, repair, T, tail=1e-14):
    """P(time down in [0, T] <= x) for a unit up at 0 with exponential
    lives and repairs (an alternating renewal process): uniformised at
    rate L, given n jumps the n + 1 spacings are Dirichlet(1, ..., 1), so
    k of them spent down sum to T * Beta(k, n + 1 - k); the chance of k
    comes from the uniformised chain."""
    L = max(fail, repair)
    up_stays, down_stays = 1.0 - fail / L, 1.0 - repair / L
    top = int(stats.poisson.isf(tail, L * T)) + 1
    u = min(max(x / T, 0.0), 1.0)
    total = 0.0
    dp = np.zeros((2, top + 2))
    dp[0, 0] = 1.0
    for n in range(top + 1):
        pk = dp.sum(axis=0)[: n + 2]
        inner = pk[0] + (pk[n + 1] if u >= 1.0 else 0.0)
        k = np.arange(1, n + 1)
        if k.size:
            inner += float(pk[1 : n + 1] @ stats.beta.cdf(u, k, n + 1 - k))
        total += stats.poisson.pmf(n, L * T) * inner
        new = np.zeros_like(dp)
        new[0] += dp[0] * up_stays + dp[1] * (1.0 - down_stays)
        new[1, 1:] += dp[0, :-1] * (1.0 - up_stays) + dp[1, :-1] * down_stays
        dp = new
    return total


def test_the_exact_downtime_distribution_at_its_ends():
    # Never failing by T has chance exp(-fail T); never up is impossible.
    assert downtime_cdf(0.0, FAIL, REPAIR, YEAR) == pytest.approx(
        math.exp(-FAIL * YEAR), rel=1e-9
    )
    assert downtime_cdf(YEAR, FAIL, REPAIR, YEAR) == pytest.approx(1.0)


def test_a_single_unit_against_the_exact_percentile():
    exact = optimize.brentq(
        lambda x: downtime_cdf(x, FAIL, REPAIR, YEAR) - 0.95, 1.0, YEAR
    )
    density = (
        downtime_cdf(exact + 1.0, FAIL, REPAIR, YEAR)
        - downtime_cdf(exact - 1.0, FAIL, REPAIR, YEAR)
    ) / 2.0
    n = 4000
    found = pump().least_spend_for_downtime(YEAR, 1e6, mc_samples=n, seed=21)
    design = found.design
    # The quantile's standard error, sqrt(p (1 - p) / n) / f(q).
    se = math.sqrt(0.95 * 0.05 / n) / density
    assert abs(design.estimate - exact) < 4 * se
    assert design.bound >= design.estimate
    assert design.planned["mean_hours"] == 0.0
    assert found.met and found.bound_method.startswith("order statistic")


def test_the_quantile_bound_is_the_binomial_order_statistic():
    ordered = np.arange(1.0, 101.0)
    bound = _risk.quantile_bound(ordered, 0.9, 0.95)
    k = int(bound)  # the k-th smallest
    assert stats.binom.cdf(k - 1, 100, 0.9) >= 0.95
    assert stats.binom.cdf(k - 2, 100, 0.9) < 0.95
    assert _risk.quantile_bound(ordered[:10], 0.99, 0.95) == math.inf
    assert _risk.tail_mean(ordered, 0.9) == pytest.approx(95.5)


def test_at_level_one_half_a_symmetric_loss_gives_its_median():
    values = stats.norm.rvs(size=2001, random_state=np.random.default_rng(6))
    found = _risk.risk(values, 0.5, "var", 0.95)
    assert found.estimate == float(np.median(values))


def test_redundancy_meets_any_target_at_a_rising_spend():
    spends = []
    for limit in (200.0, 50.0, 10.0, 1.0):
        found = pump().least_spend_for_downtime(
            YEAR, limit, redundancy={"p": 5000.0}, mc_samples=1000, seed=40
        )
        assert found.met and found.design.bound <= limit
        spends.append(found.design.spend)
    assert spends == sorted(spends) and spends[-1] > spends[0]


def test_more_spend_never_raises_the_frontier():
    found = worn().least_spend_for_downtime(
        YEAR,
        10.0,
        downtime="all",
        intervals=INTERVALS,
        mc_samples=1000,
        seed=41,
    )
    spends = [d.spend for d in found.frontier]
    risks = [d.estimate for d in found.frontier]
    assert spends == sorted(spends)
    assert all(a > b for a, b in zip(risks, risks[1:]))


def test_cutting_unplanned_downtime_raises_planned_downtime():
    found = worn().least_spend_for_downtime(
        YEAR,
        10.0,
        downtime="unplanned",
        intervals=INTERVALS,
        mc_samples=1000,
        seed=42,
    )
    planned = [d.planned["mean_hours"] for d in found.frontier]
    unplanned = [d.unplanned["bad_hours"] for d in found.frontier]
    assert found.met
    assert planned == sorted(planned) and planned[-1] > planned[0]
    assert unplanned == sorted(unplanned, reverse=True)


def test_planned_downtime_is_the_floor_named():
    found = worn().least_spend_for_downtime(
        YEAR, 10.0, intervals=INTERVALS, mc_samples=1000, seed=43
    )
    assert not found.met and found.floor == "planned"
    assert "planned task downtime" in found.stopped


def test_the_budget_is_the_floor_named():
    found = pump().least_spend_for_downtime(
        YEAR,
        100.0,
        redundancy={"p": 5000.0},
        budget=3000.0,
        mc_samples=500,
        seed=44,
    )
    assert not found.met and found.floor == "budget"
    assert "budget" in found.stopped


def test_seeded_and_in_cost():
    one = pump(downtime_cost_rate=100.0).least_spend_for_downtime(
        YEAR,
        5e4,
        measure="cost",
        redundancy={"p": 5000.0},
        mc_samples=500,
        seed=45,
    )
    again = pump(downtime_cost_rate=100.0).least_spend_for_downtime(
        YEAR,
        5e4,
        measure="cost",
        redundancy={"p": 5000.0},
        mc_samples=500,
        seed=45,
    )
    hours = pump().least_spend_for_downtime(
        YEAR, 500.0, redundancy={"p": 5000.0}, mc_samples=500, seed=45
    )
    assert [d.estimate for d in one.frontier] == [
        d.estimate for d in again.frontier
    ]
    assert one.frontier[0].estimate == pytest.approx(
        100.0 * hours.frontier[0].estimate, rel=1e-12
    )
    assert one.frontier[0].unplanned["bad_cost"] == pytest.approx(
        100.0 * hours.frontier[0].unplanned["bad_hours"], rel=1e-12
    )


def test_a_plan_alone_takes_its_diagram_route():
    plan = UnitPlan(E([FAIL]), E([REPAIR]), prices=Prices(repair=10.0))
    mine = plan.least_spend_for_downtime(
        YEAR, 100.0, redundancy={"unit": 5000.0}, mc_samples=500, seed=46
    )
    theirs = plan.diagram().least_spend_for_downtime(
        YEAR, 100.0, redundancy={"unit": 5000.0}, mc_samples=500, seed=46
    )
    assert mine.design.estimate == theirs.design.estimate
    assert plan.analysis_routes()["least_spend_for_downtime"].route == (
        "simulated"
    )


def test_refusals():
    with pytest.raises(ValueError, match="needs downtime to cost"):
        pump().least_spend_for_downtime(YEAR, 10.0, measure="cost")
    with pytest.raises(ValueError, match="no preventive task"):
        pump().least_spend_for_downtime(YEAR, 10.0, intervals={"p": [100]})
    with pytest.raises(ValueError, match="level"):
        pump().least_spend_for_downtime(YEAR, 10.0, level=1.0)
    with pytest.raises(ValueError, match="price of a copy"):
        pump().least_spend_for_downtime(YEAR, 10.0, redundancy={"p": 0.0})
    with pytest.raises(ValueError, match="limit"):
        pump().least_spend_for_downtime(YEAR, -1.0)


TWO = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def pair(**more):
    """Two pumps in parallel, a failure every 50 h and 50 h repairs."""
    return RepairableRBD(
        TWO,
        {
            n: {"reliability": E([0.02]), "repairability": E([0.02])}
            for n in "ab"
        },
        **more,
    )


def direct(diagram, n, seed):
    """A diagram's own run, as the search runs each design."""
    run = diagram.availability(
        YEAR,
        mc_samples=n,
        seed=seed,
        control_variate=False,
        conditional=False,
    )
    return YEAR - np.asarray(run.uptimes, float), run


def test_a_crew_more_is_the_diagram_with_it_and_its_salary():
    found = pair(repair_crews=1, crew_cost_rate=2.0).least_spend_for_downtime(
        YEAR, 1.0, crews={"price": 100.0}, mc_samples=1000, seed=50
    )
    # A third crew has no job two pumps do not already give the two.
    assert not found.met and found.floor == "lowest"
    assert [d.crews for d in found.frontier] == [1, 2]
    best = found.frontier[-1]
    assert best.spend == pytest.approx(100.0 + 2.0 * YEAR, rel=1e-12)
    down, _ = direct(pair(repair_crews=2, crew_cost_rate=2.0), 1000, 50)
    assert best.unplanned["mean_hours"] == pytest.approx(down.mean())
    # Against the crews' Markov chain: with two crews each pump is down
    # half the time on its own, both a quarter (2,190 h a year); with
    # one, the chain on the pumps down has shares 1 : 2 : 2, both down
    # 0.4 of the time (3,504 h).
    exact = pair(repair_crews=2).mean_unavailability() * YEAR
    assert exact == pytest.approx(YEAR / 4.0, rel=1e-9)
    assert abs(down.mean() - exact) < 4 * down.std() / math.sqrt(down.size)
    first = found.frontier[0].unplanned["mean_hours"]
    one = pair(repair_crews=1).mean_unavailability() * YEAR
    assert one == pytest.approx(0.4 * YEAR, rel=1e-9)
    assert abs(first - one) < 0.05 * one


def test_a_call_out_contract_is_priced_by_its_runs():
    contract = {"when_waiting": 1, "fee": 50.0, "rate": 3.0}
    found = pair(repair_crews=1, crew_cost_rate=2.0).least_spend_for_downtime(
        YEAR,
        1.0,
        call_out=[{"when_waiting": 2, "fee": 1.0}, contract],
        mc_samples=1000,
        seed=51,
    )
    best = found.frontier[-1]
    assert best.call_out == contract
    # Its spend is its fees and time engaged, the change in the crews'
    # cost of its runs; the salaries are the same in both.
    _, plain = direct(pair(repair_crews=1, crew_cost_rate=2.0), 1000, 51)
    down, called = direct(
        pair(repair_crews=1, crew_cost_rate=2.0, call_out=contract), 1000, 51
    )
    extra = called.cost.by_category["crews"] - plain.cost.by_category["crews"]
    assert best.spend == pytest.approx(extra, rel=1e-12)
    assert best.unplanned["mean_hours"] == pytest.approx(down.mean())
    # Called at the first job waiting, two pumps always have a crew each.
    exact = pair(repair_crews=1, call_out=contract).mean_unavailability()
    assert exact * YEAR == pytest.approx(YEAR / 4.0, rel=1e-9)


@pytest.mark.parametrize(
    "group",
    [
        {"standby": {"units": 2}},
        {"identical": {"units": 2, "k": 1}},
        {"load_sharing": {"units": 2, "k": 1, "load": 2.0}},
    ],
)
def test_a_group_of_units_is_one_node_to_the_levers(group):
    from repyability.tests import catalogue

    life = catalogue.aft() if "load_sharing" in group else E([0.01])
    spec = {"reliability": life, "repairability": E([0.02]), **group}
    rbd = RepairableRBD(ONE, {"p": spec}, repair_crews=1, crew_cost_rate=1.0)
    found = rbd.least_spend_for_downtime(
        YEAR,
        1.0,
        redundancy={"p": 1e5},
        crews={"max": 1},
        max_units={"p": 1},
        mc_samples=300,
        seed=52,
    )
    # A crew more (8,760 for the year) repairs both units at once; a copy
    # is a second group in parallel.
    assert found.frontier[0].crews == 1
    assert {d.crews for d in found.frontier} <= {1, 2}
    best = min(found.frontier, key=lambda d: d.estimate)
    assert best.copies == {"p": 1}
    copied = RepairableRBD(
        [("s", "p"), ("p", "t"), ("s", "p (copy 1)"), ("p (copy 1)", "t")],
        {"p": spec, "p (copy 1)": spec},
        repair_crews=best.crews,
        crew_cost_rate=1.0,
    )
    down, _ = direct(copied, 300, 52)
    assert best.unplanned["mean_hours"] == pytest.approx(down.mean())


def test_crew_and_call_out_refusals():
    with pytest.raises(ValueError, match="repair_crews=None"):
        pair().least_spend_for_downtime(YEAR, 1.0, crews={"price": 1.0})
    with pytest.raises(ValueError, match="must cost something"):
        pair(repair_crews=1).least_spend_for_downtime(YEAR, 1.0, crews={})
    with pytest.raises(ValueError, match="crews must be a dict"):
        pair(repair_crews=1).least_spend_for_downtime(
            YEAR, 1.0, crews={"cost": 1.0}
        )
    with pytest.raises(ValueError, match="must cost something"):
        pair(repair_crews=1).least_spend_for_downtime(
            YEAR, 1.0, call_out={"when_waiting": 1}
        )
    with pytest.raises(ValueError, match="repair_crews is None"):
        pair().least_spend_for_downtime(
            YEAR, 1.0, call_out={"when_waiting": 1, "fee": 1.0}
        )
    with pytest.raises(ValueError, match="list of them"):
        pair(repair_crews=1).least_spend_for_downtime(YEAR, 1.0, call_out=[])


def hidden(interval=2000.0):
    """A unit whose failures hide until a test finds them: a failure every
    1,000 h on average, 20 h repairs, a test costing 50."""
    return RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": E([FAIL]),
                "repairability": E([0.05]),
                "inspection": {"interval": interval, "cost": 50.0},
            }
        },
    )


def test_test_intervals_against_the_exact_downtime_and_their_count():
    found = hidden().least_spend_for_downtime(
        YEAR,
        300.0,
        inspections={"p": [1000.0, 500.0, 250.0, 100.0]},
        mc_samples=1000,
        seed=60,
    )
    # Even testing every 100 h leaves a bad year over 300 h down.
    assert not found.met and found.floor == "caps"
    chosen = [d.inspections["p"] for d in found.frontier]
    assert chosen == [None, 1000.0, 500.0, 250.0, 100.0]
    for design in found.frontier:
        interval = design.inspections["p"] or 2000.0
        down, _ = direct(hidden(interval), 1000, 60)
        assert design.unplanned["mean_hours"] == pytest.approx(down.mean())
        exact = hidden(interval).mission_unavailability(YEAR) * YEAR
        se = down.std() / math.sqrt(down.size)
        assert abs(down.mean() - exact) < 4 * se
    # The spend is the tests more, at 50 each: 4 a year every 2,000 h,
    # then 8, 17 and 35 (a test falling in a repair is not made, which
    # can happen once in a while at the shorter intervals).
    assert [d.spend for d in found.frontier[:3]] == [0.0, 200.0, 650.0]
    assert found.frontier[3].spend == pytest.approx(1550.0, abs=50.0)


def test_inspection_refusals():
    with pytest.raises(ValueError, match="no inspection"):
        pump().least_spend_for_downtime(YEAR, 10.0, inspections={"p": [100.0]})
    with pytest.raises(ValueError, match="test intervals to try"):
        hidden().least_spend_for_downtime(YEAR, 10.0, inspections={"p": []})
    offset = RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": E([FAIL]),
                "repairability": E([0.05]),
                "inspection": {"interval": 2000.0, "offset": 500.0},
            }
        },
    )
    with pytest.raises(ValueError, match="option 1 cannot be taken"):
        offset.least_spend_for_downtime(YEAR, 10.0, inspections={"p": [400.0]})


BETTER = 2.5e-4  # the better pump: 4,000 h between failures


def test_an_upgrade_against_the_better_pump_s_exact_percentile():
    exact = optimize.brentq(
        lambda x: downtime_cdf(x, BETTER, REPAIR, YEAR) - 0.95, 1.0, YEAR
    )
    density = (
        downtime_cdf(exact + 1.0, BETTER, REPAIR, YEAR)
        - downtime_cdf(exact - 1.0, BETTER, REPAIR, YEAR)
    ) / 2.0
    n = 4000
    found = pump().least_spend_for_downtime(
        YEAR,
        400.0,
        upgrades={"p": {"reliability": E([BETTER]), "cost": 3000.0}},
        mc_samples=n,
        seed=61,
    )
    assert found.met and found.design.upgrades == {"p": 1}
    assert found.design.spend == 3000.0
    se = math.sqrt(0.95 * 0.05 / n) / density
    assert abs(found.design.estimate - exact) < 4 * se
    better = RepairableRBD(
        ONE, {"p": {"reliability": E([BETTER]), "repairability": E([REPAIR])}}
    )
    down, _ = direct(better, n, 61)
    assert found.design.unplanned["mean_hours"] == pytest.approx(down.mean())


def test_an_upgrade_is_paid_for_every_unit_its_copies_too():
    found = pump().least_spend_for_downtime(
        YEAR,
        0.5,
        redundancy={"p": 5000.0},
        upgrades={"p": [{"reliability": E([BETTER]), "cost": 1000.0}]},
        mc_samples=500,
        seed=62,
    )
    assert found.met
    assert any(d.upgrades["p"] and d.copies["p"] for d in found.frontier)
    for design in found.frontier:
        units = 1 + design.copies["p"]
        paid = 1000.0 * units if design.upgrades["p"] else 0.0
        assert design.spend == 5000.0 * design.copies["p"] + paid


def test_upgrades_of_groups_and_refusals():
    from repyability.tests import catalogue

    block = RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": E([0.01]),
                "repairability": E([0.02]),
                "identical": {"units": 2, "k": 2},
            }
        },
    )
    found = block.least_spend_for_downtime(
        YEAR,
        1.0,
        upgrades={"p": {"reliability": E([0.001]), "cost": 10.0}},
        mc_samples=300,
        seed=63,
    )
    assert found.frontier[-1].upgrades == {"p": 1}
    shared = RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": catalogue.aft(),
                "repairability": E([0.5]),
                "load_sharing": {"units": 2, "load": 2.0},
            }
        },
    )
    # A load-sharing group's life must take the load: a plain life
    # cannot, and is refused before the search.
    with pytest.raises(ValueError, match="option 1 cannot be taken"):
        shared.least_spend_for_downtime(
            YEAR,
            1.0,
            upgrades={"p": {"reliability": E([0.001]), "cost": 10.0}},
        )
    with pytest.raises(ValueError, match="'reliability'"):
        pump().least_spend_for_downtime(
            YEAR, 1.0, upgrades={"p": {"reliability": E([BETTER])}}
        )
    with pytest.raises(ValueError, match="cost of an upgrade"):
        pump().least_spend_for_downtime(
            YEAR,
            1.0,
            upgrades={"p": {"reliability": E([BETTER]), "cost": 0.0}},
        )
    upgraded = RepairableRBD(
        ONE,
        {
            "p": {
                "reliability": E([FAIL]),
                "repairability": E([REPAIR]),
                "upgrade": {"at": 100.0, "reliability": E([BETTER])},
            }
        },
    )
    with pytest.raises(ValueError, match="already has an upgrade"):
        upgraded.least_spend_for_downtime(
            YEAR,
            1.0,
            upgrades={"p": {"reliability": E([BETTER]), "cost": 1.0}},
        )


def fitted_pump():
    """The pump, its life fitted to 30 failures (a rate of 1.2e-3 an
    hour), its repairs known."""
    x = np.random.default_rng(3).exponential(1000.0, 30)
    life = surv.Exponential.fit(x)
    return RepairableRBD(
        ONE, {"p": {"reliability": life, "repairability": E([REPAIR])}}
    )


def test_parameter_uncertainty_against_the_exact_mixture():
    from repyability.rbd import _downtime_risk

    rbd = fitted_pump()
    n, draws = 4000, 50
    found = rbd.least_spend_for_downtime(
        YEAR, 1e6, uncertainty=True, n_draws=draws, mc_samples=n, seed=70
    )
    # Over the draws, the downtime's distribution is the mixture of each
    # drawn rate's exact one: its 95th percentile is 914.9 h, against
    # 846 h (simulated) with the fitted rate taken as known.
    given = _downtime_risk._drawn(rbd, True, draws, n, 70)
    rates = [
        d._init_args["components"]["p"]["reliability"].params[0]
        for d in given.diagrams
    ]

    def mixture(x):
        return np.mean([downtime_cdf(x, r, REPAIR, YEAR) for r in rates])

    exact = optimize.brentq(lambda x: mixture(x) - 0.95, 1.0, YEAR)
    density = (mixture(exact + 1.0) - mixture(exact - 1.0)) / 2.0
    se = math.sqrt(0.95 * 0.05 / n) / density
    assert abs(found.design.estimate - exact) < 4 * se
    known = rbd.least_spend_for_downtime(YEAR, 1e6, mc_samples=n, seed=70)
    assert known.uncertainty is None
    assert found.design.estimate > known.design.estimate
    assert found.n_simulations == n
    assert "50 draws" in found.uncertainty
    assert "the repairability of 'p'" in found.uncertainty
    assert "approximate" in found.uncertainty


def test_parameter_uncertainty_is_seeded_and_shared_by_the_designs():
    def search():
        return fitted_pump().least_spend_for_downtime(
            YEAR,
            100.0,
            redundancy={"p": 5000.0},
            uncertainty=True,
            n_draws=20,
            mc_samples=400,
            seed=71,
        )

    one, again = search(), search()
    assert one.met and one.design.copies["p"] >= 1
    assert [d.estimate for d in one.frontier] == [
        d.estimate for d in again.frontier
    ]
    with pytest.raises(ValueError, match="at least n_draws"):
        fitted_pump().least_spend_for_downtime(
            YEAR, 10.0, uncertainty=True, n_draws=200, mc_samples=100
        )
    with pytest.raises(ValueError, match="surpyval fit"):
        pump().least_spend_for_downtime(
            YEAR, 10.0, uncertainty=True, mc_samples=100
        )
