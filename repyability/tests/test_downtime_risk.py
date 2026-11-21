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
