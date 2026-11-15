"""A minimally repaired unit replaced at its N-th failure or at an age
(#402): its long-run values are renewal-reward ratios over one
replacement cycle, exact for a power law (Crow-AMSAA, Duane) and by
quadrature for Cox-Lewis's intensity; and ``Repairable``'s failure-limit
policy is exact for an analytic model.

The references are worked by hand from the cycle: with ``H(t) = (t /
alpha) ** beta`` the ``n``-th failure comes at ``alpha * S ** (1 /
beta)``, ``S ~ Gamma(n, 1)``.
"""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate, stats
from scipy.special import gammainc, gammaincc, gammaln

from repyability import Repairable, RepairableRBD
from repyability.rbd import _minimal_cycle
from repyability.rbd import routes as r

ALPHA, BETA = 100.0, 2.0
MTTR = 5.0
CROW = surv.CrowAMSAA.from_params([ALPHA, BETA])
REPAIR = surv.Exponential.from_params([1.0 / MTTR])
COX = surv.CoxLewis.from_params([math.log(0.001), 0.005])


def one(reliability=CROW, repairability=REPAIR, **spec):
    return RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": reliability,
                "repairability": repairability,
                **spec,
            }
        },
    )


def nth(n: int) -> float:
    """E[T_n] for the power law, by hand."""
    return ALPHA * math.exp(gammaln(n + 1.0 / BETA) - gammaln(n))


# -- Hand-derived renewal-reward values ------------------------------------


def test_replaced_at_its_third_failure():
    # The cycle runs to the third failure, E[T_3] = 100 Gamma(3.5) /
    # Gamma(3) = 166.17, with three repairs of mean 5 (the third replaces
    # the unit): each failure charged 2, the replacement 20 more.
    rbd = one(replace_after=3, repair_cost=2.0, replace_cost=20.0)
    length = nth(3) + 3 * MTTR
    assert nth(3) == pytest.approx(166.16754852239, rel=1e-12)
    assert rbd.mean_availability() == pytest.approx(nth(3) / length, 1e-12)
    assert rbd.system_failure_frequency() == pytest.approx(3 / length, 1e-12)
    assert rbd.expected_cost_rate() == pytest.approx(
        (3 * 2.0 + 20.0) / length, rel=1e-12
    )
    rates = rbd.node_action_rates()["a"]
    assert rates.corrective == pytest.approx(3 / length, rel=1e-12)


def test_replaced_at_an_age():
    # Replaced at the operating age 150: H(150) = 2.25 failures, each
    # repaired in 5 on average, then the replacement, 10 on average.
    duration = surv.Exponential.from_params([0.1])
    rbd = one(
        preventive={"interval": 150.0, "cost": 15.0, "duration": duration},
        repair_cost=2.0,
        replace_cost=20.0,
    )
    length = 150.0 + 2.25 * MTTR + 10.0
    assert rbd.mean_availability() == pytest.approx(150.0 / length, 1e-12)
    assert rbd.system_failure_frequency() == pytest.approx(
        2.25 / length, 1e-12
    )
    # No failure replaces it: the replace cost is never charged.
    assert rbd.expected_cost_rate() == pytest.approx(
        (2.25 * 2.0 + 15.0) / length, rel=1e-12
    )
    assert rbd.node_action_rates()["a"].preventive == pytest.approx(
        1.0 / length, rel=1e-12
    )


def test_replaced_at_its_third_failure_or_an_age():
    # Whichever first: the third failure before 150 with probability
    # P(3, 2.25); the operating time by parts, T Q(3, H) + E[T_3]
    # P(3.5, H); the failures E[min(N(150), 3)] = H Q(2, H) + 3 P(3, H).
    h = 2.25
    replaced = gammainc(3, h)
    up = 150.0 * gammaincc(3, h) + nth(3) * gammainc(3.5, h)
    failures = h * gammaincc(2, h) + 3 * replaced
    # The same failures counted from the Poisson probabilities.
    p = stats.poisson(h).pmf(np.arange(3))
    assert failures == pytest.approx(p[1] + 2 * p[2] + 3 * (1 - p.sum()))
    rbd = one(
        replace_after=3,
        preventive={"interval": 150.0, "cost": 15.0},
        repair_cost=2.0,
        replace_cost=20.0,
    )
    length = up + failures * MTTR
    assert rbd.mean_availability() == pytest.approx(up / length, rel=1e-12)
    assert rbd.expected_cost_rate() == pytest.approx(
        (2.0 * failures + 20.0 * replaced + 15.0 * (1 - replaced)) / length,
        rel=1e-12,
    )


@pytest.mark.parametrize("spec", [{"replace_after": 4}, {"age": 120.0}])
def test_repairs_in_no_time_match_repairable(spec):
    # Repaired in no time it is never down; its cost rate is Repairable's:
    # the failure-limit policy's at N (repair 1, the N-th failure's
    # replacement 6 in all) and the overhaul policy's at the age.
    unit = Repairable(CROW)
    unit.set_repair_and_overhaul_costs(cr=1.0, co=6.0)
    if "replace_after" in spec:
        rbd = one(
            repairability="instant",
            replace_after=spec["replace_after"],
            repair_cost=1.0,
            replace_cost=5.0,
        )
        n = spec["replace_after"]
        expected = (1.0 * (n - 1) + 6.0) / unit.expected_time_to_nth_failure(n)
    else:
        rbd = one(
            repairability="instant",
            preventive={"interval": spec["age"], "cost": 6.0},
            repair_cost=1.0,
        )
        expected = unit.cost_rate(spec["age"])
    assert rbd.mean_availability() == 1.0
    assert rbd.expected_cost_rate() == pytest.approx(expected, rel=1e-12)


def test_the_best_overhaul_interval_is_repairable_s():
    # At Repairable's optimal overhaul age, the diagram's cost rate is
    # Repairable's optimum (Lambda(T*) = co / (cr (beta - 1)) = 6).
    unit = Repairable(CROW)
    unit.set_repair_and_overhaul_costs(cr=1.0, co=6.0)
    policy = unit.optimal_overhaul_policy()
    assert policy.interval == pytest.approx(ALPHA * math.sqrt(6.0), 1e-6)
    rbd = one(
        repairability="instant",
        preventive={"interval": policy.interval, "cost": 6.0},
        repair_cost=1.0,
    )
    assert rbd.expected_cost_rate() == pytest.approx(policy.cost_rate, 1e-9)


def test_duane_is_the_same_power_law():
    # Duane's b t ** a with b = 1e-4, a = 2 is (t / 100) ** 2.
    duane = surv.Duane.from_params([2.0, 1e-4])
    a = one(reliability=duane, replace_after=3).mean_availability()
    b = one(replace_after=3).mean_availability()
    assert a == pytest.approx(b, rel=1e-12)


# -- The cycle's quadrature against the closed form ------------------------


@pytest.mark.parametrize(
    "n, age", [(1, math.inf), (3, math.inf), (7, math.inf), (3, 150.0)]
)
def test_the_quadrature_agrees_with_the_closed_form(n, age):
    def hazard(t):
        return (t / ALPHA) ** BETA

    exact = _minimal_cycle.cycle(hazard, n, age, (ALPHA, BETA))
    numerical = _minimal_cycle.cycle(hazard, n, age)
    assert exact.route == "exact" and numerical.route == "numerical"
    assert numerical.up == pytest.approx(exact.up, rel=1e-10)
    assert numerical.failures == exact.failures


def cox_lewis_nth(n: int, alpha: float, beta: float) -> float:
    """E[T_n] for Cox-Lewis by its inverse: T_n = H^-1(S), S ~ Gamma(n),
    H^-1(u) = log(1 + beta u exp(-alpha)) / beta."""

    def integrand(u):
        return (
            math.log1p(beta * u * math.exp(-alpha))
            / beta
            * stats.gamma(n).pdf(u)
        )

    return integrate.quad(integrand, 0.0, np.inf, epsrel=1e-12)[0]


@pytest.mark.parametrize("n", [1, 2, 5])
def test_cox_lewis_time_to_failure_by_its_inverse(n):
    alpha, beta = (float(p) for p in COX.params)
    unit = Repairable(COX)
    assert unit.expected_time_to_nth_failure(n) == pytest.approx(
        cox_lewis_nth(n, alpha, beta), rel=1e-9
    )


# -- Best failure count -----------------------------------------------------


@pytest.mark.parametrize("beta, cr, co", [(2.0, 1.0, 6.5), (2.5, 1.0, 8.2)])
def test_the_best_failure_count_by_brute_force(beta, cr, co):
    model = surv.CrowAMSAA.from_params([ALPHA, beta])
    unit = Repairable(model)
    unit.set_repair_and_overhaul_costs(cr=cr, co=co)
    policy = unit.optimal_failure_limit_policy(max_failures=60)
    rates = [
        (cr * (n - 1) + co)
        / (ALPHA * math.exp(gammaln(n + 1.0 / beta) - gammaln(n)))
        for n in range(1, 61)
    ]
    best = int(np.argmin(rates)) + 1
    assert policy.failure_count == best
    # The rate falls while n < (co - cr) / (cr (beta - 1)), then rises.
    assert best == max(1, math.ceil((co - cr) / (cr * (beta - 1.0))))
    assert policy.cost_rate == pytest.approx(rates[best - 1], rel=1e-12)


def test_the_best_failure_count_for_cox_lewis_by_brute_force():
    alpha, beta = (float(p) for p in COX.params)
    unit = Repairable(COX)
    unit.set_repair_and_overhaul_costs(cr=1.0, co=6.0)
    policy = unit.optimal_failure_limit_policy(max_failures=15)
    rates = [
        (n - 1 + 6.0) / cox_lewis_nth(n, alpha, beta) for n in range(1, 16)
    ]
    assert policy.failure_count == int(np.argmin(rates)) + 1
    assert policy.cost_rate == pytest.approx(min(rates), rel=1e-9)


def test_a_falling_intensity_has_no_failure_limit_policy():
    unit = Repairable(surv.CoxLewis.from_params([-3.0, -0.05]))
    unit.set_repair_and_overhaul_costs(cr=1.0, co=6.0)
    with pytest.raises(ValueError, match="never reach its failure 1"):
        unit.optimal_failure_limit_policy()


# -- In a diagram -----------------------------------------------------------


def test_combined_by_the_structure_function():
    other = {
        "reliability": surv.Exponential.from_params([0.01]),
        "repairability": REPAIR,
    }
    minimal = {
        "reliability": COX,
        "repairability": REPAIR,
        "replace_after": 2,
        "preventive": {"interval": 400.0},
    }
    a = one(**{k: v for k, v in minimal.items()}).mean_availability()
    b = 100.0 / 105.0
    series = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")], {"a": minimal, "b": other}
    )
    parallel = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": minimal, "b": other},
    )
    assert series.mean_availability() == pytest.approx(a * b, rel=1e-12)
    assert parallel.mean_availability() == pytest.approx(
        1 - (1 - a) * (1 - b), rel=1e-12
    )
    report = parallel.analysis_routes()
    assert report["mean_availability"].route == r.NUMERICAL
    assert "quadrature" in report["mean_availability"].reason


WINDOW = 40000.0


@pytest.mark.parametrize(
    "spec",
    [
        dict(replace_after=3, repair_cost=2.0, replace_cost=20.0),
        dict(
            reliability=COX,
            replace_after=2,
            preventive={
                "interval": 300.0,
                "cost": 5.0,
                "duration": surv.Exponential.from_params([0.1]),
            },
            repair_cost=1.0,
            replace_cost=10.0,
        ),
    ],
    ids=["Crow-AMSAA, third failure", "Cox-Lewis, second failure or age"],
)
def test_the_simulation_agrees(spec):
    rbd = one(**spec)
    plain = dict(
        mc_samples=200, seed=11, control_variate=False, conditional=False
    )
    run = rbd.availability(WINDOW, **plain).mean_availability_interval()
    assert abs(run.estimate - rbd.mean_availability()) <= (
        4 * run.standard_error
    )
    cost = rbd.cost(WINDOW, **plain).mean_interval()
    assert abs(cost.estimate / WINDOW - rbd.expected_cost_rate()) <= (
        4 * cost.standard_error / WINDOW
    )


# -- Routes -----------------------------------------------------------------


def test_routes():
    def route(**spec):
        return one(**spec).analysis_routes()["mean_availability"]

    assert route(replace_after=3).route == r.EXACT
    assert route(preventive={"interval": 150.0}).route == r.EXACT
    # Age replacement alone has H(T) failures for any intensity.
    assert route(reliability=COX, preventive={"interval": 150.0}).route == (
        r.EXACT
    )
    assert route(reliability=COX, replace_after=2).route == r.NUMERICAL
    # Never replaced: no cycle; repaired imperfectly: no Poisson process.
    assert route().route == r.REFUSED
    gr = {"model": "kijima1", "q": 0.5}
    assert (
        route(
            reliability=surv.Weibull.from_params([100.0, 2.0]),
            repair=gr,
            replace_after=3,
        ).route
        == r.REFUSED
    )
    falling = route(
        reliability=surv.CoxLewis.from_params([-3.0, -0.05]), replace_after=2
    )
    assert falling.route == r.REFUSED
    assert "never reach its failure 2" in falling.reason
    with pytest.raises(NotImplementedError, match="never reach"):
        one(
            reliability=surv.CoxLewis.from_params([-3.0, -0.05]),
            replace_after=2,
        ).mean_availability()


def test_the_interval_search_finds_repairable_s_overhaul():
    # optimal_replacement_intervals now takes the unit: its best age is
    # Repairable's optimal overhaul interval, at the same cost rate.
    unit = Repairable(CROW)
    unit.set_repair_and_overhaul_costs(cr=1.0, co=6.0)
    policy = unit.optimal_overhaul_policy()
    rbd = one(
        repairability="instant",
        preventive={"interval": 100.0, "cost": 6.0},
        repair_cost=1.0,
    )
    plan = rbd.optimal_replacement_intervals()
    assert plan.intervals["a"] == pytest.approx(policy.interval, rel=1e-4)
    assert plan.cost_rate == pytest.approx(policy.cost_rate, rel=1e-9)
