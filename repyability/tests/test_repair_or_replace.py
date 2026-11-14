"""Repair again or replace, from a joint frailty model (UC-27):
``RepairOrReplace`` checked against closed forms, the model's own
population functions, simulation, and the costs' directions."""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy.integrate import quad
from scipy.special import gamma as gamma_fn
from scipy.stats import poisson
from surpyval import JointFrailty

from repyability import Repairable, RepairOrReplace
from repyability.maintenance._repair_or_replace import _frailty_nodes

CR, CP, CF, CW = 1.0, 10.0, 2.0, 17.0


def fleet(model, **costs):
    given = dict(
        repair_cost=CR,
        replacement_cost=CP,
        failure_cost=CF,
        write_off_cost=CW,
    )
    given.update(costs)
    return RepairOrReplace(model, **given)


@pytest.fixture(scope="module")
def fitted():
    truth = JointFrailty.from_params((100.0, 2.0), (400.0, 2.0), 0.5, 1.0)
    rows = truth.random(150, end=300.0, random_state=2)
    return JointFrailty.fit(**rows)


# -- no frailty variance: minimal repair and an independent write-off ------


class _Plain:
    """A unit repaired minimally (Weibull intensity) and written off at an
    independent Weibull hazard, priced by direct quadrature."""

    def __init__(self, rec, term, r=0.0):
        (self.rs, self.rk), (self.ts, self.tk) = rec, term
        self.r = r

    def lam(self, t):
        return self.rk / self.rs * (t / self.rs) ** (self.rk - 1)

    def Lam(self, t):
        return (t / self.rs) ** self.rk

    def h(self, t):
        return self.tk / self.ts * (t / self.ts) ** (self.tk - 1)

    def S(self, t):
        return math.exp(-((t / self.ts) ** self.tk) - self.r * t)

    def _integral(self, f, upper):
        return quad(f, 0, upper, limit=400, epsabs=0, epsrel=1e-12)[0]

    def age(self, T):
        cost = self._integral(
            lambda t: ((CR + CF) * self.lam(t) + CW * self.h(t)) * self.S(t),
            T,
        )
        return (cost + CP * self.S(T)) / self._integral(self.S, T)

    def keep(self):
        cost = self._integral(
            lambda t: ((CR + CF) * self.lam(t) + CW * self.h(t)) * self.S(t),
            np.inf,
        )
        return cost / self._integral(self.S, np.inf)

    def count(self, k):
        def below(j, t):
            return poisson.cdf(j, self.Lam(t)) if j >= 0 else 0.0

        def cost(t):
            return self.S(t) * (
                self.lam(t)
                * (
                    (CR + CF) * below(k - 2, t)
                    + (CF + CP) * poisson.pmf(k - 1, self.Lam(t))
                )
                + CW * self.h(t) * below(k - 1, t)
            )

        time = self._integral(lambda t: self.S(t) * below(k - 1, t), np.inf)
        return self._integral(cost, np.inf) / time


@pytest.mark.parametrize("r", [0.0, 0.002])
def test_no_frailty_variance_is_minimal_repair_and_an_independent_write_off(
    r,
):
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.0, 1.0)
    plain = _Plain((100.0, 2.0), (400.0, 1.5), r)
    engine = fleet(model, discount_rate=r)._engine
    for T in (20.0, 150.0, 600.0):
        assert engine.age_rate(T) == pytest.approx(plain.age(T), rel=1e-8)
    for k in (1, 2, 5, 12):
        assert engine.count_rates(k)[k - 1] == pytest.approx(
            plain.count(k), rel=1e-8
        )
    assert engine.keep_rate() == pytest.approx(plain.keep(), rel=1e-8)


def test_keep_repairing_in_closed_form_for_shapes_alike():
    # Minimal repair at (t / a)^k and a write-off at (t / b)^k: the
    # expected repairs before write-off are (b / a)^k, the expected life
    # b Gamma(1 + 1 / k).
    a, b, k = 120.0, 500.0, 1.7
    model = JointFrailty.from_params((a, k), (b, k), 0.0, 1.0)
    expected = ((CR + CF) * (b / a) ** k + CW) / (b * gamma_fn(1 + 1 / k))
    policy = fleet(model).fleet_policy()
    assert policy.keep_repairing_cost_rate == pytest.approx(expected, rel=1e-9)


def test_constant_rates_never_pay_to_replace_at_an_age():
    # Exponential failures and write-off: the unit never ages, so the
    # cost rate is (cr + cf) / a + cw / b, discounted or not, and no age
    # replacement pays.
    model = JointFrailty.from_params((50.0, 1.0), (300.0, 1.0), 0.0, 1.0)
    for r in (0.0, 0.01):
        policy = fleet(model, discount_rate=r).fleet_policy()
        assert policy.keep_repairing_cost_rate == pytest.approx(
            (CR + CF) / 50.0 + CW / 300.0, rel=1e-9
        )
        assert policy.best_age == math.inf


def test_a_small_variance_tends_to_none():
    none = fleet(
        JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.0, 1.0)
    ).fleet_policy()
    small = fleet(
        JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 1e-7, 1.0)
    ).fleet_policy()
    for name in (
        "keep_repairing_cost_rate",
        "age_cost_rate",
        "failure_count_cost_rate",
    ):
        assert small[name] == pytest.approx(none[name], rel=1e-5)
    assert small.best_failure_count == none.best_failure_count


def test_a_discount_rate_near_0_is_none():
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    plain = fleet(model).fleet_policy()
    near = fleet(model, discount_rate=1e-10).fleet_policy()
    assert near.cost_rate == pytest.approx(plain.cost_rate, rel=1e-6)
    assert near.best_age == pytest.approx(plain.best_age, rel=1e-4)


# -- with a frailty variance -------------------------------------------------


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_the_frailty_nodes_give_the_models_population_values(family):
    model = JointFrailty.from_params(
        (100.0, 2.0), (400.0, 1.5), 0.7, 1.3, family=family
    )
    u, w = _frailty_nodes(model.theta, family)
    for t in (10.0, 200.0, 800.0):
        survival = w @ np.asarray(model.sf(np.full(u.size, t), frailty=u))
        assert survival == pytest.approx(float(model.sf(t)), rel=1e-7)


@pytest.mark.parametrize("family", ["gamma", "lognormal"])
def test_an_age_rule_against_the_models_population_functions(family):
    # Replacing at age T, undiscounted: the expected repairs are the
    # model's mcf(T) (failures stopped by the write-off), the chance of a
    # write-off its ff(T), the expected time in service the integral of
    # its sf.
    model = JointFrailty.from_params(
        (100.0, 2.0), (400.0, 1.5), 0.5, 1.0, family=family
    )
    engine = fleet(model)._engine
    for T in (50.0, 200.0, 500.0):
        cost = (
            (CR + CF) * float(model.mcf(T))
            + CW * float(model.ff(T))
            + CP * float(model.sf(T))
        )
        time = quad(lambda t: float(model.sf(t)), 0, T, epsrel=1e-12)[0]
        assert engine.age_rate(T) == pytest.approx(cost / time, rel=1e-6)


def test_an_age_rule_against_the_models_simulation():
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    T, n = 250.0, 40000
    rows = model.random(n, end=T, random_state=7)
    ends = rows["c"] == 1
    failures = np.bincount(rows["i"][~ends], minlength=n)
    stop = rows["x"][ends]
    written_off = rows["terminal"][ends] == 1
    cost = (CR + CF) * failures + np.where(written_off, CW, CP)
    estimate = cost.mean() / stop.mean()
    # The ratio's standard error, by the delta method.
    z = cost - estimate * stop
    error = z.std() / math.sqrt(n) / stop.mean()
    assert fleet(model)._engine.age_rate(T) == pytest.approx(
        estimate, abs=4 * error
    )


def _simulate_counts(k, n, seed):
    """Replace at the k-th failure, by a simulation written from the
    model's definition: a gamma frailty of variance 0.5, failures at the
    rate u (t / 100)^2, write-off at the hazard u (t / 400)^1.5."""
    rng = np.random.default_rng(seed)
    u = rng.gamma(2.0, 0.5, n)
    tau = 400.0 * (rng.exponential(size=n) / u) ** (1 / 1.5)
    cum = np.cumsum(rng.exponential(size=(n, k)), axis=1) / u[:, None]
    times = 100.0 * np.sqrt(cum)
    kth = times[:, -1]
    replaced = kth < tau
    length = np.where(replaced, kth, tau)
    failures = (times < length[:, None]).sum(axis=1)
    failures = np.where(replaced, k, failures)
    repairs = np.where(replaced, k - 1, failures)
    cost = CR * repairs + CF * failures + np.where(replaced, CP, CW)
    return cost, length


def test_a_count_rule_against_simulation():
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    engine = fleet(model)._engine
    for k in (1, 3, 6):
        cost, length = _simulate_counts(k, 200000, seed=k)
        estimate = cost.mean() / length.mean()
        z = cost - estimate * length
        error = z.std() / math.sqrt(len(z)) / length.mean()
        assert engine.count_rates(k)[k - 1] == pytest.approx(
            estimate, abs=4 * error
        )


def test_counting_failures_finds_the_lemons():
    # With units that differ, replacing at a count of failures beats every
    # age; the more they differ, the more it saves.
    rates = []
    for theta in (0.2, 1.0):
        model = JointFrailty.from_params(
            (100.0, 2.0), (400.0, 1.5), theta, 1.0
        )
        policy = fleet(model).fleet_policy()
        assert policy.recommended == "replace at failure"
        rates.append(policy.age_cost_rate / policy.failure_count_cost_rate)
    assert rates[1] > rates[0] > 1


def test_an_infinite_expected_life_has_no_keep_repairing_rate():
    # A gamma frailty of variance 2 against a write-off of shape 0.4
    # (alpha = 1): E[u ** -2.5] is infinite, so is a new unit's expected
    # life; with a discount rate it is finite.
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 0.4), 2.0, 1.0)
    policy = fleet(model).fleet_policy()
    assert math.isnan(policy.keep_repairing_cost_rate)
    assert "give a discount rate" in policy.notes[0]
    assert math.isfinite(policy.cost_rate)
    discounted = fleet(model, discount_rate=0.001).fleet_policy()
    assert math.isfinite(discounted.keep_repairing_cost_rate)


# -- one unit ----------------------------------------------------------------


def test_a_new_unit_like_the_fleet_saves_the_replacement():
    # With no variance a unit is the fleet: at age 0 its best rule is the
    # fleet's, at the fleet's cost rate, so it saves just the replacement
    # it does not buy.
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.0, 1.0)
    rr = fleet(model)
    policy = rr.fleet_policy()
    decision = rr.unit_decision(0.0, frailty=1.0)
    assert decision.recommended == policy.recommended
    assert decision.best_failure_count == policy.best_failure_count
    assert decision.saving == pytest.approx(CP, rel=1e-6)
    assert decision.failure_count_cost_rate == pytest.approx(
        policy.failure_count_cost_rate, rel=1e-8
    )


def test_a_unit_at_an_age_against_quadrature():
    # Repair until written off, from age a, for a unit of frailty u: its
    # expected costs over its expected remaining life.
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    rr = fleet(model)
    a, u = 120.0, 1.7

    def S(t):
        return math.exp(-u * (((t / 400.0) ** 1.5) - (a / 400.0) ** 1.5))

    def spend(t):
        lam = u * 2.0 * t / 100.0**2
        h = u * 1.5 / 400.0 * (t / 400.0) ** 0.5
        return ((CR + CF) * lam + CW * h) * S(t)

    cost = quad(spend, a, np.inf, epsrel=1e-12, limit=400)[0]
    time = quad(S, a, np.inf, epsrel=1e-12, limit=400)[0]
    decision = rr.unit_decision(a, frailty=u, uncertainty=False)
    assert decision.keep_cost_rate == pytest.approx(cost / time, rel=1e-7)
    g = decision.fleet_cost_rate
    assert decision.keep_saving == pytest.approx(-(cost - g * time - CP))


def test_lemons_are_replaced_and_sound_units_kept():
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 2.0), 0.5, 1.0)
    rr = fleet(model)
    edge = rr.break_even_frailty(150.0, just_failed=True)
    assert 0 < edge < math.inf
    above = rr.unit_decision(
        150.0, frailty=1.05 * edge, just_failed=True, uncertainty=False
    )
    below = rr.unit_decision(
        150.0, frailty=0.95 * edge, just_failed=True, uncertainty=False
    )
    assert above.recommended == "replace now"
    assert above.saving == 0.0
    assert below.recommended != "replace now"
    assert below.saving > 0
    assert below.break_even_frailty == pytest.approx(edge, rel=1e-6)


def test_just_failed_costs_the_repair():
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 2.0), 0.5, 1.0)
    rr = fleet(model)
    later = rr.unit_decision(150.0, frailty=0.5, uncertainty=False)
    now = rr.unit_decision(
        150.0, frailty=0.5, just_failed=True, uncertainty=False
    )
    for name in ("keep_saving", "failure_count_saving"):
        assert now[name] == pytest.approx(later[name] - CR)
    # Units wear out: the older, the less frail a unit must be to be kept.
    assert rr.break_even_frailty(300.0) < rr.break_even_frailty(150.0)


def test_a_fitted_units_frailty_is_its_estimate(fitted):
    rr = fleet(fitted)
    decision = rr.unit_decision(100.0, unit=3, uncertainty=False)
    assert decision.frailty == fitted.frailties[3]
    assert decision.unit == 3
    decisions = rr.unit_decisions({3: 100.0, 5: 50.0}, uncertainty=False)
    assert list(decisions) == [3, 5]
    assert decisions[3] == decision


# -- the costs' directions ----------------------------------------------------


@pytest.mark.parametrize(
    "cost",
    ["repair_cost", "replacement_cost", "failure_cost", "write_off_cost"],
)
def test_every_cost_rate_rises_with_each_cost(cost):
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    low = fleet(model).fleet_policy()
    base = {
        "repair_cost": CR,
        "replacement_cost": CP,
        "failure_cost": CF,
        "write_off_cost": CW,
    }
    high = fleet(model, **{cost: 1.5 * base[cost]}).fleet_policy()
    assert high.cost_rate > low.cost_rate
    assert high.failure_count_cost_rate > low.failure_count_cost_rate
    if cost in ("repair_cost", "failure_cost", "write_off_cost"):
        assert high.keep_repairing_cost_rate > low.keep_repairing_cost_rate
    else:
        assert high.keep_repairing_cost_rate == low.keep_repairing_cost_rate


def test_dearer_repairs_replace_sooner_and_dearer_replacements_later():
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    base = fleet(model)
    dear_repairs = fleet(model, repair_cost=3 * CR)
    dear_new = fleet(model, replacement_cost=2 * CP, write_off_cost=2 * CW)
    k = base.fleet_policy().best_failure_count
    assert dear_repairs.fleet_policy().best_failure_count <= k
    assert dear_new.fleet_policy().best_failure_count >= k


def test_the_break_even_frailty_falls_with_age():
    # Units wear out: the older, the less frail a unit must be to be
    # replaced now; one that has just failed, its repair to pay, sooner.
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    rr = fleet(model)
    edges = [rr.break_even_frailty(a) for a in (50.0, 150.0, 300.0, 600.0)]
    failed = [
        rr.break_even_frailty(a, just_failed=True)
        for a in (50.0, 150.0, 300.0, 600.0)
    ]
    assert np.all(np.diff(edges) < 0)
    assert np.all(np.diff(failed) < 0)
    assert np.all(np.array(failed) < np.array(edges))


# -- bands -------------------------------------------------------------------


def test_bands_are_seeded_and_hold_the_estimate(fitted):
    rr = fleet(fitted)
    one = rr.fleet_policy(mc_samples=60, seed=4)
    again = fleet(fitted).fleet_policy(mc_samples=60, seed=4)
    other = fleet(fitted).fleet_policy(mc_samples=60, seed=5)
    assert one.bands == again.bands
    assert one.bands != other.bands
    assert one.bounds.startswith("90% two-sided confidence")
    for name, value in (
        ("keep repairing", one.keep_repairing_cost_rate),
        ("replace at failure", one.failure_count_cost_rate),
    ):
        lower, upper = one.bands[name]
        assert lower < value < upper
    unit = rr.unit_decision(150.0, unit=3, mc_samples=60, seed=4)
    assert unit.bands["fleet_cost_rate"] == one.bands[one.recommended]
    lower, upper = unit.bands["keep_cost_rate"]
    assert lower < unit.keep_cost_rate < upper
    assert "frailty held at its estimate" in unit.bounds


def test_a_model_built_from_parameters_has_no_bands():
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    rr = fleet(model)
    policy = rr.fleet_policy()
    assert policy.bands is None
    assert policy.bounds.startswith("none: the model carries no covariance")
    assert rr.fleet_policy(uncertainty=False).bounds.startswith("none")


def test_the_results_go_through_json(fitted):
    rr = fleet(fitted)
    for result in (
        rr.fleet_policy(mc_samples=20, seed=1),
        rr.unit_decision(80.0, unit=0, mc_samples=20, seed=1),
    ):
        text = json.dumps(result.to_dict())
        assert json.loads(text)["recommended"] == result.recommended


# -- units and covariates ----------------------------------------------------


def test_the_answers_scale_with_the_unit_of_time():
    first = fleet(
        JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    )
    p1 = first.fleet_policy()
    for c in (1e-6, 1e6):
        rr = fleet(
            JointFrailty.from_params(
                (100.0 * c, 2.0), (400.0 * c, 1.5), 0.5, 1.0
            )
        )
        pc = rr.fleet_policy()
        assert pc.cost_rate * c == pytest.approx(p1.cost_rate, rel=1e-7)
        assert pc.best_age / c == pytest.approx(p1.best_age, rel=1e-6)
        assert pc.best_failure_count == p1.best_failure_count
        assert rr.break_even_frailty(150.0 * c) == pytest.approx(
            first.break_even_frailty(150.0), rel=1e-6
        )


def test_covariates_scale_the_rates():
    # A covariate multiplies the failure rate by e^beta: the same as a
    # baseline scale of 100 e^(-beta / 2) for a shape of 2.
    beta, gamma = 0.4, -0.3
    with_z = JointFrailty.from_params(
        (100.0, 2.0), (400.0, 1.5), 0.5, 1.0, beta=[beta], gamma=[gamma]
    )
    plain = JointFrailty.from_params(
        (100.0 * math.exp(-beta / 2), 2.0),
        (400.0 * math.exp(-gamma / 1.5), 1.5),
        0.5,
        1.0,
    )
    a = fleet(with_z, Z=[1.0]).fleet_policy()
    b = fleet(plain).fleet_policy()
    assert a.cost_rate == pytest.approx(b.cost_rate, rel=1e-9)
    assert a.best_failure_count == b.best_failure_count


# -- refusals ----------------------------------------------------------------


def test_refusals(fitted):
    plain = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    with pytest.raises(TypeError, match="JointFrailtyModel"):
        fleet(surv.Weibull.from_params([10.0, 2.0]))
    with pytest.raises(TypeError):
        fleet(JointFrailty)
    gap = JointFrailty.from_params(
        (100.0, 2.0), (400.0, 1.5), 0.5, 1.0, time_scale="gap"
    )
    with pytest.raises(NotImplementedError, match="gap time scale"):
        fleet(gap)
    covariates = JointFrailty.from_params(
        (100.0, 2.0), (400.0, 1.5), 0.5, 1.0, beta=[0.1], gamma=[0.1]
    )
    with pytest.raises(ValueError, match="give the units' covariates"):
        fleet(covariates)
    with pytest.raises(ValueError, match="one per covariate"):
        fleet(covariates, Z=[1.0, 2.0])
    with pytest.raises(ValueError, match="replacement_cost"):
        fleet(plain, replacement_cost=0.0)
    with pytest.raises(ValueError, match="repair_cost"):
        fleet(plain, repair_cost=-1.0)
    with pytest.raises(ValueError, match="discount_rate"):
        fleet(plain, discount_rate=math.inf)
    with pytest.raises(ValueError, match="repair_cost"):
        fleet(plain, repair_cost="1")
    rr = fleet(plain)
    with pytest.raises(ValueError, match="age must be"):
        rr.unit_decision(-1.0, frailty=1.0)
    with pytest.raises(ValueError, match="one of them"):
        rr.unit_decision(1.0)
    with pytest.raises(ValueError, match="one of them"):
        rr.unit_decision(1.0, unit=1, frailty=1.0)
    with pytest.raises(ValueError, match="no fitted units' frailties"):
        rr.unit_decision(1.0, unit=1)
    with pytest.raises(ValueError, match="frailty must be"):
        rr.unit_decision(1.0, frailty=0.0)
    with pytest.raises(ValueError, match="has no unit"):
        fleet(fitted).unit_decision(1.0, unit="no such unit")
    with pytest.raises(ValueError, match="max_failures"):
        rr.fleet_policy(max_failures=0)
    with pytest.raises(ValueError, match="mc_samples"):
        rr.fleet_policy(mc_samples=0)
    with pytest.raises(TypeError, match="seed"):
        rr.fleet_policy(seed=np.random.default_rng(1))
    with pytest.raises(TypeError, match="mapping"):
        rr.unit_decisions([1.0])


def test_repairable_points_a_joint_frailty_model_here():
    model = JointFrailty.from_params((100.0, 2.0), (400.0, 1.5), 0.5, 1.0)
    with pytest.raises(ValueError, match="RepairOrReplace"):
        Repairable(model)
