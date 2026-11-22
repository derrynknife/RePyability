"""Contract payoffs in the simulated NPV (``npv_distribution(contract=)``)
and two plans compared on the same random numbers (``compare_npv``):
the payoff by hand, a system that never fails in closed form, identical
plans differing by exactly 0, and the variance the common numbers
remove."""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import AvailabilityContract, NPVComparison, RepairableRBD
from repyability.tests.catalogue import systems_of_every_kind

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LINE = [("s", "p"), ("p", "t")]


def pump(mttr=10.0, life=E([1e-3])):
    return RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": life,
                "repairability": E([1 / mttr]),
                "repair_cost": 500.0,
                "acquisition_cost": 2000.0,
            }
        },
    )


def terms(**kwargs):
    given = dict(target=0.99, penalty_per_point=1000.0)
    given.update(kwargs)
    return AvailabilityContract(730.0, **given)


def test_a_payoff_by_points():
    contract = AvailabilityContract(
        10.0, target=0.9, penalty_per_point=50.0, bonus_per_point=20.0, cap=300
    )
    paid = contract.payoff([0.9, 0.95, 0.85, 0.0, 1.0])
    assert paid.tolist() == pytest.approx([0.0, 100.0, -250.0, -300.0, 200.0])


def test_a_target_of_0_pays_nothing():
    rbd = pump()
    kwargs = dict(discount_rate=1e-5, value_rate=10.0, mc_samples=300, seed=4)
    plain = rbd.npv_distribution(8760.0, **kwargs)
    runs = rbd.npv_distribution(8760.0, contract=terms(target=0.0), **kwargs)
    assert np.all(runs.contract_samples == 0.0)
    assert runs.contract_payoff == 0.0
    assert np.array_equal(runs.samples, plain.samples)
    assert plain.contract is None and plain.contract_samples is None


def test_a_system_that_never_fails_is_paid_in_closed_form():
    # A life of mean 1e15 hours: no run fails within the horizon.
    rbd = pump(life=E([1e-15]))
    rate, horizon = 1e-4, 8760.0
    contract = AvailabilityContract(
        730.0,
        target=0.95,
        penalty_per_point=1000.0,
        bonus_per_point=400.0,
        cap=1500.0,
    )
    runs = rbd.npv_distribution(
        horizon, discount_rate=rate, mc_samples=200, seed=1, contract=contract
    )
    ends = 730.0 * np.arange(1, 13)
    assert runs.period_ends == pytest.approx(ends.tolist())
    assert np.all(runs.period_availability == 1.0)
    # 5 points over at 400 a point is 2,000, capped at 1,500 a month.
    each = 1500.0 * np.exp(-rate * ends).sum()
    assert runs.contract_samples == pytest.approx(np.full(200, each))
    assert runs.undiscounted_contract_payoff == pytest.approx(1500.0 * 12)
    assert runs.mean == pytest.approx(each - 2000.0)


def test_each_run_is_paid_on_its_own_availability():
    rbd = pump(mttr=50.0)
    rate, horizon, n = 2e-5, 8760.0, 300
    contract = terms(bonus_per_point=100.0)
    runs = rbd.npv_distribution(
        horizon, discount_rate=rate, mc_samples=n, seed=7, contract=contract
    )
    ends = np.array(runs.period_ends)
    paid = contract.payoff(runs.period_availability)
    assert runs.contract_samples == pytest.approx(
        paid @ np.exp(-rate * ends), rel=1e-12
    )
    assert np.allclose(
        runs.samples,
        runs.value_samples - runs.cost_samples + runs.contract_samples,
        rtol=1e-12,
    )
    # The periods cover the year: their uptime is each run's.
    up = rbd.simulate_timelines(horizon, mc_samples=n, seed=7).system.uptime
    assert runs.period_availability @ np.diff(np.r_[0.0, ends]) == (
        pytest.approx(up, rel=1e-10)
    )
    assert np.any(runs.contract_samples < 0.0)


def test_a_contract_alone_is_priced():
    rbd = RepairableRBD(
        LINE, {"p": {"reliability": E([1e-3]), "repairability": E([0.1])}}
    )
    runs = rbd.npv_distribution(
        8760.0, mc_samples=50, seed=1, contract=terms()
    )
    assert runs.mean == pytest.approx(runs.contract_payoff)
    assert runs.mean < 0.0


@pytest.mark.parametrize("contract", [None, terms(bonus_per_point=50.0)])
def test_identical_plans_differ_by_exactly_0(contract):
    gain = pump().compare_npv(
        pump(),
        8760.0,
        discount_rate=1e-5,
        value_rate=10.0,
        contract=contract,
        mc_samples=300,
        seed=2,
    )
    assert isinstance(gain, NPVComparison)
    assert np.all(gain.differences == 0.0)
    assert gain.mean == 0.0 and gain.interval.standard_error == 0.0
    assert gain.probability_other_better == 0.0


def test_the_common_numbers_shrink_the_difference_variance():
    kwargs = dict(
        discount_rate=1e-5,
        value_rate=10.0,
        mc_samples=1000,
        seed=11,
        contract=terms(bonus_per_point=50.0),
    )
    gain = pump(5.0).compare_npv(pump(10.0), 8760.0, **kwargs)
    this = pump(5.0).npv_distribution(8760.0, **kwargs)
    other = pump(10.0).npv_distribution(8760.0, **kwargs)
    # Each plan's runs are npv_distribution's, by seed.
    assert np.array_equal(gain.this.samples, this.samples)
    assert np.array_equal(gain.other.samples, other.samples)
    assert np.var(gain.differences) < np.var(this.samples) + np.var(
        other.samples
    )
    # About half here: the contract's thresholds pay on small
    # differences in a period's uptime, which the plans do not share.
    assert gain.interval.standard_error < 0.5 * gain.independent_standard_error
    # A faster repair is worth more, in nearly every run.
    assert gain.interval.lower > 0.0
    assert gain.probability_other_better < 0.1
    json.dumps(gain.to_dict())


def test_draws_that_cannot_be_matched_are_refused_by_name():
    unstreamed = systems_of_every_kind()["unstreamable"]
    with pytest.raises(NotImplementedError, match="'b'") as caught:
        unstreamed.compare_npv(
            unstreamed, 100.0, value_rate=1.0, mc_samples=10, seed=1
        )
    assert "independently" in str(caught.value)


def test_terms_go_through_plain_data():
    contract = AvailabilityContract(
        [100.0, 250.0], target=0.9, penalty_per_point=5.0, cap=20.0
    )
    data = json.loads(json.dumps(contract.to_dict()))
    assert AvailabilityContract.from_dict(data) == contract
    runs = pump().npv_distribution(
        300.0, value_rate=1.0, mc_samples=20, seed=1, contract=data
    )
    assert runs.period_ends == [100.0, 250.0]
    assert runs.contract == contract.to_dict()
    json.dumps(runs.to_dict())


@pytest.mark.parametrize(
    "args, match",
    [
        ((0.0, 0.9, 1.0), "length"),
        (([10.0, 5.0], 0.9, 1.0), "increasing"),
        ((10.0, 1.5, 1.0), "target"),
        ((10.0, 0.9, -1.0), "penalty_per_point"),
        ((10.0, 0.9, 1.0, math.nan), "bonus_per_point"),
        ((10.0, 0.9, 1.0, 0.0, 0.0), "cap"),
    ],
)
def test_terms_are_checked(args, match):
    with pytest.raises(ValueError, match=match):
        AvailabilityContract(*args)


def test_periods_must_fit_the_horizon():
    with pytest.raises(ValueError, match="after the horizon"):
        pump().npv_distribution(
            100.0,
            value_rate=1.0,
            mc_samples=10,
            seed=1,
            contract=AvailabilityContract([50.0, 150.0], 0.9, 1.0),
        )
    with pytest.raises(ValueError, match="No whole period"):
        pump().npv_distribution(
            100.0,
            value_rate=1.0,
            mc_samples=10,
            seed=1,
            contract=AvailabilityContract(200.0, 0.9, 1.0),
        )
    with pytest.raises(TypeError, match="contract"):
        pump().npv_distribution(100.0, value_rate=1.0, contract=0.9)
    with pytest.raises(TypeError, match="other"):
        pump().compare_npv("pump", 100.0, value_rate=1.0)
