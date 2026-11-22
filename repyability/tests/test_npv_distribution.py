"""The net present value of each simulated run of a ``RepairableRBD``
(``npv_distribution``): its mean against the exact ``expected_npv``,
each run at no discounting against the simulation's own cost and uptime,
the engines run for run, and the summaries against a hand calculation."""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NPVDistribution, RepairableRBD
from repyability.rbd import _npv_runs, routes

E = surv.Exponential.from_params
W = surv.Weibull.from_params
LINE = [("s", "p"), ("p", "t")]
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def pump():
    return RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": E([1e-3]),
                "repairability": E([0.1]),
                "repair_cost": 500.0,
                "acquisition_cost": 2000.0,
            }
        },
    )


def age_replaced():
    return RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": W([100.0, 2.5]),
                "repairability": W([6.0, 1.5]),
                "repair_cost": 30.0,
                "replace_cost": 5.0,
                "downtime_cost": 1.0,
                "preventive": {
                    "interval": 60.0,
                    "policy": "age",
                    "duration": W([2.0, 2.0]),
                    "cost": 10.0,
                },
            }
        },
        downtime_cost_rate=4.0,
    )


def crews(life=None):
    spec = {
        "reliability": life or E([1e-2]),
        "repairability": E([0.05]),
        "repair_cost": 10.0,
        "downtime_cost": 0.5,
    }
    return RepairableRBD(
        PAIR,
        {"a": dict(spec), "b": dict(spec)},
        repair_crews=1,
        downtime_cost_rate=100.0,
    )


@pytest.mark.parametrize("make", [pump, age_replaced, crews])
def test_the_mean_is_the_expected_npv(make):
    rbd = make()
    horizon, rate, value = 400.0, 2e-3, 3.0
    runs = rbd.npv_distribution(
        horizon, discount_rate=rate, value_rate=value, mc_samples=4000, seed=3
    )
    exact = rbd.expected_npv(horizon, discount_rate=rate, value_rate=value)
    interval = runs.mean_interval(0.9999)
    assert interval.lower < exact.npv < interval.upper
    assert runs.mean == pytest.approx(runs.value - runs.cost, rel=1e-12)
    assert np.allclose(
        runs.samples, runs.value_samples - runs.cost_samples, rtol=1e-12
    )
    # The same split of value and cost, each within its own error.
    assert set(runs.cost_by_category) == set(exact.cost_by_category)
    for category, amount in exact.cost_by_category.items():
        assert runs.cost_by_category[category] == pytest.approx(
            amount, rel=0.1, abs=1e-9
        ), category
    assert set(runs.cost_by_component) == set(exact.cost_by_component)


@pytest.mark.parametrize(
    "make", [pump, age_replaced, crews, lambda: crews(W([100.0, 2.0]))]
)
def test_undiscounted_each_run_is_its_value_less_its_simulated_cost(make):
    rbd = make()
    horizon, value, n = 300.0, 2.0, 200
    runs = rbd.npv_distribution(
        horizon, value_rate=value, mc_samples=n, seed=5
    )
    cost = rbd.cost(horizon, mc_samples=n, seed=5, control_variate=False)
    uptime = rbd.simulate_timelines(horizon, mc_samples=n, seed=5).system
    expected = value * uptime.uptime - cost.samples - cost.acquisition_cost
    assert np.allclose(runs.samples, expected, rtol=1e-10, atol=1e-8)
    assert runs.undiscounted_npv == pytest.approx(runs.mean, rel=1e-12)


def test_crews_with_other_lives_are_simulated():
    rbd = crews(W([100.0, 2.0]))
    assert rbd.analysis_routes()["expected_npv"].route == routes.REFUSED
    route = rbd.analysis_routes()["npv_distribution"]
    assert route.route == routes.SIMULATED
    runs = rbd.npv_distribution(
        200.0, discount_rate=1e-3, mc_samples=50, seed=1
    )
    assert runs.samples.shape == (50,)


def test_the_engines_agree_run_for_run():
    pytest.importorskip("numba")
    rbd = age_replaced()
    kwargs = dict(discount_rate=1e-3, value_rate=2.0, mc_samples=300, seed=9)
    python = rbd.npv_distribution(250.0, engine="python", **kwargs)
    numba = rbd.npv_distribution(250.0, engine="numba", **kwargs)
    assert np.array_equal(python.samples, numba.samples)


def test_seeded_runs_repeat_and_are_made_in_parts(monkeypatch):
    rbd = age_replaced()
    whole = rbd.npv_distribution(
        200.0, discount_rate=1e-3, mc_samples=50, seed=2
    )
    again = rbd.npv_distribution(
        200.0, discount_rate=1e-3, mc_samples=50, seed=2
    )
    assert np.array_equal(whole.samples, again.samples)
    monkeypatch.setattr(_npv_runs, "_CHUNK", 7)
    parts = rbd.npv_distribution(
        200.0, discount_rate=1e-3, mc_samples=50, seed=2
    )
    assert np.allclose(parts.samples, whole.samples, rtol=1e-13, atol=1e-12)


def test_an_unseeded_run_keeps_the_seed_it_drew():
    rbd = pump()
    first = rbd.npv_distribution(500.0, value_rate=1.0, mc_samples=20)
    again = rbd.npv_distribution(
        500.0, value_rate=1.0, mc_samples=20, seed=first.seed
    )
    assert np.array_equal(first.samples, again.samples)


def test_the_summaries_by_hand():
    rbd = age_replaced()
    runs = rbd.npv_distribution(
        200.0,
        discount_rate=1e-3,
        value_rate=1.0,
        mc_samples=10,
        seed=4,
        tail=0.3,
        hurdle=0.0,
    )
    x = sorted(runs.samples.tolist())
    # The ceil(p n)-th lowest: 1st, 5th and 9th of ten.
    assert (runs.p10, runs.p50, runs.p90) == (x[0], x[4], x[8])
    assert runs.percentile(25) == x[2]
    assert runs.percentile(100) == x[9]
    assert runs.cvar == pytest.approx(sum(x[:3]) / 3, rel=1e-15)
    assert runs.probability_below_hurdle == sum(v < 0.0 for v in x) / 10
    assert runs.probability_below(x[5]) == 0.5
    se = float(np.std(x, ddof=1)) / math.sqrt(10)
    assert runs.mean_interval().standard_error == pytest.approx(se)


def test_a_percentile_interval_is_binomial():
    # For the median of 100 runs at 95%, the 40th and 61st lowest: the
    # binomial(100, 0.5) count below the median is in [40, 60] with
    # probability 0.965.
    samples = np.arange(100.0)[::-1]
    result = NPVDistribution(
        **{
            **dict.fromkeys(NPVDistribution.__dataclass_fields__, 0.0),
            "samples": samples,
        }
    )
    assert result.percentile_interval(50) == (39.0, 60.0)
    lower, upper = result.percentile_interval(1)
    # The 1st percentile: no run low enough for a lower bound; the 4th
    # lowest above.
    assert lower == -math.inf and upper == 3.0
    with pytest.raises(ValueError, match="percentile"):
        result.percentile(0)


def test_it_goes_through_json():
    runs = pump().npv_distribution(
        500.0, value_rate=1.0, mc_samples=5, seed=1, hurdle=10.0
    )
    data = json.loads(json.dumps(runs.to_dict()))
    assert len(data["samples"]) == 5
    assert data["probability_below_hurdle"] == runs.probability_below_hurdle
    assert "NPVDistribution(mean=" in repr(runs)


def test_a_cost_the_histories_do_not_record_is_refused():
    rbd = RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": W([100.0, 2.5]),
                "repairability": E([0.2]),
                "repair_cost": 30.0,
                "preventive": {"interval": 60.0, "cost": 10.0},
            }
        },
    )
    route = rbd.analysis_routes()["npv_distribution"]
    assert route.route == routes.REFUSED
    with pytest.raises(NotImplementedError, match="duration") as caught:
        rbd.npv_distribution(100.0, mc_samples=10, seed=1)
    assert "'p'" in str(caught.value)
    standby = RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": E([1e-2]),
                "repairability": E([0.2]),
                "repair_cost": 1.0,
                "standby": {"units": 2, "k": 1},
            }
        },
    )
    with pytest.raises(NotImplementedError, match="standby"):
        standby.npv_distribution(100.0, mc_samples=10, seed=1)


def test_a_cost_given_as_a_distribution_is_charged_its_mean():
    rbd = RepairableRBD(
        LINE,
        {
            "p": {
                "reliability": E([1e-2]),
                "repairability": E([0.2]),
                "repair_cost": surv.Normal.from_params([50.0, 5.0]),
            }
        },
    )
    runs = rbd.npv_distribution(300.0, mc_samples=20, seed=1)
    failures = rbd.simulate_timelines(300.0, mc_samples=20, seed=1)
    count = failures.components["p"].failures
    assert runs.priced_at_mean == ["p"]
    assert np.allclose(runs.cost_samples, 50.0 * count)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tail": 0.0}, "tail"),
        ({"tail": 1.5}, "tail"),
        ({"hurdle": math.inf}, "hurdle"),
        ({"mc_samples": 0}, "mc_samples"),
        ({"engine": "fast"}, "engine"),
    ],
)
def test_inputs_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        pump().npv_distribution(100.0, **{"seed": 1, **kwargs})


def test_nothing_priced_and_no_value_is_refused():
    rbd = RepairableRBD(
        LINE, {"p": {"reliability": E([1e-2]), "repairability": E([0.2])}}
    )
    with pytest.raises(routes.Refused):
        rbd.npv_distribution(100.0, mc_samples=5, seed=1)
    runs = rbd.npv_distribution(100.0, value_rate=1.0, mc_samples=5, seed=1)
    assert runs.cost == 0.0 and runs.cost_by_category == {}
