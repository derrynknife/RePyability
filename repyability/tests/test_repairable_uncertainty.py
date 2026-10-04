"""Parameter uncertainty in a repairable diagram (#200), and quasi-random
draws in both diagrams: the spread of the availability and the cost rate
over plausible models of the components, each input's share of it, and
draws from a scrambled Sobol sequence."""

import numpy as np
import pytest
import scipy.stats as st
import surpyval as surv

from repyability import (
    BetaFactor,
    CCFGroup,
    NonRepairableRBD,
    RepairableRBD,
)

E = surv.Exponential.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
SERIES = [("s", "pump"), ("pump", "valve"), ("valve", "t")]


def unit(life, repair):
    return {"reliability": life, "repairability": repair}


def fitted(median, n):
    spread = 0.4 * np.linspace(-1.6, 1.6, n)
    return surv.LogNormal.fit(median * np.exp(spread))


def plant():
    return RepairableRBD(
        SERIES,
        {
            "pump": unit(E([0.01]), fitted(4.5, 12)),
            "valve": unit(E([0.005]), fitted(4.0, 8)),
        },
    )


def test_each_draw_is_the_diagram_rebuilt_with_its_models():
    # A list of alternative repair models: each draw is one of them, and
    # its availability that of the diagram with it.
    models = [E([0.5]), E([1.0]), E([2.0])]
    rbd = RepairableRBD(PAIR, {n: unit(E([0.1]), E([1.0])) for n in "ab"})

    def built(repair):
        return RepairableRBD(PAIR, {n: unit(E([0.1]), repair) for n in "ab"})

    shared = rbd.mean_availability_uncertainty(
        {("a", "b"): {"repairability": models}}, n_draws=60, seed=2
    )
    alike = {built(m).mean_availability() for m in models}
    assert set(np.round(shared.samples, 12)) <= set(np.round(list(alike), 12))
    assert shared.nominal == pytest.approx(rbd.mean_availability())
    # Drawn apart, the two pumps' repairs are often different.
    apart = rbd.mean_availability_uncertainty(
        {
            "a": {"repairability": models},
            "b": {"repairability": models},
        },
        n_draws=60,
        seed=2,
    )
    assert not set(np.round(apart.samples, 12)) <= set(
        np.round(list(alike), 12)
    )


def test_the_long_run_availability_over_a_repair_rate_known_to_a_range():
    # One unit, repaired at a rate uniform on [0.5, 1.5]: its availability
    # mu / (mu + 0.1) averaged over the range, which the Sobol points come
    # to far closer than random ones.
    rbd = RepairableRBD(
        [("s", "c"), ("c", "t")], {"c": unit(E([0.1]), E([1.0]))}
    )
    rate = {"repairability": {"failure_rate": st.uniform(0.5, 1.0)}}
    exact = 1.0 - 0.1 * np.log(1.6 / 0.6)
    errors = {}
    for sampling in ("random", "sobol"):
        result = rbd.mean_availability_uncertainty(
            {"c": rate}, n_draws=512, seed=4, sampling=sampling
        )
        errors[sampling] = abs(result.mean - exact)
    assert errors["sobol"] < 1e-5
    assert errors["sobol"] < errors["random"] / 10


def test_point_and_mission_availability_and_the_cost_rate():
    rbd = plant()
    point = rbd.point_availability_uncertainty([5.0, 50.0], n_draws=40, seed=0)
    assert point.samples.shape == (40, 2)
    np.testing.assert_allclose(
        point.nominal, rbd.point_availability([5.0, 50.0])
    )
    single = rbd.point_availability_uncertainty(5.0, n_draws=10, seed=0)
    assert single.samples.shape == (10,)
    assert isinstance(single.nominal, float)
    mission = rbd.mission_availability_uncertainty(100.0, n_draws=20, seed=0)
    assert mission.nominal == pytest.approx(rbd.mission_availability(100.0))
    lower, upper = mission.interval(0.9)
    assert lower < mission.nominal < upper
    costed = RepairableRBD(
        SERIES,
        {
            "pump": {**unit(E([0.01]), fitted(4.5, 12)), "repair_cost": 50.0},
            "valve": unit(E([0.005]), fitted(4.0, 8)),
        },
        downtime_cost_rate=100.0,
    )
    cost = costed.expected_cost_rate_uncertainty(n_draws=50, seed=0)
    assert cost.nominal == pytest.approx(costed.expected_cost_rate())
    assert cost.std > 0.0


def test_the_delta_method_against_the_draws():
    # Small uncertainty: the linearised variance is the draws', and the
    # shares agree with the Sobol indices.
    rbd = plant()
    delta = rbd.uncertainty_importance()
    assert sum(delta.first_order.values()) == pytest.approx(1.0)
    drawn = rbd.mean_availability_uncertainty(
        n_draws=1024, seed=1, sampling="sobol"
    )
    assert np.var(drawn.samples, ddof=1) == pytest.approx(
        delta.variance, rel=0.15
    )
    sobol = rbd.uncertainty_importance(
        method="sobol", n_draws=1024, seed=1, sampling="sobol"
    )
    for key, share in delta.first_order.items():
        assert sobol.first_order[key] == pytest.approx(share, abs=0.05)
        assert sobol.total[key] == pytest.approx(share, abs=0.05)
    over_time = rbd.uncertainty_importance(
        [5.0, 50.0], of="point_availability"
    )
    assert over_time.variance.shape == (2,)
    np.testing.assert_allclose(sum(over_time.first_order.values()), [1.0, 1.0])


def test_a_common_cause_group_s_model_is_an_input():
    group = CCFGroup(["a", "b"], BetaFactor(0.1))
    rbd = RepairableRBD(
        PAIR,
        {n: unit(E([0.1]), E([1.0])) for n in "ab"},
        ccf_groups=[group],
    )
    beta = {group: {"beta": st.beta(2, 18)}}
    result = rbd.mean_availability_uncertainty(beta, n_draws=50, seed=0)
    assert result.std > 0.0
    parts = rbd.uncertainty_importance(
        uncertainty={
            group: {"beta": st.beta(2, 18)},
            ("a", "b"): {
                "repairability": {"failure_rate": st.uniform(0.8, 0.4)}
            },
        }
    )
    assert set(parts.first_order) == {group, ("a", "b")}
    assert sum(parts.first_order.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="together"):
        rbd.mean_availability_uncertainty({"a": {"repairability": [E([1.0])]}})


@pytest.mark.parametrize(
    "uncertainty, match",
    [
        ({"pump": {"preventive.duration": [E([1.0])]}}, "no preventive"),
        ({"nowhere": "fit"}, "not a component"),
        ({"valve": [E([0.004])], ("valve",): "fit"}, "twice"),
        ({}, "Give the uncertain nodes"),
    ],
)
def test_what_cannot_be_drawn_is_refused(uncertainty, match):
    with pytest.raises(ValueError, match=match):
        plant().mean_availability_uncertainty(uncertainty, n_draws=5)


def test_what_the_quantities_take():
    rbd = plant()
    with pytest.raises(ValueError, match="not taken"):
        rbd.uncertainty_importance(5.0)
    with pytest.raises(ValueError, match="x is required"):
        rbd.uncertainty_importance(of="point_availability")
    with pytest.raises(ValueError, match="of must be"):
        rbd.uncertainty_importance(of="mean")
    with pytest.raises(ValueError, match="sampling"):
        rbd.mean_availability_uncertainty(n_draws=5, sampling="halton")
    with pytest.raises(ValueError, match="delta method"):
        rbd.uncertainty_importance(
            uncertainty={"valve": [E([0.004]), E([0.006])]}
        )
    same = [
        rbd.mean_availability_uncertainty(
            n_draws=16, seed=3, sampling="sobol"
        ).samples
        for _ in range(2)
    ]
    np.testing.assert_array_equal(*same)


def test_quasi_random_draws_in_a_nonrepairable_diagram():
    # Two pumps of one uncertain rate, then a valve: the reliability's
    # mean over the rate is far closer with the Sobol points, while the
    # random draws are as they were.
    edges = [("s", "p1"), ("s", "p2"), ("p1", "v"), ("p2", "v"), ("v", "t")]
    models = {
        "p1": E([0.002]),
        "p2": E([0.002]),
        "v": surv.FixedEventProbability.from_params(0.01),
    }
    rbd = NonRepairableRBD(edges, models)
    rate = {("p1", "p2"): {"failure_rate": st.uniform(0.001, 0.002)}}
    lam = np.linspace(0.001, 0.003, 200_001)
    exact = np.mean(0.99 * (1 - (1 - np.exp(-500 * lam)) ** 2))
    sobol = rbd.sf_uncertainty(
        500, rate, n_draws=1024, seed=0, sampling="sobol"
    )
    assert abs(sobol.mean - exact) < 1e-5
    plain = rbd.sf_uncertainty(500, rate, n_draws=10_000, seed=1)
    assert [round(v, 3) for v in plain.interval(0.9)] == [0.409, 0.813]
    importance = rbd.uncertainty_importance(
        500,
        {("p1", "p2"): rate[("p1", "p2")], "v": [models["v"]]},
        method="sobol",
        n_draws=256,
        seed=0,
        sampling="sobol",
    )
    assert importance.first_order[("p1", "p2")] == pytest.approx(1.0, abs=0.02)
