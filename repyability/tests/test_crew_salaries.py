"""Repair crews that cost something (``crew_cost_rate``): ``repair_crews``
crews at ``crew_cost_rate`` each cost ``n c`` per unit time, whatever
happens. Every cost result has it under ``"crews"``: ``n c`` in the long
run, ``n c T`` over a window and ``n c (1 - exp(-r T)) / r`` discounted,
and the same amount added to every simulated run. Without it every value
is as it was."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD, Timeline

E = surv.Exponential.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
LINE = [("s", "p"), ("p", "t")]
CREWS, RATE = 1, 3.0
PAID = CREWS * RATE


def unit(**extra):
    return {
        "reliability": E([0.1]),
        "repairability": E([1.0]),
        "repair_cost": 100.0,
        **extra,
    }


def pair(**options):
    """Two units in parallel, one repair crew for both."""
    return RepairableRBD(
        PAIR,
        {"a": unit(), "b": unit()},
        repair_crews=CREWS,
        downtime_cost_rate=50.0,
        **options,
    )


def present(T, r):
    """The crews' salaries from 0 to ``T``, discounted at ``r``."""
    return PAID * T if not r else PAID * (1.0 - np.exp(-r * T)) / r


def test_the_long_run_cost_rate_adds_the_salaries():
    unpaid, paid = pair(), pair(crew_cost_rate=RATE)
    assert paid.expected_cost_rate() == pytest.approx(
        unpaid.expected_cost_rate() + PAID, rel=1e-15
    )
    assert paid.total_cost(100.0) == pytest.approx(
        unpaid.total_cost(100.0) + 100.0 * PAID, rel=1e-14
    )


@pytest.mark.parametrize("r", [0.0, 0.05])
def test_a_window_adds_the_closed_forms(r):
    windows = np.array([10.0, 50.0, 100.0])
    unpaid = pair().expected_cost(windows, discount_rate=r)
    paid = pair(crew_cost_rate=RATE).expected_cost(windows, discount_rate=r)
    np.testing.assert_allclose(
        paid.by_category["crews"], present(windows, r), rtol=1e-15
    )
    np.testing.assert_allclose(unpaid.by_category["crews"], 0.0, atol=0.0)
    np.testing.assert_allclose(
        paid.mean, unpaid.mean + present(windows, r), rtol=1e-13
    )
    for category, values in unpaid.by_category.items():
        if category != "crews":
            np.testing.assert_allclose(
                paid.by_category[category], values, rtol=1e-13
            )


def test_the_cash_flow_table_adds_up_to_the_closed_form():
    r, T = 0.05, 100.0
    result = pair(crew_cost_rate=RATE).expected_npv(
        T, discount_rate=r, periods=4
    )
    rows = [row for row in result.cash_flows if row["category"] == "crews"]
    assert [row["period"] for row in rows] == [1, 2, 3, 4]
    assert all(row["component"] is None for row in rows)
    # Each period's own, from its start to its end.
    for row in rows:
        start, end = row["start"], row["end"]
        assert -row["present_value"] == pytest.approx(
            present(end, r) - present(start, r), rel=1e-12
        )
        assert -row["undiscounted"] == pytest.approx(
            PAID * (end - start), rel=1e-12
        )
    total = -math.fsum(row["present_value"] for row in rows)
    assert total == pytest.approx(present(T, r), rel=1e-13)
    assert result.cost_by_category["crews"] == pytest.approx(
        present(T, r), rel=1e-13
    )
    unpaid = pair().expected_npv(T, discount_rate=r, periods=4)
    assert result.cost == pytest.approx(unpaid.cost + present(T, r), 1e-13)
    assert result.undiscounted_cost == pytest.approx(
        unpaid.undiscounted_cost + PAID * T, rel=1e-13
    )


def test_each_simulated_run_pays_the_same_salaries():
    T = 100.0
    options = dict(t_simulation=T, mc_samples=200, seed=1)
    unpaid = pair().cost(**options)
    paid = pair(crew_cost_rate=RATE).cost(**options)
    # Added to each run after it: the runs are otherwise the same.
    np.testing.assert_array_equal(paid.samples, unpaid.samples + PAID * T)
    assert paid.by_category["crews"] == pytest.approx(PAID * T, rel=1e-15)
    assert unpaid.by_category["crews"] == 0.0
    assert paid.mean == pytest.approx(unpaid.mean + PAID * T, rel=1e-13)
    assert paid.mean == pytest.approx(
        math.fsum(paid.by_category.values()), rel=1e-12
    )
    own = pair(crew_cost_rate=RATE).cost(
        control_variate=False, conditional=False, **options
    )
    assert own.mean == pytest.approx(
        math.fsum(own.by_category.values()), rel=1e-12
    )


def test_the_spread_by_window_pays_each_window_its_salaries():
    paid = pair(crew_cost_rate=RATE).cost_by_window(
        width=25.0, count=4, mc_samples=50, seed=2
    )
    unpaid = pair().cost_by_window(width=25.0, count=4, mc_samples=50, seed=2)
    np.testing.assert_allclose(paid.mean, unpaid.mean + 25.0 * PAID)
    np.testing.assert_allclose(
        paid.samples, unpaid.samples + 25.0 * PAID, rtol=0.0, atol=1e-9
    )


def test_a_simulated_npv_pays_the_discounted_salaries():
    r, T = 0.05, 100.0
    options = dict(discount_rate=r, mc_samples=100, seed=3)
    paid = pair(crew_cost_rate=RATE).npv_distribution(T, **options)
    unpaid = pair().npv_distribution(T, **options)
    assert paid.cost_by_category["crews"] == pytest.approx(
        present(T, r), rel=1e-13
    )
    np.testing.assert_allclose(
        paid.cost_samples, unpaid.cost_samples + present(T, r), rtol=1e-13
    )


def test_a_plan_against_its_log_pays_the_crews_as_planned():
    line = RepairableRBD(
        LINE,
        {"p": unit()},
        repair_crews=1,
        crew_cost_rate=RATE,
    )
    result = line.plan_vs_actual(
        {"p": Timeline.from_outages([(30.0, 31.0)], end=50.0)},
        50.0,
        100.0,
        discount_rate=0.01,
        periods=[25.0, 50.0, 100.0],
    )
    planned = [
        r for r in result.expected_cash_flows if r["category"] == "crews"
    ]
    actual = [r for r in result.actual_cash_flows if r["category"] == "crews"]
    assert [r["period"] for r in actual] == [1, 2]
    for want, got in zip(planned, actual):
        assert got["present_value"] == pytest.approx(
            want["present_value"], rel=1e-12
        )
    assert result.variance_by_category.get("crews", 0.0) == pytest.approx(
        0.0, abs=1e-9
    )


def test_unpriced_crews_change_nothing():
    windows = np.array([20.0, 100.0])
    for unpaid, zero in ((pair(), pair(crew_cost_rate=0.0)),):
        assert unpaid.expected_cost_rate() == zero.expected_cost_rate()
        np.testing.assert_array_equal(
            unpaid.expected_cost(windows, discount_rate=0.05).mean,
            zero.expected_cost(windows, discount_rate=0.05).mean,
        )
        options = dict(t_simulation=50.0, mc_samples=40, seed=4)
        np.testing.assert_array_equal(
            unpaid.cost(**options).samples, zero.cost(**options).samples
        )
        assert "crew_cost_rate" not in zero.to_dict()


def test_salaries_alone_are_priced():
    rbd = RepairableRBD(
        PAIR,
        {"a": unit(repair_cost=0.0), "b": unit(repair_cost=0.0)},
        repair_crews=2,
        crew_cost_rate=RATE,
    )
    assert rbd.has_costs
    assert rbd.expected_cost_rate() == 2 * RATE
    assert float(rbd.expected_cost(10.0).mean) == 20 * RATE
    result = rbd.cost(t_simulation=10.0, mc_samples=20, seed=5)
    np.testing.assert_array_equal(result.samples, np.full(20, 20 * RATE))


def test_more_crews_cost_more_salaries():
    one, two = pair(crew_cost_rate=RATE), RepairableRBD(
        PAIR,
        {"a": unit(), "b": unit()},
        repair_crews=2,
        downtime_cost_rate=50.0,
        crew_cost_rate=RATE,
    )
    assert float(two.expected_cost(10.0).by_category["crews"]) == 20 * RATE
    assert float(one.expected_cost(10.0).by_category["crews"]) == 10 * RATE


def test_interval_choices_as_if_unlimited_keep_the_salaries():
    def system(**options):
        return RepairableRBD(
            PAIR,
            {
                name: {
                    "reliability": surv.Weibull.from_params([100.0, 3.0]),
                    "repairability": E([0.5]),
                    "repair_cost": 100.0,
                    "preventive": {"interval": 40.0, "cost": 10.0},
                }
                for name in "ab"
            },
            repair_crews=1,
            downtime_cost_rate=50.0,
            **options,
        )

    unpaid = system().optimal_replacement_intervals(
        assume_unlimited_crews=True
    )
    paid = system(crew_cost_rate=RATE).optimal_replacement_intervals(
        assume_unlimited_crews=True
    )
    assert paid.intervals == unpaid.intervals
    assert paid.cost_rate == pytest.approx(unpaid.cost_rate + PAID, 1e-15)
    with pytest.raises(ValueError, match="salaries alone"):
        system(crew_cost_rate=RATE).optimal_replacement_intervals(
            assume_unlimited_crews=True, max_cost_rate=PAID
        )


def test_a_crew_cost_needs_crews_and_is_a_number():
    with pytest.raises(ValueError, match="repair_crews"):
        RepairableRBD(PAIR, {"a": unit(), "b": unit()}, crew_cost_rate=1.0)
    for bad in (-1.0, math.inf, "3", True):
        with pytest.raises(ValueError, match="crew_cost_rate"):
            pair(crew_cost_rate=bad)


def test_a_crew_cost_is_saved():
    rbd = pair(crew_cost_rate=RATE)
    back = RepairableRBD.from_dict(rbd.to_dict())
    assert back.crew_cost_rate == RATE
    assert back.expected_cost_rate() == rbd.expected_cost_rate()
