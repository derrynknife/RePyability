"""Tornado data: the system's value with each lever at its low and its
high, every other lever at its value, sorted by the swing."""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    FaultTree,
    NonRepairableRBD,
    RegressionNode,
    RepairableRBD,
    TornadoResult,
    TornadoRow,
)

E, W, LN = (
    surv.Exponential.from_params,
    surv.Weibull.from_params,
    surv.LogNormal.from_params,
)
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
ONE = [("s", "a"), ("a", "t")]
RATES = {"a": (0.01, 0.5), "b": (0.02, 0.25)}


def exponential_series():
    return RepairableRBD(
        SERIES,
        {
            node: {"reliability": E([lam]), "repairability": E([mu])}
            for node, (lam, mu) in RATES.items()
        },
    )


def series_availability(rates) -> float:
    return math.prod(mu / (lam + mu) for lam, mu in rates.values())


def by_lever(result):
    return {(row.key, row.name): row for row in result.rows}


def test_a_series_of_exponential_units_swings_as_its_closed_form():
    result = exponential_series().tornado()
    assert isinstance(result, TornadoResult)
    assert result.quantity == "availability"
    assert result.base_value == pytest.approx(
        series_availability(RATES), rel=1e-12
    )
    rows = by_lever(result)
    assert len(rows) == 4 and not result.skipped
    for node, (lam, mu) in RATES.items():
        for part, index in (("reliability", 0), ("repairability", 1)):
            row = rows[(node, f"{part}.failure_rate")]
            rate = (lam, mu)[index]
            assert (row.low, row.high) == pytest.approx(
                (0.9 * rate, 1.1 * rate), rel=1e-12
            )
            assert not row.cut
            for value, at in (
                (row.value_low, row.low),
                (row.value_high, row.high),
            ):
                moved = dict(RATES)
                moved[node] = (at, mu) if index == 0 else (lam, at)
                assert value == pytest.approx(
                    series_availability(moved), rel=1e-12
                )
            assert row.swing == pytest.approx(
                abs(row.value_high - row.value_low), rel=1e-15
            )
    swings = [row.swing for row in result.rows]
    assert swings == sorted(swings, reverse=True)


def test_a_small_range_agrees_with_the_slope():
    rbd = exponential_series()
    slopes = rbd.parameter_sensitivity()
    relative = 1e-3
    for row in rbd.tornado(relative=relative).rows:
        change = slopes[row.key][row.name] * (row.high - row.low)
        # The central difference's error is of the order of the step
        # squared, relative to the change.
        assert row.value_high - row.value_low == pytest.approx(
            change, rel=1e-5
        )


def test_each_value_is_the_diagram_with_the_lever_moved():
    rbd = NonRepairableRBD(
        SERIES, {"a": LN([math.log(300.0), 0.8]), "b": W([400.0, 1.5])}
    )
    result = rbd.tornado(200.0)
    assert result.base_value == pytest.approx(float(rbd.sf(200.0)))
    rows = by_lever(result)
    # A LogNormal's mu is the log of a time: a tenth of 1, not of mu.
    mu = rows[("a", "mu")]
    assert (mu.low, mu.high) == pytest.approx(
        (math.log(300.0) - 0.1, math.log(300.0) + 0.1)
    )
    for row in result.rows:
        for value, at in (
            (row.value_low, row.low),
            (row.value_high, row.high),
        ):
            moved = rbd.with_levers({(row.key, row.name): at})
            assert value == float(moved.sf(200.0))


def test_a_repairable_diagram_over_time_and_its_cost():
    rbd = RepairableRBD(
        SERIES,
        {
            "a": {
                "reliability": W([100.0, 2.0]),
                "repairability": E([0.5]),
                "repair_cost": 10.0,
            },
            "b": {"reliability": E([0.01]), "repairability": E([0.2])},
        },
    )
    at = rbd.tornado(x=50.0)
    for row in at.rows:
        moved = rbd.with_levers({(row.key, row.name): row.high})
        assert row.value_high == pytest.approx(
            float(moved.point_availability(50.0)), rel=1e-12
        )
    cost = rbd.tornado(quantity="cost_rate")
    assert cost.base_value == pytest.approx(rbd.expected_cost_rate())
    row = by_lever(cost)[("a", "reliability.alpha")]
    moved = rbd.with_levers({("a", "reliability.alpha"): row.low})
    assert row.value_low == pytest.approx(moved.expected_cost_rate())
    window = rbd.tornado(window=200.0)
    assert window.base_value == pytest.approx(
        float(rbd.mission_availability(200.0)), rel=1e-12
    )


def test_a_counting_lever_moves_by_one():
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            node: {"reliability": E([0.01]), "repairability": E([0.1])}
            for node in ("a", "b")
        },
        repair_crews=2,
    )
    row = by_lever(rbd.tornado())[(None, "repair_crews")]
    # One crew fewer and one more; a third is never busy.
    assert (row.low, row.high) == (1.0, 3.0)
    one = rbd.with_levers({(None, "repair_crews"): 1})
    assert row.value_low == pytest.approx(float(one.mean_availability()))
    assert row.value_high == pytest.approx(float(rbd.mean_availability()))


def test_a_range_past_a_lever_s_values_is_cut_and_said():
    tree = FaultTree({"top": ("or", ["a", "b"])}, {"a": 0.95, "b": 0.2})
    rows = by_lever(tree.tornado())
    assert rows[("a", "p")].high == 1.0 and rows[("a", "p")].cut
    assert rows[("a", "p")].value_high == 1.0
    assert not rows[("b", "p")].cut
    given = tree.tornado(ranges={("b", "p"): (0.1, 1.5)})
    b = by_lever(given)[("b", "p")]
    assert (b.low, b.high, b.cut) == (0.1, 1.0, True)
    outside = tree.tornado(ranges={("b", "p"): (2.0, 3.0)})
    assert [(s["key"], s["name"]) for s in outside.skipped] == [("b", "p")]
    assert "lies outside its values" in outside.skipped[0]["reason"]


def test_a_covariate_needs_a_range():
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 2.0, 200)
    x = rng.weibull(2.0, 200) * 80.0 / np.exp(0.4 * (load - 1))
    model = surv.WeibullAFT.fit(x, Z=load.reshape(-1, 1))
    rbd = NonRepairableRBD(ONE, {"a": RegressionNode(model, covariates=[1.0])})
    lever = next(lv for lv in rbd.levers() if lv.name.startswith("covariate"))
    skipped = rbd.tornado(50.0).skipped
    assert [(s["key"], s["name"]) for s in skipped] == [("a", lever.name)]
    assert "give its range in ranges" in skipped[0]["reason"]
    ranged = by_lever(rbd.tornado(50.0, ranges={lever: (0.5, 2.0)}))
    row = ranged[("a", lever.name)]
    assert row.value_high == float(rbd.with_levers({lever: 2.0}).sf(50.0))


@pytest.mark.parametrize("c", [1e-6, 1e3])
def test_the_swings_do_not_depend_on_the_unit(c):
    def result(c):
        rbd = NonRepairableRBD(
            SERIES,
            {"a": LN([math.log(300.0 * c), 0.8]), "b": W([400.0 * c, 1.5])},
        )
        return rbd.tornado(200.0 * c)

    hours, other = result(1.0), result(c)
    assert [(r.key, r.name) for r in other.rows] == [
        (r.key, r.name) for r in hours.rows
    ]
    for a, b in zip(hours.rows, other.rows):
        assert b.value_low == pytest.approx(a.value_low, rel=1e-9)
        assert b.value_high == pytest.approx(a.value_high, rel=1e-9)


def test_the_result_goes_through_json():
    result = exponential_series().tornado()
    back = json.loads(json.dumps(result.to_dict()))
    assert back["quantity"] == "availability"
    assert len(back["rows"]) == 4
    assert isinstance(result.rows[0], TornadoRow)
    assert set(back["rows"][0]) == {
        "key",
        "name",
        "base",
        "low",
        "high",
        "value_low",
        "value_high",
        "swing",
        "cut",
    }
    assert "rows=4" in repr(result)


@pytest.mark.parametrize(
    "call, error, message",
    [
        (lambda r: r.tornado(relative=0.0), ValueError, "relative"),
        (lambda r: r.tornado(relative=1.5), ValueError, "relative"),
        (lambda r: r.tornado(quantity="mean"), ValueError, "quantity"),
        (lambda r: r.tornado(x=[1.0, 2.0]), ValueError, "one time"),
        (
            lambda r: r.tornado(x=1.0, quantity="cost_rate"),
            ValueError,
            "give window",
        ),
        (lambda r: r.tornado(x=1.0, window=2.0), ValueError, "not both"),
        (lambda r: r.tornado(state={}), ValueError, "state"),
        (lambda r: r.tornado(ranges=[1]), TypeError, "ranges"),
        (
            lambda r: r.tornado(ranges={("a", "nope"): (1.0, 2.0)}),
            ValueError,
            "no lever",
        ),
        (
            lambda r: r.tornado(
                ranges={("a", "reliability.failure_rate"): (2.0, 1.0)}
            ),
            ValueError,
            "low <= high",
        ),
    ],
)
def test_what_is_not_valid_is_refused(call, error, message):
    with pytest.raises(error, match=message):
        call(exponential_series())


def test_a_counting_lever_s_range_is_whole():
    rbd = RepairableRBD(
        ONE,
        {"a": {"reliability": E([0.01]), "repairability": E([0.1])}},
        repair_crews=1,
    )
    with pytest.raises(ValueError, match="whole numbers"):
        rbd.tornado(ranges={(None, "repair_crews"): (1.0, 2.5)})


def test_a_tree_of_repaired_events_is_sent_to_its_diagram():
    tree = FaultTree(
        {"top": ("or", ["a"])},
        {"a": {"reliability": E([0.01]), "repairability": E([0.1])}},
    )
    with pytest.raises(NotImplementedError, match="to_repairable_rbd"):
        tree.tornado(10.0)
    assert tree.analysis_routes()["tornado"].route == "refused"
