"""A block whose life is on two clocks, hours and starts: surpyval's life
at a usage rate (``EquivalenceModel.at_usage(rate)``, surpyval 0.28 and
later) as a block, and its rate as a lever.

A unit making ``rate`` starts per hour, one start worth ``k`` hours, ages
``a = 1 + k rate`` equivalent hours per hour, so its life in hours is the
combined life with time divided by ``a``: for a Weibull on the combined
clock, a Weibull with scale ``alpha / a`` and the same shape. Each check
compares against that distribution or against closed forms.
"""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    NonRepairable,
    NonRepairableRBD,
    RepairableRBD,
)
from repyability.rbd import _usage

needs_at_usage = pytest.mark.skipif(
    not _usage.has_at_usage(),
    reason="the life at a usage rate needs surpyval 0.28 or later",
)

ALPHA, BETA, K = 2000.0, 2.0, 10.0
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
OTHER = surv.Weibull.from_params([3000.0, 1.5])
REPAIR = surv.LogNormal.from_params([2.0, 0.5])


def _model():
    return surv.WeibullEquivalence.from_params(
        [ALPHA, BETA, K], clocks=("hours", "starts"), unit="hours"
    )


def _in_hours(rate):
    """The life at ``rate`` written out: Weibull(alpha / a, beta)."""
    return surv.Weibull.from_params([ALPHA / (1.0 + K * rate), BETA])


def _sf_hours(t, rate):
    a = 1.0 + K * rate
    return math.exp(-((a * t / ALPHA) ** BETA))


@needs_at_usage
def test_a_non_repairable_block_is_its_life_in_hours():
    life = _model().at_usage(0.1)
    rbd = NonRepairableRBD(SERIES, {"a": life, "b": OTHER})
    same = NonRepairableRBD(SERIES, {"a": _in_hours(0.1), "b": OTHER})
    t = np.array([100.0, 500.0, 1500.0])
    np.testing.assert_allclose(rbd.sf(t), same.sf(t), rtol=1e-12)
    np.testing.assert_allclose(
        rbd.sf(t), [_sf_hours(x, 0.1) * OTHER.sf(x) for x in t], rtol=1e-12
    )
    assert rbd.mean() == pytest.approx(same.mean(), rel=1e-6)
    # its own mean, in hours
    alone = NonRepairableRBD([("s", "a"), ("a", "t")], {"a": life})
    assert alone.mean() == pytest.approx(
        ALPHA / 2.0 * math.gamma(1.0 + 1.0 / BETA), rel=1e-6
    )


@needs_at_usage
def test_a_repairable_component_is_its_life_in_hours():
    life = _model().at_usage(0.1)

    def system(reliability, **more):
        return RepairableRBD(
            SERIES,
            {
                "a": {"reliability": reliability, "repairability": REPAIR},
                "b": {
                    "reliability": OTHER,
                    "repairability": REPAIR,
                    **more,
                },
            },
        )

    t = np.array([100.0, 1000.0])
    np.testing.assert_allclose(
        system(life).point_availability(t),
        system(_in_hours(0.1)).point_availability(t),
        rtol=1e-12,
    )
    assert NonRepairable(life, REPAIR).reliability.params == pytest.approx(
        _in_hours(0.1).params, rel=1e-14
    )


@needs_at_usage
def test_a_duty_runs_the_hours_part_of_the_time():
    # Running half the time at 0.1 starts per running hour: the life on
    # the calendar is Weibull(alpha / (a d), beta).
    life = _model().at_usage(0.1)
    on_duty = RepairableRBD(
        SERIES,
        {
            "a": {"reliability": life, "repairability": REPAIR, "duty": 0.5},
            "b": {"reliability": OTHER, "repairability": REPAIR},
        },
    )
    written = RepairableRBD(
        SERIES,
        {
            "a": {
                "reliability": surv.Weibull.from_params(
                    [ALPHA / (2.0 * 0.5), BETA]
                ),
                "repairability": REPAIR,
            },
            "b": {"reliability": OTHER, "repairability": REPAIR},
        },
    )
    t = np.array([200.0, 2000.0])
    np.testing.assert_allclose(
        on_duty.point_availability(t),
        written.point_availability(t),
        rtol=1e-12,
    )


@needs_at_usage
@pytest.mark.parametrize("rate", [0.0, 0.1, 0.5])
def test_the_rate_is_a_lever_with_the_closed_form_derivative(rate):
    rbd = NonRepairableRBD(SERIES, {"a": _model().at_usage(rate), "b": OTHER})
    t = 500.0
    found = rbd.parameter_sensitivity(t)["a"]["rate"]
    # d/d rate of exp(-(a t / alpha)^beta), a = 1 + k rate
    a = 1.0 + K * rate
    z = (a * t / ALPHA) ** BETA
    exact = -BETA * z / a * K * math.exp(-z) * float(OTHER.sf(t))
    # At rate 0 the difference is one-sided (no rate below 0), so its
    # error is of the step's size, not its square.
    assert found == pytest.approx(exact, rel=1e-3 if rate == 0 else 1e-6)
    (lever,) = [lv for lv in rbd.levers() if lv.key == "a"]
    assert (lever.name, lever.value) == ("rate", rate)
    assert lever.bounds == (0.0, math.inf)
    moved = rbd.with_levers({("a", "rate"): 0.3})
    assert float(moved.sf(t)) == pytest.approx(
        _sf_hours(t, 0.3) * float(OTHER.sf(t)), rel=1e-12
    )
    with pytest.raises(ValueError):
        rbd.with_levers({("a", "rate"): -0.1})


@needs_at_usage
def test_a_repairable_component_has_the_rate_as_a_lever():
    rbd = RepairableRBD(
        SERIES,
        {
            "a": {
                "reliability": _model().at_usage(0.1),
                "repairability": REPAIR,
            },
            "b": {"reliability": OTHER, "repairability": REPAIR},
        },
    )
    names = [lv.name for lv in rbd.levers() if lv.key == "a"]
    assert names[0] == "reliability.rate"
    found = rbd.parameter_sensitivity()["a"]["reliability.rate"]
    # more starts per hour, shorter life, lower availability
    assert found < 0
    h = 1e-4
    up = rbd.with_levers({("a", "reliability.rate"): 0.1 + h})
    down = rbd.with_levers({("a", "reliability.rate"): 0.1 - h})
    base = rbd.parameter_sensitivity()  # long-run availability
    assert base["a"]["reliability.rate"] == pytest.approx(
        (up.mean_availability() - down.mean_availability()) / (2 * h),
        rel=1e-4,
    )


@needs_at_usage
def test_diagrams_with_one_save_and_load_it():
    life = _model().at_usage(0.1)
    rbd = NonRepairableRBD(SERIES, {"a": life, "b": OTHER})
    back = NonRepairableRBD.from_dict(json.loads(json.dumps(rbd.to_dict())))
    loaded = back.reliabilities["a"]
    assert isinstance(loaded, surv.LifeAtUsage)
    assert loaded.rate == 0.1 and loaded.unit == "hours"
    np.testing.assert_allclose(back.sf([500.0]), rbd.sf([500.0]), rtol=1e-14)
    rep = RepairableRBD(
        SERIES,
        {
            "a": {"reliability": life, "repairability": REPAIR},
            "b": {"reliability": OTHER, "repairability": REPAIR},
        },
    )
    back = RepairableRBD.from_json(rep.to_json())
    assert isinstance(
        back._init_args["components"]["a"]["reliability"], surv.LifeAtUsage
    )
    assert [lv.name for lv in back.levers()] == [
        lv.name for lv in rep.levers()
    ]


@needs_at_usage
def test_the_model_on_two_clocks_is_refused_saying_to_give_a_rate():
    with pytest.raises(ValueError, match=r"model\.at_usage\(rate\)"):
        NonRepairableRBD(SERIES, {"a": _model(), "b": OTHER})
    with pytest.raises(ValueError, match="at_usage"):
        RepairableRBD(
            SERIES,
            {
                "a": {"reliability": _model(), "repairability": REPAIR},
                "b": {"reliability": OTHER, "repairability": REPAIR},
            },
        )


def test_without_surpyval_0_28_it_says_what_is_needed(monkeypatch):
    # surpyval loads its names lazily, so stand in for one without it.
    monkeypatch.setattr(_usage, "life_at_usage_class", lambda: None)
    assert not _usage.has_at_usage()
    assert not _usage.is_life_at_usage(object())
    model = surv.WeibullEquivalence.from_params([ALPHA, BETA, K])
    with pytest.raises(ValueError, match="surpyval 0.28 or later"):
        NonRepairableRBD(SERIES, {"a": model, "b": OTHER})
    saved = {"model": model.to_dict(), "rate": 0.1}
    with pytest.raises(ValueError, match="surpyval 0.28 or later"):
        _usage.from_dict(saved)
