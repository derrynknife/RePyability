"""A ``NonRepairableRBD`` as a unit's life (UC-66 of the maintenance
README): a subassembly replaced whole at its failure, taken by its
functions, its ``qf`` inverting its ``ff`` to near machine precision."""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable, NonRepairableRBD, RepairableRBD
from repyability.maintenance import AgePolicy, Prices, UnitPlan
from repyability.rbd import routes
from repyability.rbd.serialisation import rbd_from_json, rbd_to_json

LAMBDA = 0.01
E = surv.Exponential.from_params([LAMBDA])
REPAIR = surv.Exponential.from_params([0.5])
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
SHARES = np.array(
    [1e-14, 1e-9, 1e-6, 1e-3, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999, 1 - 1e-9]
)


def pair_of(edges, model):
    return NonRepairableRBD(edges, {"a": model, "b": model})


def parallel_qf(p):
    """The parallel pair of exponentials' quantile, ``1 - exp(-l t) =
    sqrt(p)``, each tail in its precise form."""
    root = np.sqrt(p)
    return (
        np.where(p <= 0.5, -np.log1p(-root), -np.log((1 - p) / (1 + root)))
        / LAMBDA
    )


def test_a_parallel_pair_s_quantile_is_its_closed_form():
    found = pair_of(PARALLEL, E).qf(SHARES)
    np.testing.assert_allclose(found, parallel_qf(SHARES), rtol=1e-13)


def test_a_series_pair_s_quantile_is_the_one_model_it_is():
    # Two Weibulls of shape 2 in series are one Weibull, scale / sqrt(2).
    weibull = surv.Weibull.from_params([100.0, 2.0])
    one = surv.Weibull.from_params([100.0 / math.sqrt(2.0), 2.0])
    np.testing.assert_allclose(
        pair_of(SERIES, weibull).qf(SHARES), one.qf(SHARES), rtol=1e-13
    )


def test_the_quantile_s_ends_and_refusals():
    pair = pair_of(PARALLEL, E)
    assert pair.qf(0.0) == 0.0
    assert pair.qf(1.0) == math.inf
    assert isinstance(pair.qf(0.5), float)
    assert pair.qf(np.array([[0.25, 0.5]])).shape == (1, 2)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pair.qf(1.5)
    fixed = NonRepairableRBD(
        SERIES,
        {"a": surv.FixedEventProbability.from_params(0.1), "b": E},
    )
    if fixed.is_fixed:
        with pytest.raises(routes.Refused):
            fixed.qf(0.5)


# -- a unit whose life is a diagram --------------------------------------------

WEIBULL = surv.Weibull.from_params([100.0, 2.0])
ONE = surv.Weibull.from_params([100.0 / math.sqrt(2.0), 2.0])


def unit(life) -> RepairableRBD:
    return UnitPlan(
        life,
        REPAIR,
        tasks=(AgePolicy(interval=60.0, cost=1.0, duration=REPAIR),),
        prices=Prices(repair=5.0, downtime=1.0),
    ).diagram()


def test_a_unit_whose_life_is_a_diagram_is_the_unit_of_that_life():
    # Exact and numerical routes, to 1e-12 (fixed before running).
    mine, theirs = unit(pair_of(SERIES, WEIBULL)), unit(ONE)
    for name in ("mean_availability", "expected_cost_rate"):
        assert getattr(mine, name)() == pytest.approx(
            getattr(theirs, name)(), rel=1e-12
        ), name
    t = np.array([10.0, 50.0, 200.0])
    np.testing.assert_allclose(
        mine.point_availability(t), theirs.point_availability(t), rtol=1e-12
    )
    assert mine.expected_npv(300.0, discount_rate=1e-3).npv == pytest.approx(
        theirs.expected_npv(300.0, discount_rate=1e-3).npv, rel=1e-12
    )


def test_its_simulations_are_those_of_that_life():
    mine, theirs = unit(pair_of(SERIES, WEIBULL)), unit(ONE)
    options = dict(
        t_simulation=300.0, mc_samples=200, seed=3, control_variate=False
    )
    a = mine.availability(**options)
    b = theirs.availability(**options)
    assert a.mean_availability == pytest.approx(b.mean_availability, rel=1e-12)
    runs = dict(discount_rate=1e-3, mc_samples=100, seed=8)
    np.testing.assert_allclose(
        mine.npv_distribution(300.0, **runs).samples,
        theirs.npv_distribution(300.0, **runs).samples,
        rtol=1e-12,
    )


def test_a_repaired_parallel_pair_is_up_its_mttf_over_its_cycle():
    pair = pair_of(PARALLEL, E)
    rbd = UnitPlan(pair, REPAIR).diagram()
    mttf = 1.5 / LAMBDA  # 1/l + 1/(2 l)
    assert rbd.mean_availability() == pytest.approx(
        mttf / (mttf + 2.0), rel=1e-12
    )


def test_a_diagram_is_a_life_wherever_a_life_is():
    pair = pair_of(PARALLEL, E)
    part = NonRepairable(pair, unplanned_cost=5.0)
    assert part.reliability is pair
    spec = RepairableRBD(
        [("s", "u"), ("u", "t")],
        {"u": {"reliability": pair, "repairability": REPAIR}},
    )
    loaded = rbd_from_json(rbd_to_json(spec))
    assert loaded.mean_availability() == pytest.approx(
        spec.mean_availability(), rel=1e-12
    )
    route = spec.analysis_routes()["mean_availability"]
    assert route.route in (routes.EXACT, routes.NUMERICAL)
