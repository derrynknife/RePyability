"""Any surpyval model with a life's functions as a repairable component's
life or repair (#93): taken by what it can do (``sf``, ``ff``, ``df``,
``hf``, ``Hf``, ``qf`` and ``mean`` of time alone), not by its class.

Each newly taken kind is checked against the simulation (its exact mission
availability within four standard errors of a plain run), seeded runs
reproduced, and saved and loaded; the models that lack a function are
refused saying which and why.
"""

import importlib.util
import json
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable, NonRepairableRBD, RepairableRBD
from repyability.rbd._model_utils import by_functions, life_functions

ONE = [("s", "a"), ("a", "t")]
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
REPAIR = surv.Exponential.from_params([0.05])
WEIBULL = surv.Weibull.from_params([100.0, 2.0])


def data():
    g = np.random.default_rng(1)
    x = 100 * g.weibull(2.0, 80)
    Z = g.normal(size=(80, 1))
    x = x * np.exp(-0.3 * Z[:, 0])
    c = (x > 160).astype(int)
    return np.minimum(x, 160.0), Z, c


def kinds():
    x, Z, c = data()
    zero = np.r_[np.zeros(6), x]
    Zz = np.r_[Z[:6], Z]
    cz = np.r_[np.zeros(6, int), c]
    g = np.random.default_rng(2)
    level = np.abs(g.normal(np.repeat([1.0, 2, 3, 4], 10), 0.3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "Royston-Parmar": surv.RoystonParmar.fit(x, c),
            "PH at Z": surv.WeibullPH.fit(x, Z, c).at([[0.5]]),
            "AFT at Z": surv.WeibullAFT.fit(x, Z, c).at([[0.5]]),
            "PO at Z": surv.WeibullPO.fit(x, Z, c).at([[0.5]]),
            "distributional at Z": surv.WeibullDistributional.fit(x, Z, c).at(
                [[0.5]]
            ),
            "zero-inflated at Z": surv.WeibullZeroInflated.fit(
                zero, Zz, cz
            ).at([[0.5]]),
            "mixture cure at Z": surv.WeibullMixtureCure.fit(x, Z, c).at(
                [[0.5]]
            ),
            "destructive degradation": surv.DestructiveDegradation.fit(
                np.repeat([10.0, 20, 30, 40], 10), level, threshold=5.0
            ),
        }


KINDS = kinds()


@pytest.mark.parametrize("name", sorted(KINDS))
def test_it_has_a_lifes_functions(name):
    model = KINDS[name]
    assert life_functions(model) == {}
    assert by_functions(model)


@pytest.mark.parametrize("role", ["life", "repair"])
@pytest.mark.parametrize("name", sorted(KINDS))
def test_exact_against_the_simulation(name, role):
    model = KINDS[name]
    spec = (
        {"reliability": model, "repairability": REPAIR}
        if role == "life"
        else {"reliability": WEIBULL, "repairability": model}
    )
    rbd = RepairableRBD(
        PARALLEL,
        {"a": spec, "b": {"reliability": WEIBULL, "repairability": REPAIR}},
    )
    t = 500.0
    options = dict(
        t_simulation=t,
        mc_samples=300,
        seed=4,
        control_variate=False,
        conditional=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        simulated = rbd.availability(**options)
        exact = rbd.mission_availability(t)
        again = rbd.availability(**options)
    error = simulated.mean_availability_interval().standard_error
    assert abs(simulated.mean_availability - exact) < 4 * error + 1e-9
    assert again.mean_availability == simulated.mean_availability


@pytest.mark.parametrize("name", sorted(KINDS))
def test_saved_and_loaded(name):
    rbd = RepairableRBD(
        ONE, {"a": {"reliability": KINDS[name], "repairability": REPAIR}}
    )
    loaded = RepairableRBD.from_dict(json.loads(json.dumps(rbd.to_dict())))
    assert loaded.mission_availability(300.0) == rbd.mission_availability(
        300.0
    )
    diagram = NonRepairableRBD(ONE, {"a": KINDS[name]})
    back = NonRepairableRBD.from_dict(
        json.loads(json.dumps(diagram.to_dict()))
    )
    assert back.sf(80.0) == diagram.sf(80.0)


def test_a_regression_at_z_is_its_regression_there():
    x, Z, c = data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regression = surv.WeibullPH.fit(x, Z, c)
    part = NonRepairable(
        regression.at([[0.5]]), planned_cost=1.0, unplanned_cost=5.0
    )
    t = np.array([20.0, 80.0, 200.0])
    assert part.reliability_function(t) == pytest.approx(
        regression.sf(t, [[0.5]]), rel=1e-12
    )
    # The maintenance calculators take it as a distribution.
    assert np.isfinite(part.find_optimal_replacement())
    assert part.failures(300.0, age=50.0).expected_failures > 0.0


def test_a_cured_share_never_fails():
    """A mixture cure model at Z: its cured share never fails, so in the
    long run the unit is up for good, and its typical failure time is the
    median life of those that fail."""
    from repyability.rbd._model_utils import failure_time_scale, never_fails

    model = KINDS["mixture cure at Z"]
    share = float(model.sf(np.inf))
    assert share > 0.0
    assert never_fails(model) == share
    assert failure_time_scale(model) == pytest.approx(
        float(model.qf(0.5 * (1 - share)))
    )
    rbd = RepairableRBD(
        ONE, {"a": {"reliability": model, "repairability": REPAIR}}
    )
    assert rbd.mean_availability() == 1.0


def test_both_engines_draw_it_alike():
    if importlib.util.find_spec("numba") is None:
        pytest.skip("numba is not installed")
    rbd = RepairableRBD(
        PARALLEL,
        {
            "a": {
                "reliability": KINDS["Royston-Parmar"],
                "repairability": REPAIR,
            },
            "b": {"reliability": WEIBULL, "repairability": REPAIR},
        },
    )
    assert rbd.analysis_routes()["availability"].engine == "numba"
    options = dict(
        t_simulation=500.0, mc_samples=50, seed=5, control_variate=False
    )
    python = rbd.availability(engine="python", **options)
    numba = rbd.availability(engine="numba", **options)
    assert python.mean_availability == numba.mean_availability


# -- what is still refused, and why ------------------------------------------


def test_a_regression_without_covariates_is_refused_naming_at():
    x, Z, c = data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regression = surv.WeibullPH.fit(x, Z, c)
    with pytest.raises(ValueError, match=r"model\.at\(Z\)"):
        RepairableRBD(
            ONE, {"a": {"reliability": regression, "repairability": REPAIR}}
        )
    # As a repair it was taken and failed in the analysis; now refused.
    with pytest.raises(ValueError, match=r"missing 1 required"):
        RepairableRBD(
            ONE, {"a": {"reliability": WEIBULL, "repairability": regression}}
        )


def test_a_cox_model_at_z_is_refused_with_its_reason():
    """Its baseline is a step function that ends at the data: no mean."""
    x, Z, c = data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = surv.CoxPH.fit(x, Z, c).at([[0.5]])
    assert "mean" in life_functions(model)
    with pytest.raises(ValueError, match="no mean life"):
        RepairableRBD(
            ONE, {"a": {"reliability": model, "repairability": REPAIR}}
        )


def test_a_non_parametric_competing_risks_estimate_is_refused():
    x, _, c = data()
    e = np.where(c == 1, np.nan, np.random.default_rng(3).integers(1, 3, 80))
    model = surv.CompetingRisks.fit(x, e, c)
    assert {"qf", "mean"} <= set(life_functions(model))
    with pytest.raises(ValueError, match="cannot work out"):
        NonRepairable(model)
