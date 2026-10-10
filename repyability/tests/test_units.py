"""Units of measure: results do not depend on the unit a diagram is in,
and a duty cycle puts a life in operating time on the calendar."""

import math

import numpy as np
import pytest
import surpyval as surv
from surpyval import AcceleratedLife, life_models

from repyability import (
    NonRepairableRBD,
    RegressionNode,
    RepairableRBD,
    StandbyModel,
)
from repyability.rbd._duty import _SCALED, on_calendar
from repyability.rbd.serialisation import rbd_from_json, rbd_to_json

W = surv.Weibull.from_params
E = surv.Exponential.from_params
LN = surv.LogNormal.from_params
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]
ONE = [("s", "a"), ("a", "t")]


# Results do not depend on the unit -----------------------------------------


def diagrams(c):
    """The same diagrams with every time multiplied by ``c``."""
    return {
        "series": NonRepairableRBD(
            SERIES, {"a": W([100 * c, 2]), "b": LN([math.log(300 * c), 0.8])}
        ),
        "cold standby": NonRepairableRBD(
            ONE, {"a": StandbyModel([W([100 * c, 2])] * 2)}
        ),
        "cold standby, different units": NonRepairableRBD(
            ONE, {"a": StandbyModel([W([100 * c, 2]), W([80 * c, 1.5])])}
        ),
        "warm standby": NonRepairableRBD(
            ONE,
            {"a": StandbyModel([W([100 * c, 2])] * 2, dormancy_factor=0.5)},
        ),
    }


@pytest.mark.parametrize("c", [1e-6, 1e-3, 1e3, 1e6])
def test_results_scale_with_the_unit(c):
    # Tiny or huge times give the same answers as times near 1: no step
    # or grid is sized in absolute time.
    t = np.array([0.0, 50.0, 100.0, 200.0])
    for name, (base, scaled) in zip(
        diagrams(1.0), zip(diagrams(1.0).values(), diagrams(c).values())
    ):
        pairs = {
            "sf": (base.sf(t), scaled.sf(t * c)),
            "ff": (base.ff(t), scaled.ff(t * c)),
            "df": (base.df(t), np.asarray(scaled.df(t * c)) * c),
            "hf": (base.hf(t[1:]), np.asarray(scaled.hf(t[1:] * c)) * c),
            "mean": (base.mean(), scaled.mean() / c),
        }
        for what, (x, y) in pairs.items():
            # Rounding aside: 1e-16 of a probability at 0, say.
            scale = np.max(np.abs(x))
            np.testing.assert_allclose(
                y, x, rtol=1e-6, atol=1e-12 * scale, err_msg=f"{name}: {what}"
            )


def test_differential_importance_does_not_depend_on_the_unit():
    # A LogNormal's mu is the log of a time: in years it is ln(8760) less
    # than in hours, so moving it by a share of itself would weigh it by
    # the unit. A proportional change of it is one of the time instead.
    def shares(c):
        rbd = NonRepairableRBD(
            SERIES,
            {"a": LN([math.log(3000 * c), 0.8]), "b": W([4000 * c, 1.5])},
        )
        return rbd.differential_importance(
            2000 * c, over="parameters", change="proportional"
        )

    hours, years = shares(1.0), shares(1 / 8760)
    for key in hours:
        assert years[key] == pytest.approx(hours[key], rel=1e-4, abs=1e-8)


def test_a_covariate_has_no_proportional_change():
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 2.0, 200)
    x = rng.weibull(2.0, 200) * 80.0 / np.exp(0.4 * (load - 1))
    model = surv.WeibullAFT.fit(x, Z=load.reshape(-1, 1))
    rbd = NonRepairableRBD(ONE, {"a": RegressionNode(model, covariates=[1.0])})
    with pytest.raises(ValueError, match="covariates, whose zero"):
        rbd.differential_importance(
            50.0, over="parameters", change="proportional"
        )
    rbd.differential_importance(50.0, over="parameters", change="uniform")


def test_an_accelerated_life_model_is_a_regression_node():
    # An Arrhenius life of one stress (a temperature, in kelvin): fitted
    # with one stress, though its life model has two parameters.
    rng = np.random.default_rng(0)
    kelvin = np.repeat([323.15, 348.15, 373.15], 30)
    x = surv.Weibull.random(90, 100, 2, random_state=rng) * np.exp(
        3000 / kelvin - 3000 / 373.15
    )
    model = AcceleratedLife(surv.Weibull, life_models.Exponential).fit(
        x, kelvin
    )
    node = RegressionNode(model, covariates=[340.0])
    rbd = NonRepairableRBD(ONE, {"a": node})
    assert rbd.sf(100.0) == pytest.approx(
        float(np.ravel(model.sf(100.0, np.array([[340.0]])))[0])
    )
    assert rbd_from_json(rbd_to_json(rbd)).mean() == pytest.approx(rbd.mean())
    assert [lever.name for lever in rbd.levers()] == ["covariate.0"]
    with pytest.raises(ValueError, match="fitted with 1 covariate"):
        RegressionNode(model, covariates=[340.0, 1.0])


def test_a_probability_is_no_life_in_the_long_run():
    # A per-demand probability has no mean time: read as one, the long run
    # took p = 0.1 for the mean life (availability 0.09, against the
    # simulations' 0.999).
    rbd = RepairableRBD(
        ONE,
        {
            "a": {
                "reliability": surv.FixedEventProbability.from_params(0.1),
                "repairability": E([1.0]),
                "repair_cost": 5.0,
            }
        },
    )
    routes = rbd.analysis_routes()
    for name, call in [
        ("mean_availability", rbd.mean_availability),
        ("system_failure_frequency", rbd.system_failure_frequency),
        ("expected_cost_rate", rbd.expected_cost_rate),
        ("birnbaum_importance", rbd.birnbaum_importance),
    ]:
        assert routes[name].route == "refused"
        with pytest.raises(NotImplementedError, match="no mean time"):
            call()
    assert routes["availability"].route == "simulated"


# Duty cycles ---------------------------------------------------------------

PARAMETERS = {
    "Exponential": [0.01],
    "Weibull": [100, 2],
    "ExpoWeibull": [100, 2, 1.5],
    "LogLogistic": [100, 3],
    "Rayleigh": [100],
    "LogNormal": [4, 0.5],
    "Galton": [4, 0.5],
    "Gamma": [2, 0.05],
    "Normal": [100, 20],
    "Gauss": [100, 20],
    "Gumbel": [100, 20],
    "GumbelLEV": [100, 20],
    "Logistic": [100, 20],
    "Uniform": [10, 200],
}


def test_every_scaled_distribution_is_checked():
    assert set(PARAMETERS) == set(_SCALED)


@pytest.mark.parametrize("name", sorted(PARAMETERS))
def test_a_life_on_the_calendar(name):
    # Operating a fraction d of the time, R_c(t) = R(d t).
    model = getattr(surv, name).from_params(PARAMETERS[name])
    d, t = 0.3, np.array([5.0, 40.0, 90.0, 150.0])
    moved = on_calendar("a", model, d)
    np.testing.assert_allclose(
        np.ravel(moved.sf(t / d)), np.ravel(model.sf(t)), atol=1e-12
    )
    offset = W([100, 2], gamma=20.0)
    np.testing.assert_allclose(
        np.ravel(on_calendar("a", offset, d).sf(t / d)),
        np.ravel(offset.sf(t)),
        atol=1e-12,
    )


def duty_rbd(life, duty, **more):
    spec = {"reliability": life, "repairability": E([0.5]), **more}
    if duty is not None:
        spec["duty"] = duty
    return RepairableRBD(ONE, {"a": spec})


def test_a_duty_is_the_life_on_the_calendar():
    # A quarter of the time: a Weibull life of scale 100 in operating
    # time lasts 400 on the calendar; repairs and maintenance stay on it.
    part = duty_rbd(W([100, 2]), 0.25, preventive={"interval": 300.0})
    calendar = duty_rbd(W([400, 2]), None, preventive={"interval": 300.0})
    assert part.mean_availability() == pytest.approx(
        calendar.mean_availability(), rel=1e-12
    )
    first = part.availability(t_simulation=2000.0, mc_samples=50, seed=3)
    second = calendar.availability(t_simulation=2000.0, mc_samples=50, seed=3)
    assert first.system_failures == second.system_failures
    # The levers are the life's in operating time, as given.
    levers = {lever.name: lever.value for lever in part.levers()}
    assert levers["reliability.alpha"] == 100.0
    again = rbd_from_json(rbd_to_json(part))
    assert again.mean_availability() == part.mean_availability()
    assert again._init_args["components"]["a"]["duty"] == 0.25


def test_a_duty_of_one_changes_nothing():
    assert duty_rbd(W([100, 2]), 1.0).mean_availability() == (
        duty_rbd(W([100, 2]), None).mean_availability()
    )


@pytest.mark.parametrize(
    "life, duty, match",
    [
        (W([100, 2]), 0.0, r"in \(0, 1\]"),
        (W([100, 2]), 1.5, r"in \(0, 1\]"),
        (W([100, 2]), "half", r"in \(0, 1\]"),
        (surv.FixedEventProbability.from_params(0.1), 0.5, "on the calendar"),
    ],
)
def test_a_duty_is_checked(life, duty, match):
    with pytest.raises(ValueError, match=match):
        duty_rbd(life, duty)
