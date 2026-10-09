"""A surpyval MixtureModel as a repairable component's life (#227).

A mixture (two modes, such as infant mortality and wear-out) has the
survival function, distribution, density and mean the analyses take, but
no quantile function (surpyval #651): the simulations' streams draw its
lives by inverting its distribution function (``mixture_quantile``), one
uniform a draw, and a life given an age through its cumulative hazard
(``MixtureLife``). Every analysis on diagrams with a mixture life is
checked in ``test_analysis_routes`` (the catalogue's "mixture life"
diagrams), and the engines against each other in ``test_catalogue``.
"""

import json

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairable, NonRepairableRBD, RepairableRBD
from repyability.rbd._sampling import (
    MixtureLife,
    inverse_sampler,
    mixture_quantile,
    stream_sampler,
)
from repyability.rbd.repairable_rbd import _aged_life
from repyability.tests.catalogue import mixture_life

REPAIR = surv.Exponential.from_params([1.0])
EDGES = [("s", "a"), ("a", "t")]


@pytest.fixture(scope="module")
def mix():
    return mixture_life()


def test_the_issue_s_diagram(mix):
    rbd = RepairableRBD(
        EDGES, {"a": {"reliability": mix, "repairability": REPAIR}}
    )
    mttf = float(mix.mean())
    assert rbd.mean_availability() == pytest.approx(mttf / (mttf + 1.0))
    assert NonRepairable(mix, REPAIR).mean_availability() == pytest.approx(
        mttf / (mttf + 1.0)
    )


def survival(mix, x):
    """The mixture's survival from its components' (its own ``sf``,
    ``1 - ff``, loses its precision in the tail, surpyval #671)."""
    return sum(w * mix.dist.sf(x, *p) for w, p in zip(mix.w, mix.params))


def test_the_quantile_inverts_the_distribution(mix):
    u = np.random.default_rng(5).random(20000)
    x = mixture_quantile(mix)(u)
    lower = u <= 0.5
    np.testing.assert_allclose(mix.ff(x[lower]), u[lower], rtol=1e-14)
    np.testing.assert_allclose(
        survival(mix, x[~lower]), 1 - u[~lower], rtol=1e-14
    )
    order = np.argsort(u)
    assert np.all(np.diff(x[order]) >= 0.0)


def test_the_quantile_s_ends(mix):
    x = mixture_quantile(mix)(np.array([0.0, 1e-300, 1.0 - 1e-16, 1.0]))
    assert x[0] == 0.0
    assert 0.0 < x[1] < 1e-290
    assert np.isfinite(x[2])
    assert x[3] == np.inf
    assert mixture_quantile(mix)(np.array([[0.25, 0.5]])).shape == (1, 2)


def test_only_a_repairable_diagram_s_streams_invert_it(mix):
    # A NonRepairableRBD's draws replay surpyval's own (its global RNG),
    # so they stay as they were.
    assert inverse_sampler(mix) is None
    assert stream_sampler(mix) is not None


def test_a_life_given_an_age(mix):
    u = np.random.default_rng(2).random(4000)
    left = np.array([_aged_life(mix, 60.0, v) for v in u])
    for x in (20.0, 50.0, 100.0):
        expected = float(mix.sf(60.0 + x) / mix.sf(60.0))
        error = np.sqrt(expected * (1 - expected) / len(u))
        assert abs(np.mean(left > x) - expected) < 4 * error


def test_the_simulation_agrees_with_the_exact_values(mix):
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": {
                "reliability": mix,
                "repairability": surv.Exponential.from_params([0.05]),
            },
            "b": {
                "reliability": surv.Weibull.from_params([150.0, 2.0]),
                "repairability": surv.Exponential.from_params([0.05]),
            },
        },
    )
    simulated = rbd.availability(
        t_simulation=2000.0,
        mc_samples=400,
        seed=11,
        control_variate=False,
        conditional=False,
    )
    exact = rbd.mission_availability(2000.0)
    error = simulated.mean_availability_interval().standard_error
    assert abs(simulated.mean_availability - exact) < 4 * error


def test_imperfect_repair_and_a_start_from_an_age(mix):
    from repyability import NodeState

    rbd = RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": mix,
                "repairability": REPAIR,
                "repair": {"model": "kijima1", "q": 0.5},
            }
        },
    )
    assert rbd.availability(t_simulation=300.0, mc_samples=20, seed=1)
    plain = RepairableRBD(
        EDGES, {"a": {"reliability": mix, "repairability": REPAIR}}
    )
    state = {"a": NodeState(age=60.0)}
    simulated = plain.availability(
        t_simulation=100.0,
        mc_samples=3000,
        seed=3,
        state=state,
        control_variate=False,
        conditional=False,
    )
    exact = plain.mission_availability(100.0, state=state)
    error = simulated.mean_availability_interval().standard_error
    assert abs(simulated.mean_availability - exact) < 4 * error


def test_saved_and_loaded(mix):
    rbd = RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": mix,
                "repairability": REPAIR,
                "preventive": {"interval": 120.0},
            }
        },
    )
    back = RepairableRBD.from_dict(json.loads(json.dumps(rbd.to_dict())))
    assert type(back.components["a"].reliability).__name__ == "MixtureModel"
    assert back.mean_availability() == rbd.mean_availability()


def test_two_mixtures_are_the_same_by_their_weights_too(mix):
    same = NonRepairableRBD._same_model
    assert same(mix, mixture_life())
    other = mixture_life()
    other.w = np.array([0.5, 0.5])
    assert not same(mix, other)
    assert not same(mix, surv.Weibull.from_params(list(mix.params[0])))


@pytest.mark.parametrize("cls", [NonRepairableRBD, RepairableRBD])
def test_its_parameters_cannot_be_drawn(mix, cls):
    if cls is NonRepairableRBD:
        rbd = cls(EDGES, {"a": mix})
        method, spec = rbd.sf_uncertainty, {"a": "fit"}
        args = (50.0,)
    else:
        rbd = cls(EDGES, {"a": {"reliability": mix, "repairability": REPAIR}})
        method = rbd.mean_availability_uncertainty
        spec, args = {"a": {"reliability": "fit"}}, ()
    with pytest.raises(
        ValueError, match="MixtureModel.*no parameter covariance"
    ):
        method(*args, spec)


def test_other_lives_are_refused_naming_the_component():
    with pytest.raises(ValueError) as caught:
        RepairableRBD(
            EDGES, {"a": {"reliability": "Weibull", "repairability": REPAIR}}
        )
    message = str(caught.value)
    assert message.startswith("Component 'a': ")
    assert "MixtureModel" in message and "got a str" in message


def test_a_standalone_unit_prices_age_replacement(mix):
    unit = NonRepairable(mix)
    unit.set_costs_planned_and_unplanned(cp=1, cu=5)
    age = float(unit.find_optimal_replacement())
    assert 0.0 < age < np.inf
    assert unit.cost_rate(age) <= unit.cost_rate(0.8 * age)
    assert unit.cost_rate(age) <= unit.cost_rate(1.25 * age)
    assert isinstance(MixtureLife(mix).qf(np.array([0.5]))[0], float)
