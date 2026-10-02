"""Non-parametric RBD nodes, and the fits to simulated lifetimes behind some
standby and load-sharing models, are deprecated in 0.11 and go in 0.12: one
minor release's notice, with a FutureWarning (which Python always shows)."""

import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    LoadSharingModel,
    NonRepairable,
    NonRepairableRBD,
    RepairableRBD,
    RepeatedNode,
    RepeatedStandbyNode,
    StandbyModel,
    __version__,
)
from repyability.utils.deprecation import REMOVAL

W, E = surv.Weibull.from_params, surv.Exponential.from_params
ONE = [("s", "a"), ("a", "t")]


def km():
    return surv.KaplanMeier.fit([10.0, 30.0, 45.0, 60.0, 80.0])


def aft(baseline):
    rng = np.random.default_rng(0)
    load = rng.uniform(0.5, 2.0, size=400)
    if baseline == "exponential":
        x = rng.exponential(100.0 / np.exp(0.6 * (load - 1.0)), size=400)
        return surv.ExponentialAFT.fit(x + 1e-3, Z=load.reshape(-1, 1))
    x = rng.weibull(2.0, size=400) * 80.0 / np.exp(0.4 * (load - 1))
    return surv.WeibullAFT.fit(x + 1e-3, Z=load.reshape(-1, 1))


def quiet(build):
    """Build, failing on any FutureWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        return build()


def test_the_removal_is_the_next_minor_release():
    # When the version reaches the removal, these deprecations must go.
    major, minor = map(int, __version__.split("."))
    assert REMOVAL == f"{major}.{minor + 1}"


@pytest.mark.parametrize(
    "node",
    [
        km,
        lambda: RepeatedNode(km(), 2, "parallel"),
        lambda: RepeatedStandbyNode(km(), 2),
        lambda: StandbyModel([km(), km()]),
    ],
    ids=["fit", "repeated", "repeated standby", "standby units"],
)
def test_a_non_parametric_node_warns(node):
    model = node()
    with pytest.warns(FutureWarning, match=rf"\['a'\].*removed in {REMOVAL}"):
        rbd = NonRepairableRBD(ONE, {"a": model})
    # Loading a saved one warns again.
    with pytest.warns(FutureWarning, match="non-parametric"):
        NonRepairableRBD.from_json(rbd.to_json())


@pytest.mark.parametrize(
    "spec",
    [
        lambda: {"reliability": km(), "repairability": E([0.5])},
        lambda: {"reliability": W([100, 2]), "repairability": km()},
        lambda: NonRepairable(km(), E([0.5])),
    ],
    ids=["life", "repair time", "NonRepairable"],
)
def test_a_non_parametric_component_warns(spec):
    component = spec()
    with pytest.warns(FutureWarning, match=rf"\['a'\].*removed in {REMOVAL}"):
        RepairableRBD(ONE, {"a": component})


def test_parametric_diagrams_do_not_warn():
    quiet(lambda: NonRepairableRBD(ONE, {"a": W([100, 2])}))
    quiet(
        lambda: RepairableRBD(
            ONE, {"a": {"reliability": W([100, 2]), "repairability": E([1])}}
        )
    )
    # A nested RBD warned when it was built: the outer one does not repeat
    # it.
    with pytest.warns(FutureWarning):
        inner = NonRepairableRBD(ONE, {"a": km()})
    quiet(lambda: NonRepairableRBD(ONE, {"a": inner}))


@pytest.mark.parametrize(
    "build, case",
    [
        (
            lambda: StandbyModel(
                [W([100, 2])] * 3,
                k=2,
                dormancy_factor=0.5,
                mc_samples=200,
                seed=1,
            ),
            "warm standby with 2 units operating",
        ),
        (
            lambda: StandbyModel(
                [W([100, 2]), W([80, 1.5]), W([100, 2])], k=2, mc_samples=200
            ),
            "cold standby with 2 different units operating",
        ),
        (
            lambda: LoadSharingModel(
                [aft("weibull"), aft("exponential")],
                load=2.0,
                mc_samples=200,
                seed=3,
            ),
            "units that are different",
        ),
    ],
    ids=["warm, two operating", "cold, two different", "load sharing"],
)
def test_a_fit_to_simulated_lifetimes_warns(build, case):
    with pytest.warns(FutureWarning, match=rf"{case}.*removed in {REMOVAL}"):
        model = build()
    assert model.is_simulated


def test_exact_and_numerical_models_do_not_warn():
    for build in (
        lambda: StandbyModel([W([100, 2])] * 2),
        lambda: StandbyModel([E([0.01])] * 2, dormancy_factor=0.5),
        lambda: StandbyModel([E([0.01])] * 3, k=2),
        lambda: StandbyModel([W([100, 2])] * 2, dormancy_factor=0.5),
        lambda: StandbyModel([W([100, 2]), W([80, 1.5])], dormancy_factor=1),
        lambda: StandbyModel([W([100, 2])] * 3, k=2),
        lambda: LoadSharingModel([aft("exponential")] * 2, load=2.0),
        lambda: LoadSharingModel([aft("weibull")] * 2, load=2.0),
    ):
        assert not quiet(build).is_simulated


def test_scoring_cold_standby_candidates_does_not_warn():
    # allocate_redundancy scores cold standby that needs two copies working
    # with a simulated StandbyModel of its own: not the user's to act on.
    rbd = NonRepairableRBD(ONE, {"a": W([100, 2])})
    result = quiet(
        lambda: rbd.allocate_redundancy(
            {"a": 1.0}, budget=3, t=50.0, required=2, strategy="cold"
        )
    )
    assert result.units == {"a": 3}
