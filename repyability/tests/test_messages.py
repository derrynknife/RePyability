"""Messages that say what to do (#232): the names 0.12 removed, a
PerfectReliability instance, a gate that lists itself, a single common-cause
group, outage logs, typos in node names, roles and parameters, seeds, the
stock with repair crews, and no doubled apostrophes."""

import ast
import math
import pathlib

import numpy as np
import pytest
import surpyval as surv
from surpyval.recurrent import CrowAMSAA

from repyability import (
    BetaFactor,
    CCFGroup,
    FaultTree,
    NonRepairableRBD,
    PerfectReliability,
    Repairable,
    RepairableRBD,
    StandbyModel,
    Timeline,
)

W, E = surv.Weibull.from_params, surv.Exponential.from_params
ONE = [("s", "a"), ("a", "t")]
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def unit(**extra):
    return {"reliability": W([100, 2]), "repairability": E([0.5]), **extra}


# -- the names 0.12 removed ---------------------------------------------------


def crow():
    model = Repairable(CrowAMSAA.from_params([100.0, 1.5]))
    model.set_repair_and_overhaul_costs(10.0, 1000.0)
    return model


@pytest.mark.parametrize(
    "call, old, new",
    [
        (
            lambda: NonRepairableRBD(ONE, {"a": W([100, 2])}).mean(
                method="simulate", N=100
            ),
            "N",
            "mc_samples",
        ),
        (
            lambda: RepairableRBD(ONE, {"a": unit()}).availability(
                10.0, tolerance=0.1, max_N=50
            ),
            "max_N",
            "max_samples",
        ),
        (
            lambda: StandbyModel([W([100, 2])] * 2, n_sims=100),
            "n_sims",
            "mc_samples",
        ),
        (lambda: crow().cost(250.0, n_simulations=10), "n_simulations", "mc_"),
    ],
    ids=["N", "max_N", "n_sims", "n_simulations"],
)
def test_a_removed_name_says_what_took_its_place(call, old, new):
    with pytest.raises(TypeError, match=f"'{old}': 0.12 removed it.*{new}"):
        call()


def test_the_misspelt_method_names_the_right_one():
    rbd = NonRepairableRBD(ONE, {"a": W([100, 2])})
    with pytest.raises(AttributeError, match="call fussell_vesely"):
        rbd.fussel_vesely()
    assert not hasattr(rbd, "fussel_vesely")
    assert not hasattr(NonRepairableRBD, "fussel_vesely")
    with pytest.raises(AttributeError, match="no attribute 'nonsense'$"):
        rbd.nonsense  # noqa: B018


def test_the_wrapped_methods_keep_their_signatures():
    import inspect

    parameters = inspect.signature(RepairableRBD.availability).parameters
    assert "mc_samples" in parameters and "N" not in parameters
    assert RepairableRBD.availability.__name__ == "availability"


# -- PerfectReliability() -----------------------------------------------------


JUNCTION = [("s", "a"), ("s", "b"), ("a", "J"), ("b", "J"), ("J", "t")]


@pytest.mark.parametrize(
    "junction, the_class",
    [
        (PerfectReliability(), PerfectReliability),
        (
            {"reliability": PerfectReliability()},
            {"reliability": PerfectReliability},
        ),
    ],
    ids=["an instance", "an instance as a spec's life"],
)
def test_a_perfect_reliability_instance_is_the_class(junction, the_class):
    given = RepairableRBD(JUNCTION, {"a": unit(), "b": unit(), "J": junction})
    reference = RepairableRBD(
        JUNCTION, {"a": unit(), "b": unit(), "J": the_class}
    )
    plain = RepairableRBD(
        JUNCTION, {"a": unit(), "b": unit(), "J": PerfectReliability}
    )
    assert given.mean_availability() == plain.mean_availability()
    assert given.to_dict() == reference.to_dict()
    # A junction's spec with no repair saves and loads.
    loaded = RepairableRBD.from_json(given.to_json())
    assert loaded.mean_availability() == plain.mean_availability()
    plain = NonRepairableRBD(
        JUNCTION,
        {"a": W([100, 2]), "b": W([90, 2]), "J": PerfectReliability()},
    )
    assert plain.to_dict()["reliabilities"]


def test_a_nonrepairable_life_that_never_ends_says_so():
    with pytest.raises(ValueError, match="never does"):
        from repyability import NonRepairable

        NonRepairable(PerfectReliability())


# -- the fault tree's loop and a single group ---------------------------------


@pytest.mark.parametrize(
    "gates",
    [
        {"T": ("or", ["T", "a"])},
        {"T": ("or", ["G", "a"]), "G": ("and", ["T", "b"])},
    ],
    ids=["a gate that lists itself", "two gates that list each other"],
)
def test_gates_in_a_loop_say_so(gates):
    events = {"a": 0.1, "b": 0.2}
    events = {
        k: v
        for k, v in events.items()
        if any(k in g[1] for g in gates.values())
    }
    with pytest.raises(ValueError, match="loop through 'T'"):
        FaultTree(gates, events)
    with pytest.raises(ValueError, match="loop through"):
        FaultTree(gates, events, top="T")


def test_a_single_common_cause_group_is_a_list_of_one():
    group = CCFGroup(["a", "b"], BetaFactor(0.1))
    gates, events = {"T": ("and", ["a", "b"])}, {"a": 0.01, "b": 0.01}
    one = FaultTree(gates, events, ccf_groups=group)
    listed = FaultTree(gates, events, ccf_groups=[group])
    assert one.top_event_probability() == listed.top_event_probability()
    models = {"a": W([100, 2]), "b": W([100, 2])}
    assert NonRepairableRBD(PAIR, models, ccf_groups=group).sf(
        20.0
    ) == NonRepairableRBD(PAIR, models, ccf_groups=[group]).sf(20.0)
    lives = {"a": unit(), "b": unit()}
    rate = {n: {**spec, "reliability": E([0.01])} for n, spec in lives.items()}
    assert (
        RepairableRBD(PAIR, rate, ccf_groups=group).mean_availability()
        == RepairableRBD(PAIR, rate, ccf_groups=[group]).mean_availability()
    )
    with pytest.raises(ValueError, match="a CCFGroup or a list of them"):
        FaultTree(gates, events, ccf_groups=5)


# -- outage logs --------------------------------------------------------------


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize(
    "outages, message",
    [
        ([(-5, 10)], "starts before the window's start"),
        ([(math.nan, 10)], "starts at nan, which is not a time"),
        ([("3", 10)], "starts at '3', which is not a time"),
        ([(3, math.nan)], "ends at nan, which is not a time"),
        ([(5, 3)], "ends before it starts"),
        ([(1, 2, 3)], "is a \\(start, end\\) pair"),
    ],
)
def test_an_outage_that_is_wrong_says_how(outages, message, merge):
    with pytest.raises(ValueError, match=message):
        Timeline.from_outages(outages, end=100, merge=merge)


def test_outages_out_of_order_still_point_to_merge():
    with pytest.raises(ValueError, match="out of order.*merge=True"):
        Timeline.from_outages([(100, 104), (102, 110)], end=1000)


# -- typos in node names ------------------------------------------------------


def test_an_unknown_node_is_given_the_closest_name():
    rbd = RepairableRBD(
        [("s", "bearing1"), ("bearing1", "t")],
        {"bearing1": unit(preventive={"interval": 50.0})},
    )
    with pytest.raises(ValueError, match="Did you mean 'bearing1'"):
        rbd.with_intervals({"bearing 1": 60.0})
    with pytest.raises(ValueError, match="Did you mean 'bearing1'"):
        rbd.spares_demand(100.0, nodes=["Bearing1"])  # case aside


def test_a_typo_in_a_part_names_the_part():
    rbd = RepairableRBD(
        [("s", "seal1"), ("seal1", "seal2"), ("seal2", "t")],
        {"seal1": unit(), "seal2": unit()},
    )
    with pytest.raises(
        ValueError, match="'seal 2' given in part 'seal'.*Did you mean 'seal2'"
    ):
        rbd.spares_demand(100.0, parts={"seal": ["seal1", "seal 2"]})


def test_a_junction_named_where_a_component_goes_is_called_a_junction():
    rbd = RepairableRBD(
        JUNCTION, {"a": unit(), "b": unit(), "J": PerfectReliability}
    )
    with pytest.raises(
        ValueError, match="'J' given in part 'p' is a junction"
    ):
        rbd.spares_demand(100.0, parts={"p": ["a", "J"]})
    with pytest.raises(ValueError, match="is a junction"):
        rbd.spares_demand(100.0, nodes=["J"])


# -- roles, parameters and the cost rate's name -------------------------------


def fitted():
    return surv.Weibull.fit([10, 20, 30, 40, 55, 70, 90])


def test_a_role_typo_is_named_with_the_roles_and_parameters():
    rbd = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": {"reliability": fitted(), "repairability": E([0.5])}},
    )
    with pytest.raises(
        ValueError, match="neither roles.*nor parameters.*mean 'reliability'"
    ):
        rbd.mean_availability_uncertainty(
            {"p": {"reliabilty": "fit"}}, n_draws=4, seed=1
        )
    with pytest.raises(ValueError, match="mix roles and parameters"):
        rbd.mean_availability_uncertainty(
            {"p": {"reliability": "fit", "alpha": E([0.1])}}, n_draws=4
        )
    with pytest.raises(ValueError, match="Did you mean 'alpha'"):
        NonRepairableRBD(ONE, {"a": fitted()}).sf_uncertainty(
            10.0, {"a": {"alfa": surv.Normal.from_params([40, 2])}}, n_draws=4
        )


def test_uncertainty_importance_takes_the_cost_rates_short_name():
    rbd = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {
            "p": {
                "reliability": fitted(),
                "repairability": E([0.5]),
                "repair_cost": 10.0,
            }
        },
    )
    short = rbd.uncertainty_importance(of="cost_rate")
    full = rbd.uncertainty_importance(of="expected_cost_rate")
    assert short.total == full.total and short.variance == full.variance


# -- seeds --------------------------------------------------------------------


@pytest.mark.parametrize(
    "seed", ["abc", 1.5, True, np.random.default_rng(1), -1, []]
)
def test_a_seed_that_is_none_says_what_one_is(seed):
    rbd = RepairableRBD(ONE, {"a": unit()})
    plain = NonRepairableRBD(ONE, {"a": W([100, 2])})
    calls = [
        lambda: rbd.availability(10.0, mc_samples=10, seed=seed),
        lambda: plain.mean(method="simulate", mc_samples=10, seed=seed),
        lambda: plain.sf_uncertainty(
            10.0, {"a": [W([100, 2]), W([90, 2])]}, n_draws=4, seed=seed
        ),
    ]
    for call in calls:
        with pytest.raises((TypeError, ValueError), match="seed must be"):
            call()


def test_a_generator_is_told_how_to_give_a_seed():
    rbd = RepairableRBD(ONE, {"a": unit()})
    with pytest.raises(TypeError, match=r"int\(rng.integers\(2\*\*32\)\)"):
        rbd.availability(10.0, mc_samples=10, seed=np.random.default_rng(1))


def test_numpy_integers_are_seeds():
    rbd = RepairableRBD(ONE, {"a": unit()})
    assert (
        rbd.availability(
            10.0, mc_samples=10, seed=np.int64(3)
        ).uptimes.tolist()
        == rbd.availability(10.0, mc_samples=10, seed=3).uptimes.tolist()
    )


# -- the stock with repair crews ----------------------------------------------


def test_the_stock_with_crews_says_it_has_no_simulation():
    lives = {"reliability": E([0.01]), "repairability": E([0.5])}
    rbd = RepairableRBD(PAIR, {"a": lives, "b": lives}, repair_crews=1)
    with pytest.raises(NotImplementedError) as caught:
        rbd.spares_stock(100.0, fill_rate=0.9)
    message = str(caught.value)
    assert "no simulation to fall back on" in message
    assert "repair_crews=None" in message
    assert rbd.analysis_routes()["spares_stock"].reason == message


# -- apostrophes --------------------------------------------------------------


def test_no_message_puts_s_after_a_quoted_name():
    # f"{node!r}'s" shows 'v1''s: the messages say "of node 'v1'" instead.
    root = pathlib.Path(__file__).resolve().parents[1]
    found = []
    for path in sorted(root.rglob("*.py")):
        if "tests" in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.JoinedStr):
                continue
            for a, b in zip(node.values, node.values[1:]):
                if (
                    isinstance(a, ast.FormattedValue)
                    and a.conversion == ord("r")
                    and isinstance(b, ast.Constant)
                    and str(b.value).startswith("'s")
                ):
                    found.append(f"{path.name}:{node.lineno}")
    assert not found, found


def test_an_uncertainty_message_names_the_node_once():
    rbd = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": {"reliability": fitted(), "repairability": E([0.5])}},
    )
    with pytest.raises(ValueError) as caught:
        rbd.mean_availability_uncertainty(
            {"p": {"reliability": {"gamma": E([1.0])}}}, n_draws=4, seed=1
        )
    assert "''s" not in str(caught.value)
    assert "The reliability of node 'p'" in str(caught.value)
