"""A diagram's levers are public (#244): ``levers()`` lists what
``parameter_sensitivity`` moves, in its order, as ``Lever`` results with
their values, their ranges and whether they are discrete or move a shared
calendar; ``with_levers`` builds the diagram with them moved, as the
sensitivities rebuild it; and the private names callers used before stay,
as they were, for 0.13."""

import json
import math
import warnings

import pytest
import surpyval as surv

from repyability import (
    MGL,
    BetaFactor,
    CCFGroup,
    Lever,
    NonRepairableRBD,
    RepairableRBD,
)
from repyability.rbd import _sensitivity
from repyability.tests.catalogue import nonrepairable_kinds, repairable_kinds

E, W = surv.Exponential.from_params, surv.Weibull.from_params
PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]


def unit(**more):
    return {"reliability": W([100.0, 2.0]), "repairability": E([0.5]), **more}


def constant(**more):
    """A unit with an exponential life, as limited crews need for exact
    long-run values."""
    return {"reliability": E([0.01]), "repairability": E([0.5]), **more}


def named(levers) -> list:
    return [(lever.key, lever.name) for lever in levers]


def values(rbd) -> dict:
    return {(lever.key, lever.name): lever.value for lever in rbd.levers()}


def kinds():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            **{f"repairable {k}": v for k, v in repairable_kinds().items()},
            **{
                f"non-repairable {k}": v
                for k, v in nonrepairable_kinds().items()
            },
        }


KINDS = kinds()


def sensitivity(rbd):
    if isinstance(rbd, NonRepairableRBD):
        return rbd.parameter_sensitivity(30.0)
    return rbd.parameter_sensitivity()


# -- what levers() lists ------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_the_levers_are_those_the_sensitivities_report(kind):
    rbd = KINDS[kind]
    levers = rbd.levers()
    assert all(isinstance(lever, Lever) for lever in levers)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reported = sensitivity(rbd)
        except (NotImplementedError, ValueError):
            return  # refused; its levers are listed all the same
    assert named(levers) == [(k, n) for k, d in reported.items() for n in d]


def test_a_lever_has_its_value_range_and_kind():
    rbd = RepairableRBD(
        [("s", "a"), ("a", "g"), ("g", "t")],
        {
            "a": unit(
                inspection={
                    "interval": 100.0,
                    "duration": E([2.0]),
                    "coverage": 0.9,
                    "full_test": 400.0,
                    "offset": 30.0,
                    "cost": 1.0,
                }
            ),
            "g": unit(standby={"units": 3, "k": 1, "dormancy_factor": 0.2}),
        },
        repair_crews=2,
    )
    levers = {(lever.key, lever.name): lever for lever in rbd.levers()}
    alpha = levers[("a", "reliability.alpha")]
    assert (alpha.value, alpha.bounds, alpha.discrete) == (
        100.0,
        (0.0, math.inf),
        False,
    )
    # A test interval is past its offset, an offset within the interval.
    assert levers[("a", "inspection.interval")].bounds == (30.0, math.inf)
    assert levers[("a", "inspection.offset")].bounds == (0.0, 100.0)
    assert levers[("a", "inspection.coverage")].bounds == (0.0, 1.0)
    assert levers[("a", "inspection.duration.failure_rate")].value == 2.0
    # A standby group needs a spare: more units than must work.
    units = levers[("g", "standby.units")]
    assert (units.value, units.bounds, units.discrete) == (
        3.0,
        (2.0, math.inf),
        True,
    )
    crews = levers[(None, "repair_crews")]
    assert (crews.value, crews.bounds, crews.discrete) == (
        2.0,
        (1.0, math.inf),
        True,
    )
    assert not any(lever.calendar for lever in levers.values())


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_a_lever_s_range_is_where_its_diagram_builds(kind):
    rbd = KINDS[kind]
    for lever in rbd.levers():
        low, high = lever.bounds
        assert low <= lever.value <= high, lever
        rbd.with_levers({lever: lever.value})
        for end, past in ((low, -1.0), (high, 1.0)):
            if not math.isfinite(end):
                continue
            outside = end + past * (1.0 if lever.discrete else 1e-6)
            with pytest.raises(ValueError, match=f"{lever.name!r} of"):
                rbd.with_levers({lever: outside})


def test_the_calendar_flag_marks_a_shared_calendar():
    block = {"policy": "block", "interval": 50.0, "cost": 1.0}
    shared = RepairableRBD(
        SERIES, {"a": unit(preventive=block), "b": unit(preventive=block)}
    )
    alone = RepairableRBD(SERIES, {"a": unit(preventive=block), "b": unit()})
    for rbd, expected in ((shared, True), (alone, False)):
        levers = {(lever.key, lever.name): lever for lever in rbd.levers()}
        assert levers[("a", "preventive.interval")].calendar is expected
        assert sum(lever.calendar for lever in levers.values()) == (
            2 if expected else 0
        )
        for lever in rbd.levers():
            internal = next(
                found
                for found in _sensitivity._levers(rbd)
                if (found.key, found.name) == (lever.key, lever.name)
            )
            assert lever.calendar == _sensitivity._calendar_lever(
                rbd, internal
            )


def test_a_lever_goes_through_json():
    rbd = RepairableRBD(
        PAIR,
        {"a": constant(), "b": constant()},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    lever = next(lever for lever in rbd.levers() if lever.name == "ccf_beta")
    out = json.loads(json.dumps(lever.to_dict()))
    assert out == {
        "key": ["a", "b"],
        "name": "ccf_beta",
        "value": 0.1,
        "discrete": False,
        "bounds": [0.0, 1.0],
        "calendar": False,
    }
    assert dict(lever)["name"] == "ccf_beta"
    assert json.loads(json.dumps(rbd.levers()[0].to_dict()))["bounds"] == [
        0.0,
        math.inf,
    ]


# -- with_levers --------------------------------------------------------------


def test_with_levers_rebuilds_as_the_sensitivities_do():
    # With one repair crew for two components the system is differenced:
    # rebuilt at each side of each lever, as with_levers rebuilds it.
    rbd = RepairableRBD(
        PAIR, {"a": constant(), "b": constant()}, repair_crews=1
    )
    step = 1e-5
    reported = rbd.parameter_sensitivity(rel_step=step)
    for lever in rbd.levers():
        if lever.discrete:
            more = rbd.with_levers({lever: lever.value + 1})
            change = rbd.mean_unavailability() - more.mean_unavailability()
        else:
            h = step * abs(lever.value)
            up = rbd.with_levers({lever: lever.value + h})
            down = rbd.with_levers({lever: lever.value - h})
            change = -(up.mean_unavailability() - down.mean_unavailability())
            change /= 2.0 * h
        assert reported[lever.key][lever.name] == change


def test_with_levers_moves_several_and_leaves_the_diagram_as_it_was():
    rbd = RepairableRBD(
        PAIR,
        {"a": constant(), "b": constant()},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        repair_crews=1,
    )
    before = values(rbd)
    levers = {(lever.key, lever.name): lever for lever in rbd.levers()}
    moved = rbd.with_levers(
        {
            levers[(("a", "b"), "ccf_beta")]: 0.2,
            (("a", "b"), "reliability.failure_rate"): 0.02,
            (None, "repair_crews"): 2,
        }
    )
    after = values(moved)
    assert after[(("a", "b"), "ccf_beta")] == 0.2
    assert after[(("a", "b"), "reliability.failure_rate")] == 0.02
    assert after[(None, "repair_crews")] == 2.0
    # Every member of the group takes the new life.
    assert moved.components["a"].reliability.params[0] == 0.02
    assert moved.components["b"].reliability.params[0] == 0.02
    assert moved.repair_crews == 2
    assert values(rbd) == before
    assert type(rbd.with_levers({})) is RepairableRBD


def test_a_test_interval_takes_its_full_tests_with_it():
    tested = unit(
        inspection={
            "interval": 100.0,
            "coverage": 0.9,
            "full_test": 400.0,
            "cost": 1.0,
        }
    )
    rbd = RepairableRBD([("s", "a"), ("a", "t")], {"a": tested})
    moved = rbd.with_levers({("a", "inspection.interval"): 50.0})
    spec = moved._init_args["components"]["a"]["inspection"]
    assert (spec["interval"], spec["full_test"]) == (50.0, 200.0)


@pytest.mark.parametrize(
    "given, error, message",
    [
        (
            {("a", "reliability.alpah"): 1.0},
            ValueError,
            "'a' has no lever 'reliability.alpah'. Did you mean "
            "'reliability.alpha'?",
        ),
        (
            {("A", "reliability.alpha"): 1.0},
            ValueError,
            "'A' has no levers.*Did you mean 'a'?",
        ),
        ({("a", "reliability.alpha"): "100"}, TypeError, "must be given a"),
        ({("a", "reliability.alpha"): True}, TypeError, "must be given a"),
        ({(None, "repair_crews"): 1.5}, ValueError, "whole number"),
        ({(None, "repair_crews"): 0}, ValueError, "between 1 and inf"),
        ({"a": 1.0}, TypeError, "by its Lever.*or as \\(key, name\\)"),
        ({("a", "reliability.alpha"): -1.0}, ValueError, "between 0 and"),
    ],
    ids=[
        "a misspelt lever",
        "a misspelt node",
        "text",
        "a boolean",
        "a fraction of a count",
        "no crews",
        "a node alone",
        "a negative scale",
    ],
)
def test_with_levers_refuses_what_is_no_lever_or_value(given, error, message):
    rbd = RepairableRBD(PAIR, {"a": unit(), "b": unit()}, repair_crews=1)
    with pytest.raises(error, match=message):
        rbd.with_levers(given)


def test_with_levers_takes_a_dict():
    rbd = RepairableRBD(PAIR, {"a": unit(), "b": unit()})
    with pytest.raises(TypeError, match="takes a dict"):
        rbd.with_levers([(("a", "reliability.alpha"), 1.0)])


# -- a non-repairable diagram -------------------------------------------------


def test_a_non_repairable_diagram_s_levers_are_its_parameters():
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("s", "c")]
        + [("a", "t"), ("b", "t"), ("c", "t")],
        {"a": W([100.0, 2.0]), "b": W([100.0, 2.0]), "c": E([0.01])},
        ccf_groups=[CCFGroup(["a", "b"], MGL(0.1))],
    )
    assert named(rbd.levers()) == [
        (("a", "b"), "alpha"),
        (("a", "b"), "beta"),
        (("a", "b"), "ccf_beta"),
        ("c", "failure_rate"),
    ]
    assert not any(lever.discrete or lever.calendar for lever in rbd.levers())
    moved = rbd.with_levers(
        {(("a", "b"), "alpha"): 200.0, (("a", "b"), "ccf_beta"): 0.05}
    )
    assert moved.reliabilities["a"].params[0] == 200.0
    assert moved.reliabilities["b"].params[0] == 200.0
    assert values(moved)[(("a", "b"), "ccf_beta")] == 0.05
    assert values(rbd)[(("a", "b"), "alpha")] == 100.0
    with pytest.raises(ValueError, match="between 0 and 1"):
        rbd.with_levers({(("a", "b"), "ccf_beta"): 1.5})


def test_non_repairable_levers_move_as_the_sensitivity_does():
    rbd = NonRepairableRBD(PAIR, {"a": W([100.0, 2.0]), "b": E([0.01])})
    step = 1e-4
    lever = rbd.levers()[0]
    h = step * lever.value
    up = rbd.with_levers({lever: lever.value + h}).sf(30.0)
    down = rbd.with_levers({lever: lever.value - h}).sf(30.0)
    reported = rbd.parameter_sensitivity(30.0, rel_step=step)["a"]["alpha"]
    assert (up - down) / (2.0 * h) == pytest.approx(reported, rel=1e-6)


# -- the private names, kept for 0.13 ----------------------------------------


def test_the_private_names_callers_used_still_work():
    rbd = RepairableRBD(
        SERIES,
        {
            "a": unit(preventive={"policy": "block", "interval": 50.0}),
            "b": unit(preventive={"policy": "block", "interval": 50.0}),
        },
    )
    old = _sensitivity.levers(rbd)
    assert all(isinstance(lever, _sensitivity.Lever) for lever in old)
    assert [(lever.key, lever.name, lever.value) for lever in old] == [
        (lever.key, lever.name, lever.value) for lever in rbd.levers()
    ]
    assert all(callable(lever.build) for lever in old)
    calendar = [_sensitivity._calendar_lever(rbd, lever) for lever in old]
    assert calendar == [lever.calendar for lever in rbd.levers()]
    spec = _sensitivity._as_spec(rbd._init_args["components"]["a"])
    assert spec["preventive"]["interval"] == 50.0
