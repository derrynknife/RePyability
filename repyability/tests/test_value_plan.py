"""``value_plan``: an NPV question asked and answered as JSON documents.

Each answer is checked against the method called directly, sent through
strict JSON, valued again from the request it echoes (the same answer,
number for number), and validated against the shipped schema (where
``jsonschema`` is installed).
"""

import json

import pytest
import surpyval as surv

from repyability import (
    NodeState,
    RepairableRBD,
    Timeline,
    ValueSchedule,
    npv_schema,
    value_plan,
)
from repyability.rbd._contract import _strict
from repyability.rbd.results import plain

E = surv.Exponential.from_params
W = surv.Weibull.from_params
FIXED = surv.ExactEventTime.from_params
EDGES = [("s", "a"), ("a", "b"), ("b", "t")]


def line(a=None, **options):
    return RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": W([300, 2.0]),
                "repairability": E([0.05]),
                "repair_cost": 6.0,
                **(a or {}),
            },
            "b": {
                "reliability": E([1e-3]),
                "repairability": E([0.1]),
                "repair_cost": 3.0,
            },
        },
        downtime_cost_rate=1.0,
        **options,
    )


def ask(rbd, analysis, inputs, **more):
    return {
        "schema": 1,
        "diagram": rbd.to_dict(),
        "analysis": analysis,
        "inputs": inputs,
        **more,
    }


def answered(request) -> dict:
    """The answer, through strict JSON, valued again from its request,
    and against the schema."""
    answer = value_plan(json.loads(json.dumps(request)))
    text = json.dumps(answer, allow_nan=False)
    assert value_plan(json.loads(text)["request"]) == answer
    try:
        import jsonschema
    except ImportError:
        return answer
    schema = npv_schema()
    jsonschema.validate(request, {**schema, "$ref": "#/$defs/request"})
    jsonschema.validate(answer, {**schema, "$ref": "#/$defs/answer"})
    return answer


def test_an_expected_npv_with_every_input_is_the_method_s():
    rbd = line({"preventive": {"policy": "block", "interval": 150.0}})
    inputs = {
        "horizon": 900.0,
        "discount_rate": 1e-3,
        "value_rate": {"times": [0.0, 400.0], "rates": [2.0, 2.5]},
        "periods": 3,
        "escalation": {"repair": 2e-4, "a": 1e-4},
        "rates": "real",
        "inflation": 3e-4,
        "rate_sensitivity": True,
        "salvage": {"a": 20.0},
        "state": {"a": {"age": 50.0}},
    }
    answer = answered(ask(rbd, "expected_npv", inputs))
    direct = rbd.expected_npv(
        900.0,
        discount_rate=1e-3,
        value_rate=ValueSchedule([0.0, 400.0], [2.0, 2.5]),
        periods=3,
        escalation={"repair": 2e-4, "a": 1e-4},
        rates="real",
        inflation=3e-4,
        rate_sensitivity=True,
        salvage={"a": 20.0},
        state={"a": NodeState(age=50.0)},
    )
    assert answer["result"]["npv"] == direct.npv
    assert answer["route"]["route"] == "numerical"
    assert answer["units"] == {"time": None, "money": "cost units"}


def test_a_part_at_several_shutdowns_and_an_upgrade_are_valued():
    shut = line(
        {"preventive": {"shutdown": ["stop", "outage"], "cost": 2.0}},
        shutdowns={
            "stop": {"period": 100.0, "duration": 1.0, "scope": ["a"]},
            "outage": {"times": [250.0], "duration": 10.0, "cost": 30.0},
        },
    )
    answer = answered(ask(shut, "expected_npv", {"horizon": 600.0}))
    assert answer["result"]["npv"] == shut.expected_npv(600.0).npv
    upgraded = line(
        {
            "preventive": {"policy": "block", "interval": 200.0},
            "upgrade": {"at": 300.0, "reliability": W([900, 2.0]), "cost": 9},
        }
    )
    answer = answered(
        ask(
            upgraded,
            "compare_npv",
            {"horizon": 800.0, "value_rate": 2.0, "mc_samples": 500},
            other=line().to_dict(),
        )
    )
    # A seed was drawn, and the request keeps it.
    seed = answer["request"]["inputs"]["seed"]
    direct = upgraded.compare_npv(
        line(), 800.0, value_rate=2.0, mc_samples=500, seed=seed
    )
    assert answer["result"]["mean"] == direct.mean


def test_a_spread_and_a_plan_against_what_happened():
    task = {"policy": "age", "interval": 200.0, "cost": 4.0}
    rbd = line({"preventive": {**task, "duration": FIXED([2.0])}})
    spread = answered(
        ask(
            rbd,
            "npv_distribution",
            {
                "horizon": 500.0,
                "value_rate": 2.0,
                "mc_samples": 300,
                "seed": 4,
            },
        )
    )
    direct = rbd.npv_distribution(
        500.0, value_rate=2.0, mc_samples=300, seed=4
    )
    assert spread["result"]["mean"] == direct.mean
    logs = {
        "a": Timeline([120.0, 130.0], 200.0),
        "b": Timeline([], 200.0),
    }
    actual = answered(
        ask(
            rbd,
            "plan_vs_actual",
            {
                "timelines": {k: v.to_dict() for k, v in logs.items()},
                "now": 200.0,
                "horizon": 600.0,
                "discount_rate": 1e-3,
            },
        )
    )
    direct = rbd.plan_vs_actual(logs, 200.0, 600.0, discount_rate=1e-3)
    assert actual["result"] == _strict(plain(direct))
    uncertain = answered(
        ask(
            rbd,
            "expected_npv_uncertainty",
            {"horizon": 500.0, "value_rate": 2.0, "n_draws": 20, "seed": 1},
        )
    )
    # Models given by their parameters have no uncertainty to draw from
    # (and a JSON request cannot give one): refused, saying what to give.
    assert uncertain["refused"]["reason"].startswith(
        "No component's model is a surpyval fit"
    )


def test_a_refusal_is_an_answer_that_says_what_to_run():
    rbd = line(
        {"preventive": {"policy": "age", "interval": 200.0, "cost": 4.0}},
        shutdowns={
            "outage": {
                "times": [150.0],
                "duration": 5.0,
                "opportunistic": {"a": 50.0},
            }
        },
    )
    answer = answered(ask(rbd, "expected_npv", {"horizon": 400.0}))
    assert "result" not in answer
    assert "by simulation" in answer["refused"]["reason"]
    assert answer["route"]["route"] == "refused"
    assert answer["refused"]["routes"]["expected_npv"] == "refused"


@pytest.mark.parametrize(
    "change, message",
    [
        ({"when": 1}, "unknown key(s) ['when']"),
        ({"schema": 2}, "schema must be 1"),
        ({"analysis": "mean_availability"}, "analysis must be one of"),
        ({"inputs": {"horizon": 9.0, "speed": 1}}, "no input(s) ['speed']"),
        ({"inputs": {"horizon": 9.0, "state": {"z": {}}}}, "'z' is not a"),
        ({"other": {}}, "only compare_npv takes 'other'"),
    ],
)
def test_a_request_is_checked_by_name(change, message):
    request = {**ask(line(), "expected_npv", {"horizon": 9.0}), **change}
    with pytest.raises(ValueError) as error:
        value_plan(request)
    assert message in str(error.value)
    assert str(error.value).startswith("value_plan: ")


def test_the_schema_ships_with_the_package():
    schema = npv_schema()
    assert set(schema["$defs"]) >= {"request", "answer"}
    analyses = schema["$defs"]["analysis"]["enum"]
    from repyability.rbd._contract import ANALYSES

    assert tuple(analyses) == ANALYSES
