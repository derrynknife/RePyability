"""``value_plan``'s ``task_value``: a service's value on dates (UC-67 of
the schedule README) asked and answered as JSON documents.

Each answer is checked against ``task_value`` called directly, sent
through strict JSON, valued again from the request it echoes, and
validated against the shipped schema (where ``jsonschema`` is
installed).
"""

import json

import numpy as np
import pytest
import surpyval as surv

from repyability import npv_schema, value_plan
from repyability.maintenance import Prices, Restoration, UnitPlan
from repyability.rbd._contract import _strict
from repyability.rbd.results import plain
from repyability.schedule import (
    Condition,
    DutyPlan,
    Recurrence,
    Services,
    Shifts,
    Task,
    UsageForecast,
    task_value,
)
from repyability.schedule._service import effect_from_dict, effect_to_dict

LIFE = surv.Weibull.from_params([8760.0, 3.0])
PLAN = UnitPlan(LIFE, "instant", prices=Prices(repair=12000.0))
SERVICE = Task(
    "gearbox service",
    every=Recurrence.every(3, "months", "2027-04-01"),
    cost=800.0,
)
USAGE = UsageForecast.calendar("2027-01-01", "2028-01-01")
KEYWORDS = {"age": 4000.0, "discount": 0.07, "period": "quarter"}


def ask(effect=None, plan=PLAN, usage=USAGE, **inputs):
    request = {
        "schema": 1,
        "analysis": "task_value",
        "inputs": {
            "task": SERVICE.to_dict(),
            "usage": usage.to_dict(),
            **({} if effect is None else {"effect": effect_to_dict(effect)}),
            **KEYWORDS,
            **inputs,
        },
    }
    if plan is not None:
        request["diagram"] = plan.to_dict()
    return request


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


def direct(effect=None, plan=PLAN, usage=USAGE, **kwargs) -> dict:
    worth = task_value(
        SERVICE, Services(plan, effect), usage, **{**KEYWORDS, **kwargs}
    )
    return _strict(plain(worth))


def test_no_effect_is_exact_and_needs_no_plan():
    answer = answered(ask(plan=None))
    assert answer["route"]["route"] == "exact"
    assert answer["units"] == {"time": "hours", "money": "cost units"}
    assert "seed" not in answer["request"]["inputs"]
    assert answer["result"] == direct(plan=None)
    # UC-67's hand numbers: minus the services' present value.
    assert answer["result"]["total"]["value"] == pytest.approx(
        -2320.95, abs=0.005
    )
    np.testing.assert_allclose(
        answer["result"]["value"], [0.0, -786.77, -773.62, -760.55], atol=5e-3
    )
    # With the plan, and with the none effect spelt out, the same.
    assert answered(ask())["result"] == answer["result"]
    spelt = ask()
    spelt["inputs"]["effect"] = {"kind": "none"}
    assert answered(spelt)["result"] == answer["result"]


def test_a_restoration_is_the_simulated_method_s():
    effect = Restoration(q=0.5)
    answer = answered(ask(effect, seed=3, mc_samples=300))
    assert answer["route"]["route"] == "simulated"
    assert answer["result"] == direct(effect, seed=3, mc_samples=300)
    assert answer["result"]["method"] == "simulated"
    assert answer["result"]["seed"] == 3


def test_a_seed_is_drawn_and_kept_for_a_stated_effect():
    answer = answered(ask(Restoration(q=0.0), mc_samples=200))
    seed = answer["request"]["inputs"]["seed"]
    assert isinstance(seed, int)
    assert answer["result"]["seed"] == seed


def test_a_factor_of_one_is_exact_as_no_effect():
    answer = answered(ask(Restoration(q=1.0, model="kijima1")))
    assert answer["route"]["route"] == "exact"
    assert answer["result"]["value"] == direct()["value"]


def test_a_fitted_condition_through_json():
    rng = np.random.default_rng(7)
    load = np.repeat([1.0, 2.0], 40)
    x = 6000.0 * rng.weibull(1.5, load.size) / load**2
    life = surv.WeibullPH.fit(x, Z=load[:, None])
    effect = Condition(life, fresh=(1.0,), stale=(2.0,), lasts=2000.0)
    saved = effect_to_dict(effect)
    loaded = effect_from_dict(json.loads(json.dumps(saved)))
    assert (loaded.fresh, loaded.stale, loaded.lasts) == ((1.0,), (2.0,), 2e3)
    answer = answered(ask(effect, age=0.0, seed=2, mc_samples=300))
    assert answer["result"] == direct(effect, age=0.0, seed=2, mc_samples=300)
    assert answer["result"]["total"]["value_lower"] > 0.0


def test_a_duty_plan_and_its_refusal_are_answers():
    shifts = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00")).forecast(
        "2027-01-01", "2028-01-01"
    )
    answer = answered(ask(usage=shifts))
    assert answer["result"] == direct(usage=shifts)
    # A virtual age on the shifts is followed in operating time.
    restored = answered(
        ask(Restoration(q=0.5), usage=shifts, seed=1, mc_samples=200)
    )
    assert restored["result"] == direct(
        Restoration(q=0.5), usage=shifts, seed=1, mc_samples=200
    )
    # A standby group on the shifts is not followed.
    pair = UnitPlan(
        LIFE,
        surv.Exponential.from_params([0.1]),
        prices=Prices(repair=12000.0),
        standby={"units": 2, "k": 1},
    )
    refused = answered(
        ask(Restoration(q=0.5), plan=pair, usage=shifts, seed=1)
    )
    assert "result" not in refused
    assert refused["route"]["route"] == "refused"
    assert "standby group" in refused["refused"]["reason"]
    assert refused["refused"]["routes"] == {"task_value": "refused"}


def test_usage_by_its_start_end_and_plan_alone():
    plan = DutyPlan(Shifts("Mon-Fri", "06:00", "22:00"))
    shifts = plan.forecast("2027-01-01", "2028-01-01")
    request = ask(plan=None)
    request["inputs"]["usage"] = {
        "start": "2027-01-01",
        "end": "2028-01-01",
        "plan": plan.to_dict(),
    }
    assert answered(request)["result"] == direct(plan=None, usage=shifts)
    request["inputs"]["usage"] = {"start": "2027-01-01", "end": "2028-01-01"}
    assert answered(request)["result"] == direct(plan=None)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"inputs": {"speed": 1}}, "no input(s) ['speed']"),
        ({"inputs": {"does": {}}}, "no input(s) ['does']"),
        ({"inputs": {"task": None}}, "task must be a Task"),
        ({"drop": "usage"}, "needs the input 'usage'"),
        ({"inputs": {"usage": {"start": "2027-01-01"}}}, "start and end"),
        (
            {"inputs": {"usage": {**USAGE.to_dict(), "when": 1}}},
            "usage takes no key(s) ['when']",
        ),
        ({"inputs": {"effect": {"kind": "fresh"}}}, "an effect is None"),
        (
            {"inputs": {"effect": {"kind": "restoration", "process": {}}}},
            "takes no key(s) ['process']",
        ),
        (
            {"inputs": {"effect": {"kind": "restoration", "q": 0.5}}},
            "needs the unit's plan",
        ),
        ({"diagram": {"kind": "rbd"}}, "the unit's plan, a UnitPlan"),
    ],
)
def test_a_request_is_checked(change, message):
    request = ask(plan=None)
    request["inputs"].update(change.get("inputs", {}))
    request["inputs"].pop(change.get("drop", ""), None)
    if "diagram" in change:
        request["diagram"] = change["diagram"]
    with pytest.raises(ValueError) as error:
        value_plan(request)
    assert message in str(error.value)
    assert str(error.value).startswith("value_plan: ")


def test_other_analyses_still_need_a_diagram():
    request = {"schema": 1, "analysis": "expected_npv", "inputs": {}}
    with pytest.raises(ValueError, match="needs 'diagram'"):
        value_plan(request)


def test_a_fitted_process_has_no_plain_form():
    effect = Restoration(process=object())
    with pytest.raises(NotImplementedError, match="Kijima model and factor"):
        effect_to_dict(effect)
    assert effect_to_dict(None) is None
    saved = effect_to_dict(Restoration(to_age=100.0, replace_every=3))
    assert saved == {
        "kind": "restoration",
        "model": "kijima2",
        "to_age": 100.0,
        "replace_every": 3,
    }
    assert effect_from_dict(saved) == Restoration(
        to_age=100.0, replace_every=3
    )
