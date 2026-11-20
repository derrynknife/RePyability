"""Opportunistic work at a planned outage: a shutdown's
``"opportunistic"`` renews, at each of its starts, the components under
age replacement that are at least as old as their ages there, as their
own age replacement would (simulated; the exact methods refuse)."""

import json
import math

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD, SimulationChunk
from repyability.rbd import _compiled, _event_loop

E, W = surv.Exponential.from_params, surv.Weibull.from_params
LIFE = W([1000.0, 3.0])
TIMES = [800.0, 1500.0, 2500.0]
SERIES = [("s", "a"), ("a", "b"), ("b", "t")]


def line(opportunistic=None, times=TIMES, duration=10.0, **shut):
    spec = {"duration": duration, "scope": ["a", "b"]}
    if times is not None:
        spec["times"] = times
    spec.update(shut)
    if opportunistic is not None:
        spec["opportunistic"] = opportunistic
    return RepairableRBD(
        SERIES,
        {
            "a": {
                "reliability": LIFE,
                "repairability": E([0.1]),
                "replace_cost": 500.0,
                "preventive": {
                    "policy": "age",
                    "interval": 1200.0,
                    "cost": 100.0,
                    "duration": E([0.5]),
                },
            },
            "b": {
                "reliability": E([1e-4]),
                "repairability": E([0.1]),
                "repair_cost": 50.0,
            },
        },
        shutdowns={"big": spec},
    )


def test_an_age_never_reached_is_the_plan_without_the_rule():
    # Same seed, same numbers: an age of inf renews nothing.
    plain = line().availability(3000.0, mc_samples=300, seed=4)
    never = line({"a": math.inf}).availability(3000.0, mc_samples=300, seed=4)
    np.testing.assert_array_equal(plain.uptimes, never.uptimes)
    np.testing.assert_array_equal(plain.cost.samples, never.cost.samples)
    assert never.opportunistic_renewals == {"a": 0, "b": 0}
    assert [job["mean"] for job in never.outage_jobs] == [0.0, 0.0, 0.0]
    # Nor does it stop the exact methods.
    assert line({"a": math.inf}).expected_cost(3000.0).mean == pytest.approx(
        line().expected_cost(3000.0).mean, rel=1e-12
    )


def test_renewed_at_the_first_outage_exactly_when_it_never_failed():
    # At an age of the first outage's time, only a unit that has not
    # failed since new is old enough: renewed there with chance R(800).
    n = 4000
    result = line({"a": 800.0}).availability(3000.0, mc_samples=n, seed=2)
    first = result.outage_jobs[0]
    assert (first["shutdown"], first["start"]) == ("big", 800.0)
    p = float(LIFE.sf(800.0))
    assert first["standard_error"] == pytest.approx(
        math.sqrt(p * (1 - p) / n), rel=0.05
    )
    assert abs(first["mean"] - p) < 4 * first["standard_error"]
    assert first["lower"] < first["mean"] < first["upper"]
    total = sum(job["mean"] for job in result.outage_jobs) * n
    assert result.opportunistic_renewals["a"] == pytest.approx(total)


def test_the_renewals_are_charged_as_its_preventive_work():
    # Renewed early, the unit is replaced at 100 instead of failing at
    # 500 and being repaired: the run costs less, and every renewal is a
    # preventive replacement, charged as one.
    plain = line().cost(3000.0, mc_samples=1000, seed=3)
    renewed = line({"a": 300.0}).cost(3000.0, mc_samples=1000, seed=3)
    assert renewed.mean < plain.mean
    assert renewed.by_category["preventive"] > plain.by_category["preventive"]


def test_a_number_renews_every_component_under_age_replacement():
    rbd = line(500.0)
    assert rbd._outage_rules == {"big": (("a", 500.0),)}
    given = line({"a": 500.0})
    a = rbd.availability(3000.0, mc_samples=200, seed=1)
    b = given.availability(3000.0, mc_samples=200, seed=1)
    np.testing.assert_array_equal(a.uptimes, b.uptimes)


def test_a_shut_in_no_time_renews_too():
    result = line({"a": 800.0}, duration=0.0).availability(
        3000.0, mc_samples=500, seed=5
    )
    assert result.outage_jobs[0]["mean"] > 0.0


def test_the_exact_methods_refuse_naming_the_simulation():
    rbd = line({"a": 500.0})
    with pytest.raises(NotImplementedError, match=r"availability\(\)"):
        rbd.expected_cost(3000.0)
    with pytest.raises(NotImplementedError, match="opportunistically"):
        rbd.expected_npv(3000.0, value_rate=1.0)
    routes = rbd.analysis_routes()
    assert routes["expected_npv"].route == "refused"
    assert "opportunistically" in routes["expected_npv"].reason
    assert routes["availability"].route == "simulated"


def test_the_compiled_engine_leaves_them_to_the_python_loop():
    rbd = line({"a": 500.0}, times=None, period=700.0, duration=0.0)
    plan, _ = _event_loop._stream_plan(rbd, 100.0, 1, False)
    assert _compiled.unsupported(rbd, plan, None, numba=True) == "shutdowns"


@pytest.mark.parametrize(
    "opportunistic, match",
    [
        ({"a": -1.0}, "0 or more"),
        ({"c": 1.0}, "not a component"),
        ({"b": 1.0}, "'policy': 'age'"),
        ({}, "names no component"),
        ("soon", "0 or more"),
    ],
)
def test_the_rule_is_checked(opportunistic, match):
    with pytest.raises(ValueError, match=match):
        line(opportunistic)


def test_a_component_out_of_the_scope_is_refused():
    with pytest.raises(ValueError, match="in its scope"):
        line({"a": 1.0}, scope=["b"])


def test_no_component_under_age_replacement_is_refused():
    with pytest.raises(ValueError, match="no component in its scope"):
        line(1.0, scope=["b"])


def test_it_is_saved_and_read_back():
    rbd = line({"a": 500.0})
    back = RepairableRBD.from_dict(json.loads(json.dumps(rbd.to_dict())))
    assert back._outage_rules == rbd._outage_rules
    a = rbd.availability(3000.0, mc_samples=100, seed=6)
    b = back.availability(3000.0, mc_samples=100, seed=6)
    np.testing.assert_array_equal(a.uptimes, b.uptimes)
    assert json.loads(json.dumps(a.to_dict()))["outage_jobs"][0]["start"] == (
        800.0
    )


def test_chunks_merged_count_the_renewals_as_one_run():
    rbd = line({"a": 500.0})
    whole = rbd.availability(3000.0, mc_samples=60, seed=8)
    chunks = [
        rbd.simulate_chunk(3000.0, k, k + 20, seed=8) for k in (0, 20, 40)
    ]
    merged = rbd.availability_from_chunks(
        [
            SimulationChunk.from_dict(json.loads(json.dumps(c.to_dict())))
            for c in chunks
        ]
    )
    assert merged.outage_jobs == whole.outage_jobs
    assert merged.opportunistic_renewals == whole.opportunistic_renewals
