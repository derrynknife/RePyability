"""A task that restores a part to a given virtual age (a preventive
spec's ``"restore_to"``; UC-58 of the maintenance README): simulated, the
exact methods refusing it by name."""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy import integrate

from repyability import RepairableRBD
from repyability.rbd import _compiled, _event_loop

E, W = surv.Exponential.from_params, surv.Weibull.from_params
TAU, D, T = 300.0, 5.0, 900.0


def outage(life, repair, **job):
    return RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": life,
                "repairability": repair,
                "preventive": {"shutdown": "overhaul", "cost": 1.0, **job},
            }
        },
        shutdowns={"overhaul": {"times": [TAU], "duration": D}},
    )


def test_restored_to_0_is_renewed_bit_for_bit():
    life, repair = W([400.0, 2.5]), E([0.05])
    renewed = outage(life, repair).availability(T, mc_samples=400, seed=3)
    zero = outage(life, repair, restore_to=0.0).availability(
        T, mc_samples=400, seed=3
    )
    np.testing.assert_array_equal(renewed.uptimes, zero.uptimes)
    np.testing.assert_array_equal(renewed.cost.samples, zero.cost.samples)
    # And the exact methods take it, as a renewal.
    assert outage(life, repair, restore_to=0.0).expected_cost(T).mean == (
        outage(life, repair).expected_cost(T).mean
    )


def mean_up(result):
    up = np.asarray(result.uptimes, float) / T
    return float(up.mean()), float(up.std(ddof=1) / math.sqrt(up.size))


def test_a_memoryless_part_is_as_if_renewed():
    life, repair = E([1 / 400.0]), E([0.05])
    exact = outage(life, repair).mission_availability(T)
    found = outage(life, repair, restore_to=5000.0).availability(
        T, mc_samples=4000, seed=7, control_variate=False
    )
    mean, error = mean_up(found)
    assert abs(mean - exact) < 4.0 * error


@pytest.mark.parametrize("age", [100.0, 500.0])
def test_a_part_that_fails_once_at_most_against_its_survival(age):
    # A repair that never ends inside the window: the part is up until
    # its first failure. Overhauled at TAU (if up), it is down for D and
    # then lives on from the virtual age: up at t > TAU + D with chance
    # R(TAU) R(age + t - TAU - D) / R(age).
    life = W([400.0, 2.5])
    found = outage(life, E([1e-9]), restore_to=age).availability(
        T, mc_samples=4000, seed=11, control_variate=False
    )

    def up(t):
        if t < TAU:
            return float(life.sf(t))
        if t < TAU + D:
            return 0.0
        return float(life.sf(TAU) * life.sf(age + t - TAU - D) / life.sf(age))

    exact = (
        integrate.quad(up, 0.0, TAU, epsabs=1e-12)[0]
        + integrate.quad(up, TAU + D, T, epsabs=1e-12)[0]
    ) / T
    mean, error = mean_up(found)
    assert abs(mean - exact) < 4.0 * error
    # It differs from a renewal by much more than its error.
    renewed = outage(life, E([1e-9])).availability(
        T, mc_samples=4000, seed=11, control_variate=False
    )
    assert abs(mean_up(renewed)[0] - mean) > 10.0 * error


def test_the_exact_methods_refuse_naming_the_age_and_the_simulation():
    rbd = outage(W([400.0, 2.5]), E([0.05]), restore_to=500.0)
    with pytest.raises(NotImplementedError, match="virtual age of 500"):
        rbd.expected_cost(T)
    routes = rbd.analysis_routes()
    assert routes["expected_npv"].route == "refused"
    assert "availability()" in routes["expected_npv"].reason
    assert routes["availability"].route == "simulated"
    # numba's loop leaves it to Python: the dated shut first, and a task
    # that restores to an age on any calendar.
    plan, _ = _event_loop._stream_plan(rbd, 100.0, 1, False)
    assert _compiled.unsupported(rbd, plan, None, numba=True) == "shutdowns"
    own = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": W([400.0, 2.5]),
                "repairability": E([0.05]),
                "preventive": {"interval": 200.0, "restore_to": 100.0},
            }
        },
    )
    plan, _ = _event_loop._stream_plan(own, 100.0, 1, False)
    assert _compiled.unsupported(own, plan, None, numba=True) == (
        "partial restoration by preventive maintenance"
    )


@pytest.mark.parametrize(
    "job, match",
    [
        ({"restore_to": -1.0}, "0 or more"),
        ({"restore_to": "old"}, "0 or more"),
        (
            {
                "restore_to": 10.0,
                "restoration": {"model": "kijima1", "q": 0.5},
            },
            "not both",
        ),
    ],
)
def test_what_is_refused(job, match):
    with pytest.raises(ValueError, match=match):
        outage(W([400.0, 2.5]), E([0.05]), **job)


def test_it_is_saved_and_read_back():
    rbd = outage(W([400.0, 2.5]), E([0.05]), restore_to=500.0)
    back = RepairableRBD.from_dict(json.loads(json.dumps(rbd.to_dict())))
    a = rbd.availability(T, mc_samples=100, seed=2)
    b = back.availability(T, mc_samples=100, seed=2)
    np.testing.assert_array_equal(a.uptimes, b.uptimes)
