"""A tested component with an inspection offset, started from a state
with no ``since_scheduled``: on its calendar as from new, first tested at
its offset and every interval after it, known up at 0. The exact methods
and the simulation agree, for each kind of tested component (instant
tests and repairs of an exponential life, any life, and timed repairs).
"""

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD

E = surv.Exponential.from_params
W = surv.Weibull.from_params
X = surv.ExactEventTime.from_params

ONE = [("s", "a"), ("a", "t")]
LAM = 1.0 / 300.0
INSPECTION = {"interval": 50.0, "offset": 20.0}
KINDS = {
    "closed": {
        "reliability": E([LAM]),
        "repairability": "instant",
        "inspection": INSPECTION,
    },
    "life": {
        "reliability": W([300.0, 2.0]),
        "repairability": "instant",
        "inspection": INSPECTION,
    },
    "unit": {
        "reliability": E([LAM]),
        "repairability": X([3.0]),
        "inspection": INSPECTION,
    },
}
AGED = {"a": NodeState(age=100.0)}


def _rbd(kind):
    return RepairableRBD(ONE, {"a": KINDS[kind]})


def test_the_mission_availability_by_hand():
    # Known up at 0; tested at 20, 70, ..., 970: 20 before the first
    # test, 19 whole intervals, then 30.
    def up(length):
        return -np.expm1(-LAM * length) / LAM

    by_hand = (up(20.0) + 19 * up(50.0) + up(30.0)) / 1000.0
    rbd = _rbd("closed")
    assert rbd.mission_availability(1000.0, state=AGED) == pytest.approx(
        by_hand, rel=1e-12
    )
    assert by_hand == pytest.approx(0.922951, abs=1e-6)
    # The same as from new: a memoryless life, on the same calendar.
    assert rbd.mission_availability(1000.0, state=AGED) == pytest.approx(
        rbd.mission_availability(1000.0), rel=1e-12
    )


def test_the_first_test_is_at_the_offset():
    x = np.array([10.0, 24.0])
    # Instant tests and repairs: renewed at 20, up 4 since.
    closed = _rbd("closed").point_availability(x, state=AGED)
    np.testing.assert_allclose(
        closed, np.exp(-LAM * np.array([10.0, 4.0])), rtol=1e-12
    )
    # Any life: one that failed by 20 is new from then.
    model = W([300.0, 2.0])
    kept = model.sf(124.0) / model.sf(100.0)
    renewed = (1.0 - model.sf(120.0) / model.sf(100.0)) * model.sf(4.0)
    life = _rbd("life").point_availability(x, state=AGED)
    np.testing.assert_allclose(
        life,
        [model.sf(110.0) / model.sf(100.0), kept + renewed],
        rtol=1e-9,
    )
    # A repair of 3: one found failed at 20 is up again, new, at 23.
    unit = _rbd("unit").point_availability(x, state=AGED)
    np.testing.assert_allclose(
        unit,
        [
            np.exp(-LAM * 10.0),
            np.exp(-LAM * 24.0) - np.expm1(-LAM * 20.0) * np.exp(-LAM),
        ],
        rtol=1e-9,
    )


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize(
    "start",
    [NodeState(age=100.0), NodeState(age=0.0)],
    ids=["aged", "new"],
)
def test_exact_matches_the_simulation(kind, start):
    rbd = _rbd(kind)
    state = {"a": start}
    exact = rbd.mission_availability(1000.0, state=state)
    run = rbd.availability(
        1000.0,
        mc_samples=20_000,
        seed=11,
        state=state,
        control_variate=False,
        conditional=False,
    )
    interval = run.mean_availability_interval(0.95)
    assert interval.method == "simulated"
    assert abs(run.mean_availability - exact) < 4.0 * interval.standard_error


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("age", [0.0, 100.0])
def test_tested_at_0_is_on_the_calendar_from_0(kind, age):
    # since_scheduled=0: tested (and found working) at 0, so tested
    # next after an interval, at 50, 100, ...: the offset no longer
    # places it, in the exact methods as in the simulation.
    rbd = _rbd(kind)
    state = {"a": NodeState(age=age, since_scheduled=0.0)}
    exact = rbd.mission_availability(1000.0, state=state)
    run = rbd.availability(
        1000.0,
        mc_samples=20_000,
        seed=11,
        state=state,
        control_variate=False,
        conditional=False,
    )
    interval = run.mean_availability_interval(0.95)
    assert abs(run.mean_availability - exact) < 4.0 * interval.standard_error
    if kind == "closed":
        whole = -np.expm1(-LAM * 50.0) / (LAM * 50.0)
        assert exact == pytest.approx(whole, rel=1e-12)
    if kind != "life":
        # A memoryless life: its age does not matter, only its calendar.
        other = {"a": NodeState(age=100.0 - age, since_scheduled=0.0)}
        assert exact == pytest.approx(
            rbd.mission_availability(1000.0, state=other), rel=1e-9
        )
