"""Block replacement on an upgraded component (UC-60): replaced every
interval with the life in service, and at the upgrade's time as at a
block time, the units from the upgrade on of the new life.

The references are block replacement alone (an upgrade to the same life
at a block time is nothing more), a plan of replacements at the merged
times (an upgrade to the same life between block times is one more
replacement), a memoryless unit's two-state chain between replacements
whose rate changes at the upgrade, and the simulation.
"""

import math

import numpy as np
import pytest
import surpyval as surv

from repyability import NodeState, RepairableRBD

E = surv.Exponential.from_params
W = surv.Weibull.from_params
FIXED = surv.ExactEventTime.from_params
EDGES = [("s", "a"), ("a", "t")]
LIFE = W([300, 2.0])
TASK = {"policy": "block", "interval": 150.0, "cost": 10.0}
Z = 4.0


def one(life=LIFE, repair=None, **more):
    return RepairableRBD(
        EDGES,
        {
            "a": {
                "reliability": life,
                "repairability": E([0.05]) if repair is None else repair,
                "repair_cost": 6.0,
                **more,
            }
        },
    )


def agree(mine, theirs, T, atol=1e-10, npv=True):
    t = np.linspace(0.0, T, 101)
    np.testing.assert_allclose(
        mine.point_availability(t), theirs.point_availability(t), atol=atol
    )
    a, b = mine.expected_events(T), theirs.expected_events(T)
    for key in ("node_failures", "node_preventive"):
        assert getattr(a, key)["a"] == pytest.approx(
            getattr(b, key)["a"], abs=atol
        )
    if not npv:
        return
    assert mine.expected_npv(
        T, discount_rate=1e-3, value_rate=3.0
    ).npv == pytest.approx(
        theirs.expected_npv(T, discount_rate=1e-3, value_rate=3.0).npv,
        rel=1e-10,
    )


@pytest.mark.parametrize("duration", [None, FIXED([3.0])])
def test_an_upgrade_to_the_same_life_at_a_block_time_is_block_replacement(
    duration,
):
    task = TASK if duration is None else {**TASK, "duration": duration}
    upgraded = one(preventive=task, upgrade={"at": 450.0, "reliability": LIFE})
    # Its curve and events are block replacement's; its NPV is integrated
    # on a plan's stretches, as the same plan of block times is.
    agree(upgraded, one(preventive=task), 1000.0, npv=False)
    plan = {key: value for key, value in task.items() if key != "interval"}
    plan["replace_at"] = [150.0 * k for k in range(1, 7)]
    plan.pop("policy")
    agree(upgraded, one(preventive=plan), 1000.0)


def test_between_block_times_it_is_one_more_replacement():
    task = {**TASK, "duration": FIXED([3.0])}
    upgraded = one(preventive=task, upgrade={"at": 400.0, "reliability": LIFE})
    times = sorted([150.0 * k for k in range(1, 8)] + [400.0])
    plan = one(
        preventive={
            "replace_at": times,
            "cost": 10.0,
            "duration": FIXED([3.0]),
        }
    )
    agree(upgraded, plan, 1000.0)


def chain(t, renewals, rate_at, mu, d):
    """A memoryless unit, new at 0, repaired at rate ``mu``, replaced in
    ``d`` at each of ``renewals`` if it is up (a unit in a repair
    carries on), failing at ``rate_at(s)`` in the stretch from ``s``:
    in each stretch, the units in a repair at its start follow the
    chain from down, and those replaced the chain from up once back."""

    def chain_at(rate, s, up):
        total = rate + mu
        return mu / total + (up - mu / total) * np.exp(-total * s)

    out = np.zeros_like(t)
    edges = [0.0] + list(renewals) + [math.inf]
    replaced, wait = 1.0, 0.0  # new at 0: as replaced, back at once

    def value(rate, s):
        back = np.where(
            s >= wait, replaced * chain_at(rate, np.maximum(s - wait, 0), 1), 0
        )
        return (1.0 - replaced) * chain_at(rate, s, 0.0) + back

    for begin, end in zip(edges[:-1], edges[1:]):
        rate = rate_at(begin)
        inside = (t >= begin) & (t < end)
        out[inside] = value(rate, t[inside] - begin)
        if not math.isfinite(end):
            break
        replaced, wait = float(value(rate, np.array(end - begin))), d
    return out


def test_a_memoryless_unit_against_its_chain():
    lam, new, mu, d, at = 1 / 200.0, 1 / 800.0, 1 / 10.0, 4.0, 400.0
    rbd = one(
        E([lam]),
        E([mu]),
        preventive={**TASK, "duration": FIXED([d])},
        upgrade={"at": at, "reliability": E([new])},
    )
    renewals = sorted([150.0 * k for k in range(1, 7)] + [at])
    t = np.array([0.0, 100.0, 149.9, 151.0, 160.0, 399.9, 401.0])
    t = np.concatenate([t, [420.0, 500.0, 601.0, 700.0, 899.0]])

    def rate_at(s):
        return lam if s < at else new

    np.testing.assert_allclose(
        rbd.point_availability(t),
        chain(t, renewals, rate_at, mu, d),
        atol=2e-6,
    )


def test_the_simulation_agrees_from_new_and_from_a_state():
    rbd = one(
        preventive={**TASK, "duration": FIXED([3.0])},
        upgrade={"at": 400.0, "reliability": W([900, 2.0]), "cost": 50.0},
    )
    T = 1000.0
    for seed, state in ((1, None), (2, {"a": NodeState(age=100.0)})):
        run = rbd.availability(
            T,
            mc_samples=20_000,
            seed=seed,
            state=state,
            control_variate=False,
            conditional=False,
        )
        interval = run.mean_availability_interval()
        exact = rbd.mission_availability(T, state=state)
        assert abs(interval.estimate - exact) <= Z * interval.standard_error
        cost = rbd.expected_cost(T, state=state)
        assert abs(run.cost.sample_mean - cost.mean) <= (
            Z * run.cost.sample_se
        )


@pytest.mark.parametrize(
    "more",
    [
        {"preventive": {"policy": "age", "interval": 100.0}},
        {"preventive": {**TASK, "replace_at": [100.0], "interval": None}},
    ],
)
def test_other_tasks_on_an_upgraded_part_are_refused(more):
    with pytest.raises(ValueError) as error:
        one(upgrade={"at": 400.0, "reliability": LIFE}, **more)
    assert str(error.value) == (
        "Component 'a' has an upgrade and a preventive task other than "
        "block replacement on its own interval ('policy': 'block'), which "
        "are not taken together yet: give it block replacement, or one or "
        "the other."
    )


def test_its_calendar_is_counted_from_now():
    rbd = one(preventive=TASK, upgrade={"at": 400.0, "reliability": LIFE})
    with pytest.raises(ValueError, match="leave its since_scheduled out"):
        rbd.mission_availability(
            500.0, state={"a": NodeState(age=10.0, since_scheduled=10.0)}
        )
    with pytest.raises(NotImplementedError, match="no long run"):
        rbd.total_cost(500.0)
