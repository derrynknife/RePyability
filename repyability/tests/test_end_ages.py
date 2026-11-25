"""Each unit's age at the end of each run (a salvage by age), as the
Python loop keeps it and as worked out from the histories for numba's
loop (``_timeline_runs._ages_at_end``), to the bit; a group's units each
by the Python loop alone."""

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _event_loop, _timeline_runs

E = surv.Exponential.from_params
W = surv.Weibull.from_params
EDGES = [("s", "a"), ("a", "b"), ("b", "t")]


def kinds():
    life = {"reliability": W([1000.0, 2.0]), "repairability": E([0.05])}
    yield "plain", RepairableRBD(EDGES, {"a": dict(life), "b": dict(life)})
    yield "age task in no time", RepairableRBD(
        EDGES,
        {
            "a": {**life, "preventive": {"interval": 600.0}},
            "b": dict(life, repairability="instant"),
        },
    )
    yield "block task taking time", RepairableRBD(
        EDGES,
        {
            "a": {
                **life,
                "preventive": {
                    "interval": 500.0,
                    "policy": "block",
                    "duration": E([0.2]),
                },
            },
            "b": dict(life),
        },
    )
    yield "repair crews", RepairableRBD(
        EDGES, {"a": dict(life), "b": dict(life)}, repair_crews=1
    )
    # Hidden failures: a unit failed unseen is down (NaN), and a test
    # that takes a working unit off line does not age it.
    yield "tests in no time", RepairableRBD(
        EDGES,
        {"a": {**life, "inspection": {"interval": 300.0}}, "b": dict(life)},
    )
    tested = {**life, "inspection": {"interval": 300.0, "duration": E([0.5])}}
    yield "tests taking time", RepairableRBD(
        EDGES, {"a": dict(tested), "b": dict(life)}
    )
    yield "tests taking time, repair crews", RepairableRBD(
        EDGES, {"a": dict(tested), "b": dict(tested)}, repair_crews=1
    )


def ages(rbd, engine, horizon=5000.0, runs=200, seed=3):
    return _timeline_runs.simulate(
        rbd,
        horizon,
        runs,
        seed,
        None,
        None,
        False,
        engine,
        None,
        0,
        None,
        hidden=True,
        ages=True,
    )


@pytest.mark.parametrize("kind", [k[0] for k in kinds()])
def test_the_histories_give_the_python_loop_s_ages(kind):
    # From the Python loop's own histories, the ages it kept, to the bit.
    rbd = dict(kinds())[kind]
    runs, _, kept = ages(rbd, "python")
    tally = _event_loop._run(
        rbd, 5000.0, set(), set(), "p", 200, False, 3, histories=True
    )
    parts, _ = tally.histories.data(5000.0)
    derived = _timeline_runs._ages_at_end(
        parts,
        tally.histories.counted(),
        5000.0,
        [None] * len(parts),
        [node in rbd._inspection for node in rbd.components],
    )
    np.testing.assert_array_equal(derived, kept)
    assert np.isfinite(kept).any() and (kept[np.isfinite(kept)] >= 0).all()


@pytest.mark.parametrize("kind", [k[0] for k in kinds()])
def test_numba_s_ages_are_the_python_loop_s(kind):
    pytest.importorskip("numba")
    rbd = dict(kinds())[kind]
    py = ages(rbd, "python")
    nb = ages(rbd, "numba")
    assert nb[0].engine == "numba"
    np.testing.assert_array_equal(nb[2], py[2])


def test_a_group_s_units_are_laid_out_after_the_other_columns():
    # A standby group of three: three columns, the Python loop's; the
    # histories leave them NaN (numba's loop does not run such a salvage,
    # _timeline_runs._python_ages) and give the plain unit's to the bit.
    group = {
        "reliability": W([1000.0, 2.0]),
        "repairability": E([0.05]),
        "standby": {"units": 3, "k": 1, "dormancy_factor": 0.5},
    }
    life = {"reliability": W([1000.0, 2.0]), "repairability": E([0.05])}
    rbd = RepairableRBD(EDGES, {"a": group, "b": life})
    assert _timeline_runs.age_slots(rbd) == [("a", 3), ("b", 1)]
    _, _, kept = ages(rbd, "python")
    assert kept.shape == (200, 4)
    tally = _event_loop._run(
        rbd, 5000.0, set(), set(), "p", 200, False, 3, histories=True
    )
    parts, _ = tally.histories.data(5000.0)
    derived = _timeline_runs._ages_at_end(
        parts, tally.histories.counted(), 5000.0, [None, None], None, [3, 1]
    )
    assert np.isnan(derived[:, :3]).all()
    np.testing.assert_array_equal(derived[:, 3], kept[:, 3])
    # Each unit up is no older than the run, and a spare's used life is
    # at most half the time it waited.
    up = kept[:, :3][np.isfinite(kept[:, :3])]
    assert up.size and (up >= 0.0).all() and (up <= 5000.0).all()


def test_a_duty_plan_s_operating_age_on_both_engines():
    # A life on a duty plan that changes: numba's loop runs it, and its
    # age from the histories is its operating time since it was put in,
    # to the bit of the Python loop's.
    from repyability.schedule import DutySchedule

    plan = DutySchedule(
        "hours", (0.0, 1000.0, 2000.0, 3000.0), (1.0, 0.25, 0.0, 0.5)
    )
    life = {"reliability": W([400.0, 2.0]), "repairability": E([0.05])}
    rbd = RepairableRBD(EDGES, {"a": {**life, "duty": plan}, "b": life})
    _, _, kept = ages(rbd, "python")
    tally = _event_loop._run(
        rbd, 5000.0, set(), set(), "p", 200, False, 3, histories=True
    )
    parts, _ = tally.histories.data(5000.0)
    clocks = [rbd.components["a"].reliability, None]
    derived = _timeline_runs._ages_at_end(
        parts,
        tally.histories.counted(),
        5000.0,
        [None, None],
        None,
        None,
        clocks,
    )
    np.testing.assert_array_equal(derived, kept)
    # (operating time: at most the plan's hours to the end)
    assert np.nanmax(kept[:, 0]) <= float(clocks[0].operating(5000.0))
    pytest.importorskip("numba")
    nb = ages(rbd, "numba")
    assert nb[0].engine == "numba"
    np.testing.assert_array_equal(nb[2], kept)
