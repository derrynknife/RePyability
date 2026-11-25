"""Each unit's age at the end of each run (a salvage by age), as the
Python loop keeps it and as worked out from the histories for numba's
loop (``_timeline_runs._ages_at_end``), to the bit."""

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
        parts, tally.histories.counted(), 5000.0, [None] * len(parts)
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
