"""Regression tests for the per-simulation component accounting in
RepairableRBD.availability().

Each component's total uptime and downtime must add up to N * t_simulation
(every simulation splits the window between up and down). The previous code
accumulated downtime against the *running cumulative* uptime rather than the
current simulation's uptime, producing wrong -- and often negative -- totals
after the first simulation.
"""

import numpy as np
import surpyval as surv

from repyability.rbd.repairable_rbd import RepairableRBD


def _rbd():
    comps = {
        "A": {
            "reliability": surv.Exponential.from_params([0.1]),
            "repairability": surv.Exponential.from_params([1.0]),
        },
        "B": {
            "reliability": surv.Exponential.from_params([0.2]),
            "repairability": surv.Exponential.from_params([1.0]),
        },
    }
    return RepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")], comps
    )


def test_component_uptime_plus_downtime_equals_window():
    N, T = 300, 20.0
    result = _rbd().availability(t_simulation=T, N=N, seed=1)
    for comp in ("A", "B"):
        up = result["components_uptime"][comp]
        down = result["components_downtime"][comp]
        assert up >= 0.0
        assert down >= 0.0
        assert np.isclose(up + down, N * T)


def test_broken_component_has_full_downtime():
    N, T = 200, 20.0
    result = _rbd().availability(
        t_simulation=T, N=N, broken_nodes=["A"], seed=1
    )
    assert result["components_uptime"]["A"] == 0.0
    assert np.isclose(result["components_downtime"]["A"], N * T)
