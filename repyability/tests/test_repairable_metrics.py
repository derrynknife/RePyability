"""Tests for the steady-state repairable-system metrics: system failure
frequency, MUT, MDT and MTBF (the Birnbaum/Vesely frequency formula), and the
simulation-side estimates on AvailabilityResult.
"""

import numpy as np
import pytest
import surpyval as surv

from repyability.non_repairable import NonRepairable
from repyability.rbd.repairable_rbd import RepairableRBD

E = surv.Exponential.from_params


def _comp(failure_rate, repair_rate):
    return {
        "reliability": E([failure_rate]),
        "repairability": E([repair_rate]),
    }


def test_single_component_identities():
    # MTTF = 5, MTTR = 1: for a single component the system metrics are the
    # component's own.
    rbd = RepairableRBD([("s", "A"), ("A", "t")], {"A": _comp(0.2, 1.0)})
    assert rbd.mean_up_time() == pytest.approx(5.0)
    assert rbd.mean_down_time() == pytest.approx(1.0)
    assert rbd.mean_time_between_failures() == pytest.approx(6.0)
    assert rbd.system_failure_frequency() == pytest.approx(1.0 / 6.0)


def test_series_exponential_mut_identity():
    # For exponential failures in series, MUT = 1 / (lambda_1 + lambda_2).
    rbd = RepairableRBD(
        [("s", "A"), ("A", "B"), ("B", "t")],
        {"A": _comp(0.2, 1.0), "B": _comp(0.5, 1.0)},
    )
    assert rbd.mean_up_time() == pytest.approx(1.0 / 0.7)
    # MTBF = MUT + MDT always.
    assert rbd.mean_time_between_failures() == pytest.approx(
        rbd.mean_up_time() + rbd.mean_down_time()
    )


def test_conditioning_forced_nodes():
    series = RepairableRBD(
        [("s", "A"), ("A", "B"), ("B", "t")],
        {"A": _comp(0.2, 1.0), "B": _comp(0.5, 1.0)},
    )
    # A always working -> the system is B alone.
    assert series.mean_up_time(working_nodes=["A"]) == pytest.approx(2.0)
    assert series.mean_down_time(working_nodes=["A"]) == pytest.approx(1.0)

    parallel = RepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
        {"A": _comp(0.2, 1.0), "B": _comp(0.5, 1.0)},
    )
    # A always working in parallel -> the system never fails.
    assert parallel.system_failure_frequency(working_nodes=["A"]) == 0.0
    assert parallel.mean_time_between_failures(working_nodes=["A"]) == float(
        "inf"
    )
    assert parallel.mean_up_time(working_nodes=["A"]) == float("inf")
    assert parallel.mean_down_time(working_nodes=["A"]) == 0.0


def test_nested_repairable_rbd_frequency_recurses():
    inner = RepairableRBD([("s", "A"), ("A", "t")], {"A": _comp(0.2, 1.0)})
    outer = RepairableRBD([("s", "X"), ("X", "t")], {"X": inner})
    assert outer.system_failure_frequency() == pytest.approx(
        inner.system_failure_frequency()
    )


def test_nonrepairable_failure_frequency():
    comp = NonRepairable(E([0.2]), E([1.0]))  # MTTF 5, MTTR 1
    assert comp.failure_frequency() == pytest.approx(1.0 / 6.0)


def test_nonrepairable_default_replacement_time_mean_availability():
    # The default time_to_replace is ExactEventTime(0), whose surpyval mean()
    # raises AttributeError; model_mean works around it, so availability is 1.
    comp = NonRepairable(E([0.2]))
    assert comp.mean_availability() == pytest.approx(1.0)
    assert comp.failure_frequency() == pytest.approx(0.2)


def test_simulation_estimates_match_analytic():
    rbd = RepairableRBD(
        [("s", "A"), ("A", "B"), ("B", "t")],
        {"A": _comp(0.2, 1.0), "B": _comp(0.5, 1.0)},
    )
    result = rbd.availability(t_simulation=100.0, N=150, seed=7)

    # System uptime + downtime account for the full simulated window.
    total = result.n_simulations * result.time_simulated_to
    assert result.system_uptime + result.system_downtime == pytest.approx(
        total
    )

    # The simulation estimates agree with the exact steady-state values.
    assert result.failure_frequency == pytest.approx(
        rbd.system_failure_frequency(), rel=0.1
    )
    assert result.mean_up_time == pytest.approx(rbd.mean_up_time(), rel=0.1)
    assert result.mean_down_time == pytest.approx(
        rbd.mean_down_time(), rel=0.1
    )


def test_result_derived_properties_zero_guards():
    # Forcing a redundant node working -> no failures -> MUT is infinite, and
    # the failure-frequency estimate is zero.
    rbd = RepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
        {"A": _comp(0.2, 1.0), "B": _comp(0.5, 1.0)},
    )
    result = rbd.availability(
        t_simulation=20.0, N=100, working_nodes=["A"], seed=3
    )
    assert result.system_failures == 0
    assert result.mean_up_time == float("inf")
    assert result.failure_frequency == 0.0
    assert np.isfinite(result.mean_down_time)
