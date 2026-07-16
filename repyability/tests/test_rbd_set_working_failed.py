"""Tests for the "force a node working / failed" logic (working_nodes /
broken_nodes) on NonRepairableRBD.sf and RepairableRBD.

Covers the core correctness identity (Birnbaum pivotal decomposition) and the
edge-case fixes: input validation, repeated-node symmetry, the zero-system-
failures crash, and broken-component accounting in the simulation.
"""

from collections.abc import Mapping

import numpy as np
import pytest
import surpyval as surv

from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.repairable_rbd import RepairableRBD

FEP = surv.FixedEventProbability


def _series():
    # s - A - B - t, R_A = 0.8, R_B = 0.6
    return NonRepairableRBD(
        [("s", "A"), ("A", "B"), ("B", "t")],
        {"A": FEP.from_params(0.2), "B": FEP.from_params(0.4)},
    )


def _parallel():
    # s - (A, B) - t, R_A = 0.8, R_B = 0.6
    return NonRepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
        {"A": FEP.from_params(0.2), "B": FEP.from_params(0.4)},
    )


def _bridge():
    edges = [
        (0, 1), (0, 3), (1, 2), (3, 4), (1, 5),
        (3, 5), (5, 2), (5, 4), (2, 6), (4, 6),
    ]  # fmt: skip
    rel = {n: FEP.from_params(1 - 0.9) for n in [1, 2, 3, 4, 5]}
    return NonRepairableRBD(edges, rel)


# --- Core correctness ------------------------------------------------------


def test_series_working_and_broken_values():
    s = _series()
    assert s.sf(1) == pytest.approx(0.48)
    assert s.sf(1, working_nodes=["A"]) == pytest.approx(0.6)  # R_B
    assert s.sf(1, broken_nodes=["A"]) == pytest.approx(0.0)


def test_parallel_working_and_broken_values():
    p = _parallel()
    assert p.sf(1) == pytest.approx(0.92)
    assert p.sf(1, working_nodes=["A"]) == pytest.approx(1.0)
    assert p.sf(1, broken_nodes=["A"]) == pytest.approx(0.6)  # R_B


@pytest.mark.parametrize("method", ["p", "c"])
def test_pivotal_decomposition_identity(method):
    # sf == R_A * sf(A working) + (1 - R_A) * sf(A broken) for every node,
    # under both the path-set and cut-set methods.
    rbd = _bridge()
    base = rbd.sf(1, method=method)
    for node in rbd.nodes:
        r = np.atleast_1d(rbd.reliabilities[node].sf(1))[0]
        working = rbd.sf(1, working_nodes=[node], method=method)
        broken = rbd.sf(1, broken_nodes=[node], method=method)
        np.testing.assert_allclose(r * working + (1 - r) * broken, base)


# --- Validation (fixes 3-6) ------------------------------------------------


def test_unknown_node_raises():
    s = _series()
    with pytest.raises(ValueError, match="Unknown node"):
        s.sf(1, working_nodes=["Z"])
    with pytest.raises(ValueError, match="Unknown node"):
        s.sf(1, broken_nodes=["Z"])


def test_node_in_both_sets_raises():
    s = _series()
    with pytest.raises(ValueError, match="both working and broken"):
        s.sf(1, working_nodes=["A"], broken_nodes=["A"])


def test_input_output_node_raises():
    s = _series()
    with pytest.raises(ValueError, match="input node"):
        s.sf(1, broken_nodes=["s"])
    with pytest.raises(ValueError, match="output node"):
        s.sf(1, working_nodes=["t"])


def test_repeated_node_symmetry_both_raise():
    # node 5 repeats node 2; neither working nor broken may target it.
    rbd = NonRepairableRBD(
        [(1, 2), (1, 3), (1, 4), (1, 5), (2, 6), (3, 6), (4, 6), (5, 6)],
        {
            2: FEP.from_params(0.2),
            3: FEP.from_params(0.1),
            4: FEP.from_params(0.15),
            5: 2,
        },
    )
    with pytest.raises(ValueError, match="repeat of node 2"):
        rbd.sf(1, working_nodes=[5])
    with pytest.raises(ValueError, match="repeat of node 2"):
        rbd.sf(1, broken_nodes=[5])


# --- sf / ff / derived paths -----------------------------------------------
# ff, reliability, unreliability and cs all delegate to sf via *args/**kwargs,
# so working_nodes/broken_nodes (and their validation) must flow through.


def test_ff_equals_one_minus_sf_with_overrides():
    p = _parallel()
    for kw in ({}, {"working_nodes": ["A"]}, {"broken_nodes": ["A"]}):
        np.testing.assert_allclose(p.ff(1, **kw), 1 - p.sf(1, **kw))


def test_reliability_unreliability_delegate_with_overrides():
    p = _parallel()
    np.testing.assert_allclose(
        p.reliability(1, broken_nodes=["A"]), p.sf(1, broken_nodes=["A"])
    )
    np.testing.assert_allclose(
        p.unreliability(1, broken_nodes=["A"]),
        1 - p.sf(1, broken_nodes=["A"]),
    )


def test_ff_method_kwarg_flows_through():
    p = _parallel()
    np.testing.assert_allclose(
        p.ff(1, broken_nodes=["A"], method="p"),
        p.ff(1, broken_nodes=["A"], method="c"),
    )


def test_override_validation_propagates_through_ff_and_friends():
    p = _parallel()
    for fn in (p.ff, p.reliability, p.unreliability):
        with pytest.raises(ValueError, match="Unknown node"):
            fn(1, working_nodes=["Z"])


def test_cs_with_broken_node_matches_surviving_branch():
    # With A broken, the parallel system reduces to B alone, so the system's
    # conditional survival equals B's conditional survival.
    w = NonRepairableRBD(
        [("s", "A"), ("s", "B"), ("A", "t"), ("B", "t")],
        {
            "A": surv.Weibull.from_params([100, 2]),
            "B": surv.Weibull.from_params([50, 1.5]),
        },
    )
    b = w.reliabilities["B"]
    expected = np.atleast_1d(b.sf(30)) / np.atleast_1d(b.sf(20))
    np.testing.assert_allclose(w.cs(10, 20, broken_nodes=["A"]), expected)


# --- RepairableRBD ---------------------------------------------------------


def _repairable_parallel():
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


def test_mean_availability_working_broken_and_validation():
    rbd = _repairable_parallel()
    av_b = 5.0 / (5.0 + 1.0)  # B alone: MTTF 5, MTTR 1
    assert rbd.mean_availability(working_nodes=["A"]) == pytest.approx(1.0)
    assert rbd.mean_availability(broken_nodes=["A"]) == pytest.approx(av_b)
    with pytest.raises(ValueError, match="Unknown node"):
        rbd.mean_availability(working_nodes=["Z"])


def test_availability_zero_system_failures_does_not_crash():
    # Fix 1: forcing a redundant node working -> the system never fails ->
    # system_failures == 0 -> previously ZeroDivisionError.
    rbd = _repairable_parallel()
    result = rbd.availability(
        t_simulation=20.0, N=200, working_nodes=["A"], seed=1
    )
    assert result["availability"].min() == pytest.approx(1.0)
    fci = result["criticalities"]["failure_criticality_index"][
        "per_system_failure"
    ]
    assert all(v == 0 for v in fci.values())

    # Every criticality/importance value stays finite (no NaN/inf) even though
    # the system never failed, so denominators like system_downtime are 0.
    def _leaves(obj):
        if isinstance(obj, Mapping):
            for v in obj.values():
                yield from _leaves(v)
        else:
            yield obj

    assert all(np.isfinite(v) for v in _leaves(result["criticalities"]))


def test_availability_broken_component_accounted_as_down():
    # Fix 2: a forced-broken component must be accounted as down (0 uptime),
    # not silently as fully up.
    rbd = _repairable_parallel()
    result = rbd.availability(
        t_simulation=20.0, N=200, broken_nodes=["A"], seed=1
    )
    assert result["node_uptime"]["A"] == 0.0


def test_availability_broken_in_series_keeps_system_down():
    # A broken component in series makes the system down the whole time, so
    # there is zero system uptime (previously over-counted at the start).
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
    rbd = RepairableRBD([("s", "A"), ("A", "B"), ("B", "t")], comps)
    result = rbd.availability(
        t_simulation=20.0, N=200, broken_nodes=["A"], seed=1
    )
    assert result["system_uptime"] == 0.0
    assert result["availability"].max() == pytest.approx(0.0)
