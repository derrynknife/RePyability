"""
Tests NonRepairableRBD's importance methods.

Uses pytest fixtures located in conftest.py in the tests/ directory.
"""

import itertools
import math
import warnings
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import RepairableRBD
from repyability.rbd.non_repairable_rbd import NonRepairableRBD


# Test birnbaum_importance() w/ composite NonRepairableRBD
def test_rbd_birnbaum_importance(rbd1: NonRepairableRBD):
    t = 2
    birnbaum_importance_dict = rbd1.birnbaum_importance(t)
    assert len(birnbaum_importance_dict) == 3
    assert (
        pytest.approx(
            rbd1.reliabilities["valve"].sf(t)
            - rbd1.reliabilities["pump2"].sf(t)
            * rbd1.reliabilities["valve"].sf(t)
        )
        == birnbaum_importance_dict["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.reliabilities["valve"].sf(t)
            - rbd1.reliabilities["pump1"].sf(t)
            * rbd1.reliabilities["valve"].sf(t)
        )
        == birnbaum_importance_dict["pump2"]
    )
    assert (
        pytest.approx(
            1
            - rbd1.reliabilities["pump1"].ff(t)
            * rbd1.reliabilities["pump2"].ff(t)
        )
        == birnbaum_importance_dict["valve"]
    )


# Test improvement_potential()
def test_rbd_improvement_potential(rbd1: NonRepairableRBD):
    t = 2
    improvement_potential = rbd1.improvement_potential(t)
    assert len(improvement_potential) == 3
    assert (
        pytest.approx(rbd1.reliabilities["valve"].sf(t) - rbd1.sf(t))
        == improvement_potential["pump1"]
    )
    assert (
        pytest.approx(rbd1.reliabilities["valve"].sf(t) - rbd1.sf(t))
        == improvement_potential["pump2"]
    )
    assert (
        pytest.approx(
            (
                1
                - rbd1.reliabilities["pump1"].ff(t)
                * rbd1.reliabilities["pump2"].ff(t)
            )
            - rbd1.sf(t)
        )
        == improvement_potential["valve"]
    )


# Test risk_achievement_worth()
def test_rbd_risk_achievement_worth(rbd1: NonRepairableRBD):
    t = 2
    raw = rbd1.risk_achievement_worth(t)
    assert len(raw) == 3
    assert (
        pytest.approx(
            (
                1
                - rbd1.reliabilities["pump2"].sf(t)
                * rbd1.reliabilities["valve"].sf(t)
            )
            / rbd1.ff(t)
        )
        == raw["pump1"]
    )
    assert (
        pytest.approx(
            (
                1
                - rbd1.reliabilities["pump1"].sf(t)
                * rbd1.reliabilities["valve"].sf(t)
            )
            / rbd1.ff(t)
        )
        == raw["pump2"]
    )
    assert pytest.approx(1 / rbd1.ff(t)) == raw["valve"]


# Test risk_reduction_worth()
def test_rbd_risk_reduction_worth(rbd1: NonRepairableRBD):
    t = 2
    rrw = rbd1.risk_reduction_worth(t)
    assert len(rrw) == 3
    assert (
        pytest.approx(rbd1.ff(t) / rbd1.reliabilities["valve"].ff(t))
        == rrw["pump1"]
    )
    assert (
        pytest.approx(
            pytest.approx(rbd1.ff(t) / rbd1.reliabilities["valve"].ff(t))
        )
        == rrw["pump2"]
    )
    assert (
        pytest.approx(
            rbd1.ff(t)
            / (
                rbd1.reliabilities["pump1"].ff(t)
                * rbd1.reliabilities["pump2"].ff(t)
            )
        )
        == rrw["valve"]
    )


# Test criticality_importance()
def test_rbd_criticality_importance(rbd1: NonRepairableRBD):
    t = 2
    criticality_importance = rbd1.criticality_importance(t, kind="success")
    assert len(criticality_importance) == 3
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                rbd1.reliabilities["valve"].sf(t)
                - rbd1.reliabilities["pump2"].sf(t)
                * rbd1.reliabilities["valve"].sf(t)
            )
            # Correction factor:
            * (rbd1.reliabilities["pump1"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["pump1"]
    )
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                rbd1.reliabilities["valve"].sf(t)
                - rbd1.reliabilities["pump1"].sf(t)
                * rbd1.reliabilities["valve"].sf(t)
            )
            # Correction factor:
            * (rbd1.reliabilities["pump2"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["pump2"]
    )
    assert (
        pytest.approx(
            # Birnbaum importance:
            (
                1
                - rbd1.reliabilities["pump1"].ff(t)
                * rbd1.reliabilities["pump2"].ff(t)
            )
            # Correction factor:
            * (rbd1.reliabilities["valve"].sf(t) / rbd1.sf(t))
        )
        == criticality_importance["valve"]
    )


def test_rbd_failure_criticality_importance(rbd1: NonRepairableRBD):
    t = 2
    ci = rbd1.criticality_importance(t)
    q = {node: rbd1.reliabilities[node].ff(t) for node in ci}
    r = {node: 1 - q[node] for node in ci}
    system_ff = 1 - rbd1.sf(t)
    # Birnbaum importance * node unreliability / system unreliability
    assert ci["pump1"] == pytest.approx(
        r["valve"] * q["pump2"] * q["pump1"] / system_ff
    )
    assert ci["pump2"] == pytest.approx(
        r["valve"] * q["pump1"] * q["pump2"] / system_ff
    )
    assert ci["valve"] == pytest.approx(
        (1 - q["pump1"] * q["pump2"]) * q["valve"] / system_ff
    )


def _series(unreliabilities):
    nodes = list(unreliabilities)
    chain = ["s", *nodes, "t"]
    return NonRepairableRBD(
        list(zip(chain, chain[1:])),
        {
            node: FixedEventProbability.from_params(q)
            for node, q in unreliabilities.items()
        },
    )


def test_failure_criticality_ranks_nodes_in_series():
    q = {"a": 0.1, "b": 0.2, "c": 0.3}
    rbd = _series(q)
    system_ff = 1 - 0.9 * 0.8 * 0.7
    # In series node i is critical exactly when every other node works.
    expected = {
        "a": 0.1 * 0.8 * 0.7 / system_ff,
        "b": 0.2 * 0.9 * 0.7 / system_ff,
        "c": 0.3 * 0.9 * 0.8 / system_ff,
    }
    ci = rbd.criticality_importance()
    for node in q:
        assert ci[node] == pytest.approx(expected[node], rel=1e-12)
    # The success-oriented form is exactly 1 for every node in series.
    success = rbd.criticality_importance(kind="success")
    for node in q:
        assert success[node] == pytest.approx(1.0, rel=1e-12)


def _enumerated_criticality(edges, probabilities):
    """Both criticality importances by brute force: every combination of
    node states, the system working when the working nodes connect the
    input "s" to the output "t"."""
    nodes = sorted(probabilities)
    graph = nx.DiGraph(edges)

    def works(up):
        return nx.has_path(graph.subgraph({"s", "t", *up}), "s", "t")

    # Integer zeros keep exact (Fraction) probabilities exact.
    system_works = 0
    critical_failed = dict.fromkeys(nodes, 0)
    critical_working = dict.fromkeys(nodes, 0)
    for states in itertools.product((True, False), repeat=len(nodes)):
        up = {node for node, state in zip(nodes, states) if state}
        weight = math.prod(
            probabilities[node] if state else 1 - probabilities[node]
            for node, state in zip(nodes, states)
        )
        if works(up):
            system_works += weight
        for node in nodes:
            if works(up | {node}) and not works(up - {node}):
                if node in up:
                    critical_working[node] += weight
                else:
                    critical_failed[node] += weight
    return (
        {node: critical_failed[node] / (1 - system_works) for node in nodes},
        {node: critical_working[node] / system_works for node in nodes},
    )


BRIDGE = [
    ("s", "a"),
    ("s", "b"),
    ("a", "c"),
    ("b", "d"),
    ("a", "e"),
    ("b", "e"),
    ("e", "c"),
    ("e", "d"),
    ("c", "t"),
    ("d", "t"),
]


def test_criticality_matches_brute_force_on_a_bridge():
    probabilities = {"a": 0.9, "b": 0.75, "c": 0.6, "d": 0.95, "e": 0.8}
    rbd = NonRepairableRBD(
        BRIDGE,
        {
            node: FixedEventProbability.from_params(1 - p)
            for node, p in probabilities.items()
        },
    )
    failure, success = _enumerated_criticality(BRIDGE, probabilities)
    ci = rbd.criticality_importance()
    ci_success = rbd.criticality_importance(kind="success")
    for node in probabilities:
        assert ci[node] == pytest.approx(failure[node], rel=1e-12)
        assert ci_success[node] == pytest.approx(success[node], rel=1e-12)


def test_repairable_criticality_matches_brute_force_on_a_bridge():
    E = surv.Exponential.from_params
    failure_rates = {"a": 0.2, "b": 0.5, "c": 1.0, "d": 0.1, "e": 0.3}
    rbd = RepairableRBD(
        BRIDGE,
        {
            node: {"reliability": E([rate]), "repairability": E([2.0])}
            for node, rate in failure_rates.items()
        },
    )
    # Long-run availability MTTF / (MTTF + MTTR) = mu / (lambda + mu)
    availability = {
        node: 2.0 / (rate + 2.0) for node, rate in failure_rates.items()
    }
    failure, success = _enumerated_criticality(BRIDGE, availability)
    ci = rbd.criticality_importance()
    ci_success = rbd.criticality_importance(kind="success")
    for node in failure_rates:
        assert isinstance(ci[node], float)
        assert ci[node] == pytest.approx(failure[node], rel=1e-12)
        assert ci_success[node] == pytest.approx(success[node], rel=1e-12)


def test_failure_criticality_keeps_its_precision():
    # A parallel pair, each failing with probability 1e-9: the system
    # unreliability, 1e-18, is lost in 1 - R_sys, but each node is critical
    # in every system failure.
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": FixedEventProbability.from_params(1e-9),
            "b": FixedEventProbability.from_params(1e-9),
        },
    )
    ci = rbd.criticality_importance()
    assert ci["a"] == pytest.approx(1.0, rel=1e-12)
    assert ci["b"] == pytest.approx(1.0, rel=1e-12)
    # A bridge failing with probability ~1e-12, against exact rational
    # arithmetic: 1 - R_sys would keep only ~4 significant figures.
    unreliabilities = {"a": 1e-6, "b": 2e-6, "c": 3e-6, "d": 1.5e-6, "e": 5e-7}
    rbd = NonRepairableRBD(
        BRIDGE,
        {
            node: FixedEventProbability.from_params(q)
            for node, q in unreliabilities.items()
        },
    )
    exact, _ = _enumerated_criticality(
        BRIDGE, {node: 1 - Fraction(q) for node, q in unreliabilities.items()}
    )
    ci = rbd.criticality_importance()
    for node in unreliabilities:
        assert ci[node] == pytest.approx(float(exact[node]), rel=1e-8)


def test_criticality_is_nan_where_it_is_undefined():
    rbd = NonRepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {
            "a": surv.Weibull.from_params([10, 2]),
            "b": surv.Weibull.from_params([20, 2]),
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Nothing can have failed at t = 0: the failure form is undefined.
        at_zero = rbd.criticality_importance(np.array([0.0, 5.0]))
        # With a working, the system cannot fail.
        perfect = rbd.criticality_importance(5.0, working_nodes=["a"])
        # With both broken, it cannot work.
        dead = rbd.criticality_importance(
            5.0, broken_nodes=["a", "b"], kind="success"
        )
    for node in ("a", "b"):
        assert np.isnan(at_zero[node][0]) and np.isfinite(at_zero[node][1])
        assert np.isnan(perfect[node])
        assert np.isnan(dead[node])
    # Each node is critical in every failure of the pair.
    assert at_zero["a"][1] == pytest.approx(1.0)


def test_criticality_kind_is_validated():
    rbd = _series({"a": 0.1, "b": 0.2})
    with pytest.raises(ValueError, match="kind must be 'failure' or"):
        rbd.criticality_importance(kind="both")
    with pytest.raises(ValueError, match="kind must be 'failure' or"):
        rbd.importances_given_state(kind="Failure")
    E = surv.Exponential.from_params
    repairable = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {"a": {"reliability": E([0.2]), "repairability": E([1.0])}},
    )
    with pytest.raises(ValueError, match="kind must be 'failure' or"):
        repairable.criticality_importance(kind="success-oriented")


def test_failure_criticality_over_time():
    t = np.array([1.0, 2.0])
    rbd = NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {
            "a": surv.Exponential.from_params([0.1]),
            "b": surv.Exponential.from_params([0.3]),
        },
    )
    ci = rbd.criticality_importance(t)
    q_a, q_b = 1 - np.exp(-0.1 * t), 1 - np.exp(-0.3 * t)
    system_ff = 1 - np.exp(-0.4 * t)
    np.testing.assert_allclose(ci["a"], q_a * (1 - q_b) / system_ff)
    np.testing.assert_allclose(ci["b"], q_b * (1 - q_a) / system_ff)


# Test fussel_vesely() w/ cut-set method: the rare-event sums


def test_fussel_vesely_incorrect_fv_type(rbd1: NonRepairableRBD):
    t = 2
    with pytest.raises(ValueError):
        rbd1.fussell_vesely(t, fv_type="a")


def test_fussel_vesely_c_rbd1(rbd1: NonRepairableRBD):
    t = 2
    fv_importance = rbd1.fussell_vesely(t, fv_type="c", method="rare_event")
    assert (
        pytest.approx(
            rbd1.reliabilities["pump1"].ff(t)
            * rbd1.reliabilities["pump2"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.reliabilities["pump1"].ff(t)
            * rbd1.reliabilities["pump2"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump2"]
    )
    assert (
        pytest.approx(rbd1.reliabilities["valve"].ff(t) / rbd1.ff(t))
        == fv_importance["valve"]
    )


def test_fussel_vesely_c_series(rbd_series: NonRepairableRBD):
    t = 2
    fv_importance = rbd_series.fussell_vesely(
        t, fv_type="c", method="rare_event"
    )
    assert (
        pytest.approx(rbd_series.reliabilities[2].ff(t) / rbd_series.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd_series.reliabilities[3].ff(t) / rbd_series.ff(t))
        == fv_importance[3]
    )
    assert (
        pytest.approx(rbd_series.reliabilities[4].ff(t) / rbd_series.ff(t))
        == fv_importance[4]
    )


def test_fussel_vesely_c_parallel(rbd_parallel: NonRepairableRBD):
    fv_importance = rbd_parallel.fussell_vesely(
        fv_type="c", method="rare_event"
    )
    # TODO: Remove need for value in FixedEventProbability
    fv_expected = (
        rbd_parallel.reliabilities[2].ff(1.0)
        * rbd_parallel.reliabilities[3].ff(1.0)
        * rbd_parallel.reliabilities[4].ff(1.0)
        / rbd_parallel.ff()
    )
    assert pytest.approx(fv_expected) == fv_importance[2]
    assert pytest.approx(fv_expected) == fv_importance[3]
    assert pytest.approx(fv_expected) == fv_importance[4]


def test_fussel_vesely_c_rbd2(rbd2: NonRepairableRBD):
    t = 2
    fv_importance = rbd2.fussell_vesely(t, fv_type="c", method="rare_event")
    assert (
        pytest.approx(rbd2.reliabilities[2].ff(t) / rbd2.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            rbd2.reliabilities[3].ff(t)
            * rbd2.reliabilities[4].ff(t)
            / rbd2.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliabilities[4].ff(t) * rbd2.reliabilities[5].ff(t)
                + rbd2.reliabilities[3].ff(t) * rbd2.reliabilities[4].ff(t)
                + rbd2.reliabilities[4].ff(t) * rbd2.reliabilities[6].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            rbd2.reliabilities[4].ff(t)
            * rbd2.reliabilities[5].ff(t)
            / rbd2.ff(t)
        )
        == fv_importance[5]
    )
    assert (
        pytest.approx(
            rbd2.reliabilities[4].ff(t)
            * rbd2.reliabilities[6].ff(t)
            / rbd2.ff(t)
        )
        == fv_importance[6]
    )
    assert (
        pytest.approx(rbd2.reliabilities[7].ff(t) / rbd2.ff(t))
        == fv_importance[7]
    )


def test_fussel_vesely_c_rbd3(rbd3: NonRepairableRBD):
    t = 2
    fv_importance = rbd3.fussell_vesely(t, fv_type="c", method="rare_event")
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[1].ff(t) * rbd3.reliabilities[3].ff(t)
                + rbd3.reliabilities[1].ff(t)
                * rbd3.reliabilities[4].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[1]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[2].ff(t) * rbd3.reliabilities[4].ff(t)
                + rbd3.reliabilities[2].ff(t)
                * rbd3.reliabilities[3].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[1].ff(t) * rbd3.reliabilities[3].ff(t)
                + rbd3.reliabilities[2].ff(t)
                * rbd3.reliabilities[3].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[2].ff(t) * rbd3.reliabilities[4].ff(t)
                + rbd3.reliabilities[1].ff(t)
                * rbd3.reliabilities[4].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[1].ff(t)
                * rbd3.reliabilities[4].ff(t)
                * rbd3.reliabilities[5].ff(t)
                + rbd3.reliabilities[2].ff(t)
                * rbd3.reliabilities[3].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[5]
    )


def test_fussel_vesely_c_repeated_component_parallel(
    rbd_repeated_component_parallel: NonRepairableRBD,
):
    rbd = rbd_repeated_component_parallel
    t = 2
    fv_importance = rbd.fussell_vesely(t, fv_type="c", method="rare_event")
    fv_expected = (
        rbd.reliabilities[2].ff(t)
        * rbd.reliabilities[3].ff(t)
        * rbd.reliabilities[4].ff(t)
        / rbd.ff(t)
    )
    assert pytest.approx(fv_expected) == fv_importance[2]
    assert pytest.approx(fv_expected) == fv_importance[3]
    assert pytest.approx(fv_expected) == fv_importance[4]


# Test fussel_vesely() w/ path-set method: the rare-event sums


def test_fussel_vesely_p_rbd1(rbd1: NonRepairableRBD):
    t = 2
    fv_importance = rbd1.fussell_vesely(t, fv_type="p", method="rare_event")
    assert (
        pytest.approx(
            rbd1.reliabilities["pump1"].ff(t)
            * rbd1.reliabilities["valve"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump1"]
    )
    assert (
        pytest.approx(
            rbd1.reliabilities["pump2"].ff(t)
            * rbd1.reliabilities["valve"].ff(t)
            / rbd1.ff(t)
        )
        == fv_importance["pump2"]
    )
    assert (
        pytest.approx(
            (
                rbd1.reliabilities["pump1"].ff(t)
                * rbd1.reliabilities["valve"].ff(t)
                + rbd1.reliabilities["pump2"].ff(t)
                * rbd1.reliabilities["valve"].ff(t)
            )
            / rbd1.ff(t)
        )
        == fv_importance["valve"]
    )


def test_fussel_vesely_p_series(rbd_series: NonRepairableRBD):
    t = 2
    fv_importance = rbd_series.fussell_vesely(
        t, fv_type="p", method="rare_event"
    )
    expected_fv_importance = (
        rbd_series.reliabilities[2].ff(t)
        * rbd_series.reliabilities[3].ff(t)
        * rbd_series.reliabilities[4].ff(t)
        / rbd_series.ff(t)
    )
    assert pytest.approx(expected_fv_importance) == fv_importance[2]
    assert pytest.approx(expected_fv_importance) == fv_importance[3]
    assert pytest.approx(expected_fv_importance) == fv_importance[4]


def test_fussel_vesely_p_parallel(rbd_parallel: NonRepairableRBD):
    t = 2
    fv_importance = rbd_parallel.fussell_vesely(
        t, fv_type="p", method="rare_event"
    )
    assert (
        pytest.approx(rbd_parallel.reliabilities[2].ff(t) / rbd_parallel.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd_parallel.reliabilities[3].ff(t) / rbd_parallel.ff(t))
        == fv_importance[3]
    )
    assert (
        pytest.approx(rbd_parallel.reliabilities[4].ff(t) / rbd_parallel.ff(t))
        == fv_importance[4]
    )


def test_fussel_vesely_p_rbd2(rbd2: NonRepairableRBD):
    t = 2
    fv_importance = rbd2.fussell_vesely(t, fv_type="p", method="rare_event")
    assert (
        pytest.approx(
            (
                (
                    rbd2.reliabilities[2].ff(t)
                    * rbd2.reliabilities[3].ff(t)
                    * rbd2.reliabilities[5].ff(t)
                    * rbd2.reliabilities[6].ff(t)
                    * rbd2.reliabilities[7].ff(t)
                )
                + (
                    rbd2.reliabilities[2].ff(t)
                    * rbd2.reliabilities[4].ff(t)
                    * rbd2.reliabilities[7].ff(t)
                )
            )
            / rbd2.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliabilities[2].ff(t)
                * rbd2.reliabilities[3].ff(t)
                * rbd2.reliabilities[5].ff(t)
                * rbd2.reliabilities[6].ff(t)
                * rbd2.reliabilities[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliabilities[2].ff(t)
                * rbd2.reliabilities[4].ff(t)
                * rbd2.reliabilities[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliabilities[2].ff(t)
                * rbd2.reliabilities[3].ff(t)
                * rbd2.reliabilities[5].ff(t)
                * rbd2.reliabilities[6].ff(t)
                * rbd2.reliabilities[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[5]
    )
    assert (
        pytest.approx(
            (
                rbd2.reliabilities[2].ff(t)
                * rbd2.reliabilities[3].ff(t)
                * rbd2.reliabilities[5].ff(t)
                * rbd2.reliabilities[6].ff(t)
                * rbd2.reliabilities[7].ff(t)
            )
            / rbd2.ff(t)
        )
        == fv_importance[6]
    )
    assert (
        pytest.approx(
            (
                (
                    rbd2.reliabilities[2].ff(t)
                    * rbd2.reliabilities[3].ff(t)
                    * rbd2.reliabilities[5].ff(t)
                    * rbd2.reliabilities[6].ff(t)
                    * rbd2.reliabilities[7].ff(t)
                )
                + (
                    rbd2.reliabilities[2].ff(t)
                    * rbd2.reliabilities[4].ff(t)
                    * rbd2.reliabilities[7].ff(t)
                )
            )
            / rbd2.ff(t)
        )
    ) == fv_importance[7]


def test_fussel_vesely_p_rbd3(rbd3: NonRepairableRBD):
    t = 2
    fv_importance = rbd3.fussell_vesely(t, fv_type="p", method="rare_event")
    assert (
        pytest.approx(
            (
                (rbd3.reliabilities[1].ff(t) * rbd3.reliabilities[2].ff(t))
                + (
                    rbd3.reliabilities[1].ff(t)
                    * rbd3.reliabilities[5].ff(t)
                    * rbd3.reliabilities[4].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[1]
    )
    assert (
        pytest.approx(
            (
                (rbd3.reliabilities[1].ff(t) * rbd3.reliabilities[2].ff(t))
                + (
                    rbd3.reliabilities[3].ff(t)
                    * rbd3.reliabilities[5].ff(t)
                    * rbd3.reliabilities[2].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[2]
    )
    assert (
        pytest.approx(
            (
                (rbd3.reliabilities[3].ff(t) * rbd3.reliabilities[4].ff(t))
                + (
                    rbd3.reliabilities[3].ff(t)
                    * rbd3.reliabilities[5].ff(t)
                    * rbd3.reliabilities[2].ff(t)
                )
            )
            / rbd3.ff(t)
        )
        == fv_importance[3]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[3].ff(t) * rbd3.reliabilities[4].ff(t)
                + rbd3.reliabilities[1].ff(t)
                * rbd3.reliabilities[4].ff(t)
                * rbd3.reliabilities[5].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[4]
    )
    assert (
        pytest.approx(
            (
                rbd3.reliabilities[1].ff(t)
                * rbd3.reliabilities[5].ff(t)
                * rbd3.reliabilities[4].ff(t)
                + rbd3.reliabilities[3].ff(t)
                * rbd3.reliabilities[5].ff(t)
                * rbd3.reliabilities[2].ff(t)
            )
            / rbd3.ff(t)
        )
        == fv_importance[5]
    )


def test_fussel_vesely_p_repeated_component_parallel(
    rbd_repeated_component_parallel: NonRepairableRBD,
):
    rbd = rbd_repeated_component_parallel
    t = 2
    fv_importance = rbd.fussell_vesely(t, fv_type="p", method="rare_event")
    assert (
        pytest.approx(rbd.reliabilities[2].ff(t) / rbd.ff(t))
        == fv_importance[2]
    )
    assert (
        pytest.approx(rbd.reliabilities[3].ff(t) / rbd.ff(t))
        == fv_importance[3]
    )
    assert (
        pytest.approx(rbd.reliabilities[4].ff(t) / rbd.ff(t))
        == fv_importance[4]
    )
