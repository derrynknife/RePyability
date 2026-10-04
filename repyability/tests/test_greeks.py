"""The sensitivity measures as one family, the Greeks (#197): they take their
arguments alike, and those that are shares add up to the whole, so that
they can be shown as a breakdown. Run on the station of the guide's page
(``docs/guide/greeks.md``)."""

import inspect

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairableRBD, RepairableRBD

E, W = surv.Exponential.from_params, surv.Weibull.from_params

EDGES = [
    ("s", "pump1"),
    ("s", "pump2"),
    ("pump1", "valve"),
    ("pump2", "valve"),
    ("valve", "t"),
]


def unit(life, repair, **more):
    return {"reliability": life, "repairability": repair, **more}


def station():
    return RepairableRBD(
        EDGES,
        {
            "pump1": unit(W([40, 2.0]), E([1.0])),
            "pump2": unit(W([40, 2.0]), E([1.0])),
            "valve": unit(
                W([160, 1.5]),
                E([2.0]),
                preventive={
                    "policy": "age",
                    "interval": 80.0,
                    "duration": E([4.0]),
                },
            ),
        },
    )


def parameters(method):
    """The method's parameters but ``self``, by name, in order."""
    return list(inspect.signature(method).parameters.values())[1:]


@pytest.mark.parametrize(
    "name",
    [
        "birnbaum_importance",
        "parameter_sensitivity",
        "differential_importance",
        "joint_importance",
        "barlow_proschan_importance",
    ],
)
def test_a_repairable_diagram_s_greeks_take_their_arguments_alike(name):
    # The nodes held come first; the times, the window and the state are
    # keyword-only, a long-run value being the default.
    given = parameters(getattr(RepairableRBD, name))
    assert [p.name for p in given[:2]] == ["working_nodes", "broken_nodes"]
    assert all(p.kind is p.POSITIONAL_OR_KEYWORD for p in given[:2])
    timing = {p.name: p for p in given if p.name in ("x", "window", "state")}
    assert set(timing) >= {"window", "state"}
    assert all(p.kind is p.KEYWORD_ONLY for p in timing.values())


def test_theta_takes_its_times_first():
    # A rate needs its times: there is no long-run one to default to.
    given = parameters(RepairableRBD.availability_rate)
    assert [p.name for p in given[:3]] == [
        "x",
        "working_nodes",
        "broken_nodes",
    ]
    assert given[0].default is inspect.Parameter.empty
    state = next(p for p in given if p.name == "state")
    assert state.kind is state.KEYWORD_ONLY


@pytest.mark.parametrize(
    "name",
    [
        "birnbaum_importance",
        "parameter_sensitivity",
        "differential_importance",
        "joint_importance",
        "reliability_rate",
        "barlow_proschan_importance",
    ],
)
def test_a_non_repairable_diagram_s_greeks_take_their_arguments_alike(name):
    given = parameters(getattr(NonRepairableRBD, name))
    assert [p.name for p in given[:3]] == [
        "x",
        "working_nodes",
        "broken_nodes",
    ]


def test_the_shares_add_up():
    rbd = station()
    assert sum(rbd.differential_importance().values()) == pytest.approx(1.0)
    kinds = {
        "lives": [
            (n, f"reliability.{p}")
            for n in ("pump1", "pump2", "valve")
            for p in ("alpha", "beta")
        ],
        "repairs": [
            (n, "repairability.failure_rate")
            for n in ("pump1", "pump2", "valve")
        ],
        "maintenance": [
            ("valve", "preventive.interval"),
            ("valve", "preventive.duration.failure_rate"),
        ],
    }
    gain = rbd.differential_importance(
        over="parameters", change="proportional", improving=True, groups=kinds
    )
    assert sum(gain.values()) == pytest.approx(1.0)
    assert all(share > 0 for share in gain.values())
    for shares in (
        rbd.barlow_proschan_importance(),
        rbd.barlow_proschan_importance(window=100.0),
    ):
        assert sum(shares.values()) == pytest.approx(1.0)


def test_theta_s_parts_add_up_to_the_system_s_rate_and_jumps():
    rbd = station()
    rate = rbd.availability_rate([5.0, 40.0, 81.0])
    np.testing.assert_allclose(
        sum(rate.node_rate.values()), rate.rate, rtol=1e-12, atol=1e-15
    )
    # The valve's age replacement, due at 80 for every valve in its first
    # life, is the one jump by 81: the chance that the valve lasted to 80
    # and that a pump works then.
    assert rate.jump_times.tolist() == [80.0]
    np.testing.assert_allclose(
        sum(rate.node_jumps.values()), rate.jumps, rtol=1e-12
    )
    assert rate.node_jumps["pump1"][0] == 0.0
    # The system's own jump, worked out apart: the point availability
    # either side of 80.
    either_side = rbd.point_availability(
        np.array([np.nextafter(80.0, 0), 80.0])
    )
    assert rate.jumps[0] == pytest.approx(np.diff(either_side)[0], rel=1e-9)


def test_vega_s_shares_add_up():
    def fitted(scale, shape, n):
        model = W([scale, shape])
        return surv.Weibull.fit(model.qf(np.linspace(0.04, 0.96, n)))

    pump = fitted(40, 2.0, 12)
    rbd = NonRepairableRBD(
        EDGES, {"pump1": pump, "pump2": pump, "valve": fitted(160, 1.5, 30)}
    )
    for parts in (
        rbd.uncertainty_importance(20.0),
        rbd.uncertainty_importance(of="mean"),
    ):
        assert set(parts.first_order) == {("pump1", "pump2"), "valve"}
        assert sum(parts.first_order.values()) == pytest.approx(1.0)
