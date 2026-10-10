"""A repairable diagram's importance measures over time (#191): at times
from new or from the components' states (``x``, ``state``), and over a
window (``window``).

At a time each measure is that of the nodes' point availabilities then:
checked against the system held with each node working and failed
(``point_availability``), against an enumeration of the nodes' states,
and at late times against the long-run measures. Over a window a ratio
measure is the ratio of the system's means over it (``mission_
availability`` held), including windows past the time the curves repeat.
With limited repair crews the chain is checked against its own matrix
exponential, and with common-cause groups the late values against the
long run."""

import itertools
import math

import numpy as np
import pytest
import scipy.linalg
import surpyval as surv

from repyability import BetaFactor, CCFGroup, NodeState, RepairableRBD
from repyability.rbd import _ccf_groups, _crews, _curves
from repyability.rbd.rbd import RBD

E, W = surv.Exponential.from_params, surv.Weibull.from_params
MEASURES = (
    "birnbaum_importance",
    "improvement_potential",
    "risk_achievement_worth",
    "risk_reduction_worth",
    "criticality_importance",
    "fussell_vesely",
)


def unit(life=None, repair=None, **more):
    return {
        "reliability": E([0.01]) if life is None else life,
        "repairability": E([0.1]) if repair is None else repair,
        **more,
    }


def bridge(**more):
    """A bridge of five units, two of them wearing out."""
    edges = [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("a", "d")]
    edges += [("c", "d"), ("b", "e"), ("c", "e"), ("d", "t"), ("e", "t")]
    return RepairableRBD(
        edges,
        {
            "a": unit(),
            "b": unit(W([200.0, 2.0]), E([0.2])),
            "c": unit(E([0.02]), E([0.5])),
            "d": unit(W([400.0, 1.5]), E([0.25])),
            "e": unit(E([0.005]), E([0.05])),
        },
        **more,
    )


def pair_then_c(**more):
    """a and b in parallel, then c."""
    return RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {
            "a": unit(),
            "b": unit(),
            "c": unit(E([0.002]), E([0.5])),
        },
        **more,
    )


def held(rbd, measure, times, node, **given):
    """The measure at ``times`` from the system held with ``node`` working
    and failed (``point_availability``)."""
    base = 1.0 - rbd.point_availability(times, **given)
    up = 1.0 - rbd.point_availability(times, working_nodes=[node], **given)
    down = 1.0 - rbd.point_availability(times, broken_nodes=[node], **given)
    return {
        "birnbaum_importance": down - up,
        "improvement_potential": base - up,
        "risk_achievement_worth": down / base,
        "risk_reduction_worth": base / up,
    }[measure]


@pytest.mark.parametrize("measure", MEASURES[:4])
def test_at_times_it_is_the_system_held(measure):
    rbd = bridge()
    times = np.array([0.5, 7.0, 60.0, 900.0])
    values = getattr(rbd, measure)(x=times)
    for node in rbd.nodes:
        # The system held subtracts availabilities near 1, to about 1e-16:
        # a ratio of unavailabilities of 1e-7 to about 1e-9.
        np.testing.assert_allclose(
            values[node],
            held(rbd, measure, times, node),
            rtol=1e-7,
            atol=1e-14,
        )
    # A single time gives floats.
    single = getattr(rbd, measure)(x=60.0)
    assert all(isinstance(v, float) for v in single.values())
    assert single["c"] == pytest.approx(float(values["c"][2]))


def enumerated(rbd, availability: dict):
    """Each node's criticality (both kinds) and Fussell-Vesely measure
    (both set types) from every state of the nodes, up with
    ``availability``."""
    nodes = list(availability)
    cuts = rbd.get_min_cut_sets()
    paths = [set(p) for p in rbd.get_min_path_sets(False)]
    totals = dict.fromkeys(["up", "down"], 0.0)
    shares: dict = {}
    for state in itertools.product([False, True], repeat=len(nodes)):
        up = {n for n, s in zip(nodes, state) if s}
        chance = math.prod(
            availability[n] if s else 1.0 - availability[n]
            for n, s in zip(nodes, state)
        )
        works = any(p <= up for p in paths)
        totals["up" if works else "down"] += chance
        for n in nodes:
            # Critical: the system works with n up and not with it down.
            with_n = any(p <= up | {n} for p in paths)
            without = any(p <= up - {n} for p in paths)
            critical = with_n and not without
            key = (n, "failure") if n not in up else (n, "success")
            if critical:
                shares[key] = shares.get(key, 0.0) + chance
            down = set(nodes) - up
            for kind, sets in (("c", cuts), ("p", paths)):
                if any(n in s and s <= down for s in sets):
                    shares[(n, kind)] = shares.get((n, kind), 0.0) + chance
    return totals, shares


def test_criticality_and_fussell_vesely_by_enumeration():
    rbd = bridge()
    t = 40.0
    curves = _curves._availability_curves(rbd, t, set())
    availability = {
        n: float(c.at(np.array([t]))[0]) for n, c in curves.items()
    }
    totals, shares = enumerated(rbd, availability)
    failure = rbd.criticality_importance(x=t)
    success = rbd.criticality_importance(x=t, kind="success")
    cut = rbd.fussell_vesely(x=t)
    path = rbd.fussell_vesely(x=t, fv_type="p")
    for n in rbd.nodes:
        assert failure[n] == pytest.approx(
            shares.get((n, "failure"), 0.0) / totals["down"], rel=1e-9
        )
        assert success[n] == pytest.approx(
            shares.get((n, "success"), 0.0) / totals["up"], rel=1e-9
        )
        assert cut[n] == pytest.approx(
            shares.get((n, "c"), 0.0) / totals["down"], rel=1e-9
        )
        assert path[n] == pytest.approx(
            shares.get((n, "p"), 0.0) / totals["down"], rel=1e-9
        )


@pytest.mark.parametrize(
    "build",
    [
        bridge,
        lambda: pair_then_c(
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))]
        ),
        lambda: pair_then_c(repair_crews=1),
    ],
    ids=["independent", "common cause", "crews"],
)
@pytest.mark.parametrize("measure", MEASURES)
def test_late_times_and_long_windows_are_the_long_run(build, measure):
    rbd = build()
    long_run = getattr(rbd, measure)()
    late = getattr(rbd, measure)(x=1e6)
    window = getattr(rbd, measure)(window=1e8)
    for node, value in long_run.items():
        assert late[node] == pytest.approx(value, rel=1e-6)
        assert window[node] == pytest.approx(value, rel=1e-4)


@pytest.mark.parametrize("measure", MEASURES[:4])
@pytest.mark.parametrize("end", [30.0, 2_000.0, 12_345.0])
def test_over_a_window_it_is_the_ratio_of_means(measure, end):
    # c is block replaced, so its curve repeats: the longest windows are
    # extended over the repeats rather than followed.
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {
            "a": unit(W([150.0, 2.5]), E([0.2])),
            "b": unit(),
            "c": unit(
                W([300.0, 3.0]),
                E([0.5]),
                preventive={"interval": 100.0, "policy": "block"},
            ),
        },
    )
    values = getattr(rbd, measure)(window=end)
    for node in rbd.nodes:
        base = 1.0 - rbd.mission_availability(end)
        up = 1.0 - rbd.mission_availability(end, working_nodes=[node])
        down = 1.0 - rbd.mission_availability(end, broken_nodes=[node])
        expected = {
            "birnbaum_importance": down - up,
            "improvement_potential": base - up,
            "risk_achievement_worth": down / base,
            "risk_reduction_worth": base / up,
        }[measure]
        assert values[node] == pytest.approx(expected, rel=1e-6, abs=1e-12)


def test_several_windows_and_times_keep_their_shape():
    rbd = bridge()
    windows = np.array([[10.0, 100.0], [1_000.0, 5_000.0]])
    values = rbd.risk_achievement_worth(window=windows)
    assert values["a"].shape == (2, 2)
    for i, end in np.ndenumerate(windows):
        alone = rbd.risk_achievement_worth(window=float(end))["a"]
        assert values["a"][i] == pytest.approx(alone)
    times = rbd.birnbaum_importance(x=[[1.0], [2.0]])
    assert times["b"].shape == (2, 1)


def test_from_a_state():
    rbd = pair_then_c()
    # Started in its long-run state, a system of exponential units is in
    # it at every time.
    for measure in MEASURES:
        long_run = getattr(rbd, measure)()
        steady = getattr(rbd, measure)(x=[0.0, 30.0], state="stationary")
        for node, value in long_run.items():
            np.testing.assert_allclose(steady[node], value, rtol=1e-9)
    # With a down at 0, b is critical then whenever c is up (it is).
    down = {"a": NodeState(alive=False)}
    at_start = rbd.birnbaum_importance(x=0.0, state=down)
    assert at_start["b"] == pytest.approx(1.0)
    later = rbd.birnbaum_importance(x=5.0, state=down)
    expected = held(
        rbd, "birnbaum_importance", np.array([5.0]), "b", state=down
    )
    assert later["b"] == pytest.approx(float(expected[0]))


def test_held_nodes_over_time():
    rbd = bridge()
    values = rbd.birnbaum_importance(x=[3.0, 30.0], working_nodes=["c"])
    times = np.array([3.0, 30.0])
    up = rbd.point_availability(times, working_nodes=["c", "a"])
    down = rbd.point_availability(
        times, working_nodes=["c"], broken_nodes=["a"]
    )
    np.testing.assert_allclose(values["a"], up - down, rtol=1e-9)


def test_with_crews_the_chain_against_its_matrix_exponential():
    rbd = pair_then_c(repair_crews=1)
    times = np.array([0.5, 4.0, 40.0])
    chain = _crews._crew_chain(rbd, frozenset())
    start = _curves._crew_start(rbd, chain, {})
    p, _ = _crews._chain_probabilities(rbd, set(), set())
    q = {n: 1.0 - v for n, v in p.items()}
    generator = np.asarray(
        chain.generator.toarray()
        if hasattr(chain.generator, "toarray")
        else chain.generator
    )
    for measure in ("criticality_importance", "fussell_vesely"):
        values = getattr(rbd, measure)(x=times)
        for k, t in enumerate(times):
            weights = start @ scipy.linalg.expm(generator * t)
            if measure == "criticality_importance":
                expected = RBD._criticality_importance(
                    rbd, p, "failure", weights, q
                )
            else:
                expected = RBD._fussell_vesely(
                    rbd, p, "c", "exact", weights, q
                )
            for node in rbd.nodes:
                assert values[node][k] == pytest.approx(
                    float(np.ravel(expected[node])[0]), rel=1e-9
                )
    # The held measures are the system held in the chain over time.
    raw = rbd.risk_achievement_worth(x=times)
    for node in rbd.nodes:
        np.testing.assert_allclose(
            raw[node],
            held(rbd, "risk_achievement_worth", times, node),
            rtol=1e-9,
        )


def test_with_common_causes_a_member_is_conditioned_on_its_state():
    rbd = pair_then_c(
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.2, basis="rate"))]
    )
    times = np.array([2.0, 20.0, 200.0])
    values = rbd.birnbaum_importance(x=times)
    # c, outside the group, is held as before.
    np.testing.assert_allclose(
        values["c"],
        held(rbd, "birnbaum_importance", times, "c"),
        rtol=1e-9,
    )
    # A member's: the system given it up less given it down, over the
    # group's joint states at each time (a with b, through the cause).
    groups = _ccf_groups._groups_over_time(rbd)
    (group,) = groups
    chances = group.probabilities(times)
    c_up = _ccf_groups._groups_curve(rbd, 200.0).curves["c"].at(times)
    position = list(group.members).index("a")
    a_up = ~group.down[:, position]
    b_up = ~group.down[:, 1 - position]
    # P(system up | a up) - P(system up | a down): with a up the system
    # needs c alone; with a down, b and c.
    p_a_up = chances[:, a_up].sum(axis=1)
    p_b_up_given_a_down = chances[:, ~a_up & b_up].sum(axis=1) / chances[
        :, ~a_up
    ].sum(axis=1)
    expected = c_up * 1.0 - c_up * p_b_up_given_a_down
    np.testing.assert_allclose(values["a"], expected, rtol=1e-9)
    assert np.all(p_a_up < 1.0)


def test_what_it_refuses():
    rbd = bridge()
    with pytest.raises(ValueError, match="not both"):
        rbd.birnbaum_importance(x=1.0, window=10.0)
    with pytest.raises(ValueError, match="state is where"):
        rbd.birnbaum_importance(state="stationary")
    with pytest.raises(ValueError, match="positive length"):
        rbd.fussell_vesely(window=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        rbd.fussell_vesely(x=-1.0)
    with pytest.raises(ValueError, match="kind"):
        rbd.criticality_importance(x=1.0, kind="both")
    with pytest.raises(ValueError, match="fv_type"):
        rbd.fussell_vesely(x=1.0, fv_type="x")
    # A unit repaired imperfectly has no curve over time.
    imperfect = {**unit(W([100.0, 2.0])), "repair": {"model": "kijima1"}}
    imperfect["repair"]["q"] = 0.5
    kijima = RepairableRBD([("s", "a"), ("a", "t")], {"a": imperfect})
    with pytest.raises(NotImplementedError):
        kijima.birnbaum_importance(x=1.0)
    # Crews around a nested RBD: the chain's states take no nested curve.
    inner = RepairableRBD([("s", "u"), ("u", "t")], {"u": unit()})
    crewed = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "n"), ("b", "n"), ("n", "t")],
        {"a": unit(), "b": unit(), "n": inner},
        repair_crews=1,
    )
    with pytest.raises(NotImplementedError, match="nested RBD"):
        crewed.criticality_importance(x=1.0)
    assert set(crewed.birnbaum_importance(x=1.0)) == {"a", "b", "n"}
