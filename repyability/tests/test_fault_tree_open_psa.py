"""Reading fault trees from Open-PSA (MEF) documents.

The writer's output reads back as the tree it was written from, on every
tree of the catalogue it exports; documents written by hand check what
the writer does not produce (XOR, NAND and NOR gates, nested formulas,
parameters, house events, expressions of the mission time), and each
element the reader does not take is refused by name.
"""

import io

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    MGL,
    AlphaFactor,
    BetaFactor,
    CCFGroup,
    FaultTree,
)
from repyability.tests.catalogue import fault_tree_kinds

E, W = surv.Exponential.from_params, surv.Weibull.from_params
T = np.array([5.0, 40.0, 120.0, 400.0])


def exported():
    """The catalogue's trees of probabilities and lifetimes the writer
    exports, by kind (the repaired ones are checked below)."""
    out = {}
    for kind, tree in fault_tree_kinds().items():
        if tree.is_repairable:
            continue
        try:
            tree.to_open_psa()
        except NotImplementedError:
            try:
                tree.to_open_psa(t=40.0)
            except NotImplementedError:
                continue
        out[kind] = tree
    return out


KINDS = exported()


def test_the_round_trips_cover_the_static_trees():
    assert set(KINDS) == {
        "fixed",
        "lifetimes",
        "mixed",
        "fitted",
        "repeated",
        "common cause",
        "common cause, fixed",
        "not gate",
        "functional dependency",
    }


def same_values(again, tree, t):
    """The top event's probability, and each event's, are the same."""
    p = again.top_event_probability(None if again.is_fixed else t)
    q = tree.top_event_probability(None if tree.is_fixed else t)
    np.testing.assert_allclose(p, q, rtol=1e-15, atol=0)
    for name, model in again.events.items():
        original = tree.events[name]
        if isinstance(model, float):
            expected = (
                original
                if isinstance(original, float)
                else np.ravel(original.ff(40.0))[0]
            )
            assert model == expected
        else:
            np.testing.assert_allclose(
                model.ff(t), original.ff(t), rtol=1e-15, atol=0
            )


# The catalogue's beta factor splits a probability past the 0.1 it is
# meant for at the later times, and says so: the round trip is the same.
@pytest.mark.filterwarnings("ignore:Common-cause group")
@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("at", [None, 40.0])
def test_a_tree_reads_back_as_it_was_written(kind, at):
    tree = KINDS[kind]
    try:
        document = tree.to_open_psa(t=at)
    except NotImplementedError:
        pytest.skip(f"{kind} is exported at a time only")
    again = FaultTree.from_open_psa(document)
    assert again.top == tree.top
    assert set(again.events) == set(tree.events)
    if kind != "functional dependency":
        assert again.gates == tree.gates
    assert [
        (g.members, g.model.beta, g.model.basis) for g in again.ccf_groups
    ] == [(g.members, g.model.beta, g.model.basis) for g in tree.ccf_groups]
    if at is None:
        # Each event as its model: lifetimes stay lifetimes.
        assert [isinstance(m, float) for m in again.events.values()] == [
            isinstance(tree.events[e], float) for e in again.events
        ]
        same_values(again, tree, T)
    else:
        assert again.is_fixed
        same_values(again, tree, np.array([at]))


def test_an_fdep_reads_back_as_its_static_equivalent():
    tree = KINDS["functional dependency"]
    again = FaultTree.from_open_psa(tree.to_open_psa())
    assert again.gates["pair"] == ("and", ["Forced_event_a", "Forced_event_b"])
    assert again.gates["Forced_event_a"] == ("or", ["a", "p"])
    assert not any(g[0] == "fdep" for g in again.gates.values())
    for t in T:
        assert again.top_event_probability(t) == pytest.approx(
            tree.top_event_probability(t), rel=1e-15, abs=0
        )


def test_names_that_are_not_identifiers_come_back_through_their_labels():
    tree = FaultTree(
        {
            "no cooling": ("or", ["no flow", "A", "a", "1st valve"]),
            "no flow": ("and", ["pump 1", "pump 2"]),
        },
        {
            "pump 1": 0.1,
            "pump 2": 0.1,
            "A": 0.01,
            "a": 0.02,
            "1st valve": 0.05,
        },
        ccf_groups=[CCFGroup(["pump 1", "pump 2"], BetaFactor(0.1))],
    )
    document = tree.to_open_psa()
    assert 'name="A"' in document and 'name="a-2"' in document
    again = FaultTree.from_open_psa(document)
    assert again.gates == tree.gates
    assert again.events == tree.events
    assert list(again.ccf_groups[0].members) == ["pump 1", "pump 2"]
    assert again.top_event_probability() == tree.top_event_probability()


def test_a_label_that_describes_keeps_the_identifier():
    document = """<?xml version="1.0"?>
    <opsa-mef><define-fault-tree name="FT">
      <define-gate name="TOP"><label>Loss of cooling</label>
        <or><basic-event name="P1"/><basic-event name="V"/></or>
      </define-gate>
      <define-basic-event name="P1"><label>Pump 1 fails</label>
        <float value="0.1"/></define-basic-event>
      <define-basic-event name="V"><float value="0.2"/></define-basic-event>
    </define-fault-tree></opsa-mef>"""
    again = FaultTree.from_open_psa(document)
    assert again.gates == {"TOP": ("or", ["P1", "V"])}
    assert again.top_event_probability() == pytest.approx(1 - 0.9 * 0.8)


HAND_WRITTEN = """<?xml version="1.0" encoding="UTF-8"?>
<opsa-mef>
  <define-fault-tree name="cooling">
    <define-gate name="TOP">
      <or>
        <gate name="G1"/>
        <event name="G2"/>
        <and><basic-event name="E"/><house-event name="H"/></and>
      </or>
    </define-gate>
    <define-gate name="G1"><xor><basic-event name="A"/><event name="B"/></xor>
    </define-gate>
    <define-gate name="G2">
      <nor><basic-event name="C"/><not><basic-event name="D"/></not></nor>
    </define-gate>
    <define-house-event name="H"><constant value="true"/></define-house-event>
  </define-fault-tree>
  <model-data>
    <define-parameter name="lambda"><float value="0.001"/></define-parameter>
    <define-parameter name="half">
      <mul><parameter name="lambda"/><float value="0.5"/></mul>
    </define-parameter>
    <define-parameter name="pB"><float value="0.2"/></define-parameter>
    <define-basic-event name="A"><float value="0.1"/></define-basic-event>
    <define-basic-event name="B"><parameter name="pB"/></define-basic-event>
    <define-basic-event name="C">
      <exponential>
        <parameter name="lambda"/><system-mission-time/>
      </exponential>
    </define-basic-event>
    <define-basic-event name="D">
      <sub><float value="1"/>
        <exp><neg><mul><parameter name="half"/><system-mission-time/></mul>
        </neg></exp>
      </sub>
    </define-basic-event>
    <define-basic-event name="E"><float value="0.3"/></define-basic-event>
  </model-data>
</opsa-mef>
"""


def test_a_hand_written_document_loads():
    tree = FaultTree.from_open_psa(HAND_WRITTEN, mission_time=100.0)
    assert tree.top == "TOP"
    # The nested AND, XOR's parts and NOR's OR are gates of their own.
    assert tree.gates["TOP"] == ("or", ["G1", "G2", "TOP/1"])
    assert tree.gates["TOP/1"] == ("and", ["E", "H"])
    assert tree.gates["G1"] == ("or", ["G1/1", "G1/2"])
    assert tree.gates["G2"] == ("not", ["G2/2"])
    assert tree.gates["G2/2"] == ("or", ["C", "G2/1"])
    assert tree.gates["G2/1"] == ("not", ["D"])
    assert tree.events["H"] == 1.0
    assert tree.events["B"] == 0.2
    # An exponential life stays a model; D is read at the mission time.
    assert isinstance(tree.events["C"], type(E([0.001])))
    # As the document writes it: 1 - exp(-(half * t)), half = lambda * 0.5.
    d = 1 - np.exp(-((0.001 * 0.5) * 100.0))
    assert tree.events["D"] == d
    a, b, e = 0.1, 0.2, 0.3
    c = -np.expm1(-0.001 * 100.0)
    xor, nor = a * (1 - b) + (1 - a) * b, (1 - c) * d
    expected = 1 - (1 - xor) * (1 - nor) * (1 - e)
    assert tree.top_event_probability(100.0) == pytest.approx(
        expected, rel=1e-14
    )


def test_an_expression_of_the_mission_time_needs_one():
    with pytest.raises(ValueError, match="Event 'D'.*pass mission_time"):
        FaultTree.from_open_psa(HAND_WRITTEN)


def test_a_nand_gate_and_a_weibull_with_an_offset():
    document = """<opsa-mef><define-fault-tree name="FT">
      <define-gate name="top"><nand><basic-event name="a"/>
        <basic-event name="b"/></nand></define-gate>
      <define-basic-event name="a"><Weibull><float value="100"/>
        <float value="2"/><float value="10"/><system-mission-time/>
        </Weibull></define-basic-event>
      <define-basic-event name="b"><float value="0.5"/></define-basic-event>
    </define-fault-tree></opsa-mef>"""
    tree = FaultTree.from_open_psa(document)
    assert tree.gates == {
        "top": ("not", ["top/1"]),
        "top/1": ("and", ["a", "b"]),
    }
    a = W([100.0, 2.0], gamma=10.0)
    assert tree.top_event_probability(60.0) == pytest.approx(
        1 - float(a.ff(60.0)) * 0.5, rel=1e-14
    )


def test_a_document_is_read_from_a_path_or_a_file(tmp_path):
    tree = KINDS["common cause"]
    path = tmp_path / "tree.xml"
    path.write_text(tree.to_open_psa(), encoding="utf-8")
    for source in (path, str(path), io.StringIO(tree.to_open_psa())):
        again = FaultTree.from_open_psa(source)
        assert again.gates == tree.gates
    with pytest.raises(FileNotFoundError, match="neither"):
        FaultTree.from_open_psa(str(tmp_path / "missing.xml"))


def wrapped(expression: str = '<float value="0.1"/>', gate: str = "") -> str:
    gate = gate or '<or><basic-event name="a"/><basic-event name="b"/></or>'
    return (
        '<opsa-mef><define-fault-tree name="FT">'
        f'<define-gate name="top">{gate}</define-gate>'
        f'<define-basic-event name="a">{expression}</define-basic-event>'
        '<define-basic-event name="b"><float value="0.2"/>'
        "</define-basic-event></define-fault-tree></opsa-mef>"
    )


@pytest.mark.parametrize(
    "element, document",
    [
        (
            "GLM",
            wrapped(
                '<mul><float value="2"/><GLM><float value="0"/>'
                '<float value="0.01"/><float value="0.1"/>'
                "<system-mission-time/></GLM></mul>"
            ),
        ),
        (
            "periodic-test",
            wrapped(
                '<periodic-test><float value="0.01"/><float value="100"/>'
                '<float value="0"/><float value="10"/></periodic-test>'
            ),
        ),
        (
            "normal-deviate",
            wrapped(
                '<normal-deviate><float value="0.1"/><float value="0.01"/>'
                "</normal-deviate>"
            ),
        ),
        (
            "lognormal-deviate",
            wrapped(
                '<lognormal-deviate><float value="0.1"/>'
                '<float value="3"/><float value="0.95"/></lognormal-deviate>'
            ),
        ),
        (
            "iff",
            wrapped(
                gate='<iff><basic-event name="a"/><basic-event name="b"/>'
                "</iff>"
            ),
        ),
        (
            "imply",
            wrapped(
                gate='<imply><basic-event name="a"/><basic-event name="b"/>'
                "</imply>"
            ),
        ),
        (
            "cardinality",
            wrapped(
                gate='<cardinality min="1" max="1"><basic-event name="a"/>'
                '<basic-event name="b"/></cardinality>'
            ),
        ),
        (
            "define-event-tree",
            '<opsa-mef><define-event-tree name="ET"/></opsa-mef>',
        ),
    ],
)
def test_what_is_not_read_is_refused_by_name(element, document):
    with pytest.raises(NotImplementedError, match=f"<{element}> is not read"):
        FaultTree.from_open_psa(document)


@pytest.mark.parametrize(
    "document, match",
    [
        (
            '<!DOCTYPE opsa-mef [<!ENTITY x "0.1">]>' + wrapped(),
            "document type.*remove the declaration",
        ),
        (
            '<!DOCTYPE opsa-mef SYSTEM "mef.dtd">' + wrapped(),
            "document type",
        ),
        (
            wrapped(
                gate='<foo><basic-event name="a"/>'
                '<basic-event name="b"/></foo>'
            ),
            "<foo>, which is not a gate the reader takes",
        ),
        (
            wrapped(
                gate='<or><basic-event name="a"/><basic-event name="z"/></or>'
            ),
            "'z'.*does not define.*define it",
        ),
        (
            '<opsa-mef><define-fault-tree name="FT">'
            '<define-gate name="top"><or><gate name="g"/>'
            '<basic-event name="a"/></or></define-gate>'
            '<define-gate name="g"><and><gate name="h"/>'
            '<basic-event name="a"/></and></define-gate>'
            '<define-gate name="h"><or><gate name="g"/>'
            '<basic-event name="a"/></or></define-gate>'
            '<define-basic-event name="a"><float value="0.1"/>'
            "</define-basic-event></define-fault-tree></opsa-mef>",
            "loop, g -> h -> g",
        ),
        (wrapped('<parameter name="q"/>'), "'q'.*add its <define-parameter>"),
        (
            wrapped().replace(
                "</opsa-mef>",
                '<define-fault-tree name="FT2"/></opsa-mef>',
            ),
            r"2 fault trees, \['FT', 'FT2'\]: name the one to read, tree=",
        ),
        (
            wrapped().replace(
                "<define-basic-event",
                '<define-gate name="other"><and><basic-event name="a"/>'
                '<basic-event name="b"/></and></define-gate>'
                "<define-basic-event",
                1,
            ),
            "2 gates that are no gate's input.*top=",
        ),
        ("<fault-tree/>", "root is <fault-tree>"),
        ("<opsa-mef><unclosed></opsa-mef>", "well-formed"),
    ],
)
def test_a_bad_document_is_refused_with_the_fix(document, match):
    with pytest.raises(ValueError, match=match):
        FaultTree.from_open_psa(document)


def test_the_tree_and_top_choose_what_is_read():
    two = wrapped().replace(
        "</opsa-mef>",
        '<define-fault-tree name="FT2"><define-gate name="t2"><and>'
        '<basic-event name="a"/><basic-event name="b"/></and></define-gate>'
        "</define-fault-tree></opsa-mef>",
    )
    tree = FaultTree.from_open_psa(two, tree="FT2")
    assert tree.gates == {"t2": ("and", ["a", "b"])}
    assert tree.top_event_probability() == pytest.approx(0.1 * 0.2)
    with pytest.raises(ValueError, match="no fault tree 'FT3'"):
        FaultTree.from_open_psa(two, tree="FT3")
    many = wrapped().replace(
        "<define-basic-event",
        '<define-gate name="other"><and><basic-event name="a"/>'
        '<basic-event name="b"/></and></define-gate><define-basic-event',
        1,
    )
    tree = FaultTree.from_open_psa(many, top="other")
    assert tree.gates == {"other": ("and", ["a", "b"])}
    with pytest.raises(ValueError, match="mission_time must be"):
        FaultTree.from_open_psa(wrapped(), mission_time=-1.0)


# -- repaired and tested events ----------------------------------------------

REPAIRED = {
    kind: tree
    for kind, tree in fault_tree_kinds().items()
    if tree.is_repairable
}
#: Times between the tests of the catalogue's tested events (a value at a
#: test is the one just after it, which a tool may give as the one before).
TIMES = np.array([3.0, 39.5, 50.5, 77.0, 123.4, 301.0, 1234.5])


def same_specs(again: dict, tree: dict) -> None:
    """Each event's spec read back is the one written: its life and
    repair by their parameters, and its inspection as given."""
    assert set(again) == set(tree)
    for name, spec in again.items():
        original = tree[name]
        assert set(spec) == set(original)
        for key in ("reliability", "repairability"):
            if isinstance(original[key], str):
                assert spec[key] == original[key]
            else:
                assert list(spec[key].params) == list(original[key].params)
                assert spec[key].dist.name == original[key].dist.name
        if "inspection" in original:
            given = {k: float(v) for k, v in original["inspection"].items()}
            assert spec["inspection"] == given


def test_the_repaired_catalogue_trees_are_exported_or_refused_by_event():
    """Every repaired tree of the catalogue: as GLM and periodic-test
    expressions where its events' models are the format's, and refused,
    naming the event and its model, where they are not."""
    assert set(REPAIRED) == {
        "repaired",
        "tested",
        "repeated, repaired",
        "repaired, not gate",
    }
    for kind, tree in REPAIRED.items():
        if kind == "tested":
            continue
        with pytest.raises(
            NotImplementedError,
            match="Repaired event 'a': its life is a Weibull.*pass t",
        ):
            tree.to_open_psa()


def exponential(tree: FaultTree) -> FaultTree:
    """``tree`` with each event's life and repair exponential, of the
    same means, so the format holds it."""
    specs = {}
    for name, spec in tree.events.items():
        spec = dict(spec)
        for key in ("reliability", "repairability"):
            if not isinstance(spec[key], str):
                spec[key] = E([1.0 / float(spec[key].mean())])
        specs[name] = spec
    return FaultTree(tree.gates, specs, top=tree.top)


@pytest.mark.parametrize("kind", sorted(REPAIRED))
def test_a_repaired_tree_reads_back_as_it_was_written(kind):
    tree = REPAIRED[kind]
    if kind != "tested":
        tree = exponential(tree)
    document = tree.to_open_psa()
    again = FaultTree.from_open_psa(document)
    assert again.gates == tree.gates and again.top == tree.top
    same_specs(again.events, tree.events)
    np.testing.assert_allclose(
        again.point_unavailability(TIMES),
        tree.point_unavailability(TIMES),
        rtol=1e-15,
        atol=0,
    )
    if kind == "tested":
        # Each form of the periodic test, and the GLM.
        assert document.count("<periodic-test>") == 2
        assert document.count("<GLM>") == 1
        assert '<bool value="false"/>' in document


@pytest.mark.parametrize("kind", sorted(REPAIRED))
def test_a_repaired_tree_at_a_time_is_its_events_unavailabilities(kind):
    tree = REPAIRED[kind]
    again = FaultTree.from_open_psa(tree.to_open_psa(t=40.0))
    assert again.is_fixed and again.gates == tree.gates
    assert again.top_event_probability() == pytest.approx(
        float(tree.point_unavailability(40.0)), rel=1e-15, abs=0
    )


def single(expression: str) -> str:
    """A document of one basic event, the top gate's only input."""
    return (
        '<opsa-mef><define-fault-tree name="FT"><define-gate name="top">'
        '<or><basic-event name="a"/></or></define-gate>'
        f'<define-basic-event name="a">{expression}</define-basic-event>'
        "</define-fault-tree></opsa-mef>"
    )


def floats(*values) -> str:
    return "".join(f'<float value="{v}"/>' for v in values)


MISSION = "<system-mission-time/>"


def test_a_glm_is_the_formula_of_the_mef():
    lam, mu = 0.003, 0.08
    tree = FaultTree.from_open_psa(
        single(f"<GLM>{floats(0, lam, mu)}{MISSION}</GLM>")
    )
    r = lam + mu
    # P(t) = lambda / r - (lambda - gamma r) / r exp(-r t), gamma = 0.
    expected = lam / r - lam / r * np.exp(-r * TIMES)
    np.testing.assert_allclose(
        tree.point_unavailability(TIMES), expected, rtol=1e-13
    )
    # Never repaired (mu = 0): the exponential life itself.
    life = FaultTree.from_open_psa(
        single(f"<GLM>{floats(0, lam, 0)}{MISSION}</GLM>")
    )
    assert not life.is_repairable
    assert life.top_event_probability(100.0) == pytest.approx(
        -np.expm1(-lam * 100.0), rel=1e-15
    )


def periodic_test(lam, mu, tau, theta, test, t):
    """The MEF's periodic test, its unavailability at ``t`` (not at a
    test): a multi-phase Markov chain, as the format describes it, of a
    unit available (A), failed unseen (F), in repair (R), off line in its
    test (T), and found failed in its test (W); with no failure in a test
    (lambda* = 0), none caused by it (gamma = 0), off line in it (x = 0),
    every failure found (sigma = 1) and good restarts (omega = 0). Tests
    at theta, theta + tau, ..., instant for ``test = 0``; a unit in repair
    is not tested; a failure found is repaired from the test's end, at
    once for ``mu = None``."""
    from scipy.linalg import expm

    A, F, R, T, W = range(5)
    rates = np.zeros((5, 5))
    rates[A, F] = lam
    if mu is not None:
        rates[R, A] = mu
    rates -= np.diag(rates.sum(axis=1))
    p, now, start = np.eye(5)[A], 0.0, theta

    def run(p, until):
        return p @ expm(rates * (until - now))

    while t >= start:
        p, now = run(p, start), start
        if test > 0.0:
            p[T], p[A] = p[A], 0.0
            p[W], p[F] = p[F], 0.0
            if t < start + test:
                return 1.0 - run(p, t)[A]
            p, now = run(p, start + test), start + test
            p[A], p[T] = p[A] + p[T], 0.0
            p[R], p[W] = p[R] + p[W], 0.0
        else:
            p[A if mu is None else R] += p[F]
            p[F] = 0.0
        start += tau
    return 1.0 - run(p, t)[A]


@pytest.mark.parametrize(
    "args, expression",
    [
        # 4 arguments: an instant repair, the first test at tau, within
        # the first interval, and at 0 (a test of a new unit, so at tau).
        ((0.004, None, 50.0, 50.0, 0.0), floats(0.004, 50, 50)),
        ((0.004, None, 50.0, 20.0, 0.0), floats(0.004, 50, 20)),
        ((0.004, None, 50.0, 0.0, 0.0), floats(0.004, 50, 0)),
        # 5 arguments: a repair rate.
        ((0.004, 0.05, 50.0, 30.0, 0.0), floats(0.004, 0.05, 50, 30)),
        # 11 arguments: a test that takes time.
        (
            (0.01, 0.02, 50.0, 10.0, 3.0),
            floats(0.01, 0, 0.02, 50, 10, 0, 3)
            + '<bool value="false"/>'
            + floats(1, 0),
        ),
    ],
)
def test_a_periodic_test_is_the_mef_multi_phase_chain(args, expression):
    tree = FaultTree.from_open_psa(
        single(f"<periodic-test>{expression}{MISSION}</periodic-test>")
    )
    expected = [periodic_test(*args, t) for t in TIMES]
    # A tested unit's values with a repair that takes time are worked out
    # numerically (to some 1e-9 here); with an instant repair, in closed
    # form.
    rtol = 1e-13 if args[1] is None else 1e-8
    np.testing.assert_allclose(
        tree.point_unavailability(TIMES), expected, rtol=rtol
    )


TEST_11 = (
    "<periodic-test>"
    + floats(0.01, "{lam_test}", 0.02, 50, 10, "{gamma}", 3)
    + '<float value="{x}"/>'
    + floats("{sigma}", "{omega}")
    + MISSION
    + "</periodic-test>"
)


@pytest.mark.parametrize(
    "expression, match",
    [
        (f"<GLM>{floats(0.1, 0.01, 0.1)}{MISSION}</GLM>", "gamma = 0.1"),
        *(
            (
                TEST_11.format(
                    **{
                        "lam_test": 0,
                        "gamma": 0,
                        "x": 0,
                        "sigma": 1,
                        "omega": 0,
                        name: value,
                    }
                ),
                match,
            )
            for name, value, match in (
                ("lam_test", 0.001, "lambda\\* .* = 0.001"),
                ("gamma", 0.01, "gamma .* = 0.01"),
                ("x", 1, "x .* = 1.0"),
                ("sigma", 0.9, "sigma .* = 0.9"),
                ("omega", 0.05, "omega .* = 0.05"),
            )
        ),
        (
            f"<periodic-test>{floats(0.01, 50, 70)}{MISSION}</periodic-test>",
            "first test at theta = 70.0",
        ),
        (
            f"<periodic-test>{floats(0.01, 0, 50, 30)}{MISSION}"
            "</periodic-test>",
            "repair rate of 0",
        ),
    ],
)
def test_a_repaired_or_tested_form_not_held_is_refused_saying_why(
    expression, match
):
    with pytest.raises(NotImplementedError, match=match):
        FaultTree.from_open_psa(single(expression))


def test_a_periodic_test_needs_its_arguments():
    with pytest.raises(ValueError, match="takes 4 or 5 or 11 arguments"):
        FaultTree.from_open_psa(
            single(
                f"<periodic-test>{floats(0.01, 50)}{MISSION}</periodic-test>"
            )
        )
    with pytest.raises(ValueError, match="positive, finite rate"):
        FaultTree.from_open_psa(
            single(f"<GLM>{floats(0, 0, 0.1)}{MISSION}</GLM>")
        )


@pytest.mark.parametrize(
    "spec, match",
    [
        (
            {"reliability": E([0.01]), "repairability": W([5.0, 2.0])},
            "its repair time is a Weibull",
        ),
        (
            {
                "reliability": E([0.01]),
                "repairability": "instant",
                "inspection": {
                    "interval": 50.0,
                    "coverage": 0.9,
                    "full_test": 500.0,
                },
            },
            "\\['coverage', 'full_test'\\]",
        ),
        (
            {
                "reliability": E([0.01]),
                "repairability": E([0.5]),
                "inspection": {"interval": 50.0, "duration": E([1.0])},
            },
            "time drawn from",
        ),
        (
            {
                "reliability": E([0.01]),
                "repairability": "instant",
                "inspection": {"interval": 50.0, "duration": 1.0},
            },
            "repaired at once after a test that takes time",
        ),
    ],
)
def test_the_writer_refuses_a_repaired_event_the_format_has_not(spec, match):
    tree = FaultTree(
        {"top": ("or", ["a", "b"])},
        {"a": spec, "b": {"reliability": E([0.01]), "repairability": E([1])}},
    )
    with pytest.raises(NotImplementedError, match=match):
        tree.to_open_psa()
    # At a time, each event is its unavailability.
    assert "<float" in tree.to_open_psa(t=10.0)


# -- common-cause models -----------------------------------------------------

VOTE = {"top": ("or", ["vote", "d"]), "vote": ("vote", 2, ["a", "b", "c"])}
ALIKE = {"a": E([0.001]), "b": E([0.001]), "c": E([0.001]), "d": 0.01}
CCF_TIMES = np.array([1.0, 10.0, 50.0])


@pytest.mark.parametrize(
    "model, written, again",
    [
        (BetaFactor(0.1), "beta-factor", BetaFactor(0.1)),
        (
            MGL(0.1, 0.3, shocks="independent"),
            "MGL",
            MGL(0.1, 0.3, shocks="independent"),
        ),
        (
            AlphaFactor(
                [0.95, 0.04, 0.01],
                testing="non-staggered",
                shocks="independent",
            ),
            "alpha-factor",
            AlphaFactor(
                [0.95, 0.04, 0.01],
                testing="non-staggered",
                shocks="independent",
            ),
        ),
        # Staggered alpha factors as the non-staggered ones of the same
        # Q_k: alpha_k / k, scaled.
        (
            AlphaFactor([0.95, 0.04, 0.01], shocks="independent"),
            "alpha-factor",
            None,
        ),
    ],
)
def test_a_common_cause_group_reads_back_as_it_was_written(
    model, written, again
):
    tree = FaultTree(
        VOTE, ALIKE, ccf_groups=[CCFGroup(["a", "b", "c"], model)]
    )
    document = tree.to_open_psa()
    assert f'model="{written}"' in document
    read = FaultTree.from_open_psa(document)
    assert list(read.ccf_groups[0].members) == ["a", "b", "c"]
    if again is not None:
        assert read.ccf_groups[0].model == again
    else:
        alphas = np.array([0.95, 0.04 / 2, 0.01 / 3])
        np.testing.assert_allclose(
            read.ccf_groups[0].model.alphas, alphas / alphas.sum(), rtol=1e-15
        )
    np.testing.assert_allclose(
        read.top_event_probability(CCF_TIMES),
        tree.top_event_probability(CCF_TIMES),
        rtol=1e-15,
        atol=0,
    )


def test_a_beta_factor_is_written_at_the_level_of_its_group():
    tree = FaultTree(
        VOTE, ALIKE, ccf_groups=[CCFGroup(["a", "b", "c"], BetaFactor(0.1))]
    )
    assert '<factor level="3">' in tree.to_open_psa()


def test_a_two_member_mgl_has_one_shared_cause_either_way():
    pair = {"top": ("and", ["a", "b"])}
    tree = FaultTree(
        pair,
        {"a": E([0.001]), "b": E([0.001])},
        ccf_groups=[CCFGroup(["a", "b"], MGL(0.2))],
    )
    again = FaultTree.from_open_psa(tree.to_open_psa())
    assert again.ccf_groups[0].model == MGL(0.2)
    assert again.top_event_probability(100.0) == pytest.approx(
        tree.top_event_probability(100.0), rel=1e-15, abs=0
    )


def mef_group(model: str, factors, q: float = 0.05, n: int = 3) -> str:
    """A document of ``n`` members in a group of ``model``, each failing
    with ``q``, under a 2-out-of-n vote."""
    names = "abcd"[:n]
    members = "".join(f'<basic-event name="{m}"/>' for m in names)
    written = "".join(
        f'<factor level="{level}"><float value="{v}"/></factor>'
        for level, v in factors
    )
    return (
        '<opsa-mef><define-fault-tree name="FT"><define-gate name="top">'
        f'<atleast min="2">{members}</atleast></define-gate>'
        f'<define-CCF-group name="g" model="{model}"><members>{members}'
        f'</members><distribution><float value="{q}"/></distribution>'
        f"<factors>{written}</factors></define-CCF-group>"
        "</define-fault-tree></opsa-mef>"
    )


def two_out_of(n: int, q_k) -> float:
    """The probability that 2 or more of ``n`` members fail, each specific
    set of ``k`` failing as an independent basic event with ``q_k[k-1]``
    (the format's expansion of a group), every state enumerated."""
    from itertools import combinations, product

    sets = [
        (frozenset(s), q_k[k - 1])
        for k in range(1, n + 1)
        for s in combinations(range(n), k)
    ]
    total = 0.0
    for state in product((0, 1), repeat=len(sets)):
        p, failed = 1.0, set()
        for on, (members, q) in zip(state, sets):
            p *= q if on else 1.0 - q
            if on:
                failed |= members
        total += p * (len(failed) >= 2)
    return total


Q = 0.05


@pytest.mark.parametrize(
    "model, factors, q_k",
    [
        # MGL, n = 3: Q_1 = (1 - b) Q, Q_2 = b (1 - g) Q / 2, Q_3 = b g Q.
        (
            "MGL",
            [(2, 0.1), (3, 0.3)],
            [0.9 * Q, 0.1 * 0.7 * Q / 2, 0.1 * 0.3 * Q],
        ),
        # The format's example of four members: rho 0.1, 0.2, 0.3.
        (
            "MGL",
            [(2, 0.1), (3, 0.2), (4, 0.3)],
            [
                0.9 * Q,
                0.1 * 0.8 * Q / 3,
                0.1 * 0.2 * 0.7 * Q / 3,
                0.1 * 0.2 * 0.3 * Q,
            ],
        ),
        # alpha-factor: Q_k = k / C(n-1, k-1) alpha_k / sum(i alpha_i) Q.
        (
            "alpha-factor",
            [(1, 0.95), (2, 0.04), (3, 0.01)],
            [
                0.95 / 1.06 * Q,
                2 / 2 * 0.04 / 1.06 * Q,
                3 * 0.01 / 1.06 * Q,
            ],
        ),
        # phi-factor: Q_k = phi_k Q, a member's sets summing to Q.
        (
            "phi-factor",
            [(1, 0.9), (2, 0.03), (3, 0.04)],
            [0.9 * Q, 0.03 * Q, 0.04 * Q],
        ),
    ],
)
def test_a_common_cause_group_is_the_mef_formula(model, factors, q_k):
    n = len(q_k)
    tree = FaultTree.from_open_psa(mef_group(model, factors, Q, n))
    assert tree.top_event_probability() == pytest.approx(
        two_out_of(n, q_k), rel=1e-13
    )


@pytest.mark.parametrize(
    "tree, match",
    [
        (fault_tree_kinds()["common cause, MGL"], "basis='rate'"),
        (fault_tree_kinds()["common cause by rate"], "basis='rate'"),
        (
            FaultTree(
                VOTE,
                ALIKE,
                ccf_groups=[CCFGroup(["a", "b", "c"], MGL(0.1, 0.3))],
            ),
            "shocks='independent'",
        ),
        (
            FaultTree(
                VOTE,
                ALIKE,
                ccf_groups=[
                    CCFGroup(["a", "b", "c"], AlphaFactor([0.9, 0.08, 0.02]))
                ],
            ),
            "shocks='independent'",
        ),
    ],
)
def test_a_group_the_format_has_not_is_refused_saying_why(tree, match):
    with pytest.raises(NotImplementedError, match=match):
        tree.to_open_psa()


def test_a_group_read_that_no_model_here_holds_is_refused_saying_why():
    # SCRAM's phi factors sum to 1 themselves: a member then fails with
    # 1.06 Q through its sets, which no model here holds.
    with pytest.raises(NotImplementedError, match="1.06 Q"):
        FaultTree.from_open_psa(
            mef_group("phi-factor", [(1, 0.9), (2, 0.06), (3, 0.04)])
        )
    with pytest.raises(ValueError, match="level \\[2, 3\\].*\\[2\\]"):
        FaultTree.from_open_psa(mef_group("MGL", [(2, 0.1)]))
    with pytest.raises(ValueError, match="not an MEF common-cause model"):
        FaultTree.from_open_psa(mef_group("delta-factor", [(2, 0.1)]))
    # Members not below the top event: the group's sets cannot be kept.
    partial = mef_group("MGL", [(2, 0.1), (3, 0.3)]).replace(
        '<atleast min="2"><basic-event name="a"/><basic-event name="b"/>'
        '<basic-event name="c"/></atleast>',
        '<and><basic-event name="a"/><basic-event name="b"/></and>',
    )
    with pytest.raises(NotImplementedError, match="\\['c'\\].*top="):
        FaultTree.from_open_psa(partial)
