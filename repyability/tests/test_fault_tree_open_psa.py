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

from repyability import BetaFactor, CCFGroup, FaultTree
from repyability.tests.catalogue import fault_tree_kinds

E, W = surv.Exponential.from_params, surv.Weibull.from_params
T = np.array([5.0, 40.0, 120.0, 400.0])


def exported():
    """The catalogue's trees the writer exports, by kind."""
    out = {}
    for kind, tree in fault_tree_kinds().items():
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
                '<GLM><float value="0"/><float value="0.01"/>'
                '<float value="0.1"/><system-mission-time/></GLM>'
            ),
        ),
        (
            "periodic-test",
            wrapped(
                '<periodic-test><float value="0.01"/><float value="100"/>'
                '<float value="0"/><system-mission-time/></periodic-test>'
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
        *(
            (
                model,
                '<opsa-mef><define-fault-tree name="FT">'
                '<define-gate name="top"><or><basic-event name="a"/>'
                '<basic-event name="b"/></or></define-gate>'
                f'<define-CCF-group name="g" model="{model}">'
                '<members><basic-event name="a"/><basic-event name="b"/>'
                '</members><distribution><float value="0.1"/></distribution>'
                '<factors><factor level="1"><float value="0.9"/></factor>'
                '<factor level="2"><float value="0.1"/></factor></factors>'
                "</define-CCF-group></define-fault-tree></opsa-mef>",
            )
            for model in ("MGL", "alpha-factor", "phi-factor")
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
