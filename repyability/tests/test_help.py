"""``help()`` reads well (#238): the package says where to start, the API
reference's cross-references read as names, and the sensitivity measures
name the Greeks the guide calls them by."""

import inspect
import pydoc

import pytest

import repyability
from repyability import FaultTree, NonRepairableRBD, RepairableRBD
from repyability.utils.docs import readable


def test_the_package_says_where_to_start():
    doc = repyability.__doc__
    for name in ("NonRepairableRBD", "RepairableRBD", "FaultTree"):
        assert name in doc
        assert hasattr(repyability, name)
    assert "analysis_routes()" in doc
    assert "https://derrynknife.github.io/RePyability/" in doc


def test_a_cross_reference_reads_as_its_name():
    target = "repyability.RepairableRBD.point_availability"
    text = f"see [`point_availability`][{target}]."
    assert readable(text) == "see ``point_availability``."
    assert readable(None) is None
    assert readable("a [link](guide/greeks.md)") == "a [link](guide/greeks.md)"


@pytest.mark.parametrize("cls", [NonRepairableRBD, RepairableRBD, FaultTree])
def test_help_shows_no_cross_reference_markup(cls):
    text = pydoc.render_doc(cls, renderer=pydoc.plaintext)
    assert "][repyability." not in text


@pytest.mark.parametrize(
    "method, greek",
    [
        ("birnbaum_importance", "*delta*"),
        ("parameter_sensitivity", "*deltas*"),
        ("differential_importance", "*DIM*"),
        ("joint_importance", "*gamma*"),
        ("barlow_proschan_importance", "*theta*"),
        ("uncertainty_importance", "*vega*"),
    ],
)
def test_the_sensitivities_name_their_greeks(method, greek):
    for cls in (NonRepairableRBD, RepairableRBD):
        assert greek in inspect.getdoc(getattr(cls, method))
    assert "*theta*" in inspect.getdoc(RepairableRBD.availability_rate)
    assert "*theta*" in inspect.getdoc(NonRepairableRBD.reliability_rate)
