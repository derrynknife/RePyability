"""The decision diagram's search and replay (#202, #207): one search
(``bdd.search``) and one replay (``shannon.replay`` and
``replay_gradient``), run as Python or compiled by numba (``_bdd_kernel``).
Either way they must make the plans recorded in ``bdd_records.json``, step
for step, give up at the step recorded, and replay the values recorded, to
the last bit. ``_bdd_cases`` says what is recorded, and re-records it."""

import json

import numpy as np
import pytest

from repyability import RBD
from repyability.rbd import _compiled, bdd, modular
from repyability.rbd.modular import decompose
from repyability.tests import _bdd_cases as cases

RECORDS = json.loads(cases.RECORDS.read_text())
CORES = {
    label: cases.rebuilt(core) for label, core in RECORDS["cores"].items()
}


@pytest.fixture(params=["python", "compiled"])
def mode(request, monkeypatch):
    """The search and replay as Python, or compiled (with numba)."""
    compiled = request.param == "compiled"
    if compiled and not _compiled.available():
        pytest.skip("numba is not installed")
    monkeypatch.setattr(bdd, "COMPILED", compiled)
    monkeypatch.setattr(modular, "COMPILED_STEPS", 1 if compiled else 10**9)
    return request.param


def test_the_records_cover_every_kind_of_core():
    labels = " ".join(RECORDS["plans"])
    for kind in ("random", "shared", "bridges 12", "grid 10x20"):
        assert kind in labels
    assert any(
        len(set(args[6].values())) < len(args[6]) for args in CORES.values()
    ), "a component drawn in several places"
    assert max(len(args[0]) for args in CORES.values()) == 200
    assert set(RECORDS["steps"]) <= set(CORES)


def test_the_plans_are_the_recorded_ones(mode):
    wrong = [
        label
        for label, args in CORES.items()
        if cases.digest(bdd.build(*args)) != RECORDS["plans"][label]
    ]
    assert not wrong


def test_too_meshed_at_the_recorded_step(mode, monkeypatch):
    # The search gives up (TooLarge) a step below what a core takes, and
    # builds the recorded plan at it.
    for label, taken in RECORDS["steps"].items():
        monkeypatch.setattr(bdd, "STEP_LIMIT", taken - 1)
        with pytest.raises(bdd.TooLarge):
            bdd.build(*CORES[label])
        monkeypatch.setattr(bdd, "STEP_LIMIT", taken)
        plan = bdd.build(*CORES[label])
        assert cases.digest(plan) == RECORDS["plans"][label], label


def test_the_values_are_the_recorded_ones(mode):
    assert cases.replay_values() == RECORDS["values"]


def test_auto_compiles_only_large_cores(monkeypatch):
    if not _compiled.available():
        pytest.skip("numba is not installed")
    from repyability.rbd import _bdd_kernel

    compiled = []
    real = _bdd_kernel.run_search

    def spy(*args):
        compiled.append(args[0][0])
        return real(*args)

    monkeypatch.setattr(_bdd_kernel, "run_search", spy)
    decompose(RBD(cases.grid(3, 4)).G, "s", "t", core="bdd")
    assert not compiled
    decompose(RBD(cases.grid(9, 18)).G, "s", "t", core="bdd")
    assert compiled


def test_probabilities_not_numbers_are_replayed_in_python(monkeypatch):
    # Values autograd traces (or anything but numbers and arrays of them)
    # keep the Python replay.
    if not _compiled.available():
        pytest.skip("numba is not installed")
    graph = decompose(RBD(cases.grid(4, 4)).G, "s", "t", core="bdd")
    monkeypatch.setattr(modular, "COMPILED_STEPS", 1)
    terms = len(graph.terms)
    assert graph._compiled_terms([0.5] * terms, [0.5] * terms, None)
    traced = [object()] * terms
    assert graph._compiled_terms(traced, traced, None) is None
    half = np.full(3, 0.5)
    assert graph._compiled_terms([half] * terms, [half] * terms, (3,))
