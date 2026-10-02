"""Simulation engines other packages add (``repyability.rbd.engines``).

An engine registered under the ``repyability.engines`` entry point group (or
with ``engines.register``) runs ``RepairableRBD`` simulations by its name,
and ``engine="auto"`` prefers it to numba when its priority is higher. These
tests register engines that run the Python loop (or numba's) under another
name, so the results must be the Python engine's to the last bit.
"""

import warnings
from importlib import metadata

import pytest

from repyability.rbd import _compiled, engines, repairable_rbd
from repyability.tests.test_simulation_engines import (
    identical,
    needs_numba,
    plain_rbds,
    repairable_rbds,
)


class Engine:
    """An engine that runs the Python loop (or, with ``numba``, numba's)
    under its own name."""

    api = engines.API

    def __init__(self, name="borrowed", priority=1, numba=False):
        self.name = name
        self.priority = priority
        self.numba = numba
        self.here = True
        self.loads = 0
        self.runs = 0
        self.failure = None

    def available(self):
        return self.here

    def worthwhile(self, draws):
        return draws > 0

    def load(self):
        self.loads += 1
        if self.failure is not None:
            raise ImportError(self.failure)

    def runner(
        self, rbd, plan, tally, progress, working, broken, method, jobs
    ):
        self.runs += 1
        if self.numba:
            return _compiled.Runner(
                rbd, plan, tally, progress, working, broken, method, jobs
            )
        widths = {name: spec.width for name, spec in plan.specs.items()}
        context = (
            tally.t_simulation,
            working,
            broken,
            method,
            None,
            plan.entropy,
            plan.antithetic,
            widths,
        )
        return repairable_rbd._PythonRunner(
            rbd, tally, progress, context, jobs
        )


@pytest.fixture(autouse=True)
def registry(monkeypatch):
    """A registry of its own for each test."""
    monkeypatch.setattr(engines, "_engines", {})
    monkeypatch.setattr(engines, "_failed", {})


def test_an_engine_runs_by_its_name_with_the_same_results():
    engine = Engine()
    engines.register(engine)
    rbd = plain_rbds()["bridge"]
    for options in (
        dict(t_simulation=150.0, mc_samples=30, seed=3),
        dict(t_simulation=150.0, mc_samples=30, seed=4, antithetic=True),
        dict(t_simulation=150.0, mc_samples=30, seed=5, working_nodes=["a"]),
    ):
        identical(
            rbd.availability(engine="python", **options),
            rbd.availability(engine="borrowed", **options),
        )
    assert engine.runs == 3 and engine.loads == 3
    costed = plain_rbds()["costed"]
    identical(
        costed.cost(100.0, mc_samples=20, seed=6, engine="python"),
        costed.cost(100.0, mc_samples=20, seed=6, engine="borrowed"),
    )


def test_auto_prefers_an_engine_of_higher_priority(monkeypatch):
    engine = Engine(priority=1)
    engines.register(engine)
    rbd = plain_rbds()["koon"]
    assert _compiled.preferred() == "borrowed"
    assert rbd.analysis_routes()["availability"].engine == "borrowed"
    identical(
        rbd.availability(100.0, mc_samples=20, seed=2),
        rbd.availability(100.0, mc_samples=20, seed=2, engine="python"),
    )
    assert engine.runs == 1
    # Not when the run is too short to repay loading it...
    monkeypatch.setattr(engine, "worthwhile", lambda draws: False)
    rbd.availability(100.0, mc_samples=20, seed=2)
    assert engine.runs == 1
    # ... nor for what the compiled engines do not simulate.
    monkeypatch.setattr(engine, "worthwhile", lambda draws: True)
    repairable_rbds()["maintained"].availability(100.0, mc_samples=5, seed=2)
    assert engine.runs == 1


def test_numba_comes_before_an_engine_of_priority_zero_or_less(monkeypatch):
    engines.register(Engine(priority=0))
    monkeypatch.setattr(_compiled, "available", lambda: True)
    assert _compiled.preferred() == "numba"
    monkeypatch.setattr(_compiled, "available", lambda: False)
    assert _compiled.preferred() == "borrowed"


def test_the_highest_priority_usable_engine_is_preferred():
    low, high = Engine("low", priority=1), Engine("high", priority=5)
    engines.register(low)
    engines.register(high)
    assert _compiled.preferred() == "high"
    high.here = False
    assert _compiled.preferred() == "low"


def test_an_engine_that_cannot_load_falls_back():
    engine = Engine()
    engine.failure = "no compiler"
    engines.register(engine)
    rbd = plain_rbds()["bridge"]
    with pytest.warns(RuntimeWarning, match="could not be loaded"):
        fallen = rbd.availability(100.0, mc_samples=20, seed=4)
    identical(
        fallen, rbd.availability(100.0, mc_samples=20, seed=4, engine="python")
    )
    assert engine.runs == 0
    # It is not tried again, and asked for outright, it raises its error.
    assert _compiled.preferred() != "borrowed"
    with pytest.raises(ImportError, match="no compiler"):
        rbd.availability(100.0, mc_samples=20, seed=4, engine="borrowed")
    assert engine.loads == 1


def test_an_engine_that_cannot_run_here_is_refused():
    engine = Engine()
    engine.here = False
    engines.register(engine)
    with pytest.raises(ImportError, match="cannot run here"):
        plain_rbds()["bridge"].availability(
            10.0, mc_samples=2, seed=1, engine="borrowed"
        )
    assert _compiled.preferred() != "borrowed"


def test_an_engine_refuses_what_the_compiled_engines_do_not_simulate():
    engines.register(Engine())
    with pytest.raises(NotImplementedError, match="preventive maintenance"):
        repairable_rbds()["maintained"].availability(
            100.0, mc_samples=5, seed=2, engine="borrowed"
        )


def test_an_unknown_engine_names_the_ones_there_are():
    engines.register(Engine())
    with pytest.raises(ValueError, match="'borrowed'"):
        plain_rbds()["bridge"].availability(
            10.0, mc_samples=2, seed=1, engine="x"
        )


@pytest.mark.parametrize(
    "change, problem",
    [
        (dict(name="numba"), "name"),
        (dict(name="auto"), "name"),
        (dict(api=engines.API + 1), "version"),
        (dict(runner=None), "lacks"),
    ],
)
def test_an_engine_that_does_not_fit_is_left_out(change, problem):
    engine = Engine()
    for key, value in change.items():
        setattr(engine, key, value)
    with pytest.warns(RuntimeWarning, match=problem):
        engines.register(engine)
    assert engines.registered() == {}


class _EntryPoint:
    def __init__(self, name, target):
        self.name = name
        self.value = f"tests:{name}"
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def test_engines_are_found_by_their_entry_points(monkeypatch):
    engine = Engine("plugged")
    points = [
        _EntryPoint("plugged", engine),
        _EntryPoint("broken", ImportError("missing dependency")),
    ]
    asked = []

    def entry_points(group):
        asked.append(group)
        return points

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    monkeypatch.setattr(engines, "_engines", None)
    with pytest.warns(RuntimeWarning, match="missing dependency"):
        assert engines.registered() == {"plugged": engine}
    assert asked == [engines.GROUP]
    # Found once.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        engines.registered()
    assert asked == [engines.GROUP]
    engines.unregister("plugged")
    assert engines.get("plugged") is None


@needs_numba
def test_an_engine_can_run_numbas_loop():
    engines.register(Engine(numba=True))
    rbd = plain_rbds()["koon"]
    options = dict(t_simulation=200.0, mc_samples=40, seed=7, n_jobs=2)
    identical(
        rbd.availability(engine="python", **options),
        rbd.availability(engine="borrowed", **options),
    )


def test_numbas_own_loop_runs_what_an_engine_does_not(monkeypatch):
    # Age and block replacement are numba's own loop's (#155): an engine of
    # the interface's version is never given them, and "auto" runs them on
    # numba even when it prefers the engine for plain components.
    engine = Engine(priority=5)
    engines.register(engine)
    monkeypatch.setattr(_compiled, "available", lambda: True)
    maintained = repairable_rbds()["maintained"]
    assert maintained.analysis_routes()["availability"].engine == "numba"
    koon = plain_rbds()["koon"].analysis_routes()["availability"]
    assert koon.engine == "borrowed"
