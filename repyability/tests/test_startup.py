"""Start-up costs: what ``import repyability`` loads, and the worker
processes of a parallel run (``n_jobs``)."""

import multiprocessing
import os
import subprocess
import sys

import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _montecarlo as montecarlo

# Loaded only by the calls that need them: FFT convolution (cold standby,
# block replacement), the progress bar, and the parallel runs' processes.
DEFERRED = (
    "scipy.signal",
    "tqdm",
    "concurrent.futures.process",
    "multiprocessing",
)

HAS_FORKSERVER = "forkserver" in multiprocessing.get_all_start_methods()
GET_CONTEXT = multiprocessing.get_context


def default_context(name):
    """``multiprocessing.get_context`` with ``name`` as the default."""
    return lambda method=None: GET_CONTEXT(method or name)


def test_import_leaves_out_what_only_some_calls_need():
    code = (
        "import sys, repyability; "
        f"print([m for m in {DEFERRED!r} if m in sys.modules])"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert loaded == "[]"


def test_all_jobs_means_the_cpus_this_process_may_use():
    cpus = montecarlo.available_cpus()
    assert 1 <= cpus <= (os.cpu_count() or 1)
    assert montecarlo.jobs(-1) == cpus
    if hasattr(os, "sched_getaffinity"):
        assert cpus == len(os.sched_getaffinity(0))


@pytest.fixture
def forkserver_by_default(monkeypatch):
    """Make forkserver the default start method (as on Linux from Python
    3.14), restoring the server's preload list afterwards."""
    from multiprocessing import forkserver

    server = forkserver._forkserver
    preload = server._preload_modules
    monkeypatch.setattr(
        multiprocessing, "get_context", default_context("forkserver")
    )
    yield server
    server._preload_modules = preload


@pytest.mark.skipif(not HAS_FORKSERVER, reason="no forkserver here")
def test_the_forkserver_preloads_repyability(forkserver_by_default):
    server = forkserver_by_default
    server._preload_modules = ["__main__", "numpy"]
    pool = montecarlo.process_pool(2)
    pool.shutdown()
    # Added to what was there, and only once.
    assert server._preload_modules == ["__main__", "numpy", "repyability"]
    montecarlo.process_pool(2).shutdown()
    assert server._preload_modules.count("repyability") == 1


@pytest.mark.skipif(not HAS_FORKSERVER, reason="no forkserver here")
def test_a_forkserver_run_matches_a_forked_one(forkserver_by_default):
    unit = {
        "reliability": surv.Exponential.from_params([0.1]),
        "repairability": surv.Exponential.from_params([1.0]),
    }
    plant = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "c"), ("b", "c"), ("c", "t")],
        {n: unit for n in "abc"},
    )
    kwargs = dict(t_simulation=50.0, N=600, seed=3, n_jobs=2)
    served = plant.availability(**kwargs)
    with pytest.MonkeyPatch.context() as m:
        m.setattr(multiprocessing, "get_context", default_context("fork"))
        forked = plant.availability(**kwargs)
    assert served.system_failures == forked.system_failures
    assert (served.uptimes == forked.uptimes).all()
