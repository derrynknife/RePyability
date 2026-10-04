"""Systems of models fitted in surpyval run in worker processes (#181).

A surpyval fit holds a closure, so pickle cannot take it (SurPyval#573),
and ``n_jobs`` could not send such a system to its workers. A model that
pickle refuses is sent in its saved form instead (``montecarlo.dumps``),
and rebuilt there: the results are those of the run in one process."""

import pickle

import numpy as np
import pytest
import surpyval as surv

from repyability import NonRepairableRBD, RepairableRBD
from repyability.rbd import _montecarlo as montecarlo

DATA = np.array([100.0, 150.0, 200.0, 260.0, 300.0, 420.0])
PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def fits():
    return surv.Weibull.fit(DATA), surv.Gamma.fit(DATA)


def test_a_fit_does_not_pickle_but_goes_to_a_worker_all_the_same():
    weibull, _ = fits()
    try:
        pickle.dumps(weibull)
    except Exception:
        pass
    else:  # pragma: no cover - once surpyval's fits pickle (SurPyval#573)
        pytest.skip("this surpyval's fits pickle")
    rbd = NonRepairableRBD(PARALLEL, {"a": weibull, "b": weibull})
    loaded = pickle.loads(montecarlo.dumps(rbd))
    # One model for both nodes still, rebuilt from its saved form.
    assert loaded.reliabilities["a"] is loaded.reliabilities["b"]
    x = np.array([50.0, 200.0, 400.0])
    np.testing.assert_array_equal(loaded.sf(x), rbd.sf(x))


def test_lifetimes_of_fitted_models_are_drawn_in_parallel():
    weibull, gamma = fits()
    rbd = NonRepairableRBD(PARALLEL, {"a": weibull, "b": gamma})
    np.testing.assert_array_equal(
        rbd.random(25_000, seed=1, n_jobs=2),
        rbd.random(25_000, seed=1, n_jobs=1),
    )
    assert rbd.mean(
        method="simulate", mc_samples=20_000, seed=1, n_jobs=2
    ) == rbd.mean(method="simulate", mc_samples=20_000, seed=1, n_jobs=1)


def test_a_repairable_system_of_fitted_models_runs_in_parallel():
    weibull, gamma = fits()
    rbd = RepairableRBD(
        PARALLEL,
        {
            "a": {
                "reliability": weibull,
                "repairability": surv.Exponential.from_params([0.5]),
            },
            "b": {"reliability": weibull, "repairability": gamma},
        },
        repair_crews=1,
    )
    alone = rbd.availability(1000.0, mc_samples=600, seed=4, engine="python")
    shared = rbd.availability(
        1000.0, mc_samples=600, seed=4, engine="python", n_jobs=2
    )
    np.testing.assert_array_equal(shared.uptimes, alone.uptimes)
    assert shared.system_uptime == alone.system_uptime
    assert rbd.simulate_timelines(
        300.0, mc_samples=40, seed=2, engine="python", n_jobs=2
    ) == rbd.simulate_timelines(300.0, mc_samples=40, seed=2, engine="python")


def test_what_cannot_be_sent_to_a_worker_is_said_plainly():
    with pytest.raises(ValueError, match="cannot be sent to worker processes"):
        montecarlo.dumps(lambda x: x)
