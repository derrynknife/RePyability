"""The first system failure of a redundant repairable diagram whose
lives or repairs are not exponential, followed as phases
(``_phase_type``; ``first_failure_sf``, ``mean_time_to_first_failure``).

Checked against chains written out by hand where the phases are exact
(Erlang lives), against the bracket of ``first_failure_bounds``, against
seeded simulations, and for what the accuracy asked for (``rtol``)
promises."""

import math

import numpy as np
import pytest
import surpyval as surv
from scipy.linalg import expm

from repyability import NodeState, RepairableRBD
from repyability.rbd import _first_failure, _phase_type

E = surv.Exponential.from_params
W = surv.Weibull.from_params
G = surv.Gamma.from_params
LN = surv.LogNormal.from_params

PAIR = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]
TIMES = np.array([20.0, 50.0, 100.0, 150.0])
RTOL = 1e-3


def pair(life, repair, **options):
    unit = {"reliability": life, "repairability": repair}
    return RepairableRBD(PAIR, {"a": unit, "b": unit}, **options)


def failing(rbd, times=TIMES, **options):
    return 1.0 - rbd.first_failure_sf(times, **options)


def accuracy(failing, rtol=RTOL):
    """What ``rtol`` promises of each probability of failing."""
    return rtol * np.maximum(failing, 1e-3 * np.max(failing))


# -- exact phases -------------------------------------------------------------

BETA, MU = 0.02, 0.1


def erlang_unit():
    """One unit of Erlang-2 life (rate BETA a phase) and exponential
    repair: its generator over (first phase, second phase, in repair)."""
    return np.array(
        [
            [-BETA, BETA, 0.0],
            [0.0, -BETA, BETA],
            [MU, 0.0, -MU],
        ]
    )


def erlang_pair():
    """The pair's generator over the states with a unit up, by hand: the
    Kronecker sum of the units', less the state with both in repair."""
    one = erlang_unit()
    both = np.kron(one, np.eye(3)) + np.kron(np.eye(3), one)
    return both[:8, :8]


def test_an_erlang_pair_is_the_chain_written_out_by_hand():
    rbd = pair(G([2, BETA]), E([MU]))
    assert _first_failure._require_exact(rbd) == _first_failure.AS_PHASES
    t = erlang_pair()
    start = np.zeros(8)
    start[0] = 1.0
    by_hand = np.array([(start @ expm(t * x)).sum() for x in TIMES])
    np.testing.assert_allclose(
        rbd.first_failure_sf(TIMES), by_hand, rtol=1e-10
    )
    assert rbd.mean_time_to_first_failure() == pytest.approx(
        start @ np.linalg.solve(-t, np.ones(8)), rel=1e-10
    )
    # Exact phases need no levels: the accuracy asked for changes nothing.
    assert np.array_equal(
        rbd.first_failure_sf(TIMES, rtol=1e-2),
        rbd.first_failure_sf(TIMES, rtol=1e-6),
    )


def test_an_aged_erlang_unit_starts_in_its_phases_at_that_age():
    # At age a, an Erlang-2 life is in its second phase with probability
    # BETA a / (1 + BETA a). The phases of what is left of it are an
    # approximation, held to rtol.
    age = 60.0
    second = BETA * age / (1.0 + BETA * age)
    t = erlang_pair()
    start = np.zeros(8)
    start[0], start[3] = 1.0 - second, second
    by_hand = np.array([(start @ expm(t * x)).sum() for x in TIMES])
    rbd = pair(G([2, BETA]), E([MU]))
    state = {"a": NodeState(age=age)}
    value = rbd.first_failure_sf(TIMES, state=state)
    assert np.all(np.abs(value - by_hand) <= accuracy(1.0 - by_hand))
    assert rbd.mean_time_to_first_failure(state=state) == pytest.approx(
        start @ np.linalg.solve(-t, np.ones(8)), rel=RTOL
    )


def test_a_unit_down_at_the_start_is_in_its_repair():
    rbd = pair(G([2, BETA]), E([MU]))
    t = erlang_pair()
    start = np.zeros(8)
    start[6] = 1.0  # (in repair, first phase)
    by_hand = np.array([(start @ expm(t * x)).sum() for x in TIMES])
    np.testing.assert_allclose(
        rbd.first_failure_sf(TIMES, state={"a": NodeState(alive=False)}),
        by_hand,
        rtol=1e-10,
    )
    down = {n: NodeState(alive=False) for n in "ab"}
    assert rbd.first_failure_sf(10.0, state=down) == 0.0
    assert rbd.mean_time_to_first_failure(state=down) == 0.0


# -- phases that approximate -------------------------------------------------

CASES = {
    "weibull": lambda: pair(W([100.0, 2.0]), W([10.0, 2.0])),
    "lognormal": lambda: pair(LN([math.log(100.0), 0.5]), E([0.1])),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_values_lie_in_the_bracket_to_their_accuracy(case):
    rbd = CASES[case]()
    value = rbd.first_failure_sf(TIMES)
    bounds = rbd.first_failure_bounds(TIMES)
    slack = accuracy(1.0 - value)
    assert np.all(value >= bounds.lower - slack)
    assert np.all(value <= bounds.upper + slack)
    assert (
        rbd.mean_time_to_first_failure()
        >= rbd.mean_time_to_first_failure_bound()
    )


@pytest.mark.parametrize("case, seed", [("weibull", 11), ("lognormal", 12)])
def test_the_values_against_seeded_simulations(case, seed):
    rbd = CASES[case]()
    n = 200_000
    value = rbd.first_failure_sf(TIMES)
    simulated = rbd.first_failure_sf(
        TIMES, method="simulate", mc_samples=n, seed=seed
    )
    error = np.sqrt(value * (1.0 - value) / n)
    assert np.all(np.abs(simulated - value) <= 4.0 * error)


def test_one_crew_for_two_units_against_seeded_simulations():
    rbd = pair(W([100.0, 2.0]), LN([math.log(10.0), 0.5]), repair_crews=1)
    n = 200_000
    value = rbd.first_failure_sf(TIMES)
    simulated = rbd.first_failure_sf(
        TIMES, method="simulate", mc_samples=n, seed=13
    )
    error = np.sqrt(value * (1.0 - value) / n)
    assert np.all(np.abs(simulated - value) <= 4.0 * error)
    # Waiting for the one crew, the pair fails sooner than with two.
    assert np.all(value[1:] < CASES["weibull"]().first_failure_sf(TIMES)[1:])


def test_a_tighter_accuracy_moves_the_value_less(monkeypatch):
    # The Erlang pair with its lives taken as any other life's, as
    # mixtures of Erlangs on a grid, against its own chain by hand.
    own = _phase_type.exact
    monkeypatch.setattr(
        _phase_type,
        "exact",
        lambda model: None if len(np.ravel(model.params)) > 1 else own(model),
    )
    rbd = pair(G([2, BETA]), E([MU]))
    t = erlang_pair()
    start = np.zeros(8)
    start[0] = 1.0
    reference = 1.0 - np.array([(start @ expm(t * x)).sum() for x in TIMES])
    mean = start @ np.linalg.solve(-t, np.ones(8))
    errors, mean_errors = [], []
    for rtol in (1e-2, 1e-3, 1e-4):
        value = failing(rbd, rtol=rtol)
        errors.append(np.max(np.abs(value - reference) / reference))
        assert errors[-1] <= rtol
        mean_errors.append(
            abs(rbd.mean_time_to_first_failure(rtol=rtol) - mean)
        )
        assert mean_errors[-1] <= rtol * mean
    assert errors[2] < errors[0]
    assert mean_errors[2] < mean_errors[0]


def test_the_levels_stop_at_the_accuracy_asked_for(monkeypatch):
    rbd = pair(W([100.0, 2.0]), E([0.1]))
    taken = []
    built = _first_failure._Factored

    def counted(rbd, states, level, horizon):
        taken.append(level)
        return built(rbd, states, level, horizon)

    monkeypatch.setattr(_first_failure, "_Factored", counted)
    rbd.first_failure_sf(TIMES, rtol=1e-2)
    loose = len(taken)
    rbd.first_failure_sf(TIMES, rtol=1e-4)
    assert _phase_type.FEWEST <= loose < len(taken) - loose
    assert taken[:loose] == list(_phase_type.LEVELS[:loose])


def test_the_series_factors_are_chained_each_alone():
    wear, repair = W([100.0, 2.0]), E([0.1])
    unit = {"reliability": wear, "repairability": repair}
    last = {"reliability": W([400.0, 1.5]), "repairability": repair}
    edges = [
        ("s", "a"),
        ("s", "b"),
        ("a", "c"),
        ("b", "c"),
        ("c", "d"),
        ("c", "e"),
        ("d", "t"),
        ("e", "t"),
    ]
    rbd = RepairableRBD(
        edges, {"a": unit, "b": unit, "c": last, "d": unit, "e": unit}
    )
    one = pair(wear, repair).first_failure_sf(TIMES)
    alone = np.asarray(last["reliability"].sf(TIMES))
    value = rbd.first_failure_sf(TIMES)
    expected = one * one * alone
    assert np.all(np.abs(value - expected) <= 3 * accuracy(1 - expected))
    # The mean is the area under the product.
    grid = np.linspace(0.0, 2000.0, 4001)
    curve = rbd.first_failure_sf(grid)
    area = float(np.sum((curve[1:] + curve[:-1]) / 2 * np.diff(grid)))
    assert curve[-1] < 1e-6
    assert rbd.mean_time_to_first_failure() == pytest.approx(area, rel=2e-3)


def test_instant_repairs_start_a_new_life():
    rbd = pair(W([100.0, 2.0]), "instant")
    assert rbd.first_failure_sf(TIMES) == pytest.approx(1.0, abs=1e-12)
    assert rbd.mean_time_to_first_failure() == math.inf


# -- sizes and refusals -----------------------------------------------------


def test_a_small_diagram_takes_few_states():
    rbd = pair(W([100.0, 2.0]), E([0.1]))
    sizes = [
        _first_failure._Absorbing(rbd, {}, None, level, 150.0).generator.shape[
            0
        ]
        for level in _phase_type.LEVELS[:4]
    ]
    # Each unit's life takes a number of phases in proportion to the level
    # (and a margin), so the pair's states grow as its square.
    assert sizes == sorted(sizes)
    assert sizes[0] < 1_000
    assert sizes[3] < 40_000
    assert sizes[3] / sizes[2] < 4.0


def test_a_chain_too_large_is_refused_with_what_to_use_instead():
    unit = {"reliability": W([100.0, 2.0]), "repairability": W([10.0, 2.0])}
    names = "abcd"
    rbd = RepairableRBD(
        [("s", n) for n in names] + [(n, "t") for n in names],
        {n: unit for n in names},
    )
    for call in (
        lambda: rbd.first_failure_sf(10.0),
        lambda: rbd.mean_time_to_first_failure(),
    ):
        with pytest.raises(NotImplementedError, match="states") as error:
            call()
        assert "first_failure_bounds" in str(error.value)
        assert "method='simulate'" in str(error.value)
    assert rbd.analysis_routes()["first_failure_sf"].route == "refused"


def test_values_that_do_not_settle_say_how_far_they_agree():
    rbd = pair(W([100.0, 2.0]), E([0.1]))
    with pytest.raises(_phase_type.NotSettled, match="agree to") as error:
        rbd.first_failure_sf(TIMES, rtol=1e-9)
    assert "first_failure_bounds" in str(error.value)
    assert "looser rtol" in str(error.value)


def test_a_long_run_start_is_refused_with_phases():
    rbd = pair(W([100.0, 2.0]), E([0.1]))
    with pytest.raises(NotImplementedError, match="stationary"):
        rbd.first_failure_sf(10.0, state="stationary")


def test_rtol_is_checked():
    rbd = pair(W([100.0, 2.0]), E([0.1]))
    for wrong in (0.0, 1.0, -1e-3, "1e-3", True):
        with pytest.raises(ValueError, match="rtol"):
            rbd.first_failure_sf(10.0, rtol=wrong)
        with pytest.raises(ValueError, match="rtol"):
            rbd.mean_time_to_first_failure(rtol=wrong)


def test_the_route_says_the_phases_are_numerical():
    route = pair(W([100.0, 2.0]), E([0.1])).analysis_routes()[
        "first_failure_sf"
    ]
    assert route.route == "numerical"
    assert "phases" in route.reason


# -- the phases themselves ----------------------------------------------------


def test_exponential_and_erlang_times_have_their_own_phases():
    assert _phase_type.phases(E([0.2]), 4) == _phase_type.Phases(
        (0.2,), (1.0,)
    )
    assert _phase_type.phases(E([0.2]), 4, age=3.0).exact
    erlang = _phase_type.phases(G([3, 0.5]), 4)
    assert erlang.exact
    assert erlang.rates == (0.5,) * 3 and erlang.ends == (0.0, 0.0, 1.0)
    assert not _phase_type.phases(G([2.5, 0.5]), 4).exact


@pytest.mark.parametrize(
    "model", [W([100.0, 2.0]), LN([math.log(50.0), 0.8]), G([2.5, 0.05])]
)
def test_the_phases_of_a_time_converge_to_it(model):
    # The phases' own distribution: the chain of one time, by hand.
    def survival(level, times):
        time = _phase_type.phases(model, level)
        count = len(time.rates)
        q = np.zeros((count, count))
        for k, (rate, ends) in enumerate(zip(time.rates, time.ends)):
            q[k, k] = -rate if k + 1 < count else -rate * ends
            if k + 1 < count:
                q[k, k + 1] = rate * (1.0 - ends)
        start = np.zeros(count)
        start[0] = 1.0
        return np.array([(start @ expm(q * t)).sum() for t in times])

    scale = _phase_type.scale(model)
    times = scale * np.array([0.5, 1.0, 2.0])
    exact = np.asarray(model.sf(times))
    errors = [
        np.max(np.abs(survival(level, times) - exact)) for level in (8, 16, 32)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.02
