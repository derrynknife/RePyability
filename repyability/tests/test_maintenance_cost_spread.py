"""A part's cost over a window from its state, with its spread (UC-8 of
``repyability/maintenance/README.md``): ``NonRepairable.cost_distribution``.
"""

import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy.stats import poisson

from repyability import CostDistribution, NodeState, NonRepairable
from repyability.maintenance._options import _unit


def exponential(rate=0.01, planned=1.0, unplanned=5.0):
    return NonRepairable(
        surv.Exponential.from_params([rate]),
        planned_cost=planned,
        unplanned_cost=unplanned,
    )


@pytest.fixture
def bearing():
    # UC-7's bearing: Weibull, scale 365 days, shape 2; $100 at a shut,
    # $400 at a failure.
    return NonRepairable(
        surv.Weibull.from_params([365.0, 2.0]),
        planned_cost=100.0,
        unplanned_cost=400.0,
    )


def pois(k, mean):
    return poisson.pmf(np.asarray(k), mean)


# -- closed forms -------------------------------------------------------------


@pytest.mark.parametrize("age", [0.0, 50.0, 400.0])
def test_an_exponential_life_run_to_failure_has_poisson_failures(age):
    # Memoryless, replaced at once: N_f ~ Poisson(lam W) whatever the age.
    # pmf exp(-m) m^k / k!, mean m cu, variance m cu^2, m = lam W = 3.
    part = exponential()
    out = part.cost_distribution(300.0, age=age, budget=20.0)
    k = np.arange(len(out.failure_probabilities))
    error = np.max(np.abs(out.failure_probabilities - pois(k, 3.0)))
    assert error <= out.precision
    assert out.precision < 1e-5
    assert out.probabilities.shape[1] == 1  # nothing planned
    assert out.mean == pytest.approx(15.0, abs=1e-5)
    assert out.standard_deviation == pytest.approx(5 * math.sqrt(3), 1e-5)
    # P90 of the cost: 5 times the 90% point of Poisson(3), 5.
    assert poisson.ppf(0.9, 3.0) == 5
    assert out.percentile(90) == 25.0
    assert list(out.percentiles) == [
        5.0 * poisson.ppf(q, 3.0) for q in (0.1, 0.5, 0.9)
    ]
    # More than 20 is 5 or more failures: 1 - P(N <= 4).
    assert out.probability_above_budget == pytest.approx(
        poisson.sf(4, 3.0), abs=1e-6
    )
    assert out.method == "numerical" and out.policy == "run to failure"
    assert out.every is None and out.planned_cost is None


def test_exponential_age_replacement_with_no_chance_to_plan():
    # From new, a window shorter than T: a planned replacement needs a part
    # to reach T, so none falls in it; failures are Poisson(lam W).
    part = exponential()
    out = part.cost_distribution(80.0, 100.0, resets=True)
    assert out.planned_probabilities[0] == pytest.approx(1.0, abs=1e-9)
    assert out.planned_probabilities[1:].sum() < 1e-9
    k = np.arange(len(out.failure_probabilities))
    assert out.failure_probabilities == pytest.approx(pois(k, 0.8), abs=1e-6)


def test_exponential_age_replacement_with_one_chance_to_plan():
    # Age 70, T = 100, W = 90: the part in now is replaced at tau = 30 if it
    # has not failed; any later part reaches 100 only after W. By hand, with
    # m = lam W = 0.9 and the stretch after tau, n = lam (W - tau) = 0.6:
    #   P(k failures, 1 planned) = exp(-lam tau) pois(k, n)
    #   P(k failures, 0 planned) = pois(k, m) - exp(-lam tau) pois(k, n)
    part = exponential()
    out = part.cost_distribution(90.0, 100.0, resets=True, age=70.0)
    P = out.probabilities
    k = np.arange(P.shape[0])
    kept = math.exp(-0.3)
    assert P[:, 1] == pytest.approx(kept * pois(k, 0.6), abs=2e-6)
    assert P[:, 0] == pytest.approx(
        pois(k, 0.9) - kept * pois(k, 0.6), abs=2e-6
    )
    assert np.all(P[:, 2:] < 1e-9)
    assert out.planned_probabilities[1] == pytest.approx(kept, abs=1e-9)
    assert (
        np.max(
            np.abs(
                P[:, :2]
                - np.stack(
                    [pois(k, 0.9) - kept * pois(k, 0.6), kept * pois(k, 0.6)],
                    axis=1,
                )
            )
        )
        <= out.precision
    )


def test_a_life_close_to_fixed_concentrates_the_count():
    # Weibull of shape 60 around 100: lives within a few of 100, so over
    # 250 two failures, nearly surely (two lives < 250 < three lives).
    part = NonRepairable(
        surv.Weibull.from_params([100.0, 60.0]), unplanned_cost=10.0
    )
    out = part.cost_distribution(250.0)
    assert out.failure_probabilities[2] > 0.999
    assert out.mean == pytest.approx(20.0, abs=0.02)
    assert list(out.percentiles) == [20.0, 20.0, 20.0]
    # Replaced every 50 on fixed times: no failure, 5 planned.
    planned = NonRepairable(
        surv.Weibull.from_params([100.0, 60.0]),
        planned_cost=1.0,
        unplanned_cost=10.0,
    )
    blocks = planned.cost_distribution(250.0, 50.0, resets=False)
    assert blocks.planned_probabilities[4] == pytest.approx(1.0)
    assert blocks.failure_probabilities[0] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("age", [0.0, 224.0, 500.0])
def test_from_an_aged_start_the_first_failure_is_the_conditional_life(
    bearing, age
):
    # 1 - P(no failure) = 1 - R(age + W) / R(age), the conditional ff;
    # with no planned replacement in the window, P(0, 0) = R(age + W) / R(age).
    life = bearing.reliability
    W = 40.0
    survive = float(life.sf(age + W) / life.sf(age))
    run = bearing.cost_distribution(W, age=age)
    assert 1.0 - run.failure_probabilities[0] == pytest.approx(
        float(life.ff(age + W) - life.ff(age)) / float(life.sf(age)),
        abs=1e-12,
    )
    assert run.probabilities[0, 0] == pytest.approx(survive, abs=1e-12)
    aged = bearing.cost_distribution(W, 600.0, resets=True, age=age)
    assert aged.probabilities[0, 0] == pytest.approx(survive, abs=1e-12)
    block = bearing.cost_distribution(
        W, 180.0, resets=False, age=age, since_renewal=100.0
    )
    assert block.probabilities[0, 0] == pytest.approx(survive, abs=1e-12)


def test_with_a_planned_replacement_due_in_the_window(bearing):
    # Age replacement at 300 from age 280, over 50: no failure in the 20
    # days to 300 (S(20)) and none in the 30 after (R(30)) is P(0, 1).
    life = bearing.reliability
    out = bearing.cost_distribution(50.0, 300.0, resets=True, age=280.0)
    S = float(life.sf(300.0) / life.sf(280.0))
    assert out.probabilities[0, 0] == 0.0
    assert out.probabilities[0, 1] == pytest.approx(
        S * float(life.sf(30.0)), abs=1e-12
    )
    # Past the age, the part is replaced at once.
    late = bearing.cost_distribution(50.0, 300.0, resets=True, age=320.0)
    assert late.planned_probabilities[0] == 0.0
    assert late.probabilities[0, 1] == pytest.approx(float(life.sf(50.0)))


# -- against the module's expected costs --------------------------------------


@pytest.mark.parametrize("age", [0.0, 224.0, 420.0])
@pytest.mark.parametrize("window", [82.0, 1086.0])
def test_the_mean_is_the_expected_cost_of_each_policy(bearing, age, window):
    run = bearing.cost_distribution(window, age=age)
    assert run.mean == pytest.approx(
        bearing.failures(window, age=age).expected_cost, rel=2e-6
    )
    aged = bearing.cost_distribution(window, 300.0, resets=True, age=age)
    unit = _unit(bearing, {"interval": 300.0, "cost": 100.0})
    state = {"unit": NodeState(age=age)}
    assert aged.mean == pytest.approx(
        unit.expected_cost(window, state=state).mean, rel=2e-6
    )
    events = unit.expected_events(window, state=state)
    assert aged.expected_failures == pytest.approx(
        float(events.node_failures["unit"]), abs=2e-6
    )
    assert aged.expected_planned == pytest.approx(
        float(events.node_preventive["unit"]), abs=2e-6
    )
    since = min(age, 159.0)
    block = bearing.cost_distribution(
        window, 180.0, resets=False, age=age, since_renewal=since
    )
    unit = _unit(
        bearing, {"interval": 180.0, "cost": 100.0, "policy": "block"}
    )
    state = {"unit": NodeState(age=age, since_scheduled=since)}
    assert block.mean == pytest.approx(
        unit.expected_cost(window, state=state).mean, rel=2e-6
    )


def test_the_uc8_figures(bearing):
    # README UC-8, over 1,086 days from age 224, 21 days to the next shut:
    # every shut, mean 1,155, 10% to 90% 600 to 1,800; run to failure,
    # 1,373, 800 to 2,000 (simulated by hand then).
    shuts = bearing.cost_distribution(
        1086.0, 180.0, resets=False, age=224.0, since_renewal=159.0
    )
    assert round(shuts.mean) == 1155
    assert list(shuts.percentiles) == [600.0, 1000.0, 1800.0]
    assert shuts.planned_probabilities[6] == pytest.approx(
        1.0
    )  # shuts at 21, 201, ... 921
    run = bearing.cost_distribution(1086.0, age=224.0)
    assert round(run.mean) == 1373
    assert list(run.percentiles) == [800.0, 1200.0, 2000.0]


# -- simulation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        {},
        {"every": 300.0, "resets": True},
        {"every": 180.0, "resets": False, "since_renewal": 159.0},
    ],
)
def test_the_simulation_agrees_with_the_numerical_route(bearing, policy):
    n = 40_000
    exact = bearing.cost_distribution(700.0, age=224.0, **policy)
    simulated = bearing.cost_distribution(
        700.0, age=224.0, method="simulate", mc_samples=n, seed=7, **policy
    )
    assert simulated.method == "simulated" and simulated.n_simulations == n
    assert abs(simulated.mean - exact.mean) < 4 * simulated.standard_error
    rows = max(exact.probabilities.shape[0], simulated.probabilities.shape[0])
    cols = max(exact.probabilities.shape[1], simulated.probabilities.shape[1])
    a, b = np.zeros((rows, cols)), np.zeros((rows, cols))
    a[: exact.probabilities.shape[0], : exact.probabilities.shape[1]] = (
        exact.probabilities
    )
    b[
        : simulated.probabilities.shape[0], : simulated.probabilities.shape[1]
    ] = simulated.probabilities
    bound = 4 * np.sqrt(a * (1 - a) / n) + 1e-4
    assert np.all(np.abs(a - b) <= bound)
    again = bearing.cost_distribution(
        700.0, age=224.0, method="simulate", mc_samples=n, seed=7, **policy
    )
    assert np.array_equal(again.probabilities, simulated.probabilities)


# -- the answer's shape -------------------------------------------------------


def test_the_cost_distribution_adds_up(bearing):
    out = bearing.cost_distribution(800.0, 300.0, resets=True, age=100.0)
    assert out.probabilities.sum() == pytest.approx(1.0, abs=out.precision)
    assert out.cost_probabilities.sum() == pytest.approx(
        out.probabilities.sum()
    )
    assert np.all(np.diff(out.costs) > 0)
    cumulative = np.cumsum(out.cost_probabilities)
    for q, value in zip(out.levels, out.percentiles):
        at = list(out.costs).index(value)
        assert cumulative[at] >= q / 100 - 1e-12
        assert at == 0 or cumulative[at - 1] < q / 100
    assert out.probability_above(out.costs[-1]) == 0.0
    assert out.probability_above(-1.0) == pytest.approx(1.0, abs=1e-9)
    outcomes = out.outcomes()
    assert outcomes[0]["probability"] == max(
        o["probability"] for o in outcomes
    )
    first = outcomes[0]
    assert first["cost"] == 400 * first["failures"] + 100 * first["planned"]


@pytest.mark.parametrize("unit", [1e-6, 24.0, 1e6])
def test_the_answer_does_not_depend_on_the_unit_of_time(unit):
    def part(scale):
        return NonRepairable(
            surv.Weibull.from_params([365.0 * scale, 2.0]),
            planned_cost=100.0,
            unplanned_cost=400.0,
        )

    for policy in (
        {},
        {"every": 300.0, "resets": True},
        {"every": 180.0, "resets": False, "since_renewal": 159.0},
    ):
        scaled = {
            k: v * unit if isinstance(v, float) else v
            for k, v in policy.items()
        }
        days = part(1.0).cost_distribution(700.0, age=224.0, **policy)
        other = part(unit).cost_distribution(
            700.0 * unit, age=224.0 * unit, **scaled
        )
        assert other.probabilities == pytest.approx(
            days.probabilities, abs=1e-9
        )


def test_the_result_goes_through_json(bearing):
    out = bearing.cost_distribution(
        400.0, 180.0, resets=False, age=50.0, budget=500.0
    )
    assert isinstance(out, CostDistribution)
    data = json.loads(json.dumps(out.to_dict()))
    assert np.array(data["probabilities"]) == pytest.approx(out.probabilities)
    assert data["percentiles"] == list(out.percentiles)
    assert data["probability_above_budget"] == out.probability_above(500.0)
    assert data["policy"] == "block replacement"


# -- inputs -------------------------------------------------------------------


@pytest.mark.parametrize(
    "args, kwargs, error, words",
    [
        ((0.0,), {}, ValueError, "window must be a number, more than 0"),
        ((100.0,), {"age": -1.0}, ValueError, "age must be"),
        ((100.0, 50.0), {}, TypeError, "resets must be True"),
        ((100.0,), {"resets": True}, ValueError, "Run to failure"),
        (
            (100.0, 50.0),
            {"resets": True, "since_renewal": 1.0},
            ValueError,
            "give no since_renewal",
        ),
        (
            (100.0, 50.0),
            {"resets": False, "since_renewal": 50.0},
            ValueError,
            "due now or past",
        ),
        ((100.0,), {"method": "fast"}, ValueError, "method must be"),
        ((100.0,), {"seed": 1}, ValueError, "for method='simulate'"),
        (
            (100.0,),
            {"method": "simulate", "mc_samples": 1},
            ValueError,
            "mc_samples",
        ),
        ((100.0,), {"levels": [101]}, ValueError, "from 0 to 100"),
        ((100.0,), {"budget": "a lot"}, ValueError, "budget must be"),
    ],
)
def test_inputs_are_refused_in_plain_words(
    bearing, args, kwargs, error, words
):
    with pytest.raises(error, match=words):
        bearing.cost_distribution(*args, **kwargs)


def test_parts_the_route_cannot_take_are_refused():
    life = surv.Weibull.from_params([100.0, 2.0])
    with pytest.raises(ValueError, match="unplanned_cost"):
        NonRepairable(life).cost_distribution(10.0)
    with pytest.raises(ValueError, match="planned_cost"):
        NonRepairable(life, unplanned_cost=5.0).cost_distribution(
            10.0, 20.0, resets=True
        )
    timed = NonRepairable(
        life, surv.Exponential.from_params([1.0]), unplanned_cost=5.0
    )
    with pytest.raises(ValueError, match="in no time"):
        timed.cost_distribution(10.0)
    estimate = surv.KaplanMeier.fit([10.0, 20.0, 30.0, 40.0])
    with pytest.raises(NotImplementedError, match="closed form"):
        NonRepairable(estimate, unplanned_cost=5.0).cost_distribution(10.0)
