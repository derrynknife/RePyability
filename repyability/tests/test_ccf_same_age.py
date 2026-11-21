"""Common causes for wear-out lives: a group splitting the failure rate
at its members' one age, against the same model drawn as nodes (each
member's own causes with survival ``R^(1 - beta)``, the shared cause one
node in series with ``R^beta``), and a shock at a constant rate as a node
in series with members of different ages, against its closed form."""

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    MGL,
    BetaFactor,
    CCFGroup,
    NodeState,
    NonRepairableRBD,
)

PARALLEL = [("s", "p1"), ("s", "p2"), ("p1", "t"), ("p2", "t")]
WITH_SHARED = [
    ("s", "p1"),
    ("s", "p2"),
    ("p1", "shared"),
    ("p2", "shared"),
    ("shared", "t"),
]
X = np.array([0.0, 500.0, 2000.0, 6000.0, 15000.0])


class Share:
    """The life whose cumulative hazard is ``share`` times a model's:
    survival ``R ** share``, written from the cumulative hazard."""

    def __init__(self, model, share):
        self.model, self.share = model, share

    def Hf(self, x):
        return self.share * self.model.Hf(x)

    def sf(self, x):
        return np.exp(-self.Hf(x))

    def ff(self, x):
        return -np.expm1(-self.Hf(x))


def group_of(model, beta):
    return NonRepairableRBD(
        PARALLEL,
        {"p1": model, "p2": model},
        ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(beta, basis="rate"))],
    )


def drawn(own, shared):
    return NonRepairableRBD(
        WITH_SHARED, {"p1": own, "p2": own, "shared": shared}
    )


def weibull_split(alpha, k, beta):
    """The Weibull's two parts in closed form: the same shape, the scale
    times ``share ** (-1 / k)``."""
    return (
        surv.Weibull.from_params([alpha * (1 - beta) ** (-1 / k), k]),
        surv.Weibull.from_params([alpha * beta ** (-1 / k), k]),
    )


def test_exponential_members_have_no_age():
    """A constant hazard's share is a shock at a constant rate, so the
    group at any common age is the group from new, and both are the pair
    of rate ``(1 - beta) lam`` in series with one of rate ``beta lam``."""
    lam, beta = 1e-4, 0.1
    group = group_of(surv.Exponential.from_params([lam]), beta)
    new = group.sf(X)
    r = np.exp(-(1 - beta) * lam * X)
    closed = np.exp(-beta * lam * X) * (1 - (1 - r) ** 2)
    assert new == pytest.approx(closed, rel=1e-12, abs=0)
    for age in (0.0, 700.0, 40000.0):
        aged = group.sf_given_state(X, {"p1": age, "p2": age})
        assert aged == pytest.approx(new, rel=1e-12, abs=0)
    by_nodes = drawn(
        surv.Exponential.from_params([(1 - beta) * lam]),
        surv.Exponential.from_params([beta * lam]),
    )
    assert by_nodes.sf(X) == pytest.approx(new, rel=1e-12, abs=0)


@pytest.mark.parametrize("beta", [0.05, 0.3])
@pytest.mark.parametrize("age", [0.0, 1500.0, 9000.0])
def test_weibull_members_against_the_split_drawn_as_nodes(beta, age):
    alpha, k = 8000.0, 2.5
    group = group_of(surv.Weibull.from_params([alpha, k]), beta)
    by_nodes = drawn(*weibull_split(alpha, k, beta))
    state = {"p1": age, "p2": age}
    everything = {**state, "shared": age}
    assert group.sf(X) == pytest.approx(by_nodes.sf(X), rel=1e-12, abs=0)
    assert group.sf_given_state(X, state) == pytest.approx(
        by_nodes.sf_given_state(X, everything), rel=1e-12, abs=0
    )
    assert group.remaining_life(0.9, state) == pytest.approx(
        by_nodes.remaining_life(0.9, everything), rel=1e-9
    )
    assert group.mean_residual_life(state) == pytest.approx(
        by_nodes.mean_residual_life(everything), rel=1e-9
    )


@pytest.mark.parametrize("age", [0.0, 3000.0])
def test_lognormal_members_against_the_split_drawn_as_nodes(age):
    """A life with no closed form for ``R ** beta``."""
    life, beta = surv.LogNormal.from_params([8.5, 0.6]), 0.2
    group = group_of(life, beta)
    by_nodes = drawn(Share(life, 1 - beta), Share(life, beta))
    state = {"p1": age, "p2": age}
    everything = {**state, "shared": age}
    assert group.sf_given_state(X, state) == pytest.approx(
        by_nodes.sf_given_state(X, everything), rel=1e-12, abs=0
    )
    assert group.mean_residual_life(state) == pytest.approx(
        by_nodes.mean_residual_life(everything), rel=1e-9
    )


def test_the_common_age_against_its_closed_form():
    """``R_s(x | a) (1 - (1 - r(x | a))^2)``, each part's survival from
    age ``a`` its share of the members' cumulative hazard from then."""
    life, beta, age = surv.LogNormal.from_params([8.5, 0.6]), 0.2, 3000.0
    H = life.Hf(age + X) - life.Hf(age)
    r, shared = np.exp(-(1 - beta) * H), np.exp(-beta * H)
    closed = shared * (1 - (1 - r) ** 2)
    group = group_of(life, beta)
    state = {"p1": NodeState(age=age), "p2": NodeState(age=age)}
    assert group.sf_given_state(X, state) == pytest.approx(
        closed, rel=1e-12, abs=0
    )
    # A member works while neither its own causes nor the shared one has
    # struck; with it failed, the other works with probability
    # P(other works, it failed) / P(it failed).
    importance = group.importances_given_state(X[1:], state)
    r, shared = r[1:], shared[1:]
    given_failed = shared * r * (1 - r) / (1 - shared * r)
    assert importance["birnbaum"]["p1"] == pytest.approx(
        1 - given_failed, rel=1e-10
    )
    assert importance["birnbaum"]["p2"] == pytest.approx(
        importance["birnbaum"]["p1"], rel=1e-12
    )


def test_an_mgl_group_by_rate_at_a_common_age():
    """Each specific set's cause takes its share of the members' hazard
    from their age on: the group from new at the conditional
    probabilities."""
    life = surv.Weibull.from_params([8000.0, 2.5])
    names = ["a", "b", "c"]
    edges = [("s", n) for n in names] + [(n, "t") for n in names]
    model = MGL(0.1, 0.3, basis="rate")
    rbd = NonRepairableRBD(
        edges,
        {n: life for n in names},
        k={"t": 2},
        ccf_groups=[CCFGroup(names, model)],
    )
    age = 4000.0
    H = life.Hf(age + X) - life.Hf(age)
    own, causes = model._causes(names)
    # Sum over which causes have struck: the system works with two of
    # the three members up.
    total = np.zeros_like(H)
    for mask in range(2 ** len(causes)):
        weight = np.ones_like(H)
        struck = set()
        for j, (members, c) in enumerate(causes):
            fires = -np.expm1(-c * H)
            if mask >> j & 1:
                weight = weight * fires
                struck |= members
            else:
                weight = weight * (1 - fires)
        left = len(names) - len(struck)
        r = np.exp(-own * H)
        if left == 3:
            works = r**3 + 3 * r**2 * (1 - r)
        elif left == 2:
            works = r**2
        else:
            works = 0.0
        total = total + weight * works
    assert rbd.sf_given_state(X, {n: age for n in names}) == pytest.approx(
        total, rel=1e-12, abs=0
    )


def test_no_state_is_from_new():
    group = group_of(surv.Weibull.from_params([8000.0, 2.5]), 0.1)
    assert group.sf_given_state(X, {}) == pytest.approx(
        group.sf(X), rel=1e-15, abs=0
    )


def test_the_simulated_mean_covers_the_exact_one():
    """The simulations draw the shared cause through the members'
    quantile function."""
    group = group_of(surv.LogNormal.from_params([8.5, 0.6]), 0.2)
    interval = group.mean_time_to_failure_interval(seed=0)
    assert interval.lower <= group.mean() <= interval.upper


@pytest.mark.parametrize(
    "state",
    [
        {"p1": 4000.0},
        {"p1": 4000.0, "p2": 1000.0},
        {"p1": NodeState(alive=False), "p2": NodeState(alive=False)},
        {
            "p1": NodeState(age=10.0),
            "p2": NodeState(
                age=10.0, model=surv.Weibull.from_params([9000.0, 2.5])
            ),
        },
    ],
)
def test_members_in_different_states_are_refused(state):
    group = group_of(surv.Weibull.from_params([8000.0, 2.5]), 0.1)
    for call in (
        lambda: group.sf_given_state(100.0, state),
        lambda: group.remaining_life(0.9, state),
        lambda: group.mean_residual_life(state),
        lambda: group.importances_given_state(100.0, state),
    ):
        with pytest.raises(NotImplementedError, match="node in series"):
            call()


def test_a_probability_split_is_refused_naming_the_rate_split():
    life = surv.Weibull.from_params([8000.0, 2.5])
    rbd = NonRepairableRBD(
        PARALLEL,
        {"p1": life, "p2": life},
        ccf_groups=[CCFGroup(["p1", "p2"], BetaFactor(0.1))],
    )
    with pytest.raises(NotImplementedError, match="basis='rate'"):
        rbd.sf_given_state(100.0, {"p1": 10.0, "p2": 10.0})
    assert rbd.analysis_routes()["sf_given_state"].route == "refused"
    by_rate = group_of(life, 0.1).analysis_routes()
    assert by_rate["sf_given_state"].route == "exact"
    assert by_rate["importances_given_state"].route == "exact"
    assert by_rate["remaining_life"].route == "numerical"
    assert by_rate["mean_residual_life"].route == "numerical"


# A shock at a constant rate that fails every member: a node in series.
PUMP = surv.Weibull.from_params([8000.0, 2.5])
VALVE = surv.Weibull.from_params([20000.0, 1.5])
RATE = 2e-5


def aged(model, age, x):
    return np.exp(-(model.Hf(age + x) - model.Hf(age)))


def test_a_lethal_shock_in_series_with_members_of_different_ages():
    """``R_sys = exp(-rate x) R_independent(x | ages)``."""
    pair = NonRepairableRBD(
        WITH_SHARED,
        {
            "p1": PUMP,
            "p2": PUMP,
            "shared": surv.Exponential.from_params([RATE]),
        },
    )
    old, young = aged(PUMP, 6000.0, X), aged(PUMP, 1000.0, X)
    closed = np.exp(-RATE * X) * (1 - (1 - old) * (1 - young))
    state = {"p1": NodeState(age=6000.0), "p2": NodeState(age=1000.0)}
    assert pair.sf_given_state(X, state) == pytest.approx(
        closed, rel=1e-12, abs=0
    )
    # The shock has no age.
    assert pair.sf_given_state(X, {**state, "shared": 5e4}) == pytest.approx(
        closed, rel=1e-12, abs=0
    )


def test_a_lethal_shock_as_a_repeated_node_across_branches():
    """Two trains, a pump and a valve each; the shock fails both pumps:
    by the law of total probability on the shock, against the trains'
    own closed form."""
    trains = NonRepairableRBD(
        [
            ("s", "c1"),
            ("c1", "p1"),
            ("p1", "v1"),
            ("v1", "t"),
            ("s", "c2"),
            ("c2", "p2"),
            ("p2", "v2"),
            ("v2", "t"),
        ],
        {
            "p1": PUMP,
            "p2": PUMP,
            "v1": VALVE,
            "v2": VALVE,
            "c1": surv.Exponential.from_params([RATE]),
            "c2": "c1",
        },
    )
    one = aged(PUMP, 6000.0, X) * aged(VALVE, 3000.0, X)
    two = aged(PUMP, 1000.0, X) * aged(VALVE, 500.0, X)
    closed = np.exp(-RATE * X) * (1 - (1 - one) * (1 - two))
    state = {"p1": 6000.0, "p2": 1000.0, "v1": 3000.0, "v2": 500.0}
    assert trains.sf_given_state(X, state) == pytest.approx(
        closed, rel=1e-12, abs=0
    )


def test_a_lethal_shock_against_a_direct_integral():
    """The mean life from new: the integral of the closed-form
    reliability, by quadrature of its own."""
    from scipy.integrate import quad

    pair = NonRepairableRBD(
        WITH_SHARED,
        {
            "p1": PUMP,
            "p2": PUMP,
            "shared": surv.Exponential.from_params([RATE]),
        },
    )

    def reliability(t):
        r = float(PUMP.sf(t))
        return np.exp(-RATE * t) * (1 - (1 - r) ** 2)

    direct = quad(reliability, 0, 4e4, epsabs=0, epsrel=1e-12, limit=200)[0]
    assert pair.mean() == pytest.approx(direct, rel=1e-9)
