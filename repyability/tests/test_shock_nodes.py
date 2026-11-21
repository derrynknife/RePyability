"""Shocks that fail each member with a probability (`shock_nodes`):
against the alpha-factor group by rate for exponential members, closed
forms for members of other lives at their own ages, and the draws."""

from math import comb

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    AlphaFactor,
    BetaFactor,
    CCFGroup,
    NonRepairableRBD,
    shock_nodes,
)

X = np.array([0.0, 100.0, 3000.0, 20000.0])


def parallel(names):
    return [("s", n) for n in names] + [(n, "t") for n in names]


def aged(model, age, x):
    return np.exp(-(model.Hf(age + x) - model.Hf(age)))


@pytest.mark.parametrize("p", [0.15, 0.4, 1.0])
@pytest.mark.parametrize("lethal", [0.0, 7e-6])
def test_exponential_members_are_an_alpha_factor_group_by_rate(p, lethal):
    """A member fails at rate ``own + rate p`` (plus the lethal shocks'),
    and the cause of each set of ``j`` members is the share ``rate p^j (1
    - p)^(m - j)`` of it: the alpha-factor model by rate, with ``alpha_j =
    C(m - 1, j - 1)`` times that share (staggered testing)."""
    own, rate, m = 1e-4, 3e-5, 3
    names = list("abc")
    total = own + rate * p + lethal
    share = {j: rate * p**j * (1 - p) ** (m - j) for j in range(1, m + 1)}
    share[1] += own
    share[m] += lethal
    alphas = [comb(m - 1, j - 1) * share[j] / total for j in range(1, m + 1)]
    assert sum(alphas) == pytest.approx(1.0, rel=1e-14)
    group = NonRepairableRBD(
        parallel(names),
        {n: surv.Exponential.from_params([total]) for n in names},
        k={"t": 2},
        ccf_groups=[CCFGroup(names, AlphaFactor(alphas, basis="rate"))],
    )
    edges, nodes = shock_nodes(
        parallel(names),
        {n: surv.Exponential.from_params([own]) for n in names},
        names,
        rate,
        p,
        lethal_rate=lethal,
    )
    written = NonRepairableRBD(edges, nodes, k={"t": 2})
    assert written.sf(X) == pytest.approx(group.sf(X), rel=1e-12, abs=0)
    assert written.ff(X) == pytest.approx(group.ff(X), rel=1e-12, abs=0)
    assert written.mean() == pytest.approx(group.mean(), rel=1e-9)


def test_lethal_shocks_alone_are_the_beta_factor_by_rate():
    own, lethal = 9e-5, 1e-5
    names = ["p1", "p2"]
    edges, nodes = shock_nodes(
        parallel(names),
        {n: surv.Exponential.from_params([own]) for n in names},
        names,
        rate=0.0,
        lethal_rate=lethal,
    )
    group = NonRepairableRBD(
        parallel(names),
        {n: surv.Exponential.from_params([own + lethal]) for n in names},
        ccf_groups=[
            CCFGroup(names, BetaFactor(lethal / (own + lethal), basis="rate"))
        ],
    )
    assert NonRepairableRBD(edges, nodes).sf(X) == pytest.approx(
        group.sf(X), rel=1e-12, abs=0
    )
    # One cause: an exponential node after each member, the second a
    # repeat of the first.
    assert len(nodes) == 4


def test_members_of_different_lives_and_ages_against_the_closed_form():
    """A parallel pair: the cause of both has not struck, and each member
    has survived its own life from its age and the cause of it alone."""
    one = surv.Weibull.from_params([8000.0, 2.5])
    two = surv.LogNormal.from_params([9.0, 0.7])
    rate, p = 2e-5, 0.6
    names = ["p1", "p2"]
    edges, nodes = shock_nodes(
        parallel(names), {"p1": one, "p2": two}, names, rate, p
    )
    rbd = NonRepairableRBD(edges, nodes)
    both = np.exp(-rate * p * p * X)
    alone = np.exp(-rate * p * (1 - p) * X)
    for a1, a2 in ((0.0, 0.0), (6000.0, 1000.0)):
        r1, r2 = aged(one, a1, X) * alone, aged(two, a2, X) * alone
        closed = both * (1 - (1 - r1) * (1 - r2))
        assert rbd.sf_given_state(X, {"p1": a1, "p2": a2}) == pytest.approx(
            closed, rel=1e-12, abs=0
        )
    assert rbd.sf(X) == pytest.approx(
        rbd.sf_given_state(X, {}), rel=1e-15, abs=0
    )


def test_two_of_three_against_every_combination_of_causes():
    """Sum over which of the seven causes have struck by ``x``."""
    life = surv.Weibull.from_params([8000.0, 2.5])
    rate, p, lethal = 3e-5, 0.3, 4e-6
    names, ages = list("abc"), {"a": 5000.0, "b": 2000.0, "c": 0.0}
    edges, nodes = shock_nodes(
        parallel(names),
        {n: life for n in names},
        names,
        rate,
        p,
        lethal_rate=lethal,
    )
    rbd = NonRepairableRBD(edges, nodes, k={"t": 2})
    causes = []
    for mask in range(1, 8):
        struck = {n for i, n in enumerate(names) if mask >> i & 1}
        j = len(struck)
        strikes = rate * p**j * (1 - p) ** (3 - j) + (lethal if j == 3 else 0)
        causes.append((struck, strikes))
    own = {n: aged(life, ages[n], X) for n in names}
    total = np.zeros_like(X)
    for fired in range(2 ** len(causes)):
        weight, down = np.ones_like(X), set()
        for i, (struck, strikes) in enumerate(causes):
            holds = np.exp(-strikes * X)
            if fired >> i & 1:
                weight, down = weight * (1 - holds), down | struck
            else:
                weight = weight * holds
        up = [own[n] for n in names if n not in down]
        if len(up) == 3:
            a, b, c = up
            works = a * b * c + a * b * (1 - c) + a * (1 - b) * c
            works = works + (1 - a) * b * c
        elif len(up) == 2:
            works = up[0] * up[1]
        else:
            works = 0.0
        total = total + weight * works
    assert rbd.sf_given_state(X, ages) == pytest.approx(
        total, rel=1e-11, abs=0
    )


def test_the_simulated_mean_covers_the_exact_one():
    life = surv.Weibull.from_params([8000.0, 2.5])
    names = list("abc")
    edges, nodes = shock_nodes(
        parallel(names), {n: life for n in names}, names, 3e-5, 0.3
    )
    rbd = NonRepairableRBD(edges, nodes, k={"t": 2})
    interval = rbd.mean_time_to_failure_interval(seed=0)
    assert interval.lower <= rbd.mean() <= interval.upper


def test_members_in_different_branches_and_the_diagram_saved():
    """Each cause follows each of its members, wherever they are."""
    pump = surv.Weibull.from_params([8000.0, 2.5])
    valve = surv.Weibull.from_params([20000.0, 1.5])
    edges = [
        ("s", "p1"),
        ("p1", "v1"),
        ("v1", "t"),
        ("s", "p2"),
        ("p2", "v2"),
        ("v2", "t"),
    ]
    models = {"p1": pump, "p2": pump, "v1": valve, "v2": valve}
    rate, p = 2e-5, 0.5
    written, nodes = shock_nodes(edges, models, ["p1", "p2"], rate, p)
    assert edges[0] == ("s", "p1") and set(models) == {"p1", "p2", "v1", "v2"}
    rbd = NonRepairableRBD(written, nodes)
    both = np.exp(-rate * p * p * X)
    alone = np.exp(-rate * p * (1 - p) * X)
    train = pump.sf(X) * valve.sf(X) * alone
    # 1 - (1 - train)^2, written without the difference: at the last
    # time a train survives with 1.5e-5, where that loses four digits.
    closed = both * train * (2 - train)
    assert rbd.sf(X) == pytest.approx(closed, rel=1e-12, abs=0)
    again = NonRepairableRBD.from_dict(rbd.to_dict())
    assert again.sf(X) == pytest.approx(rbd.sf(X), rel=1e-15, abs=0)
    # Two processes on one diagram, each with its own name.
    more, nodes = shock_nodes(written, nodes, ["v1", "v2"], 1e-5, name="surge")
    assert NonRepairableRBD(more, nodes).sf(X) == pytest.approx(
        closed * np.exp(-1e-5 * X), rel=1e-12, abs=0
    )


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"members": ["p1"]}, "2 to 10"),
        ({"members": ["p1", "p1"]}, "twice"),
        ({"members": ["p1", "zz"]}, "not one of the nodes"),
        ({"rate": -1.0}, "rate must be"),
        ({"rate": "1"}, "rate must be"),
        ({"lethal_rate": float("inf")}, "lethal_rate must be"),
        ({"p": 0.0}, "above 0 and at most 1"),
        ({"p": 1.5}, "above 0 and at most 1"),
        ({"p": True}, "above 0 and at most 1"),
    ],
)
def test_what_cannot_be_written_is_refused(arguments, message):
    life = surv.Weibull.from_params([8000.0, 2.5])
    given = {"members": ["p1", "p2"], "rate": 1e-5, **arguments}
    with pytest.raises(ValueError, match=message):
        shock_nodes(parallel(["p1", "p2"]), {"p1": life, "p2": life}, **given)


def test_a_name_in_use_and_a_repeat_as_a_member_are_refused():
    life = surv.Weibull.from_params([8000.0, 2.5])
    names = ["p1", "p2"]
    edges, nodes = shock_nodes(
        parallel(names), {n: life for n in names}, names, 1e-5
    )
    with pytest.raises(ValueError, match="name of its own"):
        shock_nodes(edges, nodes, names, 1e-5)
    with pytest.raises(ValueError, match="repeats node"):
        shock_nodes(parallel(names), {"p1": life, "p2": "p1"}, names, 1e-5)
