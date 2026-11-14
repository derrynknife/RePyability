"""NOT gates and dynamic gates (PAND, SPARE, SEQ, FDEP) in fault trees.

The values are checked against closed forms (two-input PAND and cold and
warm spares of exponential lives, an Erlang sequence), against integrals
worked out independently by scipy's quadrature, against the trees'
static equivalents (FDEP), against every combination of the events'
states (NOT gates), and against the trees' own simulation."""

import itertools
import xml.etree.ElementTree as ElementTree

import numpy as np
import pytest
import scipy.integrate as integrate
import surpyval as surv

from repyability import BetaFactor, CCFGroup, FaultTree, PrimeImplicant

E, W = surv.Exponential.from_params, surv.Weibull.from_params
T = np.array([5.0, 40.0, 120.0, 400.0])


def by_every_state(tree, t=None):
    """The top event probability, summed over every combination of the
    events' states (``occurs``)."""
    names = list(tree.events)
    q = {
        e: (m if isinstance(m, float) else float(m.ff(t)))
        for e, m in tree.events.items()
    }
    total = 0.0
    for states in itertools.product([False, True], repeat=len(names)):
        occurred = {e for e, s in zip(names, states) if s}
        if tree.occurs(occurred):
            weight = 1.0
            for e, s in zip(names, states):
                weight *= q[e] if s else 1.0 - q[e]
            total += weight
    return total


def implicants(tree):
    return {
        (frozenset(i.occurred), frozenset(i.not_occurred))
        for i in tree.prime_implicants()
    }


# -- NOT gates ----------------------------------------------------------------


def consensus_tree(**events):
    """``a b + a' c``: its prime implicants are ``a b``, ``a' c`` and their
    consensus ``b c``, which the gates do not spell out."""
    return FaultTree(
        {
            "top": ("or", ["ab", "not a, c"]),
            "ab": ("and", ["a", "b"]),
            "not a, c": ("and", ["not a", "c"]),
            "not a": ("not", ["a"]),
        },
        events or {"a": 0.1, "b": 0.2, "c": 0.3},
    )


def test_a_not_gate_s_prime_implicants_include_the_consensus():
    tree = consensus_tree()
    assert not tree.is_coherent
    assert implicants(tree) == {
        (frozenset("ab"), frozenset()),
        (frozenset("bc"), frozenset()),
        (frozenset("c"), frozenset("a")),
    }
    assert all(isinstance(i, PrimeImplicant) for i in tree.prime_implicants())


def test_the_exclusive_or_has_two_prime_implicants():
    tree = FaultTree(
        {
            "top": ("or", ["a only", "b only"]),
            "a only": ("and", ["a", "not b"]),
            "b only": ("and", ["b", "not a"]),
            "not a": ("not", ["a"]),
            "not b": ("not", ["b"]),
        },
        {"a": 0.1, "b": 0.3},
    )
    assert implicants(tree) == {
        (frozenset("a"), frozenset("b")),
        (frozenset("b"), frozenset("a")),
    }
    assert tree.top_event_probability() == pytest.approx(
        0.1 * 0.7 + 0.3 * 0.9, rel=1e-15
    )


def test_a_not_gate_s_probability_is_exact():
    tree = consensus_tree()
    assert tree.top_event_probability() == pytest.approx(
        0.1 * 0.2 + 0.9 * 0.3, rel=1e-15
    )
    assert tree.sf() == pytest.approx(1 - (0.1 * 0.2 + 0.9 * 0.3), rel=1e-15)
    assert tree.top_event_probability() == pytest.approx(
        by_every_state(tree), rel=1e-14
    )


def test_not_and_vote_gates_over_time_agree_with_every_state():
    tree = FaultTree(
        {
            "top": ("vote", 2, ["a", "not b", "g"]),
            "not b": ("not", ["b"]),
            "g": ("or", ["c", "not d"]),
            "not d": ("not", ["d"]),
        },
        {"a": W([100, 2]), "b": E([0.01]), "c": 0.05, "d": W([80, 1.5])},
    )
    for t in (1.0, 50.0, 300.0):
        assert tree.top_event_probability(t) == pytest.approx(
            by_every_state(tree, t), rel=1e-13
        )
    values = tree.ff(T)
    assert values.shape == T.shape
    assert np.allclose(values + tree.sf(T), 1.0, rtol=0, atol=1e-15)


def test_a_small_probability_keeps_its_precision():
    tree = FaultTree(
        {"top": ("and", ["a", "not b"]), "not b": ("not", ["b"])},
        {"a": 1e-12, "b": 1e-9},
    )
    assert tree.ff() == pytest.approx(1e-12 * (1 - 1e-9), rel=1e-15)
    tree = FaultTree(
        {"top": ("or", ["a", "not b"]), "not b": ("not", ["b"])},
        {"a": 1e-12, "b": 1 - 1e-9},
    )
    assert tree.sf() == pytest.approx((1 - 1e-12) * (1 - 1e-9), rel=1e-6)


def test_the_birnbaum_importance_of_an_event_under_a_not_gate_is_negative():
    tree = consensus_tree()
    birnbaum = tree.birnbaum_importance()
    assert birnbaum["a"] == pytest.approx(0.2 - 0.3)
    assert birnbaum["b"] == pytest.approx(0.1)
    given = tree.conditional_probabilities()
    assert given.given_occurred["a"] == pytest.approx(0.2)
    assert given.given_not_occurred["a"] == pytest.approx(0.3)
    assert tree.occurs({"c"}) and not tree.occurs({"a", "c"})


def test_a_coherent_tree_s_prime_implicants_are_its_minimal_cut_sets():
    tree = FaultTree(
        {
            "top": ("and", ["A", "B"]),
            "A": ("or", ["s", "a"]),
            "B": ("vote", 2, ["s", "b", "c"]),
        },
        {"s": 0.01, "a": 0.1, "b": 0.1, "c": 0.2},
    )
    found = implicants(tree)
    assert all(not off for _, off in found)
    assert {on for on, _ in found} == set(tree.minimal_cut_sets())


@pytest.mark.parametrize(
    "method, words",
    [
        ("minimal_cut_sets", "prime_implicants()"),
        ("ranked_cut_sets", "prime_implicants()"),
        ("criticality_importance", "coherent"),
        ("risk_achievement_worth", "coherent"),
        ("mean", "no"),
        ("to_rbd", "cannot hold a NOT gate"),
    ],
)
def test_what_does_not_apply_to_a_non_coherent_tree_is_refused(method, words):
    tree = consensus_tree()
    with pytest.raises(NotImplementedError, match="NOT gate") as error:
        getattr(tree, method)()
    assert words in str(error.value)


def test_a_not_gate_takes_one_input():
    with pytest.raises(ValueError, match="takes one input"):
        FaultTree({"top": ("not", ["a", "b"])}, {"a": 0.1, "b": 0.1})


def test_a_not_gate_with_common_causes_is_refused():
    with pytest.raises(NotImplementedError, match="NOT gates"):
        FaultTree(
            {"top": ("and", ["a", "not b"]), "not b": ("not", ["b"])},
            {"a": 0.1, "b": 0.1},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        )


# -- PAND ---------------------------------------------------------------------


def test_a_two_input_pand_of_exponentials_is_its_closed_form():
    la, lb = 0.01, 0.025
    tree = FaultTree(
        {"top": ("pand", ["a", "b"])}, {"a": E([la]), "b": E([lb])}
    )
    # P(A <= B <= t)
    exact = (1 - np.exp(-lb * T)) - lb / (la + lb) * (
        1 - np.exp(-(la + lb) * T)
    )
    assert np.allclose(tree.ff(T), exact, rtol=0, atol=1e-10)
    assert np.allclose(tree.sf(T), 1 - exact, rtol=0, atol=1e-10)
    # It occurs at all only in the right order: la / (la + lb).
    assert tree.ff(1e5) == pytest.approx(la / (la + lb), abs=1e-9)


def test_a_three_input_pand_occurs_in_order_with_its_chance():
    rates = [0.02, 0.01, 0.03]
    tree = FaultTree(
        {"top": ("pand", ["a", "b", "c"])},
        {name: E([r]) for name, r in zip("abc", rates)},
    )
    in_order = rates[0] / sum(rates) * rates[1] / (rates[1] + rates[2])
    assert tree.ff(1e4) == pytest.approx(in_order, abs=1e-9)


def test_a_pand_of_a_gate_and_a_weibull_is_its_integral():
    a, b, c = W([100, 2]), W([60, 0.8]), E([0.004])
    tree = FaultTree(
        {"top": ("pand", ["first", "b"]), "first": ("or", ["a", "c"])},
        {"a": a, "b": b, "c": c},
    )

    def first(s):
        return 1 - a.sf(s) * c.sf(s)

    for t in T:
        expected, _ = integrate.quad(
            lambda s: first(s) * b.df(s), 0, t, epsabs=1e-13, limit=200
        )
        assert tree.ff(t) == pytest.approx(expected, abs=1e-8)


def test_a_pand_of_probabilities_occurs_at_the_start_in_order():
    tree = FaultTree({"top": ("pand", ["a", "b"])}, {"a": 0.1, "b": 0.2})
    assert tree.is_fixed
    assert tree.ff() == pytest.approx(0.02)


# -- SPARE and SEQ -----------------------------------------------------------


def test_a_cold_spare_is_the_convolution_of_exponentials():
    l1, l2 = 0.01, 0.03
    tree = FaultTree(
        {"top": ("spare", ["a", "b"])}, {"a": E([l1]), "b": E([l2])}
    )
    exact = 1 - (l2 * np.exp(-l1 * T) - l1 * np.exp(-l2 * T)) / (l2 - l1)
    assert np.allclose(tree.ff(T), exact, rtol=0, atol=1e-8)


def test_a_cold_spare_of_weibulls_is_their_convolution():
    a, b = W([100, 2]), W([80, 1.5])
    tree = FaultTree({"top": ("spare", ["a", "b"])}, {"a": a, "b": b})
    for t in T:
        expected, _ = integrate.quad(
            lambda s: b.ff(t - s) * a.df(s), 0, t, epsabs=1e-13, limit=200
        )
        assert tree.ff(t) == pytest.approx(expected, abs=1e-8)


def test_a_hot_spare_is_the_and_of_its_units():
    a, b = W([100, 2]), W([80, 0.8])
    hot = FaultTree({"top": ("spare", 1.0, ["a", "b"])}, {"a": a, "b": b})
    assert np.allclose(hot.ff(T), a.ff(T) * b.ff(T), rtol=0, atol=1e-8)


def test_a_warm_spare_of_exponentials_is_its_closed_form():
    l1, l2, dormancy = 0.01, 0.02, 0.3
    tree = FaultTree(
        {"top": ("spare", dormancy, ["a", "b"])},
        {"a": E([l1]), "b": E([l2])},
    )
    # The primary fails at s; the spare has survived its reserve, at
    # dormancy * l2, and then its work, at l2.
    k = l1 + dormancy * l2 - l2
    exact = (
        1 - np.exp(-l1 * T) - l1 * np.exp(-l2 * T) * (1 - np.exp(-k * T)) / k
    )
    assert np.allclose(tree.ff(T), exact, rtol=0, atol=1e-8)


def test_two_cold_spares_and_a_sequence_are_an_erlang():
    rate = 0.02
    erlang = 1 - np.exp(-rate * T) * (1 + rate * T + (rate * T) ** 2 / 2)
    events = {x: E([rate]) for x in "abc"}
    spare = FaultTree({"top": ("spare", ["a", "b", "c"])}, events)
    sequence = FaultTree({"top": ("seq", ["a", "b", "c"])}, events)
    assert np.allclose(spare.ff(T), erlang, rtol=0, atol=1e-8)
    assert np.allclose(sequence.ff(T), erlang, rtol=0, atol=1e-8)


def warm_pair():
    return FaultTree(
        {
            "top": ("or", ["spares", "v"]),
            "spares": ("spare", 0.4, ["primary", "s1", "s2"]),
            "primary": ("and", ["a", "b"]),
        },
        {
            "a": W([100, 2]),
            "b": W([90, 1.5]),
            "s1": E([0.01]),
            "s2": W([120, 2.5]),
            "v": 0.01,
        },
    )


def test_the_exact_values_agree_with_the_simulation():
    tree = warm_pair()
    n = 200_000
    times = tree.random(n, seed=3)
    exact = tree.ff(T)
    simulated = np.array([np.mean(times <= t) for t in T])
    error = np.sqrt(exact * (1 - exact) / n)
    assert np.all(np.abs(simulated - exact) <= 4 * error + 1e-12)
    # So does method="simulate", which simulates the dynamic gate alone.
    estimate = tree.ff(T, "simulate", mc_samples=n, seed=4)
    assert np.all(np.abs(estimate - exact) <= 4 * error + 1e-12)


def test_the_simulation_is_reproducible():
    tree = warm_pair()
    first = tree.random(1000, seed=11)
    assert np.array_equal(first, tree.random(1000, seed=11))
    assert not np.array_equal(first, tree.random(1000, seed=12))
    a = tree.ff(T, "simulate", mc_samples=1000, seed=5)
    assert np.array_equal(a, tree.ff(T, "simulate", mc_samples=1000, seed=5))


def test_a_simulated_top_draws_what_the_whole_tree_draws():
    # The dynamic gate is the top, so method="simulate" simulates the
    # whole tree from the same streams as random().
    tree = FaultTree(
        {"top": ("pand", ["a", "b"])}, {"a": E([0.01]), "b": W([90, 2])}
    )
    times = tree.random(5000, seed=8)
    simulated = tree.ff(T, "simulate", mc_samples=5000, seed=8)
    assert np.array_equal(simulated, [np.mean(times <= t) for t in T])


# -- FDEP ---------------------------------------------------------------------


def test_a_functional_dependency_is_its_static_equivalent():
    events = {"a": W([100, 2]), "b": W([80, 1.5]), "c": 0.01, "p": E([0.002])}
    fdep = FaultTree(
        {
            "top": ("or", ["pair", "c"]),
            "pair": ("and", ["a", "b"]),
            "power": ("fdep", "p", ["a", "b"]),
        },
        events,
    )
    written = FaultTree(
        {
            "top": ("or", ["pair", "c"]),
            "pair": ("and", ["a or p", "b or p"]),
            "a or p": ("or", ["a", "p"]),
            "b or p": ("or", ["b", "p"]),
        },
        events,
    )
    assert np.allclose(fdep.ff(T), written.ff(T), rtol=1e-14, atol=0)
    assert set(fdep.minimal_cut_sets()) == set(written.minimal_cut_sets())
    assert fdep.birnbaum_importance(50.0) == pytest.approx(
        written.birnbaum_importance(50.0), rel=1e-12
    )
    assert "power" in fdep.gates and fdep.gates["power"][0] == "fdep"


def test_a_pand_whose_input_is_forced_is_still_integrated():
    a, b, p = W([100, 2]), W([60, 1.2]), E([0.003])
    tree = FaultTree(
        {"top": ("pand", ["a", "b"]), "f": ("fdep", "p", ["b"])},
        {"a": a, "b": b, "p": p},
    )

    def forced_density(s):  # of min(b, p)
        return b.df(s) * p.sf(s) + p.df(s) * b.sf(s)

    for t in T:
        expected, _ = integrate.quad(
            lambda s: a.ff(s) * forced_density(s), 0, t, epsabs=1e-13
        )
        assert tree.ff(t) == pytest.approx(expected, abs=1e-8)


def shared_trigger():
    """A PAND whose two inputs one trigger forces: not an independent
    module, so simulated, the static rest exact."""
    return FaultTree(
        {
            "top": ("or", ["order", "c"]),
            "order": ("pand", ["a", "b"]),
            "f": ("fdep", "p", ["a", "b"]),
        },
        {"a": E([0.01]), "b": E([0.02]), "c": 0.05, "p": E([0.002])},
    )


def test_a_dynamic_gate_that_is_not_a_module_is_simulated_on_request():
    tree = shared_trigger()
    with pytest.raises(NotImplementedError, match="method='simulate'"):
        tree.ff(T)
    routes = tree.analysis_routes()
    assert routes["ff"].route == "refused"
    n = 100_000
    estimate = tree.ff(T, "simulate", mc_samples=n, seed=1)
    whole = tree.random(n, seed=2)
    simulated = np.array([np.mean(whole <= t) for t in T])
    error = np.sqrt(estimate * (1 - estimate) / n)
    assert np.all(np.abs(estimate - simulated) <= 6 * error + 1e-12)
    # The event outside the module is exact: at 0 nothing has occurred
    # but c.
    assert tree.ff(0.0, "simulate", mc_samples=100, seed=1) == pytest.approx(
        0.05, rel=1e-15
    )


def test_inclusive_order_with_simultaneous_forced_failures():
    # Forced at once by p (a probability: at the start), the PAND's
    # inputs are in order.
    tree = FaultTree(
        {"top": ("pand", ["a", "b"]), "f": ("fdep", "p", ["a", "b"])},
        {"a": E([0.01]), "b": E([0.02]), "p": 1.0},
    )
    assert np.all(tree.random(100, seed=1) == 0.0)


# -- modules ------------------------------------------------------------------


def test_the_static_part_around_an_integrated_gate_is_exact():
    a, b, c, d = E([0.01]), W([70, 2]), 0.3, W([200, 1.1])
    tree = FaultTree(
        {
            "top": ("or", ["left", "right"]),
            "left": ("and", ["order", "c"]),
            "right": ("and", ["c", "d"]),
            "order": ("pand", ["a", "b"]),
        },
        {"a": a, "b": b, "c": c, "d": d},
    )
    alone = FaultTree({"top": ("pand", ["a", "b"])}, {"a": a, "b": b})
    pand, d_t = alone.ff(T), d.ff(T)
    assert np.allclose(
        tree.ff(T), c * (pand + d_t - pand * d_t), rtol=1e-12, atol=0
    )


# -- checks and refusals ------------------------------------------------------


def test_dynamic_gates_are_checked():
    with pytest.raises(ValueError, match="must be a basic event"):
        FaultTree(
            {"top": ("spare", ["a", "g"]), "g": ("or", ["b", "c"])},
            {"a": 0.1, "b": 0.1, "c": 0.1},
        )
    with pytest.raises(NotImplementedError, match="shared between SPARE"):
        FaultTree(
            {
                "top": ("and", ["s1", "s2"]),
                "s1": ("spare", ["a", "c"]),
                "s2": ("spare", ["b", "c"]),
            },
            {"a": 0.1, "b": 0.1, "c": 0.1},
        )
    with pytest.raises(ValueError, match="dormancy"):
        FaultTree({"top": ("spare", 1.5, ["a", "b"])}, {"a": 0.1, "b": 0.1})
    with pytest.raises(ValueError, match="at least two inputs"):
        FaultTree({"top": ("pand", ["a"])}, {"a": 0.1})
    with pytest.raises(NotImplementedError, match="NOT gates and dynamic"):
        FaultTree(
            {"top": ("pand", ["a", "not b"]), "not b": ("not", ["b"])},
            {"a": 0.1, "b": 0.1},
        )
    with pytest.raises(NotImplementedError, match="dynamic gates"):
        FaultTree(
            {"top": ("pand", ["a", "b"])},
            {"a": 0.1, "b": 0.1},
            ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
        )
    with pytest.raises(ValueError, match="has no output"):
        FaultTree(
            {"top": ("or", ["a", "f"]), "f": ("fdep", "p", ["a"])},
            {"a": 0.1, "p": 0.1},
        )


def test_what_a_dynamic_tree_does_not_give_is_refused():
    tree = warm_pair()
    for method in ("mean", "minimal_cut_sets", "prime_implicants", "to_rbd"):
        with pytest.raises(NotImplementedError, match="dynamic gates"):
            getattr(tree, method)()
    with pytest.raises(NotImplementedError, match="dynamic gates"):
        tree.birnbaum_importance(10.0)


def test_simulation_options_are_for_dynamic_gates():
    static = FaultTree({"top": ("and", ["a", "b"])}, {"a": 0.1, "b": 0.2})
    with pytest.raises(ValueError, match="has none"):
        static.ff(method="simulate")
    with pytest.raises(ValueError, match="only with method='simulate'"):
        static.ff(seed=1)
    with pytest.raises(ValueError, match="only with method='simulate'"):
        warm_pair().ff(10.0, mc_samples=10)


# -- saving and Open-PSA ------------------------------------------------------


@pytest.mark.parametrize(
    "tree", [consensus_tree(), warm_pair(), shared_trigger()]
)
def test_trees_with_the_new_gates_save_and_load(tree):
    again = FaultTree.from_json(tree.to_json())
    assert again.gates == tree.gates
    method = (
        "simulate"
        if tree.analysis_routes()["ff"].route == "refused"
        else "exact"
    )
    options = {"mc_samples": 500, "seed": 1} if method == "simulate" else {}
    t = None if tree.is_fixed else T
    assert np.array_equal(
        np.asarray(again.ff(t, method, **options)),
        np.asarray(tree.ff(t, method, **options)),
    )


def parsed(tree, **kwargs):
    return ElementTree.fromstring(tree.to_open_psa(**kwargs))


def test_the_open_psa_export_holds_the_static_gates():
    tree = FaultTree(
        {
            "no cooling": ("or", ["valve", "no flow", "two of three"]),
            "no flow": ("and", ["pump 1", "not ok"]),
            "not ok": ("not", ["ok"]),
            "two of three": ("vote", 2, ["x", "y", "z"]),
        },
        {
            "pump 1": E([0.001]),
            "ok": W([500, 2]),
            "valve": 0.05,
            "x": 0.1,
            "y": 0.1,
            "z": 0.1,
        },
    )
    root = parsed(tree)
    gates = {g.get("name"): g for g in root.iter("define-gate")}
    assert set(gates) == {"no_cooling", "no_flow", "not_ok", "two_of_three"}
    assert gates["no_cooling"].find("label").text == "no cooling"
    assert gates["two_of_three"].find("atleast").get("min") == "2"
    assert gates["not_ok"].find("not/basic-event").get("name") == "ok"
    events = {e.get("name"): e for e in root.iter("define-basic-event")}
    assert (
        float(events["pump_1"].find("exponential/float").get("value")) == 0.001
    )
    weibull = [float(f.get("value")) for f in events["ok"].iter("float")]
    assert weibull == [500.0, 2.0, 0.0]
    assert events["valve"].find("float").get("value") == "0.05"
    # At a time, every event is its probability then.
    at = {
        e.get("name"): e
        for e in parsed(tree, t=100.0).iter("define-basic-event")
    }
    assert float(at["ok"].find("float").get("value")) == pytest.approx(
        float(W([500, 2]).ff(100.0))
    )


def test_the_open_psa_export_writes_a_beta_factor_and_an_fdep():
    tree = FaultTree(
        {
            "top": ("or", ["pair", "c"]),
            "pair": ("and", ["a", "b"]),
            "f": ("fdep", "p", ["c"]),
        },
        {"a": 0.1, "b": 0.1, "c": 0.01, "p": 0.001},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1))],
    )
    root = parsed(tree)
    group = root.find(".//define-CCF-group")
    assert group.get("model") == "beta-factor"
    assert [m.get("name") for m in group.iter("basic-event")] == ["a", "b"]
    assert group.find("factor").get("level") == "2"
    events = {e.get("name") for e in root.iter("define-basic-event")}
    assert events == {"c", "p"}
    forced = [
        g for g in root.iter("define-gate") if g.find("label") is not None
    ]
    assert (
        len(forced) == 1
        and "functional dependency" in forced[0].find("label").text
    )


def test_the_open_psa_export_refuses_what_the_format_has_not():
    with pytest.raises(NotImplementedError, match="no dynamic gates"):
        warm_pair().to_open_psa()
    lognormal = FaultTree(
        {"top": ("or", ["a", "b"])},
        {"a": surv.LogNormal.from_params([4.0, 0.5]), "b": 0.1},
    )
    with pytest.raises(NotImplementedError, match="pass t"):
        lognormal.to_open_psa()
    assert "<float" in lognormal.to_open_psa(t=10.0)
    rate = FaultTree(
        {"top": ("and", ["a", "b"])},
        {"a": E([0.01]), "b": E([0.01])},
        ccf_groups=[CCFGroup(["a", "b"], BetaFactor(0.1, basis="rate"))],
    )
    with pytest.raises(NotImplementedError, match="BetaFactor"):
        rate.to_open_psa()
