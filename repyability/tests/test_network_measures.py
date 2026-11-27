"""A network's density, hazard, cumulative hazard, levers and parameter
sensitivity: the same measures a block diagram has, worked out from the
network's exact values."""

import math

import numpy as np
import pytest
import surpyval as surv
from surpyval import FixedEventProbability

from repyability import Lever, Network, NonRepairableRBD

E, W = surv.Exponential.from_params, surv.Weibull.from_params
F = FixedEventProbability.from_params
T = np.array([0.0, 5.0, 20.0, 50.0, 100.0, 250.0])
LINKS = {
    "a": ("s", "x"),
    "b": ("s", "y"),
    "c": ("x", "y"),
    "d": ("x", "t"),
    "e": ("y", "t"),
}


def bridge(models, nodes=None):
    return Network(
        {name: (*ends, models[name]) for name, ends in LINKS.items()},
        "s",
        "t",
        nodes=nodes,
    )


def models(c=1.0):
    return {
        name: W([(100.0 + 10.0 * i) * c, 1.5 + 0.25 * i])
        for i, name in enumerate(LINKS)
    }


def mixed():
    """A bridge of different lives, a failing node, a probability."""
    given = models()
    given["c"] = E([0.01])
    given["e"] = F(0.05)
    return bridge(given, nodes={"x": W([300.0, 2.0])})


# -- against the same system drawn as a diagram ------------------------------


SERIES = (
    Network(
        {
            "a": (0, 1, W([100.0, 2.0])),
            "b": (1, 2, E([0.01])),
            "c": (2, 3, W([150.0, 1.2])),
        },
        0,
        3,
    ),
    NonRepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "c"), ("c", "t")],
        {"a": W([100.0, 2.0]), "b": E([0.01]), "c": W([150.0, 1.2])},
    ),
)
PARALLEL = (
    Network(
        {
            "a": (0, 1, W([100.0, 2.0])),
            "b": (0, 1, E([0.01])),
            "c": (0, 1, W([150.0, 1.2])),
        },
        0,
        1,
    ),
    NonRepairableRBD(
        [
            ("s", "a"),
            ("s", "b"),
            ("s", "c"),
            ("a", "t"),
            ("b", "t"),
            ("c", "t"),
        ],
        {"a": W([100.0, 2.0]), "b": E([0.01]), "c": W([150.0, 1.2])},
    ),
)


@pytest.mark.parametrize(
    "pair", [SERIES, PARALLEL], ids=["series", "parallel"]
)
def test_a_network_that_is_a_diagram_has_the_diagrams_measures(pair):
    net, rbd = pair
    # The reliabilities agree to rounding (1e-12), and a difference of them
    # over a step of 1e-6 of the time magnifies that rounding by 1e6.
    for method, rtol in (("df", 1e-9), ("hf", 1e-9), ("Hf", 1e-12)):
        np.testing.assert_allclose(
            getattr(net, method)(T),
            getattr(rbd, method)(T),
            rtol=rtol,
            atol=1e-14,
            err_msg=method,
        )
    np.testing.assert_allclose(net.sf(T), rbd.sf(T), rtol=1e-12)
    assert net.df(50.0) == pytest.approx(rbd.df(50.0), rel=1e-9)
    ours, theirs = net.parameter_sensitivity(T), rbd.parameter_sensitivity(T)
    assert list(ours) == list(theirs)
    for name in theirs:
        assert list(ours[name]) == list(theirs[name])
        for parameter in theirs[name]:
            np.testing.assert_allclose(
                ours[name][parameter],
                theirs[name][parameter],
                rtol=1e-12,
                atol=1e-14,
            )
    assert [
        (lever.key, lever.name, lever.value, lever.bounds)
        for lever in net.levers()
    ] == [
        (lever.key, lever.name, lever.value, lever.bounds)
        for lever in rbd.levers()
    ]


# -- against closed forms ----------------------------------------------------


def bridge_slope(p):
    """dR/dp of the bridge with every link working with probability p."""
    return 4 * p + 6 * p**2 - 20 * p**3 + 10 * p**4


def test_the_bridge_density_is_the_derivative_of_its_reliability():
    # Every link the same life, so R(t) = g(p(t)) and f = g'(p) f_link;
    # the difference is central, with an error of order dx squared.
    life = W([100.0, 1.7])
    net = bridge({name: life for name in LINKS})
    t = T[1:]
    expected = bridge_slope(life.sf(t)) * life.df(t)
    np.testing.assert_allclose(net.df(t), expected, rtol=1e-7)
    np.testing.assert_allclose(net.hf(t), expected / net.sf(t), rtol=1e-7)
    # The unreliability, in the links' own: 2q^2 + 2q^3 - 5q^4 + 2q^5.
    q = life.ff(t)
    unreliability = 2 * q**2 + 2 * q**3 - 5 * q**4 + 2 * q**5
    np.testing.assert_allclose(
        net.Hf(t), -np.log1p(-unreliability), rtol=1e-12
    )


def test_a_small_unreliability_keeps_its_precision():
    life = E([1e-9])
    net = bridge({name: life for name in LINKS})
    # The bridge fails with two links down first: F = 2 q^2 as q -> 0.
    q = 1e-9 * 1.0
    assert net.Hf(1.0) == pytest.approx(2 * q**2, rel=1e-6)
    assert net.hf(1.0) == pytest.approx(4 * q * 1e-9, rel=1e-6)


def test_a_certain_failure_has_an_infinite_hazard():
    net = Network({"a": ("s", "t", 1.0)}, "s", "t")
    assert net.Hf() == math.inf
    assert net.hf() == math.inf
    assert net.df() == 0.0


def test_a_network_of_probabilities_has_a_density_of_zero():
    net = bridge({name: F(0.1) for name in LINKS})
    assert net.df() == 0.0
    assert net.hf() == 0.0
    assert net.Hf() == pytest.approx(-math.log(net.sf()), rel=1e-12)


def test_a_time_is_needed_where_a_model_depends_on_it():
    for method in ("df", "hf", "Hf", "parameter_sensitivity"):
        with pytest.raises(ValueError, match="x is required"):
            getattr(mixed(), method)()


def test_a_scalar_time_gives_numbers_and_an_array_arrays():
    net = mixed()
    assert isinstance(net.df(50.0), float)
    assert isinstance(net.parameter_sensitivity(50.0)["a"]["alpha"], float)
    assert net.df(T).shape == T.shape
    assert net.parameter_sensitivity(T)["a"]["alpha"].shape == T.shape


# -- levers and parameter sensitivity ----------------------------------------


def test_the_levers_are_those_the_sensitivities_report():
    net = mixed()
    levers = net.levers()
    assert all(isinstance(lever, Lever) for lever in levers)
    reported = net.parameter_sensitivity(50.0)
    assert [(lever.key, lever.name) for lever in levers] == [
        (k, n) for k, d in reported.items() for n in d
    ]
    assert [
        (lever.key, lever.name, lever.value)
        for lever in levers
        if lever.key == "e"
    ] == [("e", "p", 0.05)]
    assert [lever.key for lever in levers][-2:] == ["x", "x"]


def test_a_lever_s_range_is_where_its_network_builds():
    net = mixed()
    for lever in net.levers():
        low, high = lever.bounds
        assert low <= lever.value <= high, lever
        net.with_levers({lever: lever.value})
        for end, past in ((low, -1.0), (high, 1.0)):
            if not math.isfinite(end):
                continue
            with pytest.raises(ValueError, match=f"{lever.name!r} of"):
                net.with_levers({lever: end + past * 1e-6})


def test_the_sensitivities_are_the_slopes_of_the_reliability():
    net = mixed()
    step = 1e-4
    t = np.array([20.0, 60.0, 150.0])
    reported = net.parameter_sensitivity(t, rel_step=step)
    for lever in net.levers():
        h = step * lever.value
        up = net.with_levers({lever: lever.value + h}).sf(t)
        down = net.with_levers({lever: lever.value - h}).sf(t)
        np.testing.assert_allclose(
            reported[lever.key][lever.name],
            (up - down) / (2.0 * h),
            rtol=1e-6,
            atol=1e-12,
            err_msg=f"{lever.key} {lever.name}",
        )


def test_with_levers_moves_a_link_or_a_node_and_leaves_the_network_alone():
    net = mixed()
    moved = net.with_levers(
        {("a", "alpha"): 200.0, ("x", "beta"): 3.0, ("e", "p"): 0.2}
    )
    assert moved.models["a"].params[0] == 200.0
    assert moved.nodes["x"].params[1] == 3.0
    assert moved.sf(50.0) != net.sf(50.0)
    assert net.models["a"].params[0] == 100.0
    assert net.nodes["x"].params[1] == 2.0
    assert moved.links == net.links and moved.source == net.source
    # One alone: twice the scale of a single link.
    one = Network({"a": ("s", "t", W([100.0, 2.0]))}, "s", "t")
    assert one.with_levers({("a", "alpha"): 200.0}).sf(100.0) == (
        pytest.approx(math.exp(-0.25))
    )


@pytest.mark.parametrize(
    "given, error, message",
    [
        (
            {("a", "alpah"): 1.0},
            ValueError,
            "'a' has no lever 'alpah'. Did you mean 'alpha'?",
        ),
        (
            {("z", "alpha"): 1.0},
            ValueError,
            "'z' has no levers: it is no link",
        ),
        ({("a", "alpha"): -1.0}, ValueError, "'alpha' of 'a' cannot be"),
        ({("e", "p"): 1.5}, ValueError, "'p' of 'e' cannot be"),
        ({("a", "alpha"): "100"}, TypeError, "must be given a"),
        ({"a": 1.0}, TypeError, "by its Lever.*or as \\(key, name\\)"),
    ],
)
def test_with_levers_refuses_what_is_no_lever_or_value(given, error, message):
    with pytest.raises(error, match=message):
        mixed().with_levers(given)


def test_with_levers_takes_a_dict():
    with pytest.raises(TypeError, match="takes a dict"):
        mixed().with_levers([(("a", "alpha"), 1.0)])


def test_an_element_with_no_parameters_has_no_lever():
    class Bare:
        def sf(self, x):
            return np.exp(-0.01 * np.asarray(x, dtype=float))

        def ff(self, x):
            return -np.expm1(-0.01 * np.asarray(x, dtype=float))

    net = Network(
        {"a": ("s", "t", Bare()), "b": ("s", "t", E([0.01]))}, "s", "t"
    )
    assert [lever.key for lever in net.levers()] == ["b"]
    assert list(net.parameter_sensitivity(10.0)) == ["b"]


# -- the unit of time --------------------------------------------------------


@pytest.mark.parametrize("c", [1e-6, 1e-3, 1e3, 1e6])
def test_the_measures_scale_with_the_unit(c):
    base, scaled = mixed(), None
    given = models(c)
    given["c"] = E([0.01 / c])
    given["e"] = F(0.05)
    scaled = bridge(given, nodes={"x": W([300.0 * c, 2.0])})
    for what, (x, y) in {
        "df": (base.df(T), np.asarray(scaled.df(T * c)) * c),
        "hf": (base.hf(T[1:]), np.asarray(scaled.hf(T[1:] * c)) * c),
        "Hf": (base.Hf(T), scaled.Hf(T * c)),
    }.items():
        scale = np.max(np.abs(x))
        np.testing.assert_allclose(
            y, x, rtol=1e-6, atol=1e-12 * scale, err_msg=what
        )
    ours, theirs = base.parameter_sensitivity(T), scaled.parameter_sensitivity(
        T * c
    )
    # A scale or a rate moves with the unit; a shape or a probability not.
    per = {"alpha": 1.0 / c, "failure_rate": c, "beta": 1.0, "p": 1.0}
    for name, parameters in ours.items():
        for parameter, value in parameters.items():
            np.testing.assert_allclose(
                theirs[name][parameter] / per[parameter],
                value,
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"{name} {parameter}",
            )
