"""The integrals over a window on coarse pieces (#164): ``_quadrature``.

The mission availability and capacity, and the expected events and cost,
are summed over pieces cut where the curves bend and a few steps long of
the finest grid still changing, each halved until its quadrature agrees
with its halves'. They are checked here against the integral summed between
every knot of every curve (as before #164), against closed forms, and for
not depending on how coarse the pieces are; and the pieces for a long
mission on many components against the millions there were.
"""

import numpy as np
import pytest
import surpyval as surv

from repyability import RepairableRBD
from repyability.rbd import _quadrature
from repyability.rbd._point_availability import ChainDips

E = surv.Exponential.from_params
W = surv.Weibull.from_params
L = surv.LogNormal.from_params
X = surv.ExactEventTime.from_params

PARALLEL = [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")]


def unit(life, repair, **more):
    return {"reliability": life, "repairability": repair, **more}


def ladder(pairs, seed=0):
    """``pairs`` parallel pairs in series: Weibull lives, lognormal
    repairs."""
    rng = np.random.default_rng(seed)
    edges, components, previous = [], {}, ["s"]
    for i in range(pairs):
        layer = [f"A{i}", f"B{i}"]
        edges += [(p, n) for p in previous for n in layer]
        for n in layer:
            components[n] = unit(
                W([rng.uniform(2000, 20000), rng.uniform(1.2, 3.0)]),
                L([rng.uniform(1.5, 3.5), rng.uniform(0.3, 0.8)]),
            )
        previous = layer
    edges += [(p, "t") for p in previous]
    return RepairableRBD(edges, components)


def systems():
    """One of each kind of curve: plain units (one whose life is known
    exactly), age and block replacement, hidden failures, a standby group
    and a nested RBD."""
    inner = RepairableRBD(
        PARALLEL,
        {
            "a": unit(W([300.0, 2.0]), E([0.2])),
            "b": unit(E([0.004]), E([0.5])),
        },
    )
    return {
        "plain": RepairableRBD(
            PARALLEL,
            {
                "a": unit(W([500.0, 0.8]), L([1.0, 0.5])),
                "b": unit(X([300.0]), E([0.1])),
            },
        ),
        "age and block": RepairableRBD(
            PARALLEL,
            {
                "a": unit(
                    W([400.0, 2.2]),
                    E([0.5]),
                    preventive={"interval": 250.0, "duration": E([1.0])},
                ),
                "b": unit(
                    W([600.0, 1.6]),
                    L([0.5, 0.4]),
                    preventive={
                        "interval": 300.0,
                        "policy": "block",
                        "duration": E([2.0]),
                    },
                ),
            },
        ),
        "tested": RepairableRBD(
            PARALLEL,
            {
                "a": unit(
                    E([0.002]), "instant", inspection={"interval": 100.0}
                ),
                "b": unit(W([500.0, 1.5]), E([0.5])),
            },
        ),
        "standby": RepairableRBD(
            [("s", "g"), ("g", "c"), ("c", "t")],
            {
                "g": unit(E([0.01]), E([0.2]), standby={"units": 2}),
                "c": unit(W([900.0, 1.8]), E([0.25])),
            },
        ),
        "nested": RepairableRBD(
            [("s", "n"), ("n", "z"), ("z", "t")],
            {"n": inner, "z": unit(W([400.0, 1.2]), E([1.0]))},
        ),
    }


def every_knot(rbd, t):
    """The mission availability over ``[0, t]`` summed between every knot
    of every curve (``t`` before the curves settle), as before #164."""
    curves = rbd._availability_curves(t, set())
    edges = np.unique(
        np.concatenate(
            [[0.0, t]] + [curve.knots(0.0, t) for curve in curves.values()]
        )
    )
    edges = edges[(edges >= 0.0) & (edges <= t)]
    a, b = edges[:-1], edges[1:]
    x, half = _quadrature.points(a, b)
    up = rbd._curves_at(curves, x, set(), set(), "p")
    return float(_quadrature.summed(up, half).sum()) / t


@pytest.mark.parametrize("name", list(systems()))
def test_the_mission_is_what_summing_between_every_knot_gives(name):
    rbd = systems()[name]
    t = 1500.0
    assert rbd.mission_availability(t) == pytest.approx(
        every_knot(rbd, t), rel=1e-9, abs=1e-12
    )


@pytest.mark.parametrize("name", list(systems()))
def test_the_counts_do_not_depend_on_how_coarse_the_pieces_are(
    name, monkeypatch
):
    rbd = systems()[name]
    t = np.array([400.0, 1500.0])
    coarse = rbd.expected_events(t)
    monkeypatch.setattr(_quadrature, "PIECE_STEPS", 1)
    fine = rbd.expected_events(t)
    for field in ("system_failures", "system_planned_outages"):
        np.testing.assert_allclose(
            getattr(coarse, field), getattr(fine, field), rtol=1e-8, atol=1e-14
        )
    np.testing.assert_allclose(
        coarse.system_downtime, fine.system_downtime, rtol=4e-8
    )


def test_a_long_mission_on_many_components_takes_few_pieces(monkeypatch):
    # 40 pairs over ten years: their curves have about 400,000 knots
    # between them, where the integral was summed before #164.
    rbd = ladder(20)
    t = 87600.0
    seen = []
    refined = _quadrature.refined

    def counted(*args, **kwargs):
        edges, values = refined(*args, **kwargs)
        seen.append(len(edges) - 1)
        return edges, values

    monkeypatch.setattr(_quadrature, "refined", counted)
    value = rbd.mission_availability(t)
    curves = rbd._availability_curves(t, set())
    knots = sum(len(c.knots(0.0, t)) for c in curves.values())
    assert knots > 200_000
    assert seen and seen[0] < 20_000
    assert 0.999 < value < 1.0


def test_one_exponential_unit_is_its_closed_form():
    lam, mu = 0.1, 1.0
    rbd = RepairableRBD(
        [("s", "c"), ("c", "t")], {"c": unit(E([lam]), E([mu]))}
    )
    t = np.array([0.5, 10.0, 1000.0])
    total = lam + mu
    up = mu / total * t + lam / total**2 * -np.expm1(-total * t)
    # (The curve itself is good to about 4e-7 soon after the start.)
    np.testing.assert_allclose(rbd.mission_availability(t), up / t, rtol=8e-7)
    np.testing.assert_allclose(rbd.expected_failures(t), lam * up, rtol=8e-7)


def test_the_stieltjes_weights_are_exact_for_a_cubic_against_a_quintic():
    rng = np.random.default_rng(1)
    f = np.polynomial.Polynomial(rng.normal(size=4))
    M = np.polynomial.Polynomial(rng.normal(size=6))
    a = np.array([0.0, 1.5, -2.0])
    b = np.array([1.0, 4.0, -1.75])
    x, _ = _quadrature.points(a, b)
    got = _quadrature.stieltjes(f(x), M(a), M(b), M(x))
    antiderivative = (f * M.deriv()).integ()
    np.testing.assert_allclose(got, antiderivative(b) - antiderivative(a))
    # A constant takes the measure's mass in each piece exactly.
    ones = np.ones_like(x)
    np.testing.assert_allclose(
        _quadrature.stieltjes(ones, M(a), M(b), M(x)), M(b) - M(a)
    )


class _Curve:
    """A curve with breaks and a grid, for the pieces."""

    def __init__(self, breaks, grids):
        self._breaks = np.asarray(breaks, dtype=float)
        self._grids = grids

    def breaks(self, start, stop):
        keep = (self._breaks >= start) & (self._breaks <= stop)
        return self._breaks[keep]

    def grids(self):
        return self._grids


def test_the_pieces_keep_the_breaks_and_follow_the_finest_grid_there():
    fine = _Curve([3.3], [(0.01, 10.0)])  # changes up to 10
    coarse = _Curve([47.0], [(0.5, np.inf)])
    edges, finest = _quadrature.pieces(
        [fine, coarse], np.array([25.0]), 100.0, 10**6
    )
    for t in (0.0, 3.3, 10.0, 25.0, 47.0, 100.0):
        assert np.any(np.isclose(edges, t, rtol=0, atol=1e-12))
    widths = np.diff(edges)
    early = edges[:-1] < 10.0
    assert widths[early].max() <= _quadrature.PIECE_STEPS * 0.01 * (1 + 1e-9)
    assert widths[~early].max() <= _quadrature.PIECE_STEPS * 0.5 * (1 + 1e-9)
    assert widths[~early].min() > _quadrature.PIECE_STEPS * 0.01
    np.testing.assert_array_equal(finest[early], 0.01)
    np.testing.assert_array_equal(finest[~early], 0.5)
    with pytest.raises(_quadrature.TooMany):
        _quadrature.pieces([fine], np.empty(0), 10.0, 100)


def test_refining_finds_a_bend_between_the_breaks():
    # A kink at 0.3 (no break there) in one long piece: halving finds it.
    def estimate(a, b):
        x, half = _quadrature.points(a, b)
        return {"f": _quadrature.summed(np.abs(x - 0.3), half)}

    edges, values = _quadrature.refined(
        estimate, np.array([0.0, 1.0]), np.array([np.inf]), 10**6
    )
    exact = 0.3**2 / 2 + 0.7**2 / 2
    assert values["f"].sum() == pytest.approx(exact, abs=1e-9)
    assert len(edges) > 10
    # ... and for expected events (a rate, relative to its mean), not below
    # a grid's step, within which the counts are no closer.
    edges, values = _quadrature.refined(
        estimate, np.array([0.0, 1.0]), np.array([0.2]), 10**6, {"f"}
    )
    assert np.diff(edges).min() >= 0.1


def test_the_chained_maintenance_counts_do_not_dip():
    # The sums of the maintenance times of the units that each reach their
    # age: an exponential time's density jumps at 0, where their running
    # sums, and the cubic between them, once undershot below 0, and the
    # counts dipped by 1e-4 (#164). The cubic may still wiggle by its own
    # accuracy.
    chain = ChainDips(
        200.0, 0.6, E([1.0]).sf, [0.0, 1.0, 5.0, 20.0, 40.0], 3, cdfs=True
    )
    for start, step, cdf, mass in chain.cdfs:
        assert cdf.min() >= 0.0 and cdf.max() <= mass
    x = np.linspace(380.0, 660.0, 20001)
    counts = chain.before(x)
    assert np.diff(counts).min() > -1e-6
