"""Repairable spares over two echelons (``repyability.rbd._echelons``):
bases and a depot behind them, by METRIC and VARI-METRIC, and the front
of spend against a fleet's availability (the maintenance README's
UC-74)."""

import itertools
import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy import stats

from repyability import RepairableRBD, SparesAllocation
from repyability.rbd import _echelons, _spares

E = surv.Exponential.from_params
W = surv.Weibull.from_params
BASES = {"north": 12, "south": 8}


def series(lives, repair="instant"):
    names = list(lives)
    edges = list(zip(["s"] + names, names + ["t"]))
    return RepairableRBD(
        edges,
        {n: {"reliability": lives[n], "repairability": repair} for n in names},
    )


def fleet():
    """The use case's systems: a pump, a valve and two controllers, each
    swapped in no time."""
    return series(
        {
            "pump": E([1 / 2000.0]),
            "valve": E([1 / 5000.0]),
            "c1": E([1 / 8000.0]),
            "c2": E([1 / 8000.0]),
        }
    )


PARTS = {
    "pump": {
        "price": 12000.0,
        "base_repair_fraction": 0.3,
        "base_repair_time": 48.0,
        "depot_repair_time": 600.0,
        "ship_time": 72.0,
    },
    "valve": {
        "price": 3000.0,
        "base_repair_fraction": 0.5,
        "base_repair_time": 24.0,
        "depot_repair_time": 400.0,
        "ship_time": 72.0,
    },
    "controller": {
        "price": 25000.0,
        "nodes": ["c1", "c2"],
        "base_repair_fraction": 0.1,
        "base_repair_time": 72.0,
        "depot_repair_time": 900.0,
        "ship_time": 120.0,
    },
}

# -- the model written out plainly, as the use case's hand script -----------

TERMS = 200


def moments(pmf, stock):
    """The mean and variance of ``(X - stock)+``, term by term."""
    first = sum((k - stock) * p for k, p in enumerate(pmf) if k > stock)
    second = sum((k - stock) ** 2 * p for k, p in enumerate(pmf) if k > stock)
    return first, second - first**2


def poisson(mean):
    return [stats.poisson.pmf(k, mean) for k in range(TERMS)]


def by_hand(rates, bases, part, depot_stock, method):
    """Each base's pipeline (mean, variance, probabilities) of a part
    with ``depot_stock`` at the depot, the depot's expected backorders
    and the delay of an order; ``rates`` the demand per system."""
    r = part.get("base_repair_fraction", 1.0)
    demand = {j: n * rates for j, n in bases.items()}
    depot = sum(demand.values()) * (1 - r)
    owed, spread = moments(
        poisson(depot * part.get("depot_repair_time", 0.0)), depot_stock
    )
    out = {}
    for j, m in demand.items():
        own = m * (
            r * part.get("base_repair_time", 0.0)
            + (1 - r) * part.get("ship_time", 0.0)
        )
        share = m * (1 - r) / depot if depot else 0.0
        mean = own + share * owed
        variance = own + share * (1 - share) * owed + share**2 * spread
        if method == "metric" or variance <= mean * (1 + 1e-12):
            pmf = poisson(mean)
        else:
            p = mean / variance
            pmf = [
                stats.nbinom.pmf(k, mean * p / (1 - p), p)
                for k in range(TERMS)
            ]
        out[j] = (mean, variance, pmf)
    return out, owed, owed / depot if depot else 0.0


RATES = {"pump": 1 / 2000.0, "valve": 1 / 5000.0, "controller": 2 / 8000.0}
COUNT = {"pump": 1, "valve": 1, "controller": 2}


def stock_by_hand(depot, north, south, method="vari-metric"):
    """The backorders and availability of stocks given as ``{part: n}``
    at the depot and each base."""
    up = {j: 1.0 for j in BASES}
    total = 0.0
    held = {"north": north, "south": south}
    for name, part in PARTS.items():
        out, _, _ = by_hand(RATES[name], BASES, part, depot[name], method)
        for j, n in BASES.items():
            short = moments(out[j][2], held[j][name])[0]
            total += short
            up[j] *= (1 - short / (n * COUNT[name])) ** COUNT[name]
    whole = sum(BASES[j] * up[j] for j in BASES) / sum(BASES.values())
    return total, up, whole


# -- the use case's hand-worked numbers --------------------------------------


def test_nothing_held_is_the_pipelines():
    point = fleet().spares_front(BASES, PARTS, budget=0.0)[0]
    assert point.spend == 0.0
    # The pumps' depot pipeline, 20 / 2000 x 0.7 x 600, all of it owed,
    # and the north base's, 12 / 2000 x (0.3 x 48 + 0.7 x 672).
    assert point.depot_backorders["pump"] == pytest.approx(4.2, rel=1e-12)
    assert point.depot_delay["pump"] == pytest.approx(600.0, rel=1e-12)
    assert point.base_backorders["north"]["pump"] == pytest.approx(
        2.9088, rel=1e-12
    )
    assert point.backorders == pytest.approx(10.466, rel=1e-12)
    for base in BASES:
        assert point.base_availability[base] == pytest.approx(
            0.5631119632, rel=1e-9
        )


def test_one_pump_at_the_depot_by_hand():
    rbd = series({"pump": E([1 / 2000.0])})
    part = _echelons._parts(
        rbd, np.array([12.0, 8.0]), {"pump": PARTS["pump"]}
    )[0]
    depot = _echelons._depot(part)
    assert depot.mean[1] == pytest.approx(3.214996, abs=5e-7)
    assert depot.variance[1] == pytest.approx(4.088808, abs=5e-7)
    assert depot.mean[1] / depot.demand == pytest.approx(459.2851, abs=5e-5)
    mean, variance = _echelons._base_pipeline(part, depot, 0, 1, "vari-metric")
    assert mean == pytest.approx(2.317797, abs=5e-7)
    assert variance == pytest.approx(2.632370, abs=5e-7)
    for method, owed in (("vari-metric", 1.431583), ("metric", 1.416288)):
        found = _echelons._base_backorders(part, depot, 1, method)[0][1]
        assert found == pytest.approx(owed, abs=5e-7)


def test_the_stock_for_95_percent_by_hand():
    held = fleet().spares_allocation(BASES, PARTS, target_availability=0.95)
    assert isinstance(held, SparesAllocation)
    assert held.spend == 246000.0
    assert held.depot_stock == {"pump": 4, "valve": 2, "controller": 4}
    assert held.base_stock == {
        "north": {"pump": 2, "valve": 1, "controller": 1},
        "south": {"pump": 1, "valve": 1, "controller": 1},
    }
    assert held.backorders == pytest.approx(0.85627, abs=5e-6)
    assert held.availability == pytest.approx(0.95778, abs=5e-6)
    assert held.base_availability["north"] == pytest.approx(0.95986, abs=5e-6)
    assert held.base_availability["south"] == pytest.approx(0.95466, abs=5e-6)
    # Each value again from the model written out term by term.
    total, up, whole = stock_by_hand(
        held.depot_stock, held.base_stock["north"], held.base_stock["south"]
    )
    assert held.backorders == pytest.approx(total, rel=1e-10)
    assert held.availability == pytest.approx(whole, rel=1e-10)
    assert held.base_availability == pytest.approx(up, rel=1e-10)


def test_the_front_starts_and_ends_as_by_hand():
    front = fleet().spares_front(BASES, PARTS, target_availability=0.95)
    assert [p.spend for p in front[:4]] == [0.0, 3000.0, 15000.0, 27000.0]
    assert front[1].depot_stock == {"pump": 0, "valve": 1, "controller": 0}
    assert front[1].availability == pytest.approx(0.57943, abs=5e-6)
    assert front[-2].spend == 221000.0
    assert front[-2].availability == pytest.approx(0.94155, abs=5e-6)
    assert front[-1].spend == 246000.0


def test_metric_stops_short_of_the_target():
    rbd = fleet()
    metric = rbd.spares_allocation(
        BASES, PARTS, target_availability=0.95, method="metric"
    )
    assert metric.spend == 218000.0
    assert metric.method == "metric"
    assert metric.availability == pytest.approx(0.95004, abs=5e-6)
    assert metric.depot_stock == {"pump": 4, "valve": 1, "controller": 3}
    # That stock by VARI-METRIC.
    _, _, whole = stock_by_hand(
        metric.depot_stock,
        metric.base_stock["north"],
        metric.base_stock["south"],
    )
    assert whole == pytest.approx(0.93958, abs=5e-6)


# -- backorders and the depot's delay ----------------------------------------


@pytest.mark.parametrize("mean", [0.05, 0.8, 4.2, 37.0])
def test_backorders_against_the_poisson_closed_form(mean):
    # E[(X - s)+] = mean P(X >= s) - s P(X > s) for a Poisson X.
    owed, spread = _echelons._backorders(_echelons._pipeline(mean, mean))
    for stock in range(0, min(len(owed), 60)):
        closed = mean * stats.poisson.sf(stock - 1, mean) - stock * (
            stats.poisson.sf(stock, mean)
        )
        assert owed[stock] == pytest.approx(closed, rel=1e-9, abs=1e-15)
        first, variance = moments(poisson(mean), stock)
        assert owed[stock] == pytest.approx(first, rel=1e-9, abs=1e-15)
        assert spread[stock] == pytest.approx(variance, rel=1e-8, abs=1e-14)


def test_a_negative_binomial_pipeline_has_its_mean_and_variance():
    more = _echelons._pipeline(2.3, 3.1)
    pmf = -np.diff(np.concatenate(([1.0], more)))
    k = np.arange(len(pmf))
    assert pmf @ k == pytest.approx(2.3, rel=1e-12)
    assert pmf @ k**2 - (pmf @ k) ** 2 == pytest.approx(3.1, rel=1e-10)


def test_the_depot_delay_is_its_backorders_per_demand():
    front = fleet().spares_front(BASES, PARTS, budget=120000.0)
    for point in front:
        for name, part in PARTS.items():
            _, owed, delay = by_hand(
                RATES[name], BASES, part, point.depot_stock[name], "metric"
            )
            sent = 20 * RATES[name] * (1 - part["base_repair_fraction"])
            assert point.depot_backorders[name] == pytest.approx(
                owed, rel=1e-10
            )
            assert point.depot_delay[name] == pytest.approx(
                owed / sent, rel=1e-10
            )
            assert point.depot_delay[name] == pytest.approx(delay, rel=1e-10)


def test_vari_metric_owes_no_less_than_metric():
    rbd = fleet()
    for point in rbd.spares_front(BASES, PARTS, budget=150000.0):
        metric, _, _ = stock_by_hand(
            point.depot_stock,
            point.base_stock["north"],
            point.base_stock["south"],
            "metric",
        )
        assert point.backorders >= metric - 1e-12


# -- one site is spares_stock's ----------------------------------------------


@pytest.mark.parametrize("systems", [1, 7])
def test_one_base_and_no_depot_is_spares_stock(systems):
    # Every failure repaired at the base, in the lead time: the base's
    # pipeline is spares_stock's spares on order with the demand going
    # on, and its backorders and fill rate those of that distribution.
    lead = 300.0
    rbd = series({"pump": E([0.01])})
    stock = rbd.spares_stock(lead, fill_rate=0.95, fleet=systems)["pump"]
    on_order = np.asarray(stock.on_order)
    part = {"price": 1.0, "base_repair_time": lead}
    front = rbd.spares_front(
        {"site": systems}, {"pump": part}, budget=40.0, method="metric"
    )
    assert len(front) > 5
    for point in front:
        held = point.base_stock["site"]["pump"]
        assert point.depot_stock == {"pump": 0}
        assert point.spend == held
        owed = sum((k - held) * p for k, p in enumerate(on_order) if k > held)
        # spares_stock keeps its distribution to a tail of 1e-12.
        assert point.backorders == pytest.approx(
            owed, rel=1e-9, abs=100 * _spares._TAIL
        )
    # VARI-METRIC is METRIC with no depot.
    again = rbd.spares_front({"site": systems}, {"pump": part}, budget=40.0)
    assert [p.backorders for p in again] == [p.backorders for p in front]
    # The fill rate with the demand going on, from the base's pipeline.
    pipeline = _echelons._pipeline(
        systems * 0.01 * lead, systems * 0.01 * lead
    )
    going_on = float(np.cumsum(on_order)[stock.stock - 1])
    assert 1.0 - pipeline[stock.stock - 1] == pytest.approx(going_on, rel=1e-9)


def test_the_demand_is_the_diagram_s_long_run_rate():
    # A wear-out life with repairs that take time: the demand is one per
    # cycle, as spares_stock's spares on order have it on average.
    rbd = series(
        {"pump": W([2000.0, 2.5])}, surv.LogNormal.from_params([2.0, 0.5])
    )
    lead = 500.0
    on_order = np.asarray(
        rbd.spares_stock(lead, fill_rate=0.9, fleet=5)["pump"].on_order
    )
    mean = float(on_order @ np.arange(len(on_order)))
    point = rbd.spares_front(
        {"site": 5},
        {"pump": {"price": 1.0, "base_repair_time": lead}},
        budget=0.0,
    )[0]
    assert point.backorders == pytest.approx(mean, rel=1e-5)
    cycle = W([2000.0, 2.5]).mean() + math.exp(2.0 + 0.125)
    assert point.backorders == pytest.approx(5 * lead / cycle, rel=1e-9)


def test_a_demand_rate_given_replaces_the_diagram_s():
    rbd = fleet()
    parts = {name: dict(spec) for name, spec in PARTS.items()}
    given = {
        name: {
            **{k: v for k, v in spec.items() if k != "nodes"},
            "demand_rate": RATES[name],
            "quantity": COUNT[name],
        }
        for name, spec in PARTS.items()
    }
    a = rbd.spares_front(BASES, parts, budget=100000.0)
    b = rbd.spares_front(BASES, given, budget=100000.0)
    assert [p.to_dict() for p in a] == [p.to_dict() for p in b]


# -- the front ---------------------------------------------------------------

SMALL = {
    "a": {
        "price": 3.0,
        "demand_rate": 0.02,
        "base_repair_fraction": 0.2,
        "base_repair_time": 4.0,
        "depot_repair_time": 60.0,
        "ship_time": 6.0,
    },
    "b": {
        "price": 2.0,
        "demand_rate": 0.01,
        "base_repair_fraction": 0.6,
        "base_repair_time": 10.0,
        "depot_repair_time": 90.0,
        "ship_time": 1.0,
    },
}
TWO = {"x": 3, "y": 2}


def every_stock(parts, bases, budget, method):
    """(spend, backorders, stocks) of every stock vector within the
    budget, each from the model written out term by term."""
    names = list(parts)
    per_part = []
    for name in names:
        most = int(budget // parts[name]["price"])
        rows = []
        for depot in range(most + 1):
            out, _, _ = by_hand(
                parts[name]["demand_rate"], bases, parts[name], depot, method
            )
            owed = {
                j: [moments(out[j][2], s)[0] for s in range(most + 1)]
                for j in bases
            }
            for held in itertools.product(
                range(most + 1 - depot), repeat=len(bases)
            ):
                count = depot + sum(held)
                if count <= most:
                    rows.append(
                        (
                            count * parts[name]["price"],
                            sum(owed[j][s] for j, s in zip(bases, held)),
                            (depot,) + held,
                        )
                    )
        per_part.append(rows)
    out = []
    for rows in itertools.product(*per_part):
        spend = sum(row[0] for row in rows)
        if spend <= budget:
            out.append(
                (spend, sum(row[1] for row in rows), [row[2] for row in rows])
            )
    return out


@pytest.mark.parametrize("method", _echelons.METHODS)
def test_every_point_of_the_front_is_on_the_exhaustive_front(method):
    rbd = series({"n": E([1e-3])})
    budget = 17.0
    front = rbd.spares_front(TWO, SMALL, budget=budget, method=method)
    everything = every_stock(SMALL, TWO, budget, method)
    assert len(everything) > 1000
    assert len(front) >= 6
    for point in front:
        # Its values are the searched vector's.
        mine = [
            (point.depot_stock[n],)
            + tuple(point.base_stock[j][n] for j in TWO)
            for n in SMALL
        ]
        same = [row for row in everything if row[2] == mine]
        assert len(same) == 1
        assert point.spend == pytest.approx(same[0][0])
        assert point.backorders == pytest.approx(same[0][1], rel=1e-9)
        # No stock at its spend or less has fewer backorders.
        least = min(
            row[1] for row in everything if row[0] <= point.spend + 1e-9
        )
        assert point.backorders <= least * (1 + 1e-9)


@pytest.mark.parametrize("method", _echelons.METHODS)
def test_a_count_off_the_hull_is_skipped(method):
    # One large base and two small ones behind a depot that repairs
    # everything: the third spare removes little and the fourth a lot
    # (the stock moves from the depot to the bases), so three lie above
    # the line from two to four and are not on the front.
    rbd = series({"n": E([1e-3])})
    bases = {"x": 5, "y": 1, "z": 1}
    part = {
        "price": 1.0,
        "demand_rate": 0.002,
        "base_repair_fraction": 0.0,
        "depot_repair_time": 20.0,
        "ship_time": 30.0,
    }
    curve = _echelons._Curve(
        _echelons._parts(rbd, np.array([5.0, 1.0, 1.0]), {"p": part})[0],
        method,
    )
    owed = [held.backorders for held in curve.counts]
    assert owed[3] - (owed[2] + owed[4]) / 2 > 0.01
    assert curve.hull[:4] == [0, 1, 2, 4]
    front = rbd.spares_front(bases, {"p": part}, budget=6.0, method=method)
    assert [p.spend for p in front] == [0.0, 1.0, 2.0, 4.0, 5.0, 6.0]
    everything = every_stock({"p": part}, bases, 6.0, method)
    for point in front:
        least = min(
            row[1] for row in everything if row[0] <= point.spend + 1e-9
        )
        assert point.backorders <= least * (1 + 1e-9)
        assert point.backorders == pytest.approx(least, rel=1e-9)


@pytest.mark.parametrize("method", _echelons.METHODS)
def test_the_front_is_monotone(method):
    front = fleet().spares_front(
        BASES, PARTS, target_availability=0.999, method=method
    )
    assert len(front) > 25
    spend = [p.spend for p in front]
    owed = [p.backorders for p in front]
    up = [p.availability for p in front]
    assert all(b > a for a, b in zip(spend, spend[1:]))
    assert all(b < a for a, b in zip(owed, owed[1:]))
    assert all(b > a for a, b in zip(up, up[1:]))
    # Backorders removed per unit of spend fall along the front.
    gains = [
        (a.backorders - b.backorders) / (b.spend - a.spend)
        for a, b in zip(front, front[1:])
    ]
    assert all(b <= a * (1 + 1e-9) for a, b in zip(gains, gains[1:]))
    assert front[-1].availability >= 0.999 > front[-2].availability


def test_the_front_does_not_depend_on_how_far_counts_are_worked_out(
    monkeypatch,
):
    rbd = fleet()
    near = rbd.spares_front(BASES, PARTS, target_availability=0.9999)
    start = _echelons._Curve.__init__

    def far(self, part, method):
        start(self, part, method)
        self.most = 64
        self._hull()

    monkeypatch.setattr(_echelons._Curve, "__init__", far)
    again = rbd.spares_front(BASES, PARTS, target_availability=0.9999)
    assert [p.to_dict() for p in near] == [p.to_dict() for p in again]


def test_a_budget_and_a_target():
    rbd = fleet()
    within = rbd.spares_allocation(BASES, PARTS, budget=100000.0)
    assert within.spend == 94000.0
    front = rbd.spares_front(BASES, PARTS, budget=100000.0)
    assert front[-1].to_dict() == within.to_dict()
    both = rbd.spares_allocation(
        BASES, PARTS, budget=300000.0, target_availability=0.95
    )
    assert both.spend == 246000.0
    with pytest.raises(ValueError, match="Raise the budget"):
        rbd.spares_allocation(
            BASES, PARTS, budget=100000.0, target_availability=0.95
        )


def test_a_result_goes_through_json():
    held = fleet().spares_allocation(BASES, PARTS, target_availability=0.9)
    out = json.loads(json.dumps(held.to_dict()))
    assert (
        out["base_stock"]["north"]["pump"] == held.base_stock["north"]["pump"]
    )
    assert out["method"] == "vari-metric"
    assert set(out) == set(held)


@pytest.mark.parametrize("unit", [1e-6, 1e-3, 24.0, 1e6])
def test_results_do_not_depend_on_the_unit_of_time(unit):
    # Times in another unit (rates the other way): the same stocks,
    # backorders and availabilities, and the delays in the unit.
    base = fleet().spares_front(BASES, PARTS, target_availability=0.97)
    scaled_rbd = series(
        {
            "pump": E([1 / (2000.0 * unit)]),
            "valve": E([1 / (5000.0 * unit)]),
            "c1": E([1 / (8000.0 * unit)]),
            "c2": E([1 / (8000.0 * unit)]),
        }
    )
    parts = {
        name: {
            key: value * unit if key.endswith("_time") else value
            for key, value in spec.items()
        }
        for name, spec in PARTS.items()
    }
    scaled = scaled_rbd.spares_front(BASES, parts, target_availability=0.97)
    assert len(scaled) == len(base)
    for a, b in zip(base, scaled):
        assert a.depot_stock == b.depot_stock
        assert a.base_stock == b.base_stock
        assert b.backorders == pytest.approx(a.backorders, rel=1e-9)
        assert b.availability == pytest.approx(a.availability, rel=1e-12)
        for name in PARTS:
            assert b.depot_delay[name] == pytest.approx(
                a.depot_delay[name] * unit, rel=1e-9
            )


# -- a simulation of the two echelons, from the model's definition ----------

#: Fixed before any run: the runs, the warm-up, the window and the seed.
RUNS, WARM_UP, WINDOW, SEED = 2000, 6000.0, 20000.0, 20261127


def simulate(demand, part, depot_stock, base_stock, rng, fixed_depot):
    """One run of one part: Poisson demands at each base, each failed
    part repaired at the base with probability r (an exponential time)
    or at the depot (an exponential time, or a fixed one), ordered from
    the depot at once and sent, first come first served, as the depot
    has one, arriving the ship time later; a demand takes a part from
    its base's shelf, first come first served. Gives the time-averaged
    backorders of the depot and each base over the window, the mean wait
    of an order at the depot, and each base's backorders at the end."""
    end = WARM_UP + WINDOW
    r = part["base_repair_fraction"]
    times, where, local = [], [], []
    for j, rate in enumerate(demand):
        t = np.cumsum(rng.exponential(1 / rate, int(rate * end * 1.5) + 50))
        assert t[-1] > end
        t = t[t < end]
        times.append(t)
        where.append(np.full(len(t), j))
        local.append(rng.random(len(t)) < r)
    t = np.concatenate(times)
    order = np.argsort(t, kind="stable")
    t, base, at_base = (
        t[order],
        np.concatenate(where)[order],
        np.concatenate(local)[order],
    )

    def owed(asked, filled):
        """The time-average of those asked for and not yet filled."""
        a = np.clip(asked, WARM_UP, end)
        b = np.clip(filled, WARM_UP, end)
        return float(np.sum(b - a)) / WINDOW

    def filled(asked, ready, stock):
        """When each ask (in time order) is met, first come first
        served, by a shelf of ``stock`` to which parts come at
        ``ready``."""
        supply = np.concatenate((np.zeros(stock), np.sort(ready)))
        out = np.full(len(asked), np.inf)
        n = min(len(asked), len(supply))
        out[:n] = np.maximum(asked[:n], supply[:n])
        return out

    sent = t[~at_base]
    repair = (
        np.full(len(sent), part["depot_repair_time"])
        if fixed_depot
        else rng.exponential(part["depot_repair_time"], len(sent))
    )
    shipped = filled(sent, sent + repair, depot_stock)
    counted = (sent >= WARM_UP) & (shipped < np.inf)
    delay = float(np.mean((shipped - sent)[counted]))
    back = np.empty(len(t))
    back[~at_base] = shipped + part["ship_time"]
    back[at_base] = t[at_base] + rng.exponential(
        part["base_repair_time"], int(at_base.sum())
    )
    bases, last = [], []
    for j, stock in enumerate(base_stock):
        mine = base == j
        met = filled(t[mine], back[mine], stock)
        bases.append(owed(t[mine], met))
        last.append(int(np.sum(met > end)))
    return owed(sent, shipped), delay, bases, last


def runs(part, depot_stock, base_stock, fixed_depot, seed=SEED):
    demand = [n * part["demand_rate"] for n in BASES.values()]
    rng = np.random.default_rng(seed)
    rows = [
        simulate(demand, part, depot_stock, base_stock, rng, fixed_depot)
        for _ in range(RUNS)
    ]
    depot = np.array([row[0] for row in rows])
    delay = np.array([row[1] for row in rows])
    bases = np.array([row[2] for row in rows])
    last = np.array([row[3] for row in rows])
    return depot, delay, bases, last


def errors(values):
    return values.mean(axis=0), values.std(axis=0, ddof=1) / math.sqrt(
        len(values)
    )


def modelled(part, depot_stock, base_stock, method):
    rbd = series({"n": E([1e-3])})
    one = _echelons._parts(
        rbd, np.array(list(BASES.values()), float), {"p": part}
    )[0]
    depot = _echelons._depot(one)
    each = _echelons._base_backorders(one, depot, depot_stock, method)
    return (
        _echelons._at(depot.mean, depot_stock),
        _echelons._at(depot.mean, depot_stock) / depot.demand,
        [_echelons._at(each[j], s) for j, s in enumerate(base_stock)],
    )


SIM_PUMP = {**PARTS["pump"], "demand_rate": RATES["pump"]}
SIM_VALVE = {**PARTS["valve"], "demand_rate": RATES["valve"]}


def test_the_pipelines_are_exact_where_resupply_times_are_independent():
    # Nothing held at the depot, whose repairs take a fixed time: every
    # order arrives the repair and ship times later, so a base's
    # pipeline is Poisson (Palm's theorem) and both methods are exact.
    stock = (2, 1)
    depot, delay, bases, _ = runs(SIM_PUMP, 0, stock, True)
    for method in _echelons.METHODS:
        owed, wait, each = modelled(SIM_PUMP, 0, stock, method)
        mean, error = errors(depot)
        assert abs(mean - owed) < 4 * error
        assert delay == pytest.approx(wait, rel=1e-12)  # fixed
        mean, error = errors(bases)
        assert np.all(np.abs(mean - each) < 4 * error), (mean, each, error)


def test_the_depot_is_exact_and_vari_metric_is_closer_at_the_bases():
    # A depot with stock and exponential repairs: its backorders and an
    # order's wait are exact (Palm's theorem); the bases' orders then
    # wait in runs, which VARI-METRIC's variance takes in and METRIC's
    # Poisson pipeline does not.
    depot_stock, stock = 3, (2, 1)
    depot, delay, bases, _ = runs(SIM_PUMP, depot_stock, stock, False)
    owed, wait, vari = modelled(SIM_PUMP, depot_stock, stock, "vari-metric")
    _, _, metric = modelled(SIM_PUMP, depot_stock, stock, "metric")
    mean, error = errors(depot)
    assert abs(mean - owed) < 4 * error
    mean, error = errors(delay)
    assert abs(mean - wait) < 4 * error
    mean, error = errors(bases)
    for j in range(2):
        assert abs(mean[j] - vari[j]) < abs(mean[j] - metric[j])
        # METRIC understates them, by more than the runs' error.
        assert mean[j] - metric[j] > 4 * error[j]


def test_the_availability_is_the_product_over_parts():
    # Each system has one of each part, so a part's backorders at a base
    # are holes in different systems, and a system is up with no hole of
    # either part: at the end of each run, the chance of that given the
    # backorders, parts independent. In the setting where the pipelines
    # are exact.
    pump = runs(SIM_PUMP, 0, (2, 1), True)[3]
    valve = runs(SIM_VALVE, 0, (1, 0), True, seed=SEED + 1)[3]
    systems = np.array(list(BASES.values()), float)
    up = (1 - np.minimum(pump, systems) / systems) * (
        1 - np.minimum(valve, systems) / systems
    )
    whole = up @ systems / systems.sum()
    mean, error = errors(whole)
    rbd = series({"n": E([1e-3]), "m": E([1e-3])})
    point = _echelons._allocation(
        list(BASES),
        systems,
        [
            _held(rbd, "pump", SIM_PUMP, 0, (2, 1)),
            _held(rbd, "valve", SIM_VALVE, 0, (1, 0)),
        ],
        "vari-metric",
    )
    assert abs(mean - point.availability) < 4 * error
    mean, error = errors(up)
    for j, base in enumerate(BASES):
        assert abs(mean[j] - point.base_availability[base]) < 4 * error[j]


def _held(rbd, name, part, depot_stock, base_stock):
    """A part's curve held at a given stock."""
    systems = np.array(list(BASES.values()), float)
    curve = _echelons._Curve(
        _echelons._parts(rbd, systems, {name: part})[0], "vari-metric"
    )
    curve.counts = [_echelons._Held(0.0, depot_stock, tuple(base_stock))]
    curve.at = 0
    return curve


def test_several_of_a_part_in_a_system():
    # Z of a part in each of N systems: each of the N Z places holds one
    # with chance 1 - EBO / (N Z).
    rbd = series({"n": E([1e-3])})
    part = {**SIM_PUMP, "quantity": 3, "demand_rate": 3 * RATES["pump"]}
    point = rbd.spares_allocation(BASES, {"p": part}, budget=5 * 12000.0)
    for base, systems in BASES.items():
        owed = point.base_backorders[base]["p"]
        assert point.base_availability[base] == pytest.approx(
            (1 - owed / (3 * systems)) ** 3, rel=1e-12
        )


# -- what is refused ---------------------------------------------------------


def test_a_redundant_diagram_is_refused_naming_what_to_use():
    life = {"reliability": E([1e-3]), "repairability": "instant"}
    rbd = RepairableRBD(
        [("s", "a"), ("s", "b"), ("a", "t"), ("b", "t")],
        {"a": life, "b": life},
    )
    part = {"price": 1.0, "base_repair_time": 5.0}
    with pytest.raises(NotImplementedError, match="spares_stock") as error:
        rbd.spares_front({"x": 2}, {"a": part}, budget=3.0)
    assert rbd.analysis_routes()["spares_front"].reason == str(error.value)
    assert rbd.analysis_routes()["spares_allocation"].route == "refused"


def test_a_series_diagram_is_numerical():
    routes = fleet().analysis_routes()
    assert routes["spares_front"].route == "numerical"
    assert routes["spares_allocation"].route == "numerical"


def test_a_part_whose_replacements_are_not_counted_takes_a_demand_rate():
    unit = {"reliability": E([1e-3]), "repairability": E([0.1])}
    rbd = RepairableRBD(
        [("s", "a"), ("a", "b"), ("b", "t")],
        {"a": unit, "b": unit},
        repair_crews=1,
    )
    part = {"price": 1.0, "base_repair_time": 5.0}
    with pytest.raises(NotImplementedError, match="demand_rate"):
        rbd.spares_front({"x": 2}, {"a": part}, budget=3.0)
    front = rbd.spares_front(
        {"x": 2}, {"a": {**part, "demand_rate": 1e-3}}, budget=3.0
    )
    assert front[0].backorders == pytest.approx(2 * 1e-3 * 5.0)


@pytest.mark.parametrize(
    "bases, parts, options, message",
    [
        ({}, PARTS, {"budget": 1.0}, "bases must be a dict"),
        ({"x": 0}, PARTS, {"budget": 1.0}, "systems at base 'x'"),
        ({"x": 2.5}, PARTS, {"budget": 1.0}, "systems at base 'x'"),
        (BASES, {}, {"budget": 1.0}, "parts must be a dict"),
        (BASES, {"pump": 3}, {"budget": 1.0}, "must be a dict"),
        (BASES, {"pump": {"cost": 1.0}}, {"budget": 1.0}, "unknown keys"),
        (
            BASES,
            {"pump": {"base_repair_time": 1.0}},
            {"budget": 1.0},
            "'price'",
        ),
        (
            BASES,
            {"pump": {"price": 0.0, "base_repair_time": 1.0}},
            {"budget": 1.0},
            "price must be above 0",
        ),
        (
            BASES,
            {"pump": {"price": "1", "base_repair_time": 1.0}},
            {"budget": 1.0},
            "price must be a number",
        ),
        (
            BASES,
            {"pump": {"price": 1.0}},
            {"budget": 1.0},
            "'base_repair_time'",
        ),
        (
            BASES,
            {
                "pump": {
                    "price": 1.0,
                    "base_repair_fraction": 0.5,
                    "base_repair_time": 1.0,
                }
            },
            {"budget": 1.0},
            "'depot_repair_time'",
        ),
        (
            BASES,
            {
                "pump": {
                    "price": 1.0,
                    "base_repair_fraction": 1.5,
                    "base_repair_time": 1.0,
                }
            },
            {"budget": 1.0},
            "base_repair_fraction must be a number, from 0 to 1",
        ),
        (
            BASES,
            {"pump": {"price": 1.0, "base_repair_time": -1.0}},
            {"budget": 1.0},
            "base_repair_time must be a number",
        ),
        (
            BASES,
            {"gear": {"price": 1.0, "base_repair_time": 1.0}},
            {"budget": 1.0},
            "not a component of the diagram",
        ),
        (
            BASES,
            {
                "a": {
                    "price": 1.0,
                    "base_repair_time": 1.0,
                    "nodes": ["pump"],
                },
                "b": {
                    "price": 1.0,
                    "base_repair_time": 1.0,
                    "nodes": ["pump"],
                },
            },
            {"budget": 1.0},
            "a component is one part's",
        ),
        (
            BASES,
            {"pump": {"price": 1.0, "base_repair_time": 1.0, "nodes": "pump"}},
            {"budget": 1.0},
            "must be a list",
        ),
        (
            BASES,
            {
                "pump": {
                    "price": 1.0,
                    "base_repair_time": 1.0,
                    "nodes": ["pump"],
                    "demand_rate": 0.1,
                }
            },
            {"budget": 1.0},
            "both a demand_rate and nodes",
        ),
        (
            BASES,
            {"pump": {"price": 1.0, "base_repair_time": 1.0, "quantity": 0}},
            {"budget": 1.0},
            "quantity",
        ),
        (BASES, PARTS, {}, "Give a budget"),
        (BASES, PARTS, {"budget": -1.0}, "budget must be a number"),
        (BASES, PARTS, {"target_availability": 1.0}, "in \\(0, 1\\)"),
        (
            BASES,
            PARTS,
            {"budget": 1.0, "method": "mod-metric"},
            "method must be",
        ),
    ],
)
def test_inputs_are_checked(bases, parts, options, message):
    with pytest.raises(ValueError, match=message):
        fleet().spares_front(bases, parts, **options)


def test_a_part_never_needed_takes_no_stock():
    rbd = fleet()
    parts = {
        "pump": PARTS["pump"],
        "spare wheel": {
            "price": 1.0,
            "demand_rate": 0.0,
            "base_repair_time": 5.0,
        },
    }
    front = rbd.spares_front(BASES, parts, budget=50000.0)
    assert all(p.depot_stock["spare wheel"] == 0 for p in front)
    assert all(p.base_stock["north"]["spare wheel"] == 0 for p in front)
    assert front[-1].spend == 48000.0
