"""The value of running one unit on an operating schedule
(``operating_schedule_value``, ``best_switch_time``: the maintenance use
case UC-20): closed forms, an independent renewal function, the exact
values of a one-component ``RepairableRBD``, a hand-written seeded
simulation, the switch time against a scan, the unit of time, and the
refusals."""

import json
import math
import pydoc
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    NonRepairable,
    RepairableRBD,
    ScheduleValue,
    SwitchTime,
    best_switch_time,
    operating_schedule_value,
)
from repyability.maintenance import _operating


def _model(family, params):
    """A surpyval regression model of ``family`` (fitted to small data,
    then given ``params``, through its saved form)."""
    rng = np.random.default_rng(0)
    Z = rng.uniform(0.0, 1.0, (60, 1))
    x = rng.weibull(2.0, 60) * 100.0 * np.exp(-Z[:, 0])
    saved = family.fit(x, Z=Z).to_dict()
    saved["params"] = [float(p) for p in params]
    return surv.from_dict(saved)


def _close(a, b, rel):
    assert a == pytest.approx(b, rel=rel), (a, b)


# -- 1. exponential lives and repairs: a two-state Markov chain -----------


def _markov(lams, mu, starts, H, r, rates):
    """The two-state chain's values, stretch by stretch: up with
    availability ``A(t) = A_inf + (A_c - A_inf) exp(-k (t - c))`` from the
    stretch's start ``c``, ``A_inf = mu / (lam + mu)``, ``k = lam + mu``;
    failures at ``lam A``. Returns (failures, up, failures PV, value PV)."""
    ends = list(starts[1:]) + [H]
    A = 1.0
    out = np.zeros(4)
    for lam, c, e, w in zip(lams, starts, ends, rates):
        L = e - c
        k = lam + mu
        A_inf = mu / k
        B = A - A_inf
        up = A_inf * L + B * -math.expm1(-k * L) / k
        up_pv = math.exp(-r * c) * (
            A_inf * -math.expm1(-r * L) / r
            + B * -math.expm1(-(r + k) * L) / (r + k)
        )
        out += [lam * up, up, lam * up_pv, w * up_pv]
        A = A_inf + B * math.exp(-k * L)
    return out


def test_exponential_lives_and_repairs_follow_the_two_state_chain():
    # Exponential AFT: rate 0.01 at z = 0, 0.04 at z = 1 (beta = ln 4);
    # repairs exponential at rate 0.5. Cool to 60, hot to 130, cool to 200.
    model = _model(surv.ExponentialAFT, [0.01, math.log(4.0)])
    repair = surv.Exponential.from_params([0.5])
    starts, Z, rates = [0.0, 60.0, 130.0], [0.0, 1.0, 0.0], [1.0, 2.5, 1.0]
    H, r = 200.0, 0.005
    out = operating_schedule_value(
        model,
        starts,
        Z,
        rates,
        H,
        failure_cost=30.0,
        down_time=repair,
        discount_rate=r,
    )
    failures, up, failures_pv, value_pv = _markov(
        [0.01, 0.04, 0.01], 0.5, starts, H, r, rates
    )
    _close(out.failures, failures, 1e-5)
    _close(out.up_time, up, 1e-6)
    _close(out.cost, 30.0 * failures_pv, 1e-5)
    _close(out.value, value_pv, 1e-6)
    _close(out.npv, value_pv - 30.0 * failures_pv, 1e-5)
    assert out.method == "numerical" and out.error <= out.tolerance
    # Each stretch on its own: the chain from where the last left it.
    A = 1.0
    for row, lam, c, e in zip(
        out.stretches, [0.01, 0.04, 0.01], starts, starts[1:] + [H]
    ):
        k = lam + 0.5
        up = 0.5 / k * (e - c) + (A - 0.5 / k) * -math.expm1(-k * (e - c)) / k
        _close(row["up_time"], up, 1e-6)
        _close(row["failures"], lam * up, 1e-5)
        A = 0.5 / k + (A - 0.5 / k) * math.exp(-k * (e - c))


# -- 2. instant replacement, AFT: a renewal process in exposure -----------


def test_aft_with_instant_replacement_renews_in_exposure_time():
    # Weibull AFT: baseline scale 100, shape 2.5; phi = exp(1.2 z). In the
    # exposure psi(t) = integral of phi(z(u)) du the failures are an
    # ordinary renewal process of the baseline life, so the expected
    # failures by t are its renewal function at psi(t), here from
    # NonRepairable.failures (a RepairableRBD's exact expected events).
    model = _model(surv.WeibullAFT, [100.0, 2.5, 1.2])
    baseline = NonRepairable(surv.Weibull.from_params([100.0, 2.5]))
    s, H = 150.0, 400.0
    out = operating_schedule_value(model, [0.0, s], [0.0, 1.0], [1.0, 1.0], H)
    phi = math.exp(1.2)
    at_switch = baseline.failures(s).expected_failures
    at_end = baseline.failures(s + phi * (H - s)).expected_failures
    _close(out.stretches[0]["failures"], at_switch, 1e-6)
    _close(out.failures, at_end, 1e-6)
    # Instant replacement: always up.
    _close(out.up_time, H, 1e-12)


# -- 3. a constant schedule against a one-component RepairableRBD ---------


@pytest.mark.parametrize(
    "down_time",
    [surv.LogNormal.from_params([math.log(20.0), 0.6]), 15.0],
    ids=["lognormal", "fixed"],
)
def test_a_constant_schedule_is_a_one_component_repairable_rbd(down_time):
    # At z = 0.5 the AFT life is a Weibull of scale 1000 / exp(0.8 x 0.5).
    model = _model(surv.WeibullAFT, [1000.0, 1.8, 0.8])
    life = surv.Weibull.from_params([1000.0 / math.exp(0.4), 1.8])
    repair = (
        surv.ExactEventTime.from_params([down_time])
        if isinstance(down_time, float)
        else down_time
    )
    rbd = RepairableRBD(
        [("s", "u"), ("u", "t")],
        {
            "u": {
                "reliability": life,
                "repairability": repair,
                "repair_cost": 250.0,
            }
        },
    )
    H, r = 5000.0, 2e-4
    exact = rbd.expected_npv(H, discount_rate=r, value_rate=3.0)
    out = operating_schedule_value(
        model,
        [0.0],
        [0.5],
        [3.0],
        H,
        failure_cost=250.0,
        down_time=down_time,
        discount_rate=r,
    )
    _close(out.value, exact.value, 1e-6)
    _close(out.cost, exact.cost, 1e-5)
    _close(out.undiscounted_value, exact.undiscounted_value, 1e-6)
    _close(out.undiscounted_cost, exact.undiscounted_cost, 1e-5)
    _close(out.eac, exact.eac, 1e-5)
    _close(out.failures, rbd.expected_events(H).node_failures["u"], 1e-5)


# -- 4. a hand-written seeded simulation ----------------------------------


def _simulate(
    lives, starts, Z, rates, H, r, cost, down, age, n, seed
):  # noqa: C901
    """Each history: a unit put in at clock v lives along the schedule
    from v; its life is drawn by inverting its cumulative hazard
    ``sum over its own segments of phi(z) (H0(b) - H0(a))`` (proportional
    hazards, Weibull baseline). Returns per-history arrays of (failures,
    up time, failures PV, value PV)."""
    alpha, k, beta = lives
    rng = np.random.default_rng(seed)
    starts = np.asarray(starts, float)
    nxt = np.append(starts[1:], np.inf)
    phi = np.exp(beta * np.asarray(Z, float))
    H0 = lambda x: (x / alpha) ** k  # noqa: E731
    inv = lambda h: alpha * h ** (1.0 / k)  # noqa: E731

    def stretch(t):
        return int(np.searchsorted(starts, t, side="right") - 1)

    def earn(a, b):  # discounted value and time up over [a, b)
        value = up = 0.0
        while a < b:
            j = stretch(a)
            e = min(b, nxt[j])
            up += e - a
            if r:
                value += rates[j] * (math.exp(-r * a) - math.exp(-r * e)) / r
            else:
                value += rates[j] * (e - a)
            a = e
        return value, up

    out = np.zeros((n, 4))
    for i in range(n):
        clock, aged = 0.0, age
        while clock < H:
            # The life from clock, aged ``aged`` (at the first covariates).
            E = rng.exponential()
            x, t = aged, clock
            while True:
                j = stretch(t)
                L = nxt[j] - t
                gain = phi[j] * (H0(x + L) - H0(x))
                if E <= gain:
                    x_f = inv(H0(x) + E / phi[j])
                    t += x_f - x
                    break
                E -= gain
                x += L
                t = nxt[j]
            fail = t
            value, up = earn(clock, min(fail, H))
            out[i, 1] += up
            out[i, 3] += value
            if fail >= H:
                break
            out[i, 0] += 1
            out[i, 2] += math.exp(-r * fail)
            clock, aged = fail + down(rng), 0.0
    return out


@pytest.mark.parametrize(
    "case",
    [
        # Cool, then hot; a fixed down time; the unit in service 40 old.
        dict(
            starts=[0.0, 70.0],
            Z=[0.0, 1.0],
            rates=[1.0, 1.6],
            age=40.0,
            down=8.0,
            sim_down=lambda g: 8.0,
        ),
        # Hot, cool, hotter; log-normal down times.
        dict(
            starts=[0.0, 50.0, 150.0],
            Z=[1.0, 0.0, 1.5],
            rates=[2.0, 1.0, 2.5],
            age=0.0,
            down=surv.LogNormal.from_params([math.log(5.0), 0.5]),
            sim_down=lambda g: g.lognormal(math.log(5.0), 0.5),
        ),
    ],
    ids=["two-stretches-aged", "three-stretches-lognormal"],
)
def test_proportional_hazards_agree_with_a_seeded_simulation(case):
    # Weibull PH: scale 100, shape 2, hazard times exp(ln 3 z). Under
    # proportional hazards time does not rescale, so this tests the
    # shifted paths themselves.
    lives = (100.0, 2.0, math.log(3.0))
    model = _model(surv.WeibullPH, lives)
    H, r, cost = 400.0, 0.002, 30.0
    out = operating_schedule_value(
        model,
        case["starts"],
        case["Z"],
        case["rates"],
        H,
        failure_cost=cost,
        down_time=case["down"],
        discount_rate=r,
        age=case["age"],
    )
    sims = _simulate(
        lives,
        case["starts"],
        case["Z"],
        case["rates"],
        H,
        r,
        cost,
        case["sim_down"],
        case["age"],
        20_000,
        seed=11,
    )
    mean = sims.mean(axis=0)
    se = sims.std(axis=0, ddof=1) / math.sqrt(len(sims))
    numerical = [out.failures, out.up_time, out.cost / cost, out.value]
    for got, m, s in zip(numerical, mean, se):
        assert abs(got - m) < 4.0 * s, (got, m, s)


# -- 5. the switch time ---------------------------------------------------


def _story(cost):
    # Rate 0.01 at z = 0 (cool), 0.04 at z = 1 (hot).
    model = _model(surv.ExponentialAFT, [0.01, math.log(4.0)])
    return model, dict(failure_cost=cost, discount_rate=0.003)


def test_the_best_switch_time_is_the_best_of_a_fine_scan():
    # Weibull PH wear-out (scale 100, shape 3), hot (z = 1) multiplies
    # the hazard by 5 and earns 1.5 an hour to cool's 1. At 100 a failure
    # cool pays in the long run (1 - 100 / 89.3 a mean life, against 1.5 -
    # 100 / 52.2), but the first unit is young, and young units barely
    # fail: running it hot first, then cool, pays.
    model = _model(surv.WeibullPH, [100.0, 3.0, math.log(5.0)])
    kw = dict(failure_cost=100.0, discount_rate=0.002, down_time=5.0)
    H = 300.0
    best = best_switch_time(model, [1.0, 0.0], [1.5, 1.0], H, **kw)
    scan = np.linspace(0.0, 60.0, 31)
    npvs = []
    for s in scan:
        if s == 0.0:  # switching at once: cool throughout
            v = operating_schedule_value(
                model, [0.0], [0.0], [1.0], H, tolerance=1e-4, **kw
            )
        else:
            v = operating_schedule_value(
                model,
                [0.0, s],
                [1.0, 0.0],
                [1.5, 1.0],
                H,
                tolerance=1e-4,
                **kw,
            )
        npvs.append(v.npv)
    npvs = np.array(npvs)
    assert best.choice == "switch"
    # As good as any switch time scanned, to the valuations' tolerance
    # (1e-4 of the value and cost), and near the scan's best.
    noise = 1e-4 * (best.best.value + best.best.cost)
    assert best.npv >= npvs.max() - noise
    assert abs(best.switch_time - scan[np.argmax(npvs)]) <= 2.0
    assert best.switch_at_once_npv == pytest.approx(npvs[0], rel=1e-12)
    assert best.gain == pytest.approx(
        best.npv - max(best.switch_at_once_npv, best.never_switch_npv),
        rel=1e-12,
    )
    assert best.gain > 100 * noise
    assert best.best.npv == best.npv
    assert len(best.times) == len(best.npvs) == _operating._SCAN
    assert best.times[0] == 0.0 and best.times[-1] == H


@pytest.mark.parametrize(
    "cost, choice", [(20.0, "switch at once"), (40.0, "never switch")]
)
def test_an_end_can_be_best(cost, choice):
    # Exponential lives: hot earns 2 an hour at 0.04 failures an hour
    # (1.2 an hour net at 20 a failure, 0.4 at 40), cool 1 an hour at
    # 0.01 (0.8 and 0.6 net). Memoryless, so one end is best.
    model, kw = _story(cost)
    best = best_switch_time(model, [0.0, 1.0], [1.0, 2.0], 200.0, **kw)
    assert best.choice == choice
    assert best.gain == 0.0
    end = best.switch_at_once_npv if cost == 20.0 else best.never_switch_npv
    assert best.npv == pytest.approx(end, rel=1e-12)
    assert best.switch_time == (0.0 if cost == 20.0 else 200.0)


# -- 6. the unit of time and of the covariate -----------------------------


@pytest.mark.parametrize("c", [1e-6, 1e-3, 1.0, 1e3, 1e6])
def test_results_scale_with_the_unit_of_time(c):
    def value(c, zc=1.0):
        model = _model(surv.WeibullPH, [100.0 * c, 2.0, math.log(3.0) / zc])
        return operating_schedule_value(
            model,
            [0.0, 70.0 * c],
            [0.0, zc],
            [1.0 / c, 1.6 / c],
            400.0 * c,
            failure_cost=30.0,
            down_time=surv.LogNormal.from_params([math.log(5.0 * c), 0.5]),
            discount_rate=0.002 / c,
            age=40.0 * c,
            tolerance=1e-4,
        )

    base, scaled = value(1.0), value(c)
    assert scaled.npv == pytest.approx(base.npv, rel=1e-9)
    assert scaled.failures == pytest.approx(base.failures, rel=1e-9)
    assert scaled.up_time == pytest.approx(base.up_time * c, rel=1e-9)
    assert scaled.cells == base.cells
    # A covariate in other units (z times 50, its coefficient over 50).
    assert value(1.0, zc=50.0).npv == pytest.approx(base.npv, rel=1e-9)


# -- 7. inputs, results and help ------------------------------------------


@pytest.fixture
def aft():
    return _model(surv.WeibullAFT, [100.0, 2.0, 1.0])


@pytest.mark.parametrize(
    "args, kwargs, error, words",
    [
        (([1.0], [0.0], [1.0], 100.0), {}, ValueError, "first 0"),
        (([0.0, 0.0], [0, 1], [1, 1], 100.0), {}, ValueError, "increasing"),
        (([0.0, 5.0], [0.0], [1.0, 1.0], 100.0), {}, ValueError, "one row"),
        (([0.0], [[0.0, 1.0]], [1.0], 100.0), {}, ValueError, "fitted with"),
        (([0.0], [0.0], [-1.0], 100.0), {}, ValueError, "value_rates"),
        (([0.0], [0.0], [1.0, 2.0], 100.0), {}, ValueError, "value_rates"),
        (([0.0], [0.0], [1.0], 0.0), {}, ValueError, "horizon"),
        (([0.0], [0.0], [1.0], "100"), {}, ValueError, "horizon"),
        (("0", [0.0], [1.0], 100.0), {}, TypeError, "times"),
        (
            ([0.0], [0.0], [1.0], 100.0),
            {"failure_cost": -1},
            ValueError,
            "failure_cost",
        ),
        (
            ([0.0], [0.0], [1.0], 100.0),
            {"down_time": -1.0},
            ValueError,
            "down_time",
        ),
        (
            ([0.0], [0.0], [1.0], 100.0),
            {"down_time": surv.Weibull},
            TypeError,
            "distribution itself",
        ),
        (
            ([0.0], [0.0], [1.0], 100.0),
            {"down_time": "fast"},
            ValueError,
            "down_time",
        ),
        (
            ([0.0], [0.0], [1.0], 100.0),
            {"discount_rate": -0.1},
            ValueError,
            "discount_rate",
        ),
        (([0.0], [0.0], [1.0], 100.0), {"age": True}, ValueError, "age"),
        (
            ([0.0], [0.0], [1.0], 100.0),
            {"tolerance": 0.5},
            ValueError,
            "tolerance",
        ),
    ],
)
def test_inputs_are_refused_with_their_reason(aft, args, kwargs, error, words):
    with pytest.raises(error, match=words):
        operating_schedule_value(aft, *args, **kwargs)


def test_a_down_time_that_may_never_end_is_refused(aft):
    endless = surv.Weibull.from_params([10.0, 1.0], lfp_p=0.5)
    with pytest.raises(ValueError, match="never end"):
        operating_schedule_value(
            aft, [0.0], [0.0], [1.0], 100.0, down_time=endless
        )


def test_models_without_a_proper_curve_are_refused():
    rng = np.random.default_rng(0)
    Z = rng.uniform(0, 1, (50, 1))
    x = rng.weibull(2.0, 50) * 100.0
    cox = surv.CoxPH.fit(x, Z=Z)
    with pytest.raises(ValueError, match="semiparametric"):
        operating_schedule_value(cox, [0.0], [0.0], [1.0], 100.0)
    with pytest.raises(TypeError, match="distribution itself"):
        operating_schedule_value(surv.Weibull, [0.0], [0.0], [1.0], 100.0)
    with pytest.raises(ValueError, match="regression model"):
        operating_schedule_value(
            surv.Weibull.from_params([10.0, 2.0]), [0.0], [0.0], [1.0], 10.0
        )


class _DeadOnArrival:
    """A stand-in model half of whose units are dead on arrival: a
    cumulative hazard of ln 2 from age 0."""

    phi_param_map = {"beta_0": 0}  # one covariate

    def Hf_tvc(self, x, schedule):
        return np.full(np.shape(x), math.log(2.0))


def test_a_dead_on_arrival_life_is_refused():
    with pytest.raises(ValueError, match="proper curve.*0.5"):
        operating_schedule_value(_DeadOnArrival(), [0.0], [0.0], [1.0], 100.0)


class _Stub:
    """A stand-in model with the cumulative hazard ``hazard(x)``."""

    def __init__(self, hazard):
        self.hazard = hazard

    phi_param_map = {"beta_0": 0}  # one covariate

    def Hf_tvc(self, x, schedule):
        return self.hazard(np.asarray(x, dtype=float))


def test_a_life_undefined_at_some_ages_is_refused():
    stub = _Stub(lambda x: np.where(x > 1.0, np.nan, x))
    with pytest.raises(ValueError, match="not defined at some ages"):
        operating_schedule_value(stub, [0.0], [0.0], [1.0], 100.0)


def test_a_life_some_units_survive_is_valued():
    # H = min(x, 1/2): a third of the units never fail (exp(-1/2) of a
    # unit survives for ever), so the life has no upper quartile and the
    # grid is sized from its lower one. With instant replacement the
    # failures by t are a renewal count of a defective life: at most
    # 1 / exp(-1/2) - 1 = 0.6487 in all.
    stub = _Stub(lambda x: np.minimum(x, 0.5))
    out = operating_schedule_value(stub, [0.0], [0.0], [1.0], 50.0)
    assert out.failures == pytest.approx(math.exp(0.5) - 1.0, rel=1e-6)
    assert out.up_time == pytest.approx(50.0, rel=1e-12)


def test_switch_time_inputs_are_two_rows(aft):
    with pytest.raises(ValueError, match="2 stretch"):
        best_switch_time(aft, [0.0, 1.0, 2.0], [1.0, 1.0], 100.0)
    with pytest.raises(ValueError, match="value_rates"):
        best_switch_time(aft, [0.0, 1.0], [1.0], 100.0)


def test_a_grid_that_cannot_reach_the_tolerance_warns(aft, monkeypatch):
    monkeypatch.setattr(_operating, "_MAX_CELLS", 200)
    with pytest.warns(RuntimeWarning, match="did not reach the tolerance"):
        out = operating_schedule_value(
            aft, [0.0, 30.0], [0.0, 1.0], [1.0, 1.0], 1000.0, down_time=3.0
        )
    assert out.error > out.tolerance


def test_stretches_past_the_horizon_have_zeros(aft):
    out = operating_schedule_value(
        aft,
        [0.0, 50.0, 500.0],
        [0.0, 1.0, 0.0],
        [1.0, 2.0, 1.0],
        200.0,
        failure_cost=5.0,
    )
    rows = out.stretches
    assert [(r["start"], r["end"]) for r in rows] == [
        (0.0, 50.0),
        (50.0, 200.0),
        (500.0, 200.0),
    ]
    assert rows[2]["failures"] == rows[2]["value"] == 0.0
    assert sum(r["value"] for r in rows) == pytest.approx(out.value)
    assert sum(r["cost"] for r in rows) == pytest.approx(out.cost)
    assert sum(r["failures"] for r in rows) == pytest.approx(out.failures)
    assert sum(r["up_time"] for r in rows) == pytest.approx(out.up_time)


def test_results_go_through_json(aft):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = operating_schedule_value(
            aft, [0.0, 50.0], [0.0, 1.0], [1.0, 2.0], 200.0
        )
    assert isinstance(out, ScheduleValue)
    again = json.loads(json.dumps(out.to_dict()))
    assert again["npv"] == out.npv and len(again["stretches"]) == 2
    best = best_switch_time(aft, [0.0, 1.0], [1.0, 2.0], 200.0)
    assert isinstance(best, SwitchTime)
    again = json.loads(json.dumps(best.to_dict()))
    assert again["best"]["npv"] == best.npv


def test_help_reads():
    for thing in (operating_schedule_value, best_switch_time, ScheduleValue):
        text = pydoc.render_doc(thing, renderer=pydoc.plaintext)
        assert "][repyability." not in text
