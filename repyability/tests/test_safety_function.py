"""A safety function's results (``SafetyFunction``): the PFDavg, the PFH,
the spurious-trip rate and the SIL band, against closed forms, against a
multi-phase Markov chain of the channels written out here, against IEC
61508-6 Annex B's simplified formulas to the order they hold, and against
the simulation of both diagrams."""

import inspect
import itertools
import json
import math

import numpy as np
import pytest
import surpyval as surv
from scipy.linalg import expm

from repyability import (
    RepairableRBD,
    SafetyChannel,
    SafetyFunction,
    SafetyFunctionResult,
    SILBand,
    sil_band,
)
from repyability.rbd import routes
from repyability.tests.catalogue import safety_function_kinds
from repyability.tests.test_analysis_routes import behaves_as_reported

T1 = 8760.0
MTTR = 8.0


def channel(du=2e-6, dd=3e-6, safe=5e-6, **options):
    options.setdefault("repair", MTTR)
    options.setdefault("proof_test_interval", T1)
    return SafetyChannel(du, dangerous_detected=dd, safe=safe, **options)


def vote(k, n, unit=None, **options):
    unit = channel() if unit is None else unit
    return SafetyFunction({f"c{i}": unit for i in range(n)}, k, **options)


# -- the channel ------------------------------------------------------------


def test_a_rate_is_split_by_kind():
    unit = SafetyChannel.from_rate(
        1e-5,
        dangerous_fraction=0.4,
        diagnostic_coverage=0.75,
        repair=8.0,
        proof_test_interval=T1,
    )
    assert unit.dangerous_undetected == pytest.approx(1e-6, rel=1e-14)
    assert unit.dangerous_detected == pytest.approx(3e-6, rel=1e-14)
    assert unit.safe == pytest.approx(6e-6, rel=1e-14)
    assert unit.diagnostic_coverage == pytest.approx(0.75, rel=1e-14)
    # (safe + detected) / total = (6 + 3) / 10
    assert unit.safe_failure_fraction == pytest.approx(0.9, rel=1e-14)


def test_the_fractions_of_a_life_with_no_constant_rate_are_nan():
    unit = channel(du=surv.Weibull.from_params([5e5, 2.0]))
    assert unit.undetected_rate is None
    assert math.isnan(unit.safe_failure_fraction)
    assert math.isnan(unit.diagnostic_coverage)
    exponential = channel(du=surv.Exponential.from_params([2e-6]))
    assert exponential.undetected_rate == pytest.approx(2e-6, rel=1e-9)
    assert exponential.safe_failure_fraction == pytest.approx(0.8, rel=1e-9)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"dangerous_undetected": -1e-6}, "dangerous_undetected must be"),
        ({"dangerous_undetected": "2e-6"}, "dangerous_undetected must be"),
        ({"dangerous_undetected": True}, "dangerous_undetected must be"),
        ({"dangerous_detected": -1.0}, "dangerous_detected must be"),
        ({"safe": math.inf}, "safe must be"),
        ({"repair": None}, "needs its repair"),
        ({"repair": -1.0}, "repair must be a mean time"),
        ({"repair": "fast"}, "repair must be a mean time"),
        ({"undetected_repair": -1.0}, "undetected_repair must be"),
        ({"proof_test_interval": None}, "needs its proof_test_interval"),
        ({"proof_test_interval": 0.0}, "needs its proof_test_interval"),
        ({"proof_test_coverage": 1.5}, "proof_test_coverage must be"),
        ({"proof_test_offset": T1}, "proof_test_offset must be"),
    ],
)
def test_a_channel_refuses_what_it_cannot_be(options, message):
    given = {
        "dangerous_undetected": 2e-6,
        "repair": 8.0,
        "proof_test_interval": T1,
        **options,
    }
    with pytest.raises(ValueError, match=message):
        SafetyChannel(**given)


def test_a_distribution_is_not_a_model():
    with pytest.raises(TypeError, match="Weibull distribution itself"):
        channel(du=surv.Weibull)


def test_a_function_refuses_what_it_cannot_be():
    unit = channel()
    with pytest.raises(ValueError, match="at least one channel"):
        SafetyFunction({}, 1)
    with pytest.raises(ValueError, match="must be a SafetyChannel"):
        SafetyFunction({"a": 2e-6}, 1)
    with pytest.raises(ValueError, match="k must be a whole number"):
        SafetyFunction({"a": unit}, 0)
    with pytest.raises(ValueError, match="at most the number of channels"):
        SafetyFunction({"a": unit}, 2)
    with pytest.raises(ValueError, match=r"beta must be a number in \[0, 1\]"):
        SafetyFunction({"a": unit, "b": unit}, 1, beta=1.5)
    with pytest.raises(ValueError, match="detected_trips must be"):
        SafetyFunction({"a": unit}, 1, detected_trips="yes")
    with pytest.raises(ValueError, match="needs the vote's channels alike"):
        SafetyFunction({"a": unit, "b": channel(du=3e-6)}, 1, beta=0.1)
    with pytest.raises(ValueError, match="mode must be"):
        vote(1, 1).results("demand")
    with pytest.raises(ValueError, match="undetected_only must be"):
        vote(1, 1).pfh(undetected_only=1)


# -- the SIL bands ----------------------------------------------------------

PFD_EDGES = {
    4: (1e-5, 1e-4),
    3: (1e-4, 1e-3),
    2: (1e-3, 1e-2),
    1: (1e-2, 1e-1),
}
PFH_EDGES = {
    4: (1e-9, 1e-8),
    3: (1e-8, 1e-7),
    2: (1e-7, 1e-6),
    1: (1e-6, 1e-5),
}


@pytest.mark.parametrize(
    "mode, edges",
    [
        ("low_demand", PFD_EDGES),
        ("high_demand", PFH_EDGES),
        ("continuous", PFH_EDGES),
    ],
)
def test_a_value_on_an_edge_and_each_side_of_it(mode, edges):
    for level, (lower, upper) in edges.items():
        # The table's band is "at least lower, less than upper".
        assert sil_band(lower, mode).sil == level
        assert sil_band(np.nextafter(lower, 1.0), mode).sil == level
        assert sil_band(np.nextafter(upper, 0.0), mode).sil == level
        assert sil_band(upper, mode).sil == level - 1
        if level < 4:
            assert sil_band(np.nextafter(lower, 0.0), mode).sil == level + 1
        band = sil_band(math.sqrt(lower * upper), mode)
        assert (band.sil, band.lower, band.upper) == (level, lower, upper)
        assert band.margin_to_lower == pytest.approx(math.sqrt(10), rel=1e-12)
        assert band.margin_to_upper == pytest.approx(math.sqrt(10), rel=1e-12)


def test_the_tables_stop_at_sil_4_and_at_no_sil():
    below = sil_band(2e-6, "low_demand")
    assert (below.sil, below.lower, below.upper) == (4, 1e-5, 1e-4)
    assert below.margin_to_lower == pytest.approx(0.2, rel=1e-12)
    none = sil_band(0.3, "low_demand")
    assert (none.sil, none.lower, none.upper) == (0, 1e-1, math.inf)
    assert none.margin_to_upper == math.inf
    zero = sil_band(0.0, "high_demand")
    assert zero.sil == 4 and zero.margin_to_upper == math.inf


def test_the_risk_reduction_factor_is_the_pfd_s_alone():
    assert sil_band(4e-3, "low_demand").risk_reduction_factor == 250.0
    assert sil_band(4e-8, "high_demand").risk_reduction_factor is None
    assert sil_band(4e-3, "low_demand").measure == "pfd_avg"
    assert sil_band(4e-8, "continuous").measure == "pfh"


def test_a_pfh_is_judged_per_hour():
    per_year = sil_band(5e-8 * 8760.0, "high_demand", hours_per_unit=8760.0)
    assert per_year.sil == 3
    assert per_year.value == pytest.approx(5e-8, rel=1e-12)
    with pytest.raises(ValueError, match="hours_per_unit must be"):
        sil_band(1e-8, "high_demand", hours_per_unit=0.0)
    with pytest.raises(ValueError, match="mode must be"):
        sil_band(1e-3, "low")
    with pytest.raises(ValueError, match="value must be"):
        sil_band(-1e-3, "low_demand")


# -- one channel: closed forms ---------------------------------------------


def test_1oo1_against_the_tested_unit_s_closed_form():
    # Undetected failures alone, repaired in no time at the test: the unit
    # is down from its failure to the next test, so over a period
    # PFDavg = 1 - (1 - exp(-x)) / x, x = lambda T, and it fails once
    # in a period with probability 1 - exp(-x).
    lam = 2e-6
    function = vote(1, 1, channel(du=lam, dd=0.0, safe=0.0, repair=0.0))
    x = lam * T1
    exact = -math.expm1(-x)
    assert function.pfd_avg() == pytest.approx(1.0 - exact / x, rel=1e-9)
    assert function.pfh() == pytest.approx(exact / T1, rel=1e-9)
    assert function.pfh(undetected_only=True) == function.pfh()
    assert function.spurious_trip_rate() == 0.0
    assert function.safe is None


def fixed_repair_pfd(lam, interval, repair):
    """The PFDavg of a unit of hidden failures at rate ``lam``, tested
    every ``interval``, whose repair starts at the test that finds the
    failure and takes ``repair`` (less than the interval): at a test the
    unit is up (and as new, its rate being constant) or found failed, a
    two-state chain from test to test."""
    whole = -math.expm1(-lam * interval)
    part = -math.expm1(-lam * (interval - repair))
    found = whole / (1.0 + whole - part)
    down_up = interval - whole / lam
    down_found = interval - part / lam
    return ((1.0 - found) * down_up + found * down_found) / interval


def test_1oo1_with_a_repair_that_takes_time():
    lam, repair = 2e-5, 72.0
    unit = channel(
        du=lam,
        dd=0.0,
        safe=0.0,
        repair=surv.ExactEventTime.from_params([repair]),
    )
    function = vote(1, 1, unit)
    exact = fixed_repair_pfd(lam, T1, repair)
    assert function.pfd_avg() == pytest.approx(exact, rel=1e-6)
    # IEC 61508-6's lambda_DU (T1/2 + MRT) is its first-order term.
    simplified = function.simplified_formulas().pfd_avg
    assert simplified == pytest.approx(lam * (T1 / 2 + repair), rel=1e-14)
    assert simplified > exact


def test_1oo1_with_detected_failures_too():
    # The detected failures are a revealed unit in series: down
    # lambda m / (1 + lambda m) of the time, independently.
    lam, detected = 2e-6, 3e-5
    function = vote(
        1, 1, channel(du=lam, dd=detected, undetected_repair="instant")
    )
    x = lam * T1
    hidden = 1.0 - -math.expm1(-x) / x
    revealed = detected * MTTR / (1.0 + detected * MTTR)
    assert function.pfd_avg() == pytest.approx(
        hidden + revealed - hidden * revealed, rel=1e-9
    )
    # Each fails while the other is up.
    frequency = (-math.expm1(-x) / T1) * (1.0 - revealed) + (
        detected / (1.0 + detected * MTTR)
    ) * (1.0 - hidden)
    assert function.pfh() == pytest.approx(frequency, rel=1e-9)
    assert function.pfh(undetected_only=True) == pytest.approx(
        (-math.expm1(-x) / T1) * (1.0 - revealed), rel=1e-9
    )


@pytest.mark.parametrize("lam", [2e-5, 2e-6, 2e-7])
def test_the_1oo1_formula_holds_to_first_order(lam):
    # lambda_DU T1 / 2 against 1 - (1 - exp(-x)) / x = x/2 - x^2/6 + ...:
    # the formula is high by x / 3 of the value, to second order in x.
    function = vote(1, 1, channel(du=lam, dd=0.0, safe=0.0, repair=0.0))
    x = lam * T1
    gap = function.simplified_formulas().pfd_avg / function.pfd_avg() - 1.0
    assert gap == pytest.approx(x / 3.0, abs=x * x)
    assert gap > 0.0


def test_a_fitted_life_for_the_undetected_failures():
    life = surv.Weibull.from_params([2e5, 1.8])
    function = vote(1, 1, channel(du=life, dd=0.0, safe=0.0, repair="instant"))
    # Renewed only when found failed: followed by the tested unit's own
    # model, which the diagram of one such component gives.
    plain = RepairableRBD(
        [("s", "a"), ("a", "t")],
        {
            "a": {
                "reliability": life,
                "repairability": "instant",
                "inspection": {"interval": T1},
            }
        },
    )
    assert function.pfd_avg() == plain.mean_unavailability()
    assert function.pfh() == plain.system_failure_frequency()
    with pytest.raises(NotImplementedError, match="constant failure rates"):
        function.simplified_formulas()


def test_staggered_tests_against_their_closed_form():
    # 1oo2, repairs in no time, the second channel tested half a period
    # after the first: on [0, h), h = T/2, the channels were last tested
    # t and t + h ago, and the second half-period mirrors the first.
    lam = 2e-5
    first = channel(du=lam, dd=0.0, safe=0.0, repair=0.0)
    second = channel(
        du=lam, dd=0.0, safe=0.0, repair=0.0, proof_test_offset=T1 / 2
    )
    function = SafetyFunction({"a": first, "b": second}, 1)
    h = T1 / 2
    a = math.exp(-lam * h)
    integral = (
        h
        - (1.0 + a) * -math.expm1(-lam * h) / lam
        + a * -math.expm1(-2.0 * lam * h) / (2.0 * lam)
    )
    assert function.pfd_avg() == pytest.approx(integral / h, rel=1e-9)
    together = vote(1, 2, first).pfd_avg()
    x = lam * T1
    assert together == pytest.approx(
        1.0 - 2.0 * -math.expm1(-x) / x + -math.expm1(-2.0 * x) / (2.0 * x),
        rel=1e-9,
    )
    # Staggering takes the PFDavg from x^2/3 towards 5 x^2/24.
    assert integral / h < together


# -- voted channels: a multi-phase Markov chain -----------------------------


def integrated(generator, start, interval):
    """``start @ integral_0^interval expm(generator s) ds``, and the
    distribution at the end, from one exponential of the augmented
    matrix."""
    size = len(generator)
    augmented = np.zeros((2 * size, 2 * size))
    augmented[:size, :size] = generator
    augmented[:size, size:] = np.eye(size)
    whole = expm(augmented * interval)
    return start @ whole[:size, size:], start @ whole[:size, :size]


def dangerous_chain(
    n, k, du, dd, mttr, mrt, interval, beta=0.0, beta_detected=0.0
):
    """The function's PFDavg and PFH (all of it, and the part an
    undetected failure completes) from the chain of its channels, written
    from the rules: each channel's undetected failures are hidden (state
    1) until the test, which starts their repair (state 2, over at rate 1
    / mrt, or at once), and its detected failures are repaired at rate 1 /
    mttr; a shared cause fails every channel that is up in its kind. The
    chain runs from test to test, its states mapped at each."""
    kinds = [(u, d) for u in range(3 if mrt > 0 else 2) for d in range(2)]
    states = list(itertools.product(kinds, repeat=n))
    index = {state: i for i, state in enumerate(states)}
    size = len(states)
    undetected = np.zeros((size, size))
    others = np.zeros((size, size))

    def changed(state, which, part, value):
        new = list(state)
        for c in which:
            pair = list(new[c])
            pair[part] = value
            new[c] = tuple(pair)
        return index[tuple(new)]

    shared = n > 1
    own_du = du * (1.0 - beta) if shared else du
    own_dd = dd * (1.0 - beta_detected) if shared else dd
    for state, i in index.items():
        up_u = [c for c in range(n) if state[c][0] == 0]
        up_d = [c for c in range(n) if state[c][1] == 0]
        for c in up_u:
            undetected[i, changed(state, [c], 0, 1)] += own_du
        for c in up_d:
            others[i, changed(state, [c], 1, 1)] += own_dd
        if shared and beta > 0 and up_u:
            undetected[i, changed(state, up_u, 0, 1)] += beta * du
        if shared and beta_detected > 0 and up_d:
            others[i, changed(state, up_d, 1, 1)] += beta_detected * dd
        for c in range(n):
            if state[c][0] == 2:
                others[i, changed(state, [c], 0, 0)] += 1.0 / mrt
            if state[c][1] == 1:
                others[i, changed(state, [c], 1, 0)] += 1.0 / mttr
    generator = undetected + others
    generator -= np.diag(generator.sum(axis=1))
    tested = np.zeros((size, size))
    for state, i in index.items():
        found = [c for c in range(n) if state[c][0] == 1]
        tested[i, changed(state, found, 0, 2 if mrt > 0 else 0)] = 1.0
    down = np.array(
        [sum(pair == (0, 0) for pair in state) < k for state in states]
    )
    period = expm(generator * interval) @ tested
    # The distribution just after a test, in the long run.
    system = np.vstack([(period - np.eye(size)).T, np.ones(size)])
    start = np.linalg.lstsq(
        system, np.append(np.zeros(size), 1.0), rcond=None
    )[0]
    time_in, _ = integrated(generator, start, interval)

    def into_down(rates):
        return rates[:, down].sum(axis=1) * ~down

    return (
        time_in @ down / interval,
        time_in @ into_down(undetected + others) / interval,
        time_in @ into_down(undetected) / interval,
    )


def safe_chain(n, k, rate, mttr, beta=0.0):
    """The long-run frequency at which ``k`` of ``n`` channels come to be
    failed safe together, from their chain's steady state."""
    states = list(itertools.product(range(2), repeat=n))
    index = {state: i for i, state in enumerate(states)}
    size = len(states)
    generator = np.zeros((size, size))
    shared = n > 1 and beta > 0
    for state, i in index.items():
        up = [c for c in range(n) if state[c] == 0]
        for c in range(n):
            new = list(state)
            new[c] = 1 - state[c]
            generator[i, index[tuple(new)]] += (
                rate * (1.0 - beta if shared else 1.0)
                if state[c] == 0
                else 1.0 / mttr
            )
        if shared and up:
            generator[i, index[(1,) * n]] += beta * rate
    rates = generator.copy()
    generator -= np.diag(generator.sum(axis=1))
    system = np.vstack([generator.T, np.ones(size)])
    steady = np.linalg.lstsq(
        system, np.append(np.zeros(size), 1.0), rcond=None
    )[0]
    tripped = np.array([sum(state) >= k for state in states])
    return steady @ (rates[:, tripped].sum(axis=1) * ~tripped)


VOTES = [(1, 2), (2, 2), (2, 3)]
RATES = {"du": 2e-5, "dd": 3e-5, "safe": 5e-5}


@pytest.mark.parametrize("k, n", VOTES)
@pytest.mark.parametrize(
    "beta, beta_detected", [(0.0, 0.0), (0.1, 0.05), (1.0, 1.0)]
)
@pytest.mark.parametrize("mrt", [0.0, 24.0])
def test_votes_against_the_chain_of_their_channels(
    k, n, beta, beta_detected, mrt
):
    unit = channel(**RATES, undetected_repair=mrt)
    function = vote(k, n, unit, beta=beta, beta_detected=beta_detected)
    pfd, pfh, undetected = dangerous_chain(
        n, k, RATES["du"], RATES["dd"], MTTR, mrt, T1, beta, beta_detected
    )
    # Without a shared cause and with a repair that takes time after the
    # test, the values are averaged over the tested units' own grids, each
    # unit failing at its rate while it is up: 1.3e-9 off at most over
    # these cases, as through the groups' chain.
    assert function.pfd_avg() == pytest.approx(pfd, rel=1e-8)
    assert function.pfh() == pytest.approx(pfh, rel=1e-8)
    assert function.pfh(undetected_only=True) == pytest.approx(
        undetected, rel=1e-8
    )
    if beta == 0.0:
        assert not function.dangerous.ccf_groups


@pytest.mark.parametrize("k, n", [(1, 1)] + VOTES + [(1, 3)])
@pytest.mark.parametrize("beta", [0.0, 0.1, 1.0])
def test_spurious_trips_against_the_chain(k, n, beta):
    function = vote(k, n, channel(**RATES), beta_safe=beta)
    exact = safe_chain(n, k, RATES["safe"], MTTR, beta)
    assert function.spurious_trip_rate() == pytest.approx(exact, rel=1e-9)


def test_spurious_trips_to_first_order():
    # 1oo2 trips on either channel's safe failure, 2oo2 and 2oo3 on a
    # second while a first is under repair: 2 lambda, 2 lambda^2 MTTR and
    # 6 lambda^2 MTTR, each within a factor 1 + O(lambda MTTR).
    lam = RATES["safe"]
    small = lam * MTTR
    unit = channel(**RATES)
    assert vote(1, 2, unit).spurious_trip_rate() == pytest.approx(
        2 * lam, rel=2 * small
    )
    assert vote(2, 2, unit).spurious_trip_rate() == pytest.approx(
        2 * lam**2 * MTTR, rel=4 * small
    )
    assert vote(2, 3, unit).spurious_trip_rate() == pytest.approx(
        6 * lam**2 * MTTR, rel=4 * small
    )
    for k, n in [(1, 2), (2, 2), (2, 3)]:
        function = vote(k, n, unit)
        assert function.simplified_formulas().spurious_trip_rate == (
            pytest.approx(function.spurious_trip_rate(), rel=4 * small)
        )


def test_a_detected_failure_that_trips_is_a_safe_one():
    unit = channel(**RATES)
    tripping = vote(2, 3, unit, detected_trips=True)
    alone = vote(2, 3, channel(du=RATES["du"], dd=0.0, safe=0.0))
    assert tripping.pfd_avg() == alone.pfd_avg()
    assert tripping.pfh() == alone.pfh()
    assert tripping.pfh(undetected_only=True) == tripping.pfh()
    assert tripping.spurious_trip_rate() == pytest.approx(
        safe_chain(3, 2, RATES["safe"] + RATES["dd"], MTTR), rel=1e-9
    )
    assert all(kind != "DD" for _, kind in tripping.dangerous.components)


# -- common cause -----------------------------------------------------------


@pytest.mark.parametrize("k, n", [(1, 2), (2, 3), (1, 3)])
def test_beta_of_1_is_one_channel(k, n):
    # Every undetected failure shared, and found failures repaired in no
    # time. (With repairs that take time the channels fail together but
    # are restored one by one, so a 1oo2 vote is back at the first
    # repair, and is not one channel: the chain above has those cases.)
    unit = channel(du=RATES["du"], dd=0.0, safe=0.0, repair="instant")
    one = vote(1, 1, unit)
    function = vote(k, n, unit, beta=1.0)
    assert function.pfd_avg() == pytest.approx(one.pfd_avg(), rel=1e-9)
    assert function.pfh() == pytest.approx(one.pfh(), rel=1e-9)


def test_beta_of_0_is_no_group():
    function = vote(2, 3, beta=0.0, beta_detected=0.0, beta_safe=0.0)
    assert not function.dangerous.ccf_groups
    assert not function.safe.ccf_groups
    assert function.pfd_breakdown().common_cause == 0.0


def test_the_breakdown_adds_up_to_the_pfd():
    function = vote(1, 2, beta=0.1, beta_detected=0.05)
    breakdown = function.pfd_breakdown()
    assert breakdown.pfd_avg == function.pfd_avg()
    shares = sum(breakdown.components.values()) + sum(
        breakdown.groups.values()
    )
    assert shares == pytest.approx(breakdown.pfd_avg, rel=1e-12)
    assert (("c0", "DU"), ("c1", "DU")) in breakdown.groups


# -- the simplified formulas ------------------------------------------------


def annex_b(k, n, du, dd, mttr, mrt, interval, beta=0.0, beta_d=0.0):
    """IEC 61508-6 Annex B's formulas, each written out as the annex
    gives it (B.3.2.2 and B.3.3.2)."""
    lam = du + dd
    tce = du / lam * (interval / 2 + mrt) + dd / lam * mttr
    tge = du / lam * (interval / 3 + mrt) + dd / lam * mttr
    tg2e = du / lam * (interval / 4 + mrt) + dd / lam * mttr
    own = (1 - beta_d) * dd + (1 - beta) * du
    common = beta_d * dd * mttr + beta * du * (interval / 2 + mrt)
    if (k, n) == (1, 1):
        return lam * tce, du
    if (k, n) == (2, 2):
        return 2 * lam * tce, 2 * du
    if (k, n) == (1, 2):
        return (
            2 * own**2 * tce * tge + common,
            2 * own * (1 - beta) * du * tce + beta * du,
        )
    if (k, n) == (2, 3):
        return (
            6 * own**2 * tce * tge + common,
            6 * own * (1 - beta) * du * tce + beta * du,
        )
    assert (k, n) == (1, 3)
    return (
        6 * own**3 * tce * tge * tg2e + common,
        6 * own**2 * (1 - beta) * du * tce * tge + beta * du,
    )


@pytest.mark.parametrize("k, n", [(1, 1), (1, 2), (2, 2), (2, 3), (1, 3)])
@pytest.mark.parametrize("beta, beta_d", [(0.0, 0.0), (0.1, 0.05)])
def test_the_simplified_formulas_are_annex_b_s(k, n, beta, beta_d):
    unit = channel(**RATES, undetected_repair=24.0)
    function = vote(k, n, unit, beta=beta, beta_detected=beta_d)
    pfd, pfh = annex_b(
        k, n, RATES["du"], RATES["dd"], MTTR, 24.0, T1, beta, beta_d
    )
    simplified = function.simplified_formulas()
    assert simplified.pfd_avg == pytest.approx(pfd, rel=1e-13)
    assert simplified.pfh == pytest.approx(pfh, rel=1e-13)
    assert len(simplified.down_times) == n - k + 1


@pytest.mark.parametrize("k, n", [(1, 2), (2, 2), (2, 3), (1, 3)])
def test_the_formulas_hold_to_first_order_in_the_rate(k, n):
    # Undetected failures alone, repaired in no time: the formulas are
    # the leading terms of the exact values in x = lambda T1, so their
    # relative gap falls tenfold with the rate, and they err high.
    gaps = []
    for lam in (2e-6, 2e-7):
        function = vote(k, n, channel(du=lam, dd=0.0, safe=0.0, repair=0.0))
        simplified = function.simplified_formulas()
        gaps.append(
            (
                simplified.pfd_avg / function.pfd_avg() - 1.0,
                simplified.pfh / function.pfh() - 1.0,
            )
        )
    for coarse, fine in zip(*gaps):
        assert 0.0 < fine < coarse < 5 * 2e-6 * T1
        assert coarse / fine == pytest.approx(10.0, rel=0.02)


def test_the_formulas_are_refused_where_the_annex_has_none():
    with pytest.raises(NotImplementedError, match="no simplified formula"):
        vote(2, 4).simplified_formulas()
    with pytest.raises(NotImplementedError, match="channels alike"):
        SafetyFunction(
            {"a": channel(), "b": channel(du=3e-6)}, 1
        ).simplified_formulas()
    with pytest.raises(NotImplementedError, match="finds every failure"):
        vote(
            1, 2, channel(proof_test_coverage=0.9, full_test=5 * T1)
        ).simplified_formulas()
    with pytest.raises(NotImplementedError, match="several in series"):
        SafetyFunction.series(
            SafetyFunction({"a": channel()}, 1),
            SafetyFunction({"b": channel()}, 1),
        ).simplified_formulas()


# -- a proof test that can miss ---------------------------------------------


def test_a_proof_test_that_can_miss():
    # Coverage c, everything found at every fifth test: a failure waits
    # for the next test that finds it, so the PFDavg is that of the same
    # unit drawn as a diagram by hand.
    lam, coverage = 2e-6, 0.9
    unit = channel(
        du=lam,
        dd=0.0,
        safe=0.0,
        repair=0.0,
        proof_test_coverage=coverage,
        full_test=5 * T1,
    )
    function = vote(1, 1, unit)
    # To first order: the covered part waits T1/2 and the rest 5 T1/2.
    first_order = lam * (coverage * T1 / 2 + (1 - coverage) * 5 * T1 / 2)
    assert function.pfd_avg() == pytest.approx(first_order, rel=lam * 5 * T1)
    assert (
        function.pfd_avg()
        > vote(1, 1, channel(du=lam, dd=0.0, safe=0.0, repair=0.0)).pfd_avg()
    )


# -- votes in series --------------------------------------------------------


def test_votes_in_series_tested_together():
    # Two single channels, both tested at the same times and repaired in
    # no time: the function is up while neither has failed since the
    # test, which is one unit at the sum of the rates.
    a, b = 2e-5, 5e-5
    function = SafetyFunction.series(
        SafetyFunction({"a": channel(du=a, dd=0.0, repair=0.0)}, 1),
        SafetyFunction({"b": channel(du=b, dd=0.0, repair=0.0)}, 1),
    )
    x = (a + b) * T1
    assert function.pfd_avg() == pytest.approx(
        1.0 - -math.expm1(-x) / x, rel=1e-9
    )
    assert function.pfh() == pytest.approx(-math.expm1(-x) / T1, rel=1e-9)
    # Either's safe failure trips: each while the other is not tripped.
    lam = 5e-6
    up = 1.0 / (1.0 + lam * 0.0)
    assert function.spurious_trip_rate() == pytest.approx(
        2 * lam * up, rel=1e-9
    )
    assert repr(function) == "SafetyFunction(1oo1, 1oo1; channels=['a', 'b'])"


def test_votes_in_series_are_one_function():
    sensors = vote(2, 3, beta=0.1)
    valves = SafetyFunction(
        {"v1": channel(du=5e-6), "v2": channel(du=5e-6)}, 1, beta=0.05
    )
    function = SafetyFunction.series(sensors, valves)
    assert list(function.channels) == ["c0", "c1", "c2", "v1", "v2"]
    # A vote's failures are rare, so the PFDavg is near the sum, and
    # below it; the trips add up to those of either vote.
    total = sensors.pfd_avg() + valves.pfd_avg()
    assert function.pfd_avg() < total
    assert function.pfd_avg() == pytest.approx(total, rel=2e-2)
    trips = sensors.spurious_trip_rate() + valves.spurious_trip_rate()
    assert function.spurious_trip_rate() == pytest.approx(trips, rel=1e-3)
    with pytest.raises(ValueError, match="name their channels differently"):
        SafetyFunction.series(sensors, vote(1, 2))
    with pytest.raises(ValueError, match="at least one"):
        SafetyFunction.series()
    with pytest.raises(ValueError, match="takes SafetyFunctions"):
        SafetyFunction.series(sensors, "valves")


# -- the results in one call ------------------------------------------------


def test_one_call_gives_every_result():
    function = vote(2, 3, beta=0.1, beta_detected=0.05, beta_safe=0.1)
    result = function.results("low_demand")
    assert isinstance(result, SafetyFunctionResult)
    assert result.pfd_avg == function.pfd_avg()
    assert result.pfh == function.pfh()
    assert result.pfh_undetected == function.pfh(undetected_only=True)
    assert result.pfh_undetected < result.pfh
    assert result.spurious_trip_rate == function.spurious_trip_rate()
    assert result.mean_time_to_spurious_trip == 1 / result.spurious_trip_rate
    assert isinstance(result.sil, SILBand)
    assert result.sil == function.sil("low_demand")
    assert result.sil.value == result.pfd_avg
    assert result.safe_failure_fraction == {
        name: pytest.approx(0.8) for name in ("c0", "c1", "c2")
    }
    assert result.diagnostic_coverage["c0"] == pytest.approx(0.6)
    high = function.results("high_demand")
    assert high.sil.measure == "pfh" and high.sil.value == high.pfh
    assert json.loads(json.dumps(result.to_dict()))["sil"]["sil"] == 2


def test_no_result_depends_on_the_unit_of_time():
    # The same function in years: probabilities as they were, rates 8760
    # times as large, and the same SIL once told the unit.
    hours = vote(2, 3, beta=0.1, beta_detected=0.05)
    year = 8760.0
    unit = SafetyChannel(
        2e-6 * year,
        dangerous_detected=3e-6 * year,
        safe=5e-6 * year,
        repair=MTTR / year,
        proof_test_interval=T1 / year,
    )
    years = vote(2, 3, unit, beta=0.1, beta_detected=0.05)
    assert years.pfd_avg() == pytest.approx(hours.pfd_avg(), rel=1e-9)
    assert years.pfh() == pytest.approx(hours.pfh() * year, rel=1e-9)
    assert years.spurious_trip_rate() == pytest.approx(
        hours.spurious_trip_rate() * year, rel=1e-9
    )
    in_hours = hours.sil("high_demand")
    in_years = years.sil("high_demand", hours_per_unit=year)
    assert in_years.sil == in_hours.sil
    assert in_years.value == pytest.approx(in_hours.value, rel=1e-9)


# -- simulation -------------------------------------------------------------


def test_both_diagrams_against_their_simulation():
    # 20,000 seeded runs of five proof-test periods, in 20 batches whose
    # spread gives the standard errors. A run starts with every channel
    # new, which the long-run values do not: that is worth less than
    # 1e-3 of them here (a repair's length in five periods).
    rates = {"du": 2e-5, "dd": 5e-5, "safe": 1e-4}
    function = vote(
        2, 3, channel(**rates), beta=0.1, beta_detected=0.05, beta_safe=0.1
    )
    window = 5 * T1

    def batches(diagram):
        down, failures = [], []
        for seed in range(20):
            run = diagram.availability(
                window, mc_samples=1000, seed=seed, control_variate=False
            )
            down.append(1.0 - run.sample_mean_availability)
            failures.append(run.system_failures / (1000 * window))
        return np.array(down), np.array(failures)

    def within(estimates, exact):
        error = estimates.std(ddof=1) / math.sqrt(len(estimates))
        assert abs(estimates.mean() - exact) < 4 * error, (
            estimates.mean(),
            exact,
            error,
        )

    down, failures = batches(function.dangerous)
    within(down, function.pfd_avg())
    within(failures, function.pfh())
    _, trips = batches(function.safe)
    within(trips, function.spurious_trip_rate())


# -- the routes -------------------------------------------------------------

SAFETY_CALLS = {
    "pfd_avg": lambda function: function.pfd_avg(),
    "pfd_breakdown": lambda function: function.pfd_breakdown(),
    "pfh": lambda function: function.pfh(),
    "spurious_trip_rate": lambda function: function.spurious_trip_rate(),
    "safe_failure_fraction": lambda function: (
        function.safe_failure_fraction()
    ),
    "sil": lambda function: function.sil("high_demand"),
    "results": lambda function: function.results("low_demand"),
    "simplified_formulas": lambda function: function.simplified_formulas(),
}


@pytest.mark.parametrize("name", sorted(safety_function_kinds()))
def test_every_method_does_what_the_report_says(name):
    function = safety_function_kinds()[name]
    report = function.analysis_routes()
    assert set(report) == set(SAFETY_CALLS)
    for method, call in SAFETY_CALLS.items():
        behaves_as_reported(report[method], lambda: call(function))


def test_the_report_covers_every_public_analysis():
    public = {
        name
        for name, _ in inspect.getmembers(SafetyFunction, callable)
        if not name.startswith("_")
    }
    assert set(SAFETY_CALLS) == public - {"analysis_routes", "series"}


def test_the_routes_are_the_diagrams_own():
    kinds = safety_function_kinds()
    plain = kinds["1oo1, undetected failures alone"]
    report = plain.analysis_routes()
    assert report["pfd_avg"].route == routes.EXACT
    assert report["spurious_trip_rate"].route == routes.EXACT
    assert "No channel can fail safe" in report["spurious_trip_rate"].reason
    voted = kinds["2oo3, common cause"]
    report = voted.analysis_routes()
    dangerous = voted.dangerous.analysis_routes()
    assert report["pfd_avg"] == dangerous["mean_unavailability"]
    assert report["pfh"] == dangerous["system_failure_frequency"]
    assert report["spurious_trip_rate"] == (
        voted.safe.analysis_routes()["system_failure_frequency"]
    )
    assert report["results"].route == routes.NUMERICAL
    series = kinds["votes in series"].analysis_routes()
    assert series["simplified_formulas"].route == routes.REFUSED
