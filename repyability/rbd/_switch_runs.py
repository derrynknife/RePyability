"""The best time to switch blocks of a ``RepairableRBD`` from one way of
running them to another (``RepairableRBD.best_switch_time``): is it worth
running hot, for a whole diagram?

The blocks that switch run on covariate schedules (``_covariate_schedule``:
a regression life under covariates on the diagram's clock). For a switch
time ``s``, each of them runs at its first row of covariates until ``s``
and at its second after, and the system earns the first value rate per
unit of time up until ``s`` and the second after. Each candidate ``s`` is
a twin of the diagram with those schedules (the schedule is plain data,
so the twin is the diagram built again with it), simulated with the same
seed: its streams are named as the diagram's, so every candidate's runs
draw the same uniforms (common random numbers), and the differences
between candidates are far less noisy than their values. Each run is
priced as ``npv_distribution`` prices it (``_pricing``), the value split
at ``s`` between the two rates.

The NPV against ``s`` is scanned on a grid from 0 (switch at once) to the
horizon (never switch), and refined on a finer grid between the best's
neighbours; the scan is returned to audit the answer.
"""

from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np

from repyability.rbd import _npv, _npv_runs, _requirements
from repyability.rbd.results import SimulatedSwitchTime
from repyability.utils import checks

#: Switch times added between the best scanned one's neighbours.
_REFINE = 6
#: Simulations priced at a time: only their present values are kept.
_CHUNK = 2000

_NONE = (
    "No block of the diagram runs on a covariate schedule, so none can "
    "switch from one way of running to another: give each block that can "
    "a fitted surpyval regression model as its reliability and a "
    '"covariates" schedule in its spec (e.g. surpyval.StepSchedule.'
    "from_changepoints([0], [[1.0]])), or, for one unit, use "
    "repyability.best_switch_time."
)


def scheduled_blocks(rbd) -> List[Hashable]:
    """The diagram's blocks on a covariate schedule, as built."""
    return [
        node
        for node, spec in rbd._init_args["components"].items()
        if isinstance(spec, dict)
        and spec.get("covariates") is not None
        and node in rbd.components
    ]


def _require_scheduled(rbd) -> None:
    from repyability.rbd.routes import Refused

    if not scheduled_blocks(rbd):
        raise Refused(_NONE)


def route(rbd, npv_route):
    """``best_switch_time``'s route: refused if no block can switch, else
    ``npv_distribution``'s (whose runs it prices)."""
    from repyability.rbd import routes as r

    if not scheduled_blocks(rbd):
        return r.refused(_NONE)
    if npv_route.route == r.REFUSED:
        return npv_route
    return r.AnalysisRoute(
        r.SIMULATED,
        "The diagram simulated with each candidate switch time on its "
        "scheduled blocks, every candidate from the same seed (common "
        "random numbers), each run priced as npv_distribution prices it.",
        engine=npv_route.engine,
        engine_reason=npv_route.engine_reason,
    )


def _rows(rbd, covariates) -> Dict[Hashable, np.ndarray]:
    """Each switching block's two rows of covariates, checked."""
    if not isinstance(covariates, dict) or not covariates:
        raise ValueError(
            "covariates must be a dict giving, for each block that "
            "switches, its two rows of covariates: before the switch and "
            f"after (e.g. {{'pump': [[60.0], [85.0]]}}); got {covariates!r}."
        )
    blocks = scheduled_blocks(rbd)
    out = {}
    for node, rows in covariates.items():
        if node not in blocks:
            raise ValueError(
                f"Component {node!r} does not run on a covariate schedule, "
                "so it cannot switch: the blocks that do are "
                f"{', '.join(repr(b) for b in blocks)}."
            )
        Z = checks.real_array(rows, f"covariates[{node!r}]").astype(float)
        if Z.ndim == 1 and Z.size == 2:
            Z = Z.reshape(2, 1)
        if Z.ndim != 2 or Z.shape[0] != 2 or not np.all(np.isfinite(Z)):
            raise ValueError(
                f"covariates[{node!r}] must be two rows of finite "
                "covariates, before the switch and after (two numbers for "
                f"a model with one covariate); got {rows!r}."
            )
        out[node] = Z
    return out


def _schedule(Z: np.ndarray, switch: float, horizon: float):
    """The covariate schedule of a block switching at ``switch``."""
    from surpyval.univariate.regression import StepSchedule

    if switch <= 0.0:
        return StepSchedule.from_changepoints([0.0], Z[1:])
    if switch >= horizon:
        return StepSchedule.from_changepoints([0.0], Z[:1])
    return StepSchedule.from_changepoints([0.0, switch], Z)


def _twin(rbd, rows: Dict[Hashable, np.ndarray], switch, horizon):
    """The diagram with its switching blocks switched at ``switch``."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    components = dict(rbd._init_args["components"])
    for node, Z in rows.items():
        components[node] = {
            **components[node],
            "covariates": _schedule(Z, switch, horizon),
        }
    return RepairableRBD(**{**rbd._init_args, "components": components})


def _runs(twin, horizon, switch, rates, money, n, seed, engine, state):
    """Each run's NPV, value and cost (present values) of ``twin`` over
    ``horizon``, the value at ``rates[0]`` until ``switch`` and
    ``rates[1]`` after (``money`` valuing 1 per unit time up, its growth
    and rate); and the engine that ran them."""
    _npv_runs._require_priceable(twin)
    ends = np.array([min(switch, horizon), horizon])
    value = np.zeros(n)
    cost = np.zeros(n)
    used = None
    for first in range(0, n, _CHUNK):
        size = min(_CHUNK, n - first)
        ledger, runs = _npv_runs.priced_chunk(
            twin, horizon, ends, money, size, seed, engine, first, state
        )
        used = used or runs.engine
        part = slice(first, first + size)
        for (k, category, _), (present, _) in ledger.cells.items():
            if category == "value":
                value[part] += rates[k] * present
            else:
                cost[part] += present
    cost += sum(float(p) for p in twin.acquisition_costs.values() if p)
    return value - cost, value, cost, used


def best_switch_time(
    rbd,
    covariates,
    value_rates,
    horizon,
    *,
    discount_rate: float,
    mc_samples: Optional[int],
    seed,
    scan: int,
    engine: str,
    state,
    escalation=None,
    rates: str = "nominal",
    inflation=None,
) -> SimulatedSwitchTime:
    """See ``RepairableRBD.best_switch_time``."""
    rates_basis = rates
    _require_scheduled(rbd)
    _npv_runs._require_priceable(rbd)
    rows = _rows(rbd, covariates)
    rates = np.atleast_1d(checks.real_array(value_rates, "value_rates"))
    if (
        rates.shape != (2,)
        or not np.all(np.isfinite(rates))
        or np.any(rates < 0.0)
    ):
        raise ValueError(
            "value_rates must be two finite numbers, at least 0: what the "
            "system earns per unit of time up before the switch, and "
            f"after; got {value_rates!r}."
        )
    rates = rates.astype(float)
    if not np.any(rates):
        _requirements._require_costs(rbd)
    money = _npv.Money(
        rbd, discount_rate, 1.0, escalation, rates_basis, inflation
    )
    H = _npv._number(horizon, "horizon", strict=True)
    n = checks.whole_number(
        1_000 if mc_samples is None else mc_samples, "mc_samples"
    )
    count = checks.whole_number(scan, "scan", minimum=3)
    checks.one_of("engine", engine, ("auto", "python", "numba"))
    seed = checks.seed(seed)
    if seed is None:
        # Every candidate shares a seed (common random numbers): drawn
        # here, and kept in the result to repeat the search.
        seed = int(np.random.SeedSequence().generate_state(1)[0])

    valued: Dict[float, Tuple[np.ndarray, Any]] = {}

    def at(s: float) -> np.ndarray:
        if s not in valued:
            twin = _twin(rbd, rows, s, H)
            samples, _, _, used = _runs(
                twin, H, s, rates, money, n, seed, engine, state
            )
            valued[s] = (samples, used)
        return valued[s][0]

    grid = np.linspace(0.0, H, count)
    means = [float(at(float(s)).mean()) for s in grid]
    i = int(np.argmax(means))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, count - 1)]
    for s in np.linspace(lo, hi, _REFINE + 2)[1:-1]:
        at(float(s))
    times = sorted(valued)
    npvs = [float(valued[s][0].mean()) for s in times]
    best_s = times[int(np.argmax(npvs))]
    best = valued[best_s][0]
    never, at_once = valued[H][0], valued[0.0][0]
    if best_s <= 0.0:
        choice = "switch at once"
    elif best_s >= H:
        choice = "never switch"
    else:
        choice = "switch"
    npv = float(best.mean())
    return SimulatedSwitchTime(
        switch_time=float(best_s),
        choice=choice,
        npv=npv,
        never_switch_npv=float(never.mean()),
        switch_at_once_npv=float(at_once.mean()),
        gain=max(npv - max(float(never.mean()), float(at_once.mean())), 0.0),
        samples=best,
        never_switch_samples=never,
        switch_at_once_samples=at_once,
        times=[float(s) for s in times],
        npvs=npvs,
        horizon=H,
        discount_rate=money.given_rate,
        value_rates=[float(v) for v in rates],
        n_simulations=n,
        seed=seed,
        engine=valued[best_s][1] or engine,
        rate_basis=money.basis,
        inflation=money.inflation,
        escalation=dict(money.escalation),
    )
