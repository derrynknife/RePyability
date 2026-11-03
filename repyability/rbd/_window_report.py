"""A repairable diagram's work and cost in each of a run of consecutive
windows from its state now.

The exact counts and means over ``[0, t)`` (``expected_events``,
``expected_cost``) add over windows, so a window's own value is the
difference of the values at its two ends. The spread of a cost cannot be
differenced: it comes from simulated histories, each cut at every
window's end. The simulations of a seeded run draw the same lives
whatever the window, so the runs to each end are the same histories cut
off there, and a simulation's cost in a window is its cost to the
window's end less its cost to the window's start.
"""

import math
from typing import Sequence, Tuple

import numpy as np

from repyability.rbd.results import CostByWindow, EventsByWindow
from repyability.utils.checks import real_array

#: Numerical noise below this (relative to the largest value) is shown as 0.
_NOISE = 1e-9


def window_ends(ends, width, count) -> np.ndarray:
    """The windows' end times from ``ends`` (increasing, more than 0), or
    ``count`` windows of length ``width``."""
    if ends is None:
        if width is None or count is None:
            raise ValueError(
                "Give the windows' ends, or width and count (windows of "
                "that length)."
            )
        if width is True or count is True:
            raise TypeError("width and count must be numbers.")
        width = float(real_array(width, "width"))
        if not (math.isfinite(width) and width > 0.0):
            raise ValueError(f"width must be more than 0; got {width!r}.")
        if int(count) != count or count < 1:
            raise ValueError("count must be a whole number, 1 or more.")
        return width * np.arange(1, int(count) + 1)
    if width is not None or count is not None:
        raise ValueError(
            "Give the windows' ends, or width and count, not both."
        )
    out = np.atleast_1d(real_array(ends, "ends"))
    if out.ndim != 1 or out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"ends must be a list of times; got {ends!r}.")
    if out[0] <= 0.0 or np.any(np.diff(out) <= 0.0):
        raise ValueError(
            "ends must increase, the first more than 0 (the windows start "
            f"from now); got {ends!r}."
        )
    return out.astype(float)


def _between(cumulative) -> np.ndarray:
    """Each window's own value from the values at its ends, the tiny
    negatives numerics leave in a window with nothing in it set to 0."""
    values = np.diff(np.asarray(cumulative, dtype=float), prepend=0.0)
    scale = max(1.0, float(np.max(np.abs(cumulative))))
    values[(values < 0.0) & (values > -_NOISE * scale)] = 0.0
    return values


def expected_events_by_window(
    rbd, ends, width, count, state, working_nodes, broken_nodes, method
) -> EventsByWindow:
    """See ``RepairableRBD.expected_events_by_window``."""
    stops = window_ends(ends, width, count)
    total = rbd.expected_events(
        stops,
        working_nodes=working_nodes,
        broken_nodes=broken_nodes,
        method=method,
        state=state,
    )

    def per_node(table: dict) -> dict:
        return {node: _between(values) for node, values in table.items()}

    return EventsByWindow(
        starts=np.concatenate(([0.0], stops[:-1])),
        ends=stops,
        system_failures=_between(total.system_failures),
        system_planned_outages=_between(total.system_planned_outages),
        system_downtime=_between(total.system_downtime),
        node_failures=per_node(total.node_failures),
        node_corrective=per_node(total.node_corrective),
        node_preventive=per_node(total.node_preventive),
        node_inspections=per_node(total.node_inspections),
        node_downtime=per_node(total.node_downtime),
    )


def cost_by_window(
    rbd,
    ends,
    width,
    count,
    state,
    mc_samples,
    seed,
    engine,
    levels: Sequence[float],
    working_nodes,
    broken_nodes,
    method,
) -> CostByWindow:
    """See ``RepairableRBD.cost_by_window``."""
    stops = window_ends(ends, width, count)
    levels_t: Tuple[float, ...] = tuple(
        float(q) for q in np.atleast_1d(real_array(levels, "levels"))
    )
    if not all(0.0 <= q <= 100.0 for q in levels_t):
        raise ValueError(
            f"levels must be percentiles, 0 to 100; got {levels!r}."
        )
    exact = rbd.expected_cost(
        stops,
        working_nodes=working_nodes,
        broken_nodes=broken_nodes,
        method=method,
        state=state,
    ).mean
    cumulative = []
    for stop in stops:
        run = rbd.cost(
            float(stop),
            working_nodes=working_nodes,
            broken_nodes=broken_nodes,
            method=method,
            mc_samples=mc_samples,
            seed=seed,
            engine=engine,
            state=state,
            control_variate=False,
            conditional=False,
        )
        cumulative.append(np.asarray(run.samples, dtype=float))
    to_end = np.vstack(cumulative)
    samples = np.diff(to_end, axis=0, prepend=0.0)
    scale = max(1.0, float(np.max(np.abs(to_end))))
    samples[(samples < 0.0) & (samples > -_NOISE * scale)] = 0.0
    n = samples.shape[1]
    return CostByWindow(
        starts=np.concatenate(([0.0], stops[:-1])),
        ends=stops,
        mean=_between(exact),
        simulated_mean=samples.mean(axis=1),
        standard_error=samples.std(axis=1, ddof=1) / math.sqrt(n),
        samples=samples,
        n_simulations=n,
        levels=levels_t,
        percentiles=np.percentile(samples, levels_t, axis=1).T,
    )
