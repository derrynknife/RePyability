"""A ``RepairableRBD``'s long-run costs: the expected cost rate, with each
node's share and its actions (``expected_cost_rate``), the cost of buying
the components (``acquisition_cost``) and the total cost over a horizon,
discounted or not (``total_cost``). The methods of ``RepairableRBD`` of
those names call these.
"""

from functools import partial
from typing import (
    Collection,
    Hashable,
    Optional,
    Tuple,
    Union,
)

import numpy as np

from repyability.rbd import _crews, _long_run, _requirements
from repyability.rbd._common import (
    _discount_rate,
    _horizons,
    _present_horizon,
    _squeeze_values,
)
from repyability.rbd._spec import _mean_cost


def expected_cost_rate(
    rbd,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
) -> float:
    """See ``RepairableRBD.expected_cost_rate``."""
    if not rbd.has_costs:
        return 0.0

    working_nodes = set() if working_nodes is None else set(working_nodes)
    broken_nodes = set() if broken_nodes is None else set(broken_nodes)
    rbd._validate_node_overrides(working_nodes, broken_nodes)
    forced = working_nodes | broken_nodes
    setups = [group for group in rbd._maintenance.values() if group.setup_cost]
    if setups:
        _requirements._require_separate_setups(rbd)

    rate = 0.0

    # Production lost while the *system* is down.
    if rbd.downtime_cost_rate:
        unavailability = rbd.mean_unavailability(working_nodes, broken_nodes)
        rate += rbd.downtime_cost_rate * unavailability

    if not rbd.costs and not setups:
        return rate

    if rbd._crews_couple():
        # Held working or broken, a node needs no crew, and the others
        # have more of them: from the chain without it.
        probabilities, weights = _crews._chain_probabilities(
            rbd, working_nodes, broken_nodes
        )
        node_availability = {
            node: float(weights @ probabilities[node]) for node in rbd.nodes
        }
    else:
        node_availability = _squeeze_values(
            rbd._probabilities_with_overrides(
                rbd.node_availability(), working_nodes, broken_nodes
            )
        )
    for node in rbd.costs:
        rate += _node_cost_rate(
            rbd, node, node_availability[node], node in forced
        )
    # A maintenance group's set-up, at each failure and each preventive
    # replacement of a member (none is renewed early: that would have
    # been refused), a forced member making none.
    for group in setups:
        rate += group.setup_cost * sum(
            sum(_node_actions(rbd, node, node_availability[node]))
            for node in group.members
            if node not in forced
        )
    return rate


def _node_cost_rate(
    rbd, node, availability: float, forced: bool = False
) -> float:
    """A component's own running cost per unit time, in the long run:
    its corrective, preventive and inspection actions, and its own
    downtime (``availability`` its long-run availability). A forced node
    never changes state, so it incurs no actions."""
    node_costs = rbd.costs.get(node, {})
    rate = 0.0
    # Corrective actions, charged per failure, and preventive ones.
    per_action = sum(
        _mean_cost(node_costs[key])
        for key in rbd.PER_FAILURE_COST_KEYS
        if key in node_costs
    )
    preventive = node_costs.get("preventive_cost")
    if (per_action or preventive is not None) and not forced:
        failures, maintained = _node_actions(rbd, node, availability)
        rate += per_action * failures
        if preventive is not None:
            rate += _mean_cost(preventive) * maintained
    inspection = node_costs.get("inspection_cost")
    if inspection is not None and not forced:
        schedule = rbd._preventive.get(node)
        if schedule is not None and schedule.policy == "condition":
            # Inspected at each multiple of the interval at which it is
            # up (one in a repair or replacement is not).
            up = _long_run._block_cycle(rbd, node).before
            rate += _mean_cost(inspection) * up / schedule.interval
        else:
            # One test per interval, but those that fall in a repair,
            # which are not done.
            unit = _requirements._tested_unit(rbd, node)
            tests = (
                1.0 / rbd._inspection[node].interval
                if unit is None
                else unit.long_run.inspections
            )
            rate += _mean_cost(inspection) * tests
    # Optional cost of *this component* being down, whether or not the
    # system as a whole is.
    downtime_cost = node_costs.get("downtime_cost", 0.0)
    if downtime_cost:
        rate += downtime_cost * (1.0 - availability)
    return rate


def _node_actions(rbd, node, availability: float) -> Tuple[float, float]:
    """A component's corrective and preventive actions per unit time,
    in the long run (``availability`` its long-run availability)."""
    if rbd._crews_couple():
        # Waiting for a crew, as while repaired, it cannot fail: it
        # fails at its constant rate while it is up.
        life = _crews._crew_chain_rates(rbd)[node][0]
        return life * availability, 0.0
    if node in rbd._standby:
        # Each of its units' failures is a repair.
        return (
            _long_run._standby_long_run(rbd, node).unit_failure_frequency,
            0.0,
        )
    failures, maintained, _ = _long_run._node_frequencies(rbd, node)
    return failures, maintained


def acquisition_cost(rbd) -> float:
    """See ``RepairableRBD.acquisition_cost``."""
    return float(sum(rbd.acquisition_costs.values()))


def total_cost(
    rbd,
    horizon,
    working_nodes: Optional[Collection[Hashable]],
    broken_nodes: Optional[Collection[Hashable]],
    *,
    discount_rate: float,
) -> Union[float, np.ndarray]:
    """See ``RepairableRBD.total_cost``."""
    rate = _discount_rate(discount_rate)
    present = _present_horizon(
        _horizons(horizon, rate), rate, partial(_requirements._mean_lives, rbd)
    )
    total = rbd.acquisition_cost + present * rbd.expected_cost_rate(
        working_nodes, broken_nodes
    )
    return float(total) if np.ndim(total) == 0 else total
