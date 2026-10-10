"""A ``RepairableRBD``'s repair crews in the exact methods: whether the
crews limit or couple the components, and the crews' Markov chain
(``_crew_chain``, which copies the simulation's queue) with its rates,
state probabilities and outage terms in the long run.
``RepairableRBD``'s methods call these.
"""

import math
from typing import (
    Any,
    Dict,
    List,
    NoReturn,
    Tuple,
)

import numpy as np

from repyability.rbd import (
    _ccf_groups,
)
from repyability.rbd import _crew_chain as crew_chain
from repyability.rbd import _long_run, _requirements
from repyability.rbd._common import (
    _constant_rate,
    _repair_rate,
)
from repyability.rbd._events import (
    _Unit,
)


def _crew_served(rbd) -> list:
    """The jobs the repair crews work on, by key: this RBD's own
    components, not a nested RBD's (which has crews of its own), and
    each unit of a standby group (``_Unit(node, unit)``)."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    served: list = []
    for node, component in rbd.components.items():
        if isinstance(component, RepairableRBD):
            continue
        arrangement = rbd._standby.get(node)
        if arrangement is None:
            served.append(node)
        else:
            served.extend(
                _Unit(node, unit) for unit in range(arrangement.units)
            )
    return served


def _crews_limited(rbd) -> bool:
    """Whether there are fewer repair crews than components that may
    need one, so that a job may wait."""
    return rbd.repair_crews is not None and rbd.repair_crews < len(
        _crew_served(rbd)
    )


def _require_unlimited_crews(
    rbd,
    assumes: str = "these values assume",
    advice: str = "Simulate the system with availability() or cost().",
) -> None:
    """Raise if a job may wait for a repair crew: components then no
    longer fail and recover independently, which ``assumes`` (what
    assumes it, and ``assume``): do ``advice`` instead."""
    if rbd._crews_couple():
        raise NotImplementedError(
            f"With {rbd.repair_crews} repair crew(s) for "
            f"{len(_crew_served(rbd))} components, a component can "
            "wait for a crew, so the components no longer fail and "
            f"recover independently, which {assumes}. {advice}"
        )


def _no_crew_chain(rbd, why: str) -> NoReturn:
    """Raise that the repair crews' Markov chain does not cover this RBD,
    because ``why``."""
    raise NotImplementedError(
        f"With {rbd.repair_crews} repair crew(s) for "
        f"{len(_crew_served(rbd))} components, a component can wait "
        "for a crew, and the exact long-run values come from a Markov "
        "chain of the components' states and the repair queue: "
        f"{why}. Simulate the system with availability() or cost()."
    )


def _crew_chain_rates(rbd) -> Dict[Any, Tuple[float, float]]:
    """The failure and repair rates (``inf`` for an instant repair) of
    the components the repair crews work on, for their Markov chain
    (see ``_crew_chain.py``). Raise if the chain does not cover them:
    scheduled maintenance or inspection, a life or repair time that is
    not exponential, or more states than it is solved for."""
    assert rbd.repair_crews is not None  # limited crews only
    for node in rbd._standby:
        _no_crew_chain(
            rbd,
            "it has no place for a standby group, which component "
            f"{node!r} is",
        )
    rates: Dict[Any, Tuple[float, float]] = {}
    for node in _crew_served(rbd):
        component = rbd.components[node]
        if node in rbd._preventive:
            _no_crew_chain(
                rbd,
                "it has no place for scheduled maintenance, which "
                f"component {node!r} has",
            )
        if node in rbd._inspection:
            _no_crew_chain(
                rbd,
                "it has no place for inspections, which component "
                f"{node!r} has",
            )
        if node in rbd._imperfect:
            _no_crew_chain(
                rbd,
                "it has no place for imperfect repair, which component "
                f"{node!r} has",
            )
        life = _constant_rate(component.reliability)
        if life is None:
            _no_crew_chain(
                rbd,
                "it needs exponential lives (a constant failure rate), "
                f"and the life of component {node!r} is not one",
            )
        repair = _repair_rate(component.time_to_replace)
        if repair is None:
            _no_crew_chain(
                rbd,
                "it needs exponential repair times (or instant repair), "
                f"and the repair times of component {node!r} are not",
            )
        rates[node] = (life, repair)
    count = crew_chain.state_count(
        [rbd._priority.get(node, 0.0) for node in rates],
        [math.isinf(repair) for _, repair in rates.values()],
        rbd.repair_crews,
    )
    if count > crew_chain.MAX_STATES:
        _no_crew_chain(
            rbd,
            f"here it has {count:,} states, more than the "
            f"{crew_chain.MAX_STATES:,} it is solved for",
        )
    return rates


def _require_crew_chain(rbd) -> None:
    """Raise if the exact long-run values with limited repair crews
    cannot be computed: the Markov chain does not take common-cause
    groups in (#251, see ``_ccf_rates``), does not cover the components
    (see ``_crew_chain_rates``), or nested RBDs' calendars fall together
    (see ``_require_calendars``)."""
    _ccf_groups._require_ccf_long_run(rbd)
    _crew_chain_rates(rbd)
    _requirements._require_calendars(rbd)


def _crew_chain(
    rbd, forced: frozenset = frozenset()
) -> "crew_chain.CrewChain":
    """The Markov chain of the components the repair crews work on, and
    its long-run distribution (see ``_crew_chain.py``), leaving out the
    nodes in ``forced``: held working a node never fails, and held
    broken it is never repaired, so neither needs a crew. Solved once
    for each set of rates and kept."""
    _require_crew_chain(rbd)
    rates = _crew_chain_rates(rbd)
    assert rbd.repair_crews is not None  # limited crews only
    nodes = [node for node in rates if node not in forced]
    priorities = [rbd._priority.get(node, 0.0) for node in nodes]
    key = (
        rbd.repair_crews,
        tuple(zip(nodes, (rates[node] for node in nodes), priorities)),
    )
    cache = rbd.__dict__.setdefault("_crew_chains", {})
    if key not in cache:
        cache[key] = crew_chain.solve(
            nodes,
            [rates[node][0] for node in nodes],
            [rates[node][1] for node in nodes],
            priorities,
            rbd.repair_crews,
        )
    return cache[key]


def _chain_probabilities(
    rbd, working_nodes, broken_nodes
) -> Tuple[dict, np.ndarray]:
    """With limited repair crews, every node's availability in each
    state of their Markov chain (1 or 0 for the components the crews
    work on and those held working or broken, and a nested RBD's own
    long-run availability, as it has crews of its own), and the states'
    long-run probabilities: the long-run values are then averages over
    the states, as over the times of ``_long_run_grid``."""
    working_nodes = set(working_nodes or ())
    broken_nodes = set(broken_nodes or ())
    rbd._validate_node_overrides(working_nodes, broken_nodes)
    chain = _crew_chain(rbd, frozenset(working_nodes | broken_nodes))
    size = len(chain.probabilities)
    out: dict = {
        node: chain.up[:, k].astype(float)
        for k, node in enumerate(chain.nodes)
    }
    for node, component in rbd.components.items():
        if node in working_nodes:
            out[node] = np.ones(size)
        elif node in broken_nodes:
            out[node] = np.zeros(size)
        elif node not in out:
            out[node] = np.full(size, float(component.mean_availability()))
    for node in rbd.in_or_out:
        out[node] = np.ones(size)
    return out, chain.probabilities


def _chain_outage_terms(
    rbd, working_nodes, broken_nodes
) -> Tuple[List[Tuple[Any, float]], float]:
    """``_outage_terms`` with limited repair crews, over the
    states of their Markov chain: in each, a component that is up fails
    at its constant rate, and takes the system down if it is critical
    there. A nested RBD enters through its own frequencies, as it has
    crews of its own."""
    availability, unavailability, weights = (
        _long_run._long_run_unavailabilities(rbd, working_nodes, broken_nodes)
    )
    forced = set(working_nodes or ()) | set(broken_nodes or ())
    rates = _crew_chain_rates(rbd)
    birnbaum = rbd._birnbaum_importance(
        availability, node_failures=unavailability
    )
    terms: List[Tuple[Any, float]] = []
    planned = 0.0
    for node in rbd.components:
        if node in forced:
            continue
        importance = np.asarray(birnbaum[node])
        if node in rates:
            life = rates[node][0]
            node_failures, node_planned = life * availability[node], 0.0
        else:
            node_failures, _, node_planned = _long_run._node_frequencies(
                rbd, node
            )
        terms.append((node, float(weights @ (importance * node_failures))))
        planned += float(weights @ (importance * node_planned))
    return terms, planned
