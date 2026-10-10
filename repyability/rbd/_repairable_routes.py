"""A ``RepairableRBD``'s analysis routes: for each public analysis,
without running anything, whether it is exact, numerical, simulated or
refused, with the methods' own reasons (``routes``). The method
``RepairableRBD.analysis_routes`` calls this; ``test_analysis_routes``
checks it against the methods on every catalogue diagram.
"""

import dataclasses
from functools import partial
from typing import (
    Dict,
    Hashable,
    Tuple,
)

from repyability.rbd import (
    _ccf_groups,
    _crews,
    _curves,
    _event_loop,
    _intervals,
    _long_run,
    _repairable_allocation,
    _repairable_capacity,
    _requirements,
    _runs,
    _spares,
    _timeline_runs,
)
from repyability.rbd._runs import (  # noqa: E402,F401
    _UNSTREAMED,
    failure_criticality_index_per_component_failures,
    failure_criticality_index_per_system_failures,
    restoration_criticality_index_by_component,
    restoration_criticality_index_by_system,
)
from repyability.rbd.routes import AnalysisRoute


def analysis_routes(rbd) -> Dict[str, "AnalysisRoute"]:
    """See ``RepairableRBD.analysis_routes``."""
    from repyability.rbd import routes as r

    out: Dict[str, r.AnalysisRoute] = {}

    def give(names, route) -> None:
        for name in names:
            out[name] = route

    long_run = _long_run._long_run_route(rbd)

    long_run_nodes = _long_run._long_run_nodes(rbd)

    def from_long_run(route, reason):
        """A method of ``route``, explained by ``reason``, built on the
        long-run values: refused as they are, and numerical or simulated
        as the components' are."""
        if long_run.route == r.REFUSED:
            return long_run
        return r.with_nodes(route, reason, long_run_nodes, "long-run values")

    give(("mean_availability", "mean_unavailability"), long_run)
    out["node_availability"] = (
        _long_run._long_run_route(rbd, groups=False)
        if rbd.ccf_groups
        else long_run
    )
    frequencies = (
        None
        if long_run.route == r.REFUSED
        else r.refusal(partial(_ccf_groups._require_ccf_frequencies, rbd))
    )
    give(
        (
            "system_failure_frequency",
            "mean_time_between_failures",
            "mean_up_time",
            "mean_down_time",
        ),
        r.refused(frequencies) if frequencies else long_run,
    )
    give(
        ("barlow_proschan_importance",),
        (
            r.refused(frequencies)
            if frequencies
            else (
                long_run
                if long_run.route == r.REFUSED
                else dataclasses.replace(
                    long_run,
                    reason="Each component's share of the system's "
                    "long-run failure frequency (the Birnbaum/Vesely "
                    "formula's terms). " + long_run.reason,
                )
            )
        ),
    )
    # The allocations assume independent components, which limited
    # repair crews make them not; the common-cause groups' chains take
    # theirs in.
    allocation = r.refusal(
        partial(
            _crews._require_unlimited_crews,
            rbd,
            *_repairable_allocation._ALLOCATION_CREWS,
        )
    ) or r.refusal(partial(_ccf_groups._require_ccf_long_run, rbd))
    conditioned = long_run
    if rbd.ccf_groups and long_run.route != r.REFUSED:
        conditioned = dataclasses.replace(
            long_run,
            reason=long_run.reason
            + " A common-cause group's member is conditioned on its "
            "state at each time, as the shared causes tie the other "
            "members' to it.",
        )
    if rbd._crews_couple() and long_run.route != r.REFUSED:
        conditioned = dataclasses.replace(
            long_run,
            reason=long_run.reason
            + " With the crews, Birnbaum's measure, the improvement "
            "potential, RAW and RRW hold each node working and failed "
            "in the chain, solved without it; the criticality and "
            "Fussell-Vesely measures are probabilities over its "
            "states.",
        )
    give(
        (
            "birnbaum_importance",
            "improvement_potential",
            "risk_achievement_worth",
            "risk_reduction_worth",
            "criticality_importance",
        ),
        conditioned,
    )
    pairs = r.refusal(rbd._require_no_ccf_pairs)
    give(
        ("joint_importance",),
        (
            r.refused(pairs)
            if pairs
            else (
                conditioned
                if conditioned.route == r.REFUSED
                else dataclasses.replace(
                    conditioned,
                    reason="Each component's Birnbaum importance with "
                    "the other held up, less with it held down. "
                    + conditioned.reason,
                )
            )
        ),
    )
    give(
        ("differential_importance",),
        (
            conditioned
            if conditioned.route == r.REFUSED
            else dataclasses.replace(
                conditioned,
                reason="Birnbaum's measure (or, for a proportional "
                "change, the criticality importance), shared out; "
                "over='parameters', parameter_sensitivity's "
                "derivatives. " + conditioned.reason,
            )
        ),
    )
    give(
        ("fussell_vesely",),
        from_long_run(
            r.EXACT,
            "The exact probability that a minimal cut set containing "
            "each node is down, over the system's unavailability, from "
            "the exact long-run values (method='rare_event' sums the cut "
            "sets' probabilities instead).",
        ),
    )
    give(
        ("parameter_sensitivity",),
        from_long_run(
            r.NUMERICAL,
            "Central differences of the long-run availability (or cost "
            "rate) in each lever, the diagram rebuilt with the lever "
            "moved; at times or over a window, of the values over time.",
        ),
    )
    if rbd.has_costs:
        setups = r.refusal(
            partial(_requirements._require_separate_setups, rbd)
        )
        give(
            ("expected_cost_rate", "total_cost"),
            (
                r.refused(setups)
                if setups
                else from_long_run(
                    r.EXACT,
                    "The long-run cost rate, from the exact long-run "
                    "values (total_cost multiplies it by the time).",
                )
            ),
        )
    else:
        give(
            ("expected_cost_rate", "total_cost"),
            r.AnalysisRoute(
                r.EXACT,
                "No running cost is priced, so the cost rate is 0 "
                "(total_cost adds any acquisition costs).",
            ),
        )
    capacity = (
        None
        if long_run.route == r.REFUSED
        else _repairable_capacity._capacity_refusal(rbd)
    )
    out["capacity_distribution"] = (
        r.refused(*capacity)
        if capacity
        else from_long_run(
            r.EXACT,
            "The exact long-run distribution of the system's capacity, "
            + (
                "averaged over the states of the repair crews' Markov "
                "chain."
                if rbd._crews_couple()
                else "from the components' long-run availabilities."
            ),
        )
    )
    no_capacity = r.refusal(rbd._require_numbered_capacities) or (
        r.refusal(rbd._require_capacity)
    )
    out["system_capacity"] = (
        r.refused(no_capacity)
        if no_capacity
        else r.AnalysisRoute(
            r.EXACT,
            "The exact distribution of the system's capacity from the "
            "node probabilities given.",
        )
    )
    give(
        (
            "point_availability",
            "mission_availability",
            "point_unavailability",
            "mission_unavailability",
        ),
        _curves._over_time(rbd),
    )
    give(
        ("availability_rate",),
        (
            _curves._crew_over_time(rbd, "rate")
            if rbd._crews_couple()
            else _curves._over_time(
                rbd,
                "Each component's renewal equation solved on a grid (to "
                "about 1e-7) and differentiated on it, times its Birnbaum "
                "importance at the components' availabilities; the jumps "
                "at scheduled events split along the path between the "
                "values either side."
                + (
                    " A common-cause group's part is its chain's moves, "
                    "split by cause and by member repair, at the "
                    "system's availability with the group in each of "
                    "its states (exact); the jumps at a hidden group's "
                    "tests are split by the Shapley value of what jumps."
                    if rbd.ccf_groups
                    else ""
                ),
            )
        ),
    )
    over_time_capacity = _repairable_capacity._capacity_refusal(rbd)
    give(
        ("point_capacity", "mission_capacity"),
        (
            r.refused(*over_time_capacity)
            if over_time_capacity
            else _curves._over_time(
                rbd,
                "Each component's renewal equation solved on a grid (to "
                "about 1e-7), a degrading component's stages through its "
                "renewals too, and the system's capacity distribution "
                "exactly at its components' at each time.",
                kind="capacity",
            )
        ),
    )
    window = _curves._over_time(
        rbd,
        "Each component's expected events from its renewal equation, "
        "solved on a grid (to about 1e-7), and the system's failures by "
        "the time-dependent Birnbaum/Vesely formula.",
        kind="window",
    )
    give(("expected_failures", "expected_events"), window)
    out["expected_cost"] = (
        window
        if rbd.has_costs
        else r.AnalysisRoute(
            r.EXACT,
            "No running cost is priced, so the expected cost is 0 (with "
            "any acquisition costs beside it).",
        )
    )
    # A group's members' MTTF and MTTR are their group's: the
    # availability allocations hold them, and a member's copies join its
    # group.
    held_members = (
        " A common-cause group's members keep their availability: "
        "their MTTF and MTTR are their group's."
        if rbd.ccf_groups
        else ""
    )
    out["allocate_redundancy"] = (
        r.refused(allocation)
        if allocation
        else from_long_run(
            r.EXACT,
            "A search over the exact long-run values of the candidates."
            + (
                " A common-cause group member's copies join its group, "
                "whose chain takes them in: a BetaFactor group's (an MGL "
                "group's member refuses)."
                if rbd.ccf_groups
                else ""
            ),
        )
    )
    out["availability_allocation"] = (
        r.refused(allocation)
        if allocation
        else from_long_run(
            r.EXACT,
            "A search over the exact long-run values of the candidates."
            + held_members,
        )
    )
    out["mttf_mttr_allocation"] = (
        r.refused(allocation)
        if allocation
        else from_long_run(
            r.NUMERICAL,
            "A solver for the components' MTTF and MTTR targets, over the "
            "exact long-run values." + held_members,
        )
    )
    if out["availability_allocation"].route != r.REFUSED:
        # Which components are allocated one follows from the diagram
        # (and fixed, which only holds more).
        none = r.refusal(
            partial(_repairable_allocation._allocated_levers, rbd, set())
        )
        if none:
            give(
                ("availability_allocation", "mttf_mttr_allocation"),
                r.refused(none),
            )
    # The optimisers, called for every node they can choose for.
    targets = r.refusal(
        partial(
            _intervals._interval_targets,
            rbd,
            None,
            None,
        )
    )
    interval_crews = r.refusal(
        partial(_intervals._require_interval_crews, rbd)
    )
    refusal = (
        r.refusal(partial(_intervals._maintained, rbd, None))
        or targets
        or interval_crews
    )
    out["optimal_replacement_intervals"] = (
        r.refused(refusal)
        if refusal
        else from_long_run(
            r.NUMERICAL,
            "An optimiser over the age-replacement intervals, with the "
            "exact long-run values at each.",
        )
    )
    refusal = (
        r.refusal(partial(_intervals._inspected, rbd, None))
        or targets
        or interval_crews
        or next(
            (
                message
                for message in (
                    r.refusal(
                        partial(
                            _requirements._require_tested_exact,
                            rbd,
                            node,
                        )
                    )
                    for node in rbd._inspection
                )
                if message
            ),
            None,
        )
        or r.refusal(partial(_requirements._require_one_inspected, rbd))
    )
    out["optimal_inspection_intervals"] = (
        r.refused(refusal)
        if refusal
        else from_long_run(
            r.NUMERICAL,
            "An optimiser over the inspection interval, with the exact "
            "long-run values at each.",
        )
    )
    # Spares: each component's replacements, counted as a renewal
    # process, in the order the methods check them.
    for name, long_run_count in (
        ("spares_demand", False),
        ("spares_stock", True),
    ):
        refusal = r.refusal(
            partial(
                _crews._require_unlimited_crews,
                rbd,
                *(
                    _spares._STOCK_CREWS
                    if long_run_count
                    else _spares._SPARES_CREWS
                ),
            )
        )
        which: Tuple[Hashable, ...] = ()
        for node in _spares._spares_nodes(rbd, None):
            if refusal:
                break
            refusal = r.refusal(
                partial(_spares._replacements, rbd, node, long_run_count)
            )
            which = (node,)
        out[name] = (
            r.refused(refusal, which)
            if refusal
            else r.AnalysisRoute(
                r.NUMERICAL,
                "Each component's replacements, a renewal process, "
                "counted on a grid (to about 1e-6): under block "
                + (
                    "replacement, averaged over where the lead time "
                    "falls in the block interval (with repairs or "
                    "block replacements that take time, from a typical "
                    "replacement, a failure at each phase of the "
                    "interval or a block replacement, in the long run)"
                    if long_run_count
                    else "replacement, from one block interval to the " "next"
                )
                + "; with hidden failures, on its tests: exactly, tested "
                "and repaired in no time by tests that find every "
                "failure, and otherwise from the cycle between the tests "
                "that find failures, followed test by test (#159).",
            )
        )
    plan, streamed = _event_loop._stream_plan(rbd, 1.0, 0, False)
    paired = (
        ""
        if streamed
        else " Antithetic pairs are refused: some components' draws "
        "do not come from a stream."
    )
    groups = r.refusal(partial(_ccf_groups._require_groups_simulated, rbd))
    given = r.refusal(partial(_requirements._require_capacities_given, rbd))
    engine, why = _runs._engine_choice(rbd, capacity=rbd._has_capacity())
    causes = (
        " Each common-cause group's shared causes strike as Poisson "
        "processes, each failing the members it names that are up."
        if _ccf_groups._has_ccf(rbd)
        else ""
    )
    out["availability"] = (
        r.refused(groups)
        if groups
        else (
            r.refused(given, tuple(rbd._capacity_models()))
            if given
            else r.AnalysisRoute(
                r.SIMULATED,
                "A discrete-event simulation of the components' failures, "
                "repairs and maintenance." + causes + paired,
                engine=engine,
                engine_reason=why,
            )
        )
    )
    out["simulate_chunk"] = out["availability"]
    unsaved = r.refusal(partial(_runs._shard_system, rbd))
    out["shards"] = (
        out["availability"]
        if out["availability"].route == r.REFUSED
        else (
            r.refused(unsaved)
            if unsaved
            else r.AnalysisRoute(
                r.SIMULATED,
                "The run's simulations as shards, to simulate anywhere "
                "with run_shard (see shards); availability_from_chunks "
                "merges their partials into its result.",
            )
        )
    )
    out["availability_from_chunks"] = (
        out["availability"]
        if given or groups
        else r.AnalysisRoute(
            r.SIMULATED,
            "Simulated chunks of a run (see simulate_chunk), merged into "
            "its result.",
        )
    )
    engine, why = _timeline_runs.engine_choice(rbd, plan)
    out["simulate_timelines"] = (
        r.refused(groups)
        if groups
        else r.AnalysisRoute(
            r.SIMULATED,
            "The simulations availability runs, their histories kept "
            "as timelines: recorded by the event loop as it runs." + paired,
            engine=engine,
            engine_reason=why,
        )
    )
    engine, why = _runs._engine_choice(rbd, capacity=False)
    out["cost"] = (
        r.refused(groups)
        if groups and rbd.has_costs
        else r.AnalysisRoute(
            r.SIMULATED,
            "A discrete-event simulation of the components' failures, "
            "repairs and maintenance, and what they cost." + paired,
            engine=engine,
            engine_reason=why,
        )
    )
    twin = _runs._twin_report(rbd)
    # The expected values over the window, where the methods over time
    # work them out, need no simulation, and a run takes them (#187).
    exact_means = ""
    if all(
        out[name].route in (r.EXACT, r.NUMERICAL)
        for name in ("mission_availability", "expected_events")
        + (("expected_cost",) if rbd.has_costs else ())
    ):
        exact_means = (
            " Its expected values over the window need none: "
            "mission_availability, expected_events and expected_cost "
            "work them out, and by default a run's means are theirs, "
            "with no error, so a run to a tolerance stops at "
            "once. Simulate for their spread: each simulation's values, "
            "percentiles, the chance of no failure."
        )
    # Or a run's means are taken given its modules' histories (#189).
    if not exact_means:
        modules = _runs._conditional_applies(rbd)
        if modules:
            names = ", ".join(map(repr, modules))
            exact_means = (
                " By default a run's means take each "
                "simulation's expected values given the histories of "
                f"{names}, the rest exact given their states, which "
                "vary less than its own; with conditional=True only "
                f"{names} are simulated, for those means at a fraction "
                "of the work, and no spread."
            )
    for name in ("availability", "cost"):
        if out[name].route == r.SIMULATED:
            out[name] = dataclasses.replace(
                out[name], reason=out[name].reason + exact_means, twin=twin
            )
    # The difference of exact expected values needs no simulation
    # (#236): where _exact_means works them out.
    needed = [out["mission_availability"]] + (
        [out["expected_cost"]] if rbd.has_costs else []
    )
    # A diagram too meshed to work out has no exact expected values,
    # so compare simulates it.
    if rbd._too_meshed() is None and all(
        route.route in (r.EXACT, r.NUMERICAL) for route in needed
    ):
        out["compare"] = r.AnalysisRoute(
            (
                r.NUMERICAL
                if any(route.route == r.NUMERICAL for route in needed)
                else r.EXACT
            ),
            "The difference of the two systems' expected values over "
            "the window (mission_availability, and expected_cost their "
            "costs), where the other's are worked out too (#236); "
            "otherwise, or with control_variate=False, the two systems "
            "simulated with common random numbers.",
        )
    elif groups:
        out["compare"] = r.refused(groups)
    elif streamed:
        out["compare"] = r.AnalysisRoute(
            r.SIMULATED,
            "The two systems simulated with common random numbers.",
            engine=engine,
            engine_reason=why,
        )
    else:
        out["compare"] = r.refused(_UNSTREAMED)
    give(
        ("initialize_event_queue", "next_event"),
        (
            r.refused(groups)
            if groups
            else r.AnalysisRoute(
                r.SIMULATED,
                "One simulation, stepped through event by event, drawing "
                "from numpy's global RNG.",
            )
        ),
    )
    out["structural_importance"] = r.AnalysisRoute(
        r.EXACT, "From the structure alone."
    )
    out["system_probability"] = r.AnalysisRoute(
        r.EXACT,
        "The structure function over the node probabilities given.",
    )
    out["path_set_probabilities"] = r.AnalysisRoute(
        r.EXACT, "From the node probabilities given."
    )
    allocation_by_probability = r.AnalysisRoute(
        r.NUMERICAL,
        "A solver over the exact system probability, from the node "
        "probabilities given (not the component models).",
    )
    give(
        (
            "improvement_allocation",
            "equal_allocation",
            "simple_allocation",
            "cost_based_allocation",
        ),
        allocation_by_probability,
    )
    series = r.refusal(rbd._require_series)
    out["minimum_effort_allocation"] = (
        r.refused(series) if series else allocation_by_probability
    )
    # Each component's own values, and the long-run costs but the
    # system's downtime, need no structure; nor the expected cost when
    # nothing is priced.
    free = {"node_availability", "spares_demand", "spares_stock"}
    if not rbd.downtime_cost_rate:
        free |= {"expected_cost_rate", "total_cost"}
    if not rbd.has_costs:
        free.add("expected_cost")
    meshed = rbd._too_meshed()
    if meshed is not None and rbd._has_capacity():
        # The simulations that follow the system's capacity do so
        # through its reduced diagram, which a structure too meshed has
        # not.
        for name in (
            "availability",
            "simulate_chunk",
            "availability_from_chunks",
            "shards",
        ):
            out[name] = r.refused(meshed)
    out = rbd._meshed_routes(out, free)
    # Parameter uncertainty (#200): each draw's diagram rebuilt and its
    # value worked out as the diagram's own, so refused where that is
    # (a structure too meshed included).
    for name, method in (
        ("mean_availability_uncertainty", "mean_availability"),
        ("point_availability_uncertainty", "point_availability"),
        ("mission_availability_uncertainty", "mission_availability"),
        ("expected_cost_rate_uncertainty", "expected_cost_rate"),
    ):
        own = out[method]
        out[name] = (
            own
            if own.route == r.REFUSED
            else r.AnalysisRoute(
                r.SIMULATED,
                "The components' models drawn from their uncertainty, "
                f"and each draw's diagram's {method} worked out as its "
                f"own ({own.route}).",
            )
        )
    sensitivity = out["parameter_sensitivity"]
    out["uncertainty_importance"] = (
        sensitivity
        if sensitivity.route == r.REFUSED
        else r.AnalysisRoute(
            r.NUMERICAL,
            "The delta method: parameter_sensitivity's derivatives in "
            "each uncertain parameter, with their covariance "
            "(method='sobol' estimates Sobol indices from draws "
            "instead).",
        )
    )
    return dict(sorted(out.items()))
