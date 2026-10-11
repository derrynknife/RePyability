"""The README's capability matrices, made from the code (see
``test_readme_capabilities.py``, which checks the README holds what this
makes).

The first says which analyses each model type has: a public method of
the class, or a gap. The others say, for each kind of node or component
a diagram can hold, how each family of analyses is worked out on a
diagram with it (``analysis_routes()`` on the catalogue's diagrams): a
refusal that the diagram has nothing the analysis works on
(``routes.Refused``: nothing priced, no capacities, ...) is no gap, and
is left out; one the analysis cannot work out (``NotImplementedError``)
is.

Run ``python -m repyability.tests.capabilities`` to write them into the
README, between its ``capabilities`` markers.
"""

import pathlib
import warnings
from typing import Dict, List, Tuple

from repyability import (
    FaultTree,
    Network,
    NonRepairableRBD,
    PhasedMission,
    RepairableRBD,
)
from repyability.rbd import routes

START = "<!-- capabilities:start -->"
END = "<!-- capabilities:end -->"

MODELS = {
    "NonRepairableRBD": NonRepairableRBD,
    "RepairableRBD": RepairableRBD,
    "FaultTree": FaultTree,
    "Network": Network,
    "PhasedMission": PhasedMission,
}

#: Each capability, by section: for each model type, the methods that give
#: it (the first a model has is shown), or none.
CAPABILITIES: List[Tuple[str, List[Tuple[str, Tuple[str, ...]]]]] = [
    (
        "Reliability and lifetime",
        [
            ("Reliability R(t)", ("sf", "reliability")),
            ("Unreliability F(t)", ("ff", "unreliability")),
            ("Density, hazard, cumulative hazard", ("df", "hf", "Hf")),
            ("Mean time to (first) failure", ("mean", "mean_time_to_failure")),
            (
                "Each node's own R(t), F(t) and MTTF",
                ("node_sf", "node_ff", "node_mttf"),
            ),
            (
                "B-life, time to a reliability",
                ("bx_life", "time_to_reliability"),
            ),
            ("Given the system's age", ("cs",)),
            ("Given the components' ages or states", ("sf_given_state",)),
            (
                "Mean residual and remaining life",
                ("mean_residual_life", "remaining_life"),
            ),
            ("Random lifetimes", ("random", "random_block")),
            ("Top event probability", ("top_event_probability",)),
            (
                "Each phase's failure probability",
                ("phase_failure_probabilities",),
            ),
        ],
    ),
    (
        "Availability and outages",
        [
            ("Long-run availability", ("mean_availability",)),
            ("Each node's availability", ("node_availability",)),
            (
                "Availability over time",
                ("point_availability", "point_unavailability"),
            ),
            (
                "Over a mission",
                ("mission_availability", "mission_unavailability"),
            ),
            ("Unavailability (precise when small)", ("mean_unavailability",)),
            (
                "Failure frequency, MTBF",
                ("system_failure_frequency", "mean_time_between_failures"),
            ),
            ("Mean up and down times", ("mean_up_time", "mean_down_time")),
            (
                "Expected failures and events in a window",
                ("expected_events", "expected_failures"),
            ),
            (
                "Simulated availability (distribution)",
                (
                    "availability",
                    "availability_from_chunks",
                    "simulate_chunk",
                    "shards",
                ),
            ),
            ("Simulated histories", ("simulate_timelines",)),
            ("Compare two designs", ("compare",)),
        ],
    ),
    (
        "Costs and spares",
        [
            ("Long-run cost rate", ("expected_cost_rate",)),
            ("Expected cost in a window", ("expected_cost",)),
            ("Simulated cost (distribution)", ("cost",)),
            ("Total cost of ownership", ("total_cost",)),
            ("Spares demand and stock", ("spares_demand", "spares_stock")),
        ],
    ),
    (
        "Structure",
        [
            (
                "Minimal path and cut sets",
                (
                    "minimal_cut_sets",
                    "cut_sets",
                    "minimal_path_sets",
                    "path_sets",
                    "get_min_cut_sets",
                    "get_min_path_sets",
                    "get_all_path_sets",
                ),
            ),
            ("Each path set's probability", ("path_set_probabilities",)),
            ("Cut sets ranked by probability", ("ranked_cut_sets",)),
            ("Irrelevant components", ("find_irrelevant_components",)),
            ("Convert to and from an RBD", ("to_rbd", "from_rbd")),
            (
                "Capacity distribution",
                ("capacity_distribution", "system_capacity"),
            ),
            ("Capacity over time", ("point_capacity", "mission_capacity")),
            (
                "How each analysis is computed",
                (
                    "analysis_routes",
                    "is_analytically_solvable",
                    "get_non_analytic_nodes",
                ),
            ),
            (
                "Save and load (dict, JSON)",
                ("to_dict", "to_json", "from_dict", "from_json"),
            ),
        ],
    ),
    (
        "Importance measures",
        [
            ("Birnbaum", ("birnbaum_importance",)),
            ("Structural", ("structural_importance",)),
            ("Improvement potential", ("improvement_potential",)),
            ("Risk achievement worth", ("risk_achievement_worth",)),
            ("Risk reduction worth", ("risk_reduction_worth",)),
            ("Criticality", ("criticality_importance",)),
            ("Fussell-Vesely", ("fussell_vesely",)),
            ("Differential (DIM)", ("differential_importance",)),
            ("Joint (second order)", ("joint_importance",)),
            ("Barlow-Proschan", ("barlow_proschan_importance",)),
            (
                "Rate of change over time",
                ("reliability_rate", "availability_rate"),
            ),
            ("Uncertainty importance", ("uncertainty_importance",)),
            ("All measures given a state", ("importances_given_state",)),
            (
                "Parameter sensitivity, levers",
                ("parameter_sensitivity", "levers", "with_levers"),
            ),
        ],
    ),
    (
        "Uncertainty",
        [
            (
                "Of the reliability or availability",
                (
                    "sf_uncertainty",
                    "mean_availability_uncertainty",
                    "reliability_interval",
                    "unreliability_interval",
                    "point_availability_uncertainty",
                    "mission_availability_uncertainty",
                ),
            ),
            (
                "Of the MTTF",
                ("mean_uncertainty", "mean_time_to_failure_interval"),
            ),
            (
                "Of the B-life",
                ("bx_life_uncertainty", "time_to_reliability_uncertainty"),
            ),
            ("Of the cost rate", ("expected_cost_rate_uncertainty",)),
        ],
    ),
    (
        "Design and maintenance",
        [
            (
                "Reliability allocation",
                (
                    "simple_allocation",
                    "equal_allocation",
                    "cost_based_allocation",
                    "improvement_allocation",
                    "minimum_effort_allocation",
                ),
            ),
            (
                "Availability, MTTF and MTTR allocation",
                ("availability_allocation", "mttf_mttr_allocation"),
            ),
            ("Redundancy allocation", ("allocate_redundancy",)),
            (
                "Reliability-redundancy allocation",
                ("allocate_reliability_redundancy",),
            ),
            ("Cost-reliability front", ("redundancy_front",)),
            (
                "Optimal replacement intervals",
                ("optimal_replacement_intervals", "with_intervals"),
            ),
            (
                "Optimal proof-test intervals",
                ("optimal_inspection_intervals",),
            ),
        ],
    ),
]

#: Public methods that are not analyses: the structure evaluated at one
#: state, the event loop's steps and the node names.
NOT_ANALYSES = frozenset(
    {
        "initialize_event_queue",
        "is_system_working",
        "next_event",
        "node_names",
        "occurs",
        "system_probability",
        "system_timeline",
    }
)

#: The families of analyses the node tables give, by model type: the
#: methods of each.
FAMILIES: Dict[str, List[Tuple[str, Tuple[str, ...]]]] = {
    "NonRepairableRBD": [
        ("Reliability", ("sf", "ff")),
        ("MTTF", ("mean",)),
        ("B-life", ("bx_life", "time_to_reliability")),
        ("Given a state", ("sf_given_state", "mean_residual_life")),
        (
            "Importance",
            (
                "birnbaum_importance",
                "criticality_importance",
                "fussell_vesely",
                "risk_achievement_worth",
                "risk_reduction_worth",
                "improvement_potential",
                "differential_importance",
                "joint_importance",
                "barlow_proschan_importance",
            ),
        ),
        ("Sensitivity", ("parameter_sensitivity",)),
        ("Uncertainty", ("sf_uncertainty", "mean_uncertainty")),
        ("Allocation", ("allocate_redundancy", "simple_allocation")),
        ("Capacity", ("capacity_distribution",)),
        ("Simulation", ("random",)),
    ],
    "RepairableRBD": [
        ("Long run", ("mean_availability", "mean_unavailability")),
        (
            "Frequency",
            (
                "system_failure_frequency",
                "mean_up_time",
                "mean_down_time",
            ),
        ),
        (
            "Over time",
            ("point_availability", "mission_availability", "expected_events"),
        ),
        ("Costs", ("expected_cost_rate", "expected_cost")),
        (
            "Importance",
            (
                "birnbaum_importance",
                "criticality_importance",
                "fussell_vesely",
                "risk_achievement_worth",
                "risk_reduction_worth",
                "improvement_potential",
                "differential_importance",
                "joint_importance",
                "barlow_proschan_importance",
                "availability_rate",
            ),
        ),
        ("Sensitivity", ("parameter_sensitivity",)),
        (
            "Uncertainty",
            (
                "mean_availability_uncertainty",
                "expected_cost_rate_uncertainty",
            ),
        ),
        (
            "Allocation",
            ("allocate_redundancy", "availability_allocation"),
        ),
        (
            "Intervals",
            ("optimal_replacement_intervals", "optimal_inspection_intervals"),
        ),
        ("Spares", ("spares_demand", "spares_stock")),
        ("Capacity", ("capacity_distribution", "point_capacity")),
        ("Simulation", ("availability", "cost", "simulate_timelines")),
    ],
}

#: The node tables' rows: a kind of node or component, and the catalogue's
#: diagram with one.
NODES: Dict[str, List[Tuple[str, str]]] = {
    "NonRepairableRBD": [
        ("A surpyval model", "plain"),
        ("A fitted model (with its covariance)", "fitted"),
        ("A fit with a cure share and zero-inflation", "fitted shares"),
        ("A probability per demand", "fixed"),
        ("A mixture (`MixtureModel`)", "mixture"),
        ("A regression node (covariates)", "regression"),
        ("A degrading node", "degrading"),
        ("A fitted degradation process", "degradation process"),
        ("Cold standby (convolved)", "convolved standby"),
        ("Standby with no reliability of its own", "simulated standby"),
        ("Load sharing", "load sharing"),
        ("A repeated node", "repeated"),
        ("A repeated standby node", "repeated standby"),
        ("A nested RBD", "nested, worked out"),
        ("A junction", "junction in series"),
        ("A beta-factor common-cause group", "common cause"),
        ("An MGL common-cause group", "common cause, MGL"),
        ("Capacities", "capacities"),
        ("A structure too meshed to work out", "too meshed"),
    ],
    "RepairableRBD": [
        ("Any life and repair", "costed_pairs"),
        ("A fitted life", "fitted life"),
        ("A fitted Crow-AMSAA process", "fitted Crow-AMSAA process"),
        ("A probability per demand", "fixed probability"),
        ("A mixture life", "mixture life"),
        ("A fitted degradation process life", "degradation process life"),
        ("Operated part of the time (duty)", "operated part of the time"),
        (
            "A standby or load-sharing life, simulated",
            "simulated standby life",
        ),
        ("k-out-of-n", "koon"),
        ("A nested RBD", "nested_one_level"),
        ("A junction", "junction, bought"),
        ("Capacities", "capacities"),
        ("Age replacement", "age replacement, priced"),
        ("Block replacement", "block replacement"),
        ("Opportunistic maintenance", "opportunistic maintenance"),
        ("Replacement on condition", "replaced on condition"),
        (
            "Replacement on condition by level",
            "replaced on condition by level",
        ),
        ("Hidden failures, tested", "tested, Weibull"),
        ("Tests that take time", "tested, taking time"),
        ("Tests that can miss a failure", "tested, missing"),
        ("Imperfect repair (Kijima)", "imperfect repair"),
        ("Minimal repair", "minimal repair in no time"),
        ("Limited repair crews", "one repair crew"),
        ("A standby group", "standby group"),
        ("A beta-factor group, revealed failures", "common cause, revealed"),
        ("A beta-factor group, tested", "common cause, tested"),
        (
            "A group whose tests take time",
            "common cause, timed tests and repairs",
        ),
        ("A structure too meshed to work out", "too meshed"),
    ],
}

SYMBOLS = {
    routes.EXACT: "E",
    routes.NUMERICAL: "N",
    routes.SIMULATED: "S",
}


def _model_table() -> str:
    names = list(MODELS)
    lines = [
        "| | " + " | ".join(f"`{n}`" for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for section, rows in CAPABILITIES:
        lines.append(f"| **{section}** |" + " |" * len(names))
        for label, methods in rows:
            cells = []
            for cls in MODELS.values():
                have = [m for m in methods if callable(getattr(cls, m, None))]
                cells.append(f"`{have[0]}`" if have else "—")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _gap_kind(call, rbd) -> bool:
    """Whether a refused analysis is a gap: the method raises
    ``NotImplementedError``, not ``routes.Refused`` (nothing to work on)."""
    try:
        call(rbd)
    except routes.Refused:
        return False
    except NotImplementedError:
        return True
    return True


def _node_table(model: str) -> str:
    from repyability.tests.catalogue import (
        nonrepairable_kinds,
        repairable_kinds,
    )
    from repyability.tests.test_analysis_routes import (
        NONREPAIRABLE_CALLS,
        REPAIRABLE_CALLS,
    )

    kinds, calls = (
        (nonrepairable_kinds(), NONREPAIRABLE_CALLS)
        if model == "NonRepairableRBD"
        else (repairable_kinds(), REPAIRABLE_CALLS)
    )
    families = FAMILIES[model]
    lines = [
        "| | " + " | ".join(name for name, _ in families) + " |",
        "|---|" + "---|" * len(families),
    ]
    for label, kind in NODES[model]:
        rbd = kinds[kind]
        report = rbd.analysis_routes()
        cells = []
        for _, methods in families:
            seen, gaps = [], []
            for method in methods:
                route = report[method]
                if route.route != routes.REFUSED:
                    if SYMBOLS[route.route] not in seen:
                        seen.append(SYMBOLS[route.route])
                elif _gap_kind(calls[method], rbd):
                    gaps.append(method)
            seen.sort(key="ENS".index)
            cell = " ".join(seen)
            if gaps:
                if len(gaps) == len(methods):
                    refused = "✗"
                elif len(gaps) <= 2:
                    refused = "✗ " + ", ".join(f"`{m}`" for m in gaps)
                else:
                    refused = f"✗ {len(gaps)} of {len(methods)}"
                cell = f"{cell}; {refused}" if cell else refused
            cells.append(cell or "–")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def matrix() -> str:
    """The README's capability section, between its markers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nonrepairable = _node_table("NonRepairableRBD")
        repairable = _node_table("RepairableRBD")
    return "\n".join(
        [
            START,
            "",
            "Which analyses each model type has (the method; — where it "
            "has none):",
            "",
            _model_table(),
            "",
            "How each family of analyses is worked out on a diagram with "
            "each kind of node or component: **E** exact, **N** numerical "
            "(to a stated accuracy), **S** simulated, **✗** refused as not "
            "worked out (naming the methods, or counting them, where only "
            "some are), – where the diagram has nothing the analysis works "
            "on. Made from `analysis_routes()` on the test "
            "catalogue's diagrams.",
            "",
            "<details><summary><code>NonRepairableRBD</code>, by node"
            "</summary>",
            "",
            nonrepairable,
            "",
            "</details>",
            "",
            "<details><summary><code>RepairableRBD</code>, by component"
            "</summary>",
            "",
            repairable,
            "",
            "</details>",
            "",
            END,
        ]
    )


def write(path: pathlib.Path) -> None:
    """Put ``matrix()`` into the README at ``path``, between its markers."""
    text = path.read_text()
    start, end = text.index(START), text.index(END) + len(END)
    path.write_text(text[:start] + matrix() + text[end:])


if __name__ == "__main__":
    write(pathlib.Path(__file__).resolve().parents[2] / "README.md")
