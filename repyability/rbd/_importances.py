"""Every importance measure in one call, as a table.

``importances`` on each model type calls the model's own measure methods
one after another and gathers them in an
[`ImportanceTable`][repyability.ImportanceTable]. Nothing is computed here:
each measure is stated once, in its method, and the table holds what the
method returned. What the model refuses is found from ``analysis_routes()``,
so the table names the same reasons the methods raise.
"""

from typing import Callable, Dict

from repyability.rbd import routes as r
from repyability.rbd.results import ImportanceTable

#: The measures a table holds, in order. A model has the ones it has a
#: method for.
MEASURES = (
    "birnbaum_importance",
    "improvement_potential",
    "risk_achievement_worth",
    "risk_reduction_worth",
    "criticality_importance",
    "fussell_vesely",
    "differential_importance",
    "joint_importance",
    "barlow_proschan_importance",
    "structural_importance",
)

SIMULATED_REASON = "simulated: pass simulate=True"


def _sorted(report: Dict[str, r.AnalysisRoute]) -> "r.AnalysisRoutes":
    return r.AnalysisRoutes(sorted(report.items()))


def _refusal(refused: Dict[str, str]) -> str:
    return "No importance measure can be worked out here: " + "; ".join(
        f"{name}: {reason}" for name, reason in refused.items()
    )


def route(report: Dict[str, r.AnalysisRoute]) -> r.AnalysisRoute:
    """The route of ``importances``: the worst of the measures it works
    out (the exact and numerical ones; a simulated one only with
    ``simulate=True``), or a refusal when it can work out none."""
    names = [n for n in MEASURES if n in report]
    worked = [n for n in names if report[n].route in (r.EXACT, r.NUMERICAL)]
    simulated = [n for n in names if report[n].route == r.SIMULATED]
    refused = {
        n: report[n].reason for n in names if report[n].route == r.REFUSED
    }
    if not worked and not simulated:
        return r.refused(_refusal(refused))
    left = ""
    if refused:
        left += " Refused, and listed with their reasons: " + ", ".join(
            refused
        )
        left += "."
    if simulated:
        left += (
            " Simulated, and run only with simulate=True: "
            + ", ".join(simulated)
            + "."
        )
    if not worked:
        return r.AnalysisRoute(
            r.SIMULATED,
            "Only simulated measures are left; with simulate=True each "
            "is run." + left,
        )
    worst = max((report[n] for n in worked), key=lambda x: r._RANK[x.route])
    nodes = tuple(dict.fromkeys(n for m in worked for n in report[m].nodes))
    return r.AnalysisRoute(
        worst.route,
        "Each measure's own method in turn, each as its route says "
        "(the worst of them is this one's)." + left,
        nodes,
    )


def with_importances(report: Dict[str, r.AnalysisRoute]) -> "r.AnalysisRoutes":
    """``report`` with the route of ``importances`` added."""
    out = dict(report)
    out["importances"] = route(out)
    return _sorted(out)


def table(
    model, calls: Dict[str, Callable[[], dict]], simulate: bool
) -> ImportanceTable:
    """The table of the measures in ``calls`` (a zero-argument call of each
    method the model has), those ``model.analysis_routes()`` refuses listed
    with their reasons, and those it simulates run only when ``simulate``."""
    from repyability.utils.checks import one_of

    one_of("simulate", simulate, (True, False))
    report = model.analysis_routes()
    measures: Dict[str, dict] = {}
    refused: Dict[str, str] = {}
    simulated = False
    for name in MEASURES:
        if name not in calls:
            continue
        how = report[name]
        if how.route == r.REFUSED:
            refused[name] = how.reason
        elif how.route == r.SIMULATED and not simulate:
            refused[name] = SIMULATED_REASON
            simulated = True
        else:
            measures[name] = calls[name]()
    if not measures and not simulated:
        raise NotImplementedError(_refusal(refused))
    return ImportanceTable(measures=measures, refused=refused)
