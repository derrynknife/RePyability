"""How each analysis of an RBD is computed, found without running it.

An analysis is computed in one of four ways:

- **exact**: closed forms, or the exact structure function over exact
  node values;
- **numerical**: deterministic numerical methods (a convolution or a
  renewal equation on a grid, quadrature, root-finding, differences, an
  optimiser), which give the same result every time, to a small, stated
  error;
- **simulated**: Monte Carlo, reproducible with a seed;
- **refused**: the method raises.

``NonRepairableRBD.analysis_routes`` and ``RepairableRBD.analysis_routes``
give the route of each of their analyses. They take refusals from the
checks the methods themselves run, so the report and the methods agree.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Optional, Tuple

EXACT, NUMERICAL, SIMULATED, REFUSED = (
    "exact",
    "numerical",
    "simulated",
    "refused",
)
_RANK = {EXACT: 0, NUMERICAL: 1, SIMULATED: 2, REFUSED: 3}


@dataclass(frozen=True)
class AnalysisRoute:
    """How one analysis of an RBD is computed.

    Attributes
    ----------
    route : str
        ``"exact"``, ``"numerical"``, ``"simulated"`` or ``"refused"`` (see
        ``repyability.rbd.routes``).
    reason : str
        Why. For ``"refused"``, the message the method raises.
    nodes : tuple
        The nodes that decide the route, if any: those whose values are
        numerical or simulated, or that the method refuses.
    engine : str or None
        For a ``RepairableRBD`` simulation, the engine ``engine="auto"``
        runs a long simulation on: ``"numba"`` or ``"python"`` (see
        ``engine_reason``). None otherwise.
    engine_reason : str
        Why that engine.
    twin : str
        For a ``RepairableRBD``'s simulated availability and cost: the
        exact twin ``control_variate=True`` controls the estimate by (see
        [`ControlVariate`][repyability.ControlVariate]), as what it leaves
        out of the system, or why there is none. "" otherwise.
    """

    route: str
    reason: str
    nodes: Tuple[Hashable, ...] = ()
    engine: Optional[str] = None
    engine_reason: str = ""
    twin: str = ""

    def __str__(self) -> str:
        text = f"{self.route}: {self.reason}"
        if self.engine is not None:
            text += f" Engine: {self.engine} ({self.engine_reason})."
        if self.twin:
            text += f" Exact twin: {self.twin}"
        return text


def refusal(check: Callable[[], object]) -> Optional[str]:
    """The message of the ``NotImplementedError`` or ``ValueError`` that
    ``check`` raises, or None if it passes."""
    try:
        check()
    except (NotImplementedError, ValueError) as error:
        return str(error)
    return None


def model_route(model) -> Tuple[str, str]:
    """How a node model's reliability (its ``sf``) is obtained: the route,
    and a phrase saying how."""
    from surpyval import NonParametric, Parametric

    from .helper_classes import PerfectReliability, PerfectUnreliability
    from .load_sharing_node import LoadSharingModel
    from .non_repairable_rbd import NonRepairableRBD
    from .numerical_convolution import ConvolvedSurvival
    from .repeated_node import RepeatedNode
    from .repeated_standby_node import RepeatedStandbyNode
    from .standby_node import StandbyModel

    if model is PerfectReliability or model is PerfectUnreliability:
        return EXACT, "a constant"
    if isinstance(model, (StandbyModel, LoadSharingModel)):
        if model.is_simulated:
            return (
                SIMULATED,
                f"a Kaplan-Meier fit to {model.mc_samples} simulated "
                "lifetimes",
            )
        if isinstance(model._sf_model, ConvolvedSurvival):
            return NUMERICAL, "a numerical convolution of its units' lives"
        how = getattr(model._sf_model, "how", None)
        if how is not None:
            route = getattr(model._sf_model, "route", NUMERICAL)
            return (EXACT if route == "exact" else NUMERICAL), how
        return EXACT, "a closed form"
    if isinstance(model, RepeatedStandbyNode):
        return NUMERICAL, "a numerical convolution of its copies' lives"
    if isinstance(model, RepeatedNode):
        return model_route(model.model)
    if isinstance(model, NonRepairableRBD):
        inner = {n: model_route(m) for n, m in model.reliabilities.items()}
        worst = max(
            (route for route, _ in inner.values()), key=_RANK.__getitem__
        )
        if worst == EXACT:
            return EXACT, "a nested RBD of exact nodes"
        which = [n for n, (route, _) in inner.items() if route == worst]
        return worst, f"a nested RBD with {worst} nodes {_names(which)}"
    if isinstance(model, Parametric):
        return EXACT, "a closed form"
    if isinstance(model, NonParametric):
        return EXACT, "its fitted curve"
    return EXACT, "its own sf"


def mean_route(model) -> Tuple[str, str]:
    """How a node model's mean lifetime (its ``mean()``) is found: the route,
    and a phrase saying how, or the message its ``mean()`` would raise."""
    from .degrading_node import DegradingNode
    from .load_sharing_node import LoadSharingModel
    from .non_repairable_rbd import NonRepairableRBD
    from .regression_node import RegressionNode
    from .repeated_node import RepeatedNode
    from .standby_node import StandbyModel

    if isinstance(model, NonRepairableRBD):
        message = refusal(model._require_lifetimes)
        if message:
            return REFUSED, message
    if isinstance(model, (NonRepairableRBD, RepeatedNode)):
        route, how = model_route(model)
        if route == EXACT:
            return NUMERICAL, "the area under its exact reliability"
        return route, f"the area under its reliability, from {how}"
    if isinstance(model, DegradingNode):
        return EXACT, "the sum of its stages' mean times"
    if isinstance(model, (StandbyModel, LoadSharingModel)):
        if model.is_simulated:
            return (
                SIMULATED,
                f"the mean of the {model.mc_samples} lifetimes simulated "
                "when it was built",
            )
        sf_model = model._sf_model
        if getattr(sf_model, "route", None) == EXACT:
            return (
                NUMERICAL,
                "the area under its exact reliability "
                f"({getattr(sf_model, 'how', '')})",
            )
        return model_route(model)
    if isinstance(model, RegressionNode):
        return NUMERICAL, "the mean of its survival function on a grid"
    return EXACT, "its model's mean"


def with_nodes(
    route: str,
    reason: str,
    nodes: Dict[Hashable, Tuple[str, str]],
    effect: str = "values",
) -> AnalysisRoute:
    """An analysis computed from the nodes' ``effect`` (their reliabilities,
    say) by a method of ``route``: numerical or simulated too when some
    nodes' values are, naming them."""
    for worse in (SIMULATED, NUMERICAL):
        which = {n: how for n, (r, how) in nodes.items() if r == worse}
        if which and _RANK[worse] > _RANK[route]:
            detail = "; ".join(f"{n!r}: {how}" for n, how in which.items())
            return AnalysisRoute(
                worse,
                f"{reason} Some nodes' {effect} are {worse} ({detail}).",
                tuple(which),
            )
    return AnalysisRoute(route, reason)


def refused(message: str, nodes=()) -> AnalysisRoute:
    return AnalysisRoute(REFUSED, message, tuple(nodes))


def _names(nodes) -> str:
    return ", ".join(repr(n) for n in nodes)
