"""Deprecated names, arguments and models.

Old argument names and ignored arguments keep working until 1.0 (#105),
each with a ``DeprecationWarning`` pointing at the caller's line.

Non-parametric RBD nodes, and the fits to simulated lifetimes that stand in
for some standby and load-sharing models' reliability, are deprecated in
0.11 and go in 0.12 (``REMOVAL``): one minor release's notice. They warn
with a ``FutureWarning``, which Python always shows, as the notice is short.
"""

import warnings
from typing import Any, Dict, List

#: The release that removes what 0.11 deprecates.
REMOVAL = "0.12"


def renamed(
    new: str, value: Any, old: str, old_value: Any, stacklevel: int = 3
) -> Any:
    """The value of an argument renamed from ``old`` to ``new``.

    ``value`` if the new name was used, else ``old_value`` with a
    ``DeprecationWarning``. Both default to None in the signature, so an
    unused name is None.

    Parameters
    ----------
    new, old : str
        The argument's new and old names.
    value, old_value : Any
        What was passed under each name (None if nothing).
    stacklevel : int, optional
        Passed to ``warnings.warn``: 3 points at the caller of the function
        that calls ``renamed``.

    Returns
    -------
    Any
        The value to use.

    Raises
    ------
    TypeError
        If both names were used.
    """
    if old_value is None:
        return value
    if value is not None:
        raise TypeError(f"Give {new} or its old name {old}, not both.")
    warnings.warn(
        f"{old} is deprecated: use {new}. {old} will be removed in 1.0.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return old_value


def ignored(method: str, why: str, given: Dict[str, Any]) -> None:
    """Warn that the arguments in ``given`` that were passed (not None or
    False) are ignored by ``method``, for the reason ``why``."""
    passed = [
        name
        for name, value in given.items()
        if value is not None and value is not False
    ]
    if passed:
        warnings.warn(
            f"{method} ignores {', '.join(passed)}: {why} Passing "
            f"{'it' if len(passed) == 1 else 'them'} is deprecated.",
            DeprecationWarning,
            stacklevel=3,
        )


def _nonparametric(model) -> bool:
    """Whether a node's model is, or is built from, a surpyval
    non-parametric fit given by the user. A standby or load-sharing model's
    own fit to its simulated lifetimes is not one (see
    ``warn_simulated_fit``), and a nested RBD warns when it is built."""
    from surpyval import NonParametric

    from repyability.non_repairable import NonRepairable
    from repyability.rbd.degrading_node import DegradingNode
    from repyability.rbd.repeated_node import RepeatedNode
    from repyability.rbd.repeated_standby_node import RepeatedStandbyNode
    from repyability.rbd.standby_node import StandbyModel

    if isinstance(model, NonParametric):
        return True
    if isinstance(model, StandbyModel):
        return any(_nonparametric(unit) for unit in model.reliabilities)
    if isinstance(model, (RepeatedNode, RepeatedStandbyNode)):
        return _nonparametric(model.model)
    if isinstance(model, DegradingNode):
        return any(_nonparametric(stage) for _, stage in model.stages)
    if isinstance(model, NonRepairable):
        return _nonparametric(model.reliability) or _nonparametric(
            model.time_to_replace
        )
    if isinstance(model, dict):  # a RepairableRBD component spec
        return any(_nonparametric(value) for value in model.values())
    if isinstance(model, (list, tuple)):
        return any(_nonparametric(value) for value in model)
    return False


def nonparametric_nodes(models: Dict[Any, Any]) -> List[Any]:
    """The nodes whose models (or component specs) hold a surpyval
    non-parametric fit, in order."""
    return [node for node, model in models.items() if _nonparametric(model)]


def warn_nonparametric(nodes: List[Any], stacklevel: int = 3) -> None:
    """Warn that non-parametric nodes are deprecated, if ``nodes`` has
    any."""
    if not nodes:
        return
    warnings.warn(
        f"Node(s) {nodes} use a non-parametric fit (Kaplan-Meier or "
        "similar) as a lifetime or repair time. Non-parametric nodes are "
        f"deprecated and will be removed in {REMOVAL}: their curves end at "
        "the data, so the MTTF, B-lives and long-run values beyond it are "
        "artefacts, and their draws cannot be paired or streamed in "
        "simulations. Fit a parametric distribution in surpyval (e.g. "
        "surpyval.Weibull.fit(times)) and use that.",
        FutureWarning,
        stacklevel=stacklevel,
    )


def warn_simulated_fit(model: str, case: str, stacklevel: int = 3) -> None:
    """Warn that ``model``'s reliability, for ``case``, is a fit to
    simulated lifetimes, and that the fit is deprecated."""
    warnings.warn(
        f"This {model} ({case}) has no exact or numerical reliability, so "
        "its sf is a Kaplan-Meier fit to simulated lifetimes. That fit is "
        f"deprecated and will be removed in {REMOVAL}: the model will still "
        "draw lifetimes for simulations, but analyses that need its "
        "reliability will refuse, and the system's simulations (random, "
        "mean(method='simulate'), unreliability_interval, availability) "
        "will simulate it directly.",
        FutureWarning,
        stacklevel=stacklevel,
    )
