"""The differential importance measure (DIM, Borgonovo & Apostolakis,
2001; #193): each component's or parameter's share of the change in the
system's reliability or availability when they all change together, so
that the shares of any group of them add up.

``DIM_i = dQ/dtheta_i dtheta_i / sum_j dQ/dtheta_j dtheta_j``. With every
``dtheta`` equal (a uniform change, "H1") the shares are those of the
derivatives; with every ``dtheta / theta`` equal (a proportional change,
"H2"), those of ``theta dQ/dtheta``. Over components, ``theta`` is each
one's probability of failing (or of being down), or of working: a uniform
change shares out the Birnbaum importance either way (the two move
together), and a proportional one ``I_B q`` or ``I_B (1 - q)``, the
criticality importance of that kind, shared out. Over parameters, the
derivatives are ``parameter_sensitivity``'s.
"""

from typing import Any, Collection, Dict, Hashable, Mapping, Optional

import numpy as np

#: The changes the shares are of.
CHANGES = ("uniform", "proportional")
#: What they are shares of.
OVER = ("components", "parameters")
#: Which of a node's probabilities a proportional change moves.
KINDS = ("failure", "success")


def check(over: str, change: str, kind: str = "failure") -> None:
    """Raise unless ``over``, ``change`` and ``kind`` are known."""
    if change not in CHANGES:
        raise ValueError(
            f"change must be 'uniform' (every probability or parameter "
            f"moved by as much) or 'proportional' (each by the same "
            f"fraction of itself), got {change!r}."
        )
    if over not in OVER:
        raise ValueError(f"over must be one of {list(OVER)}, got {over!r}.")
    if kind not in KINDS:
        raise ValueError(
            f"kind must be 'failure' (each node's probability of failing "
            f"moved in proportion) or 'success' (of working), got {kind!r}."
        )


def flattened(sensitivities: Mapping) -> Dict[Hashable, Any]:
    """``{(key, parameter): derivative}`` from ``parameter_sensitivity``'s
    ``{key: {parameter: derivative}}``."""
    return {
        (key, name): value
        for key, values in sensitivities.items()
        for name, value in values.items()
    }


def shares(
    contributions: Mapping[Hashable, Any],
    groups: Optional[Mapping[Hashable, Collection[Hashable]]] = None,
    scalar: bool = True,
    improving: bool = False,
) -> dict:
    """Each contribution over their total (NaN where it is 0, or rounding
    against the contributions, which cancel), or, with ``groups``, each
    group's sum of its members' shares: floats where ``scalar``, else
    arrays. ``improving`` takes each contribution's size, as if each
    ``theta`` moved the way that improves the system."""
    values = {
        key: (np.abs if improving else np.asarray)(
            np.asarray(value, dtype=float)
        )
        for key, value in contributions.items()
    }
    total: Any = np.zeros(())
    size: Any = np.zeros(())
    for value in values.values():
        total = total + value
        size = size + np.abs(value)
    # Changes that cancel (an exponential unit's failure and repair rates,
    # moved in proportion, leave its availability as it was) leave a total
    # of rounding (or of the derivatives' differences, to about 1e-6 of
    # them), which shares nothing out.
    cancelled = np.abs(total) <= 1e-6 * size
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            key: np.where(cancelled, np.nan, value / total)
            for key, value in values.items()
        }
    if groups is not None:
        if not isinstance(groups, Mapping):
            raise TypeError(
                "groups must map each group's name to the keys it adds up, "
                f"got {type(groups).__name__}."
            )
        summed = {}
        for name, members in groups.items():
            members = list(members)
            unknown = [key for key in members if key not in out]
            if unknown:
                raise ValueError(
                    f"Group {name!r} names {unknown!r}, which are not among "
                    f"the keys shared out: {sorted(out, key=str)!r}."
                )
            group: Any = np.zeros(())
            for key in members:
                group = group + out[key]
            summed[name] = group
        out = summed
    if scalar:
        return {key: float(np.ravel(value)[0]) for key, value in out.items()}
    return out


def proportional_scales(levers, keys) -> Dict[Hashable, float]:
    """What a proportional change moves each of ``keys`` (``(key, name)``
    of ``levers``) by, per unit of the change (see
    ``_sensitivity._proportional_scale``): raise for a lever that has no
    such change, a regression node's covariate."""
    scales = {(lever.key, lever.name): lever.scale for lever in levers}
    unitless = [key for key in keys if scales[key] is None]
    if unitless:
        raise ValueError(
            f"Lever(s) {unitless} are covariates, whose zero is their "
            "unit's (0 degrees C is not 0 K), so a proportional change of "
            "them depends on the unit they are given in: take "
            "change='uniform', or the parameters of a diagram without them."
        )
    return {key: float(scales[key]) for key in keys}
