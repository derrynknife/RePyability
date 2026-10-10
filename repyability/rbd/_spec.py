"""A ``RepairableRBD``'s inputs, checked: the keys a component's spec may
carry and the rules each takes (preventive maintenance, inspection,
standby groups, imperfect repair, costs), the maintenance groups and the
common-cause groups. The constructor calls these; the keys are also the
class's attributes (``RepairableRBD.PREVENTIVE_KEYS``, ...).
"""

import math
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Hashable,
    Optional,
    Tuple,
)

import numpy as np

from repyability.rbd import _requirements
from repyability.rbd._degradation import (
    is_degradation,
)
from repyability.rbd._model_utils import (
    is_mixture,
)

if TYPE_CHECKING:
    pass
from repyability.rbd._events import (
    _Imperfect,
    _Inspection,
    _MaintenanceGroup,
    _Preventive,
    _Standby,
)
from repyability.utils.checks import (
    is_number,
    number_or_nan,
    unfitted_distribution,
)

#: Optional per-component cost fields accepted in a component spec dict.
COST_KEYS = ("repair_cost", "replace_cost", "downtime_cost")

#: The costs charged per failure, which may also be given as a
#: distribution of the cost (drawn afresh at each failure).
PER_FAILURE_COST_KEYS = ("repair_cost", "replace_cost")

#: Every key a component spec dict may carry.
COMPONENT_SPEC_KEYS = (
    ("reliability", "repairability")
    + COST_KEYS
    + (
        "preventive",
        "inspection",
        "acquisition_cost",
        "priority",
        "standby",
        "group",
        "repair",
        "replace_after",
        "duty",
    )
)

#: The models of imperfect repair a spec's ``"repair"`` can name.
REPAIR_MODELS = ("kijima1", "kijima2")

#: The keys of a component's ``"preventive"`` spec.
PREVENTIVE_KEYS = (
    "interval",
    "policy",
    "duration",
    "cost",
    "threshold",
    "inspection_cost",
    "opportunity",
    "level",
)

#: The policies a ``"preventive"`` spec can name: replacement at an
#: age, on a calendar (block), or on the condition an inspection finds.
PREVENTIVE_POLICIES = ("age", "block", "condition")

#: The keys of a maintenance group's options.
GROUP_KEYS = ("setup_cost", "system_down")

#: The keys of a component's ``"inspection"`` spec.
INSPECTION_KEYS = (
    "interval",
    "duration",
    "cost",
    "offset",
    "coverage",
    "full_test",
)

#: The keys of a component's ``"standby"`` spec.
STANDBY_KEYS = ("units", "k", "dormancy_factor", "switching_probability")

#: The costs charged per action (per failure, per preventive
#: replacement or per inspection), which may be distributions.
_PER_ACTION_COST_KEYS = PER_FAILURE_COST_KEYS + (
    "preventive_cost",
    "inspection_cost",
)


#: The share of its interval below which a test offset is taken as 0: the
#: first test at the interval, not one at the start of a new unit (#237).
_TINY_OFFSET = 1e-9


def _mean_cost(cost) -> float:
    """A declared cost's expected value: the number itself, or the mean of a
    cost distribution."""
    if isinstance(cost, float):
        return cost
    return float(np.ravel(cost.mean())[0])


def _validate_ccf_groups(rbd, ccf_groups) -> list:
    """The common-cause groups, checked: each a ``CCFGroup`` of
    components of this RBD (not a nested RBD or a standby group), each
    component in one group at most, the members of a group identical
    (the same life and repair models, compared through their saved
    form)."""
    if not ccf_groups:
        return []
    from repyability.rbd.ccf import checked_groups
    from repyability.rbd.repairable_rbd import RepairableRBD
    from repyability.rbd.serialisation import serialise_model

    def check_member(member, group):
        if member not in rbd.components:
            raise _requirements._not_a_component(
                rbd, member, f"common-cause group {list(group.members)}"
            )
        if (
            isinstance(rbd.components[member], RepairableRBD)
            or member in rbd._standby
        ):
            raise ValueError(
                f"CCF group member {member!r} is a nested RBD or a "
                "standby group: a group's members are single "
                "components."
            )

    def saved(member):
        component = rbd.components[member]
        return (
            serialise_model(component.reliability),
            serialise_model(component.time_to_replace),
        )

    return checked_groups(
        ccf_groups,
        check_member,
        saved,
        lambda group: (
            f"The members of a CCF group, {list(group.members)}, "
            "must be identical components, with the same life and "
            "repair models."
        ),
    )


def _validate_imperfect(rbd, node, spec: dict) -> Optional[_Imperfect]:
    """A component spec's imperfect repair (``"repair"`` and
    ``"replace_after"``), validated; None if it is renewed by every
    repair, as by default (a restoration factor of 0)."""
    repair, limit = spec.get("repair"), spec.get("replace_after")
    if repair is None and limit is None:
        return None
    q, model = 0.0, ""
    if repair is not None:
        if (
            not isinstance(repair, dict)
            or set(repair) != {"model", "q"}
            or repair["model"] not in REPAIR_MODELS
        ):
            raise ValueError(
                f"Component {node!r}: repair must be a dict of a "
                f"'model' ({' or '.join(map(repr, REPAIR_MODELS))}) "
                f"and its restoration factor 'q', got {repair!r}."
            )
        q = repair["q"]
        if (
            isinstance(q, bool)
            or not isinstance(q, (int, float, np.number))
            or not 0.0 <= float(q) <= 1.0
        ):
            raise ValueError(
                f"Component {node!r}: the restoration factor q must be "
                "a number in [0, 1] (0 renews the unit, 1 is minimal "
                f"repair), got {q!r}."
            )
        q, model = float(q), repair["model"]
    if limit is not None:
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, np.integer))
            or limit < 1
        ):
            raise ValueError(
                f"Component {node!r}: replace_after must be a whole "
                f"number of failures of at least 1, got {limit!r}."
            )
        if q == 0.0:
            raise ValueError(
                f"Component {node!r}: replace_after needs imperfect "
                "repair (a 'repair' with q above 0): renewed by every "
                "repair, the unit is as new after each failure anyway."
            )
        limit = int(limit)
    if q == 0.0:
        return None
    if spec.get("standby") is not None:
        raise ValueError(
            f"Component {node!r} is a standby group, whose units are "
            "renewed by their repairs, so it cannot be repaired "
            "imperfectly."
        )
    preventive = spec.get("preventive")
    if isinstance(preventive, dict) and preventive.get("policy") == (
        "condition"
    ):
        raise ValueError(
            f"Component {node!r} is replaced on condition, judged by its "
            "age as new, so it cannot be repaired imperfectly."
        )
    life = spec["reliability"]
    if not (is_mixture(life) or (hasattr(life, "Hf") and hasattr(life, "qf"))):
        raise ValueError(
            f"Component {node!r}: an imperfectly repaired unit's lives "
            "are drawn given its virtual age, which needs a lifetime "
            "model with a cumulative hazard (Hf) and quantiles (qf), "
            "such as a surpyval distribution."
        )
    return _Imperfect(model, q, limit)


def _validate_groups(
    rbd, options: Optional[dict]
) -> Dict[Hashable, _MaintenanceGroup]:
    """The maintenance groups (#108), from the components' ``"group"``
    and the options given for each, checked; and the members that can
    be renewed early (``_early_members``)."""
    members: Dict[Hashable, list] = {}
    for node, group in rbd._member_group.items():
        try:
            hash(group)
        except TypeError:
            raise ValueError(
                f"Component {node!r}: a group is named by a hashable "
                f"value, got {group!r}."
            ) from None
        for spec_key, what in (
            ("_inspection", "hidden failures"),
            ("_standby", "a standby group"),
        ):
            if node in getattr(rbd, spec_key):
                raise ValueError(
                    f"Component {node!r} has {what}, so it cannot be in "
                    "a maintenance group."
                )
        members.setdefault(group, []).append(node)
    rbd._early_members = {
        node
        for node, schedule in rbd._preventive.items()
        if schedule.opportunity < schedule.interval
    }
    for node in rbd._early_members:
        if node not in rbd._member_group:
            raise ValueError(
                f"Component {node!r} has an opportunity, but no "
                "maintenance group to give it one: add a 'group'."
            )
    options = dict(options or {})
    unknown = set(options) - set(members)
    if unknown:
        raise ValueError(
            f"maintenance_groups names group(s) "
            f"{sorted(map(str, unknown))} that no component is in."
        )
    out: Dict[Hashable, _MaintenanceGroup] = {}
    for group, nodes in members.items():
        given = options.get(group) or {}
        if not isinstance(given, dict) or set(given) - set(GROUP_KEYS):
            raise ValueError(
                f"Maintenance group {group!r}: its options must be a "
                f"dict of {', '.join(GROUP_KEYS)}, got {given!r}."
            )
        setup = given.get("setup_cost")
        setup = (
            0.0
            if setup is None
            else _validate_cost(group, "setup_cost", setup)
        )
        system_down = given.get("system_down", False)
        if not isinstance(system_down, bool):
            raise ValueError(
                f"Maintenance group {group!r}: system_down must be True "
                f"or False, got {system_down!r}."
            )
        out[group] = _MaintenanceGroup(tuple(nodes), setup, system_down)
    return out


def _unknown_component(node, component) -> str:
    """Why ``component`` cannot be node ``node``: it is none of the
    kinds a node can be."""
    from repyability.repairable import Repairable

    kinds = (
        "Give a spec dict with 'reliability' and 'repairability', a "
        "NonRepairable, or a nested RepairableRBD (or, for a junction "
        "that never fails, such as a k-out-of-n vote point, "
        "PerfectReliability)."
    )
    if isinstance(component, Repairable):
        return (
            f"Component {node!r} is a Repairable, which models the "
            "minimal or imperfect repair of a single unit: it cannot be "
            "a node, whose repairs renew it as new. " + kinds
        )
    spec = (
        "{'reliability': <its life>, 'repairability': <its repair "
        "times, or 'instant'>}"
    )
    distribution = unfitted_distribution(component)
    if distribution is not None:
        return (
            f"Component {node!r} is surpyval's {distribution} "
            f"distribution itself: fit it to data "
            f"(surv.{distribution}.fit(times)) or give its parameters "
            f"(surv.{distribution}.from_params([...])), and give it "
            f"with its repairs, as {spec}."
        )
    if callable(getattr(component, "sf", None)):
        return (
            f"Component {node!r} is a model of a life alone (a "
            f"{type(component).__name__}); a repairable component needs "
            f"its repairs too: give {spec}."
        )
    return f"Component {node!r} is a {type(component).__name__}. " + kinds


def _validate_component_spec(node, spec: dict) -> None:
    """Reject unknown keys in a component spec.

    A mistyped cost key (``repair_costs``) would otherwise be silently
    ignored and priced at zero, which is a quiet way to get the money
    wrong; surface it at construction instead.
    """
    unknown = set(spec) - set(COMPONENT_SPEC_KEYS)
    if unknown:
        raise ValueError(
            f"Component {node!r} has unknown key(s) "
            f"{sorted(map(str, unknown))}. A component spec takes "
            f"{', '.join(COMPONENT_SPEC_KEYS)}."
        )
    for key, what in (
        ("reliability", "its lives (time to failure)"),
        ("repairability", "its repair times"),
    ):
        if key not in spec:
            raise ValueError(
                f"Component {node!r} needs a {key!r}: the "
                f"distribution of {what}."
            )
        value = spec[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mean = "MTTF" if key == "reliability" else "MTTR"
            instant = (
                ", or 'instant' for a repair in zero time"
                if key == "repairability"
                else ""
            )
            raise TypeError(
                f"Component {node!r}: {key} is the number {value!r}, "
                f"not a distribution of {what}. For an {mean} of "
                f"{value:g}, give e.g. "
                f"surpyval.Exponential.from_params([1 / {value:g}]); "
                "for a fixed time, "
                f"surpyval.ExactEventTime.from_params([{value:g}])"
                f"{instant}."
            )


def _validate_preventive(
    node, spec, life=None
) -> Tuple[_Preventive, Any, Any]:
    """A component's ``"preventive"`` spec, validated: its schedule,
    and the cost of each replacement and of each inspection (None if
    it prices nothing); ``life`` is the component's life model, which a
    ``"level"`` needs to be a degradation process (#271)."""
    if not isinstance(spec, dict):
        raise ValueError(
            f"Component {node!r}: preventive must be a dict with "
            f"{', '.join(PREVENTIVE_KEYS)}, got {spec!r}."
        )
    unknown = set(spec) - set(PREVENTIVE_KEYS)
    if unknown:
        raise ValueError(
            f"Component {node!r} has unknown preventive key(s) "
            f"{sorted(map(str, unknown))}. A preventive spec takes "
            f"{', '.join(PREVENTIVE_KEYS)}."
        )
    try:
        interval = number_or_nan(spec["interval"])
    except KeyError:
        raise ValueError(
            f"Component {node!r}: a preventive spec needs an interval."
        ) from None
    if not interval > 0.0:
        raise ValueError(
            f"Component {node!r}: the preventive interval must be a "
            f"positive number (inf for none), got {spec['interval']!r}."
        )
    policy = spec.get("policy", "age")
    if policy not in PREVENTIVE_POLICIES:
        raise ValueError(
            f"Component {node!r}: the preventive policy must be 'age', "
            f"'block' or 'condition', got {policy!r}."
        )
    threshold = 0.0
    level = _validate_level(node, spec, policy, life)
    if policy == "condition" and level is None:
        if spec.get("threshold") is None:
            raise ValueError(
                f"Component {node!r}: the 'condition' policy needs a "
                "threshold: the probability of failing before the next "
                "inspection above which the unit is replaced (or, for "
                "a degradation process, a level)."
            )
        threshold = spec["threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float, np.number))
            or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError(
                f"Component {node!r}: the threshold must be a "
                f"probability, in [0, 1], got {threshold!r}."
            )
        threshold = float(threshold)
    elif policy != "condition":
        for key in ("threshold", "inspection_cost"):
            if spec.get(key) is not None:
                raise ValueError(
                    f"Component {node!r}: {key} applies only to the "
                    "'condition' policy, which inspects the unit."
                )
    opportunity = math.inf
    if spec.get("opportunity") is not None:
        if policy != "age":
            raise ValueError(
                f"Component {node!r}: an opportunity applies only to "
                "the 'age' policy (a unit renewed early at a stop of "
                "its maintenance group)."
            )
        opportunity = spec["opportunity"]
        if (
            isinstance(opportunity, bool)
            or not isinstance(opportunity, (int, float, np.number))
            or not 0.0 <= float(opportunity) <= interval
        ):
            raise ValueError(
                f"Component {node!r}: the opportunity must be an age "
                f"from 0 to the interval ({interval:g}), got "
                f"{opportunity!r}."
            )
        opportunity = float(opportunity)
    duration = spec.get("duration", "instant")
    if isinstance(duration, str) and duration == "instant":
        duration = None
    elif not hasattr(duration, "random"):
        raise ValueError(
            f"Component {node!r}: the preventive duration must be a "
            "time-to-maintain model (such as a fitted surpyval "
            f"distribution) or 'instant', got {duration!r}."
        )
    costs = []
    for key, name in (
        ("cost", "preventive_cost"),
        ("inspection_cost", "inspection_cost"),
    ):
        cost = spec.get(key)
        if cost is not None:
            cost = _validate_component_cost(node, name, cost)
            if isinstance(cost, float) and cost == 0.0:
                cost = None
        costs.append(cost)
    schedule = _Preventive(
        interval, policy, duration, threshold, opportunity, level
    )
    return schedule, costs[0], costs[1]


def _validate_level(node, spec: dict, policy: str, life) -> Optional[float]:
    """A ``"preventive"`` spec's ``"level"`` (#271), checked: a
    number below the failure threshold of the degradation process the
    component's life is, under the ``"condition"`` policy and in
    place of its ``"threshold"``; None if not given."""
    level = spec.get("level")
    if level is None:
        return None
    if policy != "condition":
        raise ValueError(
            f"Component {node!r}: level applies only to the "
            "'condition' policy, whose inspections measure it."
        )
    if not is_degradation(life):
        raise ValueError(
            f"Component {node!r}: a level is the measured degradation "
            "level at which an inspection replaces the unit, so its "
            "life must be a surpyval Wiener or gamma degradation "
            "process; replace on its age with a threshold instead."
        )
    if spec.get("threshold") is not None:
        raise ValueError(
            f"Component {node!r}: give the condition policy a level "
            "(of the measured degradation) or a threshold (a "
            "probability of failing by the next inspection), not both."
        )
    level = number_or_nan(level)
    if not level < float(life.threshold):
        raise ValueError(
            f"Component {node!r}: the level must be a number below the "
            "degradation process's failure threshold, "
            f"{float(life.threshold):g}, got {spec['level']!r}."
        )
    return level


def _validate_standby(node, spec: dict) -> _Standby:
    """A component spec's ``"standby"``, validated: the group of
    units it makes the component (see ``_StandbyGroup``)."""
    standby = spec["standby"]
    if not isinstance(standby, dict):
        raise ValueError(
            f"Component {node!r}: standby must be a dict of "
            f"{', '.join(STANDBY_KEYS)}, got {standby!r}."
        )
    unknown = set(standby) - set(STANDBY_KEYS)
    if unknown:
        raise ValueError(
            f"Component {node!r}: unknown standby key(s) "
            f"{sorted(map(str, unknown))}; it takes "
            f"{', '.join(STANDBY_KEYS)}."
        )
    for key in ("preventive", "inspection"):
        if spec.get(key) is not None:
            raise ValueError(
                f"Component {node!r} is a standby group, which takes no "
                f"{key!r} schedule."
            )
    if isinstance(spec["repairability"], str):
        raise ValueError(
            f"Component {node!r} is a standby group, whose units need a "
            "repair time: repaired instantly, a spare would never be "
            "needed."
        )
    counts = {"units": standby.get("units", 2), "k": standby.get("k", 1)}
    for key, value in counts.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or int(value) != value
            or value < 1
        ):
            raise ValueError(
                f"Component {node!r}: standby {key} must be a whole "
                f"number, 1 or more; got {value!r}."
            )
    units, k = int(counts["units"]), int(counts["k"])
    if k >= units:
        raise ValueError(
            f"Component {node!r}: a standby group needs a spare, so k "
            f"must be less than units; got k={k} and units={units}."
        )
    fractions = {
        "dormancy_factor": standby.get("dormancy_factor", 0.0),
        "switching_probability": standby.get("switching_probability", 1.0),
    }
    for key, value in fractions.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"Component {node!r}: standby {key} must be a number in "
                f"[0, 1]; got {value!r}."
            )
    return _Standby(
        units,
        k,
        float(fractions["dormancy_factor"]),
        float(fractions["switching_probability"]),
    )


def _validate_inspection(node, spec) -> Tuple[_Inspection, Any]:
    """A component's ``"inspection"`` spec, validated: its schedule, and
    its cost (None if it prices nothing)."""
    if not isinstance(spec, dict):
        raise ValueError(
            f"Component {node!r}: inspection must be a dict with "
            f"{', '.join(INSPECTION_KEYS)}, got {spec!r}."
        )
    unknown = set(spec) - set(INSPECTION_KEYS)
    if unknown:
        raise ValueError(
            f"Component {node!r} has unknown inspection key(s) "
            f"{sorted(map(str, unknown))}. An inspection spec takes "
            f"{', '.join(INSPECTION_KEYS)}."
        )
    try:
        interval = number_or_nan(spec["interval"])
    except KeyError:
        raise ValueError(
            f"Component {node!r}: an inspection spec needs an interval."
        ) from None
    if not (interval > 0.0 and np.isfinite(interval)):
        raise ValueError(
            f"Component {node!r}: the inspection interval must be a "
            f"positive, finite number, got {spec['interval']!r}. (A "
            "hidden failure is found only by an inspection.)"
        )
    duration = spec.get("duration", "instant")
    if isinstance(duration, str) and duration == "instant":
        duration = None
    elif not hasattr(duration, "random"):
        raise ValueError(
            f"Component {node!r}: the inspection duration must be a "
            "time-to-test model (such as a fitted surpyval distribution) "
            f"or 'instant', got {duration!r}."
        )
    cost = spec.get("cost")
    if cost is not None:
        cost = _validate_component_cost(node, "inspection_cost", cost)
        if isinstance(cost, float) and cost == 0.0:
            cost = None
    offset = number_or_nan(spec.get("offset", 0.0))
    if not 0.0 <= offset < interval:
        raise ValueError(
            f"Component {node!r}: the inspection offset, the time of its "
            "first test, must be at least 0 and less than its interval, "
            f"{interval:g}, got {spec['offset']!r}."
        )
    if offset <= _TINY_OFFSET * interval:
        # An offset of 0 puts the first test at the interval; one this
        # close to 0 (a share of the interval worked out as nearly 0,
        # say) is taken as 0, not as a test at the start (#237).
        offset = 0.0
    coverage = number_or_nan(spec.get("coverage", 1.0))
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(
            f"Component {node!r}: the inspection coverage, the chance "
            "that a test finds a failure, must be from 0 to 1, got "
            f"{spec['coverage']!r}."
        )
    full_test = spec.get("full_test")
    if full_test is not None:
        given = full_test
        full_test = number_or_nan(full_test)
        count = full_test / interval
        if not (
            np.isfinite(count)
            and round(count) >= 1
            and abs(count - round(count)) <= 1e-9 * count
        ):
            raise ValueError(
                f"Component {node!r}: the full_test interval must be a "
                f"whole multiple of its inspection interval, {interval:g} "
                f"(every so many tests is a full one), got {given!r}."
            )
        full_test = round(count) * interval
    elif coverage < 1.0:
        raise ValueError(
            f"Component {node!r}: with a coverage below 1, the failures "
            "its tests miss are found only by a full test: give its "
            "full_test interval, a whole multiple of its inspection "
            "interval (the mission time, say, after which it is "
            "renewed)."
        )
    schedule = _Inspection(interval, duration, offset, coverage, full_test)
    return schedule, cost


def _validate_cost(node, key: str, value) -> float:
    """Coerce a cost to a finite, non-negative float."""
    if hasattr(value, "qf"):
        raise ValueError(
            f"{node!r}: {key} must be a number, not a distribution. Only "
            "the costs charged per action (repair_cost, replace_cost and "
            "the preventive and inspection costs) may be distributions."
        )
    if not is_number(value):
        # Text that reads as a number, and a bool, are refused (#233).
        raise ValueError(f"{node!r}: {key} must be a number, got {value!r}.")
    cost = float(value)
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError(
            f"{node!r}: {key} must be finite and non-negative, got "
            f"{value!r}."
        )
    return cost


def _validate_component_cost(node, key: str, value):
    """Validate a per-component cost: a number (coerced to float) or, for
    the per-action costs, a distribution of the cost -- anything with
    ``qf`` and ``mean``, such as a fitted surpyval model.

    A distribution must have a finite mean and put no more than 1e-12
    probability on a negative cost.
    """
    if key not in _PER_ACTION_COST_KEYS or not hasattr(value, "qf"):
        return _validate_cost(node, key, value)
    mean = _mean_cost(value)
    if not np.isfinite(mean):
        raise ValueError(
            f"{node!r}: the {key} distribution must have a finite mean, "
            f"got {mean:g}."
        )
    lowest = float(np.ravel(value.qf(np.array([1e-12])))[0])
    if not lowest >= 0.0:
        raise ValueError(
            f"{node!r}: the {key} distribution puts appreciable "
            f"probability on a negative cost (its 1e-12 quantile is "
            f"{lowest:g}). Use one on [0, inf), such as a LogNormal, "
            "Gamma or Weibull."
        )
    return value
