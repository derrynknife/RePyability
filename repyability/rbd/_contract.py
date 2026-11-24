"""One call that values a plan from one JSON document (``value_plan``):
the request names the diagram (``RBD.to_dict()``), the analysis and its
inputs as plain data; the answer is plain data too, strict JSON, with the
request it answers, the units, the route and the result (or the refusal,
with what to run instead).

The request is the record: ``value_plan(answer["request"])`` gives the
same answer, number for number. A seeded analysis given no seed has one
drawn and written into the request it echoes, so that holds for it too.
The documents' shapes are in ``repyability/schemas/npv-1.json``.
"""

import dataclasses
import inspect
import math
from typing import Any, Callable, Dict, Hashable, NoReturn, Tuple

from repyability.rbd import _streams

#: The version of the request and answer documents.
SCHEMA = 1

#: The analyses a request may name.
ANALYSES = (
    "expected_npv",
    "expected_npv_uncertainty",
    "npv_distribution",
    "compare_npv",
    "plan_vs_actual",
    "task_value",
)

#: The keys of a request.
REQUEST_KEYS = ("schema", "diagram", "analysis", "inputs", "other")

#: Inputs no JSON document can give (models, functions): left to Python.
_NOT_JSON = {"uncertainty"}

#: The analyses that draw random numbers, and so take a ``seed``.
_SEEDED = ("expected_npv_uncertainty", "npv_distribution", "compare_npv")


def _fail(message: str) -> NoReturn:
    raise ValueError(f"value_plan: {message}")


def _strict(value):
    """``value`` (plain data) with every number JSON can hold: a float
    that is not finite written as the string ``"inf"``, ``"-inf"`` or
    ``"nan"``."""
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strict(item) for item in value]
    return value


def _node(rbd, key) -> Hashable:
    """The diagram's node a JSON key names: the node itself, or the one
    whose name it spells (a JSON object's keys are text)."""
    if key in rbd.components:
        return key
    for node in rbd.components:
        if str(node) == str(key):
            return node
    _fail(
        f"{key!r} is not a component of the diagram; its components are "
        f"{[str(node) for node in rbd.components]}."
    )


def _by_node(rbd, given, what: str, convert: Callable) -> dict:
    if not isinstance(given, dict):
        _fail(f"{what} must be an object of a value by component.")
    return {_node(rbd, key): convert(value) for key, value in given.items()}


def _state(value):
    from repyability import NodeState

    if not isinstance(value, dict):
        _fail(
            f"a state must be an object of NodeState's fields, got {value!r}."
        )
    names = {field.name for field in dataclasses.fields(NodeState)}
    unknown = sorted(set(value) - names)
    if unknown:
        _fail(
            f"a state has unknown field(s) {unknown}; NodeState takes "
            f"{sorted(names)}."
        )
    return NodeState(**value)


def _timeline(value):
    from repyability import Timeline

    if not isinstance(value, dict) or "end" not in value:
        _fail(
            "a timeline must be an object with end, changes, up and "
            f"planned (Timeline.to_dict()), got {value!r}."
        )
    return Timeline(
        value.get("changes", []),
        value["end"],
        value.get("up", True),
        planned=value.get("planned"),
        name=value.get("name"),
    )


def _value_rate(value):
    if isinstance(value, dict):
        from repyability import ValueSchedule

        unknown = sorted(set(value) - {"times", "rates"})
        if unknown or "rates" not in value:
            _fail(
                "a value_rate schedule is an object of times and rates "
                f"(ValueSchedule), got keys {sorted(value)}."
            )
        return ValueSchedule(value.get("times", []), value["rates"])
    return value


def _escalation(rbd, value):
    if not isinstance(value, dict):
        _fail("escalation must be an object of a growth rate by name.")
    out = {}
    for key, rate in value.items():
        named = [node for node in rbd.components if str(node) == str(key)]
        out[named[0] if named else key] = rate
    return out


def _salvage(rbd, value):
    if isinstance(value, dict):
        return _by_node(rbd, value, "salvage", lambda amount: amount)
    return value


def _inputs(rbd, analysis: str, given: dict) -> dict:
    """The method's keyword arguments from a request's ``inputs``,
    checked by name against its signature."""
    method = getattr(rbd, analysis)
    names = [
        name
        for name, parameter in inspect.signature(method).parameters.items()
        if parameter.kind
        in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        and name not in _NOT_JSON
        and not (analysis == "compare_npv" and name == "other")
    ]
    unknown = sorted(set(given) - set(names))
    if unknown:
        _fail(
            f"{analysis} takes no input(s) {unknown}; its inputs are "
            f"{names}."
        )
    convert: Dict[str, Callable[[Any], Any]] = {
        "state": lambda v: _by_node(rbd, v, "state", _state),
        "start_state": lambda v: _by_node(rbd, v, "start_state", _state),
        "timelines": lambda v: _by_node(rbd, v, "timelines", _timeline),
        "value_rate": _value_rate,
        "escalation": lambda v: _escalation(rbd, v),
        "salvage": lambda v: _salvage(rbd, v),
        "working_nodes": lambda v: [_node(rbd, n) for n in v],
        "broken_nodes": lambda v: [_node(rbd, n) for n in v],
    }
    return {
        name: convert.get(name, lambda v: v)(value)
        for name, value in given.items()
    }


#: The usage forecast's keys a request gives (a ``UsageForecast``'s
#: ``to_dict()`` may be given whole: what it adds is worked out again).
_USAGE_KEYS = ("start", "end", "plan", "readings", "read_at", "time_zone")
_USAGE_DERIVED = (
    "fill",
    "meters",
    "at_start",
    "at_end",
    "at_end_low",
    "at_end_high",
    "stretches",
    "description",
    "notes",
)


def _usage(value):
    """The usage forecast a request's ``usage`` gives: a duty plan's
    ``forecast`` over ``start`` to ``end`` (calendar time, around the
    clock, when it gives no ``plan``)."""
    from repyability.schedule import DutyPlan, UsageForecast

    if not isinstance(value, dict) or not {"start", "end"} <= set(value):
        _fail(
            "usage must be an object with start and end (and a duty "
            "plan's to_dict() as plan, readings and read_at), as a "
            f"UsageForecast's to_dict() has them; got {value!r}."
        )
    unknown = sorted(set(value) - set(_USAGE_KEYS) - set(_USAGE_DERIVED))
    if unknown:
        _fail(
            f"usage takes no key(s) {unknown}; it takes {list(_USAGE_KEYS)}."
        )
    readings = value.get("readings") or None
    if value.get("plan") is None:
        return UsageForecast.calendar(
            value["start"],
            value["end"],
            readings=readings,
            read_at=value.get("read_at"),
            time_zone=value.get("time_zone"),
        )
    plan = DutyPlan.from_dict(value["plan"])
    if value.get("time_zone") not in (None, plan.time_zone):
        _fail(
            f"usage gives time_zone {value['time_zone']!r} and its plan "
            f"{plan.time_zone!r}: the plan's zone is the forecast's."
        )
    return plan.forecast(
        value["start"],
        value["end"],
        readings=readings,
        read_at=value.get("read_at"),
    )


def _task_inputs() -> Tuple[str, ...]:
    """``task_value``'s inputs by name: its keywords, ``task`` and
    ``usage``, and the service's ``effect`` (in place of ``does``)."""
    from repyability.schedule import task_value

    return tuple(
        "effect" if name == "does" else name
        for name in inspect.signature(task_value).parameters
    )


def _task_value(request) -> Tuple[Any, Dict[str, str]]:
    """What a ``task_value`` request asks: the service's value (the task
    services the request's unit plan, with its stated effect), and its
    route."""
    from repyability.maintenance._plan import UnitPlan
    from repyability.schedule import Services, Task, task_value
    from repyability.schedule._service import effect_from_dict

    given = request["inputs"]
    names = _task_inputs()
    unknown = sorted(set(given) - set(names))
    if unknown:
        _fail(
            f"task_value takes no input(s) {unknown}; its inputs are "
            f"{list(names)}."
        )
    for key in ("task", "usage"):
        if key not in given:
            _fail(f"task_value needs the input {key!r}.")
    try:
        effect = effect_from_dict(given.get("effect"))
    except ValueError as error:
        _fail(str(error))
    plan = None
    if "diagram" in request:
        document = request["diagram"]
        if not isinstance(document, dict) or (
            document.get("kind") != "unit_plan"
        ):
            _fail(
                "task_value values a service of one unit: its diagram is "
                "the unit's plan, a UnitPlan's to_dict()."
            )
        plan = UnitPlan.from_dict(document)
    if effect is not None and plan is None:
        _fail(
            "a service with an effect needs the unit's plan (its life, "
            "repairs and prices) as the diagram, a UnitPlan's to_dict()."
        )
    does = Services(plan, effect)
    kwargs = {
        name: value
        for name, value in given.items()
        if name not in ("task", "usage", "effect")
    }
    exact = does.no_effect
    route = {
        "route": "exact" if exact else "simulated",
        "reason": (
            "A service with no effect on the life: minus the present "
            "value of its own cost, exactly."
            if exact
            else "A service with a stated effect: the plan's one-node "
            "diagram simulated with and without the service, on common "
            "random numbers."
        ),
    }
    if not isinstance(given["task"], dict):
        _fail("task must be a Task's to_dict(), an object.")
    result = task_value(
        Task.from_dict(given["task"]),
        does,
        _usage(given["usage"]),
        **kwargs,
    )
    return result, route


def _diagram(document, what: str):
    from repyability import RepairableRBD

    if not isinstance(document, dict):
        _fail(f"{what} must be a diagram's to_dict(), an object.")
    if document.get("kind") == "unit_plan":
        # A unit's plan, valued as its one-node diagram (UC-65 of the
        # maintenance README).
        from repyability.maintenance._plan import UnitPlan

        return UnitPlan.from_dict(document).diagram()
    rbd = RepairableRBD.from_dict(document)
    if not isinstance(rbd, RepairableRBD):
        _fail(f"{what} must be a RepairableRBD: the NPV analyses are its.")
    return rbd


def _checked(spec) -> dict:
    """The request, checked: its keys, schema, analysis and inputs as an
    object (a seed drawn for a seeded analysis given none)."""
    if not isinstance(spec, dict):
        _fail(f"the request must be an object, got {type(spec).__name__}.")
    unknown = sorted(set(spec) - set(REQUEST_KEYS))
    if unknown:
        _fail(
            f"the request has unknown key(s) {unknown}; it takes "
            f"{list(REQUEST_KEYS)}."
        )
    if spec.get("schema") != SCHEMA:
        _fail(
            f"the request's schema must be {SCHEMA} (this version's), got "
            f"{spec.get('schema')!r}."
        )
    analysis = spec.get("analysis")
    if analysis not in ANALYSES:
        _fail(f"analysis must be one of {list(ANALYSES)}, got {analysis!r}.")
    needs = () if analysis == "task_value" else ("diagram",)
    for key in needs + (("other",) if analysis == "compare_npv" else ()):
        if key not in spec:
            _fail(f"{analysis} needs {key!r}: a diagram's to_dict().")
    if analysis != "compare_npv" and "other" in spec:
        _fail("only compare_npv takes 'other', the plan to compare with.")
    inputs = spec.get("inputs", {})
    if not isinstance(inputs, dict):
        _fail("inputs must be an object of the analysis's inputs by name.")
    request = {**spec, "inputs": dict(inputs)}
    seeded = analysis in _SEEDED or (
        analysis == "task_value"
        and request["inputs"].get("effect") not in (None, {"kind": "none"})
    )
    if seeded and request["inputs"].get("seed") is None:
        request["inputs"]["seed"] = _streams.shared_seed(None)
    return request


def value_plan(spec: dict) -> dict:
    """Value a plan from one JSON document, and answer in one.

    In plain terms: "what is this plan worth?" in one call, for an agent
    or a service. The request names a diagram as
    [`to_dict`][repyability.RepairableRBD.to_dict] saves it (or a unit's
    plan as [`UnitPlan.to_dict`][repyability.UnitPlan.to_dict] saves it,
    valued as its one-node diagram), one of the
    NPV analyses (``"expected_npv"``, ``"expected_npv_uncertainty"``,
    ``"npv_distribution"``, ``"compare_npv"`` with the ``"other"`` plan,
    or ``"plan_vs_actual"``), and that method's inputs by name, as plain
    data: states as ``NodeState`` fields, timelines as
    [`Timeline.to_dict`][repyability.Timeline.to_dict] writes them, a
    ``value_rate`` schedule as ``{"times": [...], "rates": [...]}``, and
    components by name. The answer echoes the request, gives the units
    (the diagram's time unit, money in cost units), the analysis's
    route, and the result's ``to_dict()`` or, where the analysis
    refuses the diagram, the refusal and every NPV analysis's route.
    It is strict JSON: a number that is not finite is the string
    ``"inf"``, ``"-inf"`` or ``"nan"``.

    ``"task_value"`` values a service on dates, as
    [`task_value`][repyability.schedule.task_value] with
    ``Services(plan, effect=)`` does: the diagram is the unit's plan
    (``UnitPlan.to_dict()``; not needed with no effect), and the inputs
    are the ``task`` (``Task.to_dict()``), the ``usage`` (``start``,
    ``end`` and a duty plan's ``to_dict()`` as ``plan``, with its
    ``readings`` and ``read_at``, as ``UsageForecast.to_dict()`` has
    them; calendar time with no ``plan``), the service's ``effect`` and
    ``task_value``'s keywords (``age``, ``discount``, ``period``,
    ``seed``, ...). The effect is None (the default: no effect on the
    life, the service an accepted cost, valued exactly),
    ``{"kind": "restoration", "q": ..., "model": ...}`` (or ``"to_age"``,
    ``"replace_every"``, ``"replacement_cost"``), or ``{"kind":
    "condition", "life": ..., "fresh": [...], "stale": [...], "lasts":
    ...}`` with the fitted life as a diagram saves a node's model; with
    an effect it is simulated, and a seed is drawn when none is given.

    The request is the record: valuing ``answer["request"]`` again gives
    the same answer. A seeded analysis given no ``seed`` has one drawn,
    and written into the request. The shapes of both documents are in
    ``repyability/schemas/npv-1.json``
    ([`npv_schema`][repyability.npv_schema]).

    Parameters
    ----------
    spec : dict
        ``{"schema": 1, "diagram": ..., "analysis": ..., "inputs":
        {...}}`` (and ``"other"`` for ``compare_npv``; no ``"diagram"``
        for ``task_value`` with no effect).

    Returns
    -------
    dict
        ``schema``, ``analysis``, ``request``, ``units``, ``route`` and
        ``result`` (or ``refused``).

    Raises
    ------
    ValueError
        If the request is not one: an unknown key or input (named, with
        those it takes), another schema, an analysis not listed, or a
        component the diagram does not have.

    Examples
    --------
    >>> import surpyval as surv
    >>> from repyability import RepairableRBD, value_plan
    >>> pump = RepairableRBD(
    ...     [("s", "p"), ("p", "t")],
    ...     {"p": {"reliability": surv.Exponential.from_params([0.01]),
    ...            "repairability": surv.Exponential.from_params([0.5]),
    ...            "repair_cost": 100.0}},
    ... )
    >>> answer = value_plan({
    ...     "schema": 1,
    ...     "diagram": pump.to_dict(),
    ...     "analysis": "expected_npv",
    ...     "inputs": {"horizon": 100.0, "discount_rate": 0.001},
    ... })
    >>> answer["route"]["route"], round(answer["result"]["cost"], 2)
    ('numerical', 93.34)
    >>> value_plan(answer["request"]) == answer
    True
    """
    from repyability.rbd import routes as r
    from repyability.rbd.results import plain

    request = _checked(spec)
    analysis = request["analysis"]
    if analysis == "task_value":
        return _service_answer(request)
    rbd = _diagram(request["diagram"], "diagram")
    kwargs = _inputs(rbd, analysis, request["inputs"])
    args: Tuple[Any, ...] = ()
    if analysis == "compare_npv":
        args = (_diagram(request["other"], "other"),)
    route = rbd.analysis_routes()[analysis]
    answer: Dict[str, Any] = {
        "schema": SCHEMA,
        "analysis": analysis,
        "request": request,
        "units": {
            "time": plain(getattr(rbd, "unit", None)),
            "money": "cost units",
        },
        "route": {"route": route.route, "reason": route.reason},
    }
    try:
        result = getattr(rbd, analysis)(*args, **kwargs)
    except (NotImplementedError, r.Refused) as error:
        answer["refused"] = {
            "reason": str(error),
            "routes": {
                name: found.route
                for name, found in rbd.analysis_routes().items()
                if name in ANALYSES
            },
        }
    else:
        answer["result"] = plain(result)
    return _strict(plain(answer))


def _service_answer(request) -> dict:
    """``value_plan``'s answer to a ``task_value`` request."""
    from repyability.rbd.results import plain

    answer: Dict[str, Any] = {
        "schema": SCHEMA,
        "analysis": "task_value",
        "request": request,
    }
    try:
        result, route = _task_value(request)
    except NotImplementedError as error:
        answer["units"] = {
            "time": request["inputs"].get("unit"),
            "money": "cost units",
        }
        answer["route"] = {"route": "refused", "reason": str(error)}
        answer["refused"] = {
            "reason": str(error),
            "routes": {"task_value": "refused"},
        }
    else:
        answer["units"] = {"time": result.unit, "money": "cost units"}
        answer["route"] = route
        answer["result"] = plain(result)
    return _strict(plain(answer))
