"""Repairable spares over two echelons: bases, and a depot behind them.

A fleet's systems run from bases. A failed part is swapped for one from
its base's shelf (a system with none to fit waits: a backorder) and is
repaired, at the base with probability ``r`` and at the depot otherwise;
a part sent to the depot is ordered from it at once, and arrives an
order-and-ship time after the depot has one to send. Every shelf is
replenished one for one, so a site's stock is its shelf, its parts in
repair and those on order, less what it owes.

The model is Sherbrooke's (*Optimal Inventory Modeling of Systems*, 2nd
ed., 2004, ch. 2 and 3). With Poisson demand a shop's parts in repair are
Poisson with mean the demand times the mean repair time, whatever the
repair times' distribution (Palm's theorem):

- the depot's pipeline has mean ``m0 T0``, ``m0`` the demand the bases
  send it; with ``s0`` held its backorders have mean ``EBO0`` and
  variance ``VBO0``, and an order waits ``EBO0 / m0`` on average;
- a base's pipeline (in its repair, or on order) has mean ``m [r T + (1
  - r)(O + EBO0 / m0)]``. METRIC takes it as Poisson. VARI-METRIC gives
  it its variance, ``m [r T + (1 - r) O] + f (1 - f) EBO0 + f^2 VBO0``
  with ``f`` the base's share of the depot's demand, and a negative
  binomial of that mean and variance (``_base_pipeline``, one statement
  for both);
- a base's backorders with ``s`` held are ``E[(X - s)+]``, and each of
  the part's ``N Z`` places there is empty with chance ``EBO / (N Z)``;
- a base's availability is the diagram's chance of working with each
  place holding its part independently (``_up``): where every part is
  needed, the product over parts of ``(1 - EBO / (N Z))^Z``.

A base's repair fraction, repair time and ship time may each be its own
(``_Part`` keeps them per base).

The front (``spares_front``) is the textbook's marginal analysis: for
each part, the least loss of each count held in all (``_counts``), the
lower convex hull of those (``_Curve``), and the parts' hull steps taken
in the order of loss removed per unit of spend. The loss is the bases'
expected backorders (``objective="backorders"``, a part's own whatever
the others hold) or the fleet availability the part's holes cost
(``"availability"``, ``_Fleet.losses``, which changes with the other
parts' stocks, so each part's hull is drawn again from the count it has
reached after every step).
"""

import math
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from repyability.rbd import _crews, _spares
from repyability.rbd.results import SparesAllocation
from repyability.utils.checks import is_number, one_of, whole_number

#: The methods: the pipeline's distribution at a base.
METHODS = ("vari-metric", "metric")
#: A pipeline's distribution is kept to a tail of ``_TAIL``.
_TAIL = 1e-18
#: What the front removes per unit of spend.
OBJECTIVES = ("auto", "backorders", "availability")
#: An availability's rounding: a loss of availability within it is none.
_NOISE = 8.0 * float(np.finfo(float).eps)
#: The spec's keys that may differ by base.
_BY_BASE = ("base_repair_fraction", "base_repair_time", "ship_time")
#: What a part's spec may hold.
_KEYS = (
    "price",
    "demand_rate",
    "quantity",
    "nodes",
    "base_repair_fraction",
    "base_repair_time",
    "depot_repair_time",
    "ship_time",
)
#: What the crews' refusal says (as ``_spares._STOCK_CREWS``).
_CREWS = (
    "the spares' demand assumes",
    "the stock over echelons has no simulation to fall back on; the "
    "diagram without the limit (repair_crews=None) gives it when a crew "
    "is always free, or give each part's demand_rate.",
)


class _Part(NamedTuple):
    """A part: its price, how many a system has, each base's demand per
    unit time, the chance a failure is repaired at each base, the mean
    times of each base's repair, of a depot repair and of each base's
    order from the depot, and the diagram's components it is (none: a
    part every system needs)."""

    name: Hashable
    price: float
    quantity: int
    demand: np.ndarray
    fraction: np.ndarray
    base_time: np.ndarray
    depot_time: float
    ship_time: np.ndarray
    nodes: Tuple[Hashable, ...] = ()


# -- the model ---------------------------------------------------------------


def _pipeline(mean: float, variance: float) -> np.ndarray:
    """The chance that more than ``0, 1, 2, ...`` parts are in a pipeline
    of that mean and variance, to a tail of ``_TAIL``: a negative
    binomial's, or a Poisson's when the variance is the mean."""
    from scipy import stats

    if mean <= 0.0:
        return np.zeros(1)
    if variance <= mean * (1.0 + 1e-12):
        law = stats.poisson(mean)
    else:
        p = mean / variance
        law = stats.nbinom(mean * p / (1.0 - p), p)
    size = 16
    while True:
        more = law.sf(np.arange(size))
        if more[-1] < _TAIL:
            return more[: int(np.argmax(more < _TAIL)) + 1]
        size *= 2


def _backorders(more: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The mean and variance of the backorders ``(X - s)+`` at each stock
    ``s = 0, 1, ...``, from ``more[k] = P(X > k)``: the mean is the sum of
    ``more`` from ``s`` on, and the second moment that of ``(2 (k - s) +
    1) more[k]``."""
    mean = np.cumsum(more[::-1])[::-1]
    beyond = np.concatenate((np.cumsum(mean[:0:-1])[::-1], [0.0]))
    return mean, np.maximum(2.0 * beyond + mean - mean**2, 0.0)


def _at(values: np.ndarray, stock: int) -> float:
    """A stock's value: 0 past the pipeline's tail."""
    return float(values[stock]) if stock < len(values) else 0.0


class _Depot(NamedTuple):
    """A part's depot: its demand, and its backorders' mean and variance
    at each stock."""

    demand: float
    mean: np.ndarray
    variance: np.ndarray


def _depot(part: _Part) -> _Depot:
    if np.all(part.fraction == part.fraction[0]):
        # One fraction for every base: the fleet's demand times it.
        demand = float(part.demand.sum()) * (1.0 - float(part.fraction[0]))
    else:
        demand = float((part.demand * (1.0 - part.fraction)).sum())
    pipeline = demand * part.depot_time
    return _Depot(demand, *_backorders(_pipeline(pipeline, pipeline)))


def _base_pipeline(
    part: _Part, depot: _Depot, base: int, depot_stock: int, method: str
) -> Tuple[float, float]:
    """The mean and variance of a base's pipeline of a part with
    ``depot_stock`` held at the depot: its own repairs, and its orders
    from the depot, each delayed by the depot's backorders. METRIC's
    variance is the mean (a Poisson pipeline)."""
    demand = float(part.demand[base])
    fraction = float(part.fraction[base])
    own = demand * (
        fraction * float(part.base_time[base])
        + (1.0 - fraction) * float(part.ship_time[base])
    )
    if depot.demand <= 0.0:
        return own, own
    share = demand * (1.0 - fraction) / depot.demand
    owed = _at(depot.mean, depot_stock)
    mean = own + share * owed
    if method == "metric":
        return mean, mean
    spread = _at(depot.variance, depot_stock)
    return mean, own + share * (1.0 - share) * owed + share**2 * spread


def _base_backorders(
    part: _Part, depot: _Depot, depot_stock: int, method: str
) -> List[np.ndarray]:
    """Each base's expected backorders of a part at each stock of its
    own, with ``depot_stock`` at the depot."""
    return [
        _backorders(
            _pipeline(*_base_pipeline(part, depot, base, depot_stock, method))
        )[0]
        for base in range(len(part.demand))
    ]


# -- one part's best stocks --------------------------------------------------


class _Held(NamedTuple):
    """The best split of a count of a part: the loss left (the bases'
    backorders in all, or the availability the part's holes cost), the
    depot's stock and each base's."""

    loss: float
    depot: int
    bases: Tuple[int, ...]


def _counts(
    losses: Callable[[int], List[np.ndarray]], bases: int, most: int
) -> List[_Held]:
    """The least loss of ``0, 1, ..., most`` of a part held in all, and
    the split that gives each; ``losses(depot_stock)`` is each base's
    loss at each stock of its own. For a depot stock the bases' spares
    go in the order of the loss each removes, which is the best split
    among them where a base's loss is convex in its stock (its
    backorders are; the availability its holes cost is where the system
    needs the part, and need not be where several of it stand in for
    each other); the least over the depot stocks is the best of all."""
    best: List[Optional[_Held]] = [None] * (most + 1)
    for depot_stock in range(most + 1):
        room = most - depot_stock
        owed = losses(depot_stock)
        steps = sorted(
            (
                (_at(owed[base], s) - _at(owed[base], s + 1), base)
                for base in range(bases)
                for s in range(room)
            ),
            key=lambda step: -step[0],
        )
        held = [0] * bases
        for added in range(room + 1):
            count = depot_stock + added
            # Summed afresh, so that no loss left is exactly 0.
            loss = math.fsum(_at(o, s) for o, s in zip(owed, held))
            found = best[count]
            if found is None or loss < found.loss:
                best[count] = _Held(loss, depot_stock, tuple(held))
            if added < room:
                held[steps[added][1]] += 1
    return [held for held in best if held is not None]


class _Curve:
    """A part's efficient counts: the lower convex hull of its least
    loss against the count held. Counts are worked out as far as asked
    (``most``, doubled as needed); a step of the hull is given once no
    count further on can change it: a loss is never negative, so none
    can lie below the step's line where that has reached zero.

    The loss is the bases' backorders, or with a ``fleet`` the
    availability the part's holes cost at the other parts' stocks: the
    hull is then drawn from the count reached (``_hull``, again whenever
    another part's stock changes)."""

    def __init__(self, part: _Part, method: str):
        self.part = part
        self.method = method
        self.depot = _depot(part)
        self.fleet: Optional["_Fleet"] = None
        self._owed: Dict[int, List[np.ndarray]] = {}
        self.most = 8
        self.at = 0
        self._hull()

    def owed(self, depot_stock: int) -> List[np.ndarray]:
        """Each base's expected backorders at each stock of its own."""
        if depot_stock not in self._owed:
            self._owed[depot_stock] = _base_backorders(
                self.part, self.depot, depot_stock, self.method
            )
        return self._owed[depot_stock]

    def _losses(self, depot_stock: int) -> List[np.ndarray]:
        if self.fleet is None:
            return self.owed(depot_stock)
        return self.fleet.losses(self, depot_stock)

    def _hull(self) -> None:
        self.counts = _counts(self._losses, len(self.part.demand), self.most)
        start = 0 if self.fleet is None else self.at
        hull = [start]
        for count in range(start + 1, self.most + 1):
            while len(hull) > 1 and self._above(hull[-2], hull[-1], count):
                hull.pop()
            hull.append(count)
        self.hull = hull

    def _above(self, a: int, b: int, c: int) -> bool:
        """Whether count ``b`` lies above the line from ``a`` to ``c``."""
        owed = [self.counts[k].loss for k in (a, b, c)]
        return (owed[1] - owed[0]) * (c - a) > (owed[2] - owed[0]) * (b - a)

    def held(self) -> _Held:
        """The stock at the count reached."""
        return self.counts[self.at]

    def short(self, base: int) -> float:
        """A base's expected backorders at the stock reached."""
        held = self.held()
        return _at(self.owed(held.depot)[base], held.bases[base])

    def step(self) -> Optional[Tuple[float, int]]:
        """The next step of the hull from the count reached: the loss it
        removes per part added, and the count it leads to; None when no
        loss is left to remove."""
        while True:
            here = self.counts[self.at].loss
            if here <= 0.0:
                return None
            after = self.hull[self.hull.index(self.at) + 1 :]
            if after:
                then = after[0]
                removed = here - self.counts[then].loss
                slope = removed / (then - self.at)
                # The step's line has reached zero by the last count
                # worked out: no count past it lies below the line.
                if here - slope * (self.most - self.at) <= 0.0:
                    return slope, then
            self.most *= 2
            self._hull()

    def take(self, count: int) -> None:
        self.at = count


# -- a stock's values --------------------------------------------------------


def _working(part: _Part, base: int, fleets: np.ndarray, short):
    """The chance that a place of a part at a base holds one, from the
    base's expected backorders of it (a number, or an array of them)."""
    places = fleets[base] * part.quantity
    return np.maximum(1.0 - short / places, 0.0)


def _up(drawn, parts: Sequence[_Part], working: Sequence):
    """The chance that a system works, ``working[i]`` the chance that a
    place of ``parts[i]`` holds one (numbers, or arrays of one length),
    the places independent: the diagram ``drawn`` at each component's
    chance, times that of every part it does not draw. With no diagram
    every part is needed, and it is the product over the parts."""
    needed = 1.0
    chances = None if drawn is None else dict.fromkeys(drawn.nodes, 1.0)
    for part, chance in zip(parts, working):
        if chances is None or not part.nodes:
            needed = needed * chance**part.quantity
            continue
        each = chance ** (part.quantity / len(part.nodes))
        for node in part.nodes:
            chances[node] = each
    if chances is None:
        return needed
    length = max(np.size(chance) for chance in chances.values())
    return needed * drawn.system_probability(
        {n: np.broadcast_to(c, (length,)) for n, c in chances.items()}
    )


class _Fleet:
    """The availability a part's holes cost: the bases' systems, the
    diagram (None: every part needed) and every part's curve."""

    def __init__(self, fleets: np.ndarray, drawn, curves: Sequence[_Curve]):
        self.fleets = fleets
        self.drawn = drawn
        self.curves = curves
        self.parts = [curve.part for curve in curves]

    def losses(self, curve: _Curve, depot_stock: int) -> List[np.ndarray]:
        """Each base's share of the fleet availability lost to a part's
        holes at each stock of its own there, the other parts at the
        stocks they have reached: the availability with none of the
        part owed, less that at each stock."""
        out = []
        for base, owed in enumerate(curve.owed(depot_stock)):
            working = [
                _working(
                    other.part,
                    base,
                    self.fleets,
                    (
                        np.append(owed, 0.0)
                        if other is curve
                        else other.short(base)
                    ),
                )
                for other in self.curves
            ]
            up = np.broadcast_to(
                _up(self.drawn, self.parts, working), (len(owed) + 1,)
            )
            lost = self.fleets[base] / self.fleets.sum() * (up[-1] - up[:-1])
            out.append(np.where(lost > _NOISE, lost, 0.0))
        return out


def _allocation(
    sites: Sequence[Hashable],
    fleets: np.ndarray,
    curves: Sequence[_Curve],
    method: str,
    drawn=None,
    objective: str = "backorders",
) -> SparesAllocation:
    """The result for the stock each part's curve has reached."""
    spend = 0.0
    depot_stock: Dict[Hashable, int] = {}
    depot_backorders: Dict[Hashable, float] = {}
    depot_delay: Dict[Hashable, float] = {}
    stock: Dict[Hashable, Dict[Hashable, int]] = {site: {} for site in sites}
    owed: Dict[Hashable, Dict[Hashable, float]] = {site: {} for site in sites}
    working: List[List[float]] = [[] for _ in sites]
    for curve in curves:
        part, held = curve.part, curve.held()
        spend += part.price * (held.depot + sum(held.bases))
        depot_stock[part.name] = held.depot
        waiting = _at(curve.depot.mean, held.depot)
        depot_backorders[part.name] = waiting
        depot_delay[part.name] = (
            waiting / curve.depot.demand if curve.depot.demand > 0.0 else 0.0
        )
        for j, site in enumerate(sites):
            stock[site][part.name] = held.bases[j]
            short = curve.short(j)
            owed[site][part.name] = short
            working[j].append(_working(part, j, fleets, short))
    parts = [curve.part for curve in curves]
    up = np.array(
        [float(np.ravel(_up(drawn, parts, each))[0]) for each in working]
    )
    return SparesAllocation(
        spend=spend,
        availability=float(fleets @ up / fleets.sum()),
        backorders=float(sum(sum(o.values()) for o in owed.values())),
        depot_stock=depot_stock,
        base_stock=stock,
        base_availability={site: float(a) for site, a in zip(sites, up)},
        base_backorders=owed,
        depot_backorders=depot_backorders,
        depot_delay=depot_delay,
        method=method,
        objective=objective,
    )


# -- the inputs --------------------------------------------------------------


def _number(value, name: str, low: float, high: float = math.inf) -> float:
    """``value`` as a float in ``[low, high]`` (finite), or a
    ``ValueError`` naming it."""
    if not is_number(value) or not low <= float(value) <= high:
        bound = (
            f"at least {low:g}"
            if high == math.inf
            else (f"from {low:g} to {high:g}")
        )
        raise ValueError(f"{name} must be a number, {bound}, got {value!r}.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return float(value)


def _fleets(bases) -> Tuple[List[Hashable], np.ndarray]:
    if not isinstance(bases, Mapping) or not bases:
        raise ValueError(
            "bases must be a dict of each base's number of systems, such "
            f"as {{'north': 12, 'south': 8}}, got {bases!r}."
        )
    sites = list(bases)
    return sites, np.array(
        [
            whole_number(bases[site], f"The systems at base {site!r}")
            for site in sites
        ],
        dtype=float,
    )


def _spec(parts) -> Dict[Hashable, Mapping]:
    if not isinstance(parts, Mapping) or not parts:
        raise ValueError(
            "parts must be a dict of each part's spec, such as {'pump': "
            "{'price': 1200.0, 'base_repair_time': 48.0}}, got "
            f"{parts!r}."
        )
    for name, spec in parts.items():
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"The spec of part {name!r} must be a dict, got {spec!r}."
            )
        unknown = [key for key in spec if key not in _KEYS]
        if unknown:
            raise ValueError(
                f"The spec of part {name!r} has unknown keys {unknown}: "
                f"it takes {', '.join(map(repr, _KEYS))}."
            )
    return dict(parts)


def _by_base(value, name: str, sites, low: float, high: float) -> np.ndarray:
    """A number for every base, or ``{base: number}`` naming each, as an
    array over the bases."""
    if not isinstance(value, Mapping):
        return np.full(len(sites), _number(value, name, low, high))
    if set(value) != set(sites):
        raise ValueError(
            f"{name} by base must name every base once ({list(sites)}), "
            f"got {list(value)}: give one number for every base, or "
            "{base: number}."
        )
    return np.array(
        [
            _number(value[site], f"{name} at base {site!r}", low, high)
            for site in sites
        ]
    )


def _nodes(rbd, name, spec) -> List[Hashable]:
    """The diagram's components a part is: its ``nodes``, or the
    component of its name; none for a part with a ``demand_rate`` that
    the diagram does not draw (every system needs it)."""
    given = spec.get("nodes")
    if given is None:
        if spec.get("demand_rate") is not None and name not in rbd.components:
            return []
        given = [name]
    elif isinstance(given, (str, bytes)) or not isinstance(
        given, (list, tuple, set, frozenset)
    ):
        raise ValueError(
            f"The nodes of part {name!r} must be a list of the diagram's "
            f"components, got {given!r}."
        )
    nodes = list(dict.fromkeys(given))
    for node in nodes:
        if node not in rbd.components:
            raise ValueError(
                f"Part {name!r}: {node!r} is not a component of the "
                "diagram. Name the part's components with its 'nodes', "
                "or give its 'demand_rate' (failures per system per "
                "unit time)."
            )
    return nodes


def _every_part_needed(rbd) -> bool:
    """Whether the diagram is a series system: a system missing any part
    is down."""
    from repyability.rbd.routes import Refused

    try:
        rbd._require_series()
    except Refused:
        return False
    return True


def require(rbd, nodes: Optional[Sequence[Hashable]] = None) -> None:
    """Raise unless the diagram is one the stock over echelons is worked
    out for: no sub-system, a structure worked out unless every
    component is needed, and the replacements of ``nodes`` (by default
    every component's) those of a demand that goes on, with a crew
    always free."""
    from repyability.rbd.repairable_rbd import RepairableRBD

    for node, component in rbd.components.items():
        if isinstance(component, RepairableRBD):
            raise NotImplementedError(
                f"Node {node!r} is a sub-system, whose parts are not "
                "weighed over echelons yet: draw its components in the "
                "diagram itself."
            )
    if not _every_part_needed(rbd):
        meshed = rbd._too_meshed()
        if meshed is not None:
            raise NotImplementedError(
                "The stock over echelons takes a base's availability "
                f"from the diagram at each part's chance of a hole. {meshed}"
            )
    chosen = list(rbd.components) if nodes is None else list(nodes)
    if chosen:
        _crews._require_unlimited_crews(rbd, *_CREWS)
    for node in chosen:
        _spares._replacements(rbd, node, True)


def _parts(rbd, fleets: np.ndarray, parts, sites=None) -> List[_Part]:
    if sites is None:
        sites = list(range(len(fleets)))
    specs = _spec(parts)
    drawn = {name: _nodes(rbd, name, spec) for name, spec in specs.items()}
    seen: Dict[Hashable, Hashable] = {}
    for name, nodes in drawn.items():
        for node in nodes:
            if node in seen:
                raise ValueError(
                    f"Component {node!r} is in parts {seen[node]!r} and "
                    f"{name!r}: a component is one part's."
                )
            seen[node] = name
    require(
        rbd,
        [
            node
            for node, name in seen.items()
            if specs[name].get("demand_rate") is None
        ],
    )
    out = []
    for name, spec in specs.items():
        what = f"Part {name!r}:"
        nodes = drawn[name]
        if spec.get("demand_rate") is None:
            rate = sum(
                _spares.rate(_spares._replacements(rbd, node, True))
                for node in nodes
            )
        else:
            rate = _number(spec["demand_rate"], f"{what} demand_rate", 0.0)
        quantity = whole_number(
            spec.get("quantity", max(len(nodes), 1)), f"{what} quantity"
        )
        if "price" not in spec:
            raise ValueError(
                f"{what} give its 'price', what one spare costs (in "
                "any unit: the front's spend is in it)."
            )
        price = _number(spec["price"], f"{what} price", 0.0)
        if price <= 0.0:
            raise ValueError(f"{what} price must be above 0, got {price!r}.")
        fraction = _by_base(
            spec.get("base_repair_fraction", 1.0),
            f"{what} base_repair_fraction",
            sites,
            0.0,
            1.0,
        )
        said = spec.get("base_repair_fraction", 1.0)
        said = dict(said) if isinstance(said, Mapping) else f"{said:g}"
        times: Dict[str, Any] = {}
        for key, needed, why in (
            ("base_repair_time", np.any(fraction > 0.0), "a base repairs it"),
            (
                "depot_repair_time",
                np.any(fraction < 1.0),
                "the depot repairs it",
            ),
            (
                "ship_time",
                np.any(fraction < 1.0),
                "a base orders it from the depot",
            ),
        ):
            if spec.get(key) is None:
                if needed:
                    raise ValueError(
                        f"{what} give its {key!r} (a mean time, in the "
                        f"models' unit), as {why} (base_repair_fraction "
                        f"is {said})."
                    )
                times[key] = np.zeros(len(sites)) if key in _BY_BASE else 0.0
            elif key in _BY_BASE:
                times[key] = _by_base(
                    spec[key], f"{what} {key}", sites, 0.0, math.inf
                )
            elif isinstance(spec[key], Mapping):
                raise ValueError(
                    f"{what} {key} is the depot's, one number, got "
                    f"{spec[key]!r}; base_repair_fraction, "
                    "base_repair_time and ship_time may differ by base."
                )
            else:
                times[key] = _number(spec[key], f"{what} {key}", 0.0)
        out.append(
            _Part(
                name,
                price,
                quantity,
                fleets * rate,
                fraction,
                times["base_repair_time"],
                times["depot_repair_time"],
                times["ship_time"],
                tuple(nodes),
            )
        )
    return out


def _limits(budget, target_availability) -> Tuple[float, float]:
    if budget is None and target_availability is None:
        raise ValueError(
            "Give a budget (the most to spend), a target_availability, "
            "or both: the front goes on until one is reached."
        )
    most = math.inf
    if budget is not None:
        most = _number(budget, "budget", 0.0)
    target = math.inf
    if target_availability is not None:
        target = (
            float(target_availability)
            if is_number(target_availability)
            else math.nan
        )
        if not 0.0 < target < 1.0:
            raise ValueError(
                "target_availability must be in (0, 1), got "
                f"{target_availability!r}."
            )
    return most, target


# -- the front ---------------------------------------------------------------


def spares_front(
    rbd,
    bases,
    parts,
    *,
    budget,
    target_availability,
    method: str,
    objective: str,
) -> List[SparesAllocation]:
    """See ``RepairableRBD.spares_front``."""
    one_of("method", method, METHODS)
    one_of("objective", objective, OBJECTIVES)
    sites, fleets = _fleets(bases)
    most, target = _limits(budget, target_availability)
    curves = [
        _Curve(part, method) for part in _parts(rbd, fleets, parts, sites)
    ]
    drawn = None if _every_part_needed(rbd) else rbd
    if objective == "auto":
        objective = "backorders" if drawn is None else "availability"
    fleet = None
    if objective == "availability":
        fleet = _Fleet(fleets, drawn, curves)
        for curve in curves:
            curve.fleet = fleet
            curve._hull()

    def point() -> SparesAllocation:
        return _allocation(sites, fleets, curves, method, drawn, objective)

    front = [point()]
    while front[-1].availability < target:
        best: Optional[Tuple[float, _Curve, int]] = None
        for curve in curves:
            step = curve.step()
            if step is not None:
                gain = step[0] / curve.part.price
                if best is None or gain > best[0]:
                    best = (gain, curve, step[1])
        if best is None:
            _require_a_step(front[-1], objective)
            break
        _, curve, count = best
        before = curve.at
        curve.take(count)
        if fleet is not None:
            # Every part's loss is at the others' stocks: drawn again.
            for other in curves:
                other._hull()
        reached = point()
        if reached.spend > most:
            curve.take(before)
            break
        front.append(reached)
    return front


def _require_a_step(point: SparesAllocation, objective: str) -> None:
    """Raise if the front by availability has ended at a base that is
    never up: with two parts each owed more than the base has places, no
    one part's spares raise its availability, so no step is found."""
    stuck = [base for base, up in point.base_availability.items() if up <= 0.0]
    if objective == "availability" and stuck:
        raise ValueError(
            f"At base {stuck[0]!r} more than one part is owed in every "
            "place it has (the demand in a resupply time is more than "
            "the base's systems hold), so no one part's spares raise its "
            "availability and the front by availability ends at "
            f"{point.spend:g}: objective='backorders' gives the front "
            "by the backorders removed."
        )


def spares_allocation(
    rbd,
    bases,
    parts,
    *,
    budget,
    target_availability,
    method: str,
    objective: str,
) -> SparesAllocation:
    """See ``RepairableRBD.spares_allocation``."""
    front = spares_front(
        rbd,
        bases,
        parts,
        budget=budget,
        target_availability=target_availability,
        method=method,
        objective=objective,
    )
    chosen = front[-1]
    if (
        target_availability is not None
        and chosen.availability < target_availability
    ):
        raise ValueError(
            f"No stock on the front within the budget of {budget!r} "
            f"reaches an availability of {target_availability!r}: the "
            f"best within it gives {chosen.availability:.6g} for "
            f"{chosen.spend:g}. Raise the budget, or lower the target "
            "(spares_front gives every point within the budget)."
        )
    return chosen


__all__: List[Any] = []
