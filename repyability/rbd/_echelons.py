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
- a base's backorders with ``s`` held are ``E[(X - s)+]``, and its
  availability the product over parts of ``(1 - EBO / (N Z))^Z``.

The front (``spares_front``) is the textbook's marginal analysis: for
each part, the fewest backorders of each count held in all (``_counts``),
the lower convex hull of those (``_Curve``), and the parts' hull steps
taken in the order of backorders removed per unit of spend.
"""

import math
from typing import (
    Any,
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
    unit time, the chance a failure is repaired at the base, and the mean
    times of a base repair, a depot repair and an order from the depot."""

    name: Hashable
    price: float
    quantity: int
    demand: np.ndarray
    fraction: float
    base_time: float
    depot_time: float
    ship_time: float


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
    demand = float(part.demand.sum()) * (1.0 - part.fraction)
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
    own = demand * (
        part.fraction * part.base_time + (1.0 - part.fraction) * part.ship_time
    )
    if depot.demand <= 0.0:
        return own, own
    share = demand * (1.0 - part.fraction) / depot.demand
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
    """The best split of a count of a part: the bases' backorders in all,
    the depot's stock and each base's."""

    backorders: float
    depot: int
    bases: Tuple[int, ...]


def _counts(part: _Part, depot: _Depot, method: str, most: int) -> List[_Held]:
    """The fewest backorders of ``0, 1, ..., most`` of a part held in
    all, and the split that gives each. For a depot stock the bases'
    spares go in the order of the backorders each removes, which is the
    best split among them (a base's backorders are convex in its stock);
    the least over the depot stocks is the best of all."""
    best: List[Optional[_Held]] = [None] * (most + 1)
    bases = len(part.demand)
    for depot_stock in range(most + 1):
        room = most - depot_stock
        owed = _base_backorders(part, depot, depot_stock, method)
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
            # Summed afresh, so that no backorders left is exactly 0.
            backorders = math.fsum(_at(o, s) for o, s in zip(owed, held))
            found = best[count]
            if found is None or backorders < found.backorders:
                best[count] = _Held(backorders, depot_stock, tuple(held))
            if added < room:
                held[steps[added][1]] += 1
    return [held for held in best if held is not None]


class _Curve:
    """A part's efficient counts: the lower convex hull of its fewest
    backorders against the count held. Counts are worked out as far as
    asked (``most``, doubled as needed); a step of the hull is given once
    no count further on can change it: backorders are never negative, so
    none can lie below the step's line where that has reached zero."""

    def __init__(self, part: _Part, method: str):
        self.part = part
        self.method = method
        self.depot = _depot(part)
        self.most = 8
        self.at = 0
        self._hull()

    def _hull(self) -> None:
        self.counts = _counts(self.part, self.depot, self.method, self.most)
        hull = [0]
        for count in range(1, self.most + 1):
            while len(hull) > 1 and self._above(hull[-2], hull[-1], count):
                hull.pop()
            hull.append(count)
        self.hull = hull

    def _above(self, a: int, b: int, c: int) -> bool:
        """Whether count ``b`` lies above the line from ``a`` to ``c``."""
        owed = [self.counts[k].backorders for k in (a, b, c)]
        return (owed[1] - owed[0]) * (c - a) > (owed[2] - owed[0]) * (b - a)

    def held(self) -> _Held:
        """The stock at the count reached."""
        return self.counts[self.at]

    def step(self) -> Optional[Tuple[float, int]]:
        """The next step of the hull from the count reached: the
        backorders it removes per part added, and the count it leads to;
        None when no backorders are left to remove."""
        while True:
            here = self.counts[self.at].backorders
            if here <= 0.0:
                return None
            after = self.hull[self.hull.index(self.at) + 1 :]
            if after:
                then = after[0]
                removed = here - self.counts[then].backorders
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


def _allocation(
    sites: Sequence[Hashable],
    fleets: np.ndarray,
    curves: Sequence[_Curve],
    method: str,
) -> SparesAllocation:
    """The result for the stock each part's curve has reached."""
    spend = 0.0
    depot_stock: Dict[Hashable, int] = {}
    depot_backorders: Dict[Hashable, float] = {}
    depot_delay: Dict[Hashable, float] = {}
    stock: Dict[Hashable, Dict[Hashable, int]] = {site: {} for site in sites}
    owed: Dict[Hashable, Dict[Hashable, float]] = {site: {} for site in sites}
    up = np.ones(len(sites))
    for curve in curves:
        part, held = curve.part, curve.held()
        spend += part.price * (held.depot + sum(held.bases))
        depot_stock[part.name] = held.depot
        waiting = _at(curve.depot.mean, held.depot)
        depot_backorders[part.name] = waiting
        depot_delay[part.name] = (
            waiting / curve.depot.demand if curve.depot.demand > 0.0 else 0.0
        )
        each = _base_backorders(part, curve.depot, held.depot, method)
        for j, site in enumerate(sites):
            stock[site][part.name] = held.bases[j]
            short = _at(each[j], held.bases[j])
            owed[site][part.name] = short
            places = fleets[j] * part.quantity
            up[j] *= max(1.0 - short / places, 0.0) ** part.quantity
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


def _nodes(rbd, name, spec) -> List[Hashable]:
    """The diagram's components a part's demand is taken from: none when
    its ``demand_rate`` is given."""
    if spec.get("demand_rate") is not None:
        if spec.get("nodes") is not None:
            raise ValueError(
                f"Part {name!r} has both a demand_rate and nodes: give "
                "one. The nodes' long-run replacement rates are the "
                "demand when no demand_rate is given."
            )
        return []
    given = spec.get("nodes")
    if given is None:
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


def require(rbd, nodes: Optional[Sequence[Hashable]] = None) -> None:
    """Raise unless the diagram is one the stock over echelons is worked
    out for: every component needed (a series system, no sub-system),
    and the replacements of ``nodes`` (by default every component's)
    those of a demand that goes on, with a crew always free."""
    from repyability.rbd.repairable_rbd import RepairableRBD
    from repyability.rbd.routes import Refused

    for node, component in rbd.components.items():
        if isinstance(component, RepairableRBD):
            raise NotImplementedError(
                f"Node {node!r} is a sub-system, whose parts are not "
                "weighed over echelons yet: draw its components in the "
                "diagram itself."
            )
    try:
        rbd._require_series()
    except Refused:
        raise NotImplementedError(
            "The stock over echelons takes every part as needed (a "
            "system missing one is down), and this diagram is not a "
            "series system: a redundant diagram is not weighed over "
            "echelons yet. spares_stock gives one site's stock of each "
            "part for a fill rate, and least_spend_for_downtime weighs "
            "spares on one site's shelves by the system's downtime."
        ) from None
    chosen = list(rbd.components) if nodes is None else list(nodes)
    if chosen:
        _crews._require_unlimited_crews(rbd, *_CREWS)
    for node in chosen:
        _spares._replacements(rbd, node, True)


def _parts(rbd, fleets: np.ndarray, parts) -> List[_Part]:
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
    require(rbd, list(seen))
    out = []
    for name, spec in specs.items():
        what = f"Part {name!r}:"
        nodes = drawn[name]
        if nodes:
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
        fraction = _number(
            spec.get("base_repair_fraction", 1.0),
            f"{what} base_repair_fraction",
            0.0,
            1.0,
        )
        times = {}
        for key, needed, why in (
            ("base_repair_time", fraction > 0.0, "a base repairs it"),
            ("depot_repair_time", fraction < 1.0, "the depot repairs it"),
            ("ship_time", fraction < 1.0, "a base orders it from the depot"),
        ):
            if spec.get(key) is None:
                if needed:
                    raise ValueError(
                        f"{what} give its {key!r} (a mean time, in the "
                        f"models' unit), as {why} (base_repair_fraction "
                        f"is {fraction:g})."
                    )
                times[key] = 0.0
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
) -> List[SparesAllocation]:
    """See ``RepairableRBD.spares_front``."""
    one_of("method", method, METHODS)
    sites, fleets = _fleets(bases)
    most, target = _limits(budget, target_availability)
    curves = [_Curve(part, method) for part in _parts(rbd, fleets, parts)]
    front = [_allocation(sites, fleets, curves, method)]
    while front[-1].availability < target:
        best: Optional[Tuple[float, _Curve, int]] = None
        for curve in curves:
            step = curve.step()
            if step is not None:
                gain = step[0] / curve.part.price
                if best is None or gain > best[0]:
                    best = (gain, curve, step[1])
        if best is None:
            break
        _, curve, count = best
        before = curve.at
        curve.take(count)
        point = _allocation(sites, fleets, curves, method)
        if point.spend > most:
            curve.take(before)
            break
        front.append(point)
    return front


def spares_allocation(
    rbd,
    bases,
    parts,
    *,
    budget,
    target_availability,
    method: str,
) -> SparesAllocation:
    """See ``RepairableRBD.spares_allocation``."""
    front = spares_front(
        rbd,
        bases,
        parts,
        budget=budget,
        target_availability=target_availability,
        method=method,
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
