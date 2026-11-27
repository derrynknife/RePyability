# Spares

Every failure uses a spare, and so does every preventive replacement. How
many will a fleet use in a year, and how many should the store hold when a
spare takes twelve weeks to arrive? `RepairableRBD.spares_demand` and
`RepairableRBD.spares_stock` answer both for each component of a diagram,
from the models behind its availability and costs.

## Demand over a horizon

A pump that wears out (Weibull, MTTF 1,775 hours) and takes about 8 hours to
replace, and a seal with a constant failure rate, replaced in no time, in
each of a fleet of 20 systems:

```python
import surpyval as surv
from repyability import RepairableRBD

pump = {"reliability": surv.Weibull.from_params([2000.0, 2.5]),      # MTTF 1,775 h
        "repairability": surv.LogNormal.from_params([2.0, 0.5])}     # about 8 h
seal = {"reliability": surv.Exponential.from_params([1 / 4000.0]),   # MTTF 4,000 h
        "repairability": "instant"}
edges = [("s", "pump"), ("pump", "seal"), ("seal", "t")]
rbd = RepairableRBD(edges, {"pump": pump, "seal": seal})
demand = rbd.spares_demand(8760.0, fleet=20)    # a year, from new
demand["pump"].mean            # -> 90.17
demand["pump"].std             # -> 4.47
demand["pump"].stock(0.95)     # -> 98   pumps last the year with probability 0.95
demand["seal"].mean            # -> 43.80   Poisson: 20 x 8,760 / 4,000
demand["seal"].stock(0.95)     # -> 55
```

Each result is the distribution of the spares used: `probabilities` (of 0,
1, 2, ... spares) with its `mean` and `std` (properties), `covered(s)`, the
probability that `s` spares last the horizon, and `stock(p)`, the fewest
that last it with probability `p`. The components are counted from new, and
a fleet's systems independently; `nodes` picks the components (by default
all of them). A sub-system (a nested `RepairableRBD`) is counted by its
components, each as `(node, component)`, exactly through its own models,
the same as the system drawn flat, or by simulation from its own runs on
the same seed (nothing in the parent changes them, but a shutdown that
stands it still or a group that renews it, refused). A part may pool
spares across levels (`parts={"seal": ["p", ("q", "a")]}`): their demands
are independent, so its count is their sum. (The long-run stock and the
simulated shelf still refuse a part across levels.)

The seals' count is Poisson, as for any constant failure rate with instant
replacement. The pumps' is not:

- **New pumps rarely fail young.** The fleet uses 90.2 pumps in its first
  year, against 98.3 a year in the long run (20 × 8,760 / 1,783, a pump's
  cycle being its life and its replacement).
- **Worn pumps fail at regular intervals.** The count's standard deviation
  is 4.5, against 9.5 for a Poisson count with the same mean, so 98 pumps
  last the year with probability 0.95 where a Poisson estimate would hold
  106.

Preventive replacement uses spares too. Replacing each pump at 1,000 hours
of age nearly doubles the pumps used (see
[Costs](system-maintenance.md#preventive-maintenance) for whether it pays):

```python
renewed = dict(pump, preventive={"interval": 1000.0,
                                 "duration": surv.Exponential.from_params([1 / 4.0])})
maintained = RepairableRBD(edges, {"pump": renewed, "seal": seal})
maintained.spares_demand(8760.0, fleet=20, nodes=["pump"])["pump"].mean   # -> 172.34
```

## Stock with a lead time

A store that reorders each spare as it is used, and receives it a lead time
later (one-for-one, or `(S − 1, S)`, replenishment), holds its stock `S`
less the spares on order: those used in the last lead time. `spares_stock`
finds the fewest to hold, in the long run, for a `fill_rate` (the fraction
of demands met from the shelf), a `stockout_probability` (the fraction of
time the shelf is empty), or both:

```python
stock = rbd.spares_stock(2016.0, fill_rate=0.95, fleet=20)   # 12 weeks to arrive
stock["pump"].stock                   # -> 28
stock["pump"].fill_rate               # -> 0.9737
stock["pump"].stockout_probability    # -> 0.0380
stock["seal"].stock                   # -> 16
```

Each result holds the distribution of the spares on order at a random time
were the demand to go on while a repair waits (`on_order`: 22.6 pumps on
average) and as a demand finds them (`on_order_at_demand`), and
`fill_rate_for(s)` and `stockout_probability_for(s)` give what any other
stock achieves:

```python
stock["pump"].fill_rate_for(26)       # -> 0.8942
```

- **A unit waiting for a spare makes no demand.** It is down until the
  spare comes, so when the shelf is empty fewer units can fail and the
  demand slows: of the 20 seals, `k - S` are waiting when `k > S` are on
  order. The values for a stock take that in, as a finite source (a
  machine-repairman model), and `on_order_for(s)` is the distribution
  then. For units with one constant failure rate, replaced in no time,
  as the seals, it is exact (`exact` is True): one spare for one such
  unit with `λL = 0.3` meets `1 / 1.3 = 76.9%` of demands, where the
  demand going on would say `exp(-0.3) = 74.1%`. Otherwise, as for the
  pumps, it is an approximation (`exact` is False): within the
  simulated shelf's error where the lead time is short of a life, and
  short of it (on the side of holding more) where stock-outs are common;
  for wear-out lives and a lead time of several lives, by much. Check
  those with `method="simulate"` (below).
- **Poisson demand**, as the seals', would find the shelf as it is at a
  random time; with the waiting units, a demand comes less often while
  the shelf is empty, so its fill rate (0.9530) is a little more than one
  minus its stock-out probability (0.0493).
- **Wear-out demand is steadier.** A Poisson estimate at the pumps' rate
  would hold 32 pumps, where 28 meet the fill rate. And a demand finds
  fewer on order than a random time does (a pump that has just failed is
  unlikely to have failed shortly before), so the pumps' fill rate is
  higher than one minus their stock-out probability.
- **The lead time** is fixed: a supplier's delivery, or, for a repairable
  spare sent to a repair shop and back to the shelf, the shop's turnaround.
- **A fleet** shares one store, its systems' independent demands adding
  up.

## One shelf for interchangeable parts

The same part often serves several positions: the seals of a station's
three pumps come from one bin. `parts={part: [nodes]}` pools their spares
under the part's name (#183), for `spares_demand` and `spares_stock` alike.
A pooled shelf needs fewer spares than one for each position, as the
positions seldom all draw on it at once:

```python
seal = {"reliability": surv.Weibull.from_params([4000.0, 1.8]),
        "repairability": surv.LogNormal.from_params([2.0, 0.5])}
station = RepairableRBD(
    [("s", f"seal{i}") for i in (1, 2, 3)] + [(f"seal{i}", "t") for i in (1, 2, 3)],
    {f"seal{i}": dict(seal) for i in (1, 2, 3)}, k={"t": 2})   # 2 of 3 trains
six_weeks = 6 * 168.0
each = station.spares_stock(six_weeks, fill_rate=0.95, fleet=13)
sum(s.stock for s in each.values())   # -> 21   7 for each position
shelf = station.spares_stock(six_weeks, fill_rate=0.95, fleet=13,
                             parts={"seal": ["seal1", "seal2", "seal3"]})
shelf["seal"].stock                   # -> 17
shelf["seal"].fill_rate               # -> 0.9714
```

The positions' demands are independent, so a part's is their sum: for its
stock, a demand comes from position `i` with its share of the long-run
replacement rates, and finds `i`'s spares on order as its own demands do
and the others' as at a random time. The positions may differ, one under
age replacement and the others not. A node is in one part at most, and a
part is named apart from the components; given with `parts`, `nodes` (by
default none then) still counts components on their own. Two members of
one common-cause group are refused, as their shared causes replace them
together.

## How many to hold to be sure

The stock above is the long run's. To be *sure* of getting through a year
(or any horizon) without a repair waiting, ask `spares_stock` to simulate
the shelf, `method="simulate"`: each run starts with `S` on the shelf, a
spare used is reordered at once and arrives the lead time later, and a
repair that finds the shelf empty waits for the next to arrive, the unit
down meanwhile (so its next failure comes later). Stocks 0, 1, 2, ... are
run on one seed until each meets its target:

```python
sure = rbd.spares_stock(2016.0, method="simulate", horizon=8760.0,
                        no_stockout=0.9, mc_samples=10000, seed=1)
sure["pump"].stock                # -> 3
sure["pump"].no_stockout          # -> 0.9927
sure["pump"].no_stockout_bound    # -> 0.9911
sure["pump"].by_stock[2]["no_stockout"]   # -> 0.8169
```

- **`no_stockout`** is the chance of no stock-out at all over the
  horizon, from new with a full shelf. The target is met by its lower
  confidence bound (Clopper-Pearson, at `confidence=0.95` by default), so
  three pumps are "90% sure, with 95% confidence"; two get through only
  82% of years. `fill_rate` and `stockout_probability` can be asked for
  too, over the horizon (their estimates), and every stock tried is in
  `by_stock`.
- **What takes a spare:** each replacement, as `spares_demand` counts
  them (a failure repaired as new, a group's unit failures). A failure
  repaired imperfectly (the spec's `"repair"`) is a repair, and takes one
  only with `repairs=True`.
- **When the lead time is longer than the horizon**, nothing ordered
  arrives in it, and the chance is exact without simulating:
  `spares_demand(horizon)[node].covered(S)`.
- **With repair crews**, a repair that waits for its spare queues for a
  crew only once the spare is in: no crew is held while it is on its way.
- **A preventive task** that replaces the unit takes a spare too. With the
  shelf empty the task is put off until the spare is in, the unit running
  on as it is; should it fail first, the failure takes the spare ordered
  for the task, and waits for it.
- **A test that finds a hidden failure** takes a spare, and the repair
  starts once the test is over and the spare is in.
- **A fleet** (`fleet=20`) draws on one shelf: its systems are run as one
  diagram of their components (refused where repair crews, maintenance or
  common-cause groups, shutdowns or sub-systems would couple them).
- **A sub-system's components** each have a shelf of their own, keyed
  `(node, component)`, run in the sub-system's own loop. A shelf shared
  across levels (a top-level component and a sub-system's) is refused, as
  the two loops' demands would come out of time order.
- `parts=` pools shelves. The shelf is simulated in Python only (the
  compiled engine refuses it), and not yet for replacement on condition,
  maintenance groups, a sub-system's tasks, tests or crews, nor (with
  repair crews) for tested units or a unit a shutdown pauses, which are
  refused with the reason.

To keep a bad year's *downtime* under a limit rather than avoid every
stock-out, the spares are a lever of `least_spend_for_downtime` (see
[Costs](costs.md#downtime-you-can-be-sure-of)): each spare more on a
node's shelf at its holding price, every design run with the shelf.

```python
found = rbd.least_spend_for_downtime(
    8760.0, 200.0,
    spares={"pump": {"price": 1500.0, "lead_time": 2016.0},
            "seal": {"price": 50.0, "lead_time": 2016.0}},
    mc_samples=2000, seed=1)
found.design.spares["pump"], found.design.spares["seal"]   # (3, 4)
found.design.spend    # -> 4700.0
```

## A shelf in the diagram

The shelf can be part of the diagram itself, a component's `"spares"`:
its `"stock"` at the start, its `"lead_time"`, and optionally
`"repairs"` (whether a failure repaired imperfectly takes a spare). Every
simulation then runs with it: `availability()`, `cost()`,
`simulate_timelines()`, `spares_demand(method="simulate")` and the rest,
on the rules above, in Python.

```python
stocked = dict(pump, spares={"stock": 2, "lead_time": 2016.0})
kept = RepairableRBD(edges, {"pump": stocked, "seal": seal})
kept.availability(8760.0, mc_samples=2000, seed=1).mean_availability   # -> 0.9854
```

The pumps' waits for a spare take the year's availability from 0.9937,
with a pump always on hand, to 0.9854.

- **The exact and numerical analyses refuse the node** (the long run,
  the values over time, the expected events and costs, the importance
  measures, the spares used exactly, ...), saying to simulate: none of
  them takes the waits for a spare. `analysis_routes()` says which.
- **`spares_stock(method="simulate")` and the `spares` lever** try shelves
  of their own on the nodes they are given; the other nodes keep their
  own. The lever starts from the node's shelf: its stock, lead time and
  `repairs` are the lever's defaults, so `spares={"pump": {"price":
  1500.0}}` is enough. A node's copies draw on its shelf.
- **A sub-system's component** keeps its shelf in the sub-system's own
  loop, as `(node, component)` above. A shelf two levels down is refused.
- **What it does not take** is refused when the diagram is built, saying
  what to do: replacement on condition (replace at an age or on a block,
  whose task waits for its spare), a maintenance group or a shutdown's
  renewals (take the unit out of them), tests with repair crews (no
  crews, so each repair has its own), and, in a sub-system, tasks, tests
  or repair crews (simulate the sub-system as a diagram of its own). The
  diagram saves and loads with its shelves.

## Bases and a depot

A fleet's repairable spares are held where its systems run and at a depot
behind them. A failed part is swapped for one from the base's shelf and
repaired: at the base if it can be, or else at the depot, the base
ordering a replacement from the depot at once. Every shelf is replenished
one for one. A system with no part to fit waits for one, a *backorder*,
and the question is the system's: how many of each part at each site for
the most availability per dollar. `spares_front` gives the spend against
the fleet's availability, and `spares_allocation` the stock for a target
or a budget (Sherbrooke's METRIC and VARI-METRIC).

Twenty systems at two bases, each with a pump, a valve and two
controllers, every one needed:

```python
E = surv.Exponential.from_params
swap = lambda mttf: {"reliability": E([1 / mttf]), "repairability": "instant"}
names = ["pump", "valve", "c1", "c2"]
system = RepairableRBD(
    list(zip(["s"] + names, names + ["t"])),
    {"pump": swap(2000.0), "valve": swap(5000.0),
     "c1": swap(8000.0), "c2": swap(8000.0)})
bases = {"north": 12, "south": 8}
parts = {
    "pump": {"price": 12000.0, "base_repair_fraction": 0.3,
             "base_repair_time": 48.0, "depot_repair_time": 600.0,
             "ship_time": 72.0},
    "valve": {"price": 3000.0, "base_repair_fraction": 0.5,
              "base_repair_time": 24.0, "depot_repair_time": 400.0,
              "ship_time": 72.0},
    "controller": {"price": 25000.0, "nodes": ["c1", "c2"],
                   "base_repair_fraction": 0.1, "base_repair_time": 72.0,
                   "depot_repair_time": 900.0, "ship_time": 120.0},
}
held = system.spares_allocation(bases, parts, target_availability=0.95)
held.spend                           # -> 246000.0
held.availability                    # -> 0.9578
held.backorders                      # -> 0.8563
held.depot_stock                     # {'pump': 4, 'valve': 2, 'controller': 4}
held.base_stock["north"]             # {'pump': 2, 'valve': 1, 'controller': 1}
held.base_stock["south"]             # {'pump': 1, 'valve': 1, 'controller': 1}
held.base_availability["south"]      # -> 0.9547
held.depot_delay["pump"]             # -> 128.37   hours an order waits at the depot
```

A part's spec gives its `"price"`, the chance a failure is repaired at
the base (`"base_repair_fraction"`, by default 1: no depot), the mean
repair times at the base and the depot, and the `"ship_time"` from a
base's order to its arrival once the depot has one to send. Its demand
is, by default, the long-run replacement rate of its components in the
diagram (its `"nodes"`, or the component of its name), as `spares_stock`
counts them, and its `"quantity"` in a system the number of those nodes;
`"demand_rate"` (failures per system per unit time) gives it directly.

`spares_front` returns every point on the way, from nothing held, until
a `budget` or a `target_availability` is reached:

```python
front = system.spares_front(bases, parts, budget=60000.0)
len(front)                           # -> 8
front[0].availability                # -> 0.5631
front[-1].spend                      # -> 57000.0
front[-1].availability               # -> 0.7217
```

- **The pipelines.** With Poisson demand, the parts in a repair shop are
  Poisson with mean the demand times the mean repair time, whatever the
  repair times' distribution (Palm's theorem). The depot's pipeline is
  exactly that, and a base's order waits the depot's expected backorders
  over its demand. A base's pipeline (its own repairs and its orders) has
  a known mean; `method="metric"` takes it as Poisson, which understates
  the backorders, as the depot's delays come in runs. The default,
  `"vari-metric"`, gives the pipeline its variance too and a negative
  binomial of that mean and variance. Here METRIC would stop at 218,000
  and call it 0.9500, a stock VARI-METRIC puts at 0.9396.
- **Availability.** A base's is the product over parts of `(1 − EBO /
  (N Z))^Z`: `EBO` its expected backorders of the part, `N` its systems
  and `Z` the part's number in each. The fleet's is the bases' weighed by
  their systems. That is every part needed, a series system; on any
  other diagram, see [redundant parts](#redundant-parts) below.
- **The front.** For each part, the best split of each count between the
  depot and the bases is found, and the counts on the lower convex hull
  of backorders against count are kept; the parts' steps are then taken
  in the order of backorders removed per dollar (marginal analysis). Each
  point has the fewest expected backorders of any stock at its spend or
  less. A count off the hull is not on the front, and the split of a
  part's count may change from one point to the next (a valve moves from
  the depot to the bases at 30,000 above).
- **The demand goes on** while a system waits, as METRIC takes it, where
  `spares_stock` slows it (a system waiting makes no demand): the two
  agree where backorders are few, which is where a target puts the stock.
  With one base that repairs everything itself, the base's pipeline is
  `spares_stock`'s `on_order` for a constant failure rate.
- **Not yet:** parts within parts (MOD-METRIC), cannibalisation, and
  lateral supply between bases.

### A base's own times

Bases differ: a remote one repairs fewer parts itself and waits longer
for the depot. A part's `"base_repair_fraction"`, `"base_repair_time"`
and `"ship_time"` each take one number, or `{base: value}` naming every
base (the depot's repair time is the depot's, one number). The fleet
above, with the south base repairing 10% of its pumps in 96 hours and
waiting 240 hours for an order (360 for a controller):

```python
own = {
    "pump": {**parts["pump"],
             "base_repair_fraction": {"north": 0.3, "south": 0.1},
             "base_repair_time": {"north": 48.0, "south": 96.0},
             "ship_time": {"north": 72.0, "south": 240.0}},
    "valve": {**parts["valve"],
              "ship_time": {"north": 72.0, "south": 240.0}},
    "controller": {**parts["controller"],
                   "ship_time": {"north": 120.0, "south": 360.0}},
}
remote = system.spares_allocation(bases, own, target_availability=0.95)
remote.spend                         # -> 295000.0
remote.availability                  # -> 0.9618
remote.depot_stock                   # {'pump': 4, 'valve': 2, 'controller': 3}
remote.base_stock["south"]           # {'pump': 3, 'valve': 1, 'controller': 2}
```

The stock moves from the depot to the bases, and to the south: the
stock found for bases alike would give this fleet 0.9196.

### Redundant parts

On a diagram that is not a series system a missing part need not ground
a system. A base's availability is then the diagram's own, with each of
a part's places empty with chance `EBO / (N Z)`, independently: for two
pumps, either enough, `1 − q²`. And the stock is chosen by what it buys:
`objective="availability"` takes, at each step, the spares that add the
most fleet availability per dollar from the stock reached, where
`"backorders"` counts a backorder of a pump a system does without as one
that grounds it. The default, `"auto"`, is `"availability"` for a diagram
that is not a series system and `"backorders"` for one that is.

The same fleet with two pumps in each system:

```python
paired = RepairableRBD(
    [("s", "p1"), ("s", "p2"), ("p1", "valve"), ("p2", "valve"),
     ("valve", "c1"), ("c1", "c2"), ("c2", "t")],
    {"p1": swap(2000.0), "p2": swap(2000.0), "valve": swap(5000.0),
     "c1": swap(8000.0), "c2": swap(8000.0)})
pairs = {**parts, "pump": {**parts["pump"], "nodes": ["p1", "p2"]}}
held = paired.spares_allocation(bases, pairs, target_availability=0.95)
held.objective                       # 'availability'
held.spend                           # -> 210000.0
held.availability                    # -> 0.9544
held.depot_stock                     # {'pump': 4, 'valve': 2, 'controller': 4}
held.base_stock["north"]             # {'pump': 0, 'valve': 1, 'controller': 1}
owed = paired.spares_allocation(
    bases, pairs, target_availability=0.95, objective="backorders")
owed.spend                           # -> 278000.0
```

- **What is approximate.** The holes are taken as independent. For a
  pair that understates the chance of both being missing, by about a
  seventh of it in the cases checked (backorders vary more than their
  mean says), so the availability of a redundant part's stage is a
  little optimistic.
- **The front by availability** has the front by backorders' guarantee
  for one part (no stock at a point's spend or less is more available).
  With several parts each step is the best per dollar from the stock
  reached, which is not a proof for the whole stock; and the split of
  every part between the depot and the bases is chosen again at each
  point.
- A part given by its `"demand_rate"` that is no component of the
  diagram is needed by every system. A part's `"nodes"` say where it
  sits; with a `"demand_rate"` too, that replaces the nodes' own rates.
- A diagram with a sub-system is refused: draw its components in the
  diagram itself.

## Block replacement and proof tests

Two kinds of component replace on a calendar, and are counted their own way
(#147):

- **Block replacement** renews the unit at every multiple of its
  interval, whatever its age, unless it is down then (being repaired or
  replaced), when that replacement is skipped. Replacing the pumps every
  1,000 hours of the calendar, in about 4 hours, uses more pumps than
  replacing each at 1,000 hours of its age, since young pumps are replaced
  too:

  ```python
  blocked = dict(pump, preventive={"interval": 1000.0, "policy": "block",
                                   "duration": surv.Exponential.from_params([1 / 4.0])})
  calendar = RepairableRBD(edges, {"pump": blocked, "seal": seal})
  calendar.spares_demand(8760.0, fleet=20, nodes=["pump"])["pump"].mean   # -> 187.32
  ```

- **Hidden failures found by proof tests**, tested and repaired in no time,
  use a spare at each test that finds a failure. A valve that wears out
  (MTTF about 15 years), tested yearly, in a fleet of 50 over 20 years, and
  its stock for a lead time of 26 weeks:

  ```python
  valve = {"reliability": surv.Weibull.from_params([150_000.0, 2.5]),
           "repairability": "instant",
           "inspection": {"interval": 8760.0}}
  valves = RepairableRBD([("s", "v"), ("v", "t")], {"v": valve})
  used = valves.spares_demand(20 * 8760.0, fleet=50)["v"]
  used.mean         # -> 41.21
  used.stock(0.95)  # -> 48
  valves.spares_stock(26 * 7 * 24.0, fill_rate=0.95, fleet=50)["v"].stock   # -> 5
  ```

A block-replaced component's stock is worked out too (#160). With its
repairs and block replacements in no time, each block interval starts with
a new unit, so the demand repeats every interval, and a lead time's is
averaged over where in the interval it starts. Pumps swapped in no time
every 1,000 hours of the calendar need more on the shelf than the 46 for
pumps swapped at 1,000 hours of their age, as young pumps are swapped too:

```python
swapped = dict(pump, repairability="instant",
               preventive={"interval": 1000.0, "policy": "block"})
shelf = RepairableRBD(edges, {"pump": swapped, "seal": seal})
stock = shelf.spares_stock(2016.0, fill_rate=0.95, fleet=20, nodes=["pump"])["pump"]
stock.stock       # -> 52
stock.fill_rate   # -> 0.9618
stock.method      # -> 'simulated'
```

Each pump's demand in the lead time is more than one, its life is not at a
constant rate, and stock-outs are common one spare short, so the stock is
found with the shelf in the runs (the approximation said 0.9596 for 52):
`stock.reason` says so, and `stock.simulated` has every stock tried.

A demand finds at least two pumps of its own system on order: those swapped
at the two block times in the 12 weeks before it. When repairs or block
replacements take time, a pump down at a block time is not swapped there,
and one still being repaired or swapped carries over into the next
interval, which then need not start with a new pump. The demand is then
counted from a typical replacement in the long run: a failure at some
point of the interval, or a block replacement. The `calendar` pumps above,
repaired in about 8 hours and swapped in about 4:

```python
stock = calendar.spares_stock(2016.0, fill_rate=0.95, fleet=20, nodes=["pump"])["pump"]
stock.stock       # -> 52
stock.fill_rate   # -> 0.9684
```

(found with the shelf in the runs too, as above; the approximation said
0.9667.)

Their time out of service lowers their demand a little (1.160 replacements
every 1,000 hours, against 1.165), and the same stock fills more of it.

A fleet's systems are taken as on block schedules of their own, out of step
with each other; two block-replaced components in one part are refused, as
their block times keep step. A unit that can be dead on arrival fails as it
is put in, so with repairs and block replacements in no time its
replacements come several at one instant: each replacement of a unit that
lasts brings a geometric number more, which the lasting units' count is
compounded with (#322). A tested component's
spares are counted whatever its tests and repairs take, and whether or not
its tests can miss a failure: the replacements still fall on the tests that
find failures (#159).

## How it is computed

A component is replaced at each failure and each preventive replacement,
then repaired or maintained, and starts again as new: its replacements are
a renewal process. Its `n`-th replacement comes after `n − 1` whole cycles
(an up time, the smaller of its life and its replacement age, then a repair
or maintenance time) and one more up time, so the probability of `n` or more
by the horizon is that of their sum falling within it. `spares_demand`
works that out on a grid, refined until the probabilities agree to about
`1e-6`. The atoms (a replacement age, work in no time) are kept exact, and
the count is over `[0, horizon)`, as the simulation counts: a replacement at
the horizon itself falls after it, so that horizons one after another add
up.

`spares_stock` counts the same way in a lead time in the long run. From a
random time, the first replacement ends what is left of an up time, or
follows what is left of a repair or maintenance and an up time. The
stock-out probability is that of `S` or more replacements in a lead time
from a random time; the fill rate, that of fewer than `S` in the lead time
before a replacement. Both methods report `numerical` in
[`analysis_routes`](saving.md#what-is-exact-and-what-is-simulated).

Under block replacement, a new unit's next replacement is at its failure,
if that comes before the next block time, and at the block time otherwise.
So each replacement's time follows from the one before, block interval by
block interval, on a grid with the block times on it. A repair or
replacement still going on at a block time carries the next unit's start
past it, as in the simulation. A horizon on a block time is read just
before it, where the replacements' distributions start afresh. In a lead
time in the long run, with repairs and block replacements in no time, the
failures within an interval are a renewal process from new; a lead time
that runs past the interval's end adds the block replacement there and the
count from new past it. Averaged over where the lead time starts, in
exchanged order (over the phase, and over where the unit then in service
started), this needs only sums over one grid of the interval. Before a
replacement the count is that after one, as the times between replacements
are stationary from one: after a failure, at the renewal density, or after
a block replacement.

With repairs or block replacements that take time, an interval need not
start with a new unit, and the count is taken from a typical replacement in
the long run (its Palm distribution): a failure at each point of the
interval, weighted by the long-run failures there, or a block replacement,
weighted by its chance an interval, the weights coming from intervals
followed one after another until they settle. Each is followed block
interval by block interval, all of them at once on one grid, which gives
the chance that the `s`-th replacement after it falls within the lead time,
`G_s(τ)`: the count before a replacement. From a random time, the chance of
`s` or more is the long-run rate `λ` times the integral of `G_{s−1} − G_s`
over the lead time (Campbell's formula). Only the life is rounded: the
repairs and block replacements, often far shorter than an interval, are
shared between the grid points either side of each value, keeping their
mean. The grids are refined by halving the step and extrapolated, their
error falling as its square, until two extrapolations agree to `1e-5`,
which leaves about `1e-6`. The `calendar` pumps' stock takes about
3 seconds.

With proof tests, every replacement falls on a test, and the count is that
of a discrete renewal process on the tests. Tested and repaired in no time,
it is exact, with no grid: a unit renewed at a test is renewed again `k`
tests later with probability `R((k − 1)τ) − R(kτ)`. With tests or repairs
that take time, the chances of each cycle's length in tests come from the
cycle followed test by test on a grid (see [tests and repairs that take
time](system-maintenance.md#tests-and-repairs-that-take-time)). From a random time, the
next replacement is `j` tests on with probability `P(C ≥ j) / S`, `C` being
a cycle's length in tests and `S` its mean.

With tests that can miss a failure, but a full test every so often that
finds every one, a cycle's length depends on where the test that starts it
falls between the full tests: the replacements are a Markov renewal
process over those places, counted the same way, each test's place known
from the first's. From a random time, its next test at each place as
often, the next replacement ends the cycle then in progress; before a
replacement, the cycles back are those of the process reversed, from each
place in proportion to its share of the replacements.

Some components' replacements are not counted this way, and the counts
refuse, with the reason:

- a component with **hidden failures** whose tests can last as long as
  their interval;
- a **standby group**, whose units' failures depend on each other;
- any component while **repair crews** can keep components waiting.

`spares_demand(method="simulate")` counts their spares in simulations of
the whole system instead: `mc_samples` of them (by default 10,000), seeded
with `seed`. A duty pump with a cold standby that does not age while it
waits uses about as many pumps as a single pump, a few more as the standby
takes over at once:

```python
paired = RepairableRBD(edges, {"pump": dict(pump, standby={"units": 2}), "seal": seal})
used = paired.spares_demand(8760.0, fleet=20, method="simulate", seed=1)
used["pump"].mean      # -> 90.80   simulated
```

The stock needs the long run, which the simulations from new do not reach,
so `spares_stock` has no simulated route: it refuses these components.
