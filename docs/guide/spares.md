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
demand["pump"].mean()          # -> 90.17
demand["pump"].std()           # -> 4.47
demand["pump"].stock(0.95)     # -> 98   pumps last the year with probability 0.95
demand["seal"].mean()          # -> 43.80   Poisson: 20 x 8,760 / 4,000
demand["seal"].stock(0.95)     # -> 55
```

Each result is the distribution of the spares used: `probabilities` (of 0,
1, 2, ... spares) with its `mean()` and `std()`, `covered(s)`, the
probability that `s` spares last the horizon, and `stock(p)`, the fewest
that last it with probability `p`. The components are counted from new, and
a fleet's systems independently; `nodes` picks the components (by default
all but a nested RBD, whose own `spares_demand` counts its components').

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
[Costs](costs.md#preventive-maintenance) for whether it pays):

```python
renewed = dict(pump, preventive={"interval": 1000.0,
                                 "duration": surv.Exponential.from_params([1 / 4.0])})
maintained = RepairableRBD(edges, {"pump": renewed, "seal": seal})
maintained.spares_demand(8760.0, fleet=20, nodes=["pump"])["pump"].mean()   # -> 172.34
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
stock["pump"].fill_rate               # -> 0.9724
stock["pump"].stockout_probability    # -> 0.0387
stock["seal"].stock                   # -> 17
```

Each result holds the distribution of the spares on order at a random time
(`on_order`: 22.6 pumps on average) and as a demand finds them
(`on_order_at_demand`), and `fill_rate_for(s)` and
`stockout_probability_for(s)` give what any other stock achieves:

```python
stock["pump"].fill_rate_for(26)       # -> 0.8870
```

- **Poisson demand**, as the seals', finds the shelf as it is at a random
  time: its fill rate is one minus its stock-out probability, and the stock
  is the textbook Poisson one.
- **Wear-out demand is steadier.** A Poisson estimate at the pumps' rate
  would hold 32 pumps, where 28 meet the fill rate. And a demand finds
  fewer on order than a random time does (a pump that has just failed is
  unlikely to have failed shortly before), so the pumps' fill rate is
  higher than one minus their stock-out probability.
- **The lead time** is fixed: a supplier's delivery, or, for a repairable
  spare sent to a repair shop and back to the shelf, the shop's turnaround.
- **A fleet** shares one store, its systems' independent demands adding
  up.

## How it is computed

A component is replaced at each failure and each preventive replacement,
then repaired or maintained, and starts again as new: its replacements are
a renewal process. Its `n`-th replacement comes after `n − 1` whole cycles
(an up time, the smaller of its life and its replacement age, then a repair
or maintenance time) and one more up time, so the probability of `n` or more
by the horizon is that of their sum falling within it. `spares_demand`
works that out on a grid, refined until the probabilities agree to about
`1e-6`. The atoms (a replacement age, work in no time) are kept exact, so a
replacement that falls at the horizon counts in full.

`spares_stock` counts the same way in a lead time in the long run. From a
random time, the first replacement ends what is left of an up time, or
follows what is left of a repair or maintenance and an up time. The
stock-out probability is that of `S` or more replacements in a lead time
from a random time; the fill rate, that of fewer than `S` in the lead time
before a replacement. Both methods report `numerical` in
[`analysis_routes`](saving.md#what-is-exact-and-what-is-simulated).

Some components' replacements are not a renewal process, and the counts
refuse, with the reason:

- a component under **block replacement**, replaced on schedule whatever
  its age;
- a component with **hidden failures** found by inspection;
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
used["pump"].mean()    # -> 90.38   simulated
```

The stock needs the long run, which the simulations from new do not reach,
so `spares_stock` has no simulated route: it refuses these components.
