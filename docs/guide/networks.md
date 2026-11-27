# Networks

Power lines, pipes and communication links carry flow both ways, and it is
often the links that fail. The question is whether two points of the
network stay connected: some path of working links joins them; or whether
several sites, or every node, stay connected to each other. A reliability
block diagram is directed, and its nodes fail, so a network has a model of
its own, `Network`.

## Two terminals

Each link is a name mapped to the two nodes it joins and its lifetime model
(or a `FixedEventProbability`). The network works while working links join
the `source` to the `target`. A generator `G` feeding a load `L` through a
ring with a tie line:

```python
import surpyval as surv
from repyability import Network

line = surv.Exponential.from_params([0.1])     # fails 0.1 times a year
cable = surv.Exponential.from_params([0.05])
links = {
    "GA": ("G", "A", line),
    "GB": ("G", "B", line),
    "AB": ("A", "B", cable),
    "AC": ("A", "C", line),
    "BC": ("B", "C", line),
    "CL": ("C", "L", cable),
    "BL": ("B", "L", line),
}
grid = Network(links, source="G", target="L")
grid.sf(1.0)       # -> 0.98509   connected through a year
grid.ff(1.0)       # -> 0.014910
grid.mean()        # -> 8.5180   years until the load is cut off
len(grid.path_sets())    # -> 7
```

- **Paths and cuts.** `path_sets()` gives the minimal path sets, one per
  simple path between the terminals (its links, and its nodes that can
  fail); `cut_sets()` the smallest sets of links whose failure parts the
  terminals. Here two pairs of lines cut the load off on their own,
  `{"BL", "CL"}` and `{"GA", "GB"}`. `minimal_path_sets()` and
  `minimal_cut_sets()` list the same sets smallest first, as a diagram's
  and a fault tree's do.
- **Importance.** `birnbaum_importance(x)` gives, per link, the
  reliability with it working less that with it failed: the two lines out
  of the generator and the cable into the load matter most.
- **Failing nodes.** `nodes={"B": model}` makes a node fail too: every path
  through it then needs it. The terminals can be given one as well.
- **Time.** `x` can be a scalar or an array of times; with every model a
  fixed probability it can be left out.

```python
substation = surv.Exponential.from_params([0.02])
Network(links, "G", "L", nodes={"B": substation}).sf(1.0)   # -> 0.98100
```

## Several terminals, and all of them

`terminals=[...]` names the nodes that must stay connected to each other,
two or more (K-terminal reliability), and `terminals="all"` every node of
the network (all-terminal reliability): the network works while working
links join each terminal to the others. Two terminals are the same network
as `source` and `target`.

```python
sites = Network(links, terminals=["G", "L", "A"])
sites.sf(1.0)       # -> 0.98468   the generator, the load and A connected
every = Network(links, terminals="all")
every.sf(1.0)       # -> 0.98425   no node cut off
every.ff(1.0)       # -> 0.015746
every.mean()        # -> 7.6104   years until some node is cut off
len(every.path_sets())          # -> 21   the spanning trees
every.minimal_cut_sets()[:2]    # -> the lines into L, the lines out of G
```

Every method works as for two terminals: the importance measures, the
B-lives, the parameter uncertainty, `compare`, the levers, the simulation
and saving (`to_dict` keeps the terminals). What changes:

- **Path sets.** A minimal path set is a smallest set of links (and failing
  nodes) that joins the terminals: a tree, and for `terminals="all"` a
  spanning tree. They are read off the decision diagram, up to 100,000
  (`repyability.network.MAX_PATHS`); more are refused, and nothing else
  needs them.
- **Failing nodes.** A terminal that fails parts the terminals, so with
  `terminals="all"` every failing node is in series with the rest.
- **The simulation.** A sample's lifetime is that of the link which joins
  the terminals when the links are added longest-lived first.
- **`METHOD = "paths"`** is two terminals', and refuses more.

The decision diagram is the same one, with a state saying which of the
frontier's groups hold a terminal rather than which hold the two: a 10-by-10
grid with every node a terminal has 0.8 million states (corner to corner it
has 1.9 million), and with three corners as terminals 3.0 million.

## Directed links

`directed=True` makes every link work one way only, from its first node to
its second: the network works while a path along the links leads from the
`source` to the `target`. With the ring's links one way, the tie line `AB`
carries from `A` to `B` only, and five of the seven paths are left:

```python
one_way = Network(links, source="G", target="L", directed=True)
one_way.sf(1.0)             # -> 0.98430   (0.98509 with links both ways)
len(one_way.path_sets())    # -> 5
one_way.to_rbd().sf(1.0)    # -> 0.98430   the same graph as a block diagram
```

A directed network is decided from its simple paths (up to 100,000), and
every method works on it; it may have cycles. `to_rbd()` gives the
`NonRepairableRBD` of the same graph, each link and failing node a node of
the diagram and the other nodes junctions, for a network whose links from
the source to the target have no cycle (a diagram has none): its methods
then take a mesh with more paths than can be listed. A directed network
takes two terminals: whether several are joined along one-way links
depends on the order they are reached in, which the decision diagram's
states do not say, so more than two are refused.

## How it is computed

The exact values come from a binary decision diagram built from the network
itself (after Hardy, Lucet & Limnios, 2007). The links are decided one at a
time; after each, what is left to decide depends only on how the nodes with
links still to come are joined up by the working links so far, and which of
those groups hold a terminal. All the states before a decision are
worked out together, and equal ones merged, so the diagram grows with the
network's width rather than with its number of paths, which multiply in a
mesh. A 6-by-6 grid of cables, corner to corner, has over a million simple
paths, and is exact in a few hundredths of a second:

```python
cable = surv.Exponential.from_params([0.05])
mesh_links = {}
for i in range(6):
    for j in range(6):
        if j + 1 < 6:
            mesh_links[f"h{i}{j}"] = ((i, j), (i, j + 1), cable)
        if i + 1 < 6:
            mesh_links[f"v{i}{j}"] = ((i, j), (i + 1, j), cable)
mesh = Network(mesh_links, source=(0, 0), target=(5, 5))
mesh.sf(1.0)     # -> 0.994761
mesh.ff(1.0)     # -> 0.0052394
mesh.mean()      # -> 9.1502   years
```

A node that can fail is decided as its first link is, and a failed one takes
its links out. `ff` is worked out in its own right, so a small one keeps its
precision, `birnbaum_importance` gives every element's at once (as the
derivative of the reliability, from whichever of it and the unreliability
is the smaller), and `mean` integrates the exact reliability. The width
limits it: a 10-by-10 grid has 1.9 million states before its decisions,
and is exact in a second and a half, but a diagram of more than five
million (`repyability.network.MAX_STATES`), such as an 11-by-11 grid's 7.7
million, refuses the exact values, and says to simulate. Raise the limit
to try harder: the 11-by-11 grid takes eight seconds. `path_sets()`
lists the simple paths, up to 100,000 of them. Setting
`repyability.network.METHOD = "paths"` decides the network from those
paths instead, as the exact engine decides a diagram's core from its
minimal path sets: slower beyond the smallest networks.

`method="simulate"` (for `sf`, `ff` and `mean`) draws each link's and
node's lifetime, `mc_samples` times (by default 10,000), seeded with
`seed`; `random(size)` gives the lifetimes themselves. In each sample the
connection lasts as long as its longest-lasting path, found by adding links
in order of their lives, longest first, until the terminals are joined.

```python
grid.sf(1.0, method="simulate", mc_samples=200_000, seed=1)   # -> 0.98477   simulated
```

Links fail independently, as a diagram's nodes do. Given lifetime models,
a link that fails stays failed; given a life and a repair, it comes back
(next).

## Repaired links

A line that fails is repaired, and the question becomes how much of the
time the load is connected and how often it is cut off. Give each link,
and each node that can fail, a component spec as a
[`RepairableRBD`](repairable.md) takes one, its `"reliability"` and its
`"repairability"`, every element new at 0:

```python
repair = surv.LogNormal.from_params([-5.0, 0.8])        # 81 hours on average, in years
long_repair = surv.LogNormal.from_params([-4.0, 0.8])   # 221 hours
line = {"reliability": surv.Exponential.from_params([0.1]), "repairability": repair}
cable = {"reliability": surv.Exponential.from_params([0.05]), "repairability": long_repair}
supply = Network(
    {name: (u, v, cable if name in ("AB", "CL") else line)
     for name, (u, v, _) in links.items()},
    source="G", target="L",
)
supply.point_availability(1.0)       # -> 0.99999797   connected at a year
supply.point_unavailability(1.0)     # -> 2.0289e-06
supply.mission_unavailability(1.0)   # -> 1.9862e-06   the share of the first year cut off
supply.mean_unavailability()         # -> 2.0289e-06   in the long run: 1.07 minutes a year
supply.failure_frequency(1.0)        # -> 0.00035790   losses of the connection per year, then
supply.expected_failures(10.0)       # -> 0.0035738    in the first ten years
supply.system_failure_frequency()    # -> 0.00035790   in the long run: one every 2,794 years
```

- **How.** Each element's probability of being down at a time is a
  diagram's component's (in closed form for an exponential life and
  repair, else from its renewal curve, to about `1e-7`), and the elements
  fail and are repaired independently, so the connection's follows from
  the same decision diagram at those probabilities, exactly. The
  unavailability is worked out as itself, a sum of products, so a small
  one keeps its digits, and the availability is one less.
- **How often.** A link's failure cuts the load off when the link is
  critical then, so the frequency is each link's failure intensity times
  its Birnbaum importance, summed; `expected_failures(t)` is its integral
  and `system_failure_frequency()` its long-run value.
- **Any connection.** Several terminals, all of them and directed links
  work the same way.
- **Importance.** The measures made of the elements' probabilities
  (`birnbaum_importance(x)`, `criticality_importance(x)`,
  `fussell_vesely(x)`, the risk worths, `joint_importance`,
  `differential_importance`, `importances(x)`) are taken at the elements'
  availabilities at `x`.
- **Not yet.** A repaired network has no time at which its terminals part
  for good, so `sf`, `mean`, the B-lives and `random` refuse it; so do the
  levers, `parameter_sensitivity`, `tornado` and `to_rbd`. An element
  takes a life and a repair only (no tests, maintenance, crews or costs),
  every element is repaired or none is, and a repaired network is not
  simulated. A network that can be drawn as a block diagram has all of
  those as a `RepairableRBD`.

## Importance, lives, uncertainty and saving

The importance measures of a block diagram (see [Importance
measures](importance.md)) are a network's too, over its links and failing
nodes, with the same definitions: `improvement_potential`,
`risk_achievement_worth`, `risk_reduction_worth`, `criticality_importance`
(`kind="failure"` or `"success"`), `fussell_vesely` (over the cut sets or
the path sets, exact or the rare-event sum), `structural_importance`,
`differential_importance`, `joint_importance` and
`barlow_proschan_importance`. Each comes from the decision diagram replayed
with an element held working or failed (the exact Fussell-Vesely from a
decision diagram of each element's cut sets), so they are exact, and a
small one keeps its precision. The lines out of the generator account for
most of the outages:

```python
grid.criticality_importance(1.0)["GA"]        # -> 0.60334   its share of the outages at a year
grid.fussell_vesely(1.0)["GA"]                # -> 0.60991
grid.risk_achievement_worth(1.0)["GB"]        # -> 7.0029    the risk with GB down, relative
grid.barlow_proschan_importance()["GB"]       # -> 0.26291   the outages it causes, over the whole life
```

`Hf`, `df` and `hf` give the connection's cumulative hazard, failure
density and hazard rate, as a diagram's do: `Hf` is exact, `-log` of the
reliability (worked out from the unreliability while that is the smaller, so
a small one keeps its precision), and `df` and `hf` are central differences
of the exact values, with a step relative to the time, so that they do not
depend on the unit of time.

```python
grid.Hf(1.0)    # -> 0.015023
grid.df(1.0)    # -> 0.029521
grid.hf(1.0)    # -> 0.029968
```

`parameter_sensitivity(x)` gives how far the reliability at `x` moves with
each parameter of each link's and node's model: its Birnbaum importance
times how far the element's own reliability moves with the parameter, as for
a diagram (see [Sensitivities: the Greeks](greeks.md)). `levers()` lists the
parameters it moves, each with its value and range, and
`with_levers({("GA", "failure_rate"): 0.04})` builds the network with them
moved, so that the slope can be checked, or a design tried.

```python
grid.parameter_sensitivity(1.0)["GA"]["failure_rate"]   # -> -0.085537
grid.with_levers({("GA", "failure_rate"): 0.04}).sf(1.0)   # -> 0.99038
```

`tornado(x)` puts the parameters on one footing, as a tornado chart does:
the reliability at `x` with each parameter a tenth lower and a tenth
higher (or over a range given in `ranges`), the others at their values,
sorted by the swing.

```python
spread = grid.tornado(1.0)
spread.base_value                       # -> 0.98509
row = spread.rows[0]                    # the largest swing
row.key, row.value_low, row.value_high  # -> "GB", 0.98599, 0.98420
```

`bx_life(x)` and `time_to_reliability(target)` invert `sf`: the time by
which `x` percent of connections are cut, and at which the reliability
falls to `target`.

```python
grid.bx_life(10)                 # -> 2.6804   years until a tenth of the loads have been cut off
grid.time_to_reliability(0.5)    # -> 7.4167
```

`sf_uncertainty` and `mean_uncertainty` propagate the uncertainty of the
links' and nodes' fitted models, as a diagram's do (see [Parameter
uncertainty](reliability.md#uncertainty-in-the-component-models)): each draw gives every
uncertain element a plausible model (`"fit"` from its surpyval fit's
covariance, distributions over its parameters, or a list of models), and
the reliability or the mean is worked out exactly for it. Elements given
together in a tuple share their draws, as lines of one type should. With
the five lines' failure rate known only to lie between 0.05 and 0.15 a
year:

```python
import scipy.stats as st

rate = {"failure_rate": st.uniform(0.05, 0.1)}
lines = ("GA", "GB", "AC", "BC", "BL")
result = grid.sf_uncertainty(1.0, {lines: rate}, n_draws=4000, seed=1)
result.nominal                       # -> 0.98509
lower, upper = result.interval(0.9)
round(lower, 4), round(upper, 4)     # (0.9721, 0.9943)
mean = grid.mean_uncertainty({lines: rate}, n_draws=400, seed=1)
round(mean.nominal, 4)               # -> 8.518
```

`bx_life_uncertainty` and `time_to_reliability_uncertainty` give a B-life
with its band the same way, each draw's time found by inverting its exact
reliability: `grid.bx_life_uncertainty(10, {lines: rate}, n_draws=400,
seed=1)`.

A network saves and loads as a diagram does (see [Saving and
loading](saving.md#saving-and-loading)): `to_dict` and `to_json` keep the
terminals, each link's two nodes and model, and the failing nodes' models,
in surpyval's own format, so a fit's covariance round-trips; `from_dict`
and `from_json` rebuild it. `analysis_routes()` says how each analysis is
worked out, without running it: exact from the decision diagram, numerical
(`mean`, `df`, `hf`, `parameter_sensitivity`, the B-lives and
`barlow_proschan_importance`), simulated (`random` and the uncertainty's
draws), or refused, with the message the
method raises, for a network too large for its diagram.

```python
Network.from_json(grid.to_json()).sf(1.0)   # -> 0.98509
grid.analysis_routes()["mean"].route        # 'numerical'
```
