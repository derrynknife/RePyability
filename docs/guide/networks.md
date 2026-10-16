# Networks

Power lines, pipes and communication links carry flow both ways, and it is
often the links that fail. The question is whether two points of the
network stay connected: some path of working links joins them. A
reliability block diagram is directed, and its nodes fail, so a network has
a model of its own, `Network`.

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

## How it is computed

The exact values come from a binary decision diagram built from the network
itself (after Hardy, Lucet & Limnios, 2007). The links are decided one at a
time; after each, what is left to decide depends only on how the nodes with
links still to come are joined up by the working links so far, and which of
those groups hold the terminals. All the states before a decision are
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

Links fail independently, as a diagram's nodes do. The network is not
repairable: a link that fails stays failed.

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

A network saves and loads as a diagram does (see [Saving and
loading](saving.md#saving-and-loading)): `to_dict` and `to_json` keep the
terminals, each link's two nodes and model, and the failing nodes' models,
in surpyval's own format, so a fit's covariance round-trips; `from_dict`
and `from_json` rebuild it. `analysis_routes()` says how each analysis is
worked out, without running it: exact from the decision diagram, numerical
(`mean`, the B-lives and `barlow_proschan_importance`), simulated
(`random` and the uncertainty's draws), or refused, with the message the
method raises, for a network too large for its diagram.

```python
Network.from_json(grid.to_json()).sf(1.0)   # -> 0.98509
grid.analysis_routes()["mean"].route        # 'numerical'
```
