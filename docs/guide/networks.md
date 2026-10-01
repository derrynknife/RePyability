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
  `{"BL", "CL"}` and `{"GA", "GB"}`.
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

The exact values come from the minimal paths: each simple path between the
terminals is a minimal path set (two simple paths with the same links are
the same path), and the probability that one of them works is the Shannon
decomposition the exact engine uses for a diagram's core. `ff` is worked
out in its own right, so a small one keeps its precision, and `mean`
integrates the exact reliability. A network with more than 100,000 paths
between its terminals refuses the exact values, and says to simulate.

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
