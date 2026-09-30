# Simulation precision and speed

!!! tip "Learning this for the first time?"
    This page is the reference. Why a simulation has an error, and how
    these options shrink it, is taught in [Lesson
    6](../learn/availability.md#precise-enough-sooner).

Some quantities are simulated: a `NonRepairableRBD`'s lifetimes, and a
`RepairableRBD`'s histories over a window, with their costs. (A
`NonRepairableRBD`'s mean time to failure is exact, and can be simulated
too; a `RepairableRBD`'s availability over time is also exact, with
`point_availability`: see
[what is exact and what is simulated](saving.md#what-is-exact-and-what-is-simulated).)
A simulated estimate has a sampling error, which shrinks like `1/√N`:
halving it takes four times as many simulations. This page covers how a
repairable system's simulations draw their random numbers, and the options
that choose `mc_samples` for you, that get the same precision from fewer
simulations, that spread the simulations over several CPUs, that compare
two designs far more precisely than two separate runs can, and that run the
simulations compiled.

| Option | Where | What it does |
|---|---|---|
| `tolerance`, `confidence`, `max_samples` | `availability`, `cost`; `mean` and `mean_time_to_failure` with `method="simulate"`, `mean_time_to_failure_interval` | Simulate until the estimate is known to within `tolerance`. |
| `antithetic=True` | the same, and `NonRepairableRBD.random` | Simulate in antithetic pairs. |
| `n_jobs` | the same, and `RepairableRBD.compare` | Run the simulations on several CPUs. |
| `compare(other, ...)` | `RepairableRBD`, `NonRepairableRBD` | The difference between two designs, simulated with common random numbers. |
| `engine` | `RepairableRBD`'s `availability`, `cost` and `compare` | Run the simulations compiled, with numba. |

The examples use two units in parallel and the plant of
[Repairable systems](repairable.md):

```python
import numpy as np
import surpyval as surv
from repyability import NonRepairableRBD, RepairableRBD

unit = surv.Weibull.from_params([100, 2])

def parallel(n):
    """n units in parallel."""
    names = [f"u{i}" for i in range(n)]
    edges = [("s", u) for u in names] + [(u, "t") for u in names]
    return NonRepairableRBD(edges, {u: unit for u in names})

pair = parallel(2)

def repairable(failure_rate, repair_rate):
    return {
        "reliability": surv.Exponential.from_params([failure_rate]),
        "repairability": surv.Exponential.from_params([repair_rate]),
    }

#   s -> (A | B) -> C -> t        MTTF 10 / MTTR 1 pumps, MTTF 50 / MTTR 2 valve
edges = [("s", "A"), ("s", "B"), ("A", "C"), ("B", "C"), ("C", "t")]
plant = RepairableRBD(
    edges,
    {"A": repairable(0.1, 1.0), "B": repairable(0.1, 1.0), "C": repairable(0.02, 0.5)},
)
```

## Random streams

A `RepairableRBD` simulation draws each random quantity from a stream of its
own: each component's times to failure, its repair times, its maintenance or
test times, and each cost given as a distribution. A stream is named by the
component's place (its node name, and the names down through nested RBDs)
and the quantity, and seeded from the run's `seed`, and simulation `r` takes
the same numbers from it whatever the rest of the run does. So:

- a simulation does not depend on how many the run has, on the order they
  run in, on how many processes or threads run them, or on which engine: the
  first 100 simulations of a run of 1 000 are a run of 100;
- one component's draws never depend on another's: a component modelled
  the same way fails and is repaired at the same times whatever the rest of
  the system is, which is what [`compare`](#comparing-two-designs) builds
  on;
- without a `seed`, the run takes one number from numpy's global RNG as its
  seed, so `np.random.seed(s)` beforehand makes it the run `seed=s` gives;
  the global RNG is otherwise left as it was.

```python
short = plant.availability(t_simulation=100.0, mc_samples=100, seed=0)
long = plant.availability(t_simulation=100.0, mc_samples=1_000, seed=0)
bool((short.uptimes == long.uptimes[:100]).all())   # True
```

A stream's numbers are laid out in blocks of simulations, fewer to a block
the more draws a simulation is expected to take from the stream, so which
numbers a simulation gets depends on the component's models and on the
window too: a run over another window, or a component in another place,
draws other numbers. A
model whose draws cannot be streamed (other than a surpyval parametric one,
or a subclass of the component classes, which may draw its events its own
way) draws from numpy's global RNG, seeded afresh for each simulation from
the run's seed: the run stays reproducible, but that component cannot take
part in antithetic pairs or common random numbers.

## Simulating to a tolerance

`tolerance` says how precise the estimate must be: within `tolerance` either
side, at `confidence` (0.95 by default). After the first `mc_samples`
lifetimes, and after each further `mc_samples`, the run checks the
half-width of the confidence interval of the mean, and stops once it is at
most `tolerance`:

```python
ci = pair.mean_time_to_failure_interval(mc_samples=10_000, seed=1, tolerance=0.5)
ci.n_samples               # -> 30000
ci.upper - ci.estimate     # -> 0.49     at most the tolerance
ci.estimate                # -> 114.56   the exact MTTF is 114.58
```

With `method="simulate"`, `mean` and `mean_time_to_failure` take the same
arguments and return the estimate alone (by default they are exact). For a `RepairableRBD`, `availability(tolerance=...)` judges
the mean availability over the window, the fraction of it the system is up
(`result.mean_availability_interval()`), and `cost(tolerance=...)` the mean
cost of a window (`result.mean_interval()`), checking after every `mc_samples`
simulations:

```python
result = plant.availability(t_simulation=100.0, mc_samples=1_000, seed=0, tolerance=0.001)
result.n_simulations                  # -> 6000
window = result.mean_availability_interval()
window.estimate                       # -> 0.9547   the exact value is 0.9544
window.upper - window.estimate        # -> 0.00094
```

If the tolerance is not reached within `max_samples` simulations (or
lifetimes), by default 100 times `mc_samples`, the run stops there with a
`RuntimeWarning`, and the result is from those. A run that stops after `n`
simulations is the run of `N = n` from the start: the further simulations
continue the same random numbers. (`mean_availability_interval()` is the
interval of the mean over the window; the pointwise band
`availability_interval()` is the curve's, at each time: see
[Repairable systems](repairable.md#availability-over-time-simulated).)

## Antithetic pairs

With `antithetic=True` the simulations come in pairs, and in the second of
each pair every component draws `1 − u` for each uniform random number `u`
it drew in the first, in the same order: where the first drew an early
failure (or a short repair), the second draws a correspondingly late one.
Each simulation is still a correct one, so the estimate is unbiased, but the
two of a pair tend to err in opposite directions, so their mean varies less
than two independent simulations':

```python
plain = pair.mean_time_to_failure_interval(mc_samples=10_000, seed=1)
paired = pair.mean_time_to_failure_interval(mc_samples=10_000, seed=1, antithetic=True)
plain.standard_error      # -> 0.43
paired.standard_error     # -> 0.32   worth 1.8 times as many independent lifetimes
```

A coherent system's lifetime rises with every component's lifetime, which is
what makes the pairing work so well for it. A window's availability and cost
depend on the draws in a less simple way, and gain less, but still gain:

```python
single = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0)
pairs = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0, antithetic=True)
single.mean_availability_interval().standard_error   # -> 0.00082
pairs.mean_availability_interval().standard_error    # -> 0.00063
```

The pairs, not the simulations, are independent, so the intervals are worked
out from the pairs' means (`result.antithetic` records it), and
`mc_samples` must be even. `NonRepairableRBD.random(size,
antithetic=True)` returns the pairs themselves: lifetimes `2i` and `2i + 1`.
In a `RepairableRBD` every [stream](#random-streams) is paired, so an
antithetic run of `mc_samples` simulations is the first half of the run without
pairs, each followed by its mirror image.

Every component's draws must be replayable from uniform random numbers:
surpyval parametric distributions, and the composite nodes built from them.
Otherwise `antithetic=True` raises `NotImplementedError`.

## Parallel runs

`n_jobs=k` runs the simulations on `k` CPUs (`-1`: all of them). A
`RepairableRBD`'s simulations each draw from their own
[streams](#random-streams), so the result is the same to the last bit for
any `n_jobs`, and without it:

```python
fast = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0, n_jobs=2)
same = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0)
bool((fast.uptimes == same.uptimes).all())   # True
```

The [compiled engine](#the-compiled-engine) runs them on `k` threads. The
Python engine sends blocks of 250 of them to `k` processes, each with a copy
of the diagram, and adds up their results in order, so it pays off for long
simulations (many failures and repairs in a window).

A `NonRepairableRBD`'s lifetimes are split over the processes in blocks of
10 000, seeded in turn from `seed`, so the result depends on the seed and
not on the number of processes, but it is not the result of a run without
`n_jobs`, which draws every lifetime from one stream. It pays off for
lifetimes that must be simulated event by event, not for a few thousand
that are drawn in one vectorised step. Both work with `tolerance` (checked
after each batch of `mc_samples`) and `antithetic`.

A process needs RePyability, surpyval and scipy loaded before it can
simulate, and how long that takes depends on how the platform starts
processes:

- **Linux, Python up to 3.13 (fork):** processes start with everything
  already loaded.
- **Linux, Python 3.14 (forkserver):** RePyability has the server load them
  once. The first parallel run waits for that; later runs start their
  processes at once.
- **macOS and Windows (spawn):** every process loads them itself, which
  takes a second or two, so `n_jobs` pays off only for runs longer than
  that.

`n_jobs=-1` starts one process per CPU this process may run on (its CPU
affinity, where the platform reports one). A container limited by a CPU
quota rather than by affinity can report more CPUs than it may use: set
`n_jobs` explicitly there.

## Comparing two designs

Two designs are best compared with **common random numbers**: simulate both
with the same random numbers, so that the differences between their results
come from the designs rather than from chance. `a.compare(b, ...)` does that
component by component: a component with the same name in both draws the
same random numbers in both. It returns a
[`ConfidenceInterval`][repyability.ConfidenceInterval] of the mean
difference, `a`'s result minus `b`'s, which can be negative.

For a `RepairableRBD`, a component gets the same failures and repairs in
both where it is modelled the same way, and matching ones (the same
quantiles of its own models) where it is not; components of nested RBDs are
matched by their place. The difference is in the fraction of the window the
system is up (`quantity="availability"`, the default) or in its cost
(`quantity="cost"`):

```python
faster = RepairableRBD(
    edges,
    {"A": repairable(0.1, 2.0), "B": repairable(0.1, 2.0), "C": repairable(0.02, 0.5)},
)
gain = faster.compare(plant, t_simulation=100.0, mc_samples=2_000, seed=0)
gain.estimate          # -> 0.0058    the exact difference is 0.00568
gain.standard_error    # -> 0.00017
```

Two separate runs of 2 000 simulations would estimate the difference with a
standard error of about 0.0012, seven times as large: it would take fifty
times the simulations to match. The valve fails and is repaired alike in
both designs, so its outages, most of the plant's downtime, cancel; the
pumps have the same up times in both, and only their repairs differ.

For a `NonRepairableRBD`, `compare` estimates the difference in mean time to
failure:

```python
gain = parallel(3).compare(pair, mc_samples=20_000, seed=0)
gain.estimate          # -> 14.8    a third unit adds about 15 hours
gain.standard_error    # -> 0.21    two separate estimates: about 0.42
```

The exact difference, the integral of the difference in reliability, is
14.46. The reliabilities themselves need no simulation: compare `sf`.

Both kinds need every component's draws to be replayable, as antithetic
pairs do. A non-parametric model draws its own random numbers, which the two
systems would not share, so it raises `NotImplementedError`.

## The compiled engine

A `RepairableRBD`'s simulations run in Python. With
[numba](https://numba.pydata.org), an optional dependency, installed, they
can run compiled instead, about ten times as fast on one core, and faster
still on several:

```bash
pip install "repyability[fast]"
```

`engine="auto"`, the default of `availability`, `cost` and `compare`, then
compiles a run when the compiled engine simulates the system and the run is
long enough to repay loading it: about a third of a second from numba's
cache (some seconds the first time ever, while numba compiles it).
`engine="numba"` asks for it outright, and raises an error if numba is not
installed or the system is one it does not simulate; `engine="python"` keeps
to Python. The two engines give the same results, to the last bit: the
compiled loop is the Python one over arrays, reading the same
[streams](#random-streams).

```python
fast = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0)
slow = plant.availability(t_simulation=100.0, mc_samples=2_000, seed=0, engine="python")
bool((fast.uptimes == slow.uptimes).all())   # True, with numba or without
```

The compiled engine simulates plain components (`reliability` and
`repairability` specs, or `NonRepairable` objects, with surpyval parametric
models) in any structure, with nodes held working or broken, costs,
antithetic pairs, tolerances and common random numbers. Preventive
maintenance, inspections, repair crews, nested RBDs, capacities and other
models run in Python, which `"auto"` chooses by itself. With `n_jobs` it runs on that many
threads, which start at once.

On one test machine, the plant above, over 1 000 hours, ran at about a
million events (failures and repairs) a second in Python, 12 million
compiled on one core and 24 million on four; a system of 12 components with
a bridge and a vote, over 5 000 hours, at 0.8, 8.5 and 23 million; and 70
components, 35 redundant pairs in series, over 2 000 hours, at 0.6, 4.4 and
12.6 million. Above 20 components, the compiled loop works out whether the
system is up after each event that could change it, rather than looking it
up in a table of every state, so those events cost more.

## Which to use

- **A precision to meet:** give a `tolerance`, rather than guessing `mc_samples`.
- **A choice between designs:** `compare` them, rather than comparing two
  separate estimates, whose errors add up.
- **A quantity that rises with the components' lifetimes:** try
  `antithetic=True`, and check that the standard error falls.
- **Long simulations:** install numba (`pip install "repyability[fast]"`)
  for the compiled engine, and spread them over the cores with `n_jobs`.
