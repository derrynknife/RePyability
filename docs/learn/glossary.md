# Glossary

Short definitions of the terms used in the course, each with the lesson that
teaches it. Symbols follow the [notation table](index.md#notation).

**Acquisition cost.** The one-off price of buying a unit, as opposed to the
running costs it incurs while owned. [Lesson 7](costs.md)

**Aleatory uncertainty.** The variability a model describes: which units
fail, and when. More units do not reduce it. Compare epistemic uncertainty.
[Lesson 2](systems.md)

**Active redundancy.** Redundant units that all run at the same time, so
each ages and can fail while the others work. A parallel block in a diagram.
[Lesson 2](systems.md)

**Age replacement.** Replace a unit when it fails or when it reaches a set
age $T$, whichever comes first; a failure restarts the clock.
[Lesson 8](maintenance.md)

**Availability.** The probability that a repairable system is working at a
given time (*point availability* $A(t)$), or the long-run fraction of time
it is working (*long-run* or *steady-state availability* $A$). For one unit,
$A = \text{MTTF} / (\text{MTTF} + \text{MTTR})$. [Lesson 6](availability.md)

**B-life.** The age by which a given fraction of units has failed: $B_{10}$
is the age at which $F(t) = 0.1$. [Lesson 1](lifetimes.md)

**Basic event.** A leaf of a fault tree: a component failure, with a
probability or a lifetime model. [Lesson 3](structure.md)

**Bathtub curve.** A hazard rate that falls early in life (early failures),
stays flat (random failures) and then rises (wear-out).
[Lesson 1](lifetimes.md)

**Beta factor.** In the beta-factor model of common-cause failure, the
fraction $\beta$ of a unit's failure probability that fails all the
redundant units together. [Lesson 5](dependence.md)

**Birnbaum importance.** The probability that a component is critical,
$I_B(i) = R(1_i) - R(0_i)$; equivalently $\partial R / \partial p_i$, the
gain in system reliability per unit gain in the component's.
[Lesson 4](importance.md)

**Block replacement.** Replace units at fixed calendar times
$T, 2T, 3T, \ldots$ whatever their age, and at failures in between.
[Lesson 8](maintenance.md)

**Censored observation.** A unit still working when observation stopped:
its life is known to be longer than its age, but not by how much. Lifetime
fits must use such data; fitting is done in
[surpyval](https://github.com/derrynknife/SurPyval). [Lesson 1](lifetimes.md)

**Characteristic life.** The Weibull scale $\alpha$: the age by which 63.2%
of units have failed, whatever the shape. [Lesson 1](lifetimes.md)

**Cold standby.** A spare that does not run, and so does not age, until it is
switched in to replace a failed unit. *Warm standby* ages at a reduced rate
while waiting. [Lesson 5](dependence.md)

**Common-cause failure.** One cause (a shared design flaw, environment or
maintenance error) failing several redundant units at once, so that they do
not fail independently. [Lesson 5](dependence.md)

**Cost rate.** The long-run cost per unit time of running a system or a
maintenance policy. [Lessons 7](costs.md) and [8](maintenance.md)

**Critical component.** A component whose state decides the system's: with
the other components as they are, the system works if the component works
and fails if it fails. [Lesson 4](importance.md)

**Criticality importance.** The *failure-oriented* form,
$I_B(i)(1 - p_i)/(1 - R)$, is the probability that component $i$ has failed
and is critical, given that the system has failed: the share of system
failures it is decisive in. The *success-oriented* form, $I_B(i)\,p_i/R$, is
1 for every component in series with the rest. [Lesson 4](importance.md)

**Cumulative hazard.** $H(t) = \int_0^t h(u)\,du$; reliability is
$R(t) = e^{-H(t)}$. [Lesson 1](lifetimes.md)

**Cut set.** A set of components whose failure fails the system. A *minimal*
cut set has no smaller cut set inside it; a minimal cut set of one component
is a single point of failure. [Lesson 3](structure.md)

**Density.** $f(t) = dF/dt$: the fraction of the original population that
fails per unit time around age $t$. [Lesson 1](lifetimes.md)

**Epistemic uncertainty.** Not knowing a model exactly, because its
parameters are estimated from limited data; more data reduces it.
[Lesson 2](systems.md)

**Exponential distribution.** A lifetime with a constant hazard $\lambda$:
$R(t) = e^{-\lambda t}$, MTTF $= 1/\lambda$. It has no memory.
[Lesson 1](lifetimes.md)

**Failure frequency.** The long-run number of failures per unit time of a
repairable unit, $1/(\text{MTTF} + \text{MTTR})$, or of a system,
$\sum_i I_B(i)\,\omega_i$. [Lesson 6](availability.md)

**Fault tree.** A diagram of how a system fails: a top event, worked down
through OR, AND and VOTE gates to basic events. The dual of a reliability
block diagram. [Lesson 3](structure.md)

**Fussell–Vesely importance.** The share of the system's unreliability that
comes through minimal cut sets containing a component.
[Lesson 4](importance.md)

**Hazard rate.** $h(t) = f(t)/R(t)$: the rate at which units that have
survived to age $t$ fail, per unit time. Also called the failure rate.
[Lesson 1](lifetimes.md)

**Hidden failure.** A failure that nobody notices when it happens, such as
a seized relief valve or a standby pump that will not start: the part is
down until a proof test finds it. Also called an unrevealed or latent
failure. [Lesson 8](maintenance.md)

**Improvement potential.** $R(1_i) - R$: the gain in system reliability if
component $i$ were made perfect. [Lesson 4](importance.md)

**Input and output nodes.** The start and end of a reliability block diagram;
the system works when working components connect them.
[Lesson 2](systems.md)

**$k$-out-of-$n$.** A block that works when at least $k$ of its $n$ units
work. 1-out-of-$n$ is parallel and $n$-out-of-$n$ is series.
[Lesson 2](systems.md)

**Load sharing.** Redundant units that share a load, so that when one fails
the survivors carry more of it and are more likely to fail.
[Lesson 5](dependence.md)

**Mean down time (MDT).** The mean length of a system outage:
$(1 - A)$ divided by the rate of outages. [Lesson 6](availability.md)

**Mean time between failures (MTBF).** For a repairable system, the mean
time from one system failure to the next: one over the system failure
frequency. Not the same as a component's MTTF. [Lesson 6](availability.md)

**Mean time to failure (MTTF).** The mean life, $\int_0^\infty R(t)\,dt$:
the area under the reliability curve. Often half the units or more fail
before it. [Lesson 1](lifetimes.md)

**Mean time to repair (MTTR).** The mean time a failed unit is down while
it is repaired or replaced. [Lesson 6](availability.md)

**Mean up time (MUT).** The mean length of a period in which the system
works without interruption: $A$ divided by the rate of outages.
[Lesson 6](availability.md)

**Memoryless.** A lifetime whose remaining life does not depend on its age:
$\Pr(T > s + t \mid T > s) = \Pr(T > t)$. Only the exponential has this
property. [Lesson 1](lifetimes.md)

**Minimal repair.** A repair that restores a unit to working order but not
to new: the unit is "as bad as old". [Lesson 8](maintenance.md)

**Monte-Carlo simulation.** Estimating a quantity by simulating many random
histories and averaging them. Its answers carry sampling error, which
shrinks like $1/\sqrt{N}$. [Lessons 6](availability.md) and [7](costs.md)

**Parallel system.** A system that works while at least one of its units
works: $R = 1 - \prod_i (1 - R_i)$; the unreliabilities multiply.
[Lesson 2](systems.md)

**Path set.** A set of components whose working makes the system work. A
*minimal* path set has no smaller path set inside it.
[Lesson 3](structure.md)

**PFDavg.** The average probability of failure on demand of a protective
function: its long-run unavailability, when its failures are hidden until a
proof test. About $\lambda\tau/2$ for one channel tested every $\tau$.
[Lesson 8](maintenance.md)

**Pivotal decomposition.** Conditioning on one component:
$R = p_i\,R(1_i) + (1 - p_i)\,R(0_i)$. Applied repeatedly, it computes any
system's reliability exactly; it is the basis of RePyability's engine (also
called the Shannon expansion). [Lesson 3](structure.md)

**Planned outage.** Downtime for preventive maintenance or a proof test. It
counts as downtime in availability, but not as a failure.
[Lesson 8](maintenance.md)

**Preventive maintenance.** Maintenance done before a failure, on a schedule
(such as age or block replacement), to prevent failures that would cost
more. It only pays for parts that wear out. [Lesson 8](maintenance.md)

**Proof test.** A periodic inspection that finds a part's hidden failures;
the part is repaired if it is found failed. [Lesson 8](maintenance.md)

**Rare-event approximation.** Approximating a system's unreliability by the
sum, over the minimal cut sets, of the probability that all their components
have failed. Accurate when failures are rare. [Lesson 3](structure.md)

**Reduction.** Replacing a group of blocks purely in series, purely in
parallel, or $k$-out-of-$n$, by one equivalent block, and repeating. A
series-parallel system reduces to a single block; RePyability reduces every
diagram as far as it goes before applying the pivotal decomposition to the
rest. [Lessons 2](systems.md) and [3](structure.md)

**Redundancy allocation.** Choosing how many redundant copies of each
component to fit, to maximise reliability within a budget or to meet a
target at least cost; for a repairable system, to own it at the lowest
total cost. [Lesson 9](design.md), [Lesson 7](costs.md)

**Reliability.** $R(t) = \Pr(T > t)$: the probability that a unit (or a
system) is still working at age $t$. Also called the survival function.
[Lesson 1](lifetimes.md)

**Reliability allocation.** Apportioning a system's reliability target among
its components: how reliable must each part be? [Lesson 9](design.md)

**Reliability block diagram (RBD).** A diagram of a system's success logic:
blocks between an input and an output, such that the system works when some
path of working blocks connects them. It shows logic, not wiring.
[Lesson 2](systems.md)

**Renewal-reward theorem.** For a process that renews in independent,
identical cycles, the long-run cost (or reward) per unit time is the
expected cost of a cycle divided by the expected length of a cycle.
[Lessons 6](availability.md), [7](costs.md) and [8](maintenance.md)

**Repeated event.** A basic event that feeds several gates of a fault
tree, such as a shared power supply; it must be counted once, not as
independent copies. [Lesson 3](structure.md)

**Risk achievement worth (RAW).** $Q(0_i)/Q$: how many times more likely the
system is to fail while component $i$ is failed or out of service.
[Lesson 4](importance.md)

**Risk reduction worth (RRW).** $Q/Q(1_i)$: the factor by which a perfect
component $i$ would divide the system's unreliability.
[Lesson 4](importance.md)

**Series system.** A system that works only while every unit works:
$R = \prod_i R_i$. [Lesson 2](systems.md)

**Single point of failure.** A component whose failure alone fails the
system: a minimal cut set of one. [Lessons 3](structure.md) and
[4](importance.md)

**Structural importance.** Birnbaum importance with every component working
with probability ½: the fraction of the other components' states in which a
component is critical. It depends on the diagram alone.
[Lesson 4](importance.md)

**Structure function.** $\varphi(x)$: 1 if the system works when its
components are in states $x$, 0 otherwise. [Lesson 3](structure.md)

**Top event.** The undesired event at the top of a fault tree, usually
the system failing. [Lesson 3](structure.md)

**Total cost of ownership.** What owning a system costs over a horizon $H$:
buying it plus $H$ times its cost rate (a *life-cycle cost*, here
undiscounted). [Lesson 7](costs.md)

**Unreliability.** $F(t) = 1 - R(t)$: the probability that a unit has failed
by age $t$. [Lesson 1](lifetimes.md)

**Weibull distribution.** A lifetime with $R(t) = \exp[-(t/\alpha)^\beta]$.
Its shape $\beta$ gives a falling ($\beta < 1$), constant ($\beta = 1$) or
rising ($\beta > 1$) hazard. [Lesson 1](lifetimes.md)
