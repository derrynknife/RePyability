# Demonstration testing

A demonstration test shows, at a confidence level, that a design meets a
reliability target. You test `n` units and allow up to `failures` of them to
fail; if no more fail, the design has demonstrated the target. These
functions plan such a test from the target, and say what a finished test
demonstrated. They fit nothing: surpyval fits the data a test produces.

## How many units?

A unit passes if it survives the mission time. With no failures allowed,
`n` units demonstrate a reliability `R` at confidence `C` when
`R ** n ≤ 1 − C`: were the reliability only `R`, so many successes in a row
would be that unlikely. This is the success run:

```python
from repyability import demonstrated_reliability, demonstration_sample_size

demonstration_sample_size(0.95, confidence=0.95)              # -> 59
demonstration_sample_size(0.95, confidence=0.95, failures=1)  # -> 93
demonstrated_reliability(59, confidence=0.95)                 # -> 0.9505
```

Allowing failures keeps one unlucky unit from failing a good design, at the
cost of more units. With `failures` allowed, the plan is the smallest `n` for
which that many failures or fewer have a chance of at most `1 − C` at the
target (the binomial distribution). `demonstrated_reliability` gives the
reliability a finished test demonstrated: the Clopper–Pearson lower bound,
which with no failures is surpyval's `success_run`.

## Testing longer instead: Weibayes

When the lifetime's Weibull shape `β` is known, from past data or the
failure mode, each unit can run for `k` missions instead of one. A unit that
survives `k` missions survives one with reliability `R` when its reliability
over the test is `R ** (k ** β)`, so fewer units are needed:

```python
from repyability import demonstration_test_multiple

demonstration_sample_size(0.95, 0.95, test_multiple=2.0, shape=2.0)   # -> 15
demonstration_test_multiple(0.95, 20, 0.95, shape=2.0)                # -> 1.709
```

`demonstration_test_multiple` answers the other way round: how many missions
each of a given number of units must run. For wear-out (`β > 1`) a longer
test pays off quickly; at `β = 1` test time and units trade one for one; for
early failures (`β < 1`) a longer test saves few units.

## Constant failure rate: test time for an MTBF

When the failure rate does not change with age, it is the total unit time
that counts, however it is spread over the units, and failed units can be
repaired or replaced and the test carried on. A test of total time `T` that
ends with at most `r` failures demonstrates an MTBF `θ` at confidence `C`
when `T ≥ θ χ²(C; 2r + 2) / 2`:

```python
from repyability import demonstrated_mtbf, mtbf_test_time

mtbf_test_time(1000.0, confidence=0.9)               # -> 2302.6
mtbf_test_time(1000.0, confidence=0.9, failures=2)   # -> 5322.3
demonstrated_mtbf(2302.6, confidence=0.9)            # -> 1000.0
```

## What a plan risks

A plan's operating characteristic is the chance that a design passes it,
given the design's true reliability. At the target it is at most `1 − C`:
the consumer's risk of accepting a design no better than the target. For a
good design, one minus it is the producer's risk of rejecting it:

```python
from repyability import demonstration_pass_probability, mtbf_pass_probability

demonstration_pass_probability(0.95, 59)               # -> 0.0485   at the target
demonstration_pass_probability(0.99, 59)               # -> 0.5527   a much better design
demonstration_pass_probability(0.99, 93, failures=1)   # -> 0.7616   the same, allowing a failure
mtbf_pass_probability(3000.0, 2302.6)                  # -> 0.4642   three times the MTBF
```

A test that allows no failures often fails a good design: the one above
would reject a design of 99% reliability 45% of the time. Allowing a
failure, with the more units that needs, cuts that to 24%.

## Plans that keep both risks

`demonstration_plan` and `mtbf_demonstration_plan` design the test from both
risks at once (#184): the smallest test that a design at the target passes
with a chance of at most `1 − confidence`, and that a good design, which
should pass, fails with a chance of at most `producer_risk`:

```python
from repyability import demonstration_plan, mtbf_demonstration_plan

plan = demonstration_plan(0.9, 0.95, confidence=0.9, producer_risk=0.2)
plan.n, plan.failures            # (128, 8)   128 units, at most 8 failing
plan.consumer_risk               # -> 0.097   a design at 90% passes
plan.producer_risk               # -> 0.192   one at 95% fails
demonstration_pass_probability(0.95, 22)   # -> 0.324   the success run of 22 units passes it a third of the time
mtbf = mtbf_demonstration_plan(1000.0, 2000.0, confidence=0.9, producer_risk=0.2)
mtbf.test_time, mtbf.failures    # about 14,206 unit hours, at most 9 failures
mtbf.test_time                   # -> 14206
```

For each number of failures allowed, from none, the fewest units (or the
least test time) that keep the consumer's risk pass a good design most
often; the first that keeps the producer's risk too is the smallest plan of
all. A good design close to the target needs many failures allowed, and
many units: the closer, the more. Given `n=` and the lifetime's `shape`,
`demonstration_plan` searches the test length instead, and with
`test_multiple` and `shape`, it plans an extended (Weibayes) test of that
length. The MTBF plans are MIL-HDBK-781's fixed-length plans, for a
discrimination ratio `good_mtbf / mtbf`; the handbook lets some of its plans
exceed a risk a little for a shorter test, where these keep both. The
result is a [`DemonstrationPlan`][repyability.DemonstrationPlan].
