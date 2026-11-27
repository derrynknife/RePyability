# Safety functions: PFDavg, PFH, spurious trips and the SIL

A safety instrumented function is a set of channels (transmitters, logic
solvers, valves) voted M-out-of-N: it acts on a demand when M of its N
channels do. A functional-safety study (IEC 61508, IEC 61511) asks four
numbers of it:

- the **PFDavg**, the average probability that it cannot act on a demand,
  for a function demanded rarely (low demand);
- the **PFH**, the average frequency of its dangerous failure, for one
  demanded often or all the time (high demand, continuous);
- the **spurious-trip rate**: how often it acts with no demand;
- the **SIL**, the band the PFDavg or the PFH falls in.

[`SafetyFunction`][repyability.SafetyFunction] gives all four from the
channels' failure rates, repairs and proof tests. It draws two
[`RepairableRBD`][repyability.RepairableRBD]s and takes each number from a
method they already have, so the values are exact (or numerical to a
stated accuracy), not the standards' first-order formulas. Those formulas
are there too, to set beside them.

## A channel

A channel's failures are split two ways. A **dangerous** failure stops the
channel acting; a **safe** one makes it act with no demand. A dangerous
failure is **detected** where diagnostics reveal it at once, and
**undetected** where only the proof test finds it.

```python
from repyability import SafetyChannel, SafetyFunction, sil_band

transmitter = SafetyChannel(
    dangerous_undetected=2e-6,   # per hour
    dangerous_detected=3e-6,
    safe=5e-6,
    repair=8.0,                  # mean time to restoration, hours
    proof_test_interval=8760.0,  # yearly
)
transmitter.safe_failure_fraction   # -> 0.8   (5 + 3) / 10
transmitter.diagnostic_coverage     # -> 0.6   3 / (2 + 3)
```

`SafetyChannel.from_rate(1e-5, dangerous_fraction=0.5,
diagnostic_coverage=0.6, ...)` makes the same split from a total rate. The
undetected failures can be a surpyval life model (a fitted Weibull, say)
in place of a rate; a proof test can miss (`proof_test_coverage`, with the
`full_test` interval that finds the rest); channels can be tested at
different times (`proof_test_offset`); and the repair of a failure the
proof test finds can differ from a revealed one's (`undetected_repair`,
the standard's MRT beside its MTTR). A repair given as a number is an
exponential time of that mean.

## One call

```python
function = SafetyFunction(
    {"a": transmitter, "b": transmitter, "c": transmitter},
    k=2,                  # 2oo3
    beta=0.1,             # common cause, undetected failures
    beta_detected=0.05,   # detected failures
    beta_safe=0.1,        # safe failures
)
result = function.results("low_demand")
result.pfd_avg                # -> 1.1227e-03
result.sil.sil                # -> 2
result.sil.risk_reduction_factor   # -> 890.7
result.sil.margin_to_lower    # -> 1.123    the PFDavg is 1.12 times SIL 3's edge
result.pfh                    # -> 5.6465e-07   per hour
result.spurious_trip_rate     # -> 5.0101e-07   per hour
result.mean_time_to_spurious_trip / 8760   # -> 227.9   years
```

The demand mode is always given: it decides whether the SIL is read from
the PFDavg (IEC 61508-1 Table 2) or from the PFH (Table 3), and nothing in
the channels says which. Each band takes its lower edge and not its upper
one, so a PFDavg of exactly `1e-3` is SIL 2:

```python
sil_band(1e-3, "low_demand").sil      # -> 2
sil_band(9.99e-4, "low_demand").sil   # -> 3
function.sil("high_demand").sil       # -> 2   from the PFH
```

The PFH's bands are per hour. For channels whose rates are in another
unit, say how many hours it is: `function.sil("high_demand",
hours_per_unit=8760)` for rates per year.

`result.safe_failure_fraction` gives each channel's as a number. The
architectural constraints it is read against (IEC 61508-2's hardware fault
tolerance tables) are a judgement of the records behind the rates, and are
not made here.

## What is computed

`function.dangerous` is the diagram of the dangerous failures. Each
channel is two components in series: its undetected failures, `(name,
"DU")`, hidden until the proof test (a component with an `"inspection"`),
and its detected ones, `(name, "DD")`, repaired as they happen. The
channels are voted `k`-out-of-N.

- The **PFDavg** is its
  [`mean_unavailability`][repyability.RepairableRBD.mean_unavailability]:
  the probability that the function is down at each time of the proof
  tests' period, averaged over the period.
- The **PFH** is its
  [`system_failure_frequency`][repyability.RepairableRBD.system_failure_frequency]:
  each channel's failure rate times the probability that its failure takes
  the function down, averaged the same way.

`function.safe` is the diagram of the safe failures under the dual vote. A
2oo3 function trips when two channels have failed safe together, so it
stays clear of a spurious trip while two of three have not: an M-out-of-N
function is (N − M + 1)-out-of-N against spurious trips. The
**spurious-trip rate** is that diagram's `system_failure_frequency`.

Both diagrams are ordinary `RepairableRBD`s, so everything else is open on
them: the PFD over time (`function.dangerous.point_unavailability`), the
importance measures, `optimal_inspection_intervals`, the simulation.

```python
breakdown = function.pfd_breakdown()
breakdown.fractions[(("a", "DU"), ("b", "DU"), ("c", "DU"))]   # -> 0.781
```

Here 78% of the PFDavg is the undetected failures' common cause.

### Detected failures that trip

By default a channel with a detected dangerous failure stays dangerous
until it is repaired. Where the logic takes a detected failure as a vote
to trip, pass `detected_trips=True`: the detected failures then leave the
dangerous diagram and count as safe ones.

```python
tripping = SafetyFunction(
    {"a": transmitter, "b": transmitter, "c": transmitter},
    k=2, beta=0.1, beta_detected=0.05, beta_safe=0.1, detected_trips=True,
)
tripping.pfh()                  # -> 2.8278e-07
tripping.spurious_trip_rate()   # -> 8.0258e-07
```

### Votes in series

Sensors, a logic solver and final elements are votes in series: the
function acts when each does, and trips when any does.
`SafetyFunction.series` draws them into one dangerous and one safe
diagram, so votes tested at the same times are not treated as independent
on average.

```python
valve = SafetyChannel(5e-6, safe=2e-6, repair=24.0, proof_test_interval=8760.0)
valves = SafetyFunction({"v1": valve, "v2": valve}, k=1, beta=0.05)
whole = SafetyFunction.series(function, valves)
whole.pfd_avg()   # -> 2.7774e-03
```

## The simplified formulas, beside the exact values

IEC 61508-6 Annex B gives formulas for 1oo1, 1oo2, 2oo2, 2oo3 and 1oo3
votes (B.3.2.2 for the PFDavg, B.3.3.2 for the PFH).
`simplified_formulas()` evaluates them for the function's vote. With
`λD = λDU + λDD`, the channel's and the group's mean down times are

```text
t_CE = (λDU / λD) (T1/2 + MRT) + (λDD / λD) MTTR
t_GE = (λDU / λD) (T1/3 + MRT) + (λDD / λD) MTTR
```

and, for example,

```text
1oo1   PFDavg = λD t_CE                          PFH = λDU
1oo2   PFDavg = 2 ((1-βD) λDD + (1-β) λDU)² t_CE t_GE
                + βD λDD MTTR + β λDU (T1/2 + MRT)
       PFH    = 2 ((1-βD) λDD + (1-β) λDU) (1-β) λDU t_CE + β λDU
2oo3   as 1oo2, with 6 in place of 2
```

```python
simplified = function.simplified_formulas()
simplified.pfd_avg   # -> 1.1473e-03   2.2% above the exact 1.1227e-03
simplified.pfh       # -> 2.8839e-07
function.pfh(undetected_only=True)   # -> 2.8302e-07
```

The formulas are approximations, in four ways:

1. **First order in `λ T1`.** They keep the leading term of the exact
   value. With undetected failures alone and repairs in no time, the 1oo1
   formula `λDU T1 / 2` is high by `λDU T1 / 3` of the value; the error
   grows with the redundancy and with `λ T1`.
2. **Products of averages.** `t_CE t_GE` multiplies down times where the
   exact value averages the product of the channels' probabilities of
   being down. The two agree for the undetected failures (that is what
   `T1/3` is for) and differ in the terms that mix detected and undetected
   failures.
3. **Common cause and repair.** The formulas add `βD λDD MTTR`: the
   channels fail together and the vote is down for one repair. Channels
   that fail together are repaired one by one, and a 1oo2 or 2oo3 vote is
   back when the first enough of them are: the exact share of the detected
   failures' common cause is about half the formula's for 1oo2. And a vote
   one failure takes down (1oo1, 2oo2) has no common-cause term in the
   annex, where a shared cause takes one failure for two: `2 λ` is high by
   `β/2` for 2oo2.
4. **What the PFH counts.** The annex's PFH counts the function's failures
   that an undetected failure completes: it takes a detected failure to be
   answered by bringing the equipment to a safe state. `pfh()` counts
   every dangerous failure of the function, and
   `pfh(undetected_only=True)` what the annex counts. With diagnostics the
   two differ by much more than any approximation: here a detected failure
   completes about half the function's failures. If a detected failure
   does trip the channel, say so with `detected_trips=True`, and the PFH
   is that of the undetected failures alone.

For the transmitter above (`λDU = 2e-6`, `λDD = 3e-6`, `λS = 5e-6` per
hour, `T1 = 8760 h`, `MTTR = MRT = 8 h`), without common cause and with
`β = 10%`, `βD = 5%`, `βS = 10%`:

| Vote | β | PFDavg, exact | Annex B | gap | PFH, exact | of it, undetected | Annex B | gap to undetected |
|---|---|---|---|---|---|---|---|---|
| 1oo1 | – | 8.748e-03 | 8.800e-03 | +0.6% | 4.956e-06 | 1.983e-06 | 2.000e-06 | +0.9% |
| 1oo2 | 0 | 1.015e-04 | 1.035e-04 | +1.9% | 8.647e-08 | 3.459e-08 | 3.520e-08 | +1.8% |
| 2oo2 | 0 | 1.740e-02 | 1.760e-02 | +1.2% | 9.826e-06 | 3.930e-06 | 4.000e-06 | +1.8% |
| 2oo3 | 0 | 3.019e-04 | 3.105e-04 | +2.8% | 2.564e-07 | 1.026e-07 | 1.056e-07 | +3.0% |
| 1oo3 | 0 | 1.325e-06 | 1.372e-06 | +3.5% | 1.503e-09 | 6.012e-10 | 6.209e-10 | +3.3% |
| 1oo2 | 10% | 9.591e-04 | 9.683e-04 | +1.0% | 4.221e-07 | 2.279e-07 | 2.295e-07 | +0.7% |
| 2oo2 | 10% | 1.654e-02 | 1.760e-02 | +6.4% | 9.490e-06 | 3.737e-06 | 4.000e-06 | +7.0% |
| 2oo3 | 10% | 1.123e-03 | 1.147e-03 | +2.2% | 5.647e-07 | 2.830e-07 | 2.884e-07 | +1.9% |
| 1oo3 | 10% | 8.773e-04 | 8.799e-04 | +0.3% | 3.508e-07 | 2.003e-07 | 2.005e-07 | +0.1% |

At these rates (`λDU T1 = 0.0175`) the formulas are a few percent high,
which is the safe side. They are not always close. A channel ten times
less reliable (`λDU = 2e-5` per hour, nothing else) gives

```python
worn = SafetyChannel(2e-5, repair=8.0, proof_test_interval=8760.0)
voted = SafetyFunction({"a": worn, "b": worn, "c": worn}, k=2)
voted.pfd_avg()                       # -> 2.4808e-02
voted.simplified_formulas().pfd_avg   # -> 3.0835e-02   24% high
voted.pfh()                           # -> 7.9105e-06
voted.simplified_formulas().pfh       # -> 1.0531e-05   33% high
```

The spurious-trip rate's rare-event form (`N λS` for 1ooN, `2 λS² MTTR`
for 2oo2, `6 λS² MTTR` for 2oo3, plus `βS λS` where two failures are
needed) is within 0.01% of the exact rate here without common cause. With
it, a 1ooN vote's exact rate is lower, as a shared cause trips once where
`N λS` counts it N times: 9.50e-06 against 1.00e-05 for 1oo2.

```python
pair = SafetyFunction({"a": transmitter, "b": transmitter}, k=1, beta_safe=0.1)
pair.spurious_trip_rate()                       # -> 9.4993e-06
pair.simplified_formulas().spurious_trip_rate   # -> 1.0000e-05
```

## How each value is worked out

`function.analysis_routes()` says, without running anything, whether each
result is exact, numerical or refused, as the diagrams' own reports do.
The simplified formulas are refused, with the reason, for a vote the annex
has no formula for, channels that differ, a life with no constant rate, a
proof test that can miss, and votes in series; the exact values hold for
all of those.

```python
function.analysis_routes()["pfh"].route   # 'numerical'
```
