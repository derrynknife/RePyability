"""Study: does the bad-year downtime bound cover at its stated rate?

UC-68 of the maintenance README. ``least_spend_for_downtime`` keeps a
design only when an upper confidence bound on its downtime's quantile is
under the limit. This study checks that bound end to end: for a pump with
exponential lives (mean 1,000 h) and repairs (mean 50 h) over 8,760 h,
whose downtime's 95th percentile is exact (766.3532 h, by uniformisation:
``repyability/tests/test_downtime_risk.py``'s ``downtime_cdf``), it runs
the diagram's evaluation with seeds 1 to ``REPEATS`` and counts how often
the 95% bound is at or above the exact percentile.

The design was fixed before any result was seen: 1,000 repeats of 500
runs, seeds 1 to 1,000, level 0.95, confidence 0.95. The order-statistic
bound is distribution-free, so its coverage is at least 0.95 up to the
ties of a discrete outcome (none here: downtime has an atom at 0 only,
far below the percentile). Pass: the coverage's 99% binomial interval
reaches 0.95. Run at consolidation, with one process:

    PYTHONPATH=. python scripts/downtime_bound_coverage.py
"""

import math

import numpy as np
import surpyval as surv
from scipy import stats

from repyability import RepairableRBD

EXACT = 766.3532060243101  # the 95th percentile, by uniformisation
REPEATS = 1000
RUNS = 500
LEVEL = CONFIDENCE = 0.95


def main() -> None:
    E = surv.Exponential.from_params
    pump = RepairableRBD(
        [("s", "p"), ("p", "t")],
        {"p": {"reliability": E([1e-3]), "repairability": E([0.02])}},
    )
    covered = 0
    for seed in range(1, REPEATS + 1):
        found = pump.least_spend_for_downtime(
            8760.0,
            1e9,
            level=LEVEL,
            confidence=CONFIDENCE,
            mc_samples=RUNS,
            seed=seed,
        )
        covered += found.design.bound >= EXACT
    low, high = stats.binomtest(covered, REPEATS).proportion_ci(0.99)
    share = covered / REPEATS
    print(
        f"coverage {share:.4f} ({covered} of {REPEATS}), 99% interval "
        f"[{low:.4f}, {high:.4f}]; stated {CONFIDENCE}"
    )
    print("PASS" if high >= CONFIDENCE else "FAIL")
    assert math.isfinite(share) and np.isfinite(low)


if __name__ == "__main__":
    main()
