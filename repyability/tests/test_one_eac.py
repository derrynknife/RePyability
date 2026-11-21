"""One definition of the equivalent annual cost of age replacement
(``maintenance._options.cycle_eac``), which a part alone
(``NonRepairable``) and a diagram's component (``node_eac``) both take,
against Fox's closed form for an exponential life."""

import math

import pytest
import surpyval as surv

from repyability import NonRepairable, RepairableRBD
from repyability.maintenance._options import (
    cycle_eac,
    discounted_optimum,
    discounted_run_to_failure,
)
from repyability.rbd._discounted_long_run import node_eac

LAMBDA, RATE, AGE = 1e-3, 2e-4, 700.0
CP, CU = 30.0, 200.0
E = surv.Exponential.from_params([LAMBDA])


def fox(age: float) -> float:
    """``(cu l M + cp exp(-(l + r) T)) / M``, ``M = (1 - exp(-(l + r) T))
    / (l + r)``: an exponential life replaced at ``T``, discounted."""
    both = LAMBDA + RATE
    kept = math.exp(-both * age)
    length = (1.0 - kept) / both
    return (CU * LAMBDA * length + CP * kept) / length


def test_the_one_eac_is_fox_s():
    part = NonRepairable(E)
    assert cycle_eac(
        part, AGE, RATE, failure_cost=CU, planned_cost=CP
    ) == pytest.approx(fox(AGE), rel=1e-9)


def test_a_part_alone_and_a_component_take_the_one_eac():
    rbd = RepairableRBD(
        [("s", "u"), ("u", "t")],
        {
            "u": {
                "reliability": E,
                "repairability": "instant",
                "repair_cost": CU,
                "preventive": {"interval": AGE, "cost": CP},
            }
        },
    )
    part = rbd.components["u"]
    assert node_eac(rbd, "u", RATE) == cycle_eac(
        part, AGE, RATE, failure_cost=CU, planned_cost=CP
    )
    best, eac = discounted_optimum(part, RATE, CP, CU)
    if math.isinf(best):
        assert eac == discounted_run_to_failure(part, RATE, CU)
    else:
        assert eac == pytest.approx(
            cycle_eac(part, best, RATE, failure_cost=CU, planned_cost=CP),
            rel=1e-12,
        )
    assert discounted_run_to_failure(part, RATE, CU) == cycle_eac(
        part, math.inf, RATE, failure_cost=CU
    )


def test_downtime_and_work_that_takes_time_lengthen_the_cycle():
    part = NonRepairable(E)
    down, repair, planned = 5.0, 12.0, 4.0
    both = LAMBDA + RATE
    kept = math.exp(-both * AGE)
    length = (1.0 - kept) / both
    failed = LAMBDA * length
    expected = (
        failed * (CU + down * repair) + kept * (CP + down * planned)
    ) / (length + failed * repair + kept * planned)
    assert cycle_eac(
        part,
        AGE,
        RATE,
        failure_cost=CU,
        planned_cost=CP,
        downtime=down,
        repair=repair,
        planned=planned,
    ) == pytest.approx(expected, rel=1e-9)
