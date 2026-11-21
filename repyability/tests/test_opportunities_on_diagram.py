"""A diagram's shutdowns given as the site's ``Opportunities`` in model
time: each opportunity is a shutdown of its name, and the record is the
diagram its dicts make, value for value and run for run."""

import json
import warnings

import numpy as np
import pytest
import surpyval as surv

from repyability import (
    Opportunities,
    Opportunity,
    RepairableRBD,
    ShutdownPolicy,
)

E = surv.Exponential.from_params
W = surv.Weibull.from_params
EDGES = [("s", "p"), ("p", "q"), ("q", "t")]


def components():
    return {
        "p": {
            "reliability": W([500.0, 2.0]),
            "repairability": E([0.1]),
            "repair_cost": 50.0,
            "preventive": {"shutdown": "overhaul", "cost": 5.0},
        },
        "q": {
            "reliability": W([800.0, 2.5]),
            "repairability": E([0.2]),
            "repair_cost": 20.0,
            "replace_cost": 30.0,
            "preventive": {"interval": 700.0, "policy": "age", "cost": 8.0},
        },
    }


def both(opportunities, shutdowns, **record):
    site = Opportunities(opportunities, **record)
    return (
        RepairableRBD(EDGES, components(), shutdowns=site),
        RepairableRBD(EDGES, components(), shutdowns=shutdowns),
    )


def test_a_record_is_the_diagram_its_dicts_make():
    # Every 300 from 900 (both ways): on the diagram's clock, the first at
    # 300, as a dict's offset 0 puts it.
    by_record, by_dicts = both(
        [
            Opportunity(
                "overhaul",
                period=300.0,
                offset=900.0,
                duration=10.0,
                takes_down=["p"],
            )
        ],
        {
            "overhaul": {
                "period": 300.0,
                "offset": 0.0,
                "duration": 10.0,
                "scope": ["p"],
                "cost": 40.0,
            }
        },
        shut_costs={"overhaul": 40.0},
    )
    assert by_record.to_dict() == by_dicts.to_dict()
    a = by_record.expected_cost(2000.0)
    b = by_dicts.expected_cost(2000.0)
    assert a.mean == b.mean and a.by_category == b.by_category
    assert a.by_category["shutdown"] == 6 * 40.0
    runs = by_record.availability(2000.0, mc_samples=100, seed=4)
    again = by_dicts.availability(2000.0, mc_samples=100, seed=4)
    np.testing.assert_array_equal(runs.availability, again.availability)
    # Saved, it is the dicts, and comes back as the same diagram.
    saved = json.loads(json.dumps(by_record.to_dict()))
    assert RepairableRBD.from_dict(saved).to_dict() == by_dicts.to_dict()


def test_times_idle_ageing_and_opportunistic_renewals_carry_over():
    by_record, by_dicts = both(
        [
            Opportunity(
                "overhaul",
                dates=[250.0, 900.0, 1400.0],
                duration=12.0,
                idle_ageing=False,
                opportunistic={"q": 300.0},
            )
        ],
        {
            "overhaul": {
                "times": [250.0, 900.0, 1400.0],
                "duration": 12.0,
                "idle_ageing": False,
                "opportunistic": {"q": 300.0},
            }
        },
    )
    assert by_record.to_dict() == by_dicts.to_dict()
    runs = by_record.cost(2000.0, mc_samples=100, seed=7)
    again = by_dicts.cost(2000.0, mc_samples=100, seed=7)
    np.testing.assert_array_equal(runs.samples, again.samples)
    jobs = by_record.availability(2000.0, mc_samples=100, seed=7)
    assert (
        jobs.outage_jobs
        == by_dicts.availability(2000.0, mc_samples=100, seed=7).outage_jobs
    )
    assert jobs.outage_jobs


def test_a_shut_with_no_duration_takes_no_time():
    by_record, by_dicts = both(
        [Opportunity("overhaul", period=400.0)],
        {"overhaul": {"period": 400.0, "offset": 0.0, "duration": 0.0}},
    )
    assert by_record.to_dict() == by_dicts.to_dict()


def test_the_diagram_options_are_saved_with_the_opportunity():
    stop = Opportunity(
        "overhaul",
        period=300.0,
        idle_ageing=False,
        opportunistic={"q": 300.0},
    )
    assert repr(stop) == (
        "Opportunity('overhaul', period=300.0, idle_ageing=False, "
        "opportunistic=(('q', 300.0),))"
    )
    saved = json.loads(json.dumps(stop.to_dict()))
    assert Opportunity.from_dict(saved) == stop
    plain = Opportunity("overhaul", period=300.0)
    # A record saved before these options is read as it was.
    assert "idle_ageing" not in plain.to_dict()
    assert "opportunistic" not in plain.to_dict()
    site = Opportunities([stop], shut_costs={"overhaul": 12.0})
    assert Opportunities.from_dict(site.to_dict()) == site
    assert Opportunities([plain]).to_dict().get("shut_costs") is None


def test_refusals_say_what_to_give():
    dated = Opportunities([Opportunity("stop", dates=["2027-03-03"])])
    with pytest.raises(ValueError, match="on_diagram"):
        RepairableRBD(EDGES, components(), shutdowns=dated)
    with pytest.raises(ValueError, match="Opportunities the policy"):
        RepairableRBD(EDGES, components(), shutdowns=ShutdownPolicy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        priced = Opportunity("overhaul", period=300.0, cost=10.0)
    with pytest.raises(ValueError, match="shut_costs"):
        RepairableRBD(EDGES, components(), shutdowns=Opportunities([priced]))
    with pytest.raises(ValueError, match="not one of the opportunities"):
        Opportunities(
            [Opportunity("overhaul", period=300.0)], shut_costs={"x": 1.0}
        )
    with pytest.raises(ValueError, match="at least 0"):
        Opportunities(
            [Opportunity("overhaul", period=300.0)],
            shut_costs={"overhaul": -1.0},
        )
    with pytest.raises(ValueError, match="idle_ageing must be True"):
        Opportunity("overhaul", period=300.0, idle_ageing="no")
    with pytest.raises(ValueError, match="opportunistic must be an age"):
        Opportunity("overhaul", period=300.0, opportunistic=-1.0)
    # The diagram checks the scope against its nodes, as for a dict.
    with pytest.raises(ValueError, match="scope"):
        RepairableRBD(
            EDGES,
            components(),
            shutdowns=Opportunities(
                [Opportunity("overhaul", period=300.0, takes_down=["x"])]
            ),
        )
