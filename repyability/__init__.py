"""RePyability: the reliability of systems, built from fitted component
models.

A system is a reliability block diagram over its components' lifetime
models (surpyval's fits, or anything with ``sf`` and ``ff``), worked out
exactly or numerically where it can be, and simulated on request:

- ``NonRepairableRBD``: reliability over time, the MTTF, B-lives,
  importance and the Greeks, allocation, and uncertainty from fitted models.
- ``RepairableRBD``: availability (long run, over time and over a mission,
  and the unavailability to its own precision), failures and costs,
  maintenance, inspections, spares, repair crews, and simulated histories.
- ``FaultTree``, ``Network`` and ``PhasedMission``: the other ways to draw
  a system; ``StandbyModel``, ``LoadSharingModel``, ``RepeatedNode`` and
  ``CCFGroup`` the dependencies within one.
- ``demonstration_plan`` and the other ``demonstration_*`` and ``mtbf_*``
  functions: demonstration test planning.

Every class and function here is imported from the top-level package::

    from repyability import NonRepairableRBD, RepairableRBD

Each analysis says whether it is exact, numerical, simulated or refused for
a diagram: ``rbd.analysis_routes()``. Each result class has ``to_dict()``,
for ``json.dumps``.

The user guide, the course and the API reference:
https://derrynknife.github.io/RePyability/
"""

from repyability._version import __version__
from repyability.demonstration import (
    demonstrated_mtbf,
    demonstrated_reliability,
    demonstration_pass_probability,
    demonstration_plan,
    demonstration_sample_size,
    demonstration_test_multiple,
    mtbf_demonstration_plan,
    mtbf_pass_probability,
    mtbf_test_time,
)
from repyability.fault_tree import FaultTree
from repyability.maintenance import FailureLimitPolicy, MaintenancePolicy
from repyability.network import Network
from repyability.non_repairable import NonRepairable
from repyability.rbd.ccf import MGL, BetaFactor, CCFGroup
from repyability.rbd.chunks import SimulationChunk
from repyability.rbd.degrading_node import DegradingNode
from repyability.rbd.helper_classes import (
    PerfectReliability,
    PerfectUnreliability,
)
from repyability.rbd.load_sharing_node import LoadSharingModel
from repyability.rbd.node_state import NodeState
from repyability.rbd.non_repairable_rbd import NonRepairableRBD
from repyability.rbd.phased_mission import PhasedMission
from repyability.rbd.rbd import RBD
from repyability.rbd.redundancy_allocation import ComponentOption
from repyability.rbd.regression_node import RegressionNode
from repyability.rbd.repairable_rbd import RepairableRBD
from repyability.rbd.repeated_node import RepeatedNode
from repyability.rbd.repeated_standby_node import RepeatedStandbyNode
from repyability.rbd.results import (
    AvailabilityAllocation,
    AvailabilityResult,
    CapacityDistribution,
    ConditionalRun,
    ConfidenceInterval,
    ControlVariate,
    CostResult,
    Criticalities,
    DemonstrationPlan,
    ExpectedCost,
    ExpectedEvents,
    FailureCriticalityIndex,
    Lever,
    MaintenancePlan,
    RateBreakdown,
    RedundancyAllocation,
    ReliabilityRedundancyAllocation,
    RestorationCriticalityIndex,
    SparesDemand,
    SparesStock,
    TimelineSimulation,
    TotalCostAllocation,
    UncertaintyImportance,
    UncertaintyResult,
    UpDownImportance,
)
from repyability.rbd.routes import AnalysisRoute
from repyability.rbd.shards import run_shard
from repyability.rbd.standby_node import StandbyModel
from repyability.repairable import (
    Repairable,
    minimal_repair_time_to_nth_failure,
)
from repyability.timelines import Timeline, Timelines

__all__ = [
    "__version__",
    # System models
    "RBD",
    "NonRepairableRBD",
    "RepairableRBD",
    "FaultTree",
    "PhasedMission",
    "Network",
    # Component models
    "NonRepairable",
    "Repairable",
    "StandbyModel",
    "DegradingNode",
    "LoadSharingModel",
    "RepeatedNode",
    "RepeatedStandbyNode",
    # Helpers
    "PerfectReliability",
    "PerfectUnreliability",
    "NodeState",
    "RegressionNode",
    "BetaFactor",
    "MGL",
    "CCFGroup",
    "minimal_repair_time_to_nth_failure",
    # Demonstration test planning
    "demonstration_sample_size",
    "demonstrated_reliability",
    "demonstration_test_multiple",
    "demonstration_pass_probability",
    "demonstration_plan",
    "mtbf_test_time",
    "demonstrated_mtbf",
    "mtbf_pass_probability",
    "mtbf_demonstration_plan",
    "ComponentOption",
    # Up/down histories
    "Timeline",
    "Timelines",
    # Result types
    "AnalysisRoute",
    "AvailabilityResult",
    "CapacityDistribution",
    "ConditionalRun",
    "ConfidenceInterval",
    "ControlVariate",
    "CostResult",
    "ExpectedCost",
    "ExpectedEvents",
    "Lever",
    "RedundancyAllocation",
    "ReliabilityRedundancyAllocation",
    "TotalCostAllocation",
    "UncertaintyResult",
    "UncertaintyImportance",
    "Criticalities",
    "DemonstrationPlan",
    "UpDownImportance",
    "FailureCriticalityIndex",
    "RestorationCriticalityIndex",
    "MaintenancePolicy",
    "FailureLimitPolicy",
    "MaintenancePlan",
    "RateBreakdown",
    "AvailabilityAllocation",
    "SparesDemand",
    "SparesStock",
    "TimelineSimulation",
    "SimulationChunk",
    "run_shard",
]


def _readable_help() -> None:
    # The API reference's cross-references read as names in help() (#238).
    from repyability.rbd.rbd import Pairs
    from repyability.utils.docs import readable_docstrings

    readable_docstrings([globals()[name] for name in __all__[1:]] + [Pairs])


_readable_help()
