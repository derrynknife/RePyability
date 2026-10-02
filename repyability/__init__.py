"""RePyability — reliability engineering tools for Python.

The most commonly used classes are re-exported here so they can be imported
directly from the top-level package, e.g.::

    from repyability import NonRepairableRBD, StandbyModel
"""

from repyability._version import __version__
from repyability.demonstration import (
    demonstrated_mtbf,
    demonstrated_reliability,
    demonstration_pass_probability,
    demonstration_sample_size,
    demonstration_test_multiple,
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
    ConfidenceInterval,
    ControlVariate,
    CostResult,
    Criticalities,
    ExpectedCost,
    ExpectedEvents,
    FailureCriticalityIndex,
    MaintenancePlan,
    RedundancyAllocation,
    ReliabilityRedundancyAllocation,
    RestorationCriticalityIndex,
    SparesDemand,
    SparesStock,
    TotalCostAllocation,
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
    "mtbf_test_time",
    "demonstrated_mtbf",
    "mtbf_pass_probability",
    "ComponentOption",
    # Result types
    "AnalysisRoute",
    "AvailabilityResult",
    "CapacityDistribution",
    "ConfidenceInterval",
    "ControlVariate",
    "CostResult",
    "ExpectedCost",
    "ExpectedEvents",
    "RedundancyAllocation",
    "ReliabilityRedundancyAllocation",
    "TotalCostAllocation",
    "UncertaintyResult",
    "Criticalities",
    "UpDownImportance",
    "FailureCriticalityIndex",
    "RestorationCriticalityIndex",
    "MaintenancePolicy",
    "FailureLimitPolicy",
    "MaintenancePlan",
    "AvailabilityAllocation",
    "SparesDemand",
    "SparesStock",
    "SimulationChunk",
    "run_shard",
]
