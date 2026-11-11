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
- ``stress_strength`` and ``stress_strength_over_time``: the chance a
  strength exceeds a stress (interference), exact or numerical.
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
from repyability.maintenance import (
    AgeReplacementValue,
    Calendar,
    CalendarPolicy,
    CostDistribution,
    DelayEffect,
    FailureFinding,
    FailureLimitPolicy,
    LayeredVisits,
    MaintenancePolicy,
    NextReplacement,
    OnCondition,
    Opportunity,
    OptionValue,
    OverhaulReplacePolicy,
    PlannedReplacement,
    RatioChoice,
    RenewalForecast,
    ReplacementChoice,
    ReplacementComparison,
    RunToFailure,
    ScheduledReplacement,
    ScheduleValue,
    SkipValue,
    SlackEffect,
    SwitchTime,
    WindowCosts,
    best_switch_time,
    failure_finding,
    layered_cost_rate,
    on_condition,
    operating_schedule_value,
    optimal_failure_finding_interval,
    optimal_layered_visits,
    optimal_on_condition_interval,
    run_to_failure,
    scheduled_replacement,
)
from repyability.network import Network
from repyability.non_repairable import NonRepairable
from repyability.rbd._bayesian_life import PosteriorPredictive
from repyability.rbd._mission_results import (
    MissionDrivers,
    MissionPhaseSummary,
    MissionSummary,
)
from repyability.rbd._mission_settings import BlockSetting, MissionPhase
from repyability.rbd._model_utils import model_parameters
from repyability.rbd._streams import seed_entropy
from repyability.rbd.ccf import MGL, AlphaFactor, BetaFactor, CCFGroup
from repyability.rbd.chunks import SimulationChunk
from repyability.rbd.degrading_node import DegradingNode
from repyability.rbd.helper_classes import (
    PerfectReliability,
    PerfectUnreliability,
)
from repyability.rbd.joint_node import JointNode
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
    ConditionalProbabilities,
    ConditionalRun,
    ConfidenceInterval,
    ControlVariate,
    CoreStructure,
    CostByWindow,
    CostResult,
    CrewJob,
    CrewStates,
    Criticalities,
    DemonstrationPlan,
    EventsByWindow,
    ExpectedCost,
    ExpectedEvents,
    ExpectedNPV,
    FailureCriticalityIndex,
    FailureForecast,
    Lever,
    MaintenancePlan,
    ModelParameters,
    NodeActionRates,
    PlanVsActual,
    RateBreakdown,
    RedundancyAllocation,
    ReliabilityRedundancyAllocation,
    ReplacementCycle,
    RestorationCriticalityIndex,
    SimulationSamples,
    SparesDemand,
    SparesStock,
    StressStrengthCurve,
    StressStrengthResult,
    StructureSummary,
    TimelineSimulation,
    TotalCostAllocation,
    UncertaintyImportance,
    UncertaintyResult,
    UpDownImportance,
)
from repyability.rbd.routes import AnalysisRoute, AnalysisRoutes
from repyability.rbd.shards import run_shard
from repyability.rbd.standby_node import StandbyModel
from repyability.repairable import (
    Repairable,
    minimal_repair_time_to_nth_failure,
)
from repyability.stress_strength import (
    stress_strength,
    stress_strength_over_time,
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
    "MissionPhase",
    "BlockSetting",
    "MissionSummary",
    "MissionPhaseSummary",
    "MissionDrivers",
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
    "JointNode",
    "PosteriorPredictive",
    "BetaFactor",
    "MGL",
    "AlphaFactor",
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
    # Stress-strength interference
    "stress_strength",
    "stress_strength_over_time",
    # Maintenance planning
    "Opportunity",
    "Calendar",
    # Up/down histories
    "Timeline",
    "Timelines",
    # Result types
    "AnalysisRoute",
    "AnalysisRoutes",
    "AvailabilityResult",
    "CapacityDistribution",
    "ConditionalRun",
    "ConfidenceInterval",
    "ControlVariate",
    "CostResult",
    "ExpectedCost",
    "ExpectedEvents",
    "ExpectedNPV",
    "PlanVsActual",
    "EventsByWindow",
    "CostByWindow",
    "StructureSummary",
    "CoreStructure",
    "NodeActionRates",
    "CrewJob",
    "CrewStates",
    "ReplacementCycle",
    "SimulationSamples",
    "ConditionalProbabilities",
    "ModelParameters",
    "Lever",
    "RedundancyAllocation",
    "ReliabilityRedundancyAllocation",
    "TotalCostAllocation",
    "UncertaintyResult",
    "UncertaintyImportance",
    "Criticalities",
    "DemonstrationPlan",
    "StressStrengthResult",
    "StressStrengthCurve",
    "UpDownImportance",
    "FailureCriticalityIndex",
    "RestorationCriticalityIndex",
    "MaintenancePolicy",
    "FailureLimitPolicy",
    "OverhaulReplacePolicy",
    "FailureForecast",
    "SkipValue",
    "ReplacementComparison",
    "OptionValue",
    "DelayEffect",
    "AgeReplacementValue",
    "SlackEffect",
    "RenewalForecast",
    "CostDistribution",
    "WindowCosts",
    "LayeredVisits",
    "RatioChoice",
    "optimal_layered_visits",
    "layered_cost_rate",
    "RunToFailure",
    "ScheduledReplacement",
    "FailureFinding",
    "OnCondition",
    "run_to_failure",
    "scheduled_replacement",
    "failure_finding",
    "optimal_failure_finding_interval",
    "on_condition",
    "optimal_on_condition_interval",
    "NextReplacement",
    "ReplacementChoice",
    "CalendarPolicy",
    "PlannedReplacement",
    "ScheduleValue",
    "SwitchTime",
    "operating_schedule_value",
    "best_switch_time",
    "MaintenancePlan",
    "RateBreakdown",
    "AvailabilityAllocation",
    "SparesDemand",
    "SparesStock",
    "TimelineSimulation",
    "SimulationChunk",
    "run_shard",
    "seed_entropy",
    "model_parameters",
]


def _readable_help() -> None:
    # The API reference's cross-references read as names in help() (#238).
    from repyability.rbd.rbd import Pairs
    from repyability.utils.docs import readable_docstrings

    readable_docstrings([globals()[name] for name in __all__[1:]] + [Pairs])


_readable_help()
