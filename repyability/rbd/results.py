"""Typed result objects for RBD analyses.

These dataclasses give the results of the simulation methods documented,
discoverable (IDE autocomplete) attributes instead of opaque nested dicts,
e.g. ``result.criticalities.iou.up`` rather than
``result["criticalities"]["iou"]["up"]``.

For backwards compatibility they also behave as read-only mappings, so the
previous dict-style access keeps working: ``result["availability"]``,
``result.keys()``, ``dict(result)``, ``"criticalities" in result``, and
iteration all still do what they used to. (They are ``collections.abc.Mapping``
instances, not ``dict`` subclasses, so ``isinstance(result, dict)`` is now
False; use ``isinstance(result, Mapping)`` if you need such a check.)
"""

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Hashable

import numpy as np


class _ResultMapping(Mapping):
    """Read-only ``Mapping`` view over a dataclass's fields.

    Preserves the dict-style access the results used to have (``result[key]``,
    ``keys``/``items``/``values``, ``in``, ``dict(result)``, iteration) while
    the subclasses add typed, documented attributes.
    """

    def __getitem__(self, key):
        if key in self._field_names():
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return iter(self._field_names())

    def __len__(self):
        return len(self._field_names())

    def _field_names(self):
        return tuple(f.name for f in dataclasses.fields(self))


@dataclass
class UpDownImportance(_ResultMapping):
    """An importance measure split by system state.

    Attributes
    ----------
    up : dict
        Per-node measure while the system is up.
    down : dict
        Per-node measure while the system is down.
    """

    up: Dict[Hashable, float]
    down: Dict[Hashable, float]


@dataclass
class FailureCriticalityIndex(_ResultMapping):
    """Fractions relating a node's failures to system failures.

    Attributes
    ----------
    per_system_failure : dict
        For each node, the fraction of *system* failures that the node caused.
    per_component_failure : dict
        For each node, the fraction of the node's own failures that caused a
        system failure.
    """

    per_system_failure: Dict[Hashable, float]
    per_component_failure: Dict[Hashable, float]


@dataclass
class RestorationCriticalityIndex(_ResultMapping):
    """Fractions relating a node's restorations to system restorations.

    Attributes
    ----------
    by_system : dict
        For each node, the fraction of *system* restorations that the node
        caused.
    by_component : dict
        For each node, the fraction of the node's own restorations that
        restored the system.
    """

    by_system: Dict[Hashable, float]
    by_component: Dict[Hashable, float]


@dataclass
class Criticalities(_ResultMapping):
    """The importance/criticality measures from an availability simulation.

    Attributes
    ----------
    operational_criticality_index : UpDownImportance
        Time the system and a node were jointly up/down, over the system's
        up/down time.
    iou : UpDownImportance
        Intersection-over-union of a node's and the system's up/down intervals.
    failure_criticality_index : FailureCriticalityIndex
        How much each node drives system failures.
    restoration_criticality_index : RestorationCriticalityIndex
        How much each node drives system restorations.
    """

    operational_criticality_index: UpDownImportance
    iou: UpDownImportance
    failure_criticality_index: FailureCriticalityIndex
    restoration_criticality_index: RestorationCriticalityIndex


@dataclass
class AvailabilityResult(_ResultMapping):
    """The result of :meth:`RepairableRBD.availability`.

    Attributes
    ----------
    timeline : numpy.ndarray
        Event times at which the (mean) system availability changes.
    availability : numpy.ndarray
        Mean system availability at each time in ``timeline``.
    system_uptime : float
        Total system uptime summed over all simulations.
    time_simulated_to : float
        The ``t_simulation`` the simulations were run to.
    criticalities : Criticalities
        The importance/criticality measures (see :class:`Criticalities`).
    components_uptime : dict
        Total uptime per component summed over all simulations.
    components_downtime : dict
        Total downtime per component summed over all simulations.
        (``components_uptime[c] + components_downtime[c]`` equals
        ``N * time_simulated_to``.)
    """

    timeline: np.ndarray
    availability: np.ndarray
    system_uptime: float
    time_simulated_to: float
    criticalities: Criticalities
    components_uptime: Dict[Hashable, float]
    components_downtime: Dict[Hashable, float]
