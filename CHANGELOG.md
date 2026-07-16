# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

This release focuses on packaging, tooling, correctness and API-consistency
foundations (Phases 0 and 1 of the development plan in
`docs/REVIEW_AND_ROADMAP.md`).

### Added
- **Installable packaging.** `pyproject.toml` now declares a PEP 621
  `[project]` table and a `[build-system]`, so the package builds and installs
  from source (`pip install .`) with correct name, version and dependencies.
  Version is sourced from `repyability/_version.py`.
- **`scipy` dependency** is now declared (it was used directly but undeclared).
  `joblib` is declared in the `dev` extra only — it is a testing requirement
  (the suite imports surpyval's RandomSurvivalForest), not a runtime
  dependency.
- **Optional dependency groups**: `.[dev]` (pinned tooling) and `.[docs]`.
- **Seedable Monte-Carlo.** Simulation entry points accept a `seed=`
  argument (`NonRepairableRBD.random/mean/mean_time_to_failure/node_mttf`,
  `RepairableRBD.availability`, `StandbyModel`, `RepeatedNode`,
  `RepeatedStandbyNode`). Seeding is reproducible and restores the caller's
  global RNG state afterwards.
- **System-level `df` (failure density), `hf` (hazard rate) and `Hf`
  (cumulative hazard)** on `NonRepairableRBD`.
- **Curated public API.** Common classes are re-exported from the top-level
  package (`from repyability import NonRepairableRBD, StandbyModel, ...`) and
  listed in `__all__`. `repyability.__version__` is available.
- **`NonRepairableRBD.is_fixed`** property.
- CI now runs a lint/type gate and a test matrix (Python 3.10–3.12) with a
  coverage `fail_under` gate; a release workflow publishes to PyPI via trusted
  publishing.
- `CHANGELOG.md`, `CONTRIBUTING.md`, and issue/PR templates.
- **Typed result objects** for `RepairableRBD.availability()`
  (`AvailabilityResult`, `Criticalities`, `UpDownImportance`,
  `FailureCriticalityIndex`, `RestorationCriticalityIndex`), exported from the
  top-level package. They give documented, IDE-discoverable attribute access
  (`result.criticalities.iou.up`) while remaining backwards compatible with the
  old nested-dict API (subscript, `keys`/`items`/`values`, `in`, `dict()`).
- **Documentation site** (MkDocs + mkdocstrings): overview/quickstart, a user
  guide, and an auto-generated API reference. A CI job builds it with
  `--strict`.
- **Exact steady-state repairable metrics** on `RepairableRBD` via the
  Birnbaum/Vesely frequency formula: `system_failure_frequency()`,
  `mean_up_time()` (MUT), `mean_down_time()` (MDT) and
  `mean_time_between_failures()` (MTBF), all supporting
  `working_nodes`/`broken_nodes` conditioning; `NonRepairable` gains
  `failure_frequency()`. `AvailabilityResult` gains `system_downtime`,
  `system_failures`, `system_restorations` and `n_simulations` fields plus
  simulation-estimate properties `mean_up_time`, `mean_down_time` and
  `failure_frequency` to cross-check against the exact values.
- Fixed `NonRepairable.mean_availability()` crashing with the default
  (instant) replacement time — surpyval's `ExactEventTime.mean()` raises
  `AttributeError`; a workaround computes its mean from its parameter.
- **Simulation uncertainty quantification and validation.**
  `AvailabilityResult` gains `availability_se` (pointwise standard error) and
  `availability_interval(confidence)` (pointwise Wilson confidence band);
  `NonRepairableRBD` gains `mean_time_to_failure_interval()` returning a new
  `ConfidenceInterval` result type (estimate, bounds, standard error). The
  simulator's transient availability is now validated in the test suite
  against exact Markov closed forms (single-component, series and parallel
  exponential systems) within Monte-Carlo sampling error, and the
  finite-window censoring bias of the simulation MUT/MDT estimates is
  documented.

### Changed
- **API standardised across `RBD`/`NonRepairableRBD`/`RepairableRBD`**
  (breaking):
  - **Numpy-style return contract**: scalar time in → `float` out; array in →
    `numpy.ndarray` out. Applies to `sf`/`ff`/`reliability`/`unreliability`/
    `df`/`hf`/`Hf`/`cs`, the per-node accessors, and the importance measures
    (dicts of floats for scalar input). Previously everything returned
    length-1 arrays for scalar input. `mean_availability` and friends return
    plain `float`.
  - `x=None` on a time-varying RBD now raises a clear `ValueError` (it
    previously failed cryptically); fixed-probability RBDs broadcast over an
    array `x` instead of collapsing it to one value.
  - **Renames**: `sf_by_node`/`ff_by_node` → `node_sf`/`node_ff`;
    `get_nodes_names()` → `node_names()`; `time_varying_rbd()` → property
    `is_time_varying`; `initialize_event_queue(working_components,
    broken_components)` → `(working_nodes, broken_nodes)`;
    `AvailabilityResult.components_uptime`/`components_downtime` →
    `node_uptime`/`node_downtime`.
  - **Constructor parity**: `RepairableRBD` now accepts `input_node`,
    `output_node` and `on_infeasible_rbd` like `NonRepairableRBD`, validates
    its node list, and raises a clear error for graph nodes with no component
    definition (previously a `KeyError` mid-simulation).
  - **Importance measures unified**: explicit signatures on both classes,
    typed returns, and all measures now accept `working_nodes`/`broken_nodes`
    to condition the analysis. `RepairableRBD` importances return floats
    (steady state has no time dimension).
  - The `check_x`/`check_probability` decorators now use `functools.wraps`,
    so `help()`, IDE hovers and `inspect` see the real names, docstrings and
    signatures (they previously reported `wrap` with no docstring).
- **`fussel_vesely` → `fussell_vesely`** (corrected Fussell-Vesely spelling).
  The old name remains as a deprecated alias that emits a `DeprecationWarning`.
- Reliability-model dispatch no longer branches on surpyval `dist.name` string
  literals scattered across modules; it goes through capability helpers
  (`repyability/rbd/_model_utils.py`), covered by a surpyval-compatibility
  test.
- `requirements.txt` / `requirements_dev.txt` now install the package and its
  (pinned) extras from `pyproject.toml`, removing the previous mismatch where
  dev pulled surpyval from git HEAD while runtime pinned a release.
- Coverage now measures the library only (tests omitted) and enforces a
  `fail_under` threshold.
- `RepairableRBD.availability()` now returns a typed `AvailabilityResult`
  instead of a plain `dict`. It is a `collections.abc.Mapping` (not a `dict`
  subclass), so dict-style access is preserved but `isinstance(result, dict)`
  is now False (use `isinstance(result, Mapping)`).
- Requires Python >= 3.10.

### Fixed
- Removed a stray debug `print()` in `RBD.improvement_allocation`.
- Replaced `type(x) == Cls` checks with `isinstance` in `NonRepairable`.
- Replaced mutable default arguments (`{}`, `[]`) with `None` sentinels.
- Removed cross-instance access to a name-mangled attribute (`__fixed_probs`).
- **`working_nodes`/`broken_nodes` ("force node working/failed") logic:**
  - `RepairableRBD.availability()` no longer crashes with `ZeroDivisionError`
    (and the operational/IOU criticality indices no longer return `NaN`) when
    a simulation records zero system failures/restorations — e.g. when a
    redundant node is forced working. Zero-exposure measures return 0.
  - A component forced broken in `availability()` is now accounted as **down**
    from t=0 (its uptime and the initial system state were previously assumed
    "up", so a broken component reported full uptime and a system it kept down
    was over-counted as available at the start).
  - `working_nodes`/`broken_nodes` now **raise** on invalid input instead of
    silently ignoring it: unknown node names, the input/output node, or the
    same node in both sets. Previously a typo silently returned a
    plausible-but-wrong result.
  - Setting a repeated node broken now raises the same error as setting it
    working (previously it was silently ignored — an asymmetry).
- **`RepairableRBD.availability()` component downtime accounting.**
  `components_downtime` was accumulated against the *running cumulative*
  component uptime instead of the current simulation's uptime, giving wrong
  (often negative) totals after the first simulation. Each component's
  uptime + downtime now correctly sums to `N * t_simulation`. (`sf`/`ff` and
  the derived `reliability`/`unreliability`/`cs` paths were audited for the
  working/broken overrides and found correct; regression tests added.)

### Removed
- Dead/unused `repyability/rbd/rbd_args_check.py` module and the never-
  implemented `working_components`/`broken_components` parameters that its
  (commented-out) check referenced in `sf()`'s docstring.

### Deprecated
- `NonRepairableRBD.fussel_vesely` / `RepairableRBD.fussel_vesely` (use
  `fussell_vesely`).
