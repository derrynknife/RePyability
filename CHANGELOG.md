# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`LoadSharingModel`: load-sharing dynamic node (dependent failure).** A
  sibling to `StandbyModel` where *n* coupled units share a total load and the
  survivors carry more (and so age faster) as siblings fail — the group works
  while ≥ `k` survive. Each unit is a fitted AFT model, and a cumulative-exposure
  event loop advances every unit's baseline clock at rate `phi(L / survivors)`.
  Identical Exponential-baseline units get an exact hypoexponential closed form;
  otherwise the survival function is a Kaplan-Meier fit to simulated lifetimes
  (`is_simulated` reports which). With no load effect (`phi == 1`) the group
  reduces exactly to the ordinary k-out-of-n parallel result. Plugs into an RBD
  as a single simulation-backed node and serialises with it. (#38)
- **Common-cause failures (CCF) — beta-factor model.** `NonRepairableRBD` gains
  a `ccf_groups` argument taking `CCFGroup(members, BetaFactor(beta))`: a
  fraction `beta` of a symmetric redundant group's failures come from a shared
  cause that fails every member at once, the rest are independent. System
  reliability is computed **exactly** by conditioning on each group's
  shared-cause event and reusing the ordinary independent engine per branch, so
  `beta = 0` reproduces the independent result and `beta = 1` collapses the
  group to a single component. Honoured by `sf()`/`ff()` (and quantities derived
  from them) and persisted through serialisation; the probability-dependent
  importance/sensitivity and condition-based methods raise a clear error on a
  CCF RBD for now (`structural_importance`, being probability-free, is
  unaffected). Alpha-factor and Multiple Greek Letter (partial common-cause) are
  a planned extension. (#44)

## [0.7.0] - 2026-07-20

The **Maintenance & Covariates** milestone: price imperfect-repair (generalized
renewal / Kijima) and replace-at-N-th-failure maintenance policies on
`Repairable`, and drive component reliability from operating covariates with
fitted surpyval regression models — alongside a harmonised RBD constructor
signature.

### Added
- **Imperfect repair (generalized renewal / Kijima) in `Repairable`.** The
  `Repairable` component now spans the whole repair-effectiveness spectrum
  rather than only minimal repair: it sources the expected number of failures
  `E[N(t)]` from the model it is given — analytic `cif` for minimal repair
  (unchanged; e.g. a surpyval Crow-AMSAA) or a seeded `mcf` for imperfect
  repair (a fitted `GeneralizedRenewal`, Kijima I/II). Minimal repair is the
  `q = 1` special case; perfect repair (`q = 0`) is the `NonRepairable`
  boundary. The overhaul/replacement-interval policy is unchanged in form
  `(cr·E[N(t)] + co)/t`; for a simulation-backed model `optimal_overhaul_policy`
  / `find_optimal_overhaul_interval` take `seed`, `n_simulations` and
  `max_interval` for a reproducible, bounded search, and `is_simulated` reports
  which path a `Repairable` uses.
- **Replace-at-N-th-failure policy on `Repairable`.** As an alternative to
  renewing at a fixed age, `optimal_failure_limit_policy` /
  `find_optimal_replacement_failure_count` price repairing on each failure and
  replacing at the N-th, minimising `(cr·(n-1) + co) / E[T_n]`, and return a new
  typed `FailureLimitPolicy(failure_count, cost_rate)`. `E[T_n]` comes from a
  seeded simulation (`expected_time_to_nth_failure`); for the minimal-repair
  (power-law) limit the exact closed form is exposed as
  `minimal_repair_time_to_nth_failure(alpha, beta, n)`.
- **`RegressionNode`: covariate-dependent components in an RBD.** A new node
  wraps a fitted surpyval **regression** survival model (accelerated-failure-
  time, proportional-hazards/Cox, proportional-odds, ...) together with a fixed
  covariate vector `Z` — the component's operating conditions — so its
  reliability is `model.sf(x, Z)`. It is an ordinary univariate node: system
  reliability, importance, MTTF and the condition-based (`age`) layer all work
  with no special handling, because at fixed covariates
  `R(x | age) = sf(age + x, Z) / sf(age, Z)` holds for every regression family.
  The fitted model serialises with the RBD via `surpyval.from_dict`. Simulation-
  based `mean`/`random` require a proper parametric lifetime (AFT/PO/parametric);
  a semiparametric Cox baseline has no defined MTTF and says so clearly. (#37)

### Changed
- **Bumped the surpyval pin to `>=0.15,<0.16`** (from `>=0.13,<0.14`), for the
  regression-model serialisation (`to_dict`/`surpyval.from_dict`) that
  `RegressionNode` relies on. No change to existing behaviour.
- **Harmonised the RBD constructor signatures.** The base ``RBD`` now takes
  ``(edges, nodes, k, ...)`` instead of ``(edges, k, nodes, ...)``, so all
  three classes share the same positional order: the structure first (``edges``
  then the node definition -- ``nodes``/``reliabilities``/``components``), then
  the optional ``k`` (k-out-of-n) modifier. ``NonRepairableRBD`` and
  ``RepairableRBD`` are unchanged. This only affects code that constructed the
  base ``RBD`` with a *positional* ``k`` (``RBD(edges, k_dict)``); pass ``k`` by
  keyword (``RBD(edges, k=k_dict)``) or as the third argument.

## [0.6.0] - 2026-07-19

The **Condition-Based Reliability** milestone: evaluate a system from the
current, sensor-known state of each component (a "digital twin"), and round out
the importance suite with design-time and data-targeting measures.

### Added
- **`structural_importance()`** on every RBD: the Birnbaum importance with all
  node reliabilities at 1/2 — a probability-free, design-time ranking of where
  each node is pivotal in the structure. It depends only on the diagram, so it
  is identical for a `NonRepairableRBD` and a `RepairableRBD` on the same
  layout, and honours the usual `working_nodes`/`broken_nodes` conditioning.
- **`parameter_sensitivity(t)`** on `NonRepairableRBD`: the sensitivity of
  system reliability to each node's distribution parameters,
  `dR_sys/d_theta = birnbaum_importance(node) * d sf_node/d_theta`. The
  parameter derivative is computed numerically (via surpyval's `from_params`),
  so it applies to any parametric model without per-distribution formulae.
  Composite and non-parametric nodes are omitted; forced nodes report zero.
- **Condition-based ("digital twin") evaluation** on `NonRepairableRBD`, driven
  by a new public `NodeState(age, alive)` per node. Each component conditions on
  its own current life `R_i(x | X_i) = R_i(X_i + x) / R_i(X_i)` and that
  propagates exactly through the system:
  - `sf_given_state(x, state)` — system reliability a further `x` from now given
    each node's state (the conditional generalisation of `sf`; an empty state
    reproduces `sf`).
  - `remaining_life(target, state)` — remaining useful life (RUL): the further
    time until system reliability falls to `target`.
  - `importances_given_state(x, state)` — the Birnbaum and criticality
    importances re-evaluated at the conditioned reliabilities.

  Supports ordinary distribution components; dynamic (standby) and composite
  nodes raise if given a state. State is transient input — the structure still
  persists via serialisation.

### Documentation
- **Doctest-backed examples** on the primary public methods (`sf`/`ff`, the
  importance measures, `time_to_reliability`/`bx_life`, the condition-based
  methods, `mean_availability`, and serialisation). They run in CI via
  `--doctest-modules`, so the documented examples cannot silently rot. Examples
  use deterministic (non-simulated) quantities so they need no seeding.
- **Expanded documentation site**: a start-to-finish
  [Tutorial](https://derrynknife.github.io/RePyability/tutorial/) and a
  [Concepts](https://derrynknife.github.io/RePyability/concepts/) reference
  (path/cut sets, exact computation, choosing an importance measure, and
  condition-based conditioning).

## [0.5.1] - 2026-07-18

### Fixed
- **Clean installs of 0.5.0 could not `import repyability`.** Importing
  `surpyval` (which RePyability does at load time) failed with
  `ModuleNotFoundError: No module named 'joblib'`: surpyval 0.11 imported
  joblib unconditionally without declaring it as a dependency, and
  RePyability listed joblib only in its `dev` extra, so a plain
  `pip install repyability` never pulled it in. Fixed by requiring
  `surpyval >= 0.13`, which declares `joblib` as a dependency — a plain
  install now imports cleanly.

### Changed
- Bumped the `surpyval` requirement from `>=0.11.1,<0.12` to `>=0.13,<0.14`.
  surpyval 0.13 relocated its recurrent-event models
  (`surpyval.CrowAMSAA` → `surpyval.recurrent.CrowAMSAA`, and likewise
  `Duane`); `Repairable` is unaffected (it only needs an object exposing
  `cif`), and the docs and tests were updated to the new import path.
- Dropped the now-redundant `joblib` entry from the `dev` extra (surpyval
  provides it transitively).

## [0.5.0] - 2026-07-18

This release focuses on packaging, tooling, correctness and API-consistency
foundations (Phases 0 and 1 of the project's development plan; forward-looking
work is tracked as issues in the GitHub repository).

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
- CI now runs a lint/type gate and a test matrix (Python 3.11–3.12) with a
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
  `--strict`, and it is published to GitHub Pages
  (https://derrynknife.github.io/RePyability/) on every push to `master`.
- **Serialisation.** RBDs round-trip to/from a JSON-friendly structure via
  `to_dict`/`from_dict`/`to_json`/`from_json`, so a diagram can be saved,
  loaded, shared and version-controlled. The structure (edges, k-out-of-n,
  repeated nodes, nested RBDs) and node models (surpyval parametric
  distributions, and the standby/repeated/`NonRepairable` wrappers) are
  reconstructed faithfully; integer and string node names both survive JSON.
  `RBD.from_dict`/`from_json` dispatch on the document's declared type. Fitted
  non-parametric models raise a clear `NotImplementedError` (no reconstruction
  API upstream).
- **Inverse-reliability queries** on `NonRepairableRBD`:
  `time_to_reliability(target)` (solves ``R(t) = target``) and `bx_life(x)`
  (the B\ :sub:`X` life, e.g. `bx_life(10)` is the B10 life). Both accept the
  usual `working_nodes`/`broken_nodes`/`method` arguments.
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
- **Component maintenance-policy layer.** The vestigial `Repairable` class
  is now a real minimal-repair ("as bad as old") economics tool: it takes a
  surpyval recurrence model (anything exposing `cif`, e.g. Crow-AMSAA,
  Duane) and finds the optimal overhaul interval minimising
  `(cr*Lambda(t) + co)/t` (the Barlow-Hunter policy), returning `inf` when
  the unit does not wear out (HPP-like, `beta <= 1`) — validated against
  the power-law closed form. New typed result `MaintenancePolicy`
  (`interval`, `cost_rate`), returned by
  `Repairable.optimal_overhaul_policy()` and the new
  `NonRepairable.optimal_replacement_policy()`.
  `NonRepairable.find_optimal_replacement()` now returns `inf` for an
  exponential lifetime (memoryless — preventive replacement never pays),
  matching the existing Weibull shape <= 1 behaviour. A user-guide section
  explains renewal ("as good as new", `NonRepairable` — also the component
  representation inside `RepairableRBD`) vs minimal repair (`Repairable` —
  standalone; not a valid RBD node model, since RBD repairs are assumed to
  renew).

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
- Requires Python >= 3.11 (raised from 3.10: `surpyval` >= 0.11.1, the core
  dependency, itself requires Python >= 3.11, so the declared 3.10 support could
  not actually be installed).

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
- `Repairable.find_optimal_overhaul_interval()` returned a raw
  `scipy.optimize` result object instead of the interval, performed an
  unbounded search from an arbitrary starting point, and the class
  docstring described the wrong class ("stores the non-repairable
  information"). All fixed in the rebuild; the previously commented-out
  `test_repairable.py` is resurrected with closed-form validation.

### Removed
- Dead/unused `repyability/rbd/rbd_args_check.py` module and the never-
  implemented `working_components`/`broken_components` parameters that its
  (commented-out) check referenced in `sf()`'s docstring.

### Deprecated
- `NonRepairableRBD.fussel_vesely` / `RepairableRBD.fussel_vesely` (use
  `fussell_vesely`).
