# RePyability — Package Review & Development Plan

_A deep-dive review of the codebase with prioritized recommendations for
improvement and a phased roadmap for new features._

Scope of review: full source tree (`repyability/`, ~3,340 LOC), test suite
(202 tests, ~3,340 LOC), packaging, CI, and docs. All 202 tests pass on
Python 3.11 with numpy 2.x / surpyval 0.11.1; overall line coverage is **91%**.

---

## 1. Executive summary

RePyability is a reliability-engineering toolkit built around **Reliability
Block Diagrams (RBDs)**. It layers a graph model (`networkx.DiGraph`) and the
authors' own survival-analysis library (`surpyval`) into three system classes —
a structural base `RBD`, a time-based `NonRepairableRBD`, and a
simulation-based `RepairableRBD` — plus component models (`NonRepairable`,
`StandbyModel`, `RepeatedNode`, …).

**The mathematical core is genuinely strong.** Exact system reliability via a
memoised Shannon decomposition, minimal cut sets via Berge's algorithm,
k-out-of-n path-set enumeration, deterministic closed forms for cold-standby
survival (Erlang / numerical convolution), and a broad set of importance
measures are all implemented correctly and validated against textbook values.

**The gap is almost entirely "product," not "math."** The package is currently
**not installable from source** (no build metadata), has **no user
documentation or examples**, an **inconsistent public API**, and several
**hygiene/correctness papercuts**. These — not the algorithms — are what stand
between the project and its stated mission of being "accessible for students
right through to practicing professionals."

The single highest-leverage investment is **Phase 0 (packaging + docs
foundation)**: without it, new features cannot reach users and the project
cannot attract contributors.

---

## 1a. Status — progress since this review

Most of the plan below has shipped on branch
`claude/package-review-recommendations-2lmh5a`; the authoritative record is
`CHANGELOG.md`. In brief (all **C/H/M/L findings in §4 are resolved**):

- **Phase 0 (foundation) — done.** PEP 621 packaging (installs as
  `repyability`), reconciled deps (`scipy` declared; `joblib` moved to the
  `dev` extra as a test-only requirement), gating CI (black/isort/flake8/mypy
  + a 3.10–3.12 test matrix + a coverage gate), a PyPI trusted-publishing
  release workflow, and `CHANGELOG`/`CONTRIBUTING`/issue+PR templates.
- **Phase 1 (correctness & API) — done, and extended.** Curated public API +
  `__all__`; seedable Monte-Carlo (save/restore of the global RNG);
  capability-based surpyval dispatch + a compat test; system `df`/`hf`/`Hf`;
  Fussell–Vesely rename (deprecated alias). Extended with a full **API
  standardisation** across the three classes (numpy-style scalar-in/array-in
  return contract, `functools.wraps` on the `check_x`/`check_probability`
  decorators, node-vocabulary renames, `RepairableRBD` constructor parity,
  and `working_nodes`/`broken_nodes` conditioning on every importance
  measure), plus the **force-node-working/failed bug fixes** (a
  `ZeroDivisionError` crash, broken-component accounting, and the
  component-downtime accumulation error).
- **Phase 2 (docs) — done.** MkDocs-Material + mkdocstrings site (overview,
  user guide, auto-generated API reference) with a strict-build CI job.
- **Phase 3 (features) — partly done.** Monte-Carlo hardening (seeds,
  pointwise standard errors + Wilson confidence bands on availability, an
  MTTF confidence interval, and transient availability **validated against
  exact Markov solutions**); exact steady-state repairable metrics
  (MTBF/MUT/MDT/failure-frequency); inverse-reliability queries
  (`time_to_reliability`, `bx_life`); typed result objects
  (`AvailabilityResult` et al.); and **RBD serialisation**
  (`to_dict`/`from_dict`/`to_json`/`from_json`).
- **Descoped by project decision** (see `CLAUDE.md`): **visualization** lives
  in the Reliafy app; **data/distribution fitting** lives in surpyval.

The suite is now **287 tests at ~90% coverage**. The remaining forward work —
including two flagship threads born out of design discussions — is in §8–§9.

---

## 2. What the package does today (architecture map)

```
repyability/
├── __init__.py                 # exports RBD, RepairableRBD (NOT NonRepairableRBD)
├── non_repairable.py           # NonRepairable component: age-replacement cost optimisation
├── repairable.py               # Repairable component: overhaul cost (thin, ~unused)
├── rbd/
│   ├── rbd.py                  # RBD base: path/cut sets, structure fn, system_probability,
│   │                           #   importances, reliability allocation
│   ├── non_repairable_rbd.py   # NonRepairableRBD: R(t), MTTF, importances, DES for MTTF
│   ├── repairable_rbd.py       # RepairableRBD: availability via Monte-Carlo DES, criticality
│   ├── rbd_graph.py            # RBDGraph(DiGraph): structure validation, k-of-n attrs
│   ├── min_path_sets.py        # memoised k-out-of-n minimal path-set solver
│   ├── standby_node.py         # StandbyModel: cold standby (closed form / convolution / MC)
│   ├── numerical_convolution.py# ConvolvedSurvival: deterministic sum-of-lifetimes sf
│   ├── repeated_node.py        # RepeatedNode: series/parallel replication of one model
│   ├── repeated_standby_node.py# RepeatedStandbyNode: repeated cold standby
│   ├── helper_classes.py       # PerfectReliability / PerfectUnreliability
│   └── rbd_args_check.py       # arg-completeness checks (partly unused / 0% covered)
└── utils/wrappers.py           # check_probability, conditional_survival
```

**Feature inventory (implemented and working):**

- **Structure:** minimal path sets, minimal cut sets (Berge), k-out-of-n nodes,
  repeated components, irrelevant-component detection, structure validation.
- **Exact system probability:** `system_probability()` via memoised Shannon
  decomposition (path or cut set), correct for k-of-n; array-vectorised over time.
- **Non-repairable analysis:** `sf`/`ff`/`reliability`/`unreliability`, `cs`
  (conditional survival), `mean_time_to_failure` (Monte-Carlo), per-node sf/ff.
- **Repairable analysis:** transient & steady-state availability, discrete-event
  simulation with criticality indices (FCI, RCI, OCI, IOU).
- **Importance measures:** Birnbaum, improvement potential, RAW, RRW,
  criticality, Fussell–Vesely (cut- and path-set forms).
- **Allocation:** `simple_allocation`, `equal_allocation`, `improvement_allocation`.
- **Component models:** cold standby (k-of-n, imperfect switching for k=1),
  repeated nodes, age-replacement cost optimisation.

---

## 3. Strengths (keep and build on these)

1. **Correct, non-trivial algorithms, well-commented.** The Shannon-decomposition
   reliability (`rbd.py:72`), Berge cut sets (`rbd.py:161`), and memoised
   k-of-n path sets (`min_path_sets.py`) are the kind of code that is hard to
   get right; the explanatory comments are excellent.
2. **Recent numerical hardening.** Replacing noisy Monte-Carlo standby estimates
   with exact Erlang (identical-exponential) and deterministic FFT convolution
   (`numerical_convolution.py`) is a real quality upgrade, and `is_analytically_
   solvable()` transparently tells the user when a result is exact vs simulated.
3. **A differentiated, rich importance/criticality suite** — few open-source
   reliability packages offer both analytic importances and simulation-based
   criticality indices.
4. **A strong, reference-anchored test suite.** 202 tests at 91% coverage,
   asserting against closed-form and textbook values (e.g. series allocation
   `target**(1/n)`, exact `0.994780625`), not merely self-consistency.
5. **Good tooling intent** — pre-commit (isort, pyupgrade, black, flake8, mypy),
   CI on push, coverage reporting.

---

## 4. Findings & recommendations for improvement

Grouped by severity. File references are `path:line`.

### 4.1 Critical — packaging & distribution (blocks everything)

| # | Finding | Evidence | Recommendation |
|---|---------|----------|----------------|
| C1 | **Package has no build metadata.** `pyproject.toml` contains only tool configs — no `[build-system]`, no `[project]`. No `setup.py`/`setup.cfg`. `pip install .` yields `repyability-0.0.0` with no name, version, or dependencies. The repo is effectively not installable; the PyPI release must be stale. | `pyproject.toml` (no `[project]`); `pip install --dry-run .` → `repyability-0.0.0` | Add a PEP 621 `[project]` table (name, dynamic version, deps mirroring `requirements.txt`, Python `>=3.9`, classifiers, URLs, `optional-dependencies` for `dev`/`docs`) and a `[build-system]` (hatchling or setuptools). Make `requirements.txt` reference the package extras, not vice-versa. |
| C2 | **Non-reproducible dev environment.** `requirements_dev.txt` installs `git+https://…/SurPyval.git` (HEAD) while `requirements.txt` pins `surpyval==0.11.1`. Dev deps `black`/`flake8`/`mypy` are unpinned. | `requirements_dev.txt` | Pin all dev tools (or use a lockfile). Install the pinned `surpyval`, not git HEAD, in dev. |
| C3 | **CI does not enforce formatting.** The CI "black" step runs `black $SRC` (in-place), not `black --check`, so it always exits 0 and never fails a PR. Local black 26.x reformats 16 files; pre-commit pins black 22.12.0 (2022). | `.github/workflows/actions.yml`; `black --check` reformats 16 files | Change CI to `black --check`, `isort --check`, `flake8`, `mypy` as gating steps. Pin/refresh pre-commit hook versions and align them with CI. |

### 4.2 High — correctness & reproducibility

| # | Finding | Evidence | Recommendation |
|---|---------|----------|----------------|
| H1 | **Leftover debug `print(out)`** fires on every allocation call. | `rbd.py:584` | Delete it. |
| H2 | **Monte-Carlo is not seedable through the API.** Source uses the global `np.random` / surpyval `.random()`; simulations (`random`, `mean`, `availability`) accept no `rng`/`seed`. Reproducibility is only possible by mutating global numpy state (which the tests do: `np.random.seed`). | `non_repairable_rbd.py:458`, `repairable_rbd.py:availability`, `standby_node.py`, `repeated_standby_node.py:47` | Thread an optional `rng: np.random.Generator | int | None` through the simulation APIs; default to `np.random.default_rng()`. Return/accept confidence intervals (see F6). |
| H3 | **`type(x) == Class` identity checks** break for subclasses and trip flake8 E721. | `non_repairable.py:21,24,30` | Use `isinstance`. |
| H4 | **Documented `sf()` parameters that don't exist.** `working_components`/`broken_components` are described in the `sf()` docstring but are not parameters, and the consistency check is commented out. `rbd_args_check.py` is 0% covered / effectively dead. | `non_repairable_rbd.py:204–251`; `rbd_args_check.py` (0% cov) | Either implement component-level masking (the docstring promises it) or remove the stale docstring and dead module. Pick one and make code and docs agree. |
| H5 | **Fragile string-based dispatch on surpyval internals.** Logic branches on `dist.name in ["FixedEventProbability","Bernoulli"]`, `dist.name == "Weibull"`, and reaches into `dist.params[1]`, `dist.offset/zi/lfp`. A surpyval rename silently breaks behaviour. | `non_repairable_rbd.py:175–185,503`; `non_repairable.py:97–104`; `standby_node.py:20` | Introduce a small capability protocol (e.g. `is_fixed_probability(model)`, `is_analytic(model)`) or check types, not name strings. Add a compatibility test against the pinned surpyval. |
| H6 | **Cross-instance access to a name-mangled attribute.** `NonRepairableRBD.__init__` reads `node.__fixed_probs` on *other* `NonRepairableRBD` instances; this only works due to same-class name mangling and is brittle. | `non_repairable_rbd.py:163,187` | Expose a public/underscore property (`is_fixed`) instead of the dunder. |

### 4.3 Medium — API design & usability

| # | Finding | Evidence | Recommendation |
|---|---------|----------|----------------|
| M1 | **Inconsistent public API.** The flagship `NonRepairableRBD` is **not** exported from the top-level package — only `RBD` and `RepairableRBD` are. `StandbyModel`, `NonRepairable`, `RepeatedNode`, etc. also require deep imports. | `__init__.py`; `from repyability import NonRepairableRBD` → ImportError | Curate `__init__.py` with an explicit `__all__` exporting the intended public surface. |
| M2 | **Scalar-in → array-out.** `sf(50)` returns `array([…])`; under numpy 2.0 `float(result)` on a length-1 array raises. Ergonomically awkward for the "student" audience. | smoke test: `float(rbd.birnbaum_importance(50)[1])` raises | Return a scalar when the time input is scalar (or provide `.item()`-friendly helpers), consistently across `sf`/`ff`/importances. |
| M3 | **No system-level density or hazard.** Only `sf`/`ff` are exposed; there is no `df` (pdf) or `hf`/`Hf` (hazard) for the system. | `non_repairable_rbd.py` (no `df`/`hf`) | Add system `df`, `hf`, `Hf` (numerically differentiate `sf`, or convolve node densities), completing the standard reliability function set. |
| M4 | **Mutable default arguments** throughout (`k={}`, `working_nodes=[]`, `broken_nodes=[]`, `fixed=[]`). | `rbd.py:543`; `non_repairable_rbd.py:51,208,209`; `repairable_rbd.py` | Use `None` sentinels. |
| M5 | **Inconsistent method surfaces across classes.** e.g. importances take `x` on `NonRepairableRBD` but no argument on `RepairableRBD`; allocation lives on `RBD` but isn't surfaced coherently on subclasses. | `repairable_rbd.py:539` vs `non_repairable_rbd.py:511` | Define a documented, consistent API contract per class; consider an abstract base declaring the shared surface. |

### 4.4 Low — polish

- **L1 — Spelling in user-facing text/identifiers:** "Strucutral"/"strucuture"
  (`rbd.py:292`, `rbd_graph.py`), "occuring", **"Fussel-Vesely" (correct:
  Fussell–Vesely)** including the public method name `fussel_vesely`. Fixing the
  method name is a (deprecatable) API change — worth doing before 1.0.
- **L2 — Docstring `Raises: TODO` placeholders** and copy-pasted parameter docs
  (e.g. `improvement_potential` documents `x` but takes `node_probabilities`):
  `rbd.py:824`, `repairable_rbd.py:635`, `non_repairable_rbd.py:654`.
- **L3 — flake8 `per-file-ignores` in `pyproject.toml` not honoured** (F401 still
  flagged); pre-commit passes it via CLI but CI relies on the file. Align them.
- **L4 — mypy is lax:** many `[annotation-unchecked]` notes; add
  `--check-untyped-defs` and gradually type the decorators (`check_x`,
  `check_probability`) which currently erase signatures.

### 4.5 CI / infrastructure

- **CI runs a single Python (3.10)**; deps require numpy≥2 (Py≥3.9). Add a
  matrix (3.9–3.13). Upgrade the pinned actions (`checkout@v3`,
  `setup-python@v4`, `upload-artifact@v3` are all a major version behind).
- **No coverage gate** and no Codecov upload. Add a threshold (e.g. `--cov-fail-
  under=90`) and a coverage badge.
- **Slow suite (~35–60 s), dominated by Monte-Carlo standby tests** (one is
  10.8 s). Add `@pytest.mark.slow`, run fast tests by default, and adopt
  `pytest-xdist` for parallelism.
- **No CHANGELOG, CONTRIBUTING, issue/PR templates, or release process.**

---

## 5. Feature roadmap (what the RBD should have)

The package is already complete on one axis — **exact binary-state
reliability + availability + importance**. The gaps below are specific
capabilities, grouped into tiers by value-to-effort for the engine's role
(the computational backend behind the Reliafy app; fitting in surpyval; no
plotting). ✅ = shipped this branch.

**Tier 1 — foundational (backend + ergonomics)**
- ✅ **Serialisation.** `to_dict`/`from_dict`/`to_json`/`from_json` — an RBD
  round-trips to JSON so Reliafy can save, load, share and version diagrams.
- ✅ **Inverse reliability / BX-life / design life.**
  `time_to_reliability(target)`, `bx_life(x)`.

**Tier 2 — canonical RBD capabilities**
- **Warm & hot standby** — the package has *cold* standby only. Warm (spares
  age at a reduced rate) is the real gap; hot is parallel.
- **Load-sharing** — dependent-failure redundancy: survivors carry more load
  and age faster. See §9 — this is one of the two flagship threads and folds
  into the condition-based engine via surpyval's AFT models.
- **Redundancy Allocation Problem (RAP)** — choose integer redundancy under a
  cost/weight budget to hit a reliability target. Distinct from the existing
  *reliability*-allocation methods.
- **Structural importance** (probability-free) and **parameter sensitivity**
  (d(reliability)/d(Weibull shape/scale)) — round out the importance suite.

**Tier 3 — dependence & structure generalisations (specialist / larger)**
- **Common-Cause Failures (CCF)** — beta-factor / MGL / alpha-factor. Breaks
  the independence assumption; essential for safety-critical users.
- **Phased-mission systems** — the diagram changes per mission phase.
- **Multi-state / capacity (flow) systems** — degraded/derated states and
  throughput, not just working/failed.
- **Undirected network reliability** — s-t reliability on general graphs.
- **Epistemic uncertainty propagation** — distributions on node *parameters* →
  system-reliability *bounds* (distinct from the aleatory Monte-Carlo).

**Companion models & scale**
- **Fault Tree Analysis (FTA)** — AND/OR/VOTE gates, top-event probability,
  RBD↔FTA conversion; reuses the cut-set / Shannon engine.
- **BDD/ZBDD engine** for very large systems (only when scale bites).
- **Maintenance & spares optimisation (RCM)** — block replacement, inspection
  intervals, condition-based policies, spares stocking (the vestigial
  `Repairable` class is the seed).

**Flagship threads (detailed in §8–§9)**
- **★ Condition-based ("digital twin") reliability** — condition each
  component on its *current life/state* (fed from sensors) → live system
  reliability, remaining useful life, and state-dependent importances. Mostly
  composable from existing primitives; the keystone the sensor/Reliafy vision
  needs, and the substrate load-sharing sits on.
- **★ Load-sharing via AFT** — dependent redundancy through surpyval's
  accelerated-failure-time models (cumulative-exposure time-scaling).

_Removed from scope:_ visualization (Reliafy) and data-fitting (surpyval).

---

## 6. Phased development plan

The phases are ordered so that each unblocks the next. Phase 0 is a hard
prerequisite for meaningful adoption; Phases 1–2 make the project contributor-
and user-ready; Phases 3+ are feature growth.

### Phase 0 — Foundation & hygiene — ✅ done
Packaging, reconciled deps, gating CI + version matrix + coverage gate,
release workflow, community files, and the quick code cleanups (debug print,
mutable defaults, `type ==`). **(C1–C3, H1, H3, M4, §4.5)**

### Phase 1 — Correctness & API consistency — ✅ done (and extended)
Public API/`__all__`, seedable RNG, dead-code removal, capability-based
dispatch + compat test, `df`/`hf`/`Hf`, Fussell–Vesely rename, docstring
fixes — plus the full cross-class API standardisation and the
working/broken-node bug fixes. **(H2, H4–H6, M1–M3, M5, L1–L4)**

### Phase 2 — Documentation & examples — ✅ done
MkDocs-Material + mkdocstrings site (overview, guide, API reference), strict
docs build in CI, README quickstart/links. _(Remaining nice-to-have: a
hosted GitHub Pages/RTD deploy and doctest-backed examples.)_

### Phase 3 — High-value features — ◑ partly done
- [x] **Monte-Carlo hardening** — seeds, standard errors + Wilson bands, MTTF
      CI, transient availability validated against exact Markov solutions.
- [x] **Serialisation** and **inverse-reliability** (moved up from later tiers).
- [ ] Convergence diagnostics / variance reduction / parallel replications
      (the remaining MC-hardening tail).

### Phase 4 — Condition-based reliability & the dependent-failure engine _(next; see §8–§9)_
The keystone sequence — each step reuses the last:
- [ ] **Condition-based ("digital twin") evaluation layer** — a per-node
      `state` (age/exposure, alive/failed, optional current load) feeding
      `sf_given_state`, `remaining_life` (RUL), and `importances_given_state`.
      Reuses `conditional_survival` + `system_probability` + the
      `time_to_reliability` root-find. **(§8)**
- [ ] **AFT exposure model** — carry a surpyval AFT model on a node; effective
      age = accumulated exposure, current load → acceleration `φ(load)`. **(§9)**
- [ ] **Load-sharing node** — the self-loading special case (load = `L /
      survivors`), exact for identical-exponential, MC+KM otherwise. **(§9)**

### Phase 5 — New analysis capabilities _(iterative)_
- [ ] **Fault Tree Analysis** with RBD↔FTA conversion.
- [ ] **Redundancy Allocation Problem (RAP)**.
- [ ] **Warm standby**; **structural importance** & **parameter sensitivity**.
- [ ] **Epistemic uncertainty propagation** over node parameters.

### Phase 6 — Advanced / specialist _(as demand warrants)_
- [ ] Common-cause failures; phased-mission; multi-state/capacity; network
      reliability; Markov/state-space; reliability growth & demonstration
      testing; maintenance/spares optimisation; BDD engine for scale.

---

## 7. Suggested first PRs (quick wins) — ✅ all shipped

These low-risk, high-signal changes have all landed on this branch:

1. **Packaging PR** — add `[project]`/`[build-system]`; makes `pip install .`
   real. _(C1)_
2. **CI enforcement PR** — `--check` on formatters, add version matrix, bump
   actions. _(C3)_
3. **Cleanup PR** — remove `print(out)`, fix `type ==` → `isinstance`, replace
   mutable defaults. _(H1, H3, M4)_
4. **Public API PR** — `__all__` and top-level exports incl. `NonRepairableRBD`.
   _(M1)_
5. **README quickstart PR** — a runnable end-to-end example so the landing page
   shows the library in action. _(Phase 2 down payment)_

---

## 8. Flagship thread: condition-based ("digital twin") reliability

**Vision.** A user inputs the *current life/state* of each component — ideally
streamed from sensors — and gets the system's reliability **given the current
state**, updating as the state (and loading) changes. This is the
condition-based / prognostics layer, and it is the single highest-leverage
capability for the sensor + Reliafy story.

**It is mostly already in the engine.** Condition each component on its own
current life `Xᵢ` and propagate through the exact system engine:

```
Rᵢ(x | Xᵢ) = Rᵢ(Xᵢ + x) / Rᵢ(Xᵢ)        # per-component conditional survival
R_sys(x | {Xᵢ}) = Φ( {Rᵢ(x | Xᵢ)} )      # feed into system_probability()
```

Failed → 0 (`broken_nodes`), as-new → `Xᵢ = 0`, aged → its (lower) conditional
curve. This is the **heterogeneous generalisation of the existing `cs`**: `cs`
conditions on *system* survival to a single age (marginalising over which
components are alive); this conditions on the **known state of every
component**, which is what telemetry gives you and is far more informative. A
proof-of-concept composed from existing primitives (`conditional_survival` +
`system_probability` + the `time_to_reliability` root-find) already produces
correct conditional reliability and RUL that collapse appropriately as
redundancy is worn down.

**Outputs it unlocks (same machinery):**
- **Live forward reliability** given the current fleet state.
- **Remaining Useful Life (RUL)** — `time_to_reliability` on the *conditional*
  curve: "how long until the system drops below its target, given where every
  part is right now."
- **State-dependent importances** — Birnbaum/criticality re-evaluated at the
  current state: *which component is the weak link right now?* — a live
  maintenance-priority signal, not a design-time average.

**Streaming shape.** Structure is static (persist once — this is what the new
serialisation is for); state is streamed (per-component age + load per update).
Each update is a cheap, *exact* re-evaluation (sub-millisecond for realistic
systems): hold the RBD, push component states, read `{reliability, RUL,
criticality}`, Reliafy renders. A live reliability digital twin.

**Proposed surface.** A per-node `state` (age/exposure, alive/failed, optional
current load) plus `sf_given_state(x, state)`, `remaining_life(target, state)`,
and `importances_given_state(state)`.

**Scope note.** Clean for ordinary distribution components (repeated components
share one state — they are one physical unit). Dynamic nodes (standby) carry
internal sequence-state that is messier to condition; scope v1 to ordinary
components.

---

## 9. Flagship thread: load-sharing via AFT (and the unification)

**What it is.** *n* active units split a total load; when one fails the
survivors carry more and **age faster** (dependent failure). Works while ≥ `k`
survive. Distinct from parallel (independent) and cold standby (sequential).

**Architecture.** It must be a **single dynamic node** (the coupled group
cannot be *n* independent RBD nodes, which the exact engine assumes) — exactly
the pattern `StandbyModel` already uses. So it is a new node model
(`LoadSharingModel`), not a change to the structural engine.

**Each unit carries a surpyval AFT model.** surpyval's accelerated-failure-time
model computes `H(x | Z) = H₀(φ(Z)·x)` — literally the Nelson cumulative-
exposure / time-scale model, which is *exactly* what load-sharing needs. So the
acceleration is fit from data (not an ad-hoc knob), any life-stress law
(power/Arrhenius/Eyring/…) is supported, and `φ(Z)` is a direct call
(`model.phi(Z, *phi_params)`). This handles a survivor's load jumping mid-life
correctly for any baseline distribution — the subtlety a naive implementation
gets wrong.

**Engine (exact cumulative exposure), reusing the standby three-way split:**

```
draw baseline threshold  τ_i = M_i.dist.qf(U_i)        # baseline life, φ = 1
e_i = 0; t = 0; survivors = all n
while len(survivors) >= k:
    Z   = per-unit load for this survivor count         # e.g. L / j
    φ_i = M_i.phi(Z)                                     # direct call
    dt  = min_i (τ_i − e_i) / φ_i;   t += dt             # real time to next failure
    e_i += φ_i · dt   for survivors;   drop the failer   # load redistributes
system lifetime = t                                     # at the (n−k+1)-th failure
```

Exact closed form for identical-exponential baselines (each phase is
`Exp(j·λ₀·φ_j)` → hypoexponential, generalising the Erlang cold-standby
result with `φ ≡ 1`); Monte-Carlo + Kaplan-Meier otherwise.

**The unification — build §8 first.** "Current life" is really *accumulated
exposure*, and current load sets the go-forward `φ`. With an AFT node,
`R(x | eᵢ, load ℓᵢ) = R₀(eᵢ + φ(ℓᵢ)·x) / R₀(eᵢ)` — the *same* conditional-
survival formula as §8. So condition-based reliability and load-sharing are
**one engine**, differing only in where the load comes from: **fed from
sensors** (digital twin) vs. **computed from the group's survivors** (load-
sharing). Building the §8 conditional-state layer first makes load-sharing the
self-loading special case.

**Open decisions.**
1. Load→covariate mapping — default "covariate = per-unit load", user supplies
   total load `L` (so `Z_j = L/j`) and, if needed, a `load → Z` transform.
2. AFT serialisation — reconstruct a fitted AFT as `(baseline dist, life-stress
   model, params)`; verify surpyval's reconstruction API before committing.
3. Scope: non-repairable first; equal load share; one shared or per-unit AFT.

_(surpyval also has Cox PH — the hazard-scaling sibling, cumulative-hazard
clock — available if stress should bend the hazard shape rather than rescale
time. AFT is the right lead for load-sharing.)_

---

_Review notes (original): 202/202 tests pass (Py 3.11, numpy 2.x,
surpyval 0.11.1); line coverage 91%. Findings in §4 reference `master` at the
time of the initial review and are now resolved (§1a). Current state: 287 tests,
~90% coverage._
