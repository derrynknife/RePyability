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

## 5. Recommendations for additional features

Ordered roughly by value-to-effort for the stated audience (students →
professionals). Items marked ★ are the highest-impact.

1. **★ Visualization.** Draw the RBD (graphviz/networkx layout) and plot
   `R(t)`, `F(t)`, `h(t)`, and availability curves via an optional matplotlib
   backend. This is the biggest single usability win for teaching and for
   communicating results, and it reuses the existing graph and `sf` machinery.
2. **★ Fault Tree Analysis (FTA).** AND/OR/k-of-n(VOTE) gates, minimal cut sets,
   top-event probability, and RBD↔FTA conversion. It reuses the existing
   cut-set and Shannon-decomposition engine and is the natural companion model
   to RBDs.
3. **★ Monte-Carlo engine hardening.** Seedable `Generator`, confidence
   intervals and convergence diagnostics on availability/MTTF, variance
   reduction, and parallel replications. Turns the simulator from
   "point estimate" into "estimate ± CI."
4. **Data → model workflow.** Helpers to fit surpyval distributions from field
   failure data (CSV/pandas) and drop them straight into RBD nodes — a
   "from failure data to system reliability" story that showcases the surpyval
   integration.
5. **Maintenance & spares optimization (RCM).** Extend the single-component
   age-replacement model to block replacement, inspection intervals,
   condition-based policies, and spare-parts stocking — with system-level cost
   optimization on top of the RBD.
6. **Redundancy Allocation Problem (RAP).** Given per-component cost/reliability,
   optimize redundancy to meet a system target — a direct, high-value extension
   of the existing allocation methods.
7. **Uncertainty propagation.** Put distributions on node parameters and
   propagate to system reliability (the array-vectorised `system_probability`
   already supports Monte-Carlo over parameter draws).
8. **Common-Cause Failures (CCF).** Beta-factor / MGL models — essential for
   safety-critical (nuclear, aerospace, process) users.
9. **Markov / state-space models** for repairable systems with dependencies
   (shared repair crews, switching failures) beyond what simulation covers
   analytically.
10. **Reliability growth** (Crow-AMSAA/NHPP, Duane) and **reliability
    demonstration test** planning.
11. **Serialization / interchange.** Save/load an RBD to JSON/YAML; optional
    import/export with other tools (OpenFTA, etc.).
12. **Advanced structures:** phased-mission analysis, load-sharing, and
    degradation/PoF models.

---

## 6. Phased development plan

The phases are ordered so that each unblocks the next. Phase 0 is a hard
prerequisite for meaningful adoption; Phases 1–2 make the project contributor-
and user-ready; Phases 3+ are feature growth.

### Phase 0 — Foundation & hygiene _(prerequisite; ~1–2 weeks)_
**Goal: the package installs, builds, and enforces its own quality bars.**
- [ ] Add PEP 621 `[project]` + `[build-system]` to `pyproject.toml`; dynamic
      version; `dev`/`docs` extras; classifiers; Python `>=3.9`. **(C1)**
- [ ] Reconcile requirements files; pin dev tools; stop installing surpyval from
      git HEAD in dev. **(C2)**
- [ ] Make CI gate on `black --check`, `isort --check`, `flake8`, `mypy`; add a
      Python version matrix and refresh action versions; add a coverage
      threshold. **(C3, §4.5)**
- [ ] Set up a release workflow (tag → build → publish to PyPI via trusted
      publishing) and add `CHANGELOG.md`, `CONTRIBUTING.md`, PR/issue templates.
- [ ] Delete the debug `print` and fix the mutable-default and `type ==` issues
      (fast, low-risk). **(H1, H3, M4)**

### Phase 1 — Correctness & API consistency _(~1–2 weeks)_
**Goal: no papercuts; a coherent, discoverable public API.**
- [ ] Curate `__init__.py` / `__all__`; export `NonRepairableRBD` et al. **(M1)**
- [ ] Add seedable RNG to all simulation entry points. **(H2)**
- [ ] Resolve the components-vs-nodes masking promise (implement or remove);
      delete or cover `rbd_args_check.py`. **(H4)**
- [ ] Replace surpyval string dispatch with capability checks + a compat test;
      remove cross-instance dunder access. **(H5, H6)**
- [ ] Scalar-in/scalar-out ergonomics; add system `df`/`hf`/`Hf`. **(M2, M3)**
- [ ] Fix docstring TODOs and the Fussell–Vesely naming (with deprecation
      shim). **(L1, L2)**

### Phase 2 — Documentation & examples _(~2–3 weeks)_
**Goal: a student can go from install to a solved system in 10 minutes.**
- [ ] Stand up a docs site (MkDocs-Material or Sphinx) with autodoc/API
      reference, hosted on GitHub Pages/Read the Docs.
- [ ] Write a "Getting started" tutorial and a **worked example gallery**
      (series/parallel, k-of-n, standby, repairable availability, importances)
      as runnable notebooks — reusing the reference systems already in
      `conftest.py`.
- [ ] Expand the README with a quickstart, feature list, and links; add
      theory/reference notes and citations for each method.
- [ ] Add doctest-backed examples to key public methods.

### Phase 3 — High-value features _(iterative)_
- [ ] **Visualization** module (RBD drawing + reliability/availability plots).
      ★ Do this first — it also makes the docs gallery far better.
- [ ] **Monte-Carlo hardening** (CIs, convergence, parallelism).
- [ ] **Data→model** helpers (fit from failure data → RBD nodes).

### Phase 4 — New analysis capabilities _(iterative)_
- [ ] **Fault Tree Analysis** with RBD↔FTA conversion.
- [ ] **Redundancy Allocation** optimization.
- [ ] **Uncertainty propagation** over node parameters.

### Phase 5 — Advanced / specialist _(as demand warrants)_
- [ ] Common-cause failures; Markov/state-space; reliability growth &
      demonstration testing; phased-mission / load-sharing / degradation.

---

## 7. Suggested first PRs (quick wins, ≤ a day each)

These are low-risk, high-signal changes that immediately improve the project
and can land before the larger phases:

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

_Review notes: 202/202 tests pass (Py 3.11, numpy 2.x, surpyval 0.11.1); line
coverage 91% (`repairable.py` and `rbd_args_check.py` at 0%). Findings reference
current `master` at the time of review._
