# 3D FEM Dataset Generation Pipeline — Consolidated Specification (Final)

**Status:** this document is the single authoritative architectural spec for
`Training_pipeline`'s 3D dataset-generation stage. It supersedes and replaces
two earlier documents:
- ~~`implementation_plan_corrections.md`~~ (corrections pass on the original plan)
- ~~`3d_dataset_generation_prompt.md`~~ (original base coding-agent prompt)

Both are now folded into this document as a single, non-contradictory spec.
Do not work from either superseded file. **`opus_task_spec_load_config_corrections.md`
remains the separate execution layer** — it defines the ordered, verifiable
tasks (fillet-radius implementation, leakage-safe splitting, the Peterson's-
chart calibration suite, etc.) that bring an existing partial implementation
into compliance with this spec. Read this document for *what the pipeline
should be*; read the task spec for *what to do next, in what order, and how
to verify it*.

---

## 0. Six Resolved Design Decisions

These were found to be inconsistent or conflicting across the two superseded
documents. Each is now a single source-of-truth decision, applied
consistently everywhere below rather than patched locally:

| # | Decision | Resolution |
|---|---|---|
| 1 | Point loads | **Retracted entirely.** Surface traction only, project-wide. A point load is, like a sharp reentrant corner, a classic cause of a non-convergent FEA singularity (ADR-005). `solve/loads.py` implements traction application only — no point-load code path exists anywhere in this pipeline. |
| 2 | Calibration-pass intent | Calibration validates that the **safety-factor scaling machinery works correctly** (sane scale factors, correct cap behavior, intended SF spread achieved) — it does **not** check "stress stays below yield," since SF ≤ 1 samples are intentional training signal, not a violation. |
| 3 | Scale-factor denominator | **P95 von Mises**, never literal peak, in the `scale_factor` formula everywhere it appears. Peak stress at a reentrant corner or notch is mesh-sensitive and can suppress the scale factor toward zero — the exact failure mode P95 exists to avoid. |
| 4 | Split-strategy grouping key | `base_sample_id` (not the older "shape + load variants" phrasing). All safety-factor variants of one base sample must land in the same train/val/test split, composed in a defined order with the per-stratum floor and the scale-extrapolation holdout (Section 12). |
| 5 | Mesh sizing near curved features | The global `mesh_size = 0.05 × characteristic_length` rule is necessary but not sufficient. A local override applies near any filleted/curved feature (Section 4). |
| 6 | `max_displacement_fraction_of_length` | **0.05**, for the bulk-solid family class. (The bending-family class uses an unrelated, thickness-relative parameter, default 0.5 — see Section 12; the two were never actually in conflict, only the bulk-class value had drifted between documents.) |

---

## 1. Overview & Non-Negotiable Constraints

A from-scratch 3D pipeline built directly on Gmsh (geometry + meshing) and
FEniCS/Dolfin (solving) — not the `main_package`/MGN-Public library, which is
confirmed 2D-only. Timing validated on Google Colab: ~1.1–1.2 sec/sample
steady-state (post JIT warm-up) at ~1,600 nodes / ~6,400 tets.

**Purpose:** generate training data for a physics-informed GNN:

```
Node features (position, BC flags, load flags)
  -> Encoder (node/edge/global MLPs)
  -> Processor (K rounds of message passing)
  -> Decoder A: displacement (learned)
  -> Strain: deterministic B-matrix from displacement, per-element-type dispatch
  -> Decoder B: stress from strain + material (learned, strain DETACHED from
     displacement-predicting weights)
  -> Von Mises + safety factor: exact algebraic post-processing (NOT learned)
```

Two things the dataset must deliver beyond basic correctness:
1. **Scale robustness** — exposure to meaningfully larger/smaller objects and
   denser meshes, not one fixed absolute size regime.
2. **Shape generalization** — multiple distinct geometry families, so
   train/val/test splits test shape generalization, not interpolation within
   one parametric template.

**No-Monte-Carlo constraint (non-negotiable):** every *base* sample is
exactly one FEM solve. Displacement, stress, strain, and reaction force for
that base sample all come from that same solve call — never averaged,
aggregated, or combined across randomized realizations. A base sample
legitimately produces multiple *output* files (one per target safety factor,
Section 10) via exact linear scaling of that one solve's fields — this is
post-solve scaling of a single coherent solution, not aggregation, and does
not weaken this constraint.

This is a deliberate departure from stochastic-aggregate approaches used by
some published FEM-surrogate datasets (e.g. CMU SFEM, which aggregates 50
Monte Carlo point-load realizations per geometry via per-node percentile
statistics, in service of predicting a converged *distributional* stress
field for uncertainty quantification). SFEM's design is correct for its own
objective; it is rejected here because that objective is incompatible with
this project's deterministic, single-solve, internally-coherent-field
requirement — not because SFEM's methodology is flawed in general. (SFEM's
material properties, E≈2.3 GPa / ν≈0.40, are also far outside structural-steel
range, an independent reason it couldn't be reused regardless.) See ADR-006.

**Toolchain:**
- Geometry + meshing: Gmsh Python API, OCC kernel (`gmsh.model.occ`) — 3D
  solids (box, cylinder, boolean union/cut/fragment), tetrahedral mesh
  generation (`gmsh.model.mesh.generate(3)`)
- Solving: FEniCS/Dolfin (legacy, via `fem-on-colab`), linear elasticity
  variational form, `VectorFunctionSpace` degree 1
- Mesh interchange: `meshio`, Gmsh `.msh` → Dolfin-readable XDMF
- No `torch`, `torch-geometric`, or `main_package`/MGN-Public — this pipeline
  is data generation only

---

## 2. File Structure

```
dataset_gen_3d/
├── README.md
├── requirements.txt              # gmsh, meshio, numpy, pandas, networkx, pyyaml, h5py — NOT torch
├── config.yaml                   # every tunable parameter, see Section 16
├── corrections_progress.json     # task-level status tracking (see task spec)
├── geometry/
│   ├── __init__.py
│   ├── families/
│   │   ├── __init__.py
│   │   ├── block_with_holes.py       # box + 1-4 cylindrical through-holes
│   │   ├── l_bracket.py               # box with a FILLETED rectangular notch (non-convex); see Section 3.2
│   │   ├── elongated_bar.py           # high aspect-ratio bar, optional holes
│   │   ├── plate_3d.py                # thin, wide/long block (low height:width ratio)
│   │   └── block_with_fillet.py       # box with a filleted/rounded edge or corner
│   └── scale_sampler.py               # samples overall bounding-box scale, see Section 4
├── solve/
│   ├── __init__.py
│   ├── mesh_conversion.py             # Gmsh .msh -> meshio -> Dolfin XDMF
│   ├── material.py                    # E, nu, yield strength (full 3D, no plane assumptions)
│   ├── boundary_conditions.py         # configurable fixed-face selection
│   ├── loads.py                       # SURFACE TRACTION ONLY (ADR-005: point loads banned, singularity risk)
│   ├── run_fem_3d.py                  # one FEniCS solve per call, returns u, stress, reaction
│   ├── load_scaling.py                # pure linear-algebra scaling of one solve -> N safety-factor variants; no FEniCS dependency, locally testable
│   └── reaction_forces.py             # extract reaction forces at constrained DOFs
├── validation/
│   ├── __init__.py
│   ├── coherence_check.py             # tetra B-matrix -> Hooke's law -> compare to solver stress
│   ├── b_matrix_tetra.py              # linear tetrahedron shape-function-derivative strain
│   ├── receptive_field_check.py       # graph diameter computation (networkx)
│   ├── scale_distribution_check.py    # confirms dataset spans intended size range + per-stratum/per-split counts
│   └── calibration/
│       ├── percentile_calibration.py  # thin-case + plane-strain-BC + Kt-vs-z checks (Section 11)
│       └── linearity_spot_check.py    # independent re-solve verification (Section 11)
├── pipeline/
│   ├── __init__.py
│   ├── sample_spec.py                  # dataclass: family + scale + params + load direction/location + BC + mesh_size (a base config; SF sweep is post-solve, Section 10)
│   ├── generate_dataset.py             # main orchestrator
│   ├── split_strategy.py               # stratified split by family, scale_bucket, AND base_sample_id (Section 12)
│   └── resume.py
├── output/
│   ├── geometries/                     # generated .step files
│   ├── raw/                            # per-sample node-level HDF5
│   └── manifest.csv                    # every param + pass/fail flag + timing + scale_bucket + base_sample_id + target_safety_factor per sample
├── logs/
│   ├── generation.log
│   └── calibration.log                 # required calibration run, Section 11
└── tests/
    ├── test_geometry_families.py       # each family produces valid, non-degenerate solids; l_bracket never produces r=0
    ├── test_coherence_check.py         # validated against a known analytical case (uniaxial tension block)
    ├── test_scale_range.py             # confirms samples actually span configured size range
    ├── test_receptive_field_distribution.py
    ├── test_split_leakage.py           # no base_sample_id split across train/val/test
    └── test_pipeline_end_to_end.py     # 5-10 samples across all families + scale buckets
```

---

## 3. Geometry Families

Implement **5 distinct basic families**, randomly interleaved per sample (not
sequentially blocked), so train/val/test splits stratify cleanly. Record
`geometry_family` per sample in the manifest.

### 3.1 Block With Holes
Rectangular box, 1–4 cylindrical through-holes, randomized position/radius/
axis orientation (need not all pierce the same face).

### 3.2 L-Bracket / Notched Block
Box with a rectangular notch cut from one edge → non-convex solid. Tests
reentrant-corner stress concentration.

**The notch root carries a randomized, non-zero fillet radius.** A sharp
(zero-radius) internal corner is a genuine mathematical singularity in linear
elasticity — peak stress grows without bound under mesh refinement and never
converges. This is both non-physical (every manufactured part has some
corner radius from machining/casting/forming) and numerically invalid for a
dataset that needs mesh-convergent samples.

- Parameterize as `fillet_radius`, expressed as ratio `r/d` (radius /
  relevant section depth), matching the Peterson's/Shigley's chart
  convention (Table A-15).
- Sample `r/d` uniformly from `[0.02, 0.3]` by default. Below ~0.02 the
  geometry re-approaches singular, mesh-sensitive behavior; above ~0.3 the
  stress concentration flattens and the family stops meaningfully testing
  concentration behavior.
- Reserve a held-out sub-range for test only, e.g. train on
  `r/d ∈ [0.02,0.10] ∪ [0.20,0.30]`, test on `r/d ∈ (0.10,0.20)` — an
  explicit extrapolation-along-concentration-severity test, independent of
  the scale-bucket holdout (Section 12).
- The geometry remains concave/reentrant at the notch — this still tests
  reentrant-corner-driven stress concentration, it is simply no longer
  singular.
- Record `fillet_radius` and `radius_ratio` per sample in the manifest.
- Reject (don't silently pass) any sample where mesh generation at the
  fillet fails or produces inverted/degenerate elements.

### 3.3 Elongated Bar
High aspect ratio (length:cross-section ≥ 8:1), with or without holes.
**Mandatory** — exists specifically to guarantee high graph-diameter samples
(Section 6). Directly precedented by M4GN's DeformingBeam/DeformingBeam-Large
family, which exists for the same reason (maximizing graph diameter to
stress-test long-range coupling in message-passing GNNs).

### 3.4 Thin Plate-like Block
Low height:width:length ratio (height << width, length), optionally with
holes. Tests thin-section, sheet-metal-like behavior.

### 3.5 Block With Fillet
Box with a filleted/rounded edge or corner (smooth curvature, not sharp).
Tests generalization to curved-boundary geometry vs. planar-faced solids.

---

## 4. Scale Robustness

`scale_sampler.py`:
- Samples an overall bounding-box scale factor per sample from a wide,
  explicitly configured range (e.g. ~20mm to ~2m+ characteristic length —
  bracket plausible small-bracket-to-large-panel component sizes; exact
  bounds in `config.yaml`).
- Applies the scale factor to whichever family was selected (scale is a
  modifier on top of family + shape params, not a separate family).
- Buckets samples into discrete scale bins (small/medium/large) for
  stratification. At least one bucket (the largest) is deliberately
  held out / under-represented in training and over-represented in test,
  specifically to probe scale-extrapolation.

**Mesh sizing — two rules, not one:**
1. **Global rule:** `mesh_size = 0.05 * characteristic_length`, so relative
   mesh resolution stays comparable across the scale range. Log both the
   absolute `mesh_size` and resulting node/element count per sample.
2. **Local override near curved/filleted features** (L-Bracket's notch
   fillet, Block-With-Fillet's rounded edge): the global rule alone is
   insufficient once a fillet radius gets small. At `r/d` near the low end
   of its range (~0.02), `0.05 × characteristic_length` will typically be
   far too coarse to place an adequate number of elements around the curve
   to resolve its stress gradient (standard FEA practice wants at least 4–6
   elements around a curved feature). Apply, via Gmsh local refinement
   fields (distance + threshold field around the fillet edge/surface):
   ```
   local_mesh_size = min(global_mesh_size, fillet_radius / 4)
   ```
   Without this, small-`r/d` L-Bracket samples will be geometrically correct
   but numerically under-resolved — which directly invalidates the
   Peterson's-chart calibration in Section 11 regardless of how correct the
   analytical setup is otherwise.

Log actual node/element counts per sample — larger objects at proportional
mesh density naturally produce larger graphs; this is intentional (Section
5), not something to correct for.

`validation/scale_distribution_check.py`: after generation, confirm the
dataset actually spans the configured characteristic-length and node-count
range (not clustered near one value from a sampling bug) before considering
generation complete. Extend this to also report per-stratum
(family × scale_bucket) and per-split base_sample_id counts (Section 12).

---

## 5. Object-Size → Mesh-Size → Graph-Size Interaction

- Larger objects, at proportional mesh density, produce larger graphs — this
  is expected and desired; the dataset needs a real spread of graph sizes,
  not a spread of physical dimensions with node count held artificially
  constant.
- Log and later analyze whether coherence-check pass rate or solve time
  degrades at the largest scale bucket — diagnostic information, not
  something to silently average away.
- Do not quietly cap mesh resolution to keep solve times uniform across
  scales. Log solve time per sample so any time/resolution trade-off is
  visible and can inform a later, explicit decision (e.g. excluding the most
  extreme scale bucket from *training* while keeping some for *testing*) if
  solve time becomes a real bottleneck.

---

## 6. Receptive Field / Graph Diameter

- Compute graph diameter (BFS hop count via `networkx`) per sample, log to
  manifest as `graph_diameter`.
- Elongated Bar + the large-scale bucket together should guarantee some
  samples with deliberately large diameter. Cross-check after generation
  that diameter correlates with expectation (elongated + large scale =
  largest diameters).
- The GNN processor stack is 15 message-passing layers; samples with
  diameter meaningfully exceeding this are the intended stress test of the
  architecture's global-broadcast mechanism (empirical test, post-training).

---

## 7. Node/Element-Level Output Schema

Full 6-component Voigt order `[xx, yy, zz, xy, yz, xz]`, this exact order,
every file.

| Column | Description |
|---|---|
| `node_id`, `x`, `y`, `z` | geometry |
| `is_fixed` | 1 = Dirichlet constrained on ≥1 DOF |
| `is_loaded` | 1 = nonzero applied traction at this node |
| `load_fx`, `load_fy`, `load_fz` | applied force components (0 if unloaded) |
| `disp_x`, `disp_y`, `disp_z` | displacement solution |
| `stress_xx`…`stress_xz` | 6-component Voigt stress |
| `strain_xx`…`strain_xz` | solver-reported strain — cross-check only, never a primary training target |
| `von_mises` | computed if not directly exported |
| `reaction_fx`, `reaction_fy`, `reaction_fz` | reaction force at constrained nodes (Section 8) |

**Multi-hot edge case (required):** some nodes must have both `is_fixed=1`
and `is_loaded=1` (a fixed face that also carries traction, or a pinned node
on the load face). Enforce this as an **automated manifest-level assertion**
at the end of generation — fail loudly if fewer than `min_multihot_samples`
(config, default 50) qualifying samples exist. A unit test on a small
fixture proves the logic *can* produce this case; only a manifest assertion
proves the full generation run actually *did*.

Element-level: tetrahedral connectivity `(E, 4)`, 0-based indexing; material
ID per element (constant across the dataset at this stage — single isotropic
material).

---

## 8. Reaction Forces

Implement explicitly in `solve/reaction_forces.py` — required for the
architecture's equilibrium regularization term
(`|predicted_internal_force - applied_external_force|`). Compute via
assembling the residual/test-function inner product at the constrained
boundary (e.g. `assemble(action(a, u_sol) - L)` restricted to boundary
DOFs). If this genuinely cannot be extracted cleanly from the legacy FEniCS
API within reasonable effort, document it as an explicit, named limitation
in README.md rather than silently omitting the columns. (Already flagged as
a risk area: legacy `dolfin` reaction-force extraction is API-fragile.)

---

## 9. Global (Graph-Level) Attributes

Per-sample manifest columns: `total_load_magnitude` (state explicitly
whether vector-sum or L2-norm-sum of per-node force vectors; document the
choice), `pinned_node_count`, `total_node_count`, `geometry_family`,
`scale_bucket`, `characteristic_length`, `base_sample_id`,
`target_safety_factor`.

---

## 10. Material and Load Configuration

**Material — SSAB Domex/Strenx 420** (single isotropic linear-elastic
material for this dataset stage):

| Property | Value | Status |
|---|---|---|
| E | 210 GPa | **CONFIRMED** — cross-checked against Eurocode 3 and the SSAB Domex 420MC datasheet |
| ν | 0.3 | **CONFIRMED** — as above |
| σ_yield | 420 MPa | **CONFIRMED** — as above |
| UTS | 480 MPa | **CONFIRMED** — as above |
| Representativeness of this grade for the actual Volvo component scope | — | **OPEN** — not yet confirmed against real component use case; see README open question |

These are two independent claims. Resolving the numeric/datasheet accuracy
(done) does not resolve whether Domex 420MC specifically is the right grade
for the real parts this surrogate targets (open) — document both, separately
dated, in README.md, and do not conflate them into one "material values
confirmed" statement.

**Loads — surface traction only.** Point loads are banned project-wide
(ADR-005): a point load, like a sharp reentrant corner, is a classic cause
of a non-convergent FEA singularity. `solve/loads.py` implements traction
application only — no point-load code path exists anywhere in this
pipeline, not even as a disabled/deprioritized option.

**Load strategy — reference solve + safety-factor scaling sweep.** Sampling
load magnitude from one fixed global range does not produce consistent
mild-to-failure coverage across the dataset: the load needed to reach yield
differs entirely between a small bracket and a large panel, and again
between a thin plate and a solid block. Fixed-range sampling makes
small/thin geometries fail catastrophically while large/thick geometries
barely feel any load — neither end is useful training signal.

Instead, because this is linear-elastic FEM (stress scales *exactly*
linearly with applied load):

1. For each base sample (one geometry + one BC + one load direction/
   location), run **one real FEM solve** at a small reference traction
   magnitude, safely elastic.
2. Read off **P95 von Mises** (never literal peak — see Section 0, decision
   3) from that solve.
3. Compute the exact scale factor to hit a target safety factor:
   ```
   scale_factor = (yield_strength / target_safety_factor) / reference_p95_von_mises
   ```
4. Generate `variants_per_base_sample` (default 5) output samples by
   multiplying displacement, strain, and stress by `scale_factor` — exact
   linear algebra, not an approximation; scaling a valid linear-elastic
   solution by a constant produces another valid linear-elastic solution,
   trivially preserving physical coherence.
5. Sample target safety factors across a defined mild-to-failure spread,
   e.g. `[10.0, 4.0, 2.0, 1.3, 0.8]` — very mild through at/beyond nominal
   yield. Configure the spread and variant count in `config.yaml`, never
   hardcode.

Implement the scaling logic in `solve/load_scaling.py` — pure linear
algebra, no FEniCS dependency, locally testable without Colab.
`sample_spec.py` represents the base configuration only; the safety-factor
sweep is a post-solve step in `generate_dataset.py`, producing one HDF5 file
per target safety factor per base solve.

**Two caveats, stated explicitly in README.md and any future write-up:**

1. **"Failure" here means a linear-elastic stress value exceeding yield —
   not a plasticity or large-deformation failure simulation.** The solver
   has no concept of yielding; a sample scaled past its yield point produces
   a stress value that, compared against yield strength, indicates the real
   material would have yielded. This is legitimate and useful training
   signal (the model should predict correct linear-elastic stress even
   near/past nominal yield, since that's exactly what a safety-factor check
   needs) — do not let this get overclaimed as nonlinear failure modeling.
2. **Small-deformation assumption must not be silently violated at high
   scale factors** — see Section 12's geometric-linearity gate.

**Peak-stress cap:** discard (don't silently keep) any sample whose scaled
stress exceeds `0.9 × UTS = 432 MPa`.

---

## 11. Calibration (Required Before Full Generation)

Calibration validates the **scaling machinery**, not "stress stays below
yield" — SF ≤ 1 samples are intentional (Section 10). Two tiers:

**Tier 1 — general pilot batch** (~20–30 base samples spanning the full
scale range and all 5 families):
1. Confirm the reference (small-magnitude) solve stays safely elastic, as
   intended.
2. Confirm the P95-based `scale_factor` computation produces sane, finite
   values across all families — not near-zero or unbounded, especially for
   L-Bracket.
3. Confirm the `peak_stress_cap` (432 MPa) correctly discards over-limit
   samples.
4. Confirm the resulting spread of actual safety factors, per family and
   per scale bucket, roughly matches the configured `target_safety_factors`
   spread.
5. Log everything to `logs/calibration.log`, separate from the main
   generation log.

**Tier 2 — L-Bracket-specific Peterson's-chart / through-thickness Kt
suite** (required before trusting Tier 1's result for this family; L-Bracket
is the family most exposed to singularity/mesh-sensitivity risk):

Peterson's/Shigley's charts (Table A-15: notched/filleted rectangular bar,
`Kt = σ_max/σ_nom` vs. `r/d`) assume no through-thickness stress variation —
a thin flat bar. A real 3D block does not satisfy this: published 3D FEM
studies show the peak `Kt` occurs at mid-plane only for thin plates
(converging to the 2D plane-stress chart value there); for thick plates the
true maximum shifts toward the **free surface** and can be 24–123% higher
than the free-surface value itself. Mid-plane stress in a thick block
instead approaches a generalized **plane-strain** solution, a different
number from the plane-stress chart value. A naive 1:1 comparison between the
production-thickness L-Bracket and the raw chart number is wrong by
construction.

1. **Thin-case check:** build a dedicated calibration-only thin L-Bracket
   variant (thickness/notch_depth ≤ 0.2) at 4–5 `r/d` values spanning the
   configured range. Extract `Kt = σ_max/σ_nom` at mid-thickness (`σ_nom` =
   mean stress on a net cross-section taken away from the notch, along the
   load axis — define this concretely in the manifest computation, not as a
   post-hoc analysis choice). Compare against Peterson's chart value at
   matching `r/d`, `w/d`. Tolerance ~15%, given the known residual 3D
   deviation even in the thin limit.
2. **Plane-strain-BC check:** using the normal production-thickness
   geometry, run one calibration variant per `r/d` with z-displacement
   constrained to zero on the front/back faces (forcing a genuine
   plane-strain state within the standard 3D solver/mesh path). Compare
   against a plane-strain reference value (not the plane-stress chart
   number).
3. **Through-thickness profile:** for a pilot batch of normal
   (unconstrained, production-thickness) L-Bracket samples, compute
   `peak_von_mises`, `p95`, `p90`, `p99`, and `Kt` at multiple through-
   thickness z-slices (minimum: mid-plane and near-surface). Plot `Kt` vs.
   `z`. It should qualitatively match the literature pattern (roughly flat,
   shifting toward a surface peak as relative thickness grows) — flag,
   don't just note, any profile that stays flat with no surface trend
   regardless of thickness, since that suggests a modeling problem, not
   real 3D behavior.
4. **P95 robustness check:** confirm whether `stress_percentile: 95` is
   validated as singularity-robust for L-Bracket specifically, given the new
   finite-radius geometry, or whether a per-family override is warranted.

Produce `calibration_percentile_report.md` covering all four checks, each
with an explicit pass/fail or confirmed/flagged status — no open-ended
"seems fine" statements. Do not change production `config.yaml` values based
on this tier without the report existing and being referenced in the commit.

**Tier 3 — empirical linear-scaling spot-check** (after Tiers 1–2 pass):
Phase-4 scaling (Section 10) assumes exact linearity holds at any target
load — true in exact linear elasticity, but could be silently violated by
solver tolerance settings or an accidental nonlinear flag on a subset of
samples. L-Bracket, near its (now finite but still tight) fillet root, is
the family most likely to reveal this first.

1. `linearity_verification_sample_rate` (config, default 0.01): for sampled
   base samples, after generating the variant at the **lowest** target
   safety factor (highest scaled load), run one additional real FEniCS
   solve directly at that scaled load (not via scaling). Compare against
   the scaled-from-reference field: report max relative error in von Mises
   stress and displacement magnitude.
2. Flag (never silently discard) any sample exceeding a configurable
   tolerance (default 1%), and flag its entire base_sample_id
   family/scale-bucket stratum for manual review.
3. Cross-reference any L-Bracket failures against both the geometric-
   linearity gate (Section 12) and the Tier 2 calibration report for the
   same sample — a failure correlated across more than one check is a
   stronger signal than any single one alone.
4. Spot-checks must be deterministic given a fixed random seed.
5. Produce `linearity_spot_check_report.md`, all 5 families, pass/fail and
   worst-case relative error per family. The full generation run must not
   proceed past calibration if any family's failure rate exceeds
   `max_family_failure_rate` (default 5%) without an explicit logged manual
   acknowledgment.

---

## 12. Train/Val/Test Split

Stratify by **three** composed constraints, in this order:

1. **Scale-extrapolation holdout** (existing intentional design): the
   largest scale bucket is deliberately under-represented in training and
   over-represented in test, to probe scale extrapolation specifically.
   Assign this first.
2. **Per-stratum floor:** guarantee a minimum number of base samples per
   `(geometry_family, scale_bucket)` stratum (15 strata: 5 families × 3
   buckets). `min_base_samples_per_stratum` in `config.yaml` (e.g. 250,
   tuned via the empirical learning-curve check below). If random
   family/scale sampling would leave a stratum under-filled, keep generating
   additional base samples for that stratum specifically rather than
   stopping at a fixed total sample count — **total sample count is an
   output of the balancing requirement, not a hard input.** Apply the same
   floor logic to the safety-factor tiers from Section 10 (each
   `target_safety_factor` should hit a minimum count overall; per-stratum
   per-SF-tier balance, a stricter 15×5=75-cell target, is worth attempting
   first and relaxing only if generation time becomes a real constraint).
3. **Leakage-safety grouping:** all safety-factor variants sharing one
   `base_sample_id` land in the same split — never divided across
   train/val/test. This is not stylistic: the variants are linearly-scaled
   copies of one solve (identical normalized spatial field, differing only
   by a scalar), so splitting them apart is a direct leakage bug that
   inflates apparent validation accuracy.

Implementation order: (1) assign `base_sample_id`s honoring the
scale-extrapolation holdout for the largest bucket first, (2) fill remaining
strata to satisfy the per-stratum floor, (3) verify no `base_sample_id`'s
variants ended up split across sets. If honoring all three constraints is
infeasible for some stratum, log which constraint was relaxed and by how
much — never relax silently.

Target ~80/10/10 overall; document actual achieved ratios per family/scale
bucket in the manifest summary. Extend `scale_distribution_check.py` to
report per-stratum counts *and* per-split `base_sample_id` counts, as the
artifact proving both balance and leakage-safety were actually achieved, not
just attempted.

**Geometric-linearity gate** (mode-aware, two independent checks —
Section 0, decision 6):

- **Bulk-solid families** (Block With Holes, Block With Fillet, L-Bracket):
  `max_displacement / characteristic_length < 0.05`. Flag/discard samples
  exceeding this.
- **Bending-dominated families** (Elongated Bar, Thin Plate): a
  length-relative gate is the wrong proxy — the governing engineering rule
  of thumb is deflection relative to member **thickness** (nonlinearity
  becomes relevant once deflection exceeds roughly half the governing
  thickness, or rotation exceeds ~10°). Add a second, independent check:
  `max_displacement / governing_cross_section_dimension < 0.5`, where
  `governing_cross_section_dimension` is the bar's minimum net cross-section
  (accounting for holes) or the plate's actual instantiated thickness —
  never a fixed constant, computed from the real generated geometry. Flag if
  **either** the bulk-style or bending-style check trips; log which one
  fired, per family, as separate counts (not merged into one flag).

**Empirical sample-count check:** rather than picking a target count
up-front, generate an initial batch (e.g. 1,500 base samples), train a first
pass, plot a learning curve (validation error vs. training-set size). Plot
this against **base-sample count**, not total training-pair count — the
safety-factor variants of one base sample are not statistically independent
(they share an identical spatial field, differing only by a scalar), so a
curve plotted against total pairs will look flattened earlier than the data
actually supports. If the curve is still dropping meaningfully, generate
more; if flattened, that's likely enough. Target range: 3,000–5,000 base
samples (→15,000–25,000 total pairs) as the primary plan; 7,000–10,000 base
samples if GPU access supports a stronger final result rather than just
architecture validation.

---

## 13. Coherence Validation (Automated Per-Sample Gate)

1. Compute strain from displacement via the linear-tetrahedron B-matrix.
2. Apply Hooke's law using the sample's E, ν (full 3D constitutive relation
   — this is a genuine 3D problem, not plane stress/strain).
3. Compare resulting von Mises to the solver's own stress output.
4. Apply a configured relative-error tolerance, automated pass/fail, logged.
5. Exclude failed samples from the usable split by default; retain raw
   output for debugging; never silently drop without logging.

Validate the coherence-check code itself against a known closed-form case
before trusting it on generated samples — a simple rectangular block under
pure uniaxial tension (`σ = F/A` uniform), as a sanity baseline. The
Peterson's-chart calibration in Section 11 is the same validation philosophy
applied to L-Bracket specifically, using a published closed-form reference
instead of a trivial uniform-stress case — treat it as an extension of this
principle, not a separate concern.

---

## 14. Resumability and Failure Handling

- Skip completed sample IDs on rerun (check manifest).
- Catch and log per-sample exceptions (mesh failure, solver non-convergence,
  degenerate scaled geometry) without halting the run.
- Progress logging every ~50 samples: running success/failure/coherence-pass
  counts, average solve time, estimated time remaining.
- Write output/manifest incrementally (append per sample, not held in memory
  until the end) — a mid-run Colab disconnect should lose at most the
  current sample.

---

## 15. Performance Expectations (Empirically Validated)

- ~1.1–1.2 sec/sample steady-state after JIT warm-up, at ~1,600 nodes /
  ~6,400 tets.
- Expect this to increase for larger-scale-bucket samples with
  proportionally larger meshes — log actual per-sample solve time, don't
  assume constant across the scale range.
- First sample in a session pays a one-time JIT compilation cost (~20–30 sec
  observed) — use steady-state average, not first-sample timing, for
  runtime estimates.
- Target runtime for 3,000–5,000 base samples: ~1–1.5 hours of solve time,
  fits a single Colab session with incremental writes to mounted Drive.

---

## 16. Consolidated `config.yaml` Schema

```yaml
# DESIGN CONSTRAINT: No Monte Carlo aggregation anywhere in this pipeline.
# Every base sample = one geometry + one BC + one load direction/location +
# one FEM solve. Safety-factor variants (load_sweep, below) are generated
# via EXACT linear scaling of that single solve's result, never re-solved.
# This preserves physical coherence by construction. See Section 1.

material:
  name: "SSAB Domex/Strenx 420"
  E: 210.0e9              # Pa — CONFIRMED vs. Eurocode 3 / SSAB datasheet
  nu: 0.3                 # CONFIRMED
  yield_strength: 420.0e6 # Pa — CONFIRMED
  UTS: 480.0e6            # Pa — CONFIRMED
  # Representativeness of this specific grade for the real Volvo component
  # scope is OPEN — see README. Resolving the numbers above does not
  # resolve this.

geometry:
  families: [block_with_holes, l_bracket, elongated_bar, plate_3d, block_with_fillet]
  scale_range_mm: [20, 2000]          # characteristic length bounds
  scale_buckets: [small, medium, large]
  scale_extrapolation_holdout_bucket: large
  l_bracket:
    fillet_radius_ratio_range: [0.02, 0.3]      # r/d, matches Peterson's chart convention
    fillet_radius_ratio_holdout: [0.10, 0.20]   # reserved test-only, disjoint from train

mesh:
  global_mesh_size_fraction: 0.05     # of characteristic_length
  local_refinement:
    enabled_for_features: [l_bracket_fillet, block_with_fillet_edge]
    local_mesh_size_formula: "min(global_mesh_size, fillet_radius / 4)"
    min_elements_around_curve: 4

loads:
  type: surface_traction_only         # point loads banned project-wide (ADR-005)

load_sweep:
  reference_load_magnitude: <value, small enough to stay safely elastic>
  stress_percentile: 95               # P95 von Mises — NEVER literal peak (singularity robustness)
  target_safety_factors: [10.0, 4.0, 2.0, 1.3, 0.8]  # mild -> at/beyond nominal yield
  variants_per_base_sample: 5
  peak_stress_cap: 432.0e6            # 0.9 x UTS; discard sample if exceeded

geometric_linearity_gate:
  bulk_families: [block_with_holes, block_with_fillet, l_bracket]
  bulk_max_displacement_fraction_of_length: 0.05
  bending_families: [elongated_bar, plate_3d]
  bending_max_displacement_fraction_of_thickness: 0.5

sampling:
  min_base_samples_per_stratum: 250   # tune via empirical learning-curve check, Section 12
  target_base_samples_range: [3000, 5000]

split:
  ratios: {train: 0.8, val: 0.1, test: 0.1}
  grouping_key: base_sample_id        # leakage-safety: all SF variants of one base sample stay together
  min_multihot_samples: 50            # automated manifest assertion, Section 7

calibration:
  pilot_batch_size: 25                # Tier 1, Section 11
  l_bracket_kt_tolerance: 0.15        # Tier 2 thin-case vs. Peterson chart
  linearity_verification_sample_rate: 0.01   # Tier 3
  linearity_relative_error_tolerance: 0.01
  max_family_failure_rate: 0.05
```

---

## 17. Code Standards

- Python 3.12+, type hints, NumPy-style docstrings.
- Small, single-responsibility modules matching the folder structure.
- Explicit imports, no wildcards.
- Every tunable value (scale range, mesh-size ratios, load sweep, gate
  thresholds, material properties, family selection weights) lives in
  `config.yaml` — nothing hardcoded in pipeline logic.
- Unit tests per the `tests/` file list in Section 2, plus the end-to-end
  5–10 sample test spanning multiple families and scale buckets.

---

## 18. Deliverable Checklist

- [ ] All 5 geometry families implemented, producing valid non-degenerate
      solids (tested)
- [ ] L-Bracket's notch root has a randomized, non-zero fillet radius
      (`r/d` in the configured range, with held-out test sub-range) — no
      sharp reentrant corner anywhere in the dataset
- [ ] Local mesh refinement applied near L-Bracket's fillet (and any other
      small curved feature), not just the global size-to-scale rule
- [ ] Scale sampler spans the configured range, mesh size scales with object
      size, confirmed via `scale_distribution_check.py`
- [ ] Elongated-Bar + large-scale samples confirmed (via test) to produce
      high graph-diameter samples
- [ ] Multi-hot `is_fixed ∧ is_loaded` case present, enforced as an
      automated manifest-level assertion (not only a unit test)
- [ ] Every sample: single solve, coherence-checked, logged
- [ ] Reaction forces included, or gap explicitly documented if infeasible
- [ ] Global attributes computable per sample, including `base_sample_id`
      and `target_safety_factor`
- [ ] Point-load code path does not exist anywhere in `solve/loads.py` —
      surface traction only
- [ ] Load strategy implemented as reference-solve + P95-based linear
      scaling to target safety factors, via `solve/load_scaling.py`
- [ ] Calibration run and logged before full generation: Tier 1 (general
      pilot), Tier 2 (L-Bracket Peterson's-chart / through-thickness suite),
      Tier 3 (linear-scaling spot-check) — all three, in order
- [ ] Mode-aware geometric-linearity gate implemented (length-relative for
      bulk families, thickness-relative for bending families), each logged
      separately
- [ ] Train/val/test split composed correctly across all three constraints
      (scale-extrapolation holdout, per-stratum floor, `base_sample_id`
      leakage-safety grouping) — zero cross-split leakage, verified by test
- [ ] README.md and ADRs contain accurate, separately-dated provenance
      statements for material values (numeric CONFIRMED, Volvo-scope
      representativeness OPEN) and for the no-Monte-Carlo/SFEM rejection
      rationale
- [ ] New ADR entry documents the L-Bracket fillet-radius design change,
      linked to `calibration_percentile_report.md`
- [ ] Output written incrementally (Colab-disconnect-safe), resumable
- [ ] Fully config-driven, no hardcoded tunables in pipeline logic

---

## 19. Relationship to the Execution Plan

This document defines *what the pipeline should be*. Actual implementation
work — reading the current codebase, making changes, verifying against
acceptance criteria, committing checkpoints — is governed by
`opus_task_spec_load_config_corrections.md`, whose six tasks map onto this
spec as follows:

| Task | Spec section(s) it implements |
|---|---|
| 1 — L-Bracket fillet radius | §3.2, §4 (local mesh refinement), §16 |
| 2 — Mode-aware linearity gate | §12 (geometric-linearity gate), §16 |
| 3 — Leakage-safe split + multi-hot assertion | §7 (multi-hot), §12 (split) |
| 4 — Peterson's-chart / through-thickness calibration | §11 (Tier 2) |
| 5 — Linear-scaling spot-check | §11 (Tier 3) |
| 6 — README/ADR provenance corrections | §1 (SFEM rationale), §10 (material provenance) |

Read this spec first for context on *why* each task exists, then execute
the task file in the stated dependency order.
