# FEM Dataset Generation Pipeline — Physics Verification & Corrected Implementation Plan

**Status:** Supersedes `implementation_plan_latest.md` (v2) and the third-party
verification report (`FEM_Dataset_Plan_Verification.pdf`) on the points below.
**Scope:** Addresses only the items the verification report raised. Everything
in `implementation_plan_latest.md` not mentioned here is unchanged.

---

## 0. Method

Every load-bearing physics claim in the verification report was checked
against a primary or near-primary source: Eurocode 3 (EN 1993-1-1 §3.2.6),
SSAB's own Domex 420MC datasheet, standard FEA vendor documentation
(Abaqus/Ansys/COMSOL) on element locking and stress singularities, and the
MeshGraphNets literature on message-passing receptive fields. Findings fall
into three buckets:

1. **Confirmed correct** — keep as recommended.
2. **Right conclusion, wrong mechanism** — the numeric recommendation
   survives, but the physical justification in the report is incorrect and
   is corrected here.
3. **Real gap the report didn't surface** — a consequence of its own
   recommendation that it didn't account for.

---

## 1. Confirmed correct (no dispute)

| Claim | Verification |
|---|---|
| E = 210 GPa, ν = 0.3 | Exact match to Eurocode 3 EN 1993-1-1 §3.2.6 design values (E = 210,000 MPa, ν = 0.30, G ≈ 81,000 MPa). |
| σ_yield = 420 MPa, UTS 480–620 MPa for Domex 420MC | Matches SSAB's own product datasheet (ReH min 420 MPa, Rm 480–620 MPa) exactly. |
| Exact linear scaling of load variants | Mathematically exact for linear elasticity: `Ku=F` is linear in `F`, so `K(αu)=αF` holds identically — no approximation, no numerical noise from re-solving. This is the single best design decision in the plan (ADR-006). Keep it. |
| Point loads → non-convergent singularity | Confirmed: applying a finite force at a single mesh node produces peak stress that *increases without bound* under mesh refinement, unlike a true stress concentration (hole, fillet) which converges to a finite value as the mesh refines. This is standard, well-documented FEA behavior, not an edge case. |
| Linear (4-node) tetrahedra are too stiff in bending | Confirmed: first-order tetrahedra are constant-strain elements; under bending they generate spurious ("parasitic") shear strain that artificially stiffens the element. This is universally documented and second-order (10-node) tetrahedra are the standard remedy. |
| MeshGraphNets receptive field = message-passing layer count | Confirmed: the original MeshGraphNets architecture uses ~15 processor blocks as its standard depth, and each layer extends the receptive field by exactly one graph hop. A graph diameter that exceeds the deployed model's layer count genuinely causes "message-passing under-reaching" — this is an active research problem (over-squashing/over-smoothing), not a report exaggeration. |

**Action: no changes to these parts of the plan.**

---

## 2. Right number, wrong physics — displacement threshold

### What the report got wrong

The report frames `max_displacement_fraction_of_length` as an **infinitesimal
strain tensor** validity check, and cites the standard Cauchy strain
derivation (dropping the `½(∇u)ᵀ∇u` term from Green-Lagrange strain) to argue
for an "existing" 5% strain limit.

This conflates two physically distinct things:

- **Material strain** (ε = ∂u/∂x, locally): governs whether Hooke's law is
  still a valid stress-strain relationship.
- **Displacement-to-length ratio** (δ/L, globally): governs whether the
  *geometry* has moved enough that the original stiffness matrix `K` is no
  longer a good approximation (geometric/kinematic nonlinearity).

These are not proportional. A long slender cantilever can have a tip
deflection that is a large fraction of its length while the underlying
material strain stays tiny — this is ordinary beam mechanics: deflection
scales roughly with `L³` while bending strain scales with `σ/E`. Given this
pipeline's own material (E = 210 GPa), even stress *at* yield corresponds to
strain of only ~0.2%, and the plan's most aggressive safety-factor target
(SF = 0.8, i.e. 125% of yield) still only reaches ~0.25% strain. **Material
strain will essentially never be the binding constraint for this steel at
these load levels.** The report's stated mechanism doesn't actually apply to
this pipeline's regime.

### What's actually correct

Displacement-to-length thresholds around 5% (the "1/20 rule") *are* a
real, widely used engineering heuristic — but for a different reason:
**geometric nonlinearity**, i.e., whether the stiffness matrix `K` itself
would need to change under load (P-Delta effects, follower-load effects,
large-rotation kinematics). This is independently well documented across
FEA vendor guidance as a rule of thumb (not a hard theorem) for when to
switch on large-deflection/NLGEOM analysis.

This reframing matters a lot for this specific pipeline, because **the
entire "exact linear scaling" design in ADR-006 is a bet on geometric
linearity holding**: linear scaling of `F → u,σ` is only exact if `K`
doesn't change with deformation. If displacements get large enough that `K`
should really be `K(u)`, the linear-scaling premise silently breaks —
independent of whether the material is still elastic. So this check isn't
a nice-to-have data-quality filter; it's the direct validity gate for the
pipeline's core architectural decision.

### Corrected recommendation

- **Keep the threshold at 0.05** (agree with the report's number) — but
  document it in `config.yaml` and the README as a **geometric-linearity /
  stiffness-matrix-invariance check tied to ADR-006**, not a strain-validity
  check.
- **Add a genuine local strain diagnostic** alongside it. It's nearly free
  (the strain field is already computed for the coherence check) and gives
  a real audit trail: a sample can be flagged "large deflection, small
  strain" (benign — elastica-like slender-member bending, common and not a
  data-quality problem) vs. flagged for a reason that would actually
  indicate trouble. Log both; don't conflate them into one flag.

```yaml
load_sweep:
  max_displacement_fraction_of_length: 0.05   # geometric-linearity gate (K-invariance for ADR-006 scaling), NOT a strain limit — was 0.08

validation:
  max_local_strain_diagnostic: 0.02   # informational only; logged for traceability, does not gate/discard samples
```

---

## 3. Confirmed real fixes (report is right, plan needs to change)

### 3.1 Ban point loads

`loads.py` must not support point loads at all. Replace with mandatory
surface tractions (pressure) applied over a face or a bounded nodal patch of
minimum area, so that as the mesh refines, the loaded area stays fixed
relative to the geometry and stress at the load application region
converges to a finite value instead of diverging.

```yaml
loads:
  application_type: surface_traction   # point_load removed — not a valid option
  min_patch_area_fraction: 0.01        # minimum loaded-face area as fraction of characteristic_length^2
```

### 3.2 Percentile-based stress for the scale factor

`load_scaling.py`'s `scale_factor = (yield_strength / target_SF) / reference_peak_von_mises`
must not use the absolute peak. Even with point loads banned, sharp
re-entrant corners (the `l_bracket`, `block_with_holes` fillet edges) are
real geometric stress concentrators that can behave near-singularly in a
purely linear-elastic model. A single such node shouldn't be allowed to
suppress the load applied to the entire structure.

```yaml
load_scaling:
  stress_percentile: 95   # use 95th percentile von Mises across all elements/nodes, not the absolute max
  # 99th percentile available as a stricter alternative — document whichever is chosen and why
```

Use `scale_factor = (yield_strength / target_SF) / percentile_95_von_mises`.

### 3.3 Quadratic tetrahedra — correct, but with real consequences the report didn't address

`run_fem_3d.py` should move from linear (CG1) to quadratic (CG2) tetrahedra
to eliminate shear locking in the bending-dominated families
(`l_bracket`, `elongated_bar`). This part of the report is right. But three
downstream consequences follow that the report treats as "unchanged":

**(a) Compute cost.** Quadratic tetrahedra roughly double-to-triple the DOF
count per element versus linear tets, and the resulting stiffness matrix is
denser. At 3,000–5,000 base FEM solves, this is a real, non-trivial increase
in wall-clock time and memory per Colab run. Budget for this explicitly —
run a timing pilot on ~30 calibration samples with CG2 before committing to
the full run, and compare against the existing CG1 calibration numbers.

**(b) `b_matrix_tetra.py` / `coherence_check.py` are NOT unchanged.** For a
linear (CG1) tetrahedron, the strain-displacement B-matrix is constant
across the element — that's exactly *why* it's a constant-strain element,
and it's why the current coherence check (single B-matrix per element,
multiply by nodal displacement, compare to solver stress) works as written.
For a quadratic (CG2) tetrahedron, strain varies linearly within the
element — B is a function of position, evaluated at quadrature points, not
a single constant matrix. The coherence check must be rewritten to evaluate
B at specific points (e.g. at each corner node or at Gauss points) rather
than once per element. This is new work, not a pass-through.

**(c) HDF5 schema / MeshGraphNets compatibility.** Solving with CG2 produces
displacement/stress/strain at 10 nodes per tetrahedron (4 corners + 6 edge
midpoints), but the existing schema and the downstream `gnn_project_version_2`
architecture (per its ADRs) expect a linear graph over corner nodes only.
**Recommended resolution:** solve at CG2 (for accuracy — the physics is
computed correctly across the whole element, including bending), but
**export only the corner-node subset** of the solution into the HDF5 file,
keeping `elements/connectivity` as `(E, 4)` exactly as already specified.
This gets the shear-locking fix without touching the downstream GNN
architecture or the HDF5 schema at all. Because stress/strain recovered at a
shared corner node will generally differ slightly between adjacent elements
(this is a standard FE discontinuity, present at any element order), apply
simple nodal averaging over all elements sharing that node — call this out
explicitly as new logic in `run_fem_3d.py`'s output-extraction step, and add
a corresponding new test in `test_coherence_check.py` (which the current
plan also lists as "unchanged" — it isn't, once the B-matrix stops being
constant).

```yaml
solver:
  element_order: 2                    # CG2 / quadratic (10-node) tetrahedra — was CG1/linear
  output_node_subset: corner_only     # export displacement/stress/strain at the 4 corner nodes per element only
  nodal_stress_recovery: averaged     # average recovered stress/strain over all elements sharing a corner node
```

**New/changed modules (correcting the "unchanged from v1" labels):**
- `validation/b_matrix_tetra.py` — **changed**: must evaluate B at specific points within a quadratic element rather than treating it as element-constant.
- `validation/coherence_check.py` — **changed**: compare solver stress against B-matrix-derived stress at corner nodes specifically, accounting for nodal averaging.
- `tests/test_coherence_check.py` — **changed**: add a case exercising the CG2 coherence path, not just the CG1 analytical case.

---

## 4. Graph diameter / message-passing threshold

The report's specific "30–60 hops" figure for a 20-elements-per-side mesh is
a plausible order-of-magnitude estimate, not a derived number — actual graph
diameter depends heavily on element connectivity, aspect ratio, and mesh
regularity, which is exactly why the plan already computes `graph_diameter`
per sample rather than assuming it. Keep that as-is.

One correction to the report's suggested `max_graph_diameter` warning
threshold (~25–30 hops): that number should be **derived from the actual
downstream GNN's message-passing depth** (per the `gnn_project_version_2`
architecture ADRs), not picked as a generic MeshGraphNets-paper default.
If the downstream model uses a different processor depth than the original
paper's 15 layers, the threshold should track that, plus a margin.

```yaml
validation:
  max_graph_diameter_warning: null   # set from gnn_project_version_2's actual message-passing depth + margin, not a generic default
```

---

## 5. Summary of changes vs. `implementation_plan_latest.md` (v2)

| Item | v2 plan | Corrected |
|---|---|---|
| `max_displacement_fraction_of_length` | 0.08, framed as unrelated to core design | 0.05, explicitly tied to ADR-006's K-invariance assumption |
| Local strain tracking | Not present | Added as informational diagnostic (does not gate samples) |
| `loads.py` | Supports point load *or* surface traction | Surface traction / bounded patch only — point loads removed entirely |
| `load_scaling.py` denominator | Absolute `reference_peak_von_mises` | 95th-percentile von Mises (configurable) |
| Element order | Linear (CG1) tetrahedra, "unchanged from v1" | Quadratic (CG2) tetrahedra, solved at CG2 / exported at corner nodes only |
| `b_matrix_tetra.py`, `coherence_check.py` | "Unchanged from v1" | Rewritten for non-constant B-matrix; new coherence test case required |
| HDF5 `elements/connectivity` | `(E, 4)` | Unchanged `(E, 4)` — CG2 solved internally, corner-only export preserves schema |
| `max_graph_diameter` warning | Generic ~25–30 hop default | Derived from actual downstream GNN processor depth |
| Compute budget | Not addressed | Explicit CG2 timing pilot (~30 samples) required before full 3,000–5,000 sample run |

---

## 6. What to discard from the verification report

- The claim that 5% is *the* infinitesimal-strain validity limit, derived
  from dropping Green-Lagrange nonlinear terms — this is a real derivation
  for a different quantity (local strain) misapplied to a different one
  (displacement/length ratio). Corrected in §2 above.
- The "30–60 hop" graph diameter estimate presented as if it were a
  computed fact rather than a rough guess — treat it as illustrative only.
- Any implication that adopting quadratic elements is a drop-in swap with
  no cost — it changes the coherence-check math and the compute budget;
  neither was addressed.
