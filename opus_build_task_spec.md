<!--
  This file REPLACES opus_task_spec_load_config_corrections.md. That file assumed
  an existing, partially-correct codebase to read and fix. There is no existing
  code — this file sequences a from-scratch build instead, with every requirement
  that used to be a "correction" (L-Bracket fillet radius, P95 scaling, mode-aware
  linearity gate, leakage-safe splitting, point-load ban, provenance docs) built
  in natively from the first line of each file, per dataset_pipeline_spec_CONSOLIDATED.md.

  dataset_pipeline_spec_CONSOLIDATED.md is unchanged and remains the sole
  architectural source of truth. This file deliberately does not restate its
  technical content — only references which section governs each build phase —
  to avoid the exact two-document drift problem (0.05 vs 0.08, peak vs P95) that
  the consolidation pass just fixed. Feed both files to Opus together, spec first.

  Phase order follows the one piece of sequencing guidance validated from the
  original plan review and never contradicted since: locally-testable phases
  (no FEniCS/Colab needed) before FEniCS-dependent phases, so cheap pytest
  feedback happens before the expensive environment is ever needed.
-->

<role>
You are the implementation engineer building `gnn_project_version_2`'s 3D
dataset-generation pipeline (`Training_pipeline` repo) from scratch. No
baseline code exists — every file described in `dataset_pipeline_spec_CONSOLIDATED.md`
Section 2 needs to be created by you, exactly as specified there. That
document is the single architectural source of truth; this document
sequences the build into eight verifiable phases and defines what "done"
means for each. Read `dataset_pipeline_spec_CONSOLIDATED.md` in full before
starting Phase 1, and re-read the specific section each phase references
before implementing it — do not implement from a paraphrased memory of an
earlier read. Confirm whether `Training_pipeline` uses its own ADR sequence
or references the parent `gnn_project_version_2` project's (check for an
existing docs/ADR folder first; if none exists, start a local ADR-001
sequence within this repo, consistent with this pipeline's correct
architectural separation from the parent project). Create or update
AGENTS.md as part of Phase 1, pointing at the consolidated spec as the
design reference.
</role>

<success_criteria>
This build is complete when all of the following hold:
1. Every file in `dataset_pipeline_spec_CONSOLIDATED.md` Section 2's file
   structure exists and does what its spec section describes.
2. L-Bracket has never had a sharp reentrant corner at any point in this
   repo's history — the fillet radius is native to the geometry from the
   first commit, not retrofitted.
3. `solve/loads.py` never contains a point-load code path.
4. The full safety-factor load-sweep (P95-based scaling) is implemented and
   calibrated per spec Section 11 before any full-scale generation is
   attempted.
5. `split_strategy.py` composes all three constraints (scale-extrapolation
   holdout, per-stratum floor, `base_sample_id` leakage-safety grouping)
   correctly, verified by test, not assumed.
6. The dataset has actually been generated: every stratum at its floor,
   zero split leakage, calibration reports clean.
7. README.md and ADRs document every design decision made during the
   build, with accurate, dated provenance for material values and the
   no-Monte-Carlo/SFEM rationale.
</success_criteria>

<context>
Building a synthetic 3D FEM dataset to train and validate a physics-informed
GNN (encoder -> processor -> displacement decoder -> B-matrix strain ->
stress decoder -> algebraic von Mises/safety-factor). Full architectural
spec, file structure, config schema, and technical rationale for every
requirement live in `dataset_pipeline_spec_CONSOLIDATED.md` — read it in
full before Phase 1. This document only sequences the build and defines
phase-level acceptance criteria; it deliberately does not restate spec
content.
</context>

<global_constraints>
- Read the `dataset_pipeline_spec_CONSOLIDATED.md` section each phase
  references, in full, before writing code for that phase.
- Before creating any file, check whether it already exists (in case of a
  partial/resumed build) — do not blindly overwrite without checking.
- No overengineering: implement exactly what the referenced spec section
  describes for this phase. Do not add configurability, abstractions, or
  files the spec doesn't call for.
- No hardcoding, no test-gaming: implement the general, correct logic the
  spec describes, not something special-cased to pass one fixture.
- Single active phase at a time, strictly in the order given in
  `<execution_protocol>`.
- Verify each phase against its own `<acceptance_criteria>` before starting
  the next.
- Maintain `build_progress.json` at the repo root (phase id, status,
  verification result, timestamp) so the build is resumable across sessions.
- Git commit after each verified phase, message referencing the phase
  number and any ADR it produced.
- Ask before anything destructive or irreversible (force-push, rewriting
  git history, deleting generated data). Everything else — creating files,
  running tests, running pilot/calibration batches — proceed without asking.
</global_constraints>

<tasks>

  <task id="1" severity="critical" title="Project scaffolding and config">
    <instructions>
    1. Create the full directory/file skeleton per spec Section 2 (empty
       modules with docstrings/TODOs are fine at this stage for files owned
       by later phases — geometry/, solve/, validation/, pipeline/, output/,
       logs/, tests/).
    2. Create `config.yaml` with the exact schema from spec Section 16.
    3. Create `requirements.txt` (gmsh, meshio, numpy, pandas, networkx,
       pyyaml, h5py — not torch).
    4. Initialize `build_progress.json` with all 8 phases listed pending.
    5. Create `README.md` containing: the no-Monte-Carlo declaration with
       the SFEM-objective-mismatch wording from spec Section 1 (not
       "SFEM is incoherent"), and the two-tier material-provenance statement
       from spec Section 10 with explicit CONFIRMED/OPEN markers.
    6. Create or update `AGENTS.md` pointing at
       `dataset_pipeline_spec_CONSOLIDATED.md` as the design source of
       truth, after resolving the ADR-sequence question above.
    </instructions>
    <acceptance_criteria>
    - Directory tree matches spec Section 2 exactly.
    - `config.yaml` loads without error and contains every key from spec
      Section 16.
    - `README.md` contains both required statements with the exact framing
      from spec Sections 1 and 10 — not paraphrased into something weaker.
    </acceptance_criteria>
  </task>

  <task id="2" severity="critical" title="Geometry families (locally testable, no FEniCS)" depends_on="1">
    <instructions>
    1. Implement all 5 families per spec Section 3: `block_with_holes.py`,
       `l_bracket.py`, `elongated_bar.py`, `plate_3d.py`,
       `block_with_fillet.py`.
    2. `l_bracket.py` implements the fillet-radius parameterization from
       spec Section 3.2 natively — r/d in [0.02, 0.3], held-out test
       sub-range (0.10, 0.20), `fillet_radius`/`radius_ratio` recorded per
       sample, geometry-validity rejection for degenerate mesh at the
       fillet. There is no sharp-corner version to correct later.
    3. Implement `scale_sampler.py` per spec Section 4, including the LOCAL
       mesh-size override near curved features
       (`min(global_mesh_size, fillet_radius/4)`) — not just the global
       `0.05 * characteristic_length` rule.
    4. Write `test_geometry_families.py`: each family produces valid
       non-degenerate solids; l_bracket specifically asserts zero samples
       with r=0 across a batch; `scale_sampler` output spans the configured
       range on a pilot batch.
    </instructions>
    <acceptance_criteria>
    - All 5 families pass `test_geometry_families.py`.
    - A 50+ sample pilot batch of l_bracket shows r/d distribution matching
      the configured train/holdout ranges with zero holdout leakage into
      train.
    - Mesh element density near the fillet visibly increases at low r/d
      under the local refinement rule (Gmsh mesh-only check, no solve
      needed yet).
    </acceptance_criteria>
  </task>

  <task id="3" severity="critical" title="Pure-Python support modules (locally testable, no FEniCS)" depends_on="1,2">
    <instructions>
    1. Implement `solve/load_scaling.py` per spec Section 10: pure
       linear-algebra scaling of one solve's fields to N safety-factor
       variants, using P95 von Mises as the denominator (never peak). No
       FEniCS dependency — test with synthetic field arrays.
    2. Implement `validation/b_matrix_tetra.py` (linear tetrahedron
       B-matrix) and `validation/coherence_check.py` (B-matrix strain ->
       Hooke's law -> compare to a provided stress array). Validate against
       a hand-computed closed-form case (uniaxial tension, sigma = F/A)
       using synthetic displacement input — no real FEniCS solve needed.
    3. Implement `pipeline/sample_spec.py`: dataclass representing one base
       configuration (geometry + BC + load direction/location + mesh_size)
       per spec Section 10.
    4. Implement `pipeline/split_strategy.py` per spec Section 12:
       mode-aware geometric-linearity gate (bulk vs. bending families, per
       spec Section 12's two formulas) AND the three-constraint split
       composition (scale-extrapolation holdout -> per-stratum floor ->
       `base_sample_id` leakage-safety grouping), in that order, with
       explicit logging if any constraint must be relaxed.
    5. Write `test_coherence_check.py` (validated against the uniaxial
       case) and `test_split_leakage.py` (synthetic manifest: N
       base_sample_ids x 5 variants each, assert zero cross-split leakage;
       also assert the mode-aware gate fires correctly per family class on
       synthetic fixtures).
    </instructions>
    <acceptance_criteria>
    - `coherence_check.py` reproduces the analytical uniaxial-tension stress
      to within a tight numerical tolerance on synthetic input.
    - `test_split_leakage.py` passes on synthetic fixtures covering all
      three composed constraints.
    - `load_scaling.py` produces the correct scale factor on a
      hand-computed synthetic example (known reference P95, known target
      SF, known expected scale factor).
    </acceptance_criteria>
  </task>

  <task id="4" severity="critical" title="FEniCS solve modules (requires FEniCS/Colab environment)" depends_on="1,2,3">
    <instructions>
    1. Implement `solve/mesh_conversion.py` (Gmsh .msh -> meshio -> Dolfin
       XDMF), `solve/material.py` (Domex 420MC properties per spec Section
       10, full 3D constitutive relation), `solve/boundary_conditions.py`
       (configurable fixed-face selection).
    2. Implement `solve/loads.py` — SURFACE TRACTION ONLY. Do not write a
       point-load code path at all, not even a disabled one (spec Section 0
       decision 1 / Section 10).
    3. Implement `solve/run_fem_3d.py`: one FEniCS solve per call, returns
       full displacement/stress/strain field data for the reference (small,
       safely-elastic) load.
    4. Implement `solve/reaction_forces.py` per spec Section 8; if
       genuinely infeasible from the legacy dolfin API within reasonable
       effort, document the specific limitation in README.md rather than
       silently omitting the columns.
    5. Grep the finished `solve/` directory for any reference to a point
       load and confirm zero matches, as a direct check on instruction 2.
    </instructions>
    <acceptance_criteria>
    - `run_fem_3d.py` produces a coherent solve on a simple test geometry (a
      plain block), verified against `coherence_check.py` from Phase 3.
    - `solve/loads.py` contains no point-load code path (grep-verified).
    - `reaction_forces.py` either produces correct values (spot-checked:
      sum of reactions equals sum of applied loads on the simple block
      case) or the limitation is documented in README.md.
    </acceptance_criteria>
  </task>

  <task id="5" severity="critical" title="Pipeline orchestration and end-to-end run" depends_on="4">
    <instructions>
    1. Implement `pipeline/generate_dataset.py`: orchestrates one base
       solve (Phase 4) + `load_scaling.py` sweep (Phase 3) into
       `variants_per_base_sample` output HDF5 files, applies the
       `peak_stress_cap` discard, applies the multi-hot
       `is_fixed AND is_loaded` requirement (spec Section 7) as an
       automated end-of-run manifest assertion (`min_multihot_samples`),
       writes `manifest.csv` incrementally per spec Section 14.
    2. Implement `pipeline/resume.py`: skip completed sample IDs on rerun,
       per manifest.
    3. Implement `validation/receptive_field_check.py` (graph diameter via
       networkx) and `validation/scale_distribution_check.py`
       (characteristic-length/node-count range check, per-stratum count
       table, per-split `base_sample_id` count table).
    4. Write `test_pipeline_end_to_end.py`: 5-10 samples spanning multiple
       families and scale buckets, full pipeline path including a real
       (small) FEniCS solve.
    </instructions>
    <acceptance_criteria>
    - `test_pipeline_end_to_end.py` passes, producing real manifest rows and
      HDF5 output for a small batch.
    - A deliberate mid-run interruption + rerun on that small batch resumes
      correctly (no duplicate or missing samples).
    - The multi-hot assertion correctly fails on a manifest engineered to
      have zero qualifying samples and passes once enough exist.
    </acceptance_criteria>
  </task>

  <task id="6" severity="critical" title="Calibration suite (Tiers 1-3)" depends_on="2,4,5">
    <instructions>
    Implement per spec Section 11, all three tiers:
    1. Tier 1: general pilot-batch validation (~20-30 base samples, all
       families/scales) — reference solve stays elastic, P95 scale factors
       sane, `peak_stress_cap` discards correctly, achieved SF spread
       matches the configured spread. Log to `logs/calibration.log`.
    2. Tier 2 (`validation/calibration/percentile_calibration.py`):
       L-Bracket-specific thin-case vs. Peterson's-chart comparison,
       plane-strain-BC comparison, through-thickness Kt-vs-z profile, P95
       robustness check — exactly as spec Section 11 Tier 2 describes.
       Produce `calibration_percentile_report.md`.
    3. Tier 3 (`validation/calibration/linearity_spot_check.py`):
       independent re-solve spot-check at the lowest target SF,
       cross-referenced against the Phase 3 linearity gate and the Tier 2
       report. Produce `linearity_spot_check_report.md`.
    4. Run all three tiers now, on the pilot batch, before Phase 8's full
       generation.
    </instructions>
    <acceptance_criteria>
    - `calibration_percentile_report.md` and `linearity_spot_check_report.md`
      both exist with explicit pass/fail status per check — no open-ended
      "seems fine" language.
    - Thin-case Kt matches Peterson's chart within the configured tolerance
      (0.15) for at least 4 of 5 tested r/d values.
    - No family exceeds `max_family_failure_rate` on the Tier 3 spot-check
      without an explicit logged manual acknowledgment.
    </acceptance_criteria>
  </task>

  <task id="7" severity="moderate" title="README/ADR documentation" depends_on="2,6">
    <instructions>
    1. Confirm the no-Monte-Carlo / SFEM-rejection wording in README.md
       (from Phase 1) matches spec Section 1 exactly — objective mismatch,
       not "incoherent," plus the independent material-mismatch note.
    2. Confirm the two-tier material-provenance statement (from Phase 1) is
       present with CONFIRMED/OPEN markers per spec Section 10.
    3. Write a new ADR documenting the L-Bracket fillet-radius design
       decision (spec Section 3.2), referencing
       `calibration_percentile_report.md` from Phase 6.
    4. Document the reaction-force outcome from Phase 4 (implemented, or
       named limitation) in README.md if not already done.
    </instructions>
    <acceptance_criteria>
    - README.md and the new ADR are grep-able for "CONFIRMED" and "OPEN"
      markers next to the two separate material claims.
    - The new ADR links to `calibration_percentile_report.md`.
    </acceptance_criteria>
  </task>

  <task id="8" severity="critical" title="Run and verify the full dataset generation" depends_on="1,2,3,4,5,6,7">
    <instructions>
    1. Confirm all of Phases 1-7 are marked done and verified in
       `build_progress.json` before starting.
    2. Run full generation targeting `sampling.target_base_samples_range`
       from `config.yaml` (default 3,000-5,000 base samples), respecting
       `min_base_samples_per_stratum` as a floor, not a fixed total — let
       generation continue past the low end of the range for any stratum
       still under floor.
    3. Use the incremental/resumable write path (spec Section 14); resume
       from the manifest on any Colab disconnect rather than restarting.
    4. Run `scale_distribution_check.py`: confirm every stratum meets its
       floor, per-split `base_sample_id` counts show zero cross-split
       leakage, and the achieved size distribution actually spans the
       configured range.
    5. Re-run the Tier 3 linearity spot-check (Phase 6) against the full
       generated dataset, not just the Phase 6 pilot batch — report any
       change in failure rate at full scale.
    6. Produce `generation_summary.md`: total base samples, total training
       pairs, per-family/per-scale-bucket/per-SF-tier counts,
       coherence-check pass rate, calibration status, total generation
       time, and any excluded samples with reasons.
    </instructions>
    <acceptance_criteria>
    - `output/manifest.csv` exists, is complete (no partial/orphaned rows
      from an interrupted run), every row has a valid `base_sample_id`,
      `target_safety_factor`, `geometry_family`, and `scale_bucket`.
    - All 15 strata (5 families x 3 scale buckets) meet
      `min_base_samples_per_stratum`.
    - `test_split_leakage.py` passes against the full generated manifest,
      not just synthetic fixtures.
    - `generation_summary.md` exists and is referenced in the final commit.
    </acceptance_criteria>
  </task>

</tasks>

<execution_protocol>
Work through phases strictly in this order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
-> 8. Phases 1-3 require no FEniCS/Colab environment — complete and fully
test these locally first, since pytest feedback here is far cheaper than
debugging inside Colab. Phase 4 is the first phase that needs the actual
FEniCS/Colab environment.

Before starting each phase: read, in full, the spec section(s) it
references in `dataset_pipeline_spec_CONSOLIDATED.md`. State briefly what
that section requires before writing code — do not implement from a
paraphrased memory of an earlier read.

After finishing each phase: run its acceptance criteria yourself, update
`build_progress.json`, commit with a message referencing the phase number.
Only then move to the next phase.

If any acceptance criterion cannot be met after a reasonable implementation
attempt, stop and report exactly what failed and why, rather than relaxing
the criterion or marking the phase done anyway.
</execution_protocol>

<final_instruction>
Start by reading `dataset_pipeline_spec_CONSOLIDATED.md` in full. Then begin
Phase 1: scaffolding and config.
</final_instruction>
