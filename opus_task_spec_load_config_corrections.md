<!--
  This file is the final, consolidated corrections plan for gnn_project_version_2's
  3D dataset-generation load-configuration pipeline. It supersedes the earlier prose-
  style validation report and is restructured specifically for Claude Opus running in
  Antigravity, following Anthropic's current prompting best practices:
  https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices

  Structural choices made deliberately per that guidance:
  - XML tags delineate role / context / constraints / tasks / execution protocol,
    since Opus is trained to parse tag boundaries unambiguously.
  - Long reference material (<context>) is placed before the actionable <tasks>,
    per the long-context guidance that queries/instructions near the end of a
    prompt improve response quality on multi-document inputs.
  - <global_constraints> includes explicit anti-overengineering, anti-hardcoding,
    and investigate-before-editing guardrails, since Opus models default toward
    broader changes and confident-but-unverified edits unless told otherwise.
  - Each <task> carries its own <acceptance_criteria>, so correctness is checked
    against a stated test, not left to judgment.

  Paste the block below (everything from <role> to the closing </final_instruction>)
  directly into Antigravity as the task prompt, or reference this file from AGENTS.md.
-->

<role>
You are the implementation engineer for the `gnn_project_version_2` 3D dataset-generation
pipeline (`Training_pipeline` repo). You have deep expertise in FEA/solid mechanics,
linear elasticity, and GNN-based surrogate modeling. AGENTS.md is the source of truth
for repository conventions. Every design decision you make that isn't purely mechanical
(renaming, formatting) should be logged as an ADR, following the project's existing
ADR-001 through ADR-007 pattern.
</role>

<success_criteria>
This corrections pass is complete and correct when all of the following hold:
1. No geometry family can produce a mesh-non-convergent (singular) stress field —
   specifically, the L-Bracket family no longer contains a sharp reentrant corner.
2. The geometric-linearity gate correctly flags bending-dominated families
   (Elongated Bar, Thin Plate) using a thickness-relative criterion, not a
   length-relative one.
3. Train/val/test splits contain zero cross-split leakage from the linearly-scaled
   safety-factor variants of any base sample.
4. L-Bracket's computed stress-concentration behavior is checked against published
   Peterson/Shigley Kt trends via a calibration procedure that correctly accounts
   for the difference between a 2D chart assumption and a real 3D block — not a
   naive 1:1 comparison.
5. The linear-scaling assumption (1 solve -> 5 variants) is empirically spot-checked
   against independent re-solves, not just assumed to hold everywhere.
6. README.md and the relevant ADRs contain accurate, separately-dated provenance
   statements — no claim is left unsourced or conflated with a different claim.
</success_criteria>

<context>
  <document index="1">
    <source>Load Configuration & I/O Report — original pipeline design</source>
    <document_content>
The pipeline generates 5 geometry families (block_with_holes, l_bracket,
elongated_bar, plate_3d, block_with_fillet) using surface-traction loading only
(point loads banned). For each base sample: one reference FEniCS/Dolfin linear-
elastic solve is run, then displacement/strain/stress fields are exactly linearly
scaled to hit 5 target safety factors (config: safety_factor_targets
[1.2, 1.5, 2.0, 3.0, 5.0] or, per the corrections doc, a mild-to-failure spread
such as [10.0, 4.0, 2.0, 1.3, 0.8]). Scaling denominator uses P95 von Mises
(not peak) specifically to avoid singularity suppression. A peak-stress cap at
0.9xUTS (432 MPa) discards over-limit samples. A geometric-linearity gate
(max_displacement / characteristic_length &lt; 5-8%) flags/discards samples with
excessive deflection. Material: Domex 420MC steel, E=210 GPa, nu=0.3,
sigma_yield=420 MPa, UTS=480 MPa. The pipeline explicitly rejects Monte Carlo
aggregation (as used by the CMU SFEM dataset) as a design constraint, and uses
stratified floor sampling (minimum base samples per family x scale_bucket) to
guarantee balanced coverage. Target: 3,000-5,000 base samples -> 15,000-25,000
training pairs via the load sweep.

L-Bracket specifically: "Box with a rectangular notch cut from one edge ->
non-convex solid... the reentrant corner produces a stress singularity under
linear elasticity." As originally specified, this notch has a sharp (zero-radius)
internal corner.
    </document_content>
  </document>

  <document index="2">
    <source>Literature validation findings (prior research pass)</source>
    <document_content>
- Point-load ban is well-supported: point loads/restraints and sharp reentrant
  corners are the two classic causes of mesh-non-convergent FEA stress
  singularities.
- The reference-solve + linear-scaling technique is physically exact (superposition
  of a linear-elastic solution) but methodologically uncommon: neither CMU SFEM
  (independent stochastic Monte Carlo solves) nor DeepMind's MeshGraphNets/
  DeformingPlate dataset (independent transient trajectories) nor M4GN's
  DeformingBeam family generate load diversity this way. The 5 scaled variants
  of one base sample share an identical normalized spatial field and are NOT
  statistically independent samples.
- CMU SFEM (Ezemba, McComb & Tucker, 2025, ASME J. Mech. Design): ~16,000
  BrepGen geometries, FEniCSx/DOLFINx solves, E~2.3 GPa / nu~0.40 (soft,
  non-steel material), stochastic point loads (200/2,000/20,000 N), 50 Monte
  Carlo realizations per geometry/load combo, objective is predicting the
  converged DISTRIBUTIONAL stress field for uncertainty quantification -- a
  different, not incompatible-by-flaw, learning objective from this project's
  deterministic paired-field regression.
- M4GN's DeformingBeam / DeformingBeam-Large family exists specifically to
  maximize graph diameter and stress-test long-range coupling in message-passing
  GNNs -- a near-identical published precedent for this project's Elongated Bar
  rationale. MeshGraphNets' reference config (128-d hidden, 15 processor layers)
  independently matches this project's architecture.
- Geometric-nonlinearity rule of thumb (COMSOL / standard structural-mechanics
  guidance): nonlinearity becomes relevant once deflection in a linear analysis
  exceeds roughly HALF the member's THICKNESS (or rotation exceeds ~10 degrees)
  -- a thickness-relative criterion, not a length-relative one. This matters for
  Elongated Bar and Thin Plate specifically, where characteristic_length is a
  poor proxy for the governing bending dimension.
- Peterson's/Shigley's stress-concentration charts (Table A-15: notched/filleted
  rectangular bar in tension or bending) give Kt = sigma_max/sigma_nom as a
  function of r/d (fillet radius / section depth), well-characterized over
  roughly r/d in [0.02, 0.3]. Below ~0.02 the geometry re-approaches singular,
  mesh-sensitive behavior; above ~0.3 the concentration flattens out.
- Critical 3D caveat found on Peterson's-chart applicability: these charts assume
  NO through-thickness stress variation (thin flat bar / plane stress). Real 3D
  FEM studies show the stress concentration factor genuinely varies through the
  thickness: the peak occurs at MID-PLANE only for thin plates and converges to
  the 2D plane-stress chart value there; for thick plates the true maximum shifts
  toward the FREE SURFACE and can be 24-123% higher than the free-surface value
  itself, tending toward a Poisson's-ratio-dependent constant as thickness grows.
  Mid-plane stress in a thick block approaches a generalized PLANE-STRAIN
  solution instead of the plane-stress chart value. One validated technique for
  isolating a clean comparison: constrain z-displacement to zero on the front/
  back faces of a normal 3D solid model to force a genuine plane-strain state,
  then compare directly against classical plane-strain (not plane-stress)
  reference solutions.
    </document_content>
  </document>
</context>

<global_constraints>
- Investigate before editing: open and read the actual current contents of any
  file before modifying it. Never assume a schema, function signature, or config
  key exists -- confirm it in the repo first. Never make claims about pipeline
  behavior you have not verified by reading the relevant code.
- No overengineering: make only the changes each task specifies. Do not refactor
  surrounding code, add configurability beyond what's requested, or introduce new
  abstractions for one-time operations. A fix to l_bracket.py does not need the
  other four family modules touched unless a task explicitly says so.
- No hardcoding, no test-gaming: implement the general, correct logic described
  in each task. Do not special-case behavior to make a specific unit test pass
  without the underlying logic being genuinely correct for all valid inputs.
- Single active task at a time: work through <tasks> in the order given in
  <execution_protocol>. Do not start a task whose listed dependency isn't
  complete and verified.
- Verify before advancing: after implementing a task, check it against its own
  <acceptance_criteria> yourself (run the unit tests / produce the report
  artifact specified) before moving to the next task. Log the result.
- State tracking: maintain a `corrections_progress.json` at the repo root with
  one entry per task (id, status: pending/in_progress/done, verification result,
  timestamp). Update it as you go so work is resumable across sessions.
- Git checkpoints: commit after each completed and verified task, with a commit
  message referencing the task id and, where applicable, the ADR it produced or
  updated.
- Ask before anything destructive or irreversible (force-push, deleting
  generated data, rewriting git history). Everything else -- local file edits,
  running tests, running the pilot/calibration batches -- proceed without asking.
</global_constraints>

<tasks>

  <task id="1" severity="critical" title="Replace L-Bracket's sharp reentrant corner with a parameterized fillet radius">
    <rationale>
    A sharp (zero-radius) internal corner produces a true mathematical singularity
    in linear elasticity: peak stress grows without bound under mesh refinement
    and never converges. This is the root cause of needing the P95-percentile
    workaround in the first place, and it also isn't physically realistic --
    every manufactured L-bracket has some corner radius from machining, casting,
    or forming. Removing the singularity at the geometry level is a better fix
    than continuing to work around it statistically.
    </rationale>
    <instructions>
    1. In `l_bracket.py`, add a randomized shape parameter `fillet_radius` at the
       notch root (the internal reentrant corner), replacing the sharp corner
       entirely -- every generated l_bracket sample must have a non-zero notch-
       root radius.
    2. Express and sample this parameter as a ratio `r/d` (radius / relevant
       section depth, matching the Peterson's-chart convention), drawn uniformly
       from r/d in [0.02, 0.3] by default (expose both bounds in `config.yaml`).
    3. Reserve a held-out sub-range of r/d for test-only, e.g. train on
       r/d in [0.02, 0.10] union [0.20, 0.30], test on r/d in (0.10, 0.20) --
       expose the split boundaries in `config.yaml` as
       `l_bracket_radius_ratio_holdout: [0.10, 0.20]`. This gives an explicit
       extrapolation-along-concentration-severity test, consistent with the
       project's existing scale-bucket holdout pattern -- do not conflate the
       two holdouts, keep them as independent config keys.
    4. Add `fillet_radius` and `radius_ratio` to the per-sample manifest output
       so downstream calibration and analysis can condition on it.
    5. Confirm mesh generation (Gmsh OCC kernel) correctly fillets the notch root
       at small r/d without producing degenerate or self-intersecting geometry --
       add a geometry-validity check that rejects (not silently passes) any
       sample where mesh generation fails or produces inverted elements near the
       fillet.
    </instructions>
    <acceptance_criteria>
    - Zero generated l_bracket samples have a zero-radius notch root; assert this
      as a manifest-level check.
    - r/d distribution across a pilot batch (50+ samples) matches the configured
      train/holdout ranges with no leakage of held-out r/d values into train.
    - A 3-4 step mesh-refinement study on one fixed r/d value shows peak von
      Mises stress converging (<5% change between the two finest refinements) --
      contrast this explicitly against the old sharp-corner behavior, which
      must NOT converge under the same refinement study (run both, document the
      difference in the calibration report from Task 4).
    </acceptance_criteria>
  </task>

  <task id="2" severity="critical" title="Mode-aware geometric-linearity gate">
    <rationale>
    The current gate (max_displacement / characteristic_length &lt; 5-8%) is a
    reasonable proxy for bulk-solid families but is the wrong denominator for
    bending-dominated families, where the literature's rule of thumb is
    deflection relative to member THICKNESS, not overall length. As specified,
    the gate can silently pass Elongated Bar and Thin Plate samples that are
    already geometrically invalid.
    </rationale>
    <instructions>
    1. In `config.yaml`, replace the single `max_displacement_fraction_of_length`
       key with per-family-class settings: a "bulk" class (block_with_holes,
       block_with_fillet, l_bracket) keeping the existing length-relative check,
       and a "bending" class (elongated_bar, plate_3d) adding a second,
       independent check: displacement / governing_cross_section_dimension,
       default threshold 0.5 (deflection exceeding ~half the governing
       thickness). Both checks run for the bending class; flag if either trips.
    2. Each bending-class family module must expose `governing_thickness` in its
       manifest output, computed from the actual instantiated geometry
       parameters (minimum net cross-section for elongated_bar, accounting for
       holes; actual plate thickness for plate_3d) -- never a fixed constant.
    3. The validation/coherence-check step must log which specific check fired,
       per family, as separate counts -- not merged into one flag.
    4. Preserve exact existing behavior for the 3 bulk families; do not change
       their default threshold or flag/discard semantics.
    </instructions>
    <acceptance_criteria>
    - Unit test: for elongated_bar and plate_3d, a synthetic case with
      displacement/characteristic_length &lt; 5% but displacement/
      governing_thickness &gt; 50% is flagged.
    - Unit test: block_with_holes output is bit-for-bit identical to the
      pre-change gate on 5 fixed fixtures.
    - `scale_distribution_check.py` reports both check types separately per
      family.
    </acceptance_criteria>
  </task>

  <task id="3" severity="critical" title="Base-sample-aware, leakage-safe splitting + automated multi-hot assertion">
    <rationale>
    The 5 safety-factor variants of one base sample are linearly dependent
    (identical normalized spatial field, differing only by a scalar). If
    `split_strategy.py` doesn't group by base_sample_id, near-duplicate variants
    can land in both train and test, inflating validation accuracy.
    </rationale>
    <instructions>
    1. `split_strategy.py` must treat `base_sample_id` as the atomic splitting
       unit: all safety-factor variants of one base sample go to the same split,
       never divided across train/val/test.
    2. This must compose correctly with the two existing constraints: (a) the
       per-stratum (family x scale_bucket) minimum floor, and (b) the intentional
       large-scale-bucket extrapolation holdout. Order of operations: assign
       base_sample_ids honoring (b) for the large-scale bucket first, then fill
       remaining strata to satisfy (a), then verify no base_sample_id's variants
       are split across sets.
    3. Add an automated manifest assertion (not a manual review step): fail
       loudly if fewer than `min_multihot_samples` (new config key, default 50)
       samples exist with a node having both is_fixed=1 and is_loaded=1.
    4. If honoring both the leakage-safety grouping and the per-stratum floor is
       infeasible for some stratum, log which constraint was relaxed and by how
       much -- never relax silently.
    </instructions>
    <acceptance_criteria>
    - Unit test: 20 synthetic base_sample_ids x 5 variants each -> zero
      base_sample_ids with variants split across more than one set.
    - Unit test: multi-hot check correctly fails at 0 qualifying samples and
      passes at >= min_multihot_samples.
    - `scale_distribution_check.py` reports per-split base_sample_id counts
      alongside per-stratum counts.
    </acceptance_criteria>
  </task>

  <task id="4" severity="moderate" title="Peterson-chart / through-thickness Kt calibration suite" depends_on="1">
    <rationale>
    Peterson's/Shigley's charts assume no through-thickness stress variation
    (thin flat bar). A real 3D L-bracket block does not satisfy that assumption:
    published 3D FEM studies show the peak Kt shifts toward the free surface in
    thick blocks and can exceed the free-surface value by 24-123%, while
    mid-plane stress in a thick block approaches a plane-strain solution, not
    the plane-stress chart value. A naive 1:1 comparison between your production
    L-bracket and the raw chart number will be wrong by construction. This task
    replaces the earlier flat "P95-per-family validation" idea with a more
    complete calibration procedure.
    </rationale>
    <instructions>
    1. Build a dedicated, calibration-only thin variant of the l_bracket
       geometry (thickness / notch_depth &lt;= 0.2), at 4-5 r/d values spanning
       the configured range from Task 1. Run it through the normal solve
       pipeline. Extract Kt = sigma_max / sigma_nom at mid-thickness (sigma_nom
       = mean stress on a net cross-section taken away from the notch, along the
       load axis -- define this formula concretely and bake it into the
       manifest computation, not as a post-hoc analysis choice). Compare against
       Peterson's chart value at the matching r/d, w/d.
    2. Separately, using the normal production-thickness l_bracket geometry, run
       one calibration variant per r/d value with z-displacement constrained to
       zero on the front/back faces (forcing a plane-strain state within the
       standard 3D solver/mesh path). Compare this against a plane-strain
       reference value (not the plane-stress chart number).
    3. For a pilot batch of normal (unconstrained, production-thickness)
       l_bracket samples, compute and log peak_von_mises, p95_von_mises,
       p90_von_mises, p99_von_mises, and Kt at multiple through-thickness z-
       slices (at minimum: mid-plane and near-surface). Plot Kt vs. z.
    4. Produce `calibration_percentile_report.md` covering: (a) thin-case vs.
       Peterson comparison, pass/fail per r/d, tolerance ~15% given known 3D
       deviation; (b) plane-strain-BC case vs. plane-strain reference; (c) the
       Kt-vs-z profile for production-thickness samples, with a written check
       that it qualitatively matches the literature pattern (roughly flat,
       shifting toward a surface peak as relative thickness grows) -- flag, do
       not just note, any profile that looks flat with no surface trend
       regardless of thickness, since that would suggest a modeling problem;
       (d) whether stress_percentile: 95 in `config.yaml` is validated as
       singularity-robust for l_bracket specifically, given the new finite-
       radius geometry from Task 1, or whether a per-family override is needed.
    5. Do not change production `config.yaml` values based on this task without
       the report existing and being referenced in the commit.
    </instructions>
    <acceptance_criteria>
    - `calibration_percentile_report.md` exists and contains all four
      sub-reports listed above, each with an explicit pass/fail or
      confirmed/flagged status -- no open-ended "seems fine" statements.
    - Thin-case Kt values are within the stated tolerance of Peterson's chart
      for at least 4 of 5 tested r/d values; any failure is explained, not
      silently dropped.
    </acceptance_criteria>
  </task>

  <task id="5" severity="moderate" title="Empirical spot-check of the linear-scaling assumption" depends_on="4">
    <rationale>
    Phase 2 of the load strategy assumes exact linear scaling holds at any
    target load. This is true in exact linear elasticity but could be silently
    violated by solver tolerance settings or an accidental nonlinear flag on a
    subset of samples. l_bracket (near its new fillet root) is the family most
    likely to reveal this first.
    </rationale>
    <instructions>
    1. Add `linearity_verification_sample_rate` to `config.yaml` (default 0.01).
    2. For sampled base samples, after generating the variant at the LOWEST
       target safety factor (highest scaled load), run one additional real
       FEniCS solve directly at that scaled load (not via scaling). Compare
       against the scaled-from-reference field: report max relative error in
       von Mises stress and displacement magnitude.
    3. Flag (do not silently discard) any sample exceeding a configurable
       tolerance (default 1%), and flag its entire base_sample_id
       family/scale-bucket stratum for manual review.
    4. Cross-reference any l_bracket failures against the Task 2 linearity-gate
       output and the Task 4 calibration report for the same sample -- a
       failure correlated across more than one of these checks is a stronger
       signal than any single one alone; log this cross-reference explicitly.
    5. Spot-checks must be deterministic given a fixed random seed.
    </instructions>
    <acceptance_criteria>
    - `linearity_spot_check_report.md` produced, covering all 5 families, with
      pass/fail and worst-case relative error per family.
    - The full generation run must not proceed past calibration if any family's
      spot-check failure rate exceeds `max_family_failure_rate` (default 5%)
      without an explicit logged manual acknowledgment.
    </acceptance_criteria>
  </task>

  <task id="6" severity="minor" title="README/ADR provenance corrections" depends_on="1,4">
    <rationale>
    Two documentation-accuracy issues: the no-Monte-Carlo ADR mischaracterizes
    CMU SFEM as "physically incoherent" when it is actually built for a
    different, legitimate objective; and material-value provenance conflates
    two distinct validation layers (numeric accuracy vs. use-case
    representativeness) into one claim.
    </rationale>
    <instructions>
    1. In the ADR discussing the no-Monte-Carlo constraint, reword the SFEM
       characterization: SFEM performs genuine Monte Carlo aggregation (50
       stochastic point-load realizations per geometry/load combination) and is
       deliberately designed to predict the converged statistical/distributional
       stress field for uncertainty quantification -- a different learning
       objective from this project's deterministic single-solve paired-field
       regression, not a flawed methodology in general. Separately note SFEM's
       material properties (E~2.3 GPa, nu~0.40) are far outside structural-steel
       range, an independent reason it can't be reused even absent the
       objective mismatch.
    2. Split the material-provenance statement in README.md into two explicitly
       separate, separately-dated entries: (a) numeric/datasheet accuracy --
       E=210 GPa, nu=0.3, yield=420 MPa cross-checked against Eurocode 3 and the
       SSAB Domex 420MC datasheet, marked CONFIRMED with date and source; (b)
       representativeness of Domex 420MC for the actual Volvo component scope,
       marked OPEN, restating the open question verbatim from
       `implementation_plan_corrections.md` Section 3. State plainly that
       resolving (a) does not resolve (b).
    3. Add a new ADR entry documenting the Task 1 L-Bracket geometry change
       (sharp corner -> parameterized fillet radius) with the reasoning from
       Task 1's rationale and a reference to the Task 4 calibration report.
    4. This is a documentation-only task -- do not modify pipeline code here.
    </instructions>
    <acceptance_criteria>
    - README.md and the relevant ADR files contain all corrected sections.
    - The markers "CONFIRMED" and "OPEN" appear next to the two provenance
      entries so a future reader (or agent) can grep for open items.
    - The new ADR for the L-Bracket fillet-radius change exists and links to
      `calibration_percentile_report.md`.
    </acceptance_criteria>
  </task>

</tasks>

<execution_protocol>
Work through the tasks in this order, respecting the stated dependencies:
Task 1 -> Task 2 -> Task 3 -> Task 4 (needs Task 1 done) -> Task 5 (needs Task 4
done) -> Task 6 (needs Tasks 1 and 4 done). Tasks 2 and 3 have no dependency on
Task 1 and may be reordered relative to it if that's more efficient, but Task 4
must not start until Task 1 is verified complete.

Before starting each task: read every file the task references. State briefly
what you found before proposing changes -- do not describe intended behavior of
code you have not opened.

After finishing each task: run its acceptance criteria yourself, update
`corrections_progress.json`, and commit with a message referencing the task id.
Only then move to the next task.

If any acceptance criterion cannot be met after a reasonable implementation
attempt, stop and report exactly what failed and why, rather than relaxing the
criterion or marking the task done anyway.
</execution_protocol>

<final_instruction>
Start with Task 1. Read `l_bracket.py` and the relevant sections of
`config.yaml` first, confirm your understanding of the current notch-generation
logic, then implement the fillet-radius parameterization.
</final_instruction>
