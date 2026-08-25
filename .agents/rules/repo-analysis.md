---
trigger: model_decision
glob:
description: Use when the task requires understanding, navigating, or analyzing a large or unfamiliar repository/codebase — repo-wide questions, "explain this project", multi-file refactors.
---

# Repository Analysis Strategy

Large repos are expensive to read in full — don't default to scanning everything.

1. Do NOT scan the entire repository immediately.
2. First determine which files are actually relevant to the question (entry points, imports, directory names — infer scope before reading).
3. Read only the minimum number of files required to answer.
4. If the request genuinely requires repo-wide analysis, say so and outline the plan (which files/dirs, why, rough read count) before reading everything — so I can redirect if your plan is off.
5. Prefer incremental exploration: read a few files, check if that's enough, expand only if not.

(This is the detailed version of the "Large-repo strategy" section in AGENTS.md — kept here as a separate model_decision rule so it only loads into context on repo-analysis tasks, not every request.)
