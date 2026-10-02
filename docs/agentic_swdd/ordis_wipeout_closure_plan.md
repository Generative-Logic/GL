<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# OR branch closure wipeout

## Completed contract

- Implemented from main. The tested source tip d2afdea34 landed on main as exact-tree squash 37c784961, preserving the sandbox branch. Source checkpoint:; first clean-run evidence:.
- A branch is resolved when every current goal at its cohort parent or an ancestor is known at the branch or above. Unrelated goals are ignored; zero relevant goals qualifies.
- Resolved roots join the existing ordered end-of-burst subtree wipe. Runtime rows and descendant subgoals are removed and future deposits are filtered. Parent-chain goals, cohort counts, booked contributions and proof history remain.
- Only derivation-level runs of current goals booked at the closing branch are retained separately. These are statement proof contributions, not values taken from the goal registry. Reduced-count convergence can use them only while the expression remains a current chain goal, preserving the existing first-survivor choice and exact levels. Non-goal levels are not retained.
- The new statified map uses the ordinary deloadable arena and appended tags 826–828. It survives its own branch wipe, clears when the cohort parent is wiped, and clears at discharge. No new pool, capacity, verifier change or configuration was needed.
- Acceptance complete: baseline and both clean full GPU runs have the same theorem counts, theorem rows, theorem sets and verifier totals. The requested main landing is complete; the new decision is D-347 in the separate documentation renumbering step.

## Verification

All runs used native Windows `main.py --GPU` in the primary working directory. The clean Release build used MSBuild `-t:Rebuild` through `env=dict(os.environ)` and passed all 1,659 C++ unit tests. Both clean runs used the same executable, SHA-256 `e92e66d99947372dd67678a923135e9cc406234f10646a2ee58a5565caa1267e`.

| Run | Run log directly under `.debug` | Verifier checks | Failures | Runtime, seconds |
|---|---|---:|---:|---:|
| Main baseline | `run_ordis_wipeout_baseline.log` | 151919 | 0 | 565.82400 |
| Clean run 1 | `run_ordis_wipeout_full_1.log` | 151919 | 0 | 589.74274 |
| Clean run 2 | `run_ordis_wipeout_full_2.log` | 151919 | 0 | 623.11117 |

The eight theorem counts in run-log order are `518, 79, 76, 105, 25, 4, 19, 39` in all three runs. The verifier parts are 25,968 and 125,951 checks. All 6095 snapshot files match except the operational Lean `kernel_check.log`; theorem and proof artifacts are byte-identical. Full manifests and completion records remain in ; comparison reports are . Run logs remain directly in `.debug` so the maintainer does not need to search nested artifact folders.

Direct tests cover partial/all goal coverage, ancestor-known and unrelated goals, vacuous resolution, descendant cleanup and rejected later deposits, ordinary convergence after a branch wipe, reduced-count convergence using saved goal levels, exclusion after a goal is removed, non-goal exclusion, enclosing-scope cleanup, disk roundtrip, and discharge cleanup.

## Material diagnosis and decisions

- A plain subtree wipe erased derivation levels used by reduced-count convergence. The full audit reached five current-goal rechecks without those levels and crashed in FTA at `hostCounters.reverseOwnerCount <= capacity_.reverseOwners`. At burst 18 it had 41 active blocks and 116,243 expressions versus baseline 34 and 52,963. The crash log is .
- The maintainer approved testing retention of only current-goal derivation levels after clarification of their meaning. Retaining levels of every booked statement was not justified or implemented. Parent OR rules can independently rederive facts after refutation, so a missing lookup alone was not proof of a lost theorem; the workload established the impact.
- The goal-level audit completed with 151919 checks, no failures, and the baseline theorem/artifact result. Six goal rechecks used saved levels while 1,067 non-goal rechecks retained none. FTA burst 18 returned to 34 active blocks with 48,598 expressions; the original CUDA assertion did not fire. Evidence:  and the archived full-chain `traps.log` in its artifact folder.
- Temporary closure tracing was removed before the clean build and acceptance runs; both clean trap logs are empty. The pre-existing CUDA assertion, capacities and sacred hashburst dump remain unchanged. The initial diagnostic build's MSVC `fopen` warning was corrected before the audit; no diagnostic code remains.
- Durable behavior is documented in both design-document families. The existing MPU memory-hierarchy and sizing diagrams remain valid because the retained goal-level column uses the existing arena and allocation hierarchy. The main landing assigns D-347 to the pending decision; the preserved sandbox keeps its branch-local placeholder.

Implementation acceptance and the requested exact-tree main landing are complete. This ledger is frozen. Runtime figures are recorded evidence; this task makes no acceleration claim.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
