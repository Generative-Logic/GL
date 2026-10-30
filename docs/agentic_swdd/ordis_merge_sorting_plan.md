<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# OR convergence without deposit sorting

## Execution contract

- Objective: remove the four statement-list sorts immediately before OR convergence and establish unchanged theorem results plus repeatable output.
- Branch:, created from main under the user's corrected branch instruction.
- Approval: consume the existing statement-list order, including the shared post-registration admission/OR loop. Different proof bytes from the sorted baseline are allowed; theorem contents must remain equal and repeated runs must match each other.
- Constraints: retain hash-engine ordering, class enumeration, scope rules, origin caps, assertions, and all scratch-lifetime boundaries. Development runs use CUDA only. Do not cancel an active run.
- Acceptance: native full rebuild and unit tests; two `main.py --GPU` runs and two `main.py --shortcut --GPU` runs; zero verifier failures and successful full-run Lean checks; exact theorem-set/count equality against the current baseline; byte-identical theorem and processed-proof artifacts within each repeated pair.
- Baseline: the production tree matches the validated canonical-unused implementation. Full GPU evidence is  and ; shortcut evidence is  and .
- Current state: complete and frozen after the exact-tree squash onto main; the ordering decision is D-343. The four sorts are removed; all 1,655 unit tests and all four requested GPU runs passed. The full pair matches across 1,661 files and the shortcut pair across 118 files. Theorem contents/counts equal baseline: 62 main, 1,460 incubator, and 64 shortcut. All four Lean checks passed.
- Next action: none; implementation, repeated-run validation, and the requested main landing are complete.
- Blockers: none.

## Evidence and decisions

- The prior direct `ordisMerge` test proves that the first deposit selects a capped contradiction witness. It does not prove that sorting is necessary for run-to-run determinism.
- The four sites are `logical_block/equality_propagation.hpp` and `addExprToMemoryBlockKernel`'s post-registration loop, plus `logical_block/equality_propagation.hpp` and `drainStagedEquiProducts`, `canonicalizeUnusedLists`, and `applyEquiClasses`'s `mergeProducts` lambda. Each already owns an ordered statement buffer; removal consumes that buffer directly.
- Baseline full GPU: 272.59237 seconds overall, 130.77052 seconds summed prover time, 142,755 proof checks with zero failures and Lean passed. Baseline shortcut GPU: 60.27769 seconds overall, 40.2036 seconds prover time, 9,192 proof checks with zero failures.

- Baseline theorem files were verified byte-for-byte against the current generated files before any run: 62 main theorems, 1,460 incubator theorems, and 64 shortcut theorems. Their hashes and counts are recorded in .

- Build/unit milestone:  and , success with 1,655/1,655 unit tests. The new test reaches both ordinary equivalence-product OR convergence and unused-list replacement, pins q-before-p emission without lexicographic sorting, reverses the sequence, and repeats both cases. The user chapter and all 12 embedded SVG diagrams parse; `git diff --check` passes.

- Implementation milestone is committed and pushed. The first full GPU comparison is running; each run has its own log, completion marker, complete theorem/processed-proof snapshot, and result JSON. The snapshots compare theorem sets as well as counts, and repeated pairs compare SHA-256 manifests for every captured file.

- Full GPU run 1 passed: , 282.54011 seconds overall and 140.30104 seconds summed prover time. Lean passed; 142,755 proof checks have zero failures. Main 62 and incubator 1,460 theorem files match baseline byte for byte, with no additions or losses. All 1,661 captured theorem/processed-proof artifacts match the sorted baseline. Evidence: .

- Full GPU run 2 passed: , 266.45155 seconds overall and 127.69655 seconds summed prover time. Lean passed and all 142,755 proof checks passed. All 1,661 artifacts match both run 1 and the sorted baseline byte for byte; theorem counts and contents remain 62 main and 1,460 incubator. Evidence: . The full-run determinism milestone is complete.

- Shortcut GPU run 1 passed: , 58.32244 seconds overall and 39.66750 seconds prover. Lean passed and 9,126 checks have zero failures. All 64 theorem lines are byte-identical to baseline. One of 118 captured artifacts differs from the sorted baseline: `files/shortcut/processed_proof_graph/6_check_induction_condition.txt`. Different baseline proof bytes are within the approved scope; repeat equality is checked by run 2.

## Final acceptance

| GPU mode | Run | Overall seconds | Prover seconds | Proof checks | Failures |
|---|---:|---:|---:|---:|---:|
| full | 1 | 282.54011 | 140.30104 | 142,755 | 0 |
| full | 2 | 266.45155 | 127.69655 | 142,755 | 0 |
| shortcut | 1 | 58.32244 | 39.66750 | 9,126 | 0 |
| shortcut | 2 | 54.39526 | 38.97810 | 9,126 | 0 |

- Full determinism: both runs have identical SHA-256 manifests for all 1,661 captured theorem and processed-proof files. Both also match the sorted baseline byte for byte. Main and incubator theorem files remain exactly 62 and 1,460 lines, without additions or losses.
- Shortcut determinism: both runs have identical SHA-256 manifests for all 118 captured files. Both preserve the 64-theorem baseline file byte for byte. Relative to the sorted baseline, exactly `files/shortcut/processed_proof_graph/6_check_induction_condition.txt` differs, consistently in both runs; verification falls from 9,192 to 9,126 successful checks, with zero failures and Lean passing. This is the approved change in proof selection, not a theorem-set change or a repeated-run mismatch.
- Source and execution: all four runs used the same native build. The full runs record implementation commit; the shortcut runs record documentation milestone, whose production tree is identical. Every run used `--GPU`; no CPU run or cancellation occurred. The hash-engine and class-ordering paths, capacity ceilings, assertions, and scratch-lifetime boundaries remain intact.
- Evidence: , , and the four  snapshots contain manifests, theorem files, processed proofs, diagnostics, hashburst copies, and per-run result JSON. Dedicated pipeline logs are . These two runs per mode establish repeatability for the exercised workload; no broader speedup claim follows from the observed timing spread.
- Outcome: the requested existing-order implementation and all acceptance gates are complete, based on main. This ledger is frozen.

## Squash landing

- User approval: squash onto main with the repository playbook. The reviewed source tip is; after fetching, local and remote main remain and the source equals its remote. Main is the branch base, so no destination-side changes need integration.
- User-documentation sweep: `docs/user_swdd/mechanisms.html` and its statement-order diagram accurately describe the four existing-order consumers and capped-witness selection. The page and its 12 SVG diagrams passed parsing in the validated implementation. The GPU ordering and MPU contracts are unchanged. No production edit or new pipeline run is required for this identical-tree landing.

- Final source snapshot:, preserved locally and on origin as. Squash has the sole parent and the identical source tree. The squash is pushed and the primary checkout is on main.
- Main numbering: the repository tool assigns D-343 to the existing-list-order decision and updates its declaration, cross-references, and decision high-water. This renumbering and final landing record are a separate documentation-only commit. Production source remains identical to the implementation that passed 1,655 unit tests and all four GPU pipelines. No new run was needed for the history-only landing. The task ledger is frozen.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
