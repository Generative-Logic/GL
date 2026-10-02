<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Phase 3 OR-branch exclusion and bulk cleanup

- **Objective:** exclude resolved OR subtrees from Phase 3 inference and replace repeated end-of-phase subtree wipes with one stable union cleanup.
- **Branch:**, created from `main`.
- **Approved design:** at Phase 3 entry inspect current goals at the OR branch itself and every ancestor through main. Exclude the whole subtree when that goal set is empty or every goal is known at the branch or above. Select once at entry; preserve history, convergence, release timing, and the actual end-of-phase cleanup point. Use one branch-inclusive predicate for exclusion and closure. Bulk cleanup preserves sequential wipe contents, survivor order, and scope-filter mint order.
- **Acceptance:** direct positive and negative tests; Windows Release rebuilds, C++ unit gate, matched CUDA shortcuts and full GPU runs; unchanged theorem sets and passing verifier/connected Lean checks. Inference may change artifacts with a complete difference report; bulk cleanup alone must be byte-identical. Two final full runs must reproduce complete artifact manifests. Measure the two improvements separately.
- **Constraints:** heap-free production paths, unchanged assertions and capacities, no discarded work; main landing authorized on 2026-09-08 after independent validation. Dump disabled only through the `_RT` unmatched leaf target with its full parent chain intact. Source/configuration milestones are complete snapshots committed and pushed separately.
- **Current state:** LANDED on main. Exact-tree squash is pushed, with source preserved. Independent full-run acceptance is complete: identical source/executable and all 5,565 artifacts, archive members, theorem sets and verifier rows; 152,809 checks and connected Lean pass. Pending documentation identifiers are assigned D-359/D-360 and I-232/I-233 in the separate documentation commit.
- **Next action:** none; landing is complete and this ledger is frozen.
- **Blockers:** none. The preserved baseline FTA and first landing-acceptance IncubatorPeano2 access violations remain unexplained; successful retries are not represented as crash fixes.

## Milestones

1. Retarget the hashburst dump, establish the baseline, and commit and push the complete snapshot.
2. Implement branch-inclusive Phase 3 exclusion with direct tests and design documentation, measure and verify it, then commit and push.
3. Implement union subtree cleanup with exact sequential-oracle tests, complete repeated full acceptance and documentation, then commit and push.

## Evidence

- Exclusion build `ordis_ignore_exclusion_build_03`: Release rebuild passes in 114.280 seconds and 1,692/1,692 C++ tests pass. The two preceding unit attempts exposed test-fixture omissions (equality provenance, then own-scope class placement); correcting those fixtures leaves production assertions unchanged. No investigation-only production traps were introduced.
- Exclusion shortcuts `ordis_ignore_exclusion_shortcuts_shortcut_01` / `_02`: 60.95967 / 61.20865 seconds overall, 46.5278 / 46.8265 seconds proving, 16.7328043 / 16.9895123 seconds Phase 3. Both retain the 127-theorem pool and all 26,848 checks. The 950-file manifests and every verifier row equal each other and the baseline. Mean overall shortcut time falls from 63.393245 to 61.08416 seconds. Reports use the corresponding  names.
- Exclusion full run : 306.14702 seconds overall, FTA proving 46.0583 seconds, connected Lean accepted, 152,809 checks / zero failures. All 5,565 artifacts, archive members, theorem sets and verifier rows equal the baseline exactly. The full overall time increased despite the FTA improvement, so no whole-pipeline speedup is claimed for exclusion. Report: .
- Architecture documentation and user/MPU descriptions are updated. On-sight cookbook corrections align identifier widths, statement-membership semantics, owner records, paging, worker-slot references and an allocation example with the current source.

- Unchanged retry uses executable SHA-256 `67e63890c9bf8b8a971aae76253634abed6abc75dad4e1da4ec0138af2430772`.  reports 298.72710 seconds overall, FTA prover 48.6216 seconds, 152,809 checks / zero failures, and connected Lean kernel acceptance. The captured full artifact manifest contains 5,565 files / 800,225,142 bytes. Shortcuts `ordis_ignore_baseline_retry_shortcut_01` and `_02` report 63.13234 / 63.65415 seconds overall and 48.4944 / 48.9216 seconds proving; both pass 26,848 checks and reproduce every entry in their 950-file shortcut manifests. Every run leaves the hashburst trace empty.
- Source review identified queued OR seeds that have not yet entered `orLiveBranches`; the exclusion snapshot must classify existing NameMap scopes, not only that live set. Direct equivalence-class products and descendant admission entries also need scope filters beyond the ordinary deposit door. These are implementation details of the approved exclusion, not a scheduling change.

- Baseline build `ordis_ignore_baseline_build_01`: full native rebuild succeeds in 117.886 seconds; 1,682/1,682 C++ tests pass. The first full baseline, , exits after 254.349 seconds with native FTA exit code 3221225477. Its diagnostics, traps and zero-byte hashburst trace are preserved under . Inference and cleanup source still match main.

- The investigation / `docs/fta_ladder/D1/runtime_investigation.md` is supporting evidence, not an implementation constraint. Its saved dump-off shortcut is 62.8103151 seconds with 26,848 passing proof checks and 967 captured output files;  confirms the dump-off output identity. New matched runs establish this task's baselines.

## Verification

All builds use native Windows Python environment normalization and MSBuild `-t:Rebuild`. Pipeline logs live directly under `.debug` with unique names. Each long command has one completion-driven watcher; logs are inspected after completion.

```powershell
& C:/Users/nikol/anaconda3/python.exe main.py --shortcut --GPU
& C:/Users/nikol/anaconda3/python.exe main.py --GPU
```

Pass criteria are the acceptance conditions above plus a zero-byte  after normal startup truncation.

- Bulk implementation uses an earliest-covering-root rank for each closed identifier, avoiding the incorrect global identifier sort. Four tests compare all canonical columns and persistent goals against sequential calls and cover overlaps, repeated roots, page crossings, high-identifier aliases, saved levels and empty runs. The initial bulk build passes 1,696 tests; final production build validation follows.
- Shortcut captures contain the input file `full_input.zip`. `fta_shortcut.capture` writes current ZIP timestamps, so a new full run changes its container bytes even when every input member is identical. Comparison reports retain that raw difference separately, verify every member hash, and require all generated proof files to remain byte-identical; full-run acceptance already compares archived input members separately from its 5,565 proof artifacts.

## Final outcome

- Final production rebuild `ordis_ignore_final_build_01` passes in 101.673 seconds; all 1,696 C++ tests pass. No temporary production instrumentation or capacity/configuration changes remain.
- Bulk shortcuts `ordis_ignore_bulk_shortcut_01` and `_02` take 58.49267 / 58.15897 seconds overall, 45.3406 / 44.9342 seconds proving, and 15.9217928 / 15.9015091 seconds in Phase 3. Their proof files, input members, theorem sets and verifier rows are identical to exclusion; their complete 950-entry captures also equal each other. The separately regenerated input ZIP differs from the earlier exclusion capture only in container metadata.
- Mean shortcut comparison (baseline → exclusion → bulk): overall **63.393245 → 61.08416 → 58.32582 seconds**; native proving **48.708 → 46.67715 → 45.1374 seconds**; Phase 3 **20.47855365 → 16.8611583 → 15.91165095 seconds**. These are matched shortcut measurements, not a claim of unchanged proof work. Summary: .
- Final full run  passes in **294.50140 seconds**, with FTA proving **43.5976 seconds**, connected Lean acceptance, and **152,809 proof checks / zero failures**. All **5,565** captured artifacts, input-archive members, theorem sets and every per-check verifier row match exclusion exactly, which already matches baseline. Report: . The hashburst trace is empty.
- At initial delivery the second independent final full repeat was **not run** after the maintainer said “Come to an end.” The later main-landing request resumed and completed that repeat, as recorded below.
- At initial delivery the updated OR and cleanup SVG diagrams parsed successfully and source/diff checks passed; main was not advanced until the subsequent authorized landing below.

## Main landing acceptance

- Authorization: the maintainer requests the squash playbook to `main`, superseding the earlier delivery-only boundary and resuming the outstanding independent final full repeat.
- Ref refresh confirms local and remote main and sandbox. The sandbox is four commits ahead and has no pre-existing working changes.
- Exact executable SHA-256 remains `baf74b129a4c63ed29ad9c6d05b66578d8a6a8133d25639da8710b3b3c4aa1f7`; all 274 files in the final run's source manifest match current files. Reuse this fully rebuilt executable for the independent run; no implementation changes or new build are required.

- First landing-acceptance attempt `ordis_ignore_main_acceptance_full_01` exits after 65.329005 seconds in IncubatorPeano2 with native access violation 3221225477. Diagnostics and traps are preserved beside its source/input capture. No source changes preceded this attempt. Main remains unchanged; one unchanged retry is started. No common root cause with the earlier baseline FTA access violation is asserted.

- Unchanged retry `ordis_ignore_main_retry_full_01` passes in 311.39049 seconds with connected Lean and 152,809 checks / zero failures. Exact comparison against `ordis_ignore_final_full_01` verifies the same executable and all source hashes, all 5,565 artifact hashes, input archive members, theorem sets and every verifier row. The earlier missing final repeat is now completed. Report: . The failed first attempt remains preserved and is not claimed fixed.

- Landing completed: has the previous main as its only parent and tree, exactly equal to sandbox source. The source is preserved locally and remotely. The checkout is on main; the following documentation-only commit assigns D-359/D-360 and I-232/I-233 and records closeout.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
