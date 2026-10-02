<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Compressor acceleration

## Current state

- Status: complete and frozen for the user's approved exact-tree squash onto `main`. Implementation milestones and verification remain recorded below; the landing record is .
- Objective: reduce compressor rule-installation and premise/goal-seeding time, preserving exact output. `buildStack` remains outside scope.
- Acceptance: direct index-equivalence coverage for batched rule loading; a measured setup reduction on a full CUDA run; unchanged output hashes, elimination decisions, verifier results, and passing native tests and Lean kernel verification.
- User approval: performance work on this named branch. Changes to theorem selection, provenance, scope handling, or proof-graph semantics require a separate design approval.
- Constraints: full Release rebuilds with normalized Windows environment; pipeline execution through `main.py --GPU`; preserve assertions and existing diagnostics; one completion watcher per long command; retain separate logs and artifact manifests.
- Measured state: matched full CUDA runs reduced combined compression from 24.4756505 to 5.6829667 seconds (4.31 times faster, 76.78 percent less time). The complete pipeline measured 305.4530426 to 287.6844819 seconds; this single pair includes ordinary variation in unchanged stages.
- Implementation: parse each theorem once before independent rule installation; index each extracted graph by full expression and scope, and evaluate its least fixed point through reverse dependency links instead of repeated whole-graph string-set scans. Candidate ordering and the obligation to derive every extracted head remain unchanged.
- Next action: none in the implementation. The approved landing uses the committed source tree with one parent at the existing main tip; its final branch, tree, and renumber verification are recorded in the landing artifact and squash commit.
- Follow-up requested by the user: after setup optimization, inspect the compressor prover's current CPU/GPU route and measure the remaining opportunity for GPU execution within its roughly one-second duration. Preserve this order; no GPU change is part of the setup milestone.
- Scope cleanup: the user explicitly approved removing the two uncommitted chapter-walker optimizations. They have been removed; no chapter-walker source changes remain.

## Evidence and completed steps

- 2026-09-06: created the requested branch from the clean, fetched main tip. No production changes or measurements yet.
- Baseline evidence: , , , and . The full Release rebuild and pipeline exited zero; 1,637 native tests and 407 verifier unit tests passed; the generated Lean project passed its kernel check. Compressor decisions: Peano kept 37 of 130; Gauss kept 62 of 73.
- The initial completion-watcher transport timed out at 300 seconds; its kernel was reset and the watcher reattached. The replacement received the completion record. No active-run log polling was used.
- Source review found stale compressor documentation claiming that only surviving heads were checked, usage ties followed insertion order, and an optimizer-disabling pragma was active. Current source checks every extracted head, breaks ties by theorem text, and contains no active pragma; these documentation claims are being corrected alongside the optimization.

## Final verification

| Full CUDA stage | Baseline seconds | Optimized seconds | Speedup |
|---|---:|---:|---:|
| Peano compressor | 20.0558215 | 4.3119618 | 4.65 times |
| Gauss compressor | 4.4198290 | 1.3710049 | 3.22 times |
| Combined compressor | 24.4756505 | 5.6829667 | 4.31 times |
| Complete pipeline | 305.4530426 | 287.6844819 | 1.06 times |

- Optimized run: `rt_parts_optimized_20260906`, full Release rebuild and `main.py --GPU --run-descriptor rt_parts_optimized_20260906`, both exit zero. The same build flags, configuration, inputs, and Windows checkout were used for the baseline.
- Native unit tests: 1,640/1,640, including three new direct reachability tests. The generated oracle test covers 256 graph shapes times 16 unavailable-theorem sets. Verifier unit tests: 407/407.
- Full verification: 141,136 checks, zero failures; generated Lean project passed its kernel check. Verifier result rows exactly match the baseline.
- Exact comparison: all 3,490 captured files have identical SHA-256 hashes, with no additions or removals. The manifest covers the main and incubator theorem files, raw proof graphs, processed proof graphs, and GL binaries. Tracked files under `files/` have no changes. Every compressor elimination-pass decision matches the baseline, including the same 84,906 Peano and 53,273 Gauss extracted origin entries.
- Optimized phase timings: Peano Phase 1 total 4.23893 seconds and greedy elimination 0.0696469 seconds; Gauss Phase 1 total 1.33489 seconds and greedy elimination 0.0327878 seconds. These Phase 1 totals include preparation, proving, extraction, and cleanup; they are not isolated graph-extraction measurements.
- Evidence: , , , , and .
- No temporary production instrumentation was introduced. Two coarse compressor phase durations remain as permanent run-log observability. The chapter-walker source is unchanged after the user-approved removal of its two experimental optimizations.
- Documentation: updated the compressor implementation description and user-facing diagram; corrected the stale compressor and historical chapter-runtime claims read during the investigation. The new diagram parses as valid SVG. Final static verification: `git diff --check` passed.

## Remaining cost and multithreading question

The user asked whether multithreading is possible and whether hash bursts now
dominate. The optimized run's 15 logged compressor iterations sum to 0.8177007
seconds for Peano and 0.3301327 seconds for Gauss: 1.1478334 seconds combined.
Phase 1 totals 5.57382 seconds, leaving 4.4259866 seconds outside those measured
iterations. This remainder includes preparation, extraction, and other work
outside the iteration timers; those parts have not been timed separately.
Phase 2 totals only 0.1024347 seconds.

The prover kernel already schedules independent logic blocks across workers.
Preparation and graph extraction are prospective parallelization targets, but
should be timed separately first. Preparation currently uses the reserved
single-threaded scratch slots through `addToHashMemory`; parallel workers would
need distinct worker slots and an audit of shared initialization and compiler
state. Extraction writes one graph per logic block and is a narrower candidate.
Parallel candidate removals would also have to preserve the existing greedy
order. No multithreading change was made in response to this feasibility question.

## Phase 1 attribution follow-up — complete

Full native CUDA run `rt_parts_phase1_profile_20260906`:

| Exclusive Phase 1 stage | Peano seconds | Gauss seconds | Combined seconds |
|---|---:|---:|---:|
| Parse theorems | 0.0003135 | 0.0002838 | 0.0005973 |
| Create logic blocks | 0.0020768 | 0.0010142 | 0.0030910 |
| Install rules | 2.4819100 | 0.5215820 | 3.0034920 |
| Register rule statements | 0.0180220 | 0.0058164 | 0.0238384 |
| Seed premises and goals | 0.7500620 | 0.4095010 | 1.1595630 |
| Complete prover call | 0.7113320 | 0.3424270 | 1.0537590 |
| Decode origins | 0.0381762 | 0.0247262 | 0.0629024 |
| Assemble graphs | 0.0299235 | 0.0173914 | 0.0473149 |
| Release decoded snapshots and extraction overhead | 0.0025195 | 0.0016606 | 0.0041801 |
| Destroy logic blocks | 0.0251296 | 0.0106595 | 0.0357891 |
| Remaining orchestration | 0.0010449 | 0.0003779 | 0.0014228 |
| **Phase 1 total** | **4.0605100** | **1.3354400** | **5.3959500** |

- Result: rule installation is 55.66 percent and premise/goal seeding 21.49 percent of Phase 1; together 4.163055 seconds, or 77.15 percent. Complete proving is 19.53 percent. All graph extraction, including snapshot release, totals only 0.1143974 seconds. The unattributed remainder is 0.0264 percent of Phase 1.
- The old `dt=` lines measure `proveKernel` only. They sum to 1.042844 seconds here; the complete `prove` calls take 1.053759 seconds. Deactivation sweeps, steward startup/shutdown, and other work outside those iteration timers account for only 0.010915 seconds, not the previous multi-second remainder.
- Source explanation: each of 130 Peano blocks installs all 130 theorem rules; each of 73 Gauss blocks installs all 73. Those explicit loops therefore perform 22,229 top-level installations. Each standalone `addToHashMemory` closes its outermost staging window through `closeRuleIndexStaging`. The timer includes span preparation and the complete installation, including that flush; it does not separately attribute the flush share. Seeding separately calls `addExprToMemoryBlock` for every premise and goal.
- Recommendation boundary: setup is the measured next target. Batching existing staging windows or distributing independent block setup across worker slots are candidates to investigate; neither has been implemented or measured. Graph-extraction parallelism has little available time to recover on this workload. Shared initialization, compiler state, and deterministic ordering still require review before a threading change.
- Verification: normalized native Release rebuild exited zero; full `main.py --GPU` exited zero in 278.4553365 seconds (`run.total`, rebuild excluded). Native tests 1,640/1,640; verifier unit tests 407/407; full verification 141,136 checks with zero failures; Lean kernel check passed. All 3,490 captured files, every elimination decision, and every verifier result row match `rt_parts_optimized_20260906` exactly.
- Timing interpretation: this change adds observability only. The whole-run difference from the preceding 287.6844819-second run is ordinary run variation, not a claimed additional acceleration. Combined compressor time was 5.5099604 seconds, including 0.1055061 seconds in Phase 2.
- Evidence: , , , , , and .
- Instrumentation status: the permanent compressor report now contains these component times. Timers wrap serial orchestration and do not enter statified inference functions; no inference, provenance, or invocation order changed. `buildStack` remains unchanged.

## Setup optimization — complete

One existing staging window now covers each block's complete rule load, and a
second covers its premise/goal deposits. The rule load closes before seeding;
seeding closes before proving. Input and mint order within each block remain
unchanged. The direct regression compares all seven index maps in each of the
four hash memories and decoded origin rows against the per-install-flush path,
before and after seeding; both paths finish with empty staging and zero depth.

| Combined Peano and Gauss stage | Before seconds | Batched seconds |
|---|---:|---:|
| Rule installation | 3.003492 | 1.121936 |
| Premise/goal seeding | 1.159563 | 0.485693 |
| **Targeted setup** | **4.163055** | **1.607629** |
| Complete prover calls | 1.053759 | 0.748569 |
| Phase 1 total | 5.395950 | 2.523701 |
| Complete compressor | 5.5099604 | 2.6396070 |

- Targeted setup improved by 61.38 percent; complete compression improved 2.09 times on this matched pair and 9.27 times against the original 24.4756505-second baseline. Prover-call timing also varied; no separate inference optimization is claimed.
- Full run `rt_parts_setup_batched_20260906`: normalized Release rebuild passed; `main.py --GPU` passed in 273.675187 seconds (`run.total`, rebuild excluded). Native tests 1,641/1,641, verifier unit tests 407/407, full verification 141,136 checks with zero failures, and Lean kernel check passed.
- Every one of the 3,490 captured files is SHA-256 identical to `rt_parts_phase1_profile_20260906`; elimination decisions and verifier result rows are identical. Tracked `files/` artifacts and `visualizer.cpp` have no changes.
- Evidence: , , , , , and .
- The hash-engine chapter, I-221 entry and quick reference, compressor chapter, and user-facing setup diagram now include these two windows. The diagram parses as valid SVG; `git diff --check` passed. No temporary diagnostics or new parallel execution were introduced.

## Exact-tree landing contract

- User approval: squash down to `main` using the GL squash playbook; finish on clean `main` and preserve the sandbox locally and on origin.
- Refreshed topology: destination matches origin/main and is an ancestor of the reviewed source. The source also matches its remote. The final source preparation adds only this landing record and user-document corrections; production source and tests remain the verified implementation.
- Required user-document sweep: corrected the dependency graph's description to include expression scopes and alternative dependency lists, and corrected the origin-cap wording. Retaining the first recorded origins does not by itself prove that their dependency graph is acyclic. The compressor setup and reachability diagrams otherwise match the implemented behavior.
- Acceptance: one-parent squash whose tree exactly equals the final committed source tree; destination advances from its expected old tip; sandbox refs remain at the final source tip; SwDD renumber dry run and actual invocation complete on main, with any changes in a separate commit; primary checkout ends clean on main, matching origin/main.
- No rebuild is required for this history operation: the source/test content matches the completed full-run gate. Documentation syntax and `git diff --check` are checked before landing; exact tree identity transfers the validated implementation without a merge reconstruction.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
