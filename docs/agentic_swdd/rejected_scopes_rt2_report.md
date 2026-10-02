<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# IncubatorGauss3 runtime investigation

## Post-landing shortcut verification: complete

User request: run the complete shortcut pipeline on main. Source: clean; command `main.py --shortcut --GPU --run-descriptor main_shortcut_20260905_153728`. Retain the current configuration, including enabled runtime measurement. No source/configuration edits or fixes are requested.

Acceptance: full native Release rebuild using the normalized Windows environment, then the full shortcut entry point including unit, verifier, and Lean gates. Preserve dedicated build/run logs, diagnostics, hashburst, and runtime aggregates; inspect only after completion. No `gl_quick` process was running at the initial pre-build check. Full Release rebuild passed in 109.140 seconds (). The complete `main.py --shortcut --GPU` invocation completed with exit 0. Its reported overall runtime is 74.78116 seconds (wrapper launch-to-exit 74.94705 seconds, rebuild excluded); prover runtime is 52.1965 seconds. All 1,637 C++ tests and 407 verifier unit tests passed; Lean kernel export passed; the shortcut verifier completed 9,217 checks with zero failures. The native run reports 39 proven theorems. Runtime measurement remained enabled, matching the requested main snapshot.

Evidence: , its `.completion.json`, , archived diagnostics/hashburst/traps prefixed `main_shortcut_20260905_153728`, and . No source/configuration edits or fixes were made; this result entry is the only tracked change. Verification is complete.

## Two-stage squash landing: complete

User request: squash into its parent, then squash that parent into `main`, using the GL squash playbook. Preserve every source branch and its original history; no force push or history deletion. Final checkout: `main`.

Reviewed starting references (all match freshly fetched origin): source, parent, main. Both destination tips are ancestors of their sources and the working tree is clean. Land exact source trees with `commit-tree`, verify tree identity and expected parents, advance destinations with old-value-checked `update-ref`, and push each destination before the next stage. Renumber pending SwDD entries only after landing on main, in a separate commit.

The complete source snapshot is intended: accepted ancestor-park release and exact zoom-in history/export/verifier/Lean support, the stable statement-vector compaction, and the current enabled runtime instrumentation. The canonical-only unused-expression idea remains an unimplemented proposal. No code or configuration cleanup is inferred from the squash request.

Pre-main user-SwDD sweep: `mechanisms.html` and `provenance.html` describe the current ancestor release, park retention, exact zoom-in source/target, and exporter behavior, with matching diagrams. Compaction changes copying cost without changing the represented scope lifecycle; timing additions and the unimplemented proposal require no diagram change. Their HTML parses, SVGs are well-formed, and local anchor links resolve. No source change since the successful full Phase 3 profile; that run passed 1,637 C++ tests, 407 verifier tests, 141,136 proof checks, and Lean, with 3,496 artifact identities preserved. No new prover run is needed for an exact-tree landing and documentation-ID substitutions.

Both exact-tree landings completed and were pushed in the requested order:

| Stage | Source snapshot | Destination squash | Previous destination |
|---|---|---|---|
| RT2 to parent | | | |
| Parent to main | | | |

Both squash trees equal. Each commit has exactly one parent, the previous destination tip. is preserved; is preserved, whose ancestry retains the entire original rejected-scopes development history. The primary checkout is now main.

The separate main renumber promotes the five pending entries to D-341/D-342, I-223/I-224, and G-74 and updates the navigation high-water marks to D-342, I-224, G-74. Only documentation/comment identifiers and this receipt change after the exact-tree squash; no executable behavior changed and the already-passing full-run verification remains applicable. Evidence receipts are  and . Final references and clean status are checked after the renumber commit is pushed.

## Current Phase 3 costs measured; canonical-only design under review

The user requests the concrete costly operations within the remaining Phase 3 time and whether implementation costs can be reduced. Diagnose the post-compaction source; no fixes. Enable the existing runtime aggregate and add timing-only subdivisions for disintegration history, instruction renaming/recursion, rejected-record snapshots/writes/fired equality pairs, and resent-registry resets. Reuse the archived exact EnumerationSet3 chain trace and retain the normal anchor dump for timing comparability. All inputs, semantic operations, and assertions stay unchanged.

The full profile completed successfully. It measured 22.6764 seconds for IncubatorGauss3, including instrumentation and this run's host conditions; this must not replace the verified uninstrumented 17.1672-second result. Phase 3 wall was 15.9487 seconds; its aggregate was 16.59 worker-seconds (1.04x parallelism). All 1,637 C++ tests, 407 verifier tests, 141,136 proof checks, and Lean passed. The 3,496 checked artifacts still match the verified compaction output by length and SHA-256.

| Non-overlapping Phase 3 category | Inclusive worker-seconds |
|---|---:|
| Internal statement absorption, `SP_ABSORB_STMTS_INT` | 8.00 |
| Equivalence-class application, `SP_APPLY_EQUI_CLASSES` | 5.85 |
| Expression cleanup, `SP_CLEANUP_EXPRESSIONS` | 1.61 |
| Scope wipe, `WIPE_SUBTREES` | 0.09 |

The absorption category contains 4.18 seconds in `disintegrateExpr2` (`DOOR_DISINTEGRATE`), 1.17 in entrance `canonicalFormAtScope` (`DOOR_CANONICALIZE`, 104,387 calls), and 1.34 in the product-statement walk. Inside disintegration, the core issues 311,000 local history writes and 311,000 mail-history writes (about 0.51/0.53 seconds inclusive respectively), builds 103,058 renamed existence instructions (0.28 seconds), and performs 206,116 existence-child recursions (0.19 seconds exclusive). Pass A/B outside its new nested writers contributes 1.04 seconds. These child figures must not be added to their containing categories.

Equivalence application contains 1.25 seconds in the rejected-algebra hook, 0.83 in rejected integration, 0.99 in the main admission-algebra hook, and 0.54 in compact-implication handling. The encoded-statement batch itself costs 0.04 scan + 0.61 closure + 0.63 commit/convergence = 1.28 seconds. The 0.63 includes 0.46 in `ordisMerge` convergence. Rejected-record snapshot copying is about 0.01 seconds, and fired-pair construction about 0.14 across the two main hooks; neither is a large target. Resent-registry reset costs 0.27 seconds in the principal rejected-algebra path. Existing cleanup scans total 596,222 local rows, 291,465 local-delta rows, and 626,273 overall rows, with their three filter scopes taking about 1.39 seconds.

**Reduction boundary for the user's proposal:** skipping equivalence application only for unusable encoded expressions targets some of the 1.28-second statement batch and the 1.61-second cleanup, plus any secondary effect from storing fewer variants. It does not directly remove the rejected/admission-map work, and the 4.18-second disintegration work already follows entrance canonicalization. `drainStagedEquiProducts` registers equivalence variants directly; it does not run the full absorption/disintegration door on each one. Therefore the entire 8.00-second absorption category cannot be advertised as savings from this guard alone. These are measured work budgets, not a tested speedup prediction.

Evidence: , , , and . Timing-only additions remain on the branch; no additional performance or semantic fix has been implemented.

### User-proposed canonical-only encoded-expression policy (analysis)

The user clarified the condition as AND: a core predicate, considering both polarities, must be neither usable for request generation nor represented among goals. Reuse the canonical form already built at `addExprToMemoryBlock` entry. Store only that canonical form in the encoded-expression collections (internal/local, external, and overall), and skip equivalence-class application to those unusable encoded expressions. Rejected maps are explicitly excluded from the skip; their existing canonical re-keying and other processing remain intact. The earlier rejected-map suggestion was withdrawn. No semantic implementation has been made.

Check the legitimacy of the added expressions and their request/goal relevance, and quantify which measured Phase 3 work this policy can actually avoid before claiming savings. Any correctness edge must be reported against this exact user-specified scope, not silently solved by a different design.

## Remaining runtime: traced workload multiplier

Read-only follow-up; no new prover run, instrumentation, or code fix. The remaining runtime is dominated by the construction and processing of a near-Cartesian set of addition/multiplication existence statements in the two enlarged interval or-integration scopes. The compaction removed their quadratic deletion cost, while preserving the generation and reasoning work.

The traced chain is:

1. `ExpressionAnalyzer::releaseParkedProductIntoScope` in [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) offers both the parked operator fact and its siblings at the demanding descendant scope. The trace records `zoom in` on natural-number typing siblings such as `(in[it_0_lev_1_304,1])`, sourced from the interval subproof scope.
2. Those typed witnesses join local witnesses from disintegration. At ENTRY #13, the equality-to-6 and equality-to-7 scopes contain 119 and 122 explicit `(in[t,1])` rows respectively, including 12 and 13 with `zoom in` origins. This is term-name growth, not a claim of 119/122 mathematically unequal numbers; the scopes also contain equivalence classes.
3. The totality clause of [`fXYZ.mpl`](../../files/definitions/fXYZ.mpl), instantiated for `+` and `*` by [`NaturalNumbers.mpl`](../../files/definitions/NaturalNumbers.mpl), takes two natural-number typing premises and derives an operation-result existence statement. The saved origin rows explicitly cite this rule and the two typing facts.
4. Before the expensive burst, these existence statements already cover almost every ordered pair of the operand-name sets shown below, for both operation slots 4 and 5. They trigger the observed disintegration, history writing, and equivalence-class work before the scopes are eventually closed.

| Scope branch | Operand names represented in each input slot | Addition + multiplication existence statements | Full two-operation Cartesian product |
|---|---:|---:|---:|
| `(=[6,repl_lev_1_5])` | 80 | 12,530 | 12,800 |
| `(=[7,repl_lev_1_5])` | 83 | 13,504 | 13,778 |

Both actual counts are about 98% of the Cartesian-product size. In the equality-to-7 scope, 13,468 stored origin rows for those statements have tag `implication`; only 81 have tag `equality1` (an expression can have multiple origins). The equality-to-6 counts are 12,496 `implication` and 53 `equality1`. The dominant recorded producer is therefore the two-premise arithmetic closure rule, not simply equivalence-class multiplication of an existing existence statement. Equivalence processing still consumes substantial time downstream.

One exact recorded production is `(existence1[1,2,it_0_lev_1_506,4])` from `(in[2,1]) @ main` and `(in[it_0_lev_1_506,1])` at the equality-to-7 scope. In the compiled binary, `existence1[N,a,b,f]` represents an existential result `z` with `(in[z,N])` and `(in3[a,b,z,f])`. This exposes the pairwise growth directly; it is not inferred merely from matching statement counts.

The current measured Phase 3 wall is **11.6774 seconds**, versus **2.1949 seconds** on main: **9.4825 seconds** of the **12.9042-second** total remaining gap. Phase 2 is 3.3873 versus 1.5941 seconds; Phase 1 is 1.1707 versus 0.2718 seconds. The saved detailed profile placed the non-wipe work primarily in internal statement absorption/disintegration and equivalence-class application. Those detailed labels are pre-compaction measurements; only the phase totals above are from the fixed run.

**Boundary:** this establishes the causal workload mechanism and its dominant producer. It does not establish that every revived witness is required, that every release is unnecessarily permissive, or which admission/canonicality change could suppress the expansion while preserving the approved ancestor-scope semantics. The next diagnostic target would be the demand provenance for the typing-bearing park releases, plus whether equivalent operand names are redundantly admitted. No such semantic fix is proposed as already proven.

Evidence: , , and the saved exact-chain trace. Reuse of that trace is justified by the compaction's unchanged scope behavior, identical burst states, and byte-identical 3,496-file proof-artifact comparison. Follow-up finding recorded; code remains unchanged.

## Approved implementation: complete

Implemented and verified, production change. The three statement-vector wipes in `Memory::wipeSubtree` now use stable forward compaction with the same scope predicate, survivor order, whole-record bytes, dirty-state contract, and page reclamation. Each kept row moves at most once. No heap storage, additional persistent state, scope semantics, history behavior, assertions, or GPU logic changed.

| Matched full CUDA run | Before | After |
|---|---:|---:|
| IncubatorGauss3 prover | 30.4387 s | 17.1672 s |
| Phase 1 wall | 1.2167651 s | 1.1707059 s |
| Phase 2 wall | 3.4562788 s | 3.3873115 s |
| Phase 3 wall | 24.8125290 s | 11.6773946 s |
| Burst 10 wall | 20.312 s | 7.88969 s |

Savings: **13.2715 seconds / 43.6%**, or **1.773x speedup**, in this matched measurement. Phase 3 accounts for 13.1351 seconds of the reduction. Both runs use the normal runtime gate (`RT_MEASUREMENT = 0`) and the same pre-investigation exact anchor hashburst target/path. Every burst retains the same active-block and expression counts. The remaining runtime includes the intentionally retained ancestor-release workload; this change does not restore the old main's scope behavior.

**Artifact identity:** all **3,496 checked files / 42,153,814 bytes** match path, length, and SHA-256, with zero added, removed, or changed files. Coverage: 358 main raw proof files, 176 main processed proof files, 1,463 incubator raw proof files, 1,464 incubator processed proof files, 21 theorem files, eight compiled binaries, and six Lean-source/certificate files. Both JSON manifests are byte-identical; manifest SHA-256: `e4808ec45837d2ab653d86f7f6e4d9b972fb109f62226ba549b79cd10e23e16f`. HTML and Lean build-cache/log files are outside this manifest.

**Validation:** full rebuild passed; all **1,637 C++ unit tests**, **407 verifier unit tests**, **141,136 proof-graph checks**, and the **Lean kernel export** passed. The two new direct tests cover interleaved closed/descendant/main/prefix-sibling scopes, three differently populated vectors, whole-record identity and ordering across pages, exact retained data-page counts, restructured state on removal, and empty/all-kept/all-removed inputs.

**Separate failed attempt:** the first fixed run hit the unchanged CUDA projection metadata-capacity assertion in the first IncubatorPeano1 warm-up Phase 2, before the changed Phase 3 wipe was reachable. The source and capacities were left unchanged; a full rebuild and identical-source retry passed. The failure remains documented below and in its archived logs. Its exact cause remains unresolved; no CUDA capacity or scheduling fix is claimed.

Temporary investigation timing annotations/counters are removed and the normal gate and sacred dump target/path restored under the approved cleanup. All prior evidence is retained. The current executable and live generated artifacts are from the verified fixed run. Implementation and verification are complete; this report is frozen.

Evidence: ; ; ; immutable artifact folders ending `_outputs`; the corresponding `run_*.log`, diagnostics, and completion records.

## Diagnosis results (historical)

The slowdown is a transient expansion inside the existing EnumerationSet3 logic block, compounded by quadratic scope cleanup. Ancestor park release admits facts into two or-integration scopes; downstream reasoning expands those scopes, and the unchanged wipe deletes their rows one by one. That deletion copies 919,147,513 statement records over the batch and costs 12.68 seconds in the detailed measurement. The ancestor traversal and zoom-in history writes themselves are small.

| Measurement | Clean main reference | Investigation profile 1 |
|---|---:|---:|
| Source checkpoint | | |
| IncubatorGauss3 prover | 4.263 s | 31.586 s |
| Phase 1 wall | 0.272 s | 1.379 s |
| Phase 2 wall, including setup and host work | 1.594 s | 3.622 s |
| Phase 3 wall | 2.195 s | 25.621 s |
| Burst 10 input expressions, whole batch | 9,173 | 37,085 |
| Burst 10 wall | 0.662 s | 21.196 s |
| Peak pool blocks | 767 | 4,176 |

Both are complete native Windows `main.py --GPU` runs with identical IncubatorGauss3 configuration, 1,452 loaded proved theorems, and 47 loaded external theorems. Profile 1 enables timing and retains the pre-existing anchor dump; main is uninstrumented. This is a fresh reference comparison, not an instrumentation-matched speed ratio. The earlier uninstrumented investigation parent recorded 34.3672 seconds with exactly the same expression-count sequence. Profile 2 isolates the cleanup mechanism but its 2.1 GB diagnostic dump raises total time, so it is not the representative runtime.

Created from clean on 2026-09-05. The user authorized instrumentation, hashburst redirection, and a main reference, while requiring diagnosis only. No fixes were implemented. Prover semantics, configurations, assertions, verifier, and hashburst section writers remain unchanged. Timing annotations and the exact enumeration-chain dump target remain on the investigation branch.

Recommended next work, requiring a separate implementation task: stable forward compaction of the three wiped statement vectors, retaining the current membership predicate and survivor order. This targets roughly 12–13 seconds of measured work; it does not establish a return to 5 seconds because the new scope work also consumes time outside cleanup. Detailed evidence and limits follow. Investigation complete; report frozen. The investigation branch and its executable are restored, and the instrumentation remains enabled for follow-up work.

## Evidence ledger

- Initial working directory clean; requested branch created without changing HEAD.
- Existing hashburst target already selects the exact Incubator3 anchor chain. Archived full trace is 28,898,405 bytes; it will be preserved.
- Runtime measurement was disabled (`RT_MEASUREMENT = 0`), with a five-second per-call trigger. Aggregate profiling supports existing split configuration.

- User additionally authorized running `main` as a fresh reference. Its current tip is. IncubatorGauss3 configuration has no difference from the investigation parent.
- Archived uninstrumented burst 10: phase 1 = 0.304233 seconds, phase 2 = 0.977905 seconds, phase 3 = 21.4059 seconds; the measured GPU device work within phase 2 is 0.287891 seconds. The extra work is therefore primarily in phase 3.
- Instrumentation milestone committed and pushed. Full Release rebuild passed in 100.894 seconds. Complete CUDA profile run started with descriptor `rejected_scopes_rt2_profile1`; output is inspected only after completion.

## First detailed profile

- Complete CUDA pipeline: exit 0; 324.931 seconds overall; 1,635/1,635 C++ and 407/407 verifier unit tests; 141,136 proof-graph checks, zero failures. IncubatorGauss3 prover = 31.5858 seconds, with exactly the saved branch expression counts.
- Measured phase walls: phase 1 = 1.3793433 seconds, phase 2 = 3.6224638 seconds, phase 3 = 25.6209022 seconds. Runtime aggregate: phase 3 = 26.30 worker-seconds / 25.62 wall-seconds (1.03 parallelism), so the long work is nearly serial.
- Largest aggregate path: `END_OF_BURST_SANITIZE > WIPE_SUBTREES`, 13.45 exclusive worker-seconds (11 calls). Internal statement absorb consumes 5.96 inclusive worker-seconds; equivalence-class application 4.75; expression cleanup 1.31. These are non-overlapping categories.
- The triggered per-call report identifies the exact chain: root -> `(AnchorIncubator3[1,2,3,4,5,6,7,8,9])` -> `(EnumerationSet3[2,6,7,10])`. At its hashburst index 10, phase 3 lasts 19.32 seconds, including 12.86 in the scope wipe (66.55%).
- Evidence preserved under , , and the descriptor's run/diagnostics/hashburst files. The instrumentation does not change the observed expression counts.
- Next experiment adds only timing labels/counts around the three existing per-row statement erases, scope-membership collection, hash-memory cleanup, and ancestor release. The erase counters count actual suffix elements copied by the unchanged `PagedVector::erase`, not modeled bytes. Retarget the sacred dump to the exact enumeration-premise chain above, with user-authorized output redirection to . All dump writers stay unchanged.

## Second detailed profile: confirmed cleanup mechanism

- Complete CUDA pipeline: exit 0; 326.502 seconds overall; all 1,635 C++ tests, 407 verifier tests, 141,136 proof-graph checks, and Lean kernel export passed. IncubatorGauss3 prover = 41.3631 seconds with the same expression counts. Its redirected 2,100,572,073-byte trace adds substantial dump time outside the labelled work; use the first profile for representative timings, not this trace-heavy total or its `wall~s` estimates.
- The unchanged erase loops perform 375,144,803 statement-registry copies (5.22 seconds), 374,747,269 local-registry copies (5.24 seconds), and 169,255,441 local-delta copies (2.22 seconds): 919,147,513 successor copies, 12.68 exclusive worker-seconds. Scope collection and hash-memory cleanup are negligible by comparison.
- In the single target call at burst 10: 46,690 main-registry erases copy 354,793,296 successors; 46,690 local-registry erases copy 354,783,494; 30,610 delta erases copy 164,159,305. Total: 873,736,095 copies / 123,990 erases, 12.09 seconds. `IntEncodedExpr` is statically asserted at 352 bytes, so those copies represent 307.555 GB of logical record payload written; this is an operation count, not measured physical memory traffic.
- Source mechanism: `Memory::wipeSubtree` in [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp) erases closed-scope rows backwards one by one; `PagedVector::erase` in [`paged_vector.hpp`](../../GL_Quick_VS/GL_Quick/src/memory_infra/paged_vector.hpp) copies every successor left. Backwards traversal preserves indices but still repeatedly copies retained rows. The source is unchanged from main; the new ancestor-release workload exposes its quadratic cost.
- Ancestor-release counters: 9,647 product attempts and 8,752 known skips, leaving 895 successful release calls / 1,790 product-plus-sibling offers and history attempts. These are offers, not distinct accepted facts. Most traversal time rounds to 0.00 seconds per displayed path; the dominant path is only 0.02 seconds. This rules out ancestor walking or zoom-in recording as the primary direct runtime cost.
- Next: run clean main with the same native full CUDA pipeline, compare input/load state, burst counts and theorem artifacts, then return to the investigation branch. No fix has been implemented.

## Fresh main reference

- Clean `main`, rebuilt natively in 99.360 seconds, then complete `main.py --GPU --run-descriptor rejected_scopes_rt2_main_reference`: exit 0, 280.043 seconds overall. Main stayed unchanged.
- IncubatorGauss3 prover = 4.26299 seconds, peak 767 blocks. Phase 1 = 0.2718344 seconds; phase 2 = 1.5940933 seconds; phase 3 = 2.1949091 seconds. Burst 10 = 0.661686 seconds with 9,173 expressions. The historical main execution shape is reproduced exactly.
- All 1,624 C++ tests, 403 verifier tests, 141,136 proof-graph checks, and Lean kernel export passed.
- Compared with profile 1, the saved IncubatorGauss3 compiled binary (449,729 bytes), incubator theorem file (152,193 bytes), raw global theorem list (185,789 bytes), and processed global theorem list (203,103 bytes) are byte-identical by length and SHA-256. Full hashes are in ; this is not a claim that every proof chapter is byte-identical.
- Returned. Generated live proof outputs are the main reference's outputs; both measured runs' comparison artifacts and the profiling evidence remain separately archived under `.debug/`. A final rebuild restores the executable to the investigation branch's source; no additional pipeline run is needed for unchanged source.
## Trace-backed causal chain

The exact-chain dump locates the growth in two or-integration scopes under the goal `(interval[1,4,2,7,10])`, whose branch equalities are `(=[6,repl_lev_1_5])` and `(=[7,repl_lev_1_5])`. The first `zoom in` rows appear at dump EXIT #8 (remaining-iteration burst 5), matching the first divergence seen at the next burst entry. One exact example releases `(in2[it_0_lev_1_42,it_1_lev_1_228,3])` from the interval subproof scope into each of those equality branches.

At ENTRY #13 (remaining-iteration burst 10), this logic block has 33,348 statements: 17,618 in the equality-to-7 scope and 14,987 in the equality-to-6 scope. Together those two scopes hold 32,605 statements (97.8%). They contain 13,504 and 12,530 distinct `existence1` statements respectively; the whole logic block has 26,220 `existence1` statements (78.6%). In the saved compiled binary, `existence1` has elements `(in[1,u_1])` and `(in3[u_2,u_3,1,u_4])`. The increased state is therefore mainly operation-result existence facts in the two proof branches, not an explosion of new logic blocks.

At EXIT #13 the logic block drops to 731 statements and both large scopes have disappeared. Its history grows from 240,384 to 434,216 keys because history survives the wipe. Within that call, the 123,990 erase operations count rows built during processing as well as rows present at entry; they must not be interpreted as the entry-minus-exit count. The source and measured timers identify repeated successor copying during these scope deletions as the dominant avoidable cost.

Evidence: , , and the full 2,100,572,073-byte trace. Dump sequence numbers include the two warm-up iterations; dump #13 corresponds to remaining-iteration burst 10.

## Proposed fix, not implemented

Replace each of the three per-row erase loops in `Memory::wipeSubtree` with a stable forward compaction and one final `PagedVector::truncate`. Existing `PagedVector::setAt` and `PagedVector::truncate` already provide the required storage operations; kept rows can retain their exact relative order and the same scope-membership predicate. Each retained record needs at most one copy rather than one copy for each earlier deleted row. Keep all other wipe sections, scope decisions, history, and ancestor releases intact.

Acceptance for an implementation: direct mixed-scope/interleaved-row tests across page boundaries, same kept-row order and dirty/page-release behavior, full CUDA checks, and byte comparisons of the relevant theorem and proof artifacts. Remeasure afterward. The measured erase work offers roughly 12–13 seconds of potential savings on this host; it does not prove a return to 5 seconds because additional statement processing and equivalence-class work remain. No performance fix or semantic pruning was attempted.

## Completion

Final native Release rebuild passed in 99.792 seconds (). It restores the binary to the same instrumentation source whose complete profile-2 pipeline passed every gate. Main remains at its original commit, unchanged. Only instrumentation, its permitted dump target/path, and this report were added on the requested branch. No semantic, performance, configuration, or verifier fixes were made.

## Approved change implementation ledger

- Temporary diagnostic annotations and counters removed; `RT_MEASUREMENT` returned to zero and the initial exact anchor dump target/path restored. All C++ source matched the pre-investigation parent before the baseline build (100.280 seconds, exit 0).
- The three statement loops now share a local stable compaction lambda. No heap allocation, additional persistent state, scope decision, history read/write, or assertion change. Membership is the same `inClosedBit` predicate. Existing `setAt`/`truncate` preserve dirty-state and page-release behavior.
- Added direct tests for complete-record equality and survivor order with mixed parent/descendant/sibling scopes and three differently populated vectors; verify exact data-page counts, restructured state on removal, empty/all-kept/all-removed cases, and index membership. Corrected the observed obsolete erase-cost comment and the logic-block chapter's allocation/registry descriptions while documenting the local cleanup algorithm. No diagram or sizing contract changes.

- Matched pre-fix run: complete CUDA pipeline exit 0, 322.216 seconds overall, IncubatorGauss3 prover 30.4387 seconds, phase 1 = 1.2167651, phase 2 = 3.4562788, phase 3 = 24.812529 seconds. Burst 10 = 20.312 seconds, 37,085 input expressions. All 1,635 C++ / 407 verifier tests, 141,136 proof checks, and Lean kernel export passed.
- Baseline manifest and immutable copies cover 3,496 files / 42,153,814 bytes under ; manifest . The source patch was prepared while the unchanged pre-fix executable ran; the pipeline never rebuilt or loaded the edited C++ source.

- Fixed source full Release rebuild: exit 0 in 98.568 seconds. Direct native C++ suite: 1,637/1,637 passed, including both new compaction tests (). The implementation is a complete verified milestone; next is the full fixed CUDA run and artifact comparison.

- First fixed full-run attempt (`rejected_scopes_rt2_compaction_after`) passed 1,637 C++ and 407 verifier tests, then asserted at `Phase2ProjectionArena::appendLogicalBlock`: `metadataBytes.size + expressionKey.len <= capacity.metadataBytes`. It failed in the first IncubatorPeano1 warm-up Phase 2, before any production `Memory::wipeSubtree` call (the sole caller is the Phase 3 tail in `performElemPhase3`). Native exit 3221226505, pipeline exit 1. Launch-to-exit time 373.317 seconds includes the failed process and is not a successful runtime measurement. No Gauss result or output-identity claim exists for this attempt.
- The unchanged projection code allocates bounded construction shards and assigns blocks through a shared atomic cursor. The failure alone does not establish the exact capacity cause; no CUDA fix or bound relaxation is part of this task. The failed log and diagnostics remain archived. Retry the identical compaction source after a full rebuild under a new descriptor.

## Final implementation verification

- The first failed attempt remains preserved under `rejected_scopes_rt2_compaction_after`; it supplies no Gauss runtime or artifact-identity evidence.
- Identical-source retry `rejected_scopes_rt2_compaction_after_retry1`: full rebuild exit 0; complete native CUDA pipeline exit 0, 288.629 seconds overall; 1,637/1,637 C++ tests, 407/407 verifier tests, 141,136 checks with zero failures, and Lean kernel export passed. No production code changed between the failed attempt and the successful retry.
- Path/length/SHA-256 manifests are identical for all 3,496 files / 42,153,814 bytes. IncubatorGauss3 prover 30.4387 -> 17.1672 seconds, phase 3 24.8125290 -> 11.6773946 seconds; every burst has identical active-block and expression counts. No additional optimization was attempted.

## Encoded-only proposal: legitimacy and eligibility evidence

The saved exact-chain trace contains 26,034 existence1 statements in the two large or-integration scopes at ENTRY #13. All 26,034 have at least one valid local derivation: 25,964 implication-origin checks and 134 equality1-origin checks pass with the unchanged verifier functions, including their scope checks. For the arithmetic closure rows, the two bound operands are alpha-renamed before resolving the recorded anchor parameters, avoiding variable capture. This validates local rule applications; it is not a full independent kernel replay of every intermediate ancestor dependency. Evidence:  and the audit script.

Across all saved ENTRY/EXIT snapshots for that logic-block chain, no `existence1` occurrence of either polarity appears in the dumped request-premise keys or goals. The audit scans 17,237 overall, 8,895 local, 2,369 local-delta, and 664 working-memory key rows. This identifies `existence1` as a concrete candidate for the proposed predicate-level guard in the observed snapshots. A production eligibility test must use the live request indexes and goals and account for later rule/goal changes, rather than treating this trace census as a permanent property. Evidence: .

## Canonical-only policy: clarified by the user

The user explicitly excluded rejected maps: retain every existing rejected-map and admission-map operation. The encoded-only condition is not request-usable AND not a goal predicate, considering either polarity. Entrance canonicalization remains mandatory.

A later equality can change an already-stored row's canonical representative. For example, `P(a)` and `!P(b)` at main can precede `a=b`, even while P appears in no rule premise or goal. `dischargeContradiction` reads the encoded registry and probes the exact negated expression key; entrance-only canonicalization would leave the two older rows unmatched. Re-keying the single stored canonical form when classes change, without enumerating variants, addresses this case. The user explicitly selected the equivalence-class update as the trigger: whenever a class changes, affected unused encoded expressions are updated to their new canonical forms, without generating non-canonical variants. Needed expressions retain ordinary processing. No implementation of the new policy has been made in this diagnosis step. Eligibility transitions when a new rule or goal starts using a previously unused core must also preserve the required processing of existing canonical rows.

## Final Phase 3 diagnosis state

The concrete current costs are measured, the large existence families have passed their local legitimacy checks, and `existence1` is unused as a request premise or goal in the saved boundary snapshots. The user's intended optimization is now explicit: for a core that is neither request-usable (either polarity) nor a goal predicate, encoded-expression storage remains canonical on arrival and after later class changes, without variant multiplication. This applies to internal/local, external, and overall encoded-expression collections only. Rejected maps and admission maps retain their existing processing. Future rule/goal relevance changes must allow existing canonical rows to resume the needed processing.

No semantic optimization has been implemented here. Only timing annotations and the runtime gate changed; the profile completed all pipeline/Lean checks and preserved all 3,496 compared artifacts. The measured direct work that can be targeted by the encoded-only policy is within the 1.28-second statement-equivalence batch and 1.61-second expression-cleanup category; those totals include needed predicates too and are upper work budgets rather than promised savings. Entry canonicalization and normal disintegration still consume time. The profile and policy analysis are ready for implementation review; diagnostic annotations remain available on the branch.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
