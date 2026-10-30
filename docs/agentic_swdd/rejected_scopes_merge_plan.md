<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Rejected scopes: merge main and run the full pipeline

## Current state

- Objective complete: zoom in is implemented across recording, exact-endpoint export, processing, verification, typed certificates, and Lean replay. Both full and shortcut GPU runs passed.
- Acceptance met: one unchanged-expression inheritance edge cites the exact strict-ancestor source; parent state/history remain intact and recording reads no parent origins. C++/verifier/export tests and Lean fixtures pass. Both requested production runs passed uninterrupted with Lean and zero verifier failures; their retained proofs contain zero zoom in rows, so dedicated fixtures supply new-tag coverage.
- Approved scope: the user approved the explicit zoom in design, including its dedicated verifier and Lean support, and required full plus shortcut GPU verification. Existing release policy and other verifier checks remain unchanged.
- Starting destination:; source:; common base:.
- Both branches matched their remote tips after a successful fetch. The working directory was clean; no prover or build process was running. Destination has two unique commits; source has 68.
- Existing limitation: the destination records an aborted IncubatorGauss3 run with a runtime explosion. Its cause remains undiagnosed; this merge must preserve the ancestor-park release design.
- Outcome: zoom in implementation 44680cd3e2b9b1c64c82b872cbc3fd9921d78ad9 is committed and pushed. Full and shortcut GPU invocations completed successfully with their in-pipeline Lean checks. The earlier generic-copy and environment investigations remain recorded as history.
- Next action: none for this implementation. Results and the new-tag coverage boundary are recorded below.
- Constraints: preserve history and main's tip; no pending-ID renumbering or unrelated proof-policy changes. The approved tag adds one dedicated checker; existing checks stay intact. History stays enabled and never determines release. All runs use main.py with --GPU, unique logs, and recorded exit status.

## Evidence

- Initial branch and ancestry inspection completed; the only destination feature is ancestor-park release plus its aborted-run record.

## Verification

```powershell
& 'C:/Users/nikol/anaconda3/python.exe' -c 'import os, subprocess, sys; sys.exit(subprocess.call(sys.argv[1:], env=dict(os.environ)))' 'C:/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe' GL_Quick_VS/GL_Quick.sln -p:Configuration=Release -p:Platform=x64 -t:Rebuild -verbosity:minimal '-p:CudaToolkitDir=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/'
& './GL_Quick_VS/GL_Quick/gl_quick.exe' --unit-tests
& 'C:/Users/nikol/anaconda3/python.exe' tests/test_harness.py
& 'C:/Users/nikol/anaconda3/python.exe' main.py --GPU
```

Each command writes to its own uniquely named `.debug/` log. Final exit codes, verifier totals, and runtime scopes will be recorded here.

- Merge review: all source and tests merged automatically; five documentation conflicts retain both contracts. The source difference against main is exactly the destination feature (328 lines in prover.cpp, 233 in prover.hpp, and its existing tests/verifier change). No new verifier edits or architecture changes were made.

- Exact merge check: the added/removed lines in both prover files, C++ tests, verifier, and Python structural tests match the original destination patch exactly (328, 233, 315, 32, and 82 lines respectively).
- Initial rebuild failed before compilation because this tool session has no CUDA toolkit path. CUDA 13.3 exists at `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/`; retry passes that path through the MSBuild `CudaToolkitDir` property. No project configuration was changed.

- Python verifier unit tests: 406/406 passed, exit 0 ().
- Corrected two stale descriptions encountered during merge review: the glossary now describes mail-only revival, and the prover chapter describes the current canonical rejected-integration rewrite rather than the retired permutation walk.

- Compilation exposed one source-level merge collision: both parents defined test helper `mintChildScope` in the same anonymous namespace. Renamed the destination helper and its eight callers to `mintAncestorReleaseChildScope`; both test fixtures remain intact, and production source is unchanged.
- The same build reported an MSBuild FileTracker environment-path error. The next rebuild supplies standard Windows ProgramData/ALLUSERSPROFILE and the installed CUDA path in its process environment.

- Registry verification: every decision, invariant, and gotcha heading from both parents survives; the ancestor-release placeholders remain unnumbered. Main remains at its original tip.

- Environment diagnosis: the node-launched PowerShell returns an empty `Environment.GetFolderPath(CommonApplicationData)`; the shell execution tool returns `C:/ProgramData`. Merely setting environment variables did not repair that API result. The full rebuild is now launched through the shell with an event-driven completion-file watcher; no build logs are read while it runs.

- Shell-launched rebuild reached a separate environment error: MSBuild received both `PATH` and `Path`, which its case-insensitive dictionary rejects (MSB6001). The next launcher uses Python's Windows-normalized environment dictionary, verifies case-insensitive key uniqueness, and passes the same full-rebuild command. Production source is unchanged.

- Successful full Release rebuild: exit 0, 94.5562727 seconds (). The child environment has one case-insensitive entry per name and an explicit installed CUDA toolkit path.
- C++ unit tests: 1,631/1,631 passed, exit 0 (). Both parents' test fixtures are retained.
- Merge milestone is ready: production source and verifier contain the original destination patch unchanged against main; the sole additional source edit is the test-helper rename. The full pipeline result is still pending.

- Merge committed and pushed:, parents and. Post-commit working directory was clean; main ancestry verified.
- Full run command: `C:/Users/nikol/anaconda3/python.exe -u main.py --GPU --run-descriptor rejected_scopes_merge_20260905_1788590257543`. Output: . One file-event watcher receives the exit record; no progress polling. Diagnostics and hashburst evidence are copied under the run descriptor on completion.

- Maintainer follow-up: make the recurrent PATH/Path fix persistent. Updated Rule 25 and the canonical build command in both AGENTS.md and the project conventions to require native Python child-environment normalization before every Windows build. Saved the explicit user-requested memory correction at `C:/Users/nikol/.codex/memories/extensions/ad_hoc/notes/20260905-085312-windows-path-environment.md`. The confirmed command and launcher limitation are recorded there; no machine environment was changed.

## Initial uninstrumented run result

- Executed the complete pipeline entry point on merge, with every batch explicitly using CUDA.
- The pipeline repeated both gates successfully: 1,631/1,631 C++ unit tests and 406/406 Python verifier unit tests.
- Completed IncubatorPeano1, IncubatorPeano2, Peano, IncubatorGauss1, and IncubatorGauss2. IncubatorGauss3 reached remaining-iteration burst 6 (6 active bodies, 11,076 total expressions) and aborted.
- Exact assertion: `copied >= 1 && "releaseParkedProductIntoScope: a parked product has no disintegration origin at its park scope"`, in `logical_block/rejected_source_replay.cpp` / `RejectedSourceReplay::releaseParkedProductIntoScope`.
- Native exit status: 3,221,226,505. Pipeline exit status: 1. Measured wall time from pipeline launch to failure: 197.275845 seconds; this is not a completed-run runtime or a prover-only measurement.
- That run did not reach final proof-graph verification. No zero-failure or complete-output claim was established. The cause was undiagnosed at that checkpoint; the later trap diagnosis below resolves it. The assertion, verifier, and scope-release architecture are unchanged.
- Evidence:  and its completion JSON; archived diagnostics  (2,752,695 bytes). The archived sacred hashburst file is empty, so it does not identify the failing product or logic-block chain.
- This merge-and-run record is closed with a failed full-pipeline result. Any repair requires separate trace-backed investigation under the existing rules.

## Trap investigation setup

- User requested trap debugging on 2026-09-05 and reaffirmed the intended behavior: release a parent-scope park into a child scope where the product is needed and absent. Preserve that behavior.
- The assertion checks the source `(product, parkValidity)`, not presence of the product at `targetValidity`. The copy helper selects only `OriginTag::disintegration`; missing source rows and source rows with other tags must be distinguished from actual runtime evidence.
- The dump target is determined before execution: `addTheoremToMemory` builds one logic block per premise, sets `exprKey` to that premise, and links the parent. The current IncubatorGauss3 conjectures begin with `(AnchorIncubator3[1,2,3,4,5,6,7,8,9])`; its parent is the empty root sentinel. Only this exact two-node chain is selected for the sacred dump.
- Failure-only instrumentation will print the actual complete failing logic-block chain, product and siblings, source and target scopes, same-product history at other scopes, and the complete reachable origin-key closure using bounded stack storage and non-minting reads. Output goes to ; the original assertion remains immediately afterward.
- Initial hypothesis before the diagnostic run: a renamed parked product may have an equality1 foundation rather than a direct disintegration row. No source mutation or skipped release is authorized by that hypothesis.

- Trap build reached compilation with normalized Windows environment; MSVC rejected the diagnostic fopen call (C4996). The trap now uses fopen_s under MSVC and the standard fopen on other compilers, with explicit failure assertions. The original proof assertion and provenance reads are unchanged.

- Instrumentation verification: removing only the diagnostic lambda and its two conditional calls in an in-memory comparison reproduces the original prover.cpp exactly; all hashburst_dump.cpp bytes outside isTargetLB are unchanged. The source assertion itself is untouched.
- Corrected trap build: exit 0, 106.8576257 seconds (). C++ unit tests: 1,631/1,631 passed, exit 0 ().

## Confirmed diagnosis

- Diagnostic checkpoint:, committed and pushed before reproduction. The full `main.py --GPU` run repeated 1,631/1,631 C++ and 406/406 Python unit-test passes, then reproduced the original assertion in IncubatorGauss3, remaining-iteration burst 6. Pipeline exit 1; native exit 3,221,226,505. Pipeline launch-to-abort wall time was 223.0504632 seconds with diagnostic dumping; this is not the duration of burst 6.
- The failure trap captured two instances. Complete logic-block chains (innermost to root): `(EnumerationSet2[2,6,10])` → `(AnchorIncubator3[1,2,3,4,5,6,7,8,9])` → empty root sentinel; and `(EnumerationSet3[2,6,7,10])` → the same anchor → empty root sentinel. The sacred dump separately captured the exact anchor-to-root chain.
- Both instances use source scope `main_boundary_(interval[1,4,2,6,10])_subproof_(implication28[1,4,2,6,10])`. Their targets append, respectively, `_boundary_orint_(or2[2,repl_lev_1_2,6])_((=[6,repl_lev_1_2]))` and `_boundary_orint_(or4[6,repl_lev_1_2,2,7])_((=[6,repl_lev_1_2]))`. Both targets are strict descendants of the park scope, as intended.
- The failed product is `(in3[repl_lev_1_2,repl_lev_1_2,it_0_lev_1_356,4])`; its sibling is `(in[it_0_lev_1_356,1])`. Source-key lookup succeeds in each logic block and finds exactly one source origin row: `equality1`.
- That row cites two parent-scope premises: earlier product `(in3[int_lev_1_5,repl_lev_1_2,it_0_lev_1_356,4])` and equality `(=[int_lev_1_5,repl_lev_1_2])`. The earlier product has a `disintegration` row citing `!(>[it_0_lev_1_356](in[it_0_lev_1_356,1])!(in3[int_lev_1_5,repl_lev_1_2,it_0_lev_1_356,4]))`. The equality exactly rewrites the earlier product to the failed one. Each captured closure finishes with 32 keys; neither trap exceeded its capacity.
- Root cause: `logical_block/rejected_source_replay.cpp` / `copyDisintegrationOriginsToInternalMail` discards every row whose tag is not `OriginTag::disintegration`. The rewritten product's immediate `equality1` row is therefore ignored and the helper returns zero, triggering `releaseParkedProductIntoScope`'s unchanged assertion. The parent history exists; the disintegration foundation is one dependency step behind the equality rewrite. The row is not the one-premise self-source convenience row: it cites a different product plus a concrete equality. The child expression's absence is expected, not the defect.
- Source correspondence: the rejected-map canonical rewrite hooks in `logical_block/equivalence_admission_rewriter.hpp` (`applyEquivalenceClassToRejectedMap` and its integration twin, their `writeEquality1` lambdas) explicitly maintain this rewritten-product history shape. The relevant existing tests check direct disintegration copying and equality1-only rekey history separately.
- Primary evidence:  (18,153 bytes), events `PARK-ORIGIN-SOURCE`, `PARK-ORIGIN-NODE`, `PARK-ORIGIN-ROW`, and both `PARK-ORIGIN-END` markers;  (12,696,545 bytes); . A read-only consistency extraction verified both rows, scopes, rewrites, and closure counts and wrote .
- No semantic fix, verifier change, assertion change, or new run is needed for this diagnosis. Temporary failure instrumentation remains in `releaseParkedProductIntoScope`, and the sacred dump remains targeted to the anchor chain. Remove the temporary failure instrumentation after a separately authorized repair, before its final production verification (Rule 30); do not remove the sacred dump.
- Diagnostic record closed with the cause established. The older pre-merge runtime explosion remains a separate, undiagnosed observation.

## Approved repair

- The user approved copying every stored parent history row, preserving its tag, dependency expressions, dependency scopes, and order through the normal writer. They then explicitly narrowed the task: do not change verifier.py; verifier failures will be inspected individually later.
- No verifier changes had been made before that correction. The release continues to depend only on demands, parks, scopes, and known statements. History maintenance returns no release decision; an empty source history is an empty copy.

- Implementation: copyOriginsToInternalMail returns void, copies every stored tag and dependency key in order, and treats absent history as an empty copy. releaseParkedProductIntoScope has no copied-count assertion or history-dependent release result. The temporary failure trap is removed; the sacred dump target and machinery are unchanged from the diagnostic checkpoint.
- Full Release rebuild passed (exit 0, 111.5752243 seconds); 1,634/1,634 C++ unit tests passed. Tests cover all OriginTag values, repeated-copy deduplication, preservation of parent rows and dependency scopes, both rejected maps with the captured equality1 history shape and sibling, and history-enabled release with no history available.
- git diff confirms verifier.py and the entire Python tests directory are unchanged. The agent contract and user diagram describe the generic recorder; the revised SVG parses and its rectangles remain inside the enlarged view box. Build and test logs use the rejected_scopes_copy_20260905_094857 descriptor.

- Repair committed and pushed as; post-commit working directory was clean. The full GPU run was launched through the normalized native Windows helper with one completion-file watcher. Output: ; diagnostics and hashburst evidence will be archived under that descriptor at process exit.

## Full-run result and finalization

- The generic-copy main.py --GPU run completed all native batches: IncubatorPeano1, IncubatorPeano2, Peano, IncubatorGauss1, IncubatorGauss2, IncubatorGauss3, and Gauss, including the main-batch compressors. The former origin assertion did not recur. IncubatorGauss3's complete prover runtime was 36.5434 seconds; it is a batch-level measurement, not one burst.
- The process stopped after 338.8993869 seconds at Lean finalization. The kernel_check.log contained only a GitHub download failure; no Lean proof-check result had been produced. The processed main graph contained 64 Peano and 30 Gauss theorems for export.
- Environment diagnosis: sandboxed elan toolchain list returned no installed toolchains; the host command listed both leanprover/lean4:v4.30.0-rc2 (the project's exact pin) and v4.33.0. The installed toolchain was hidden from the sandbox, causing an unnecessary download attempt. No installation or toolchain-version change was made.
- The failed kernel log is preserved at . Finalization resumes from the existing outputs by calling the existing kernel_check, HTML generation, and verifier functions outside the sandbox, with normalized environment and a completion watcher. It does not rerun the prover or modify the verifier. Its log is .
- The first completion-watcher RPC timed out after 300 seconds; that was a notification-transport failure, not a process exit. The completed run's independent JSON record supplied its actual exit status and duration.

## Repair completion

- Host finalization exited 0 in 31.4375508 seconds. The existing Lean v4.30.0-rc2 toolchain built the exported Peano/Gauss project successfully (7 jobs), with no download or toolchain installation. Main and incubator HTML export then completed.
- The unchanged verifier reported incubator success=124224, failure=0 and main success=16912, failure=0: 141,136 checks and zero failures in total. No verifier failure was altered, suppressed, or reclassified.
- Evidence: , its completion JSON, and . The first sandbox failure log is retained separately. The original process duration (338.8993869 seconds) and resumed-finalization duration (31.4375508 seconds) are separate measurements; this was not one uninterrupted successful main.py invocation.
- Verification of scope: git diff against the pre-repair checkpoint shows no changes to verifier.py or the Python tests directory. Production/test source is unchanged from repair dfe7c4f96; the full rebuild and 1,634 C++ plus 406 Python unit-test results apply to the completed run.
- The temporary failure-only trap has been removed. The sacred hashburst dump remains intact and targeted to the exact AnchorIncubator3-to-root chain. All source changes are on sandbox/rejected_scopes; main remains untouched.
- Repair record closed. The full native run passed the former IncubatorGauss3 failure point and completed that batch; the older pre-merge performance observation is retained as historical evidence without an invented diagnosis.

## Lean launch reliability

- The user requested reliable full runs after the successful resumed finalization. This is a launch-environment repair; no prover, proof-record, verifier, or kernel acceptance rule changes are authorized or needed.
- More precise diagnosis supersedes the earlier broad statement that the sandbox hides the toolchain files: the pinned bin/lake.exe exists and runs directly inside the sandbox, while elan with implicit home reports no toolchains and warns that it cannot canonicalize the user-profile path.
- Giving the same sandboxed command explicit ELAN_HOME=C:/Users/nikol/.elan makes elan list both installed toolchains. From the generated project directory, the same lake launcher then reports Lean v4.30.0-rc2, honoring lean-toolchain without a download.
- Planned fix: kernel_check passes a copied child environment with ELAN_HOME explicitly defaulted to Path.home/.elan; existing overrides remain authoritative. find_lake's non-PATH lookup uses that configured home. Machine settings, installed versions, the project pin, and failure assertions remain unchanged.

- Launch repair implemented in proof_export/live.py: explicit child ELAN_HOME, parent environment untouched, existing override respected, configured-home discovery supported. All 21 live-export unit tests passed, including default-home forwarding, custom-home forwarding for both Lean commands, unchanged failure assertion, and configured-home discovery.
- A real sandboxed call to the updated kernel_check passed (7 Lake jobs) with ELAN_HOME absent from the parent process. No explicit environment setting or escalation was supplied by the test command; the pipeline code supplied it. Evidence: .

- Launch fix committed and pushed as after unit tests and the successful real sandboxed Lean preflight. Fresh full Release rebuild launched through the normalized Windows helper. Verification uses the lean_home_full_20260905_103021 descriptor; one file-event completion promise is retained across any notification-transport timeout, without process/log polling.

## Zoom in tag proposal — analysis only

- While the uninterrupted run is pending, the user proposed a dedicated `zoom in` tag for copying an unchanged expression into a child scope while keeping its parent-scope original. No implementation of this proposal is authorized by the request to analyze it; the existing verifier-change restriction remains in force.
- Proposed row: `E child_scope zoom in E parent_scope`. It has exactly one dependency, identical expression text, and a source scope that is a strict ancestor of the result scope (any ancestor, matching the existing release path). Same-scope, sibling, descendant-source, and changed-expression cases are rejected.
- The recorder would emit this edge directly from the existing release data, once for the product and each sibling. It would not inspect or copy the parent's derivation rows. Parent history remains intact; a parked product need not be admitted in the active parent statement registry. The reasoning algorithm and demand/budget/known-skip decisions remain unchanged.
- Verifier requirement: check the exact `(expression, source_scope)` dependency, not expression text alone. The current generic origin meta-check indexes only expression text, so this tag's same-expression dependency could otherwise appear satisfied by the child row itself. Normal origin-chain termination and type checks must still apply.
- Exporter requirement: preserve the explicit child zoom row and its source scope. `proof/proof_reconstructor.cpp` / `liftToShallowestOriginAncestor` currently lifts to ancestor origins, except across OR-branch boundaries; without a deliberate preservation rule it can erase a zoom row in ordinary child scopes or change the recorded source. The source derivation must remain traversed and exported, not replaced with an unproved leaf.
- Typed certificate and Lean requirement: add an explicit scope-inheritance action that reuses the cited parent proof in the descendant context, respecting the existing bound-variable and scope-context representation. No new mathematical axiom or proof search is needed.
- Integration surface: append the OriginTag value without renumbering existing tags; add the tag name, description, verifier checker, certificate action, Lean renderer, processing/export round-trip tests, and matching agent/user design documentation. A child-side canonical rewrite remains a separate equality1 step after zoom in.
- Expected tradeoff: one direct parent dependency per release and an explicit transfer row in the exported chapter. This is clearer than duplicating parent derivation rows, but exported graphs and row counts will change. Test both rejected maps and siblings, nested scopes, exact source preservation through processing, negative scope/source cases, and Lean replay before a full-run gate.

## Uninterrupted sandbox verification completed

- Source:. Fresh full Release rebuild exited 0 in 117.7319159 seconds.
- One uninterrupted `main.py --GPU` invocation completed inside the sandbox with exit 0. The program reported Overall runtime 380.21521 seconds (the wrapper measured 380.4020356 seconds). No finalization was resumed, no host escalation was used for the run, and the wrapper recorded parent_elan_home=null.
- Both in-pipeline gates passed: 1,634/1,634 C++ unit tests and 406/406 Python verifier unit tests. Lean built the exported project successfully (7 jobs); both HTML trees completed; the unchanged verifier reported 124,224 incubator plus 16,912 main checks, totaling 141,136 checks with zero failures.
- Evidence: , its completion JSON, , and .
- Conclusion: Lean is not inherently incompatible with this sandbox. Elan's implicit-home discovery was the failing part; explicit ELAN_HOME in the pipeline child environment resolves it using the installed pinned toolchain. The earlier coarse description of hidden toolchain files is superseded by the direct-binary and explicit-home probes.
- The zoom in proposal was analyzed against the existing code while this run proceeded; none of its proposed production, verifier, or Lean changes were made. This successful run validates the generic-copy and launch-environment fixes only.

## Zoom in implementation approved

- The user approved the analyzed design, including the necessary dedicated verifier checker and Lean action, then explicitly required `main.py --shortcut --GPU` in addition to the full GPU run. This approval supersedes the earlier no-verifier-change boundary only for the new tag's support; existing checks remain intact.
- Row contract: `E child_scope zoom in E parent_scope`, one exact dependency, identical expression, strict ancestor source. Parent statement/park and history remain intact. Recording uses release data directly and does not inspect parent history.
- Export must preserve both recorded endpoint scopes, including ordinary child scopes where generic lifting would erase the transfer. The parent derivation must remain in the exported dependency chain. Verifier must require the exact source pair, not an expression-only match.
- New tag is appended to the OriginTag vocabulary; existing numeric tag values remain stable. All required code, tests, tag description, and agent/user design documents belong to a complete verified milestone. No prototype is accepted as final.

- Implementation checkpoint: OriginTag::zoomIn is appended; recordZoomInOrigin emits one direct parent dependency without reading parent history. The exporter preserves zoom results and traverses their parent dependencies at exact scope. The processor leaves exact zoom source theorem spelling intact. The dedicated verifier checker and typed certificate require an exact source pair; no other checker rule is changed.
- Lean replay enters the child context and explicitly applies the parent proof with its parameters, matching parent assumptions, and witness guards. Scoped witness bundles are carried to the child. Seven focused tests pass, including graph-processing identity and exact certificate resolution; four generated Lean smoke theorems (main, nested, guarded witness, existential) kernel-check successfully.
- Full Release rebuild passed in 107.0209848 seconds. C++ unit tests passed 1,635/1,635; Python verifier tests passed 407/407; all 80 Lean-export unit tests passed. No full or shortcut gate has yet been run on this implementation.

- Implementation committed and pushed as 44680cd3e2b9b1c64c82b872cbc3fd9921d78ad9. Full GPU gate started with descriptor zoom_in_20260905_112000_full; shortcut will follow sequentially. A raw XML check initially rejected pre-existing HTML named entities in the SVGs; parsing with HTML entities resolved validates all SVGs in both touched user pages. No diagram change was needed for that parser issue.

## Zoom in verification completed

- Implementation source:; no production or test source changed after its successful build and unit tests.
- Full run: `main.py --GPU --run-descriptor zoom_in_20260905_112000_full`, exit 0, Overall runtime 346.39572 seconds (wrapper 346.5540822 seconds). Lean built Peano/Gauss successfully (7 jobs). Verifier: 124,224 incubator plus 16,912 main checks, total 141,136, zero failures.
- Shortcut run: `main.py --shortcut --GPU --run-descriptor zoom_in_20260905_112000_shortcut`, exit 0, Overall runtime 73.10568 seconds (wrapper 73.2602043 seconds). Lean built FTA successfully (6 jobs). Verifier: 9,217 checks, zero failures. The run saved 64 theorem rows and reported 39 newly proven theorems; those counts have their existing distinct meanings.
- Both runs repeated 1,635/1,635 C++ and 407/407 Python verifier unit-test passes. Both reports show zoom in success=0, failure=0: the retained production proof graphs do not use a zoom in row. This is regression validation, not direct runtime coverage of the new tag.
- Direct tag coverage: C++ tests emit zoom rows and preserve both endpoints despite a higher ancestor origin; Python tests validate exact source pairs and graph processing; typed certificate and renderer tests pass. Four generated Lean smoke theorems exercise main, nested, guarded-witness and existential inheritance. An additional isolated Gauss test module, generated from the current full-run certificate with explicit main-to-child and nested parent-to-child zoom steps, kernel-checked with exit 0. Evidence: , its manifest, and . This is a synthetic verification fixture, not a production GL theorem artifact.
- Run evidence: , , their independent completion JSON records, and archived `_kernel_check.log` files for each run. No verifier failure was weakened or suppressed.
- Completion notifications were not timely enough for the user. Final results were confirmed from the exit records and logs. The requested shortcut status check found the launched Python wrapper live, then its completed 73.1-second result; no running state is inferred from an empty console or an idle GPU.
- Task complete. Existing origin values remain stable; parent derivations are not copied or interpreted by the recorder; source and target scopes remain explicit; the new checker and Lean action preserve the full cited proof path.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
