<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# D2 — uniqueness of quotient and remainder — current proof state



## Active runtime-explosion investigation — 2026-09-11



## Active remainder-uniqueness stall investigation — 2026-09-10

- **Objective and acceptance:** Apply the stall playbook to the new remainder-uniqueness goal. Identify the first missing human-proof step, its exact logic block and scope, the relevant live statements/rules/admission state, and whether the cause is a coding defect, an architectural limit, or a configured restriction. Diagnosis requires evidence of the stopping condition, not just a missing final equality.
- **Authorization and constraints:** The maintainer explicitly requested applying the playbook. Use the saved full run first; retain the supplied conjecture, capacities, assertions and prover policy. Any required fresh measurement uses native full rebuild, `main.py --GPU` (or a fingerprint-valid GPU shortcut), a distinct descriptor and one completion watcher. Change only the sacred full-chain target when instrumenting a different block; obtain direction before semantic/configuration repairs.
- **Baseline:** The completed experiment below avoids exhaustion with two induction candidates (`a,d`) but leaves `(=[11,12])` pending at main. Its final target frame has both quotient/product decompositions and both strict remainder bounds. These witnesses therefore exist; predicate expansion is not the first missing step on the direct route.
- **Working map:** `a=9`, `d=10`, `r=11`, `r'=12`; the first quotient/product are `int_lev_1_0` / `int_lev_1_1`, the second are `int_lev_2_0` / `int_lev_2_1`. Their products plus the respective remainders both equal `9`.
- **Next action:** The sequential playbook result is recorded in the appended steps 4–8. Await the maintainer's selection of further proof-route investigation or a design direction; do not apply a repair.
- **Current result:** Two source-grounded blockers are established: the negated-head block cannot open cases without explicit goals, and the direct first quotient case parks both difference witnesses without matching admission demand. No coding-contract violation is established; the two induction families remain untraced.

## Completed remainder-uniqueness experiment — 2026-09-10

- **Objective and acceptance:** Replace the two expanded D2 conjectures with one remainder-uniqueness conjecture for the same dividend and divisor. Run one full GPU pipeline and record whether host-pool exhaustion persists and whether the exact new conjecture is proved.
- **Authorization and constraints:** The maintainer requested this replacement to test exhaustion. Keep the first 66 supplied conjectures, predicate definitions, prover policy, capacities and assertions unchanged. Retarget only the sacred dump's complete-chain matcher to the replacement premise block.
- **Baseline:** The latest completed run below exhausted all 49,152 blocks of the 12 GiB host main pool at FTA burst 16 during Phase 3; both expanded D2 equality goals remained open.
- **Hypothesis:** One equality head has only two induction-eligible variables, `a` and `d`. In `ConfigFTA.json`, `remainder` has `input_args=["a","d"]` and `output_args=["r"]`; the two outputs `r` and `r'` are excluded. `logical_block/scope_type_resolver.hpp`, `ScopeTypeResolver::findDigitArgs`, selects distinct configured input arguments minus all output and anchor arguments. The quotient and product witnesses stay internal to the predicate. The earlier expanded rows each have five eligible variables (`q,d,r,q',r'`), so the supplied goals move from ten candidates across two heads to two across one head. The configured-input/output calculation is saved in . The completed measurement below confirms that this input avoids the previous exhaustion on the unchanged 12 GiB main pool; it does not isolate the contribution of each input simplification.
- **Replacement:** `remainder(N,+,*,a,d,r)` and `remainder(N,+,*,a,d,r')` imply `r=r'`. Slots `9=a`, `10=d`, `11=r`, `12=r'` follow the unchanged eight-slot FTA anchor. Quotient uniqueness is outside this experiment.

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](remainder[1,4,5,9,10,11])(>[12](remainder[1,4,5,9,10,12])(=[11,12]))))
```

- **Trace identity:** `(remainder[1,4,5,9,10,12])` → `(remainder[1,4,5,9,10,11])` → `(AnchorFTA[1,2,3,4,5,6,7,8])` → empty root with null parent. This follows `proof/proof_grid_builder.cpp`, `ProofGridBuilder::addTheoremToMemory`, which creates one child per premise and sets its expression key to that premise.
- **Input milestone:** Committed and pushed as. The input has 67 unique rows; the first 66 are unchanged and the replacement parses with no free arguments. All 266 source/input hashes from the previous failing run were compared: only the conjecture file and sacred dump matcher differ. Evidence: .
- **Completed run:** Descriptor `d2_remainder_unique_20260910_01`, tested input/source commit. The native full rebuild passed in **103.142 seconds**. `main.py --GPU` exited **0** after **296.926 seconds** (pipeline-reported runtime **296.78738 seconds**); FTA prover runtime was **55.1819 seconds**. All **1,724 native tests**, **423 verifier unit tests**, **150,663 proof checks with zero failures**, and the connected Lean build passed. The eight logged theorem counts are **518, 79, 76, 105, 25, 4, 19, 40**.
- **Memory result:** FTA's main static-pool peak was **12,933 blocks / 3,390,308,352 bytes / 3.15747 GiB**, with the same **49,152-block / 12 GiB** capacity. The previous expanded input exhausted that capacity during Phase 3. Across all eight batches the largest main-pool peak was **6.32520 GiB**. This run completed without either host-pool or GPU-capacity exhaustion; no capacity or assertion was changed.
- **Proof result:** The new remainder-uniqueness conjecture remains **unproved**. The exact target's final frame, **EXIT #26**, still contains `(=[11,12])` at `main`, with 8,436 statements and 4,169 hash-memory originals. Neither saved theorem list contains the new statement or a two-remainder equality row. FTA reaches burst 42; bursts 24–42 report seven active bodies, zero swept and seven skipped. The successful verifier and Lean results validate the proofs produced, not this pending conjecture.
- **Evidence:** , , and the descriptor's `_diagnostics.log`, `_hashburst.txt`, `_analysis.json`, `_proof_and_peak.json`, `_inputs.json`, `_artifacts.json`, `_baseline_comparison.json` and `_induction_inputs.json`. The saved sacred trace is **407,507,642 bytes**, SHA-256 `e2b173cc594244fae8bfa29a2dee6d765684815a975a9d5e05f4bbbe8a6e8041`; the artifact manifest records **5,551 files**. The full pipeline refreshed the supplied-input capture normally.
- **Watcher:** One filesystem-event completion watcher read the atomic completion record. Its client connection timed out after five minutes; reconnecting to the same promise returned the completed result. No running logs or process state were polled.
- **Disposition:** Requested replacement and exhaustion experiment complete. Keep the new row and matching sacred trace target. Memory exhaustion is avoided in this measured run; the remaining proof stall is not diagnosed. No additional run, capacity adjustment, or prover-policy change is part of this task. Freeze this result; a later proof investigation should start from the saved target trace and its induction/contradiction children.

## Completed full retest after accepted main landing — 2026-09-10

- **Merge completed:** main squash, documentation renumber
, and D2 merge are
 committed and pushed. Main is an ancestor of D2. Both D2 conjectures and the
 full-chain sacred dump target remain unchanged; documentation conflicts kept
 both histories. The only source differences from main are D2's existing
 capacity-description comment and eight-byte owner-work assertion in a test.
- **Run completed with failure:** descriptor `d2_no_requests_2_20260910_01`.
 Native full rebuild passed in **104.746 seconds**. `main.py --GPU` exited 1
 after **442.157 seconds**, with native FTA exit 3. All **1,724 native tests**
 and **423 verifier unit tests** passed. The seven pre-FTA theorem counts
 match main: **518, 79, 76, 105, 25, 4, 19**.
- **Failure:** the same host assertion, `carveCursor_ < totalBlocks_`, in
 `memory_infra/global_memory_manager.cpp` / `GlobalMemoryManager`, at **FTA
 hashburst 16 during Phase 3**. The Phase 2 barrier completed in 13.174 seconds;
 no Phase 3 completion barrier follows it. The old GPU capacity failures were
 not observed before this host failure. **The requested crash removal has not
 been achieved.**
- **Host census:** all **49,152 blocks / 12 GiB** occupied. Worker container
 scratch: **31,795 blocks / 7.76245 GiB**. Logic-block storage: **17,322 blocks /
 4.22900 GiB**. String scratch: 33 blocks; unattributed: 2 blocks. These are
 concurrent approximate reads. The `request-generation` label sums shared
 `genScratchArenas` storage; it does not establish that these bytes are GPU
 requests. The preceding GPU stream completed 346,208 requests and 315,516
 firings, with a 710,046,512-byte frontier peak; that local frontier backing
 has been released when the controller returns before Phase 3.
- **D2 proof state:** the last sacred target frame is EXIT #19 with both
 `(=[9,14])` and `(=[12,16])` still pending at main. FTA never completed.
 Final proof verification and connected Lean were not reached; preserved old
 proof files are not evidence of completed D2 verification.
- **Evidence:** , the distinct
 build/full-run/diagnostics logs, saved sacred dump, input and artifact
 manifests,  and
 . The main landing's
 exact tree and 266 accepted input/source hashes are recorded in
 .
- **Watcher:** one filesystem-event completion watcher observed the run's
 atomic completion record. Its client connection timed out after five minutes;
 retrieving that same completed watcher result did not stop or restart the
 native process. No periodic log/process polling was used.
- **Disposition:** the approved squash, merge and full D2 retest are complete;
 the old host-pool crash persists. No assertion or capacity was weakened, no
 additional run was started, and no unrelated repair was attempted. Remaining
 work is to attribute the ordinary Phase 3 worker-scratch growth before proposing
 a storage or inference change.

## Completed capacity retest — 2026-09-10

- **Objective and acceptance:** Merge main, perform one native full rebuild and full GPU pipeline attempt, and determine whether a GPU capacity assertion still stops D2. Completed; the run now stops on host memory capacity.
- **Authorization and constraints:** The maintainer requested the merge and run. Both supplied equality conjectures, assertions, proof policy and the complete-chain sacred dump target remain intact. No capacity change or additional run follows this measurement.
- **Merged and pushed:** Merge joins D2 with main; main is an ancestor. This is the tested source commit.
- **Measured outcome:** Full rebuild passed in 111.994 seconds. The full GPU process ran for 476.637 seconds, passed 1,723 native and 423 verifier unit tests, and completed all seven pre-FTA batches. FTA passed the old burst-11/12 GPU failures and aborted at burst 17: the 12 GiB host main static pool exhausted all 49,152 blocks. No GPU capacity assertion occurred before this abort. Native FTA exited 3; `main.py` exited 1.
- **D2 proof state:** Both equality goals remain open at exact `main` in the last sacred target snapshot, ENTRY #20. Neither D2 theorem was printed as proved. FTA proof export, final proof-graph verification and connected Lean were not reached; existing theorem files are partial pre-FTA outputs, not a completed D2 result.
- **Proof-route distinction:** A mathematical uniqueness proof can use a contradiction LB seeded with `!(=[9,14])`, then direct cancellation for `(=[12,16])`. GL enables direct deduction, induction and negated-head contradiction; no successful D2 method is established. A contradiction discharge can carry registry method `direct`, so proof-history tags and the source block determine the mechanism.
- **Evidence:** , , , , , and .
- **Disposition:** Requested merge and capacity retest complete. Freeze this result. The next unresolved blocker is host main-pool consumption; no repair or sizing experiment is part of this task.

## Previous handoff — 2026-09-09

- **Objective:** Record the D2 findings and the proposed GPU-streaming and joint-AND-head designs, restore the original reverse-owner reservation, and hand off implementation to separate sessions starting from main.
- **Additional authorized documentation:** create an MPL cookbook in the agent SwDD, put the separate-conjecture rule for AND heads first, and require reading it before writing MPL in both `AGENTS.md` and the project conventions.
- **Acceptance:** This handoff records both failed proof attempts and their evidence, the two design directions and correctness constraints, and the restored 4,194,304-entry source reservation. D2 proof success remains a later acceptance gate after both implementations return to this branch.
- **Authorization:** The maintainer explicitly requested restoring the previous boundary to its original value and documenting both designs here. Separate sessions starting from main will implement the changes and merge them back to D2. This handoff performs no further builds, runs, GPU redesign or AND-head implementation.
- **Baseline:** branch starts at main; tracked working tree was clean. D1's frozen predecessor ledger records its successful proof and full validation.
- **Current state:** HANDOFF READY. D2 remains two equality-head conjectures at rows 67 and 68. Both are unproved in the saved runs. The reverse-owner source reservation is restored to 4,194,304 entries (32 MiB); accepted-event and ordering ceilings remain 6,291,456. Temporary capacity-reporting code stays removed, and the sacred dump remains targeted at the complete D2 parent chain.
- **Constraints:** native Windows; full rebuild with normalized environment; all pipeline runs through `main.py --GPU`; dedicated logs and one completion-driven watcher. Preserve assertions, proof history and the sacred dump format. Any diagnostic dump target uses its full parent chain.
- **Execution requirement:** `fta_shortcut.restore` checks supplied-conjecture fingerprints. Adding D2 requires a full GPU run to refresh `files/shortcut/full_input.zip` before the requested shortcut; do not bypass that check or modify the archive.
- **Next action:** The separate main-based sessions implement bounded GPU streaming and native joint AND heads, then bring both reviewed changes back to sandbox/FTA_D2. Rebuild before any D2 run; refresh the full input capture if definitions, configuration or conjectures change, then reassess both D2 conclusions.
- **Blockers:** The current D2 workload exceeds the original reverse-owner reservation; the enlarged-owner experiment then exceeded accepted-event storage. These are resource aborts before the proof horizon, not completed logical-stall diagnoses. The on-disk executable is from the enlarged-owner experiment and is stale relative to the restored source.

## Historical expanded statement and encoding

For natural numbers, `q*d + r = q'*d + r'`, `r < d`, and `r' < d` imply both `q = q'` and `r = r'`.

Anchor slots are `1=N`, `2=zero`, `3=successor`, `4=addition`, `5=multiplication`, `6=one`, `7=two`, `8=identity`. The expanded variables are `9=q`, `10=d`, `11=q*d`, `12=r`, `13=a`, `14=q'`, `15=q'*d`, `16=r'`. The two addition premises share the same result `13`; no extra equality premise or function notation is needed. The earlier durable input supplied these two separate equality heads; the active experiment above replaces both rows.

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14,15](in3[14,10,15,5])(>[16](in3[15,16,13,4])(>[](strictOrder[1,4,12,10])(>[](strictOrder[1,4,16,10])(=[9,14]))))))))
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](in3[9,10,11,5])(>[12,13](in3[11,12,13,4])(>[14,15](in3[14,10,15,5])(>[16](in3[15,16,13,4])(>[](strictOrder[1,4,12,10])(>[](strictOrder[1,4,16,10])(=[12,16]))))))))
```

## Human proof and investigation frontier

1. Trichotomy compares `q` and `q'`.
2. Under `q < q'`, the successor bound gives `q+1 <= q'`. Multiplication monotonicity, distributivity, `r < d` and nonnegativity give `q*d+r < (q+1)*d <= q'*d <= q'*d+r'`, contradicting the shared sum.
3. The symmetric argument excludes `q' < q`.
4. Thus the quotient-equality conjecture closes. Addition cancellation gives `r = r'`, closing the separate remainder-equality conjecture. There is no final conjunction-integration step.

For a stall, map these steps to the compiled expressions and exact scopes, derive the target's full parent chain from the creation code, and use the sacred trace to locate the last present and first absent step. Check live statements, installed rules, scope compatibility, request gates and admission state. Check batch configuration before describing a mechanism as missing.

## Evidence and milestones

- Read the predecessor `D1/current_proof_state.md`, the FTA shortlist, the prover chapter's full/shortcut input contract, and the stall-investigation playbook.
- Preflight : 67 conjectures, D2 once at row 67, no unbound arguments, and the conjecture file is the only source fingerprint changed from the previous full capture. Preserved that capture as . D2 expression SHA-256: `367332dada22e23a66753bcfd637079fe5b8fd5e0afdea1f784042cbd5f15f12`.
- Maintainer correction: GL uses two conjectures for the equality heads. Replaced the source shorthand with two explicit rows, preserving the premises. Stopped the initial runner process tree before restarting; retained  and . That interrupted attempt is not a proof result.

- Corrected-input execution : native full rebuild passed in 104.675 seconds; `main.py --GPU --run-descriptor d2_split_01_full` exited 1 after 234.609 seconds, with native FTA exit 3 at burst 11. Exact failure: `hostCounters.reverseOwnerCount <= capacity_.reverseOwners` in `phase2_cuda.cu`, `CudaPhase2EvaluationBuffer::expandEvaluationWork`. The runner stopped before its shortcut stage. Saved 5,453 files in  and retained , , and the empty sacred dump/trap files. The input archive was captured before FTA; SHA-256 `90e4a44008a208675f6e9c5d11e79ace18f6dd7c513cff31fb619aa64c3a43bc`.
- MPL cookbook milestone: added `03b_mpl_cookbook.md`, with explicit AND-head splitting as rule 1 and the two exact D2 rows as its example; extended identical Rule 26 text in both operational files and linked SwDD navigation. Corrected the read MPL reference's binary conjunction, binding, schematic predicate spelling and empty-binder wording, plus the conjecturer chapter's obsolete shortcut-snapshot description.  validates both D2 examples, local cookbook links and the paired rule text. This is documentation of existing behavior; no runtime or diagram contract changed.

### Diagnostic preparation

- Derived the shared D2 target directly from `ProofGridBuilder::addTheoremToMemory` and `ce::disintegrateImplication`. Its inner-to-root chain is `strictOrder[1,4,16,10]`, `strictOrder[1,4,12,10]`, `in3[15,16,13,4]`, `in3[14,10,15,5]`, `in3[11,12,13,4]`, `in3[9,10,11,5]`, `AnchorFTA[1,2,3,4,5,6,7,8]`, empty root with null parent; each nonempty key retains its MPL parentheses. Retargeted only `infra/hashburst_dump.cpp::isTargetLB`.
- Added temporary  diagnostics: `proveKernel` records each projected block's complete parent chain; `CudaPhase2EvaluationBuffer::expandEvaluationWork` records the overflow total and independently sums synchronized request states in bounded 1,024-record host chunks, including per-block owner totals and maximum owners per request. No capacity, assertion, device kernel or proof gate is changed. These diagnostics remain until the diagnosis/fix disposition is known.
- Working hypothesis: the common 4,194,304-entry reverse-owner work buffer is too small for the D2 evaluation image. The recount and block attribution distinguish real request expansion from an inconsistent counter before any sizing decision.
- While reading required cookbooks, clarified that the string-cookbook identity migration covers converted hot paths, not every owning-string caller, and labeled the GPU evaluator's old small ceilings as historical. The current common capacity table remains unchanged.

- Diagnostic gate : clean rebuild exit 0 in 107.113 seconds; native unit gate exit 0 in 10.245 seconds. Diagnostic binary SHA-256 `fbe729efb7f2f74dca77b414ae320e4a81b8af054738b9545c5dc34271f60f7f`. The refreshed archive still matches all current input fingerprints and contains all 68 durable conjecture rows.


## Diagnosis checkpoint — 2026-09-09

**Immediate classification: capacity gap in the existing fixed CUDA evaluation buffer.** The failure is not a successful proof and not evidence that the conjecture is false. The counter agrees exactly with an independent 64-bit sum of synchronized request states. No coding-contract violation has been demonstrated by this investigation. Because execution stops early, a separate logical stall or its cause has not been established.

### Measured request expansion

The overflowing evaluation has **60,782 requests** and **4,224,638 reverse-owner work items**. The common reservation is **4,194,304**, exceeded by **30,334 (0.7232 percent)**. The recount and the device aggregate both equal 4,224,638. These are reverse-index candidate owners before the remaining-argument subset and encoded-key checks, not successful rule firings.

Every contributing block in this evaluation belongs to D2's full parent chain. `ProofGridBuilder::addTheoremToMemory` creates these as recursion block #1 under the shared last-premise block, using `(in2[recArg,digitArg,sName])`.

| Projected blocks | Induction variable | Leaf keys | Requests per block | Owners per block | Maximum owners per request |
|---|---|---|---:|---:|---:|
| 0, 3 | divisor `10` | `(in2[rec0,10,3])`, `(in2[rec5,10,3])` | 18,925 | 1,348,797 | 321 |
| 1, 4 | second quotient `14` | `(in2[rec2,14,3])`, `(in2[rec7,14,3])` | 5,733 | 381,761 | 292 |
| 2, 5 | first quotient `9` | `(in2[rec4,9,3])`, `(in2[rec9,9,3])` | 5,733 | 381,761 | 292 |

The matching counts across pairs are measurements for separate conjecture obligations; they do not establish that either induction block can be skipped.

`phase2EvaluationRequestStateKernel` normalizes each eligible request, finds its reverse-index entry, stores its owner count and adds that count to the aggregate. `CudaPhase2EvaluationBuffer::expandEvaluationWork` checks the aggregate before `phase2EvaluationEmitOwnersKernel` can fill `DeviceEvaluationOwnerWork`. `phase2EvaluationCandidateKernel` applies the remaining-argument subset check afterward. Thus the assertion is the intended boundary for a genuinely larger candidate image; it is not a device allocation failure or a counter-wrap symptom.

### Proof state and limits of the diagnosis

The sacred dump records 14 ENTRY and 13 EXIT snapshots of the shared D2 goal block. Both main-scope goals, `(=[9,14])` and `(=[12,16])`, remain pending in every snapshot. Stored statements grow from 6 to 3,223 and hash-memory original-chain entries from 0 to 635. The final ENTRY retains the four arithmetic premises, both strict remainder bounds and the derived divisor positivity `(preorder[1,4,6,10])` at `main`.

Neither quotient equality nor remainder equality appears as a live statement in the final target snapshot. The direct human-proof comparison between `9` and `14` is also absent from that block's live order statements; this identifies an unfinished proof step, not the reason it would remain unfinished after more iterations. The only non-main scope in this target has four statements under an unrelated constant-order cohort. The run aborts while processing the six induction blocks, before the configured proof horizon. Capacity must be addressed before distinguishing a normal unfinished search from a logical stall. Do not infer a missing OR mechanism or change admission policy from these incomplete snapshots.

### Evidence

- : shortcut process exit 1; native FTA exit 3; elapsed process time 37.439 seconds. Source checkpoint; diagnostic binary SHA-256 matches the passed build/unit gate.
- : FTA burst 11 and the unchanged reverse-owner assertion; verifier and connected Lean were not reached.
- : 105,261,532 bytes, SHA-256 `b4c302e6e07655eda87c47cbf7856b607842def106fc1b75a119a5c92cc01142`.
- : full projection parent chains, aggregate/recount and six per-block demand rows; SHA-256 `1b701b5720a8f2b51dea7a15d89c3481daddd5335f7134b19d864406164cf23e`.
- : structured goal trajectory, scope and predicate counts, live expected statements and owner attribution.  preserves the last target statement slice. Analyzer: .
- : preserved execution artifacts. The replay archive SHA-256 remains `90e4a44008a208675f6e9c5d11e79ace18f6dd7c513cff31fb619aa64c3a43bc`, the same refreshed 68-row input used for the failing full run.
- : all **1,703/1,703** native tests pass with the diagnostic code.

### Repair directions for a maintainer decision

1. Enlarge the active common profile's reverse-owner work reservation and rerun. For example, `kFullRunPhase2EvaluationCapacity.reverseOwners` from 4,194,304 to 8,388,608 adds 32 MiB to the eight-byte owner-work array. This exceeds the measured failed-burst demand, but later peak demand is unknown; it is not a proven final sizing requirement.
2. A bounded multi-segment owner expansion could avoid reserving the whole image at once, but that is a new execution design requiring explicit approval and ordering/provenance validation.

No capacity, proof gate or configuration was changed, and every existing assertion remains intact. Temporary projection-chain and overflow-recount diagnostics remain in `json.hpp` and `gpu/phase2_cuda.cu`; the sacred dump remains targeted at D2. This is a diagnosis-only checkpoint, so no cleanup masquerading as a completed fix or successful final verification is claimed.

## Approved capacity continuation — 2026-09-09

The maintainer approved 4,194,304 -> 8,388,608 common-profile reverse-owner entries. This adds 33,554,432 bytes to the owner-work array and leaves all other capacity fields unchanged. Cleanup removes only the two temporary reporting blocks introduced for the completed capacity diagnosis, including their diagnostic buffers and file-I/O scaffolding; all pre-existing production assertions and the sacred D2 dump remain. The clean run will provide the proof and memory-allocation result.

- Capacity gate : full rebuild passed in 105.745 seconds; all 1,703 native tests passed in 9.823 seconds. Binary SHA-256 `a3f138219657074bf445751534aa35a9f0d3fd02e6147c78d0def73dcc001088`.  verifies only the common reverse-owner field changed, the two cleaned production files match their pre-diagnostic contents exactly, and the saved input fingerprints remain valid.  verifies all three sizing documents agree on 64 MiB and the calculated total of 7,220,118,148 device bytes. Shortcut result remains pending.


## Main-based implementation handoff — 2026-09-09

The maintainer stopped further experimentation here and selected two complementary directions. These are documented designs for the separate sessions, not implementations already present in D2. Use this existing `current_proof_state.md` as the single D2 state document; do not create a competing `current_state.md` ledger.

### Completed enlarged-owner experiment

The 8,388,608-entry reservation cleared the original owner-work failure. The clean shortcut, from source, completed an evaluation with **5,511,208 reverse-owner work items**. The startup allocation report measured **7,220,118,148 fixed device bytes**, exactly the expected 32 MiB increase. All 1,703 unit tests had passed on the clean build; temporary reporting traps were absent.

The run then aborted at **FTA hash burst 12** in `gpu/phase2_cuda.cu`, `appendPooledGrowthEvent`, at **`eventIndex < capacity.acceptedEvents`**. The accepted-event reservation is **6,291,456**, and `kFullRunPhase2OrderingCapacity.events` has the matching ceiling. The exact demand of the failed pass is unknown: the assertion establishes only that it needs at least **6,291,457 events**. The later host `cudaErrorAssert (710)` at a copy is the consequence of this device assertion, not a new diagnosis of allocation failure. The maximum among completed passes was 4,959,028 accepted events, 101,979 unique requests, 5,511,208 reverse owners and 81,559 subset-surviving candidates. Completed-pass maxima do not measure the failed pass.

Both D2 equality goals remain pending in all 29 target snapshots (15 ENTRY and 14 EXIT). The last target ENTRY has 3,277 stored statements and 644 hash-memory original-chain entries. The additive-order totality theorem and its `or11` consequence are emitted at burst 11, immediately before the next burst aborts. This is progress toward the human proof's comparison step; the run does not establish that the comparison mechanism is absent or permanently stalled.

Evidence:

- : shortcut process exit 1, native FTA exit 3, 59.461 seconds; source and binary identity, artifact and trace hashes.
- : burst-11 totality result, burst-12 first device assertion and subsequent host failure.
- : measured allocation and completed evaluation counters.
- : 120,026,425 bytes; SHA-256 `72189ab61959992a113914d638a96921891a82de17dd0c0ba559202e0f732780`.
-  and : target goal trajectory and final live statement slice.
- : both the successful-owner measurements and the new accepted-event lower bound. Its proposed event enlargement was never applied.
-  is empty. The input archive remains the same 68-row capture, SHA-256 `90e4a44008a208675f6e9c5d11e79ace18f6dd7c513cff31fb619aa64c3a43bc`.

### GPU design: bounded generation, evaluation and deloading

**Maintainer proposal:** do not require all work to reside on the GPU together. Load logic blocks batchwise, and allow a large block's request generation to pause at a bounded work limit. Evaluate the available work; retire request storage after evaluation; resume generation from its saved position.

**Evaluation is resumable too.** It may fill its result storage, deload completed results, reclaim that storage, and continue evaluating. It does not have to fit the complete result set into one allocation. A request remains live until its entire expansion has been evaluated and every result still referring to it has been safely retained. Only then may its request/event storage be reused. Result deloading is storage management, not premature application of the results to the proof state.

Correctness requirements for the implementation:

1. Bound the working storage at both levels: the batch of projected logic blocks and the work stream within an oversized block. A request-count ceiling alone is insufficient: the observed limits are the accepted-event ledger before request deduplication and the owner expansion after it; frontier/prefix, candidate, generated-byte and provenance storage also need defined pause/drain behavior.
2. Treat a full processing buffer as a defined pause-and-resume state, never as permission to drop, truncate or silently skip work. Keep assertions for violated invariants and failures. Evaluation can make repeated result-deload cycles, including while finishing one large request.
3. Preserve the exact next generation/evaluation position and all still-live continuation data: call, run, path/candidate position, prefix/frontier ownership and partial owner/output progress. No candidate may be omitted or repeated accidentally at a boundary.
4. Keep the Phase 2 logical input snapshot unchanged across the batches. Generation after a drain must see the same premises, classes, scopes and rule indexes as generation before it; newly evaluated statements enter through the existing finalization contract.
5. Carry global ordering, duplicate handling, accepted-subkey/split-work counts and doom-line decisions across batches. Sorting each batch independently is insufficient unless the implementation proves that the combined order is the required order. A later doom decision must still correctly filter already deloaded output.
6. Deload complete provenance-bearing results with their canonical ordering keys and required identifiers. Do not reconstruct missing proof evidence by rerunning semantics or route proof decisions through the origin map. Do not recycle event/request identifiers while a result or continuation still references them.
7. Verify lossless results across several batch sizes and forced drain boundaries, including an oversized single block and an oversized request expansion. For unchanged inputs, preserve the processor-equivalent search and the repository's applicable proof/artifact identity contract. Measure bounded GPU memory separately from elapsed time.

Implementation entry points: `prover.cpp`, `Prover::proveKernel`; `gpu/phase2_cuda.cu`, `CudaPhase2GrowthBuffer::runRequestGrowth`, `appendPooledGrowthEvent`, `CudaPhase2OrderingBuffer::orderAndDeduplicate`, `CudaPhase2EvaluationBuffer::expandEvaluationWork`, `materializeFiringExpressions` and `selectDoomPrefixes`. Consult the GPU and statification cookbooks before selecting the continuation and deload representations. The exact queue, storage and batching design belongs to the main-based implementation session; no such machinery is added here.

### AND-head design: share induction, retain separate contradiction blocks

**Maintainer proposal:** support a native joint AND conclusion instead of always turning a conjunction into unrelated proof obligations. For D2, let `A` mean quotient equality and `B` mean remainder equality under the existing common premises.

- One induction family proves `A and B`: its induction hypothesis carries both preceding conclusions, and both the base and step obligations require both conclusions. Share the eligible-variable induction machinery and its common context instead of constructing a separate family for each head.
- Retain **two contradiction logic blocks**, one seeded with `not A` and one with `not B`, each discharging its corresponding conjunct. The maintainer explicitly corrected the earlier suggestion of halving these blocks. Do not assume `not A and not B` to prove the conjunction, and do not replace the agreed two-block arrangement with a single negated-AND OR cohort.
- Record partial closure separately. Proving one conjunct cannot mark the combined goal proved. Shared induction must not lose typing, base-case, step-case, scope or dependency obligations. Export must retain auditable evidence for both conclusions and valid introduction/projection of the conjunction as required by the chosen representation.
- Implement an actual joint goal and its induction/proof contract; do not silently substitute an input-only alias or an unaudited scheduling shortcut. The existing ordinary D2 premise chain is already shared; the intended saving is the duplicated induction auxiliaries.

The original overflowing evaluation contained six recursion-block-#1 instances in three pairs: divisor blocks `rec0`/`rec5` each contributed 1,348,797 owner candidates, and quotient blocks `rec2`/`rec7` and `rec4`/`rec9` each contributed 381,761. Their total was 4,224,638. This identifies substantial structural duplication. A joint induction family could remove paired machinery, but matching counts do not prove identical proof states, and the stronger joint induction hypothesis can change search behavior. Do not claim exactly half the memory or runtime without a measured comparison. No contradiction-block reduction is required for the proposal to be useful.

At this handoff, the D2 file retained two separate conjecture rows pending native AND-head support; the later remainder-uniqueness experiment above replaces those inputs. The MPL cookbook's first rule and Rule 26 in `AGENTS.md`/the project conventions describe the separate-head authoring restriction. An AND-head implementation must revisit those instructions and the relevant agent/user SwDD chapters in the same change, explaining the newly supported form and its proof obligations. Do not leave the blanket prohibition as the permanent architecture after adding joint goals.

### Restored boundary and integration contract

The maintainer explicitly requested restoring the original boundary. `kFullRunPhase2EvaluationCapacity.reverseOwners` is therefore **4,194,304** again: **32 MiB** of eight-byte owner-work entries. The experimental D2-headroom test is removed with that sizing decision; the owner-record size check and all production overflow assertions remain. Growth accepted events and ordering entries stay at **6,291,456**. The current sizing documents are synchronized to the restored source, while all experiment evidence and commits are preserved.

No build or pipeline run is performed after this restoration. The existing executable, SHA-256 `a3f138219657074bf445751534aa35a9f0d3fd02e6147c78d0def73dcc001088`, still contains the 8,388,608-owner experiment; it must not be mistaken for a build of the restored source. The restored allocation's prior measured total is 7,186,563,716 bytes, not a new post-restoration measurement.

Separate sessions starting from main will address GPU streaming and AND heads on their own implementation branches, then merge the reviewed changes back. This session does not create those tasks, switch branches or land work on main. Preserve D2's inputs, evidence and full-chain dump target during integration. After integration, rebuild and reassess both equality conclusions; changing D2 to a joint head or changing definitions/configuration requires refreshing the full FTA input capture before shortcut replay. Final proof acceptance must distinguish actual theorem registration and validated proof evidence from merely emptied local goals.

## Main integration — 2026-09-10

Source merged without conflicts. Four documentation conflicts preserve the D2 historical decision and main's current bounded execution, allocation and runtime descriptions. The automatic-planner paragraph and both operational GPU run rules now match production and I-211. Relative to main, runtime source differs only in the inherited evaluation-capacity comment; the inherited projection test retains its eight-byte owner-record assertion. Both D2 supplied rows remain present. No capacity or proof policy was changed during this integration. Main and D2 already share the same sacred dump target.

## Retest evidence — 2026-09-10

The exact assertion is `carveCursor_ < totalBlocks_` in `memory_infra/global_memory_manager.cpp`, with the message naming `static_pool_bytes`. The exhaustion report identifies pool kind 0 (Main), `totalBlocks=49152`, `blocksInUse=49152`, `blockBytes=262144`, and no recycled block. This is the host pool configured by `parameters.hpp` at 12,884,901,888 bytes, not a CUDA device-allocation or fixed event/owner-buffer assertion.

The existing census attributes 31,853 blocks (8,350,072,832 bytes, approximately 7.78 GiB) to request-generation scratch and 17,265 blocks (4,525,916,160 bytes) to grid logic blocks. It also reports 33 string-scratch blocks and one unattributed block. Scratch reads are explicitly labelled racy, so this is the abort-time census, not an exact allocation-owner diagnosis. No paging eviction occurred; the CUDA route remains resident-only. Further classification requires a separate investigation, not an inferred capacity increase.

The completed burst-16 second CUDA pass processed 13,303,450 accepted events with an 8,388,608-entry event window, 346,208 requests and 315,516 sealed firing records. Bounded execution therefore crossed the previous event ceiling without asserting. Burst 16 completed in 162.859 seconds; its recorded Phase 2 time was 12.9452 seconds and Phase 3 time was 147.494 seconds. Burst 17 then aborted before another completed CUDA-pass report.

The last target snapshot preserves `(=[9,14])` and `(=[12,16])` as the two unresolved goals. The saved trace contains 225,057,513 bytes with SHA-256 `3f1e43a6f0c2f54c122507b3f2ef957e9abdaefb056b8b90588a799d509e79b7`. Tested executable SHA-256: `432da457a0e723cd44ee2f02915741c974cecadf981adc1bb2e8dc53d972a90a`. No temporary diagnostics were introduced; `traps.log` remains empty.

The first Node completion watcher lost its tool transport at 300 seconds. Its kernel was reset and a native process-exit waiter replaced it without restarting the runner. The maintainer explicitly requested progress, authorizing the two live checks; final evidence was inspected after process completion.

While answering the proof-method question, the read prover chapter was found to describe an obsolete separate typing block and `typingProofOnly` flag. Its induction section now documents the existing exact-main typing lookup in `TheoremPublisher::updateGlobal` and the parent-block proof export. This is a documentation correction to current source, not a prover change, and requires no new execution.

## 2026-09-10 — Stall playbook steps 1 and 2: proof map and target

**Step 1 complete.** Read the current D2 statement and human argument, `D1/current_proof_state.md` (current outcome, glossary and witness/induction conventions), the D2 entry and supporting A/B lemmas in `fta_shortlist.md`, and the current predicate definitions. D2 has no separate human-proof file; the argument below adapts the ledger's quotient-comparison proof to the remainder-predicate input. The D1 glossary contains historical compiled names; the current matching `GL_binary_FTA.json` defines the D1 wrapper as `existence17` and trichotomy as `or12`.

| Human step | Expected live expressions and scope |
|---|---|
| Expose the two decompositions | At the final premise block's `main`: `in3[q,10,t,5]`, `in3[t,11,9,4]`, `strictOrder[1,4,11,10]`, and the corresponding `q',t',12` rows. The saved concrete names are `q=int_lev_1_0`, `t=int_lev_1_1`, `q'=int_lev_2_0`, `t'=int_lev_2_1`; this is a run-local map, not a guessed map for the contradiction child. |
| Compare the quotients | `or12[1,4,q,q']` at `main`, with leaves `strictOrder[1,4,q,q']`, `=[q,q']`, `strictOrder[1,4,q',q]`. Each live leaf has its own `_ordis_` scope; sibling premises cannot be combined. |
| Exclude `q<q'` | At that leaf's scope: obtain `s(q)<=q'`; use `r<d`, multiplication monotonicity and distributivity to derive `q*d+r < s(q)*d <= q'*d <= q'*d+r'`. The equal sums are both `9`, giving a branch contradiction. An equivalent difference-witness argument is allowed, but its exact witnesses and rules must be identified. |
| Exclude `q'<q` | The symmetric argument at the other strict-order leaf's scope. |
| Close the equality case | Under `q=q'`, rewrite the two products to equality and apply addition cancellation to obtain `=[11,12]`; converge the cases to `main`. In the negated-head block this conflicts with the seeded `!(=[11,12])`, discharging the original conjecture. |

The original saved direct target has both decompositions and the quotient trichotomy. Its `q<q'` branch is live; the equality and reverse strict-order leaves are pending. This identifies the branch arithmetic as a candidate frontier, not yet its cause. The only induction candidates remain `a=9` and `d=10`; the direct witness comparison does not imply induction on those existential witnesses.

**Step 2 complete.** Choose the negated-head contradiction child to examine the explicit reductio route. The source creation site is `proof/proof_grid_builder.cpp`, `ProofGridBuilder::addTheoremToMemory`, under `try_contradiction_negated_head`: it sets `negHead=negate(head)`, `contradictionKey="__contradiction__"+negHead`, links the child to the final premise block, and sets its expression key to that same string. `ConfigFTA.json` enables this route; its separate positive-head contradiction route is disabled.

Full target chain, innermost first:

```text
__contradiction__!(=[11,12])
(remainder[1,4,5,9,10,12])
(remainder[1,4,5,9,10,11])
(AnchorFTA[1,2,3,4,5,6,7,8])
<empty root; parentMemory == nullptr>
```

Next: step 3 changes only `hashburst_dump.cpp`, `isTargetLB`, to this chain, then performs the native full rebuild and unit-test gate before committing and pushing. Step 4 performs one FTA GPU shortcut with descriptor `d2_stall_neghead_20260910_01` and preserves its evidence. The maintainer directed that all FTA runs from this point use `main.py --shortcut --GPU`; keep the fingerprint check intact and do not claim a new connected Lean result from a shortcut. The earlier direct trace remains under its distinct saved name.

## 2026-09-10 — Stall playbook step 3: target gate complete

The sacred matcher now selects only the complete negated-head child chain recorded in step 2.  proves the parsed conjecture/parent chain matches the source matcher and that no source bytes outside the matcher changed. The 266-file comparison against the completed remainder run differs only in `infra/hashburst_dump.cpp`; conjectures, definitions and configuration are unchanged.

Native full rebuild passed in **105.375 seconds**; all **1,724 unit tests** passed in **10.076 seconds**. Completion record: ; separate `_build_build.log` and `_build_unit_tests.log` retain both outputs. One completion watcher reported the result; no running logs were inspected. Commit and push this coherent diagnostic milestone before step 4.

The maintainer's updated execution instruction is authoritative: all subsequent FTA runs use `main.py --shortcut --GPU`. The full-input capture comes from the completed remainder-uniqueness run, and the ordinary restore fingerprint assertion remains intact.

## 2026-09-10 — Stall playbook steps 4 and 5: contradiction trace and frontier

Step 3 landed as before execution. The input archive's source fingerprints match the unchanged supplied goals, definitions and configuration.

**Step 4 complete:** `main.py --shortcut --GPU --run-descriptor d2_stall_neghead_20260910_01_shortcut` exited **0** after **69.382 seconds** (reported runtime **69.27811 seconds**); FTA prover time was **55.4818 seconds**. All **1,724 native tests**, **423 verifier unit tests** and **26,557 shortcut proof checks** passed. FTA's main-pool peak was **12,955 / 49,152 blocks**. The shortcut theorem and processed global lists are byte-identical to the preceding full remainder run; the new goal is still absent. This shortcut performs no connected Lean check.

Evidence: , , and the descriptor's `_diagnostics.log`, `_hashburst.txt`, `_inputs.json`, `_artifacts.json`, `_frames.json`, `_final_sections.json` and `_frontier.json`. The saved sacred trace is **502,154,473 bytes**, SHA-256 `909d9f85c5c4422eea81e7aaa94ab4ab48681b9ca88feb81c0317275c4b2a68e`. The completion watcher reported the atomic record; no running output was inspected. No temporary traps were added.

**Step 5 complete:** The negated-head block has **52 snapshots**, all with `toBeProved=0`, `primedForContradiction=1`, `isActive=1`, and statements only at `main`. An empty explicit goal registry is not proof completion: this block seeks contradiction from its seed.

| Human-proof evidence | First live occurrence | Final state |
|---|---|---|
| Seed `!(=[11,12])` | ENTRY #1 at `main` | Present |
| Both quotient/product decompositions, quotient typing and remainder bounds | EXIT #2 at `main` | Present |
| `or12[1,4,int_lev_1_0,int_lev_2_0]` | EXIT #20 at `main` | Present |
| Either strict quotient-order case, or quotient equality | Never | Absent at every scope |
| `=[11,12]` | Never | Absent |

The last present step is the quotient trichotomy; the first absent step is opening a quotient case. Final EXIT #26 has **9,799 statements**, **4,658 hash-memory originals**, zero OR bookkeeping/count/pending entries and zero OR parks. The trichotomy's generating rule is present in hash-memory originals and both quotient typings are live at the same `main`, so this is not failure to generate the comparison theorem.

## 2026-09-10 — Stall playbook steps 6 and 7: causal classification

### Negated-head route: explicit-goal requirement prevents every case split

The creation site in `proof/proof_grid_builder.cpp`, `ProofGridBuilder::addTheoremToMemory`, seeds the negated head with status 0, without registering a status-2 goal. In `CaseProcessor::consumeOrLeavesCohort`:

```cpp
const bool lbHasGoals = !memoryBlock.intToBeProved.empty();
const bool orAdmitted = lbHasGoals && (routeB || probeHit || demandHit);
```

The trace establishes `lbHasGoals=false` throughout. Opening is therefore false regardless of admission-route results. Parking also requires `lbHasGoals`; `drainPendingOrReleases` clears its staging and returns when the goal registry is empty. The live flat OR rules remain available, but no assumed-case scope exists. This agrees with the documented [D-306](../../agentic_swdd/40_decisions.md#d-306) restriction and [OR branching](../../agentic_swdd/20_core_concepts/07_or_branching.md); no coding-contract violation is established.

Configuration survey:  covers every `Config*.json` with prover parameters. FTA explicitly enables `try_contradiction_negated_head`, uses `max_or_depth=1`, `maxNumberSecondaryVariables=2` and `maxIterationNumberVariable=1`. The first split would be at depth zero. None of these settings bypasses the empty-goal gate. `allow_multiplication=false` controls implication partitioning, not the existence of arithmetic multiplication; it is not evidence that the `in3[...,5]` arithmetic mechanism is disabled.

**Classification:** a documented architectural restriction of the explicit contradiction route, not a missing trichotomy rule, memory failure, or disabled contradiction flag. It does not by itself explain the direct route, which has an explicit equality goal and already opens the first quotient case.

### Direct route: the first quotient case cannot use its difference witnesses

Continue from the earlier preserved direct target, whose complete parent chain is recorded above. Its exact live scope is:

```text
main_boundary_ordis_(or12[1,4,int_lev_1_0,int_lev_2_0])_((strictOrder[1,4,int_lev_1_0,int_lev_2_0]))
```

At this scope, `q<q'`, `q<=q'` and even `q*d<q'*d` are live. The next difference-witness step does not enter live arithmetic. `disintegrateExpr2` has generated and parked both versions:

```text
rejectedMap:
  key=(in3[int_lev_1_0,marker,int_lev_2_0,4])
  concrete=(in3[int_lev_1_0,it_0_lev_2_4408,int_lev_2_0,4])
  source=(strictOrder[1,4,int_lev_1_0,int_lev_2_0])
rejectedMapIntegration:
  key=(in3[int_lev_1_0,marker,int_lev_2_0,4])
  concrete=(in3[int_lev_1_0,int_lev_2_4409,int_lev_2_0,4])
  source=(strictOrder[1,4,int_lev_1_0,int_lev_2_0])
```

Both entries carry the exact branch scope and their witness-typing sibling. No `in3[q,gap,q',4]` fact is live at any scope. The marked key is absent from the entire algebra admission map, including `main`, so `isAdmittedIncludingAncestors` cannot admit it. The integration map likewise lacks `(in3[u_int_lev_1_0,marker,u_int_lev_2_0,u_4])` at the branch, and the integration admission set is empty. `isAllowedAsOperatorInput` cannot supply the alternative: `ConfigFTA.json` gives `in3` two inputs, while [I-6](../../agentic_swdd/30_invariants.md#i-6) permits exactly one. Thus both witness bodies remain parked under the existing Pass-B contract.

The absent admission key is the immediate stopping condition; the dump does not establish why every possible future demand-producing proof route fails. The algebra witness has generation zero, within the configured limit; no count/depth increase is justified by this evidence. The first quotient case remains unresolved, so the sequenced cohort retains its equality and reverse-order leaves as pending. Evidence: , , and the preserved full-run trace/section extract.

**Classification:** a demonstrated admission-policy limitation on this direct difference-witness route. No incorrect lookup, lost admitted fact or violated coding contract has been shown. This and the explicit-goal OR restriction are concrete blockers; they are not a proof that one change alone would close D2. The two induction families have not been individually traced in this investigation.

## 2026-09-10 — Stall playbook step 8: recorded result and decision boundary

Steps 1–8 are recorded with exact source predicates, scopes and evidence. The requested diagnosis is complete for the inspected direct and negated-head routes. Remainder uniqueness is still unproved. Production inference, supplied conjectures, capacities and configuration are unchanged; the only source edit is the sacred target matcher, which remains on the explicit contradiction block. No temporary traps remain because none were added.

Possible next directions for a maintainer decision:

1. Investigate which proved arithmetic rule should demand `q+gap=q'` at the live direct branch, then distinguish missing rule coverage from a rejected demand before selecting any admission change. Preserve the single-input invariant.
2. If contradiction blocks are intended to perform case analysis, design a represented contradiction obligation with branch-resolution and provenance rules; changing only the opening boolean is insufficient because release and resolved-branch handling also assume explicit goals.
3. Trace the dividend and divisor induction families separately before attributing the entire unproved result to only the two inspected routes.

These are investigation/design directions, not approved fixes. No additional run or semantic/configuration change follows this diagnosis.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
