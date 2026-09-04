<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Full-run GPU plan

> **Purpose:** this is the persistent working memory for the full-run GPU
> campaign. Read it after every context compaction. Update it after every
> measurement, design conclusion, failed experiment, commit, and change of next
> action. Conversation memory is not authoritative.

## Current state

- **Branch:** 
- **Worktree:** 
- **Base:** current `main`
- **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU, native `sm_89`
- **Status:** configuration policy, resident-only steward, full-run CUDA
 streaming, zero-work omission, exact immutable-column sharing, selected
 hash-memory projection, and first CUDA CE orchestration are implemented and
 exact across complete runs, but the current source milestone is not committed
- **Active milestone:** M2/M3 — split GPU filtering/growth from evaluation so
 the dominant encoded-statement state is projected only for surviving LBs
- **Next action:** commit the exact infrastructure milestone, then implement the
 survivor-first projection split and generation-tracked device retention

## Objective and acceptance

Make the current deterministic CUDA Phase 2 route the default for the complete
GL pipeline, including every incubator batch, every Peano and Gauss main batch,
and every counterexample-filter proof attempt.

The finished branch must satisfy all of the following:

1. `parameters.hpp` exposes a boolean `use_gpu` parameter whose global default
 is `true`.
2. Every batch configuration may override `use_gpu` to `false`; that batch then
 uses the processor route for Phase 2 and counterexample filtering.
3. SSD deload has an explicit global boolean policy parameter if no equivalent
 live parameter already exists. Every batch configuration may override it.
4. GPU use and SSD deload are mutually exclusive for this campaign. Selecting
 both is an assertion failure at configuration ownership, never a fallback or
 silent policy rewrite. Enabling SSD deload in GPU mode is outside scope.
5. A GPU-selected batch performs its counterexample filtering on the GPU too.
 The implementation must avoid unnecessary host/device transfer and must not
 route counterexample-filter proof semantics back through the processor.
6. Every retained theorem, proof artifact, and verifier result is identical to
 the current-main processor reference. The theorem count and verifier check
 count must match exactly, with zero verifier failures.
7. Output remains deterministic across repeated GPU runs and exactly matches the
 processor route at the semantic-artifact SHA-256 boundary.
8. The complete Phase 2 bracket for every incubator and Peano/Gauss main batch
 reaches at least the shortcut campaign's approximately ten-times acceleration
 over its corresponding processor reference.
9. Counterexample-filter Phase 2 reaches the same ten-times target. The campaign
 also measures the complete counterexample-filter bracket and pursues a
 ten-times overall improvement where the measured removable time permits it.
10. The final result uses `main.py` for every pipeline or reproduction run, a
 full Windows Release rebuild, all native unit tests, and a zero-failure
 verifier run. No direct executable run is permitted except `--unit-tests`.

## User-authorized architecture boundary

This is an experimental research branch. The maintainer explicitly authorized
architecture decisions needed to extend CUDA across the full run, especially
the counterexample filter, without a separate approval checkpoint. The branch
must still preserve deterministic semantics, exact output, assertion-based
failure, fixed ownership, the processor oracle, and the documentation family.

## Inherited CUDA baseline

Current `main` already contains the initial CUDA port, `gpu_rt1`, and the
accepted `gpu_rt2` candidate-window work.

- The initial same-tip shortcut result was 96.212168 processor seconds versus
 19.835229 CUDA seconds for complete Phase 2.
- `gpu_rt1` reduced accepted complete CUDA Phase 2 to 9.74343 and 9.46378
 seconds while a fresh processor reference measured 100.043 seconds: 10.27 and
 10.57 times faster, with 9,994 verifier checks and 311 exact artifacts.
- The accepted `gpu_rt2` candidate-window rerun measured 8.8462047 seconds for
 complete Phase 2, 5.94762836 seconds of device work, 114.337 seconds of prover
 time, 119.41707 seconds overall, 9,994 verifier checks with zero failures, and
 exact identity for all 311 shortcut artifacts.
- The CUDA route currently remains explicitly selected through
 `--phase2-backend cuda` and is restricted by the Python entry point to
 `main.py --shortcut`. Removing that orchestration restriction is necessary but
 is not by itself acceptance: full-run capacities, semantics, and performance
 must be established from fresh evidence.

Historical performance values are context only. Every acceptance claim on this
branch uses fresh same-tip processor and GPU measurements. Numeric timing parsers
must accept scientific notation with `[0-9.eE+-]+`.

## Non-negotiable contracts

1. The processor route remains a selectable oracle for every batch.
2. GPU mode has no processor fallback. Missing hardware, fixed-capacity
 exhaustion, allocation failure, launch failure, or ownership violation
 asserts at the responsible boundary.
3. Observable request order, provenance, dependency order, doom selection,
 witness mint order, and canonical artifact bytes remain unchanged.
4. Physical CUDA append order, launch order, and transfer scheduling never enter
 semantics.
5. Production CUDA ownership is fixed-capacity and assertion-checked. No live
 resize, truncation, replay, unified-memory fallback, or silent batch split may
 change the logical route.
6. GPU mode must not enter any SSD-deload door. The mutual exclusion assertion
 is backed by direct tests of the valid and invalid parameter combinations.
7. Counterexample-filter conjectures preserve their private single-use logical
 blocks and disjoint contradiction-table slots. GPU batching may change the
 physical schedule only when those ownership boundaries remain exact.
8. A new main-pipeline function receives full GL-style Doxygen documentation and
 direct positive and negative unit coverage in the same commit.
9. Source and configuration milestones are complete snapshots, committed with
 `git add -A`, detailed messages, and pushed immediately.
10. Every hardware-relevant architecture milestone updates the agent SwDD, user
 SwDD, and MPU booklet when their current explanation or diagram would become
 wrong.

## Measurement strategy

The campaign is measurement-first and workload-specific.

1. Preserve a current-main processor reference tree and record theorem counts,
 verifier counts, semantic file hashes, per-batch Phase 2 totals, complete
 counterexample-filter brackets, prover time, and overall time.
2. Run a current-main CUDA shortcut control first to prove the inherited route
 and reproduce the approximate ten-times Phase 2 boundary on this hardware.
3. After full-run configuration routing is valid, run CUDA through the smallest
 complete full-run batch that establishes real capacities without narrowing
 semantics.
4. Record per-batch projection sizes, scheduled calls, frontier maxima, growth
 attempts, output counts, transfer bytes, fixed allocation, device work,
 route time, and complete Phase 2 time. Capacity changes are census-backed and
 assertion-checked.
5. Profile the counterexample-filter route before changing its ownership. Rank
 projection, transfer, filter/growth/evaluation kernels, synchronization, host
 sealing, clone construction, and teardown by exact wall-time contribution.
6. Select only changes whose measured removable time can reach the remaining
 target. Re-profile after every retained large change.
7. Preserve every failed performance experiment and its evidence in the failure
 ledger. Fix forward; never discard branch work without maintainer consent.

## Counterexample-filter investigation questions

The first design audit must answer these from current code and measurements:

1. Which state is shared across all private counterexample-filter clones, and
 which projected columns differ per conjecture?
2. Can the immutable fact/rule projection be uploaded once per batch while only
 per-conjecture goal and contradiction state is staged for each GPU item?
3. Can multiple clone Phase 2 tasks share one global filtering/growth/evaluation
 launch without changing the private-LB and contradiction-table ownership
 contract?
4. What is the stable semantic order across conjecture ordinal, iteration,
 logical block, call, path, and candidate, and where is it reconstructed now?
5. Which host/device downloads are final decisions needed by Phase 3, and which
 are merely intermediate frontier control that can remain resident?
6. Does counterexample-filter teardown currently destroy reusable CUDA
 projection state between conjectures or iterations?
7. Which fixed capacities are safe for every Peano, Gauss, and incubator
 counterexample-filter batch, and what exact census proves them?

## Milestones

### M0 — branch, ledger, call-path audit, and references

- [x] Create from current `main` in a dedicated worktree.
- [x] Read the SwDD navigation and complete invariant quick-reference.
- [x] Read the prior GPU campaign work practices and accepted evidence.
- [x] Create this persistent plan and progress ledger.
- [x] Read the complete statification container, string, and GPU cookbooks.
- [x] Trace configuration ownership and every SSD-deload gate.
- [x] Trace full-run Phase 2 selection through Python and C++.
- [x] Trace the complete counterexample-filter clone, scheduling, and Phase 2
 lifecycle.
- [x] Full Windows Release rebuild and all native unit tests.
- [ ] Fresh current-main CUDA shortcut control with exact retained artifact
 manifest.
- [ ] Fresh current-main processor full-run reference with per-batch timing,
 theorem counts, verifier counts, and exact retained artifact manifest.

### M1 — batch parameters and hard mutual exclusion

- [x] Add global-default `use_gpu = true` and the SSD-deload policy parameter or
 reuse the proven equivalent live parameter.
- [x] Load both from every batch configuration with per-batch override support.
- [x] Assert that GPU and SSD deload are not both enabled at the single owning
 configuration boundary and at any lower boundary whose direct contract needs
 a tripwire.
- [x] Route every full-run prover subprocess from the batch parameters instead
 of a shortcut-only command-line policy.
- [x] Add direct positive and negative unit tests plus configuration tests.
- [x] Full rebuild, native tests, and GPU full-pipeline smoke; processor smoke
 remains pending.
- [ ] Update configuration and design documentation; commit and push.

### M2 — full-run CUDA correctness and capacity

- [ ] Run every incubator and Peano/Gauss main batch through CUDA Phase 2.
- [ ] Establish exact fixed capacities from complete live censuses; assert every
 ceiling and remove observation overhead from production specialization.
- [ ] Match processor theorem counts, verifier counts, and semantic artifact
 hashes exactly.
- [ ] Repeat the complete GPU run and prove deterministic artifact identity.
- [ ] Record per-batch Phase 2 and device-route timing; optimize any batch below
 the ten-times target using measured attribution.
- [ ] Update the complete design-document family; commit and push coherent
 milestones immediately.

### M3 — GPU counterexample filter

- [ ] Establish a processor counterexample-filter reference per batch: input
 conjectures, filtered survivors, contradiction decisions, Phase 2 time, and
 complete filter time.
- [ ] Add the direct deterministic GPU twin for the selected shared-projection
 and private-clone ownership design.
- [ ] Keep immutable batch state resident and stage only measured per-conjecture
 deltas when the code and profile prove that boundary exact.
- [ ] Execute all Phase 2 counterexample-filter semantics on GPU whenever
 `use_gpu` is true, with no processor fallback.
- [ ] Match every contradiction decision and filtered-conjecture byte exactly.
- [ ] Reach at least ten-times Phase 2 acceleration for every counterexample
 filter and pursue ten-times complete-filter acceleration from measured seams.
- [ ] Repeat the complete filter and prove deterministic identity.
- [ ] Update the complete design-document family; commit and push coherent
 milestones immediately.

### M4 — final full-run acceptance

- [ ] Full clean Windows Release rebuild and all native unit tests.
- [ ] Fresh GPU-first and processor full runs through `main.py`.
- [ ] Exact theorem counts and verifier check counts with zero failures.
- [ ] Exact semantic SHA-256 identity between processor and GPU outputs.
- [ ] Repeated GPU full-run artifact identity.
- [ ] At least ten-times complete Phase 2 acceleration for every incubator,
 Peano, Gauss, and counterexample-filter batch.
- [ ] Complete counterexample-filter acceleration reported separately and the
 ten-times overall target met or bounded by exact remaining attribution.
- [ ] No SSD deload activity in GPU mode, with the invalid combination asserted.
- [ ] Final transfer, capacity, route, device, prover, verifier, and overall
 evidence recorded here.
- [ ] Agent SwDD, user SwDD, MPU booklet, configuration documentation, and this
 ledger reconciled with the final source.
- [ ] Every source/configuration milestone pushed; branch clean.

## Verification command pattern

Use the canonical clean Windows child environment already recorded in
`gpu_rt1_plan.md` and `gpu_rt2_plan.md`, with one case-insensitive `Path`, an
explicit CUDA 13.3 `CudaToolkitDir` ending in `\`, and worktree-local `TEMP` and
`TMP`.

```powershell
MSBuild.exe GL_Quick_VS\GL_Quick.sln -p:Configuration=Release -p:Platform=x64 -t:Rebuild -verbosity:minimal
GL_Quick_VS\GL_Quick\gl_quick.exe --unit-tests
python main.py --shortcut --phase2-backend cuda --run-descriptor <descriptor>
python main.py <full-run arguments established by the implemented batch parameters>
python verifier.py
```

Every pipeline run writes a distinct ; no log is
reused. Timing sums use anchored rows and `[0-9.eE+-]+`. Artifact comparisons
cover relative path, byte length, and SHA-256, not only row counts.

## Evidence ledger

| Date | Commit or worktree | Evidence | Result |
|---|---|---|---|
| 2026-08-30 |, | `git worktree add -b sandbox/GPU_main_run C:\Users\nikol\pycharmprojects\GL\.worktree\GPU_main_run main` | Dedicated clean worktree created from current local and remote `main` tip |
| 2026-08-30 | inherited `gpu_rt1` evidence | Fresh processor 100.043 seconds versus CUDA 9.74343 and 9.46378 seconds complete shortcut Phase 2 | Accepted prior route established 10.27 and 10.57 times acceleration, 9,994 checks, zero failures, and 311 exact artifacts |
| 2026-08-30 | inherited accepted `gpu_rt2` evidence | Candidate-window rerun | 8.8462047 seconds complete shortcut Phase 2, 9,994 checks, zero failures, and 311 exact artifacts |
| 2026-08-30 | dirty M1/M2/M3 worktree | Full CUDA 13.3 Release rebuild plus `gl_quick.exe --unit-tests` | Build succeeded; 1,560 of 1,560 native tests passed |
| 2026-08-30 |  | First streamed IncubatorPeano1 warm-up burst, 1,767 active logical blocks | Six CUDA chunks completed; complete Phase 2 bracket 0.186074 seconds |
| 2026-08-30 |  | IncubatorPeano1 and IncubatorPeano2 complete prover execution | All 50 iterations of each completed through CUDA; complete Phase 2 cumulative times 77.1871 and 9.63342 seconds before the pipeline advanced through Peano |
| 2026-08-30 |  | Peano CE passes, 1,001 then 215 conjectures | CUDA Phase 2 2.89291 and 0.934525 seconds; complete filters 6.74274 and 2.0158 seconds; 215 then 171 survivors |
| 2026-08-30 |  | First run through all five incubators and into Gauss main | IncubatorPeano1/2 complete Phase 2 82.914/9.409 seconds and overall 134.548/23.2827 seconds; IncubatorGauss1/2/3 complete Phase 2 13.831/3.763/5.002 seconds and overall 103.268/30.4817/91.1764 seconds; Peano main overall 56.7257 seconds; Gauss reached iteration 70 before a 1,048,576-node growth-frontier assertion |
| 2026-08-30 |  | IncubatorPeano1 Phase 2 attribution | 82.914 seconds total across 1,313 CUDA chunks: 46.301 seconds host preparation including 32.865 seconds projection, 19.092 seconds device work, 5.066 seconds finalization, and about 11.45 seconds capacity planning/orchestration outside chunk timers |
| 2026-08-30 |  | Peano CE passes, 1,001 then 215 conjectures | CUDA Phase 2 2.96435 and 0.930068 seconds; complete filters 6.96107 and 2.07383 seconds; 215 then 171 survivors |
| 2026-08-30 |  | First capacity-clean default-GPU full run; machine partially occupied | All seven prover batches completed through CUDA, all proof graphs passed 139,241 checks with zero failures, and overall runtime was 666.65492 seconds. The maintainer reported concurrent machine load, so every timing from this run is impure and excluded from performance acceptance; the run remains valid only for correctness and capacity evidence. |
| 2026-08-30 |  | Zero-work omission full-run semantic gate; machine initially occupied | All seven batches completed and all proof graphs passed the exact 122,994 incubator plus 16,247 main checks, 139,241 total, with zero failures. Overall runtime was 607.13925 seconds, but the run began during unrelated machine use and remains correctness-only. |
| 2026-08-30 | ,  | Constant-time exact projection census | Full CUDA 13.3 Release rebuild succeeded and all 1,560 native tests passed after replacing repeated key-byte census walks with mutation-maintained exact counters. |
| 2026-08-30 |  | First full run begun after the machine became exclusively available | All seven batches and proof graphs completed through CUDA with the exact 122,994 incubator plus 16,247 main checks, 139,241 total, and zero failures. Overall runtime was 614.21430 seconds. IncubatorPeano1 Phase 2 was 83.8572 seconds versus the 44.444-second processor reference, proving the missing win is architectural rather than measurement contamination. |
| 2026-08-30 | , ,  | Exact sharing of repeated immutable name and rule-string columns | Full CUDA 13.3 Release rebuild and all 1,561 native tests passed. The clean full run completed all seven batches and 139,241 checks with zero failures in 580.84874 seconds. Phase 2 totals became 66.350704, 7.874544, 9.109199, 11.446142, 3.113738, 4.023321, and 12.950826 seconds: a real but insufficient 1.12-1.56-times improvement. |
| 2026-08-30 | , ,  | Exact executable-family hash-memory selection | Full CUDA 13.3 Release rebuild and all 1,561 native tests passed. The clean full run completed all seven batches and 139,241 checks with zero failures in 607.36436 seconds. Phase 2 totals were 80.586939, 9.638415, 8.755005, 10.831017, 2.805991, 3.592373, and 13.234814 seconds. The selected-away maps reduced aggregate IncubatorPeano1 entries by only 10,896 of 130,188,316 and key bytes by only 1,303,168 of 17,657,766,596, so selection is exact but not the missing acceleration. Peano CE retained 215 then 171 conjectures in 7.32097 and 2.27957 seconds; Gauss CE retained 80 in 7.79676 seconds. |
| 2026-08-31 |,  | Diagnostic Phase 1 and Phase 3 barrier-wall brackets on the exact `gpu-main-run-best-607s` source tree | Full CUDA 13.3 Release rebuild and all 1,561 native tests passed. The idle-machine full run completed 139,241 checks with zero failures in 574.64098 seconds. Across the seven prover calls, Phase 1 consumed 151.402187 seconds, Phase 2 consumed 115.657140 seconds, Phase 3 consumed 119.794726 seconds, and 32.675847 seconds remained elsewhere inside the 419.529900-second prover wall. Phase 1 is now the largest aggregate barrier, but IncubatorPeano1 remains Phase-2-dominated and the two final Gauss-family provers are Phase-3-dominated. |
| 2026-08-31 |, , ,  | Contention-free internal Phase 1/3 attribution on the exact tagged semantic tree | Full CUDA 13.3 Release rebuild and all 1,561 native tests passed. The idle-machine full run retained 215/171 Peano and 80 Gauss CE survivors and completed 139,241 checks with zero failures in 576.10843 seconds. Phase 1 is routing external-statement absorption in every non-CE prover, reaching 99.28 and 99.66 percent of worker time in IncubatorGauss1/2. The long IncubatorGauss3 and Gauss Phase 3 barriers are low-parallelism internal-statement absorption plus equivalence application and subtree wipes. |

## Decision ledger

| Date | Decision | Evidence and consequence |
|---|---|---|
| 2026-08-30 | Treat `use_gpu` as a batch-owned semantic execution policy with global default `true` | Required by the maintainer; processor remains the explicit `false` route and oracle |
| 2026-08-30 | GPU and SSD deload are mutually exclusive for this campaign | Required by the maintainer; invalid selection asserts rather than rewriting either choice |
| 2026-08-30 | A GPU-selected batch includes its counterexample filter | Required by the maintainer; a mixed GPU prover/processor filter is not an accepted implementation |
| 2026-08-30 | Architecture decisions are authorized on this experimental branch | Maintainer explicitly delegated the counterexample-filter and full-run GPU design; exactness and evidence gates remain binding |
| 2026-08-30 | Stream full-run CUDA work through deterministic capacity-bounded chunks | IncubatorPeano1 exposes 1,767 active logical blocks versus shortcut's 512-block allocation; split task families stay intact and all 24 projection columns plus task columns bound the greedy chunks |
| 2026-08-30 | Keep GPU mode resident-only despite permission to consider SSD deload | Current failures are CUDA fixed-output ceilings, not host residency; SSD traffic would add an unrelated transfer seam and is not needed yet |
| 2026-08-30 | Enable the statistics-driven LB split in every incubator configuration | Maintainer explicitly removed the prior incubator restriction; `isStraggler` still leaves balanced small LBs unsplit, while measured heavy LBs may fan out deterministically |
| 2026-08-30 | Replace full projection per iteration with resident projection plus deltas | IncubatorPeano1 spends only 19.092 of 82.914 Phase 2 seconds on device work; projection/preparation and 1,313 chunk routes dominate, so kernel-only tuning cannot reach ten times |
| 2026-08-30 | Omit structurally impossible CUDA request batches and whole zero-work tasks | The processor executes an empty mandatory search cheaply, while CUDA previously projected the complete LB and launched the complete device route. A request batch whose every mandatory term contains an empty view cannot produce a candidate, firing, doom line, or submatch; omitting it preserves exact output and zero work tally. |
| 2026-08-30 | Project only hash memories named by executable request batches | Each request batch reads one whole-key/subkey pair, but the first full-run route copied all four pairs for every LB. The overall encoded and remaining-argument evaluation tables remain universal; unselected whole-key/subkey pairs are absent and therefore cannot consume preparation, transfer, or device capacity. |
| 2026-08-30 | Make resident device state the target architecture, not repeated host snapshots | The maintainer proposed transfer of only Phase 2-relevant hash memory, only statement rows surviving filtering, and zero transfer for an unchanged LB. The selected design keeps generation-tracked LB columns resident, appends or replaces only changed ranges, and has GPU filtering feed compact surviving statement indices directly into growth/evaluation. |
| 2026-08-30 | Put the dominant evaluation state behind GPU survivor detection | Selected hash-memory projection removed only about 0.01 percent of IncubatorPeano1's aggregate entries and bytes, while the universal encoded and remaining-argument tables still dominate reconstruction. The first device stage therefore receives only filtering/growth columns, returns the compact set of LBs with unique surviving requests, and only those LBs receive evaluation-only state. Fixed descriptor ordinals remain present as empty views in the first stage. |

## Failure ledger

| Date | Failure | Resolution or next action |
|---|---|---|
| 2026-08-30 | The first context summary recalled only the initial 4.851-times CUDA port result | Maintainer corrected the baseline: `gpu_rt1` and `gpu_rt2` already reduce shortcut Phase 2 by about ten times; this ledger now uses the current-main compounded result |
| 2026-08-30 | Normal rebuild could not locate CUDA, then the explicit CUDA build hit a duplicate case-insensitive `Path`, then link missed ignored `mimalloc.dll.lib` | Use the clean child environment from `gpu_rt1`; copy the ignored import library from the prior GPU worktree; full rebuild succeeds |
| 2026-08-30 | Full-run CUDA first asserted at 512 logical blocks, then at metadata and blob projection ceilings | Added deterministic streaming and replaced the temporary block-count limit with exact greedy packing across every projection and task column |
| 2026-08-30 | IncubatorPeano1 full state exceeded shortcut evaluator marker and candidate-owner ceilings | Sized evaluator work columns at their existing request/reverse-owner scale; the complete IncubatorPeano1 prover and its CUDA CE filter then passed |
| 2026-08-30 | IncubatorPeano2 later exceeded shortcut growth prefix-payload capacity | Increase the growth prefix/candidate-window ownership for the next full pipeline run; no processor fallback or truncation |
| 2026-08-30 | The first progress entry mislabeled IncubatorPeano2's 9.09502-second Phase 2 total as IncubatorPeano1 | Re-read the live batch boundaries: IncubatorPeano1 is about 77 seconds; IncubatorPeano2 is about 9 seconds; corrected the evidence and progress ledgers immediately |
| 2026-08-30 | Gauss main iteration 71 exceeded the inherited 1,048,576-node growth frontier inside one indivisible split-LB family | Preserve the family and increase its fixed frontier/accepted-event ownership to 4,194,304; no retry, truncation, or processor fallback |
| 2026-08-30 | The widened Gauss frontier then emitted more accepted events than the downstream 2,097,152-event ordering column owned | Align the fixed ordering event column with growth at 4,194,304; live GPU telemetry still showed about 2.8 GB free |
| 2026-08-30 | The first capacity-clean timing run overlapped unrelated machine use | Retain its successful seven-batch, verifier, and capacity evidence, but reject all runtime values for acceptance. Repeat the identical benchmark on an otherwise idle machine after the architecture is complete. |
| 2026-08-30 | The first key-counter rebuild used the typed façade as though it were the underlying cold map | Reach `BytesKeyStore` through `TypedCold::inner`; the next full rebuild succeeded and all 1,560 tests passed. |
| 2026-08-30 | The first selected-map pipeline asserted in `findProjectedByteMapEntry` because omitted views shifted fixed enum ordinals | Preserve all ten descriptor positions and represent each unselected whole-key/subkey pair as a zero-length view. Entries, slots, keys, and blobs remain omitted, while constant-time device addressing stays unchanged. |

## Commit ledger

| Milestone | Commit | Verification | Push |
|---|---|---|---|
| Branch and persistent plan | | `git diff --check`; documentation-only complete snapshot ||

## Progress log

### 2026-08-30 — campaign opened from current main tip

Created the dedicated branch and worktree, which already contains
the landed initial CUDA port, `gpu_rt1`, and accepted `gpu_rt2` candidate-window
route. The current checkout outside this worktree carries an unrelated
uncommitted `prover.hpp` change and is intentionally untouched.

Prior campaign evidence establishes the inherited approximately ten-times
shortcut Phase 2 acceleration and exact 9,994-check/311-artifact boundary. The
next action is a source-grounded audit of configuration ownership, every SSD
deload door, full-run Phase 2 selection, and the private counterexample-filter
clone lifecycle before production code changes.

### 2026-08-30 — first full-run CUDA and CE execution

Added batch-owned `use_gpu` and `allow_ssd_deload` parameters, made configuration
selection the default full-run route, retained the explicit command-line backend
as a test override, and made the steward resident-only when SSD deload is off.
The owning constructor asserts the CUDA/SSD invalid combination. A clean CUDA
13.3 Release rebuild succeeds and all 1,560 native tests pass.

The shortcut projection cannot own a full incubator sweep in one image:
IncubatorPeano1 begins with 1,767 active logical blocks. CUDA now keeps each
split logical block's task family intact and streams greedily packed chunks
through persistent buffers. Packing is determined by exact post-Phase-1 usage
of every projection and task column. IncubatorPeano1 completed all 50 prover
iterations with 77.1871 seconds cumulative Phase 2; IncubatorPeano2 completed
its 50 iterations with 9.63342 seconds.

The CE filter now builds sixteen private clones at a time and runs their one
hashburst through the CUDA `counterExample` task kind. The first complete live
Peano filter retained 215 of 1,001 conjectures, then 171 of 215, with 3.827435
seconds CUDA Phase 2 and 8.75854 seconds across both complete filters. This
establishes the direct GPU semantics but is not the performance endpoint: each
clone still repeats the immutable fact-base projection and transfer. The next CE
architecture milestone aliases or retains that common projection and stages only
the conjecture-specific rule state.

### 2026-08-30 — full-run attribution and incubator split authorization

The first capacity-closed run through every incubator proved that the modest
runtime improvement is outside the CUDA kernels. IncubatorPeano1 accumulated
82.914 seconds in Phase 2, but only 19.092 seconds was device work. Rebuilding
and scheduling 1,313 capacity chunks consumed 46.301 seconds, including 32.865
seconds of projection; sealing consumed 5.066 seconds and pre-chunk capacity
planning accounts for most of the remaining time. The ten-times target therefore
requires a resident projection with deterministic deltas and fewer routes.

The maintainer explicitly allowed LB splitting in incubator batches. All five
incubator configurations now opt in. The existing integer `isStraggler` policy
still leaves balanced small LBs at one part, so the config change makes splitting
available without forcing it. The same run reached Gauss main iteration 70 and
then proved that one 32-part LB family needs more than the inherited 1,048,576
growth-frontier records; fixed ownership is increased to 4,194,304 before the
next full rebuild and run.

The split-enabled A/B completed all five incubators with unchanged proven-
theorem counts but consistently worse time: IncubatorPeano1 Phase 2 rose from
82.914 to 97.819 seconds and overall time from 134.548 to 157.652 seconds;
IncubatorPeano2 overall rose from 23.283 to 25.678 seconds; IncubatorGauss1/2/3
overall rose from 103.268/30.482/91.176 to 117.238/33.595/96.938 seconds. The
permission remains encoded, but processor-style 32-part execution is not counted
as a GPU optimization. The resident/delta route must avoid repeated family
projection and filter work.

### 2026-08-30 — full-run capacity closure and zero-work boundary

The expanded fixed frontier completed every incubator, Peano, Gauss, both CUDA
counterexample-filter passes, and the full verifier. The verifier performed
139,241 checks with zero failures. Because the maintainer reported concurrent
machine use, the 666.65492-second overall time and every component timing in
`run_gpu_main_packing_order1.log` are explicitly impure and cannot support a
speed claim.

The successful trace exposed a more direct source of the absent runtime win:
CUDA projected and launched LBs even when every normal-mode request family was
structurally incapable of producing a candidate. The task projection now counts
only executable families, the combined new-this-burst family exists only when
one of its mandatory terms can be satisfied, and a whole zero-work LB bypasses
projection and device execution while retaining a zero submatch count and empty
sealed output. A full rebuild and all 1,560 native tests pass. The complete
zero-work semantic run then reproduced all 139,241 verifier checks with zero
failures across the same seven batches. It lowered an impure overall run from
666.65492 to 607.13925 seconds, but the run began while the machine was occupied
and remains excluded from performance acceptance.

The capacity planner also used an element walk to count every projected cold
key even though each key store already owns the mutations that determine the
exact logical byte total. `BytesKeyStore` now maintains that total through
append, compaction, clear, copy, and reload, so the projection census is
constant-time for name, rule, key, and blob byte columns. A clean rebuild and
all 1,560 unit tests pass. The first full run that began after the maintainer
released the whole machine is `run_gpu_main_pure_key_counter1.log`; it is the
current admissible performance floor, not yet the target result. It completed
all seven batches and 139,241 verifier checks with zero failures in 614.21430
seconds. Its clean per-batch Phase 2 totals are 83.8572, 9.8739, 11.9563,
12.8856, 3.80907, 6.28557, and 14.5407 seconds. The clean GPU counterexample
filter brackets are 10.3212 seconds for Peano and 7.92093 seconds for Gauss.

IncubatorPeano1 attribution is decisive: only 19.769432 seconds are measured
device work, while host projection consumes 43.418392 seconds of wall time and
144.067743 seconds of aggregate worker processor time. Name reconstruction,
byte-map reconstruction, rule strings, reverse maps, and shard merge account
for the bulk of that processor work. Exact immutable name/rule sharing passes a
full rebuild, 1,561 tests, and a complete exact run. Overall runtime fell to
580.84874 seconds, while the seven Phase 2 totals fell to 66.350704, 7.874544,
9.109199, 11.446142, 3.113738, 4.023321, and 12.950826 seconds. This is only
1.12-1.56 times faster than the clean pre-sharing route and remains slower than
the processor on the largest incubator. IncubatorPeano1 still spends 42.189262
seconds in preparation, 140.303831 aggregate processor-seconds rebuilding
projection, and 18.723964 seconds on device work. Exact sharing is therefore
retained as a correctness-neutral physical reduction but rejected as the
primary acceleration mechanism. The next implementation narrows each LB to the
hash-memory pairs selected by its executable batches; the final architecture
retains those columns and encoded-statement state on device across iterations,
updates only mutations, and lets the device filter expose only surviving rows
to downstream kernels.

### 2026-08-30 — selected hash-memory control closes the wrong seam

The executable-family mask now projects only the working, overall/new, local,
and local-delta hash-memory pairs actually named by each LB's surviving request
batches. It preserves the fixed ten-view descriptor table with empty views for
unselected pairs. A first implementation omitted the descriptors themselves and
asserted when device code addressed a shifted enum ordinal; the fixed-table form
passes the full rebuild and all 1,561 native tests.

The clean full run is exact: all seven batches completed, Peano and Gauss CE
retained the same 215/171 and 80 conjectures, and the verifier performed 139,241
checks with zero failures. Overall runtime was 607.36436 seconds. Per-batch Phase
2 totals were 80.586939, 9.638415, 8.755005, 10.831017, 2.805991, 3.592373, and
13.234814 seconds. IncubatorPeano1 lost only 10,896 of 130,188,316 aggregate
entries and 1,303,168 of 17,657,766,596 key bytes, so the maps excluded by the
mask were almost empty. The experiment is retained because it is exact and
prevents irrelevant future state from entering a task, but it cannot provide
the runtime target.

The next split is now evidence-selected: GPU filtering, growth, and ordering run
from their narrow columns first; only LBs with unique surviving requests receive
the dominant overall encoded/remaining-argument maps, reverse maps, rule strings,
and evaluation-only tables. This creates the exact survivor boundary needed
before unchanged columns can be retained by generation on device.

### 2026-08-31 — Phase 1 and Phase 3 barrier attribution

The 607.36436-second log did not record Phase 1 and Phase 3 independently. A
diagnostic branch based on its exact tagged source tree adds wall clocks around
the existing Phase 1 and Phase 3 worker barriers without changing scheduling,
backend selection, or proof semantics. The fresh idle-machine run completed in
574.64098 seconds with the same 139,241 checks and zero failures. Therefore the
table is a clean remeasurement of the tagged code, not a reconstruction of the
older 607.36436-second wall-clock sample. The remainder column is derived as
the complete prover or filter wall minus the three measured phase barriers.

| Prover or CE pass | Complete wall (seconds) | Phase 1 | Phase 2 | Phase 3 | Derived non-phase remainder |
|---|---:|---:|---:|---:|---:|
| IncubatorPeano1 prover | 120.527000 | 30.142457 | 69.866652 | 4.034604 | 16.483287 |
| IncubatorPeano2 prover | 19.990500 | 9.712055 | 8.159811 | 0.303676 | 1.814959 |
| Peano CE 1,001 to 215 | 6.851120 | 0.880464 | 2.820904 | 0.544535 | 2.605217 |
| Peano CE 215 to 171 | 1.965070 | 0.264266 | 0.820501 | 0.267603 | 0.612701 |
| Peano prover | 27.076300 | 4.754822 | 6.005454 | 9.989441 | 6.326584 |
| IncubatorGauss1 prover | 75.390100 | 58.234775 | 11.970231 | 2.246499 | 2.938595 |
| IncubatorGauss2 prover | 22.282200 | 17.663056 | 3.615679 | 0.473215 | 0.530251 |
| IncubatorGauss3 prover | 93.640800 | 27.939084 | 5.882937 | 58.024561 | 1.794218 |
| Gauss CE 674 to 80 | 7.587530 | 1.956473 | 4.064223 | 0.199146 | 1.367689 |
| Gauss prover | 60.623000 | 2.955939 | 10.156377 | 44.722730 | 2.787954 |
| **Seven provers total** | **419.529900** | **151.402187** | **115.657140** | **119.794726** | **32.675847** |

Phase 2 is no longer the aggregate prover leader: Phase 1 represents 36.09
percent of prover wall, Phase 3 28.55 percent, and Phase 2 27.57 percent.
However, the next bottleneck depends on the batch. IncubatorGauss1 and
IncubatorGauss2 are Phase-1-dominated; Peano, IncubatorGauss3, and Gauss are
Phase-3-dominated; IncubatorPeano1 is still Phase-2-dominated. Every non-empty
CE pass is also still Phase-2-dominated. Peano and Gauss additionally execute
small post-prover Phase 1/2/3 work of 0.047045/0.491030/0.097943 seconds and
0.034465/0.194709/0.093729 seconds respectively.

### 2026-08-31 — internal Phase 1 and Phase 3 attribution

The next diagnostic keeps the complete barrier clocks and adds 27 internal
categories. Each worker writes elapsed nanoseconds only into its private row;
the rows are folded after the join, so the measurement adds no shared timing
counter or lock to the phase. Category totals are **summed worker seconds** and
may exceed barrier wall because workers overlap. The effective parallelism is
summed worker seconds divided by barrier wall. For the category tables below,
the derived barrier contribution is the measured barrier wall multiplied by a
category's share of summed worker seconds. It is a proportional model, not a
second wall-clock measurement. Every category at or above 0.5 percent appears;
the rest are combined.

The full CUDA 13.3 rebuild and all 1,561 native tests passed. The idle-machine
run retained the exact 215/171 Peano and 80 Gauss counterexample survivors and
completed 122,994 incubator plus 16,247 main verifier checks, 139,241 total,
with zero failures. Overall runtime was 576.10843 seconds, 1.46745 seconds or
0.255 percent above the preceding 574.64098-second run. This difference also
contains normal run-to-run variation, but rules out a large measurement-regime
distortion.

| Prover or CE pass | Phase 1 barrier | Phase 1 worker seconds | Effective parallelism | Phase 3 barrier | Phase 3 worker seconds | Effective parallelism |
|---|---:|---:|---:|---:|---:|---:|
| IncubatorPeano1 prover | 29.427333 | 919.221131 | 31.24 | 3.980773 | 103.061509 | 25.89 |
| IncubatorPeano2 prover | 9.537721 | 291.219631 | 30.53 | 0.289890 | 3.334999 | 11.50 |
| Peano CE 1,001 to 215 | 0.898800 | 9.819927 | 10.93 | 0.598582 | 2.416524 | 4.04 |
| Peano CE 215 to 171 | 0.282088 | 3.120358 | 11.06 | 0.294855 | 1.004000 | 3.41 |
| Peano prover | 4.745575 | 96.400266 | 20.31 | 10.166981 | 99.372800 | 9.77 |
| IncubatorGauss1 prover | 57.487574 | 1,777.373397 | 30.92 | 2.231761 | 28.887267 | 12.94 |
| IncubatorGauss2 prover | 18.044743 | 503.760007 | 27.92 | 0.452563 | 1.198407 | 2.65 |
| IncubatorGauss3 prover | 27.772594 | 52.560191 | 1.89 | 57.261548 | 72.482610 | 1.27 |
| Gauss CE 674 to 80 | 1.960317 | 24.094718 | 12.29 | 0.204779 | 1.127853 | 5.51 |
| Gauss prover | 2.939941 | 32.873263 | 11.18 | 44.452680 | 96.021573 | 2.16 |

Across the seven prover calls, Phase 1 barrier wall is 149.955481 seconds and
Phase 3 barrier wall is 118.836196 seconds. CE passes remain separate above.

#### Phase 1 internal split

Each entry is `summed worker seconds / worker share / derived barrier seconds`.

| Prover or CE pass | Phase 1 categories |
|---|---|
| IncubatorPeano1 prover | External statement absorption `839.691 / 91.35% / 26.881`; external origin absorption `33.372 / 3.63% / 1.068`; contradiction discharge `28.618 / 3.11% / 0.916`; routing-mail pull `6.928 / 0.75% / 0.222`; other `10.612 / 1.15% / 0.340` |
| IncubatorPeano2 prover | External statement absorption `280.291 / 96.25% / 9.180`; external origin absorption `5.534 / 1.90% / 0.181`; contradiction discharge `2.200 / 0.76% / 0.072`; routing-mail pull `1.463 / 0.50% / 0.048`; other `1.732 / 0.59% / 0.057` |
| Peano CE 1,001 to 215 | Goal discharge `6.337 / 64.54% / 0.580`; mail-out fill `2.128 / 21.67% / 0.195`; contradiction discharge `1.151 / 11.72% / 0.105`; contradiction-scope discharge `0.196 / 2.00% / 0.018`; other `0.007 / 0.07% / 0.001` |
| Peano CE 215 to 171 | Goal discharge `2.055 / 65.85% / 0.186`; mail-out fill `0.663 / 21.24% / 0.060`; contradiction discharge `0.328 / 10.52% / 0.030`; contradiction-scope discharge `0.073 / 2.33% / 0.007`; other `0.002 / 0.05% / 0.000` |
| Peano prover | External statement absorption `67.293 / 69.81% / 3.313`; external origin absorption `12.394 / 12.86% / 0.610`; equivalence application `9.911 / 10.28% / 0.488`; routing-mail pull `2.191 / 2.27% / 0.108`; contradiction discharge `1.447 / 1.50% / 0.071`; expression cleanup `1.374 / 1.42% / 0.068`; mail-out fill `1.093 / 1.13% / 0.054`; other `0.698 / 0.72% / 0.034` |
| IncubatorGauss1 prover | External statement absorption `1,764.656 / 99.28% / 57.076`; every other category combined `12.717 / 0.72% / 0.411` |
| IncubatorGauss2 prover | External statement absorption `502.063 / 99.66% / 17.984`; every other category combined `1.697 / 0.34% / 0.061` |
| IncubatorGauss3 prover | External statement absorption `47.500 / 90.37% / 25.099`; burst setup `1.916 / 3.65% / 1.013`; external origin absorption `1.893 / 3.60% / 1.000`; equivalence application `0.682 / 1.30% / 0.360`; other `0.569 / 1.08% / 0.301` |
| Gauss CE 674 to 80 | Goal discharge `15.610 / 64.79% / 1.270`; mail-out fill `5.466 / 22.69% / 0.445`; contradiction discharge `2.465 / 10.23% / 0.201`; contradiction-scope discharge `0.549 / 2.28% / 0.045`; other `0.005 / 0.02% / 0.000` |
| Gauss prover | External statement absorption `18.032 / 54.85% / 1.613`; equivalence application `7.862 / 23.91% / 0.703`; external origin absorption `3.254 / 9.90% / 0.291`; contradiction discharge `0.948 / 2.88% / 0.085`; internal statement absorption `0.776 / 2.36% / 0.069`; mail-out fill `0.657 / 2.00% / 0.059`; routing-mail pull `0.575 / 1.75% / 0.051`; expression cleanup `0.346 / 1.05% / 0.031`; burst setup `0.238 / 0.72% / 0.021`; other `0.186 / 0.57% / 0.017` |

#### Phase 3 internal split

| Prover or CE pass | Phase 3 categories (`worker seconds / share / derived barrier seconds`) |
|---|---|
| IncubatorPeano1 prover | Internal statement absorption `50.643 / 49.14% / 1.956`; contradiction discharge `26.436 / 25.65% / 1.021`; internal origin absorption `11.853 / 11.50% / 0.458`; equivalence application `8.468 / 8.22% / 0.327`; mail-out fill `2.257 / 2.19% / 0.087`; goal discharge `1.266 / 1.23% / 0.049`; ancestor-row sweep `1.098 / 1.06% / 0.042`; other `1.042 / 1.01% / 0.040` |
| IncubatorPeano2 prover | Contradiction discharge `1.584 / 47.51% / 0.138`; internal statement absorption `0.800 / 23.99% / 0.070`; equivalence application `0.506 / 15.17% / 0.044`; internal origin absorption `0.191 / 5.74% / 0.017`; ancestor-row sweep `0.056 / 1.69% / 0.005`; mail-out fill `0.043 / 1.29% / 0.004`; subtree wipe `0.042 / 1.25% / 0.004`; goal discharge `0.034 / 1.01% / 0.003`; expression cleanup `0.032 / 0.95% / 0.003`; internal-mail clear `0.031 / 0.92% / 0.003`; other `0.016 / 0.48% / 0.001` |
| Peano CE 1,001 to 215 | Equivalence application `1.338 / 55.37% / 0.331`; contradiction discharge `0.368 / 15.21% / 0.091`; expression cleanup `0.330 / 13.65% / 0.082`; goal discharge `0.127 / 5.27% / 0.032`; internal statement absorption `0.094 / 3.89% / 0.023`; mail-out fill `0.091 / 3.78% / 0.023`; ancestor-row sweep `0.034 / 1.42% / 0.008`; internal origin absorption `0.023 / 0.93% / 0.006`; other `0.012 / 0.48% / 0.003` |
| Peano CE 215 to 171 | Equivalence application `0.538 / 53.60% / 0.158`; contradiction discharge `0.178 / 17.69% / 0.052`; expression cleanup `0.175 / 17.41% / 0.051`; internal statement absorption `0.032 / 3.21% / 0.009`; mail-out fill `0.032 / 3.17% / 0.009`; goal discharge `0.031 / 3.13% / 0.009`; ancestor-row sweep `0.010 / 0.95% / 0.003`; internal origin absorption `0.006 / 0.62% / 0.002`; other `0.002 / 0.23% / 0.001` |
| Peano prover | Internal statement absorption `44.504 / 44.78% / 4.553`; equivalence application `39.378 / 39.63% / 4.029`; mail-out fill `7.658 / 7.71% / 0.784`; expression cleanup `5.103 / 5.13% / 0.522`; contradiction discharge `1.608 / 1.62% / 0.165`; internal origin absorption `0.696 / 0.70% / 0.071`; other `0.426 / 0.43% / 0.044` |
| IncubatorGauss1 prover | Internal statement absorption `12.045 / 41.70% / 0.931`; equivalence application `11.039 / 38.21% / 0.853`; contradiction discharge `1.342 / 4.65% / 0.104`; subtree wipe `1.017 / 3.52% / 0.079`; mail-out fill `0.907 / 3.14% / 0.070`; internal origin absorption `0.869 / 3.01% / 0.067`; goal discharge `0.787 / 2.72% / 0.061`; expression cleanup `0.617 / 2.13% / 0.048`; other `0.263 / 0.91% / 0.020` |
| IncubatorGauss2 prover | Equivalence application `0.408 / 34.08% / 0.154`; internal statement absorption `0.303 / 25.27% / 0.114`; subtree wipe `0.249 / 20.76% / 0.094`; contradiction discharge `0.137 / 11.42% / 0.052`; internal origin absorption `0.027 / 2.24% / 0.010`; mail-out fill `0.021 / 1.73% / 0.008`; expression cleanup `0.019 / 1.60% / 0.007`; goal discharge `0.016 / 1.30% / 0.006`; disproved-goal drain `0.007 / 0.61% / 0.003`; ancestor-row sweep `0.006 / 0.50% / 0.002`; other `0.006 / 0.49% / 0.002` |
| IncubatorGauss3 prover | Internal statement absorption `39.542 / 54.55% / 31.238`; subtree wipe `13.383 / 18.46% / 10.573`; equivalence application `12.663 / 17.47% / 10.004`; expression cleanup `3.255 / 4.49% / 2.571`; quiescence and dump checks `2.886 / 3.98% / 2.280`; internal origin absorption `0.402 / 0.56% / 0.318`; other `0.352 / 0.49% / 0.278` |
| Gauss CE 674 to 80 | Contradiction discharge `0.731 / 64.77% / 0.133`; internal statement absorption `0.245 / 21.68% / 0.044`; ancestor-row sweep `0.101 / 8.94% / 0.018`; internal origin absorption `0.015 / 1.31% / 0.003`; mail-out fill `0.014 / 1.27% / 0.003`; goal discharge `0.014 / 1.23% / 0.003`; other `0.009 / 0.79% / 0.002` |
| Gauss prover | Internal statement absorption `51.329 / 53.46% / 23.762`; equivalence application `26.122 / 27.20% / 12.093`; subtree wipe `8.003 / 8.33% / 3.705`; expression cleanup `5.784 / 6.02% / 2.678`; mail-out fill `2.899 / 3.02% / 1.342`; contradiction discharge `1.082 / 1.13% / 0.501`; other `0.801 / 0.83% / 0.371` |

The optimization seams are now unambiguous. Working-set claim/load time is
negligible in both phases. Phase 1 is the routing external-statement absorption
door in every prover; IncubatorGauss1/2 use nearly all 32 workers, whereas
IncubatorGauss3 reaches only 1.89 effective workers and is a serial-straggler
case. The long Phase 3 barriers are also low-parallelism stragglers:
IncubatorGauss3 reaches 1.27 effective workers and Gauss reaches 2.16. Their
dominant chain is internal statement absorption, equivalence application, and
subtree wiping. Counterexample Phase 1 is different: it is goal discharge plus
mail-out fill; its Phase 3 is small and dominated by equivalence/cleanup for
Peano and contradiction discharge for Gauss.

## Main-merge acceptance (2026-09-01)

The branch merged current main (the canonical deposit door and its capacity
census refresh), replaced every direct numeric capacity assignment at the
execution site with the named audited profiles, separated the physical task
ceiling from the projected-block ceiling, and pinned Incubator Peano 1 to the
processor route by config. All four acceptance runs against fresh same-tree
main references passed exactly:

| Run | Verifier | Theorems | Artifacts vs main reference |
|---|---|---|---|
| shortcut processor | 9,207 / 0 | 39 proven, 64 rows | 310/311 byte-identical |
| shortcut CUDA | 9,207 / 0 | 39 proven, 64 rows | 310/311 byte-identical |
| full processor | 139,296 / 0 | 518/79/76/105/25/4/19 | 5,048/5,049 byte-identical |
| full mixed CUDA | 139,296 / 0 | 518/79/76/105/25/4/19 | 5,048/5,049 byte-identical |

The single non-identical file in every comparison is
`externally_provided_theorems.txt` — a tracked durable input whose working
copies differ only by checkout line-ending normalization; every produced
artifact is byte-identical. Observation-only telemetry (`[GPU-*]`,
`[PHASE2-TIMING]`, `[PHASE13-*]`, `[DELOAD]`, `[SPLIT]`) now writes to the
common per-run diagnostics log  with batch and
hashburst markers, keeping the main run log proof-narrative only.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
