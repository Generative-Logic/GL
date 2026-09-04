<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Phase 2 GPU runtime 1 plan

> **Purpose:** this is the persistent working memory for the second Phase 2 GPU
> performance campaign. Read it after every context compaction. Update it after
> every measurement, design conclusion, failed experiment, commit, and change of
> next action. Conversation memory is not authoritative.

## Current state

- **Branch:** 
- **Worktree:** 
- **Base:** `main`
- **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU, native `sm_89`
- **Status:** campaign complete; dependency-staged projection upload is exact and
 repeatably clears the strict gate at 9.74343 and 9.46378 seconds
- **Active milestone:** M4 complete
- **Next action:** none inside this campaign; preserve the exact branch as the
 starting point for any later dirty-generation or deeper GPU restructuring work

## Objective and acceptance

Accelerate the complete FTA shortcut Phase 2 CUDA bracket by at least two times
on the current RTX 4070 Laptop GPU, without weakening exact processor/CUDA
semantics, provenance, ordering, doom selection, fixed ownership, or failure
contracts.

The landed same-tip reference is 19.835229 seconds complete CUDA Phase 2, with 15.034592 seconds inside the exact upload-plus-device event
interval. Its strict two-times target is **9.917614 seconds or less**. A fresh
 source-tree baseline is 20.1945325 seconds, whose two-times threshold
is 10.0972663 seconds. Final acceptance requires at least 2.000 times against
that fresh control and must still meet the stricter 9.917614-second historical
threshold. Prover and overall shortcut time are reported beside the Phase 2
number and may not regress relative to the fresh control.

Final acceptance also requires:

- a full clean Windows Release rebuild and every native unit test passing;
- processor and CUDA shortcut runs through `main.py`, never the executable;
- zero verifier failures on both routes;
- exact SHA-256 identity for every retained semantic artifact;
- native CUDA telemetry proving the optimized path executed with no fallback;
- fixed-capacity assertions, complete provenance, canonical order, and doom
 prefix semantics unchanged;
- every source milestone committed with `git add -A`, a detailed message, and an
 immediate push;
- relevant agent SwDD, user SwDD, and MPU claims audited before each
 hardware-relevant architecture commit.

## Evidence inherited from the landed port

The final landing pair on current-then-main measured 96.212168 processor seconds
and 19.835229 CUDA seconds for complete Phase 2, with 203.22474 and 127.55500
seconds overall. Both passed 9,994 checks with zero failures and produced 305
byte-identical semantic artifacts.

The strongest pre-merge attribution measured:

| Seam | Time or hardware fact |
|---|---:|
| Complete Phase 2 | 19.257039 seconds |
| Exact upload plus device interval | 14.331816 seconds |
| Host projection | 3.987276 seconds |
| All transfers | about 0.525 seconds |
| Frontier-growth kernels | 13.005 seconds, 87.0% of kernel time |
| Representative achieved occupancy | 54.35% |
| Representative compute throughput | 46.49% |
| Representative DRAM throughput | 1.96% |
| Growth kernel compiled registers | 63 per thread |
| Growth kernel compiled thread stack | 3,184 bytes |

`phase2GrowthExpandKernel` assigns one thread to one `DeviceGrowthNode`. That
thread builds the node prefix once, then serially scans every later filtered
candidate position. The retained FTA census recorded 4,077,562,830 expansion
attempts across the run and 358,565,795 attempts in one sweep. The work is
latency/occupancy constrained rather than device-memory-bandwidth constrained.

Projection upload currently copies the used prefixes of all 24 resident columns
in canonical order. Transfer is included in the exact device interval. It is not
the first bottleneck. The two possible CUDA passes in one shortcut subprocess
project disjoint non-straggler and straggler logical-block sets, so an in-process
unchanged-block cache has zero hits on the current target.

## Non-negotiable semantic contracts

1. The processor implementation remains the oracle and selectable peer.
2. Selected CUDA mode never falls back; unavailable hardware, capacity failure,
 allocation failure, launch failure, or identity failure asserts at its owner.
3. The GPU continues to perform request filtering, growth, ordering, evaluation,
 provenance, canonical ordering, and doom selection. The host only projects,
 downloads, seals, and deposits already-decided content.
4. Atomic append order and kernel completion order remain non-semantic. Existing
 deterministic order-token reconstruction selects the observable stream.
5. The compact 56-byte persistent `DeviceGrowthNode` remains the frontier record.
 A full normalized-key payload is not copied into every million-row frontier.
6. Production buffers remain process-owned, preallocated, fixed-capacity arrays.
 No hot-path allocation, live resize, truncation, or unified-memory fallback.
7. Identifier mint order, decoded-byte lexical order, request first occurrence,
 growth position, provenance dependency order, and doom-prefix inclusivity stay
 exact.
8. Every new production function receives full GL-style Doxygen documentation
 and a direct positive/negative unit test in the same commit.

## Approved restructuring direction

The original approved work grain is a global bulk frontier spanning logical
blocks, parts, request batches, stumps, and nodes. The second campaign changes
only how independent candidate positions inside one frontier node consume GPU
threads. Semantic inputs, outputs, capacities, order reconstruction, evaluation,
and host ownership boundaries remain unchanged.

### First implementation: adaptive cooperative growth

Keep the current one-thread-per-node path for short remaining candidate spans.
For a long span, the small-node kernel appends only that node's index to a bounded
large-node list and performs no candidate semantics. A second kernel assigns one
cooperative thread block to each listed node:

1. build the node's normalized-key and request-gate prefix once in shared state;
2. distribute later candidate positions across block lanes;
3. fold only the lane's appended statement into the shared prefix;
4. run the unchanged whole-key and owner-subkey gates independently;
5. append the same accepted event and child-node records to the existing bounded
 ledgers;
6. leave all observable ordering to the existing exact ordering stage.

The large-node list is bounded by the existing frontier ceiling. Its atomic
append order is not observable because entries name immutable source nodes and
the cooperative kernels append into ledgers that are already canonically
reconstructed. The threshold is performance policy only. It will be selected
from retained span telemetry and direct timing, not guessed into the final route.

The cooperative path must not copy the whole shared prefix into every lane's
local stack. Byte-map probes need a segmented prefix-plus-appended-expression
view or an equivalent shared-buffer reader whose hash and equality are directly
twinned against the existing contiguous key representation.

### Follow-on seams, only after the growth result

1. **Host-controlled frontier waves:** remove repeated counter downloads and
 synchronizations only if post-growth profiling makes them material.
2. **Projection parallelism:** distribute independent logical-block/name work or
 move rank construction to the device if host projection remains more than
 about 25% of the new Phase 2 bracket.
3. **Selective upload:** transfer only task-reachable packed tables or explicit
 dirty ranges when measurement shows material copy or projection avoidance.
 Current all-column transfer is about 0.5 seconds, so this cannot deliver the
 first two-times gain by itself.
4. **Persistent changed-memory residency:** retain the already-approved
 generation-backed contract for future modes with repeated logical blocks.
 Do not introduce cross-process persistence to manufacture cache hits for the
 current disjoint-pass shortcut.

## Milestones

### M0 — branch, plan, and baseline

- [x] Create in its own worktree from current `main`.
- [x] Read the complete container, string, and GPU cookbooks.
- [x] Read the completed first-port plan and final landing evidence.
- [x] Create this persistent plan.
- [x] Full Windows Release rebuild.
- [x] Run all 1,557 native unit tests.
- [x] Run and retain a fresh CUDA shortcut baseline.
- [x] Parse all Phase 2, route, projection, device, prover, and overall timings
 with exponent-safe numeric parsing.
- [x] Retain the complete 311-file shortcut tree and record its SHA-256 manifest.

### M1 — adaptive work evidence and direct twin

- [x] Add observation-only remaining-span telemetry or derive the complete
 distribution from existing retained data without entering proof flow.
- [x] Fix the first small/large threshold before production timing: 256 remaining
 candidates, with 128 and 512 retained as measured tuning candidates.
- [x] Add the bounded large-node list and exact capacity/high-water telemetry.
- [x] Add a direct CUDA twin covering short-node, long-node, rejection, owner
 rejection, whole-only emission, child growth, repeated variables, and maximum
 live depth.
- [x] Full rebuild and unit gate for the census seam.
- [x] Commit and push the coherent census milestone.

### M2 — cooperative growth production route

- [x] Implement shared prefix state and candidate-lane distribution.
- [x] Prove segmented key hash/equality byte-identical to the contiguous oracle.
- [x] Route only long nodes cooperatively; preserve the small-node kernel.
- [x] Run the complete CUDA shortcut and verifier.
- [x] Compare the semantic artifact manifest to M0 exactly.
- [x] Record complete Phase 2 and exact device improvement.
- [x] Profile the accepted compiler resource shape before choosing the next seam.
- [x] Update affected design documentation; commit and push.

### M3 — close the remaining measured seam

- [ ] If growth remains dominant, tune block size/threshold or eliminate the next
 measured growth serialization while retaining the direct twin.
- [x] If projection becomes dominant, parallelize the measured projection
 component before adding residency machinery.
- [ ] If synchronization becomes dominant, restructure host-controlled frontier
 waves without changing frontier/event semantics.
- [ ] Add selective task-reachable or dirty-range upload only if measured bytes
 and time can contribute materially to the strict target.
- [x] Dependency-stage the nine evaluation-only column uploads across independent
 host schedule construction while retaining all 24 exact device prefixes.
- [ ] Commit and push each independently complete, exact milestone.

### M4 — final two-times gate

- [x] Full clean Release rebuild and all native tests.
- [x] Fresh processor and CUDA shortcut runs through `main.py`.
- [x] Zero verifier failures and exact processor/CUDA artifact SHA-256 identity.
- [x] Complete CUDA Phase 2 at or below 9.917614 seconds and at least 2.000 times
 faster than the fresh M0 CUDA control.
- [x] No regression in prover or overall shortcut time against M0.
- [x] Final profile, transfer bytes, fixed ownership, queue peaks, occupancy,
 compute/DRAM throughput, branch/commit, and logs recorded here.
- [x] Agent SwDD, user SwDD, and MPU audit complete.
- [x] Final source/documentation commit pushed; branch clean.

## Verification commands

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe' GL_Quick_VS\GL_Quick.sln -p:Configuration=Release -p:Platform=x64 -t:Rebuild -verbosity:minimal
& 'GL_Quick_VS\GL_Quick\gl_quick.exe' --unit-tests
& 'C:\Users\nikol\anaconda3\python.exe' main.py --shortcut --phase2-backend cpu --run-descriptor gpu_rt1_final_cpu
& 'C:\Users\nikol\anaconda3\python.exe' main.py --shortcut --phase2-backend cuda --run-descriptor gpu_rt1_final_cuda
```

Every run redirects both streams to its own  file.
The baseline and every accepted source milestone retain their own artifact set or
manifest. Phase 2 is the sum of `[PHASE2-TIMING] iteration_seconds`; exact device
time is the sum of `[GPU-PHASE2] device_seconds`. Numeric extraction accepts
decimal and exponent forms.

## Evidence ledger

| Date | Commit or worktree | Evidence | Result |
|---|---|---|---|
| 2026-08-29 |, | `git worktree add -b sandbox/gpu_rt1 .worktree\gpu_rt1 main` | Dedicated clean worktree created from current `main` |
| 2026-08-29 | | Complete reads of `09b_statification_cookbook.md`, `09c_string_statification_cookbook.md`, `09d_gpu_cookbook.md`, and `docs/GPU/gpu_plan.md` | Container, string lifetime, GPU exactness, landed architecture, bottleneck, and final timing contracts loaded before source work |
| 2026-08-29 | | Landed same-tip shortcut evidence from `gpu_plan.md` | 19.835229-second CUDA Phase 2, 15.034592-second exact device interval, 127.55500-second overall, 9,994 checks with zero failures, 305 exact semantic artifacts |
| 2026-08-29 |, source tree | Full Release rebuild  and  | Native `sm_89`/`sm_120` link succeeded; 1,557/1,557 native tests passed |
| 2026-08-29 |,  | Fresh CUDA `main.py --shortcut --phase2-backend cuda --run-descriptor gpu_rt1_baseline_cuda` | 45 Phase 2 rows and 48 CUDA passes; Phase 2 20.1945325 seconds, exact device 15.2917872, route 19.4982658, preparation 3.9629086, projection 3.8889809, scheduling 0.0739263, finalization 0.2297749, prover 125.212, overall 130.12574; 9,994 checks, zero failures |
| 2026-08-29 | ,  | Complete retained shortcut output and SHA-256 manifest | 311 files, 11,541,549 bytes, 311 manifest rows; working tree remained clean |
| 2026-08-29 |; ,  | Full Release rebuild and direct CUDA span-census twin | Native link succeeded; 1,557/1,557 tests passed; the fixture reports exactly four nodes/ten attempts in the 2-to-3 bucket and two nodes/nine attempts in the 4-to-7 bucket; fixed growth ownership is 263,225,552 bytes |
| 2026-08-29 |;  | Complete CUDA shortcut census through `main.py` | 18,940,682 live frontier nodes own 4,498,333,440 candidate attempts. Spans of at least 128 candidates own 89.7395% of attempts across 61.9906% of nodes; at least 256 own 68.7134% across 35.0923%; at least 512 own 31.3414% across 10.2520%. Exact device time is 15.1043092 seconds; route is 20.5408475 seconds including the observation kernel; prover is 128.426 and overall is 133.46301 seconds; 9,994 checks, zero failures |
| 2026-08-29 |  versus current `files/shortcut` | SHA-256 comparison after complete census run | All 311 paths, lengths, and bytes are identical; zero differences |
| 2026-08-29 |; ,  | Full Release rebuild and mixed short/cooperative CUDA twin | Native link succeeded; 1,557/1,557 tests passed. The cutoff-four fixture executes both routes, and the existing accepted, owner-rejected, absent, whole-only, child-growth, repeated-variable, and live-depth checks pass |
| 2026-08-29 |  | Complete cutoff-256 CUDA shortcut through `main.py` | Phase 2 15.1108074 seconds, exact device 10.1747262, route 14.4156141, preparation 3.9950932, projection 3.9191152, scheduling 0.0759778, finalization 0.2296915, prover 119.196, overall 124.74793; cooperative peak 301,811 nodes; 9,994 checks, zero failures. Versus M0: Phase 2 1.3364 times faster and device interval 1.5024 times faster |
| 2026-08-29 |  versus cutoff-256 `files/shortcut` | Complete semantic SHA-256 comparison | All 311 paths, lengths, and bytes are identical; zero differences |
| 2026-08-29 |  | Native `cuobjdump --dump-resource-usage` | `sm_89` cooperative kernel: 53 registers, 1,280-byte lane stack, 2,016-byte shared block state; short-node kernel: 62 registers, 3,104-byte stack |
| 2026-08-29 | cutoff-64/64-lane worktree; ,  | Full Release rebuild and direct CUDA gates | Native link succeeded; 1,557/1,557 tests passed |
| 2026-08-29 |  | Complete CUDA shortcut through `main.py` | Phase 2 12.4757739 seconds, exact device 7.5260573, route 11.7834596, projection 3.9306738, prover 116.023, overall 120.89349; cooperative peak 552,505; 9,994 checks, zero failures; all 311 artifact hashes identical to M0. Phase 2 is 1.6187 times faster than M0 |
| 2026-08-29 | ,  | Complete process-tree CUDA attribution | Cooperative growth 3.732334 seconds, short-node growth 0.985799, filter count 1.171789, filter emit 0.566028, seed 0.092825, host-to-device transfers 0.364500, device-to-host 0.011897 seconds |
| 2026-08-29 | ,  | Full metric profile of one 311,218-node cooperative launch | 65.57% achieved occupancy, 46.72% compute, 2.11% DRAM, 99.90% L2 hit rate, 10.83 active threads per warp; remaining kernel is branch/latency constrained, not bandwidth constrained |
| 2026-08-29 | ,  | Four-shard Release rebuild and direct all-column merge twin | Native link succeeded; 1,558/1,558 tests passed. The twin rebases non-empty name, rule, byte/blob, reverse-owner, plain-data/run, mandatory-key, and metadata columns and matches the original serial 24-column checksum exactly |
| 2026-08-29 | ,  | Complete four-shard CUDA shortcut and identity gate | Phase 2 10.6506 seconds, projection 1.6821862, construction 4.2007232 processor-seconds, merge 0.3905575, exact device 7.9531376, prover 115.235, overall 120.14793; 9,994 checks, zero failures; all 311 paths, lengths, and hashes identical to M0. Fresh-baseline speedup is 1.896 times; 0.733 seconds remain to the strict gate |
| 2026-08-29 | ,  | Eight-shard Release rebuild and direct gates | Native link succeeded; 1,558/1,558 tests passed |
| 2026-08-29 | ,  | Complete eight-shard CUDA shortcut and identity gate | Phase 2 9.9825 seconds, projection 1.3315546, construction 4.5878480 processor-seconds, merge 0.4056626, exact device 7.6424692, prover 113.739, overall 118.71161; 9,994 checks, zero failures; all 311 paths, lengths, and hashes identical to M0. Fresh-baseline speedup is 2.023 times, but 0.0649 seconds remain to the stricter historical gate |
| 2026-08-29 | ,  | Twelve-shard Release rebuild and direct gates | Native link succeeded; 1,558/1,558 tests passed |
| 2026-08-29 | ,  | Complete twelve-shard CUDA shortcut and identity gate | Phase 2 10.1734 seconds, projection 1.3033454, construction 4.8731522 processor-seconds, merge 0.4433709, exact device 7.8794975, prover 113.482, overall 118.39938; 9,994 checks, zero failures; all 311 paths, lengths, and hashes identical to M0. The 0.0282-second projection gain did not offset merge and run variation; shard scaling is saturated |
| 2026-08-29 | ,  | Fixed pinned-staging Release rebuild and direct staged all-column checksum twin | Native link succeeded; 1,558/1,558 tests passed; ordinary and begin/finish upload doors produce the exact same device checksum across all 24 non-empty semantic columns |
| 2026-08-29 |  | Complete dependency-staged CUDA shortcut and identity gate | Phase 2 9.74343 seconds, projection 1.3181980, construction 4.9423075 processor-seconds, merge 0.4197105, staged preparation 1.8580455, task-upload/join/kernels 6.9558142, route 9.0659093, prover 113.791, overall 118.81494; 9,994 checks, zero failures; all 311 paths, lengths, and hashes identical to M0. Strict-reference speedup is 2.036 times and fresh-control speedup is 2.073 times |
| 2026-08-29 |  | Independent complete CUDA confirmation and identity gate | Phase 2 9.46378 seconds, projection 1.3362983, merge 0.4339587, staged preparation 1.8428385, task-upload/join/kernels 6.6995169, route 8.7941203, prover 113.587, overall 118.60189; 9,994 checks, zero failures; all 311 paths, lengths, and hashes identical to M0. Strict-reference speedup is 2.096 times and fresh-control speedup is 2.134 times |
| 2026-08-29 |  | Fresh processor shortcut and identity gate | Phase 2 100.043 seconds, prover 205.151, overall 209.94481; 9,994 checks, zero failures; all 311 paths, lengths, and hashes identical to M0. Accepted CUDA runs are 10.27 and 10.57 times faster in complete Phase 2 |

## Failure ledger

| Date | Failure | Resolution or next action |
|---|---|---|
| 2026-08-29 | First rebuild lacked `CudaToolkitDir` in the long-lived process environment | Passed the installed CUDA 13.3 directory explicitly; CUDA compilation started |
| 2026-08-29 | Host compilation stopped at known duplicate inherited `PATH`/`Path` key `MSB6001` | Launched MSBuild with one canonical child `Path` and the installed CUDA 13.3 variables |
| 2026-08-29 | The ignored vcpkg dependency copy carried headers but omitted the `lib` directory, causing `LNK1181` for `mimalloc.dll.lib` | Copied the already-installed local library to the worktree's expected fixed path; the following full rebuild linked cleanly |
| 2026-08-29 | First projection-shard rebuild captured the `ExpressionAnalyzer::steward` member as if it were an enclosing local (`C3480`) | Accessed the member through the captured analyzer instance; the following full rebuild and all 1,558 tests passed |
| 2026-08-29 | Registering the live projection vectors and overlapping their evaluation-only copies with semantic kernels remained exact but regressed complete Phase 2 to 10.8707 seconds | Kept projection memory pageable, copied only the nine deferred prefixes into fixed pinned staging, started transfer before host schedule construction, and joined before the first kernel; consecutive exact runs reached 9.74343 and 9.46378 seconds |

## Commit ledger

| Milestone | Commit | Verification | Push |
|---|---|---|---|
| Branch plan | | `git diff --check`; documentation-only snapshot | |
| Fresh CUDA baseline | | Full Release rebuild; 1,557 tests; 48 CUDA passes; 9,994 checks with zero failures; 311-file retained manifest | |
| Exact remaining-span census | | Full Release rebuild; 1,557 tests; exact six-node/nineteen-attempt CUDA twin; cookbook and design-family audit | |
| Adaptive cooperative growth | | Full Release rebuild; 1,557 tests; complete CUDA shortcut; 9,994 checks with zero failures; 311-file exact manifest; native resource report; full design-family update | |
| Cutoff-64/64-lane tuning | | Full Release rebuild; 1,557 tests; exact complete shortcut; 311-file identity; Nsight Systems and Compute attribution | |
| Four-shard host projection | | Full Release rebuild; 1,558 tests; exact complete shortcut; 311-file identity; 1.896-times Phase 2 | |
| Eight-shard host projection tuning | | Full Release rebuild; 1,558 tests; exact complete shortcut; 311-file identity; 2.023-times fresh-baseline Phase 2 | |
| Twelve-shard saturation point | | Full Release rebuild; 1,558 tests; exact complete shortcut; 311-file identity; projection improvement too small to retain as the final seam | |
| Dependency-staged projection upload | | Full Release rebuild; 1,558 tests; two exact complete CUDA shortcuts below the strict gate; 311-file identity; complete design-family update | |

## Progress log

### 2026-08-29 — campaign opened

Created the isolated branch/worktree from current `main`, loaded the complete
statification and GPU contracts, and fixed the target and first restructuring
shape. No production source changed.

### 2026-08-29 — M0 baseline complete

The clean build and all 1,557 native tests pass. The fresh untouched CUDA control
measures 20.1945325 seconds complete Phase 2 and 15.2917872 seconds exact
upload-plus-device work, with 3.8889809 seconds of projection. It passes 9,994
checks with zero failures. The complete 311-file shortcut tree and SHA-256
manifest are retained. M1 now measures candidate-span distribution before fixing
the adaptive cooperative threshold.

### 2026-08-29 — exact growth-span census verified

The aggregate processor growth census is zero in selected CUDA mode and cannot
choose a candidate-parallel threshold. The device growth owner now retains twelve
observation-only node/attempt buckets. A per-block shared reduction limits global
atomics, no proof branch reads the values, and the existing CUDA growth fixture
directly proves the independently derived six-node/nineteen-attempt distribution.
Fixed growth ownership rises by only 192 bytes to 263,225,552. The full Release
build and all 1,557 tests pass.

### 2026-08-29 — complete growth census and first cutoff

The complete CUDA shortcut counts 18,940,682 live frontier nodes and
4,498,333,440 candidate attempts. The concentration is strong enough to justify
candidate-parallel growth: spans of at least 256 candidates contain 68.7134% of
all attempts while representing 35.0923% of nodes; the adjacent 128 and 512
cutoffs contain 89.7395%/61.9906% and 31.3414%/10.2520%, respectively. The first
production cutoff is therefore 256, with 128 and 512 reserved for direct tuning.
The observation-only kernel adds about 1.04 seconds to route time versus M0 while
device time remains within normal run variation. All 311 shortcut artifacts are
byte-identical to M0 and the verifier reports 9,994 checks with zero failures.

### 2026-08-29 — first adaptive cooperative route retained

The production route disables the census, keeps spans below 256 on the original
one-thread path, warp-compacts long-node indices into a frontier-sized fixed list,
and assigns one 128-lane block to each long node. Thread zero builds the prefix
and gate summaries once in 2,016 bytes of `sm_89` shared memory; each lane stores
only a 1,280-byte stack and hashes the shared prefix plus its local expression
suffix. The complete CUDA shortcut is exact: 9,994 checks with zero failures and
all 311 artifact hashes match M0. Complete Phase 2 falls 20.1945325 → 15.1108074
seconds and exact device time 15.2917872 → 10.1747262 seconds. This is a retained
1.3364-times Phase 2 milestone, not the two-times finish. Projection remains about
3.92 seconds, leaving roughly 5.19 seconds to remove for the strict target. The
next step times cutoff and block-width peers before attributing the remaining
device interval.

### 2026-08-29 — cutoff 64 and 64 lanes retained

The complete profile showed the cutoff-256 short kernel still owned 4.963 seconds.
The census says spans of at least 64 contain 97.4104% of candidate attempts across
81.2753% of nodes, while the excluded 32-to-63 bucket contains only 1.9633% of
attempts but 9.7558% of nodes. Cutoff 64 with 64 lanes therefore increases useful
work per lane without allocating blocks to that low-work bucket. The exact run
reaches 12.4757739 seconds Phase 2 and 7.5260573 seconds device time, with all 311
artifacts identical and 9,994 checks/zero failures. Growth remains 4.718 seconds,
but projection is now the largest single seam at 3.9306738 seconds: name work
2.3527650, byte maps 0.6957810, reverse maps 0.2232920, pod maps 0.2172618.
The cooperative profile is compute/latency and branch-divergence constrained;
selective upload cannot remove the host construction time. The next retained
architecture step is the approved independent logical-block projection parallelism.

### 2026-08-29 — four-shard projection retained

Four process-owned fixed-capacity arenas now claim independent logical blocks and
reuse the unchanged serial builder. The exact merge copies all 24 column prefixes,
rebases every nested reference, and restores canonical descriptor order. The direct
twin and complete shortcut are exact: 1,558 tests, 9,994 verifier checks with zero
failures, and all 311 artifact hashes match M0. Projection falls 3.9306738 to
1.6821862 seconds and complete Phase 2 falls 12.4757739 to 10.6506 seconds. The
remaining 0.733-second gap is smaller than the measured parallel construction
headroom: the four workers consume 4.2007232 processor-seconds while merge consumes
0.3905575 seconds. The next measured tuning point is eight fixed shards; selective
upload remains deferred because transfer is still only about 0.38 seconds.

### 2026-08-29 — eight-shard tuning retained

Doubling the fixed construction shards reduces projection from 1.6821862 to
1.3315546 seconds and complete Phase 2 from 10.6506 to 9.9825 seconds. The fresh
20.1945325-second control is now beaten by 2.023 times, but the campaign retains
the stricter historical 9.917614-second threshold, so 0.0649 seconds still remain.
The run passes 1,558 unit tests, 9,994 verifier checks with zero failures, and all
311 artifact hashes. Construction still owns 4.5878480 processor-seconds against
1.3315546 seconds wall, so twelve shards are the next bounded tuning point; the
0.4056626-second merge and 0.38-second transfer are not changed in this step.

### 2026-08-29 — twelve shards establish saturation

Twelve shards remain exact but improve projection by only 0.0282 seconds, from
1.3315546 to 1.3033454, while merge rises from 0.4056626 to 0.4433709 seconds.
The complete run measures 10.1734 seconds because exact device time varies upward
to 7.8794975 seconds. All 1,558 tests, 9,994 verifier checks, and 311 artifact
hashes pass. The failed performance point is preserved rather than silently
discarded. More host arenas are no longer justified; current profiling must now
separate the roughly 0.38-second transfer from device kernels and target only
task-reachable or changed prefixes whose avoided bytes can close the strict gate.

### 2026-08-29 — dependency-staged upload clears the strict gate

Kernel dependency inspection separates fifteen columns needed by filter, growth,
and ordering from nine columns first needed by evaluation: rule-string records and
bytes, reverse-map views/entries/slots/key bytes/owners, plain-data run values,
and metadata. The first experiment registered the live projection vectors and
allowed the transfer to overlap kernels. It stayed exact but regressed Phase 2 to
10.8707 seconds, proving that pinned live builder memory and copy/compute
competition cost more than they hid.

The accepted route owns fixed pinned staging instead. Immediately after projection
it synchronously copies the fifteen early columns, copies the nine deferred used
prefixes into staging, and queues their host-to-device transfers on a nonblocking
stream. Host task/filter/growth scheduling proceeds while that stream runs; after
the small task upload, the stream joins before filter. No kernel observes a partial
image and the all-column device checksum covers both the ordinary and staged doors.
Device ownership remains 729,721,629 bytes; the new fixed pinned host staging is
31,219,712 bytes. This milestone changes dependency timing, not transfer volume;
generation-backed dirty-prefix residency remains a separate follow-on.
The clean build passes 1,558 tests. Consecutive complete CUDA shortcuts measure
9.74343 and 9.46378 seconds, with 9,994 checks, zero failures, and all 311 hashes
identical to M0. The strict 19.835229-second reference is beaten by 2.036 and
2.096 times; the fresh 20.1945325-second control is beaten by 2.073 and 2.134 times.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
