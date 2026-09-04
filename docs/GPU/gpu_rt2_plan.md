<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Phase 2 GPU runtime 2 plan

> **Purpose:** this file is the persistent working memory for the third Phase 2
> GPU performance campaign. Read it after every context compaction. Update it
> after every measurement, design conclusion, failed experiment, commit, and
> change of next action. Conversation memory is not authoritative.

## Current state

- **Branch:** 
- **Worktree:** 
- **Base:** current `main` tip 
- **Hardware:** NVIDIA GeForce RTX 4070 Laptop GPU, native `sm_89`
- **Status:** deterministic fixed candidate windows are exact and 9.3 percent
 faster than the pooled-prefix checkpoint, but the final target is not met
- **Active milestone:** M3 — close the measured 3.81688-second remainder
- **Next action:** profile the dominant survivor-probe and cheap-gate kernels,
 then select only an exact fixed-capacity seam that can close that remainder

## Objective and acceptance

Accelerate the complete FTA shortcut Phase 2 CUDA bracket by at least two times
relative to a fresh CUDA control built from this branch's unchanged `main` tip.
The retained independent confirmation on the tree that landed to `main` is
9.46378 seconds, so the provisional strict target is **4.73189 seconds or less**.
The two accepted fresh M0 controls are 10.0993 and 10.1686 seconds. Their
two-times thresholds are 5.04965 and 5.08430 seconds, so the final strict target
remains the smaller inherited threshold: **4.73189 seconds or less**.

The improvement must preserve exact processor/CUDA semantics, complete
provenance, canonical ordering, deterministic doom selection, fixed ownership,
and all assertion-based failure contracts. Prover and overall shortcut time are
reported beside Phase 2 and may not regress relative to the fresh control.

Final acceptance also requires:

- a full clean Windows Release rebuild and every native unit test passing;
- CUDA-first and processor shortcut runs through `main.py`, never direct
 executable runs except `--unit-tests`;
- zero verifier failures on both routes;
- exact SHA-256 identity for every retained semantic artifact;
- native CUDA telemetry proving the optimized route executed with no fallback;
- every numeric timing parser accepting decimal and exponent forms;
- fixed-capacity assertions, complete provenance, canonical record order, and
 inclusive doom-prefix semantics unchanged;
- no new hot-path allocation, live resize, truncation, unified-memory fallback,
 or processor semantic replay;
- every new production function documented with a complete GL-style Doxygen
 block and covered by direct positive and negative unit tests;
- every complete source milestone committed with `git add -A`, a detailed
 message, a clean scoped status check, and an immediate push;
- relevant agent SwDD, user SwDD, and MPU claims audited in the same commit as
 every hardware-relevant architecture change.

### Accepted M0 CUDA control

| Run | Phase 2 | Route | Preparation | Projection wall | Projection worker processor total | Schedule | Device route | Finalize | Prover | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpu_rt2_m0_cuda_free_v1` | 10.0993 | 9.4220125 | 1.8676236 | 1.2839528 | 4.9063418 | 0.5836707 | 7.3168369 | 0.2375538 | 115.648 | 121.00890 |
| `gpu_rt2_m0_cuda_free_v2` | 10.1686 | 9.4649476 | 1.9335160 | 1.3460415 | 4.9773255 | 0.5874739 | 7.2988975 | 0.2325357 | 116.164 | 121.88687 |

All values are seconds. Projection worker processor time is summed across
workers and therefore is not additive with projection wall time. The complete
Phase 2 spread is 0.69 percent. Both runs executed 48 GPU pass rows, passed
9,994 verifier checks with zero failures, and produced identical 311-file
path/length/SHA-256 manifests.

### M0 current-route attribution

The fastest accepted control reconciles as follows. The small difference in the
last decimal is printed-timer rounding.

| Mutually exclusive interval | Seconds | Share of Phase 2 |
|---|---:|---:|
| Preparation: projection plus scheduling | 1.8676236 | 18.49% |
| Device route | 7.3168369 | 72.45% |
| Download and sealing finalization | 0.2375538 | 2.35% |
| Phase 2 outside logged GPU passes | 0.6772857 | 6.71% |
| Complete Phase 2 | 10.0993 | 100.00% |

Preparation is projection 1.2839528 plus scheduling 0.5836707. Nsight Systems
accounts for the device route as 6.8771104 seconds of kernels, 0.4264739 seconds
of transfers and device memory operations, and 0.0132526 seconds of remaining
launch gaps and device-route overhead.

| Device kernel family | Seconds | Share of kernel time | Launches |
|---|---:|---:|---:|
| Cooperative growth | 3.754719839 | 54.6% | 194 |
| Filter count | 1.327047146 | 19.3% | 51 |
| Short-node growth | 0.985512417 | 14.3% | 224 |
| Filter emit | 0.486511117 | 7.1% | 51 |
| All other kernels | 0.323319847 | 4.7% | remaining |

The four dominant families total 6.5537905 seconds and own 95.3 percent of all
kernel time. The target requires removing 5.3674100 seconds from the fastest
control. If preparation, finalization, transfers, other kernels, and the
outside-pass remainder do not change, the dominant families may retain only
1.1863805 seconds: they require at least a 5.52-times combined speedup. Their
ideal-removal Amdahl floor is 3.5455095 seconds complete Phase 2, so this is the
only measured family with enough removable time to reach the target.

Compute profiles explain why launch tuning alone is insufficient:

- Cooperative growth: 65.59% achieved occupancy, 46.73% issue-slot use,
 10.83 active threads per warp, 50.7% long-scoreboard share, 99.98% L2 hit
 rate, 2.28% DRAM throughput, 53 registers, 1,280-byte compiled stack, and
 2,016-byte shared prefix state on the live `sm_89` image.
- Filter count: 83.27% achieved occupancy but only 3.18% issue-slot use and
 3.41% eligible-scheduler cycles; local-memory traffic owns 80.92% of L1
 sectors, memory-operation queue stalls own 41.4% of issue intervals, barrier
 imbalance owns 31.2%, and the kernel uses 34 registers plus a 1,168-byte
 compiled stack.
- Short-node growth: 40.40% achieved occupancy, 36.60% issue-slot use, only
 3.75 active threads per warp, 44.7% long-scoreboard share, 62 registers, and
 a 3,104-byte compiled stack. Long nodes are discovered inside this launch,
 so their threads stop doing short work while the launch still spans the full
 frontier.

### M1 approved architecture

**Bounded signature-and-candidate pipeline.** This is one coordinated redesign
of the filter and growth families; implementing only one half cannot meet the
measured Amdahl requirement.

1. Intern filter calls by the exact tuple `(logical block, memory kind,
 iteration ceiling, whole-key widening, statement count)`. Every original
 call retains its processor ordinal but points at one immutable filter class.
2. Add fixed projection columns containing each statement's exact canonical
 one-expression normalized key slice. The host projects bytes only; the GPU
 still performs frozen-validity, selected-map, whole/subkey, and iteration
 decisions. No host verdict enters proof flow.
3. Replace count-plus-predicate-replay with one class-row verdict array, one
 segmented exclusive scan, and one stable compact emission. The scan caps each
 class at its first 8,192 accepted statement indices, and original calls reuse
 the identical compact span without changing their growth-call ordering.
4. Build one fixed prefix record per live growth node, then compact short and
 long node indices separately before candidate work. Short growth launches
 only compacted short nodes instead of the full frontier.
5. Process candidate attempts through bounded deterministic windows: cheap
 reachability and validity gates, compact survivors, then normalized-key and
 owner probes, followed by event and child emission. Prefix slices plus one
 candidate scalar replace per-thread statement, position, and normalized-key
 arrays. Window order is the canonical `(call, run, path, candidate)` ordinal;
 atomic append order remains non-semantic and the existing ordering tokens
 reconstruct every observable stream.

Before production routing, an observation-only census records unique filter
classes, class rows, short/long nodes, candidate attempts and survivor peaks per
gate and per frontier depth. The new class, verdict, prefix, window, survivor,
event, and child buffers receive explicit measured ceilings; construction
allocates once, every append asserts, and there is no resize, truncation,
fallback, unified memory, or processor replay. If the census contradicts the
required reuse or capacity, implementation stops for a new maintainer decision;
it does not silently change this architecture.

The direct CUDA twin covers duplicate-class reuse, frozen validity, all four
memory kinds, whole-only and subkey hits, iteration edges, exactly 8,192 plus
overflow candidates, stable equal-name order, mandatory reachability,
hypothesis and secondary-variable gates, owner rejection, repeated variables,
whole-only events, child growth, depth eight, fixed-window boundaries, and
capacity assertions. Complete verification remains the two 9,994-check routes
plus exact 311-file SHA-256 identity. The relevant agent SwDD, user SwDD, and
MPU diagrams change in the same production commit.

The measured upper bound is the full 6.5537905 seconds owned by the redesigned
families. Acceptance requires their combined time at or below 1.1863805 seconds
when all other intervals are unchanged; every milestone re-profiles the full
route rather than claiming that upper bound as an expected result.

### M2 observed reuse and frontier census

The complete CUDA shortcut census on source commit validates the
approved class seam. Across 48 device passes, 24,229 processor-order filter
calls contain 8,853 exact five-field classes. The other 15,376 calls are exact
duplicates: 63.461 percent of calls. One representative per class examines
8,178,826 rows instead of 89,618,957 rows, removing 81,440,131 repeated rows,
or 90.874 percent of the current predicate work before class compaction. The
weighted row reduction is 10.957 times and maximum class multiplicity is 32.

The same run exactly reproduces the earlier growth census: 18,940,682 live
frontier nodes and 4,498,333,440 candidate attempts, or 237.496 attempts per
node. Aggregate fixed span buckets are:

| Bucket | Nodes | Candidate attempts |
|---:|---:|---:|
| 0 | 24,285 | 0 |
| 1 | 28,597 | 28,597 |
| 2 | 72,761 | 184,562 |
| 3 | 192,851 | 1,075,655 |
| 4 | 458,961 | 5,309,952 |
| 5 | 921,308 | 21,574,990 |
| 6 | 1,847,822 | 88,314,816 |
| 7 | 3,652,659 | 345,064,145 |
| 8 | 5,094,708 | 945,820,874 |
| 9 | 4,704,936 | 1,681,117,843 |
| 10 | 1,744,909 | 1,181,262,800 |
| 11 | 196,885 | 228,579,206 |

This observation run is not a performance candidate because both the
allocation-free quadratic class scan and one extra span kernel per frontier
depth are enabled. It measured 10.3548975 seconds complete Phase 2, 9.6434785
seconds in logged GPU routes, and 7.1124543 seconds in device routes. It passed
all 9,994 checks with zero failures and its 311-file manifest exactly matches
the accepted clean control. The remaining pre-routing evidence is the number
of candidates surviving each cheap gate at each frontier depth and the peak
survivor prefix that fixes each compact-window capacity.

The second observation run completes the gate census. Aggregate counts are:

| Depth | Attempts | Mandatory reachable | Validity comparable | Hypothesis compatible | Secondary/key-length compatible |
|---:|---:|---:|---:|---:|---:|
| 1 | 570,064 | 457,841 | 457,841 | 457,841 | 457,841 |
| 2 | 121,346,869 | 86,461,349 | 84,512,258 | 70,823,542 | 55,191,171 |
| 3 | 2,662,156,411 | 2,587,300,727 | 2,532,425,096 | 2,356,699,134 | 1,308,065,007 |
| 4 | 1,560,295,666 | 1,514,467,214 | 1,463,751,615 | 1,394,477,594 | 685,239,715 |
| 5 | 153,964,313 | 85,718,136 | 81,075,224 | 78,556,093 | 33,858,349 |
| 6 | 117 | 113 | 113 | 113 | 113 |
| 7 | 0 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **4,498,333,440** | **4,274,405,380** | **4,162,222,147** | **3,901,014,317** | **2,082,812,196** |

The overall key-length gate removes no candidate after the shape gates on this
workload, so its counts exactly equal the final column above. Branch outcomes
after key probing are:

| Depth | Subkey present | Owner satisfied | Whole key present | Terms satisfied | Accepted events | Children |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 457,841 | 457,841 | 304,614 | 198,881 | 457,841 | 395,015 |
| 2 | 11,382,711 | 11,382,711 | 625,241 | 30,858,414 | 11,382,711 | 11,357,614 |
| 3 | 7,065,994 | 6,292,075 | 817,748 | 580,225,057 | 6,415,267 | 6,291,246 |
| 4 | 675,013 | 665,397 | 493,118 | 340,872,813 | 668,820 | 662,807 |
| 5 | 9,726 | 9,726 | 9,726 | 33,857,888 | 9,726 | 14 |
| 6 | 0 | 0 | 0 | 38 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Total** | **19,591,285** | **18,807,750** | **2,250,447** | **986,013,091** | **18,934,365** | **18,706,696** |

Maximum counts in one GPU pass, which bound any non-windowed representation,
are retained in . The capacity-driving
depths are:

| Depth | Attempts peak | Secondary/key-length peak | Subkey peak | Owner peak | Event peak | Child peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 54,893 | 50,007 | 50,007 | 50,007 | 50,007 | 39,532 |
| 2 | 9,056,768 | 3,953,719 | 719,103 | 719,103 | 719,103 | 718,754 |
| 3 | 203,642,409 | 90,873,097 | 467,869 | 453,792 | 454,834 | 453,792 |
| 4 | 172,437,072 | 66,029,213 | 60,953 | 60,902 | 60,915 | 60,902 |
| 5 | 22,782,059 | 7,112,035 | 1,948 | 1,948 | 1,948 | 4 |
| 6 | 31 | 31 | 0 | 0 | 0 | 0 |

The complete twelve-field aggregate and peak tables are retained in
 and
. Depths three and four own 93.87
percent of all attempts. Their post-cheap-gate single-pass peaks rule out a
complete survivor allocation: production must stream deterministic bounded
windows as approved. Across all depths, cheap gates remove 53.70 percent before
normalized-key probes; only 0.941 percent of key-probed candidates have a
subkey, and owner signatures reject 783,535 of those present subkeys.

The gate-census run measured 12.1850 seconds complete Phase 2 and 9.3320876
seconds in device routes because its high-volume observation atomics are active;
it is not a performance candidate. It passed all 9,994 checks with zero failures
and its 311-file manifest exactly matches M0. The approved pre-routing census is
therefore complete.

### M2 production filter-class milestone

The first production slice implements exact five-field class interning in the
fixed host schedule and one count, scan, stable compact emission, and radix sort
per class on CUDA. One fixed original-to-class column expands the class count and
offset back to each original call. Growth reads the shared statement span but
retains the original processor call ordinal in every event and downstream order
token. The direct twin changes each signature field independently, executes an
exact duplicate on native CUDA, and proves that two original calls share the
filter span while producing distinct original-call events.

The valid idle production run is:

| Run | Phase 2 | Route | Preparation | Projection | Schedule | Device route | Finalize | Prover | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gpu_rt2_filter_class_v2` | 9.40098 | 8.7148406 | 1.8930928 | 1.3264584 | 0.5666349 | 6.5884318 | 0.2333147 | 113.374 | 118.80591 |

Compared with accepted M0 v1, complete Phase 2 falls by 0.69832 seconds, or
6.91 percent; the device route falls by 0.7284051 seconds and the enclosing route
by 0.7071719 seconds. The schedule still contains 24,229 original calls and
8,853 classes. Class filtering examines 8,178,826 rows and retains 827,746 class
rows instead of 8,815,888 duplicated per-call rows. The output is exact:
19,194,381 accepted events, 841,301 unique requests, 9,994 checks with zero
failures, and zero differences across the 311-file M0 manifest.

Fixed filter ownership rises from 25,512,447 to 25,577,983 bytes for the class
mapping. The retained 864-byte gate-census record makes fixed growth ownership
267,420,728 bytes. Complete application-owned device memory is now 729,788,029
bytes, 66,400 bytes above M0. All arrays allocate once and every capacity breach
asserts.

This slice does not approach the final 4.73189-second target by itself, as the
approved Amdahl analysis predicted. The post-change profile below fixes growth
prefix/window work as the measured remaining dominant seam.

### M2 post-filter profile

Nsight Systems report  attributes
6.386228408 seconds to kernels and 0.465629454 seconds to device memory
operations. Transfers and memory operations are 0.445402938 host-to-device,
0.012104518 device-to-host, 0.007860621 memset, and 0.000261377 device-to-device
seconds. The profiled run includes profiler overhead and is not a timing
candidate; the accepted 9.40098-second idle run remains the performance result.

| Kernel family | M0 seconds | Post-class seconds | Change |
|---|---:|---:|---:|
| Cooperative growth | 3.754719839 | 4.589203073 | +0.834483234 |
| Short-node growth | 0.985512417 | 1.166508319 | +0.180995902 |
| Filter count | 1.327047146 | 0.118330508 | -1.208716638 |
| Filter emit | 0.486511117 | 0.117727976 | -0.368783141 |
| Filter class-span expansion | 0 | 0.000185058 | +0.000185058 |

Count plus emit falls from 1.813558263 to 0.236058484 seconds, a 7.683-times
speedup and 1.577499779-second removal. The new expansion kernel is 0.185
milliseconds. Cooperative plus short growth rises from 4.740232256 to
5.755711392 seconds and now owns 90.13 percent of all kernel time. Complete
kernel time still falls by 0.490881992 seconds, consistent with the valid route
improvement after the transfer and launch intervals are included.

The native `sm_89` production resource image is also exact evidence. The
compile-time-false cooperative specialization uses 59 registers, 1,280 stack
bytes, and 2,016 shared bytes versus M0's 53 registers with the same stack and
shared sizes. The short specialization uses 63 registers and 3,104 stack bytes
versus M0's 62 registers and the same stack. The observation specialization is
separate and does not execute in production. The resource increase and growth
time increase are both measured; this plan does not claim an unprofiled causal
mechanism between them. The approved fixed-prefix, compact-index, and bounded
candidate-window route replaces this remaining growth shape rather than tuning
the class slice further.

### M2 prefix-pool capacity completion

The earlier gate census fixed candidate-window and survivor peaks but did not
retain the simultaneous `NameId` counts required by the approved pooled-prefix
representation. The existing mode-2 kernels now add each productive node's
normalized payload length, distinct normalization-variable count, and distinct
secondary-variable count into three otherwise-unused depth-zero census fields.
The host resets those fields per frontier and reports the maximum of each across
the complete device pass. Production mode compiles every write out, no proof
branch reads a counter, and device ownership does not change.

The result-only ABI grows by 24 bytes, from 1,072 to 1,096. The direct mode-2
fixture asserts that all three measured pools are populated while retaining its
exact six-node, nineteen-candidate, seven-event, four-request, and five-subkey
contracts. The first clean build and 1,559-unit gate pass. The observation
activation uses mode 2 only for the complete sizing run; the approved production
mode remains zero.

The complete 48-pass CUDA run fixes the following simultaneous live-frontier
ceilings and production capacities. Each capacity is the measured maximum plus
25 percent, rounded upward to the next 262,144-value block. Every value is one
four-byte `NameId`; pool exhaustion is an assertion and has no fallback.

| Pool | Measured maximum | Fixed capacity | Headroom | Fixed bytes |
|---|---:|---:|---:|---:|
| Normalized payload | 14,074,478 | 17,825,792 | 26.653% | 71,303,168 |
| Normalization variables | 4,369,164 | 5,505,024 | 25.997% | 22,020,096 |
| Secondary variables | 951,515 | 1,310,720 | 37.751% | 5,242,880 |
| **Total** | **19,395,157** | **24,641,536** | — | **98,566,144 (94 MiB)** |

The observation-only run measured complete Phase 2 at 11.9766 seconds: 2.4627240
seconds of preparation, 8.5703674 seconds on the device route, 0.2386094 seconds
of finalization, and 0.7048992 seconds outside logged GPU passes. Its timing is
not a production candidate. All 9,994 verifier checks pass, and the 311-file
path/length/SHA-256 manifest has zero differences from accepted M0.

### M2 pooled-prefix production milestone

The production growth owner now prepares one immutable 32-byte prefix header
for every live frontier node. Its three offsets and bounded lengths name slices
in the measured 94 MiB pools; the header also carries hypothesis validity,
non-exempt validity, and packed path summaries. The 56-byte frontier node stays
unchanged. Both short and cooperative candidate kernels read the same pooled
prefix and build only the appended-expression suffix.

The preparation pass writes complementary short and long flags. Stable CUB
selection over the ascending frontier index produces both compact work lists;
pool allocation offsets and block arrival order are never ordering identities.
The first implementation instead atomically appended short and long indices.
All 40 completed GPU pass rows matched the accepted filter run on every reported
semantic count, but the later processor deposit stopped at
`removeOwnerFromRun: owner is not on key`. That trace isolated the newly changed
physical traversal seam. The fix-forward stable selection passed the complete
route and is now part of the invariant.

The retained production route reports exactly the original census maxima:
14,074,478 payload values, 4,369,164 normalization variables, and 951,515
secondary variables. Fixed growth ownership is 405,840,199 bytes, including
the pool arrays, prefix headers, two flag arrays, two index arrays, and reusable
CUB scratch. Complete fixed device ownership is 868,207,500 bytes:

| Owner | Fixed bytes |
|---|---:|
| Projection | 299,951,120 |
| Tasks | 731,144 |
| Filtering | 25,577,983 |
| Growth | 405,840,199 |
| Ordering | 98,838,019 |
| Evaluation | 37,269,035 |
| **Total** | **868,207,500** |

The clean production run measured 9.42072 seconds complete Phase 2, 8.7270083
seconds in the enclosing route, 1.8753261 seconds of preparation, 1.2914174
seconds of projection, 0.5839091 seconds of scheduling, 6.59380093 seconds of
exact device work, 6.6169076 seconds in the device route, 0.2347745 seconds of
finalization, 114.231 seconds in the prover, and 119.48693 seconds overall. It
passed all 9,994 checks and all 311 artifact hashes match M0. The 0.01974-second
increase from the accepted 9.40098-second filter route is 0.21 percent. This
slice is retained for the approved suffix-only window pipeline; it is not
claimed as a standalone speedup.

### M2 deterministic candidate-window milestone

The production growth route now sorts each live frontier by the canonical
`(call, run, path)` node key using stable radix passes. Exact candidate counts
are scanned into one ordinal space. Consecutive half-open windows of at most
16,777,216 attempts cover that space without gaps or overlap; the largest
measured depth requires thirteen windows. Short-thread and 64-lane cooperative
kernels apply reachability, validity, hypothesis, secondary-variable, and key-
length gates. Stable selection compacts only survivors, after which one kernel
performs suffix normalization, subkey/owner/whole-key probes, event emission,
and child emission.

The direct CUDA twin deliberately sets the capacity to four records. It forces
a ten-attempt depth and a nine-attempt depth to cross several boundaries, then
matches the separately compiled observation route on all seven events, four raw
requests, and nineteen census attempts. The production capacity owns 256 MiB of
16-byte attempts, 16 MiB of flags, 64 MiB of survivor indices, canonical node
indices and 64-bit span arrays, and 12,868,095 bytes of reusable CUB scratch.
Fixed ownership is now:

| Owner | Fixed bytes |
|---|---:|
| Projection | 299,951,120 |
| Tasks | 731,144 |
| Filtering | 25,577,983 |
| Growth | 812,965,447 |
| Ordering | 98,838,019 |
| Evaluation | 37,269,035 |
| **Total** | **1,275,332,748** |

The clean complete run measured 8.54877 seconds for Phase 2, 7.8623967 seconds
inside logged GPU routes, 5.72042443 seconds of exact device work, 1.8873552
seconds of preparation, 1.2531050 seconds of projection, 0.6342499 seconds of
scheduling, 0.2280701 seconds of finalization, 113.294 seconds in the prover,
and 118.37951 seconds overall. It is 0.87195 seconds or 9.256 percent faster
than the pooled-prefix checkpoint. All 1,559 tests pass, the verifier reports
9,994 checks and zero failures, and all 311 path/length/SHA-256 rows exactly
match M0.

The new Systems profile attributes 4.961267042 seconds to kernels. Four growth
kernels own 4.098578580 seconds or 82.61 percent: survivor probes
1.768231566, cooperative cheap gates 1.305358563, prefix preparation
0.680364406, and short cheap gates 0.344624045 seconds. Host-to-device copies
own 0.485711417 seconds; device-to-host copies 0.013755820; memsets
0.019233521; device-to-device copies 0.000272802. The strict target remains
3.81688 seconds below this exact checkpoint, so M3 must reduce this measured
growth family rather than tune the already-small window sorts and scans.

## Evidence inherited from `gpu_rt1`

The exact branch landed onto `main` before this campaign. Its independent CUDA
confirmation measured 9.46378 seconds complete Phase 2, 1.3362983 seconds host
projection, 0.4339587 seconds projection merge, 1.8428385 seconds staged
preparation, 6.6995169 seconds task upload plus join plus semantic kernels,
8.7941203 seconds enclosing CUDA route, 113.587 seconds prover time, and
118.60189 seconds overall shortcut time. It passed 9,994 checks with zero
failures and all 311 retained artifacts matched the original processor/CUDA
baseline by SHA-256.

The latest complete Systems attribution before the final staged-upload change
measured approximately:

| Seam | Retained evidence |
|---|---:|
| Cooperative growth | 3.732334 seconds |
| Short-node growth | 0.985799 seconds |
| Filter count | 1.171789 seconds |
| Filter emit | 0.566028 seconds |
| Host projection | about 1.33 seconds |
| Host-to-device transfer | 0.364500 seconds |
| Device-to-host transfer | 0.011897 seconds |

The representative cooperative launch reached 65.57% achieved occupancy,
46.72% compute throughput, 2.11% DRAM throughput, and a 99.90% L2 hit rate, but
only 10.83 active threads per warp. The current device limit is therefore
branch divergence and instruction latency, not device-memory bandwidth. This is
historical attribution only; M0 must profile the landed staged-upload route
before it selects a target.

The current GPU already owns fixed-capacity projection, filtering, normalized-
key growth, exact request ordering and deduplication, evaluation, provenance,
doom selection, and firing-record materialization. Host work is projection,
schedule construction, bounded transfers, downloads, sealing, and the shared
deposit door. The explicit backend remains shortcut-only and never falls back.

## Non-negotiable semantic and ownership contracts

1. The processor implementation remains the oracle and selectable peer.
2. Selected CUDA mode never falls back. Missing hardware, capacity failure,
 allocation failure, launch failure, identity failure, or impossible state
 asserts at its owner.
3. GPU filtering, growth, ordering, evaluation, provenance, doom selection, and
 firing-record materialization stay on the GPU. The host may only project,
 schedule, transfer, seal, and deposit already-decided content.
4. Atomic append order, warp scheduling, and kernel completion order remain
 non-semantic. Exact ordering tokens reconstruct every observable stream.
5. Persistent frontier and event records remain compact, pointer-free, and
 fixed-capacity. No optimization may copy full normalized keys into every
 frontier row without measured ownership and explicit architecture approval.
6. Identifier mint order, decoded-byte lexical order, stable equal-name order,
 request first occurrence, growth position, provenance dependency order, and
 inclusive doom-prefix semantics remain exact.
7. Production buffers are process-owned and preallocated. Capacity widening is
 evidence-driven and assertion-checked; overflow never truncates or resizes.
8. Observation counters and profiler hooks are process documentation only. No
 proof branch may read them.
9. Processor and CUDA verification runs are sequential, with distinct logs and
 retained artifact manifests.
10. Any change to Phase 2 ownership, frontier shape, hash-engine behavior,
 synchronization semantics, projection generations, or processor/GPU
 boundary requires explicit maintainer approval before production code.

## Measurement-first campaign strategy

M0 establishes the fresh target and attributes the current staged-upload route.
No production architecture is selected from historical profiles alone.

The profile must reconcile the complete Phase 2 bracket into mutually exclusive
host and device intervals and break device time down by kernel family. At a
minimum it records filtering, short and cooperative growth, ordering and
deduplication, evaluation expansion and materialization, canonical ordering,
doom selection, every transfer class, projection construction and merge,
schedule construction, downloads, sealing, launch counts, and synchronizations.
It also records compiled registers, local stack, shared memory, achieved
occupancy, active threads per warp, branch efficiency, compute throughput, DRAM
throughput, and L2 behavior for the dominant kernels.

After M0, this file will contain one evidence-backed architecture proposal with
an Amdahl bound showing that its measured removable time can reach the target.
Production implementation pauses at that checkpoint until the maintainer
approves the specific design. A smaller signature-preserving tuning change may
proceed without a new architecture decision only when it preserves the existing
ownership and semantics and its exact twin already covers the changed behavior.

Candidate directions are hypotheses, not approved designs:

- reduce cooperative-growth divergence by grouping nodes or candidate lanes by
 remaining-span/work shape while retaining exact stream reconstruction;
- replace the two-pass filter replay with one bounded compaction or a fused
 schedule only if the exact first-8,192 acceptance and stable order remain
 directly twinned;
- fuse adjacent growth/order or evaluation stages when profiler data shows
 material launch, memory-traffic, or synchronization cost and fixed ownership
 remains explicit;
- reduce short-node cost only if its current time is still material after the
 landed staged-upload route;
- reduce projection construction or merge only after the fresh profile proves
 its wall-time share can materially contribute to a 4.73189-second target;
- add generation-backed residency only if a measured same-process workload has
 reuse; the current disjoint shortcut passes do not justify cross-process
 caching.

## Milestones

### M0 — branch, plan, fresh control, and attribution

- [x] Create in its own worktree from current `main` tip.
- [x] Read the complete statification container, string, and GPU cookbooks.
- [x] Read the complete `gpu_rt1` campaign plan and landed evidence.
- [x] Create this persistent plan and progress log.
- [x] Commit and push the branch plan.
- [x] Full Windows Release rebuild with the canonical child environment.
- [x] Run all native unit tests.
- [x] Run and retain two fresh CUDA shortcut controls first.
- [x] Parse complete Phase 2, route, projection, device, prover, and overall
 timings with exponent-safe numeric extraction.
- [x] Retain the complete shortcut artifact trees and SHA-256 manifests.
- [x] Profile the exact landed CUDA control with Systems, then Compute on the
 dominant kernel launches.
- [x] Reconcile the complete Phase 2 bracket and fix the final two-times target.

### M1 — evidence-backed design checkpoint

- [x] Rank current seams by exact wall-time contribution and calculate their
 maximum removable time.
- [x] Reject any direction whose Amdahl bound cannot reach the target.
- [x] Derive the exact semantic twin and fixed-capacity ownership needed for the
 leading direction.
- [x] Record one concrete architecture proposal here, including data shape,
 ordering proof, capacity, failure contract, documentation impact, direct
 tests, and expected time removal.
- [x] Obtain explicit maintainer approval before production architecture code.

### M2 — first approved dominant-seam milestone

- [x] Add the observation-only census or direct fixture required by the approved
 design without entering proof flow.
 - [x] Measure exact filter classes using all five approved signature fields.
 - [x] Enable the existing directly tested frontier-span node and candidate
 counters.
 - [x] Add candidate survival counts for each approved gate and frontier depth.
 - [x] Run the complete survivor census and record all aggregate and peak
 capacities.
- [ ] Implement the approved bounded representation and direct exact twin.
 - [x] Exact filter-class interning, one class span, and original-call mapping.
 - [x] Fixed growth prefixes and stable compact short/long indices.
 - [x] Deterministic bounded candidate windows.
- [x] Full rebuild and native unit gate for the filter-class slice.
- [x] Complete CUDA shortcut, zero-failure verifier, and 311-file identity gate
 for the filter-class slice.
- [x] Profile the changed route and record the measured contribution here.
- [x] Update the complete design-document family for the filter-class slice.
- [x] Commit and push the coherent filter-class milestone.
- [x] Complete CUDA shortcut, zero-failure verifier, and 311-file identity gate
 for the pooled-prefix slice.
- [x] Update the complete design-document family for pooled prefixes and stable
 compact work lists.
- [x] Commit and push the coherent pooled-prefix milestone.
- [x] Complete CUDA shortcut, zero-failure verifier, and 311-file identity gate
 for the candidate-window slice.
- [x] Update the complete design-document family for deterministic windows.

### M3 — close the measured remainder

- [x] Re-profile; do not reuse M0 attribution after a large change.
- [ ] Select only a seam whose measured removable time closes the remaining gap.
- [ ] Repeat the approval, exact-twin, fixed-capacity, complete-run, identity,
 documentation, commit, and push gates.
- [ ] Preserve exact failed performance experiments in this ledger; fix forward
 and never silently discard them.

### M4 — final two-times gate

- [ ] Full clean Release rebuild and all native tests.
- [ ] Fresh CUDA-first and processor shortcut runs through `main.py`.
- [ ] Zero verifier failures and exact processor/CUDA artifact SHA-256 identity.
- [ ] Complete CUDA Phase 2 at or below the final target and at least 2.000 times
 faster than the fresh M0 CUDA control.
- [ ] No regression in prover or overall shortcut time against M0.
- [ ] Final profile, transfer bytes, fixed ownership, queue peaks, occupancy,
 compute/DRAM throughput, branch/commit, and logs recorded here.
- [ ] Agent SwDD, user SwDD, and MPU audit complete.
- [ ] Final source/documentation commit pushed; branch clean.

## Verification commands

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe' GL_Quick_VS\GL_Quick.sln -p:Configuration=Release -p:Platform=x64 -t:Rebuild -verbosity:minimal
& 'GL_Quick_VS\GL_Quick\gl_quick.exe' --unit-tests
& 'C:\Users\nikol\anaconda3\python.exe' main.py --shortcut --phase2-backend cuda --run-descriptor gpu_rt2_m0_cuda
& 'C:\Users\nikol\anaconda3\python.exe' main.py --shortcut --phase2-backend cpu --run-descriptor gpu_rt2_m0_cpu
```

Every build and run writes both streams to a unique `.debug` log. Phase 2 is
the sum of anchored `[PHASE2-TIMING] iteration_seconds` rows. Exact device time
is the sum of `[GPU-PHASE2] device_seconds`. Numeric extraction accepts
`[0-9.eE+-]+`; a decimal-only parser is forbidden because it corrupts scientific
notation. The baseline and every accepted milestone retain separate artifact
sets or complete manifests.

## Evidence ledger

| Date | Commit or worktree | Evidence | Result |
|---|---|---|---|
| 2026-08-29 |, | `git worktree add -b sandbox/gpu_rt2 .worktree\gpu_rt2 main` | Dedicated clean worktree created from current `main` tip |
| 2026-08-29 | | Complete reads of `09b_statification_cookbook.md`, `09c_string_statification_cookbook.md`, `09d_gpu_cookbook.md`, and `docs/GPU/gpu_rt1_plan.md` | Static-memory, string lifetime, GPU exactness, landed architecture, performance, and verification contracts loaded before source work |
| 2026-08-29 | inherited landed route | Independent `gpu_rt1` confirmation | Phase 2 9.46378 seconds; route 8.7941203; prover 113.587; overall 118.60189; 9,994 checks, zero failures, 311 exact artifacts |
| 2026-08-29 | | Branch plan snapshot | Persistent plan committed and pushed before measurements |
| 2026-08-29 | worktree | ;  | Clean Release rebuild passed; 1,558 of 1,558 native unit tests passed |
| 2026-08-29 | rejected occupied control |  | 11.6578429-second Phase 2 and 9,994 zero-failure checks retained as correctness evidence only; maintainer reported the computer was occupied, so it is excluded from performance evidence |
| 2026-08-29 | accepted clean control v1 | ;  | Phase 2 10.0993; route 9.4220125; device route 7.3168369; prover 115.648; overall 121.00890; 9,994 checks, zero failures; 311 artifacts retained |
| 2026-08-29 | accepted clean control v2 | ;  | Phase 2 10.1686; route 9.4649476; device route 7.2988975; prover 116.164; overall 121.88687; 9,994 checks, zero failures; 311 artifacts retained |
| 2026-08-29 | accepted clean controls | Complete path, length, and SHA-256 manifest comparison | 311 files on each run; zero differences; 0.69 percent Phase 2 spread; final strict target fixed at 4.73189 seconds |
| 2026-08-29 |  and derived CSV reports | Complete `main.py` process-tree CUDA attribution | Kernels 6.8771104 seconds; host-to-device 0.4055674; device-to-host 0.0118627; memset 0.0087664; device-to-device 0.0002774; dominant kernels and launches recorded above |
| 2026-08-29 |  | Full metrics for the live 311,218-node, 64-thread cooperative launch | 65.59% occupancy; 46.73% issue slots; 10.83 active threads per warp; 50.7% long-scoreboard share; 99.98% L2; 2.28% DRAM |
| 2026-08-29 |  | Full metrics for the largest matched filter-count launch | 83.27% occupancy but 3.18% issue slots; 80.92% local-memory L1 sectors; 41.4% memory-queue and 31.2% barrier stall shares |
| 2026-08-29 | ;  | Full metrics and native resource report for dominant kernels | Short growth: 40.40% occupancy, 3.75 active threads per warp, 44.7% long-scoreboard share; native stacks and registers recorded above |
| 2026-08-29 | | M0/M1 evidence and explicitly approved architecture checkpoint | Complete plan committed and pushed; generated vcpkg deltas removed with approval; tracked worktree clean |
| 2026-08-29 | observation census | ;  | Clean Release rebuild passed; 1,559 of 1,559 native unit tests passed, including exact five-field class distinction and the existing six-node/19-candidate span fixture |
| 2026-08-29 | full CUDA census | ;  | 48 passes; 24,229 calls to 8,853 exact classes; 90.874% repeated rows; 18,940,682 nodes; 4,498,333,440 candidates; 9,994 checks, zero failures; 311 artifacts exactly match M0 |
| 2026-08-29 | per-gate census source | ;  | Clean Release rebuild passed; 1,559 of 1,559 native tests passed; direct CUDA fixture reconciles 19 gate attempts to 19 span candidates and checks every survivor relation |
| 2026-08-29 | full survivor census | ; aggregate, peak, and 311-file manifest CSVs | 4,498,333,440 attempts reconcile exactly; 2,082,812,196 survive cheap gates; 19,591,285 subkeys present; 18,807,750 owners pass; 9,994 checks, zero failures; 311 artifacts exactly match M0 |
| 2026-08-29 | rejected filter-class production v1 | ; ;  | Clean build and 1,559 tests passed; correctness remained exact, but runtime-disabled census state inflated hot growth resources and Phase 2 regressed to 12.4457 seconds; result rejected |
| 2026-08-29 | filter-class production v2 | ; ; ;  | Clean rebuild; 1,559 of 1,559 tests; Phase 2 9.40098; device route 6.5884318; 9,994 checks, zero failures; 311 artifacts exactly match M0 |
| 2026-08-29 | post-filter Systems profile | ; four derived stats CSVs;  | Filter count plus emit 0.236058484 seconds, 7.683 times faster than M0; cooperative plus short growth 5.755711392 seconds and 90.13 percent of all 6.386228408 kernel seconds; fixed growth route remains the target |
| 2026-08-29 | prefix-pool census source and activation | ; ; ;  | Two clean Release rebuilds passed, including the live observation activation; both native gates passed all 1,559 tests; direct mode-2 CUDA fixture populates all three prefix-pool counters without changing its exact semantic counts |
| 2026-08-29 | complete prefix-pool census | ; ;  | Payload 14,074,478; normalization variables 4,369,164; secondary variables 951,515; Phase 2 11.9766 observation-only seconds; 9,994 checks, zero failures; all 311 artifacts exactly match M0 |
| 2026-08-29 | production-mode restoration | ;  | Live census selector returned from mode 2 to mode zero; clean Release rebuild passed; all 1,559 native tests passed; separately compiled and directly tested observation specialization retained |
| 2026-08-29 | pooled-prefix source before stable selection | ; ;  | Clean rebuild and all 1,559 tests passed; 40 reported GPU passes matched accepted filter semantics exactly, then downstream owner removal asserted after atomic-arrival work-list compaction changed physical traversal |
| 2026-08-29 | stable pooled-prefix production route | ; ; ; ; ;  | Full route passed, then the final documented tree passed another clean rebuild and all 1,559 tests; Phase 2 9.42072; device route 6.6169076; exact census maxima; 9,994 checks, zero failures; all 311 artifacts exactly match M0 |
| 2026-08-29 | deterministic candidate-window production route | ; ; ;  | Four-record direct twin and all 1,559 tests pass; Phase 2 8.54877; route 7.8623967; device work 5.72042443; 9,994 checks, zero failures; all 311 artifacts exactly match M0; fixed device ownership 1,275,332,748 bytes |
| 2026-08-29 | post-window Systems profile | ; four derived stats CSVs | Kernels 4.961267042 seconds; survivor probes 1.768231566; cooperative cheap gates 1.305358563; prefix preparation 0.680364406; short cheap gates 0.344624045; four growth kernels own 82.61 percent |

## Failure ledger

| Date | Failure | Resolution or next action |
|---|---|---|
| 2026-08-29 | Initial sandboxed worktree creation could not lock `refs/heads/sandbox/gpu_rt2` | Retried the same requested main-tip worktree creation with repository metadata write access; worktree created |
| 2026-08-29 | The ignored worktree-local vcpkg tree lacked `mimalloc.dll.lib`; the sandboxed canonical restore could not reach the package registry | Ran the same canonical `vcpkg install --triplet x64-windows` with network access; mimalloc 2.2.3#1 restored from the local binary cache |
| 2026-08-29 | Build attempts v1-v4 exposed an empty CUDA property, duplicate case-insensitive path variables, and an inaccessible inherited `C:\Windows\TEMP` in clean children | Used one clean `System.Diagnostics.ProcessStartInfo` child with one `Path`, explicit worktree `TEMP`/`TMP`, CUDA 13.3, and full `-t:Rebuild`; v5 passed |
| 2026-08-29 | First completed CUDA control ran while the maintainer reported the computer was occupied; its confirmation had already started | Rejected the 11.6578429-second timing, stopped only that exact confirmation prover process, retained both logs, waited for the parent to exit, then ran two new controls from a confirmed idle GPU |
| 2026-08-29 | First clean artifact preservation attempted the obsolete `files/full_proof_graph` location and created an empty ignored destination | Located the live descriptor tree at `files/shortcut`, copied its four retained subtrees into the same destination, and generated the complete 311-row manifest |
| 2026-08-29 | First explicit Systems-stat extraction rejected an SQLite export a fraction older than its report; automatic statistics also printed a non-fatal invalid-UTF-8 protobuf warning | Regenerated the derived SQLite from the immutable `.nsys-rep` with `--force-export=true`; all four requested CSV reports processed successfully |
| 2026-08-29 | Nsight Compute could not deploy optional section copies under the read-only Documents path | The profiler used its installed stock sections and wrote all requested reports successfully; no tool or metric was substituted |
| 2026-08-29 | Canonical vcpkg restore modified seven tracked package-cache outputs while restoring the ignored missing import library | Maintainer explicitly approved discarding only those seven generated deltas; restored them from `HEAD`, retained `vcpkg_installed/x64-windows/lib/mimalloc.dll.lib`, and left the plan as the only tracked modification |
| 2026-08-29 | The first post-checkpoint upstream-status command left PowerShell's `@{u}` revision unquoted, so PowerShell parsed it as a hashtable | Re-ran the read-only status command with `'@{u}'`; local and upstream both resolved to |
| 2026-08-29 | Census rebuild v1 received an empty CUDA toolkit property; v2 omitted the separator required before `bin`; v3 inherited duplicate `PATH` and `Path` entries | Retained all three failed logs, then used the validated clean child environment with one `Path`, explicit worktree temporary storage, and `CudaToolkitDir` ending in `\`;  passed |
| 2026-08-29 | Sandboxed `git add -A` could not create the worktree `index.lock`; the immediately following commit therefore also had no staged changes | Re-ran `git add -A` with repository-metadata access, committed, and pushed it without changing the verified tree |
| 2026-08-29 | First per-gate unit run stopped because the fixed-allocation test still expected the pre-census counter size | The new record is exactly 864 bytes: nine depths times 96 bytes. Updated the expected growth-buffer total from 267,419,864 to 267,420,728 bytes, rebuilt cleanly, and all 1,559 tests passed |
| 2026-08-29 | First production class-sharing run regressed Phase 2 to 12.4457 seconds despite reducing retained rows to 827,746 | Runtime `collectSpanCensus == 0` still left the large gate-counter state in each short thread and cooperative block. Retained the failed evidence and specialized the growth kernels on a compile-time census mode so production instantiates no census state. |
| 2026-08-29 | The first compile-time-disabled census build made the direct gate test fail because counters were absent in every launch | Replaced the one unconditional compile-time constant with explicit `<false>` production/span-only launches and `<true>` full-census launches. The direct mode-2 fixture again reconciles all 19 candidates; production remains resource-free. |
| 2026-08-29 | Nsight Systems profile v1 rejected `--trace=cuda,nvtx,osrt` because the installed Windows 2026.1.3 profiler has no `osrt` trace category | No GL process launched. Retained  and reran with the supported `cuda,nvtx` categories under a new v2 descriptor. |
| 2026-08-29 | The first `cuobjdump` resource command used the Visual Studio intermediate output path, where no executable exists | Retained ; reran against the canonical `GL_Quick_VS\GL_Quick\gl_quick.exe` output as v2 and extracted both native `sm_89` growth specializations. |
| 2026-08-29 | Prefix-census rebuild v1 stopped at the `Phase2GrowthResult` ABI size assertion after three 64-bit host-result maxima were added | Updated the exact assertion from 1,072 to 1,096 bytes. Device ownership is unchanged because live prefix totals reuse the otherwise-unused depth-zero gate-census fields. |
| 2026-08-29 | Prefix-census rebuild v2 compiled CUDA, then `CL.exe` stopped at MSBuild's duplicate case-insensitive `PATH` / `Path` environment collision | Retained the log and reran through the validated clean `ProcessStartInfo` child with one `Path`, explicit worktree `TEMP`/`TMP`, and CUDA 13.3. |
| 2026-08-29 | Pooled-prefix rebuild v1 stopped because CUDA device code could not call the host-only `std::numeric_limits<uint16_t>::max` constant expression | Replaced the device assertion operand with the exact `0xffffu` representation and retained the assertion; clean rebuild v2 passed. |
| 2026-08-29 | The first full pooled-prefix run stopped after iteration 22 at `removeOwnerFromRun: owner is not on key` | All 40 completed GPU pass records matched the accepted filter run on every reported semantic count. The new atomic-arrival short-list compaction was the only changed traversal seam; replaced it with stable CUB selection in original frontier-index order. Full run v2 passed all 9,994 checks and 311 artifact hashes. |
| 2026-08-29 | Final pooled-prefix rebuild v4 found that removal of an unused short-kernel parameter had also removed the preparation kernel's required `terms` parameter | Restored the preparation-only parameter and Doxygen entry; the short kernel remains free of the unused argument. Clean rebuild v5 and all 1,559 unit tests passed. |
| 2026-08-29 | First candidate-window unit run stopped at the exact fixed-allocation assertion inherited from the pooled-prefix owner | Retained the failure; the complete pipeline allocation report established exact growth ownership at 812,965,447 bytes and the final direct test pins that value plus its one-GiB class ceiling. |
| 2026-08-29 | Candidate-window rebuild v2 inherited an empty CUDA toolkit property | Retained the log and returned to the validated clean `ProcessStartInfo` child with one `Path`, worktree `TEMP`/`TMP`, and explicit CUDA 13.3. |
| 2026-08-29 | Removing the survivor probe's unused request-limit parameter also matched the pooled candidate helper and short-gate launch in the first broad patch | Restored both required request-limit arguments, kept the survivor probe parameter-free, and retained the compile failure as ; rebuild v4 passed. |
| 2026-08-29 | The first temporary allocation-range build used unavailable `ASSERT_GT` test syntax | Replaced it with the supported `ASSERT_TRUE`, retained , and clean rebuild v6 plus all 1,559 tests passed. |

## Commit ledger

| Milestone | Commit | Verification | Push |
|---|---|---|---|
| Branch plan | | `git diff --check`; documentation-only snapshot ||
| M0/M1 evidence and approved architecture | | Clean rebuild; 1,558 unit tests; two 9,994-check CUDA runs; 311-file exact manifest comparison; explicit maintainer approval ||
| M2 observation census slice | | Clean rebuild; 1,559 unit tests; 9,994-check CUDA census; 311-file exact identity ||
| M2 per-gate survivor census | | Clean rebuild; 1,559 unit tests; 9,994-check CUDA census; 311-file exact identity ||
| M2 exact filter-class production slice | | Clean rebuild; 1,559 unit tests; 9,994-check CUDA run; 311-file exact identity; complete design-document family ||
| M2 post-filter profile checkpoint | | Nsight Systems kernel/memory attribution; native `sm_89` resource extraction; living plan reconciled with exact filter and growth ownership ||
| M2 prefix-pool census activation | | Two clean rebuilds; two 1,559-test gates; compile-time-isolated mode-2 prefix totals; live activation checkpoint ||
| M2 pooled-prefix production slice | | Clean Release rebuild; 1,559 native tests; 9,994-check CUDA run; exact 311-file identity; complete design-document family ||
| M2 deterministic candidate windows | pending this commit | Clean Release rebuild; 1,559 native tests; 9,994-check CUDA run; exact 311-file identity; post-change Systems profile; complete design-document family | push immediately after commit |

## Progress log

### 2026-08-29 — campaign opened from current main tip

Created the isolated branch and worktree from `main`, after the
exact `gpu_rt1` tree and its documentation identifiers landed. Loaded the full
static-container, string-lifetime, and GPU cookbooks plus the complete prior
campaign ledger. No production source changed.

The new campaign doubles the already-optimized GPU mode. The independent prior
confirmation makes 4.73189 seconds the provisional strict target, subject to a
fresh unchanged-main control. Historical attribution says growth and filtering
jointly own about 6.46 seconds before the staged-upload final change, while host
projection owns about 1.34 seconds. That evidence is sufficient to define the
profile, not to choose a production redesign. M0 now builds, validates, retains
the fresh CUDA control, and profiles the exact landed route before the Rule-8
architecture checkpoint.

### 2026-08-29 — M0 clean CUDA control established

Restored the missing worktree-local mimalloc package through the canonical
vcpkg command, then completed a clean Release rebuild and all 1,558 native unit
tests. The first completed CUDA run was excluded from performance evidence when
the maintainer reported that the computer had been occupied, and its in-flight
confirmation was stopped without changing source or tracked artifacts.

Two new controls began from an idle P8 GPU at 0 percent utilization. They
measured 10.0993 and 10.1686 seconds for complete Phase 2, a 0.69 percent spread,
and each passed all 9,994 checks. Their separately retained 311-file manifests
are byte-identical. The fresh two-times thresholds are both weaker than the
inherited independent-best threshold, so final acceptance remains 4.73189
seconds. M0 now profiles this exact landed staged-upload route; no production
architecture direction has been selected.

### 2026-08-29 — M0 attribution complete, M1 approval required

Systems accounts for 99.82 percent of the 7.3168369-second device route with
6.8771104 seconds of kernels and 0.4264739 seconds of memory operations. Four
kernels own 6.5537905 seconds, while every other kernel together owns only
0.3233198 seconds. Their required 5.52-times combined speedup rules out isolated
launch tuning, transfer work, projection-only work, filter-only work, and either
growth kernel alone.

Compute shows three complementary structural costs: branch and dependency
latency in cooperative growth, repeated dynamically indexed normalized-key
materialization plus barrier imbalance in filtering, and inactive lanes in the
full-frontier short launch. The proposed bounded signature-and-candidate
pipeline addresses all three measured seams while retaining GPU decisions,
first-8,192 filter order, canonical event reconstruction, fixed ownership, and
assert-only failure. M1 reached the explicit architecture-approval checkpoint.

### 2026-08-29 — architecture and dependency cleanup approved

The maintainer explicitly approved both the bounded signature-and-candidate
pipeline and discarding only the seven generated vcpkg package-cache deltas.
The seven tracked files now match `HEAD`; the restored ignored mimalloc import
library remains available for clean rebuilds. Production work may proceed with
the observation-only capacity/reuse census defined by the approved design.

### 2026-08-29 — first M2 observation slice unit-verified

Added an observation-only exact filter-class census over the complete approved
signature tuple: logical block, memory kind, iteration ceiling, whole-key
widening, and statement count. The allocation-free scan reports unique classes,
class rows, duplicate calls and rows, and maximum multiplicity, but neither
rewrites the filter schedule nor feeds any proof decision. A direct host test
proves that one exact duplicate merges while a change to each individual tuple
field does not.

Enabled the existing observation-only growth-span census for the forthcoming
sizing run. Its direct CUDA fixture already proves six nodes and nineteen
candidate attempts in the expected two span buckets. The clean Release rebuild
and all 1,559 native tests pass. The agent SwDD, user SwDD, and MPU diagrams need
no update for this observation-only, route-neutral slice; production routing
will carry the complete design-family update. The next run measures class reuse
and frontier spans, after which per-gate survivor counters complete the approved
pre-routing census.

The exact observation slice is committed as and. Its full-pipeline census is the active next action.

### 2026-08-29 — live class reuse and frontier sizes validated

The complete observation run proves the approved reuse seam is substantial,
not hypothetical: exact class representatives remove 63.461 percent of calls
and 90.874 percent of examined rows, with maximum multiplicity 32. The growth
shape also exactly reproduces the prior 18,940,682-node and
4,498,333,440-candidate census, giving a stable cross-campaign input size.

Correctness is unchanged: all 9,994 verifier checks pass and all 311 retained
artifacts match the accepted M0 control byte for byte. Per-gate survivors by
frontier depth remain required before fixed candidate-window capacities can be
approved for production routing, so that observation-only counter extension is
the active next action.

### 2026-08-29 — per-gate survivor counters unit-verified

Added twelve observation fields at each candidate depth: attempts, mandatory
reachability, validity comparability, hypothesis compatibility, secondary-
variable compatibility, key-length acceptance, subkey presence, owner
satisfaction, whole-key presence, mandatory-term satisfaction, accepted events,
and emitted children. Short-node threads aggregate locally before fixed counter
atomics; cooperative blocks aggregate in shared memory before one fixed counter
update per nonzero field. The counters are written only when the existing
observation switch is active and no semantic code reads them.

The fixed counter owner grows by exactly 864 bytes and remains allocated once.
A clean Release rebuild and all 1,559 native tests pass. The existing direct
CUDA fixture independently counts nineteen candidates, reconciles that total
against all depth-attempt counters, and checks every cumulative and branch
relationship. This route-neutral instrumentation does not change any agent
SwDD, user SwDD, or MPU diagram; the complete production redesign will update
the design-document family after the full live survivor census fixes capacities.

The per-gate source is committed as and; its full CUDA census is now running next.

### 2026-08-29 — approved pre-routing census complete

The full survivor run reconciles all 4,498,333,440 candidate attempts against
the independent span census and preserves all 9,994 checks plus all 311 artifact
hashes. Shape gates reduce the key-probe stream to 2,082,812,196 candidates;
19,591,285 find a subkey and 18,807,750 pass owner signatures. Depths three and
four dominate both total work and single-pass peaks.

The 90,873,097- and 66,029,213-candidate post-shape peaks prove that no complete
survivor array fits the fixed ownership contract. The approved deterministic
window pipeline is therefore required rather than optional. Exact filter-class
reuse is the first production slice because it removes 90.874 percent of
repeated filter rows while preserving the original call ordinals that growth
ordering consumes.

### 2026-08-29 — exact filter-class production slice retained

Implemented the approved class boundary with fixed host and CUDA ownership. The
first exact run exposed that a runtime-disabled observation branch still shaped
the compiled hot growth kernels; its 12.4457-second regression is retained. A
compile-time removal then broke the direct census fixture, so the fix-forward
route specializes production/span-only and full-census launches separately.
This preserves the observation instrument while compiling its per-thread and
shared state entirely out of production.

The clean rebuild and all 1,559 native tests pass. The confirmed idle CUDA run
measures 9.40098 seconds complete Phase 2, 6.5884318 seconds in the device route,
113.374 seconds in the prover, and 118.80591 seconds overall. All 9,994 verifier
checks pass and every one of 311 retained artifact hashes matches M0. Exact
class sharing therefore removes 0.69832 seconds of complete Phase 2 without
changing the original-call search or output. The agent SwDD, user SwDD diagram,
and MPU control-word diagram now record the live class contract and corrected
729,788,029-byte fixed owner. The complete slice is commit,. A new complete device profile is the next action.

### 2026-08-29 — post-filter profile fixes growth as the remainder

The supported CUDA/NVTX Systems run completed through `main.py` and produced a
fresh native report. Exact class sharing works at the intended scale: filter
count and emit are 7.683 times faster and their combined time falls by
1.577499779 seconds. The complete kernel stream falls by only 0.490881992 seconds
because cooperative plus short growth rises by 1.015479136 seconds and now owns
90.13 percent of all kernel time.

Native resource extraction shows the production cooperative image at 59
registers versus M0's 53 and the short image at 63 versus 62, with stack and
shared sizes otherwise unchanged. That resource delta and the Systems times are
the established boundary; no semantic or scheduling guess is used. The next
source slice implements the already-approved fixed per-node prefix, compact
short/long node indices, and deterministic candidate windows, with direct twins
and fixed capacities before another complete run.

### 2026-08-29 — prefix-pool capacity census completed in the direct twin

Extended the existing observation specialization with per-frontier normalized
payload, normalization-variable, and secondary-variable totals. The counters
reuse depth zero, which cannot represent an appended premise and was otherwise
unused, so fixed device ownership stays unchanged. A clean Release rebuild and
all 1,559 native tests pass; the direct CUDA fixture proves all three totals are
nonzero while every prior exact count remains fixed.

Mode 2 is now explicitly active for one complete CUDA sizing run. A second clean
Release rebuild and all 1,559 native tests pass with that live activation. Its
maxima will set the fixed prefix-pool capacities; the next production build
returns to mode zero while retaining the separately compiled observation
specialization.

### 2026-08-29 — live prefix-pool capacities fixed

The complete observation pass produced 48 GPU records and found independent
simultaneous-frontier maxima of 14,074,478 payload values, 4,369,164
normalization-variable values, and 951,515 secondary-variable values. The
production capacities apply 25 percent headroom and round each pool upward to a
262,144-value block: 17,825,792, 5,505,024, and 1,310,720 values respectively,
for 94 MiB total. Every boundary will assert; there is no alternate allocation
or processor replay.

This observation run passed all 9,994 verifier checks and its 311 retained
artifact hashes exactly match accepted M0. Its 11.9766-second Phase 2 time is
deliberately excluded from performance evidence because mode 2 compiled the
full census into the growth kernels. The next source checkpoint returns the
live flag to production mode zero while retaining the tested observation code.

### 2026-08-29 — production mode restored after capacity census

Returned the live selector from mode 2 to mode zero without removing the
observation specialization, result telemetry, or direct mode-2 test. A clean
Release rebuild passes and all 1,559 native tests pass. The branch is now back
on its production resource image with the three exact fixed capacities recorded
above; implementation proceeds only along the approved pooled-prefix,
compact-index, and deterministic-window route.

### 2026-08-29 — pooled-prefix and stable-index production slice exact

Added one prefix header per live node and the three measured fixed value pools.
A preparation kernel performs the full path normalization and summary work once;
the short and cooperative kernels now consume those immutable slices and build
only each candidate's expression suffix. The exact production owner rises to
405,840,199 bytes and total application device ownership to 868,207,500 bytes.

The first clean build exposed one CUDA-incompatible host constant expression and
was fixed with its exact unsigned 16-bit representation. The first full run then
proved that atomic-arrival short/long compaction was not an acceptable physical
traversal change: all 40 completed GPU pass rows remained semantically identical,
but a later processor owner-removal assertion fired. Stable CUB selection over
the original frontier-index sequence fixed forward without reverting the pooled
representation.

The stable implementation passes a clean Release rebuild and all 1,559 native
tests. Its full CUDA route passes all 9,994 verifier checks, reproduces the three
prefix census maxima exactly, and matches every one of 311 M0 artifacts. Complete
Phase 2 is 9.42072 seconds, only 0.01974 seconds above the accepted filter route.
This is retained as the exact representation boundary required by deterministic
candidate windows; no independent acceleration is claimed. The agent SwDD, user
SwDD diagram, and MPU diagram now describe the pooled prefix, stable selection,
fixed capacities, and 868,207,500-byte total. The next source slice is the
approved deterministic bounded candidate-window pipeline. The final documented
tree passes  and all 1,559 tests in
. The complete milestone is commit
,.

### 2026-08-29 — deterministic candidate windows exact and profiled

Added stable call/run/path node ordering, exact scanned candidate spans, and
consecutive fixed windows capped at 16,777,216 attempts. Separate short-thread
and cooperative cheap-gate kernels fill every in-window ordinal exactly once;
stable selection compacts survivors before suffix-only key, owner, event, and
child work. The direct CUDA twin uses a four-record window and crosses both of
its productive depths several times. It matches the observation specialization,
and all 1,559 native tests pass.

The complete production run passes all 9,994 verifier checks and its 311-row
manifest exactly matches M0. Phase 2 falls from 9.42072 to 8.54877 seconds while
prover and overall time improve to 113.294 and 118.37951 seconds. Growth owns
812,965,447 fixed bytes and the full CUDA route owns 1,275,332,748 bytes; no
window allocates, resizes, truncates, falls back, or replays processor work.

The immediate Systems profile replaces the pre-window attribution. Survivor
probes, cooperative cheap gates, prefix preparation, and short cheap gates own
4.098578580 of 4.961267042 kernel seconds. The milestone is therefore retained
as an exact 9.256-percent improvement, but it does not satisfy the 4.73189-second
goal. M3 must close 3.81688 seconds inside this newly measured growth family.

### 2026-08-29 — accepted candidate-window commit rerun before main landing

The maintainer selected commit as the accepted optimization and
requested one fresh run before squashing it to `main`. The dedicated worktree
was placed on that exact commit. The first clean rebuild invocation omitted the
directory separator after the explicit CUDA 13.3 root and therefore formed
`v13.3bin`; it stopped before compilation and changed no source. The corrected
full rebuild passed in
, followed by all 1,559
native tests in .

The shortcut-only CUDA run in
 passes all 9,994 checks with
zero failures. Its 311-row path, length, and SHA-256 manifest at
 has zero differences
from M0. Exponent-safe sums over 45 Phase 2 iterations give 8.8462047 seconds
complete Phase 2, preparation 1.9256776, projection 1.2833057, scheduling
0.6423715, device work 5.94762836, device route 5.9754134, finalization
0.2375394, and enclosing route 8.1386283 seconds. Prover time is 114.337 seconds
and overall shortcut time is 119.41707 seconds.

The fresh measurement remains 0.5745153 seconds below the 9.42072-second pooled
prefix parent checkpoint, so the candidate-window optimization again produces a
measurable complete-Phase-2 gain. It is 0.2974347 seconds above the original
8.54877-second candidate-window observation; no cause is assigned to cross-run
movement without a trace. The accepted source is preserved. The merge-time user-SwDD sweep also copied the
existing CUDA filter-class navigation entry to all shared sidebars; all 11 pages
now validate with 196 identifiers and zero warnings.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
