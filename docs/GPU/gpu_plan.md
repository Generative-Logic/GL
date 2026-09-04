<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Phase 2 GPU port plan

> **Purpose:** this is the persistent working memory for the Phase 2 GPU port.
> Read it before every work session and after every context compaction. Update it
> whenever a milestone, assumption, decision, measurement, failure, commit, or
> next action changes. Conversation memory is never the authoritative state.

## Current state

- **Branch:** 
- **Worktree:** 
- **Base:** `main`
- **Implementation:** native `sm_89` and `sm_120`, the launch contract, the complete 24-column
 resident Phase 2 semantic image, and exact device probes for its name, byte,
 reverse, and eleven plain-data tables are live. The pointer-free executor-task,
 request-batch, mandatory-term, and expression-stump image now has measured
 FTA ceilings plus fixed host and CUDA ownership with a byte-exact upload twin.
 The global bulk filter/scan/sort stage is also live and directly identical to
 processor output across all four rule registries. The retained FTA growth
 census now fixes the frontier, accepted-event, and raw-request capacities. A
 native CUDA twin builds exact multi-premise normalized keys and applies both
 whole-key presence and owner-subkey u_ signature semantics. The measured
 request-growth path now runs in bulk on CUDA through exact mandatory masks,
 validity/request gates, stump seeds, normalized map probes, and two global
 frontiers. Exact processor stack order, task-local cumulative growth positions,
 and collision-checked first-occurrence request deduplication now run on CUDA.
 The complete growth owner is 263,225,360 bytes (about 251.031 MiB),
 including call links, direct/suffix masks, frontiers, the accepted-event
 ledger with compact filtered-position codes, raw requests, exact per-task
 subkey-work counters, and shared counters. A
 second fixed ordering owner carries the radix, segmented-scan, and semantic
 deduplication arrays plus one maximum-sized reusable CUB scratch allocation.
 The retained processor evaluation census now fixes the next device stage:
 72,674 requests, 1,049,978 reverse owners, 47,930 subset candidates, 46,654
 encoded hits, 82,620 local values, and 62,834 complete firing records at the
 sweep peaks. Generated output peaks at 1,935,835 bytes, 189,815 level values,
 and 184,494 provenance dependencies. `CudaPhase2EvaluationBuffer` now owns
 every retained work/output array plus one reusable scan scratch allocation;
 its deterministic base is 37,267,500 bytes and the complete production owner
 is 37,269,035 bytes, remaining below 64 MiB. The device work path is live through dependency and
 validity gates, reverse expansion, subset compaction, mapped encoded lookup,
 and `LocalMemoryValue` expansion. A second kernel now reconstructs each selected
 value's exact reverse substitution, streams greedy-longest replacement plus
 repeated token-leading `u_` removal without dynamic or intermediate storage,
 repeats the rule/deposit validity filter, applies marker/demand purity and
 ordis2 depth gates, and writes byte-exact expressions plus LMV source flags to
 the fixed firing arenas. Surviving heads and demands also receive the exact
 sorted-unique premise/rule level union; heads receive the source implication
 followed by request premises in decoded expression/validity order, entirely in
 the fixed provenance arena. Projected names and rule strings carry the compact
 absent/atomic/non-atomic compiled-core distinction; each logical block also
 carries its anchor name, level, evaluator switches, secondary-number ceiling,
 and both mail-eligibility sets. The evaluator now computes exact disintegration,
 mail, ancestor-known, OR-disintegration, and marker-category verdicts. Marker
 records carry substituted keys, decoded-byte-sorted unique remaining arguments,
 bare arguments, and admission ceilings in fixed device arenas.
 Two fixed firing-index arrays reproduce canonical content order. A fixed
 512-entry doom-line and triggering-request arrays then mirror sole-goal,
 contradiction, recursion, counter-example, and compressor verdicts on projected
 state; a stable scan retains only the winning part through its inclusive first
 triggering request, including when later requests share the growth position.
 `Phase2SealingArena` now downloads those used prefixes through fixed processor
 arrays, decodes the same projected NameMap/rule-interner identifiers, and seals
 complete canonical `FiringRecord` chains without semantic replay. The existing
 processor producer still requests canonical sorting; the GPU handoff explicitly
 bypasses that duplicate sort only for device-ordered records. Production routing
 is live behind `--phase2-backend cpu|cuda`, defaults to the processor, rejects
 CUDA outside shortcut mode, and never falls back. The CUDA route retains the
 processor stump producer for split-work discovery, then projects the exact
 read-only logical-block state and executes filtering, growth, ordering,
 evaluation, provenance, doom selection, and firing-record materialization on
 the device before host-only sealing and the shared deposit door. The first
 complete CUDA shortcut executed 48 device passes, passed 10,038 verifier checks
 with zero failures, and produced all 311 artifacts byte-identically to the
 retained processor baseline. The first paired timing gate measured 110.333
 processor seconds versus 228.782 CUDA seconds for complete Phase 2. Nsight
 Systems attributes 217.594 seconds, 99.0% of all CUDA kernel time, to the
 request-frontier growth expansion kernel; host-to-device transfers total only
 0.330 seconds. Hoisting the processor-equivalent normalized-key and request-gate
 prefix summaries out of that kernel's per-candidate loop reduced exact CUDA
 time from 220.274 to 71.034 seconds and complete Phase 2 from 228.782 to 78.845
 seconds. Against the paired processor baseline, CUDA Phase 2 is now 1.399 times
 faster and the complete shortcut is 1.137 times faster. The counter-guided
 local-state reduction then lowered the compiled stack from 5,024 to 3,184 bytes
 and crossed the 64-register residency threshold. Exact CUDA-owned time fell to
 15.929 seconds, complete Phase 2 to 25.178 seconds, and the shortcut to 161.911
 seconds. Against the paired processor run this is now a 4.382-times Phase 2 and
 1.349-times end-to-end acceleration. All 305 current proof artifacts remain
 byte-identical. Post-optimization profiling assigns 13.005 seconds and 87.0%
 of kernel time to frontier growth; the representative launch reaches 54.35%
 occupancy and 46.49% compute throughput while using only 1.96% DRAM
 throughput. Exact route attribution reconciles within 3 microseconds: 8.270
 seconds of host projection/scheduling, 15.684 seconds of CUDA-segment wall
 time containing a 15.672-second event interval, and 0.283 seconds of download
 and sealing. A further 0.808 seconds lies in the complete Phase 2 bracket after
 the CUDA route. The six process-owned CUDA buffers now report their exact
 allocation requests: projection 299,951,120 bytes, tasks 731,144 bytes,
 filtering 25,512,447 bytes, growth 263,225,360 bytes, ordering 98,838,019
 bytes, and evaluation 37,269,035 bytes. Their complete fixed ownership is
 725,527,125 bytes (691.92 MiB, 0.676 GiB, 8.45% of the 8,188 MiB device),
 excluding only driver and CUDA-context memory. The measured live peaks remain
 far below the fixed output ceilings: 50,732 firing records and 154,845
 provenance dependencies. Provenance therefore occupies 1,238,760 bytes at
 the peak (1.181 MiB, 59.07% of its 2 MiB arena). Mail eligibility is a flag on
 the already-required 96-byte firing record and creates no second GPU mail
 queue or payload arena. The refined route run split preparation into 6.905
 seconds of logical-block projection and 0.073 seconds of scheduling. Projection
 is 98.95% of preparation and 30.09% of the measured 22.948-second complete
 Phase 2; scheduling is closed as immaterial. That run is 4.808 times faster
 than the retained 110.333-second processor Phase 2 baseline and retains all
 305 current artifacts byte-identically with 10,038 verifier checks and zero
 failures. Internal projection attribution then assigned 5.112 seconds to names,
 including 4.304 seconds of decoded lexical sorting. Reusing the already-packed
 name bytes for that exact sort reduced sorting to 1.936 seconds, projection to
 4.308 seconds, and complete Phase 2 to 19.777 seconds. This is a 5.579-times
 acceleration over the retained 110.333-second processor Phase 2 baseline; all
 305 artifacts remain byte-identical and 10,038 checks still pass with zero
 failures. A fixed-capacity most-significant-byte radix sort with common-prefix
 collapse reduced sorting again to 1.545 seconds, projection to 3.987 seconds,
 and complete Phase 2 to 19.257 seconds. That pre-main-merge Phase 2 was 5.730
 times faster than its retained processor baseline and 11.880 times faster than
 the first complete CUDA route. After merging current main and resizing only the
 two newly exceeded fixed capacities, the final same-tip landing pair measures
 96.212 processor seconds and 19.835 CUDA seconds for complete Phase 2: a 4.851
 times acceleration and 76.377 seconds saved. The CUDA shortcut completes in
 127.555 seconds versus 203.225 seconds for the processor, a 1.593-times end-to-end
 acceleration. Its exact device interval is 15.035 seconds; all 305 semantic
 artifacts remain byte-identical and both runs pass 9,994 checks with zero
 failures. Against the retained 19.257-second pre-merge CUDA result, exact device
 time increased by 0.703 seconds while projection decreased by 0.179 seconds;
 the enclosing route increased by 0.526 seconds and work outside it by 0.052
 seconds, producing the 0.578-second complete-Phase-2 delta. The FTA shortcut's
 at-most-two passes project disjoint
 non-straggler and straggler sets inside each fresh prover subprocess, so an
 in-process projection cache has no reuse hits on this target.
- **Status:** complete and landed on `main`. The exact GPU squash is,
 the separate SwDD allocation is, the source branch is preserved,
 and no process is running.
- **Active milestone:** none; the approved FTA shortcut GPU port and landing gates
 are complete.
- **Next action:** broader GL modes remain a separately approved future port; no
 first-port or landing work is open.

## Objective

Port the FTA shortcut's complete Phase 2 hashburst to the NVIDIA GPU on Windows
while retaining the existing processor backend, full provenance, deterministic
doom-line and firing semantics, canonical firing-record replay, proof artifacts,
and verifier results. The first successful end-to-end target is
`main.py --shortcut`; broader GL modes are out of the first landing gate.

## User-approved boundaries

1. The existing processor Phase 2 remains available in parallel with the GPU
 implementation during the initial port.
2. A configuration flag explicitly selects processor or GPU Phase 2 so the two
 paths can be compared on the same source tree.
3. There is no automatic fallback. A selected GPU backend must either satisfy
 its contract or assert at the failing boundary.
4. Full provenance stays enabled. GPU mode must produce the same proof history,
 not a theorem-only result.
5. The first pipeline target is FTA shortcut mode through `main.py --shortcut`,
 never a direct `gl_quick.exe` pipeline run.
6. Processor and GPU runs must be byte-identical at the proof-artifact boundary
 and must pass the same verifier checks.
7. The agentic SwDD is the implementation memory. Add a GPU cookbook with the
 same practical purpose and structure as the statification cookbooks, and keep
 the relevant agentic SwDD contracts current throughout the port.
8. Do not create a separate user-SwDD GPU chapter. Make only narrow required
 consistency corrections if an existing shipped page or diagram becomes
 factually false.
9. Maintain this file throughout the port so it survives context compactions.

## Project rules that shape the port

- No production architecture is written before the exact device layout,
 scheduling, ownership, backend flag, and provenance boundary receive user
 approval.
- Every new production function carries full GL-style Doxygen documentation and
 a direct unit test in the same commit.
- Unexpected device, allocation, queue, transfer, launch, or identity failure is
 an assertion, never a fallback to the processor path.
- Existing source or experimental work is never reverted without explicit user
 consent. Fix forward.
- Any source or configuration milestone is committed with `git add -A`, a
 detailed commit message, status verification, and an immediate branch push.
- Every build is a full Windows MSBuild rebuild. Pipeline work always runs
 through `main.py`; only `gl_quick.exe --unit-tests` may run directly.
- Hashburst dump infrastructure remains untouched except for an explicitly
 retargeted full logical-block parent chain when a trap is needed.
- Hardware-relevant architecture changes update the appropriate agentic SwDD
 chapters in the same commit. New decision, invariant, and gotcha identifiers
 on this sandbox branch use pending slugs.

## Confirmed environment

Audit date: 2026-08-27.

| Component | Confirmed state |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9 |
| Video memory | 8,188 MiB |
| Driver | NVIDIA 596.08; reports CUDA 13.2 driver compatibility |
| Device power ceiling | 140 W maximum, 135 W current configured ceiling |
| Host link | PCI Express generation 4 by 8 lanes maximum |
| CUDA Toolkit | 13.3, compiler build 13.3.73 |
| CUDA compiler | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.exe` |
| CUDA runtime | Headers plus dynamic and static runtime libraries present |
| Visual Studio integration | CUDA 13.3 build customizations installed for Visual Studio 2022 and Visual Studio 18 |
| Native build | Visual Studio 2022 MSBuild 17.14.23 present |
| Profiling | NVIDIA Nsight Compute 2026.2.1 and Nsight Systems 2026.1.3 installations present |
| Runtime launch | A Toolkit 13.3 kernel compiled as native `sm_89` code launched and returned the expected value |

The Codex parent process started before Toolkit installation, so its inherited
`PATH` and `CUDA_PATH` remain stale. Use the confirmed absolute Toolkit path in
commands until a fresh process exposes the installed environment. This is an
environment fact, not a repository workaround.

The driver exposes CUDA runtime level 13.2. Default Toolkit 13.3 PTX failed at
kernel launch as an unsupported toolchain, while the same probe compiled with
`-arch=sm_89` succeeded. The first build integration must emit native compute
capability 8.9 code and may not rely on PTX JIT on this machine.

## Live Phase 2 seams to preserve

These symbols are the initial source map; expand this list in the GPU cookbook
after the mandatory full cookbook reads.

- `prover.cpp::ExpressionAnalyzer::proveKernel`: builds the flat `(logical
 block, part)` executor list, runs the producer/bucket rounds, preserves the
 deterministic doom line, and finalizes each logical block.
- `prover.cpp::ExpressionAnalyzer::performElem2`: executes all request batches,
 reads logical-block state, and writes only the task's sealed record pages plus
 the external doom line.
- `memory.cpp::ExpressionAnalyzer::produceExpressionStumps`: emits the exact
 expression-search frontier used to split a straggler.
- `memory.cpp::ExpressionAnalyzer::generateEncodedRequestsStatic`: filters and
 sorts the statement universe, grows exact request candidates, deduplicates
 requests, and streams them to the consumer.
- `prover.hpp::BurstSink`: performs dependency membership checks, invokes
 request evaluation, and publishes deterministic early-exit positions.
- `memory.cpp::ExpressionAnalyzer::checkLocalEncodedMemoryStatic`: performs
 integer gates, encoded-map lookup, exact substitution and rule-specific
 evaluation, and captures complete provenance-bearing `FiringRecord` values.
- `memory.cpp::ExpressionAnalyzer::applyFiringRecords`: canonically sorts and
 applies the record set after the executor join.
- `memory.hpp::FiringRecord`: the existing semantic and provenance output
 contract. GPU mode must not weaken it.

Confirmed data facts:

- request-generation input is already dominated by flat 32-bit records:
 `IntEncodedExpr` is a 352-byte trivially-copyable record with eight expressions
 maximum per request and sixteen arguments maximum per expression;
- host `StaticRequest`, `IntNormalizedKey`, `IntStmtView`, and `SplitStumpRef`
 contain pointers or host-container views and therefore require offset-based
 device twins;
- the generator performs an exact filtered, decoded-name-stable-sorted search,
 uses a stack order that is not interchangeable with generic breadth-first or
 depth-first order, counts accepted subkeys, and keeps the first occurrence of
 each emitted request per generator call;
- evaluation reads `NameMap`, the rule interner, validity ancestry and filters,
 statement membership and levels, recursion-product identifiers, the remaining
 argument reverse index, normalized keys and owner blobs, encoded-map
 `LocalMemoryValue` blobs, local-statement membership, compiled marker category,
 and a small set of batch flags;
- head, marker, and ordis2-demand outputs contain generated strings, sorted
 integer runs, identifiers, booleans, and complete provenance; processor-side
 semantic replay would leave request evaluation unported;
- `applyFiringRecords` already supplies the correct single-threaded canonical
 deposit seam after parallel producers and should remain the initial host
 finalize boundary.

The detailed host-to-device type table and pitfalls are in
`docs/agentic_swdd/20_core_concepts/09d_gpu_cookbook.md`.

## Architecture questions requiring user approval

The user approved every proposal in this table on 2026-08-27 and asked the port
to proceed without further architectural micro-approvals. Exact numerical arena
capacities remain evidence-driven under the approved capacity policy.

| Question | Current state |
|---|---|
| Exact configuration key and values | **Approved:** command-line `--phase2-backend cpu|cuda`, accepted by `main.py` and forwarded to the native batch. Any missing value, duplicate flag, or other value asserts. |
| Default backend during the sandbox port | **Approved:** `cpu` until the complete FTA identity gate passes; selection is reported once per native batch. |
| Device projection ownership and lifetime | **Approved:** one process-owned CUDA engine and preallocated device arenas; batch-immutable tables live for the batch; each logical-block projection carries a stable identity and generation; iteration scratch and output cursors reset only after their synchronization boundary. |
| Host-to-device update contract | **Approved:** compact logical snapshots, never cold-page or full-pool copies; cache projections by logical-block generation and transfer only changed packed tables/ranges after Phase 1. Cache presence affects transfers only, never semantics. |
| Request-generation work grain | **Approved:** global bulk frontiers containing many logical blocks, parts, registry batches, stumps, and frontier nodes. Prefix scans compact gates and recover exact processor growth positions; semantic request dedup keeps the minimum processor-order token. No one-kernel-per-stump design. |
| Request-evaluation boundary | **Approved:** the GPU executes the dependency check and every integer, byte-string, rule, scope, admission, mail, ancestor-known, marker-category, and firing gate through complete semantic output. The processor performs no semantic re-evaluation. |
| Provenance construction location | **Approved:** the GPU emits all provenance content as existing identifiers plus generated byte slices; the processor only decodes identifiers and seals the already-decided content into `SealedPageSet`. |
| Device output and canonical-order key | **Approved:** a fixed `DeviceFiringRecord` carries kind, flags, validity identifier, source and premise identifiers, level-run offsets, generated-string offsets, and processor stream token. A host adapter builds ordinary `FiringRecord`; existing `applyFiringRecords` remains the content-key sort and deposit boundary. |
| Doom-line implementation | **Approved:** first correct version computes the bounded work, finds each part's first deactivating request in exact processor stream order, reduces by `(growth position, part ordinal)`, and retains the winning part's inclusive triggering-request prefix. Cooperative early cancellation is deferred until identity. |
| Queue capacities and allocation source | **Approved policy; task, filter/sort, and request-growth capacities measured:** all production input, double-frontier, request, firing-record, integer-run, and byte arenas come from engine-owned startup allocations with fixed capacities and high-water telemetry. The retained FTA census fixes the input, frontier, accepted-event, and raw-request ceilings; firing-output numbers remain pending. Exact byte counts are derived from retained measurements before the matching allocation lands; overflow asserts and never resizes or falls back. |
| Device availability contract | **Approved:** selected `cuda` mode asserts on no device, wrong compute capability, native-image mismatch, allocation failure, copy failure, launch failure, or synchronization failure. |
| FTA shortcut residency policy | **Approved:** deterministic capacity-bounded task batches with a performance-only logical-block projection cache. A cache miss uploads the exact compact snapshot; insufficient space divides work into more deterministic batches, not processor fallback. |

### Measured FTA projection ceilings

The retained `gpu_capacity_cpu_v1` shortcut run emitted 26 Phase 2 censuses.
Its observed high-water marks were 306 logical blocks, 129,399 statements,
348,066 name records, and 11,811,329 canonical-name bytes in one sweep. The
largest single logical block held 20,486 statements, 67,806 name records, and
2,916,976 canonical-name bytes.

The first four measured columns therefore use the next power-of-two batch
ceilings: 512 logical blocks, 262,144 statements, 524,288 name records, and
16,777,216 canonical-name bytes. Those four arrays occupy about 112 MiB. Their
margins are 1.67, 2.03, 1.51, and 1.42 times the measured sweep peaks
respectively. Every further input class has its own measured ceiling below; it
does not consume an arbitrary remainder of these arrays. Exceeding any batch
ceiling asserts until the approved deterministic batching implementation owns
that boundary.

The corrected retained `gpu_complete_projection_cpu_v1` run measured every
resident semantic-input column, including the name lookup table, the frozen OR
branch filter, and safe upper bounds for the derived remaining-argument reverse
index. Component-wise sweep peaks and fixed first-FTA ceilings are:

| Projection column | Measured peak | Fixed ceiling |
|---|---:|---:|
| Name lookup hash slots | 997,376 | 1,048,576 |
| Rule-string records | 117,807 | 131,072 |
| Rule-string bytes | 7,423,016 | 8,388,608 |
| Byte-map views | 3,060 | 5,120 (10 per logical-block slot) |
| Byte-map entries | 407,031 | 524,288 |
| Byte-map hash slots | 1,051,000 | 2,097,152 |
| Byte-map key bytes | 61,482,816 | 67,108,864 |
| Value-blob records | 364,291 | 524,288 |
| Value-blob bytes | 19,929,121 | 33,554,432 |
| Reverse-map views | 306 | 512 (one per logical-block slot) |
| Reverse-map entry upper bound | 56,462 | 65,536 |
| Reverse-map hash-slot upper bound | 154,624 | 262,144 |
| Reverse-map key-byte upper bound | 9,220,736 | 16,777,216 |
| Reverse-map owners | 56,462 | 65,536 |
| Plain-data map views | 3,366 | 5,632 (11 per logical-block slot) |
| Plain-data map entries | 444,461 | 524,288 |
| Plain-data map hash slots | 1,245,796 | 2,097,152 |
| Plain-data run values | 409,420 | 524,288 |
| Mandatory statement keys | 144,890 | 262,144 |
| Logical-block metadata bytes | 8,945 | 16,384 |

The byte-map slot and key-byte ceilings deliberately use the strict next power
of two even though their margins are narrow. They cover the deterministic FTA
shortcut evidence; any larger sweep must be divided by the approved batching
boundary, never accommodated by resizing a live arena.
`measurePhase2ProjectionUsage` is a pure census twin. Direct columns are exact;
the reverse-map entry, slot, and key-byte figures are safe no-dedup upper bounds
derived from the measured forward blobs. The builder coalesces equal normalized
keys into a smaller used prefix without exceeding those ceilings.

The complete fixed resident image contains 24 arrays and reserves 263.5 MiB on
the device. This is 3.2 percent of the RTX 4070 Laptop's 8,188 MiB. Construction
and upload assert every ceiling; no array grows after engine startup.

The retained `gpu_task_census_cpu_v1` shortcut run measured 497 task records,
1,953 request batches, 1,987 mandatory terms, and 9,823 expression stumps in its
largest Phase 2 sweep. One part used at most one task, four batches, four terms,
and 55 stumps. The fixed task-image ceilings are therefore 512 tasks, 2,048
batches, 2,048 terms, and 16,384 stumps. The four arrays reserve 731,136 bytes,
or about 0.70 MiB, and `CudaPhase2TaskBuffer` allocates them once. Each upload
copies only the bounded used prefixes and its native checksum twin covers all
four columns in canonical order.

The retained `gpu_filter_census_cpu_v1` shortcut run measured the whole global
filter/sort sweep, including repeated split parts and stump producers. Its peaks
were 1,959 filter calls, 10,481,129 examined statement rows, and 955,583 retained
rows. One call examined at most 20,486 rows and retained at most 1,752 of the
processor's first-8,192 accepted-row envelope. The first fixed bulk ceilings are
therefore 2,048 calls, 16,777,216 examined rows, 1,048,576 retained rows, and
32,768 examined rows per call. The measured next-power-of-two retained maximum
is 2,048 per call, but no per-call array uses it: the kernel preserves the
processor's semantic 8,192 cap and asserts only the global retained arena. These
counts size the schedule, predicate/scan scratch, and sort prefixes; they never
change the processor acceptance rule.

`CudaPhase2FilterSortBuffer` now allocates the measured production ceilings,
including two compact 64-bit key arrays and one reusable CUB scan/radix scratch
region. One count launch covers every call, one exclusive scan assigns compact
segments, and one emit launch replays the pure predicate before a global radix
sort. Keys order by call ordinal, decoded-name rank, and original statement
index; this exactly implements the processor's stable name-only sort. The
device predicate covers frozen validity ancestry, single-expression normalized
key construction, all four whole/subkey registry pairs, whole-key widening, the
first-8,192 accepted-row envelope, and the iteration ceiling.

The retained `gpu_grow_census_cpu_v1` shortcut run measured every normalized-key
expansion attempt, accepted owner-subkey event, growable frontier node, raw
whole-key request, and stump-producer node by exact expression depth. One sweep
performed at most 358,565,795 expansion attempts and produced at most 1,596,160
accepted owner-subkey events, 1,584,375 frontier pushes, and 72,674 raw requests.
Across the whole run the generator performed 4,077,562,830 expansion attempts.
The largest single-depth populations were 1,025,308 accepted events and
1,024,936 growable nodes at depth two. The deepest attempted request had six
expressions; accepted and emitted requests reached five expressions.

The first fixed growth ceilings are therefore 1,048,576 records in each of the
two frontier arrays, 2,097,152 accepted-event records accumulated across depths,
and 131,072 raw request records. Expansion attempts are evaluated directly from
frontier rows and filtered statement spans; no 358-million-row attempt array is
materialized. The stump producer remains inside the existing fixed task/stump
ownership: its sweep peaks were 9,823 singleton survivors, 1,275 depth-two
attempts, and 660 depth-two survivors. These ceilings preserve the measured FTA
shape and assert on overflow; deterministic batching remains the approved route
for a larger workload.

The first native growth primitive is live in
`gpu/phase2_cuda.cu::buildProjectedNormalizedKey` and
`projectedOwnerSetUSatisfied`. It builds the exact `NormKey` payload directly
from projected `IntEncodedExpr` rows, probes the selected registry's whole and
subkey tables, treats one- and two-premise subkeys as presence-only, and walks
the canonical `OwnerSet` blob from three premises onward. The direct CUDA twin
compares the full payload and separately proves accepted, present-but-u_-rejected,
absent, and short-subkey cases. It is the semantic core the bulk frontier now
calls; the production request stream is still processor-owned.

`gpu/phase2_projection.hpp::DeviceGrowthNode` stores only a call identifier,
stump-run ordinal, up to eight filtered positions, start position, folded
validity, and mandatory-term mask: 56 bytes. It deliberately does not carry a
256-identifier normalized-key buffer. `DeviceAcceptedGrowthEvent` persists the
statement path, verdict, and compact filtered-position path in 64 bytes after a
source frontier is reused; each position code stores the filtered position plus
one, reserving zero for a whole-only stump element that has no filtered
position. `DeviceRawGrowthRequest` adds the assigned growth position in 72
bytes.
`CudaPhase2GrowthBuffer` allocates 2,048 pointer-free call links, two
1,048,576-byte mandatory direct/suffix masks, two 1,048,576-node frontiers,
2,097,152 accepted events, 131,072 raw requests, 2,048 exact per-task subkey
work counters, and one shared counter record once. The task counts match the
processor's `g_growthMatchCount` input to the later split policy without
downloading or replaying the event ledger. The exact fixed ownership is
263,225,360 bytes, about 251.031 MiB; a
production-capacity RTX unit test allocates, reports, and releases it
successfully.

`Phase2GrowthScheduleArena` links each projected request batch to the filter
span it consumes. `CudaPhase2GrowthBuffer::runRequestGrowth` builds mandatory
view and suffix masks, seeds empty unsplit roots or exact stump nodes, and
expands the global frontier without materializing rejected attempts. Stump seeds
are grouped by absolute starting depth, so every wave remains depth-homogeneous
and the measured 1,024,936-node depth peak safely bounds each 1,048,576-record
frontier despite varying stump depths. The device
implements parent-chain validity comparability and deeper scope, hypothesis
agreement with the main-anchor exemption, distinct secondary variables with
recursion-product exclusion and `_orint_` widening, mandatory reachability,
normalized whole-key presence, and owner-subkey signatures. Persistent events
carry statement-index paths, so a whole-only stump probe does not depend on a
filtered position. The direct RTX twin proves the exact semantic
content, including mandatory pruning, scope and request-gate rejection,
whole-only requests, and a present three-premise subkey rejected by its owner.

`CudaPhase2OrderingBuffer` converts the unordered append ledger into the exact
processor stream without host semantic replay. Eleven stable radix passes order
the task, call, stump run, and eight path components according to the real
stack walk: immediate siblings emit in ascending filtered position while
continuing subtrees are visited in reverse push order. A segmented inclusive
scan assigns each task's cumulative accepted-subkey growth position. A fixed
open-address table then compares the complete semantic key — generator call,
statement count, and every original/validity identifier pair — and atomically
retains the minimum exact event order for each key. The unique request tokens
are finally radix-sorted by that order. Its deterministic base arrays consume
81,788,932 bytes at retained production ceilings before the single reusable
maximum CUB scratch allocation; `fixedAllocationBytes` reports the complete
runtime allocation, and the production RTX test proves it remains below
512 MiB.

The processor `[GPU-EVAL-CAPACITY]` census retained 26 FTA sweeps. Its maxima
are 72,674 requests and dependency passes; 1,049,978 reverse owners with at
most 804 per request; 47,930 subset candidates with at most four per request;
46,654 encoded hits; 82,620 local values with at most 39 per hit; 62,834 total
firing records; 1,935,835 generated bytes; 189,815 level values; 184,494 origin
dependencies; 1,608 marker-key references; 780 marker remaining-argument
references; and 1,345 marker-argument slices. The fixed device ceilings are the
next powers of two: 131,072 requests, 2,097,152 reverse-owner work items, 65,536
candidate owners and encoded hits, 131,072 local values, 65,536 firing records,
2,097,152 generated bytes, 262,144 levels and origin dependencies, 2,048 marker
keys and argument slices, and 1,024 marker remaining arguments.

`CudaPhase2EvaluationBuffer` allocates all of those work and output arrays once.
The deterministic arrays consume 37,267,500 bytes before one reusable maximum
CUB scan scratch allocation; the production RTX test measures 37,269,035 bytes
for the complete owner, below 64 MiB. `expandEvaluationWork` applies projected known-statement
dependency checks, premise validity folding, hypothesis and closed-scope gates,
then scans the reverse runs. Capacity-checked device atomics compact exact
remaining-argument subsets; each survivor rebuilds the ignore-u mapped key and
probes `overallEncoded`; the final scan expands encoded hits into flat
`LocalMemoryValue` blob work. The direct twin covers dependency rejection and
the fully known path, an extra non-subset owner, four exact mapped hits, and a
multi-value encoded run. Candidate append order is intentionally unobservable:
every record carries its ordered request index, later firing output is
canonically sorted by complete content, and doom reduction uses that request's
exact growth position.

## Implementation milestones and commit plan

Each numbered item becomes one or more complete commits. Update its status and
evidence immediately after every commit; never carry a completed milestone only
in conversation.

1. **Plan and environment — COMPLETE.** Created the sandbox worktree and this
 plan, record the installed toolchain, then commit and push the documentation
 snapshot.
2. **GPU cookbook and architecture proposal — COMPLETE AND APPROVED.** Read the
 two existing cookbooks completely, mapped Phase 2 data and ownership, wrote
 the GPU cookbook and relevant agentic SwDD navigation/pipeline note, and
 recorded the user's blanket architecture approval.
3. **CUDA build and device-contract substrate — COMPLETE.** The Visual Studio
 target builds native `sm_89` CUDA, asserts the device contract, and directly
 tests a real kernel launch. Backend parsing lands with integration so no
 selectable no-op CUDA mode exists.
4. **Read-only device projection — RESIDENT AND TASK INPUT IMAGES COMPLETE.**
 All 24 resident columns, fixed host/device ownership, complete non-empty
 upload checksum twins, corrected FTA capacity census, and native probes for
 every name, byte-map, reverse-map, and plain-data view are complete.
 The fixed host task image now represents executor parts, ordered request
 batches, mandatory terms, and exact expression-stump buckets without pointers.
 Its FTA capacity census, fixed 0.70 MiB CUDA ownership, bounded upload, and
 byte-exact device checksum twin are complete.
5. **Exact request-generation replay — DEVICE GENERATOR COMPLETE; PRODUCTION
 ROUTING PENDING.** The resident
 lookup layer, retained global filter/sort capacity census, fixed CUDA
 allocation, exact statement predicate, compaction, and stable decoded-name
 order are complete. The retained depth census fixes double-frontier,
 accepted-event, and raw-request capacities. Normalized-key growth, owner probes,
 request gates, mandatory reachability, and stump seeding are complete for
 semantic content. Exact processor stack order, task-local cumulative growth
 positions, and collision-checked per-call first-occurrence deduplication are
 complete. Direct logical-block fixtures prove the complete ordered request
 stream, mixed-stump order, duplicate-row handling, and per-call counter
 persistence. Production selection lands only after evaluation exists.
6. **Exact request evaluation and provenance — COMPLETE ON DEVICE.** The
 retained observation-only `[GPU-EVAL-CAPACITY]` census measures emitted and
 dependency-passing requests, reverse/subset owners, encoded hits, local
 values, all three firing kinds, generated bytes, and every variable-length
 output run without entering proof flow. Its 26 FTA rows are retained and
 byte-identity is proven. Fixed work/output ownership and evaluation expansion
 through dependency, validity, reverse-owner subset, mapped encoded-key, and
 local-value selection are live. Exact greedy substitution, repeated `u_`
 stripping, rule-scope folding/filtering, marker/demand purity, ordis2 depth,
 source classification, and fixed firing-expression output are live.
 Sorted-unique combined levels and complete head provenance are live. Exact
 disintegration, mail, ancestor-known, OR-disintegration, and marker-category
 verdicts are live. Marker keys, decoded-byte-sorted unique remaining
 arguments, bare arguments, and admission ceilings are complete in fixed
 output arenas. A direct RTX twin compares every header flag and every
 variable-length head/marker payload.
7. **Phase 2 integration — COMPLETE.** Complete device records now
 retain logical-block, part, and exact growth-position identity. Two fixed
 index arrays run a parallel merge sort over `(logical block, existing
 FiringRecord content key)` without moving payloads; the direct RTX twin pins
 the exact five-record order. Exact device doom reduction now mirrors every
 processor trigger, preserves the no-doom and compressor cases, and stably
 retains only the winning part's inclusive request prefix. The full evaluator
 ownership is 37,269,035 bytes. Fixed host staging now downloads every used
 output prefix and seals complete processor `FiringRecord` chains in device
 canonical order; exact head provenance and every marker payload pass the
 direct twin. Exact per-task subkey counters preserve the processor's later
 split-work statistic without treating whole-key-only request events as
 growth matches. `applyFiringRecords` skips its processor sort only under the
 explicit already-canonical contract, while all processor callers retain the
 existing route. `--phase2-backend cpu|cuda` now routes the live shortcut
 scheduler explicitly. CUDA mode runs only the split stump producer on the
 processor, then executes the semantic Phase 2 pipeline on-device with fixed
 owners and no fallback; host finalization only seals and deposits the already
 decided canonical records.
8. **FTA shortcut identity gate — COMPLETE.** The first CUDA shortcut completed
 48 real device passes, passed 1,542 native tests, 403 verifier unit tests, and
 10,038 proof-graph checks with zero failures. Its 311 retained artifacts are
 byte-identical to the current-source processor baseline.
9. **Performance and memory gate — COMPLETE.** The explicit barrier-to-barrier
 measurement is live and the paired baseline is retained separately from
 expanded Phase 3: processor Phase 2 is 110.333 seconds and CUDA Phase 2 is
 228.782 seconds, while end-to-end shortcut time is 218.442 versus 343.659
 seconds. CUDA events report 220.274 seconds for uploads plus device work and
 the enclosing CUDA route reports 228.121 seconds. Nsight Systems assigns
 217.594 seconds, 99.0% of all CUDA kernel time, to
 `phase2GrowthExpandKernel`; host-to-device copies total 0.330 seconds and
 device-to-host copies total 0.011 seconds. The first optimization therefore
 ports the processor's prefix-summary hoist into that kernel. The rerun reduced
 complete Phase 2 to 78.845 seconds, exact CUDA-owned time to 71.034 seconds,
 prover time to 186.043 seconds, and overall shortcut time to 192.096 seconds.
 This was 1.399 times faster than the processor in Phase 2 and 1.137 times faster
 end to end. The next counter-guided local-state reduction then reached 15.929
 seconds exact CUDA-owned time, 25.178 seconds complete Phase 2, 155.947 seconds
 prover time, and 161.911 seconds overall. CUDA Phase 2 is now 4.382 times faster
 than the processor and the complete shortcut 1.349 times faster. The new
 profile assigns 13.005 seconds and 87.0% of kernel time to growth; a
 representative launch reaches 54.35% occupancy and 46.49% compute throughput
 with 1.96% DRAM throughput. Route attribution assigns 8.270 seconds to host
 projection/scheduling, 15.684 seconds to the CUDA segment including the
 15.672-second event interval, and 0.283 seconds to download/sealing, reconciling
 to the 24.238-second route within 3 microseconds. The six fixed CUDA owners
 now total 725,527,125 bytes (691.92 MiB, 8.45% of the device). The refined
 run reports 50,732 firing records, 154,845 provenance dependencies, and no
 separate mail queue: the mail verdict occupies a firing-record flag. Its
 6.905-second projection and 0.073-second scheduling split proves that repeated
 logical-block projection, not scheduler construction, owns the remaining
 substantial processor seam. Internal attribution then measured 5.112 seconds
 of name work, dominated by 4.304 seconds of decoded lexical sorting. Sorting
 against the already-packed name image preserves the same byte comparator and
 reduced sort time to 1.936 seconds, projection to 4.308 seconds, complete Phase
 2 to 19.777 seconds, and the non-CUDA remainder to 5.268 seconds. A first radix
 implementation regressed on FTA's long common prefixes; collapsing each range's
 common prefix before its first real byte split fixed that coding flaw and reduced
 sorting to 1.545 seconds, projection to 3.987 seconds, and complete Phase 2 to
 19.257 seconds. Exact CUDA time is 14.332 seconds and the remaining non-CUDA
 portion is 4.925 seconds. The route is now 5.730 times faster than the retained
 processor Phase 2 baseline. Source audit also closes in-process projection reuse
 for FTA shortcut: each fresh prover subprocess has at most two passes and their
 non-straggler and straggler projection sets are disjoint. NVIDIA's official
 RTX 5090 versus the actual RTX 4070 Laptop specifications give a 4.722-times
 CUDA-core ratio and a 5.232-times rated core-clock-product ratio; the laptop's
 128-bit 16-Gbps memory shape is 256 GB/s versus 1,792 GB/s, a 7-times ratio.
 The local driver reports the exact device identity plus 3,105-MHz maximum SM
 and 8,001-MHz maximum memory clocks. Holding 4.925 measured host seconds and
 about 0.525 transfer seconds fixed while scaling the remaining device work by
 3 to 4 gives an 8.902-to-10.052-second modeled Phase 2: 1.916 to 2.163 times
 faster than the current RTX 4070 Laptop route and 10.976 to 12.394 times faster
 than the processor baseline. The rated 5.23-times sensitivity point is 8.090
 seconds and 2.380 times, not an expected benchmark. Native `sm_120` is embedded
 beside `sm_89`, so the estimate is runnable on a future RTX 5090 without PTX
 just-in-time compilation.
 Full shortcut prover and overall time remain the ultimate usefulness gate.
10. **Initial-port closeout — COMPLETE.** Update the cookbook with measured rules,
 document remaining non-shortcut work without claiming it complete, and leave
 both backends reproducibly selectable.

## Initial-port closeout boundary

The supported first-port commands are:

```powershell
C:\Users\nikol\anaconda3\python.exe main.py --shortcut --phase2-backend cpu --run-descriptor gpu_cpu
C:\Users\nikol\anaconda3\python.exe main.py --shortcut --phase2-backend cuda --run-descriptor gpu_cuda
```

Omitting `--phase2-backend` selects `cpu`. `main.py::main` asserts that `cuda` is
used only with `--shortcut`; `run_modes.py::shortcut_run` propagates the exact
selection to every native prover subprocess; `run_modes.cpp::fullRun` constructs
the analyzer with the selected enum; and `prover.cpp::proveKernel` skips ordinary
processor burst execution only when CUDA is selected. Selected CUDA mode asserts
on unsupported platform, capability, capacity, allocation, launch, or semantic
contract failures. There is no automatic processor fallback.

The completed claim is deliberately narrow:

- Windows FTA `main.py --shortcut` only; `main.py` rejects CUDA for the full run.
- Phase 1, Phase 3, Python graph processing, and verification remain on the
 processor.
- The processor still performs the approved split-stump discovery prepass and
 fixed-array sealing; request filtering, growth, ordering, evaluation,
 provenance, marker payloads, doom selection, and canonical content run on CUDA.
- Fixed capacities are measured for the retained FTA shortcut corpus. Other
 anchors, configs, or full-pipeline workload distributions require fresh censuses
 and identity gates before CUDA enablement.
- Native `sm_89` is runtime-proven on the RTX 4070 Laptop GPU. Native `sm_120` is
 compile- and embedding-proven but remains runtime-unmeasured until an RTX 5090
 is available.

The initial-port requirements are closed: persistent plan and cookbook, separate
CPU/CUDA backends, processor default, no fallback, FTA shortcut first, complete
device-side semantic work, full provenance, deterministic ordering and doom,
fixed ownership, exact Phase 2 and CUDA timing, current artifact identity,
stronger-GPU bound, and explicit non-shortcut boundary are all present and
evidence-linked below.

## Verification gates

### Per source commit

```powershell
& 'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe' GL_Quick_VS\GL_Quick.sln -p:Configuration=Release -p:Platform=x64 -t:Rebuild -verbosity:minimal
& 'GL_Quick_VS\GL_Quick\gl_quick.exe' --unit-tests
```

Pass means the full rebuild succeeds and every direct unit test passes. Adjust
the executable path only to the actual solution output established by the first
rebuild; record that fact here rather than guessing.

### FTA shortcut processor/GPU identity

Run from this Windows worktree through `main.py --shortcut`, with a distinct log
and retained artifact set for each backend. Never run the pipeline executable
directly.

Pass requires all of the following:

- both runs finish without assertion or fallback;
- theorem and proof-graph artifacts are byte-identical;
- provenance rows and their canonical order are byte-identical;
- verifier totals and every failure count are identical, with zero failures;
- GPU mode actually executed device kernels, proved by backend telemetry;
- the hashburst dump remains contract-identical if a targeted comparison is
 required.

### Performance

Performance is evaluated only after identity. The diagnostic comparison brackets
the complete Phase 2 scheduler window, including producer/executor work,
projection, device work, sealing, and finalization, but excluding Phase 1 and the
expanded Phase 3. Report processor and GPU Phase 2 wall time so the port itself is
measured, and report prover plus overall shortcut time beside it as the acceptance
criterion: the CUDA path is not called a performance win unless end-to-end shortcut
time improves. Transfer bytes, device-memory peak, queue peaks, utilization, and
energy-relevant run duration complete the report. A kernel-only or whole-pipeline
number is never reported as Phase 2 acceleration.

The implementation emits one `[PHASE2-TIMING]` row per `proveKernel` call. Sum
`iteration_seconds` across the complete shortcut log; do not sum analyzer-local
`cumulative_seconds`, because each pipeline subprocess owns a fresh analyzer.
Every CUDA pass also emits `[GPU-PHASE2] device_seconds` from reusable CUDA events
around projection/task uploads and every semantic device operation through
doom-prefix selection. Its `route_seconds` is the enclosing CUDA route wall time,
including host projection, scheduling, device work, downloads, and sealing. Report
both; neither replaces the complete Phase 2 or end-to-end shortcut measurement.

For host-overhead diagnosis, the same row also reports `prepare_seconds` from the
start of the selected CUDA route through logical-block projection and task/schedule
construction, `device_route_seconds` as host wall time around the exact event
interval, and `finalize_seconds` for used-prefix downloads plus processor sealing.
Their sum is `route_seconds` apart from clock-read precision. `device_seconds`
remains the authoritative CUDA-event interval including uploads; the host
sub-timers explain it but never replace it.

The exact CUDA-owned event interval deliberately includes projection and task
uploads. Data transfer is part of the selected backend's cost, not overhead to
subtract from the result. Nsight's separate copy totals remain diagnostic; the
paired baseline shows that transfers are negligible compared with frontier
growth on the current workload.

### Stronger commodity GPU bound

The model uses the live device identity and official NVIDIA specifications, not
game or synthetic benchmark claims. The
[RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
has 21,760 CUDA cores at 2.41 GHz. The
[RTX 4070 Laptop specification](https://www.nvidia.com/en-us/geforce/laptops/40-series/)
lists 4,608 CUDA cores, a 1.230-to-2.175-GHz boost range, 8 GB GDDR6, and a
128-bit interface. The local `nvidia-smi` query identifies an RTX 4070 Laptop GPU
with 3,105-MHz maximum SM and 8,001-MHz maximum memory clocks. The two devices
therefore have a 4.722-times core-count ratio, a 5.232-times official rated
core-clock-product ratio, and 1,792 versus 256 GB/s of memory bandwidth. The
measured kernel is latency/occupancy constrained and uses only 1.96% DRAM
throughput, so none of these hardware ratios is asserted as observed GL scaling.

For a conservative bound, retain 4.925 seconds of non-CUDA Phase 2 and 0.525
seconds of measured transfers, then scale the remaining 13.807 CUDA seconds.
Device factors 3 through 4 predict 10.052 through 8.902 seconds of complete
Phase 2, a 1.916-to-2.163-times gain over the current 19.257-second RTX 4070
Laptop run and a 10.976-to-12.394-times gain over the 110.333-second processor
baseline. Scaling by the 5.23 rated core-clock-product ratio gives an aggressive
8.090-second, 2.380-times sensitivity point, not a claimed ceiling or expected
result. Actual RTX 5090 measurements remain required. The project embeds
native `sm_89` and `sm_120` cubins for both CUDA translation units; it does not
depend on forward PTX compilation.

## Evidence ledger

Append concise dated rows. Keep measurements tied to a branch, commit, command,
and retained log or artifact path.

| Date | Evidence | Result |
|---|---|---|
| 2026-08-27 | Clean `main` and `origin/main`; absent before creation | Valid branch base |
| 2026-08-27 | `git worktree add -b sandbox/gpu .worktree/gpu main` | Dedicated worktree created at the exact base |
| 2026-08-27 | CUDA Toolkit files and `nvcc --version` by absolute path | CUDA 13.3, compiler 13.3.73 present |
| 2026-08-27 | Visual Studio CUDA build-customization directories | CUDA 13.3 integration present for Visual Studio 2022 and 18 |
| 2026-08-27 | `nvidia-smi` | RTX 4070 Laptop, driver 596.08, compute capability 8.9, 8,188 MiB |
| 2026-08-27 | Complete reads of `09b_statification_cookbook.md` and `09c_string_statification_cookbook.md` | Mandatory container, arena, span, string-lifetime, determinism, and verification rules loaded before production C++ |
| 2026-08-27 | Toolkit 13.3 runtime probe compiled and launched on the RTX 4070 | Default PTX launch failed against the 13.2-level driver; native `sm_89` launch succeeded and returned `4070` |
| 2026-08-27 | Live reads of `performElem2`, `generateEncodedRequestsStatic`, `checkLocalEncodedMemoryStatic`, `applyFiringRecords`, their data records, scheduler, build project, and shortcut argument path | Complete first-port device input/output and ownership map recorded in the GPU cookbook |
| 2026-08-27 | CPU `main.py --shortcut --run-descriptor gpu_cpu_baseline_v2` after the FTA mirror-refutation configuration repair | 1,527/1,527 C++ unit tests, 403/403 verifier unit tests, 10,038 proof-graph checks, 0 failures; prover 196.824 seconds, overall 200.65901 seconds |
| 2026-08-27 |  | Retained 314 CPU-reference files (11,602,682 bytes): run/frame/memory logs plus shortcut theorem, raw, processed, and full proof-graph outputs |
| 2026-08-27 | Full Release rebuild with CUDA 13.3 Visual Studio customization;  | Both production and test CUDA translation units compiled; `gl_quick.exe` linked successfully |
| 2026-08-27 | CUDA-enabled `gl_quick.exe --unit-tests`;  and  | 1,529/1,529 tests passed; direct device-contract and native-kernel round-trip tests passed |
| 2026-08-27 | CUDA 13.3 `cuobjdump --list-elf gl_quick.exe` | Two embedded native `sm_89` cubins confirmed, one per CUDA translation unit |
| 2026-08-27 | Full Release rebuild and  | 1,532/1,532 tests passed; lossless statement/name projection, fixed-allocation reuse, bounded device upload, and device-computed byte checksum twins passed |
| 2026-08-27 | Full Release rebuild, , and processor `main.py --shortcut --run-descriptor gpu_capacity_cpu_v1` | 1,532/1,532 native tests, 403/403 verifier tests, and 10,038 proof-graph checks with 0 failures; prover 200.047 seconds, overall 204.70891 seconds |
| 2026-08-27 | 26 deterministic `[GPU-CAPACITY]` rows in  | Sweep peaks: 306 logical blocks, 129,399 statements, 348,066 name records, 11,811,329 name bytes; per-block peaks: 20,486 statements, 67,806 name records, 2,916,976 name bytes |
| 2026-08-27 | SHA-256 comparison of theorems plus raw, processed, and full shortcut proof graphs against  | 311 current files versus 311 baseline files; zero missing, extra, or byte-different artifacts |
| 2026-08-27 | Full Release rebuild, , and processor `main.py --shortcut --run-descriptor gpu_semantic_capacity_cpu_v1` | 1,533/1,533 native tests, 403/403 verifier tests, and 10,038 proof-graph checks with 0 failures; prover 195.569 seconds, overall 199.62025 seconds |
| 2026-08-27 | 26 initial `[GPU-SEMANTIC-CAPACITY]` rows in  | First sweep peaks for rule strings, ten byte maps, blobs, plain-data maps, value runs, mandatory keys, and metadata; the 2026-08-28 live-read audit below corrected the omitted inputs before allocation |
| 2026-08-27 | SHA-256 comparison after the complete semantic census against  | 311 current files versus 311 baseline files; zero missing, extra, or byte-different artifacts |
| 2026-08-28 | Full Release rebuild and  | 1,533/1,533 native tests passed; all 24 fixed columns were non-empty where applicable, uploaded to the RTX, and matched the device-computed checksum |
| 2026-08-28 | Processor `main.py --shortcut --run-descriptor gpu_complete_projection_cpu_v1` and  | 1,533/1,533 native tests, 403/403 verifier tests, and 10,038 proof-graph checks with 0 failures; prover 193.952 seconds, overall 197.95149 seconds |
| 2026-08-28 | 26 corrected `[GPU-SEMANTIC-CAPACITY]` rows | Measured name slots, ten forward byte maps, derived reverse bounds, seven actually-read plain-data maps including frozen branches, runs, mandatory keys, and metadata |
| 2026-08-28 | SHA-256 comparison after complete 24-column projection against  | 311 current files versus 311 baseline files; zero missing, extra, or byte-different artifacts |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,533/1,533 native tests passed; native CUDA probes matched name hits/misses, all ten byte-map views and payload runs, reverse owners, and all seven plain-data views and scalar/run payloads |
| 2026-08-28 | Full Release rebuild, , and processor `main.py --shortcut --run-descriptor gpu_mandatory_indexes_cpu_v1` | 1,533/1,533 native tests, 403/403 verifier tests, and 10,038 proof-graph checks with zero failures; prover 195.393 seconds, overall 199.46115 seconds |
| 2026-08-28 | Then-current nine-view plain-data census | Sweep peaks at that snapshot: 2,754 views, 438,571 entries, 1,228,636 slots, and 409,420 run values; the later eleven-view current-source census supersedes these capacity figures |
| 2026-08-28 | SHA-256 comparison after derived mandatory indexes against  | 311 current files versus 311 baseline files; zero missing, extra, or byte-different artifacts |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,534/1,534 native tests passed; direct task twin covered all four normal request batches, ordered terms, split stumps, fixed-allocation reuse, and the unsplit counter-example task |
| 2026-08-28 | Processor `main.py --shortcut --run-descriptor gpu_task_census_cpu_v1` and  | 1,534/1,534 native tests, 403/403 verifier tests, and 10,038 proof-graph checks with zero failures; prover 194.943 seconds, overall 198.96207 seconds |
| 2026-08-28 | 26 deterministic `[GPU-TASK-CAPACITY]` rows | Sweep peaks: 497 tasks, 1,953 batches, 1,987 mandatory terms, and 9,823 stumps; per-part maxima: one task, four batches, four terms, and 55 stumps; fixed task image reserves 0.70 MiB |
| 2026-08-28 | SHA-256 comparison after the task-capacity census against  | 311 current files versus 311 baseline files; zero missing, extra, or byte-different artifacts |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,535/1,535 native tests passed; all four task columns uploaded to the RTX and matched the device-computed byte checksum |
| 2026-08-28 | Full Release rebuild , , and processor `main.py --shortcut --run-descriptor gpu_filter_census_cpu_v1` | 1,535/1,535 native tests, 403/403 verifier tests, and 10,038 proof-graph checks with zero failures; prover 202.038 seconds, overall 206.04497 seconds |
| 2026-08-28 | 26 deterministic `[GPU-FILTER-CAPACITY]` rows | Sweep peaks: 1,959 combined request/producer calls, 10,481,129 examined rows, and 955,583 retained rows; per-call peaks: 20,486 examined and 1,752 retained |
| 2026-08-28 | SHA-256 comparison after the filter/sort census against  | 311 current files versus 311 baseline files; zero missing, extra, or byte-different artifacts |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,536/1,536 native tests passed; the exact production-sized filter/sort allocation processed all four registries and matched processor retained counts and ordered statement indices across whole-key widening, frozen ancestry, iteration rejection, and equal-name stability |
| 2026-08-28 | Full Release rebuild , , and processor `main.py --shortcut --run-descriptor gpu_grow_census_cpu_v1` | 1,536/1,536 native tests, 403/403 verifier tests, and 10,038 proof-graph checks with zero failures; prover 196.835 seconds, overall 200.98066 seconds |
| 2026-08-28 | 26 deterministic `[GPU-GROW-CAPACITY]` rows | Sweep peaks: 358,565,795 expansion attempts, 1,596,160 accepted owner-subkey events, 1,584,375 frontier pushes, and 72,674 raw requests; largest depth: 1,025,308 accepted events and 1,024,936 frontier nodes |
| 2026-08-28 | SHA-256 comparison after the normalized-key growth census against  | 311 current files versus 311 baseline files; zero missing, extra, or byte-different artifacts |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,537/1,537 native tests passed; a real CUDA kernel matched the processor's multi-premise normalized key and owner-subkey verdicts for accepted, present-but-rejected, absent, and short presence-only cases |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,538/1,538 native tests passed; the RTX allocated and released the exact two-frontier, accepted-event, and raw-request production ownership totaling 215 MiB |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,539/1,539 native tests passed; unsplit bulk CUDA growth matched seven exact semantic events and four raw requests across mandatory pruning, validity and request gates, whole-only hits, and owner rejection; a second run matched three events and two requests from terminal depth-one plus growable depth-two stumps; fixed ownership rose only for call links and the two retained-row masks to 227,565,584 bytes |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,540/1,540 native tests passed; eleven stable radix passes reproduced the exact processor stack walk, the segmented scan reproduced task-local growth positions, mixed stumps retained their exact order, and collision-checked deduplication retained the first semantic request independently per generator call; growth ownership is 263,217,168 bytes and the production ordering owner remains below 512 MiB |
| 2026-08-28 | Full Release rebuild  and ; processor `main.py --shortcut --run-descriptor gpu_eval_census_cpu_v1` | 1,540/1,540 native tests, 403/403 verifier tests, and 10,038 proof-graph checks passed with zero failures; prover 195.044 seconds and overall 200.60304 seconds |
| 2026-08-28 | 26 deterministic `[GPU-EVAL-CAPACITY]` rows in  | Sweep peaks: 72,674 requests, 1,049,978 reverse owners, 47,930 subset candidates, 46,654 encoded hits, 82,620 local values, 62,834 firing records, 1,935,835 generated bytes, 189,815 levels, and 184,494 provenance dependencies; per-item maxima: 804 reverse owners per request, four candidates per request, and 39 values per encoded hit |
| 2026-08-28 | SHA-256 comparison after the evaluation census against  | 311 current files versus 311 baseline files; zero missing, extra, or byte-different artifacts |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,540/1,540 native tests passed; device evaluation matched the four-request fixture at both the dependency-rejected 2/1/1/1 and fully known 5/4/4/5 reverse-owner/candidate/hit/value shapes; mapped ignore-u keys, non-subset rejection, multi-value expansion, and production allocation below 64 MiB passed |
| 2026-08-28 | Full Release rebuild , native gate , and CUDA shortcut  | 1,542/1,542 native tests, 403/403 verifier unit tests, 48 real CUDA Phase 2 passes, 10,038 proof-graph checks, zero failures; prover 338.939 seconds and overall 343.691 seconds |
| 2026-08-28 | Retained  versus the completed CUDA shortcut artifacts | 311 baseline and 311 CUDA artifacts; zero missing, extra, or different files |
| 2026-08-28 | Full Release rebuild  and native gate  | Link succeeded and 1,543/1,543 native tests passed; the direct CUDA-event timer test measured a positive real-kernel device interval |
| 2026-08-28 | Paired processor  and CUDA  | Processor: Phase 2 110.333405 seconds, prover 213.808, overall 218.44163. CUDA: exact device 220.274061, enclosing CUDA route 228.121285, complete Phase 2 228.782324, prover 338.703, overall 343.65850. CUDA is 2.073× slower in Phase 2 and 1.573× slower overall; device work is 96.28% of CUDA Phase 2, so transfer/sealing overhead is not the primary bottleneck |
| 2026-08-28 | Paired timing-run artifacts in  and  | 305 current shortcut artifacts compared, zero missing and zero different; both runs passed 10,038 checks with zero failures |
| 2026-08-28 | Nsight Systems process-tree profile  and , produced through `main.py --shortcut` | `phase2GrowthExpandKernel` consumed 217.594 seconds across 224 launches, 99.0% of CUDA kernel time. Host-to-device copies consumed 0.330 seconds, device-to-host copies 0.011 seconds, and device-to-device copies below 0.001 seconds; transfer is not the limiting seam |
| 2026-08-28 | Live comparison of processor `Memory::generateEncodedRequestsStatic` with CUDA `phase2GrowthExpandKernel` | The processor builds normalized-key and request-gate prefix summaries once per frontier pop, while CUDA rebuilt both across every candidate position. The first optimization hoists those exact order-free summaries once per CUDA frontier node and folds only the appended expression |
| 2026-08-28 | Full Release rebuild  and native gate  | Native `sm_89` link succeeded and all 1,543 tests passed, including the direct bulk growth-content twin over repeated variables, request gates, owner signatures, recursion-product exclusions, exact events, requests, and subkey-work counts |
| 2026-08-28 | Optimized CUDA shortcut  | 48 device passes and 45 Phase 2 rows completed. Exact CUDA-owned uploads plus device work totaled 71.033664 seconds, the enclosing CUDA route 78.191829 seconds, complete Phase 2 78.844768 seconds, prover 186.043 seconds, and overall shortcut 192.09611 seconds. The prefix hoist accelerated CUDA-owned time 3.101 times and complete CUDA Phase 2 2.902 times. Against the paired processor baseline, Phase 2 is 1.399 times faster and end to end is 1.137 times faster. The verifier passed 10,038 checks with zero failures |
| 2026-08-28 |  compared with  | 305 theorem, raw, processed, and full proof-graph files compared by SHA-256; zero missing, extra, or byte-different artifacts. The retained optimized set contains those 305 artifacts plus its run, frame-timing, and memory logs |
| 2026-08-28 | Optimized process-tree Nsight Systems profile  and  | `phase2GrowthExpandKernel` still consumes 68.320 seconds across 224 launches, 97.6% of all CUDA kernel time. Filter count and emit consume 0.933 and 0.470 seconds; every other semantic kernel is below 0.073 seconds. Host-to-device copies total 0.337 seconds and device-to-host copies 0.011 seconds. Growth remains the only material optimization seam after the prefix hoist |
| 2026-08-28 | Nsight Compute basic-set attempt  | The complete shortcut still passed 10,038 checks with zero failures, but Nsight Compute emitted `ERR_NVGPUCTRPERM` before collecting the selected launch because the Windows account lacks NVIDIA performance-counter permission. No occupancy or throughput estimate is substituted |
| 2026-08-28 | Enabled-counter probe  and production report  | Counter access succeeded; 1,543/1,543 native tests and the profiled shortcut's 10,038 verifier checks passed. The representative 694-block growth launch took 633.13 milliseconds under collection, reached 58.78% memory but only 8.04% compute throughput, used 65 registers and a 5,024-byte stack per thread, and achieved 43.89% occupancy against 50% theoretical. Registers limit residency to three blocks per multiprocessor; Nsight classifies the launch as latency-bound and estimates a 50% local opportunity from the occupancy ceiling |
| 2026-08-28 | Full Release rebuild , native gate , and direct resource report  | Link succeeded and 1,543/1,543 exact tests passed. Reusing the final serialized buffer and reading prior appended arguments directly reduced compiled growth resources from 65 to 63 registers and from a 5,024-byte to 3,184-byte thread stack. Register-limited residency rose from three to four blocks per multiprocessor and theoretical occupancy from 50% to 66.67%; the one-block direct fixture is not used for achieved-occupancy timing |
| 2026-08-28 | Counter-guided CUDA shortcut  | 48 device passes and 45 Phase 2 rows completed. Exact CUDA-owned uploads plus device work totaled 15.928690 seconds, enclosing CUDA route 24.358358 seconds, complete Phase 2 25.177654 seconds, prover 155.947 seconds, and overall shortcut 161.91054 seconds. The local-state change accelerated CUDA-owned time 4.459 times and complete GPU Phase 2 3.132 times over. Against the paired processor baseline, Phase 2 is 4.382 times faster and end to end 1.349 times faster. The verifier passed 10,038 checks with zero failures |
| 2026-08-28 |  compared with  | 305 theorem, raw, processed, and full proof-graph files compared by SHA-256; zero missing, extra, or byte-different artifacts. The retained set contains those 305 artifacts plus its run, frame-timing, and memory logs |
| 2026-08-28 | Optimized process-tree profile  and production counters  | Growth fell to 13.005 seconds and 87.0% of kernel time; filter count/emit now consume 1.058/0.626 seconds and all transfers about 0.525 seconds. The representative growth launch fell from 633.13 to 97.41 milliseconds, reached 54.35% occupancy and 46.49% compute throughput with 63 registers and a 3,184-byte stack, and used only 1.96% DRAM throughput. The remaining 8.43-second route-minus-device gap is now comparable to growth and requires host sub-timing before another architecture change |
| 2026-08-28 | Full Release rebuild  and native gate  | Link succeeded and 1,543/1,543 tests passed; timing-only telemetry now brackets host preparation, CUDA-segment wall, and host finalization without changing the existing CUDA-event, route, proof, or allocation contracts |
| 2026-08-28 | Route-attribution shortcut  | Across 48 CUDA passes, preparation/projection/scheduling totaled 8.270264 seconds, CUDA-segment wall time 15.684074 seconds, exact CUDA-event time 15.671945 seconds, and download/sealing 0.283341 seconds. The attributed sum is 24.237679 seconds versus 24.237682 seconds enclosing route time, a 2.8-microsecond reconciliation delta. Complete Phase 2 was 25.045766 seconds; the remaining post-route Phase 2 work was 0.808084 seconds. The diagnostic run passed 10,038 checks with zero failures; its 165.645-second overall runtime is retained as diagnostic variation, not the authoritative performance result |
| 2026-08-28 | Full Release rebuild , native gate , and CUDA shortcut  | All 1,543 native tests passed. The six process-static owners report exact `cudaMalloc` requests totaling 704,555,605 bytes: projection 278,979,600, tasks 731,144, filtering 25,512,447, growth 263,225,360, ordering 98,838,019, and evaluation 37,269,035. The total is 671.92 MiB, 8.20% of the 8,188 MiB device. Across 48 passes, complete Phase 2 was 24.760829 seconds and exact CUDA-event time 15.858907 seconds. The run passed 10,038 checks with zero failures and all 305 current proof artifacts matched the retained processor baseline byte for byte |
| 2026-08-28 | Full Release rebuild , native gate , and CUDA shortcut  | All 1,543 native tests passed. Across 48 passes, complete Phase 2 was 22.948067 seconds, preparation 6.977704, projection 6.904738, scheduling 0.072964, exact CUDA-event time 14.611697, CUDA-segment wall time 15.104797, finalization 0.230716, and route time 22.313216 seconds. Projection is 98.95% of preparation and 30.09% of complete Phase 2. Peak traffic was 1,088,142 accepted events, 49,380 unique requests, 1,007,955 reverse owners, 50,732 firing records, 1,619,493 generated bytes, 150,183 level values, and 154,845 provenance dependencies. This is 4.808 times faster than the retained processor Phase 2 baseline; 10,038 checks passed and all 305 current artifacts remained exact |
| 2026-08-28 | Projection-stage attribution  | Across 48 passes, projection consumed 6.754050 seconds and its mutually exclusive internal intervals accounted for 6.601027 seconds. Name projection owned 5.112147 seconds: 0.618587 record construction, 4.304092 decoded lexical sorting, 0.005251 rank writes, and 0.184214 lookup-slot construction. Names were 77.44% of accounted projection and lexical sorting was 84.19% of name work |
| 2026-08-28 | Packed-name sort clean rebuild , native gate , and CUDA shortcut  | All 1,543 native tests passed. Sorting exact decoded ranks from already-packed name spans reduced name sort from 4.304092 to 1.935975 seconds, projection from 6.754050 to 4.308227 seconds, and complete Phase 2 from 22.4424 to 19.7768 seconds. Exact CUDA-event time was 14.509291 seconds, so complete Phase 2 is now 5.579 times faster than the 110.333405-second processor baseline. Both verifier passes completed 10,038 checks with zero failures and all 305 current semantic artifacts matched the retained processor baseline by SHA-256 |
| 2026-08-28 | First fixed-capacity name-radix run  | The complete shortcut remained semantically exact with 10,038 checks and zero failures, but one 257-bucket pass per shared-prefix byte increased name sorting to 4.706019 seconds, projection to 7.098890 seconds, and complete Phase 2 to 23.3115 seconds. This run is failure evidence, not the accepted performance result |
| 2026-08-28 | Prefix-collapsed radix clean rebuild , native gate , and CUDA shortcut  | All 1,543 native tests passed, including full rank equality against `compareSpans` for shared-prefix and prefix-terminator names. Common-prefix collapse reduced sort time from the accepted packed-sort 1.935975 to 1.544649 seconds, projection from 4.308227 to 3.987276 seconds, and complete Phase 2 from 19.7768 to 19.2570 seconds. Exact CUDA-event time was 14.331816 seconds; Phase 2 is 5.730 times faster than the 110.333405-second processor baseline. Both verifier passes completed 10,038 checks with zero failures and all 305 semantic artifacts matched the retained baseline by SHA-256 |
| 2026-08-28 | Live `proveKernel` pass and `main.py --shortcut` process-boundary audit | Every shortcut prover subprocess performs at most two CUDA passes. Pass one excludes `produceOnly` stragglers and pass two contains only those stragglers, so their projected logical-block sets are disjoint; the subprocess then exits. An in-process generation cache therefore has zero reuse hits on the current FTA shortcut target |
| 2026-08-28 | Live RTX 4070 Laptop identity, NVIDIA official laptop/RTX 5090 specifications, and current 19.257-second Phase 2 split | The actual 4,608-core laptop GPU versus 21,760-core RTX 5090 gives a 4.722-times core ratio and 5.232-times official rated core-clock-product ratio; 256 versus 1,792 GB/s gives 7 times bandwidth. Holding 4.925 host and 0.525 transfer seconds fixed, 3-to-4-times device scaling models 10.052-to-8.902-second Phase 2, 1.916-to-2.163 times faster than the current laptop route; a 5.23-times sensitivity point is 8.090 seconds and 2.380 times. This remains a model until measured on the card |
| 2026-08-28 | Dual-native clean rebuild , `cuobjdump --list-elf`, and  | CUDA 13.3 confirmed support for both targets. The executable embeds native `sm_89` and `sm_120` cubins for each of two CUDA translation units, and all 1,543 tests passed on the RTX 4070 through native `sm_89`; no forward PTX dependency remains |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,540/1,540 native tests passed; GPU output matched exact numeric reverse substitution plus repeated `u_` stripping at four request shapes, inherited the deeper rule/request validity, preserved LMV source flags, and rejected an impure marker before consuming firing output |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,540/1,540 native tests passed; four GPU head records matched 12 sorted-unique premise/rule level values and 12 ordered provenance dependencies, including removal of the `-1` tier, source-implication-first order, decoded expression/validity premise order, and the corrected 35,164,192-byte deterministic evaluator base |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,540/1,540 native tests passed; four exact head records and one exact marker record matched all verdict flags, 12 levels, 12 provenance dependencies, two substituted marker keys, decoded-byte-sorted unique remaining arguments, one bare argument, the 35,688,492-byte deterministic evaluator base, and the 35,690,027-byte complete allocation including reusable scan scratch |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,540/1,540 native tests passed; the device retained logical-block/part/growth identity and reproduced the five-record processor content order through fixed parallel merge-sort index arrays; deterministic evaluator base 37,261,356 bytes and complete allocation 37,262,891 bytes |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,540/1,540 native tests passed; no-doom and compressor cases retained all five records; sole-goal and contradiction cases selected the exact position-3 request-0 prefix; a shared-position case retained request 2's complete batch while excluding later request 3; deterministic evaluator base 37,267,500 bytes and complete allocation 37,269,035 bytes |
| 2026-08-28 | Full Release rebuild  and  | Link succeeded and 1,541/1,541 native tests passed; five device-canonical records sealed through fixed host arrays with exact expressions, validities, level runs, source-first provenance, verdict flags, marker keys, sorted remaining arguments, bare arguments, and admission fields; the canonical-input deposit route preserved append order without a second sort |
| 2026-08-28 | Full Release rebuild , , and processor `main.py --shortcut --run-descriptor gpu_projection_reaudit_cpu_v1` | 1,541/1,541 native tests, 403/403 verifier tests, and 10,038 proof-graph checks passed with zero failures; 311/311 retained artifacts are byte-identical; the device task counter matched five subkey-accepted nodes while excluding whole-key-only events; current eleven-view sweep peaks are 3,366 views, 444,461 entries, and 1,245,796 slots, while metadata peaks at 8,945 bytes; prover 195.110 seconds, overall 199.23079 seconds |
| 2026-08-28 | Final same-tip landing pair  and  after merging main | Processor complete Phase 2 was 96.212168 seconds and overall runtime 203.22474 seconds. CUDA complete Phase 2 was 19.835229 seconds, exact device time 15.034592 seconds, and overall runtime 127.55500 seconds: 4.851-times Phase 2 and 1.593-times end-to-end acceleration. Both runs passed 9,994 checks with zero failures; 305 semantic artifacts had zero missing, extra, or SHA-256-different files |

## Decision ledger

| Date | Decision | Authority |
|---|---|---|
| 2026-08-27 | Keep provenance in GPU mode | User |
| 2026-08-27 | Keep processor and GPU paths in parallel initially | User |
| 2026-08-27 | Select the backend with a flag | User |
| 2026-08-27 | Port FTA `main.py --shortcut` first | User |
| 2026-08-27 | Maintain the GPU cookbook in the agentic SwDD | User |
| 2026-08-27 | `docs/GPU/gpu_plan.md` is the compaction-survival plan | User |
| 2026-08-27 | Approve every architecture proposal in this plan and proceed without further architectural micro-approvals; user will evaluate the completed work afterwards | User |

## Failure ledger

| Date | Failure | Resolution or next action |
|---|---|---|
| 2026-08-27 | Initial audit found the CUDA Toolkit and compiler absent | User installed WinGet package `Nvidia.CUDA` 13.3; re-audit passed |
| 2026-08-27 | Current Codex process does not inherit the newly installed CUDA environment variables | Use confirmed absolute paths until a fresh process; do not treat driver compatibility as compiler presence |
| 2026-08-27 | Toolkit 13.3 default PTX kernel launch failed with `the provided PTX was compiled with an unsupported toolchain` | Compile the current target as native `sm_89`; the launch probe then passed. Do not rely on PTX JIT with the current driver |
| 2026-08-27 | Visual Studio inherited both `PATH` and `Path`, and MSBuild stopped with duplicate-key `MSB6001` | Launch MSBuild through a clean child environment containing one canonical `PATH`; the full rebuild then reached the linker |
| 2026-08-27 | The new worktree's ignored vcpkg tree lacked the existing mimalloc library and debug directories | Populate those ignored dependency directories from the main checkout's installed tree; the full rebuild then produced `gl_quick.exe` |
| 2026-08-27 | Fresh FTA shortcut baseline passed all unit tests, then asserted because `mirror_refutation` required a conjecturer-written `mirror_pairs.txt` that shortcut mode deliberately never creates | Set `ConfigFTA.json` `mirror_refutation` to `false`; its counterexample-filter iteration count is zero, so this removes an impossible artifact precondition without changing a proof decision |
| 2026-08-27 | First CUDA project rebuild could not resolve `CudaToolkitDir` because this long-lived process predates the Toolkit installation | Inject the installer-owned machine values `CUDA_PATH` and `CUDA_PATH_V13_3` into the clean child build environment; project files still use the installed Visual Studio customization |
| 2026-08-27 | CUDA translation units initially parsed nested namespace syntax as pre-C++17 | Add `-std=c++17` to both project `CudaCompile` configurations, matching the existing host language standard |
| 2026-08-27 | CUDA objects compiled, but the host link could not locate `cudart.lib` | Add the customization-provided `$(CudaToolkitLibDir)` to both link configurations; do not hard-code a toolkit library path |
| 2026-08-28 | The first semantic census called itself complete before allocation but omitted the request generator's `frozenOrBranches` input, the name lookup slots, and the derived remaining-argument reverse index | Audited the live generator/evaluator reads, replaced two unused send-filter counts with the seven actual plain-data inputs, added safe reverse-index bounds, reran the full shortcut, and corrected the plan and cookbook before freezing capacities |
| 2026-08-28 | The complete resident image carried delta and external mandatory statement keys but no fixed-load device membership indexes for them | Added two derived `DevicePodMapKind` views inside the existing arrays and retained the raw ordered key slices for byte-exact projection evidence; later mail-eligibility additions raised the current image to eleven views, and the current-source census above supersedes the nine-view capacity row |
| 2026-08-28 | CUDA 13.3 CUB headers reject MSVC's legacy preprocessor mode | Add `/Zc:preprocessor` to both CUDA project configurations through the CUDA compiler option; the next full rebuild compiled the bulk scan and radix primitives |
| 2026-08-28 | The first doom compaction used only the packed growth-position/part line, which would retain a later request sharing the trigger's growth position even though the winning processor part stops immediately | Add a second fixed per-block minimum over exact ordered-request indices at the winning line, compact through that request inclusively, and pin the request-2-versus-request-3 shared-position case in the direct RTX twin |
| 2026-08-28 | The first production-routed CUDA pass stopped before upload because the 8,192-byte metadata ceiling came from the pre-mail-eligibility projection census | Use the current-source `gpu_projection_reaudit_cpu_v1` peak of 8,945 bytes and raise the immutable metadata ceiling to the next power of two, 16,384 bytes; retain the assertion and do not resize live |
| 2026-08-28 | The second production-routed CUDA run rejected a projected rule whose advertised maximum key length exceeded the eight-expression frontier capacity before that rule was ever traversed | Accept long-rule metadata at projection time and assert only when a live frontier would cross from eight to nine expressions; the next run completed all 48 CUDA passes with exact artifacts |
| 2026-08-28 | The first exact-device-timer build included `cuda_runtime_api.h` from ordinary host `prover.cpp`, whose MSVC include path intentionally does not inherit the CUDA compiler headers | Hide the two reusable events behind `CudaPhase2DeviceTimer` in the existing CUDA module; `prover.cpp` sees no CUDA runtime type, and the direct native-work timer test pins the interface |
| 2026-08-28 | Nsight Compute rejected hardware-counter collection with `ERR_NVGPUCTRPERM`; the application itself completed and verified | Resolved by enabling Developer Settings and allowing all users access under Developer → Manage GPU Performance Counters.  and the representative production report prove collection is live |
| 2026-08-28 | The first fixed-ownership unit command used an obsolete executable path and therefore ran no test | Re-ran the established `GL_Quick_VS\GL_Quick\gl_quick.exe --unit-tests` path; all 1,543 tests passed |
| 2026-08-28 | The first projection-traffic unit attempt started before the long clean link had produced the executable and therefore ran no test | Observed the still-running MSBuild/link processes, waited for the clean rebuild to complete, then ran the native gate; all 1,543 tests passed |
| 2026-08-28 | Two first artifact comparisons selected obsolete shortcut output roots and were rejected before being treated as evidence | Compared the three current proof-graph roots under `files/shortcut` plus the current and retained filtered-conjecture files; all 305 current semantic artifacts matched byte for byte |
| 2026-08-28 | The first packed-name comparator build captured member vectors as local variables and MSVC rejected the lambda | Capture `this` and the immutable record offset; the following full clean rebuild linked, all 1,543 native tests passed, and the complete shortcut verified byte-identically |
| 2026-08-28 | The first most-significant-byte name radix performed one full bucket pass per common-prefix byte and regressed exact sorting from 1.936 to 4.706 seconds | Keep the fixed-capacity radix design but collapse the complete common prefix of each range before counting its first differing byte. The next clean build and full shortcut reduced sorting to 1.545 seconds and complete Phase 2 to 19.257 seconds with exact artifacts |
| 2026-08-28 | The first stronger-GPU model compared the RTX 5090 with a desktop RTX 4070 even though the measured device is an RTX 4070 Laptop GPU | Re-query the live device, use NVIDIA's official 4,608-core laptop specification and the local clock/memory telemetry, replace every desktop-derived ratio, and retain the incorrect model only in immutable commit history as rejected evidence |
| 2026-08-28 | The first clean build after merging main found the GPU projection test still calling main's removed `ensureEmptySubkeyRecord` fixture helper | Install the same short-subkey test row through `addShortSubkeyOwner` with an explicit owner, matching main's new exact-removal representation; production GPU code had already merged onto the owner-bearing form |
| 2026-08-28 | The first merged unit run reached the repaired projection test but its synthetic packed owner named rule identifier 1 while the fixture's rule interner was empty, causing the owner projection to assert on a `PagedVector` bounds check | Intern one fixture rule first and pack that real identifier with the main validity identifier; the owner row is now a valid miniature of main's production representation |
| 2026-08-28 | The valid owner fixture still reached the same bounds assertion without a native stack trace | Add flushed log-only stage traps to the merged processor/CUDA filter twin so fixture, projection, upload, processor-oracle, and CUDA-filter boundaries identify the exact failing call while leaving production behavior untouched |
| 2026-08-28 | Stage traps proved projection packing and upload completed before the bounds assertion, narrowing the fault to the five processor-oracle filter calls | Name and flush each oracle call separately so the failing hash-memory instance and whole-key widening mode are trace-visible |
| 2026-08-28 | The named oracle traps proved only `overall-union` failed: the old GPU fixture used `.mint(key)` on main's new whole-key owner map, leaving a key with a zero-length owner run | Build the whole-key row through `addWholeKeyOwner` with the same interned fixture owner used by short subkeys; this matches main's production invariant and preserves the filter test's intended presence-only verdict |
| 2026-08-28 | The next merged unit test reached CUDA but asserted at the end of the old projected `OwnerSet` reader because main appended provenance-owner records to the canonical blob | Consume and validate the owner count and twelve-byte owner records after the signatures without using provenance as a proof input; update every GPU fixture to carry real owners and install whole keys through `addWholeKeyOwner` |
| 2026-08-28 | The first post-merge CUDA shortcut passed two real device iterations, then the owner-bearing value-blob projection exceeded its pre-main 16,777,216-byte fixed ceiling | Use the paired processor census peak of 19,929,121 bytes and raise only the immutable blob arena to the next power of two, 33,554,432 bytes; overflow still asserts and CUDA still never resizes or falls back |
| 2026-08-28 | The same-tip CUDA retry passed the enlarged blob arena and stopped at the next stale projection ceiling: 1,051,000 forward-map hash slots versus 1,048,576 allocated | Audit every paired processor census field against its compiled ceiling, confirm all other current fields fit, and raise only forward-map slots to the next power of two, 2,097,152 |
| 2026-08-28 | A quick timing extractor reported 41.835 seconds for the final CUDA Phase 2, apparently doubling the accepted 19.257-second result | The extractor accepted only decimal characters, so one `3e-07` row, two `2e-07` rows, and fifteen `1e-07` rows were read as 3, 2, and 1 seconds, adding exactly 21.999998 false seconds. Parse the complete numeric token with `[0-9.eE+-]+`; row-by-row extraction gives 19.835229 seconds after merge versus 19.257039 seconds before merge |

## Commit ledger

| Milestone | Commit | Verification | Push |
|---|---|---|---|
| Plan and environment | | `git diff --cached --check`; one-file documentation snapshot | |
| GPU cookbook and architecture proposal | | `git diff --cached --check`; four-file agentic SwDD snapshot; native `sm_89` runtime probe passed | |
| Fresh shortcut prerequisite and CPU baseline | | 1,527/1,527 native unit tests; 403/403 verifier unit tests; 10,038 proof-graph checks, 0 failures; retained CPU artifacts | |
| Native CUDA build and launch contract | | Full Release rebuild; two native `sm_89` cubins; 1,529/1,529 unit tests including device query and real kernel round trip | |
| Fixed-capacity logical-block projection | | Full Release rebuild; 1,532/1,532 tests including host lossless twins, allocation reuse, bounded device upload, and device checksum identity | |
| FTA projection capacity census | | Full Release rebuild; 1,532/1,532 native tests; 403/403 verifier tests; 10,038 proof-graph checks with zero failures; 311/311 retained artifacts byte-identical | |
| Complete semantic-input capacity census | | Full Release rebuild; 1,533/1,533 native tests; 403/403 verifier tests; 10,038 proof-graph checks with zero failures; 311/311 retained artifacts byte-identical | |
| Complete 24-column resident image | | Full Release rebuild; 1,533/1,533 native tests; complete CUDA upload checksum twin; 403/403 verifier tests; 10,038 proof-graph checks with zero failures; 311/311 retained artifacts byte-identical; 263.5 MiB fixed device image | |
| Native resident lookup layer | | Full Release rebuild; 1,533/1,533 native tests; exact native hits, misses, and payloads for names, all 10 byte maps, the reverse map, and all 7 plain-data maps | |
| Derived mandatory-membership indexes | | Full Release rebuild; 1,533/1,533 native tests; 403/403 verifier tests; 10,038 checks with zero failures; 311/311 artifact identity; all 9 plain-data views probed on-device | |
| Pointer-free Phase 2 task image | | Full Release rebuild; 1,534/1,534 native tests; exact normal and counter-example batches, ordered mandatory terms, split stumps, and fixed host allocation reuse | |
| Fixed Phase 2 task capacity and CUDA upload | | Retained FTA census and 311/311 artifact identity; 0.70 MiB fixed task image; Full Release rebuild; 1,535/1,535 native tests including exact four-column device upload | |
| Global filter/sort capacity census | | Full Release rebuild; 1,535/1,535 native tests; 403/403 verifier tests; 10,038 checks with zero failures; 311/311 artifact identity; fixed ceilings for 2,048 calls, 16,777,216 examined rows, and 1,048,576 retained rows | |
| Bulk CUDA statement filter and stable ordering | | Exact production-sized fixed allocation; Full Release rebuild; 1,536/1,536 native tests; processor-identical retained counts and ordered indices for all four registries, whole-key widening, frozen ancestry, iteration cap, and equal-name ties | |
| Normalized-key growth-capacity census | | Full Release rebuild; 1,536/1,536 native tests; 403/403 verifier tests; 10,038 checks with zero failures; 311/311 artifact identity; fixed ceilings for two 1,048,576-record frontiers, 2,097,152 accepted events, and 131,072 raw requests | |
| CUDA normalized-key and owner-subkey primitive | | Full Release rebuild; 1,537/1,537 native tests; exact key payload and accepted, present-but-u_-rejected, absent, and short presence-only owner verdicts on the RTX | |
| Fixed CUDA normalized-key growth arenas | | Full Release rebuild; 1,538/1,538 native tests; two production frontiers plus accepted-event and raw-request ledgers allocated and released on the RTX; exact fixed ownership 215 MiB | |
| Bulk CUDA request frontier content | | Clean native `sm_89` Release rebuild; 1,539/1,539 tests; exact unsplit and mixed-depth stump content across every generation gate; depth-homogeneous waves; fixed growth ownership 227,565,584 bytes | |
| Exact CUDA request stream reconstruction | | Clean native `sm_89` Release rebuild; 1,540/1,540 tests; exact stack-walk event order, task-local growth positions, mixed-stump ordering, duplicate-row coverage, and collision-checked per-call first-occurrence deduplication; fixed growth ownership 263,217,168 bytes | |
| Request-evaluation capacity census | | Clean Release rebuild; 1,540/1,540 tests; processor shortcut passed 403/403 verifier tests and 10,038 checks; 311/311 artifacts byte-identical; retained fixed evaluator work/output ceilings from 26 sweeps | |
| Fixed CUDA evaluator and request-to-LMV expansion | | Clean native `sm_89` Release rebuild; 1,540/1,540 tests; exact dependency/validity gates, reverse expansion, subset rejection, ignore-u mapped encoded hits, and multi-value LMV expansion; complete production evaluator owner below 64 MiB | |
| CUDA firing-expression materialization | | Clean native `sm_89` Release rebuild; 1,540/1,540 tests; exact reverse substitution, repeated token-leading `u_` removal, deeper rule-scope filtering, marker/demand purity, ordis2 depth, source flags, fixed byte/header output, and an impure-marker rejection twin | |
| CUDA firing levels and head provenance | | Clean native `sm_89` Release rebuild; 1,540/1,540 tests; 12 exact sorted-unique level values and 12 exact source-first, decoded-premise-ordered provenance dependencies across four GPU heads; corrected deterministic evaluator base 35,164,192 bytes | |
| Complete CUDA firing verdicts and marker payloads | | Clean native `sm_89` Release rebuild; 1,540/1,540 tests; four exact heads plus one exact marker matched all verdict flags and every variable payload; deterministic evaluator base 35,688,492 bytes and complete allocation 35,690,027 bytes | |
| Canonical CUDA firing-record order | | Clean native `sm_89` Release rebuild; 1,540/1,540 tests; exact five-record processor-equivalent order with a marker-last case; fixed parallel index-array merge; deterministic evaluator base 37,261,356 bytes and complete allocation 37,262,891 bytes | |
| Exact CUDA doom-prefix selection | | Clean native `sm_89` Release rebuild; 1,540/1,540 tests; no-doom and compressor retention, sole-goal and contradiction position-3 exits, plus exact request-2/request-3 shared-position separation; deterministic evaluator base 37,267,500 bytes and complete allocation 37,269,035 bytes | |
| Processor-only canonical GPU record sealing | | Clean native `sm_89` Release rebuild; 1,541/1,541 tests; fixed host downloads and exact five-record sealing twin; existing processor sort preserved and explicit device-canonical bypass pinned | |
| Exact CUDA split-work counts and current projection re-audit | | Clean native `sm_89` Release rebuild; 1,541/1,541 native tests; 403/403 verifier tests; 10,038 checks with zero failures; 311/311 artifact identity; subkey-only task counts; corrected eleven-view projection peaks and 5,632-view ceiling | |
| Explicit live CUDA FTA Phase 2 route and identity | | Clean native `sm_89` Release rebuild; 1,542/1,542 native tests; 403/403 verifier unit tests; 48 CUDA Phase 2 passes; 10,038 proof-graph checks, zero failures; 311/311 artifact byte identity; no fallback | |
| Exact Phase 2 and CUDA device timing | | Clean native `sm_89` Release rebuild; 1,543/1,543 tests including reusable CUDA-event timing of real native work; paired processor/CUDA shortcut measurement later established 110.333 versus 228.782 seconds for complete Phase 2 and retained exact artifacts | |
| Processor-equivalent request-growth prefix hoist | | Clean native `sm_89` Release rebuild; 1,543/1,543 tests; optimized shortcut 78.845-second complete Phase 2 and 192.096-second end-to-end runtime; 10,038 verifier checks with zero failures; 305/305 current artifacts byte-identical | |
| Counter-guided growth local-state reduction | | Clean native `sm_89` Release rebuild; 1,543/1,543 tests; 63 registers and 3,184-byte compiled stack; 25.178-second complete Phase 2 and 161.911-second end-to-end runtime; 10,038 verifier checks with zero failures; 305/305 current artifacts byte-identical | |
| Exact CUDA-route host attribution | | Clean native `sm_89` Release rebuild; 1,543/1,543 tests; 48-pass attribution reconciled to route wall time within 3 microseconds; 10,038 verifier checks with zero failures | |
| Exact fixed CUDA ownership telemetry | | Clean native `sm_89` Release rebuild; 1,543/1,543 tests; all six process-static owners report exact allocation requests totaling 704,555,605 bytes; 48 CUDA passes; 10,038 verifier checks with zero failures; 305/305 current artifacts byte-identical | |
| Projection and traffic attribution telemetry | | Clean native `sm_89` Release rebuild; 1,543/1,543 tests; 6.905-second projection versus 0.073-second scheduling split; complete Phase 2 22.948 seconds; exact traffic peaks; 10,038 verifier checks with zero failures; 305/305 current artifacts byte-identical | |
| Projection-stage attribution and packed-name sorting | | Clean native `sm_89` Release rebuild; 1,543/1,543 tests; name sort 4.304 -> 1.936 seconds; projection 6.754 -> 4.308 seconds; complete Phase 2 19.777 seconds; 10,038 verifier checks with zero failures; 305/305 current artifacts byte-identical | |
| Prefix-collapsed fixed-capacity name radix | | Clean native `sm_89` Release rebuild; 1,543/1,543 tests; name sort 1.936 -> 1.545 seconds; projection 4.308 -> 3.987 seconds; complete Phase 2 19.257 seconds; 10,038 verifier checks with zero failures; 305/305 current artifacts byte-identical | |
| Dual-native Ada/Blackwell build and stronger-GPU model | | Clean Release rebuild; native `sm_89` and `sm_120` cubins for both CUDA translation units; 1,543/1,543 tests on RTX 4070 Laptop; model corrected forward in the closeout snapshot after the live-device audit rejected its desktop-4070 baseline | |
| Initial-port boundary and corrected live-device model | | Live backend-door audit; Windows FTA shortcut boundary; CPU default and no fallback; explicit processor/GPU ownership split; non-shortcut work recorded; exploratory draft superseded by measured 19.257-second Phase 2, 5.730-times processor acceleration, 10,038 zero-failure checks, and 305 exact artifacts | |
| Main integration and full processor identity |, | Real main merge on the source branch; clean rebuild; 1,557/1,557 native tests; exact-main and merged-GPU processor runs each passed 139,241 checks; all 5,040 retained semantic files were byte-identical | |
| Current-main capacity integration and final shortcut gate |,, | Owner-bearing blob and forward-map capacities raised from paired processor census only; final processor/CUDA Phase 2 96.212/19.835 seconds; overall 203.225/127.555 seconds; 9,994 zero-failure checks and 305 exact semantic artifacts | |
| Exact-tree main squash | | Sole parent; tree exactly equals preserved; `git diff --check` clean; source branch retained locally and on origin | `origin/main` |
| Final SwDD identifier allocation | | Separate documentation commit allocated D-313 and I-211, advanced both navigation high-water marks, and left zero pending identifiers | `origin/main` |

## Completed implementation handoff record

This section retains the compaction history; the implementation
and landing are complete, so its intermediate profiling and next-action statements
are evidence chronology rather than open work. The two statification cookbooks and
`docs/agentic_swdd/20_core_concepts/09d_gpu_cookbook.md` own any future extension.
The corrected census, complete 24-column resident image,
native upload twin, and processor byte-identity gate passed. The pointer-free
part/batch, mandatory-term, and stump image has measured fixed host/device
ownership and no resident state is missing. The
production-sized global filter/scan/sort allocation and exact processor-order
twin pass on the RTX. The retained growth census fixes two 1,048,576-record
frontiers, 2,097,152 accepted-event records, and 131,072 raw request records.
The native normalized-key and owner-subkey primitive passes its direct RTX twin.
The compact 56-byte node shape and all request-growth arrays now own fixed
263,225,360-byte device storage. Bulk seeding and expansion are live with exact
semantic content. Eleven stable radix passes, task-segmented growth scans, and
 collision-checked minimum-order request deduplication reconstruct the exact
 processor stream entirely on the device. The ordered stream is connected to
 device evaluation. The retained evaluation census fixes 131,072 requests,
2,097,152 reverse-owner work items, 65,536 candidate owners and firing records,
131,072 local values, 2,097,152 generated bytes, and the documented provenance
runs. The fixed evaluator and its dependency/validity, reverse/subset,
mapped-encoded, and local-value expansion kernels are live. The firing-expression
kernel also reconstructs reverse maps, performs exact greedy substitution and
repeated u_ stripping, folds rule validity, applies closed-scope, purity, and
ordis2 depth gates, and writes source-classified expression headers. It now also
 merges exact sorted-unique premise/rule levels and writes complete ordered head
 provenance. Final head verdicts and full marker payloads pass their direct RTX
 content twin. Canonical per-logical-block device-record ordering and exact
 doom-winner prefix reduction are live through fixed arrays. Fixed host-only
 sealing now converts that stream into complete ordinary `FiringRecord` chains,
 and the canonical deposit door avoids a second processor sort. The live FTA
 shortcut route is selectable through `--phase2-backend cpu|cuda`; CUDA executes
 48 real device passes with no fallback and produces all retained artifacts
 byte-identically with 10,038 verifier checks and zero failures. Whole-prover and
 original overall time was slower. The explicit complete-Phase-2 timer measured
 110.333 processor seconds versus 228.782 initial CUDA seconds, while the initial
 end-to-end shortcut measured 218.442 versus 343.659 seconds. Nsight assigned
 217.594 seconds, 99.0% of CUDA kernel time, to frontier growth. Hoisting the
 exact processor-style prefix summaries out of the CUDA per-candidate loop then
 reduced exact CUDA-owned time to 71.034 seconds, complete Phase 2 to 78.845
 seconds, and end-to-end shortcut time to 192.096 seconds. This is a 1.399-times
 Phase 2 and 1.137-times end-to-end acceleration over the paired processor run,
 with 305/305 current proof artifacts byte-identical. Profile the optimized
 shape next and close memory, utilization, and remaining bottlenecks. The
 optimized Systems profile confirms growth still owns 68.320 seconds and 97.6%
 of CUDA kernel time, while transfers remain below 0.35 seconds. Nsight Compute
 hardware counters now prove that one representative launch uses 65 registers
 and a 5,024-byte thread stack, reaches only 43.89% occupancy and 8.04% compute
 throughput, and is latency-bound. Removing the duplicated prefix payload, its
 per-candidate copy, and two appended-expression arrays lowers the compiled
 shape to 63 registers and a 3,184-byte stack, raising theoretical occupancy to
 66.67%. The complete rerun now measures 15.929 seconds exact CUDA-owned work,
 25.178 seconds complete Phase 2, and 161.911 seconds end to end: 4.382 and 1.349
 times faster than the paired processor run with 305/305 artifacts exact. The
 new profile puts growth at 13.005 seconds and 87.0% of kernel time, all transfers
 near 0.525 seconds, representative achieved occupancy at 54.35%, compute
 throughput at 46.49%, and DRAM throughput at 1.96%. Route attribution then
 assigns 8.270 seconds to host projection/scheduling, 15.684 seconds to the CUDA
 segment containing the 15.672-second exact event interval, and 0.283 seconds to
 download/sealing; the three intervals reconcile to 24.238 seconds within 3
 microseconds. Exact allocation telemetry now proves that the six process-static
 CUDA owners total 725,527,125 bytes (691.92 MiB, 8.45% of the device):
 299,951,120 projection, 731,144 tasks, 25,512,447 filtering, 263,225,360
 growth, 98,838,019 ordering, and 37,269,035 evaluation. The refined 48-pass
 run measured complete Phase 2 at 22.948 seconds, including 6.905 seconds of
 host projection and only 0.073 seconds of scheduling. Its peak 154,845
 provenance dependencies occupy 1.181 MiB; mail eligibility stays a flag on
 the required firing record and creates no separate GPU queue. Projection-stage
 telemetry then isolated decoded-name lexical sorting at 4.304 seconds of a
 6.754-second projection. Comparing the already-packed name spans instead of
 repeatedly decoding the cold table reduced sorting to 1.936 seconds, projection
 to 4.308 seconds, and complete Phase 2 to 19.777 seconds. The prefix-collapsed
 fixed-capacity radix sorter then reached 1.545 seconds of sorting, 3.987 seconds
 of projection, 14.332 seconds of exact CUDA work, and 19.257 seconds of complete
 Phase 2. The clean Release build, all 1,543 native tests, 10,038 verifier checks,
 and 305-artifact SHA-256 comparison pass. In-process projection caching is closed
 for FTA shortcut because each fresh prover subprocess projects disjoint first-pass
 non-stragglers and second-pass stragglers, then exits. is
 pushed. The corrected stronger-GPU model predicts 8.902-to-10.052-second Phase 2
 on an RTX 5090 under 3-to-4-times device scaling, with an 8.090-second aggressive
 rated-spec sensitivity point; these
 are not benchmark claims. The executable now contains native `sm_89` and `sm_120`
 cubins for both CUDA translation units and passes all 1,543 tests on the RTX 4070.
 Dual-architecture commit and initial-port closeout are
 pushed. The exact landing and final branch/remote audits are recorded below.

## Main integration and squash landing — complete 2026-08-28

Resolved source and destination are -> `main`. Current `main`
and `origin/main` were both;
the clean GPU source was.
The required first operation is a real merge of that main tip, preserving both histories on the source branch. The merge had
two documentation-only conflicts because main finalized D-312/I-210 while the
GPU branch carries its own pending D/I entries. Resolution retains both and
leaves the GPU entries pending until the destination-only renumber commit.

Landing gates, in order:

1. Complete the merge commit, full-rebuild it, run all native
 unit tests, and push the merged source branch.
2. Run the full processor pipeline at exact pre-merge `main` and at merged
; retain independent artifacts and require exact byte identity
 plus zero verifier failures.
3. Run processor and CUDA FTA shortcut paths on merged; retain
 independent artifacts and require exact byte identity, the same theorem
 set, and zero verifier failures. CUDA transfer time remains included in the
 exact CUDA interval.
4. Sweep `docs/user_swdd/` and `docs/MPU/index.html` against the merged behavior;
 change them only if an existing claim or diagram is stale.
5. Create a plumbing squash commit whose tree is exactly the merged
 tree and whose sole parent is the verified main tip; prove an
 empty tree diff before advancing and pushing `main`. Preserve
 locally and on origin.
6. Run the SwDD renumber tool on `main` and commit only the pending-identifier
 promotion as a separate documentation snapshot; then prove final refs,
 ancestry, source preservation, and clean relevant worktrees.

Final status: the full clean Release build
 passes, and
 records 1,557/1,557
native tests passing. The user SwDD accurately describes the explicit processor
and CUDA backends, complete GPU provenance, split-block processor stump producer,
canonical deposit, no fallback, and the existing shortcut identity gate. The MPU
booklet remains accurate because its read-only Phase 2, expression split, and
canonical merge diagrams are backend-independent. The real main-to-GPU merge is
commit and is pushed. Exact-main full processor run
 passed 139,241 checks with zero failures
in 723.55295 seconds; merged-GPU processor run
 passed the same 139,241 checks with
zero failures in 719.26016 seconds. The retained ordinary and incubator theorem,
raw, processed, and full proof-graph roots contain 5,040 files and 665,372,565
bytes on each side; complete SHA-256 comparison found zero missing, extra, or
different files. The final same-tip shortcut processor/CUDA pair passed 9,994
checks with zero failures and produced 305 byte-identical semantic artifacts.
Complete Phase 2 measured 96.212168 processor seconds versus 19.835229 CUDA
seconds, while overall runtime measured 203.22474 versus 127.55500 seconds. The
source branch finished and remains on origin. Its exact tree was
squashed onto `main` as, followed only by the separate SwDD allocation
commit for D-313 and I-211; no pending identifiers remain.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
