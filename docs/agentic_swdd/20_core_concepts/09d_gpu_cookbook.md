<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — GPU cookbook `[DRAFT]`

> You have a processor Phase 2 type or operation and must give it a GPU form
> without changing proof semantics. This is the practical lookup: classify the
> state, choose the device representation, preserve the exact semantic order,
> and compare against the processor oracle. Both the FTA shortcut and the
> complete `main.py` full pipeline have explicit processor and CUDA backends;
> the processor route is the product default, `main.py --GPU` selects CUDA
> for the whole run, a batch config may pin one batch through `use_gpu`, and
> `--phase2-backend` remains the explicit override. Every later expansion beyond the approved
> full-run boundary remains an architecture decision rather than an implicit
> widening of this implementation.
>
> Read the container and string cookbooks first:
> [`09b_statification_cookbook.md`](09b_statification_cookbook.md) and
> [`09c_string_statification_cookbook.md`](09c_string_statification_cookbook.md).
> The persistent implementation plan and evidence ledger are
> [`docs/GPU/gpu_plan.md`](../../GPU/gpu_plan.md).

---

## Definition — GPU-ported means exact semantic work on the device

A Phase 2 operation is GPU-ported only when all of the following are true:

- the combinatorial operation and its semantic gates execute on the GPU;
- the GPU reads a bounded, pointer-free projection whose ownership and lifetime
 are explicit;
- the GPU emits every datum needed to reconstruct the ordinary
 provenance-bearing `FiringRecord` without processor-side semantic replay;
- the selected GPU backend never falls back to the processor backend;
- queue, byte-arena, transfer, launch, and device failures assert at their
 owning boundary;
- the processor implementation remains the oracle and selectable peer;
- processor and GPU proof artifacts are byte-identical at the agreed identity
 gate.

Launching a kernel around processor-generated requests is not a Phase 2 port.
Returning only theorem heads while provenance is reconstructed by re-running
the rule on the processor is not a Phase 2 port. Host-side ownership conversion
is allowed: decoding an already-decided identifier and sealing already-produced
bytes into `SealedPageSet` does not repeat semantics.

---

## Step 0 — classify every byte before moving it

| Class | Meaning | Device rule |
|---|---|---|
| Process immutable | CUDA context, device identity, kernel code, allocation metadata | Create once for the selected GPU backend; destroy once. |
| Batch immutable | Compiled operator metadata and other configuration shared by the batch | Upload once per batch and address by integer identifier or offset. |
| Logical-block snapshot | Phase 2 reads that belong to one resident logical block | Pack into a pointer-free image with an explicit generation; never expose host arena pointers. |
| Iteration delta | State changed by Phase 1 before the Phase 2 barrier | Upload only the changed ranges or replace the affected compact table; record transferred bytes. |
| Task scratch | Frontier nodes, candidate keys, request deduplication, hit indexes | Allocate from pre-sized device scratch buffers; reset cursors, never allocate per request. |
| Semantic output | Complete firing decisions, generated strings, provenance identifiers, flags, doom evidence | Write to bounded device record and byte arenas; copy the compact used prefix back. |
| Host-only finalize | `SealedPageSet` ownership conversion and `applyFiringRecords` canonical replay | May copy/decode device results, but may not redo request gates or rule evaluation. |

If a value does not fit one row, stop and define its lifetime. A host pointer,
`std::string`, cold-page address, `StrSpan`, `SealedString`, or arena cursor is not
a device interchange representation.

---

## Phase 2 conversion table

This table maps the live processor path to the approved device forms. Rows are
implemented incrementally; the implementation state immediately below names
the exact live subset.

| Processor type or role | Proposed device form | Exactness rule | Live source |
|---|---|---|---|
| `IntEncodedExpr` statement pages | Contiguous `IntEncodedExpr[]` plus logical-block slice | Copy field-for-field; the 352-byte record is already flat and trivially copyable. | `memory_infra/int_encoded_expr.hpp · IntEncodedExpr` |
| `IntStmtView` | `(offset, count)` into the statement array | Offset is relative to the owning projection, never a device or host pointer persisted in a record. | `memory.hpp · IntStmtView` |
| `ExpressionStump` bucket | Contiguous stump array plus `(offset, count, ordinal, total)` | Preserve statement indices, `terminalOnly`, bucket order, and sibling ordinal exactly. | `memory.hpp · ExpressionStump`, `SplitStumpRef` |
| `NameMap` strings | `stringOffsets[]`, `stringBytes[]`, byte-key lookup table, validity-parent runs, decoded-lex rank per identifier | Identifier equality stays identifier equality; every observable order uses decoded-byte rank, never mint order. | `memory.hpp · NameMap` |
| `ValueInterner` rule strings | Separate offsets, bytes, lookup table, and decoded-lex rank | Never mix its identifier space with `NameMap`; substitutions read exact stored bytes. | `memory.hpp · ValueInterner` |
| `normalizedEncodedKeys` | Read-only open-addressed table from normalized-key bytes to presence | Probe uses the exact `Codec<NormKey>` byte layout and hash; fixed load factor asserts during projection build. | `memory_infra/hash_memory.hpp · HashMemory::normalizedEncodedKeys` |
| `normalizedEncodedSubkeys` | Read-only key table plus packed `OwnerSetBlob` values | `subkeyUSatisfied` must reproduce key presence and every owner signature gate. | `memory_infra/hash_memory.hpp · HashMemory::normalizedEncodedSubkeys` |
| `encodedMap` | Read-only normalized-key table to a run descriptor; `LocalMemoryValue` blobs in one byte pool | Preserve every blob byte and value-run membership; value order is by decoded head bytes where observed. | `memory_infra/hash_memory.hpp · HashMemory::encodedMap`, `memory.hpp · LmvBlobView` |
| Remaining-argument forward and reverse indexes | Packed key runs and normalized-key-to-owner runs | Candidate owners are filtered by exact subset semantics and ranked by signed identifier-set lexicographic order. | `HashMemory::remainingArgsNormalizedEncodedMap`, `remainingArgsReverseIndex` |
| Packed statement membership sets | Read-only fixed-load open-addressed tables keyed by `(originalId, validityId)` | Preserve the distinction among known, local, level-carrying, and registered state. Map presence may not replace the correct flag. | `Memory::intKnownStatements`, `intLocalEncodedStatementsSet`, `intStatementLevelsMap` |
| `intValidityNamesToFilter` and validity ancestry | Packed identifier set plus parent/ancestor representation | Both consensus and deeper rule scopes must perform the same ancestor filter walk. | `checkLocalEncodedMemoryStatic` |
| `productsOfRecursionIds` | Read-only identifier set | Pure/impure classification reads every iteration-bearing argument exactly as the processor path does. | `HashMemory::productsOfRecursionIds` |
| `MandatoryTerm` views | Up to two term masks over projected statement slices | Term satisfaction and suffix reachability stay bit-identical to the processor masks. | `memory.hpp · MandatoryTerm`, `generateEncodedRequestsStatic` |
| `StaticRequest` | Inline statement indices, count, maximum iteration, normalized-key offset and length, canonical stream token | No device pointer escapes. The token identifies the processor-equivalent batch/stump/candidate position. | `memory.hpp · StaticRequest` |
| `FiringRecord` | Fixed device record plus offsets into device integer, identifier, and byte output arenas | Device computes the full content; host only seals bytes/identifiers into the existing record shape. | `memory.hpp · FiringRecord` |
| `SealedPageSet` | Remains a host ownership boundary | CUDA output is converted after synchronization; production kernels never write host sealed pages. | `memory_infra/sealed_pages.hpp · SealedPageSet` |

The first projection should be logical, not a byte copy of the host cold-memory
pages. Cold pages contain host-specific directories and scattered addresses;
the GPU needs compact arrays and offsets. Projection twins must prove that the
logical rows, runs, decoded bytes, and identifier spaces are identical.

### Implemented projection foundation

[`phase2_projection.hpp`](../../GL_Quick_VS/GL_Quick/src/gpu/phase2_projection.hpp)
now defines the complete resident pointer-free image:

- arena-relative logical-block descriptors for all 24 columns;
- byte-identical contiguous `IntEncodedExpr` rows in original statement-index
 order;
- one name record per local `NameMap` identifier plus identifier-zero sentinel,
 carrying canonical byte offset/length, validity parent identifier, and
 decoded-byte lexical rank, plus a fixed-load byte lookup table;
- rule-interner records and bytes preserving their separate identifier space;
- ten typed byte-map views over canonical keys, open-addressed slots, raw value
 blobs, and blob-run descriptors;
- one derived normalized-key-to-forward-owner reverse map per logical block,
 rebuilt from the remaining-argument forward blobs and deduplicated exactly;
- eleven typed plain-data views covering the nine resident inputs (recursion
 products, known statements, local membership, statement levels, validity
 filtering, frozen OR branches, goals, mail-eligible statements, and
 mail-eligible markers) plus derived fixed-load membership indexes for the delta
 and external mandatory statement views;
- the three ordered mandatory statement-key slices, expression-key bytes, four
 maximum-key lengths, and the contradiction-role scalars needed by the doom
 predicate;
- a host arena whose vectors reserve fixed ceilings once, assert before every
 append, and retain their allocations across `clear`;
- twelve process-owned projection shards that claim independent logical blocks in
 parallel, followed by `Phase2ProjectionArena::appendShard`, which bulk-copies
 all 24 used prefixes, rebases every nested offset and slot index, and restores
 the canonical logical-block descriptor order before upload;
- a CUDA owner whose 24 device arrays allocate once, accept only bounded used
 prefixes, never grow during upload, and are all covered by the transfer twin;
- a dependency-staged upload door whose fixed page-locked arrays queue the nine
 evaluation-only column prefixes on a nonblocking stream while the host builds
 task/filter/growth schedules; the join occurs before the first kernel, and the
 fifteen early columns retain their ordinary synchronous copy; the staging owns
 31,219,712 host bytes while fixed device ownership is 1,275,332,748 bytes
 after the measured pooled-prefix and candidate-window growth owner;
- native fixed-load probes for decoded names, all ten byte-map views, the
 remaining-argument reverse map, and all eleven plain-data views, using the same
 FNV-1a or SplitMix64 hash and linear-probe contract as the host builder.

The direct twins compare host rows, decoded bytes, parent links, ranks, map
kinds, key and run slices, scalar flags, reverse ownership, allocation reuse,
and a device-computed checksum over every uploaded prefix. A direct lookup probe
also verifies name hits and misses, every byte-map kind, reverse owner runs, and
every plain-data kind on native device code. Probe inputs use a small inline key
only for the diagnostic launch; production request kernels call the underlying
lookup primitives on task-arena spans and do not inherit that envelope.
Expression stumps and mandatory-term combinations are per-task inputs and join
request generation; provenance-bearing firing output joins request evaluation.
Neither is resident logical-block state.

`gpu/phase2_projection.hpp::Phase2TaskProjectionArena` now gives those task
inputs their own fixed, pointer-free host image. One `DevicePhase2Task` selects a
resident logical block and owns slices of ordered `DeviceRequestBatch` records
and exact `DeviceExpressionStump` copies. Each batch selects overall, local,
local-delta, or working hash memory by enumeration and owns zero to two ordered
mandatory terms; every term names one or two of the resident local, delta, and
external membership indexes. Split ordinal/total, terminal-only markers, the
iteration cap, and the counter-example route are explicit scalar fields. The
arena reserves once and reuses its storage across clears.
`gpu/phase2_projection.cpp::measurePhase2TaskProjectionUsage` counts the exact
normal or counter-example batch shape without allocating or changing proof
state. `Prover::proveKernel` aggregates it in `[GPU-TASK-CAPACITY]`; the retained
FTA sweep peaks are 497 tasks, 1,953 batches, 1,987 mandatory terms, and 9,823
stumps, with per-part maxima of one, four, four, and 55 respectively. The fixed
ceilings are 512, 2,048, 2,048, and 16,384 records, occupying about 0.70 MiB.
`gpu/phase2_cuda.cu::CudaPhase2TaskBuffer` allocates those four device columns
and its checksum word once, accepts only bounded used prefixes, and proves the
transfer byte-for-byte on native CUDA. Request kernels are the next boundary;
the live processor does not consume this image yet.

`Prover::proveKernel` emits one deterministic `[GPU-CAPACITY]` census for each
Phase 2 sweep. The first executor task to claim a logical block records that
post-Phase-1 read image; an atomic per-block latch collapses split siblings and
the optional second pass. The line reports totals and per-block maxima for
encoded statements, name records (including record zero), and canonical name
bytes. These values size the fixed projection from measured FTA evidence. The
census is observation only: no proof decision, firing, or allocation branch
reads it.

The retained FTA shortcut census measured sweep peaks of 306 logical blocks,
129,399 statements, 348,066 name records, and 11,811,329 canonical-name bytes.
The first fixed FTA batch therefore reserves the next power-of-two ceilings:
512 logical blocks, 262,144 statements, 524,288 name records, and 16,777,216
name bytes. These are batch capacities, not semantic limits. An overflow asserts
until deterministic multi-batch ownership is connected; it never resizes inside
Phase 2 or falls back to the processor.

`gpu/phase2_projection.cpp::measurePhase2ProjectionUsage` is the complete
read-image capacity twin. It walks one logical block without allocation or proof
side effects and counts name lookup slots; rule-interner strings; all ten
byte-key maps and their encoded keys, value blobs, and fixed-load hash slots;
 the derived reverse map's safe no-dedup upper bound; all seven actual plain-data
 inputs, both derived mandatory-membership indexes, both mail-eligibility
 sets, and their value runs;
 mandatory statement-key sets; and logical-block metadata. `proveKernel`
 aggregates those rows into one
`[GPU-SEMANTIC-CAPACITY]` line per sweep using the same first-task latch. The
census is not a proof input.

The corrected retained FTA shortcut census measured these component-wise sweep
peaks: 997,376 name slots; 117,807 rule-string records and 7,423,016 bytes;
3,060 forward byte-map views, 407,031 entries, 1,051,000 hash slots, and
61,482,816 key bytes; the post-main owner-bearing representation raises the
current value-blob peak to 19,929,121 bytes; 306 reverse
views bounded by 56,462 entries and owners, 154,624 hash slots, and 9,220,736
 key bytes; 3,366 plain-data map views, 444,461 entries, 1,245,796 hash slots,
and 409,420 run values; 144,890 mandatory statement keys; and 8,945 metadata
bytes. The matching fixed ceilings are maintained in
[`docs/GPU/gpu_plan.md`](../../GPU/gpu_plan.md); all 24 arrays together reserve
263.5 MiB. A narrow margin triggers deterministic batching when exceeded; it
does not justify a live resize or processor fallback.

---

## Request generation — preserve the search, recover parallelism

The processor `[GPU-FILTER-CAPACITY]` twin measures the exact fixed-cost prefix
before growth: every request-generator filter plus every stump-producer filter,
their full examined statement slices, retained prefixes, and per-call maxima.
The retained FTA peaks are 1,959 calls, 10,481,129 examined rows, and 955,583
retained rows in one sweep; one call examined 20,486 and retained 1,752. The
first global fixed ceilings are 2,048 calls, 16,777,216 examined rows, 1,048,576
retained rows, and 32,768 examined rows per call. The observed retained maximum
rounds to 2,048 per call, but it is not an allocation boundary: the CUDA kernel
preserves the processor's first-8,192 acceptance envelope and only the global
retained prefix has a fixed arena ceiling. The schedule processes calls in one
bulk launch family and orders retained indices by decoded-name lexical rank then
original statement index. The precomputed name rank makes that composite key
exactly equivalent to the processor's stable name-only sort without comparing
variable-length strings inside the sort.

`gpu/phase2_projection.hpp::Phase2FilterScheduleArena` owns both the fixed host
call column and its exact class image. Each call descriptor selects one resident
logical block, one of the four whole/subkey registry pairs, the iteration
ceiling, the processor whole-key widening flag, and the statement count. A
fixed open-address table interns all five fields, checks equality after hashing,
preserves first-occurrence class order, and stores one class index for every
original processor call. The complete FTA shortcut contains 24,229 calls but
only 8,853 exact classes: class representatives examine 8,178,826 rows instead
of 89,618,957, a 10.957-times reduction in predicate rows.

`gpu/phase2_cuda.cu::CudaPhase2FilterSortBuffer` owns original calls, class
calls, the original-to-class map, original and class count/offset columns, two
compact 64-bit retained-key arrays, and one shared CUB scratch allocation. Count,
exclusive scan, predicate replay, stable compact emission, and radix sort run
once per class. A fixed expansion kernel maps the class count and offset back to
every original call without copying retained rows. The global sort key is class
ordinal, decoded-name rank, and original statement index; growth decodes only
the low statement-index field from the shared span, while its events and every
downstream ordering token retain the original processor call ordinal. Equal-name
stability is explicit rather than dependent on kernel arrival order. The
predicate directly covers frozen-scope ancestry, exact one-expression
normalized-key bytes, all four registry pairs, whole-key widening, and the
inclusive iteration cap. The direct twin changes every signature field one at a
time, exercises a duplicate class on native CUDA, and proves shared counts and
offsets plus distinct original-call events. See
[D-317](../40_decisions.md#d-317)
and [I-212](../30_invariants.md#i-212).

The observation-only `[GPU-GROW-CAPACITY]` twin at
`prover.cpp::ExpressionAnalyzer::proveKernel` resets six thread-local depth
ledgers for every executor task, lets
`memory.cpp::ExpressionAnalyzer::generateEncodedRequestsStatic` and
`produceExpressionStumps` increment them at the live gates, and publishes their
integer sums only after the task returns. No proof branch reads a census value.
The retained FTA shortcut peaked at 358,565,795 expansion attempts, 1,596,160
accepted owner-subkey events, 1,584,375 growable frontier pushes, and 72,674 raw
whole-key requests in one sweep. Depth two was largest: 1,025,308 accepted events
and 1,024,936 frontier nodes. The run performed 4,077,562,830 expansion attempts,
which is the directly parallel work pool rather than a queue to materialize.

The matching fixed ceilings are two 1,048,576-record frontier arrays, a
2,097,152-record accepted-event ledger accumulated across depths, and 131,072
raw request records. Kernels derive expansion attempts from frontier rows and
their compact filtered spans; allocating a row for every attempt would reserve
memory for the wrong abstraction. The stump producer stays bounded by the task
image: 9,823 singleton survivors, 1,275 depth-two attempts, and 660 depth-two
survivors at its sweep peaks. Capacity assertions preserve these FTA limits;
larger workloads use the approved deterministic batch boundary.

`gpu/phase2_cuda.cu::buildProjectedNormalizedKey` is the first live device
growth primitive. It folds projected statement rows into the exact sequential
first-occurrence `NormKey` payload without storing key bytes in frontier records.
`projectedOwnerSetUSatisfied` consumes the matched subkey entry's canonical
single `OwnerSet` blob and compares flattened `argFullId` values from three
premises onward; shorter keys remain presence-only. The direct
`CudaPhase2ProjectionBuffer::launchGrowthCandidateProbe` twin returns the full
device-built key plus separate subkey-presence, owner-satisfaction, and whole-key
flags. Its accepted, present-but-u_-rejected, absent, and short-key cases pass on
the RTX. `CudaPhase2GrowthBuffer::runRequestGrowth` now reuses those helpers in
the bulk frontier; the probe itself remains a fixed-scratch semantic twin, not a
processor fallback.

The persistent frontier records do not store normalized-key payloads.
`DeviceGrowthNode` remains 56 bytes: call and stump-run identity, the eight-
position filtered path, next start, folded validity, and mandatory-term mask.
`DeviceAcceptedGrowthEvent` copies the 64-byte statement-path/verdict identity
plus eight compact filtered-position codes before its source frontier is
reused. Each code is filtered position plus one, with zero reserved for a
whole-only stump element that has no filtered position.
`DeviceRawGrowthRequest` adds an eight-byte growth position for a 72-byte
record.

One preparation kernel builds an immutable 32-byte `DeviceGrowthPrefix` for
every live node. The header points into three fixed global pools containing the
exact normalized payload, distinct normalization variables, and distinct
secondary variables, and packs the hypothesis, non-exempt-scope, and mandatory-
term summaries. Candidate kernels read that header and fold only the appended
expression suffix. Pool placement order is not a semantic key. Two fixed flag
arrays plus stable CUB selection produce complementary short and cooperative
node lists in ascending original-frontier index order; atomic block arrival is
not used to construct either list. This contract is
[D-318](../40_decisions.md#d-318)
and
[I-213](../30_invariants.md#i-213).

The complete prefix census measured 14,074,478 payload values, 4,369,164
normalization variables, and 951,515 secondary variables at the simultaneous
frontier peaks. Production applies 25 percent headroom and rounds each upward
to 262,144-value blocks: 17,825,792, 5,505,024, and 1,310,720 values, or exactly
94 MiB across the three pools. With one header per frontier node, canonical
node-order arrays, two index lists, two flag arrays, fixed candidate-window
storage, and retained CUB scratch, `CudaPhase2GrowthBuffer` owns exactly
812,965,447 bytes (775.304 MiB). The task
counters increment
only for owner-accepted subkeys, matching `g_growthMatchCount` without counting
whole-key-only evaluation events or downloading the event ledger. A
production-sized RTX test allocates and releases the complete
owner and asserts the exact byte total. Copying maximum-size prefix payloads
into every 56-byte frontier row remains forbidden because it would dominate
device memory.

Production candidate growth is fixed and complete. Stable radix passes order
node spans by original call, run, statement path, and candidate; one inclusive
scan assigns exact candidate ordinals. Consecutive half-open windows of at most
16,777,216 records cover every ordinal exactly once. Short-thread and 64-lane
cooperative kernels apply only the map-independent gates, stable flagged
selection compacts survivors in ordinal order, and one suffix-only probe kernel
performs exact subkey, owner, whole-key, and term work. The retained window owns
16-byte attempts, one-byte flags, and four-byte survivor indices; every boundary
asserts and no window size can alter proof order. Reusable CUB scratch is
12,868,095 bytes and complete fixed device ownership is 1,275,332,748 bytes.
See
[D-319](../40_decisions.md#d-319)
and
[I-214](../30_invariants.md#i-214).

The shared counter record also carries an observation-only twelve-bucket census
of each live node's remaining filtered positions: zero, one, powers-of-two ranges
through 1,023, and 1,024 or more. A separate device kernel combines node and
candidate totals in per-block shared counters before twelve bounded pairs of
global atomics. No semantic kernel reads the census, and production disables the
observation kernel after profiling. The direct growth fixture pins six nodes and
nineteen attempts across the 2-to-3 and 4-to-7 buckets. The complete FTA census
counts 18,940,682 live nodes and 4,498,333,440 attempts: spans of at least 256
positions own 68.7134% of attempts across 35.0923% of nodes, and spans of at least
64 own 97.4104% across 81.2753%. The first production threshold was 256; direct
timing then selected 64 without guessing from aggregate work.

`Phase2GrowthScheduleArena` links projected task and global request-batch rows to
their already-sorted filter spans. `runRequestGrowth` builds direct mandatory
membership plus backward suffix masks, seeds one empty main-scope root or each
exact stump, and expands all live nodes through the two global frontiers. Stump
seeds run in absolute-starting-depth groups, keeping every frontier wave
depth-homogeneous; this is why the measured maximum for one depth safely bounds
the arrays even though stumps begin at different depths. The device gates match
the processor's validity ancestry/deeper-scope fold,
hypothesis agreement with main-anchor exemption, recursion-product exclusion,
distinct-secondary cap and `_orint_` widening, mandatory reachability,
normalized whole-key presence, and owner-subkey signature test. Stump-alone
whole-only requests remain representable because persistent events carry
statement indices rather than requiring every stump element to have a filtered
position.

Growth is adaptive inside each wave. `phase2GrowthExpandKernel` keeps spans below
64 positions on the one-thread-per-node path and warp-compacts longer nodes into
the fixed cooperative index list. After the compact count is read,
`phase2GrowthCooperativeExpandKernel` assigns one block to each listed node.
Thread zero reconstructs its normalized-key prefix and hypothesis, scope,
secondary-variable, and mandatory-term summaries once in shared memory; 64 lanes
own disjoint candidate positions. `findProjectedSegmentedByteMapEntry` hashes and
compares the canonical header, shared prefix payload, and lane-local expression
suffix as one byte-exact key, so no lane copies the at-most-256-identifier prefix
to its stack. The ordinary event/frontier ledgers and later canonical ordering are
unchanged. The direct fixture lowers the threshold to four so short nodes, long
nodes, presence, owner rejection, absence, event emission, and child growth all
execute in one exact CUDA test.

`CudaPhase2OrderingBuffer` now reconstructs the exact observable stream from
the unordered append ledger without processor semantic replay. Eleven stable
least-significant-first radix passes encode task, call, stump run, and all eight
path components. At each path level the direct child uses its ascending
filtered position; a continuing subtree uses a disjoint upper range with the
position inverted. This exactly represents the processor stack rule: emit all
immediate children ascending, then visit pushed children descending. A
task-segmented inclusive scan over the subkey-satisfied bit assigns the exact
cumulative growth position, including persistence across calls in one task.

The per-call deduplication table is fixed-capacity open addressing. Its slot
hash is only a placement aid: collision resolution compares the complete key —
call identifier, statement count, and every original/validity identifier pair.
An atomic minimum retains the lowest exact event order, so kernel arrival timing
cannot select the representative. The retained unique tokens are radix-sorted
by event order. At production ceilings its deterministic arrays consume
81,788,932 bytes before one reusable maximum-sized CUB scratch allocation;
`CudaPhase2OrderingBuffer::fixedAllocationBytes` reports the complete fixed
ownership, and the production RTX allocation test asserts it remains below
512 MiB.

The processor generator is not a flat Cartesian product. It:

1. filters the statement universe against the selected rule registry;
2. stable-sorts by decoded expression name, retaining statement-index order for
 equal names;
3. builds mandatory-term masks and suffix reachability;
4. traverses one unsplit search or one search per stump;
5. applies scope, hypothesis, secondary-variable, recursion-product, normalized
 subkey, and whole-key gates;
6. increments the part's growth counter only for accepted subkeys;
7. deduplicates emitted requests per generator call, keeping the first request
 in processor stream order.

The GPU may change wall-clock execution order, but it must reconstruct the same
logical sequence where order is observable. The parallel pattern is:

- represent each candidate by its statement-index and compact filtered-position
 paths — live;
- expand frontier nodes in bulk into a second preallocated frontier — live;
- compute every generation gate independently on candidates — live;
- compact survivors with prefix scans;
- assign each candidate its exact growth position by a task-segmented scan over
 `subOk` in processor-equivalent order — live;
- deduplicate requests by full semantic request key, retaining the minimum
 stream token, then order the unique stream by that token for doom-prefix
 semantics — live.

The processor's stack walk processes all children of a popped prefix in
ascending filtered position and later pops the most recently pushed prefix.
Do not call a generic depth-first or breadth-first order equivalent. The
processor-order token and its twin test must encode the actual
`generateEncodedRequestsStatic` walk.

Decoded string comparison in the hot generator should become integer
comparison against a projection-time decoded-lex rank. Rank is derived from
bytes, not identifier mint order. Equal expression names retain the original
statement index as the stable tie-break.

---

## Request evaluation and provenance

The evaluation boundary begins at the dependency check in `BurstSink::consume`
and ends when complete semantic firing records exist. GPU mode must perform:

- known-statement dependency checks;
- premise validity folding and closed-scope filtering;
- purity classification and combined-level construction;
- remaining-argument reverse lookup, subset filtering, and normalized-key
 remapping;
- encoded-map lookup and all `LocalMemoryValue` gates;
- exact greedy-longest back substitution and `u_` stripping;
- head, marker, and ordis2-demand classification;
- disintegration, mail eligibility, ancestor-known, marker-category, iteration,
 and admission flags;
- complete provenance content.

The observation-only `[GPU-EVAL-CAPACITY]` row measures the allocation shape
before the evaluator owner is fixed. It reports emitted and dependency-passing
requests; reverse-index owners, subset survivors, encoded hits, and local values;
head, marker, and ordis2-demand records; generated output bytes; level and
origin-dependency runs; marker key, remaining-argument, and bare-argument runs;
and the maximum reverse owners per request, candidates per request, and local
values per encoded hit. Executor tasks accumulate these counters in
thread-local storage and publish them only after returning. They are never read
by proof flow.

The retained FTA peaks are 72,674 requests and dependency passes; 1,049,978
reverse owners with at most 804 per request; 47,930 subset candidates with at
most four per request; 46,654 encoded hits; 82,620 local values with at most 39
per hit; 62,834 total firing records; 1,935,835 generated bytes; 189,815 level
values; 184,494 origin dependencies; 1,608 marker keys; 780 marker remaining
arguments; and 1,345 marker-argument slices. Use fixed next-power-of-two
ceilings: 131,072 requests, 2,097,152 reverse-owner work items, 65,536 candidate
owners and encoded hits, 131,072 local values, 65,536 firing records, 2,097,152
generated bytes, 262,144 levels and origin dependencies, 2,048 marker keys and
argument slices, and 1,024 marker remaining arguments. The reverse expansion is
wide but highly compactable: fewer than five percent of the peak reverse owners
survive the subset gate.

`CudaPhase2EvaluationBuffer` owns every work and final-output column at those
ceilings. Its deterministic arrays consume 37,267,500 bytes before one reusable
maximum CUB scan scratch allocation, and the production RTX test measures the
complete owner at 37,269,035 bytes, below 64 MiB. `expandEvaluationWork` is live through:

1. known-statement dependency checks;
2. comparable premise validity folding, hypothesis-anchor discipline, and
 ancestor closed-scope filtering;
3. exact normalized-key reverse lookup and fixed scan expansion;
4. remaining-argument subset testing and capacity-checked atomic compaction;
5. ignore-u mapped normalized-key construction and `overallEncoded` probing;
6. encoded-hit run scans into flat request/`LocalMemoryValue` blob work.

`materializeFiringExpressions` consumes that value work without processor
semantic replay. Each thread reconstructs numeric-key reverse identifiers from
the exact request plus remaining-argument owner. A forward cursor then applies
the processor token rule: only a complete numeric token after `[` or `,` may map,
and an out-of-range longer token is not retried as a shorter key. A second cursor
removes adjacent token-leading `u_` pairs with the same erase-and-recheck
semantics as `replaceUSubstringsScratch`. The cursor is replayed for measurement
and fill, so the thread owns no heap, variable stack buffer, or substitution
scratch arena and rejected values consume no generated-byte capacity.

Before output, the kernel checks LMV/request scope comparability, chooses the
deeper validity, repeats the ancestor closed-scope filter when the LMV deepens
the deposit, applies the marker/ordis2 purity exception only at OR boundaries,
and scans transformed bytes for the exact ordis2 iteration cap. Surviving rows
write complete fixed firing headers plus arena-relative payload runs. The same
kernel fills levels for heads and demands with a bounded multiway
merge over the already-sorted premise and LMV runs: negative premise tiers are
discarded, duplicates collapse, and no dynamic temporary is needed. Head
provenance starts with `(source rule-interner id, LMV validity id)`, followed by
premises sorted through projection-time decoded ranks for expression then
validity; identifiers remain in their source namespace until host sealing.

The resident projection does not duplicate the analyzer-wide compiled-expression
map. Instead, every projected `NameMap` and rule-interner string carries the only
compiled-core distinction Phase 2 consumes: absent, atomic, or non-atomic. Two
additional plain-data views project `canBeSentIds` and `canBeSentMarkerIds`; the
logical-block descriptor carries the anchor-name bytes, block level, standard
secondary-number ceiling, and incubator/disintegration/compressor switches.
With those compact annotations the kernel executes the exact processor verdicts:
mail eligibility including `int_lev_` and single-marker substitution, incubator
anchor/local-premise disintegration gating, OR-disintegration permission,
ancestor-known suppression, compressor handling, and marker non-atomic category.

Marker records write transformed keys in source order, remaining rule-interner
identifiers in decoded-byte sorted-unique order, and sorted-unique bare argument
slices into fixed arenas. Their admission-depth and secondary-number ceilings
are part of the fixed header. The implementation deliberately scans for each next
remaining argument instead of allocating a per-thread variable array; marker
runs are small and this preserves occupancy and the static-memory contract.
The direct RTX twin covers mapped and unmapped numeric keys, repeated `u_`
removal, deeper validity, source flags, an impure-marker rejection outside an OR
branch and acceptance inside one, the non-derived-level exclusion, four exact
level unions, complete source-first/premise-sorted provenance runs, every head
verdict flag, transformed marker keys, decoded-byte-sorted unique remaining
arguments, bare arguments, category, and admission ceilings.

The compacted candidate/value append order is not semantic. Every firing header
retains logical-block index, part ordinal, exact ordered-request index, and growth
position. `orderFiringRecords` builds a permutation in two fixed index arrays and
runs power-of-two parallel merge passes over `(logical block, existing
FiringRecord content key)`. The comparator decodes local identifiers only through
the record's own projected block and reproduces the processor's head, marker, and
ordis2-demand field order; variable payloads never move. `selectDoomPrefixes`
mirrors the processor deactivation predicate from projected roles, goals, known
statements, and compressor mode. One CAS-min reduces the exact packed line per
block; a second reduction selects the first triggering ordered request at that
line. Stable exclusive-scan compaction keeps only the winning part through that
request, excluding later requests even when they share its growth position. The
direct RTX twin proves dependency rejection, a
non-subset reverse owner, mapped key hits, a multi-value encoded run, and the exact
five-record canonical permutation.

The proposed device output stores semantic identifiers wherever the source
already has them and stores generated bytes only where substitution creates new
text. Examples:

- validity scopes remain `NameId` values until host sealing;
- head provenance dependencies remain ordered `(originalId, validityId)` pairs;
- the source implication remains the rule-interner identifier plus its validity;
- levels remain a sorted-unique integer run;
- marker remaining arguments may remain rule-interner identifiers after the GPU
 has established their decoded-byte order;
- substituted heads, substituted marker keys, and demand text live in the device
 byte arena with `(offset, length)` references;
- marker bare arguments may be slices of the emitted head bytes when that slice
 exactly matches the processor result.

The host conversion may decode these identifiers and copy these byte slices to
`SealedPageSet`. It must not look up the rule again, rerun substitution, repeat a
gate, infer provenance, or suppress a device firing.

`gpu/phase2_sealing.hpp::Phase2SealingArena` is the implemented ownership
conversion. Its constructor sizes every processor download column, canonical
index array, and per-record sealing scratch array once from
`Phase2EvaluationCapacity`; `downloadAndSeal` reuses those addresses. It copies
the used device prefixes, walks the doom-compacted canonical permutation, and
creates one ordinary sealed record chain per logical block. Generated
expressions, marker keys, and marker arguments copy from device byte slices.
Validity and premise identifiers decode through the same projected `NameMap`;
source implications and marker remaining arguments decode through the same
projected rule interner. The literal `implication` tag and already-computed
device flags complete the existing `FiringRecord` shape without reopening the
rule or any proof-state gate.

`memory.cpp::ExpressionAnalyzer::applyFiringRecords` now has an explicit
`firingRecordsCanonical` contract. Processor part chains pass `false` and retain
the existing pointer-index content sort. A GPU-sealed chain passes `true` only
after device ordering and doom-prefix compaction, so deposit walks its frozen
append order directly and does not pay for a second processor sort. Both routes
still share the same deposit and drain code.

Variable-length strings use two passes: measure exact byte counts, exclusive
scan into a bounded byte arena, then fill. A fixed per-record maximum that
truncates is forbidden. The arena capacity is explicit and an exceeded capacity
asserts with measured high-water telemetry.

---

## Doom line and observable order

Current processor semantics publish the lexicographic minimum
`(growth position, part ordinal)`. The triggering part stops after the first
request whose complete firing-record batch contains a deactivating head; every
record produced by that request remains in its chain. Finalize merges only the
winning part.

**Implemented first correct GPU form:** the bounded candidate/evaluation batch
runs to completion, reduces each part to its first deactivating request in exact
processor stream order, reduce parts by `(growth position, part ordinal)`, and
compact the winning part's output to the inclusive triggering-request prefix.
Sibling outputs are ignored exactly as the processor finalize ignores losing
chains. This removes timing from correctness and does not require cooperative
device cancellation. It may do wasted work on a doomed logical block; an
identity-proven later optimization may add cancellation without changing the
prefix result.

The reduction needs both values:

- the published growth position, which counts accepted subkeys across all five
 request batches of the part;
- the request stream token, which distinguishes multiple emitted requests that
 share one growth position and identifies the first trigger.

Do not replace the doom line with first-completed thread, atomic append order,
kernel launch order, or a head-content minimum.

---

## Allocation and transfer tools

| Need | GPU tool | Rule |
|---|---|---|
| Long-lived device memory | One backend-owned allocation set | Allocate at initialization or deliberate resize boundary; never in a request kernel. |
| Frontiers and compaction | Double-buffered fixed-cap arrays plus prefix scan | Record high water; assert before a write would exceed capacity. |
| Read-only lookup | Fixed-load open-addressed tables over packed bytes or integer keys | Build deterministically; no insertion during Phase 2. |
| Variable output | Fixed record arena plus fixed byte/integer arenas and atomic reservation or scan offsets | Capacity failure asserts; output may not truncate. |
| Host transfer | Explicit used-range copies, preferably pinned staging after the identity baseline | Report bytes by class and direction. Unified-memory page faults are not an implicit transfer design. |
| Synchronization | Named stream/event boundaries matching the Phase 1/2/3 barriers | A host read waits for the exact producing event. |
| Device algorithms | CUDA primitives whose ordering contract is wrapped by GL twin tests | Library choice never substitutes for semantic-order proof. |

The MPU 0.1 shape helps directly: fixed-capacity pools, flat records, integer
identifiers, read-only Phase 2 state, split logical blocks, and sealed
producer-to-consumer handoff already expose the boundaries a GPU needs. It does
not make branch-heavy work automatically fast; speed comes from batching many
independent frontier nodes and rule hits while keeping their packed state
resident.

---

## Windows CUDA build contract

The audited development machine has CUDA Toolkit 13.3, Visual Studio 2022, and
an RTX 4070 Laptop GPU with compute capability 8.9. The current driver exposes a
CUDA 13.2 runtime level.

A real probe established:

- default Toolkit 13.3 PTX reaches the device but launch fails because the
 driver cannot JIT that newer PTX toolchain;
- compiling an explicit `sm_89` cubin launches and returns the expected value.

The Visual Studio project now imports the installed CUDA 13.3 build
customization and compiles [`phase2_cuda.cu`](../../GL_Quick_VS/GL_Quick/src/gpu/phase2_cuda.cu)
with native `compute_89,sm_89` and `compute_120,sm_120` images. CUDA translation units also pass `/Zc:preprocessor`
because the Toolkit 13.3 CUB headers require MSVC's conforming preprocessor.
`cuobjdump --list-elf` confirms four cubins: native `sm_89` and `sm_120` for both
CUDA translation units. Ada and Blackwell execution therefore do not depend on
PTX JIT. The complete unit gate calls
`queryCudaDeviceContract` and `launchCudaContractProbe`; the second function
allocates a device word, launches a real kernel, synchronizes, and copies the
deterministic result back. Device enumeration alone is not accepted as a
launch test. The RTX 4070 gate exercises `sm_89`; the embedded `sm_120` image is
compile-proven but requires a Blackwell device for runtime verification.

Use absolute Toolkit paths while the long-lived Codex process retains its
pre-install environment. That is a shell concern only; project files should use
the installed CUDA Visual Studio build customization rather than hard-coded
developer-machine compiler paths.

---

## Pitfalls

### Host pointers disguised as plain records

`StaticRequest::intExprs`, `IntNormalizedKey::data`, `SplitStumpRef::stumps`,
`StrSpan`, and cold-container peek pointers are host addresses. Replace them
with projection-relative offsets or inline bounded arrays before transfer.

### Mint order used as lexical order

Name and rule identifiers are stable equality tokens, not string ranks. Any
filter sort, candidate sort, rule-value sort, premise sort, marker-argument sort,
or canonical output comparison that was byte-lexical must use decoded-byte rank
or compare bytes.

### Kernel completion used as semantic order

Atomic append order and warp scheduling are nondeterministic. Every observable
record carries a deterministic content key or processor stream token and is
sorted/compacted at the defined boundary.

### Provenance postponed into processor re-evaluation

The device output must already identify the exact source implication, ordered
premises, validity, levels, tag, marker data, and flags. Host sealing is an
ownership conversion, not a second evaluator.

### Device allocation in the hot path

`cudaMalloc`, host heap growth, and unbounded queue expansion inside a burst hide
the real capacity. Preallocate, measure, and assert.

### Per-logical-block tiny launches

One kernel per small logical block reproduces processor under-occupancy with
extra launch and transfer overhead. Batch independent logical-block parts and
frontier nodes into global work arrays while retaining the logical-block and
part identifiers on every row.

### Rejecting harmless long-rule metadata too early

A registry may advertise a maximum rule-key length beyond the device frontier
capacity even when the filtered live search never reaches that depth. Projection
accepts that metadata. The exact growth seam asserts only when a live frontier
node at the eight-expression ceiling has another retained position to traverse;
the assertion therefore names a real ninth-expression requirement rather than a
harmless long rule elsewhere in the registry.

### Copying the full static pool

The host process reserves more static memory than the GPU owns, and reservation
is not live content. Transfer compact logical snapshots and dirty ranges; report
actual bytes. Never size a device copy from the host pool ceiling.

### PTX-only output on the current driver

Toolkit compilation success is insufficient. The current machine requires a
native `sm_89` image for successful launch until the driver supports the
Toolkit's PTX version. The stronger-GPU target likewise carries native `sm_120`;
do not replace either with forward PTX as a portability shortcut.

### Silent processor fallback

An unavailable device, unsupported capability, allocation failure, queue
overflow, or kernel error is an assertion in selected GPU mode. The processor
backend runs only when explicitly selected.

---

## Verifying a GPU conversion

1. **Representation twin.** Pack a real logical-block fixture, copy it through
 the device representation, and compare every decoded statement, string,
 key, run, blob, flag, and identifier to the host source.
2. **Primitive twins.** Compare device hashes, normalized keys, scope folds,
 request gates, substitution bytes, owner tests, and ordering ranks against
 the existing processor helpers, including maximum-capacity and negative
 cases.
3. **Generator twin.** Capture complete processor and GPU request streams for
 the same batch, stump bucket, and logical block. Compare semantic keys,
 first-occurrence order, growth positions, and deduplication.
4. **Evaluator twin.** Compare every field and variable run of every generated
 firing record before deposits are applied.
5. **Doom twin.** Exercise multiple triggers, equal growth positions, different
 part ordinals, and multiple requests at one position. Compare the selected
 winner and inclusive record prefix.
6. **Partition twin.** Run unsplit and several bucket counts; compare the
 contractually expected record set or deterministic doom winner.
7. **Pipeline identity.** Full rebuild, then run processor and GPU FTA shortcut
 paths only through `main.py --shortcut`; retain separate logs and artifacts,
 compare proof outputs byte for byte, and run the same verifier.
8. **Performance after identity.** Sum `[PHASE2-TIMING] iteration_seconds` (now written to the common diagnostics log , batch- and hashburst-marked) across
 the complete processor and CUDA shortcut logs. Report that barrier-to-barrier
 Phase 2 comparison together with prover and overall shortcut time; the latter
 remains the usefulness gate. Then report transfer bytes, device and host
 high-water memory, frontier/output peaks, utilization, and the exact
 processor/GPU commit.

For CUDA, sum `[GPU-PHASE2] device_seconds` for the reusable-event device
timeline. On the dependency-staged route it begins after the projection upload
has been queued and host scheduling has overlapped it; it contains the task
upload, the projection-stream join, and every semantic kernel through doom-prefix
selection. Sum `route_seconds` for the enclosing per-pass wall time. Preparation
contains host projection, the staged projection-copy start, and scheduling;
finalization contains downloads and sealing. These nested values explain
overhead; the complete Phase 2 bracket and end-to-end shortcut still decide
whether the port is useful.

When device optimization makes `route_seconds - device_seconds` material, split
the host route before changing ownership. Measure preparation from route entry
through projection, staged transfer start, and schedule construction; measure
CUDA-segment wall time around the event interval; measure finalization across
used-prefix downloads and sealing. The three host-clock intervals must sum to the
enclosing route. Treat CUDA-event time as the exact interval named by the active
upload schedule, not automatically as all projection-upload time: an upload
deliberately started before its event belongs to preparation and the complete
Phase 2 bracket. Runtime-call wall overhead belongs in the CUDA-segment wall
interval, not in sealing.

A kernel benchmark is evidence about one kernel, not evidence of Phase 2 or
complete-run acceleration.

### Profile before changing the frontier shape

The first paired FTA measurement put complete Phase 2 at 110.333
seconds on the processor and 228.782 seconds on CUDA. The exact CUDA-event
interval, which deliberately includes projection and task uploads, was 220.274
seconds. Nsight Systems then attributed 217.594 seconds across 224 launches,
99.0% of all CUDA kernel time, to `phase2GrowthExpandKernel`; host-to-device
copies totaled 0.330 seconds and device-to-host copies totaled 0.011 seconds.
Do not optimize transfer or host sealing first on this shape.

`Memory::generateEncodedRequestsStatic` already performs the applicable
processor optimization: for one frontier pop it builds the normalized-key
prefix and the hypothesis, scope, and secondary-variable summaries once, then
folds only each appended statement. The first CUDA growth kernel instead rebuilt
those prefix facts for every candidate position. Preserve the compact 56-byte
`DeviceGrowthNode`; storing a full normalized payload in every node would expand
both million-record frontiers substantially. Build the order-free prefix arrays
once in the owning CUDA thread, keep them live while it scans later positions,
and fold only the appended expression. The direct request-growth content twin
must remain exact before any timing result is accepted.

On, that prefix hoist reduced the exact upload-plus-device interval
from 220.274 to 71.034 seconds and complete Phase 2 from 228.782 to 78.845
seconds. The optimized Phase 2 is 1.399 times faster than the paired 110.333-second
processor baseline, while the complete shortcut is 1.137 times faster: 192.096
versus 218.442 seconds. The verifier retained 10,038 checks with zero failures and
all 305 current proof artifacts remained byte-identical. Rerun the profiler after
a large optimization; the old 99.0% attribution is no longer a valid guide to the
new shape.

The post-hoist Nsight Compute report for a representative 694-block launch shows
the next resource boundary: 65 registers and a 5,024-byte stack per thread,
43.89% achieved occupancy against 50% theoretical, 58.78% memory throughput, and
only 8.04% compute throughput. Registers permit only three resident blocks per
multiprocessor and Nsight classifies the launch as latency-bound. Before changing
frontier ownership or parallelizing candidate positions, remove redundant
thread-local state from the current exact kernel. Build the prefix directly in
the final serialized-key buffer so later positions overwrite only the appended
suffix; detect duplicates among one appended expression by reading its prior
arguments rather than allocating peer arrays. Re-run the exact growth-content
twin, inspect the newly compiled register/stack shape, and measure the complete
shortcut before claiming an improvement.

That local-state reduction crossed the important compiler threshold on the
current `sm_89` target: registers fell from 65 to 63 per thread, the thread stack
from 5,024 to 3,184 bytes, register-limited residency from three to four blocks
per multiprocessor, and theoretical occupancy rose from 50% to 66.67%. A direct
one-block twin proves the resource shape and exact content, but its achieved
occupancy is not a workload measurement. Only the complete shortcut decides the
runtime effect.

The complete shortcut confirms that the compiler threshold mattered:
exact CUDA-owned time fell from 71.034 to 15.929 seconds and complete Phase 2
from 78.845 to 25.178 seconds. Against the paired processor baseline, Phase 2 is
4.382 times faster and the full shortcut is 1.349 times faster, 161.911 versus
218.442 seconds. The same 10,038 verifier checks pass with zero failures and all
305 current artifacts remain byte-identical.

The post-change Systems profile assigns 13.005 seconds and 87.0% of kernel time
to growth. A representative 694-block Compute launch takes 97.41 milliseconds
under collection, reaches 54.35% occupancy and 46.49% compute throughput, and
uses only 1.96% DRAM throughput with 63 registers and a 3,184-byte stack. The
kernel is no longer DRAM-bound; register-limited occupancy, instruction latency,
and partial waves are the remaining device constraints.

The route breakdown reconciles across all 48 passes within 3
microseconds. Host projection and scheduling consume 8.270 seconds, CUDA-segment
wall time consumes 15.684 seconds and contains the 15.672-second exact event
interval, and downloads plus sealing consume 0.283 seconds. Complete Phase 2 is
25.046 seconds, leaving 0.808 seconds after the CUDA route. Transfer and sealing
are therefore closed as small seams on this shape; repeatedly rebuilding the host
projection is the only substantial non-kernel seam. Measure fixed ownership and
resident high-water memory before deciding whether to cache that projection.

The fixed-ownership gate accounts for every production `cudaMalloc` request in
the six process-static owners. After main appended provenance-owner records to
the projected value blobs and raised the current forward-map slot peak, projection
owns 299,951,120 bytes, tasks 731,144,
filtering 25,577,983, growth 812,965,447, ordering 98,838,019, and evaluation
37,269,035. The complete application-owned device reservation is
therefore 1,275,332,748 bytes (1,216.252 MiB, 1.188 GiB, or 14.85% of the 8,188 MiB
device).
Driver and CUDA-context allocations are outside
this application-owner figure. No production route performs a transient
allocation; the contract probe is test infrastructure and CUB uses the owners'
preallocated reusable scratch.

The traffic gate makes the remaining host seam unambiguous. Across
48 passes, preparation consumes 6.977704 seconds: 6.904738 seconds of projection
and 0.072964 seconds of schedule construction. Projection is 98.95% of
preparation and 30.09% of the 22.948067-second complete Phase 2 bracket. The
same run peaks at 1,088,142 accepted events, 49,380 unique requests, 1,007,955
reverse owners, 50,732 firing records, 1,619,493 generated bytes, 150,183 level
values, and 154,845 provenance dependencies. A provenance dependency is one
8-byte `DeviceEvaluationDependency`, so the peak is 1,238,760 bytes (1.181 MiB,
59.07% of the fixed 2 MiB arena). Mail creates no parallel output stream:
`DevicePhase2FiringRecord` already carries the `deviceFiringAllGood` eligibility
bit, and the host seal converts that decided bit into the ordinary record. Do
not estimate mail memory as a second firing queue.

Projection reuse must preserve this accounting discipline. Cache presence is
performance-only; an unchanged verdict needs an explicit generation or complete
content identity, never pointer identity, resident-page identity, queue timing,
or an inferred lack of writes. A changed logical block is projected completely
before its device slice becomes visible. Reuse may remove host traversal and
used-range copies, but it may not move evaluation, provenance, mail verdicts,
canonical order, or doom selection back to the processor.

Internal projection attribution is required before introducing a persistent
cache. On , mutually exclusive
builder intervals accounted for 6.601027 of 6.754050 projection seconds. Names
owned 5.112147 seconds: 0.618587 seconds to copy records and compiled categories,
4.304092 seconds to sort decoded lexical order, 0.005251 seconds to write ranks,
and 0.184214 seconds to build name lookup slots. The outer projector overhead was
only about 0.153 seconds, so caching the complete 24-column image would have been
an overbroad first response to a localized cold-table traversal.

The first rank improvement compared packed `DeviceNameRecord` spans in
`nameBytes` after those bytes had been copied, instead of calling
`NameMap::decodeView` on the cold paged table for every comparison. It retained
the same `compareSpans` byte ordering, identifier set, rank write, fixed
reservation, and device image. That verified run reduced decoded lexical sorting
to 1.935975 seconds, name projection to 2.718540 seconds, total projection to
4.308227 seconds, and complete Phase 2 to 19.7768 seconds.

The current sorter uses fixed-capacity most-significant-byte radix scratch over
the same packed spans. End-of-string is bucket zero and each unsigned byte maps
to buckets one through 256, which is exactly `compareSpans` order. Each pending
range first collapses its complete common prefix; only the first differing byte
gets a counting/scatter pass. Without that collapse, FTA's long validity-name
prefixes made the first radix attempt regress sorting to 4.706019 seconds. The
fixed-forward run reduced sorting to 1.544649 seconds, name projection to
2.322149 seconds, total projection to 3.987276 seconds, and complete Phase 2 to
19.2570 seconds. Exact CUDA-event time was 14.331816 seconds. All 1,543 native
tests, both 10,038-check verifier passes, and the 305-artifact SHA-256 comparison
passed. Prefer consuming the packed projection already under construction before
adding another cache, generation, owner, or transfer protocol.

FTA shortcut supplies no in-process projection reuse. Each fresh prover
subprocess executes at most two CUDA passes: the first projects non-stragglers,
while the second projects only the `produceOnly` stragglers skipped by the first.
The sets are disjoint and the process exits afterwards. A generation-backed
in-process cache therefore has zero hits on this target. Retain the approved cache
contract for a future mode that demonstrably reprojects the same logical block in
one process; do not add cross-process persistence merely to manufacture reuse.

The stronger commodity GPU estimate is an Amdahl-style bound, not a benchmark.
NVIDIA lists [21,760 CUDA cores at 2.41 GHz for the RTX 5090](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/).
The actual measured device is an RTX 4070 Laptop GPU, not the desktop card;
[NVIDIA's laptop specification](https://www.nvidia.com/en-us/geforce/laptops/40-series/)
lists 4,608 CUDA cores, a 1.230-to-2.175-GHz boost range, 8 GB GDDR6, and a
128-bit interface. The local driver reports 3,105-MHz maximum SM and 8,001-MHz
maximum memory clocks. That gives a 4.722-times core-count ratio, a 5.232-times
official rated core-clock-product ratio, and 1,792 versus 256 GB/s of bandwidth.
GL's measured growth kernel is latency and occupancy constrained with 1.96% DRAM
throughput, so those hardware ratios are sensitivity inputs rather than claimed
scaling.

Keep the measured 4.925 non-CUDA seconds and about 0.525 transfer seconds fixed,
then scale the remaining 13.807 device seconds. A 3-to-4-times device factor gives
10.052-to-8.902-second complete Phase 2, 1.916-to-2.163 times faster than the
current RTX 4070 Laptop route and 10.976-to-12.394 times faster than the processor
baseline. Scaling by the 5.23 rated core-clock-product ratio gives an aggressive
8.090-second, 2.380-times sensitivity point, not an expected result. Run the
native `sm_120` binary on the actual card before replacing this range with a
measured result.

---

## Approved port contract and implementation state

The approved first-port answers are maintained in
[`docs/GPU/gpu_plan.md`](../../GPU/gpu_plan.md); the full-run campaign
ledger is [`docs/GPU/gpu_main.md`](../../GPU/gpu_main.md). In short: batch
config ownership of the backend (`ProverParameters::use_gpu`, default false —
the processor route; `main.py --GPU` selects CUDA for the whole run)
with `--phase2-backend cpu|cuda` as the explicit command-line override;
process-owned preallocated CUDA memory sized by the named audited capacity
profiles (`Phase2ProjectionProfile::ftaShortcut` / `fullRun` and the named
task/filter/growth/ordering/evaluation constants); packed logical-block
projections; bulk frontier generation with exact order tokens; complete
device-side evaluation and provenance content; host-only sealing plus the
existing `applyFiringRecords`; and deterministic post-compute doom-prefix
reduction.

The Windows CUDA build seam, real-kernel device contract, complete fixed
resident projection, corrected capacity census, all-column transfer twin, and
native lookup layer, pointer-free host task image, exact bulk filter/sort stage,
retained growth-capacity census, complete request-growth content, the
812,965,447-byte pooled-prefix and candidate-window fixed growth owner, exact
per-task split-work counts, exact remaining-span census, stable
short/cooperative selection, canonical candidate windows, cooperative long-span
cheap gates, stable survivor compaction, and exact on-device stream reconstruction
are live. Exact processor event order, task-local growth positions, and
collision-checked per-call deduplication pass direct unsplit, mixed-stump, and
duplicate-row twins. Production routing is live for FTA shortcut mode: the
processor remains the default, `--phase2-backend cuda` selects CUDA explicitly,
and selected CUDA mode asserts instead of falling back. The processor stump
producer remains the split-work discovery prepass; semantic request filtering,
growth, ordering, evaluation, provenance, doom selection, and firing-record
materialization then execute on the device. Evaluation work expansion through
exact LocalMemoryValue selection and byte-exact firing-record
materialization is live, including combined levels, head provenance, final head
verdicts, and complete marker payloads. Canonical record sorting and deterministic
doom-prefix reduction are live through fixed arrays. Fixed processor staging now
downloads every used output column and seals complete canonical `FiringRecord`
chains without semantic replay; `applyFiringRecords` consumes those chains
without sorting again. The first CUDA shortcut executed 48 device passes, passed
10,038 proof-graph checks with zero failures, and produced all 311 retained
 artifacts byte-identically to the processor baseline. The first paired timing measured processor
Phase 2 at 110.333 seconds and CUDA Phase 2 at 228.782 seconds; CUDA's exact
upload-plus-device interval was 220.274 seconds and its enclosing route was
228.121 seconds. Device work therefore owns 96.28% of CUDA Phase 2: host transfer,
sealing, and finalization are not the primary bottleneck. End-to-end shortcut time
 was 218.442 seconds on the processor and 343.659 seconds on CUDA. Two exact
 growth optimizations then lowered complete CUDA Phase 2 to 25.178 seconds and
 the complete shortcut to 161.911 seconds: 4.382 and 1.349 times faster than the
 paired processor run. The performance gate keeps the explicit Phase 2 bracket
 for diagnosis and the complete shortcut as the usefulness decision. Exact
 fixed device ownership is now 1,275,332,748 bytes (1,216.252 MiB); the refined traffic
 run measured complete Phase 2 at 22.948 seconds and isolated 6.905 seconds of
 projection from only 0.073 seconds of scheduling. Internal attribution then
 localized the dominant host cost to repeated cold-table name comparisons. The
 packed-span sorter reduced complete Phase 2 to 19.777 seconds, then the
 prefix-collapsed fixed radix sorter reached 19.257 seconds, a 5.730-times
 acceleration over the retained processor Phase 2 baseline, with every artifact
 exact. The first adaptive candidate-parallel route then reduced the fresh
 20.1945325-second CUDA control to 15.1108074 seconds: exact device time fell
 from 15.2917872 to 10.1747262 seconds, the cooperative peak was 301,811 nodes,
 all 311 artifacts remained byte-identical, and the verifier retained 9,994
 checks with zero failures. The native `sm_89` cooperative kernel uses 53
 registers, a 1,280-byte lane stack, and 2,016 bytes of block shared memory,
 versus 62 registers and a 3,104-byte stack for the short-node kernel. The live
 cutoff-64/64-lane tuning then reached 12.4757739 seconds complete Phase 2 and
 7.5260573 seconds exact device time, with a 552,505-node cooperative peak and
 all artifacts exact. Nsight Systems attributes 3.732 seconds to cooperative
 growth, 0.986 seconds to short-node growth, and 1.738 seconds to filter count
 plus emit. A 311,218-node `sm_89` cooperative profile reaches 65.57% occupancy,
 46.72% compute, 2.11% DRAM, and 99.90% L2 hits; only 10.83 threads per warp are
 active on average, so divergence/latency rather than device-memory bandwidth is
 the remaining kernel limit. Four parallel host projection shards then reduced
 projection from 3.9306738 to 1.6821862 seconds and complete Phase 2 to
 10.6506 seconds. Their exact merge costs 0.3905575 seconds; construction owns
 4.2007232 processor-seconds across the workers. All 311 artifacts remain exact
 and the verifier retains 9,994 checks with zero failures. This is a retained
 1.896-times intermediate, 0.733 seconds above the strict gate; the measured
 worker work justified shard-count tuning before transfer residency. Eight
 shards then reduced projection to 1.3315546 seconds and complete Phase 2 to
 9.9825 seconds, 2.023 times faster than the fresh control but still 0.0649
 seconds above the stricter historical 9.917614-second gate. The exact device
 interval was 7.6424692 seconds, merge was 0.4056626 seconds, and all 311
 artifacts plus 9,994 zero-failure checks remained exact. Twelve shards lowered
 projection only to 1.3033454 seconds while merge rose to 0.4433709 and the
 complete Phase 2 run measured 10.1734 seconds. This exact result shows host
 shard scaling has saturated; further arena multiplication is not the next seam.
 Dependency-staged transfer then split the fifteen filter/growth/ordering
 columns from nine evaluation-only columns. Registering the live projection
 vectors and allowing copies to compete with kernels regressed the exact result
 to 10.8707 seconds. The accepted route instead owns fixed pinned staging,
 starts transfer immediately after projection, overlaps it only with host
 schedule construction, and joins before filter. Consecutive exact runs measure
 9.74343 and 9.46378 seconds complete Phase 2: 2.036 and 2.096 times faster than
 the strict 19.835229-second historical reference. All 1,558 native tests,
 9,994 zero-failure verifier checks, and 311 artifact hashes remain exact.
 The live
 shortcut process/pass shape gives an in-process projection cache
 no reuse hits. Queue, firing, provenance, level, generated-byte, and marker peaks are
 recorded above; mail adds no separate device queue. The corrected stronger-GPU
 model is recorded above and both native architectures are embedded.

**Build flag (release 13, [D-340](../40_decisions.md#d-340)).**
The CUDA route is compiled under the preprocessor definition `GL_CUDA`, which
the Visual Studio project defines unconditionally and the Makefile defines
under `make USE_CUDA=1` (nvcc for `src/gpu/*.cu` and `src/tests/*.cu`,
`sm_89` + `sm_120`, `cudart` linked, `CUDA_HOME` default `/usr/local/cuda`).
`GL_CUDA` gates the CUDA includes and the CUDA block of `proveKernel` in
[`prover.cpp`](../../../GL_Quick_VS/GL_Quick/src/prover.cpp), the whole
sealing translation unit `gpu/phase2_sealing.cpp` (it downloads the device
buffer, whose methods live in the `.cu` file), and the CUDA unit tests. A
build without `GL_CUDA` has no CUDA route at all: selecting one
(`--GPU` / `--phase2-backend cuda` / `use_gpu: true`) asserts at prover
construction, never falls through to the processor. Both Linux builds are
gated on Ubuntu (WSL2, CUDA 13.3, g++ 15): processor and CUDA, 1626/1626
unit tests each.

The closed boundary is `main.py` complete on Windows and Linux — the FTA
shortcut AND the full incubator/main pipeline. The processor route is the product default
(`ProverParameters::use_gpu`, false; no shipped config pins a batch),
`main.py --GPU` selects CUDA for every batch of the run,
`--phase2-backend cpu|cuda` is the explicit whole-run override, and selected
CUDA asserts rather than falling back. A CUDA-selected batch also owns its counterexample filter: sixteen
private conjecture clones per direct CUDA Phase 2 call, never a processor
replay. CUDA and SSD deload are mutually exclusive — a CUDA batch derives
`allow_ssd_deload` false and runs the resident-only steward, while a
processor batch pages through the working-set steward by default. Phase 1, Phase 3, Python proof-graph processing, and
verification remain on the processor. The approved split-stump discovery
prepass and fixed-array sealing also remain on the processor, while semantic
request work through provenance, marker payloads, canonical ordering, and doom
selection executes on the device. Fixed device ownership is sized by the named
audited capacity profiles; a full run streams deterministic projection chunks
that keep each split LB's task family whole. Non-Windows hosts and new anchors
still require new capacity, identity, and performance gates. Native `sm_120`
is embedded but remains runtime-unmeasured until a Blackwell card is
available. Do not widen any of these claims from direct helper coverage alone.

---

## See also

- [`09_static_memory.md`](09_static_memory.md) — the host static-memory hierarchy.
- [`09b_statification_cookbook.md`](09b_statification_cookbook.md) — container and arena recipes.
- [`09c_string_statification_cookbook.md`](09c_string_statification_cookbook.md) — string ownership and span rules.
- [`02_hash_engine.md`](02_hash_engine.md) — normalized keys and rule lookup.
- [`../10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — Phase 2 execution model.
- [`../../GPU/gpu_plan.md`](../../GPU/gpu_plan.md) — persistent port plan, approvals, and evidence.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
