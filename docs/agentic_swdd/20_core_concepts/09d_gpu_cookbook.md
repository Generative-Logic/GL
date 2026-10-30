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

## Production bounded execution

`proof/phase2_executor.cpp` / `Phase2Executor::execute` measures consecutive complete LB
families and packs immutable projection, task and filter storage with minimum
execution scratch against the automatic 2 MiB block budget. The one-LB fit assertion
is retained. `makePhase2BufferPlanInput` maps every measured projection/task column explicitly,
reserving one element for empty arrays. A minimum-only allocation query avoids
demand optimization during each packing probe; the full planner runs once for the chosen chunk. The
automatic budget stops at `kPhase2DefaultGpuBudgetBytes` (4 GiB,
[D-412](../40_decisions.md#d-412));
`GL_GPU_BUDGET_MIB` replaces that ceiling. Ordinary `main.py --GPU` needs no
hardware-specific settings. `CudaPhase2BlockPool` additionally reserves up to
`kPhase2FrontierStoreReserveBytes` (896 MiB) of whole blocks for the frontier
store, taken only from device memory left after the working budget and the
runtime reserve (`Phase2GpuBudget::frontierStoreBlocks`); `planPhase2Buffers`
charges the store's whole blocks to that reservation, not to the working blocks
([D-465](../40_decisions.md#d-465)).

Each chunk borrows six disjoint block leases for projection, tasks, filter,
growth, ordering and evaluation. Consecutive complete logic blocks form host
construction windows targeting `kPhase2HostWindowTargetBytes` (32 MiB) of measured
column storage (two are alive at once when the independent host budget permits);
the executor and the host planner read that one constant, so the planned windows
are the built ones. A block larger
than that target owns its exact measured window. Host pressure can reduce a device chunk only at complete logical-block family boundaries; it never splits a family or drops work. Within each window, parallel builders keep disjoint round-robin
assignments, shared backing columns and canonical descriptor order. One set of
persistent workers per chunk builds and adopts every window; adoption adds the
accumulated complete-image offsets directly. An upload thread uploads each finished
window and keeps its name/rule lookup data and source statement ordinals on the
host while the next window is built, so at most two windows are alive. The upload
thread empties the window's slot as soon as both synchronous copies finish, so its
contiguous GPU-host lease returns at once. Device kernels begin after the complete image is
sealed. See [D-436](../40_decisions.md#d-436),
[D-447](../40_decisions.md#d-447) and
[D-491](../40_decisions.md#d-491).
Chunk/task directories use fixed GPU-host pool leases; repeated filter-call examined-row
accounting uses 64-bit totals.

`Phase2SealingArena::runWindows` owns the pause/drain/resume orchestration. Generation
stays frozen while locally ordered requests pass bounded owner/value evaluation. An aggregate per-block stop takes effect only after a completed logical growth level. Raw device output downloads repeatedly;
a marker publishes one complete record after its key fragments assemble. Pages seal
after generation and evaluation finish. Production sets the canonical flag false so
`applyFiringRecords` uses the shared RAM comparator and paged merge sorter. No host
inference, substitution, provenance reconstruction or early statement deposit occurs.

Completed chunks feed observed occupancy and drain pressure into the next allocation.
Permanent diagnostics separate allocated blocks, frontier and output transfers,
host backing, and elapsed route time. CUDA event duration includes host gaps between
kernels and must not be described as pure kernel busy time. Pipeline acceptance for
this controller is recorded in `docs/GPU/batched_gpu_plan.md`.

## Bounded Phase 2 execution and early exit

Before constructing any of the six CUDA buffers,
its `measureAllocation` query reports device payload bytes and the exact aligned
arena footprint. Queries include current CUB workspace and every array, with
256-byte alignment between arrays; they create no device-array ownership. Direct
tests compare reported payload with each real owner and compare aligned footprint
with the shared arena cursor. Adding a device column requires updating its sizing
query as well as the constructor. Production selects capacities through the common automatic planner.

`CudaPhase2EvaluationBuffer::nextFiringWindow` packs consecutive values using
GPU-measured simultaneous bounds for all seven output columns. Bounds accumulate
across 1,024-row metadata transfers until an actual output column fills; the
metadata transfer size does not impose a materialization or download boundary. A large marker
uses the largest fitting consecutive key slice. Ready output must be downloaded
before reuse. If even the next minimum fragment does not fit, the method returns
`needsCapacity` with its requirements and an unchanged value/key cursor; it neither
writes nor drops the request. The production planner reserves the shared CPU single-fragment bound before
execution; the explicit smaller-buffer API retains this non-consuming scheduling state.

`Phase2SealingArena::downloadAndSeal` can consume raw device order and leave its
output page sets filling across drains. A receiver collects marker-key fragments
without appending partial records; repeated fragments reuse already-sealed head,
validity, marker arguments and remaining arguments. `SealedSpanBuilder` reserves
the final host span, copies consecutive fragments and exposes it only when complete.
This is ownership conversion, with no CPU inference or substitution. The existing
single-host-span allocation contract remains; arbitrary generated-byte and other
provenance-column segmentation are still unfinished.

The processor `BurstSink` retains its original stop for the ASIC path. GPU
deactivation eligibility remains on CUDA in `deviceFiringDeactivates` and
`markDoomTriggers`. `Phase2SealingArena::runWindows` keeps one 64-bit state per
logical block: active, discharge observed in this level, or stopped. Every
result in the current logical growth level is retained. Physical capacity drains
do not commit a stop. After all seed calls at one starting depth, and after each
complete frontier wave, the controller drains remaining requests, commits observed
stops and removes those blocks' next frontiers. Later seed groups still use the
original input and kernels; stopped blocks' results and descendants are excluded.
Completed firing records append directly to their final pages, including every
repeated firing and its proof information. There is no request-identity ledger,
retrospective firing-selection list, or request-identity download. Phase 3 alone
closes goals and discharges blocks. The original RAM frontier, broad execution
windows, split-stump construction and shared RAM content sort remain. See
I-237 and `docs/GPU/no_requests_plan.md`.
Candidate-window launches in CudaPhase2GrowthBuffer::runRequestGrowthWindows use growthCandidateNodeWindow to binary-search the existing stable short/cooperative index lists against cumulative candidate boundaries. The returned slices cover every positive overlapping span; internal empty spans may remain. Two fixed counter records size the launches, while a fitting whole preparation uses its complete lists directly. This changes launch work only, preserving every candidate, request and output ordering key.

Frontier transfer scratch uses three dedicated blocks from the existing host static pool, registered with CUDA for the duration of one growth sweep. Copy pages prefer 4,096 rows but are clamped by the larger node/prefix row size and the actual host block size. The blocks are unregistered before return to the pool; at the default 256 KiB block size the temporary reservation is 768 KiB. No request identities are downloaded for early exit.


The shared RAM-side firing sorter uses `PagedSortedOrdinals`: bounded initial runs, page-backed run directories and two-way merge passes. It has no fixed 64-run limit. Production GPU supplies raw-order records and uses this same RAM sorting path. See [I-234](../30_invariants.md#i-234).


## Counterexample finalization

The `standardProcessing` counterexample gate runs only contradiction checks;
completed heads are checked directly from the internal channel against loaded
main-scope facts. Retaining a complete GPU level therefore does not invoke the
ordinary equivalence and mail-processing sequence on counterexample clones.
The generation, evaluation and multiple-firing contracts are unchanged. See
I-236.

## Requests are evaluated without duplicate history

`phase2_cuda.cu` and `CudaPhase2OrderingBuffer::orderRequests` emit a token for
every recordable event, including repeated ordered premise identities within a
call or across drains. There is no request-key hash table or history owner in RAM
or VRAM. Deterministic event ordering remains. Multiple firing records for an
expression and validity scope preserve their levels, disintegration signals and
proof origins; existing deposit code merges those contributions. See
I-238 and `docs/GPU/no_requests_plan.md`.

## Bounded generation windows

### Prefix-directed candidate ranges

Production projection adds four derived byte-map views, one for each selected
hash-memory kind, after the ten existing views. `Phase2ProjectionArena::appendNextPredicateMaps`
groups the immediate ordered normalized prefixes of the actual whole keys and
subkeys into decoded-rank-ordered predicate runs. Each run's 32-bit word carries the
polarity mask and up to two argument slots of the appended premise that every
licensing key binds to the same prefix variable (`packNextPredicateWord`,
`mergeNextPremiseBindings`). Both cheap-gate kernels resolve those slots against the
node's distinct normalization-variable pool (`growthArgumentGate`) and reject a
statement with a different argument there before `preparePooledGrowthCandidate`; such a
statement normalizes outside every key of the transition, so positions, ordinals and
accepted events are unchanged ([D-448](../40_decisions.md#d-448)). Retained
empty short-subkey records participate. Singleton key lengths determine predicate
arities; the prefix retains the complete argument-sharing pattern. Temporary
deduplication uses a private static-pool arena. The final index reuses the existing
24 packed columns and enters the normal projection sizing and block-budget calculation.

`measurePhase2SharedKeyUsage` counts the distinct byte-key union and exact
prefix entry, hash-slot and transition counts in private static-pool scratch
before the host/device allocation is planned. `appendLogicalBlock` uses a
separate construction dictionary for each logical block: byte-identical keys
in different maps or prefix tables share one `byteKeyBytes` range, but every
map keeps its original entries, value runs and membership. Reverse-index key
offsets address the already projected forward `blobBytes`; the reverse-key
transport column stays empty while its ordinal and positive minimum capacity
remain. Shard adoption rebases reverse key offsets by the blob-byte base.
Neither native proof storage nor owner validation changes. See
[D-424](../40_decisions.md#d-424).

Projection-only measurement on the full F3 workload (2026-09-28): largest host projection decreased
from 3,113.3 to 1,615.4 MiB in both sequential runs. Peak native working set
decreased from 9,939.0 to 8,576.7 / 8,569.5 MiB; the 6,144 MiB main pool
still peaked at 5,895.25 / 5,888.0 MiB. Exact theorem and proof bytes matched
the reference. These independent peaks are not additive.


`phase2GrowthPreparePrefixesKernel` queries one transition entry per prepared
prefix. `growthPredicateRange` then locates the allowed predicate names within
the existing retained, rank-sorted statement span, clipped at its original cursor.
The 40-byte prefix header carries the transition entry and reduced span length.
The candidate-count scan and both cheap-gate shapes cover these disjoint name ranges;
the polarity mask rejects an unlicensed sign before ordinary candidate checks.
The configured eligible-row prefix retention, original positions, mandatory suffix masks, split stumps,
terminal probes, full key/owner checks, accepted-subkey counts, complete-level stops
and canonical firing output remain unchanged. Every currently accepted key supplies
its own transition, so no accepted event or descendant is removed by the prefilter.

The explicit ten-view projection mode remains the low-level test oracle; all
production GPU projections request the fourteen-view image. Missing indexed
continuations have zero candidates, never an unfiltered fallback. Per-pass diagnostics
report candidate positions before/after name-range selection, derived key/predicate
counts, used index bytes, and index construction worker time. That time is nested
inside byte-map construction and must not be added to total projection time.
See `I-248` and `D-384`.

### Cached statement argument gates

`phase2GrowthMandatoryMaskKernel` prepares each retained statement's distinct
iterative argument positions once per growth call and removes identities present
in that logical block's immutable recursion-product map. `growthDistinctSecondaryArgumentMask`
selects the first iterative occurrence of an identity, even when an earlier
non-iterative occurrence exists. A 32-bit device word keeps the original mandatory
membership in bits 0..7 and this argument mask in bits 8..23. Suffix unions and
request membership consume only the low byte. Compared with the one-byte membership
array, this costs three extra bytes per retained row, included in allocation sizing.
The 24 projection columns and encoded statement bytes remain unchanged.

`preparePooledGrowthCandidate` checks hypothesis length/scope compatibility before
secondary-variable counting, then visits only cached argument positions. It still
compares those identities against the exact prefix's distinct secondary set and
applies the original secondary-variable limits. Recursion-product lookups and
within-statement duplicate scans therefore leave the per-candidate path; no gate
verdict, candidate position, frontier boundary or firing order changes.

The same candidate gate reuses each ancestry answer for both comparability and
the deeper validity. If the current validity is an ancestor of the appended
statement, the statement's validity is the result; otherwise the reverse test
selects the current validity, or rejects an incomparable pair. It does not walk
the same ancestry path again merely to recover that already established result.

`CudaPhase2GrowthBuffer::runRequestGrowthWindows` runs the existing CUDA semantic
kernels with explicit seed-call/stump, frontier-page, prefix and candidate-window
positions. A seed window launches one thread per (call, stump) pair
(`phase2GrowthSeedPairsKernel`), so a few calls with long stump runs still fill
the device; the whole-frontier reference path keeps one thread per call
(`phase2GrowthSeedKernel`), and both run the same per-pair body `seedGrowthPair`
([D-449](../40_decisions.md#d-449)). A private
host static-pool arena backs bounded resident request-frontier tails and transfer scratch. Excess rows use RequestFrontierStore; see the request-frontier disk storage section below.
CUDA measures each prefix's three payload lengths; host scheduling packs consecutive
prefixes within the available pools, then CUDA fills them. Cheap rejection fills a window bounded by the candidate buffer, independently of output slack. GPU-selected survivors then execute in consecutive probe windows, reserving worst-case room for one child, event and request per survivor before launch. Probe windows consume remaining event/request space. A tail below one quarter of full probe-window capacity (or the final smaller remaining run) triggers a synchronous drain, retaining all dependent results before request slots reset. Sparse output therefore does not force a drain after every large probe. Candidate indices and prefix payloads stay live across these drains, so rejection is not replayed.

Per-task accepted-subkey counts accumulate into 64-bit host metadata across drains.
The result reports logical event/request totals, peak frontier/backing occupancy
and frontier upload/download bytes. Host code copies rows and schedules from sizes;
all request construction, gates, normalization and exact probes remain on CUDA.
Original call/run/path fields survive physical pages. Complete Phase 2 processing
allows pages to execute in physical order before shared canonical result sorting.

The windowed driver keeps the frontier in a pass-local device store: `deviceFrontiers_[0..1]` hold `frontierStoreRecords` rows each (at least `frontierRecords`), one the level being grown and the other its children. Pages of a level are read in place, device rows first; seeds append to the current buffer, probe children to the other, and the buffers swap at each level end. The host keeps upper bounds of the four append counters (seeds, children, events, requests; at most one per seed pair or survivor) and reads the exact counters only before a window the bounds cannot cover, at prefix windows (one read for both node selections and the attempt total), at each candidate window (with its survivor count) and at level ends. Probe windows are half the window capacity, so two run in flight: after each probe launch the four leading counters are published into one of two host-mapped snapshots, and before the next window the host tightens its bounds from the older snapshot (its exact counts plus one row per survivor of the newer window) while the newer probe runs; it settles exactly only when even that cannot cover the minimum window ([D-450](../40_decisions.md#d-450)). Every small read-back, exact or lagged, goes through `publishDeviceWords`: one block on the compute engine writes the words into host-mapped pinned memory and an event behind it makes them visible, because a small copy-engine transfer waits for an engine switch on both sides under the Windows display driver model. The probe kernel strides over survivor ordinals and the windowed driver launches at most the resident blocks; it probes the whole-key map only when the answer can reach an event (beside a satisfied subkey, or once the mandatory terms hold). A buffer that cannot take the next window spills its rows to the host pages (`spillRows`, counted in `frontierSpills`); spilled pages upload into the consumed current buffer after its device rows. A request drain restores the two frontier counters after resetting the others, so pending store rows survive it. The level-end cutoff prunes the host pages (`pruneFrontierByCall`) and the device rows (`pruneDeviceFrontierByCall`, a stable `cub::DeviceSelect::If` into the free buffer). The planner sizes the store from the `frontierStore` work dimension (the unclipped measured level, raised by spill pressure) up to `phase2FrontierStoreMaximumRows` (8,388,608 rows, 448 MiB per buffer); with dedicated frontier-store blocks the store instead takes the largest power of two up to that ceiling whose two buffers fit those blocks, whatever the demand (the measured demand lags a level larger than every earlier one), otherwise the budget's common scale factor applies; nothing in it outlives the pass ([D-441](../40_decisions.md#d-441), [D-465](../40_decisions.md#d-465)).

The original whole-frontier method remains an isolated reference for direct tests. Pinned frontier transfer scratch uses five GPU-host-pool blocks with copy pages clamped to their actual capacity: downloads, packed prefix sizes, counters, a second frontier-upload stage, and mapped counter snapshots. Frontier-page uploads alternate between two staging blocks with asynchronous copies, so filling one block from the host frontier (`PagedVector::copyRunTo`, one copy per page span) overlaps the other block's transfer; stream order keeps every upload ahead of the download that reuses the first block. Consecutive transfers fill one GPU execution window of up to 65,536 frontier rows before prefix preparation. After the measuring preparation, `packPhase2PrefixSizes` packs each prefix's payload, variable and secondary lengths into one 64-bit word on the device (in the growth buffer's second node sort-key scratch, unused by this driver); the host downloads 8 bytes per node in whole pinned blocks instead of the 36-byte prefix records, keeps the words in GPU-host static-pool pages, and packs prefixes into windows across transfer boundaries, walking the words span by span (`unpackPhase2PrefixSizes`). The frontier stores reset their logical disk prefixes and reuse bounded resident pages; the prefix-size list still resets while retaining its pages. The level-end cutoff asks the sealing controller once for one keep flag per scheduled call (`retainedCalls`); when no call is cut off the frontier is not scanned, otherwise `pruneFrontierByCall` compacts it in place in order. Actual frontier, prefix and candidate capacities still determine continuation. Only spilled rows cross to the host and back. The call/task/filter image is supplied as a measured, fitting immutable schedule.


## Automatic block-budget planning

`phase2_allocation.hpp` / `planPhase2Buffers` measures immutable projection, task
and filter costs first, then fits growth, ordering and evaluation
capacities into whole 2 MiB blocks. Relative demand starts from a retained complete
GPU run. Exact array/alignment/CUB costs govern the fit; unused blocks are lent to
flexible processing columns. Prefix and candidate allocations stop at the amount
one bounded execution window can use, independently of host transfer-page size. `observePhase2WorkDemand` updates proportions
from completed-batch occupancy and drain pressure; these values never select proof
results. Production uses the planner for every chunk.

Minimum output storage covers one firing supported by the existing CPU storage
contract: transformed strings and sealed spans fit a host arena block (currently
256 KiB). The GPU reserves a head plus one marker key, one complete level run,
bounded head dependencies, remaining-argument identifiers and marker argument
slices. Larger result collections drain through value/key windows. This preserves
the CPU representation rather than adding a new GPU payload limit. A capacity
request beyond that contract still requires a shared payload-format extension.

A non-fitting minimum input asks the packer to reduce its LB chunk; the production
packer must assert when its single LB still cannot fit. A pending output floor can
borrow from processing storage; if the floor itself cannot fit, planning reports
that further segmentation is required without consuming work.

## Capacity profile selection

`CudaPhase2BlockPool` provides the common allocation substrate for the streaming
integration. It leaves the larger of 256 MiB or five percent of total VRAM free,
reserves the remaining working budget, capped at 4 GiB by default, in 2 MiB
blocks, and lends validated
contiguous runs. Full storage is observed through free-run counts before a
reservation; release synchronizes before a generation is reused. Its ownership
directory uses the existing host static pool. `CudaPhase2Arena` suballocates
256-byte-aligned arrays from a lease. Projection, task, filter, growth, ordering
and evaluation owners accept that arena and leave borrowed device arrays to its
owner at destruction; pinned host allocations and CUDA streams retain their own
lifetimes. Existing production calls still select independent reservations until
the streaming controller supplies pooled arenas.

`run_modes.cpp` / `fullRun` selects the common full-run capacity tuple for both
full execution and shortcut replay (`Prover::phase2ShortcutProfile`
is false). Replay consumes the same FTA workload and therefore needs the same
reservation. The historical smaller tuple remains available to isolated native
capacity tests; the production shortcut no longer selects it.
The common full-run profile reserves fixed storage at process initialization.
Its arrays use these ceilings; kernel loop bounds use the actual occupied counts.
The retained pre-D2 corpus census supplied the sizing baseline below. These
reservations provided at least twenty percent headroom for that baseline; the
new D2 attempts exceed two of the limits, as recorded after the table:

| Full-run buffer | Validated baseline maximum | Current reservation |
|---|---:|---:|
| Accepted events and matching ordering entries | 5,214,814 | 6,291,456 |
| Frontier records | 2,888,475 | 3,670,016 |
| Prefix payload values | 88,141,662 | 109,051,904 |
| Prefix variable values | 23,900,090 | 29,360,128 |
| Prefix secondary values | 4,426,235 | 6,291,456 |
| Evaluation reverse-owner work | 2,880,228 | 4,194,304 |

The common raw-request ceiling remains 524,288 and the candidate window remains
16,777,216. Projection, filtering and evaluation algorithms are unchanged. The sizing removes 417 MiB of fixed columns from the enlarged
profile, plus library-workspace savings. The startup `[GPU-PHASE2-MEMORY]` report
includes all fixed owners and library workspace: the validated full profile
previously reserved 7,169,784,452 device bytes. The induction-producer descriptor adds
2,048 bytes and the D1 reverse-owner expansion adds 16,777,216 bytes, giving a
measured pre-streaming total of 7,186,563,716 bytes. The wide owner/value window
offsets add 8,388,616 bytes in the full profile, giving a calculated
7,194,952,332 bytes. The bounded output-planning column adds 57,344 bytes,
giving a calculated 7,195,009,676 bytes for the historical fixed profile before automatic allocation. The reverse-owner field sizes only
the eight-byte `DeviceEvaluationOwnerWork` array; request, encoded-hit and firing
counts size the unchanged CUB workspace. The complete D1 trace measured 2,880,228
owners from 60,799 requests before the former 2,097,152 limit asserted. See
[D-355](../40_decisions.md#d-355). D2 required 4,224,638 owners from 60,782
requests. An experimental 8,388,608-owner reservation passed that point and
reached 5,511,208 owners, then exceeded the separate 6,291,456 accepted-event
ceiling. The maintainer restored the original reservation and deferred bounded
GPU generation/evaluation and joint AND-head induction to main-based sessions.
See the retained experiment
[D-371](../40_decisions.md#d-371)
and the [D2 handoff](../../fta_ladder/D2/current_proof_state.md).
A successful allocation is not
a performance guarantee: acceptance compares existing-stage runtime alongside
full proof-artifact identity, verifier results and the connected Nat kernel build.
See `D-348` and
[`rt_restoration_plan.md`](../../GPU/rt_restoration_plan.md) for measured evidence.

## Definition — GPU-ported means exact semantic work on the device

The evaluation owner also exposes `beginEvaluationWindows` and
`nextEvaluationValues`. Request gates execute once; wide owner offsets name
the complete logical expansion. Each owner window fits the owner, candidate
and encoded-hit columns in the worst case. Wide value offsets allow even one
encoded hit to span many value windows. The caller must retain each window's
output before requesting another. These doors are directly tested with
one-item work buffers; production routing is integrated and the remaining shared-payload segmentation limitation is recorded in [`batched_gpu_plan.md`](../../GPU/batched_gpu_plan.md).

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
- GPU proofs are valid and deterministic; cross-backend proof histories may differ. `prover.cpp`, `Prover::analyzeExpressions`, reports `Number proven theorems` from the complete `globalTheoremList.size`. Compare the authoritative exported theorem files and proof artifacts before landing; counts alone do not establish equivalence.

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
| Task scratch | Frontier nodes, candidate keys, request tokens, hit indexes | Allocate from pre-sized device scratch buffers; reset cursors, never allocate per request. |
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
| `IntEncodedExpr` statement pages | Contiguous `IntEncodedExpr[]` plus logical-block slice | Copy field-for-field; the 240-byte record is already flat, padding-free and trivially copyable. | `memory_infra/int_encoded_expr.hpp · IntEncodedExpr` |
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

[`phase2_projection.hpp`](../../../GL_Quick_VS/GL_Quick/src/gpu/phase2_projection.hpp)
now defines the complete resident pointer-free image:

- arena-relative logical-block descriptors for all 24 columns;
- byte-identical contiguous `IntEncodedExpr` rows in original relative order,
 prefiltered on the host by the logical block's signed premise-core union
 (`memory.hpp` and `forEachPremiseCore`); singleton subkeys and singleton whole
 keys across all four hash memories cover the request universe. Goal-only cores
 are not uploaded, and neither is a row no live rule premise accepts on its
 `u_` literals, repetitions and scope (`memory.hpp` and `premisePatternAdmits`,
 the same union the processor statement filter applies). Argument patterns,
 iteration limits, and frozen scopes remain
 device filters. The reserved host scratch uses one two-polarity byte per name.
 A host-only source-ordinal column follows retained rows through shard merging;
 `Phase2ProjectionArena::projectedStatementIndex` translates processor split
 stumps to dense device ordinals before task upload;
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
- a host arena whose fixed-capacity columns (`ProjectionColumn<T>`) reserve
 their ceilings once, assert before every append, and retain their
 allocations across `clear`;
- a build-once projection ([D-425](../40_decisions.md#d-425)):
 each bounded construction window reserves its twelve shards'
 measured capacities laid end to end (`addProjectionCapacity`), and each shard
 is a slice arena whose columns view its part of that storage, so the shards
 claim independent logical blocks in parallel and write straight into the
 final buffer; `Phase2ProjectionArena::adoptShards` then computes every
 shard's column bases, adds the window's stream bases, and lets the chunk's
 persistent `Phase2ProjectionWorkers` take whole columns (each moves
 that column's used prefix of every shard, in order, to the backing column's
 end inside the same storage and rebases its nested offsets and slot indices
 through the per-column core `adoptShardColumn`), byte-identical to building
 every block in one arena; the caller restores the canonical logical-block
 descriptor order before append upload into the complete device image. A
 retained-only host image holds sealing lookups and scheduling ordinals.
 The backing columns, each shard's construction scratch, retained image,
 schedules and object directories use the independent GPU-host reservation.
 Upload completion precedes lease reuse; no projection image is deloaded.
 Sealing download columns acquire their fixed lease only after construction
 windows have returned theirs (D-484);
- a CUDA owner whose 24 device arrays accept only bounded used prefixes and
 never grow during upload; production borrows the planner's block leases,
 while standalone fixed allocations remain available to the transfer twins;
- a dependency-staged upload door whose fixed page-locked arrays queue the nine
 evaluation-only column prefixes on a nonblocking stream while the host builds
 task/filter/growth schedules; the join occurs before the first kernel, and the
 fifteen early columns retain their ordinary synchronous copy; actual device
 ownership follows the measured block plan, and direct allocation tests cover
 the standalone fixed-capacity owners;
- native fixed-load probes for decoded names, the ten source byte-map views
 and four derived prefix-transition views, the
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
transfer byte-for-byte on native CUDA. Production request growth, evaluation and
firing construction consume this image through the bounded execution owners.

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
read-image capacity twin. It walks one logical block with private arena scratch
for shared keys and exact prefixes, without proof side effects, and counts name lookup slots; rule-interner strings; all ten
byte-key maps and their encoded keys, value blobs, and fixed-load hash slots;
 the derived reverse map's safe owner/entry upper bound (key bytes alias forward blobs); all seven actual plain-data
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
preserves the processor's configured `maxNumberFilteredStatements` acceptance prefix and only the global
retained prefix has a fixed arena ceiling. The schedule processes calls in one
bulk launch family and orders retained indices by decoded-name lexical rank then
original statement index. The precomputed name rank makes that composite key
exactly equivalent to the processor's stable name-only sort without comparing
variable-length strings inside the sort.

`gpu/phase2_projection.hpp::Phase2FilterScheduleArena` owns both the fixed host
call column and its exact class image. Each call descriptor selects one resident
logical block, one of the four whole/subkey registry pairs, the iteration
ceiling, the processor whole-key widening flag, and the statement count. A
fixed open-address table interns all six fields, checks equality after hashing,
preserves first-occurrence class order, and stores one class index for every
original processor call. The complete FTA shortcut contains 24,229 calls but
only 8,853 exact classes: class representatives examine 8,178,826 rows instead
of 89,618,957, a 10.957-times reduction in predicate rows.

`gpu/phase2_cuda.cu::CudaPhase2FilterSortBuffer` owns original calls, class
calls, the original-to-class map, original and class count/offset columns, two
compact 64-bit retained-key arrays, and one shared CUB scratch allocation. Count,
exclusive scan, predicate replay, stable compact emission, and radix sort run
once per class. A fixed expansion kernel maps the class count and offset back to
every original call without copying retained rows. CUB segmented radix sort uses
each class's offset range, including one final end-offset sentinel. Each 64-bit
key contains a full 32-bit decoded-name rank and 32-bit original statement index;
class identity lives in the span directory instead of consuming key bits. Growth
decodes only the low statement-index field from the shared span, while its events and every
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
`proof/phase2_executor.cpp`, `Phase2Executor::execute`, resets six thread-local depth
ledgers for every executor task, lets
`memory.cpp::HashRequestGenerator::generateEncodedRequestsStatic` and
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

The historical fixed-profile prefix census measured 14,074,478 payload values, 4,369,164
normalization variables, and 951,515 secondary variables at the simultaneous
frontier peaks. That profile applied 25 percent headroom and rounds each upward
to 262,144-value blocks: 17,825,792, 5,505,024, and 1,310,720 values, or exactly
94 MiB across the three pools. With one header per frontier node, canonical
node-order arrays, two index lists, two flag arrays, fixed candidate-window
storage, and retained CUB scratch, the historical growth owner used
812,965,447 bytes (775.304 MiB). Its direct allocation fixture now expects
812,965,463 bytes after adding the two candidate-window counter records.
Current production capacities come from the automatic budget planner. The task
counters increment
only for owner-accepted subkeys, matching `g_growthMatchCount` without counting
whole-key-only evaluation events or downloading the event ledger. A
production-sized RTX test allocates and releases the complete
owner and asserts the exact byte total. Copying maximum-size prefix payloads
into every 56-byte frontier row remains forbidden because it would dominate
device memory.

The historical complete-growth route uses fixed candidate storage. Stable radix passes order
node spans by original call, run, statement path, and candidate; one inclusive
scan assigns exact candidate ordinals. Consecutive half-open windows of at most
16,777,216 records cover every ordinal exactly once. Short-thread and 64-lane
cooperative kernels apply only the map-independent gates, stable flagged
selection compacts survivors in ordinal order, and one suffix-only probe kernel
performs exact subkey, owner, whole-key, and term work. The retained window owns
16-byte attempts, one-byte flags, and four-byte survivor indices; every boundary
asserts and no window size can alter proof order. Reusable CUB scratch is
12,868,095 bytes; the recorded historical complete ownership was 1,275,332,748 bytes.
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
to its stack. A caller that probes several maps with one key (the survivor probe's
subkey and whole-key maps) digests the key once with `hashProjectedSegmentedKey`,
which folds each identifier as one word low byte first (bit-identical to the
byte-wise FNV-1a over the contiguous key), and probes each map with
`findProjectedSegmentedByteMapEntryHashed`. The ordinary event/frontier ledgers and
later canonical ordering are unchanged. The direct fixture lowers the threshold to four so short nodes, long
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

`growthEventPathComponent` bounds each of the eight path keys to 131070.
`kDeviceGrowthPathRadixBits` therefore limits their CUB radix range to bits
[0,17), with a compile-time bound and exhaustive position-code test. The other
three component sorts retain 32 bits. All eleven stable sorts, tie handling,
event membership and the segmented growth-position scan remain unchanged.

The ordering owner retains every recordable request. Its arrays contain event
sort keys, indices, task-local growth scans, request tokens and reusable library
scratch; no key table or duplicate history remains. At the documented ordering
capacity the arrays consume 79,691,780 bytes before library scratch.
`CudaPhase2OrderingBuffer::fixedAllocationBytes` reports the complete ownership.

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
- emit a token for every recordable event and order tokens for cutoff
 selection; repeated semantic request identities are preserved.

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

**Historical evaluation-sizing baseline.** The following peaks and initial
ceilings describe the original evaluation campaign; current production uses
the common profile in [Capacity profile selection](#capacity-profile-selection).

The retained FTA peaks were 72,674 requests and dependency passes; 1,049,978
reverse owners with at most 804 per request; 47,930 subset candidates with at
most four per request; 46,654 encoded hits; 82,620 local values with at most 39
per hit; 62,834 total firing records; 1,935,835 generated bytes; 189,815 level
values; 184,494 origin dependencies; 1,608 marker keys; 780 marker remaining
arguments; and 1,345 marker-argument slices. The initial fixed next-power-of-two
ceilings were: 131,072 requests, 2,097,152 reverse-owner work items, 65,536 candidate
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
the deposit, enforces marker purity in every scope (including OR branches),
retains only the separate integration-justified compound-demand exemption,
and scans transformed bytes for the exact ordis2 iteration cap. Surviving rows
write complete fixed firing headers plus arena-relative payload runs. The same
kernel fills levels for heads and demands with a bounded multiway
merge over the already-sorted premise and LMV runs: negative premise tiers are
discarded, duplicates collapse, and no dynamic temporary is needed. Head
provenance starts with `(source rule-interner id, LMV validity id)`, followed by
premises sorted through projection-time decoded ranks for expression then
validity; identifiers remain in their source namespace until host sealing.

The resident projection does not duplicate the analyzer-wide compiled-expression
map. Projected strings retain the compiled-core annotation needed by record
validity assertions. No whole-expression or marker mail-permission views are
projected. The block descriptor retains anchor bytes, block level, secondary
ceilings and execution switches. The kernel preserves incubator anchor/local
premise disintegration gating, OR-disintegration permission, ancestor-known
suppression and compressor handling; mail-only permission evaluation is absent.

Marker records write transformed keys in source order, remaining rule-interner
identifiers in decoded-byte sorted-unique order, and sorted-unique bare argument
slices into fixed arenas. Their admission-depth and secondary-number ceilings
are part of the fixed header. The implementation deliberately scans for each next
remaining argument instead of allocating a per-thread variable array; marker
runs are small and this preserves occupancy and the static-memory contract.
The direct RTX twin covers mapped and unmapped numeric keys, repeated `u_`
removal, deeper validity, source flags, CPU/GPU impure-marker rejection inside an
OR branch and acceptance after recursion-product classification, the non-derived-level exclusion, four exact
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
conversion. Its constructor records `Phase2EvaluationCapacity` and allocates
nothing; `downloadAndSeal` grows every processor download column, the canonical
index array and the per-record sealing scratch to the used counts of its window
and keeps the largest for later windows of the chunk, never beyond the capacity
([D-428](../40_decisions.md#d-428)). It copies
the used device prefixes, walks the doom-compacted canonical permutation, and
creates one ordinary sealed record chain per logical block. Generated
expressions, marker keys, and marker arguments copy from device byte slices.
Validity and premise identifiers decode through the same projected `NameMap`;
source implications and marker remaining arguments decode through the same
projected rule interner. The literal `implication` tag and already-computed
device flags complete the existing `FiringRecord` shape without reopening the
rule or any proof-state gate.

`memory.cpp::HashRuleEvaluator::applyFiringRecords` now has an explicit
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
customization and compiles [`phase2_cuda.cu`](../../../GL_Quick_VS/GL_Quick/src/gpu/phase2_cuda.cu)
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
 event order, growth positions, and preservation of repeated requests.
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

The historical fixed-ownership gate accounted for every production `cudaMalloc` request in
the six process-static owners. These recorded figures describe that profile,
not current automatic allocations. After main appended provenance-owner records to
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
fixed pooled-prefix and candidate-window growth owner, exact
per-task split-work counts, exact remaining-span census, stable
short/cooperative selection, canonical candidate windows, cooperative long-span
cheap gates, stable survivor compaction, and exact on-device stream reconstruction
are live. Exact processor event order, task-local growth positions, and
the then-active per-call deduplication passed direct unsplit, mixed-stump, and
duplicate-row twins. Production routing is live for the full pipeline and FTA shortcut mode: the
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
 fixed device ownership at that milestone was 1,275,332,748 bytes (1,216.252 MiB); the refined traffic
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
`GL_CUDA` gates the CUDA includes and the CUDA block of `Phase2Executor::execute` in
[`proof/phase2_executor.cpp`](../../../GL_Quick_VS/GL_Quick/src/proof/phase2_executor.cpp), the whole
sealing translation unit `gpu/phase2_sealing.cpp` (it downloads the device
buffer, whose methods live in the `.cu` file), and the CUDA unit tests. A
build without `GL_CUDA` has no CUDA route at all: selecting one
(`--GPU` / `--phase2-backend cuda` / `use_gpu: true`) asserts at prover
construction, never falls through to the processor. Both Linux builds are
gated on Ubuntu (WSL2, CUDA 13.3, g++ 15): processor and CUDA.
Every test compiled only under `GL_CUDA` is registered with
`CUDA_TEST(suite, name)`, not `TEST`: `main.py`'s processor-route gate
(`gl_quick --unit-tests --phase2-backend cpu`) leaves those tests out, so the
default run needs no CUDA device
([D-457](../40_decisions.md#d-457)).

The closed boundary is `main.py` complete on Windows and Linux — the FTA
shortcut AND the full incubator/main pipeline. The processor route is the product default
(`ProverParameters::use_gpu`, false; no shipped config pins a batch),
`main.py --GPU` selects CUDA for every batch of the run,
`--phase2-backend cpu|cuda` is the explicit whole-run override, and selected
CUDA asserts rather than falling back. A CUDA-selected batch also owns its counterexample filter: sixty-four independent conjecture clones share setup,
and automatic device planning divides their images into bounded chunks. There
is no processor replay. Both backends page through the working-set steward by default
(`allow_ssd_deload`): the CUDA route packs chunks from the measurement each
logical block records at the end of its phase 1, projects and schedules each
block under a claim, keeps a steward window open while the device output grows
the pool, and shares name / rule columns by comparing against the shard's own
copy; counterexample-filter clones run resident-only
([D-409](../40_decisions.md#d-409)). Phase 1, Phase 3, Python proof-graph processing, and
verification remain on the processor. The approved split-stump discovery
prepass and fixed-array sealing also remain on the processor, while semantic
request work through provenance, marker payloads and deactivation eligibility
executes on the device. Host metadata selects the GPU request-key cutoff and
orders retained firing content. Automatic device ownership accounts for all
arrays and library scratch within one budget; a full run streams projection chunks
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

### Bounded parallel construction waves

`planProjectionConstructionWaves` partitions each existing device chunk into
consecutive host-construction waves. The sum of unshared usage in each wave
fits one construction shard in every column. This is a worst-case guarantee
for dynamic claiming: even one worker receiving the entire wave fits the
reserved arrays. Each wave resets the twelve construction shards and appends
their results into the same final host projection. (The production route no
longer builds in waves: each chunk builds once, the shards sized from the
recorded per-block usage — see the build-once bullet above; the planner remains
unit-tested.) Canonical descriptor order,
source-ordinal remapping, GPU chunk boundaries, and device capacities stay
unchanged. Two average shares alone were not a bound on a worker's actual load.

## Witness-class mail boundary

Processor and CUDA firing records carry no transitive mail-eligibility or marker-permission fields. The two permission sets and their projection columns are removed; remaining projection kinds retain their numeric identities. Circulating mail is checked at its publishing boundary by ordinary integration-witness presence. The GPU still computes every surviving inference gate and provenance field; removing a mail-only predicate does not transfer inference evaluation to the processor.

## Request-frontier disk storage

`RequestFrontierStore` replaces the two unbounded host frontier vectors in
`CudaPhase2GrowthBuffer::runRequestGrowthWindows`. Each logical sequence is an
ordered disk prefix followed by a resident tail containing at most
`phase2FrontierHostCacheRows` rows (at most 32 MiB of row payload). Page slack,
page directories and the fixed transfer buffers are additional bounded storage.
A full tail is synchronously written through bounded contiguous transfers before
another append can allocate more resident rows. There is no row-proportional
disk index. Checked 64-bit row counts and file offsets cover the disk sequence.

`Phase2Executor::execute` prepares two reusable `.deload/requests_frontier_*.bin`
files and their current-thread positioned handles before growth. Their owner
closes every handle at phase exit, before another batch can purge the directory.
Growth receives borrowed files through `RequestFrontierStorageOptions`, passed
by `Phase2SealingArena::runWindows`. The policy is
`parameters.allow_ssd_deload && !ceFilteringActive`; null file pointers retain
the previous resident-only path, including its existing capacity assertion. Both paths allocate resident pages from the independent GPU-host pool; permission to spill is unchanged.
`enable_extent_deload` continues to select the logic-block image representation.

Device rows still run first. Host windows copy the same ordered rows from disk
and the resident tail into the existing upload stages. Stable call pruning uses
bounded reads and writes survivors no farther than the read cursor; it cannot
overwrite unread rows. The stored identities, split stumps, request multiplicity,
accepted-subkey counts, firing contributions and complete-level stopping seam
are unchanged. A file or bounds failure asserts; there is no request replay,
discard, heap spill or processor evaluation.

`Phase2GrowthStreamResult::frontierStorage` reports request disk reads/writes and
bytes separately from `frontierSpills`, which still counts device-to-host spills.
Resident payload and retained storage peaks (including each fixed disk transfer buffer) are separate from logical
`maximumHostBackingBytes`, which includes the disk prefix. These are observation
counters only. Request files are transient raw records, not canonical logic-block
images or restart checkpoints.

Tests: `test_phase2_streaming.cpp` / `request_frontier` and
`test_phase2_projection.cpp` / `cuda_bulk_growth_matches_request_gate_content`;
negative probes run through `.scripts/tests/test_request_frontier_asserts.py`.
See I-296.


## Independent GPU-host capacity planning

`gpu/phase2_host_plan.hpp` / `planPhase2HostStorage` counts block-rounded host
leases before any chunk allocation. The 4 GiB default is independent of the
device budget and the logical-block pools. Pool metadata, device-pool metadata
and pass directories are already live and subtracted first. The planner also
requires a sufficiently large contiguous free region, so free bytes split across
holes cannot justify a lease that will not fit.

Fixed columns include retained sealing lookups, task/filter/growth schedules,
chunk directories and window/shard/worker object storage. Construction combines
one backing window, the previous upload when overlapping, first-fit address gaps
from alternating window sizes, twelve independently
rounded shard scratch leases and each shard's largest per-block dictionary
bound. Paged dictionary bounds include key locations, page-contained key bytes,
map values, both hash generations, prefix heads/transitions and pointer
directories. Download columns and request-growth support occupy the later
execution stage, after construction windows are released; their storage is not
summed with released window payloads.

`selectPhase2HostWindows` tries two upload windows, then one. If neither fits,
`Phase2Executor::execute` retries a consecutive prefix containing fewer whole
logical-block families, preserving their first-appearance order and each family's
task order. Every family is eventually processed. One family that cannot fit
prints its fixed/construction/execution requirements and asserts. A serial
window joins the uploader before reconstructing its slot.

Request support includes two bounded resident tails, page directories, prefix
measurements, five registered transfer blocks, retained-call flags and the two
disk transfer blocks when existing policy permits spilling. Resident-only growth
may consume remaining pool capacity beyond that provision and retains its hard
exhaustion assertion. Host pressure never enables disk or projection deload.
CPU production batches never call the lazy pool initialization door.

The spill files share the 50 GiB deload ceiling with the extent file
(`PositionedFile::kMaxDeloadBytes`, [I-350](../30_invariants.md#i-350)):
a growth that would pass it asserts.

`memory_census.cpp` / `writeReport` and `proof_run_reporter.cpp` / run reporting
separate reserved, committed, live, peak-live, registered and peak-registered
bytes. Arena component capacities are a breakdown of this pool, not extra heap
bytes. The component sum is not a simultaneous-lifetime peak; use the pool's
peak-live counter for that quantity.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
