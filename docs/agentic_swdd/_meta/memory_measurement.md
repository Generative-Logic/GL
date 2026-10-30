<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Memory measurement

## Ordinary process and phase counters

The process census remains active in ordinary runs. `run_modes.cpp`, `fullRun`,
writes it through `memory_census::writeReport` into the batch diagnostics and
. It reports operating-system lifetime working-set
and private-byte peaks, sampled phase maxima, pool purpose/phase high-waters,
GPU host/device accounting and logic-block geometry. These denominators stay
separate: resident RAM, committed bytes, reserved address space and logical
container bytes cannot be added indiscriminately.

The old `MEM_MEASUREMENT` iteration aggregate and its `.rt/_memory_<batch>.log`
and `.html` writers are retired. The code, per-worker tag tables, sampling calls
and aggregate-only tests have been removed. Historical measurements using that
instrument remain historical evidence; they combine block samples from different
times and do not describe one simultaneous RAM peak.

## Selecting a detailed peak report

`GL_PEAK_MODE` is one mutually exclusive runtime mode selector:

| Setting | Behavior |
|---|---|
| Unset | Ordinary counters only; no scan file or captured snapshot. |
| `scan` | First full run: record peak X and a nearby sampled execution position for each batch. |
| `capture` | Second full run: read the saved positions and capture state Y for each batch. |

The modes do not automatically launch another run. Use the same workload and
matching executable for both passes. A capture asserts if a saved position is
missing or its phase differs. `main.py` supplies its own interpreter to the native Python helper;
`GL_PEAK_PYTHON` optionally overrides it.
`GL_PEAK_FULL_DUMP=1` retains the raw state for later typed offline refinement.
A full dump can fault additional pages into RAM after Y is saved: the second
run's later lifetime peak is not the snapshot denominator or a runtime reference.

The snapshot path does not depend on `HEAP_MEASUREMENT`. That separate compile-time
switch defaults to zero and controls only replacement C++ allocation operators
for the optional heap profile. With it enabled, `GL_HEAP_PROFILE_KIB` selects
allocation-site sampling. Keep it disabled for ordinary runtime comparisons.

## Structure ownership

The detailed report uses one saved resident-page list. Named byte ranges cover
Memory and LbMemory fields, owning containers, host GPU buffers, shared pools,
retained capacity, allocator storage, modules and thread stacks. A borrowed
pointer adds no second charge. Every parent equals its disjoint children and
the root equals Y; unidentified ownership stays explicitly unresolved.

Container registration belongs to the diagnostic ownership walk and saved-type
decoders, not an end-of-burst aggregate. `LbMemory::visitLiveBytes` and individual
container observers remain available for the snapshot and existing size queries.

## The heap census — live bytes and allocation sites

The whole-process census (`infra/memory_census.{hpp,cpp}`) sees the malloc
heap only as mimalloc's committed bytes, which cannot tell live data from
memory the allocator retains, nor name the owner. `infra/heap_profile.{hpp,cpp}`
adds two instruments, both measurement-only:

- **Live C++ heap.** In `HEAP_MEASUREMENT` builds that link mimalloc,
 `infra/heap_profile_hooks.cpp` replaces the global `operator new` /
 `operator delete` family. Each block still comes from mimalloc; its usable
 size is added to or subtracted from sixty-four thread-sharded counters.
 `heap_profile::liveBytes` feeds a new "Heap live (C++)" column of the
 per-step process table (`max_heap_live_bytes` in `[process-phase]`). Committed
 heap minus live heap is the allocator's retention and fragmentation. Before
 [D-427](../40_decisions.md#d-427)
 it also held mimalloc's never-released blocks above 64 MiB
 ([G-78](../50_gotchas.md#g-78));
 the heap-site profile cannot show those, because their frees are counted.
 `malloc` from C libraries and the CUDA runtime is outside the count.
- **Allocation sites (opt-in).** With `GL_HEAP_PROFILE_KIB=<interval>` in the
 batch's environment, `main` calls `heap_profile::enable` (not under
 `--unit-tests`, whose heap-profile test enables sampling itself). Call stacks
 are keyed on their innermost 16 frames. Every thread then
 samples one allocation per interval bytes it allocates, plus every
 allocation of at least the interval, recording the call stack and the
 sample's weight (the larger of its size and the interval) in a fixed table
 reserved once through `reserveAddressSpace`; a free removes its sample (a
 counting filter keeps the free path lock-free for non-samples). The census
 hook `noteCensusSample` snapshots the live weights per call stack at the
 batch's heap-live peak and at its largest end-of-barrier sample;
 `writeReport` prints the top 30 stacks of each snapshot as "Heap live by
 allocation site" tables and `[heap-site]` lines, symbolized through DbgHelp
 (Windows, the PDB beside the executable) or `dladdr` (elsewhere). A frame
 carries a name only when its address lies inside the named function
 (`symbolName`): without a PDB, or for a function the dynamic symbol table
 does not list, it stays unnamed instead of borrowing the nearest listed
 symbol's name. A label reads innermost program frame first (`a < b` = a
 called from b) with the container operation in brackets.

Sampling costs a stack capture per sample (tens of thousands per batch at
256 KiB) and about 90 MiB of reserved tables (262 144 call stacks, about a
million live samples); the live counter alone costs a
`mi_usable_size` and a relaxed atomic add per allocation and free. Nothing
reads either as a proof input (Rule 16).

## Two-run resident snapshot dissection

The diagnostic `GL_PEAK_MODE=scan` pass records the largest sampled working set
and its batch-local census ordinal and phase. The operating-system lifetime
peak is saved separately. `GL_PEAK_MODE=capture` replays that ordinal and
phase and captures one resident-page list for each native batch. The first
run's peak X and the second run's snapshot Y may differ; the structure tree
uses Y as its sole denominator. Neither the first peak nor the second run's
later lifetime peak is substituted for the snapshot total.

`infra/peak_snapshot.hpp`, `Position`, keeps the execution position.
`infra/peak_snapshot.cpp`, `checkpoint` and `capture`, call the existing
steward quiescence door at the selected single-threaded seam.
`gpu/phase2_cuda.cu`, `Phase2WindowUploader::quiesceForSnapshot`, waits for
the current upload while retaining its window; no buffer is freed merely
to make enumeration safe. The seam thread does not submit more work while
the snapshot is collected. These opt-in measurement hooks never
control a proof decision.

A separate Python process obtains the batch's working-set page list before
the owner walk starts. Its query buffer belongs to the collector process.
The paused batch then emits exact physical address spans for named fields,
live data, directory storage, held capacity, arena bookkeeping and recycled
pool blocks. Only intersections with the already saved resident pages count.
Reading a previously nonresident directory during enumeration cannot enlarge Y.

The requesting native process locks its Windows heaps before the external
collector suspends its threads. The collector verifies each thread's process
identity, records stack and thread-environment allocation ranges, and restores
the previous suspension counts. `GL_PEAK_FULL_DUMP=1` additionally preserves
a full memory dump after recording Y. Dump-induced page faults are excluded
from Y. The native owner walk follows after the helper resumes the process.

`infra/peak_heap.hpp`, `HeapView`, expands typed standard-library allocations
and nested definition fields. `infra/peak_snapshot.cpp`, `allocation`, uses
Microsoft's standard allocator alignment adjustment and mimalloc's usable
size to separate requested capacity from prefix and rounding bytes. That overhead
observation is compiled only with both `_MSC_VER` and `USE_MIMALLOC`; other
platforms retain the capacity and live-byte observations without those intervals. Windows
heap enumeration and the main-thread mimalloc visitor provide physical backing
ranges below semantic structure owners. Other mimalloc regions remain explicitly
uninspected until decoded; their membership is not evidence of complete ownership.
GPU projection snapshots include both host columns and retained construction
scratch. A named child takes precedence over physical pool backing, so a deep
allocator label cannot hide a shallower identified structure.


Paged-container census methods explicitly traverse directory and data pages
while capture is armed. Ordinary `LbArena::pageAt` resolves pages without
observer calls or observer thread-local reads; inactive field measurement
bypasses ownership scopes.

Byte spans, rather than one owner per operating-system page, preserve two
different 2 KiB GL pages sharing a 4 KiB Windows page. Shared backing stores
are counted once; non-owning references add no duplicate allocation. Every
parent must equal its disjoint children, and every final leaf belongs to the
same snapshot. Missing ownership remains an explicit incomplete result;
arithmetic closure alone does not establish a complete structure census.



## Restructured prover snapshot symbols

The stable snapshot owner label `ExpressionAnalyzerHeap` remains unchanged.
The C++ root type is `Prover`; the observed compiled map belongs to its value
member `definitions` (`CompiledDefinitionRegistry::entries`). Offline readers
subtract both field offsets to recover the root address. Nested registry fields
canonicalize to the same observed owner labels, and the registry wrapper maps to
the common ancestor so its bytes cannot conflict with a field's attribution.

GPU capture locals and the device pool belong to `Phase2Executor::execute` in
`proof/phase2_executor.cpp`. Symbol extraction resolves that frame and its four
capture anchors; the existing executable and symbol-file identity checks remain
mandatory. This correction changes offline attribution only.

Offline heap refinement recognizes native-observed fields both directly under
`Prover` and inside `CompiledDefinitionRegistry`. It leaves those complete native
observations authoritative, including the active alternative of definition
variants, and traverses only fields not already observed. Owner-path normalization
alone is insufficient: it merges labels after reading and cannot prevent a generic
reader from inspecting an inactive variant alternative.


## Returned-pool backing counters

`[pool-commitment]` reports current bytes, independent lifetime peak bytes, cumulative decommit/recommit bytes and trim calls for each pool. The process table's pool commitment is current backing, not the carve high-water. Windows `[process-memory] page_fault_count` is the process-lifetime page-fault count; unsupported platforms report the existing unavailable sentinel. A commitment reduction is not automatically an equal reduction in resident RAM or the lifetime process peak. Proposal comparisons retain those separate denominators.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
