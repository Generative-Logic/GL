<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# GPU acceleration of Phase 2 hashburst — analysis draft

> **Status:** historical exploratory analysis from 2026-08-27, superseded by the
> measured implementation below. The original forecasts remain visible as dated
> hypotheses; they are not current performance claims. The maintained authority is
> [`gpu_plan.md`](gpu_plan.md) plus the
> [agentic GPU cookbook](../agentic_swdd/20_core_concepts/09d_gpu_cookbook.md).
>
> **Source basis:** live `main`.
> Runtime and memory evidence named below came from the preceding
> acceptance run at
>; it is retained as dated evidence,
> not claimed as a fresh `main` measurement.

## Measured outcome — 2026-08-28

The FTA shortcut Phase 2 port is live with explicit
`--phase2-backend cpu|cuda` selection, processor default, no selected-CUDA
fallback, full provenance, deterministic doom selection, and canonical output.
The retained processor Phase 2 baseline is 110.333405 seconds. The first complete
CUDA route was 228.782 seconds; profiling and exact fixes reduced it to 19.2570
seconds, a measured 5.730-times acceleration on the RTX 4070 Laptop GPU. The
current split is 14.331816 seconds of exact CUDA work, including transfers, and
4.925 seconds outside that event interval.

The six fixed CUDA owners reserve 704,555,605 bytes (671.92 MiB). Peak provenance
is 154,845 eight-byte dependencies, or 1.181 MiB; provenance was retained rather
than removed. Both verifier passes complete 10,038 checks with zero failures, and
all 305 retained theorem and proof-graph artifacts match the processor baseline
byte for byte.

The corrected RTX 5090 planning band holds measured host and transfer time fixed
and scales the remaining device work by 3 to 4. It predicts 8.902-to-10.052-second
Phase 2, 1.916-to-2.163 times faster than the current laptop GPU and
10.976-to-12.394 times faster than the processor baseline. This is a model, not a
5090 benchmark. The binary embeds native `sm_89` and `sm_120` cubins so the actual
measurement can run without forward PTX compilation.

## Historical pre-port conclusion

The targets in this section were discussion-era forecasts before a prototype or
isolated Phase 2 timer existed. The measured outcome above supersedes them.

Phase 2 can profit substantially from a commodity GPU, despite request
generation and request evaluation both containing branchy work. The useful
property is not that every operation is naturally GPU-shaped. It is that Phase
2 is read-only on each logical block, its expression-stump subtrees are exact
independent work partitions, and its firing output is already captured as
records and merged canonically after the parallel join.

The defensible targets against the current 32-thread central-processor run are:

| Device and implementation maturity | Sustained Phase-2 acceleration |
|---|---:|
| RTX 4070 Laptop, first useful resident implementation | **4–8 times** |
| RTX 4070 Laptop, tuned queues and batching | **8–12 times** |
| RTX 4070 Laptop, aggressive upper target | about **15 times** |
| RTX 5090, literal port with coarse work | **2–5 times** |
| RTX 5090, resident flattened request and evaluation queues | **12–25 times** |
| RTX 5090, heavily tuned and batched across logical blocks | **25–40 times** |
| RTX 5090, largest request-generation kernels only | **40–60 times** |

The strongest practical planning number is therefore **15–25 times sustained
Phase-2 acceleration on an RTX 5090-class device**. The higher figures are
kernel-local or depend on enough batched work to fill the device. A one-hundred-
times whole-Phase-2 expectation is not supported by the present work shape.

These are engineering projections, not measurements. There is no GPU prototype
and no isolated current-`main` Phase-2 wall-time measurement yet.

## What Phase 2 currently does

The current Phase-2 path is centred on these symbols:

- [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp),
 `ExpressionAnalyzer::performElem2`: drives the request batches and streams
 each generated request directly into evaluation.
- [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp),
 `ExpressionAnalyzer::generateEncodedRequestsStatic`: filters and sorts the
 visible statement universe, then depth-first grows normalized request keys.
- [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp), `BurstSink`: performs
 the fixed dependency check and calls `checkLocalEncodedMemoryStatic` without
 buffering all requests first.
- [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp),
 `ExpressionAnalyzer::checkLocalEncodedMemoryStatic`: performs integer gates
 first, probes the encoded rule map, and on a hit enters the string,
 substitution, admission, OR, and equivalence-class paths that produce sealed
 firing records.
- [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp),
 `ExpressionAnalyzer::performElemPhase2`: after all parts join, applies the
 captured firing records to the logical block.
- [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp),
 `ExpressionAnalyzer::applyFiringRecords`: sorts and replays the records in
 canonical order, making the result independent of part completion order.

`performElem2` does not mutate the logical block while the request work runs.
Each executor writes only its own sealed page set and may lower the external
deterministic doom line. The post-join finalize is where logical-block mutation
resumes. That existing separation is the main reason GPU acceleration is
plausible without changing proof results.

## Request generation

### Why it is not naturally GPU-friendly

`generateEncodedRequestsStatic` is a variable-depth tree search. Different
candidates encounter different outcomes:

- a key may fail an owner probe;
- mandatory-containment state may reject it;
- scope, recursion-product, secondary-variable, or iteration gates may reject
 it;
- one candidate may be a complete request while another must grow further;
- subtrees have unequal sizes;
- normalized-key and cold-hash probes create irregular memory access;
- emitted-request deduplication adds another data-dependent hash lookup.

Putting one arbitrary tree node on one GPU thread would therefore create severe
branch divergence and load imbalance. Merely translating the present
central-processor recursion or stack loop into a kernel is the low-value
`2–5 times` case, even on a large GPU.

### Why it can still accelerate

The current expression split already exposes a correct parallel unit.
`produceExpressionStumps` names nodes in the same depth-first search tree.
`generateEncodedRequestsStatic` documents that the resulting stump subtrees are
disjoint and their union is the complete unsplit enumeration. The split is an
exact partition, not an approximation or a filter.

This supports three progressively wider GPU work shapes:

1. **One warp per stump.** Lanes cooperate on candidate fields, owner probes,
 normalization, and child emission. Divergence remains inside a warp, but
 independent stumps progress independently.
2. **Flattened frontier queues.** Each kernel consumes a level or batch of
 candidate states, writes surviving children into a compacted next queue, and
 sends complete keys into an evaluation queue. Prefix sums or equivalent
 queue allocation replace per-thread dynamic containers.
3. **Batch several logical blocks.** One launch carries work from multiple
 read-only logical blocks. This is important on a large GPU because a single
 logical block may not expose enough simultaneous stumps.

The existing split is currently tuned to the host: `proveKernel` deals stumps
into at most `logicalCores` buckets, which is 32 on the discussed machine, and
`kStumpsPerBucketTarget` asks the producer for four stumps per bucket. The dated
acceptance log nevertheless records completed split cases with 541–1,435
stumps, all collapsed into 32 host buckets. A GPU should consume the stumps or
finer candidate-frontier items directly; preserving the 32-bucket execution
shape would starve a high-end device.

The 541–1,435 observed stumps correspond to the same number of possible
warp-level work items. That is useful width for the RTX 4070 Laptop. On an RTX
5090 it is only a starting point: sustained upper-range performance requires
frontier flattening or batching across logical blocks so that uneven stump
subtrees do not leave most compute units idle.

## Request evaluation

Request evaluation is also mixed rather than uniformly GPU-friendly.

### GPU-friendly front half

The front half is predominantly bounded integer work over read-only data:

- the `BurstSink` dependency membership checks;
- validity identifier comparability and deepest-scope selection;
- ancestor-filter membership;
- recursion-product checks;
- sorted-unique level runs;
- normalized-key and encoded-map probes;
- iteration and rule metadata gates.

Once represented as packed arrays with offset-based hash buckets, this work can
run as a request-evaluation queue. Failed requests return cheaply. Requests that
reach a rule hit can be compacted into a smaller hit queue.

### GPU-hostile tail

After a rule hit, `checkLocalEncodedMemoryStatic` crosses into more irregular
work:

- decoding variable-length names and rule fragments;
- string span replacement and exact string materialization;
- rule-dependent admission and OR paths;
- equivalence-class expansion;
- variable numbers and shapes of firing records;
- sealing variable-length strings into the output record pages.

This tail has branch divergence, variable output size, and exact byte-level
ordering obligations. It is possible to put it on a GPU, but it is not the
right first proof of value.

The lower-risk experimental boundary is:

1. generate requests on the GPU;
2. execute the integer evaluation gates and encoded-map probes on the GPU;
3. compact rule hits into a deterministic queue;
4. perform exact string substitution and firing-record construction on the
 host initially;
5. preserve the existing canonical `applyFiringRecords` replay.

This boundary is a proposal for measurement, not an approved architecture. If
the host tail becomes dominant after generation is accelerated, the hit queue
provides a narrow and measurable second migration target.

## What the historical profile says

The dated FTA profile in
[`runtime_ideas.md`](../agentic_swdd/runtime_ideas.md), section “Where the time
goes now,” attributed 1,095.4 burst-seconds over 6,657 bursts:

| Region | Attributed burst-seconds | Share |
|---|---:|---:|
| Folded batch 2+3 grow search | 887.4 | 81.0% |
| Batch 1 pairing or grow work | 193.2 | 17.7% |
| Batch 2+3 filter and sort | 5.9 | 0.5% |
| Request firing, `STATIC_REQGEN_FIRE_EVAL` | 2.1 | 0.19% |

An event-tracing profile in the same document placed
`generateEncodedRequestsStatic` at about 92% of process central-processor time.
Inside the generator, the approximate split was:

| Generator work | Generator share |
|---|---:|
| Two owner-key probes | 56.87% |
| Request gates | 13.93% |
| Normalized-key construction | 10.41% |
| Candidate-pointer refill | 7.57% |
| Phase-4 merge sort | 3.98% |
| Generator body | 2.60% |
| Scope fold | 2.17% |
| Filter and emit | 0.70% |

This profile strongly supports attacking request generation first. It does not
prove that request evaluation will remain negligible after acceleration:
`STATIC_REQGEN_FIRE_EVAL` covers the firing-evaluation scope after the dependency
skip, and the measurement predates the current `main` source. A fresh segmented
profile is required before fixing the final migration boundary.

## MPU 0.1 static allocation and splitting

The MPU 0.1 shape is beneficial to a GPU. Static allocation and exact splitting
are prerequisites for an efficient implementation, but they do not make the
current host structures directly GPU-readable.

### Static allocation helps

The statified tree already has several device-friendly properties, documented
in [`09_static_memory.md`](../agentic_swdd/20_core_concepts/09_static_memory.md):

- persistent content is referred to through integer identifiers and arena
 offsets rather than ownership-rich heap objects;
- logical-block content is stable throughout Phase 2;
- scratch lifetimes are bounded to one executor task;
- pool use has measured upper bounds and asserts on exhaustion;
- the final record merge has an explicit deterministic contract.

These properties make capacity planning and bulk transfer possible. They also
permit fixed-capacity device queues whose overflow is an assertion during
development, consistent with GL's no-fallback rule.

The dated FTA memory report  recorded these
independent physical high-water marks:

| Pool | Peak |
|---|---:|
| Main | 1,840.750 MiB |
| Persistent | 10.719 MiB |
| Mail | 208.750 MiB |
| Logical-block body | 19.500 MiB |

The peaks are independent and must not be added as simultaneous measurements,
but even their conservative sum is only about 2.08 GiB. The active data
therefore fit in the discussed 8 GiB RTX 4070 Laptop and very comfortably in a
32 GB RTX 5090. The 12 GiB main-pool reservation is virtual capacity, not the
amount that must be copied to the device; only committed live pages and their
device indices matter.

### A device projection is still required

The current arena offsets resolve through host-side block tables, and current
cold containers contain host implementation details. Copying their bytes and
launching a kernel is not enough. A useful device projection would need, at
minimum:

- packed encoded statements and expression metadata;
- offset tables for variable-length rows;
- packed normalized-key sets and encoded-map buckets;
- validity ancestry and comparability data;
- immutable rule metadata;
- bounded candidate, request, hit, and firing queues;
- stable identifiers or offsets instead of host pointers.

The projection should remain resident across many bursts. Copying the whole
logical-block state over PCI Express for every call would erase much of the
gain. Host-to-device updates should consist of appended or dirtied regions;
device-to-host traffic should consist mainly of compacted hits or final firing
records.

### Splitting helps, with a different grain

The existing expression-stump split proves that Phase 2 can be partitioned
without changing the firing set. That is more valuable than inventing a new
approximate partition. For GPU use, however, a “part” should no longer mean one
of 32 host-worker buckets. It should mean a stump, frontier node, or compacted
candidate range. Several such ranges can still contribute to the same
per-logical-block canonical record merge.

The external doom-line semantics also have to survive exactly. GPU completion
order cannot choose the winning part. Each work item must retain its
deterministic stream position and invocation ordinal so the same lexicographic
minimum and the same winning firing chain are selected.

## Historical pre-port device estimates

### Discussed local machine

The machine used in the discussion has an Intel Core i9-13900HX with 32 logical
processors and an NVIDIA GeForce RTX 4070 Laptop GPU. A live `nvidia-smi` query
reported 8,188 MiB video memory, a 140 W maximum power limit, and maximum PCI
Express generation 4 by 8 lanes. NVIDIA lists 4,608 CUDA cores and 8 GB GDDR6
for this laptop model class.

On this device, a correct first resident implementation should target **4–8
times Phase-2 acceleration**. Queue tuning, reduced transfers, and batching may
reach **8–12 times**, with about **15 times** as an aggressive upper target.
Thermal and power limits matter because this is a laptop device.

NVIDIA source: [GeForce RTX 40 Series Laptop GPU
specifications](https://www.nvidia.com/en-us/geforce/laptops/40-series/).

### Strong commodity device: RTX 5090

The concrete stronger device considered is the desktop GeForce RTX 5090:

- 21,760 CUDA cores;
- 32 GB GDDR7;
- 512-bit memory interface;
- 1.792 TB/s memory bandwidth;
- PCI Express generation 5.

NVIDIA sources: [GeForce RTX 5090
specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/)
and [RTX 50 Series launch
specifications](https://www.nvidia.com/en-us/geforce/news/rtx-50-series-graphics-cards-gpu-laptop-announcements/).

Its CUDA-core count is about 4.7 times the RTX 4070 Laptop count, its video
memory is four times as large, and its bandwidth is much higher. GL will not
scale by those ratios automatically. The likely sustained range is **12–25
times Phase 2**, with **25–40 times** requiring a mature flattened and batched
implementation. A largest-logical-block request-generation kernel may reach
**40–60 times**, but that is not a whole-Phase-2 number.

### Why the estimate stops below hardware peak ratios

The limiting effects are:

- unequal stump-subtree sizes;
- branch divergence in gates and early exits;
- random cold-hash probes rather than streaming arithmetic;
- 64-bit integer hash operations;
- bounded width on small and medium logical blocks;
- duplicate suppression;
- variable-length strings and output records;
- host-device synchronization at the Phase-2 finalize;
- the deterministic doom line and canonical record-order obligations.

The RTX 5090 therefore improves both capacity and throughput, but it also makes
insufficient work width more visible. Batching and queue compaction determine
whether the device reaches the upper half of the estimate.

## Historical effect on the observed complete run

The dated acceptance run
 recorded:

- prover time: 206.087 seconds;
- overall pipeline time: 212.44392 seconds;
- outside-prover time: 6.35692 seconds.

The current log does not isolate Phase 2, so the table below is an Amdahl-law
scenario, not a measured prediction. It asks what happens if Phase 2 accounts
for either 90% or 95% of the observed prover time:

| Assumed prover share in Phase 2 | Phase-2 acceleration | Projected overall time | Projected overall acceleration |
|---:|---:|---:|---:|
| 90% | 12 times | 42.4 s | 5.0 times |
| 90% | 25 times | 34.4 s | 6.2 times |
| 90% | 40 times | 31.6 s | 6.7 times |
| 95% | 12 times | 33.0 s | 6.4 times |
| 95% | 25 times | 24.5 s | 8.7 times |
| 95% | 40 times | 21.6 s | 9.9 times |

Under those assumptions, the RTX 4070 Laptop target corresponds roughly to a
**40–70 second** complete run. A proper RTX 5090 implementation corresponds to
roughly **25–40 seconds**, with approximately **20–30 seconds** possible only
after aggressive tuning and only if the present run is at least 95% Phase 2.
The whole pipeline saturates far below the Phase-2 kernel acceleration because
the non-Phase-2 remainder is unchanged.

## Historical evidence plan — now completed

The implementation campaign completed the following evidence plan:

1. Measure current-`main` wall time separately for request filtering and sort,
 stump or frontier growth, dependency rejection, integer evaluation, string
 evaluation, firing-record construction, and canonical merge.
2. Record the distribution, not only the maximum, of stumps, candidate nodes,
 emitted requests, encoded-map hits, and firing records per logical block.
3. Measure bytes that would enter the immutable device projection and bytes
 dirtied per iteration.
4. Prototype only the exact request-generation replay against a captured
 read-only logical-block snapshot and require byte-identical request keys.
5. Add integer evaluation only after generation throughput and transfer cost
 are known; require byte-identical compacted hit records.
6. Compare the RTX 4070 Laptop and RTX 5090-class model using the same captured
 workload before deciding whether the variable-length firing tail belongs on
 the GPU.

The pass criteria are exact output identity and a sustained wall-time gain over
the existing 32-thread path. A fast kernel that depends on per-burst whole-state
copies, changes doom-line selection, or changes canonical firing records does
not qualify.

## Final assessment

The MPU 0.1 static-memory and split Phase-2 shape does give a GPU something
valuable to exploit. Static allocation makes residency and bounded queues
possible; the expression stumps give exact parallel ownership; read-only Phase
2 prevents cross-worker proof-state races; sealed records and canonical replay
preserve determinism.

The implementation projects immutable integer state into contiguous device
arrays, expresses the search as compacted queues, preserves deterministic stream
identities, and materializes evaluation, provenance, marker payloads, ordering,
and doom selection on the GPU. The processor performs fixed-array download and
sealing without semantic replay. The measured RTX 4070 Laptop result is 5.730
times Phase 2 acceleration; the evidence-backed RTX 5090 planning band is
10.976-to-12.394 times versus the processor baseline, pending an actual card run.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
