<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Pipeline · Stage 3 — Prover `[DRAFT]`

> **Input:** `files/theorems/filtered_conjectures.txt` (from CE filter) + `files/theorems/compressed_external_theorems.txt` (external theorems + mirrors) + compiled `CoreExpressionConfig` map + `files/theorems/theorems.txt` (when running as later batch in a multi-batch pipeline).
> **Output:** `files/theorems/theorems.txt` (append); per-theorem origin maps used by stage 5 (raw proof graph emission).
> **Owner:** `prover.cpp` / `prover.hpp` / `memory.hpp`.
> **Entry:** `ExpressionAnalyzer::analyzeExpressions` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), called from `run_modes::fullRun` at [`run_modes.cpp`](../../GL_Quick_VS/GL_Quick/src/run_modes.cpp).

---
## Automatic GPU streaming controller

`prover.cpp` / `ExpressionAnalyzer::proveKernel` measures complete LB families,
packs consecutive families into the automatic VRAM block budget, and retains the
single-LB snapshot-plus-minimum-scratch assertion. All six device owners borrow
reusable block leases; deterministic host-builder assignments use measured per-column
reservations and merge descriptors back into original order.

`gpu/phase2_sealing.cpp` / `Phase2SealingArena::runWindows` connects request drains,
bounded owner/value evaluation, output reservation, repeated downloads and marker
assembly. Aggregate early exit retains complete logical growth levels before
complete output pages enter the shared RAM canonical sorter. Input stays immutable and no result deposits
statements before finalization. The working budget is the free VRAM after the
runtime reserve, capped at 4 GiB (`kPhase2DefaultGpuBudgetBytes`,
[D-412](../40_decisions.md#d-412));
`GL_GPU_BUDGET_MIB` replaces the cap. Capacities derive from that budget and
recorded/measured demand.
Pipeline acceptance is tracked in `docs/GPU/batched_gpu_plan.md`.

## Bounded Phase 2 execution and early exit

The automatic block planner now fits measured input plus coupled processing
capacities, seeds proportions from recorded pipeline demand and updates them from
completed-batch occupancy/drain pressure. Minimum output scratch covers the existing
CPU single-span contract; marker-key collections stream. Production selects these capacities for every measured chunk.

Before constructing any of the six CUDA buffers,
its `measureAllocation` query reports device payload bytes and the exact aligned
arena footprint. Queries include current CUB workspace and every array, with
256-byte alignment between arrays; they create no device-array ownership. Direct
tests compare reported payload with each real owner and compare aligned footprint
with the shared arena cursor. Adding a device column requires updating its sizing
query as well as the constructor. Production selection uses the common automatic block planner.

`CudaPhase2EvaluationBuffer::nextFiringWindow` packs consecutive values using
GPU-measured simultaneous bounds for all seven output columns. A large marker
uses the largest fitting consecutive key slice. Ready output must be downloaded
before reuse. If even the next minimum fragment does not fit, the method returns
`needsCapacity` with its requirements and an unchanged value/key cursor; it neither
writes nor drops the request. The production planner reserves the shared CPU single-fragment bound before
execution. Explicitly smaller component buffers retain this non-consuming scheduling state.

The bounded generation component `CudaPhase2GrowthBuffer::runRequestGrowthWindows`
keeps each frontier level in a pass-local device frontier store (two buffers of
`frontierStoreRecords` rows: the level and its children), spills only the rows of a
level that outgrows a buffer to host static-pool pages, and executes seed,
prefix and candidate windows on CUDA ([D-441](../40_decisions.md#d-441)). Cheap rejection uses candidate-buffer capacity; its surviving indices pass through separately bounded probe windows. It reserves append capacity before each probe launch and invokes a synchronous request drain before buffer reuse, preserving pending candidate indices and prefix payloads. Original call and
path identities survive the windows; cumulative split-work counters are 64-bit.
Production uses this bounded route; the whole-frontier method is a direct-test reference.

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
retrospective firing-selection list, or request-identity download. Phase 3 consumes these captured results and closes goals or discharges blocks;
phase 1 performs the same ordinary processing for incoming mail. The original RAM frontier, broad execution
windows, split-stump construction and shared RAM content sort remain. See
I-237 and `docs/GPU/no_requests_plan.md`.

Repeated requests are evaluated without duplicate-key history in RAM or VRAM.

The shared RAM-side firing sorter uses `PagedSortedOrdinals`: bounded initial runs, page-backed run directories and two-way merge passes. It has no fixed 64-run limit. Production GPU supplies raw-order records to this same RAM sorter. See [I-234](../30_invariants.md#i-234).


## Counterexample-mode gate

`prover.hpp` / `ExpressionAnalyzer::standardProcessing` begins with a
`ceFilteringActive` gate. Counterexample mode checks registered contradictions
and completed heads in `internalMailIn` against the loaded statement registry,
requiring both sides at `main`, then returns. New heads are still in the internal
channel at this seam, so checking the registry alone would miss their evidence.
This path performs no absorb, equivalence application, goal processing, mail
publication or delta clearing. It adds no inference and retains no request history.
The ordinary processing sequence remains unchanged below the gate. Each
counterexample clone runs one burst and is destroyed with its untouched channels
and unused class deltas. Both processor and GPU counterexample runs use the gate.

## What this stage does

The prover is the main proof engine. Given:

- a set of candidate conjectures (from the CE filter),
- a set of externally-provided theorems (axioms + previously-proven theorems),
- the compiled expression map (definitions),

it attempts to derive each conjecture by iterated hash-based inference on a grid of logic blocks, with full provenance recording for every emitted fact. Theorems that get proved are written to `theorems.txt`; theorems that fail stay on the candidate list and are discarded.

The prover's central data structure is `ExpressionAnalyzer` — a single per-batch object that owns all proof state. Every per-theorem proof lives in a child `Memory` block parented to `ExpressionAnalyzer::body`.

---

## Chapter map

- [ExpressionAnalyzer — the state container](#expressionanalyzer--the-state-container)
- [Full and shortcut FTA input identity](#full-and-shortcut-fta-input-identity)
- [The execution model](#the-execution-model)
- [Theorem load path](#theorem-load-path)
- [Core step primitives](#core-step-primitives)
- [Disintegration](#disintegration)
- [Integration](#integration)
- [`multiplyImplication` — Bell-partition equalisation](#multiplyimplication--bell-partition-equalisation)
- [Induction](#induction)
- [Equivalence classes](#equivalence-classes-and-addstatement)
- [OR & contradiction](#or-and-contradiction)
- [Origin tracking](#origin-tracking)
- [Where invariants live](#where-invariants-live)
- [Weaknesses](#weaknesses)
- [Open questions](#open-questions)

---

## ExpressionAnalyzer — the state container

Declared at [`prover.hpp+`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) and implemented across `prover.cpp`. Instantiated once per batch in `run_modes::fullRun`.

Selected top-level fields:

| Field | Purpose |
|---|---|
| `body` | Root `Memory` block — the top of the per-batch LB tree. |
| `anchorInfo` | Resolved anchor metadata (name, slot count, signature). |
| `anchorID_` | String tag of the batch (`"Peano"`, `"Gauss"`, `"IncubatorPeano"`, …). |
| `coreExpressionMap` | `map<string, CoreExpressionConfig>` — compiled expression configs (from stage 1). |
| `compiledExpressions` | `map<string, LogicalEntity>` — per-name metadata (category, elements, signature, definedSet). |
| `operators` | Set of expression names that are "operators" (have `output_args`). Consulted by Pass B's `isAllowedAsOperatorInput`. |
| `globalTheoremList` | The authoritative list of proved theorems + their method + metadata. Written to `theorems.txt` and `global_theorem_list.txt` after the run. Every insertion goes through the single door `appendGlobalTheorem` — insert-if-absent on the theorem string, first emission wins (the sink dedup, [D-216](../40_decisions.md#d-216)); a theorem proved through two paths (contradiction discharge + ordinary goal closure) lands exactly once. The companion `globalTheoremStrings` set is cleared together with the list at the export rebuild. |
| `permanentBodies` | `vector<Memory*>` — every LB that is currently alive (target proofs, sub-goals, auxiliary blocks). Retained across iterations. |
| `permanentBodiesCE` | Same, but the CE-filter-specific grid. |
| `box` / `boxes` | Per-core mailbox arrays for the main grid. |
| `contradictionTable` | Per-conjecture contradiction status (set during CE + main prover). |
| `implCounter`, `existenceCounter`, `andCounter`, `orCounter`, `variableCounter` | Monotone counters — name-minting sources for spontaneous compact operators (`implication<N>`, `existence<N>`, `and<N>`, `or<N>`) and freshly-renamed bound variables. The first four are seeded from the loaded `GL_binary_<Tag>.json` at startup so identifiers stay stable across batches; `variableCounter` resets to zero per batch. |
| `parameters` | `ProverParameters` from `parameters.hpp` + config. |
| `globalDependencies` | Per-expression dependency set maintained during main prover runs. |
| `ceFilteringActive` | Flag distinguishing CE-filter mode from main mode; consulted by hash-request generators. |
| `ceBody`, `permanentBodiesCE` | CE-filter grid mirror of the main `body` / `permanentBodies`. (The push-model routing mirrors `indexCE` / `boxesCE` are deleted — the parent→children index went with the vestigial `ParentChildrenMap` plumbing, 2026-07-03.) |

The class is large because most per-step primitives are inline methods on `ExpressionAnalyzer`. File-level refactor is on the post-FTA plan.

---

## Full and shortcut FTA input identity

`run_modes.py` / `_run_batch` calls `fta_shortcut.capture` after preparing FTA
conjectures and seeding `GL_binary_FTA.json`, immediately before native execution.
The generated `files/shortcut/full_input.zip` contains the exact compiler files,
simple facts, theorem pool and raw proof chapters at that boundary, with SHA-256
fingerprints for every member and for the definitions, configurations and goals.
The full run refreshes this archive automatically.

`shortcut_run` calls `fta_shortcut.restore` before native FTA. It validates the
archive and current input fingerprints before cleanup, restores shared scratch
files, and redirects the theorem pool and preceding chapters to `files/shortcut/`.
The captured conjectures are written as `replay_conjectures.txt`; the durable
shortlist remains the full-run source. The historical external snapshot stays
intact but is no longer a prover input. Captured proved theorems stay proved;
captured compressed externals stay external, with no remirroring or reclassification.
Both routes use the same GPU capacity profile and append both paired theorem
columns, so graph pruning keeps the same roots. Missing or changed inputs assert
and require a new full run. The shortcut omits earlier computation and connected
Lean certification, while retaining ordinary graph generation and verification.

## The execution model

A GL proof run is a sequence of **cycles**. Each cycle:

1. For each active LB, run the elementary step — produce hash requests from known expressions, look them up in the local `HashMemory`, emit new statements. `proveKernel` drives this as three barriered phase sweeps across all active LBs (see *Phase split* below), not one per-LB call.
2. After every LB has processed its local state for the cycle, drain mail: move `mailOut` contents into recipient `mailIn` buffers.
3. Next cycle: each LB processes its `mailIn`. Loop until no LB produces anything new, or a per-batch iteration cap is reached.

`prove(numberIterations, permanentBodies)` is the outer loop for the main prover and the compressor. The CE filter does NOT share it: `filterConjecturesWithCE` (`filter.cpp`) drives its own per-conjecture worker pool over `permanentBodiesCE` and never enters `prove` / `proveKernel` ([I-82](../30_invariants.md#i-82)).

The **elementary logical step** — today three phase helpers (`performElemPhase1` / `performElem2` / `performElemPhase3`) driven by `proveKernel` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), see the phase-split note below — is the inner core. It:

- Generates hash requests via the single generator `generateEncodedRequestsStatic`. Every caller drives it the same way — a list of mandatory-containment terms, checked at the finished whole key — and an empty list is a first-class input meaning no ingredient is mandatory (the CE filter and batch 5 both pass it). The obligatory stump, its builder and the pairing merge are deleted ([D-295](../40_decisions.md#d-295)); the target map is always the whole-key map and the grow depth is always the longest installed key. Its grow search prepares each stack pop's fixed premise prefix once — the premise pointers, the normalized key and the request-gate summary — and per candidate position redoes only the appended premise ([D-296](../40_decisions.md#d-296)).
- Matches each request against `overallHashMemory.encodedMap`.
- On a match, fires the rule — builds the conclusion instance, runs `addStatement` to install it locally, and optionally routes it upstream via `mailOut`.
- Handles OR-branch bookkeeping (`classifyOrScope`) and admission-map updates. Contradiction detection is no longer in the firing path — it moved to `dischargeContradiction`, run once per step from `standardProcessing` (see [Contradiction discharge](#contradiction-discharge)).

**Phase split + barriers (LB-split refactor, [D-113](../40_decisions.md#d-113) / [D-114](../40_decisions.md#d-114)).** The elementary step is three phase helpers — `performElemPhase1` (pre-hashburst mail absorb), `performElemPhase2` (the hashburst), `performElemPhase3` (post-burst absorb + sanitize). `proveKernel` runs them as **barriered sweeps with a join between each**: phase 1, then phase 2, then phase 3. Phases 1 and 3 are plain across-LB work-stealing sweeps (a local `runPhase` generic-lambda). Phase 2 is the **flat executor pool** ([D-115](../40_decisions.md#d-115)): one work-stealing pool over the flat list of `(LB, part)` tasks across all active LBs (an LB with `numberOfParts == N` contributes N tasks, never nested), each task running `performElem2` — the per-part request-generation + fixpoint unit — appending its firings to its own per-task `SealedPageSet` record chain, with the per-slot scratch arenas as its only other write surface ([I-83](../30_invariants.md#i-83)). After that pool joins, a per-LB finalize sweep (`performElemPhase2`) hands each LB's sealed part sets in part order to `applyFiringRecords` — a pointer-index sort over the chains' stable payload addresses ([I-77](../30_invariants.md#i-77)) — and applies them. The whole pool + finalize is wrapped in a ≤ 2-round loop (producer, then buckets — see *Statistics-driven split* below). The flat pool is what runs one heavy LB's expression buckets across N cores. The single-LB orchestrator `performElementaryLogicalStep` (entry `isActive` early-return, `RT_TRACKER_DECL`, all three phases inline) became dead once `proveKernel` called the phases directly and has been **removed** — the phase model is the only path. The hashburst dump now has two traps bracketing the live path: ENTRY at the top of phase 1, EXIT at the bottom of phase 3 (the EARLY-EXIT trap was removed once deactivation deferral made it dead — Rule 14, user-approved). The whole restructure is byte-identical at `splitCount == 1`; the byte-identity rests on [I-28](../30_invariants.md#i-28) (no cross-LB intra-cycle observation; sorted post-join drains).

**Initial GPU port (historical FTA shortcut route).** The live scheduler selects the processor or CUDA Phase 2 explicitly through `--phase2-backend cpu|cuda`; processor is the default, both full execution and shortcut replay support CUDA, and the selected CUDA route asserts rather than falling back. Split logical blocks still use the processor stump producer as the read-only work-discovery prepass. Every semantic request stage after that boundary runs on native `sm_89` CUDA: projection, task imaging, filtering, normalized-key growth, exact request ordering and deduplication, evaluation, expression materialisation, full provenance and marker payloads, canonical firing-record order, and deterministic doom-prefix selection. `CudaPhase2EvaluationBuffer::orderFiringRecords` compares decoded semantic content without moving variable payloads. `selectDoomPrefixes` mirrors the processor's sole-goal, contradiction, recursion, counter-example, and compressor rules, reduces the packed growth-position/part line, resolves the first triggering request when positions tie, and stably retains the winner's inclusive prefix. `Phase2SealingArena::downloadAndSeal` then downloads the fixed used prefixes and converts generated slices plus projected NameMap/rule-interner identifiers into complete ordinary `FiringRecord` chains without semantic replay. `applyFiringRecords` accepts those already-canonical chains without a second processor sort; processor part chains retain the unchanged sorting route and both paths share every deposit below it. Exact duplicate filter calls share one immutable class span ([D-317](../40_decisions.md#d-317), [I-212](../30_invariants.md#i-212)): the five-field signature is checked exactly, count/scan/emit/sort run once per class, and a fixed map restores each original processor call ordinal before growth and canonical ordering. Growth preparation builds one immutable measured pooled prefix per live node, then stable short and cooperative selections preserve ascending frontier-index traversal before suffix-only candidate expansion ([D-318](../40_decisions.md#d-318), [I-213](../30_invariants.md#i-213)). Production orders exact node spans by call, run, statement path, and candidate, covers them through consecutive fixed windows, stably compacts cheap-gate survivors, and performs normalized-key and owner probes only for that compact set ([D-319](../40_decisions.md#d-319), [I-214](../30_invariants.md#i-214)). The fixed growth owner is 812,965,447 bytes. Complete fixed device ownership is 1,275,332,748 bytes. The deterministic evaluator arrays are 37,267,500 bytes before reusable scan scratch and the retained production-size RTX allocation is 37,269,035 bytes including scratch. The first complete CUDA shortcut executed 48 device passes, passed 10,038 proof-graph checks with zero failures, and produced 311/311 artifacts byte-identically to the processor baseline. Performance is measured across the complete Phase 2 barrier window only; expanded Phase 3 and whole-prover totals are reported separately. Practical rules live in [`09d_gpu_cookbook.md`](../20_core_concepts/09d_gpu_cookbook.md); the first-port evidence lives in [`docs/GPU/gpu_plan.md`](../../GPU/gpu_plan.md), and the current acceleration campaign lives in [`docs/GPU/gpu_rt2_plan.md`](../../GPU/gpu_rt2_plan.md).

**Full-run GPU campaign.** [D-329](../40_decisions.md#d-329) supersedes only the shortcut/default-selection boundary above: absent an explicit override, each batch's `use_gpu` configuration selects the route — the processor route by default since [D-338](../40_decisions.md#d-338) (`main.py --GPU` passes `--phase2-backend cuda` to every batch) — and the same backend owns that batch's counterexample filter. `allow_ssd_deload` is a separate batch policy that both backends honour: the CUDA route pages through the working-set steward like the processor route, and only counterexample-filter clones run resident-only ([D-409](../40_decisions.md#d-409)). Full-run CUDA preserves each split-LB task family and streams exact capacity-bounded chunks through the fixed device allocation documented in the GPU cookbook. Structurally impossible request families and whole zero-work LBs never enter a projection. The host projection shares exactly identical immutable name/rule slices within a chunk and includes only the hash-memory kinds named by executable batches, while preserving every fixed descriptor ordinal with an empty view for an omitted kind. `BytesKeyStore::logicalByteCount` maintains exact projected key-byte usage at mutation doors so capacity census does not walk every key. The counterexample route groups sixteen private fact-template clones per CUDA call, projects the dedicated `counterExample` request kind, runs filter through sealing on device, and applies only the canonical retained records before contradiction discharge. The current implementation is an exact transfer baseline, not the target architecture: the next stage runs filtering/growth from a narrow projection and admits evaluation-only LB state only after the GPU reports unique surviving requests. Complete evidence and the live next action are in [`docs/GPU/gpu_main.md`](../../GPU/gpu_main.md).

**Whole-LB expression/bucket split ([D-201](../40_decisions.md#d-201)).** A straggler (see *Statistics-driven split* below), or a newly activated induction-zero LB with no prior work statistic, is split by partitioning its EXPRESSION search, not its rules. `proveKernel` dispatches it as one round-1 PRODUCER task that runs `produceExpressionStumps` on the whole LB at `g_splitCount == 1` (so `partitionAccepts` accepts every rule — the rule dimension is off): a bounded enumeration that grows the LB's base candidates level by level until it holds enough distinct ones (up to `kStumpsPerBucketTarget × logicalCores`), firing nothing. Before a short level is replaced by its children, any node the overall hash memory records as a request is retained as a **terminal pre-stump** ([D-202](../40_decisions.md#d-202)): it is dealt to exactly one bucket, rechecked by each batch against that batch's actual hash memory through the normal emitter → `BurstSink` path, and stops before depth-first growth. Its children remain the owners of every larger combination. `overallHashMemory` is sufficient only for conservative work discovery because every status-specific hash memory is populated from that superset; it never substitutes for the per-batch validation. The classify seam deals all stump records round-robin into `min(nStumps, logicalCores)` buckets and requeues one bucket part per bucket for round 2. The CUDA route deals them instead right after round 1's producer pool and runs the parts in round 1 ([D-442](../40_decisions.md#d-442)). The old two-cap rule→stump escalation is gone.

Every request a bucket part emits contains one of its stumps: the stump IS the growing candidate, seeded into the depth-first stack with its cursor past its own last element, so the run owns that node's subtree and nothing else ([D-283](../40_decisions.md#d-283) retired the older scheme that attached the stump per probe). A stump whose elements this batch's narrower rule registry does not accept is skipped before the seed frame ([I-156](../30_invariants.md#i-156)). The statement filter is built once per bucket, and the emitter's seen-set collapses a request two of the bucket's stumps both reach. The union of all buckets is the whole search, and the sorted firing-record merge makes it byte-identical to an unsplit burst ([I-77](../30_invariants.md#i-77)). A producer that grows only one base candidate cannot fan wider than one bucket — its whole cost is a single grow-tree that expression-bucketing cannot subdivide (the `[SPLIT] ineffective` report flags this; it is the one case the rule dimension used to parallelise and this axis does not).

The native growth primitive at `gpu/phase2_cuda.cu::buildProjectedNormalizedKey` and `projectedOwnerSetUSatisfied` builds exact multi-premise normalized keys and applies the projected owner-subkey u_ signature gate. `CudaPhase2GrowthBuffer::runRequestGrowth` drives those primitives through global frontiers after applying all map-independent and mandatory gates. `phase2GrowthPreparePrefixesKernel` builds one fixed pooled prefix and gate summary per live node. Stable selection compacts short and cooperative indices in original frontier order. Stable node-field radix passes and exact scanned spans then establish canonical candidate ordinals. Consecutive fixed windows apply cheap gates in the matching short or cooperative execution shape, compact survivors stably, and send only those records to suffix-only map and owner probes. The fixed growth owner is 812,965,447 bytes at retained FTA ceilings. `CudaPhase2OrderingBuffer::orderAndDeduplicate` reconstructs the exact processor event order with eleven stable radix passes, assigns cumulative task-local growth positions, compares full semantic keys after hash collisions, and retains the first request independently per generator call. `BurstSink::consume` and `checkLocalEncodedMemoryStatic` populate the observation-only `[GPU-EVAL-CAPACITY]` request, owner, hit, value, firing-kind, generated-byte, and variable-run census. Its retained peaks fix 2,097,152 reverse-owner work items, 131,072 local values, 65,536 firing records, 2,097,152 generated bytes, and the smaller provenance-run arrays. `CudaPhase2EvaluationBuffer::expandEvaluationWork` executes dependency and validity gates, reverse-owner expansion, subset compaction, ignore-u mapped encoded lookup, and `LocalMemoryValue` run expansion inside a complete fixed owner below 64 MiB. `materializeFiringExpressions` adds exact reverse substitution, repeated `u_` removal, rule-scope/purity/depth gates, complete head verdicts, and complete marker payloads. Exact combined levels and ordered head provenance are live. `orderFiringRecords` reproduces the processor comparator with fixed parallel index-array merges; `selectDoomPrefixes` reproduces the exact packed winner plus first triggering request and compacts the canonical output stably. `Phase2SealingArena::downloadAndSeal` converts that retained stream to processor-owned sealed pages using fixed host arrays and exact identifier decoding, while `applyFiringRecords`'s explicit canonical-input route preserves the device order without sorting again. Production routing remains before the GPU stream can replace `generateEncodedRequestsStatic`.

Sealed record sets accumulate ACROSS the iteration's rounds; an LB is finalised once, in the round after which it has no tasks left. Requeued work runs in the NEXT round, never appended to the running one — the working-set pager has registered this round's task list and dispatch cursor. The CUDA route's bucket parts respect the same constraint: `dealProducerStumps` appends them only after the producer pool has joined and closed its executor window, and the device chunk opens its own windows over the extended list.

**Statistics-driven split ([D-201](../40_decisions.md#d-201)).** The split TRIGGER is not a mid-burst cap — every main-path burst runs to COMPLETION (no cap, no truncation, no discard-and-redo). After the round loop, a single-threaded stats pass in `proveKernel` sets each active main-path LB's split for the NEXT iteration from this iteration's completed work. Each LB's `work` is the SUM of its parts' submatch counts (`lbTotalSub`), the split-invariant total ([D-117](../40_decisions.md#d-117)); with `T = Σ work` and `C = logicalCores`, the static helper `isStraggler(work, T, C, min_split_work)` returns `work > T / C && work >= min_split_work` — the idle-core fair-share (an LB alone exceeding the balanced per-core load, so parallelising it uses the idle cores) gated by the per-bucket setup break-even. A straggler's `Memory::numberOfParts` is set to `logicalCores`, everything else to `1`; recomputed every iteration, so an LB whose work falls back below the bar returns to unsplit.

Classifying on the split-invariant SUM (never the max-over-parts, which scales ≈ `work / parts` and would thrash — a split LB looks light, de-splits, re-splits, the retired [D-111](../40_decisions.md#d-111) band) is the anti-thrash property: a heavy LB measures heavy whether split or not, so it holds at split. All-integer arithmetic makes the verdict — and thus the split set — a deterministic function of the deterministic work totals. `numberOfParts` persists across iterations (never reset). Whether a batch runs the split at all is the per-config `lb_split` flag ([D-231](../40_decisions.md#d-231)): the gate is `mainPath = parameters.lb_split && !parameters.disable_lb_split` — `incubator_mode` is no longer consulted. Every production incubator config sets `lb_split: true` ([D-328](../40_decisions.md#d-328)); balanced small-LB grids still remain at one part because the straggler predicate is false. `disable_lb_split` keeps its separate diagnostic / RT-profiling meaning; the CE filter is forced unsplit by `ceFilteringActive`. A `[SPLIT] straggler` line reports each unsplit → split transition (work, `T`, parts, exprKey — pure observation, no control-flow effect). A `[SPLIT] ineffective` line reports a straggler whose busiest bucket still exceeds the fair-share — an irreducibly-serial grow-tree that bucketing cannot help (pure observation, no control-flow effect); on this corpus these are exactly the recursion / induction-step LBs (`(=[N,2])_induction_recM_`, `(in2[recM,N,3])`).

The run is **deterministic** (two runs byte-identical: `theorems.txt` + verifier check total) and **preserves the theorem set** (46 theorems, Gauss fold present): the split SET is a deterministic function of the prior iteration's deterministic totals, the applied deposit is partition-independent (the sorted firing-record merge, [D-117](../40_decisions.md#d-117) / [I-77](../30_invariants.md#i-77)), and the whole-LB expression split runs at `g_splitCount == 1` so it drops no cross-term. The burst early-exit works for split LBs too via the **doom line** ([I-191](../30_invariants.md#i-191), superseding the old [I-76](../30_invariants.md#i-76) suppression): a doom trigger publishes its part's own submatch counter plus invocation ordinal by CAS-min onto a per-LB external atomic; every part stops once its OWN counter lies strictly past the line's position; and the finalize merges ONLY the winning part's chain — timing shapes wall-clock, never output. It is **NOT** byte-identical to a never-split baseline at the proof-graph level (an unsplit deactivating LB's early-exit prefix differs from the winner part's prefix — the expected unsplit-vs-split difference), but that difference is itself deterministic.

**Firing-record capture (LB-split groundwork).** `checkLocalEncodedMemoryStatic` no longer mutates its deposit containers inline. Each firing appends a `FiringRecord` (`memory.hpp`); after the `FIXPOINT_LOOP`, `applyFiringRecords` (`memory.cpp`) sorts processor-produced records by a total content key and applies them — head deposits to `sameIterationInternalMail.statements` / `.exprOriginMap` / `.disintegrationSignals`, marker deposits to `deferredIntegrationPreps` / `admissionKeysAlgebra`. Its explicit `firingRecordsCanonical` input is false for this processor route. The GPU adapter may pass true only for a device-sorted, doom-compacted chain; that route skips only the duplicate sort and shares every deposit below it. This makes the order-sensitive deposits (cap-bounded `addOrigin`, per-head `disintegrationSignals` last-write, the staged-vector drain order) a function of the firing SET, not the request-evaluation ORDER — the determinism the LB split's expression-bucket merge depends on. The phase-2 burst performs no deactivation or discharge at all — captured firing results reach those operations through phase 3's `standardProcessing`, while mail-derived closures also run in phase 1 ([I-66](../30_invariants.md#i-66)). The fixpoint also mints **nothing** into the LB's `NameMap`: the `encode` sites in `checkLocalEncodedMemoryStatic` became the non-minting `NameMap::lookup` ([D-116](../40_decisions.md#d-116)), so the whole hashburst is read-only on the shared LB — the precondition for parallel executor threads. See [D-117](../40_decisions.md#d-117) / [I-77](../30_invariants.md#i-77).

**Deferred admission tail.** The algebra-`admissionMap` writes the hashburst marker branch (`checkLocalEncodedMemoryStatic`) discovers — `admissionMap` insert, `admissionStatusMap`, `varsInAdmissionMapKeys`, and the `revisitRejected2` revival — are not applied inline. Each is staged on the per-burst buffer `Memory::admissionKeysAlgebra` (now populated by `applyFiringRecords` in canonical sorted order) and replayed once, in that order, by `drainAdmissionKeysAlgebra` immediately before the post-burst `standardProcessing`, keeping the hashburst pass free of algebra-container mutation. The drain replays each record against current admission state; admission keys remain available until canonical re-keying or subtree cleanup. The drained template's `admissionStatusMap` value is its recursion ownership and merges like `updateAdmissionMap`'s write — the record's recursion flag OR the existing status — so an ordinary demand on the induction-hypothesis template no longer silences `updateAdmissionMapRecursion` for it ([D-413](../40_decisions.md#d-413); the former unconditional 0 was the inline write D-103 preserved). See [D-103](../40_decisions.md#d-103) / [I-68](../30_invariants.md#i-68).

**Warm-up + main iterations.** Per the project conventions, every prover run includes warm-up cycles before the main-iteration budget. Warm-up cycles "prime" the hash memory by letting rule-fire cascades reach steady state. Warm-up is the first `prove` call (`preIterations`, default 2); `analyzeExpressions` sets `warmUpPhase` around it so the quiescent-skip is disabled for warm-up bursts.

**Quiescent-burst skip ([D-194](../40_decisions.md#d-194) / [I-153](../30_invariants.md#i-153)).** `proveKernel`'s single-threaded active-build (`for (Memory* b: bodies) if (b && b->isActive) …`) excludes from the sweep every active LB whose burst would provably do nothing:

```
sweep X  ⟺  X->isActive AND
            ( warmUpPhase OR !enable_quiesce_skip OR compressor_mode
              OR X->hasWork OR mailLog.mailPeek(X) )
```

`Memory::hasWork` is a producer-side latch on the never-deloaded LB slab ([I-109](../30_invariants.md#i-109)): SLEEP clears it at `performElemPhase3` exit when the burst mutated nothing (statement count unchanged vs the phase-1 baseline, the `mutatedThisBurst` non-statement-mutation flag false, and both INTERNAL mail channels — `sameIterationInternalMail` / `nextIterationInternalMail` — empty; `mailOut` is deliberately NOT a SLEEP term — it is outgoing-only under the pull model, no burst path derives local state from it, and the commit barrier sweeps `bodies` regardless of sweep status, so pending outbound content ships whether or not the LB sweeps: the post-join drains park proven-theorem mail on the ROOT's `mailOut` after the barrier, which would otherwise permanently re-arm a genuinely idle root); WAKE sets it at every cross-LB write door (`updateGlobal`'s settled-goal erase latch — the settled/reformulated-head deposits are removed, [D-324](../40_decisions.md#d-324) — `drainDeferredAncestorAdmissions`, `broadcastTheorems` root self-inject, `activateZeroCondition`). `mailPeek` polls the never-deloaded mail log (`mailHeads.count > mailCursor`) for un-ingested ancestor mail, waking a converged LB the same iteration the pull would deliver ([I-55](../30_invariants.md#i-55) latency preserved). The predicate reads ONLY never-deloaded logical state — never `resident` / `blocksInUse` — so it is deterministic ([I-106](../30_invariants.md#i-106) / [I-108](../30_invariants.md#i-108)); residency is a consequence of the skip, not an input. A skipped LB stays `isActive` and keeps its claim/pager state untouched, so the pager's sweep window never includes it and it evicts and stays evicted at 4 GiB (the motivating win). The loop is cap-driven, so the skip changes neither the iteration count nor which iteration a fact appears in — correctness reduces to "each skipped burst is individually a no-op," validated by the `QUIESCE_SHADOW_CHECK` compile flag (sweep would-be-skipped LBs anyway and assert their burst was empty). The [I-48](../30_invariants.md#i-48) deactivation survey (`deactivateRecursively`) is a tree-wide post-join walk reading only never-deloaded state, so it still surveys skipped LBs on schedule with no reload. Telemetry: the `dt` line prints `swept=` (post-skip) and `skipped=`. Gated by `parameters.enable_quiesce_skip`.

**Runtime measurement.** `proveKernel` always brackets the complete Phase 2 scheduler window from the phase-1 barrier to immediately before phase 3. Each call prints `[PHASE2-TIMING]` — to the common diagnostics log  (with `[GPU-*]`, `[PHASE13-*]`, `[DELOAD]`, `[SPLIT]`, `[mail-memory]`, `[pool-memory]`, `[mem]`, and `[head-switch pre-emit]` telemetry, structured by `=== batch ===` and `--- hashburst ---` markers), not the run log — with the explicit backend, iteration wall time, analyzer-local cumulative wall time, iteration count, and active logical-block count. The bracket includes scheduler setup, processor or CUDA execution, projection and transfers, sealing, finalization, capacity telemetry, and split statistics; it excludes Phase 1 and expanded Phase 3. Summing `iteration_seconds` across every subprocess in one shortcut log gives the comparable Phase 2 total, while prover and overall shortcut time remain the usefulness gate. CUDA adds two nested per-pass values: reusable CUDA events produce `[GPU-PHASE2] device_seconds` across projection/task uploads and every semantic device operation through doom-prefix selection; `route_seconds` is the enclosing wall interval including host projection, scheduling, downloads, and sealing. The deeper permanent, off-by-default instrumentation layer brackets the elementary step's natural phases — now spread across the phase helpers `performElemPhase1/2/3` + `performElem2` — with RAII `RT_SCOPE_HERE` calls that attribute to a per-call tracker via the thread-local `g_currentThreadTracker`. When the compile-time gate `RT_MEASUREMENT` is `1` *and* a single call exceeds `RT_TIME_TRIGGER_SECONDS`, the tracker writes a live timing table to `.rt/<sanitized-LB-chain>.log`, refreshing online as the call proceeds. Each of `performElemPhase1`, `performElem2`, and `performElemPhase3` owns a per-call tracker. The separate sweeps therefore contribute their own scopes to the cross-burst aggregate; `REQGEN_*` measures processor request generation, while CUDA reports its device interval separately. The same gate also controls the `proveKernel` phase-and-attribution traps (`[RT phase measurement]`, `.rt/burst_phases.log`, the C15 `g_crt*` tallies). The compile-time switch controls measurement; compare its actual checked-in setting when interpreting run times. Full design + open question: [`_meta/rt_measurement.md`](../_meta/rt_measurement.md). Separate from the hashburst dump under the project conventions — different file, different gate, different code path.

**Single-pass head-switch model.** `analyzeExpressions` runs exactly **one** prove pass per batch. The contrapositive (head-switched) mirrors of every conjecture that structurally qualifies are folded into the conjecture pool **before** that single prove call, so the prover gets the same proof opportunities in one pass that the legacy multi-iteration path reached across many. Immediately after the prove pass, the same `headSwitchOne` walk runs once more on `globalTheoremList` to populate the class member `orPairsFromHeadSwitch` — `run_modes.cpp` consumes those pairs to construct OR theorems (with disjunct-set dedup and parent-removal at construction time per [D-55](../40_decisions.md#d-55); full mechanics in [`20_core_concepts/07_or_branching.md`](../20_core_concepts/07_or_branching.md#or-theorem-construction-run_modescppfullrun)). The compressor is invoked from `run_modes.cpp::fullRun` *after* `analyzeExpressions` returns (gated by `parameters.skipCompression`) — see [D-25](../40_decisions.md#d-25); it is no longer part of `analyzeExpressions` itself. There is no grid teardown / rebuild / re-prove; the multi-iteration loop and its `bigIteration` / `maxBigIterations` / `pre_emit_head_switch` machinery were deleted at [D-24](../40_decisions.md#d-24).

**Contrapositive construction.** `headSwitchOne` ([`prover.cpp+`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) is purely structural — it takes the implication string, scans the chain for a `prem[0] == '!'` premise with empty inner bound-vars (the rightmost such premise wins), swaps the head with that premise's body, and rebuilds the implication via direct string concatenation. An already-negated head loses its `!` when it becomes the new premise instead of gaining a second one (double-negation cancellation, [D-217](../40_decisions.md#d-217)) — GL has no universal `!!` normalizer (OPEN-MPL-1) and a raw `!!(...)` premise breaks every compiled-core consumer downstream. The same helper backs both the pre-emit pass and the post-prove walk; output is byte-identical between the two call sites.

---

## Theorem load path

Every theorem that enters the prover must first be rewritten by `precompileStructuralOperators` (see [I-1](../30_invariants.md#i-1)). There are three distinct load paths:

### 1. External theorems

`run_modes.cpp` and `fullRun` load the external and previously proved theorem lists before calling `analyzeExpressions`. After the warm-up proof iterations, `analyzeExpressions` precompiles their structural operators and calls `broadcastTheorems`; mail absorption installs their rule fuel. External theorems do not enter through the conjecture-registration function `addTheoremToMemory`.

### 2. Conjectures to be proved

After CE filtering, each surviving conjecture is precompiled and registered as a proof target via `addTheoremToMemory(conj, this->body, 0, false, globalDependencies)`. The `false` argument indicates "unproved — this is a goal, not a given".

### 3. Broadcast / mail

When an LB derives a theorem that becomes available to siblings, the broadcast path forwards the theorem via `mailOut.statements`. Recipients invoke `addStatement` on receipt, which internally handles re-precompilation if needed. In incubator batches the broadcast reaches only the anchor LB — the root's log has a single reader there ([I-220](../30_invariants.md#i-220)); the externals seed is stored in the anchor LB's log instead and still reaches every LB below it.

**Load-path invariant.** Any new load site must call `precompileStructuralOperators` **first**. See [I-1](../30_invariants.md#i-1) and.

---

## Core step primitives

### `addStatement` — central ingestion

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Takes:

- an expression (string),
- the target `Memory` block,
- levels `(statementLevel, equalityLevel)`,
- validity name,
- a `local` flag (whether to also register in local-only delta maps).

Does:

1. **Eager equality mirror push.** If the expression is `(=[a,b])` with `a!= b` **and `local == true`**, push `(=[b,a])` onto `newStatements` (guarded by `args[0]!= args[1]` — see [I-9](../30_invariants.md#i-9)). The `local` gate ([D-50](../40_decisions.md#d-50)) restricts the mirror push to status 0/1 (local additions); disabling this gate (the [](../) `&& false` band-aid) cost 366 incubator-Peano theorems before D-50 restored it.
2. **Shape dispatch and paired registration.** Positive equality enters `addEquality`; negated equality enters `addNegatedEquality`. These helpers register the expression and its mirror together with their levels and history. An admitted row in `intKnownStatements` represents membership; there is no separate `registered` flag. The disintegration-product path reaches this door for statuses 0/1/3, while statuses 2 and 4 leave `addExprToMemoryBlock` earlier. Negated-equality expansion also calls `addStatement` directly. Its optional `negatedExpansionReady` output reports arrival at the expansion seam without recursively entering it; the arena-backed depth-first driver finishes the child before resuming parent publication. Ordinary callers omit that continuation output.
3. **Filter gates.** Max-iteration and count-pattern early-exits.
4. **Non-equality deposit / negated-equality expansion.** For non-equality shapes (or `skip_eq_classes`), deposit into `intStatementLevelsMap`, `intKnownStatements`, `intEncodedStatements` and (when `local`) the local-delta channels. For `!(=[a,b])`, additionally call `applyEquivalenceClassToNegatedEquality` to emit one-sided sibling inequalities ([I-12](../30_invariants.md#i-12)).
5. **Equality branch.** For `(=[a,b])` (and not `skip_eq_classes`), call `updateEquivalenceClasses` (class merges and rewrites) and push the equality itself onto `newStatements`.
6. **Append to the caller-owned `newStatements` buffer** for the post-loop to consume. `mailOut.statements` is no longer populated from here — `fillMailOut` is the single writer (centralisation landed earlier on this branch).

**Output parameter — `PagedVector<IntEncodedExpr>& newStatements` ([D-33](../40_decisions.md#d-33), [I-25](../30_invariants.md#i-25)).** `addStatement` returns no value. Each appended encoded row carries its actual expression and validity identifiers. Same-scope deposits, descendant-scope deposits and ancestor-scope rewrites use this caller-owned channel. The recursive negated-equality caller supplies a scratch output buffer while the accepted rows are registered in the owning block.

The post-`addStatement` loop in `prover.cpp`, `addExprToMemoryBlock`, consumes the appended encoded rows and uses each row's decoded validity scope for:

- `intStatementLevelsMap` lookup (the non-minting `lookupStatementLevels`) — the same-scope assert at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) holds because the lookup is keyed by the deposit's actual scope.
- `updateAdmissionMapIntegration` / `updateAdmissionMapRecursion` calls.
- `toBeProved` discharge for recursion-LB head match, single-chapter theorem proof, and validity-name promotion (OR-integration / NotOrScope).

This routing closed the cross-scope discharge gap that the WIP D-33 commit had introduced — see [D-33](../40_decisions.md#d-33) for the incident and the fix.

The asserted precondition on negated-equality entry is `isNegatedEquality(expr)` — callers must gate on the same check. See `prover.hpp`, `applyEquivalenceClassToNegatedEquality`.

### `addTheoremToMemory`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Takes an unproved conjecture and the root memory block, walks its ordered premise chain, creates or reuses one child logic block per premise, seeds that premise, and installs the final goal and applicable proof auxiliaries. Its `proved` argument is asserted false. Established external and previously proved theorems enter through broadcast and mail absorption, not this conjecture-registration door.

**Antisymmetry-shaped conjectures deposit a variable-copy axiom** ([D-235](../40_decisions.md#d-235)). When the chain carries two positive premises of one operator with argument lists equal except two swapped positions, and the head is an equality over exactly that pair, the innermost premise LB additionally receives the dead-end free axiom `(=[x,x_copy])` (status 0, `main`, `variableCopy` origin, with the LB-creation paired mailOut write) — `x` the swapped argument appearing first left-to-right. Detection is the pure helper `detectAntisymmetryCopyVar`; the deposit is the fourth emission site of the `(=[Y,Y_copy])` contract shared with `checkNecessityForEquality`, `disintegrateExprHypothetically`, and `reactToHypo`. Rationale: the premise pair double-links both its variables, so every downstream rule instance — including the demand-marker completion that revives the parked totality witness — needs two template-distinct slots bound to one name, which the exact normalized-key match cannot assemble; the copy equality re-names one occurrence and the equivalence class supplies the all-distinct variants.

**Output-collision conjectures deposit the same axiom at the second colliding premise LB** ([D-265](../40_decisions.md#d-265)). When the chain carries exactly two positive premises of one operator whose non-input/output arguments are all byte-equal and whose single output slot holds the same name r (inputs differing — the MPL encoding of an equality between two compound terms, e.g. B8's a·c = b·c as two `in3` premises sharing result name 11), the SECOND colliding premise LB receives `(=[r,r_copy])` under the identical contract — the fifth emission site. Detection is the pure helper `detectOutputCollisionCopyVar` (config-classified slots via `inputIndices`/`outputIndices`). The deposit sits at the second colliding premise because mail flows only ancestor-to-descendant ([I-57](../30_invariants.md#i-57)): every genuine consumer — deeper premise LBs, the contradiction twins under the innermost premise, recursion auxiliaries — sits at or below it, while the first colliding premise is typically an anchor-adjacent chain prefix shared with foreign conjectures, and a deposit there mails the copy into subtrees that never asked for it (the full-run Peano breach behind the D-entry's placement decision). Rationale: a pool rule written with two distinct result variables for such a premise pair can never match the fused-name facts (matching renames, never collapses; partition copies are multiplication-gated off), so its firings and demand markers starve; the copy equality re-names one occurrence and the class supplies the all-distinct variants.

### `addToHashMemory`

Defined at [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp). Stores a normalised implication chain in `HashMemory::encodedMap`, along with its `LocalMemoryValue` metadata (`value`, `levels`, `originalImplication`, `justification`, `key`, `remainingArgs`, `validityName`, and — D-32 — `productOfDisintegration`). This is the actual installation of an implication rule in the hash engine.

`encodedMap` is populated from **two distinct insertion sites**, both reached from this function:

1. **Head-implication insert** at [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp) — the rule registry entry. The LMV's `.value` is the head's normalised template; the LMV carries full provenance (`originalImplication`, `justification`, `validityName`). This is the LMV that fires when an LB hash-matches the chain. **`productOfDisintegration` is stamped here**, true iff at least one premise has an arg starting with `"u_"`.
2. **Derivative LMV insert** at [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp), inside `makeNormalizedKeysForAdmission`. Regular and `ordisOnly` variants carry a marker-form missing premise; `ordis2Demand` carries it verbatim. An actual rule's anchor is never selected as missing and remains in every derivative key. Its shared normalization with the other retained premises preserves which anchor slot each variable denotes; eligibility and non-anchor binding gates are unchanged. `productOfDisintegration` defaults to `false` here — derivative LMVs never drive OR-disintegration.

When adding metadata to `LocalMemoryValue`, choose explicitly which insertion site receives it. See [`20_core_concepts/02_hash_engine.md`](../20_core_concepts/02_hash_engine.md#two-encodedmap-insertion-sites-head-lmvs-vs-marker-lmvs) for the full discussion.

After installation, `multiplyImplication` is invoked to generate partition-based equalisation copies (see below).

### `createAuxyImplication`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Generates auxiliary implications for recursion: separates digit arguments from immutable arguments (via `findDigitArgs` / `findImmutableArgs`), wraps the implication in rec-chain machinery, and registers the result. Called by `addTheoremToMemory` as part of theorem installation.

### `precompileStructuralOperators`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). In-place rewrite: replaces every raw `!(&...)` with the compiled `or<N>` name and every `!(>...)` with the compiled `existence<N>` name, recursively. Produces a string that contains no uncompiled structural operators.

Invariant [I-1](../30_invariants.md#i-1): **every theorem-load path must call this**. The assert that blows up inside disintegration when this is skipped is the symptom.

Note `precompileStructuralOperators` compiles only the **negated** structural forms (`!(&...)` → `or<N>`, `!(>...)` → `existence<N>`). A plain top-level `(>[])` implication is never its input and is therefore never compiled to `implication<N>` on a theorem-load path. The `implication<N>` name is allocated only for implications that appear as *subexpressions* of a compiled definition body, plus — on this branch — the mail-broadcast compaction wrapper below.

The phase-4 barrier also hosts the in-run OR-theorem seam `constructOrTheoremsInRun` (D-266): new proved rows fold into or theorems there and ride the same deferred-compaction drain as implication-compact broadcasts, so a run's own or theorems become consumable statements two iterations after their source proves. Full mechanism in [`20_core_concepts/07_or_branching.md`](../20_core_concepts/07_or_branching.md).

### Named compiled predicates and retirement

`visualizer.cpp` and `ExpressionAnalyzer::loadGlBinary` load every definition
before reconstructing normalized reuse entries for public logical predicates.
`splitCompilationKey` takes constituent arities from the encoded source
expressions because the shared registry omits atomic predicates. It never
inserts placeholder definitions while reading. Counter seeding applies only to generated
category-plus-number names; reuse registration is independent of that spelling.

`compilation_state.cpp` and `registerCompilationReuse` give public predicates
precedence over numbered operators. The first public registration for a
normalized body and category remains its reuse winner, while every public name
retains its own definition. `finalizeCompilationReplacements` removes generated
equivalents, rewrites dependencies and cache values with the explicit argument
permutation, and completes the rewrite before proof-state creation. The direct
OR interfaces carry the retained signature's argument order too.

Schema-1 `GL_compilation_state_<Tag>.json` sidecars contain `next_counters`
for implication, existence, and, and or, plus `replacements` entries with a
retained `target` and one-based `arguments` permutation. Original-format
registries without sidecars derive counters from their generated names.
The compiled-definition JSON structure is unchanged. Retired numbers remain
issued: their counters persist even after their definitions are removed.

`run_modes.cpp` and `fullRun` normalize inherited theorem, conjecture and OR-pair
inputs before proof work. `compilation_state.py` and `migrate_batch` then rewrite
active registry and proof-file references before the shared merge and its next
consumers. The replacement map is migration metadata, not a second compiled
alias. Shortcut capture includes the sidecars with the existing registry tree.
See [I-226](../30_invariants.md#i-226).

### `compileImplicationToCompact`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) (member of `ExpressionAnalyzer`). Compiles an implication into its compact `(implication<N>[args])` form for the single mail `statements` channel (see [`20_core_concepts/03_mail_system.md`](../20_core_concepts/03_mail_system.md)). The single-threaded load-time `broadcastTheorems` calls it inline; in-run parallel producers enqueue through `recordPendingCompaction`, and the post-join drain compiles those records. Registry preparation also calls it at single-threaded seams. Parallel calls would race global `implCounter` / `compiledExpressions` / `repetitionExclusionMap` ([D-76](../40_decisions.md#d-76), [I-28](../30_invariants.md#i-28)). It does **not** introduce a new compile path: it normalises the implication's free arguments via `stripUPrefixAST` (a leading `u_` removed from every argument token at every nesting level — unlike `removeUPrefixFromArguments`, which only sees the first `[...]`) and then calls the existing `compileCoreExpressionMapCore`. For a top-level non-negated implication that function reaches only its `(>` branch and returns the compact instance. A fully-bound proved theorem (every variable bound — [D-75](../40_decisions.md#d-75)) has zero free arguments and compiles to `(implication<N>[])`; an implication with genuinely-free arguments compiles to `(implication<N>[<freeArgs>])`. A mailed implication that fails to compile to a simple `(implication<N>[...])` asserts at its origin rather than returning an empty/sentinel string.

Compiling a *multi-premise* theorem recurses its inner `>` into its own `implication<N>`, so the outer implication-entry's head becomes an `implication<N>`; cross-batch shared-registry reuse likewise makes `implication<N>` a constituent of hand-defined `and` operators (`fXY`, `identity`). The historical `checkCompiledCoreExpressionMap` invariant forbidding "an implication as a constituent of another compiled expression" was **retired per explicit user decision** ([D-76](../40_decisions.md#d-76)): the `DeepChecker` implication branch now accepts a nested implication as a recursion-stop leaf instead of asserting. The unrelated "core not found" guard in the same walk is preserved.

> **🔴 ROOT-CAUSED + FIXED (2026-05-17).** The non-determinism was a data race, not a key flaw: `updateGlobalSingleChapter` is also called from the parallel `addExprToMemoryBlock(... coreId...)` contradiction path, so inline `compileImplicationToCompact` mutated global `implCounter` / `compiledExpressions` / `repetitionExclusionMap` unsynchronized (violates [I-28](../30_invariants.md#i-28)) — proven by torn diagnostic-trace lines and run-to-run `implication<N>` drift (1359 vs 1361) on a byte-identical `theorems.txt`. Fix (Option A, user-approved): the eight sites enqueue via `recordPendingCompaction`; a single-threaded sorted pass after `pool.join` performs the compile + deposit, flushed per core via `sendMail` — deterministic and injective. The `excludeRepetitions` zero-arg signature fix and the retired nested-implication assert are unaffected. The separate ~22 GB incubator memory-expansion concern is unchanged in volume (compaction deferred, not added) and did not reproduce in the two verification runs. See [D-76](../40_decisions.md#d-76) and [G-45](../50_gotchas.md#g-45).

### `findDigitArgs` / `findImmutableArgs`

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) and [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Extract the "flowing" versus "pinned" arguments of an implication chain. `findDigitArgs` is called by `addTheoremToMemory` (each digit becomes an induction triad), by `findImmutableArgs`, and by `updateAdmissionMap3` on a lone goal (the goal door's admission seeding).

**Typed `=` arguments** ([I-267](../30_invariants.md#i-267)). `=` is configured with `input_args [1,2]`, but it may relate sets or relations. `findDigitArgs` therefore types the whole chain (`typeLeafInto`) on its first `=` element and counts an `=` input only when it resolves to `(1)`. The supplied lemma "two intervals with the same bounds are equal" thus schedules no induction over its two sets. A lone `=` goal types to the starting point `(1)`, so the goal-door seeding is unchanged.

**Typed inputs** ([D-413](../40_decisions.md#d-413)). Every other configured input counts only when its definition slot is `(1)` (asserted present for every input slot). Every configured input of every batch is currently `(1)`-typed, so this is a guard: a set- or relation-typed input would otherwise become an extra induction triad AND an immutable argument of the real induction, fixing that object in the hypothesis. The verifier port `_find_digit_args` reads its input slots from `ConfigVisu.json` and applies no typing (see [08_verifier.md](08_verifier.md)).

### Definition-set typing and the LB records

[I-267](../30_invariants.md#i-267), [D-407](../40_decisions.md#d-407).

**The typing rule.** `ExpressionAnalyzer::typeLeafInto` types flat leaves into a stack union-find, `VarTypeTable`:
- a config slot fixes its argument's label;
- `(=[a,b])` joins `a` and `b`;
- a compact without a config entry is typed through its compiled elements;
- two labels in one group assert;
- an unfixed group resolves to `=`'s configured label (`equalityStartLabel`, `resolvedLabel`).

**The records.** `addTheoremToMemory` records, right after setting each new LB's key, the definition sets of that key's arguments (`assignLbVarDefinitionSets`). The record is built from the key chain from the root down to the LB, with `leadingExpressionOfKey` stripping the `__contradiction__` prefix and the `_induction_<recN>_` suffix. An LB reached again through a shared chain prefix is re-checked (`lbVarDefinitionSetsMatch`). Compressor nodes record their target theorem's variables before its premises and head are seeded (`assignTheoremVarDefinitionSets`).

**Verifier parity.** The Python verifier replicates these algorithms — see `_find_digit_args` and `_find_immutable_args` in `verifier.py`. A divergence between C++ and Python is a sound-ness red flag; [I-16](../30_invariants.md#i-16) applies.

### `reconstructImplication` / `reconstructImplicationFullBind`

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Rebuild an implication from a chain + head. There is now **one** binder rule (see [I-4](../30_invariants.md#i-4)): bind every variable whose name does not start with `u_`, at the left-most premise that mentions it. `reconstructImplicationFullBind` is the single implementation; `reconstructImplication` is a thin forwarder to it, kept so the historical theorem-level call sites and the I-4/I-5 citation symbols stay grep-stable. The earlier theorem-vs-rest split (the retired sparse "bind only ≥2× occurrences" rule, old [I-5](../30_invariants.md#i-5)) is gone. A theorem carries no `u_` args, so every variable in a theorem — including every anchor slot — is bound.

**`reformulateTheorem` interaction.** Its peeling-layer trigger used to read the *reconstructed* last-link `>[...]` cardinality (`boundVars.size == 1`). Under the unified binder that cardinality no longer encodes the condition, so the trigger is re-expressed to compute the same predicate directly from the original chain — the target Definition's set-argument is the single non-`u_` arg that occurs ≥2× across (all premises + head) and appears in no other premise. The same reformulations fire; only the emitted theorem's outer `>[...]` widen. The inner negated-existence still binds exactly the peeled set-argument. See [G-44](../50_gotchas.md#g-44) and [D-75](../40_decisions.md#d-75).

### `addExprToMemoryBlock` — `status` parameter reference

`addExprToMemoryBlock` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) is the central deposit primitive called from dozens of sites. Its 4th argument `int status` is a mode selector that controls which deposit path runs. The values are not documented inline at the function site — callers learn them by copy-paste from neighbouring call sites, which is error-prone. Reference table:

**The canonical door (every status except 2).** After Site F (raw text) and Site H, `addExprToMemoryBlock` rewrites every deposit that is neither a goal, nor a positive / negated equality, nor an anchor predicate to its class-canonical form under the equivalence classes at the deposit's OWN scope (`canonicalFormAtScope`, chooser `chooseCanonicalId`; [I-217](../30_invariants.md#i-217)). A changed deposit whose canonical form is already known at the scope or a strict ancestor ends there (both ids release the or-cohort signal); otherwise the door proceeds with the canonical text LOCAL iff an applied equality is a local statement here (the rewrite is then this LB's class product; under purely mailed equalities the arrival's locality stays) — the raw text keeps its producer line (shipped to `mailOut` when the row is local), the canonical form gets one `equality1` bridge when it has none (I-34), the level run gains the applied pairs' class levels — and reports the encoded canonical row through the optional `registeredOut` (the absorb drain stages a status-3 arrival from it only when the row stayed non-local). Only canonical forms are admitted; `applyEquiClasses` still multiplies them.

| `status` | Path | `forceDeep` arg | Disintegration? | Origin tracked? | Typical caller |
|---|---|---|---|---|---|
| `0` | Local derivation, force-deep disintegrate | `true` | yes | yes | Pass B deferred emissions, OR branch seeds, CE filter admission |
| `1` | Local derivation, normal disintegrate | `false` | yes | yes | Canonical "produced and absorbed by LB" — `addStatement` main branch (`:1975`), fixpoint deposits (`:4648`), `addToHashMemory` head writes (`:4956`), **integration-revival absorb** (from `sameIterationInternalMail`); the internal-mail absorb offers every row RAW — the door canonicalizes |
| `2` | Goal / `toBeProved` insert | n/a | skipped — `prepareIntegration` runs instead | yes (goal-origin) | New head registered for proving; OR branch head emission |
| `3` | **External-mail absorb (ASIC 0.1 reshuffle).** Enters `disintegrateExpr2` to recover the implication / fact from the compact mail form. The earlier branch experiment of running with `allowExistenceDisintegration=false` was reverted — the parameter was removed and `disintegrateExpr2` / `disintegrateExprCore2` are now byte-identical to main HEAD's control flow. Recovered implications are status-distributed: `overallHashMemory` (full visibility) **plus** the per-burst `workingMemory` (the same-burst Batch-1 source) — and explicitly **NOT** `localHashMemory`/`localHashMemoryDelta` (those are local-impl-only; an external rule there would be mis-treated as a local delta by Batch 5). Recovered facts go through the kernel and are additionally staged into `externalStatements`/`intExternalStatements` for the same burst's mail-pair batches. Runs in the **pre-fixpoint mail-absorption block** (before request generation), unified across CE and normal mode. The post-fixpoint placement introduced by commit was reverted on 2026-05-20 ([D-79](../40_decisions.md#d-79)) because it pushed `__contradiction__(=[a,b])` LBs past `MAX_NAME_IDS` during Peano-incube combinatorial substitution. | `false` | yes (origin from `mailIn.exprOriginMap`) | The pre-fixpoint `mailIn.statements` drain in `standardProcessing`, offered RAW; the `intExternalStatements` row is staged from the door's returned canonical row (I-217). |
| `4` | Fast-path fact load | n/a | skipped (no gate, direct push) | no (levels only) | Simple-facts bootstrap `prover.cpp` |
| `5` | **Flag-5 relay arrival ([D-284](../40_decisions.md#d-284)).** A routing statement whose mail side column carries flag 5 ("external, may be disintegrated"): the sender's local disintegration parked this compact's products for lack of admission demand, and this receiver may hold the demand. Enters `disintegrateExpr2` WITH witness minting at the carried generation stamp (the side-column iteration, I-171 across the seam). The CARRIER registers non-local (never re-forwarded) and joins `intExternalStatements` (staged under the mail-level status 3 before the per-statement upgrade); admitted PRODUCTS register LOCAL (mailable, local-delta-batch-visible) and ordinary integration-bearing products stay local; undemanded products park under the normal Pass-B discipline (revival covers late demand). A status-5 disintegration never re-stages a relay — no cascade. Reuses the integer of the old rejected pre-reshuffle status-5 mode (unrelated). | `false` | yes (witness minting on) | The routing `mailIn` drain when the statement's flag side column reads 5. |

**Choosing the right status.** Ask: has this expression ALREADY been disintegrated by whoever produced it?

- If the producer is another LB's `mailOut → smashMail → mailIn` routing: use `3` — post-ASIC-0.1 the receiver re-derives compact implications and negated compact existences through `disintegrateExpr2`, while positive compact existences remain opaque (no fresh `it_`/`int_`); recovered rule carriers install through the persistent 3-way path.
- If the producer is *this* LB's own body (eq-class rewrite, OR-branch seed, revival absorb): no, or incomplete. Use `1` (normal) or `0` (force deep).
- If inserting a theorem to be proved (vs asserted), use `2` — `prepareIntegration` runs to set up integration rules.
- If reloading simple facts at grid setup, use `4` — no semantic processing needed.

**Silent-failure mode (historical).** Before the ASIC 0.1 reshuffle, `status=3` skipped `disintegrateExpr2` entirely (a `status!= 3` gate). Using it where `1` was needed (e.g. for revival constituents) deposited the expression into `intEncodedStatements` and `intKnownStatements` but ran no disintegration — Pass B admission never fired, `int_` vars never minted, no integration progressed; the statement "arrived" but was inert. That is why the integration-revival channel uses `status=1` ([D-19](../40_decisions.md#d-19)). Post-reshuffle `status=3` enters `disintegrateExpr2` for compact implications and negated compact existences, but positive compact existences remain opaque. Thus the old all-expression inert mode is gone, while positive existence-witness minting still distinguishes status `3` from status `1`.

### Admission / rejection-map key shapes

There are FIVE maps in `HashMemory` that key on variations of a marker-form expression. They are NOT interchangeable — each uses a different prefix convention, and a lookup using the wrong form silently misses. Reference table:

| Map | Field in `HashMemory` | Key shape | Produced at |
|---|---|---|---|
| `admissionMap` (algebra) | `.admissionMap` | Producer templates can carry `u_` on fixed arguments; canonical concrete storage also uses bare fixed arguments. In particular, unary integration-demand transfer inserts its canonical marked premise, such as `(and27[marker,12,i,10])`. Entries retain the installing or matched source scope (a branch-scope rule demands only in its branch; [D-369](../40_decisions.md#d-369)). | `prover.hpp`, `updateAdmissionMap`; `unary_composition.cpp`, `transferUnaryIntegrationDemands` |
| `admissionMapIntegration` | `.admissionMapIntegration` | `u_`-prefixed on all non-marker args — same as above, integration-side | `prover.hpp` — built from `le.signature` in `prepareIntegrationCore2` Case C |
| `admissionSetIntegration` | `.admissionSetIntegration` | **bare concrete with `repl_` preserved** — e.g. `(in3[repl_lev_2_0, marker, repl_lev_2_3, 4])` | `prover.cpp` — `removeUPrefixFromArguments(mappedElement)` (strips `u_`, keeps `repl_`) |
| `rejectedMap` (algebra) | `.rejectedMap` | **bare concrete** — matches Pass B's `makeMarkedExpr` output | `prover.hpp` (via `updateRejectedMap`) — key built by Pass B at `prover.cpp` |
| `rejectedMapIntegration` | `.rejectedMapIntegration` | **bare concrete** — same shape as `rejectedMap` | `prover.hpp` (via `updateRejectedMapIntegration`) — key built by Pass B at `prover.cpp` |

**Bridging between shapes.** A few gateways exist:

- `isAdmittedIntegration` at `prover.hpp` — converts Pass B's bare marker form to `u_`-prefixed before looking up `admissionMapIntegration`. Mandatory transform for the lookup to hit.
- `removeUPrefixFromArguments` — strips `u_` from args in-place. Used at `prover.cpp` to go from `u_`-prefixed template to `admissionSetIntegration`'s `repl_`-preserving form, and used in the integration-revival path (`applyEquivalenceClassToRejectedMapIntegration` at `prover.hpp`) to strip `u_` off the admissionMap key before probing `rejectedMapIntegration`.

**Silent-failure modes.**

- Revisiting `rejectedMapIntegration` with the `u_`-prefixed key (without stripping): every lookup misses. No rejections get revived. The revival mechanism fires but accomplishes nothing. See [G-32](../50_gotchas.md#g-32-rejectedmapintegration-key-shape-bare-concrete-not-u_-prefixed) for the concrete debug trail.
- Probing `admissionMapIntegration` with a bare key (without `u_`-prefixing): same — every probe misses.
- Probing `admissionSetIntegration` with a `u_`-prefixed key: misses (the set stores bare `repl_`-form).

Rule of thumb: distinguish producer templates from canonical concrete keys. Integration admission keeps its prepared `u_` template convention; algebra admission also contains canonical concrete keys, including those produced by `transferUnaryIntegrationDemands`. Admission sets and rejection maps use concrete instances. Use the matching gateway for each map instead of assuming one prefix convention for every admission entry.

---

## Startup integration demand

`prover.cpp`, `analyzeExpressions` / local `buildGrid`, does not call `prefillIntegrationMapsRecursive`. Known non-atomic operator premises are still registered and disintegrated normally, but grid construction does not replace their output argument by `marker` and request alternative-output constructions. Integration preparation remains reachable from explicit goals, installed rule premises (including induction hypotheses), CNFE and ordinary demand replay. The unused prefill helper is retained during the bounded experiment; its logic is not invoked by the main pipeline. No inference rule, scope boundary, supporting-level or history contract is changed at the remaining preparation doors.



## Disintegration

Disintegration is the prover's "expand compound expressions into their parts" operation. Two paths:

### Pass A — structural disintegration

The standard path: when an `and` or `existence` node is received, split it into its element expressions via `disintegrateExprCore2` and recurse. Straightforward. A **negated existence** `!(existence<N>[args])` (a universal in De-Morgan clothing) splits into its two rules `left → !right` / `right → !left` — since [D-310](../40_decisions.md#d-310) not as on-the-spot rule text but as the existence entity's compiled `implications` compacts (`compileExistenceImplications` at the `!(>` mint), resolved through the registry (the prepared instruction is empty for a negated compound), instantiated positionally — except the typing direction `in[k,S] → !body(k)`, which the door skips ([D-382](../40_decisions.md#d-382)): only `body(k) → !in[k,S]` is instantiated — and handed to `prepareIntegrationCore` + the implication branch as product statements: the compact carries the `expansion` origin citing the negated existence, its expansion into the rule is a second `expansion` row, and the rule's bound variable is the registry placeholder rather than a fresh `startInt` integer ([I-209](../30_invariants.md#i-209)). Negated category-`and` compounds bypass the OR consumer. A public predicate name supplies one flat exclusion implication per conjunct, with expansion/disintegration history at its source scope; generated `and<N>` names remain ordinary negative facts. Neither path registers an OR source or creates case scopes (I-179). Positive category-`or` compounds register an on-demand exclusion source after finalized known/levels admission. Their case opening and parking keep the ordinary gates; their compiled exclusion family is no longer installed in hash memory. `standardProcessing` drains canonical OR matches after `materializeWatchHits`, emits residuals or literals through internal mail, and activates residual continuation only after the deposit door admits a known form.

### Admission and permanent publication

A negative implication compact with exactly one local bound variable also supplies a counterexample construction. `prover.hpp` / `prepareIntegrationCore` enables this only for `disintegrateExpr2`: an expansion-local `counterexample` instruction retains the negative compact as its source and contains every premise followed by the negated head. Free arguments remain fixed; the single bound placeholder is fresh within that instruction. Recursive preparation propagates the same mode without adding shared compiled definitions. Ordinary integration keeps its negative-compound polarity guard. Zero- and multiple-witness negative implications remain passive.

`prover.cpp` / `disintegrateExprCoreView` creates exactly one integration-kind slot for this instruction, never a parallel algebraic copy. On the ordinary route, `disintegrateExpr2` admits that slot without demand only when its original assigned depth is within `maxIntegrationDepth`, its source has no algebra-witness input, and its enclosing witness is admitted; it does not pull an unadmitted or depth-forbidden ordinary parent into permanent state. Forced routes retain their separate witness class and depth exemption. Source/scope receipts suppress repeated creation, and ordinary disintegration-ban/compressor gates remain. Nested ordinary existences still need their own admission. The `int_` kind keeps downstream premise-completion rules pure; an `it_` divisor would block the cofactor admission marker. `expandSignature` reconstructs the original negated universal with the admitted witness binder and existing expansion/disintegration provenance. See [D-392](../40_decisions.md#d-392).

The disintegration preparation mode also unfolds a negative category-`or` compact as category `and`, with each registered disjunct negated and double negation cancelled. It reuses conjunction recursion and history, without OR demand, case scopes or a new witness. Nested witness-bearing children remain subject to their own construction policy; an unresolved enclosing witness still prevents publication. Positive OR handling and integration preparation are unchanged. See [I-255](../30_invariants.md#i-255).

`prover.cpp` / `disintegrateExpr2` first calls `disintegrateExprCore2` with expansion-local temporary witnesses and history/OR effects disabled. Each slot records its construction kind, algebraic generation, shared assigned integration depth, enclosing slot, structural position and source existence. `Memory::startInt` and persistent witness/history stores are untouched by this discovery pass. Temporary identity comes from slot metadata, never permanent-name parsing. `memory.hpp` / `DisintInstructionView` keeps the prepared instruction shared through witness recursion and applies each enclosing substitution when a signature or element is read. Sibling views are independent and expansion-local. The negated-existence instruction-extension path and concrete OR-leaf consumer materialize only their current effective view, preserving the original field order, marked goal and preparation counters. See I-250.

Admission keeps the existing routes and gates:

- Algebra: `isAdmittedIncludingAncestors` probes the exact scope, then strict ancestors in their existing order; `ordisOnly` values do not qualify. The standalone route remains limited to an operator's single input slot (I-6).
- Integration: `isAdmittedIntegration`, then `admissionSetIntegration`. The instruction-map route excludes configured predicates with two inputs and no output, except positive equality, through the shared eligibility predicate at Case C key creation, admission lookup and known-statement replay (I-227). The existing exclusions for typing `in` in either polarity and negative equality remain; predicates with other input counts or outputs retain the other gates. The separate admission-set route is unchanged.
- Generation and secondary-variable budgets use explicit slot metadata. Secondary variables are counted by occurrence; permanent recursion products retain their exemption.
- Force-deep selects forced constructions on integration-only enclosing paths and mints `force_lev_` names. The compressor and disintegration-ban gates remain.
- Ordinary integration constructions use one depth computed from the original outer expression: zero without integration names, otherwise one plus their maximum depth. `maxIntegrationDepth` gates witness minting and parking, never the structural traversal. Witness-independent statements and implications remain available at the limit. Blocked witness chains cannot be coadmitted indirectly.
- Ordinary integration also rejects a source existence with an algebra-witness input. `disintegrateExpr2` checks its complete argument tokens after template-prefix stripping using `classifyWitnessName`; an expansion-local argument uses `TemporaryWitnessStore` kind metadata. The new bound witness is not an input to its own compact source. Rejection uses the depth gate's disabled path: no minting, no integration parking, and no descendant coadmission. The check is repeated on replay's current source independently of the saved depth. Forced extraction, algebra admission and witness-independent structural extraction remain available. See [I-268](../30_invariants.md#i-268).

A successful probe coadmits the complete enclosing chain, assigns permanent names in deterministic order and immediately applies the ordinary successful admission updates with substituted names. Shared-expression cascade closure completes before publication. A second traversal of the prepared instruction visits selected paths with permanent names and writes the existing expansion/disintegration history, rules and OR effects. No OR inside an unresolved existence can be published or parked. Temporary slots and expanded instructions end with the call.

### Parked existences and normal replay

Atomic leaves retain their input sign in `prover.hpp` / `prepareIntegrationCore` before instruction-signature deduplication and admission-key construction. Positive and negative equalities with the same arguments remain distinct identities.

`prover.hpp` / `isIntegrationInstructionKeyEligible` excludes predicates with exactly two configured inputs and no outputs from `admissionMapIntegration` marker keys, except positive equality. Typing (`in`) in both polarities and negative equality remain excluded. Case C in `prepareIntegrationCore2`, pure and committing `isAdmittedIntegration`, and known-statement `updateAdmissionMapIntegration` share this predicate. Unary predicates such as `prime`, other input counts, and predicates with outputs retain their existing gates. Exclusion affects keys only: order bounds and other excluded constituents remain required premises of construction rules. See [I-227](../30_invariants.md#i-227).

Case C of `prepareIntegrationCore2` permits repeated fixed arguments when registering a new integration admission key. The earlier `allowedRepeatedVars` instruction check remains, as do the same-core exclusion, the configured-predicate eligibility check and the requirement for exactly one non-`u_` argument occurrence. Thus an eligible singleton interval can register a key with equal lower and upper endpoints. Repetition introduced by canonical rewriting is also retained. A constituent's presence in another key's instruction payload does not itself register an admission key for that constituent.

Both replay callers keep a sorted snapshot solely to determine the original iteration set. Before each consumed-value write they reload the current map, locate the original instruction by content, assert its presence, and add the value to that current map. Recursive preparation may add instructions under the same key; subsequent writes preserve those additions without extending the current iteration set. The update remains arena-backed and leaves pure probes unchanged.

Both `rejectedMap` and `rejectedMapIntegration` keep their concrete marked keys and source scope. Their values contain the **source existence**, algebraic generation, original assigned integration depth, construction/OR/carrier flags, source levels and optional deferred equality triples (`before`, `after`, `equality`). They contain no expanded products, sibling lists, instruction trees or witness-name tables. The construction flags distinguish the algebra and integration alternatives of the same source without retaining structural-path progress.

An unresolved inner source is parked with concrete outer arguments after its enclosing witness is admitted. While an enclosing witness is unresolved, its enclosing source is retained instead. Keys containing another unresolved temporary are never stored.

`revisitRejected2` and `revisitRejectedIntegration2` find exact and ancestor parks. `prover.hpp` / `queueParkedExistence` revalidates admission and queues the source on `sameIterationInternalMail`. `standardProcessing` absorbs it through `addExprToMemoryBlock` and regular disintegration. The internal signal preserves the original assigned integration depth; replay does not recompute eligibility from a nested source containing newly introduced witnesses. Real pending assignments min-merge, and completed constructions are not reminted. An explicit queued-replay construction mask passes the known-source and fully-disintegrated-equivalent refusal; ordinary arrivals retain those gates and replay still obeys closed-scope checks.

`Memory::existenceReplays` records source/scope completion and pending construction bits, without witness names. Successful descendant replay leaves the ancestor source available for other scopes. Repeated demand at the same scope cannot mint the completed construction again; unresolved nested sources remain independently eligible. Park-after-demand checks include already-present descendant demands.

### History, coverage and lifecycle

`Memory::sampleParkedSourceStorage` measures the two source maps, integration key argument indexes and source/scope receipts while the block is resident at the phase-three tail. Its always-resident eight-byte peak is telemetry only: it is neither cloned nor deloaded. The end-of-prover report sums the observed per-block peaks; this is not a simultaneous grid peak, and excludes shared text interners and arena padding. The witness count is the sum of `startInt` growth after grid construction, so it counts permanent allocations during the measured prover interval and excludes copied initial counters. Mail-history volume remains separately reported by the existing mail telemetry.

`ExpressionAnalyzer::wakeParkedExistencesAtExistingDemands` uses `LbMemory::existenceLookups` to find demand keys by the same prefix-stripped marked text. Newly appended keys extend the derived lookup; key removal/reset rebuilds it before querying. Only matching packed keys enter the original decoded-expression/scope order and ordinary admission checks. The index never mints into a logical text interner.

`retireCompletedExistences` selects receipt keys at the requested scope, then follows source/scope postings to parked keys. Candidate keys are ordered by their original map positions before applying the unchanged completion predicate and removal operations. Source insertion and equality replacement maintain the postings; different provenance rows share one source/key edge. These pool-backed lookups are derived: clones and canonical reloads reconstruct them, raw eviction preserves them, and scope wipe/discharge/canonical release clear them while resident. Their bytes participate in `LbMemory::liveBytes` but remain outside canonical deload facets and the source-container peak counter. See I-251.

The source equality hook uses `forEachSubstitutionToken` to check both marked keys and stored source arguments before sorting affected keys or constructing replacements. Its token set is call-local scratch; deferred equality steps and successful replay ordering are unchanged. Each source applies only substitutions for its own tokens, ordered by their original class-member positions. Its old record run is cleared before ordered replacement and immediate wakeup. Empty old keys carry no replay source and are compacted together before the hook returns; assertions check that none survives. The block remains claimed throughout, so this temporary empty-run state never enters deload. `isAdmitted` reads live scalar record fields for a pure probe; full record snapshots and decoded update strings belong to the committing path. Explicit budget, key-size and status assertions are retained.

History is produced during selected publication. Descendant replay first records `zoom in` for the source compound, then disintegrates it using fresh witnesses in the target scope. Source equality transformations are retained as ordered triples until materialization and are shipped with the source chain. No admission or replay decision reads `exprOriginMap`.

`fullyDisintegrated` remains a coverage claim: an unresolved eligible source prevents full coverage. Highest uncovered sources still feed flag-5 relay; the existing local-status/main-scope relay eligibility is unchanged. An inherited carrier retains its non-locality and no-cascade role through later replay. Facts clones copy source context and completed constructions; pending delivery resets with their existing empty-mail/empty-demand contract. Source records live in the existing rejected-map deload bands. Source/scope receipts are ordinary cold block storage and participate in clone, scope cleanup and both deload formats.

The public configuration and proof tags remain unchanged. See I-252 and I-224.

## Integration

Integration is the mirror of disintegration: reassemble a compound expression from its constituents so it can be asserted as a conclusion.

Three preparatory tags appear in the processed proof graph:

- `reformulation for integration and` — for `and`-category conclusions.
- `reformulation for integration >[bound]` — for `existence`-category conclusions whose outermost `>[...]` bound-variable list is non-empty; the bound variable is carried through.
- `reformulation for integration >[]` — for `existence`-category conclusions whose outermost `>[...]` list was stripped because the witness slot was already occupied by an external value.

Helpers involved:

- `expandSignatureForIntegration` — uses `reconstructImplicationFullBind` (see [I-4](../30_invariants.md#i-4)).
- `buildIntegrationInstruction` — ditto.
- `updateAdmissionMap` — consumer-side output-slot registration, feeding the disintegration admission map; fed the rule's PREMISES only, never its head ([I-244](../30_invariants.md#i-244)).

Each integration step emits a `validity name` tag row — declaring the scope into which the integrated expression is bound — and then an `expansion for integration` tag row showing the re-expansion into compiled form.

**Typed `=` goals** ([I-267](../30_invariants.md#i-267)). The goal door (`addExprToMemoryBlock`'s status-2 branch) calls `prepareIntegration` with `typedEqualityGoal`. A positive `=` goal is typed on the spot (`resolveEqualityGoalLabel`):
- the LB records up the parent chain;
- else the element that opened the goal's scope — the implication of a Case-A subproof (the goal is its renamed head) or the OR of an OR-case branch (the goal is a disjunct over the OR's actuals) — typed through the element's interface slots, position by position, and iterating up the scopes (`scopeVarLabel`) when a formal's actual carries the type;
- else assert.

A `(1)` goal keeps the atomic path. A set or relation goal enters `prepareIntegrationCore` with the extensionality entity as its root: `and` over the two subset implications `E[A,B]` and `E[B,A]`, where `E` is `(>[x](in[x,A])(in[x,B]))` or the `in2` / `in3` form. The unchanged `and` machinery opens one subproof per subset (`<goal>_subproof_(E[A,B])`) and installs the rule "`E[A,B]` and `E[B,A]` → `(=[A,B])`". Its expansion row is `(&(E[A,B])(E[B,A]))_integration_goal`, and its reformulation tag is `reformulation for integration and`. The entities are compiled at registration (`ensureTypedEqualityEntityForTheorem`, before the head is seeded) and reuse an existing subset compact when the registry has one.

**Finished-task wipe** ([I-284](../30_invariants.md#i-284), [D-443](../40_decisions.md#d-443)). Each `prepareIntegration` expression at its scope is a task in the block's task table (`LbMemory::integrationTasks`). A task prepared while no task is active is a root; everything prepared, installed, opened or parked while a task is active (`IntegrationTaskScope`) belongs to a node of that root — the task itself, a sub-task, or the gate node of one subproof opening, case split or rule installation. Marker firings pass the root on through the fired rule copy (`FiringRecord::ruleCopyOwner`). A door that finds a derivative already made for another root, or for an independent trigger, marks its node shared. A root without a marker finishes when its expression is known at its scope or an ancestor, up to the scope's classes; a root with one marker at a declared output slot (the value of a fold, say) finishes when any value is known there. Only integration products count ([D-444](../40_decisions.md#d-444)): a row fired in the block by a rule copy an integration task owns, or mailed with the sender's integrated bit, is marked `StatementFlags::integrated` at the absorb; a value another rule built leaves the root open. `sweepFinishedIntegrationRoots` detects finished roots from staged arrivals and class changes and, just before the settled-subproof retirement, removes the admission entries, parks, rules, algebra and admission-set keys, triggers and scopes of the root's unshared nodes. Facts, proof history and prepared keys stay. In the slowest F3 block this closes the losing accumulator candidate's subproof when its fold task is solved at main burst 8.

---

## `multiplyImplication` — Bell-partition equalisation

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Invoked by `addToHashMemory` on every newly-installed implication.

**What it does.** Collects every variable that appears at a `(1)`-typed argument position anywhere in the rule (both bound and free `u_*` anchor parameters), then enumerates Bell partitions of that set. Each partition defines one "equalisation copy" where the variables in the same class are forced equal simultaneously. Each copy is installed as its own rule — broadening the implication's applicability.

**Example.** An implication `(>[v1,v2,v3](P(v1,v2,v3)))` with three bound `(1)`-typed variables has the Bell partitions:

- `{{v1},{v2},{v3}}` — no equalisations.
- `{{v1,v2},{v3}}`, `{{v1,v3},{v2}}`, `{{v2,v3},{v1}}` — one pair equalised.
- `{{v1,v2,v3}}` — all three equal.

For each partition except the original, `multiplyImplication` produces the equalised copy, passes it through `precompileStructuralOperators` (if needed), and registers it as a separate rule.

**Why.** Universally bound variables may be instantiated with the same value. The hash engine's exact normalized-key match preserves argument repetition, so a rule written with distinct variables needs an equalised copy to match facts whose arguments coincide. `multiplyImplication` installs those specialisations of the source theorem without requiring the conjecturer to enumerate them; it does not assume new equalities between distinct free parameters.

**Soundness gate (no double-`u_` equivalence classes).** A partition is *skipped* whenever any equivalence class would contain two or more distinct free `u_*` anchor parameters. Free anchors are pre-bound to specific elements of the underlying definition sets — equating two distinct ones would silently rewrite a free slot of the rule body, producing a logically stronger rule than the source. Bound-and-free merging (one `u_*` together with one or more bound vars) is still permitted: that is the documented mechanism for specialising a rule's bound variables to a known constant.

Emitted chapters carry the `multiplied from` tag (see [Proof tags](../20_core_concepts/08_proof_tags.md)).

---

## Induction

Induction is scheduled on a bound variable `n` identified structurally (currently: `(1)`-typed bound variables that appear in certain canonical positions). The prover creates sub-goal LBs for:

1. **Base case** — substitute `i0 = 0` for `n`; prove the head under the substituted premises.
2. **Step case** — assume the head for `n`; prove the head for `s(n)`.

Promotion of an induction theorem to `globalTheoremList` uses `method = induction` with the induction variable recorded in the reference column. The registered string is the FullBind-canonical reconstruction (`reconstructImplicationFullBind(ky, value)`) of the proven implication, **not** the as-scheduled `expr` string carried in `dependencyTable.originalAuxyMap[…].expr`. The conjecturer / scheduler may emit a theorem whose outer `>[…]` binds only the anchor-slot names that appear in the body (e.g. `(>[N,i0,s](AnchorPeano[N,i0,s,+,*,i1])(…))` for `existence3`); the registry must store the I-4 form (`(>[N,i0,s,+,*,i1](…))`) because the chapter's `implication`-tag citation is also FullBind, reconstructed from the GL binary by the deferred-compaction drain. See [D-80](../40_decisions.md#d-80).

### Induction typing obligation

See [I-18](../30_invariants.md#i-18) and [`docs/agentic_swdd/induction_typing_plan.md`](../induction_typing_plan.md). The historical prover scheduled induction on any structurally-typed bound variable without first verifying `(in[n, N])`. For bound variables appearing only in negations, bare equalities, or existence heads, this is unsound — the theorem ranges over all entities, not just over `N`.

The current guard in `prover.cpp` and `ExpressionAnalyzer::updateGlobal` requires `(in[indVar,N])` to be known at the conjecture chain block's exact `main` scope before induction promotion. `lookupStatementFlags` probes `intKnownStatements` through that block's `nameMap`; an absent typing fact prevents insertion into `globalTheoremList`.

The typing evidence is exported from the parent conjecture block by the `inductionTypingStack` lambda in `visualizer.cpp`, alongside the zero and induction-condition chapters. The verifier requires the corresponding `<N>_induction_typing.txt` chapter and checks its typing goal. This is an existing proof obligation; the plan file records its original staged design.

---

## Equivalence classes and `addStatement`

`Memory::equivalenceClassesMap` is per-validity. Each entry is an `EquivalenceClass` grouping variables known-equal within that validity. When `addStatement` receives `(=[a,b])`, it updates the relevant class; when it receives `!(=[a,b])`, it emits sibling inequalities one-sidedly via `applyEquivalenceClassToNegatedEquality` (see [I-12](../30_invariants.md#i-12)).

The deliberate one-sidedness — not the symmetric cross-product — is the combinatorial containment knob. Two-sided expansion would blow up without clear semantic gain.

### Pair-invariant asserts

`addStatement` carries several pair-invariant asserts (the equality-mirror guard, the negation-shape guard, the validity-namespace guard). These are load-bearing — weakening one to "make a failing test pass" is exactly the [I-19](../30_invariants.md#i-19) anti-pattern. Memory file records the rationale.

---

## OR and contradiction

### OR disintegration / convergence

`classifyOrScope` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) classifies a validity scope as `NotOrScope` / `Integration` / `Disintegration` based on the NameMap stack payload. Used by:

- the elementary-step phases (driven by `proveKernel`) — to decide whether to run OR-aware bookkeeping.
- `ordisMerge` (inline member in [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp), called from `addExprToMemoryBlock`'s post-`addStatement` loop) — `_ordis_` convergence detection, per-branch cleanup of the converged expression, parent-scope promotion via `sameIterationInternalMail`. See [D-34](../40_decisions.md#d-34).
- `cleanUpOrIntegrationBranches` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) — wipes only sibling `_orint_` branches with the same direct parent validity and outer OR signature when one branch proves the goal; equal signatures under other parents remain live.

OR disintegration first recursively flattens every contiguous nested OR in the prepared instruction into ordered non-OR leaves, then creates one validity scope per leaf inside the current LB and asserts that leaf as the branch seed. It does not create child LBs, intermediate OR branches, or seed the other leaves' negations. The original outer OR signature names every branch and remains the cohort/provenance identity; the flat leaf count is the convergence threshold ([D-218](../40_decisions.md#d-218)). OR convergence fires when every branch scope in the exact parent-scoped cohort has independently reached the same conclusion — the conclusion is then promoted to that cohort's parent scope. `orDisjunctCount` is keyed by the interned `(parentValidity, originalOrSignature)` cohort; `orBookkeeping` is keyed by `(conclusion, cohort)`, so equal signatures under different parents cannot combine ([D-219](../40_decisions.md#d-219)).

OR integration applies the same contiguous flattening before branch preparation. `prepareIntegrationCore2` skips nested OR entities already owned by an eligible outer root, renames the root's complete ordered leaf list once, and creates one `_orint_` scope per atomic leaf. Each scope receives that leaf as its goal and every other leaf's negation as assumptions; no branch body or assumption names an intermediate compiled OR ([D-220](../40_decisions.md#d-220)).

**MAIN-goal closure is level-free; registration is level-gated** ([D-278](../40_decisions.md#d-278), [I-192](../30_invariants.md#i-192)). When a delta statement matches a MAIN `intToBeProved` goal and `isProved` holds, `dischargeToBeProved`'s non-recursion branch closes the goal unconditionally — goal-row erase, goal-scope wipe mint, twin-retirement staging — and seals an `UpdateGlobalSingleChapterRec` carrying the `allLevelsInvolved` verdict (`lvN == level+1`, or `lvN == level` without level 0) computed from the head's level row. The post-join `drainUpdateGlobalSingleChapter` index-sorts the records on `(theorem bytes, level verdict TRUE-first, producer chain)` — every key a pure function of proof state; the dispatch `coreId` is banned from the order because one theorem can seal from two producers in one iteration (the chain LB and its primed `__contradiction__` twin), the sink's string dedup makes the first-drained record's method the registration, and a race-ordered drain flipped that method run to run ([D-291](../40_decisions.md#d-291)). It then forwards each verdict into `updateGlobalSingleChapter(…, registerGlobally)`: a false verdict runs the lifecycle walks (`deactivateUnnecessary`, both polarities) and returns before every circulation surface — the closure registers only into the proved-not-broadcast tier ([D-290](../40_decisions.md#d-290)): a recorded row plus chapter, never a circulating fact, and a later first-class registration of the same theorem upgrades the tier row in place. The over-premised conjecture's LB chain still dies either way. The recursion branch always had this shape (closure unconditional; `recLevelsInvolved` — including the digit-arg membership exemption — rides `updateGlobalTuples` to `updateGlobal`'s promotion gate).

`dischargeToBeProved` closes scoped goals through lexical visibility, not exact-scope duplication. Its statement-delta walk still handles exact arrivals first. It then snapshots unresolved goals in decoded lexical order and walks each non-main goal's `NameMap` ancestry root-to-self. The first known matching statement supplies the canonical source validity and level row. One local closure body emits from the exact goal scope in both cases: `_orint_` emits the wrapping OR at the goal's parent and cleans only that goal's cohort; an ordinary implication scope emits its bare signature at `main`. Provenance cites the actual ancestor source, and the source statement is never stored again at the child. Goals below a scope already queued for removal are skipped ([D-221](../40_decisions.md#d-221), [I-164](../30_invariants.md#i-164)).

The OR-disintegration admission gate at `disintegrateExprCore2`'s OR case (the shared `consumeOrLeavesCohort`) is gated three times. First on `hasOrProofWork`: an LB without explicit goals or active primed contradiction work (evaluated at every open attempt) neither opens nor parks a cohort — the flat 1a rules still land ([D-306](../40_decisions.md#d-306)). Second on `currentOrDepth < parameters.max_or_depth`. Third on the two-route opening of [D-252](../40_decisions.md#d-252): route (b) admits when the firing rule is a product of disintegration (`LocalMemoryValue::productOfDisintegration`, stamped at install time in `addToHashMemory` from a `u_*` arg in any premise, threaded as `allowOrDisintegration = allowOrDisintegration && !doNotDisintegrate` at `addExprToMemoryBlock`'s entry — D-241's intro-fired suppression and the other `doNotDisintegrate` contexts still block; the hypothetical-disintegration path, which bypasses that door, keeps the default `false` so no cohort mints at its sentinel scopes), route (a) when some disjunct's product template is already an `admissionMap` key, route (c) when a clean leaf is an `admissionMapOrdis2` demand; everything else parks in `rejectedMapOrdis` / `rejectedMapOrdis2`. The explosion protection is the demand filter plus the sequenced one-branch-at-a-time release ([I-174](../30_invariants.md#i-174)). After outgoing publication, changed-class clearing, goal sanitization, exclusion unbinding and `drainDisprovedGoals`, the shared `standardProcessing` order is `drainDeadOrBranches` → `retireDuplicateOrCohorts` → `freezeResolvedOrBranches` (a live branch whose current chain goals are all known at the branch or above is marked resolved and queued for subtree wipeout; parent-chain goals, booked contributions and history remain — [I-206](../30_invariants.md#i-206)) → `drainPendingOrReleases` (mints each cohort's next pending branch when the live one resolved — a resolution signal that also survives an ancestor-known deposit refusal via `stageOrReleaseForRefusedDeposit` ([I-187](../30_invariants.md#i-187)); releases nothing without explicit goals or active primed contradiction work) → the `pendingWipeScopes` drain. The common cleanup tail's LAST step is `sweepAncestorKnownRows` (after the wipe drain): it drops statement-list rows at non-main scopes whose text a strict ancestor knows — the branch-first residue of the ancestor-known dedup contract — keeping the registry rows as tombstones. See [`20_core_concepts/07_or_branching.md`](../20_core_concepts/07_or_branching.md) for the full walkthrough.

### Contradiction discharge

**Ordinary implication vacuity.** After the existing main-scope contradiction sweep, `dischargeVacuousImplicationScopes` inspects unresolved positive compiled implication scopes before `dischargeToBeProved`. It selects a known opposing pair with one fact at the exact goal scope and the other at that scope or an ancestor. The scoped head receives a `vacuous truth` record citing both facts at their actual scopes and the union of their real premise levels. The normal goal closure then records the compact implication and queues the scope for the existing cleanup tail. It neither deactivates the parent logical block nor turns a child contradiction into a main-scope contradiction. Sibling-only and descendant-only pairs, OR scopes, reductio scopes and already-closing goals are excluded. Goals and fact selection use decoded lexical order; the known registry covers either fact arrival order. See [I-258](../30_invariants.md#i-258).

**The watch drain (step 4b).** Directly before the contradiction sweep, `materializeWatchHits` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) turns every canonical contradiction or goal pair into the exact pair the sweeps below need: the rows the registration doors staged since the last step are probed against the per-scope canonical ledgers (`Memory::negationLedger`, `Memory::goalLedger`) or the known registry, a per-step batch sweep serves the rows whose core is not used as a premise, live `_ordis_` branches probe their assumption's negation, and each hit writes the canonical twin of the non-canonical side (plus a goal's literal spelling) as a derived local row with one `equality1` bridge — multiplication on demand, replacing the blanket class multiplication of negated rows and goals ([D-380](../40_decisions.md#d-380), [I-246](../30_invariants.md#i-246)). The existing main-scope and reductio sweeps keep their contracts; ordinary implication vacuity uses the separate scoped-goal pass described above.

`dischargeContradiction` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)), run once per ordinary `standardProcessing` call before the scoped-vacuity and goal-discharge passes, handles the **incubator**, **vacuous-truth**, and **CE-filter** cases — whose fired heads all enter `intEncodedStatements`. It sweeps the LB's whole `intEncodedStatements` set but considers only rows at `main`; for the first main-scope statement whose negation is also known at `main`, it fires one of three mutually-exclusive reactions and deactivates the LB ([D-222](../40_decisions.md#d-222), [I-78](../30_invariants.md#i-78)):

- **Incubator** (`primedForContradiction`) — records a `contradiction` origin for `negate(cleanOp)` (= the stored theorem's head, via `negateScratch` — the `'!'`-prefix for a verbatim LB's positive seed, the leading-`!` slice for a complement LB's negated seed) and broadcasts the stored contradiction theorem (`contradictionTheoremId`, decoded) via `updateGlobalSingleChapter`. The seal also carries the route's `allLevelsInvolved` verdict, computed from the twin's own resident state: the colliding pair's level-row union EXCLUDING the twin's own level (the assumed seed's tier), against the same count-based full-run test as `dischargeToBeProved` — the premise chain occupies levels `0..twin.level-1` in all three twin geometries ([D-278](../40_decisions.md#d-278)); a false verdict still settles the parent goal through the D-240 deposit but registers nothing. Per [D-51](../40_decisions.md#d-51) the record stays local to the `__contradiction__` LB. A branch-local pair cannot broadcast this theorem.
- **Vacuous truth** (`isPartOfRecursion` at validity `main`) — records a `vacuous truth` origin (deps: expr, negation, the decoded `recursionHypothesisId`) and deposits the `toBeProved` head via `addStatement`, so the immediately-following `dischargeToBeProved` (which snapshots `intLocalEncodedStatementsDelta` at entry) closes the induction goal the same step.
- **CE filter** (`contradictionIndex >= 0`) — marks `contradictionTable[contradictionIndex].successful` and deactivates the CE LB; the refuted conjecture is dropped. CE clones are never primed / recursion, so this is the only reaction they ever take.

Contradiction LBs are created only at conjecture registration (`addTheoremToMemory`, innermost chain node), never during proving. `try_contradiction` seeds each conjecture's head verbatim into a `__contradiction__`+head child — it can only ever *disprove* the head (stored theorem = chain ⇒ negate(head)). `try_contradiction_negated_head` additionally creates the complement for every conjecture: `__contradiction__`+negate(head), seeded with negate(head), stored theorem = the conjecture itself (reductio) — the polarity that can *prove* a head, in particular a negated-equality head ([D-214](../40_decisions.md#d-214)). Both branches dedup by routing key with skip-on-exists; opposite-head conjectures on the same chain produce byte-identical colliding LBs, so the skip is lossless ([I-165](../30_invariants.md#i-165)).

**Vacuous-premise suppression** ([D-236](../40_decisions.md#d-236)). `dischargeContradiction` handles a fourth role: a NORMAL LB (no discharge role) holding a full E / !E pair at exact `main` has an inconsistent premise set — the sweep sets `Memory::mainContradiction`, deactivates the LB, and stages its subtree for the post-join `drainSubtreeDeactivations`. Theorem emission is gated at the single-threaded seams (`drainUpdateGlobalSingleChapter` via the producer LB now carried in `UpdateGlobalSingleChapterRec`; the induction promotion in `updateGlobal`) by the `hasContradictedAncestor` self-or-ancestor walk, so nothing vacuous ever reaches the global theorem list — the GL-native `ContradictoryAxioms` verdict. (The record's sealed `allLevelsInvolved` verdict is the second registration gate at the same seam — [D-278](../40_decisions.md#d-278); the vacuity `continue` skips the whole record including the D-240 deposit, while a level refusal runs lifecycle + deposit and skips only registration.) A genuine reductio is untouched: its seed lives only in the `__contradiction__` child, so the parent never flags. The same pair at the batch root or an anchor premise LB asserts (definition-set bug).

CE detection used to stay at the `addExprToMemoryBlock` insertion gate (CE fired heads were discarded, never entering `intEncodedStatements`). The CE-filter RT-opt ([D-118](../40_decisions.md#d-118)) rebuilt CE filtering as a per-conjecture thread pool where each CE LB is single-owner, so its fired heads can safely enter `intEncodedStatements` and the gate is gone — CE joins incubator + vacuous-truth in the sweep, and `burstDeactivates` stops the burst the instant the refuting head fires.

The sweep replaces the incubator + vacuous halves of the former per-insertion "Site G" block inside `addExprToMemoryBlock`. Site G fired only at insertion time on the incoming expression, so it could miss a main-scope pair whose opposite arrived later; the whole-set sweep is order-independent. It fires at most once per call and is a no-op on an already-inactive LB, so the phase-1/phase-3 double-run of `standardProcessing` cannot double-fire ([I-78](../30_invariants.md#i-78)). The phase-2 `burstDeactivates` predicate mirrors the same main/main condition so a branch-local pair cannot truncate a hashburst that phase 3 will not discharge.

Memory: `primedForContradiction` keeps contradicted LBs alive in `deactivateRecursively` so the discharge can fire; `contradictionTheoremId` and `recursionHypothesisId` carry the per-LB NameMap ids whose decoded strings the reactions cite.

**Induction-auxiliary retirement on settlement ([D-223](../40_decisions.md#d-223)).** `updateGlobalSingleChapter` retires recursion auxiliaries for BOTH polarities of every emitted theorem: the existing `deactivateUnnecessary(theorem, root)` walk covers the theorem itself (which for a complement-LB reductio emission *is* the registered conjecture), and a second walk on the head-negated twin form (`reconstructImplication(chain, negate(head))`) covers the opposite-polarity conjecture — in particular the conjecture a verbatim contradiction LB just *disproved*, whose emitted negation matches no `originalAuxyMap` entry directly. A twin form matching no entry is a defined no-op.

**Contradiction-twin retirement at emission ([D-230](../40_decisions.md#d-230)).** `deactivateUnnecessary` itself also retires both `__contradiction__` twins of the settled head at the theorem's final chain node, immediately after the path walk resolves. This is the emission-time counterpart of the [D-224](../40_decisions.md#d-224) goal-closure staging: the staged path fires only when the chain LB's own main-scope goal closes, which for a theorem proved by induction (auxy drain) or in another LB happens one or more bursts after emission — on the broadcast theorem's re-fire — so both twins idled 1–3 bursts per proof (measured: 280 twin-burst units across a Peano main run, 2–44 live twins of already-proved theorems per burst). The staged path remains as the second trigger; the flip is idempotent and runs on the same single-threaded post-join seam.

**Twin deactivation ([D-224](../40_decisions.md#d-224)).** A contradiction LB whose purpose is fulfilled retires instead of idling to batch end: when a contradiction LB fires, it stages its polarity sibling (`__contradiction__` + negate(own seed) under the same parent); when a conjecture's main goal closes through the ordinary path, the chain LB stages both its contradiction twins (a primed closer stages its sibling instead). Staging (`stageTwinDeactivation` / `stageContradictionTwinsOfHead` / `stageContradictionSiblingTwin`) runs in the worker phase, so the actual `isActive` flip is deferred per [I-28](../30_invariants.md#i-28) to `drainPendingTwinDeactivations`, drained single-threaded in `proveKernel` directly after `drainUpdateGlobalSingleChapter`. Deactivation is idempotent and commutative (no ordering work); an absent twin (reformulated-operator LB, or a batch without the complement flag) is contracted absence; an already-fired twin is a defined no-op. Twin keys are built span-only on the scratch arena via the `SimpleMapStore::findChild` span overload — the discharge functions stay heap-free.

### `reactToHypo`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Handles the hypothetical-disintegration variable-copy tag path — scopes marked with `_hypo_` payloads. It runs after mail absorption and input clearing, before class application, in both standard-processing calls. A distinguished variable with at least one peer and only `IntLev` peers receives its parent-scope copy equality through the normal deposit door. Existing class processing, discharge and outgoing publication consume the result in that same call; no additional pass is added. Earlier qualification can change proof timing and theorem coverage. The user-facing tag in the processed proof graph is `variable copy` (having subsumed the retired `reaction to hypo` and `necessity for equality (hypo)` tags).

---

## Origin tracking

Every emission records provenance into `exprOriginMap` (field of `Memory`). An entry maps:

```
(expression, validity) → list<origin-citation>
```

where each `origin-citation` names:

- The justification category (corresponds to the proof-graph tag).
- The antecedent `(expression, validity)` pairs the derivation cites.

Since `D-131` the map is stored in id form: keys and antecedents are `packOriginKey(expressionId, validityId)` int64 packs of two int32 ids from the dedicated per-LB `Memory::originInterner` (NOT the NameMap — see the decision entry), and the tag is the closed `OriginTag` enum (`originTagFromString` asserts on an unknown tag). Emission sites encode via `addOriginEncoded`. Mail history is also stored in id form: routing uses the global mail interner, outgoing history uses its private per-LB interner, and internal mail uses the LB origin interner; the commit and absorption boundaries translate between these spaces. Every order-sensitive walk (the dump section, `findEnds`, the compressor extraction) derives a decoded `(expression, validity)` lex-sorted snapshot via `decodeOriginMapSorted` — exactly the former `std::map` key order.

When stage 5 (raw proof graph emission — `visualizer.cpp`) runs, it walks `exprOriginMap` backwards from each proved theorem to build a chapter: `buildStack` at [`visualizer.cpp`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp).

The origin map is the raw material of every subsequent auditability claim. A missing origin entry means the verifier has to fail the theorem; a *wrong* origin entry (emitting a justification that isn't actually valid) is the silent-unsoundness case verified against.

### Origin map locality (D-51)

Every origin record is **local to the LB that produced it**. Mail transports (parent → children, via `mailOut → sendMail → child.mailIn → child's exprOriginMap on next iteration`) carry origins **downward only**. There is no upward propagation: a child LB's origin records do not reach the parent's `exprOriginMap`. The `pendingAncestorOrigins` queue in `proveKernel` (originally a child-side `parentMemory`-walk that wrote a contradiction LB's `("contradiction", deps)` record into every ancestor's `exprOriginMap`, deferred for thread safety per D-39) is **retired** as of D-51 — the queue is never pushed and the drain iterates an empty container. Cleanup of the dead struct/mutex/loop is deferred.

The principled consequence for chapter emission: a contradiction LB's recipe lives only inside the `__contradiction__(head)` LB. The chapter walker (`buildStack`) enters that LB explicitly via the chapter-boundary `__contradiction__` `SimpleMapStore` edge fallback when a negated head has no acyclic direct origin in the LB before the head. The former [D-49](../40_decisions.md#d-49) cap-full preference rule is superseded: current origin insertion retains existing rows and never displaces one at capacity, independently of tag priority ([I-216](../30_invariants.md#i-216)).

### `buildStack` chapter walker (D-51 algorithm + lifting per `D-56`)

`buildStack` at [`visualizer.cpp`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp) walks the proof tree starting from the chapter goal and emits one row per visited expression. Five mechanisms work together:

1. **Ancestor lifting.** At entry, `proved` is lifted via `liftToShallowestOriginAncestor(memoryBlock, proved)` — the helper walks the validity-stack ancestor chain of `proved.validityName` (recoverable from the canonical `parent + "_boundary_" + payload` string per [I-2](../30_invariants.md#i-2)) from `main` outward and returns the **closest-to-`main` ancestor** for which `(expr, ancestor)` exists as a key in `memoryBlock.exprOriginMap`. From this point on, `proved` is the lifted form: chapter rows are emitted with the lifted validity, the path-cycle filter inserts/erases the lifted form, and every dependency is independently lifted before being written into a row cell and before recursion. The dep-side lift uses the same helper. Net effect: every chapter cell (`row[1]` and every `row[3+2k+1]`) carries the lifted (closest-to-`main`-with-origin) validity, and each `(expr, lifted_v)` pair has at most one row per chapter via the `covered` dedup set. The pre-lifting fallback ("exact-key try, then `(expr, "main")` shadow lookup") is retired: lifting subsumes it and produces a truthful `row[1]` instead of one that re-tagged the row with the requested deep validity while citing a shallower origin.

 **OR-branch barrier.** The lift may not cross `_boundary_orint_` or `_boundary_ordis_` delimiters. OR-branch scopes are conditional on a disjunct hypothesis; preserving branch-distinct namespaces is required by the OR-family verifier checkers (`check_or_convergence`, `check_or_branch_proven`, `check_or_branch_assumption`, `check_or_disintegration`). The deepest `orint_`/`ordis_` ancestor sets the shallowest allowed lift target. Without the barrier, both branches of an OR converge to identical `(expr, parent_boundary)` cells (the parent boundary holds the post-convergence origin), breaking the verifier's branch-distinctness expectation and producing `origin chain termination` cycles. Boundary detection: payload (segment after `parent + "_boundary_"`) starts with `"orint_"` or `"ordis_"`.
2. **Path-cycle filter.** A `thread_local std::set<ExpressionWithValidity> g_buildStackPath` tracks the recursion path. Before picking an origin candidate, any candidate whose deps (after lifting) include an expression already on the path is rejected (would form a chapter-row cycle the verifier's `origin chain termination` check would flag).
3. **Chapter goal in path.** At `directStack` entry, the chapter goal expression (the wrapped theorem from `theoremList`) is inserted into `g_buildStackPath` and erased on return. The path-cycle filter then rejects the prover's self-applying forward-inference origin (origin tag `implication` with deps `[wrapped theorem, anchor]`) — the chapter walker would otherwise emit a `theorem`-tag leaf row matching the chapter goal, tripping `verifier.py::check_chapter` self-reference.
4. **`__contradiction__` LB fallback.** When all sorted candidates of a negated head fail (rejected by path-cycle filter, or no direct origin existed for the lifted `proved`), `buildStack` walks the LB chain (current → ancestors via `parentMemory`) for a child memory block keyed `"__contradiction__" + positive. On hit, it switches into that contradiction LB and resolves the head locally — the contradiction record is local there. **Resolution guard ([D-277](../40_decisions.md#d-277)):** both fallback sites probe the twin first via `contradictionLbHoldsRecord` (read-only reload, lift within the twin, non-minting origin-key probe, non-empty run — the exact mirror of `buildStack`'s entry-side origin check). A primed-but-never-converged twin holds no record and resolves nothing; without the guard the switch ended at the twin's entry no-origin assert even while a caller one backtrack level up still had acyclic candidates (the B10 export crash). With the guard, a record-less twin means the fallback does not apply: the entry site falls through to the loud no-origin assert (genuine gap), the post-loop site falls through to the degraded front-emit and returns false so backtracking proceeds.

`buildStack` performs **no other LB switch**. Each contradiction is an independent LB with its own chapter; nested contradiction LBs are NOT children of an outer contradiction LB. Recursion on a contradiction's three deps stays in the current LB; if a dep itself needs a contradiction proof, the chapter-boundary fallback the next level down enters another sibling `__contradiction__` LB attached to the anchor.

5. **Cross-chapter admissibility** ([D-276](../40_decisions.md#d-276), [I-190](../30_invariants.md#i-190)). The path-cycle filter is chapter-local, and a `theorem`-tagged origin carries an empty dependency vector — a used theorem is a foundation leaf, so two chapters could each ground themselves in the other's theorem with every chapter-local mechanism satisfied. The export loop therefore maintains `ExpressionAnalyzer::exportTheoremUsage` (a `TheoremUsageGraph`, declared in [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)): before any proof walk, fixed synthetic-chapter citations are seeded through `TheoremUsageGraph::addEdgesFromStack`; after each chapter file is written, `writeStackIndexed` commits the chosen chapter's citation edges (`theorem` rows; `reformulated from` / `incubator back reformulation` sources at `row[3]`; `or theorem` sources at `row[3]`/`row[5]`). In the candidate loop, directly after the path-cycle test, a `theorem` candidate citing U (the citation is `proved.original` itself) is rejected while building the chapter of theorem T (`exportCurrentChapterTheorem`, set per export-loop iteration) iff `reaches(U, T)` in the committed graph — U = T included. Rejection falls through to the existing machinery: next candidate → contradiction-LB fallback → degraded `front` emit, which the verifier's `theorem usage termination` check flags. Known reformulation, back-reformulation, OR-construction and OR-elimination obligations are present before origin selection, even when their synthetic chapters appear later. Chosen proof edges commit after writing; backtracking needs no edge rollback. Without that preparation, an early proof could cite its own later reformulation before the mandatory reverse edge became visible. An empty `exportCurrentChapterTheorem` disables the check — the defined state for `findEnds` and direct unit-test callers.

Sorting / backtracking. Per-key origin candidates are ordered by [D-49-style](../40_decisions.md#d-49) preference (non-equality tags first preserving insertion order, then `equality1`/`equality2` preserving insertion order). `buildStack` returns `bool`: `true` on success, `false` to signal the caller (recursive `buildStack`) to backtrack — roll back stack/covered snapshots and try the next candidate. Top-level callers (`directStack`, `checkZeroStack`, `checkInductionConditionStack`) ignore the return.

**Discard-aware fallback** ([D-297](../40_decisions.md#d-297), [I-201](../30_invariants.md#i-201)). The last-resort fallback is skipped in the one context where its output cannot reach the chapter. `buildStack` carries a defaulted `rowsSurviveOnFailure` parameter; the candidate-loop recursion — the sole caller that truncates `stack` back to its mark and rolls `covered` back through the insertion journal the instant it receives false — passes `false`, and a frame holding that value returns false immediately instead of emitting the front origin and walking its whole dependency subtree. The two contradiction tail calls inherit the flag (their verdict becomes the calling frame's verdict) and so does the fallback's own dependency walk (that frame ignores the verdict and never rolls back), so the degraded chapter shape is produced exactly as before wherever it is actually kept. On the FTA shortcut this removed the bulk of a pathological chapter's 4.6 M recursive calls with byte-identical chapter output.

**Rollback mechanics.** A failing candidate restores `covered` through `g_coveredJournal`, a thread-local insertion journal beside `g_buildStackPath`: every genuine `covered.insert` appends its key, a candidate records the journal length as its mark, and rollback erases exactly the keys inserted since. Exact because `covered` is insert-only inside the walk and membership-only outside it. It replaced a per-candidate deep copy of the whole set that measured 38% of the entire chapter export.

**TRIPWIRE** at function entry: the hypothetical-disintegration sentinel scope (`_boundary_hypothetical_disintegration`) is checked on the **incoming** `provedIn` before lifting, because the sentinel is a structural probe — it is not an ancestor of anything legitimate and must never reach buildStack regardless of where the consumer might lift it.

**Soundness of lifting.** An origin entry at `(expr, V)` in `exprOriginMap` exists only because the prover fired a rule producing `expr` with every premise available at scope `V`. Per [I-2](../30_invariants.md#i-2), every non-root scope inherits all parent-scope facts. Therefore: lifting from `(expr, V_deep)` to `(expr, V_root)` where `V_root` is the closest-to-`main` ancestor with origin → the rule did fire at `V_root` (origin was recorded there), so its premises were available at `V_root` or shallower → the lifted row truthfully claims `V_root`. The asymmetric direction holds: ancestor facts are universally available, descendant-only facts are not — and lifting only moves toward `main`, never away from it.

---

## Theorem export at end of batch

When `proveKernel` exits its outer iteration loop and the batch terminates, the prover writes the proved-theorem set to disk in two parallel files. Both files carry the same theorem set; the only difference is the form of compiled structural operators (`existence0..N`, `or0..N`, `and0..N`, etc.).

### The two-file dual-form export

[`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp):

```cpp
// Second file: compiled forms (for proof graph pruning — keeps or0, existence2, etc.)
const auto compiledPath = theoremsDir / "compiled_theorems.txt";
std::ofstream ofsCompiled(compiledPath, std::ios::trunc);

int written = 0;
for (const std::string& theorem : essentialTheorems) {
    if (externalTheorems.count(theorem)) continue;

    auto it = compactToExpanded.find(theorem);
    std::string toWrite = (it != compactToExpanded.end()) ? it->second : theorem;

    // compiled_theorems.txt: write as-is (keeps compiled heads)
    if (ofsCompiled.is_open()) ofsCompiled << toWrite << "\n";

    // theorems.txt: expand runtime expressions (or0, etc.) to base form
    // so subsequent batches can parse them.
    toWrite = expandToBaseForm(toWrite);
    ofs << toWrite << "\n";
    ++written;
}
```

`essentialTheorems` is the compressor-survivor set (the theorems that remain essential after [`05_compressor.md`](05_compressor.md)). Surviving externals are skipped — they weren't proved here, they were imported. (The prover no longer fabricates `mirrored statement` rows, so there are no mirror-of-external entries to exclude — D-112; the reverse direction is now a genuinely-proved conjecture, see [`02_conjecturer.md`](02_conjecturer.md).)

For each survivor, the loop emits:

- **`compiled_theorems.txt`** — compact form. Keeps every compiled head (`existence2`, `or0`, …) as the literal token allocated during this batch's run. Truncate-each-batch (`std::ios::trunc`).
- **`theorems.txt`** — same content, but `expandToBaseForm` rewrites every compiled head into its base-form definition. `existence2[N,i,s]` becomes `!(>[8](in[8,N])!(in2[8,i,s]))` (the `¬∀y∈N: ¬in2[y,i,s]` definition); `or0[a,b]` becomes `!(&!(a)!(b))`; etc.

### Why two forms

The expansion at line 7817 exists for one specific reason: a downstream batch (e.g. Gauss main reading Peano main's theorems) might not have the same compact-name dictionary loaded at parse time. Compact heads are batch-local allocations — Peano's `existence2` is *Peano's* compact name for that expansion. If Gauss main saw `existence2` as a raw token without the Peano dictionary, parsing would fail. Writing `theorems.txt` in expanded form gives downstream batches an operator-dictionary-free input.

`compiled_theorems.txt` keeps the compact heads because **proof-graph artifacts also use compact heads**:

- `visualizer.cpp` writes raw chapters with `rest[]` fields citing rules in compact form (whatever the prover's runtime had in `exprOriginMap`).
- `process_proof_graphs.py` propagates these compact citations into the processed chapter rows.
- The verifier's `origin` check (see [`08_verifier.md`](08_verifier.md#origin-meta-check)) looks `rest[0]` up in the registry by alpha-canonical match — without operator expand/compact normalisation. So the registry has to contain the rule in the same form the chapter cites: compact.

This duality is load-bearing. Removing either side breaks something:

- Drop `compiled_theorems.txt` → the Stage-2 verifier-registry route (see [D-40](../40_decisions.md#d-40)) loses its compact-form source; chapter `origin` lookups miss.
- Drop the `expandToBaseForm` step in `theorems.txt` → downstream-batch parsing of non-shared-binary compact heads fails.

### Cross-batch propagation: which file feeds the next batch's externals

The orchestrator (`run_modes.py`) selects between the two files when seeding a downstream batch:

| Channel | Source file | Reader | Form |
|---|---|---|---|
| Inter-batch parser feed (default) | `theorems.txt` | C++ prover loading externals at startup | expanded |
| Externals seed for next-tag incubator (since [D-40](../40_decisions.md#d-40)) | `compiled_theorems.txt` | C++ `--mirror-externals` mode + verifier registry | compact |

The compact-form route is safe because `GL_binary_shared.json` (`run_modes.py:_seed_per_batch_binary` + `_merge_into_shared`) carries the spontaneous-category compact dictionary across batches — see [`09_incubator.md` § Cross-batch externals seed](09_incubator.md#cross-batch-externals-seed).

### Per-batch GL binaries

Alongside the two theorem files, the prover writes `files/GL_binaries/GL_binary_<tag>.json` — the compact-name dictionary specific to this batch. Schema entries are name → `{category, signature, …}`. The write path is project-rooted (resolved by `project_paths.cpp`, `projectRoot`, in `visualizer.cpp`, `generateRawProofGraph`), so every batch — incubator and main alike — writes into the single canonical `files/GL_binaries/` directory; see [D-54](../40_decisions.md#d-54) for the unification of a previously asymmetric writer path.

The orchestrator post-batch step `_merge_into_shared` (`run_modes.py`) selects the spontaneous categories (`implication`, `existence`, `or`, `and`) and unions them into `GL_binary_shared.json`, which the next batch's `_seed_per_batch_binary` (`run_modes.py`) copies back as that batch's per-batch starting dictionary. **Tags whose name begins with `Incubator` are skipped** — incubator-allocated spontaneous names are batch-local; merging them into shared would shift the next main batch's counters and rename its operators, breaking [I-23](../30_invariants.md#i-23) for the main pipeline.

Non-spontaneous categories (anchor entries, atomic entries) stay in the per-batch file only — they are by-design batch-local and never cross.

### File-write summary

| File | Form | Truncate semantics | Audience |
|---|---|---|---|
| `files/theorems/theorems.txt` | expanded | per-batch trunc (`prover.cpp`) | inter-batch parser-input fallback |
| `files/theorems/compiled_theorems.txt` | compact | per-batch trunc | proof-graph pruning + cross-batch externals seed (D-40) |
| `files/theorems/compressed_external_theorems.txt` | rewritten by `--mirror-externals` | per-step write | C++ prover externals load |
| `files/GL_binaries/GL_binary_<tag>.json` | per-batch compact dictionary | per-batch overwrite | next batch's startup load |
| `files/GL_binaries/GL_binary_shared.json` | accumulating spontaneous-category union | append (entries not overwritten) | cross-batch dictionary |

---

## Where invariants live

| Invariant | Code location |
|---|---|
| [I-1](../30_invariants.md#i-1) — precompile before theorem-load | `precompileStructuralOperators` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) |
| [I-2](../30_invariants.md#i-2), [I-3](../30_invariants.md#i-3) — scope mint + ref copy | `NameMap::encodePush` at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp) |
| [I-4](../30_invariants.md#i-4) — one binder rule, bind every non-`u_` var ([I-5](../30_invariants.md#i-5) retired) | `reconstructImplicationFullBind` + `reconstructImplication` forwarder at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) |
| [I-6](../30_invariants.md#i-6) — single-input Pass B gate | `isAllowedAsOperatorInput` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) |
| [I-7](../30_invariants.md#i-7) — Pass B gated on `!parameters.ban_disintegration` | `disintegrateExpr2` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) |
| [I-12](../30_invariants.md#i-12) — equivalence-class one-sided expansion | `applyEquivalenceClassToNegatedEquality` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) |
| [I-17](../30_invariants.md#i-17) — `savedStartInt` single-counter | `disintegrateExpr2` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) |
| [I-18](../30_invariants.md#i-18) — induction typing | shipped; auxiliary `(in[digitArg, N])` triad before promotion, `<N>_induction_typing.txt` chapter, verifier `induction typing` checker |

---

### Induction heads and flat integration

The recursion-step LB records its installed induction implication identity. A head fired by that exact implication carries `recursionHead` through processor/CUDA firing, sealing and internal mail, and takes `forceDeep` at `addExprToMemoryBlock`. This admits forced hypothesis witnesses and their defining products without demand, drops the new algebra-side products, and does not enroll products of recursion.

`prepareIntegrationCore2` Case B additionally calls `emitFlatIntegrationVariants` for its root marked goal. Generated conjunction/existence wrappers dissolve; configured containers branch intact or one-level-expanded; other expressions remain leaves. Unique deeper premise multisets install as ordinary integration rules with explicit flattened-construction history. The existing preparation and admission keys remain active. Integration demand variants retain their justification, require six non-anchor premises and are exempt from the firing purity gate; general demands retain four and their existing purity rule. Both firing implementations and route-(c) opening exclude already-known same/ancestor facts. See [D-352](../40_decisions.md#d-352) and the adjacent demand decisions.


The flat-integration walk in `prover.hpp` / `emitFlatIntegrationVariants`
keeps an immutable stack-local continuation for each expanded child. Resuming
a sibling may enumerate choices at the same depth while the earlier child
still has alternatives; those alternatives retain their own parent and next
element. Thus every emitted construction includes all siblings, including
after nested configured choices. The walk remains heap-free and retains the
same intact-before-expanded order, depth assertions and premise-multiset deduplication.

### Induction admission needs a numbers-only original

An admission that comes from an induction chain (a recursion-owned template, [D-413](../40_decisions.md#d-413)) serves a disintegration only when the original expression being disintegrated has numbers as its only arguments: nonempty decimal tokens or `rec` (`memory_infra/witness_names.hpp` / `hasOnlyNumberArguments`). A witness of any class, an anchor copy such as `x2` or a replacement name makes the original dirty. This is the stricter sibling of the integration depth read from the original by `assignedIntegrationDepth`.

In `prover.cpp` / `addExprToMemoryBlock` every out-statement of a disintegrated deposit other than the deposit itself is a product of that disintegration, and only for those is the predicate evaluated on the deposit (a preparation `u_` prefix is stripped first, as in the algebra-input gate); for a product of a dirty original `updateAdmissionMapRecursion` receives `inductionAllowed = false`, and a recursion-owned template that matches the product does not move its chain: no re-keyed chain, no derived entry, no integration preparation. The door's probes and the default-false status entries ([D-172](../40_decisions.md#d-172)) are unchanged. The deposit itself and every deposit that is not disintegrated propagate as before, so a derived chain such as `fold[1,3,4,5,W,2,rec,marker]` still advances when its fold value arrives. Integration preparation, integration-route admission and forced disintegration are unchanged.

The disintegration admission applies the same rule to operator outputs. `prover.hpp` / `isAdmitted` skips a recursion-owned value (`flagByte!= 0`) when `inductionAllowed` is false: it neither admits the witness nor moves its chain in the commit; an ordinary value on the same template admits as before. `isAdmittedIncludingAncestors` forwards the flag to every scope it probes. `prover.cpp` / `disintegrateExpr2` evaluates the predicate once on its original inside the Pass B branch and passes it to the route-1 probe and commit. `prover.hpp` / `queueParkedExistence` evaluates it on a parked algebra source, which becomes the original of its replay, so a recursion-owned demand does not wake a source it could not admit.

In the forward-F3 induction step the hypothesis key `(sequence[1,4,2,9,marker])` thus moves its chain for the restriction of the step sequence (an original over 9, 11, 1, 4, 2) and not for the appends of the witness sequence `int_0_lev_6_8`. See [I-285](../30_invariants.md#i-285) and [D-445](../40_decisions.md#d-445).

### Induction chains do not feed on the hypothesis' own constructions

The forced disintegration of a statement fired by the installed induction implication ([D-356](../40_decisions.md#d-356)) records every witness it publishes in `Memory::recursionHeadWitnesses` (`prover.cpp` / `disintegrateExpr2`, `recursionHead`). A recursion-owned template that matches a statement whose bound argument is such a witness ends its walk in `updateAdmissionMapRecursion` before any derived entry, products-of-recursion enrollment, integration preparation or revisit. In the forward-F3 induction step the hypothesis fires a fold over [2,9] of the restricted step sequence; its accumulator `force_lev_6_71` is a sequence on [2,9], which used to meet the hypothesis premise demand `(sequence[1,4,2,9,marker])` again and open the never-finishing task `fold[1,3,4,5,force_lev_6_71,2,rec,marker]`. See [I-286](../30_invariants.md#i-286) and [D-446](../40_decisions.md#d-446).

### Parent proof delivery during mail absorption

A convergence completed by a mail arrival sends its parent statement and convergence history to the current `standardProcessing` output channel, passed explicitly through `addExprToMemoryBlock` to `ordisMerge`. The input is still cleared after its fixed snapshot drains. Calls from equivalence processing or shared scope retirement retain their ordinary same-iteration destination. This preserves the existing schedules while preventing newly produced parent proofs from being cleared as already-consumed input.

## Weaknesses

### Known & tracked


- **An induction demand can stop at an unbound compound relation and mutually missing operator premises.** `prover.hpp`, `updateAdmissionMap`, retains the D-257 exclusion of one-free-argument compound-relation marker demands, and `prover.cpp`, `updateAdmissionMapRecursion`, advanced known facts only for configured operators and prepared no integration (both lifted for recursion-owned chains by the experiment below). If the later rule has two operator premises with distinct output variables, `memory.cpp`, `makeNormalizedKeysForAdmission`, can make each completion key retain the other premise. Neither construction starts when both are absent. The reverse-Fold induction trace exhibits this combination with a sequence relation and two Fold premises, despite known restriction existences. This records a completeness boundary, not authorization to relax D-257 or restore broad premise prefill. The maintainer-approved experiment [D-413](../40_decisions.md#d-413) opens the exception for recursion-owned chains, without configuration changes: every positive non-relation compound premise with exactly one unbound argument takes the marker route at install, and `updateAdmissionMapRecursion` binds an operator's output and any argument of such a compound, and derives, with prepared integration, a demand for an operator left with only its output free and for a compound left with one free argument. Two-input relations keep the skip. The traced admission chain reaches `(sequence[1,4,2,<m>,11])` with only 11 unbound, so the demand wakes the parked restrictions; an arriving restriction re-keys the hypothesis, and its value at `<m>` and its folds are demanded as constructions. On the branch's previous rule the fold integrations ran into a cascade of subproof scopes that overflowed the step block's canonical class store (see the decision). 

- **Independent inner demand can activate a larger enclosing construction.** In `prover.cpp`, `disintegrateExpr2` / local `selectWitness`, admission recursively coadmits enclosing witnesses even when the matched constituent does not mention them. Explicit-goal and rule-premise demand retain this contract. Automatic startup demands from known operator premises are disabled as described above; this removes one producer of broad Fold demands without changing coadmission itself. 

- **Canonical class rebuilding remains a runtime cost under broad live class growth.** Watch identity updates now select only changed normal representatives/new members, reuse the live original/scope index, and canonicalize identifiers without proof metadata ([I-269](../30_invariants.md#i-269)). Complete descendant class views and all equality evidence retain their existing refresh timing. 

- **Resolved: ordinary implication scopes can discharge vacuously.** `dischargeVacuousImplicationScopes` supplies the recorded scoped head and normal goal closure retires the scope. The motivating saved singleton Fold run retained 140,206 rows in 19 contradictory transition scopes; validation of the correction is recorded in the task ledger. The independent statement-selection window is now configured by `maxNumberFilteredStatements`; increasing its bound does not establish complete search.

- **The statement filter can hide an already-proved premise.** [`memory.cpp`](../../GL_Quick_VS/GL_Quick/src/memory.cpp), `filterIntEncodedStatements`, stops after `maxOut` accepted rows; `generateEncodedRequestsStatic` and `produceExpressionStumps` pass `maxNumberFilteredStatements` (default 16384), with their index buffers allocated from the static per-worker arena. The CUDA route preserves the same configured prefix in [`gpu/phase2_cuda.cu`](../../GL_Quick_VS/GL_Quick/src/gpu/phase2_cuda.cu), `DevicePhase2FilterCall::maximumFilteredStatements`, `phase2FilterCountKernel` and `phase2FilterEmitKernel`. Selection precedes the stable name sort and has no continuation through later eligible statements. In the saved singleton Fold trace before scoped-vacuity repair, all five root-integration premises were known at `main` and their complete normalized key was installed, but the transition premise lay beyond this selection in both overall and local rule pools. Scoped closure removes that workload's overflow; the generic bounded-window limitation remains. The forward F3 trace likewise contains a proved recurrence premise at eligible rank 9452 of 13405; the previous 8192 cap excludes it even though its unary graph-inclusion rule is installed. The configurable increase is recorded in [D-408](../40_decisions.md#d-408). The block can become quiescent with the goal still open. This is a search-completeness limitation of the retained candidate window, separate from the sound `u_` literal prune; changing its semantics requires an approved design. Because the window is a prefix in statement order, any change that removes rows ahead of an eligible statement moves that statement toward the window: the finished-task wipe ([I-284](../30_invariants.md#i-284)) removes 16.9 k rows of the slowest forward-F3 block, which then stays inside the window until burst 27 instead of crossing it at burst 25 (18,344 eligible rows; every eligible row past statement index about 43.8 k dropped, including the typing facts of three accumulator-candidate subproofs). The released search adds 63 k statements in bursts 28-31 without proving the block's goal and costs about 32 worker-seconds; the window, not the wipe, bounded that search before.


- **Counterexample construction remains one-witness only.** Negative implication compacts with zero or multiple local variables remain stored facts. A negative compound assumption can produce an exclusion rule whose later fired head takes ordinary rather than assumption disintegration; no assumption treatment is implicitly inherited through that firing. The counterexample exception admits its one witness, not all nested existences or arbitrary derived heads.

- **FTA burst-16 compiler trigger — repaired.** Public-predicate reuse and
 numbered retirement prevent the duplicate existence alias inside strictOrder.
 In the original failing profile, false branches amplified two costs:
 the former `checkForEquivalence` Cartesian enumeration repeated argument
 occurrences (63^4 trials for one 63-choice variable), and
 `MergeClassAccum::serializedSize`, `serializeInto`, and `forEachOriginSorted`
 repeatedly grouped flat history lines through nested scans. The identity
 check now uses complete-scoped posting indexes, and the history walks group
 the lines once by (key, insertion index) (`orderLinesByKey`); the historical
 probe timing does not measure this implementation. Class write-back
 and changed-class tracking serialize the same merged state twice. The measured
 target spends about 9.55 seconds in equivalence probes, 12.14 in these two
 serializations and 3.40 in history synchronization. The valid surviving
 9-to-41 merge also occurs in the legacy fast run and is not the discriminator.
 See [`rt_restoration_plan.md`](../../GPU/rt_restoration_plan.md) for the exact
 chain, controlled inputs and diagnostics. The compiler repair passes the
 complete GPU pipeline, unchanged verifier and connected Lean.


- **Ancestor-park history copy — resolved.** Traps established that the old direct-disintegration-only copy rejected an existing parent-scope `equality1` row. The maintainer-approved generic copier preserves every stored row and imposes no history prerequisite on release. All native GPU batches subsequently completed, Lean checked the exported Peano/Gauss project, and the unchanged verifier passed 141,136 checks with zero failures. Finalization was resumed outside the sandbox because the sandbox hid the already-installed Lean toolchain; the original process failure remains recorded. The older pre-merge runtime explosion remains a separate historical observation. See [`rejected_scopes_merge_plan.md`](../rejected_scopes_merge_plan.md).
- **Park-after-demand at a descendant scope — resolved.** `prover.hpp`, `wakeParkedExistencesAtExistingDemands`, matches newly parked sources against existing demand through the derived canonical indexes and the ordinary admission probes. Both arrival orders retain ancestor-to-descendant replay; see *Parked existences and normal replay* above.
- **Induction-typing soundness — closed.** Historically the prover scheduled induction without first verifying membership in `N`, which was unsound for bound variables appearing only in negations / existence heads / bare equalities. **Closed:** the prover now proves the typing triad before promotion; see [I-18](../30_invariants.md#i-18). Listed here for historical context.
- **Pass B single-input gate is empirical.** [I-6](../30_invariants.md#i-6) is enforced because widening broke Gauss. There is no theoretical argument here — only a measurement-grounded guardrail. A hypothetical operator with single-input semantics but admission-map requirements that the gate does not check would slip through.
- **Incubator contradiction scope under OR disintegration — closed.** `dischargeContradiction` now requires both opposite expressions at `main`; a deeper duplicate is ignored by its branch row but does not hide a separate main row. The phase-2 early-exit mirror has the same condition. This prevents the IncubatorGauss3 rung-2 false `EnumerationSet2` / `EnumerationSet3` negated-interval theorems. Rejecting a contradictory `_ordis_` branch and re-checking convergence over the surviving branches was closed later by dead-branch retirement ([D-242](../40_decisions.md#d-242), [I-172](../30_invariants.md#i-172)) — the main/main contradiction contract itself is unchanged.
- **`excludeRepetitions` zero-free-arg signature — fixed.** `excludeRepetitions` (`prover.hpp`) builds a compiled-operator signature by appending `"u_<i>,"` per unchangeable arg then unconditionally `pop_back`-ing the trailing comma. With `numUnchArgs == 0` the pop ate the `[` and produced the malformed `"(nameN])"`. The trailing-comma strip is now guarded by `numUnchArgs > 0`, so a zero-free-arg compile (a fully-bound theorem, every variable bound per [D-75](../40_decisions.md#d-75)) yields the well-formed `"(nameN[])"`. Byte-identical for the ≥1-arg case (all pre-existing `or<N>`/`existence<N>`/`and<N>`/`implication<N>` allocation), so it cannot regress existing proofs; the corrected branch is only reachable on the new top-level-implication compaction path.

- **Chapter-walker cost depends on the workload.** The earlier multi-second shortcut backtracking and incubator reload costs are documented, with their fixes, in [`runtime_ideas.md`](../runtime_ideas.md). Current coarse measurements are available in the frame-timing ledger. The full native CUDA baseline in [`rt_parts_plan.md`](../rt_parts_plan.md) measured IncubatorPeano1's 946 chapter walks at 0.1093451 seconds on 2026-09-06. Profile a new workload before proposing chapter parallelism, reload changes, or another search algorithm; the earlier measurements do not establish a current bottleneck.

### Suspected fragility

- **`ExpressionAnalyzer` is a monolith spanning `prover.hpp` and `prover.cpp`. Every proof primitive is a method on the same object. Refactor planned post-FTA (see ).
- **Parallel grid is live (no longer a fragility).** `logicalCores = std::max(1u, std::thread::hardware_concurrency)` (`prover.cpp`; the old `logicalCores = 1` override is commented out), and `proveKernel` runs real worker pools — across-LB sweeps for phases 1/3, a flat `(LB, part)` executor pool for phase 2. The push-mail `smashMail` / per-core-mailbox batching this note once warned about is **retired**, replaced by the pull-model `MailLog` ([D-137](../40_decisions.md#d-137) / [I-94](../30_invariants.md#i-94)); determinism under parallelism rests on [I-28](../30_invariants.md#i-28) (no cross-LB intra-cycle observation; sorted post-join drains) and the canonical-sort merge in `applyFiringRecords`.
- **Proof-cycle iteration cap.** `parameters.numberIterationsConjectureFiltering` and the main-mode iteration budget are batch-wide constants. A single unusually-hard conjecture can consume the full budget while the rest starve. No per-conjecture timeout.
- **`savedStartInt` freshness contract.** Any refactor that introduces a parallel counter violates [I-17](../30_invariants.md#i-17) silently.
- **Readers that still take `=`'s slots as `(1)`** ([I-267](../30_invariants.md#i-267)).
 - `multiplyImplication` counts a variable as `(1)`-typed when it sits in any `(1)` slot, `=` included, so a rule with a set-typed `=` would get equalisation copies mixing sets and numbers. Today no such rule reaches it: multiplication is off in FTA and runs only in the incubators and the counterexample filter, which never see a set `=`.
 - The compile-time `ArgumentAnalyzer` and the compact slot typing (`findDefinitionSetRecursive`) type `=` as `(1)`. They see definitions only, and no definition has a set `=`; one that did would assert at config load.
- **Operator set inference — resolved.** `ExpressionAnalyzer::ExpressionAnalyzer` populates `operators` directly from `coreExpressionMap` using the same non-empty input/output test as the conjecturer; there is no second operator definition that can drift. See resolved OPEN-4 below.

### Worked example: delayed integration construction

An outer expression with no ordinary integration names receives assignment 0. Its structural expansion exposes an existence whose body contains `(in3[a,w,b,4])`. If the matching `(in3[a,marker,b,4])` demand is absent, `disintegrateExpr2` parks the concrete existence source, its scope, levels, processing flags and assigned depth 0. It does not mint `w` or retain witness-dependent products.

When an equality replaces `a` with `9`, `rewriteParkedExistences` canonicalizes the source and marked key together and carries the depth unchanged. Deferred equality evidence remains attached to the source record. A matching demand in that scope or a descendant lets `queueParkedExistence` queue the source through internal mail with the saved assignment. Duplicate pending routes merge to the minimum assignment; completed source/scope constructions remain suppressed.

`standardProcessing` absorbs the source and passes its assignment to `disintegrateExpr2`. The admitted construction mints `int_0_lev_<level>_<serial>` and publishes its supported local products. A delayed nested source may itself contain a depth-0 or depth-1 witness from the outer expansion; it still uses its original saved assignment. Membership in `rejectedMapIntegration` establishes prior depth eligibility. Newly encountered outer expressions compute their own assignment, and those above `maxIntegrationDepth` can still yield structural facts and implications without minting or parking another ordinary integration witness.

The source record remains available for independently demanding descendant scopes. Proof history records the source transformation, scope transfer, expansion and supported disintegration products; it is never an admission input. Outgoing facts and compact rules containing the resulting ordinary integration name are refused by the mail boundary, while internal replay and proof-history transport continue.

---

### Case A known parts and settled implication subproofs

**Known-part check.** Before `prepareIntegrationCore2` Case A opens a subproof for an implication compact, `compactKnownUpToEquality` asks whether the compact is already a known statement at the preparation scope — exactly, or through that scope's complete classes (the shared statement posting index of `checkForEquivalence`, without its fully-disintegrated condition). A known compact needs no subproof. The check must compare through the classes because admission is canonical ([I-254](../30_invariants.md#i-254)): after an equality the admission re-instantiates an instruction with the class representative, and an exact-text check misses the parts known under the old member. Measured in the F3 goal block (`ledger/fta_f3_rt3_set_equality_ledger.md`): after row 84 merged the [0,10] intervals, the fold goal's re-admission opened 90 subproofs for the three `fXY` parts of 30 known facts (`implication0` / `4` / `5` over `int_lev_2_0`, about 420 scopes); with the check none opens.

**Implication gate (parked subproofs).** Case A lives in `openImplicationSubproof`, the one opening path of an implication element. After the known-part check and the prepared gate it applies the gate of [I-270](../30_invariants.md#i-270): every and of the instruction that lists the implication must have all its gate elements (non-implication elements with fixed arguments, negated and compound included) known at the scope or an ancestor, up to the scope's classes (`elementKnownAtOrAbove`). Otherwise the implication is parked in `rejectedMapSubproof` under the first unknown element (`parkImplicationSubproof`) and nothing is prepared or opened. Every newly known row, and every scope whose complete class view changed, is staged while a park exists; `drainSubproofArrivals`, first in step 4b of `standardProcessing`, re-keys and rechecks the parks under changed scopes (a merge can make a gate known without a new row), matches the staged rows against the park keys at their scope and below and hands each hit to `revisitRejectedSubproof`, which reopens the records exactly as stored — the instruction's own text, which the and's integration rule expects — through `openImplicationSubproof` (or parks them under the next unknown element; the key is canonical) and drops records whose root goal is gone or whose parent and is known. Ands never park. In F3's fold step this keeps every wrong candidate's `implication32` / `implication33` subproofs closed until `in2[10,14,g]` is known for that candidate ([D-406](../40_decisions.md#d-406)).

**Settled-subproof retirement.** The end-of-burst `retireSettledImplicationSubproofs` sweep (after the OR freeze and release drains, immediately before the `pendingWipeScopes` drain) wipes an open bare-payload implication subproof — opened by an arriving statement, or nested below another scope — once its implication is known at its parent, and of such subproofs whose implications became class-equal under one parent keeps the one with the most statements in its subtree (ties to the smaller payload). Goal-driven `<goal>_subproof_` scopes follow their goal. The finished-task wipe (section **Integration**) runs just before it and queues a finished root's scopes on the same drain. See [I-271](../30_invariants.md#i-271) and [D-414](../40_decisions.md#d-414).

---

### 3-way admission-map insert fan-out

`prepareIntegrationCore2` Case B at `prover.hpp` calls `addToHashMemory(...)` **three times** — once each with target = `mb.overallHashMemory`, `mb.localHashMemory`, `mb.localHashMemoryDelta`. Each `addToHashMemory` invocation iterates `triggersForAdmissionSetIntegration` and calls `makeAdmissionKeys(..., &mb)`, which — per [D-19](../40_decisions.md#d-19) — includes:

- `admissionMapIntegration[keyString, validity][instructionCopy];` (default-insert of the instructionCopy sub-key into the template map).
- `revisitRejectedIntegration2(bareKey, mb, validityName)` (stripped of u_).
- `admissionSetIntegration.insert(...)` + `revisitRejectedIntegration2(admissionKey, mb, validityName)`.

**Implication:** for one logical "admission template is now registered for compound C" event, `revisitRejectedIntegration2` can fire up to **3× per `makeAdmissionKeys` call** (overall + local + localDelta) **× number of trigger templates** (outer loop in `addToHashMemory`). Each call is idempotent — the second and third calls at the same marker key find `rejectedMapIntegration[(key, validity)]` empty (since the first call already erased it), returns fast. No correctness issue, but a perf hotspot at FTA scale where trigger counts grow.

**Do not** add guards of the form "skip if not overallHashMemory" — the three-way insert is structural (see `prover.hpp` Case B for rationale: local/delta mirrors are needed for static-hotpath request generation). The cheap idempotent revisit is the right design; any "optimization" that conditionally skips the revisit for `local*` targets must first prove that `overallHashMemory` will always have the key available at the revisit moment.

### `rejectedMapIntegration` and ancestor-scope eq classes

`applyEquivalenceClassToRejectedMapIntegration` matches a class against an rmi entry when either:

- `keyEv.validityName == validityName` (class's scope), same-NS case; OR
- `validityName` a strict ancestor of `keyEv.validityName` (`memoryBlock.nameMap.strictAncestorNames`), ancestor-NS case.

This is the same rule the main `applyEquivalenceClass` uses at its ancestor-NS call site (`prover.hpp`). The *class* applies at whichever scope it was registered, the *rewritten entry's validity is the rmi key's original validity*. So a class at `main` validly rewrites a rejected constituent whose compound lived at a deeper scope (e.g. `main_boundary_(implication23[…])_…`) without promoting the rewrite's validity.

**Edge case to watch:** if the rewritten constituent emits to `sameIterationInternalMail` with the rmi key's (descendant) validity, the next hashburst's absorb adds it at that validity. But `updateAdmissionMapIntegration` is gated on `validityName == "main"` at `prover.hpp` — so a non-main revival emission does NOT trigger admission-map updates at main, and integration templates that would admit the rewritten form at main never see it. This is a known asymmetry; if FTA rungs need branch-scope revivals to cross-trigger main-scope integrations, the gate needs relaxation (flag as open question, not a 2026-04 fix).

---

### Scaling pressures on `rejectedMapIntegration` beyond `{0,1}=[0,1]`

Current design works for the Gauss `{0,1}=[0,1]` FTA-ladder rung. Three pressures are already visible and expected to bite on subsequent rungs (Euclid, FTA proper):

1. **`rejectedMapIntegration` growth is unbounded.** Pass B keeps emitting fresh entries per rejection; with full-permutation, the no-match-persist path inserts multiple new keys per old entry per class call (up to the number of distinct mappings that produce distinct rewritten keys). Observed ~10^5 entries mid-Gauss; richer fact bases will push this 10–100×. Today's linear walk in `applyEquivalenceClassToRejectedMapIntegration` is O(|rmi|) per class call despite the short-circuits. **Planned mitigation:** reverse index `varToKeys: unordered_map<string, unordered_set<markedKey>>` maintained on insert/erase, so each class call touches only overlap-relevant entries (O(|affected|) instead of O(|rmi|)). Would also subsume the current `varsInRejectedMapIntegrationKeys` cache.

2. **`sameIterationInternalMail` is per-LB, unbounded.** Revival emissions accumulate across a hashburst and drain at the top of the next. A runaway revival chain at one LB throttles that LB with no intrinsic cap. Per-hashburst dedup of `(stmt, validity)` emissions would help (partially handled by `std::set` semantics on `statements`, but the equality1-origin trackHistory path still allocates).

3. **`varsInRejectedMapIntegrationKeys` is monotonic.** Grows forever — never shrinks when entries leave rmi via revival. At FTA scale it converges to ~"every arg ever seen" and stops being selective. Subsumed by mitigation (1).

On the current rung these pressures cost ~30% runtime (405s → 525s going canonical-only → full-permutation) but don't regress theorems. On the next rung they may become load-bearing.

### Debug trap patterns for Pass B / admission / revival paths

When diagnosing silent drops or revival misfires, the idiom that worked for the `rejectedMapIntegration` development was a **static `std::ofstream` at the emission site**, gated on a counter:

```cpp
{
    static int __dbgCount = 0;
    static std::ofstream __dbg(".debug/NAME_trace.txt", std::ios::out);
    if (__dbg.is_open()) {
        __dbgCount++;
        if (__dbgCount <= 50 || (__dbgCount % 500) == 0) {
            __dbg << "event #" << __dbgCount
                  << " key=" << markedKey
                  << " v="   << validityName
                  << " LB="  << mb.exprKey
                  << "\n";
            __dbg.flush();
        }
    }
}
```

Properties:

- **Thread-safe without a mutex** — the static is per-thread-init, so each worker thread touches only its own copy (no shared state across the parallel grid).
- **Self-bounded output** — first 50 events logged verbatim, then every 500th. Catches the start (when the pattern first fires) + periodic samples (to track steady state).
- **Opens once per process** — no file-open-per-emission overhead.
- **Clean removal** — one `git rm` of the block after diagnosis, no header changes, no call-site changes elsewhere.

During this session five such blocks were placed at `updateRejectedMapIntegration` (write count), `emitIntegrationRevivalToInternalMailIn` (emit count), `applyEquivalenceClassToRejectedMapIntegration` (class invocation), `revisitRejectedIntegration2` (find-count with `found=YES/no` tag), and the admissionMapIntegration insert site — then all five removed. Future agents reintroducing them for another silent-drop investigation can `git show 51750a6^:prover.hpp` and copy the exact blocks back.

**Complement this with `git show 51750a6 -- prover.hpp | grep -B2 -A15 "__dbg"`** for the ready-made block template. The counter-gating pattern works for any emission/insert site where you suspect a silent drop or a low-frequency corner case.

### Known silent-drop sites (historical + current)

A "silent drop" is a code path where a statement, admission, or rejection is discarded without a log, counter, or assert — invisible unless you happen to trap at that exact site. Catalog:

- **Pass B `int_` branch — FIXED (now buffers to `rejectedMapIntegration`).** Before [D-19](../40_decisions.md#d-19), the `int_` branch at `prover.cpp` silently dropped any fresh `int_` witness that failed both `isAdmittedIntegration` and the `admissionSetIntegration` fallback. No buffer, no revisit path, no counter. Symptom: theorems requiring deferred admission (after an equality rewrite would admit the int_ witness) never proved. Gauss `sequence / limitSequence / fold` theorems were blocked on this. Fixed by the `rejectedMapIntegration` mechanism.
- **Pass B `it_` branch `else` clause when core is not an operator** (`prover.cpp`). If `extractExpression(removedU)` is not in `operators`, the whole `hasOperator` branch is skipped and no buffering happens. Whether this is a real silent-drop depends on whether non-operator body elements can ever arise for an `it_`-bound stmt — not currently exercised by known theorems, but the path is un-asserted. Tag: suspected fragility, not confirmed.
- **`applyEquivalenceClass` statement-growth cap.** If `countPatternOccurrences > maxNumberSecondaryVariables` the rewritten statement is silently dropped (`prover.hpp`). This is intentional fan-out control, but the drop itself is unlogged — a theorem that depends on the rewrite never proving may look like an unrelated prover issue.
- **`addExprToMemoryBlock` ancestor-scan duplicate suppression (`prover.cpp`).** Redundant deposits are silently ignored — correct behaviour, but a buggy caller that depends on the *second* deposit having different semantics would fail silently.

**Why this matters.** The algebra side (`rejectedMap`) was a fix for a silent-drop pattern years ago; the `rejectedMapIntegration` fix (2026-04) was the equivalent for integration. The pattern repeats: if a new admission category is added later (e.g. contradiction-side, or a new anchor-family), check for its own silent-drop equivalent.

### Not exercised by tests

- **Contradiction-from-multi-hypothesis scope.** Tested implicitly by incubator runs, not by a minimal regression. A refactor of `primedForContradiction` propagation would go unverified.
- **`multiplyImplication` partition correctness for arity > 3.** The Bell numbers grow fast (B₃=5, B₄=15, B₅=52). In practice Peano/Gauss theorems rarely have more than 3 `(1)`-typed bound variables, but a future anchor expansion with more typed slots could expose a partition-generation bug.

---

## Open questions

- **OPEN-WALKER-NOGOOD — RESOLVED upstream (2026-08-30): the export abort was a hole in the origin map, not a walker weakness.** A least-fixpoint derivability pass over the exact graph `buildStack` searches (lifting, contradiction-twin fallback, inadmissible-`theorem` skip, terminal tags) showed the `check_zero` goal had no acyclic derivation; 191,804 of 191,814 walker failures were on provably non-derivable nodes. Cause and fix: [D-325](../40_decisions.md#d-325). The memo below stays as written; whether the walker should become a fixpoint walker remains a runtime question only. Original entry: `visualizer.cpp::buildStack` now carries a no-good memo: a frame that exhausts every candidate records the on-path nodes that blocked it (its *conflict set*, unioned with the conflict sets of children that failed under it), the dependencies it tested against `covered` (its *consulted set*), and which of those were already covered at the time. A later re-entry returns false immediately when every blocker is still on the path and no consulted dependency is newly covered. Both conditions are one-directional and that is what makes them sound: extra path members only remove candidates (a non-cyclic candidate can become cyclic, never the reverse), while extra covered members only remove recursions and could therefore turn failure into success — so the second condition forbids exactly that. A frame that skipped a candidate as inadmissible is never cached, because the cross-chapter usage graph and `exportCurrentChapterTheorem` both move between chapters; consulted footprints above `kNoGoodConsultedCap` are not cached either, so a probe cannot cost more than the re-walk it saves.

 **This does not yet fix the failure it was written for.** On the Peano batch's chapter export aborts on the `buildStack` 5M-call tripwire while building the `check_zero` chapter of the first exported theorem (an induction theorem on the shared multiplier of right-distributivity, whose zero block collapses most of its terms into one equivalence class). Measured in that block: 55,423 origin keys, of which 10,074 carry exactly one origin row and that row is a canonical-door `equality1` bridge; 113 walk nodes exhaust every candidate; 283 distinct nodes absorb 5,000,000 calls. An earlier, cruder variant of this memo (covered-dependence treated as a bar rather than a recorded condition) took 2,040,676 hits yet still overran the cap, with 114,867 of 190,937 failures barred from the cache by the `covered` skip. A class-sibling re-route for blocked equality bridges was also tried and measured inert: only 1,212 of 262,537 cyclic rejections are `equality1` at all, and in 1,211 of those the blocked dependency is the licensing equality rather than the bridge source. The residual blow-up is ordinary backtracking over `implication` / `disintegration` / `expansion` candidates, so the open question is whether memoization can work here at all, or whether the walker should stop backtracking and compute the derivable set as a monotone least fixpoint (origins whose dependencies are all derivable) before emitting any row.


- **OPEN-3.** The single-input operator gate — theoretical reason or empirical? See [I-6](../30_invariants.md#i-6). Likely requires a concept-level argument about Pass B admission to answer.
- **OPEN-4 — RESOLVED.** `ExpressionAnalyzer::operators` is populated at [`prover.cpp–200`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) immediately after `coreExpressionMap = ce::modifyCoreExpressionMap(anchorID)`. Loop: `for (const auto& kv: coreExpressionMap) { if (!kv.second.inputArgs.empty && !kv.second.outputArgs.empty) operators.insert(kv.first); }`. The invariant *linking it to the conjecturer's notion of "operator"* is structural: both consume the same `coreExpressionMap` from stage 1, and both use the same `inputArgs`/`outputArgs`-non-empty condition. No separate conjecturer-side operator set exists that could drift; the set is canonical.
- **OPEN-6 — RESOLVED.** `performDisintegration` does not exist in the current source tree (grep returns no matches across `GL_Quick_VS/GL_Quick/src/`). the project conventions's naming is stale — it likely refers to a historical name pre-split. The actual disintegration surface today is `ce::disintegrateImplication` in `compiler.hpp` (structural walk, no LB side effects) plus `disintegrateExpr2` in `prover.cpp` (Pass B, with admission gating). the project conventions should be updated to remove the `performDisintegration` reference.
- **OPEN-12 — RESOLVED.** Induction-variable identification happens at [`prover.cpp–5107`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) during auxiliary-implication setup. For each theorem, the prover iterates over `digitArg` values (members of the digit-args set computed by `findDigitArgs`). For each digit-arg, it creates a recursion sub-block (`tempMb2`) and records the choice: `dependencyTable.originalInductionVariableMap[originalIndex] = make_tuple(digitArg, recursionCounter)`. So: **induction is scheduled on every digit-arg as a candidate**; the base case and step case are proved for each; the `digitArg` whose triad succeeds gets promoted with `method = induction` and its name in the reference column of `globalTheoremList`. Multiple digit-args can each yield separate induction theorems from the same chain. With the induction-typing fix ([I-18](../30_invariants.md#i-18)), a third sub-block proves `(in[digitArg, N])` before promotion.

---

## See also

- [`20_core_concepts/01_logic_blocks.md`](../20_core_concepts/01_logic_blocks.md) — the LB grid, `Memory` structure.
- [`20_core_concepts/02_hash_engine.md`](../20_core_concepts/02_hash_engine.md) — hash-based inference detail.
- [`20_core_concepts/03_mail_system.md`](../20_core_concepts/03_mail_system.md) — inter-block comm.
- [`20_core_concepts/04_validity_stack.md`](../20_core_concepts/04_validity_stack.md) — scope names, NameMap.
- [`20_core_concepts/05_equivalence_classes.md`](../20_core_concepts/05_equivalence_classes.md) — equality classes + negated-equality expansion.
- [`20_core_concepts/07_or_branching.md`](../20_core_concepts/07_or_branching.md) — OR handling detail.
- [`30_invariants.md`](../30_invariants.md) — the numbered invariant register.

## Initial deposit staging

`prover.cpp` and `ExpressionAnalyzer::addTheoremToMemory` wrap every initial
expression deposit in a rule-index staging window. All expressions and goals
still pass through `addExprToMemoryBlock` in the original order. The window
batches rule-index writes produced by one compound deposit, then closes and
flushes all four hash-memory instances before registration continues. Existing
nested window depth and removal flushes remain in effect. This reuses the
compressor seeding mechanism and preserves the I-221 reader boundary.

The printed `Runtime prover` timer begins after theorem registration and grid
construction. Initial registration therefore belongs to preparation, outside
that printed interval. See `D-344`.


### Phase 3 entry-selected OR exclusion

`ExpressionAnalyzer::performElemPhase3` binds a temporary per-block root set around inference. `orBranchGoalsResolved` considers every current goal at the OR branch itself or an ancestor through main; empty and fully known sets qualify. `collectPhase3IgnoredOrRoots` also finds queued seeds, and `Memory::phase3IgnoresScope` covers their descendants without recomputing goal resolution. Mail history, convergence evidence and parent-scope processing remain. The shared tail unbinds the context after goal sanitization and before lifecycle drains; the phase-3 wrapper checks that selected live roots were wiped. See [the OR chapter](../20_core_concepts/07_or_branching.md#phase-3-inference-exclusion) and [I-232](../30_invariants.md#i-232).


### Ordered union subtree cleanup

The final `standardProcessing` wipe drain in phases 1 and 3 retains decoded-name root ordering and calls `Memory::wipeSubtrees` once. A scratch bitmap covers the union of descendants; each affected container is filtered once with stable survivors. Scope-filter mints retain first-covering-root order, with ascending identifiers inside each root. `wipeSubtree` remains the single-root interface. History, cohort bookkeeping and the saved-goal-level parent rule are unchanged; see [I-233](../30_invariants.md#i-233).


### Contradiction-case lifecycle

Active primed contradiction blocks can open and sequence admitted OR cases without artificial explicit goals. Empty relevant goal sets remain unresolved in these blocks, preventing immediate Phase 3 exclusion or freezing. A case's arbitrary registered contradiction follows the ordinary parent refutation, retirement and next-case release path; only a main/main contradiction discharges the whole block. Both refutation and convergence history use the current absorption output channel. See [I-239](../30_invariants.md#i-239).


### Inherited OR entry

**Inherited positive ORs.** Ordinary status-3 ancestor mail also enters disintegration for a positive compiled `or<N>` when `hasOrProofWork` holds at the recipient. The arrival keeps its external classification; recovered rules install in `overallHashMemory` and `workingMemory`. Normal demand, depth, duplicate and suppression checks still control case admission, and both OR parking indexes support later demand. The expansion history of each recovered rule cites its own compact implication at the rule scope, preserving the OR expansion and disintegration intermediates. No new mail flag is required. See [I-240](../30_invariants.md#i-240) and [D-367](../40_decisions.md#d-367).

## Necessity check (CNFE) and its goal rescan

`prover.cpp`, `checkNecessityForEquality`, prepares for a goal the integration of every premise of every rule chain whose head establishes the goal up to one `(1)`-typed constant, and adds the demanded variable copies. The candidates come from the masked-key family of `OriginalsHeadIndex`, whose key carries the sign: a head concluding the goal's negation is never matched ([D-415](../40_decisions.md#d-415), [I-242](../30_invariants.md#i-242)). The door calls it once per new goal against every chain, and after every implication-installing deposit (the goal rescan) for every goal against the chains no earlier rescan visited, above `HashMemory::goalScanChainMark`; a repeated (goal, chain) visit changes nothing ([D-416](../40_decisions.md#d-416), [I-272](../30_invariants.md#i-272)).

## Implication-chain composition and CNFE demand transfer

`unary_composition.cpp`, `rememberUnaryImplication` and `drainUnaryComposition`, compose actual ordered producer premise chains with unary consumers, retaining every producer premise in order. Producer heads and consumer premises join by signed operator and arity. `composeImplicationChain` matches the complete binder sets bijectively through the intermediate ports: bound-to-fixed specialization, binder collapse, unmatched binders and binders absent from the result are rejected. `normalizeCompositionChain` normalizes the complete ordered premise run and head at registration independently of composition eligibility. Fixed `u_` parameters and repeated-argument patterns remain unchanged. Matching ports retain the types of the already-typed sources. Results use the deeper of comparable scopes and the union of supporting levels. Sibling scopes never combine. Distinct scope/level support remains separate; duplicate results and identity implications add no work. Generated rules participate until closure stabilizes.

`matchUnaryAdmissionHead` is shared by composition eligibility and admission transfer. `indexUnaryCompositionDemand` enrolls only consumers whose heads match a currently installed integration demand through that predicate. All producer heads remain indexed; only one-premise rules can be consumers; the consumer-premise index contains only eligible rules. `wakeUnaryCompositionDemands` processes newly observed demands against the signed head index and queues existing consumers. New rules also inspect existing demands. Each attempted composition checks that a matching demand still exists and is comparable with the resulting source scope. Demand eligibility does not add a proof premise or alter that source scope. This backward propagation of an eligible final head discovers longer chains without materializing unrelated intermediate closures.

`standardProcessing` drains composition after equivalence-class application. Results use ordinary hash-rule installation and ordinary `implication` firing. The whole closure shares one rule-index staging window: the composition indexes expose new nodes immediately, while ordinary hash indexes flush once after the queue empties, before admission transfer or inference. The general scratch frame remains alive until that flush so staged payloads cannot be reclaimed by a candidate-loop frame. Their `implication composition` origin cites both exact source rules at their original scopes. History never controls eligibility, deduplication or lifecycle. Multiplied copies retain their actual rule scope in history.

`transferUnaryIntegrationDemands`, also reached from `checkNecessityForEquality`, handles either rule/demand arrival order. The marked witness becomes `u_marker` only in the matching view; the one remaining non-`u_` argument must have digit type `(1)`. Other argument classifications remain unchanged. The substitution must determine the complete premise and retain the marked witness. Its prefix-stripped template enters ordinary algebra admission with existing generation and secondary-variable bounds. Normal rejected-source revival then releases matching parked existences. The original integration instruction is unchanged; neither a backwards fact nor an invented integration instruction is installed.

`transferUnaryIntegrationDemands` also scans every premise of a rule whose final head matches an existing integration admission key. A positive equality must connect that same marked witness to a determined fixed argument, in either orientation. Source-fixed arguments are available at the source scope, and demand-fixed arguments at the demand scope; the copy is queued only at their deepest compatible scope. A visible canonical equality class supplies an existing second name; otherwise ordinary internal mail carries the variable-copy declaration and its empty-origin history. Receipts and mail deduplication prevent repeated requests. Every effect of the walk follows a receipt mint, so a call is skipped while neither the integration-demand key set nor the composition rule registry (`UnaryCompositionStore::rules`, whose reset also clears the receipts) has gained a key since the previous walk started, as told by their `insertEpoch` values recorded in the store; such a walk would visit receipted pairs only. Other premises remain required for firing, and the unchanged integration instruction collects the shared witness from derived facts. The old unary algebra-admission transfer remains unary. No intermediate ordinary OR-admission key is consulted for the copy decision.

`UnaryCompositionStore` retains independent roots, scoped closure nodes, level-sensitive identities, indexed joins, generated owners, consumer-demand links, observed-demand receipts and admission-transfer receipts on the block arena. The normalized premise chain is one newline-separated rule-interner value, with its count in `UnaryRuleInfo`; a one-premise chain retains its atom spelling. All twenty-two facets survive deload. `Memory::cloneFactsTemplate` deliberately starts with empty rules, demands and composition work, matching its asserted facts-only input contract; the clone registers its own rules normally. The extended source and rule records require fresh deload images; images from a different executable are not reused. Root removal or scope cleanup rebuilds derived closure from active independent roots and clears demand indexes, so cycles cannot support themselves. Removed demand receipts retire; reintroducing a demand wakes consumers again. Independent roots sharing a generated owner retain their recorded local/external routing when that owner is rebuilt. See [I-261](../30_invariants.md#i-261) and [I-262](../30_invariants.md#i-262).

### Equality support in OR bookkeeping

`materializeWatchHits` adds case-scoped requests for existing ancestor goals to its canonical matching batch. Unequal-text matches call `bookOrEqualitySupport`, retain the real source and local equalities, and feed `ordisMerge` without registering a goal twin. `orBranchGoalsResolved` recognizes that support. Frozen support remains usable by normal or reduced-count convergence until the cohort parent closes. See [I-263](../30_invariants.md#i-263).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
