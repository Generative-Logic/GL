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
| `globalTheoremList` | The authoritative list of proved theorems + their method + metadata. Written to `theorems.txt` and `global_theorem_list.txt` after the run. |
| `permanentBodies` | `vector<Memory*>` — every LB that is currently alive (target proofs, sub-goals, auxiliary blocks). Retained across iterations. |
| `permanentBodiesCE` | Same, but the CE-filter-specific grid. |
| `box` / `boxes` | Per-core mailbox arrays for the main grid. |
| `contradictionTable` | Per-conjecture contradiction status (set during CE + main prover). |
| `implCounter`, `existenceCounter`, `andCounter`, `orCounter`, `variableCounter` | Monotone counters — name-minting sources for spontaneous compact operators (`implication<N>`, `existence<N>`, `and<N>`, `or<N>`) and freshly-renamed bound variables. The first four are seeded from the loaded `GL_binary_<Tag>.json` at startup so identifiers stay stable across batches; `variableCounter` resets to zero per batch. |
| `parameters` | `ProverParameters` from `parameters.hpp` + config. |
| `globalDependencies` | Per-expression dependency set maintained during main prover runs. |
| `ceFilteringActive` | Flag distinguishing CE-filter mode from main mode; consulted by hash-request generators. |
| `ceBody`, `permanentBodiesCE` | CE-filter grid mirror of the main `body` / `permanentBodies`. (The push-model routing mirrors `indexCE` / `boxesCE` are deleted — the parent→children index went with the vestigial `ParentChildrenMap` plumbing, 2026-07-03.) |

The class is large (`prover.hpp` ≈ 5925 lines) because most per-step primitives are inline methods on `ExpressionAnalyzer`. File-level refactor is on the post-FTA plan.

---

## The execution model

A GL proof run is a sequence of **cycles**. Each cycle:

1. For each active LB, run the elementary step — produce hash requests from known expressions, look them up in the local `HashMemory`, emit new statements. `proveKernel` drives this as three barriered phase sweeps across all active LBs (see *Phase split* below), not one per-LB call.
2. After every LB has processed its local state for the cycle, drain mail: move `mailOut` contents into recipient `mailIn` buffers.
3. Next cycle: each LB processes its `mailIn`. Loop until no LB produces anything new, or a per-batch iteration cap is reached.

`prove(numberIterations, permanentBodies)` is the outer loop for the main prover and the compressor. The CE filter does NOT share it: `filterConjecturesWithCE` (`filter.cpp`) drives its own per-conjecture worker pool over `permanentBodiesCE` and never enters `prove` / `proveKernel` ([I-82](../30_invariants.md#i-82)).

The **elementary logical step** — today three phase helpers (`performElemPhase1` / `performElem2` / `performElemPhase3`) driven by `proveKernel` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), see the phase-split note below — is the inner core. It:

- Generates hash requests via the single generator `generateEncodedRequestsStatic`, called with an obligatory-stump length of 1 (mandatory single), 2 (mandatory pair) or 0 (CE mode) — depending on `ceFilteringActive` and the per-LB role.
- Matches each request against `overallHashMemory.encodedMap`.
- On a match, fires the rule — builds the conclusion instance, runs `addStatement` to install it locally, and optionally routes it upstream via `mailOut`.
- Handles OR-branch bookkeeping (`classifyOrScope`) and admission-map updates. Contradiction detection is no longer in the firing path — it moved to `dischargeContradiction`, run once per step from `standardProcessing` (see [Contradiction discharge](#contradiction-discharge)).

**Phase split + barriers (LB-split refactor, [D-113](../40_decisions.md#d-113) / [D-114](../40_decisions.md#d-114)).** The elementary step is three phase helpers — `performElemPhase1` (pre-hashburst mail absorb), `performElemPhase2` (the hashburst), `performElemPhase3` (post-burst absorb + sanitize). `proveKernel` runs them as **barriered sweeps with a join between each**: phase 1, then phase 2, then phase 3. Phases 1 and 3 are plain across-LB work-stealing sweeps (a local `runPhase` generic-lambda). Phase 2 is the **flat executor pool** ([D-115](../40_decisions.md#d-115)): one work-stealing pool over the flat list of `(LB, part)` tasks across all active LBs (an LB with `numberOfParts == N` contributes N tasks, never nested), each task running `performElem2` — the per-part request-generation + fixpoint unit — appending its firings to its own per-task `SealedPageSet` record chain, with the per-slot scratch arenas as its only other write surface ([I-83](../30_invariants.md#i-83)). After that pool joins, a per-LB finalize sweep (`performElemPhase2`) hands each LB's sealed part sets in part order to `applyFiringRecords` — a pointer-index sort over the chains' stable payload addresses ([I-77](../30_invariants.md#i-77)) — and applies them. The whole pool + finalize is wrapped in a ≤ 2-round loop (producer, then buckets — see *Statistics-driven split* below). The flat pool is what runs one heavy LB's expression buckets across N cores. The single-LB orchestrator `performElementaryLogicalStep` (entry `isActive` early-return, `RT_TRACKER_DECL`, all three phases inline) became dead once `proveKernel` called the phases directly and has been **removed** — the phase model is the only path. The hashburst dump now has two traps bracketing the live path: ENTRY at the top of phase 1, EXIT at the bottom of phase 3 (the EARLY-EXIT trap was removed once deactivation deferral made it dead — Rule 14, user-approved). The whole restructure is byte-identical at `splitCount == 1`; the byte-identity rests on [I-28](../30_invariants.md#i-28) (no cross-LB intra-cycle observation; sorted post-join drains).

**Whole-LB expression/bucket split ([D-201](../40_decisions.md#d-201)).** A straggler (see *Statistics-driven split* below), or a newly activated induction-zero LB with no prior work statistic, is split by partitioning its EXPRESSION search, not its rules. `proveKernel` dispatches it as one round-1 PRODUCER task that runs `produceExpressionStumps` on the whole LB at `g_splitCount == 1` (so `partitionAccepts` accepts every rule — the rule dimension is off): a bounded enumeration that grows the LB's base candidates level by level until it holds enough distinct ones (up to `kStumpsPerBucketTarget × logicalCores`), firing nothing. Before a short level is replaced by its children, any node accepted by the overall hash memory's minus-one or minus-two maps is retained as a **terminal pre-stump** ([D-202](../40_decisions.md#d-202)): it is dealt to exactly one bucket, rechecked by each obligatory-stump batch against that batch's actual hash memory through the normal `preEvaluateFromEncoded` → emitter → `BurstSink` path, and stops before depth-first growth. Its children remain the owners of every larger combination. `overallHashMemory` is sufficient only for conservative work discovery because every status-specific hash memory is populated from that superset; it never substitutes for the per-batch validation. The classify seam deals all stump records round-robin into `min(nStumps, logicalCores)` buckets and requeues one bucket part per bucket for round 2. The old two-cap rule→stump escalation is gone.

Every request a bucket part emits contains one of its stumps: the generator joins the stump to each growing candidate for both owner-set probes and materialises the union into a `BaseCandidate` only where the target map accepts. The statement filter is built once per bucket, and the emitter's seen-set collapses a request two of the bucket's stumps both reach. The union of all buckets is the whole search, and the sorted firing-record merge makes it byte-identical to an unsplit burst ([I-77](../30_invariants.md#i-77)). A producer that grows only one base candidate cannot fan wider than one bucket — its whole cost is a single grow-tree that expression-bucketing cannot subdivide (the `[SPLIT] ineffective` report flags this; it is the one case the rule dimension used to parallelise and this axis does not).

Sealed record sets accumulate ACROSS the iteration's rounds; an LB is finalised once, in the round after which it has no tasks left. Requeued work runs in the NEXT round, never appended to the running one — the working-set pager has registered this round's task list and dispatch cursor.

**Statistics-driven split ([D-201](../40_decisions.md#d-201)).** The split TRIGGER is not a mid-burst cap — every main-path burst runs to COMPLETION (no cap, no truncation, no discard-and-redo). After the round loop, a single-threaded stats pass in `proveKernel` sets each active main-path LB's split for the NEXT iteration from this iteration's completed work. Each LB's `work` is the SUM of its parts' submatch counts (`lbTotalSub`), the split-invariant total ([D-117](../40_decisions.md#d-117)); with `T = Σ work` and `C = logicalCores`, the static helper `isStraggler(work, T, C, min_split_work)` returns `work > T / C && work >= min_split_work` — the idle-core fair-share (an LB alone exceeding the balanced per-core load, so parallelising it uses the idle cores) gated by the per-bucket setup break-even. A straggler's `Memory::numberOfParts` is set to `logicalCores`, everything else to `1`; recomputed every iteration, so an LB whose work falls back below the bar returns to unsplit.

Classifying on the split-invariant SUM (never the max-over-parts, which scales ≈ `work / parts` and would thrash — a split LB looks light, de-splits, re-splits, the retired [D-111](../40_decisions.md#d-111) band) is the anti-thrash property: a heavy LB measures heavy whether split or not, so it holds at split. All-integer arithmetic makes the verdict — and thus the split set — a deterministic function of the deterministic work totals. `numberOfParts` persists across iterations (never reset). The **incubator** and `disable_lb_split` run UNSPLIT; the CE filter never enters `proveKernel`. A `[SPLIT] ineffective` line reports a straggler whose busiest bucket still exceeds the fair-share — an irreducibly-serial grow-tree that bucketing cannot help (pure observation, no control-flow effect); on this corpus these are exactly the recursion / induction-step LBs (`(=[N,2])_induction_recM_`, `(in2[recM,N,3])`).

The run is **deterministic** (two runs byte-identical: `theorems.txt` + verifier check total) and **preserves the theorem set** (46 theorems, Gauss fold present): the split SET is a deterministic function of the prior iteration's deterministic totals, the applied deposit is partition-independent (the sorted firing-record merge, [D-117](../40_decisions.md#d-117) / [I-77](../30_invariants.md#i-77)), and the whole-LB expression split runs at `g_splitCount == 1` so it drops no cross-term. The early-exit is gated on the per-burst `g_isMultiPart` ([I-76](../30_invariants.md#i-76)), off across a straggler's buckets, so a split does not reshape which theorems are found. It is **NOT** byte-identical to a never-split baseline at the proof-graph level (a split deactivating LB records more incidental derivations — the expected unsplit-vs-split difference), but that difference is itself deterministic.

**Firing-record capture (LB-split groundwork).** `checkLocalEncodedMemoryStatic` no longer mutates its deposit containers inline. Each firing appends a `FiringRecord` (`memory.hpp`); after the `FIXPOINT_LOOP`, `applyFiringRecords` (`memory.cpp`) sorts the records by a total content key and applies them — head deposits to `sameIterationInternalMail.statements` / `.exprOriginMap` / `.disintegrationSignals` + `canBeSentIds`, marker deposits to `deferredIntegrationPreps` / `canBeSentMarkerIds` / `admissionKeysAlgebra`. This makes the order-sensitive deposits (cap-bounded `addOrigin`, per-head `disintegrationSignals` last-write, the staged-vector drain order) a function of the firing SET, not the request-evaluation ORDER — the determinism the LB split's expression-bucket merge depends on. The phase-2 burst performs no deactivation or discharge at all — those run only in phase 3's post-burst `standardProcessing` ([I-66](../30_invariants.md#i-66)). The fixpoint also mints **nothing** into the LB's `NameMap`: the `encode` sites in `checkLocalEncodedMemoryStatic` became the non-minting `NameMap::lookup` ([D-116](../40_decisions.md#d-116)), so the whole hashburst is read-only on the shared LB — the precondition for parallel executor threads. See [D-117](../40_decisions.md#d-117) / [I-77](../30_invariants.md#i-77).

**Deferred admission tail.** The algebra-`admissionMap` writes the hashburst marker branch (`checkLocalEncodedMemoryStatic`) discovers — `admissionMap` insert, `admissionStatusMap`, `varsInAdmissionMapKeys`, and the `revisitRejected2` revival — are not applied inline. Each is staged on the per-burst buffer `Memory::admissionKeysAlgebra` (now populated by `applyFiringRecords` in canonical sorted order) and replayed once, in that order, by `drainAdmissionKeysAlgebra` immediately before the post-burst `standardProcessing`, keeping the hashburst pass free of algebra-container mutation. The drain re-applies the consumed-key gate per record so an earlier record's revival consuming a later record's key resolves deterministically within the burst. See [D-103](../40_decisions.md#d-103) / [I-68](../30_invariants.md#i-68).

**Warm-up + main iterations.** Per the project conventions, every prover run includes warm-up cycles before the main-iteration budget. Warm-up cycles "prime" the hash memory by letting rule-fire cascades reach steady state. Warm-up is the first `prove` call (`preIterations`, default 2); `analyzeExpressions` sets `warmUpPhase` around it so the quiescent-skip is disabled for warm-up bursts.

**Quiescent-burst skip ([D-194](../40_decisions.md#d-194) / [I-153](../30_invariants.md#i-153)).** `proveKernel`'s single-threaded active-build (`for (Memory* b: bodies) if (b && b->isActive) …`) excludes from the sweep every active LB whose burst would provably do nothing:

```
sweep X  ⟺  X->isActive AND
            ( warmUpPhase OR !enable_quiesce_skip OR compressor_mode
              OR X->hasWork OR mailLog.mailPeek(X) )
```

`Memory::hasWork` is a producer-side latch on the never-deloaded LB slab ([I-109](../30_invariants.md#i-109)): SLEEP clears it at `performElemPhase3` exit when the burst mutated nothing (statement count unchanged vs the phase-1 baseline, the `mutatedThisBurst` non-statement-mutation flag false, and both INTERNAL mail channels — `sameIterationInternalMail` / `nextIterationInternalMail` — empty; `mailOut` is deliberately NOT a SLEEP term — it is outgoing-only under the pull model, no burst path derives local state from it, and the commit barrier sweeps `bodies` regardless of sweep status, so pending outbound content ships whether or not the LB sweeps: the post-join drains park proven-theorem mail on the ROOT's `mailOut` after the barrier, which would otherwise permanently re-arm a genuinely idle root); WAKE sets it at every cross-LB write door (the two `updateGlobalDirect` deposits, `updateGlobal`, `drainDeferredAncestorAdmissions`, `broadcastTheorems` root self-inject, `activateZeroCondition`). `mailPeek` polls the never-deloaded mail log (`mailHeads.count > mailCursor`) for un-ingested ancestor mail, waking a converged LB the same iteration the pull would deliver ([I-55](../30_invariants.md#i-55) latency preserved). The predicate reads ONLY never-deloaded logical state — never `resident` / `blocksInUse` — so it is deterministic ([I-106](../30_invariants.md#i-106) / [I-108](../30_invariants.md#i-108)); residency is a consequence of the skip, not an input. A skipped LB stays `isActive` and keeps its claim/pager state untouched, so the pager's sweep window never includes it and it evicts and stays evicted at 4 GiB (the motivating win). The loop is cap-driven, so the skip changes neither the iteration count nor which iteration a fact appears in — correctness reduces to "each skipped burst is individually a no-op," validated by the `QUIESCE_SHADOW_CHECK` compile flag (sweep would-be-skipped LBs anyway and assert their burst was empty). The [I-48](../30_invariants.md#i-48) deactivation survey (`deactivateRecursively`) is a tree-wide post-join walk reading only never-deloaded state, so it still surveys skipped LBs on schedule with no reload. Telemetry: the `dt` line prints `swept=` (post-skip) and `skipped=`. Gated by `parameters.enable_quiesce_skip`.

**Runtime measurement.** A permanent, off-by-default instrumentation layer brackets the elementary step's natural phases — now spread across the phase helpers `performElemPhase1/2/3` + `performElem2` — with RAII `RT_SCOPE_HERE` calls that attribute to a per-call tracker via the thread-local `g_currentThreadTracker`. When the compile-time gate `RT_MEASUREMENT` is `1` *and* a single call exceeds `RT_TIME_TRIGGER_SECONDS`, the tracker writes a live timing table to `.rt/<sanitized-LB-chain>.log`, refreshing online as the call proceeds. **Known limitation (LB-split barriers, [D-114](../40_decisions.md#d-114)):** `RT_TRACKER_DECL`'s only call site was in `performElementaryLogicalStep`, now removed, so no tracker is declared on the live path, `g_currentThreadTracker` is never set, and the scopes no-op — RT currently produces no output until the tracker is re-homed to span the three barriered sweeps. RT is compile-time-off by default, so the determinism gate is unaffected. Full design + open question: [`_meta/rt_measurement.md`](../_meta/rt_measurement.md). Separate from the hashburst dump under the project conventions — different file, different gate, different code path.

**Single-pass head-switch model.** `analyzeExpressions` runs exactly **one** prove pass per batch. The contrapositive (head-switched) mirrors of every conjecture that structurally qualifies are folded into the conjecture pool **before** that single prove call, so the prover gets the same proof opportunities in one pass that the legacy multi-iteration path reached across many. Immediately after the prove pass, the same `headSwitchOne` walk runs once more on `globalTheoremList` to populate the class member `orPairsFromHeadSwitch` — `run_modes.cpp` consumes those pairs to construct OR theorems (with disjunct-set dedup and parent-removal at construction time per [D-55](../40_decisions.md#d-55); full mechanics in [`20_core_concepts/07_or_branching.md`](../20_core_concepts/07_or_branching.md#or-theorem-construction-run_modescppfullrun)). The compressor is invoked from `run_modes.cpp::fullRun` *after* `analyzeExpressions` returns (gated by `parameters.skipCompression`) — see [D-25](../40_decisions.md#d-25); it is no longer part of `analyzeExpressions` itself. There is no grid teardown / rebuild / re-prove; the multi-iteration loop and its `bigIteration` / `maxBigIterations` / `pre_emit_head_switch` machinery were deleted at [D-24](../40_decisions.md#d-24).

**Contrapositive construction.** `headSwitchOne` ([`prover.cpp+`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) is purely structural — it takes the implication string, scans the chain for a `prem[0] == '!'` premise with empty inner bound-vars (the rightmost such premise wins), swaps the head with that premise's body, and rebuilds the implication via direct string concatenation. The same helper backs both the pre-emit pass and the post-prove walk; output is byte-identical between the two call sites.

---

## Theorem load path

Every theorem that enters the prover must first be rewritten by `precompileStructuralOperators` (see [I-1](../30_invariants.md#i-1)). There are three distinct load paths:

### 1. External theorems

Loaded at the start of `analyzeExpressions`. Contents of `files/theorems/compressed_external_theorems.txt` are read line by line; each is precompiled, then passed to `addTheoremToMemory`.

In a multi-batch pipeline, theorems proved in an earlier batch are loaded here for the next batch. Example: the branch's `run_modes.cpp` hard-codes a specific Peano theorem to *remove* from the loaded `proved_set` during the Gauss batch, as a mitigation for the Gauss fold/limitSequence cascade regression — visible at [`run_modes.cpp–150`](../../GL_Quick_VS/GL_Quick/src/run_modes.cpp).

### 2. Conjectures to be proved

After CE filtering, each surviving conjecture is precompiled and registered as a proof target via `addTheoremToMemory(conj, this->body, 0, false, globalDependencies)`. The `false` argument indicates "unproved — this is a goal, not a given".

### 3. Broadcast / mail

When an LB derives a theorem that becomes available to siblings, the broadcast path forwards the theorem via `mailOut.statements`. Recipients invoke `addStatement` on receipt, which internally handles re-precompilation if needed.

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
2. **Shape dispatch + `registered`-bit registration.** If `isEquality(expr)`, call `addEquality` (registers expression + mirror into `intStatementLevelsMap`, `intKnownStatements`, `intEncodedStatements`, and the local-delta channels when `local`). If `isNegatedEquality(expr)`, call `addNegatedEquality` (same registration pattern, mirror form `!(=[b,a])`). Then upsert the statement's packed key into `intKnownStatements` with the `registered` bit (the former `wholeExpressions` membership per [D-128](../40_decisions.md#d-128)). Moved here from the caller-side disintegration-product loop — `addStatement` has exactly one call site and only statuses 0/1/3 reach it (status 2 returns at `addExprToMemoryBlock` early, status 4 even earlier), so no gate is needed. The two helpers are private members; this is the single dispatch entry. The `allowSymmetry` parameter the helpers previously carried has been retired; the mirror block inside each helper now runs unconditionally.
3. **Filter gates.** Max-iteration and count-pattern early-exits.
4. **Non-equality deposit / negated-equality expansion.** For non-equality shapes (or `skip_eq_classes`), deposit into `intStatementLevelsMap`, `intKnownStatements`, `intEncodedStatements` and (when `local`) the local-delta channels. For `!(=[a,b])`, additionally call `applyEquivalenceClassToNegatedEquality` to emit one-sided sibling inequalities ([I-12](../30_invariants.md#i-12)).
5. **Equality branch.** For `(=[a,b])` (and not `skip_eq_classes`), call `updateEquivalenceClasses` (class merges and rewrites) and push the equality itself onto `newStatements`.
6. **Return** `newStatements` for the post-loop to consume. `mailOut.statements` is no longer populated from here — `fillMailOut` is the single writer (centralisation landed earlier on this branch).

**Return type — `std::vector<ExpressionWithValidity>` ([D-33](../40_decisions.md#d-33), [I-25](../30_invariants.md#i-25)).** Each entry in `newStatements` is an `ExpressionWithValidity` pair carrying the deposit's actual scope. Same-scope deposits, descendant-scope deposits (D-33 new direction — class deeper than the incoming expression), and ancestor-scope rewrites all flow through this single channel. There is no separate "side sink" for cross-scope deposits.

The kernel's post-`addStatement` loop in [`addExprToMemoryBlockKernel`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) iterates the returned pairs and uses each entry's own `validityName` (the local `effectiveValidity`) for:

- `intStatementLevelsMap` lookup (the non-minting `lookupStatementLevels`) — the same-scope assert at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) holds because the lookup is keyed by the deposit's actual scope.
- `updateAdmissionMapIntegration` / `updateAdmissionMapRecursion` calls.
- `toBeProved` discharge for recursion-LB head match, direct-theorem proof, and validity-name promotion (OR-integration / NotOrScope).

This routing closed the cross-scope discharge gap that the WIP D-33 commit had introduced — see [D-33](../40_decisions.md#d-33) for the incident and the fix.

The asserted precondition on negated-equality entry is `isNegatedEquality(expr)` — callers must gate on the same check. See [`applyEquivalenceClassToNegatedEquality`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) at line 4878.

### `addTheoremToMemory`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Takes a theorem string and a target memory block; registers the theorem as an encoded statement, builds its disintegration chain, and creates auxiliary implications via `createAuxyImplication`. This is the path for loading an established theorem (external or previously-proven) as an inference rule.

### `addToHashMemory`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Stores a normalised implication chain in `HashMemory::encodedMap`, along with its `LocalMemoryValue` metadata (`value`, `levels`, `originalImplication`, `justification`, `key`, `remainingArgs`, `validityName`, and — D-32 — `productOfDisintegration`). This is the actual installation of an implication rule in the hash engine.

`encodedMap` is populated from **two distinct insertion sites**, both reached from this function:

1. **Head-implication insert** at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) — the rule registry entry. The LMV's `.value` is the head's normalised template; the LMV carries full provenance (`originalImplication`, `justification`, `validityName`). This is the LMV that fires when an LB hash-matches the chain. **`productOfDisintegration` is stamped here**, true iff at least one premise has an arg starting with `"u_"`.
2. **Marker LMV insert** at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), inside `makeNormalizedKeysForAdmission`. The LMV's `.value` is a marker-form expression (the original premise with its output arg replaced by the literal token `"marker"`); provenance fields default-empty. This LMV fires during Pass B's `int_` admission probe via `isAdmittedIntegration`. `productOfDisintegration` defaults to `false` here — marker LMVs never drive OR-disintegration.

When adding metadata to `LocalMemoryValue`, choose explicitly which insertion site receives it. See [`20_core_concepts/02_hash_engine.md`](../20_core_concepts/02_hash_engine.md#two-encodedmap-insertion-sites-head-lmvs-vs-marker-lmvs) for the full discussion.

After installation, `multiplyImplication` is invoked to generate partition-based equalisation copies (see below).

### `createAuxyImplication`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Generates auxiliary implications for recursion: separates digit arguments from immutable arguments (via `findDigitArgs` / `findImmutableArgs`), wraps the implication in rec-chain machinery, and registers the result. Called by `addTheoremToMemory` as part of theorem installation.

### `precompileStructuralOperators`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). In-place rewrite: replaces every raw `!(&...)` with the compiled `or<N>` name and every `!(>...)` with the compiled `existence<N>` name, recursively. Produces a string that contains no uncompiled structural operators.

Invariant [I-1](../30_invariants.md#i-1): **every theorem-load path must call this**. The assert that blows up inside disintegration when this is skipped is the symptom.

Note `precompileStructuralOperators` compiles only the **negated** structural forms (`!(&...)` → `or<N>`, `!(>...)` → `existence<N>`). A plain top-level `(>[])` implication is never its input and is therefore never compiled to `implication<N>` on a theorem-load path. The `implication<N>` name is allocated only for implications that appear as *subexpressions* of a compiled definition body, plus — on this branch — the mail-broadcast compaction wrapper below.

### `compileImplicationToCompact`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) (member of `ExpressionAnalyzer`). Compiles an implication that enters the mail "implications" channel into its compact `(implication<N>[args])` form, additionally deposited as a mail expression (see [`20_core_concepts/03_mail_system.md`](../20_core_concepts/03_mail_system.md)). **It is invoked only from the single-threaded post-`pool.join` deferred-compaction drain, never inline at the eight broadcast sites** — those call `recordPendingCompaction` to enqueue the implication; calling it inline raced global `implCounter` / `compiledExpressions` / `repetitionExclusionMap` on the parallel `addExprToMemoryBlock`-contradiction path ([D-76](../40_decisions.md#d-76), [I-28](../30_invariants.md#i-28)). It does **not** introduce a new compile path: it normalises the implication's free arguments via `stripUPrefixAST` (a leading `u_` removed from every argument token at every nesting level — unlike `removeUPrefixFromArguments`, which only sees the first `[...]`) and then calls the existing `compileCoreExpressionMapCore`. For a top-level non-negated implication that function reaches only its `(>` branch and returns the compact instance. A fully-bound proved theorem (every variable bound — [D-75](../40_decisions.md#d-75)) has zero free arguments and compiles to `(implication<N>[])`; an implication with genuinely-free arguments compiles to `(implication<N>[<freeArgs>])`. Per the project conventions there is no defensive fallback: a mailed implication that fails to compile to a simple `(implication<N>[...])` asserts at its origin rather than returning an empty/sentinel string.

Compiling a *multi-premise* theorem recurses its inner `>` into its own `implication<N>`, so the outer implication-entry's head becomes an `implication<N>`; cross-batch shared-registry reuse likewise makes `implication<N>` a constituent of hand-defined `and` operators (`fXY`, `identity`). The historical `checkCompiledCoreExpressionMap` invariant forbidding "an implication as a constituent of another compiled expression" was **retired per explicit user decision** ([D-76](../40_decisions.md#d-76)): the `DeepChecker` implication branch now accepts a nested implication as a recursion-stop leaf instead of asserting. The unrelated "core not found" guard in the same walk is preserved.

> **🔴 ROOT-CAUSED + FIXED (2026-05-17).** The non-determinism was a data race, not a key flaw: `updateGlobalDirect` is also called from the parallel `addExprToMemoryBlock(... coreId...)` contradiction path, so inline `compileImplicationToCompact` mutated global `implCounter` / `compiledExpressions` / `repetitionExclusionMap` unsynchronized (violates [I-28](../30_invariants.md#i-28)) — proven by torn diagnostic-trace lines and run-to-run `implication<N>` drift (1359 vs 1361) on a byte-identical `theorems.txt`. Fix (Option A, user-approved): the eight sites enqueue via `recordPendingCompaction`; a single-threaded sorted pass after `pool.join` performs the compile + deposit, flushed per core via `sendMail` — deterministic and injective. The `excludeRepetitions` zero-arg signature fix and the retired nested-implication assert are unaffected. The separate ~22 GB incubator memory-expansion concern is unchanged in volume (compaction deferred, not added) and did not reproduce in the two verification runs. See [D-76](../40_decisions.md#d-76) and [G-45](../50_gotchas.md#g-45).

### `findDigitArgs` / `findImmutableArgs`

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) and [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Extract the "flowing" versus "pinned" arguments of an implication chain. Used by auxiliary-implication construction and by `updateAdmissionMap`.

**Verifier parity.** The Python verifier replicates these algorithms — see `_find_digit_args` and `_find_immutable_args` in `verifier.py`. A divergence between C++ and Python is a sound-ness red flag; [I-16](../30_invariants.md#i-16) applies.

### `reconstructImplication` / `reconstructImplicationFullBind`

Defined at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Rebuild an implication from a chain + head. There is now **one** binder rule (see [I-4](../30_invariants.md#i-4)): bind every variable whose name does not start with `u_`, at the left-most premise that mentions it. `reconstructImplicationFullBind` is the single implementation; `reconstructImplication` is a thin forwarder to it, kept so the historical theorem-level call sites and the I-4/I-5 citation symbols stay grep-stable. The earlier theorem-vs-rest split (the retired sparse "bind only ≥2× occurrences" rule, old [I-5](../30_invariants.md#i-5)) is gone. A theorem carries no `u_` args, so every variable in a theorem — including every anchor slot — is bound.

**`reformulateTheorem` interaction.** Its peeling-layer trigger used to read the *reconstructed* last-link `>[...]` cardinality (`boundVars.size == 1`). Under the unified binder that cardinality no longer encodes the condition, so the trigger is re-expressed to compute the same predicate directly from the original chain — the target Definition's set-argument is the single non-`u_` arg that occurs ≥2× across (all premises + head) and appears in no other premise. The same reformulations fire; only the emitted theorem's outer `>[...]` widen. The inner negated-existence still binds exactly the peeled set-argument. See [G-44](../50_gotchas.md#g-44) and [D-75](../40_decisions.md#d-75).

### `addExprToMemoryBlock` — `status` parameter reference

`addExprToMemoryBlock` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) is the central deposit primitive called from dozens of sites. Its 4th argument `int status` is a mode selector that controls which deposit path runs. The values are not documented inline at the function site — callers learn them by copy-paste from neighbouring call sites, which is error-prone. Reference table:

| `status` | Path | `forceDeep` arg | Disintegration? | Origin tracked? | Typical caller |
|---|---|---|---|---|---|
| `0` | Local derivation, force-deep disintegrate | `true` | yes | yes | Pass B deferred emissions, OR branch seeds, CE filter admission |
| `1` | Local derivation, normal disintegrate | `false` | yes | yes | Canonical "produced and absorbed by LB" — `addStatement` main branch (`:1975`), fixpoint deposits (`:4648`), `addToHashMemory` head writes (`:4956`), **integration-revival absorb** (new — from `sameIterationInternalMail`) |
| `2` | Goal / `toBeProved` insert | n/a | skipped — `prepareIntegration` runs instead | yes (goal-origin) | New head registered for proving; OR branch head emission |
| `3` | **External-mail absorb (ASIC 0.1 reshuffle).** Enters `disintegrateExpr2` to recover the implication / fact from the compact mail form. The earlier branch experiment of running with `allowExistenceDisintegration=false` was reverted — the parameter was removed and `disintegrateExpr2` / `disintegrateExprCore2` are now byte-identical to main HEAD's control flow. Recovered implications are status-distributed: `overallHashMemory` (full visibility) **plus** the per-burst `workingMemory` (the same-burst Batch-1 source) — and explicitly **NOT** `localHashMemory`/`localHashMemoryDelta` (those are local-impl-only; an external rule there would be mis-treated as a local delta by Batch 5). Recovered facts go through the kernel and are additionally staged into `externalStatements`/`intExternalStatements` for the same burst's mail-pair batches. Runs in the **pre-fixpoint mail-absorption block** (before request generation), unified across CE and normal mode. The post-fixpoint placement introduced by commit was reverted on 2026-05-20 ([D-79](../40_decisions.md#d-79)) because it pushed `__contradiction__(=[a,b])` LBs past `MAX_NAME_IDS` during Peano-incube combinatorial substitution. | `false` | yes (origin from `mailIn.exprOriginMap`) | The pre-fixpoint `mailIn.statements` drain in `performElementaryLogicalStep`. Replaced the rejected status=5 mode. |
| `4` | Fast-path fact load | n/a | skipped (no gate, direct push) | no (levels only) | Simple-facts bootstrap `prover.cpp` |

**Choosing the right status.** Ask: has this expression ALREADY been disintegrated by whoever produced it?

- If the producer is another LB's `mailOut → smashMail → mailIn` routing: use `3` — post-ASIC-0.1 the receiver re-derives via `disintegrateExpr2` but with existence disintegration banned (no fresh `it_`/`int_`); recovered implications install through the persistent 3-way path.
- If the producer is *this* LB's own body (eq-class rewrite, OR-branch seed, revival absorb): no, or incomplete. Use `1` (normal) or `0` (force deep).
- If inserting a theorem to be proved (vs asserted), use `2` — `prepareIntegration` runs to set up integration rules.
- If reloading simple facts at grid setup, use `4` — no semantic processing needed.

**Silent-failure mode (historical).** Before the ASIC 0.1 reshuffle, `status=3` skipped `disintegrateExpr2` entirely (a `status!= 3` gate). Using it where `1` was needed (e.g. for revival constituents) deposited the expression into `intEncodedStatements` and `intKnownStatements` but ran no disintegration — Pass B admission never fired, `int_` vars never minted, no integration progressed; the statement "arrived" but was inert. That is why the integration-revival channel uses `status=1` ([D-19](../40_decisions.md#d-19)). Post-reshuffle `status=3` *does* enter `disintegrateExpr2` (existence disintegration banned), so the "inert deposit" mode no longer exists for it; the remaining contrast with `status=1` is purely the existence-witness ban (status 3 mints no fresh `it_`/`int_`).

### Admission / rejection-map key shapes

There are FIVE maps in `HashMemory` that key on variations of a marker-form expression. They are NOT interchangeable — each uses a different prefix convention, and a lookup using the wrong form silently misses. Reference table:

| Map | Field in `HashMemory` | Key shape | Produced at |
|---|---|---|---|
| `admissionMap` (algebra) | `.admissionMap` | `u_`-prefixed on all non-marker args — e.g. `(in3[u_a, marker, u_b, u_+])` | `prover.hpp` (in `updateAdmissionMap` path) |
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

Rule of thumb: think of `u_` as the "universal template" prefix and the bare form as the "concrete runtime instance." Admission MAPS (map + mapIntegration) store templates (`u_` everywhere). Admission SET (`admissionSetIntegration`) and rejection MAPS (both) store concrete instances (no `u_`). Always transform at the boundary.

---

## Disintegration

Disintegration is the prover's "expand compound expressions into their parts" operation. Two paths:

### Pass A — structural disintegration

The standard path: when an `and` or `existence` node is received, split it into its element expressions via `disintegrateExprCore2` and recurse. Straightforward.

### Pass B — operator-aware disintegration

Defined at [`disintegrateExpr2`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) (`prover.cpp`). Produces sets of new statements and integration expressions with iteration tracking, plus a 4th return element `fullDisintegrationHappened` — true iff the compound entered disintegration and every existence inside it got at least one admitted witness, or it had no existence at all. The `addExprToMemoryBlock` call site records that flag as `StatementFlags::fullyDisintegrated` on the expression's `wholeExpressions` / `intKnownStatements` entry (see [I-72](../30_invariants.md#i-72)).

**Entry gate (`addExprToMemoryBlock`).** `disintegrateExpr2` is called only when `!doNotDisintegrate && !checkForEquivalence(expr, validityName, memoryBlock) && (status!= 3 || isCompactImplication)`. The `checkForEquivalence` term skips disintegration when an equivalence-class variant of the expression is already registered in `wholeExpressions` for the same validity scope **and flagged `fullyDisintegrated`** — not on mere presence and not on locality (a local-but-rejecting twin must not suppress the canonical; see [I-72](../30_invariants.md#i-72)). [D-88](../40_decisions.md#d-88) records the previous removal and its Gauss-fold motivation; this branch reintroduces the gate keyed on full disintegration ([D-107](../40_decisions.md#d-107), [D-108](../40_decisions.md#d-108)).

Central complexity: freshly-minted `it_…` variables must be *admitted* before they enter `finalStringStatements`. Two admission paths compete:

1. **Map-based admission** — `isAdmitted` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Consults `admissionMap` (populated by `updateAdmissionMap` from consumer-side registrations). Authoritative; failure routes to `pendingRejections`. A freshly-minted `it_…` is admitted only if its max iteration number is `<= maxAdmissionDepth` **and** its constituent-form secondary-variable count (`countPatternOccurrences`, which skips `productsOfRecursionIds` members) is `<= maxNumberSecondaryVariables`. The cnt cap uses `maxNumberSecondaryVariables` — **not** the per-slot `standardMaxSecondaryNumber` — so it stays symmetric with the marked-form cnt sites (the `isAllowedAsOperatorInput` guard, the `applyEquivalenceClass` growth cap, `revisitRejected2`), which scan the marked form where the output slot is the literal `marker` and thus count one fewer. See [D-89](../40_decisions.md#d-89).
2. **Single-input-operator standalone** — `isAllowedAsOperatorInput` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp). Fallback; fires only for `reIt`-matching variables. Admits iff:
 - `extractExpression(stmt)` is in `ExpressionAnalyzer::operators`;
 - `cfg.inputIndices.size == 1` — **single-input operators only** (in practice `in`, `in2`);
 - The lone input-arg position holds the variable;
 - Standard guards pass (`maxIterationNumberVariable`, `maxNumberSecondaryVariables`).

First-match-wins — the outer loop `break`s as soon as one path admits. The standalone rule does not consult the admission map at all. This is the guardrail against RT explosion: widening to multi-input operators broke Gauss summation. See [I-6](../30_invariants.md#i-6).

**The `int_…` path** has two admission checks: map-based `isAdmittedIntegration` against `admissionMapIntegration`, then `admissionSetIntegration` fallback. Porting the `isAllowedAsOperatorInput` single-input rule into the `int_` branch is forbidden without careful RT measurement.

### `int_` deferred-rejection buffer (`rejectedMapIntegration`)

As of commit landing [D-19](../40_decisions.md#d-19), the `int_` branch also has a deferred-rejection buffer symmetric to the `it_` branch's `pendingRejections → rejectedMap` flow. When a fresh `int_` variable fails both `isAdmittedIntegration` and `admissionSetIntegration`, the rejection is NOT silently dropped — it is buffered into `pendingRejectionsIntegration` and, if the cascade-admission loop still leaves the var un-admitted, committed to `HashMemory::rejectedMapIntegration` via `updateRejectedMapIntegration`.

Shape:

| Field | Role |
|---|---|
| `rejectedMapIntegration` key | `ExpressionWithValidity(markedConstituent, validityName)` — same shape as algebra's `rejectedMap` key, keyed on the marker form (`int_var → "marker"`) of the body element. Skipped for `(in[...])` typing elements per user spec (they are never a key but still carried in siblings). |
| `rejectedMapIntegration` value (`RejectedMapIntegrationValue`) | `concreteConstituent` (body element with `int_` arg intact) + `siblings` (all OTHER body elements of the compound, incl. `(in[...])` typing) + `compoundExpression` (for origin). No `iteration` field — `int_` mint is level+startInt only. |

Revival paths (once stored):

- **Equivalence-class rewrite.** `applyEquivalenceClassToRejectedMapIntegration` ([eq-classes chapter](../20_core_concepts/05_equivalence_classes.md#extension--rejectedmapintegration)) rewrites keys when the class touches their args; on matching admission, emits rewritten constituents to `sameIterationInternalMail`. Uses the full `allMappingsAna` permutation loop (same machinery as the main helper), with per-entry dedup across mappings that collapse to the same rewritten key.
- **New-admission-key trigger.** `revisitRejectedIntegration2` fires at `admissionMapIntegration` insert (`prover.hpp`, with u_-strip — see [G-32](../50_gotchas.md#g-32)) and at `admissionSetIntegration` insert (`prover.cpp`, bare-form). Emits matching `rejectedMapIntegration` entries to `sameIterationInternalMail`.

Both revival paths deposit to `sameIterationInternalMail` only — no `addExprToMemoryBlock` call. See [D-19](../40_decisions.md#d-19) for why (cyclic-re-entry avoidance). The `sameIterationInternalMail` absorb at the top of next hashburst uses `status=1` (full disintegration pipeline), not the legacy mail's `status=3`.

Asymmetries with algebra-side:

- Integration revival does **NOT** call `cleanAdmissionMap` on the admission-map entry after successful revival (see [I-22](../30_invariants.md#i-22)). The admission rule stays live for future matching rejections.
- Integration revival does NOT cascade through `addExprToMemoryBlock`. Linear path only.

### Disintegration gate

Pass B fires when `!parameters.compressor_mode && !parameters.ban_disintegration` (see [I-7](../30_invariants.md#i-7)). `ban_disintegration` also gates back-reformulation, hypothetical disintegration, and necessity-for-equality-hypo — every disintegration-shaped path in the prover. Pre-2026-04-29 the gate was `!parameters.incubator_mode`; that decoupling-from-`incubator_mode` is the lasting change ([D-27](../40_decisions.md#d-27)). A short-lived `allow_disintegration` flag was introduced and then collapsed into `ban_disintegration` later the same day ([D-28](../40_decisions.md#d-28)) once the per-config matrix turned out to be symmetric (`ban_disintegration == !allow_disintegration` in every config). `ConfigIncubatorGauss1.json`'s SE2-migration combination — Pass B on, `multiplyImplication` off, `incubator_mode` on — sets `ban_disintegration=false` to fire Pass B.

### `savedStartInt` freshness

Pass B's freshness check uses `savedStartInt` — a monotonically-increasing counter. A parallel counter breaks the contract; see [I-17](../30_invariants.md#i-17).

### `performDisintegration`

the project conventions lists `performDisintegration` as a key function; the current source search returned "not found" (see OPEN-6 in [`AGENT_SwDD.md`](../AGENT_SwDD.md#open-questions)). The actual current code has `disintegrateExpr2` plus a set of related helpers; `performDisintegration` may be a historical name. Requires clarification on the next expansion pass.

---

## Integration

Integration is the mirror of disintegration: reassemble a compound expression from its constituents so it can be asserted as a conclusion.

Three preparatory tags appear in the processed proof graph:

- `reformulation for integration and` — for `and`-category conclusions.
- `reformulation for integration >[bound]` — for `existence`-category conclusions whose outermost `>[...]` bound-variable list is non-empty; the bound variable is carried through.
- `reformulation for integration >[]` — for `existence`-category conclusions whose outermost `>[...]` list was stripped because the witness slot was already occupied by an external value.

Helpers involved:

- `expandSignatureForIntegration` — uses `reconstructImplicationFullBind` (see [I-4](../30_invariants.md#i-4)).
- `buildIntegrationInstruction` — ditto.
- `updateAdmissionMap` — consumer-side output-slot registration, feeding the disintegration admission map.

Each integration step emits a `validity name` tag row — declaring the scope into which the integrated expression is bound — and then an `expansion for integration` tag row showing the re-expansion into compiled form.

---

## `multiplyImplication` — Bell-partition equalisation

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Invoked by `addToHashMemory` on every newly-installed implication.

**What it does.** Collects every variable that appears at a `(1)`-typed argument position anywhere in the rule (both bound and free `u_*` anchor parameters), then enumerates Bell partitions of that set. Each partition defines one "equalisation copy" where the variables in the same class are forced equal simultaneously. Each copy is installed as its own rule — broadening the implication's applicability.

**Example.** An implication `(>[v1,v2,v3](P(v1,v2,v3)))` with three bound `(1)`-typed variables has the Bell partitions:

- `{{v1},{v2},{v3}}` — no equalisations.
- `{{v1,v2},{v3}}`, `{{v1,v3},{v2}}`, `{{v2,v3},{v1}}` — one pair equalised.
- `{{v1,v2,v3}}` — all three equal.

For each partition except the original, `multiplyImplication` produces the equalised copy, passes it through `precompileStructuralOperators` (if needed), and registers it as a separate rule.

**Why.** Some theorems hold only when two or more bound variables are identified; `multiplyImplication` surfaces those cases automatically without requiring the conjecturer to have enumerated them.

**Soundness gate (no double-`u_` equivalence classes).** A partition is *skipped* whenever any equivalence class would contain two or more distinct free `u_*` anchor parameters. Free anchors are pre-bound to specific elements of the underlying definition sets — equating two distinct ones would silently rewrite a free slot of the rule body, producing a logically stronger rule than the source. Bound-and-free merging (one `u_*` together with one or more bound vars) is still permitted: that is the documented mechanism for specialising a rule's bound variables to a known constant.

Emitted chapters carry the `multiplied from` tag (see [Proof tags](../20_core_concepts/08_proof_tags.md)).

---

## Induction

Induction is scheduled on a bound variable `n` identified structurally (currently: `(1)`-typed bound variables that appear in certain canonical positions). The prover creates sub-goal LBs for:

1. **Base case** — substitute `i0 = 0` for `n`; prove the head under the substituted premises.
2. **Step case** — assume the head for `n`; prove the head for `s(n)`.

Promotion of an induction theorem to `globalTheoremList` uses `method = induction` with the induction variable recorded in the reference column. The registered string is the FullBind-canonical reconstruction (`reconstructImplicationFullBind(ky, value)`) of the proven implication, **not** the as-scheduled `expr` string carried in `dependencyTable.originalAuxyMap[…].expr`. The conjecturer / scheduler may emit a theorem whose outer `>[…]` binds only the anchor-slot names that appear in the body (e.g. `(>[N,i0,s](AnchorPeano[N,i0,s,+,*,i1])(…))` for `existence3`); the registry must store the I-4 form (`(>[N,i0,s,+,*,i1](…))`) because the chapter's `implication`-tag citation is also FullBind, reconstructed from the GL binary by the deferred-compaction drain. See [D-80](../40_decisions.md#d-80).

### The typing gap (current debug focus)

See [I-18](../30_invariants.md#i-18) and [`docs/agentic_swdd/induction_typing_plan.md`](../induction_typing_plan.md). The historical prover scheduled induction on any structurally-typed bound variable without first verifying `(in[n, N])`. For bound variables appearing only in negations, bare equalities, or existence heads, this is unsound — the theorem ranges over all entities, not just over `N`.

The fix is in progress on the current branch:

- A third recursion sub-block `tempMb3` is created at induction setup, with `(in[digitArg, anchor_args[0]])` as its head.
- `parameters.typingProofOnly = true` during the typing proof, forbidding induction re-entry.
- If the typing sub-block does not complete successfully, the induction is silently rejected (no `globalTheoremList` insertion).
- A chapter file `<N>_induction_typing.txt` is emitted with all the usual tags (direct-proof).
- The verifier adds a new `induction typing` checker that walks every `method = induction` row and verifies its accompanying typing chapter.

See the plan file for the full staged implementation.

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
- `ordisMerge` (inline member in [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp), called from `addExprToMemoryBlockKernel`'s post-`addStatement` loop) — `_ordis_` convergence detection, per-branch cleanup of the converged expression, parent-scope promotion via `sameIterationInternalMail`. See [D-34](../40_decisions.md#d-34).
- `cleanUpOrIntegrationBranches` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) — wipes sibling `_orint_` branches' statements when one branch proves the OR-introduction goal's head.

OR disintegration creates a child LB per disjunct, with the other disjuncts' negations seeded as branch-local assumptions. OR convergence fires when every child has independently reached the same conclusion — the conclusion is then promoted to the parent scope.

The OR-disintegration admission gate at `disintegrateExprCore2`'s OR case (per [D-32](../40_decisions.md#d-32)) reads a threaded `allowOrDisintegration` parameter — set by `checkLocalEncodedMemoryStatic` only when the firing implication is a product of disintegration (`LocalMemoryValue::productOfDisintegration`, stamped at install time in `addToHashMemory` based on the presence of a `u_*` arg in any premise). Coupled with `addExprToMemoryBlock`'s `doNotDisintegrate` so OR-disint cannot fire when general disint is forbidden. See [`20_core_concepts/07_or_branching.md`](../20_core_concepts/07_or_branching.md) for the full walkthrough.

### Contradiction discharge

`dischargeContradiction` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)), run once per elementary step from `standardProcessing` directly before `dischargeToBeProved`, handles the **incubator**, **vacuous-truth**, and **CE-filter** cases — whose fired heads all enter `intEncodedStatements`. It sweeps the LB's whole `intEncodedStatements` set; for the first statement whose negation is already proved at an ancestor scope (self included — the `nameMap.ancestorsOf` + `intKnownStatements` scan), it fires one of three mutually-exclusive reactions and deactivates the LB:

- **Incubator** (`primedForContradiction`) — records a `contradiction` origin for `!`+cleanOp and broadcasts the stored contradiction theorem (`contradictionTheoremId`, decoded) via `updateGlobalDirect`. Per [D-51](../40_decisions.md#d-51) the record stays local to the `__contradiction__` LB.
- **Vacuous truth** (`isPartOfRecursion` at validity `main`) — records a `vacuous truth` origin (deps: expr, negation, the decoded `recursionHypothesisId`) and deposits the `toBeProved` head via `addStatement`, so the immediately-following `dischargeToBeProved` (which snapshots `intLocalEncodedStatementsDelta` at entry) closes the induction goal the same step.
- **CE filter** (`contradictionIndex >= 0`) — marks `contradictionTable[contradictionIndex].successful` and deactivates the CE LB; the refuted conjecture is dropped. CE clones are never primed / recursion, so this is the only reaction they ever take.

CE detection used to stay at the `addExprToMemoryBlock` insertion gate (CE fired heads were discarded, never entering `intEncodedStatements`). The CE-filter RT-opt ([D-118](../40_decisions.md#d-118)) rebuilt CE filtering as a per-conjecture thread pool where each CE LB is single-owner, so its fired heads can safely enter `intEncodedStatements` and the gate is gone — CE joins incubator + vacuous-truth in the sweep, and `burstDeactivates` stops the burst the instant the refuting head fires.

The sweep replaces the incubator + vacuous halves of the former per-insertion "Site G" block inside `addExprToMemoryBlock`. Site G fired only at insertion time on the incoming expression, so it missed a contradiction when the ancestor-scope negation arrived *after* the deeper positive (neither insertion's scan saw the other); the whole-set sweep is order-independent. It fires at most once per call and is a no-op on an already-inactive LB, so the phase-1/phase-3 double-run of `standardProcessing` cannot double-fire ([I-78](../30_invariants.md#i-78)).

Memory: `primedForContradiction` keeps contradicted LBs alive in `deactivateRecursively` so the discharge can fire; `contradictionTheoremId` and `recursionHypothesisId` carry the per-LB NameMap ids whose decoded strings the reactions cite.

### `reactToHypo`

Defined at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Handles the hypothetical-disintegration variable-copy tag path — scopes marked with `_hypo_` payloads. In the current code it handles the book-keeping; the user-facing tag in the processed proof graph is `variable copy` (having subsumed the retired `reaction to hypo` and `necessity for equality (hypo)` tags).

---

## Origin tracking

Every emission records provenance into `exprOriginMap` (field of `Memory`). An entry maps:

```
(expression, validity) → list<origin-citation>
```

where each `origin-citation` names:

- The justification category (corresponds to the proof-graph tag).
- The antecedent `(expression, validity)` pairs the derivation cites.

Since `D-131` the map is stored in id form: keys and antecedents are `packOriginKey(expressionId, validityId)` int64 packs of two int32 ids from the dedicated per-LB `Memory::originInterner` (NOT the NameMap — see the decision entry), and the tag is the closed `OriginTag` enum (`originTagFromString` asserts on an unknown tag). Emission sites encode via `addOriginEncoded`; the mail origin maps stay string (mail is the cross-LB boundary), so the absorb path encodes mail records in and `fillMailOut` decodes records out. Every order-sensitive walk (the dump section, `findEnds`, the compressor extraction) derives a decoded `(expression, validity)` lex-sorted snapshot via `decodeOriginMapSorted` — exactly the former `std::map` key order.

When stage 5 (raw proof graph emission — `visualizer.cpp`) runs, it walks `exprOriginMap` backwards from each proved theorem to build a chapter: `buildStack` at [`visualizer.cpp`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp).

The origin map is the raw material of every subsequent auditability claim. A missing origin entry means the verifier has to fail the theorem; a *wrong* origin entry (emitting a justification that isn't actually valid) is the silent-unsoundness case verified against.

### Origin map locality (D-51)

Every origin record is **local to the LB that produced it**. Mail transports (parent → children, via `mailOut → sendMail → child.mailIn → child's exprOriginMap on next iteration`) carry origins **downward only**. There is no upward propagation: a child LB's origin records do not reach the parent's `exprOriginMap`. The `pendingAncestorOrigins` queue in `proveKernel` (originally a child-side `parentMemory`-walk that wrote a contradiction LB's `("contradiction", deps)` record into every ancestor's `exprOriginMap`, deferred for thread safety per D-39) is **retired** as of D-51 — the queue is never pushed and the drain iterates an empty container. Cleanup of the dead struct/mutex/loop is deferred.

The principled consequence for chapter emission: a contradiction LB's recipe lives only inside the `__contradiction__(head)` LB. The chapter walker (`buildStack`) enters that LB explicitly via the chapter-boundary `__contradiction__` `SimpleMapStore` edge fallback when a negated head has no acyclic direct origin in the LB before the head. This replaces the [D-49](../40_decisions.md#d-49) cap-full preference rule (now superseded — the cap-full preference logic remains in `addOrigin` but is rarely triggered with `max_origin_per_expr = 30` and is no longer load-bearing).

### `buildStack` chapter walker (D-51 algorithm + lifting per `D-56`)

`buildStack` at [`visualizer.cpp`](../../GL_Quick_VS/GL_Quick/src/visualizer.cpp) walks the proof tree starting from the chapter goal and emits one row per visited expression. Four mechanisms work together:

1. **Ancestor lifting.** At entry, `proved` is lifted via `liftToShallowestOriginAncestor(memoryBlock, proved)` — the helper walks the validity-stack ancestor chain of `proved.validityName` (recoverable from the canonical `parent + "_boundary_" + payload` string per [I-2](../30_invariants.md#i-2)) from `main` outward and returns the **closest-to-`main` ancestor** for which `(expr, ancestor)` exists as a key in `memoryBlock.exprOriginMap`. From this point on, `proved` is the lifted form: chapter rows are emitted with the lifted validity, the path-cycle filter inserts/erases the lifted form, and every dependency is independently lifted before being written into a row cell and before recursion. The dep-side lift uses the same helper. Net effect: every chapter cell (`row[1]` and every `row[3+2k+1]`) carries the lifted (closest-to-`main`-with-origin) validity, and each `(expr, lifted_v)` pair has at most one row per chapter via the `covered` dedup set. The pre-lifting fallback ("exact-key try, then `(expr, "main")` shadow lookup") is retired: lifting subsumes it and produces a truthful `row[1]` instead of one that re-tagged the row with the requested deep validity while citing a shallower origin.

 **OR-branch barrier.** The lift may not cross `_boundary_orint_` or `_boundary_ordis_` delimiters. OR-branch scopes are conditional on a disjunct hypothesis; preserving branch-distinct namespaces is required by the OR-family verifier checkers (`check_or_convergence`, `check_or_branch_proven`, `check_or_branch_assumption`, `check_or_disintegration`). The deepest `orint_`/`ordis_` ancestor sets the shallowest allowed lift target. Without the barrier, both branches of an OR converge to identical `(expr, parent_boundary)` cells (the parent boundary holds the post-convergence origin), breaking the verifier's branch-distinctness expectation and producing `origin chain termination` cycles. Boundary detection: payload (segment after `parent + "_boundary_"`) starts with `"orint_"` or `"ordis_"`.
2. **Path-cycle filter.** A `thread_local std::set<ExpressionWithValidity> g_buildStackPath` tracks the recursion path. Before picking an origin candidate, any candidate whose deps (after lifting) include an expression already on the path is rejected (would form a chapter-row cycle the verifier's `origin chain termination` check would flag).
3. **Chapter goal in path.** At `directStack` entry, the chapter goal expression (the wrapped theorem from `theoremList`) is inserted into `g_buildStackPath` and erased on return. The path-cycle filter then rejects the prover's self-applying forward-inference origin (origin tag `implication` with deps `[wrapped theorem, anchor]`) — the chapter walker would otherwise emit a `theorem`-tag leaf row matching the chapter goal, tripping `verifier.py::check_chapter` self-reference.
4. **`__contradiction__` LB fallback.** When all sorted candidates of a negated head fail (rejected by path-cycle filter, or no direct origin existed for the lifted `proved`), `buildStack` walks the LB chain (current → ancestors via `parentMemory`) for a child memory block keyed `"__contradiction__" + positive`. On hit, it switches into that contradiction LB and resolves the head locally — the contradiction record is local there.

`buildStack` performs **no other LB switch**. Each contradiction is an independent LB with its own chapter; nested contradiction LBs are NOT children of an outer contradiction LB. Recursion on a contradiction's three deps stays in the current LB; if a dep itself needs a contradiction proof, the chapter-boundary fallback the next level down enters another sibling `__contradiction__` LB attached to the anchor.

Sorting / backtracking. Per-key origin candidates are ordered by [D-49-style](../40_decisions.md#d-49) preference (non-equality tags first preserving insertion order, then `equality1`/`equality2` preserving insertion order). `buildStack` returns `bool`: `true` on success, `false` to signal the caller (recursive `buildStack`) to backtrack — roll back stack/covered snapshots and try the next candidate. Top-level callers (`directStack`, `checkZeroStack`, `checkInductionConditionStack`) ignore the return.

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

Alongside the two theorem files, the prover writes `files/GL_binaries/GL_binary_<tag>.json` — the compact-name dictionary specific to this batch. Schema entries are name → `{category, signature, …}`. The write path is project-rooted (computed from `__FILE__` ascent in `visualizer.cpp::generateRawProofGraph`), so every batch — incubator and main alike — writes into the single canonical `files/GL_binaries/` directory; see [D-54](../40_decisions.md#d-54) for the unification of a previously asymmetric writer path.

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
| [I-7](../30_invariants.md#i-7) — Pass B gated on `!incubator_mode` | `disintegrateExpr2` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) |
| [I-12](../30_invariants.md#i-12) — equivalence-class one-sided expansion | `applyEquivalenceClassToNegatedEquality` at [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp) |
| [I-17](../30_invariants.md#i-17) — `savedStartInt` single-counter | `disintegrateExpr2` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) |
| [I-18](../30_invariants.md#i-18) — induction typing | shipped; auxiliary `(in[digitArg, N])` triad before promotion, `<N>_induction_typing.txt` chapter, verifier `induction typing` checker |

---

## Weaknesses

### Known & tracked

- **Induction-typing soundness — closed.** Historically the prover scheduled induction without first verifying membership in `N`, which was unsound for bound variables appearing only in negations / existence heads / bare equalities. **Closed:** the prover now proves the typing triad before promotion; see [I-18](../30_invariants.md#i-18). Listed here for historical context.
- **Pass B single-input gate is empirical.** [I-6](../30_invariants.md#i-6) is enforced because widening broke Gauss. There is no theoretical argument here — only a measurement-grounded guardrail. A hypothetical operator with single-input semantics but admission-map requirements that the gate does not check would slip through.
- **Hard-coded Peano theorem drop.** On the current branch, `run_modes.cpp–150` hard-codes a filter that removes a specific Peano theorem from the inherited `proved_set` at the start of the Gauss batch. This is a known debug mitigation, not a shipping fix. See commit.
- **`excludeRepetitions` zero-free-arg signature — fixed.** `excludeRepetitions` (`prover.hpp`) builds a compiled-operator signature by appending `"u_<i>,"` per unchangeable arg then unconditionally `pop_back`-ing the trailing comma. With `numUnchArgs == 0` the pop ate the `[` and produced the malformed `"(nameN])"`. The trailing-comma strip is now guarded by `numUnchArgs > 0`, so a zero-free-arg compile (a fully-bound theorem, every variable bound per [D-75](../40_decisions.md#d-75)) yields the well-formed `"(nameN[])"`. Byte-identical for the ≥1-arg case (all pre-existing `or<N>`/`existence<N>`/`and<N>`/`implication<N>` allocation), so it cannot regress existing proofs; the corrected branch is only reachable on the new top-level-implication compaction path.

### Suspected fragility

- **`ExpressionAnalyzer` is a monolith.** ≈5925 lines in `prover.hpp` + ≈8770 in `prover.cpp`. Every proof primitive is a method on the same object. Refactor planned post-FTA (see ).
- **Parallel grid is live (no longer a fragility).** `logicalCores = std::max(1u, std::thread::hardware_concurrency)` (`prover.cpp`; the old `logicalCores = 1` override is commented out), and `proveKernel` runs real worker pools — across-LB sweeps for phases 1/3, a flat `(LB, part)` executor pool for phase 2. The push-mail `smashMail` / per-core-mailbox batching this note once warned about is **retired**, replaced by the pull-model `MailLog` ([D-137](../40_decisions.md#d-137) / [I-94](../30_invariants.md#i-94)); determinism under parallelism rests on [I-28](../30_invariants.md#i-28) (no cross-LB intra-cycle observation; sorted post-join drains) and the canonical-sort merge in `applyFiringRecords`.
- **`performDisintegration` phantom.** the project conventions names this function, but a grep does not find it — possibly renamed/retired. Stale documentation in the project conventions is a minor fragility but a symptom: doc drift on a core surface. See [OPEN-6](../AGENT_SwDD.md#open-questions).
- **Proof-cycle iteration cap.** `parameters.numberIterationsConjectureFiltering` and the main-mode iteration budget are batch-wide constants. A single unusually-hard conjecture can consume the full budget while the rest starve. No per-conjecture timeout.
- **`savedStartInt` freshness contract.** Any refactor that introduces a parallel counter violates [I-17](../30_invariants.md#i-17) silently.
- **Operator set inference.** `ExpressionAnalyzer::operators` is populated somewhere — OPEN-4 in AGENT_SwDD.md. A mismatch between this set and the conjecturer's notion of "operator" could silently alter Pass B behaviour.

### Worked example — a Gauss revival, end to end

Concrete trace of the mechanism on the first theorem it unblocked: `sequence[1,4,2,12,10] → sequence[1,4,2,9,11]` under `limitSequence[1,4,9,10,11] ∧ in2[9,12,3]` (conjectures.txt line 296 on the pre-merge branch). Anchor mapping `AnchorGauss[N, i0, s, +, *, i1, i2, id] = [1..8]`, so `4 = +`, `2 = i0`, `3 = s`.

**Cycle N — Pass B rejection.** Somewhere mid-Gauss, a compound of shape `(sequence[1, 4, 2, X, Y])` lands at some LB at `v=main`. `disintegrateExprCore2` (prover.cpp) unfolds it through `fXY / interval` expansion and mints an integration witness `int_lev_2_42` (level 2, startInt 42) for one of the inner `preorder` existentials. One body element of that existential is `(in3[a, int_lev_2_42, b, 4])` — "a + int_lev_2_42 = b in +", with `a`, `b` concrete from the compound's args.

Pass B's `int_` branch at `prover.cpp`:

1. `isAdmittedIntegration(mb, removedU="(in3[a,int_lev_2_42,b,4])", var="int_lev_2_42", markedExpr="(in3[a,marker,b,4])", …)` u_-prefixes non-marker args → looks up `admissionMapIntegration[(in3[u_a, marker, u_b, u_4])]` → miss (no template registered for this specific concrete `a`/`b`).
2. `admissionSetIntegration.find((in3[a, marker, b, 4]))` → miss.
3. Fall-through. New code kicks in: builds `siblings = [(in[int_lev_2_42, 1])]` (the typing element, preserved for re-emission), packages into `PendingRejectionIntegration{concreteConstituent, markedExpr, siblings}`, buffers. After cascade admission loop (around `prover.cpp`), the var is still un-admitted → `updateRejectedMapIntegration(markedExpr, concreteConstituent, siblings, topLevelExprClean, mb.overallHashMemory, "main")` fires.

State of `mb.overallHashMemory`:

```
rejectedMapIntegration[(in3[a,marker,b,4]) @ main] = {
    { concreteConstituent = "(in3[a,int_lev_2_42,b,4])",
      siblings            = ["(in[int_lev_2_42,1])"],
      compoundExpression  = "(sequence[1,4,2,X,Y])" }
}
varsInRejectedMapIntegrationKeys += {a, b, 4}
```

No visible fact was added to `intEncodedStatements`; the compound proof is stuck on this rung.

**Cycle N+k — equivalence class propagation.** Some subsequent LB proves `(=[X, 9])` (X can be identified with 9 given the other premises). The equality is absorbed at `v=main`; `updateEquivalenceClasses` creates/merges class `{X, 9}` with canonical `9` (say lex-min). `applyEquivalenceClass` (main helper) rewrites matching `intEncodedStatements`. Simultaneously, the new `applyEquivalenceClassToRejectedMapIntegration` fires at the same call sites (`prover.hpp` same-NS, `5982-5988` ancestor-NS, fixpoint re-iter at `6049`).

The helper:

1. `varsInRejectedMapIntegrationKeys` contains `{a, b, 4,...}`. Class `{X, 9}` — if `X` or `9` is in the cache, proceed. (One of a/b equals X in this trace → overlap.)
2. Iterate rmi snapshot. Find the `(in3[a,marker,b,4])` entry where `a = X`. Indices of X in args: `[0]`. Look up `allMappingsAna[(1, 2)]` → mappings `[[0], [1]]`.
3. Mapping `[0]`: X → eqList[0] = 9 (assuming `9 < X` lex). Rewritten: `(in3[9, marker, b, 4])`. Non-identity.
4. Mapping `[1]`: X → eqList[1] = X. Identity, skipped.
5. Dedup — one unique rewrite: `(in3[9, marker, b, 4])`.
6. Admission probe: `admissionMapIntegration[(in3[u_9, marker, u_b, u_4]) @ main]` — MATCH (some earlier compound at main had an integration template whose body reformulates "add inside N, output in +" at concrete anchor-slot values; its Case-C insert populated this entry).
7. Match path — emit to `sameIterationInternalMail`:
 - Primary: pre = `(in3[a, int_lev_2_42, b, 4])`, post = `(in3[9, int_lev_2_42, b, 4])` (via `substMap = {a→9}` applied to pre).
 - Sibling: pre = `(in[int_lev_2_42, 1])`, post = unchanged (no class member in it).
 - Both pushed to `mb.sameIterationInternalMail.statements.insert(tuple(post, lvls, "main"))`.
 - Each gets an origin entry: `origin.first = "equality1"`, `origin.second = [pre, (=[a,9])]`.
8. Entry erased from `rejectedMapIntegration` (rejection resolved). Admission-map entry NOT erased ([I-22](../30_invariants.md#i-22)).

**Cycle N+k+1 — absorb, re-disintegrate, prove.** Top of next hashburst at this LB:

1. `sameIterationInternalMail` absorb loop (`prover.cpp+`) iterates `sortedInternal`. For each tuple, calls `addExprToMemoryBlock(stmt, body, -1, status=1, levels, origin, coreId, -1, validityName, false)`. `status=1` ≠ 3, so `disintegrateExpr2` runs (`prover.cpp` gate).
2. For `(in3[9, int_lev_2_42, b, 4])`: disintegrate → atom, goes into `intEncodedStatements` at main.
3. For `(in[int_lev_2_42, 1])`: atom, goes into `intEncodedStatements` at main.
4. `updateAdmissionMapIntegration` fires on each deposit (main-only gate at `prover.hpp`) — walks each arg, probes admissionMapIntegration with marker-forms, fires `prepareIntegrationCore2` for any that match. This is where downstream integration rules pick up the new facts and drive toward the compound's proof.
5. Subsequent hash bursts: the compound's enclosing implication (`sequence → sequence`) has its premise chain now satisfied. Head `(sequence[1,4,2,9,11])` is derived. Moves out of `toBeProved` into `theorems.txt`.

**Origin trail as seen by the verifier.** The chapter for the proved theorem has a row for `(in3[9, int_lev_2_42, b, 4])` tagged `equality1`, with `rest = [source="(in3[a,int_lev_2_42,b,4])", ns="main", eq="(=[a,9])", ns="main"]`. Verifier's `check_equality1` at `verifier.py`:
- result_core == source_core (`in3`) ✓
- arity match (4) ✓
- differing args: position 0 — `(a, 9)` ∈ equality set? Yes (the origin provides `(=[a,9])`) ✓ — accepted.

Non-differing positions (`int_lev_2_42`, `b`, `4`) need no equality. Row passes.

**If canonical-only instead of full permutation.** For this trace the class `{X, 9}` has size 2 and only X is in the key, so canonical-only coincides with full permutation — one rewrite, same result. A future rung where two class members land in the same key (e.g. `{X, Y}` with key `(in3[X, marker, Y, 4])`) would have canonical-only produce `(in3[9, marker, 9, 4])` (collapse), while full permutation would also produce `(in3[9, marker, Y, 4])` and `(in3[X, marker, 9, 4])` — three distinct admission-probe chances. The Gauss `{0,1}=[0,1]` rung doesn't hit this, but FTA-ladder downstream almost certainly will.

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

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
