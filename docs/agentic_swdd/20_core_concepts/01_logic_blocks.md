<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Logic blocks `[DRAFT]`

> The Logic Block (LB) is the unit of execution in the GL grid. Every proof state lives in some LB's `Memory`. Every inter-step communication goes through mail. The LB-grid model is what makes GL both logically clean (no shared state inside a cycle) and forward-compatible with the ASIC roadmap — each LB maps to a silicon core.

---

## What an LB is

A Logic Block is a self-contained hash-inference engine. Each LB owns:

- A **local hash memory** — the `HashMemory` it queries and writes to.
- A set of **known expressions** — `intKnownStatements` (the packed-key registry carrying the former `wholeExpressions` membership as its `registered` bit), the int16 statement registries `intEncodedStatements` / `intLocalEncodedStatements{,Delta}` (string form reconstructed on demand via `decodeExpression`), and the packed-key indexes `intLocalEncodedStatementsSet` / `intStatementLevelsMap`.
- An **origin map** — `exprOriginMap`, recording provenance for every emission.
- An **equivalence-class registry** — `equivalenceClassesMap`, one per validity.
- A **validity stack** — `stackOfValidity`, the active scope IDs for this LB.
- A **name map** — `NameMap`, the scope-ID encoder.
- A **mailbox pair** — `mailIn` as mail-pool `RoutingColdMail`; `mailOut` plus its private interner as deloadable `DeloadableMailOut` in the LB arena.
- A **parent pointer** — `parentMemory`, or `nullptr` for the root.
- **Children (down-edges)** — held in `SimpleMapStore` (parent → child by interned routing key), off the `Memory` shell, no longer a member map.
- A **status** — `isActive`, `primedForContradiction`, `toBeProved`, etc.

The class that holds all of this is `Memory`, defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). The name `Memory` is the current name; historically it was `BodyOfProves`, renamed during the `analyze_expressions` monolith split.

---

## The hierarchy

An `ExpressionAnalyzer` has one root LB: `body`. Children are created dynamically for:

1. **Target proofs** — one LB per conjecture being proved. The LB's `exprKey` is the conjecture's head.
2. **Hypothesis LBs** — when the prover assumes a premise, a child LB opens with the premise as its `exprKey` and the assumed-premise's scope as its validity.
3. **OR branch LBs** — one per disjunct on an OR disintegration. Each branch runs with the other disjuncts' negations as seeds.
4. **Integration LBs** — created during reformulation-for-integration.
5. **Auxiliary LBs** — for recursion sub-goals (induction base, step, and now typing).
6. **Compressor LBs** — one per theorem during Phase 1 of the compressor. These are independent trees rooted at fresh `Memory` instances, not children of the main prover body.

The `parentMemory` pointer lets any LB walk up to the root. Cross-scope operations (promotion from branch to parent, hypothesis discharge, integration binding) follow this chain explicitly — never via shared state.

---

## The one-cycle-at-a-time execution model

A GL run proceeds in cycles:

1. **Cycle start.** Each active LB pulls the prior retained `MailLog` window from its ancestors into `mailIn` and advances its cumulative cursors.
2. **Per-LB execution.** `proveKernel` runs the elementary step over every active LB on a pool of `logicalCores` real worker threads (work-stealing over a shared `active` vector), as three barriered phase sweeps — `performElemPhase1` (mail absorb) → `performElemPhase2` (the hashburst) → `performElemPhase3` (post-burst absorb), with a join between each ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp), [D-114](../40_decisions.md#d-114)). The hashburst (phase 2) is where, per LB:
 - Generate hash requests from known expressions.
 - Query `overallHashMemory.encodedMap`.
 - On a hit, fire the rule — build the conclusion instance, invoke `addStatement`, record origin.
 - Enqueue outgoing messages into `mailOut`.
3. **Cycle end.** After phase 3 joins, an all-active-at-grid-build batch retires the delivered blob/ref window; a batch with any initially dormant LB keeps full history. The commit barrier then serializes each non-empty `mailOut` into the next retained window and clears it.
4. **Next cycle** — loop until no LB produces anything new, or an iteration budget is exhausted.

**Key property.** LBs never read each other's state *during* a cycle. All cross-LB observation happens via mail, at cycle boundaries. This is both the correctness guarantee (no data races) and the parallelism affordance (the sequential inner loop can become parallel without changing semantics; for the ASIC roadmap).

---

## Memory — the fields

Selected fields of `Memory` (not exhaustive — the struct is large):

| Field | Type | Role |
|---|---|---|
| `exprKey` | accessor over `int32_t exprKeyId` | The expression this LB is proving or storing — statified: a 4-byte id into the never-deloaded `skeletonInterner`, decoded byte-identically by `exprKey` (0 = empty root sentinel; [I-97](../30_invariants.md#i-97)). |
| `level` | `int` | Scope depth relative to root. |
| `isActive` | `bool` | Whether this LB participates in the cycle. |
| `primedForContradiction` | `bool` | Keeps LB alive in `deactivateRecursively` for contradiction discharge. |
| `intToBeProved` | `unordered_map<int32_t, …>` | Goals yet to derive — packed `(originalId, validityId)` keys ([D-130](../40_decisions.md#d-130)). |
| `intKnownStatements` | `std::unordered_map<int32_t, StatementFlags>` | Packed `(originalId, validityId)` keys; `registered` / `known` membership bits ([I-85](../30_invariants.md#i-85)). |
| `intEncodedStatements` | `std::vector<IntEncodedExpr>` | The statement registry — int16 rows, string form on demand via `decodeExpression`. |
| `intLocalEncodedStatements{,Delta}`, `intLocalEncodedStatementsSet` | local-only variants | Written only when `addStatement(local=true)`. The `Set` variant ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)) is a packed-key `std::unordered_set<int32_t>` for O(1) membership, maintained beside the vector — used by the D-29 disintegration gate ([I-7](../30_invariants.md#i-7) clause 2). |
| `intStatementLevelsMap` | `packStatementKey(originalId, validityId) → std::set<int>` | Per-statement set of LB depths (`Memory::level` values, anchor=0, child=1, …) whose state contributed to the derivation of that statement. Levels propagate as union(rule.levels, premise.levels) at every firing. Read by `prover.cpp::addExprToMemoryBlock`'s discharge gate `allLevelsInvolved` — see [`02_glossary.md::levels`](../02_glossary.md#levels) and [I-51](../30_invariants.md#i-51). |
| `overallHashMemory` | `HashMemory` | This LB's implication-rule index. |
| `localHashMemory`, `localHashMemoryDelta` | scoped hash memories | Per-branch or per-hypothesis rule registries. |
| `stackOfValidity` | `std::vector<int16_t>` | Active scope IDs. |
| `nameMap` | `NameMap` | Scope-ID encoder. |
| `equivalenceClassesMap` | `unordered_map<validityId, vector<EquivalenceClass>>` | Equality-class registry, keyed by scope id ([D-134](../40_decisions.md#d-134)); probe via the non-minting `classesAt`. |
| `mailIn` | `RoutingColdMail` | Inter-LB inbox staging on the never-deloaded mail pool; cleared after phase-1 absorb. |
| `mailOut` | `DeloadableMailOut` | Inter-LB outbox + private ids on the LB's deloadable main-pool arena; shell pending bit selects serial commit reload. |
| `sameIterationInternalMail` | `Mail` | Per-LB integration-revival inbox (typed `Mail` since 2026-05-07; [D-53](../40_decisions.md#d-53) unification, renumbered from main's D-46— pre-unification was a separate `struct InternalMail`). Populated during hashburst body by `applyEquivalenceClassToRejectedMapIntegration` / `revisitRejectedIntegration2`; drained at top of next hashburst with `status=1` absorb. Not routed across LBs. See [`03_mail_system.md`](03_mail_system.md#integration-revival-channel). |
| `parentMemory` | `Memory*` | Up-pointer. |
| (down-edges) | `SimpleMapStore` (off-shell, not a `Memory` member) | Children, by interned routing key. |
| `exprOriginMap` | `IdOriginMap` (packed origin-interner keys → `(OriginTag, packed deps)` lines) | Provenance (id form per `D-131`; decode via `Memory::originInterner`). |
| `startInt`, `startIntRepl`, `startIntPi` | counter snapshots | Used by Pass B freshness checks. |

---

## LB lifecycle

1. **Birth.** `new Memory` — root and compressor LBs; child-creation paths inside the prover (`makeHypothesisBlock`, `makeBranchBlock`, `makeIntegrationBlock` — exact names vary per kind).
2. **Registration.** Inserted into `permanentBodies` (or `permanentBodiesCE` for CE filter) and linked under its parent in `SimpleMapStore` (`linkChild`; `ceSimpleMapStore` for the CE tree).
3. **Active execution.** Participates in cycles while `isActive == true`.
4. **Deactivation.** `deactivateRecursively` — the LB and its descendants are marked inactive when their target is proved or the scope discharges. This active-to-inactive transition is permanent. The only inactive-to-active path is an induction-zero LB born dormant and later woken by `activateZeroCondition` ([I-112](../30_invariants.md#i-112)); this is why any initially dormant grid retains full mail history.
5. **Tear-down.** Main-pipeline LBs persist for the duration of the run; compressor Phase 1 LBs are explicitly `delete`d after their proof-graph extraction via the `destroyGrid` lambda at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp).

---

## Per-batch grid — the `permanentBodies` model

For the main prover, `permanentBodies` is the full vector of all grid LBs, active or dormant. The outer `prove` loop iterates cycles over the current active subset. Cross-LB mail is the pull model — each LB's `mailOut` is committed to its [`MailLog`](03_mail_system.md#the-pull-model) chain at the cycle-end barrier and pulled by descendants in phase 1. The completed-grid dormant scan fixes either rolling delivered-window retirement or full-history catch-up for the batch ([I-161](../30_invariants.md#i-161)). The old per-core `boxes` + `sendMail` / `smashMail` machinery is deleted ([D-137](../40_decisions.md#d-137)); `MailLog` lives on `ExpressionAnalyzer`, not `Memory`.

For CE filtering, the analogous structure is `permanentBodiesCE`. CE runs the same hash-request generator with an empty obligatory stump (`generateEncodedRequestsStatic`, stump length 0) and a different outer budget (`numberIterationsConjectureFiltering`); its isolated clone LBs have no descendants, so the pull is gated off for them.

The grid runs in parallel: `logicalCores = std::max(1u, std::thread::hardware_concurrency)` (the old `=1` hardcode is commented out), so `proveKernel` spawns real worker threads over three barriered phase sweeps per cycle.

---

## Compressor LBs vs main LBs

Different from main LBs in that each compressor LB:

- Is a freshly-allocated `Memory` not connected to the main hierarchy.
- Has *all* proved theorems loaded as implication rules (`addToHashMemory` × N).
- Runs hash bursts to extract a per-theorem derivation graph, then is discarded.
- Lifetime: milliseconds to seconds per LB; explicit `delete` at end of Phase 1.

See [`10_pipeline/05_compressor.md`](../10_pipeline/05_compressor.md) for detail.

---

## The ASIC correspondence

The LB model maps cleanly to silicon. Every LB becomes a core:

- **Hash memory** → on-die SRAM per core.
- **wholeExpressions / intKnownStatements** → per-core register file.
- **mailIn / mailOut** → network-on-chip packets between cores.
- **Cycle boundaries** → clock edges at the fabric level.

Post-FTA roadmap targets a 100–1000× memory reduction (LB split, static, no history) — enough to cross the on-die SRAM threshold and enable a meaningful silicon prototype.

---

## Weaknesses

### Known & tracked

- **Parallel grid is live.** `logicalCores = std::max(1u, std::thread::hardware_concurrency)` — the old `=1` hardcode is commented out, so `proveKernel` runs real worker threads, and since the LB-split barriers ([D-114](../40_decisions.md#d-114)) each cycle is three barriered phase sweeps. Cross-LB writes during the parallel phase stay forbidden ([I-28](../30_invariants.md#i-28)) — ancestor-direction effects go through the class-level deferred-action collectors, drained sorted after the join. Cross-LB mail uses the pull model: commits run single-threaded at the post-join barrier, and the phase-1 pull reads frozen logs ([I-94](../30_invariants.md#i-94)). (`smashMail` and the parallel-drain question it raised are retired.)
- **`Memory` monolith.** ≈1430 lines in `memory.hpp`. Refactor on the post-FTA plan — split into concerns (proof state, hash engine, mail, scope). Premature before FTA.

### Suspected fragility

- **Child-LB book-keeping is implicit.** Child LBs are linked under their parent in `SimpleMapStore` by an interned routing key (`std::to_string(index)` or similar) — a per-parent counter that is not itself audited. A bug that miscounts (or collides keys) would corrupt the hierarchy silently.
- **`parentMemory` null-check discipline.** Every walk-up path must handle the root case. A missing null check in a new walk-up site crashes.
- **`startInt` / `startIntRepl` / `startIntPi` freshness.** Three parallel counters. The Pass B freshness check assumes `savedStartInt` compares monotonically — but with three counters, care is needed that the *right* one is being checked. See [I-17](../30_invariants.md#i-17).

### Not exercised by tests

- **LB deactivation correctness.** The `deactivateRecursively` + `primedForContradiction` interplay is not unit-tested. Regression would show up as incorrect discharge, but potentially silently (wrong proof emitted).
- **Compressor Phase 1 LB teardown.** `destroyGrid` walks the tree and deletes. A cycle (unlikely but not forbidden) would loop forever; a dangling pointer (possible if a mail queue held a reference) would fault.

---

## See also

- [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — main prover operation.
- [`10_pipeline/05_compressor.md`](../10_pipeline/05_compressor.md) — compressor Phase 1 LBs.
- [`20_core_concepts/02_hash_engine.md`](02_hash_engine.md) — per-LB `HashMemory`.
- [`20_core_concepts/03_mail_system.md`](03_mail_system.md) — inter-LB messaging.
- [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md) — per-LB scope stack.
-, — ASIC roadmap.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
