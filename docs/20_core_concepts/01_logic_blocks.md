<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Logic blocks `[DRAFT]`

> The Logic Block (LB) is the unit of execution in the GL grid. Every proof state lives in some LB's `Memory`. Every inter-step communication goes through mail. The LB-grid model is what makes GL both logically clean (no shared state inside a cycle) and forward-compatible with the ASIC roadmap — each LB maps to a silicon core.

---

## What an LB is

A Logic Block is a self-contained hash-inference engine. Each LB owns:

- A **local hash memory** — the `HashMemory` it queries and writes to.
- A set of **known expressions** — `wholeExpressions`, `intKnownStatements`, `encodedStatements`, `localEncodedStatements`, and their integer-encoded counterparts.
- An **origin map** — `exprOriginMap`, recording provenance for every emission.
- An **equivalence-class registry** — `equivalenceClassesMap`, one per validity.
- A **validity stack** — `stackOfValidity`, the active scope IDs for this LB.
- A **name map** — `NameMap`, the scope-ID encoder.
- A **mailbox** — `mailIn` (inbox) and `mailOut` (outbox) `Mail` buffers.
- A **parent pointer** — `parentMemory`, or `nullptr` for the root.
- A **children map** — `simpleMap`, keyed by child identifier.
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

1. **Cycle start.** All LBs are idle; `mailIn` buffers hold messages queued from the previous cycle.
2. **Per-LB execution.** For each active LB (iterated in some order — currently sequentially in the single-process build), call `performElementaryLogicalStep` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp):
 - Generate hash requests from known expressions.
 - Query `overallHashMemory.encodedMap`.
 - On a hit, fire the rule — build the conclusion instance, invoke `addStatement`, record origin.
 - Enqueue outgoing messages into `mailOut`.
3. **Cycle end.** Drain mail: every `mailOut` merges into the recipient's `mailIn`.
4. **Next cycle** — loop until no LB produces anything new, or an iteration budget is exhausted.

**Key property.** LBs never read each other's state *during* a cycle. All cross-LB observation happens via mail, at cycle boundaries. This is both the correctness guarantee (no data races) and the parallelism affordance (the sequential inner loop can become parallel without changing semantics; for the ASIC roadmap).

---

## Memory — the fields

Selected fields of `Memory` (not exhaustive — the struct is large):

| Field | Type | Role |
|---|---|---|
| `exprKey` | `std::string` | The expression this LB is proving or storing. |
| `level` | `int` | Scope depth relative to root. |
| `isActive` | `bool` | Whether this LB participates in the cycle. |
| `primedForContradiction` | `bool` | Keeps LB alive in `deactivateRecursively` for contradiction discharge. |
| `toBeProved` | set | Goals yet to derive. |
| `wholeExpressions` | `std::unordered_set<EncodedExpression>` | Every expression known to this LB at string level. |
| `intKnownStatements` | `std::unordered_set<int>` | Integer-encoded equivalent. |
| `encodedStatements`, `intEncodedStatements` | statement registry | Keyed by `EncodedExpression`. |
| `localEncodedStatements{,Delta}`, `intLocalEncodedStatements{,Delta}`, `localEncodedStatementsSet` | local-only variants | Written only when `addStatement(local=true)`. The `Set` variant ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)) is a parallel `std::set<EncodedExpression>` for O(log N) lookup, maintained alongside the vector — used by the D-29 disintegration gate ([I-7](../30_invariants.md#i-7) clause 2). |
| `statementLevelsMap` | `EncodedExpression → levels` | Admission ordering. |
| `overallHashMemory` | `HashMemory` | This LB's implication-rule index. |
| `localHashMemory`, `localHashMemoryDelta` | scoped hash memories | Per-branch or per-hypothesis rule registries. |
| `stackOfValidity` | `std::vector<int16_t>` | Active scope IDs. |
| `nameMap` | `NameMap` | Scope-ID encoder. |
| `equivalenceClassesMap` | `map<validity, EquivalenceClass>` | Equality-class registry. |
| `orAdmissionSet` | set | OR branch admission bookkeeping. |
| `mailIn`, `mailOut` | `Mail` | Inter-LB messaging (main-scope broadcast channel). |
| `internalMailIn` | `Mail` | Per-LB integration-revival inbox (typed `Mail` since 2026-05-07; [D-53](../40_decisions.md#d-53) unification, renumbered from main's D-46— pre-unification was a separate `struct InternalMail`). Populated during hashburst body by `applyEquivalenceClassToRejectedMapIntegration` / `revisitRejectedIntegration2`; drained at top of next hashburst with `status=1` absorb. Not routed across LBs. See [`03_mail_system.md`](03_mail_system.md#integration-revival-channel). |
| `parentMemory` | `Memory*` | Up-pointer. |
| `simpleMap` | `std::map<string, Memory*>` | Children. |
| `exprOriginMap` | `map<EncodedExpression, list<Origin>>` | Provenance. |
| `startInt`, `startIntRepl`, `startIntPi` | counter snapshots | Used by Pass B freshness checks. |

---

## LB lifecycle

1. **Birth.** `new Memory` — root and compressor LBs; child-creation paths inside the prover (`makeHypothesisBlock`, `makeBranchBlock`, `makeIntegrationBlock` — exact names vary per kind).
2. **Registration.** Inserted into `permanentBodies` (or `permanentBodiesCE` for CE filter) and its parent's `simpleMap`.
3. **Active execution.** Participates in cycles while `isActive == true`.
4. **Deactivation.** `deactivateRecursively` — the LB and its descendants are marked inactive when their target is proved or the scope discharges. `primedForContradiction` is the override flag that keeps a contradiction-LB alive past its nominal deactivation so the discharge path can fire.
5. **Tear-down.** Main-pipeline LBs persist for the duration of the run; compressor Phase 1 LBs are explicitly `delete`d after their proof-graph extraction via the `destroyGrid` lambda at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp).

---

## Per-core batching — the `permanentBodies` + `boxes` model

For the main prover, `permanentBodies` is the full vector of active LBs, `index` is a parent/children map over them, and `boxes` is per-core mailbox arrays. The outer `prove` loop iterates cycles, calling per-LB step functions and then draining `boxes`.

For CE filtering, the analogous structures are `permanentBodiesCE`, `indexCE`, `boxesCE`. The distinction is there because CE runs with a different hash-request generator (`generateEncodedRequestsStaticCE`) and a different outer budget (`numberIterationsConjectureFiltering`).

Note: per, `logicalCores` is hardcoded to 1. The `boxes` structure is therefore a single-element array in practice; the parallelism hooks exist but are not exercised.

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

- **`logicalCores = 1` hardcode.** The grid's parallel affordance is not exercised in the current binary. Changing this without adding the parallel-smashMail logic would break (see — parallel smashMail was explicitly rejected as a previous attempt).
- **`Memory` monolith.** ≈1430 lines in `memory.hpp`. Refactor on the post-FTA plan — split into concerns (proof state, hash engine, mail, scope). Premature before FTA.

### Suspected fragility

- **Child-LB book-keeping is implicit.** Child LBs are registered into `simpleMap` keyed by `std::to_string(index)` or similar — a per-parent counter that is not itself audited. A bug that miscounts (or collides keys) would corrupt the hierarchy silently.
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
