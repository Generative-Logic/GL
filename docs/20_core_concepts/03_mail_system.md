<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Mail system `[DRAFT]`

> LBs communicate between cycles via mail. This is what keeps per-LB execution in a single cycle race-free while still letting the grid as a whole converge — observations made by one LB this cycle become observable to others only next cycle.

---

## Why mail

Two design goals compete:

1. **Parallelism.** The LB grid is the architectural abstraction for distributed execution (current single-process; roadmap ASIC). Multiple LBs should be able to compute in parallel.
2. **Determinism.** Proof results must be reproducible. An LB must not observe a partial update to another LB's state, or race on shared data.

Mail resolves the tension by pinning all inter-LB observation to cycle boundaries. Within a cycle, each LB sees only its own state as it was at cycle start. At cycle end, outgoing messages are merged into recipients' inboxes; next cycle, they become observable.

---

## The `Mail` struct

Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Fields:

| Field | Type | Role |
|---|---|---|
| `statements` | `set<pair<ExpressionWithValidity, levels>>` | Expressions to be delivered — each becomes an `addStatement` call on the recipient. The `ExpressionWithValidity` (`memory.hpp::ExpressionWithValidity`, fields `original` and `validityName`) carries the install scope per element. For routing-channel traffic (`mailIn`/`mailOut`) the `validityName` is `"main"` by [I-26](../30_invariants.md#i-26) sender contract; per-LB `internalMailIn` traffic may carry non-main scopes. Migrated 2026-05-07 — was previously `set<pair<string, levels>>` with `"main"` hardcoded at the receiver. |
| `implications` | `set<tuple<chain, head, args, levels, theorem>>` | Implication rules to be installed on the recipient. |
| `exprOriginMap` | `map<ExpressionWithValidity, list<Origin>>` | Provenance entries to merge into the recipient's origin map. |

Every `Memory` has two `Mail` instances: `mailIn` (inbox) and `mailOut` (outbox).

In addition, every `Memory` carries a third `Mail` instance, **`internalMailIn`**, dedicated to the integration-revival path ([D-19](../40_decisions.md#d-19)). It uses the same struct as the routing channels (since 2026-05-07 — [D-53](../40_decisions.md#d-53), renumbered from main's D-46; pre-unification this was a separate `struct InternalMail` with a `set<tuple<string, levels, validityName>>` `statements` shape). Revival emissions carry per-element scope in the EWV's `validityName` (typically non-main; e.g. Branch A inside an OR integration). The `implications` field exists on this `Mail` instance but is unused — no producer pushes implications into the revival channel, and the absorb path consumes only the statements set. The channel has no outbox counterpart — it is per-LB, produced and consumed within the same LB across cycle boundaries. See the [Integration-revival channel](#integration-revival-channel) section below.

---

## The cycle-boundary protocol

1. **Cycle start.** `mailIn` holds whatever was queued to this LB in the previous cycle. `mailOut` is empty.
2. **Cycle body.** During `performElementaryLogicalStep`, every new statement emitted at `validityName == "main"` also pushes into `mailOut.statements`. Implications likewise push into `mailOut.implications`. Provenance entries accumulate in `mailOut.exprOriginMap`.
3. **Cycle end.** The outer `prove` driver drains each LB's `mailOut`, routing entries to the appropriate recipient's `mailIn`. The routing map is `buildParentChildrenMap` + `buildPerCoreMailboxes` computed at grid setup time.
4. **Next cycle start.** Each LB processes its `mailIn`:
 - Statements → `addStatement(expr, this, levels, validityName, local=false)`.
 - Implications → `addToHashMemory(chain, head, args, levels,...)`.
 - Origin entries → merged into `exprOriginMap`.
 - **Equality origins** for positive 2-arg equality keys `(=[a,b])` are *additionally* synced into every `EquivalenceClass.equalityOriginMap` whose `validityName` matches and whose `variables` already contain both `a` and `b`. Without this sync the class state is incomplete: `mergeTwoEquivalenceClasses` (`prover.hpp::mergeTwoEquivalenceClasses`) consults class state when generating `equality2` cross-pair origin records and would treat mail-arrived equalities as not-yet-derived, emitting redundant transitive-closure records that can form cycles when distinct bridge variables (`commonArg`) generate mutually-pointing origins. Mailed equalities whose vars are not yet class-bound are picked up later via `prover.hpp::updateEquivalenceClasses` line 5886 when their own absorption fires.
 - `mailIn` is then cleared.

Key property: **LB execution during cycle N cannot depend on any state produced by other LBs in cycle N.** The mail drain is the synchronisation point.

---

## Main-only gate for `Mail::statements` and `Mail::implications`

`Mail::statements` and `Mail::implications` are **MAIN-ONLY** by contract. Only expressions whose install validity is `"main"` are mailed; non-main expressions stay local to the deriving LB.

Two enforcement sites:

- **Statements**: `addStatement` at [`prover.hpp` (around the `local` branch in `addStatement`'s body)](../../GL_Quick_VS/GL_Quick/src/prover.hpp) gates on `validityName == "main"` before pushing into `memoryBlock.mailOut.statements`. Non-main deposits enter `localEncodedStatements*` and `wholeExpressions` only.
- **Implications**: `addExprToMemoryBlock`'s post-disintegration loop at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) gates on `impValidity == "main" && allowedForMail(impStr, memoryBlock)`. The `impValidity` here is each imp's actual install scope as returned by `disintegrateExpr2`'s `imps` set, *not* the function's outer `validityName` — disintegration may produce imps at sub-scopes via push semantics.

Receivers split by channel:

- **`mailIn.implications`** absorb at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) hardcodes `"main"` as the install validity. The implications 5-tuple already has a validity field in its 5th element (the `theorem`/`orImpl` slot, used for OR-derivation context, not install scope), but the receiver does not consult it — the sender contract guarantees only main-scope rules ship through this channel.
- **`mailIn.statements`** absorb at `performElementaryLogicalStep` reads `validityName` from the EWV in each tuple's first element AND carries a per-element `assert(vName == "main")` as the runtime enforcement of I-26 for the statements channel. For routing-channel traffic the EWV always carries `"main"` by sender convention; the assert traps any sender that ever pushes non-main into `mailOut.statements` (it would propagate via `smashMail` into `mailIn.statements` and surface here). Non-`"main"` reaches a `Mail`-shaped absorb only through the separate `internalMailIn` drain block (top of hashburst, `status=1`); that block has no main-only assert because revival traffic is validity-aware by design. Migrated 2026-05-07.

**Why the gate exists.** Hashmem rules are tied to the hypothesis stack at their install scope. A rule at `v=main_boundary_(impl24[…])` is conditional on the impl24 hypothesis being active; shipping it to another LB and reinstalling at `v=main` discards that conditionality and produces unsound rule firings. Receivers re-derive non-main rules from their own disintegration of the mailed v=main statements — yielding properly-scoped origins via `trackExpansionHistory` and properly-scoped admission via `addToHashMemory`.

**`mailOut.exprOriginMap` is NOT main-gated** — provenance entries for ALL scopes (main and non-main) flow through mail. Receivers' `body.exprOriginMap` (post-merge) carries the sender's full provenance graph at every validity it lived at, so the visualizer's `buildStack` walk can resolve cross-LB dependency chains even when the depending rule lives at a sub-scope.

History note. The implications-side gate landed with [D-34](../40_decisions.md#d-34) in 2026-05-02. Pre-D-34, `prover.cpp` only consulted `allowedForMail` (which gates on expression shape, not validity); non-main rules shipped silently and got reinstalled at `v=main` at the receiver, with origins keyed at the sender's deeper scope. The visualizer's firing-time lookup at `(rule, "main")` found no exprOriginMap entry and asserted (`buildStack: no origin found`). The crash was latent until D-34's `ordisMerge` unblocked the FTA-rung-1 §4.1 chain and reached the contradiction LB's saved-theorem visualization.

---

## Who receives what

The delivery rules depend on scope and channel:

- **`mailOut.statements`** — main-only by contract (see above). Broadcast to *every* LB in the grid via the routing map. Each recipient adds the statement to its own `wholeExpressions` and runs the usual admission logic.
- **`mailOut.implications`** — main-only by contract (see above). Broadcast to LBs that would be expected to consume them; in the current build the broadcast is over-approximating (every LB in the grid).
- **`mailOut.exprOriginMap`** — carries entries for ALL scopes per the [`trackExpansionHistory`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) invariant. Propagates along the same routing paths as statements/implications and merges into the recipient's `exprOriginMap` at the same keys.
- **Statements at non-main scopes** never enter `mailOut`. They stay local to the deriving LB. If a sibling/child needs the same fact, it re-derives via its own disintegration of the mailed v=main statements.

The parent/children mapping `buildParentChildrenMap` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) is the routing backbone. It computes reachable nodes and direct-child pointers starting from the root LB(s).

---

## Per-core mailboxes

In principle, to avoid contention in a parallel-core setup, each "core" has its own inbox-outbox pair. The `boxes` structure is `vector<pair<mailIn*, mailOut*>>`, one entry per core, with LBs distributed across cores at setup.

In practice, `logicalCores = std::max(1u, std::thread::hardware_concurrency)` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp); the `=1` line is commented out at line 245). `proveKernel` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) spawns `workers = logicalCores` real threads, each pulling LBs from a shared `bodies` vector via `std::atomic<size_t> next.fetch_add(1)` and calling `performElementaryLogicalStep` independently. The parallel-mailbox infrastructure IS exercised — `boxes[mb].resize(logicalCores)` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) gives each LB one mailbox slot per core. Two LBs can run on different threads at the same time. The non-determinism investigation that produced [D-39](../40_decisions.md#d-39) discovered exactly this: a descendant LB's contradiction handler walked across into ancestor LBs' `exprOriginMap` from a worker thread while the ancestor was being processed on another worker thread. The fix is the deferred-action collector pattern documented in the [Class-level deferred-action collectors](#class-level-deferred-action-collectors) section below.

`buildPerCoreMailboxes` lives at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) — see [OPEN-21 RESOLVED](#open-questions).

---

## Integration-revival channel

`internalMailIn` is the second inbox introduced with the `rejectedMapIntegration` mechanism ([D-19](../40_decisions.md#d-19)). It exists so that integration-side rejection recovery does not have to route through the grid-broadcast `mailOut → smashMail → mailIn` pipeline.

**Producers.**

- `applyEquivalenceClassToRejectedMapIntegration` — on an equivalence-class rewrite that produces a marker-form matching an existing `admissionMapIntegration` or `admissionSetIntegration` entry, emits the rewritten concrete constituent + siblings via `emitIntegrationRevivalToInternalMailIn`.
- `revisitRejectedIntegration2` — fired at `admissionMapIntegration` insert (`prover.hpp`, with u_-strip on the key) and at `admissionSetIntegration` insert (`prover.cpp`, key already bare). Emits any matching `rejectedMapIntegration` entries.

**Consumers.** The hashburst entry absorbs `internalMailIn` at the top, immediately before the legacy `mailIn` absorb, with `status=1` (full disintegration pipeline — unlike legacy `mailIn` which uses `status=3` to skip disintegration; see [D-19](../40_decisions.md#d-19) rationale).

**Lifecycle asymmetry.** Legacy `mailIn` is cleared at the end of each hashburst (`prover.cpp`) — inserts come from between-cycle routing. `internalMailIn` is cleared at the **top** of each hashburst, immediately after absorb, because its inserts happen *during* cycle body (from eq-class rewrites and admission-key revisits triggered inside `addStatement`). See [I-21](../30_invariants.md#i-21).

**Scope.** As of 2026-05-07 ([D-53](../40_decisions.md#d-53) unification, renumbered from main's D-46), `Mail::statements` is `set<pair<ExpressionWithValidity, levels>>` carrying scope per element, and the integration-revival channel `Memory::internalMailIn` uses the same struct. `mailOut.statements.insert` is still gated on `validityName == "main"` at the routing senders (so the EWV always carries the literal `"main"` for routing traffic), and the receiver absorb reads `validityName` from the EWV. `internalMailIn` senders push EWVs with the actual revival scope (typically non-main; e.g. Branch A inside an OR integration). The two channels are now distinguished only by lifecycle (top-of-burst vs end-of-burst absorb, [I-21](../30_invariants.md#i-21)) and absorb status (`status=1` vs `status=3`) — not by struct type. The standalone `struct InternalMail` was deleted.

---

## The parallel-smashMail episode

Per memory file: a past session proposed parallelising the `smashMail` step (the mail-drain inner loop). The proposal was rejected. Reason: the determinism guarantees depend on mail-drain ordering matching the LB-iteration ordering; a parallel drain could reorder operations in ways that change whether a specific statement is admitted first in LB A or LB B.

This does not rule out a parallel grid — it rules out *one specific implementation strategy*. The RT campaign explores alternatives: per-LB lock-free mailbox, coarser cycle semantics with explicit commit barriers, etc.

---

## Class-level deferred-action collectors

Mail (`mailOut → smashMail → mailIn`) is the right channel for grid-broadcast cycle communication: a statement or implication or origin entry produced by one LB and intended for the LBs in its routing map (descendants/successors, per `parentChildrenMap`). It is not the right channel for **ancestor-direction** writes from a descendant LB. The mail routing goes the wrong way (parents are not in the descendant's routing map), and forcing ancestor writes through mail would require either a parallel mailbox infrastructure or a special-case routing pass.

The codebase's accepted pattern for ancestor-direction effects is the **class-level deferred-action collector**: a `std::vector<…>` field on `ExpressionAnalyzer` (not on `Memory`), guarded by a `std::mutex`, drained inside `proveKernel` after `pool.join` (single-threaded), in sorted order. Each entry typically holds `(emitter LB pointer, payload)`; the drain walks `emitter->parentMemory` and applies the payload to ancestors directly — race-free because no worker threads remain.

Existing collectors:

| Collector | Mutex | Producer | Drain action |
|---|---|---|---|
| `inductionMemoryBlocks` | `inductionMemoryBlocksMutex` | `addStatement` `isPartOfRecursion` branch ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) | `activateZeroCondition` (touches `parentMemory->simpleMap`) |
| `pendingAncestorOrigins` | `pendingAncestorOriginsMutex` | `addStatement` `primedForContradiction` branch ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) | walk `parentMemory` chain + `addOrigin(ancestor->exprOriginMap, …)` |
| `updateGlobalTuples` | `updateGlobalMutex` | `updateGlobal` calls from worker threads | apply to global theorem registry |
| `updateGlobalDirectTuples` | `updateGlobalDirectMutex` | `updateGlobalDirect` calls from worker threads | direct theorem emission |

Drain block lives in `proveKernel` after `pool.join` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)). Sort key per collector is chosen so that arrival-order variation across worker threads cannot affect the drain output. See [I-28](../30_invariants.md#i-28) for the rule, [D-39](../40_decisions.md#d-39) for the worked example.

---

## Weaknesses

### Known & tracked

- **Parallel smashMail rejected.** Any future attempt to parallelise must preserve the LB-iteration-order observation-equivalence; not trivial.

### Suspected fragility

- **Main-scope statement broadcast cost.** Every main-scope emission is pushed to every LB's `mailIn` — O(grid-size) cost per emission. For Peano-sized grids (≈60 LBs) this is fine; for FTA-era grids (expected much larger) it becomes a serialisation bottleneck at the mail-drain phase.
- **Implication broadcast is implicit.** New theorems proved during a cycle are broadcast as implications; it is not always clear which LBs *need* the implication. Currently over-broadcast (to everyone) trading memory for simplicity.
- **Origin-map merge.** Merging `mailOut.exprOriginMap` into recipient's `exprOriginMap` must preserve per-expression unique-dependency semantics. A duplicate-key handling bug would produce either lost origins (verifier fails on `origin` check) or bloated origin maps (verifier passes but slowly).

### Not exercised by tests

- **Cycle-boundary invariant.** No test explicitly asserts "a statement emitted in cycle N is NOT visible in cycle N to a non-emitting LB". The invariant is upheld by construction (LBs don't read each other's state intra-cycle), but a refactor that introduced shared state would violate it silently.
- **Mail drain ordering.** The drain order determines the order statements appear in recipient `mailIn`s. If two LBs produce the same statement in the same cycle, the recipient sees it twice with potentially-different origins. Deduplication happens at `addStatement`, but the *origin-merge* behaviour at that point depends on order.

---

## Open questions

- **OPEN-21 — RESOLVED.** `buildPerCoreMailboxes` at [`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp). Iterates `index` (parent→children map) to collect every recipient LB; for each recipient, assigns `std::vector<Mail>(cores)` where `cores = logicalCores`. With `logicalCores = 1` hardcoded, each recipient has exactly one mailbox slot. Type: `PerCoreMailboxes = unordered_map<Memory*, vector<Mail>>` at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). The name is historical — in the current single-thread build, it's effectively a per-LB mailbox, not per-core.
- **OPEN-22 — RESOLVED.** Mail-drain ordering is deterministic by **recipient `exprKey` sort**. `smashMail` at [`prover.cpp–5921`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) collects `(recipient, slots)` pairs and sorts them via `std::sort` with a comparator on `body->exprKey`. Then for each recipient in that order, merges statements (`set::insert` — order-independent within a recipient), implications (same), and origins (accumulated without duplicates via linear scan). So: inter-recipient ordering is `exprKey`-sorted; intra-recipient merge is set-based, hence commutative. This guarantees byte-reproducible mail drain regardless of the order `mailOut` was filled. A parallel implementation could preserve this invariant by sorting *after* collection, which is what the current code already does.

---

## See also

- [`20_core_concepts/01_logic_blocks.md`](01_logic_blocks.md) — the LB that owns `mailIn` + `mailOut`.
- [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — the cycle-driver `prove`.
- — parallelism history.
- — RT campaign roadmap including mail redesign considerations.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
