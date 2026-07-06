<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Mail system `[DRAFT]`

> **Id-form / mint-at-commit status (current, ).** All mail now carries INTEGER IDS, not `std::string`. Cross-LB routing mail (`mailIn` / `mailOut`, `RoutingColdMail`) carries GLOBAL `mailInterner` ids (an int32 `ColdStringTable` on the never-deloaded mail pool, never reset); internal mail (`ColdMail`) carries per-LB `NameMap` / `originInterner` ids. Value types: `IntMailStatementKey` (statements) + `IntMailOrigin` (origins) + `packOriginKey`; the string `MailStatementKey` / `EwvKey` / `IntEwv` are DELETED (`OriginLine` / `ExpressionWithValidity` survive only as the decoded boundary form). Design = **lock-free mint-at-commit**: the global `mailInterner` is FROZEN (decode / lookup only) during the parallel phase; the ONLY mint sites are single-threaded seams (the `proveKernel` commit barrier after `pool.join`, the load-time `broadcastTheorems`, the post-join `updateGlobal*` drains). `mailOut` holds SENDER `NameMap` / `originInterner` ids, `mailIn` + the committed blob hold GLOBAL ids — the asymmetry is translated at the commit seam. Every mail-inbox writer `ensureArena`s before its first insert. The `expandedImplications` column is DELETED. See [I-91](../30_invariants.md#i-91), [I-101](../30_invariants.md#i-101), [I-102](../30_invariants.md#i-102), `I-127` ([30_invariants.md](../30_invariants.md)), `G-55` ([50_gotchas.md](../50_gotchas.md)). **The `std::string` mail representation, the `HotMail` naming, and the `expandedImplications` field described below all predate this migration — read them for the delivery STRUCTURE, with the payload now id-form.**

> **Statification status (current, ).** `MailLog` is now **statified** onto a dedicated, never-deloaded **mail pool** ([D-141](../40_decisions.md#d-141), [I-95](../30_invariants.md#i-95)): its three internal containers (the per-LB batch blob runs, the recipient→ancestor lists, and the per-(recipient, ancestor) cursors) live on the mail pool's pages instead of heap `std::unordered_map`s. The `mailIn`/`mailOut` staging buffers have since moved onto a HOT arena ([I-101](../30_invariants.md#i-101)) and the two internal-mail channels onto the COLD deloadable path as `ColdMail` ([I-102](../30_invariants.md#i-102)) — see below. The public `registerLb`/`commit`/`pull`/`clear` surface, all delivery semantics, and the race-safety discipline are byte-preserved — only the storage moved. See [The pull model](#the-pull-model). The heap `std::` form the rest of this section describes is the prototype it replaced.

> **Pull-model status.** The push/broadcast routing this chapter historically described — `mailOut → sendMail → per-core boxes → smashMail → every descendant's mailIn` — has been **replaced by a pull model** ([D-137](../40_decisions.md#d-137)) and the push machinery (`sendMail`, `smashMail`, `boxes`, `buildPerCoreMailboxes`, `destroyMailboxes`, the `PerCoreMailboxes` typedef) **deleted**. Each LB now stores the mail batches it emits ONCE in its own append-only log (`MailLog` on `ExpressionAnalyzer`, [`mail_log.hpp`](../../GL_Quick_VS/GL_Quick/src/mail_log.hpp)); a receiver pulls un-ingested batches from its `parentMemory`-chain ancestors in `performElemPhase1`, tracking a per-(recipient, ancestor) cursor. `mailIn`/`mailOut` survive as per-LB staging buffers. Delivery, scopes, gates, and latencies are byte-preserved (see [The pull model](#the-pull-model) below); the D-76 / `Mail::implications`-deletion facts in the banner below still hold. Sections describing `smashMail` / per-core mailboxes are retained as historical context, flagged inline.

> **ASIC 0.1 reshuffle status (final branch state, 2026-05-20).** `Mail::implications` has been **removed** ([D-78](../40_decisions.md#d-78)); implications travel as the compact `(implication<N>[…])` form on `Mail::statements`, deposited by the D-76 deferred-compaction drain after `pool.join`, recovered receiver-side via `status=3` disintegration. The mail-absorb block was relocated post-fixpoint by commit and then **reverted to pre-fixpoint** on 2026-05-20 ([D-79](../40_decisions.md#d-79)) — the post-fixpoint experiment added two PKs of latency for the contradiction-LB rule arrival (one PK from the D-76 deferred broadcast plus one PK from post-hashburst absorb), which pushed `__contradiction__(=[a,b])` LBs past `MAX_NAME_IDS` during the Peano-incube combinatorial substitution phase before they could discharge. The pre-fixpoint position restored the same-burst absorb-fixpoint semantics; only the one PK from the D-76 deferred broadcast remains as residual latency vs main HEAD. The persistent per-LB fields `workingMemory` / `externalStatements` / `intExternalStatements` (added by commit ) are kept and now serve as same-burst staging: cleared and refilled by the pre-fixpoint absorb, read by the same burst's request-generation batches.

> LBs communicate between cycles via mail. This is what keeps per-LB execution in a single cycle race-free while still letting the grid as a whole converge — observations made by one LB this cycle become observable to others only next cycle.

---

## Why mail

Two design goals compete:

1. **Parallelism.** The LB grid is the architectural abstraction for distributed execution (current build: real worker threads over `logicalCores`; roadmap ASIC). Multiple LBs compute in parallel.
2. **Determinism.** Proof results must be reproducible. An LB must not observe a partial update to another LB's state, or race on shared data.

Mail resolves the tension by pinning all inter-LB observation to cycle boundaries. Within a cycle, each LB sees only its own state as it was at cycle start. At cycle end, outgoing messages are merged into recipients' inboxes; next cycle, they become observable.

---

## The `Mail` struct

Defined at [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp). Fields:

| Field | Type | Role |
|---|---|---|
| `statements` | `set<pair<ExpressionWithValidity, levels>>` | Expressions to be delivered — each becomes an `addStatement` call on the recipient. The `ExpressionWithValidity` (`memory.hpp::ExpressionWithValidity`, fields `original` and `validityName`) carries the install scope per element. For routing-channel traffic (`mailIn`/`mailOut`) the `validityName` is `"main"` by [I-26](../30_invariants.md#i-26) sender contract; per-LB `sameIterationInternalMail` traffic may carry non-main scopes. Migrated 2026-05-07 — was previously `set<pair<string, levels>>` with `"main"` hardcoded at the receiver. Compact-form implications `(implication<N>[…])` arrive here too (post-`Mail::implications` deletion). |
| `exprOriginMap` | `map<ExpressionWithValidity, list<Origin>>` | Provenance entries to merge into the recipient's origin map. |
| `expandedImplications` | `set<ExpressionWithValidity>` | Index of `(implication<N>[…])` compacts whose body was expanded into rules at the sender; the receiver merges this set so its own end-of-burst `sanitizeHashMemory` walk can rewrite-or-eradicate the same compacts when an equi-class downprioritizes their `it_`/`int_` args. |
| `~~implications~~` | ~~removed~~ | The `set<tuple<chain, head, args, levels, theorem>>` tuple channel was deleted. Implications now travel only as the compact `(implication<N>[…])` form via `statements` ([D-78](../40_decisions.md#d-78)). |

Every `Memory` has two routing-mailbox instances — `mailIn` / `mailOut` — now backed by a dedicated per-LB **HOT `LbArena`** rather than heap `std::set`/`std::map` (the `HotMail` struct: their three live containers — `statements`, `exprOriginMap`, `expandedImplications` — ride the cold-map family off the heap, never deload-registered; the fourth heap-`Mail` field `disintegrationSignals` is dropped, assert-guarded, since the routing channels never write it — [I-101](../30_invariants.md#i-101)). They are **per-LB staging buffers** for the pull model (no longer routed cross-LB): `mailOut` is the per-cycle outbound buffer that `fillMailOut` fills (via the hot write doors) and the commit barrier serializes directly (via `Codec<Mail>::serialize(const HotMail&)`, no `toHeapMail` snapshot) into the LB's `MailLog` log; `mailIn` is filled by the phase-1 pull from ancestors' logs and drained by the absorb reading the `HotMail` sorted snapshots directly (no transient heap `Mail`), then cleared by the caller. (Pre-pull-model they were the cross-LB routing inbox/outbox carried by `sendMail`/`smashMail`.)

In addition, every `Memory` carries **two per-LB internal-mail instances** for the cross-cycle revival path ([D-19](../40_decisions.md#d-19), with the two-channel split landing in the follow-up consolidation 2026-05-24 — see [D-90](../40_decisions.md#d-90)):

- **`sameIterationInternalMail`** — the ephemeral hashburst-output channel. Filled during the burst by `checkLocalEncodedMemoryStatic`'s rule firings and during the **pre-burst** `standardProcessing` call by `dischargeToBeProved`'s parent-scope emissions (routed via the `internalMailOut` parameter). Drained by the same step's **post-burst** `standardProcessing` call. Lifecycle: one elementary step. Was named `internalMailIn` until commit 2 of the follow-up consolidation; renamed mechanically to disambiguate from `nextIterationInternalMail`.
- **`nextIterationInternalMail`** — the cross-iteration deferral channel. Filled by the **post-burst** `standardProcessing` call's `dischargeToBeProved` (via `internalMailOut`), by `addExprToMemoryBlock`'s vacuous-truth path, and by the `updateGlobal`/`updateGlobalDirect` cross-LB deposit sites (each targeting the recipient block's own `nextIterationInternalMail`). Drained at the start of the next step's **pre-burst** `standardProcessing` call, alongside `mailIn`. Lifecycle: cross-iteration; carries one-step-delayed emissions.

**Storage (statification): both channels are COLD.** As the two internal channels are `ColdMail` (`memory_infra/cold_mail.hpp`) — reference aliases into `LbMemory::sameInternalMail` / `nextInternalMail` — on the LB's DELOADABLE per-LB arena, deload-registered at bases 455 / 505 and `survivesDischarge` ([I-102](../30_invariants.md#i-102)). They MUST survive deload: `nextIterationInternalMail` carries one-step-delayed emissions across the deload seam, and `sameIterationInternalMail` carries the end-of-burst `sanitizeHashMemory` rewrites (written after the post-burst clear) to the next step. `ColdMail` is the cold sibling of `HotMail`: same column types/codecs (`MailStatementKey` / `EwvKey` / `OriginLine`), but it CARRIES `disintegrationSignals` (the column `HotMail` dropped, here a `TypedColdMap<EwvKey, uint8_t>` of the two firing-time bools packed) and DROPS the always-empty-for-internal `expandedImplications`. Writers use the doors (`insertStatement` / `addMailOrigin` / `setDisintegrationSignal`); the absorb reads the cold columns DIRECTLY — the sorted snapshots (`sortedStatements` / `sortedOrigins`) plus the `getDisintegrationSignal` read door — while the sacred dump + tests materialize a transient heap `Mail` via `makeHeapMail`. Both re-impose the canonical `ExpressionWithValidity::operator<` order, so the absorb result and the dump bytes stay byte-identical to the former heap channel ([D-142](../40_decisions.md#d-142)). The one non-mirror, forced by cold storage: a cross-LB `nextIter` deposit `ensureLoaded`s the recipient and skips a `dischargedForever` target. The `Mail` struct itself stays in `memory.hpp` only as that transient snapshot — no persistent internal-mail heap allocation remains.

Both internal channels use the `Mail` struct (since 2026-05-07 — [D-53](../40_decisions.md#d-53), renumbered from main's D-46; pre-unification the integration-revival channel was a separate `struct InternalMail` with a `set<tuple<string, levels, validityName>>` `statements` shape). Revival emissions carry per-element scope in the EWV's `validityName` (typically non-main; e.g. Branch A inside an OR integration). Neither internal channel has an outbox counterpart — they are per-LB, produced and consumed within the same LB across cycle boundaries.

The two-channel split exists to disambiguate "this fires this step" (sameIter) from "this fires next step" (nextIter) — a single channel could not carry both lifecycles without subtle clear-ordering bugs. See the [Integration-revival channel](#integration-revival-channel) section below.

---

## The cycle-boundary protocol

1. **Cycle start (phase-1 pull, then pre-burst absorb).** `mailIn` starts empty; the **phase-1 pull** (`MailLog::pull`, see [The pull model](#the-pull-model)) fills it first — walking this LB's `parentMemory` chain and merging each ancestor's un-ingested batches (statements + exprOriginMap) into `mailIn`, then advancing the per-(recipient, ancestor) cursor. `nextIterationInternalMail` holds the previous step's cross-iter deferral emissions (vacuous-truth, post-burst discharge, cross-LB updateGlobalDirect deposits). `mailOut` is empty. `sameIterationInternalMail` is empty (was drained by the prior step's post-burst pass). The **pre-burst `standardProcessing`** call then runs:
 - Drains `mailIn` (status=3) into `intLocalEncodedStatementsDelta`. Status=3 disintegrates each entry: regular facts land in `overallHashMemory` + `externalStatements`/`intExternalStatements`; compact `(implication<N>[…])` forms recover their `(chain, head, …)` and install into `overallHashMemory` + `workingMemory`.
 - Drains `nextIterationInternalMail` (status=1) into the same delta. Status=1 routes through the full disintegration pipeline so rewritten constituents re-fire.
 - Merges `mailIn.exprOriginMap` and `nextIterationInternalMail.exprOriginMap` into `body.exprOriginMap` via `addOrigin` (D-49). **Equality origins** for positive 2-arg equality keys `(=[a,b])` are *additionally* synced into every `EquivalenceClass.equalityOriginMap` whose `validityName` matches and whose `variables` already contain both `a` and `b`. Without this sync the class state is incomplete: `mergeTwoEquivalenceClasses` (`prover.hpp::mergeTwoEquivalenceClasses`) consults class state when generating `equality2` cross-pair origin records and would treat mail-arrived equalities as not-yet-derived, emitting redundant transitive-closure records that can form cycles when distinct bridge variables (`commonArg`) generate mutually-pointing origins. Mailed equalities whose vars are not yet class-bound are picked up later via `prover.hpp::updateEquivalenceClasses` when their own absorption fires.
 - Merges `mailIn.expandedImplications` into `body.expandedImplications`. (`nextIterationInternalMail.expandedImplications` is always empty — no producer writes to it.)
 - Clears `mailIn.statements`/`exprOriginMap`/`expandedImplications` and `nextIterationInternalMail.statements`/`exprOriginMap` immediately after drain, before applying equivalence classes.
 - Applies equivalence classes (`applyEquiClasses`), discharges `toBeProved` with `internalMailOut = sameIterationInternalMail` (so OR-integration / NotOrScope parent-scope emissions land in sameIter and feed the same step's hashburst — zero-delay routing). Populates `mailOut.statements` and `mailOut.exprOriginMap` via `fillMailOut`. Clears `changedClassesThisStep`.
2. **Cycle body.** Request generation reads `workingMemory` (Batch 1), `externalStatements`/`intExternalStatements` (Batch 3/4 mail side), and the local containers (Batch 2/5, CE). The three per-step delta containers (`intLocalEncodedStatementsDelta`, `intLocalEncodedStatementsDelta`, `localHashMemoryDelta`) are cleared in a single unconditional pre-fixpoint site. The hashburst evaluation pass runs — a single pass over the static requests. Every rule firing's output is deposited into `body.sameIterationInternalMail` (single-channel write — the pre-2026-05-24 parallel direct `mailOut` deposit has been retired), absorbed by the post-burst `standardProcessing` rather than into `intKnownStatements`; the request skip-set is therefore fixed across the pass, so one pass reaches the fixpoint.
3. **Post-burst absorb.** `standardProcessing` runs a second time with `externalMailIn = nullptr`, `internalMailIn = body.sameIterationInternalMail` (hashburst output + same-step revival deposits), and `internalMailOut = body.nextIterationInternalMail` (so OR-integration / NotOrScope emissions during this absorb land in nextIter for next-step processing — the documented one-step-delayed contract). Same recipe as pre-burst: drain → clear → apply → discharge → fillMailOut → clear `changedClassesThisStep`.
4. **Cycle end (commit barrier).** At the single-threaded `proveKernel` post-`pool.join` seam — where `smashMail` used to consolidate — the **commit barrier** appends each LB's `mailOut` (statements + exprOriginMap) as one batch to its `MailLog` log, then clears `mailOut`. It iterates `bodies` with an empty-`mailOut` skip (so it also covers the root, which can deactivate). The `proveKernel` deferred-compaction drain (D-76) and the `updateGlobalDirect`/`updateGlobal` `"theorem"`-origin sends run *after* the barrier and append to the **root's** `mailOut` via `mergeBatchInto`, so they ride the **next** iteration's barrier — preserving their one-iteration-later (net two-cycle) delivery. `mailIn` is no longer written here.
5. **End of `proveKernel`.** *(retired)* The `smashMail` per-core consolidation is gone; the commit barrier in step 4 is now the single-threaded mail seam, and receivers pick up the committed batches via their own phase-1 pull next cycle.

Key property: **LB execution during cycle N cannot depend on any state produced by other LBs in cycle N.** The commit barrier (cycle end) plus the next cycle's phase-1 pull is the synchronisation point.

---

## The pull model

The cross-LB carrier is `MailLog` — a heap `std::` struct on `ExpressionAnalyzer` ([`mail_log.hpp`](../../GL_Quick_VS/GL_Quick/src/mail_log.hpp)), with two maps:

- **`batches`** — `unordered_map<const Memory*, vector<Mail>>`: each LB's append-only log of the mail batches it has emitted this execution batch. This is the **single stored copy** of everything that LB sends; descendants read it, no per-recipient copy is made.
- **`cursor`** — `unordered_map<const Memory*, unordered_map<const Memory*, size_t>>`: `cursor[recipient][ancestor]` = how many of that ancestor's batches the recipient has already ingested.

`MailLog` keys on `const Memory*` but never dereferences it (the caller supplies the ancestor list at registration; the pull then derives its ancestor set from the registered cursor row).

**Statified storage (current).** The two logical maps above are now five cold containers on the dedicated mail pool (the `ExpressionAnalyzer`-owned `mailArena{ &mailMemory }`, the "local memory manager" that draws blocks from the pool and hands pages to them). All five are pure RUNTIME containers — never in `LbMemory::visitContainers`, so never deloaded / dirty-tracked / reshuffled (the `intToBeProved` precedent); their `DirtyState` is never read. Keys are the `uintptr_t` of a `const Memory*` (`int64_t` for the POD-key maps), still never dereferenced.

The `batches` map is THREE containers forming **independent per-LB append-only logs** — the pool equivalent of the heap prototype's `unordered_map<Memory*, vector<Mail>>`, where each LB's batches grow independently so a commit is O(1) and never touches another LB's bytes:

- **`mailBlobPool`** — `PagedVector<char>`: every committed batch's `Codec<Mail>` blob (statements + exprOriginMap only, strings inlined), APPEND-ONLY — a written blob never moves.
- **`mailRefs`** — `PagedVector<BlobRef>` (`{uint32 start; uint32 len; int32 prev}`): one ref per batch, APPEND-ONLY; each LB's refs form a newest-first back-linked chain via `prev` (the pool form of one LB's `vector<Mail>`).
- **`mailHeads`** — `TypedColdMap<int64_t, MailHead>` (`{int32 lastRef; int32 count}`): producing-LB key → its chain head. `commit` appends the blob + a ref and bumps the head; `pull` reads the head and walks the `count − cursor` newest refs, decoding each STRAIGHT into the recipient's `mailIn` (`readBlobInto` → `Codec<Mail>::deserializeInto`, no transient heap `Mail`; the test `pull<Mail>` path keeps `readBlob` + `mergeBatchInto`) ([D-143](../40_decisions.md#d-143)).

> **Why not one `TypedColdBlobMap`.** The first cut stored the batches in a single `TypedColdBlobMap<int64_t, Mail>` whose CSR concatenates every LB's run into one byte pool; a commit to any non-tail LB then `memcpy`-shifts every later LB's bytes — superlinear, a measured massive slowdown (per-burst `dt` grew 7→14→18→27 s as the grid filled). The append-only pool + per-LB chain restores the heap's O(1) commit ([D-141](../40_decisions.md#d-141)); the per-burst `dt` then tracks the actual inference work, not the grid size.

The routing index is the other two containers:

- **`mailEdges`** — `ColdMultiMap<int64_t, int64_t>`: recipient → its ancestor-key list (set once at registration via `appendToTail`). It exists because a hash map cannot enumerate the cursor's keys by recipient prefix — it is what the pull walks to find a recipient's ancestors (the heap prototype read them from `cursor[recipient]`'s keys).
- **`mailCursor`** — `TypedColdMap<CursorKey, int32_t>`: the `cursor` map, flattened to a packed `(recipient, ancestor)` key. Every cell is pre-created at registration (`insert`), advanced by the parallel pull via `setValueAtRelaxed` (the disjoint no-dirty write).

The mail pool is never deloaded, so the pointer-valued keys and grant-order layout carry **no canonical-bytes obligation** — proof output stays content-deterministic via the downstream set-merge + the absorb's pre-fold origin sort, exactly as before ([I-103](../30_invariants.md#i-103) governs deload streams only, of which the mail log has none).

**Lifecycle.**
- **Register** (single-threaded, at `buildGrid`): every LB pre-creates `batches[lb]` and `cursor[lb][anc] = 0` for each ancestor on its `parentMemory` chain. Every LB — the root `&body` included — is in `permanentBodies` and none is born mid-run, so all keys exist before any parallel pull, and the pull only ever advances pre-existing cells.
- **Stage** (`fillMailOut`, unchanged): writes `body.mailOut` in phase 1 and phase 3.
- **Commit** (the cycle-end barrier, step 4): appends each LB's `mailOut` (statements + exprOriginMap) to its log via `MailLog::commit(const HotMail&)` — serializing the routing mailbox directly, no `toHeapMail` — then clears `mailOut`.
- **Pull** (`MailLog::pull` in `performElemPhase1`, immediately before the pre-burst absorb, gated `!ceFilteringActive && !parameters.compressor_mode`): for each registered ancestor, fold `batches[ancestor][ cursor.. end ]` into `body.mailIn` — decoded straight in via `readBlobInto` on the production `pull<HotMail>` path — then advance the cursor to the ancestor's log length.

**Why this is the dual of the old routing.** `buildParentChildrenMap` gave each sender its transitive descendants; `sendMail` pushed to all of them (both deleted with the push model). "Pull from every `parentMemory`-chain ancestor" reaches the identical sender→receiver set ([I-57](../30_invariants.md#i-57)). The *content* delivered to each receiver is unchanged — only the storage moved from O(descendants) copies to one.

**Dormant LBs.** A parked induction-zero LB never runs while dormant, so its cursors stay at 0. On wake (`activateZeroCondition`), its first phase-1 pull catches up its ancestors' whole logs at once — the same union the old accumulate-in-`mailIn` would have held, but stored once in the ancestors' logs instead of copied into the dormant LB's inbox.

### Three behaviour-preservation rules

1. **Load-time broadcast → store once.** `broadcastTheorems` (previous-batch externals + proved-theorem rules) and the `buildGrid` startup dispatch reach *all* LBs, not just one producer's descendants. They store the seed batch once in the **root's** log (every descendant pulls it on the normal walk) and **self-inject the root's `mailIn`** directly (the root has no ancestor to pull from). Single-threaded at load, before any pull, so the batch is in place for burst 1.
2. **D-76 two-cycle latency preserved.** The compact-implication drain and the `updateGlobalDirect`/`updateGlobal` `"theorem"`-origin sends run *after* the commit barrier, append to the root's `mailOut`, and ride the next barrier's commit — net two-cycle delivery, exactly the old post-`smashMail` timing ([G-49](../50_gotchas.md#g-49) contradiction-LB latency sensitivity unchanged). The per-step delta stays one cycle (committed at the seam it is produced, pulled next cycle).
3. **Field coverage.** Only `statements` + `exprOriginMap` ever reached `mailIn` (the old `smashMail` dropped `expandedImplications`); `mergeBatchInto` carries exactly those two. `disintegrationSignals` is never written to `mailOut` (internal-channel only), so it is naturally untouched. Copying the whole `Mail` would deliver `expandedImplications` cross-LB for the first time — a forbidden semantics change (Rule 8).

### Race-safety

Commits happen ONLY at the single-threaded post-join barrier, so the batch logs are frozen during the parallel phase-1 sweep. The pull does concurrent **reads** of the frozen `mailBatches` runs + `mailEdges` lists plus **one disjoint write per edge** to advance the recipient's own `mailCursor` cells (and writes its own `mailIn`), with no insert/rehash (every cursor cell pre-created) — race-free. See [I-94](../30_invariants.md#i-94).

**Statified race-safety (current).** The statified pull is race-free by the same discipline, now resting on three concrete properties: (1) the mail pool is never deloaded or compacted, so a vid resolves to a stable physical address for the whole run — a captured blob stays readable at any later pull; (2) every read accessor (`lookup` / `runLen` / `valueAt` / `recordAt` / `find` / the `Codec<Mail>` decode into a stack-local `Mail`) is pure — it touches no shared mutable state during the parallel phase; (3) the one write — the cursor advance — uses `setValueAtRelaxed`, which writes ONLY the recipient's own (disjoint) value slot and **skips the shared dirty-flag write** that would otherwise be a data race even on disjoint slots ([`09_static_memory.md`](09_static_memory.md) cold-map "Mutation beyond append"). The string-interning that earlier statified trials raced on is gone: strings are inlined into each blob (`Codec<Mail>`), so a pull never mints into a shared interner — it only decodes. The bug those trials actually died on — `PagedHashIndex::reset` silently overrunning its single directory page once the index passed `dirCap`, corrupting neighbouring arena pages — is fixed (the two-level directory + hard assert, [D-144](../40_decisions.md#d-144)); a forced-small-page unit test drives thousands of LB keys through the spilled index with correct delivery.

---

## Main-only gate for `Mail::statements`

`Mail::statements` is **MAIN-ONLY** by contract. Only expressions whose install validity is `"main"` are mailed; non-main expressions stay local to the deriving LB. The pre-existing companion gate on `Mail::implications` is retired with the channel itself (see [D-78](../40_decisions.md#d-78)) — implications now travel as compact `(implication<N>[…])` strings on `Mail::statements`, subject to the same gate.

Sender-side enforcement:

- **Statements (including compact implications)**: `addStatement` at `prover.hpp` (the `local` branch in `addStatement`'s body) gates on `validityName == "main"` before pushing into `memoryBlock.mailOut.statements`. The D-76 deferred-compaction drain in `proveKernel` also deposits the compact `(implication<N>[…])` form into `mailOut.statements` at `validityName == "main"` only (the queue holds main-scope implications only by the senders' gates above).
- **(Retired) Implications**: on main HEAD, `addExprToMemoryBlock`'s post-disintegration loop gated `mailOut.implications.insert` on `impValidity == "main" && allowedForMail(impStr, memoryBlock)`. The channel is removed on this branch ([D-78](../40_decisions.md#d-78)); the equivalent compact-form deposit goes through `Mail::statements`'s gate above.

Receiver-side enforcement:

- **`mailIn.statements`** absorb at `performElementaryLogicalStep` reads `validityName` from the EWV in each tuple's first element AND carries a per-element `assert(vName == "main")` as the runtime enforcement of I-26 for the statements channel. For routing-channel traffic the EWV always carries `"main"` by sender convention; the assert traps any sender that ever pushes non-main into `mailOut.statements` (it would propagate via `smashMail` into `mailIn.statements` and surface here). Non-`"main"` reaches a `Mail`-shaped absorb only through the separate `sameIterationInternalMail` drain block (in the same pre-fixpoint absorb region, `status=1`); that block has no main-only assert because revival traffic is validity-aware by design. Migrated 2026-05-07.

**Why the gate exists.** Hashmem rules are tied to the hypothesis stack at their install scope. A rule at `v=main_boundary_(impl24[…])` is conditional on the impl24 hypothesis being active; shipping it to another LB and reinstalling at `v=main` discards that conditionality and produces unsound rule firings. Receivers re-derive non-main rules from their own disintegration of the mailed v=main statements — yielding properly-scoped origins via `trackExpansionHistory` and properly-scoped admission via `addToHashMemory`.

**`mailOut.exprOriginMap` is NOT main-gated** — provenance entries for ALL scopes (main and non-main) flow through mail. Receivers' `body.exprOriginMap` (post-merge) carries the sender's full provenance graph at every validity it lived at, so the visualizer's `buildStack` walk can resolve cross-LB dependency chains even when the depending rule lives at a sub-scope.

History note. The implications-side gate landed with [D-34](../40_decisions.md#d-34) in 2026-05-02. Pre-D-34, `prover.cpp` only consulted `allowedForMail` (which gates on expression shape, not validity); non-main rules shipped silently and got reinstalled at `v=main` at the receiver, with origins keyed at the sender's deeper scope. The visualizer's firing-time lookup at `(rule, "main")` found no exprOriginMap entry and asserted (`buildStack: no origin found`). The crash was latent until D-34's `ordisMerge` unblocked the FTA-rung-1 §4.1 chain and reached the contradiction LB's saved-theorem visualization.

### Additive compact-implication deposit (ASIC 0.1 prep, [D-76](../40_decisions.md#d-76))

At every site that pushes a tuple onto `mailOut.implications` (the eight `updateGlobalDirect` / `updateGlobal` broadcast sites — both proven-theorem broadcast and the internal-implication-to-children path; the latter is automatic because `sendMail` / `smashMail` copy `mailOut.statements`), the prover *additionally* records that implication for deferred compaction via `recordPendingCompaction`; a single-threaded pass after `pool.join` (sorted `pendingCompactionQueue`) then compiles it to its compact `(implication<N>[...])` form via `compileImplicationToCompact` and deposits it as a `mailOut.statements` element at `validityName == "main"`, flushed per originating core via `sendMail` (compiling inline at the broadcast sites raced the global name-allocation state on the parallel contradiction path — [D-76](../40_decisions.md#d-76), [I-28](../30_invariants.md#i-28)) (consistent with the main-only `Mail::statements` contract — the implications channel is itself main-only by [I-26](../30_invariants.md#i-26), so "its respective namescope" is always `"main"` here). The existing implication tuple is left untouched — the deposit is purely additive.

Each such deposit also emits a paired `compilation` origin into `mailOut.exprOriginMap` keyed by the compact form, with the original expanded implication as the single antecedent (both at `"main"`). This origin doubles as the receiver's mandatory paired-origin record (the `mailIn.statements` absorb hard-asserts a paired origin under `trackHistory`, [D-45](../40_decisions.md#d-45)); it is gated on `parameters.trackHistory`, mirroring the adjacent `theorem`-origin idiom.

**Antecedent body is the binary's canonical reconstruction, not the queued `original`.** `compileImplicationToCompact` dedups alpha-equivalent inputs to the same `implication<N>` name; the binary stores only the first-seen body. Citing the queued `original` as the `compilation` row's `rest[0]` can produce a string that is alpha-equivalent to but not structurally identical with the binary's stored body (specifically: the inner `>[w_i,w_j]` group's binder list paired with body args in opposite order from the binder declaration). The verifier's `check_compilation` reconstructs from the binary's `elements` and compares modulo `_normalize_with_unchangeables`; the normalizer does not try binder permutations, so non-canonical citations are rejected. The drain must therefore look up `compiledExpressions[ce::extractExpressionUniversal(compactImpl)]`, split `elements` into `(key=elements[:-1], head=elements[-1])`, and pass `this->reconstructImplicationFullBind(key, head)` as the antecedent. See [I-52](../30_invariants.md#i-52), [D-81](../40_decisions.md#d-81), [G-47](../50_gotchas.md#g-47).

Non-functional by construction: `exprOriginMap` is process documentation, not a proof input ([I-44](../30_invariants.md#i-44)); the compact statement is a fresh synthetic `(implication<N>[…])` atom absorbed at `status=3` (no disintegration), is no existing rule's premise, no goal, and is never negated, so it cannot change which theorems prove. It is preparatory for ASIC 0.1. The new tag is validated by `check_compilation` (see [`08_proof_tags.md`](08_proof_tags.md#compilation) / [`../10_pipeline/08_verifier.md`](../10_pipeline/08_verifier.md)).

**Level set on the compact-form deposit MUST be empty.** The `pair<ExpressionWithValidity, std::set<int>>` inserted into `mailOut.statements` by the deferred-compaction drain has to carry `std::set<int>` (empty) for the level component, exactly as every retired `mailOut.implications.insert` site did. The receiver's `addExprToMemoryBlock(..., status=3, levels=<from mail>,...)` forwards this set into the recovered rule's `LocalMemoryValue::levels` via `addToHashMemory`, and at every subsequent firing the rule contributes its installation levels into the derived-statement union (see [`02_glossary.md::levels`](../02_glossary.md#levels) and [`02_hash_engine.md::Fire = emit`](02_hash_engine.md#the-query-fire-cycle)). A non-empty deposited set silently pollutes every derivation that ever fires against the recovered rule — the rule fires, the head is derived, but the discharge gate `allLevelsInvolved` rejects the promotion because `levels.size > memoryBlock.level + 1`. This swallowed 345 incube theorems before [D-77](../40_decisions.md#d-77) fixed the drain. See [I-51](../30_invariants.md#i-51) and [G-46](../50_gotchas.md#g-46).

---

## Hashburst mail-deposit honors ref's Site F dedup

The follow-up consolidation (commit 17) rewired the hashburst rule-firing site at `memory.cpp::checkLocalEncodedMemoryStatic` to deposit derived statements + their history lines to `sameIterationInternalMail` directly, instead of calling `addExprToMemoryBlock` (which used to be the path on ref). The motivation was the "single mailOut writer policy" — `fillMailOut` becomes the sole writer to outbound mail by draining the internal mail channel post-burst.

The unintended consequence: `addExprToMemoryBlock`'s Site F early-return at function entry (`prover.cpp::4141-4145` in ref) — which suppresses BOTH the statement insert AND any new origin write when the head is already known at any ancestor scope — is bypassed by the mail-deposit path. `addOrigin(sameIterationInternalMail.exprOriginMap, …)` only dedupes on (key, ENTIRE-origin-tuple), so every distinct (rule, premise) firing for the same head lands as a separate origin line. Over five iterations, common atoms like `(in[*,1])` hit the 30-origin cap; `(=[*,*])` atoms reach 21+; the rest is exponential `buildStack` candidate exploration at chapter export.

The fix lifts ref's Site F check into the mail-deposit site, gating both writes on the same ancestor-scope `intKnownStatements` scan that ref runs at `addExprToMemoryBlock` entry:

```cpp
bool alreadyKnown = false;
if (!parameters.compressor_mode) {
    const int16_t origId = memoryBlock.nameMap.encode(rplExpr2);
    const int16_t valId  = memoryBlock.nameMap.encode(expressionListValidityName);
    for (int16_t anc : memoryBlock.nameMap.ancestorsOf[valId]) {
        if (memoryBlock.intKnownStatements.count(packStatementKey(origId, anc))) {
            alreadyKnown = true;
            break;
        }
    }
}
if (!alreadyKnown) {
    /* existing sameIterationInternalMail.statements.insert + addOrigin block */
}
```

Compressor mode keeps multiples — same `!compressor_mode` carve-out ref uses.

**Companion fixes (across commits in 2026-05-25 — "runs through 5 iterations" milestone and the subsequent Peano-main fix).** Four smaller pieces carry the documentation-only chain that the delta loop alone misses:

1. `trackExpansionHistory` (in `disintegrateExprCore2`) — paired `mailOut.exprOriginMap` writes at both the expansion-origin and disintegration-origin sites in `prover.cpp::disintegrateExprCore2`. Ships expansion-conjunction histories so the chain in receivers' `exprOriginMap` is complete after one mail round.
2. LB-creation paired writes at the three `addExprToMemoryBlock(... task formulation...)` sites in `addTheoremToMemory` (chain-walk LB, reformulated-contradiction LB, standard-contradiction LB). Pushes each LB's own task-formulation origin into its `mailOut` at LB creation time so the buildGrid dispatch carries them.
3. `buildGrid` runs `sendMail` over `permanentBodies` before the existing `smashMail(boxes)` — dispatches the LB-creation paired writes into the per-core mailbox slots so `smashMail` has content to drain into descendants' `mailIn` before step 1 begins.
4. `prepareIntegrationCore2` paired writes at all seven local `mb.exprOriginMap` addOrigin sites in `prover.hpp::prepareIntegrationCore2` — Case A `implication-expansion` + `premise-element`, Case OR `or-branch-goal` + `or-branch-assumption`, Case B `expansion-integration` + `iiv` + `iivHash`. Ships the integration-instruction-related documentation rows so descendant LBs that fire the integration rule via hashburst (and cite the integration-instruction string as origin dep) have the key entry available in their own `exprOriginMap` for `buildStack`'s chain walk at chapter export. Restoration of this set fixed the `(>[pi_lev_0_1](in[pi_lev_0_1,u_1])(>[](in2[pi_lev_0_1,u_6,u_3])(existence2[u_1,u_6,u_3])))` `buildStack` crash on Peano main. See [D-91](../40_decisions.md#d-91).

The big conjunction example below — the canonical `expansion of (NaturalNumbers[1,2,3,4,5])` documentation-only entry — is addressed by item 1 (the paired write at the expansion-origin site in `trackExpansionHistory`).

---

### Documentation-only chain example

The canonical case is the big conjunction produced by `expansion` of a compact form like `(NaturalNumbers[1,2,3,4,5])`:

```text
(NaturalNumbers[1,2,3,4,5])
  expansion →  (&(&(&…(in[2,1])(fXY[3,1,1]))(implication4[1,2,3])…(implication17[1,3,5,4])))
                  | disintegration ⇣ (one per element)
                  ├─ (in[2,1])
                  ├─ (fXY[3,1,1])
                  ├─ (implication4[1,2,3])
                  ├─ …
                  └─ (implication17[1,3,5,4])
```

The compact `NaturalNumbers[…]` and the disintegration products (`(in[2,1])`, `(implication4[1,2,3])`, …) are real statements: they enter `intEncodedStatements` + `intLocalEncodedStatementsDelta` and are shipped by the delta loop. The big conjunction is *not* a statement — `disintegrateExprCore2`'s `trackExpansionHistory` lambda only records `expansion | (NaturalNumbers[…])` in `exprOriginMap` and `disintegration | <conjunction>` in each child's origin. The conjunction is referenced as a dep by every child but never added to any delta channel.

If the conjunction's expansion-origin line is **not** shipped to children, `buildStack` (chapter export, `visualizer.cpp::buildStack`) walks bottom-up from each child, follows the disintegration dep to the conjunction, looks it up in the child's `exprOriginMap`, finds no entry, and asserts. This bit on 2026-05-25: both `__contradiction__(=[2,6])` (the diagnostic-trap target) and `__contradiction__(in2[10,10,3])` (the actual crash site) had every atomic disintegration product in `intEncodedStatements` + an origin entry citing the conjunction, but zero entries for the conjunction itself.

**The shipment contract — transitive walk in `fillMailOut`.** For each delta entry that fillMailOut ships, the function (a) writes the entry's own origin lines to `mailOut.exprOriginMap`, and (b) walks every dep referenced by those origins via `shipExprHistoryTransitively`. The walk is recursive:

```text
shipExprHistoryTransitively(dep):
    if dep already in mailOut.exprOriginMap:  return        # done — also breaks cycles
    if dep not in this LB's exprOriginMap:
        # Ancestor-only deps (e.g. child contradiction LB's anchor-handling
        # cites the original anchor whose task-formulation lives in the
        # parent AnchorIncubator LB): the ancestor's own fillMailOut ships
        # the entry to all its descendants — a superset of this LB's
        # descendants — so re-shipping is redundant.
        for each ancestor LB on chain to root:
            if dep in ancestor.exprOriginMap:  return       # ancestor will ship
        assert                                              # bug — Rule 19
    ship every origin line of dep
    for each sub-dep of each shipped origin:
        shipExprHistoryTransitively(sub-dep)
```

The early-return on "already in mailOut" is the dedup mechanism — each documentation-only key walks exactly once per `fillMailOut` call regardless of how many delta entries cite it. The mid-step `addOrigin(mailOut, dep, …)` runs *before* the recursive sub-dep calls, so a cycle (dep → … → dep) hits the early return on its second visit. The assert is per Rule 19: a dep cited by a shipped origin must have its own entry in the LB's `exprOriginMap` — silent skip would re-create the symptom this fix exists to remove.

**Coverage.** The walk picks up every link in the chapter chain that lives only in `exprOriginMap`, including multiple levels of nested expansion (e.g. `AnchorIncubator` → anchor-level conjunction → contains `NaturalNumbers` → expands into NaturalNumbers conjunction → disintegrates into atomic facts). Each level's expansion-conjunction is reached transitively from the atomic-fact delta entry that cited it.

**Cost.** Bounded by the visited set (the mailOut-presence check). Worst case the walk visits every key in `exprOriginMap` once per `fillMailOut` call. For the IncubatorPeano 5-iter run, the contradiction LB's `exprOriginMap` has ~4 000 entries; the walk's runtime contribution is in the noise vs the hashburst fixpoint itself.

**Why not a separate `localExprOriginDelta`.** An alternative is a sister-delta tracking new `exprOriginMap` keys per step, paralleling `intLocalEncodedStatementsDelta`. This would let fillMailOut iterate both deltas without recursion. Rejected because it adds bookkeeping at every `addOrigin(memoryBlock.exprOriginMap, …)` site (~30 call sites) for no behavioural improvement — the transitive walk reaches the same set of keys for free, and is local to `fillMailOut`.

See [I-56](../30_invariants.md#i-56) and [D-92](../40_decisions.md#d-92).

---

## Proven-theorem rule delivery — receiver-local head materialisation, and the residual one PK of latency on this branch (G-48)

Once a theorem `(>[bound](Anchor[bound])head)` is proved (typically by a contradiction LB's discharge calling `updateGlobalDirect`), the prover does **not** emit `head` as a statement at the proving site. What is broadcast is the **rule** — the full implication — and it reaches every receiver whose hashMemory the rule belongs in. Each receiver then **materialises the head locally** at its own hashburst by firing the rule against its in-scope antecedents. There is no "head-as-statement" mail mechanism — `head` only ever enters a receiver as the local derivation of `addExprToMemoryBlockKernel` (`status=0/1`, `isLocal=true`).

### Reference path *(historical — the push-model flow this comparison was written against; `sendMail` / `buildParentChildrenMap` / the per-core mailboxes are all deleted)*

1. **Rule broadcast (proving LB's PK).** The proving LB calls `ExpressionAnalyzer::updateGlobalDirect(theorem, coreId)` (`prover.cpp::updateGlobalDirect`). After disintegrating `(chain, head)` and doing a local `addStatement(head, *memoryBlockR, false, …)` at the chain-resolved ancestor (book-keeping; `local=false` so no mail emission), the function pushes the implication tuple onto `mailOut.implications` and calls `sendMail(this->body, mailOut, …)`. `sendMail` looks up `this->body` in the routing index; `buildParentChildrenMap` (`prover.cpp::buildParentChildrenMap`) computes the **transitive descendants** of each LB, so the entry for `this->body` covers every reachable LB in the grid. The tuple lands in every descendant's per-core mailbox slot.
2. **Receiver-side absorb (each descendant's next PK, pre-hashburst).** Every LB whose `mailIn.implications` received the tuple absorbs it at the top of its burst by calling `addToHashMemory(chain, head, …, "implication", true, orImpl, "main")`. The rule is now in the receiver's own `overallHashMemory`.
3. **Head materialised by receiver's hashburst (same PK as absorb).** The hashburst fires the new rule because the antecedent `(Anchor[..])` is in scope by anchor inheritance. The head reaches `addExprToMemoryBlockKernel` with `status=0/1` → `isLocal=true` → `addNegatedEquality`/`addEquality`/etc. with `local=true`. The local branch inserts into both `intLocalEncodedStatementsDelta` AND `memoryBlock.mailOut.statements` at `validityName == "main"` (`prover.cpp::addEquality`, `addNegatedEquality`).
4. **Side-effect: the head also reaches the receiver's children via regular sends.** Each receiver's `sendMail(receiver, receiver.mailOut, …)` at end of that PK delivers the locally-derived raw `head` to the receiver's own children. So a deeper-tree descendant gets `head` *twice* — once via its own rule-fire at the same PK, and once via its parent's `mailOut.statements` at the next PK. The first delivery is the load-bearing one for discharge.

For `__contradiction__(=[7,10])` under `(AnchorIncubator[..])` on ref's burst 3, this is exactly what fires: the implication tuple for `head=!(=[2,6])` arrives in its `mailIn.implications`; its own hashburst at burst 3 absorbs the rule and fires it locally; `!(=[2,6])` lands in its `intEncodedStatements`; the assumed `(=[7,10])` propagates `7↔10` across the existence/typing chain to produce `(=[2,6])` somewhere; X∧¬X reached; the `primedForContradiction` discharge sets `isActive = false` by end of burst 3.

### deltas — one PK of residual latency

The branch deletes `Mail::implications`. The rule no longer travels as the inline implication tuple. The proving LB pushes the original implication string onto the deferred-compaction queue via `recordPendingCompaction` ([D-76](../40_decisions.md#d-76)); a single-threaded drain after `pool.join` (still inside the proving PK) compiles the compact form and deposits `(implication<N>[…])` into `mailOut.statements`. The drain runs after the parallel workers, but the mailbox slots fill late enough that the next `smashMail` consolidation lands the compact form in receivers' `mailIn.statements` **one PK later** than the inline implication tuple would have arrived on main HEAD. On the reshuffle data, `(implication46[])` arrives at the contradiction LB's burst 4 `mailIn.statements`; on ref the equivalent tuple arrived at burst 3 `mailIn.implications`. This one PK of latency is unavoidable as long as compact-form compilation is deferred for I-28 safety.

The receiver-side absorb runs **pre-fixpoint** on this branch (post-fixpoint experiment was reverted on 2026-05-20 — see [D-79](../40_decisions.md#d-79)). So once the compact form arrives at the receiver's `mailIn.statements` at burst K, it is absorbed at the top of burst K via `addExprToMemoryBlock(... status=3,...)`, disintegrated back into `(chain, head, …)`, and installed in `overallHashMemory` + `workingMemory` before the burst's request generation runs. The rule then fires in the same burst K's hashburst — no additional PK of latency from the absorb position itself.

For `__contradiction__(=[7,10])` on the reshuffle: `(implication46[])` arrives at burst 4 mailIn.statements, pre-fixpoint absorb at burst 4 recovers `!(=[2,6])` as a rule, burst 4 hashburst fires it locally, the LB discharges by end of burst 4. The Peano-incube nameMap overflow at burst 5 (which existed when the absorb was post-fixpoint) is gone with this branch state. The remaining one PK of delay vs ref (burst 4 discharge here vs burst 3 on ref) is acceptable for Peano-incube; whether it stays acceptable at FTA scale is open.

The user's design intent (Generative Logic, 2026-05-20): the head is materialised **only** by local rule-firing in each receiver's own hashburst; there is no separate "head-as-statement" delivery to add. The fix landed for the two-PK-latency case is the absorb-position revert; the residual one PK from the D-76 deferred broadcast is a separate decision and a separate fix if it bites at FTA.

---

## Who receives what

The delivery rules depend on scope and channel:

- **`mailOut.statements`** — main-only by contract (see above). Committed once to the sender's `MailLog` log at the cycle-end barrier; **pulled** by every descendant (each walks its own `parentMemory` chain). Each recipient adds the statement to its own `wholeExpressions` and runs the usual admission logic. Compact-form implications travel here too.
- **(Retired) `mailOut.implications`** — channel deleted on this branch ([D-78](../40_decisions.md#d-78)). The compact-form deposit on `mailOut.statements` is the sole carrier for implication content.
- **`mailOut.exprOriginMap`** — carries entries for ALL scopes per the [`trackExpansionHistory`](../../GL_Quick_VS/GL_Quick/src/prover.cpp) invariant. Propagates along the same routing paths as statements/implications and merges into the recipient's `exprOriginMap` at the same keys.
- **Statements at non-main scopes** never enter `mailOut`. They stay local to the deriving LB. If a sibling/child needs the same fact, it re-derives via its own disintegration of the mailed v=main statements.

Under the pull model the routing backbone is the receiver's `parentMemory`-chain walk in `MailLog::pull`, not a precomputed descendant map. `buildParentChildrenMap` survived the push model's deletion for a while as a vestige (its product was threaded through the compressor and CE-filter call shapes but never read) and is now deleted outright (2026-07-03, with `indexCE` and the `ParentChildrenMap` alias); its transitive-descendants set was the exact inverse of the ancestor walk, which is why delivery was unchanged when the pull model replaced it.

The routing constraint is normative: see [I-57](../30_invariants.md#i-57). Every expression that lands in an LB's `mailIn` was authored by one of that LB's direct `parentMemory`-chain ancestors. A debugger looking for the producer of a malformed mail item can therefore enumerate exactly the ancestor chain — no sibling-subtree search needed.

---

## Per-core mailboxes *(historical — deleted with the push model)*

The push model gave each recipient one mailbox slot per worker core (`boxes` = `PerCoreMailboxes`, `unordered_map<Memory*, vector<Mail>>`, built by `buildPerCoreMailboxes`), so a worker could write its slot lock-free during the parallel phase; `smashMail` then consolidated the slots into each recipient's `mailIn` after the join. **The pull model removes all of this** — there is no outbound copy to a recipient, so no per-core slot and no consolidation. Worker threads still run `proveKernel`'s phases over the shared `bodies` vector (`logicalCores` threads, `next.fetch_add(1)` dispatch); the parallel-phase race-safety now rests on the commit-at-seam / read-frozen-logs discipline ([I-94](../30_invariants.md#i-94)) rather than per-core slots. The non-determinism investigation that produced [D-39](../40_decisions.md#d-39) (a worker walking into another LB's `exprOriginMap`) is still addressed by the deferred-action collector pattern below.

---

## Integration-revival channel

`sameIterationInternalMail` is the second inbox introduced with the `rejectedMapIntegration` mechanism ([D-19](../40_decisions.md#d-19)). It exists so that integration-side rejection recovery does not have to route through the grid-broadcast `mailOut → smashMail → mailIn` pipeline.

**Producers.**

- `applyEquivalenceClassToRejectedMapIntegration` — on an equivalence-class rewrite that produces a marker-form matching an existing `admissionMapIntegration` or `admissionSetIntegration` entry, emits the rewritten concrete constituent + siblings via `emitIntegrationRevivalToInternalMailIn`.
- `revisitRejectedIntegration2` — fired at `admissionMapIntegration` insert (`prover.hpp`, with u_-strip on the key) and at `admissionSetIntegration` insert (`prover.cpp`, key already bare). Emits any matching `rejectedMapIntegration` entries.

**Consumers.** The hashburst entry absorbs `sameIterationInternalMail` at the top, immediately before the legacy `mailIn` absorb, with `status=1` (full disintegration pipeline — unlike legacy `mailIn` which uses `status=3` to skip disintegration; see [D-19](../40_decisions.md#d-19) rationale).

**Lifecycle asymmetry.** Legacy `mailIn` is cleared at the end of each hashburst (`prover.cpp`) — inserts come from between-cycle routing. `sameIterationInternalMail` is cleared at the **top** of each hashburst, immediately after absorb, because its inserts happen *during* cycle body (from eq-class rewrites and admission-key revisits triggered inside `addStatement`). See [I-21](../30_invariants.md#i-21).

**Scope.** As of 2026-05-07 ([D-53](../40_decisions.md#d-53) unification, renumbered from main's D-46), `Mail::statements` is `set<pair<ExpressionWithValidity, levels>>` carrying scope per element, and the integration-revival channel `Memory::sameIterationInternalMail` uses the same struct. `mailOut.statements.insert` is still gated on `validityName == "main"` at the routing senders (so the EWV always carries the literal `"main"` for routing traffic), and the receiver absorb reads `validityName` from the EWV. `sameIterationInternalMail` senders push EWVs with the actual revival scope (typically non-main; e.g. Branch A inside an OR integration). The two channels are now distinguished only by lifecycle (top-of-burst vs end-of-burst absorb, [I-21](../30_invariants.md#i-21)) and absorb status (`status=1` vs `status=3`) — not by struct type. The standalone `struct InternalMail` was deleted.

---

## The parallel-smashMail episode *(historical — smashMail deleted)*

`smashMail` is gone, so the old caution against parallelising its drain is moot. The lesson it taught is exactly what the pull model adopts: a single-threaded **commit barrier** at the cycle-end seam plus per-LB pulls — the "explicit commit barriers" the RT campaign had flagged as the sound alternative to a parallel drain.

---

## Class-level deferred-action collectors

Mail (now the pull model — each LB's log, pulled by its descendants) is the right channel for **descendant-direction** cycle communication: a statement or origin entry produced by one LB and consumed by the LBs that pull from it (its descendants). It is not the right channel for **ancestor-direction** writes from a descendant LB — a descendant's ancestors do not pull from it. Forcing ancestor writes through mail would require a separate upward channel.

The codebase's accepted pattern for ancestor-direction effects is the **class-level deferred-action collector**: a `std::vector<…>` field on `ExpressionAnalyzer` (not on `Memory`), guarded by a `std::mutex`, drained inside `proveKernel` after `pool.join` (single-threaded), in sorted order. Each entry typically holds `(emitter LB pointer, payload)`; the drain walks `emitter->parentMemory` and applies the payload to ancestors directly — race-free because no worker threads remain.

Existing collectors:

| Collector | Mutex | Producer | Drain action |
|---|---|---|---|
| `inductionMemoryBlocks` | `inductionMemoryBlocksMutex` | `addStatement` `isPartOfRecursion` branch ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) | `activateZeroCondition` (touches the parent's `SimpleMapStore` edges) |
| `deferredAncestorAdmissions` | `deferredAncestorAdmissionsMutex` | `updateAdmissionMap3`'s strict-ancestor admission seed, staged when `g_inParallelWorkerPhase` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)) | `drainDeferredAncestorAdmissions` — replays `updateAdmissionMap` on the pre-resolved ancestor ([D-187](../40_decisions.md)) |
| `updateGlobalTuples` | `updateGlobalMutex` | `updateGlobal` calls from worker threads | apply to global theorem registry |
| `updateGlobalDirectTuples` | `updateGlobalDirectMutex` | `dischargeToBeProved` (non-recursion-proven) **and** `dischargeContradiction` (incubator branch) — both enqueue rather than call `updateGlobalDirect` inline ([D-145](../40_decisions.md#d-145)) | direct theorem emission |

Drain block lives in `proveKernel` after `pool.join` ([`prover.cpp`](../../GL_Quick_VS/GL_Quick/src/prover.cpp)). Sort key per collector is chosen so that arrival-order variation across worker threads cannot affect the drain output. See [I-28](../30_invariants.md#i-28) for the rule, [D-39](../40_decisions.md#d-39) for the worked example.

The former `pendingAncestorOrigins` collector — the [D-39](../40_decisions.md#d-39) deferred ancestor-origin queue, which walked `emitter->parentMemory` and wrote `addOrigin(ancestor->exprOriginMap, …)` into every ancestor — was **retired by [D-51](../40_decisions.md#d-51)**, which localised the contradiction record to the `__contradiction__` LB (mail flows parent → children only); its struct, field, mutex, and drain loop are all gone.

---

## Weaknesses

### Known & tracked

- **Statification-phase concurrency (RESOLVED).** The statified form (`MailLog` on the mail pool) is race-free without a lock — see [The pull model → Statified race-safety](#race-safety). The earlier trials' "cold-blob deserialize race" was a symptom of the `PagedHashIndex` single-directory overflow corrupting neighbouring pages (now fixed by the two-level directory); the genuine concurrency design is: frozen-during-parallel-phase logs (commits only at the seam), pure read accessors, inlined blob strings (no shared interner mint at pull), and the disjoint no-dirty `setValueAtRelaxed` cursor advance. No shared mutex. The forced-small-page unit test exercises the spilled-index path under correct delivery.

### Suspected fragility

- **Per-cycle broadcast STORAGE cost — fixed.** The push model copied every main-scope emission into every descendant's `mailIn` (O(grid-size) stored copies per emission), the FTA-scale bottleneck. The pull model stores each emission once in the producer's log; this weakness is resolved (the point of [D-137](../40_decisions.md#d-137)).
- **Empty-batch skip relies on `mailOut` emptiness.** The commit barrier skips an LB whose `mailOut` is empty (to avoid empty batches). This is correct only because the sole writers of a non-root LB's `mailOut` are `fillMailOut` (per-step delta). A future writer that put content into a non-root LB's `mailOut` outside `fillMailOut` would still be committed (non-empty → committed), but the assumption is worth stating.
- **Over-pull to all descendants.** A proven rule is available to every descendant (each pulls the root's log), not just those that need it — memory traded for simplicity, as before.
- **Origin-map merge.** Merging an ancestor batch's `exprOriginMap` into the recipient's `mailIn` (then into `body.exprOriginMap`) must preserve per-expression unique-dependency semantics. A duplicate-key handling bug would produce either lost origins (verifier fails on `origin` check) or bloated origin maps (verifier passes but slowly). `mergeBatchInto`'s `std::find` dedup mirrors the old `smashMail` merge.
- **Mail-payload history-record order is not pinned (open, user-accepted 2026-06-10).** A dump-on A/B against main head from the same starting disk state — Gauss-induction-LB trap, 56 dump blocks, 227 MB traces — showed the identical line multiset but FOUR lines swapped, all inside `-- mailIn.exprOriginMap`: the same two history records for one statement (an `implication` origin and a `disintegration` origin) shipped in opposite order in 2 blocks. Every artifact (theorems, raw + processed chapters, verifier output, hash-burst lines) and every receiver-side `exprOriginMap` section stayed byte-identical — `addOrigin`'s dedup absorbs the merge-order difference. Root cause not identified: all membership gates migrated by [D-128](../40_decisions.md#d-128) are set-equal by construction, so the suspect is a subtler interaction in `fillMailOut`'s per-key shipping order. Evidence preserved in  / `hashburst_gauss_main.txt` (local). If this resurfaces: retarget the dump at the parent LB (`(in2[9,10,3])` chain) and diff its `mailOut` population at the source.

### Not exercised by tests

- **Cycle-boundary invariant.** No test explicitly asserts "a statement emitted in cycle N is NOT visible in cycle N to a non-emitting LB". The invariant is upheld by construction (LBs don't read each other's state intra-cycle), but a refactor that introduced shared state would violate it silently.
- **Pull order across ancestors.** `MailLog::pull` iterates the recipient's ancestors in `cursor` (`unordered_map`) order, so the order origins land in `mailIn.exprOriginMap` is unspecified. This is washed out downstream: statements merge set-wise, and the absorb sorts each origin vector before the capped `addOrigin` fold, so the persistent `exprOriginMap` is order-independent. The determinism anchor is that pre-fold sort, not the pull order.

---

## Open questions

*(OPEN-21 / OPEN-22 below describe the retired push carrier — `buildPerCoreMailboxes` and `smashMail`, both deleted. Kept as historical record of the design they resolved.)*

- **OPEN-21 — RESOLVED (push model, now retired).** `buildPerCoreMailboxes` iterated `index` to give each recipient `std::vector<Mail>(logicalCores)` slots, so a worker wrote its own slot lock-free during the parallel phase; `smashMail` consolidated the per-core slots into each recipient's `mailIn` after the join. The pull model replaces this with one log per LB + per-LB pulls (no per-core slots).
- **OPEN-22 — RESOLVED (push model, now retired).** `smashMail`'s drain was deterministic by recipient `exprKey` sort + set-based intra-recipient merge. The pull model's determinism instead rests on set-based statement merge + the absorb's pre-fold origin sort (see Weaknesses → *Pull order across ancestors*).

---

## See also

- [`mail_log.hpp`](../../GL_Quick_VS/GL_Quick/src/mail_log.hpp) — the `MailLog` struct (`batches` + `cursor`, `register`/`commit`/`pull`/`mergeBatchInto`).
- [`40_decisions.md#d-137`](../40_decisions.md#d-137) — the pull-model decision.
- [`30_invariants.md#i-94`](../30_invariants.md#i-94) — the commit-at-seam / frozen-logs race-safety rule.
- [`20_core_concepts/01_logic_blocks.md`](01_logic_blocks.md) — the LB; `mailIn`/`mailOut` are now per-LB staging.
- [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — the cycle-driver `prove`.
- — RT campaign roadmap including mail redesign considerations.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
