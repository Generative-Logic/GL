<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Invariants `[DRAFT]`

> Numbered, named, cross-referenced. Every invariant in this document has:
>
> - A stable anchor (`#i-N`) for cross-linking from chapter weaknesses sections and from [`AGENT_SwDD.md`](AGENT_SwDD.md#invariant-quick-reference).
> - **Scope** — what code paths it applies to.
> - **Rule** — the exact statement.
> - **Why** — the past incident, semantic constraint, or design commitment that motivates it.
> - **Spot** — the symptom(s) that indicate a violation.
> - **Fix** — the correct pattern to restore the invariant.
> - **Code** — the relevant file + symbol references.
>
> Invariants are numbered in the order they entered the document; the numbering is a stable identifier, *not* a priority ordering.

When you (future agent) add an invariant, append a new `## I-N` section, update the quick-reference table in [`AGENT_SwDD.md`](AGENT_SwDD.md#invariant-quick-reference), and do **not** renumber existing invariants — the numbers are referenced from external places (memory files, commit messages, code comments).

---

<a id="i-153"></a>
## I-153 Every cross-LB deposit into an LB sets the recipient's `hasWork` (or is reachable by `mailPeek`); the quiescent-skip predicate reads only never-deloaded logical state


**Scope.** The quiescent-burst skip ([D-194](40_decisions.md#d-194)): the `Memory::hasWork` latch, the `performElemPhase3` SLEEP detector, the wake doors in `prover.cpp`, and `MailLog::mailPeek`.

**Rule.** Two obligations hold jointly:

1. **Wake-door completeness.** Every site that deposits a statement, rule, admission entry, origin, or `nextIterationInternalMail` item into an LB OTHER than via that LB's own swept burst MUST set the recipient's `hasWork = true` — OR the recipient must be reachable by `mailPeek` (mail committed to the log, polled pull-side). A missed wake is a missed theorem (unsound); a redundant one is harmless. The direct-write doors: the two `updateGlobalDirect` deposits, `updateGlobal`'s proven-head deposit + goal erase, `drainDeferredAncestorAdmissions` (and the inline `updateAdmissionMap3` twin), `broadcastTheorems`' root self-inject (a `mailIn` write — a genuine burst input), and `activateZeroCondition` (case F). The `mailPeek`-covered doors: the commit-barrier log commits and the deferred-compaction drain into the root's `mailOut`. **A write to a recipient's `mailOut` is NOT a wake obligation:** `mailOut` is outgoing-only (I-64) — no burst path derives local state from it, and the barrier's commit sweep iterates `bodies` regardless of sweep status, so its content ships whether or not the owner sweeps. (Once committed, it wakes the DOWNSTREAM recipients via `mailPeek`; the owner itself needs no sweep for that to happen.)

2. **Over-approximation + no residency read.** The SLEEP detector clears `hasWork` only on a provable no-op burst (statement count unchanged vs the phase-1 baseline, `mutatedThisBurst` false, both INTERNAL mail channels — `sameIterationInternalMail` / `nextIterationInternalMail`, the inputs the LB's own next burst absorbs — empty) — when in doubt it stays dirty (the I-70 / I-79 discipline). `mailOut` is deliberately NOT a SLEEP term: outgoing-only content is not activity, and the post-join drains park proven-theorem mail on the ROOT's `mailOut` after the barrier, which false-fired the shadow check at the root until both `mailOut` terms were removed (see D-194). The skip predicate reads `hasWork` (LB slab, I-109), `mailPeek` counts (mail pool, I-101), and `isActive` — all never-deloaded — and NEVER `resident` / `blocksInUse`.

**Why.** The skip is byte-identity-safe only if each skipped burst is a genuine no-op and each wake fires the same iteration the reference would act. Missing a wake drops a theorem; reading residency makes the decision timing-dependent, breaking determinism (the I-106 / I-108 doctrine). Over-approximating trades a little speed for soundness.

**Spot.** The `QUIESCE_SHADOW_CHECK` assert firing at a specific LB (a would-be-skipped burst mutated state). Or an A/B `main.py` (skip ON vs OFF at 8 GiB) that is NOT byte-identical in `theorems.txt` / proof graphs / verifier counts.

**Fix.** Add the missing `hasWork = true` at the offending deposit door (or confirm `mailPeek` reachability). Never gate the predicate on residency; keep the SLEEP condition an over-approximation.

**Code.** [`memory.hpp::Memory::hasWork`](../GL_Quick_VS/GL_Quick/src/memory.hpp); [`mail_log.hpp::MailLog::mailPeek`](../GL_Quick_VS/GL_Quick/src/mail_log.hpp); [`prover.cpp::proveKernel`](../GL_Quick_VS/GL_Quick/src/prover.cpp) active-build + wake doors; `performElemPhase1` / `performElemPhase2` / `performElemPhase3` SLEEP detector. See also [D-194](40_decisions.md#d-194), [I-48](#i-48), [I-55](#i-55), [I-106](#i-106), [I-108](#i-108).

---

<a id="i-59"></a>
## I-59 Every runtime measurement belongs to one `performElem2` (hashburst) call — never accumulated across calls, threads, or outer iterations

**Scope.** RT measurement infrastructure (`src/infra/rt_tracker.hpp` / `src/infra/rt_tracker.cpp`) and the `RT_SCOPE_HERE` call sites inside `prover.cpp::ExpressionAnalyzer::performElem2`. The LB split removed the original `performElementaryLogicalStep` home; the per-call tracker now lives in `performElem2`, the phase-2 hashburst executor ([D-110](40_decisions.md#d-110)).

**Rule.** The `RTTracker` is a stack-local object whose lifetime equals one call of `performElem2`. It records elapsed self-time per labelled section using a `steady_clock` started in its constructor and stopped in its destructor. No state of the tracker — start time, section totals, hit counts, open-scope stack — is ever shared with a tracker from a different call, a different worker thread, or a different outer iteration of `ExpressionAnalyzer::prove`. RT records meaningfully only under `disable_lb_split` (one part per LB), so the call is the LB's whole hashburst on one thread; with splitting on, an LB's N parts each construct a tracker that would collide on the shared `.rt/<chain>.log`. The per-LB output file `.rt/<chain>.log` is rewritten from scratch on every refresh and on every later call to the same LB; previous-call contents are not merged in.

**Why.** The infrastructure exists to diagnose LBs whose single hash burst never returns. Accumulating measurements across calls would dilute the signal of *this* hang into a running average that includes earlier, possibly healthy calls. Sharing tracker state across threads would require synchronisation that defeats the purpose (a worker thread that hangs cannot be safely sampled from another thread without locks that themselves stall). Per-call scope is the design's central simplification — and the reason `RT_MAX_SECTIONS` can be a small fixed number with no global registry.

**How to spot.** Search `grep -nE 'static.*RT|thread_local.*RT|std::atomic.*RT' GL_Quick_VS/GL_Quick/src/` — every hit on a name containing `RT` other than the four `RT_*` compile-time constants in `parameters.hpp` is suspect. A tracker member that is not on the stack (a `static` inside the class, a `thread_local` cache of section totals, a global accumulator) violates this invariant.

**How to fix on violation.** Delete the cross-call state. If the offending caller wanted across-call comparison (e.g. "did this LB hang in two consecutive bursts?"), the user-facing recipe is `cp .rt/<chain>.log .rt/<chain>__burst1.log` by hand between bursts — not a refactor of the tracker.

**Code.** `src/infra/rt_tracker.hpp` / `src/infra/rt_tracker.cpp` (the tracker class and its members). `prover.cpp::ExpressionAnalyzer::performElem2` (the `RT_TRACKER_DECL(body)` at the top + the `REQGEN_*` / `FIXPOINT_LOOP` `RT_SCOPE_HERE` call sites it owns; the phase-1/phase-3 `RT_SCOPE_HERE` markers run in separate sweeps with no tracker active and do not record).

**See also.** [D-95](40_decisions.md#d-95), [D-110](40_decisions.md#d-110) (the `performElem2` home + the split-disable flag RT requires), [I-13](#i-13) (`ChunkPool` static `char[]` convention — same no-heap-no-shared-state philosophy applied to a different subsystem), the project conventions (hashburst dump untouched).

---

<a id="i-58"></a>
## I-58 `intKnownStatements` is the immortal statement record — rows never erased outside wholesale teardowns

**Scope.** Prover. `Memory::intKnownStatements` (packed `(originalId, validityId)` key → `StatementFlags`, carrying the `registered` / `known` membership bits per [I-85](#i-85)).

**Rule.** No code path on the per-step / per-burst pipeline erases rows from `intKnownStatements`. Once a write site upserts a packed `(originalId, validityId)` key, that row stays for the LB's lifetime. Legitimate row-erase sites are exactly the wholesale teardowns: `Memory::wipeSubtree` (closed-scope teardown — removes rows whose `validityId` falls inside the closed subtree), `prover.hpp::eradicateImplicationFromLB` (specific-implication teardown), and `prover.hpp::resetResentExpressionRegistries` (single resent-compound teardown, [D-106](40_decisions.md#d-106) delete → send → reabsorb) — each retires a whole item and removes every membership at once. The CE teardown `filter.cpp::releaseCEBatchMemory` additionally clears the `registered` membership (erasing registered-only rows, keeping `known` rows) — a membership reset, not a per-class drop. Per-class cleanup is *not* legitimate: `cleanUpExpressions` once erased rows during class canonicalization and desynced the Site F gate from the registration gates.

**Concretely banned erase sites:** `cleanUpExpressions` (local-encoded sweep + encoded sweep — both used to erase the packed key paired with the `intStatementLevelsMap.erase(pk)` index erase; that registry erase is removed).

**Why.** The `known` bits are the ground truth for `addExprToMemoryBlock`'s **Site F** ancestor-scan duplicate-suppression gate (`for (anc: ancestorsOf[valId])`, `known`-bit test per packed key). When mail re-delivers a statement that the LB has already processed, Site F must fire and short-circuit the entire addExpr pipeline — otherwise `addEquality` runs, hits its own `registered`-bit skip, falls through to `addStatement`, which then pushes the equality onto `newStatements`, which trips the post-loop `intStatementLevelsMap` assert at `prover.cpp::addExprToMemoryBlock` in the `for (ev: stmts) { sortedNew = added;... }` block.

The pre-fix `cleanUpExpressions` erased non-canonical class members from BOTH `intStatementLevelsMap` (correct — those entries no longer represent a live statement at this validity) AND the packed registry (wrong — that desynced the Site F gate from the registration gates, opening the re-arrival window described above). The fix: erase only from `intStatementLevelsMap`. The registry rows record "we've already added this statement, do not re-enter the kernel for it"; the runtime containers (`intEncodedStatements`, `intLocalEncoded*`, `intStatementLevelsMap`) carry the canonical-only live state.

**How to spot.** Search `grep -nE 'intKnownStatements\.erase' GL_Quick_VS/GL_Quick/src/`. Expected hits: `prover.hpp::eradicateImplicationFromLB`, `prover.hpp::resetResentExpressionRegistries`, `memory.cpp::Memory::wipeSubtree`, and the membership reset in `filter.cpp::releaseCEBatchMemory`. Any hit inside `cleanUpExpressions` (or any other per-step / per-class pass) is a violation. Symptom of violation: `addStatement post-loop intStatementLevelsMap invariant violated` assert in `addExprToMemoryBlock` during a Peano main / IncubatorPeano run with non-trivial equivalence-class formation.

**How to fix on violation.** Delete the offending erase. If the surrounding code also erases from `intStatementLevelsMap`, that erase stays — only the registry erase is the bug. Verify with the SwDD invariant's reproduction recipe (Peano main, an equality whose args belong to a class containing other non-canonical members; the assert fires within 3–5 hash bursts).

**Code.** `prover.hpp::cleanUpExpressions` (local-encoded sweep + encoded sweep, both sites preserve the `intStatementLevelsMap.erase(pk)` index erase with no registry erase). `prover.cpp::addExprToMemoryBlock` Site F (the `known`-bit reader). `prover.cpp::addEquality` (the `registered`-bit short-circuit). The four legitimate erasers above.

**See also.** [I-85](#i-85), [D-93](40_decisions.md#d-93), [I-60](#i-60), [I-27](#i-27) (Site F / Site H ancestor-scan dedupe at `addExprToMemoryBlock` entry).

---

<a id="i-60"></a>
## I-60 `addExprToMemoryBlock` is a flat container-insertion routine; no equivalence-class apply burst inside it

**Scope.** Prover. `addExprToMemoryBlock` and the helpers it delegates to (`addEquality`, `addNegatedEquality`, `addStatement`, `updateEquivalenceClasses`).

**Rule.** The addExpr chain inserts the input statement into the LB's containers (intEncodedStatements + intLocalEncoded* + intStatementLevelsMap + intKnownStatements + exprOriginMap), routes equality through `updateEquivalenceClasses` for class merging, and disintegrates compound expressions. It does **not** fire `applyEquivalenceClass` or the four admission/rejected helpers; it does **not** discharge `toBeProved`; it does **not** recursively re-enter `addExprToMemoryBlock` for OR-integration / NotOrScope parent emissions.

**Why.** Pre-refactor, the addExpr chain fired apply transitively (via `addStatement`'s in-line three-direction loops and the post-merge apply loop inside `updateEquivalenceClasses`) and the Kernel's post-`addStatement` loop ran toBeProved discharge with recursive addExpr emissions. Both produced a tangled control flow with unpredictable per-step real-time cost and recursive cross-scope emissions that fought the parent→child mail-direction contract.

**How to spot.** Search `addExprToMemoryBlock` body (prover.cpp) for `applyEquivalenceClass`, `dischargeToBeProved`, or any `addExprToMemoryBlock` recursive call — all should be absent. The equality-class application happens exclusively from `applyEquiClasses`; toBeProved discharge happens exclusively from `dischargeToBeProved`; parent-scope emissions enqueue into `sameIterationInternalMail`. (The contradiction-detection branch does still call `updateGlobalDirect` to broadcast a proved contradiction theorem — that is the cross-LB broadcast concern, separate from the apply burst, and is not forbidden by this invariant.)

**How to fix on violation.** Move the misplaced call into the correct per-step site (`applyEquiClasses` for apply / `dischargeToBeProved` for discharge / `fillMailOut` for mail / `sameIterationInternalMail` enqueue for parent-scope emission). Do **not** re-add the call inside addExpr.

**See also.** [D-96](40_decisions.md#d-96), [I-61](#i-61).

---

<a id="i-61"></a>
## I-61 Per-step delta containers cleared once per LB in the phase-2 finalize — after every part's request generation, before the record merge

**Scope.** Prover. `performElemPhase2` (the per-LB phase-2 finalize; D-126).

**Rule.** Once per LB per burst, inside `performElemPhase2`'s entrance `isActive` block, the LB clears `intLocalEncodedStatementsDelta` and resets `localHashMemoryDelta` (`resetToFresh`) — after EVERY split part's request generation has read them (a later part's batches 2 and 5 still consume the deltas, which is why the clear must NOT sit inside `performElem2`; D-126) and before `applyFiringRecords` merges the burst's captured records (the staging vectors `admissionKeysAlgebra` / `deferredIntegrationPreps` clear in the same block, so the merge repopulates them fresh). Single site — no `contradictionIndex == -1` gate. The generated requests hold arena-allocated `IntEncodedExpr` COPIES, not pointers into the live delta vectors, so the clear does not invalidate in-flight request data. The per-step delta-class tracker `changedClassesThisStep` is **not** cleared here — that tracker is managed by `standardProcessing` (cleared at the end of each call) and the start-of-burst clear at the top of `performElemPhase1`.

**Why.** Under the channel-split architecture (`standardProcessing` two-call shape with `sameIterationInternalMail` / `nextIterationInternalMail` split), the **pre-burst** `standardProcessing` call populates `intLocalEncodedStatementsDelta` with `mailIn`-derived and `nextIter`-derived statements. Request generation reads the delta into `reqBuf`. After request gen completes, the delta has served its purpose for hashburst — the **post-burst** `standardProcessing` call wants to populate a clean delta with hashburst-derived (sameIter-channel) statements only. Single-site pre-fixpoint clear achieves that without leaving stale entries.

The `contradictionIndex == -1` gate (predating visible git history, preserved as "matching the pre-relocate gate" by every prior refactor) used to protect delta entries needed by post-burst `dischargeToBeProved` when contradiction had fired but `isActive` stayed true. Under the new architecture, `dischargeToBeProved` already ran inside the pre-burst `standardProcessing` call before this clear point, so the contradicted-and-proved emission path has already done its work. The gate becomes obsolete.

**How to spot.** Search `performElemPhase2` for `intLocalEncodedStatementsDelta.clear` — it appears exactly once, inside the entrance `isActive` block, before the `applyFiringRecords` merge. A per-executor clear inside `performElem2` (wipes the input a later split part still reads), a post-merge clear, or any `contradictionIndex` gate around the clear is a regression.

**How to fix on violation.** Consolidate to the single finalize site. Drop any `contradictionIndex` gate.

**See also.** [D-96](40_decisions.md#d-96), [D-90](40_decisions.md#d-90), [I-60](#i-60), [I-62](#i-62).

---

<a id="i-62"></a>
## I-62 `Memory` has two distinct internal-mail channels with disjoint lifecycles

**Scope.** Prover + memory layout. `Memory::sameIterationInternalMail` + `Memory::nextIterationInternalMail`.

**Rule.** Every `Memory` has two `Mail` instances for the per-LB internal-mail path:

- **`sameIterationInternalMail`** is the **same-step hashburst-feed** channel. Drained by the **post-burst** `standardProcessing` call. Writers:
 - `memory.cpp::checkLocalEncodedMemoryStatic` — hashburst rule-firing output.
 - `dischargeToBeProved` — parent-scope emissions, routed via the `internalMailOut` parameter when `standardProcessing` is called in the **pre-burst** phase.
 - **Equi-class machinery revival sites** (carried over unchanged from the pre-split architecture):
 - `prover.hpp::emitIntegrationRevivalToInternalMailIn` (called from `revisitRejected2` and `revisitRejectedIntegration2`).
 - `prover.hpp::applyEquivalenceClassToRejectedMapIntegration` (revival deposit when an admission template gets rewritten under a class).
 - `prover.hpp::applyEquivalenceClassToRejectedMap` (algebra-side revival counterpart).
 - `prover.hpp::sanitizeHashMemory` (end-of-burst revival when a rule gets eradicated and its head needs re-emission).
 - `prover.hpp::ordisMerge` (or-disintegration revival when a re-evaluated witness gets re-emitted).

 Lifecycle: entries from revival sites that fire during the **post-burst** `applyEquiClasses` survive into the next step's **post-burst** absorb (1-cycle delay, matching pre-split behavior). Whether to phase-route these revival sites — emit to `nextIter` when running post-burst so the next step's pre-burst drains them — is an open architectural question; current behavior preserves the pre-split semantics.

- **`nextIterationInternalMail`** is the **cross-iteration** channel. Written by:
 - `updateGlobalDirect` siblings (×3, cross-LB deposits targeting the recipient block's own `nextIterationInternalMail`).
 - `dischargeToBeProved` — parent-scope emissions, routed via the `internalMailOut` parameter when `standardProcessing` is called in the **post-burst** phase.

 Drained at the start of the **next** step's **pre-burst** `standardProcessing` call, alongside `mailIn`.

The two channels MUST NOT mix for the **routed** writers (hashburst and discharge). Hashburst output never goes to `nextIter`. `dischargeToBeProved` routes via its `internalMailOut` parameter — pre-burst → sameIter (zero-delay same-step processing), post-burst → nextIter (one-step cross-iter deferral). The equi-class revival writers listed above are **not** phase-routed; they always emit to `sameIter` and carry pre-split timing semantics.

**Why.** Pre-split, `Memory::internalMailIn` was a single channel that carried both lifecycles (hashburst output AND cross-iter deferral), distinguished only by **when** the write happened (during hashburst loop = same-iter, after the post-burst clear = next-iter). Subtle clear-ordering bugs ensued — moving the clear by one position lost either same-iter or next-iter traffic. The two-channel split makes the lifecycle explicit at the type level: same-iter / next-iter is no longer a timing distinction but a container distinction.

**How to spot.** `grep -rn "sameIterationInternalMail\.statements\.insert\|nextIterationInternalMail\.statements\.insert" GL_Quick_VS/GL_Quick/src/` should return exactly the writers listed in the **Rule** section above (six `sameIter` sites including the five equi-class revival sites; three `nextIter` sites — the `updateGlobalDirect` ×3 cross-LB deposits. `dischargeToBeProved` routes through its `internalMailOut` parameter and so is not a direct insert; the `addExprToMemoryBlock` vacuous-truth path now deposits via `addStatement`, not via either internal-mail channel). Any **additional** site is a violation. Any swap of channel (e.g. hashburst writing to `nextIter`) is a violation.

**How to fix on violation.** New writers join the appropriate channel by lifecycle: hashburst-output-style writes ⇒ `sameIter`; cross-LB or always-cross-iter deferral ⇒ `nextIter`. For `dischargeToBeProved`, route via the `internalMailOut` parameter and let the `standardProcessing` caller pick the channel by phase. For revival sites, current default is `sameIter`; phase-routing them is a future architectural decision (Rule 8).

**See also.** [D-96](40_decisions.md#d-96), [D-90](40_decisions.md#d-90), [I-21](#i-21), [I-61](#i-61).

---

<a id="i-63"></a>
## I-63 `standardProcessing` is the sole driver of mail absorb + apply + discharge + fillMailOut per call phase

**Scope.** Prover. The elementary-step phases (`performElemPhase1` pre-burst, `performElemPhase3` post-burst).

**Rule.** Every burst cycle makes exactly **two** `standardProcessing` calls — the pre-burst call in `performElemPhase1`, the post-burst call in `performElemPhase3`:

1. **Pre-burst:** `standardProcessing(body, &body.mailIn, body.nextIterationInternalMail, body.sameIterationInternalMail, coreId)`. Drains `mailIn` (status=3) + `nextIter` (status=1) into `intLocalEncodedStatementsDelta`, clears both source mails, applies equivalence classes, discharges `toBeProved` (parent-scope emissions land in `sameIter`), populates `mailOut`, clears `changedClassesThisStep`.

2. **Post-burst:** `standardProcessing(body, nullptr, body.sameIterationInternalMail, body.nextIterationInternalMail, coreId)`. Drains `sameIter` (status=1) into `intLocalEncodedStatementsDelta`, clears `sameIter`, applies equivalence classes, discharges `toBeProved` (parent-scope emissions land in `nextIter`), populates `mailOut`, clears `changedClassesThisStep`.

No other site in the prover drains mail, runs `applyEquiClasses`, calls `dischargeToBeProved`, or invokes `fillMailOut`. The four pipeline stages are members of the same per-call recipe.

**Why.** Pre-refactor, the absorb pipeline was scattered: pre-hashburst block drained `mailIn` only (no apply/discharge/fillMailOut), post-hashburst block did the full pipeline. The asymmetric two-step processing was hard to reason about and introduced a 1-step delay for any mail-derived emission that needed to fire in the same step. The consolidated two-call shape — same recipe twice with different I/O channel selection — eliminates the asymmetry and lets the caller route `internalMailOut` by phase (same-iter for pre-burst, next-iter for post-burst), achieving zero-delay processing for pre-burst discharge emissions.

**How to spot.** Search `performElemPhase1` / `performElemPhase3` for direct calls to `applyEquiClasses`, `dischargeToBeProved`, `fillMailOut`, or any inline `addExprToMemoryBlock` mail-drain loop. Expected: zero such direct sites — all four pipeline stages run only inside `standardProcessing`'s body. Any inline copy is a regression.

**How to fix on violation.** Replace any inline absorb/apply/discharge/fillMailOut block with a `standardProcessing` call carrying the appropriate four-mail-argument shape.

**See also.** [D-96](40_decisions.md#d-96), [D-90](40_decisions.md#d-90), [I-60](#i-60), [I-62](#i-62), [I-64](#i-64).

---

<a id="i-64"></a>
## I-64 `fillMailOut` is the primary writer to `mailOut.statements` and `mailOut.exprOriginMap`; the commit barrier is the sole reader (pull model)

**Scope.** Prover. `mailOut.statements` and `mailOut.exprOriginMap` fields on `Memory::mailOut`.

**Rule.** `fillMailOut` is the sole writer to `memoryBlock.mailOut.statements`. `fillMailOut` is the sole writer to `memoryBlock.mailOut.exprOriginMap` **with one documented exception**: the multiplied-implication "multiplied from" history row inside `memory.cpp::addToHashMemory` is restored as a direct `addOrigin(mb.mailOut.exprOriginMap,...)` because the multiplied implication installs as a *rule* in `overallHashMemory` and never reaches `intLocalEncodedStatementsDelta`, so the delta-driven `fillMailOut` path cannot propagate the row. Every other writer to either field has been deleted:

- `mailOut.statements`: pre-refactor scattered writers (`addStatement`, `applyEquivalenceClass`, `addEquality` original + mirror, `addNegatedEquality` original + mirror, anchor handling, the hashburst-direct deposit at `checkLocalEncodedMemoryStatic`) were deleted in commit 11 of the original refactor and commit 7 of the follow-up consolidation.
- `mailOut.exprOriginMap`: pre-refactor scattered writers (`addEquality` / `addNegatedEquality` original + mirror history rows, `applyEquivalenceClass` per-class commit, `disintegrateExpr2` expansion / disintegration / OR-disintegration origin rows, `addExprToMemoryBlock`'s inlined-kernel implication-expansion / premise / goal / assumption / expansion / instruction rows, the vacuous-truth row, the recursion-pre-emission row on `tempMb`, the `prehandleAnchor` axed-anchor history row) were deleted in commit 10 of the follow-up consolidation. The `addToHashMemory` multiplied-implication "multiplied from" origin in `memory.cpp` was deleted in commit 10 and then restored as the documented exception in commit 12, because the multiplied implication never reaches `intLocalEncodedStatementsDelta` (see exception subsection below).

`fillMailOut` iterates `intLocalEncodedStatementsDelta`, gates on `validityName == "main" && allowedForMail`, copies levels from `intStatementLevelsMap` (packed probe via the delta row's own ids) into `mailOut.statements`, and copies origin lines from `body.exprOriginMap` into `mailOut.exprOriginMap`. The LB's containers (`intLocalEncodedStatementsDelta` + `intStatementLevelsMap` + `body.exprOriginMap`) carry every piece of information needed for the outbound mail.

**Pull-model update (2026-06-24, [D-137](40_decisions.md#d-137)).** `mailOut` is now a per-cycle staging buffer, not a routed outbox. `fillMailOut` remains the writer of the per-step delta, but for the **root LB only** there are additional sanctioned writers: the post-join `updateGlobalDirect`/`updateGlobal` `"theorem"`-origin sends and the D-76 compact-implication drain, plus load-time `broadcastTheorems`, each via `mergeBatchInto(localMail, this->body.mailOut)` (single-threaded; staging for the next commit). The **sole reader/drain** is the commit barrier at `proveKernel`'s post-join seam (`MailLog::commit`), NOT `sendMail` (deleted). The single-writer *spirit* holds — every writer is single-threaded and disciplined — but the literal "fillMailOut is the only writer" claim is superseded by these root-only staging merges. The "eighteen hits" grep below counts `mailOut.statements.insert` / `addOrigin(...mailOut.exprOriginMap...)` only; the `mergeBatchInto` root-staging writers are a separate, sanctioned set.

**Why.** Single-writer policy. Scattered writes meant ~17 sites had to agree on the gate (main + allowedForMail), the levels source, and the origin source. Divergence between sites produced subtle bugs where children received slightly different mail content than the parent's delta would have predicted. One writer reading off one delta makes the contract trivial to inspect and trivial to verify. Per the user's 2026-05-24 direction: "FillMaiOut must be the only place where mailOut is written. LB containers have all the necessary information."

**Structural corollary: LBs without children make `mailOut` writes useless.** An LB's `mailOut` is drained by the commit barrier into the LB's own `MailLog` log, which its descendants pull. An LB with no children has no one pulling its log — every `mailOut` write on such an LB is vestigial (the commit still runs, but no recipient reads the batch). Recursion / induction LBs are the canonical no-children case: the three sub-LBs of an induction triad (recursion-premise, zero-case, successor-case) exist to discharge the induction goal up to the parent via `dischargeToBeProved`'s nextIter route, not to seed downstream LBs. Writes to `mailOut.exprOriginMap` from inside the vacuous-truth path (gated on `isPartOfRecursion`) and the recursion pre-emission on `tempMb` were deleted in commit 10 and intentionally NOT restored, because the LB they fire on has no recipients in the first place. This is the principle behind why the multiplied-implication exception is genuinely the only one: every other deleted candidate either reaches the delta (fillMailOut covers) or fires on a no-children LB (mailOut is moot).

**How to spot.** `grep -rn "mailOut\.statements\.insert\|addOrigin([^)]*mailOut\.exprOriginMap" GL_Quick_VS/GL_Quick/src/` against `body.mailOut` writes (excluding the function-scope local `Mail mailOut` in `prover.cpp::updateGlobalDirect` and the sibling broadcast functions — see the "Verification note" subsection) should return exactly **eighteen** hits:

- **Two** inside `fillMailOut` — one for `mailOut.statements.insert`, one for the `addOrigin(mailOut.exprOriginMap,...)` history copy. The canonical sole-writer.
- **One** inside `memory.cpp::addToHashMemory`'s multiplied-implication `for (std::size_t c = 0; c < copies.size; ++c)` loop (Documented exception 1).
- **Six** inside `prover.cpp::addTheoremToMemory`'s three LB-creation sites (chain-walk, reformulated-contradiction, standard-contradiction — each contributes one `mailOut.statements.insert` + one `addOrigin(mailOut.exprOriginMap,...)`; Documented exception 2).
- **Two** inside `prover.cpp::disintegrateExprCore2` — paired `addOrigin(memoryBlock.mailOut.exprOriginMap,...)` writes at the expansion-origin and disintegration-origin sites (Documented exception 3).
- **Seven** inside `prover.hpp::prepareIntegrationCore2` — paired `addOrigin(mb.mailOut.exprOriginMap,...)` writes at all seven local `mb.exprOriginMap` write sites (Case A implication-expansion + premise-element; Case OR or-branch-goal + or-branch-assumption; Case B expansion-integration + iiv + iivHash). Documented exception 4 — restored 2026-05-25 to fix the `buildStack` crash on Peano main where the pi-bound integration-instruction (`(>[pi_lev_0_1](in[pi_lev_0_1,u_1])(>[](in2[pi_lev_0_1,u_6,u_3])(existence2[u_1,u_6,u_3])))`) was cited as an origin dep at a descendant LB but had no key entry in the descendant's `exprOriginMap`.

Any additional hit is a regression.

**How to fix on violation.** Route any new mailOut-bound write through `fillMailOut` by ensuring the source statement reaches `intLocalEncodedStatementsDelta` via `addExprToMemoryBlock` and that the corresponding origin reaches `body.exprOriginMap` via the local `addOrigin` call adjacent to the producer's logic. If the new write covers an origin whose subject is a *rule* (installs into `overallHashMemory` and never enters `intEncodedStatements`), document the exception in the **Documented exceptions** subsection below; do not add the carve-out silently. If the gate needs to differ from `main + allowedForMail`, that is an architectural decision (Rule 8) requiring user approval, not a per-site fork.

**Documented exceptions.**

- **`memory.cpp::addToHashMemory` multiplied-implication "multiplied from" origin** (restored 2026-05-24, commit 12 of the follow-up consolidation, per user direction "this mailOut history entry must be restored"). The multiplied implication is a new rule produced by `multiplyImplication`, installed into `overallHashMemory`. It does **not** land in `intEncodedStatements` / `intLocalEncodedStatementsDelta`, so `fillMailOut`'s delta-driven copy cannot propagate the row. The direct `addOrigin(mb.mailOut.exprOriginMap, copyEv, mulOrigin,...)` call adjacent to the parallel `mb.exprOriginMap` write preserves the "multiplied from" chain for receivers' visualizer / verifier walks.

- **LB-creation paired mailOut write at the three theorem-load `addExprToMemoryBlock` sites in `prover.cpp::addTheoremToMemory`** (chain-walk LB creation, reformulated-contradiction LB creation, standard-contradiction LB creation — added 2026-05-25 per user direction "when a LB for an expression is created code can explicitly put the expression and its hist line to the mailOut"). At each site, immediately after the `addExprToMemoryBlock(element, *child,..., task formulation,...)` call, the code also writes `child->mailOut.statements.insert(...)` and `addOrigin(child->mailOut.exprOriginMap, ev, origin,...)` for the LB's own task-formulation entry. **Why the exception is needed.** LB creation runs at theorem load — *before* any prover step. `fillMailOut` runs only inside step 1's pre-burst `standardProcessing`. `buildGrid`'s `smashMail(boxes)` at the close of theorem load dispatches every LB's current `mailOut` to descendants' `mailIn`; without the LB-creation paired write, every LB's `mailOut` is empty at that point and the dispatch is a no-op. The descendant's step 1 then starts with empty `mailIn`, its pre-burst `absorb` finds nothing to roll into `exprOriginMap`, and the transitive walk in `fillMailOut` asserts on a dep whose history the parent had locally but never shipped (because the parent's first `sendMail` hasn't run yet — parallel cores race). With the paired write, the initial dispatch carries each LB's task-formulation origin to every descendant before any step begins. Symptom that surfaced this exception: `__contradiction__(=[10,11])` LB's step 1 `fillMailOut` asserted on the original `(AnchorIncubator[1,2,3,4,5,6,7,8,9,10,11,12,13,14])` task-formulation dep — the parent had it locally + in `mailOut.exprOriginMap` at the moment of assert, but the child had not absorbed because parent's `sendMail` had not run. See  and for the trap output.

- **`disintegrateExprCore2` paired mailOut writes at the expansion-origin and disintegration-origin sites in `prover.cpp::disintegrateExprCore2`** (added 2026-05-25 with the "runs through 5 iterations" milestone, mirroring ref's pre-refactor paired writes). At each site the code writes `addOrigin(memoryBlock.exprOriginMap,...)` for the local entry and a paired `addOrigin(memoryBlock.mailOut.exprOriginMap,...)` for cross-LB shipment. **Why the exception is needed.** The expansion-origin row (`expansion of compact compound`) and the disintegration-origin row (`disintegration of conjunction`) are *documentation-only*: their KEYs are not added to `intEncodedStatements` / `intLocalEncodedStatementsDelta` (they describe the existence of an expansion tree, not a new statement to fire on). `fillMailOut`'s delta-driven copy therefore never sees them. Without the paired write, descendant LBs that import the compact compound's history walk the disintegration dep and find no expansion-origin entry, asserting in `buildStack` at chapter export.

- **`prepareIntegrationCore2` paired mailOut writes at all seven local `mb.exprOriginMap` addOrigin sites in `prover.hpp::prepareIntegrationCore2`** (restored 2026-05-25, this branch — matches ref's pre-refactor structure). The seven sites are:
 - Case A (implication, `le.category == "implication"`): the `expansion for integration` row for the expanded-implication signature, and the `premise element` row for every renamed-chain premise.
 - Case OR (`le.category == "or"`): the `expansion for integration` row for the per-branch sub-implication goal, and the `or branch assumption` row for every other-branch negated disjunct.
 - Case B (existence / and, `le.category == "existence" || "and"`): the `expansion for integration` row for the expanded-signature goal, the `reformulation for integration {and / >[bound] / >[]}` row for the integration-instruction history form (`iiv`), and the same row for the u_-preserved hash form (`iivHash`).

 **Why the exception is needed.** All seven KEYs are *documentation-only*: `expandedSignature + "_integration_goal"`, `expandedImplication + "_integration_goal"`, `subImpl + "_integration_goal"`, the per-element renamed chain entries, the negated-disjunct assumptions, and the integration-instruction forms (`iiv`, `iivHash`). None of these enter `intLocalEncodedStatementsDelta` — they are either suffixed marker strings, hashmemory rule strings, or scope-bound assumption rows. `fillMailOut`'s delta-driven copy never sees them. **Symptom that surfaced this exception:** the pi-bound integration-instruction `(>[pi_lev_0_1](in[pi_lev_0_1,u_1])(>[](in2[pi_lev_0_1,u_6,u_3])(existence2[u_1,u_6,u_3])))` was cited as origin dep `dep[0]` in `(existence2[1,6,3]) <- implication | <iiv> | (in2[2,6,3]) | (in[2,1])` rows written at descendant LBs (by hashburst rule-firing in `memory.cpp::checkLocalEncodedMemoryStatic` via `lmv.originalImplication`) but the producer LB's local `addOrigin(mb.exprOriginMap, iiv, originInstruction,...)` was the only mailOut-eligible write, and without the paired mailOut shipment the descendant LB's `exprOriginMap` never received the iiv key. `buildStack` at chapter export walked the dep and asserted "no origin found". Restoring all seven paired writes brings the refactor in line with ref. See [D-91](40_decisions.md#d-91).

All other cross-iter rule deposits that need cross-LB history routing should also use this pattern and add a bullet here.

**Verification note** (covered by the LB-container invariant or by the no-children corollary, retained for clarity):

- `prehandleAnchor` (`prover.cpp::prehandleAnchor`) pushes the axed-anchor expression into `mb->intLocalEncodedStatementsDelta` (plus `intEncodedStatements`, `intStatementLevelsMap`, etc.) on every non-recursion LB where it fires; the function early-returns on `isPartOfRecursion` so it never touches recursion LBs. The accompanying `addOrigin(mb->exprOriginMap, encVal,...)` makes the origin available to `fillMailOut` via the delta-driven lookup. Descendants inherit the origin through the normal mailOut → mailIn → body.exprOriginMap merge path on the LB's first elementary step.
- **Vacuous-truth path inside `addExprToMemoryBlock`** (deleted in commit 10, NOT restored). The branch is gated on `isPartOfRecursion`, so it fires only on recursion / induction LBs — which have no children. The deleted `addOrigin(memoryBlock.mailOut.exprOriginMap, exprVal, origin,...)` would have shipped `exprVal`'s history to descendants, but recursion LBs have no descendants to ship to (their role is to discharge the induction goal up to the parent via `dischargeToBeProved`'s nextIter route). The body.exprOriginMap entry at line 4156 is preserved for the in-LB visualizer walk; that is sufficient.
- **Recursion pre-emission on `tempMb` inside `addTheoremToMemory`** (deleted in commit 10, NOT restored). Same reason as vacuous truth: `tempMb` is an induction-LB-in-construction with no children to receive its mailOut. The `tempMb->exprOriginMap` entry on the parallel line is preserved.
- `updateGlobalDirect` / `updateGlobal` build a **local** `Mail mailOut` of `"theorem"`-origin lines; under the pull model these are merged into the **root's** `body.mailOut` via `mergeBatchInto` (post-join, single-threaded) and ride the next commit barrier. These root-only staging writers are noted in the Pull-model update above; `sendMail` is deleted.
- (The `addToHashMemory` multiplied-implication "multiplied from" caveat that previously lived here has been promoted to the **Documented exceptions** subsection above and restored in code at commit 12.)

**See also.** [D-96](40_decisions.md#d-96), [D-90](40_decisions.md#d-90), [I-26](#i-26), [I-44](#i-44), [I-63](#i-63), [I-56](#i-56).

---

<a id="i-56"></a>
## I-56 Hashburst mail-deposit honors ref's Site F dedup before writing to `sameIterationInternalMail`

**Scope.** Prover. `memory.cpp::checkLocalEncodedMemoryStatic` — the rule-firing site inside the hashburst fixpoint loop that emits derived statements + history lines to `sameIterationInternalMail`.

**Rule.** Before depositing a derived statement to `sameIterationInternalMail`, run an ancestor-scope scan against `intKnownStatements` mirroring ref's `addExprToMemoryBlock` Site F early-return at `prover.cpp::4141-4145`:

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
    /* existing mail.statements.insert + addOrigin block */
}
```

Skip both `sameIterationInternalMail.statements.insert(...)` AND `addOrigin(sameIterationInternalMail.exprOriginMap,...)` when the head is already known at any ancestor scope. Compressor mode keeps multiples — same `!compressor_mode` carve-out ref uses.

**Why.** Pre-commit-17, ref's hashburst rule-firing path routed through `addExprToMemoryBlock` which hits Site F at function entry and `return`s immediately when the head is already in `intKnownStatements` at any ancestor scope. That early-return suppresses BOTH the statement insert AND any new origin write. So the same head being re-derived by N different (rule, premise) combinations within one burst records exactly ONE origin in ref.

Commit 17 of the follow-up consolidation rewired the hashburst output to deposit to `sameIterationInternalMail` instead of calling `addExprToMemoryBlock` — bypassing Site F. `addOrigin(mail.exprOriginMap, …)` dedupes only on (key, ENTIRE-origin-tuple), so N distinct (rule, premise) firings for the same head all land as N distinct origin lines.

Symptom (`__contradiction__(in2[10,10,3])`, 2026-05-25, both branches' hashburst dumps compared at the same LB):

- Equi `(=[9,11])` at burst 3 EXIT: **10 origins** (1 equality2 + 8 distinct implication-rule-variant + 1 symmetry).
- Ref `(=[9,11])` at burst 3 EXIT: **2 origins** (1 equality2 + 1 implication — the 8 extra equi origins are alpha-variant re-firings ref's Site F suppressed).
- Equi `(in[2,1])` at burst 5 EXIT: **30 origins** (cap hit — 1 disintegration + 29 implication re-firings).
- Ref `(in[2,1])` at burst 5 EARLY-EXIT: **1 origin** (just the disintegration).

`buildStack` at chapter export then has 30^depth × 21^depth candidates per chain → exponential exploration → 5M+ buildStack calls, single-thread CPU-bound, hang. Adding the Site F mirror gate at the mail-deposit site brings equi's per-key origin counts back in line with ref, chapter export completes cleanly, and verifier passes airtight.

**Earlier attempt that did not work.** A recursive transitive walk in `fillMailOut` (`shipExprHistoryTransitively`) was implemented first: ship documentation-only deps' history transitively along with each delta-entry's origin. It addressed shipping documentation-only entries like expansion conjunctions, but did not address the more fundamental per-key origin bloat from re-firing on already-known heads. Chapter export still hung. The walk was retired; the lesson is that origin bloat at the mail-deposit site needs a head-dedup, not a deeper history walk.

**How to spot.** Hashburst-trace diff equi-vs-ref at any contradiction LB: same `intEncodedStatements` count, same `overallHashMemory.originals` rule registry size (rules identical), same atomic in2/in3 facts present — but per-key origin counts diverge starting around burst 3 (equi top expression at 10+, ref at 2-3). `awk` top-N over the `exprOriginMap` section of any burst makes the divergence visible immediately.

**How to fix on violation.** Confirm the Site F mirror gate above wraps the `sameIterationInternalMail.statements.insert(...)` + `addOrigin(sameIterationInternalMail.exprOriginMap,...)` block. Without it, the post-commit-17 mail-deposit path silently records duplicate-head origins per (rule, premise) combination and `buildStack`'s candidate exploration explodes at chapter export.

**Companion fixes landing in the same commit (2026-05-25 milestone — runs through 5 iterations of IncubatorPeano cleanly).**

1. `trackExpansionHistory` paired `mailOut.exprOriginMap` writes restored at the expansion-origin and disintegration-origin sites in `prover.cpp::disintegrateExprCore2`, matching ref's lines 6479 + 6492.
2. LB-creation paired `mailOut.statements.insert` + `addOrigin(mailOut.exprOriginMap, …)` writes added at the three `addExprToMemoryBlock(... task formulation...)` sites in `prover.cpp::addTheoremToMemory` (chain-walk, reformulated-contradiction, standard-contradiction).
3. `buildGrid` **commits** each LB's startup `mailOut` into its `MailLog` log (the LB-creation paired writes), so descendants pull them at their step-1 phase-1. (Pre-pull-model this was a `sendMail`-into-boxes + `smashMail` dispatch.)

**See also.** [D-92](40_decisions.md#d-92), [I-64](#i-64), [I-44](#i-44), [I-19](#i-19).

---

<a id="i-65"></a>
## I-65 `applyEquiClasses` covers both (delta-class × all intEncodedStatements) AND (non-delta-class × new intEncodedStatements), with equality / negated-equality statements filtered out

**Scope.** Prover. `prover.hpp::applyEquiClasses`.

**Rule.** Inside the outer `while (intEncodedStatements.size grows)` fixpoint, two passes run in sequence on each iteration:

- **Pass 1** — for each class in `changedClassesThisStep`, run the 4 admission/rejected helpers (`applyEquivalenceClassToRejectedMapIntegration`, `applyEquivalenceClassToAdmissionMapIntegration`, `applyEquivalenceClassToAdmissionMap`, `applyEquivalenceClassToRejectedMap`) and apply the class via `applyEquivalenceClass` to `intEncodedStatements[startIdx[k].. end)` (with `startIdx[k] = 0` on entry → covers ALL existing statements; advanced to current size after the pass).

- **Pass 2** — for each class in `equivalenceClassesMap` that is NOT also in `changedClassesThisStep` (Pass 1 already handled the delta classes), run the same 4 admission/rejected helpers and apply the class to `intEncodedStatements[eqClassSttmntIndexMapMap[v][cls.variables].. end)`. After the inner loop, advance `eqClassSttmntIndexMapMap[v][cls.variables]` to current `intEncodedStatements.size`.

Both passes use the same equality / negated-equality filter inside the inner loop: `if (isEquality(stmt.original) || isNegatedEquality(stmt.original)) continue;`. The validity-key list and per-validity class count are snapshotted at the top of each Pass 2 outer-iteration, and class copies (not references) are used inside the inner loop, to defend against mid-iteration mutation of `equivalenceClassesMap[validity]` by rewrite-induced equality formation.

**Why.** Pre-fix, only Pass 1 existed. A class registered at LB creation time (e.g., `__contradiction__(=[10,11])` seeded with `(=[10,11])` task formulation → `addStatement` → `updateEquivalenceClasses` populates `changedClassesThisStep`) is wiped from the delta tracker at the start of the child LB's first burst (the `changedClassesThisStep` clear at the top of `performElemPhase1`; see [I-61](#i-61) for the delta-container clear in the phase-2 finalize). The class persists in `equivalenceClassesMap` but is invisible to the delta-only Pass 1. Statements that arrive later in the LB's life (e.g., `(in2[9,10,3])` lands at `intEncodedStatements[44]` at burst 2 via the AnchorIncubator chain disintegration mail) never get back-rewritten by the class — zero `equality1` rewrites fire across the entire LB's 5 bursts; the contradiction chain dies; the 33 `!(=[X,Y])` family inequalities (and ~26 downstream `!(in3[...])` variants) never prove.

The equality / negated-equality filter exists because those two shapes are covered by their own dedicated paths: positive `(=[a,b])` → `updateEquivalenceClasses` (class merge); negated `!(=[a,b])` → `applyEquivalenceClassToNegatedEquality` (called inline from `addStatement` on incoming arrivals). `applyEquivalenceClass` is for everything else (`in`, `in2`, `in3`, `fXY`, `fXYZ`, operator expressions,...). Pre-reshuffle ref's `applyEquivalenceClass` had only the `isEquality` early return (skipping positive equalities) — negated equalities went through both `applyEquivalenceClass` (via the in-line burst in `addStatement`) AND `applyEquivalenceClassToNegatedEquality`. The double processing is treated as a ref-side bug; the equi contract is tightened to single-path coverage for each shape.

**How to spot.** Hashburst-trace any contradiction LB seeded with `(=[a,b])` (e.g., the `__contradiction__(=[a,b])` family on IncubatorPeano). Pre-fix symptoms: `equivalenceClassesMap` contains the class `{a, b}` from ENTRY #1 onward (the persistent class is correctly created at LB creation), but `<- equality1` count is 0 across all bursts in the entire LB, and the canonical substitution targets (`(in2[X,b,3])` from `(in2[X,a,3])`) are absent from both `intEncodedStatements` and `exprOriginMap`. `.scripts/compare_chapter_vs_originmap.py` flags such cases as the canonical "MISSING + ALL_DEPS_AVAIL" pattern — both deps (the source statement and the equality) are AVAILABLE, the head is NOT.

**How to fix on violation.** Restore both Pass 1 and Pass 2 inside the outer `while (size grows)` fixpoint; keep the equality / negated-equality filter in both passes; do NOT re-introduce inline class application inside `addStatement` (that goes against [I-60](#i-60)); do NOT remove the `changedClassesThisStep.clear` at the top of `performElemPhase1` (it correctly isolates delta state per step — Pass 2 reads the persistent store directly via `equivalenceClassesMap`).

**See also.** [D-97](40_decisions.md#d-97), [I-60](#i-60), [I-61](#i-61), [D-96](40_decisions.md#d-96).

---

<a id="i-66"></a>
## I-66 — the phase-2 hashburst never deactivates or discharges an LB; the `isActive` flip + all discharge bookkeeping happen only in phase 3's post-burst `standardProcessing`

**Scope.** Prover. The phase-2 executor path (`performElem2`, `checkLocalEncodedMemoryStatic`, `applyFiringRecords`) versus the phase-3 post-burst path (`standardProcessing` → `dischargeContradiction`, `dischargeToBeProved`, and the `addExprToMemoryBlock` mail absorb).

**Rule.** The phase-2 hashburst is read-only on the shared LB ([D-116](40_decisions.md#d-116)): it captures firings as `FiringRecord`s and MUST NOT flip `memoryBlock.isActive`, broadcast (`updateGlobalDirect` / `updateGlobalTuples` / `updateGlobalDirectTuples`), erase `toBeProved`, push `inductionMemoryBlocks`, mark `contradictionTable`, write `exprOriginMap` / `mailOut.exprOriginMap`, or clear `contradictionTheoremId`. Every deactivation and discharge — induction-proof close, incubator / CE / vacuous-truth contradiction, OR-integration — happens only in phase 3, when `standardProcessing` runs `dischargeContradiction` then `dischargeToBeProved` and absorbs `sameIterationInternalMail` through `addExprToMemoryBlock`. This is what lets N executors share one LB on parallel threads without racing on `isActive`. The phase-2 streaming **early-exit** ([D-121](40_decisions.md#d-121)) does NOT weaken this: when `burstDeactivates` detects the LB is doomed it sets an **external** per-LB `std::atomic<bool>` stop flag (owned by `proveKernel`, shared by the LB's parts), never a field on the LB, so the burst still writes nothing on the LB itself; the actual deactivation/discharge still happens only in phase 3. The early-exit is honored only when the LB runs unsplit (`g_splitCount == 1`); under split the `stop` is ignored and every part runs to completion, because a sibling bailing on another part's `stop` is a determinism-breaking race ([I-76](#i-76)).

**History.** A mid-burst RT predicate `deactivationCheck` once detected closes during the burst, recording into a per-executor `sawDeactivation` early-stop signal without mutating state. That signal went unused in the flat-executor phase 2, and the predicate was removed entirely ([D-122](40_decisions.md#d-122)) — contradiction detection + reactions now live solely in `dischargeContradiction`. The earlier `while(changed)` / EARLY-EXIT-`break` framing is also obsolete: the loop is single-pass since [D-104](40_decisions.md#d-104) and the EARLY-EXIT dump trap was removed once deactivation deferral made it dead. The early-exit was later **reintroduced with a real consumer** ([D-121](40_decisions.md#d-121)): `burstDeactivates` (mirroring the phase-3 discharge detection on each fired head — contradiction, vacuous-truth contradiction, toBeProved reached, and, added by the CE-filter pool [D-118](40_decisions.md#d-118), a CE-filter refutation when the head's negation is a loaded fact) runs read-only on each fired head and, on a hit, sets the external stop flag (honored only at `g_splitCount == 1`; under split the early-exit is disabled — a sibling bail is a determinism race, [I-76](#i-76)). Unlike the old `deactivationCheck` the signal is now consumed (it stops request generation), and unlike it nothing is written on the LB (the flag is external, so the read-only guarantee is structural, not just by convention).

**How to spot.** A phase-2 function (`performElem2` / `checkLocalEncodedMemoryStatic` / `applyFiringRecords`) mutating `body.isActive`, writing `exprOriginMap`, erasing `toBeProved`, or marking `contradictionTable`; or a split run (N>1) that races or loses theorems versus N==1.

**How to fix on violation.** Move the offending mutation out of phase 2 into phase 3's `standardProcessing` discharge path (`dischargeContradiction` / `dischargeToBeProved` / the `addExprToMemoryBlock` absorb); keep the burst capturing `FiringRecord`s only.

**See also.** [D-121](40_decisions.md#d-121), [D-123](40_decisions.md#d-123), [D-116](40_decisions.md#d-116), [D-122](40_decisions.md#d-122), [I-78](#i-78), [I-60](#i-60).

---

<a id="i-1"></a>
## I-1 Precompile structural operators on every theorem-load path

**Scope.** Compiler + prover. Every path that feeds a theorem string into `addTheoremToMemory` / `disintegrateImplication` / `addToHashMemory` — the initial load in `analyzeExpressions`, broadcast paths that forward theorems between LBs, and the compressor's proof-pool ingestion.

**Rule.** Before such a call, invoke `ExpressionAnalyzer::precompileStructuralOperators(thm)` so that raw `!(&...)` and `!(>...)` subexpressions are rewritten into their compiled `or<N>` / `existence<N>` names.

**Why.** Uncompiled structural operators become LB `exprKey`s verbatim. The hash engine keys on string form, so two LBs with the same semantic content but one carrying `!(&...)` and another carrying `or3` will never match. Downstream, disintegration asserts that every compound category (`and` / `existence` / `implication`) resolves to an entry in `compiledExpressions`; a raw `!(&...)` resolves to nothing, and the disintegration helper asserts and crashes the core.

**Spot.**

- An LB assertion fires inside `disintegrateExprCore2` or a similar disintegration helper, with an `exprKey` visibly containing `!(&` or `!(>` at string level.
- Silent CE-filter divergence: the same theorem admitted to one LB matches its hash target, admitted to another LB does not — because one LB got the compiled form and the other did not.

**Fix.** Every enumerate-theorems-to-feed-LBs site must filter through `precompileStructuralOperators`. The theorem-load paths currently known to honour this:

- Initial load of `compressed_external_theorems.txt` — at the entry of the prover.
- Broadcast paths — `mailOut.statements` preparation sites.
- Compressor pool — `Compressor::run` ingestion.

If a new path is added (e.g. a second external-theorem file, or a merging utility), it must call `precompileStructuralOperators` as the *first* operation on the raw string.

**Code.** `ExpressionAnalyzer::precompileStructuralOperators` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Referenced from the project conventions rule set.

---

<a id="i-2"></a>
## I-2 Non-`main` validityName minted only via `NameMap::encodePush`

**Scope.** Validity-stack machinery in every LB.

**Rule.** Every per-LB `validityName` other than the literal `"main"` root must be minted via `NameMap::encodePush(parentId, payload)`, where `parentId` resolves through `MAIN_ID` (either directly or via a chain of previous `encodePush` calls). Raw string concatenation of scope names — for example `"foo_" + validityName` — bypasses `ancestorsOf` registration, produces orphan roots, and breaks `comparable` / `deeperOf`.

**Why.** Scope-depth comparisons and ancestor queries depend on `ancestorsOf` being the authoritative ancestor record. An orphan scope (never registered in `ancestorsOf`) yields false answers to "is scope A deeper than scope B?", which corrupts admission logic and OR-branch bookkeeping.

**Spot.**

- A validity-scope comparison returns an unexpected answer during OR-branch handling.
- A statement is emitted into a scope that was never registered — e.g. it disappears under escalation because its ancestors are not recognised.

**Fix.** When introducing a new scope kind (hypothesis, integration goal, OR branch, sentinel), encode the scope *role* in the `encodePush` payload prefix. Do not synthesize the full scope-name string by hand elsewhere. Extract payloads via `decodeSub(stackBack(id))`, and **copy** any `decodeView` span before a nested mint (see [I-3](#i-3)).

**Code.** `NameMap::encodePush` at [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp).

---

<a id="i-3"></a>
## I-3 NameMap views into paged storage — copy before any nested mint

**Scope.** Anywhere a `StrSpan` from `NameMap::decodeView` (or `ColdStringTable::view`) or an element read of the paged validity metadata is held across a mint or deload. (`decode` now returns an **owned `std::string`** and is always safe; the heap `idToSub` / `stackOf`-reference forms are retired.)

**Rule.** A `decodeView` span aliases the cold name bytes, and a paged-metadata element aliases page memory. **Copy** (or use the owning `decode`) before any nested call that could mint or relocate a page. The old `stackOf(id)` that returned a `const std::vector<int16_t>&` into heap metadata is gone — callers use the element accessors (`stackLen`/`stackAt`/`stackBack`/`ancLen`/`ancAt`), and `encodePush` reads a parent run into a transient scratch before the append that grows the page, so no long-lived reference into the paged containers escapes.

**Why.** Any nested mint path (`encodePush`, `encodeExpression`, `nameMap.encode(...)`, …) can grow a `PagedVector` onto a fresh page, and a deload frees the pages outright. A span / element reference taken before then dangles, and subsequent reads produce garbage — or read memory handed out elsewhere.

**Spot.**

- Mysterious string corruption in a scope-name or payload during a path that involves nested encoding.
- Intermittent crashes in scope-comparison paths that are not reproducible on every run.

**Fix.** Pattern:

```cpp
// WRONG — span into cold/paged storage dangles after a nested mint
StrSpan payload = nameMap.decodeView(scopeId);
doSomethingThatMayMintAnotherScope();
// payload now may be garbage (a page relocated)

// RIGHT — owning copy (decode), or finish using the span first
std::string payload = nameMap.decode(scopeId);   // owned copy
doSomethingThatMayMintAnotherScope();
// payload still valid
```


---

<a id="i-4"></a>
## I-4 One implication-binder rule — bind every non-`u_` variable, everywhere

**Scope.** Prover, compiler, conjecturer, verifier — every implication-reconstruction and implication-checking site.

**Rule.** An implication `(>[bound-vars](premise)(body))` binds, in `>[...]`, **every** variable whose name does **not** start with `u_`; each is bound at the left-most premise that mentions it. `u_` formal parameters of GL-binary-expanded definitions stay free. This is a single uniform rule with no theorem-vs-rest distinction. There is one implementation, `reconstructImplicationFullBind`; `reconstructImplication` is a thin forwarder to it, kept only so the ~9 historical theorem-level call sites and the I-4/I-5 citation symbols stay grep-stable. A theorem carries no `u_` args, so every variable in a theorem — including every anchor slot — is bound. This replaces the retired sparse rule (old I-5: "bind only variables occurring ≥2× across chain+head"); see [I-5](#i-5) (tombstone) and [D-75](40_decisions.md#d-75).

**Why.** The old theorem-vs-rest split forced the verifier to special-case anchor-first implications (a whole "all vars changeable" branch in `check_implication`) and made the conjecturer leave body-unreferenced anchor slots free. The free/bound status of single-occurrence anchor symbols was never load-bearing for the hash engine — the kernel consumes premise text and `ce::getArgs`, not the `>[...]` binder list — so the sparse rule's stated soundness rationale was obsolete. One rule lets the same reconstruction and the same verifier path serve every implication.

**Spot.**

- A reconstruction site that emits a sparse binder (a non-`u_` variable left out of `>[...]`).
- A theorem row whose anchor slots are not all listed in the outer `>[...]`.
- A verifier `implication` failure traceable to a theorem-vs-rest handling mismatch.

**Fix.** Route every reconstruction through `reconstructImplicationFullBind` (directly or via the `reconstructImplication` forwarder). Any control-flow that historically read a *reconstructed* binder's cardinality (notably `reformulateTheorem`'s peeling-layer trigger) must be re-expressed from the chain directly — see [G-44](50_gotchas.md#g-44).

**Code.** `reconstructImplicationFullBind` and the `reconstructImplication` forwarder at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp); `reformulateTheorem` peeling-layer predicate at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp); verifier single-path `check_implication` at [`verifier.py`](../verifier.py).

---

<a id="i-5"></a>
## I-5 (retired) — folded into I-4

**Retired** by [D-75](40_decisions.md#d-75) (sandbox/unify_implications). The non-FullBind "bind only variables occurring ≥2× across chain+head" theorem-level rule no longer exists: there is now one binder rule everywhere — see [I-4](#i-4). `reconstructImplication` survives as a thin forwarder to `reconstructImplicationFullBind` so pre-existing call sites and `#i-5` cross-references stay valid. Heading kept so existing anchors do not dangle.

---

<a id="i-6"></a>
## I-6 Pass B single-input-operator gate — do not widen

**Scope.** Prover — Pass B admission.

**Rule.** The standalone fallback admission rule `isAllowedAsOperatorInput` in `prover.hpp` — the non-map-based path for `it_…` (Pass B) variables — must fire only for operators with `cfg.inputIndices.size == 1`. In practice this means `in` and `in2`. Widening to multi-input operators (`in3`, `fold`, `residual`, `interval`, `preorder`) broke Gauss summation.

**Why.** `isAllowedAsOperatorInput` admits a variable under a minimal guard set (single-input only, the lone input-arg position holds the variable, standard max-iteration guards). This guard set is insufficient for multi-input operators — without the full admission-map reasoning, multi-input operators fire far too eagerly and explode the RT surface.

**Spot.**

- A Gauss batch suddenly produces a wall-clock regression with the prover consuming many seconds per theorem where previously it took a fraction of a second. This is the classic symptom.
- A Pass B proof graph contains iteration variables that were never registered on any admission-map producer path.

**Fix.** Keep the `inputIndices.size == 1` check. When adding a new operator, either ensure it has the single-input shape (and rely on this gate), or ensure its admission map is populated from consumer-side registrations via `updateAdmissionMap`. Do not add a new operator to the fallback rule without careful RT measurement.

**Code.** `isAllowedAsOperatorInput` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). the project conventions.

---

<a id="i-7"></a>
## I-7 Pass B guarded by `!parameters.ban_disintegration` (and Pass B + back-reformulation + hypo-disintegration share that flag)

**Scope.** Prover — Pass B entry, plus back-reformulation, hypothetical disintegration, and necessity-for-equality-hypo paths.

**Rule.** The guard that decides whether Pass B runs is `!parameters.ban_disintegration` (combined with `!parameters.compressor_mode`). The same flag also gates back-reformulation ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)), hypothetical disintegration ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)), and necessity-for-equality-hypo ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) — every disintegration-shaped path in the prover. Pass B used to gate on `!parameters.incubator_mode` until 2026-04-29; the short-lived `parameters.allow_disintegration` flag introduced earlier on 2026-04-29 was collapsed into `ban_disintegration` later the same day after the per-config matrix turned out to be perfectly symmetric (`ban_disintegration == !allow_disintegration` in every existing config). See [D-27](40_decisions.md#d-27) and [D-28](40_decisions.md#d-28).

**Why.** Pre-2026-04-29, every config that needed Pass B had to also be a non-incubator config, and every incubator config lost Pass B as a consequence. That foreclosed the FTA-ladder rung-1 §4.1 proof, which needs Pass B (for `EnumerationSet2` / `interval` body disintegration) AND `multiplyImplication` off AND `incubator_mode` true (for contradiction LBs / skip CE filter). The decoupling lets a config independently set them. The `incubator_mode`/`ban_disintegration` collapse keeps the flag set minimal — `incubator_mode` still governs the other things it always did (the head-already-registered short-circuit, OR-generation suppression, integration reformulation, tempArgs assert, conjecturer behaviours), but no longer governs disintegration.

**Spot.**

- Pass B unexpectedly fires (or doesn't fire) for a config that has `incubator_mode=true`. Check `ban_disintegration` first, not `incubator_mode`.

**Fix.** The guard must read `if (!parameters.compressor_mode && !parameters.ban_disintegration)` at the Pass B entry. Do not substitute `!incubator_mode`.

**Code.** Pass B entry at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Sibling decoupling for `multiplyImplication` lives at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp), gated by `parameters.allow_multiplication` (with the `!ceFilteringActive` carve-out preserved).

**Local-premise refinement (D-29, 2026-04-29).** Inside `checkLocalEncodedMemoryStatic` at [`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp) (the body moved from `prover.cpp`, 2026-05-04), when both `parameters.incubator_mode == true` and `!parameters.ban_disintegration` (the SE2-migration combination), disintegration of a matched rule's head is additionally gated by a two-part rule:

1. **Anchor LB always blocks.** If `memoryBlock.exprKey` starts with `"(" + anchorInfo.name` (i.e. this is the anchor LB itself), `doNotDisintegrate = true` unconditionally. The anchor LB never disintegrates in this mode — its job is to handle external rules and broadcast their conclusions to descendants without local fan-out.
2. **Non-anchor LB needs a local premise.** Otherwise, gate on at least one of the matched premises being in the LB's `intLocalEncodedStatementsSet` — an O(1) probe with `packStatementKey(originalId, validityId)` taken directly from the request's int rows (`req.intExprs`), no decode, no string-key construction (re-keyed from the former `std::set<EncodedExpression>` by `D-129`). The set is populated by both the LB's own work AND `prehandleAnchor` (the anchor x-prefixed `(in[X, N])` deposits at `prover.cpp/8236`); without rule (1), the anchor LB's own anchor-deposit-saturated set would always satisfy this check and the gate would be inert.

Without this gate the prover crashed with NameMap exhaustion at `prover.hpp` after burst 2 (~4570 expressions) when fold/sequence/etc. external rules from a prior incubator batch fanned out across local `(in[X, N])` rows. With the gate, the same prove pass plateaus at ~7730 expressions and runs to completion. The gate is dormant for every other config. See [D-29](40_decisions.md#d-29).

---

<a id="i-8"></a>
## I-8 Trivial equality `(=[x,x])` forbidden in head only, allowed in premises

**Scope.** Conjecturer + prover.

**Rule.** Conjectures whose *head* is the trivial equality `(=[x,x])` (same variable on both sides) must be rejected. Trivial equalities in premises are fine — they reduce to a tautological hypothesis that the prover simply discharges.

**Why.** `(=[x,x])` as a head is universally true independently of any premise — the conjecturer would generate an explosion of useless tautology theorems otherwise.

**Spot.**

- Conjecturer output `conjectures.txt` contains `(=[v1,v1])` heads.

**Fix.** Guard in the conjecturer's head-construction path: reject any proposed head where both arg-positions of `(=[...])` resolve to the same variable name.

**Code.** Conjecturer filter; see [`10_pipeline/02_conjecturer.md`](10_pipeline/02_conjecturer.md) `[STUB]`. the project conventions for the equivalent incubator-mode handling.

---

<a id="i-9"></a>
## I-9 Equality mirror guarded by `args[0]!= args[1]`

**Scope.** Compiler — mirror generation.

**Rule.** When generating a mirrored form of a theorem, the equality mirror emission is guarded by `args[0]!= args[1]`. Trivial equalities are not mirrored.

**Why.** Mirroring `(=[a,b])` produces `(=[b,a])`, which is semantically distinct. Mirroring `(=[a,a])` produces `(=[a,a])` again — no new information. Emitting the identical mirror wastes space and can silently break deduplication checks.

**Spot.**

- Duplicate mirror entries in `reshuffled_mirrored_conjectures.txt`.

**Fix.** Preserve the `args[0]!= args[1]` guard in `createReshuffledMirrored`.

**Code.** `createReshuffledMirrored` at [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp).

---

<a id="i-10"></a>
## I-10 Chapter v-numbering seeded from theorem expression

**Scope.** `process_proof_graphs.py` — variable renaming.

**Rule.** The renamer must assign `v1`, `v2`, … to non-anchor variables by scanning the **theorem expression** left-to-right *first*, then continuing with any remaining variables found in chapter lines. This seeding ensures chapter and global theorem list share the same v-numbering.

**Why.** Without seeding from the theorem, the chapter might assign `v1` to a variable that appears first in chapter lines — which is a different variable from the one `v1` refers to in `global_theorem_list.txt`. Verifier checks that cite the theorem by v-number then spuriously fail.

**Spot.**

- Verifier fails `task formulation` or `theorem` checks with "expected v1, got v3" messages.

**Fix.** Preserve the four-priority renaming in `process_proof_graphs.py` (anchor → theorem scan → chapter scan → `_copy` derivation) — see the project conventions "Variable renaming in process_proof_graphs.py" for the full specification.

**Code.** `process_proof_graphs.py`. See [`10_pipeline/06_process_proof_graph.md`](10_pipeline/06_process_proof_graph.md) `[STUB]`.

---

<a id="i-11"></a>
## I-11 Anchor slots ARE bound in a theorem's `>[...]` (inverted)

**Scope.** Theorem construction — prover + conjecturer.

**Inverted** by [D-75](40_decisions.md#d-75) (sandbox/unify_implications). The previous rule said the opposite: *anchor-slot names must never appear in any theorem's `>[...]`*. Under the unified binder rule ([I-4](#i-4)) that is no longer true and was never load-bearing for the hash engine.

**Rule (current).** Every variable in a theorem is bound (a theorem carries no `u_` formal parameters), so every anchor slot the theorem references **does** appear in the outer `>[...]`, in occurrence order. A theorem is an implication with zero unchangeable (`u_`) arguments. The anchor atom still pins the slots positionally; binding them universally is the deliberate uniform semantics, not a meaning change the prover acts on.

**Why the old rule was dropped.** The old rationale — "binding an anchor slot makes the theorem quantify over all instantiations and breaks the anchor-handling tag chain" — did not hold: the hash kernel consumes premise text and `ce::getArgs`, never the `>[...]` binder list, and the anchor-handling tag chain keys on the anchor atom, not on binder membership. The split's only real effect was forcing a verifier special-case and a conjecturer divergence; [I-4](#i-4) removes both.

**Spot.**

- A theorem-construction site that still emits a *sparse* anchor binder (an anchor slot referenced by the theorem but absent from `>[...]`) — that is now the bug, the inverse of the old spot condition.

**Fix.** Bind every anchor slot via the unified reconstruction ([I-4](#i-4)) / the conjecturer's anchor-attach binder. `_find_digit_args` / `_find_immutable_args` are unaffected — they subtract the anchor-atom args, so the digit/immutable sets are identical whether or not the slots also appear in `>[...]`; they were never the enforcement mechanism for the old rule and continue to serve induction unchanged.

**Code.** Unified binder at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (`reconstructImplicationFullBind`); conjecturer anchor-attach binder at [`conjecturer.cpp`](../GL_Quick_VS/GL_Quick/src/conjecturer.cpp) (`connectExpressions` / `connectExpressionsInt`). `_find_digit_args` / `_find_immutable_args` at [`verifier.py`](../verifier.py).

---

<a id="i-12"></a>
## I-12 `addStatement` applies equivalence classes to `!(=[a,b])` one-sidedly

**Scope.** Prover — `addStatement`.

**Rule.** When `addStatement` receives a negated equality `!(=[a,b])`, it applies equivalence classes to emit sibling inequalities, but only one-sidedly:

- For each `c ∈ class(a) \ {a}` — emit `!(=[c, b])`.
- For each `d ∈ class(b) \ {b}` — emit `!(=[a, d])`.

The symmetric cross-product (both args substituted simultaneously) is **deliberately not emitted**.

**Why.** Two-sided expansion would blow up combinatorially (|class(a)| × |class(b)| emissions per input) without clear semantic gain — anything the cross-product would conclude is already derivable by two one-sided steps.

**Spot.**

- A proof unexpectedly stalls because a two-sided-expansion fact that should be derivable is not being emitted.

**Fix.** Before changing this, profile the alternatives. The two-sided form has never been implemented, and the one-sided form is the load-bearing simplification.

**Code.** `applyEquivalenceClassToNegatedEquality` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Caller: `addStatement` at [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp).

---

<a id="i-13"></a>
## I-13 ~~`ChunkPool` uses static `char[]` — never `malloc`/`new`~~ — retired, superseded by [I-95](#i-95)

**Status.** Retired. The `ChunkPool` class this invariant named no longer exists in the codebase. The contract it carried — hot-path allocators pre-allocate fixed storage up front and never touch the heap per object (a pattern the user requested explicitly at least four times) — lives on, generalized, in the statification static-memory hierarchy: one program-start reservation dispensed as fixed-size blocks (`GlobalMemoryManager`, [`memory_infra/global_memory_manager.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/global_memory_manager.hpp)). See [I-95](#i-95) and [`20_core_concepts/09_static_memory.md`](20_core_concepts/09_static_memory.md).


---

<a id="i-14"></a>
## I-14 Never destroy git history

**Scope.** Process.

**Rule.** No `git push --force`, no `git rebase` that removes commits, no deletion of branches with unmerged work. The commit log is sacred and eternal. This is the ONE absolute restriction in the project conventions.

**Why.** Multi-agent collaboration depends on the history being a reliable shared reference. A force-push silently invalidates other agents' local branches and can erase in-flight work.

**Spot.**

- A branch that was at commit `X` on origin no longer contains `X` in its history.

**Fix.** Preventative: never run history-rewriting commands against shared branches. Detection: `git reflog` of the affected local clone is the only recovery.

**Code.** the project conventions.

---

<a id="i-15"></a>
## I-15 `git reset --hard` only; never soft or mixed

**Scope.** Process.

**Rule.** When moving `HEAD` between commits, always use `git reset --hard`. Never use `--soft` or `--mixed`.

**Why.** Soft/mixed resets leave the working tree polluted with stale changes that look like "your" changes but belong to the commit you just moved away from. The next commit then unintentionally rolls them in. Hard reset is the only form that guarantees a clean slate.

**Spot.**

- Unexpected "modified" entries in `git status` after a `git reset`.

**Fix.** `git reset --hard <target>`. If you meant to preserve local changes, stash them *before* the reset.

**Code.** the project conventions.

---

<a id="i-16"></a>
## I-16 `verifier.py` is sacred — failures are real bugs

**Scope.** Verifier.

**Rule.** Never modify `verifier.py` without explicit user consent. When a verifier check fails, it is a real bug, not a "false positive". The correct response is to find the upstream cause in the prover / processor / compiler, not to "make the verifier pass".

**Why.** The verifier is the sole independent oracle for proof-graph correctness. Relaxing a checker to make a failure disappear hides real semantic errors. The verifier's independence (no imports from `expression_utils` or prover code, own copies of every algorithm) is explicitly designed to resist this pattern.

**Spot.**

- A PR diff touches `verifier.py` with the effect of loosening a check.
- A commit message claims "verifier false positive fixed".

**Fix.** If you believe the verifier is genuinely wrong, raise it with the user with a concrete example. Otherwise, treat the failure as a bug in whatever produced the data.

**Code.** See also [`10_pipeline/08_verifier.md`](10_pipeline/08_verifier.md).

---

<a id="i-17"></a>
## I-17 `savedStartInt` freshness check assumes one monotonic counter

**Scope.** Prover — Pass B.

**Rule.** `disintegrateExpr2`'s `savedStartInt` freshness check assumes there is exactly one monotonically-increasing counter feeding the iteration-level numbers. A parallel counter breaks the freshness contract.

**Why.** The freshness check tests whether a level seen now is strictly greater than the saved start — i.e. whether it was produced after this Pass B invocation began. Two counters advancing independently produce values that are not totally ordered, so the freshness test returns wrong answers.

**Spot.**

- Pass B produces iteration variables that should have been rejected as stale, or vice versa.

**Fix.** Maintain a single global counter. If a second counter is needed for a legitimate reason (e.g., a separate numbering scheme for a new scope kind), update `savedStartInt` logic to reason about the Cartesian product, not just the scalar.

**Code.** `disintegrateExpr2` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp).

---

<a id="i-18"></a>
## I-18 Induction scheduled on a bound variable must first prove its typing

**Scope.** Prover — induction setup.

**Rule.** Before scheduling induction on a bound variable `n`, the prover must prove `(in[n, N])` from the current chain, where `N = anchor_args[0]` — the first slot of the anchor (in Peano, `N` itself). The typing sub-theorem inherits the original chain's non-anchor premises and anchor; only the head changes.

Induction succeeds **iff all three discharges complete**:

1. typing: `(>[bounds](Anchor)(chain_premises)(in[n, N]))` — direct-proof only.
2. base: `(>[bounds](Anchor)(chain_premises[n:= i0])(head[n:= i0]))`.
3. step: `(>[bounds](Anchor)(chain_premises[n:= s(m)], head[m:= n])(head[n:= s(m)]))`.

**Why.** Without the typing discharge, induction on `n` is soundness-unsound for any `n` not in `N` — the proof's range silently widens from "all natural numbers" to "all entities whatsoever". The current branch was created precisely because a negative Peano theorem was going through via this unsound path.

**Spot.**

- A theorem in `global_theorem_list.txt` with method `induction` whose induction variable has no typing derivation in its premises.
- A `check_zero` or `check_induction_condition` chapter with no corresponding `induction_typing` chapter at `<N-1>`.

**Fix.** **Shipped.** The prover now proves `(in[digitArg, N])` as an auxiliary induction triad before promoting an induction theorem. The chapter set carries `<N>_induction_typing.txt` files holding the typing derivation. The verifier's `induction typing` checker walks every `method = induction` row in `global_theorem_list.txt` and verifies its accompanying typing chapter. See [`docs/agentic_swdd/induction_typing_plan.md`](induction_typing_plan.md) for the historical implementation plan.

**Code.** Implemented in `parameters.hpp` (config flag), `memory.hpp` (induction-triad bookkeeping), `prover.cpp` (induction setup + promotion gate), `prover.hpp`, `visualizer.cpp` (chapter rendering), `generate_full_proof_graph.py` (induction-typing chapter ingest), `process_proof_graphs.py` (chapter emission), `run_modes.py`, `run_modes.cpp`, `verifier.py` (`induction typing` checker registry).

---

<a id="i-19"></a>
## I-19 Asserts are first-class — never weaken or remove to pass a test

**Scope.** Process — C++ and Python alike.

**Rule.** Add `assert(...)` whenever a function has a precondition that the caller already guards. GL must have all states defined; an unguarded-but-expected state is a silent contract. If an assert fires, it has surfaced a previously unknown code path — do not suppress it. The correct response is to understand why that path is reached and fix the caller.

**Why.** GL's correctness is deterministic — every state is definable or the system is in an inconsistent state. Asserts are the contract-enforcement surface. Weakening an assert to pass a test hides the exact bug the assert was designed to catch.

**Spot.**

- A PR comment reading "removed assert because it was firing" — without an accompanying explanation of *why* the caller reached the precondition-violating path.
- A test that previously passed is "fixed" by weakening an assert rather than by fixing the test harness or the code under test.

**Fix.** Find the upstream caller. Document why it reached the asserted-violating state. If the assert is genuinely wrong, discuss with the user before changing it.

**Code.** the project conventions.

---

<a id="i-20"></a>
## I-20 Auto-commit + push on source/config changes, detailed message

**Scope.** Process.

**Rule.** After each prompt that modifies source code or configs (`*.cpp`, `*.hpp`, `*.h`, `*.py`, definition files in `files/definitions/`, config JSON in `files/config/`):

- `git add <changed files> && git commit -m "<prompt summary>"`.
- Verify with `git status` that the commit succeeded.
- Push the current branch: `git push -u origin <current-branch>`.

Commit message must include: (1) one-line summary of what changed, (2) root cause or motivation — why this change was needed, (3) what was investigated/ruled out during debugging if applicable, (4) before/after results if measurable.

HTML or other non-source files do **not** require auto-commit.

**Why.** Multi-agent collaboration on the private origin depends on every source change being visible promptly. A detailed commit message is the only artifact a future agent has to reconstruct *why* a change was made — code shows *what*, message shows *why*.

**Spot.**

- A local branch diverges from origin by many unpushed commits.
- A commit message reads only "fix bug" with no motivation.

**Fix.** Follow the rule as stated. If a commit goes in with a terse message, the next commit's message should include the missing motivation for the prior one.

**Code.** the project conventions.

---

<a id="i-21"></a>
## I-21 `sameIterationInternalMail` cleared immediately after its absorb (not at end of burst)

**Scope.** Prover — the `standardProcessing` absorb (its `internalMailIn` drain-then-clear). `sameIterationInternalMail` is drained by the post-burst call in `performElemPhase3` (I-63).

**Rule.** `Memory::sameIterationInternalMail` (integration-revival channel) is cleared **immediately after its absorb loop**, within the same block. Inserts during the hashburst body (from `applyEquivalenceClassToRejectedMapIntegration` and `revisitRejectedIntegration2`) must survive to the NEXT burst's absorb. The clear-right-after-absorb adjacency is the load-bearing part; the absolute position within the function is not (the invariant is position-agnostic).

**Branch history.** On, the absorb (and its clear) was relocated post-fixpoint by commit (ASIC 0.1 reshuffle 3/8), and then reverted back to pre-fixpoint on 2026-05-20 ([D-79](40_decisions.md#d-79)) after the post-fixpoint placement was shown to push `__contradiction__(=[a,b])` LBs past `MAX_NAME_IDS` during the Peano-incube combinatorial substitution phase. Legacy `mailIn` is cleared in the *same* pre-fixpoint block (right after its `status=3` drain), so both inboxes are consumed-and-cleared in the same pre-fixpoint pass; the only distinction is absorb status (`sameIterationInternalMail` = `status=1`, full disintegration; `mailIn` = `status=3`, existence-banned).

**Why.** The cleanup must be adjacent to the absorb because the *producer* runs during the burst body (eq-class rewrites and admission-key revisits inside `addStatement`). If `sameIterationInternalMail` were cleared at a later point a mid-burst producer could precede, entries produced mid-burst would be lost. Clearing immediately after the absorb loop guarantees every insert during a burst body lands intact in the next burst's inbox, regardless of where in the burst cycle the absorb sits.

**Spot.**

- A `sameIterationInternalMail` clear placed anywhere other than directly after its own absorb loop (e.g. folded into the end-of-burst cleanup, or before the absorb).
- Revival statements missing after an equi-class rewrite — next burst's absorb sees an empty `sameIterationInternalMail` when traces show mid-burst inserts did happen.

**Fix.** Clear the `sameIterationInternalMail` columns inside the absorb, on the lines immediately following its absorb loop. See `prover.hpp::standardProcessing` — the `internalMailIn` drain-then-clear (the "Cleared on exit" parameter contract).

**Code.** `memory_infra/cold_mail.hpp` — `ColdMail` bound as `Memory::sameIterationInternalMail` (id-form since I-102; historically a heap `Mail`, typed since 2026-05-07 per [D-53](40_decisions.md#d-53), renumbered from main's D-46); `prover.hpp::standardProcessing`, the internal-mail absorb-then-clear (post-burst driver: `performElemPhase3`).

---

<a id="i-22"></a>
## I-22 `admissionMapIntegration` entry NOT cleaned on successful revival or consumption

**Scope.** Prover — `revisitRejectedIntegration2`, `applyEquivalenceClassToAdmissionMapIntegration`, and any future integration-side hook that fires on consumed admission templates.

**Rule.** When an integration-side rejection is revived (via equi-class rewrite or new-admission-key trigger), the `admissionMapIntegration` / `admissionSetIntegration` entry that enabled the revival is **not** erased afterwards. Asymmetric with algebra: `revisitRejected2` (the algebra counterpart) calls `cleanAdmissionMap` after successful revival, and `cleanAdmissionMap`'s D-62 canonicalization closure additionally erases every algebra admission key with the same canonical form under the current classes. The integration counterparts deliberately have NO such cleanup or closure — no analog of `cleanAdmissionMap` fires on integration revival, and no analog of D-62's closure fires when an integration admission template is consumed.

**Why.** A single integration admission template can admit multiple distinct `int_` witnesses over the proof lifetime. The admission rule is a *template*; consuming it once does not exhaust it. Cleaning it would disable future revivals of different compound shapes whose rewritten keys happen to match the same template. Algebra can afford to clean because its revisit goes through `addExprToMemoryBlock` which can re-populate admission maps on future need; integration revival is mailIn-only and has no such re-population pathway — once gone, the template is unrecoverable.

**Spot.**

- Any `cleanAdmissionMap(...)`-on-hit call — or a revival of the deleted `cleanUpAdmissionMapIntegration` sweep as an on-hit call — added inside `revisitRejectedIntegration2` or inside the post-insert revival path of `applyEquivalenceClassToAdmissionMapIntegration`. (`cleanUpAdmissionMapIntegration` was deleted as dead code 2026-07-02; the prohibition on recreating any such on-hit cleanup stands.)
- A canonicalization-closure block added to `cleanAdmissionMap`'s `markerIsOutput` branch that actually finds and erases integration-map entries (the existing `admissionMapIntegration.erase(...)` calls inside `cleanAdmissionMap` are key-form no-ops — algebra uses bare-marker keys, integration uses u_-form keys — and remain as harmless vestiges; adding a u_-form lookup or a per-canon-class walk that does match would violate this invariant).
- A second integration revival of the same marker-template failing because the admission entry was silently removed by a previous revival.

**Fix.** Do not add cleanup. The `rejectedMapIntegration` entry itself IS erased after revival (the rejection is resolved), but the admission-map entry stays live. The retired post-class-update sweep `cleanUpAdmissionMapIntegration` ([I-43](#i-43); definition deleted 2026-07-02) dropped only non-canonical entries after class formation — canonicalization-driven, not consumption-driven, so it never violated this invariant; its D-106 successor, the inline re-key inside `applyEquivalenceClassToAdmissionMapIntegration`, is likewise canonicalization-driven and equally compatible.

**Code.** `prover.cpp::revisitRejectedIntegration2`; `prover.hpp::applyEquivalenceClassToAdmissionMapIntegration` (the post-insert revisit-fire path); `prover.hpp::cleanAdmissionMap` (the existing key-form-mismatched `admissionMapIntegration.erase` lines — benign, do not "fix" them).

---

<a id="i-23"></a>
## I-23 Spontaneous compact operator names are stable across batches

**Scope.** C++ compiler (`ExpressionAnalyzer` constructor + `excludeRepetitions` + `compileCoreExpressionMapCore`) and Python pipeline (`run_modes.py::_run_batch`).

**Rule.** Every spontaneous operator name (`implication<N>`, `existence<N>`, `or<N>`, `and<N>`) allocated by any batch in a run keeps the same `<N>` for every later batch in the same run. **Every batch — incubator and main alike — contributes to and reads from the shared registry.** The shared registry file `files/GL_binaries/GL_binary_shared.json` is the canonical source of truth; per-batch files `GL_binary_<Tag>.json` are seeded from shared before each `gl_quick.exe <Tag>` invocation, and any new spontaneous-allocator entries the batch creates are merged back into shared after the invocation returns. The four spontaneous-operator counter members (`implCounter`, `existenceCounter`, `andCounter`, `orCounter`) are seeded from the maxima of names already in the per-batch file rather than zeroed; only `variableCounter` resets to zero per batch.

**Why.** The proof-graph emission pipeline keys on string-form compact names. `compiled_theorems.txt` (the Python pruner's essential set) and `raw_proof_graph/global_theorem_list.txt` (the Python pruner's all-theorems set) must use the same name for the same logical fact, otherwise `_prune_proof_graph` computes an empty intersection for that theorem and drops both forms; chapter contents that cite the Gauss-batch internal name then fail the verifier's origin check because the renamed template is absent from `state.global_theorems`. The chapter-85 successor-existence failure under sandbox/cancellation_v2 was the concrete incident. See [D-22](40_decisions.md#d-22) for the architecture record.

**Spot.**

- A spontaneous operator appears in `compiled_theorems.txt` under one name (e.g. `existence3`) and in `raw_proof_graph/global_theorem_list.txt` under a different name (e.g. `existence2`) for the same structural form.
- `process_proof_graphs.py:_prune_proof_graph` drops a theorem because `essential ∩ all_thm_exprs` is empty.
- The verifier reports an `origin` failure citing an implication template `(>[…](AnchorXxx[…])(<spontaneous>[…]))` that is absent from `files/processed_proof_graph/global_theorem_list.txt`.
- `loadGlBinary` log line is missing or shows `loaded 0 entries` after the first batch of a run that used spontaneous operators in an earlier batch.

**Fix.** Verify `_seed_per_batch_binary` runs immediately before every `run_gl_quick(tag)` call and `_merge_into_shared` runs immediately after; the merge admits every tag (no `Incubator` early-return per [D-61](40_decisions.md#d-61)) and filters by category only — atomic entries (`=`) are excluded, every entry with a spontaneous category is merged. Verify `loadGlBinary` is called from the `ExpressionAnalyzer` constructor before any compilation work; verify `precompileStructuralOperators` was called only after the loader has populated `repetitionExclusionMap` so `excludeRepetitions` finds existing entries. The C++ lookup is keyed by `(elements, category)` per [D-60](40_decisions.md#d-60) — without that key choice, cross-batch sharing is unsound (an incubator-allocated implication and a main-batch existence with the same elements would collide). If a config-level change has invalidated existing shared entries (e.g. an `existence_variable_position` change), delete `files/GL_binaries/GL_binary_shared.json` manually and rerun — there is no automatic staleness detection. Never edit `GL_binary_shared.json` by hand to "fix" a name collision; if the registry contains an unwanted entry, wipe and rerun.

**Code.** `GL_Quick_VS/GL_Quick/src/visualizer.cpp::loadGlBinary`; `GL_Quick_VS/GL_Quick/src/prover.cpp` constructor (counter init + loader call); `run_modes.py::_seed_per_batch_binary`, `_merge_into_shared`, and `_run_batch`.

---

<a id="i-24"></a>
## I-24 `multiplyImplication` may not equate two distinct free `u_*` anchor parameters

**Scope.** Prover — `ExpressionAnalyzer::multiplyImplication` and the Python verifier's `check_equalize_variable`.

**Rule.** When `multiplyImplication` enumerates Bell partitions over the `(1)`-typed argument set of a rule, any partition whose equivalence classes contain two or more distinct free `u_*` anchor parameters in the same class must be skipped. Bound→bound merging (the standard Bell partition use case) and bound→free merging (specialising a bound variable to a known anchor) are both permitted; only free→free with distinct names is forbidden. The verifier enforces the same gate on the Python side as a defence in depth: `check_equalize_variable` parses every `>[…]` binder to classify each name as bound or free, then rejects any `(orig_arg → copy_arg)` mapping where both ends are free and the names differ.

**Free ⟺ `u_` prefix (post [D-75](40_decisions.md#d-75)).** Under the unified binder rule, EVERY non-`u_` variable is bound in some `>[…]`; only `u_*` formal parameters are ever free (the `u_` prefix never appears in a binder). So "classify by `>[…]` membership" now coincides exactly with "free ⟺ `u_`-prefixed". The gate is **not weakened**: the `u_*` parameters it protects are still classified free (their classification never depended on the now-widened theorem binders — `u_` is excluded from `>[…]` before and after), so distinct-free-`u_*` merges are still rejected. Theorem anchor slots that are now bound (e.g. `+`, `*`, `N`) were never the gate's target — merging two *bound* variables is the legitimate Bell-partition case and stays permitted. The gate and its code are unchanged; only this rationale is reworded.

**Why.** Free `u_*` parameters are pre-bound to specific elements of the rule's underlying definition sets. Equating two distinct ones rewrites a free slot of the rule body and emits a logically stronger rule than the source — provable conclusions then include theorems the source rule never authorised. The chapter-1115 incubator regression (HTML title *"sum(i=0..0) ≠ 0"* — a mathematically false claim derived through this exact path) is the load-bearing incident; commit cf358271 had deliberately removed an earlier `if (hasDoubleU) continue;` skip while pursuing a u\_-equalisation extension, and the unsoundness sat dormant on Peano/Gauss until the FTA-ladder branch exercised the bad shape. See [D-25](40_decisions.md#d-25) and the prover-chapter [Soundness gate](10_pipeline/04_prover.md#multiplyimplication-bell-partition-equalisation) write-up.

**Spot.**

- A `multiplied from` chapter row whose source and copy differ in a free anchor slot — e.g. source `existence2[v2,N,0,id,0,1,s,+]`, copy `existence2[v2,N,0,id,0,0,s,+]` (slot 6 collapsed `1 → 0`).
- The Python verifier emits a `multiplied from` failure with the row payload above.
- A theorem appears in `theorems.txt` whose statement is mathematically false and whose chapter graph traces back through a `multiplied from` step that touches a free anchor.

**Fix.** Re-enable the `hasDoubleU` skip inside `multiplyImplication` (commit 992fe2b1 restored it). Independently, keep the `check_equalize_variable` free-anchor-merge guard live (commit 38701644) so that any future regression that bypasses the prover-side gate is caught before the chapter ships.

**Code.** `multiplyImplication` at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp); `hasDoubleU` skip at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). Verifier guard: `check_equalize_variable` at [`verifier.py`](../verifier.py); `_extract_bound_vars` helper colocated.

---

<a id="i-26"></a>
## I-26 Mail-out statements are MAIN-ONLY; mail-out exprOriginMap is ALL-SCOPES

**Scope.** Mail subsystem — [`addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`addExprToMemoryBlock`](../GL_Quick_VS/GL_Quick/src/prover.cpp), and every site that inserts into `Memory::mailOut.statements`.

**Branch note (ASIC 0.1 reshuffle).** The `Mail::implications` channel was retired (commit, [D-78](40_decisions.md#d-78)). Implications now travel as the compact `(implication<N>[…])` form deposited into `Mail::statements` by the deferred-compaction drain (D-76). The MAIN-ONLY gate therefore applies only to the statements channel on this branch; the implication-specific receiver-side `addToHashMemory(... "main")` hard-coding is gone. The wording below covers both sides of the contract — the pre-deletion (`Mail::implications`) wording is preserved as historical context for the merge with main HEAD.

**Rule.** A non-main expression must NEVER enter `mailOut.statements`. The channel is gated at the sender:

- **Statements**: `addStatement`'s `local`-branch push into `mailOut.statements` is conditional on `validityName == "main"`.
- **(Retired on this branch) Implications**: `addExprToMemoryBlock`'s post-disintegration loop on main HEAD gates the `mailOut.implications.insert` on `impValidity == "main" && allowedForMail(impStr, memoryBlock)`. On the channel is gone; the compact-form deposit into `mailOut.statements` (via `recordPendingCompaction` + the deferred drain) carries the same MAIN-ONLY contract through the statements-channel gate.

`mailOut.exprOriginMap` is the exception: it carries entries for ALL scopes per `trackExpansionHistory`'s convention. Receivers' merged `body.exprOriginMap` therefore covers the sender's full provenance graph — main, hypothetical, ordis-branch, and orint-branch alike — even though the rules themselves stay local to the sender at non-main scopes.

**Why.** Hashmem rules are conditional on the hypothesis stack at their install scope. A rule at `v=main_boundary_(impl24[…])` is conditional on the impl24 hypothesis being active; shipping it to another LB and reinstalling at hardcoded `v=main` (the receiver's implication absorb — historically the `mailIn.implications` block, today the compact `(implication<N>[…])` statements drain in the phase-1 `standardProcessing` absorb, [I-54](#i-54)) discards that conditionality and produces unsound rule firings — and, secondarily, breaks visualizer's `buildStack` walk because the firing recorder writes the dep at `(rule, "main")` while the rule's `exprOriginMap` entry came in keyed at the sender's deeper scope. The `sameIterationInternalMail` channel is the per-LB cross-cycle inbox that DOES carry validity for non-main revival messages — see [I-21](#i-21).

**Spot.**

- A new `mailOut.implications.insert` site that does not gate on `validityName == "main"` (or its equivalent).
- A new `mailOut.statements.insert` site whose `ExpressionWithValidity` is constructed with anything other than the literal `"main"` for `validityName` — every routing-channel sender must wrap with `ExpressionWithValidity(expr, "main")` (the sender's own `validityName == "main"` gate guarantees the surrounding scope, but the EWV constructor argument is what gets shipped).
- A change that adds a validity field to the `mailOut.implications` 5-tuple beyond the existing 5th element. The tuple's 5th element carries OR-derivation context (`orImpl`/`theorem`), not install scope — adding a parallel validity field is a code-smell that the install gate is being bypassed.
- A receiver-side `addToHashMemory` in the `mailIn.implications` absorb that uses anything other than `"main"` for the install validity. The hardcoded `"main"` is the contract counterpart of the sender gate.
- A new `mailIn.statements` consumption site (or a refactor of the existing CE-mode / normal-mode absorb loops) that does not assert `it->first.validityName == "main"` before consuming the EWV. The runtime enforcement of I-26 for the statements channel is the per-element `assert(vName == "main")` carried inside both absorb loops (the `standardProcessing` statements drain, driven pre-burst by `performElemPhase1`). The `mailIn.statements` shape is `pair<ExpressionWithValidity, levels>` per [D-53](40_decisions.md#d-53) (renumbered from main's D-46); the EWV's `validityName` is read at consumption rather than hardcoded because the non-`"main"` branch of the same code shape is exercised by the per-LB `sameIterationInternalMail` channel on a separate code path (the top-of-burst drain block has no main-only assert).

**Fix.** Add the `impValidity == "main"` gate at the sender. If a non-main rule legitimately needs to cross LB boundaries, route it via `sameIterationInternalMail` (the integration-revival channel, which DOES carry validity in the tuple) — but only if the receiver actually needs the rule at the same non-main scope, which is rare.

**Code.** Sender gates: [`prover.hpp::addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (`mailOut.statements` site, plus `addEquality` / `addNegatedEquality` / `addAnchorHandling` / `applyEquivalenceClass` deposit branch — every routing sender wraps with `ExpressionWithValidity(expr, "main")`), [`prover.cpp::addExprToMemoryBlock`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (`mailOut.implications` site). Receiver absorbs: [`prover.hpp::standardProcessing`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (pre-burst driver `performElemPhase1`) — the `mailIn.statements` absorb loops carry `assert(vName == "main")` per-element as the runtime enforcement of the statement-channel contract; the former `mailIn.implications` block hardcoded `"main"` directly (that tuple channel is deleted — [I-54](#i-54)). History: implication-side gate landed with [D-34](40_decisions.md#d-34) in 2026-05-02; pre-D-34 the implications side was missing the gate, ran latent until the FTA-rung-1 §4.1 chain reached contradiction-LB closure, then crashed `visualizer.cpp buildStack`. Statement-side EWV migration + receiver assert landed 2026-05-07 ([D-53](40_decisions.md#d-53), renumbered from main's D-46); pre-migration the receiver hardcoded `"main"` for both channels.

---

<a id="i-27"></a>
## I-27 Site F / Site H — ancestor-scan dedupe at `addExprToMemoryBlock` entry

**Scope.** Prover — [`addExprToMemoryBlock`](../GL_Quick_VS/GL_Quick/src/prover.cpp) at the function entry, before any disintegration / kernel logic.

**Rule.** Every deposit into `addExprToMemoryBlock` first walks `memoryBlock.nameMap.ancestorsOf[valId]` (which includes `valId` itself plus every strict prefix scope registered via `encodePush`). If the expression is already in `memoryBlock.intKnownStatements` at ANY of those validities, the function returns immediately without depositing.

Two parallel scans, both at function entry:

- **Site F** ([`prover.cpp–4804`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) — duplicate suppression. Returns early if any ancestor scope already has the expression.
- **Site H** ([`prover.cpp+`](../GL_Quick_VS/GL_Quick/src/prover.cpp)) — blacklist scan over `memoryBlock.nameMap.ancestorsOf[valId]` against `intValidityNamesToFilter`. Filters out depositary scopes that have been blacklisted.

**Why.** Comparable-scope semantics: a fact at scope `S` is observably true at every descendant of `S`. Re-installing the same fact at a strict descendant adds no information the prover couldn't already see via inheritance. Site F's early return prevents the redundant deposit and short-circuits any downstream work (disintegration, mail emission, ordisMerge convergence registration) for the duplicate.

**Spot.**

- New code that pushes into `intKnownStatements` outside `addStatement`'s `local`-branch.
- A new entry path into the kernel that bypasses Site F (e.g. directly calling `addExprToMemoryBlockKernel` without the outer dedupe). This is generally wrong unless the caller has independently established that the deposit is non-redundant.
- A flat-namespace `validityName` (no `_boundary_`) where `ancestorsOf[valId]` degenerates to `{valId}`. Safe — the scan still runs, just with a single entry — but worth noting for sanity.

**Fix.** Trust Site F. If the deposit "should" land but doesn't, the diagnostic is to check `intKnownStatements` for any ancestor entry — almost always the redundancy is real.

**Compressor mode bypass.** Site F is gated on `!parameters.compressor_mode`. The compressor stage allows redundant deposits at strict-descendant scopes for reasons specific to the compression algorithm (currently undocumented in detail; an open question for the compressor chapter).

**Interactions.**

- **D-34's `ordisMerge`** pushes converged expressions onto `sameIterationInternalMail` for the next-hashburst absorb. The absorb's `addExprToMemoryBlock` call hits Site F. If the converged expression is already at a strict ancestor (e.g. via an unrelated derivation chain), the deposit short-circuits silently and no `or convergence` entry is emitted at the convergence target scope. This is correct behaviour — see the impl24-down case in [D-34](40_decisions.md#d-34)'s verification trace, where the same `(or2[2,repl_lev_1_0,6])` orSig at two stack positions races and the impl24-internal promotion gets dedupe'd by Site F because the top-level OR-convergence promotion already landed the fact at `v=main`.
- **The `_orint_` block at [`prover.cpp+`](../GL_Quick_VS/GL_Quick/src/prover.cpp)** emits the wrapping OR at the parent scope via a fresh `addExprToMemoryBlock` call. Site F dedupe applies — if the OR is already known at any ancestor of the parent scope, the emit is silent.

**Code.** [`prover.cpp–4812`](../GL_Quick_VS/GL_Quick/src/prover.cpp). The `nameMap.ancestorsOf` table is built at every `encodePush` site — see [`memory.hpp::NameMap::encodePush`](../GL_Quick_VS/GL_Quick/src/memory.hpp).

---

<a id="i-29"></a>
## I-29 Variable-port type consistency: every variable in a chapter row connects ports with identical type labels

**Scope.** Producer side: [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp) (`ArgumentAnalyzer`) called from [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (`compileCoreExpressionMap`). Consumer side: [`verifier.py:check_defset_consistency`](../verifier.py) (D-41).

**Rule.** Within any single MPL expression in a proof-graph chapter row, every variable that appears at multiple operator-call positions must connect to ports whose `definition_sets` type label is identical. Bound-variable rebinding at `>[v1,v2]` quantifiers introduces alpha-distinct names — the same string `v1` inside an inner quantifier scope is not the same variable as one outside. Variables whose name is reused across alpha-distinct scopes are independent.

**Why.** Consistency of types across an expression's variable connections is the structural prerequisite for correct semantics. A variable used at one position with type `(1)` (an element of N) and at another with type `P(1)` (a subset of N) cannot meaningfully refer to the same object. The compiler enforces this at producer time via the per-batch `ArgumentAnalyzer`'s `mergeMaps` mismatch raise; the verifier confirms the contract holds on the artefacts that get shipped (chapter rows in `processed_proof_graph/*.txt`). Together these form the formal-completeness pillar.

**Spot.**

- A new producer-side code path that emits a chapter row with mismatched argument types would surface as a `definition set consistency` failure in the verifier tally.
- A new compact operator added to a per-batch GL binary whose elements lead to inter-batch arity collisions will manifest if `gl_binaries[<batch>]` and `gl_binaries['shared']` both define the operator with different signatures.
- A change to ConfigVisu.json's atomic operator definition_sets that drifts away from per-batch configs will surface as widespread chapter-row failures (the  script catches this case).

**Fix.**

- Producer-side type mismatch in a chapter row: trace upstream from the verifier's per-row failure report. The compiler's analyzer would have asserted at compile time if the rule itself were ill-typed; surface from a verifier failure on a chapter row indicates either a typing edge case the compiler missed or a rule application at runtime that ignores the operator's defsets. Investigate the row's tag (which producer site emitted it).
- ConfigVisu drift: sync to the per-batch authoritative version; per-batch configs win on disagreement (they're the live source for the prover).
- Inter-batch compact-name collision: per-tag resolution in `build_resolved_defsets_per_tag` selects the right batch's allocation per chapter; if a new batch is added, ensure its binary is loaded.

**Code.** Producer-side: [`compileCoreExpressionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Consumer-side: [`verifier.py:check_defset_consistency`](../verifier.py) + helpers (`build_resolved_defsets_per_tag`, `_parse_subtree`, `_merge_maps`, `_process_leaf`).

---

<a id="i-28"></a>
## I-28 Cross-LB writes during `proveKernel`'s parallel phase are forbidden — defer to post-`pool.join` collectors

**Scope.** [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (`proveKernel`'s barriered phases — the phase-1 and phase-3 across-LB sweeps, and phase 2's flat `(LB, part)` executor pool + per-LB finalize sweep — plus the post-join drains), every code path reachable from `performElem2` / `performElemPhase1` / `performElemPhase2` / `performElemPhase3` on a worker thread.

**Rule.** Code that runs on a worker thread inside any of `proveKernel`'s three barriered phase sweeps (each sweep spawns `workers` threads via the local `runPhase` helper and joins them — the join is the barrier, [D-114](40_decisions.md#d-114)) **must not write to any other LB's mutable state** — `exprOriginMap`, `intEncodedStatements`, `mailIn`, `sameIterationInternalMail`, `simpleMap`, `toBeProved`, etc. Worker threads may freely write their **own** LB's `mailOut.*` (which is single-LB-scoped during the parallel phase), and may stage entries into class-level collectors on `ExpressionAnalyzer` under a mutex (`inductionMemoryBlocksMutex`, `pendingAncestorOriginsMutex`, etc.). The class-level collectors are drained single-threaded after the final sweep's join, in sorted order; that drain is the only place where ancestor-side mutable state is touched. Phase 2's flat executor pool is a strictly stronger case: its `(LB, part)` tasks are **read-only** on the LB ([D-116](40_decisions.md#d-116)) and write only their own per-task `firingRecords`, so they make no LB write at all — even same-LB parts running concurrently cannot race; the per-LB finalize sweep that follows the pool's join does the (single-LB) writes.

**Why.** `logicalCores = std::max(1u, std::thread::hardware_concurrency)` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp); the `=1` workaround is commented out). Each phase sweep spawns `workers` real threads, each pulling LBs from the shared `active` vector via `std::atomic<size_t> next` and calling the phase helper independently. Two threads can be on different LBs at the same time; one reaching across into another's `std::map` is undefined behavior — the bimodal pattern fixed by [D-39](40_decisions.md#d-39) was exactly this. The barriered split keeps the proof byte-identical precisely *because* this invariant holds: with no cross-LB intra-cycle write, the phase interleaving is unobservable (the byte-identity argument in [D-114](40_decisions.md#d-114)).

**Spot.**
- Any descendant code that does `pred = memoryBlock.parentMemory; while (pred) { someWrite(pred->...); pred = pred->parentMemory; }` outside the post-join block.
- Any code that takes a raw `Memory*` pointer to a LB other than the one currently on this thread's stack and writes to its containers.
- A new collector that buffers cross-LB effects but doesn't go under a mutex, or doesn't sort before drain — both break the invariant.

**Fix.** Stage the effect into a class-level vector under the appropriate mutex (or add one); drain it in `proveKernel` after the existing `inductionMemoryBlocks` drain (`prover.cpp`), single-threaded, sorted by deterministic key. The drain may freely touch any LB's state. See `inductionMemoryBlocks` (+ `activateZeroCondition`) and `deferredAncestorPages` (+ `drainDeferredAncestorAdmissions`, [D-187](40_decisions.md); the deferred-admission carrier is a shared `SealedPageSet` record chain since S9 — the sealed-handoff variant of the pattern, [D-164](40_decisions.md#d-164)) as canonical instances. When the cross-LB write site itself runs in BOTH single-threaded contexts (grid build, the drain's own re-entry) and parallel workers, gate the defer on a thread-local "in a parallel worker" flag (`g_inParallelWorkerPhase`) so the single-threaded paths apply inline and the drain does not re-defer.

**Code.** Existing collectors: `inductionMemoryBlocks` + `inductionMemoryBlocksMutex`, `updateGlobalTuples` + `updateGlobalMutex`, `updateGlobalDirectTuples` + `updateGlobalDirectMutex`, `deferredAncestorPages` (a shared `SealedPageSet` record chain, statified from the former `std::vector<DeferredAncestorAdmission>` in S9) + `deferredAncestorAdmissionsMutex` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)). Drain block: [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). (`pendingAncestorOrigins` + `pendingAncestorOriginsMutex`, cited here historically, is retired — [D-51](40_decisions.md#d-51).) `updateAdmissionMap3`'s strict-ancestor admission seed is a deferred instance: the resolved call's strings are SEALED onto `deferredAncestorPages` under the mutex when `g_inParallelWorkerPhase` (the sealed handoff, [D-164](40_decisions.md#d-164) — no worker-phase interner mint, [I-83](#i-83)); `proveKernel` `emplace`s a fresh Filling set before phase-1 and `drainDeferredAncestorAdmissions` replays them single-threaded via an index sort under `deferredAncestorAdmissionLess` (the retired 6-field chain, byte-for-byte; [D-187](40_decisions.md)), then `seal`s + `freePages` + resets the `std::optional`.

---

<a id="i-25"></a>
## I-25 `addStatement`'s `newStatements` out-param carries every deposit (id form) with its own scope; cross-scope deposits ride the single channel

**Scope.** Prover — [`addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`applyEquivalenceClass`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`applyEquivalenceClassToNegatedEquality`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp), and the kernel post-loop in [`addExprToMemoryBlockKernel`](../GL_Quick_VS/GL_Quick/src/prover.cpp). (`cleanUpExpressions` no longer participates — its dead `newStatements` branch was dropped in the L5 id-form flip.)

**Rule.** `addStatement` returns `void` and appends to a caller-owned out-param `PagedVector<IntEncodedExpr>& newStatements` (id form) on the per-slot `genScratchArenas`; every helper that pushes new facts into that channel (`applyEquivalenceClass`'s `products` sink, `applyEquivalenceClassToNegatedEquality`, the three direct `addStatement` pushes) appends the same id-form row type. Each row's decoded `(originalId, validityId)` carries the deposit's actual scope. The kernel's post-`addStatement` loop index-sorts the buffer with `sortStatementRows` (reproducing the former `std::sort` under `ExpressionWithValidity::operator<` byte-for-byte) and decodes each row's `validityId` for the `intStatementLevelsMap` lookup, admission-map updates, `toBeProved` discharge, and validity-name promotion — never the kernel-call's own `validityName` parameter. There must be no separate "side sink" for cross-scope deposits. (PagedVector is non-movable, which is why the channel is an out-param, not a return value.)

**Why.** Cross-scope eq-class deposits ([D-33](40_decisions.md#d-33)) land at `deeperOf(class.scope, expr.scope)`, which can differ from the kernel-call's `validityName`. A WIP design that routed them to side sinks (`crossScopeSink`, `descendantSink`, `ancestorSink`) avoided tripping the kernel's same-scope assert at [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) but bypassed the kernel's discharge logic for those facts. Matching `toBeProved` entries never closed; FTA-rung-1 §9b's `(implication23[2,8,int_lev_4_2349])` validity-name promotion never fired even with the boundary fact present in `intEncodedStatements`; Gauss summation regressed from proved to unproved.

The per-row channel carries the deposit's scope into the kernel (as the row's `validityId`) so the lookup uses the deposit's own validity, the assert holds, and the discharge logic runs uniformly for same-scope and cross-scope deposits.

**Spot.**

- A new code path that pushes into the emit channel with a bare `std::string` or a heap `ExpressionWithValidity` (compile error after the L5 flip — the channel type is `PagedVector<IntEncodedExpr>`; append an id-form row via `encodeExpression`).
- A new local `crossScopeSink`, `descendantSink`, or `ancestorSink` variable inside `addStatement` or `updateEquivalenceClasses`. Routes that bypass `newStatements` re-open the discharge gap.
- The kernel post-loop building `EncodedExpression(addExpression, validityName)` with the function-parameter `validityName`. After this invariant the loop uses `effectiveValidity` (the per-entry scope from the pair).

**Fix.** Append the id-form row into `newStatements` — `encodeExpression(StrSpan(applied), StrSpan(depositValidity), nameMap)` (the ids are already minted by the deposit itself, so this re-finds them). The consumer materializes the strings at its edge via `nameMap.decode(row.originalId)` / `decode(row.validityId)`; do not re-introduce a parallel string channel.

**Code.** Helper signatures: [`prover.hpp::applyEquivalenceClass`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::applyEquivalenceClassToNegatedEquality`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::addStatement`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Kernel loop + `sortStatementRows` consumption: [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp). The `sortStatementRows` index sort, the `IntEncodedExpr` row type, and the `ExpressionWithValidity` `operator<` it reproduces are in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp).

---

<a id="i-30"></a>
## I-30 ~~`applyEquivalenceClassToRejectedMapIntegration` is additive — original rmi entries are never erased~~ *(retired — superseded by [I-37](#i-37))*

**Status.** Retired (2026-05-13). The additive rule recorded here was the integration-side analog of the pre-D-63 algebra stance and carried the same provenance-leak bug acknowledged in the pre-revision I-37: substituted constituents inserted directly at K' lacked post-substitution `disintegration` provenance. The integration hook now follows the algebra D-63 drop+mail pattern; the never-written-directly rule is unified into the revised [I-37](#i-37), which now covers both algebra `rejectedMap` and integration `rejectedMapIntegration`.

**Anchor preserved** for external cross-references in code comments, commit messages, and memory files.

**See.** [I-37](#i-37) — unified rule for both algebra and integration sides; [D-64](40_decisions.md#d-64) — the rewrite that retired this invariant; [D-43](40_decisions.md#d-43) — the original D-43 (additive-on-no-match) is correspondingly superseded.

---

<a id="i-31"></a>
## I-31 `updateEquivalenceClasses` ancestor-pass never modifies ancestor-scope class state

**Scope.** [`prover.hpp::updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp). The cross-scope merge pass added by [D-44](40_decisions.md#d-44).

**Rule.** When merging absorbs an ancestor class `C_a @ V_a` into `mergedClass @ validityName`:
- `mb.equivalenceClassesMap[V_a]` is **never** written.
- `mb.eqClassSttmntIndexMapMap[V_a]` is **never** written (no `erase`, no overwrite).
- The ancestor class is read via a `const&` argument to `mergeTwoEquivalenceClasses`; the merge logic mutates only `mergedClass` (the destination at `validityName`).

**Why.** A class at `V_a` is only valid under the equalities that have been admitted at `V_a` and its ancestors. The new equality bridging into `mergedClass @ V` was admitted at the descendant scope `V` only — `V_a` cannot see it under the same-or-deeper visibility rule. Writing `V_a`'s class state from this descendant-scope merge would conjure equivalences that `V_a` should not yet know, breaking the invariant that each scope's class state is consistent with the equalities visible AT that scope. This mirrors the additive principle established by [D-33](40_decisions.md#d-33) (apply-side cross-scope deposits land at the deeper scope; ancestor original is never overwritten) and [I-30](#i-30) / [D-43](40_decisions.md#d-43) (rmi rewrites are additive at the new key; original ancestor-scope rmi entries kept).

**Spot.**

- Any write to `mb.equivalenceClassesMap[ancestorV]` inside `updateEquivalenceClasses`'s ancestor-pass block.
- Any `mb.eqClassSttmntIndexMapMap[ancestorV].erase(...)` or `[ancestorV][...] =...` inside the ancestor-pass block.
- A `mergeTwoEquivalenceClasses` overload or call that takes `classB` by mutable reference instead of `const&`.
- Theorem regression where a fact provable at an ancestor scope vanishes after a descendant scope ingests an equality — the ancestor class was destructively mutated.

**Fix.** Restore the read-only contract: the ancestor pass must only mutate `mergedClass` (the destination class at `validityName`). `mergeTwoEquivalenceClasses` already enforces this at the type level (`const EquivalenceClass& classB`); the call-site discipline is to pass the ancestor class as the `classB` argument and never the `classA` argument.

**Code.** [`updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp), the ancestor-pass block guarded by `mb.nameMap.strictAncestorNames(validityName)`. See also [D-44](40_decisions.md#d-44), [D-33](40_decisions.md#d-33), [I-30](#i-30).

---

<a id="i-33"></a>
## I-33 `mergeTwoEquivalenceClasses` cross-vN preconditions: ancestor-only direction; eqArgs-subset assert is same-vN only

**Scope.** [`prover.hpp::mergeTwoEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp). The function takes `validityName` (mergedClass / `classA` scope) and `classBValidityName` as its two scope parameters.

**Rule.** Three sub-rules:

1. **Ancestor-only direction.** When `classBValidityName!= validityName`, `classBValidityName` must be a strict ancestor of `validityName` per `memoryBlock.nameMap.isStrictAncestor`. Descendant-direction merge is forbidden by symmetry with [D-33](40_decisions.md#d-33)'s class-deeper exclusion (a descendant class is invisible at the ancestor's scope and carries no information for an equality admitted there).

2. **Cross-vN parent-subset early-exit.** When cross-vN AND `classB.variables ⊆ classA.variables`, return immediately. The descendant `mergedClass` already covers every variable the ancestor class would contribute; no merge action is required, no cross-pair emissions, no origin folding. The ancestor class stays at its scope per [I-31](#i-31).

3. **eqArgs-subset assert is same-vN only.** The `assert(!isSubsetOf(eqArgs, classB.variables))` precondition holds for same-vN merges only. Cross-vN allows `eqArgs ⊆ classB.variables` because the equality `(=[eqArgs[0], eqArgs[1]])` may already be established at the ancestor's scope (via mail, prior derivation, or independent admission at `V_a`). Symmetrically, the `tmp.size == 1` assert (single-bridge invariant) holds for same-vN only — cross-vN allows `tmp.size == 2`, in which case `commonArg = *tmp.begin` picks the lexicographically smallest as a deterministic bridge. Cross-pair records via that single bridge cover every (varA, varB) pair; alternatives via the second bridge would be parallel origin records that [I-32](#i-32) suppresses as already-known.

**Why.** [D-44](40_decisions.md#d-44) extended `updateEquivalenceClasses` with an ancestor-scope merge pass. The original `mergeTwoEquivalenceClasses` was written under same-vN assumptions: if any same-vN class contained both `eqArgs`, that class would be iterated first by `updateEquivalenceClasses`'s sequential overlap loop and absorbed via the subset path before `mergedClass` could grow past `eqArgs`. The eqArgs-subset and single-bridge asserts encoded that invariant. Cross-vN ancestor classes break the assumption — they may legitimately contain both `eqArgs` independently of the descendant's iteration order. The Gauss main batch on the branch hit the asserted condition during Hash burst 2, aborting via `0xC0000409`. The cross-vN early-exit (sub-rule 2) further short-circuits the case where the ancestor adds nothing new, avoiding spurious cross-pair emissions for variables already in the descendant class.

**Spot.**

- Assertion failure `!isSubsetOf(eqArgs, classB.variables)` at the function in production runs of the prover that mix descendant scopes with ancestor classes containing both args.
- Assertion failure `tmp.size == 1` immediately following — the single-bridge invariant is broken when both eqArgs are in classB.
- A cross-vN call site that passes the same `validityName` argument twice (collapsing the cross-vN distinction into apparent same-vN, hiding the bug).
- A descendant class accidentally passed as `classB` (descendant-direction merge) — caught by the ancestorship assertion.

**Fix.** Both call sites of `mergeTwoEquivalenceClasses` in `updateEquivalenceClasses` must pass `classBValidityName` correctly:
- Same-vN call (line ~5950): `validityName, validityName`.
- Ancestor-vN call (line ~5987): `validityName, ancestorV`.

The function then validates the relationship and gates the assertions and the early-exit accordingly. Do not weaken the assertions — they are correct for same-vN. Do not skip the ancestorship validation — it is the only gate against descendant-direction misuse.

**Code.** [`mergeTwoEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp). See also [D-44](40_decisions.md#d-44), [I-31](#i-31), [I-32](#i-32).

---

<a id="i-32"></a>
## I-32 Cross-pair `equality2` emission gated on existing class/LB origin

**Scope.** [`prover.hpp::mergeTwoEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp), the merged-pair history block (the `if (parameters.trackHistory)` branch with `varA!= commonArg && varB!= commonArg`).

**Rule.** Before pushing an `equality2` cross-pair origin record for target `(=[varA, varB]) @ validityName` (or its mirror `(=[varB, varA])`), check whether any of the following already carry a non-empty origin entry for that target:

- `mergedOriginMap` — origins accumulated in the in-flight merge so far (covers prior cross-pair pushes within the same `updateEquivalenceClasses` call where multiple existing classes overlap `eqArgs` and get folded sequentially).
- `classB.equalityOriginMap` — origins on the class being absorbed (these are about to fold into `mergedClass` via the post-loop overwrite at `prover.hpp::mergeTwoEquivalenceClasses`'s `tmpMap` blocks; treat them as already-known).
- `memoryBlock.exprOriginMap` — origins at LB level (covers mail-arrived origins synced into the class via the phase-1 absorb's origins bulk-merge + equality-sync inside [`prover.hpp::standardProcessing`](../GL_Quick_VS/GL_Quick/src/prover.hpp); also covers any other prior derivation that already wrote to the LB's exprOriginMap).

If any of the three holds, **skip** all three `addOrigin` calls (`mergedOriginMap`, `memoryBlock.exprOriginMap`, `memoryBlock.mailOut.exprOriginMap`) for that target.

**Why.** When a clique of equalities arrives via mail (e.g. the contradiction-cascade clique `{(=[v1,v2]), (=[v1,i1]), (=[v2,i1])}` mailed into a child LB), the bulk-merge places mail origins in `body.exprOriginMap` and the mail-sync places them in the class's `equalityOriginMap`. Subsequent `mergeTwoEquivalenceClasses` calls iterate possible bridge variables (`commonArg`) and unconditionally generate `equality2` cross-pair records for every (varA, varB) combination. Different bridge choices for the same target produce DIFFERENT cross-pair source pairs that are individually valid but mutually dependent: e.g. `(=[v1,v2]) ← equality2 | (=[v1,i1]) (=[i1,v2])` (bridge `i1`) and `(=[v1,i1]) ← equality2 | (=[v1,v2]) (=[v2,i1])` (bridge `v2`). The verifier's origin-chain DFS at [`verifier.py`](../verifier.py) detects the cycle. The historical case is the chapter-22 / theorem-12 (induction zero-case) failure documented in [G-31](50_gotchas.md#g-31) and [D-46](40_decisions.md#d-46).

The gate is sound because `equality2` cross-pair records are *transitive convenience records* — they document that the equality is derivable through the merge bridge. When the equality is already established by a separate path (mail, prior merge, anchor handling, etc.), the convenience record contributes no new deductive content but introduces a parallel origin that can form cycles with other parallel origins.

**Spot.**

- New chapter rows of the form `(=[a,b]) equality2 (=[a,c]) (=[c,b])` where `(=[a,b])` already has another origin row in the same chapter.
- Verifier failures under the `origin chain termination` tag — the DFS reports cyclic nodes within an `equality2` clique.
- Direct inspection of `memoryBlock.exprOriginMap` for an equality showing both a non-`equality2` origin (e.g. `recursion`, `merging origin`, mail-derived) AND an `equality2` origin whose sources transitively depend on the equality itself.

**Fix.** Restore the gate. Do **not** weaken the check — all three sources (`mergedOriginMap`, `classB.equalityOriginMap`, `memoryBlock.exprOriginMap`) must be consulted. Removing any one re-opens the cycle path it covers (e.g. dropping `mergedOriginMap` re-opens cycles between two same-call merges; dropping `memoryBlock.exprOriginMap` re-opens cycles against mail-arrived origins not yet visible to the class's local map).

**Code.** [`mergeTwoEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp). See also [D-46](40_decisions.md#d-46), [G-31](50_gotchas.md#g-31), [I-12](#i-12), [I-31](#i-31).

---

<a id="i-34"></a>
## I-34 Cross-substitution `equality1` emission gated on existing target origin

**Scope.** [`prover.hpp::applyEquivalenceClass`](../GL_Quick_VS/GL_Quick/src/prover.hpp), the `if (parameters.trackHistory)` block that emits the `equality1` origin record for a class-rewritten expression `applied @ depositValidity`.

**Rule.** Before pushing an `equality1` origin record for target `applied @ depositValidity`, check whether `memoryBlock.exprOriginMap[appliedWithValidity]` already has at least one entry. If it does, **skip** both `addOrigin` calls (`memoryBlock.exprOriginMap` and `memoryBlock.mailOut.exprOriginMap`) for that target.

The `exprOriginMapLocal` populate at the rewrite-enumeration site stays unconditional — this is required so the FIRST emission for a target carries its `setEqualities` justifiers. Skipping the populate would produce a malformed `len(rest) < 4` origin record that the verifier's `check_equality1` rejects (see the comment block at the populate site in `applyEquivalenceClass`). The gate fires at the *emission* step only.

**Why.** `applyEquivalenceClass` runs once per equivalence class per relevant statement. When two members of the same class both appear in admissible expressions, the rewrite loop fires both directions: `(in2[i0,i0,id]) [i0→v10] → (in2[i0,v10,id])` and the reverse `(in2[i0,v10,id]) [v10→i0] → (in2[i0,i0,id])`. Pre-gate, both directions emit `equality1` origin records that point at each other — a 2-cycle. The verifier's `origin chain termination` DFS at [`verifier.py`](../verifier.py) walks the chain and reports the cycle.

The gate is a **producer-side redundancy guard**, not the system-level cycle-resolution mechanism. It suppresses redundant equality1 emissions inside `applyEquivalenceClass` when the target already has any origin in `body.exprOriginMap`. Two important caveats:

1. The gate alone **does not close the swap-cycle case** (`(in2[i0,v10,id]) ↔ (in2[v10,i0,id])` mutually-substituted via the equivalence class with both targets being first-emissions). Both syntactically distinct targets pass the gate independently because each target's origin map is empty when its emission fires. The system-level resolution is **[I-35](#i-35)** / [D-49](40_decisions.md#d-49) — `addOrigin`'s cap-full preference replacement at intake.
2. The gate's scope is the **producer LB only** (the LB that runs `applyEquivalenceClass`). Cycles formed by mail import (multiple LBs each emitting one half of a cycle, then aggregated via `smashMail`) are unaffected by this gate. They are resolved at the receiver via the bulk-merge routing through `addOrigin` (see [I-35](#i-35)).

Both observations were established empirically by the chapter-96/97 (theorem 96 Gauss `fold` induction zero-case + step) failures: the original cycle was `(in2[i0,v10,id]) ↔ (in2[i0,i0,id])`; the swap-cycle was `(in2[i0,v10,id]) ↔ (in2[v10,i0,id])`. Both shapes were ultimately resolved by I-35/D-49, not by this gate. See [G-39](50_gotchas.md#g-39) and [D-48](40_decisions.md#d-48) for the gate; [G-40](50_gotchas.md#g-40) and [D-49](40_decisions.md#d-49) for the resolution.

**Spot.**

- New chapter rows of the form `(P[…X…]) equality1 (P[…Y…]) (=[X,Y])` where the same chapter also has a row `(P[…Y…]) equality1 (P[…X…]) (=[Y,X])` — both rows reference each other and neither has a non-`equality1` origin.
- Verifier failures under the `origin chain termination` tag whose cyclic node set consists of two expressions related by an equivalence-class substitution.
- Direct inspection of `memoryBlock.exprOriginMap` showing two expressions whose only origins are `equality1` records pointing at each other.

**Fix.** Keep the gate — `auto alreadyHasOrigin = [&](const ExpressionWithValidity& ev) -> bool { auto it = memoryBlock.exprOriginMap.find(ev); return it!= memoryBlock.exprOriginMap.end && !it->second.empty; };` — applied before the `addOrigin` calls inside the `trackHistory` block. The gate is producer-side noise reduction; the actual cycle-resolution mechanism is [I-35](#i-35). Sites 2 (`applyEquivalenceClassToNegatedEquality`) and 3 (`emitIntegrationRevivalToInternalMailIn`) can also emit `equality1`; they are not gated because their internal early-exits (`intStatementLevelsMap` dedup at site 2; mail absorbance routing through I-35 at site 3) cover the cycle shape via the receiver's preference replacement.

**Code.** [`applyEquivalenceClass`](../GL_Quick_VS/GL_Quick/src/prover.hpp). See also [D-48](40_decisions.md#d-48), [I-35](#i-35), [D-49](40_decisions.md#d-49), [G-40](50_gotchas.md#g-40), [I-32](#i-32).

---

<a id="i-35"></a>
## I-35 `addOrigin` cap-full preference: anything beats `equality1`/`equality2` — superseded by [D-51](40_decisions.md#d-51), 2026-05-08

> **Status amendment (2026-05-08).** Superseded by [D-51](40_decisions.md#d-51) for the chapter-emission cycle case. The cap-full preference logic remains in `addOrigin` at HEAD but is rarely active under the post-D-51 configuration: `max_origin_per_expr = 30` (non-compressor configs match compressor mode) means the per-key origin vector is rarely at cap, so the replacement branch runs only on a handful of high-traffic keys. The structural cycle-prevention work has moved into `buildStack` itself (chapter goal in `g_buildStackPath` so the existing path-cycle filter rejects self-applying origins; chapter-boundary `__contradiction__` simpleMap fallback as the only LB switch). The text below describes the original rule for traceability of the chapter-100/101 fix; the rule is no longer load-bearing for any current chapter shape. Cleanup of the now-mostly-inert preference branch is deferred.

**Scope.** [`prover.hpp::addOrigin`](../GL_Quick_VS/GL_Quick/src/prover.hpp), the universal origin-emission helper used by every producer site (`applyEquivalenceClass`, `mergeTwoEquivalenceClasses`, `addStatement` mirror block, `addExprToMemoryBlock` history, the phase-1 absorb's mailIn bulk-merge in `standardProcessing`, etc.).

**Rule.** When the per-key origin vector is at cap (`maxOrigins`) and a new origin arrives:

- If the new origin's tag is **not** `equality1` or `equality2`, scan the vector for the first slot whose tag **is** `equality1` or `equality2` and replace it in place with the new origin. Return.
- Otherwise (new origin is equality-convenience-tagged, or no slot is replaceable), keep existing — drop the new origin per the legacy "insertion order" tiebreak.

Below cap, the legacy append behavior is unchanged (push at end, dedup on exact-match, D-44 trap intact).

**Why.** `equality1` and `equality2` are *transitive convenience records*: they document derivability via equivalence-class substitution (`equality1` for argument substitution, `equality2` for cross-pair transitivity). Both are inherently susceptible to swap/bridge cycles when the class has multiple members — see [I-32](#i-32) for the equality2 cross-pair gate, [I-34](#i-34) for the equality1 substitution gate. Any other origin tag (`implication`, `recursion`, `theorem`, `expansion`, `disintegration`, `task formulation`, `premise element`, `reformulation for integration...`, `vacuous truth`, `symmetry of equality/inequality`,...) refers to a direct deductive step that does not have this cyclic structure.

When two origins exist for the same key (legitimate situation — e.g. mail-bulk-merge brings a foundational `implication` origin AND a cyclic `equality1` origin both shipped in the same broadcast for the same target) and `max_origin_per_expr = 1` (non-compressor) forces the choice down to one, **the choice must not be left to insertion order** — that silently picks whichever origin happened to be pushed first, which has been the cyclic `equality1` in the chapter-100/101 swap-cycle case (theorem-96 Gauss `fold` induction). Foundation must displace convenience. The verifier's `origin chain termination` walks the surviving origin; a foundational origin terminates, a cyclic one loops.

**Spot.**

- `body.exprOriginMap[key]` showing only an `equality1`/`equality2` origin for an expression that *also* has a foundational origin earlier in the same hashburst's `mailIn.exprOriginMap` snapshot. Compare burst N's mailIn dump (multi-origin) against burst N+1's `body.exprOriginMap` dump (single-origin); a downgrade to equality-only is the symptom.
- Verifier failures under the `origin chain termination` tag where the cyclic node set is exactly two `equality1`-tagged or `equality2`-tagged rows and the prover-side trace shows a competing non-equality origin in `mailIn.exprOriginMap`.

**Fix.** Restore the cap-full preference replacement in `addOrigin`. Do **not** weaken to "first non-equality wins" (existing equality slot must be replaced by an arriving non-equality, regardless of arrival order). Do **not** extend preference to all tags — the rule is binary (equality-convenience vs. anything else) because that is the categorical distinction; a finer-grained per-tag priority list would couple the helper to producer-side semantics and is unnecessary for cycle suppression.

**Code.** [`addOrigin`](../GL_Quick_VS/GL_Quick/src/prover.hpp). See also [D-49](40_decisions.md#d-49), [I-32](#i-32), [I-34](#i-34).

---

<a id="i-39"></a>
## I-39 Chapter rows are emitted at the closest-to-`main` ancestor with an origin


**Scope.** [`visualizer.cpp::buildStack`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp) — the chapter-row emission walker. All consumers of `files/raw_proof_graph/*.txt`: [`verifier.py`](../verifier.py), [`process_proof_graphs.py`](../process_proof_graphs.py), [`generate_full_proof_graph.py`](../generate_full_proof_graph.py).

**Rule.** Every chapter row's `row[1]` (new-expression validity) and every dep cell's validity at `row[3 + 2k + 1]` is the **closest-to-`"main"` ancestor of the requested validity for which `(expression, ancestor)` has an origin in the emitting LB's `exprOriginMap`**, subject to the **OR-branch barrier**: the walk may not cross `_boundary_orint_` or `_boundary_ordis_` delimiters. The deepest `orint_`/`ordis_` ancestor sets the shallowest allowed lift target. Ancestor chains are recovered by splitting the validity string on `"_boundary_"` per [I-2](#i-2); the helper `liftToShallowestOriginAncestor` in [`visualizer.cpp`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp) is the single authority. Producer-side: applied at `buildStack` entry, at the dep-emission lambda, at every recursive `buildStack` call, at the candidate-loop cycle filter, and at the last-resort `front`-emit fallback. Consumer-side: a chapter row's `row[1]` and each dep cell's validity already agree at the same lifted scope, so verifier checkers can compare dep validities with exact `==` against the corresponding stored validity.

**Why the OR barrier.** OR-branch scopes (`_boundary_orint_<sig>_(<disjunct>)`, `_boundary_ordis_<sig>_(<disjunct>)`) are conditional on a disjunct hypothesis. The OR-family verifier checkers (`check_or_convergence`, `check_or_branch_proven`, `check_or_branch_assumption`, `check_or_disintegration`) require branch-distinct namespaces on the dep cells to recognise the convergence / branch-proof pattern. Without the barrier, both branches' deps would lift to the parent boundary (where the post-convergence origin lives), collapsing the convergence row to identical dep pairs and tripping `origin chain termination` cycles in the verifier.

**Why.** Pre-lifting, `buildStack` did an exact `(expr, validity)` lookup with a "fall through to `(expr, "main")`" shadow when the exact-key missed, and `emitRow` wrote the *requested* `proved.validityName` into `row[1]` regardless of which scope the origin was actually fetched from. Result: chapter rows could claim a derivation happened at a deep boundary scope while the origin's deps were at `main` — a falsified row. Concrete instance: `files/raw_proof_graph/193_check_induction_condition.txt` line 102 emitted `(=[it_0_lev_0_32,2])` at `main_boundary_(implication23[2,8,int_lev_4_2365])` with deps cited at `main`, despite the only origin record being at `main`. Lifting eliminates the falsification by emitting at the scope where the origin actually lives. The chapter shape becomes truthful, deduplicates by construction (one row per `(expr, lifted_v)`), and makes HTML namespace-tag jumps deterministic (every cited validity has a card because that is exactly where buildStack emitted).

**Spot.**

- A chapter row whose `row[1]` is a non-`main` scope but every dep cell is on `main`. Pre-lifting, this was the falsification signal; post-lifting it should not appear unless the origin really is recorded at that scope.
- Verifier failures under `expansion`, `disintegration`, `symmetry of equality`, `symmetry of inequality`, or other dep-validity-matching checkers where the stored validity exists at an ancestor of the row's cited dep validity. Pre-lifting, the fix was to widen verifier matching; post-lifting, the fix is to confirm `buildStack` is invoking `liftToShallowestOriginAncestor` at every dep emission and recursion site.
- Inspect  if it appears — the "no origin found" assertion's dump now reflects a *lifted* `proved`, so a missing origin really means no ancestor (including `main`) has an entry.

**Fix.** Restore the lifting at any site in `buildStack` where it was removed. The helper `liftToShallowestOriginAncestor(memoryBlock, ExpressionWithValidity)` is the single point of authority. Do not bypass it for "performance"; the cost is one hash-map lookup per ancestor level, capped by validity-stack depth.

**Code.** [`visualizer.cpp::liftToShallowestOriginAncestor`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp), [`visualizer.cpp::buildStack`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp). See also [I-2](#i-2), [I-38](#i-38), [D-56](40_decisions.md#d-56), [G-41](50_gotchas.md#g-41).

**Relation to [I-38](#i-38).** Complementary fixes for the same falsified-row class. This invariant is producer-side (lifting prevents emission of the falsified row in the first place). The deeperOf-equality invariant is verifier-side (rejects any falsified row that slips through). With both in place, the chapter-193 line-102 example is fixed at the producer (lift moves the row to `main`); the verifier check stays as forcing-function for any future producer-side regression.

---

<a id="i-38"></a>
## I-38 `implication`-row deposit lives at `deeperOf` of constituents

**Scope.** Verifier — `check_implication` at [`verifier.py`](../verifier.py). Mirrors C++ prover behaviour: [`generateEncodedRequestsStatic`](../GL_Quick_VS/GL_Quick/src/memory.cpp) accumulates validity ids via `nm.deeperOf(...)`, so the result of combining facts lives at the deepest scope of the inputs.

**Rule.** For every chapter row tagged `implication`:

1. **Pair-wise comparability.** Every pair of namespaces among the constituents (the rule itself + every cited premise) must be **comparable** in the validity-stack sense (`nm.comparable(a, b)` per [`04_validity_stack.md` § `comparable / deeperOf`](20_core_concepts/04_validity_stack.md)). One must be an ancestor of the other (or they are equal). Sibling scopes are rejected.

2. **Deeperof-equality on result.** The row's namespace (`line.namespace`) must EQUAL the deepest constituent namespace. Equivalently, `line.namespace ∈ {impl_ns} ∪ {premise_nss}`. If every constituent sits at `"main"` while `line.namespace` is `"main_boundary_<X>"`, the row is rejected: an implication firing deposits its conclusion at `deeperOf(constituents)`, never at a strictly deeper scope no constituent reaches.

**Relation to [D-35](40_decisions.md#d-35).** D-35 (comparable-scope premise inheritance) is the weaker baseline: every constituent ns must be at-or-above `result_ns`. The deeperOf-equality rule is the strengthening: at least one constituent must REACH `result_ns`. D-35 alone admitted rows where every constituent was an ancestor of `result_ns` but none reached it; those rows imply a derivation step the prover cannot have performed.

**Relation to [I-39](#i-39).** Producer-side complement. After `buildStack` lifting, no chapter row should have `line.namespace` strictly deeper than every dep — lifting moves the result up to the closest-to-`main` ancestor with origin, so at least one dep is at the result's scope. This invariant therefore acts as a backstop: if a future producer-side regression bypasses lifting, the verifier still catches the falsified row.

**Why.** Sound under GL's validity-stack semantics ([`20_core_concepts/04_validity_stack.md`](20_core_concepts/04_validity_stack.md) — `deeperOf(a, b)` returns the deeper of two comparable ids and is the canonical "fact's effective scope when joined"). A verified proof graph that admits result_ns strictly deeper than every input means a producer-side bug forged a scope-promotion the prover's hash kernel cannot emit. Catching such rows at the verifier (rather than chasing them through downstream regressions) follows the project's "failures are first-class" stance.

**Spot.**

- Verifier failure under `implication` on a chapter row whose `line.namespace` is strictly deeper than every namespace in `rest[1::2]`.
- Concrete example (rung-1 incubator branch, pre-fix): `(=[it_0_lev_0_32,2]) main_boundary_(implication23[2,8,int_lev_4_2365]) implication (>[1](in[1,u_1])(>[2](in2[2,1,u_3])(>[3](in2[3,1,u_3])(=[2,3])))) main (in2[2,6,3]) main (in2[it_0_lev_0_32,6,3]) main (in[6,1]) main`. Every constituent at `main`; result at `main_boundary_(implication23[2,8,int_lev_4_2365])`. D-35 alone passes; the deeperOf-equality rule rejects. Post-lifting, the row no longer exists (lifted to `main`).

**Fix.** Producer side — fix the prover-side path that emitted the misnamespaced row (typically a `validityName` set/copy that bypassed `nm.deeperOf` accumulation). The verifier check is forcing-function for the producer-side audit; do NOT weaken the check. For the chapter-193 line-102 instance, the producer-side fix is [I-39](#i-39): `buildStack` now lifts the row's namespace to the scope where its origin actually lives.

**Code.** `check_implication` namespace block at [`verifier.py`](../verifier.py). Tests covering the failure mode: `tests/test_verifier_implication.py` (`test_implication_result_deeper_than_every_constituent`, `test_implication_result_deeper_than_every_constituent_simple`, `test_implication_premise_ns_sibling_via_pair_comparable`). C++ source of truth: [`generateEncodedRequestsStatic`](../GL_Quick_VS/GL_Quick/src/memory.cpp).

**Decision ref.** [D-58](40_decisions.md#d-58).

---

<a id="i-37"></a>
## I-37 `rejectedMap` and `rejectedMapIntegration` are never written **directly** by equi-class application

**Scope.** Prover — equivalence-class hook surface on both algebra and integration sides. Specifically `prover.hpp::applyEquivalenceClassToAdmissionMap`, `prover.hpp::applyEquivalenceClassToRejectedMap`, `prover.hpp::applyEquivalenceClassToAdmissionMapIntegration`, `prover.hpp::applyEquivalenceClassToRejectedMapIntegration`, and any future hook on either side.

**Rule.** Equi-class application may **erase** entries from `HashMemory::rejectedMap` or `HashMemory::rejectedMapIntegration` (when the entry contains a non-canonical class member) but **never inserts** into either map directly. Each map is mutated only by:

1. The original disintegration path inserting fresh rejection records via `prover.hpp::updateRejectedMap` (algebra) or `prover.hpp::updateRejectedMapIntegration` (integration).
2. The revival entry points snapshotting + erasing entries during revival: `prover.cpp::revisitRejected2` (algebra) and `prover.cpp::revisitRejectedIntegration2` (integration).
3. The equi-class cleanup hooks **erasing** entries: `prover.hpp::applyEquivalenceClassToRejectedMap` ([D-63](40_decisions.md#d-63), algebra) and `prover.hpp::applyEquivalenceClassToRejectedMapIntegration` ([D-64](40_decisions.md#d-64), integration).

Any code that takes an existing `rejectedMap` / `rejectedMapIntegration` entry, substitutes its variables via an equivalence class, and **directly re-inserts** the substituted form is a violation.

**Why.** Both `rejectedMap` (algebra) and `rejectedMapIntegration` (integration) hold real disintegration products whose `disintegration` origins were written at production site — by `prover.cpp::disintegrateExprCore2::trackExpansionHistory` on the algebra side, and by the integration-side equivalent expansion path on the integration side. Those origins are bound to the original (un-substituted) expressions and live in `Memory::exprOriginMap` from the moment the disintegration walk completes. Inserting a substituted constituent **directly** into either rejection map would create a deferred-match record for an expression that has no corresponding `disintegration` origin in `exprOriginMap` — a provenance gap. Subsequent revival of that record would either (a) emit a chapter row with a soft `equality1` self-source origin (no foundation origin present to displace it via `addOrigin`'s cap-full preference, so the verifier sees `len(rest) < 4` and fails), or (b) tag the constituent under a borrowed `disintegration` origin pointing to the pre-substitution expanded form — a row that does not correspond to a real prover step.

**The cleanup hook's mail-and-disintegrate workaround.** Both algebra D-63 and integration D-64 close the provenance gap without violating the direct-write rule: when the hook drops a non-canonical entry, it also mails the substituted **compound** onto `sameIterationInternalMail.statements` with an `equality1` history line in `sameIterationInternalMail.exprOriginMap` pointing back to the original compound. The kernel's natural disintegration of the absorbed compound at the next hashburst re-runs the appropriate `disintegrateExprCore2::trackExpansionHistory` (algebra) or integration-side expansion path, which emits proper `disintegration` origins for the canonical-form constituents; unadmitted constituents are routed to `rejectedMap[K']` / `rejectedMapIntegration[K']` via the standard `updateRejectedMap` / `updateRejectedMapIntegration` writer (channel 1 above) — with full provenance. So either rejection map may end up populated as an **indirect** consequence of the equi-class hook, but the hook itself never writes the map.

**History — pre-revision asymmetry.** Before (2026-05-13), the integration side performed a direct substituted insertion via the D-43 additive-on-no-match behaviour, acknowledged as a provenance-leak bug ("we do not correct it for integration but for algebra it must be like..."). That asymmetry has been retired; the integration hook now follows the same drop+mail playbook, and [I-30](#i-30) (the prior rmi-additive rule) is retired in favor of this unified statement.

**Spot.**

- A new write to `HashMemory::rejectedMap` or `HashMemory::rejectedMapIntegration` (`insert`, `[].insert`, `emplace`, or container modification) inside a function whose name or doc references equivalence-class application. The six legitimate writers are listed under the *Rule* heading above (three per side).
- A chapter row tagged `equality1` or `disintegration` whose source expression cannot be located in any earlier chapter row at any ancestor scope — likely symptom of a substituted entry whose pre-substitution form was never independently registered.

**Fix.** Route broadening through the admission-map hooks (`applyEquivalenceClassToAdmissionMap` on algebra, `applyEquivalenceClassToAdmissionMapIntegration` on integration) for admission-side metadata changes, and through the rejection-map cleanup hooks for class-driven rejection cleanup. The latter mail the rewritten compound; do not bypass that path by directly inserting into `rejectedMap` / `rejectedMapIntegration`.

**Code.** [`prover.hpp::applyEquivalenceClassToAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::applyEquivalenceClassToRejectedMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::applyEquivalenceClassToAdmissionMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::applyEquivalenceClassToRejectedMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.cpp::revisitRejected2`](../GL_Quick_VS/GL_Quick/src/prover.cpp), [`prover.cpp::revisitRejectedIntegration2`](../GL_Quick_VS/GL_Quick/src/prover.cpp), [`prover.cpp::disintegrateExprCore2`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (the production-site origin emitter). See also [D-57](40_decisions.md#d-57), [D-63](40_decisions.md#d-63), [D-64](40_decisions.md#d-64), [I-22](#i-22), [I-36](#i-36).

---

<a id="i-36"></a>
## I-36 Algebra equi-class rewrites preserve positional collision pattern

> **Superseded on the algebra admission path by [D-106](40_decisions.md#d-106) (2026-06-02).** `applyEquivalenceClassToAdmissionMap` no longer enumerates rewrites or runs the arg-equalization filter — it collapses the class to one canonical representative (`chooseCanonical`) and rewrites the key once, allowing slot collapses (repetitions). The positional-collision preservation below is no longer load-bearing because the companion rejected-key hook collapses identically (both sides land on the same representative). The admission-*integration* mirror filter is retired the same way by that decision (the integration hook now collapses to one representative too). The rest of this entry describes the retired enumerate-path behaviour.

**Scope.** Prover — `prover.hpp::applyEquivalenceClassToAdmissionMap`. Algebra equi-class hook on the admission map.

**Rule.** For every candidate rewrite of an `admissionMap` key K → K' produced by `enumerateEqClassRewrites`, the rewrite is admissible only if it preserves the positional collision pattern. Concretely:

> For every pair `(i, j)` with `i < j` in K's arg list: if `origArgs[i]!= origArgs[j]` then `newArgs[i]!= newArgs[j]` must hold.

If a class substitution would collapse two previously-distinct arg slots into the same value, the candidate rewrite is dropped — no admission insert, no `revisitRejected2` call. The filter lives inside the sink lambda passed to `enumerateEqClassRewrites` in `applyEquivalenceClassToAdmissionMap`.

**Why.** An `admissionMap` key K is a marker-form expression encoding the positional structure of the admission rule. Two slots holding the same arg encode "these two positions take the same value"; two slots holding different args encode "these positions can take independent values". Substituting via an equi-class so that distinct slots collapse to the same value produces a K' whose positional structure differs from K's. `revisitRejected2(K', mb, validityName)` would then probe `rejectedMap[K']` — which only contains rejections rejected under K''s collapsed shape, NOT rejections rejected under K's original distinct-arg shape. Matching mixes semantically distinct rejections; reviving them under K's rule semantics is unsound.

The integration side does NOT have this filter (it allows arg-equalizing rewrites; see `prover.hpp::applyEquivalenceClassToRejectedMapIntegration`). User clarification: "regarding application of equi classes. i see for integration it admits equal variables where original had none. it is nonsense. we do not correct it for integration but for algebra it must be like: normalized signature after renaming == before renaming. no arg equalization here." For algebra the filter is mandatory.

**Spot.**

- A chapter row generated post-revival under a rule whose marker key has identical args at slots that were distinct in the producer admission key.
- A spurious admission-fire at a fact whose argument positions don't match the original admission rule's distinct-slot signature.
- Removal of the arg-equalization filter loop in `applyEquivalenceClassToAdmissionMap` (the nested `for (i) for (j > i)` collision check before each `uniqueRewrites[...] =` insert in the sink lambda).

**Fix.** Restore the filter. The check is O(|args|²) per candidate rewrite — negligible at GL's admission-key arity (typically ≤ 5 args). Do not bypass for "performance"; correctness depends on the filter.

**Code.** [`prover.hpp::applyEquivalenceClassToAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (the sink-lambda filter). See also [D-57](40_decisions.md#d-57), [I-37](#i-37).

---

<a id="i-40"></a>
## I-40 Post-class-update admissionMap canonical sweep

> **Retired on the algebra admission path by [D-106](40_decisions.md#d-106) (2026-06-02).** `applyEquivalenceClassToAdmissionMap` now drops the changed key inline (re-key), so `cleanUpAdmissionMap` is no longer called from the Step-4b deferred-cleanup block; the inline drop subsumes every key this sweep removed. The `cleanUpAdmissionMap` definition itself was DELETED as dead code on 2026-07-02 (user-authorized; dead by subsumption, together with its integration sibling `cleanUpAdmissionMapIntegration` — [I-43](#i-43)). The rest of this entry describes the retired sweep; its Code citations are historical.

**Scope.** [`prover.hpp::cleanUpAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp), historically called from the Step-4b deferred-cleanup block in `prover.hpp::standardProcessing`.

**Rule.** After `updateEquivalenceClasses` merges classes and `applyEquivalenceClassToAdmissionMap` (D-57) has additively inserted canonical-form K' entries, the sweep walks `admissionMap` entries at `validityName` and drops every K for which `filterIterations(K.original, eqClass)` returns `false` against any class at `validityName`. The mirror erase on `admissionStatusMap` is paired with each `admissionMap` erase. The sweep does NOT touch `admissionMapIntegration` (integration domain) and does NOT insert into `consumedAdmissionKeys` (entries are replaced by canonical K', not consumed by a head). Gated on `!parameters.skip_eq_classes`.

**Why.** Mirrors the expression-side two-phase shape `applyEquivalenceClass` → `cleanUpExpressions` (`prover.hpp::cleanUpExpressions`) that `updateEquivalenceClasses` already uses for `intEncodedStatements`/`intLocalEncodedStatements*`/`intStatementLevelsMap`/`intKnownStatements`. The expression-side sweep drops `(P[non-canonical])` whose canonical-form rewrite `(P[canonical])` `applyEquivalenceClass` just inserted; the admissionMap sweep does the symmetric drop for non-canonical admission keys whose canonical form `applyEquivalenceClassToAdmissionMap` (D-57) just inserted. Without the sweep, K and K' coexist forever, costing memory and duplicate `isAdmitted` probe work on facts that admit the same shape under the class.

**Spot.**

- A `filterIterations` (`prover.hpp::filterIterations`) call signature changed to take additional arguments without updating the admissionMap sweep call site.
- `admissionMap` grows monotonically per equality and never shrinks during a single LB's prover loop, even when `cleanUpExpressions` is dropping expressions at the same call site.
- Removed `this->cleanUpAdmissionMap(mb, validityName);` line from `updateEquivalenceClasses` (immediately after the `cleanUpExpressions` call).
- A new admission-key insertion path that does not update `varsInAdmissionMapKeys` — the sweep's short-circuit then fails to detect overlap and skips legitimate cleanup work.

**Fix.** Restore the call site and the helper. The helper's short-circuit guard, scope filter, and `filterIterations` predicate are the load-bearing pieces — do not weaken any of them.

**Code.** [`prover.hpp::cleanUpAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (call site between `cleanUpExpressions` and `updateWeakVariables`), [`prover.hpp::filterIterations`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (canonical-selection rule), [`prover.cpp::updateWeakVariables`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (parallel canonical-selection bookkeeping). See also [D-62](40_decisions.md#d-62), [D-57](40_decisions.md#d-57), [I-41](#i-41).

---

<a id="i-41"></a>
## I-41 cleanAdmissionMap canonicalization closure on consumed K

**Scope.** [`prover.hpp::cleanAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp), the `markerIsOutput` branch.

**Rule.** When a disintegration head consumes admission key K at `validity` (`isAdmitted` triggers `cleanAdmissionMap(markedExpr, validity, mb)` with the operator's output slot in K), the cleanup erases not only K but also every admissionMap entry K' at the same `validity` whose canonical form under classes at `validity` equals `canon(K)`. Canonicalization is `canonicalizeUnderClasses` (`prover.hpp::canonicalizeUnderClasses`) — substitute every class member with the class's canonical (lex-smallest `int_lev_*`, fallback lex-smallest `it_*_lev_*`), per the same rule `filterIterations` and `updateWeakVariables` use. Each erased K' is removed from `admissionMap`, `admissionStatusMap`, `admissionMapIntegration`, and added to `consumedAdmissionKeys` — symmetric to the existing per-K erase. Gated on `!parameters.skip_eq_classes`. The function is FED as spans (`StrSpan markedExpr` / `validity` — the head gates run zero-copy via `extractExpressionSpan` / a transparent `operators` probe / `getArgsSpans`, and the one key mint rides the `mintTemplateKey` span door); the closure block itself stays string machinery, fed by a single materialization of each span at its entry — same bytes, same verdicts.

**Why.** K and K' (equi-class-equivalent under the current classes at `validity`) admit the same fact — the additive hook (`applyEquivalenceClassToAdmissionMap`, D-57) inserted K' precisely because the class makes them interchangeable. When K is consumed, K' has been consumed in the same sense. Leaving K' stale would re-fire the same operator output via a different surface form — duplicate work at best, divergent admission state at worst. Option 3 (canonicalization) covers transitive class-composition reachability (`K → K' via C₁ → K'' via C₂` is captured by `canon(K) = canon(K'') = canon(K') = canonical-of-all-classes`) — 1-hop would leave 2-hop entries stranded.

**Spot.**

- A new admission-key insertion path that bypasses `varsInAdmissionMapKeys`-cache population, causing the closure's short-circuit to miss legitimate K' entries.
- A new `cleanAdmissionMap` call site that does not pass the consumed K's `validity` correctly — the closure walks the wrong scope and either over-erases (different scope's keys) or under-erases (missing same-scope K').
- A new equi-class hook (algebra) that updates classes mid-LB without `cleanUpAdmissionMap` running afterwards — the closure may then erase entries the producer thought it had registered, but the producer's expectation is satisfied (those entries' canonical form is consumed).
- Restored 1-hop logic (`enumerateEqClassRewrites` per class) instead of canonicalization — silently loses 2-hop K' entries.

**Fix.** Preserve canonicalization, not 1-hop. The closure walks all admissionMap entries at `validity`; the cost is bounded by `cleanAdmissionMap`'s already-constrained firing condition (`markerIsOutput` only). The short-circuit on `varsInAdmissionMapKeys` keeps cost zero when no class overlaps any admission key.

**Code.** [`prover.hpp::cleanAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (the `markerIsOutput` branch closure block), [`prover.hpp::canonicalizeUnderClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (helper), [`prover.hpp::filterIterations`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (same canonical-selection rule). See also [D-62](40_decisions.md#d-62), [D-57](40_decisions.md#d-57), [I-40](#i-40).

---

<a id="i-42"></a>
## I-42 `applyEquivalenceClassToAdmissionMapIntegration` is additive — original K stays

> **Retired on this branch by [D-106](40_decisions.md#d-106) (2026-06-02).** `applyEquivalenceClassToAdmissionMapIntegration` now **drops the changed key and inserts the canonical K'** (re-key), exactly like the algebra admission hook — no longer additive. The integration map has no `admissionStatusMap`, so the re-key moves nothing. [I-22](#i-22) is preserved (K' persists; `revisitRejectedIntegration2` does not clean it). The rest of this entry describes the retired additive contract.

**Scope.** [`prover.hpp::applyEquivalenceClassToAdmissionMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp). The integration-side equivalence-class hook on `HashMemory::admissionMapIntegration`.

**Rule.** When a class C rewrites an admission-integration entry at u_-form key K to a new u_-form key K':
- The original K entry stays in `admissionMapIntegration` (key and inner `map<Instruction, set<string>>` value untouched).
- K' is **added** to `admissionMapIntegration` with the substituted `Instruction` value and the substituted applied-vars set.
- Multiple K' rewrites at multiple class instances accumulate without erasing earlier entries.

Mirror of [D-57](40_decisions.md#d-57)'s principle 3 for the integration side. The function never calls `admissionMapIntegration.erase` on its own.

**Why.** Admission templates are reusable across multiple `int_` witnesses (see [I-22](#i-22)). The original K may still match incoming compound shapes that the rewritten K' does not (and vice versa). Erasing K under class application would silently lose those future admissions. The post-class-update sweep `cleanUpAdmissionMapIntegration` ([I-43](#i-43)) handles the bulk drop of non-canonical entries as a separate canonicalization step, mirroring the expression-side `applyEquivalenceClass` → `cleanUpExpressions` two-phase shape.

**Spot.**

- Any `admissionMapIntegration.erase` call inside `applyEquivalenceClassToAdmissionMapIntegration` or any new helper it calls.
- A revisitRejectedIntegration2 call followed by an inline `cleanAdmissionMap`-like erase of K' — see [I-22](#i-22).

**Fix.** Restore the additive contract: the rewrite loop only queues inserts; the post-loop apply block contains only insertions, cache populates, and `revisitRejectedIntegration2` calls.

**Code.** [`applyEquivalenceClassToAdmissionMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp). See also [D-65](40_decisions.md#d-65), [D-57](40_decisions.md#d-57), [I-22](#i-22).

---

<a id="i-43"></a>
## I-43 Post-class-update admissionMapIntegration canonical sweep

> **Retired on this branch by [D-106](40_decisions.md#d-106) (2026-06-02).** The hook now drops the changed key inline (re-key), so `cleanUpAdmissionMapIntegration` is no longer called from the Step-4b deferred-cleanup block; the inline drop subsumes it. The definition itself was DELETED as dead code on 2026-07-02 (user-authorized; dead by subsumption, together with the algebra `cleanUpAdmissionMap` — [I-40](#i-40)). The rest of this entry describes the retired sweep; its Code citations are historical.

**Scope.** [`prover.hpp::cleanUpAdmissionMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp), historically called from the Step-4b deferred-cleanup block in `prover.hpp::standardProcessing`.

**Rule.** After `updateEquivalenceClasses` merges classes and `applyEquivalenceClassToAdmissionMapIntegration` ([D-65](40_decisions.md#d-65)) has additively inserted canonical-form K' entries, the sweep walks `admissionMapIntegration` entries at `validityName` and drops every K for which `filterIterations(removeUPrefixFromArguments(K.original), eqClass)` returns `false` against any class at `validityName`. There is NO mirror erase on `admissionStatusMap` (the integration map has no parallel status map) and NO insertion into `consumedAdmissionKeys` (entries are replaced by canonical K', not consumed by a head). Gated on `!parameters.skip_eq_classes`.

The u_-strip via `removeUPrefixFromArguments` before `filterIterations` is required because `filterIterations` matches `int_lev_*` / `it_*_lev_*` regex tokens, which only match against bare-form names; admissionMapIntegration keys are u_-form. The cache `varsInAdmissionMapIntegrationKeys` stores bare-form non-marker args, so the short-circuit overlap check uses bare-name comparison naturally.

**Asymmetry vs algebra.** This sweep is the canonicalization-driven drop only. There is **no integration-side analog of [I-41](#i-41)** (the on-hit canonicalization closure on `cleanAdmissionMap`'s `markerIsOutput` branch). Per [I-22](#i-22), integration admission templates persist across consumption; the closure would violate that. The existing `admissionMapIntegration.erase(...)` calls inside `cleanAdmissionMap`'s `markerIsOutput` branch are key-form no-ops (algebra bare-marker key vs integration u_-form key never match) and stay as harmless vestiges — they are NOT an exception to this rule.

**Why.** Mirrors the expression-side / algebra-admission-side two-phase shape `applyEquivalenceClass*` → `cleanUp*` that `updateEquivalenceClasses` already uses for `intEncodedStatements` ([I-40](#i-40) ports it to `admissionMap`; this invariant ports it to `admissionMapIntegration`). Without the sweep, K and K' coexist forever, costing memory and duplicate `isAdmittedIntegration` probe work on facts that admit the same shape under the class.

**Spot.**

- A `filterIterations` call inside `cleanUpAdmissionMapIntegration` passed the raw u_-form key instead of the u_-stripped form. `filterIterations`' regex tokens never match u_-prefixed forms; the function silently returns `true` (no class member found) and the sweep does nothing.
- `admissionMapIntegration` grows monotonically per equality and never shrinks during a single LB's prover loop, even when `cleanUpExpressions` / `cleanUpAdmissionMap` are dropping entries at the same call site.
- Removed `this->cleanUpAdmissionMapIntegration(mb, validityName);` line from `updateEquivalenceClasses` (immediately after the `cleanUpAdmissionMap` call).
- A new admission-integration-key insertion path that does not update `varsInAdmissionMapIntegrationKeys` — the sweep's short-circuit then fails to detect overlap and skips legitimate cleanup work.

**Fix.** Restore the call site and the helper. The helper's short-circuit guard, scope filter (same-`validityName` only), and `filterIterations` predicate with u_-strip are the load-bearing pieces — do not weaken any of them.

**Code.** [`prover.hpp::cleanUpAdmissionMapIntegration`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::updateEquivalenceClasses`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (call site immediately after `cleanUpAdmissionMap`), [`prover.hpp::filterIterations`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.cpp::removeUPrefixFromArguments`](../GL_Quick_VS/GL_Quick/src/prover.cpp). See also [D-66](40_decisions.md#d-66), [D-65](40_decisions.md#d-65), [I-40](#i-40), [I-22](#i-22).

---

<a id="i-44"></a>
## I-44 `exprOriginMap` is process documentation, not a proof input

**Scope.** Every prover code path. `exprOriginMap`, `mailOut.exprOriginMap`, `mailIn.exprOriginMap`, and any peer history container.

**Rule.** Proof-side control flow must not depend on what these maps contain. No rule firing, admission decision, hashburst routing, contradiction handling, or other algorithmic branching may be gated on `originMap.contains(...)` or any presence/absence check against history state. The forthcoming ASIC / fast-PC build will run the proof without an originMap entirely; the runtime state graph is the source of truth, the history is documentation.

**But maintenance is mandatory.** When a state mutation changes a *derived* expression — equi-class rewrite of a non-goal expression, integration-revival rewrite, any analogous mutation — the corresponding history line must be written to keep the documentation current. Goals in `toBeProved` are exempt (no upstream history line exists for a goal). Reads incident to writing (lookup the existing entry to construct its successor, `addOrigin`'s cap-full dedup and preference logic) are part of the write path and remain permitted.

**Why.** GL's algorithmic core targets a deterministic memory-access architecture. originMap is a string-keyed, append-mostly side log that scales with proof size and is not on the hot path. ASIC budgeting depends on the prover being independent of originMap; any algorithmic dependency introduced now blocks that build. The doc/process-graph role is preserved because every fact's derivation history is still recorded — just not consulted.

**Spot.** Grep new code for reads of `exprOriginMap` outside `addOrigin`, the hashburst dump, the visualiser, the verifier, and any code path explicitly marked "write-side dedup". Each remaining read should be a maintenance call (looking up an existing line to extend it), not a decision input.

**Fix.** When a needed-looking gate against originMap is identified during development, refactor: route the equivalent decision through the runtime state graph (the maps that drive actual proof progress — `intEncodedStatements`, `admissionMap`, `equivalenceClassesMap`, etc.).

**Code.** Rule 16 in the project conventions. [`prover.hpp::addOrigin`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (the canonical maintenance write site). See also [D-49](40_decisions.md#d-49), [I-35](#i-35).

### Weaknesses

- **Pre-existing violation at `applyEquivalenceClass`'s deposit gate.** The per-rewrite commit block in `prover.hpp::applyEquivalenceClass` reads `notInOriginMap = memoryBlock.exprOriginMap.find(appliedWithValidity) == end` and AND-combines it with `notInStmtLvl` to gate the commit of the rewritten expression into `intStatementLevelsMap`, `intKnownStatements`, `intEncodedStatements`, `intLocalEncodedStatements{,Set,Delta}`, the parallel int vectors, and (conditionally) `mailOut.statements`. That is a proof-side commit decision gated on originMap presence — exactly what this rule forbids. The gate predates the invariant and was not removed in the commit that introduced this rule. Resolution path: drop the `notInOriginMap` term from the gate and rely on `notInStmtLvl` (and/or `intKnownStatements`) alone for dedup. The ASIC build cannot ship while this gate remains.

---

<a id="i-45"></a>
## I-45 `toBeProved` goals reach canonical form under equi-class rewrites without changing namespace

**Scope.** `Memory::toBeProved` and the prover loop's discharge path.

**Rule.** Once per burst, in `performElemPhase3`'s end-of-burst sanitize block, `sanitizeToBeProved` walks the goal registry `intToBeProved` via its decoded lex-sorted snapshot (`decodeToBeProvedSorted` — same processing order the former string-keyed `std::map` produced). For each entry, every arg matching `it_*_lev_*_*` / `int_lev_*_*` whose equi-class (at the entry's validity scope or any visible ancestor scope) contains a strictly-higher-priority `it_/int_` peer is substituted in place. The entry's validity id is reused verbatim in the rewritten packed key; only the body args at `it_/int_` positions are rewritten. The original entry is erased; the new key takes its `(auxies, tags)` value. Duplicate-target collisions collapse.

`it_/int_` → `it_/int_` only. `repl_*` and plain (non-`it_/int_`) names never qualify and never participate in the priority ranking. This restriction is what eliminates the need for a position catalogue, a cascade-discharge, an additive insert + sweep, and three call sites.

**Why.** Discharge consumers do exact-match packed finds — `intToBeProved.find(packStatementKey(ie.originalId, ie.validityId))` on the delta row's ids. Rewriting the goal's validity would produce a different validity id and silently miss those finds; the re-key therefore reuses the old packed key's validity id verbatim, making the namespace preservation structural ([I-87](#i-87)). Keeping the goal's scope fixed preserves discharge integrity while normalising the body for canonical-form matching. The `it_/int_`-only replacement rule ensures plain Peano goals see zero substitution (no class peer can win priority against a non-`it_/int_` argument) and ensures cross-iteration substitution cannot widen the substitution surface (a rewrite never introduces `it_/int_` at a previously-non-`it_/int_` position).

**Spot.** A pending goal whose body still names a non-canonical class member after the canonical form has been derived — `toBeProved` walk shows an entry that the verifier would expect to be discharged. Or a TBP rewrite that touches a non-`it_/int_` position (e.g. a `repl_*` slot) — that would violate the `it_/int_` → `it_/int_` rule and is forbidden; the priority test rejects non-`it_/int_` peers structurally.

**Fix.** `sanitizeToBeProved` runs at the end of every burst. New TBP write sites do not need any catalogue / propagation logic — the once-per-burst sanitize handles canonicalization.

**Code.** [`prover.hpp::sanitizeToBeProved`](../GL_Quick_VS/GL_Quick/src/prover.hpp). Call site in `prover.cpp::performElemPhase3`'s `END_OF_BURST_SANITIZE` block, immediately after `sanitizeHashMemory(body)`. See also [D-67](40_decisions.md#d-67), [I-46](#i-46).

---

<a id="i-46"></a>
## I-46 hashMem rule registry is canonicalised by end-of-burst `sanitizeHashMemory` via the `expandedImplications` index

**Scope.** `Memory::expandedImplications` (per-LB index), `Mail::expandedImplications` (propagation), and the underlying registries that `eradicateImplicationFromLB` touches: `intEncodedStatements`, `intLocalEncodedStatements{,Delta}` + `intLocalEncodedStatementsSet`, `intStatementLevelsMap`, `intKnownStatements` (whole rows — both membership bits), `HashMemory::encodedMap` head LMVs, and `originals` chains across `overallHashMemory`, `localHashMemory`, `localHashMemoryDelta`.

**Rule.** Every implication installed into a LB's hash rules (via `addToHashMemory` from `addExprToMemoryBlock`'s implication branch) must also be inserted into `memoryBlock.expandedImplications` AND `memoryBlock.mailOut.expandedImplications`. Once per burst, at the end of `performElemPhase3` (after `reactToHypo`, before the EXIT trap), `sanitizeHashMemory(body)` walks `body.expandedImplications`. For each entry whose `it_/int_` args are now downprioritized under the active equi-classes (entry's own scope OR strict ancestor), the rewrite is mailed onto `mb.sameIterationInternalMail.statements` at the entry's `validityName` with an `equality1` line into `mb.sameIterationInternalMail.exprOriginMap`, and the old form is removed by `eradicateImplicationFromLB` from every per-LB registry that would otherwise dedup the re-push. The kernel's `addStatement` → `addExprToMemoryBlock` → disintegration → `addToHashMemory` chain on the next burst re-disintegrates and re-installs both fact AND rule sides with `disintegration` origins for every chain element via `trackExpansionHistory`.

`Mail::expandedImplications` is a staging field that the cross-LB pull does **not** carry: `mergeBatchInto` (and the retired `smashMail` before it) merges only `statements` + `exprOriginMap` into a receiver's `mailIn`, so `mailIn.expandedImplications` is never populated by routing. A child LB inheriting rules builds its OWN `expandedImplications` locally when it disintegrates the mailed compact statements into rules, and sanitizes them on its own next burst with its own visible equi-classes. (The `body.mailIn.expandedImplications` merge at the start of `performElemPhase1` is retained but sees only local staging.)

`it_/int_` → `it_/int_` only. `repl_*` and plain (non-`it_/int_`) names never qualify as replacements and never participate in the priority ranking. Priority: `int_*` outranks `it_*`; within the same prefix, lex-smallest wins.

**Why drop+mail, not direct install.** A prototype using `addToHashMemory(rewrittenImpl, …)` directly was implemented first. It bypassed disintegration: chain elements ended up without `disintegration` origins at the deposit scope, and `visualizer.cpp::buildStack` asserted `"no origin found"` for premises like `(in[7,1])` during chapter export. The drop+mail design routes through `addStatement`, which triggers disintegration, which emits the proper provenance via `trackExpansionHistory`.

**Why `sameIterationInternalMail`.** The registration gates at `addExprToMemoryBlock`'s entry would block a top-level re-push of the rewritten implication. `sameIterationInternalMail` bypasses them; kernel processing of mailed statements runs disintegration regardless.

**Spot.** A hashMem rule firing under stale names after class formation. A chain orphaned in `originals` because eradication missed a `localHashMemory*` LMV that still holds it. A `buildStack: no origin found` assertion firing during chapter export with the offending exprKey being a chain element (premise) of an equi-class-rewritten implication.

**Fix.** Every `addToHashMemory` call site that installs an implication into a LB must also update `memoryBlock.expandedImplications` and `memoryBlock.mailOut.expandedImplications` in the same code path. `sanitizeHashMemory(body)` must be invoked exactly once per burst, in `performElemPhase3` after `reactToHypo`. `eradicateImplicationFromLB` must scan all three hashMems (`overallHashMemory` + `localHashMemory` + `localHashMemoryDelta`) before erasing from `originals`. The mail-to-`sameIterationInternalMail` path is non-negotiable — direct `addToHashMemory` bypasses provenance.

**Code.** [`prover.hpp::sanitizeHashMemory`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`prover.hpp::eradicateImplicationFromLB`](../GL_Quick_VS/GL_Quick/src/prover.hpp), [`memory.hpp::Memory::expandedImplications`](../GL_Quick_VS/GL_Quick/src/memory.hpp), [`memory.hpp::Mail::expandedImplications`](../GL_Quick_VS/GL_Quick/src/memory.hpp), [`prover.cpp::performElemPhase3`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (the end-of-burst call site), [`prover.cpp::addExprToMemoryBlock`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (index population at implication install). See also [D-69](40_decisions.md#d-69), [D-33](40_decisions.md#d-33), [D-63](40_decisions.md#d-63), [D-64](40_decisions.md#d-64), [I-44](#i-44), [I-45](#i-45).

---

<a id="i-47"></a>
## I-47 RETIRED — `applyEquivalenceClass` compiled-implication branch (prototype, reverted)

**Status.** Retired. The invariant described a per-rewrite branch inside `applyEquivalenceClass` that was prototyped and reverted in the same squash. See [D-68](40_decisions.md#d-68) for the retirement rationale. The case is now covered structurally by [I-46](#i-46) — the `Memory::expandedImplications` index + end-of-burst `sanitizeHashMemory` re-absorbs the rewritten implication through the canonical `addStatement` chain, installing both fact and rule sides with `disintegration` origins in one pass.

---

<a id="i-48"></a>
## I-48 LB-disable bubble-up gates on main-namescope `toBeProved` only

**Scope.** The bubble-up block in `prover.cpp::deactivateUnnecessary` (the LB deactivation pass that walks `chainBlocks` bottom-up). Affects LB lifecycle, scheduler workload, runtime budget on bursts where deeper-scope `toBeProved` residue accumulates.

**Rule.** An LB is disabled when both conditions hold simultaneously:

1. No direct child in `block->simpleMap` is active (`!anyActiveChild`).
2. No `toBeProved` entry at `validityName == "main"` remains.

Entries at deeper scopes (hypothetical, OR-branch, integration boundaries) do not block disabling on their own — they require an active child to keep the LB alive.

**Why.** A deeper-scope `toBeProved` entry can only be discharged by an active child sub-block at that scope; the discharge then routes back up via integration. Once `!anyActiveChild` holds, no child can do that work and the entry is structurally stranded — no future GL state can touch it. The conjunction "no active children AND no main TBP" is the smallest sufficient condition for "the LB cannot produce future main-scope state." Keeping the LB active on stranded entries spends scheduler time per burst on a block that cannot make progress.

`sanitizeToBeProved` ([I-45](#i-45) / [D-67](40_decisions.md#d-67)) creates measurable deeper-scope residue by canonicalising entries at terminal scopes; under the looser pre-change rule (`toBeProved.size == 0`) these residuals would have kept LBs active indefinitely.

**Spot.** A burst-time profile showing LBs with `!anyActiveChild` and only deeper-scope TBP entries staying active across many bursts. Or a rung-1-style runtime ballooning on a proof that completes correctly but takes far longer than expected — root cause often "deeper-scope residue keeping otherwise-finished LBs in the scheduler."

**Fix.** The bubble-up block must check `(key & 0xFFFF) == NameMap::MAIN_ID` when surveying `block->intToBeProved` (the low 16 packed-key bits carry the validity id), not just `intToBeProved.size`. A single main-namescope hit is sufficient to keep the LB alive; deeper-scope hits alone are not.

**Interaction with the quiescent-burst skip ([D-194](40_decisions.md#d-194)).** The bubble-up survey (`deactivateRecursively`, invoked from `deactivateUnnecessary`) is a tree-wide POST-JOIN walk from the root, run whenever a theorem is proven. It surveys every ACTIVE node — including an LB that was SKIPPED (unswept) this iteration — reading only never-deloaded state (`intToBeProved` persistent I-108; child `isActive` + `simpleMap` edges on the never-deloaded LB slab, I-109 / I-110), so it forces no reload of a cold skipped LB. Skipping therefore preserves the exact I-48 deactivation schedule with no extra machinery: deactivation is driven by proofs (which skip cannot change), not by whether the parent was itself swept.

**Code.** [`prover.cpp::deactivateUnnecessary`](../GL_Quick_VS/GL_Quick/src/prover.cpp), bubble-up block. See also [D-70](40_decisions.md#d-70), [D-67](40_decisions.md#d-67), [I-45](#i-45).

---

<a id="i-50"></a>
## I-50 Radical subtree wipe runs at end-of-burst, never mid-kernel

**Scope.** `Memory::wipeSubtree` and its callers. Established by [D-72](40_decisions.md#d-72). Affects every site that wants to wipe the subtree of a closed impl scope.

**Rule.** Code that closes an implication-integration scope MUST queue the closed scope into `Memory::pendingWipeScopes` and let `performElemPhase3` drain the queue at the end of the burst (after `sanitizeToBeProved`, before the EXIT trap). Direct mid-kernel invocation of `Memory::wipeSubtree(scope)` from inside `addExprToMemoryBlockKernel`'s `sortedNew` loop is forbidden.

**Why.** The wipe erases `intStatementLevelsMap` and `equivalenceClassesMap` entries among many others. The kernel's `sortedNew` loop in `addExprToMemoryBlock` processes one new statement per iteration and asserts the statement's `intStatementLevelsMap` entry exists (the non-minting `lookupStatementLevels` probe). If the wipe runs at the impl-discharge sites mid-loop and the next item in `sortedNew` lives at a descendant of the closed subtree, its lookup misses and the kernel asserts out. The reintroduced `checkForEquivalence` gate also asserts that `equivalenceClassesMap` still contains the expression's validity scope before it enumerates equivalent variants, so the deferred-wipe invariant again protects both mid-loop look-ups.

The deferred drain pattern moves the wipe past the loop's exit, so every assert-look-up in the loop runs against the pre-wipe state.

**Spot.** A C++ assertion at `prover.cpp ~3812` or `~3993` / `~4068` triggered during incubator or Gauss processing; the assertion message names a validity that contains the closed impl scope as a prefix.

**Fix.** Wherever an impl discharge or OR convergence wants to eradicate its subtree, replace any direct `wipeSubtree(closedVid)` call with `pendingWipeScopes.mint(closedVid)` (the vid comes from a non-minting NameMap `lookup` + assert — see [I-93](#i-93)). The drain block at the end of the burst (`performElemPhase3`) is the only authorised invocation point.

**Drain block shape.** Snapshot the cold set's ids onto the per-slot gen-scratch byte-bump tier, reset the set (so `wipeSubtree`'s own inserts into other fields run against a fresh container), sort the ids by their decoded names (`compareSpans` byte-lex — the former string-set order, tie-free because vids are deduped and the interner injective), then wipe per vid:
```cpp
int16_t* ids = /* gen-scratch byte-bump run of pendingWipeScopes.decode(1..n) */;
body.pendingWipeScopes.resetToFresh();
std::sort(ids, ids + n, /* compareSpans over nameMap.decodeView */);
for (int32_t k = 0; k < n; ++k) body.wipeSubtree(ids[k]);
```

**Code.** [`memory.hpp::Memory::pendingWipeScopes`](../GL_Quick_VS/GL_Quick/src/memory.hpp); drain block at the end of [`prover.cpp::performElemPhase3`](../GL_Quick_VS/GL_Quick/src/prover.cpp). See also [D-72](40_decisions.md#d-72), [I-44](#i-44).

---

<a id="i-53"></a>
## I-53 Anchor predicates are never rewritten by equivalence-class substitution


**Scope.** Equivalence-class machinery — the shared rewrite helper `prover.hpp::enumerateEqClassRewrites` and the algebra-side admission hook `prover.hpp::applyEquivalenceClassToAdmissionMap`.

**Rule.** No equivalence-class substitution may produce a rewritten anchor predicate. Two gates enforce this jointly:

1. **Helper-level early return.** `enumerateEqClassRewrites` returns immediately when `baseExpr.rfind("Anchor", 0) == 0`. The helper emits zero rewrites for any expression whose base operator is an `Anchor*`. Both `applyEquivalenceClass` (algebra statements) and `applyEquivalenceClassToRejectedMapIntegration` (integration drop+mail) inherit the exclusion.

2. **Admission value-loop refuse.** `applyEquivalenceClassToAdmissionMap`'s value-key substitution loop tracks an `anchorChanged` flag and refuses the entire admission update if any element with prefix `(Anchor` would change under `ce::replaceKeysInString`. The value loop iterates `AdmissionMapValue::key` directly with `replaceKeysInString` rather than routing through the helper, so it needs its own gate.

The additive principle is preserved: the original concrete anchor stays unchanged in every container; only the *generation* of variant forms and the *insertion* of modified-anchor admission templates are suppressed.

**Why.** An anchor predicate (`AnchorPeano[N,i0,s,+,*,i1]`, `AnchorGauss[…]`, `AnchorIncubator[…]`) is the positional scope identity of its LB. Its argument slots carry conjecturer-assigned canonical names that the chapter exporter and the verifier treat as the immutable contract of the LB's scope. A class containing `(=[i0, i0_copy])` could otherwise rewrite the anchor to `AnchorPeano[N,i0_copy,s,+,*,i1]`, producing a duplicate anchor predicate the LB never declared. Downstream consumers (chapter row writer, verifier's `anchor handling` and `origin` meta-checks) have no canonical-choice rule for "which anchor does the scope mean?" and emit failures. The substitution also wastes the per-class budget enumerating identities that never enable a deduction — anchor predicates are not premises, not goals, and not admission keys whose substitution changes the integration outcome.

**Spot.**

- A rewritten anchor variant appearing in `intEncodedStatements` / `mailOut.statements` / `admissionMap`, in any chapter row, or in the hashburst dump's per-LB containers. On before the gates landed, the trapped LB's run produced 512 `AnchorPeano[…,repl_lev_…]` lines from this exact path.
- A verifier failure on `anchor handling` or `origin` meta-checks where the chapter cites one anchor string and the registry stores a substituted variant.
- The two-gate structure must stay symmetric: a future caller of `enumerateEqClassRewrites` that bypasses the early return (e.g. by computing rewrites inline) reintroduces the leak. A future admission hook that drops the `anchorChanged` flag reintroduces the leak.

**Fix.** Keep both gates in place. Where new substitution paths are added, route them through `enumerateEqClassRewrites` (gets the helper gate for free) or replicate the `(Anchor` prefix check at the new site.

**Code.** [`prover.hpp::enumerateEqClassRewrites`](../GL_Quick_VS/GL_Quick/src/prover.hpp) — top-of-function early return. [`prover.hpp::applyEquivalenceClassToAdmissionMap`](../GL_Quick_VS/GL_Quick/src/prover.hpp) — value-loop `anchorChanged` flag and post-loop `continue`. See [D-82](40_decisions.md#d-82). Compare to [I-36](#i-36) (arg-equalization collision filter) — same shape (drop a rewrite that breaks a positional invariant); [I-36](#i-36) is per-arg-slot, this invariant is per-base-operator.

---

<a id="i-54"></a>
## I-54 Implications cross the mail system only as compact `(implication<N>[…])` statements via `Mail::statements`


**Scope.** Mail subsystem on this branch.

**Rule.** No code path may deposit an implication into `Mail::implications` — the channel does not exist on this branch ([D-78](40_decisions.md#d-78)). The sole carrier for implication content across LBs is the compact `(implication<N>[…])` string deposited into `Mail::statements` at `validityName == "main"` by the D-76 deferred-compaction drain after `pool.join`. Receivers absorb the compact statement via `addExprToMemoryBlock(... status=3,...)`, which disintegrates it back into `(chain, head, …)` and installs the rule in `overallHashMemory` + `workingMemory` ([D-84](40_decisions.md#d-84)).

**Why.** Two parallel routing systems for the same logical content were redundant after D-76 made the compact-form deposit universal. Deleting `Mail::implications` simplifies the mail struct, smashMail/sendMail loops, and the per-burst clears. The hashburst dump (Rule 14) is updated to drop the three retired `.implications` sub-sections (mailIn/mailOut/sameIterationInternalMail).

**Spot.**

- A new `mailOut.implications.insert(...)` call site — illegal, the struct member is gone.
- A producer (broadcast site) that emits an implication only as a tuple without the paired `recordPendingCompaction` — silent loss on this branch. The two formerly-backup-less producers (`broadcastTheorems` load-time, `addExprToMemoryBlock` recovered re-broadcast) had their compact-form deposits added by [D-83](40_decisions.md#d-83).

**Fix.** Use `recordPendingCompaction` at the producer site. The single-threaded post-`pool.join` drain handles the actual compile + the deposit into the root's `mailOut` via `mergeBatchInto` (the next commit barrier ships it to descendants' pulls; `sendMail` is deleted).

**Code.** `memory.hpp::Mail` (no `implications` field). `prover.cpp` — eight `updateGlobalDirect`/`updateGlobal` broadcast sites + `broadcastTheorems` + `addExprToMemoryBlock` recovered re-broadcast site, each call `recordPendingCompaction`. `prover.hpp::standardProcessing` — the `mailIn.statements` `status=3` drain (pre-burst driver `performElemPhase1`).

---

<a id="i-51"></a>
## I-51 Every mail deposit ships rules with an empty level set


**Scope.** Mail subsystem on this branch — the D-76 deferred-compaction drain in `prover.cpp::proveKernel` and any historical or future site that constructs a `pair<ExpressionWithValidity, std::set<int>>` for `mailOut.statements`. Also covered every `mailOut.implications.insert` site on the retired tuple channel (which always used `std::set<int>` — preserved as historical evidence of the right answer).

**Rule.** The `std::set<int>` paired with each `ExpressionWithValidity` in a `mailOut.statements` deposit MUST be empty (`std::set<int>`). No deposit may synthesise a non-empty set from chain length, sender LB depth, sender hashmem state, or any other producer-side quantity. The receiver computes the recovered rule's effective levels from its OWN scope when `addExprToMemoryBlock(... status=3,...)` calls `addToHashMemory` → installs into the receiver's `LocalMemoryValue::levels` and, on every subsequent firing, propagates into the derived statement's `intStatementLevelsMap` entry via `union(rule.levels, premise[i].levels)`. A non-empty deposited set leaks producer-side levels into the receiver's union, which has no anchor in the receiver's LB tree and silently fails the discharge gate.

**Why.** The discharge gate `allLevelsInvolved` in `prover.hpp::dischargeToBeProved` (statement-level run form since the S5b levels-run flip; the historical set form read `addExpressionLevels.size` / `.find(0)`) is:

```cpp
bool has0 = false;
for (int32_t j = 0; j < lvN; ++j)
    if (lvRun[j] == 0) { has0 = true; break; }
bool allLevelsInvolved = (lvN == memoryBlock.level + 1);
if (lvN == memoryBlock.level && !has0)
    allLevelsInvolved = true;
```

Both branches compare `levels.size` against the receiver LB's depth (`memoryBlock.level + 1` for the primary, `memoryBlock.level` for the alternate that skips anchor scope). An extra integer in the set — any integer — makes `size > level + 1`, fails both branches, and the gate returns false. The derived expression lands in the LB's `intEncodedStatements` (rule did fire, head is materialised, all dep checks pass) but is never promoted to `globalTheoremList`. The theorem disappears with no assert, no verifier error, no toBeProved residue — only a missing line in `theorems.txt`.

**Spot.**

- Any `for (int i = 0; i <= kySize; ++i) compactLevels.insert(i);` (or a non-empty `std::set<int>{…}` literal) in the D-76 drain or any future mail-deposit code.
- A descending stmt-count plateau on the trapped LB at a hypothesis-LB target, with the toBeProved set frozen at its initial size and the head expression visible in intEncodedStatements at v=main. Cross-check by reading the head's `intStatementLevelsMap` entry — if its size exceeds `memoryBlock.level + 1`, the gate is silently swallowing the discharge.
- 64 missing addition-by-2 uniqueness theorems in `theorems.txt` (or any similar cascade where N=2 base is proved but every rung from N=6 up is missing — that pattern is the surface symptom).

**Fix.** Use `std::set<int>` (empty). Apply at the producer site; the receiver does the rest.

**Code.** `prover.cpp::proveKernel`'s deferred-compaction drain block (the per-tuple body inside the `for (const std::tuple<std::string, int, int>& e: pendingCompactionQueue)` loop). Receiver-side propagation is in `prover.cpp::addExprToMemoryBlock`'s `status=3` branch → `addToHashMemory` call. See [D-77](40_decisions.md#d-77), [G-46](50_gotchas.md#g-46), and [`02_glossary.md::levels`](02_glossary.md#levels) for the full semantics of the level set.

---

<a id="i-52"></a>
## I-52 `compilation` row's `rest[0]` cites the binary's canonical reconstruction


**Scope.** Prover — `prover.cpp::proveKernel`'s deferred-compaction drain (the per-tuple body inside the loop over `pendingCompactionQueue`). Applies to every `compilation` origin emitted from that drain.

**Rule.** The `compilation` origin row's `rest[0]` (the "original expanded implication" reference) MUST be the canonical body reconstructed from `compiledExpressions[compactCore].elements` via `this->reconstructImplicationFullBind(canonicalKey, canonicalHead)`. It MUST NOT be the raw `original` string popped off `pendingCompactionQueue`.

**Why.** `compileImplicationToCompact` dedups alpha-equivalent inputs to the same `implication<N>` name; the binary stores only the first-seen body. A later input whose binder ordering differs from the binary registered form will be alpha-equivalent but not structurally identical. The verifier's `check_compilation` reconstructs from the binary and compares with `rest[0]` modulo `_normalize_with_unchangeables`; the normalizer does not try binder permutations, so non-canonical citations are rejected.

**Spot.** `compilation success 0, failure N` in the verifier report on chapters that emit compaction rows (e.g. `check_induction_condition`). Inspecting the failing row's `rest[0]` vs the binary's `implication<N>` `elements` shows the inner `>[w_i,w_j]` group's body args swapped (e.g. `in3[w2,w_j,w_i,+]` in the chapter vs `in3[w2,w_i,w_j,+]` in the binary's reconstruction).

**Fix.** Look up `compiledExpressions[ce::extractExpressionUniversal(compactImpl)]`; split `.elements` into key (premises) + value (head); call `this->reconstructImplicationFullBind(key, value)`; pass the result as the compilation origin's sole dep. See [D-81](40_decisions.md#d-81).

**Code.** `prover.cpp::proveKernel` — the deferred-compaction drain block's `addOrigin(... "compilation"...)` call. Reconstruction helpers `ce::extractExpressionUniversal` (in `compiler.hpp`) and `this->reconstructImplicationFullBind` (in `prover.hpp::ExpressionAnalyzer`).

---

<a id="i-55"></a>
## I-55 Mail absorb runs pre-fixpoint; rules absorbed at burst N's mailIn fire in burst N's hashburst


**Scope.** Prover — `prover.cpp::performElemPhase1` (the pre-burst absorb phase).

**Rule.** The mail-absorption block (internal-mail absorb+clear, mailIn.exprOriginMap merge, eq-class origin sync, mailIn.statements `status=3` drain, mailIn clears — the phase-1 `standardProcessing` call) runs **before** the request-generation + hashburst pass (phase 2), in the same burst cycle. The clear of `workingMemory` / `externalStatements` / `intExternalStatements` is also at the top of this block; the absorb refills them. Result: the rule (or fact) arriving at burst K's `mailIn` is installed in `overallHashMemory` + `workingMemory` (rule case) or `externalStatements` + `intExternalStatements` (fact case) **before** the same burst's request generation runs, and fires in the same burst's hashburst.

**Branch history.** The post-fixpoint placement introduced by commit (ASIC 0.1 reshuffle 3/8) added one PK of latency at the receiver between mail arrival and the rule firing. Combined with D-76's deferred-compaction broadcast (one PK at the producer), the contradiction-LB rule landed two PKs late, which pushed `__contradiction__(=[a,b])` LBs past `MAX_NAME_IDS` during combinatorial substitution. The pre-fixpoint position was restored on 2026-05-20 ([D-79](40_decisions.md#d-79)), eliminating the receiver-side PK; only the D-76 producer-side PK remains as residual latency vs main HEAD.

**Why.** Contradiction sub-LBs assume `(=[a,b])` and inflate combinatorially with the equivalence-class machinery; their only pruner is a discharge premise arriving via mail. Each extra PK between proving the premise and firing it locally is an extra round of unchecked combinatorial inflation. The same-burst absorb-fixpoint property keeps the receiver side latency-free.

**Spot.**

- `mailIn.statements` absorb call sites located after the phase-2 hashburst (i.e. in phase 3) instead of in the phase-1 absorb.
- `workingMemory`/`externalStatements`/`intExternalStatements` clear placed after the fixpoint instead of at the top of the absorb block.
- Request-generation batches reading `body.workingMemory` when the absorb hasn't yet populated it (would silently miss this burst's mail rules).

**Fix.** Keep the pull + `mailIn` absorb at the phase-1 position, after the entry trap. `mailOut` is no longer flushed in phase 3 (the old `sendMail` + clears are deleted) — the commit barrier drains it at the post-join seam.

**Code.** `prover.cpp::performElemPhase1` — the `PRE_FIXPOINT_MAIL_ABSORB` block.

---

<a id="i-49"></a>
## I-49 HashMemory subkey containers carry an owner-set, dropped key-by-key when the set empties

**Scope.** The four `HashMemory` containers — `normalizedEncodedKeys`, `normalizedEncodedSubkeys`, `normalizedEncodedSubkeysMinusOne`, `normalizedEncodedSubkeysMinusTwo`. Established by [D-71](40_decisions.md#d-71).

**Rule.** Every insertion into one of these four maps MUST record the originating implication + scope as an `ExpressionWithValidity` in the value-side `std::set<ExpressionWithValidity>`. Wipes MUST iterate the owner-set, remove only owners whose validity matches the closed subtree, and erase the key only when the owner-set becomes empty.

**Why.** A subkey can legitimately be shared by multiple `(impl, scope)` pairs — `makeNormalizedSubkeys` walks every permutation of an implication's key array; two different implications can produce the same subkey under different permutations. Without owner tracking the subtree wipe ([D-72](40_decisions.md#d-72)) cannot tell which subkeys it may erase: erasing a subkey owned by a surviving implication breaks that implication's hash-lookup fast path.

**Spot.** A wipe that removes a subkey while at least one un-wiped implication still references it. Downstream symptom: `preEvaluateFromEncoded`, or the request generator's grow-DFS probe, returns `false` for a query that should hit, because the subkey is gone. Manifests as missing hash inferences rather than as a crash — hard to spot without instrumentation.

**Fix.** Two insertion-site rules:

1. `addToHashMemory` inserts `ExpressionWithValidity(curOrigImpl, validityName)` at every subkey / key insert (lines around the four `[intSubKey].insert(owner)` calls in `memory.cpp`).
2. `makeNormalizedKeysForAdmission` takes `const std::string& originalImpl, const std::string& validityName` parameters and uses them at every subkey / key insert in its body. Marker LMVs in `encodedMap` are also tagged `originalImplication = originalImpl` / `validityName = validityName`.

The lookup contract is unchanged — `find!= end` / `count` behave identically on the new map type.

**Code.**
- [`memory.hpp::HashMemory`](../GL_Quick_VS/GL_Quick/src/memory.hpp) — declarations of the four maps.
- [`memory.cpp::addToHashMemory`](../GL_Quick_VS/GL_Quick/src/memory.cpp) — insertion sites.
- [`memory.cpp::makeNormalizedKeysForAdmission`](../GL_Quick_VS/GL_Quick/src/memory.cpp) — insertion sites + owner parameters.
- `Memory::wipeSubtree` — wipe contract (owner-set filter + drop-on-empty + secondary-index sync into `remainingArgsNormalizedEncodedMap`).

See also [D-71](40_decisions.md#d-71), [D-72](40_decisions.md#d-72), [`20_core_concepts/02_hash_engine.md`](20_core_concepts/02_hash_engine.md).

---

<a id="i-57"></a>
## I-57 External-expression mail comes only from direct ancestors

**Scope.** The cross-LB mail system — `Mail::statements`, `Mail::exprOriginMap`, and any mail container that propagates an expression (with or without history) from one LB to another. Established 2026-05-25 as the routing-direction rule for external mail.

**Rule.** Mail delivery is strictly parent → child. An LB's `mailIn` receives **only** mail authored by one of its direct ancestors (parent, grandparent, …, root). A sibling never mails to a sibling; a child never mails upward to a parent; an unrelated LB on a different subtree never mails at all. Equivalently: when an expression with history lands in an LB's `mailIn`, the producer can be exactly identified as one of the LB's `parentMemory`-chain entries.

**Why.** The kernel's correctness reasoning depends on this routing constraint: every fact reaching an LB via mail is valid in a scope that visibly subsumes the LB's own scope, so its truth at the LB is by descendant-inheritance. Sibling-to-sibling or child-to-parent mail would break the inheritance contract and require a heavier scope check at the receiver. The `parentMemory`-chain-only routing also localises the producer-set for debug purposes — chasing a malformed mail item back to its origin reduces to walking the recipient's ancestor chain, never enumerating the LB tree.

**Spot.** A new mail-delivery code path that targets an LB that is **not** a direct descendant of the producer. Specifically:
- A `MailLog::pull` that walks anything other than the recipient's own `parentMemory` chain, or a commit that deposits a batch into a log whose owner's subtree does not contain the producer.
- `mailOut` reads followed by writes into siblings' / cousins' `mailIn`.
- Cross-subtree propagation paths added during refactors of the mail pipeline (the most likely regression site).

If a chapter-export `buildStack` crash reports an expression that is in this LB's `exprOriginMap` but not in its `intEncodedStatements`, the dep is producer-local — the producer is exactly one of this LB's `parentMemory` ancestors; instrument each ancestor's `fillMailOut` (with that ancestor's full chain to root) to find which one shipped the dep.

**Fix.** Mail routing must walk the producer's descendant subtree and refuse any recipient that is not a descendant. The receiver-side check is simpler: `body->parentMemory`-chain enumeration of the producer's chain (via the producer LB pointer threaded through `Mail`) confirms ancestry; mismatch is an assert.

**Code.** (Mechanism replaced by the pull model — [D-137](40_decisions.md#d-137). The routing law is unchanged; only the carrier moved from push to pull.)
- `mail_log.hpp::MailLog::pull` — receiver-side ancestor walk: a recipient ingests only the un-cursored batches of its `parentMemory`-chain ancestors. This is the exact dual of the old descendant routing, so the "only direct ancestors" rule holds by construction.
- `prover.cpp` — the MailLog commit barrier at `proveKernel`'s post-join seam: each LB appends its own `mailOut` to its own log single-threaded; a producer never deposits into another LB's log.
- `prover.hpp::standardProcessing` — pre-burst and post-burst absorb of `body.mailIn` on the receiver side.
- `prover.hpp::fillMailOut` — producer-side write to `mailOut`.

The retired push carrier (`smashMail` per-core mailbox collector sorted by recipient `exprKey`) was deleted with the rest of the broadcast machinery; the load-time grid-wide broadcast (`broadcastTheorems`) — the one non-ancestor path — stores its batch once in the root's log and self-injects the root, so every receiver still gets it via the ancestor walk.

Mirror in: (FOUNDATIONAL: mail flows parent → children only; D-51 confines contradiction record to `__contradiction__` LB).

---

<a id="i-67"></a>
## I-67 `g_buildStackPath` per-frame insertion ownership; erase only when THIS frame inserted

**Scope.** `visualizer.cpp::ExpressionAnalyzer::buildStack` — the chapter-export walker's path-cycle filter `g_buildStackPath` (thread-local `std::set<ExpressionWithValidity>`). Established 2026-05-26 after the cycle-filter false-negative on `(=[2,8]) ↔ (in2[8,6,3])` (chapter `86_check_induction_condition.txt`).

**Rule.** Every `buildStack` invocation captures whether ITS own attempt at inserting `proved` was the first insert (`insertedHere = g_buildStackPath.insert(proved).second`). Every exit path's `erase(proved)` is guarded by `if (insertedHere)...`. A `buildStack` frame whose `insertedHere == false` (because an outer frame is already walking `proved` further up the stack) MUST NOT erase `proved` from `g_buildStackPath` on its way out.

**Why.** `g_buildStackPath` is keyed by `(expression, validity)` and uses set semantics — no refcount. When a node `X` legitimately appears at two different recursion depths (outer frame walking `X`; inner frame re-entering `X` because `X` is on the dep chain of a candidate currently being explored), the inner frame's `g_buildStackPath.insert(X)` is a no-op (the entry is already there from the outer frame). The inner frame's exit `erase(X)` removes the entry that the outer frame had inserted, breaking the cycle filter for the outer frame's remaining candidates. Subsequent candidates of the outer frame's `proved` then see `X` NOT on the path, accept cyclic deps, and emit the cyclic chapter row.

**Spot.** A regression-emerging cycle in a chapter `origin chain termination` verifier failure whose nodes are mutually-citing (e.g. `A` cites `B` as a dep AND `B` cites `A` as a dep) — when `A` and `B` are both reachable from the chapter root and both have at least one origin candidate. Particularly likely with:
- An expression whose ONLY origin at a given LB is an `equality1` rewrite citing a recently-derived equality (creating a back-edge from the rewrite-target to the equality).
- A multi-candidate node whose D-49-preferred (non-equality) candidates also reference the rewrite-target as a dep.

**Fix.** Five exit sites in `buildStack` are guarded by `insertedHere`: early-return broadcast inside the candidate loop, subtreeOk-success return, contradiction-LB-switch tail-call, fallback broadcast early-return, last-resort fallback return. The single insert site at the top captures `.second` into `insertedHere`. No other `buildStack` semantics changed.

Alternative considered and rejected: `std::map<Key,int>` refcount. The boolean-per-frame approach is local to each invocation, costs one `bool` per stack frame, and the semantics match exactly: each invocation's exit pairs with its own entry's `.second`. The refcount variant adds heap overhead per insert and complicates the read-side without adding correctness.

**Code.**
- `visualizer.cpp::ExpressionAnalyzer::buildStack` — single insert at top captures `.second`; five erase exits guarded by `insertedHere`.
- `tests/test_equi_reshuffle.cpp::buildstack_path_refcount_doubleentry` — synthetic double-entry case; pre-fix emits cyclic chapter row, post-fix picks the acyclic candidate.

See also [D-101](40_decisions.md#d-101), [D-51](40_decisions.md#d-51) (path-stack cycle filter design — this invariant patches a hole in that design).

---

<a id="i-68"></a>
## I-68 Algebra `admissionMap` writes from the hashburst marker branch are staged and drained post-fixpoint, never applied inline

**Scope.** Prover. `memory.cpp::checkLocalEncodedMemoryStatic` (producer) and `memory.cpp::ExpressionAnalyzer::drainAdmissionKeysAlgebra` (consumer); the per-burst buffer `Memory::admissionKeysAlgebra`.

**Rule.** The hashburst marker branch performs **no** algebra-`admissionMap` mutation inline. For each new admission key it discovers it appends one `AdmissionKeyAlgebraRecord{key, value}` to `memoryBlock.admissionKeysAlgebra`. The four writes — `admissionMap` insert, `admissionStatusMap = false`, `varsInAdmissionMapKeys` population, `revisitRejected2` revival — happen only in `drainAdmissionKeysAlgebra`, called once after the fixpoint loop and before the post-burst `standardProcessing`. The drain replays records in append (firing) order and re-applies the consumed-key gate per record. The buffer is cleared at burst start beside the other per-step delta clears ([I-61](#i-61)).

**Why.** Keeps the hashburst hot path free of algebra-container mutation so the forthcoming LB-split/merge step has a single well-defined admission drain point ([D-103](40_decisions.md#d-103)). Behaviour-preserving: the burst never reads the *real-scope* `admissionMap` entries the marker branch defers — the only in-burst `admissionMap` reader, `isAdmitted` (reached via `prepareIntegration`'s hypothetical disintegration), operates on a disjoint sentinel scope — and `revisitRejected2`'s revival cohorts land on `sameIterationInternalMail`, which the post-burst `standardProcessing` drains regardless — so the post-loop placement is invisible to the burst, provided the firing-order and per-record consumed-gate properties hold (`cleanAdmissionMap`'s closure is order-sensitive — [I-41](#i-41)).

**How to spot.** Search `checkLocalEncodedMemoryStatic`'s marker branch for `admissionMap.insert` / `admissionStatusMap[` / `varsInAdmissionMapKeys.insert` / `revisitRejected2(` — all should be absent (replaced by one `admissionKeysAlgebra.push_back`). The four writes appear only inside `drainAdmissionKeysAlgebra`, which is called exactly once per burst in `performElemPhase2`, before phase 3's post-burst (`POST_FIXPOINT_MAIL_FLUSH`) `standardProcessing`.

**How to fix on violation.** Move the misplaced write back into the staged record + drain. Do **not** re-inline an admission write inside the hashburst loop. Preserve append (firing) order and the per-record consumed-gate.

**See also.** [D-103](40_decisions.md#d-103), [I-60](#i-60), [I-61](#i-61), [I-41](#i-41).

---

<a id="i-69"></a>
## I-69 Integration `prepareIntegration` seed registration from the hashburst marker branch is staged and drained post-fixpoint, never applied inline

**Scope.** Prover. `memory.cpp::checkLocalEncodedMemoryStatic`'s marker branch (producer) and `memory.cpp::ExpressionAnalyzer::drainDeferredIntegrationPreps` (consumer); the per-burst buffer `Memory::deferredIntegrationPreps`. Integration-side analog of [I-68](#i-68).

**Rule.** The marker branch performs **no** integration-admission registration inline. Where it would call `prepareIntegration(rplExpr2, argSet, mb, validityName)` it instead appends one `DeferredIntegrationPrep{expression, unchangeableArgs, validityName}` to `memoryBlock.deferredIntegrationPreps`. The `prepareIntegration` call — which writes `overallHashMemory.admissionMapIntegration` and the `varsInAdmissionMapIntegrationKeys` cache via `prepareIntegrationCore2` — runs only in `drainDeferredIntegrationPreps`, called once after the fixpoint loop, immediately before the post-burst `standardProcessing` (beside `drainAdmissionKeysAlgebra`). The drain replays records in append (firing) order. The buffer is cleared at burst start beside the other per-step delta clears ([I-61](#i-61)).

**Why.** Keeps the hashburst hot path free of integration-admission mutation for the forthcoming LB-split step, with one well-defined post-fixpoint drain point — the integration-side companion of the algebra deferral ([I-68](#i-68), [D-103](40_decisions.md#d-103)). The `prepareIntegration` seed registration is the **only** `admissionMapIntegration` write reached from inside the fixpoint loop (`checkLocalEncodedMemoryStatic`); the `updateAdmissionMapIntegration` cascade runs in `addExprToMemoryBlock` at the post-burst absorb — already outside the hashburst — and is left inline. Behaviour-preserving: the producer fires inside the clear→drain window, so the drain captures every staged record; the registered templates are present before the post-burst absorb's `updateAdmissionMapIntegration` cascade consumes them; the one-burst delay of the seed registration is benign (Gauss fold + the full AnchorGauss batch prove, incubator output byte-identical including rung1, verifier airtight; ~6% runtime cost vs main).

**How to spot.** `checkLocalEncodedMemoryStatic`'s marker branch contains `deferredIntegrationPreps.push_back(...)`, not a direct `prepareIntegration(...)`. The `prepareIntegration` replay appears only inside `drainDeferredIntegrationPreps`, called exactly once per burst in `performElemPhase2`, immediately after `drainAdmissionKeysAlgebra`.

**How to fix on violation.** Move the misplaced `prepareIntegration` back into the staged record + drain. Do **not** re-inline it inside the hashburst loop; do **not** instead defer the `updateAdmissionMapIntegration` cascade — that producer runs in `standardProcessing`'s absorb, *outside* the burst-start-clear→drain window, so its staged records are wiped before the drain ever runs (empirically 89 staged / 0 drained for the Gauss fold LB), silently dropping the cascade and leaving the fold theorem unproved.

**See also.** [I-68](#i-68) (algebra-side analog), [D-103](40_decisions.md#d-103), [I-61](#i-61), [I-22](#i-22).

---

<a id="i-70"></a>
## I-70 The request-generation validity prune is a sound over-approximation of the firing gate — it may never drop a request that could fire

**Scope.** Prover request generation. `ExpressionAnalyzer::ownerSetHasComparable` (predicate) and its call sites: `preEvaluateFromEncoded`, the grow-DFS + seed + merge inside `generateEncodedRequestsStatic`, `makeMandatoryEncodedStatementLists2Static`, and `filterIntEncodedStatements` (whose `alsoAcceptFullKeys` flag widens the gate to the full-key map for a zero-length stump). Reads each owner's stored validity id (`OwnerSet::owners` maps owner → id); the firing-side mirror is `checkLocalEncodedMemoryStatic`.

**Rule.** At every owner-set match site a request is pruned only when its consensus validity is non-`main` AND `nm.comparable` to **no** owner of the matched key. The predicate must stay implied by "the request can fire": it is a runtime optimisation, never a soundness gate. The firing gate enforces the same comparability independently, so the prune is forbidden from changing results — `files/theorems/theorems.txt` must be byte-identical with the prune active or stripped.

**Why it is sound.** A request fires only if some rule R at scope S is comparable to the request's consensus V. R stamps S as an owner of every (sub)key projection the request matches — including each single-premise projection — in the same `addToHashMemory` / `makeNormalizedKeysForAdmission` pass that installs R's firing LMV. Scopes form a tree (single-parent `encodePush`), and the partial validity at any growth stage is an ancestor of V, so `comparable(S, V)` implies `comparable(S, partial)`. Hence "∃ owner comparable to the partial validity" holds at every match length from the single-statement pre-filter up to the full key; pruning when none is comparable removes only non-firing requests. The `main`-skip (`requestVid == MAIN_ID → keep`) is correct because `main` is the root ancestor of every scope and a matched key always has ≥1 owner.

**How to spot.** Each prune site reads `ownerSetHasComparable(it->second, <consensus vid>, nm)` after a `find` hit; the consensus vid is the `deeperOf` of the request's premises (in scope at the site, recomputed via `deeperOf` inside `preEvaluateFromEncoded`). each owner's validity id is stored with it in `OwnerSet::owners` (encoded once at insert, removed with its owner at wipe) ([D-105](40_decisions.md#d-105)).

**How to fix on violation.** If `theorems.txt` diverges with the prune active, the predicate is over-pruning (a real bug). Do **not** loosen the firing gate. Find the site whose consensus vid or owner-set is wrong — prime suspects: a `deeperOf` over premises that were not actually pairwise comparable (wrong consensus), or an owner stored with the wrong validity id at insert (the id is encoded once and rides with its owner, so the wipe cannot desync it).

**See also.** [D-105](40_decisions.md#d-105), [D-55](40_decisions.md#d-55) (firing-site comparability gate), [I-49](#i-49) (owner-set refcount), [D-72](40_decisions.md#d-72) (owner-set introduction).

---

<a id="i-71"></a>
## I-71 A key in `admissionMap` is never also in `consumedAdmissionKeys` — every admissionMap insert skips a consumed key

**Scope.** Prover admission registration. The three inline `admissionMap` insert sites — `updateAdmissionMap`, `updateAdmissionMapRecursion`, `applyEquivalenceClassToAdmissionMap` (`prover.hpp` / `prover.cpp`) — plus the post-fixpoint `drainAdmissionKeysAlgebra` (`memory.cpp`), which already carried the gate.

**Rule.** Before inserting key K into `overallHashMemory.admissionMap`, each site checks `consumedAdmissionKeys.find(K) == end` and skips the insert otherwise. Consume (`cleanAdmissionMap`) erases K from `admissionMap` in the same step it inserts K into `consumedAdmissionKeys`, so the two are disjoint at consume time; this invariant keeps them disjoint afterward by refusing to re-add a consumed key. `isAdmitted` asserts the exclusion once it finds the marker in `admissionMap`.

**Why.** The delete → send → reabsorb completion of [D-106](40_decisions.md#d-106) re-disintegrates a resent compound, which re-enters admission for its product marker. If that marker's admission was already consumed and an inline insert re-added it, the marker would sit in both maps and the `isAdmitted` assert would fire (observed in Peano without the gate). The drain already skipped consumed keys; extending the same skip to the inline sites makes the exclusion hold on every path.

**How to spot.** The `isAdmitted` `assert(consumedAdmissionKeys.find(marker) == end)` firing after the marker is found in `admissionMap`.

**How to fix on violation.** Add the `consumedAdmissionKeys` skip to the offending insert site (mirror the others). Never weaken the assert.

**See also.** [D-106](40_decisions.md#d-106), [I-22](#i-22).

---

<a id="i-72"></a>
## I-72 `checkForEquivalence` suppresses only on a `fullyDisintegrated` variant; the flag is a sound under-approximation

**Scope.** Prover disintegration gate. The cFE variant check inside `addExprToMemoryBlock`; the `fullyDisintegrated` field of `StatementFlags` (the value type of `intKnownStatements`, `memory.hpp`); the producer in `disintegrateExpr2`.

**Rule.** `checkForEquivalence` returns true (suppress re-disintegration of an equivalence-class variant) only when the matched variant's registry row is `registered` with `fullyDisintegrated == true` — never on locality or mere presence. `fullyDisintegrated` is set true exactly when an expression entered `disintegrateExpr2` and came back fully disintegrated: **no existence inside** (vacuously true — nothing to witness), or **every existence inside got at least one admitted witness** (an `it_`/`int_` in `admittedVars`; one per existence is enough). It is an **under-approximation**: the producer must err toward `false` whenever full disintegration is uncertain — a false `false` costs only a redundant re-disintegration, whereas a false `true` suppresses an expansion that was not actually completed (the mirror bug).

**Why.** Locality is too weak a gate. A local existence whose witness is rejected (the Gauss `existence1[1,it_0_lev_4_0,7,5]`, `it_0_lev_4_0 ≡ 9`, whose `in3[it_0_lev_4_0,7,marker,5]` is never admitted) is local but not fully disintegrated; under a local gate it suppressed the canonical `existence1[1,9,7,5]` (witness admitted against `in3[9,7,marker,5]`), the two mutually blocked, and the Gauss fold-`n+1` step never produced `in3[9,7,…,5]`. Gating on full disintegration leaves the rejecting twin unflagged, so it stops suppressing the canonical and Gauss closes.

**How to spot.** A theorem that depends on an existential witness stalls even though its marker is in `admissionMap`; the target-LB `[CFE-EXISTENCE-TRAP]` shows the equivalent twin disintegrating-and-rejecting each burst while the canonical never lands its witness.

**How to fix on violation.** If a witness stalls, confirm the rejecting twin's entry is `fullyDisintegrated == false` (it must be, since its witnesses are not in `admittedVars`); if it is wrongly `true`, the producer's witness-grouping or coverage test over-approximated — tighten it toward `false`. Never gate cFE back on locality or presence.

**See also.** [D-108](40_decisions.md#d-108), [I-70](#i-70) (request-prune sound over-approximation — same "never drop what could fire" discipline), [I-71](#i-71).

---

<a id="i-77"></a>
## I-77 hashburst deposits are applied in canonical sorted order, independent of request-generation order

**Scope.** Prover hashburst. The single-pass streaming fixpoint in `performElem2` (`BurstSink::consume`), `checkLocalEncodedMemoryStatic` (the `FiringRecord` producer), and `applyFiringRecords` (the sort + apply).

**Rule.** A hashburst's order-sensitive deposits — the cap-bounded `addOrigin` selection on `sameIterationInternalMail.exprOriginMap`, the per-head last-write on `disintegrationSignals`, and the firing-order drains of `admissionKeysAlgebra` and `deferredIntegrationPreps` — must be a function of the firing SET, not the order requests were generated/evaluated. `checkLocalEncodedMemoryStatic` therefore appends a `FiringRecord` per firing (onto the task's sealed-page record chain, `I-135`) instead of depositing inline; `applyFiringRecords` sorts a pointer INDEX over the parts' records by a total content key (expression, validity, kind, then per-kind fields) before applying — a strict-total-order sort has a unique output sequence and content-identical duplicates are byte-indistinguishable in the deposit stream, so the index sort's deposit sequence is content-determined regardless of input permutation and sort algorithm. The plain sets (`statements`, `canBeSentIds`, `canBeSentMarkerIds`) are order-independent already and need no sort.

**Why.** The LB split partitions an LB's rules across `n` copies and pools their firings; the merged deposits must be byte-identical to the single-LB burst regardless of the partition. Inline, order-dependent deposits would make the merge depend on which rule landed in which copy. The sort is the property that makes the partition non-functional. Deactivation is deliberately NOT part of this — the phase-2 burst is read-only and never deactivates; the close is handled in phase 3's post-burst `standardProcessing` ([I-66](#i-66)).

**How to spot.** A split run's processed proof graph diverges from the single-copy reference on a NON-deactivating LB (origin selection or admission-revival order differs); or `test_lb_split.cpp`'s `apply_firing_records_order_independent` fails (two permutations of one record set produce different deposits).

**How to fix on violation.** Make the `applyFiringRecords` comparator a strict total order over every field that affects a deposit, so no two distinct records tie — that totality is what makes the pointer-index sort's output content-determined. Never move an order-sensitive deposit back inline into `checkLocalEncodedMemoryStatic`; never edit the comparator body without user approval (its order is a frozen byte contract).

**See also.** [D-117](40_decisions.md#d-117), [D-124](40_decisions.md#d-124), [I-44](#i-44) (origin map is process documentation — why a different surviving origin preserves theorems), `I-135`.

---

<a id="i-80"></a>
## I-80 `OwnerSet::partitionIds` IS the owner record — one packed composite id per owner, serving ownership, comparability, and the LB-split partition at once

**Scope.** Hash engine. The four `normalizedEncoded*` owner-set maps in `HashMemory`; the owner inserts in `addToHashMemory` and `makeNormalizedKeysForAdmission`; the owner erase in `Memory::wipeSubtree`'s `wipeOwnerMap`; the readers `ownerSetHasComparable` and `partitionAccepts`.

**Rule.** An owner exists iff its composite id `makePartitionId(NameMap::encode(expandedOriginal), scopeVid)` is in `OwnerSet::partitionIds` — there is no separate owner container (`D-133` unified the former string-keyed `owners` map into this set; the historical two-container lockstep is structural now). Both halves are NameMap-minted at install; the composite is a pure function of the owner, never minted by a separate counter. The D-72 ownership rule reads through it (a key drops when the set empties, [I-49](#i-49)); the D-105 comparability prune reads each id's LOW half (the scope validity id); the D-119 split filter takes `id % N`. The wipe erases an owner by decoding the low half for the closed-scope prefix test — one erase removes the whole owner record.

**Why.** A split executor `n` of `N` accepts a (sub)key iff some id in `partitionIds` has `id % N == n`. A stale id (owner's scope closed, id left behind) makes an executor claim a key whose surviving owners are not its rules — wasted requests (the firing site re-checks per-LMV, never unsound — [I-70](#i-70)). A missing id (owner installed, id absent) drops a rule's keys from every executor's slice and loses firings, and simultaneously breaks the D-105 prune (the owner's scope is invisible to the comparability test). One container means the three consumers can never disagree about who owns a key.

**How to spot.** A split run (`N > 1`) that proves fewer theorems than `N == 1` (a rule's keys fell out of every slice); an owner insert site that computes a composite differently from `makePartitionId(NameMap::encode(expandedOriginal), scopeVid)`; any reintroduction of a side container that shadows the set.

**How to fix on violation.** Route the insert/erase through the single `partitionIds` site with the canonical `makePartitionId` composite — never special-case it away, never add a parallel record.

**See also.** [D-119](40_decisions.md#d-119), `D-133` (40_decisions.md), [I-49](#i-49) (drop a key when its owner-set empties), [I-70](#i-70) (request prune is a sound over-approximation — a stale id is wasteful, not unsound).

---

<a id="i-78"></a>
## I-78 contradiction discharge fires at most once per LB per step, gated on `isActive`

**Scope.** Prover. `ExpressionAnalyzer::dischargeContradiction` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)) and its single call site in `standardProcessing`.

**Rule.** `dischargeContradiction` returns immediately when `!memoryBlock.isActive`, and after firing any one of its three reactions (incubator / CE / vacuous) it sets `isActive = false` and returns — at most one reaction per call. `standardProcessing` runs in BOTH phase 1 (pre-burst) and phase 3 (post-burst), and `proveKernel` does not re-filter the phase-3 sweep by `isActive` (an LB closed in phase 1 is still swept in phase 3 over the same `active` set). The leading `!isActive` early return is what makes that second pass a no-op; without it the vacuous-truth branch would deposit the head twice and record a duplicate origin row, and the incubator branch would re-enter with a cleared `contradictionTheoremId`. The early return is contracted control flow (an inactive LB has no contradiction to discharge), NOT banned defensive programming ([I-19](#i-19)).

**Why.** The sweep replaced the per-insertion Site-G reactions; a whole-`intEncodedStatements` re-scan each step would otherwise re-detect the same contradiction every phase, and every later step the LB remained in the active set.

**How to spot.** Duplicate `contradiction` / `vacuous truth` rows for one LB in the processed proof graph; a verifier failure on doubled vacuous-truth heads; or a theorem broadcast twice from one `__contradiction__` LB.

**How to fix on violation.** Restore the leading `if (!memoryBlock.isActive) return;` and the per-reaction `return`; never let the sweep continue after a reaction (the `return` also guards against iterating an `intEncodedStatements` vector the vacuous branch's `addStatement` may have reallocated).

**See also.** [D-122](40_decisions.md#d-122), [I-60](#i-60) (`addExprToMemoryBlock` is a flat insertion routine — why the reactions left it), [I-66](#i-66) (deactivation deferred to post-burst absorb).

---

<a id="i-79"></a>
## I-79 the request-generation u_ literal prune is a sound over-approximation of the firing gate — it may never drop a request that could fire

**Scope.** Prover request generation. `ExpressionAnalyzer::ownerSetUSatisfied` (predicate) and its call sites — the same owner-set match sites as the D-105 scope prune: `preEvaluateFromEncoded` (the merge step), `filterIntEncodedStatements` (both flag settings), the grow-DFS and the seed inside `generateEncodedRequestsStatic` (every obligatory-stump length), and `makeMandatoryEncodedStatementLists2Static`. The insert-side producer is `recordUSignature`, called at every owner insert in `addToHashMemory` and `makeNormalizedKeysForAdmission`. The firing-side mirror is `checkLocalEncodedMemoryStatic`'s `encodedMap` (`ignoreU=true`) lookup.

**Rule.** At every owner-set match site a request is pruned only when, in addition to failing D-105/partition, **no owner's cached u_ signature is satisfiable** by the request's flattened `argFullId`. `ownerSetUSatisfied` keeps the request when `OwnerSet::hasLooseOwner` (some owner has no u_ constraint, or a literal was not interned at insert), when `OwnerSet::uSignatures` is empty, or when any signature's every `(slot, id)` pair equals the request's `argFullId[slot]`. The predicate must stay implied by "the request can fire" — a runtime optimisation, never a soundness gate. The firing gate enforces the exact u_ literal match independently (the `encodedMap` key pins each unchangeable arg's literal), so the prune is forbidden from changing results — `files/theorems/theorems.txt` must be byte-identical with the prune active or stripped.

**Why it is sound.** A request fires owner R only if R's `encodedMap` key (`ignoreU=true`) equals the request's `WithMap` key, which at every unchangeable slot requires `request.argFullId[slot] == NameMap::encode(R.arg[1])` — exactly the `(slot, id)` pair `recordUSignature` stored for R (`argFullId == encode(arg[1])`, see `encodeExpression`). Request premises and stored (sub)keys are both name-sorted, so the linear arg slots align. Hence a request that can fire some owner satisfies that owner's stored signature; `ownerSetUSatisfied` returns `false` only when no owner can be satisfied — i.e. only for non-firing requests. The conservative paths (loose owner, un-interned literal → loose, empty signatures) only ever return `true`, so they never drop a firing request.

**Why the read mints nothing and the wipe leaves stale entries.** `recordUSignature` reads literals via the non-minting `NameMap::lookup` (an un-interned literal flags the key loose rather than recording an incomplete signature), so building the cache changes no id-assignment order and the run stays byte-identical. `ownerSetUSatisfied` reads only cached ids + `argFullId` (no `encode`), so it is safe on the shared read-only LB the LB-split executors run in parallel. Unlike `owners`/`partitionIds` ([I-80](#i-80)) the signatures are deliberately NOT maintained on `Memory::wipeSubtree` — a stale signature or stale `hasLooseOwner` only ever makes the prune *weaker* (keeps more), never unsound, so leaving them is a zero-cost simplification.

**How to spot.** `theorems.txt` differs with the prune wired vs. the 2/3 (record-only) commit — a real soundness break (a firing request was dropped); or `test_memory.cpp`'s `owner_set_u_satisfied_*` tests fail.

**How to fix on violation.** Never tighten `ownerSetUSatisfied` past a necessary condition for firing. If a real firing is lost, the signature recorded a literal the firing gate does not require (a `recordUSignature` slot/literal-extraction bug) — fix the producer to mirror `encodeExpression`/`makeIntNormalizedKey(ignoreU=true)` exactly, or widen the conservative loose path; never weaken the firing gate to match.

**See also.** [D-120](40_decisions.md#d-120), [D-105](40_decisions.md#d-105) (sibling scope prune — same "never drop what could fire" discipline), [I-70](#i-70) (the scope-prune over-approximation), [I-49](#i-49) (owner-set lifecycle).

---

<a id="i-81"></a>
## I-81 the reverse direction of a two-directional theorem is proved through the normal engine, never injected as an unproven registry row

**Scope.** Conjecturer → prover. `conj::mergeMirrorConjecturesIntoPool` (folds reverse-direction mirrors into `conjectures.txt`) and the absence of any `globalTheoremList.emplace_back(..., "mirrored statement",...)` writer in `prover.cpp`.

**Rule.** When a theorem has a distinct reverse direction (the head-swapped mirror), that direction enters the prove pool as an ordinary conjecture, passes through the counterexample filter, and is proved (or refuted) by the same machinery as any conjecture. The prover never registers a theorem it did not derive — there is no proof-graph row whose method asserts a result without a proof chapter. A reverse direction that cannot be proved is simply absent (surfaced as a missing theorem), never fabricated.

**Why.** The retired `mirrored statement` step asserted the converse of a functional relation, which is not a logical equivalence; stamping it "proved" was unsound. Routing both directions through CE + real proof restores the property that every registry row is backed by a derivation.

**How to spot.** A `mirrored statement` / `mirrored from` token anywhere in `files/processed_proof_graph/` or `globalTheoremList`; or a `globalTheoremList.emplace_back(reshuffledMirrored,...)` reintroduced in the prover.

**See also.** [D-112](40_decisions.md#d-112), [I-9](#i-9) (the equality mirror, which IS a sound equivalence and is kept), [I-44](#i-44) (origin map is documentation, not a proof input).

---

<a id="i-82"></a>
## I-82 each CE-filter conjecture runs on a private single-use clone LB; the only cross-thread write is its disjoint `contradictionTable` slot

**Scope.** Prover / CE filter. The `filterConjecturesWithCE` worker pool ([`filter.cpp`](../GL_Quick_VS/GL_Quick/src/filter.cpp)) and `Memory::cloneFactsTemplate`.

**Rule.** Each conjecture is processed by exactly one worker thread on its **own** `Memory` clone (its own `nameMap` / `keyArena` / mail / `intEncodedStatements`), produced by `cloneFactsTemplate` from a shared read-only facts template. Across the pool the ONLY write to shared `ExpressionAnalyzer` state is `contradictionTable[memoryBlock.contradictionIndex].successful = true`, and the index is unique per conjecture (handed out by the atomic queue counter), so the slots are disjoint; `contradictionTable` is sized before the pool and never resized while it runs. A CE clone is never `primedForContradiction` / `isPartOfRecursion`, so `dischargeContradiction`'s shared-write branches (`updateGlobalDirect`, `inductionMemoryBlocks`) never fire for it. The result is therefore independent of worker count and scheduling — `logicalCores` 1 and N produce the identical survivor set.

**Skeleton-interner caveat (race fixed 2026-06-27, [D-186](40_decisions.md#d-186)).** Each worker also calls `lb->setExprKey(std::to_string(i))`, which mints the conjecture index into the **process-global** `skeletonInterner` — a shared, lock-free `ColdStringTable` whose write side is single-threaded by contract ([I-97](#i-97), I-83). Minting it from the worker pool was a true data race: concurrent `mint` calls corrupt the cold key store + `PagedHashIndex`, tripping `assert(lookup(k) == id && "cold-index desync: minted key not findable at its id")` (sporadic, and independent of deload because the CE LBs are fully resident — the same desync family as the `ColdHashSet::indexPlace: duplicate id` abort). **Fix:** `filterConjecturesWithCE` PRE-INTERNS every `std::to_string(i)` for `i ∈ [0, conjectures.size)` single-threaded **before** spawning the pool, so each worker's `setExprKey` takes the lookup-only (read) path — concurrent reads on the now-unchanging table are safe. With that, the "only shared write is `contradictionTable`" rule holds again.

**Why.** The per-conjecture single-owner model is what lets the burst write to its LB (enter `intEncodedStatements`, flip `isActive`, mint names) without the I-66 read-only discipline — that discipline exists only for LBs SPLIT across threads, which a CE clone never is (`splitCount = 1`). Sharing a clone, or overlapping `contradictionIndex` across conjectures, would reintroduce a data race.

**How to spot.** A CE run whose survivor set differs between `logicalCores = 1` and `logicalCores = N`, or run-to-run; a crash / corruption under the CE pool; two conjectures handed the same `contradictionIndex`; a `cold-index desync: minted key not findable at its id` (or `indexPlace: duplicate id`) abort during the CE pool — the signature of a process-global interner minted from a worker.

**How to fix on violation.** Keep one clone per conjecture and one `contradictionIndex` per conjecture; never let two workers share a clone; route any new shared-state write out of the per-conjecture run. For an unavoidable process-global interner key (the `setExprKey` index), pre-intern every key single-threaded before the pool so the workers only read it.

**See also.** [D-118](40_decisions.md#d-118), [I-66](#i-66) (phase-2 read-only, which CE clones are exempt from by being single-owner), [I-28](#i-28) (no cross-LB writes in the parallel phase).

---

<a id="i-83"></a>
## I-83 phase-2 parallel parts write only per-slot scratch and their own sealed page set — never shared LB state

**Scope.** Prover. The `performElem2` call tree (`generateEncodedRequestsStatic` / `...Pairs` / `...CE`, `makeMandatory*`, `preEvaluateFromEncoded`, `checkLocalEncodedMemoryStatic`) run by the flat phase-2 executor pool.

**Rule.** An LB's N parts run concurrently on one shared `Memory`. Each part may WRITE only: its own per-task `SealedPageSet` (the record chain its firings append to plus the sealed strings/spans they reference, `I-135`) and its own per-slot scratch arenas (`scratchArenas` / `genScratchArenas`, released at `performElem2` exit — the request keys and request-expr copies ride the gen arena, `I-130`). (The former per-thread `reqBuf` is gone — the streaming `BurstSink` checks each request inline and buffers nothing, [D-109](40_decisions.md#d-109).) It must NOT write any shared per-LB state — no `NameMap::encode` (mint), no insert into `intEncodedStatements` / `overallHashMemory` / any `body` / `memoryBlock` container, no `isActive` flip. Reads of shared state use non-minting accessors (`NameMap::lookup`). All shared mutation is deferred to the single-threaded per-LB finalize (`applyFiringRecords` in `performElemPhase2`) and phase 3.

**Why.** Two concurrent parts writing one shared container is a data race → corruption → non-deterministic output and crashes. The read-only refactor ([D-116](40_decisions.md#d-116)) converted the `NameMap` mints to `lookup` but MISSED `Memory::keyArena.store` (request keys) — a shared write that raced and segfaulted Gauss main under heavy split ([D-125](40_decisions.md#d-125)). This invariant is the standing guard against the next such miss.

**How to spot.** In the `performElem2` call tree, any `body.` / `memoryBlock.` write — `grep -nE '(body|memoryBlock)\.[A-Za-z_]+\.(store|push_back|insert|emplace)|nameMap\.encode'` over the phase-2 functions. A reproducible crash / corruption that scales with split depth (worse at low `maxNumberHashRequests`); verifier check count that varies with the cap or `split_growth_factor`; non-determinism between `logicalCores` 1 and N.

**How to fix on violation.** Move the written state to per-slot scratch (the worker's `genScratchArenas` arena) or the task's own `SealedPageSet`, or defer the write to the single-threaded finalize. Never add a lock on the shared container in the hot path — it serializes the split.

**Code.** `prover.cpp::performElem2` + its call tree; the per-slot scratch registries (`scratch_arena.hpp`) + the per-task `SealedPageSet` (`sealed_pages.hpp`). The finalize writer: `applyFiringRecords` (`performElemPhase2`).

**See also.** [D-125](40_decisions.md#d-125), [D-116](40_decisions.md#d-116), [D-117](40_decisions.md#d-117), [I-28](#i-28), [I-66](#i-66).

---

<a id="i-73"></a>
## I-73 the submatch cap has zero effect in CE-filter mode

**Scope.** Prover / CE filter. `BurstSink::canAccept` and the request generator `generateEncodedRequestsStatic` at an obligatory-stump length of 0.

**Rule.** While `ceFilteringActive`, the per-part submatch cap (`maxNumberHashRequests`) does NOT apply — `BurstSink::canAccept` returns true regardless of `g_growthMatchCount`. A CE burst runs its full enumeration; the only thing that halts it is the contradiction early-exit (the external `stop` flag set by `burstDeactivates`).

**Why.** The CE filter decides whether a conjecture's negation contradicts the fact base. It is one un-split LB per conjecture (no parts to bound), and a completeness check. Truncating its burst on a count could drop the refuting head, so the filter would wrongly KEEP a conjecture it should discard — a completeness failure that floods the main prover. Confirmed: capping CE kept 936 → 909 conjectures and triggered a main-prover explosion; uncapping cut survivors to 230 and removed the blow-up.

**How to spot.** A conjecture survives CE that the unsplit reference filters out; CE survivor counts that vary with `maxNumberHashRequests`.

**How to fix on violation.** Keep the `ceFilteringActive` short-circuit first in `canAccept`; never add a count-based stop to the CE generator's grow loop.

**Code.** `prover.hpp::BurstSink::canAccept` (the `self->ceFilteringActive ||` short-circuit); `memory.cpp::generateEncodedRequestsStatic` (its grow-DFS consults `canAccept`, so the cap bypass reaches the search; the empty-stump path asserts the LB is unsplit, which is why its submatch tally may be read by nobody).

**See also.** [D-109](40_decisions.md#d-109), [D-118](40_decisions.md#d-118).

---

<a id="i-76"></a>
## I-76 the burst early-exit is honored only when the LB runs as a SINGLE part this burst (`g_isMultiPart == false`); multi-part LBs run to completion (determinism)

**Scope.** Prover. `BurstSink::canAccept` (the `stop` check) and `BurstSink::consume` (the `burstDeactivates` → `stop` set) in `prover.hpp`, driven by `performElem2`; the per-burst thread-local `gl::g_isMultiPart` (`memory.hpp`/`memory.cpp`), set by `performElem2` from its `partCount` argument (`= partCount > 1`).

**Rule.** The phase-2 burst early-exit ([D-121](40_decisions.md#d-121)) is gated on the per-burst multi-part signal `g_isMultiPart`: honored ONLY when the LB runs as ONE part this burst (`g_isMultiPart == false`). When it runs as several parts (`g_isMultiPart == true`) neither the SET (`consume` skips the `burstDeactivates` loop — no `stop->store`, no early `return false`) nor the BAIL (`canAccept` skips the `stop->load` check) fires; every part runs to completion.

The gate is `g_isMultiPart`, NOT `g_splitCount`, because the split axis changed ([D-201](40_decisions.md#d-201)): a straggler is split into `logicalCores` EXPRESSION buckets that run at `g_splitCount == 1` (so `partitionAccepts` accepts every rule — the rule dimension is off). Keying the early-exit on `g_splitCount>1` would leave it ON across those bucket parts and reintroduce the race below; keying on `g_isMultiPart` (true iff `partCount > 1`, in EITHER dimension) covers them. `performElem2` asserts a stump bucket rides a multi-part LB (`splitStump.count == 0 || partCount > 1`) rather than trusting it. `g_splitCount` still gates only `partitionAccepts` and must stay `1` for a whole-LB expression split.

**Why.** When an LB runs as several parts they are siblings on one shared LB. One part firing a deactivating head and setting the shared `stop`, with the other parts bailing in `canAccept`, is a **thread-timing race**: a sibling that has not yet generated its (useful) firing is cut, so the firing set — and the proved-theorem set — depends on scheduling. That is non-deterministic output, which breaks GL's core determinism claim. It surfaced as a heisenbug — the hashburst dump's own mutex-serialized latency reliably flipped a lost Peano `1·x=x`-mirror theorem (`(>[v1](in[v1,N])(>[v2](in3[i1,v2,v1,*])(=[v1,v2])))`) from absent at split back to present — and was confirmed: gating the early-exit off multi-part makes two full split runs byte-identical (`proved_theorems` + `global_theorem_list`), recovering the dropped Peano and Gauss theorems with the verifier airtight. When the LB runs as one part there are no siblings (a single part bailing on its own deactivating head is deterministic), so the early-exit stays valid — and load-bearing for the unsplit incubator and the CE filter ([I-73](#i-73), which still halts on this stop because CE runs single-part, `g_isMultiPart == false`).

**How to spot.** Two same-config runs differing in `proved_theorems` / `global_theorem_list`; a theorem present unsplit (`disable_lb_split`) but absent when the LB is split; `canAccept` or `consume` honoring `stop` without a `g_isMultiPart` guard; the early-exit gate reading `g_splitCount` again (it must read `g_isMultiPart`).

**How to fix on violation.** Keep both gates: `canAccept` returns true when `g_isMultiPart` regardless of `stop`; `consume`'s `burstDeactivates` loop is wrapped in `if (!g_isMultiPart)`. The skipped requests at multi-part are wasted-once-doomed, so single-part and multi-part produce the same firing set — minus the race.

**Code.** `prover.hpp::BurstSink::canAccept` (the `g_isMultiPart ||` guard), `prover.hpp::BurstSink::consume` (the `if (!g_isMultiPart)` wrap), `prover.cpp::performElem2` (`g_isMultiPart = (partCount > 1)`).

**See also.** [D-121](40_decisions.md#d-121), [D-201](40_decisions.md#d-201), [I-66](#i-66), [I-73](#i-73), [I-83](#i-83).

---

<a id="i-75"></a>
## I-75 the phase-2 pass loop runs at most TWO rounds — producer, then buckets; there is no cap-escalation

**Scope.** Prover. `proveKernel`'s phase-2 pass loop and its classify step.

**Rule.** The split is decided BEFORE the iteration (the end-of-iteration stats pass, [D-201](40_decisions.md#d-201)), so the pass loop no longer escalates on a cap-hit. It runs at most TWO rounds, asserted at `passNo <= 2`:

- **Round 1.** Each active main-path LB dispatches ONE task: a straggler (`numberOfParts > 1`) dispatches a `produceOnly` PRODUCER (which runs `produceExpressionStumps` whole-LB, fires nothing); every other LB dispatches one unsplit burst part, kept.
- **Round 2.** Each producer's stumps are dealt into `min(nStumps, logicalCores)` buckets in the classify step, and one bucket part per bucket is requeued; these run and are kept. (A producer that yielded no stumps requeues one unsplit burst instead.)

No burst is ever discarded (no cap, no truncation); every part is kept and merged. The finalize applies the kept parts and does NOT decide the split (that is the stats pass).

**Superseded.** This entry formerly bounded a THREE-level cap-escalation ladder (`passNo <= 3`: unsplit → rule split → stump split) driven by `adaptiveSplitDecision`. Both are gone with [D-111](40_decisions.md#d-111): the trigger is preemptive, the fan is one expression dimension, and the loop is producer-then-buckets.

**Why.** Termination plus a bounded cost. A producer's buckets requeue exactly once (they never produce stumps of their own — the classify step only deals stumps from `produceOnly` tasks), so the loop cannot grow past round 2. No discard-and-redo means no wasted capped pass.

**How to spot.** The `passNo <= 2` assert firing; a hang where `nextTasks` never empties; a bucket part (`t.stump.count > 0`) being treated as a producer; a `produceOnly` task appearing in round 2.

**How to fix on violation.** Only `produceOnly` tasks requeue (into bucket parts); a bucket or unsplit burst part is always kept, never requeued.

**Code.** `prover.cpp::proveKernel` (the pass loop's `passNo <= 2` assert; the classify step's producer-deal vs keep).

**See also.** [D-201](40_decisions.md#d-201), [D-111](40_decisions.md#d-111), [I-74](#i-74), [I-77](#i-77).

---

<a id="i-74"></a>
## I-74 RETIRED ([D-201](40_decisions.md#d-201)) — there is no discarded / re-run burst

**Scope.** Prover (historical). `proveKernel`'s former escalate-discard-redo path.

**Rule.** RETIRED. The mid-burst cap and its same-iteration discard-and-redo are gone ([D-111](40_decisions.md#d-111) superseded): every main-path burst runs to completion and is kept, and a heavy LB is split PREEMPTIVELY next iteration (no truncated burst to discard). The property this invariant guarded — that a discarded burst left no LB residue — is therefore vacuous. What survives, and matters, is that phase 2 (`performElem2`) is **read-only on the LB** ([D-116](40_decisions.md#d-116)); that guarantee is now carried by [I-66](#i-66) / [I-83](#i-83). The producer task (`produceExpressionStumps`) is likewise read-only on the LB.

**See also.** [D-201](40_decisions.md#d-201), [D-111](40_decisions.md#d-111), [D-116](40_decisions.md#d-116), [I-66](#i-66), [I-83](#i-83), [I-75](#i-75).

---

<a id="i-84"></a>
## I-84 the int16 vectors are the ONLY stored statement form; string structs are transient, decoded at boundaries

**Scope.** Prover / Memory. `Memory::intEncodedStatements`, `intLocalEncodedStatements`, `intLocalEncodedStatementsDelta`, `intExternalStatements`; the converter pair `encodeExpression` / `decodeExpression` (`memory.hpp`).

**Rule.** A statement is stored exactly once per registry, as an `IntEncodedExpr` row. No `Memory` member may hold a persistent `vector<EncodedExpression>` mirror of a statement registry. String-form `EncodedExpression` values are transient: constructed at insert sites (then discarded after `encodeExpression`), or reconstructed on demand via `decodeExpression` at the boundaries that genuinely need text — the hashburst diagnostic dump, `fillMailOut` (mail crosses LBs and `NameMap` ids are LB-local), the visualizer, and equivalence-class rewriting. The statement indexes (`intLocalEncodedStatementsSet`, `intStatementLevelsMap`) are packed-key (`I-86`), so no index operation needs a string key. Three discipline clauses: (1) read paths use the non-minting `NameMap::lookup`, never `encode` — a lookup miss means "no stored row can match"; (2) decoded references are copied into locals before any call that may mint ([I-3](#i-3)); (3) rows are matched by the `(originalId, validityId)` pair — full identity, because every stored row is a canonical-pipeline encoding whose other fields derive from those two strings — and never sorted or compared by raw id value (mint order is not lexicographic).

**Why.** One stored form kills the lockstep dual-write hazard (every push/erase previously had to touch two containers; test-side bypasses forced defensive size guards), removes the fattest per-statement storage in `Memory` (several heap strings + a vector-of-vector-of-strings per row vs one flat 176-byte struct) on the ASIC 0.1 static-memory path, and makes the hot path's int-only consumption (static request generation, firing, `intKnownStatements`) the same data the rest of the prover uses. Losslessness is what makes it sound: `originalId` interns the whole original text, `validityId` the scope name, and `EncodedExpression(original, validityName)` re-derives every other field; `encodeExpression` asserts arity fits `MAX_ARITY`.

**How to spot.** A new `vector<EncodedExpression>` member appearing on `Memory`; a reader calling `nameMap.encode` on a probe string; a decode reference held across a minting call; output order changing because something sorted by int id.

**How to fix on violation.** Store the int row and decode at the boundary instead; switch probe-side `encode` to `lookup`; copy decoded strings before mints; sort by decoded strings where order feeds emitted artifacts.

**Code.** `memory.hpp::encodeExpression` / `memory.hpp::decodeExpression`; push sites in `prover.cpp::addEquality` / `addExprToMemoryBlock` / `prehandleAnchor` and `prover.hpp::addStatement` / `applyEquivalenceClass`; boundary decodes in `prover.hpp::fillMailOut` / `dischargeToBeProved` / `dischargeContradiction` / `applyEquiClasses`, `infra/hashburst_dump.cpp` section writers, `visualizer.cpp` `containsEncoded`.

**See also.** [D-127](40_decisions.md#d-127), [I-3](#i-3), [I-58](#i-58).

---

<a id="i-85"></a>
## I-85 — `intKnownStatements` bits are the membership truth: `known`-reads vs `registered`-reads, OR-only upsert

**Scope.** Prover / Memory. `Memory::intKnownStatements`, `StatementFlags::registered` / `StatementFlags::known`, the write door `upsertStatementKey` (`memory.hpp`).

**Rule.** `intKnownStatements` carries TWO memberships in one packed-key map, distinguished by `StatementFlags` bits: `registered` (the statement passed an add-path registration door) and `known` (the statement entered the level registry — the Site F dedup record). Every read tests the bit its contract names, never bare map presence: the Site F ancestor scans (`addExprToMemoryBlockKernel`, `checkLocalEncodedMemoryStatic`), the contradiction negation scans (`dischargeContradiction`, `burstDeactivates`), and the burst dependency skip (`BurstSink::consume`) test `known`; the statement-registration gates (`addEquality` / `addNegatedEquality` dedup + mirror asserts, the integration-preparation gates, `fillMailOut`'s transitive-walk statement skip, `checkForEquivalence` / `disintegrateExpr2` variant checks, the recursion-mail / typing-goal / CE-head / anchor gates) test `registered`. Bits are OR-only through `upsertStatementKey`; the only bit-clearing writers are whole-entry erases (`wipeSubtree`, `eradicateImplicationFromLB`, `resetResentExpressionRegistries`) and the CE teardown's `registered`-membership reset (`releaseCEBatchMemory`).

**Why.** The two memberships were historically two containers (`wholeExpressions` / `intKnownStatements`) with deliberately different content: `addStatement` registers unconditionally but admits to the level registry only behind its iteration-cap / secondary-variable-count / equivalence-filter gates; the equivalence-class commit grants `known` only; the compressor rule load grants `registered` only. A bare-presence read widens Site F (cap-filtered statements would suppress kernel entry for fresh derivations) or flips registration gates (equi-rewrite rows would suppress integration preparation and mail-walk shipment), changing proof output. The bits keep each gate's answer exactly while one container holds both records.

**How to spot.** A `count` or bare `find!= end` on `intKnownStatements` inside a gate; a direct `insert({key, flags})` bypassing `upsertStatementKey`; any code clearing a bit outside the named erase/teardown sites.

**How to fix on violation.** Test the bit the gate's contract names; route the write through `upsertStatementKey`; if a clear seems needed, the operation is a teardown — erase the entry through the named teardown paths instead.

**Code.** `memory.hpp::upsertStatementKey` (write door), `memory.hpp::StatementFlags`; `known` reads in `prover.cpp::addExprToMemoryBlockKernel` (Site F), `memory.cpp::checkLocalEncodedMemoryStatic`, `prover.hpp::dischargeContradiction` / `burstDeactivates` / `BurstSink::consume`; the CE `registered` reset in `filter.cpp::releaseCEBatchMemory`.

**See also.** [D-128](40_decisions.md#d-128), [I-58](#i-58), [I-84](#i-84).

---

<a id="i-86"></a>
## I-86 — the statement indexes are packed-key; string probes are non-minting; the dump derives its section by decode + lex-sort


**Scope.** Prover / Memory. `Memory::intLocalEncodedStatementsSet` (a `ColdHashSet<PodKeyStore<int32_t>>` since Batch 1), `Memory::intStatementLevelsMap` (a `ColdSetMap<PodKeyStore<int32_t>, int>` — the cold-map set form — since Batch 2; `lookupStatementLevels` returns the cold key id, the level run reconstructed by `coldIntSetAt`), the non-minting probes `isLocalEncodedStatement` / `lookupStatementLevels` (`memory.hpp`), the dump section writer `writeStatementLevelsMap` (`infra/hashburst_dump.cpp`).

**Rule.** Both statement indexes are keyed by `packStatementKey(originalId, validityId)` — the `intKnownStatements` key, bijective with the former `(original, validityName)` string-pair identity because the per-LB `NameMap` interns each string exactly once. Four clauses:

1. **Probes with ids in hand pack directly.** A site holding an `IntEncodedExpr` row (request rows in `checkLocalEncodedMemoryStatic`, registry rows in `applyEquiClasses`, delta rows in `fillMailOut` / `dischargeToBeProved`, `ieStmt` in `addStatement`) probes with `packStatementKey(row.originalId, row.validityId)` — no decode, no string-key construction.
2. **Probes with only strings in hand go through the non-minting helpers** (`isLocalEncodedStatement`, `lookupStatementLevels`) — `NameMap::lookup` with the id-0 miss sentinel, exact because every index entry interned both strings at its insert site. Never `encode` on a probe-only path: minting there shifts the LB's id-assignment order and breaks dump byte-identity. The two deliberately-minting erase helpers (`resetResentExpressionRegistries`, `eradicateImplicationFromLB`) keep `encode` — they minted before the re-key too, documented inline.
3. **No reader iterates either index.** The only iterations are `wipeSubtree`'s erase-if sweep (order-independent; closed-bitmap membership on the key's low 16 bits is exactly the forest predicate, the [D-128](40_decisions.md#d-128) argument carried over by [I-139](#i-139)) and the dump writer.
4. **The dump section is derived, byte-identical.** `writeStatementLevelsMap` unpacks each key, decodes via `idToName`, lex-sorts rows by the decoded `(original, validityName)` pair, and prints under the unchanged section label `-- statementLevelsMap (N):` — reproducing the former `std::map<EncodedExpression, std::set<int>>` iteration order exactly (keys are unique pairs; no ties). The level payload is reconstructed as an ordered `std::set<int>` (`coldIntSetAt` over the sorted cold run) and prints ascending. Rule 14 applies to the section format; only the data source was retargeted (user-authorized, the [D-127](40_decisions.md#d-127)/[D-128](40_decisions.md#d-128) consent shape).

**Why.** The D-29 locality gate and the per-request combined-levels union sat in the burst hot path, paying a decode plus (for the gate) the parsing `EncodedExpression` constructor per premise per firing; the packed probe is O(1) integer hashing with zero string work. Every write site already computes the packed key for its adjacent `upsertStatementKey` call, so maintaining the indexes costs no extra encodes.

**How to spot.** A probe building an `EncodedExpression` just to test index membership; `nameMap.encode` feeding an index probe on a path that never minted before; a new iteration over either index whose order reaches any output; a dump edit that sorts by raw key value instead of decoded strings.

**How to fix on violation.** Pack from the row's own ids, or route through the non-minting helpers; sort by decoded strings wherever order feeds an emitted artifact ([I-84](#i-84) clause 3).

**Code.** Gate + levels union in `memory.cpp::checkLocalEncodedMemoryStatic`; write sites in `prover.cpp::addEquality` / `addNegatedEquality` / `addExprToMemoryBlock` (status-4) / `prehandleAnchor` and `prover.hpp::addStatement` / `applyEquivalenceClass`; sweeps in `memory.cpp::Memory::wipeSubtree` and `prover.hpp::cleanUpExpressions`; the dump writer in `infra/hashburst_dump.cpp::writeStatementLevelsMap`.

**See also.** `D-129` (40_decisions.md), [I-84](#i-84), [I-85](#i-85), [I-7](#i-7) (the D-29 gate's home).

---

<a id="i-87"></a>
## I-87 — the goal registry is packed-key; probes are non-minting; order-sensitive walks iterate the decoded lex-sorted snapshot


**Scope.** Prover / Memory. `Memory::intToBeProved` (a `ColdSetMap<PodKeyStore<int32_t>, int>` — the cold-map set form — since Batch 2; the value is the auxy set, the former reserved tags member long dropped; `lookupToBeProved` returns the cold key id, the auxy run reconstructed by `coldIntSetAt`), the non-minting probe `lookupToBeProved` and the snapshot builder `decodeToBeProvedSorted` (`memory.hpp`), the dump section writer `writeToBeProved` (`infra/hashburst_dump.cpp`).

**Rule.** The goal registry is keyed by `packStatementKey(originalId, validityId)` — bijective with the former `(original, validityName)` string-pair identity, which is exactly what `EncodedExpression::operator<` compared. The value is the auxy set (`std::set<int>`, induction-discharge bookkeeping); the former reserved tags member was dropped as dead (never written anywhere — every entry carried `tags={}`), and the dump row prints a literal `tags={}` to keep the Rule-14 format byte-identical. Four clauses:

1. **Probes with ids in hand pack directly.** `dischargeToBeProved`'s three find sites probe with the delta row's `packStatementKey(ie.originalId, ie.validityId)`; the `addExprToMemoryBlock` status-2 insert reuses the ids encoded at function entry. The `effectiveValidity == "main"` gates compare `ie.validityId == NameMap::MAIN_ID` — the same predicate, id-side.
2. **Probes with only strings in hand go through the non-minting `lookupToBeProved`** (`NameMap::lookup`, id-0 miss sentinel — exact because every goal interned both names at its insert site). This includes `burstDeactivates`' sole-goal probe, which runs in the phase-2 read-only parallel context where minting would also be a data race, and `updateGlobal`'s induction-promotion probe. Never `encode` on a probe-only path.
3. **Order-sensitive walks iterate `decodeToBeProvedSorted`** — owned decoded copies (the consumers mint downstream) lex-sorted on `(original, validityName)`, reproducing the former `std::map` iteration order exactly. The walks: `sanitizeToBeProved`'s staging pass ([I-45](#i-45) — rewrite collapse depends on processing order), the post-absorb `checkNecessityForEquality` sweep, the vacuous-truth first-main-goal pick, and the dump writer. Order-free reads (the [I-48](#i-48) main-scope surveys in `deactivateRecursively` / `deactivateUnnecessary`, `wipeSubtree`'s closed-bitmap erase-if) iterate the live map directly; the main-scope test is `(key & 0xFFFF) == NameMap::MAIN_ID`.
4. **The dump section is derived, byte-identical.** `writeToBeProved` prints the snapshot rows under the unchanged section label `-- toBeProved (N):` with the unchanged per-row format; the header's `toBeProved=` count reads the packed registry's size. Rule 14 applies to the format; only the data source was retargeted (the [D-127](40_decisions.md#d-127)/[D-129](40_decisions.md#d-129) consent shape).

The `sanitizeToBeProved` re-key reuses the old key's validity id verbatim and encodes only the rewritten original — [I-45](#i-45)'s namespace-preservation clause enforced structurally (a rewrite cannot mint a different scope).

**Why.** The discharge finds and the `burstDeactivates` probe each ran the parsing `EncodedExpression` constructor per call just to build a lookup key; the [I-48](#i-48) surveys string-compared `validityName` per entry per deactivation pass. The packed forms are O(1) integer work with the ids already in hand. ASIC 0.1 direction: goal rows shed their tree-map string keys.

**How to spot.** A probe constructing an `EncodedExpression` just to test goal membership; `nameMap.encode` feeding a goal probe on a path that never minted before; a new walk over `intToBeProved` whose processing order reaches any output without going through `decodeToBeProvedSorted`; a dump edit sorting by raw key value.

**How to fix on violation.** Pack from the row's own ids or route through `lookupToBeProved`; route any order-sensitive walk through `decodeToBeProvedSorted` ([I-84](#i-84) clause 3: mint order is not lex order).

**Code.** Insert + post-absorb sweep in `prover.cpp::addExprToMemoryBlock`; discharge finds in `prover.hpp::dischargeToBeProved`; sole-goal probe in `prover.hpp::burstDeactivates`; promotion erase in `prover.cpp::updateGlobal`; re-key in `prover.hpp::sanitizeToBeProved`; surveys in `prover.cpp::deactivateRecursively` / `deactivateUnnecessary`; sweep in `memory.cpp::Memory::wipeSubtree`; state-1 erase in `prover.cpp::removeExpressionFromMemoryBlock`; the dump writer in `infra/hashburst_dump.cpp::writeToBeProved`.

**See also.** `D-130` (40_decisions.md), [I-84](#i-84), [I-86](#i-86), [I-45](#i-45), [I-48](#i-48).

---

<a id="i-88"></a>
## I-88 — equivalence-class state is id-form; canonical selection is decoded-lex storage order, never id order

**Scope.** Prover / equivalence classes — `EquivalenceClass`, `Memory::equivalenceClassesMap`, `Memory::intWeakVariables`, `Memory::eqClassSttmntIndexMapMap`, `Memory::changedClassesThisStep`, the `EqClassNameCaches` memoization, and every reader listed under **Code** below.

**The invariant.**

1. **`memberIds` is the class.** Members are NameMap ids stored sorted by DECODED name — the first member of a tier (`int_lev_*` over `it_*_lev_*`; `chooseCanonical` adds the normal tier above both) IS that tier's lex-min, so canonical selection is a first-of-tier scan with zero sorting and zero id-order dependence ([I-84](#i-84)). Writers keep the order structurally: `setMembersFromNames` pushes in `std::set` iteration order; `unionMemberIdsByName` merges by decoded-name compare. `intEqualityLevelsMap` keys are `packEqPairKey` unordered id pairs — id normalization inside the key is identity-only and never ordering. `equalityOriginMap` stays string-keyed (the originMap batch owns its migration).
2. **Probes are non-minting; misses are definitive.** Member/argument membership goes through `NameMap::lookup`: a never-interned name cannot be a member, because every member is interned at class commit (`updateEquivalenceClasses` encodes eqArgs up front; merge unions only existing ids). Same argument covers `reduceEqClassIds`' validity lookup against the packed `intWeakVariables` and `Memory::classesAt`'s scope probe (`nullptr` = the defined no-classes state).
3. **The name caches are pure and reset only with their NameMap.** `EqClassNameCaches` memoizes `classifyName` / `scanSpecialTokens` per id — pure functions of the decoded string; never wiped on scope teardown, reset in lockstep with the `destroyGrid` `nameMap` reset (ids re-bind there). Lazy fill is a shared-state write — probes stay out of phase-2 parallel parts ([I-83](#i-83)).
4. **Order-sensitive walks decode then lex-sort.** `reactToHypo`, `applyEquiClasses`' pass-2 validity loop, and the dump sections iterate decoded-name lex-sorted snapshots of the id-keyed class store; the dump (`writeEquivalenceClassesMap`, `writeWeakVariables`) derives byte-identical output (members print in `memberIds` storage order = decoded-lex; weak rows decode + pair-sort). Rule 14 applies to the format; only data sources were retargeted (the [D-127](40_decisions.md#d-127)/[D-129](40_decisions.md#d-129)/[D-130](40_decisions.md#d-130) consent shape).

**Why.** `filterIterations` regex-scanned every class member plus the whole statement text per (statement × class) probe — the dominant string cost of the subsystem; the cached id path is integer work. ASIC 0.1 direction: class members shed heap strings. The decoded-lex storage order makes the canonical member a structural property instead of a per-call computation.

**How to spot violations.** A sort or comparison of `memberIds` by raw id value on any path that reaches a substitution, an emission order, or the dump; a class-membership probe that calls `encode` instead of `lookup`; a new walk over `equivalenceClassesMap` whose iteration order reaches output without a decoded lex-sort; a cache probe from a phase-2 parallel part; an `equalityLevelsMap`-style string container reappearing on the class.

**How to fix on violation.** Route canonical selection through `firstSpecialMemberId` / `chooseCanonical`; route membership through `lookup` + `memberIds` find; route order-sensitive walks through a decoded snapshot; route writer-side member construction through `setMembersFromNames` / `unionMemberIdsByName`.

**Code.** Struct + helpers in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp) (`EquivalenceClass`, `packEqPairKey`, `unionMemberIdsByName`, `EqClassNameCaches`, `classifyName`, `scanSpecialTokens`, `Memory::classesAt`); canonical readers in [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) (`firstSpecialMemberId`, `filterIterationsCore` + overloads, `chooseCanonical`, `reduceEqClassIds`, `canonicalizeUnderClasses`, `enumerateEqClassRewrites`, `applyEquivalenceClass`, `applyEquiClasses`, `mergeTwoEquivalenceClasses`, `updateEquivalenceClasses`); `prover.cpp` (`reactToHypo`, `updateWeakVariables`, `checkForEquivalence`, the registration site in `addEquality`'s path, the `destroyGrid` cache reset); the dump writers in `infra/hashburst_dump.cpp`.

**See also.** `D-134` (40_decisions.md), [I-84](#i-84), [I-85](#i-85), [I-86](#i-86), [I-87](#i-87), [I-83](#i-83).

---

<a id="i-89"></a>
## I-89 — admission/rejected keys are packed (templateId, validityId) in a DEDICATED template-id space; probes never mint; parallel staging rides sealed page views

**Scope.** Prover / admission + rejection — `TemplateInterner`, `mintTemplateKey` / `lookupTemplateKey` / `decodeTemplateKey`, the five maps (`admissionMap`, `admissionStatusMap`, `admissionMapIntegration`, `rejectedMap`, `rejectedMapIntegration`), `consumedAdmissionKeys` / `revisitInProgress`, the three `varsIn*Keys` caches, and every reader/writer listed in `D-132`.

**The invariant.**

1. **Two id spaces, never mixed.** The template half of every packed key comes from `Memory::templateInterner`; the validity half from `Memory::nameMap`. A template string is NEVER interned in the `NameMap` (it would mint mid-run and shift statement ids — the dump prints the full `nameMap` table and raw-id sections, so id stability is load-bearing for trace comparability). The interner resets only in lockstep with `destroyGrid`'s `nameMap` reset.
2. **Writers mint, probes look up.** Registration writes (`updateAdmissionMap`, the drains, `prepareIntegrationCore2`, the equi-class hooks' K' inserts, `updateRejectedMap(Integration)`) go through `mintTemplateKey`. Probes (`isAdmitted`, `isAdmittedIntegration`, the consumed/revisit guards, `revisitRejected(Integration)2` cohort finds, the staging-time consumed gate) go through the non-minting `lookupTemplateKey` — a lookup miss on either half is a definitive container miss, a defined result.
3. **Phase-2 parallel staging never mints.** `admissionKeysAlgebra` / `deferredIntegrationPreps` records carry their template text as SEALED PAGE VIEWS (`SealedString` into the producing task's `SealedPageSet` — [D-164](40_decisions.md#d-164); strings until the strings campaign); the single-threaded post-fixpoint drains intern through the span doors ([I-68](#i-68)/[I-69](#i-69)/[I-83](#i-83)) — no drain-side materialization; the residual edge strings live inside the callees at their compiled-definition boundaries — and `performElemPhase2` clears both staging vectors post-drain, before the page sweep. The staging-time consumed gate reads with `lookup` only.
4. **Order-sensitive walks decode then lex-sort.** The equi-class hooks (whose walk/insert order drives `revisitRejected(Integration)2` and mail emission order) iterate decoded `(template, validity)` lex-sorted snapshots — the former EWV `std::map` order exactly. Erase-only sweeps (`cleanAdmissionMap`'s [I-41](#i-41) closure, the cleanup sweeps) may walk unordered (content-deterministic). Dump sections derive byte-identical output — with ONE exception: the three `varsIn*Keys` sections print decoded names lex-sorted (a dump-order change; see `D-132`).
5. **Contracts carried over unchanged.** [I-22](#i-22) (integration templates persist; the algebra-side integration erases remain the key-form no-ops they always were), [I-37](#i-37) (rejected maps: drop + mail, never direct equi-class insert), [I-71](#i-71) (consumed/admission mutual exclusion, now on packed keys), and the three-key-shape contract (marker / u_ / bare conversions stay string operations at write/rewrite time).

**Why.** Admission probes and key bookkeeping ran on `std::map<ExpressionWithValidity>` string keys (log-N string compares per probe); packed keys are O(1) integer work. ASIC 0.1 direction: the bookkeeping maps shed their string keys while the dedicated id space keeps the statement-id stream byte-stable.

**How to spot violations.** `templateInterner.encode` on a probe path or anywhere in phase-2 parallel code; a template string fed to `nameMap.encode`; a new full-map walk whose effects reach mail/revival/dump without a decoded lex-sort; a staged record carrying a packed key; an equi-class hook inserting into a rejected map.

**How to fix on violation.** Route writes through `mintTemplateKey` at single-threaded sites, probes through `lookupTemplateKey`, order-sensitive walks through `decodeTemplateKey` + sort.

**Code.** `TemplateInterner` + helpers in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp); consumers across [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) / [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) / [`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp) per `D-132`; dump writers in `infra/hashburst_dump.cpp`.

**See also.** `D-132` (40_decisions.md), [I-68](#i-68), [I-69](#i-69), [I-71](#i-71), [I-22](#i-22), [I-37](#i-37), [I-41](#i-41), [I-83](#i-83), [I-84](#i-84), [I-99](#i-99).

---

<a id="i-90"></a>
## I-90 — admission/rejected VALUES are id-form in the int32 value space; observable orderings via stateful decoded comparators; working forms stay string

**Scope.** Prover / admission + rejection — `ValueInterner`, the id-form value structs (`AdmissionMapValue`, `RejectedMapValue`, `RejectedMapIntegrationValue`, `IntInstruction`/`IntLogicalEntity`), their comparators and find-or-emplace helpers, the template-space EWV sets, and every consumer in `D-136`.

**The invariant.**

1. **Value strings live in `Memory::valueInterner` (int32).** Never in the `NameMap` (id-shift/trace-stability) and never in the int16 `TemplateInterner` (values sit in id vectors, not packed pair keys; Gauss-scale value populations exceed int16 comfort). Encode at single-threaded write sites; non-minting `lookup` probes; the interner resets only with `destroyGrid`'s `nameMap` reset.
2. **Every observable ordering is decoded order.** The value sets (`AdmissionValueSet`, `RejectedValueSet`, `RejectedIntegrationValueSet`), the stored-instruction map (`IntegrationEntryMap`), and the payload sets (`ValueIdSet`) order through stateful comparators that hold a `const ValueInterner*` and replicate the historical string `operator<` field orders exactly — never raw id order ([I-84](#i-84)). Containers are created ONLY through their value helpers so the comparator state is always supplied — a default-constructed comparator (null interner) is a bug: the heap rejected / integration maps via the find-or-emplace `rejectedValuesAt` / `rejectedIntegrationValuesAt` / `integrationEntryAt` / `payloadAt`; the now-cold `admissionMap` ([I-99](#i-99)) via the read-snapshot `admissionRecordsAt` + the RMW `insertAdmissionValue`, which decode the run into the same comparator-bearing `AdmissionValueSet` before mutating and re-emit it sorted.
3. **Working forms stay string at the boundaries; staging rides sealed views.** The parallel staging path carries `StagedAdmissionValue` as SEALED PAGE VIEWS (sealed key elements + sorted-unique sealed remaining args — [D-164](40_decisions.md#d-164); strings until the strings campaign), and the drain (`stagedToIdValue`) materializes exactly at the intern point ([I-68](#i-68)/[I-83](#i-83)); the string `Instruction` remains the processing form (`cleanInstruction`, `prepareIntegration` flow) with `encodeInstruction`/`decodeInstruction` at the map touchpoints; rewrite/analysis sites decode to OWNED copies before any mint-capable call ([I-3](#i-3) discipline) — `isAdmitted`'s tuple loop, the recursion walk, the hooks' per-value rewrites, the revival emissions.
4. **The global `LogicalEntity` is untouched.** Only the STORED instructions inside `admissionMapIntegration` use the id-form twins; `compiledExpressions` and the disintegration machinery keep the string type.
5. **The EWV sets are template-space packed.** `admissionSetIntegration` / `triggersForAdmissionSetIntegration` members are template-population strings (marker forms, repl_ triggers) keyed by `mintTemplateKey`/`lookupTemplateKey`; the per-trigger `makeAdmissionKeys` production walk iterates a decoded lex-sorted snapshot.

**Why.** The values dominated admission storage (key vectors, sibling lists, compounds duplicated across ~10^5 entries at Gauss scale); id vectors dedupe them. Set ordering is observable in the dump and in every snapshot loop that drives mail/revival emission order, so decoded comparators are correctness, not style.

**How to spot violations.** A value set or instruction map constructed without its helper; a comparator comparing raw ids; an `encode` on a probe path or in phase-2 parallel code; a decode reference held across an encode; the global `LogicalEntity` acquiring id fields.

**How to fix on violation.** Route container creation through the helpers; route order through the decoded comparators; decode to owned copies at boundaries; keep value minting at the single-threaded write sites.

**Code.** Structs + comparators + helpers in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp); consumers across [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) / [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) / [`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp) per `D-136`; dump writers in `infra/hashburst_dump.cpp`.

**See also.** `D-136` (40_decisions.md), `I-89`, [I-84](#i-84), [I-68](#i-68), [I-83](#i-83), [I-3](#i-3), [I-99](#i-99).

---

<a id="i-91"></a>
## I-91 — the origin maps are id-form in the dedicated per-LB origin space; tags are a closed enum; mail is id-form too (routing via the global `mailInterner`, internal per-LB); every observable ordering is decoded order

**Scope.** Prover / visualizer / compressor — `Memory::exprOriginMap`, `EquivalenceClass::equalityOriginMap`, `Memory::originInterner`, the `OriginTag` tables, the pack/mint/lookup/decode key helpers, `addOriginId` / `addOriginEncoded` / `overwriteOriginsId` / `decodeOriginMapSorted`, and every consumer in `D-131`.

**The invariant.**

1. **Origin strings live in `Memory::originInterner` (int32 ids, int64 packed `(expressionId, validityId)` keys).** Never in the `NameMap`: the trace dumps the full `nameMap.idToName` table, and mail-carried dependencies can name child-LB scopes never interned locally — a mint would shift the dumped table and destroy A/B comparability. Both origin maps share the one per-LB space, so class↔body history copies and merges are pure id operations. The interner resets ONLY in lockstep with `destroyGrid`'s `nameMap` reset and is never wiped on scope teardown (`exprOriginMap` survives `wipeSubtree` per [I-44](#i-44), so its id space must too).
2. **The tag vocabulary is closed.** `OriginTag` enumerates every historical tag literal; `originTagFromString` asserts on anything else — a new emission site must extend the enum and the name table (a firing assert is a report, never a fallback). No observable ordering may sort by the enum value; the only semantic tag comparisons are equality and the equality1/equality2 convenience test of the [D-49](40_decisions.md#d-49) cap-full policy.
3. **Every observable ordering is decoded order.** Order-sensitive whole-map walks (the dump's `exprOriginMap` section, `findEnds`, the compressor phase-1 extraction) iterate `decodeOriginMapSorted` snapshots — decoded `(expression, validity)` lex order, exactly the former `ExpressionWithValidity::operator<` map order; per-key history-line vectors keep insertion order ([I-84](#i-84)).
4. **Mail origin maps are id-form.** Cross-LB routing mail carries GLOBAL `mailInterner` ids (an id means the same at sender and receiver — a per-LB space could not); internal mail carries per-LB `NameMap` / `originInterner` ids. Records travel as `IntMailOrigin` id blobs, not string `OriginLine`s (`OriginLine` survives only as the decoded boundary form). `fillMailOut` copies the body's `IdOrigin` id records straight into `mailOut.origins_` (no decode); the commit seam translates each sender id → global id ([I-127](#i-127)); the pull / absorb decode global ids to owned string copies only at the read boundary ([I-3](#i-3)).
5. **Probe paths never mint.** Presence checks (the equality1 commit gate, the D-48 / I-32 emission gates, `liftToShallowestOriginAncestor`, the `buildStack` root find, `fillMailOut`'s origin copy) go through the non-minting `lookupOriginKey` — a never-interned pair is a definitive miss. Encodes happen only at single-threaded write sites ([I-83](#i-83)).

**Why.** The origin maps are the largest string containers in persistent prover state and ship in the ASIC build (the compressor consumes `exprOriginMap`) — id form is a memory necessity. The dump section and the chapter/compressor walks make iteration order and tag spelling observable, so decoded-order snapshots and the byte-exact tag table are correctness, not style. Rule 16 is untouched: the maps stay process documentation; no proof decision routes on their contents (the surviving presence checks are its sanctioned dedup-on-insert reads).

**How to spot violations.** An `encode` on a probe path or in phase-2 parallel code; a walk iterating the unordered map directly where output order is observable; a sort keyed on raw packed keys or on `OriginTag` values; a new tag literal passed as a string without an enum entry; a `mailInterner` MINT reached from a parallel worker (mint is single-threaded-seam only — [I-127](#i-127)); an interner reset outside `destroyGrid`.

**How to fix on violation.** Route probes through `lookupOriginKey`; route observable walks through `decodeOriginMapSorted`; extend `OriginTag` + `originTagName` together (the `static_assert` pins the table size); keep mail id-form and mint global ids only at the single-threaded commit / broadcast seams, decoding to strings at the read boundary.

**Code.** Interner/enum/helpers in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp); consumers across [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) / [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) / [`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp) / [`visualizer.cpp`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp) / [`compressor.cpp`](../GL_Quick_VS/GL_Quick/src/compressor.cpp); the dump writer in `infra/hashburst_dump.cpp`.

**See also.** `D-131` (40_decisions.md), [I-44](#i-44), [I-84](#i-84), [I-83](#i-83), [I-3](#i-3), [D-49](40_decisions.md#d-49).

---

<a id="i-92"></a>
## I-92 — the rule registry is id-form in the dedicated per-LB rule space; installs encode, the parallel burst only decodes; observable orderings are decoded order

**Scope.** Hash engine — `Memory::ruleInterner`, the id-form `LocalMemoryValue`, `HashMemory::originals`, the `RuleJustification` tables, and every consumer in `D-133`.

**The invariant.**

1. **Rule strings live in `Memory::ruleInterner` (int32).** Never the `NameMap` (the trace dumps the full id table) and never the admission/origin interners (disjoint populations). Encodes happen ONLY at the single-threaded install sites (`addToHashMemory`, `makeNormalizedKeysForAdmission`); the parallel hashburst decodes by array index — `const` refs, no mint, no rehash ([I-83](#i-83), [D-116](40_decisions.md#d-116)). The interner resets only in lockstep with `destroyGrid`'s `nameMap` reset.
2. **`LocalMemoryValue.validityId` is the install-time NameMap encode** (the same one that feeds the owner composite) — the firing path's D-55 scope-comparability check reads it directly; no per-firing lookup, no new NameMap mints. `isMarker` is the install-time head/marker classification; the firing path never re-scans the template.
3. **The justification vocabulary is closed.** `RuleJustification` (`none` = the marker-LMV empty string, `implication`, `integration`); `ruleJustificationFromString` asserts on anything else.
4. **Every observable ordering is decoded order.** The per-hit LMV candidate sort compares decoded heads (same outcomes as the former string member, including ties); the two `originals` write-path walks (`checkNecessityForEquality`, the integration-prep existence walk) and the dump's originals section iterate decoded lex-sorted snapshots; the marker dump sorts on decoded `(value, key, remainingArgs)`; the encodedMap dump decodes in place (the map itself is untouched, so its iteration order is unchanged) ([I-84](#i-84)).
5. **Post-substitution products stay string.** `FiringRecord` contents, staged admission values, and everything bound for mail are decoded/composed strings — the cross-LB space; the decode happens inside the burst as owned copies or const refs never held across a mint ([I-3](#i-3)).

**Why.** The rule registry sits under the firing/binding hot path and is duplicated across overall/local/delta/working registries and multiplied rule copies — the dominant string population after the origin maps. Decode-in-place keeps the burst read-only; install-time classification removes per-firing string scans without behavior change.

**How to spot violations.** A `ruleInterner.encode` inside `checkLocalEncodedMemoryStatic` or any phase-2 path; a sort keyed on raw ids; an originals walk iterating the unordered id set where writes or output depend on order; a new justification string without an enum entry.

**How to fix on violation.** Move the encode to the install site; route order through decoded snapshots or decoded comparators; extend the enum and its table together.

**Code.** [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp) (struct, enum, interner member); [`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp) (installs, firing path, wipe); [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) / [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (eradicate sweep, originals walks); `infra/hashburst_dump.cpp` (derived sections).

**See also.** `D-133` (40_decisions.md), [I-80](#i-80), [I-83](#i-83), [I-84](#i-84), [I-3](#i-3).

---

<a id="i-93"></a>
## I-93 — the LB state maps are id-form; observable walk orders are decoded order; the lbStateInterner never resets (expandedImplications survives destroyGrid)

**Scope.** Prover — `Memory::lbStateInterner`, `orBookkeeping` / `orDisjunctCount`, `integrationPrepared` / `integrationPreparedMarker` / `integrationStartIntMap`, `pendingWipeScopes`, `expandedImplications`, and every consumer in `D-135`.

**The invariant.**

1. **OR-state and expanded-implication strings live in `Memory::lbStateInterner` (int32, int64 `packLbStateKey` pairs).** Never the NameMap — mail-absorbed implications may be locally un-interned, and a mint there would shift the dumped id table. All writers are single-threaded. The interner does NOT reset at `destroyGrid`: `expandedImplications` deliberately survives grid teardown (existing behavior), and the space is NameMap-decoupled — ids never re-bind, decode stays valid across grids.
2. **The integration-prep gates key in the TEMPLATE space** (packed (templateId, validityId); `integrationStartIntMap` by the bare template id — its key never had a validity dimension). Gates are non-minting `lookupTemplateKey` probes; writes mint; `wipeSubtree` filters by the low-16-bits validity predicate.
3. **`pendingWipeScopes` holds NameMap validity ids.** Every queued scope was created via `encodePush`, so the insert is a non-minting `lookup` + assert (a firing assert = an un-interned scope was queued — report, not fallback). The drain sorts the ids by their decoded names (`compareSpans` over `decodeView`, tie-free — deduped vids, injective interner) before the per-vid `wipeSubtree` calls — the former string-set order ([I-84](#i-84)) with zero owned strings.
4. **Every observable ordering is decoded order.** `ordisMerge`'s disjunct sets are decoded-lex ordered storage (`orBookkeeping`, a `ColdSetMap` since Batch 2, kept sorted by a `DecodedIdLess` supplied per call to `insertSorted` — comparator state is the `lbStateInterner`; read in RUN order via `valueAt`, never `coldIntSetAt`); the seen-disjunct copy that feeds the D-36 origin rows and the per-branch cleanup decodes in that order; `sanitizeHashMemory` walks a decoded lex-sorted `expandedImplications` snapshot; every dump section derives decode + lex-sort.
5. **`orAdmissionSet` no longer exists.** The legacy gate is the `allowOrDisintegration` flag alone (the container had no insert site — [D-31](40_decisions.md#d-31)); the dump prints the literal empty section header for byte stability.
6. **Mail stays string.** `Mail::expandedImplications` is the cross-LB form; the absorb encodes into the packed registry.

**Why.** These were the last per-statement string populations in persistent prover state. Walk orders (the wipe-drain sequence, the sanitize rewrite plan, the convergence origin rows) are observable in proof output and the trace, so decoded-order snapshots are correctness, not style.

**How to spot violations.** An `encode` on a gate path; a walk iterating a packed set where writes or output depend on order; an `lbStateInterner` reset anywhere; a `pendingWipeScopes` insert without the lookup+assert; a new EWV container on `Memory` (the campaign's shapes are packed ids now).

**How to fix on violation.** Route gates through the non-minting probes; route observable walks through decoded lex-sorted snapshots; keep the interner append-only for the LB's lifetime.

**Code.** [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp) (declarations, `packLbStateKey`, `orDisjunctsAt`); [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) / [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp) (ordisMerge, prepareIntegration, the drain, sanitize/eradicate, installs/absorb); [`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp) (`wipeSubtree`); `infra/hashburst_dump.cpp` (derived sections).

**See also.** `D-135` (40_decisions.md), [I-84](#i-84), [I-83](#i-83), [D-31](40_decisions.md#d-31).

---

<a id="i-95"></a>
## I-95 One program-start reservation backs all statified per-LB memory — no per-object heap allocation, exhaustion is an assert

**Scope.** Statification memory hierarchy ([`memory_infra/global_memory_manager.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/global_memory_manager.hpp) / `.cpp`). Successor of the retired [I-13](#i-13).

**Rule.** `GlobalMemoryManager::init` makes ONE main reservation per process (sized by `static_pool_bytes`), never frees it during the run, never grows it, and dispenses fixed-size blocks (`static_block_bytes`) under a mutex. Statified container element payloads must come from this hierarchy — never from per-object `malloc`/`new`. Pool exhaustion is an assert naming `static_pool_bytes`; a fallback heap allocation on exhaustion is banned.

**The three sanctioned extra reservations — persistent, mail, and LB-body.** Selected by `StaticMemoryConfig::kind` (`PoolKind {Main, Persistent, Mail, Lb}`), each a separate `GlobalMemoryManager` instance with its OWN exhaustion assert and telemetry, none depleting the main pool, the no-heap-fallback contract binding all four identically:

- **Persistent** (`persistentMemory`, `static_persistent_pool_bytes`, smaller `static_persistent_block_bytes` blocks) backs the never-deloaded `Memory::intToBeProved` so the deactivation survey can read the goal registry while the main arena is deloaded (the determinism fix — [I-108](#i-108), [D-149](40_decisions.md#d-149)).
- **Mail** (`mailMemory`, `static_mail_pool_bytes`, main-pool-sized `static_mail_block_bytes` blocks) backs the never-deloaded cross-LB pull-model mail log; nothing reads its grant ledger, so it plays no role in any deload/throttle/steward decision ([`03_mail_system.md`](20_core_concepts/03_mail_system.md)). The single `ExpressionAnalyzer`-owned `mailArena` (an `LbArena{ &mailMemory }`) draws its blocks.
- **LB-body** (`lbMemory`, `static_lb_pool_bytes`, main-pool-sized `static_lb_block_bytes` blocks) backs the never-deloaded LB object store (`LbStore`) that holds the `Memory` node shells off the malloc heap; nothing reads its grant ledger either, so it plays no role in any deload/throttle/steward decision ([I-109](#i-109), [D-150](40_decisions.md#d-150)).

These three are the ONLY sanctioned extra reservations; a fifth, or a second MAIN pool, is a violation.

**Why.** The ASIC 0.1 milestone requires all prover memory static — deterministic capacity, no allocator churn in hot paths, the software model of the ASIC's fixed SRAM. A silent heap fallback would hide exactly the sizing signal the assert exists to surface, and reintroduce the unbounded consumption that makes FTA infeasible. The persistent pool stays within this doctrine: it is static, mutex-dispensed, and asserts on exhaustion — it just has a different lifetime (never deloaded, reclaimed per LB at discharge) and a separate sizing knob.

**Spot.**

- A statified container or manager calls `new`/`malloc` for element payloads.
- Exhaustion handled by allocating "just this once", or by widening a pool at runtime.
- A FIFTH reservation appears, or a second MAIN pool, or an "overflow" side arena (persistent, mail, and LB-body are the only three sanctioned extra reservations — anything beyond them is a violation).

**Fix.** Route the allocation through the block/page hierarchy of the appropriate pool. If a pool is genuinely too small, raise `static_pool_bytes` / `static_persistent_pool_bytes` / `static_mail_pool_bytes` / `static_lb_pool_bytes` in `parameters.hpp` (compile-time, same for every batch; [D-139](40_decisions.md#d-139)) — never weaken the assert ([I-19](#i-19)).

**Boundary (tier 1).** Bookkeeping containers — block lists, page tables, the recycle queue — are heap `std::vector`/`std::deque` by design; the invariant binds element payloads. A later statification tier moves bookkeeping into the blocks. See [D-174](40_decisions.md#d-174).

**Code.** `GlobalMemoryManager::init` / `acquireBlock` / `releaseBlock`. Memory mirror:.

---

<a id="i-108"></a>
## I-108 The persistent pool backs `Memory::intToBeProved` — never deloaded, reclaimed only at discharge, readable while the main arena is cold

**Scope.** The persistent (second) static-memory pool and the goal registry it backs ([`memory_infra/global_memory_manager.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/global_memory_manager.hpp) `persistentMemory`; [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp) `Memory::persistentArena` / `Memory::intToBeProved`).

**Rule.** `Memory::intToBeProved` (the packed-key goal registry) is an OWNED `Memory` member bound to `Memory::persistentArena`, a per-LB `LbArena` drawn from `persistentMemory` — NOT the deloadable `lbMemory` arena. The persistent arena is never deloaded and never compacted: it stays resident for the LB's whole active life, so `intToBeProved.count` / `keyAt` / `lookup` are legal even while the main `lbMemory` arena is deloaded. It is in NO deload stream — `LbMemory::visitContainers` does not enumerate it, and its tags 35-37 are retired (reserved, never reused) — so a deload/reload of the main arena leaves it untouched. It is reclaimed exactly once, at discharge: `Memory::dischargeStatementContent` calls `intToBeProved.resetToFresh` then `persistentArena.releaseAll` (that order — resetToFresh frees pages into the arena before the arena returns its blocks; the reverse asserts). The CE-filter clone starts it empty (fresh arena, default-constructed member). Declaration order is load-bearing: `persistentArena` is declared before `intToBeProved` so the map destructs before its arena.

**Why.** The deactivation survey (`deactivateRecursively` / `deactivateUnnecessary`) reads `intToBeProved` for main-scope goals to decide `isActive = false`. When it lived on the deloadable arena the survey was residency-gated and SKIPPED on a deloaded LB; which LBs are deloaded is timing-dependent (the hard-bound mass deload), so the deactivation decision — and thus which theorems are dropped — was non-deterministic. A persistent goal registry removes the only timing-dependent input to that decision.

**Spot.**

- `intToBeProved` re-added to `LbMemory` / `visitContainers` / the deload stream (it would deload with the main arena and re-introduce the residency gate).
- A deactivation survey gated on `lbMemory.manager.resident` instead of `isActive`.
- `persistentArena.releaseAll` before `intToBeProved.resetToFresh` (frees pages into a released arena — assert).
- `persistentArena` declared after `intToBeProved` (the map destructs into a dead arena).

**Fix.** Keep `intToBeProved` an owned `Memory` member on `persistentArena`; gate the survey on `isActive`; reclaim in `dischargeStatementContent` only, resetToFresh before releaseAll. See [D-149](40_decisions.md#d-149).

**Code.** `Memory::intToBeProved` / `Memory::persistentArena` / `Memory::dischargeStatementContent`; `deactivateRecursively` / `deactivateUnnecessary` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)).

---

<a id="i-107"></a>
## I-107 Statified containers address storage by per-LB virtual offsets — no observable depends on physical block identity or grant order

**Scope.** Statification cold substrate ([`memory_infra/lb_arena.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/lb_arena.hpp), [`memory_infra/arena_vector.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/arena_vector.hpp)); every statified container.

**Rule.** Cold containers store **virtual offsets** (`ArenaOffset`) into their LB's `LbArena` — a byte position relative to the arena base, resolved to a physical address per access by `resolve` (`block = offset >> blockShift`, `within = offset & blockMask`). An offset is a pure function of allocation order (single-threaded per LB); the only operation that changes an offset is the copying compaction (`LbMemory::reshuffle`), which runs with exclusive LB access, reassigns every offset densely, and leaves logical content untouched ([D-162](40_decisions.md#d-162)). Physical pointers are resolved per access and are never stored across deload- or compaction-capable boundaries, never compared, never ordered on, never dumped. Block grant order from the global manager (mutex under parallel sweeps — nondeterministic) must be invisible: every output (trace dumps, proof artifacts, deload bytes) derives from logical content only.

**Why.** A deload/reload cycle may bind entirely different physical blocks, and parallel grant order varies run to run. GL's determinism doctrine (every cross-run deviation is a bug) is reconciled with a shared physical pool exactly by this indirection — offset sequences are a pure function of the LB's own deterministic mutation history.

**Spot.**

- A container member holding `char*` / `T*` instead of an `ArenaOffset`.
- A sort or hash keyed on pointer values; a dump printing addresses.
- Behavior branching on pointer comparisons across blocks.

**Fix.** Store offsets; resolve via `LbArena::resolve` at the access site; derive any output from decoded logical state.

**Code.** `LbArena::alloc` / `popTo` / `resolve` / `releaseAll`; `LbMemory::reshuffle`.

---

<a id="i-103"></a>
## I-103 Deload files are a pure function of logical container content — element order in, bytes out, nothing else

**Scope.** The **v3 canonical** deload ([`memory_infra/lb_deload.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/lb_deload.hpp) / `.cpp`, `dumpLbMemory` / `dumpLbMemoryTail` / `loadLbMemory`); the `LbMemory` aggregate and every container that joins it. **NARROWED (`D-195`, 2026-07-07): this invariant governs only the DISCHARGE / chapter-export images.** The v4 **raw** arena image (`dumpLbMemoryRaw` / `loadLbMemoryRaw`, the eviction/reload hot loop) is DELIBERATELY EXEMPT — its bytes are nondeterministic (arena fragmentation + block grant order leak into the file), user-approved, because it is written and read back within one run and never A/B-compared. Determinism tooling must not point at `lb<ordinal>_raw.bin` files; content-determinism there is carried by the RESTORED LOGICAL STATE being byte-identical (same vids → same bytes, container bookkeeping untouched), not by the file bytes. **Extended (`D-198`):** the raw images now live at nondeterministic SLAB OFFSETS inside one preallocated `.deload/extent.bin` (allocation order varies run-to-run) — a second nondeterminism source, equally sound because slab offsets never enter proof output (the reload seeks to the LB's stored `rawExtentOffset_`, and the tightened `ordinal == expectedOrdinal` header assert is the slab-reuse tripwire).

**Rule.** `dumpLbMemory` / `dumpLbMemoryTail` serialize containers element-by-element in logical order, containers in ascending `ContainerTag` order, headers built only from deterministic facts (verbatim chain, the per-LB deload ordinal, counts, sizes, version, kind). Nothing allocation-historical — block boundaries, offset values, physical addresses, fragmentation holes — may influence a single output byte. File names are the deterministic per-LB **ordinal** (`lb<ordinal>_...`, registry-mapped), never a content hash; the in-house FNV-1a 64 survives only as the cold-string interner's probe index, never in a file name or header. The ordinal is cross-run/cross-host stable (a process-monotonic counter assigned in deterministic barrier order from 0), so one LB's header is reproducible. With tail deltas the file SET is a pure function of the content HISTORY (deterministic across runs — compaction points derive from the deterministic mutation history); every full dump restores per-state canonical form: two dumps of one LB's logically equal aggregate are byte-identical, across runs and hosts. (The v4 raw path has none of these obligations — see the Scope exemption above.)

**Why.** This is the "straightening" of the approved design and the property that makes `.deload/` files A/B-comparable debugging artifacts (the hashburst-dump methodology extended to whole LB images). It is also what lets reload rebuild a fresh consecutive arena (fresh offsets) with zero fidelity loss after arbitrary fragmentation.

**Spot.**

- A dump writer emitting raw arena bytes (block padding, interior holes) instead of walking elements.
- Offsets, pointers, or block counts appearing in the header or payload.
- A content hash (or any allocation-historical value) sneaking into a file name in place of the deterministic ordinal.

**Fix.** Stream through the container's logical indexing (`operator[]`, `size`); keep headers logical-facts-only; keep `fnv1a64`.

**Code.** `lbdeload::dumpLbMemory` / `loadLbMemory` / `fnv1a64`; pinned by the `churned_container_dumps_canonical_bytes` and `double_dump_is_byte_identical` unit tests.

---

<a id="i-111"></a>
## I-111 Arena accessors assert residency; reloads happen only at enumerated touch points, never lazily inside the hot path

**Scope.** Statification lifecycle — `LbArena::resolve`, `Memory::deloadStaticContainers` / `ensureLoaded`, every prover-side touch point of statified containers.

**Rule.** Touching a deloaded LB's statified storage is an assert — `resolve`'s residency check on element access AND `size`/`empty` on the containers. The size assert is load-bearing: a deloaded container reads as empty in RAM, so without it every loop over the container silently no-ops while the LB's RAM-resident side tables (`intStatementLevelsMap`, `intKnownStatements`, …) stay mutable — the registry/side-table pairing corruption the tier-1 branch gate caught live. Cold metadata is read ONLY from the counts recorded at deload (`Memory::deloadedCounts` / `intEncodedStatementsCount`). Reloads run only at the explicitly enumerated touch points, and every KERNEL-SIDE touch point goes through the CLAIM-CORRECT DOOR — the uniform handshake `claimAndLoadForWork` + release-`Idle` — never a bare `ensureLoaded` (a bare seam reload leaves the claim word `Dumped` on a resident arena, invisible to the pager's victim selection: the 4 GiB forensic census's 290-LB anomaly, [D-196](40_decisions.md#d-196)). **The touch-point enumeration** (each site carries a `SEAM DOOR` or equivalent comment): (1) the worker handshake itself — phases 1/2/3 entry + the phase-2 finalize re-claim; (2) the steward executor's prefetch-load task + the barrier head prefetch (`prefetchHead`); (3) the commit-barrier mail sweep (`proveKernel`, windowed, doored); (4) the `updateGlobal` proven-head + cross-LB recipient deposits and (5) the `updateGlobalDirect` cross-LB recipient deposit (both under the barrier seam window, doored); (6) the `drainDeferredAncestorAdmissions` ancestor reload (doored); (7) the barrier discharge reload before `dischargeStatementContent` (doored; post-quiesce); (8) the post-prove visualizer equality-node read (bare `ensureLoaded` — sanctioned: the steward is destroyed, no pager exists, claim words are meaningless post-prove) and the chapter export's `ensureLoadedForRead` (same, plus legal on discharged LBs, D-158). `ensureLoaded` on a resident LB is a defined no-op (the touch points call unconditionally); `deloadStaticContainers` on a deloaded LB is an assert (double deload = lifecycle bug). The door's census invariant is asserted: a successful `Dumped → Busy` claim over a RESIDENT arena is a bypassed reload (`claimAndLoadForWork` + the executor load task).

**Why.** A lazy reload buried in `operator[]` would hide every missed touch point forever — the run would limp on with pathological I/O instead of stopping at the exact line where an unenumerated consumer touched cold storage (Rule 19: the firing assert on a "can't happen" path is a gift). The enumerated-touch-point list is the documentation of who reads LB state between bursts; a new assert hit means the list — not the assert — must grow.

**Spot.**

- A `resident` check followed by an inline reload anywhere but `ensureLoaded`.
- A `resolve` assert "fixed" by reloading inside the arena.

**Fix.** Add the missing touch point as an explicit claim-correct door (`claimAndLoadForWork` + release; a bare `ensureLoaded` only on the post-prove steward-free paths) and document it in the enumeration above; never weaken the assert.

**Code.** `LbArena::resolve` / `markDeloaded` / `markResident`; `Memory::deloadStaticContainers` / `ensureLoaded` (`memory.cpp`); the doors in `proveKernel` / `updateGlobal` / `updateGlobalDirect` / `drainDeferredAncestorAdmissions` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)). Unit test: `test_steward.cpp` `seam_door_phase4_reloads_claim_correct`.

---

<a id="i-112"></a>
## I-112 An LB that leaves the active set never returns — `dischargedForever` marks it; only born-parked blocks may flip `isActive` back on

**Scope.** LB lifecycle — the kernel's end-of-iteration sweep (`proveKernel`), `activateZeroCondition`, the steward's discharge machinery.

**Rule.** Deactivation is permanent. The kernel's end-of-iteration sweep runs `Memory::dischargeStatementContent` on every LB that left the active set — flag `dischargedForever`, capture the exact registry pair set, empty the dischargeable containers, reshuffle (all blocks return, zero I/O) — and enqueues it on `ExpressionAnalyzer::pendingDischarge` for the pressure-lazy near-empty image dump ([D-157](40_decisions.md#d-157)). From that point nothing reactivates it, ever, and the kernel never reloads it: `ensureLoaded` asserts `!dischargedForever`; the post-prove gates probe RAM records (`dischargedRegistryKeys`, `intLocalEncodedStatementsSet`). ONE sanctioned read-only reload exists outside the kernel: the chapter export's `Memory::ensureLoadedForRead` ([D-158](40_decisions.md#d-158)) brings a drained LB's cold string content back for the origin walk — `isActive` / `dischargedForever` untouched, no kernel runs afterwards; "never returns" means never returns to the ACTIVE set. No mail-reactivation path exists (`smashMail` writes `mailIn` only and never touches `isActive`). The ONLY `false → true` `isActive` flip on a live LB is `activateZeroCondition` waking an induction zero-condition block that was parked at birth (`isActive = false` at creation) and therefore never entered an active snapshot — the wake asserts `!dischargedForever`, distinguishing "was never active" (parked, wakeable, never deloaded) from "was deactivated" (discharged, dead forever).

**Why.** The steward's relief ordering (pending discharges first — cheapest, content frozen, blocks never needed again) is sound only if "discharged" really means forever: a discharged LB resurrected by some future path would reload a stale image against live side tables. The flag plus the wake-site assert turn any such resurrection into a loud stop at its origin instead of silent registry corruption.

**Spot.**

- Any new code path setting `isActive = true` outside `activateZeroCondition` / grid reset.
- A discharged LB appearing in an active snapshot (the sweep's enqueue assert).
- "Fixing" the `activateZeroCondition` assert by clearing `dischargedForever` instead of asking why a discharged block was wake-targeted.

**Fix.** Treat any firing assert as a real lifecycle bug; never clear the flag. If a legitimate new activation path ever appears (Rule 8 territory), this invariant must be revised explicitly, not bypassed.

**Code.** `Memory::dischargedForever` (`memory.hpp`); the sweep enqueue + `activateZeroCondition` assert (`prover.cpp`); `ExpressionAnalyzer::pendingDischarge` (`prover.hpp`). See [D-156](40_decisions.md#d-156).

---

<a id="i-106"></a>
## I-106 Steward DELOAD-SET decisions read quiesced barrier counts and the grant ledger only — mid-iteration `blocksInUse` is telemetry and the exhaustion assert, never a deload-set decision input

**Scope.** Steward / pressure decisions — `GlobalMemoryManager::grantsSinceBarrier` / `armGrantTrigger`, the kernel's barrier decision point, all future steward planning code.

**Rule.** Every decision that selects WHAT gets dumped, evicted, or reshuffled is a pure function of logical state read at a deterministic point: quiesced block counts at the kernel's end-of-iteration barrier, or the monotone grant ledger mid-iteration (the one-shot `armGrantTrigger` crossing). `blocksInUse` mid-iteration — which dips as asynchronously executed frees complete, making its value timing-dependent — must never feed a decision; it serves telemetry (the peak line) and the exhaustion assert only.

**Why.** GL's determinism doctrine admits asynchronous EXECUTION but not timing-dependent DECISIONS: the BARRIER-time deload decisions (file sets, epochs, eviction victim lists) must be identical across runs. The ledger is order-independent (an iteration's grant SET is deterministic; the count after k grants does not depend on interleaving), so "did pressure cross in iteration N" is run-invariant even though which thread observes the crossing varies.

**Relaxation (D-148) — RETIRED ([D-161](40_decisions.md#d-161); throttle code removed).** The mid-burst admission throttle that once relaxed this invariant (a TOTAL-usage gate driving the steward's force-deload) is gone — the working-set pager subsumed mid-burst relief ([I-115](#i-115)) and the throttle code (`throttled_` / `waitWhileThrottled` / the `kThrottle*` band) is deleted. I-106 is absolute again: no mid-iteration usage read feeds any steward decision; the only mid-iteration signal is the monotone grant ledger.

**Spot.**

- A steward or kernel decision branching on `blocksInUse` between barriers.
- A partial mid-iteration drain ("until below the stop mark") instead of a whole deterministic unit.
- A trigger threshold computed from anything but barrier-quiesced logical state.

**Fix.** Move the read to the barrier, or re-express it against the grant ledger; drain in whole barrier-fixed units.

**Code.** `GlobalMemoryManager::grantsSinceBarrier` / `resetGrantLedger` / `armGrantTrigger` / `disarmGrantTrigger` (`memory_infra/global_memory_manager.*`); watermark constants in `memory_infra/steward.hpp`. See [D-160](40_decisions.md#d-160).

---

<a id="i-122"></a>
## I-122 Every steward touch of an LB goes through the per-LB claim word; a worker never waits on the steward except through the claim, and phases 2/3 never see a claimable LB

**UPDATED ([D-161](40_decisions.md#d-161), 2026-06-21).** A SINGLE `stewardClaim` word now governs ALL THREE phases (the second `burstClaim` word and the `Planned` state are retired). The unified handshake `claimAndLoadForWork` claims via `Idle/Dumped → Busy`, runs the load (reload + make-room) UNDER `Busy`, then PUBLISHES `WorkerOwned` only once the LB is fully loaded (a release-store). So `WorkerOwned` ⟹ fully-loaded: a split sibling part in phase 2 that sees `WorkerOwned` reads a complete LB, never a half-rebuilt cold-map index. A worker that finds `Busy` (a sibling loading, or the steward mid-op) waits — deadline-bounded by [I-113](#i-113) — until it resolves to `WorkerOwned` (ready → read) or `Idle`/`Dumped` (re-claim). The steward also CASes `Idle → Busy` for its own deload/reload/compaction, so `Busy` means "a worker is loading OR the steward is mid-op". The worker releases to `Idle` when done with the LB (asserted to have been held `WorkerOwned` at every release site). There is no barrier-armed eviction plan and no kernel-entry install — eviction is continuous in the phase windows ([I-114](#i-114)). **The `Busy`-during-load step is load-bearing:** publishing `WorkerOwned` before `ensureLoaded` returned (the first cut) raced a sibling read against the index rebuild — `reloadFromImage` marks `resident` before `loadLbMemory` fills the containers — producing `ColdHashSet::indexPlace: duplicate id` only under heavy phase-2 churn on split LBs. The historical description below predates all this and describes the retired two-word / Planned scheme.

**Scope.** Steward execution windows — `Memory::stewardClaim`, the kernel-entry plan arming, `MemorySteward`'s eviction pass, the `performElemPhase1` handshake.

**Rule.** The claim word (`Memory::stewardClaim`, atomic byte: `Idle / Planned / Busy / Dumped / WorkerOwned`) is the ONLY arbitration between the steward and the phase workers. Writers: the kernel entry installs `Planned` single-threaded before the phase-1 pool spawns; the steward and the phase-1 handshake move states exclusively by compare-and-swap (`Planned → Busy → Dumped` steward side; `Planned → WorkerOwned` worker self-service; `Dumped → WorkerOwned` worker reload-claim); the barrier resets terminal states to `Idle` single-threaded post-quiesce. A worker that loses the race to a `Busy` steward yield-spins on the word — never on a lock. The steward must hold a successful claim before touching ANY LB storage, and every claim source it CASes from (`Planned`) exists only between kernel entry and the LB's phase-1 slot — therefore phases 2/3 (whose hashburst holds raw element pointers, I-66) can never observe a cold or mid-dump LB on THIS word. **The mid-burst throttle's force-deload uses a SEPARATE per-LB word, `burstClaim`** ([D-148](40_decisions.md#d-148), [I-115](#i-115)), precisely so it never disturbs `stewardClaim`'s eviction-plan / discharge state; `stewardClaim` is untouched throughout phase 2.

**Why.** Pages must never move or vanish under a live reader. The single-word protocol gives exactly-once dumping with byte-identical content regardless of who dumps (content is frozen between barrier and slot), keeps the worker wait lock-free and bounded, and makes "steward never touches an LB a worker is processing" structural instead of scheduled.

**Spot.**

- Steward code touching an LB without a successful CAS from `Planned` (or, for reloads/reshuffles in later steps, the documented claim sources).
- A second writer of `Planned` / `Idle` outside the single-threaded kernel entry / barrier fold.
- A worker blocking on a mutex (instead of the claim word) to wait for the steward.

**Fix.** Route the touch through the claim word; extend the state machine explicitly (new states are a Rule-8 discussion), never bypass it.

**Code.** `Memory::stewardClaim` (`memory.hpp`); plan arming + barrier fold + `performElemPhase1` handshake (`prover.cpp`); the eviction pass (`memory_infra/steward.cpp`). See [D-175](40_decisions.md#d-175). The SEPARATE phase-2 `burstClaim` word is governed by [I-115](#i-115).

---

<a id="i-115"></a>
## I-115 A mid-burst pool-depletion throttle gates phase-2 task starts and force-deloads other LBs; relief comes from cold force-deload, never from finishing tasks; genuine exhaustion still asserts

**RETIRED ([D-161](40_decisions.md#d-161), 2026-06-21).** The throttle gate + force-deload are removed: the pre-allocation `waitWhileThrottled` wait was the silent-freeze deadlock (a worker parks before any `acquireBlock`, so when nothing is force-deloadable — split LBs / hot scratch — the exhaustion assert never fires). The working-set pager (`maintainWorkingSet` + the worker load↔evict exchange, [I-114](#i-114)) subsumes mid-burst relief: a worker that needs room evicts a deloadable LB inline, and genuine exhaustion asserts at `acquireBlock` (never a wait). The `GlobalMemoryManager` throttle (`throttled_` / `waitWhileThrottled` / `setThrottleOnsetCallback` / `isThrottled` / `totalBlocksInUse` / the `kThrottle*` band), the `Memory::burstClaim` word, `forceDeloadPass`, and the relief window are ALL now removed. The historical description below is preserved for the anchor.

**Scope.** The mid-burst admission protocol — `GlobalMemoryManager`'s throttle (`kThrottleHigh/Low`, `throttled_`, `waitWhileThrottled`, `setThrottleOnsetCallback`), the per-LB `Memory::burstClaim` word, the phase-2 executor pickup gate + `claimAndReloadForBurst`, the steward's `forceDeloadPass` / relief window.

**Rule.** When total pool usage (`blocksInUse + hotBlocksInUse`) crosses `kThrottleHigh` (7/8) a grant sets the throttle; a phase-2 worker calls `waitWhileThrottled` BEFORE pulling its next `(LB, split-part)` task, so started tasks finish while new ones pause. The steward, woken by the throttle onset, force-deloads not-running, non-split active LBs (biggest-first) until a release drops usage below `kThrottleLow` (3/4); a gated worker reloads its LB at pickup if the steward deloaded it. Steward and workers arbitrate over the SEPARATE per-LB `burstClaim` word (reset to `Idle` at phase-2 start; worker `Idle → WorkerOwned`, reloading a `Dumped` one; steward `Idle → Busy → Dumped`) — NOT `stewardClaim`, which the eviction plan / discharge own and phase 2 leaves untouched. The split LB being worked is NEVER deloaded (it stays resident). Relief comes ONLY from cold force-deload — a finished task's hot scratch is bulk-freed after phase 2, not per-task — so the throttle is a pure function of usage (set ≥High / cleared <Low). If the steward exhausts every candidate and the throttle still holds, the actively-running working set alone exceeds the pool: genuine exhaustion, surfaced by the next grant's `static_pool_bytes` assert (a gated worker always has an `Idle` candidate behind it, so no silent hang).

**Why.** With the transient hashmems on the cold arena ([D-176](40_decisions.md#d-176)), an FTA-scale burst can reach the pool limit mid-flight, which previously only crashed. The throttle degrades gracefully (serialize-under-pressure) instead. Sound because it changes only task-start timing, not which tasks run or the proof output (the [I-106](#i-106) relaxation). Inert below 7/8 — the current batches never cross it, so the protocol is provably idle on today's pipeline.

**Spot.**

- A phase-2 worker running `performElem2` without first `waitWhileThrottled` + `claimAndReloadForBurst`.
- `forceDeloadPass` deloading a split LB (`numberOfParts > 1`) or a worker-owned / in-flight one.
- The relief window left open into phase 3 (a force-deload straggling onto an LB phase 3 needs resident).
- A gate on the finalize or phase-3 pools (no `Idle` victims there → a hang instead of relief).

**Fix.** Keep the gate at the executor pickup only; keep force-deload candidates to not-running unsplit active LBs; close the relief window (with its quiesce) before phase 3.

**Code.** `GlobalMemoryManager` throttle (`memory_infra/global_memory_manager.*`); `MemorySteward::forceDeloadPass` / `requestRelief` / `begin`/`endThrottleReliefWindow` (`memory_infra/steward.*`); the pickup gate + `claimAndReloadForBurst` + claim reset + relief window (`prover.cpp`). See [D-148](40_decisions.md#d-148).

---

<a id="i-96"></a>
## I-96 RETIRED — the segregated hot grant path is deleted; every arena draws cold `acquireBlock`

The hot-arena block-traffic segregation is gone: `GlobalMemoryManager::acquireBlockHot` / `releaseBlockHot` / `hotBlocksInUse` / `peakHotBlocksInUse` and the `HotArena` / `HotArenaRegistry` infrastructure are deleted with the HOT substrate ([D-185](40_decisions.md)). Every arena — the per-LB store, the per-worker scratch arenas ([I-124](#i-124)), the sealed-page handoff ([I-101](#i-101)) — now draws the cold `acquireBlock` path, so its grants ARE visible to the steward (`grantsSinceBarrier`). Determinism holds not by HIDING the grants but by RELEASING them to zero before each barrier (per-task scratch release, pre-barrier sealed free), so the deload SET stays a pure function of quiesced logical counts ([I-106](#i-106) — which is absolute again now that its one throttle exception is also gone).

---

<a id="i-116"></a>
## I-116 RETIRED — superseded by `I-124`

The "a HotString never outlives its hot scope" liveness contract is unchanged in substance but moved off the segregated HOT substrate: transient calculation strings now live as `ScratchString` views into the worker slot's COLD scratch arena, released per worker task. See [I-124](#i-124).

<a id="i-124"></a>
## I-124 Per-worker scratch strings ride the COLD grant path, are RELEASED per worker task, and no `ScratchString` view outlives its scope

**Scope.** The per-worker scratch arenas (`ScratchArena` = `LbArena` bound cold, [`memory_infra/scratch_arena.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/scratch_arena.hpp) / `.cpp`; `ScratchArenaRegistry` / `scratchArenas` / `initScratchArenas`) and every consumer of `ScratchString` / `ScratchScope` ([`memory_infra/scratch_string.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/scratch_string.hpp)) — `checkLocalEncodedMemoryStatic`, the admission key builders, the drain-side transients. ([D-185](40_decisions.md)); replaces the retired [I-116](#i-116) and lifts the segregated [I-96](#i-96) off this substrate.

**Rule.** Each worker slot owns one `ScratchArena` — an `LbArena` `bind`-ed COLD (the `acquireBlock` grant path, no reserve cap, cold poison 0xCD), bound once in the `ExpressionAnalyzer` constructor, its blocks acquired lazily and RELEASED back to the pool (`releaseAll`) at every `performElem2` exit — no retention. Transient calculation strings live as `ScratchString` views into that arena and die with their window: per-call `ScratchScope`s rewind them; the per-task `releaseAll` bumps the generation that invalidates them wholesale. Nothing persistent — no `Memory` member, no staging record, no mail field, no container that survives the executor — stores a `ScratchString` or its raw `data` pointer; what must survive is COPIED out explicitly (ColdString table, sealed handoff pages, or `toStdString` at boundaries). Stack discipline binds rewinds: a view never crosses a rewind mark below its birth. Liveness = the arena generation + `usedBytes` (the byte-bump cursor position) at birth.

**Why.** The HOT substrate bought nothing scratch needs (already static); folding it onto the cold grant path removes a second memory model and its segregated ledger. Determinism holds: cold scratch grants are now visible to the steward (`grantsSinceBarrier`), but the per-task release nets them to ZERO before every barrier, so the deload SET stays a pure function of quiesced logical counts ([I-106](#i-106)/[I-107](#i-107)) and deload is content-invisible ([I-103](#i-103)) — output byte-identical, only grant-trigger timing shifts. The view's generation + used-bytes asserts catch escapes at the access site, but only until the span is refilled (no per-allocation metadata), so the rule is the contract and the asserts are the tripwire.

**Spot.** A `ScratchString` (or its `data` pointer) stored in any structure that outlives the allocating executor; a `reset`-and-retain at executor entry instead of `releaseAll` at exit; a `toStdString` inside burst hot paths; a rewind mark crossing live views; `usedBytes` decoupled from the byte-bump cursor the scratch fill advances (would silence the rewind tripwire).

**Fix.** Copy out explicitly at the escape boundary, or move the allocation to the longer-lived tier (executor scope, or sealed pages for staging). Keep the arena cold (`bind`), uncapped, released per task; keep `usedBytes` = the byte-bump cursor the scratch fill advances.

**Code.** `ScratchString::assertLive` / `toStdString`, `ScratchScope` (`memory_infra/scratch_string.hpp`); `ScratchArena` `bind` / `releaseAll` / `usedBytes` (`lb_arena.*`); the per-task `releaseAll` at `performElem2` exit (`prover.cpp`); `ScratchArenaRegistry` (`scratch_arena.*`). Unit tests: `test_scratch_arena.cpp`, `test_scratch_string.cpp`. See [D-163](40_decisions.md#d-163), [D-185](40_decisions.md).

---

<a id="i-104"></a>
## I-104 NameMap validity metadata is a flat paged parent-pointer forest (`validityNodes`), deload-persisted, surviving discharge; `verdict` walks it (no `pairMap`)

**Scope.** The NameMap validity metadata (`validityNodes` in `LbMemory`; `NameMap::nodes` façade pointer) and the scope-comparison primitives `verdict` / `comparable` / `deeperOf`.

**Rule.** The per-id scope hierarchy lives in ONE flat paged container on the LB arena — `validityNodes`, a `PagedVector<ValidityNode{parentId, ownSubId}>` (not heap `std::vector<std::vector<int16_t>>`, nor a jagged CSR) — serialized as ONE tag (`ValidityNodes` = 18, visited directly like the statement vectors) and reloaded with the LB. It **survives discharge** (`LbMemory::survivesDischarge`, range 4–18): an emptied container would read as unseeded and assert on a non-root id, and the post-prove readers still resolve validity ids into it. `stackOfValidity` / `ancestorsOf` are DERIVED by walking the `parentId` chain; `pairMap` is deleted — `verdict(a, b)` is derived (`a` strict-ancestors `b` iff `a` is on `b`'s parent chain), and `comparable` / `deeperOf` route through it. The walks run only on single-threaded paths (the parallel burst uses non-minting `lookup`).

**Why.** Statification's rule: state that survives between bursts goes to a cold paged container, never the heap. The metadata survives the LB's whole lifetime, so it goes cold + paged + deload-persisted. The flat parent-pointer forest is ~7× leaner than the jagged lists (4 B/id), O(1) to extend, and equally fast to query (an O(scope-depth) chain walk) — best for RT. `pairMap` held no information the forest does not, and its O(N)-per-scope fill loop in `encodePush` was pure waste.

**Spot.**

- A bounds / `ancAt position` assert from a metadata accessor, or a wrong ancestor walk → a reader touched an unseeded map for a non-root id, or an id beyond the seeded high-water.
- A discharged LB's `verdict` / `comparable` returning wrong answers → the forest was wrongly emptied on discharge (the `survivesDischarge` range).

**Fix.** Keep `ValidityNodes` inside the `survivesDischarge` range; read scope-comparison only through `verdict` / `comparable` / `deeperOf`; never re-introduce a separate ancestor cache without proving it carries information the parent-pointer forest lacks.

**Code.** `NameMap::verdict` / `ancContains` / `parentOf` / `pairCount`, `ValidityNode` + `LbMemory::validityNodes` (`memory_infra/lb_memory.hpp`), `LbMemory::survivesDischarge`. See [D-138](40_decisions.md#d-138).

---

<a id="i-105"></a>
## I-105 A NameMap id equals its cold-table id directly — "main" lazily interned as id 1; eternal-root fallbacks cover the pre-intern window

**Scope.** `NameMap` id ↔ string mapping: `mintName` / `lookup` / `decode` / `decodeView` / `nameCount` / `internMain` / `seedIfEmpty`, and every metadata accessor's unseeded branch.

**Rule.** A public NameMap id equals its cold-table id with no offset (as on `main`). "main" (id 1) is lazily interned as cold-table id 1 by `internMain` on the LB's first `encode` / `encodePush` (folded into `seedIfEmpty`), so a transient `Memory` that never encodes allocates nothing. Before that intern, `lookup("main")` / `decode(MAIN_ID)` / `decodeView(MAIN_ID)` return the eternal-root value, and the metadata accessors (`ancLen` / `ancAt` / `ancContains` / `stackLen` / `stackEmpty`) resolve `MAIN_ID` / slot 0 to their defined values — byte-identical to the seeded answer, with an assert that only root ids appear unseeded.

**Why.** The off-table-`"main"` special case forced `nameId == tableId + 1` everywhere; `main` itself was always direct. Restoring the direct mapping removes the offset arithmetic and lets the metadata index by id directly. `nameCount` is invariant across the change (old `count+1` with off-table main == new `count` with on-table main), so the hashburst dump stays byte-identical and theorem ids are unchanged.

**Spot.**

- A decoded name off by one position, or `decode` / `view` reading the wrong row → a stray `±1` survived the offset removal.
- An `internMain after the names table already grew` assert → seeding ran after a non-main name was already interned.

**Fix.** Keep id == cold-table id everywhere; route the first-touch intern through `internMain` only; resolve `MAIN_ID` through the eternal-root branches, never a re-added offset.

**Code.** `NameMap::internMain` / `seedIfEmpty` / `mintName` / `lookup` / `decode` / `decodeView` / `nameCount`. Reproduces the "main on-table" technique. See [D-138](40_decisions.md#d-138).

---

<a id="i-117"></a>
## I-117 A cold map's lookup index is DERIVED (rebuilt on reload, never persisted, never dirties the aggregate); keys + values live cold and paged — the index LOCATION is the experiment variable (heap hash | cold sorted | throw-away paged hash)

**Scope.** The cold-map family ([`memory_infra/cold_hash_map.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_hash_map.hpp)): the one `HashMap<KeyStore, ValueStore>` class (aliased `ColdHashSet` / `ColdStringTable`, `ColdHashMap`, `ColdMultiMap`), over the key-store policies `BytesKeyStore` / `PodKeyStore<K>` and the value-store policies `EmptyValueStore` / `SingleValueStore<V>` / `CsrValueStore<V>`.

**Rule.** Every cold map splits its state in two. The **index** maps key→id and is DERIVED: never serialized, rebuilt deterministically from the cold keys at reload (`rebuildIndex`), and it never escalates the aggregate's deload dirty flag. The **data** — keys (a `PagedVector` per key store) and any values (`ColdHashMap`'s value column, `ColdMultiMap`'s runs index + value column) — lives COLD on the LB arena and deload-persists. So the deload byte stream is a pure function of the logical key/value content in id order (append order == id order); the index is invisible to it ([I-103](#i-103)). `mint` / `lookup` assert residency first, so a deloaded map never answers a silent miss. The index's LOCATION is the experiment variable — three strategies: (1) **heap hash** on the parent (`D-165`) — a `std::vector` open-addressing hash, O(1) but malloc; (2) **fully-cold sorted** on the binary branch (`D-167`) — a sorted `PagedVector<int32_t>`, zero heap but O(log n), measured ~34% slower; (3) **throw-away paged hash** on this branch (`D-166`) — a `PagedHashIndex` (open-addressing slots on arena pages with in-place writes, never deloaded, rebuilt on reload), O(1) AND zero heap. All three keep the index derived; only its storage differs.

**Why.** The index is cheap to rebuild from the keys, so it is always derived; the only question is where its buckets live. In GL there is no heap — the right home for a hot derived structure is throw-away static (pool-backed, mutable, discarded on deload), which the paged hash gives: O(1) like the heap hash, zero malloc like the cold sorted index. The cold-sorted experiment proved zero-heap correct but O(log n) costs ~34% RT; the throw-away paged hash recovers O(1). Either way the irreplaceable content goes cold and the deload stays canonical.

**Spot.**

- A reload that loses dedup (a re-interned key gets a fresh id) → `rebuildIndex` was not run after the cold keys were bulk-loaded.
- A deload diff between two runs with equal content → something dumped the index, or sorted by raw id (forbidden — [I-84](#i-84)); the stream follows id order only. (The cold sorted index is ordered by KEY content, never raw id, and is never deloaded.)
- A silent "not found" on a deloaded map → a `lookup` skipped the residency assert.
- The index dirtying the aggregate (spurious deloads) → a throw-away index was given the aggregate's dirty flag; a `PagedHashIndex` carries none (it is never deloaded), and the cold-sorted variant used a private flag.

**Fix.** Keep the index derived and rebuilt from the cold keys on every reload / `copyFrom`; never serialize it or let it touch the aggregate dirty; stream keys and values in id order, never by index slot.

**Code.** `ColdHashSet::mint` / `lookup` / `rebuildIndex` / `indexInsert` / `indexPlace` + `buckets_` (`PagedHashIndex`, `memory_infra/paged_hash_index.hpp`); the key stores' `hashStored` / `hashProbe` / `equalStored`. See [D-165](40_decisions.md#d-165), [D-167](40_decisions.md#d-167), [D-166](40_decisions.md#d-166).

---

<a id="i-119"></a>
## I-119 Cold-map mutation beyond append (per-key `erase` / `eraseIf`, in-place `setValueAt`) preserves the canonical-bytes contract: survivor order kept, the mutation forces a full rewrite, the derived index rebuilt — POD-key set + single-value map only

**Scope.** The cold-map family ([`memory_infra/cold_hash_map.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_hash_map.hpp)) when used as a mutable per-LB container (the int-keyed-map migration). Erase/update instantiate for `PodKeyStore` AND `BytesKeyStore` set / single-value map (the byte-key store gained erase on Batch 2 for the flattened `eqClassSttmntIndexMapMap` — [D-168](40_decisions.md#d-168)); only `ColdMultiMap` (the CSR bag) stays out of the generic erase — the CSR set form `ColdSetMap` erases run-aware through `eraseSetIf` ([I-118](#i-118)).

**Rule.** Three mutators extend the append-only base without breaking [I-103](#i-103):

- **`setValueAt(id, v)`** overwrites a value in place through the one reviewed `PagedVector::setAt`. An in-place write is NOT an append, so it escalates the aggregate dirty state to `Restructured` — the next deload is a full canonical rewrite, never a tail-delta (which would silently drop the change).
- **`erase(key)` / `eraseIf(pred)`** run ONE forward **compaction** pass: survivors slide to the front of the key + value columns in lockstep (`moveKeyTo` / `moveValue` → `PagedVector::setAt`), the dead tail is `truncate`d (→ `Restructured`), and the throw-away index is rebuilt once. `moveKeyTo` is position-based: a POD key copies the key value; a BYTE key copies only the location entry, leaving the erased key's bytes as a hole in the append-only byte pool the copying compaction reclaims (the deload streams the survivors' logical bytes in id order, so the holes never reach the canonical image). O(count), not O(removed × count) — the "rebuild dense from live content" reload/reshuffle already does, scoped to one container. `erase(key)` routes through `eraseIf` with a one-key predicate.
- **Survivor order is preserved** — post-erase ids are insertion order minus the erased — so the deload byte stream stays a deterministic pure function of the surviving content (two runs with equal erase sequences dump byte-identically; an erase reached two ways yields the same bytes).

**Why.** The heap containers Batch 1 replaces both erase (`Memory::wipeSubtree` by closed scope) and update (`upsertStatementKey`'s flag-bit OR). The family had to gain both without making the deload non-canonical. Forcing `Restructured` is the load-bearing guard: it stops a mutated container from being tail-delta'd. Byte-key erase joined on Batch 2 (`eqClassSttmntIndexMapMap`, flattened to a `(validity ++ memberIds)` byte key, needed it): `moveKeyTo` slides only the location index, so the byte pool needs no O(N) in-place rewrite — its holes are reclaimed by the copying compaction. Only the `ColdMultiMap` CSR runs stay out of the generic erase (no per-key `moveValue`); the set form `ColdSetMap` erases run-aware via `eraseSetIf`.

**Spot.**

- A deload that drops an erase or an in-place update (the reloaded image still has the old row) → the mutator did not escalate to `Restructured`, so a tail-delta dump streamed only the appended tail.
- A stale `find` / `lookup` after an erase → `rebuildIndex` was skipped after compaction renumbered ids.
- A cross-run deload diff after equal erase sequences → the compaction was not stable (it must preserve survivor order — a forward compaction, never a back-swap).
- A compile error naming `erase` on a `ColdMultiMap` → intended: a CSR bag has no per-key value move; the CSR set form uses `eraseSetIf` instead.

**Fix.** `erase` / `eraseIf` rebuild the index and (via `truncate`) mark `Restructured`; `setValueAt` (via `setAt`) marks `Restructured`; keep the stable forward compaction (survivor order), never a reordering remove.

**Code.** `HashMap::erase` / `eraseIf` / `setValueAt`; `PagedVector::setAt` / `truncate`; `PodKeyStore` / `BytesKeyStore` `moveKeyTo` / `truncate`; `SingleValueStore::moveValue` / `truncateValues` / `setValueAt`; `StrSpan operator==`. See [D-170](40_decisions.md#d-170), [D-168](40_decisions.md#d-168).

---

<a id="i-118"></a>
## I-118 A cold SET-MAP (`ColdSetMap` = `HashMap<KeyStore, SetValueStore<V>>`) keeps each key's value run SORTED + DUPLICATE-FREE under a per-call comparator; interior insert, whole-run replace, and run-aware per-key erase all preserve the canonical-bytes contract

**Scope.** The cold-map family's set form ([`memory_infra/cold_hash_map.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_hash_map.hpp)): `SetValueStore<V>` + the `HashMap` set surface (`insertSorted` / `assignSet` / `setContains` / `eraseSet` / `eraseSetIf`), the Batch-2 substrate for the int→set maps. POD-key only (the run-aware erase uses `setKeyAt`).

**Rule.** The run for a key is a sorted-unique SET, not a bag:

- **`insertSorted(k, v, cmp)`** binary-searches the run with `cmp`, returns unchanged on a `cmp`-equal hit (the set dedup), else splices `v` at its sorted position (`PagedVector::insertAt`) and bumps every LATER key's run start by one. The comparator is PASSED PER CALL (the caller holds it), never stored — a stateful comparator (decoded-id order) stays valid because the caller binds it to the live interner. The order MUST be consistent across calls for one key, which holds because the caller always passes the same comparator.
- **`assignSet(k, vals, m)`** replaces a key's whole run with pre-sorted-unique values: overwrites the common prefix in place (no shift when the size is unchanged — the dominant case), then grows / shrinks the tail and adjusts later run starts by the size delta.
- **`eraseSet` / `eraseSetIf`** run ONE forward run-aware compaction: each survivor's key, run start, and whole value run slide to the front in lockstep, the dead tails are `truncate`d, the index is rebuilt once. Survivor order preserved → the deload bytes stay a deterministic pure function of the surviving content. Every write is no-op-guarded, so a predicate matching nothing leaves the container and its dirty state untouched.

All splices route through `PagedVector::setAt` / `insertAt` / `erase` / `truncate`, which force `Restructured` (never a tail-delta that would drop the change) — the canonical-bytes guard ([I-103](#i-103), [I-119](#i-119)). The two CSR deload tags (run-starts + values) plus the key tag stream in id order through `KeysView` / `RunStartsView` / `RunValuesView`; the index is derived ([I-117](#i-117)).

**Why.** The append-to-tail `ColdMultiMap` cannot serve a map whose value is a SET that grows on an existing key, dedups, and is erased per scope. The set form adds exactly those without breaking the canonical deload — the interior shift is O(value-tail + key-tail), the RT cost the batch gate watches.

**Spot.**

- A run with a duplicate value → `insertSorted` was called with a different comparator than the run was built with (the per-call comparator must be stable per key).
- A cross-run deload diff after equal erase sequences → the compaction was not stable (survivor order must be preserved).
- A wrong run length / off-by-one after an interior insert → the later keys' run starts were not bumped, or `assignSet`'s delta adjust was skipped.
- A compile error naming `insertSorted` on a `ColdMultiMap` → intended: the set surface is `SetValueStore`-only.

**Fix.** Always pass the same comparator for a given key; keep the forward (stable) compaction; bump later run starts on every interior insert / size-changing replace.

**Code.** `HashMap::insertSorted` / `assignSet` / `setContains` / `eraseSet` / `eraseSetIf`; `SetValueStore` (`insertValueAt` / `eraseValueAt` / `setRunStartRaw` / `valueRaw` / …); `PagedVector::insertAt`; the `RunStartsView` / `RunValuesView` facets. See [D-168](40_decisions.md#d-168).

<a id="i-98"></a>
## I-98 A cold BLOB-map (`ColdBlobMap` = `HashMap<KeyStore, BlobCsrValueStore>`) stores a key → an ordered RUN of variable-length byte BLOBS (a record value store); a per-record call-site codec is the single determinism point

**Scope.** The cold-map family ([`memory_infra/cold_hash_map.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_hash_map.hpp)) when a value is a multi-field record no single trivially-copyable `V` can hold — Batch 3's `equivalenceClassesMap`, and the Batch-4 `HashMemory` record-set values. The store is record-AGNOSTIC: it holds opaque bytes; a per-record-type (de)serializer at the call site produces / consumes each blob.

**Shape.** A two-level CSR, three dense paged columns, every boundary DERIVED (no redundant length column): `runStarts_` (key → first blob index), `blobStarts_` (blob → byte offset), `blobPool_` (dense blob bytes, may straddle pages — read through `contiguousRun`). Four deload tags with a POD key (key + the three value columns); each column streams via `appendSpanBytes` / `bulkAppendBytes` directly (no lengths/bytes staging handshake). `assignRun` is the whole-run replace (the bucket-rebuild door), `eraseBlobIf` the run-aware compacting scope-wipe, both O(tail) via `PagedVector::replaceRange` / `moveBytes`.

**Canonical bytes.** The deload image is a pure function of logical content ([I-103](#i-103)) IF AND ONLY IF the codec is canonical: every hash-backed field (e.g. `EquivalenceClass::equalityOriginMap`, an `unordered_map`) is SORTED before emit; insertion-ordered observable vectors are preserved verbatim. The codec — not the store — is the reviewed determinism artifact. See [D-169](40_decisions.md#d-169).

**Code.** `BlobCsrValueStore`; `HashMap::assignRun` / `blobAt` / `eraseBlobIf` / `blobCount` / `poolByteCount`; the `BlobStartsView` / `BlobPoolView` facets; `ColdBlobMap` alias; `PagedVector::replaceRange`; `serializeEquivalenceClass` / `deserializeEquivalenceClass`, `serializeSpecialTokenScan` / `deserializeSpecialTokenScan`.

<a id="i-120"></a>
## I-120 RETIRED — superseded by `I-125`

The "derived / transient container rides a DEDICATED non-deload HOT arena" pattern is removed: `changedClassesThisStep` (tag 655) and `eqClassNameCaches` (tag 705) now ride the per-LB COLD DELOADABLE arena, enrolled in `LbMemory::visitContainers` with the real deload-dirty (`I-125`). The routing mailboxes that briefly shared the pattern went to the never-deloaded mail pool instead ([I-101](#i-101)). No `acquireBlockHot` / `initHot` per-LB-container user remains; the hot-arena infrastructure deletion follows ([D-185](40_decisions.md)).

<a id="i-125"></a>
## I-125 Derived per-LB containers ride the COLD DELOADABLE arena, enrolled in `visitContainers`, no segregated hot accounting

**Scope.** `Memory::changedClassesThisStep` (`ChangedClassesBuffer`, [`memory_infra/changed_classes_buffer.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/changed_classes_buffer.hpp), tag base 655) and `Memory::eqClassNameCaches` (`EqClassNameCaches`, [`memory_infra/eq_class_name_caches.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/eq_class_name_caches.hpp), tag base 705). ([D-185](40_decisions.md)); replaces the retired [I-120](#i-120).

**Rule.** Both bind their columns to the LB's deloadable `manager` arena + the REAL deload-`dirty`, enrolled in `LbMemory::visitContainers` at a reserved 50-tag base, so they deload / reload / release / discharge like any other cold container — no private arena, no write-only sink. `changedClassesThisStep` is **dischargeable** (cleared every elementary step → empty at every deload boundary). `eqClassNameCaches` **survives discharge** (a persistent cross-grid memo, a pure function of `nameMap`; `survivesDischarge` true for 705..754). Both are written only at single-threaded seams (`standardProcessing` / the equi-class application; phase 2 is read-only on the LB, I-66/I-83), so the cold accessors' residency assert is never hit on a resident-during-its-burst LB.

**Why.** The hot arena bought nothing these need (already static); folding them onto the cold deloadable arena removes a second memory substrate and its segregated ledger (the retired I-96). `changedClassesThisStep` is empty at the seam so it deloads trivially; `eqClassNameCaches` is reconstructible, so deload-persist is the uniform choice (drop-and-rebuild stays a future memory option). Determinism holds: deload bytes are a pure function of insertion order (I-103); the grant traffic now visible to the steward is content-deterministic (one-shot trigger per barrier, deload SET from quiesced logical counts, I-106/I-107).

**Spot.** Either container given a private arena; `changedClassesThisStep` in `survivesDischarge`; `eqClassNameCaches` NOT in `survivesDischarge` (the post-prove readers would lose it); a `.size` / `kindOf` read on a deloaded LB without `ensureLoaded`.

**Fix.** Bind to `&manager` + real `dirty`; enroll in `visitContainers` + `liveBytes`; `changedClassesThisStep` dischargeable, `eqClassNameCaches` survivesDischarge.

**Code.** `changed_classes_buffer.hpp::ChangedClassesBuffer` (+ out-of-line `push`/`classAt` in `memory.hpp`); `eq_class_name_caches.hpp::EqClassNameCaches` (+ out-of-line `kindOf`/`tokensOf`, `Codec<SpecialTokenScan>` in `memory.hpp`); `lb_memory.hpp` (members, `visitContainers` splices, `survivesDischarge`, `liveBytes`). Unit tests: `test_memory.cpp` (`changed_classes_buffer_round_trip`, `eqclass_name_caches_*`).

<a id="i-100"></a>
## I-100 The four cold owner-set maps (`normalizedEncoded*`) are READ on the request-generation hot path via a zero-allocation byte peek, never a full `OwnerSet` decode — the prune's short-circuits stay free

**Scope.** The four `normalizedEncoded{Keys,Subkeys,SubkeysMinusOne,SubkeysMinusTwo}` maps in every `HashMemory`, statified onto `TypedColdBlobMap<NormKey, OwnerSet>` (one blob per key, run-length-1, whole-value replace). The request-generation prune probes them 10^4–10^6× per burst, and the common case (all-`main` scope, unsplit, loose owner) reads only the `hasLooseOwner` byte before short-circuiting. So the hot read must NOT materialize the `OwnerSet` (two `std::set`s, the cost the blob map's `recordsAt` would pay): `ExpressionAnalyzer::ownerKeyAccepts` does the `lookup` first (a miss returns `false` before any predicate or scope computation), then `TypedCold::peekRecordBytes` (zero-copy off the arena, a `thread_local` scratch only on a rare page straddle) wrapped in an `OwnerSetBlob` view whose readers the three byte-overload predicates (`ownerSetHasComparable` / `partitionAccepts` / `ownerSetUSatisfied`) consume in place. The verdict is identical to the former `find` + `const OwnerSet&` form, so the prune stays a sound over-approximation ([I-70](#i-70), [I-79](#i-79)). Writes are install-time RMW (`mergeOwnerRecord`), single-threaded, not on this path.

**Code.** `ExpressionAnalyzer::ownerKeyAccepts` / `mergeOwnerRecord`; `OwnerSetBlob` (byte view, layout-coupled to `Codec<OwnerSet>`); `TypedCold::peekRecordBytes` → `HashMap::peekBlobContiguous` → `BlobCsrValueStore::peekBlob` (`PagedVector::contiguousRun`); the `OwnerSetBlob` overloads of `partitionAccepts` / `ownerSetUSatisfied` / `ownerSetHasComparable`. See [D-140](40_decisions.md#d-140), [I-49](#i-49), [I-80](#i-80).

<a id="i-99"></a>
## I-99 The algebra AND integration admission/rejection containers are cold-stored; the record run is kept canonical (sorted) by read-modify-write, reads snapshot, erases batch, writes stay single-threaded

**Scope.** The algebra admission/rejection `HashMemory` containers statified by [D-172](40_decisions.md#d-172): `admissionMap` / `rejectedMap` (`TypedColdBlobMap<int32_t, AdmissionMapValue|RejectedMapValue>`), `admissionStatusMap` (`TypedColdMap<int32_t, uint8_t>`), `consumedAdmissionKeys` / `revisitInProgress` / `varsInAdmissionMapKeys` (`TypedColdSet`). The **integration** twins join on the same substrate: `admissionMapIntegration` (`TypedColdBlobMap<int32_t, IntegrationEntry>` — its nested inner `map<IntInstruction, ValueIdSet>` flattened one `IntegrationEntry` blob per inner entry), `rejectedMapIntegration` (`TypedColdBlobMap<int32_t, RejectedMapIntegrationValue>`), and `varsInAdmissionMapIntegrationKeys` / `varsInRejectedMapIntegrationKeys` (`TypedColdSet<int16_t>`); the integration side has NO `consumedAdmissionKeys` / `admissionStatusMap` analog, and `admissionSetIntegration` / `triggersForAdmissionSetIntegration` (the EWV set side) stay on the heap. The packed `(templateId, validityId)` int32 keys them all directly through the identity `Codec<int32_t>`. Layered ON TOP of [I-89](#i-89) (keys packed in the template space) / [I-90](#i-90) (values id-form) — those were already interned; this invariant governs the cold STORAGE.

**The invariant.**

1. **The blob run is canonical (sorted), maintained by RMW.** Each `AdmissionMapValue` / `RejectedMapValue` is one record blob; the run is kept sorted-unique by the decoded comparator (`DecodedAdmissionValueLess` / `DecodedRejectedValueLess`), exactly reproducing the former `std::set`. A value insert is read-modify-write — `lookup` → `recordsAt` decode into the comparator-bearing set → `insert` → `assignRun` the whole run — never a raw append (which would break order/dedup). So the deload bytes AND the sacred dump derive byte-identical to the heap set, and `isAdmitted`'s snapshot-and-iterate order is unchanged.
2. **Reads snapshot, never hold an arena reference.** `admissionRecordsAt` returns an OWNED `AdmissionValueSet` (decoded copy), so a later mutating call (e.g. `updateAdmissionMap` inside `isAdmitted`'s loop) cannot invalidate the snapshot ([I-3](#i-3) discipline). The dump reads through the same snapshot (kept alive in `rows`, mirroring `writeEncodedMapMarkers`), preserving Rule 14's derived-view contract.
3. **Erases batch through `eraseBlobIf` / `eraseIf`.** Single-key removals (`cleanAdmissionMap`'s [I-41](#i-41) closure, the subtree wipe) collect the keys then run ONE compacting pass per container; `rejectedMap` is never erased ([I-37](#i-37)). The compaction marks `Restructured` so a mutated container is never tail-delta'd ([I-119](#i-119)).
4. **Writes single-threaded; the one parallel read is a non-minting lookup.** All cold mutation rides the post-fixpoint `drainAdmissionKeysAlgebra` drain + the single-threaded synchronous helpers ([I-68](#i-68)/[I-83](#i-83)); the phase-2 staging consumed-key gate is a pure `contains` (lookup) on the resident cold set, parallel-safe. `admissionMapPropagate`'s deliberate default-false status `[]`-probe (the dump counts the created entry) is reproduced as `find`-miss → `upsert(0)`.
5. **The integration admission value is a nested map flattened to blobs.** `admissionMapIntegration`'s value is `map<IntInstruction, ValueIdSet>` (not a flat set); on the cold run each inner `(instruction, payload-set)` entry is one `IntegrationEntry` blob, the run held in the inner map's decoded order (`DecodedInstructionLess` outer, `DecodedIdLess` inner). The RMW round trip is `admissionIntegrationRecordsAt` (decode the run back into the nested `IntegrationEntryMap`, both stateful comparators threaded from the `ValueInterner`) → mutate via `payloadAt` → `assignRun(flattenIntegrationEntryMap(em))`. `isAdmittedIntegration` / `updateAdmissionMapIntegration` write back PER snapshot entry — the literal cold mirror of mutating the live map before the re-entrant `prepareIntegrationCore2`, which only ever writes a structurally different marker-form key (never the renamed key under iteration). `prepareIntegrationCore2`'s own write is the "ensure the instruction entry exists (empty payload)" round trip. [I-22](#i-22) holds unchanged: integration admission entries are NEVER cleaned on revival/consumption — `eraseBlobIf` fires only in the equi-class re-key drop and the subtree wipe (the `cleanAdmissionMap` integration erases stay the bare-marker-form no-ops they always were); `rejectedMapIntegration` IS erased after revival (`revisitRejectedIntegration2`).

**How to spot violations.** A raw `appendRecord` into `admissionMap` / `rejectedMap` (breaks sort/dedup); a per-key `eraseBlobIf` in a loop (use a batched key set); a reference into `recordsAt`'s result held across a mutating call; a `templateInterner` / `valueInterner` mint on the phase-2 staging consumed-gate path; an admission-integration write-back deferred PAST a re-entrant `prepareIntegrationCore2` that could read the same key.

**Code.** Algebra: `admissionRecordsAt` / `insertAdmissionValue` + `Codec<int32_t>` ([`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp) / [`memory_infra/typed_cold_map.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/typed_cold_map.hpp)); `updateAdmissionMap` / `isAdmitted` / `cleanAdmissionMap` / `applyEquivalenceClassToAdmissionMap` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)); `admissionMapPropagate` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)); `drainAdmissionKeysAlgebra` + the subtree wipe ([`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp)); the cold-adapted dump readers ([`infra/hashburst_dump.cpp`](../GL_Quick_VS/GL_Quick/src/infra/hashburst_dump.cpp)). Integration twins: `admissionIntegrationRecordsAt` / `flattenIntegrationEntryMap` / `rejectedIntegrationRecordsAt` / `insertRejectedIntegrationValue` + the `IntegrationEntry` record and `Codec<IntegrationEntry>` ([`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)); `isAdmittedIntegration` / `updateAdmissionMapIntegration` / `prepareIntegrationCore2` / `applyEquivalenceClassToAdmissionMapIntegration` / `applyEquivalenceClassToRejectedMapIntegration` ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)); `revisitRejectedIntegration2` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)); the integration `wipeHashMem` erases ([`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp)). Unit tests: `integration_admission_cold_roundtrip` / `integration_rejected_cold_roundtrip` (`tests/test_memory.cpp`). See [D-172](40_decisions.md#d-172), [I-98](#i-98), [I-22](#i-22), [I-89](#i-89), [I-90](#i-90).

<a id="i-123"></a>
## I-123 `HashMemory` is an `LbMemory` member; `LbMemory::visitContainers` is the SOLE enumeration driving its dump / load / release / discharge — no parallel hand-threaded path

**Scope.** The four per-LB `HashMemory` instances (`overallHashMemory` / `localHashMemory` / `localHashMemoryDelta` / `workingMemory`) folded into `LbMemory` by [D-147](40_decisions.md#d-147). They are `LbMemory` members declared after `manager`; `Memory` reaches them through reference aliases (`HashMemory& overallHashMemory = lbMemory.overallHashMemory`).

**The invariant.**

1. **One enumeration.** `LbMemory::visitContainers` (const + mutable) is the single source of container enumeration for the WHOLE LB — its own tags 0..50 AND each `HashMemory` instance's facets at its reserved 100-tag base (51 / 151 / 251 / 351, defined in `hash_memory.hpp`), bridged into a `ContainerTag` by a cast. Every arena-lifecycle op (the deload dump, the reload, `releaseStaticBlocks`, `clearDischargeableContainers`, `liveBytes`) drives off this one walk; there is NO separate hand-threaded `HashMemory` path (`buildHashMemoryDeloadColumns` / `extraDump` / `extraLoad` and the manual per-instance release are retired). The `lb_deload` `extraColumns` parameter survives defaulted-and-unused.
2. **Byte-identical tags.** The bridge records `HashMemory` facets at directory tags 51..450 — identical to the former extra-column scheme — so the on-disk deload image (directory + tag-ordered payload + tail-delta counts) is unchanged by the fold; proof output is byte-identical.
3. **Survives discharge.** `survivesDischarge(tag)` returns true for tags 51..450, so `clearDischargeableContainers` never empties `HashMemory`: a discharged LB keeps its hash engine resident for the post-prove chapter export, then the pending drain dumps the full image (the pre-fold behaviour, where `HashMemory` lived outside `LbMemory` and discharge could not reach it).
4. **Natural destruction order.** The instances are declared after `manager` in `LbMemory`, so they destruct before the arena their cold containers ride — the explicit `~Memory` `releaseAllCold` safeguard is retired (the `freePage`-on-dead-arena hazard is structurally impossible). `HashMemory::visitContainers` has a const overload (shared static impl) because `dumpLbMemory` reads a `const LbMemory&`.

**How to spot violations.** A reintroduced `extraColumns` argument carrying `HashMemory` facets; a `HashMemory` deload/release path that bypasses `LbMemory::visitContainers`; a `HashMemory` tag added below 51 or at/above 451 (collides with `LbMemory`'s own range); a `survivesDischarge` that drops the 51..450 range (would empty the hash engine on discharge and break the export); a `HashMemory` instance declared before `manager` in `LbMemory` (resurrects the destruction-order hazard).

**Code.** `LbMemory` members + ctor + `visitContainers` + `survivesDischarge` + `liveBytes` ([`memory_infra/lb_memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/lb_memory.hpp)); `HashMemory::visitContainers` const/non-const + the deload-base constants ([`memory_infra/hash_memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/hash_memory.hpp)); the `Memory` reference aliases + retired `~Memory` ([`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)); `dumpStaticContainers` / `releaseStaticBlocks` / `reloadFromImage` ([`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp)). Unit test: `memory_deload_reload_preserves_all_four_hashmemories` ([`tests/test_lb_deload.cpp`](../GL_Quick_VS/GL_Quick/src/tests/test_lb_deload.cpp)). See [D-147](40_decisions.md#d-147).

<a id="i-113"></a>
## I-113 Every worker wait for an LB is deadline-bounded — a claim-spin past `kStuckSeconds` asserts at its origin, never a silent freeze

**Scope.** The forward-progress backstop for the working-set deload/reload policy. Every point where a phase worker can wait for an LB it must process is bounded by a wall-clock deadline (`steward::kStuckSeconds`, 30 s): the phase-1 `stewardClaim` spin and the phase-2 `burstClaim` spin (`claimAndReloadForBurst`) stamp a steady-clock start before the loop and, on each `Busy` iteration, call `steward::stuckDeadlineExceeded(start)`; a `true` result prints the stuck LB's full parent chain (`buildLbChainString`, Rule 12) and asserts (Rule 19). The intent is that a worker needing a non-resident LB makes room by eviction or asserts on genuine pool exhaustion (`static_pool_bytes`) rather than parking, so the deadline is only ever reached on an UNEXPECTED hang (e.g. a steward thread stalled mid-op). It directly closes the silent-freeze failure mode of the phase-2 `waitWhileThrottled` gate ([I-115](#i-115)), whose unbounded spin could latch forever with no `acquireBlock` ever firing the exhaustion assert.

**Safety tripwire, not a proof input.** The deadline reads the monotonic clock, so its firing is timing-dependent; like the hashburst diagnostic it never influences proof output, only converts a hang into a loud abort at its origin. Asserts are live in the Release build (the project does not define `NDEBUG`), so the tripwire fires in production runs, not only in debug.

**Code.** `steward::kStuckSeconds` / `steward::stuckDeadlineExceeded` ([`memory_infra/steward.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/steward.hpp) / `.cpp`); the `Busy`-claim wait inside `MemorySteward::claimAndLoadForWork` (the unified handshake used by all three phases). Unit test: `test_steward.cpp` `stuck_deadline_predicate`.

<a id="i-114"></a>
## I-114 RAM is a cache of the working set — the steward keeps only the processing LBs + the next few resident in EVERY phase and drains the rest; the claim word (not the index window) guarantees an in-flight LB is never deloaded

**Scope.** The active-LB deload/reload policy — the working-set pager ([D-161](40_decisions.md#d-161), reserve-target drain per [D-197](40_decisions.md#d-197), planner/executor split per [D-196](40_decisions.md#d-196)). Each of the three phases opens a steward window (`beginPhaseWindow` / `endPhaseWindow`) over its dispatch cursor + the active vector; while open, the PLANNER pass `maintainWorkingSet` ENQUEUES — onto two fixed 256-slot rings drained by the I/O EXECUTOR pool (`ioThreadCountFor(workers)` threads; HIGH lane = prefetch loads, LOW lane = evictions + reshuffles, HIGH served first except below the emergency floor) — prefetch loads over the UNIFIED prefetch region `[cursor, cursor+workers+kLookaheadWorkerMultiple·workers)` (starting AT the cursor, ending where the keep window ends; reload a cold upcoming LB; reshuffle a fragmented resident one) and the **reserve-shortfall evictions**: while the PROJECTED free blocks (live free + in-flight dump blocks + already-queued eviction blocks) fall short of `kReserveBlocks` (2048) and an eligible farthest-next-use victim exists, plan one more; the pass ends when the projection meets the reserve, no victim qualifies (a DEFINED result — the working set fits, the stream pauses; genuine exhaustion still asserts at `acquireBlock`, Rule 19), or the ring fills (designed load-shedding, counted, re-planned next pass). Executor task execution is claim-first drop-on-lose with a window-generation re-validation under the CURRENT cursor; the async dump contract holds an eviction's LB `Busy` + blocks until the raw image is complete, so `Dumped` is never observable before image-complete + blocks-returned. `quiesce` = planner parked + rings empty + executors idle (load-bearing for the barrier's `deloadRegistry` reference read); the discharge drain stays on the planner thread ([I-106](#i-106) execution unchanged). Victim selection is TWO-TIER (`pickVictimTwoTier`, [D-196](40_decisions.md#d-196)): tier 1 is the floor-preferred Belady scan (`pickVictimBehindCursor`: next-use distance `d(i)=(i−cursor) mod n`, a backward scan from `keepLo−1` with wraparound visiting strictly decreasing `d`, floor `blocksHeld >= kMinEvictBlocks` (4) — tiny LBs stay implicit residents because the per-eviction fixed cost dominates); when the reserve is breached and NO above-floor victim exists, tier 2 re-runs the SAME scan with a one-block floor (the survival branch — a floor-only drain finds no victim in the median-2-block early-batch population while grants continue, the post-pager 4 GiB wall). Eligible = active, unsplit, resident, `Idle` (not in flight), outside the kept working set `[cursor−workers, cursor+workers+kLookaheadWorkerMultiple·workers)`. The prefetch region and the keep window share the end, so a freshly prefetched LB is never drained by the same pass (`kLookaheadWorkerMultiple = 2` — the lookahead scales with the worker count). This REPLACES the former watermark drain (mass-evict at 3/4, one-for-one at 1/2), which starved eviction victims and hit the `IncubatorPeano1` exhaustion assert at 4 GiB; the biggest-first selector `pickBiggestDeloadable` is retired entirely. **The claim word is the correctness guarantee, the index window the churn optimization:** the steward CASes `Idle → Busy` before any deload, so a worker that claimed the LB (`WorkerOwned`) makes that CAS fail — an in-flight LB is NEVER deloaded regardless of the cursor's staleness. Phase 2 registers its REAL dispatch atomics: an EXECUTOR window over the pool's `next` atomic + the flat per-pass `execOrder` (one entry per task; split-part duplicates collapse via dedup + CAS), then a FINALIZE window over `nextLi` / `toRun`; redo passes window their own vectors; the frozen `phase2Cursor` is retired. The barrier additionally calls `prefetchHead` — one-shot `kAnyWindowGeneration` HIGH-lane loads for the next iteration's head, after the discharge decision (inactive heads skipped; the barrier quiesce empties the rings before `dischargedForever` is ever set).

**Worker side.** A worker makes the LB it must process resident through the unified handshake `claimAndLoadForWork` (claim → `ensureLoaded` straight into the reserve's free blocks, NO eviction on the normal path) and **releases the claim (`Idle`) the moment it is done with the LB**, so the LB becomes deloadable again. The old load↔evict exchange is gone; the WIDENED WORKER PRESSURE VALVE replaces it ([D-196](40_decisions.md#d-196)) — EVERY claim (resident or cold: the pure-grant path, where a worker's task grants scratch / statement blocks with no reload in sight, previously had no valve at all) checks `freeBlocks` against the emergency floor `kEmergencyFloorBlocks` (= `kReserveBlocks / kEmergencyFloorDivisor` = 512) and, on breach, evicts ONE behind-cursor victim (`evictOneForReload`, two-tier Belady) and bumps `emergencyEvictCount` (expected 0; nonzero = the planner cannot keep up); the cold-miss path adds the reload-fit trigger (free blocks cannot cover THIS reload, estimated from `lastRawImageBytes`). At most one eviction per claim, no loop — if one eviction is not enough, the reload's `acquireBlock` surfaces genuine exhaustion (Rule 19) — this is the load-bearing half: a processed LB that stays `WorkerOwned` is never drained, so the resident set grows to ALL active LBs and the pool exhausts (the first-run `IncubatorPeano1` depletion). Phases 1 and 3 release at the body end. Phase 2's executor — where one LB has N concurrent parts — releases via a per-LB remaining-parts counter: the part that decrements it to zero (the last to seal, so no part still reads the LB) flips the claim to `Idle`; the finalize then re-claims + reloads it. All three phases use the same handshake (CE-filter clones run with no steward — always resident — and skip it). Genuine exhaustion (the concurrent working set alone exceeds the pool) asserts at `acquireBlock` naming `static_pool_bytes`, never a silent wait; every `Busy`-claim wait is deadline-bounded ([I-113](#i-113)).

**Determinism.** The eviction set is timing-dependent (the gitignored `.deload/` file set varies run-to-run — accepted, as for the retired [I-115](#i-115)); proof output stays deterministic because deload/reload is content-invisible ([I-103](#i-103), [I-107](#i-107)) and the last timing-dependent decision input — the residency-gated deactivation survey — was removed ([I-108](#i-108)). **Inert without pressure:** a batch whose `freeBlocks` stays at or above `kReserveBlocks` evicts nothing → reloads nothing (light batches, and the whole pipeline at 8 GiB where peak « total − reserve, run exactly as before). Relaxes [I-106](#i-106): mid-phase `blocksInUse` drives the pager's reserve-target drain in all phases.

**Code.** `MemorySteward::maintainWorkingSet` / `pickVictimBehindCursor` / `pickVictimTwoTier` / `evictOneForReload` / `claimAndLoadForWork` / `beginPhaseWindow` / `endPhaseWindow` / `enqueueIoTask` / `popIoTaskLocked` / `executeIoTask` / `ioThreadMain` / `inFlightBlocks` + `kMinEvictBlocks` / `kReserveBlocks` / `kEmergencyFloorBlocks` / `kLookaheadWorkerMultiple` / `kIoRingCapacity` / `ioThreadCountFor` / `kIoThreadsOverride` ([`memory_infra/steward.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/steward.hpp) / `.cpp`); the three phase windows + the worker handshake + the simplified barrier in `ExpressionAnalyzer::proveKernel` / `performElemPhase1` / `performElemPhase3` ([`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp)). Unit tests: `test_steward.cpp` `pick_victim_behind_cursor_selects_and_skips`, `pick_victim_two_tier_floor_preferred_then_floorless`, `thrash_regime_reserve_target_evicts_the_deficit_not_the_set`, `unified_window_prefetched_lb_survives_same_pass`, `claim_and_load_for_work_handshake`, `worker_valve_fires_below_emergency_floor_on_pure_grant_path`, `worker_reload_forced_exhaustion_fires_fallback_exactly_once`, `io_thread_count_clamps_and_overrides`, `executor_lane_priority_flips_to_low_below_emergency_floor`, `planner_enqueue_dedups_drops_count_and_stale_tasks_drop`, `async_dump_publishes_dumped_only_after_blocks_returned`, `stress_concurrent_dump_load_disjoint_lbs_across_executors`.

---

<a id="i-121"></a>
## I-121 `Memory::exprOriginMap` is a cold blob map — the last standalone heap origin map goes off the heap; mail stays string and the equi-class merge mints keys deterministically

**Scope.** Prover / visualizer / compressor / hashburst dump — `Memory::exprOriginMap` (now `TypedColdBlobMap<int64_t, IdOrigin>`, an `LbMemory` member aliased on `Memory`), its `Codec<int64_t>` key + `Codec<IdOrigin>` record codecs, the cold overloads of `addOriginId` / `addOriginEncoded` / `decodeOriginMapSorted`, and every read / write / merge site. Layered on [I-91](#i-91) (id-form) and the cold-map family ([I-98](#i-98)).

**The invariant.**

1. **The working container is cold, not heap.** [I-91](#i-91) interned the origin maps to id-form but left the container a heap `std::unordered_map`; Batch 5 moves `Memory::exprOriginMap` into the LB arena as a `TypedColdBlobMap<int64_t, IdOrigin>` (tags 451–454 — appended after the HashMemory band, the next free `LbMemory` tags). One key (packed `(expressionId, validityId)`) → a run of `IdOrigin` blobs; `Record = IdOrigin` serializes one history line (`OriginTag` + packed dep keys) byte-identical to `serializeEquivalenceClass`'s per-line stream. The equi-class `EquivalenceClass::equalityOriginMap` is NOT a separate container — it already rides the `equivalenceClassesMap` blob (Batch 3) and stays a transient heap decode, keeping the heap `IdOriginMap` helper overloads.

2. **Same observable semantics via cold twins.** The hot path reaches it through cold overloads of `addOriginId` / `addOriginEncoded` (RMW: decode the key's run, apply the EXACT [D-49](40_decisions.md#d-49) cap-full policy, `appendRecord` below cap / `assignRun` for the in-place convenience-slot swap) and `decodeOriginMapSorted` (walk ids `1..count`, decode, sort by decoded key). Overload resolution routes each call site by the map type — a site passing `mb.exprOriginMap` binds cold, a site passing a class's `equalityOriginMap` binds heap. The ad-hoc raw-API sites the heap container served (`.find` / `.swap` / range-for / `.count(k)` / `.at`) became `lookup` / `recordsAt` / an id-walk.

3. **Bulk key-mint is sorted.** The cold deload streams keys in insertion order, so the one bulk merge (the equi-class→body union in `updateEquivalenceClasses`) processes the class keys in SORTED order and rebuilds each key's run IN PLACE (`[class lines, then body lines]`, deduped, capped — the heap `merged:= classOrigins; overwriteOriginsId(merged, body)` order), so a class-only key is minted deterministically. The heap form relied on the dump's key-sort and did not need this; the cold deload's canonical-bytes contract ([I-103](#i-103)) does.

4. **Survives discharge and `wipeSubtree`.** `LbMemory::survivesDischarge` covers tags 451–454 — the post-prove chapter export reads origin history on discharged LBs (matching the pre-statification heap map, never emptied on teardown, [I-44](#i-44)). Reset to empty only at CE-clone teardown (`resetToFresh`). The `originInterner` its keys index resets in lockstep at `destroyGrid`.

5. **Mail stays string ([I-91](#i-91).4 unchanged).** `Mail::exprOriginMap` is a separate string-form member; `fillMailOut` decodes the cold body run into it and absorb encodes mail records back via the cold `addOriginEncoded`. No mail subsystem code changed — the encode/decode boundary already isolated mail from the body container's representation, so statifying the body did not need to touch mail.

**Why.** The origin map is the last standalone per-LB heap container in the statification campaign (the SwDD's "Batch 5"); moving it cold serves the all-static goal that gates FTA. Memory, not runtime, is the FTA bottleneck, so the cold RMW's added cost is accepted ([D-178](40_decisions.md#d-178)). Proof output stays byte-identical: the dump / export / compressor read `decodeOriginMapSorted` (which sorts), and per-key line order is preserved by the merge.

**How to spot violations.** A `.find` / `.swap` / range-for / `operator[]` on `exprOriginMap` (the cold surface has none — they will not compile); an unsorted bulk key-mint into it (nondeterministic deload); a mail container holding `IdOrigin`; a write that bypasses the cold `addOriginId` D-49 policy.

**How to fix on violation.** Route reads through `lookup` + `recordsAt` (or `decodeOriginMapSorted` for observable walks); sort before any bulk key-mint; keep mail string and convert at the absorb / fill boundary.

**Code.** Codecs + cold helpers in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp); the member / tags / `visitContainers` / `survivesDischarge` in [`memory_infra/lb_memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/lb_memory.hpp); `Codec<int64_t>` in [`memory_infra/typed_cold_map.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/typed_cold_map.hpp); the merge + probes in [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp) / [`prover.cpp`](../GL_Quick_VS/GL_Quick/src/prover.cpp); reads in [`visualizer.cpp`](../GL_Quick_VS/GL_Quick/src/visualizer.cpp); the header count in `infra/hashburst_dump.cpp` (`.count`). Unit tests: `codec_idorigin_round_trip_and_byte_identity` (`test_memory.cpp`), `int64_key_identity_and_blob_map` (`test_typed_cold_map.cpp`).

**See also.** [I-91](#i-91), [I-44](#i-44), [I-98](#i-98), [I-103](#i-103), `D-131` / [D-178](40_decisions.md#d-178).

---

<a id="i-94"></a>
## I-94 `MailLog` writes are single-threaded seams; the parallel phase-1 pull reads a frozen retained window and writes only its own cursor row

**Scope.** The pull-model mail system ([`mail_log.hpp`](../GL_Quick_VS/GL_Quick/src/mail_log.hpp); `MailLog::pull` in `performElemPhase1`; the commit barrier at `proveKernel`'s post-join seam). Established 2026-06-24 ([D-137](40_decisions.md#d-137)).

**Rule.** Every structural write to `mailBlobPool`, `mailRefs`, `mailHeads`, or `mailEdges` is single-threaded: registration at grid build, optional rolling retirement after phase 3 joins, and the following commit barrier — never during a parallel phase. The phase-1 pull only READS the frozen retained window and WRITES the recipient's own `mailCursor` cells (via `setValueAtRelaxed`) and its own `mailIn`. Every cursor cell is pre-created at registration, so the pull never inserts or rehashes. Retirement is separately gated by [I-161](#i-161).

**Why.** This is what makes the parallel pull race-free. Retained pages and chain links remain stable from commit through the next phase-1/2/3 sweeps; no retirement begins until all workers join. Read accessors and `Codec<Mail>::deserializeInto` are pure with respect to shared state, and the global mail interner is frozen during the pull. The only shared-container write is a disjoint cursor slot through `setValueAtRelaxed`, which skips the shared dirty flag. A commit or retirement during a worker phase, a pull-side insert, or a cursor advance through `setValueAt` would reintroduce a race.

**Spot.** A `commit` or `retireDeliveredBatches` call inside phase 1/2/3 worker execution; a cursor advance through `setValueAt`; a blob/ref/head append outside the commit seam; a pull that mints into the global interner; mail containers added to `LbMemory::visitContainers`.

**Fix.** Keep all batch-log writes at the single-threaded seams (register at grid build, commit at the post-join barrier); keep `MailLog::pull` read-only on `mailBatches`/`mailEdges` and write-only on the recipient's own `mailCursor` cells (via `setValueAtRelaxed`) + own `mailIn`; register every LB at grid build before any pull; keep the mail containers OUT of `visitContainers` (never deloaded/compacted, so vids stay stable). The statification re-established this discipline — see the mail-system chapter's [Statified race-safety](20_core_concepts/03_mail_system.md#race-safety) (the cold-read concurrency risk is now resolved).

**Code.** `mail_log.hpp::MailLog` (`registerLb`, `pull`, `retireDeliveredBatches`, `commit`, and the five cold containers); `prover.cpp::performElemPhase1`, `ExpressionAnalyzer::proveKernel`, and `analyzeExpressions` `buildGrid`. Unit tests: `test_mail_log.cpp::repeated_retire_pull_cycles_are_bounded`, `dormant_catch_up`, and `forced_two_level_spill_no_corruption`.

---

<a id="i-101"></a>
## I-101 The per-LB routing `mailIn` rides the never-deloaded MAIL POOL and returns its blocks immediately after phase-1 absorb

**Scope.** `Memory::mailIn`, `RoutingColdMail`, `MailLog::pull`, and `performElemPhase1`.

**Rule.** Each `mailIn` owns a lazily bound cold `LbArena` from `mailMemory`. Phase 1 claim/reloads the destination LB before pull writes global `mailInterner` ids into the inbox; after `standardProcessing` absorbs it under the same worker claim, `mailIn.clear` releases both columns and returns every arena block to the mail pool. It is never enrolled in `LbMemory::visitContainers` and never deloaded. Observable reads decode through the global interner.

**Why.** The inbox is a transient phase-1 staging surface, not retained history: it is filled and consumed under one worker claim and has no reason to remain resident afterward. Its self-owned arena gives `clear` an immediate whole-inbox block return instead of tying those blocks to the rest of the LB arena.

**Spot / fix.** Any live inbox after phase 1, sender-local ids in an inbox, or a global-interner mint from a parallel worker. Keep `mailIn` on its self-owned mail-pool arena, fill it only with global ids, and clear it after absorb.

**Code.** `routing_cold_mail.hpp::RoutingColdMail`; `memory.hpp::decodeMailIn*` / `routingMailInToHeap`; `mail_log.hpp::MailLog::pull`; `prover.cpp::performElemPhase1`.

**See also.** `I-163`, [I-94](#i-94), [I-127](#i-127).

<a id="i-102"></a>
## I-102 The two internal-mail channels go COLD — `ColdMail` on the LB's deloadable arena, not the HOT routing path and not the never-deloaded mail pool

**Scope.** `Memory::sameIterationInternalMail` / `nextIterationInternalMail` (reference aliases into `LbMemory::sameInternalMail` / `nextInternalMail`) and their backing struct `ColdMail` ([`memory_infra/cold_mail.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/cold_mail.hpp); `addInternalMailOrigin(ColdMail&, ValueInterner&, …)`, the `getDisintegrationSignal` read door, `kSameInternalMailDeloadBase` = 455 / `kNextInternalMailDeloadBase` = 505), the id-form mail value types ([`memory_infra/mail_types.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/mail_types.hpp)), the direct absorb in [`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)`::standardProcessing`, and the `makeHeapMail(const ColdMail&, …)` dump/test snapshot in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp). Established ([D-180](40_decisions.md#d-180)); the direct absorb ([D-142](40_decisions.md#d-142)); migrated string → per-LB id.

**Rule.** Internal mail's three live columns — `statements` (an `IntMailStatementKey`: `(originalId, validityId, levels)` in per-LB `NameMap` ids), `exprOriginMap` (`origins_`, an `int64`-keyed `IntMailOrigin` id blob run in the LB's `originInterner` space, with the D-49 cap via `addInternalMailOrigin`), and `disintegrationSignals` (an `int64`-keyed `uint8_t`, the two firing-time bools packed) — ride the cold-map family bound to the LB's **DELOADABLE** per-LB `LbArena` (`LbMemory::manager`) with the LB's REAL deload-`dirty`, enumerated by `LbMemory::visitContainers` at the two reserved bases so they deload / reload / discharge like any other cold container. Internal mail is produced AND consumed in the same LB, so per-LB ids suffice (no global interner — that is the routing mail's job, [I-127](#i-127)). `survivesDischarge` is TRUE for tags 455..554 — the heap never cleared internal mail at discharge, so the faithful mirror keeps them. `ColdMail` DROPS `expandedImplications` (always empty for internal mail) and CARRIES `disintegrationSignals` (the field the routing mailbox drops); `makeHeapMail` therefore repopulates `disintegrationSignals` (unlike `RoutingColdMail::toHeapMail`). Every observable order is re-imposed by `ExpressionWithValidity::operator<` at the read boundary — the absorb iterates `sortedStatements` / `sortedOrigins` directly and probes signals via `getDisintegrationSignal`; the sacred dump + tests materialize the same order through `makeHeapMail` (`sortedStatements` / `sortedOrigins` / `sortedDisintegrationSignals`). The deload bytes are insertion-ordered, kept deterministic by the sorted deposit boundaries (`applyFiringRecords`; the post-join `updateGlobal*` drain).

**Why.** Internal mail MUST survive deload: `nextIterationInternalMail` carries one-step-delayed emissions across the seam, and `sameIterationInternalMail` carries end-of-burst rewrites to the next step. The cold deloadable path is therefore its home, as it is now for routing `mailOut`; `mailIn` remains the separate transient mail-pool case whose self-owned arena is released immediately after absorb. The absorb reads the cold columns directly and re-imposes decoded ordering, so behaviour is byte-identical.

**Cross-LB residency — through the CLAIM-CORRECT DOOR.** The cross-LB deposit sites (`updateGlobalDirect` / `updateGlobal`) write a DIFFERENT LB's `nextIterationInternalMail`, which on the cold path may be deloaded. Each deposit goes through the uniform handshake `claimAndLoadForWork` (phase 4, the barrier bucket), holds `WorkerOwned` across the write, and releases `Idle` after — and SKIPS a `dischargedForever` target (it never drains its `nextIter` again — dead deposit; the reload asserts on a discharged LB). The door replaced the original bare `ensureLoaded`: a bare seam reload left the claim word `Dumped` on a resident arena — invisible to the pager's victim selection — and the deposit sweep ran with no pager window open, so residency grew monotonically to the 4 GiB wall (the forensic census: 1467 Idle-resident LBs / 14106 blocks + 290 Dumped-but-resident / 2146 blocks; [D-196](40_decisions.md#d-196)). The drains now run under the pinned-cursor barrier seam window, so recipients are evicted behind the next iteration's head while the deposits churn — recipient churn is correct behavior (the deposit lives in the arena and rides the raw image).

**Spot.** A `ColdMail` channel switched to `RoutingColdMail` (`nextIter` would lose cross-burst + cross-deload content) or to a never-deloaded pool (its content must travel with the LB); a `.statements` / `.exprOriginMap` / `.disintegrationSignals` member access on a channel (cold now — use the doors / `makeHeapMail`); a cross-LB deposit without the door + `dischargedForever` guard (residency assert, or the door's Dumped-but-resident assert); a future internal-mail writer depositing from a `std::unordered_*` / pointer-ordered source (breaks the canonical deload bytes); `makeHeapMail` leaving `disintegrationSignals` empty (would drop the firing-time OR-disintegration signals — `or0` facts un-disintegrated).

**Fix.** Keep the channels as `ColdMail` aliases into `LbMemory`; write via the doors (`insertStatement` / `addInternalMailOrigin` / `setDisintegrationSignal`, all id-form on the LB's own `NameMap` / `originInterner`); read via the id-form decode helpers + `getDisintegrationSignal` (the absorb) or `makeHeapMail` (dump / tests); guard every cross-LB deposit with `!dischargedForever` + the claim-correct door (`claimAndLoadForWork` + release, never a bare `ensureLoaded`); deposit only from sorted / positional sources.

**Code.** `memory_infra/cold_mail.hpp::ColdMail` (+ `addInternalMailOrigin`, `getDisintegrationSignal`, the deload bases; 8 deload facets — statements 2 + origins 4 + disintegrationSignals 2, was 10 before the id-form key drop); `memory_infra/mail_types.hpp` (`IntMailStatementKey` / `IntMailOrigin` id codecs + `ExpressionWithValidity` / `OriginLine` decoded-boundary types); `memory.hpp::makeHeapMail` (dump / tests) + the `Memory` aliases; the writers in `memory.cpp::applyFiringRecords`, `prover.hpp` (`sanitizeHashMemory`, `emitIntegrationRevivalToInternalMailIn`, `ordisMerge`, `dischargeToBeProved`), `prover.cpp` (`updateGlobal` / `updateGlobalDirect`); the direct absorb in `prover.hpp::standardProcessing`; `memory.cpp::wipeSubtree`; the sacred dump `infra/hashburst_dump.cpp::writeInternalMailIn`. See also [I-21](#i-21), [I-26](#i-26), [I-55](#i-55), [I-62](#i-62), [I-44](#i-44). Unit tests: `test_cold_mail.cpp` (incl. `get_disintegration_signal_read_door`), `test_lb_deload.cpp::internal_mail_coldmail_roundtrip`.

---

<a id="i-127"></a>
## I-127 — the global `mailInterner` is the cross-LB routing-mail id space, FROZEN during the parallel phase; mint only at single-threaded seams

**Scope.** `mailInterner` (defined in [`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp), declared in [`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp)) — a never-reset `ColdStringTable` (int32 ids) on the never-deloaded mail pool `mailMemory`; the commit seam `Codec<Mail>::serialize(const RoutingColdMail&, NameMap&, ValueInterner&)` / `deserializeInto` / `mergeBatchIntoMailIn` / `MailLog::pull`; the mailbox writers `insertStatement` / `addRoutingMailOrigin` / `mergeBatchIntoMailIn` (all `ensureArena` first). Migrated.

**Rule.** Committed cross-LB blobs and `mailIn` travel as GLOBAL `mailInterner` ids so a receiver can decode them independently of the sender. The interner is FROZEN throughout the parallel phase. Its mint sites are single-threaded seams: the post-join commit, load-time self-inject, and post-join drains. `mailOut` holds ids from its own per-LB `mailOutInterner`; commit claim/reloads the producer, decodes that private table, and mints global ids. Every mail-inbox writer `ensureArena`s before first insert ([G-55](50_gotchas.md#g-55)).

**Why.** This is the lock-free design the earlier racing shared-interner trials lacked. A `ColdStringTable` mint mutates its `ColdHashSet` index + `BytesKeyStore` (NOT parallel-safe); `decode` / `lookup` are pure const reads (ARE parallel-safe). Freezing the interner during the parallel phase and confining every mint to a single-threaded seam makes concurrent decode race-free without a lock — exactly as the `MailLog` blobs are already frozen for the parallel pull. int32 (not the `NameMap`'s int16) because one table spanning all LBs far exceeds the per-LB `MAX_NAME_IDS` ceiling.

**Spot.** A `mailInterner.intern(...)` — directly or via `serialize` / `mergeBatchIntoMailIn` / `addRoutingMailOrigin` — reachable from a parallel worker (`performElemPhase1/2/3`, `standardProcessing`, `fillMailOut`, the pull); an `intern` on a decode path; a `mailInterner` reset outside `destroyGrid`; a mail-inbox insert before `ensureArena`; a `mailIn` decoded via the sender `NameMap` or a `mailOut` decoded via the global `mailInterner`.

**Fix.** Keep mints at the three single-threaded seams; decode / lookup only in parallel code; `ensureArena` before any inbox insert; decode each id-space with its own interner (sender `NameMap` / `originInterner` for `mailOut`, global `mailInterner` for `mailIn`).

**Code.** `mailInterner` (`memory.cpp`); `Codec<Mail>::serialize(const RoutingColdMail&, …)` / `deserializeInto` / the `decodeMailIn*` / `decodeMailOut*` helpers (`memory.hpp`); `mergeBatchIntoMailIn` / `MailLog::pull` / `commit` (`mail_log.hpp`); the commit barrier + `broadcastTheorems` + `updateGlobal*` in `prover.cpp`; `fillMailOut` + the absorb in `prover.hpp`. See also [I-91](#i-91), [I-101](#i-101), [I-102](#i-102), [I-83](#i-83) (single-threaded mint discipline), [G-55](50_gotchas.md#g-55). Unit tests: `test_mail_log.cpp`, `test_memory.cpp` (`internal_mail_origin_cap_full_preference`).

---

<a id="i-109"></a>
## I-109 — LB node objects live in a non-relocating slab on a fourth never-deloaded pool; every `Memory*` stays address-stable for the object's life

**Scope.** The `LbStore` ([`memory_infra/lb_store.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/lb_store.hpp) / `.cpp`) and its backing pool `lbMemory` (`PoolKind::Lb`, the fourth `GlobalMemoryManager`), owned by `ExpressionAnalyzer` (`lbStore`). Established ([D-150](40_decisions.md#d-150)).

**Rule.** A `Memory` node object is constructed once into a fixed-size slot and NEVER relocated; the slot is recycled in place on destroy (intrusive free-list). The store draws `static_lb_block_bytes` blocks from the never-deloaded LB-body pool and carves each into `sizeof(Memory)`-sized, `alignof(Memory)`-aligned slots; within a store's life it never returns a block to the pool MID-run — a freed slot is recycled via the intrusive free-list — so the pool sizes to the peak simultaneously-live LB count; at TEARDOWN `releaseAll` (in `~LbStore`, mirroring `LbArena::~LbArena`) returns every block, so the several `ExpressionAnalyzer`s a process builds reuse the pool rather than summing their peaks. `create<Memory>` / `destroy` are the `new` / `delete` replacement; `allocateSlot` / `freeSlot` are mutex-guarded (the CE-clone path allocates on worker threads — the main grid is built single-threaded, "no LB is born after `buildGrid`"). Because slots never move, every raw `Memory*` the prover holds (`parentMemory`, the `SimpleMapStore` edge children, the `mailLog` `uintptr` keys, every `vector<Memory*>`) stays valid for the object's whole life — they stay raw pointers, NOT handles.

**Why.** `Memory` is non-copyable and non-movable, and the whole prover assumes pointer-stable LBs; a slab that constructs in place is the only container that fits both (a `std::vector<Memory>` reallocates, a `PagedVector<Memory>` demands trivially-copyable `T`). The pool is never deloaded because the shells are the always-resident LB directory (their bulky content already deloads via the per-LB arenas), so the store plays no role in any deload / throttle / steward decision (the mail-pool precedent). The intrusive free-list keeps the store off the heap.

**Spot.** A `new Memory` / `delete` that bypasses the store (heap shell, not pooled); a `std::vector<Memory>` / `PagedVector<Memory>` holding shells (relocation — `Memory` is non-movable, and a moved shell dangles every `Memory*`); the store returning blocks MID-run — only `releaseAll` at teardown may, when no shell is live (a reused slot's old `Memory*` would dangle); an `allocateSlot` / `freeSlot` outside the mutex during the CE-clone parallel phase (race); a raw `new Memory` / `delete` (or test scaffolding) that builds a grid node outside the store, then `destroyGrid` routes it through `lbStore.destroy` — the `freeSlot` underflow assert catches it.

**Fix.** Route every LB birth / death through `lbStore.create<Memory>` / `lbStore.destroy`; keep links raw `Memory*`; never relocate a slot; keep the slab on the never-deloaded `lbMemory` pool.

**Code.** `memory_infra/lb_store.hpp` / `.cpp` (`LbStore`); `memory_infra/global_memory_manager.hpp` / `.cpp` (`PoolKind::Lb`, `lbMemory` / `initLbMemory`); `parameters.hpp` (`static_lb_pool_bytes` / `static_lb_block_bytes`); `prover.hpp` (the `lbStore` member); `prover.cpp` (the ctor `initLbMemory`; the five grid `create` sites + the `destroyGrid` `destroy`). The allocation cutover also routes `compressor.cpp`, `filter.cpp` (CE template + the per-conjecture clone), `memory.cpp::cloneFactsTemplate(LbStore&)`, and `releaseCEBatchMemory` through the store. Unit tests: `test_lb_store.cpp`. See also [I-95](#i-95), [I-107](#i-107).

---

<a id="i-97"></a>
## I-97 — the LB identity `exprKey` is a 4-byte id into a shared never-deloaded skeleton interner; `exprKey` decodes byte-identically

**Scope.** `Memory::exprKeyId` + the `exprKey` / `setExprKey` accessors ([`memory.hpp`](../GL_Quick_VS/GL_Quick/src/memory.hpp), defs in [`memory.cpp`](../GL_Quick_VS/GL_Quick/src/memory.cpp)) and `skeletonInterner` (the shared `ColdStringTable` on `LbArena{ &lbMemory }`). Established ([D-151](40_decisions.md#d-151)).

**Rule.** An LB's expression key is stored as `int32_t exprKeyId` (0 = empty / root sentinel; ≥1 = an id in `skeletonInterner`). Reads go through `exprKey` (`decodeString`, returns "" for id 0) — byte-identical to the former `std::string exprKey` member — or the raw `exprKeyId` for id-equality only; writes go through `setExprKey` (interns; empty → 0). The interner is ONE process-wide table on the never-deloaded LB-body pool, so it is always resident — readable while an LB's main arena is cold and after discharge. `exprKey` ids are NEVER observable (always decoded back to the string at every print / compare / chain site), so the mint order — a pure function of grid-build order — is invisible to every proof artifact and dump.

**Why.** The identity string must survive deload AND discharge; per-LB never-deloaded storage would waste a 256 KiB block per LB; a shared deduped interner is compact and the `mailArena` / `lbStore` precedent. Statifying the identity off the heap is the Batch-2 step of the LB-body campaign.

**Spot.** A new `m->exprKey` field access (now a method — compile error C3867); a write `m->exprKey = x` (now `setExprKey` — C2659); a site comparing or sorting on `exprKeyId` instead of the decoded `exprKey` (id order is NOT lexicographic — a determinism break); the sacred hashburst dump emitting anything other than `exprKey` (byte drift); `decodeString` on a deloadable / discharge-reclaimed arena (the skeleton interner must stay on `lbMemory`).

**Fix.** Read via `exprKey` (string) or `exprKeyId` (id-equality only); write via `setExprKey`; sort / compare on the decoded string; keep `skeletonInterner` on the never-deloaded pool.

**Code.** `memory.hpp` (`exprKeyId`, `exprKey` / `setExprKey` decls, `skeletonInterner` decl); `memory.cpp` (the defs); every `exprKey` read + `setExprKey` write across `prover.cpp` / `prover.hpp` / `filter.cpp` / `compressor.cpp` / `memory.cpp` / `visualizer.cpp` / `infra/hashburst_dump.cpp` / `infra/rt_tracker.cpp`. The `simpleMap` routing edges followed ([I-110](#i-110)). Unit test: `test_memory.cpp::exprkey_interns_and_round_trips_through_skeleton`. See also [I-109](#i-109), [I-95](#i-95).

---

<a id="i-110"></a>
## I-110 — the LB-tree routing edges live in a never-deloaded `SimpleMapStore`, not a per-`Memory` `std::map`; keys interned, child pointers raw

**Scope.** `SimpleMapStore` ([`simple_map_store.hpp`](../GL_Quick_VS/GL_Quick/src/simple_map_store.hpp)) and the two `ExpressionAnalyzer` instances `simpleMapStore` (main tree) / `ceSimpleMapStore` (CE-filter tree) ([`prover.hpp`](../GL_Quick_VS/GL_Quick/src/prover.hpp)). The `Memory::simpleMap` member is removed. Established ([D-152](40_decisions.md#d-152)).

**Rule.** An LB tree's down-edge `parent --routing-key--> child` is one `EdgeNode { Memory* child; int32_t keyId; int32_t prev; }` in the store's append-only `edges` (`PagedVector`), with the parent's edges threaded newest-first through `prev` and headed by `heads` (`TypedColdMap<int64_t, EdgeHead>`, keyed by `lbKey(parent)` = the parent `uintptr`, NEVER dereferenced — Rule 12). The routing-key STRING interns into the shared `skeletonInterner` (a `keyId`, a DISTINCT slot from `exprKey` — they may differ, Rule 12); the child VALUE is a raw `Memory*` (an address-stable `LbStore` slot, dereferenced on navigation). `linkChild` (intern + append + head bump) runs single-threaded (LB birth, the `setExprKey` context); the parallel burst only READS via the non-minting `skeletonInterner.lookup` + frozen `edges` / `heads` (race-free, no lock). A `(parent, routing-key)` pair is linked at most once (the prover guards every insert with a prior `findChild` miss), so a parent's chain holds no duplicate `keyId`. `findChild` returns `nullptr` for an absent edge — a defined query result (the `find == end` twin), never a defensive fallback. `forEachChild` yields children SORTED by the decoded routing-key string (`compareSpans` over `skeletonInterner.view`), reproducing the old `std::map<std::string, Memory*>` iteration order byte-for-byte. The store is NEVER deloaded (no `ContainerTag`, no `visitContainers`, no `dirty`), so its pointer-valued keys / values and grant-order layout carry NO canonical-bytes obligation; proof output stays content-deterministic via the logical edge content plus the decoded-key sort. The two trees can coexist (CE filtering interleaves with proving), so each store is cleared at its OWN teardown — `simpleMapStore` at `destroyGrid`, `ceSimpleMapStore` at `filterConjecturesWithCE`.

**Why.** The edges are navigated regardless of an LB's deload state, so they cannot ride the deloadable per-LB arena; a never-deloaded subsystem keyed on `lbKey` (the `MailLog` model) is the fit. The per-parent back-linked chain takes interleaved cross-parent births in O(1) — a flat append-only CSR cannot, its `appendToTail` rejects an interior key. The child stays a raw pointer because `LbStore` slots are address-stable ([I-109](#i-109)), and the store is cleared before any child is freed. This statifies the last large per-LB heap user on the `Memory` shell, toward zero transitive malloc per LB.

**Spot.** A new `mb->simpleMap` access (the member is gone — compile error); a `findChild` result used without a null check where the old code branched on `it == end` (an absent edge is `nullptr`); a site sorting / comparing on `keyId` instead of the decoded string (id order is mint order, NOT lexicographic — a determinism break); a `linkChild` off the single-threaded LB-birth path (minting on a worker thread); a main-path site using `ceSimpleMapStore` or a CE-path site using `simpleMapStore` (the wrong tree); the store added to `LbMemory::visitContainers` or the deload stream (it carries no deload obligation and must not).

**Fix.** Route every down-edge through `linkChild` / `findChild` / `forEachChild`; keep child values raw `Memory*`; intern keys only single-threaded; iterate via `forEachChild` (decoded-key sorted); pick the store by tree (main vs CE); keep both stores off the deload path; clear each at its own teardown.

**Code.** `simple_map_store.hpp` (`SimpleMapStore`, `EdgeNode`, `EdgeHead`, `lbKey`, `linkChild` / `findChild` / `forEachChild` / `clear` / `empty`); `prover.hpp` (the two store members + their arenas); `prover.cpp` (`accessMemory`, `addTheoremToMemory`, `deactivateRecursively` / `deactivateUnnecessary`, `prefillIntegrationMapsRecursive`, `prehandleAnchor`, `activateZeroCondition`, `destroyGrid`); `filter.cpp` (`loadFactsForCEFiltering`, `filterConjecturesWithCE`); `visualizer.cpp` (the chapter-walk finds + the two filtered child scans). Unit test: `test_simple_map_store.cpp`. See also [I-109](#i-109), [I-97](#i-97).

---

<a id="i-126"></a>
## I-126 — `LbArena`'s block/page bookkeeping is pool-backed via a small-buffer `PtrDirectory`, never the heap; byte-transparent

**Scope.** `LbArena` ([`lb_arena.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/lb_arena.hpp) / `.cpp`) and `PtrDirectory` ([`ptr_directory.hpp`](../GL_Quick_VS/GL_Quick/src/memory_infra/ptr_directory.hpp)). Established (`D-188` in [`40_decisions.md`](40_decisions.md), re-opening [D-174](40_decisions.md#d-174)).

**Rule.** `LbArena`'s three pointer tables — the byte-bump block table (`blocks_`), the carved-block list (`pageBlocks_`), and the page table (`pageTable_`) — are `PtrDirectory<kInline>`, NOT `std::vector<char*>`. A `PtrDirectory` keeps its first `kInline` entries INLINE in the directory object (already pool-backed via the `LbStore` slab) and spills the overflow onto pool blocks drawn DIRECTLY from the `GlobalMemoryManager` (`acquireBlock` / `releaseBlock`), addressed by a small wired inline root `root_[kArenaDirRootCap]` — one level, no recursion; overflowing the root asserts (assert-not-recurse). The page free-list is an intrusive LIFO with the next-free link in each free page's LAST `sizeof(char*)` bytes (so `freePage`'s poison still covers offset 0, the use-after-free tripwire). `pop_back` / `truncate` RETAIN capacity; spilled blocks return only at `clear` (deload / teardown) or `shrinkToFit` (the `compactPages` coarse seam). The change is byte-transparent: arena bookkeeping is never serialized ([I-107](#i-107)), so vids / offsets and the deload bytes are unchanged. `GlobalMemoryManager`'s `granted_` / `recycled_` and the `compactPages` scratch stay on the heap (out of scope / a follow-up).

**Why.** "All of the LB goes static" (the MPU target) requires lifting `LbArena`'s last heap carve-out. The inline buffer is the decisive choice over a pure paged table: a pure-paged table costs +1 pool block per page-tier arena, which is pervasive (the per-worker scratch, routing-mail, and a typical `persistentArena` are all small page-tier arenas), exhausts tightly-sized pools, and pressures the never-deloaded persistent pool at FTA scale; with the inline buffer those small arenas cost zero pool block while only the large main deloadable arena spills. Retain-on-shrink keeps a scratch `rewind` / refill from churning pool grants, which would perturb the steward's monotone grant ledger ([I-106](#i-106)). `LbArena` is the substrate every container resolves through, so it is blanketed with a deep `assertInvariants` (Rule 19) — the maintainer's explicit requirement for this code.

**Spot.** A `std::vector<char*>` member back in `LbArena` (the heap carve-out returning); a `blocks_` / `pageBlocks_` / `pageTable_` access via `operator[]` on the hot `resolve` / `pageAt` path (use the non-asserting `peek`); a `pop_back` / `truncate` that releases a spilled block inline (must retain — only `clear` / `shrinkToFit` release); the free-list link placed at a page's START (overwrites the offset-0 poison tripwire); a directory growing past `kArenaDirRootCap` without the assert firing (a silent third level); arena bookkeeping appearing in the deload stream ([I-107](#i-107) — it must not).

**Fix.** Route the three tables through `PtrDirectory`; size `kInline` per role (`kArenaPageTableInline` 128 for the page table, `kArenaBlockTableInline` 16 for the block lists); keep the free-list link at the page end; retain on shrink, release at `clear` / `shrinkToFit`; assert the root cap (never recurse); keep arena bookkeeping out of the deload bytes.

**Code.** `ptr_directory.hpp` (`PtrDirectory`, `kArenaDirRootCap` / `kArenaPageTableInline` / `kArenaBlockTableInline`, `GL_ARENA_PARANOID`); `lb_arena.hpp` / `.cpp` (`blocks_` / `pageBlocks_` / `pageTable_`, `freeHead_`, `resolve` / `pageAt` via `peek`, `compactPages` + `shrinkToFit`, `assertInvariants`). Unit tests: `test_static_memory.cpp` (suite `ptr_directory`), `test_lb_arena.cpp`. See also [I-107](#i-107), [I-95](#i-95), [I-109](#i-109), [I-124](#i-124).

<a id="i-130"></a>
## I-130 — phase-2 request-generation scratch is heap-free

**Scope.** The phase-2 hashburst read path: `requestGatesPass` + `preEvaluateFromEncoded` (`prover.hpp`), `checkLocalEncodedMemoryStatic` (`memory.cpp`), the single request generator `generateEncodedRequestsStatic` (`memory.cpp`), and `StaticRequestEmitter` (`memory.hpp`). Established by the transient-statification campaign's Batch T1; the request keys + request-expr copies added by Batch T2.

**Rule.** None of the per-burst request-generation scratch is on the malloc heap. **Bounded** scratch uses stack arrays: `requestGatesPass`'s distinct-secondary dedup (`seenSecondary`, cap `maxNumberSecondaryVariables` — the predicate `preEvaluateFromEncoded` runs before its map probe, split out so the grow loop can build one key and probe two owner-set maps with it) and `checkLocalEncodedMemoryStatic`'s arg-membership set (`intAllArgs`). **Growable / dedup** scratch rides a SECOND per-slot scratch arena, `genScratchArenas` — distinct from the string-scratch `scratchArenas` ([I-124](#i-124)) — released per worker task at `performElem2`'s single exit alongside it. On that arena: the generator's DFS frontier (`generateEncodedRequestsStatic::stack`) rides the **byte-bump** tier as `ArenaStack` (`popTo` reclaim on backtrack, so the footprint tracks the live frontier, not total nodes); `baseCandidates` and `sortedStumps` ride the **page** tier as `PagedVector`; the emitted-request dedup `StaticRequestEmitter::seen` is a `ColdHashSet<BytesKeyStore>` keyed by the request's packed bytes (assembled on the stack). The two tiers are independent state on one `LbArena`, so the byte-bump stacks and the page-tier containers coexist without interference. **Batch T2 routes the last two heap users on this path onto the same byte-bump tier:** the normalized request keys (formerly a `thread_local` `new int16_t[]` chunk arena) and the `IntEncodedExpr` request-expr copies (formerly a per-thread `::operator new` arena), both via `alloc(bytes, alignof(T))` + `resolve`, persistent for the task (a `StaticRequest` points at them through `consume`) and freed by the same per-task `releaseAll`. They sit below the per-request `combinedLevels` mark, so `checkLocalEncodedMemoryStatic`'s function-exit `popTo` never touches them; the transient grow-DFS keys are dead after `ownerKeyAccepts` (a `BaseCandidate` keeps indices, not the key pointer) so the `ArenaStack` unwind reclaiming them is harmless, and the merge / seed / CE keys + copies are consumed synchronously inside `emit` so a later unwind frees only dead bytes. Byte-transparent: every replacement preserves content and order, and the per-task `releaseAll` nets grants to zero before each barrier, so the deload set and proof output are unchanged. The int/struct-side mirror of [I-124](#i-124).

**Why.** Statification moved the LB's storage *at rest* onto the static pool; the read path was the last per-burst minter — and the hottest, `seenSecondary`, ran millions of times per step. Killing it removes the residual transient heap and the malloc/cache tax on the hot path, a prerequisite for the all-static MPU model and FTA-scale memory.

**How to spot.** A `std::set` / `std::map` / `std::vector` / `unordered_set` local on the request-generation path; a `thread_local` reused heap buffer (still heap — defeats the purpose); byte-bumping the string-scratch arena `scratchArenas` (muddies the `ScratchString` `usedBytes` liveness tripwire); a `PagedVector` / `ColdHashSet` placed on the string arena (a `ScratchScope` rewind there frees its pages).

**How to fix on violation.** Stack array for bounded scratch; `ArenaStack` (byte-bump tier) for LIFO frontiers; `PagedVector` (page tier) for append-then-index lists; `ColdHashSet` for content dedup — all bound to `genScratchArenas.forSlot(coreId)`, released per task.

**Code.** `arena_stack.hpp` (`ArenaStack`); `scratch_arena.hpp` / `.cpp` (`genScratchArenas` / `initGenScratchArenas`); `prover.cpp` (init beside `initScratchArenas`, `releaseAll` at `performElem2` exit); `memory.cpp` (the generator); `memory.hpp` (`StaticRequestEmitter`). Unit test: `test_arena_stack.cpp`. See also [I-124](#i-124), [I-107](#i-107), [I-95](#i-95).

---

<a id="i-129"></a>
## I-129 — absorb-door interior scratch is heap-free

**Scope.** The single-threaded-per-LB absorb door (phases 1 & 3): `reconstructImplicationFullBind` (`prover.hpp`) as the batch opens; `addExprToMemoryBlock` / `addStatement` / the U-prefix/marker helpers as the transient-statification campaign's Batch T3 extends; `encExpr`/`encodedExpr`, `prefixArgumentsWithU`, and the `addStatement`-return copy as Batch T3b completes. What is minted inside another batch's function stays with that batch — the deferred items and their target batches are tracked in `transient_statification.md`.

**Rule.** The absorb door's call-local scratch carries no malloc heap. In `reconstructImplicationFullBind`, the former interior `std::vector<std::vector<std::string>>` (`chain` / `argsChain` / `whenRemoved`) and `std::set<std::string>` (`removedArgs` / `placed`) are fixed stack arrays of `StrSpan`: every argument name is a substring of the stable inputs `key` / `value`, so the spans — filled by the zero-allocation `getArgsSpans` — stay valid for the call. Set membership is a sorted `StrSpan` array + `binary_search` (`compareSpans`, byte-identical to the `std::set<std::string>` ordering) with a parallel `bool` for the first-occurrence `placed` dedup (it only ever holds `removedArgs` members, so the bsearch index is its key). The returned implication is the one `std::string` — the deferred-to-T8 boundary — built directly. The per-expression arg cap is `MAX_ARITY`; the chain cap is a fixed 64 (the measured max chain is 14 — a 13-premise implication — and `MAX_EXPRESSIONS` does NOT bound this premise `key`), asserted (Rule 19). The build-only `ce::getArgs` calls in `addExprToMemoryBlock` / `addStatement` (the `int_`/`it_` regex-assert args, matched span-wise via `std::regex_match`'s iterator overload; the `remainingArgs` set source; the equality `mirrored` builder) likewise read a stack `StrSpan[MAX_ARITY]` — the set / string they feed still materializes, blocked on its sink. The U-prefix/marker helpers `removeUPrefixFromArguments` / `makeMarkedExpr` drop their `std::map` + `ce::getArgs` for a stack `StrReplacement[MAX_ARITY]` (keys/values span the input / a literal) + the no-arena `replaceKeysToString` — a byte-exact, scratch-free twin of `ce::replaceKeysInString` whose result IS the returned `std::string`. (`prefixArgumentsWithU`, whose `"u_"+arg` values are new bytes, rides the per-slot scratch arena — see the Batch T3b paragraph below.) The origin deposits `exprVal` / `exprWithValidity` (`addExprToMemoryBlock` / `addStatement`) drop the transient `ExpressionWithValidity` — the key is interned from `StrSpan`s over the stable input via additive `ValueInterner::encode(StrSpan)` / `mintOriginKey(StrSpan)` / cold `addOriginEncoded(StrSpan)` overloads (byte-identical ids/keys; the `origin` history record stays string, the deferred mail/EWV boundary). Byte-transparent: content and order preserved.

**Batch T3b** completes the door. `encExpr` / `encodedExpr` — `EncodedExpression` instances built only to feed `encodeExpression(ee, nm)` — are replaced by the span overload `encodeExpression(StrSpan original, StrSpan validity, NameMap&)`, which parses + encodes in one pass (`getArgsSpans` + the additive `NameMap::encode(StrSpan)`) and computes the int fields by arithmetic (the struct path's materialized `to_string(lev+1)` was immediately `atoi`'d back → `lev+1`); byte-identical `IntEncodedExpr`, no nested `vector<vector<string>>`, no `ce::getArgs` vector. `prefixArgumentsWithU` drops its `vector<string>` + `map<string,string>` for a stack `StrSpan[]` + `StrReplacement[]` + `replaceKeysToString`, with its `"u_"+arg` NEW bytes built on the per-slot scratch arena (`scratchArenas.forSlot`, rewound by a `ScratchScope`); the slot is published by a `thread_local g_currentCoreId` set at the phase-1/3 worker entries (the door sits frames below them without `coreId`; mirrors the RT tracker's `g_currentThreadTracker`), and single-threaded setup reads -1 → a reserved slot (`initScratchArenas(logicalCores + 1)`). The `addStatement` return is sorted in place in the door (the `sortedNew` deep copy dropped; EWV `operator<` is total).

**Residuals.** The absorb-door items this batch did not statify, and their target batches, are tracked in `transient_statification.md` (the campaign doc) — not duplicated here.

**Why.** The absorb door runs per absorbed expression on the single-threaded phase-1/3 path; `reconstructImplicationFullBind` alone previously minted two vector-of-vectors, two `std::set<std::string>`, and per-link `std::string`s, plus a `ce::getArgs` heap vector per chain link. Killing the call-local heap removes the malloc tax on the absorb path — the last per-burst minter outside the parallel phases — a prerequisite for the all-static MPU model.

**How to spot.** A `std::vector` / `std::set` local on the absorb path whose contents are substrings of an input; `ce::getArgs` where `getArgsSpans` (writes into a caller `StrSpan[]`) would serve; a transient `std::string` / `ExpressionWithValidity` built only to hand to an interner.

**How to fix on violation.** Stack `StrSpan` arrays (fixed chain cap + `MAX_ARITY`, asserted) filled by `getArgsSpans`; sorted-array + `binary_search` for membership; build the one escaping `std::string` directly.

**Code.** `prover.hpp` (`reconstructImplicationFullBind`, `addStatement`, `makeMarkedExpr`, the `g_currentCoreId` declaration), `prover.cpp` (`addExprToMemoryBlock`, `removeUPrefixFromArguments`, `prefixArgumentsWithU`, the `g_currentCoreId` definition + the phase-1/3 publication + `initScratchArenas(logicalCores + 1)`), `str_ops.hpp` (`replaceKeysToString`), `memory.hpp` (`NameMap::encode` / `ValueInterner::encode` / `mintOriginKey` / `addOriginEncoded` `StrSpan` overloads, the span `encodeExpression` overload). Unit tests: `test_reconstruct_implication.cpp` (`origin_mint_byte_identical`, `prefix_arguments_with_u`), `test_memory.cpp` (`namemap_encode_strspan_byte_identical`, `encodeexpression_span_twin_byte_identical`), `test_str_ops.cpp` (`replace_keys_to_string_twin`). See also [I-130](#i-130), [I-131](#i-131), [I-124](#i-124), [I-95](#i-95).

---

<a id="i-131"></a>
## I-131 — the firing-record staging structs are heap-free POD

**Scope.** The phase-2 firing capture + finalize path: `FiringRecord`, `StagedAdmissionValue`, `DeferredIntegrationPrep`, `AdmissionKeyAlgebraRecord` (`memory.hpp`); the producer `checkLocalEncodedMemoryStatic` and the consumer `applyFiringRecords` (`memory.cpp`). Established by the transient-statification campaign's Batch T2.

**Rule.** No firing record carries a heap container. The former spines — `FiringRecord::levels` (`std::set<int>`), `originDeps` / `markerArgsSorted` (`std::vector`), and `StagedAdmissionValue::key` / `remainingArgsSorted` + `DeferredIntegrationPrep::unchangeableArgsSorted` (`std::vector<SealedString>`) — are now `SealedSpan<T>` views over runs bump-allocated onto the same per-task `SealedPageSet` (D-164) that already holds the record's strings, so every record is trivially copyable POD. The producer assembles each run heap-free before sealing: **bounded** runs ride stack buffers (`originDeps` ≤ 1+`MAX_EXPRESSIONS`+2, `markerArgsSorted` ≤ `MAX_ARITY`, `sealedPremises`); **unbounded** runs ride the slot's gen-scratch arena (`genScratchArenas.forSlot(coreId)`, the [I-130](#i-130) arena) on the **byte-bump tier** via `alloc(bytes, alignof(T))` + `resolve` — `combinedLevels`, the per-firing level set, `admv.key`, `remainingArgsSorted` — all reclaimed in one `popTo` at the single function exit (no early return follows the mark; the byte-bump scratch sits below the mark and never interferes with T1's page-tier `PagedVector`s, the independent tier). The consumer reads the spans directly; the canonical sort ([I-77](#i-77)) compares them element-wise (`compareSpans` / ascending int-span lex — byte-identical to the former `std::set`/`std::vector` orders). The level set materializes into a `std::set<int>` only at the single-threaded `insertStatement` deposit — the deferred mail/NameMap boundary, statified by a later mail batch. Byte-transparent: content and order preserved.

**Why.** `checkLocalEncodedMemoryStatic` runs per firing — the hottest deposit path; each head/marker firing previously minted up to five heap containers (the level set + the four sealed-view vector spines). Killing them removes the per-firing malloc, a prerequisite for the all-static MPU model and FTA-scale memory. The int/struct-side peer of the string sealing (D-164).

**How to spot.** A `std::set` / `std::map` / `std::vector` member on `FiringRecord` or a staging struct; a build buffer on the heap in `checkLocalEncodedMemoryStatic`; a `SealedSpan` built without a `popTo`-bracketed gen-scratch window. (Historic note: `allocBytes` was once a page-tier fill that collided with `allocPage` containers; it is now a resolved byte-bump `alloc` — the byte tier, independent of the page tier — so scratch fill and paged containers never collide, see D-189.)

**How to fix on violation.** Stack buffer for bounded runs; gen-scratch byte-bump `alloc(bytes, alignof(T))` + `resolve` for unbounded runs, sealed via `SealedSpan<T>::copyFrom` and reclaimed by one function-exit `popTo`.

**Code.** `sealed_pages.hpp` (`SealedSpan<T>`); `memory.hpp` (the structs); `memory.cpp` (`checkLocalEncodedMemoryStatic`, `applyFiringRecords`). Unit test: `test_sealed_pages.cpp` (`SealedSpan`), `test_lb_split.cpp` (the canonical sort). See also [I-77](#i-77), [I-130](#i-130), [I-124](#i-124), D-164.

<a id="i-132"></a>
## I-132 — disintegrateExpr2's accumulators ride a page-tier interner, never byte-bump

**Scope.** `ExpressionAnalyzer::disintegrateExpr2` / `disintegrateExprCore2` (phases 1 & 3), Batch T4 of the transient-statification campaign — established for `collected` and `orBranchStatements`.

**Rule.** The disintegration's EWV-bearing accumulators are off the malloc heap as `CollectedArena` (`prover.hpp`): a `PagedVector<CollectedRec>` of interned-id records (`keyId`/`aId`/`bId` + `kind` 0 impl / 1 child / 2 empty-key) over a `ColdHashSet<BytesKeyStore>` byte interner, BOTH on the per-slot scratch arena's PAGE tier. Keys compare by interned id (the interner dedups, so id-equality == byte-equality); values `decode(id) → StrSpan`. Order is not observable (everything funnels into the final `std::set`s; `addToFinal` is idempotent), so records append raw. `collected`, `orBranchStatements`, and `admittedVars` ride the per-slot **`genScratchArenas`** arena (the page-tier container arena), NOT the `scratchArenas` string arena — see constraint 1. The leaf scratch (`getArgsSpans` + the `matchItLevId`/`matchIntLevId` regex twins + the cascade level `char[]`), `admittedVars` (transient `ColdHashSet`), and the single-entry renaming maps (stack `StrReplacement` + `replaceKeysToString`) are likewise heap-free. `IntEwv` is NOT used — a raw-byte interner, not NameMap ids, so the NameMap dump (I-105) is not reordered.

Two hard constraints this batch surfaced (each cost a gate cycle):

1. **`LbArena::allocBytes` (page-tier scratch FILL) must not share an arena with any live `allocPage` container.** `allocBytes` is NOT the byte-bump tier — it owns the page-table tail (`pageAt(pageHighWater-1)`) plus one arena-wide `scratchWithin_`, and its `ScratchScope` rewind frees and `0xCD`-poisons that tail page. A page container's `allocPage` pushes the tail out from under it, so the next `allocBytes` writes into — and rewind poisons — the container's page. The byte-bump tier (`alloc`, `cursor_`/`blocks_`) and the page tier (`allocPage`, `pageTable_`/`pageBlocks_`) ARE independent and coexist freely (a per-LB arena uses both at once); the conflict is `allocBytes`-vs-`allocPage`, not byte-bump-vs-page. Two manifestations, same rule: (a) the first `CollectedArena` stored strings via `allocBytes` alongside its `PagedVector` records and segfaulted on read — fixed by moving strings to the page-tier `ColdHashSet` interner; (b) `collected` was then bound to `scratchArenas` (where `prefixArgumentsWithU` does `allocBytes`), so the re-entrant integration / hypothetical / equality-necessity paths re-entered `prefixArgumentsWithU` while an outer `collected` held pages and poisoned its tail — Gauss `collected` records silently corrupted, the dropped statements failed to match, and the theorems vanished with no crash (membership-only `admittedVars` masked the identical clobber because its bytes are never decoded as content). Fix: the accumulators ride `genScratchArenas` (the container arena), never `scratchArenas`. This NARROWS the T1 "two tiers coexist" claim ([I-130](#i-130)): `ArenaStack`(byte-bump `alloc`)+`PagedVector` coexist; `allocBytes`+`PagedVector` did NOT. **Since structurally eliminated:** `allocBytes` was retired to a resolved byte-bump `alloc` ([D-189](40_decisions.md#d-189)), so it can no longer touch the page tier — `allocBytes`+`PagedVector` now coexist freely, and the `genScratchArenas` move above is belt-and-suspenders rather than load-bearing.
2. **Never `StrSpan(rawCharArray)`** — with no `const char*` ctor it picks `StrSpan(const std::string&)`, builds a temporary `std::string`, and the span dangles past the statement (it masked the `collected` migration as a Peano `isInt` divergence). Always pass the explicit length: `StrSpan(buf, len)`.

**Batch T5 finishes the function** — the remaining accumulators ride the same `genScratchArenas` page tier, byte-transparent: `finalStringStatements` → `ColdHashSet` (its one order-sensitive site, the forceDeep `canBeSentIds` mint, sorts a byte-bump id snapshot to keep the I-105 NameMap mint order); `newVarMap` → `NewVarStore` (`ColdMultiMap` var→elem-id run + interner, decoded-lex `forEachVarSorted` for the order-sensitive Pass B); `existenceGroups` → two `ColdHashSet`s (`allSigs` / `coveredSigs`, full disintegration ⟺ equal counts); `pendingRejections` / `pendingRejectionsIntegration` → one `RejectionStore` (append-only interned-id records drained in Pass-B = lex order). `expandSignature` is statified too — a flat `collectExprTokens` bracket scan replaces the `ce::parseExpr` walk (the shared compiler parser untouched). New `ScratchString` twins: `removeUPrefixScratch` / `addMissingUScratch` / `reconstructImplicationFullBindScratch` / `makeMarkedExprScratch`. Only the two root-type batches remain — `instructions` / `LogicalEntity`, and the RETURNED `set<ExpressionWithValidity>` finals (`finalImplications` / `finalStatements`, the T10 EWV root-type); `prepareIntegrationCore`'s `replacementMap` is LogicalEntity-coupled and rides with that batch.

**Code.** `prover.hpp` (`CollectedArena`, `NewVarStore`, `RejectionStore`, the four scratch twins); `prover.cpp` (`disintegrateExpr2` / `disintegrateExprCore2` / `expandSignature`); `memory_infra/str_ops.hpp` (`matchItLevId` / `matchIntLevId` / `collectExprTokens`). Unit tests: `test_str_ops.cpp` (`collected_arena`, `match_lev_id_twin`, `disintegrate_scratch_twins`, `new_var_store`, `rejection_store`, `collect_expr_tokens_twin`). See also [I-129](#i-129), [I-84](#i-84).

---

<a id="i-133"></a>
## I-133 — the integration path's working Instruction/LogicalEntity rides an arena-backed interned-id form

**Scope.** The integration path (`prepareIntegration` / `prepareIntegrationCore` / `prepareIntegrationCore2`, the admission/rejection-integration probes and the equi-class hooks) — the transient-statification campaign's integration arm, the "T10 EWV root-type" batch that [I-132](#i-132) deferred (the disintegration path left the working `Instruction` / `LogicalEntity` on the heap). The foundation (the type + codecs) lands first, unwired; the per-function wiring follows batch by batch.

**Rule.** The transient WORKING `Instruction` / `LogicalEntity` the integration path builds, rewrites, and reads each burst is off the malloc heap as `WorkInstruction` ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)): a `ColdHashSet<BytesKeyStore>` that interns every string field, a `PagedVector<WorkLogicalEntity>` of flat POD records (`catId`/`sigId`/`dsId`/`arity` plus the half-open `[elemsStart, elemsStart+elemsCount)` element-id run), and a `PagedVector<int32_t>` element column — all on ONE per-slot `genScratchArenas` arena (the disintegration accumulator idiom, `CollectedArena`/`RejectionStore`). Build is append-only (`elemMark` → `addElement` → `commitEntity`, plus `setMarkedGoal`); the path's one rewrite (`prepareIntegrationCore2`'s count-preserving field substitution) rebuilds into a FRESH `WorkInstruction`, never mutates in place (the records are read-only after commit). The cold boundary `encodeWorkInstruction(wi, vi) → IntInstruction` / `loadFromIntInstruction(wi, ii, vi)` is byte/id-identical to the heap `encodeInstruction` / `decodeInstruction` (the SAME `ValueInterner`, so a load-then-encode reproduces the exact ids), the sanctioned `IntInstruction` materialization handed to the already-cold `admissionMapIntegration` store. The string `LogicalEntity` / `Instruction` survive only as the boundary/decode types (the sacred hashburst dump's `compiledExpressions`; the cold codecs); the WORKING form is the arena twin.

The same arena constraints as [I-132](#i-132) apply: page containers ride `genScratchArenas`, never the `scratchArenas` string arena; never `StrSpan(rawCharArray)`; every observable ordering sorts a decoded id snapshot ([I-84](#i-84)).

**Status.** B1 landed the foundation (the `memory.hpp` type + codecs + the `entityAt` per-iteration shadow materializer). B2 wired the object swap across `prepareIntegration` / `prepareIntegrationCore` / `prepareIntegrationCore2`: the heap `Instruction` is gone, the working entities ride a `WorkInstruction` on the per-slot `genScratchArenas`, and the marker admission key encodes through `encodeWorkInstruction`. The 600-line `prepareIntegrationCore2` body stays byte-identical — block 1's former in-place rewrite is a rebuild into a fresh `WorkInstruction` (records are read-only after commit). B2 read each entity through a per-iteration `entityAt` LogicalEntity shadow; **B3 retired those shadows** — the three loops (`allowedRepeatedVars`, block 1, block 2 incl. the marker copy) read the `WorkInstruction` span accessors directly, so NO `LogicalEntity` object remains in `prepareIntegrationCore2`. B3 is field materialization (the span bytes copied into same-valued `std::string`/`std::vector` locals), byte-identical: nearly every entity-field use feeds a `std::string`/`vector`-taking boundary (the heap helpers `renamingChain2` / `expandSignatureForIntegration` / `buildIntegrationInstruction` / `addToHashMemory`, the deposits, the origin records), so the boundary-bound leaf heap remains (its removal needs reworking those helpers, a separate concern); `findAllUArgs` took `(signature, elements)` instead of a `LogicalEntity`. `entityAt` survives only for the disint-side `disintegrateExprHypothetically` bridge. **B5 then retired the two decoded-caller bridges** (`isAdmittedIntegration` / `updateAdmissionMapIntegration`): they now `loadFromIntInstruction` the snapshot `IntInstruction` straight into a `WorkInstruction` and clean it via `cleanInstructionWork` — a heap-free `WorkInstruction` twin of the (now-deleted) heap `cleanInstruction`, doing the same first-match-erase + recursive-unused-element drop as a span-compare rebuild, unit-verified equivalent — so no heap `Instruction` / `decodeInstruction` / `encodeInstruction` round-trip remains there. The disint-side `std::vector<LogicalEntity>` bridge (built in `disintegrateExpr2` to feed `disintegrateExprCore2`) was DELETED in the cycle-interface batch: `disintegrateExprCore2` now takes `const WorkInstruction&` directly and materializes ONE `LogicalEntity` per matched entity via `WorkInstruction::entityAt(hit)` at the read sites (the `trackExpansionHistory` / `expandSignature` / `listLastRemovedArgsLE` boundaries), so no full heap `std::vector<LogicalEntity>` remains — the per-hit materialization + the `processPath` fresh-`WorkInstruction` rebuild leave `disintegrateExprCore2` row-false pending the innards batch (which removes the per-hit `LogicalEntity` via a span/Work `expandSignature` overload). **B7 / B8 then statified the equi-class hooks' dominant per-call heap** — the decoded `(template,validity)` snapshot (`rmiRows` in `applyEquivalenceClassToRejectedMapIntegration`, `amiRows` in `applyEquivalenceClassToAdmissionMapIntegration`) rides a per-slot `genScratchArenas` `ColdHashSet` interner + a byte-bump `compareSpans` index-sort (the `forEachVarSorted` idiom; the `pk ↔ (template,validity)` bijection means no ties, so the order is byte-identical). **B8 removed the last heap `LogicalEntity` from the integration hooks**: `origInstr` loads via `loadFromIntInstruction` into a `WorkInstruction`, the `augSubst` rewrite builds a fresh `WorkInstruction`, and the `toInsert` staging record holds the `IntInstruction` (`encodeWorkInstruction`) instead of a heap `Instruction`. The hooks' boundary-bound / ordering-critical transients (`substMap` / `augSubst`, the `toMail` staging, `uniqueCompounds`) stay heap. **B4 — the `IntegrationEntryMap` snapshot — was STATIFIED** onto `ArenaIntegrationMap` (memory.hpp): flat int-POD `PagedVector`s (entries + `ArenaIntLE` records + an element column + append-only per-entry value chains) on the per-slot `genScratchArenas`. Since `IntInstruction` is interned (one id per decoded form), `DecodedInstructionLess`-equivalence == id-equality, so `findOrEmplace` is a plain id-compare (no decode); the `DecodedIdLess` / `DecodedInstructionLess` ORDERS are imposed at read/flatten via a `valueIdLess` / `compareEntries` index-sort — byte-identical to the heap `flattenIntegrationEntryMap`. All five consumers thread onto it, including the snapshot-copy + mutate-while-iterate in `isAdmittedIntegration` / `updateAdmissionMapIntegration`: a stable `sortedIndices` snapshot before the loop, a live `countValue` (equal to the heap's frozen-snapshot count because each entry is mutated only at its own iteration), and the re-entrant `prepareIntegrationCore2` builds its OWN map at a different pk so LIFO page reclaim keeps the loop's map intact. Unit-tested (`arena_integration_map_matches_heap`: flatten / find-or-emplace / sorted-value-insert / post-mutation parity to the heap `std::map`, plus the B4b cold round trip). **B4b then made the two cold-STORE I/O boundaries DIRECT**: `ArenaIntegrationMap::buildFromCold` deserializes each cold record's canonical bytes straight into the arena `PagedVector`s through the substrate's no-allocation `peekRecordBytes` read door (zero-copy single-page, the owner-set-prune lever; `scratch` only on a page straddle), and `ArenaIntegrationMap::writeToCold` serializes the `DecodedInstructionLess`-sorted arena entries straight into the blob run through the engine's raw `assignRun(KeyView, bytes, lens, M)` byte door (reached via the typed wrapper's `inner` escape hatch; `Codec<int32_t>::view(pk) == pk`, so the call is identical to the typed `assignRun`'s own forwarding). No intermediate heap `IntegrationEntry` / `std::vector<IntegrationEntry>` crosses either boundary — only the flat `bytes` / `lens` buffers the typed `assignRun` itself builds. Both paths mirror `Codec<IntegrationEntry>::serialize`'s field order EXACTLY and are byte-identical to the former `recordsAt` / `flatten`→`assignRun` cycle (the B4b cold round-trip unit test peeks both maps' record blobs and asserts `memcmp == 0`, then asserts `buildFromCold` reconstructs the same `flatten` run). **No substrate edit was needed** — `peekRecordBytes` + `runLen` + `lookup` (read) and `inner.assignRun` (write) already exist; the change is additive in `memory.hpp` only. The deletion sweep found **no fully-orphaned symbol**: the heap-form `build` / `flatten` / `flattenIntegrationEntryMap` survive as the test golden references the direct paths are checked against; `admissionIntegrationRecordsAt` (returning the heap `IntegrationEntryMap`) stays as the SACRED hashburst dump's read path (Rule 14 — a sanctioned debug-time boundary, never a prover hot path); `IntegrationEntry` stays the cold-store record type + `Codec` layout. **S10 C1 closed the last `ArenaIntegrationMap` heap surface**: the two heap-returning read methods `sortedIndices` / `valuesAt(ei)` gained caller-fill `(out, cap)` twins that write the identical `DecodedInstructionLess` entry-index / `DecodedIdLess` value sequence into a caller stack run (caps `MAX_INTEGRATION_ENTRIES` / `MAX_INTEGRATION_ENTRY_VALUES`, loud Rule-19 STOP-widen asserts, page-tier `PagedVector` widening path). All five call sites thread onto the fill forms — the internal `writeToCold` plus the four external snapshot sites (`isAdmittedIntegration` / `updateAdmissionMapIntegration` / `applyEquivalenceClassToAdmissionMapIntegration`) — while the heap `sortedIndices` / `valuesAt` / `flatten` survive as `flatten`'s reference path + the twin-test oracles.

**S10 C2 — `prepareIntegrationCore` span-native + the chain-rename / u_-collection twins.** `prepareIntegrationCore` gained an additive `StrSpan` real body (a `const std::string&` delegator forwards) that runs its signature→expression argument map as a `StrReplacement` run over `getArgsSpans` spans, mints the `existence` `pi_lev_*` fresh vars on the string-tier arena (first-seen dedup via a linear span-scan, `startIntPi` order frozen), rewrites via `replaceKeysScratch`, and self-recurses on spans over its arena-built new elements (each nested `ScratchScope` nests above the parent's element spans, byte-bump LIFO — no heap `std::string`/`std::map`/`std::vector` remains). `findAllUArgs`'s interior became `getArgsSpans` + span `u_`-check (the RETURN stays the `std::set<std::string>` `addToHashMemory` edge form, materialized only at the per-hit `insert`). `renamingChain2Scratch` landed as the byte-twin of `renamingChain2` (unwired in C2; Core2 Cases A/OR wire it in C4/C5) — the same `u_`-strip-slice / fresh-`repl_lev_*` logic with the `startIntRepl` progression frozen; mints nothing. Named cap `MAX_INSTRUCTION_ELEMENTS` sizes the element-run stack buffers. Twin tests: `renaming_chain2_scratch_matches_vector` (test_str_ops.cpp); the `prepareIntegrationCore` / `findAllUArgs` interior conversions are byte-checked by the batch gate.

**S10 C3 — the integration fresh-expression builder scratch twins (unwired; Core2 §C4/§C5 wire them).** Under the J1 ruling (user 2026-07-03: convert the RESULTS too — the fresh-expression builders' final strings become arena builds end-to-end, heap ONLY at the `addToHashMemory` boundary) the four integration builders `expandSignatureForIntegration` / `stripUPrefixAST` / `buildIntegrationInstruction` / `reconstructImplicationForIntegration` gained byte-twin scratch forms (`expandSignatureForIntegrationScratch` / `stripUPrefixASTScratch` / `buildIntegrationInstructionScratch` / `reconstructImplicationForIntegrationScratch`), plus `negateScratch` (the `negate` slice-or-`prefixBang` twin — `negate` KEEPS live production callers beyond Core2 Case OR: `prover.cpp` `reconstructImplication`/`expanded` sites + `prover.hpp` the OR-branch/negation-lookup sites) and `sortOriginalChainIndex` (the shared decoded-lex index over the `originals` rule registry replacing the `originalChains` snapshot + `std::sort`, consumed by Core2 Case B + `checkNecessityForEquality`). The static-parse path is `collectExprTokens` (the registered T5 `ce::parseExpr` token-walk twin — no arena tree parser exists or is needed); `reconstructImplicationFullBindScratch` (also T5) is REUSED, not re-implemented. All six mint NOTHING; the heap forms survive as the twin-test oracles. Named cap `MAX_EXPR_TOKENS` sizes the `stripUPrefixASTScratch` token runs. Twin tests: `integration_builder_scratch_twins` (the five builders) + `sort_original_chain_index_matches_std_sort` (test_str_ops.cpp).

**S10 C4+C5 — `prepareIntegrationCore2` interior span-native + the map door (landed as ONE commit).** The `std::map<std::string,std::string>& replacementMap` param became a `const StrReplacement*` run + count (the map door); the three callers — `prepareIntegration` (the multi-entry `pi_lev_*` `changeableMap` → a `changeablePairs` run on the widened `uScope`, `startIntPi` order frozen), `isAdmittedIntegration` + `updateAdmissionMapIntegration` (the one-pair `"marker" -> "u_"+var` map → a `StrReplacement[1]` on the caller's held arena) — cascade in the same commit; the closed `S10 EDGE (ledger)` comments are deleted (Rule 23). The whole Core2 interior is span-native under ONE function-level `ScratchScope` (the existing per-record origin scopes `gScope`/`peScope`/`igScope`/`ogScope`/`oaScope` nest inside and rewind only their own allocations, so the case-level strings survive — byte-bump LIFO): `makeMarkedExprLambda` / the `allowedRepeatedVars` prescan / the rewrite loop / the case headers (`leSignature`/`leCategory` → spans, `leElements` → a span run into the stable `rewritten` WorkInstruction) all run on `getArgsSpans` + `StrReplacement` + `replaceKeysScratch`; Case A wires `renamingChain2Scratch` + `expandSignatureForIntegrationScratch`; Case OR wires `negateScratch` + the arena `(&…)` / `orint_` / `(>[]…)` builders; Case B wires `sortOriginalChainIndex` (the existence `originalChains` snapshot → a byte-bump `int32[]` index on `genScratchArenas`, each chain decoded per `makeAdmissionKeys` call into a reused vector) + `expandSignatureForIntegrationScratch` + `buildIntegrationInstructionScratch`; Case C wires the marker-rewrite `StrReplacement` + `removeUPrefixScratch`. **The ONLY heap materialization is at the `addToHashMemory` boundary** (one `std::string` `leSignatureStr` + `std::vector<std::string> leElementsVec` + the two instruction strings, built once per li) and at the genuinely-`std::string`-only sinks (`findAllUArgs`, `makeAdmissionKeys`, `revisitRejectedIntegration2`, and `encodePush` — whose payload is a fresh `std::string` built inline this batch (`encodePush` itself HAS a `StrSpan` overload, `memory.hpp · encodePush(int16_t, const StrSpan&)`, so the innards batch builds the payload on the string arena and feeds it directly): byte-identical, no mint reorder). Whole-hook conversion — no standalone twin; the batch gate is the byte-identity proof (09c whole-hook doctrine).

**S10 C6 — the equality doors (`checkForEquivalence` + `checkNecessityForEquality`) span-native + `prefixArgumentsWithUScratch`.** `prefixArgumentsWithUScratch` landed as the byte-twin of `prefixArgumentsWithU` (prefixes EVERY arg — contrast the marker-skipping `prefixNonMarkerArgumentsWithUScratch`; an already-`u_` arg is prefixed again, NOT special-cased; mints nothing; test `prefix_arguments_with_u_scratch_matches_string`). `checkForEquivalence`'s interior went span-native: `args` → `getArgsSpans`; the `possibilities` `vector<vector<string>>` → per-arg RANGE descriptors over the ALREADY-landed member-run snapshot (`possStart`/`possLen`/`possSelf`/`possSelfSpan`); the Cartesian `replacements` map + `ce::replaceKeysInString` → a per-tuple `StrReplacement` run whose class-member value is a zero-copy `decodeView` (SAFE — the only in-loop interner touch is the non-minting `lookupStatementFlags` probe, so no NameMap mint intervenes, I-3 / 09c §1) + `replaceKeysScratch` → `StrSpan` variant fed to the `(StrSpan,StrSpan)` `lookupStatementFlags` overload. `checkNecessityForEquality`'s interior: `prefixArgumentsWithUScratch` under ONE function-level `ScratchScope`; `inputName`/`inputArgs`/`headName`/`headArgs`/`pArgs`/`origArgs` → span twins; the `originalChains` snapshot + `std::sort` → the shared `sortOriginalChainIndex` (its SECOND consumer after Core2 Case B), each chain decoded per iteration into a reused `std::vector`; the one-entry replacement map → a `StrReplacement[1]`; the premise's `removeUPrefixFromArguments` + `getArgs` set → `removeUPrefixScratch` + a sorted-unique `StrSpan` run fed to the `prepareIntegration` SPAN CORE (`prepareIntegration(StrSpan, const StrSpan*, int32_t, Memory&, StrSpan)`, retiring the string adapter at this site); the `(=[Y,Y_copy])` build → a `ScratchString` + `removeUPrefixScratch` + the `addExprToMemoryBlock` span door (the reactToHypo precedent). The ALREADY-landed `hasExistingEquality` view walk is untouched (its `targetVar` is now a span fed to the non-minting `nameMap.lookup(StrSpan)`). Whole-hook conversions — the batch gate is the byte-identity proof.

**S10 C7 — `disintegrateExprHypothetically` interior span-native.** `inputVars` / `sArgs` / `eArgs` → `getArgsSpans` (the `"marker"` guard → `equalSpans`); `targetVars` → a first-seen-order `StrSpan` run over `expr` (the `std::find` dedup → a linear `equalSpans` scan, storage order preserved for the hypoPayload build); `Y`/`Y_copy`/`equalityExpr`/`finalExpr` → a `ScratchString` + `removeUPrefixScratch` + the `addExprToMemoryBlock` span door (the reactToHypo precedent). The step-7 "register NEW STATEMENTS" loop drops the transient `EncodedExpression enc` for the EXISTING span doors — `lookupStatementFlags(…, StrSpan, StrSpan)`, `encodeExpression(StrSpan, StrSpan, NameMap&)`, span `nameMap.encode` — with the four container deposits unchanged (id-form). **`newStatements` kept as the honest edge `std::set<std::string>`** (materialized once from the out-of-scope `disintegrateExpr2` EWV set) — option (a), byte-trivially identical (no adjacency-order proof risk); the sentinel + `hypoPayload` stay `std::string` because their sole sink `encodePush` has no `StrSpan` overload (byte-identical edges); `disintegrateExpr2` is the out-of-scope EDGE. The ALREADY-landed `hasExistingEquality` view walk + `lvRun[256]` run are untouched. Whole-hook — the batch gate is the byte-identity proof.

**S10 C8 — the `cleanAdmissionMap` closure island + `canonicalizeUnderClasses`.** The closure's `decodeClassesAt` heap `std::vector<EquivalenceClass>` snapshot became a member-run snapshot (a `PagedVector<int16_t>` member pool + `PagedVector<int32_t>` class-start index on `genScratchArenas`, the checkForEquivalence S8 pattern — its second consumer); the `anyOverlap` scan reads `templateInterner.lookup(nameMap.decodeView(mid))` (both non-minting span doors); `canonicalizeUnderClasses` was retyped — param `const std::vector<EquivalenceClass>&` → the member-run view (via a small `EqMemberRunView` that `firstSpecialMemberId` templates on), return `.toStdString` → a `ScratchString` on a caller-owned `outArena` (its internal copies + `StrReplacement` run ride the gen-scratch byte-bump tier, a DIFFERENT arena family, so the result outlives them), `expr` → `StrSpan`; `canonOfTrigger` is a held `ScratchString`, each per-key canon rides a nested rewound inner scope, the `==` verdict → `equalSpans`; the `toErase` `vector<pair<int32,string>>` + `unordered_set<int32>` collapsed to ONE `ColdHashSet<PodKeyStore<int32_t>>` (the `kpTemplate` in the pair was only for the erase decision, dropped — the `eraseBlobIf` lambda keys on the int32 only), the consumedAdmissionKeys mint order = admissionMap scan order preserved. The admissionMap scan mints only `consumedAdmissionKeys` / `admissionStatusMap` / the scratch erase set (different sources than `equivalenceClassesMap` / NameMap / `templateInterner`), so the snapshot ids + decodeView spans stay valid (I-3). `decodeClassesAt` survives as the retained oracle (other callers); `canonicalize_under_classes_subst_twin` is a self-contained substitution mirror (does not call the function) and stays green. Whole-hook — the batch gate is the byte-identity proof. **(This closes the S10 `_integration_templates` subsession: the last coding commit.)**

**B9 — `compiledExpressions` (`std::map<std::string, LogicalEntity>`) — was assessed and DEFERRED.** Two blockers beyond cross-cutting volume (~16 field reads + ≥6 iterations + `DeepChecker`'s map-ref member + the sacred dump): (1) it is NOT load-once — `prover.cpp` inserts OR expressions into it at RUNTIME during proving (single-threaded between bursts, lock-free read in the burst), so a statified form must support runtime insert + concurrent non-minting read; (2) its `std::map` sorted-key iteration order is OBSERVABLE across the dump, the visualizer, and four prover iterations, so a statified registry must reproduce that decoded-key sort byte-for-byte everywhere. A large, deeply-entangled, byte-sensitive change for a modest, slowly-growing, NON-per-burst (non-FTA-scaling) registry — byte-identity risk far outweighs the win. A safe version would need a never-deloaded analyzer interner with runtime single-threaded insert + concurrent non-minting read + decoded-key sorted iteration, the field reads on span accessors, `DeepChecker` / the iterations / the visualizer adapted, and the sacred `dumpEntry` fed a decoded temp `std::map` at its call site (signature/body untouched) — done only if the registry is shown to matter for scaling. B6 found the rejection-revival path (`revisitRejectedIntegration2` / `emitIntegrationRevivalToInternalMailIn`) already near-fully static (one `getArgs` → `getArgsSpans`; the rest boundary-bound or I-3-owned). The remaining leaf transients (the subst maps, `replaceKeysInString`, the origin records) stay heap pending later batches.

**IntInstruction working-struct elimination — foundation (, unwired).** The `ArenaIntegrationMap` gains four heap-free doors that let its consumers shed the transient heap `IntInstruction` / `IntLogicalEntity` locals CodeQL cannot see (a `std::vector` inside a user struct — [I-138](#i-138)'s remaining `standardProcessing`-tree OPEN violation): `loadInstructionInto(ei, wi, vi)` inlines `loadFromIntInstruction(wi, instructionAt(ei), vi)` (reads entry `ei`'s records straight into a `WorkInstruction`, minting only `wi`'s own transient interner, never `vi`); `flattenWorkInstructionInto(wi, vi, pool)` mints `wi`'s fields into `vi` and appends the flattened id run in `encodeWorkInstruction`'s EXACT mint sequence (per entity: category, elements, signature, definedSet; markedGoal LAST — the frozen order the stored `Codec<IntegrationEntry>` blob + `.deload/` id table depend on, [I-84](#i-84)) with no throwaway `IntInstruction`; `findOrEmplaceFlattened(run, len)` id-compares/appends a flattened run (mints NOTHING); and `findOrEmplaceWork(wi, vi)` composes the flatten (into a byte-bump run on the map's own `*arena` under a `ScratchScope`) with `findOrEmplaceFlattened`, minting ALL fields UNCONDITIONALLY before the compare (the mint-on-HIT rule — a probe-then-mint would diverge the id table on hits). The pool STORAGE layout (`[entityCount][per entity: catId, sigId, arity, dsId, elemCount, elemIds…][markedGoalId]`) is an internal free choice; only the MINT order is frozen. The doors land unwired; the three integration consumers (`isAdmittedIntegration` / `updateAdmissionMapIntegration` / `applyEquivalenceClassToAdmissionMapIntegration`) plus the `prepareIntegration` marker-key site wire onto them batch by batch. Fresh-interner byte-twins assert the mint order strict.

**Code.** `memory.hpp` (`WorkLogicalEntity`, `WorkInstruction`, `encodeWorkInstruction`, `loadFromIntInstruction`, `entityAt`, `cleanInstructionWork`, `ArenaIntegrationMap` / `ArenaIntLE`, incl. the direct cold-I/O `buildFromCold` / `writeToCold`, and the four IntInstruction-boundary doors `loadInstructionInto` / `flattenWorkInstructionInto` / `findOrEmplaceFlattened` / `findOrEmplaceWork`); `prover.hpp` (`prepareIntegration` / `prepareIntegrationCore` / `prepareIntegrationCore2`; the two bridge sites `isAdmittedIntegration` / `updateAdmissionMapIntegration`; the heap `cleanInstruction` deleted by B5; the equi-class hooks `applyEquivalenceClassToRejectedMapIntegration` / `applyEquivalenceClassToAdmissionMapIntegration` — `RmiSortRow` / `AmiSortRow`; the `updateRejectedMapIntegration` leaf); `prover.cpp` (`disintegrateExprHypothetically` bridge). Unit tests: `test_memory.cpp` (`work_instruction_build_and_read`, `work_instruction_codec_round_trip`, `clean_instruction_work_matches_heap`, `arena_integration_map_matches_heap`, `arena_integration_sorted_indices_fill_matches_vector`, `arena_integration_values_at_fill_matches_vector`, `load_instruction_into_matches_load_from_int`, `flatten_work_instruction_into_matches_encode`, `find_or_emplace_flattened_matches_heap`, `find_or_emplace_work_matches_encode_find`). See also [I-132](#i-132), [I-84](#i-84).

---

<a id="i-128"></a>
## I-128 — the equivalence-class processing path's transient heap is statified; decoded class snapshots read via a zero-copy blob view

**Scope.** The equivalence-class processing surface (`prover.hpp`: `applyEquiClasses`, `applyEquivalenceClass`, the four admission/rejected equi-class hooks, `mergeTwoEquivalenceClasses`, `updateEquivalenceClasses`, `cleanUpExpressions`, `cleanUpAdmissionMap{,Integration}`, `enumerateEqClassRewrites`, `reduceEqClassIds`, `chooseCanonical`, `canonicalizeUnderClasses`, `filterIterations`; `prover.cpp`: `applyEquivalenceClassToNegatedEquality`, `addEquality`, `addNegatedEquality`, `updateWeakVariables`) — the transient-statification campaign's equivalence-class arm. The PERSISTENT class state is already static (`equivalenceClassesMap` is the cold `TypedColdBlobMap<int16_t, EquivalenceClass>`, Batch-3 tags 47–50; levels / origin / name-cache / changed-class state already cold or id-form), so this batch removes only the per-call WORKING heap. The foundation (the read-view type) lands first, unwired; the per-function wiring follows batch by batch.

**Rule.** The dominant per-call transient — the decoded `std::vector<EquivalenceClass>` snapshot that `decodeClassesAt` / `decodeClassesById` materialize at ~10 read sites (each element re-materializing its own `std::vector` / `std::map` / `IdOriginMap`) — is read instead through `EquivalenceClassView` ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)): a zero-copy `(p, len)` over one class's cold blob, obtained from the blob map via `TypedCold::peekBlobContiguous` / `peekBlobAt`, reading members / per-pair levels / origin history straight off the arena (the `OwnerSetBlob` idiom, layout-coupled to `serializeEquivalenceClass`). No `EquivalenceClass` heap decode. The remaining working scratch is statified with the campaign's tools: substitution maps → stack `StrReplacement[]` + `replaceKeysScratch`; arg lists → `getArgsSpans` / `StrSpan`; decoded strings → `StrSpan` / `ScratchString`; member-id lists / per-loop queues / id-set dedup → page-tier `PagedVector` / `ColdHashSet` on the per-slot `genScratchArenas`. Cross-subsystem RETURN vectors (`newStatements` / `products` consumed by the kernel's post-`addStatement` loop) stay heap and are materialized at the function edge — the boundary the batch deliberately does not cross.

The same arena constraints as [I-132](#i-132) apply: page containers ride `genScratchArenas`, never the `scratchArenas` string arena; never `StrSpan(rawCharArray)`; every observable ordering sorts a decoded id snapshot ([I-84](#i-84)).

**Status.** Foundation landed (`EquivalenceClassView` + round-trip tests). Wiring in progress: `chooseCanonical` fuses the weak-filter into the tier scan (no `strong` vector); `enumerateEqClassRewrites` runs its per-mapping scratch on stack `StrSpan`/`int` arrays (the `tempList` / `indices` / `argIds` / `tempIds` / `joined` heap gone; the dead `substMap` / `isIdentity` fields dropped now the rmi hook no longer routes through it); `applyEquivalenceClass` feeds that helper `getArgsSpans` argument spans + on-demand `decodeView` member names (its `argsExpr` / `eqList` string vectors gone). Its two per-call maps (`exprLevelsMap` / `exprOriginMapLocal`) are gone too — the rewrites accumulate as append-only rows on the per-slot `genScratchArenas` (interned `rewrittenExpr` + level/eq id runs), deposited once per equal-key group via a `compareSpans`-sorted, `seq`-tie-broken index walk that reproduces the former ascending-`std::map` iteration and `operator[]` last-write-wins exactly (deposit at each group's last element = max-seq winner). The small `EqClassRewrite` per-mapping sorted sets (`extraLevels` / `setEqualities`) became loop-local stack buffers in **equi-9a**: `extraLevels` a sorted-unique `int[kEqRewriteLevelCap]`, and the justifiers a sorted-unique `EqClassSpanPair[MAX_ARITY]` (from/to member-name spans) the sink formats into `"(=[from,to])"` and interns — byte-identical order because member names carry no comma, so the `(from,to)` sort equals the former full-string `std::set` order. The two ALGEBRA hooks (`applyEquivalenceClassToAdmissionMap` / `applyEquivalenceClassToRejectedMap`) now build their full-map `(template, validity)` snapshot on the per-slot `genScratchArenas` via the same `ColdHashSet<BytesKeyStore>` interner + `PagedVector<SortRow>` + `compareSpans` index-sort the integration `rmi`/`ami` hooks use — reproducing the former heap `std::vector` + `a.first < b.first` `std::sort` order exactly (that walk order drives the `revisitRejected2` / mail emission order). All four hooks' `substMap` (`std::map<string,string>`, every non-canon member → the `chooseCanonical` representative) is now an arena `StrReplacement` run on `genScratchArenas` (member NAMES are stable `decodeView` spans; no NameMap mint before the post-loop drain), and every `ce::replaceKeysInString(x, substMap)` is `replaceKeysToString(StrSpan(x), subPairs, subCount)` — the proven byte-exact, source-scanned, order-independent twin. `subPairs` iterates `memberIds` (decoded-lex), so it preserves the former `std::map` key order (the `usedEqs` origin tail stays byte-identical); the integration `ami` hook's `augSubst` becomes one `augPairs` run (bare pairs then `u_`-pairs sharing the buffer, `u_`-names arena-materialized) and its per-row arity-bounded `uMap` becomes stack `StrReplacement[MAX_ARITY]` under a per-row `ScratchScope`. The trivial `toErase` / `eraseSet` are now one scratch `TypedColdSet<int32_t>` per hook (membership erase is order-independent → byte-identical: `mint` during the walk, `contains` for `eraseBlobIf`, `decodeKey`-iteration for the algebra-admission `admissionStatusMap` drop). The two MAIL hooks' value staging is done: `PendingMail` / `std::vector<PendingMail> toMail` → arena `MailRec` interned-id records (compoundPost / Pre / preVld / depVld ids + level and equality1-tail id runs) on the per-slot `genScratchArenas`, appended in walk order and drained post-walk in the same order (the drain mints in NameMap via `resetResentExpressionRegistries`, so it must stay deferred — the arena interner is NOT the NameMap, so those mints never touch the staged strings). The rmi hook's `uniqueCompounds` (`std::map<string,std::set<int>>`) → a per-entry fresh `ColdHashSet<BytesKeyStore>` + a `compareSpans` ascending-compound index-sort (the level union is idempotent — every value at one compound looks up the same `(compound, validity)` levels — so it reduces to distinct-compounds-sorted, the former `std::map` iteration order that drives mail emission), and `usedEqs` → an interned id run. The ALGEBRA insert hook is done: `PendingAdmissionInsert` / `std::vector<PendingAdmissionInsert> toInsert` → arena `AmInsertRec` (interned newKeyTemplate / depositValidity + the all-int `AdmissionMapValue` — depth / sec / flag + valueInterner-id key / remainingArgs runs), appended in walk order and drained in the same order; the value's ids are minted in `valueInterner` during the walk (a different interner from NameMap, so the `decodeView` subPairs stay valid), and the `AdmissionMapValue` is rebuilt once per record at the `insertAdmissionValue` boundary. The INTEGRATION insert hook is done too: `PendingAdmissionIntegrationInsert` / `toInsert` → arena `AmiInsertRec` (interned newKeyUForm / newKeyBare / depositValidity + a flattened id run (`flattenWorkInstructionInto`, layout |data|, then per entity category / signature / arity / definedSet / |elements| / elements; markedGoal LAST) consumed at the `findOrEmplaceFlattened` boundary — no heap `IntInstruction`, plus a sorted `newAppliedVars` interned-id run). **All four equi-class hooks are now heap-free** (the inner per-record `ArenaIntegrationMap` shares the same scratch slot as it always did; the staging pools are page-tier and read via stable vids across its allocations). `updateEquivalenceClasses`'s ANCESTOR merge loop now reads each ancestor scope's classes through `EquivalenceClassView` over the cold blob (`equivalenceClassesMap.lookup` + `runLen` + `peekRecordBytes`) instead of decoding the whole scope to a heap `std::vector<EquivalenceClass>`: the overlap test scans the view's member ids (byte-identical to the heap `memberIds` scan), and only the rare OVERLAPPING ancestor is materialized (`deserializeEquivalenceClass`) to a heap class for the const& merge — the common non-overlapping majority stay view-only, no heap decode. Ancestor classes are read-only (D-44 / I-31), so no run mutates inside the loop and the peeked pointers stay valid across it. Its `argsList` (`ce::getArgs`) → `getArgsSpans`. This was the foundation view's first read-path wiring; equi-6b then converts the SAME-SCOPE loop the same way and — the user's full-package, no-deferral choice — splices the NON-overlapping KEPT classes' raw blobs into the rewritten run VERBATIM through the existing raw byte door (`equivalenceClassesMap.inner.assignRun`), no decode and no new cold-map variant: pass 1 merges the rare overlapping classes + counts the kept blobs, pass 2 copies the kept blobs (BEFORE the run-replacing `assignRun`, so the peeked pointers never dangle) followed by the serialized `mergedClass` into one contiguous `genScratchArenas` byte-bump buffer. A kept blob equals re-serializing its decoded class (a lossless pure-function codec, I-103), so the raw run is byte-identical to the former `assignClassesById([kept..., merged])` — pinned by `test_memory.cpp::eqclass_raw_door_run_splice`. The whole-run same-scope heap decode is gone; only the rare overlapping-class materialize + the single `mergedClass` merge-output stay heap (the ancestor-loop pattern). **equi-6-accum** then lands the mutable arena-class accumulator `MergeClassAccum` (`memory.hpp`) — members `PagedVector<int16_t>`, per-key ascending-int level runs, flat insertion-order origin lines — with a `serialize` BYTE-IDENTICAL to `serializeEquivalenceClass` (proven by `merge_accum_serialize_matches_heap`), an `addOriginId` twin carrying the D-49 cap-preference, and `unionMembersByName` (the decoded-lex two-pointer merge); `EquivalenceClassView` gains `findLevels` / `hasOrigin` point lookups. It is DORMANT (unwired) → the run stays byte-identical, and it is the seam the producer/consumer split rests on: the producer pushes accumulator-serialized bytes byte-identical to `serialize(mergedClass)`, so `changedClassesThisStep`'s producer and consumer decouple across that byte-stable interchange (I-125). **equi-6-accum-2** adds the B-base origin-merge op `MergeClassAccum::mergeOriginsBBase`: the accumulator holds A, but the merge's origin combine is B-BASE (`tmp = classB.equalityOriginMap; overwriteOriginsId(tmp, A, cap)`), so the op snapshots A's lines, clears the origin section, re-seeds it verbatim from the classB view (`forEachOriginLine`), then merges the A snapshot back per key below cap and deduped — `overwriteOriginsId`'s inner, NO D-49 cap-preference (that belongs to the equality2-emission `addOriginId`). Per-key line order (B-first-then-A) is observable, so it is the byte contract; proven byte-identical to the heap `overwriteOriginsId` by `merge_accum_origins_bbase_matches_heap`. **equi-6-accum-3/4** complete the accumulator API — the cross-product / A-subset-B combine ops (`mergeLevelsBBase` / `mergeLevelsFromAccum` / `mergeOriginsFromAccum`), plus `unionMembersFromView`, `forEachOriginSorted`, `addOriginEnc`, a member-run `upsertEqClassIndex` / `eraseEqClassIndex` byte-key, and a `ChangedClassesBuffer` push-from-bytes — each differential-tested. **equi-6-merge** then wires it: `mergeTwoEquivalenceClasses(MergeClassAccum&, EquivalenceClassView, …)` runs both branches on the accumulator (A-subset-B = union + `mergeLevelsBBase` + `mergeOriginsBBase`; cross-product builds a SECOND `MergeClassAccum` for the former `mergedMap` / `mergedOriginMap` in the pair loop — the equality2 I-32 gate reads the accumulator / view `hasOrigin`, the `exprOriginMap` side effects preserved — then folds it in with `mergeLevelsFromAccum` / `mergeOriginsFromAccum`), and `updateEquivalenceClasses`'s six `mergedClass` uses (init; the two merge calls via views, no deserialize; `serialize`→raw-door; `changedClassesThisStep.push`-from-bytes; `upsertEqClassIndex`×2; origin-sync via `forEachOriginSorted`) read the accumulator. The merge-internal heap (`mergedMap` / `mergedOriginMap` / `newLevels` / `tmpMap` / `bridgeCandidates` / `or1` / `or2`) and the whole-scope heap decode are gone — only the rare overlapping-class blob peek stays view-only. Byte-identity proven by `merge_full_serialize_matches_heap` (A-subset-B + cross-product against the heap algebra oracle) on top of the per-op differential tests. **equi-7** is DONE. The five consumers (`applyEquivalenceClass` + the four admission/rejected hooks) and the leaf helpers (`chooseCanonical` / `filterIterations` / `reduceEqClassIds` / `firstSpecialMemberId` / `enumerateEqClassRewrites`) are TEMPLATED on the class type (a shared `memberCount` / `memberId` / `forEachLevel` read interface on both `EquivalenceClass` and the view; byte-identical while every existing caller still instantiates for `EquivalenceClass`). `applyEquiClasses` then reads its delta class (Pass 1, via `ChangedClassesBuffer::classViewAt`) AND the non-delta cold-bucket class (Pass 2, via `equivalenceClassesMap.inner.peekBlobContiguous`) through `EquivalenceClassView` — no heap `EquivalenceClass` decode remains on either pass (proven byte-identical on the hot path by the equi-7e gate run). Its setup transients are statified too: `deltaIds` (`std::set<pair<int16_t, vector<int16_t>>>`, formerly fed by heap `classAt`) → a per-slot `genScratchArenas` `ColdHashSet<BytesKeyStore>` keyed by `encodeEqClassKeyFromView` and built from the changed-class VIEWS (bijective encoded key → identical membership); `startIdx` (`std::vector<size_t>`) → a page-tier `PagedVector<int64_t>` (an unobservable per-class waterline); `validityKeys` (`std::vector<pair<string, int16_t>>`) → an arena `PagedVector<int16_t>` iterated in decoded-name ascending order via a byte-bump `compareSpans` index-sort — byte-identical to the former `std::sort` of `(decodedName, id)` pairs (each id has a unique name, so the name is the sole ordering key). `products` / `newStatements` stay heap (the user's standing cross-subsystem edge). New `EquivalenceClassView` `lookupEqClassIndex` / `upsertEqClassIndex` overloads (`encodeEqClassKeyFromView` + the cold byte-key door) replace the `cls.memberIds` keys. **equi-8** statifies `cleanUpExpressions` (the active Step-4b sweep): its `cleanupClasses` heap decode → an `EquivalenceClassView` walk over the cold blob run (`equivalenceClassesMap.lookup` + `runLen` + `inner.peekBlobContiguous`; the bucket is never mutated in-loop, so the count / blobs stay stable, and `filterIterations` takes the view via its equi-7b template); the five `IntEncodedExpr` row rebuilds (`newIntLocalEncoded` / `droppedLocalRows` / `newIntLocalEncodedDelta` / `newIntEncodedStatements` / `droppedRows`) → per-slot `genScratchArenas` `PagedVector`s, the kept rows swapped into the arena-backed runtime registries via `clear` + `push_back` in the SAME order; the four `std::set<int32_t>` dropped-key membership sets → two reused `ColdHashSet<PodKeyStore<int32_t>>` (`resetToFresh` per block, `lookup!= 0` membership). Byte-identical: same class order, same keep/drop verdicts, same row order, same `intStatementLevelsMap` erase set (`intKnownStatements` still NOT erased — I-58 / D-93); `newStatements` / `filtered` stay heap (the ignored cross-subsystem Step-4b sink, D-93), and `clsScratch` is the sanctioned `peekBlobContiguous` page-straddle fallback (one reused buffer, as in `applyEquiClasses`). `cleanUpAdmissionMap` / `cleanUpAdmissionMapIntegration` are confirmed DEAD (no live caller, subsumed by the D-106 inline drop-and-rekey) — left as-is, no runtime heap, not deleted (out of this batch's scope). **equi-9b** finishes the tail: `applyEquivalenceClassToNegatedEquality`'s `args` → `getArgsSpans` and its two `decodeClassesAt` heap walks (same-NS + strict ancestors) → an `EquivalenceClassView` blob-run walk (a LOCAL page-straddle `clsScratch` keeps a recursive `emitNew` → `addStatement` → self nested call from clobbering the outer peek; negated equalities form no classes, so the run is never mutated mid-walk and the peeked blob + membership stay valid); `addEquality` / `addNegatedEquality`'s `args` → `getArgsSpans` (guard → `equalSpans`, the mirror a boundary `std::string` built from the spans for `encodeExpression` / `intKnownStatements`); `updateWeakVariables`'s `eqVars` → `getArgsSpans` and its `decodeClassesById` decode → a view walk that records the target class's weak members inline (steps 3-4 mint only NameMap / `intWeakVariables` / the name-kind cache, never `equivalenceClassesMap`, so the peeked blob stays valid). **The batch is code-complete: the equivalence-class processing path holds no transient heap `EquivalenceClass` / arg-list vector / set / map except the standing cross-subsystem `newStatements` / `products` returns and the sanctioned `peekBlobContiguous` page-straddle `clsScratch` fallback — pending the manager's final full-pipeline byte-identity gate.**

**Code.** `memory.hpp` (`EquivalenceClassView` + its `findLevels`/`hasOrigin`, `MergeClassAccum` + its combine/view-union/origin-iterate/`addOriginEnc` ops, the accum/view `eqClassSttmntIndexMapMap` byte-key helpers, `ChangedClassesBuffer::push`-from-bytes); `prover.hpp` (the equi-class hooks, `enumerateEqClassRewrites` / `applyEquivalenceClass`, `mergeTwoEquivalenceClasses` on the accumulator + view, `updateEquivalenceClasses`'s ancestor + same-scope view/splice paths + the accum-based `mergedClass`, `applyEquiClasses` / `cleanUpExpressions` / `applyEquivalenceClassToNegatedEquality` on the view, the `EqClassSpanPair` residual); `prover.cpp` (`addEquality` / `addNegatedEquality` / `updateWeakVariables` — `getArgsSpans` + view). Unit tests: `test_memory.cpp` (`eqclass_view_reads_every_field`, `eqclass_view_empty`, `eqclass_raw_door_run_splice`, `merge_accum_serialize_matches_heap`, `merge_accum_add_origin_id_cap_preference`, `merge_accum_union_members_decoded_lex`, `merge_accum_origins_bbase_matches_heap`, `merge_accum_levels_bbase_matches_heap`, `merge_accum_levels_from_accum_matches_heap`, `merge_accum_origins_from_accum_matches_heap`, `merge_accum_union_members_from_view`, `merge_accum_for_each_origin_sorted`, `merge_accum_add_origin_enc_matches_heap`, `changed_classes_buffer_push_from_bytes`, `upsert_eq_class_index_from_accum`, `merge_full_serialize_matches_heap`). See also [I-132](#i-132), [I-133](#i-133), [I-84](#i-84).

---

<a id="i-138"></a>
## I-138 — the `standardProcessing` call tree's interior transient containers are statified, EXCEPT the compiled-definition layer and lone decode-strings, which are still heap — an OPEN violation to statify, not sanctioned (the absorb entrance decode snapshots flip to direct cold-id iteration in L8)

**Scope.** The transitive call tree of `standardProcessing` ([`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp)) — the per-step mail-absorb → apply-equi-classes → discharge → fill-mail-out driver run in prover phases 1 and 3 ([I-63](#i-63)). This is the transient-statification campaign's final arm, unifying the per-path batches ([I-128](#i-128), [I-129](#i-129), [I-133](#i-133), [I-132](#i-132), plus the mail statification) and closing the residual per-call heap they left across the driver body, the discharge path (`dischargeToBeProved` / `dischargeContradiction` and callees), `fillMailOut`, and the `addExprToMemoryBlock` residuals.

**Rule.** Every OWNED transient container inside the tree (arg-list vectors, substitution maps, decoded-class snapshots reused as scratch, level sets, page-straddle `char` buffers, accumulator vectors/sets) rides the per-slot `genScratchArenas` (page + byte-bump tiers) or `scratchArenas` (string tier), released per worker task at `performElem2` exit — using the campaign's established recipes ([`09b_statification_cookbook.md`](20_core_concepts/09b_statification_cookbook.md)); no new mechanism is introduced. TWO boundaries are still heap — OPEN violations to statify, NOT sanctioned by decision (the rule: in `performElem1/2/3` the only permitted heap is `hashburst_trace.txt` generation): (1) the compiled-definition layer — `compiledExpressions`, `coreExpressionMap`, and the string `LogicalEntity`/`Instruction` working forms flowing through disintegration (the existing string-typed-`LogicalEntity` contract); (2) entrance decode snapshots (`decodeMailInStatements`/`decodeMailInOrigins`/`decodeInternalMail*`/`decodeClassesById` returned `std::vector`s) plus values returned to out-of-scope subsystems, materialized at the edge — the absorb-path consumption of these is itself statified by **L8** below (direct cold-id iteration + arena decoded-lex indices reproducing the former `std::sort`/`lower_bound` orders); the decode helpers survive for the out-of-scope-subsystem edges (the sacred dump, tests, other prover sites). Byte-transparent — proof artifacts byte-identical. **Definition sharpened ([D-192](40_decisions.md)):** a function counts as statified only at 0% heap — the CodeQL inventory (`statification/inventory.md`) row `true`, with no by-value heap parameter or return either; the boundary copy lives in the still-heap caller and static→heap→static walk-arounds are banned. The remaining ≤ #100 heap leaves are retired under that bar.

**Status. COMPLETE for the `standardProcessing` (phase-1/3 fan-out) interior; the phase-2 firing check + admission-gate + integration-template subsystem is a SEPARATE target — see the S7-census extension note below.** The full `standardProcessing` interior cascade landed (layers L1–L9): every OWNED transient container AND every lone `std::string` / `ExpressionWithValidity` / `OriginLine` in the tree is now span/id-form. Only the `compiledMap` / compiled-definition layer and the cross-LB heap-`Mail` `addOrigin` boundary are still heap — both OPEN violations to statify, not sanctioned end-states. All layers gated byte-identical — `main.py` 0 verifier failures, `git status files/` clean, per-commit `--unit-tests` green, adversarial diff review clean. `canonicalizeUnderClasses`'s `std::map<std::string,std::string> substMap` → a contiguous `StrReplacement` run on `genScratchArenas` (each member / canonical name copied onto `scratchArenas` so the span outlives its `ScratchString` local and any later NameMap growth, [I-3](#i-3); `replaceKeysScratch` is the byte-exact greedy-longest twin of `ce::replaceKeysInString`). The equi-class page-straddle `char` scratch buffers (`clsScratch` in `cleanUpExpressions`, `peekScratch` + `ancScratch` in `updateEquivalenceClasses`, `clsScratch` in the recursive `applyEquivalenceClassToNegatedEquality`, and — completing the sweep in S8 `_firing_check` — `clsScratch` in `updateWeakVariables`) now ride the per-slot `genScratchArenas` byte-bump tier via a `ScratchArena&` overload of `HashMap::peekBlobContiguous` / `TypedCold::peekRecordBytes` — each straddle takes a FRESH `alloc(len,1)` with NO rewind, so an outer peek's bytes survive the recursive `applyEquivalenceClassToNegatedEquality` → `addStatement` → self re-entrant peek (byte-bump accumulation, reclaimed by the per-task `releaseAll`). `applyEquiClasses`'s own `clsScratch` — the last of the equi-class page-straddle buffers — is now retired too: its two `ChangedClassesBuffer::classViewAt` peeks ride a new `ScratchArena&` overload of that method (the arena twin of the `std::vector<char>` `classViewAt`) and its `peekBlobContiguous` peek rides the existing overload, all onto the apply's already-present `deltaArena` slot handle by the same FRESH-`alloc`-NO-rewind accumulation. The remaining container work landed in the same batch: `cleanUpAdmissionMap`(+`Integration`) (`toErase` → `PagedVector<int32>`, `eraseSet` → `ColdHashSet<PodKeyStore<int32>>`); `updateEquivalenceClasses` + `mergeTwoEquivalenceClasses` (`eqArgs` → sorted `StrSpan[]`, `eqArgIds` → stack `int16[]`, level runs → byte-bump; `mergeTwo`'s `const std::vector<int16_t>&` param → `(const int16_t*, int32_t)`); `dischargeToBeProved` + `cleanUpOrIntegrationBranches` (`deltaSnapshot`/`victimIds` → `PagedVector`, `auxiesCopy` → direct cold-run walk, `victimNames` collapsed into `victimIds`); `prepareIntegration` + `updateAdmissionMapIntegration` + `createAuxyImplication` (arg-list vectors → `getArgsSpans`, substitution maps → `StrReplacement`, observable-order membership → sorted `StrSpan[]` + `compareSpans`); `fillMailOut` (`lines` `std::vector<IdOrigin>` → direct `runLen`/`recordAt` walk; and, in S8 `_firing_check`, the per-row `EncodedExpression`/`EWV` decode → `decodeView` spans — fully id/span, no per-row string, `allowedForMail`/`lookupOriginKey` span-fed); and the `standardProcessing` direct body + `absorb` lambda (`_deferredCleanupVals` → `ColdHashSet`, absorb `args` → `getArgsSpans`). `addExprToMemoryBlock`, `dischargeContradiction`, `getGlobalKey`, `addEquality`/`addNegatedEquality`, `classifyOrScope`, `reduceEqClassIds` carried no interior container work (all boundary or already-static). **S8 `_firing_check` correction:** the eq-class-apply cluster DID carry residual heap the standardproc scope missed — now flipped: `reduceEqClassIds` returns an out-param `int16_t[]` run + count (the `enumerateEqClassRewrites` param retyped to `(const int16_t*, int32_t)` in step); `applyEquivalenceClass`'s deposit leaves (`depositValidity` → `ScratchString` copyFrom, `applied` → a raw `exprKeys` `StrSpan`, the dropped `appliedEnc`/`appliedWithValidity`, and the trackHistory `OriginLine` → the `OriginDep` span door with `OriginTag::equality1`) are span/id-form; and `applyEquiClasses`'s two `classValidity` decodes (Pass 1 + Pass 2) are `ScratchString` copyFrom on the string-scratch arena (they survive the apply* / applyEquivalenceClass NameMap mints, I-3). **Scope deferral:** lone standalone `std::string` scalar temporaries (I-3 decode copies at mint-adjacent sites) are NOT converted this batch — the *containers* are off the heap, the scalar decode-strings remain (the same line the prior [I-128](#i-128) equi-class batch drew); the full-string-freedom follow-up landed in **L9** below. **That follow-up's shared span/scratch doors begin landing additively (byte-identical, wired in later batches):** `encodeValueSpanSetSorted` ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)) is the value-set span twin of `encodeValueSetSorted` — it dedups member spans in a scratch `ColdHashSet`, `compareSpans`-sorts the distinct ids, and mints into the `ValueInterner` in that decoded-lex order, reproducing the `std::set<std::string>` mint-order id vector byte-for-byte (the [row-50](20_core_concepts/09b_statification_cookbook.md) `rewrittenRemaining` / `newApplied` container pass's sorted-encode step; transients reclaimed per call via `ColdHashSet::release` + `popTo`). The origin `(=[x,y])`-construction and gate doors those builders need already exist (`encodeExpression(StrSpan,StrSpan)`, `addOriginEncoded(StrSpan)`, `mintOriginKey(StrSpan)`, `lookupStatementFlags`/`lookupStatementLevels(StrSpan,StrSpan)`, `ValueInterner::encode(StrSpan)`), so the origin-record deposit path is span-complete and needs no new EWV-from-spans ctor. `addStatement` encodes its statement to an `IntEncodedExpr` at entry via `encodeExpression(StrSpan,StrSpan)` but then consumes the raw `std::string expr` DEEP — the `addEquality` / `addNegatedEquality` / `updateEquivalenceClasses` / `applyEquivalenceClassToNegatedEquality` sub-builders (all `const std::string&`), the `isEquality` / `isNegatedEquality` / `extractMaxIterationNumber` / `countPatternOccurrences` shape helpers, and the `ExpressionWithValidity(expr,…)` return boundary — so a clean `addStatement(StrSpan,StrSpan,…)` door (and the `applyEquivalenceClassToNegatedEquality` `emitNew`→`addStatement` exit that rides it) was deferred to a dedicated later step that first spanifies those sub-builders + shape helpers — **completed by L6 (statement param) + L7 (`validityName` param) below**, so those sub-builders + `validityName` now thread `StrSpan`, not `const std::string&`.

**L5 container flip (landed — the "full string-freedom" follow-up above).** The two transient statement buffers themselves — `applyEquiClasses`'s per-statement `products` and `addStatement`'s returned `newStatements`, both formerly `std::vector<ExpressionWithValidity>` — flip to `PagedVector<IntEncodedExpr>` (id form) on the per-slot `genScratchArenas`, so the containers build no heap `std::string`/EWV for statements in this path. The ONE new foundation door is `sortStatementRows` ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)) — a byte-bump `int32_t[]` index over the id rows that reproduces `std::sort` under `ExpressionWithValidity::operator<` exactly (decode-then-lex-sort: `compareSpans(decodeView(originalId))` then, on a tie, `decodeView(validityId)`; byte-identical because `IntEncodedExpr::originalId == nm.encode(original)` and `validityId == nm.encode(validityName)`, and `compareSpans == std::string::compare`). The consumers (`ordisMerge` at the `applyEquiClasses::mergeProducts` edge; `updateAdmissionMapIntegration` / `updateAdmissionMapRecursion` / `ordisMerge` at the `addExprToMemoryBlock` stmts loop) originally kept their `const std::string&` signatures fed a per-row `nameMap.decode(id)` (superseded by the S4 flip below — they now take `StrSpan` fed a per-row `ScratchString`) at the compiled-definition / mint-adjacent edge (the sanctioned "materialize at the edge" boundary — a `decodeView` span would dangle at `ordisMerge`'s first mint, I-3). **`products` landed** (self-contained): `applyEquivalenceClass`'s sink param → `PagedVector<IntEncodedExpr>&`, its one push appends the already-minted `ie` (no new mint), and `mergeProducts` index-sorts via `sortStatementRows` then edge-decodes to `ordisMerge`. **`newStatements` landed** (the connected component): `addStatement` returns `void` and takes a trailing `PagedVector<IntEncodedExpr>& newStatements` out-param (PagedVector is non-movable — cannot be returned); the mirror push id-mints `(=[b,a])` at the push site (byte-identical id order — `addEquality` re-encodes only already-minted strings before its mirror registration, so nothing grabs an id in between), the main-local + equality-self pushes reuse the entry-minted `ieStmt`, `applyEquivalenceClassToNegatedEquality`'s sink flips (its `emitNew` passes a per-slot throwaway to the inner `addStatement`), `updateEquivalenceClasses` keeps the param as an in-place no-op carrier, `cleanUpExpressions`'s dead `newStatements` branch is dropped (param removed), and the `addExprToMemoryBlock` consumption edge builds the buffer on `genScratchArenas`, index-sorts via `sortStatementRows`, and edge-decodes each row to `updateAdmissionMapIntegration` / `updateAdmissionMapRecursion` / `ordisMerge`. The two discarded `addStatement` call sites (`emitNew`, the vacuous-truth head) pass per-slot throwaway buffers. See [I-25](#i-25) for the deposit-channel contract.

**S4 consumer edge flip (landed).** The per-row heap `std::string` decode L5 left at the two consumption edges is removed — the id-form flip is zero-heap through to the consumers. The three consumers now take `StrSpan` (`ordisMerge`, `updateAdmissionMapIntegration`, `updateAdmissionMapRecursion` — the two edges are their sole callers, so the signatures change outright, no overload); each edge decodes a row's `(originalId, validityId)` into a per-row `ScratchString` on the string tier (`scratchArenas`, `ScratchString::copyFrom(nameMap.decodeView(id))`) under a per-row `ScratchScope`, byte-identical to `nameMap.decode`. The `ScratchString` rides a DIFFERENT arena than the NameMap / `lbStateInterner` / `valueInterner` cold tables the consumers mint into, so the `StrSpan` stays valid across those mints where a raw `decodeView` span would dangle (I-3) — the exact dangle that forced L5 to keep the heap decode. Consumer interiors were already span-ready (L1/L2): `equalSpans(v, "main")` for the guard, `getArgsSpans`, `replaceKeysScratch`, the `StrSpan` overloads of `lookupTemplateKey` / `mintTemplateKey` / `lookupStatementLevels`, `NameMap::encode(StrSpan)`, `lbStateInterner.encode(StrSpan)`, `classifyOrScopeView`. THREE sanctioned std::string materializations remained at S4, each minimized to its edge — item (1) since RETIRED by the S6 reader: (1) `updateAdmissionMapRecursion`'s `core` — retired by the S6 `_reader` subsession (`I-137`): the `operators` probe now takes a `std::string_view` over the `extractExpressionSpan` slice (`operators` is `std::set<std::string, std::less<>>`, transparent) and the `coreExpressionMap` read goes through the `coreConfig` span reader, so no `std::string` is materialized on this path at all; (2) `prepareIntegrationCore2`'s by-value `std::string validityName` (the compiled-definition-layer integration sink) fed `validityName.toStdString` — `validityName` is `"main"` there (guarded); (3) `updateAdmissionMapRecursion`'s match-only `revisitRejected2` call, whose by-value `std::string validityName` already copied — `.toStdString` at that edge is no net-new heap (a NON-compiled edge, reached only on a match, not per-row). The `ordisMerge` EWV/origin boundary kept std::string per row-64 at S4 (`addExpression.toStdString` at each `ExpressionWithValidity` build — the internal-mail door then still took EWV); L4 below flips it to the `StrSpan` door. Byte-transparent.

**L4 internal-mail door span flip.** `insertInternalStatement` and `addInternalMailOrigin` ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)) gain ADDITIVE `StrSpan` doors — `insertInternalStatement(ColdMail&, NameMap&, StrSpan original, StrSpan validityName, const std::set<int>&)` and `addInternalMailOrigin(ColdMail&, ValueInterner&, StrSpan original, StrSpan validityName, const OriginLine&, int)` — that encode the statement/validity KEY from spans into the id-form `ColdMail` columns byte-identically to the `ExpressionWithValidity` overload (the `NameMap::encode(StrSpan)` / `mintOriginKey(StrSpan)` span twins mint the same key bytes to the same ids in the same evaluation order; same `encodeOrigin` + cap-full `addMailOriginRecord`). Only the KEY is spanned; the `origin` RECORD (`OriginLine.first` tag + `.second` antecedent `ExpressionWithValidity` vector) stays string — the deferred L3 origin-antecedent boundary. The doors land with byte-twin unit tests; the callers whose transient `ExpressionWithValidity` was built ONLY to feed these two sinks flip to spans over their stable `std::string` / `ScratchString` inputs. **`ordisMerge` landed first** — its statement + origin-key deposits pass `addExpression` and the `mergeParentView` slice of `effectiveValidity` straight to the doors, dropping the two `addExpression.toStdString` + two `std::string(mergeParentView…)` heap builds per convergence; both spans alias caller-owned `ScratchString` buffers (`StrSpan(prodOriginal)`/`StrSpan(prodValidity)` at the `applyEquiClasses` edge, `StrSpan(addExpression)`/`StrSpan(effectiveValidity)` at the `addExprToMemoryBlock` edge), NOT the NameMap the door mints into, so they survive the encode ([I-3](#i-3)). The remaining `prover.hpp` internal-mail writers followed: `emitIntegrationRevivalToInternalMailIn`, `sanitizeHashMemory`, the algebra + rejection equi-class admission/rejection hooks, and `dischargeToBeProved`'s two parent-scope emissions (OR-integration + NotOrScope) — each passing its stable `std::string` locals as spans, the `origin` record left string. `prover.cpp`'s `updateGlobal` / `updateGlobalDirect` cross-LB theorem-broadcast deposits completed the sweep (`StrSpan(valueR)` / `StrSpan(value)` + `StrSpan("main", 4)`, dropping the per-broadcast `evR` / `ev` EWV shared by both sinks). `memory.cpp::applyFiringRecords` keeps its EWV deliberately: that `ev` also feeds `setInternalDisintegrationSignal`, a THIRD internal-mail sink outside this layer's two-sink scope (its span door is a later follow-up). Byte-transparent.

**L3 origin-record span-antecedent doors.** The layer L4 deferred — the origin RECORD (`OriginLine.first` tag + `.second` antecedent `ExpressionWithValidity` vector) — is now id-formed for the `standardProcessing` tree. The transient heap `OriginLine` (`std::pair<std::string, std::vector<ExpressionWithValidity>>`) built at each in-scope origin-emission site fed exactly `encodeOrigin`, which does `originTagFromString(justification)` + a positional `mintOriginKey` per antecedent, discarding the `vector<EWV>` and its `2·N` `std::string`s. **Foundation** ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)): `struct OriginDep{ StrSpan original; StrSpan validity; }` (a stack antecedent), `struct TransientOrigin{ bool present; OriginTag tag; const OriginDep* deps; int depN; }` (the S6 pass-through carrier — `present==false` is the empty-`OriginLine` sentinel, so an empty origin never reaches `encodeOriginSpans`, whose `originTagFromString("")` would assert), and `encodeOriginSpans(OriginTag, const OriginDep*, int, ValueInterner&) -> IdOrigin` — byte-identical to `encodeOrigin` on the equivalent `OriginLine` (the tag is the same enumerator `originTagFromString(literal)` returns, taken directly; the deps mint via the `mintOriginKey` span twin over the same bytes in the same order; the tag is interned by neither path, so the `originInterner` mint sequence is the dependency mints alone). **Five additive span-record door overloads** each mirror an existing `OriginLine` door 1:1 with the KEY as `StrSpan`s (the L1/L2 form) and the RECORD as `OriginTag` + `OriginDep` span: the persistent cold `addOriginEncoded(TypedColdBlobMap<int64_t,IdOrigin>&)`, the heap `addOriginEncoded(IdOriginMap&)`, the `MergeClassAccum::addOriginEnc` accumulator method, `addInternalMailOrigin(ColdMail&)`, and `addRoutingMailOrigin(RoutingColdMail&)`. Each door reproduces its twin's exact call/statement shape (key mint first where the twin sequences it, `mo.ensureArena` first for routing), so the `originInterner` mint order and the D-49 cap-full preference are preserved. **STAYS HEAP:** the cross-LB heap-`Mail` string `addOrigin(std::map<ExpressionWithValidity, std::vector<OriginLine>>&, …)` path (`mailOut.exprOriginMap` in the theorem-load path, `broadcastMail`, the compaction fold, the runtime-`originTag` broadcast site) — the compiled/cross-LB string boundary, where the mail wire carries origins as inlined strings with no cross-LB interning; L3 removes the RECORD *construction* in the `standardProcessing` tree, not the `OriginLine` type. The doors land additively with byte-twin unit tests; the ~35–40 in-scope construction sites feeding ~57 door calls are wired cluster by cluster (equi-class → contradiction/vacuous → disintegration/expansion → anchor/mail leaves → the P9 `addExprToMemoryBlock → addStatement → addEquality/addNegatedEquality` pass-through cascade). **Equi-class cross-product cluster landed first** (`mergeTwoEquivalenceClasses`): the two `equality2` history records — whose antecedents are fresh `(=[X,Y])` concatenations of `varA`/`varB`/`commonArg` — are built on the per-slot string-scratch arena (`scratchArenas`, one `ScratchScope`, explicit-length `allocBytes` fills, never a bare-array `StrSpan`) and fed to the span-record accumulator + cold-map doors; the `or1`/`or2` `OriginLine` vectors and the `eq1Enc`/`eq2Enc` key EWVs are gone (the `eq1`/`eq2` key `std::string`s stay — a non-minting `lookupOriginKey` probe consumes them, and they are the statement KEY, not the origin RECORD). The sibling `updateEquivalenceClasses` origin doors (its `origin` is the threaded pass-through from `addStatement`, not a local build) are deferred to the P9 cascade, where the `TransientOrigin` carrier supplies known-size deps at the source. **Contradiction / vacuous-truth cluster landed next** (`dischargeContradiction`): the two direct cold-map doors — the incubator `contradiction` record `{expr, negation, cleanOp}` and the induction-step `vacuous truth` record `{expr, negation, recursionPremise}`, both three in-hand `std::string` locals plus a `"main"` literal — span their antecedents directly (no scratch — the strings are already stable locals; `StrSpan("main", 4)` is a static literal) and drop the `evResult`/`evHead` key EWVs, the two `deps` vectors, and the `make_pair` `OriginLine` temporaries. The vacuous-truth path's SECOND origin build — the `origin` `OriginLine` threaded into `addStatement` (the same three antecedents) — stays for the P9 cascade. **Disintegration / or-expansion cluster landed** (`disintegrateExprCore2` / `disintegrateExpr2`, prover.cpp): the negated-existence `expansion` pair, `trackExpansionHistory`'s `expansion` + child `disintegration` records (+ their paired `addRoutingMailOrigin` mailOut writes), the OR De-Morgan `disintegration` per-implication record, and the `or disintegration` per-branch record all move to the span-record doors. Fresh u_-stripped antecedents/keys reuse the existing `removeUPrefixScratch` ScratchString twin (the `.toStdString` heap copies dropped), held as named `ScratchString` locals so the spans stay live across the paired door calls; loop-invariant antecedents (`expandedSignature` / `expandedOrSignature` / `orSignature`) span stable `std::string` locals with the `OriginDep` built once outside the loop. The `origin`/`expOrigin`/`originExpansion`/`originDisintegration`/`originOrDisintegration` `OriginLine`s and the `expandVal`/`elemVal`/`impEv`/`branchEv` key EWVs are gone (`branchEv`'s key spans now also feed `orBranchStatements.insertImpl`). **Integration-expansion cluster landed** (`prepareIntegration`, prover.hpp): the four direct-door-only records — `expansion for integration` on the implication path, the OR sub-implication goal, the existence/and `expansion for integration`, and the category-selected `reformulation for integration {and,>[bound],>[]}` instruction — flip to the paired span-record `addOriginEncoded` + `addRoutingMailOrigin` doors. Their `base + "_integration_goal"` KEYs/antecedents are built on the per-slot string-scratch arena via a local `goalOf` concat lambda (explicit `sizeof-1` suffix length, `allocBytes` byte-bump so multiple spans coexist under one `ScratchScope`); the instruction KEYs span stable `integrationInstructionFor*` `std::string` locals directly. The instruction tag is category-computed into an `OriginTag` with an explicit unset-tag assert mirroring the string door's implicit `originTagFromString("")` contract. The `originImplicationExpansion`/`originGoal`/`originExpansion`/`originInstruction` `OriginLine`s and the `ev`/`iiv`/`iivHash` key EWVs are gone. The sibling `premise element` / `or branch assumption` records (which also thread into `addExprToMemoryBlock`) stay for the P9 cascade. **Anchor / recursion leaves landed** (prover.cpp): `handleAnchor`'s + `prehandleAnchor`'s `anchor handling` records (KEY the x-prefixed anchor, antecedent the source anchor expression, both in-hand `std::string` locals + a `"main"` literal) and the induction `recursion` record (empty antecedents, P1) flip to the span-record cold-map door; the `evReplacedAnchor`/`encVal` key EWVs and a pre-existing DEAD `handledOrigin` `OriginLine` (built, never used — the door had an inline `make_pair`) are dropped. **Internal-mail RECORD flip landed** — the L4 door left only the `addInternalMailOrigin` KEY spanned; L3 now flips the RECORD too across all ten `addInternalMailOrigin` sites (`updateGlobal`/`updateGlobalDirect` `implication` broadcasts in prover.cpp; the equi-class `emitOne` / absorb-path `equality1`, the substitution-rewrite `equality1`, `dischargeToBeProved`'s `or branch proven` + `validity name`, and `ordisMerge`'s `or convergence` in prover.hpp). Dynamic-length records (`1 + N` antecedents) fill a bounded stack `OriginDep[64]` with an in-loop overflow assert (mirroring the codebase's `[64]` premise-chain convention); the absorb-path `equality1` sites span their already-interned `rmiStrings`/`rmStrings` antecedent keys DIRECTLY (dropping the per-antecedent `std::string` heap copies), the substitution-rewrite + `or convergence` sites build their fresh `(=[k,v])` / `branchPrefix+djView` antecedents on the per-slot string-scratch arena (explicit length, coexisting under one `ScratchScope` live to the door). The routing-mail leaves that pair with an `addExprToMemoryBlock` call (`task formulation`, contradiction-LB) share a threaded origin and stay for the P9 cascade. Byte-transparent.

**P9 pass-through cascade landed (S6 — completes the L3 origin cascade).** The last in-scope `OriginLine` construction — the record threaded down the `addExprToMemoryBlock → addStatement → addEquality`/`addNegatedEquality`/`updateEquivalenceClasses` chain — is now the `TransientOrigin` carrier. The five functions' `origin` parameter retypes `const std::pair<std::string, std::vector<ExpressionWithValidity>>&` → `const TransientOrigin&` (an all-or-nothing signature cascade — one commit; the pass-through calls between the five are unchanged, the record is now uniformly `TransientOrigin`). Each in-function door that consumed the threaded record fires under `if (origin.present) { assert(origin.tag!= OriginTag::COUNT); span-door(…); }`: the `addExprToMemoryBlock` status-0/1/3 deposit, `addStatement`'s `trackHistory` deposit, `addEquality`/`addNegatedEquality`'s original-equality deposit, and `updateEquivalenceClasses`'s two cluster-B doors (the `MergeClassAccum::addOriginEnc` accumulator + the `trackHistory` cold-map `addOriginEncoded`). `present==false` reproduces today's empty-`std::pair<>` sentinel EXACTLY — the status-2 (toBeProved) and status-4 (do-not-disintegrate) paths return before any door, so an empty origin never reached `encodeOrigin`'s `originTagFromString("")` assert, and `if (origin.present)` preserves that (no deposit, no assert; the doors were unconditional before, so in every executed path `origin` is present and the guard never skips). The `addEquality`/`addNegatedEquality` in-function `mirroredOrigin` (a LOCAL `symmetry of equality` / `symmetry of inequality` record, always present) becomes a stack `OriginDep[1]` over the original equality's `(expr, validityName)` spans — its door stays unconditional. **Leaf callers:** `variable copy`, `task formulation`, `recursion` (or1/or2), compressor `premise`/`goal`, filter `CE_building_block` become `TransientOrigin{ true, tag, nullptr, 0 }` (empty antecedents); the empty status-2 heads become `TransientOrigin{}`. `updateGlobal`'s `equality1` (2 in-hand deps) and the vacuous-truth `addStatement`'s `vacuous truth` (3 in-hand deps) span their stable `std::string` locals directly. `premise element` / `or branch assumption` build their fresh `base + "_integration_goal"` antecedent on the per-slot string-scratch arena under a `ScratchScope` that SPANS the `addExprToMemoryBlock` pass-through (the pass-through's status-0/1/3 span door mints the antecedent synchronously into `originInterner`, never the arena — I-3), and each ALSO flips its two paired direct doors (`addOriginEncoded` + `addRoutingMailOrigin`, deferred to S6 because they share the local record with the pass-through); the `task formulation` / contradiction-LB routing-mail leaves flip their paired `addRoutingMailOrigin` in step, reusing the just-built `TransientOrigin`'s `tag`/`deps`/`depN`. **Two DECODE sites** re-form a `TransientOrigin` from an already-stored record: the `addExprToMemoryBlock` kernel's `frontOrigin` (the first `exprOriginMap` record per disintegration-product statement) and the mail-absorb drain's `oit->second.front` — each decodes to owned strings, then spans them into a bounded stack `OriginDep[64]` (the codebase's equality1-chain cap, with an in-loop overflow assert), the tag taken from the `IdOrigin` directly (kernel) or via `originTagFromString` (drain, whose snapshot is already a string `OriginLine`). The `exprWithValidity` / `eqltyEnc` key EWVs the doors no longer need are dropped. **The `OriginLine` typedef now survives only for the cross-LB heap-`Mail` `addOrigin` boundary** (the theorem-load `mailOut.exprOriginMap`, `broadcastMail`, the compaction fold, the runtime-`originTag` broadcast, and the `origins`-snapshot the drain decodes) — the `standardProcessing`-tree origin RECORD construction is span-complete. Byte-transparent.

**Cycle-innards batch landed (the S7-census phase-2 arm).** The disintegration / admission / integration-template subsystem the Status note flagged as a SEPARATE target is now 0% heap: all NINE mutually-recursive `standardProcessing`-cycle bodies flip to CodeQL-inventory `true` — the disintegration axis (`disintegrateExprCore2`, `disintegrateExpr2`, `disintegrateExprHypothetically`, `checkNecessityForEquality`) and the admission/integration/central-add axis (`addToHashMemory`, `prepareIntegration`, `prepareIntegrationCore2`, `updateAdmissionMap3`, `addExprToMemoryBlock`). The two shared foundations are the raw/id-run doors reused from the makeNormalizedKeysForAdmission machinery (`encodeIdVecKeyInto` + the `originals` raw `inner.mint`, the `appendLmvIdsRecord` head-install extension, the `makeAdmissionKeys` / `makeNormalizedKeysForAdmission` span-run overloads, `ruleJustificationFromString(StrSpan)`) and two zero-copy blob views (`IdOriginBlobView` reading `Codec<IdOrigin>` for the central-add front-origin, `IdVecKeyView` reading `Codec<IdVecKey>` for the `originals` chain — which also closed the `sortOriginalChainIndex` CodeQL blind-spot lie, its former per-compare heap `decodeKey.ids` now a zero-copy view). The disintegration WORKING form's last `LogicalEntity` island (`disintegrateExprCore2`'s per-hit `entityAt`) and the two `disintegrateExpr2` / `disintegrateExprHypothetically` `std::set<ExpressionWithValidity>` channels are gone — the `DisintProducts` channel is consumed directly via `forEachSorted` (== `EWV::operator<`). Only the TWO roots (`applyEquivalenceClassToAdmissionMapIntegration`, `standardProcessing`) and the compiled-definition layer remain heap. Byte-transparent (per-commit `--unit-tests` green; the batch's `main.py` byte-identity gate is the merge criterion).

**L6/L7 `addStatement` param span door landed (completes the *Scope deferral* deferral).** The clean `addStatement(StrSpan expr, …, StrSpan validityName, …)` door is now built in two coordinated flips. **L6 (statement param):** the `expr`/`eqlty` statement threaded through the add path — `addStatement` + `addEquality` / `addNegatedEquality` / `updateEquivalenceClasses` / `applyEquivalenceClassToNegatedEquality` / `updateWeakVariables` — threads as `StrSpan`; the two in-tree statement builders (`addStatement`'s `(=[a1,a0])` mirror, `applyEquivalenceClassToNegatedEquality`'s `emitNew` sibling `!(=[…])`/`(=[…])` forms) move to the per-slot string-scratch arena; the shape helpers ride their existing `StrSpan` overloads and `encodeExpression(StrSpan,StrSpan)` is the mint-order twin of the `EncodedExpression` form. **L7 (validity scope name):** the `validityName` param — a NameMap-interned RUNTIME scope name (`"main"` or a decoded scope, NOT the compiled-definition layer) — threads as `StrSpan` across the same six PLUS the apply path: `applyEquivalenceClass`, the four `applyEquivalenceClassTo{Rejected,Admission}Map{,Integration}` hooks, `mergeTwoEquivalenceClasses` (BOTH scope params `validityName`+`classBValidityName`), the `emitFromClassesAt` lambda, and the lookup-only leaf helpers `reduceEqClassIds` / `chooseCanonicalId` (pulled in — each consumes `validityName` solely for `nameMap.lookup(StrSpan)`; `chooseCanonical`, with no production caller, stays `std::string`). **BOUNDED cluster:** every above-cluster caller already holds `validityName` as a stable OWNED `std::string` — `applyEquiClasses`'s `nameMap.decode` class-validity copy fed to the whole apply* family + `applyEquivalenceClass`, `addExprToMemoryBlock`'s `ev.validityName` — and hands its span down, so NO caller signature cascades (the sole caller edit is the vacuous-truth `"main"` literal → `StrSpan("main", 4)`, StrSpan having no bare-literal ctor). Two additive foundation span twins round it out — `NameMap::strictAncestorNames(StrSpan)` (ancestor-scope enumeration) and `lookupOriginKey(StrSpan, StrSpan)` (the merge `alreadyKnown` origin probe) — and the scope comparisons retype to `equalSpans` / the id-form `isStrictAncestor(lookup(a), lookup(b))` (`memory.hpp · isStrictAncestor` comparability twin), with `deeperOf(StrSpan,StrSpan)` / `lookupStatementFlags` / `lookupStatementLevels(StrSpan,StrSpan)` already present. The one heap-origin leaf (`applyEquivalenceClass`'s not-yet-statified `std::pair<string,vector<EWV>>` path) materializes `validityName.toStdString` into its already-heap `ExpressionWithValidity` — a single leaf copy, no further threading. Compiled-definition-layer / non-cluster scope holders (`checkForEquivalence`, `prepareIntegration*`, `cleanUpAdmissionMap*`) keep `const std::string&`. Byte-transparent.

**L8 absorb entrance decode-snapshot elimination (landed).** The two heap `std::vector` snapshots the `standardProcessing::absorb` lambda decoded from the already-id-form cold mail — `sortedStatements` (`std::pair<EWV, std::set<int>>`) and `origins` (`std::pair<EWV, std::vector<OriginLine>>`), each `std::sort`ed into canonical order before its consumers ran — plus the equality-sync's `decodeClassesById` `std::vector<EquivalenceClass>` snapshot, are removed: the absorb iterates the cold mail ids directly and reproduces the former `std::sort` / `std::lower_bound` orders byte-for-byte via arena-backed decoded-lex indices (`int32_t[]` byte-bump on the per-slot `genScratchArenas`, the `applyEquiClasses` `vkOrder` / `sortStatementRows` recipe). The cold maps mint in insertion order, so BOTH the statement drain order AND the per-run origin order are observable (they drive `intExternalStatements` push order, rule-firing order, the `exprOriginMap` cold key mint order, and — routing only — the `originInterner` re-mint order, hence the deload stream); every walk therefore goes through an index, never raw cold-id order ([I-84](#i-84)). Only `compiledMap` stays heap. **S0 foundation (landed).** Three tested comparators in [`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp) — `originLineLessId` (THE CRUX: reproduces `OriginLine::operator<` = `std::pair<std::string, std::vector<EWV>>::operator<` exactly — the tag as a STRING via `compareSpans(originTagName)`, NOT the `OriginTag` enum order which differs, then the antecedent vector lexicographically under `EWV::operator<`, shorter-run-first), `decodedOriginKeyLess` (EWV-key order), `decodedStatementLess` (`pair<EWV, set<int>>::operator<` — EWV then ascending-levels lexicographic) — plus the `mailIdView` id→span dispatcher (`NameMap` / `ValueInterner` / `ColdStringTable`, so one templated comparator serves the internal per-LB and routing global id spaces) and the zero-copy `viewMailOriginBlob` / `mailOriginDepAt` `IntMailOrigin`-blob parse (unaligned-safe `int64` dep reads). Each new comparator has a heap-`std::sort` oracle unit test; `originLineLessId`'s deliberately includes a tag whose STRING order disagrees with the enum order, so an enum-order regression fails the test. **S1 statements (landed).** The statement drain iterates `mail.statements_` ids via a byte-bump index sorted by `decodedStatementLess` (internal `NameMap` / routing global `mailInterner`), materializing the two owning-`std::string` boundaries the `addExprToMemoryBlock` door then took into ONE reused per-drain buffer each (the honest residual at S1 — the `StrSpan` door and the drain's per-row `ScratchString` copy that retire these two buffers land in **L9 Item 3** below); the disintegration-signal probe packs the decoded key's `NameMap` ids directly (no re-lookup). The `.front` origin pick still reads the `origins` snapshot this stage. **S2 origins (landed).** The bulk `exprOriginMap` merge, the equality-class `equalityOriginMap` sync, and the per-statement `.front` origin pick all iterate `mail.origins_` cold ids directly: a cross-row index (`decodedOriginKeyLess`) reproduces the EWV-key row order, a per-run index (`originLineLessId`, peeking each `IntMailOrigin` blob zero-copy via `peekRecordBytes`) reproduces the sorted-run order. Internal deposits are pure id copies (`addOriginId(exprOriginMap, coldKey, IdOrigin{tag, deps})` — `coldKey` IS the `exprOriginMap` key, no re-mint); routing deposits decode the key + deps via `mailInterner` and re-mint through the span door (`addOriginEncoded(…, originInterner, keyExpr, keyVal, tag, deps, depN)`), preserving the `originInterner` mint order. The `.front` pick becomes a direct origin-key lookup (routing: the statement key ids ARE the origin key; internal: `lookupOriginKey` NameMap→originInterner span probe) + a decoded min-scan (min-of-run == the former sorted run's `.front`, no non-identical ties); the winning record's deps are copied onto the scratch arena before `addExprToMemoryBlock` (which mints into `originInterner` — I-3). The equality-sync's `decodeClassesById` stays a heap `std::vector<EquivalenceClass>` snapshot this stage. **S3 equi-class class snapshot (landed).** The equality-sync's `decodeClassesById` heap `std::vector<EquivalenceClass>` snapshot is replaced by an `EquivalenceClassView` cold RMW mirroring the `updateEquivalenceClasses` twin: a zero-copy view walk over `equivalenceClassesMap`'s cold blob run finds the single class containing BOTH equality args (classes at a scope are DISJOINT — at most one contains both), decodes ONLY that class (`recordAt`), appends the sorted-run origins to its `equalityOriginMap`, re-serializes it (`serializeEquivalenceClass`), and splices the run in original order (the matched class replaced, the rest raw blobs verbatim — a raw blob == re-serializing its decoded class, the codec being a lossless pure function of content, I-103) via `equivalenceClassesMap.inner.assignRun`. Byte-identical to the former `decodeClassesById` + origin-append + `assignClassesById`. With S3, the absorb consumes the cold mail (statements + origins) AND the equi-class store entirely by direct id iteration; only `compiledMap` stays heap. **S8 `_firing_check` extension of the `decodeClassesAt`/`decodeClassesById` flip to the phase-tree GATE callers:** `addStatement` / `checkNecessityForEquality` / `disintegrateExprHypothetically` (mint-free membership scans) read via a DIRECT `EquivalenceClassView` `peekRecordBytes` walk; `checkForEquivalence` (snapshot reused across an args loop that later mints) reads via a reactToHypo-style member-run snapshot; `cleanAdmissionMap` KEEPS its heap `decodeClassesAt` snapshot by decision (its `closureClasses` feeds the shared `canonicalizeUnderClasses(const std::vector<EquivalenceClass>&)`, whose heap-vector interface is out of scope). `decodeClassesAt`/`decodeClassesById` are RETAINED (the sacred dump, tests, `cleanAdmissionMap`, other callers, oracles). **S8 `_firing_check` C6 (decoded-lex iteration under a minting window):** `dischargeContradiction`'s `decodeToBeProvedSorted` heap snapshot → a decoded-lex `(original,validity)` key index over `intToBeProved` ids + a per-consumed-row `ScratchString` copy (the FIRST main-scope goal, byte-identical order); the `addExprToMemoryBlock` necessity sweep KEEPS `decodeToBeProvedSorted` by decision (its consumer `checkNecessityForEquality(const std::string&, Memory&, std::string)` is NOT `StrSpan`, an out-of-scope edge); the two `strictAncestorNames` stragglers (`updateEquivalenceClasses`, `applyEquivalenceClassToNegatedEquality`) → `strictAncestorSpans` (direct where mint-free, `ScratchString` snapshot where the emit window mints); `applyEquivalenceClassToNegatedEquality`'s `a`/`b`/`member` decode-strings → `ScratchString`s on `emitStrArena`. Per-commit `--unit-tests` green; the full-pipeline byte-identity gate is run by the maintainer. Byte-transparent.

**L9 in-scope `std::string`-residual span flips (landed).** The scalar `std::string` residuals the *Scope deferral* left in the admission / hash-memory add path flip to spans, closing the campaign's last in-scope heap. **Item 1 (`revisitRejected2` scope param, landed).** The drain-side revival `revisitRejected2(const std::string& markedExpr, Memory&, std::string validityName)` retypes its by-value `validityName` → `StrSpan`: the unconditional per-call scope-name copy is gone; the reentrancy-guard, `revisitInProgress` mint, and done-erase template-key probes consume the span through the `mintTemplateKey` / `lookupTemplateKey(StrSpan,StrSpan)` overloads (`markedExpr` stays `const std::string&`, so the mixed `(std::string, StrSpan)` call resolves unambiguously to the span overload — the string overload is non-viable, `StrSpan` has no `std::string` conversion), and the two non-compiled `const std::string&` sinks it still reaches — `emitIntegrationRevivalToInternalMailIn` and `cleanAdmissionMap`, both kept OUTSIDE this campaign's scope — take a SINGLE `validityName.toStdString` materialized once past the early-return guards, so the scope-name copy is now conditional on reaching the revival path rather than paid on every call (the common guard-rejected path pays none). The `updateAdmissionMapRecursion` caller drops its per-match `.toStdString`; the `drainAdmissionKeysAlgebra` caller passes its owned `std::string` scope name unchanged (implicit span). **Item 2 (`updateAdmissionMapRecursion` value working forms, landed).** The recursion admission-key rewrite's `argSpans[outIdx].toStdString` output arg drops its `.toStdString` — now a `StrSpan` slice of the caller-stable `expression`, fed as a span to `nameMap.encode`, the substitution VALUE, and the remaining-args set. Its `std::map<std::string,std::string> replacementMap` collapses to a stack `StrReplacement[1]` consumed by `replaceKeysToString` (the byte-exact single-key `ce::replaceKeysInString` twin; `newKey` stays a `std::vector<std::string>` — the I-90 id-vector working set fed to `encodeValueVector`, kept per I-90's contract). Its `std::set<std::string> newRemainingArgs` becomes a sorted-unique `StrSpan` run on the per-slot byte-bump `genScratchArenas` (the `createAuxyImplication` untouchables recipe, row 41 — spans alias `valRemaining`'s stable `std::set` nodes + the output-arg span, `compareSpans`-sorted + dedup): membership rides `std::binary_search(compareSpans)`, and the sorted-encode swaps `encodeValueSetSorted` → `encodeValueSpanSetSorted` (row 50, byte-identical id vector). `valKey` / `valRemaining` (decoded snapshots) + `kMarkerMap` / `newMarkerMap` (decoded working forms) + `core` / `kOutputArg` (the `operators` / `coreExpressionMap` compiled-definition keys) stay heap by decision. **Item 3 (`addExprToMemoryBlock` `StrSpan` door, landed — the L8 S1 residual).** The central hash-memory add door retypes BOTH string params — `const std::string& expr` → `StrSpan expr` and by-value `std::string validityName` → `StrSpan validityName`. The span-native fast paths (Site F/H ancestor-dedupe, axed-var scan, status-4 statement-only) consume the spans directly (`nameMap.encode(StrSpan)`, `encodeExpression(StrSpan,StrSpan)`, `getArgsSpans`, `containsSpan`, `addOriginEncoded(StrSpan)`), so the common early-out paths materialize NOTHING — the win over the old by-value `validityName` copy paid on every call. Past the status-4 return, a SINGLE `exprStr` / `validityNameStr` pair is materialized once and fed the compiled-definition-layer + prover doors that keep `const std::string&`: `disintegrateExpr2`, `ce::extractExpression`, `ExpressionWithValidity`, `checkForEquivalence`, `checkNecessityForEquality`, `prepareIntegration`, `updateAdmissionMap3` (and `isCompactImplication`, which reuses `exprStr`'s `std::string` ops for zero-risk byte identity). The ~19 callers ripple trivially: stable `std::string` expr/scope variables convert implicitly, the `"main"` literals become `StrSpan("main", 4)` (`StrSpan` has no bare-literal ctor), and the **`standardProcessing::absorb` statement drain drops its two reused heap `std::string` buffers** (`statement` / `vName`) for per-row `ScratchString` copies on the string-scratch arena — the door mints into the NameMap / mailInterner, so a raw `decodeView` span would dangle (I-3); the established S4 consumer-edge recipe. `compressor.cpp` / `filter.cpp` out-of-tree callers flip their `"main"` literals in step. With Item 3, the `standardProcessing` tree's central add door is span-native and the L8 S1 "honest residual" is retired — `compiledMap` and the compiled-definition string layer are the only heap that remains by decision. Byte-transparent.

**S6 regex sweep (landed).** The **Status. COMPLETE.** claim above was overstated: seven regex readers inside the fan-out (`allowedForMail`, `extractRemainingArgs` ×2, the `addExprToMemoryBlock` assert pair, `extractSubstringsForAuxy`, `countPatternOccurrences` + `extractMaxIterationNumber`, `isProved`) still carried per-match `std::regex` matcher allocations and satellite string/set/map transients. All seven now run regex-free lexical twins (`str_ops` occurrence/shape scanners + the new prefix-shape core `matchItLevPrefixOccurrenceAt`/`scanItLevPrefixOccurrences`, the distinct-collection `scanSingleDistinctIntLev`, and the existence scans `containsItLevPrefixShape`/`containsCDigit`), `std::regex` demoted to in-test oracles; `allowedForMail` is span-native end to end (its memory.cpp caller decodes via `decodeView`). Production regex machinery over `prover.cpp` + `prover.hpp` is now zero (the only `grep -E "std::regex|sregex|smatch"` residuals are the word "Mismatch"/"mismatching"). **Carve-out (S8 census correction):** `filter.cpp::readSimpleFacts` retains a `std::regex` `simple_facts_*.txt` filename matcher — BATCH-SETUP-only (it runs once at load, before the prover, over no phase call tree), NOT a statification target, the same out-of-scope class as the compile/init seams. The TRUE post-S6 statement: inside the phase call trees the remaining heap is (1) the compiledMap complex behind the `compiledEntity` / `coreConfig` reader fence (`I-137`), (2) the compiled-definition/parse-install boundary (the `ce::` working forms; the `extractRemainingArgs` returned set; the `createAuxyImplication` tuple returns; the D-172 RMW read snapshots — `admissionRecordsAt` / `rejectedRecordsAt` / `decodeValueVector*` AND their integration twins `admissionIntegrationRecordsAt` / `rejectedIntegrationRecordsAt` (all sanctioned P-class read snapshots, so a census hit on any is traceable to this named list); the enumerated edge materializations), and (3) the sacred dump's read paths (Rule 14). See also [I-137](#i-137).

**S7-census extension (2026-07-03).** The "TRUE post-S6 statement" above is scoped to the `standardProcessing` fan-out only. The S7 final census (VERDICT: FINDINGS) found live heap OUTSIDE that scope, on other phase-call-tree paths the earlier layers never touched: `checkLocalEncodedMemoryStatic`'s interior (the phase-2 firing burst — the candidate vector-of-sets, the `NormKey`/`Int16SetKey` heap constructions, the per-hit `recordsAt` value decode, `orderedPremises`), the minor residuals (`reduceEqClassIds` return, `updateWeakVariables` `clsScratch`, `applyEquiClasses` `classValidity`, `applyEquivalenceClass` deposit leaves, `applyEquivalenceClassToNegatedEquality` a/b/member, `fillMailOut`'s per-row decode), and the remaining `decodeClassesAt`/`decodeClassesById`/`decodeToBeProvedSorted`/`strictAncestorNames` heap snapshots on phase-tree gate paths. The user ruled these a full statification TARGET (subsessions **S8 `_firing_check`**, **S9 `_admission_gates`**, **S10 `_integration_templates`**), not a sanctioned boundary. **S8 (this branch) landed** the firing check + the residuals + the twin conversions, with two documented boundaries KEPT heap by window verdict — `cleanAdmissionMap`'s `decodeClassesAt` snapshot (feeds the shared `canonicalizeUnderClasses(const std::vector<EquivalenceClass>&)`) and `addExprToMemoryBlock`'s necessity-sweep `decodeToBeProvedSorted` (feeds `checkNecessityForEquality(const std::string&, …)`). S9/S10 target the admission gates + integration templates and remain pending. **Pre-cycle rump batches additionally flip the remaining lone decode-`std::string`s the phase-3 discharge sweep left:** `dischargeContradiction`'s whole body is now span/`ScratchString`-native (`expr` / `validityName` / `negation` / `theorem` / `cleanOp` / `mbKey` / `negCleanOp` / `recursionPremise`) — a per-iteration `ScratchScope` on `scratchArenas` bounds the sweep (the RAII-per-iteration free the heap version had), `theorem` is a bare `decodeView` (no NameMap mint between it and its `SealedString::copyFrom`; the cleanOp/mbKey work reads the frozen skeletonInterner and addOriginEncoded mints originInterner only), `expr` / `validityName` / `recursionPremise` are `copyFrom` onto `sArena` (the `encode(negation)` / `addStatement` NameMap mints intervene), `negation` a `negateScratch` span on `sArena` (survives NameMap mints), `cleanOp` / `mbKey` zero-copy `exprKeyView` sub-spans, `negCleanOp` an explicit-length `"!"`-prefix build on `sArena` — flipping its inventory row. The zero-heap phase-tree claim is re-assertable only after the S10 re-census returns zero findings.

**Roots batch landed — the campaign's finish line: ZERO false rows.** The two last-remaining `false` rows in the `standardProcessing`-rooted CodeQL inventory — `standardProcessing` itself (the root) and `applyEquivalenceClassToAdmissionMapIntegration` (the integration-side admission hook reached via `applyEquiClasses`) — flip to `true`, so the entire `standardProcessing` call tree now reads **0% heap** by the inventory bar. `standardProcessing`'s body: the `std::pair<bool,bool> sig` disintegration-signal local → the out-param `ColdMail::getDisintegrationSignal(int64_t, bool&, bool&)` door (c1); the step-4b `std::string _deferredValidity` → a string-tier `ScratchString::copyFrom` on `scratchArenas` (c2, mandatory copy — `cleanUpExpressions` re-encodes into NameMap, [I-3](#i-3)); the internal-channel `exprOriginMap` bulk merge's heap `IntMailOrigin`/`IdOrigin` → a `peekRecordBytes` + `viewMailOriginBlob` + stack `int64 deps[64]` read feeding the POD cold `addOriginId` (c3, mirroring the routing branch above it); the statement drain's heap `IntMailStatementKey` decode → a zero-copy `keyAt` + `memcpy` field reads + a stack `int lvRun[256]` (c4, a loud Rule-19 level-run cap — widen-not-truncate on STOP); the equality-class `equalityOriginMap` sync's heap `EquivalenceClass mcls` decode + `serializeEquivalenceClassInto` → a `MergeClassAccum` seeded from the matched class's zero-copy `EquivalenceClassView` (verbatim members/levels/origins on an empty accum), mail origins appended via the accum's byte-twin `addOriginEnc`/`addOriginId` doors, `serializeInto` replacing the free serializer (c5 — the batch's highest byte-identity risk; `serializeEquivalenceClassInto` loses its only in-tree caller and leaves the tree, retained as its test oracle). `applyEquivalenceClassToAdmissionMapIntegration`'s three flagged decode-strings — the `std::pair<std::string,std::string> tv` walk-entry decode and the drain loop's `std::string newKeyBare` / `std::string depositValidity` — become direct reuse of the already-present `amiStrings` spans (`amiTmplSpan`/`amiVldSpan`/`bareSpan`/`depSpan`), all I-3-safe zero-copy `keyAt` reuse (no `amiStrings` mint intervenes) (c6). **The `IntInstruction` working-struct boundary — ELIMINATED (the follow-on batch).** The roots batch left the integration path constructing heap `IntInstruction` locals (`std::vector<IntLogicalEntity> data`, each owning `std::vector<int32_t> elements`) at the `instructionAt` / `encodeWorkInstruction` / drain-rebuild sites shared by `applyEquivalenceClassToAdmissionMapIntegration`, `isAdmittedIntegration`, and `updateAdmissionMapIntegration`. CodeQL does **not** flag these (a `std::vector` INSIDE a user struct is a blind spot), so the rows read `true` while genuine heap remains — surfaced here, not hidden. This was a tree-wide boundary, not introduced by the roots batch; the dedicated follow-on batch removed it with four new `ArenaIntegrationMap` doors ([I-133](#i-133)) — `loadInstructionInto` (replacing `instructionAt` then `loadFromIntInstruction`), the mint-order-safe `flattenWorkInstructionInto` (preserving `encodeWorkInstruction`'s exact per-entity category / elements / signature / definedSet then markedGoal-LAST mint sequence), `findOrEmplaceFlattened`, and `findOrEmplaceWork` — wired across all three consumers plus the `prepareIntegration` marker-key site with fresh-interner Rule-18 twins (the batch note below). No heap `IntInstruction` / `IntLogicalEntity` is constructed on the `standardProcessing` tree any more; the heap forms (`instructionAt` / `encodeWorkInstruction` / `findOrEmplace(IntInstruction)` / `loadFromIntInstruction`) are retained in place as the twin oracles. The compiled-definition layer ([I-137](#i-137)) is now the SOLE remaining standing OPEN violation on the `standardProcessing` tree. With the roots landed the S7/S8/S9/S10-census forward claims above are closed for the `standardProcessing` tree: its inventory has zero false rows; with the `IntInstruction` boundary eliminated (above) only the compiled-definition layer carries heap, by explicit decision. Byte-transparent (per-commit `--unit-tests` green + the new c1 twin `disintegration_signal_out_param_matches_pair`; the whole-hook byte-identity gate is the maintainer's `main.py` run).

**IntInstruction working-struct elimination batch — COMPLETE.** The `IntInstruction` working-struct boundary the roots batch flagged (above) is removed via the four `ArenaIntegrationMap` doors landed in [I-133](#i-133) (`loadInstructionInto` / `flattenWorkInstructionInto` / `findOrEmplaceFlattened` / `findOrEmplaceWork`). The PERSISTENT byte layouts (`Codec<IntegrationEntry>`, the `valueInterner` id encodings, every deload stream) are FROZEN; the mint-order-preserving flatten (per entity: category, elements, signature, definedSet; markedGoal LAST) makes strict byte-identity achievable without moving one stored byte. **c2 wired sites A + B** — `isAdmittedIntegration` and `updateAdmissionMapIntegration` drop the `IntInstruction pairFirst = entryMap.instructionAt(snapEi)` → `loadFromIntInstruction(wiSrc, pairFirst, …)` bridge for a direct `entryMap.loadInstructionInto(snapEi, wiSrc, mb.valueInterner)` (no heap `IntInstruction`). **c3 wired site D** — the `prepareIntegration` marker-key ensure-entry drops `admEntries.findOrEmplace(encodeWorkInstruction(markedCopy, …))` for `admEntries.findOrEmplaceWork(markedCopy, mb.valueInterner)`: the door flattens `markedCopy` into a byte-bump run, minting ALL fields UNCONDITIONALLY in the frozen `encodeWorkInstruction` order (markedGoal LAST) BEFORE the id-compare (the mint-on-HIT rule), then find-or-emplaces the run — no heap `IntInstruction`. **c4 wired site C** — the `applyEquivalenceClassToAdmissionMapIntegration` ami hook's three touch points: C1's read loads via `innerMap.loadInstructionInto(innerIdx, wiSrc, memoryBlock.valueInterner)`; C2's flatten replaces the `encodeWorkInstruction` + manual `amiInstrPool.push_back` block with `innerMap.flattenWorkInstructionInto(newInstrWork, memoryBlock.valueInterner, amiInstrPool)` (minting at the identical loop position, so the walk-order of the mints is unchanged; the `amiInstrPool` layout moves to entity-count-FIRST / markedGoal-LAST); C3's drain replaces the heap `IntInstruction` rebuild + `findOrEmplace` with a byte-bump copy of the pool run (a page straddle is possible — `amiInstrPool` is a `PagedVector`) onto `innerMap`'s own `*arena` under a `ScratchScope`, then `findOrEmplaceFlattened(run, len)` (mints nothing). No heap `IntInstruction` remains at any of the three touch points. **c5 (close-out)** flips the roots-batch OPEN-violation sub-paragraph (above) to ELIMINATED — leaving the compiled-definition layer ([I-137](#i-137)) the SOLE remaining documented OPEN violation on the `standardProcessing` tree — and regenerates the Tier-1/Tier-2 inventory: the four doors enter as `true` rows, every existing row stays `true`, zero `false` rows, zero `true→false` flips. Byte-transparent (per-commit `--unit-tests` green; the whole-hook byte-identity gate is the maintainer's `main.py` run).

**Mail-pull batch — the MPU 0.1 statification FINALE: `performElemPhase1/2/3` production paths at 0% heap.** The last three production-reachable heap functions in the `performElemPhase1` tree — `Codec<Mail>::deserialize(const char*, int32_t)` and `MailLog::readBlob` (rows 317/318, which inherently return a heap `Mail` — they ARE the twin oracle) and `MailLog::readBlobInto` (row 371, a `std::vector<char>` reassembly buffer) — are removed from the tree, and the CodeQL-blind heap on `Codec<Mail>::deserializeInto(const char*,...)` (row 370, the `std::vector<int64_t> deps` inside its transient `IntMailOrigin` record — the same `std::vector`-inside-a-user-struct blind spot the `IntInstruction` batch closed) is converted, so the row is HONESTLY `true`. Mechanism: the frozen-`Codec<Mail>`-blob parse becomes ONE templated free body `deserializeMailBlobInto<Src>(Src&, RoutingColdMail&)` ([`memory.hpp`](../../GL_Quick_VS/GL_Quick/src/memory.hpp)) driven by two sequential POD sources — `CharMailSource` (contiguous, the retained char* oracle) and `PoolMailSource` (straddle-aware over the `PagedVector<char>` mail blob pool, the production sink), both `memcpy`-only (the straddle path assembles a field byte-run-by-byte across pages, never a `reinterpret_cast` of a straddling pool pointer). The body feeds the id-form `RoutingColdMail` write doors: statements via `insertStatement`, origins via the POD `addMailOriginRecord(origins_, key, tag, deps, depN, INT_MAX)` from a stack `int64_t d[kMaxOriginDeps]` — NO owning `IntMailOrigin` built (byte-identical to the record overload by `add_mail_origin_record_pod_matches_intmailorigin`). `inbox.ensureArena` is the body's FIRST statement, before both loops ([G-55]); the `depCount <= kMaxOriginDeps` assert fires BEFORE the stack fill and `src.atEnd` closes the parse (both loud Rule-19 tripwires). `MailLog::readBlobInto` drops its reassembly buffer and forwards to the new `Codec<Mail>::deserializeInto(const PagedVector<char>&, uint32 start, uint32 len, RoutingColdMail&)` pool overload. `MailLog::pull<Inbox>` (an `if constexpr` template that textually named BOTH `readBlobInto` AND `readBlob`) is split into two non-template overloads with DUPLICATED chain walks — `pull(const Memory*, RoutingColdMail&)` (production, names ONLY `readBlobInto`) and `pull(const Memory*, Mail&)` (the heap oracle, names `readBlob` + `mergeBatchInto`); production resolves to the former, tests to the latter, so `readBlob` / `deserialize` / `mergeBatchInto` have no production caller and leave the phase-1 inventory tree (dispatch-only, semantics preserved, oracle retained — the standard twin-oracle severance, NOT a Rule-8 change). **The FROZEN `Codec<Mail>` blob layout is untouched** — `serialize` (both overloads) unchanged; this batch only changes how the frozen bytes are READ. **Result:** the PRODUCTION paths of `performElemPhase1/2/3` are now 0% heap by the inventory bar; the ONLY heap remaining in those trees is (1) the sanctioned `hashburst_trace.txt` dump island (the `performElementaryLogicalStep` dump, Rule 14) and (2) the compiled-definition layer ([I-137]) — a data-on-heap layer burst-read through the `compiledEntity` / `coreConfig` span fences. The cross-LB heap-`Mail` WRITE/commit seam (`Codec<Mail>::serialize`, the theorem-load `mailOut.exprOriginMap`, `broadcastMail`, the compaction fold) stays heap but is OUT of the phase-1/2/3 trees — it runs only at the single-threaded commit barrier, on the frozen layout. Unit tests: the existing `deserialize_into_round_trips` now exercises `CharMailSource` + the shared body via the char* wrapper; `deserialize_into_pool_matches_char` (`test_mail_log.cpp`, the pool-vs-char straddle twin — default 8 KiB page AND a 32-byte page + 2-byte lead pad forcing PoolMailSource's mid-field straddle path); `pull_routing_cold_mail_matches_heap` (the first end-to-end test of the production pool-cursor path, with an origins-only newest batch re-covering the [G-55] `ensureArena`-first path). Byte-transparent.

**Code.** `prover.hpp` (`canonicalizeUnderClasses`, `cleanUpExpressions`, `updateEquivalenceClasses`, `applyEquivalenceClassToNegatedEquality`, `applyEquiClasses`; `applyEquivalenceClass` products sink [L5]; `addStatement` return→`void` + `newStatements` `PagedVector<IntEncodedExpr>&` out-param, and the `updateEquivalenceClasses`/`applyEquivalenceClassToNegatedEquality` sinks + `cleanUpExpressions` dead-branch drop that ride it [L5]; further functions batch by batch); `prover.cpp` `addExprToMemoryBlock` `sortStatementRows` consumption edge + vacuous-truth throwaway [L5]; `memory.hpp` `sortStatementRows` [L5 foundation door]; `prover.hpp` (`ordisMerge`, `updateAdmissionMapIntegration`, `updateAdmissionMapRecursion` decl `const std::string&`→`StrSpan`; `applyEquiClasses::mergeProducts` consumer edge) + `prover.cpp` (`updateAdmissionMapRecursion` def; `addExprToMemoryBlock` stmts-loop consumer edge) [S4 consumer edge flip]; the `ScratchArena&` peek overload in `memory_infra/cold_hash_map.hpp` (`HashMap::peekBlobContiguous`, over `readBlob(int32_t,char*)` / `blobAt(int32_t,int32_t,char*)`) and `memory_infra/typed_cold_map.hpp` (`TypedCold::peekRecordBytes`); the `ScratchArena&` overload of `ChangedClassesBuffer::classViewAt` (declared in `memory_infra/changed_classes_buffer.hpp`, defined in `memory.hpp`). Unit tests: `test_str_ops.cpp::canonicalize_under_classes_subst_twin`, `test_typed_cold_map.cpp::blob_peek_arena_matches_vector`, `test_memory.cpp::changed_classes_view_arena_matches_vector` (arena-overload `EquivalenceClassView` byte-identical to the `std::vector<char>` oracle, contiguous + forced page straddle), `test_memory.cpp::encode_value_span_set_sorted_byte_identical` (the `encodeValueSpanSetSorted` value-set span door's id vector byte-identical to the `std::set<std::string>` `encodeValueSetSorted` oracle on fresh interners — reverse-lex + dup, length tiebreak, single, empty), `test_memory.cpp::sort_statement_rows_matches_std_sort_ewv` (the `sortStatementRows` L5 index byte-identical to `std::sort(std::vector<EWV>)` — duplicate key + validity tie-break). `memory.hpp` (`insertInternalStatement` / `addInternalMailOrigin` `StrSpan` doors [L4 internal-mail door span flip]; the callers wired batch by batch). Unit tests: `test_memory.cpp::internal_mail_statement_span_door_twin` / `internal_mail_origin_span_door_twin` (each L4 span door's `ColdMail` deposit byte-identical to the `ExpressionWithValidity` overload through `makeHeapMail`). `memory.hpp` (`OriginDep` / `TransientOrigin` / `encodeOriginSpans` foundation + the five span-record door overloads `addOriginEncoded` [cold + heap], `MergeClassAccum::addOriginEnc`, `addInternalMailOrigin`, `addRoutingMailOrigin` [L3 origin-record span-antecedent doors]; the callers wired cluster by cluster). Unit tests: `test_memory.cpp::encode_origin_spans_twin` (the encoder byte-identical to `encodeOrigin`, depN 0/1/3), `add_origin_encoded_spans_twin` (heap `IdOriginMap` door + the D-49 cap-full replacement), `add_origin_encoded_cold_spans_twin` (cold `exprOriginMap` door + dedup), `merge_accum_add_origin_enc_spans_twin` (accumulator `serialize` twin), `internal_mail_origin_record_span_door_twin` / `routing_mail_origin_record_span_door_twin` (each mail span-record door byte-identical to the `OriginLine` overload through `makeHeapMail` / `decodeMailOutOrigins`). `prover.hpp` (`addExprToMemoryBlock` / `addStatement` / `addEquality` / `addNegatedEquality` / `updateEquivalenceClasses` `origin` param → `const TransientOrigin&`; the two `updateEquivalenceClasses` cluster-B doors; the `premise element` / `or branch assumption` scratch-antecedent callers + `updateGlobal` / vacuous-truth / mail-drain callers) + `prover.cpp` (the four defs' `origin` param, the kernel `frontOrigin` `OriginDep[64]` build, the recursion / contradiction / `variable copy` leaf callers) + `compressor.cpp` / `filter.cpp` (out-of-tree `addExprToMemoryBlock` callers) [S6 P9 pass-through cascade — no new door; the foundation tests above cover the encode path, the wiring is exercised by the full-pipeline byte-identity gate]. `prover.hpp` + `prover.cpp` (`addStatement` / `addEquality` / `addNegatedEquality` / `updateEquivalenceClasses` / `applyEquivalenceClassToNegatedEquality` / `updateWeakVariables` statement param → `StrSpan` [L6]; the same six's `validityName` + `applyEquivalenceClass` + the four `applyEquivalenceClassTo{Rejected,Admission}Map{,Integration}` hooks + `mergeTwoEquivalenceClasses`'s two scope params + the `emitFromClassesAt` lambda + the `reduceEqClassIds`/`chooseCanonicalId` leaf helpers → `StrSpan` [L7]) + `memory.hpp` (`NameMap::strictAncestorNames(StrSpan)` + `lookupOriginKey(StrSpan,StrSpan)` additive span twins [L7 foundation]). Unit tests: `test_memory.cpp::namemap_strict_ancestor_names_span_matches_string` (span twin == string across a three-deep chain + un-interned empty), `origin_key_lookup_span_matches_string` (span verdict+key == string form before/after mint + un-interned validity); the param flips ride the existing twin-tested span doors + the full-pipeline byte-identity gate. `memory.hpp` (`mailIdView` overloads, `MailOriginBlobView` + `viewMailOriginBlob` + `mailOriginDepAt`, `originLineLessId` / `decodedOriginKeyLess` / `decodedStatementLess` comparators [L8 S0 foundation]) + `prover.hpp` (`standardProcessing::absorb` statement-drain cold-id iteration [L8 S1]; origins bulk-merge + equality-sync cross-row + per-run cold iteration, `.front` direct-lookup min-scan [L8 S2]; equality-sync `decodeClassesById` snapshot → `EquivalenceClassView` cold RMW splice [L8 S3]). Unit tests: `test_memory.cpp::origin_line_less_id_matches_std_sort_originline` (the crux — tag string-vs-enum-order disagreement, dep tie, shorter-vector prefix, empty deps), `decoded_origin_key_less_matches_ewv_order`, `decoded_statement_less_matches_std_sort_pair` (both NameMap + ColdStringTable id spaces); the S1 drain flip rides them + the full-pipeline byte-identity gate. `prover.hpp` + `prover.cpp` (`revisitRejected2` `validityName` param `std::string`→`StrSpan`; the `validityNameStr` single materialization for the `emitIntegrationRevivalToInternalMailIn` + `cleanAdmissionMap` `const std::string&` sinks; the `updateAdmissionMapRecursion` caller drops `.toStdString` [L9 Item 1 — no new door, rides the twin-tested `mintTemplateKey`/`lookupTemplateKey(StrSpan,StrSpan)` overloads + the full-pipeline byte-identity gate]) + `prover.cpp` (`updateAdmissionMapRecursion` `inputOutputArg` → `StrSpan`, `replacementMap` → `StrReplacement[1]`+`replaceKeysToString`, `newRemainingArgs` → sorted `StrSpan` run + `std::binary_search(compareSpans)`, `encodeValueSetSorted` → `encodeValueSpanSetSorted` [L9 Item 2 — rides `replaceKeysToString` / `encodeValueSpanSetSorted`, both twin-tested, + the full-pipeline byte-identity gate]) + `prover.hpp` + `prover.cpp` (`addExprToMemoryBlock` `expr` / `validityName` params `const std::string&` / `std::string` → `StrSpan`; the single `exprStr` / `validityNameStr` materialization for the compiled-definition + prover `const std::string&` doors; `expr.find` → `containsSpan`, `expr.size/compare/[]` → `exprStr` ops, `encodeExpression` wrapper drop; the ~19 callers' `"main"` → `StrSpan("main", 4)` incl. `compressor.cpp` / `filter.cpp`; the `standardProcessing::absorb` drain's two reused `std::string` buffers → per-row `ScratchString` copies [L9 Item 3 — no new door, rides the twin-tested `encodeExpression(StrSpan,StrSpan)` / `addOriginEncoded(StrSpan)` / `nameMap.encode(StrSpan)` doors + the full-pipeline byte-identity gate]). See also [I-128](#i-128), [I-84](#i-84), [I-95](#i-95).

---

<a id="i-139"></a>
## I-139 — wipe-subtree membership is the `validityNodes` forest walk, provably equivalent to the retired text-prefix predicate


**Scope.** `Memory::wipeSubtree` (`memory.cpp`) and its membership builder `NameMap::collectClosedSubtreeIds` (`memory.hpp`). Established by the S1 `_phase3_drain` subsession of the transient-statification close-out.

**Rule.** The closed-subtree membership test inside `wipeSubtree` is `nameMap.ancContains(id, closedVid)` — a `validityNodes` parent-chain walk (self included) — materialized once per wipe by `collectClosedSubtreeIds` into a 4 KB stack bitmap plus an ascending id vector on the per-slot gen-scratch page tier ([I-124](#i-124)). Every NameMap-id sweep consults the bitmap through the guarded `inClosedBit` lambda (the negative-cast twin of the former `closedIds.count`); step 11 mints the ascending vector (the D-190 order). The retired predicate — decode EVERY minted id to an owned `std::string` and test `v == closedScope || v starts with closedScope + "_boundary_"` over a heap `unordered_set` — is gone. The ONE non-NameMap id space, `expandedImplications`' `lbStateInterner` scope half, keeps the text gate as a zero-copy span twin over `decodeView`s (byte-semantics identical, including the strict `>` length gate).

**Why equivalent (the theorem, not a grammar hope).** The precondition set, assert-enforced at EVERY mint path (`encodePush` payloads AND both `encode` overloads' flat-name paths): a payload / flat name (a) contains no `"_boundary_"`, (b) does not END with `"_boundary"`, and (c) does not START with `"boundary_"`. (b)/(c) close the delimiter's period-9 self-overlap: an overlap-shaped fragment passes (a) yet creates a spurious delimiter occurrence at a concatenation junction — e.g. parent `P` + payload `"X_boundary"` → child `"P_boundary_X_boundary"`, whose own child reads `"…_boundary_boundary_(q)"` and a text-prefix scan diverges from the recorded parentage. Under (a)-(c): (1) no minted fragment can embed, extend into, or complete a delimiter occurrence; (2) `NameMap::encode` of a name CONTAINING the delimiter splits at the LAST delimiter, recursively encodes the prefix as the parent, and `encodePush`es the tail — so even a mail-absorbed deep name gets the maximal-split parent chain (no first-mint-wins parentage hazard); (3) therefore every minted name's delimiter decomposition is unique and equals its recorded parent chain: text-prefix-with-delimiter == forest ancestry; flat names (`ValidityNode{0,0}`) contain no delimiter, can never carry the prefix, and their chain is {self} — both predicates say "self only". Adversarial shapes are impossible at origin (the asserts — such a name cannot even be constructed in a unit test without aborting), so the twin coverage targets multi-delimiter grandchild names (nested `encodePush` AND whole-string `encode`), flat expression names, and the literal-prefix sibling trap instead.

**How to spot violations.** A decode-to-`std::string` loop over all NameMap ids inside `wipeSubtree`; a new sweep predicate that string-compares instead of consulting the bitmap; an unguarded bitmap index at a former-`count` site (the negative-cast values at the mail filters are LEGAL input and must short-circuit exactly as `count(negative)` did); a TOLERANT guard at a decode-heritage site (the 10a LMV `validityId` read and the 10b partition low-half read consume stored minted vids — out-of-range there is corruption and must ASSERT, as the retired per-record `decode` did, never silently classify as "not closed").

**How to fix on violation.** Route NameMap-id membership through `inClosedBit` / `collectClosedSubtreeIds`; keep non-NameMap id spaces on span twins over `decodeView`; never index the bitmap outside `[1, nameHighWater]`.

**Code.** `memory.hpp` (`NameMap::collectClosedSubtreeIds`, beside `ancContains`); `memory.cpp` (`Memory::wipeSubtree` — bitmap prelude, `inClosedBit`, the `expandedImplications` span sweep, step-11 `closedAsc` mint loop). Unit tests: `test_memory.cpp::collect_closed_subtree_matches_string_predicate` (the retained decode-all-ids string-predicate oracle, per-id bit + ascending-vector equality), `wipe_subtree_vid_scope_sweep` (whole-function sweep incl. the `expandedImplications` span-twin branches). See also [I-2](#i-2), [I-50](#i-50), [I-84](#i-84), [I-124](#i-124).

---

<a id="i-140"></a>
## I-140 — `NameMap::strictAncestorSpans` fills sorted strict-ancestor SPANS into a caller stack array; `MAX_SCOPE_DEPTH` is the depth tripwire


**Scope.** `NameMap::strictAncestorSpans` (`memory.hpp`, beside the two `strictAncestorNames` overloads), `ExecutionParameters::MAX_SCOPE_DEPTH` (`parameters.hpp`). Established by the S2 `_sanitize` subsession of the transient-statification close-out; consumed by the end-of-burst sanitize twins' scope walks. **S8 `_firing_check` consumers:** `updateEquivalenceClasses`'s ancestor merge loop reads the spans DIRECTLY (mint-free — `mergeTwoEquivalenceClasses` only `nameMap.lookup`s and mints `originInterner`, reads ancestor `equivalenceClassesMap` runs read-only per D-44/I-31); `applyEquivalenceClassToNegatedEquality`'s ancestor emit loop MINTS (`emitFromClassesAt → emitScratch → addStatement`), so it snapshots the spans to `ScratchString`s on `emitStrArena` BEFORE the first emit (`ancStable[]`).

**Rule.** `strictAncestorSpans(v, out, cap)` is the zero-heap twin of `strictAncestorNames`: it resolves @p v via the non-minting `lookup(StrSpan)`, fills `out` with `decodeView` spans of the strict ancestors (self excluded), sorts them by `compareSpans` (byte-lex == the string overload's `std::sort`), and returns the count — 0 for an unknown or root name. The strict ancestors are pairwise-distinct interned names, so the order is strict and the permutation unique (tie-free). The spans alias the NameMap's cold byte pool and are valid until the next mint into THAT NameMap ([I-3](#i-3)); callers finish reading (or copy) before any apply-phase mint. The `cap` overflow assert — sized by callers with `ExecutionParameters::MAX_SCOPE_DEPTH` (64, >6x observed nesting, 1 KB of stack per array) — is the Rule-19 tripwire: scope nesting has no structural bound (every `encodePush` deepens by one; `MAX_NAME_IDS` is uselessly large for a stack array), so an FTA-scale overflow stops at its origin instead of silently truncating a scope walk.

**How to spot violations.** A caller holding the returned spans across a `nameMap.encode` / `encodePush` / statement-mint into the same NameMap; a cap sized by anything other than the named constant; the assert moved AFTER the fill loop (it must precede any write to `out`); a re-sort by id instead of `compareSpans` ([I-84](#i-84)).

**How to fix on violation.** Copy the spans (or re-derive after the mint); restore the named-constant cap and the write-before assert.

**Code.** `memory.hpp` (`NameMap::strictAncestorSpans`); `parameters.hpp` (`ExecutionParameters::MAX_SCOPE_DEPTH`). Retained oracle: both `strictAncestorNames` overloads (their remaining production callers `applyEquivalenceClassToNegatedEquality` / `updateEquivalenceClasses` stay on the string form). Unit test: `test_memory.cpp::strict_ancestor_spans_matches_string_overload`. See also [I-3](#i-3), [I-84](#i-84), [I-104](#i-104).

---

<a id="i-141"></a>
## I-141 — the sanitize twins' shared scan core is heap-free: `bestSanitizePeer` blob walk in the historical visit order, stack `StrReplacement` staging, order-free substitution with an ascending observable drain


**Scope.** `bestSanitizePeer`, `addSanitizeSubstPair`, `sortSanitizeSubstPairs` (`memory.hpp`, free functions after the `Memory` class). Established by the S2 `_sanitize` subsession; the two end-of-burst sanitize twins (`sanitizeToBeProved` / `sanitizeHashMemory`, `prover.hpp`) wire onto this core.

**Rule.**

1. **Identical visit order to the retired heap snapshot.** `bestSanitizePeer` walks (scope, class, peer) triples in EXACTLY the sequence the twins' historical `decodeClassesAt` loops visited: scopes in caller order (own validity, then lex-sorted strict ancestors), classes in blob-run order (`peekRecordBytes` for `j` ascending == the decoded `recordsAt` vector order), members in the blob's stored run order (== the deserialized `memberIds` order). The running (bestPeer, bestPeerIsInt) state therefore takes the identical value sequence and lands on identical final bytes. Priority rule unchanged: `int_lev_*` outranks `it_*_lev_*`, lex-smallest (`compareSpans` == byte-lex `std::string::operator<`) within a tier, `repl_*`/plain names never rank.
2. **Scan-mint-freedom.** The scan path calls only non-minting probes (`NameMap::lookup`, `equivalenceClassesMap.lookup`), read-only peeks, and `decodeView`. `kindOf`'s memo write hits a DIFFERENT container than the NameMap byte pool ([I-3](#i-3) different-container rule). The returned peer span aliases NameMap cold bytes — valid until the caller's next NameMap mint; callers copy at staging.
3. **Substitution is order-free; the drain order is ascending and observable.** The `replaceKeysToString` / `replaceKeysScratch` position scan takes the unique greedy longest key match at each token boundary, so the output is a pure function of the (key → value) SET — pair-array order affects nothing (pinned by the reversed-order twin test). `sortSanitizeSubstPairs` still sorts ascending `compareSpans` because `sanitizeHashMemory`'s history block drains the pairs into `(=[k,v])` equality1 antecedents, and that sequence historically iterated a `std::map` ascending — an order that reaches `originInterner` mints and deload bytes.
4. **Duplicate staging is assert-pure.** `addSanitizeSubstPair` dedups on the key and ASSERTS the value matches on a duplicate — the scan is pure (a repeated argName recomputes the same bestPeer), and a divergent duplicate is a Rule-19 stop, not an overwrite.
5. **The twins' interiors are two-phase: a mint-free SCAN stages OWNED pending records, then a minting APPLY drains them.** `sanitizeToBeProved` snapshots the registry as packed keys on the gen-scratch page tier and index-sorts by decoded spans (byte-identical to the retired `decodeToBeProvedSorted` heap snapshot's `(original, validityName)` order — pairwise-distinct packed keys + interner injectivity make it tie-free); the scan phase performs NO NameMap mint (decodeView reads, non-minting lookups, `strictAncestorSpans`, `bestSanitizePeer`, scratch-only builders), so no held span can dangle; every pending record's payloads are OWNED — the rewritten original as string-tier `ScratchString` bytes (`replaceKeysScratch` product), the value run as an ascending gen-arena byte-bump copy of the sorted-unique CSR run (`runLen`/`valueAt` order == the retired `coldIntSetAt` set iteration), the packed key by value. The apply loop then re-keys in the historical per-record op order (assert-lookup → `eraseSet(old)` → `encode(new)` → collapse-gate `lookup` → `assignSetRange` from the run pointers). Two-registry split per [I-124](#i-124): strings on `scratchArenas`, spines + int runs + peek straddles on `genScratchArenas` — never mixed. `PendingTbpRec` carries a `ScratchString` by value inside a `PagedVector` — sanctioned because `ScratchString` is trivially copyable (the `PagedVector` static_assert enforces it) and the spine plus every record die before the function-exit `ScratchScope` rewind reclaims the bytes.
6. **The hashmem half mirrors point 5, with three of its own obligations.** `sanitizeHashMemory` snapshots `expandedImplications` as packed int64 keys and index-sorts by the decoded `lbStateInterner` spans — the same order as the retired decoded-EWV `std::sort` under `ExpressionWithValidity::operator<` (the decoded strings ARE the EWV fields; distinct packed pairs decode to distinct string pairs, so tie-free); its scan is mint-free with respect to nameMap AND lbStateInterner. (a) EVERY pending payload — old original, old validity, rewritten implication, each substitution pair — is an OWNED string-arena `ScratchString` copy (`PendingSanRec` + the shared `SanPairRec` pool), even though `lbStateInterner` is never minted during apply (`eradicateImplicationFromLB` only `lookup`s it): the owned-copy design does not RELY on that, per the approved worklist shape and [I-3](#i-3) discipline. (b) The levels probe stays at APPLY time, per record — earlier records' eradications erase `intStatementLevelsMap` entries, so a staging-time prefetch would resurrect deleted state; the copied run feeds the levels-RUN mail door (key bytes identical to the retired `coldIntSetAt` set form). (c) The history drain replays the record's `[pairStart, pairStart + pairCount)` pool window — the ascending key order `sortSanitizeSubstPairs` froze at staging == the retired `std::map` iteration, so the `(=[k,v])` `equality1` antecedent mints hit `originInterner` in the identical positional order. The eradication callee takes spans (`eradicateImplicationFromLB(Memory&, StrSpan, StrSpan)` — same probes and encodes on the same bytes in the same order).

**How to spot violations.** A re-sort of classes or members by id; a `decodeClassesAt` reintroduced inside the scan; the peer span held across a NameMap mint; the assert in `addSanitizeSubstPair` weakened to last-write-wins; `sortSanitizeSubstPairs` dropped ("the substitution doesn't need it" — the HISTORY drain does); a pending payload that aliases interner cold bytes instead of an owned copy; a mint added to the scan phase; the hashmem levels probe hoisted to staging time; the apply's per-record op sequence reordered "for clarity".

**How to fix on violation.** Restore the blob-run visit order and the ascending drain sort; copy spans before mints; keep pending payloads owned; keep the levels probe at apply time.

**Code.** `memory.hpp` (`bestSanitizePeer`, `addSanitizeSubstPair`, `sortSanitizeSubstPairs`); `prover.hpp` (`sanitizeToBeProved` — packed-key snapshot + owned `PendingTbpRec` staging; `sanitizeHashMemory` — packed-key snapshot + owned `PendingSanRec`/`SanPairRec` staging + apply-time levels probe; `eradicateImplicationFromLB` — the span-retyped callee). Retained oracles: `Memory::decodeClassesAt` / `decodeClassesById` (live for other consumers + the test oracle), `ce::replaceKeysInString`, `decodeToBeProvedSorted` and `coldIntSetAt` (both keep their other production consumers). Unit tests: `test_memory.cpp::best_sanitize_peer_matches_decoded_class_oracle`, `test_str_ops.cpp::sanitize_subst_pairs_dedup_and_sort_match_map`, `test_str_ops.cpp::replace_keys_pairs_match_map_oracle_sanitize_shapes`, `test_memory.cpp::tbp_snapshot_idx_sort_matches_decode_to_be_proved_sorted`, `test_memory.cpp::tbp_assign_set_range_run_matches_set`, `test_memory.cpp::expimpl_snapshot_idx_sort_matches_ewv_sort`, `test_memory.cpp::sanitize_levels_run_probe_matches_cold_int_set`. See also [I-3](#i-3), [I-45](#i-45), [I-46](#i-46), [I-84](#i-84), [I-88](#i-88), [I-124](#i-124), `I-140`.

---

<a id="i-142"></a>
## I-142 — `reactToHypo`'s per-row class read is an arena member-run snapshot taken before the class loop, never live blob views


**Scope.** `ExpressionAnalyzer::reactToHypo` (`prover.cpp`) — the once-per-burst phase-3 reaction to hypothetical disintegration — plus its parse foundation `parseHypoScopeVars` / `HypoScopeParse` (`prover.hpp`). Established by the S3 `_hypo` subsession of the transient-statification close-out.

**Rule.**

1. **The per-row class read is a SNAPSHOT, not a view.** Before a hypo row's class loop runs, the row's whole bucket is copied into arena pools (`memberPool` member ids + `classStarts` run starts, gen-scratch page tier, cleared per row); the loop reads ONLY the pools. Live `EquivalenceClassView`s held across the loop are forbidden: the loop's own `addExprToMemoryBlock` (the parent-scope `(=[var,var_copy])` deposit) can reach `updateEquivalenceClasses` and `assignRun` a DIFFERENT key of the SAME `equivalenceClassesMap`, splicing the map's shared blob pool — moving other runs' bytes. A live view would therefore both (a) see mid-row mutations the retired heap `decodeClassesById` vector never saw — a semantic divergence, and (b) dangle. The snapshot reproduces the heap decode's isolation exactly.
2. **Content-completeness.** The snapshot carries member runs ONLY, because the class loop reads nothing else (membership, the class-mate `kindOf` tier probe, and the id-compare skip) — no levels, no origins.
3. **Current-content read at row start.** The bucket is read (`lookup` + `runLen` + `peekRecordBytes`) at the START of each row, exactly where the retired code decoded it — an earlier row's mutations to a later row's bucket ARE seen, and keys minted into the map during the walk are invisible to the vid snapshot (both semantics inherited from the heap form).
4. **Row order is the decoded-name lex sort of a vid array** (`compareSpans(decodeView)` over `keyAt(1..count)`, gen-scratch byte-bump, mint-free sort window) — tie-free because vids are pairwise distinct and the NameMap injective ([I-84](#i-84), [I-3](#i-3)).
5. **The parse spans ride an owned per-row copy.** Gates + variable extraction run via `parseHypoScopeVars` on a per-row `ScratchString` copy of the scope name (string tier), so the parent-scope and variable spans survive the class loop's NameMap mints; the payload-variable array is insertion-order with linear `equalSpans` membership (the retired `std::set<std::string>`'s order was unobservable — membership + empty only), capped at `ExecutionParameters::MAX_ARITY` with Rule-19 tripwire asserts (the only hypo-name producer, `disintegrateExprHypothetically`, writes a deduplicated subset of one expression's arguments).

**How to spot violations.** An `EquivalenceClassView` (or `peekRecordBytes` pointer) from `reactToHypo`'s bucket held across an `addExprToMemoryBlock` call; the snapshot hoisted out of the row loop (breaks the current-content read, point 3); a levels/origins column added to the pools "for completeness" (dead weight — point 2); a raw-id row sort; parse spans taken on the raw `decodeView` name instead of the owned copy.

**How to fix on violation.** Restore the row-start pool copy and read only the pools inside the loop; keep the vid sort on decoded names; keep the parse on the owned `ScratchString` row name.

**Code.** `prover.cpp` (`reactToHypo`); `prover.hpp` (`parseHypoScopeVars`, `HypoScopeParse`); `memory_infra/str_ops.hpp` (`findSpanFrom`, `rfindSpanBefore`, `writeDecimalDigits` — the std-twin foundations). Retained oracle: `Memory::decodeClassesById` (live for other consumers + the test oracle). Unit tests: `test_memory.cpp::react_to_hypo_vid_sort_matches_string_pair_sort`, `test_memory.cpp::hypo_member_snapshot_matches_decode_classes_by_id`, `test_memory.cpp::parse_hypo_scope_vars_matches_string_oracle`, `test_str_ops.cpp::find_rfind_span_match_std_string_oracle`, `test_str_ops.cpp::write_decimal_digits_matches_to_string`; the firing-product path rides the pre-existing `removeUPrefixScratch` twin (`test_str_ops.cpp::disintegrate_scratch_twins`). See also [I-3](#i-3), [I-84](#i-84), [I-88](#i-88), [I-124](#i-124).

---

<a id="i-134"></a>
## I-134 — the token memo's cold blob is the single persistent representation; production writes it regex-free and reads it via `SpecialTokenScanView`; `std::regex` on this path is test-oracle-only


**Scope.** The `EqClassNameCaches` token memo path: `tokensViewOf` (decl `memory_infra/eq_class_name_caches.hpp`, def `memory.hpp`), `SpecialTokenScanView` (`memory.hpp`), the occurrence scanners `matchIntLevOccurrenceAt` / `matchItLevOccurrenceAt` / `scanIntLevOccurrences` / `scanItLevOccurrences` (`memory_infra/str_ops.hpp`), and the consumer funnel `filterIterations(id)` → `filterIterationsCore` (`prover.hpp`). Established by the S3 `_hypo` subsession.

**Rule.**

1. **The memo's PERSISTENT representation is untouched.** `tokensByExprId_` (one canonical `SpecialTokenScan` blob per expression id, `TypedColdBlobMap<int16_t, SpecialTokenScan>`) keeps its container, facet base 705+1..4, deload enrollment, `survivesDischarge`, and the [I-88](#i-88) clause-3 purity/reset discipline (pure function of `nameMap.decode(id)`, reset only with `destroyGrid`). Only the TRANSIENT interchange changed — no carve-out, no representation change.
2. **Production writes are regex-free and byte-identical to the retired codec path.** The `tokensViewOf` miss builds the canonical blob DIRECTLY on the caller's gen-scratch byte-bump tier via the occurrence scanners in two passes (count, then exact-length fill — asserted) and installs it through the raw `inner.assignRun` byte door (identity `int16_t` key). The scanners are byte-exact `std::sregex_iterator` twins (the four-step leftmost / unique-extent / resume-at-end / advance-by-one argument on their Doxygen), so the blob bytes equal `serializeSpecialTokenScan(scanSpecialTokens(text))`; same misses in the same caller order mint the same keys — deload facets 705+1..4 stay byte-identical.
3. **Production reads are a parameter-only zero-copy view.** `SpecialTokenScanView` (layout-coupled to `serializeSpecialTokenScan` — edited together) wraps the peeked record; it dies before the next `tokensByExprId_` mutation or the caller's arena pop ([I-116] discipline). `tokensViewOf` performs NO internal rewind — the caller (`filterIterations(id)`) owns the mark/pop window; that pop is LIFO-safe against `cleanUpExpressions`' own peeked class views (below the mark on the same slot arena).
4. **The consumer's probe order is deload-observable and FROZEN.** `filterIterationsCore`'s per-token `assert(kindOf(...))` lazily fills the deload-enrolled `kindById_` facet (705+0), so the token probe SEQUENCE — int section first, then it, early exit on the first forbidden token — is byte-observable. The view core's sequential `TokenCursor` walk replicates the retired vector loop's statement order exactly; any "visit all tokens then decide" restructuring is a deload-byte regression.
5. **`std::regex` on this path exists ONLY as unit-test oracles.** `scanSpecialTokens(const std::string&)` and `filterIterations(const std::string&, …)` are retained, production-caller-free oracle forms (the `classifyName(std::string)` precedent); the codec pair (`serializeSpecialTokenScan` / `deserializeSpecialTokenScan` / `Codec<SpecialTokenScan>`) and the `SpecialTokenScan` struct remain the codec/oracle interchange. The heap `tokensOf` is DELETED — replaced by its view twin (the `wipeSubtree(string)` → vid replaced-twin pattern).

**How to spot violations.** A `std::regex` reappearing on a production token-scan path; a `SpecialTokenScanView` stored in a struct or held across a memo mutation; a `tokensViewOf` internal `popTo` (would free the straddle/build bytes it returns); the `runLen == 1` assert weakened; a consumer restructured to pre-collect tokens before judging (breaks the kindOf fill order, point 4); the memo's facet base / enrollment / `survivesDischarge` "fixed" in passing.

**How to fix on violation.** Route the scan through the occurrence scanners; keep the view parameter-only and the arena window caller-owned; restore the cursor walk's statement order.

**Code.** `memory.hpp` (`SpecialTokenScanView`, `EqClassNameCaches::tokensViewOf` def, `scanSpecialTokens` — retained oracle, `serializeSpecialTokenScan` / `deserializeSpecialTokenScan` / `Codec<SpecialTokenScan>`); `memory_infra/eq_class_name_caches.hpp` (decl + memo container); `memory_infra/str_ops.hpp` (the four occurrence scanners; `isIntLevShape` / `isItLevShape` delegate to the cores); `prover.hpp` (`filterIterationsCore` view overload, `filterIterations(id)` flip; the string overload — retained oracle). Unit tests: `test_str_ops.cpp::int_it_lev_occurrence_scanners_match_regex_iterator_oracle` (the scanner == `sregex_iterator` sequence pin), `test_memory.cpp::special_token_scan_view_matches_deserialize_oracle`, `test_memory.cpp::filter_iterations_core_view_matches_heap_scan`, `test_memory.cpp::tokens_view_of_facets_match_typed_assign_oracle` (the facet 705+1..4 byte proof + the hit-path pin); the S2a anchored-scanner tests are the delegation's regression net. See also [I-3](#i-3), [I-83](#i-83), [I-84](#i-84), [I-88](#i-88), [I-116](#i-116), [I-124](#i-124), [I-125](#i-125).

---

<a id="i-135"></a>
## I-135 — staging records ride an intrusive single-type node chain on their task's `SealedPageSet`; append order is the iteration order; the chain dies with the payload


**Scope.** `SealedPageSet::appendRecord` / `recordCount` / `forEachRecord`, `SealedRecordNode`, `SealedRecordCursor` (`memory_infra/sealed_pages.hpp`; the chain-state zeroing in `SealedPageSet::freePages`, `memory_infra/sealed_pages.cpp`). Established by the S5 `_interfaces` subsession of the transient-statification close-out; the phase-2 producer/consumer wiring (`checkLocalEncodedMemoryStatic` → `applyFiringRecords`) consumes it.

**Rule.**

1. **The records live on the pages they already reference.** Each `appendRecord` places one intrusive `SealedRecordNode` header plus the memcpy-filled payload in a single page-set allocation (node aligned up first, payload to `alignof(T)` after it — the `SealedSpan::copyFrom` align-up idiom twice); the nodes link head → tail. There is no separate spine container and no per-record heap.
2. **Append order is the frozen iteration order.** `forEachRecord` walks head → tail; `SealedRecordCursor` hands out each record exactly once in the same order, resuming across later appends (appending never mutates existing nodes, only the old tail's `next`). Order-sensitive consumers build a sorted permutation over the payload addresses, never reorder the chain.
3. **Payload addresses are stable until `freePages`.** `appendRecord` returns a `const T&` to the stored payload; that address stays valid through `Filling` and `Sealed` — the pointer-permutation sort in `applyFiringRecords` relies on it.
4. **One record type per set.** The nodes are type-erased, so a `sizeof(T)` stride recorded at the first append is asserted on every later append/read (the realistic-mistake tripwire).
5. **The chain dies with the payload.** Chain nodes ride the same `bump_` arena as the strings/spans, so `freePages` reclaims spine + payload in one sweep — including records never read (the proveKernel redo-discard path drops an LB's records unread with zero extra code). Reads (`recordCount` / `forEachRecord` / `SealedRecordCursor::next`) are legal in `Filling` and `Sealed` only; all assert on `Freed`. The cursor's caught-up `nullptr` is a DEFINED result (the cache-miss shape), never a failure signal.

**How to spot violations.** A heap `std::vector<FiringRecord>` spine reappearing beside the chain; a consumer sorting the chain in place or depending on anything but append order + a sorted permutation; a chain read after `freePages`; a second record type appended to one set; a `std::function` parameter on the walk (the functor is a template parameter — zero-alloc).

**How to fix on violation.** Route records through `appendRecord` and read via `forEachRecord` / the cursor before `freePages`; keep per-type chains on separate sets.

**Code.** `memory_infra/sealed_pages.hpp` (`SealedRecordNode`, `SealedPageSet::appendRecord` / `recordCount` / `forEachRecord` / `recordPayload`, `SealedRecordCursor`); `memory_infra/sealed_pages.cpp` (`SealedPageSet::freePages` chain zeroing). Unit tests: `test_sealed_pages.cpp::record_chain_append_foreach_order_and_count`, `record_chain_interleaved_with_string_allocs`, `record_cursor_incremental_drain`, `record_chain_dies_with_free_pages`, `record_payload_address_stable`. See also [I-77](#i-77), [I-83](#i-83), `I-131`, [D-164](40_decisions.md#d-164).

---

<a id="i-136"></a>
## I-136 — levels travel the `addExprToMemoryBlock` kernel chain as caller-owned ascending-unique `(const int*, int32_t)` runs, never as pointers into a callee-mutable cold column


**Scope.** The level-set parameters of the `addExprToMemoryBlock` kernel web (`addExprToMemoryBlock`, `addEquality`, `addNegatedEquality`, `addToHashMemory`, `updateRejectedMap`, `applyEquivalenceClass`, `applyEquivalenceClassToNegatedEquality`, `mergeTwoEquivalenceClasses`, `updateEquivalenceClasses`, `ordisMerge`, `addStatement`) and every builder feeding them; the run foundations `coldIntRunAt` / `insertLevelSorted` (`memory.hpp`), the `RoutingColdMail::insertStatement` run sibling (`memory_infra/routing_cold_mail.hpp`), and the `RejectedMapValue` run constructor (`memory.hpp`). Established by the S5b `_levels` subsession of the transient-statification close-out.

**Rule.**

1. **The run form.** A levels parameter is a raw ascending-unique run `(const int* levels, int32_t levelCount)` — not a span struct. `(nullptr, 0)` is the empty set; the door assert shape is `assert(levelCount >= 0 && (levelCount == 0 || levels!= nullptr))`. An ascending duplicate-free run is exactly a `std::set<int>`'s iteration sequence, so every consumer (an `assignSetRange` splice, a linear merge, a codec/key `assign`, a count compare) sees identical bytes.
2. **Runs are CALLER-OWNED.** Every run passed into the web lives on memory the caller frame owns — a stack array (`coldIntRunAt` / `insertLevelSorted` fills, singleton/ascending-loop literals), a gen-arena fill made for this call, or an owned decode copy (`k.levels.data`). A run must NEVER alias `intStatementLevelsMap` or any cold container a callee can mutate: the chain re-enters (`addStatement` → `updateEquivalenceClasses` → `applyEquivalenceClassToNegatedEquality` → `addStatement`) and `addEquality` / `addStatement` / `applyEquivalenceClass` all `assignSetRange` into that same map — a CSR splice shifts later runs, and a `PagedVector` run may straddle pages besides. The stack-buffer copy reproduces exactly the retired per-frame `std::set<int>` copy semantics, so recursion re-entrance stays byte-safe with zero aliasing proof obligations.
3. **The sink set-overloads are retained.** `ColdMail::insertStatement(set)` / `RoutingColdMail::insertStatement(set)` / the `insertInternalStatement` EWV+set and span+set forms / the `RejectedMapValue` set constructor stay: they serve the heap-`Mail` wire boundary (the single-threaded commit/broadcast seams, I-51's subject) and are the byte oracles the run-door twin tests compare against.
4. **`coldIntSetAt` is the retained test oracle.** After the C3 straggler sweep it has zero callers on the phase call trees; its only production callers are two sanctioned uses inside `infra/hashburst_dump.cpp` (the Rule-14 sacred debug dump, which is off the phase call trees). The twin tests (`cold_int_run_at_matches_cold_int_set`, `sanitize_levels_run_probe_matches_cold_int_set`) keep it as the heap reference.
5. **The web is FLIPPED.** All eleven kernel-chain signatures carry the run form in place (no overloads — the all-or-nothing cascade, compiler-enforced completeness), and every builder feeding them converts by one of five idioms: (a) singleton stack literal `{ level }`; (b) empty `(nullptr, 0)`; (c) ascending `0..N` stack loop with a capacity assert; (d) cold-run stack copy via `coldIntRunAt` (the 256 literal per the in-tree `levArr[256]` precedent); (e) owned-vector pass-through (`k.levels.data` on a `decodeKey` row copy — frame-owned, ascending-unique by the mail-door contracts). `revisitRejectedIntegration2` deposits through the run twin of `emitIntegrationRevivalToInternalMailIn`; the string+set twin form is a retained test oracle with zero production callers.

**How to spot violations.** A `const std::set<int>&` levels parameter reappearing on a web signature; a call passing `&map.valueAt(...)`-derived or `mailLvlPool`-interior pointers into a door; a builder whose insert sequence is not provably ascending-unique feeding a run door; a widened capacity literal instead of a capacity assert.

**How to fix on violation.** Copy the source run to a caller stack buffer via `coldIntRunAt` (or build it with `insertLevelSorted`) and pass `(buf, n)`; keep the capacity asserts (Rule 19).

**Code.** `memory.hpp` (`coldIntRunAt`, `insertLevelSorted`, the `RejectedMapValue` run constructor, `coldIntSetAt` — retained oracle); `memory_infra/routing_cold_mail.hpp` (`insertStatement` run sibling); `memory_infra/cold_mail.hpp` (`insertStatement` run door — the S2 precedent this extends). Unit tests: `test_memory.cpp::cold_int_run_at_matches_cold_int_set`, `test_memory.cpp::insert_level_sorted_matches_set_insert`, `test_memory.cpp::rejected_value_run_ctor_matches_set_ctor`, `test_mail_log.cpp::statement_levels_run_door_twin` (suite `routing_cold_mail`). See also [I-51](#i-51), [I-84](#i-84), [I-103](#i-103), [I-118](#i-118), [I-124](#i-124).

---

<a id="i-137"></a>
## I-137 — the compiledMap complex is still on the heap — an OPEN violation to statify, NOT a sanctioned island; the only heap permitted in `performElem1/2/3` is `hashburst_trace.txt` generation. While it remains, phase-tree reads go through two read-out accessors over transparent (`std::less<>`) maps


**Scope.** `ExpressionAnalyzer::compiledExpressions` (`gl::CompiledExpressionMap` — `std::map<std::string, LogicalEntity, std::less<>>`) and `ExpressionAnalyzer::coreExpressionMap` (`ce::CoreExpressionMap` — `std::map<std::string, CoreExpressionConfig, std::less<>>`), plus the two reader accessors `ExpressionAnalyzer::compiledEntity(StrSpan)` / `coreConfig(StrSpan)`. Established by the S6 `_reader` subsession (the FINAL statification subsession) of the MPU-0.1 transient-statification close-out. This is an OPEN heap violation, NOT a user-approved carve-out: the compiled-definition layer is still a heap `std::map` and must be statified. The rule (user-set, 2026-07-04): in `performElem1/2/3` the ONLY permitted heap is `hashburst_trace.txt` generation (the Rule-14 debug dump, off the compute path); everything else on the phase trees must be static, this layer included. It is not yet — on silicon this layer is ROM/SRAM.

**Rule.**

1. **The open violation.** The compiledMap complex is still on the heap after the campaign — config-load populated, never deloaded, never minted into or serialized — but that is an unfinished statification target, NOT a sanctioned exception. The remaining string `LogicalEntity` / `Instruction` disintegration working forms (the `ce::` calls, the `EncodedExpression` parse boundary) are the same open gap. It must go static.
2. **Read side is fenced by the two accessors.** Every phase-tree point read goes through `compiledEntity` / `coreConfig`: a span converts to a `std::string_view` and probes the transparent map heterogeneously — no `std::string` is built to key the lookup. `nullptr` is the DEFINED miss, the exact analogue of `find == end` (Rule 19 bivalent contract — at every converted site the miss branch is a defined result: an existing `if (it == end)` branch or an existing `assert`, preserved verbatim). Read-out ONLY: a transparent `find` and a pointer return — no transformation, caching, side effect, or logging (adding any is a Rule-8 conversation). The empty span maps to the same defined miss `find("")` produced.
3. **Byte-order contract.** `std::less<>` compares two stored `std::string` keys via `std::string::operator<` — byte-lexicographic, the SAME strict weak order the default `std::less<std::string>` produced. Tree shape, insertion results, and every whole-map iteration order are byte-identical to the pre-switch map. A heterogeneous probe compares a stored `std::string` against a `std::string_view` via `std::string_view::operator<` — also byte-lex — so a span probe returns exactly the entry a materialized-`std::string` probe returned.
4. **Whole-map recursive consumers keep `const map&` parameters** — retyped mechanically to `const gl::CompiledExpressionMap&` / `const ce::CoreExpressionMap&` (the comparator switch makes them a distinct type; the compiler enforces closure). Zero logic change at any consumer. The Rule-14 sacred `infra/hashburst_dump.{hpp,cpp}` `dumpEntry` / `writeCompiledExpressions` signatures were retyped to `const CompiledExpressionMap&` under explicit user consent (2026-07-03) — type-spelling-only, nothing else in those files touched.
5. **Writers are NOT fenced.** Every existing write site keeps direct member access on its single-threaded seam: the compile/init seams (`compileCoreExpressionMapCore`, `excludeRepetitions` / `splitNormalizedKey`, `findDefinitionSetRecursive` / `extendCoreExpressionMap` / `compileCoreExpressionMap`), the proveKernel deferred-compaction drain, `constructOrTheorem` via `run_modes`, and `loadGlBinary`. The returned entry pointer aliases the map node — stable for the program's life on the read side (node-based `std::map`); phase-parallel sweeps are read-only per the carve-out, and no converted site holds the pointer across a writer seam (each consumes it synchronously).

6. **`expressionsFromConfig` is NOT the coreConfig fence.** `prepareIntegrationCore2`'s two core-expression MEMBERSHIP probes read the SEPARATE `ExpressionAnalyzer::expressionsFromConfig` set — a snapshot of the `coreExpressionMap` keys taken at construction, BEFORE `extendCoreExpressionMap` / the OR-core insert add further keys. So `coreConfig(x)!= nullptr` matches a strict SUPERSET of `expressionsFromConfig.count(x)`; routing these two probes through the `coreConfig` fence would silently WIDEN membership. The 0%-heap conversion instead retyped `expressionsFromConfig` to a transparent `std::set<std::string, std::less<>>` and probes it with a `std::string_view` — semantics-identical (the set's contents are unchanged), zero-allocation. STOP-and-report if a transparent-`find` verdict ever disagrees with the materialized-`std::string` form.

**How to spot violations.** A materialized `std::string core = ce::extractExpression(...)` feeding a `compiledExpressions.find` / `coreExpressionMap.find` on a phase-tree read path; a new `if (map.contains(...))` proof branch (Rule 16 — the map is never a control-flow input); caching or normalization added inside an accessor; a reader pointer stored across a nested call that can reach a writer seam.

**How to fix on violation.** Route the read through `compiledEntity(extractExpressionSpan(...))` / `coreConfig(...)`, preserving the site's own miss verdict (assert stays assert, branch stays branch). Never widen an accessor beyond `find` + pointer return.

**Code.** `prover.hpp` (`ExpressionAnalyzer::compiledEntity`, `ExpressionAnalyzer::coreConfig`); `memory.hpp` (`gl::CompiledExpressionMap`); `compiler.hpp` (`ce::CoreExpressionMap`). Unit tests: `test_memory.cpp::compiled_entity_reader_matches_map_at`, `test_memory.cpp::core_config_reader_matches_map_at`. See also [I-84](#i-84), [I-124](#i-124), `I-138`, the project conventions.

**Status.** C1 landed the aliases + accessors + the mechanical consumer sweep. C2 routed the memory.cpp phase-tree reads: the `checkLocalEncodedMemoryStatic` hot marker-capture block (the phase-2 parallel staging probe now feeds the existing `(StrSpan, StrSpan)` `lookupTemplateKey` overload; the marker category read goes through `compiledEntity`) and the `makeNormalizedKeysForAdmission` admission-key probe (through `coreConfig`) — the per-row `rplStd` / `validityStd` / `coreExpr` materializations are gone. C3 routed the prover.cpp fan-out (expandExpr, multiplyImplication, findImmutableArgs, tryBackReformulateOperatorHead, reformulateTheorem ×2, checkNecessityForEquality, addExprToMemoryBlock, addTheoremToMemory, updateAdmissionMapRecursion ×3, makeAdmissionKeys, prefillIntegrationMapsRecursive ×2, disintegrateExprHypothetically) — the `core` / `kCore` / `newCore` / `coreExpr` locals whose only uses were `operators` + the map probe are deleted (`operators` is transparent). `allowedForMail`'s map read stays deferred to C5; the proveKernel deferred-compaction drain stays a direct WRITER-seam read by decision. C4 routed the prover.hpp fan-out (updateAdmissionMap, elementMatchesLocalUCriterion ×2, outputMatchesHeadOutputSlot, isAllowedAsOperatorInput, isAdmittedIntegration, cleanAdmissionMap, prepareIntegrationCore, prepareIntegration) — the `cleanAdmissionMap` `coreStr` S6-carve-out probe is retired, `isAdmittedIntegration`'s presence-only `it0` folds into the assert, and `prepareIntegration`'s former unchecked `find->second` deref becomes a loud `compiledEntity` + `assert` (latent UB removed; the dead `//return;` category branch deleted under user standing dead-code authorization). `findDigitArgs` keeps its `const ce::CoreExpressionMap&` parameter reads (a C1 whole-map consumer, not a member point read); the compile/init seams (`compileCoreExpressionMapCore`, `extendCoreExpressionMap`, `findDefinitionSetRecursive`, `compileCoreExpressionMap`) stay direct WRITER seams. C5 completed the deferred `allowedForMail` read: the function is now span-native end to end (its regex gate + collection fold into `str_ops::scanSingleDistinctIntLev`, the marker form builds on the string-scratch arena, and the `compiledExpressions` read goes through `compiledEntity`); its memory.cpp caller decodes via `decodeView`. C6 dropped the last production `std::regex` in `prover.*` (`extractRemainingArgs` two independent lexical passes; `extractSubstringsForAuxy` paren-group scan). C7 (the S6d value-object arm) retired the non-drain admission/rejected/rejected-integration caller VALUE OBJECTS: two id-run blob doors `insertAdmissionIdsBlob` / `insertRejectedIdsBlob` deposit the `Codec` byte layout DIRECTLY from pre-minted id runs (no `AdmissionMapValue` / `RejectedMapValue` built), and the manager-added `updateRejectedMapIntegration` conversion landed a full mirror blob-splice path (`RejectedIntegrationValueBlobView`, `rejectedIntegrationBlobLess`, `serializeRejectedIntegrationValueToArena`, `insertRejectedIntegrationBlobSorted`, `insertRejectedIntegrationIdsBlob`) — the value-form `insertRejectedIntegrationValue` stays as the byte oracle. The four ctor-argument mint sites (`updateAdmissionMap`, `updateRejectedMap`, `updateRejectedMapIntegration`, and the already-sequenced `updateAdmissionMapRecursion`) hoist their `vi.encode` calls into explicit sequenced statements in MSVC's observed right-to-left ctor-argument evaluation order; the order is compiler-verified by pinning tests (`admission_value_ctor_mint_order_frozen`, `rejected_value_ctor_mint_order_frozen`, `rejected_integration_value_ctor_mint_order_frozen`) that assert the OLD ctor and the NEW sequenced build produce byte-equal value-interner id tables. The `applyEquivalenceClassToAdmissionMap` drain rebuild (no mint) copies the paged id pools into contiguous arena runs and deposits through `insertAdmissionIdsBlob`. The compiled-definition island + its `compiledEntity`/`coreConfig` fence are unaffected by the continuing phase-tree statification (S8 `_firing_check` firing check + residuals + twin conversions; S9/S10 the admission gates + integration templates) — see the `I-138` S7-census extension note.

---

<a id="i-149"></a>
## I-149 — the origin doors mint dependency keys into a stack run and serialize from POD fields; the `IdOrigin` / `IntMailOrigin` heap intermediate is eliminated at the door layer, key-first


**Scope.** The batch-6 origin-record cascade in `memory.hpp`: the two dep-mint helpers `mintOriginDepsInto` (span antecedents) / `mintOriginDepsFromEWVInto` (EWV antecedents); the POD `(uint8_t tag, const int64_t* deps, int32_t depN)` overloads of `serializeOriginTo` / `serializeMailOriginTo` / the cold+heap `addOriginId` / `addMailOriginRecord`; and the door bodies that now route through them — `addOriginEncoded` (heap + cold, span-antecedent AND EWV/`OriginLine`-antecedent), `addInternalMailOrigin`, `addRoutingMailOrigin`, `MergeClassAccum::addOriginEnc`. Plus `prover.cpp · addExprToMemoryBlock`'s inline of `decodeOrigin`.

**Rule.**

1. **No `IdOrigin` / `IntMailOrigin` heap vector at the door layer.** A door minting a history line mints the dependency keys straight into a caller `int64_t[kMaxOriginDeps]` stack buffer via `mintOriginDepsInto` / `mintOriginDepsFromEWVInto` (positional order — byte-identical id run to `encodeOriginSpans` / `encodeOrigin`), then serializes / RMWs from the raw `(tag, deps, depN)` POD fields. No transient `IdOrigin` (the D-49 origin `std::vector<int64_t>`) or `IntMailOrigin` is materialized. The dependency-count assert names `kMaxOriginDeps` (Rule-19 tripwire; realistic counts are single-digit).

2. **Key-first, everywhere.** Every door mints the origin KEY first, then the dependency ids. The mail / `MergeClassAccum` doors were already key-first (key minted before deps inside `encodeOrigin(Spans)`), so their `originInterner` mint sequence is unchanged. The single-expression `addOriginEncoded` (heap + cold) doors' old body was `addOriginId(map, mintOriginKey(...), encode…(...), cap)` whose key-vs-deps evaluation order was MSVC-UNSPECIFIED; the explicit key-first statements pin it deterministically under the USER byte-identity waiver — `files/` proof artifacts are id-value-independent (the chapter export decodes `originInterner` ids to strings), so only the gitignored `.deload/` id VALUES may reassign; the gate is a 2×-sequential `main.py` check, not `git status files/`.

3. **`originTagFromString` reads a separate static table**, so the tag resolution never affects the `originInterner` mint sequence; the EWV-form doors resolve it via the SPAN twin `originTagFromString(StrSpan(origin.first))`, dropping the heap `unordered_map` lookup.

4. **Retained oracles.** `encodeOrigin` / `encodeOriginSpans` / `decodeOrigin` (and the `IdOrigin` / `IntMailOrigin` overloads of the serializers and doors) leave the `standardProcessing` tree but are KEPT in `memory.hpp` as the byte-twin oracles (the `decodeValueVector` precedent). `decodeOrigin`'s one in-tree caller (`addExprToMemoryBlock`) inlines its antecedent walk over owned decoded strings (the tag path was already the enum `frontId.first`).

**How to spot violations.** A door body constructing `IdOrigin` / `IntMailOrigin` before a serialize/RMW; a `mintOriginKey` evaluated as an unsequenced argument beside an `encode…` call (unspecified order); a new production caller of `encodeOrigin` / `encodeOriginSpans` (they are oracle-only).

**Code.** `memory.hpp` (`mintOriginDepsInto`, `mintOriginDepsFromEWVInto`, the POD `serializeOriginTo` / `serializeMailOriginTo` / `addOriginId` / `addMailOriginRecord` overloads, the rewritten door bodies); `prover.cpp` (`addExprToMemoryBlock` `decodeOrigin` inline). Unit tests: `test_memory.cpp` `serialize_origin_to_pod_matches_idorigin`, `serialize_mail_origin_to_pod_matches_intmailorigin`, `add_origin_id_cold_pod_matches_idorigin`, `add_mail_origin_record_pod_matches_intmailorigin`, `add_origin_id_heap_pod_matches_idorigin`, `mint_origin_deps_into_matches_encode_origin_spans`, `mint_origin_deps_from_ewv_into_matches_encode_origin`. See also `I-138`, [I-84](#i-84), the project conventions.

---

<a id="i-150"></a>
## I-150 — `generateSetPartitions` is a heap-free arena CSR of element indices; the recursive Bell enumeration is an iterative ping-pong reproducing the heap emission order byte-for-byte


**Scope.** `ExpressionAnalyzer::generateSetPartitions` (`prover.cpp`, declared `prover.hpp`) and its sole consumer `multiplyImplication`.

**Rule.** The former heap `std::vector<std::vector<std::vector<std::string>>>` partition family is three caller-owned `PagedVector<int32_t>` CSR columns on the per-slot gen-scratch page tier: `outMember` (concatenated class members as INDICES into the caller-stable `elements`), `outClassStart` (per-class start + terminator), `outPartStart` (per-partition start + terminator). Both the partition ENUMERATION order and the within-class member order are OBSERVABLE downstream (`multiplyImplication`'s `seen`/`result` dedup and representative selection), so both are byte-identical to the heap builder: the `n > cap` regime is the flat singletons/pairwise/triples port; the `1 ≤ n ≤ cap` recursive Bell enumeration is an iterative ping-pong over two arena CSR buffers — the heap recursion peels `elements[0]` and recurses on the rest, so the outermost peel is applied LAST, and expanding indices `n-1.. 0` reproduces the Option-1-then-Option-2 emission order; within-class order is the `compareSpans(elements[idx])` byte-lex sort (== the heap `std::sort` on member strings). the span-form `multiplyImplication` maps each member index → `sortedVars[idx]` (a `StrSpan` run) and runs its double-u guard / representative pick / `replaceArgSpanScratch` (the `replaceArgInString` twin). The heap builder is kept TEST-LOCAL as the byte-twin oracle (ce-twin survivorship — prover-only, no out-of-tree reader). **Batch 7:** an additive `generateSetPartitions(const StrSpan* elements, int32_t n, …)` overload is the REAL implementation (its interior reads each element only through `compareSpans` + the count, so the retype is mechanical); the `std::vector<std::string>` overload builds a `StrSpan` run over its strings on the arena byte-bump tier and FORWARDS to it, byte-identical. `multiplyImplication` calls the span form directly in its span-native rewrite.

**Code.** `prover.cpp` / `prover.hpp` (`ExpressionAnalyzer::generateSetPartitions`, `multiplyImplication`). Unit test: `test_memory.cpp · generate_set_partitions_arena_matches_heap` (empty / 1 / 2 / cap / cap+2 / `u_`-member battery). See also `I-138`, [I-84](#i-84), [I-124](#i-124).

---

<a id="i-145"></a>
## I-145 — the mail-merge class write-back serializes the merged `EquivalenceClass` through a 0%-heap arena-fill twin; the I-103 canonical byte layout does not move


**Scope.** `serializeEquivalenceClassInto` (`memory.hpp`) and its sole in-tree reacher, the mail-merge class write-back inside `standardProcessing` (`prover.hpp`, the `mcls` splice).

**Rule.** The write-back serializes the merged class through `serializeEquivalenceClassInto(ScratchArena&, const EquivalenceClass&) -> StrSpan`, a 0%-heap arena-fill twin of the retained heap codec `serializeEquivalenceClass`. Output is byte-for-byte identical to the heap codec — the I-103 canonical-bytes contract (the single determinism point reaching the `.deload/` stream): same field order, `memberIds` verbatim (decoded-lex, I-84), `intEqualityLevelsMap` ascending `std::map` walk with each `std::set<int>` ascending, `equalityOriginMap` keys sorted ascending via an arena `std::sort(ok, ok + G)` (identical to the heap `std::vector<int64_t>` sort — an `unordered_map` whose bucket order must not leak), per-key history lines + per-line deps verbatim (insertion order, observable). The ONLY interior changes are the sink (`std::vector<char>` return → a caller-arena byte-bump buffer) and the origin-key sort scratch (`std::vector<int64_t>` → arena `int64_t*`); neither moves a byte value or an emit order. Both `ok` and the result `buf` ride the caller's `ogArena` byte-bump tier; blocks are stable (I-107) so the returned `StrSpan` survives the later run-splice allocs + `peekRecordBytes` peeks; there is NO nested `ScratchScope` (a rewind would free `buf`). The heap codec `serializeEquivalenceClass` is KEPT as the byte-twin oracle plus its many `test_memory.cpp` uses; it and `deserializeEquivalenceClass` leave the `standardProcessing` tree via the twin.

**Code.** `memory.hpp · serializeEquivalenceClassInto`; site `prover.hpp` (the `mcls` mail-merge class write-back). Unit test: `test_memory.cpp · serialize_equivalence_class_into_matches_heap` (empty / members-only / members+levels / full ascending-insert / full reverse-insert exercising the `ok` sort). See also [I-84](#i-84), [I-124](#i-124), `I-103`, `I-107`, `I-138`.

---

<a id="i-148"></a>
## I-148 — the rejected + rejected-integration read snapshots are arena blob-copy siblings of `snapshotAdmissionRun`; `revisitRejectedIntegration2` snapshots-then-erases-then-processes


**Scope.** `snapshotRejectedRun` / `snapshotRejectedIntegrationRun` (plus the `RejectedRunSnapshot` / `RejectedIntegrationRunSnapshot` structs) in `memory.hpp`, and the migrated `revisitRejectedIntegration2` (`prover.cpp`).

**Rule.** `snapshotRejectedRun(m, pk, gArena)` and `snapshotRejectedIntegrationRun(m, pk, gArena)` are the literal rejected-side siblings of `snapshotAdmissionRun`: look up the key, peek each record blob, copy the bytes VERBATIM onto the caller's gen-scratch arena, and return an arena-held array of zero-decode `RejectedValueBlobView` / `RejectedIntegrationValueBlobView` over the copies. The copies survive any later `eraseBlobIf` restructure of @c m (09b pitfall 6, `I-99`), so a snapshot-then-erase-then-process consumer never dangles. Because the cold run is stored canonical under `DecodedRejected[Integration]ValueLess` (the `insertRejected[Integration]Value` RMW contract), the view sequence == the former `rejectedRecordsAt` / `rejectedIntegrationRecordsAt` set order byte-for-byte; the set's dedup is a no-op on a canonical run. Mints nothing; `count == 0` is a defined never-minted-key miss (Rule 19). `revisitRejectedIntegration2` adopts the `revisitRejected2` shape (snapshot → erase → process → `popTo`): snapshot the cohort, erase `markedPk` from `rmi`, then walk the views reading `concreteConstituentId` / `siblingCount` / `siblingId(i)` / `compoundExpressionId` in place of the former heap-set members — behaviour-identical because the emission loop deposits onto `sameIterationInternalMail` and never reads `rmi`, so the erase order relative to the loop is unobservable. Removing the only heap in `revisitRejectedIntegration2` (the `RejectedIntegrationValueSet`) flips its inventory row `true`. The heap `rejected*RecordsAt` stay as the retained oracles for the sacred hashburst dump + the twin tests; the two hooks (`applyEquivalenceClassToRejectedMap[Integration]`) migrate onto the snapshots in a later commit.

**Code.** `memory.hpp · snapshotRejectedRun` / `snapshotRejectedIntegrationRun`; `prover.cpp · revisitRejectedIntegration2`. Unit tests: `test_memory.cpp · snapshot_rejected_run_matches_set` / `snapshot_rejected_integration_run_matches_set`. See also `I-99`, [I-84](#i-84), the `snapshotAdmissionRun` precedent (09b pitfall 6).

---

<a id="i-151"></a>
## I-151 — multiplyImplication has a 0%-heap span form emitting CopyRefs; the vector<string> form is the retained heap oracle; the file-static replaceArgInString is deleted


**Scope.** `ExpressionAnalyzer::multiplyImplication` (`prover.cpp`, both overloads), its sole in-tree caller `addToHashMemory` (`memory.cpp`), and the deleted file-static `replaceArgInString`.

**Rule.** The production path is `void multiplyImplication(StrSpan implication, ScratchArena& outStrArena, PagedVector<CopyRef>& out)`: it collects the unique vars + non-Anchor exprs as `(offset,len)` `VarRef`s into the caller-stable `implication` (no bytes copied, row 62), dedups the vars in a gen-scratch `ColdHashSet`, sorts the `(1)`-typed vars by `numericLessSpan` (the `numericLess` twin) into a contiguous `StrSpan` run, feeds the `generateSetPartitions` `StrSpan`-run overload, and per partition builds the substituted copy on `outStrArena`'s string tier via iterated `replaceArgSpanScratch` (the `replaceArgInString` twin) + `deduplicateBoundVarsScratch`, skips trivial-equality heads, content-dedups through a gen-scratch `ColdHashSet`, and emits each survivor as a `CopyRef` (dropped partitions reclaim their per-partition scratch via `popTo`). The `(1)`-typed slot probe replaces `std::to_string(i+1)` + `definitionSets.find` with a `writeDecimalDigits` stack buffer + a span-compare scan of the non-transparent `definitionSets` map (byte-identical). The emitted copy SEQUENCE (bytes + order) is byte-identical to the retained heap oracle `std::vector<std::string> multiplyImplication(const std::string&)`, which stays compiled as a self-contained oracle (a local word-boundary lambda replaces the deleted `replaceArgInString`) but has no `standardProcessing`-reachable caller, so it leaves the inventory (the `admissionRecordsAt` way). `addToHashMemory` (the still-heap boundary owner) drives the copies through a `PagedVector<CopyRef>` on gen-scratch + a per-copy `curOrigImpl` `std::string` materialized from each `CopyRef` span, held live by a `ScratchScope` across the loop; its outer loop and the `c == 0` key/value special case are unchanged. The file-static `replaceArgInString` is DELETED (its span twin `replaceArgSpanScratch` + a test-local oracle carry it, row 311 completes).

**Code.** `prover.cpp · multiplyImplication` (span form + retained heap oracle); `memory.cpp · addToHashMemory`. Unit tests: `test_memory.cpp · multiply_implication_span_matches_heap` (differential vs the heap oracle); the primitive twins `replace_arg_span_scratch_matches_heap` / `numeric_less_span_matches_numeric_less` (`test_str_ops.cpp`) and `generate_set_partitions_arena_matches_heap`. See also [I-84](#i-84), [I-124](#i-124), I-150, `I-138`.

---

<a id="i-144"></a>
## I-144 — makeNormalizedKeysForAdmission is 0% heap; the encodedMap / owner / remaining-args writes go through raw-key + id-run doors


**Scope.** `makeNormalizedKeysForAdmission` (`memory.cpp`) and the four write doors it wires: `encodeNormKeyInto` / `appendLmvIdsRecord` (`memory.hpp`), `mergeOwnerRecord` raw-key overload / `insertRemainingArgsNormKey` raw-run overload (`prover.hpp`).

**Rule.** The install-time admission-key builder is 0% heap: every owning container in its frame becomes a stack/span form. `outputArg` is a `StrSpan` slice of `key[index]`; `binary` / `subkey` / `sids` / `tempList` are stack arrays (cap `MAX_ADMISSION_KEY_ELEMENTS + 1`, loud Rule-19 asserts); `replaced` / `valueVariant` / the per-`replKey` element are `ScratchString`s on the string-tier arena (held under an index-level `ScratchScope` + a per-permutation nested scope); the `mp2` rename map is a `StrReplacement` run built in id order (`replaceKeysScratch` is greedy-longest, so order is NOT observable — byte-identical to the former lex-ordered `std::map`), its key spans `NameMap::decodeView` (I-3-safe: no NameMap mint falls between the decode and the last `replaceKeysScratch` use — the intervening mints are `ruleInterner`). The `remainingArgs` / `intRemainingArgs` sets are dropped: the `getRemainingArgs` `compareSpans` sorted-unique run is minted into NameMap in RUN order (I-84 — mint order reaches deload), and a SORTED-ASCENDING copy is the `Int16SetKey` (the former `std::set<int16_t>` iteration order); `lmv.remainingArgIds` mints `ruleInterner` in run order (== the former set-lex order). The three cold writes go through the raw / id-run doors — `appendLmvIdsRecord` (the `Codec<LocalMemoryValue>` blob serialized straight from id runs with the marker-install defaults), `mergeOwnerRecord`'s raw-key overload (`encodeNormKeyInto` stack key bytes + `inner` doors), `insertRemainingArgsNormKey`'s raw-run overload — so no owning `NormKey` or `LocalMemoryValue` is materialized. Each door is byte-identical to its retained owning-form oracle (`encode_norm_key_into_matches_codec`, `merge_owner_record_raw_matches_owning`, `append_lmv_ids_record_matches_value`, `insert_remaining_args_normkey_raw_matches_owning`). `outputMatchesHeadOutputSlot` and `getRemainingArgs` gained additive `StrSpan` overloads (the `std::vector<std::string>` forms retained). Every mint sequence (NameMap: subkey encodes, then `intRemArgs`; `ruleInterner`: valueVariant, keyIds in binary order, remainingArgIds in run order, originalImpl) is preserved verbatim — byte-identical deload, no waiver.

**Code.** `memory.cpp · makeNormalizedKeysForAdmission`; `memory.hpp · encodeNormKeyInto` / `appendLmvIdsRecord`; `prover.hpp · mergeOwnerRecord` (raw overload) / `insertRemainingArgsNormKey` (raw overload) / `getRemainingArgs` (span overload) / `outputMatchesHeadOutputSlot` (span overload). Unit tests: the four door twins (`test_memory.cpp`). See also [I-84](#i-84), [I-124](#i-124), `I-3`, `I-138`.

---

<a id="i-143"></a>
## I-143 — the raw-key `mergeOwnerRecord` is fully blob-native: it reads the existing `OwnerSet` blob and writes a byte-identical merged blob, materializing NEITHER an `OwnerSet` NOR a `serialize` vector


**Scope.** `ExpressionAnalyzer::mergeOwnerRecord` (the raw-key overload, `prover.hpp`), `buildUSignatureRunInto` + the `OwnerSetBlob` uSignature read accessors (`memory.hpp`). The owner maps are `TypedColdBlobMap<NormKey, OwnerSet>`, run-length-1 (one whole-value blob per key).

**Rule.** Merging one owner into a `normalizedEncoded*` map is a whole-blob RMW that touches NEITHER the `OwnerSet` struct NOR the `Codec<OwnerSet>` blob bytes. The owner's u_ signature is produced by `buildUSignatureRunInto` (the slot-ascending `(slot, argFullId)` run + `hasUArg` flag, the byte-exact producer twin of the `IntEncodedExpr` `recordUSignature`); the existing owner blob is read zero-copy through `OwnerSetBlob` (`peekRecordBytes` arena straddle overload); the merged blob is assembled directly on the gen byte-bump tier (reached internally via `g_currentCoreId`, per-registry `slotCount-1` fallback). Byte 0 is `existingLoose || !hasUArg` (the decode-then-`recordUSignature` loose path); `partitionIds` is the existing ascending run with `partitionId` inserted sorted-unique (`std::set<int32_t>` order); `uSignatures` is the existing lex-ascending run with the new sig inserted at its `std::vector<std::pair>::operator<` position (element-wise, `.first` then `.second`, shorter-is-prefix — NOT any count-prefixed order), skipped when byte-equal (`std::set` idempotent) or when the owner is loose. Result byte-identical to `Codec<OwnerSet>::serialize` of the decode-merge-reencode oracle (the owning overloads + `Codec<OwnerSet>::serialize`, retained as the differential twin). The merged blob is assembled in a DISJOINT buffer BEFORE the run-replacing `assignRun`, so the peeked existing bytes never dangle (09b pitfall 6). The `OwnerSetBlob` read accessors deref `p`, so the sig pre-scan is guarded by `haveExisting` (a fresh key has a null `ob.p`).

**Spot.** A revived `OwnerSet os;` local or a `Codec<OwnerSet>::serialize(...)` call inside the raw overload; a signature comparator that compares the pair-count field first (the `Int16SetKey` byte order, not `std::vector<std::pair>::operator<`); an unguarded `ob.firstSigOffset` / `ob.partitionId(i)` on a fresh key (null-deref).

**Fix.** Keep the merge byte-native; if any clean path appears to need a change to `OwnerSet`'s representation or the serialized bytes, STOP and raise with the user (Rule 8) — do not choose it silently.

**Code.** `prover.hpp · mergeOwnerRecord` (raw overload); `memory.hpp · buildUSignatureRunInto`, `OwnerSetBlob` (`uSigCount` / `firstSigOffset` / `sigPairCount` / `sigBytes` / `sigPairFirstAt` / `sigPairSecondAt`). Unit tests: `build_usignature_run_matches_record_usignature`, `merge_owner_record_raw_blob_matches_owning` (`test_memory.cpp`). See also [I-83](#i-83), [I-84](#i-84), `I-99`, `I-100`, `I-138`.

---

<a id="i-147"></a>
## I-147 — proven direct theorems cross the parallel → join boundary as `SealedString` records on a shared `SealedPageSet`, drained in the former `std::sort` order


**Scope.** `ExpressionAnalyzer::updateGlobalDirectPages` (the `std::optional<SealedPageSet>` member) + the `UpdateGlobalDirectRec` POD + `updateGlobalDirectLess` + `drainUpdateGlobalDirect` (`prover.hpp` / `prover.cpp`); the two producer sites `dischargeToBeProved` / `dischargeContradiction` and the `proveKernel` emplace/drain. Replaces the former `std::vector<std::tuple<std::string,int>> updateGlobalDirectTuples`.

**Rule.** A proven direct (non-auxy) theorem discovered in a parallel worker must NOT be deferred as a decode — the LB deloads after its burst, so the NameMap-decoded theorem bytes cannot be recovered post-join. The bytes are sealed VERBATIM onto the shared `updateGlobalDirectPages` set (`SealedString::copyFrom`) at the producer, under `updateGlobalDirectMutex` (the set's `alloc` / `appendRecord` are not internally thread-safe — the `deferredAncestorPages` precedent, [D-187](40_decisions.md#d-187)). `proveKernel` `emplace`s a fresh Filling set before phase-1; `drainUpdateGlobalDirect` (single-threaded, post-`pool.join`) gathers the record chain, index-sorts on `updateGlobalDirectLess` (the `(theorem bytes via compareSpans, coreId)` total order reproducing the former `std::sort` over `std::tuple<std::string,int>` EXACTLY), replays each into `updateGlobalDirect` (materializing the `std::string` at that still-heap sink edge), then `seal` / `freePages` / `reset`. Byte-identical to the former sort-then-loop: the replay order is the same, so the global proof state and `theorems.txt` do not move. The sealed lifecycle is `proveKernel`-owned (emplace) + drain-owned (reset) — `releaseCEBatchMemory` does NOT touch it (nullopt between `proveKernel` calls). Part A of the same conversion made `dischargeToBeProved` 0% heap: `addExpression` / `effectiveValidity` are `ScratchString::copyFrom` on the string arena (mandatory — `encodePush` / `encode` mint NameMap mid-entry, I-3), `orEmitScope` is a `rfindSpanBefore` prefix slice or the `"main"` literal, and the theorem rides `sArena` (reconstruct) or a NameMap `decodeView` (contradiction, no mint before the sealed copy).

**Spot.** A revived `updateGlobalDirectTuples` vector; a worker minting a process interner to build the theorem (I-127 — the sealed-bytes route deliberately avoids interning); a `SealedPageSet` append outside `updateGlobalDirectMutex`; a drain comparator on raw ids instead of decoded bytes; `releaseCEBatchMemory` resetting the page set (it must not).

**Fix.** Seal the bytes at the producer under the mutex; drain sorted post-join; keep the lifecycle in `proveKernel` + `drainUpdateGlobalDirect`.

**Code.** `prover.hpp · UpdateGlobalDirectRec` / `updateGlobalDirectPages` / `updateGlobalDirectLess` / `drainUpdateGlobalDirect` / `dischargeToBeProved` / `dischargeContradiction`; `prover.cpp · updateGlobalDirectLess` / `drainUpdateGlobalDirect` / `proveKernel`. Unit test: `update_global_direct_sealed_drain_matches_sorted` (`test_reconstruct_implication.cpp`). See also [I-28](#i-28), `I-84`, `I-124`, `I-135`, [D-187](40_decisions.md#d-187).

---

<a id="i-146"></a>
## I-146 — `updateEquivalenceClasses`'s origin sync rebuilds each class key's `exprOriginMap` run via a 0%-heap serialize+`memcmp` raw-door replace; no `newRun` / `recordsAt` heap


**Scope.** `exprOriginRunReplace` (`memory.hpp`, the extracted per-key run-replace) and its sole production caller, the origin-sync block at the tail of `ExpressionAnalyzer::updateEquivalenceClasses` (`prover.hpp`). Established by the pre-cycle rump batch of the transient-statification close-out (row 361, ). `exprOriginMap` is `TypedColdBlobMap<int64_t, IdOrigin>`.

**Rule.** The equi-class → body origin merge (`merged:= classOrigins; overwriteOriginsId(merged, body)`) is a per-key whole-run RMW that materializes NO heap `std::vector<IdOrigin> newRun` and NO `recordsAt` body snapshot. Under a function-scoped `ScratchScope` on the accumulator's own byte-bump arena (`mcArena`, a `genScratchArenas` slot — nested ABOVE `forEachOriginSorted`'s reused `okeys`/`idx`, which byte-bump the SAME arena before the visitor runs), pass 1 serializes every class `LineView` line (`serializeOriginTo`, byte-exact to `Codec<IdOrigin>::serialize`) into a `cap * kMaxOriginBlobBytes` staging buffer (`classLines.size <= cap` by the accumulator's per-key cap — a loud Rule-19 assert); pass 2 peeks each body blob (`peekRecordBytes`, a page straddle spilling onto the SAME arena ABOVE the buffer, reclaimed by the scope), dedups against ALL accepted lines by whole-blob `memcmp` (the codec is injective, so byte-equality IS `IdOrigin` equality — the twin of the heap `std::find` `IdOrigin::operator==`), and appends below the cap (== the heap `newRun.size >= cap` break); one raw `exprOriginMap.inner.assignRun` COPIES the buffer into the map's own blob pool (a different arena) before the scope rewinds, so nothing dangles (09b pitfall 6). The merge mints NOTHING — class and body share the LB's `originInterner` int64 keys — so the produced cold blobs AND the id-ordered deload stream are byte-identical to `assignRun(key, newRun)`. `exprOriginMap` is verifier-walked process documentation (the project conventions, [I-44](#i-44)); its content is unchanged.

**Spot.** A revived `std::vector<IdOrigin> newRun` / `std::vector<int64_t> deps` / `recordsAt(bid)` inside the origin-sync block; a `ScratchScope` opened OUTSIDE the `forEachOriginSorted` visitor (would free `okeys`/`idx` on rewind); a `bp` peek held across the next peek or across the terminal `assignRun`; a dedup comparator on decoded `IdOrigin` fields instead of the whole-blob `memcmp`; the `classLines.size <= cap` assert softened.

**Fix.** Keep the merge blob-native through `exprOriginRunReplace`; if any clean path appears to need a heap `newRun` or a change to the `IdOrigin` blob layout, STOP and raise with the user (Rule 8) — do not choose it silently.

**Code.** `memory.hpp · exprOriginRunReplace`; `prover.hpp · updateEquivalenceClasses` (the origin-sync `forEachOriginSorted` call). Unit test: `expr_origin_run_replace_matches_heap` (`test_memory.cpp` — class-lines-only, class+body dedup hit, cap-gated body truncation, page-straddling body blob). See also [I-44](#i-44), `I-91`, [I-121](#i-121), `I-98`, `I-138`, `I-149`.

---

<a id="i-152"></a>
## I-152 — `disintegrateExpr2`'s two `std::set<ExpressionWithValidity>` returns become arena-backed sorted-unique channels reproducing `EWV::operator<` byte-for-byte; nothing observable mints


**Scope.** `DisintPairChannel` / `DisintProducts` (`memory.hpp`, the two-channel return holder) and the consumers of `disintegrateExpr2`'s former `std::tuple<std::set<EWV>, std::set<EWV>, int, bool>` return (`addExprToMemoryBlock`'s imps/stmts loops, `disintegrateExprHypothetically`'s statements walk). Established by the cycle-interface batch.

**Rule.** Each channel is a distinct-`(original, validity)` set held as a `ColdHashSet<BytesKeyStore>` interner + parallel `PagedVector<int32_t>` `origIds`/`valIds` on a per-slot `genScratchArenas` arena. Dedup is O(1) via a composite key `original` + `0x01` + `validity` built on the byte-bump tier under a `ScratchScope` and probed/minted in `keys`; the `0x01` separator is never present in MPL text (the same separator `disintegrateExpr2` already uses), so a composite can never collide with a component string or another pair. A NEW row mints `original` and `validity` into that SAME throwaway arena interner — NEVER `NameMap` / `valueInterner` / `originInterner` / `lbStateInterner` — so building or reading a channel mints nothing observable ([I-84](#i-84)). `forEachSorted` reproduces `ExpressionWithValidity::operator<` = `(original, validityName)` lexicographic byte-for-byte through a decoded-lex index on the byte-bump tier (09c §5): `compareSpans(decode(origId))` then, on a tie, `compareSpans(decode(valId))`; the sort does pure `decode` reads (no mint), so the zero-copy spans stay valid across `std::sort` ([I-3](#i-3)). Dedup key equality (`original` byte-equal AND `validity` byte-equal) == `EWV::operator==`, and the comparator IS `EWV::operator<`, so the channel holds exactly the distinct `EWV`s the `std::set` held and consumers fire in byte-identical order — the interner mint sequence they drive and the `.deload/` stream are unchanged. The redundant `startInt` (former `get<2>`) is DROPPED (delivered by the live `memoryBlock.startInt` mutation, a proven self-assign at the sole consumer), never re-added as an `int&` out-param (that would be defensive — Rule 19).

**Spot.** A channel comparator on raw ids (mint order ≠ lex order, [I-84](#i-84)); `forEachSorted` re-minting instead of pure `decode` reads (would dangle the sort spans); a component minted into `NameMap`/`valueInterner`/`originInterner` (would shift an observable id); the composite separator changed from `0x01`; a resurrected `startInt` out-param.

**Fix.** Keep the two channels the exact byte-content and byte-order of the former sets; if any clean path appears to need a different channel content/order, STOP and raise with the user (Rule 8) — the STOP tripwire is any pressure to change WHAT `disintegrateExpr2` emits.

**Code.** `memory.hpp · DisintPairChannel` / `DisintProducts` (wired at `prover.cpp · disintegrateExpr2`). Unit test: `disint_products_channel_matches_set_ewv` (`test_memory.cpp` — scrambled multiset with duplicates + tie cases, byte-identical to a `std::set<ExpressionWithValidity>` oracle). See also [I-84](#i-84), [I-3](#i-3), `I-138`.

---

<a id="i-154"></a>
## I-154 — the `remainingArgsNormalizedEncodedMap` reverse membership side-index is DERIVED: exact NormKey→owning-key answers, never deloaded/enrolled/dumped, rebuilt on canonical reload, verbatim in the raw image


**Scope.** `ReverseArgsIndex` (`memory_infra/reverse_args_index.hpp`) — one instance per `HashMemory` next to `remainingArgsNormalizedEncodedMap` (`TypedColdBlobMap<Int16SetKey, NormKey>`). Established by the request-generation reverse-index batch that inverted the firing-check candidate loop's former O(keys) forward scan + per-candidate O(run) byte-peek membership into one hash probe. Layered on the I-117 derived-index pattern.

**Rule.** The reverse index maps a `NormKey`'s serialized bytes (`Codec<NormKey>::serialize` == key `encode`) to the run of forward-map key ids whose stored run contains that exact NormKey. It is DERIVED exactly as the cold-map family's `key→id` index (I-117): it rides the LB's deloadable `LbArena` (zero heap — a `ColdHashSet<BytesKeyStore>` NormKey interner + three `PagedVector<int32_t>` chain columns, on its OWN private `DirtyState` so a derived mutation never marks the LB dirty), but is NEVER enrolled in `LbMemory::visitContainers`, NEVER in the canonical deload stream, and NEVER dumped. Lifecycle mirrors the cold-map `PagedHashIndex` buckets one-for-one: (a) maintained incrementally by `appendEdge` at the single-threaded `insertRemainingArgsNormKey` install sites (one edge per NormKey newly added to a forward key's run — the `!dup` path only); (b) cleared then rebuilt wholesale by `rebuildReverseIndex` from the forward map's cold runs at the canonical-reload seam (`Memory::reloadFromImage`'s canonical `else` branch, after `loadLbMemory`) and after `wipeRemainingArgsForClosed`'s survivor re-derivation; (c) freed page-by-page by `clear` at the canonical release seam (`Memory::releaseStaticBlocks`, BEFORE `manager.releaseAll` — the throw-away pages must not outlive the arena as stale vids); (d) captured verbatim by the v4 RAW eviction image (whole-arena memcpy) and restored on raw reload WITHOUT a rebuild, so `releaseStaticBlocksRaw` (no container walk) and the raw reload path add NO hook — the members' `~PagedVector`/`~ColdHashSet` residency branches handle a raw-deloaded teardown. Discharge needs no hook: the four `HashMemory` bands survive discharge (`survivesDischarge` true for tags 51..450) and `reshuffle`/`compactPages` is content-invisible (stable vids, I-107). The answer is EXACT — no false positives (membership is by exact bytes in `normKeys_`, so a hash collision between distinct NormKeys is resolved by the `BytesKeyStore` byte compare), no omissions (every run member gets an edge), no duplicates (the forward run is sorted-unique and `appendEdge` fires once per newly-added NormKey) — so the candidate loop's set `{subset} ∩ {run contains tpleNorm}` and its `int16SetKeyLexCompare` emission order are byte-identical to the former full scan.

**Spot.** The reverse index enrolled in `visitContainers` or given a `ContainerTag` (it must never touch the deload stream); `appendEdge` fed the LB's shared `dirty` instead of `derivedDirty_` (would force needless deload dumps); a rebuild hook placed in the RAW reload branch (double-frees the image-restored pages); a `clear` MISSING at canonical release (stale vids double-freed at the next rebuild); a rebuild MISSING at canonical reload (a stale-empty index silently drops candidates — an unsound theorem loss, not merely slow); the candidate loop keeping the old per-candidate `nkPresent` byte-peek recheck (dead once the reverse index is exact) or dropping the exactness by bucketing on hash without the byte compare.

**Fix.** Keep the reverse index a faithful derived twin of the cold-map buckets; if any clean path appears to need it in the deload stream or an over-approximate (hash-bucketed) answer, STOP and raise with the user (Rule 8) — the candidate-set exactness is the soundness contract.

**Code.** `memory_infra/reverse_args_index.hpp · ReverseArgsIndex` (`appendEdge` / `reverseIndexRunOf` / `rebuildReverseIndex` / `clear`); wired at `memory.cpp · checkLocalEncodedMemoryStatic` (probe), `prover.hpp · insertRemainingArgsNormKey` (append), `memory.hpp · wipeRemainingArgsForClosed` (rebuild), `memory.cpp · Memory::reloadFromImage` / `releaseStaticBlocks` (reload/release hooks). Unit tests: `rebuild_membership_matches_brute_force` / `append_edge_matches_brute_force` / `clear_then_rebuild_is_identical` (`test_reverse_args_index.cpp`). See also [I-117](#i-117), `I-107`, `I-95`, `I-103`, `I-111`, [D-199](40_decisions.md).

---

<a id="i-155"></a>
## I-155 — there is exactly ONE request generator; the obligatory-stump length (0, 1, 2) is its only mode, and the grow loop bumps the submatch tally exactly once per node


**Scope.** `ExpressionAnalyzer::generateEncodedRequestsStatic` (`memory.cpp`) and its five call sites in `performElem2` (`prover.cpp`) plus the counter-example call site's `stumpLen = 0`. Supporting: `filterIntEncodedStatements` (`memory.cpp`, the `alsoAcceptFullKeys` flag), `requestGatesPass` (`prover.hpp`), the two obligatory-stump builders, and `Stump` (`memory.hpp`).

**Rule.**
1. **One function.** No second request generator may be added. A new mode is a new `stumpLen`, or it is a parameter — never a copy of the grow DFS. The retired `generateEncodedRequestsStaticPairs`, `generateEncodedRequestsStaticCE`, `growBaseCandidates` and `filterIntEncodedStatementsCE` are gone; re-introducing any of them is the violation this invariant exists to catch.
2. **`stumpLen` determines everything.** Target owner-set map = `normalizedEncodedKeys` (0) / `…SubkeysMinusOne` (1) / `…SubkeysMinusTwo` (2). Grow depth = `maxKeyLength - stumpLen`. `alsoAcceptFullKeys = (stumpLen == 0)`. Seed and merge phases are empty when `stumpLen == 0`, because a stump of no elements has no instances.
3. **The submatch tally is bumped once per grow node, by the subkey probe alone.** The node runs `requestGatesPass`, builds one normalized key, and probes the subkey map and the target map through the non-counting `ownerKeyAccepts`. It must NOT call `preEvaluateFromEncoded` (which bumps) more than the merge step already does. The tally caps the burst and drives the LB-split policy (D-109), so an extra bump is a proof change, not a metric wobble.
4. **An empty stump emits inside the search.** With `stumpLen == 0` a recorded candidate IS the request and is handed to the consumer at the node that found it. Buffering it into `baseCandidates` and emitting after the DFS would keep the answer and destroy the counter-example filter's contradiction early-exit (I-73). The empty-stump path also asserts `g_splitCount == 1` — that assert is what licenses its (different) tally.
5. **A candidate carrying a stump must remain growable.** Its record gate is `ownerKeyAccepts(targetKeys) && subOk`. With no stump the `subOk` conjunct is dropped: the candidate need not be extendable, because nothing will be attached to it. Dropping `subOk` for a non-empty stump widens the base-candidate pool; keeping it for an empty one loses every request whose whole key is not also a subkey.
6. **The merge concatenates `base ++ stump`, then one stable sort by name.** A stump element tying a base element on core-expression name lands AFTER it. This does NOT decide hit/miss — `addToHashMemory` installs every weakly-name-sorted permutation of a rule's premises, so both tie orders are present as keys. It IS observable further downstream: the emitted tuple order feeds `StaticRequestEmitter::seen`'s dedup bytes and `checkLocalEncodedMemoryStatic`'s probe, and changing it moved one Gauss burst census by five intermediate expressions (no artifact changed). Treat the concatenation order as frozen until that sensitivity is understood; see `D-200`'s open question.

**Why.** Three copies of one DFS is how the CE variant silently drifted: it grew a dual full-key / subkey check the other two never had, and a filter that accepted a union the other two never accepted. Neither difference was a mode — both were consequences of the stump being empty. The prunes are sound over-approximations (I-70, I-79), so the correctness argument is the same at every stump length; only the bookkeeping differs. The singles and pairs variants had also drifted apart on merge tie order without either being right; unifying them surfaced a pre-existing order sensitivity downstream of request generation (`D-200`, open question).

**How to spot violations.** A second function whose body contains `ArenaStack<StackItem>` over `filteredIdx`; a `preEvaluateFromEncoded` call inside the grow DFS; a `stumpLen`-shaped `if` that selects between two whole loops rather than between two values; `baseCandidates.push_back` on the `stumpLen == 0` path; a merge that puts the stump before the base.

**How to fix on violation.** Fold the second generator back in as a `stumpLen` value (or a parameter of the existing one). If the grow DFS genuinely needs a counting probe, prove the tally is unobservable on that path the way the empty-stump assert does — do not assume it.

**Code.** `memory.cpp · generateEncodedRequestsStatic` / `filterIntEncodedStatements` / `makeMandatoryEncodedStatementLists1Static` / `makeMandatoryEncodedStatementLists2Static`; `prover.hpp · requestGatesPass` / `BurstSink::canAccept` / `g_growthMatchCount`; `memory.hpp · Stump` / `BaseCandidate` / `StaticRequestEmitter`; `prover.cpp · performElem2` (the six call sites). Unit tests: `test_memory.cpp · generate_encoded_requests_static_all_stump_lengths` (all three lengths find the same single request by three different routes), `filter_int_encoded_statements_also_accept_full_keys`, `request_gates_pass_hypo_scope_and_length`. See also `D-200`, [I-70](#i-70), [I-79](#i-79), [I-73](#i-73), [I-130](#i-130).

---

## Meta-remarks

<a id="i-156"></a>
## I-156 the split stump joins the BASE CANDIDATE, is attached only for the probes, and its empty-candidate node is recorded on its own

**Scope.** Prover / request generation. `memory.cpp::generateEncodedRequestsStatic` (the grow phase), `memory.cpp::produceExpressionStumps`, `prover.cpp::performElem2`, `memory.hpp::SplitStumpRef` / `ExpressionStump`.

**Rule.** Three things, each load-bearing.

1. **The stump is part of the base candidate, never of the obligatory stump.** `stumpLen` stays 0/1/2 and the target map stays `normalizedEncodedKeys` / `…SubkeysMinusOne` / `…SubkeysMinusTwo`. Attaching the split stump to the obligatory sequence would make a base candidate a key minus `|O| + |S|` elements, requiring a minus-`k` owner-set map for every stump length — an unbounded family of cold containers in the deload stream.
2. **A growing candidate never carries the stump.** Per node the union is built into a stack array, both probes run on it, and it is dropped; only a candidate the target map accepts materialises the union into a `BaseCandidate`. Grow depth counts the union. The search skips filtered statements that are in the stump, so the union is a merge of two disjoint ascending runs under `(decoded name, statement index)` — the order the filtered list is sorted into.
3. **The empty-candidate node is probed on its own, before the search.** Unsplit, the base candidate `{x}` is recorded inside the loop of the candidate one level up. A sub-part stumped on `{x}` never runs that loop.

**Why.** (1) is what makes the dimension implementable at all. (2) keeps the union a merge rather than a set operation, and keeps the submatch tally one bump per node (D-109). (3) is the difference between the sub-part emitting `{x}` + obligatory stump and losing it silently.

Completeness of (2) rests on `normalizedEncodedSubkeys` holding every name-sorted SUBSET of a key, not merely its prefixes — `addToHashMemory` walks every permutation and installs each prefix that is still weakly name-increasing — and on `requestGatesPass` being closed downward under subsets. Otherwise probing `C ∪ S` at each node would dead-end: reaching `{a,b,c}` from stump `{c}` passes through `{a,c}`, which is no prefix.

**How to spot.** A request emitted by a stump sub-part that does not contain its stump; a base candidate of size 1 vanishing under a stump split (the empty-candidate node was not probed); the unit test `split_stump_alone_is_recorded_as_a_base_candidate` failing; a `MinusThree` map appearing.

**How to fix on violation.** Keep the union transient. Keep the empty-candidate probe with its tally bump. Never lengthen the obligatory stump.

**Code.** `memory.cpp::generateEncodedRequestsStatic` (the `unionWithStump` / `inSplitStump` lambdas and the pre-search probe), `memory.cpp::produceExpressionStumps`.

**See also.** [D-203](40_decisions.md#d-203), [I-155](#i-155), [I-70](#i-70), [I-79](#i-79), [I-130](#i-130).

---

<a id="i-157"></a>
## I-157 every recordable node from a replaced stump level survives as terminal-only work, is validated in the real request batch, and never grows

**Scope.** Prover / request generation. `memory.cpp::produceExpressionStumps`, `memory.cpp::generateEncodedRequestsStatic`, `memory.hpp::ExpressionStump`, `prover.cpp::proveKernel`.

**Rule.** When `produceExpressionStumps` successfully constructs a replacement level, it probes every node in the level being replaced against `overallHashMemory.normalizedEncodedSubkeysMinusOne` and `...MinusTwo`. An accepted node is appended as `ExpressionStump::terminalOnly == 1`; no recordable shallow node is discarded. `proveKernel` includes that record in the same deterministic round-robin deal as regular stumps, so exactly one bucket owns it. The consumer repeats the stump-alone gates and owner-map probes against the call's actual `intMemory`, may append the node as a `BaseCandidate`, and then MUST skip the depth-first search for that record. Regular children in the replacement level remain the sole owners of every larger combination.

The producer's overall hash memory is a conservative discovery superset: recovered implications enter it before the status-specific working or local memories. It may identify terminal work a particular batch rejects, but cannot omit terminal work that batch needs. The producer MUST NOT call `checkLocalEncodedMemoryStatic` or any firing path; validation and firing remain `generateEncodedRequestsStatic` → `preEvaluateFromEncoded` → `StaticRequestEmitter` → `BurstSink::consume` → `checkLocalEncodedMemoryStatic`.

**Why.** A length-`L+1` stump cannot reproduce a recordable base candidate of length `L`. Treating the retained node as an ordinary seed would instead overlap the child frontier. Terminal-only ownership preserves both the shallow request and the non-overlapping larger search.

**How to spot.** The old `stump fan drops a recordable base candidate` assert; a recordable shallow request missing only when the producer grows beyond level 1; `terminalOnly` records entering the depth-first stack; a producer calling the checker directly; a terminal record tested only against `overallHashMemory` and trusted without the per-batch probes.

**How to fix on violation.** Retain the node before advancing the producer frontier, deal it once, keep the existing stump-alone probe, and stop that run immediately afterward. Do not remove the child node or bypass the normal request pipeline.

**Code.** `memory.cpp::produceExpressionStumps` (the replaced-level scan and terminal append), `memory.cpp::generateEncodedRequestsStatic` (the terminal stop after the stump-alone probe), `prover.cpp::proveKernel` (the unchanged all-record bucket deal).

**See also.** [D-202](40_decisions.md#d-202), `I-156`, `I-158`, [I-155](#i-155), [I-70](#i-70), [I-79](#i-79).

---

<a id="i-158"></a>
## I-158 a stump sub-part owns a BUCKET of stumps; the filter is built once per bucket and the emitter collapses the overlap

**Scope.** Prover / request generation. `memory.hpp::SplitStumpRef`, `memory.cpp::generateEncodedRequestsStatic` (the per-stump run loop), `prover.cpp::proveKernel` (the round-robin deal).

**Rule.** A straggler's round-1 PRODUCER task ([D-201](40_decisions.md#d-201)) returns the regular frontier its whole-LB filter grows (up to `kStumpsPerBucketTarget × logicalCores`) plus any recordable terminal pre-stumps from replaced levels. `proveKernel` deals all records round-robin into `min(stumpCount, logicalCores)` buckets, permuted so each bucket is a contiguous run, and one bucket part runs per BUCKET. Inside the generator the statement filter and its name sort are built ONCE per bucket; the search then runs once per regular stump and performs only the shallow probe for a terminal stump, accumulating into one `baseCandidates` column, and the merge runs once. Requests two of the bucket's stumps both reach are emitted twice into the same `StaticRequestEmitter`, whose `seen` collapses them.

**Why.** One part per stump would multiply the part count. Bucketing pays the fixed per-bucket setup (`filterIntEncodedStatements` + the obligatory-stump builders) once instead of once per stump, and the emitter's dedup — a request two of a bucket's stumps both reach fires once — is a second win unavailable to separate parts. Growing more stumps than buckets (the `kStumpsPerBucketTarget` multiple) then round-robin dealing balances the buckets' grow-tree sizes, so no bucket becomes the LB's straggler (the split-ineffective report catches the residual serial cases).

**How to spot.** Part counts far above `logicalCores` for one LB; a hash burst dominated by `filterIntEncodedStatements`; `SplitStumpRef::count` always 1 where `proveKernel` should have dealt several; buckets keyed on `fixed_number_splits` (retired) rather than `logicalCores`.

**How to fix on violation.** Keep the filter above the per-stump loop. Keep the deal round-robin (a bucket must draw stumps from across the name-sorted list, not one contiguous slice of it) and the permutation that makes each bucket contiguous. Deal into `logicalCores` buckets, not a config count.

**Code.** `prover.cpp::proveKernel` (the classify step's bucket deal, `min(nStumps, logicalCores)`), `memory.cpp::generateEncodedRequestsStatic` (the `runCount` loop), `memory.cpp::produceExpressionStumps`.

**See also.** [D-201](40_decisions.md#d-201), [D-203](40_decisions.md#d-203), [D-202](40_decisions.md#d-202), [D-109](40_decisions.md#d-109), `I-157`, `I-156`, [I-130](#i-130).

---

<a id="i-159"></a>
## I-159 the burst early-exit reads the per-burst `g_isMultiPart`, never `g_splitCount`; the expression split runs at `g_splitCount == 1`

**Scope.** Prover. `gl::g_isMultiPart` (`memory.hpp`/`memory.cpp`), `prover.cpp::performElem2`, `prover.hpp::BurstSink::canAccept` / `::consume`, `memory.hpp::partitionAccepts`.

**Rule.** Two thread-locals with DISJOINT roles that must never be conflated:
- `g_splitCount` (+ `g_splitProcessID`) is the RULE dimension, and ONLY `partitionAccepts` reads it. A whole-LB expression/bucket split keeps `g_splitCount == 1` so `partitionAccepts` accepts EVERY rule — the buckets partition the expression search, not the rules.
- `g_isMultiPart` is the per-burst "this LB runs as > 1 part" signal, set by `performElem2` from `partCount` (`= partCount > 1`), and ONLY the burst early-exit reads it ([I-76](#i-76)).

Setting `g_splitCount > 1` for the expression split (to reuse the old gate) would rule-partition the buckets: a request whose rule falls in partition `j` but whose expression falls in bucket `k ≠ j` is then covered by NO part — incomplete, LOST theorems. So the two signals stay separate: `g_splitCount == 1` (complete rule coverage) AND `g_isMultiPart == true` (early-exit off across buckets).

**Why.** The expression split is sound only because every bucket sees every rule; the early-exit must be off only because the buckets are siblings. These are different conditions and `g_splitCount` can express only the first. `performElem2` asserts `splitStump.count == 0 || partCount > 1` to catch a bucket dispatched as single-part.

**How to spot.** `partitionAccepts` reading `g_isMultiPart`, or the early-exit reading `g_splitCount`; a bucket part dispatched with `splitCount > 1` (would drop cross-terms); theorems present unsplit but absent when a straggler splits.

**How to fix on violation.** Dispatch bucket parts at `splitCount == 1`, `partCount = buckets`. Keep `partitionAccepts` on `g_splitCount` and the early-exit on `g_isMultiPart`.

**Code.** `prover.cpp::performElem2` (sets both from `splitCount` / `partCount`), `prover.cpp::proveKernel` (dispatches buckets at `splitCount == 1`, `partCount = buckets`).

**See also.** [D-201](40_decisions.md#d-201), [D-121](40_decisions.md#d-121), [I-76](#i-76), `I-160`.

---

<a id="i-160"></a>
## I-160 the split trigger is an end-of-iteration straggler statistic over split-invariant TOTAL work; the split set is a deterministic function of proof state

**Scope.** Prover. `prover.cpp::proveKernel` (the classify seam's `lbTotalSub` sum-reduce and the end-of-iteration stats pass) and `ExpressionAnalyzer::isStraggler`.

**Rule.** After each iteration a single-threaded pass over the active main-path LBs sets each LB's NEXT-iteration split from this iteration's completed work:
- `work(L)` = the SUM of `L`'s parts' submatch counts (`lbTotalSub`), NOT the max-over-parts. The sum is split-invariant (a function of `L`'s firing work, [D-117](#d-117)); the max scales `≈ work/parts` and would thrash (a split LB looks light → de-splits → re-splits, the retired [D-111](40_decisions.md#d-111) band).
- `isStraggler(work, T, logicalCores, min_split_work)` = `work > T / logicalCores && work >= min_split_work` (`T = Σ work`), all `int64` — integer arithmetic, no sort, no float, so the verdict is a pure deterministic function of the deterministic work totals.
- A straggler's `numberOfParts` is set to `logicalCores`, everything else to `1`. Recomputed every iteration.

Because the split set is a deterministic function of the PRIOR iteration's deterministic totals (never wall-clock RT), two full runs are byte-identical. Wall-clock RT here would make the split set timing-dependent and the proof graph non-deterministic — forbidden.

**Why.** The straggler decision needs the whole active set (the fair-share `T/C`), so it cannot be a per-LB finalize; it is one pass after the pass loop. Sum-not-max is the anti-thrash property. Integer-not-float removes the last determinism hazard (FP associativity).

**How to spot.** The stats pass reading wall-clock / `RTTracker` instead of `lbTotalSub`; a max-over-parts reduce feeding the trigger; two runs differing in which LBs split; a float threshold.

**How to fix on violation.** Feed `isStraggler` only the summed submatch totals. Keep the arithmetic integer. Keep `lbMaxSub` for the split-ineffective report only, never the trigger.

**Code.** `prover.cpp::proveKernel` (`lbTotalSub` / `lbMaxSub` reduce; the `if (mainPath)` stats pass), `prover.cpp::ExpressionAnalyzer::isStraggler`.

**See also.** [D-201](40_decisions.md#d-201), [D-117](#d-117), [D-109](40_decisions.md#d-109), [I-59](#i-59), [I-77](#i-77), `I-159`.

---

<a id="i-161"></a>
## I-161 MailLog history may roll only when every LB was active at completed-grid inspection; the decision is fixed for the execution batch


**Scope.** `prover.cpp::analyzeExpressions` `buildGrid`, `ExpressionAnalyzer::proveKernel`, and `mail_log.hpp::MailLog::retireDeliveredBatches`.

**Rule.** After all `permanentBodies` entries exist and anchor prehandling is complete, the prover counts initially inactive LBs exactly once. Rolling retirement is enabled if and only if the count is zero and stays fixed for the execution batch. In rolling mode, the single-threaded post-phase-3 seam may release `mailBlobPool` and `mailRefs`, preserve `mailEdges`, `mailCursor`, and every cumulative `MailHead::count`, and reset only `MailHead::lastRef` to `-1` before the commit barrier opens the next retained window. If the initial count is non-zero, no mid-batch history retirement is allowed: the whole log remains available for dormant catch-up.

**Why.** An induction-zero LB born dormant may activate later with an old cursor ([I-112](#i-112)); deleting any ancestor batch before that first pull loses mail. When no LB is born dormant, every receiver that can remain relevant pulls the prior window in phase 1 before retirement, and later deactivation is permanent. Cumulative counts and cursors let `mailPeek` and `count - cursor` span rolling windows without changing latency or delivery semantics.

**How to spot.** A dynamic switch to rolling after dormant LBs wake; clearing `mailHeads`, `mailEdges`, or `mailCursor`; resetting `MailHead::count`; retiring before phase 3 joins or after the new commit sweep; `MailLog::pull` underrunning a ref chain; a non-zero grid-build dormant count printed with `mode=rolling`.

**How to fix on violation.** Restore the one-time completed-grid count and fixed policy. In rolling mode clear only blob/ref history at the post-phase-3, pre-commit seam and reset only retained-chain heads. In any initially dormant grid retain full history through teardown.

**Code.** `prover.cpp::ExpressionAnalyzer::analyzeExpressions` (`buildGrid` dormant scan), `prover.cpp::ExpressionAnalyzer::proveKernel` (retire seam), `mail_log.hpp::MailLog::retireDeliveredBatches`, `tests/test_mail_log.cpp::repeated_retire_pull_cycles_are_bounded` and `::dormant_catch_up`.

**See also.** [D-204](40_decisions.md#d-204), [I-94](#i-94), [I-112](#i-112), [I-55](#i-55).

---

<a id="i-162"></a>
## I-162 a rolling mail window resets its global id table only after every old-id carrier is empty or retired


**Scope.** `prover.cpp::ExpressionAnalyzer::proveKernel`, `memory.cpp::mailInterner` / `resetMailInterner`, routing `Memory::mailIn` / `mailOut`, and `MailLog` committed blobs.

**Rule.** `resetMailInterner` is allowed only in the zero-initially-dormant rolling mode fixed by [I-161](#i-161), at the single-threaded post-phase-3, pre-commit seam, in this order:

1. every `Memory::mailIn` is empty (asserted);
2. `MailLog::retireDeliveredBatches` removes every blob/reference carrying global ids;
3. `resetMailInterner` invalidates all ids and returns its pages to its arena;
4. the commit barrier translates sender-local `mailOut` ids and remints the next global-id window.

`mailOut` never blocks the reset because it carries ids from its own per-LB interner. A grid with any initially dormant LB performs none of steps 2 or 3 until execution-batch teardown; its old cursor may require any retained blob and its matching id table.

**Why.** A global id has meaning only while the table that minted it is unchanged. Resetting while a blob or inbox still carries an old id makes later decode read the wrong string or assert; retaining the table after all such carriers are gone accumulates distinct strings across hashbursts for no semantic benefit. The ordered quiescent seam makes the lifetime exact and race-free.

**How to spot.** A reset outside `rollingMailHistoryEnabled`; reset before `retireDeliveredBatches`; a non-empty `mailIn` at reset; a global-id `mailOut`; a dormant batch whose interner count restarts; theorem or verifier drift after enabling the reset.

**How to fix on violation.** Restore the four-step seam and the fixed zero-dormant gate. Never weaken the `mailIn.empty` assert. If another container begins retaining global mail ids, either retire it before step 3 or keep the interner; do not decode through a reset table.

**Code.** `prover.cpp::ExpressionAnalyzer::proveKernel` (retire/reset/commit seam), `memory.cpp::resetMailInterner`, `memory.hpp::mailInterner`, `mail_log.hpp::MailLog::retireDeliveredBatches`, `tests/test_memory.cpp::mail_interner_reset_restarts_the_delivery_window`.

**See also.** [D-206](40_decisions.md#d-206), [I-161](#i-161), [I-127](#i-127), [I-101](#i-101).

---

<a id="i-163"></a>
## I-163 outgoing routing mail and every id it contains deload as one LB-owned unit; the shell pending bit is the only cold-safe commit probe


**Scope.** `memory_infra/deloadable_mail_out.hpp::DeloadableMailOut`, `LbMemory::mailOut`, `Memory::mailOutInterner` / `mailOutPending` / outgoing write doors, `ExpressionAnalyzer::proveKernel` commit sweep, and post-join root deposits.

**Rule.** Each LB's outgoing statements, origin keys, origin dependencies, and dedicated string table occupy one `DeloadableMailOut` on `lbMemory.manager`, enumerated at append-only tag band 755..804. Every stored id belongs to that mailbox's private interner. Every write goes through `Memory::insertMailOutStatement` or `Memory::addMailOutOrigin`, which sets always-resident `mailOutPending` and refreshes the logical-byte snapshot. The commit sweep MUST NOT inspect cold mailbox columns before claiming the LB: it skips only on `mailOutPending == false`; otherwise it claim/reloads, asserts the mailbox is non-empty, commits, calls `clearMailOut` (columns + interner + shell summaries), then releases the claim. The tag band survives discharge until that final commit. Any post-join writer to the root outbox holds the root claim across the deposit.

**Why.** A per-LB mailbox id has meaning only with the exact table that minted it; deloading or clearing one without the other corrupts decode. The shell bit removes the old need to keep every outbox resident merely to test emptiness. Under pressure the steward can release completed phase-3 LBs and the serial barrier materializes at most its working window rather than every producer's output arena simultaneously.

**How to spot.** Direct `mailOut.insertStatement` / `origins_` writes; an origin dependency encoded through `originInterner`; a commit-side `mailOut.empty` probe before `claimAndLoadForWork`; clearing columns without resetting the private interner and pending bit; tag 755..804 missing from `visitContainers` or `survivesDischarge`; a post-join root merge without a claim.

**How to fix on violation.** Route all writes through the `Memory` doors, keep the private id table and both columns in the same deload band, use only the shell bit before claim, and reset the entire unit with `clearMailOut` after delivery.

**Code.** `memory_infra/deloadable_mail_out.hpp::DeloadableMailOut`; `memory_infra/lb_memory.hpp::LbMemory`; `memory.hpp::Memory::insertMailOutStatement`, `::addMailOutOrigin`, `::clearMailOut`; `prover.cpp::ExpressionAnalyzer::proveKernel`; `mail_log.hpp::MailLog::commit`. Tests: `test_mail_log.cpp` deloadable-mail-out tests and `test_lb_deload.cpp::memory_deload_reload_roundtrip`.

**See also.** [D-205](40_decisions.md#d-205), [I-101](#i-101), [I-102](#i-102), [I-162](#i-162).

---

- **Ordering rationale.** Invariants are sorted by when they entered the document, not by priority. For a priority-ordered view (what an agent checks first when debugging), the quick-reference table in [`AGENT_SwDD.md`](AGENT_SwDD.md#invariant-quick-reference) groups by scope.
- **Stability of numbering.** Once an invariant is numbered, the number is immutable. If an invariant is retired, its section becomes a short "retired — superseded by I-K" stub, preserving the anchor.
- **External cross-references.** Memory files referenced by name are the long-form rationale; this document is the operational rule.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
