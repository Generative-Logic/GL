/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschränkt)

 This program is free software : you can redistribute it and /or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.If not, see < https://www.gnu.org/licenses/>.

 ------------------------------------------------------------------------------

 This software is also available under a commercial license.For details,
 see: https://generative-logic.com/license

 Contributions to this project must be made under the terms of the
 Contributor License Agreement(CLA).See the project's CONTRIBUTING.md file.*/
#include "phase2_executor.hpp"
#include "../project_paths.hpp"
#include "../infra/peak_heap.hpp"
#include "../infra/phase2_trace.hpp"
#include "../infra/diagnostics_log.hpp"

#include "../parameters.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../infra/rt_tracker.hpp"
#include "../infra/memory_census.hpp"
#include "../memory_infra/global_memory_manager.hpp"
#include "../memory_infra/scratch_arena.hpp"
#include "../memory_infra/lb_deload.hpp"
#include "../memory_infra/deload_stats.hpp"
#include "../gpu/phase2_projection.hpp"
#ifdef GL_CUDA
#include "../gpu/phase2_cuda.hpp"
#include "../gpu/phase2_sealing.hpp"
#include "../gpu/phase2_host_plan.hpp"
#endif
#include <iostream>
#include <fstream>
#include <array>
#include <numeric>
#include <algorithm>
#include <optional>
#include <utility>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <chrono>
#include <cstdio>
#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif


namespace gl {
/// @brief Bind the phase2 executor operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param phase2Backend Borrowed Phase2Backend interface or state; must outlive this component.
/// @param phase2CumulativeSeconds Borrowed double interface or state; must outlive this component.
/// @param phase2MeasuredIterations Borrowed uint64_t interface or state; must outlive this component.
/// @param phase2MeasurementEpoch Borrowed uint64_t interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param requestGenerator Borrowed HashRequestGenerator interface or state; must outlive this component.
/// @param ruleEvaluator Borrowed HashRuleEvaluator interface or state; must outlive this component.
/// @param anchorID_ Borrowed std::string interface or state; must outlive this component.
/// @param logicalCores Borrowed unsigned interface or state; must outlive this component.
/// @param steward Borrowed std::unique_ptr interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param anchorInfo Borrowed ce::AnchorInfo interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
Phase2Executor::Phase2Executor(const ProverParameters& parameters, Phase2Backend& phase2Backend, double& phase2CumulativeSeconds, uint64_t& phase2MeasuredIterations, uint64_t& phase2MeasurementEpoch, bool& ceFilteringActive, HashRequestGenerator& requestGenerator, HashRuleEvaluator& ruleEvaluator, std::string& anchorID_, unsigned& logicalCores, std::unique_ptr<MemorySteward>& steward, AdmissionMemory& admissionMemory, const ce::AnchorInfo& anchorInfo, const CompiledDefinitionRegistry& definitions) : parameters(parameters), phase2Backend(phase2Backend), phase2CumulativeSeconds(phase2CumulativeSeconds), phase2MeasuredIterations(phase2MeasuredIterations), phase2MeasurementEpoch(phase2MeasurementEpoch), ceFilteringActive(ceFilteringActive), requestGenerator(requestGenerator), ruleEvaluator(ruleEvaluator), anchorID_(anchorID_), logicalCores(logicalCores), steward(steward), admissionMemory(admissionMemory), anchorInfo(anchorInfo), definitions(definitions) {}

/// @brief One burst part — request generation + the fixpoint
        ///        evaluation pass, writing deposits to a per-part buffer.
        ///
        /// @details
        /// The inner unit of phase 2 — one `(LB, part)` task in `proveKernel`'s
        /// flat executor pool. Generates the hash requests for its part (an
        /// unsplit LB, one expression bucket of a straggler, or the CE path),
        /// then runs the single-pass `FIXPOINT_LOOP`
        /// capturing each firing's deposit onto the task's sealed-page record
        /// chain (`sealedPages.appendRecord`; it does **not**
        /// apply them — the per-LB finalize `performElemPhase2` hands all of an
        /// LB's sealed part sets, in part order, to `applyFiringRecords` after
        /// the pool join). The
        /// per-step delta clears are likewise NOT done here — the finalize clears
        /// them once per LB before the merge (so every part still reads the deltas
        /// it generates from). Read-only on the LB ([D-116])
        /// so concurrent parts on one LB never race. The arena buffers
        /// are per-thread scratch (caller-owned, reset per task), and
        /// `sealedPages` — records and their strings alike — is the task's own
        /// output slot.
        ///
        /// @param body          The shared LB (read for request generation /
        ///                      firing; deposits go to the sealed-page record
        ///                      chain, not the LB).
        /// @param coreId        Worker core id (passed to
        ///                      `checkLocalEncodedMemoryStatic`).
        /// @param partCount     Total concurrent parts of this LB this burst
        ///                      (`>= 1`): a whole-LB expression split runs
        ///                      `partCount` bucket parts. The doom line's winner
        ///                      is one of these parts.
        /// @param splitStump    The LB split's expression dimension: this bucket
        ///                      part's stump, which every request it generates
        ///                      must contain, plus its place among its siblings
        ///                      (which the generator's seed phase reads).
        ///                      Default-constructed for an unsplit LB and for the
        ///                      counter-example filter. Handed to every
        ///                      request-generator call in the burst.
        /// @param sealedPages   This task's page set: record chain + the sealed
        ///                      strings/spans the records reference. Bound and
        ///                      `Filling` at entry; the caller seals it after
        ///                      this returns.
        /// @param doomLine      External per-LB packed (position, ordinal) doom
        ///                      line, shared by all the LB's parts (lives OUTSIDE
        ///                      the LB so the hashburst stays read-only on the
        ///                      LB). `kNoDoomLine` at burst start. Read by the
        ///                      streaming sink's `canAccept`; lowered by CAS-min
        ///                      when `burstDeactivates` fires.
        /// @return (void). The split-policy marker is this part's submatch count
        ///         (`gl::HashRequestGenerator::growthMatchCount()`, the subkey growth-probe tally),
        ///         read by the worker from the thread_local right after this call;
        ///         `proveKernel` SUMS the LB's parts into its total work, which the
        ///         end-of-iteration `isStraggler` pass consumes.
        /// @see `performElemPhase2` — the orchestrator that calls this and merges.
        /// @see `isStraggler` — consumes the LB's summed submatch work.
        /// @see `produceExpressionStumps` — where a stump comes from.
void Phase2Executor::performElem2(const Memory& body, unsigned coreId,
    int partCount,
    const SplitStumpRef& splitStump,
    SealedPageSet& sealedPages,
    std::atomic<int64_t>& doomLine) {

    // The LB split's expression dimension. A stump-split sub-part carries a bucket
    // of stumps, and every request it generates contains one of them; an ordinary
    // whole-LB part carries none.
    assert((splitStump.count == 0) == (splitStump.stumps == nullptr)
        && "performElem2: stump bucket present iff non-empty");
    // A stump bucket only ever rides a MULTI-PART LB: the whole-LB expression split
    // runs partCount bucket parts, and the doom line's winner
    // selection relies on each bucket part carrying its bucket ordinal among
    // partCount siblings.
    assert((splitStump.count == 0 || partCount > 1)
        && "performElem2: a stump bucket belongs to a MULTI-PART LB");
    // The burst reads the rule indexes: every staged install write was
    // flushed at its window's close (D-333), so this
    // slot's stagings are empty.
    assert(ruleStagings().slotIsEmpty(coreId)
        && "performElem2: staged rule-index writes reached the burst - a flush seam was missed");
    assert((splitStump.count == 0 || splitStump.total >= 1)
        && "performElem2: a stump sub-part has a place among its siblings");
    assert(partCount >= 1 && "performElem2: partCount is the LB's part count (>= 1)");
    // The doom line identifies its winner by the part's invocation ordinal; an
    // unsplit part is ordinal 0 by construction (default SplitStumpRef).
    assert((splitStump.count > 0 || splitStump.ordinal == 0)
        && "performElem2: an unsplit part carries ordinal 0");

    // Per-call RT tracker home. performElem2 is the hashburst (request
    // generation + the inline fixpoint check) — where a runaway LB spends its
    // time — and its REQGEN_BATCH* / FIXPOINT_LOOP RT_SCOPE_HERE markers already
    // record into the thread-local tracker this declares. Stripped to nothing
    // when RT_MEASUREMENT == 0. RT requires disable_lb_split (one part per LB) so
    // this call measures the LB's whole hashburst on one thread, writing one
    // .rt/<chain>.log per LB with no cross-part races. See
    // D-110, I-59.
    RT_TRACKER_DECL(body);

    // The request keys + the IntEncodedExpr copies now ride this slot's gen
    // scratch arena (genScratchArenas().forSlot(coreId)), released per worker
    // task at this function's single exit — so each executor starts with a
    // fresh arena and no per-entry reset is needed. Keys stay private to this
    // executor's slot (one executor per slot at a time), so parallel LB-split
    // parts never write a shared key arena (D-125).

    // This slot's scratch string arena is filled by the firing-check path
    // below (transient strings under per-call rewind windows; FiringRecord
    // fields stay owned copies / sealed pages, so nothing points into the
    // arena across the call) and RELEASED back to the pool at this executor's
    // exit (see the end of this function) — no blocks retained after the
    // worker finishes.

    // Submatch counter reset: zero the per-split-part tally at the start of this
    // part. The generator's grow-DFS bumps it on every match this part owns;
    // canAccept caps the burst on it, and the worker
    // reads it after this call to drive the split policy. See D-109.
    gl::HashRequestGenerator::resetGrowthMatchCount();

#if RT_MEASUREMENT
    // [RT phase measurement] arm the C15 attribution for the dump-target
    // LB's parts only; reassigned on every entry so no stale thread state.
    g_crtArmed = gl::hashburst_dump::isTargetLB(body);
    if (g_crtArmed) {
        g_crtPreorderId = body.nameMap.lookup(std::string("preorder"));
        g_crtFiveId = body.nameMap.lookup(std::string("5"));
        if (g_crtPreorderId == 0 || g_crtFiveId == 0) g_crtArmed = false;
    }
#endif

    // Processor requests stop at the original deterministic cutoff. Complete
    // output of the winning request is retained; Phase 3 owns discharge.
    BurstSink sink{ &ruleEvaluator, &body, coreId, &sealedPages, &doomLine,
        static_cast<int32_t>(splitStump.ordinal),
        SealedRecordCursor<FiringRecord>(sealedPages) };

    // ===================================================================
    // Request generation. performElemPhase1's absorb filled
    // body.workingMemory / body.intExternalStatements from THIS burst's
    // freshly-arrived mailIn.statements (status=3 disintegration), so the
    // same-burst
    // request blocks below see all newly-arrived rules and facts.
    // ===================================================================

    if (ceFilteringActive) {
        RT_SCOPE_HERE("REQGEN_CE_MODE");
        // --- CE mode: no ingredient is obligatory, so the term list is empty and
        // every whole key is a request.
        //
        // The counter-example filter runs one unsplit LB per conjecture, which is
        // what licenses its submatch tally going unread (the cap is bypassed in
        // BurstSink::canAccept, and an unsplit LB has no split policy). It is the
        // only caller for which that holds, so the statement lives here and not in
        // the generator, which cannot tell this call from a termless prover batch.
        assert(partCount == 1
            && "performElem2: the counter-example filter runs one unsplit LB per "
               "conjecture");
        requestGenerator.generateEncodedRequestsStatic(body, body.overallHashMemory,
            /*terms=*/nullptr, /*termCount=*/0,
            SplitStumpRef{}, coreId, sink);
    } else {
        // --- Normal mode: four request batches, one per rule registry ---
        // Each batch names, as mandatory-containment terms, the ingredient a
        // request must be new because of; batch 5 names none, because there the
        // rule itself is what is new. The registries differ, which is why the
        // batches cannot share one call: the registry drives the key length, the
        // target owner map and every owner probe. Batch 1 reads the persistent
        // `body.workingMemory` and the mail batches read
        // `body.intExternalStatements`, all filled by the absorb above.

        // --- Batch 1: mail-recovered rules (body.workingMemory) ---
        // Filled by the absorb above (status=3 recovered implications). What is
        // new here is the RULE, so a request has to carry it against a fact this
        // LB already holds: one term naming the local statements. The leading
        // sink.canAccept() skips the batch once the LB hit the cap or was
        // early-exited by another part; the emptiness test skips a search whose
        // only term no candidate could satisfy.
        if (sink.canAccept() && !body.workingMemory.encodedMap.empty()
            && !body.intLocalEncodedStatements.empty()) {
            RT_SCOPE_HERE("REQGEN_BATCH1_WORKING_MEMORY");
            MandatoryTerm terms[1];
            terms[0].views[0] = IntStmtView(body.intLocalEncodedStatements);
            terms[0].viewCount = 1;

            requestGenerator.generateEncodedRequestsStatic(body, body.workingMemory,
                terms, /*termCount=*/1,
                splitStump, coreId, sink);
        }

        // --- Batch 2+3: everything new this burst, in ONE search ---
        // Both former batches read the same rule registry (overallHashMemory) and
        // the same statement universe (intEncodedStatements); they differed only
        // in what they obliged a request to contain. The mandatory-containment
        // control states that difference directly, so one enumeration covers both
        // and reaches each request exactly once — no pairing merge, and no
        // cross-batch duplicate for a request that carries both a fresh local
        // statement and a fresh arrival.
        //
        // Term 1: a statement derived here this burst.
        // Term 2: a mail arrival AND a local statement — a purely-external
        //         combination already fired at the ancestor that mailed it (I-57),
        //         and batch 4 below covers arrivals against this LB's own rules.
        if (sink.canAccept()) {
            RT_SCOPE_HERE("REQGEN_BATCH23_NEW_THIS_BURST");
            // The mail side is staged by the same absorb that offers each arrival
            // to the statement registry, so a staged arrival that clears the
            // iteration cap is registered here and the one grow universe below
            // reaches it. A miss would silently drop every request that arrival
            // could serve, so it is a hard stop, not a widened universe.
            for (int32_t k = 0; k < body.intExternalStatements.size(); ++k) {
                const IntEncodedExpr& ext = body.intExternalStatements[k];
                if (ext.maxIteration > parameters.maxIterationNumberVariable)
                    continue;
                assert(body.intKnownStatements.find(
                           StatementKey{ ext.originalId, ext.validityId }) != nullptr
                    && "performElem2: a staged mail arrival must still be "
                       "registered at this LB - the absorb stages only what the "
                       "registry accepted, and every registry erase purges the "
                       "matching staged row");
            }

            MandatoryTerm terms[2];
            terms[0].views[0] = IntStmtView(body.intLocalEncodedStatementsDelta);
            terms[0].viewCount = 1;
            terms[1].views[0] = IntStmtView(body.intExternalStatements);
            terms[1].views[1] = IntStmtView(body.intLocalEncodedStatements);
            terms[1].viewCount = 2;

            requestGenerator.generateEncodedRequestsStatic(body, body.overallHashMemory,
                terms, /*termCount=*/2,
                splitStump, coreId, sink);
        }

        // --- Batch 4: localHashMemory x fresh mail ---
        // Every rule this LB already owns has to be tested against the arrivals
        // this burst brought in, so the mandatory ingredient is the mail side:
        // one term naming body.intExternalStatements (filled by the absorb above).
        if (sink.canAccept() && !body.localHashMemory.encodedMap.empty()
            && !body.intExternalStatements.empty()) {
            RT_SCOPE_HERE("REQGEN_BATCH4_LOCAL_X_MAIL_SINGLES");
            MandatoryTerm terms[1];
            terms[0].views[0] = IntStmtView(body.intExternalStatements);
            terms[0].viewCount = 1;

            requestGenerator.generateEncodedRequestsStatic(body, body.localHashMemory,
                terms, /*termCount=*/1,
                splitStump, coreId, sink);
        }

        // --- Batch 5: localHashMemoryDelta ---
        // The rule is what is new here, and a rule that lands this burst must
        // meet every visible statement, not only the ones that arrived with it.
        // So there is no mandatory ingredient at all: the term list is EMPTY and
        // every whole key of this registry is a request. (Naming the whole
        // statement universe as a term would say the same thing and then pay a
        // per-candidate mask test that can never fail.)
        if (sink.canAccept() && !body.localHashMemoryDelta.encodedMap.empty()) {
            RT_SCOPE_HERE("REQGEN_BATCH5_LOCAL_HASH_DELTA");
            requestGenerator.generateEncodedRequestsStatic(body, body.localHashMemoryDelta,
                /*terms=*/nullptr, /*termCount=*/0,
                splitStump, coreId, sink);
        }
    } // end normal mode

    // Per-step delta clears are deliberately NOT done here. A later part of a
    // multi-part LB still reads body.localHashMemoryDelta /
    // body.intLocalEncodedStatementsDelta; clearing per-part would wipe the
    // input the next part needs. performElemPhase2 clears them ONCE, after every part's
    // reqgen and before the merge (D-126). The clear is
    // still pre-merge, so admissionKeysAlgebra / deferredIntegrationPreps are
    // reset before applyFiringRecords repopulates them.

    // Checking is inline in BurstSink::consume; there is no separate fixpoint
    // loop. The hashburst captures firings onto the task's sealed-page record
    // chain (merged by the
    // caller), never into `intKnownStatements`, so the dependency skip-set is
    // burst-fixed and the single generate-and-check pass reaches the same fixed
    // point an iterated loop would (D-104). Per I-66 phase 2 never mutates the
    // LB's `isActive`; deactivation / discharge happen in phase 3's post-burst
    // absorb (`dischargeContradiction` / `dischargeToBeProved`). The streaming
    // early-exit only lowers the external `doomLine` atomic, never the LB.


    // No return: the split-policy marker is this part's submatch count
    // (gl::HashRequestGenerator::growthMatchCount()), read by the worker right after this call and reduced to
    // the busiest part in the finalize sweep. The former emitted-request count
    // (sink.producedCount()) no longer drives the cap or the split.

    // Return this slot's string and request-generation scratch blocks to the
    // pool and bump both generations, so any view that escaped this executor
    // asserts at its access site (I-116). No retention — the next executor on
    // this slot re-acquires lazily. performElem2 is single-exit, so this runs
    // on every path.
    releaseSlotScratch(scratchArenas(), genScratchArenas(), ruleStagings(), coreId);
}

/// @brief Whether an LB is a straggler that should split into `cores`
        ///        expression buckets next iteration — the split TRIGGER.
        ///
        /// @details The idle-core fair-share test. `totalWork / cores` is the
        /// ideally-balanced per-core load for this iteration's active main-path LBs;
        /// an LB whose own work exceeds it is, alone, a makespan bottleneck that
        /// leaves cores idle, so parallelising it across `cores` is justified. The
        /// second clause is the setup break-even (`minSplitWork`): below it a
        /// bucket's share of the work is smaller than the fixed per-bucket setup, so
        /// splitting provably cannot help — this suppresses splitting when the whole
        /// iteration is trivially cheap. Fair-share is masking-resistant (one
        /// straggler lifts the bar by only `work / cores`) and self-limiting (at
        /// most `cores - 1` LBs can exceed it -> "a few"). Integer arithmetic only,
        /// so the verdict is a deterministic function of the (deterministic) work
        /// totals — the property that keeps two runs byte-identical. Pure function of
        /// its arguments; `static` so it is unit-testable without an analyzer.
        ///
        /// @param work         This LB's total work this iteration = sum of its
        ///                     parts' submatch counts (split-invariant, D-117; `>= 0`).
        /// @param totalWork    Sum of `work` over all active main-path LBs (`>= 0`).
        /// @param cores        `logicalCores`, the split fan-out (`>= 1`).
        /// @param minSplitWork `parameters.min_split_work`, the setup break-even
        ///                     floor (`>= 0`).
        /// @return `true` iff this LB should run as `cores` expression buckets next
        ///         iteration; `false` to run it unsplit.
        /// @see `proveKernel` — the end-of-iteration stats pass that calls this.
        /// @see `Memory::numberOfParts` — the per-LB state this drives (1 or `cores`).
bool Phase2Executor::isStraggler(int64_t work, int64_t totalWork,
    int cores, int64_t minSplitWork) {
    assert(cores >= 1 && "isStraggler: cores is the machine core count (>= 1)");
    // Idle-core fair-share: an LB whose work alone exceeds the ideally-balanced
    // per-core load leaves cores idle, so splitting it across cores is justified.
    // The floor suppresses splitting a trivially cheap iteration (per-bucket work
    // below the fixed setup cost). Integer division -> deterministic verdict.
    const int64_t fairShare = totalWork / static_cast<int64_t>(cores);
    return work > fairShare && work >= minSplitWork;
}

/// @brief Phase 2 finalize — merge one LB's flat executor parts and drain
        ///        the staged buffers.
        ///
        /// @details
        /// The executors already ran in `proveKernel`'s **flat** (LB, part)
        /// work-stealing pool, each read-only on the LB
        /// ([D-116]) and appending its records to its own sealed page set;
        /// `proveKernel` hands this LB's sealed part sets in part order and
        /// calls this once per LB after the pool join. Reads `isActive` (the
        /// phase-1 absorb may have discharged the LB, in which case it generated
        /// no executor tasks and `partCount` is 0): if active, clears the
        /// per-step delta containers once and merges via the canonical-sorted
        /// `applyFiringRecords` (partition- and thread-order-independent,
        /// [D-117]). Always drains the staged
        /// algebra-admission and integration-prep buffers
        /// (`drainAdmissionKeysAlgebra` / `drainDeferredIntegrationPreps`) before
        /// phase 3's post-burst absorb. Single-LB — `proveKernel` runs it as an
        /// across-LB sweep (I-28-safe), so it may freely mutate `body`. The split
        /// decision (which LBs are stragglers next iteration) lives in
        /// `proveKernel`'s end-of-iteration stats pass via `isStraggler`, NOT here;
        /// every burst runs to completion, so `performElemPhase2` always applies a
        /// COMPLETE burst (D-201).
        ///
        /// @param body      The LB being finalized.
        /// @param parts     This LB's sealed part sets in part order; the
        ///                  records ride each set's chain (consumed / index-
        ///                  sorted by `applyFiringRecords`).
        /// @param partCount Number of part sets (`0` for a phase-1-discharged
        ///                  LB — nothing to merge, drains still run).
        /// @param doomLine  The LB's final packed doom line (`kNoDoomLine` when
        ///                  no doom trigger fired this burst). When set, the
        ///                  merge takes ONLY the winning part's chain — the part
        ///                  named by `doomLineOrdinal` — and the other parts'
        ///                  records are discarded unread (sound: the trigger
        ///                  guarantees phase 3 discharges the LB, so sibling
        ///                  records are wasted-once-doomed).
        /// @param firingRecordsCanonical True only for the CUDA adapter's one
        ///                  already-sorted, already-doom-compacted chain. In
        ///                  that route `partCount` is one and `doomLine` is
        ///                  `kNoDoomLine`; processor executor chains pass false.
        /// @return (void)
        /// @see `performElem2` — the per-part executor that ran in the flat pool.
        /// @see `applyFiringRecords` — the canonical-order deposit merge.
        /// @see `isStraggler` — the split trigger (applied in `proveKernel`).
void Phase2Executor::performElemPhase2(Memory& body,
    SealedPageSet* const* parts, int32_t partCount,
    int64_t doomLine,
    bool firingRecordsCanonical) {
    gl::Phase2TraceScope phase2FinalizeTrace("GL.Phase2.Finalize", partCount);
    // Per-LB finalize of the FLAT executor pool: apply this LB's captured firing
    // records and drain the per-burst admission / integration staging. The
    // executors already ran in proveKernel's flat (LB, part) sweep, each read-only
    // on its LB (D-116) and appending its records to its own sealed page set;
    // `parts` hands THIS LB's sealed part sets in part order. This step is
    // single-LB and runs after the pool join, so it may freely mutate the LB. The
    // split decision (which LBs are stragglers next iteration) lives in proveKernel's
    // end-of-iteration stats pass via isStraggler, not here.

    // ENTRANCE isActive gate. performElemPhase1's absorb may have discharged the
    // LB (primed/CE contradiction, vacuous-truth induction step, toBeProved
    // match), flipping body.isActive=false; such an LB generated no executor
    // tasks, so partCount is 0 and the merge is skipped. The drains below
    // still run so any pre-deactivation staging is replayed.
    if (body.isActive) {
        // Per-step delta clears — ONCE per LB, before the merge (kept here, not
        // in performElem2, so a later part's request generation still sees the
        // deltas; D-126). Pre-merge, so admissionKeysAlgebra
        // / deferredIntegrationPreps are reset before applyFiringRecords fills
        // them.
        body.intLocalEncodedStatementsDelta.clear();
        body.localHashMemoryDelta.resetToFresh();
        body.admissionKeysAlgebra.clear();
        body.deferredIntegrationPreps.clear();
        body.admissionKeysOrdis2.clear();

        // Merge the parts' captured deposits in canonical sorted order —
        // partition-independent (D-117): the same
        // firing SET sorts identically regardless of how it was split or in what
        // thread order the parts completed. This is why flat parallel execution
        // stays deterministic.
        //
        // Doom line set: a doom trigger fired this burst, so the merge takes
        // ONLY the winning part's chain (the lexicographic-minimum
        // (position, ordinal) trigger — a pure function of the parts'
        // deterministic streams). The other parts' chains are discarded unread:
        // the trigger guarantees phase 3 discharges the LB, so sibling records
        // are wasted-once-doomed and dropping them keeps the merged set
        // deterministic (their tails end wherever the line caught them). An
        // unsplit LB is its own winner, so its early-exit merge is unchanged.
        if (firingRecordsCanonical) {
            assert(partCount == 1
                && "canonical GPU output is one compacted chain per LB");
            assert(doomLine == kNoDoomLine
                && "canonical GPU output already applied doom selection");
            ruleEvaluator.applyFiringRecords(body, parts, partCount, true);
        } else if (doomLine != kNoDoomLine) {
            assert(partCount > 0
                && "performElemPhase2: a doom trigger implies a fired part");
            const int32_t w = doomLineOrdinal(doomLine);
            assert(w < partCount
                && "performElemPhase2: doom-line winner outside this LB's parts");
            SealedPageSet* const winner[1] = { parts[w] };
            ruleEvaluator.applyFiringRecords(body, winner, 1, false);
        } else {
            ruleEvaluator.applyFiringRecords(body, parts, partCount, false);
        }
    }

    // Quiescence (D-194): admission-map churn this burst is a
    // mutation the phase-3 statement-count diff cannot see — a marker firing
    // registers an admission template without depositing a statement, yet it can
    // enable a fresh it_/int_ admission (and thus a firing) next burst. Flag it
    // so the LB stays awake. Read while the staging vectors still hold this
    // burst's records, before the drains below consume them.
    if (!body.admissionKeysAlgebra.empty() || !body.deferredIntegrationPreps.empty()
        || !body.admissionKeysOrdis2.empty())
        body.mutatedThisBurst = true;

    // Drain the per-burst admission / integration / demand records (replay
    // in firing order before phase 3's post-burst standardProcessing
    // absorb). Run for every LB, active or discharged in phase 1.
    admissionMemory.drainAdmissionKeysAlgebra(body);
    admissionMemory.drainDeferredIntegrationPreps(body);
    admissionMemory.drainAdmissionKeysOrdis2(body);

    // The drained staging vectors carried sealed views into the tasks' page
    // sets; clear them NOW, before the post-join sweep frees the pages, so
    // no dangling view ever sits in a Memory container — and no staging can
    // leak across iterations into a later drain whose pages are long freed.
    // (The pre-merge clears above remain: they cover the inactive path's
    // bookkeeping and the D-126 delta contract.)
    body.admissionKeysAlgebra.clear();
    body.deferredIntegrationPreps.clear();
    body.admissionKeysOrdis2.clear();

}

/// @brief Record the CUDA Phase 2 read-image measurement of one claimed,
///        resident LB at the end of its phase 1.
///
/// @param body The active LB whose phase 1 just finished.
void Phase2Executor::recordPhase2Measurement(Memory& body) const {
    assert(phase2MeasurementEpoch != 0
        && "recordPhase2Measurement before proveKernel started a burst");
    body.phase2TaskUsage =
        gpu::measurePhase2TaskProjectionUsage(body, 0, ceFilteringActive);
    if (body.phase2TaskUsage.tasks != 0) {
        body.phase2SelectedHashMemories =
            gpu::selectPhase2HashMemories(body, ceFilteringActive);
        assert(body.phase2SelectedHashMemories != 0);
        body.phase2ProjectionUsage = gpu::measurePhase2ProjectionUsage(
            body, anchorInfo, body.phase2SelectedHashMemories, true);
    } else {
        body.phase2SelectedHashMemories = 0;
        body.phase2ProjectionUsage = gpu::Phase2ProjectionUsage{};
    }
    body.phase2MeasurementEpoch = phase2MeasurementEpoch;
}
/// @brief Execute and finalize the current burst's phase-two work.
/// @details Runs the existing flat worker rounds, CUDA projection, sealed-record merge and split measurements. The caller retains the trace lifetime and joins phase one before this call.
/// @param active Logical blocks selected at the burst boundary.
/// @param workers Existing worker count.
/// @param phase2TraceCall Existing trace sequence identifier.
/// @param phase2Started Start time captured after phase one.
/// @param phase2Trace Caller-owned trace finished at the boundary.
/// @param trapPh2Ns Per-block phase-two timings when runtime measurement is enabled.
/// @param trapFinNs Per-block finalization timings when runtime measurement is enabled.
/// @param trapWork Per-block work totals when runtime measurement is enabled.
/// @return Existing main-path split eligibility sampled at the original phase-two point.
/// @invariant All executor and finalizer workers join before returning.
bool Phase2Executor::execute(const std::vector<Memory*>& active, unsigned workers, uint64_t phase2TraceCall, std::chrono::steady_clock::time_point phase2Started, Phase2TraceScope& phase2Trace
#if RT_MEASUREMENT
, std::vector<int64_t>& trapPh2Ns, std::vector<int64_t>& trapFinNs, std::vector<int64_t>& trapWork
#endif
) {
    if (gl::phase2TraceEnabled()) {
        diagnosticsLog() << "[PHASE2-IDENTITY] call=" << phase2TraceCall
            << " anchor=" << anchorID_ << " ce_filter=" << ceFilteringActive
            << " measured_iteration=" << phase2MeasuredIterations + 1 << '\n';
        for (std::size_t li = 0; li < active.size(); ++li) {
            diagnosticsLog() << "[PHASE2-LB] call=" << phase2TraceCall
                << " li=" << li << " chain=";
            for (const Memory* p = active[li]; p != nullptr; p = p->parentMemory) {
                if (p != active[li]) diagnosticsLog() << " <- ";
                const std::string key = p->exprKey();
                diagnosticsLog() << (key.empty() ? "(root)" : key);
            }
            diagnosticsLog() << '\n';
        }
    }


    // Phase 2 opens TWO working-set windows PER PASS
    // (D-196): an EXECUTOR window over the real dispatch
    // atomic (`next` over the flat `execOrder` task list — duplicates for
    // split parts are fine: the enqueue dedup and the claim CAS collapse
    // them, and split LBs are ineligible victims anyway), then, after the
    // executor join, a FINALIZE window over `nextLi` / `toRun`. This retires
    // the frozen `phase2Cursor` (a completion cursor stored only by the
    // finalize, which sat at 0 through the whole hashburst — the steward was
    // blind in the heaviest phase and every phase-2 reload ran inline on
    // workers). Redo passes open their own windows over their own vectors,
    // so indices are always in the right space. The unified handshake uses
    // the single stewardClaim word — the separate burst-claim word and the
    // mid-burst throttle gate are retired (the pager subsumes them).

    // ---- Phase 2: FLAT executor pool + per-LB finalize, EXPRESSION-BUCKET split ----
    // FLAT parallelism, never nested: one work-stealing pool over the flat list of
    // (LB, part) executor tasks across the LBs of this pass. No LB spawns
    // sub-threads, so the cores are never oversubscribed by nesting. With the
    // read-only fixpoint (D-116), distinct tasks on one LB only READ it (the shared
    // NameMap is read via lookup) and each writes its OWN sealed page set's record
    // chain, so same-LB tasks never race; tasks on different LBs are independent.
    //
    // The split is decided BEFORE the iteration (the end-of-iteration stats pass
    // below, D-201): a straggler (numberOfParts > 1) carries
    // logicalCores expression buckets. No cap, no escalation, no discarded burst.
    // The pass loop is at most TWO rounds (I-75):
    //
    //   round 1: a straggler dispatches one produceOnly PRODUCER (produceExpressionStumps,
    //            whole-LB, fires nothing); every other LB dispatches one unsplit burst
    //   round 2: the producer's regular and terminal pre-stumps are dealt into
    //            buckets, one bucket part each
    //
    // Requeued work runs in the NEXT round, never appended to the running one: the
    // working-set pager has registered this round's task list and dispatch cursor, so
    // the task vector cannot grow under it. An LB is finalised ONCE, in the round after
    // which it has no tasks left. The merge is partition-independent (applyFiringRecords
    // sorts, D-117 / I-77), so which parts came from which round cannot matter.
    //
    // disable_lb_split and the incubator run UNSPLIT; the CE filter is unaffected
    // (its own single-part loop in filter.cpp).
    struct ExecTask {
        Memory* lb;
        std::size_t li;          // index into `active`
        int partCount;           // concurrent parts of this LB this burst (= bucket count)
        bool produceOnly;        // a round-1 stump PRODUCER (runs produceExpressionStumps,
                                 // not performElem2); its buckets requeue for round 2
        SplitStumpRef stump;     // empty for an unsplit part; set for a bucket part
        std::atomic<int64_t>* doomLine;  // the LB's shared packed (position, ordinal) stop line
        std::atomic<int>* partsLeft;
    };
    const bool mainPath = parameters.lb_split && !parameters.disable_lb_split
        && !ceFilteringActive;
    // A straggler splits into logicalCores expression buckets; that is the only
    // split dimension now, so the whole-machine core count is the per-LB part
    // ceiling (Rule 19 -- a machine past the named constant stops HERE).
    assert(static_cast<int>(logicalCores) <= gl::HashRequestGenerator::kMaxSplitParts
        && "logicalCores exceeds kMaxSplitParts - raise the named constant "
           "deliberately for a machine with more cores than the ceiling");

    if (!active.empty()) {
    const std::size_t M = active.size();
    // Per-LB phase-2 early-exit doom lines, EXTERNAL to the LB so the hashburst
    // stays strictly read-only on it (I-66). One packed (position, ordinal)
    // atomic per LB, shared by all its parts and lowered only by CAS-min; sized
    // once so the &doomLines[li] handed to tasks stay stable across passes
    // (the vector is never resized; std::atomic is not movable).
    std::vector<std::atomic<int64_t>> doomLines(M);
    // Per-LB remaining-parts counter (working-set pager): each executor part
    // decrements it after sealing, and the part that drops it to zero releases the
    // LB's claim to Idle -- so an executor-done LB becomes deloadable immediately,
    // bounding the phase-2 resident set to the working set (I-114). Re-armed per pass.
    std::vector<std::atomic<int>> partsRemaining(M);

    // Sealed record sets accumulate ACROSS this iteration's rounds: a non-straggler's
    // round-1 burst is kept while a straggler's buckets run in round 2. Deque:
    // SealedPageSet is non-movable, so the views' owner pointers stay stable (D-164).
    std::deque<SealedPageSet> pageStore;
    std::vector<std::vector<SealedPageSet*>> keptParts(M);
    // Per-LB total work this iteration = SUM of its parts' submatch counts
    // (split-invariant, D-117) -- the straggler classifier's input (the end-of-
    // iteration stats pass below). lbMaxSub is the busiest single part, kept only
    // for the split-ineffective self-control report.
    std::vector<int64_t> lbTotalSub(M, 0);
    std::vector<int64_t> lbMaxSub(M, 0);
    // Doom-line latch: 1 when any of this LB's parts hit a burst early-exit
    // (a burstDeactivates trigger) in ANY pass this iteration. An early-exited
    // part stops at a timing-dependent overshoot past the published line in
    // its own stream (I-191 protects the merged CONTENT, not the tallies), so
    // a doomed LB's submatch tally is not a deterministic work figure — the
    // straggler statistic below excludes it. Whether a doom trigger fires at
    // all IS deterministic (the winning line's position is always reached by
    // its part), so the exclusion keeps the split set a pure function of
    // proof state (I-160). Latched after each pass's join because the
    // per-pass doomLines reset would erase a pass-1 doom.
    std::vector<char> lbDoomed(M, 0);
    // Diagnostics for the split-ineffective report: how many stump work items the
    // producer returned and how many buckets it dealt them into (1 = not really split).
    std::vector<int32_t> lbStumps(M, 0);
    std::vector<int32_t> lbBuckets(M, 1);
    std::vector<char> lbFinalized(M, 0);
    std::vector<char> lbFiringRecordsCanonical(M, 0);
    // Device-projection capacity census. The first executor task that claims an
    // LB records its post-phase-1 read image once; split bucket siblings and a
    // second pass see the same read-only LB and are collapsed by projectionCounted.
    // Atomics make the observation race-free; addition and maximum are independent
    // of which sibling wins the exchange. This telemetry never gates proof flow.
    std::vector<std::atomic<uint8_t>> projectionCounted(M);
    for (std::atomic<uint8_t>& counted : projectionCounted)
        counted.store(0, std::memory_order_relaxed);
    std::atomic<uint64_t> projectionLogicalBlocks{ 0 };
    std::atomic<uint64_t> projectionStatements{ 0 };
    std::atomic<uint64_t> projectionMaximumStatements{ 0 };
    std::atomic<uint64_t> projectionNameRecords{ 0 };
    std::atomic<uint64_t> projectionMaximumNameRecords{ 0 };
    std::atomic<uint64_t> projectionNameBytes{ 0 };
    std::atomic<uint64_t> projectionMaximumNameBytes{ 0 };
    constexpr std::size_t kProjectionSemanticMetricCount = 20;
    std::array<std::atomic<uint64_t>,
               kProjectionSemanticMetricCount> projectionSemanticTotals;
    std::array<std::atomic<uint64_t>,
               kProjectionSemanticMetricCount> projectionSemanticMaximums;
    for (std::size_t metric = 0;
         metric < kProjectionSemanticMetricCount; ++metric) {
        projectionSemanticTotals[metric].store(0, std::memory_order_relaxed);
        projectionSemanticMaximums[metric].store(0, std::memory_order_relaxed);
    }
    constexpr std::size_t kTaskProjectionMetricCount = 4;
    std::array<std::atomic<uint64_t>,
               kTaskProjectionMetricCount> taskProjectionTotals;
    std::array<std::atomic<uint64_t>,
               kTaskProjectionMetricCount> taskProjectionMaximums;
    for (std::size_t metric = 0;
         metric < kTaskProjectionMetricCount; ++metric) {
        taskProjectionTotals[metric].store(0, std::memory_order_relaxed);
        taskProjectionMaximums[metric].store(0, std::memory_order_relaxed);
    }
    requestGenerator.resetFilterCounters();





    constexpr std::size_t kGpuGrowDepthCount =
        ExecutionParameters::MAX_EXPRESSIONS + 1;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuGrowAttemptsByDepth;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuGrowFrontierByDepth;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuGrowSubkeysByDepth;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuGrowRequestsByDepth;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuProducerAttemptsByDepth;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuProducerSurvivorsByDepth;
    for (std::size_t depth = 0; depth < kGpuGrowDepthCount; ++depth) {
        gpuGrowAttemptsByDepth[depth].store(0, std::memory_order_relaxed);
        gpuGrowFrontierByDepth[depth].store(0, std::memory_order_relaxed);
        gpuGrowSubkeysByDepth[depth].store(0, std::memory_order_relaxed);
        gpuGrowRequestsByDepth[depth].store(0, std::memory_order_relaxed);
        gpuProducerAttemptsByDepth[depth].store(0, std::memory_order_relaxed);
        gpuProducerSurvivorsByDepth[depth].store(0, std::memory_order_relaxed);
    }
    constexpr std::size_t kGpuEvaluationTotalCount = 15;
    constexpr std::size_t kGpuEvaluationMaximumCount = 3;
    std::array<std::atomic<uint64_t>, kGpuEvaluationTotalCount>
        gpuEvaluationTotals;
    std::array<std::atomic<uint64_t>, kGpuEvaluationMaximumCount>
        gpuEvaluationMaximums;
    for (std::atomic<uint64_t>& total : gpuEvaluationTotals)
        total.store(0, std::memory_order_relaxed);
    for (std::atomic<uint64_t>& maximum : gpuEvaluationMaximums)
        maximum.store(0, std::memory_order_relaxed);
    // The stumps a producer task returns, copied off its sealed pages so those pages
    // go back to the pool at once. The bucket tasks point into these runs, so the
    // storage must outlive the pass: deque, never reallocated.
    std::deque<std::vector<ExpressionStump>> stumpStore;

    // Task build. A straggler (numberOfParts > 1, set last iteration by the stats
    // pass below) dispatches ONE producer task that enumerates the LB's expression
    // stumps; its buckets run in round 2. Every other LB dispatches ONE unsplit part.
    std::vector<ExecTask> tasks;
    for (std::size_t li = 0; li < M; ++li) {
        Memory* b = active[li];
        if (!b->isActive) continue;  // discharged in phase 1 -> no executor tasks
        // Split preemptively if the submatch stat flagged it last iteration
        // (numberOfParts > 1) OR it just activated this iteration (justActivated,
        // one-shot: no prior burst for the stat to see). Consume the flag here.
        const bool straggler = mainPath && (b->numberOfParts > 1 || b->justActivated);
        b->justActivated = false;
        tasks.push_back(ExecTask{ b, li,
            /*partCount=*/1, /*produceOnly=*/straggler, SplitStumpRef{},
            &doomLines[li], &partsRemaining[li] });
    }

    int passNo = 0;
    for (;;) {
        ++passNo;
        gl::Phase2TraceScope phase2PassTrace("GL.Phase2.Pass", passNo);
        if (gl::phase2TraceEnabled()) {
            for (std::size_t task = 0; task < tasks.size(); ++task)
                diagnosticsLog() << "[PHASE2-TASK] call=" << phase2TraceCall
                    << " pass=" << passNo << " task=" << task << " li=" << tasks[task].li
                    << " parts=" << tasks[task].partCount
                    << " producer=" << tasks[task].produceOnly << '\n';
        }

        // Round 1: producers + unsplit bursts. Round 2 (processor route): the
        // producers' buckets; the CUDA route runs them in round 1.
        assert(passNo <= 2
            && "phase-2 pass loop exceeded two rounds - a bucket part requeued");
        assert((passNo == 1 || phase2Backend == Phase2Backend::cpu)
            && "the CUDA route runs its producers' buckets in round 1");

        for (std::size_t li = 0; li < M; ++li) {
            doomLines[li].store(kNoDoomLine, std::memory_order_relaxed);
            partsRemaining[li].store(0, std::memory_order_relaxed);
        }
        for (const ExecTask& t : tasks) {
            if (phase2Backend == Phase2Backend::cpu || t.produceOnly)
                partsRemaining[t.li].fetch_add(1, std::memory_order_relaxed);
        }

        // This pass's page sets append to the store; `base` is where they start.
        const std::size_t base = pageStore.size();
        for (std::size_t i = 0; i < tasks.size(); ++i) pageStore.emplace_back();
        // A second set per task, for the stumps a round-1 producer returns: the
        // record chain is single-type per set (I-135) and the firing records already
        // own the first.
        std::deque<SealedPageSet> stumpPages(tasks.size());
        std::vector<char> stumpBound(tasks.size(), 0);
        std::vector<int32_t> taskStumps(tasks.size(), 0);
        // Per-(LB, part) submatch tally: each worker reads gl::HashRequestGenerator::growthMatchCount() right
        // after its performElem2 returns (the submatches this part visited). See
        // D-109.
        std::vector<int64_t> taskSubMatches(tasks.size(), 0);
        // CUDA sealing emits one already-canonical chain per projected LB. The
        // first task for that LB owns the page set; sibling tasks still retain
        // their independent split-work counts but do not own duplicate chains.
        GpuHostArena gpuDirectoryStorage;
        gpu::ProjectionColumn<int32_t> gpuTaskIndices;
        gpu::ProjectionColumn<char> gpuOutputOwners;
#if RT_MEASUREMENT
        // [RT phase measurement] per-task compute wall time (the
        // produce/burst call only; claim/load IO excluded).
        std::vector<int64_t> taskNs(tasks.size(), 0);
#endif
        // Flat executor order for this pass's pager window: one entry per task.
        std::vector<Memory*> execOrder;
        execOrder.reserve(tasks.size());
        for (const ExecTask& t : tasks) execOrder.push_back(t.lb);

        if (!tasks.empty()) {
            std::atomic<std::size_t> next{ 0 };
            // EXECUTOR window: the pool's real dispatch atomic is the pager cursor,
            // so the steward prefetches upcoming tasks' LBs and evicts behind the
            // executor sweep in real time.
            gl::memory_census::markPhase(gl::PoolPhase::Phase2Execute);
            steward->beginPhaseWindow(/*phase=*/2, &next, &execOrder, workers,
                                      lbdeload::kDeloadDirectory);
            auto worker = [this, &tasks, &pageStore, base, &stumpPages, &stumpBound,
                            &taskStumps, &taskSubMatches,
                            &projectionCounted, &projectionLogicalBlocks,
                            &projectionStatements, &projectionMaximumStatements,
                            &projectionNameRecords, &projectionMaximumNameRecords,
                            &projectionNameBytes, &projectionMaximumNameBytes,
                            &projectionSemanticTotals,
                            &projectionSemanticMaximums,
                            &taskProjectionTotals,
                            &taskProjectionMaximums,
                            &gpuGrowAttemptsByDepth,
                            &gpuGrowFrontierByDepth,
                            &gpuGrowSubkeysByDepth,
                            &gpuGrowRequestsByDepth,
                            &gpuProducerAttemptsByDepth,
                            &gpuProducerSurvivorsByDepth,
                            &gpuEvaluationTotals,
                            &gpuEvaluationMaximums,
#if RT_MEASUREMENT
                           &taskNs,
#endif
                           &next, workers,
                           mainPath, phase2TraceCall](unsigned coreId) {
                const unsigned cid = workers ? (coreId % workers) : 0U;
                // The phase-2 executor pool runs in parallel like phases 1/3, so
                // publish this worker's slot. Every gl::WorkerContext::coreId()-resolved per-slot
                // arena consumer reached from performElem2 must pick THIS worker's
                // slot, not the single reserved slot every unpublished worker would
                // share (that slot is single-threaded-only).
                gl::WorkerContext::setCoreId(static_cast<int>(cid));
                gl::Phase2TraceScope phase2WorkerTrace("GL.Phase2.CpuWorker", phase2TraceCall, gl::phase2TraceEnabled(), true);
                for (;;) {
                    std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= tasks.size()) break;
                    const ExecTask& t = tasks[i];
                    if (this->phase2Backend == Phase2Backend::cuda
                        && !t.produceOnly) continue;
                    gl::Phase2TraceScope phase2CpuTaskTrace("GL.Phase2.CpuTask", i);
                    // Unified working-set handshake: claim this LB so the steward
                    // will not deload it under us, reloading if it is cold. A split
                    // sibling that already owns it returns immediately (resident).
                    steward->claimAndLoadForWork(*t.lb, /*phase=*/2,
                                                 lbdeload::kDeloadDirectory);
                    if (projectionCounted[t.li].exchange(
                            1, std::memory_order_relaxed) == 0) {
                        const gpu::Phase2ProjectionUsage projectionUsage =
                            gpu::measurePhase2ProjectionUsage(*t.lb, anchorInfo, 0x0fu, true);
                        const uint64_t statementCount =
                            projectionUsage.statements;
                        const uint64_t nameRecordCount =
                            projectionUsage.nameRecords;
                        const uint64_t nameByteCount = projectionUsage.nameBytes;
                        projectionLogicalBlocks.fetch_add(
                            1, std::memory_order_relaxed);
                        projectionStatements.fetch_add(
                            statementCount, std::memory_order_relaxed);
                        projectionNameRecords.fetch_add(
                            nameRecordCount, std::memory_order_relaxed);
                        projectionNameBytes.fetch_add(
                            nameByteCount, std::memory_order_relaxed);
                        auto publishMaximum = [](std::atomic<uint64_t>& maximum,
                                                 uint64_t value) {
                            uint64_t observed = maximum.load(
                                std::memory_order_relaxed);
                            while (observed < value
                                && !maximum.compare_exchange_weak(
                                    observed, value,
                                    std::memory_order_relaxed,
                                    std::memory_order_relaxed)) {
                            }
                        };
                        publishMaximum(
                            projectionMaximumStatements, statementCount);
                        publishMaximum(
                            projectionMaximumNameRecords, nameRecordCount);
                        publishMaximum(
                            projectionMaximumNameBytes, nameByteCount);
                        const uint64_t semanticValues[
                            kProjectionSemanticMetricCount] = {
                            projectionUsage.ruleStringRecords,
                            projectionUsage.ruleStringBytes,
                            projectionUsage.nameSlots,
                            projectionUsage.byteMapViews,
                            projectionUsage.byteMapEntries,
                            projectionUsage.byteMapSlots,
                            projectionUsage.byteKeyBytes,
                            projectionUsage.blobRecords,
                            projectionUsage.blobBytes,
                            projectionUsage.reverseMapViews,
                            projectionUsage.reverseMapEntries,
                            projectionUsage.reverseMapSlots,
                            projectionUsage.reverseKeyBytes,
                            projectionUsage.reverseOwners,
                            projectionUsage.podMapViews,
                            projectionUsage.podMapEntries,
                            projectionUsage.podMapSlots,
                            projectionUsage.podRunValues,
                            projectionUsage.mandatoryStatementKeys,
                            projectionUsage.metadataBytes
                        };
                        for (std::size_t metric = 0;
                             metric < kProjectionSemanticMetricCount; ++metric) {
                            projectionSemanticTotals[metric].fetch_add(
                                semanticValues[metric],
                                std::memory_order_relaxed);
                            publishMaximum(
                                projectionSemanticMaximums[metric],
                                semanticValues[metric]);
                        }
                    }
                    // The task's exclusive write window on its page set opens here
                    // and closes at the seal below -- the records' strings then cross
                    // the pool join read-only.
                    SealedPageSet& ps = pageStore[base + i];
                    ps.bind(&staticMemory());
#if RT_MEASUREMENT
                    // [RT phase measurement] compute wall time of this
                    // task; index i belongs to exactly this worker (race-free).
                    const auto trapT0 = std::chrono::steady_clock::now();
#endif
                    for (std::size_t depth = 0;
                         depth < kGpuGrowDepthCount; ++depth) {
                        gl::HashRequestGenerator::resetDepthCounters(depth);





                    }
                    gl::HashRuleEvaluator::resetEvaluationUsage();
                    if (t.produceOnly) {
                        // Round-1 stump PRODUCER: enumerate the whole LB's
                        // expression stumps, retaining terminal pre-stumps for
                        // recordable nodes replaced by a deeper level. It fires
                        // nothing and deposits nothing (ps stays empty); the
                        // classify deals all work items into buckets that run in
                        // round 2. Grow MORE stumps than buckets (a small multiple
                        // of logicalCores) so the round-robin deal evens out the
                        // buckets' grow-tree sizes.
                        ps.seal();
                        stumpPages[i].bind(&staticMemory());
                        stumpBound[i] = 1;
                        const int32_t stumpTarget = static_cast<int32_t>(
                            logicalCores) * kStumpsPerBucketTarget;
                        taskStumps[i] = requestGenerator.produceExpressionStumps(
                            *t.lb, cid, stumpTarget, stumpPages[i]);
                        stumpPages[i].seal();
                        taskSubMatches[i] = 0;  // the producer does no firing work
                        // The producer's level column held this slot's gen arena;
                        // the stumps were copied onto the sealed pages, so nothing
                        // points into it — hand the blocks back now.
                        assert(ruleStagings().slotIsEmpty(cid)
                            && "producer exit: staged rule-index writes outlive their window");
                        genScratchArenas().forSlot(cid).releaseAll();
                    } else {
                        // A burst part: an unsplit LB, or one expression bucket of a
                        // straggler. Runs to completion unless the LB's doom line
                        // stops it at a deterministic stream position.
                        const gpu::Phase2TaskProjectionUsage taskUsage =
                            gpu::measurePhase2TaskProjectionUsage(
                                *t.lb,
                                static_cast<uint32_t>(t.stump.count),
                                this->ceFilteringActive);
                        const uint64_t taskValues[kTaskProjectionMetricCount] = {
                            taskUsage.tasks, taskUsage.batches,
                            taskUsage.terms, taskUsage.stumps
                        };
                        for (std::size_t metric = 0;
                             metric < kTaskProjectionMetricCount; ++metric) {
                            taskProjectionTotals[metric].fetch_add(
                                taskValues[metric], std::memory_order_relaxed);
                            uint64_t observed = taskProjectionMaximums[metric].load(
                                std::memory_order_relaxed);
                            while (observed < taskValues[metric]
                                && !taskProjectionMaximums[metric].compare_exchange_weak(
                                    observed, taskValues[metric],
                                    std::memory_order_relaxed,
                                    std::memory_order_relaxed)) {
                            }
                        }
                        performElem2(*t.lb, cid, t.partCount, t.stump, ps,
                                           *t.doomLine);
                        ps.seal();
                        taskSubMatches[i] = gl::HashRequestGenerator::growthMatchCount();
                    }
                    for (std::size_t depth = 0;
                         depth < kGpuGrowDepthCount; ++depth) {
                        gpuGrowAttemptsByDepth[depth].fetch_add(
                            gl::HashRequestGenerator::gpuGrowAttemptsByDepth(depth),
                            std::memory_order_relaxed);
                        gpuGrowFrontierByDepth[depth].fetch_add(
                            gl::HashRequestGenerator::gpuGrowFrontierByDepth(depth),
                            std::memory_order_relaxed);
                        gpuGrowSubkeysByDepth[depth].fetch_add(
                            gl::HashRequestGenerator::gpuGrowSubkeysByDepth(depth),
                            std::memory_order_relaxed);
                        gpuGrowRequestsByDepth[depth].fetch_add(
                            gl::HashRequestGenerator::gpuGrowRequestsByDepth(depth),
                            std::memory_order_relaxed);
                        gpuProducerAttemptsByDepth[depth].fetch_add(
                            gl::HashRequestGenerator::gpuProducerAttemptsByDepth(depth),
                            std::memory_order_relaxed);
                        gpuProducerSurvivorsByDepth[depth].fetch_add(
                            gl::HashRequestGenerator::gpuProducerSurvivorsByDepth(depth),
                            std::memory_order_relaxed);
                    }
                    const uint64_t evaluationTotals[
                        kGpuEvaluationTotalCount] = {
                        gl::HashRuleEvaluator::evaluationUsage().requests,
                        gl::HashRuleEvaluator::evaluationUsage().dependencyPassRequests,
                        gl::HashRuleEvaluator::evaluationUsage().reverseOwners,
                        gl::HashRuleEvaluator::evaluationUsage().candidateOwners,
                        gl::HashRuleEvaluator::evaluationUsage().encodedHits,
                        gl::HashRuleEvaluator::evaluationUsage().localValues,
                        gl::HashRuleEvaluator::evaluationUsage().headRecords,
                        gl::HashRuleEvaluator::evaluationUsage().markerRecords,
                        gl::HashRuleEvaluator::evaluationUsage().demandRecords,
                        gl::HashRuleEvaluator::evaluationUsage().generatedBytes,
                        gl::HashRuleEvaluator::evaluationUsage().levelValues,
                        gl::HashRuleEvaluator::evaluationUsage().originDependencies,
                        gl::HashRuleEvaluator::evaluationUsage().markerKeys,
                        gl::HashRuleEvaluator::evaluationUsage().markerRemainingArgs,
                        gl::HashRuleEvaluator::evaluationUsage().markerArgs
                    };
                    for (std::size_t metric = 0;
                         metric < kGpuEvaluationTotalCount; ++metric) {
                        gpuEvaluationTotals[metric].fetch_add(
                            evaluationTotals[metric],
                            std::memory_order_relaxed);
                    }
                    const uint64_t evaluationMaximums[
                        kGpuEvaluationMaximumCount] = {
                        gl::HashRuleEvaluator::evaluationUsage().maximumReverseOwnersPerRequest,
                        gl::HashRuleEvaluator::evaluationUsage().maximumCandidateOwnersPerRequest,
                        gl::HashRuleEvaluator::evaluationUsage().maximumLocalValuesPerHit
                    };
                    for (std::size_t metric = 0;
                         metric < kGpuEvaluationMaximumCount; ++metric) {
                        uint64_t observed = gpuEvaluationMaximums[metric].load(
                            std::memory_order_relaxed);
                        while (observed < evaluationMaximums[metric]
                            && !gpuEvaluationMaximums[metric].compare_exchange_weak(
                                observed, evaluationMaximums[metric],
                                std::memory_order_relaxed,
                                std::memory_order_relaxed)) {
                        }
                    }
#if RT_MEASUREMENT
                    // [RT phase measurement]
                    taskNs[i] = std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - trapT0).count();
#endif
                    // Last part of this LB to finish releases its claim to Idle (all
                    // parts have sealed -> no part still reads it), so the steward may
                    // now deload it. The finalize re-claims + reloads it.
                    if (t.partsLeft->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        assert(t.lb->stewardClaim.load(std::memory_order_relaxed)
                                   == static_cast<uint8_t>(
                                          Memory::StewardClaim::WorkerOwned)
                               && "phase-2 executor release of an LB not held "
                                  "WorkerOwned");
                        t.lb->stewardClaim.store(
                            static_cast<uint8_t>(Memory::StewardClaim::Idle),
                            std::memory_order_release);
                    }
                }
                };
            gl::Phase2TraceScope phase2CpuPoolTrace("GL.Phase2.CpuPool", passNo);
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (unsigned t = 0; t < workers; ++t) pool.emplace_back(worker, t);
            for (auto& th : pool) th.join();
            // Close the executor window at its join (its cursor and order die with
            // this block; queued stale tasks drop on the generation).
            steward->endPhaseWindow();
        }

        // Deal producer i's stumps into logicalCores expression buckets and append
        // one bucket part each to `parts`.
        const auto dealProducerStumps = [&](std::size_t i, std::vector<ExecTask>& parts) {
            assert(tasks[i].produceOnly);
            Memory* const lb = tasks[i].lb;
            const std::size_t li = tasks[i].li;
            if (taskStumps[i] == 0) {
                // No statement survives the filter -> the LB's real burst would
                // generate nothing either. Run it once, unsplit, this iteration
                // (guarantees the burst happens; its work then feeds the stats).
                parts.push_back(ExecTask{ lb, li,
                    /*partCount=*/1, /*produceOnly=*/false,
                    SplitStumpRef{}, &doomLines[li], &partsRemaining[li] });
                return;
            }
            stumpStore.emplace_back();
            std::vector<ExpressionStump>& run = stumpStore.back();
            run.reserve(static_cast<std::size_t>(taskStumps[i]));
            stumpPages[i].forEachRecord<ExpressionStump>(
                [&run](const ExpressionStump& s) { run.push_back(s); });
            assert(static_cast<int32_t>(run.size()) == taskStumps[i]
                && "stump record chain disagrees with the producer's count");

            // Deal the regular and terminal pre-stumps into BUCKETS, one part
            // per bucket. Round-robin, so
            // a bucket draws stumps from across the name-sorted list rather than
            // one contiguous slice; the stumps are permuted into bucket order so
            // each bucket is a contiguous run. Grown to a multiple of buckets
            // (kStumpsPerBucketTarget), so the deal evens out the buckets'
            // grow-tree sizes; fewer stumps than buckets means the LB could not
            // fan wider. The generator builds the filter once per bucket and the
            // emitter's seen-set collapses a request two of a bucket's stumps
            // both reach (I-158).
            const int32_t nStumps = static_cast<int32_t>(run.size());
            lbStumps[li] = nStumps;
            const NameId buckets = static_cast<NameId>(
                std::min<int32_t>(nStumps, static_cast<int32_t>(logicalCores)));
            lbBuckets[li] = buckets;
            std::vector<ExpressionStump> dealt;
            dealt.reserve(run.size());
            std::vector<int32_t> bucketStart(
                static_cast<std::size_t>(buckets) + 1, 0);
            for (NameId k = 0; k < buckets; ++k) {
                bucketStart[static_cast<std::size_t>(k)] =
                    static_cast<int32_t>(dealt.size());
                for (int32_t j = k; j < nStumps; j += buckets)
                    dealt.push_back(run[static_cast<std::size_t>(j)]);
            }
            bucketStart[static_cast<std::size_t>(buckets)] =
                static_cast<int32_t>(dealt.size());
            run.swap(dealt);
            for (NameId k = 0; k < buckets; ++k) {
                const int32_t lo = bucketStart[static_cast<std::size_t>(k)];
                const int32_t hi = bucketStart[static_cast<std::size_t>(k) + 1];
                parts.push_back(ExecTask{ lb, li,
                    /*partCount=*/buckets, /*produceOnly=*/false,
                    SplitStumpRef{ run.data() + lo,
                                   static_cast<NameId>(hi - lo), k, buckets },
                    &doomLines[li], &partsRemaining[li] });
            }
        };

        // CUDA route: the producers' bucket parts join THIS pass, so a straggler's
        // parts share the pass's device chunk with the unsplit LBs -- one
        // projection, one device route and one finalize sweep per iteration
        // (D-442). Parts own their page sets after the
        // pass's tasks; the processor route keeps round 2 (its parts are threads
        // that need the producers' stumps before they start).
        if (phase2Backend == Phase2Backend::cuda) {
            std::vector<ExecTask> dealtParts;
            for (std::size_t i = 0; i < tasks.size(); ++i)
                if (tasks[i].produceOnly) dealProducerStumps(i, dealtParts);
            for (const ExecTask& part : dealtParts) {
                if (gl::phase2TraceEnabled())
                    diagnosticsLog() << "[PHASE2-TASK] call=" << phase2TraceCall
                        << " pass=" << passNo << " task=" << tasks.size() << " li=" << part.li
                        << " parts=" << part.partCount << " producer=0\n";
                tasks.push_back(part);
                pageStore.emplace_back();
            }
            stumpBound.resize(tasks.size(), 0);
            taskStumps.resize(tasks.size(), 0);
            taskSubMatches.resize(tasks.size(), 0);
#if RT_MEASUREMENT
            taskNs.resize(tasks.size(), 0);
#endif
        }

#ifdef GL_CUDA
        if (phase2Backend == Phase2Backend::cuda && !tasks.empty()) {
            // Request files are owned by this serial GPU phase. Prepare each
            // thread handle before growth; every handle closes before the next
            // batch can purge the deload directory.
            PositionedFile requestFrontierFiles[2];
            gpu::RequestFrontierStorageOptions frontierStorage;
            if (parameters.allow_ssd_deload && !ceFilteringActive) {
                std::error_code directoryError;
                std::filesystem::create_directories(lbdeload::kDeloadDirectory, directoryError);
                assert(!directoryError && "request frontier directory creation failed");
                for (unsigned index = 0; index < 2; ++index) {
                    auto& file = requestFrontierFiles[index];
                    file.open(std::filesystem::path(lbdeload::kDeloadDirectory)
                        / (index == 0 ? "requests_frontier_0.bin" : "requests_frontier_1.bin"));
                    file.preallocate(0);
                    file.writeAt(0, nullptr, 0); // prepare this thread's positioned handle
                    frontierStorage.files[index] = &file;
                }
            }
            static gpu::CudaPhase2BlockPool devicePool([]() -> uint64_t {
#ifdef _WIN32
                char budgetText[32]{};
                std::size_t required = 0;
                const errno_t environmentResult = getenv_s(&required, budgetText,
                    sizeof(budgetText), "GL_GPU_BUDGET_MIB");
                assert(environmentResult == 0);
                if (required == 0) return 0;
                const char* text = budgetText;
#else
                const char* text = std::getenv("GL_GPU_BUDGET_MIB");
                if (text == nullptr) return 0;
#endif
                assert(*text != '\0');
                uint64_t value = 0;
                for (; *text != '\0'; ++text) {
                    assert(*text >= '0' && *text <= '9');
                    const uint64_t digit = static_cast<uint64_t>(*text - '0');
                    assert(value <= (UINT64_MAX - digit) / 10);
                    value = value * 10 + digit;
                }
                assert(value > 0 && value <= UINT64_MAX / (1024 * 1024));
                return value * 1024 * 1024;
            }(), gpu::kPhase2FrontierStoreReserveBytes);
            gl::memory_census::noteGpuDevice(devicePool.deviceTotalBytes(),
                devicePool.deviceFreeBeforePoolBytes(), devicePool.reservedBytes());
            static gpu::Phase2WorkDemand workDemand = gpu::initialPhase2WorkDemand();
            constexpr std::size_t kGpuProjectionShardCount =
                gpu::kPhase2ProjectionConstructionShardCount;
            const auto addProjectionUsage = [](
                gpu::Phase2ProjectionUsage& total,
                const gpu::Phase2ProjectionUsage& add) {
                total.logicalBlocks += add.logicalBlocks;
                total.statements += add.statements;
                total.nameRecords += add.nameRecords;
                total.nameBytes += add.nameBytes;
                total.nameSlots += add.nameSlots;
                total.ruleStringRecords += add.ruleStringRecords;
                total.ruleStringBytes += add.ruleStringBytes;
                total.byteMapViews += add.byteMapViews;
                total.byteMapEntries += add.byteMapEntries;
                total.byteMapSlots += add.byteMapSlots;
                total.byteKeyBytes += add.byteKeyBytes;
                total.blobRecords += add.blobRecords;
                total.blobBytes += add.blobBytes;
                total.reverseMapViews += add.reverseMapViews;
                total.reverseMapEntries += add.reverseMapEntries;
                total.reverseMapSlots += add.reverseMapSlots;
                total.reverseKeyBytes += add.reverseKeyBytes;
                total.reverseOwners += add.reverseOwners;
                total.podMapViews += add.podMapViews;
                total.podMapEntries += add.podMapEntries;
                total.podMapSlots += add.podMapSlots;
                total.podRunValues += add.podRunValues;
                total.mandatoryStatementKeys += add.mandatoryStatementKeys;
                total.metadataBytes += add.metadataBytes;
            };
            const auto addTaskUsage = [](
                gpu::Phase2TaskProjectionUsage& total,
                const gpu::Phase2TaskProjectionUsage& add) {
                total.tasks += add.tasks;
                total.batches += add.batches;
                total.terms += add.terms;
                total.stumps += add.stumps;
            };
            const auto hashMemoryBit = [](gpu::DeviceHashMemoryKind memory) {
                return 1u << static_cast<uint32_t>(memory);
            };
            const auto inputFor = gpu::makePhase2BufferPlanInput;
            const auto representable = [](const gpu::Phase2ProjectionUsage& usage,
                const gpu::Phase2TaskProjectionUsage& tasks, uint64_t retainedRows) {
                return retainedRows <= INT32_MAX
                    && usage.logicalBlocks <= INT32_MAX
                    && usage.statements <= INT32_MAX
                    && usage.nameRecords <= INT32_MAX
                    && usage.nameBytes <= INT32_MAX
                    && usage.nameSlots <= INT32_MAX
                    && usage.ruleStringRecords <= INT32_MAX
                    && usage.ruleStringBytes <= INT32_MAX
                    && usage.byteMapViews <= INT32_MAX
                    && usage.byteMapEntries <= INT32_MAX
                    && usage.byteMapSlots <= INT32_MAX
                    && usage.byteKeyBytes <= INT32_MAX
                    && usage.blobRecords <= INT32_MAX
                    && usage.blobBytes <= INT32_MAX
                    && usage.reverseMapViews <= INT32_MAX
                    && usage.reverseMapEntries <= INT32_MAX
                    && usage.reverseMapSlots <= INT32_MAX
                    && usage.reverseKeyBytes <= INT32_MAX
                    && usage.reverseOwners <= INT32_MAX
                    && usage.podMapViews <= INT32_MAX
                    && usage.podMapEntries <= INT32_MAX
                    && usage.podMapSlots <= INT32_MAX
                    && usage.podRunValues <= INT32_MAX
                    && usage.mandatoryStatementKeys <= INT32_MAX
                    && usage.metadataBytes <= INT32_MAX
                    && tasks.tasks <= INT32_MAX && tasks.batches <= INT32_MAX
                    && tasks.terms <= INT32_MAX && tasks.stumps <= INT32_MAX;
            };
            gl::Phase2TraceScope phase2PackingTrace("GL.Phase2.Packing", passNo);
            struct PackedChunk {
                std::size_t begin = 0, count = 0;
                gpu::Phase2BufferPlanInput input{};
            };
            GpuHostLayout directoryLayout;
            directoryLayout.add<int32_t>(tasks.size());
            directoryLayout.add<char>(tasks.size());
            directoryLayout.add<std::size_t>(M);
            directoryLayout.add<std::size_t>(M + 1);
            directoryLayout.add<std::size_t>(M);
            directoryLayout.add<std::size_t>(tasks.size());
            directoryLayout.add<std::size_t>(tasks.size());
            directoryLayout.add<gpu::Phase2TaskProjectionUsage>(tasks.size());
            directoryLayout.add<uint32_t>(M);
            directoryLayout.add<gpu::Phase2ProjectionUsage>(M);
            directoryLayout.add<PackedChunk>(M);
            directoryLayout.add<gpu::Phase2ProjectionUsage>(M);
            directoryLayout.add<std::size_t>(M + 1);
            gpuDirectoryStorage.reserve(directoryLayout.bytes);
            const auto reserveDirectory = [&](auto& column, std::size_t count) {
                column.bindStorage(gpuDirectoryStorage);
                column.reserve(count);
            };
            reserveDirectory(gpuTaskIndices, tasks.size());
            gpuTaskIndices.resize(tasks.size(), -1);
            reserveDirectory(gpuOutputOwners, tasks.size());
            gpuOutputOwners.resize(tasks.size(), 0);
            gpu::ProjectionColumn<std::size_t> familyCounts;
            reserveDirectory(familyCounts, M);
            familyCounts.resize(M);
            gpu::ProjectionColumn<std::size_t> familyStarts;
            reserveDirectory(familyStarts, M + 1);
            familyStarts.resize(M + 1);
            gpu::ProjectionColumn<std::size_t> familyWritten;
            reserveDirectory(familyWritten, M);
            familyWritten.resize(M);
            gpu::ProjectionColumn<std::size_t> familyPositions;
            reserveDirectory(familyPositions, tasks.size());
            familyPositions.resize(tasks.size());
            gpu::ProjectionColumn<std::size_t> packedTaskPositions;
            reserveDirectory(packedTaskPositions, tasks.size());
            gpu::ProjectionColumn<gpu::Phase2TaskProjectionUsage> cudaTaskUsages;
            reserveDirectory(cudaTaskUsages, tasks.size());
            cudaTaskUsages.resize(tasks.size());
            gpu::ProjectionColumn<uint32_t> selectedHashMemoriesByLi;
            reserveDirectory(selectedHashMemoriesByLi, M);
            selectedHashMemoriesByLi.resize(M);
            gpu::ProjectionColumn<gpu::Phase2ProjectionUsage> projectionUsageByLi;
            reserveDirectory(projectionUsageByLi, M);
            projectionUsageByLi.resize(M);
            gpu::ProjectionColumn<PackedChunk> cudaTaskChunks;
            reserveDirectory(cudaTaskChunks, M);
            gpu::ProjectionColumn<gpu::Phase2ProjectionUsage> chunkFamilyUsages;
            reserveDirectory(chunkFamilyUsages, M);
            gpu::ProjectionColumn<std::size_t> chunkFamilyStarts;
            reserveDirectory(chunkFamilyStarts, M + 1);
            // Every CUDA task's usage, its logical block's hash-memory selection
            // and projection usage come from the measurement the block recorded
            // at the end of this burst's phase 1 (recordPhase2Measurement): the
            // packing reads no LB content, so it never reloads a deloaded LB. The
            // block has been read-only since that phase 1, so the record is exact.
            for (std::size_t taskPosition = 0;
                 taskPosition < tasks.size(); ++taskPosition) {
                const ExecTask& task = tasks[taskPosition];
                if (task.produceOnly) continue;
                const Memory& recorded = *task.lb;
                assert(recorded.phase2MeasurementEpoch == phase2MeasurementEpoch
                    && "CUDA packing read a Phase 2 measurement not recorded in "
                       "this burst's phase 1");
                cudaTaskUsages[taskPosition] = gpu::phase2TaskUsageWithStumps(
                    recorded.phase2TaskUsage,
                    static_cast<uint32_t>(task.stump.count), ceFilteringActive);
                if (cudaTaskUsages[taskPosition].tasks == 0) continue;
                assert(recorded.phase2SelectedHashMemories != 0);
                selectedHashMemoriesByLi[task.li] =
                    recorded.phase2SelectedHashMemories;
                projectionUsageByLi[task.li] = recorded.phase2ProjectionUsage;
                ++familyCounts[task.li];
            }
            for (std::size_t li = 0; li < M; ++li)
                familyStarts[li + 1] = familyStarts[li] + familyCounts[li];
            for (std::size_t position = 0; position < tasks.size(); ++position) {
                if (cudaTaskUsages[position].tasks == 0) continue;
                const std::size_t li = tasks[position].li;
                familyPositions[familyStarts[li] + familyWritten[li]++] = position;
            }
            PackedChunk chunk;
            gpu::Phase2ProjectionUsage chunkProjectionUsage{};
            gpu::Phase2TaskProjectionUsage chunkTaskUsage{};
            gpu::Phase2ProjectionUsage passProjectionUsage{};
            gpu::Phase2TaskProjectionUsage passTaskUsage{};
            uint64_t chunkExaminedRows = 0, chunkRetainedRows = 0, chunkMaximumRows = 0;
            for (std::size_t taskPosition = 0; taskPosition < tasks.size(); ++taskPosition) {
                const ExecTask& task = tasks[taskPosition];
                if (task.produceOnly) continue;
                const GpuHostSpan<const std::size_t> family(
                    familyPositions.data() + familyStarts[task.li], familyCounts[task.li]);
                if (family.empty()) continue;
                const uint32_t selected = selectedHashMemoriesByLi[task.li];
                assert(selected != 0);
                const gpu::Phase2ProjectionUsage familyProjection =
                    projectionUsageByLi[task.li];
                gpu::Phase2TaskProjectionUsage familyTasks{};
                for (const auto position : family) addTaskUsage(familyTasks, cudaTaskUsages[position]);
                uint64_t classes = 0;
                for (uint32_t kind = 0; kind < 4; ++kind) classes += (selected >> kind) & 1u;
                const uint64_t familyExamined = familyTasks.batches * familyProjection.statements;
                const uint64_t familyRetained = classes * std::min(static_cast<uint64_t>(parameters.maxNumberFilteredStatements), familyProjection.statements);
                auto candidateProjection = chunkProjectionUsage;
                auto candidateTasks = chunkTaskUsage;
                addProjectionUsage(candidateProjection, familyProjection);
                addTaskUsage(candidateTasks, familyTasks);
                const auto fits = [&](const gpu::Phase2ProjectionUsage& projected,
                    const gpu::Phase2TaskProjectionUsage& scheduled, uint64_t examined,
                    uint64_t retained, uint64_t maximum) {
                    if (!representable(projected, scheduled, retained)) return false;
                    return gpu::planPhase2Buffers(inputFor(projected, scheduled,
                        examined, retained, maximum), devicePool.budget().blocks,
                        workDemand, {}, true).state == gpu::Phase2BufferPlanState::fits;
                };
                if (chunk.count != 0 && !fits(candidateProjection, candidateTasks,
                    chunkExaminedRows + familyExamined, chunkRetainedRows + familyRetained,
                    std::max(chunkMaximumRows, familyProjection.statements))) {
                    cudaTaskChunks.push_back(chunk);
                    chunk = PackedChunk{};
                    chunk.begin = packedTaskPositions.size();
                    chunkProjectionUsage = {};
                    chunkTaskUsage = {};
                    chunkExaminedRows = chunkRetainedRows = chunkMaximumRows = 0;
                    candidateProjection = familyProjection;
                    candidateTasks = familyTasks;
                }
                assert(fits(candidateProjection, candidateTasks,
                    chunkExaminedRows + familyExamined, chunkRetainedRows + familyRetained,
                    std::max(chunkMaximumRows, familyProjection.statements))
                    && "one LB input snapshot and minimum execution scratch must fit usable VRAM");
                for (const std::size_t position : family) packedTaskPositions.push_back(position);
                chunk.count += family.size();
                chunkProjectionUsage = candidateProjection;
                chunkTaskUsage = candidateTasks;
                chunkExaminedRows += familyExamined;
                chunkRetainedRows += familyRetained;
                chunkMaximumRows = std::max(chunkMaximumRows, familyProjection.statements);
                chunk.input = inputFor(chunkProjectionUsage, chunkTaskUsage,
                    chunkExaminedRows, chunkRetainedRows, chunkMaximumRows);
                addProjectionUsage(passProjectionUsage, familyProjection);
                addTaskUsage(passTaskUsage, familyTasks);
                familyCounts[task.li] = 0;
            }
            if (chunk.count != 0) cudaTaskChunks.push_back(chunk);
            phase2PackingTrace.finish();

            diagnosticsLog() << "[GPU-PACKING] pass=" << passNo
                      << " chunks=" << cudaTaskChunks.size()
                      << " logical_blocks="
                      << passProjectionUsage.logicalBlocks
                      << " statements=" << passProjectionUsage.statements
                      << " name_records=" << passProjectionUsage.nameRecords
                      << " name_bytes=" << passProjectionUsage.nameBytes
                      << " name_slots=" << passProjectionUsage.nameSlots
                      << " rule_string_records="
                      << passProjectionUsage.ruleStringRecords
                      << " rule_string_bytes="
                      << passProjectionUsage.ruleStringBytes
                      << " byte_map_views="
                      << passProjectionUsage.byteMapViews
                      << " byte_map_entries="
                      << passProjectionUsage.byteMapEntries
                      << " byte_map_slots="
                      << passProjectionUsage.byteMapSlots
                      << " byte_key_bytes="
                      << passProjectionUsage.byteKeyBytes
                      << " blob_records="
                      << passProjectionUsage.blobRecords
                      << " blob_bytes=" << passProjectionUsage.blobBytes
                      << " reverse_map_views="
                      << passProjectionUsage.reverseMapViews
                      << " reverse_map_entries="
                      << passProjectionUsage.reverseMapEntries
                      << " reverse_map_slots="
                      << passProjectionUsage.reverseMapSlots
                      << " reverse_key_bytes="
                      << passProjectionUsage.reverseKeyBytes
                      << " reverse_owners="
                      << passProjectionUsage.reverseOwners
                      << " pod_map_views="
                      << passProjectionUsage.podMapViews
                      << " pod_map_entries="
                      << passProjectionUsage.podMapEntries
                      << " pod_map_slots="
                      << passProjectionUsage.podMapSlots
                      << " pod_run_values="
                      << passProjectionUsage.podRunValues
                      << " mandatory_statement_keys="
                      << passProjectionUsage.mandatoryStatementKeys
                      << " metadata_bytes="
                      << passProjectionUsage.metadataBytes
                      << " tasks=" << passTaskUsage.tasks
                      << " batches=" << passTaskUsage.batches
                      << " terms=" << passTaskUsage.terms
                      << " stumps=" << passTaskUsage.stumps
                      << std::endl;

            uint64_t phase2TraceChunk = 0;
            for (const PackedChunk& packedChunk : cudaTaskChunks) {
            chunkFamilyUsages.clear();
            chunkFamilyStarts.clear();
            std::size_t previousLi = M;
            for (std::size_t position = packedChunk.begin;
                 position < packedChunk.begin + packedChunk.count; ++position) {
                const std::size_t li = tasks[packedTaskPositions[position]].li;
                if (li == previousLi) continue;
                chunkFamilyUsages.push_back(projectionUsageByLi[li]);
                chunkFamilyStarts.push_back(position);
                previousLi = li;
            }
            chunkFamilyStarts.push_back(packedChunk.begin + packedChunk.count);
            std::size_t firstFamily = 0;
            while (firstFamily < chunkFamilyUsages.size()) {
            std::size_t selectedFamilies = chunkFamilyUsages.size() - firstFamily;
            gpu::Phase2BufferPlanInput input{};
            gpu::Phase2BufferPlan plan{};
            gpu::Phase2HostFootprint hostPlan{};
            auto& hostPool = gpuHostStorage();
            const uint64_t availableHostBytes = static_cast<uint64_t>(std::min(
                hostPool.totalBlocks() - hostPool.blocksInUse(), hostPool.largestGpuHostRun()))
                * hostPool.blockBytes();
            for (;;) {
                gpu::Phase2ProjectionUsage projected{};
                gpu::Phase2TaskProjectionUsage scheduled{};
                uint64_t examined = 0, retained = 0, maximum = 0;
                for (std::size_t family = firstFamily;
                     family < firstFamily + selectedFamilies; ++family) {
                    const auto& usage = chunkFamilyUsages[family];
                    addProjectionUsage(projected, usage);
                    gpu::Phase2TaskProjectionUsage familyTasks{};
                    for (std::size_t position = chunkFamilyStarts[family];
                         position < chunkFamilyStarts[family + 1]; ++position)
                        addTaskUsage(familyTasks, cudaTaskUsages[packedTaskPositions[position]]);
                    addTaskUsage(scheduled, familyTasks);
                    const auto li = tasks[packedTaskPositions[chunkFamilyStarts[family]]].li;
                    uint64_t classes = 0;
                    for (uint32_t kind = 0; kind < 4; ++kind)
                        classes += (selectedHashMemoriesByLi[li] >> kind) & 1u;
                    examined += familyTasks.batches * usage.statements;
                    retained += classes * std::min<uint64_t>(
                        parameters.maxNumberFilteredStatements, usage.statements);
                    maximum = std::max(maximum, usage.statements);
                }
                input = inputFor(projected, scheduled, examined, retained, maximum);
                plan = gpu::planPhase2Buffers(input, devicePool.budget().blocks,
                    workDemand, {}, false, devicePool.budget().frontierStoreBlocks);
                assert(plan.state == gpu::Phase2BufferPlanState::fits);
                hostPlan = gpu::planPhase2HostStorage(input, plan,
                    GpuHostSpan<const gpu::Phase2ProjectionUsage>(
                        chunkFamilyUsages.data() + firstFamily, selectedFamilies),
                    M, availableHostBytes, parameters.allow_ssd_deload && !ceFilteringActive);
                if (hostPlan.windows != 0) break;
                diagnosticsLog() << "[GPU-HOST-CAPACITY] available_bytes=" << availableHostBytes
                    << " logical_blocks=" << selectedFamilies
                    << " fixed_bytes=" << hostPlan.fixedBytes
                    << " construction_one_bytes=" << hostPlan.constructionOneBytes
                    << " construction_two_bytes=" << hostPlan.constructionTwoBytes
                    << " execution_bytes=" << hostPlan.executionBytes << std::endl;
                assert(selectedFamilies > 1
                    && "one complete LB family and its GPU-host support exceed GL_GPU_HOST_BUDGET_MIB");
                selectedFamilies = (selectedFamilies + 1) / 2;
            }
            gl::Phase2TraceScope phase2ChunkTrace("GL.Phase2.Chunk", ++phase2TraceChunk);
            gl::Phase2TraceScope phase2ChunkSetupTrace("GL.Phase2.ChunkSetup", phase2TraceChunk);
            const std::size_t chunkBegin = chunkFamilyStarts[firstFamily];
            const std::size_t chunkEnd = chunkFamilyStarts[firstFamily + selectedFamilies];
            const GpuHostSpan<const std::size_t> cudaTaskPositions(
                packedTaskPositions.data() + chunkBegin, chunkEnd - chunkBegin);
            const auto& projectionCapacity = input.projection;
            const auto& taskCapacity = input.tasks;
            const auto& filterCapacity = input.filter;
            diagnosticsLog() << "[GPU-HOST-PLAN] logical_blocks=" << selectedFamilies
                << " upload_windows=" << hostPlan.windows
                << " available_bytes=" << availableHostBytes
                << " planned_peak_bytes=" << (hostPlan.windows == 2
                    ? hostPlan.twoWindowPeakBytes : hostPlan.oneWindowPeakBytes)
                << " fixed_bytes=" << hostPlan.fixedBytes
                << " execution_bytes=" << hostPlan.executionBytes << std::endl;
            assert(devicePool.availableBlocks() == devicePool.budget().totalBlocks());
            gpu::CudaPhase2Arena projectionArena(devicePool, plan.blocks[0]);
            gpu::CudaPhase2Arena taskArena(devicePool, plan.blocks[1]);
            gpu::CudaPhase2Arena filterArena(devicePool, plan.blocks[2]);
            gpu::CudaPhase2Arena growthArena(devicePool, plan.blocks[3]);
            gpu::CudaPhase2Arena orderingArena(devicePool, plan.blocks[4]);
            gpu::CudaPhase2Arena evaluationArena(devicePool, plan.blocks[5]);
            const auto chunkDirectoryLayout = gpu::phase2HostDirectoryLayout(
                M, projectionCapacity.logicalBlocks, taskCapacity.tasks);
            GpuHostArena chunkDirectoryStorage;
            chunkDirectoryStorage.reserve(chunkDirectoryLayout.bytes);
            const auto reserveChunkDirectory = [&](auto& column, std::size_t count) {
                column.bindStorage(chunkDirectoryStorage);
                column.reserve(count);
            };
            gpu::ProjectionColumn<char> projectionSeen;
            reserveChunkDirectory(projectionSeen, M);
            projectionSeen.resize(M);
            gpu::ProjectionColumn<std::size_t> projectionLis;
            reserveChunkDirectory(projectionLis, projectionCapacity.logicalBlocks);
            gpu::ProjectionColumn<Memory*> projectionOrder;
            reserveChunkDirectory(projectionOrder, projectionCapacity.logicalBlocks);
            gpu::ProjectionColumn<uint32_t> projectionSelectedHashMemories;
            reserveChunkDirectory(projectionSelectedHashMemories, projectionCapacity.logicalBlocks);
            gpu::ProjectionColumn<std::size_t> outputTaskByBlock;
            reserveChunkDirectory(outputTaskByBlock, projectionCapacity.logicalBlocks);
            outputTaskByBlock.resize(projectionCapacity.logicalBlocks);
            gpu::ProjectionColumn<int32_t> projectedBlockByLi;
            reserveChunkDirectory(projectedBlockByLi, M);
            gpu::ProjectionColumn<std::size_t> windowEnds;
            reserveChunkDirectory(windowEnds, projectionCapacity.logicalBlocks);
            gpu::ProjectionColumn<gpu::DeviceLogicalBlockProjection> canonicalDescriptors;
            reserveChunkDirectory(canonicalDescriptors, projectionCapacity.logicalBlocks);
            gpu::ProjectionColumn<uint8_t> descriptorSeen;
            reserveChunkDirectory(descriptorSeen, projectionCapacity.logicalBlocks);
            gpu::ProjectionColumn<Memory*> scheduleOrder;
            reserveChunkDirectory(scheduleOrder, taskCapacity.tasks);
            gpu::ProjectionColumn<uint64_t> taskSubkeyCounts;
            reserveChunkDirectory(taskSubkeyCounts, taskCapacity.tasks);
            gpu::ProjectionColumn<SealedPageSet*> outputs;
            reserveChunkDirectory(outputs, projectionCapacity.logicalBlocks);
            gpu::ProjectionColumn<int64_t> gpuDoomLines;
            reserveChunkDirectory(gpuDoomLines, projectionCapacity.logicalBlocks);
            for (const std::size_t taskPosition : cudaTaskPositions) {
                const ExecTask& task = tasks[taskPosition];
                if (projectionSeen[task.li] != 0) continue;
                projectionSeen[task.li] = 1;
                assert(projectionLis.size()
                    < projectionCapacity.logicalBlocks);
                projectionLis.push_back(task.li);
                projectionOrder.push_back(task.lb);
                assert(selectedHashMemoriesByLi[task.li] != 0);
                projectionSelectedHashMemories.push_back(
                    selectedHashMemoriesByLi[task.li]);
                outputTaskByBlock[projectionLis.size() - 1] = taskPosition;
                gpuOutputOwners[taskPosition] = 1;
            }

            gpu::Phase2ProjectionArena hostProjection(projectionCapacity,
                gpu::Phase2ProjectionArena::SealingOnly{});
            uint64_t peakHostProjectionBytes = hostProjection.hostBytes();
            uint64_t projectionShardHostBytes = 0;
            gpu::Phase2TaskProjectionArena hostTasks(taskCapacity);
            gpu::Phase2FilterScheduleArena filterSchedule(filterCapacity);
            gpu::Phase2GrowthScheduleArena growthSchedule(plan.growth.calls);
            gpu::CudaPhase2ProjectionBuffer deviceProjection(projectionCapacity, &projectionArena);
            gpu::CudaPhase2TaskBuffer deviceTasks(taskCapacity, &taskArena);
            gpu::CudaPhase2FilterSortBuffer deviceFilter(filterCapacity, &filterArena);
            gpu::CudaPhase2GrowthBuffer deviceGrowth(plan.growth, &growthArena);
            gpu::CudaPhase2OrderingBuffer deviceOrdering(plan.ordering, &orderingArena);
            gpu::CudaPhase2EvaluationBuffer deviceEvaluation(plan.evaluation, &evaluationArena);
            gpu::Phase2SealingArena sealing(plan.evaluation);
            diagnosticsLog() << "[GPU-PHASE2-MEMORY] budget_bytes=" << devicePool.budget().bytes
                << " reserved_blocks=" << plan.totalBlocks << " input_blocks="
                << plan.blocks[0] + plan.blocks[1] + plan.blocks[2]
                << " frontier_store_blocks=" << plan.frontierStoreDedicatedBlocks
                << std::endl;
            phase2ChunkSetupTrace.finish();
            const auto gpuPassStarted = std::chrono::steady_clock::now();

            if (!projectionOrder.empty()) {
                projectedBlockByLi.resize(M, -1);
                // The host planner permits one or two windows. The measured
                // payload target and whole-block boundaries are identical.
                constexpr uint64_t hostWindowTargetBytes = gpu::kPhase2HostWindowTargetBytes;
                windowEnds.clear();
                std::size_t windowBegin = 0;
                gpu::Phase2ProjectionUsage windowUsage{};
                for (std::size_t block = 0; block < projectionOrder.size(); ++block) {
                    auto candidate = windowUsage;
                    addProjectionUsage(candidate, projectionUsageByLi[projectionLis[block]]);
                    const auto candidateCapacity = inputFor(candidate, {}, 0, 0, 0).projection;
                    if (block > windowBegin
                        && gpu::phase2ProjectionHostBytes(candidateCapacity) > hostWindowTargetBytes) {
                        windowEnds.push_back(block);
                        windowBegin = block;
                        windowUsage = {};
                    }
                    addProjectionUsage(windowUsage, projectionUsageByLi[projectionLis[block]]);
                }
                windowEnds.push_back(projectionOrder.size());
                windowBegin = 0;
                gpu::Phase2ProjectionArena::ShardBases uploadedEnds{};
                deviceProjection.beginStreamedUpload();
                // One set of construction threads serves every window of the
                // chunk. With two slots, upload of window k overlaps building
                // window k + 1; one slot joins before reuse.
                gpu::Phase2ProjectionWorkers projectionWorkers(
                    static_cast<uint32_t>(kGpuProjectionShardCount));
                gpu::Phase2WindowUploader windowUploader(deviceProjection, hostProjection);
                peak_snapshot::ExtraRootScope uploadSnapshot("GpuUploadWindow",&windowUploader,
                    [](void* p){static_cast<gpu::Phase2WindowUploader*>(p)->quiesceForSnapshot();},
                    [](void* p){static_cast<gpu::Phase2WindowUploader*>(p)->observeForSnapshot();});
                peak_snapshot::ExtraRootScope retainedSnapshot("GpuRetainedProjection",&hostProjection,nullptr,
                    [](void* p){static_cast<gpu::Phase2ProjectionArena*>(p)->snapshotBytes();});

                GpuHostObjectArray<std::optional<gpu::Phase2ProjectionArena>> windowSlots(hostPlan.windows);
                std::size_t windowOrdinal = 0;
                uint64_t inFlightWindowBytes = 0;
                for (const std::size_t windowEnd : windowEnds) {
                    // One oversized logical block owns its exact measured window;
                    // host staging never truncates or repartitions device work.
                    gl::Phase2TraceScope phase2ProjectionWindowTrace("GL.Phase2.Projection.Window",
                        windowEnd - windowBegin);
                    gl::Phase2TraceScope phase2WindowSetupTrace("GL.Phase2.Projection.Setup");
                    std::array<gpu::Phase2ProjectionCapacity, kGpuProjectionShardCount> shardCapacities{};
                    gpu::Phase2ProjectionCapacity windowStorage{};
                    gpu::Phase2ProjectionUsage totalUsage{};
                    for (std::size_t shard = 0; shard < kGpuProjectionShardCount; ++shard) {
                        gpu::Phase2ProjectionUsage usage{};
                        for (std::size_t block = windowBegin + shard; block < windowEnd;
                             block += kGpuProjectionShardCount)
                            addProjectionUsage(usage, projectionUsageByLi[projectionLis[block]]);
                        addProjectionUsage(totalUsage, usage);
                        shardCapacities[shard] = inputFor(usage, {}, 0, 0, 0).projection;
                        windowStorage = gpu::addProjectionCapacity(windowStorage, shardCapacities[shard]);
                    }
                    const auto windowCapacity = inputFor(totalUsage, {}, 0, 0, 0).projection;
                    if (hostPlan.windows == 1) {
                        windowUploader.waitUntilUploaded();
                        inFlightWindowBytes = 0;
                    }
                    // The upload thread empties a slot as soon as its window's
                    // upload and sealing copy finish, and a slot is reused only
                    // after the next submit or waitUntilUploaded saw that happen.
                    auto& windowSlot = windowSlots[windowOrdinal % windowSlots.size()];
                    assert(!windowSlot.has_value() && "a projection window outlived its upload");
                    windowSlot.emplace(windowCapacity, windowStorage);
                    auto* window = &*windowSlot;
                    peakHostProjectionBytes = std::max(peakHostProjectionBytes,
                        hostProjection.hostBytes() + inFlightWindowBytes + window->hostBytes());
                    canonicalDescriptors.resize(windowEnd - windowBegin);
                    descriptorSeen.assign(windowEnd - windowBegin, 0);
                    GpuHostObjectArray<std::optional<gpu::Phase2ProjectionArena>> projectionShards(kGpuProjectionShardCount);
                    gpu::Phase2ProjectionCapacity sliceStart{};
                    uint64_t windowShardBytes = 0;
                    for (std::size_t shard = 0; shard < kGpuProjectionShardCount; ++shard) {
                        projectionShards[shard].emplace(
                            shardCapacities[shard], *window, sliceStart);
                        sliceStart = gpu::addProjectionCapacity(sliceStart, shardCapacities[shard]);
                        windowShardBytes += projectionShards[shard]->hostBytes();
                    }
                    projectionShardHostBytes = std::max(projectionShardHostBytes, windowShardBytes);

                    peak_snapshot::ExtraRootScope shardSnapshot("GpuProjectionShards",&projectionShards,nullptr,
                        [](void* p){
                            const auto& shards=*static_cast<GpuHostObjectArray<std::optional<gpu::Phase2ProjectionArena>>*>(p);
                            for(std::size_t i=0;i<shards.size();++i) if(shards[i]) shards[i]->snapshotBytes();
                        });
                    peak_snapshot::ExtraRootScope constructionSnapshot("GpuConstructionWindow",&window,nullptr,
                        [](void* p){
                            const auto& owner=*static_cast<gpu::Phase2ProjectionArena**>(p);
                            if(owner) {peak_snapshot::note(owner,sizeof(*owner),3);owner->snapshotBytes();}
                        });
                    phase2WindowSetupTrace.finish();
                    gl::Phase2TraceScope phase2WorkersTrace("GL.Phase2.Projection.Workers");
                    std::atomic<std::size_t> projectionCursor{windowBegin};
                    gl::memory_census::markPhase(gl::PoolPhase::Phase2Projection);
                    steward->beginPhaseWindow(2, &projectionCursor, MemoryOrderView(projectionOrder.data(), projectionOrder.size()),
                        static_cast<unsigned>(kGpuProjectionShardCount), lbdeload::kDeloadDirectory);
                    auto projectionJob = [this, &projectionCursor, &projectionOrder,
                        &projectionSelectedHashMemories, &projectionShards, windowBegin, windowEnd]
                        (uint32_t shard) {
                        gl::Phase2TraceScope projectionCpuTrace("GL.Phase2.ProjectionWorker",
                            shard, gl::phase2TraceEnabled(), true);
                        auto& image = *projectionShards[shard];
                        for (std::size_t block = windowBegin + shard; block < windowEnd;
                             block += kGpuProjectionShardCount) {
                            Memory* logicalBlock = projectionOrder[block];
                            this->steward->claimAndLoadForWork(*logicalBlock, 2, lbdeload::kDeloadDirectory);
                            image.appendLogicalBlock(*logicalBlock, definitions, anchorInfo, parameters,
                                projectionSelectedHashMemories[block], true);
                            assert(logicalBlock->stewardClaim.load(std::memory_order_relaxed)
                                == static_cast<uint8_t>(Memory::StewardClaim::WorkerOwned));
                            logicalBlock->stewardClaim.store(static_cast<uint8_t>(Memory::StewardClaim::Idle),
                                std::memory_order_release);
                            projectionCursor.fetch_add(1, std::memory_order_relaxed);
                        }
                    };
                    projectionWorkers.run(static_cast<uint32_t>(kGpuProjectionShardCount), projectionJob);
                    steward->endPhaseWindow();
                    phase2WorkersTrace.finish();
                    gl::Phase2TraceScope phase2AdoptTrace("GL.Phase2.Projection.Adopt");
                    std::array<gpu::Phase2ProjectionArena*, kGpuProjectionShardCount> shardSources{};
                    std::array<uint32_t, kGpuProjectionShardCount> shardFirstBlocks{};
                    for (std::size_t shard = 0; shard < kGpuProjectionShardCount; ++shard)
                        shardSources[shard] = &*projectionShards[shard];
                    gl::memory_census::sampleNow();
                    // Adoption adds the uploaded column ends, so every reference
                    // already addresses its complete-image offset.
                    window->adoptShards(shardSources.data(),
                        static_cast<uint32_t>(kGpuProjectionShardCount), shardFirstBlocks.data(),
                        uploadedEnds, projectionWorkers);
                    for (std::size_t shard = 0; shard < kGpuProjectionShardCount; ++shard) {
                        uint32_t local = 0;
                        for (std::size_t block = windowBegin + shard; block < windowEnd;
                             block += kGpuProjectionShardCount) {
                            const auto position = block - windowBegin;
                            assert(descriptorSeen[position] == 0);
                            descriptorSeen[position] = 1;
                            canonicalDescriptors[position] =
                                window->logicalBlocks[shardFirstBlocks[shard] + local++];
                        }
                        projectionShards[shard].reset();
                    }
                    assert(window->logicalBlocks.size() == windowEnd - windowBegin);
                    for (std::size_t block = windowBegin; block < windowEnd; ++block) {
                        assert(descriptorSeen[block - windowBegin] == 1);
                        window->logicalBlocks[block - windowBegin] =
                            canonicalDescriptors[block - windowBegin];
                        assert(projectedBlockByLi[projectionLis[block]] < 0);
                        projectedBlockByLi[projectionLis[block]] = static_cast<int32_t>(block);
                    }
                    phase2AdoptTrace.finish();
                    const auto nextEnds = window->streamedEnd(uploadedEnds);
                    inFlightWindowBytes = window->hostBytes();
                    windowUploader.submit(windowSlot, uploadedEnds);
                    // The upload thread now owns and frees the window, so the
                    // construction-window snapshot root must not observe it.
                    window = nullptr;
                    ++windowOrdinal;
                    uploadedEnds = nextEnds;
                    gl::Phase2TraceScope phase2CensusTrace("GL.Phase2.Projection.Census");
                    gl::memory_census::sampleNow();
                    phase2CensusTrace.finish();
                    windowBegin = windowEnd;
                }
                windowUploader.finish();
                for (std::size_t slot = 0; slot < windowSlots.size(); ++slot)
                    assert(!windowSlots[slot].has_value() && "a projection window outlived its upload");

                assert(hostProjection.logicalBlocks.size() == projectionOrder.size());
                const auto gpuProjectionFinished = std::chrono::steady_clock::now();

                gl::Phase2TraceScope phase2ScheduleTrace("GL.Phase2.Schedule");
                scheduleOrder.clear();
                for (const std::size_t taskPosition : cudaTaskPositions)
                    scheduleOrder.push_back(tasks[taskPosition].lb);
                std::atomic<std::size_t> scheduleCursor{ 0 };
                gl::memory_census::markPhase(gl::PoolPhase::Phase2Device);
                steward->beginPhaseWindow(
                    /*phase=*/2, &scheduleCursor, MemoryOrderView(scheduleOrder.data(), scheduleOrder.size()), 1,
                    lbdeload::kDeloadDirectory);
                std::size_t scheduleIndex = 0;
                for (const std::size_t taskPosition : cudaTaskPositions) {
                    const ExecTask& task = tasks[taskPosition];
                    scheduleCursor.store(
                        scheduleIndex, std::memory_order_relaxed);
                    steward->claimAndLoadForWork(
                        *task.lb, /*phase=*/2,
                        lbdeload::kDeloadDirectory);
                    const int32_t blockIndexSigned =
                        projectedBlockByLi[task.li];
                    assert(blockIndexSigned >= 0);
                    const uint32_t blockIndex =
                        static_cast<uint32_t>(blockIndexSigned);

                    // Match the processor's hard staged-arrival invariant before
                    // the device sees the immutable projection.
                    for (int32_t index = 0;
                         index < task.lb->intExternalStatements.size(); ++index) {
                        const IntEncodedExpr& external =
                            task.lb->intExternalStatements[index];
                        if (external.maxIteration
                            > parameters.maxIterationNumberVariable) continue;
                        assert(task.lb->intKnownStatements.find(
                                   StatementKey{
                                       external.originalId,
                                       external.validityId }) != nullptr
                               && "a staged mail arrival must remain registered "
                                  "before CUDA Phase 2 projection");
                    }

                    std::array<gpu::Phase2RequestBatchInput, 4> batches{};
                    uint32_t batchCount = 0;
                    if (ceFilteringActive) {
                        assert(task.partCount == 1 && task.stump.count == 0);
                        batches[0].kind =
                            gpu::DeviceRequestBatchKind::counterExample;
                        batches[0].memory =
                            gpu::DeviceHashMemoryKind::overall;
                        batchCount = 1;
                    } else {
                        if (!task.lb->workingMemory.encodedMap.empty()
                            && !task.lb->intLocalEncodedStatements.empty()) {
                            gpu::Phase2RequestBatchInput& input =
                                batches[batchCount++];
                            input.kind =
                                gpu::DeviceRequestBatchKind::workingRules;
                            input.memory = gpu::DeviceHashMemoryKind::working;
                            input.terms[0].views[0] =
                                gpu::DeviceMandatoryViewKind::local;
                            input.terms[0].viewCount = 1;
                            input.termCount = 1;
                        }

                        const bool hasNewThisBurst =
                            !task.lb->intLocalEncodedStatementsDelta.empty()
                            || (!task.lb->intExternalStatements.empty()
                                && !task.lb->intLocalEncodedStatements.empty());
                        if (hasNewThisBurst) {
                            gpu::Phase2RequestBatchInput& input =
                                batches[batchCount++];
                            input.kind =
                                gpu::DeviceRequestBatchKind::newThisBurst;
                            input.memory = gpu::DeviceHashMemoryKind::overall;
                            input.terms[0].views[0] =
                                gpu::DeviceMandatoryViewKind::localDelta;
                            input.terms[0].viewCount = 1;
                            input.terms[1].views[0] =
                                gpu::DeviceMandatoryViewKind::external;
                            input.terms[1].views[1] =
                                gpu::DeviceMandatoryViewKind::local;
                            input.terms[1].viewCount = 2;
                            input.termCount = 2;
                        }

                        if (!task.lb->localHashMemory.encodedMap.empty()
                            && !task.lb->intExternalStatements.empty()) {
                            gpu::Phase2RequestBatchInput& input =
                                batches[batchCount++];
                            input.kind = gpu::DeviceRequestBatchKind::
                                localRulesWithMail;
                            input.memory = gpu::DeviceHashMemoryKind::local;
                            input.terms[0].views[0] =
                                gpu::DeviceMandatoryViewKind::external;
                            input.terms[0].viewCount = 1;
                            input.termCount = 1;
                        }

                        if (!task.lb->localHashMemoryDelta.encodedMap.empty()) {
                            gpu::Phase2RequestBatchInput& input =
                                batches[batchCount++];
                            input.kind =
                                gpu::DeviceRequestBatchKind::localDeltaRules;
                            input.memory = gpu::DeviceHashMemoryKind::localDelta;
                        }
                    }
                    assert(batchCount > 0 && batchCount <= batches.size());
                    const uint32_t selectedHashMemories =
                        selectedHashMemoriesByLi[task.li];
                    assert(selectedHashMemories != 0);
                    for (uint32_t localBatch = 0;
                         localBatch < batchCount; ++localBatch) {
                        assert((selectedHashMemories
                            & hashMemoryBit(batches[localBatch].memory)) != 0);
                    }

                    const uint32_t gpuTaskIndex =
                        static_cast<uint32_t>(hostTasks.tasks.size());
                    const uint32_t batchOffset =
                        static_cast<uint32_t>(hostTasks.batches.size());
                    const gpu::DevicePhase2Task projectedTask = hostTasks.appendTask(
                        blockIndex, batches.data(), batchCount,
                        task.stump.stumps,
                        static_cast<uint32_t>(task.stump.count),
                        task.stump.ordinal, task.stump.total,
                        parameters.maxIterationNumberVariable,
                        ceFilteringActive ? 1u : 0u);
                    for (uint32_t si = 0; si < projectedTask.stumpCount; ++si) {
                        auto& stump = hostTasks.stumps[projectedTask.stumpOffset + si];
                        for (NameId pi = 0; pi < stump.count; ++pi)
                            stump.statementIndices[pi] = hostProjection.projectedStatementIndex(
                                blockIndex, stump.statementIndices[pi]);
                    }
                    gpuTaskIndices[taskPosition] =
                        static_cast<int32_t>(gpuTaskIndex);

                    const uint32_t statementCount = hostProjection.
                        logicalBlocks[blockIndex].statementCount;
                    for (uint32_t localBatch = 0;
                         localBatch < batchCount; ++localBatch) {
                        const uint32_t filterCallIndex =
                            static_cast<uint32_t>(filterSchedule.calls.size());
                        filterSchedule.appendCall(
                            blockIndex, batches[localBatch].memory,
                            parameters.maxIterationNumberVariable,
                            /*alsoAcceptFullKeys=*/1, statementCount,
                            static_cast<uint32_t>(parameters.maxNumberFilteredStatements));
                        growthSchedule.appendCall(
                            gpuTaskIndex, batchOffset + localBatch,
                            filterCallIndex);
                    }
                    assert(task.lb->stewardClaim.load(
                               std::memory_order_relaxed)
                           == static_cast<uint8_t>(
                                  Memory::StewardClaim::WorkerOwned));
                    task.lb->stewardClaim.store(
                        static_cast<uint8_t>(Memory::StewardClaim::Idle),
                        std::memory_order_release);
                    ++scheduleIndex;
                    scheduleCursor.store(
                        scheduleIndex, std::memory_order_relaxed);
                }
                assert(scheduleIndex == scheduleOrder.size());
                steward->endPhaseWindow();
                phase2ScheduleTrace.finish();

                if (gl::phase2TraceEnabled()) {
                    for (uint32_t task = 0; task < hostTasks.tasks.size(); ++task) {
                        const auto& image = hostTasks.tasks[task];
                        diagnosticsLog() << "[PHASE2-GPU-TASK] call=" << phase2TraceCall
                            << " pass=" << passNo << " chunk=" << phase2TraceChunk
                            << " gpu_task=" << task << " gpu_lb=" << image.logicalBlockIndex
                            << " li=" << projectionLis[image.logicalBlockIndex]
                            << " stump_ordinal=" << image.stumpOrdinal << " stump_total=" << image.stumpTotal
                            << " stumps=" << image.stumpCount << " batches=" << image.batchCount << '\n';
                        for (uint32_t offset = 0; offset < image.batchCount; ++offset) {
                            const auto& batch = hostTasks.batches[image.batchOffset + offset];
                            diagnosticsLog() << "[PHASE2-GPU-REQUEST-KIND] call=" << phase2TraceCall
                                << " pass=" << passNo << " chunk=" << phase2TraceChunk
                                << " gpu_task=" << task << " batch=" << image.batchOffset + offset
                                << " kind=" << static_cast<uint32_t>(batch.kind)
                                << " memory=" << static_cast<uint32_t>(batch.memory)
                                << " terms=" << batch.termCount << '\n';
                        }
                    }
                }
                const auto gpuDeviceRouteStarted =
                    std::chrono::steady_clock::now();
                const double gpuPreparationSeconds =
                    std::chrono::duration<double>(
                        gpuDeviceRouteStarted - gpuPassStarted).count();
                const double gpuProjectionSeconds =
                    std::chrono::duration<double>(
                        gpuProjectionFinished - gpuPassStarted).count();
                const double gpuScheduleSeconds =
                    std::chrono::duration<double>(
                        gpuDeviceRouteStarted - gpuProjectionFinished).count();
                const gpu::Phase2FilterClassCensus filterClassCensus =
                    filterSchedule.measureClassReuse();

                // CUDA event time covers the stream, including host drain gaps.
                // Separate host transfer and complete route measurements retain
                // those costs instead of attributing them to individual kernels.
                static gpu::CudaPhase2DeviceTimer gpuDeviceTimer;
                gpuDeviceTimer.start();

                deviceTasks.upload(hostTasks);
                deviceProjection.finishPhase2Upload();
                const uint32_t retainedRows = deviceFilter.filterAndSort(
                    deviceProjection, filterSchedule);
                taskSubkeyCounts.assign(hostTasks.tasks.size(), 0);
                outputs.resize(projectionOrder.size());
                for (std::size_t block = 0; block < projectionOrder.size(); ++block) {
                    auto& pages = pageStore[base + outputTaskByBlock[block]];
                    pages.bind(&staticMemory());
                    outputs[block] = &pages;
                    assert(doomLines[projectionLis[block]].load(std::memory_order_relaxed) == kNoDoomLine);
                    lbFiringRecordsCanonical[projectionLis[block]] = 0;
                }
                gpu::Phase2StreamBuffers buffers{deviceProjection, deviceTasks, deviceFilter,
                    deviceGrowth, deviceOrdering, deviceEvaluation};
                gpuDoomLines.assign(outputs.size(), kNoDoomLine);
                // Device window: the device reads only the uploaded projection
                // and the sealing writes pool pages owned by no logic block, so
                // every logic block is Idle here. The open window lets the
                // working-set steward keep its reserve by evicting logic blocks
                // while final sealed output grows the logical-block pool;
                // its cursor stays at the head of the active order, which the
                // finalize sweep reads first.
                std::atomic<std::size_t> deviceCursor{ 0 };
                steward->beginPhaseWindow(/*phase=*/2, &deviceCursor, &active,
                    workers, lbdeload::kDeloadDirectory);
                const auto streamed = sealing.runWindows(buffers, hostProjection, hostTasks,
                    growthSchedule, gpu::DevicePhase2GrowthParameters{parameters.maxLenHypoKey,
                        parameters.maxNumberSecondaryVariables, parameters.maxNumberSecondaryVariablesOrint,
                        gpu::kDeviceGrowthProductionCooperativeSpan, 0}, plan, outputs.data(),
                    static_cast<uint32_t>(outputs.size()), taskSubkeyCounts.data(),
                    static_cast<uint32_t>(taskSubkeyCounts.size()), gpuDoomLines.data(), frontierStorage);
                steward->endPhaseWindow();
                gl::Phase2TraceScope phase2RouteReportTrace("GL.Phase2.RouteReport");
                for (std::size_t block = 0; block < projectionLis.size(); ++block)
                    doomLines[projectionLis[block]].store(gpuDoomLines[block], std::memory_order_relaxed);
                // The projection measurement captures its largest construction
                // window plus retained sealing data while that window was live.
                gl::memory_census::noteGpuChunk({ peakHostProjectionBytes,
                    projectionShardHostBytes, hostTasks.hostBytes(),
                    filterSchedule.hostBytes(), growthSchedule.hostBytes(),
                    sealing.hostBytes(), deviceProjection.pinnedHostBytes() });
                gl::memory_census::sampleNow();
                const auto& growthResult = streamed.growth;
                for (const auto taskPosition : cudaTaskPositions) {
                    assert(gpuTaskIndices[taskPosition] >= 0);
                    const uint64_t subkeys = taskSubkeyCounts[gpuTaskIndices[taskPosition]];
                    assert(subkeys <= INT64_MAX);
                    taskSubMatches[taskPosition] = static_cast<int64_t>(subkeys);
                    if (gl::phase2TraceEnabled())
                        diagnosticsLog() << "[PHASE2-TASK-WORK] call=" << phase2TraceCall
                            << " pass=" << passNo << " chunk=" << phase2TraceChunk
                            << " task=" << taskPosition << " li=" << tasks[taskPosition].li
                            << " gpu_task=" << gpuTaskIndices[taskPosition] << " subkeys=" << subkeys << '\n';

                }
                gpu::observePhase2WorkDemand(workDemand, streamed.feedback);
                const double gpuDeviceSeconds = gpuDeviceTimer.stopSeconds();
                const auto gpuFinalizeStarted = std::chrono::steady_clock::now();
                const double gpuDeviceRouteSeconds = std::chrono::duration<double>(
                    gpuFinalizeStarted - gpuDeviceRouteStarted).count();
                diagnosticsLog() << "[GPU-STREAM] requests=" << streamed.evaluatedRequests
                    << " sealed_records=" << streamed.sealedRecords
                    << " event_capacity=" << plan.growth.acceptedEvents
                    << " request_capacity=" << plan.growth.rawRequests
                    << " cutoff_drains=" << growthResult.cutoffDrainCount
                    << " pruned_frontier_nodes=" << growthResult.prunedFrontierNodes
                    << " candidate_node_window_queries=" << growthResult.candidateNodeWindowQueries
                    << " unsliced_gate_nodes=" << growthResult.unslicedGateNodes
                    << " scheduled_gate_nodes=" << growthResult.scheduledGateNodes
                    << " candidate_positions_before_prefilter=" << growthResult.candidatePositionsBeforePrefilter
                    << " candidate_positions_after_prefilter=" << growthResult.candidatePositionsAfterPrefilter
                    << " frontier_transfer_calls=" << growthResult.frontierTransferCalls
                    << " frontier_copy_capacity=" << growthResult.frontierTransferRows
                    << " frontier_capacity=" << plan.growth.frontierRecords
                    << " frontier_store=" << plan.growth.frontierStoreRecords
                    << " frontier_spills=" << growthResult.frontierSpills
                    << " request_disk_writes=" << growthResult.frontierStorage.writes
                    << " request_disk_reads=" << growthResult.frontierStorage.reads
                    << " request_disk_write_bytes=" << growthResult.frontierStorage.writtenBytes
                    << " request_disk_read_bytes=" << growthResult.frontierStorage.readBytes
                    << " request_resident_payload_peak=" << growthResult.frontierStorage.peakResidentPayloadBytes
                    << " request_resident_storage_peak=" << growthResult.frontierStorage.peakResidentStorageBytes
                    << " probe_windows=" << growthResult.probeWindows
                    << " exact_count_reads=" << growthResult.exactCountReads
                    << " candidate_capacity=" << plan.growth.candidateWindowRecords
                    << " value_capacity=" << plan.evaluation.localValues
                    << " firing_capacity=" << plan.evaluation.firingRecords
                    << " generation_and_drain_seconds=" << streamed.generationAndDrainSeconds
                    << " sealing_seconds=" << streamed.sealingSeconds
                    << " firings=" << streamed.firingRecords << " request_drains=" << growthResult.drainCount
                    << " downloads=" << streamed.outputDownloads << " output_transfer_bytes=" << streamed.outputTransferBytes
                    << " frontier_upload_bytes=" << growthResult.uploadedFrontierBytes
                    << " frontier_download_bytes=" << growthResult.downloadedFrontierBytes
                    << " frontier_host_bytes=" << growthResult.maximumHostBackingBytes
                    << std::endl;

                const auto gpuPassFinished = std::chrono::steady_clock::now();
                const double gpuFinalizeSeconds = std::chrono::duration<double>(
                    gpuPassFinished - gpuFinalizeStarted).count();
                const double gpuPassSeconds = std::chrono::duration<double>(
                    gpuPassFinished - gpuPassStarted).count();
                const gpu::Phase2ProjectionTiming& projectionTiming =
                    hostProjection.timing;
                assert(projectionTiming.logicalBlocks
                    == projectionOrder.size());
                constexpr double kNanosecondsPerSecond = 1000000000.0;
                const double projectionWorkerCpuSeconds =
                    static_cast<double>(
                        projectionTiming.preflightNanoseconds
                        + projectionTiming.statementNanoseconds
                        + projectionTiming.nameNanoseconds
                        + projectionTiming.ruleStringNanoseconds
                        + projectionTiming.byteMapNanoseconds
                         + projectionTiming.reverseMapNanoseconds
                         + projectionTiming.podMapNanoseconds
                         + projectionTiming.finalNanoseconds)
                    / kNanosecondsPerSecond;
                const double projectionMergeSeconds = static_cast<double>(
                    projectionTiming.mergeNanoseconds) / kNanosecondsPerSecond;
                diagnosticsLog() << "[GPU-PHASE2] pass=" << passNo
                          << " logical_blocks=" << projectionOrder.size()
                          << " construction_workers=" << kGpuProjectionShardCount
                          << " source_statement_rows=" << projectionTiming.sourceStatementRows
                          << " uploaded_statement_rows=" << projectionTiming.retainedStatementRows
                          << " uploaded_statement_bytes=" << projectionTiming.retainedStatementRows * sizeof(IntEncodedExpr)
                          << " tasks=" << hostTasks.tasks.size()
                          << " batches=" << hostTasks.batches.size()
                          << " filter_calls="
                          << filterClassCensus.callCount
                          << " filter_classes="
                          << filterClassCensus.uniqueClassCount
                          << " filter_duplicate_calls="
                          << filterClassCensus.duplicateCallCount
                          << " filter_class_rows="
                          << filterClassCensus.uniqueExaminedRows
                          << " filter_duplicate_rows="
                          << filterClassCensus.duplicateExaminedRows
                          << " filter_max_class_multiplicity="
                          << filterClassCensus.maximumClassMultiplicity
                          << " retained_rows=" << retainedRows
                          << " maximum_frontier="
                          << growthResult.maximumFrontierCount
                          << " accepted_events="
                          << growthResult.acceptedEventCount
                          << " raw_requests=" << growthResult.rawRequestCount
                          << " evaluated_requests=" << streamed.evaluatedRequests
                          << " firing_records=" << streamed.firingRecords
                          << " generated_bytes=" << streamed.outputColumns[1]
                          << " level_values=" << streamed.outputColumns[2]
                          << " provenance_dependencies=" << streamed.outputColumns[3]
                          << " marker_keys=" << streamed.outputColumns[4]
                          << " marker_remaining_args=" << streamed.outputColumns[5]
                          << " marker_args=" << streamed.outputColumns[6];
                diagnosticsLog() << " prepare_seconds=" << gpuPreparationSeconds
                          << " projection_seconds=" << gpuProjectionSeconds
                          << " projection_worker_cpu_seconds="
                          << projectionWorkerCpuSeconds
                          << " projection_merge_seconds="
                          << projectionMergeSeconds
                          << " projection_preflight_seconds="
                          << static_cast<double>(
                                 projectionTiming.preflightNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_statement_seconds="
                          << static_cast<double>(
                                 projectionTiming.statementNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_name_seconds="
                          << static_cast<double>(
                                 projectionTiming.nameNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_name_record_seconds="
                          << static_cast<double>(
                                 projectionTiming.nameRecordNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_name_sort_seconds="
                          << static_cast<double>(
                                 projectionTiming.nameSortNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_name_rank_seconds="
                          << static_cast<double>(
                                 projectionTiming.nameRankNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_name_slot_seconds="
                          << static_cast<double>(
                                 projectionTiming.nameSlotNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_rule_seconds="
                          << static_cast<double>(
                                 projectionTiming.ruleStringNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_byte_map_seconds="
                          << static_cast<double>(
                                 projectionTiming.byteMapNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_prefix_filter_seconds="
                          << static_cast<double>(projectionTiming.prefixFilterNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_prefix_filter_keys=" << projectionTiming.prefixFilterKeys
                          << " projection_prefix_filter_predicates=" << projectionTiming.prefixFilterPredicates
                          << " projection_prefix_filter_bytes=" << projectionTiming.prefixFilterBytes
                          << " projection_reverse_map_seconds="
                          << static_cast<double>(
                                 projectionTiming.reverseMapNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_pod_map_seconds="
                          << static_cast<double>(
                                 projectionTiming.podMapNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_final_seconds="
                          << static_cast<double>(
                                 projectionTiming.finalNanoseconds)
                                 / kNanosecondsPerSecond
                          << " schedule_seconds=" << gpuScheduleSeconds
                          << " device_seconds=" << gpuDeviceSeconds
                          << " device_route_seconds=" << gpuDeviceRouteSeconds
                          << " finalize_seconds=" << gpuFinalizeSeconds
                          << " route_seconds=" << gpuPassSeconds << std::endl;
            }
            firstFamily += selectedFamilies;
            }
        }
        }
#else
        assert(phase2Backend == Phase2Backend::cpu
            && "the CUDA Phase 2 backend requires a CUDA build "
               "(the Visual Studio project, or make USE_CUDA=1)");
#endif

        // Latch this pass's doom lines (post-join relaxed loads see the final
        // CAS-min values) before the next pass's reset can erase them.
        for (std::size_t li = 0; li < M; ++li)
            if (doomLines[li].load(std::memory_order_relaxed) != kNoDoomLine)
                lbDoomed[li] = 1;

        // ---- Classify every part, single-threaded: deal a producer's stumps into
        // buckets (requeue for round 2), or keep a burst part ----
        gl::Phase2TraceScope phase2PassOutputsTrace("GL.Phase2.PassOutputs", passNo);
        std::vector<ExecTask> nextTasks;
        for (std::size_t i = 0; i < tasks.size(); ++i) {
            const ExecTask& t = tasks[i];
            const std::size_t li = t.li;
#if RT_MEASUREMENT
            // [RT phase measurement] producer and burst parts both count.
            trapPh2Ns[li] += taskNs[i];
#endif
            SealedPageSet& ps = pageStore[base + i];

            // A round-1 PRODUCER. It fired nothing (ps is empty). The processor
            // route deals its stumps here and requeues the parts for round 2; the
            // CUDA route dealt them into this pass right after the producers ran.
            if (t.produceOnly) {
                ps.freePages();
                if (phase2Backend == Phase2Backend::cpu)
                    dealProducerStumps(i, nextTasks);
                continue;
            }

            // A burst part (an unsplit LB or one expression bucket): keep it and add
            // its submatch count to the LB's split-invariant TOTAL work (the straggler
            // classifier's input); lbMaxSub tracks the busiest single part for the
            // split-ineffective report.
            if (phase2Backend == Phase2Backend::cpu
                || gpuOutputOwners[i] != 0) {
                keptParts[li].push_back(&ps);
            }
            lbTotalSub[li] += taskSubMatches[i];
            if (taskSubMatches[i] > lbMaxSub[li]) lbMaxSub[li] = taskSubMatches[i];
        }
        for (std::size_t i = 0; i < tasks.size(); ++i)
            if (stumpBound[i]) stumpPages[i].freePages();

        phase2PassOutputsTrace.finish();
        // ---- Finalize every LB that has no work left ----
        gl::Phase2TraceScope phase2FinalizeSweepTrace("GL.Phase2.FinalizeSweep", passNo);
        std::vector<char> hasNext(M, 0);
        for (const ExecTask& t : nextTasks) hasNext[t.li] = 1;
        std::vector<Memory*> toFinalize;
        std::vector<std::size_t> finalizeLi;
        for (std::size_t li = 0; li < M; ++li) {
            if (hasNext[li] || lbFinalized[li]) continue;
            lbFinalized[li] = 1;
            toFinalize.push_back(active[li]);
            finalizeLi.push_back(li);
        }
        if (!toFinalize.empty()) {
            std::atomic<std::size_t> nextLi{ 0 };
            // FINALIZE window: nextLi is a real fetch_add dispatch cursor over
            // toFinalize -- the pager tracks the finalize sweep like phases 1/3.
            gl::memory_census::markPhase(gl::PoolPhase::Phase2Finalize);
            steward->beginPhaseWindow(/*phase=*/2, &nextLi, &toFinalize, workers,
                                      lbdeload::kDeloadDirectory);
            auto finalizeWorker = [this, &toFinalize, &finalizeLi, &keptParts,
                                   &doomLines, &lbFiringRecordsCanonical,
                                   &nextLi
#if RT_MEASUREMENT
                                   , &trapFinNs
#endif
                                   ](unsigned tIdx) {
                // The finalize pool runs in parallel like phases 1/3, so publish this
                // worker's slot. The per-slot scratch / gen arenas reached deep in the
                // drains must pick THIS worker's arena, not the single reserved slot
                // every worker would otherwise share.
                gl::WorkerContext::setCoreId(static_cast<int>(tIdx));
                for (;;) {
                    std::size_t j = nextLi.fetch_add(1, std::memory_order_relaxed);
                    if (j >= toFinalize.size()) break;
                    Memory* b = toFinalize[j];
                    const std::size_t li = finalizeLi[j];
                    // Make the LB resident before finalize writes it: an active LB was
                    // released to Idle by its last executor part; an inactive one (no
                    // executor task) may be cold. The handshake reloads it.
                    steward->claimAndLoadForWork(
                        *b, /*phase=*/2, lbdeload::kDeloadDirectory);
                    // The split decision is no longer here (no cap, no escalation):
                    // it is the single-threaded end-of-iteration stats pass below.
                    // Hand the LB's sealed part sets, in part order, to the finalize
                    // (applyFiringRecords reads each chain in append order, then
                    // sorts - the merge is partition- and pass-independent). Empty for
                    // a phase-1-discharged LB: nothing deposits. The LB's final doom
                    // line rides along: when a doom trigger fired, the finalize
                    // merges only the winning part's chain. The pool has joined, so
                    // this relaxed load sees the final CAS-min value.
                    const int count = static_cast<int>(keptParts[li].size());
                    assert(count <= gl::HashRequestGenerator::kMaxSplitParts
                        && "LB part count exceeds kMaxSplitParts - raise the named "
                           "constant deliberately, never cap the split silently");
#if RT_MEASUREMENT
                    // [RT phase measurement] finalize wall time; each li
                    // is finalized by exactly one worker (race-free).
                    const auto trapT0 = std::chrono::steady_clock::now();
#endif
                    const bool firingRecordsCanonical =
                        lbFiringRecordsCanonical[li] != 0;
                    gl::Phase2TraceScope phase2LbFinalizeTrace("GL.Phase2.LBFinalize", li, gl::phase2TraceEnabled(), true);
                    performElemPhase2(
                        *b, keptParts[li].data(), count,
                        phase2Backend == Phase2Backend::cuda
                            ? kNoDoomLine
                            : doomLines[li].load(std::memory_order_relaxed),
                        firingRecordsCanonical);
#if RT_MEASUREMENT
                    trapFinNs[li] += std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - trapT0).count();
#endif
                    // Phase 2 is done with this LB: release the claim so the steward
                    // may reclaim it.
                    assert(b->stewardClaim.load(std::memory_order_relaxed)
                               == static_cast<uint8_t>(
                                      Memory::StewardClaim::WorkerOwned)
                           && "phase-2 finalize release of an LB not held WorkerOwned");
                    b->stewardClaim.store(
                        static_cast<uint8_t>(Memory::StewardClaim::Idle),
                        std::memory_order_release);
                    // The finalize task is over: hand this slot's scratch
                    // blocks back to the pool (no retention across tasks).
                    releaseSlotScratch(scratchArenas(), genScratchArenas(),
                                       ruleStagings(), tIdx);
                }
                };
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (unsigned t = 0; t < workers; ++t) pool.emplace_back(finalizeWorker, t);
            for (auto& th : pool) th.join();
            steward->endPhaseWindow();
            // Records consumed -- free the finalised LBs' pages in one sweep. After
            // this point any straggler view asserts at its access site.
            for (std::size_t j = 0; j < finalizeLi.size(); ++j)
                for (SealedPageSet* set : keptParts[finalizeLi[j]]) set->freePages();
        }
        phase2FinalizeSweepTrace.finish();

        if (nextTasks.empty()) break;
        tasks = std::move(nextTasks);
    }

    // GPU capacity projections go to the common diagnostics log so the
    // run log stays readable; one block per burst per batch.
    std::ostream& gpuCapacityLog = diagnosticsLog();
    gpuCapacityLog << "[GPU-CAPACITY] logical_blocks="
              << projectionLogicalBlocks.load(std::memory_order_relaxed)
              << " statements="
              << projectionStatements.load(std::memory_order_relaxed)
              << " maximum_statements_per_lb="
              << projectionMaximumStatements.load(std::memory_order_relaxed)
              << " name_records="
              << projectionNameRecords.load(std::memory_order_relaxed)
              << " maximum_name_records_per_lb="
              << projectionMaximumNameRecords.load(std::memory_order_relaxed)
              << " name_bytes="
              << projectionNameBytes.load(std::memory_order_relaxed)
              << " maximum_name_bytes_per_lb="
              << projectionMaximumNameBytes.load(std::memory_order_relaxed)
              << std::endl;
    const auto semanticTotal = [&](std::size_t metric) {
        return projectionSemanticTotals[metric].load(
            std::memory_order_relaxed);
    };
    const auto semanticMaximum = [&](std::size_t metric) {
        return projectionSemanticMaximums[metric].load(
            std::memory_order_relaxed);
    };
    gpuCapacityLog << "[GPU-SEMANTIC-CAPACITY] rule_string_records="
              << semanticTotal(0)
              << " maximum_rule_string_records_per_lb=" << semanticMaximum(0)
              << " rule_string_bytes=" << semanticTotal(1)
              << " maximum_rule_string_bytes_per_lb=" << semanticMaximum(1)
              << " name_slots=" << semanticTotal(2)
              << " maximum_name_slots_per_lb=" << semanticMaximum(2)
              << " byte_map_views=" << semanticTotal(3)
              << " byte_map_entries=" << semanticTotal(4)
              << " maximum_byte_map_entries_per_lb=" << semanticMaximum(4)
              << " byte_map_slots=" << semanticTotal(5)
              << " maximum_byte_map_slots_per_lb=" << semanticMaximum(5)
              << " byte_key_bytes=" << semanticTotal(6)
              << " maximum_byte_key_bytes_per_lb=" << semanticMaximum(6)
              << " blob_records=" << semanticTotal(7)
              << " maximum_blob_records_per_lb=" << semanticMaximum(7)
              << " blob_bytes=" << semanticTotal(8)
              << " maximum_blob_bytes_per_lb=" << semanticMaximum(8)
              << " reverse_map_views=" << semanticTotal(9)
              << " reverse_map_entries_upper=" << semanticTotal(10)
              << " maximum_reverse_map_entries_upper_per_lb="
              << semanticMaximum(10)
              << " reverse_map_slots_upper=" << semanticTotal(11)
              << " maximum_reverse_map_slots_upper_per_lb="
              << semanticMaximum(11)
              << " reverse_key_bytes_upper=" << semanticTotal(12)
              << " maximum_reverse_key_bytes_upper_per_lb="
              << semanticMaximum(12)
              << " reverse_owners=" << semanticTotal(13)
              << " maximum_reverse_owners_per_lb=" << semanticMaximum(13)
              << " pod_map_views=" << semanticTotal(14)
              << " pod_map_entries=" << semanticTotal(15)
              << " maximum_pod_map_entries_per_lb=" << semanticMaximum(15)
              << " pod_map_slots=" << semanticTotal(16)
              << " maximum_pod_map_slots_per_lb=" << semanticMaximum(16)
              << " pod_run_values=" << semanticTotal(17)
              << " maximum_pod_run_values_per_lb=" << semanticMaximum(17)
              << " mandatory_statement_keys=" << semanticTotal(18)
              << " maximum_mandatory_statement_keys_per_lb="
              << semanticMaximum(18)
              << " metadata_bytes=" << semanticTotal(19)
              << " maximum_metadata_bytes_per_lb=" << semanticMaximum(19)
              << std::endl;
    const auto taskTotal = [&](std::size_t metric) {
        return taskProjectionTotals[metric].load(std::memory_order_relaxed);
    };
    const auto taskMaximum = [&](std::size_t metric) {
        return taskProjectionMaximums[metric].load(std::memory_order_relaxed);
    };
    gpuCapacityLog << "[GPU-TASK-CAPACITY] tasks=" << taskTotal(0)
              << " maximum_tasks_per_part=" << taskMaximum(0)
              << " batches=" << taskTotal(1)
              << " maximum_batches_per_part=" << taskMaximum(1)
              << " terms=" << taskTotal(2)
              << " maximum_terms_per_part=" << taskMaximum(2)
              << " stumps=" << taskTotal(3)
              << " maximum_stumps_per_part=" << taskMaximum(3)
              << std::endl;
    gpuCapacityLog << "[GPU-FILTER-CAPACITY] request_calls="
              << requestGenerator.gpuRequestFilterCallsValue()
              << " producer_calls="
              << requestGenerator.gpuProducerFilterCallsValue()
              << " input_statements="
              << requestGenerator.gpuFilterInputStatementsValue()
              << " maximum_input_statements_per_call="
              << requestGenerator.gpuFilterMaximumInputStatementsValue()
              << " output_statements="
              << requestGenerator.gpuFilterOutputStatementsValue()
              << " maximum_output_statements_per_call="
              << requestGenerator.gpuFilterMaximumOutputStatementsValue()
              << std::endl;
    gpuCapacityLog << "[GPU-GROW-CAPACITY]";
    for (std::size_t depth = 0; depth < kGpuGrowDepthCount; ++depth) {
        gpuCapacityLog << " attempts_d" << depth << "="
                  << gpuGrowAttemptsByDepth[depth].load(std::memory_order_relaxed)
                  << " frontier_d" << depth << "="
                  << gpuGrowFrontierByDepth[depth].load(std::memory_order_relaxed)
                  << " subkeys_d" << depth << "="
                  << gpuGrowSubkeysByDepth[depth].load(std::memory_order_relaxed)
                  << " requests_d" << depth << "="
                  << gpuGrowRequestsByDepth[depth].load(std::memory_order_relaxed)
                  << " producer_attempts_d" << depth << "="
                  << gpuProducerAttemptsByDepth[depth].load(
                         std::memory_order_relaxed)
                  << " producer_survivors_d" << depth << "="
                  << gpuProducerSurvivorsByDepth[depth].load(
                         std::memory_order_relaxed);
    }
    gpuCapacityLog << std::endl;
    const auto evaluationTotal = [&](std::size_t metric) {
        return gpuEvaluationTotals[metric].load(std::memory_order_relaxed);
    };
    const auto evaluationMaximum = [&](std::size_t metric) {
        return gpuEvaluationMaximums[metric].load(std::memory_order_relaxed);
    };
    gpuCapacityLog << "[GPU-EVAL-CAPACITY] requests=" << evaluationTotal(0)
              << " dependency_pass_requests=" << evaluationTotal(1)
              << " reverse_owners=" << evaluationTotal(2)
              << " maximum_reverse_owners_per_request="
              << evaluationMaximum(0)
              << " candidate_owners=" << evaluationTotal(3)
              << " maximum_candidate_owners_per_request="
              << evaluationMaximum(1)
              << " encoded_hits=" << evaluationTotal(4)
              << " local_values=" << evaluationTotal(5)
              << " maximum_local_values_per_hit=" << evaluationMaximum(2)
              << " head_records=" << evaluationTotal(6)
              << " marker_records=" << evaluationTotal(7)
              << " demand_records=" << evaluationTotal(8)
              << " generated_bytes=" << evaluationTotal(9)
              << " level_values=" << evaluationTotal(10)
              << " origin_dependencies=" << evaluationTotal(11)
              << " marker_keys=" << evaluationTotal(12)
              << " marker_remaining_args=" << evaluationTotal(13)
              << " marker_args=" << evaluationTotal(14)
              << std::endl;

    // ---- End-of-iteration straggler classification (the split TRIGGER) ----
    // Set each LB's split for the NEXT iteration from this iteration's completed,
    // deterministic work totals. work(L) = sum of L's parts' submatch counts
    // (split-invariant, D-117). A straggler is an LB whose work exceeds the ideally-
    // balanced per-core load T / logicalCores (so it alone leaves cores idle) AND
    // clears the setup break-even min_split_work; it runs as logicalCores expression
    // buckets next iteration, every other LB unsplit. Integer arithmetic only, over a
    // fixed-order single-threaded sweep -> the split set is a deterministic function
    // of proof state (two runs stay byte-identical). Recomputed every iteration, so an
    // LB whose work falls back below the bar returns to unsplit. A doomed LB
    // (burst early-exit) contributes NOTHING: its tally was cut at a
    // timing-dependent overshoot, so folding it into T (or classifying it)
    // would make the statistic — and any borderline straggler verdict —
    // a race outcome.
    if (mainPath) {
        int64_t T = 0;
        for (std::size_t li = 0; li < M; ++li)
            if (!lbDoomed[li]) T += lbTotalSub[li];
        const int64_t fairShare = T / static_cast<int64_t>(logicalCores);
        // Split reports go to the common diagnostics log so the run log
        // stays readable; one line per onset / self-control hit.
        std::ostream& splitLog = diagnosticsLog();
        for (std::size_t li = 0; li < M; ++li) {
            // An early-exited LB is discharging out of the grid: its cut
            // tally is not comparable work and it never bursts again — the
            // unsplit default is its final split state.
            if (lbDoomed[li]) {
                active[li]->numberOfParts = 1;
                continue;
            }
            const bool straggler = gl::Phase2Executor::isStraggler(
                lbTotalSub[li], T, static_cast<int>(logicalCores),
                static_cast<int64_t>(parameters.min_split_work));
            const bool wasUnsplit = active[li]->numberOfParts == 1;
            active[li]->numberOfParts =
                straggler ? static_cast<int>(logicalCores) : 1;
            // Split-onset report: printed once per unsplit -> split transition so
            // the run log shows which LBs the trigger promoted. Pure observation ->
            // determinism intact.
            if (straggler && wasUnsplit)
                splitLog << "[SPLIT] straggler: work=" << lbTotalSub[li]
                         << " T=" << T
                         << " parts=" << active[li]->numberOfParts
                         << " lb=" << active[li]->exprKey() << "\n";
            // Self-control report: a split LB whose busiest bucket is still a global
            // outlier has an irreducibly-serial core (a "runs and runs" induction LB)
            // that bucketing cannot subdivide. Pure observation -> determinism intact.
            if (straggler && lbMaxSub[li] > fairShare)
                splitLog << "[SPLIT] ineffective: work=" << lbTotalSub[li]
                         << " maxPart=" << lbMaxSub[li]
                         << " buckets=" << lbBuckets[li]
                         << " stumps=" << lbStumps[li]
                         << " fairShare=" << fairShare
                         << " lb=" << active[li]->exprKey() << "\n";
        }
    }
#if RT_MEASUREMENT
    // [RT phase measurement] carry the split-invariant work totals out
    // to the post-phase-3 report (lbTotalSub is scoped to this block).
    for (std::size_t li = 0; li < M; ++li) trapWork[li] = lbTotalSub[li];
#endif
    if (gl::phase2TraceEnabled())
        for (std::size_t li = 0; li < M; ++li)
            diagnosticsLog() << "[PHASE2-LB-WORK] call=" << phase2TraceCall
                << " li=" << li << " work=" << lbTotalSub[li] << '\n';

    }  // active non-empty

    phase2Trace.finish();
    const double phase2IterationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase2Started).count();
    phase2CumulativeSeconds += phase2IterationSeconds;
    ++phase2MeasuredIterations;
    diagnosticsLog() << "[PHASE2-TIMING] backend="
              << (phase2Backend == Phase2Backend::cuda ? "cuda" : "cpu")
              << " iteration_seconds=" << phase2IterationSeconds
              << " cumulative_seconds=" << phase2CumulativeSeconds
              << " iterations=" << phase2MeasuredIterations
              << " active_logical_blocks=" << active.size() << std::endl;
#if RT_MEASUREMENT
    gl::rt_tracker::addRtPhaseWallSeconds(2, phase2IterationSeconds);
#endif

#ifdef USE_MIMALLOC
    // Return the heap Phase 2 freed (the host projection and its scratch,
    // allocated and freed on this thread) to the operating system before
    // phase 3, instead of letting the allocator keep it committed and resident
    // until the next Phase 2 reuses it (D-426).
    mi_collect(true);
#endif

    return mainPath;
}

#if RT_MEASUREMENT
thread_local bool Phase2Executor::g_crtArmed = false;
thread_local NameId Phase2Executor::g_crtPreorderId = 0;
thread_local NameId Phase2Executor::g_crtFiveId = 0;
std::atomic<int64_t> Phase2Executor::g_crtAttempts{ 0 };
std::atomic<int64_t> Phase2Executor::g_crtAttemptsUnlinked{ 0 };
std::atomic<int64_t> Phase2Executor::g_crtAccepted{ 0 };
std::atomic<int64_t> Phase2Executor::g_crtAcceptedUnlinked{ 0 };
#endif

#if RT_MEASUREMENT
/// @brief Clear the target block's observation counters at burst entry.
/// @details Performs the four relaxed atomic stores in their order.
/// @return No value.
void Phase2Executor::resetTargetCounters() {
        g_crtAttempts.store(0, std::memory_order_relaxed);
        g_crtAttemptsUnlinked.store(0, std::memory_order_relaxed);
        g_crtAccepted.store(0, std::memory_order_relaxed);
        g_crtAcceptedUnlinked.store(0, std::memory_order_relaxed);
}
/// @brief Read the target block accepted observation counter.
/// @details Uses the relaxed atomic load at the diagnostic output site.
/// @return Current observation count.
int64_t Phase2Executor::targetAccepted() { return g_crtAccepted.load(std::memory_order_relaxed); }
/// @brief Read the target block acceptedunlinked observation counter.
/// @details Uses the relaxed atomic load at the diagnostic output site.
/// @return Current observation count.
int64_t Phase2Executor::targetAcceptedUnlinked() { return g_crtAcceptedUnlinked.load(std::memory_order_relaxed); }
/// @brief Read the target block attempts observation counter.
/// @details Uses the relaxed atomic load at the diagnostic output site.
/// @return Current observation count.
int64_t Phase2Executor::targetAttempts() { return g_crtAttempts.load(std::memory_order_relaxed); }
/// @brief Read the target block attemptsunlinked observation counter.
/// @details Uses the relaxed atomic load at the diagnostic output site.
/// @return Current observation count.
int64_t Phase2Executor::targetAttemptsUnlinked() { return g_crtAttemptsUnlinked.load(std::memory_order_relaxed); }
#endif

}
