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
#include "proof_barrier.hpp"
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
#endif
#include <iostream>
#include <fstream>
#include <array>
#include <numeric>
#include <algorithm>
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
/// @brief Bind the proof barrier operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param body Borrowed Memory interface or state; must outlive this component.
/// @param updateGlobalTuples Borrowed std::vector interface or state; must outlive this component.
/// @param inductionMemoryBlocks Borrowed std::vector interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param mailLog Borrowed MailLog interface or state; must outlive this component.
/// @param rollingMailHistoryEnabled Borrowed bool interface or state; must outlive this component.
/// @param peakDeloadableMailOutBytes Borrowed int64_t interface or state; must outlive this component.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param dischargeScratch Borrowed LbArena interface or state; must outlive this component.
/// @param pendingCompactionQueue Borrowed std::vector interface or state; must outlive this component.
/// @param pendingDischarge Borrowed std::vector interface or state; must outlive this component.
/// @param steward Borrowed std::vector interface or state; must outlive this component.
/// @param structuralCompiler Borrowed StructuralCompiler interface or state; must outlive this component.
/// @param orRuleCompiler Borrowed OrRuleCompiler interface or state; must outlive this component.
/// @param deactivationSweepOwed Borrowed bool interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param scopeLifecycle Borrowed ScopeLifecycle interface or state; must outlive this component.
/// @param blockMailProcessor Borrowed BlockMailProcessor interface or state; must outlive this component.
/// @param anchorScopeQueries Borrowed AnchorScopeQueries interface or state; must outlive this component.
/// @param theoremPublisher Borrowed TheoremPublisher interface or state; must outlive this component.
/// @param orTheoremBuilder Borrowed OrTheoremBuilder interface or state; must outlive this component.
/// @param orEliminationBuilder Borrowed OrEliminationBuilder interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
ProofBarrier::ProofBarrier(const ProverParameters& parameters, Memory& body, std::vector< std::tuple<int, uint64_t, int>>& updateGlobalTuples, std::vector<Memory*>& inductionMemoryBlocks, bool& ceFilteringActive, const CompiledDefinitionRegistry& definitions, MailLog& mailLog, bool& rollingMailHistoryEnabled, int64_t& peakDeloadableMailOutBytes, SimpleMapStore& simpleMapStore, LbArena& dischargeScratch, std::vector<std::tuple<std::string, int, int>>& pendingCompactionQueue, std::vector<Memory*>& pendingDischarge, std::unique_ptr<MemorySteward>& steward, StructuralCompiler& structuralCompiler, OrRuleCompiler& orRuleCompiler, bool& deactivationSweepOwed, AdmissionMemory& admissionMemory, ProofHistory& proofHistory, ScopeLifecycle& scopeLifecycle, BlockMailProcessor& blockMailProcessor, AnchorScopeQueries& anchorScopeQueries, TheoremPublisher& theoremPublisher, OrTheoremBuilder& orTheoremBuilder, OrEliminationBuilder& orEliminationBuilder) : parameters(parameters), body(body), updateGlobalTuples(updateGlobalTuples), inductionMemoryBlocks(inductionMemoryBlocks), ceFilteringActive(ceFilteringActive), definitions(definitions), mailLog(mailLog), rollingMailHistoryEnabled(rollingMailHistoryEnabled), peakDeloadableMailOutBytes(peakDeloadableMailOutBytes), simpleMapStore(simpleMapStore), dischargeScratch(dischargeScratch), pendingCompactionQueue(pendingCompactionQueue), pendingDischarge(pendingDischarge), steward(steward), structuralCompiler(structuralCompiler), orRuleCompiler(orRuleCompiler), deactivationSweepOwed(deactivationSweepOwed), admissionMemory(admissionMemory), proofHistory(proofHistory), scopeLifecycle(scopeLifecycle), blockMailProcessor(blockMailProcessor), anchorScopeQueries(anchorScopeQueries), theoremPublisher(theoremPublisher), orTheoremBuilder(orTheoremBuilder), orEliminationBuilder(orEliminationBuilder) {}

//#pragma optimize("", off)


// loadFactsForCEFiltering(), addConjectureForCEFiltering(),
// releaseCEBatchMemory() — moved to filter.cpp.


// filterConjecturesWithCE() — moved to filter.cpp.
void ProofBarrier::activateZeroCondition(Memory& memoryBlock) {
    std::vector<std::string> args0 = ce::getArgs(memoryBlock.exprKey());
    std::string zeroArgName = anchorScopeQueries.findZeroArgName(memoryBlock);
    std::string eqExpr = std::string("(=[s(") + args0[0] + ")," + zeroArgName + "])";
    if (memoryBlock.parentMemory != NULL) {
        Memory* zeroChild = simpleMapStore.findChild(memoryBlock.parentMemory, eqExpr);
        if (zeroChild) {
            // The wake target is a block parked at birth that never
            // entered an active snapshot — waking a DISCHARGED block
            // (one that ran and deactivated) would resurrect frozen
            // content (I-112).
            assert(!zeroChild->dischargedForever
                && "activateZeroCondition on a discharged LB - "
                   "deactivation is permanent");
            zeroChild->isActive = true;
            // WAKE DOOR 6 / case F (D-194): this births an
            // induction zero block into the active set bypassing every mailbox,
            // so it must be marked dirty to be swept its first burst.
            zeroChild->hasWork = true;
            // First active burst processes a full backlog with no prior submatch
            // history — split it preemptively this iteration (D-201).
            zeroChild->justActivated = true;
        }
    }
}
/// @brief The step-win seam: settle every step block that proved its goal
///        in the finished burst (D-517).
///
/// @details
/// Runs inside `finish` where the zero blocks used to be woken, after the
/// mail commit and before the theorem drains and the discharge loop. First
/// every winning triad whose promotion waits for its typing fact is
/// re-checked (`ScopeLifecycle::retryWaitingPromotions`) and, once the fact
/// arrived, re-queued into `updateGlobalTuples` for the drain that follows.
/// Then the closed step blocks are sorted in the total order of their full
/// block chains (`TheoremPublisher::compareProducerChains`) and settled one
/// by one with this burst's closure levels
/// (`ScopeLifecycle::settleStepClosure`); every closure except a tie loser
/// wakes its zero block through `activateZeroCondition`. Each step-win event
/// writes one `[STEP-WIN]` line to the run diagnostics log. In compressor
/// mode, which reuses this prover on blocks without step blocks, the seam
/// does nothing.
///
/// @return No value.
/// @invariant Single-threaded post-join seam (I-28); the steward is live.
/// @see ScopeLifecycle::settleStepClosure, ScopeLifecycle::retryWaitingPromotions.
void ProofBarrier::settleStepWins() {
    // The compressor reruns this prover on its own flat blocks, which hold
    // no step blocks; the main run's step wins stay untouched.
    if (parameters.compressor_mode) {
        assert(inductionMemoryBlocks.empty()
            && "settleStepWins: a step block closed in compressor mode");
        return;
    }
    assert(steward && "settleStepWins: the step-win seam needs the live steward");
    using Event = ScopeLifecycle::StepWinEvent;
    const auto logEvent = [](const Event& event, const Memory* stepBlock) {
        const char* kind = "unaffected";
        switch (event.kind) {
        case Event::Kind::Winner: kind = "winner"; break;
        case Event::Kind::TieLoser: kind = "tie_loser"; break;
        case Event::Kind::PromotionRetried: kind = "promotion_retried"; break;
        case Event::Kind::Uncovered: kind = "uncovered"; break;
        case Event::Kind::Unaffected: kind = "unaffected"; break;
        }
        std::ostringstream line;
        line << "[STEP-WIN] " << kind
             << " variable=" << event.inductionVariable
             << " induction=" << (event.strongInduction ? "strong" : "weak")
             << " siblings_off=" << event.siblingsSwitchedOff;
        if (event.kind == Event::Kind::Uncovered) {
            line << " missing_levels=0x" << std::hex << event.missingLevels << std::dec;
        }
        if (stepBlock != nullptr) line << " step=" << stepBlock->exprKey();
        line << " conjecture=" << event.conjecture << "\n";
        diagnosticsLog() << line.str() << std::flush;
    };

    for (const Event& event : scopeLifecycle.retryWaitingPromotions(*steward)) {
        // The theorem drain right behind this seam runs the promotion again;
        // the triad's level union is already recorded.
        updateGlobalTuples.emplace_back(event.retryAuxyIndex, uint64_t(0), 0);
        logEvent(event, nullptr);
    }

    std::sort(inductionMemoryBlocks.begin(), inductionMemoryBlocks.end(),
        [](const Memory* a, const Memory* b) {
            return TheoremPublisher::compareProducerChains(a, b) < 0;
        });
    for (Memory* stepBlock : inductionMemoryBlocks) {
        const Event event = scopeLifecycle.settleStepClosure(*stepBlock, *steward, updateGlobalTuples);
        if (event.kind != Event::Kind::TieLoser) {
            activateZeroCondition(*stepBlock);
        }
        if (event.kind != Event::Kind::Unaffected) {
            logEvent(event, stepBlock);
        }
    }
}
/// @brief The useless-step seam: switch off every weak step block whose
///        usefulness test caught its hypothesis head in the finished burst
///        (D-521).
///
/// @details
/// Runs inside `finish` right after `settleStepWins`. Collects the blocks of
/// @p bodies whose test is `Caught` (`InductionUsefulness::settle` set it in
/// standard processing), sorts them in the total order of their full block
/// chains (`TheoremPublisher::compareProducerChains`) and switches each off
/// (`ScopeLifecycle::switchOffUselessStep`). Each event writes one
/// `[STEP-USELESS]` line to the run diagnostics log. Reads only the
/// never-deloaded block slab, so it loads no block.
///
/// @param bodies The complete burst grid.
/// @return No value.
/// @invariant Single-threaded post-join seam (I-28).
/// @see ScopeLifecycle::switchOffUselessStep, InductionUsefulness::settle.
void ProofBarrier::settleUselessSteps(const std::vector<Memory*>& bodies) {
    std::vector<Memory*> caught;
    for (Memory* lb : bodies) {
        if (lb != nullptr && lb->usefulnessTest == Memory::UsefulnessTest::Caught) caught.push_back(lb);
    }
    std::sort(caught.begin(), caught.end(),
        [](const Memory* a, const Memory* b) {
            return TheoremPublisher::compareProducerChains(a, b) < 0;
        });
    for (Memory* stepBlock : caught) {
        const ScopeLifecycle::UselessStepEvent event = scopeLifecycle.switchOffUselessStep(*stepBlock);
        std::ostringstream line;
        line << "[STEP-USELESS] variable=" << event.inductionVariable
             << " switched_off=" << (event.switchedOff ? 1 : 0)
             << " step=" << stepBlock->exprKey()
             << " conjecture=" << event.conjecture << '\n';
        diagnosticsLog() << line.str() << std::flush;
    }
}
/// @brief Commit mail and drain the completed burst's proof results.
/// @details Preserves the post-phase-three order of mail commitment, theorem publication, deactivation, OR construction, compaction, discharge and prefetch. Existing recipients, queues and storage owners remain unchanged.
/// @param bodies Complete burst logical-block grid.
/// @param active Logical blocks selected at the burst boundary.
/// @param workers Existing worker count used by the steward windows.
/// @param seamScratchSlot Reserved single-threaded scratch slot selected at burst entry.
/// @return No value.
/// @invariant Called only after every phase-three worker has joined.
void ProofBarrier::finish(const std::vector<Memory*>& bodies, const std::vector<Memory*>& active, unsigned workers, unsigned seamScratchSlot) {
    int64_t deloadableMailOutBytes = 0;
    for (const Memory* lb : bodies) {
        if (lb != nullptr) deloadableMailOutBytes += lb->mailOutLiveBytes;
    }
    peakDeloadableMailOutBytes =
        std::max(peakDeloadableMailOutBytes, deloadableMailOutBytes);

    // A grid with no initially dormant LB has no future catch-up reader. Every
    // LB that can consume the currently retained window ran phase 1 before this
    // post-phase-3 join, so release the delivered blob/ref pages and their
    // global-id dictionary before the next commit opens a fresh window. Every
    // active LB cleared mailIn after its phase-1 absorb; mailOut still carries
    // sender-local ids, so no live global mail id crosses this seam. The
    // grid-build policy is fixed for the whole execution batch: any initially
    // dormant LB keeps full history and the matching global-id space even if it
    // later activates (I-162).
    if (!parameters.compressor_mode && !ceFilteringActive
        && rollingMailHistoryEnabled) {
        for (const Memory* lb : bodies) {
            assert(lb == nullptr || lb->mailIn.empty()
                && "mail interner retirement requires every mailIn to be empty");
        }
        this->mailLog.retireDeliveredBatches();
        resetMailInterner();
    }

    // New mail system (D-137): the commit barrier. Each LB's
    // mailOut -- this step's fillMailOut output, plus (for the root) the
    // post-join updateGlobal*/D-76 drains from the PREVIOUS iteration -- becomes
    // one appended batch in its log; mailOut is then cleared. Single-threaded
    // (post pool.join), so the logs are frozen for the next iteration's parallel
    // phase-1 pulls. Replaces the per-step smashMail.
    //
    // Iterates `bodies`, not just the active set: the post-join updateGlobal* /
    // D-76 drains below append to the ROOT's mailOut after this barrier, and the
    // root can itself deactivate (the deactivation predicate fires once all its
    // children are done), so an active-only barrier could strand the root's
    // proven-rule mail. The empty-mailOut skip keeps inactive / produced-nothing
    // LBs from accruing empty batches. Anything written to an LB's mailOut AFTER
    // this point rides the NEXT iteration's commit -- a one-iteration delay,
    // matching the old post-smashMail sendMail timing. The compressor's LBs are
    // flat and never registered, so it is gated out.
    if (!parameters.compressor_mode && !ceFilteringActive) {
        // BARRIER SEAM WINDOW A (D-196): the commit sweep
        // is a linear walk over `bodies`, so it gets a dispatch-style pager
        // window like a phase — the planner prefetches ahead of the commit
        // cursor and drains behind it, so the sweep's reloads no longer
        // accumulate monotonically with the planner parked (the 4 GiB
        // between-iterations wall). Each committed LB goes through the
        // UNIFORM DOOR (claimAndLoadForWork -> commit -> release Idle): the
        // claim hold keeps the executors off the LB while commit serializes
        // its private deloadable mailOut ids, and the claim word ends
        // consistent — a seam reload can never leave Dumped-but-resident.
        //
        // Discharge distinction: an LB that merely DEACTIVATED this iteration
        // (isActive false; dischargedForever not yet set — the discharge
        // block runs later in this same barrier) is legal to door-reload for
        // its final commit; a PREVIOUSLY-discharged LB is structurally
        // skipped by the empty-mailOut check — its last commit (this sweep,
        // the barrier before its discharge) cleared mailOut, and nothing
        // appends to its mailOut afterwards (fillMailOut needs a phase run).
        // The two exceptions are the post-join drains' mail targets: the
        // ROOT (updateGlobal* / D-76 drains) and, in incubator mode, the
        // anchor LB (the D-332 nested compacts). Both may receive mail after
        // their own discharge, so the discharge block below never hands them
        // to the drain: they stay resident and this claim finds them loaded.
        std::atomic<std::size_t> commitCursor{ 0 };
        steward->beginPhaseWindow(/*phase=*/4, &commitCursor, &bodies, workers,
                                  lbdeload::kDeloadDirectory);
        for (std::size_t bi = 0; bi < bodies.size(); ++bi) {
            commitCursor.store(bi, std::memory_order_relaxed);
            Memory* lb = bodies[bi];
            if (!lb) continue;
            if (!lb->mailOutPending) continue;
            steward->claimAndLoadForWork(*lb, /*phase=*/4,
                                         lbdeload::kDeloadDirectory);
            assert(!lb->mailOut.empty()
                && "mailOutPending set for an empty outgoing mailbox");
            this->mailLog.commit(lb, lb->mailOut);
            lb->clearMailOut();
            lb->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        }
        steward->endPhaseWindow();
    }

    // Sort thread-collected vectors for deterministic processing order (the
    // closed step blocks are sorted by settleStepWins).
    std::sort(updateGlobalTuples.begin(), updateGlobalTuples.end());

    // BARRIER SEAM WINDOW B (D-196): the post-join drains
    // below (deferred ancestor admissions, updateGlobal, updateGlobalSingleChapter,
    // the compaction flush) deposit into RANDOM-ACCESS recipient LBs (per-
    // theorem accessMemory tree lookups) — there is no sweep order to
    // register, so the window's cursor is PINNED AT 0 over `active`. That
    // pin is the honest Belady origin between iterations: the next use of
    // active[i] IS position i (the next kernel re-sweeps from 0), so the
    // planner keeps the next iteration's head resident (pre-warming phase 1)
    // and evicts everything else farthest-first WHILE the drains churn —
    // bounding the residency that previously grew monotonically with no
    // window open. Each drain reload goes through the uniform door
    // (claimAndLoadForWork -> write -> release Idle): recipients are
    // WorkerOwned across their deposit (never evicted mid-write), evictable
    // the moment they release (the deposit lives in the arena and rides the
    // raw image — recipient churn under pressure is correct behavior).
    {
        std::atomic<std::size_t> drainCursor{ 0 };
        steward->beginPhaseWindow(/*phase=*/4, &drainCursor, &active, workers,
                                  lbdeload::kDeloadDirectory);

    // Wake the zero blocks of the closed step blocks and switch off the
    // other proof blocks of each conjecture whose step won (D-517).
    settleStepWins();
    // Switch off the weak step blocks whose usefulness test caught the
    // hypothesis head (D-521).
    settleUselessSteps(bodies);

    // I-28 detect-and-defer (trial): replay the cross-LB ancestor admission
    // seeds the parallel phase-1/phase-3 workers deferred, single-threaded here.
    admissionMemory.drainDeferredAncestorAdmissions();

    for (const std::tuple<int, uint64_t, int>& t : updateGlobalTuples)
    {
        theoremPublisher.updateGlobal(std::get<0>(t), std::get<1>(t), std::get<2>(t));
    }

    // Replay the proven single-chapter theorems staged on updateGlobalSingleChapterPages, sorted
    // (the former std::sort over updateGlobalSingleChapterTuples + the replay loop).
    theoremPublisher.drainUpdateGlobalSingleChapter();
    // Settle the global prune owed by the deactivateUnnecessary walks of the
    // two theorem drains above (updateGlobal, updateGlobalSingleChapter): one
    // tree-wide sweep for all of them, before the twin / subtree deactivation
    // drains and the barrier discharge read isActive. An iteration whose
    // drains ran no complete walk owes no sweep, exactly as before
    // (D-451).
    if (deactivationSweepOwed) {
        deactivationSweepOwed = false;
        scopeLifecycle.deactivateRecursively();
    }
    // Retract already-submitted rows whose producer turned out
    // unintentionally contradictory this iteration (the vacuous-premise
    // flag): the emission gates cover new submissions, this sweep covers
    // prior-iteration ones.
    theoremPublisher.revertVacuousGlobalTheorems();
    // Twin deactivation rides directly behind the theorem drain so every
    // firing/closure of this iteration has staged its twins first.
    scopeLifecycle.drainPendingTwinDeactivations();
    // Vacuous-premise subtree deactivation: every LB whose main turned
    // contradictory this iteration takes its whole subtree out of the
    // search — nothing sound and non-vacuous can be proved below an
    // inconsistent premise set.
    scopeLifecycle.drainSubtreeDeactivations();

    // In-run OR-theorem construction (A16 Phase 2,
    // D-266): fold this iteration's new proved
    // rows into or theorems and queue their implication-compact broadcast
    // on pendingCompactionQueue so the drain below ships them with this
    // iteration's batch. Placed after the vacuous retraction sweep so a
    // row retracted this iteration is never scanned as an or source.
    orTheoremBuilder.constructOrTheoremsInRun();

    // Pre-split merge directly after the or seam: an or theorem minted
    // this window can license a merge in the same window, and the merged
    // theorem's broadcast rides the compaction drain just below.
    orEliminationBuilder.constructOrEliminationInRun();

    // Deferred compaction drain (D-76,
    // Option A). Single-threaded, post-pool.join(): sorting makes the
    // implication<N> allocation a deterministic function of the broadcast
    // set (by string), so compileImplicationToCompact's global implCounter /
    // compiledExpressions / compilationReuse.entries() mutation is injective and
    // reproducible instead of the racy parallel allocation that violated
    // I-28 and broke determinism. Each compact form is then merged into the
    // root's mailOut, shipped by the MailLog commit barrier -- a two-iteration
    // delay vs the producing burst, matching the old post-smashMail timing.
    std::sort(pendingCompactionQueue.begin(), pendingCompactionQueue.end());
    {
        std::map<int, Mail> compactionMailByCore;
        // Incubator shape routing (D-332):
        // the root's log is read by the anchor LB alone, so a nested rule
        // (anchorOnlyRule false) is ALSO merged into the anchor LB's mailOut --
        // the anchor's log reaches every LB below it, the root's log reaches
        // the anchor LB itself.
        std::map<int, Mail> nestedMailByCore;
        Memory* const nestedStore = (parameters.incubator_mode && !pendingCompactionQueue.empty())
            ? blockMailProcessor.incubatorAnchorLb() : nullptr;
        for (const std::tuple<std::string, int, int>& e : pendingCompactionQueue)
        {
            const std::string& original = std::get<0>(e);
            const int cId = std::get<2>(e);
            const bool nested = nestedStore != nullptr && !blockMailProcessor.anchorOnlyRule(original);
            const std::string compactImpl = structuralCompiler.compileImplicationToCompact(original);
            // Level set MUST be empty. The implication rule always
            // deposits std::set<int>() for its levels.
            // The receiver-side addToHashMemory call propagates these levels
            // into the rule's intStatementLevelsMap entry, and when the rule
            // fires, the derived statement's levels are computed as the union
            // of the rule levels and the matching premises' levels. The
            // empty set keeps the derived levels equal to the premise-side
            // union only, which is what the allLevelsInvolved registration
            // verdict computed in prover.hpp::dischargeToBeProved expects
            // (size == memoryBlock.level + 1). Filling compactLevels with
            // {0, 1, ..., kySize} would inject an extra level into every
            // derived statement that fired against a mail-arrived rule,
            // making size > level + 1 so the sealed verdict turns false and
            // the drain refuses appendGlobalTheorem — the goal still closes
            // (closure is level-free), but the theorem is silently lost.
            std::set<int> compactLevels;
            ExpressionWithValidity compactEv(compactImpl, "main");
            Mail& m = compactionMailByCore[cId];
            m.statements.insert(std::make_pair(compactEv, compactLevels));
            if (parameters.trackHistory) {
                // Cite the binary's CANONICAL reconstruction in the
                // `compilation` row's rest[0], not the input `original`.
                // compileImplicationToCompact dedups alpha-equivalent
                // inputs (two distinct theorem reformulations whose only
                // difference is the order of two co-bound variables in
                // a `>[w_i,w_j]` group, e.g. `[w2,w4,w5][w3,w4,w6]
                // [w1,w5,w6]` vs `[w2,w5,w4][w3,w5,w6][w1,w4,w6]`) to
                // the SAME implication<N> name. The binary stores only
                // the first-seen body; any later input whose binder
                // ordering differs from the binary registered form will
                // be alpha-equivalent but NOT structurally identical.
                //
                // The verifier's `check_compilation` reconstructs the
                // implication from the binary's `elements` via
                // `_build_implication_from_elements` (the Python mirror
                // of reconstructImplicationFullBind) and compares with
                // rest[0] modulo `_normalize_with_unchangeables`. The
                // normalizer rewrites BOTH binder content and arg
                // content to a fresh v<k> scheme, so any alpha-equivalent
                // form WHOSE BINDER LIST MATCHES THE BODY'S FIRST-
                // APPEARANCE ORDER will pass. Citing `original` as-is
                // can produce a body whose binder list `>[w_i,w_j]` is
                // inconsistent with the first-appearance order of w_i
                // and w_j in the body (the inner premise reads
                // `in3[w2,w_j,w_i,+]` so w_j appears before w_i but the
                // binder names them w_i first) — the resulting
                // normalized strings differ in `>[v_k,v_l]` vs
                // `>[v_l,v_k]` and the structural check fails.
                //
                // The canonical reconstruction below is the same form
                // the verifier rebuilds: take the binary's stored
                // elements (placeholder-numeric form `1..K`),
                // reconstruct via reconstructImplicationFullBind with
                // its own first-appearance binder rule. The resulting
                // numeric body is structurally identical to what
                // check_compilation's _try_expand builds, so the
                // normalize-then-compare matches by construction.
                // Citing the input `original` instead would risk a
                // binder-order misalignment that the verifier rejects.
                const std::string compactCore = extractExpressionUniversalSpan(StrSpan(compactImpl)).toStdString();
                auto cit = this->definitions.entries().find(compactCore);
                assert(cit != this->definitions.entries().end()
                    && "deferred-compaction drain: compact form must resolve to a compiledExpressions entry");
                const std::vector<std::string>& elems = cit->second.elements;
                assert(!elems.empty()
                    && "deferred-compaction drain: implication entry must have at least one element (head)");
                std::vector<std::string> canonicalKey(elems.begin(), elems.end() - 1);
                const std::string& canonicalHead = elems.back();
                const std::string canonicalOriginal = gl::ImplicationBuilder::reconstructImplicationFullBind(canonicalKey, canonicalHead);
                proofHistory.addOrigin(m.exprOriginMap, compactEv,
                    std::make_pair("compilation", std::vector<ExpressionWithValidity>{ ExpressionWithValidity(canonicalOriginal, "main") }),
                    (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }
            if (nested) {
                Mail& nm = nestedMailByCore[cId];
                nm.statements.insert(std::make_pair(compactEv, compactLevels));
                if (parameters.trackHistory) {
                    // The same rows as the root batch (keyed by the compact).
                    const auto oit = m.exprOriginMap.find(compactEv);
                    assert(oit != m.exprOriginMap.end()
                        && "deferred-compaction drain: the compact's compilation row must exist");
                    for (const OriginLine& line : oit->second) {
                        proofHistory.addOrigin(nm.exprOriginMap, compactEv, line,
                            (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    }
                }
            }
        }
        if (!compactionMailByCore.empty()) {
            steward->claimAndLoadForWork(this->body, /*phase=*/4,
                                         lbdeload::kDeloadDirectory);
        }
        for (std::map<int, Mail>::iterator mit = compactionMailByCore.begin(); mit != compactionMailByCore.end(); ++mit) {
            // New mail system (D-137): the D-76 compact
            // (implication<N>[...]) forms target the root's descendants, so merge
            // every per-core batch into the root's mailOut. The NEXT commit
            // barrier ships them into the root's log; descendants pull and recover
            // the rule via status=3 disintegration. This preserves the old
            // post-smashMail TWO-iteration compact-implication latency -- the
            // drain runs after the barrier, so its mail rides the next barrier and
            // is pulled the cycle after, exactly the old timing (G-49:
            // contradiction LBs are sensitive to this delay; it is unchanged). The
            // per-core key no longer routes anything; the assert stays as a
            // tripwire for a corrupt core id (Rule 19).
            assert(mit->first >= -1);
            mergeBatchIntoMailOut(mit->second, this->body);
        }
        if (!compactionMailByCore.empty()) {
            this->body.stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        }
        // Shape-routed nested compacts: the anchor LB's mailOut too (its log
        // reaches every LB below it), under the anchor's claim.
        if (!nestedMailByCore.empty()) {
            assert(nestedStore != nullptr);
            steward->claimAndLoadForWork(*nestedStore, /*phase=*/4,
                                         lbdeload::kDeloadDirectory);
            for (std::map<int, Mail>::iterator mit = nestedMailByCore.begin(); mit != nestedMailByCore.end(); ++mit) {
                mergeBatchIntoMailOut(mit->second, *nestedStore);
            }
            nestedStore->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        }
    }
    pendingCompactionQueue.clear();

    // The compaction drain registers fresh compacts; close the reduced-or
    // closure again so any or it minted is covered before the next
    // iteration's bursts (idempotent; D-268).
    orRuleCompiler.preMintReducedOrs();

        // Close the barrier seam window: the drains are done; the discharge
        // barrier block below quiesces the steward before reading any state.
        steward->endPhaseWindow();
    }

    deloadableMailOutBytes = 0;
    for (const Memory* lb : bodies) {
        if (lb != nullptr) deloadableMailOutBytes += lb->mailOutLiveBytes;
    }
    peakDeloadableMailOutBytes =
        std::max(peakDeloadableMailOutBytes, deloadableMailOutBytes);

    // (D-39 pendingAncestorOrigins drain retired by D-51 — contradiction record
    // now lives only in the __contradiction__ LB, no upward propagation. Drain
    // queue + sort + iteration removed in the post-D-52 cleanup commit.)

	inductionMemoryBlocks.clear();
	updateGlobalTuples.clear();
	// updateGlobalSingleChapterPages is reset by drainUpdateGlobalSingleChapter above.

    // --- Statification: LAZY deload — dump-on-release only ---
    // (D-155) Still the LAST act of the
    // kernel, after EVERY post-join drain (activateZeroCondition,
    // updateGlobal*, the compaction flush) — no same-iteration consumer
    // can touch a cold LB; the next touch is the next iteration's
    // phase-1 ensureLoaded or a post-prove reader's explicit
    // ensureLoaded. The CE filter never enters proveKernel, so its
    // clones never deload.
    //
    // POLICY — eager discharge drain for main batches (D-486,
    // narrowing D-156), pressure-lazy discharge for incubator batches
    // (D-551), decided HERE, executed by the
    // steward (D-159).
    // An LB that left the active set this iteration is flagged
    // dischargedForever (deactivation is permanent — no path flips
    // isActive back), keeps only its discharge survivors after the
    // compaction, and joins pendingDischarge in deterministic `active`
    // order. The barrier protocol (I-106): disarm any grant trigger,
    // QUIESCE the steward (every previously released drain completes —
    // quiesced block counts are logical again), fold what it drained,
    // enqueue this iteration's discharges and install the list. A main
    // batch wakes the steward now: the drain writes each survivor set as
    // one canonical image in the background while the next iteration
    // computes (discharged LBs are disjoint from everything the phases
    // touch), returns its blocks, and trims the main pool. An incubator
    // batch wakes it only at or above the steward::kWake watermark and
    // below it arms the grant trigger at the blocks-to-cross distance, so
    // a mid-iteration crossing wakes the steward, which drains the WHOLE
    // installed list; an incubator run that never crosses the watermark
    // writes no discharge image and its chapter export reads the
    // discharged LBs where they are. The manifest is rewritten only when
    // something was dumped. The dumped file SET is the deterministic
    // discharge order (and, for incubators, the run-invariant crossing
    // of the monotone grant ledger); only the dumping thread and its
    // timing vary (unobservable). The pool stays the hard bound
    // (exhaustion still asserts).
    {
        assert(steward && "proveKernel barrier without a steward scope");
        // The barrier drains are done with the reserved seam slot: return its
        // scratch blocks before the pool count below is read.
        releaseSlotScratch(scratchArenas(), genScratchArenas(), ruleStagings(),
                           seamScratchSlot);
        staticMemory().disarmGrantTrigger();
        steward->quiesce();
        // Batch byte sharing: no worker and no steward executor runs between
        // this quiesce and the discharge wake below, so this is the one
        // window in which the batch dictionary may grow. Payloads nominated
        // by at least two rows during phase 3 become shared entries; the
        // blocks convert their copies at their next maintenance seams.
        {
            const auto promotionStarted = std::chrono::steady_clock::now();
            const int32_t nominated = batchByteCandidates().distinctCount();
            const int64_t nominations = batchByteCandidates().nominationTotal();
            const int32_t added = batchByteDictionary().promoteNominations(batchByteCandidates());
            diagnosticsLog() << "[BATCH-BYTES] nominated=" << nominated
                << " nominations=" << nominations
                << " added=" << added
                << " entries=" << batchByteDictionary().count()
                << " entry_bytes=" << batchByteDictionary().logicalByteCount()
                << " seconds=" << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - promotionStarted).count()
                << std::endl;
        }
        bool dumpedSomething = false;
        // Fold + clear the list the steward drained since the last
        // barrier (its content was handed over by a previous install).
        // Per-LB discharge fold: a discharged LB is dumped exactly once and
        // never reloaded, so non-residency IS its done-marker. Drop the
        // entries already deloaded (by the steward's background drain or a
        // synchronous pass) and keep the still-resident, not-yet-drained ones
        // for re-install. Removing per-LB rather than clearing the whole list
        // on a single drained flag is what stops an already-deloaded LB from
        // being re-installed and deloaded a second time under heavy pressure.
        {
            const std::size_t beforeFold = pendingDischarge.size();
            pendingDischarge.erase(
                std::remove_if(pendingDischarge.begin(),
                               pendingDischarge.end(),
                               [](const Memory* b) {
                                   return !b->lbMemory.manager.resident();
                               }),
                pendingDischarge.end());
            if (pendingDischarge.size() != beforeFold) dumpedSomething = true;
        }
        // No eviction plan to fold — active-LB eviction is continuous (the
        // working-set pager), not barrier-armed. Only inactive discharges.
        for (Memory* b : active) {
            // Every phase releases its claim when done, so no active LB may
            // still be WorkerOwned here — a missed release would make the next
            // iteration's handshake skip its load (Rule 19). Idle (resident or
            // cold) and Dumped (cold) are both valid carry-over states, so no
            // reset is needed.
            assert(b->stewardClaim.load(std::memory_order_relaxed)
                       != static_cast<uint8_t>(
                              Memory::StewardClaim::WorkerOwned)
                   && "active LB still WorkerOwned at the barrier");
            if (!b->isActive) {
                assert(!b->dischargedForever
                    && "discharged LB re-entered an active snapshot");
                // SEAM DOOR (D-196): the pager may have
                // deloaded this LB while it was active+Idle in a phase or the
                // barrier seam window; reload before discharge
                // (dischargeStatementContent reads the registry and asserts
                // residency). The uniform handshake keeps the claim word
                // consistent (never Dumped-but-resident); it runs here
                // post-quiesce with no window open, so its valve is a defined
                // no-op and nothing can race the claim. The discharge
                // protocol (D-157): capture the exact
                // gate records, empty the dischargeable containers, reshuffle —
                // every block returns NOW, zero I/O. The pressure-lazy pending
                // dump later writes the near-empty image.
                steward->claimAndLoadForWork(*b, /*phase=*/4,
                                             lbdeload::kDeloadDirectory);
                b->stewardClaim.store(
                    static_cast<uint8_t>(Memory::StewardClaim::Idle),
                    std::memory_order_release);
                b->dischargeStatementContent(dischargeScratch);
                // The post-join drains' mail targets (the root, and the
                // incubator anchor LB of D-332) keep receiving mailOut merges
                // after their discharge, which a drained image could not take:
                // they stay resident until the export, everything else leaves
                // at this barrier (D-486).
                const bool mailTarget = b == &body
                    || (parameters.incubator_mode && b->parentMemory == &body
                        && b == blockMailProcessor.incubatorAnchorLb());
                if (steward->allowsSsdDeload() && !mailTarget) {
                    // Stamp the ordinal single-threaded now, before any later
                    // steward dump names the file.
                    b->ensureDeloadOrdinal();
                    pendingDischarge.push_back(b);
                }
            }
        }
        // Hand the discharged list to the steward. A main batch wakes its
        // background canonical dump at once (D-486): a discharged LB never
        // runs again (I-112), so its discharge survivors leave RAM at their
        // first barrier instead of waiting for pool pressure; the chapter
        // export reloads them read-only (D-158). An incubator batch, whose
        // peak the eager drain did not lower, waits for pool pressure
        // (D-551): wake at the watermark, else
        // arm the grant trigger to wake on a mid-iteration crossing. Each
        // barrier folds the drained entries out above, so the list holds the
        // discharges the drain has not reached. Active-LB pressure is handled
        // continuously by the working-set pager (the phase windows), so the
        // barrier never mass-deloads active LBs; the pool stays the hard
        // bound and genuine exhaustion asserts in a phase, never here.
        assert(steward->allowsSsdDeload() || pendingDischarge.empty());
        if (!pendingDischarge.empty()) {
            steward->installDischargeWork(pendingDischarge,
                                          lbdeload::kDeloadDirectory);
            const int64_t totalBlocks = staticMemory().totalBlocks();
            const int64_t usedNow = staticMemory().blocksInUse();
            if (!parameters.incubator_mode
                    || usedNow * steward::kWakeDen >= totalBlocks * steward::kWakeNum) {
                steward->wake();
            }
            else {
                const int64_t crossAt =
                    (totalBlocks * steward::kWakeNum + steward::kWakeDen - 1)
                    / steward::kWakeDen;
                const int64_t threshold = crossAt - usedNow;
                assert(threshold >= 1);
                MemorySteward* s = steward.get();
                staticMemory().armGrantTrigger(threshold,
                                               [s]() { s->wake(); });
            }
        }
        staticMemory().resetGrantLedger();
        if (dumpedSomething)
            lbdeload::rewriteRegistry(staticMemory().deloadRegistry(),
                                      staticMemory().extentSlabRegistry(),
                                      lbdeload::kDeloadDirectory);
        // BARRIER HEAD PREFETCH (D-196): warm the head of
        // the next iteration's sweep — one-shot HIGH-lane loads for the
        // Dumped, still-active LBs among the first window-width slots, so
        // phase 1 does not cold-start every iteration. Runs AFTER the
        // discharge decision (inactive heads are skipped) and after the
        // quiesce above, so no queued load can ever observe a
        // dischargedForever LB. Content-invisible cross-boundary I/O.
        // NOT harmless when unbounded: these loads execute with no window
        // open — no planner eviction can relieve them and the executor load
        // path has no valve — so prefetchHead caps its issues at the
        // prefetch budget (pending loads always fit inside the reserve's
        // free blocks), and after the LAST iteration the prove-scope guard
        // discards whatever is still queued (there is no next iteration to
        // warm). The unbudgeted, undiscarded form was the 4 GiB teardown
        // wall.
        steward->prefetchHead(
            &active,
            static_cast<std::size_t>(workers)
                + steward::kLookaheadWorkerMultiple
                      * static_cast<std::size_t>(workers),
            lbdeload::kDeloadDirectory);
    }
    gl::memory_census::markPhase(gl::PoolPhase::Outside);
}

}
