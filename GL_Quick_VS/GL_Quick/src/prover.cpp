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

#include "project_paths.hpp"
#include "prover.hpp"
#include "infra/peak_heap.hpp"
#include "infra/phase2_trace.hpp"
#include "infra/diagnostics_log.hpp"

#include "parameters.hpp"
#include "infra/hashburst_dump.hpp"
#include "infra/rt_tracker.hpp"
#include "infra/memory_census.hpp"
#include "memory_infra/global_memory_manager.hpp"
#include "memory_infra/scratch_arena.hpp"
#include "memory_infra/lb_deload.hpp"
#include "memory_infra/deload_stats.hpp"
#include "gpu/phase2_projection.hpp"
#ifdef GL_CUDA
#include "gpu/phase2_cuda.hpp"
#include "gpu/phase2_sealing.hpp"
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
    thread_local FtaBlowupAudit* ftaBlowupAudit = nullptr;



    //Declarations


    // Assumes n, m > 0.



/// @brief Construct the prover for a named anchor batch.
///
/// @details
/// One-shot constructor. Steps performed:
/// 1. Default-initialize the per-batch state slots (parameters, body LB,
///    permutation/mapping caches, dependency tracker, mailboxes,
///    contradiction table, etc.) via the member initialiser list.
/// 2. Locate `files/config/Config<anchorID>.json` via a search-path walk
///    (the default search list covers typical run environments).
/// 3. Load the JSON config into `coreExpressionMap` via
///    `ce::modifyCoreExpressionMap` and build `anchorInfo` via
///    `ce::initAnchor` (which asserts the anchor key exists).
/// 4. Populate the `operators` and `expressionsFromConfig` sets from
///    `coreExpressionMap` so hot-path classification queries are O(log N).
/// 5. Seed the per-batch counters (`implCounter`, `existenceCounter`,
///    `andCounter`, `orCounter`, `variableCounter`) from
///    `ProverParameters` defaults; these get bumped on every fresh
///    expression-name mint during the run.
/// 6. Build `allMappingsAna` and `allPermutationsAna` for the fixed
///    arity range so the static request pipeline does not have to
///    enumerate at runtime.
///
/// Construction does **not** run the pipeline — it only loads
/// configuration and prepares the per-batch state. The pipeline is
/// driven by `run_modes::fullRun(anchorID)`.
///
/// @param anchorID Short anchor name (e.g. `"Peano"`, `"Gauss"`,
///                 `"IncubatorPeano"`). The `"Anchor"` prefix is added
///                 internally by `ce::initAnchor`.
/// @param selectedPhase2Backend Explicit Phase 2 execution backend retained
///                              for every proof iteration in this batch.
/// @pre  `files/config/Config<anchorID>.json` exists and is well-formed.
/// @post `coreExpressionMap`, `anchorInfo`, `operators`,
///       `expressionsFromConfig`, and the per-batch counters are
///       populated. The body LB tree is empty (no theorem loaded yet).
/// @invariant [I-1](../../docs/agentic_swdd/30_invariants.md#i-1) — every
///            theorem-load path must call
///            `precompileStructuralOperators` before disintegration.
///            That call is in the per-theorem load site, not the
///            constructor.
/// @see [`ce::modifyCoreExpressionMap`](compiler.hpp#modifycoreexpressionmap),
///      [`ce::initAnchor`](compiler.hpp#initanchor).
// Default constructor definition
Prover::Prover(
    std::string anchorID,
    std::optional<Phase2Backend> selectedPhase2Backend,
    std::optional<std::filesystem::path> glBinaryFile,
    const PrecompiledDefinitions* precompiledDefinitions)
    :parameters(),
    phase2Backend(Phase2Backend::cpu),
    body(),
    allMappingsAna(),
    allPermutationsAna(),
    globalDependencies(),
    inductionMemoryBlocks(),
    contradictionTable()
{
    proverInitialization.initialize(anchorID, selectedPhase2Backend, glBinaryFile, precompiledDefinitions);
}








// =================== Main driver ===================

// Per-split-part submatch counter — the hashburst work metric. See prover.hpp.
// I-28 detect-and-defer trial: set true only inside a parallel phase-1/phase-3
// worker task (the runPhase wrappers); gates updateAdmissionMap3's ancestor write.

// Absorb-door scratch accessor: the phase-1/phase-3 workers publish their
// coreId here on entry; prefixArgumentsWithU reads it to pick its per-slot
// scratch arena. -1 (the default, never overwritten on single-threaded setup
// threads) maps to the reserved scratch slot.

#if RT_MEASUREMENT
// [RT phase measurement] C15 submatch-attribution state (see prover.hpp).







#endif







/// @brief Phase 1 of the elementary step — pre-hashburst mail absorb.
///
/// @details
/// The first of the three phases the elementary step is split into for
/// the LB split. Runs the ENTRY hashburst-dump trap
/// (Rule 14 — relocated here unchanged, follows its bracketed code)
/// and runs the pre-burst
/// `standardProcessing` (`PRE_FIXPOINT_MAIL_ABSORB`): empties the
/// per-burst `workingMemory` / `intExternalStatements` staging and
/// refills them from this cycle's
/// `mailIn` + `nextIterationInternalMail`. After this phase the LB's
/// request-generation inputs are current; the phase may discharge the
/// LB (flip `isActive`) via the absorb's `dischargeToBeProved`.
///
/// `PHASE13_DEEP_TIMING` attributes claim/load, routing pull, external
/// and internal origin/statement absorption, equivalence processing,
/// discharge, and cleanup into per-worker rows. These are summed worker
/// times and remain explicitly separate from the complete barrier wall.
///
/// @param body   The LB being stepped.
/// @param coreId The worker core id (routes `standardProcessing`).
/// @return (void)
/// @see `performElemPhase2`, `performElemPhase3` — the other phases.
/// @see `standardProcessing` — the absorb pipeline this phase invokes.
void Prover::performElemPhase1(Memory& body, unsigned coreId) {
    // Publish this worker's slot for the absorb door's per-slot scratch arena
    // (read deep in disintegration by prefixArgumentsWithU).
    gl::WorkerContext::setCoreId(static_cast<int>(coreId));
    // Per-call phase-1 tracker: feeds the cross-burst aggregate so the
    // pre-burst absorb (standardProcessing and the deposit-door tree under
    // it) is attributed down to its atomic sections. Per-LB `.rt` files stay
    // trigger-gated; the aggregate records every call.
    RT_TRACKER_DECL(body);
#if PHASE13_DEEP_TIMING
    const auto recordPhase1Detail =
        [this, coreId](Phase13TimingSlot slot,
                       std::chrono::steady_clock::time_point started) {
            // No rows bound = the defined no-measurement state (unit tests /
            // CE clones run phase helpers outside proveKernel's binding) —
            // same contract as standardProcessing's recordPhase13Detail.
            if (phase13TimingRows == nullptr) return;
            assert(coreId < phase13TimingWorkers
                && "Phase 1 timing row is not bound to this worker");
            phase13TimingRows[
                static_cast<std::size_t>(coreId) * kPhase13TimingSlotCount
                + static_cast<std::size_t>(slot)] +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count();
        };
    const auto claimLoadStarted = std::chrono::steady_clock::now();
#endif
    // Unified working-set handshake (the user's "a worker uploads itself"):
    // claim this LB and make it resident before ANYTHING reads it — the dump
    // trap below included. All three phases use the same handshake
    // (I-114). It reloads a cold LB, making room first
    // via the load<->evict exchange, and bounds its `Busy` wait with the
    // stuck-assert (I-113). The claim is released at the end
    // of the body so the steward may reclaim the LB once done. CE-filter
    // clones run with NO steward (always resident, never deloaded) and skip
    // the handshake entirely.
    {
        RT_SCOPE("PH1_CLAIM_LOAD");
        if (steward)
            steward->claimAndLoadForWork(body, /*phase=*/1,
                                         lbdeload::kDeloadDirectory);
    }
#if PHASE13_DEEP_TIMING
    recordPhase1Detail(Phase13TimingSlot::claimLoad, claimLoadStarted);
    const auto burstSetupStarted = std::chrono::steady_clock::now();
#endif

    // Quiescence SLEEP detector reset (D-194): the per-burst
    // non-statement mutation flag starts clean, and the statement-count baseline
    // is snapped BEFORE the pre-burst mail absorb below, so any mail-absorbed
    // deposit this burst counts as work (keeps the LB awake). Read back at
    // phase-3 exit to decide whether hasWork clears.
    body.mutatedThisBurst = false;
    body.encodedCountAtBurstStart = body.intEncodedStatementsCount();

    // ENTRY trap — delegate to hashburst_dump (relocated into phase 1 with the
    // performElem phase split; the dump sections / file path / target predicate
    // are unchanged, Rule 14). Mirrored in `feedback_dump_structure_immutable.md`.
    if (hashburst_dump::isTargetLB(body)) {
        hashburst_dump::dumpEntry(body, this->definitions.entries());
#if RT_MEASUREMENT
        // [RT phase measurement] fresh per-burst attribution tallies;
        // burst separator in the hit log (truncate on the first burst of a run).
        phase2Executor.resetTargetCounters();
        {
            static std::atomic<int> crtBurstNo{ 0 };
            const int b = crtBurstNo.fetch_add(1, std::memory_order_relaxed);
            std::ofstream hf(".debug/c15_successful_submatches.txt",
                             b == 0 ? std::ios::trunc : std::ios::app);
            hf << "== target-LB burst " << (b + 1) << " ==\n";
        }
#endif
    }

    // The per-step delta-class tracker `changedClassesThisStep` is
    // deliberately NOT cleared here: its sole clear is the tail of
    // `standardProcessing` (Step 7), after every consumer has read it. A
    // delta minted OUTSIDE any `standardProcessing` call — LB seeding
    // (`addTheoremToMemory` copy axioms / equality premises), post-join
    // barrier deposits — must survive to this step's `applyEquiClasses`,
    // which back-applies it to every pre-existing statement (Pass 1). An
    // entry clear here discards exactly those deltas unconsumed (the
    // waterline seeded at class creation then hides the pre-existing
    // statements from Pass 2 forever).

    { RT_SCOPE_HERE("PRE_FIXPOINT_MAIL_ABSORB");
    // ===================================================================
    // PRE-FIXPOINT MAIL ABSORB.
    // Absorbing mail before the fixpoint avoids one PK of latency between
    // a proven implication arriving at a receiver's mailIn and its rule
    // firing locally — the latency that pushes the Peano-incube
    // `__contradiction__(=[a,b])` LBs past MAX_NAME_IDS before they can
    // discharge.
    //
    // Mail arriving at this PK is absorbed FIRST, so the resulting
    // rules/facts are visible to THIS burst's request generation and
    // fixpoint. The persistent per-LB fields
    // workingMemory/externalStatements/intExternalStatements serve as
    // same-burst staging: cleared and refilled by this absorb, read by
    // the request-gen blocks below. D-76 deferred compact-form broadcast
    // and the Mail::implications deletion stay unchanged.
    // ===================================================================

    // Empty the per-burst external scratch, then refill from THIS cycle's
    // fresh mail below. These are NOT persistent across bursts (unlike the
    // local-premise containers); this absorb is their sole writer.
    body.workingMemory.resetToFresh();
    body.intExternalStatements.clear();
#if PHASE13_DEEP_TIMING
    recordPhase1Detail(Phase13TimingSlot::burstSetup, burstSetupStarted);
    const auto routingMailPullStarted = std::chrono::steady_clock::now();
#endif

    // The sameIterationInternalMail absorb runs AFTER the hashburst loop
    // (the game resets after hashburst). The mailIn (parent-emitted)
    // absorb that follows STAYS in its pre-hashburst position.

    // Pre-hashburst absorb pipeline — a single `standardProcessing`
    // call routing three mail channels:
    //
    //   - externalMailIn = &body.mailIn — external arrivals from
    //     other LBs' mail, delivered by the MailLog phase-1 pull (status=3).
    //   - internalMailIn = body.nextIterationInternalMail —
    //     cross-iteration deferred emissions from the previous
    //     step's pre-burst discharge AND post-burst discharge
    //     AND the vacuous-truth path inside addExprToMemoryBlock
    //     AND the updateGlobalSingleChapter siblings (status=1).
    //   - internalMailOut = body.sameIterationInternalMail —
    //     any discharge parent-scope emission during THIS absorb
    //     writes here; it then feeds the same step's hashburst
    //     rule firings (which also write to sameIter) and gets
    //     drained by the post-burst standardProcessing call.
    //     Zero-delay routing for pre-burst discharge emissions.
    //
    // The pre-burst call runs the full pipeline — mailIn-derived
    // statements get applied and discharged in the same step they
    // arrive, rather than waiting for the post-burst block's
    // first-pass.

    // New mail system (D-137): pull every un-ingested ancestor
    // batch into mailIn before this absorb drains it. The pull reads ancestors'
    // frozen logs and writes only this LB's own cursor row and its own mailIn —
    // parallel-safe, because commits happen only at the single-threaded
    // post-join seam. CE / compressor LBs are isolated/flat and never registered,
    // so they are gated out.
    int64_t trackedMailInBlocks = 0;
    if (!ceFilteringActive && !parameters.compressor_mode) {
        RT_SCOPE("PH1_MAIL_PULL");
        this->mailLog.pull(&body, body.mailIn);
        trackedMailInBlocks = body.mailIn.blocksHeld();
        const int64_t simultaneousMailInBlocks =
            routingMailInBlocksInFlight.fetch_add(
                trackedMailInBlocks, std::memory_order_relaxed)
            + trackedMailInBlocks;
        int64_t observedPeak =
            peakRoutingMailInBlocks.load(std::memory_order_relaxed);
        while (simultaneousMailInBlocks > observedPeak
            && !peakRoutingMailInBlocks.compare_exchange_weak(
                observedPeak, simultaneousMailInBlocks,
                std::memory_order_relaxed)) {
        }
    }
#if PHASE13_DEEP_TIMING
    recordPhase1Detail(Phase13TimingSlot::routingMailPull,
                       routingMailPullStarted);
#endif

    // mailIn is HOT (I-101): the absorb reads its canonical
    // sorted snapshots directly — no transient heap Mail. Cleared here, after the
    // drain returns; nothing between standardProcessing's drain and its return
    // reads it.
    logicBlockOperations.standardProcessing(body,
                             /*externalMailIn=*/&body.mailIn,
                             /*internalMailIn =*/body.nextIterationInternalMail,
                             /*internalMailOut=*/body.sameIterationInternalMail,
                             coreId);
#if PHASE13_DEEP_TIMING
    const auto routingMailCleanupStarted =
        std::chrono::steady_clock::now();
#endif
    body.mailIn.clear();
    if (!ceFilteringActive && !parameters.compressor_mode) {
        const int64_t beforeRelease = routingMailInBlocksInFlight.fetch_sub(
            trackedMailInBlocks, std::memory_order_relaxed);
        assert(beforeRelease >= trackedMailInBlocks
            && "routing mailIn attribution counter underflow");
    }
#if PHASE13_DEEP_TIMING
    recordPhase1Detail(Phase13TimingSlot::routingMailCleanup,
                       routingMailCleanupStarted);
#endif
    } // RT_SCOPE PRE_FIXPOINT_MAIL_ABSORB

    body.lbMemory.sharedNativeKeys.collect(compactionScratchArenas().forSlot(coreId));
    // Rows nominated before the last barrier now find their bytes in the batch
    // dictionary; counterexample clones are short-lived and skipped.
    if (!ceFilteringActive)
        body.lbMemory.shareBatchBytesAfterPhase1(genScratchArenas().forSlot(coreId));
    body.lbMemory.originStrings.keyStore().packPrefixes(genScratchArenas().forSlot(coreId));

    // Return whole blocks of freed pages before phase 2 reads this LB (the
    // per-burst resets above free pages the refill may not reuse). The page
    // pack is content-invisible (I-107) and runs under this worker's
    // exclusive claim; counterexample clones are short-lived and skipped.
    if (!ceFilteringActive)
        steward::compactIfFragmented(body.lbMemory,
                                     compactionScratchArenas().forSlot(coreId));

    // CUDA route: record this LB's Phase 2 read image while it is still
    // claimed and resident; the chunk packing reads the record instead of the
    // LB (D-409). A discharged LB gets no task.
    if (phase2Backend == Phase2Backend::cuda && body.isActive)
        phase2Executor.recordPhase2Measurement(body);

    // Release the claim — phase 1 is done with this LB, so it is deloadable
    // again (the steward may now reclaim it). CE clones have no steward.
#if PHASE13_DEEP_TIMING
    const auto releaseClaimStarted = std::chrono::steady_clock::now();
#endif
    if (steward) {
        assert(body.stewardClaim.load(std::memory_order_relaxed)
                   == static_cast<uint8_t>(Memory::StewardClaim::WorkerOwned)
               && "phase-1 release of an LB not held WorkerOwned");
        body.stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle),
            std::memory_order_release);
    }
#if PHASE13_DEEP_TIMING
    recordPhase1Detail(Phase13TimingSlot::releaseClaim,
                       releaseClaimStarted);
#endif
}



/// @brief Phase 3 of the elementary step — post-hashburst absorb
///        and shared scope cleanup.
///
/// @details
/// Runs the post-burst `standardProcessing` (`POST_FIXPOINT_MAIL_FLUSH`)
/// absorbing the hashburst's `sameIterationInternalMail` deposits — this
/// is where the authoritative deactivation / discharge fires (I-66) when
/// a closing head was deposited. The LB's `mailOut` is left populated for
/// the post-join `MailLog` commit barrier in `proveKernel` to drain.
/// Standard processing owns hypothesis reaction and ordered scope
/// cleanup. This wrapper owns the temporary exclusion, its completion
/// assertion, quiescence bookkeeping and EXIT hashburst dump (Rule 14).
/// `PHASE13_DEEP_TIMING` attributes the standard-processing substeps,
/// reaction, each sanitize/drain seam, quiescence bookkeeping, and the
/// working-set claim/release into contention-free per-worker rows.
///
/// @param body   The LB being stepped.
/// @param coreId Worker core id (routes `standardProcessing`).
/// @return (void)
/// @see `performElemPhase1`, `performElemPhase2` — the other phases.
void Prover::performElemPhase3(Memory& body, unsigned coreId) {
    FtaBlowupAudit audit;
    static const bool auditEnabled = [] {
#ifdef _WIN32
        size_t required = 0;
        const errno_t result = getenv_s(&required, nullptr, 0, "GL_FTA_BLOWUP_TRACE");
        assert(result == 0);
        return required != 0;
#else
        return std::getenv("GL_FTA_BLOWUP_TRACE") != nullptr;
#endif
    }();
    const bool auditTarget = auditEnabled && hashburst_dump::isTargetLB(body);
    assert(ftaBlowupAudit == nullptr);
    if (auditTarget) {
#if RT_MEASUREMENT
        audit.burst = gl::rt_tracker::g_currentHashburstIndex;
#endif
        ftaBlowupAudit = &audit;
    }

    // Publish this worker's slot for the absorb door's per-slot scratch arena
    // (read deep in disintegration by prefixArgumentsWithU).
    gl::WorkerContext::setCoreId(static_cast<int>(coreId));
#if PHASE13_DEEP_TIMING
    const auto recordPhase3Detail =
        [this, coreId](Phase13TimingSlot slot,
                       std::chrono::steady_clock::time_point started) {
            // No rows bound = the defined no-measurement state — same
            // contract as standardProcessing's recordPhase13Detail.
            if (phase13TimingRows == nullptr) return;
            assert(coreId < phase13TimingWorkers
                && "Phase 3 timing row is not bound to this worker");
            phase13TimingRows[
                static_cast<std::size_t>(coreId) * kPhase13TimingSlotCount
                + static_cast<std::size_t>(slot)] +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count();
        };
    const auto claimLoadStarted = std::chrono::steady_clock::now();
#endif
    // Per-call phase-3 tracker: feeds the cross-burst aggregate so the
    // post-burst absorb, the discharge machinery, and the end-of-burst
    // sanitize drains are attributed down to their atomic sections.
    RT_TRACKER_DECL(body);
    // Unified working-set handshake (all phases equivalent): claim + load
    // this LB before phase 3 reads it; CE clones run with no steward.
    {
        RT_SCOPE("PH3_CLAIM_LOAD");
        if (steward)
            steward->claimAndLoadForWork(body, /*phase=*/3,
                                         lbdeload::kDeloadDirectory);
    }
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::claimLoad, claimLoadStarted);
#endif

    assert(body.phase3IgnoredOrRoots == nullptr
        && "Phase 3 exclusion context is already bound");
    ScratchArena& phase3Arena = genScratchArenas().forSlot(coreId);
    DirtyState phase3Dirty = DirtyState::Clean;
    ColdHashSet<PodKeyStore<NameId>> ignoredOrRoots(&phase3Arena, &phase3Dirty);
    PagedVector<NameId> ignoredLiveRoots(&phase3Arena, &phase3Dirty);
    caseProcessor.collectPhase3IgnoredOrRoots(body, ignoredOrRoots);
    for (int32_t id = 1; id <= ignoredOrRoots.count(); ++id) {
        const NameId root = ignoredOrRoots.decode(id);
        if (body.orLiveBranches.lookup(root) != 0)
            ignoredLiveRoots.push_back(root);
    }
    body.phase3IgnoredOrRoots = ignoredOrRoots.empty() ? nullptr : &ignoredOrRoots;

    { RT_SCOPE_HERE("POST_FIXPOINT_MAIL_FLUSH");
    logicBlockOperations.standardProcessing(body,
                             /*externalMailIn=*/nullptr,
                             /*internalMailIn =*/body.sameIterationInternalMail,
                             /*internalMailOut=*/body.nextIterationInternalMail,
                             coreId);

    // mailOut is no longer flushed here. It accumulates this step's fillMailOut
    // output (phase 1 + phase 3) and is drained once, single-threaded, by the
    // commit barrier in proveKernel (D-137). Clearing it here
    // would leave that commit nothing to ship.
    } // RT_SCOPE POST_FIXPOINT_MAIL_FLUSH

    for (int32_t id = 0; id < ignoredLiveRoots.size(); ++id) {
        assert(body.intValidityNamesToFilter.lookup(ignoredLiveRoots[id]) != 0
            && "an excluded live OR branch must close and be wiped in this phase");
    }

#if PHASE13_DEEP_TIMING
    const auto quiescenceAndDumpsStarted =
        std::chrono::steady_clock::now();
#endif

    // Quiescence SLEEP (D-194): the burst has finished
    // mutating. Clear hasWork unless this burst produced work — a statement
    // deposit / equi-class rewrite / subtree-wipe erase (count changed vs the
    // phase-1 baseline), a flagged non-statement mutation (admission churn or
    // wipe), or a pending emission on an INTERNAL mail channel (revival on
    // sameIter, a cross-iteration deferral on nextIter — both are inputs this
    // LB's own NEXT burst absorbs). A sound over-approximation: when any of
    // these holds the LB stays dirty and is swept again next iteration
    // (Rule 19 — a missed wake would be unsound, an extra sweep is merely
    // slow). Reads only this LB's own resident/never-deloaded state; no
    // residency input (I-106 / I-108).
    //
    // mailOut is deliberately NOT in this fold. It is OUTGOING-ONLY under the
    // pull model (I-64: fillMailOut and the deposit doors write it; the
    // single-threaded commit barrier is its sole consumer): no burst path
    // derives local state from it — the one historical burst-path read, the
    // retired transitive-ship dedup, is dead code — so pending mailOut content
    // can never make this LB's NEXT burst productive. And the barrier's commit
    // sweep iterates `bodies` regardless of sweep status, so that content is
    // committed whether or not this LB sweeps. Including it was never a valid
    // activity signal, and it false-fired the shadow check on the ROOT: the
    // post-join drains (the updateGlobal* theorem sends, the D-76 compaction
    // flush) append to the root's mailOut AFTER the barrier, so a genuinely
    // no-op root burst saw "mailOut non-empty" here and re-armed hasWork.
    const bool sleepMutated = body.mutatedThisBurst;
    const int32_t sleepCountNow = body.intEncodedStatementsCount();
    const bool sleepSameIterNonEmpty = !body.sameIterationInternalMail.empty();
    const bool sleepNextIterNonEmpty = !body.nextIterationInternalMail.empty();
    body.hasWork =
          sleepMutated
       || sleepCountNow != body.encodedCountAtBurstStart
       || sleepSameIterNonEmpty
       || sleepNextIterNonEmpty;

#ifdef QUIESCE_SHADOW_CHECK
    // Empirical soundness validator (off by default, in-tree). Under the flag the
    // skip filter does NOT exclude a would-be-skipped LB — it sweeps it anyway and
    // records the predicate's verdict in shadowWouldSkip. If that "quiescent"
    // burst actually produced work, hasWork is now set and this fires at the exact
    // LB: the predicate is unsound (a missed wake door or a missed mutation choke
    // point). Rule 19 in spirit — surface it loudly at its origin. The forensic
    // dump below prints exactly WHICH SLEEP condition re-triggered and the full
    // parentMemory chain (Rule 12 — full chain, never exprKey alone), so one run
    // identifies the miss precisely.
    if (body.shadowWouldSkip && body.hasWork) {
        std::ostringstream fx;
        fx << "[QUIESCE-SHADOW] predicate miss at burst="
           << ::gl::rt_tracker::g_currentHashburstIndex
           << " phase=3 (performElemPhase3 exit)\n";
        fx << "[QUIESCE-SHADOW] LB chain (innermost -> root):\n";
        for (const Memory* cur = &body; cur != nullptr;
             cur = cur->parentMemory) {
            fx << "[QUIESCE-SHADOW]   '" << cur->exprKey() << "'"
               << (cur->parentMemory == nullptr ? "  (root sentinel)" : "")
               << "\n";
        }
        fx << "[QUIESCE-SHADOW] SLEEP conditions re-triggered this burst:\n";
        fx << "[QUIESCE-SHADOW]   mutatedThisBurst        = "
           << (sleepMutated ? "TRUE" : "false") << "\n";
        fx << "[QUIESCE-SHADOW]   encodedStatements count = "
           << sleepCountNow << " now vs " << body.encodedCountAtBurstStart
           << " at phase-1 entry"
           << (sleepCountNow != body.encodedCountAtBurstStart
                   ? "  <-- CHANGED" : "  (unchanged)") << "\n";
        fx << "[QUIESCE-SHADOW]   sameIterationInternalMail non-empty = "
           << (sleepSameIterNonEmpty ? "TRUE" : "false") << "\n";
        fx << "[QUIESCE-SHADOW]   nextIterationInternalMail non-empty = "
           << (sleepNextIterNonEmpty ? "TRUE" : "false") << "\n";
        fx << "[QUIESCE-SHADOW] latch state: shadowWouldSkip=true (so at this "
              "iteration's active-build hasWork was false AND mailPeek was "
              "false); hasWork now=true\n";
        fx << "[QUIESCE-SHADOW] mailPeek(now, frozen logs) = "
           << (mailLog.mailPeek(&body) ? "TRUE" : "false")
           << "  (commits happen only at the barrier, so this equals the "
              "active-build value)\n";
        fx << "[QUIESCE-SHADOW] isActive=" << (body.isActive ? "true" : "false")
           << " dischargedForever=" << (body.dischargedForever ? "true" : "false")
           << " primedForContradiction="
           << (body.primedForContradiction ? "true" : "false")
           << " toBeProved count=" << body.intToBeProved.count() << "\n";
        std::cerr << fx.str() << std::flush;
        assert(!body.hasWork
            && "QUIESCE_SHADOW_CHECK: a would-be-skipped LB's burst mutated state "
               "— the quiescence predicate is unsound (missed wake or missed "
               "mutation choke point)");
    }
#endif

    body.lbMemory.sharedNativeKeys.collect(compactionScratchArenas().forSlot(coreId));
    // Rows minted this iteration share bytes the batch dictionary holds; the
    // rest is nominated for the barrier. Counterexample clones are skipped.
    if (!ceFilteringActive)
        body.lbMemory.shareBatchBytesAfterPhase3(batchByteCandidates(),
                                                 genScratchArenas().forSlot(coreId));
    body.lbMemory.originStrings.keyStore().packPrefixes(genScratchArenas().forSlot(coreId));
    // Scope cleanup and shared-key collection can leave whole blocks of free
    // pages. Pack them while this worker still owns the block exclusively;
    // virtual page identities and proof content remain unchanged.
    if (!ceFilteringActive)
        steward::compactIfFragmented(body.lbMemory,
                                     compactionScratchArenas().forSlot(coreId));
    body.sampleParkedSourceStorage();


    // EXIT trap — delegate to hashburst_dump (relocated here from the inline
    // performElem tail with the phase split; dump unchanged, Rule 14).
    if (hashburst_dump::isTargetLB(body)) {
        hashburst_dump::dumpExit(body);
        // No abort — let the pipeline run to completion. The trap
        // continues firing for every EXIT of this LB; the trace
        // accumulates all of them.
#if RT_MEASUREMENT
        // Per-burst submatch attribution for the hashburst dump's target LB.
        // File-bound like the phase report beside it, never stdout.
        std::filesystem::create_directories(".rt");
        std::ofstream c15Log(".rt/burst_phases.log", std::ios::app);
        c15Log << "[C15-SUB] accepted="
                  << phase2Executor.targetAccepted()
                  << " acceptedUnlinkedDiv="
                  << phase2Executor.targetAcceptedUnlinked()
                  << " attempts="
                  << phase2Executor.targetAttempts()
                  << " attemptsUnlinkedDiv="
                  << phase2Executor.targetAttemptsUnlinked()
                  << "\n";
#endif
    }

    // Release the claim — phase 3 is done with this LB; deloadable again
    // (the steward may now reclaim it). CE clones have no steward.
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::quiescenceAndDumps,
                       quiescenceAndDumpsStarted);
    const auto releaseClaimStarted = std::chrono::steady_clock::now();
#endif
    if (steward) {
        assert(body.stewardClaim.load(std::memory_order_relaxed)
                   == static_cast<uint8_t>(Memory::StewardClaim::WorkerOwned)
               && "phase-3 release of an LB not held WorkerOwned");
        body.stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle),
            std::memory_order_release);
    }
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::releaseClaim,
                       releaseClaimStarted);
#endif
    ftaBlowupAudit = nullptr;
    if (auditTarget) {
        FILE* out = nullptr;
#ifdef _WIN32
        const errno_t openResult = fopen_s(&out, ".debug/traps.log", "a");
        assert(openResult == 0);
#else
        out = std::fopen(".debug/traps.log", "a");
#endif
        assert(out);
        std::fprintf(out, "[FTA-CFE-SUM] burst=%d calls=%llu matches=%llu direct=%llu variants=%llu direct_variants=%llu setup_ns=%llu probes_ns=%llu snapshot_members=%llu equality_calls=%llu equality_members=%llu\n",
            audit.burst, audit.calls, audit.matches, audit.directMatches,
            audit.variants, audit.directVariants, audit.setupNs, audit.probesNs,
            audit.snapshotMembers, audit.equalityCalls, audit.equalityMembers);
        std::fprintf(out, "[FTA-EQ-SUM] burst=%d setup=%llu same=%llu ancestor=%llu closure=%llu writeback=%llu delta=%llu ledger=%llu weak=%llu origins=%llu origin_b_calls=%llu origin_b_comparisons=%llu origin_b_ns=%llu origin_a_calls=%llu origin_a_comparisons=%llu origin_a_ns=%llu\n",
            audit.burst, audit.classStageNs[0], audit.classStageNs[1], audit.classStageNs[2],
            audit.classStageNs[3], audit.classStageNs[4], audit.classStageNs[5],
            audit.classStageNs[6], audit.classStageNs[7], audit.classStageNs[8],
            audit.originBCalls, audit.originBComparisons, audit.originBNs,
            audit.originACalls, audit.originAComparisons, audit.originANs);
        std::fclose(out);
    }

}



// Count disjuncts in an expanded OR expression like !(&!(&!(d1)!(d2))!(d3))
// Parses the nested right-associative !(&...) structure.
static int countOrDisjuncts(const std::string& expr) {
    if (expr.size() < 4 || expr.substr(0, 3) != "!(&") return 1;
    // Strip !(&...) → inner content (two children, each starting with !)
    std::string inner = expr.substr(3, expr.size() - 4);
    // Find end of first child by tracking paren depth
    int depth = 0;
    bool seenOpen = false;
    for (size_t i = 0; i < inner.size(); ++i) {
        if (inner[i] == '(' || inner[i] == '[') { depth++; seenOpen = true; }
        if (inner[i] == ')' || inner[i] == ']') depth--;
        if (seenOpen && depth == 0) {
            return countOrDisjuncts(inner.substr(0, i + 1)) + 1;
        }
    }
    return 1; // malformed fallback
}



/// @brief Drive one outer iteration over every LB in `bodies` — the
/// prover's main parallel hash-burst loop.
///
/// @details
/// Spawns a worker pool sized at `logicalCores`; the cycle runs as three
/// barriered phase sweeps (`performElemPhase1` -> join -> phase 2 -> join ->
/// `performElemPhase3` -> join), each a work-stealing sweep over the active LBs.
/// Phase 2 is the **flat (LB, part) executor pool** (one work-stealing pool over
/// every active LB's parts — a straggler contributes `logicalCores` expression
/// buckets, everything else one part) followed by a per-LB finalize
/// (`performElemPhase2`), in a <= 2-round loop: round 1 runs the producers +
/// unsplit bursts, round 2 runs the stragglers' buckets. The split is decided by
/// the end-of-iteration `isStraggler` stats pass (D-201).
/// Cross-LB state mutations are forbidden during the worker phase per
/// [I-28](../../docs/agentic_swdd/30_invariants.md#i-28); each worker writes only
/// to its own LB's local state and to the per-thread collectors
/// (`updateGlobalTuples`, the `updateGlobalSingleChapterPages` sealed set, the
/// per-core-slot mail outboxes).
///
/// After `pool.join()`, the post-parallel collector phase runs:
/// 1. The `MailLog` commit barrier — appends each LB's `mailOut` as one
///    batch (statements + exprOriginMap) to its log, which descendants pull
///    at the next iteration's phase 1; the per-item validity assert per
///    [I-26](../../docs/agentic_swdd/30_invariants.md#i-26) runs at the
///    receiver-side absorb.
/// 2. `updateGlobal` / `updateGlobalSingleChapter` — drains the per-thread
///    tuple queues into the global theorem list / dependencies.
/// 3. Equivalence-class merger pass — folds any new equalities
///    discovered this iteration.
///
/// The loop continues until either:
/// - no LB produced any new statement this iteration
///   (`Memory::deltaNumberStatements == 0` everywhere), or
/// - the iteration cap from `parameters` is reached.
///
/// @param bodies        LB pointers to iterate over.
/// @param iteration     Outer iteration counter.
/// @param ...           Routing parameters.
/// @invariant [I-26](../../docs/agentic_swdd/30_invariants.md#i-26),
///            [I-28](../../docs/agentic_swdd/30_invariants.md#i-28).
/// @see `performElemPhase1` / `performElemPhase2` / `performElemPhase3` — the per-LB phase workers.
/// @see `MailLog` — the per-LB pull logs that carry cross-LB mail (commit barrier here).
void Prover::proveKernel(const std::vector<Memory*>& bodies) {
    // The opt-in capture seam owns the full allocation walk.
    peak_snapshot::RootScope peakRoots(this,
        [](void* context) {
            auto& analyzer=*static_cast<Prover*>(context);
            assert(analyzer.steward);
            analyzer.steward->quiesce();
        },
        [](void* context) {
            auto& analyzer=*static_cast<Prover*>(context);
            GlobalMemoryManager* managers[]={&staticMemory(),&persistentMemory(),&mailMemory(),&lbMemory(),&gpuHostMemory()};
            const char* poolNames[]={"Main","Persistent","Mail","LogicBlockBodies","GpuHost"};
            for(std::size_t index=0;index<std::size(managers);++index) {
                peak_snapshot::Scope pools("Pools");
                peak_snapshot::Scope pool(poolNames[index]);
                managers[index]->observeResidentRanges();
            }
            const auto observe=[](Memory& block) {
                peak_snapshot::Scope object("Memory");
                peak_snapshot::note(&block,sizeof(block),0);
                { peak_snapshot::Scope state("lbMemory");
                  { peak_snapshot::Scope arena("arena"); block.lbMemory.manager.observeResidentRanges(); }
                  if(block.lbMemory.manager.resident())
                      block.lbMemory.visitLiveBytes([](const char*,int64_t){});
                }
                { peak_snapshot::Scope goals("intToBeProved");
                  { peak_snapshot::Scope arena("arena"); block.persistentArena.observeResidentRanges(); }
                  block.intToBeProved.liveBytes();
                }
                { peak_snapshot::Scope mail("mailIn");
                  block.mailIn.arena_.observeResidentRanges();
                  if(block.mailIn.arenaInited_) {
                    peak_snapshot::measure("statements",block.mailIn.statements_,[&]{return block.mailIn.statements_.liveBytes();});
                    peak_snapshot::measure("origins",block.mailIn.origins_,[&]{return block.mailIn.origins_.liveBytes();});
                    peak_snapshot::measure("statementFlags",block.mailIn.statementFlags_,[&]{return block.mailIn.statementFlags_.liveBytes();});
                  }
                }
            };
            observe(analyzer.body);observe(analyzer.ceBody);
            analyzer.lbStore.observeLiveSlots([&](void* slot){observe(*static_cast<Memory*>(slot));});
            { peak_snapshot::Scope owner("MailLog");
              analyzer.mailArena.observeResidentRanges();
              peak_snapshot::measure("mailBlobPool",analyzer.mailLog.mailBlobPool,[&]{return analyzer.mailLog.mailBlobPool.liveBytes();});
              peak_snapshot::measure("mailRefs",analyzer.mailLog.mailRefs,[&]{return analyzer.mailLog.mailRefs.liveBytes();});
              peak_snapshot::measure("mailHeads",analyzer.mailLog.mailHeads,[&]{return analyzer.mailLog.mailHeads.liveBytes();});
              peak_snapshot::measure("mailEdges",analyzer.mailLog.mailEdges,[&]{return analyzer.mailLog.mailEdges.liveBytes();});
              peak_snapshot::measure("mailCursor",analyzer.mailLog.mailCursor,[&]{return analyzer.mailLog.mailCursor.liveBytes();});
            }
            { peak_snapshot::Scope owner("SimpleMapStore");
              analyzer.simpleMapArena.observeResidentRanges();
              peak_snapshot::measure("edges",analyzer.simpleMapStore.edges,[&]{return analyzer.simpleMapStore.edges.liveBytes();});
              peak_snapshot::measure("heads",analyzer.simpleMapStore.heads,[&]{return analyzer.simpleMapStore.heads.liveBytes();});
            }
            { peak_snapshot::Scope owner("CounterexampleSimpleMapStore");
              analyzer.ceSimpleMapArena.observeResidentRanges();
              peak_snapshot::measure("edges",analyzer.ceSimpleMapStore.edges,[&]{return analyzer.ceSimpleMapStore.edges.liveBytes();});
              peak_snapshot::measure("heads",analyzer.ceSimpleMapStore.heads,[&]{return analyzer.ceSimpleMapStore.heads.liveBytes();});
            }
            { peak_snapshot::Scope owner("skeletonInterner"); skeletonInterner().liveBytes(); }
            { peak_snapshot::Scope owner("mailInterner"); mailInterner().liveBytes(); }
            { peak_snapshot::Scope owner("batchByteDictionary"); batchByteDictionary().liveBytes(); }
            { peak_snapshot::Scope owner("batchByteCandidates");
              for(unsigned i=0;i<batchByteCandidates().stripeCount();++i)
                  peak_snapshot::measure("stripe",batchByteCandidates().stripe(i),[&]{return batchByteCandidates().stripe(i).liveBytes();}); }
            { peak_snapshot::Scope owner("stringScratch");
              for(unsigned i=0;i<scratchArenas().slotCount();++i) scratchArenas().forSlot(i).observeResidentRanges(); }
            { peak_snapshot::Scope owner("containerScratch");
              for(unsigned i=0;i<genScratchArenas().slotCount();++i) genScratchArenas().forSlot(i).observeResidentRanges(); }
            { peak_snapshot::Scope owner("dischargeScratch"); analyzer.dischargeScratch.observeResidentRanges(); }
            { peak_snapshot::Scope owner("ExpressionAnalyzerHeap");
                peak_snapshot::field("compiledExpressions",analyzer.definitions.entries());
                peak_snapshot::field("coreExpressionMap",analyzer.definitions.configurations());
                peak_snapshot::field("updateGlobalTuples",analyzer.updateGlobalTuples);
                peak_snapshot::field("globalTheoremList",analyzer.globalTheoremList);
                peak_snapshot::field("globalTheoremProducers",analyzer.globalTheoremProducers);
                peak_snapshot::field("retractedVacuousTheorems",analyzer.retractedVacuousTheorems);
                peak_snapshot::field("fullTheoremList",analyzer.fullTheoremList);
                peak_snapshot::field("lastCompressionSurvivors",analyzer.lastCompressionSurvivors);
                peak_snapshot::field("inductionMemoryBlocks",analyzer.inductionMemoryBlocks);
                peak_snapshot::field("pendingTwinDeactivations",analyzer.pendingTwinDeactivations);
                peak_snapshot::field("pendingSubtreeDeactivations",analyzer.pendingSubtreeDeactivations);
                peak_snapshot::field("allMappingsAna",analyzer.allMappingsAna);
                peak_snapshot::field("allPermutationsAna",analyzer.allPermutationsAna);
                peak_snapshot::field("permanentBodies",analyzer.permanentBodies);
                peak_snapshot::field("permanentBodiesCE",analyzer.permanentBodiesCE);
                peak_snapshot::field("pendingCompactionQueue",analyzer.pendingCompactionQueue);
                peak_snapshot::field("pendingDischarge",analyzer.pendingDischarge);
                peak_snapshot::field("stewardEvictionPlan",analyzer.stewardEvictionPlan);
                peak_snapshot::field("operators",analyzer.definitions.operatorNames());
                peak_snapshot::field("expressionsFromConfig",analyzer.definitions.configuredNames());
            }
        });


    if (bodies.empty()) return;

    // Every deactivateUnnecessary walk runs in this kernel's theorem drains
    // or in the in-run or seam, and is settled there; an owed sweep at entry
    // means a walk ran outside them, and its prune would miss the barrier
    // discharge (D-451).
    assert(!deactivationSweepOwed
        && "proveKernel entry: a deactivateUnnecessary walk outside the theorem drains left its sweep unsettled");

    // The reserved single-threaded slot serves the seams between iterations;
    // no scope on it is open here, so its blocks go back to the pool before
    // this iteration's phases draw from it.
    const unsigned seamScratchSlot = genScratchArenas().slotCount() - 1;
    releaseSlotScratch(scratchArenas(), genScratchArenas(), ruleStagings(),
                       seamScratchSlot);


    // Reduced-or closure for every or already in the registry (GL-binary
    // load, external-theorem precompile, prior seams) BEFORE this
    // iteration's bursts can flat-consume an or statement. Single-threaded
    // here — workers spawn later (I-137); idempotent, so the per-iteration
    // re-run costs one deduping registry scan.
    orRuleCompiler.preMintReducedOrs();

    const unsigned workers = logicalCores;

    // Active set at this cycle's start. The barriered phases run phase 1 / 2 / 3
    // over exactly the LBs a per-LB loop would have stepped
    // (active when pulled): intra-cycle deactivation only flips active -> inactive
    // (a child's reactivation is in the post-join drain, I-28), and an LB never
    // reads another's state intra-cycle, so collecting the set once here matches
    // the old `if(!isActive) continue` admission. See D-114.
    std::vector<Memory*> active;
    active.reserve(bodies.size());
    // Quiescent-burst skip (D-194): sweep an active LB iff it
    // may produce work this burst. `skipEnabled` gates the whole filter off during
    // warm-up (belt), when the config disables it, and in compressor mode (LBs
    // unregistered in the mail log, so mailPeek would be invalid — gated out). An
    // eligible LB is skipped only when it has NO producer-side work (hasWork clear)
    // AND NO un-ingested ancestor mail (mailPeek false). mailPeek is polled fresh
    // each iteration, so a cross-LB commit at iteration N wakes the recipient at
    // the N+1 active-build — the identical iteration the reference would pull, so
    // no mail latency is added (I-55). The predicate reads ONLY never-deloaded
    // logical state (hasWork on the LB slab, mailPeek on the mail pool); it never
    // reads resident() / blocksInUse() — the I-106 / I-108 / D-149 determinism
    // doctrine. A skipped LB stays isActive and keeps its claim/pager state
    // untouched: it is unswept this round, not deactivated, so the pager's sweep
    // window never includes it and it evicts and stays evicted at 4 GiB.
    // One burst stamp for the Phase 2 measurements recorded in this call's
    // phase 1 (recordPhase2Measurement).
    ++phase2MeasurementEpoch;

    const bool skipEnabled = parameters.enable_quiesce_skip
        && !parameters.compressor_mode && !ceFilteringActive && !warmUpPhase;
    int skipped = 0;
    for (Memory* b : bodies) {
        if (!(b && b->isActive)) continue;
#ifdef QUIESCE_SHADOW_CHECK
        // Shadow: record the predicate's verdict but NEVER actually skip — the LB
        // is swept and performElemPhase3 asserts its "quiescent" burst was a no-op.
        b->shadowWouldSkip = skipEnabled && !b->hasWork && !mailLog.mailPeek(b);
        active.push_back(b);
#else
        if (skipEnabled && !b->hasWork && !mailLog.mailPeek(b)) { ++skipped; continue; }
        active.push_back(b);
#endif
    }
    lastSweptCount = static_cast<int>(active.size());
    lastSkippedCount = skipped;

#if RT_MEASUREMENT
    // [RT phase measurement - file-bound, see .rt/burst_phases.log]
    // Per-LB wall-time ledgers for this iteration, indexed like `active`:
    // phase-1 sweep, phase-2 executor compute, phase-2 finalize, phase-3 sweep,
    // plus a copy of the split-invariant submatch work for the report.
    std::vector<int64_t> trapPh1Ns(active.size(), 0);
    std::vector<int64_t> trapPh2Ns(active.size(), 0);
    std::vector<int64_t> trapFinNs(active.size(), 0);
    std::vector<int64_t> trapPh3Ns(active.size(), 0);
    std::vector<int64_t> trapWork(active.size(), 0);
#endif

    // Once-per-batch LB-size telemetry (this batch's first kernel barrier,
    // single-threaded): the active-LB count and the resident LBs' blocksHeld
    // distribution — raw inputs for later tuning constants (Rule 16 / I-44,
    // never a prover input). Gathered only on the first barrier via the guard.
    if (deloadStats().lbHistogramPending()) {
        std::vector<int64_t> blocksHeld;
        blocksHeld.reserve(active.size());
        for (Memory* b : active)
            if (b->lbMemory.manager.resident())
                blocksHeld.push_back(b->lbMemory.manager.blocksHeld());
        deloadStats().reportLbHistogram(static_cast<int>(active.size()),
                                        blocksHeld);
    }

    // Active-LB eviction is now CONTINUOUS — the steward drains the working
    // set in every phase window (the working-set pager,
    // D-161), not via a barrier-armed plan. No
    // kernel-entry arming step.

    // Fresh per-iteration staging page set for deferred cross-LB admission seeds
    // (I-28): emplace a Filling set BEFORE phase-1 (single-threaded), sealed +
    // freed + reset by the post-join drain below. Non-movable SealedPageSet in a
    // std::optional stays at a stable address so its SealedString views hold.
    deferredAncestorPages.emplace();
    deferredAncestorPages->bind(&staticMemory());

    // Fresh per-iteration staging page set for proven single-chapter theorems (Part B of
    // the row-338 statification): dischargeToBeProved / dischargeContradiction
    // seal each theorem's bytes here so they cross the parallel -> join boundary
    // verbatim (the LB deloads after its burst); drained by
    // drainUpdateGlobalSingleChapter below. Non-movable in a std::optional -> stable
    // address for its SealedString views.
    updateGlobalSingleChapterPages.emplace();
    updateGlobalSingleChapterPages->bind(&staticMemory());

    // Barriered phase model: phase 1 for ALL LBs, join, phase 2 for ALL, join,
    // phase 3 for ALL, join. Each phase is its own work-stealing sweep over the
    // active set; the join between phases is the barrier. This is byte-identical
    // to the per-LB performElem loop (phase 1/2/3 are pure-extraction helpers and
    // no cross-LB observation happens intra-cycle — all cross-LB mail is delayed
    // to the next cycle via the MailLog commit barrier + phase-1 pull). Phase 2 is the
    // hashburst — the phase where a heavy LB's split gives it N executor threads.
    // `runPhase` is a local generic-lambda helper (Rule 18 exempt).
    auto runPhase = [&active, workers](std::atomic<std::size_t>& next,
                                       auto&& body_fn
#if RT_MEASUREMENT
                                       , int64_t* perLbNs = nullptr
#endif
                                       ) {
        auto worker = [&active, &next, workers, &body_fn
#if RT_MEASUREMENT
                       , perLbNs
#endif
                       ](unsigned coreId) {
            const unsigned cid = workers ? (coreId % workers) : 0U;
            for (;;) {
                std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= active.size()) break;
#if RT_MEASUREMENT
                // [RT phase measurement] per-LB wall time; each index i
                // is dispatched to exactly one worker, so the write is race-free.
                const auto trapT0 = std::chrono::steady_clock::now();
#endif
                body_fn(*active[i], cid);
#if RT_MEASUREMENT
                if (perLbNs != nullptr)
                    perLbNs[i] += std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - trapT0).count();
#endif
            }
            };
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (unsigned t = 0; t < workers; ++t) pool.emplace_back(worker, t);
        for (auto& th : pool) th.join();
        };

#if PHASE13_DEEP_TIMING
    std::vector<int64_t> phase1DetailRows(
        static_cast<std::size_t>(workers) * kPhase13TimingSlotCount, 0);
    std::vector<int64_t> phase3DetailRows(
        static_cast<std::size_t>(workers) * kPhase13TimingSlotCount, 0);
    static constexpr const char* phase13TimingLabels[] = {
        "claim_load",
        "burst_setup",
        "routing_mail_pull",
        "external_origin_absorb",
        "external_statement_absorb",
        "internal_origin_absorb",
        "internal_statement_absorb",
        "internal_mail_clear",
        "apply_equivalence_classes",
        "enrich_products_of_recursion",
        "cleanup_expressions",
        "discharge_contradiction",
        "discharge_to_be_proved",
        "discharge_contradiction_scopes",
        "fill_mail_out",
        "changed_classes_clear",
        "routing_mail_cleanup",
        "react_to_hypothesis",
        "sanitize_to_be_proved",
        "drain_disproved_goals",
        "drain_dead_or_branches",
        "freeze_resolved_or_branches",
        "drain_pending_or_releases",
        "wipe_subtrees",
        "sweep_ancestor_known_rows",
        "quiescence_and_dumps",
        "release_claim"
    };
    static_assert(sizeof(phase13TimingLabels) / sizeof(phase13TimingLabels[0])
                      == kPhase13TimingSlotCount,
                  "Phase 1/3 timing labels must cover every slot");
    const auto printPhase13Detail =
        [workers](int phase, double barrierSeconds,
                  const std::vector<int64_t>& rows) {
            int64_t categorizedNs = 0;
            diagnosticsLog() << "[PHASE13-DETAIL] phase=" << phase
                      << " barrier_seconds=" << barrierSeconds;
            for (std::size_t slot = 0; slot < kPhase13TimingSlotCount; ++slot) {
                int64_t slotNs = 0;
                for (unsigned worker = 0; worker < workers; ++worker) {
                    slotNs += rows[static_cast<std::size_t>(worker)
                                       * kPhase13TimingSlotCount
                                   + slot];
                }
                categorizedNs += slotNs;
                diagnosticsLog() << ' ' << phase13TimingLabels[slot] << "_worker_seconds="
                          << static_cast<double>(slotNs) / 1e9;
            }
            diagnosticsLog() << " categorized_worker_seconds="
                      << static_cast<double>(categorizedNs) / 1e9
                      << std::endl;
        };
#endif

    // Phase 1 opens a working-set window over its dispatch cursor
    // (D-161, I-114): the steward
    // prefetches the upcoming LBs and, above the watermarks, drains the
    // deloadable rest. Every LB's handshake (claimAndLoadForWork) makes it
    // resident before its body and releases the claim after, so the steward
    // can reclaim it once done.
    const auto phase1Started = std::chrono::steady_clock::now();
    {
        std::atomic<std::size_t> phase1Cursor{ 0 };
#if PHASE13_DEEP_TIMING
        phase13TimingRows = phase1DetailRows.data();
        phase13TimingWorkers = workers;
#endif
        gl::memory_census::markPhase(gl::PoolPhase::Phase1);
        steward->beginPhaseWindow(/*phase=*/1, &phase1Cursor, &active, workers,
                                  lbdeload::kDeloadDirectory);
        runPhase(phase1Cursor, [this](Memory& b, unsigned cid) {
            gl::WorkerContext::setParallelWorkerPhase(true);
            this->performElemPhase1(b, cid);
            gl::WorkerContext::setParallelWorkerPhase(false);
            // The task is over and every scratch-backed local of the body is
            // destroyed: hand this slot's scratch blocks back to the pool.
            releaseSlotScratch(scratchArenas(), genScratchArenas(),
                               ruleStagings(), cid);
        }
#if RT_MEASUREMENT
        , trapPh1Ns.data()
#endif
        );
        steward->endPhaseWindow();
#if PHASE13_DEEP_TIMING
        phase13TimingRows = nullptr;
        phase13TimingWorkers = 0;
#endif
    }
    const double phase1IterationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase1Started).count();

    // The always-on Phase 2 measurement starts after the phase-1 barrier and
    // stops before phase 3. It includes scheduler setup, processor or CUDA
    // execution, projection/transfers, sealing, finalization, and split stats.
    const auto phase2Started = std::chrono::steady_clock::now();
    static std::atomic<uint64_t> phase2TraceSequence{0};
    const uint64_t phase2TraceCall = gl::phase2TraceEnabled()
        ? phase2TraceSequence.fetch_add(1, std::memory_order_relaxed) + 1 : 0;
    gl::Phase2TraceScope phase2Trace("GL.Phase2", phase2TraceCall, gl::phase2TraceEnabled(), true);
    const bool mainPath = phase2Executor.execute(active, workers, phase2TraceCall, phase2Started, phase2Trace
#if RT_MEASUREMENT
        , trapPh2Ns, trapFinNs, trapWork
#endif
    );

    // Phase 3 opens the same working-set window (all phases equivalent).
    const auto phase3Started = std::chrono::steady_clock::now();
    {
        std::atomic<std::size_t> phase3Cursor{ 0 };
#if PHASE13_DEEP_TIMING
        phase13TimingRows = phase3DetailRows.data();
        phase13TimingWorkers = workers;
#endif
        gl::memory_census::markPhase(gl::PoolPhase::Phase3);
        steward->beginPhaseWindow(/*phase=*/3, &phase3Cursor, &active, workers,
                                  lbdeload::kDeloadDirectory);
        runPhase(phase3Cursor, [this](Memory& b, unsigned cid) {
            gl::WorkerContext::setParallelWorkerPhase(true);
            this->performElemPhase3(b, cid);
            gl::WorkerContext::setParallelWorkerPhase(false);
            // The task is over and every scratch-backed local of the body is
            // destroyed: hand this slot's scratch blocks back to the pool.
            releaseSlotScratch(scratchArenas(), genScratchArenas(),
                               ruleStagings(), cid);
        }
#if RT_MEASUREMENT
        , trapPh3Ns.data()
#endif
        );
        steward->endPhaseWindow();
        for (gl::GlobalMemoryManager* manager : { &gl::staticMemory(),
                &gl::persistentMemory(), &gl::mailMemory(), &gl::lbMemory() })
            manager->trimRecycledBlocks(gl::kRecycledWarmBytes);
        gl::memory_census::markPhase(gl::PoolPhase::Barrier);
#if PHASE13_DEEP_TIMING
        phase13TimingRows = nullptr;
        phase13TimingWorkers = 0;
#endif
    }
    const double phase3IterationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase3Started).count();
    diagnosticsLog() << "[PHASE13-TIMING] phase1_seconds="
              << phase1IterationSeconds
              << " phase3_seconds=" << phase3IterationSeconds
              << " active_logical_blocks=" << active.size() << std::endl;
#if RT_MEASUREMENT
    // Register this iteration's phase wall-clock with the RT aggregate so
    // its header can state attributed worker-seconds against the single
    // timeline (effective parallelism per phase).
    gl::rt_tracker::addRtPhaseWallSeconds(1, phase1IterationSeconds);
    gl::rt_tracker::addRtPhaseWallSeconds(3, phase3IterationSeconds);
#endif
#if PHASE13_DEEP_TIMING
    printPhase13Detail(1, phase1IterationSeconds, phase1DetailRows);
    printPhase13Detail(3, phase3IterationSeconds, phase3DetailRows);
#endif


#if RT_MEASUREMENT
    // [RT phase measurement - file-bound, see .rt/burst_phases.log]
    // Per-iteration per-LB wall-time report: per-phase totals, then the top LBs
    // by summed wall time with their FULL parentMemory chain (Rule 12 - the
    // [SPLIT] lines print only the ambiguous leaf exprKey). Observation only.
    //
    // Written to `.rt/burst_phases.log`, never stdout: this is measurement the
    // acceleration campaigns read, and the main run log stays free of it. The
    // per-batch / per-section split lives beside it in `.rt/_aggregate_<tag>.log`.
    if (mainPath && !active.empty()) {
        auto trapSec = [](int64_t ns) {
            return static_cast<double>(ns) / 1e9; };
        std::filesystem::create_directories(".rt");
        std::ofstream phaseLog(".rt/burst_phases.log", std::ios::app);
        int64_t s1 = 0, s2 = 0, sf = 0, s3 = 0;
        for (std::size_t li = 0; li < active.size(); ++li) {
            s1 += trapPh1Ns[li]; s2 += trapPh2Ns[li];
            sf += trapFinNs[li]; s3 += trapPh3Ns[li];
        }
        phaseLog << "[BURST-PH] ph1=" << trapSec(s1) << "s ph2=" << trapSec(s2)
                  << "s fin=" << trapSec(sf) << "s ph3=" << trapSec(s3) << "s\n";
        std::vector<std::size_t> trapOrder(active.size());
        for (std::size_t li = 0; li < trapOrder.size(); ++li) trapOrder[li] = li;
        std::sort(trapOrder.begin(), trapOrder.end(),
            [&](std::size_t a, std::size_t b) {
                const int64_t ta = trapPh1Ns[a] + trapPh2Ns[a]
                                 + trapFinNs[a] + trapPh3Ns[a];
                const int64_t tb = trapPh1Ns[b] + trapPh2Ns[b]
                                 + trapFinNs[b] + trapPh3Ns[b];
                return ta > tb; });
        const std::size_t trapTopK = trapOrder.size();
        for (std::size_t k = 0; k < trapTopK; ++k) {
            const std::size_t li = trapOrder[k];
            const int64_t tot = trapPh1Ns[li] + trapPh2Ns[li]
                              + trapFinNs[li] + trapPh3Ns[li];
            std::string chain;
            for (const Memory* p = active[li]; p != nullptr;
                 p = p->parentMemory) {
                if (!chain.empty()) chain += " <- ";
                const std::string key = p->exprKey();
                chain += key.empty() ? std::string("(root)") : key;
            }
            phaseLog << "[RT-LB] tot=" << trapSec(tot)
                      << "s ph1=" << trapSec(trapPh1Ns[li])
                      << "s ph2=" << trapSec(trapPh2Ns[li])
                      << "s fin=" << trapSec(trapFinNs[li])
                      << "s ph3=" << trapSec(trapPh3Ns[li])
                      << "s work=" << trapWork[li]
                      << " chain=" << chain << "\n";
        }
    }
#endif

    proofBarrier.finish(bodies, active, workers, seamScratchSlot);
}


























// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void Prover::prove(int numberIterations,
    const std::vector<Memory*>& bodies) {

    // Steward scope (D-159): the background memory
    // custodian lives exactly as long as this iteration loop. Started
    // parked; every kernel barrier quiesces it before reading state;
    // the guard quiesces + stops + destroys it before prove() returns,
    // so destroyGrid, the CE filter, and every post-prove reader see a
    // steward-free world.
    assert(!steward && "nested prove() steward scopes");
    // Plans never cross prove scopes: a plan computed at a previous
    // scope's final barrier was never armed and is dropped here (the
    // kernel re-plans from live state at its next pressured barrier).
    stewardEvictionPlan.clear();
    // Counterexample-filter clones (the CUDA filter proves them in batches
    // through this scope) are destroyed right after it, so their steward is
    // resident-only: no clone is ever written to SSD.
    steward = std::make_unique<MemorySteward>(
        parameters.allow_ssd_deload && !ceFilteringActive);
    // The worker count sizes the I/O executor pool (steward::ioThreadCountFor
    // — 4 executors at 32 workers).
    steward->start(logicalCores);
    // FORENSIC EXHAUSTION TRAP: at a main-pool wall, grantLocked prints the
    // pool counters and this census (the grid + steward view) to stderr
    // BEFORE the exhaustion assert aborts — resolving policy-missed-victims
    // versus pinned-working-set at the point of death. The census runs under
    // the pool mutex; printExhaustionCensus is manager-mutex-free by
    // construction (its contract). Cleared first in the scope guard so no
    // post-prove grant ever runs a reporter over a stale grid reference.
    staticMemory().setExhaustionReporter(
        [&bodies, s = steward.get()](int64_t inUse, int64_t total) {
            printExhaustionCensus(bodies, s, inUse, total);
        });
    // GRANT BACKPRESSURE (D-422): a grant that would
    // leave fewer free blocks than the emergency floor first evicts one idle
    // logic block behind the open window's cursor, so growth inside claimed
    // blocks cannot outrun the SSD evictions. Cleared with the reporter
    // below, while the steward still lives.
    if (steward->allowsSsdDeload())
        staticMemory().setPressureValve(
            [s = steward.get()] { s->relieveGrantPressure(lbdeload::kDeloadDirectory); },
            steward->emergencyFloorBlocks());
    struct StewardScope {
        std::unique_ptr<MemorySteward>& s;
        ~StewardScope() {
            // TEARDOWN LOAD-SHEDDING: after the last iteration, the final
            // barrier's head-prefetch loads (and any straggler window tasks)
            // are pure waste — there is no next iteration to warm, no window
            // is open (the planner cannot evict to make room), and the
            // executor load path has no valve. Discard everything queued but
            // unstarted (defined load-shedding, the ring-drop doctrine);
            // in-flight tasks finish and the quiesce below waits for them.
            // The unbudgeted, undiscarded form of these loads was the 4 GiB
            // teardown wall (D-196).
            s->discardQueuedIoTasks();
            // The final barrier may have re-armed the grant trigger; the
            // chapter export's post-prove reloads acquire blocks and would
            // fire a trigger pointing at this stopped steward (the
            // wake-without-work assert). No armed trigger may outlive its
            // steward — post-prove block traffic is not a steward signal
            // (D-158).
            staticMemory().disarmGrantTrigger();
            s->quiesce();
            s->stop();
            // The exhaustion reporter clears LAST — after every teardown
            // action that can still grant (the in-flight executor loads the
            // quiesce just waited out), so a teardown exhaustion prints a
            // full census. Lifetime: the reporter captures prove()'s
            // `bodies` reference (alive for the whole prove scope, including
            // this guard) and the steward pointer (alive until the reset
            // below) — both outlive every possible invocation.
            staticMemory().setExhaustionReporter(nullptr);
            staticMemory().setPressureValve(nullptr, 0);
            s.reset();
        }
    } stewardScope{ steward };

    for (int it = 0; it < numberIterations; ++it) {
        // The batch's proof bursts, warm-up included; the counterexample
        // filter's bursts on its clones do not count
        // (D-521).
        if (!ceFilteringActive) ++proofBurstNumber;
        // Debug: count total expressions across all active bodies
        std::size_t totalExprs = 0;
        std::size_t activeBodies = 0;
        for (std::size_t bi = 0; bi < bodies.size(); ++bi) {
            if (bodies[bi] && bodies[bi]->isActive) {
                // Deload-tolerant count: most LBs are cold between
                // iterations and must not reload just to be counted.
                totalExprs += static_cast<std::size_t>(
                    bodies[bi]->intEncodedStatementsCount());
                ++activeBodies;
            }
        }
        std::cout << "Hash burst: " << it
                  << "  active_bodies=" << activeBodies
                  << "  total_exprs=" << totalExprs << std::endl;
        std::cout.flush();
        // Mirror the burst boundary into the diagnostics log so its telemetry
        // lines stay attributable to their hashburst.
        diagnosticsLog() << "--- hashburst " << it
                         << " active_bodies=" << activeBodies
                         << " total_exprs=" << totalExprs
                         << " ---" << std::endl;

        // Publish the burst index so every RTTracker constructed by the
        // upcoming proveKernel captures it in its `.rt/<chain>.log`
        // header. Written here on the main thread between bursts; read
        // by worker threads inside the phase helpers. No
        // concurrent write while workers run, so plain int is safe.
        ::gl::rt_tracker::g_currentHashburstIndex = it;

        auto burstStart = std::chrono::high_resolution_clock::now();
        proveKernel(bodies);
        auto burstEnd = std::chrono::high_resolution_clock::now();
        double burstSec = std::chrono::duration<double>(burstEnd - burstStart).count();
        // active_bodies (above) is the PRE-skip count; swept is the post-skip
        // sweep size and skipped the quiescent LBs excluded this iteration
        // (D-194). Telemetry only (Rule 16 / I-44).
        std::cout << "  dt=" << burstSec << "s"
                  << "  swept=" << lastSweptCount
                  << "  skipped=" << lastSkippedCount << std::endl;
        std::cout.flush();

        // Global deactivation sweep once per iteration, post-drain. The
        // sweep used to run only from deactivateUnnecessary's tail — i.e.
        // only when a theorem was emitted — so LBs whose last main-scope
        // goal closed after the batch's final emission stayed active to
        // batch end. Same I-48 predicate as before, evaluated every
        // iteration on this single-threaded seam; reads only never-deloaded
        // state (intToBeProved persistent I-108, isActive + tree edges on
        // the LB slab I-109/I-110).
        scopeLifecycle.deactivateRecursively();

    }

}













std::vector<std::string> Prover::runCeFilterOnly(
    const std::vector<std::string>& theorems) {

    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "Counter example based filtering started." << std::endl;

    std::vector<std::string> filteredConjectures = theorems;

    if (!parameters.skip_ce_filter) {
        if (parameters.mirror_refutation) {
            const auto mirrorPairsPath =
                gl::projectRoot() / "files" / "theorems"
                    / "mirror_pairs.txt";
            mirrorPartnerMap = loadMirrorPairs(mirrorPairsPath);
            std::cout << "CE filter: " << mirrorPartnerMap.size()
                      << " mirror-partner keys loaded." << std::endl;
        }
        std::vector<std::vector<std::string>> simpleFactsLists = counterexampleFilter.readSimpleFacts();
        std::cout << "CE filter: " << filteredConjectures.size() << " conjectures, "
                  << simpleFactsLists.size() << " fact files." << std::endl;
        for (const auto& simpleFacts : simpleFactsLists)
        {
            filteredConjectures = filterConjecturesWithCE(filteredConjectures, simpleFacts);
            std::cout << "CE filter: " << filteredConjectures.size() << " survived." << std::endl;
        }
        counterexampleFilter.saveFilteredConjectures(filteredConjectures);
    }

    std::cout << "Counter example based filtering finished." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "Runtime counter example based filtering: " << dt.count() << " seconds\n";

    return filteredConjectures;
}



// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void Prover::analyzeExpressions(const std::vector<std::string>& theorems,
    const std::vector<std::string>& provedTheorems,
    const std::vector<std::string>& externalTheorems) {

    compilationReuse.freezePublicNames();

    const auto startupSeconds = [](std::chrono::steady_clock::time_point from,
                                   std::chrono::steady_clock::time_point to) {
        return std::chrono::duration<double>(to - from).count();
    };
    const auto purgeStarted = std::chrono::steady_clock::now();
    // Statification: one gl_quick invocation = one batch — empty the
    // deload directory at batch START only (end-of-run files stay on
    // disk for post-run inspection). The accumulated manifest resets
    // with it.
    // Close any prior batch's extent file BEFORE the purge deletes it (an open
    // handle blocks the delete on Windows), then purge + reopen a fresh one.
    staticMemory().closeExtentFile();
    lbdeload::purgeDeloadDirectory(lbdeload::kDeloadDirectory);
    staticMemory().resetDeloadRegistry();
    if (parameters.allow_ssd_deload && parameters.enable_extent_deload) {
        // Preallocate 1.5x the pool: live extent bytes are the Dumped (non-
        // resident) LBs' slabs, so the extent legitimately exceeds pool size
        // (disk > RAM is the point of deload). Growth covers under-estimates.
        const int64_t initialBytes =
            staticMemory().totalBlocks() * staticMemory().blockBytes() * 3 / 2;
        staticMemory().openExtentFile(
            std::filesystem::path(lbdeload::kDeloadDirectory) / "extent.bin",
            initialBytes);
    }
    pendingDischarge.clear();
    stewardEvictionPlan.clear();
    const auto ceFilterStarted = std::chrono::steady_clock::now();

    // 1. CE Filtering — extracted to `runCeFilterOnly` so the `--ce-only`
    // fast-iteration entry point can reuse the exact same code path. Returns
    // the surviving conjectures (== `theorems` when `skip_ce_filter` is true).
    std::vector<std::string> filteredConjectures = runCeFilterOnly(theorems);
    const auto preparationStarted = std::chrono::steady_clock::now();

    // =========================================================================
    // Helper: compile existence heads and register conjectures into grid
    // =========================================================================
    // Maps raw (uncompiled) conjecture string → compiled form (for existence heads)
    std::unordered_map<std::string, std::string> rawToCompiled;

    proofGridBuilder.prepareBatch(filteredConjectures, rawToCompiled,
        structuralCompiler, orTheoremBuilder, globalDependencies,
        blockMailProcessor, dormantLogicBlocksAtGridBuild, rollingMailHistoryEnabled);
    const auto preparationDone = std::chrono::steady_clock::now();
    startupTimings.extentPurgeSeconds = startupSeconds(purgeStarted, ceFilterStarted);
    startupTimings.ceFilterSeconds = startupSeconds(ceFilterStarted, preparationStarted);
    startupTimings.batchPreparationSeconds = startupSeconds(preparationStarted, preparationDone);
    std::cout << "Runtime batch preparation: " << startupTimings.batchPreparationSeconds
              << " seconds" << std::endl;

    int64_t initialWitnessCounterSum = 0;
    const size_t initialBodyCount = permanentBodies.size();
    for (const Memory* memory : permanentBodies) initialWitnessCounterSum += memory->startInt;
    std::cout << "Prover started." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // SINGLE PASS: prove → compress → head-switch (orPairsFromHeadSwitch only).
    //
    // The head-switched mirrors of every structurally-qualifying conjecture
    // were folded into the conjecture pool already (see the pre-emit pass at
    // the top of analyzeExpressions, just before compileAndRegister). So one
    // prove call is enough — there is nothing the legacy big-iteration second
    // pass would have reached that the first pass does not. The post-prove
    // head-switch block here exists solely to populate `orPairsFromHeadSwitch`
    // for run_modes.cpp's OR-theorem construction. See D-24.
    // =========================================================================

    // ----- PROVE (warm-up + main) -----
    {
        int preIterations = 2;
        if (parameters.maxIterationNumberProof < preIterations)
            preIterations = parameters.maxIterationNumberProof;

        if (preIterations > 0) {
            std::cout << "Phase 1: Running " << preIterations << " warm-up iterations..." << std::endl;
            // Quiescence: warm-up bursts never skip (D-194) —
            // they are productive priming and the belt keeps every active LB swept.
            this->warmUpPhase = true;
            this->prove(preIterations, permanentBodies);
            this->warmUpPhase = false;
        }

        int remainingIterations = parameters.maxIterationNumberProof - preIterations;

        // Precompile structural operators in proved theorems before broadcasting.
        // Proved theorems from previous batches may contain !(&...) (OR) or
        // !(>[...]) (existence) that need compilation BEFORE disintegration.
        std::vector<std::string> compiledProved = provedTheorems;
        for (auto& pt : compiledProved) {
            structuralCompiler.precompileStructuralOperators(pt);
        }

        if (!compiledProved.empty()) {
            std::cout << "Injecting " << compiledProved.size() << " proved theorems via broadcast..." << std::endl;
            // Theorems proved by earlier batches: stored in the root's log
            // (every LB outside incubator mode; only the anchor LB inside it --
            // D-332). In incubator mode a
            // nested rule (anchorOnlyRule false) goes to the anchor LB's log
            // instead, so the premise LBs below the anchor receive it.
            blockMailProcessor.broadcastTheorems(compiledProved, "broadcast", this->body,
                parameters.incubator_mode ? blockMailProcessor.incubatorAnchorLb() : nullptr);
        }

        if (!externalTheorems.empty()) {
            // Same precompile treatment as provedTheorems — externals may
            // carry raw !(&...) / !(>...) that would otherwise reach LB
            // memory uncompiled.
            std::vector<std::string> compiledExternals = externalTheorems;
            for (auto& et : compiledExternals) {
                structuralCompiler.precompileStructuralOperators(et);
            }
            // Keep the compiled twins for the post-run registry export —
            // base-form externals are registry-independent on the wire, but
            // chapter citations carry these compiled forms.
            this->precompiledExternalsExport = compiledExternals;
            std::cout << "Injecting " << compiledExternals.size()
                      << " external theorems via broadcast..." << std::endl;
            // Externals are multi-premise pool lemmas that fire in the premise
            // LBs below the anchor, so they must reach EVERY LB: in incubator
            // mode the root's log has one reader (the anchor LB), hence the
            // batch is stored in the anchor LB's log, which the anchor LB
            // (self-injected) and every LB below it read.
            Memory& externalsStore = parameters.incubator_mode
                ? *blockMailProcessor.incubatorAnchorLb()
                : this->body;
            blockMailProcessor.broadcastTheorems(compiledExternals, "externally provided theorem",
                                    externalsStore, nullptr);
        }

        if (remainingIterations > 0) {
            std::cout << "Phase 2: Running remaining " << remainingIterations << " iterations..." << std::endl;
            this->prove(remainingIterations, permanentBodies);
        }
    }


    // Compression is invoked from run_modes.cpp post-analyzeExpressions,
    // gated by skipCompression (true under incubator_mode / ban_disintegration).
    // See D-25.

    // ----- HEAD SWITCH (orPairsFromHeadSwitch only) -----
    // The mirrors themselves are already in the conjecture pool (pre-emit
    // pass) and were proved in the single prove call above. The walk here
    // exists solely to populate `orPairsFromHeadSwitch` (class member) so
    // run_modes.cpp:267+ can construct OR theorems from (existence,
    // companion) pairs.
    {
        std::vector<std::tuple<std::string, std::string, std::string, std::string>> snapshot;
        {
            std::lock_guard<std::mutex> lock(theoremListMutex);
            snapshot = globalTheoremList;
        }

        for (const auto& tpl : snapshot) {
            const std::string& theorem = std::get<0>(tpl);
            // Zero-circulation contract: proved-not-broadcast rows never
            // seed the export seam's or construction either.
            if (std::get<1>(tpl) == "proved not broadcast") continue;
            std::string rebuilt = orTheoremBuilder.headSwitchOne(theorem);
            if (rebuilt.empty()) continue;

            std::cout << "Head switch: " << rebuilt << std::endl;
            orPairsFromHeadSwitch.push_back({theorem, rebuilt});
        }
    }
    proofRunReporter.reportCompletion(initialBodyCount, initialWitnessCounterSum, t0);
}



















} // namespace gl
