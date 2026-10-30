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
#include "scope_lifecycle.hpp"
namespace gl {
/// @brief Bind inputs for scope lifecycle.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param body Borrowed Memory interface or state; must outlive this component.
/// @param deactivationSweepOwed Borrowed bool interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param globalDependencies Borrowed Dependencies interface or state; must outlive this component.
/// @param pendingSubtreeDeactivations Borrowed std::vector interface or state; must outlive this component.
/// @param pendingSubtreeDeactivationsMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param pendingTwinDeactivations Borrowed std::vector interface or state; must outlive this component.
/// @param pendingTwinDeactivationsMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param anchorScopeQueries Borrowed AnchorScopeQueries interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
ScopeLifecycle::ScopeLifecycle(Memory& body, bool& deactivationSweepOwed, const CompiledDefinitionRegistry& definitions, Dependencies& globalDependencies, std::vector<Memory*>& pendingSubtreeDeactivations, std::mutex& pendingSubtreeDeactivationsMutex, std::vector<Memory*>& pendingTwinDeactivations, std::mutex& pendingTwinDeactivationsMutex, SimpleMapStore& simpleMapStore, AnchorScopeQueries& anchorScopeQueries) : body(body), deactivationSweepOwed(deactivationSweepOwed), definitions(definitions), globalDependencies(globalDependencies), pendingSubtreeDeactivations(pendingSubtreeDeactivations), pendingSubtreeDeactivationsMutex(pendingSubtreeDeactivationsMutex), pendingTwinDeactivations(pendingTwinDeactivations), pendingTwinDeactivationsMutex(pendingTwinDeactivationsMutex), simpleMapStore(simpleMapStore), anchorScopeQueries(anchorScopeQueries) {}

//#pragma optimize("", off)


// Helper to recursively deactivate a block and ONLY its induction-related descendants.
// Helper to recursively deactivate blocks starting from the root.
// Traverses the entire tree and deactivates any node where:
// 1. All its children are inactive.
// 2. It has no pending proofs (toBeProved is empty).
/// @brief Walk the entire LB tree marking inactive LBs as torn down.
///
/// @details
/// Called between outer iterations to reclaim memory for LBs whose
/// purpose is complete. Walks every LB depth-first; when the
/// deactivation predicate fires, sets `isActive = false` and frees
/// per-LB state (intEncodedStatements, hash memories,
/// equivalence-class state). Children of a deactivated LB are also
/// deactivated.
///
/// Cached proof stacks (`cachedProofStacks`) are populated in a
/// pre-teardown pass before this walk so theorems proven inside
/// torn-down LBs still have their proof stack at export time.
///
/// @see `deactivateUnnecessary` — narrower variant that targets a
///      specific theorem's LBs.
/// @see `cachedProofStacks` (member) — per-theorem stack cache.
void ScopeLifecycle::deactivateRecursively() {
    // Recursive lambda for post-order traversal
    std::function<void(Memory*)> traverse =
        [&](Memory* node) {
        if (!node) return;

        // 1. Recurse into children first (Post-order)
        simpleMapStore.forEachChild(node, [&](const gl::StrSpan&, Memory* child) {
            if (child) {
                traverse(child);
            }
        });

        // 2. Check if all children are deactivated — an order-free question,
        //    so the unordered probe answers it without sorting the children.
        const bool allChildrenDeactivated = !simpleMapStore.anyChild(node,
            [](const Memory* child) { return child && child->isActive; });

        // 3. Deactivate node if conditions are met:
        //    - All children are inactive
        //    - No pending proofs in the current node

        // The survey reads intToBeProved, which now lives in the PERSISTENT pool
        // and is always resident (I-108) — so it runs on
        // every ACTIVE node regardless of whether the node's MAIN arena is
        // deloaded. This is also what makes the quiescent-burst skip
        // (D-194) preserve the I-48 deactivation schedule with
        // zero extra work: this survey is a tree-wide, post-join walk from the
        // root, so a SKIPPED LB (unswept this iteration but still isActive) is
        // still surveyed here, reading only never-deloaded state (intToBeProved
        // persistent I-108; child isActive + simpleMap edges on the never-deloaded
        // LB slab, I-109 / I-110) — it forces no reload of a cold skipped LB.
        // Gating on isActive (not residency) is the determinism fix:
        // the old residency gate SKIPPED a deloaded-but-active node, and which
        // nodes are deloaded is timing-dependent, so the deactivation decision
        // (and thus the theorem set) was non-deterministic. An already-inactive
        // node needs no survey (deactivation is monotonic; isActive implies
        // !dischargedForever, so the persistent registry is still populated).
        if (node->isActive) {
            const int mainValidityCount = mainGoalCount(*node);

            if (allChildrenDeactivated && mainValidityCount == 0
                && !node->primedForContradiction) {
                node->isActive = false;
            }
        }
        };

    // Start traversal from the root of the Prover
    traverse(&this->body);
}

//#pragma optimize("", off)


/// @brief Deactivate the LBs that were created exclusively for proving
/// the given theorem and are no longer needed.
///
/// @details
/// Targeted variant of `deactivateRecursively`. Used at the end of
/// each theorem's main-batch proof to trim LBs that supported only
/// that theorem's hash bursts. The criterion is: an LB whose chain
/// of ancestors traces back to a unique `theorem`-creation site and
/// whose `intLocalEncodedStatementsSet` is empty after the proof
/// completes.
///
/// Same teardown protocol as `deactivateRecursively` — proof stacks
/// are cached pre-teardown via `cachedProofStacks`.
///
/// The closing tree-wide prune is owed (`deactivationSweepOwed`), not run
/// here: `proveKernel` settles it with one `deactivateRecursively` sweep
/// after the iteration's theorem drains
/// (D-451).
///
/// @param theorem Theorem text identifying the per-theorem LBs.
/// @param root    Root sentinel from which to walk.
void ScopeLifecycle::deactivateUnnecessary(const std::string& theorem, Memory& root) {
    // 1) Collect original indices for this theorem
    std::vector<int> originals;
    originals.reserve(8);
    for (const auto& kv : this->globalDependencies.originalAuxyMap) {
        if (kv.second.expr == theorem)
            originals.push_back(kv.first);
    }

    // 2) Disintegrate theorem to find its Memory path
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    const std::string head =
        ce::disintegrateImplication(theorem, tempChain, this->definitions.configurations());
    if (tempChain.empty()) return;

    // Walk down the chain from root, collecting the blocks on the path
    Memory* node = &root;
    std::vector<Memory*> chainBlocks;
    chainBlocks.reserve(tempChain.size());

    for (const auto& t : tempChain) {
        const std::string& key = std::get<0>(t);
        Memory* nextNode = simpleMapStore.findChild(node, key);
        if (nextNode == nullptr) {
            // Path not present (may have been pruned earlier) — nothing to do safely.
            return;
        }
        node = nextNode;
        chainBlocks.push_back(node);
    }

    // Retire both contradiction twins of this settled head at the final
    // node. Emission-time counterpart of the D-224 goal-closure staging:
    // the staging path fires only when the chain LB's own goal closes,
    // which for a theorem proved by induction or in another LB happens one
    // or more bursts after emission (on the broadcast re-fire) — this flip
    // closes that lag at the emission drain, the same single-threaded seam
    // the staged drain uses. A missing twin is a defined case (the head
    // never took contradiction LBs, or they were retired already).
    for (const std::string& seed : { head, gl::DefinitionExpansion::negate(head) }) {
        if (Memory* twin = simpleMapStore.findChild(node, "__contradiction__" + seed)) {
            twin->isActive = false;
        }
    }

    // Names depend only on the last node on the path
    const std::string zeroArgName = anchorScopeQueries.findZeroArgName(*node);
    const std::string sName = anchorScopeQueries.findSName(*node);
    if (zeroArgName.empty() || sName.empty()) {
        // Can't construct the expected keys — bail gracefully.
        return;
    }

    // 3) Deactivate per-original auxiliaries under the final node
    for (int originalIndex : originals) {
        auto itIV = this->globalDependencies.originalInductionVariableMap.find(originalIndex);
        if (itIV == this->globalDependencies.originalInductionVariableMap.end()) continue;

        const std::string& indVar = std::get<0>(itIV->second); // e.g. "i2"
        const std::string& recCounter = std::get<1>(itIV->second); // numeric string
        const std::string  recArg = std::string("rec") + recCounter;

        const std::string k_rec_in2 = "(in2[" + recArg + "," + indVar + "," + sName + "])";
        const std::string k_eq_side = "(=[s(" + recArg + ")," + zeroArgName + "])";

        if (Memory* c1 = simpleMapStore.findChild(node, k_rec_in2)) {
            c1->isActive = false;

        }
        if (Memory* c2 = simpleMapStore.findChild(node, k_eq_side)) {
            c2->isActive = false;

        }
        // No asserts: partial deactivation is still useful/safe.
    }

    // 4) Bubble up: if a block has no active children, deactivate it too
    for (auto rit = chainBlocks.rbegin(); rit != chainBlocks.rend(); ++rit) {
        Memory* block = *rit;
        // Order-free question: the unordered probe stops at the first active
        // child instead of sorting all of them (an anchor block can hold
        // thousands).
        const bool anyActiveChild = simpleMapStore.anyChild(block,
            [](const Memory* child) { return child && child->isActive; });
        // Disable the LB when no active children AND no MAIN-namescope
        // toBeProved entries remain. Entries at deeper scopes (hypothetical,
        // OR-branch, integration boundaries) do NOT keep the LB alive —
        // their proof obligations belong to the deeper context and the LB
        // itself has no work left at its own scope. If only deeper-scope
        // entries linger and no children are active, disable.
        // intToBeProved is in the PERSISTENT pool (always resident,
        // I-108), so the survey runs on every ACTIVE block
        // regardless of its MAIN arena's deload state. Gating on isActive (not
        // residency) is the determinism fix — the old residency gate skipped a
        // deloaded-but-active block, and which blocks are deloaded is
        // timing-dependent, so the deactivation decision was non-deterministic.
        if (!anyActiveChild && block->isActive)
        {
            // I-48 survey on the packed keys: the low 32 bits carry the
            // validity id, and "main" is the pre-registered MAIN_ID.
            if (mainGoalCount(*block) == 0) {
                block->isActive = false;
            }
        }
    }

    // The global prune of the tree is owed, not run here: proveKernel settles
    // it with one deactivateRecursively sweep right after the iteration's
    // theorem drains, before anything else reads isActive. The sweep is a
    // monotone fixpoint, so one sweep after all walks leaves the same inactive
    // set as one sweep per walk (D-451).
    deactivationSweepOwed = true;
}

/// @brief Whether the induction variable's typing fact `(in[n,N])` is known
///        at the goal block's `main` scope.
///
/// @details
/// The induction typing gate (I-18): an induction theorem ranges over the
/// anchor's natural-number set only when the conjecture's goal block knows
/// that the induction variable lies in it. `N` is the anchor argument
/// `AnchorScopeQueries::findNName` reads from the block chain. The induction
/// promotion (`TheoremPublisher::updateGlobal`), the step-win cover test and
/// the retry of a waiting promotion share this probe.
///
/// @param goalBlock         The conjecture's goal block (its innermost
///                          premise block).
/// @param inductionVariable The induction variable of the triad.
/// @return True when `(in[inductionVariable,N])` is known at `main`.
/// @invariant @p goalBlock is resident: the caller holds it through the
///            steward door or it was never deloaded.
/// @see TheoremPublisher::updateGlobal, settleStepClosure, retryWaitingPromotions.
bool ScopeLifecycle::inductionTypingKnown(Memory& goalBlock, const std::string& inductionVariable) {
    const std::string anchorN = anchorScopeQueries.findNName(goalBlock);
    const std::string typingGoal =
        std::string("(in[") + inductionVariable + "," + anchorN + "])";
    return lookupStatementFlags(goalBlock.intKnownStatements, goalBlock.nameMap,
                                typingGoal, "main") != nullptr;
}

/// @brief The premise levels an induction's proof must cover before the
///        induction theorem registers.
///
/// @details
/// The conjecture's chain keys sit at levels `0..n-1` (the anchor at
/// level 0); the step and base blocks live one level below the last key.
/// Required are all premise levels `1..n-1`, the induction variable's own
/// membership premise included: the induction principle consumes it, and
/// its levels enter the union through the typing fact
/// (`inductionTypingLevels`). The anchor level and the step or base block's
/// own level are never required (D-516).
///
/// @param chainKeys The conjecture's premise keys, anchor first.
/// @return One bit per required level.
/// @invariant @p chainKeys holds the anchor and at most 63 keys in all
///            (asserted).
/// @see TheoremPublisher::updateGlobal, settleStepClosure.
uint64_t ScopeLifecycle::inductionPremiseLevels(const std::vector<std::string>& chainKeys) {
    assert(!chainKeys.empty() && chainKeys.size() < 64
        && "inductionPremiseLevels: a chain without its anchor or deeper than the level mask");
    uint64_t required = 0;
    for (std::size_t level = 1; level < chainKeys.size(); ++level) {
        required |= uint64_t(1) << level;
    }
    return required;
}

/// @brief The levels of the induction variable's typing fact `(in[n,N])`
///        known at the goal block's `main` scope.
///
/// @details
/// The induction principle consumes the typing fact, so its levels belong
/// to the induction's level union beside the step and base proofs' levels
/// (D-516). A fact delivered as a premise
/// carries its premise's level; a derived fact carries the levels of its
/// derivation; the non-derived tier `{-1}` contributes nothing.
///
/// @param goalBlock         The conjecture's goal block.
/// @param inductionVariable The induction variable of the triad.
/// @return One bit per level of the fact's level row.
/// @invariant The fact is known in @p goalBlock (asserted; probe with
///            `inductionTypingKnown` first) and @p goalBlock is resident.
/// @see inductionTypingKnown, TheoremPublisher::updateGlobal.
uint64_t ScopeLifecycle::inductionTypingLevels(Memory& goalBlock, const std::string& inductionVariable) {
    const std::string anchorN = anchorScopeQueries.findNName(goalBlock);
    const std::string typingGoal =
        std::string("(in[") + inductionVariable + "," + anchorN + "])";
    const int32_t rowId = lookupStatementLevels(goalBlock.intStatementLevelsMap,
                                                goalBlock.nameMap, typingGoal, "main");
    assert(rowId != 0
        && "inductionTypingLevels: the typing fact has no level row in the goal block");
    int levels[256];
    const int32_t levelCount =
        coldIntRunNonNegAt(goalBlock.intStatementLevelsMap, rowId, levels, 256);
    uint64_t mask = 0;
    for (int32_t j = 0; j < levelCount; ++j) {
        assert(levels[j] <= goalBlock.level && levels[j] < 64
            && "inductionTypingLevels: a typing level outside the goal block's level universe");
        mask |= uint64_t(1) << levels[j];
    }
    return mask;
}

/// @brief Count the block's goals at the `main` scope.
///
/// @details
/// The I-48 survey: `intToBeProved` keys whose validity id is
/// `NameMap::MAIN_ID`. The goal registry lives in the persistent pool, so the
/// survey needs no residency door (I-108).
///
/// @param block The block to survey.
/// @return The number of main-scope goals.
int ScopeLifecycle::mainGoalCount(const Memory& block) const {
    int count = 0;
    for (int32_t tbpId = 1; tbpId <= block.intToBeProved.count(); ++tbpId) {
        // Low 32 key bits carry the validity id; "main" is MAIN_ID (I-48).
        if (Codec<StatementKey>::decode(block.intToBeProved.keyAt(tbpId)).validity
            == NameMap::MAIN_ID) {
            ++count;
        }
    }
    return count;
}

/// @brief Switch off the winning conjecture's other proof blocks.
///
/// @details
/// Both contradiction blocks of the head (`__contradiction__` + head and
/// `__contradiction__` + the negated head) and the step block of every
/// other triad of the conjecture are switched off with the flip contract of
/// `drainPendingTwinDeactivations` (`isActive` false,
/// `contradictionTheoremId` cleared, `dischargedForever` untouched). So is
/// every other triad's zero block that an earlier uncovered step woke; the
/// parked ones stay parked, since their step blocks can no longer close.
/// Absent contradiction blocks are the contracted no-twin case.
///
/// @param conjecture The conjecture expression of the win.
/// @param win        The conjecture's step-win record.
/// @return The number of blocks switched off.
/// @invariant Every triad of the conjecture has its step and zero block
///            under the goal block (asserted).
int ScopeLifecycle::retireStepWinSiblings(const std::string& conjecture, const StepWinRecord& win) {
    Memory* goalBlock = win.goalBlock;
    int switchedOff = 0;
    const auto switchOff = [&switchedOff](Memory* block) {
        if (!block->isActive) return;
        block->contradictionTheoremId = 0;
        block->isActive = false;
        ++switchedOff;
    };

    for (const std::string& seed : { win.head, gl::DefinitionExpansion::negate(win.head) }) {
        if (Memory* twin = simpleMapStore.findChild(goalBlock, "__contradiction__" + seed)) {
            switchOff(twin);
        }
    }

    const std::string zeroArgName = anchorScopeQueries.findZeroArgName(*goalBlock);
    const std::string sName = anchorScopeQueries.findSName(*goalBlock);
    for (const auto& kv : this->globalDependencies.originalAuxyMap) {
        if (kv.first == win.originalIndex || kv.second.expr != conjecture) continue;
        const auto itVariable =
            this->globalDependencies.originalInductionVariableMap.find(kv.first);
        assert(itVariable != this->globalDependencies.originalInductionVariableMap.end()
            && "retireStepWinSiblings: a triad without its induction variable");
        const std::string recArg = std::string("rec") + std::get<1>(itVariable->second);
        Memory* step = simpleMapStore.findChild(goalBlock,
            "(in2[" + recArg + "," + std::get<0>(itVariable->second) + "," + sName + "])");
        Memory* zero = simpleMapStore.findChild(goalBlock,
            "(=[s(" + recArg + ")," + zeroArgName + "])");
        assert(step != nullptr && zero != nullptr
            && "retireStepWinSiblings: a triad's step or zero block is missing under the goal block");
        switchOff(step);
        // An uncovered step closure of this conjecture woke its zero block
        // without winning; the win switches that base case off as well.
        switchOff(zero);
    }
    return switchedOff;
}

/// @brief Settle one step block that proved its goal in the finished burst.
///
/// @details
/// The step-win seam's per-closure decision, run single-threaded at the
/// barrier in the total order of the closed step blocks. The step block's
/// triad comes from `Dependencies::stepBlockOriginal`. When the
/// conjecture already has a step win this closure is a `TieLoser`. When
/// the conjecture's goal block left the active set, or its head goal is no
/// longer open there, the closure is `Unaffected`. When the step proof's
/// levels (from this burst's closures), together with the typing fact's
/// levels if the goal block knows that fact, leave a premise level unused,
/// the closure is `Uncovered`: nothing is switched off and the base case
/// decides (maintainer rule: a win needs a publishable step). Otherwise it
/// is the `Winner`: a `StepWinRecord` is stored, and every other triad's
/// step and zero blocks and both contradiction blocks of the head are
/// switched off. The premise chain, the goal block included, stays alive:
/// its direct route can prove the head before the winning base case does
/// (maintainer, 2026-10-10). The goal block is made resident through the
/// steward door for the goal and typing probes and released to Idle
/// afterwards. The caller wakes the zero block for every outcome except
/// `TieLoser`.
///
/// @param stepBlock The step block whose goal closed in this burst.
/// @param steward   The steward whose door loads the goal block.
/// @param closures  This burst's closure tuples `(auxyIndex, levelsMask,
///                  coreId)`; the step's own closure is among them
///                  (asserted).
/// @return The event for the barrier's diagnostics line.
/// @invariant Runs only at the single-threaded post-join barrier (I-28),
///            before the barrier's discharge loop.
/// @see retryWaitingPromotions, ProofBarrier::settleStepWins.
ScopeLifecycle::StepWinEvent ScopeLifecycle::settleStepClosure(Memory& stepBlock, MemorySteward& steward,
    const std::vector<std::tuple<int, uint64_t, int>>& closures) {
    const auto itTriad = this->globalDependencies.stepBlockOriginal.find(&stepBlock);
    assert(itTriad != this->globalDependencies.stepBlockOriginal.end()
        && "settleStepClosure: a closed step block without its triad record");
    const int originalIndex = itTriad->second;
    const std::string conjecture =
        this->globalDependencies.originalAuxyMap.at(originalIndex).expr;
    const auto itVariable =
        this->globalDependencies.originalInductionVariableMap.find(originalIndex);
    assert(itVariable != this->globalDependencies.originalInductionVariableMap.end()
        && "settleStepClosure: a step block's triad has no induction variable");

    StepWinEvent event;
    event.conjecture = conjecture;
    event.inductionVariable = std::get<0>(itVariable->second);
    event.strongInduction =
        this->globalDependencies.strongInductionOriginals.count(originalIndex) != 0;

    // A win switches every other step block of its conjecture off at its own
    // barrier, so a second closure can only share the winner's burst.
    const auto itWin = this->globalDependencies.stepWins.find(conjecture);
    if (itWin != this->globalDependencies.stepWins.end()) {
        assert(itWin->second.originalIndex != originalIndex
            && "settleStepClosure: one step block closed twice");
        event.kind = StepWinEvent::Kind::TieLoser;
        return event;
    }

    Memory* goalBlock = stepBlock.parentMemory;
    assert(goalBlock != nullptr && "settleStepClosure: a step block without its goal block");
    if (!goalBlock->isActive) return event;

    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    const std::string head =
        ce::disintegrateImplication(conjecture, tempChain, this->definitions.configurations());

    steward.claimAndLoadForWork(*goalBlock, /*phase=*/4, lbdeload::kDeloadDirectory);
    if (lookupToBeProved(goalBlock->intToBeProved, goalBlock->nameMap, head, "main") == 0) {
        // The conjecture is proved already; the zero block wakes as before.
        goalBlock->stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle), std::memory_order_release);
        return event;
    }

    // A win needs a step that can publish on its own: its proof's levels,
    // with the typing fact's levels once the goal block knows that fact,
    // cover every premise level (D-516).
    uint64_t stepLevels = 0;
    bool stepClosureFound = false;
    for (const std::tuple<int, uint64_t, int>& closure : closures) {
        if (this->globalDependencies.auxyOriginalMap.at(std::get<0>(closure)) != originalIndex) continue;
        stepLevels |= std::get<1>(closure);
        stepClosureFound = true;
    }
    assert(stepClosureFound && "settleStepClosure: a closed step block without its closure in this burst");
    std::vector<std::string> chainKeys;
    chainKeys.reserve(tempChain.size());
    for (const auto& element : tempChain) chainKeys.push_back(std::get<0>(element));
    const uint64_t premiseLevels = inductionPremiseLevels(chainKeys);
    const uint64_t typingLevels = inductionTypingKnown(*goalBlock, event.inductionVariable)
        ? inductionTypingLevels(*goalBlock, event.inductionVariable) : 0;
    const uint64_t missingLevels = premiseLevels & ~(stepLevels | typingLevels);
    goalBlock->stewardClaim.store(
        static_cast<uint8_t>(Memory::StewardClaim::Idle), std::memory_order_release);
    if (missingLevels != 0) {
        event.kind = StepWinEvent::Kind::Uncovered;
        event.missingLevels = missingLevels;
        return event;
    }

    StepWinRecord win;
    win.originalIndex = originalIndex;
    win.goalBlock = goalBlock;
    win.head = head;
    win.inductionVariable = event.inductionVariable;
    win.strongInduction = event.strongInduction;

    event.siblingsSwitchedOff = retireStepWinSiblings(conjecture, win);
    this->globalDependencies.stepWins.emplace(conjecture, win);
    if (event.siblingsSwitchedOff > 0) {
        deactivationSweepOwed = true;
    }
    event.kind = StepWinEvent::Kind::Winner;
    return event;
}

/// @brief Switch off a weak step block whose usefulness test caught its
///        hypothesis head.
///
/// @details
/// The barrier's per-block act for a `Caught` usefulness test
/// (`InductionUsefulness::settle`): an active step block is switched off
/// with the flip contract of `retireStepWinSiblings` (`isActive` false,
/// `contradictionTheoremId` cleared, `dischargedForever` untouched) and the
/// barrier's tree-wide prune is owed (`deactivationSweepOwed`). Its zero
/// block stays parked, since the step can no longer close; the goal block
/// and the conjecture's other triads are untouched. The test becomes `Done`.
/// The triad comes from `Dependencies::stepBlockOriginal`
/// (D-521).
///
/// @param stepBlock The step block whose test caught its head (asserted).
/// @return The event for the barrier's diagnostics line.
/// @invariant Runs only at the single-threaded post-join barrier (I-28).
/// @see InductionUsefulness::settle, ProofBarrier::settleUselessSteps.
ScopeLifecycle::UselessStepEvent ScopeLifecycle::switchOffUselessStep(Memory& stepBlock) {
    assert(stepBlock.usefulnessTest == Memory::UsefulnessTest::Caught
        && "switchOffUselessStep: the block's usefulness test has not caught its head");
    const auto itTriad = this->globalDependencies.stepBlockOriginal.find(&stepBlock);
    assert(itTriad != this->globalDependencies.stepBlockOriginal.end()
        && "switchOffUselessStep: a step block without its triad record");
    const int originalIndex = itTriad->second;
    assert(this->globalDependencies.strongInductionOriginals.count(originalIndex) == 0
        && "switchOffUselessStep: only a weak step is tested");
    const auto itVariable =
        this->globalDependencies.originalInductionVariableMap.find(originalIndex);
    assert(itVariable != this->globalDependencies.originalInductionVariableMap.end()
        && "switchOffUselessStep: a step block's triad has no induction variable");

    UselessStepEvent event;
    event.conjecture = this->globalDependencies.originalAuxyMap.at(originalIndex).expr;
    event.inductionVariable = std::get<0>(itVariable->second);
    if (stepBlock.isActive) {
        stepBlock.contradictionTheoremId = 0;
        stepBlock.isActive = false;
        deactivationSweepOwed = true;
        event.switchedOff = true;
    }
    stepBlock.usefulnessTest = Memory::UsefulnessTest::Done;
    return event;
}

/// @brief Re-queue every winning triad's promotion that waits for its
///        typing fact, once the goal block knows the fact.
///
/// @details
/// Walks `Dependencies::stepWins` in conjecture order and visits the
/// records whose promotion met a goal block without the typing fact
/// (`StepWinRecord::promotionAwaitsTyping`). A record whose goal block left
/// the active set, or whose head goal is no longer open there, stops
/// waiting. Otherwise, once the goal block knows the typing fact, a
/// `PromotionRetried` event hands the waiting auxiliary index back to the
/// barrier, which re-queues it into the theorem drain (maintainer choice,
/// 2026-10-10). Each visited goal block is made resident through the
/// steward door and released to Idle afterwards.
///
/// @param steward The steward whose door loads the goal blocks.
/// @return One `PromotionRetried` event per promotion due again, in
///         conjecture order.
/// @invariant Runs only at the single-threaded post-join barrier (I-28).
/// @see settleStepClosure, TheoremPublisher::updateGlobal.
std::vector<ScopeLifecycle::StepWinEvent> ScopeLifecycle::retryWaitingPromotions(MemorySteward& steward) {
    std::vector<StepWinEvent> events;
    for (auto& kv : this->globalDependencies.stepWins) {
        StepWinRecord& win = kv.second;
        if (!win.promotionAwaitsTyping) continue;
        Memory* goalBlock = win.goalBlock;
        if (!goalBlock->isActive) {
            win.promotionAwaitsTyping = false;
            continue;
        }
        steward.claimAndLoadForWork(*goalBlock, /*phase=*/4, lbdeload::kDeloadDirectory);
        if (lookupToBeProved(goalBlock->intToBeProved, goalBlock->nameMap, win.head, "main") == 0) {
            win.promotionAwaitsTyping = false;
        } else if (inductionTypingKnown(*goalBlock, win.inductionVariable)) {
            assert(win.awaitingAuxyIndex >= 0
                && "retryWaitingPromotions: a waiting promotion without its auxiliary index");
            StepWinEvent event;
            event.kind = StepWinEvent::Kind::PromotionRetried;
            event.conjecture = kv.first;
            event.inductionVariable = win.inductionVariable;
            event.strongInduction = win.strongInduction;
            event.retryAuxyIndex = win.awaitingAuxyIndex;
            events.push_back(event);
            win.promotionAwaitsTyping = false;
        }
        goalBlock->stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle), std::memory_order_release);
    }
    return events;
}
}
