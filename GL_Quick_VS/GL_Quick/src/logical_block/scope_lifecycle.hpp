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
#pragma once
#include "../project_paths.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory.hpp"
#include "../filter.hpp"
#include "../compilation_state.hpp"
#include "../compiler/expression_parser.hpp"
#include "../compiler/compilation_normalizer.hpp"
#include "../compiler/compilation_persistence.hpp"
#include "../compiler/compiled_definition_registry.hpp"
#include "../inference/hash_key_builder.hpp"
#include "../inference/implication_multiplier.hpp"
#include "../inference/binding_partitions.hpp"
#include "../inference/hash_rule_index.hpp"
#include "../inference/hash_rule_classifier.hpp"
#include "../inference/hash_request_generator.hpp"
#include "../inference/hash_rule_evaluator.hpp"
#include "../inference/burst_sink.hpp"
#include "../logical_block/logic_block_identity.hpp"
#include "../logical_block/statement_knowledge.hpp"
#include "../logical_block/deferred_ancestor_admission.hpp"
#include "../logical_block/update_global_single_chapter_record.hpp"
#include "../logical_block/or_implication_refuter.hpp"
#include "../logical_block/or_implication_selection.hpp"
#include "../logical_block/watch_bridge.hpp"
#include "../logical_block/canonical_form_cache.hpp"
#include "../logical_block/compact_canon_pair.hpp"
#include "../logical_block/numeric_less.hpp"
#include "../logical_block/hypo_scope_parse.hpp"
#include "../logical_block/implication_subproof_outcome.hpp"
#include "../logical_block/collected_record.hpp"
#include "../logical_block/collected_arena.hpp"
#include "../logical_block/new_variable_store.hpp"
#include "../logical_block/witness_meta_store.hpp"
#include "../logical_block/rejection_record.hpp"
#include "../logical_block/rejection_store.hpp"
#include "../logical_block/scanned_token_runs.hpp"
#include "../logical_block/equivalence_span_pair.hpp"
#include "../logical_block/equivalence_rewrite.hpp"
#include "../logical_block/admission_policy.hpp"
#include "../logical_block/scope_type_resolver.hpp"
#include "../logical_block/canonicalizer.hpp"
#include "../logical_block/canonical_template_matcher.hpp"
#include "../logical_block/rejected_source_memory.hpp"
#include "../logical_block/integration_instruction_builder.hpp"
#include "../logical_block/unary_rule_memory.hpp"
#include "../logical_block/admission_marker_builder.hpp"
#include "../compiler/existence_rule_compiler.hpp"
#include "../compiler/implication_builder.hpp"
#include "../compiler/or_rule_compiler.hpp"
#include "../compiler/structural_compiler.hpp"
#include "../compiler/supporting_rule_compiler.hpp"
#include "../compiler/compilation_reuse.hpp"
#include "../compiler/definition_expansion.hpp"
#include "../proof/worker_context.hpp"
#include "../infra/rt_tracker.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory_infra/steward.hpp"
#include "../memory_infra/lb_store.hpp"
#include "../mail_log.hpp"
#include "../simple_map_store.hpp"
#include "../memory_infra/or_implication_match_index.hpp"
#include "../memory_infra/temporary_witness_store.hpp"
#include "phase13_timing_slot.hpp"
#include "watch_ledger.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>
namespace gl {
class AnchorScopeQueries;
namespace tests { struct ScopeLifecycleFixture; }
/// @brief Scope Lifecycle operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class ScopeLifecycle {
friend struct tests::ScopeLifecycleFixture;
public:
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
ScopeLifecycle(Memory& body, bool& deactivationSweepOwed, const CompiledDefinitionRegistry& definitions, Dependencies& globalDependencies, std::vector<Memory*>& pendingSubtreeDeactivations, std::mutex& pendingSubtreeDeactivationsMutex, std::vector<Memory*>& pendingTwinDeactivations, std::mutex& pendingTwinDeactivationsMutex, SimpleMapStore& simpleMapStore, AnchorScopeQueries& anchorScopeQueries);

private:
/// @brief Stage one contradiction LB for post-join deactivation.
        ///
        /// @details
        /// The staging primitive of the twin-deactivation machinery: the
        /// discharge sites run in the parallel worker phase and must not
        /// write another LB's `isActive` (I-28), so they stage the target
        /// here and `drainPendingTwinDeactivations` flips it post-join.
        /// A null @p twin is a CONTRACTED absence, not an error: the
        /// reformulated-operator LB has no polarity twin, and batches
        /// without the complement flag create only one polarity — both are
        /// defined "nothing to stage" cases.
        ///
        /// @param twin The contradiction LB to retire, or `nullptr` for the
        ///             contracted no-twin case.
void stageTwinDeactivation(Memory* twin);

public:
/// @brief Stage both contradiction twins of a closed conjecture head.
        ///
        /// @details
        /// Called from the ordinary goal closure (`dischargeToBeProved`) on
        /// the innermost chain LB: the conjecture is settled, so the LBs
        /// that would prove (`__contradiction__` + gl::DefinitionExpansion::negate(head)) or disprove
        /// (`__contradiction__` + head) it have no remaining purpose. Both
        /// keys are built on @p arena (no heap — the caller is statified)
        /// and probed via the span `findChild`; absent twins are the
        /// contracted no-twin case.
        ///
        /// @param chainLB The innermost chain LB that owns the twins.
        /// @param head    The closed goal head (span over a stable buffer).
        /// @param arena   Scratch arena for the key bytes.
void stageContradictionTwinsOfHead(Memory& chainLB, StrSpan head,
                                           ScratchArena& arena);

public:
/// @brief Stage the polarity sibling of a contradiction LB.
        ///
        /// @details
        /// Called when a contradiction LB itself settles its conjecture (its
        /// own fire, or a goal closure routed through it): the sibling that
        /// assumed the opposite polarity — `__contradiction__` +
        /// gl::DefinitionExpansion::negate(own seed) under the same parent — has no remaining
        /// purpose. The seed is the caller's exprKey with the
        /// `__contradiction__` prefix stripped
        /// (I-165); the sibling key
        /// differs from the caller's own key by exactly one leading `!`, so
        /// the lookup can never return the caller.
        ///
        /// @param contraLB The fired / settled contradiction LB.
        /// @param arena    Scratch arena for the key bytes.
void stageContradictionSiblingTwin(Memory& contraLB,
                                           ScratchArena& arena);

public:
/// @brief Post-join drain: deactivate every staged contradiction twin.
        ///
        /// @details
        /// Runs single-threaded in `proveKernel`'s collector, directly after
        /// `drainUpdateGlobalSingleChapter`, so every firing/closure of the
        /// iteration has staged its twins first. An already-inactive twin
        /// (it fired on its own before its counterpart's stage drained) is a
        /// defined no-op — deactivation is idempotent and commutative, so no
        /// ordering work is needed. Clears the staging vector.
void drainPendingTwinDeactivations();

public:
/// @brief Stage one vacuous-premise LB for post-join subtree
        ///        deactivation.
        ///
        /// @details
        /// The staging primitive of the vacuous-premise suppression:
        /// `dischargeContradiction`'s normal-LB branch runs in the parallel
        /// worker phase and must not write descendant LBs' `isActive`
        /// (I-28), so it stages the contradicted root here and
        /// `drainSubtreeDeactivations` walks the subtree post-join.
        ///
        /// @param root The contradicted premise LB whose descendants are to
        ///             be deactivated.
void stageSubtreeDeactivation(Memory* root);

private:
/// @brief Deactivate every descendant LB of @p node (the node itself
        ///        is not touched).
        ///
        /// @details
        /// Recursive walk over the LB tree's routing edges
        /// (`simpleMapStore.forEachChild` — deterministic decoded-key
        /// order). Each still-active descendant is deactivated and its
        /// pre-built contradiction theorem cleared, mirroring
        /// `drainPendingTwinDeactivations`' flip contract:
        /// `dischargedForever` stays untouched, so no barrier assert can
        /// fire for a descendant that never re-enters an active snapshot.
        /// Already-inactive descendants are recursed through regardless —
        /// their own children may still be active.
        ///
        /// @param node The subtree root whose descendants are deactivated.
void deactivateDescendantsOf(Memory* node);

public:
/// @brief Post-join drain: deactivate the subtree of every staged
        ///        vacuous-premise LB.
        ///
        /// @details
        /// Runs single-threaded in `proveKernel`'s collector, directly
        /// after `drainPendingTwinDeactivations`. Cross-LB `isActive`
        /// writes are legal here (the sanctioned I-28 seam). Deactivation
        /// is idempotent and commutative, so staging order and
        /// double-staging need no ordering work. Clears the staging
        /// vector.
void drainSubtreeDeactivations();

public:
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
void deactivateRecursively();

public:
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
void deactivateUnnecessary(const std::string& theorem, Memory& root);

public:
/// @brief One barrier event of the step win, reported by the barrier as a
///        `[STEP-WIN]` diagnostics line.
///
/// @details
/// `Unaffected`: the closed step block's conjecture was already proved, or
/// its goal block had left the active set, so the zero block wakes as
/// before and nothing is switched off. `Uncovered`: the step proof, with
/// the typing fact's levels when the goal block knows that fact, leaves a
/// premise level unused (`missingLevels`); the zero block wakes, nothing is
/// switched off and no win is recorded, so the base case decides.
/// `Winner`: the first step closure of an open conjecture that covers every
/// premise level; its siblings are switched off and the premise chain stays
/// alive. `TieLoser`: another step of the same conjecture won in this
/// barrier; this step's zero block stays parked. `PromotionRetried`: the
/// winning triad's promotion, stopped earlier only by the missing typing
/// fact, is due again because the fact arrived; `retryAuxyIndex` is the
/// auxiliary index the barrier re-queues (D-517).
struct StepWinEvent {
    enum class Kind { Unaffected, Uncovered, Winner, TieLoser, PromotionRetried };
    Kind kind = Kind::Unaffected;
    std::string conjecture;
    std::string inductionVariable;
    bool strongInduction = false;
    int siblingsSwitchedOff = 0;
    int retryAuxyIndex = -1;
    uint64_t missingLevels = 0;
};

public:
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
static uint64_t inductionPremiseLevels(const std::vector<std::string>& chainKeys);

public:
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
uint64_t inductionTypingLevels(Memory& goalBlock, const std::string& inductionVariable);

public:
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
bool inductionTypingKnown(Memory& goalBlock, const std::string& inductionVariable);

public:
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
StepWinEvent settleStepClosure(Memory& stepBlock, MemorySteward& steward,
                               const std::vector<std::tuple<int, uint64_t, int>>& closures);

public:
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
std::vector<StepWinEvent> retryWaitingPromotions(MemorySteward& steward);

public:
/// @brief One barrier event of a useless weak induction step, reported by the
///        barrier as a `[STEP-USELESS]` diagnostics line.
///
/// @details `conjecture` and `inductionVariable` name the step block's triad.
/// `switchedOff` is `false` when the block had already left the active set
/// in the same burst (D-521).
struct UselessStepEvent {
    std::string conjecture;
    std::string inductionVariable;
    bool switchedOff = false;
};

public:
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
UselessStepEvent switchOffUselessStep(Memory& stepBlock);

private:
/// @brief Count the block's goals at the `main` scope.
///
/// @details
/// The I-48 survey: `intToBeProved` keys whose validity id is
/// `NameMap::MAIN_ID`. The goal registry lives in the persistent pool, so the
/// survey needs no residency door (I-108).
///
/// @param block The block to survey.
/// @return The number of main-scope goals.
int mainGoalCount(const Memory& block) const;

private:
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
int retireStepWinSiblings(const std::string& conjecture, const StepWinRecord& win);

private:

Memory& body;
bool& deactivationSweepOwed;
const CompiledDefinitionRegistry& definitions;
Dependencies& globalDependencies;
std::vector<Memory*>& pendingSubtreeDeactivations;
std::mutex& pendingSubtreeDeactivationsMutex;
std::vector<Memory*>& pendingTwinDeactivations;
std::mutex& pendingTwinDeactivationsMutex;
SimpleMapStore& simpleMapStore;
AnchorScopeQueries& anchorScopeQueries;
};
}
#include "../logical_block/anchor_scope_queries.hpp"
namespace gl {

/// @brief Stage one contradiction LB for post-join deactivation.
        ///
        /// @details
        /// The staging primitive of the twin-deactivation machinery: the
        /// discharge sites run in the parallel worker phase and must not
        /// write another LB's `isActive` (I-28), so they stage the target
        /// here and `drainPendingTwinDeactivations` flips it post-join.
        /// A null @p twin is a CONTRACTED absence, not an error: the
        /// reformulated-operator LB has no polarity twin, and batches
        /// without the complement flag create only one polarity — both are
        /// defined "nothing to stage" cases.
        ///
        /// @param twin The contradiction LB to retire, or `nullptr` for the
        ///             contracted no-twin case.
inline void ScopeLifecycle::stageTwinDeactivation(Memory* twin) {
            if (twin == nullptr) return;
            std::lock_guard<std::mutex> lock(this->pendingTwinDeactivationsMutex);
            this->pendingTwinDeactivations.push_back(twin);
        }

/// @brief Stage both contradiction twins of a closed conjecture head.
        ///
        /// @details
        /// Called from the ordinary goal closure (`dischargeToBeProved`) on
        /// the innermost chain LB: the conjecture is settled, so the LBs
        /// that would prove (`__contradiction__` + gl::DefinitionExpansion::negate(head)) or disprove
        /// (`__contradiction__` + head) it have no remaining purpose. Both
        /// keys are built on @p arena (no heap — the caller is statified)
        /// and probed via the span `findChild`; absent twins are the
        /// contracted no-twin case.
        ///
        /// @param chainLB The innermost chain LB that owns the twins.
        /// @param head    The closed goal head (span over a stable buffer).
        /// @param arena   Scratch arena for the key bytes.
inline void ScopeLifecycle::stageContradictionTwinsOfHead(Memory& chainLB, StrSpan head,
                                           ScratchArena& arena) {
            const StrSpan prefix("__contradiction__", 17);
            const StrSpan negHead = gl::DefinitionExpansion::negateScratch(arena, head);
            const StrSpan suffixes[2] = { head, negHead };
            for (int k = 0; k < 2; ++k) {
                char* kb = arena.allocBytes(prefix.len + suffixes[k].len);
                std::memcpy(kb, prefix.ptr, prefix.len);
                std::memcpy(kb + prefix.len, suffixes[k].ptr,
                            static_cast<size_t>(suffixes[k].len));
                stageTwinDeactivation(simpleMapStore.findChild(
                    &chainLB, StrSpan(kb, prefix.len + suffixes[k].len)));
            }
        }

/// @brief Stage the polarity sibling of a contradiction LB.
        ///
        /// @details
        /// Called when a contradiction LB itself settles its conjecture (its
        /// own fire, or a goal closure routed through it): the sibling that
        /// assumed the opposite polarity — `__contradiction__` +
        /// gl::DefinitionExpansion::negate(own seed) under the same parent — has no remaining
        /// purpose. The seed is the caller's exprKey with the
        /// `__contradiction__` prefix stripped
        /// (I-165); the sibling key
        /// differs from the caller's own key by exactly one leading `!`, so
        /// the lookup can never return the caller.
        ///
        /// @param contraLB The fired / settled contradiction LB.
        /// @param arena    Scratch arena for the key bytes.
inline void ScopeLifecycle::stageContradictionSiblingTwin(Memory& contraLB,
                                           ScratchArena& arena) {
            const StrSpan prefix("__contradiction__", 17);
            const StrSpan mbKey = contraLB.exprKeyView();
            assert(mbKey.len > prefix.len
                && equalSpans(StrSpan(mbKey.ptr, prefix.len), prefix)
                && "sibling-twin staging requires a __contradiction__ LB");
            assert(contraLB.parentMemory != nullptr);
            const StrSpan seed(mbKey.ptr + prefix.len, mbKey.len - prefix.len);
            const StrSpan negSeed = gl::DefinitionExpansion::negateScratch(arena, seed);
            char* kb = arena.allocBytes(prefix.len + negSeed.len);
            std::memcpy(kb, prefix.ptr, prefix.len);
            std::memcpy(kb + prefix.len, negSeed.ptr,
                        static_cast<size_t>(negSeed.len));
            stageTwinDeactivation(simpleMapStore.findChild(
                contraLB.parentMemory, StrSpan(kb, prefix.len + negSeed.len)));
        }

/// @brief Post-join drain: deactivate every staged contradiction twin.
        ///
        /// @details
        /// Runs single-threaded in `proveKernel`'s collector, directly after
        /// `drainUpdateGlobalSingleChapter`, so every firing/closure of the
        /// iteration has staged its twins first. An already-inactive twin
        /// (it fired on its own before its counterpart's stage drained) is a
        /// defined no-op — deactivation is idempotent and commutative, so no
        /// ordering work is needed. Clears the staging vector.
inline void ScopeLifecycle::drainPendingTwinDeactivations() {
            std::lock_guard<std::mutex> lock(this->pendingTwinDeactivationsMutex);
            for (Memory* twin : this->pendingTwinDeactivations) {
                if (!twin->isActive) continue;
                twin->contradictionTheoremId = 0;
                twin->isActive = false;
            }
            this->pendingTwinDeactivations.clear();
        }

/// @brief Stage one vacuous-premise LB for post-join subtree
        ///        deactivation.
        ///
        /// @details
        /// The staging primitive of the vacuous-premise suppression:
        /// `dischargeContradiction`'s normal-LB branch runs in the parallel
        /// worker phase and must not write descendant LBs' `isActive`
        /// (I-28), so it stages the contradicted root here and
        /// `drainSubtreeDeactivations` walks the subtree post-join.
        ///
        /// @param root The contradicted premise LB whose descendants are to
        ///             be deactivated.
inline void ScopeLifecycle::stageSubtreeDeactivation(Memory* root) {
            std::lock_guard<std::mutex> lock(this->pendingSubtreeDeactivationsMutex);
            this->pendingSubtreeDeactivations.push_back(root);
        }

/// @brief Deactivate every descendant LB of @p node (the node itself
        ///        is not touched).
        ///
        /// @details
        /// Recursive walk over the LB tree's routing edges
        /// (`simpleMapStore.forEachChild` — deterministic decoded-key
        /// order). Each still-active descendant is deactivated and its
        /// pre-built contradiction theorem cleared, mirroring
        /// `drainPendingTwinDeactivations`' flip contract:
        /// `dischargedForever` stays untouched, so no barrier assert can
        /// fire for a descendant that never re-enters an active snapshot.
        /// Already-inactive descendants are recursed through regardless —
        /// their own children may still be active.
        ///
        /// @param node The subtree root whose descendants are deactivated.
inline void ScopeLifecycle::deactivateDescendantsOf(Memory* node) {
            simpleMapStore.forEachChild(node,
                [&](const StrSpan& /*key*/, Memory* child) {
                    if (child->isActive) {
                        child->contradictionTheoremId = 0;
                        child->isActive = false;
                    }
                    deactivateDescendantsOf(child);
                });
        }

/// @brief Post-join drain: deactivate the subtree of every staged
        ///        vacuous-premise LB.
        ///
        /// @details
        /// Runs single-threaded in `proveKernel`'s collector, directly
        /// after `drainPendingTwinDeactivations`. Cross-LB `isActive`
        /// writes are legal here (the sanctioned I-28 seam). Deactivation
        /// is idempotent and commutative, so staging order and
        /// double-staging need no ordering work. Clears the staging
        /// vector.
inline void ScopeLifecycle::drainSubtreeDeactivations() {
            std::lock_guard<std::mutex> lock(this->pendingSubtreeDeactivationsMutex);
            for (Memory* root : this->pendingSubtreeDeactivations) {
                deactivateDescendantsOf(root);
            }
            this->pendingSubtreeDeactivations.clear();
        }
}
