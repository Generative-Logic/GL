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
#include "../logical_block/admission_memory.hpp"
#include "../logical_block/integration_processor.hpp"
#include "../logical_block/implication_subproofs.hpp"
#include "../logical_block/rejected_source_replay.hpp"
#include "../inference/hash_rule_installer.hpp"
#include "../logical_block/compact_rule_processor.hpp"
#include "../logical_block/statement_memory.hpp"
#include "../logical_block/equivalence_processor.hpp"
#include "../logical_block/equality_propagation.hpp"
#include "../logical_block/equivalence_admission_rewriter.hpp"
#include "../logical_block/watch_processor.hpp"
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/case_processor.hpp"
#include "../logical_block/disintegration_processor.hpp"
#include "../logical_block/or_implication_processor.hpp"
#include "../logical_block/proof_history.hpp"
#include "../logical_block/unary_integration_processor.hpp"
#include "../logical_block/goal_processor.hpp"
#include "../logical_block/scope_lifecycle.hpp"
#include "../logical_block/logic_block_operations.hpp"
#include "../logical_block/block_mail_processor.hpp"
#include "../logical_block/anchor_scope_queries.hpp"
#include "../logical_block/equivalence_queries.hpp"
#include "../logical_block/block_key_reader.hpp"
#include "../logical_block/phase13_timing_slot.hpp"
#include "../logical_block/watch_ledger.hpp"
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
#include "../proof/or_candidate.hpp"
#include "../infra/rt_tracker.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory_infra/steward.hpp"
#include "../memory_infra/lb_store.hpp"
#include "../mail_log.hpp"
#include "../simple_map_store.hpp"
#include "../memory_infra/or_implication_match_index.hpp"
#include "../memory_infra/temporary_witness_store.hpp"
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
class ScopeLifecycle;
class TheoremPublisher;
namespace tests { struct OrTheoremBuilderFixture; }
/// @brief Or Theorem Builder operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class OrTheoremBuilder {
friend struct tests::OrTheoremBuilderFixture;
public:
/// @brief Bind the or theorem builder operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param fullTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param globalTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param orBuiltByPair Borrowed std::vector interface or state; must outlive this component.
/// @param orCandidates Borrowed std::vector interface or state; must outlive this component.
/// @param orCandidatesMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param orInRunScannedRows Borrowed std::vector interface or state; must outlive this component.
/// @param orPairsFromHeadSwitch Borrowed std::vector interface or state; must outlive this component.
/// @param orRuleCompiler Borrowed OrRuleCompiler interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param theoremListMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param theoremPublisher Borrowed TheoremPublisher interface or state; must outlive this component.
/// @param orInRunCompanions Existing shared state or operation interface, borrowed for this component lifetime.
/// @param scopeLifecycle Existing shared state or operation interface, borrowed for this component lifetime.
/// @param deactivationSweepOwed Existing shared state or operation interface, borrowed for this component lifetime.
/// @return None; construction binds the supplied references.
OrTheoremBuilder(const CompiledDefinitionRegistry& definitions, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& fullTheoremList, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList, std::map<std::pair<std::string, std::string>, std::string>& orBuiltByPair, std::vector<OrCandidate>& orCandidates, std::mutex& orCandidatesMutex, std::unordered_set<std::string>& orInRunScannedRows, std::vector<std::pair<std::string, std::string>>& orPairsFromHeadSwitch, OrRuleCompiler& orRuleCompiler, const ProverParameters& parameters, ProofHistory& proofHistory, std::mutex& theoremListMutex, TheoremPublisher& theoremPublisher, std::unordered_set<std::string>& orInRunCompanions, ScopeLifecycle& scopeLifecycle, bool& deactivationSweepOwed);

public:
// Apply the head-switch (contrapositive) construction to a single
        // theorem string. Returns the rebuilt implication if the chain has a
        // negated, non-quantified premise; returns an empty string otherwise.
        // The negated head takes that premise's place, or, when a later
        // premise binds a variable of the head, the place right after the
        // last such premise, so it is never used before its binder.
        // An already-negated head loses its `!` when it becomes the new
        // premise (double-negation cancellation) — GL has no universal `!!`
        // normalizer and a raw `!!(...)` breaks compiled-core consumers.
        // Stateless — does not register the result anywhere. Used by both the
        // pre-emit pass at conjecture-load time and the post-prove walk that
        // populates `orPairsFromHeadSwitch` for OR-theorem construction.
std::string headSwitchOne(const std::string& theorem) const;

private:
/// @brief Fold a proved hypothesis-chain theorem into a compiled
        /// n-ary OR theorem and register its `or<N>` compact.
        ///
        /// @details
        /// Disintegrates @p existenceThm and folds EVERY binder-free
        /// negated premise into the OR head: classically
        /// `!d_1 → (!d_2 → h)` is equivalent to `d_1 ∨ d_2 ∨ h`, so a
        /// chain with K binder-free negated premises yields K+1 disjuncts
        /// — each folded premise un-negated, the head last, all in chain
        /// order and TRUE polarity (a negated head stays a negated
        /// disjunct). A negated premise that binds variables stays a
        /// shared premise: folding it would orphan its binder. Non-negated
        /// premises survive as the reassembled theorem's premise chain.
        ///
        /// The u_-canonical ordered element list is the or-operator's
        /// registry identity: an existing `or`-category entry with the
        /// identical element vector is reused (I-23); otherwise a fresh
        /// `or<orCounter>` is minted into `compiledExpressions` +
        /// `coreExpressionMap`.
        ///
        /// @param existenceThm  The proved direction; supplies premises,
        ///        folded disjuncts, and the head disjunct.
        /// @param companionThm  The head-switched companion; recorded by
        ///        the caller as parent reference only — not consulted for
        ///        construction.
        /// @return The reassembled OR theorem with compiled `(or<N>[…])`
        ///         head, or the empty string when the chain carries no
        ///         binder-free negated premise (caller asserts against
        ///         this for licensed pairs).
        /// @invariant Single-threaded seam — runs post-`prove()` only.
        /// @see `constructOrTheoremsFromPairs` — the walking caller.
        /// @see `headSwitchOne` — pair producer.
std::string constructOrTheorem(const std::string& existenceThm, const std::string& companionThm);

public:
/// @brief Construct every OR theorem licensed by the head-switch pairs
/// against the CURRENT `globalTheoremList`, register each in the global
/// registries, and report the parents it subsumes.
///
/// @details
/// Walks `orPairsFromHeadSwitch` — one `(theorem, companion)` pair per
/// proved theorem whose straightened shape carries a binder-free negated
/// premise (`headSwitchOne`). A SINGLE proved direction licenses the
/// disjunction: classically `!d_1 → (!d_2 → h)` is equivalent to
/// `d_1 ∨ d_2 ∨ h`, and every other direction is derivable from it, so
/// the companion's own proof is never required — it is recorded only as
/// the or-theorem row's second parent reference.
///
/// A pair whose source theorem is no longer in `globalTheoremList` is
/// skipped — compression pruning and vacuity retraction remove rows
/// between the pair walk and this call, and both removals are defined
/// pipeline states, not failures. Mirror pairs `(x, y)` and `(y, x)`
/// describe the same disjunction; the canonical (min, max) disjunct-set
/// dedup constructs it once.
///
/// @param consumedParents Out-parameter accumulating the proved parent
///        theorems subsumed by a constructed OR; the caller drops them
///        from the saved theorem files.
/// @return The constructed OR theorems in compiled form, in pair order.
/// @invariant Single-threaded seam — must run after `prove()` has joined.
/// @see `constructOrTheorem` — per-pair OR builder.
/// @see `headSwitchOne` — pair producer.
std::vector<std::string> constructOrTheoremsFromPairs(
            std::set<std::string>& consumedParents);

public:
/// @brief In-run OR-theorem construction and broadcast — the A16
/// Phase 2 seam.
///
/// @details
/// Runs on the phase-4 barrier of every `proveKernel` iteration
/// (single-threaded, post-`pool.join()`, after the vacuous-theorem
/// retraction sweep). Walks `globalTheoremList` for rows not yet in
/// `orInRunScannedRows`; per new row, `headSwitchOne` licenses the
/// or (a straightened shape with a binder-free negated premise),
/// `constructOrTheorem` folds and registers the `or<N>` compact
/// (I-23 registry dedup), and the or theorem is registered via
/// `appendGlobalTheorem(or, "or theorem", row, companion)` +
/// `fullTheoremList`.
///
/// The broadcast rides the existing D-76 deferred-compaction drain:
/// `recordPendingCompaction` queues the or theorem, the drain this
/// seam feeds compiles it to its implication compact and merges it
/// into the root's mailOut with an empty level set, and every LB
/// installs it through the status-3 rule door two iterations later.
/// Where the rule's premises hold it fires, the `(or<N>[…])` head
/// deposits locally, disintegrates, and the cohort machinery opens
/// under its normal demand gates. A premise-free or theorem has no
/// implication-compact shape: it registers but is not broadcast.
///
/// Every broadcast or theorem also gets the `("theorem", [])` origin at
/// `main` in the root's mailOut, the documentation every other proved
/// theorem's broadcast carries (`updateGlobalSingleChapter`). Receivers
/// thereby hold a proved-theorem leaf for it besides the compaction's
/// compilation/expansion pair, which alone is a two-node cycle the
/// chapter walker (`buildStack`) cannot ground.
///
/// Idempotence is the string dedup in `appendGlobalTheorem`: a
/// mirror row scanned later — or the post-run export seam
/// (`constructOrTheoremsFromPairs`) — reconstructs the SAME or
/// string through the registry reuse, lands `false`, and neither
/// re-registers nor re-broadcasts. In compressor mode the seam is
/// a defined no-op: the compressor's internal re-derivation is a
/// redundancy probe whose theorem list must stay byte-identical to
/// the run it audits; the export seam constructs the ors on the
/// survivor list.
///
/// Every or theorem the seam registers records its companion in
/// `orInRunCompanions` and retires the companion's logic blocks
/// (`retireSubsumedCompanion`). The seam runs after the theorem
/// drains' deactivation sweep, so it settles the prune those walks
/// owe itself and leaves `deactivationSweepOwed` false (D-451).
///
/// @return Nothing.
/// @invariant Single-threaded seam — phase-4 barrier only; registry
///            mints (`compiledExpressions` / `coreExpressionMap` /
///            `orCounter`) are legal here for the same reason the
///            deferred-compaction drain's are (D-76).
/// @see `constructOrTheorem` — per-pair builder (registry dedup).
/// @see `constructOrTheoremsFromPairs` — the post-run export seam.
/// @see `recordPendingCompaction` — the broadcast queue.
/// @see `mergeTheoremOriginMailIntoRoot` — the documentation door.
/// @see `retireSubsumedCompanion` — the companion's retirement.
void constructOrTheoremsInRun();

public:


private:
std::unordered_set<std::string>& orInRunCompanions;
ScopeLifecycle& scopeLifecycle;
bool& deactivationSweepOwed;

const CompiledDefinitionRegistry& definitions;
std::vector< std::tuple<std::string, std::string, std::string, std::string >>& fullTheoremList;
std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList;
std::map<std::pair<std::string, std::string>, std::string>& orBuiltByPair;
std::vector<OrCandidate>& orCandidates;
std::mutex& orCandidatesMutex;
std::unordered_set<std::string>& orInRunScannedRows;
std::vector<std::pair<std::string, std::string>>& orPairsFromHeadSwitch;
OrRuleCompiler& orRuleCompiler;
const ProverParameters& parameters;
ProofHistory& proofHistory;
std::mutex& theoremListMutex;
TheoremPublisher& theoremPublisher;

public:
/// @brief Register an input OR candidate pair.
/// @details Appends at the existing single-threaded input-loading boundary.
/// @param candidate Pair to append without changing its current flags.
/// @return No value.
void addCandidate(const OrCandidate& candidate) { orCandidates.push_back(candidate); }
/// @brief Count registered input OR candidate pairs.
/// @details Reads the existing candidate collection without mutation.
/// @return Number of candidate pairs.
size_t candidateCount() const { return orCandidates.size(); }
};
}
#include "../proof/theorem_publisher.hpp"
namespace gl {
}

#include "../logical_block/scope_lifecycle.hpp"
