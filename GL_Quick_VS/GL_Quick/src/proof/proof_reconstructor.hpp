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
#include "proof_walk_node.hpp"
#include "proof_no_good_entry.hpp"
#include "theorem_usage_graph.hpp"
#include "../proof/theorem_publisher.hpp"
#include "../proof/proof_grid_builder.hpp"
#include "../proof/or_theorem_builder.hpp"
#include "../proof/or_elimination_builder.hpp"
#include "../proof/counterexample_filter.hpp"
#include "../inference/encoded_expression_normalizer.hpp"
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
#include "../proof/variable_copy_detector.hpp"
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
namespace tests { struct ProofReconstructorFixture; }
/// @brief Proof Reconstructor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class ProofReconstructor {
friend struct tests::ProofReconstructorFixture;
public:
/// @brief Bind the proof reconstructor operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param exportTheoremUsage Borrowed TheoremUsageGraph interface or state; must outlive this component.
/// @param exportCurrentChapterTheorem Borrowed std::string interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
ProofReconstructor(SimpleMapStore& simpleMapStore, TheoremUsageGraph& exportTheoremUsage, std::string& exportCurrentChapterTheorem);

public:
/// @brief Reset proof reconstruction state for a new export.
/// @details Clears the existing thread-local path, journal and failed-walk memo at the export boundary.
/// @return No value.
static void clearBuildStackPath();

private:
/// @brief Order candidate origins for the reconstruction walk.
/// @details Preserves insertion order within non-equality and equality groups.
/// @param origins Candidate origin records.
/// @return Ordered candidate indices.
static std::vector<size_t> sortedOriginIndicesD49(
    const std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>& origins);

private:
/// @brief Find the shallowest origin scope allowed by the lift barriers.
/// @details Uses the existing origin lookup. Two barriers bound the lift: an
/// OR-branch scope (`orint_` / `ordis_`), which keeps every statement inside
/// its conditional branch, and a contradiction scope whose assumption is the
/// expression itself, which keeps that assumption cited where it is assumed.
/// The deepest barrier on the path wins.
/// @param mb Logical block containing the origin records.
/// @param v Expression and initial validity.
/// @return The existing expression with its selected validity.
static ExpressionWithValidity liftToShallowestOriginAncestor(
        const Memory& mb,
        const ExpressionWithValidity& v);

private:
/// @brief List every scope at which an expression carries origin records,
///        from the shallowest permitted ancestor down to its own scope.
///
/// @details
/// The first entry is the scope `liftToShallowestOriginAncestor` selects:
/// the shallowest of `main` and the `_boundary_` prefixes of
/// `v.validityName` whose `(expression, scope)` key holds an origin record,
/// at or below the lift barriers; `v` itself for an exact `zoom in` record or
/// an expression without records. The deeper prefixes, ending with the full
/// validity, follow in order whenever their own key holds a record. A first
/// entry equal to `v` leaves nothing deeper.
///
/// Only the deeper-scope walk of `buildChapterStack` reads the deeper
/// entries: a statement derived at main through the very implication a
/// subproof is proving can still be cited inside that subproof through its
/// own records.
///
/// @param mb Logical block holding the origin records.
/// @param v  Expression and the scope its citing record names.
/// @return The origin-bearing scopes, shallowest first; never empty.
/// @see liftToShallowestOriginAncestor, buildChapterStack.
static std::vector<ExpressionWithValidity> originScopeAlternatives(
        const Memory& mb,
        const ExpressionWithValidity& v);


// =====================================================================
// D-51 (option 1 + backtracking) — path-stack cycle filter for buildStack
// origin choice.
//
// With max_origin_per_expr raised to 30 (matching compressor mode), each
// expression typically carries multiple candidate origins. buildStack:
//   1. Sorts origins per D-49-style preference: non-equality tags first
//      (preserving insertion order within group), then equality1/equality2
//      tags. Within each group, insertion order survives — matches the
//      legacy cap=1 + D-49 surviving-origin choice for the common case.
//   2. Walks candidates in that order; skips any whose deps include an
//      expression already on the current proof-tree path (would form a
//      cycle with the recursion stack — Knuth (1977) AND-OR graph
//      acyclic-derivation extraction).
//   3. Backtracks: if the recursive walk fails for a candidate's deps,
//      rolls back stack/covered/path snapshots and tries the next
//      candidate. Guarantees a valid acyclic tree if one exists.
//
// Returns true on success. False signals to the recursive caller that no
// acyclic origin exists for this node given the current path; caller
// retries with its own next candidate. Top-level callers (directStack,
// checkZeroStack, ...) ignore the return.
// =====================================================================


// Insertion journal for the walk's `covered` set — the exact-rollback
// replacement for the per-candidate set copy.
//
// A candidate that fails must leave `covered` exactly as it found it, because
// its emitted rows are dropped by `stack.resize` and a node left marked
// covered with no row would lose its row from the chapter. `covered` is
// insert-only inside the walk (no site erases from it), so the set at a
// candidate's entry is recovered by erasing exactly the keys inserted since
// that candidate's mark — a suffix truncation, which is always well defined
// because marks nest with the recursion.
//
// The copy this replaces was 38% of the whole chapter export: 4.5 M
// per-candidate deep copies of a set holding up to 311 string pairs, of which
// 97% were discarded unused because the candidate succeeded.





// Conflict set, consulted set and context-dependence of the frame that most
// recently returned false - read by the parent immediately after the child
// returns.




// A frame whose consulted set outgrows this is not cached: the probe walks
// the set on every re-entry, so an unbounded footprint would cost more than
// the re-walk it saves.
















private:
static thread_local std::set<ExpressionWithValidity> g_buildStackPath;
private:
static thread_local std::vector<ExpressionWithValidity> g_coveredJournal;
private:
static thread_local std::map<WalkNode, NoGoodEntry> g_noGood;
private:
static thread_local std::set<ExpressionWithValidity> g_failConflict;
private:
static thread_local std::set<ExpressionWithValidity> g_failConsulted;
private:
static thread_local bool g_failContextDependent;
private:
static constexpr std::size_t kNoGoodConsultedCap = 2048;
private:
// Set only during the retry walk of `buildChapterStack`: the candidate loop
// then adds the deeper-scope pass.
static thread_local bool g_deeperScopes;
public:
/// @brief Reset the journal and failed-walk memo for a root traversal.
/// @details The export driver invokes this immediately before each top-level buildStack call.
/// @return No value.
static void beginRootTraversal();
public:
/// @brief Mark a chapter goal on the reconstruction path.
/// @details The path filter must reject origins that depend on the chapter goal.
/// @param goal Chapter expression and validity to mark or remove.
/// @return No value.
static void enterChapterGoal(const ExpressionWithValidity& goal);
public:
/// @brief Remove the chapter goal after its reconstruction.
/// @details The export driver removes the same marker after the top-level walk.
/// @param goal Chapter expression and validity to mark or remove.
/// @return No value.
static void leaveChapterGoal(const ExpressionWithValidity& goal);

private:
/// @brief Probe whether a `__contradiction__` twin LB can actually resolve
///        `proved` — i.e. holds at least one origin record for it — before
///        buildStack's D-51 fallback switches in.
///
/// @details
/// Mirrors the entry-side origin check of the walk the switch would start
/// exactly: read-only reload (D-158), the same scope choice (lifted within
/// the twin, or exact when that walk keeps its scope), non-minting key
/// probe, non-empty run. A twin that primed but never converged (no
/// discharge record) fails the probe, and so does a twin that holds the
/// statement only at a shallower scope than an exact walk asks for; the
/// fallback then does not apply and the walk unwinds as an ordinary
/// candidate failure so backtracking can reach acyclic records one level
/// up — instead of dying on the twin's entry no-origin assert.
///
/// @param contraLB      The `__contradiction__` twin candidate.
/// @param proved        The expression/validity the walk must justify.
/// @param preserveScope The switched walk's scope mode: true keeps
///                      `proved`'s exact scope (a `zoom in`,
///                      `induction scope` or `int scope` dependency), false
///                      lifts it.
/// @return true if the twin holds at least one origin record for `proved`
///         in the form the switched walk checks; false otherwise.
/// @invariant Non-minting (I-91).
/// @see buildStack — both D-51 fallback sites are the only callers.
bool contradictionLbHoldsRecord(Memory& contraLB,
                const ExpressionWithValidity& proved, bool preserveScope);

public:
/// @brief Emit one chapter's derivation rows for `proved`, walking
        ///        this memoryBlock's origin records ([D-51](../../docs/agentic_swdd/40_decisions.md#d-51)).
        ///
        /// @details
        /// Recursive backtracking search for an acyclic derivation tree.
        /// Candidates are the node's origin records in D-49 preference order;
        /// a candidate whose lifted dependencies touch the current recursion
        /// path is skipped, and a candidate whose subtree fails is rolled back
        /// (rows truncated, `covered` insertions undone through the journal)
        /// before the next is tried.
        ///
        /// When every candidate fails, the contradiction-LB fallback is tried
        /// and then, as a last resort, the front origin is emitted anyway with
        /// its dependency subtree walked in full, so the chapter has content
        /// even though the verifier will flag the residual cycle.
        ///
        /// @param memoryBlock          The LB whose origin records justify
        ///                             `proved`.
        /// @param proved               The expression / validity to justify.
        ///                             Lifted to its shallowest origin-bearing
        ///                             ancestor on entry.
        /// @param stack                Accumulating chapter rows; a failing
        ///                             candidate truncates back to its mark.
        /// @param covered              Nodes already given a row this chapter,
        ///                             so each `(expression, lifted validity)`
        ///                             appears once. Membership only — never
        ///                             iterated.
        /// @param preserveScope Preserve the exact requested source scope of a zoom in dependency.
        /// @param rowsSurviveOnFailure Whether rows this call emits still
        ///                             reach the chapter if it returns false.
        ///                             False only for a recursion from a
        ///                             parent's candidate loop, which truncates
        ///                             the stack the moment it receives false —
        ///                             so the last-resort fallback is skipped
        ///                             there, being work whose only consumer is
        ///                             that truncation. Defaults to true, the
        ///                             conservative value every top-level
        ///                             caller wants.
        /// @return True when an acyclic derivation tree was emitted; false to
        ///         tell a recursive caller to backtrack onto another origin.
        ///
        /// @invariant A false return leaves `covered` and `g_buildStackPath`
        ///            exactly as the call found them.
        /// @see contradictionLbHoldsRecord
bool buildStack(Memory& memoryBlock,
                const ExpressionWithValidity& proved,
                std::vector<std::vector<std::string>>& stack,
                std::set<ExpressionWithValidity>& covered,
                bool rowsSurviveOnFailure = true,
                bool preserveScope = false);

public:
/// @brief Emit one chapter's rows for `proved`: the shallowest-scope walk,
///        and a deeper-scope walk only when that one finds no acyclic tree.
///
/// @details
/// The first walk is `buildStack` exactly as every chapter has always been
/// built: each dependency cited at its shallowest origin-bearing scope
/// (I-39). When it returns true, its rows are the chapter.
///
/// When it returns false (its rows are the degraded chapter with a residual
/// cycle), the walk is repeated from a fresh traversal with
/// `g_deeperScopes` set: a node whose candidates all fail at the shallowest
/// scopes may then cite a dependency at a deeper scope that holds its own
/// records (`originScopeAlternatives`). A statement that also holds at main,
/// derived there only through the implication a subproof is proving, is
/// then cited inside the subproof, where its own record derives it. Every
/// cited row is an origin record, so nothing is invented. An acyclic retry
/// replaces the chapter; a failed retry leaves the first walk's rows, so a
/// chapter without any acyclic derivation is emitted exactly as before.
///
/// @param memoryBlock The LB whose origin records justify `proved`.
/// @param proved      The chapter's head expression and validity.
/// @param stack       Receives the chapter rows; empty on entry.
/// @param covered     Receives the covered nodes of the emitted walk; empty
///                    on entry.
/// @return True when the emitted chapter is an acyclic derivation tree.
///
/// @invariant A chapter the first walk resolves is identical to the
///            `buildStack` chapter; `g_deeperScopes` is false on entry and
///            on return.
/// @see buildStack, originScopeAlternatives, beginRootTraversal.
bool buildChapterStack(Memory& memoryBlock,
                       const ExpressionWithValidity& proved,
                       std::vector<std::vector<std::string>>& stack,
                       std::set<ExpressionWithValidity>& covered);

public:
/// @brief Order expressions by descending proof-stack size.
/// @details Sorts index/value pairs and projects the original expressions in that order; mismatched lengths retain the existing invalid-argument failure.
/// @param expressions Expressions to order.
/// @param values Corresponding proof-stack sizes.
/// @return Expressions in descending size order.
std::vector<ExpressionWithValidity> sortByValuesDesc(const std::vector<ExpressionWithValidity>& expressions,
            const std::vector<int>& values);
private:

SimpleMapStore& simpleMapStore;
TheoremUsageGraph& exportTheoremUsage;
std::string& exportCurrentChapterTheorem;
};
}
namespace gl {
}
