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
class TheoremPublisher;
namespace tests { struct OrEliminationBuilderFixture; }
/// @brief Or Elimination Builder operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class OrEliminationBuilder {
friend struct tests::OrEliminationBuilderFixture;
public:
/// @brief Bind the or elimination builder operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param fullTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param globalTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param orElimBuiltByPair Borrowed std::vector interface or state; must outlive this component.
/// @param orElimCitedOrByMerged Borrowed std::vector interface or state; must outlive this component.
/// @param orElimGuardIndex Borrowed std::vector interface or state; must outlive this component.
/// @param orElimInRunScannedRows Borrowed std::vector interface or state; must outlive this component.
/// @param orElimPendingPairs Borrowed std::vector interface or state; must outlive this component.
/// @param orRuleCompiler Borrowed OrRuleCompiler interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param theoremListMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param theoremPublisher Borrowed TheoremPublisher interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
OrEliminationBuilder(const CompiledDefinitionRegistry& definitions, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& fullTheoremList, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList, std::map<std::pair<std::string, std::string>, std::string>& orElimBuiltByPair, std::map<std::string, std::string>& orElimCitedOrByMerged, std::map<std::string, std::vector<std::pair<std::string, std::string>>>& orElimGuardIndex, std::unordered_set<std::string>& orElimInRunScannedRows, std::vector<std::array<std::string, 4>>& orElimPendingPairs, OrRuleCompiler& orRuleCompiler, const ProverParameters& parameters, ProofHistory& proofHistory, std::mutex& theoremListMutex, TheoremPublisher& theoremPublisher);

private:
/// @brief Token-aligned variable unification of two compiled
/// expressions — the or-elimination license probe's comparator.
///
/// @details
/// Walks @p a and @p b as alternating delimiter runs and token runs
/// (`[A-Za-z0-9_]+`). Delimiter runs must match byte-for-byte —
/// this makes polarity literal (a `!` present on one side only
/// fails the walk, I-175). Token pairs: a numeric token with value
/// >= 9 is a bound non-anchor variable and binds through the
/// @p forward / @p reverse maps (a consistent bijection across the
/// whole call sequence — both maps are caller-owned and shared
/// across the two disjunct alignments and the side-premise images);
/// every other token — operator names, `u_` tokens, anchor-slot
/// numerals 1..8 — must be byte-equal (anchor slots are pinned by
/// the shared `Anchor<...>[1..8]` authoring convention).
///
/// Both verdicts are contractual: `true` = the bijection extends,
/// `false` = the expressions do not align (a defined mismatch the
/// caller skips, not a failure state).
///
/// @param a       First expression (or-theorem side).
/// @param b       Second expression (variant-guard side).
/// @param forward Accumulated a→b variable bindings (extended).
/// @param reverse Accumulated b→a variable bindings (extended).
/// @return True iff the walk completes with a consistent bijection.
/// @see `findOrEliminationLicense` — sole caller.
static bool alignOrEliminationExprs(const std::string& a,
                                            const std::string& b,
                                            std::map<std::string, std::string>& forward,
                                            std::map<std::string, std::string>& reverse);

private:
/// @brief Find a proved or theorem licensing the merge of two
/// guard premises; empty string = the defined no-license verdict.
///
/// @details
/// Scans a snapshot of `globalTheoremList` in list order
/// (deterministic — the first licensing row wins the citation).
/// A row licenses the pair iff: its head's core resolves through
/// `compiledEntity` to an or-category entity with exactly TWO
/// registry leaves (`flattenRegistryOrLeaves`, k=2 first build);
/// the leaves, instantiated with the head's actual args
/// (signature-args → head-args substitution), align with
/// (@p guardA, @p guardB) in either order under ONE variable
/// bijection (`alignOrEliminationExprs`); and every premise of the
/// or-theorem row is covered — its image under the bijection is
/// byte-present in @p commonChain, or the premise is a typing atom
/// `(in[v,...])` whose image variable occurs in @p commonChain
/// (the pragmatic prover-side gate; the external verifier redoes
/// this rigorously via the definition-set machinery).
///
/// The empty return is a contractual verdict, not a failure
/// sentinel (the `headSwitchOne` precedent): a licenseless pair
/// parks and is re-probed next iteration.
///
/// @param guardA      First guard premise (variant A's innermost).
/// @param guardB      Second guard premise (variant B's innermost).
/// @param commonChain The variants' shared premise chain
///                    (root-first, guard excluded).
/// @return The licensing or-theorem row, or "" if none exists yet.
/// @see `constructOrEliminationInRun` — sole caller.
/// @see `alignOrEliminationExprs` — the comparator.
std::string findOrEliminationLicense(const std::string& guardA,
                                             const std::string& guardB,
                                             const std::vector<std::string>& commonChain);

public:
/// @brief In-run pre-split merge (or elimination): derive the
/// unguarded theorem from a proved guard-variant pair under a
/// proved or theorem.
///
/// @details
/// Runs on the phase-4 barrier of every `proveKernel` iteration,
/// single-threaded, immediately after `constructOrTheoremsInRun`
/// (so an or theorem minted this iteration can license a merge in
/// the same window) and before the deferred-compaction drain (so
/// the merged theorem's broadcast rides this iteration's drain).
///
/// Scan: `globalTheoremList` rows not yet in
/// `orElimInRunScannedRows` are disintegrated once; a row whose
/// innermost premise is binder-free files into `orElimGuardIndex`
/// under its common-part key. Two rows under one key with
/// different guards form a candidate pair, parked on
/// `orElimPendingPairs`.
///
/// Merge: for each pending pair, `findOrEliminationLicense` looks
/// for a proved or theorem whose two disjuncts align with the two
/// guards under one variable bijection (literal polarity, I-175)
/// and whose side premises are covered by the common chain. On a
/// license: the merged theorem is rebuilt canonically via
/// `reconstructImplicationFullBind(commonChain, head)` and lands as
/// a DERIVED row — `appendGlobalTheorem(merged, "or elimination",
/// variantA, variantB)`, one `fullTheoremList` mirror, the or
/// citation into `orElimCitedOrByMerged`, and the standard
/// deferred-compaction broadcast for implication-shaped results,
/// documented with the `("theorem", [])` origin at `main` through
/// `mergeTheoremOriginMailIntoRoot`. The merged theorem is never a
/// pool conjecture and no LB proof state is touched; the variants
/// stay first-class proved rows (never subsumed). In compressor mode
/// the seam is a defined no-op, exactly like the or seam.
///
/// @return Nothing.
/// @invariant Single-threaded seam — phase-4 barrier only. The
///            merge mints nothing into the operator registries; its
///            only writes are the theorem list, the seam ledgers,
///            the compaction queue, and the theorem-origin
///            documentation in the root's mailOut.
/// @see `findOrEliminationLicense` — the license probe.
/// @see `constructOrTheoremsInRun` — the sibling seam this one
///      mirrors (and must follow in the window).
void constructOrEliminationInRun();
private:

const CompiledDefinitionRegistry& definitions;
std::vector< std::tuple<std::string, std::string, std::string, std::string >>& fullTheoremList;
std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList;
std::map<std::pair<std::string, std::string>, std::string>& orElimBuiltByPair;
std::map<std::string, std::string>& orElimCitedOrByMerged;
std::map<std::string, std::vector<std::pair<std::string, std::string>>>& orElimGuardIndex;
std::unordered_set<std::string>& orElimInRunScannedRows;
std::vector<std::array<std::string, 4>>& orElimPendingPairs;
OrRuleCompiler& orRuleCompiler;
const ProverParameters& parameters;
ProofHistory& proofHistory;
std::mutex& theoremListMutex;
TheoremPublisher& theoremPublisher;
};
}
#include "../proof/theorem_publisher.hpp"
namespace gl {
}
