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
#include "cached_proof_stack.hpp"
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
class ProofReconstructor;
namespace tests { struct ProofGraphWriterFixture; }
/// @brief Proof Graph Writer operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class ProofGraphWriter {
friend struct tests::ProofGraphWriterFixture;
public:
/// @brief Bind the proof graph writer operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param body Borrowed Memory interface or state; must outlive this component.
/// @param globalTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param globalTheoremStrings Borrowed std::vector interface or state; must outlive this component.
/// @param globalTheoremProducers Borrowed std::vector interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param cachedProofStacks Borrowed std::vector interface or state; must outlive this component.
/// @param exportTheoremUsage Borrowed TheoremUsageGraph interface or state; must outlive this component.
/// @param exportCurrentChapterTheorem Borrowed std::string interface or state; must outlive this component.
/// @param orElimCitedOrByMerged Borrowed std::vector interface or state; must outlive this component.
/// @param compilationPersistence Borrowed CompilationPersistence interface or state; must outlive this component.
/// @param theoremPublisher Borrowed TheoremPublisher interface or state; must outlive this component.
/// @param proofReconstructor Borrowed ProofReconstructor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
ProofGraphWriter(Memory& body, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList, std::unordered_set<std::string>& globalTheoremStrings, std::vector<const Memory*>& globalTheoremProducers, const CompiledDefinitionRegistry& definitions, SimpleMapStore& simpleMapStore, std::map<std::string, CachedProofStack>& cachedProofStacks, TheoremUsageGraph& exportTheoremUsage, std::string& exportCurrentChapterTheorem, std::map<std::string, std::string>& orElimCitedOrByMerged, CompilationPersistence& compilationPersistence, TheoremPublisher& theoremPublisher, ProofReconstructor& proofReconstructor);

public:
/// @brief Write debug proof chapters for the requested logical block.
/// @details Resolves the original child path, reconstructs every recorded end and publishes the indexed chapters and mapping in the existing order.
/// @param path Logical-block routing keys from the root.
/// @param outDirParam Destination for debug chapters and their mapping.
/// @return No value.
void findEnds(const std::vector<std::string>& path, const std::filesystem::path& outDirParam);

public:

/// @brief Export the recorded theorem proofs into raw chapter files.
/// @details Preserves chapter selection, reconstruction, cross-chapter usage checks, serialization and deload release order. Synthetic citations are seeded before traversal; chosen edges are committed after writing each chapter.
/// @param theoremList Ordered theorem records to export.
/// @param outDir Destination for chapter files and their mapping.
/// @return No value.
void generateRawProofGraph(
            const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& theoremList,
            const std::filesystem::path& outDir = std::filesystem::path("files") / "raw_proof_graph");

/// @brief Whether a block's history records a `vacuous truth` line for a goal
///        at main (D-573).
/// @details The vacuous-truth branch of `GoalProcessor::dischargeContradiction`
/// closes an induction goal without depositing it as a statement; the closure
/// is recorded only as the goal's history line at main. The induction-chapter
/// gates of `generateRawProofGraph` accept such a block as the goal's closing
/// block, besides a block holding the goal as a statement; the chapter is then
/// built from that history line like any other. Reads the origin interner and
/// `exprOriginMap`, which a deload leaves in memory.
/// @param block The induction block whose history is read.
/// @param goal The goal expression in its literal spelling.
/// @return `true` iff the history holds a `vacuous truth` record for
///         (@p goal, main).
static bool hasVacuousTruthAtMain(const Memory& block, const std::string& goal);
private:

Memory& body;
std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList;
std::unordered_set<std::string>& globalTheoremStrings;
std::vector<const Memory*>& globalTheoremProducers;
const CompiledDefinitionRegistry& definitions;
SimpleMapStore& simpleMapStore;
std::map<std::string, CachedProofStack>& cachedProofStacks;
TheoremUsageGraph& exportTheoremUsage;
std::string& exportCurrentChapterTheorem;
std::map<std::string, std::string>& orElimCitedOrByMerged;
CompilationPersistence& compilationPersistence;
TheoremPublisher& theoremPublisher;
ProofReconstructor& proofReconstructor;
};
}
#include "../proof/proof_reconstructor.hpp"
namespace gl {
}
