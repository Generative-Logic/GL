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
#include "../proof/phase2_backend.hpp"
#include "../proof/phase2_executor.hpp"
#include "../proof/proof_barrier.hpp"
#include "../proof/cached_proof_stack.hpp"
#include "../proof/theorem_usage_graph.hpp"
#include "../proof/proof_reconstructor.hpp"
#include "../proof/proof_graph_writer.hpp"
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
#include <optional>
#include <memory>
namespace gl { namespace tests { struct ProverInitializationFixture; }
namespace gpu { class CudaContextWarmup; }
/// @brief The compiled state of one configuration: the definition registry and
///        the reuse state exactly as `ProverInitialization::initialize` leaves
///        them after compiling the configured definitions.
/// @details Recorded from a constructed prover and handed to the construction
/// of a later prover of the same configuration, which then installs it instead
/// of compiling again (`ProverInitialization::initialize`). The compilers hold
/// only references to these two stores, so the pair is the complete output of
/// the definition compile.
struct PrecompiledDefinitions {
    CompiledDefinitionRegistry registry;
    CompilationReuseState reuse;
};
class ProverInitialization {
friend struct tests::ProverInitializationFixture;
public:
/// @brief Bind the inputs used to initialize one prover batch.
/// @details Stores explicit references to the owners; construction does not run initialization.
/// @param parameters Existing batch state or operation used during initialization.
/// @param phase2Backend Existing batch state or operation used during initialization.
/// @param body Existing batch state or operation used during initialization.
/// @param allMappingsAna Existing batch state or operation used during initialization.
/// @param allPermutationsAna Existing batch state or operation used during initialization.
/// @param globalDependencies Existing batch state or operation used during initialization.
/// @param definitions Existing batch state or operation used during initialization.
/// @param anchorInfo Existing batch state or operation used during initialization.
/// @param anchorID_ Existing batch state or operation used during initialization.
/// @param permanentBodies Existing batch state or operation used during initialization.
/// @param logicalCores Existing batch state or operation used during initialization.
/// @param structuralCompiler Existing batch state or operation used during initialization.
/// @param supportingRuleCompiler Existing batch state or operation used during initialization.
/// @param compilationPersistence Existing batch state or operation used during initialization.
/// @param compilationReuse Reuse component whose state a precompiled configuration replaces.
/// @return Constructed initialization interface.
ProverInitialization(ProverParameters& parameters, Phase2Backend& phase2Backend, Memory& body, std::map<int64_t, std::vector<std::vector<int> > >& allMappingsAna, std::map<int, std::vector<std::vector<int> > >& allPermutationsAna, Dependencies& globalDependencies, CompiledDefinitionRegistry& definitions, ce::AnchorInfo& anchorInfo, std::string& anchorID_, std::vector<Memory*>& permanentBodies, unsigned& logicalCores, StructuralCompiler& structuralCompiler, SupportingRuleCompiler& supportingRuleCompiler, CompilationPersistence& compilationPersistence, CompilationReuse& compilationReuse);
/// @brief Initialize the configuration, compiler and memory services for one batch.
/// @details Executes the existing configuration loads, backend selection, pool
/// setup, mapping construction, registry load and rule preparation in their
/// order. The caller owns every referenced store.
///
/// With @p precompiled, the definition compile (`compileCoreExpressionMap`,
/// `compileContrapositionFamilies`) is replaced by installing the recorded
/// registry and reuse state; every other step runs unchanged. The record must
/// come from a prover of the same configuration constructed without a
/// compiled-registry input, so the installed state is exactly what this
/// prover's own compile would produce.
/// @param anchorID Requested anchor configuration.
/// @param selectedPhase2Backend Optional caller-selected phase-two backend.
/// @param glBinaryFile Optional compiled-registry input.
/// @param precompiled Optional recorded compile of the same configuration;
///        exclusive with @p glBinaryFile (asserted).
/// @return No value.
/// @invariant Runs at the prover construction boundary, before any proof work.
void initialize(const std::string& anchorID, const std::optional<Phase2Backend>& selectedPhase2Backend, const std::optional<std::filesystem::path>& glBinaryFile, const PrecompiledDefinitions* precompiled);
/// @brief Release the initialization interface.
/// @details On the CUDA route this waits for the background creation of the
/// CUDA context that `initialize` started (`gpu::CudaContextWarmup`); it is
/// defined where that type is complete.
/// @return No value.
~ProverInitialization();
private:
/// @brief Enumerate the finite argument mappings used by the request pipeline.
/// @details Preserves packed-key order and mixed-radix enumeration of every function for the configured arity ranges.
/// @param n Exclusive domain-size limit.
/// @param m Exclusive target-size limit.
/// @return Existing mapping table indexed by packed domain and target sizes.
static std::map<int64_t, std::vector<std::vector<int>>> generateAllMappings(int n, int m);
ProverParameters& parameters;
Phase2Backend& phase2Backend;
Memory& body;
std::map<int64_t, std::vector<std::vector<int> > >& allMappingsAna;
std::map<int, std::vector<std::vector<int> > >& allPermutationsAna;
Dependencies& globalDependencies;
CompiledDefinitionRegistry& definitions;
ce::AnchorInfo& anchorInfo;
std::string& anchorID_;
std::vector<Memory*>& permanentBodies;
unsigned& logicalCores;
StructuralCompiler& structuralCompiler;
SupportingRuleCompiler& supportingRuleCompiler;
CompilationPersistence& compilationPersistence;
CompilationReuse& compilationReuse;
#ifdef GL_CUDA
// Started by `initialize` on the CUDA route: creates the CUDA context while the
// batch sets up, so the first CUDA phase-two pass does not wait for it.
std::unique_ptr<gpu::CudaContextWarmup> cudaContextWarmup;
#endif
}; }

