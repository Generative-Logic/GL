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
namespace tests { struct ProofBarrierFixture; }
/// @brief Proof Barrier operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class ProofBarrier {
friend struct tests::ProofBarrierFixture;
public:
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
ProofBarrier(const ProverParameters& parameters, Memory& body, std::vector< std::tuple<int, uint64_t, int>>& updateGlobalTuples, std::vector<Memory*>& inductionMemoryBlocks, bool& ceFilteringActive, const CompiledDefinitionRegistry& definitions, MailLog& mailLog, bool& rollingMailHistoryEnabled, int64_t& peakDeloadableMailOutBytes, SimpleMapStore& simpleMapStore, LbArena& dischargeScratch, std::vector<std::tuple<std::string, int, int>>& pendingCompactionQueue, std::vector<Memory*>& pendingDischarge, std::unique_ptr<MemorySteward>& steward, StructuralCompiler& structuralCompiler, OrRuleCompiler& orRuleCompiler, bool& deactivationSweepOwed, AdmissionMemory& admissionMemory, ProofHistory& proofHistory, ScopeLifecycle& scopeLifecycle, BlockMailProcessor& blockMailProcessor, AnchorScopeQueries& anchorScopeQueries, TheoremPublisher& theoremPublisher, OrTheoremBuilder& orTheoremBuilder, OrEliminationBuilder& orEliminationBuilder);

public:
// … earlier declarations …
void activateZeroCondition(Memory& memoryBlock);
public:
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
/// block chains
/// (`TheoremPublisher::compareProducerChains`) and settled one by one
/// (`ScopeLifecycle::settleStepClosure`); every closure except a tie loser
/// wakes its zero block through `activateZeroCondition`. Each step-win event
/// writes one `[STEP-WIN]` line to the run diagnostics log. In compressor
/// mode, which reuses this prover on blocks without step blocks, the seam
/// does nothing.
///
/// @return No value.
/// @invariant Single-threaded post-join seam (I-28); the steward is live.
/// @see ScopeLifecycle::settleStepClosure, ScopeLifecycle::retryWaitingPromotions.
void settleStepWins();
public:
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
void settleUselessSteps(const std::vector<Memory*>& bodies);
public:
/// @brief Commit mail and drain the completed burst's proof results.
/// @details Preserves the post-phase-three order of mail commitment, theorem publication, deactivation, OR construction, compaction, discharge and prefetch. Existing recipients, queues and storage owners remain unchanged.
/// @param bodies Complete burst logical-block grid.
/// @param active Logical blocks selected at the burst boundary.
/// @param workers Existing worker count used by the steward windows.
/// @param seamScratchSlot Reserved single-threaded scratch slot selected at burst entry.
/// @return No value.
/// @invariant Called only after every phase-three worker has joined.
void finish(const std::vector<Memory*>& bodies, const std::vector<Memory*>& active, unsigned workers, unsigned seamScratchSlot);

private:

const ProverParameters& parameters;
Memory& body;
std::vector< std::tuple<int, uint64_t, int>>& updateGlobalTuples;
std::vector<Memory*>& inductionMemoryBlocks;
bool& ceFilteringActive;
const CompiledDefinitionRegistry& definitions;
MailLog& mailLog;
bool& rollingMailHistoryEnabled;
int64_t& peakDeloadableMailOutBytes;
SimpleMapStore& simpleMapStore;
LbArena& dischargeScratch;
std::vector<std::tuple<std::string, int, int>>& pendingCompactionQueue;
std::vector<Memory*>& pendingDischarge;
std::unique_ptr<MemorySteward>& steward;
StructuralCompiler& structuralCompiler;
OrRuleCompiler& orRuleCompiler;
bool& deactivationSweepOwed;
AdmissionMemory& admissionMemory;
ProofHistory& proofHistory;
ScopeLifecycle& scopeLifecycle;
BlockMailProcessor& blockMailProcessor;
AnchorScopeQueries& anchorScopeQueries;
TheoremPublisher& theoremPublisher;
OrTheoremBuilder& orTheoremBuilder;
OrEliminationBuilder& orEliminationBuilder;
};
}
namespace gl {
}
