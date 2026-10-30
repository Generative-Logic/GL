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
namespace tests { struct CounterexampleFilterFixture; }
/// @brief Counterexample Filter operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class CounterexampleFilter {
friend struct tests::CounterexampleFilterFixture;
public:
/// @brief Bind the counterexample filter operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param anchorInfo Borrowed ce::AnchorInfo interface or state; must outlive this component.
/// @param ceBody Borrowed Memory interface or state; must outlive this component.
/// @param ceSimpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param filteredConjecturesDirectory Borrowed std::filesystem::path interface or state; must outlive this component.
/// @param hashRuleInstaller Borrowed HashRuleInstaller interface or state; must outlive this component.
/// @param inductionMemoryBlocks Borrowed std::vector interface or state; must outlive this component.
/// @param lbStore Borrowed LbStore interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param permanentBodiesCE Borrowed std::vector interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param updateGlobalTuples Borrowed std::vector interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
CounterexampleFilter(const ce::AnchorInfo& anchorInfo, Memory& ceBody, SimpleMapStore& ceSimpleMapStore, const CompiledDefinitionRegistry& definitions, std::filesystem::path& filteredConjecturesDirectory, HashRuleInstaller& hashRuleInstaller, std::vector<Memory*>& inductionMemoryBlocks, LbStore& lbStore, const ProverParameters& parameters, std::vector<Memory*>& permanentBodiesCE, StatementMemory& statementMemory, std::vector< std::tuple<int, uint64_t, int>>& updateGlobalTuples);

public:
// ------------------------------------------------------------------
    // CE-filter file IO.
    // ------------------------------------------------------------------

    /// @brief Load the per-anchor *simple facts* — small ground-fact tables
    /// the CE filter probes each conjecture against — from the `files/simple_facts/`
    /// directory.
    ///
    /// @details
    /// Resolves the anchor's short name (strips the `"Anchor"` prefix and
    /// lowercases — `"AnchorPeano"` → `"peano"`, `"AnchorGauss"` → `"gauss"`),
    /// then walks `files/simple_facts/` for files matching
    /// `simple_facts_<name>(_<n>)?\.txt` (case-insensitive). Hits are sorted
    /// by the optional integer `n` ascending so j-copies appear in
    /// `simple_facts_<name>_5.txt`, `simple_facts_<name>_6.txt`, … order.
    /// Each surviving file is loaded line-by-line; trailing `\r` from Windows
    /// line endings is stripped. The result is a vector-of-vectors: one inner
    /// vector per file, in sorted order.
    ///
    /// j-copy ordering matters because the CE filter generates encoded
    /// requests against each fact-list separately and the prover's
    /// determinism depends on stable iteration order across runs.
    ///
    /// @return Vector of fact-lists, sorted by their numeric suffix.
    /// @pre  `this->anchorInfo.name` has already been initialized (it is set
    ///       in the `Prover` constructor by `initAnchor`).
    /// @post Files are read once; no caching — each call re-walks the
    ///       directory. The CE filter calls this once per batch.
    /// @see `docs/agentic_swdd/10_pipeline/03_ce_filter.md` — fact-base shape rationale.
std::vector<std::vector<std::string>> readSimpleFacts() const;

public:
/// @brief Write the post-CE-filter conjecture list to
    /// `filteredConjecturesDirectory/filtered_conjectures.txt`.
    ///
    /// @details
    /// Called once per batch after the CE filter has finished probing every
    /// conjecture. Survivors (those that did NOT produce a confirmed
    /// contradiction in the CE pass, or that the CE filter is configured to
    /// pass through verbatim) are persisted to disk so the main prover stage
    /// can read them as its theorem list. The file is overwritten on each
    /// call.
    ///
    /// @param filteredConjectures The full list of conjectures to persist,
    ///                            one per output line.
    /// @post The selected directory contains `filtered_conjectures.txt` with
    ///       `filteredConjectures.size()` lines of MPL text.
    /// @see `proof/counterexample_filter.hpp::CEFilter` — primary caller.
    /// @see `docs/agentic_swdd/10_pipeline/03_ce_filter.md` — pipeline stage chapter.
void saveFilteredConjectures(const std::vector<std::string>& lines);

public:
// ------------------------------------------------------------------
    // CE-filter batch lifecycle.
    // ------------------------------------------------------------------

    /// @brief Seed the CE-filter LB tree with the simple-fact base — every
    /// fact is installed as either a statement or a hash-engine rule,
    /// depending on its shape.
    ///
    /// @details
    /// One CE-filter LB is created per conjecture batch slot
    /// (`0..batchSize-1`) under `ceBody` in `ceSimpleMapStore`. For each slot, every
    /// entry of `simpleFacts` is dispatched through the generic
    /// `addExprToMemoryBlock` so atomic facts become statements in
    /// `Memory::intEncodedStatements` while implication-shaped facts become
    /// hash rules in `Memory::overallHashMemory`. The polymorphism is what
    /// makes the CE filter setup tractable across batches with very
    /// different fact-base shapes (Peano = mostly atomic; Gauss = mix of
    /// atomic + implication).
    ///
    /// Per I-1, this function is called AFTER `precompileStructuralOperators`
    /// for the conjecture list, so any structural operator (`or0`,
    /// `existence2`, etc.) has already been disintegrated to base form
    /// before the simple facts hit the admission map.
    ///
    /// @param simpleFacts  Flat list of fact texts. The CE-filter loop
    ///                     reuses the same fact set across every batch
    ///                     slot.
    /// @param batchSize    Number of CE-filter slots to seed. One LB is
    ///                     installed under `ceBody` at routing key
    ///                     `std::to_string(i)` in `ceSimpleMapStore`, for each
    ///                     `i` in `[0, batchSize)`.
    /// @pre  `precompileStructuralOperators` has run for the batch's
    ///       theorem list (I-1).
    /// @post Each slot LB's `intEncodedStatements`
    ///       and `…->overallHashMemory` carry the simple-fact base, ready
    ///       for the per-slot conjecture probe.
    /// @see `addExprToMemoryBlock` — installer.
void loadFactsForCEFiltering(
            std::vector<std::string> simpleFacts,
            int batchSize);

public:
/// @brief Install one conjecture into the CE-filter LB so the prover can
    /// probe whether its negation contradicts the simple-fact base.
    ///
    /// @details
    /// Wraps the conjecture in its negated form (the contradiction probe
    /// looks for `!conjecture` becoming derivable from the simple facts;
    /// success means the conjecture is true on this fact base) and submits
    /// it via `addExprToMemoryBlock`. The result feeds into the per-row
    /// `ContradictionItem` of the CE-filter contradiction table.
    ///
    /// Called once per conjecture per CE-filter pass; the per-batch budget
    /// is `parameters.numberIterationsConjectureFiltering` outer iterations.
    ///
    /// @param conjecture        Raw MPL conjecture text.
    /// @param body              CE-filter LB (already seeded with simple
    ///                          facts via `loadFactsForCEFiltering`).
    /// @param coreId            Logical-core id (CE filter uses 0 in single
    ///                          core mode).
    /// @param conjectureIndex   Row index into `contradictionTable`.
    /// @post `body` carries the negated conjecture as a pending obligation;
    ///       on hash-burst, the prover may set
    ///       `contradictionTable[conjectureIndex].successful = true`.
void addConjectureForCEFiltering(const std::string& conjecture,
            Memory *mb,
            int cIndex);

public:
/// @brief Tear down the CE-filter batch state — clear the simple-fact
    /// LB, free per-batch arenas, reset the contradiction table.
    ///
    /// @details
    /// Called once per batch after the CE filter has finished probing every
    /// conjecture and `saveFilteredConjectures` has persisted the survivors.
    /// Drops the simple-fact LB tree
    /// (so the main prover stage starts from a clean root), and clears the
    /// per-row `ContradictionItem` records so re-running on a fresh batch
    /// is well-defined.
    ///
    /// @post All CE-filter-only state is released; subsequent calls to the
    ///       main prover stage operate on a fresh LB tree.
void releaseCEBatchMemory();
private:

const ce::AnchorInfo& anchorInfo;
Memory& ceBody;
SimpleMapStore& ceSimpleMapStore;
const CompiledDefinitionRegistry& definitions;
std::filesystem::path& filteredConjecturesDirectory;
HashRuleInstaller& hashRuleInstaller;
std::vector<Memory*>& inductionMemoryBlocks;
LbStore& lbStore;
const ProverParameters& parameters;
std::vector<Memory*>& permanentBodiesCE;
StatementMemory& statementMemory;
std::vector< std::tuple<int, uint64_t, int>>& updateGlobalTuples;
};
}
namespace gl {
}
