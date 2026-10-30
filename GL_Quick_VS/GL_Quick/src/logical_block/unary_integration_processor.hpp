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
class IntegrationProcessor;
class HashRuleInstaller;
class RejectedSourceReplay;
namespace tests { struct UnaryIntegrationProcessorFixture; }
/// @brief Unary Integration Processor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class UnaryIntegrationProcessor {
friend struct tests::UnaryIntegrationProcessorFixture;
public:
/// @brief Bind inputs for unary integration processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param unaryMemory Borrowed UnaryRuleMemory interface or state; must outlive this component.
/// @param hashRuleInstaller Borrowed HashRuleInstaller interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param integrationProcessor Existing shared state or operation interface, borrowed for this component lifetime.
/// @return None; construction binds the supplied references.
UnaryIntegrationProcessor(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const ProverParameters& parameters, UnaryRuleMemory& unaryMemory, HashRuleInstaller& hashRuleInstaller, RejectedSourceReplay& rejectedSourceReplay, IntegrationProcessor& integrationProcessor);

public:
/// @brief Compose producer chains with demanded unary consumers.
    /// @details New rules and newly observed demands enqueue both join roles;
    /// only eligible consumers enter the premise index. New
    /// results enter normal hash-rule installation, then participate in further
    /// joins. Removed roots invalidate the derived closure as a whole; rebuilding
    /// from live independent roots prevents cyclic support from becoming an axiom.
    /// Closure installs share one rule-index staging window. The independent
    /// composition indexes expose each new rule immediately; hash indexes are
    /// flushed once before returning to admission transfer or inference.
    /// @param mb Resident claimed block with flushed rule indexes.
    /// @return Nothing.
void drainUnaryComposition(Memory& mb);

public:
/// @brief Transfer unary witness demands and request demanded equality copies.
    /// @details The marked head argument becomes u_marker for matching; exactly
    /// one other non-u_ digit argument remains. A successful substitution must
    /// determine every premise argument and preserve the same marked witness.
    /// Complete producer chains additionally request an existing fixed argument's
    /// alias when a positive equality connects it to the demanded marker. Other
    /// premises remain firing obligations. Copies use the deepest compatible
    /// scope and the ordinary variable-copy mail and provenance contract.
    /// Receipts process each rule/demand pair once and are cleared with closure
    /// invalidation. The original integration instruction is never modified.
    /// Every effect follows a receipt mint, and a walk mints a receipt for each
    /// pair it can match. A walk is therefore skipped while the demand key set
    /// and the rule registry (whose reset also clears the receipts) have gained
    /// no key since the previous walk started: it would visit receipted pairs
    /// only. Erased demands only shrink that pair set.
    /// @param mb Owning block with current rule and instruction registries.
    /// @return Nothing.
void transferUnaryIntegrationDemands(Memory& mb);
private:
IntegrationProcessor& integrationProcessor;

CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
const ProverParameters& parameters;
UnaryRuleMemory& unaryMemory;
HashRuleInstaller& hashRuleInstaller;
RejectedSourceReplay& rejectedSourceReplay;
};
}
#include "../inference/hash_rule_installer.hpp"
#include "../logical_block/rejected_source_replay.hpp"
namespace gl {
}

#include "integration_processor.hpp"
