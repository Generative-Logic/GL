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
namespace tests { struct BlockKeyReaderFixture; }
/// @brief Block Key Reader operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class BlockKeyReader {
friend struct tests::BlockKeyReaderFixture;
public:
/// @brief Bind inputs for block key reader.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @return None; construction binds the supplied references.
BlockKeyReader();

public:
/// @brief Compute the *global key* for a memory block — the canonical
/// vector identifying its position in the LB tree.
///
/// @details
/// The global key is a sequence of `exprKey` values from the LB up to
/// the root sentinel, in deepest-first order. Used as a `std::map` key
/// in cross-LB tables (e.g. proof-stack caches) where a `Memory*`
/// pointer alone would not survive teardown / re-creation of the LB.
///
/// @param memoryBlock LB to compute the key for.
/// @return Vector of `exprKey` values from this LB up to the root,
///         deepest first. Returns an empty vector for the root sentinel
///         itself (its `exprKey` is empty by definition).
/// @see The full-LB-chain trace requirement for any debug trap that
///      identifies a target LB; this function returns exactly the
///      chain such traps must match.
std::vector<std::string> getGlobalKey(const Memory& memoryBlock);

public:
/// @brief Compute the *global key* for a memory block into a
/// caller-owned span array — the 0%-heap production form.
///
/// @details
/// Walks `parentMemory` to the root sentinel, writing each non-empty
/// `exprKeyView()` span (a zero-copy view over the process-wide
/// `skeletonInterner`, never deloaded and frozen during the parallel
/// phase, so no walk mints it — I-97) into `out` in deepest-first
/// order, then reverses the filled range in place. The spans alias
/// the interner's cold bytes and inherit its lifetime; they stay
/// valid until the next mint into `skeletonInterner`, which does not
/// happen on the discharge path. Byte-identical to the retained
/// `std::vector<std::string>` oracle above (`exprKeyView` bytes ==
/// `exprKey`). Body in `logical_block/block_key_reader.cpp`.
///
/// @param memoryBlock LB to compute the key for.
/// @param out         Caller-owned span array of capacity `cap`;
///                    filled `[0, n)` with the reversed key spans.
/// @param cap         Capacity of `out`; a loud Rule-19 assert fires
///                    if the LB chain is deeper than `cap`
///                    (STOP-widen at FTA scale).
/// @return The number of spans written (`n <= cap`).
/// @see The `std::vector<std::string>` oracle overload above.
int32_t getGlobalKey(const Memory& memoryBlock, StrSpan* out,
                             int32_t cap);
private:

};
}
namespace gl {
}
