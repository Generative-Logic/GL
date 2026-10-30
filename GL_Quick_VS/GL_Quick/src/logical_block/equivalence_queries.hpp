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
namespace tests { struct EquivalenceQueriesFixture; }
/// @brief Equivalence Queries operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class EquivalenceQueries {
friend struct tests::EquivalenceQueriesFixture;
public:

/// @brief Whether a ground statement is known at a scope or one of its
        ///        ancestors, up to the scope's complete classes
        ///        (D-406).
        ///
        /// @details
        /// The known probe of the implication gate. In order: a reflexive
        /// equality `(=[x,x])` is known; a literal known row at the scope or
        /// an ancestor (`ancestorKnown`) is known; otherwise the shared
        /// statement posting index compares the text with the rows owned by
        /// each scope on the ancestor chain, both normalized through the
        /// complete classes of @p scopeId (I-254), so a row that names another
        /// class member counts. The disintegration state of a matching row is
        /// irrelevant.
        ///
        /// @param element Clean (u_-stripped) ground statement text on
        ///                caller-owned storage.
        /// @param scopeId Minted scope where the statement is needed.
        /// @param mb      Owning block, its classes and known-statement store.
        /// @return True iff the statement is known at @p scopeId or an
        ///         ancestor, literally or up to the classes of @p scopeId.
        /// @invariant Creates no proof row; only arena-backed derived
        ///            comparison indexes are synchronized.
        /// @see compactKnownUpToEquality (the exact-scope twin),
        ///      ancestorKnown, matchingCanonicalTemplates.
bool elementKnownAtOrAbove(StrSpan element, NameId scopeId, Memory& mb);

/// @brief Bind inputs for equivalence queries.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
EquivalenceQueries(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer);

public:
/// @brief Check for a fully disintegrated equivalent expression at one scope.
        /// @details Forwards to the span implementation, whose identity uses the
        /// complete target-scope classes. Ancestor and sibling receipt ownership
        /// does not satisfy this scope's disintegration gate.
        /// @param expr Candidate expression, owned by the caller.
        /// @param validityName Exact disintegration scope.
        /// @param memoryBlock Owning block and complete class views.
        /// @return Whether an equivalent known row is fully disintegrated here.
        /// @see checkForEquivalence(StrSpan, StrSpan, Memory&).
bool checkForEquivalence(const std::string& expr,
            const std::string& validityName, Memory& memoryBlock);

public:
/// @brief Test the canonical identity of fully disintegrated known rows.
        /// @details An already completed literal row is a direct hit. Otherwise
        /// the shared posting index compares both sides at the exact target scope
        /// and reads the live fullyDisintegrated bit from each matching original
        /// record. Classes come from the complete view; no Cartesian expression
        /// family is materialized. Existing application multiplication is separate.
        /// @param expr Candidate expression on caller-owned storage.
        /// @param validityName Registered scope of this disintegration attempt.
        /// @param memoryBlock Owning block, classes and known-statement store.
        /// @return Whether a matching row is fully disintegrated at this scope.
        /// @invariant Uses arena-backed derived indexes; creates no proof row.
        /// @see matchingCanonicalTemplates, CanonicalTemplateIndex.
bool checkForEquivalence(StrSpan expr, StrSpan validityName, Memory& memoryBlock);

public:
/// @brief Test whether a compact is a known statement at one scope, up to
///        that scope's complete equivalence classes.
///
/// @details The known-part check of `prepareIntegrationCore2` Case A: an
/// implication compact that is already a known statement needs no subproof.
/// A literal row at the exact scope is a direct hit. Otherwise the shared
/// statement posting index compares the compact with every known statement
/// owned by that exact scope after both are normalized through the scope's
/// complete class view (I-254), so a statement that names another member of a
/// class counts. Unlike `checkForEquivalence`, the disintegration state of the
/// matching row is irrelevant: a known statement is known whether or not it
/// was disintegrated.
///
/// @param compact Candidate compact statement on caller-owned storage.
/// @param validityName Exact scope that owns the matching statement; it must
///        already be registered in the block's name map.
/// @param memoryBlock Owning block, its classes and known-statement store.
/// @return Whether the compact, or a class variant of it at this scope, is a
///         known statement owned by that scope.
/// @invariant Creates no proof row; only arena-backed derived comparison
///            indexes are synchronized.
/// @see checkForEquivalence, matchingCanonicalTemplates.
bool compactKnownUpToEquality(StrSpan compact, StrSpan validityName, Memory& memoryBlock);
private:

CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
};
}
namespace gl {
}

namespace gl {
/// @brief Whether a ground statement is known at a scope or one of its
        ///        ancestors, up to the scope's complete classes
        ///        (D-406).
        ///
        /// @details
        /// The known probe of the implication gate. In order: a reflexive
        /// equality `(=[x,x])` is known; a literal known row at the scope or
        /// an ancestor (`ancestorKnown`) is known; otherwise the shared
        /// statement posting index compares the text with the rows owned by
        /// each scope on the ancestor chain, both normalized through the
        /// complete classes of @p scopeId (I-254), so a row that names another
        /// class member counts. The disintegration state of a matching row is
        /// irrelevant.
        ///
        /// @param element Clean (u_-stripped) ground statement text on
        ///                caller-owned storage.
        /// @param scopeId Minted scope where the statement is needed.
        /// @param mb      Owning block, its classes and known-statement store.
        /// @return True iff the statement is known at @p scopeId or an
        ///         ancestor, literally or up to the classes of @p scopeId.
        /// @invariant Creates no proof row; only arena-backed derived
        ///            comparison indexes are synchronized.
        /// @see compactKnownUpToEquality (the exact-scope twin),
        ///      ancestorKnown, matchingCanonicalTemplates.
inline bool EquivalenceQueries::elementKnownAtOrAbove(StrSpan element, NameId scopeId, Memory& mb) {
            assert(scopeId != 0 && "elementKnownAtOrAbove: expected a minted scope");
            if (startsWithSpan(element, "(=[", 3)) {
                StrSpan pair[ExecutionParameters::MAX_ARITY];
                const int32_t n = getArgsSpans(element, pair, ExecutionParameters::MAX_ARITY);
                if (n == 2 && equalSpans(pair[0], pair[1])) return true;
            }
            const NameId origId = mb.nameMap.lookup(element);
            if (origId != 0 && gl::StatementKnowledge::ancestorKnown(mb, origId, scopeId, true)) return true;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            DirtyState dirty = DirtyState::Clean;
            const int32_t ancN = mb.nameMap.ancLen(scopeId);
            for (int32_t k = 0; k < ancN; ++k) {
                PagedVector<int64_t> matches(&arena, &dirty);
                canonicalTemplates.matchingCanonicalTemplates(mb, mb.intKnownStatements,
                    mb.lbMemory.canonicalUses.statementMatches, element,
                    mb.nameMap.ancAt(scopeId, k), scopeId, matches, arena, true);
                if (matches.size() > 0) return true;
            }
            return false;
        }
}
