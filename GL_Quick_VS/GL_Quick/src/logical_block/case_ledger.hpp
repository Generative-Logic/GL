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
namespace tests { struct CaseLedgerFixture; }
/// @brief Case Ledger operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class CaseLedger {
friend struct tests::CaseLedgerFixture;
public:
/// @brief Bind inputs for case ledger.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
CaseLedger(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions);

public:
/// @brief Unpack the canonical half of a processed-or ledger row.
        ///
        /// @details
        /// A `Memory::processedOrLedger` value is `packInt32Pair(canonicalId,
        /// originalId)`; this reads the high half — the or text's
        /// equality-and-symmetry processing id under the scope's classes as of the
        /// last `recanonicalizeProcessedOrLedger` at that scope.
        ///
        /// @param row A ledger run value.
        /// @return The canonical NameMap id (never 0 for a recorded row).
        /// @see ledgerOriginalId; recordProcessedOr.
static NameId ledgerCanonicalId(int64_t row);

public:
/// @brief Unpack the original half of a processed-or ledger row.
        ///
        /// @details
        /// The low half of `packInt32Pair(canonicalId, originalId)` — the
        /// door-canonical text recorded when the or received full
        /// processing; the cohort signature every clear path and the D-331
        /// drain key on.
        ///
        /// @param row A ledger run value.
        /// @return The original NameMap id (never 0 for a recorded row).
        /// @see ledgerCanonicalId; clearProcessedOr.
static NameId ledgerOriginalId(int64_t row);

private:
/// @brief Record original and canonical token dependencies of an OR marker.
        /// @details The logical owner remains the original expression and exact
        /// scope. Replacing the marker updates only that owner's posting edges.
        /// @param mb Owning block.
        /// @param original Original OR expression identifier.
        /// @param canonical Current canonical processing identifier.
        /// @param scope Exact marker scope.
        /// @param arena Caller-owned page-tier scratch.
        /// @return Nothing.
inline void indexCanonicalOrMarker(Memory& mb, NameId original,
                NameId canonical, NameId scope, ScratchArena& arena);

private:
/// @brief Initialize OR-marker dependencies once from authoritative runs.
        /// @details Marker writers maintain later additions and replacements;
        /// scope cleanup retires closed entries; canonical release clears all state.
        /// @param mb Owning block.
        /// @param arena Caller-owned page-tier scratch.
        /// @return Nothing.
inline void ensureCanonicalOrMarkerUses(Memory& mb, ScratchArena& arena);

private:
/// @brief Orient the validated digit pair for OR processing identity.
///
/// @details Swaps argument positions, never all occurrences of their values.
/// Other slots, including a repeated value in another slot, remain unchanged.
/// The returned text uses caller-owned string scratch only when changed.
/// @param expression Positive numbered OR instance.
/// @param strings Caller-owned string arena.
/// @return Input text or its lexicographically smaller validated orientation.
/// @invariant No proof expression, definition or interner is changed.
/// @see compileOrSymmetry; canonicalOrIdAtScope.
StrSpan canonicalOrProcessingForm(StrSpan expression,
            ScratchArena& strings) const;

public:
/// @brief Compute an OR processing id under the scope's equality classes.
///
/// @details Applies the existing equality chooser, then the compiled symmetric
/// slot orientation. Only the resulting processing identity is interned.
/// @param mb Owning block.
/// @param origId Minted original OR id.
/// @param validityName Scope whose existing equality bucket applies.
/// @return Minted processing id; original proof text is retained separately.
/// @invariant No proof history or known statement is rewritten.
/// @see canonicalOrProcessingForm; recordProcessedOr.
NameId canonicalOrIdAtScope(Memory& mb, NameId origId,
            StrSpan validityName);

public:
/// @brief Test whether an equivalent OR already received processing.
        ///
        /// @details The input has the deposit door's equality-canonical spelling.
        /// Its compiled digit-pair orientation is looked up without minting and
        /// compared with each processing id in the exact scope's ledger. Original
        /// ids also match verbatim retries. Other scopes remain independent.
        /// @param mb Owning resident block.
        /// @param expr Equality-canonical positive numbered OR instance.
        /// @param validityName Exact scope being tested.
        /// @return Whether a recorded representative supplies OR processing.
        /// @invariant Does not mint or modify the ledger or proof state.
        /// @see canonicalOrProcessingForm; recordProcessedOr.
bool orEquiRepresentativeRecorded(Memory& mb,
            StrSpan expr,
            StrSpan validityName);

public:
/// @brief Record an or compact as fully processed at a scope — the
        ///        ledger insert paired with the willDisintegrate branch.
        ///
        /// @details
        /// Inserts `packInt32Pair(canonicalOrIdAtScope(origId), origId)`
        /// into the scope's `processedOrLedger` run (sorted-unique, numeric
        /// order — the run is never observable). Called by real nested OR consumption and the
        /// completed deposit; repeated insertion is idempotent. The
        /// canonical half is computed under the classes present now and
        /// kept current by the class-change seam afterwards.
        ///
        /// @param mb     Owning LB.
        /// @param origId The or statement's NameMap id (door-canonical text).
        /// @param valId  The scope's NameMap id.
        /// @return (void)
        /// @see orEquiRepresentativeRecorded; clearProcessedOr;
        ///      recanonicalizeProcessedOrLedger.
void recordProcessedOr(Memory& mb, NameId origId, NameId valId);

public:
/// @brief Drop one or statement's row from the processed-or ledger
        ///        — the clear paired with every path that intentionally
        ///        re-runs an or consumption.
        ///
        /// @details
        /// Rebuilds the scope's run without the row whose ORIGINAL half is
        /// @p origId (`assignSet`; the run is small — the distinct
        /// fully-processed ors of one scope). Idempotent by contract:
        /// several park keys of one or each wake once, and each wake must
        /// leave the ledger cleared — an absent row is the defined
        /// already-cleared state, never a failure. Callers:
        /// `resetParkedOrStatementRegistries` (the revival un-know), the
        /// two ordis park hooks' re-mail loops (the mailed rewritten
        /// statement must re-enter the door as the representative, or the
        /// gate would swallow the canonical re-file), and the D-331 drain's
        /// loser retirement.
        ///
        /// @param mb     Owning LB.
        /// @param origId The or statement's original NameMap id.
        /// @param valId  The scope's NameMap id.
        /// @return (void)
        /// @see recordProcessedOr; orEquiRepresentativeRecorded.
void clearProcessedOr(Memory& mb, NameId origId, NameId valId);

public:
/// @brief Re-key one scope's processed-or ledger rows to the
        ///        scope's current classes — the class-change seam of I-219.
        ///
        /// @details
        /// For every row of @p valId's run, recomputes the canonical half
        /// from the original half (`canonicalOrIdAtScope`) and, when any
        /// half changed, replaces the run with the re-sorted pairs
        /// (`assignSet`; the original halves are untouched, so no two rows
        /// collide — two originals that now share a canonical form stay two
        /// rows, which is exactly what the D-331 drain groups). Called from
        /// `updateEquivalenceClasses` right after the merged class is
        /// stored and pushed to `changedClassesThisStep`, the one place a
        /// scope's class bucket changes; the cost is the run length times
        /// one canonicalization per class change, instead of one
        /// canonicalization per record per or deposit at the gate. A scope
        /// with no rows is the defined no-op. Heap-free: the rebuilt run
        /// rides a gen-scratch mark, the canonical texts a string-tier
        /// scope, both reclaimed on return; the scope name is copied into
        /// the arena before the first mint (I-3).
        ///
        /// @param mb    Owning LB.
        /// @param valId The scope whose class bucket just changed.
        /// @return (void)
        /// @invariant After the call every row's canonical half equals
        ///            `canonicalOrIdAtScope(originalHalf)` under the
        ///            scope's classes.
        /// @see recordProcessedOr; orEquiRepresentativeRecorded;
        ///      updateEquivalenceClasses.
void recanonicalizeProcessedOrLedger(Memory& mb, NameId valId,
            const PagedVector<NameId>* members = nullptr);
private:

CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
const CompiledDefinitionRegistry& definitions;
};
}
namespace gl {

/// @brief Unpack the canonical half of a processed-or ledger row.
        ///
        /// @details
        /// A `Memory::processedOrLedger` value is `packInt32Pair(canonicalId,
        /// originalId)`; this reads the high half — the or text's
        /// equality-and-symmetry processing id under the scope's classes as of the
        /// last `recanonicalizeProcessedOrLedger` at that scope.
        ///
        /// @param row A ledger run value.
        /// @return The canonical NameMap id (never 0 for a recorded row).
        /// @see ledgerOriginalId; recordProcessedOr.
inline NameId CaseLedger::ledgerCanonicalId(int64_t row) {
            return static_cast<NameId>(static_cast<uint64_t>(row) >> 32);
        }

/// @brief Unpack the original half of a processed-or ledger row.
        ///
        /// @details
        /// The low half of `packInt32Pair(canonicalId, originalId)` — the
        /// door-canonical text recorded when the or received full
        /// processing; the cohort signature every clear path and the D-331
        /// drain key on.
        ///
        /// @param row A ledger run value.
        /// @return The original NameMap id (never 0 for a recorded row).
        /// @see ledgerCanonicalId; clearProcessedOr.
inline NameId CaseLedger::ledgerOriginalId(int64_t row) {
            return static_cast<NameId>(
                static_cast<uint64_t>(row) & 0xffffffffu);
        }

/// @brief Record original and canonical token dependencies of an OR marker.
        /// @details The logical owner remains the original expression and exact
        /// scope. Replacing the marker updates only that owner's posting edges.
        /// @param mb Owning block.
        /// @param original Original OR expression identifier.
        /// @param canonical Current canonical processing identifier.
        /// @param scope Exact marker scope.
        /// @param arena Caller-owned page-tier scratch.
        /// @return Nothing.
inline void CaseLedger::indexCanonicalOrMarker(Memory& mb, NameId original,
                NameId canonical, NameId scope, ScratchArena& arena) {
            auto& index = mb.lbMemory.canonicalUses.processedOr;
            const int64_t key = packStatementKey(original, scope);
            index.erase(key, arena);
            canonicalTemplates.indexCanonicalText(index, key, scope, mb.nameMap.decodeView(original));
            canonicalTemplates.indexCanonicalText(index, key, scope, mb.nameMap.decodeView(canonical));
        }

/// @brief Initialize OR-marker dependencies once from authoritative runs.
        /// @details Marker writers maintain later additions and replacements;
        /// scope cleanup retires closed entries; canonical release clears all state.
        /// @param mb Owning block.
        /// @param arena Caller-owned page-tier scratch.
        /// @return Nothing.
inline void CaseLedger::ensureCanonicalOrMarkerUses(Memory& mb, ScratchArena& arena) {
            auto& uses = mb.lbMemory.canonicalUses;
            if (uses.orLedgersBuilt) return;
            const auto add = [&](const auto& ledger) {
                for (int32_t i = 1; i <= ledger.count(); ++i) {
                    const NameId scope = ledger.keyAt(i);
                    for (int32_t j = 0; j < ledger.runLen(i); ++j) {
                        const int64_t pair = ledger.valueAt(i, j);
                        indexCanonicalOrMarker(mb,
                            static_cast<NameId>(uint32_t(pair)),
                            static_cast<NameId>(uint64_t(pair) >> 32), scope, arena);
                    }
                }
            };
            add(mb.processedOrLedger);
            uses.orLedgersBuilt = true;
        }
}
