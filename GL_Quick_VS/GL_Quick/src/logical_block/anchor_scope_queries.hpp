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
namespace tests { struct AnchorScopeQueriesFixture; }
/// @brief Anchor Scope Queries operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class AnchorScopeQueries {
friend struct tests::AnchorScopeQueriesFixture;
public:
/// @brief Bind inputs for anchor scope queries.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param definitionExpansion Borrowed DefinitionExpansion interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
AnchorScopeQueries(DefinitionExpansion& definitionExpansion);

private:

inline std::string extractNaturalNumbersExpression(const std::string& expression);

public:

inline std::string findZeroArgNameFromAnchor(const std::string& anchorExpression);

private:
// Returns the anchor argument that names `N` (the type-ground set).
        // Same extraction path as findZeroArgNameFromAnchor but takes args[0]
        // of the NaturalNumbers sub-expression inside the expanded anchor MPL.
        // Used by the induction-typing sub-block (see docs/agentic_swdd/induction_typing_plan.md)
        // to form the sub-goal `(in[digitArg,N])` before scheduling induction.
inline std::string findNArgNameFromAnchor(const std::string& anchorExpression);

public:

inline std::string findSArgNameFromAnchor(const std::string& anchorExpression);

public:
/// @brief The anchor argument that names addition.
///
/// @details
/// Expands the anchor and reads the fourth argument of its
/// `NaturalNumbers[N,i0,s,+,*]` sub-expression — the same extraction path as
/// `findZeroArgNameFromAnchor` (second argument) and `findSArgNameFromAnchor`
/// (third argument). The strong-induction hypothesis writes its bound
/// `m<n` as the atomic `(in3[m,x,n,+])`, so it needs the anchor's own name for
/// addition.
///
/// @param anchorExpression The anchor expression as it occurs in a conjecture,
///                         for example `(AnchorFTA[1,2,3,4,5,6,7,8])`.
/// @return The anchor argument in the addition slot of `NaturalNumbers`.
/// @invariant The expanded anchor contains a `NaturalNumbers` expression with
///            at least four arguments (asserted).
/// @see findZeroArgNameFromAnchor, findSArgNameFromAnchor,
///      ProofGridBuilder::createStrongAuxyImplication.
inline std::string findPlusArgNameFromAnchor(const std::string& anchorExpression);

public:
/// @brief The anchor argument that names multiplication.
///
/// @details
/// Expands the anchor and reads the fifth argument of its
/// `NaturalNumbers[N,i0,s,+,*]` sub-expression, the extraction path of
/// `findPlusArgNameFromAnchor`. The strong step's multiplication catcher
/// writes its bound premise `x·m = n` as the atomic `(in3[x,m,n,*])`, so it
/// needs the anchor's own name for multiplication
/// (D-566).
///
/// @param anchorExpression The anchor expression as it occurs in a conjecture,
///                         for example `(AnchorFTA[1,2,3,4,5,6,7,8])`.
/// @return The anchor argument in the multiplication slot of `NaturalNumbers`.
/// @invariant The expanded anchor contains a `NaturalNumbers` expression with
///            at least five arguments (asserted).
/// @see findPlusArgNameFromAnchor, ProofGridBuilder::createStrongAuxyImplication.
inline std::string findTimesArgNameFromAnchor(const std::string& anchorExpression);

public:

inline std::string findZeroArgName(Memory& memoryBlock);

public:

inline std::string findSName(Memory& memoryBlock);

public:
// Chain-walks up to the anchor LB and returns the anchor's N slot
        // (slot 0 — the type-ground set). Used by the induction-typing
        // sub-block setup to form `(in[digitArg,N])` as the typing goal.
inline std::string findNName(Memory& memoryBlock);
private:

DefinitionExpansion& definitionExpansion;
};
}
namespace gl {


inline std::string AnchorScopeQueries::extractNaturalNumbersExpression(const std::string& expression) {
            const std::string needle = "(NaturalNumbers[";
            const std::size_t start = expression.find(needle);
            if (start == std::string::npos) {
                return {};
            }

            const std::size_t end = expression.find(']', start + needle.size());
            if (end == std::string::npos) {
                return {};
            }

            return expression.substr(start, end - start + 1);
        }


inline std::string AnchorScopeQueries::findZeroArgNameFromAnchor(const std::string& anchorExpression) {
            std::string fullMPL = definitionExpansion.expandExpr(anchorExpression);

			std::string naturalNumbersExpression = extractNaturalNumbersExpression(fullMPL);
			std::vector<std::string> args = ce::getArgs(naturalNumbersExpression);

            return args[1];
        }

// Returns the anchor argument that names `N` (the type-ground set).
        // Same extraction path as findZeroArgNameFromAnchor but takes args[0]
        // of the NaturalNumbers sub-expression inside the expanded anchor MPL.
        // Used by the induction-typing sub-block (see docs/agentic_swdd/induction_typing_plan.md)
        // to form the sub-goal `(in[digitArg,N])` before scheduling induction.
inline std::string AnchorScopeQueries::findNArgNameFromAnchor(const std::string& anchorExpression) {
            std::string fullMPL = definitionExpansion.expandExpr(anchorExpression);

            std::string naturalNumbersExpression = extractNaturalNumbersExpression(fullMPL);
            std::vector<std::string> args = ce::getArgs(naturalNumbersExpression);

            return args[0];
        }


inline std::string AnchorScopeQueries::findSArgNameFromAnchor(const std::string& anchorExpression) {
            std::string fullMPL = definitionExpansion.expandExpr(anchorExpression);

            std::string naturalNumbersExpression = extractNaturalNumbersExpression(fullMPL);
            std::vector<std::string> args = ce::getArgs(naturalNumbersExpression);

            return args[2];
        }


/// @brief The anchor argument that names addition.
///
/// @details
/// Expands the anchor and reads the fourth argument of its
/// `NaturalNumbers[N,i0,s,+,*]` sub-expression — the same extraction path as
/// `findZeroArgNameFromAnchor` (second argument) and `findSArgNameFromAnchor`
/// (third argument). The strong-induction hypothesis writes its bound
/// `m<n` as the atomic `(in3[m,x,n,+])`, so it needs the anchor's own name for
/// addition.
///
/// @param anchorExpression The anchor expression as it occurs in a conjecture,
///                         for example `(AnchorFTA[1,2,3,4,5,6,7,8])`.
/// @return The anchor argument in the addition slot of `NaturalNumbers`.
/// @invariant The expanded anchor contains a `NaturalNumbers` expression with
///            at least four arguments (asserted).
/// @see findZeroArgNameFromAnchor, findSArgNameFromAnchor,
///      ProofGridBuilder::createStrongAuxyImplication.
inline std::string AnchorScopeQueries::findPlusArgNameFromAnchor(const std::string& anchorExpression) {
            const std::string fullMPL = definitionExpansion.expandExpr(anchorExpression);
            const std::string naturalNumbersExpression = extractNaturalNumbersExpression(fullMPL);
            assert(!naturalNumbersExpression.empty()
                && "findPlusArgNameFromAnchor: the anchor names no NaturalNumbers expression");
            const std::vector<std::string> args = ce::getArgs(naturalNumbersExpression);
            assert(args.size() >= 4
                && "findPlusArgNameFromAnchor: NaturalNumbers has no addition slot");
            return args[3];
        }


/// @brief The anchor argument that names multiplication.
///
/// @details
/// Expands the anchor and reads the fifth argument of its
/// `NaturalNumbers[N,i0,s,+,*]` sub-expression, the extraction path of
/// `findPlusArgNameFromAnchor`. The strong step's multiplication catcher
/// writes its bound premise `x·m = n` as the atomic `(in3[x,m,n,*])`, so it
/// needs the anchor's own name for multiplication
/// (D-566).
///
/// @param anchorExpression The anchor expression as it occurs in a conjecture,
///                         for example `(AnchorFTA[1,2,3,4,5,6,7,8])`.
/// @return The anchor argument in the multiplication slot of `NaturalNumbers`.
/// @invariant The expanded anchor contains a `NaturalNumbers` expression with
///            at least five arguments (asserted).
/// @see findPlusArgNameFromAnchor, ProofGridBuilder::createStrongAuxyImplication.
inline std::string AnchorScopeQueries::findTimesArgNameFromAnchor(const std::string& anchorExpression) {
            const std::string fullMPL = definitionExpansion.expandExpr(anchorExpression);
            const std::string naturalNumbersExpression = extractNaturalNumbersExpression(fullMPL);
            assert(!naturalNumbersExpression.empty()
                && "findTimesArgNameFromAnchor: the anchor names no NaturalNumbers expression");
            const std::vector<std::string> args = ce::getArgs(naturalNumbersExpression);
            assert(args.size() >= 5
                && "findTimesArgNameFromAnchor: NaturalNumbers has no multiplication slot");
            return args[4];
        }


inline std::string AnchorScopeQueries::findZeroArgName(Memory& memoryBlock) {
            Memory* current = &memoryBlock;
            while (true) {
                if (!current->exprKey().empty() &&
                    current->exprKey().find("(Anchor") == 0) { // starts with

                    std::string zeroArg = findZeroArgNameFromAnchor(current->exprKey());
                    return zeroArg;
                }
                else if (current->parentMemory != NULL) {
                    current = current->parentMemory;
                }
                else {
                    assert(false); // not found up the chain
                }
            }
            // Unreachable, but keeps some compilers happy:
            return std::string();
        }


inline std::string AnchorScopeQueries::findSName(Memory& memoryBlock) {
            Memory* current = &memoryBlock;
            while (true) {
                if (!current->exprKey().empty() &&
                    current->exprKey().find("(Anchor") == 0) { // starts with

                    std::string sArg = findSArgNameFromAnchor(current->exprKey());
                    return sArg;
                }
                else if (current->parentMemory != NULL) {
                    current = current->parentMemory;
                }
                else {
                    assert(false); // not found up the chain
                }
            }
            // Unreachable, but keeps some compilers happy:
            return std::string();
        }

// Chain-walks up to the anchor LB and returns the anchor's N slot
        // (slot 0 — the type-ground set). Used by the induction-typing
        // sub-block setup to form `(in[digitArg,N])` as the typing goal.
inline std::string AnchorScopeQueries::findNName(Memory& memoryBlock) {
            Memory* current = &memoryBlock;
            while (true) {
                if (!current->exprKey().empty() &&
                    current->exprKey().find("(Anchor") == 0) {
                    return findNArgNameFromAnchor(current->exprKey());
                }
                else if (current->parentMemory != NULL) {
                    current = current->parentMemory;
                }
                else {
                    assert(false);
                }
            }
            return std::string();
        }
}

namespace gl {
/// @brief Bind inputs for anchor scope queries.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param definitionExpansion Borrowed DefinitionExpansion interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
inline AnchorScopeQueries::AnchorScopeQueries(DefinitionExpansion& definitionExpansion) : definitionExpansion(definitionExpansion) {}
} // namespace gl
