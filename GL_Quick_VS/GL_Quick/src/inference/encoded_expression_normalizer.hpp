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
namespace tests { struct EncodedExpressionNormalizerFixture; }
/// @brief Encoded Expression Normalizer operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class EncodedExpressionNormalizer {
friend struct tests::EncodedExpressionNormalizerFixture;
public:
/// @brief Bind the encoded expression normalizer operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @return None; construction binds the supplied references.
EncodedExpressionNormalizer();

private:

inline std::set<std::string> getAllEncodedArgs(const std::vector<EncodedExpression>& expressionList);

private:

inline void setUnchangeables(std::vector<EncodedExpression>& expressionList,
            const std::set<std::string>& unchangeables);

private:

inline std::set<std::string> extractUnchangeables(const std::vector<EncodedExpression>& expressionList);

private:

inline void resetUnchangeables(std::vector<EncodedExpression>& expressionList,
            const std::set<std::string>& unchangeables);

private:
/// Hot path: builds NameId normalized key into caller's stack buffer.
        /// Returns length written. No heap allocation.
        /// ignoreU: if true, unchangeable vars get their NameMap ID with changeableStatus=1.
        ///          if false, all vars are normalized sequentially with changeableStatus=0.
inline NameId makeIntNormalizedKey(
            const std::vector<EncodedExpression>& lst,
            NameMap& nm,
            bool ignoreU,
            NameId* outBuf,
            NameId bufCapacity);

private:
/// Hot path variant: also produces a reverse map (normId -> original varId)
        /// for back-substitution when decoding heads.
inline NameId makeIntNormalizedKeyWithMap(
            const std::vector<EncodedExpression>& lst,
            NameMap& nm,
            bool ignoreU,
            NameId* outBuf,
            NameId bufCapacity,
            NameId* reverseMap,     // reverseMap[normId] = original varId
            NameId& numNormVars);

public:
/// @brief Comparator on `EncodedExpression::name` — used when sorting
    /// statements by their core expression name.
    /// @details Stable lexicographic order over `.name`. Used when the static
    /// pipeline wants name-grouped iteration; pairs with `lessByOriginal` for
    /// the alternative ordering.
static bool lessByName(const EncodedExpression& a, const EncodedExpression& b);

public:
/// @brief Comparator on `EncodedExpression::original` — used to sort by
    /// raw source text.
    /// @details Stable lexicographic order over `.original`, the un-rewritten
    /// expression string. Used by paths that need ordering by the
    /// user-visible form rather than by the internal name; pairs with
    /// `lessByName`.
static bool lessByOriginal(const EncodedExpression& a, const EncodedExpression& b);
private:

};
}
namespace gl {


inline std::set<std::string> EncodedExpressionNormalizer::getAllEncodedArgs(const std::vector<EncodedExpression>& expressionList) {
            std::set<std::string> allArgs;
            for (std::size_t i = 0; i < expressionList.size(); ++i) {
                const std::vector<std::vector<std::string> >& args = expressionList[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    if (args[j].size() >= 2) {
                        allArgs.insert(args[j][1]);
                    }
                }
            }
            return allArgs;
        }


inline void EncodedExpressionNormalizer::setUnchangeables(std::vector<EncodedExpression>& expressionList,
            const std::set<std::string>& unchangeables) {
            for (std::size_t i = 0; i < expressionList.size(); ++i) {
                std::vector<std::vector<std::string> >& args = expressionList[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    std::vector<std::string>& argument = args[j];
                    if (argument.size() >= 2) {
                        if (unchangeables.find(argument[1]) != unchangeables.end()) {
                            argument[0] = "True";
                        }
                    }
                }
            }
        }


inline std::set<std::string> EncodedExpressionNormalizer::extractUnchangeables(const std::vector<EncodedExpression>& expressionList) {
            std::set<std::string> unchangeables;
            for (std::size_t i = 0; i < expressionList.size(); ++i) {
                const std::vector<std::vector<std::string> >& args = expressionList[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::vector<std::string>& argument = args[j];
                    if (argument.size() >= 2 && argument[0] == "True") {
                        unchangeables.insert(argument[1]);
                    }
                }
            }
            return unchangeables;
        }


inline void EncodedExpressionNormalizer::resetUnchangeables(std::vector<EncodedExpression>& expressionList,
            const std::set<std::string>& unchangeables) {
            for (std::size_t i = 0; i < expressionList.size(); ++i) {
                std::vector<std::vector<std::string> >& args = expressionList[i].arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    std::vector<std::string>& argument = args[j];
                    if (argument.size() < 2) {
                        continue;
                    }
                    argument[0] = "False";
                    if (unchangeables.find(argument[1]) != unchangeables.end()) {
                        argument[0] = "True";
                    }
                }
            }
        }

/// Hot path: builds NameId normalized key into caller's stack buffer.
        /// Returns length written. No heap allocation.
        /// ignoreU: if true, unchangeable vars get their NameMap ID with changeableStatus=1.
        ///          if false, all vars are normalized sequentially with changeableStatus=0.
inline NameId EncodedExpressionNormalizer::makeIntNormalizedKey(
            const std::vector<EncodedExpression>& lst,
            NameMap& nm,
            bool ignoreU,
            NameId* outBuf,
            NameId bufCapacity) {

            // Small linear map instead of 8KB memset: keys have few distinct vars
            NameId varIds[ExecutionParameters::MAX_KEY_SLOTS];
            NameId normIds[ExecutionParameters::MAX_KEY_SLOTS];
            NameId nVars = 0;
            NameId nextNormId = 1;
            NameId pos = 0;

            for (std::size_t i = 0; i < lst.size(); ++i) {
                const EncodedExpression& expr = lst[i];

                // expressionID
                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = nm.encode(expr.name);
                // negation
                outBuf[pos++] = expr.negation ? 1 : 0;

                const std::vector<std::vector<std::string>>& args = expr.arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::vector<std::string>& arg = args[j];
                    if (arg.size() < 2) continue;

                    assert(pos + 2 <= bufCapacity);
                    bool isUnchangeable = (arg[0] == "True");
                    NameId varId = (!ignoreU && isUnchangeable)
                        ? nm.encode("u_" + arg[1])
                        : nm.encode(arg[1]);
                    assert(varId < ExecutionParameters::MAX_NAME_IDS);

                    if (ignoreU && isUnchangeable) {
                        outBuf[pos++] = varId;
                        outBuf[pos++] = 1;
                    }
                    else {
                        // Linear scan for normalization (nVars typically < 20)
                        NameId normId = 0;
                        for (NameId v = 0; v < nVars; ++v) {
                            if (varIds[v] == varId) { normId = normIds[v]; break; }
                        }
                        if (normId == 0) {
                            normId = nextNormId++;
                            varIds[nVars] = varId;
                            normIds[nVars] = normId;
                            ++nVars;
                        }
                        outBuf[pos++] = normId;
                        outBuf[pos++] = 0;
                    }
                }
            }
            return pos;
        }

/// Hot path variant: also produces a reverse map (normId -> original varId)
        /// for back-substitution when decoding heads.
inline NameId EncodedExpressionNormalizer::makeIntNormalizedKeyWithMap(
            const std::vector<EncodedExpression>& lst,
            NameMap& nm,
            bool ignoreU,
            NameId* outBuf,
            NameId bufCapacity,
            NameId* reverseMap,     // reverseMap[normId] = original varId
            NameId& numNormVars) {

            // Small linear map instead of 8KB memset
            NameId varIdsL[ExecutionParameters::MAX_KEY_SLOTS];
            NameId normIdsL[ExecutionParameters::MAX_KEY_SLOTS];
            NameId nVars = 0;
            NameId nextNormId = 1;
            NameId pos = 0;

            for (std::size_t i = 0; i < lst.size(); ++i) {
                const EncodedExpression& expr = lst[i];

                assert(pos + 2 <= bufCapacity);
                outBuf[pos++] = nm.encode(expr.name);
                outBuf[pos++] = expr.negation ? 1 : 0;

                const std::vector<std::vector<std::string>>& args = expr.arguments;
                for (std::size_t j = 0; j < args.size(); ++j) {
                    const std::vector<std::string>& arg = args[j];
                    if (arg.size() < 2) continue;

                    assert(pos + 2 <= bufCapacity);
                    bool isUnchangeable = (arg[0] == "True");
                    NameId varId = (!ignoreU && isUnchangeable)
                        ? nm.encode("u_" + arg[1])
                        : nm.encode(arg[1]);
                    if (varId >= ExecutionParameters::MAX_NAME_IDS) {
                        std::cerr << "NAME ID OVERFLOW: varId=" << varId
                                  << " MAX=" << ExecutionParameters::MAX_NAME_IDS
                                  << " arg[1]=" << arg[1]
                                  << " isUnchangeable=" << isUnchangeable << std::endl;
                    }
                    assert(varId < ExecutionParameters::MAX_NAME_IDS);

                    if (ignoreU && isUnchangeable) {
                        outBuf[pos++] = varId;
                        outBuf[pos++] = 1;
                    }
                    else {
                        NameId normId = 0;
                        for (NameId v = 0; v < nVars; ++v) {
                            if (varIdsL[v] == varId) { normId = normIdsL[v]; break; }
                        }
                        if (normId == 0) {
                            normId = nextNormId++;
                            varIdsL[nVars] = varId;
                            normIdsL[nVars] = normId;
                            ++nVars;
                            reverseMap[normId] = varId;
                        }
                        outBuf[pos++] = normId;
                        outBuf[pos++] = 0;
                    }
                }
            }
            numNormVars = static_cast<NameId>(nextNormId - 1);
            return pos;
        }
}
