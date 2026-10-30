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
#include "../compiler.hpp"
#include "analysis_node.hpp"

namespace gl {
/// @brief Analyze definition types and recognize set definitions.
/// @details Reads configured types without accessing prover state.
class ArgumentAnalyzer {
public:
/// @brief Bind source definitions for argument analysis.
/// @details The read-only map outlives this analyzer.
/// @param config Source definition configurations.
/// @return Constructed argument analyzer.
explicit ArgumentAnalyzer(const ce::CoreExpressionMap& config);
/// @brief Analyze the free argument types of an expression.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param rawExpr Expression text; formatting whitespace is ignored.
/// @return An owned analysis tree, or null for empty input.
AnalysisNode* analyze(const std::string& rawExpr);
/// @brief Check free argument types against a configured signature.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param node Analyzed expression, or null for absent input.
/// @param cfg Declared argument types.
/// @param callSignature Expression signature supplying argument names.
/// @return None.
static void checkDefinitionConsistency(const AnalysisNode* node,
            const ce::CoreExpressionConfig& cfg,
            const std::string& callSignature);
/// @brief Recognize a mirrored implication pair defining a set.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param root Analyzed expression tree.
/// @param outSetArg Receives the defined set argument.
/// @param outDefSet Receives its definition-set label.
/// @return Whether the tree defines a set without self-reference.
bool isSetDefinition(const AnalysisNode* root, std::string& outSetArg, std::string& outDefSet) const;

private:
/// @brief Compare two analysis trees structurally.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param a First tree, or null.
/// @param b Second tree, or null.
/// @return Whether both trees have identical values and children.
static bool areTreesEqual(const AnalysisNode* a, const AnalysisNode* b);
/// @brief Recognize a membership leaf carrying a set argument.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param node Tree node to classify.
/// @param outSetArg Receives the set argument on a match.
/// @param outDefSet Receives the set definition label on a match.
/// @return Whether the leaf names a supported membership predicate with a set argument.
static bool isMembershipClause(const AnalysisNode* node, std::string& outSetArg, std::string& outDefSet);
const ce::CoreExpressionMap& coreConfig;
};
} // namespace gl
