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

namespace gl {

/// @brief MPL syntax operations used by definition compilation.
/// @details Owns parsing and structural text transformations without prover state.
class ExpressionParser {
public:

/// @brief Parse one MPL expression into its expression tree.
///
/// @details Uses whitespace removal, node labels, argument sets and recursive parse order. The caller owns the returned tree.
/// @param treeStrIn Expression text to parse.
/// @return Owned expression tree.
static ce::TreeNode1* parse(const std::string& treeStrIn);

/// @brief Expand conjunction premises into the existing implication-chain form.
///
/// @details Repeats the established tree transformation until no conjunction premise remains, preserving binders and expression spelling.
/// @param expr Expression to transform.
/// @return Transformed expression.
static std::string smoothenExpr(const std::string& expr);

/// @brief Enumerate independent conclusion expressions.
///
/// @details Walks the expression in the existing left-to-right order and retains compound premises and negated structures.
/// @param treeStr Expression to inspect.
/// @return Ordered conclusion expressions.
static std::vector<std::string> groomExpr(const std::string& treeStr);

/// @brief Read the first implication binder argument list.
///
/// @details Returns the existing ordered binder spelling, including its established empty-list representation. Expressions without a binder have an empty result.
/// @param expr Expression to inspect.
/// @return Ordered bound arguments.
static std::vector<std::string> listLastRemovedArgs(const std::string& expr);

/// @brief Rename the single outer bound argument for compilation.
///
/// @details Preserves the compiler variable counter and token-aware substitution. The expression must have exactly one referenced outer bound argument.
/// @param expr Implication or negated implication.
/// @param startInt Next compiler variable number.
/// @return Rewritten expression, next counter and chosen variable name.
static std::tuple<std::string, int, std::string> renameLastRemoved(const std::string& expr, int startInt);

private:

/// @brief Traverse one expression while collecting conclusion alternatives.
///
/// @details Recursively preserves the caller-selected premise or conclusion role and advances the shared parse position.
/// @param s Expression text.
/// @param side Zero for a premise, one for a conclusion.
/// @param index Current parse position, advanced in place.
/// @return Ordered expression alternatives.
static std::vector<std::string> groomSubexpr(const std::string& s, int side, std::size_t& index);

/// @brief Append one tree using implication-chain premise formatting.
///
/// @details Retains the existing traversal and marks any conjunction-premise rewrite so the caller repeats normalization.
/// @param node Tree node, with null denoting an empty child.
/// @param out Destination expression text.
/// @param foundAnd Whether a conjunction premise was expanded.
/// @return None.
static void appendSmoothedTree(const ce::TreeNode1* node, std::string& out, bool& foundAnd);

/// @brief Parse the next MPL subtree.
///
/// @details Builds children and argument sets in the established recursive order and updates the shared input position.
/// @param s Whitespace-normalized expression.
/// @param index Current parse position, advanced in place.
/// @return Owned subtree or the defined empty-child representation.
static ce::TreeNode1* parseSubtree(const std::string& s, std::size_t& index);
};

} // namespace gl
