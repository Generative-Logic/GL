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
/// @brief Parse expressions into trees carrying free argument types.
/// @details Reads configured types without accessing prover state.
class ArgumentTypeParser {
public:
/// @brief Bind an expression and its definition configurations.
/// @details Both inputs outlive parsing; the cursor begins at the first character.
/// @param expression Expression text to parse.
/// @param config Source definition configurations.
/// @return Constructed argument-type parser.
explicit ArgumentTypeParser(const std::string& expression, const ce::CoreExpressionMap& config);
/// @brief Parse one expression and propagate its argument types.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @return An owned analysis tree.
AnalysisNode* parse();

private:
/// @brief Parse the next subtree and remove locally bound arguments.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @return The owned subtree beginning at the current cursor.
AnalysisNode* parseSubtree();
/// @brief Assign the configured definition sets to one leaf.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param node Leaf receiving its argument types.
/// @param label Leaf expression text.
/// @return None.
void processLeaf(AnalysisNode* node, const std::string& label);
/// @brief Combine argument types while asserting shared-variable consistency.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param a First argument-type map.
/// @param b Second argument-type map.
/// @return The combined argument-type map.
std::map<std::string, std::string> mergeMaps(const std::map<std::string, std::string>& a,
            const std::map<std::string, std::string>& b) const;
const std::string& expr;
std::size_t index;
const ce::CoreExpressionMap& coreConfig;
};
} // namespace gl
