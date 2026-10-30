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
#include "argument_type_parser.hpp"

namespace gl {
/// @brief Bind an expression and its definition configurations.
/// @details Both inputs outlive parsing; the cursor begins at the first character.
/// @param expression Expression text to parse.
/// @param config Source definition configurations.
/// @return Constructed argument-type parser.
ArgumentTypeParser::ArgumentTypeParser(const std::string& expression, const ce::CoreExpressionMap& config) : expr(expression), index(0), coreConfig(config) {}
/// @brief Parse one expression and propagate its argument types.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @return An owned analysis tree.
AnalysisNode* ArgumentTypeParser::parse() {
                return parseSubtree();
            }
/// @brief Parse the next subtree and remove locally bound arguments.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @return The owned subtree beginning at the current cursor.
AnalysisNode* ArgumentTypeParser::parseSubtree() {
                assert(index < expr.size() && "Unexpected end of expression");

                AnalysisNode* node = new AnalysisNode();
                std::string nodeLabel;

                if (expr[index] == '(') {
                    index++;

                    // 1.1 Implication
                    if (index < expr.size() && expr[index] == '>') {
                        index++;
                        nodeLabel += '>';

                        std::string argsPart = expr.substr(index);
                        std::vector<std::string> boundVars = ce::getArgs(argsPart);

                        size_t closeBracket = expr.find(']', index);
                        assert(closeBracket != std::string::npos && "Missing ']' in implication");

                        nodeLabel += expr.substr(index, closeBracket - index + 1);
                        index = closeBracket + 1;

                        node->left = parseSubtree();
                        node->right = parseSubtree();

                        std::map<std::string, std::string> combined = mergeMaps(
                            node->left ? node->left->remainingArgsDefs : std::map<std::string, std::string>{},
                            node->right ? node->right->remainingArgsDefs : std::map<std::string, std::string>{}
                        );

                        for (const auto& var : boundVars) {
                            combined.erase(var);
                        }
                        node->remainingArgsDefs = combined;
                    }
                    // 1.2 Conjunction
                    else if (index < expr.size() && expr[index] == '&') {
                        index++;
                        nodeLabel += '&';

                        node->left = parseSubtree();
                        node->right = parseSubtree();

                        node->remainingArgsDefs = mergeMaps(
                            node->left ? node->left->remainingArgsDefs : std::map<std::string, std::string>{},
                            node->right ? node->right->remainingArgsDefs : std::map<std::string, std::string>{}
                        );
                    }
                    // 1.3 Leaf
                    else {
                        size_t endIndex = expr.find(')', index);
                        assert(endIndex != std::string::npos && "Missing ')' for leaf");

                        nodeLabel = expr.substr(index, endIndex - index);
                        index = endIndex;

                        processLeaf(node, nodeLabel);
                    }
                }
                // 2. Negation
                else if (expr.substr(index, 2) == "!(") {
                    index += 2;

                    // 2.1 Negated Implication (!>)
                    if (index < expr.size() && expr[index] == '>') {
                        index++;
                        nodeLabel += "!>";

                        // Extract bound variables (Essential for remainingArgsDefs!)
                        std::string argsPart = expr.substr(index);
                        std::vector<std::string> boundVars = ce::getArgs(argsPart);

                        size_t closeBracket = expr.find(']', index);
                        assert(closeBracket != std::string::npos && "Missing ']' in negated implication");

                        nodeLabel += expr.substr(index, closeBracket - index + 1);
                        index = closeBracket + 1;

                        node->left = parseSubtree();
                        node->right = parseSubtree();

                        // Merge maps from children
                        std::map<std::string, std::string> combined = mergeMaps(
                            node->left ? node->left->remainingArgsDefs : std::map<std::string, std::string>{},
                            node->right ? node->right->remainingArgsDefs : std::map<std::string, std::string>{}
                        );

                        // Remove bound variables from the result (Crucial fix for "0 remainingArgs")
                        for (const auto& var : boundVars) {
                            combined.erase(var);
                        }
                        node->remainingArgsDefs = combined;
                    }
                    // 2.2 Negated Conjunction (!&)
                    else if (index < expr.size() && expr[index] == '&') {
                        index++;
                        nodeLabel += "!&";

                        node->left = parseSubtree();
                        node->right = parseSubtree();

                        // Merge maps from children (Crucial fix)
                        node->remainingArgsDefs = mergeMaps(
                            node->left ? node->left->remainingArgsDefs : std::map<std::string, std::string>{},
                            node->right ? node->right->remainingArgsDefs : std::map<std::string, std::string>{}
                        );
                    }
                    // 2.3 Negated Leaf
                    else {
                        size_t endIndex = expr.find(')', index);
                        assert(endIndex != std::string::npos && "Missing ')' for negated leaf");

                        std::string innerLabel = expr.substr(index, endIndex - index);
                        nodeLabel = "!(" + innerLabel + ")";
                        index = endIndex;
                        processLeaf(node, innerLabel);
                    }
                }

                if (index < expr.size() && expr[index] == ')') {
                    index++;
                }

                node->value = nodeLabel;
                return node;
            }
/// @brief Assign the configured definition sets to one leaf.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param node Leaf receiving its argument types.
/// @param label Leaf expression text.
/// @return None.
void ArgumentTypeParser::processLeaf(AnalysisNode* node, const std::string& label) {
                std::string coreExpr = ce::extractExpression(label);
                std::vector<std::string> args = ce::getArgs(label);

                auto it = coreConfig.find(coreExpr);
                if (it != coreConfig.end()) {
                    const ce::CoreExpressionConfig& cfg = it->second;
                    for (size_t i = 0; i < args.size(); ++i) {
                        std::string argName = args[i];
                        std::string paramIndex = std::to_string(i + 1);

                        auto defIt = cfg.definitionSets.find(paramIndex);
                        if (defIt != cfg.definitionSets.end()) {
                            node->remainingArgsDefs[argName] = defIt->second.first;
                        }
                    }
                }
            }
/// @brief Combine argument types while asserting shared-variable consistency.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param a First argument-type map.
/// @param b Second argument-type map.
/// @return The combined argument-type map.
std::map<std::string, std::string> ArgumentTypeParser::mergeMaps(const std::map<std::string, std::string>& a,
            const std::map<std::string, std::string>& b) const {
            std::map<std::string, std::string> result = a;
            for (const auto& kv : b) {
                auto it = result.find(kv.first);
                if (it != result.end()) {
                    // If the argument exists in both branches, the definition sets MUST match.
                    assert(it->second == kv.second && "Definition set mismatch for shared argument");
                }
                else {
                    result.insert(kv);
                }
            }
            return result;
        }
} // namespace gl
