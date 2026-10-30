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
#include "argument_analyzer.hpp"
#include "argument_type_parser.hpp"

namespace gl {
/// @brief Bind source definitions for argument analysis.
/// @details The read-only map outlives this analyzer.
/// @param config Source definition configurations.
/// @return Constructed argument analyzer.
ArgumentAnalyzer::ArgumentAnalyzer(const ce::CoreExpressionMap& config) : coreConfig(config) {}
/// @brief Analyze the free argument types of an expression.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param rawExpr Expression text; formatting whitespace is ignored.
/// @return An owned analysis tree, or null for empty input.
AnalysisNode* ArgumentAnalyzer::analyze(const std::string& rawExpr) {
            std::string cleanExpr;
            cleanExpr.reserve(rawExpr.size());
            for (char c : rawExpr) {
                if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                    cleanExpr.push_back(c);
                }
            }
            if (cleanExpr.empty()) return nullptr;
            ArgumentTypeParser parser(cleanExpr, coreConfig);
            return parser.parse();
        }
/// @brief Check free argument types against a configured signature.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param node Analyzed expression, or null for absent input.
/// @param cfg Declared argument types.
/// @param callSignature Expression signature supplying argument names.
/// @return None.
void ArgumentAnalyzer::checkDefinitionConsistency(const AnalysisNode* node,
            const ce::CoreExpressionConfig& cfg,
            const std::string& callSignature) {
            if (!node) return;

            // 1. Get Actual Argument Names from the call (e.g. "x", "y")
            std::vector<std::string> actualArgs = ce::getArgs(callSignature);

            // 2. Strict Count Check
            if (node->remainingArgsDefs.size() != cfg.definitionSets.size()) {
                std::cerr << "Definition argument count mismatch for " << callSignature << std::endl;
                assert(false && "Definition argument count mismatch");
            }

            // 3. Verify each configured definition set
            for (const auto& kv : cfg.definitionSets) {
                // Key is position string "1", "2"...
                int paramIndex = std::stoi(kv.first);
                const std::string& expectedType = kv.second.first;

                if (paramIndex < 1 || paramIndex > static_cast<int>(actualArgs.size())) {
                    assert(false && "Configuration index out of bounds for call signature");
                }

                std::string argName = actualArgs[paramIndex - 1];

                auto it = node->remainingArgsDefs.find(argName);
                if (it != node->remainingArgsDefs.end()) {
                    if (it->second != expectedType) {
                        std::cerr << "Type mismatch for arg " << argName
                            << ". Expected: " << expectedType
                            << " Found: " << it->second << std::endl;
                        assert(false && "Definition type mismatch");
                    }
                }
                else {
                    assert(false && "Configured argument not found in analysis");
                }
            }
        }
/// @brief Recognize a mirrored implication pair defining a set.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param root Analyzed expression tree.
/// @param outSetArg Receives the defined set argument.
/// @param outDefSet Receives its definition-set label.
/// @return Whether the tree defines a set without self-reference.
bool ArgumentAnalyzer::isSetDefinition(const AnalysisNode* root, std::string& outSetArg, std::string& outDefSet) const {
            if (root == nullptr) return false;

            // 1. Root must be AND (&)
            if (root->value != "&" && root->value != "(&") return false;

            AnalysisNode* leftBranch = root->left;
            AnalysisNode* rightBranch = root->right;

            if (!leftBranch || !rightBranch) return false;

            // 2. Both branches must be Implications (>)
            if (leftBranch->value.find('>') == std::string::npos) return false;
            if (rightBranch->value.find('>') == std::string::npos) return false;

            // 3. Extract components
            // Left:  C -> D
            AnalysisNode* C = leftBranch->left;
            AnalysisNode* D = leftBranch->right;

            // Right: D' -> C'
            AnalysisNode* D_prime = rightBranch->left;
            AnalysisNode* C_prime = rightBranch->right;

            if (!C || !D || !D_prime || !C_prime) return false;

            // 4. Check Mirroring (C == C' and D == D')
            if (!areTreesEqual(C, C_prime)) return false;
            if (!areTreesEqual(D, D_prime)) return false;

            // 5. Find the Set Argument in either C or D
            // The membership clause determines the Set variable (outSetArg).
            // We then check the *other* side (the definition) to ensure outSetArg is NOT present.

            // Case A: C is the membership clause (e.g., x in S). D is the definition.
            if (isMembershipClause(C, outSetArg, outDefSet)) {
                // Ensure S does NOT appear in the definition D (Self-Reference check).
                // relying on remainingArgsDefs which aggregates used arguments.
                if (D->remainingArgsDefs.find(outSetArg) == D->remainingArgsDefs.end()) {
                    return true;
                }
            }

            // Case B: D is the membership clause. C is the definition.
            if (isMembershipClause(D, outSetArg, outDefSet)) {
                // Ensure S does NOT appear in the definition C.
                if (C->remainingArgsDefs.find(outSetArg) == C->remainingArgsDefs.end()) {
                    return true;
                }
            }

            return false;
        }
/// @brief Compare two analysis trees structurally.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param a First tree, or null.
/// @param b Second tree, or null.
/// @return Whether both trees have identical values and children.
bool ArgumentAnalyzer::areTreesEqual(const AnalysisNode* a, const AnalysisNode* b) {
            // Both null -> Equal
            if (a == nullptr && b == nullptr) return true;
            // One null -> Not Equal
            if (a == nullptr || b == nullptr) return false;

            // Compare operator/value string (e.g. ">[p]", "(&", "(in[p,M])")
            if (a->value != b->value) return false;

            // Recursively check children
            return areTreesEqual(a->left, b->left) && areTreesEqual(a->right, b->right);
        }
/// @brief Recognize a membership leaf carrying a set argument.
///
/// @details Preserves source traversal, binder scope and configured definition-set identity.
/// @param node Tree node to classify.
/// @param outSetArg Receives the set argument on a match.
/// @param outDefSet Receives the set definition label on a match.
/// @return Whether the leaf names a supported membership predicate with a set argument.
bool ArgumentAnalyzer::isMembershipClause(const AnalysisNode* node, std::string& outSetArg, std::string& outDefSet) {
            if (node == nullptr) return false;

            // Check if this is a leaf node that is a membership predicate
            // extractExpression removes parens and brackets: "(in[x,M])" -> "in"
            std::string coreName = ce::extractExpression(node->value);

            if (coreName == "in" || coreName == "in2" || coreName == "in3") {
                // Look through the arguments defined at this node to find one with Type "P(...)"
                for (const auto& kv : node->remainingArgsDefs) {
                    const std::string& argName = kv.first;
                    const std::string& defSet = kv.second;

                    // Check for "P(" prefix (Power Set)
                    if (defSet.rfind("P(", 0) == 0) {
                        outSetArg = argName;
                        outDefSet = defSet;
                        return true;
                    }
                }
            }
            return false;
        }
} // namespace gl
