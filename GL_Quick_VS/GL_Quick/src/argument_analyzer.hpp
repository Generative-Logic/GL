/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschränkt)

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

#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

#include "create_expressions_shim.hpp"

namespace gl {

    struct AnalysisNode {
        std::string value; // Operator (e.g., ">", "&") or leaf expression (e.g. "(in[x,A])")
        AnalysisNode* left = nullptr;
        AnalysisNode* right = nullptr;
        std::map<std::string, std::string> remainingArgsDefs;

        AnalysisNode() {}

        ~AnalysisNode() {
            delete left;
            delete right;
        }
    };

    class ArgumentAnalyzer {
    private:
        const std::map<std::string, ce::CoreExpressionConfig>& coreConfig;

        std::map<std::string, std::string> mergeMaps(const std::map<std::string, std::string>& a,
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

        class RecursiveParser {
            const std::string& expr;
            size_t index;
            const ArgumentAnalyzer& parent;

        public:
            RecursiveParser(const std::string& s, const ArgumentAnalyzer& p)
                : expr(s), index(0), parent(p) {
            }

            AnalysisNode* parse() {
                return parseSubtree();
            }

        private:
            AnalysisNode* parseSubtree() {
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

                        std::map<std::string, std::string> combined = parent.mergeMaps(
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

                        node->remainingArgsDefs = parent.mergeMaps(
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
                        std::map<std::string, std::string> combined = parent.mergeMaps(
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
                        node->remainingArgsDefs = parent.mergeMaps(
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

            void processLeaf(AnalysisNode* node, const std::string& label) {
                std::string coreExpr = ce::extractExpression(label);
                std::vector<std::string> args = ce::getArgs(label);

                auto it = parent.coreConfig.find(coreExpr);
                if (it != parent.coreConfig.end()) {
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
        };

    public:
        explicit ArgumentAnalyzer(const std::map<std::string, ce::CoreExpressionConfig>& config)
            : coreConfig(config) {
        }

        AnalysisNode* analyze(const std::string& rawExpr) {
            std::string cleanExpr;
            cleanExpr.reserve(rawExpr.size());
            for (char c : rawExpr) {
                if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                    cleanExpr.push_back(c);
                }
            }
            if (cleanExpr.empty()) return nullptr;
            RecursiveParser parser(cleanExpr, *this);
            return parser.parse();
        }

        // ---------------------------------------------------------------------------------
        // NEW FUNCTIONALITY: Tree Equality and Set Definition Recognition
        // ---------------------------------------------------------------------------------

        // Recursively compares two trees for strict structural equality (values and children).
        // This confirms that the right branch is a true mirror of the left branch.
        static bool areTreesEqual(const AnalysisNode* a, const AnalysisNode* b) {
            // Both null -> Equal
            if (a == nullptr && b == nullptr) return true;
            // One null -> Not Equal
            if (a == nullptr || b == nullptr) return false;

            // Compare operator/value string (e.g. ">[p]", "(&", "(in[p,M])")
            if (a->value != b->value) return false;

            // Recursively check children
            return areTreesEqual(a->left, b->left) && areTreesEqual(a->right, b->right);
        }

        // Checks if a node represents a membership clause (in, in2, in3) involving a Set argument.
        // A "Set argument" is defined as having a definition set starting with "P(".
        // Returns true if found, and populates output params.
        static bool isMembershipClause(const AnalysisNode* node, std::string& outSetArg, std::string& outDefSet) {
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

        // ---------------------------------------------------------------------------------
        // NEW FUNCTIONALITY: Definition Consistency Check
        // ---------------------------------------------------------------------------------

        // Verifies that the node's remaining arguments (free variables) exactly match 
        // the definition sets specified in the configuration.
        static void checkDefinitionConsistency(const AnalysisNode* node,
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

        //#pragma optimize("", off)

        // Determines if the parsed tree represents a Set Definition:
        // Structure: (& (>[x] C D) (>[x] D C))
        // Where either C or D is a membership clause for a Set variable.
        // AND the Set variable must NOT appear in the definition body (No Self-Reference).
        bool isSetDefinition(const AnalysisNode* root, std::string& outSetArg, std::string& outDefSet) const {
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
    };

} // namespace gl