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

#include "compiler.hpp"

namespace ce {

    std::vector<std::vector<int>> generateBinarySequencesAsLists(int n) {
        std::vector<std::vector<int>> out;

        if (n < 0) {
            return out;
        }
        if (n == 0) {
            out.emplace_back();
            return out;
        }

        const std::size_t total = static_cast<std::size_t>(1ULL) << n;
        out.reserve(total);

        for (std::size_t mask = 0; mask < total; ++mask) {
            std::vector<int> row;
            row.reserve(static_cast<std::size_t>(n));
            for (int i = n - 1; i >= 0; --i) {
                int bit = static_cast<int>((mask >> i) & 1ULL);
                row.push_back(bit);
            }
            out.push_back(std::move(row));
        }
        return out;
    }

    std::map<int, std::vector<std::vector<int>>> generateAllPermutations(int n) {
        std::map<int, std::vector<std::vector<int>>> allPermutations;

        if (n < 0) {
            return allPermutations;
        }

        allPermutations[0] = std::vector<std::vector<int>>();

        for (int i = 1; i <= n; ++i) {
            std::vector<int> current;
            current.reserve(static_cast<std::size_t>(i));
            for (int k = 0; k < i; ++k) {
                current.push_back(k);
            }

            std::vector<std::vector<int>> perms;
            do {
                perms.push_back(current);
            } while (std::next_permutation(current.begin(), current.end()));

            allPermutations[i] = perms;
        }

        return allPermutations;
    }

    TreeNode1* parseExpr(
        const std::string& treeStrIn) {

        std::string s;
        s.reserve(treeStrIn.size());
        for (std::size_t i = 0; i < treeStrIn.size(); ++i) {
            const char c = treeStrIn[i];
            if (c != '\n' && c != ' ' && c != '\t') {
                s.push_back(c);
            }
        }

        std::size_t index = 0;

        struct Parser {
            static TreeNode1* parseSubtree(
                const std::string& s,
                std::size_t& index) {

                if (s.empty()) {
                    throw std::runtime_error("Input 's' cannot be empty.");
                }

                TreeNode1* node = new TreeNode1("", 0);
                std::string nodeLabel;

                if (index < s.size() && s[index] == '(') {
                    ++index;

                    if (index < s.size() && s[index] == '>') {
                        ++index;
                        nodeLabel.push_back('>');

                        std::vector<std::string> argsToRemove = ce::getArgs(s.substr(index));
                        std::size_t close = s.find(']', index);
                        if (close == std::string::npos) {
                            delete node;
                            throw std::runtime_error("No closing ']' for '>' node.");
                        }
                        index = close + 1;

                        nodeLabel.push_back('[');
                        for (std::size_t ai = 0; ai < argsToRemove.size(); ++ai) {
                            if (ai > 0) nodeLabel.push_back(',');
                            nodeLabel += argsToRemove[ai];
                        }
                        nodeLabel.push_back(']');

                        TreeNode1* leftRes = parseSubtree(s, index);
                        TreeNode1* rightRes = parseSubtree(s, index);
                        node->left = leftRes;
                        node->right = rightRes;

                        if (node->left) {
                            node->arguments.insert(node->left->arguments.begin(), node->left->arguments.end());
                        }
                        if (node->right) {
                            node->arguments.insert(node->right->arguments.begin(), node->right->arguments.end());
                        }

                        std::vector<std::string> removeArgs = ce::getArgs(nodeLabel);
                        for (std::size_t ri = 0; ri < removeArgs.size(); ++ri) {
                            node->arguments.erase(removeArgs[ri]);
                        }

                    }
                    else if (index < s.size() && s[index] == '&') {
                        ++index;
                        nodeLabel.push_back('&');

                        TreeNode1* leftRes = parseSubtree(s, index);
                        TreeNode1* rightRes = parseSubtree(s, index);
                        node->left = leftRes;
                        node->right = rightRes;

                        if (node->left) {
                            node->arguments.insert(node->left->arguments.begin(), node->left->arguments.end());
                        }
                        if (node->right) {
                            node->arguments.insert(node->right->arguments.begin(), node->right->arguments.end());
                        }

                    }
                    else {
                        std::size_t endIndex = s.find(')', index);
                        if (endIndex == std::string::npos) {
                            delete node;
                            throw std::runtime_error("No closing ')'.");
                        }
                        nodeLabel = s.substr(index, endIndex - index);
                        index = endIndex;

                        const std::string expr = ce::extractExpression(nodeLabel);

                        std::vector<std::string> args = ce::getArgs(nodeLabel);
                        for (std::size_t i = 0; i < args.size(); ++i) {
                            node->arguments.insert(args[i]);
                        }
                    }

                }
                else if (index + 1 < s.size() && s[index] == '!' && s[index + 1] == '(') {
                    index += 2;

                    if (index < s.size() && s[index] == '>') {
                        ++index;
                        nodeLabel += "!>";

                        std::vector<std::string> argsToRemove = ce::getArgs(s.substr(index));
                        std::size_t close = s.find(']', index);
                        if (close == std::string::npos) {
                            delete node;
                            throw std::runtime_error("No closing ']' for '!>' node.");
                        }
                        index = close + 1;

                        nodeLabel.push_back('[');
                        for (std::size_t ai = 0; ai < argsToRemove.size(); ++ai) {
                            if (ai > 0) nodeLabel.push_back(',');
                            nodeLabel += argsToRemove[ai];
                        }
                        nodeLabel.push_back(']');

                        TreeNode1*leftRes = parseSubtree(s, index);
                        TreeNode1* rightRes = parseSubtree(s, index);
                        node->left = leftRes;
                        node->right = rightRes;

                        if (node->left) {
                            node->arguments.insert(node->left->arguments.begin(), node->left->arguments.end());
                        }
                        if (node->right) {
                            node->arguments.insert(node->right->arguments.begin(), node->right->arguments.end());
                        }

                        std::vector<std::string> removeArgs = ce::getArgs(nodeLabel);
                        for (std::size_t ri = 0; ri < removeArgs.size(); ++ri) {
                            node->arguments.erase(removeArgs[ri]);
                        }

                    }
                    else if (index < s.size() && s[index] == '&') {
                        ++index;
                        nodeLabel += "!&";

                        TreeNode1* leftRes = parseSubtree(s, index);
                        TreeNode1* rightRes = parseSubtree(s, index);
                        node->left = leftRes;
                        node->right = rightRes;

                        if (node->left) {
                            node->arguments.insert(node->left->arguments.begin(), node->left->arguments.end());
                        }
                        if (node->right) {
                            node->arguments.insert(node->right->arguments.begin(), node->right->arguments.end());
                        }

                    }
                    else {
                        std::size_t endIndex = s.find(')', index);
                        if (endIndex == std::string::npos) {
                            delete node;
                            throw std::runtime_error("No closing ')'.");
                        }
                        nodeLabel = s.substr(index, endIndex - index);
                        nodeLabel = "!(" + nodeLabel + ")";
                        index = endIndex;

                        const std::string expr = ce::extractExpressionFromNegation(nodeLabel);

                        std::vector<std::string> args = ce::getArgs(nodeLabel);
                        for (std::size_t i = 0; i < args.size(); ++i) {
                            node->arguments.insert(args[i]);
                        }
                    }

                }
                else if (index < s.size() && s[index] == ')') {
                    index = index - 1;
                }

                ++index;
                node->value = nodeLabel;
                if (node->value.empty()) {
                    delete node;
                    node = NULL;
                }
                return node;
            }
        };

        TreeNode1* result = Parser::parseSubtree(s, index);
        return result;
    }

} // namespace ce
