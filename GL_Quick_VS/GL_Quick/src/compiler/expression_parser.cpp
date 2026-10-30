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
#include "expression_parser.hpp"

namespace gl {
using ce::TreeNode1;

/// @brief Parse one MPL expression into its expression tree.
///
/// @details Uses whitespace removal, node labels, argument sets and recursive parse order. The caller owns the returned tree.
/// @param treeStrIn Expression text to parse.
/// @return Owned expression tree.
ce::TreeNode1* ExpressionParser::parse(
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



        TreeNode1* result = parseSubtree(s, index);
        return result;
    }

/// @brief Parse the next MPL subtree.
///
/// @details Builds children and argument sets in the established recursive order and updates the shared input position.
/// @param s Whitespace-normalized expression.
/// @param index Current parse position, advanced in place.
/// @return Owned subtree or the defined empty-child representation.
/// @invariant A conjunction `(&..)` / `!(&..)` has exactly two children
///            (asserted); a longer conjunction nests, left or right.
ce::TreeNode1* ExpressionParser::parseSubtree(const std::string& s, std::size_t& index) {

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
                        assert(leftRes != NULL && rightRes != NULL && index < s.size() && s[index] == ')'
                            && "ExpressionParser::parseSubtree: a conjunction has exactly two children");
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
                        assert(leftRes != NULL && rightRes != NULL && index < s.size() && s[index] == ')'
                            && "ExpressionParser::parseSubtree: a conjunction has exactly two children");
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

/// @brief Append one tree using implication-chain premise formatting.
///
/// @details Retains the existing traversal and marks any conjunction-premise rewrite so the caller repeats normalization.
/// @param node Tree node, with null denoting an empty child.
/// @param out Destination expression text.
/// @param foundAnd Whether a conjunction premise was expanded.
/// @return None.
void ExpressionParser::appendSmoothedTree(const ce::TreeNode1* node, std::string& out, bool& foundAnd) {
            if (node == NULL) {
                return;
            }
            const std::string& v = node->value;

            if (!v.empty() && v[0] == '>') {
                if (node->left != NULL && node->left->value == "&") {
                    foundAnd = true;

                    std::string leftLeftExpr = ce::treeToExpr(node->left->left);
                    std::string leftRightExpr = ce::treeToExpr(node->left->right);
                    std::string rightExpr = ce::treeToExpr(node->right);

                    std::string tempExpr = "(";
                    tempExpr += ">[]";
                    tempExpr += leftRightExpr;
                    tempExpr += rightExpr;
                    tempExpr += ")";

                    out += "(";
                    out += v;
                    out += leftLeftExpr;
                    out += tempExpr;
                    // fall through to the common ')' append below; no recursion on children in this case
                }
                else {
                    out += "(";
                    out += v;
                    appendSmoothedTree(node->left, out, foundAnd);
                    appendSmoothedTree(node->right, out, foundAnd);
                }
            }
            else if (v == "&") {
                out += "(&";
                appendSmoothedTree(node->left, out, foundAnd);
                appendSmoothedTree(node->right, out, foundAnd);
            }
            else if (v.size() >= 2 && v[0] == '!' && v[1] == '>') {
                out += "!(";
                out += v.substr(1); // strip leading '!' from value like "!>..."
                appendSmoothedTree(node->left, out, foundAnd);
                appendSmoothedTree(node->right, out, foundAnd);
            }
            else if (v == "!&") {
                out += "!(&";
                appendSmoothedTree(node->left, out, foundAnd);
                appendSmoothedTree(node->right, out, foundAnd);
            }
            else if (!v.empty() && v[0] == '!') {
                // value like "!( ... )" — take inside between "!(" and trailing ")"
                out += "!(";
                if (v.size() >= 3) {
                    out += v.substr(2, v.size() - 3);
                }
            }
            else {
                out += "(";
                out += v;
            }

            out.push_back(')');
        }

/// @brief Expand conjunction premises into the existing implication-chain form.
///
/// @details Repeats the established tree transformation until no conjunction premise remains, preserving binders and expression spelling.
/// @param expr Expression to transform.
/// @return Transformed expression.
std::string ExpressionParser::smoothenExpr(const std::string& expr) {


    bool found = true;
    std::string smoothenedExpr = expr;

    while (found) {
        // smoothen_one_and(smoothenedExpr)
        ce::TreeNode1* root = parse(smoothenedExpr);

        std::string localExpr;
        bool foundAnd = false;
        appendSmoothedTree(root, localExpr, foundAnd);

        ce::deleteTree(root);

        smoothenedExpr = localExpr;
        found = foundAnd;
    }

    return smoothenedExpr;
}

/// @brief Traverse one expression while collecting conclusion alternatives.
///
/// @details Recursively preserves the caller-selected premise or conclusion role and advances the shared parse position.
/// @param s Expression text.
/// @param side Zero for a premise, one for a conclusion.
/// @param index Current parse position, advanced in place.
/// @return Ordered expression alternatives.
std::vector<std::string> ExpressionParser::groomSubexpr(const std::string& s, int side, std::size_t& index) {
    std::vector<std::string> subexprList;

    if (s[index] == '(') {
        index += 1;
        if (index < s.size() && s[index] == '>') {
            index += 1;
            std::vector<std::string> argsToRemove = ce::getArgs(s.substr(index));
            std::size_t close = s.find(']', index);
            if (close == std::string::npos) {
                throw std::runtime_error("No closing ']'.");
            }
            index = close + 1;

            std::vector<std::string> left = groomSubexpr(s, 0, index);
            std::vector<std::string> right = groomSubexpr(s, side, index);

            std::string argsJoined = ce::joinWithComma(argsToRemove);
            for (std::size_t i = 0; i < left.size(); ++i) {
                for (std::size_t j = 0; j < right.size(); ++j) {
                    std::string t;
                    t.reserve(4 + argsJoined.size() + left[i].size() + right[j].size());
                    t += "(>[";
                    t += argsJoined;
                    t += "]";
                    t += left[i];
                    t += right[j];
                    t += ")";
                    subexprList.push_back(t);
                }
            }
        }
        else if (index < s.size() && s[index] == '&') {
            index += 1;

            std::vector<std::string> left = groomSubexpr(s, side, index);
            std::vector<std::string> right = groomSubexpr(s, side, index);

            if (side) {
                subexprList.reserve(left.size() + right.size());
                for (std::size_t i = 0; i < left.size(); ++i) subexprList.push_back(left[i]);
                for (std::size_t j = 0; j < right.size(); ++j) subexprList.push_back(right[j]);
            }
            else {
                std::string t;
                t.reserve(3 + left[0].size() + right[0].size());
                t += "(&";
                t += left[0];
                t += right[0];
                t += ")";
                subexprList.push_back(t);
            }
        }
        else {
            std::size_t endIndex = s.find(')', index);
            if (endIndex == std::string::npos) {
                throw std::runtime_error("No closing ')'.");
            }
            std::string nodeLabel = s.substr(index, endIndex - index);
            index = endIndex;
            std::string t;
            t.reserve(2 + nodeLabel.size());
            t.push_back('(');
            t += nodeLabel;
            t.push_back(')');
            subexprList.push_back(t);
        }
    }
    else if (index + 1 < s.size() && s[index] == '!' && s[index + 1] == '(') {
        index += 2;
        if (index < s.size() && s[index] == '>') {
            index += 1;
            std::vector<std::string> argsToRemove = ce::getArgs(s.substr(index));
            std::size_t close = s.find(']', index);
            if (close == std::string::npos) {
                throw std::runtime_error("No closing ']'.");
            }
            index = close + 1;

            std::vector<std::string> left = groomSubexpr(s, 0, index);
            std::vector<std::string> right = groomSubexpr(s, 0, index);

            std::string argsJoined = ce::joinWithComma(argsToRemove);
            std::string t;
            t.reserve(6 + argsJoined.size() + left[0].size() + right[0].size());
            t += "!(>[";
            t += argsJoined;
            t += "]";
            t += left[0];
            t += right[0];
            t += ")";
            subexprList.push_back(t);
        }
        else if (index < s.size() && s[index] == '&') {
            index += 1;

            std::vector<std::string> left = groomSubexpr(s, 0, index);
            std::vector<std::string> right = groomSubexpr(s, 0, index);

            // !(&A B) is a disjunction (De Morgan: ¬(A∧B) ≡ ¬A∨¬B).
            // Never flatten — splitting disjuncts into independent facts would
            // conflate OR with AND. Always keep it wrapped as a single expr.
            std::string t;
            t.reserve(5 + left[0].size() + right[0].size());
            t += "!(&";
            t += left[0];
            t += right[0];
            t += ")";
            subexprList.push_back(t);
        }
        else {
            std::size_t endIndex = s.find(')', index);
            if (endIndex == std::string::npos) {
                throw std::runtime_error("No closing ')'.");
            }
            std::string nodeLabel = s.substr(index, endIndex - index);
            index = endIndex;
            std::string t;
            t.reserve(4 + nodeLabel.size());
            t += "!(";
            t += nodeLabel;
            t += ")";
            subexprList.push_back(t);
        }
    }

    index += 1;
    return subexprList;
}

/// @brief Enumerate independent conclusion expressions.
///
/// @details Walks the expression in the existing left-to-right order and retains compound premises and negated structures.
/// @param treeStr Expression to inspect.
/// @return Ordered conclusion expressions.
std::vector<std::string> ExpressionParser::groomExpr(const std::string& treeStr) {
    std::size_t index = 0;
    return groomSubexpr(treeStr, 1, index);
}

/// @brief Read the first implication binder argument list.
///
/// @details Returns the existing ordered binder spelling, including its established empty-list representation. Expressions without a binder have an empty result.
/// @param expr Expression to inspect.
/// @return Ordered bound arguments.
std::vector<std::string> ExpressionParser::listLastRemovedArgs(const std::string& expr) {
    std::vector<std::string> out;
    std::size_t start = expr.find(">[");
    if (start == std::string::npos) {
        return out;
    }
    std::size_t end = expr.find(']', start + 2);
    if (end == std::string::npos) {
        return out;
    }
    std::string inner = expr.substr(start + 2, end - (start + 2));
    std::size_t pos = 0;
    while (pos <= inner.size()) {
        std::size_t comma = inner.find(',', pos);
        if (comma == std::string::npos) {
            out.push_back(inner.substr(pos));
            break;
        }
        out.push_back(inner.substr(pos, comma - pos));
        pos = comma + 1;
    }
    return out;
}

/// @brief Rename the single outer bound argument for compilation.
///
/// @details Preserves the compiler variable counter and token-aware substitution. The expression must have exactly one referenced outer bound argument.
/// @param expr Implication or negated implication.
/// @param startInt Next compiler variable number.
/// @return Rewritten expression, next counter and chosen variable name.
std::tuple<std::string, int, std::string>
ExpressionParser::renameLastRemoved(const std::string& expr, int startInt) {
    const bool startsWithImp = expr.size() >= 2 && expr[0] == '(' && expr[1] == '>';
    const bool startsWithNotImp = expr.size() >= 3 && expr[0] == '!' && expr[1] == '(' && expr[2] == '>';
    assert(startsWithImp || startsWithNotImp);

    std::vector<std::string> args = listLastRemovedArgs(expr);
    std::set<std::string> argSet(args.begin(), args.end());
    std::vector<std::string> orderedArgs = ce::orderByPattern(expr, argSet);

    assert(orderedArgs.size() == 1);
    const std::string arg = orderedArgs[0];

    const std::string newVar =
        std::string("comp_") + std::to_string(startInt);

    std::map<std::string, std::string> replacementMap;
    replacementMap.insert(std::make_pair(arg, newVar));

    const std::string newExpr = ce::replaceKeysInString(expr, replacementMap);

    const int newStartInt = startInt + 1;
    return std::make_tuple(newExpr, newStartInt, newVar);
}

} // namespace gl
