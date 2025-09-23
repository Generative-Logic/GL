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

#include "analyze_expressions.hpp"
#include "parameters.hpp"
#include <iostream>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <utility>
#include <filesystem>
#include <stdexcept>
#include <sstream>
#include <unordered_set>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>



namespace gl {

    class SimpleBarrier {
    public:
        explicit SimpleBarrier(unsigned n) : n_(n) {}

        void arrive_and_wait(std::function<void()> last_fn) {
            std::unique_lock<std::mutex> lk(m_);
            const unsigned g = gen_;
            if (++arrived_ == n_) {
                arrived_ = 0;
                ++gen_;
                lk.unlock();              // run outside the lock
                if (last_fn) last_fn();
                cv_.notify_all();
            }
            else {
                cv_.wait(lk, [&] { return gen_ != g; });
            }
        }

    private:
        std::mutex m_;
        std::condition_variable cv_;
        unsigned n_;
        unsigned arrived_ = 0;
        unsigned gen_ = 0;
    };

    //Declarations
    static std::map<std::pair<int, int>, std::vector<std::vector<int>>> generateAllMappings(int n, int m);

    // Assumes n, m > 0.
    // Returns a map from (i, j) with 1 <= i < n and 1 <= j < m
    // to the list of all functions [0..i-1] -> [0..j-1], each function
    // represented as a list<int> of length i whose entries are in [0, j).
    std::map<std::pair<int, int>, std::vector<std::vector<int>>> generateAllMappings(int n, int m) {
        std::map<std::pair<int, int>, std::vector<std::vector<int>>> functionMap;

        for (int i = 1; i < n; ++i) {
            for (int j = 1; j < m; ++j) {
                // total number of functions is j^i
                long long total = 1;
                for (int t = 0; t < i; ++t) {
                    total *= static_cast<long long>(j);
                }

                std::vector<std::vector<int>> functions;
                functions.reserve(static_cast<std::size_t>(total));

                // Mixed-radix counting in base j over i positions.
                // Order matches Python itertools.product(range(j), repeat=i):
                // leftmost position changes slowest, rightmost fastest.
                for (long long index = 0; index < total; ++index) {
                    std::vector<int> f(static_cast<std::size_t>(i), 0);
                    long long value = index;
                    for (int pos = i - 1; pos >= 0; --pos) {
                        f[static_cast<std::size_t>(pos)] = static_cast<int>(value % j);
                        value /= j;
                    }
                    functions.push_back(f);
                }

                functionMap[std::make_pair(i, j)] = functions;
            }
        }

        return functionMap;
    }

// Global state analogous to the Python module-level variables.
static BodyOfProves* gBody = nullptr;
static std::ofstream gQuickFile;

// Small helper to print to both cout and quick_mode.txt
static void logLine(const std::string& s) {
    std::cout << s << std::endl;
    if (gQuickFile.is_open()) gQuickFile << s << '\n';
}







// Default constructor definition
ExpressionAnalyzer::ExpressionAnalyzer()
    : body(),
    allBinariesAna(),
    allMappingsAna(),
    maxNumLeafsPerKey(0),
    allPermutationsAna(),
    globalDependencies(),
	inductionMemoryBlocks(),
	contradictionTable(),
	index(),
    boxes(),
    indexCE(),
    boxesCE()
{
    std::map<int, std::vector<std::vector<int> > > binarySeqsMap;
    for (int num = 0; num <= parameters::sizeAllBinariesAna; ++num) {
        binarySeqsMap[num] = ce::generateBinarySequencesAsLists(num);
    }
    allBinariesAna = binarySeqsMap;

    allMappingsAna = generateAllMappings(
        parameters::maxSizeDefSetMapping,
        parameters::maxSizeTargetSetMapping
    );

    maxNumLeafsPerKey = 0;

    allPermutationsAna = ce::generateAllPermutations(
        parameters::sizeAllPermutationsAna
    );

    Dependencies dependencyTable;
    globalDependencies = dependencyTable;

    coreExpressionMap = ce::modifyCoreExpressionMap();

    anchorInfo = ce::AnchorInfo(
        "(NaturalNumbers[1,2,3,4,5,6])",
        coreExpressionMap["NaturalNumbers"].arity,
        coreExpressionMap["NaturalNumbers"].placeholders,
        "NaturalNumbers"
	);

    permanentBodies.clear();
    permanentBodies.push_back(&body);

    logicalCores = std::max(1u, std::thread::hardware_concurrency());
    //logicalCores = 1;
    assert(logicalCores >= 1);




    std::string expr = "(>[5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[7,10,11,5])(>[12](in3[9,10,12,5])(in3[11,8,12,5])))))";
    
    int test = 0;
}


std::string ExpressionAnalyzer::smoothenExpr(const std::string& expr) {
    struct Helper {
        static void deleteTree(ce::TreeNode1* n) {
            if (n == NULL) { return; }
            deleteTree(n->left);
            deleteTree(n->right);
            delete n;
        }

        static void nodeToStr(const ce::TreeNode1* node, std::string& out, bool& foundAnd) {
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
                    nodeToStr(node->left, out, foundAnd);
                    nodeToStr(node->right, out, foundAnd);
                }
            }
            else if (v == "&") {
                out += "(&";
                nodeToStr(node->left, out, foundAnd);
                nodeToStr(node->right, out, foundAnd);
            }
            else if (v.size() >= 2 && v[0] == '!' && v[1] == '>') {
                out += "!(";
                out += v.substr(1); // strip leading '!' from value like "!>..."
                nodeToStr(node->left, out, foundAnd);
                nodeToStr(node->right, out, foundAnd);
            }
            else if (v == "!&") {
                out += "!(&";
                nodeToStr(node->left, out, foundAnd);
                nodeToStr(node->right, out, foundAnd);
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
    };

    bool found = true;
    std::string smoothenedExpr = expr;

    while (found) {
        // smoothen_one_and(smoothenedExpr)
        std::pair<ce::TreeNode1*, int> pr = ce::parseExpr(smoothenedExpr, this->coreExpressionMap);
        ce::TreeNode1* root = pr.first;

        std::string localExpr;
        bool foundAnd = false;
        Helper::nodeToStr(root, localExpr, foundAnd);

        Helper::deleteTree(root);

        smoothenedExpr = localExpr;
        found = foundAnd;
    }

    return smoothenedExpr;
}

std::vector<std::string> ExpressionAnalyzer::groomSubexpr(const std::string& s, int side, std::size_t& index) {
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

            if (side) {
                subexprList.reserve(left.size() + right.size());
                for (std::size_t i = 0; i < left.size(); ++i) subexprList.push_back(left[i]);
                for (std::size_t j = 0; j < right.size(); ++j) subexprList.push_back(right[j]);
            }
            else {
                std::string t;
                t.reserve(5 + left[0].size() + right[0].size());
                t += "!(&";
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

std::vector<std::string> ExpressionAnalyzer::groomExpr(const std::string& treeStr) {
    std::size_t index = 0;
    return groomSubexpr(treeStr, 1, index);
}

std::vector<std::string> ExpressionAnalyzer::listLastRemovedArgs(const std::string& expr) {
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

std::tuple<std::string, int, std::string>
ExpressionAnalyzer::renameLastRemoved(const std::string& expr, int startInt, int iteration, int level) {
    const bool startsWithImp = expr.size() >= 2 && expr[0] == '(' && expr[1] == '>';
    const bool startsWithNotImp = expr.size() >= 3 && expr[0] == '!' && expr[1] == '(' && expr[2] == '>';
    assert(startsWithImp || startsWithNotImp);

    std::vector<std::string> args = listLastRemovedArgs(expr);
    std::set<std::string> argSet(args.begin(), args.end());
    std::vector<std::string> orderedArgs = ce::orderByPattern(expr, argSet);

    assert(orderedArgs.size() == 1);
    const std::string arg = orderedArgs[0];

    const std::string newVar =
        std::string("it_") + std::to_string(iteration) +
        "_lev_" + std::to_string(level) +
        "_" + std::to_string(startInt);

    std::map<std::string, std::string> replacementMap;
    replacementMap.insert(std::make_pair(arg, newVar));

    const std::string newExpr = ce::replaceKeysInString(expr, replacementMap);

    const int newStartInt = startInt + 1;
    return std::make_tuple(newExpr, newStartInt, newVar);
}

std::string ExpressionAnalyzer::expandExpr(const std::string& expr) {
    std::string expandedExpr = expr;

    if (!expr.empty() && expr.front() == '!')
    {
		return expandedExpr;
    }

    const std::string magicString = "@19023847@";

    for (std::map<std::string, ce::CoreExpressionConfig>::const_iterator it = this->coreExpressionMap.begin();
        it != this->coreExpressionMap.end(); ++it) {
        const std::string& coreExpr = it->first;
        const ce::CoreExpressionConfig& cfg = it->second;

        if (cfg.levelFlag == 1) {
            const std::string needle = std::string("(") + coreExpr + "[";
            std::size_t index = expr.find(needle);
            if (index != std::string::npos) {
                index = index + 1;

                std::vector<std::string> replacingArgs = ce::getArgs(expr.substr(index));
                std::vector<std::string> argsToBeReplaced = ce::getArgs(cfg.signature);

                std::map<std::string, std::string> replacementMap;
                for (std::size_t ind = 0; ind < argsToBeReplaced.size(); ++ind) {
                    replacementMap[argsToBeReplaced[ind]] = replacingArgs[ind] + magicString;
                }

                std::string pattern = std::holds_alternative<std::string>(cfg.definition)
                    ? std::get<std::string>(cfg.definition)
                    : std::string();

                expandedExpr = ce::replaceKeysInString(pattern, replacementMap);

                std::size_t pos = 0;
                while (true) {
                    std::size_t p = expandedExpr.find(magicString, pos);
                    if (p == std::string::npos) break;
                    expandedExpr.erase(p, magicString.size());
                    pos = p;
                }
            }
        }
    }




    return expandedExpr;
}


std::vector<std::string> ExpressionAnalyzer::getGlobalKey(const BodyOfProves& memoryBlock) {
    std::vector<std::string> globalKey;
    const BodyOfProves* current = &memoryBlock;
    while (true) {
        if (!current->exprKey.empty()) {
            globalKey.push_back(current->exprKey);
        }
        if (current->parentBodyOfProves != NULL) {
            current = current->parentBodyOfProves;
        }
        else {
            break;
        }
    }
    std::reverse(globalKey.begin(), globalKey.end());
    return globalKey;
}

void ExpressionAnalyzer::addToHashMemory(const std::vector<std::string>& key,
    const std::string& value,
    const std::set<std::string>& remainingArgs,
    LocalMemory& hashMemory,
    const std::set<int>& levels,
    const std::string& originalImplication,
    int maxAdmissionDepth,
    int maxSecondaryNumber,
    bool partOfRecursion) {
    this->makeNormalizedKeysForAdmission(key, hashMemory, value);

    std::vector<std::string> keyPlusValue = key;
    keyPlusValue.push_back(value);
    updateAdmissionMap(hashMemory,
        keyPlusValue,
        remainingArgs,
        maxAdmissionDepth,
        maxSecondaryNumber,
        partOfRecursion);

    std::vector<int> allOnes(key.size(), 1);

    const std::set<VariantItem> variants = this->createVariants(key, value);

    this->makeNormalizedSubkeys(key, hashMemory);

    for (std::set<VariantItem>::const_iterator it = variants.begin(); it != variants.end(); ++it) {
        const VariantItem& v = *it;

        LocalMemoryValue lmv;
        lmv.value = v.valueVariant;
        lmv.levels = levels;
        lmv.originalImplication = originalImplication;
        lmv.remainingArgs = remainingArgs;

        std::map<NormalizedKey, std::set<LocalMemoryValue> >::iterator emIt =
            hashMemory.encodedMap.find(v.normalizedIgnoredKey);
        if (emIt != hashMemory.encodedMap.end()) {
            emIt->second.insert(lmv);
        }
        else {
            std::set<LocalMemoryValue> bucket;
            bucket.insert(lmv);
            hashMemory.encodedMap.insert(std::make_pair(v.normalizedIgnoredKey, bucket));
        }

        hashMemory.normalizedEncodedKeys.insert(v.normalizedNotIgnoredKey);

        std::map<std::set<std::string>, std::set<NormalizedKey> >::iterator remIt =
            hashMemory.remainingArgsNormalizedEncodedMap.find(remainingArgs);
        if (remIt != hashMemory.remainingArgsNormalizedEncodedMap.end()) {
            remIt->second.insert(v.normalizedNotIgnoredKey);
        }
        else {
            std::set<NormalizedKey> s;
            s.insert(v.normalizedNotIgnoredKey);
            hashMemory.remainingArgsNormalizedEncodedMap.insert(std::make_pair(remainingArgs, s));
        }
    }

    std::map<std::string, int> coreExpressionCounterMap;
    for (std::size_t i = 0; i < key.size(); ++i) {
        const std::string coreExpr = ce::extractExpression(key[i]);
        std::map<std::string, int>::iterator itc = coreExpressionCounterMap.find(coreExpr);
        if (itc == coreExpressionCounterMap.end()) {
            coreExpressionCounterMap.insert(std::make_pair(coreExpr, 1));
        }
        else {
            itc->second += 1;
        }
    }

    hashMemory.maxKeyLength = std::max(static_cast<int>(key.size()), hashMemory.maxKeyLength);
}

bool ExpressionAnalyzer::lessByName(const EncodedExpression& a, const EncodedExpression& b) {
    return a.name < b.name;
}
bool ExpressionAnalyzer::lessByOriginal(const EncodedExpression& a, const EncodedExpression& b) {
    return a.original < b.original;
}

std::vector<EncodedExpression>
ExpressionAnalyzer::sortEncodedExpressionsByName(const std::vector<EncodedExpression>& in) {
    std::vector<EncodedExpression> out = in;
    std::stable_sort(out.begin(), out.end(), &ExpressionAnalyzer::lessByName);
    return out;
}


// .cpp
std::vector<EncodedExpression>
ExpressionAnalyzer::mergeInsertSortedEncoded(const std::vector<EncodedExpression>& listA,
    const std::vector<std::string>& valuesA,
    const std::vector<EncodedExpression>& listB,
    const std::vector<std::string>& valuesB)
{
    // Precondition: valuesA are sorted by name (they come from filteredAll sorted by name). :contentReference[oaicite:1]{index=1}
    // valuesB MAY BE UNSORTED. We sort B by its keys first, stably.
    const std::size_t nA = listA.size();
    const std::size_t nB = std::min(listB.size(), valuesB.size());

    std::vector<std::size_t> idxB(nB);
    for (std::size_t i = 0; i < nB; ++i) idxB[i] = i;

    // Stable sort ensures B’s original relative order for equal keys (matches repeated inserts).
    std::stable_sort(idxB.begin(), idxB.end(),
        [&](std::size_t i, std::size_t j) { return valuesB[i] < valuesB[j]; });

    std::vector<EncodedExpression> out;
    out.reserve(nA + nB);

    std::size_t iA = 0, iB = 0;
    while (iA < nA && iB < nB) {
        const std::string& aKey = valuesA[iA];
        const std::string& bKey = valuesB[idxB[iB]];

        // B-first on ties to match the original lower_bound+insert behavior. :contentReference[oaicite:2]{index=2}
        if (aKey < bKey) {
            out.push_back(listA[iA++]);
        }
        else {
            out.push_back(listB[idxB[iB++]]);
        }
    }
    while (iA < nA) out.push_back(listA[iA++]);
    while (iB < nB) out.push_back(listB[idxB[iB++]]);
    return out;
}




std::vector< std::vector<EncodedExpression> >
ExpressionAnalyzer::makeMandatoryEncodedStatementLists1(const LocalMemory& localMemory,
    const std::vector<EncodedExpression>& localStatements) {
    std::vector<EncodedExpression> tmp = filterEncodedStatements(localStatements, localMemory);
    std::stable_sort(tmp.begin(), tmp.end(), &ExpressionAnalyzer::lessByOriginal);

    std::vector< std::vector<EncodedExpression> > out;
    out.reserve(tmp.size());
    for (std::size_t i = 0; i < tmp.size(); ++i) {
        std::vector<EncodedExpression> one;
        one.push_back(tmp[i]);
        out.push_back(one);
    }
    return out;
}

std::vector< std::vector<EncodedExpression> >
ExpressionAnalyzer::makeMandatoryEncodedStatementLists2(BodyOfProves& body,
    const LocalMemory& localMemory,
    const std::vector<EncodedExpression>& firstLayer,
    const std::vector<EncodedExpression>& secondLayer) {
    std::vector< std::vector<EncodedExpression> > mandatory;

    if (!firstLayer.empty() && !secondLayer.empty()) {
        std::vector<EncodedExpression> f1 = filterEncodedStatements(firstLayer, localMemory);
        std::vector<EncodedExpression> f2 = filterEncodedStatements(secondLayer, localMemory);

        std::stable_sort(f1.begin(), f1.end(), &ExpressionAnalyzer::lessByOriginal);
        std::stable_sort(f2.begin(), f2.end(), &ExpressionAnalyzer::lessByOriginal);

        for (std::size_t i = 0; i < f1.size(); ++i) {
            for (std::size_t j = 0; j < f2.size(); ++j) {
                if (!(f1[i] == f2[j])) {
                    std::vector<EncodedExpression> list2;
                    list2.push_back(f1[i]);
                    list2.push_back(f2[j]);

                    std::pair<bool, NormalizedKey> pe = preEvaluateEncodedKey(list2, body, body.localMemory.normalizedEncodedKeys);
                    if (pe.first) {
                        mandatory.push_back(list2);
                    }
                }
            }
        }
    }
    return mandatory;
}




std::vector< std::tuple< std::vector<EncodedExpression>, bool, NormalizedKey > >
ExpressionAnalyzer::generateEncodedRequests(BodyOfProves& body,
    const LocalMemory& localMemory,
    const std::vector< std::vector<EncodedExpression> >& mandatoryLists,
    const std::vector<EncodedExpression>& allStatements)
{
    typedef std::tuple< std::vector<EncodedExpression>, bool, NormalizedKey > ReqT;
    std::set<ReqT> rqsts; // dedup by full request tuple

    if (mandatoryLists.empty()) {
        return std::vector<ReqT>();
    }

    // --- Universe prep (filter + sort once) ---
    std::vector<EncodedExpression> filteredAll = filterEncodedStatements(allStatements, localMemory);
    filteredAll = sortEncodedExpressionsByName(filteredAll); // stable name order
    std::vector<std::string> valuesFas; valuesFas.reserve(filteredAll.size());
    for (std::size_t i = 0; i < filteredAll.size(); ++i) valuesFas.push_back(filteredAll[i].name);

    // MSL name vectors for merges later
    std::vector< std::vector<std::string> > valuesMsl;
    valuesMsl.reserve(mandatoryLists.size());
    for (std::size_t i = 0; i < mandatoryLists.size(); ++i) {
        const std::vector<EncodedExpression>& lst = mandatoryLists[i];
        std::vector<std::string> names; names.reserve(lst.size());
        for (std::size_t j = 0; j < lst.size(); ++j) names.push_back(lst[j].name);
        valuesMsl.push_back(std::move(names));
    }

    const int maxKeyLen = localMemory.maxKeyLength;
    const int sizeOne = static_cast<int>(mandatoryLists[0].size()); // per contract: only 1 or 2
    const int targetLen = std::max(0, maxKeyLen - sizeOne);

    // Pick the subkey set: (n-1) when sizeOne==1, (n-2) when sizeOne==2
    const std::set<NormalizedKey>* targetSubkeys =
        (sizeOne == 1)
        ? &localMemory.normalizedEncodedSubkeysMinusOne
        : &localMemory.normalizedEncodedSubkeysMinusTwo;

    // --- Seed: bare mandatory lists that are already complete keys ---
    for (std::size_t i = 0; i < mandatoryLists.size(); ++i) {
        const std::vector<EncodedExpression>& rq = mandatoryLists[i];
        NormalizedKey nk = makeNormalizedEncodedKeyOnly(rq, /*ignoreU=*/false);
        if (localMemory.normalizedEncodedKeys.find(nk) != localMemory.normalizedEncodedKeys.end()) {
            rqsts.insert(ReqT(rq, true, nk));
        }
    }

    if (targetLen <= 0) {
        // Nothing to grow; the seed phase covers the degenerate case.
        return std::vector<ReqT>(rqsts.begin(), rqsts.end());
    }

    // --- Grow base requests only until they hit the chosen (n-1)/(n-2) subkey set ---
    struct StackItem {
        int start;
        std::vector<int> indicesSubset;
        std::vector<EncodedExpression> baseReq;
        std::vector<std::string>       baseVals;
    };
    std::vector<StackItem> stack;
    stack.push_back(StackItem{ 0, std::vector<int>(), std::vector<EncodedExpression>(), std::vector<std::string>() });
    stack.back().baseReq.reserve(static_cast<std::size_t>(targetLen));
    stack.back().baseVals.reserve(static_cast<std::size_t>(targetLen));

    // Hold all base (near-complete) requests we’ll later pair with each MSL
    std::vector< std::vector<EncodedExpression> > baseCandidates;

    while (!stack.empty()) {
        StackItem top = std::move(stack.back());
        stack.pop_back();

        for (int i = top.start; i < static_cast<int>(filteredAll.size()); ++i) {
            // do not grow past targetLen
            if (static_cast<int>(top.baseReq.size()) + 1 > targetLen) {
                break; // further i only increases size
            }

            // extend subset in name-order (indices strictly increasing)
            top.indicesSubset.push_back(i);

            // append only the new tail element — no clear/rebuild
            top.baseReq.push_back(filteredAll[static_cast<std::size_t>(i)]);
            top.baseVals.push_back(valuesFas[static_cast<std::size_t>(i)]);

            // peek & prune using the existing guard (secondary count, length, and generic subkey membership)
            std::pair<bool, NormalizedKey> pr =
                preEvaluateEncodedKey(top.baseReq, body, localMemory.normalizedEncodedSubkeys);
            if (pr.first) {
                if (targetSubkeys->find(pr.second) != targetSubkeys->end()) {
                    baseCandidates.push_back(top.baseReq);
                }

                // keep growing if we haven't reached targetLen yet
                if (static_cast<int>(top.baseReq.size()) < targetLen) {
                    stack.push_back(StackItem{ i + 1, top.indicesSubset, top.baseReq, top.baseVals });
                }
            }

            // remove the last element before the next i
            top.baseReq.pop_back();
            top.baseVals.pop_back();
            top.indicesSubset.pop_back();
        }
    }

    // --- For each base candidate, merge with every MSL and accept if the full key exists ---
    for (std::size_t bi = 0; bi < baseCandidates.size(); ++bi) {
        const std::vector<EncodedExpression>& base = baseCandidates[bi];
        // build once
        std::vector<std::string> baseVals; baseVals.reserve(base.size());
        for (std::size_t k = 0; k < base.size(); ++k) baseVals.push_back(base[k].name);

        for (std::size_t j = 0; j < mandatoryLists.size(); ++j) {
            const std::vector<EncodedExpression>& msl = mandatoryLists[j];

            // Skip if any dup would exist (as before)
            bool dup = false;
            for (std::size_t s = 0; s < msl.size() && !dup; ++s) {
                for (std::size_t r = 0; r < base.size(); ++r) {
                    if (msl[s] == base[r]) { dup = true; break; }
                }
            }
            if (dup) continue;

            // O(n+m) stable merge (B-first on ties), already sorted by name
            std::vector<EncodedExpression> rqst =
                mergeInsertSortedEncoded(base, baseVals, msl, valuesMsl[j]);

            // No preEvaluate here; just check the final full-key table.
            std::pair<bool, NormalizedKey> pr2 =
                preEvaluateEncodedKey(rqst, body, localMemory.normalizedEncodedKeys);
            if (pr2.first) {
                rqsts.insert(ReqT(rqst, true, pr2.second));
            }
        }
    }

    return std::vector<ReqT>(rqsts.begin(), rqsts.end());
}





// =================== Main driver ===================

BodyOfProves& ExpressionAnalyzer::performElementaryLogicalStep(BodyOfProves& body,
    unsigned coreId,
    const ParentChildrenMap& index,
    PerCoreMailboxes& boxes) {
    if (!body.isActive) {
        return body;
    }

    if (body.encodedStatements.size() > 30000)
	    std::cout << body.exprKey << " " << body.encodedStatements.size() << " " << std::endl;

    if (body.exprKey == "(>[4,5,6](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10,11](in3[8,10,11,5])(>[](in2[8,7,4])(in3[9,10,11,6])))))")
    {
        int test = 0;
        test++;
    }

    // ingest implications into local memory
    for (std::set< std::tuple< std::vector<std::string>,
        std::string,
        std::set<std::string>,
        std::set<int>,
        std::string> >::const_iterator it = body.mailIn.implications.begin();
        it != body.mailIn.implications.end(); ++it) {
        const std::vector<std::string>& chain = std::get<0>(*it);
        const std::string& head = std::get<1>(*it);
        const std::set<std::string>& remainingArgsKey = std::get<2>(*it);
        const std::set<int>& levels = std::get<3>(*it);
        const std::string& orImpl = std::get<4>(*it);

        this->addToHashMemory(chain, head, remainingArgsKey, body.localMemory, levels,
            orImpl, parameters::standardMaxAdmissionDepth, parameters::standardMaxSecondaryNumber, false);
    }

    // expr_origin_map = mail_in | current   (right wins on conflicts)
    {
        std::map<std::string, std::vector<std::string> > merged = body.mailIn.exprOriginMap;
        for (std::map<std::string, std::vector<std::string> >::const_iterator it = body.exprOriginMap.begin();
            it != body.exprOriginMap.end(); ++it) {
            merged[it->first] = it->second;
        }
        body.exprOriginMap.swap(merged);
    }

    // working memory (only from mail_in implications)
    LocalMemory working;
    for (std::set< std::tuple< std::vector<std::string>,
        std::string,
        std::set<std::string>,
        std::set<int>,
        std::string> >::const_iterator it = body.mailIn.implications.begin();
        it != body.mailIn.implications.end(); ++it) {
        const std::vector<std::string>& chain = std::get<0>(*it);
        const std::string& head = std::get<1>(*it);
        const std::set<std::string>& remainingArgsKey = std::get<2>(*it);
        const std::set<int>& levels = std::get<3>(*it);
        const std::string& orImpl = std::get<4>(*it);

        this->addToHashMemory(chain, head, remainingArgsKey, working, levels,
            orImpl, parameters::standardMaxAdmissionDepth, parameters::standardMaxSecondaryNumber, false);
    }

    std::vector< std::tuple< std::vector<EncodedExpression>, bool, NormalizedKey > > workingReqs;

    if (!working.encodedMap.empty()) {
        std::vector< std::vector<EncodedExpression> > msl1 =
            this->makeMandatoryEncodedStatementLists1(working, body.localEncodedStatements);

        if (!msl1.empty()) {
            workingReqs = this->generateEncodedRequests(body, working, msl1, body.encodedStatements);
        }
    }

    // Convert to a sorted vector (sorted by the pair's first: the string), then iterate it.
    std::vector< std::pair<std::string, std::set<int> > > sortedStatements(
        body.mailIn.statements.begin(), body.mailIn.statements.end());

    // absorb incoming statements
    for (std::vector< std::pair<std::string, std::set<int> > >::const_iterator it = sortedStatements.begin();
        it != sortedStatements.end(); ++it) {
        const std::string& statement = it->first;
        const std::set<int>& levels = it->second;

        if (body.statementLevelsMap.find(statement) == body.statementLevelsMap.end()) {
            std::vector<std::string> origin;
            if (parameters::trackHistory) {
                std::map<std::string, std::vector<std::string> >::const_iterator oit =
                    body.mailIn.exprOriginMap.find(statement);
                if (oit != body.mailIn.exprOriginMap.end()) origin = oit->second;
            }
            addExprToMemoryBlock(statement, body, -1, 3, levels, origin, coreId, -1);
        }
    }


    // new local delta -> requests
    std::vector< std::vector<EncodedExpression> > mNew1 =
        this->makeMandatoryEncodedStatementLists1(body.localMemory, body.localEncodedStatementsDelta);

    std::vector< std::tuple< std::vector<EncodedExpression>, bool, NormalizedKey > > newLocalReqs;
    if (!mNew1.empty()) {
        newLocalReqs = this->generateEncodedRequests(body, body.localMemory, mNew1, body.encodedStatements);
    }

    // local + mail statements (pairs)
    std::vector<EncodedExpression> mailEncodedStatements;
    for (std::set< std::pair<std::string, std::set<int> > >::const_iterator it = body.mailIn.statements.begin();
        it != body.mailIn.statements.end(); ++it) {
        mailEncodedStatements.push_back(EncodedExpression(it->first));
    }

    std::vector< std::vector<EncodedExpression> > mLocalMail2 =
        this->makeMandatoryEncodedStatementLists2(body, body.localMemory, body.localEncodedStatements, mailEncodedStatements);

    std::vector< std::tuple< std::vector<EncodedExpression>, bool, NormalizedKey > > localMailReqs;
    if (!mLocalMail2.empty()) {
        localMailReqs = this->generateEncodedRequests(body, body.localMemory, mLocalMail2, body.encodedStatements);
    }

    // combine
    std::vector< std::tuple< std::vector<EncodedExpression>, bool, NormalizedKey > > encodedRequests;
    encodedRequests.reserve(workingReqs.size() + newLocalReqs.size() + localMailReqs.size());
    for (std::size_t i = 0; i < workingReqs.size(); ++i)  encodedRequests.push_back(workingReqs[i]);
    for (std::size_t i = 0; i < newLocalReqs.size(); ++i) encodedRequests.push_back(newLocalReqs[i]);
    for (std::size_t i = 0; i < localMailReqs.size(); ++i) encodedRequests.push_back(localMailReqs[i]);

    // clear mail_in and local delta (like Python)
    body.mailIn.statements.clear();
    body.mailIn.implications.clear();
    body.mailIn.exprOriginMap.clear();
    body.localEncodedStatementsDelta.clear();

    // evaluate requests
    for (std::size_t r = 0; r < encodedRequests.size(); ++r) {
        if (!body.isActive)
        {
            return body;
        }

        const std::vector<EncodedExpression>& reqKey = std::get<0>(encodedRequests[r]);

        bool toContinue = false;
        for (std::size_t s = 0; s < reqKey.size(); ++s) {
            const std::string& orig = reqKey[s].original;
            if (body.statementLevelsMap.find(orig) == body.statementLevelsMap.end()) {
                toContinue = true; break;
            }
        }
        if (toContinue) continue;

        int itr = -1;
        for (std::size_t s = 0; s < reqKey.size(); ++s) {
            if (itr < reqKey[s].maxIterationNumber) {
                itr = reqKey[s].maxIterationNumber;
            }
        }

        this->checkLocalEncodedMemory(reqKey, body, itr + 1, std::get<2>(encodedRequests[r]), coreId);
        int test = 0;
        test++;
    }

    if (body.exprKey == "(NaturalNumbers[1,2,3,4,5,6])")
    {
        int test = 0;
        test++;
    }

    // Send to per-core mailboxes for children of 'body'
    this->sendMail(body, body.mailOut, coreId, index, boxes);
    body.mailOut.statements.clear();
    body.mailOut.implications.clear();
    body.mailOut.exprOriginMap.clear();

#if 0
    // recurse into children
    for (std::map<std::string, BodyOfProves*>::const_iterator it = body.simpleMap.begin();
        it != body.simpleMap.end(); ++it) {
        if (it->second != NULL) {
            this->performElementaryLogicalStep(*it->second);
        }
    }
#endif

    return body;
}







std::set<std::string>
ExpressionAnalyzer::findImmutableArgs(const std::string& theorem,
    const std::string& digit) {
    std::set<std::string> immutables;

    // Disintegrate implication to a chain of left subexpressions.
    std::vector< std::tuple<
        std::string,                  // leftExpr
        std::vector<std::string>,     // args of current '>'
        std::set<std::string>         // left node arguments (unused here)
    > > tempChain;

    ce::disintegrateImplication(theorem, tempChain, this->coreExpressionMap);

    std::vector<std::string> chain;
    chain.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) {
        chain.push_back(std::get<0>(tempChain[i]));
    }

    // Start with all digit args except the given 'digit'
    std::set<std::string> digits = ce::findDigitArgs(theorem, anchorInfo, coreExpressionMap);
    immutables.insert(digits.begin(), digits.end());
    immutables.erase(digit);

    // Propagate immutability forward along the chain
    bool found = true;
    while (found) {
        found = false;

        for (std::size_t i = 0; i < chain.size(); ++i) {
            const std::string& expression = chain[i];
            const std::string coreExpr = ce::extractExpression(expression);
            if (ce::operators.find(coreExpr) != ce::operators.end()) {
                const std::vector<std::string> args = ce::getArgs(expression);
                if (args.size() < 2) {
                    continue;
                }

                // candidate is the penultimate argument
                const std::string& cand = args[args.size() - 2];

                // inputs are all except the last two (outputs)
                std::set<std::string> inputs;
                for (std::size_t k = 0; k + 2 < args.size(); ++k) {
                    inputs.insert(args[k]);
                }

                // if all inputs are already immutable and cand is not yet immutable, add it
                bool inputsSubset = true;
                for (std::set<std::string>::const_iterator it = inputs.begin();
                    it != inputs.end(); ++it) {
                    if (immutables.find(*it) == immutables.end()) {
                        inputsSubset = false;
                        break;
                    }
                }

                if (inputsSubset && immutables.find(cand) == immutables.end()) {
                    immutables.insert(cand);
                    found = true;
                }
            }
        }
    }

    return immutables;
}

BodyOfProves* ExpressionAnalyzer::accessBodyOfProves(const std::vector<std::string>& theoremKey,
    BodyOfProves& bodyOfProves1) {
    BodyOfProves* memoryBlock = &bodyOfProves1;
    for (std::size_t i = 0; i < theoremKey.size(); ++i) {
        const std::string& key = theoremKey[i];
        std::map<std::string, BodyOfProves*>::iterator it = memoryBlock->simpleMap.find(key);
        if (it == memoryBlock->simpleMap.end() || it->second == NULL) {
            return NULL;
        }
        memoryBlock = it->second; // follow the pointer
    }
    return memoryBlock;
}

// analyze_expressions.cpp
// Thread-safe: locks globalTheoremList only when mutating it.
void ExpressionAnalyzer::updateGlobalDirect(const std::string& theorem, int coreId) {
    // 1) Disintegrate to chain + head
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    Mail mailOut;

    deactivateUnnecessary(theorem, this->body);  // deactivates all blocks tied to this theorem.

    std::string value = ce::disintegrateImplication(theorem, tempChain, this->coreExpressionMap);

    std::vector<std::string> ky;
    ky.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

    // 2) Emit implication into the global mail_out
    mailOut.implications.insert(std::make_tuple(ky, value, std::set<std::string>(), std::set<int>(), theorem));
    if (parameters::trackHistory) {
        mailOut.exprOriginMap[theorem] = std::vector<std::string>(1, "theorem");
    }

    // 3) Record theorem and print (guarded)
    {
        std::lock_guard<std::mutex> lock(this->theoremListMutex);
        this->globalTheoremList.emplace_back(theorem, "direct", "-1", "-1");
    }
    std::cout << theorem << std::endl;

    // 4) Try to create reshuffled/mirrored variant (anchor-first)
    const std::string reshuffledMirrored =
        ce::createReshuffledMirrored(theorem, this->anchorInfo.name, true, this->coreExpressionMap);  // :contentReference[oaicite:3]{index=3}

    if (!reshuffledMirrored.empty()) {
        {
            std::lock_guard<std::mutex> lock(this->theoremListMutex);
            this->globalTheoremList.emplace_back(reshuffledMirrored, "mirrored statement", theorem, "-1");
        }

        if (parameters::trackHistory) {
            mailOut.exprOriginMap[reshuffledMirrored] = std::vector<std::string>(1, "theorem");
        }

        // rebuild chain for the mirrored expr
        tempChain.clear();
        value = ce::disintegrateImplication(reshuffledMirrored, tempChain, this->coreExpressionMap);

        ky.clear();
        ky.reserve(tempChain.size());
        for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

        BodyOfProves* memoryBlockM = accessBodyOfProves(ky, this->body);
        if (memoryBlockM != NULL) {
            std::vector<std::string> originM;
            originM.push_back("implication");
            originM.push_back(reshuffledMirrored);
            originM.insert(originM.end(), ky.begin(), ky.end());

            std::set<int> levelsM;
            for (int i = 0; i <= static_cast<int>(ky.size()); ++i) levelsM.insert(i);

            addStatement(value, *memoryBlockM, false, levelsM, originM);
        }

        // Also queue the mirrored implication
        mailOut.implications.insert(
            std::make_tuple(ky, value, std::set<std::string>(), std::set<int>(), reshuffledMirrored)
        );
    }

    // 5) Flush mail
    assert(coreId >= 0);
    sendMail(this->body, mailOut, coreId, index, boxes);  // :contentReference[oaicite:4]{index=4}
}



void ExpressionAnalyzer::deactivateUnnecessary(const std::string& theorem, BodyOfProves& root) {
    // 1) Collect original indices for this theorem
    std::vector<int> originals;
    for (const auto& kv : this->globalDependencies.originalAuxyMap) {
        if (kv.second.expr == theorem) originals.push_back(kv.first);
    }
    if (originals.empty()) return;

    // 2) Disintegrate theorem to find its BodyOfProves path
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    ce::disintegrateImplication(theorem, tempChain, this->coreExpressionMap);

    std::vector<std::string> chain;
    chain.reserve(tempChain.size());
    for (const auto& t : tempChain) chain.push_back(std::get<0>(t));    

    BodyOfProves* memoryBlock = &root;
    for (std::size_t index = 0; index < chain.size(); ++index) {
        const std::string& element = chain[index];

        std::map<std::string, BodyOfProves*>::iterator it = memoryBlock->simpleMap.find(element);
        if (it != memoryBlock->simpleMap.end() && it->second != NULL) {
            memoryBlock = it->second;
        }
        else
        {
            assert(false);
        }
    }

        for (int originalIndex : originals) {
            // one induction variable and its recursion counter per original
            auto itIV = this->globalDependencies.originalInductionVariableMap.find(originalIndex);
            if (itIV == this->globalDependencies.originalInductionVariableMap.end()) continue;

            const std::string& indVar = std::get<0>(itIV->second); // e.g., "i2" or "j3"
            const std::string& recCounter = std::get<1>(itIV->second); // numeric string
            const std::string recArg = std::string("rec") + recCounter;

            const std::string zeroArgName = this->findZeroArgName(*memoryBlock);
            const std::string sName = this->findSName(*memoryBlock);

            // Keys created in addTheoremToMemory:
            const std::string k_rec_in2 = std::string("(in2[") + recArg + "," + indVar + "," + sName +"])";
            const std::string k_eq_side = std::string("(=[s(") + recArg + ")," + zeroArgName + "])";

            auto it1 = memoryBlock->simpleMap.find(k_rec_in2);
            if (it1 != memoryBlock->simpleMap.end() && it1->second)
                it1->second->isActive = false;
            else
            {
                assert(false);
            }

            auto it2 = memoryBlock->simpleMap.find(k_eq_side);
            if (it2 != memoryBlock->simpleMap.end() && it2->second)
                it2->second->isActive = false;
            else
            {
                assert(false);
            }

    }
}


// analyze_expressions.cpp
// Thread-safe: locks globalDependencies briefly, and locks globalTheoremList only when pushing.
void ExpressionAnalyzer::updateGlobal(int auxyIndex, bool allLevelsInvolved, int coreId) {
    // ---- dependency bookkeeping (short critical section) ----
    int originalIndex = -1;
    std::string expr;
    std::string indVar;
    std::string recCounter;
    bool promote = false;

    {
        std::lock_guard<std::mutex> lock(this->dependenciesMutex);

        // Resolve original and update its bookkeeping
        originalIndex = this->globalDependencies.auxyOriginalMap[auxyIndex];

        auto& origItem = this->globalDependencies.originalAuxyMap[originalIndex];
        origItem.auxies.erase(auxyIndex);
        origItem.allLevelsInvolved = allLevelsInvolved || origItem.allLevelsInvolved;

        // If ready to promote, capture the data we need before releasing the lock
        if (origItem.auxies.empty() && origItem.allLevelsInvolved) {
            promote = true;
            expr = origItem.expr;

            auto itIV = this->globalDependencies.originalInductionVariableMap.find(originalIndex);
            if (itIV != this->globalDependencies.originalInductionVariableMap.end()) {
                indVar = std::get<0>(itIV->second);
                recCounter = std::get<1>(itIV->second);
            }
        }
    }

    if (!promote) return;

    // ---- heavy work without holding dependency lock ----
    Mail mailOut;

    deactivateUnnecessary(expr, this->body);  // deactivates all blocks tied to this theorem.

    // Disintegrate expr -> (chain, head)
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    std::string value = ce::disintegrateImplication(expr, tempChain, this->coreExpressionMap);

    std::vector<std::string> ky;
    ky.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

    // Walk to the memory block for this key
    BodyOfProves* memoryBlock = accessBodyOfProves(ky, this->body);
    if (memoryBlock != NULL) {
        std::map<std::string, std::set<int>>::iterator it = memoryBlock->toBeProved.find(value);
        if (it != memoryBlock->toBeProved.end()) {
            // Python asserts this is an empty set
            assert(it->second.empty());
            memoryBlock->toBeProved.erase(it);

            // Build origin and levels
            std::vector<std::string> origin;
            origin.push_back("implication");
            origin.push_back(expr);
            origin.insert(origin.end(), ky.begin(), ky.end());

            std::set<int> levels;
            for (int i = 0; i <= static_cast<int>(ky.size()); ++i) levels.insert(i);

            // Add the head as a statement at this memory block
            addStatement(value, *memoryBlock, false, levels, origin);

            // Queue the implication in global mail_out
            mailOut.implications.insert(std::make_tuple(ky, value, std::set<std::string>(), std::set<int>(), expr));

            if (parameters::trackHistory) {
                mailOut.exprOriginMap[expr] = std::vector<std::string>(1, "theorem");
            }

            // ---- record in globalTheoremList (short critical section) ----
            {
                std::lock_guard<std::mutex> lock(this->theoremListMutex);
                this->globalTheoremList.emplace_back(expr, "induction", indVar, recCounter);
            }
            std::cout << expr << std::endl;

            // Try anchor-first reshuffled / mirrored variant
            const std::string reshuffledMirrored =
                ce::createReshuffledMirrored(expr, this->anchorInfo.name, true, this->coreExpressionMap);  // :contentReference[oaicite:1]{index=1}

            if (!reshuffledMirrored.empty()) {
                {
                    std::lock_guard<std::mutex> lock(this->theoremListMutex);
                    this->globalTheoremList.emplace_back(reshuffledMirrored, "mirrored statement", expr, "-1");
                }

                if (parameters::trackHistory) {
                    mailOut.exprOriginMap[reshuffledMirrored] = std::vector<std::string>(1, "theorem");
                }

                // Disintegrate mirrored and try to add its head too
                tempChain.clear();
                std::string valueM =
                    ce::disintegrateImplication(reshuffledMirrored, tempChain, this->coreExpressionMap);

                ky.clear();
                ky.reserve(tempChain.size());
                for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

                BodyOfProves* memoryBlockM = accessBodyOfProves(ky, this->body);
                if (memoryBlockM != NULL) {
                    std::vector<std::string> originM;
                    originM.push_back("implication");
                    originM.push_back(reshuffledMirrored);
                    originM.insert(originM.end(), ky.begin(), ky.end());

                    std::set<int> levelsM;
                    for (int i = 0; i <= static_cast<int>(ky.size()); ++i) levelsM.insert(i);

                    addStatement(valueM, *memoryBlockM, false, levelsM, originM);
                }

                // Also queue the mirrored implication in global mail_out
                mailOut.implications.insert(
                    std::make_tuple(ky, valueM, std::set<std::string>(), std::set<int>(), reshuffledMirrored)
                );
            }

            // Flush mail
            assert(coreId >= 0);
            sendMail(this->body, mailOut, coreId, index, boxes);  // :contentReference[oaicite:2]{index=2}
        }
    }
}


inline void ExpressionAnalyzer::updateAdmissionMap3(const std::string& expr,
    BodyOfProves& memoryBlock,
    int maxAdmissionDepth,
    int maxSecondaryNumber,
    bool partOfRecursion) {
    // 1) Compute digit arguments of the expression (uses analyzer's anchor)
    std::set<std::string> digitArgs = findDigitArgs(expr, this->anchorInfo, this->coreExpressionMap);

    // 2) Walk up the BodyOfProves chain to find the first block whose exprKey shares a digit arg
    BodyOfProves* tempMb = &memoryBlock;
    while (true) {
        // Build set of arguments from this block's exprKey
        const std::vector<std::string> keyArgsVec = ce::getArgs(tempMb->exprKey);
        std::set<std::string> exprKeyArgs;
        for (std::size_t i = 0; i < keyArgsVec.size(); ++i) {
            exprKeyArgs.insert(keyArgsVec[i]);
        }

        // Check intersection with digitArgs
        bool intersects = false;
        for (std::set<std::string>::const_iterator it = digitArgs.begin();
            it != digitArgs.end(); ++it) {
            if (exprKeyArgs.find(*it) != exprKeyArgs.end()) {
                intersects = true;
                break;
            }
        }

        if (intersects) {
            // Remaining args = digitArgs ∪ { last argument of expr }
            const std::vector<std::string> exprArgs = ce::getArgs(expr);
            if (!exprArgs.empty()) {
                std::set<std::string> remainingArgs = digitArgs;
                remainingArgs.insert(exprArgs.back());

                // Key is just [expr]
                std::vector<std::string> key;
                key.push_back(expr);

                // Delegate to previously converted updateAdmissionMap
                this->updateAdmissionMap(tempMb->localMemory,
                    key,
                    remainingArgs,
                    maxAdmissionDepth,
                    maxSecondaryNumber,
                    partOfRecursion);
            }
            break;
        }
        else {
            if (tempMb->parentBodyOfProves != NULL) {
                tempMb = tempMb->parentBodyOfProves;
            }
            else {
                break;
            }
        }
    }
}


std::string ExpressionAnalyzer::negate(std::string expr) {
    if (!expr.empty() && expr.front() == '!') return expr.substr(1);
    return "!" + expr;
}



void ExpressionAnalyzer::addExprToMemoryBlock(const std::string& expr,
    BodyOfProves& memoryBlock,
    int iteration,
    int status,
    const std::set<int>& involvedLevels,
    const std::vector<std::string>& origin,
    int coreId,
    int auxyIndex) {
    if (memoryBlock.wholeExpressions.find(expr) != memoryBlock.wholeExpressions.end()) return;

    /*if (status == 4)
    {
        memoryBlock.localEncodedStatements.push_back(EncodedExpression(expr));
		memoryBlock.localEncodedStatementsDelta.push_back(EncodedExpression(expr));
        memoryBlock.encodedStatements.push_back(EncodedExpression(expr));
		memoryBlock.statementLevelsMap[expr] = involvedLevels;
		memoryBlock.wholeExpressions.insert(expr);
        memoryBlock.
    }*/

	std::string negation = this->negate(expr);
    if (memoryBlock.wholeExpressions.find(negation) != memoryBlock.wholeExpressions.end())
    {
        // Contradiction detected
        if (memoryBlock.contradictionIndex >= 0) 
        {
			this->contradictionTable[memoryBlock.contradictionIndex].successful = true;
            memoryBlock.isActive = false;
			//std::cout << "contradiction found for " << memoryBlock.exprKey << std::endl;
            return;
        }
        else
        {
            //throw std::runtime_error("Unintended contradiction found");
            std::cout << expr << " " << memoryBlock.exprKey << std::endl;
			memoryBlock.isActive = false;
        }
    }

    if (status != 2) {
        if (this->isEquality(expr)) {
            std::vector<std::string> args = ce::getArgs(expr);
            if (args.size() == 2) {
                const std::string mirrored = std::string("(=[") + args[1] + "," + args[0] + "])";
                memoryBlock.wholeExpressions.insert(mirrored);
                if (status == 1) {
                    memoryBlock.mailOut.statements.insert(std::make_pair(expr, involvedLevels));
                    memoryBlock.mailOut.statements.insert(std::make_pair(mirrored, involvedLevels));
                    if (parameters::trackHistory) {
                        if (memoryBlock.mailOut.exprOriginMap.find(expr) == memoryBlock.mailOut.exprOriginMap.end()) {
                            memoryBlock.mailOut.exprOriginMap[expr] = origin;
                        }
                        if (memoryBlock.mailOut.exprOriginMap.find(mirrored) == memoryBlock.mailOut.exprOriginMap.end()) {
                            std::vector<std::string> mirroredOrigin;
                            mirroredOrigin.push_back("symmetry of equality");
                            mirroredOrigin.push_back(expr);
                            memoryBlock.mailOut.exprOriginMap[mirrored] = mirroredOrigin;
                        }
                    }
                }
            }
        }
        memoryBlock.wholeExpressions.insert(expr);
    }

    if (status == 2) {
        std::map<std::string, std::set<int> >::iterator itTBP = memoryBlock.toBeProved.find(expr);
        if (itTBP != memoryBlock.toBeProved.end()) {
            if (auxyIndex >= 0) itTBP->second.insert(auxyIndex);
        }
        else {
            if (auxyIndex >= 0) {
                std::set<int> s; s.insert(auxyIndex);
                memoryBlock.toBeProved.insert(std::make_pair(expr, s));
            }
            else {
                memoryBlock.toBeProved.insert(std::make_pair(expr, std::set<int>()));
            }
        }

        const std::string coreExpr = ce::extractExpression(expr);
        if (ce::operators.find(coreExpr) != ce::operators.end()) {
            this->updateAdmissionMap3(expr,
                memoryBlock,
                parameters::inductionMaxAdmissionDepth,
                parameters::inductionMaxSecondaryNumber,
                true);
        }
        return;
    }
    else {
        if (parameters::trackHistory) {
            if (memoryBlock.exprOriginMap.find(expr) == memoryBlock.exprOriginMap.end()) {
                memoryBlock.exprOriginMap[expr] = origin;
                memoryBlock.mailOut.exprOriginMap[expr] = origin;
            }
        }

        bool isSimple = ce::expressionIsSimple(expr);
        std::vector<std::string> newStatements;

        if (isSimple) {
            bool isLocal = (status == 0 || status == 1);
            std::vector<std::string> added = this->addStatement(expr,
                memoryBlock,
                isLocal,
                involvedLevels,
                origin);
            newStatements = added;
        }

        std::vector<std::string> sortedNew = newStatements;
        std::sort(sortedNew.begin(), sortedNew.end());
        for (std::size_t idx = 0; idx < sortedNew.size(); ++idx) {
            const std::string& addExpression = sortedNew[idx];

            std::map<std::string, std::set<int> >::const_iterator sit =
                memoryBlock.statementLevelsMap.find(addExpression);
            if (sit == memoryBlock.statementLevelsMap.end()) continue;
            const std::set<int>& addExpressionLevels = sit->second;

            bool allLevelsInvolved = (addExpressionLevels.size() == static_cast<std::size_t>(memoryBlock.level + 1));

            if (memoryBlock.isPartOfRecursion) {
                std::map<std::string, std::set<int> >::iterator itProof =
                    memoryBlock.toBeProved.find(addExpression);
                if (itProof != memoryBlock.toBeProved.end()) {
                    std::set<int> auxiesCopy = itProof->second;
                    {
                        std::lock_guard<std::mutex> lock(this->updateGlobalMutex);
                        for (std::set<int>::const_iterator ait = auxiesCopy.begin(); ait != auxiesCopy.end(); ++ait) {
                            this->updateGlobalTuples.push_back(std::make_tuple(*ait, allLevelsInvolved, coreId));
                        }
                    }

                    memoryBlock.toBeProved.erase(itProof);
                    memoryBlock.isActive = false;

                    if (startsWithStr(memoryBlock.exprKey, "(in2[rec")) {

                        {
                            std::lock_guard<std::mutex> lock(this->inductionMemoryBlocksMutex);
							this->inductionMemoryBlocks.push_back(&memoryBlock);
                        }
#if 0
                        std::vector<std::string> args0 = ce::getArgs(memoryBlock.exprKey);
                        std::string zeroArgName = this->findZeroArgName(memoryBlock);
                        std::string eqExpr = std::string("(=[s(") + args0[0] + ")," + zeroArgName + "])";
                        if (memoryBlock.parentBodyOfProves != NULL) {
                            std::map<std::string, BodyOfProves*>::iterator itChild =
                                memoryBlock.parentBodyOfProves->simpleMap.find(eqExpr);
                            if (itChild != memoryBlock.parentBodyOfProves->simpleMap.end() && itChild->second) {
                                itChild->second->isActive = true;
                            }
                        }
#endif //#if 0
                    }
                }
            }
            else {
                if (this->isProved(addExpression) && allLevelsInvolved && status != 0) {
                    std::vector<std::string> globalKey = this->getGlobalKey(memoryBlock);
                    std::string fullTheorem = this->reconstructImplication(globalKey, addExpression);

                    std::map<std::string, std::set<int> >::iterator itTBP2 =
                        memoryBlock.toBeProved.find(addExpression);
                    if (itTBP2 != memoryBlock.toBeProved.end()) {
                        {
                            std::lock_guard<std::mutex> lock(this->updateGlobalDirectMutex);
                            this->updateGlobalDirectTuples.push_back(std::make_tuple(fullTheorem, coreId));
                        }
                        memoryBlock.toBeProved.erase(itTBP2);
                    }
                }
            }
        }

        std::string expandedExpr = this->expandExpr(expr);
        if (parameters::trackHistory) {
            if (memoryBlock.exprOriginMap.find(expandedExpr) == memoryBlock.exprOriginMap.end()) {
                std::vector<std::string> expOrigin;
                expOrigin.push_back("expansion");
                expOrigin.push_back(expr);
                memoryBlock.exprOriginMap[expandedExpr] = expOrigin;
                memoryBlock.mailOut.exprOriginMap[expandedExpr] = expOrigin;
            }
        }

        if (expandedExpr == expr && isSimple) return;

        if (memoryBlock.statementLevelsMap.find(expandedExpr) == memoryBlock.statementLevelsMap.end()) {
            memoryBlock.statementLevelsMap[expandedExpr] = involvedLevels;
        }

        if (status != 3) {
            std::set<std::string> imps;
            std::set<std::string> stmts;
            int newStart = memoryBlock.startInt;

            std::tuple< std::set<std::string>, std::set<std::string>, int> tpl = this->disintegrateExpr(expandedExpr,
                newStart,
                iteration,
                memoryBlock.level,
                memoryBlock.localMemory);
            imps = std::get<0>(tpl);
			stmts = std::get<1>(tpl);
            memoryBlock.startInt = std::get<2>(tpl);

            std::vector<std::string> localOrigin;
            if (parameters::trackHistory) {
                localOrigin.push_back("disintegration");
                localOrigin.push_back(expandedExpr);
            }

            for (std::set<std::string>::const_iterator itImp = imps.begin(); itImp != imps.end(); ++itImp) {
                std::pair<std::string, std::string> kv = ce::extractKeyValue(*itImp, coreExpressionMap);
                const std::string& ky = kv.first;
                const std::string& vlue = kv.second;

                std::set<std::string> remainingArgsKey = ce::extractDifference(ky);

                std::set<std::string> remainingArgsImpl = ce::extractDifference(*itImp);
                std::map<std::string, std::string> replacementMapRA;
                for (std::set<std::string>::const_iterator ra = remainingArgsImpl.begin(); ra != remainingArgsImpl.end(); ++ra) {
                    replacementMapRA.insert(std::make_pair(*ra, std::string("u_") + *ra));
                }
                std::string replacedImpl = ce::replaceKeysInString(*itImp, replacementMapRA);

                std::vector< std::tuple<
                    std::string,                    // leftExpr
                    std::vector<std::string>,       // args of the current implication node
                    std::set<std::string>           // node->left.arguments  (SET, like Python)
                > >tempChain;
                std::string head = ce::disintegrateImplication(replacedImpl, tempChain, coreExpressionMap);
                std::vector<std::string> chain;
                for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

                this->addToHashMemory(chain,
                    head,
                    remainingArgsKey,
                    memoryBlock.localMemory,
                    involvedLevels,
                    *itImp,
                    parameters::standardMaxAdmissionDepth,
                    parameters::standardMaxSecondaryNumber,
                    false);

                std::set< std::tuple< std::vector<std::string>, std::string, std::set<std::string>, std::set<int>, std::string > >& impSet =
                    memoryBlock.mailOut.implications;

                impSet.insert(std::make_tuple(chain,
                    head,
                    remainingArgsKey,
                    involvedLevels,
                    *itImp));

                if (parameters::trackHistory) {
                    if (memoryBlock.exprOriginMap.find(*itImp) == memoryBlock.exprOriginMap.end()) {
                        memoryBlock.exprOriginMap[*itImp] = localOrigin;
                        memoryBlock.mailOut.exprOriginMap[*itImp] = localOrigin;
                    }
                }
            }

            for (std::set<std::string>::const_iterator itSt = stmts.begin(); itSt != stmts.end(); ++itSt) {
                this->addExprToMemoryBlock(*itSt,
                    memoryBlock,
                    iteration,
                    0,
                    involvedLevels,
                    localOrigin,
                    coreId,
                    auxyIndex);
            }
        }
    }
}

std::vector<std::string> ExpressionAnalyzer::extractSubstringsForAuxy(const std::string& expr2) {
    // Matches: "(" + (first char not >, (,[ or () ) + "[...]" + ")"
    const std::regex re(R"(\(([^>(\[]+\[[^\]]*\])\))");
    std::sregex_iterator it(expr2.begin(), expr2.end(), re);
    std::sregex_iterator end;
    std::vector<std::string> out;
    for (; it != end; ++it) {
        out.push_back(it->str(0)); // whole match with surrounding "()"
    }
    return out;
}

    std::tuple<
    std::string,
    std::vector<std::string>,
    std::string,
    std::set<std::string>,
    std::string,
    std::string
>
ExpressionAnalyzer::createAuxyImplication(const std::string& expr,
    const std::string& arg,
    const std::string& recArg,
    const std::set<std::string>& digitArgs,
    const std::string& anchorName) {
    std::set<std::string> untouchables = digitArgs;
    untouchables.insert(recArg);

    std::vector<std::string> chain;
    std::string zeroArgName;
    std::string sName;

    // replace arg -> recArg in the whole expression
    std::map<std::string, std::string> replMap;
    replMap[arg] = recArg;
    const std::string replExpr = ce::replaceKeysInString(expr, replMap);

    // collect non-top-level subexpressions "(X[...])" with X != '>'
    const std::vector<std::string> subexprs = extractSubstringsForAuxy(replExpr);
    for (std::size_t i = 0; i < subexprs.size(); ++i) {
        const std::string& sub = subexprs[i];
        if (sub.find(anchorName) != std::string::npos) {
            const std::vector<std::string> anchorArgs = ce::getArgs(sub);
            // add all anchor args to untouchables
            for (std::size_t k = 0; k < anchorArgs.size(); ++k) {
                untouchables.insert(anchorArgs[k]);
            }
            if (anchorArgs.size() > 1) zeroArgName = anchorArgs[1];
            if (anchorArgs.size() > 3) sName = anchorArgs[3];
        }
        else {
            chain.push_back(sub);
        }
    }

    // remaining_args_key: args from all but the last element of chain that intersect untouchables
    std::set<std::string> remainingArgsKey;
    if (chain.size() >= 1U) {
        for (std::size_t elementIndex = 0; elementIndex + 1 < chain.size(); ++elementIndex) {
            const std::vector<std::string> elementArgs = ce::getArgs(chain[elementIndex]);
            for (std::size_t a = 0; a < elementArgs.size(); ++a) {
                if (untouchables.find(elementArgs[a]) != untouchables.end()) {
                    remainingArgsKey.insert(elementArgs[a]);
                }
            }
        }
    }

    // prefix all untouchables with "u_" inside chain
    if (!untouchables.empty()) {
        std::map<std::string, std::string> replMap2;
        for (std::set<std::string>::const_iterator it = untouchables.begin();
            it != untouchables.end(); ++it) {
            replMap2[*it] = std::string("u_") + *it;
        }
        for (std::size_t i = 0; i < chain.size(); ++i) {
            chain[i] = ce::replaceKeysInString(chain[i], replMap2);
        }
    }

    // head is last element; chain are the preceding elements
    assert(!chain.empty());
    const std::string head = chain.back();
    chain.pop_back();

    const std::string implication = reconstructImplication(chain, head);

    return std::make_tuple(implication, chain, head, remainingArgsKey, zeroArgName, sName);
}

// Member of gl::ExpressionAnalyzer
void ExpressionAnalyzer::addTheoremToMemory(const std::string& expr,
                                            BodyOfProves& memory,
                                            int iteration,
                                            bool proved,
                                            Dependencies& dependencyTable) {
    // 1) Disintegrate implication -> chain (left nodes) + head (rightmost)
    std::vector< std::tuple<
        std::string,                  // left expression
        std::vector<std::string>,     // node args
        std::set<std::string>         // left-node arguments
    > > tempChain;
    const std::string head = ce::disintegrateImplication(expr, tempChain, this->coreExpressionMap);

    if (expr == "(>[4,5](NaturalNumbers[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,5])(>[10](in2[7,10,4])(>[11](in2[9,11,4])(in3[10,8,11,5])))))")
    {
        int test = 0;
        test++;
    }

    std::vector<std::string> chain;
    chain.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

    // 2) Walk/build the memory path for the chain
    BodyOfProves* memoryBlock = &memory;
    for (std::size_t index = 0; index < chain.size(); ++index) {
        const std::string& element = chain[index];

        std::map<std::string, BodyOfProves*>::iterator it = memoryBlock->simpleMap.find(element);
        if (it != memoryBlock->simpleMap.end() && it->second != NULL) {
            memoryBlock = it->second;
        } else {
            BodyOfProves* child = new BodyOfProves();
            memoryBlock->simpleMap[element] = child;

            child->parentBodyOfProves = memoryBlock;
            child->level = static_cast<int>(index);
            child->exprKey = element;

            this->permanentBodies.push_back(child);

            std::set<int> lv; lv.insert(child->level);
            std::vector<std::string> origin; origin.push_back("task formulation");
            this->addExprToMemoryBlock(element, *child, iteration, 0, lv, origin, -1, -1);

            memoryBlock = child;
        }

        // 3) At the last node: proved or set up recursion/auxiliaries
        if (index == chain.size() - 1) {
            if (proved) {
                this->addExprToMemoryBlock(head, *memoryBlock, iteration, 1, std::set<int>(), std::vector<std::string>(), -1, -1);
            } else {
                // digits / immutables
                const std::set<std::string> digits = ce::findDigitArgs(expr, this->anchorInfo, coreExpressionMap);
                for (std::set<std::string>::const_iterator dit = digits.begin(); dit != digits.end(); ++dit) {
                    const std::string digitArg = *dit;
                    std::set<std::string> immutables = this->findImmutableArgs(expr, digitArg);

                    // recX label for equality block, generic 'rec' for the auxy implication chain
                    const std::string recArg = std::string("rec") + std::to_string(memoryBlock->recursionCounter);
                    

                    std::string auxyImplication;
                    std::vector<std::string> auxyChain;
                    std::string auxyHead;
                    std::set<std::string> remainingArgsKey;
                    std::string zeroArgName;
                    std::string sName;

                    // createAuxyImplication(expr, digitArg, "rec", immutables, anchorName)
                    {
                        std::tuple<
                            std::string,
                            std::vector<std::string>,
                            std::string,
                            std::set<std::string>,
                            std::string,
                            std::string
                        > tpl = createAuxyImplication(expr, digitArg, "rec", immutables, this->anchorInfo.name);

                        auxyImplication  = std::get<0>(tpl);
                        auxyChain        = std::get<1>(tpl);
                        auxyHead         = std::get<2>(tpl);
                        remainingArgsKey = std::get<3>(tpl);
                        zeroArgName      = std::get<4>(tpl);
                        sName            = std::get<5>(tpl);
                    }

                    // --- Recursion block #1 ----------------------------------------------------
                    const std::string tempExpr  = std::string("(in2[") + recArg + "," + digitArg + "," + sName + "])";
                    const std::string tempExpr2 = std::string("(in2[") + "rec"   + "," + digitArg + "," + sName + "])";

                    // ensure unique
                    if (memoryBlock->simpleMap.find(tempExpr) == memoryBlock->simpleMap.end()) {
                        BodyOfProves* tempMb = new BodyOfProves();
                        memoryBlock->simpleMap[tempExpr] = tempMb;

                        tempMb->level = static_cast<int>(index + 1);
                        tempMb->parentBodyOfProves = memoryBlock;
                        tempMb->exprKey = tempExpr;
                        tempMb->isPartOfRecursion = true;

                        // history tag for auxy implication
                        tempMb->exprOriginMap[auxyImplication] = std::vector<std::string>(1, "recursion");
                        tempMb->mailOut.exprOriginMap[auxyImplication] = std::vector<std::string>(1, "recursion");

                        this->permanentBodies.push_back(tempMb);

                        std::set<int> lv;
                        lv.insert(tempMb->level);

                        this->addToHashMemory(auxyChain, auxyHead, remainingArgsKey,
                            tempMb->localMemory, lv,
                            auxyImplication,
                            parameters::inductionMaxAdmissionDepth,
                            parameters::inductionMaxSecondaryNumber,
                            true);


                        // add (in2[rec,...]) to temp block
                        {
                            std::set<int> lv; lv.insert(tempMb->level);
                            std::vector<std::string> or1; or1.push_back("recursion");
                            this->addExprToMemoryBlock(tempExpr2, *tempMb, iteration, 0, lv, or1, -1, -1);
                        }

                        // queue head as "to be proved" with auxy index
                        this->addExprToMemoryBlock(head,
                                                   *tempMb,
                                                   iteration,
                                                   2,
                                                   std::set<int>(),
                                                   std::vector<std::string>(),
							                       -1,     
                                                   dependencyTable.auxyIndex);

                        // update dependency table (original entry collects this auxy)
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex] = DependencyItem();
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex].auxies.insert(dependencyTable.auxyIndex);
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex].expr = expr;
                        dependencyTable.auxyOriginalMap[dependencyTable.auxyIndex] = dependencyTable.originalIndex;
                        dependencyTable.auxyIndex += 1;
                    }

                    // --- Recursion block #2 (equality side-chain) -----------------------------
                    const std::string tempExpr3 = std::string("(=[s(") + recArg + ")," + zeroArgName + "])";
                    const std::string tempExpr4 = std::string("(=[") + digitArg + "," + zeroArgName + "])";

                    if (memoryBlock->simpleMap.find(tempExpr3) == memoryBlock->simpleMap.end()) {
                        BodyOfProves* tempMb2 = new BodyOfProves();
                        memoryBlock->simpleMap[tempExpr3] = tempMb2;

                        tempMb2->level = static_cast<int>(index + 1);
                        tempMb2->parentBodyOfProves = memoryBlock;
                        tempMb2->exprKey = tempExpr4;
                        tempMb2->isPartOfRecursion = true;

                        {
                            std::set<int> lv; lv.insert(tempMb2->level);
                            std::vector<std::string> or2; or2.push_back("recursion");
                            this->addExprToMemoryBlock(tempExpr4, *tempMb2, iteration, 0, lv, or2, -1, -1);
                        }

                        this->addExprToMemoryBlock(head,
                                                   *tempMb2,
                                                   iteration,
                                                   2,
                                                   std::set<int>(),
                                                   std::vector<std::string>(),
                                                   -1,
                                                   dependencyTable.auxyIndex);

                        tempMb2->isActive = false;

                        this->permanentBodies.push_back(tempMb2);

                        dependencyTable.auxyOriginalMap[dependencyTable.auxyIndex] = dependencyTable.originalIndex;
                        dependencyTable.originalAuxyMap[dependencyTable.originalIndex].auxies.insert(dependencyTable.auxyIndex);
                        dependencyTable.originalInductionVariableMap[dependencyTable.originalIndex] = std::make_tuple(digitArg, std::to_string(memoryBlock->recursionCounter));
                        dependencyTable.auxyIndex += 1;
                        dependencyTable.originalIndex += 1;
                        memoryBlock->recursionCounter += 1;
                    }
                }

                // finally mark head as "to be proved" at the original block too
                this->addExprToMemoryBlock(head, *memoryBlock, iteration, 2, std::set<int>(), std::vector<std::string>(), -1, -1);
            }
        }
    }
}


// In analyze_expressions.hpp (declaration)
// void revisitRejected(const std::string& markedExpr, BodyOfProves& memoryBlock);

void ExpressionAnalyzer::revisitRejected(const std::string& markedExpr,
    BodyOfProves& memoryBlock) {
    // Alias to the actual type: map<string, set<RejectedMapValue>>
    std::map<std::string, std::set<RejectedMapValue> >& rm = memoryBlock.localMemory.rejectedMap;

    std::map<std::string, std::set<RejectedMapValue> >::iterator itRM = rm.find(markedExpr);
    if (itRM == rm.end()) {
        return;
    }

    // Snapshot the set so we can iterate safely while mutating memory/mail
    std::set<RejectedMapValue> rejectedExprsCopy = itRM->second;

    // Process each rejected record
    for (std::set<RejectedMapValue>::const_iterator it = rejectedExprsCopy.begin();
        it != rejectedExprsCopy.end(); ++it) {
        const std::string& rejectedExpr = it->renamedExpression; // first in Python tuple
        const std::string& expandedExpr = it->expression;        // second in Python tuple
        const int iteration = it->iteration;         // third in Python tuple

        // Levels for expandedExpr
        std::map<std::string, std::set<int> >::const_iterator itLev =
            memoryBlock.statementLevelsMap.find(expandedExpr);
        if (itLev == memoryBlock.statementLevelsMap.end()) {
            continue; // no levels known -> skip
        }
        const std::set<int>& involvedLevels = itLev->second;

        // Disintegrate expandedExpr
        int newStart = memoryBlock.startInt;
        std::tuple< std::set<std::string>, std::set<std::string>, int > tpl =
            this->disintegrateExpr(expandedExpr,
                newStart,
                iteration,
                memoryBlock.level,
                memoryBlock.localMemory);

        std::set<std::string> implications = std::get<0>(tpl);
        std::set<std::string> statements = std::get<1>(tpl);
        memoryBlock.startInt = std::get<2>(tpl);

        // Local origin (history)
        std::vector<std::string> localOrigin;
        if (parameters::trackHistory) {
            localOrigin.push_back("disintegration");
            localOrigin.push_back(expandedExpr);
        }

        // Handle implications
        for (std::set<std::string>::const_iterator itImp = implications.begin();
            itImp != implications.end(); ++itImp) {

            // Split into key/value
            std::pair<std::string, std::string> kv = ce::extractKeyValue(*itImp, this->coreExpressionMap);
            const std::string& ky = kv.first;
            const std::string& vlue = kv.second;

            // Remaining-args for key and implication
            std::set<std::string> remainingArgsKey = ce::extractDifference(ky);
            std::set<std::string> remainingArgsImpl = ce::extractDifference(*itImp);

            // Replace remaining args inside the implication with "u_<arg>"
            std::map<std::string, std::string> replacementMap;
            for (std::set<std::string>::const_iterator ra = remainingArgsImpl.begin();
                ra != remainingArgsImpl.end(); ++ra) {
                replacementMap[*ra] = std::string("u_") + *ra;
            }
            const std::string replacedImpl = ce::replaceKeysInString(*itImp, replacementMap);

            // Disintegrate replaced implication to get chain + head
            std::vector< std::tuple<
                std::string,
                std::vector<std::string>,
                std::set<std::string>
            > > tempChain;
            const std::string head = ce::disintegrateImplication(replacedImpl, tempChain, this->coreExpressionMap);

            std::vector<std::string> chain;
            chain.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) {
                chain.push_back(std::get<0>(tempChain[i]));
            }

            // Update hash memory
            this->addToHashMemory(chain,
                head,
                remainingArgsKey,
                memoryBlock.localMemory,
                involvedLevels,
                *itImp,
                parameters::standardMaxAdmissionDepth,
                parameters::standardMaxSecondaryNumber,
                false);

            // Emit into mailOut (5-tuple)
            std::set< std::tuple<
                std::vector<std::string>,
                std::string,
                std::set<std::string>,
                std::set<int>,
                std::string
            > >& impSet = memoryBlock.mailOut.implications;

            impSet.insert(std::make_tuple(chain,
                head,
                remainingArgsKey,
                involvedLevels,
                *itImp));

            // History for this implication
            if (parameters::trackHistory) {
                if (memoryBlock.exprOriginMap.find(*itImp) == memoryBlock.exprOriginMap.end()) {
                    memoryBlock.exprOriginMap[*itImp] = localOrigin;
                    memoryBlock.mailOut.exprOriginMap[*itImp] = localOrigin;
                }
            }
        }

        // Handle resulting statements
        for (std::set<std::string>::const_iterator itSt = statements.begin();
            itSt != statements.end(); ++itSt) {
            this->addExprToMemoryBlock(*itSt,
                memoryBlock,
                iteration,
                0,
                involvedLevels,
                localOrigin,
                -1,
                -1);
        }
    }

    // Remove the processed bucket
    rm.erase(itRM);
}


// Remove "u_" prefixes that begin a token inside bracket lists, i.e. after '[' or ','
std::string ExpressionAnalyzer::replaceUSubstrings(const std::string& s) {
    std::string out = s;
    if (out.size() < 2) return out;

    std::size_t i = 1; // we look back at i-1
    while (i + 1 < out.size()) {
        if (out[i] == 'u' && out[i + 1] == '_' &&
            (out[i - 1] == '[' || out[i - 1] == ',')) {
            // erase "u_"
            out.erase(i, 2);
            // do not advance i; check again at the same spot
            continue;
        }
        ++i;
    }
    return out;
}

void ExpressionAnalyzer::checkLocalEncodedMemory(const std::vector<EncodedExpression>& expressionList,
    BodyOfProves& memoryBlock,
    int iteration,
    const NormalizedKey& tple,
    unsigned coreId) {
    // --- pure check: all args with iteration >=0 must be productsOfRecursion
    bool pure = true;
    for (std::size_t i = 0; i < expressionList.size(); ++i) {
        const EncodedExpression& ex = expressionList[i];
        for (std::size_t a = 0; a < ex.arguments.size(); ++a) {
            const std::vector<std::string>& arg = ex.arguments[a];
            if (arg.size() >= 3) {
                const int itNum = std::atoi(arg[2].c_str());
                if (itNum > -1) {
                    const std::string& varName = arg[1];
                    if (memoryBlock.localMemory.productsOfRecursion.find(varName) ==
                        memoryBlock.localMemory.productsOfRecursion.end()) {
                        pure = false;
                        break;
                    }
                }
            }
        }
        if (!pure) break;
    }

    // or_unchangeables
    const std::set<std::string> orUnchangeables = this->extractUnchangeables(expressionList);

    // combined levels across all subexpr originals
    std::set<int> combinedLevels;
    for (std::size_t i = 0; i < expressionList.size(); ++i) {
        const std::string& orig = expressionList[i].original;
        std::map<std::string, std::set<int> >::const_iterator itL =
            memoryBlock.statementLevelsMap.find(orig);
        if (itL != memoryBlock.statementLevelsMap.end()) {
            const std::set<int>& lv = itL->second;
            for (std::set<int>::const_iterator it = lv.begin(); it != lv.end(); ++it) {
                combinedLevels.insert(*it);
            }
        }
    }

    // all args from encoded expressions
    const std::set<std::string> allArgs = this->getAllEncodedArgs(expressionList);

    if (!memoryBlock.localMemory.encodedMap.empty()) {
        // Iterate over remainingArgsNormalizedEncodedMap keys (sets)
        std::map< std::set<std::string>, std::set<NormalizedKey> >::const_iterator itRA =
            memoryBlock.localMemory.remainingArgsNormalizedEncodedMap.begin();
        for (; itRA != memoryBlock.localMemory.remainingArgsNormalizedEncodedMap.end(); ++itRA) {
            const std::set<std::string>& st = itRA->first;

            // check subset: st ⊆ allArgs
            bool subsetOk = true;
            for (std::set<std::string>::const_iterator it = st.begin(); it != st.end(); ++it) {
                if (allArgs.find(*it) == allArgs.end()) {
                    subsetOk = false;
                    break;
                }
            }
            if (!subsetOk) continue;

            // check that tple is in the associated set of normalized keys
            const std::set<NormalizedKey>& nkSet = itRA->second;
            if (nkSet.find(tple) == nkSet.end()) continue;

            // We need to temporarily set unchangeables from 'st', compute normalized key, then reset.
            // Work on a copy to avoid mutating caller's vector.
            std::vector<EncodedExpression> workList = expressionList;

            this->setUnchangeables(workList, st);
            std::pair<NormalizedKey, std::map<std::string, std::string> > normPair =
                this->makeNormalizedEncodedKey(workList, true);
            this->resetUnchangeables(workList, orUnchangeables);

            const NormalizedKey& normalized = normPair.first;
            const std::map<std::string, std::string>& normMap = normPair.second;

            // Build reverse map (value -> key)
            std::map<std::string, std::string> backReplacementMap;
            for (std::map<std::string, std::string>::const_iterator it = normMap.begin();
                it != normMap.end(); ++it) {
                backReplacementMap[it->second] = it->first;
            }

            // If normalized key is present in encodedMap, process its LocalMemoryValue set
            std::map<NormalizedKey, std::set<LocalMemoryValue> >::const_iterator itEnc =
                memoryBlock.localMemory.encodedMap.find(normalized);
            if (itEnc == memoryBlock.localMemory.encodedMap.end()) continue;

            const std::set<LocalMemoryValue>& valueSet = itEnc->second;

            // Copy to vector and sort by .value (to mimic Python "sorted(..., key=item.value)")
            std::vector<LocalMemoryValue> ordered;
            ordered.reserve(valueSet.size());
            for (std::set<LocalMemoryValue>::const_iterator it = valueSet.begin();
                it != valueSet.end(); ++it) {
                ordered.push_back(*it);
            }
            struct ByValue {
                bool operator()(const LocalMemoryValue& a, const LocalMemoryValue& b) const {
                    return a.value < b.value;
                }
            };
            std::sort(ordered.begin(), ordered.end(), ByValue());

            // Build tail for origin: sorted list of originals in expressionList
            std::vector<std::string> originalsTail;
            originalsTail.reserve(expressionList.size());
            for (std::size_t i = 0; i < expressionList.size(); ++i) {
                originalsTail.push_back(expressionList[i].original);
            }
            std::sort(originalsTail.begin(), originalsTail.end());

            for (std::size_t vi = 0; vi < ordered.size(); ++vi) {
                const LocalMemoryValue& lmv = ordered[vi];

                // replace normalized placeholders back to variables
                const std::string rplExpr1 = ce::replaceKeysInString(lmv.value, backReplacementMap);
                const std::string rplExpr2 = ExpressionAnalyzer::replaceUSubstrings(rplExpr1);

                // If NO "marker" present -> add proved statement
                if (lmv.value.find("marker") == std::string::npos) {
                    // tempLevels = combinedLevels ∪ lmv.levels
                    std::set<int> tempLevels = combinedLevels;
                    for (std::set<int>::const_iterator it = lmv.levels.begin();
                        it != lmv.levels.end(); ++it) {
                        tempLevels.insert(*it);
                    }

                    std::vector<std::string> origin;
                    origin.push_back("implication");
                    origin.push_back(lmv.originalImplication);
                    for (std::size_t oi = 0; oi < originalsTail.size(); ++oi) {
                        origin.push_back(originalsTail[oi]);
                    }

                    this->addExprToMemoryBlock(rplExpr2,
                        memoryBlock,
                        iteration,
                        1,
                        tempLevels,
                        origin,
                        coreId,
                        -1);
                }
                else {
                    // marker case: only proceed if pure
                    if (!pure) continue;

                    // Replace back in each element of lmv.key, also strip "u_"
                    std::vector<std::string> replacedKey;
                    replacedKey.reserve(lmv.key.size());
                    for (std::size_t k = 0; k < lmv.key.size(); ++k) {
                        const std::string tmp = ce::replaceKeysInString(lmv.key[k], backReplacementMap);
                        replacedKey.push_back(ExpressionAnalyzer::replaceUSubstrings(tmp));
                    }

                    // Remaining args as a set
                    std::set<std::string> remainingArgs = lmv.remainingArgs;

                    // Add (or create) admissionMap entry
                    std::map<std::string, std::set<AdmissionMapValue> >::iterator itAdm =
                        memoryBlock.localMemory.admissionMap.find(rplExpr2);
                    AdmissionMapValue admv;
                    admv.key = replacedKey;
                    admv.remainingArgs = remainingArgs;
                    admv.standardMaxAdmissionDepth = parameters::standardMaxAdmissionDepth;
                    admv.standardMaxSecondaryNumber = parameters::standardMaxSecondaryNumber;
                    admv.flag = false;

                    if (itAdm != memoryBlock.localMemory.admissionMap.end()) {
                        itAdm->second.insert(admv);
                    }
                    else {
                        std::set<AdmissionMapValue> newSet;
                        newSet.insert(admv);
                        memoryBlock.localMemory.admissionMap.insert(
                            std::make_pair(rplExpr2, newSet));
                    }

                    // admissionStatusMap[rplExpr2] = false
                    memoryBlock.localMemory.admissionStatusMap[rplExpr2] = false;

                    // Revisit any previously rejected items for this marker
                    this->revisitRejected(rplExpr2, memoryBlock);
                }
            }
        }
    }
}


void ExpressionAnalyzer::buildStack(BodyOfProves& memoryBlock,
    const std::string& proved,
    std::vector<std::vector<std::string>>& stack,
    std::set<std::string>& covered) {
    // Lookup origin list for `proved`
    std::map<std::string, std::vector<std::string>>::const_iterator it =
        memoryBlock.exprOriginMap.find(proved);
    if (it == memoryBlock.exprOriginMap.end()) {
        return; // nothing to expand
    }
    const std::vector<std::string>& origin = it->second;

    // Push one row: [proved] + origin
    std::vector<std::string> row;
    row.reserve(1 + origin.size());
    row.push_back(proved);
    row.insert(row.end(), origin.begin(), origin.end());
    stack.push_back(row);

    // Recurse for each ingredient in origin[1:]
    for (std::size_t i = 1; i < origin.size(); ++i) {
        const std::string& ingredient = origin[i];
        // only visit once
        if (covered.insert(ingredient).second) {
            buildStack(memoryBlock, ingredient, stack, covered);
        }
    }
}



std::vector<std::string> gl::ExpressionAnalyzer::sortByValuesDesc(const std::vector<std::string>& strings,
    const std::vector<int>& values) {
    if (strings.size() != values.size()) {
        std::ostringstream oss;
        oss << "got " << strings.size() << " strings but " << values.size() << " values";
        throw std::invalid_argument(oss.str());
    }
    std::vector< std::pair<int, std::string> > paired;
    paired.reserve(strings.size());
    for (std::size_t i = 0; i < strings.size(); ++i) {
        paired.push_back(std::make_pair(values[i], strings[i]));
    }
    // sort by integer descending
    std::sort(paired.begin(), paired.end(),
        [](const std::pair<int, std::string>& a, const std::pair<int, std::string>& b) {
            return a.first > b.first;
        });
    std::vector<std::string> ordered;
    ordered.reserve(paired.size());
    for (std::size_t i = 0; i < paired.size(); ++i) {
        ordered.push_back(paired[i].second);
    }
    return ordered;
}

void gl::ExpressionAnalyzer::findEnds(const std::vector<std::string>& path) {
    namespace fs = std::filesystem;

    // ---- prepare output dir: PROJECT_ROOT/files/raw_proof_graph ----
    std::filesystem::path outDir = std::filesystem::path("files") / "raw_proof_graph";
    if (fs::exists(outDir)) {
        std::error_code ec1;
        fs::remove_all(outDir, ec1);
    }
    std::error_code ec2;
    fs::create_directories(outDir, ec2);

    // ---- navigate to requested memory block ----
    BodyOfProves* memoryBlock = &this->body;
    for (std::size_t i = 0; i < path.size(); ++i) {
        const std::string& elt = path[i];
        std::map<std::string, BodyOfProves*>::iterator it = memoryBlock->simpleMap.find(elt);
        if (it == memoryBlock->simpleMap.end() || it->second == NULL) {
            return; // path invalid
        }
        memoryBlock = it->second;
    }

    // ---- collect candidate ends (keys of exprOriginMap) ----
    std::set<std::string> allExprs;
    std::map<std::string, std::vector<std::string> >::const_iterator itE = memoryBlock->exprOriginMap.begin();
    for (; itE != memoryBlock->exprOriginMap.end(); ++itE) {
        allExprs.insert(itE->first);
    }

    // ---- compute stack sizes for sorting ----
    std::vector<std::string> endsVec;
    endsVec.reserve(allExprs.size());
    std::set<std::string>::const_iterator itS = allExprs.begin();
    for (; itS != allExprs.end(); ++itS) endsVec.push_back(*itS);

    std::vector<int> stackSizes;
    stackSizes.reserve(endsVec.size());
    for (std::size_t i = 0; i < endsVec.size(); ++i) {
        std::vector< std::vector<std::string> > stack;
        std::set<std::string> covered;
        this->buildStack(*memoryBlock, endsVec[i], stack, covered);
        stackSizes.push_back(static_cast<int>(stack.size()));
    }

    // ---- order ends by size (desc) ----
    std::vector<std::string> endsOrdered = this->sortByValuesDesc(endsVec, stackSizes);

    // ---- rebuild globalTheoremList ----
    this->globalTheoremList.clear();
    std::string joinedPath;
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (i > 0) joinedPath.push_back(';');
        joinedPath += path[i];
    }
    for (std::size_t i = 0; i < endsOrdered.size(); ++i) {
        std::string theoremStr;
        if (!joinedPath.empty()) {
            theoremStr = joinedPath + "+" + endsOrdered[i];
        }
        else {
            theoremStr = std::string("+") + endsOrdered[i];
        }
        this->globalTheoremList.push_back(std::make_tuple(theoremStr, std::string("debug"), std::string("-1"), std::string("-1")));
    }

    // ---- write stacks as <index>_debug.txt and mapping file ----
    // mapping file lines: theorem \t method \t var
    std::ofstream mapFile((outDir / "global_theorem_list.txt").string().c_str(), std::ios::out | std::ios::trunc);
    if (!mapFile.is_open()) {
        return;
    }

    for (std::size_t idx = 0; idx < this->globalTheoremList.size(); ++idx) {
        const std::tuple<std::string, std::string, std::string, std::string>& entry = this->globalTheoremList[idx];
        const std::string& theoremStr = std::get<0>(entry);
        const std::string& method = std::get<1>(entry);
        const std::string& var = std::get<2>(entry);

        // parse "a;b;...+END"
        std::string pathPart;
        std::string endExpr;
        std::size_t plusPos = theoremStr.find('+');
        if (plusPos == std::string::npos) {
            pathPart = std::string();
            endExpr = theoremStr;
        }
        else {
            pathPart = theoremStr.substr(0, plusPos);
            endExpr = theoremStr.substr(plusPos + 1);
        }

        // navigate to target block
        BodyOfProves* mb = &this->body;
        if (!pathPart.empty()) {
            std::size_t start = 0;
            while (true) {
                std::size_t pos = pathPart.find(';', start);
                std::string node = (pos == std::string::npos)
                    ? pathPart.substr(start)
                    : pathPart.substr(start, pos - start);
                if (!node.empty()) {
                    std::map<std::string, BodyOfProves*>::iterator itChild = mb->simpleMap.find(node);
                    if (itChild == mb->simpleMap.end() || itChild->second == NULL) {
                        mb = NULL; break;
                    }
                    mb = itChild->second;
                }
                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        }
        if (mb == NULL) continue;

        // build stack
        std::vector< std::vector<std::string> > stack;
        std::set<std::string> covered;
        this->buildStack(*mb, endExpr, stack, covered);

        // write stack to "<index>_debug.txt"
        fs::path filePath = outDir / (std::to_string(static_cast<long long>(idx)) + "_debug.txt");
        std::ofstream ofs(filePath.string().c_str(), std::ios::out | std::ios::trunc);
        if (ofs.is_open()) {
            for (std::size_t r = 0; r < stack.size(); ++r) {
                const std::vector<std::string>& row = stack[r];
                for (std::size_t c = 0; c < row.size(); ++c) {
                    if (c > 0) ofs << '\t';
                    ofs << row[c];
                }
                ofs << '\n';
            }
            ofs.close();
        }

        // record mapping line
        mapFile << theoremStr << '\t' << method << '\t' << var << '\n';
    }
    mapFile.close();
}

void ExpressionAnalyzer::generateRawProofGraph(
    const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& theoremList,
    const std::filesystem::path& outDirParam)
{
    namespace fs = std::filesystem;

    std::cout << "Number proven theorems: "
        << theoremList.size() / 2 << "\n";


    // ---------- out dir: "files/raw_proof_graph" by default ----------
    fs::path outDir = outDirParam.empty() ? (fs::path("files") / "raw_proof_graph") : outDirParam;
    std::error_code ec;
    fs::remove_all(outDir, ec);       // best-effort (ignore if not present)
    fs::create_directories(outDir, ec);

    auto toLower = [](std::string s) {
        for (std::size_t i = 0; i < s.size(); ++i) {
            s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
        }
        return s;
        };

    auto startsWith = [](const std::string& s, const std::string& pref) -> bool {
        return s.size() >= pref.size() && std::equal(pref.begin(), pref.end(), s.begin());
        };
    auto endsWith = [](const std::string& s, const std::string& suf) -> bool {
        return s.size() >= suf.size() && std::equal(s.end() - suf.size(), s.end(), suf.begin());
        };

    auto containsEncoded = [](const std::vector<EncodedExpression>& vec, const std::string& expr) -> bool {
        EncodedExpression needle(expr);
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (vec[i] == needle) return true;
        }
        return false;
        };

    auto writeStackIndexed = [&](int idx, const std::string& part,
        const std::vector<std::vector<std::string>>& stackRows) {
            fs::path fp = outDir / (std::to_string(idx) + "_" + part + ".txt");
            std::ofstream f(fp.c_str());
            for (std::size_t r = 0; r < stackRows.size(); ++r) {
                const std::vector<std::string>& row = stackRows[r];
                for (std::size_t c = 0; c < row.size(); ++c) {
                    if (c) f << '\t';
                    f << row[c];
                }
                f << '\n';
            }
        };

    auto directStack = [&](const std::string& theorem) -> std::vector<std::vector<std::string> > {
        std::vector< std::tuple<
            std::string,                    // leftExpr
            std::vector<std::string>,       // args of the current implication node
            std::set<std::string>           // node->left.arguments
        > > tempChain;

        std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

        std::vector<std::string> chain;
        chain.reserve(tempChain.size());
        for (std::size_t i = 0; i < tempChain.size(); ++i) {
            chain.push_back(std::get<0>(tempChain[i]));
        }

        BodyOfProves* mb = &body;
        for (std::size_t i = 0; i < chain.size(); ++i) {
            std::map<std::string, BodyOfProves*>::iterator it = mb->simpleMap.find(chain[i]);
            if (it == mb->simpleMap.end() || it->second == NULL) return std::vector<std::vector<std::string> >();
            mb = it->second;
        }

        std::vector<std::vector<std::string> > stack;
        std::set<std::string> covered;
        this->buildStack(*mb, head, stack, covered);
        return stack;
        };

    auto checkZeroStack = [&](const std::string& theorem,
        const std::string& inductionVar,
        const std::string& recCounter) -> std::vector<std::vector<std::string> > {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
            std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

            std::vector<std::string> chain;
            chain.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) {
                chain.push_back(std::get<0>(tempChain[i]));
            }

            BodyOfProves* mb = &body;
            for (std::size_t i = 0; i < chain.size(); ++i) {
                std::map<std::string, BodyOfProves*>::iterator it = mb->simpleMap.find(chain[i]);
                if (it == mb->simpleMap.end() || it->second == NULL) return std::vector<std::vector<std::string> >();
                mb = it->second;
            }

            std::vector<std::string> args0 = ce::getArgs(chain[0]);
            if (args0.size() < 2) return std::vector<std::vector<std::string> >();
            const std::string zeroName = args0[1];

            for (std::map<std::string, BodyOfProves*>::iterator it = mb->simpleMap.begin();
                it != mb->simpleMap.end(); ++it) {
                const std::string& key = it->first;
                if (!startsWith(key, std::string("(=[s(rec") + recCounter)
                    || !endsWith(key, std::string(",") + zeroName + "])")) {
                    continue;
                }

                BodyOfProves* eqNode = it->second;
                if (eqNode == NULL) continue;

                if (!containsEncoded(eqNode->localEncodedStatements, head)) continue;

                std::vector<std::string> ev = ce::getArgs(eqNode->exprKey);
                if (ev.empty() || ev[0] != inductionVar) continue;

                std::vector<std::string> keyArgs = ce::getArgs(key);
                if (keyArgs.size() < 2) continue;
                const std::string recName = keyArgs[0];

                const std::string tempExpr = std::string("(=[") + recName + "," + zeroName + "])";
                std::map<std::string, BodyOfProves*>::iterator it2 = mb->simpleMap.find(tempExpr);
                if (it2 == mb->simpleMap.end() || it2->second == NULL) continue;

                BodyOfProves* mbTarget = it2->second;
                std::vector<std::vector<std::string> > stack;
                std::set<std::string> covered;
                this->buildStack(*mbTarget, head, stack, covered);
                return stack;
            }
            return std::vector<std::vector<std::string> >();
        };

    auto checkInductionConditionStack = [&](const std::string& theorem,
        const std::string& inductionVar,
        const std::string& recCounter) -> std::vector<std::vector<std::string> > {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
            std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

            std::vector<std::string> chain;
            chain.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

            BodyOfProves* mb = &body;
            for (std::size_t i = 0; i < chain.size(); ++i) {
                std::map<std::string, BodyOfProves*>::iterator it = mb->simpleMap.find(chain[i]);
                if (it == mb->simpleMap.end() || it->second == NULL) return std::vector<std::vector<std::string> >();
                mb = it->second;
            }

            std::vector<std::string> args0 = ce::getArgs(chain[0]);
            if (args0.size() < 4) return std::vector<std::vector<std::string> >();
            const std::string sName = args0[3];

            for (std::map<std::string, BodyOfProves*>::iterator it = mb->simpleMap.begin();
                it != mb->simpleMap.end(); ++it) {
                const std::string& key = it->first;
                if (!startsWith(key, std::string("(in2[rec") + recCounter)) continue;
                if (!endsWith(key, std::string("") + inductionVar + "," + sName + "])")) continue;

                BodyOfProves* node = it->second;
                if (node == NULL) continue;
                if (!containsEncoded(node->localEncodedStatements, head)) continue;

                std::vector<std::vector<std::string> > stack;
                std::set<std::string> covered;
                this->buildStack(*node, head, stack, covered);
                return stack;
            }
            return std::vector<std::vector<std::string> >();
        };

    auto debugStack = [&](const std::string& pathPlusEnd) -> std::vector<std::vector<std::string> > {
        std::string::size_type pos = pathPlusEnd.find('+');
        if (pos == std::string::npos) return std::vector<std::vector<std::string> >();

        std::string pathPart = pathPlusEnd.substr(0, pos);
        std::string endExpr = pathPlusEnd.substr(pos + 1);

        BodyOfProves* mb = &body;
        if (!pathPart.empty()) {
            std::string token;
            for (std::size_t i = 0; i <= pathPart.size(); ++i) {
                if (i == pathPart.size() || pathPart[i] == ';') {
                    if (!token.empty()) {
                        std::map<std::string, BodyOfProves*>::iterator it = mb->simpleMap.find(token);
                        if (it == mb->simpleMap.end() || it->second == NULL)
                            return std::vector<std::vector<std::string> >();
                        mb = it->second;
                        token.clear();
                    }
                }
                else {
                    token.push_back(pathPart[i]);
                }
            }
        }

        std::vector<std::vector<std::string> > stack;
        std::set<std::string> covered;
        this->buildStack(*mb, endExpr, stack, covered);
        return stack;
        };

    // ---------- emit stacks + mapping file ----------
    std::ofstream mapping((outDir / "global_theorem_list.txt").c_str());
    int idx = 0;

    for (std::size_t i = 0; i < theoremList.size(); ++i) {
        const std::string& name = std::get<0>(theoremList[i]);
        const std::string& methodOrig = std::get<1>(theoremList[i]);
        const std::string& var = std::get<2>(theoremList[i]);
        const std::string& recCtr = std::get<3>(theoremList[i]);

        const std::string method = toLower(methodOrig);

        if (method == "induction") {
            std::vector<std::vector<std::string> > st0 = checkZeroStack(name, var, recCtr);
            writeStackIndexed(idx, "check_zero", st0);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;

            std::vector<std::vector<std::string> > st1 = checkInductionConditionStack(name, var, recCtr);
            writeStackIndexed(idx, "check_induction_condition", st1);
            ++idx;
        }
        else if (method == "direct") {
            std::vector<std::vector<std::string> > st = directStack(name);
            writeStackIndexed(idx, "direct_proof", st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "debug") {
            std::vector<std::vector<std::string> > st = debugStack(name);
            writeStackIndexed(idx, "debug", st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "mirrored statement") {
            std::vector<std::vector<std::string> > st;
            st.push_back(std::vector<std::string>());
            st.back().push_back(name);
            st.back().push_back("mirrored from");
            st.back().push_back(var);
            writeStackIndexed(idx, "mirrored_statement", st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else {
            std::vector<std::vector<std::string> > empty;
            writeStackIndexed(idx, "unknown", empty);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
    }
    mapping.close();
}

static inline std::string mk_in3(long long a, long long b, long long c, char op) {
    return "(in3[i" + std::to_string(a) + ",i" + std::to_string(b) +
        ",i" + std::to_string(c) + "," + op + "])";
}

static inline std::string mk_succ(long long a) {
    return "(in2[i" + std::to_string(a) + ",i" + std::to_string(a + 1) + ",s])";
}

static inline std::string mk_neq(long long a, long long b) {
    return "!(=[i" + std::to_string(a) + ",i" + std::to_string(b) + "])";
}

std::tuple<
    std::vector<std::string>,
    std::vector<std::string>,
    std::vector<std::string>,
    std::vector<std::string>
> ExpressionAnalyzer::makeMPLTables(int n,
    bool print,
    std::function<bool(const std::string&)> print_if) {
    std::vector<std::string> mult, add, succ, neq;
    if (n < 0) return { mult, add, succ, neq };

    auto mk_id = [](char kind, int v) { return std::string(1, kind) + std::to_string(v); };
    auto mk_in3_ijr = [&](char xa, int a, char xb, int b, int c, char op, char rc) {
        return "(in3[" + mk_id(xa, a) + "," + mk_id(xb, b) + "," + mk_id(rc, c) + "," + op + "])";
        };
    auto mk_not = [](const std::string& s) { return "!" + s; };
    auto distinct3 = [&](const std::string& A, const std::string& B, const std::string& C) {
        return A != B && A != C && B != C;
        };
    auto allow_ijr = [](char ka, char kb, char rc) {
        return ((ka == 'j') + (kb == 'j') + (rc == 'j')) <= 1; // at most one 'j' total
        };
    const char kinds[2] = { 'i','j' };

    // Addition: result ∈ [0..n]; inputs i/j; results i and j; filter repeats; ≤1 'j'
    for (int a = 0; a <= n; ++a) {
        for (int b = 0; b <= n; ++b) {
            int s_ab = a + b; if (s_ab > n) continue;
            for (char ka : kinds) for (char kb : kinds) {
                const std::string A = mk_id(ka, a), B = mk_id(kb, b);
                if (allow_ijr(ka, kb, 'i')) {
                    const std::string Ri = mk_id('i', s_ab);
                    if (distinct3(A, B, Ri)) add.emplace_back(mk_in3_ijr(ka, a, kb, b, s_ab, '+', 'i'));
                }
                if (allow_ijr(ka, kb, 'j')) {
                    const std::string Rj = mk_id('j', s_ab);
                    if (distinct3(A, B, Rj)) add.emplace_back(mk_in3_ijr(ka, a, kb, b, s_ab, '+', 'j'));
                }
            }
        }
    }

    // Multiplication: result ∈ [0..n]; inputs i/j; results i and j; filter repeats; ≤1 'j'
    for (int a = 0; a <= n; ++a) {
        for (int b = 0; b <= n; ++b) {
            long long p_ab = 1LL * a * b; if (p_ab > n) continue;
            for (char ka : kinds) for (char kb : kinds) {
                const std::string A = mk_id(ka, a), B = mk_id(kb, b);
                if (allow_ijr(ka, kb, 'i')) {
                    const std::string Ri = mk_id('i', (int)p_ab);
                    if (distinct3(A, B, Ri)) mult.emplace_back(mk_in3_ijr(ka, a, kb, b, (int)p_ab, '*', 'i'));
                }
                if (allow_ijr(ka, kb, 'j')) {
                    const std::string Rj = mk_id('j', (int)p_ab);
                    if (distinct3(A, B, Rj)) mult.emplace_back(mk_in3_ijr(ka, a, kb, b, (int)p_ab, '*', 'j'));
                }
            }
        }
    }

    // Succession: intermix i/j for source and target, but ≤1 'j' per expression
    for (int k = 0; k < n; ++k) {
        for (char ks : kinds) for (char kt : kinds) {
            if (((ks == 'j') + (kt == 'j')) <= 1)
                succ.emplace_back("(in2[" + mk_id(ks, k) + "," + mk_id(kt, k + 1) + ",s])");
        }
    }

    // Inequalities: negated + and *; inputs i/j; results i and j; filter repeats; ≤1 'j'
    for (int a = 0; a <= n; ++a) {
        for (int b = 0; b <= n; ++b) {
            const int sum = a + b;
            const long long prod = 1LL * a * b;
            for (int p = 0; p <= n; ++p) {
                for (char ka : kinds) for (char kb : kinds) {
                    const std::string A = mk_id(ka, a), B = mk_id(kb, b);
                    // + inequality
                    if (p != sum) {
                        if (allow_ijr(ka, kb, 'i')) {
                            const std::string Ri = mk_id('i', p);
                            if (distinct3(A, B, Ri)) neq.emplace_back(mk_not(mk_in3_ijr(ka, a, kb, b, p, '+', 'i')));
                        }
                        if (allow_ijr(ka, kb, 'j')) {
                            const std::string Rj = mk_id('j', p);
                            if (distinct3(A, B, Rj)) neq.emplace_back(mk_not(mk_in3_ijr(ka, a, kb, b, p, '+', 'j')));
                        }
                    }
                    // * inequality
                    if ((long long)p != prod) {
                        if (allow_ijr(ka, kb, 'i')) {
                            const std::string Ri = mk_id('i', p);
                            if (distinct3(A, B, Ri)) neq.emplace_back(mk_not(mk_in3_ijr(ka, a, kb, b, p, '*', 'i')));
                        }
                        if (allow_ijr(ka, kb, 'j')) {
                            const std::string Rj = mk_id('j', p);
                            if (distinct3(A, B, Rj)) neq.emplace_back(mk_not(mk_in3_ijr(ka, a, kb, b, p, '*', 'j')));
                        }
                    }
                }
            }
        }
    }

    if (print) {
        auto print_list = [&](const char* title, const std::vector<std::string>& v) {
            std::cout << title << " (" << v.size() << ")\n";
            if (!print_if) for (const auto& s : v) std::cout << "  " << s << '\n';
            else           for (const auto& s : v) if (print_if(s)) std::cout << "  " << s << '\n';
            };
        print_list("Multiplication", mult);
        print_list("Addition", add);
        print_list("Succession", succ);
        print_list("Inequalities (negated +,*)", neq);
        std::cout.flush();
    }

    return { mult, add, succ, neq };
}





void ExpressionAnalyzer::addConjectureForCEFiltering(const std::string& conjecture,
    const std::tuple<
    std::vector<std::string>,
    std::vector<std::string>,
    std::vector<std::string>,
    std::vector<std::string>
    >& tables,
    int cIndex) {
    // Create or reuse LB0 under ceBody
    BodyOfProves* lb0 = nullptr;
    {
        std::map<std::string, BodyOfProves*>::iterator it = ceBody.simpleMap.find(conjecture);
        if (it != ceBody.simpleMap.end() && it->second) {
            lb0 = it->second;
        }
        else {
            lb0 = new BodyOfProves();
            ceBody.simpleMap[conjecture] = lb0;
            lb0->parentBodyOfProves = &ceBody;
            lb0->level = 0;
            lb0->exprKey = conjecture;
			lb0->contradictionIndex = cIndex;
            permanentBodiesCE.push_back(lb0);
        }
    }

    const std::vector<std::string>& mult = std::get<0>(tables);
    const std::vector<std::string>& add = std::get<1>(tables);
    const std::vector<std::string>& succ = std::get<2>(tables);
    const std::vector<std::string>& neq = std::get<3>(tables);

    // Origins and levels
    std::vector<std::string> origin;
    if (parameters::trackHistory) {
        origin.push_back("CE_building_block");
    }
    const std::set<int> lvl0{ 0 };

    // LB0: +, *, s
    for (const auto& s : mult) this->addExprToMemoryBlock(s, *lb0, 0, 1, lvl0, origin, -1, -1);
    for (const auto& s : add)  this->addExprToMemoryBlock(s, *lb0, 0, 1, lvl0, origin, -1, -1);
    for (const auto& s : succ) this->addExprToMemoryBlock(s, *lb0, 0, 1, lvl0, origin, -1, -1);
    for (const auto& s : neq)  this->addExprToMemoryBlock(s, *lb0, 0, 1, lvl0, origin, -1, -1);
    this->addExprToMemoryBlock("(NaturalNumbers[N,i0,i1,s,+,*])", *lb0, 1, 3, lvl0, origin, -1, -1);
    this->addExprToMemoryBlock("(NaturalNumbers[N,i0,j1,s,+,*])", *lb0, 1, 3, lvl0, origin, -1, -1);
    this->addExprToMemoryBlock("(NaturalNumbers[N,j0,i1,s,+,*])", *lb0, 1, 3, lvl0, origin, -1, -1);
    this->addExprToMemoryBlock("(NaturalNumbers[N,j0,j1,s,+,*])", *lb0, 1, 3, lvl0, origin, -1, -1);

    // Store conjecture in the LB0 hash memory
    const std::map<std::string, std::string> replacementMap{
    {"1","N"}, {"2","i0"}, {"3","i1"}, {"4","s"}, {"5","+"}, {"6","*"}
    };
    std::string replacedConjecture = ce::replaceKeysInString(conjecture, replacementMap);


    // 1) Disintegrate implication -> chain (left nodes) + head (rightmost)
    std::vector< std::tuple<
        std::string,                  // left expression
        std::vector<std::string>,     // node args
        std::set<std::string>         // left-node arguments
    > > tempChain;
    const std::string head = ce::disintegrateImplication(replacedConjecture, tempChain, this->coreExpressionMap);


    std::vector<std::string> chain;
    chain.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));



    this->addToHashMemory(chain,
        head,
        std::set<std::string>{},
        lb0->localMemory,
        lvl0,
        replacedConjecture,
        parameters::standardMaxAdmissionDepth,
        parameters::standardMaxSecondaryNumber,
        false);
}

void ExpressionAnalyzer::releaseCEBatchMemory() {
    // ---- 0) Defensive: clear tiny per-iteration queues (cheap, may carry strings) ----
    inductionMemoryBlocks.clear();
    std::vector<BodyOfProves*>().swap(inductionMemoryBlocks);

    updateGlobalTuples.clear();
    std::vector<std::tuple<int, bool, int>>().swap(updateGlobalTuples);

    updateGlobalDirectTuples.clear();
    std::vector<std::tuple<std::string, int>>().swap(updateGlobalDirectTuples);

    // ---- 1) Collect ALL potential CE roots from every holder ----
    std::vector<BodyOfProves*> roots;
    roots.reserve(ceBody.simpleMap.size()
        + permanentBodiesCE.size()
        + indexCE.size() * 2);

    // from ceBody root map
    for (std::map<std::string, BodyOfProves*>::iterator it = ceBody.simpleMap.begin();
        it != ceBody.simpleMap.end(); ++it) {
        if (it->second) roots.push_back(it->second);
    }
    // from permanentBodiesCE
    for (std::size_t i = 0; i < permanentBodiesCE.size(); ++i)
        if (permanentBodiesCE[i]) roots.push_back(permanentBodiesCE[i]);
    // from indexCE keys + values (covers any stray node not attached under ceBody)
    for (ParentChildrenMap::const_iterator it = indexCE.begin(); it != indexCE.end(); ++it) {
        if (it->first) roots.push_back(it->first);
        const std::vector<BodyOfProves*>& vs = it->second;
        for (std::size_t j = 0; j < vs.size(); ++j)
            if (vs[j]) roots.push_back(vs[j]);
    }

    // ---- 2) Iterative DFS to delete each node exactly once ----
    std::unordered_set<BodyOfProves*> seen;
    std::vector<BodyOfProves*> stack;
    stack.reserve(roots.size());
    for (BodyOfProves* r : roots) if (r) stack.push_back(r);

    while (!stack.empty()) {
        BodyOfProves* node = stack.back(); stack.pop_back();
        if (!node || !seen.insert(node).second) continue;

        for (std::map<std::string, BodyOfProves*>::iterator it = node->simpleMap.begin();
            it != node->simpleMap.end(); ++it)
            if (it->second) stack.push_back(it->second);

        delete node; // d-tor frees STL members
    }

    // ---- 3) Drop/compact all CE containers & maps (release capacity) ----
    // ceBody’s root map
    std::map<std::string, BodyOfProves*>().swap(ceBody.simpleMap);

    // explicit reset of ceBody's heavy members via swap-with-empty
    {
        BodyOfProves empty;
        using std::swap;
        swap(ceBody.startInt, empty.startInt);
        swap(ceBody.toBeProved, empty.toBeProved);
        swap(ceBody.encodedStatements, empty.encodedStatements);
        swap(ceBody.statementLevelsMap, empty.statementLevelsMap);
        swap(ceBody.exprKey, empty.exprKey);
        ceBody.parentBodyOfProves = nullptr;
        swap(ceBody.level, empty.level);
        swap(ceBody.localMemory, empty.localMemory);
        swap(ceBody.equivalenceClasses, empty.equivalenceClasses);
        swap(ceBody.localEncodedStatements, empty.localEncodedStatements);
        swap(ceBody.localEncodedStatementsDelta, empty.localEncodedStatementsDelta);
        swap(ceBody.mailIn, empty.mailIn);
        swap(ceBody.mailOut, empty.mailOut);
        swap(ceBody.wholeExpressions, empty.wholeExpressions);
        swap(ceBody.eqClassSttmntIndexMap, empty.eqClassSttmntIndexMap);
        swap(ceBody.isActive, empty.isActive);
        swap(ceBody.isPartOfRecursion, empty.isPartOfRecursion);
        swap(ceBody.deltaNumberStatements, empty.deltaNumberStatements);
        swap(ceBody.exprOriginMap, empty.exprOriginMap);
        swap(ceBody.recursionCounter, empty.recursionCounter);
        swap(ceBody.contradictionIndex, empty.contradictionIndex);
    }

    // CE containers
    destroyParentChildrenMap(indexCE);
    destroyMailboxes(boxesCE);
    std::vector<BodyOfProves*>().swap(permanentBodiesCE);

}

void ExpressionAnalyzer::destroyParentChildrenMap(ParentChildrenMap& m) {
    for (ParentChildrenMap::iterator it = m.begin(); it != m.end(); ++it)
        std::vector<BodyOfProves*>().swap(it->second);
    m.clear();
    m.rehash(0);
}

void ExpressionAnalyzer::destroyMailboxes(PerCoreMailboxes& boxes) {
    for (PerCoreMailboxes::iterator it = boxes.begin(); it != boxes.end(); ++it) {
        std::vector<Mail>& slots = it->second;
        for (std::size_t i = 0; i < slots.size(); ++i) {
            Mail& mb = slots[i];
            mb.statements.clear();
            mb.implications.clear();
            mb.exprOriginMap.clear();
            Mail empty; using std::swap; swap(mb, empty); // free capacity
        }
        std::vector<Mail>().swap(slots);
    }
    boxes.clear();
    boxes.rehash(0);
}




#if 0
std::vector<std::string> ExpressionAnalyzer::filterConjecturesWithCE(const std::vector<std::string>& conjectures) {

    
    const auto tables = this->makeMPLTables(parameters::counterExampleBoundary, /*print=*/false);

	// Add each conjecture under ceBody
    for (int cIndex = 0; cIndex < conjectures.size(); cIndex++) {
        contradictionTable.push_back(ContradictionItem(conjectures[cIndex], false));

        this->addConjectureForCEFiltering(conjectures[cIndex], tables, cIndex);
    }

    // Build parent->children index once
    indexCE = buildParentChildrenMap(permanentBodiesCE);

    // Prepare per-core mailboxes sized to logical cores
    boxesCE = buildPerCoreMailboxes(indexCE);

    this->prove(parameters::numberIterationsConjectureFiltering, permanentBodiesCE, indexCE, boxesCE);

#if 0
	std::cout << "Filtering results:\n";
    for (int cIndex = 0; cIndex < conjectures.size(); cIndex++)
    {
        if (!contradictionTable[cIndex].successful) {
            std::cout << contradictionTable[cIndex].expr << std::endl;
        }
    }
#endif //#if 0

	std::vector<std::string> filteredConjectures;
    for (int cIndex = 0; cIndex < conjectures.size(); cIndex++)
    {
        if (!contradictionTable[cIndex].successful) {
            filteredConjectures.push_back(conjectures[cIndex]);
        }
    }

	return filteredConjectures;
}
#endif //#if 0

std::vector<std::string> ExpressionAnalyzer::filterConjecturesWithCE(
    const std::vector<std::string>& conjectures)
{
    // Build once for all batches
    const auto tables =
        this->makeMPLTables(parameters::counterExampleBoundary, /*print=*/false);

    // Keep contradiction indices stable across batches
    contradictionTable.reserve(contradictionTable.size() + conjectures.size());
    for (int i = 0; i < static_cast<int>(conjectures.size()); ++i)
        contradictionTable.push_back(ContradictionItem(conjectures[i], false));

    const unsigned batchSize = std::max(1u, logicalCores);

    for (std::size_t start = 0; start < conjectures.size(); start += batchSize) {
        const std::size_t end = std::min(start + batchSize, conjectures.size());

        // Build CE memory for THIS batch only
        for (std::size_t i = start; i < end; ++i) {
            const int cIndex = static_cast<int>(i);
            this->addConjectureForCEFiltering(conjectures[i], tables, cIndex);
        }

        // Parent->children index and mailboxes for THIS batch
        indexCE = buildParentChildrenMap(permanentBodiesCE);
        boxesCE = buildPerCoreMailboxes(indexCE);

        // Run the prover on this batch
        this->prove(parameters::numberIterationsConjectureFiltering,
            permanentBodiesCE, indexCE, boxesCE);

        // Release everything created for THIS batch
        this->releaseCEBatchMemory();
    }

    // Extra explicit cleanup “as if between batches” (no-op if already clean)
    this->releaseCEBatchMemory();

    // Keep non-contradictory conjectures
    std::vector<std::string> filtered;
    filtered.reserve(conjectures.size());
    for (int i = 0; i < static_cast<int>(conjectures.size()); ++i)
        if (!contradictionTable[i].successful) filtered.push_back(conjectures[i]);

    return filtered;
}







using gl::ParentChildrenMap;
using gl::PerCoreMailboxes;

ParentChildrenMap ExpressionAnalyzer::buildParentChildrenMap(const std::vector<BodyOfProves*>& roots) const {
    // 1) Collect reachable nodes and direct children
    std::unordered_map<BodyOfProves*, std::vector<BodyOfProves*>> direct;
    std::unordered_set<BodyOfProves*> seen;
    std::vector<BodyOfProves*> stack;
    for (BodyOfProves* r : roots) if (r) stack.push_back(r);

    while (!stack.empty()) {
        BodyOfProves* cur = stack.back(); stack.pop_back();
        if (!cur || !seen.insert(cur).second) continue;

        std::vector<BodyOfProves*> kids;
        kids.reserve(cur->simpleMap.size());
        for (std::map<std::string, BodyOfProves*>::const_iterator it = cur->simpleMap.begin();
            it != cur->simpleMap.end(); ++it) {
            BodyOfProves* ch = it->second;
            if (!ch) continue;
            kids.push_back(ch);
            stack.push_back(ch);
        }
        direct[cur] = std::move(kids);
    }

    // 2) Transitive closure with memoization; cycle-safe
    std::unordered_map<BodyOfProves*, std::unordered_set<BodyOfProves*>> memo;
    std::unordered_set<BodyOfProves*> visiting;

    std::function<const std::unordered_set<BodyOfProves*>& (BodyOfProves*)> dfs =
        [&](BodyOfProves* node) -> const std::unordered_set<BodyOfProves*>&{
        std::unordered_map<BodyOfProves*, std::unordered_set<BodyOfProves*>>::iterator mit = memo.find(node);
        if (mit != memo.end()) return mit->second;

        if (!visiting.insert(node).second) {
            static const std::unordered_set<BodyOfProves*> empty;
            return empty; // break cycles defensively
        }

        std::unordered_set<BodyOfProves*> acc;
        std::vector<BodyOfProves*>& kids = direct[node]; // default-constructed empty if missing
        for (std::size_t i = 0; i < kids.size(); ++i) {
            BodyOfProves* ch = kids[i];
            if (!ch) continue;
            acc.insert(ch);
            const std::unordered_set<BodyOfProves*>& sub = dfs(ch);
            acc.insert(sub.begin(), sub.end());
        }

        visiting.erase(node);
        return memo.emplace(node, std::move(acc)).first->second;
        };

    for (std::unordered_map<BodyOfProves*, std::vector<BodyOfProves*>>::const_iterator it = direct.begin();
        it != direct.end(); ++it) {
        dfs(it->first);
    }

    // 3) Convert sets to vectors
    ParentChildrenMap out;
    out.reserve(memo.size());
    for (std::unordered_map<BodyOfProves*, std::unordered_set<BodyOfProves*>>::const_iterator it = memo.begin();
        it != memo.end(); ++it) {
        const std::unordered_set<BodyOfProves*>& s = it->second;
        std::vector<BodyOfProves*> v; v.reserve(s.size());
        for (std::unordered_set<BodyOfProves*>::const_iterator sit = s.begin(); sit != s.end(); ++sit)
            v.push_back(*sit);
        out.emplace(it->first, std::move(v));
    }

    return out;
}


PerCoreMailboxes ExpressionAnalyzer::buildPerCoreMailboxes(const ParentChildrenMap& index) const {
    unsigned cores = logicalCores;

    // Recipients are all values in the index
    std::unordered_set<BodyOfProves*> recipients;
    for (const auto& kv : index) 
        for (BodyOfProves* ch : kv.second) 
            if (ch) recipients.insert(ch);

    PerCoreMailboxes boxes;
    for (BodyOfProves* b : recipients) 
        boxes[b] = std::vector<Mail>(cores);
    return boxes;
}

void ExpressionAnalyzer::smashMail(PerCoreMailboxes& boxes) const {
    for (auto& kv : boxes) {
        BodyOfProves* body = kv.first;
        if (!body) continue;

        bool any = false;
        std::vector<Mail>& slots = kv.second;

        for (Mail& m : slots) {
            if (!m.statements.empty() || !m.implications.empty() || !m.exprOriginMap.empty()) any = true;

            body->mailIn.statements.insert(m.statements.begin(), m.statements.end());
            body->mailIn.implications.insert(m.implications.begin(), m.implications.end());

            // rhs wins
            for (const auto& kv2 : m.exprOriginMap) body->mailIn.exprOriginMap[kv2.first] = kv2.second;

            m.statements.clear();
            m.implications.clear();
            m.exprOriginMap.clear();
        }
        
    }
}

void ExpressionAnalyzer::activateZeroCondition(BodyOfProves& memoryBlock)
{
    std::vector<std::string> args0 = ce::getArgs(memoryBlock.exprKey);
    std::string zeroArgName = this->findZeroArgName(memoryBlock);
    std::string eqExpr = std::string("(=[s(") + args0[0] + ")," + zeroArgName + "])";
    if (memoryBlock.parentBodyOfProves != NULL) {
        std::map<std::string, BodyOfProves*>::iterator itChild =
            memoryBlock.parentBodyOfProves->simpleMap.find(eqExpr);
        if (itChild != memoryBlock.parentBodyOfProves->simpleMap.end() && itChild->second) {
            itChild->second->isActive = true;
        }
    }
}

void ExpressionAnalyzer::proveKernel(const std::vector<BodyOfProves*>& bodies,
    const ParentChildrenMap& index,
    PerCoreMailboxes& boxes) {

    if (bodies.empty()) return;

    const unsigned workers = logicalCores;

    std::atomic<std::size_t> next{ 0 };

    auto worker = [this, &bodies, &index, &boxes, &next, workers](unsigned coreId) {
        const unsigned cid = workers ? (coreId % workers) : 0U;
        for (;;) {
            std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= bodies.size()) break;

            BodyOfProves* b = bodies[i];
            if (!b || !b->isActive) continue;

            this->performElementaryLogicalStep(*b, cid, index, boxes);
        }
        };

    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned t = 0; t < workers; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    this->smashMail(boxes);  // activate recipients for next iteration

    for (int blockIndex = 0; blockIndex < inductionMemoryBlocks.size(); blockIndex++)
    {
        BodyOfProves* memoryBlock = inductionMemoryBlocks[blockIndex];
        this->activateZeroCondition(*memoryBlock);
    }

    for (const std::tuple<int, bool, int>& t : updateGlobalTuples) 
    {
        updateGlobal(std::get<0>(t), std::get<1>(t), std::get<2>(t));
    }

    for (const std::tuple<std::string, int>& t : updateGlobalDirectTuples) 
    {
        updateGlobalDirect(std::get<0>(t), std::get<1>(t));
    }



	inductionMemoryBlocks.clear();
	updateGlobalTuples.clear();
	updateGlobalDirectTuples.clear();
}













void ExpressionAnalyzer::prove(int numberIterations,
    const std::vector<BodyOfProves*>& bodies,
    const ParentChildrenMap& index,
    PerCoreMailboxes& boxes) {

    for (int it = 0; it < numberIterations; ++it) {
        std::cout << "Hash burst: " << it << std::endl;
        proveKernel(bodies, index, boxes);                     // runs, then smashMail(boxes)
    }
}





void ExpressionAnalyzer::analyzeExpressions(const std::vector<std::string>& theorems) {

    auto t0 = std::chrono::high_resolution_clock::now();
	std::cout << "Counter example based filtering started." << std::endl;

#if 1
    std::vector<std::string> filteredConjectures = filterConjecturesWithCE(theorems);

    // Add theorems (constructor already set up globals and maps)
    for (std::size_t i = 0; i < filteredConjectures.size(); ++i) {
        this->addTheoremToMemory(filteredConjectures[i], this->body, 0, false, this->globalDependencies);
    }
#else
    for (std::size_t i = 0; i < theorems.size(); ++i) {
        this->addTheoremToMemory(theorems[i], this->body, 0, false, this->globalDependencies);
    }
#endif // #if

    std::cout << "Counter example based filtering finished." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "Runtime counter example based filtering: " << dt.count() << " seconds\n";

	//std::cout << permanentBodies.size() << " permanent bodies in main memory." << std::endl;

    // Run prove() and time it
    t0 = std::chrono::high_resolution_clock::now();

    // Build parent->children index once
    index = buildParentChildrenMap(permanentBodies);

    // Prepare per-core mailboxes sized to logical cores
    boxes = buildPerCoreMailboxes(index);

    std::cout << "Prover started." << std::endl;

    this->prove(parameters::maxIterationNumberProof, permanentBodies, index, boxes);
    t1 = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration<double>(t1 - t0);

    std::cout << "Prover finished." << std::endl;


    std::cout << "Runtime prover: " << dt.count() << " seconds" << std::endl;
}











} // namespace gl
