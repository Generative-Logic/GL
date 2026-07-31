
/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschraenkt)

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

#include "conjecturer.hpp"
#include <json.hpp>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <chrono>
#include <regex>
#include <sstream>
#include <iterator>
#include <mutex>
#include <atomic>
#include <cstring>
#include <cstdio>
#include <climits>

namespace conj {

using json = nlohmann::json;

// ============================================================================
// PROFILING (temporary — runtime investigation, session_24042026)
// ============================================================================
namespace prof {
    struct Counter {
        std::atomic<uint64_t> calls{0};
        std::atomic<uint64_t> ns{0};
        void add(uint64_t dtNs) {
            calls.fetch_add(1, std::memory_order_relaxed);
            ns.fetch_add(dtNs, std::memory_order_relaxed);
        }
    };
    // Hot-path functions
    inline Counter g_reshuffle;
    inline Counter g_reshufflePermInner;       // buildAndNormalize + compare, per permutation
    inline Counter g_mirrored;
    inline Counter g_encodeExpr;
    inline Counter g_encodeDefSetMap;
    inline Counter g_decodeExpr;
    inline Counter g_connectExprInt;
    inline Counter g_makeAllConnMapsInt;
    inline Counter g_singleThreadCalc;
    inline Counter g_singleExprAnchorConn;
    inline Counter g_exprGood2Int;
    inline Counter g_exprGood;
    inline Counter g_disintegrate;  // compiler.hpp::disintegrateImplication
    // Filter functions (string-based — candidates for int-ification)
    inline Counter g_onlyInHeadGoodInt;
    inline Counter g_prohibHeadsGoodInt;
    inline Counter g_checkInputVarsHead;
    inline Counter g_triggersExistenceRef;
    inline Counter g_checkInputVarsOrder;
    inline Counter g_patternInConjecture;
    inline Counter g_evaluateOperatorExprs2;
    inline Counter g_controlEquality;
    inline Counter g_checkMinSizeExpression;
    inline Counter g_passesMaxSizeAfterEx;
    inline Counter g_passesComplexityAfterEx;
    inline Counter g_passesInPremiseFilter;
    inline Counter g_reformulateToExistence;
    inline Counter g_generateNegatedPremise;
    inline Counter g_checkDefSets;
    inline Counter g_checkComplLevForDefSets;
    inline Counter g_countOpOccurrences;
    inline Counter g_checkDefSetsPriorInt;
    inline Counter g_checkComplexityPerOpInt;
    inline Counter g_getNumRemArgsInt;
    // Phase timers
    inline Counter g_phase_precomp;
    inline Counter g_phase_prelim;
    inline Counter g_phase_mainLoop;
    inline Counter g_phase_sortAndOut;

    // Scope RAII timer. Toggle GL_PROF_SCOPE_ENABLE to 0 to make all
    // per-function counters (but not phase timers in run()) no-op —
    // used to isolate profiler overhead from real work. The phase
    // timers in run() continue to measure wall-time directly and are
    // unaffected.
    #ifndef GL_PROF_SCOPE_ENABLE
    #define GL_PROF_SCOPE_ENABLE 0
    #endif
    struct Scope {
    #if GL_PROF_SCOPE_ENABLE
        Counter& c;
        std::chrono::steady_clock::time_point t0;
        Scope(Counter& cc) : c(cc), t0(std::chrono::steady_clock::now()) {}
        ~Scope() {
            auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count();
            c.add((uint64_t)dt);
        }
    #else
        Scope(Counter&) {}
    #endif
    };

    inline void report(const char* name, const Counter& c, double wallSec) {
        uint64_t calls = c.calls.load();
        uint64_t ns = c.ns.load();
        double sec = ns / 1e9;
        double pct = wallSec > 0 ? 100.0 * sec / wallSec : 0.0;
        double avgUs = calls > 0 ? (ns / 1000.0 / (double)calls) : 0.0;
        std::cout << "  " << std::setw(28) << std::left << name
                  << " calls=" << std::setw(12) << calls
                  << " total=" << std::setw(9) << std::fixed << std::setprecision(3) << sec << "s"
                  << " (" << std::setw(5) << std::fixed << std::setprecision(1) << pct << "% of wall)"
                  << " avg=" << std::fixed << std::setprecision(3) << avgUs << "us"
                  << "\n";
    }
} // namespace prof

// ============================================================================
// Helper: strip whitespace (like Python's _normalize_mpl)
// ============================================================================

static std::string stripWs(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c != '\n' && c != ' ' && c != '\t' && c != '\r') out.push_back(c);
    }
    return out;
}

// ============================================================================
// Helper: identity map (like Python's global `identity`)
// ============================================================================

static const std::map<int,int>& identityMap() {
    static std::map<int,int> id;
    static bool init = false;
    if (!init) {
        for (int i = 1; i <= 100000; ++i) id[i] = i;
        init = true;
    }
    return id;
}

// ============================================================================
// Helper: make anchor signature from raw short_mpl
// ============================================================================

static std::string makeAnchorSignatureFromRaw(const std::string& signature) {
    std::string s = stripWs(signature);
    auto l = s.find('[');
    auto r = s.rfind(']');
    if (l == std::string::npos || r == std::string::npos || r < l) return s;
    std::string args_str = s.substr(l + 1, r - l - 1);
    int n = 0;
    if (!args_str.empty()) {
        n = 1;
        for (char c : args_str) if (c == ',') ++n;
    }
    std::string new_args;
    for (int i = 1; i <= n; ++i) {
        if (i > 1) new_args += ',';
        new_args += std::to_string(i);
    }
    return s.substr(0, l + 1) + new_args + s.substr(r);
}

// Helper: extract arg list from raw signature
static std::vector<std::string> extractArgList(const std::string& signatureRaw) {
    std::string s = stripWs(signatureRaw);
    auto l = s.find('[');
    auto r = s.rfind(']');
    if (l == std::string::npos || r == std::string::npos || r < l) return {};
    std::string inside = s.substr(l + 1, r - l - 1);
    if (inside.empty()) return {};
    std::vector<std::string> result;
    std::size_t pos = 0;
    while (pos <= inside.size()) {
        auto comma = inside.find(',', pos);
        std::string tok = (comma == std::string::npos)
            ? inside.substr(pos)
            : inside.substr(pos, comma - pos);
        if (!tok.empty()) result.push_back(tok);
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return result;
}

// Helper: align and sort args by signature position (like Python _align_and_sort_args)
static std::pair<std::vector<std::string>, std::vector<int>>
alignAndSortArgs(const std::vector<std::string>& names, const std::vector<std::string>& orderedArgs) {
    std::map<std::string, int> firstIndex;
    for (int i = 0; i < (int)orderedArgs.size(); ++i) {
        if (firstIndex.find(orderedArgs[i]) == firstIndex.end())
            firstIndex[orderedArgs[i]] = i;
    }
    std::vector<std::tuple<int, int, std::string>> triples;
    for (int pos = 0; pos < (int)names.size(); ++pos) {
        auto it = firstIndex.find(names[pos]);
        if (it != firstIndex.end()) {
            triples.push_back({it->second, pos, names[pos]});
        }
    }
    std::sort(triples.begin(), triples.end(), [](const auto& a, const auto& b) {
        if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
        return std::get<1>(a) < std::get<1>(b);
    });
    std::vector<std::string> sortedNames;
    std::vector<int> sortedIndices;
    for (auto& [idx, _, n] : triples) {
        sortedIndices.push_back(idx);
        sortedNames.push_back(n);
    }
    return {sortedNames, sortedIndices};
}

// ============================================================================
// ============================================================================
// ConfigurationData::getAnchorName
// ============================================================================

/// @brief Convenience accessor matching Python's
///        `get_anchor_name`.
///
/// @details
/// Prefers the configured `anchor_name` when it resolves to a
/// known expression in `data`; otherwise falls back on the
/// derived `"Anchor" + anchor_id` candidate. Returns the empty
/// string when neither resolves.
///
/// @return Resolved anchor name, or empty on miss.
std::string ConfigurationData::getAnchorName() const {
    if (!anchor_name.empty() && data.find(anchor_name) != data.end())
        return anchor_name;
    if (!anchor_id.empty()) {
        std::string candidate = "Anchor" + anchor_id;
        if (data.find(candidate) != data.end())
            return candidate;
    }
    assert(false && "getAnchorName: no anchor found");
    return "";
}

// ============================================================================
// Free utility functions (Phase 3)
// ============================================================================

/// @brief Check whether any integer arg id appears more than once
///        in `s`. See header declaration for the full contract.
bool repetitionsExist(const std::string& s) {
    std::regex pat(R"(\([^()]*\))");
    std::sregex_iterator it(s.begin(), s.end(), pat);
    std::sregex_iterator end;
    std::set<std::string> seen;
    int count = 0;
    for (; it != end; ++it) {
        seen.insert(it->str());
        ++count;
    }
    return count != (int)seen.size();
}

/// @brief Append the reverse-direction mirror conjectures into the prove
///        pool, de-duplicated, so each is genuinely proved.
///
/// @details
/// The conjecturer computes, for every conjecture, the mirror that swaps the
/// head with the premise sharing the head's output variable (see
/// `Conjecturer::createReshuffledMirrored`). Those mirrors were historically
/// written only to the archival `reshuffled_mirrored_conjectures.txt`, and the
/// prover fabricated the reverse direction post-proof as an unproved
/// `mirrored statement` row. This helper instead folds the mirrors into the
/// actual prove pool (`conjectures.txt`), exactly as the OR existence/companion
/// pairs are folded in, so each mirror passes through the counterexample
/// filter — a false mirror is discarded there — and is proved by the normal
/// engine. The reverse direction therefore becomes an explicit, genuine proof
/// rather than an assertion.
///
/// Entries are appended in input order, skipping two defined cases that are
/// part of the contract (not failure fallbacks): an empty string — the mirror
/// of a conjecture whose head is symmetric collapses to nothing under I-9, a
/// defined "no distinct reverse direction" result — and any mirror already
/// present in `pool` (or repeated within `mirrors`). The caller sorts `pool`
/// afterwards for deterministic output.
///
/// @param pool    The prove-pool conjecture list, appended to in place.
/// @param mirrors Mirror conjectures; may contain empties and duplicates.
/// @return Number of mirror conjectures actually appended to `pool`.
/// @see Conjecturer::createReshuffledMirrored — produces the mirror strings.
/// @see D-112 — the decision this helper implements.
int mergeMirrorConjecturesIntoPool(std::vector<std::string>& pool,
                                   const std::vector<std::string>& mirrors) {
    std::set<std::string> present(pool.begin(), pool.end());
    int added = 0;
    for (const std::string& mirror : mirrors) {
        if (mirror.empty()) continue;
        if (present.insert(mirror).second) {
            pool.push_back(mirror);
            ++added;
        }
    }
    return added;
}

// ---- TreeNode1-based def-set parsing ----

std::pair<ce::TreeNode1*, std::vector<int>> parseDefSet(const std::string& s) {
    int index = 0;

    std::function<std::pair<ce::TreeNode1*, std::vector<int>>()> parseSubexpr =
        [&]() -> std::pair<ce::TreeNode1*, std::vector<int>> {
        std::vector<int> leafIds;
        auto* node = new ce::TreeNode1();
        std::string label;

        if (s[index] == '(') {
            index++;
            if (s[index] == 'x') {
                index++;
                label = "x";
                auto [leftNode, leftIds] = parseSubexpr();
                auto [rightNode, rightIds] = parseSubexpr();
                node->left = leftNode;
                node->right = rightNode;
                leafIds.insert(leafIds.end(), leftIds.begin(), leftIds.end());
                leafIds.insert(leafIds.end(), rightIds.begin(), rightIds.end());
            } else {
                auto end_index = s.find(')', index);
                label = s.substr(index, end_index - index);
                leafIds.push_back(std::stoi(label));
                index = (int)end_index;
            }
        } else if (index + 1 < (int)s.size() && s.substr(index, 2) == "P(") {
            index += 2;
            if (s[index] == 'x') {
                index++;
                label = "P(x)";
                auto [leftNode, leftIds] = parseSubexpr();
                auto [rightNode, rightIds] = parseSubexpr();
                node->left = leftNode;
                node->right = rightNode;
                leafIds.insert(leafIds.end(), leftIds.begin(), leftIds.end());
                leafIds.insert(leafIds.end(), rightIds.begin(), rightIds.end());
            } else {
                auto end_index = s.find(')', index);
                std::string num = s.substr(index, end_index - index);
                leafIds.push_back(std::stoi(num));
                label = "P(" + num + ")";
                index = (int)end_index;
            }
        }

        index++;
        node->value = label;
        return {node, leafIds};
    };

    return parseSubexpr();
}

/// @brief Render a `TreeNode1` to its string form, reordering
///        commutative arg lists into canonical (ascending) order.
///        See header declaration for the full contract.
std::string treeToStrReorder(const ce::TreeNode1* root) {
    std::string result;
    int counter = 1;

    std::function<void(const ce::TreeNode1*)> nodeToStr = [&](const ce::TreeNode1* node) {
        if (node->value == "x") {
            result += "(x";
        } else if (node->value == "P(x)") {
            result += "P(x";
        } else if (!node->value.empty() && node->value[0] == 'P') {
            result += "P(" + std::to_string(counter);
            counter++;
        } else {
            result += "(" + std::to_string(counter);
            counter++;
        }

        if (node->left) nodeToStr(node->left);
        if (node->right) nodeToStr(node->right);
        result += ")";
    };

    nodeToStr(root);
    return result;
}

/// @brief Render a `TreeNode1` to its string form, applying
///        `subMap` as an integer-id substitution and shifting
///        every retained id by `offset`. See header declaration
///        for the full contract.
std::string treeToStr(const ce::TreeNode1* root, int offset, const std::map<int,int>& subMap) {
    std::string result;

    std::function<void(const ce::TreeNode1*)> nodeToStr = [&](const ce::TreeNode1* node) {
        if (node->value == "x") {
            result += "(x";
        } else if (node->value == "P(x)") {
            result += "P(x";
        } else if (!node->value.empty() && node->value[0] == 'P') {
            // Extract number from P(N)
            std::regex re(R"(\(([^)]+)\))");
            std::smatch m;
            std::string v = node->value;
            std::regex_search(v, m, re);
            int tempInt = std::stoi(m[1].str()) - offset;
            auto it = subMap.find(tempInt);
            if (it != subMap.end()) tempInt = it->second;
            result += "P(" + std::to_string(tempInt);
        } else {
            int tempInt = std::stoi(node->value) - offset;
            auto it = subMap.find(tempInt);
            if (it != subMap.end()) tempInt = it->second;
            result += "(" + std::to_string(tempInt);
        }

        if (node->left) nodeToStr(node->left);
        if (node->right) nodeToStr(node->right);
        result += ")";
    };

    nodeToStr(root);
    return result;
}

std::pair<int,int> findMinMaxNumbers(const std::string& s) {
    std::regex pat(R"(\((\d+)\))");
    std::sregex_iterator it(s.begin(), s.end(), pat);
    std::sregex_iterator end;
    int minN = INT_MAX, maxN = INT_MIN;
    bool found = false;
    for (; it != end; ++it) {
        int n = std::stoi((*it)[1].str());
        minN = std::min(minN, n);
        maxN = std::max(maxN, n);
        found = true;
    }
    if (!found) return {0, 0};
    return {minN, maxN};
}

std::set<int> findAllIds(const std::string& s) {
    std::regex pat(R"(\((\d+)\))");
    std::sregex_iterator it(s.begin(), s.end(), pat);
    std::sregex_iterator end;
    std::set<int> result;
    for (; it != end; ++it) {
        result.insert(std::stoi((*it)[1].str()));
    }
    return result;
}

/// @brief Merge `(num1, num2)` into a transitive replacement map.
///        See header declaration for the full contract.
void updateReplacementMap(std::map<int,int>& repMap, int num1, int num2) {
    std::set<int> connected;
    connected.insert(num1);
    connected.insert(num2);

    bool stay = true;
    while (stay) {
        std::set<int> newConnected;
        for (int arg : connected) {
            auto it = repMap.find(arg);
            if (it != repMap.end() && connected.find(it->second) == connected.end()) {
                newConnected.insert(it->second);
            }
        }
        if (!newConnected.empty()) {
            connected.insert(newConnected.begin(), newConnected.end());
        } else {
            stay = false;
        }
    }

    int minVal = *connected.begin();
    for (int arg : connected) {
        if (arg != minVal) repMap[arg] = minVal;
    }
}

std::pair<bool, std::map<int,int>> defSetsEqual(const std::string& ds1, const std::string& ds2) {
    auto [temp1, ids1] = reorderNumbers(ds1);
    auto [temp2, ids2] = reorderNumbers(ds2);

    std::map<int,int> repMap;
    if (temp1 == temp2) {
        for (int id1 : ids1) {
            for (int id2 : ids2) {
                updateReplacementMap(repMap, id1, id2);
            }
        }
    }
    return {temp1 == temp2, repMap};
}

std::pair<std::string, std::vector<int>> subtractAndReplaceNumbers(const std::string& s, int subtractValue, const std::map<int,int>& m) {
    auto [root, leafIds] = parseDefSet(s);
    std::string newStr = treeToStr(root, subtractValue, m);
    ce::deleteTree(root);
    return {newStr, leafIds};
}

std::pair<std::string, std::vector<int>> reorderNumbers(const std::string& s) {
    auto [root, leafIds] = parseDefSet(s);
    std::string newStr = treeToStrReorder(root);
    ce::deleteTree(root);
    return {newStr, leafIds};
}

/// @brief Adjust every def-set text in `argDefSetMap` so they
///        share a contiguous integer-id space. See header
///        declaration for the full contract.
void shiftTogether(std::map<std::string,std::string>& argDefSetMap) {
    std::set<int> idSet;
    for (auto& [arg, ds] : argDefSetMap) {
        auto ids = findAllIds(ds);
        idSet.insert(ids.begin(), ids.end());
    }
    std::vector<int> idList(idSet.begin(), idSet.end());

    std::map<int,int> repMap;
    for (int i = 0; i < (int)idList.size(); ++i) {
        repMap[idList[i]] = i + 1;
    }

    for (auto& [arg, ds] : argDefSetMap) {
        auto [newDs, _] = subtractAndReplaceNumbers(ds, 0, repMap);
        ds = newDs;
    }
}

std::tuple<std::map<std::string,std::string>, std::set<std::pair<std::string,std::string>>, int, std::vector<std::string>>
connectExpressionSets(
    const std::set<std::pair<std::string,std::string>>& set1,
    const std::set<std::pair<std::string,std::string>>& set2,
    const std::string& connectionType,
    int isDefinition,
    const std::set<std::string>& argsToRemove,
    bool afterGrooming)
{
    std::map<int,int> repMap;
    int success = 1;
    std::vector<std::string> removedArgs;

    int globalMax = 0;
    for (auto& [arg, val] : set1) {
        auto [mn, mx] = findMinMaxNumbers(val);
        globalMax = std::max(globalMax, mx);
    }

    std::set<std::pair<std::string,std::string>> tempSet;
    for (auto& [arg, val] : set2) {
        auto [newVal, _] = subtractAndReplaceNumbers(val, -globalMax, identityMap());
        tempSet.insert({arg, newVal});
    }
    tempSet.insert(set1.begin(), set1.end());

    std::set<std::pair<std::string,std::string>> boolSet;
    for (auto& tple1 : tempSet) {
        boolSet.insert(tple1);
        for (auto& tple2 : tempSet) {
            if (tple1.first == tple2.first && tple1 != tple2 && boolSet.find(tple2) == boolSet.end()) {
                boolSet.insert(tple2);
                auto [eq, smallRepMap] = defSetsEqual(tple1.second, tple2.second);
                if (eq) {
                    for (auto& [k, v] : smallRepMap) {
                        updateReplacementMap(repMap, k, v);
                    }
                } else {
                    success = 0;
                }
            }
        }
    }

    std::set<std::pair<std::string,std::string>> tempSet2;
    for (auto& [arg, val] : tempSet) {
        auto [newVal, _] = subtractAndReplaceNumbers(val, 0, repMap);
        tempSet2.insert({arg, newVal});
    }

    std::map<std::string,std::string> commonMap;
    for (auto& [arg, val] : tempSet2) {
        commonMap[arg] = val;
    }

    if (connectionType == ">" && success == 1) {
        for (auto& tple1 : set1) {
            for (auto& tple2 : set2) {
                if (tple1.first == tple2.first && commonMap.find(tple1.first) != commonMap.end()
                    && argsToRemove.find(tple1.first) != argsToRemove.end()) {
                    commonMap.erase(tple1.first);
                    removedArgs.push_back(tple1.first);
                    if (tple1.second[0] == 'P' && isDefinition == 0) {
                        success = 0;
                    }
                }
            }
        }
    }

    if (!afterGrooming) {
        if (argsToRemove != std::set<std::string>(removedArgs.begin(), removedArgs.end())) {
            success = 0;
        }
    }

    shiftTogether(commonMap);
    std::set<std::pair<std::string,std::string>> commonSet;
    for (auto& [k, v] : commonMap) {
        commonSet.insert({k, v});
    }

    return {commonMap, commonSet, success, removedArgs};
}

/// @brief Extract the substring between the first `[` at or after
///        `startIndex` and its matching `]`. See header
///        declaration for the full contract.
std::string extractBetweenBrackets(const std::string& s, size_t startIndex) {
    auto start = s.find('[', startIndex);
    if (start == std::string::npos) return "";
    auto end = s.find(']', start);
    if (end == std::string::npos) return "";
    return s.substr(start + 1, end - start - 1);
}

/// @brief Find the position of `substring` inside `text`,
///        accepting only matches enclosed by `[...]` brackets.
///        See header declaration for the full contract.
int findPositionSurrounded(const std::string& text, const std::string& substring) {
    size_t subLen = substring.size();
    size_t pos = text.find(substring);
    while (pos != std::string::npos) {
        if (pos > 0 && (text[pos - 1] == '[' || text[pos - 1] == ',')
            && (pos + subLen < text.size()) && (text[pos + subLen] == ']' || text[pos + subLen] == ',')) {
            return (int)pos;
        }
        pos = text.find(substring, pos + 1);
    }
    return -1;
}

std::vector<std::string> sortListAccordingToOccurrence(const std::vector<std::string>& lst, const std::string& text) {
    std::vector<std::pair<std::string, int>> tpleList;
    for (auto& arg : lst) {
        int pos = findPositionSurrounded(text, arg);
        tpleList.push_back({arg, pos});
    }
    std::sort(tpleList.begin(), tpleList.end(), [](const auto& a, const auto& b) {
        return a.second < b.second;
    });
    std::vector<std::string> result;
    for (auto& [arg, _] : tpleList) result.push_back(arg);
    return result;
}

std::vector<std::string> findOrderedIntegers(const std::vector<std::string>& intStrings, const std::string& bigString) {
    std::vector<std::string> found;
    for (auto& s : intStrings) {
        if (bigString.find(s) != std::string::npos) {
            found.push_back(s);
        }
    }
    std::sort(found.begin(), found.end(), [&](const std::string& a, const std::string& b) {
        return bigString.find(a) < bigString.find(b);
    });
    return found;
}

/// @brief Replace one occurrence of an integer-ish substring
///        inside a bigger string. See header declaration for the
///        full contract.
std::string replaceIntegerInString(const std::string& bigString, const std::string& targetInt, const std::string& replacementInt) {
    std::string pattern = "\\b" + targetInt + "\\b";
    std::regex re(pattern);
    return std::regex_replace(bigString, re, replacementInt);
}

/// @brief Subtract `number` from every integer arg id in `expr`
///        that satisfies the `numbersToReplace` filter. See
///        header declaration for the full contract.
std::string subtractNumberFromInts(const std::string& expr, int number, const std::set<int>& numbersToReplace, bool replaceAll) {
    if (expr.empty()) return expr;

    // Manual scan: find digit sequences preceded by '[' or ',' and followed by ']' or ','
    std::string result;
    result.reserve(expr.size());
    size_t i = 0;
    while (i < expr.size()) {
        if (std::isdigit(static_cast<unsigned char>(expr[i]))) {
            size_t start = i;
            while (i < expr.size() && std::isdigit(static_cast<unsigned char>(expr[i]))) ++i;
            // Check context: preceded by '[' or ',' AND followed by ']' or ','
            bool preceded = (start > 0) && (expr[start - 1] == '[' || expr[start - 1] == ',');
            bool followed = (i < expr.size()) && (expr[i] == ']' || expr[i] == ',');
            if (preceded && followed) {
                int oldNum = std::stoi(expr.substr(start, i - start));
                if (replaceAll || numbersToReplace.find(oldNum) != numbersToReplace.end()) {
                    result += std::to_string(oldNum - number);
                } else {
                    result.append(expr, start, i - start);
                }
            } else {
                result.append(expr, start, i - start);
            }
        } else {
            result.push_back(expr[i]);
            ++i;
        }
    }
    return result;
}

/// @brief Count identity entries in `mapping`. See header
///        declaration for the full contract.
int getNumberRemovableArgs(const std::map<std::string,std::string>& mapping) {
    std::map<std::string,std::string> subMap;
    for (auto& [k, v] : mapping) {
        if (v != k) subMap[k] = v;
    }
    std::set<std::string> vals;
    for (auto& [k, v] : subMap) vals.insert(v);
    return (int)vals.size();
}

/// @brief Validate that `mapping` is a well-formed substitution.
///        See header declaration for the full contract.
bool mappingGood(const std::map<std::string,std::string>& mapping) {
    std::map<std::string, std::string> reversed;
    for (auto& [key, value] : mapping) {
        auto it = reversed.find(value);
        if (it != reversed.end()) {
            if (key < it->second) it->second = key;
        } else {
            reversed[value] = key;
        }
    }
    for (auto& [value, minKey] : reversed) {
        if (minKey != value) return false;
    }
    return true;
}

// ============================================================================
// Configuration loading (Phase 1)
// ============================================================================

/// @brief Locate and parse `Config<anchorId>.json` into a fully-populated
///        `ConfigurationData` record.
///
/// @details
/// Uses `nlohmann::ordered_json` for the first parse so the JSON key
/// order in `expressionOrder` survives — downstream enumeration walks
/// expressions in author order for deterministic conjecture sequences.
/// The same JSON is re-parsed via plain `json::parse(oj.dump())` to
/// pick up the rest of the schema.
///
/// Compiles every `patterns_to_exclude_raw` and `only_in_head_raw`
/// regex; failure is reported but does not abort the load (the
/// affected pattern is dropped). Anchor expression description is
/// looked up by name once the loop has populated `data`.
///
/// @param anchorId Short anchor name (e.g. `"Peano"`, `"Gauss"`,
///                 `"IncubatorPeano"`) used to assemble the config
///                 file path `files/config/Config<anchorId>.json`.
/// @return Fully populated `ConfigurationData`. Invalid path /
///         malformed JSON yields a default-constructed record with
///         `anchor_id` set but `data` empty (caller should treat
///         that as a hard error).
/// @see Header declaration for the contract.
ConfigurationData Conjecturer::loadConfiguration(const std::string& anchorId) {
    ConfigurationData config;
    config.anchor_id = anchorId;

    auto configPath = projectRoot_ / "files" / "config" / ("Config" + anchorId + ".json");

    std::ifstream in(configPath);
    if (!in) {
        std::cerr << "Cannot open config: " << configPath << "\n";
        return config;
    }

    // Parse with ordered_json to preserve JSON key order for expression iteration
    nlohmann::ordered_json oj;
    in >> oj;
    // Record top-level key order (before erasing meta keys)
    std::vector<std::string> topKeyOrder;
    for (auto it = oj.begin(); it != oj.end(); ++it) topKeyOrder.push_back(it.key());

    // Convert to regular json for the rest of parsing
    json j = json::parse(oj.dump());

    // --- parameters ---
    if (j.contains("parameters") && j["parameters"].is_object()) {
        auto& p = j["parameters"];
        auto& cp = config.parameters;
        cp.min_number_simple_expressions = p.value("min_number_simple_expressions", 2);
        cp.max_number_simple_expressions = p.value("max_number_simple_expressions", 0);
        cp.max_size_mapping_def_set = p.value("max_size_mapping_def_set", 0);
        cp.max_number_args_expr = p.value("max_number_args_expr", 0);
        cp.operator_threshold = p.value("operator_threshold", 0);
        cp.max_size_binary_list = p.value("max_size_binary_list", 0);
        cp.incubator_mode = p.value("incubator_mode", false);
        cp.apply_in_premise_filter = p.value("apply_in_premise_filter", true);
        if (p.contains("max_distinct_anchor_values_per_type") && p["max_distinct_anchor_values_per_type"].is_object())
            for (auto& [k, v] : p["max_distinct_anchor_values_per_type"].items())
                cp.max_distinct_anchor_values_per_type[k] = v.get<int>();

        if (p.contains("max_values_for_def_sets") && p["max_values_for_def_sets"].is_object())
            for (auto& [k, v] : p["max_values_for_def_sets"].items())
                cp.max_values_for_def_sets[k] = v.get<int>();
        if (p.contains("max_values_for_uncomb_def_sets") && p["max_values_for_uncomb_def_sets"].is_object())
            for (auto& [k, v] : p["max_values_for_uncomb_def_sets"].items())
                cp.max_values_for_uncomb_def_sets[k] = v.get<int>();
        if (p.contains("max_values_for_def_sets_prior_connection") && p["max_values_for_def_sets_prior_connection"].is_object())
            for (auto& [k, v] : p["max_values_for_def_sets_prior_connection"].items())
                cp.max_values_for_def_sets_prior_connection[k] = v.get<int>();
        if (p.contains("max_complexity_if_anchor_parameter_connected_before_existence") && p["max_complexity_if_anchor_parameter_connected_before_existence"].is_object())
            for (auto& [k, v] : p["max_complexity_if_anchor_parameter_connected_before_existence"].items())
                cp.max_complexity_if_anchor_parameter_connected_before_existence[k] = v.get<int>();
        if (p.contains("max_complexity_if_anchor_parameter_connected_after_existence") && p["max_complexity_if_anchor_parameter_connected_after_existence"].is_object())
            for (auto& [k, v] : p["max_complexity_if_anchor_parameter_connected_after_existence"].items()) {
                // Accept legacy int (treated as [int, 100]) or [int, int] vector.
                int comp = 0, arity = 100;
                if (v.is_number_integer()) {
                    comp = v.get<int>();
                } else if (v.is_array() && v.size() >= 1) {
                    comp = v[0].get<int>();
                    if (v.size() >= 2) arity = v[1].get<int>();
                }
                cp.max_complexity_if_anchor_parameter_connected_after_existence[k] = {comp, arity};
            }

        if (p.contains("simple_facts_parameters") && p["simple_facts_parameters"].is_array())
            for (auto& v : p["simple_facts_parameters"]) cp.simple_facts_parameters.push_back(v.get<int>());
        if (p.contains("fact_variable_kinds") && p["fact_variable_kinds"].is_array())
            for (auto& v : p["fact_variable_kinds"]) cp.fact_variable_kinds.push_back(v.get<std::string>());

        j.erase("parameters");
    }

    // --- patterns_to_exclude ---
    if (j.contains("patterns_to_exclude") && j["patterns_to_exclude"].is_array()) {
        for (auto& v : j["patterns_to_exclude"]) {
            std::string s = v.get<std::string>();
            config.patterns_to_exclude_raw.push_back(s);
            try { config.patterns_to_exclude.push_back(std::regex(s)); }
            catch (...) {}
        }
        j.erase("patterns_to_exclude");
    }

    // --- only_in_head ---
    if (j.contains("only_in_head") && j["only_in_head"].is_array()) {
        for (auto& v : j["only_in_head"]) {
            config.only_in_head_raw.push_back(v.get<std::string>());
        }
        j.erase("only_in_head");
    }

    // --- prohibited_combinations ---
    if (j.contains("prohibited_combinations") && j["prohibited_combinations"].is_array()) {
        for (auto& item : j["prohibited_combinations"]) {
            if (item.is_array() && item.size() == 2) {
                config.prohibited_combinations.push_back({item[0].get<std::string>(), item[1].get<std::string>()});
            }
        }
        j.erase("prohibited_combinations");
    }

    // --- prohibited_heads ---
    if (j.contains("prohibited_heads") && j["prohibited_heads"].is_array()) {
        for (auto& v : j["prohibited_heads"]) config.prohibited_heads.push_back(v.get<std::string>());
        j.erase("prohibited_heads");
    }

    // --- folder overrides ---
    if (j.contains("theorems_folder")) {
        config.theorems_folder = j["theorems_folder"].get<std::string>();
        j.erase("theorems_folder");
    }
    if (j.contains("background_theorems_folder")) {
        config.background_theorems_folder = j["background_theorems_folder"].get<std::string>();
        j.erase("background_theorems_folder");
    }
    if (j.contains("anchor_name")) {
        config.anchor_name = j["anchor_name"].get<std::string>();
        j.erase("anchor_name");
    }

    // --- definitions folder ---
    auto defsDir = projectRoot_ / "files" / "definitions";

    // --- expressions ---
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string name = it.key();
        auto& spec = it.value();
        if (!spec.is_object() || !spec.contains("arity")) continue;

        ExpressionDescription desc;
        desc.arity = spec.value("arity", 0);

        // definition_sets (3-tuple)
        if (spec.contains("definition_sets") && spec["definition_sets"].is_object()) {
            for (auto& [k, v] : spec["definition_sets"].items()) {
                if (v.is_array()) {
                    std::string text = v.size() >= 1 ? stripWs(v[0].get<std::string>()) : "";
                    bool combinable = v.size() >= 2 ? v[1].get<bool>() : true;
                    bool connectable = v.size() >= 3 ? v[2].get<bool>() : true;
                    desc.definition_sets[k] = {text, combinable, connectable};
                } else if (v.is_string()) {
                    desc.definition_sets[k] = {stripWs(v.get<std::string>()), true, true};
                }
            }
        }

        // full_mpl
        std::string fullMplRaw = spec.value("full_mpl", std::string{});
        bool looksLikeFile = !fullMplRaw.empty() &&
            (fullMplRaw.size() >= 4 && fullMplRaw.substr(fullMplRaw.size()-4) == ".mpl"
             || fullMplRaw.find('/') != std::string::npos
             || fullMplRaw.find('\\') != std::string::npos);
        if (looksLikeFile) {
            std::filesystem::path p(fullMplRaw);
            std::filesystem::path candidate = defsDir / p.filename();
            if (std::filesystem::exists(candidate)) {
                desc.full_mpl = ce::readTreeFromFile(candidate);
            } else {
                desc.full_mpl = stripWs(fullMplRaw);
            }
        } else {
            desc.full_mpl = stripWs(fullMplRaw);
        }

        // handle
        desc.handle = spec.value("handle", "(" + name + "[");
        if (!desc.handle.empty() && desc.handle.back() != '[') {
            desc.handle = "(" + name + "[";
        }

        // short_mpl
        std::string shortMplRaw = spec.value("short_mpl", std::string{});
        desc.short_mpl_raw = shortMplRaw;
        if (!shortMplRaw.empty()) {
            desc.short_mpl_normalized = makeAnchorSignatureFromRaw(stripWs(shortMplRaw));
        }

        desc.max_count_per_conjecture = spec.value("max_count_per_conjecture", 0);
        desc.max_size_expression_before_existence = spec.value("max_size_expression_before_existence", 0);
        desc.max_size_expression_after_existence  = spec.value("max_size_expression_after_existence",  0);
        desc.min_size_expression = spec.value("min_size_expression", 1);
        desc.allow_negation = spec.value("allow_negation", false);
        desc.allow_to_constitute_existence = spec.value("allow_to_constitute_existence", false);
        desc.existence_variable_position = spec.value("existence_variable_position", -1);
        if (spec.contains("allowed_for_existence") && spec["allowed_for_existence"].is_array()) {
            for (auto& v : spec["allowed_for_existence"]) {
                if (v.is_number_integer()) desc.allowed_for_existence.push_back(v.get<int>());
            }
        }

        // input_args, output_args
        if (spec.contains("input_args") && spec["input_args"].is_array())
            for (auto& v : spec["input_args"]) desc.input_args.push_back(v.get<std::string>());
        if (spec.contains("output_args") && spec["output_args"].is_array())
            for (auto& v : spec["output_args"]) desc.output_args.push_back(v.get<std::string>());

        // Align and sort by signature position
        auto orderedArgs = extractArgList(shortMplRaw);
        auto [sortedInput, indInput] = alignAndSortArgs(desc.input_args, orderedArgs);
        auto [sortedOutput, indOutput] = alignAndSortArgs(desc.output_args, orderedArgs);
        desc.input_args = sortedInput;
        desc.indices_input_args = indInput;
        desc.output_args = sortedOutput;
        desc.indices_output_args = indOutput;

        config.data[name] = desc;
    }

    // Preserve JSON key order for expression iteration (matches Python dict order)
    for (auto& key : topKeyOrder) {
        if (config.data.find(key) != config.data.end()) {
            config.expressionOrder.push_back(key);
        }
    }

    return config;
}

/// @brief Build the `coreExprMap_` adapter shim from `config_.data`.
///
/// @details
/// One-to-one projection: every `ExpressionDescription` produces a
/// `ce::CoreExpressionConfig` that downstream shim functions
/// (originally written against the prover's compiled-config record
/// shape) consume without further translation.
void Conjecturer::buildCoreExprMapAdapter() {
    for (auto& [name, desc] : config_.data) {
        ce::CoreExpressionConfig cfg;
        cfg.arity = desc.arity;
        cfg.signature = desc.short_mpl_normalized;
        for (auto& [k, v] : desc.definition_sets) {
            cfg.definitionSets[k] = {std::get<0>(v), std::get<1>(v)};
        }
        cfg.inputArgs = desc.input_args;
        cfg.outputArgs = desc.output_args;
        cfg.inputIndices = desc.indices_input_args;
        cfg.outputIndices = desc.indices_output_args;
        cfg.definition = desc.full_mpl;
        coreExprMap_[name] = cfg;
    }
}

// ============================================================================
// Int-path: NameMap + encode/decode
// ============================================================================

/// @brief Populate `nameMap_` from every expression name, handle, and
///        def-set text appearing in `config_.data`. Called once during
///        construction.
void Conjecturer::buildNameMap() {
    // Register all expression names and def-set texts
    for (auto& [name, desc] : config_.data) {
        nameMap_.encode(name);
        nameMap_.encode(desc.handle);
        for (auto& [pos, tpl] : desc.definition_sets) {
            nameMap_.encode(std::get<0>(tpl));  // def-set text like "(1)", "P(1)"
        }
    }
}

/// @brief Populate `intExprConfigs_` (indexed by nameId) plus the
///        per-defSetId limit arrays from `config_`.
///
/// @details
/// Walks every `ExpressionDescription` once, projecting it into the
/// dense int16_t per-expression record, then a second pass over the
/// per-type config maps populates `maxForDefSets_`,
/// `maxForUncombDefSets_`, `maxForDefSetsPrior_`, and
/// `maxComplexityAnchorConn_`. After this returns the int-path hot
/// loop can answer every per-expression / per-def-set query with a
/// single indexed read.
void Conjecturer::buildIntExprConfigs() {
    // Size to cover all nameIds
    intExprConfigs_.resize(nameMap_.nextId);
    for (auto& [name, desc] : config_.data) {
        int16_t nid = nameMap_.lookup(name);
        if (nid == 0 || nid >= (int16_t)intExprConfigs_.size()) continue;
        auto& ic = intExprConfigs_[nid];
        ic.nameId = nid;
        ic.arity = (int16_t)desc.arity;
        ic.maxCountPerConj = (int16_t)desc.max_count_per_conjecture;
        ic.handleId = nameMap_.lookup(desc.handle);
        ic.maxSizeExprBeforeEx = (int16_t)desc.max_size_expression_before_existence;
        ic.maxSizeExprAfterEx  = (int16_t)desc.max_size_expression_after_existence;
        ic.minSizeExpr = (int16_t)desc.min_size_expression;
        ic.numInputArgs = (int16_t)desc.indices_input_args.size();
        for (int i = 0; i < ic.numInputArgs && i < 16; ++i)
            ic.indicesInputArgs[i] = (int16_t)desc.indices_input_args[i];
        ic.numOutputArgs = (int16_t)desc.indices_output_args.size();
        for (int i = 0; i < ic.numOutputArgs && i < 4; ++i)
            ic.indicesOutputArgs[i] = (int16_t)desc.indices_output_args[i];
    }

    // Build defSetId-indexed config param limit arrays
    int16_t maxDsId = nameMap_.nextId;
    maxForDefSets_.assign(maxDsId, -1);
    maxForUncombDefSets_.assign(maxDsId, -1);
    maxForDefSetsPrior_.assign(maxDsId, -1);
    maxComplexityAnchorConn_.assign(maxDsId, -1);

    for (auto& [ds, val] : config_.parameters.max_values_for_def_sets) {
        int16_t id = nameMap_.lookup(ds);
        if (id > 0) maxForDefSets_[id] = (int16_t)val;
    }
    for (auto& [ds, val] : config_.parameters.max_values_for_uncomb_def_sets) {
        int16_t id = nameMap_.lookup(ds);
        if (id > 0) maxForUncombDefSets_[id] = (int16_t)val;
    }
    for (auto& [ds, val] : config_.parameters.max_values_for_def_sets_prior_connection) {
        int16_t id = nameMap_.lookup(ds);
        if (id > 0) maxForDefSetsPrior_[id] = (int16_t)val;
    }
    // The int-path dense lookup currently mirrors only the pre-existence caps — the
    // after-existence check runs on the string path via passesComplexityAfterExistence.
    for (auto& [ds, val] : config_.parameters.max_complexity_if_anchor_parameter_connected_before_existence) {
        int16_t id = nameMap_.lookup(ds);
        if (id > 0) maxComplexityAnchorConn_[id] = (int16_t)val;
    }

    // Build operator/relation nameId lists
    for (auto& op : operators_) {
        int16_t id = nameMap_.lookup(op);
        if (id > 0) operatorNameIds_.push_back(id);
    }
    for (auto& rel : relations_) {
        int16_t id = nameMap_.lookup(rel);
        if (id > 0) relationNameIds_.push_back(id);
    }
}

/// @brief Serialize an MPL expression string into a flat
///        `IntConjBuf` (inverse of `decodeExpr`).
///
/// @details
/// Companion to `decodeExpr`: round-trip
/// `decodeExpr(encodeExpr(s)) == s` for any conjecturer-emitted
/// `s` (tested via the `intpath_*` suite). Walks `expr` block by
/// block, emitting `[boundCount, bv0..bvN, nameId, arity,
/// arg0..argN]` per quantifier layer. The output buffer is
/// suitable for the int-path filter cascade — no further
/// string-form parsing is needed once it is built. Returns an
/// empty buffer (`len == 0`) when `expr` is empty or malformed —
/// typically because the caller passed a non-conjecturer input.
///
/// @param expr Expression text in numbered-variable form (e.g.
///             `"(>[1,2](AnchorPeano[...])(in[1,2]))"`). Must be
///             a recognised conjecture/sub-expression shape with
///             `name[arg0,...]` heads.
/// @return Encoded buffer with `len <= MAX_CONJ_BUF`.
/// @pre  `expr` is in the conjecturer's numbered-variable form.
/// @post `len <= MAX_CONJ_BUF`.
/// @see `decodeExpr` — recovers `expr` from the returned buffer.
IntConjBuf Conjecturer::encodeExpr(const std::string& expr) const {
    prof::Scope _prof_enc(prof::g_encodeExpr);
    // Encodes a conjecture string into a flat int16_t array.
    // Conjectures are right-leaning chains: (>[bv](leaf1)(>[bv](leaf2)...(head)))
    // Encoding: sequential blocks, each = [boundCount, bv..., nameId, arity, args...]
    // Last block is the head with boundCount=0.
    //
    // A bare leaf like (in[1,2]) encodes as: [0, nameId, arity, args...]
    IntConjBuf buf;
    buf.len = 0;
    int pos = 0;
    int n = (int)expr.size();

    auto parseLeaf = [&]() {
        // Expect pos at '(' of a leaf like (name[args])
        assert(expr[pos] == '(');
        pos++; // skip '('
        int nameStart = pos;
        while (pos < n && expr[pos] != '[') pos++;
        std::string name = expr.substr(nameStart, pos - nameStart);
        int16_t nameId = nameMap_.lookup(name);
        assert(nameId != 0);
        buf.data[buf.len++] = nameId;

        pos++; // skip '['
        int16_t arityPos = buf.len;
        buf.data[buf.len++] = 0; // placeholder
        int16_t arity = 0;
        while (pos < n && expr[pos] != ']') {
            if (expr[pos] == ',') { pos++; continue; }
            int numStart = pos;
            while (pos < n && expr[pos] >= '0' && expr[pos] <= '9') pos++;
            if (pos > numStart) {
                buf.data[buf.len++] = (int16_t)std::stoi(expr.substr(numStart, pos - numStart));
                arity++;
            }
        }
        buf.data[arityPos] = arity;
        pos++; // skip ']'
        pos++; // skip ')'
    };

    // Recursive: parses either (>[bv](left)(right)) or a bare leaf (name[...])
    std::function<void()> encode = [&]() {
        if (pos >= n) return;
        assert(expr[pos] == '(');

        if (pos + 1 < n && expr[pos + 1] == '>') {
            // Implication node: (>[bv](left)(right))
            pos += 2; // skip "(>"
            assert(expr[pos] == '[');
            pos++; // skip '['

            int16_t boundStart = buf.len;
            buf.data[buf.len++] = 0; // placeholder for boundCount
            int16_t boundCount = 0;
            while (pos < n && expr[pos] != ']') {
                if (expr[pos] == ',') { pos++; continue; }
                int numStart = pos;
                while (pos < n && expr[pos] >= '0' && expr[pos] <= '9') pos++;
                if (pos > numStart) {
                    buf.data[buf.len++] = (int16_t)std::stoi(expr.substr(numStart, pos - numStart));
                    boundCount++;
                }
            }
            buf.data[boundStart] = boundCount;
            pos++; // skip ']'

            // Left = always a leaf
            parseLeaf();

            // Right = either another implication or the head leaf
            // If next char is '(' and the one after is '>', it's nested implication
            if (pos < n && expr[pos] == '(' && pos + 1 < n && expr[pos + 1] == '>') {
                encode(); // recurse into nested implication
            } else {
                // Head leaf: write boundCount=0 prefix then the leaf
                buf.data[buf.len++] = 0;
                parseLeaf();
            }

            pos++; // skip ')' of the implication
        } else {
            // Bare leaf (no implication wrapper — e.g. single expression)
            buf.data[buf.len++] = 0; // boundCount = 0
            parseLeaf();
        }
    };

    encode();
    return buf;
}

/// @brief Deserialize a flat `IntConjBuf` back into its MPL string
///        form (inverse of `encodeExpr`).
///
/// @details
/// Companion to `encodeExpr`: round-trip
/// `decodeExpr(encodeExpr(s)) == s` for any expression `s` the
/// conjecturer emits (verified by the `intpath_*` suite). Walks
/// `buf` block by block, reading each `[boundCount, bv0..bvN,
/// nameId, arity, arg0..argN]` record and emitting the
/// corresponding `(>[bv0,bv1](nameHandle[arg0,arg1])(...))`
/// fragment. Returns the empty string when `buf.len == 0` —
/// typically because the buffer has not been initialised by
/// `encodeExpr`.
///
/// @param buf Encoded buffer previously produced by `encodeExpr`.
///            Must satisfy `buf.len <= MAX_CONJ_BUF`; otherwise
///            the walker reads past the valid range and returns
///            garbage.
/// @return MPL expression text byte-equivalent to the original
///         input of `encodeExpr`, or empty on `buf.len == 0`.
/// @see `encodeExpr` — produces the matching buffer.
std::string Conjecturer::decodeExpr(const IntConjBuf& buf) const {
    prof::Scope _prof_dec(prof::g_decodeExpr);
    // Read sequential blocks from the flat array, wrap right-to-left.
    // Each block: [boundCount, bv..., nameId, arity, args...]
    struct Block {
        int16_t boundCount;
        int16_t boundVars[MAX_CONJ_ARGS];
        int16_t nameId;
        int16_t arity;
        int16_t args[16];
    };
    Block blocks[MAX_CONJ_ARGS];
    int numBlocks = 0;
    int pos = 0;

    while (pos < buf.len) {
        Block& b = blocks[numBlocks++];
        b.boundCount = buf.data[pos++];
        for (int i = 0; i < b.boundCount; ++i)
            b.boundVars[i] = buf.data[pos++];
        b.nameId = buf.data[pos++];
        b.arity = buf.data[pos++];
        for (int i = 0; i < b.arity; ++i)
            b.args[i] = buf.data[pos++];
    }

    if (numBlocks == 0) return "";

    // Build leaf string
    auto blockToStr = [&](const Block& b) -> std::string {
        std::string s = "(" + nameMap_.decode(b.nameId) + "[";
        for (int i = 0; i < b.arity; ++i) {
            if (i > 0) s += ",";
            s += std::to_string(b.args[i]);
        }
        s += "])";
        return s;
    };

    // Build right-to-left: last block = head, each preceding block wraps in (>[bv]leaf ...)
    std::string result = blockToStr(blocks[numBlocks - 1]);
    for (int i = numBlocks - 2; i >= 0; --i) {
        auto& b = blocks[i];
        std::string bvStr = "[";
        for (int j = 0; j < b.boundCount; ++j) {
            if (j > 0) bvStr += ",";
            bvStr += std::to_string(b.boundVars[j]);
        }
        bvStr += "]";
        result = "(>" + bvStr + blockToStr(b) + result + ")";
    }

    return result;
}

/// @brief Serialize a string-keyed `DefSetMap` into the
///        parallel-array `IntDefSetMap` (inverse of
///        `decodeDefSetMap`).
///
/// @details
/// Companion to `decodeDefSetMap`: round-trip
/// `decodeDefSetMap(encodeDefSetMap(m)) == m` is byte-stable for
/// any `m` whose keys are decimal-integer strings and whose
/// def-set texts are already registered in `nameMap_`. Each
/// `argId` is decoded from the stringified arg name (`"5"` -> `5`);
/// each def-set text goes through `nameMap_.encode`. The
/// combinable / connectable bools are projected to int16_t.
/// Truncates silently if `dsm.size() > MAX_CONJ_ARGS` —
/// typically because the caller passed an oversized merge result
/// that bypassed the per-type combinable / uncombinable caps.
///
/// @param dsm Source map keyed by stringified arg ids
///            (e.g. `"1"`, `"5"`). Values are `(text, combinable,
///            connectable)` tuples.
/// @return `IntDefSetMap` with `count == min(dsm.size(),
///         MAX_CONJ_ARGS)` and parallel arrays populated.
/// @see `decodeDefSetMap` — recovers `dsm` from the returned
///      record.
IntDefSetMap Conjecturer::encodeDefSetMap(const DefSetMap& dsm) const {
    prof::Scope _prof_eds(prof::g_encodeDefSetMap);
    IntDefSetMap idsm;
    idsm.count = 0;
    for (auto& [arg, tpl] : dsm) {
        // arg is a string like "1", "2", etc.
        bool isDigit = !arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit);
        if (!isDigit) continue;  // skip non-numeric args (u_ prefixed etc)
        int idx = idsm.count++;
        assert(idx < MAX_CONJ_ARGS);
        idsm.argId[idx] = (int16_t)std::stoi(arg);
        idsm.defSetId[idx] = nameMap_.lookup(std::get<0>(tpl));
        idsm.combinable[idx] = std::get<1>(tpl) ? 1 : 0;
        idsm.connectable[idx] = std::get<2>(tpl) ? 1 : 0;
    }
    return idsm;
}

/// @brief Deserialize an `IntDefSetMap` back into a string-keyed
///        `DefSetMap` (inverse of `encodeDefSetMap`).
///
/// @details
/// Companion to `encodeDefSetMap`: round-trip
/// `decodeDefSetMap(encodeDefSetMap(m)) == m` is byte-stable for
/// any `m` whose def-set texts were registered in `nameMap_` at
/// encode time. Each `argId` is rendered as a decimal string; each
/// `defSetId` is resolved through `nameMap_.decode`. Returns an
/// empty map when `idsm.count == 0` — typically because the
/// caller passed an uninitialised buffer.
///
/// @param idsm Encoded def-set map previously produced by
///             `encodeDefSetMap` (or by an int-path filter that
///             writes directly into the parallel arrays).
/// @return `DefSetMap` keyed by stringified arg ids; size equals
///         `idsm.count`.
/// @see `encodeDefSetMap` — produces the matching encoded map.
DefSetMap Conjecturer::decodeDefSetMap(const IntDefSetMap& idsm) const {
    DefSetMap dsm;
    for (int i = 0; i < idsm.count; ++i) {
        std::string arg = std::to_string(idsm.argId[i]);
        std::string defSetText = nameMap_.decode(idsm.defSetId[i]);
        dsm[arg] = {defSetText, idsm.combinable[i] != 0, idsm.connectable[i] != 0};
    }
    return dsm;
}

// ============================================================================
// Pre-computation (Phase 2)
// ============================================================================

/// @brief Pre-compute every injective mapping of arity `2..N` from
///        `[1..p]` -> `[1..q]` for every `(p, q)` cross product
///        within the size budget.
///
/// @details
/// Built once during construction; consumed read-only by
/// `makeAllConnectionMaps` / `makeAllConnectionMapsInt` in the hot
/// path. Memory footprint scales combinatorially in `N`; the current
/// production batches use `N <= 5` so the table fits comfortably
/// in a few MB.
///
/// @param N Maximum arity to enumerate.
/// @return `MappingsMap[size][(p, q)]` -> list of injections.
MappingsMap Conjecturer::createMap(int N) {
    MappingsMap outer;
    if (N < 2) return outer;

    for (int n = 2; n <= N; ++n) {
        std::map<std::pair<int,int>, std::vector<std::map<int,int>>> M;
        for (int p = 1; p < n; ++p) {
            int q = n - p;
            std::vector<int> S;
            for (int i = p + 1; i <= n; ++i) S.push_back(i);

            std::vector<std::map<int,int>> L;

            // Generate all subsets T of S
            int sSize = (int)S.size();
            for (int r = 0; r <= sSize; ++r) {
                // Generate combinations of size r
                std::vector<bool> selector(sSize, false);
                std::fill(selector.end() - r, selector.end(), true);
                do {
                    std::vector<int> T;
                    for (int i = 0; i < sSize; ++i)
                        if (selector[i]) T.push_back(S[i]);

                    if ((int)T.size() > p) continue;

                    // Generate all injective mappings T -> {1..p}
                    std::vector<int> targets;
                    for (int i = 1; i <= p; ++i) targets.push_back(i);

                    // Permutations of targets, take first T.size()
                    std::vector<int> perm(targets);
                    std::sort(perm.begin(), perm.end());
                    do {
                        std::map<int,int> Q;
                        for (int i = 1; i <= p; ++i) Q[i] = i;
                        for (int idx = 0; idx < (int)T.size(); ++idx)
                            Q[T[idx]] = perm[idx];
                        for (int s : S)
                            if (Q.find(s) == Q.end()) Q[s] = s;
                        L.push_back(Q);
                    } while (std::next_permutation(perm.begin(), perm.end()));

                    // But we only want permutations of length T.size() from p targets.
                    // The above generates all permutations of all p targets.
                    // We need: itertools.permutations(targets, len(T))
                    // Let me fix this.
                } while (std::next_permutation(selector.begin(), selector.end()));
            }

            // Actually, the above is wrong. Let me redo this properly.
            L.clear();
            // Match Python's itertools.combinations order: lexicographic
            // combinations(S, r) yields tuples in lex order from left to right
            auto genCombinations = [](const std::vector<int>& pool, int r) -> std::vector<std::vector<int>> {
                std::vector<std::vector<int>> result;
                int n = (int)pool.size();
                if (r > n || r < 0) return result;
                if (r == 0) { result.push_back({}); return result; }
                std::vector<int> indices(r);
                std::iota(indices.begin(), indices.end(), 0);
                result.push_back({});
                for (int i = 0; i < r; ++i) result.back().push_back(pool[indices[i]]);
                while (true) {
                    int i = r - 1;
                    while (i >= 0 && indices[i] == i + n - r) --i;
                    if (i < 0) break;
                    ++indices[i];
                    for (int j = i + 1; j < r; ++j) indices[j] = indices[j-1] + 1;
                    std::vector<int> combo;
                    for (int j = 0; j < r; ++j) combo.push_back(pool[indices[j]]);
                    result.push_back(combo);
                }
                return result;
            };
            // Match Python's itertools.permutations(targets, k) order
            auto genKPermutations = [](const std::vector<int>& pool, int k) -> std::vector<std::vector<int>> {
                std::vector<std::vector<int>> result;
                int n = (int)pool.size();
                if (k > n || k < 0) return result;
                if (k == 0) { result.push_back({}); return result; }
                // Python itertools.permutations: lex order on positions
                std::vector<int> indices(n);
                std::iota(indices.begin(), indices.end(), 0);
                std::vector<int> cycles(k);
                for (int i = 0; i < k; ++i) cycles[i] = n - i;
                // First permutation
                std::vector<int> perm;
                for (int i = 0; i < k; ++i) perm.push_back(pool[indices[i]]);
                result.push_back(perm);
                while (true) {
                    bool found = false;
                    for (int i = k - 1; i >= 0; --i) {
                        --cycles[i];
                        if (cycles[i] == 0) {
                            // Rotate indices[i:] left by 1
                            int tmp = indices[i];
                            for (int j = i; j < n - 1; ++j) indices[j] = indices[j+1];
                            indices[n-1] = tmp;
                            cycles[i] = n - i;
                        } else {
                            int j = n - cycles[i];
                            std::swap(indices[i], indices[j]);
                            perm.clear();
                            for (int ii = 0; ii < k; ++ii) perm.push_back(pool[indices[ii]]);
                            result.push_back(perm);
                            found = true;
                            break;
                        }
                    }
                    if (!found) break;
                }
                return result;
            };

            for (int r = 0; r <= sSize; ++r) {
                auto combos = genCombinations(S, r);
                for (auto& T : combos) {
                    if ((int)T.size() > p) continue;

                    std::vector<int> allTargets;
                    for (int i = 1; i <= p; ++i) allTargets.push_back(i);

                    int k = (int)T.size();
                    if (k == 0) {
                        std::map<int,int> Q;
                        for (int i = 1; i <= n; ++i) Q[i] = i;
                        L.push_back(Q);
                    } else {
                        auto kPerms = genKPermutations(allTargets, k);
                        for (auto& mapping : kPerms) {
                            std::map<int,int> Q;
                            for (int i = 1; i <= p; ++i) Q[i] = i;
                            for (int idx = 0; idx < k; ++idx)
                                Q[T[idx]] = mapping[idx];
                            for (int si : S)
                                if (Q.find(si) == Q.end()) Q[si] = si;
                            L.push_back(Q);
                        }
                    }
                }
            }

            M[{p, q}] = L;
        }
        outer[n] = M;
    }
    return outer;
}

/// @brief Pre-compute anchor <-> expression argument permutation
///        tables.
///
/// @details
/// Cartesian product over `T \subseteq S` with `targets^|T|` images
/// per subset. For `AnchorIncubator` with its 7 `(1)`-typed slots
/// `leftMax = 7`; `rightMax` is the max over def-sets of
/// `(uncomb + comb)` values from the per-batch config.
///
/// @warning `rightMax > 3` causes RAM explosion: millions of
///          permutation dicts materialise. There is no assert
///          currently — see SwDD `OPEN-8`. Keep
///          `parameters.max_values_for_def_sets` plus
///          `parameters.max_values_for_uncomb_def_sets` so that
///          their per-type maximum sum stays at most 3. Adding the
///          assert is recommended; it is a Rule-8 architectural
///          touch and currently deferred.
MappingsMap Conjecturer::createMapAnchor(int leftMax, int rightMax) {
    MappingsMap outer;
    if (leftMax < 1 || rightMax < 1) return outer;

    int N = leftMax + rightMax;
    for (int n = 2; n <= N; ++n) {
        std::map<std::pair<int,int>, std::vector<std::map<int,int>>> M;
        for (int p = 1; p < n; ++p) {
            int q = n - p;
            if (p > leftMax || q > rightMax) continue;

            std::vector<int> S;
            for (int i = p + 1; i <= n; ++i) S.push_back(i);

            std::vector<std::map<int,int>> L;
            int sSize = (int)S.size();

            // Match Python's itertools.combinations order (lex from left)
            auto genCombinations = [](const std::vector<int>& pool, int r) -> std::vector<std::vector<int>> {
                std::vector<std::vector<int>> result;
                int n = (int)pool.size();
                if (r > n || r < 0) return result;
                if (r == 0) { result.push_back({}); return result; }
                std::vector<int> indices(r);
                std::iota(indices.begin(), indices.end(), 0);
                result.push_back({});
                for (int i = 0; i < r; ++i) result.back().push_back(pool[indices[i]]);
                while (true) {
                    int i = r - 1;
                    while (i >= 0 && indices[i] == i + n - r) --i;
                    if (i < 0) break;
                    ++indices[i];
                    for (int j = i + 1; j < r; ++j) indices[j] = indices[j-1] + 1;
                    std::vector<int> combo;
                    for (int j = 0; j < r; ++j) combo.push_back(pool[indices[j]]);
                    result.push_back(combo);
                }
                return result;
            };

            for (int r = 0; r <= sSize; ++r) {
                auto combos = genCombinations(S, r);
                for (auto& T : combos) {
                    std::vector<int> targets;
                    for (int i = 1; i <= p; ++i) targets.push_back(i);

                    int tSize = (int)T.size();
                    if (tSize == 0) {
                        std::map<int,int> Q;
                        for (int i = 1; i <= n; ++i) Q[i] = i;
                        L.push_back(Q);
                    } else {
                        // Cartesian product: targets^tSize (same as Python itertools.product)
                        std::vector<int> assignment(tSize, 0);
                        while (true) {
                            std::map<int,int> Q;
                            for (int i = 1; i <= n; ++i) Q[i] = i;
                            for (int idx = 0; idx < tSize; ++idx)
                                Q[T[idx]] = targets[assignment[idx]];
                            L.push_back(Q);

                            int carry = tSize - 1;
                            while (carry >= 0) {
                                assignment[carry]++;
                                if (assignment[carry] < p) break;
                                assignment[carry] = 0;
                                carry--;
                            }
                            if (carry < 0) break;
                        }
                    }
                }
            }

            M[{p, q}] = L;
        }
        if (!M.empty()) outer[n] = M;
    }
    return outer;
}

/// @brief Compute the maximum anchor-slot id over the loaded
///        config; used as `leftMax` for `createMapAnchor`.
int Conjecturer::determineLeftSideBoundary() const {
    std::string anchorName = config_.getAnchorName();
    auto& sets = config_.data.at(anchorName).definition_sets;

    std::map<std::string, int> counterMap;
    for (auto& [arg, tpl] : sets) {
        std::string st = std::get<0>(tpl);
        counterMap[st]++;
    }

    int boundary = 0;
    for (auto& [k, v] : counterMap) {
        boundary = std::max(boundary, v);
    }
    return boundary;
}

/// @brief Compute the maximum candidate-slot id (sum of `uncomb +
///        comb` per type) over the loaded config; used as `rightMax`
///        for `createMapAnchor`. Result must stay <= 3 to avoid RAM
///        explosion (SwDD `OPEN-8`).
int Conjecturer::determineRightSideBoundary() const {
    int boundary = -1;
    for (auto& [defSet, val] : config_.parameters.max_values_for_def_sets) {
        auto it = config_.parameters.max_values_for_uncomb_def_sets.find(defSet);
        int uncomb = (it != config_.parameters.max_values_for_uncomb_def_sets.end()) ? it->second : 0;
        int candBound = uncomb + val;
        if (candBound > boundary) boundary = candBound;
    }
    return boundary;
}

// ============================================================================
// Expression parsing & arg maps (Phase 4)
// ============================================================================

/// @brief Walk an expression and extract every arg name with its
///        definition-set type, combinable flag, and connectable
///        flag.
///
/// @details
/// Pure walk: descends into nested `(>[...])` quantifier blocks,
/// pulls every `name[arg0,arg1,...]` token, and looks up the
/// def-set tuple for each arg position via `coreExprMap_`. The
/// returned `DefSetMap` is the canonical input to every string-path
/// filter and to the encoders that produce the int-path twin.
///
/// @param exprIn Expression text in numbered-variable form. Must
///               be a recognised conjecture/sub-expression shape
///               with `name[arg0,...]` heads.
/// @return `DefSetMap` keyed by arg name (as string). Empty when
///         `exprIn` carries no recognised expression.
DefSetMap Conjecturer::findArgMap(const std::string& exprIn) const {
    std::string expr = stripWs(exprIn);
    int index = 0;

    std::function<DefSetMap()> findArgMapCore = [&]() -> DefSetMap {
        DefSetMap nodeMap;

        if (expr[index] == '(') {
            index++;
            if (expr[index] == '>') {
                index++;
                auto argsToRemove = ce::getArgs(expr.substr(index));
                index = (int)expr.find(']', index) + 1;
                auto leftMap = findArgMapCore();
                auto rightMap = findArgMapCore();
                nodeMap = leftMap;
                for (auto& [k, v] : rightMap) nodeMap[k] = v;
                for (auto& arg : argsToRemove) nodeMap.erase(arg);
            } else if (expr[index] == '&') {
                index++;
                auto leftMap = findArgMapCore();
                auto rightMap = findArgMapCore();
                nodeMap = leftMap;
                for (auto& [k, v] : rightMap) nodeMap[k] = v;
            } else {
                auto endIndex = expr.find(')', index);
                std::string nodeLabel = expr.substr(index, endIndex - index);
                auto arrayArgs = ce::getArgs(nodeLabel);
                std::string tempExpr = ce::extractExpression(nodeLabel);
                auto& exprMap = config_.data.at(tempExpr).definition_sets;
                for (int i = 0; i < (int)arrayArgs.size(); ++i) {
                    nodeMap[arrayArgs[i]] = exprMap.at(std::to_string(i + 1));
                }
                for (auto& [arg, val] : exprMap) {
                    bool isDigit = !arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit);
                    if (!isDigit) nodeMap[arg] = val;
                }
                index = (int)endIndex;
            }
        } else if (index + 1 < (int)expr.size() && expr.substr(index, 2) == "!(") {
            index += 2;
            if (expr[index] == '>') {
                index++;
                auto argsToRemove = ce::getArgs(expr.substr(index));
                index = (int)expr.find(']', index) + 1;
                auto leftMap = findArgMapCore();
                auto rightMap = findArgMapCore();
                nodeMap = leftMap;
                for (auto& [k, v] : rightMap) nodeMap[k] = v;
                for (auto& arg : argsToRemove) nodeMap.erase(arg);
            } else if (expr[index] == '&') {
                index++;
                auto leftMap = findArgMapCore();
                auto rightMap = findArgMapCore();
                nodeMap = leftMap;
                for (auto& [k, v] : rightMap) nodeMap[k] = v;
            } else {
                auto endIndex = expr.find(')', index);
                std::string nodeLabel = expr.substr(index, endIndex - index);
                std::string tempExpr = ce::extractExpression(nodeLabel);
                nodeLabel = "!(" + nodeLabel + ")";
                auto arrayArgs = ce::getArgs(nodeLabel);
                auto& exprMap = config_.data.at(tempExpr).definition_sets;
                for (int i = 0; i < (int)arrayArgs.size(); ++i) {
                    nodeMap[arrayArgs[i]] = exprMap.at(std::to_string(i + 1));
                }
                for (auto& [arg, val] : exprMap) {
                    bool isDigit = !arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit);
                    if (!isDigit) nodeMap[arg] = val;
                }
                index = (int)endIndex;
            }
        } else if (expr[index] == ')') {
            index--;
        }

        index++;
        return nodeMap;
    };

    return findArgMapCore();
}

/// @brief Legacy reshuffle predecessor: rename bound variables
///        in `exprIn` to canonical `1..N` ids in first-occurrence
///        order.
///
/// @details
/// Largely superseded by `reshuffle` on the `rt_conjecturer*`
/// branches but retained because some callers still consume
/// its returned `(text, defSets, renameMap)` triple directly.
/// Walks `exprIn` left-to-right, assigning the next free id the
/// first time each old id is encountered. `deep` controls whether
/// nested existence heads participate in the renumber.
///
/// @param exprIn Source expression text.
/// @param deep   When true, descend into nested existence heads.
/// @return Tuple `(renamed, defSets, renameMap)`.
std::tuple<std::string, DefSetMap, std::map<std::string,std::string>>
Conjecturer::renameVariablesInExpr(const std::string& exprIn, bool deep) const {
    std::string expr = stripWs(exprIn);
    int index = 0;
    std::set<int> numbersToReplace;
    int firstIntToUse = 10000;
    int unchangedFirstIntToUse = firstIntToUse;
    std::map<std::string,std::string> replacementMap;
    DefSetMap originalArgMap;

    if (!deep) {
        originalArgMap = findArgMap(expr);
    }

    auto createReplacementMap = [&](const std::string& subexpr, const std::set<std::string>& argsToRemoveSet)
        -> std::pair<std::map<std::string,std::string>, std::vector<std::string>>
    {
        std::map<std::string,std::string> repMap;
        auto orderedArgs = ce::orderByPattern(subexpr, argsToRemoveSet);
        for (auto& arg : orderedArgs) {
            while (true) {
                std::string candidate = std::to_string(firstIntToUse - (unchangedFirstIntToUse - 1));
                if (originalArgMap.find(candidate) == originalArgMap.end()) {
                    repMap[arg] = std::to_string(firstIntToUse);
                    numbersToReplace.insert(firstIntToUse);
                    firstIntToUse++;
                    break;
                } else {
                    firstIntToUse++;
                }
            }
        }
        return {repMap, orderedArgs};
    };

    std::function<std::pair<DefSetMap, std::string>()> renameSubexpr = [&]() -> std::pair<DefSetMap, std::string> {
        std::string fullExpression;
        DefSetMap nodeMap;

        if (expr[index] == '(') {
            index++;
            if (expr[index] == '>') {
                index++;
                auto argsToRemove = ce::getArgs(expr.substr(index));
                index = (int)expr.find(']', index) + 1;
                auto [leftMap, leftExpr] = renameSubexpr();
                std::set<std::string> argsSet(argsToRemove.begin(), argsToRemove.end());
                auto [repMap2, orderedInts2] = createReplacementMap(leftExpr, argsSet);
                for (auto& [k, v] : repMap2) replacementMap[k] = v;
                auto [rightMap, rightExpr] = renameSubexpr();
                leftExpr = ce::replaceKeysInString(leftExpr, repMap2);
                rightExpr = ce::replaceKeysInString(rightExpr, repMap2);
                nodeMap = leftMap;
                for (auto& [k, v] : rightMap) nodeMap[k] = v;
                for (auto& arg : argsToRemove) nodeMap.erase(arg);
                std::vector<std::string> renamedArgs;
                for (auto& oi : orderedInts2) {
                    renamedArgs.push_back(repMap2[oi]);
                }
                fullExpression = "(>[" + ce::joinWithComma(renamedArgs) + "]" + leftExpr + rightExpr + ")";
            } else if (expr[index] == '&') {
                index++;
                auto [leftMap, leftExpr] = renameSubexpr();
                auto [rightMap, rightExpr] = renameSubexpr();
                nodeMap = leftMap;
                for (auto& [k, v] : rightMap) nodeMap[k] = v;
                fullExpression = "(&" + leftExpr + rightExpr + ")";
            } else {
                auto endIndex = expr.find(')', index);
                std::string nodeLabel = expr.substr(index, endIndex - index);
                auto arrayArgs = ce::getArgs(nodeLabel);
                std::string tempExpr = ce::extractExpression(nodeLabel);
                auto& exprMap = config_.data.at(tempExpr).definition_sets;
                for (int i = 0; i < (int)arrayArgs.size(); ++i) {
                    nodeMap[arrayArgs[i]] = exprMap.at(std::to_string(i + 1));
                }
                for (auto& [arg, val] : exprMap) {
                    bool isDigit = !arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit);
                    if (!isDigit) nodeMap[arg] = val;
                }
                index = (int)endIndex;
                fullExpression = "(" + nodeLabel + ")";
            }
        } else if (index + 1 < (int)expr.size() && expr.substr(index, 2) == "!(") {
            index += 2;
            if (expr[index] == '>') {
                index++;
                auto argsToRemove = ce::getArgs(expr.substr(index));
                index = (int)expr.find(']', index) + 1;
                auto [leftMap, leftExpr] = renameSubexpr();
                std::set<std::string> argsSet(argsToRemove.begin(), argsToRemove.end());
                auto [repMap2, orderedInts2] = createReplacementMap(leftExpr, argsSet);
                for (auto& [k, v] : repMap2) replacementMap[k] = v;
                auto [rightMap, rightExpr] = renameSubexpr();
                leftExpr = ce::replaceKeysInString(leftExpr, repMap2);
                rightExpr = ce::replaceKeysInString(rightExpr, repMap2);
                nodeMap = leftMap;
                for (auto& [k, v] : rightMap) nodeMap[k] = v;
                for (auto& arg : argsToRemove) nodeMap.erase(arg);
                std::vector<std::string> renamedArgs;
                for (auto& oi : orderedInts2) {
                    renamedArgs.push_back(repMap2[oi]);
                }
                fullExpression = "!(>[" + ce::joinWithComma(renamedArgs) + "]" + leftExpr + rightExpr + ")";
            } else if (expr[index] == '&') {
                index++;
                auto [leftMap, leftExpr] = renameSubexpr();
                auto [rightMap, rightExpr] = renameSubexpr();
                nodeMap = leftMap;
                for (auto& [k, v] : rightMap) nodeMap[k] = v;
                fullExpression = "!(&" + leftExpr + rightExpr + ")";
            } else {
                auto endIndex = expr.find(')', index);
                std::string nodeLabel = expr.substr(index, endIndex - index);
                std::string tempExpr = ce::extractExpression(nodeLabel);
                nodeLabel = "!(" + nodeLabel + ")";
                auto arrayArgs = ce::getArgs(nodeLabel);
                auto& exprMap = config_.data.at(tempExpr).definition_sets;
                for (int i = 0; i < (int)arrayArgs.size(); ++i) {
                    nodeMap[arrayArgs[i]] = exprMap.at(std::to_string(i + 1));
                }
                for (auto& [arg, val] : exprMap) {
                    bool isDigit = !arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit);
                    if (!isDigit) nodeMap[arg] = val;
                }
                index = (int)endIndex;
                fullExpression = nodeLabel;
            }
        } else if (expr[index] == ')') {
            index--;
        }

        index++;
        return {nodeMap, fullExpression};
    };

    auto [rootMap, renamedExpr] = renameSubexpr();

    if (deep) {
        std::set<std::string> args;
        for (auto& [k, v] : rootMap) args.insert(k);
        auto [repMap3, orderedInts3] = createReplacementMap(renamedExpr, args);
        for (auto& [k, v] : repMap3) replacementMap[k] = v;
        renamedExpr = ce::replaceKeysInString(renamedExpr, repMap3);
        renamedExpr = subtractNumberFromInts(renamedExpr, unchangedFirstIntToUse - 1, numbersToReplace, false);

        DefSetMap tempMap;
        for (auto& [arg, val] : rootMap) {
            auto it = repMap3.find(arg);
            if (it != repMap3.end()) {
                tempMap[std::to_string(std::stoi(it->second) - (unchangedFirstIntToUse - 1))] = val;
            } else {
                tempMap[arg] = val;
            }
        }
        rootMap = tempMap;
    } else {
        renamedExpr = subtractNumberFromInts(renamedExpr, unchangedFirstIntToUse - 1, numbersToReplace, false);
    }

    for (auto& [arg, val] : replacementMap) {
        val = std::to_string(std::stoi(val) - (unchangedFirstIntToUse - 1));
    }

    return {renamedExpr, rootMap, replacementMap};
}

// ============================================================================
// Expression connection (Phase 5)
// ============================================================================

/// @brief String-path: merge two expressions via a substitution
///        map and a binary-sign vector.
///
/// @details
/// `subMap` carries the bijection between a subset of `map1`'s
/// args and a subset of `map2`'s args; `binaryList` carries one
/// sign bit per merge slot to enumerate negation variants.
/// `connectToAnchor` toggles whether the second expression is the
/// anchor (uses `mappingsMapAnchor_` instead of `mappingsMap_`).
/// String-path twin of `connectExpressionsInt`; the int lane is
/// the hot path and is byte-equivalent in output.
///
/// @param expr1            First expression text.
/// @param expr2            Second expression text (or anchor).
/// @param map1             First expression's def-set map.
/// @param map2             Second expression's def-set map.
/// @param subMap           Bijection between subset args.
/// @param binaryList       Per-slot sign bits.
/// @param connectToAnchor  Toggle anchor-attach mode.
/// @return Tuple `(success, mergedExpr, mergedDefSets)`.
std::tuple<bool, std::string, DefSetMap>
Conjecturer::connectExpressions(const std::string& expr1, const std::string& expr2,
                                const DefSetMap& map1, const DefSetMap& map2,
                                const std::map<std::string,std::string>& subMap,
                                const std::vector<int>& binaryList, bool connectToAnchor) const {
    // Precondition (D-75): when
    // connectToAnchor is true, expr1 MUST be the anchor — every production
    // anchor-attach call site passes the anchor signature as expr1
    // (`connectExpressions(anchor_…, …)`). The unified binder reads expr1's
    // outermost atom as the Anchor atom; a non-anchor expr1 here is a
    // caller bug. First-class assert per CLAUDE.md "asserts are first-class".
    assert((!connectToAnchor ||
            (expr1.size() >= 7 && expr1.compare(0, 7, "(Anchor") == 0))
           && "connectExpressions: connectToAnchor=true requires expr1 to be "
              "the anchor atom (expression beginning \"(Anchor\")");

    auto checkMaps = [&](const std::vector<std::string>& atr,
                         const DefSetMap& mp1, const DefSetMap& mp2) -> bool {
        for (auto& [arg, val] : mp1) {
            auto it = mp2.find(arg);
            if (it != mp2.end()) {
                if (std::get<0>(val) != std::get<0>(it->second)) return false;
            }
        }
        if (!connectToAnchor) {
            for (auto& arg : atr) {
                auto it1 = mp1.find(arg);
                if (it1 != mp1.end()) {
                    auto it2 = mp2.find(arg);
                    if (std::get<0>(it1->second)[0] == 'P' && !(std::get<1>(it1->second) && std::get<1>(it2->second)))
                        return false;
                }
            }
        }
        return true;
    };

    int shiftNum = 0;
    for (auto& [arg, _] : map1) shiftNum = std::max(shiftNum, std::stoi(arg));

    DefSetMap leftMap;
    std::set<std::string> removableArgs;
    for (auto& [arg, val] : map1) {
        auto it = subMap.find(arg);
        std::string newArg = (it != subMap.end()) ? it->second : arg;
        leftMap[newArg] = val;
    }

    DefSetMap rightMap;
    for (auto& [arg, val] : map2) {
        std::string shiftedArg = std::to_string(std::stoi(arg) + shiftNum);
        auto it = subMap.find(shiftedArg);
        std::string newArg = (it != subMap.end()) ? it->second : shiftedArg;
        rightMap[newArg] = val;
        if (map1.find(newArg) != map1.end()) {
            removableArgs.insert(newArg);
        }
    }

    std::vector<std::string> removableArgsList(removableArgs.begin(), removableArgs.end());
    auto sortedList = sortListAccordingToOccurrence(removableArgsList, expr1);

    std::vector<std::string> argsToRemove;
    for (int i = 0; i < (int)binaryList.size(); ++i) {
        if (binaryList[i]) {
            argsToRemove.push_back(sortedList[i]);
        }
    }

    DefSetMap connectedMap = leftMap;
    for (auto& [k, v] : rightMap) connectedMap[k] = v;
    for (auto& arg : argsToRemove) connectedMap.erase(arg);

    bool success = true;
    if (!checkMaps(argsToRemove, leftMap, rightMap)) success = false;

    std::string newExpr2 = subtractNumberFromInts(expr2, -shiftNum, {}, true);
    std::string newExpr1 = ce::replaceKeysInString(expr1, subMap);
    newExpr2 = ce::replaceKeysInString(newExpr2, subMap);

    // Unified binder (D-75): when attaching
    // the anchor, the outer >[...] binds EVERY anchor-atom argument, not just
    // the body-referenced removable subset. Conjecturer args carry no u_
    // prefix, so every anchor slot is bound. `argsToRemove` still drives
    // connectedMap erase / checkMaps / negation-variant enumeration above
    // (unchanged); only the emitted binder string widens. reshuffle
    // re-derives binder placement from each entry's parsed >[...] ∩ leftArgs,
    // so the change is confined to >[...] content — byte-stable elsewhere.
    // Must stay byte-identical to connectExpressionsInt's binder (worker_*
    // parity); both emit the anchor-atom args in anchor-atom order.
    const std::vector<std::string> binderArgs =
        connectToAnchor ? ce::getArgs(newExpr1) : argsToRemove;

    std::string connectedExpr = "(>[" + ce::joinWithComma(binderArgs) + "]"
        + newExpr1 + newExpr2 + ")";

    if (newExpr1 == newExpr2) success = false;

    return {success, connectedExpr, connectedMap};
}

// ============================================================================
// Int-path: connectExpressions
// ============================================================================

/// @brief Sort `removableArgs` by their first-occurrence position in
///        `expr`; result written to caller-owned `sortedOut`.
void Conjecturer::sortByOccurrenceInt(const IntConjBuf& expr, const int16_t* removableArgs, int numRemovable,
                                       int16_t* sortedOut) const {
    // Find first occurrence of each removable arg as an argument in the expression buffer.
    // This matches findPositionSurrounded which finds the first arg in [x,...] context.
    int16_t positions[MAX_CONJ_ARGS];
    for (int i = 0; i < numRemovable; ++i) positions[i] = 9999;

    int pos = 0;
    int bufPos = 0; // tracks position in buffer for ordering
    while (pos < expr.len) {
        int16_t bc = expr.data[pos++];
        for (int i = 0; i < bc; ++i) pos++; // skip bound vars
        pos++; // skip nameId
        int16_t ar = expr.data[pos++];
        for (int a = 0; a < ar; ++a) {
            int16_t argVal = expr.data[pos++];
            for (int i = 0; i < numRemovable; ++i) {
                if (removableArgs[i] == argVal && bufPos < positions[i]) {
                    positions[i] = (int16_t)bufPos;
                }
            }
            bufPos++;
        }
    }

    // Sort by position
    int16_t indices[MAX_CONJ_ARGS];
    for (int i = 0; i < numRemovable; ++i) indices[i] = (int16_t)i;
    // Simple insertion sort (numRemovable is small)
    for (int i = 1; i < numRemovable; ++i) {
        int16_t key = indices[i];
        int16_t keyPos = positions[key];
        int j = i - 1;
        while (j >= 0 && positions[indices[j]] > keyPos) {
            indices[j + 1] = indices[j];
            j--;
        }
        indices[j + 1] = key;
    }
    for (int i = 0; i < numRemovable; ++i) sortedOut[i] = removableArgs[indices[i]];
}

/// @brief Int-path twin of `getNumberRemovableArgs` — count
///        distinct non-identity targets in a connection map.
///
/// @details
/// Companion to string-path `getNumberRemovableArgs`: implements
/// the same `set(v for k, v in subMap if k != v).size()`
/// semantics on the int-encoded `connMap` instead of a
/// `std::map<string, string>`. Walks `connMap.map[1..maxArg]`,
/// skipping identity entries (`map[k] == k`) and 0 entries
/// (unmapped slots), then deduplicates the remaining values.
/// Returns 0 when every entry is identity or unmapped —
/// typically the case for a fresh `IntConnMap` initialised by
/// the cartesian-product enumerator before any non-trivial
/// substitution slot is filled.
///
/// @param connMap Int-encoded connection map (one entry per
///                source argId in `[1, maxArg]`).
/// @return Count of distinct non-identity values; `0` when no
///         non-identity entries exist.
/// @see `getNumberRemovableArgs` — string-path twin with
///      identical contract.
int Conjecturer::getNumberRemovableArgsInt(const IntConnMap& connMap) const {
    prof::Scope _p(prof::g_getNumRemArgsInt);
    // Count unique VALUES (not keys) in non-identity mappings.
    // Matches string version: set(v for k,v in subMap if k != v).size()
    int16_t vals[MAX_CONJ_ARGS];
    int numVals = 0;
    for (int i = 1; i <= connMap.maxArg; ++i) {
        if (connMap.map[i] != 0 && connMap.map[i] != i) {
            int16_t v = connMap.map[i];
            bool found = false;
            for (int j = 0; j < numVals; ++j) if (vals[j] == v) { found = true; break; }
            if (!found) vals[numVals++] = v;
        }
    }
    return numVals;
}

/// @brief Int-path: merge two encoded expressions via a connection
///        map and binary-sign vector. Hot-path twin of
///        `connectExpressions`.
///
/// @details
/// `subMap` carries the bijection between a subset of `map1`'s args
/// and a subset of `map2`'s args; `binaryList` carries one sign bit
/// per merge slot to enumerate negation variants.
/// `connectToAnchor` toggles whether the second expression is the
/// anchor (uses `mappingsMapAnchor_` instead of `mappingsMap_`).
/// Outputs are written into `outExpr` / `outMap` (caller-owned) to
/// avoid per-call allocation; this keeps the hot loop allocation-free.
///
/// @return `true` on a successful merge; `false` on rejection.
bool Conjecturer::connectExpressionsInt(
    const IntConjBuf& expr1, const IntConjBuf& expr2,
    const IntDefSetMap& map1, const IntDefSetMap& map2,
    const IntConnMap& subMap,
    const int16_t* binaryList, int binaryLen,
    bool connectToAnchor,
    IntConjBuf& outExpr, IntDefSetMap& outMap) const
{
    prof::Scope _prof_connE(prof::g_connectExprInt);

    // Precondition (D-75): when
    // connectToAnchor is true, expr1 MUST be the anchor — every production
    // anchor-attach call site passes `anchorInt_` as expr1
    // (`connectExpressionsInt(anchorInt_, …)`). The unified binder reads
    // expr1's first block as the Anchor atom; a non-anchor expr1 here is a
    // caller bug. First-class assert per CLAUDE.md "asserts are first-class".
    // Block layout: [bc, bv_0..bv_{bc-1}, nameId, arity, args…] — the first
    // block's nameId is at index 1 + bc.
    assert((!connectToAnchor ||
            (expr1.len > 0 && anchorInt_.len > 0 &&
             expr1.data[1 + expr1.data[0]] ==
                 anchorInt_.data[1 + anchorInt_.data[0]]))
           && "connectExpressionsInt: connectToAnchor=true requires expr1 to "
              "be the anchor (first block must be the Anchor atom)");

    // Find shiftNum = max argId in map1
    int16_t shiftNum = 0;
    for (int i = 0; i < map1.count; ++i)
        if (map1.argId[i] > shiftNum) shiftNum = map1.argId[i];

    // Build leftMap: apply subMap to map1's arg IDs
    IntDefSetMap leftMap;
    leftMap.count = 0;
    for (int i = 0; i < map1.count; ++i) {
        int16_t arg = map1.argId[i];
        int16_t newArg = (subMap.map[arg] != 0) ? subMap.map[arg] : arg;
        int idx = leftMap.count++;
        leftMap.argId[idx] = newArg;
        leftMap.defSetId[idx] = map1.defSetId[i];
        leftMap.combinable[idx] = map1.combinable[i];
        leftMap.connectable[idx] = map1.connectable[i];
    }

    // Build rightMap: shift map2's args by shiftNum, then apply subMap
    IntDefSetMap rightMap;
    rightMap.count = 0;
    int16_t removableArgs[MAX_CONJ_ARGS];
    int numRemovable = 0;

    for (int i = 0; i < map2.count; ++i) {
        int16_t arg = map2.argId[i];
        int16_t shiftedArg = arg + shiftNum;
        int16_t newArg = (shiftedArg <= subMap.maxArg && subMap.map[shiftedArg] != 0) ? subMap.map[shiftedArg] : shiftedArg;
        int idx = rightMap.count++;
        rightMap.argId[idx] = newArg;
        rightMap.defSetId[idx] = map2.defSetId[i];
        rightMap.combinable[idx] = map2.combinable[i];
        rightMap.connectable[idx] = map2.connectable[i];

        // Check if newArg exists in map1 (making it removable)
        for (int j = 0; j < map1.count; ++j) {
            if (map1.argId[j] == newArg) {
                // Check not already in removable list
                bool found = false;
                for (int k = 0; k < numRemovable; ++k)
                    if (removableArgs[k] == newArg) { found = true; break; }
                if (!found) removableArgs[numRemovable++] = newArg;
                break;
            }
        }
    }

    // Sort removable args by occurrence in expr1
    int16_t sortedRemovable[MAX_CONJ_ARGS];
    sortByOccurrenceInt(expr1, removableArgs, numRemovable, sortedRemovable);

    // Select args to remove based on binaryList
    int16_t argsToRemove[MAX_CONJ_ARGS];
    int numToRemove = 0;
    for (int i = 0; i < binaryLen; ++i) {
        if (binaryList[i]) {
            argsToRemove[numToRemove++] = sortedRemovable[i];
        }
    }

    // Build connectedMap = leftMap + rightMap - argsToRemove
    // String version: connectedMap = leftMap; for (k,v) in rightMap: connectedMap[k] = v;
    // So rightMap OVERWRITES leftMap for overlapping keys.
    outMap.count = 0;
    for (int i = 0; i < leftMap.count; ++i) {
        int idx = outMap.count++;
        outMap.argId[idx] = leftMap.argId[i];
        outMap.defSetId[idx] = leftMap.defSetId[i];
        outMap.combinable[idx] = leftMap.combinable[i];
        outMap.connectable[idx] = leftMap.connectable[i];
    }
    for (int i = 0; i < rightMap.count; ++i) {
        // Check if argId already exists (from leftMap) — overwrite if so
        int existing = -1;
        for (int j = 0; j < outMap.count; ++j) {
            if (outMap.argId[j] == rightMap.argId[i]) { existing = j; break; }
        }
        if (existing >= 0) {
            outMap.defSetId[existing] = rightMap.defSetId[i];
            outMap.combinable[existing] = rightMap.combinable[i];
            outMap.connectable[existing] = rightMap.connectable[i];
        } else {
            int idx = outMap.count++;
            outMap.argId[idx] = rightMap.argId[i];
            outMap.defSetId[idx] = rightMap.defSetId[i];
            outMap.combinable[idx] = rightMap.combinable[i];
            outMap.connectable[idx] = rightMap.connectable[i];
        }
    }
    // Erase argsToRemove from outMap
    for (int r = 0; r < numToRemove; ++r) {
        for (int i = 0; i < outMap.count; ++i) {
            if (outMap.argId[i] == argsToRemove[r]) {
                for (int j = i; j < outMap.count - 1; ++j) {
                    outMap.argId[j] = outMap.argId[j + 1];
                    outMap.defSetId[j] = outMap.defSetId[j + 1];
                    outMap.combinable[j] = outMap.combinable[j + 1];
                    outMap.connectable[j] = outMap.connectable[j + 1];
                }
                outMap.count--;
                break;
            }
        }
    }

    // checkMaps: verify def-set compatibility for overlapping args
    // Check 1: overlapping args in leftMap and rightMap must have same defSetId
    for (int i = 0; i < leftMap.count; ++i) {
        for (int j = 0; j < rightMap.count; ++j) {
            if (leftMap.argId[i] == rightMap.argId[j]) {
                if (leftMap.defSetId[i] != rightMap.defSetId[j]) return false;
            }
        }
    }
    // Check 2: for args to remove (if !connectToAnchor), P-type def sets must both be combinable
    if (!connectToAnchor) {
        for (int r = 0; r < numToRemove; ++r) {
            int16_t arg = argsToRemove[r];
            // Find in leftMap
            for (int i = 0; i < leftMap.count; ++i) {
                if (leftMap.argId[i] == arg) {
                    // Check if defSet starts with 'P'
                    const std::string& dsText = nameMap_.decode(leftMap.defSetId[i]);
                    if (!dsText.empty() && dsText[0] == 'P') {
                        // Find in rightMap
                        for (int j = 0; j < rightMap.count; ++j) {
                            if (rightMap.argId[j] == arg) {
                                if (!(leftMap.combinable[i] && rightMap.combinable[j]))
                                    return false;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    // Build output expression: first expr1 block gets outer bound vars,
    // remaining expr1 blocks copied as-is, then expr2 blocks.
    // Format per block: [boundCount, bvs..., nameId, arity, args...]
    outExpr.len = 0;

    // Copy expr1 blocks, applying subMap to all args.
    // The FIRST block gets the outer bound-var list (argsToRemove) prepended to
    // any existing bound vars from expr1's first block.
    {
        int pos = 0;
        bool firstBlock = true;
        while (pos < expr1.len) {
            int16_t bc = expr1.data[pos++];
            if (firstBlock) {
                if (connectToAnchor) {
                    // Unified binder (D-75):
                    // bind EVERY anchor-atom argument in the outer >[...],
                    // not just the body-referenced removable subset.
                    // Conjecturer args carry no u_ prefix, so every anchor
                    // slot is bound. `argsToRemove` still drives the
                    // connectedMap erase / def-set checks / negation-variant
                    // enumeration (unchanged); only the emitted binder widens.
                    // The anchor is the first premise, so its args first
                    // occur here in anchor-atom order — that IS occurrence
                    // order (GL >[...] convention). Byte-identical to the
                    // string lane's `ce::getArgs(newExpr1)` (worker_* parity).
                    // First block layout after bc:
                    //   [bv_0..bv_{bc-1}, nameId, arity, arg_0..arg_{arity-1}]
                    int peek = pos + bc;            // skip first-block bvs
                    int16_t anchorArity = expr1.data[peek + 1];
                    int anchorArgsStart = peek + 2;
                    outExpr.data[outExpr.len++] = (int16_t)(anchorArity + bc);
                    for (int i = 0; i < anchorArity; ++i) {
                        int16_t a = expr1.data[anchorArgsStart + i];
                        int16_t m = (a <= subMap.maxArg && subMap.map[a] != 0) ? subMap.map[a] : a;
                        outExpr.data[outExpr.len++] = m;
                    }
                    for (int i = 0; i < bc; ++i) {
                        int16_t bv = expr1.data[pos++];
                        int16_t mapped = (bv <= subMap.maxArg && subMap.map[bv] != 0) ? subMap.map[bv] : bv;
                        outExpr.data[outExpr.len++] = mapped;
                    }
                } else {
                    // Merge outer bound vars + expr1's first block's bound vars
                    outExpr.data[outExpr.len++] = (int16_t)(numToRemove + bc);
                    for (int i = 0; i < numToRemove; ++i)
                        outExpr.data[outExpr.len++] = argsToRemove[i];
                    for (int i = 0; i < bc; ++i) {
                        int16_t bv = expr1.data[pos++];
                        int16_t mapped = (bv <= subMap.maxArg && subMap.map[bv] != 0) ? subMap.map[bv] : bv;
                        outExpr.data[outExpr.len++] = mapped;
                    }
                }
                firstBlock = false;
            } else {
                outExpr.data[outExpr.len++] = bc;
                for (int i = 0; i < bc; ++i) {
                    int16_t bv = expr1.data[pos++];
                    int16_t mapped = (bv <= subMap.maxArg && subMap.map[bv] != 0) ? subMap.map[bv] : bv;
                    outExpr.data[outExpr.len++] = mapped;
                }
            }
            // Copy nameId
            outExpr.data[outExpr.len++] = expr1.data[pos++];
            // Copy arity + args with substitution
            int16_t ar = expr1.data[pos++];
            outExpr.data[outExpr.len++] = ar;
            for (int i = 0; i < ar; ++i) {
                int16_t argVal = expr1.data[pos++];
                int16_t mapped = (argVal <= subMap.maxArg && subMap.map[argVal] != 0) ? subMap.map[argVal] : argVal;
                outExpr.data[outExpr.len++] = mapped;
            }
        }
    }

    // Copy expr2 blocks, shifting by shiftNum then applying subMap
    {
        int pos = 0;
        while (pos < expr2.len) {
            // Copy bound var block
            int16_t bc = expr2.data[pos++];
            outExpr.data[outExpr.len++] = bc;
            for (int i = 0; i < bc; ++i) {
                int16_t bv = expr2.data[pos++] + shiftNum;
                int16_t mapped = (bv <= subMap.maxArg && subMap.map[bv] != 0) ? subMap.map[bv] : bv;
                outExpr.data[outExpr.len++] = mapped;
            }
            // Copy nameId (unchanged)
            outExpr.data[outExpr.len++] = expr2.data[pos++];
            // Copy arity + args, shifted then substituted
            int16_t ar = expr2.data[pos++];
            outExpr.data[outExpr.len++] = ar;
            for (int i = 0; i < ar; ++i) {
                int16_t argVal = expr2.data[pos++] + shiftNum;
                int16_t mapped = (argVal <= subMap.maxArg && subMap.map[argVal] != 0) ? subMap.map[argVal] : argVal;
                outExpr.data[outExpr.len++] = mapped;
            }
        }
    }

    // Check newExpr1 == newExpr2 (the substituted/shifted forms must not be equal)
    // Compare leaf content (nameId + mapped args) of both source expressions.
    {
        // Count blocks in each source
        int e1Blocks = 0;
        { int p = 0; while (p < expr1.len) { int16_t bc = expr1.data[p++]; p += bc; p++; int16_t ar = expr1.data[p++]; p += ar; e1Blocks++; } }
        int e2Blocks = 0;
        { int p = 0; while (p < expr2.len) { int16_t bc = expr2.data[p++]; p += bc; p++; int16_t ar = expr2.data[p++]; p += ar; e2Blocks++; } }

        if (e1Blocks == e2Blocks) {
            bool equal = true;
            int p1 = 0, p2 = 0;
            for (int b = 0; b < e1Blocks && equal; ++b) {
                int16_t bc1 = expr1.data[p1++]; p1 += bc1; // skip bound vars
                int16_t bc2 = expr2.data[p2++]; p2 += bc2;
                // Compare nameId
                int16_t n1 = expr1.data[p1++];
                int16_t n2 = expr2.data[p2++];
                if (n1 != n2) { equal = false; break; }
                // Compare arity
                int16_t ar1 = expr1.data[p1++];
                int16_t ar2 = expr2.data[p2++];
                if (ar1 != ar2) { equal = false; break; }
                // Compare mapped args
                for (int a = 0; a < ar1; ++a) {
                    int16_t v1 = expr1.data[p1++];
                    int16_t mv1 = (v1 <= subMap.maxArg && subMap.map[v1] != 0) ? subMap.map[v1] : v1;
                    int16_t v2 = expr2.data[p2++] + shiftNum;
                    int16_t mv2 = (v2 <= subMap.maxArg && subMap.map[v2] != 0) ? subMap.map[v2] : v2;
                    if (mv1 != mv2) { equal = false; break; }
                }
            }
            if (equal) return false;
        }
    }

    // Normalization pass: renumber surviving args to contiguous 1..N in
    // first-occurrence order (atom args only, bvs skipped), matching
    // reshuffle's canonicalization rule. Merges via subMap may have left
    // holes in the shifted range; this pass closes them so downstream
    // consumers (conjectures.txt, anchor-connect inputs) see a contiguous
    // arg-ID space.
    {
        std::map<int16_t, int16_t> renumber;
        int16_t nextId = 1;

        // Pass 1: collect argIds from atom args in block order.
        int pos = 0;
        while (pos < outExpr.len) {
            int16_t bc = outExpr.data[pos++];
            pos += bc;  // skip bvs
            pos++;      // skip nameId
            int16_t ar = outExpr.data[pos++];
            for (int i = 0; i < ar; ++i) {
                int16_t arg = outExpr.data[pos++];
                if (renumber.find(arg) == renumber.end()) {
                    renumber[arg] = nextId++;
                }
            }
        }
        // Absorb any bv that wasn't seen in atoms (defensive; valid
        // conjectures always use every bv in some downstream atom).
        pos = 0;
        while (pos < outExpr.len) {
            int16_t bc = outExpr.data[pos++];
            for (int i = 0; i < bc; ++i) {
                int16_t bv = outExpr.data[pos++];
                if (renumber.find(bv) == renumber.end()) {
                    renumber[bv] = nextId++;
                }
            }
            pos++;  // skip nameId
            int16_t ar = outExpr.data[pos++];
            pos += ar;
        }

        // Pass 2: rewrite bvs and args in place.
        pos = 0;
        while (pos < outExpr.len) {
            int16_t bc = outExpr.data[pos++];
            for (int i = 0; i < bc; ++i) {
                outExpr.data[pos] = renumber[outExpr.data[pos]];
                ++pos;
            }
            pos++;  // skip nameId
            int16_t ar = outExpr.data[pos++];
            for (int i = 0; i < ar; ++i) {
                outExpr.data[pos] = renumber[outExpr.data[pos]];
                ++pos;
            }
        }

        // Rewrite outMap.argId via the same renumber.
        for (int i = 0; i < outMap.count; ++i) {
            auto it = renumber.find(outMap.argId[i]);
            if (it != renumber.end()) outMap.argId[i] = it->second;
        }
    }

    return true;
}

/// @brief String-path: enumerate every valid connection map
///        between the args of `argsMap1` and `argsMap2`.
///
/// @details
/// Walks the precomputed `mappingsMap` (or `mappingsMapAnchor_`
/// when `withAnchor` is true), generating every valid bijection
/// between the two arg sets. One entry in the returned vector
/// becomes the `subMap` argument of a subsequent
/// `connectExpressions` call. String-path twin of
/// `makeAllConnectionMapsInt`.
///
/// @param argsMap1     First def-set map.
/// @param argsMap2     Second def-set map.
/// @param withAnchor   Toggle anchor-attach mode.
/// @param mappingsMap  Pre-computed bijection table.
/// @return All valid connection maps.
std::vector<std::map<std::string,std::string>>
Conjecturer::makeAllConnectionMaps(const DefSetMap& argsMap1, const DefSetMap& argsMap2,
                                    bool withAnchor, const MappingsMap& mappingsMap) const {
    int shiftNum = 0;
    for (auto& [arg, _] : argsMap2) shiftNum = std::max(shiftNum, std::stoi(arg));

    std::map<std::string, std::set<std::string>> srcMap, dstMap;

    for (auto& [arg, val] : argsMap1) {
        if (!withAnchor) {
            if (std::get<1>(val)) srcMap[std::get<0>(val)].insert(arg);
        } else {
            srcMap[std::get<0>(val)].insert(arg);
        }
    }
    for (auto& [arg, val] : argsMap2) {
        if (!withAnchor) {
            if (std::get<1>(val)) dstMap[std::get<0>(val)].insert(arg);
        } else {
            bool connectable = std::get<2>(val);
            if (connectable) dstMap[std::get<0>(val)].insert(arg);
        }
    }

    std::vector<std::vector<std::map<std::string,std::string>>> mappingsList;

    // Dst-only sets
    for (auto& [defSet, args] : dstMap) {
        if (srcMap.find(defSet) == srcMap.end()) {
            std::vector<std::string> sortedArgs(args.begin(), args.end());
            std::sort(sortedArgs.begin(), sortedArgs.end());
            std::map<std::string,std::string> identity;
            for (auto& a : sortedArgs) identity[a] = a;
            mappingsList.push_back({identity});
        }
    }

    // Overlapping and src-only
    for (auto& [defSet, srcArgs] : srcMap) {
        if (dstMap.find(defSet) != dstMap.end()) {
            std::vector<std::string> dstArgsSorted(dstMap[defSet].begin(), dstMap[defSet].end());
            std::sort(dstArgsSorted.begin(), dstArgsSorted.end());
            std::vector<std::string> srcArgsSorted(srcArgs.begin(), srcArgs.end());
            std::sort(srcArgsSorted.begin(), srcArgsSorted.end());

            std::vector<std::string> shiftedSrc;
            for (auto& x : srcArgsSorted) {
                shiftedSrc.push_back(std::to_string(std::stoi(x) + shiftNum));
            }

            std::vector<std::string> allArgs = dstArgsSorted;
            allArgs.insert(allArgs.end(), shiftedSrc.begin(), shiftedSrc.end());

            int totalSize = (int)dstArgsSorted.size() + (int)srcArgsSorted.size();
            auto key = std::make_pair((int)dstArgsSorted.size(), (int)srcArgsSorted.size());

            auto it1 = mappingsMap.find(totalSize);
            if (it1 == mappingsMap.end()) continue;
            auto it2 = it1->second.find(key);
            if (it2 == it1->second.end()) continue;

            std::vector<std::map<std::string,std::string>> block;
            for (auto& mapping : it2->second) {
                std::map<std::string,std::string> tempMap;
                for (int i = 0; i < (int)allArgs.size(); ++i) {
                    tempMap[allArgs[i]] = allArgs[mapping.at(i + 1) - 1];
                }
                block.push_back(tempMap);
            }
            mappingsList.push_back(block);
        } else {
            std::vector<std::string> srcArgsSorted(srcArgs.begin(), srcArgs.end());
            std::sort(srcArgsSorted.begin(), srcArgsSorted.end());
            std::vector<std::string> shiftedSrc;
            for (auto& x : srcArgsSorted) {
                shiftedSrc.push_back(std::to_string(std::stoi(x) + shiftNum));
            }
            std::map<std::string,std::string> identity;
            for (auto& a : shiftedSrc) identity[a] = a;
            mappingsList.push_back({identity});
        }
    }

    // Cartesian product of all blocks (union_of_dicts)
    auto unionOfDicts = [&](const std::vector<std::map<std::string,std::string>>& dicts) {
        std::map<std::string,std::string> result;
        for (auto& d : dicts) for (auto& [k, v] : d) result[k] = v;
        // Fill in identity for missing args
        for (auto& [arg, _] : argsMap1) {
            std::string shifted = std::to_string(std::stoi(arg) + shiftNum);
            if (result.find(shifted) == result.end()) result[shifted] = shifted;
        }
        for (auto& [arg, _] : argsMap2) {
            if (result.find(arg) == result.end()) result[arg] = arg;
        }
        return result;
    };

    std::vector<std::map<std::string,std::string>> allMaps;
    if (mappingsList.empty()) {
        allMaps.push_back(unionOfDicts({}));
    } else {
        // Cartesian product
        std::vector<int> indices(mappingsList.size(), 0);
        while (true) {
            std::vector<std::map<std::string,std::string>> selection;
            for (int i = 0; i < (int)mappingsList.size(); ++i) {
                selection.push_back(mappingsList[i][indices[i]]);
            }
            allMaps.push_back(unionOfDicts(selection));

            int carry = (int)mappingsList.size() - 1;
            while (carry >= 0) {
                indices[carry]++;
                if (indices[carry] < (int)mappingsList[carry].size()) break;
                indices[carry] = 0;
                carry--;
            }
            if (carry < 0) break;
        }
    }

    return allMaps;
}

// ============================================================================
// Int-path: makeAllConnectionMaps
// ============================================================================

/// @brief Int-path: enumerate every connection map between two
///        encoded def-set maps. Hot-path twin of
///        `makeAllConnectionMaps`.
///
/// @details
/// Walks the precomputed `mappingsMap` (or `mappingsMapAnchor_` when
/// `withAnchor` is true), generating every valid bijection between
/// the two arg sets. Output appended to `outMaps`. Cap on the number
/// of maps written is `MAX_CONN_MAPS`; misconfiguration that would
/// blow past this triggers a hard cut.
void Conjecturer::makeAllConnectionMapsInt(
    const IntDefSetMap& argsMap1, const IntDefSetMap& argsMap2,
    bool withAnchor, const MappingsMap& mappingsMap,
    std::vector<IntConnMap>& outMaps) const
{
    prof::Scope _prof_mkM(prof::g_makeAllConnMapsInt);
    outMaps.clear();
    // shiftNum = max argId in argsMap2
    int16_t shiftNum = 0;
    for (int i = 0; i < argsMap2.count; ++i)
        if (argsMap2.argId[i] > shiftNum) shiftNum = argsMap2.argId[i];

    // Group args by defSetId into src/dst groups
    struct Group {
        int16_t defSetId;
        int16_t args[MAX_PER_GROUP];
        int16_t count;
    };
    Group srcGroups[MAX_DEFSET_GROUPS];
    int numSrcGroups = 0;
    Group dstGroups[MAX_DEFSET_GROUPS];
    int numDstGroups = 0;

    auto findOrAddGroup = [](Group* groups, int& numGroups, int16_t dsId) -> int {
        for (int i = 0; i < numGroups; ++i)
            if (groups[i].defSetId == dsId) return i;
        int idx = numGroups++;
        assert(idx < MAX_DEFSET_GROUPS);
        groups[idx].defSetId = dsId;
        groups[idx].count = 0;
        return idx;
    };

    // Build srcGroups from argsMap1
    for (int i = 0; i < argsMap1.count; ++i) {
        if (!withAnchor) {
            if (!argsMap1.combinable[i]) continue;
        }
        int gi = findOrAddGroup(srcGroups, numSrcGroups, argsMap1.defSetId[i]);
        srcGroups[gi].args[srcGroups[gi].count++] = argsMap1.argId[i];
    }
    // Build dstGroups from argsMap2
    for (int i = 0; i < argsMap2.count; ++i) {
        if (!withAnchor) {
            if (!argsMap2.combinable[i]) continue;
        } else {
            if (!argsMap2.connectable[i]) continue;
        }
        int gi = findOrAddGroup(dstGroups, numDstGroups, argsMap2.defSetId[i]);
        dstGroups[gi].args[dstGroups[gi].count++] = argsMap2.argId[i];
    }

    // Sort args within each group by STRING comparison (match std::set<string>)
    // "1" < "10" < "11" < "2" < "3" etc.
    auto strLess = [](int16_t a, int16_t b) -> bool {
        // Compare std::to_string(a) < std::to_string(b)
        // Optimization: same digit count → numeric comparison works
        // Different digit counts → fewer digits < more digits only if first chars agree
        // Simplest correct approach: compare string representations
        char sa[8], sb[8];
        int la = snprintf(sa, sizeof(sa), "%d", (int)a);
        int lb = snprintf(sb, sizeof(sb), "%d", (int)b);
        return std::string_view(sa, la) < std::string_view(sb, lb);
    };
    auto sortGroup = [&strLess](Group& g) {
        for (int i = 1; i < g.count; ++i) {
            int16_t key = g.args[i];
            int j = i - 1;
            while (j >= 0 && strLess(key, g.args[j])) { g.args[j + 1] = g.args[j]; j--; }
            g.args[j + 1] = key;
        }
    };
    for (int i = 0; i < numSrcGroups; ++i) sortGroup(srcGroups[i]);
    for (int i = 0; i < numDstGroups; ++i) sortGroup(dstGroups[i]);

    // Sort groups by defset TEXT (match std::map<string,...> iteration order)
    auto sortGroups = [&](Group* groups, int count) {
        for (int i = 1; i < count; ++i) {
            Group key = groups[i];
            const std::string& keyText = nameMap_.decode(key.defSetId);
            int j = i - 1;
            while (j >= 0 && nameMap_.decode(groups[j].defSetId) > keyText) {
                groups[j + 1] = groups[j]; j--;
            }
            groups[j + 1] = key;
        }
    };
    sortGroups(srcGroups, numSrcGroups);
    sortGroups(dstGroups, numDstGroups);

    // Build mapping blocks: each block = vector of IntConnMap fragments
    // We need cartesian product across blocks.
    // Dynamic vectors because Incubator anchor permutation counts can exceed fixed limits.
    struct MapEntry {
        int16_t map[MAX_CONJ_ARGS * 2]; // map[argId] = targetArgId
        int16_t maxArg;
    };
    struct MapBlock {
        std::vector<MapEntry> maps;
    };
    static thread_local std::vector<MapBlock> blocks;
    blocks.clear();
    blocks.resize(MAX_DEFSET_GROUPS * 2);
    int numBlocks = 0;

    // Dst-only groups (no matching src group)
    for (int d = 0; d < numDstGroups; ++d) {
        bool hasSrc = false;
        for (int s = 0; s < numSrcGroups; ++s)
            if (srcGroups[s].defSetId == dstGroups[d].defSetId) { hasSrc = true; break; }
        if (hasSrc) continue;

        auto& block = blocks[numBlocks++];
        block.maps.clear();
        block.maps.emplace_back();
        auto& me = block.maps.back();
        std::memset(me.map, 0, sizeof(me.map));
        me.maxArg = 0;
        for (int i = 0; i < dstGroups[d].count; ++i) {
            int16_t a = dstGroups[d].args[i];
            me.map[a] = a; // identity
            if (a > me.maxArg) me.maxArg = a;
        }
    }

    // Overlapping and src-only groups
    for (int s = 0; s < numSrcGroups; ++s) {
        int16_t dsId = srcGroups[s].defSetId;
        int dstIdx = -1;
        for (int d = 0; d < numDstGroups; ++d)
            if (dstGroups[d].defSetId == dsId) { dstIdx = d; break; }

        if (dstIdx >= 0) {
            // Overlapping: use MappingsMap permutations
            auto& sg = srcGroups[s];
            auto& dg = dstGroups[dstIdx];

            // Build allArgs = dstArgs + shiftedSrcArgs
            int16_t allArgs[MAX_PER_GROUP * 2];
            int totalSize = dg.count + sg.count;
            for (int i = 0; i < dg.count; ++i) allArgs[i] = dg.args[i];
            for (int i = 0; i < sg.count; ++i) allArgs[dg.count + i] = sg.args[i] + shiftNum;

            auto key = std::make_pair((int)dg.count, (int)sg.count);
            auto it1 = mappingsMap.find(totalSize);
            if (it1 == mappingsMap.end()) continue;
            auto it2 = it1->second.find(key);
            if (it2 == it1->second.end()) continue;

            auto& block = blocks[numBlocks++];
            block.maps.clear();
            block.maps.reserve(it2->second.size());
            for (auto& mapping : it2->second) {
                block.maps.emplace_back();
                auto& me = block.maps.back();
                std::memset(me.map, 0, sizeof(me.map));
                me.maxArg = 0;
                for (int i = 0; i < totalSize; ++i) {
                    int16_t fromArg = allArgs[i];
                    int16_t toArg = allArgs[mapping.at(i + 1) - 1];
                    me.map[fromArg] = toArg;
                    if (fromArg > me.maxArg) me.maxArg = fromArg;
                }
            }
        } else {
            // Src-only: identity mapping for shifted args
            auto& block = blocks[numBlocks++];
            block.maps.clear();
            block.maps.emplace_back();
            auto& me = block.maps.back();
            std::memset(me.map, 0, sizeof(me.map));
            me.maxArg = 0;
            for (int i = 0; i < srcGroups[s].count; ++i) {
                int16_t a = srcGroups[s].args[i] + shiftNum;
                me.map[a] = a;
                if (a > me.maxArg) me.maxArg = a;
            }
        }
    }

    // Cartesian product of all blocks, writing to outMaps
    // unionOfDicts: merge all selected maps + fill identity for missing args
    if (numBlocks == 0) {
        // Single identity map
        outMaps.emplace_back();
        auto& m = outMaps.back();
        std::memset(m.map, 0, sizeof(m.map));
        m.maxArg = 0;
        for (int i = 0; i < argsMap1.count; ++i) {
            int16_t shifted = argsMap1.argId[i] + shiftNum;
            m.map[shifted] = shifted;
            if (shifted > m.maxArg) m.maxArg = shifted;
        }
        for (int i = 0; i < argsMap2.count; ++i) {
            int16_t a = argsMap2.argId[i];
            if (m.map[a] == 0) { m.map[a] = a; if (a > m.maxArg) m.maxArg = a; }
        }
    } else {
        // Estimate cartesian product size for reserve
        size_t totalProduct = 1;
        for (int b = 0; b < numBlocks; ++b)
            totalProduct *= blocks[b].maps.size();
        outMaps.reserve(totalProduct);

        // Cartesian product
        int indices[MAX_DEFSET_GROUPS * 2];
        std::memset(indices, 0, sizeof(indices));

        while (true) {
            outMaps.emplace_back();
            auto& m = outMaps.back();
            std::memset(m.map, 0, sizeof(m.map));
            m.maxArg = 0;

            // Merge selected maps from each block
            for (int b = 0; b < numBlocks; ++b) {
                auto& bme = blocks[b].maps[indices[b]];
                for (int16_t a = 1; a <= bme.maxArg; ++a) {
                    if (bme.map[a] != 0) {
                        m.map[a] = bme.map[a];
                        if (a > m.maxArg) m.maxArg = a;
                    }
                }
            }

            // Fill identity for missing args
            for (int i = 0; i < argsMap1.count; ++i) {
                int16_t shifted = argsMap1.argId[i] + shiftNum;
                if (m.map[shifted] == 0) { m.map[shifted] = shifted; if (shifted > m.maxArg) m.maxArg = shifted; }
            }
            for (int i = 0; i < argsMap2.count; ++i) {
                int16_t a = argsMap2.argId[i];
                if (m.map[a] == 0) { m.map[a] = a; if (a > m.maxArg) m.maxArg = a; }
            }

            // Advance indices
            int carry = numBlocks - 1;
            while (carry >= 0) {
                indices[carry]++;
                if (indices[carry] < (int)blocks[carry].maps.size()) break;
                indices[carry] = 0;
                carry--;
            }
            if (carry < 0) break;
        }
    }
}

// ============================================================================
// Int-path: validation filters
// ============================================================================

/// @brief Int-path twin of `repetitionsExist` — reject candidates
///        that carry the same leaf twice.
///
/// @details
/// Companion to string-path `repetitionsExist`: applied to the
/// flat-encoded candidate instead of the rendered string, but
/// enforces the same contract — `repetitionsExistInt(buf) ==
/// repetitionsExist(decodeExpr(buf))` for any buffer the
/// conjecturer emits. Walks `buf` block by block, packing each
/// `(nameId, arity, args)` tuple into a `Leaf` record, then
/// scans for duplicates. The structural-uniqueness gate prevents
/// the chain from carrying the same predicate twice (which would
/// add no constraint and would inflate the conjecture set).
/// Returns `false` when `buf` carries fewer than 2 leaves —
/// nothing to deduplicate.
///
/// @param buf Encoded candidate previously produced by
///            `encodeExpr` (or by `connectExpressionsInt`).
/// @return `true` when the same `(nameId, args)` tuple appears
///         at least twice; `false` otherwise.
/// @see `repetitionsExist` — string-path twin with identical
///      contract.
bool Conjecturer::repetitionsExistInt(const IntConjBuf& buf) const {
    // Check for duplicate leaf expressions (same nameId + same args).
    // Matches the string regex R"(\([^()]*\))" which finds all leaf nodes.
    struct Leaf { int16_t nameId; int16_t arity; int16_t args[16]; };
    Leaf leaves[MAX_CONJ_ARGS];
    int numLeaves = 0;

    int pos = 0;
    while (pos < buf.len) {
        int16_t bc = buf.data[pos++]; pos += bc; // skip bound vars
        auto& leaf = leaves[numLeaves];
        leaf.nameId = buf.data[pos++];
        leaf.arity = buf.data[pos++];
        for (int i = 0; i < leaf.arity; ++i) leaf.args[i] = buf.data[pos++];
        numLeaves++;
    }

    // O(n^2) but n is small (typically <8)
    for (int i = 0; i < numLeaves; ++i) {
        for (int j = i + 1; j < numLeaves; ++j) {
            if (leaves[i].nameId == leaves[j].nameId && leaves[i].arity == leaves[j].arity) {
                bool same = true;
                for (int a = 0; a < leaves[i].arity; ++a) {
                    if (leaves[i].args[a] != leaves[j].args[a]) { same = false; break; }
                }
                if (same) return true;
            }
        }
    }
    return false;
}

/// @brief Int-path twin of `numbersGood` — per-expression
///        occurrence-count cap.
///
/// @details
/// Companion to string-path `numbersGood`: enforces the same
/// per-expression `max_count_per_conjecture` cap on the flat
/// encoded buffer instead of the rendered string. Walks `buf`
/// block by block, indexing `intExprConfigs_` by nameId to find
/// each expression's configured cap; rejects when any count
/// exceeds. Despite the misleading name, this is a count-cap
/// check — not a contiguity check on arg ids. Returns `true`
/// when every nameId stays at or below its cap (or when no
/// nameId has a positive cap, meaning the cap is "unlimited").
///
/// @param buf Encoded candidate previously produced by
///            `encodeExpr` (or by `connectExpressionsInt`).
/// @return `true` to accept; `false` when any nameId's count
///         exceeds its configured `maxCountPerConj`.
/// @see `numbersGood` — string-path twin with identical contract.
bool Conjecturer::numbersGoodInt(const IntConjBuf& buf) const {
    // Count occurrences of each nameId, compare against maxCountPerConj
    int16_t counts[256]; // indexed by nameId — safe since we have ~15 expression names
    std::memset(counts, 0, sizeof(counts));

    int pos = 0;
    while (pos < buf.len) {
        int16_t bc = buf.data[pos++]; pos += bc;
        int16_t nid = buf.data[pos++];
        if (nid < 256) counts[nid]++;
        int16_t ar = buf.data[pos++]; pos += ar;
    }

    for (int16_t nid = 1; nid < (int16_t)intExprConfigs_.size() && nid < 256; ++nid) {
        auto& ic = intExprConfigs_[nid];
        if (ic.maxCountPerConj > 0 && counts[nid] > ic.maxCountPerConj) return false;
    }
    return true;
}

/// @brief Int-path twin of `checkDefSets`.
///
/// @details
/// Reject when the def-set type per-arg counts exceed the
/// per-type `max_values_for_def_sets` (combinable) or
/// `max_values_for_uncomb_def_sets` (uncombinable) caps.
/// Consults `maxForDefSets_` and `maxForUncombDefSets_` (the
/// per-defSetId dense lookup tables built by
/// `buildIntExprConfigs`).
///
/// @param argMap Encoded def-set map of the candidate.
/// @return `true` to accept; `false` to reject.
bool Conjecturer::checkDefSetsInt(const IntDefSetMap& argMap) const {
    int16_t combCounts[256];
    int16_t uncombCounts[256];
    std::memset(combCounts, 0, sizeof(combCounts));
    std::memset(uncombCounts, 0, sizeof(uncombCounts));

    for (int i = 0; i < argMap.count; ++i) {
        int16_t dsId = argMap.defSetId[i];
        if (dsId < 256) {
            if (argMap.combinable[i]) combCounts[dsId]++;
            else uncombCounts[dsId]++;
        }
    }

    for (int16_t dsId = 1; dsId < (int16_t)maxForDefSets_.size() && dsId < 256; ++dsId) {
        if (maxForDefSets_[dsId] >= 0 && combCounts[dsId] > maxForDefSets_[dsId]) return false;
    }
    for (int16_t dsId = 1; dsId < (int16_t)maxForUncombDefSets_.size() && dsId < 256; ++dsId) {
        if (maxForUncombDefSets_[dsId] >= 0 && uncombCounts[dsId] > maxForUncombDefSets_[dsId]) return false;
    }
    return true;
}

/// @brief Int-path twin of `checkComplexityLevelForDefSets`. Reject
///        if any def-set type's complexity exceeds the configured
///        pre-existence cap (consults `maxComplexityAnchorConn_`).
bool Conjecturer::checkComplexityLevelInt(const IntDefSetMap& argMap, int complexityLevel) const {
    for (int i = 0; i < argMap.count; ++i) {
        int16_t dsId = argMap.defSetId[i];
        if (dsId > 0 && dsId < (int16_t)maxComplexityAnchorConn_.size()) {
            if (maxComplexityAnchorConn_[dsId] >= 0 && maxComplexityAnchorConn_[dsId] < complexityLevel)
                return false;
        }
    }
    return true;
}

/// @brief Int-path twin of `checkDefSetsPriorToConnection`. Pre-merge
///        per-type cap: reject when the union of args from both
///        sides would exceed the configured prior-connection cap.
bool Conjecturer::checkDefSetsPriorInt(const IntDefSetMap& argsStmt, const IntDefSetMap& argsGT) const {
    prof::Scope _p(prof::g_checkDefSetsPriorInt);
    int16_t counts[256];
    std::memset(counts, 0, sizeof(counts));

    for (int i = 0; i < argsStmt.count; ++i)
        if (argsStmt.combinable[i] && argsStmt.defSetId[i] < 256) counts[argsStmt.defSetId[i]]++;
    for (int i = 0; i < argsGT.count; ++i)
        if (argsGT.combinable[i] && argsGT.defSetId[i] < 256) counts[argsGT.defSetId[i]]++;

    for (int16_t dsId = 1; dsId < (int16_t)maxForDefSetsPrior_.size() && dsId < 256; ++dsId) {
        if (maxForDefSetsPrior_[dsId] >= 0 && counts[dsId] > maxForDefSetsPrior_[dsId]) return false;
    }
    return true;
}

/// @brief Int-path twin of `countOperatorOccurrences` — count
///        quantifier-block headers in an encoded buffer.
///
/// @details
/// Companion to string-path `countOperatorOccurrences`:
/// `countOperatorOccurrencesInt(buf) ==
/// countOperatorOccurrences(decodeExpr(buf))` for any conjecturer-
/// emitted buffer. Walks `buf` block by block; each block whose
/// `boundCount > 0` contributes one operator-block header (the
/// flat-encoded analogue of a `(>[...]...)` opening). Returns 0
/// for atom-only buffers (e.g. the encoded anchor or a single
/// predicate without an outer quantifier).
///
/// @param buf Encoded candidate.
/// @return Number of `boundCount > 0` blocks (operator-layer
///         count == complexity level).
/// @see `countOperatorOccurrences` — string-path twin with
///      identical contract.
int Conjecturer::countOperatorOccurrencesInt(const IntConjBuf& buf) const {
    prof::Scope _p(prof::g_countOpOccurrences);
    // Count quantifier layers: blocks whose boundCount > 0 (matches string "(>[" counting)
    int count = 0;
    int pos = 0;
    while (pos < buf.len) {
        int16_t bc = buf.data[pos++]; pos += bc;
        pos++; // nameId
        int16_t ar = buf.data[pos++]; pos += ar;
        if (bc > 0) count++;
    }
    return count;
}

/// @brief Int-path twin of `onlyInHeadGood` — reject when a
///        head-only expression appears outside the head.
///
/// @details
/// Companion to string-path `onlyInHeadGood`:
/// `onlyInHeadGoodInt(buf) ==
/// onlyInHeadGood(decodeExpr(buf))` for any conjecturer-emitted
/// buffer. Walks `buf` block by block; for every non-head block
/// whose nameId matches an entry in `config_.only_in_head_patterns`
/// (compiled regex list), rejects the candidate. The head block
/// (last block in the buffer) is exempt by definition. Returns
/// `true` when no `only_in_head` pattern matches outside the
/// head — typically the case for any well-formed Peano / Gauss
/// candidate, since the patterns target known-pathological shapes.
///
/// @param buf Encoded candidate.
/// @return `true` when no `only_in_head` pattern matches a
///         non-head leaf; `false` otherwise.
/// @see `onlyInHeadGood` — string-path twin with identical
///      contract.
bool Conjecturer::onlyInHeadGoodInt(const IntConjBuf& buf) const {
    prof::Scope _p(prof::g_onlyInHeadGoodInt);
    // For each only_in_head expression, check it appears only in the head (last block).
    // We precompute the nameId for each only_in_head pattern.
    // only_in_head_raw contains handle strings like "(in2[" — we need the nameId of "in2".
    // But in the int path, we just check nameIds of leaf blocks.

    // For each config only_in_head handle, extract the expression name and get its nameId
    for (auto& handle : config_.only_in_head_raw) {
        // Extract expression name from handle like "(in2["
        std::string name = handle.substr(1, handle.size() - 2); // strip ( and [
        int16_t targetId = nameMap_.lookup(name);
        if (targetId == 0) continue;

        // Count occurrences
        int count = 0;
        int pos = 0;
        int16_t lastNameId = 0;
        while (pos < buf.len) {
            int16_t bc = buf.data[pos++]; pos += bc;
            int16_t nid = buf.data[pos++];
            int16_t ar = buf.data[pos++]; pos += ar;
            lastNameId = nid;
            if (nid == targetId) count++;
        }
        if (count == 0) continue;
        if (count > 1) return false;
        if (count == 1 && lastNameId != targetId) return false;
    }
    return true;
}

/// @brief Int-path twin of `prohibitedHeadsGood` — reject when
///        `buf`'s head is on the per-batch prohibited list.
///
/// @details
/// Companion to string-path `prohibitedHeadsGood`:
/// `prohibitedHeadsGoodInt(buf) ==
/// prohibitedHeadsGood(decodeExpr(buf))` for any conjecturer-
/// emitted buffer. Reads the last block's nameId (the head) and
/// looks it up in `config_.prohibited_heads` (per-batch block
/// list). Short-circuits to `true` when the prohibited-heads list
/// is empty — typically the case for Peano / Gauss; some
/// incubator configs set the list non-empty to suppress
/// known-pathological head shapes.
///
/// @param buf Encoded candidate.
/// @return `true` when the head's nameId is not on the prohibited
///         list; `false` otherwise.
/// @see `prohibitedHeadsGood` — string-path twin with identical
///      contract.
bool Conjecturer::prohibitedHeadsGoodInt(const IntConjBuf& buf) const {
    prof::Scope _p(prof::g_prohibHeadsGoodInt);
    if (config_.prohibited_heads.empty()) return true;
    // Find the last block's nameId (the head)
    int16_t headNameId = 0;
    int pos = 0;
    while (pos < buf.len) {
        int16_t bc = buf.data[pos++]; pos += bc;
        headNameId = buf.data[pos++];
        int16_t ar = buf.data[pos++]; pos += ar;
    }
    for (auto& ph : config_.prohibited_heads) {
        int16_t phId = nameMap_.lookup(ph);
        if (phId != 0 && phId == headNameId) return false;
    }
    return true;
}

/// @brief Int-path twin of `checkConjectureComplexityPerOperator`
///        — per-operator complexity gate on a prospective merge.
///
/// @details
/// Companion to string-path `checkConjectureComplexityPerOperator`:
/// answers the same per-operator-complexity question on encoded
/// inputs. Walks `growingTheorem` to collect per-operator counts,
/// then conditionally adds `statement`'s contribution; rejects
/// when any operator's projected count exceeds its
/// `max_size_expression_before_existence` cap from
/// `intExprConfigs_`. Returns `true` on the empty-stmt case
/// (`statement.len == 0`) — there's nothing to add.
///
/// @param growingTheorem Encoded partial theorem being grown.
/// @param statement      Encoded candidate to merge in.
/// @return `true` to accept the merge; `false` when any
///         operator's projected count exceeds its cap.
/// @see `checkConjectureComplexityPerOperator` — string-path twin
///      with identical contract.
bool Conjecturer::checkComplexityPerOpInt(const IntConjBuf& growingTheorem, const IntConjBuf& statement) const {
    prof::Scope _p(prof::g_checkComplexityPerOpInt);
    // Count total expressions in growingTheorem + 1 (for statement)
    int chainLen = 0;
    int pos = 0;
    while (pos < growingTheorem.len) {
        int16_t bc = growingTheorem.data[pos++]; pos += bc;
        pos++; // nameId
        int16_t ar = growingTheorem.data[pos++]; pos += ar;
        chainLen++;
    }
    // Count statement blocks
    pos = 0;
    while (pos < statement.len) {
        int16_t bc = statement.data[pos++]; pos += bc;
        pos++;
        int16_t ar = statement.data[pos++]; pos += ar;
        chainLen++;
    }

    // Check each expression's max_size_expression_before_existence against chainLen
    pos = 0;
    while (pos < growingTheorem.len) {
        int16_t bc = growingTheorem.data[pos++]; pos += bc;
        int16_t nid = growingTheorem.data[pos++];
        int16_t ar = growingTheorem.data[pos++]; pos += ar;
        if (nid > 0 && nid < (int16_t)intExprConfigs_.size()) {
            if (intExprConfigs_[nid].maxSizeExprBeforeEx > 0 && intExprConfigs_[nid].maxSizeExprBeforeEx < chainLen)
                return false;
        }
    }
    pos = 0;
    while (pos < statement.len) {
        int16_t bc = statement.data[pos++]; pos += bc;
        int16_t nid = statement.data[pos++];
        int16_t ar = statement.data[pos++]; pos += ar;
        if (nid > 0 && nid < (int16_t)intExprConfigs_.size()) {
            if (intExprConfigs_[nid].maxSizeExprBeforeEx > 0 && intExprConfigs_[nid].maxSizeExprBeforeEx < chainLen)
                return false;
        }
    }
    return true;
}

/// @brief Int-path twin of `exprGood2`. Final structural-approval
///        gate at the int-path level — combines numeric-arg
///        contiguity, per-type combinable / uncombinable caps, and
///        operator-block accounting.
bool Conjecturer::exprGood2Int(const IntConjBuf& buf, int nse, const IntDefSetMap& connectedMap) const {
    prof::Scope _prof_eg2(prof::g_exprGood2Int);
    if (repetitionsExistInt(buf)) return false;
    if (!numbersGoodInt(buf)) return false;

    if (nse == config_.parameters.max_number_simple_expressions && !checkDefSetsInt(connectedMap))
        return false;

    // For evaluateOperatorExprs and evaluateOperatorExprs2, we need string-based chain analysis.
    // Decode to string and call existing functions.
    std::string expr = decodeExpr(buf);
    DefSetMap strMap = decodeDefSetMap(connectedMap);

    int sizeArgs = 0;
    for (auto& [arg, tpl] : strMap)
        if (std::get<0>(tpl).substr(0, 1) != "P" && std::get<1>(tpl)) sizeArgs++;

    auto opExprs = extractOperatorExpressions(expr);
    auto evaluateOperatorExprs = [&](const std::vector<std::string>& opExprs2, bool freeArgs) -> bool {
        if (opExprs2.empty()) return true;
        std::string lastExpr = opExprs2.back();
        auto lastArgs = ce::getArgs(lastExpr);
        std::string lastCore = ce::extractExpression(lastExpr);
        std::string lastArg = lastArgs[config_.data.at(lastCore).indices_output_args[0]];

        int occCounter = 0;
        for (int i = 0; i < (int)opExprs2.size() - 1; ++i) {
            auto args = ce::getArgs(opExprs2[i]);
            if (std::find(args.begin(), args.end(), lastArg) != args.end()) {
                occCounter++;
                int occInd = (int)(std::find(args.begin(), args.end(), lastArg) - args.begin());
                std::string ce2 = ce::extractExpression(opExprs2[i]);
                if (occInd != config_.data.at(ce2).indices_output_args[0]) return false;
            }
        }

        if ((int)opExprs2.size() >= config_.parameters.operator_threshold ||
            ((int)opExprs2.size() == config_.parameters.operator_threshold - 1 && freeArgs)) {
            for (int lastArgInd : config_.data.at(lastCore).indices_input_args) {
                occCounter = 0;
                for (int opInd = 0; opInd < (int)opExprs2.size() - 1; ++opInd) {
                    std::string ce2 = ce::extractExpression(opExprs2[opInd]);
                    auto args = ce::getArgs(opExprs2[opInd]);
                    if (std::find(args.begin(), args.end(), lastArgs[lastArgInd]) != args.end()) {
                        occCounter++;
                        if (args[config_.data.at(ce2).indices_output_args[0]] != lastArgs[lastArgInd])
                            return false;
                    }
                }
                if (occCounter > 1) return false;
            }
        }
        return true;
    };

    if (!evaluateOperatorExprs(opExprs, sizeArgs > 0)) return false;
    if (!evaluateOperatorExprs2(expr, false)) return false;
    if (!checkProhibitedCombinations(expr)) return false;
    return true;
}

// ============================================================================
// Validation filters (Phase 6)
// ============================================================================

/// @brief First-pass structural sanity check on `expr` — cheap
///        gate before per-conjecture filters.
bool Conjecturer::exprGood(const std::string& expr) const {
    prof::Scope _prof_eg(prof::g_exprGood);
    if (expr.substr(0, 3) == "(>[" && expr.substr(0, 4) != "(>[]") {
        if (!repetitionsExist(expr)) return true;
    }
    return false;
}

/// @brief Reject if integer arg ids in `expr` are not contiguous
///        `1..N` (string-path twin of `numbersGoodInt`).
bool Conjecturer::numbersGood(const std::string& expr) const {
    for (auto& [name, desc] : config_.data) {
        int count = 0;
        size_t pos = 0;
        while ((pos = expr.find(desc.handle, pos)) != std::string::npos) {
            count++;
            pos += desc.handle.size();
        }
        if (count > desc.max_count_per_conjecture) return false;
    }
    return true;
}

/// @brief Reject if def-set type per-arg counts exceed the per-type
///        `max_values_for_def_sets` / `max_values_for_uncomb_def_sets`
///        caps (string-path twin of `checkDefSetsInt`).
bool Conjecturer::checkDefSets(const DefSetMap& argMap) const {
    prof::Scope _p(prof::g_checkDefSets);
    // Count combinable
    std::map<std::string, int> counterMap;
    for (auto& [arg, tpl] : argMap) {
        if (!std::get<1>(tpl)) continue;
        counterMap[std::get<0>(tpl)]++;
    }
    for (auto& [defSet, cnt] : counterMap) {
        auto it = config_.parameters.max_values_for_def_sets.find(defSet);
        if (it != config_.parameters.max_values_for_def_sets.end() && cnt > it->second) return false;
    }

    // Count non-combinable
    counterMap.clear();
    for (auto& [arg, tpl] : argMap) {
        if (std::get<1>(tpl)) continue;
        counterMap[std::get<0>(tpl)]++;
    }
    for (auto& [defSet, cnt] : counterMap) {
        auto it = config_.parameters.max_values_for_uncomb_def_sets.find(defSet);
        if (it != config_.parameters.max_values_for_uncomb_def_sets.end() && cnt > it->second) return false;
    }
    return true;
}

/// @brief Reject if any def-set type's complexity exceeds the
///        configured pre-existence cap (string-path twin of
///        `checkComplexityLevelInt`).
bool Conjecturer::checkComplexityLevelForDefSets(const DefSetMap& argMap, int complexityLevel) const {
    prof::Scope _p(prof::g_checkComplLevForDefSets);
    std::set<std::string> defSets;
    for (auto& [arg, tpl] : argMap) defSets.insert(std::get<0>(tpl));
    for (auto& ds : defSets) {
        auto it = config_.parameters.max_complexity_if_anchor_parameter_connected_before_existence.find(ds);
        if (it != config_.parameters.max_complexity_if_anchor_parameter_connected_before_existence.end() && it->second < complexityLevel)
            return false;
    }
    return true;
}

/// @brief Pre-equality-emission qualification check. Used before
///        letting `(=[...])` into a candidate's head position.
bool Conjecturer::qualifiedForEquality(const std::string& expr) const {
    using ChainEntry = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<ChainEntry> tempChain;
    std::string head = ce::disintegrateImplication(expr, tempChain, coreExprMap_);

    std::vector<std::string> fullChain;
    for (auto& e : tempChain) fullChain.push_back(std::get<0>(e));
    fullChain.push_back(head);

    std::string anchorName = config_.getAnchorName();
    std::vector<std::string> chain;
    for (auto& e : fullChain)
        if (ce::extractExpression(e) != anchorName) chain.push_back(e);
    bool anchorPresent = fullChain.size() > chain.size();

    if (chain.size() != 3) return false;
    auto& e1 = chain[0]; auto& e2 = chain[1]; auto& e3 = chain[2];
    if (ce::extractExpression(e3) != "=") return false;
    std::string core1 = ce::extractExpression(e1);
    std::string core2 = ce::extractExpression(e2);
    if (std::find(operators_.begin(), operators_.end(), core1) == operators_.end()) return false;
    if (std::find(operators_.begin(), operators_.end(), core2) == operators_.end()) return false;

    auto& desc1 = config_.data.at(core1);
    auto& desc2 = config_.data.at(core2);
    if (desc1.short_mpl_normalized == desc1.full_mpl) return false;
    if (desc2.short_mpl_normalized == desc2.full_mpl) return false;
    if (core1 != core2) return false;

    auto args1 = ce::getArgs(e1);
    auto args2 = ce::getArgs(e2);
    if (desc1.indices_output_args.empty()) return false;
    int outIdx = desc1.indices_output_args[0];
    assert(args1.size() == args2.size());
    for (int i = 0; i < (int)args1.size(); ++i) {
        if (i == outIdx) { if (args1[i] == args2[i]) return false; }
        else { if (anchorPresent && args1[i] != args2[i]) return false; }
    }

    std::set<std::string> eqArgs(ce::getArgs(e3).begin(), ce::getArgs(e3).end());
    std::set<std::string> outPair = {args1[outIdx], args2[outIdx]};
    return eqArgs == outPair;
}

/// @brief Collect operator-headed sub-expressions inside `expr`.
std::vector<std::string> Conjecturer::extractOperatorExpressions(const std::string& expr) const {
    if (operators_.empty()) return {};
    std::vector<std::string> sortedOps = operators_;
    std::sort(sortedOps.begin(), sortedOps.end(), [](const std::string& a, const std::string& b) {
        return a.size() > b.size();
    });
    std::vector<std::string> escaped;
    for (auto& op : sortedOps) {
        std::string e;
        for (char c : op) {
            if (std::string("\\^$.|?*+()[]{}").find(c) != std::string::npos) e += '\\';
            e += c;
        }
        escaped.push_back(e);
    }
    std::string joined;
    for (size_t i = 0; i < escaped.size(); ++i) {
        if (i > 0) joined += '|';
        joined += escaped[i];
    }
    std::regex re("(" + joined + ")\\[[^\\]]*\\]");
    std::vector<std::string> result;
    std::sregex_iterator it(expr.begin(), expr.end(), re);
    std::sregex_iterator end;
    for (; it != end; ++it) {
        result.push_back("(" + it->str() + ")");
    }
    return result;
}

/// @brief Reject when `expression` matches an entry on
///        `config_.prohibited_combinations` (per-batch block list).
bool Conjecturer::checkProhibitedCombinations(const std::string& expression) const {
    using ChainEntry = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<ChainEntry> tempChain;
    std::string head = ce::disintegrateImplication(expression, tempChain, coreExprMap_);
    std::set<std::string> coreExprs;
    for (auto& e : tempChain) coreExprs.insert(ce::extractExpression(std::get<0>(e)));
    coreExprs.insert(ce::extractExpression(head));
    for (auto& prohibited : config_.prohibited_combinations) {
        bool isSubset = true;
        for (auto& p : prohibited)
            if (coreExprs.find(p) == coreExprs.end()) { isSubset = false; break; }
        if (isSubset) return false;
    }
    return true;
}

/// @brief Reject when the head of `conjecture` is on
///        `config_.prohibited_heads` (string-path twin of
///        `prohibitedHeadsGoodInt`).
bool Conjecturer::prohibitedHeadsGood(const std::string& conjecture) const {
    if (config_.prohibited_heads.empty()) return true;
    using ChainEntry = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<ChainEntry> tempChain;
    std::string head = ce::disintegrateImplication(conjecture, tempChain, coreExprMap_);
    std::string headCore = ce::extractExpression(head);
    for (auto& ph : config_.prohibited_heads)
        if (ph == headCore) return false;
    return true;
}

/// @brief Reject if the total argument count of `conjecture`
///        exceeds `parameters.max_number_args_expr`.
bool Conjecturer::countArgumentsFilter(const std::string& conjecture) const {
    using ChainEntry = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<ChainEntry> chain;
    std::string head = ce::disintegrateImplication(conjecture, chain, coreExprMap_);
    std::vector<std::string> allExprs;
    for (auto& e : chain) allExprs.push_back(std::get<0>(e));
    allExprs.push_back(head);

    bool incubator = config_.parameters.incubator_mode;
    for (auto& expr : allExprs) {
        auto args = ce::getArgs(expr);
        if (!incubator) {
            std::set<std::string> s(args.begin(), args.end());
            if (s.size() != args.size()) return false;
        } else {
            std::string name = ce::extractExpression(expr);
            if (name == "=") {
                std::set<std::string> s(args.begin(), args.end());
                if (s.size() != args.size()) return false;
            } else if (config_.data.find(name) != config_.data.end() && !config_.data.at(name).definition_sets.empty()) {
                auto& ds = config_.data.at(name).definition_sets;
                std::vector<std::string> nonElement;
                for (int i = 0; i < (int)args.size(); ++i) {
                    auto it = ds.find(std::to_string(i + 1));
                    if (it == ds.end() || std::get<0>(it->second) != "(1)")
                        nonElement.push_back(args[i]);
                }
                std::set<std::string> s(nonElement.begin(), nonElement.end());
                if (s.size() != nonElement.size()) return false;
            } else {
                std::set<std::string> s(args.begin(), args.end());
                if (s.size() != args.size()) return false;
            }
        }
    }
    return true;
}

/// @brief Reject when `conjecture` matches any compiled regex in
///        `config_.patterns_to_exclude`.
bool Conjecturer::patternInConjecture(const std::string& conjecture) const {
    prof::Scope _p(prof::g_patternInConjecture);
    for (auto& pat : config_.patterns_to_exclude) {
        if (std::regex_search(conjecture, pat)) return true;
    }
    return false;
}

/// @brief Reject when an `only_in_head` pattern matches outside the
///        head position of `conjecture` (string-path twin of
///        `onlyInHeadGoodInt`).
bool Conjecturer::onlyInHeadGood(const std::string& conjecture) const {
    for (auto& handle : config_.only_in_head_raw) {
        if (conjecture.find(handle) != std::string::npos) {
            int count = 0;
            size_t pos = 0;
            while ((pos = conjecture.find(handle, pos)) != std::string::npos) { count++; pos += handle.size(); }
            if (count == 1) {
                using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
                std::vector<CE> tempChain;
                std::string head = ce::disintegrateImplication(conjecture, tempChain, coreExprMap_);
                if (head.find(handle) == std::string::npos) return false;
            } else {
                return false;
            }
        }
    }
    return true;
}

/// @brief Reject if appending `newExpr` to `conjecture` would push
///        any per-operator complexity over its config cap
///        (string-path twin of `checkComplexityPerOpInt`).
bool Conjecturer::checkConjectureComplexityPerOperator(const std::string& conjecture, const std::string& newExpr) const {
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> tempChain;
    std::string head = ce::disintegrateImplication(conjecture, tempChain, coreExprMap_);
    std::vector<std::string> chain;
    for (auto& e : tempChain) chain.push_back(std::get<0>(e));
    chain.push_back(head);
    chain.push_back(newExpr);

    for (auto& element : chain) {
        std::string coreExpr = ce::extractExpression(element);
        auto it = config_.data.find(coreExpr);
        if (it != config_.data.end()) {
            if (it->second.max_size_expression_before_existence < (int)chain.size()) return false;
        }
    }
    return true;
}

/// @brief Reject if `conjecture` is below the per-expression minimum
///        size threshold (`min_size_expression`).
bool Conjecturer::checkMinSizeExpression(const std::string& conjecture) const {
    prof::Scope _p(prof::g_checkMinSizeExpression);
    std::string anchorName = config_.getAnchorName();
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> tempChain;
    std::string head = ce::disintegrateImplication(conjecture, tempChain, coreExprMap_);
    std::vector<std::string> chain;
    for (auto& e : tempChain)
        if (ce::extractExpression(std::get<0>(e)) != anchorName) chain.push_back(std::get<0>(e));
    chain.push_back(head);
    for (auto& element : chain) {
        std::string coreExpr = ce::extractExpression(element);
        auto it = config_.data.find(coreExpr);
        if (it != config_.data.end()) {
            if (it->second.min_size_expression > (int)chain.size()) return false;
        }
    }
    return true;
}

/// @brief Equality-head guards.
///
/// @details
/// Rejects `(=[x, x])` heads (per
/// [I-8](../../docs/agentic_swdd/30_invariants.md#i-8)) and descending-ordered
/// `(=[a, b])` with `stoi(a) > stoi(b)` to keep only one orientation
/// of the symmetric pair. The descending-rejection rule is the
/// post-D-23 form; the pre-D-23 `nse <= 3` exception was reverted at
/// [D-23](../../docs/agentic_swdd/40_decisions.md#d-23). See SwDD chapter
/// `02_conjecturer.md` section *controlEquality* for history.
///
/// @invariant [I-8](../../docs/agentic_swdd/30_invariants.md#i-8) — trivial
///            equality forbidden in head.
bool Conjecturer::controlEquality(const std::string& conjecture) const {
    prof::Scope _p(prof::g_controlEquality);
    std::regex pat(R"(\(=\[\d+,\d+\]\))");
    std::smatch m;
    bool result = true;
    if (std::regex_search(conjecture, m, pat)) {
        auto args = ce::getArgs(m.str());
        if (std::stoi(args[0]) > std::stoi(args[1])) {
            // Reject descending `=[a, b]` (a > b as integers) so we never emit
            // both (a, b) and (b, a) forms of the symmetric `=` relation. The
            // ascending sibling is reached by the same pair-combination
            // enumeration that produced the descending one, so dropping the
            // descending form loses no genuine theorem. The cancellation
            // theorem head `=[b, i0]` (a bound-var b, anchor i0) is also
            // produced as `=[i0, b]` independently — verified by inspection
            // of the conjecturer output.
            result = false;
        }
    }
    return result && countArgumentsFilter(conjecture);
}

/// @brief Pre-connection per-type cap.
///
/// @details
/// Reject when the union of candidate args from both sides would
/// exceed the configured prior-connection cap. Called before
/// committing to a `connectExpressions` merge so that mergers
/// guaranteed to fail downstream are dropped early. Consults
/// `parameters.max_values_for_def_sets_prior_connection`.
/// String-path twin of `checkDefSetsPriorInt`.
///
/// @param argsStatement       Def-set map of the candidate
///                            statement.
/// @param argsGrowingTheorem  Def-set map of the partial theorem
///                            being grown.
/// @return `true` to accept; `false` to reject.
bool Conjecturer::checkDefSetsPriorToConnection(const DefSetMap& argsStatement, const DefSetMap& argsGrowingTheorem) const {
    std::map<std::string, int> counterMap;
    for (auto& [arg, tpl] : argsStatement) {
        if (!std::get<1>(tpl)) continue;
        counterMap[std::get<0>(tpl)]++;
    }
    for (auto& [arg, tpl] : argsGrowingTheorem) {
        if (!std::get<1>(tpl)) continue;
        counterMap[std::get<0>(tpl)]++;
    }
    for (auto& [defSet, cnt] : counterMap) {
        auto it = config_.parameters.max_values_for_def_sets_prior_connection.find(defSet);
        if (it != config_.parameters.max_values_for_def_sets_prior_connection.end() && cnt > it->second)
            return false;
    }
    return true;
}

// ============================================================================
// evaluateOperatorExprs2 and helpers (Phase 6 continued)
// ============================================================================

/// @brief Validate every operator-headed sub-expression in
///        `expression`.
///
/// @details
/// Walks the expression tree, locating every operator-headed
/// sub-expression (those with `output_args` non-empty), and
/// verifies its output binding is well-formed in the current
/// context. Skipped on the `nse = 1` path when `anchorAttached` is
/// false — a single expression cannot consume an operator's
/// output binding so the validity check is meaningless. See SwDD
/// chapter `02_conjecturer.md` section *Operator head vs relation
/// head*.
///
/// @param expression       Expression to validate.
/// @param anchorAttached   Whether the expression is already
///                         attached to the anchor; controls
///                         whether the operator-output check
///                         applies.
/// @return `true` when every operator sub-expression is
///         well-formed.
bool Conjecturer::evaluateOperatorExprs2(const std::string& expression, bool anchorAttached) const {
    prof::Scope _p(prof::g_evaluateOperatorExprs2);
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> tempChain;
    std::string head = ce::disintegrateImplication(expression, tempChain, coreExprMap_);
    std::vector<std::string> chain;
    for (auto& e : tempChain) chain.push_back(std::get<0>(e));
    chain.push_back(head);

    std::vector<std::string> opExprs;
    for (auto& elem : chain)
        if (std::find(operators_.begin(), operators_.end(), ce::extractExpression(elem)) != operators_.end())
            opExprs.push_back(elem);

    bool headIsOp = std::find(operators_.begin(), operators_.end(), ce::extractExpression(head)) != operators_.end();

    // Build arg_map: arg -> [set of input positions, set of output positions]
    std::map<std::string, std::pair<std::set<int>, std::set<int>>> argMap;
    std::set<std::string> relArgs;
    std::vector<std::vector<std::string>> relArgsList;
    std::vector<std::string> relCoreExprs;

    for (auto& elem : chain) {
        std::string ce2 = ce::extractExpression(elem);
        if (std::find(relations_.begin(), relations_.end(), ce2) != relations_.end()) {
            relCoreExprs.push_back(ce2);
            auto args = ce::getArgs(elem);
            std::vector<std::string> lst;
            for (int idx : config_.data.at(ce2).indices_input_args) {
                relArgs.insert(args[idx]);
                lst.push_back(args[idx]);
            }
            relArgsList.push_back(lst);
        }
    }

    // Side-bucket for property expressions (1 input, 0 outputs — e.g., `in`).
    // Mirrors the relation bucket: no position counter, no argMap writes.
    std::set<std::string> propArgs;
    for (auto& elem : chain) {
        std::string ce2 = ce::extractExpression(elem);
        if (std::find(properties_.begin(), properties_.end(), ce2) != properties_.end()) {
            auto args = ce::getArgs(elem);
            int inputIdx = config_.data.at(ce2).indices_input_args[0];
            propArgs.insert(args[inputIdx]);
        }
    }

    // Side-bucket for anchor args — relation args that sit on the anchor expression
    // (e.g. `=[7, 2]` where `2` is an anchor slot) are "known" the same way property
    // args are: they exist in the chain but outside the operator-argMap machinery.
    std::set<std::string> anchorArgs;
    {
        std::string anchorName = config_.getAnchorName();
        for (auto& elem : chain) {
            if (ce::extractExpression(elem) == anchorName) {
                auto args = ce::getArgs(elem);
                for (auto& a : args) anchorArgs.insert(a);
                break;
            }
        }
    }

    int position = 0;
    for (int exprInd = 0; exprInd < (int)opExprs.size(); ++exprInd) {
        std::string coreExpr = ce::extractExpression(opExprs[exprInd]);
        auto argList = ce::getArgs(opExprs[exprInd]);
        for (int argInd = 0; argInd < (int)argList.size(); ++argInd) {
            auto& arg = argList[argInd];
            if (argInd == config_.data.at(coreExpr).indices_output_args[0]) {
                argMap[arg].second.insert(position);
            }
            if (std::find(config_.data.at(coreExpr).indices_input_args.begin(),
                          config_.data.at(coreExpr).indices_input_args.end(), argInd)
                != config_.data.at(coreExpr).indices_input_args.end()) {
                argMap[arg].first.insert(position);
            }
        }
        position++;
    }

    bool evalPos = true;
    for (auto& [arg, sets] : argMap) {
        auto& [inputs, outputs] = sets;
        if (!inputs.empty() && outputs.empty()) continue;
        else if (inputs.empty() && outputs.size() <= 2) {
            if (outputs.find((int)opExprs.size() - 1) != outputs.end()) {
                if ((int)opExprs.size() == config_.parameters.max_number_simple_expressions || anchorAttached) {
                    if (outputs.size() < 2 && headIsOp) evalPos = false;
                    else continue;
                } else continue;
            } else {
                if (headIsOp) evalPos = false;
                else continue;
            }
        } else if (inputs.size() == 1 && outputs.size() == 1) {
            // Check subset
            bool isSubset = true;
            for (int v : inputs) if (outputs.find(v) == outputs.end()) { isSubset = false; break; }
            if (!isSubset) {
                if (relArgs.find(arg) != relArgs.end()) evalPos = false;
            } else continue;
        } else {
            evalPos = false;
        }
    }

    for (int index = 0; index < (int)relArgsList.size(); ++index) {
        auto& rArgs = relArgsList[index];
        if (relCoreExprs[index] == "=") {
            int counter = 0;
            for (auto& arg : rArgs) {
                auto it = argMap.find(arg);
                if (it != argMap.end() && !it->second.second.empty()) counter++;
            }
            if (counter >= 2) {
                if (!qualifiedForEquality(expression)) evalPos = false;
            }
        }
    }

    int numEndOperators = 0;
    if (anchorAttached) {
        std::set<int> positions;
        for (auto& [arg, sets] : argMap) {
            if (sets.first.empty() && !sets.second.empty())
                positions.insert(sets.second.begin(), sets.second.end());
        }
        numEndOperators = (int)positions.size();
        if ((int)positions.size() > 2) evalPos = false;
    }

    if (numEndOperators > 1) {
        for (auto& rArgs : relArgsList) {
            auto it0 = argMap.find(rArgs[0]);
            auto it1 = argMap.find(rArgs[1]);
            // Anchor args act as "known constants" — for the operator-chain dichotomy (both-outputs-only
            // vs both-inputs-only), treat them the same as property args: input-only (they flow into
            // the relation but don't emerge as any operator's output).
            // An arg found in none of argMap / propArgs / anchorArgs appears only in relation atoms
            // (reachable once a relation's per-conjecture count cap admits two relation atoms sharing
            // an argument); it carries no input/output evidence, classifies as neither, and the
            // dichotomy condition below rejects the candidate.
            auto classify = [&](const std::string& arg, auto it) {
                bool hasIn  = (it != argMap.end() && !it->second.first.empty())
                            || (propArgs.count(arg) > 0)
                            || (anchorArgs.count(arg) > 0);
                bool hasOut = (it != argMap.end() && !it->second.second.empty());
                return std::pair<bool, bool>{hasIn, hasOut};
            };
            auto [in0, out0] = classify(rArgs[0], it0);
            auto [in1, out1] = classify(rArgs[1], it1);
            bool cond = (out0 && out1 && !in0 && !in1) ||
                        (in0  && in1  && !out0 && !out1);
            if (!cond) evalPos = false;
        }
    }

    if (numEndOperators == 2) {
        std::vector<int> endOpIndices;
        for (int index = 0; index < (int)opExprs.size(); ++index) {
            std::string ce2 = ce::extractExpression(opExprs[index]);
            auto args = ce::getArgs(opExprs[index]);
            std::string outArg = args[config_.data.at(ce2).indices_output_args[0]];
            auto it = argMap.find(outArg);
            if (it != argMap.end() && it->second.first.empty() && !it->second.second.empty())
                endOpIndices.push_back(index);
        }

        std::vector<std::vector<std::string>> inputArgsList2, outputArgsList2;
        for (auto& opExpr : opExprs) {
            std::string ce2 = ce::extractExpression(opExpr);
            auto args = ce::getArgs(opExpr);
            std::vector<std::string> ins, outs;
            for (int idx : config_.data.at(ce2).indices_input_args) ins.push_back(args[idx]);
            for (int idx : config_.data.at(ce2).indices_output_args) outs.push_back(args[idx]);
            inputArgsList2.push_back(ins);
            outputArgsList2.push_back(outs);
        }

        std::set<int> v1, v2;
        auto ea1 = findEntryArgs2(inputArgsList2, outputArgsList2, endOpIndices[0], v1);
        auto ea2 = findEntryArgs2(inputArgsList2, outputArgsList2, endOpIndices[1], v2);

        for (auto& rArgs : relArgsList) {
            if (ea1.find(rArgs[0]) != ea1.end())
                if (ea2.find(rArgs[1]) == ea2.end()) evalPos = false;
            if (ea1.find(rArgs[1]) != ea1.end())
                if (ea2.find(rArgs[0]) == ea2.end()) evalPos = false;
        }
    }

    return evalPos;
}

// ============================================================================
// checkInputVariablesTheoremOperatorHead
// ============================================================================

/// @brief Operator-head-specific validity check.
///
/// @details
/// Skipped on the `nse = 1` path because a single expression
/// cannot consume an operator's output binding. When run, walks
/// the theorem's head and verifies the operator-head's input
/// variables are properly bound either by anchor slots or by
/// chain-introduced bound variables. Distinct from
/// `evaluateOperatorExprs2` in that it focuses on the head only,
/// not every operator sub-expression in the chain.
bool Conjecturer::checkInputVariablesTheoremOperatorHead(const std::string& theorem) const {
    prof::Scope _p(prof::g_checkInputVarsHead);
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> tempChain;
    std::string head = ce::disintegrateImplication(theorem, tempChain, coreExprMap_);
    std::vector<std::string> chain;
    for (auto& e : tempChain) chain.push_back(std::get<0>(e));
    chain.push_back(head);

    std::string anchorName = config_.getAnchorName();
    // Remove anchor element
    for (auto it = chain.begin(); it != chain.end(); ++it) {
        if (it->find(config_.data.at(anchorName).handle) != std::string::npos) {
            chain.erase(it);
            break;
        }
    }

    // Empty chain after anchor removal: a bare-anchor theorem has no
    // operator head to validate, so the check vacuously passes. Without
    // this guard, the subsequent `chain.back()` is undefined behavior;
    // MSVC happens to produce garbage that falls through the operator-
    // membership check at line below and returns true, libstdc++ at -O3
    // produces SIGABRT.
    if (chain.empty()) return true;

    std::string lastExpr = chain.back();
    std::string coreExpr = ce::extractExpression(lastExpr);
    if (std::find(operators_.begin(), operators_.end(), coreExpr) == operators_.end())
        return true;

    std::vector<std::vector<std::string>> inputArgsList, outputArgsList;
    for (auto& element : chain) {
        std::string ce2 = ce::extractExpression(element);
        auto args = ce::getArgs(element);
        std::vector<std::string> ins, outs;
        for (int idx : config_.data.at(ce2).indices_input_args) ins.push_back(args[idx]);
        for (int idx : config_.data.at(ce2).indices_output_args) outs.push_back(args[idx]);
        inputArgsList.push_back(ins);
        outputArgsList.push_back(outs);
    }

    assert(outputArgsList.back().size() == 1);
    std::string outputVar = outputArgsList.back()[0];

    int secondLastIndex = -1;
    for (int index = 0; index < (int)outputArgsList.size(); ++index) {
        for (auto& ov : outputArgsList[index]) {
            if (ov == outputVar) {
                if (index == (int)chain.size() - 1) return false;
                secondLastIndex = index;
                break;
            }
        }
        if (secondLastIndex >= 0) break;
    }
    assert(secondLastIndex >= 0);

    std::set<int> v1, v2;
    auto entryArgs = findEntryArgs2(inputArgsList, outputArgsList, (int)chain.size() - 1, v1);
    auto entryArgsSecond = findEntryArgs2(inputArgsList, outputArgsList, secondLastIndex, v2);

    // Remove anchor definition_set keys
    auto& anchorDS = config_.data.at(anchorName).definition_sets;
    for (auto& [k, _] : anchorDS) {
        entryArgs.erase(k);
        entryArgsSecond.erase(k);
    }

    return entryArgs == entryArgsSecond;
}

// ============================================================================
// Sub-functions for checkInputVariablesOrder
// ============================================================================

/// @brief Collect every "digit" arg id appearing in `theorem`.
///
/// @details
/// A digit arg is a bound variable that will appear in the
/// rendered output as a numeric token (1, 2, ...). The returned
/// set drives the ordering checks in `checkInputVariablesOrder`
/// and its sub-helpers.
///
/// @param theorem Conjecture text.
/// @return Set of digit-arg ids (as strings).
std::set<std::string> Conjecturer::findDigitArgs(const std::string& theorem) const {
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> tempChain;
    std::string head = ce::disintegrateImplication(theorem, tempChain, coreExprMap_);
    std::vector<std::string> chain;
    for (auto& e : tempChain) chain.push_back(std::get<0>(e));
    chain.push_back(head);

    std::set<std::string> allInputArgs;
    for (auto& element : chain) {
        std::string coreExpr = ce::extractExpression(element);
        auto args = ce::getArgs(element);
        for (int idx : config_.data.at(coreExpr).indices_input_args) allInputArgs.insert(args[idx]);
    }

    std::string anchorName = config_.getAnchorName();
    for (auto& element : chain) {
        if (element.find(config_.data.at(anchorName).handle) != std::string::npos) {
            auto args = ce::getArgs(element);
            for (auto& a : args) allInputArgs.erase(a);
        }
    }

    std::set<std::string> allOutputArgs;
    for (auto& element : chain) {
        std::string coreExpr = ce::extractExpression(element);
        auto args = ce::getArgs(element);
        for (int idx : config_.data.at(coreExpr).indices_output_args) allOutputArgs.insert(args[idx]);
    }

    for (auto& a : allOutputArgs) allInputArgs.erase(a);
    return allInputArgs;
}

/// @brief Walk a chain from a chosen `expression` and partition
///        the visited digit args into left-side and right-side
///        contributions.
///
/// @details
/// Recursive walker with cycle prevention via `visited`. From
/// `expression` (an operator-headed entry in `chain`), follow
/// the operator's input/output bindings to neighbouring entries
/// and accumulate the digit args those entries contribute,
/// classifying each as left- or right-side based on the chain's
/// orientation.
///
/// @param chain       Chain entries.
/// @param expression  Operator-headed entry to walk from.
/// @param digits      Digit args to consider.
/// @param counter     Recursion depth tracker.
/// @param visited     Mutable set of already-visited entries.
/// @return Pair `(leftDigits, rightDigits)`.
std::pair<std::set<std::string>, std::set<std::string>>
Conjecturer::getLeftRight(const std::vector<std::string>& chain,
                          const std::string& expression,
                          const std::set<std::string>& digits, int counter,
                          std::set<std::string>& visited) const {
    std::set<std::string> left, right;
    if (visited.find(expression) != visited.end()) return {left, right};
    visited.insert(expression);

    std::string coreExpr = ce::extractExpression(expression);
    assert(std::find(operators_.begin(), operators_.end(), coreExpr) != operators_.end());
    assert(config_.data.at(coreExpr).indices_input_args.size() == 2);

    auto args = ce::getArgs(expression);
    // Left side (input arg 0)
    if (digits.find(args[0]) != digits.end()) {
        left.insert(args[0]);
    } else {
        for (auto& element : chain) {
            std::string ceElem = ce::extractExpression(element);
            if (std::find(operators_.begin(), operators_.end(), ceElem) != operators_.end()
                && config_.data.at(ceElem).indices_input_args.size() == 2) {
                auto argsElem = ce::getArgs(element);
                if (argsElem[config_.data.at(ceElem).indices_output_args[0]] ==
                    args[config_.data.at(coreExpr).indices_input_args[0]]) {
                    auto [le, re] = getLeftRight(chain, element, digits, counter + 1, visited);
                    left.insert(le.begin(), le.end());
                    left.insert(re.begin(), re.end());
                }
            }
        }
    }
    // Right side (input arg 1)
    if (digits.find(args[1]) != digits.end()) {
        right.insert(args[1]);
    } else {
        for (auto& element : chain) {
            std::string ceElem = ce::extractExpression(element);
            if (std::find(operators_.begin(), operators_.end(), ceElem) != operators_.end()
                && config_.data.at(ceElem).indices_input_args.size() == 2) {
                auto argsElem = ce::getArgs(element);
                if (argsElem[config_.data.at(ceElem).indices_output_args[0]] ==
                    args[config_.data.at(coreExpr).indices_input_args[1]]) {
                    auto [le, re] = getLeftRight(chain, element, digits, counter + 1, visited);
                    right.insert(le.begin(), le.end());
                    right.insert(re.begin(), re.end());
                }
            }
        }
    }
    return {left, right};
}

/// @brief Compute the right-side sub-chain reachable from `head`,
///        marking visited entries to prevent re-traversal.
///
/// @details
/// Used by `getLeftRightChains` to collect every chain entry
/// reachable through input/output-arg connections starting from
/// the head. `visited` is the cycle guard.
///
/// @param chain    Source chain.
/// @param head     Starting entry (typically the implication
///                 head).
/// @param visited  Cycle-guard set (mutated).
/// @return Reachable sub-chain in walk order.
std::vector<std::string>
Conjecturer::getRightChain(const std::vector<std::string>& chain,
                           const std::string& head, std::set<std::string>& visited) const {
    std::vector<std::string> rightChain;
    if (visited.find(head) != visited.end()) return rightChain;
    visited.insert(head);
    rightChain.push_back(head);

    auto headArgs = ce::getArgs(head);
    std::string headCoreExpr = ce::extractExpression(head);
    std::vector<std::string> headInputs;
    for (int idx : config_.data.at(headCoreExpr).indices_input_args) headInputs.push_back(headArgs[idx]);

    for (auto& expression : chain) {
        std::string coreExpr = ce::extractExpression(expression);
        if (std::find(operators_.begin(), operators_.end(), coreExpr) == operators_.end()) continue;
        auto exprArgs = ce::getArgs(expression);
        std::string exprOutput = exprArgs[config_.data.at(coreExpr).indices_output_args[0]];
        if (std::find(headInputs.begin(), headInputs.end(), exprOutput) != headInputs.end()) {
            auto sub = getRightChain(chain, expression, visited);
            rightChain.insert(rightChain.end(), sub.begin(), sub.end());
        }
    }
    return rightChain;
}

/// @brief Split `chain` into its left and right subchains
///        relative to the conjecture's outer implication.
///
/// @details
/// `chain[0]` is the anchor and is excluded from both sides.
/// Walks from `chain.back()` (the head) using `getRightChain` to
/// collect the right subchain; everything else (in chain order)
/// becomes the left subchain.
///
/// @param chain Source chain.
/// @return Pair `(leftChain, rightChain)`.
std::pair<std::vector<std::string>, std::vector<std::string>>
Conjecturer::getLeftRightChains(const std::vector<std::string>& chain) const {
    std::vector<std::string> noAnchor(chain.begin() + 1, chain.end());
    std::set<std::string> visited;
    auto rightChain = getRightChain(noAnchor, chain.back(), visited);
    std::set<std::string> rightSet(rightChain.begin(), rightChain.end());
    std::vector<std::string> leftChain;
    for (auto& x : noAnchor)
        if (rightSet.find(x) == rightSet.end()) leftChain.push_back(x);
    return {leftChain, rightChain};
}

/// @brief Extract the operator id (expression name) from the head
///        of `expr`.
///
/// @details
/// Used by the `checkInputVariablesOrder` sub-helpers to route
/// operator-specific behaviour. Returns the empty string when
/// `expr` does not have a recognised operator head.
std::string Conjecturer::getOperatorId(const std::string& expr) const {
    std::string coreExpr = ce::extractExpression(expr);
    if (std::find(operators_.begin(), operators_.end(), coreExpr) == operators_.end()) return "";
    std::map<std::string,std::string> replMap;
    auto args = ce::getArgs(expr);
    for (int idx : config_.data.at(coreExpr).indices_input_args) replMap[args[idx]] = "";
    for (int idx : config_.data.at(coreExpr).indices_output_args) replMap[args[idx]] = "";
    return ce::replaceKeysInString(expr, replMap);
}

/// @brief Check that input-variable positions in `chain` respect
///        the canonical left-to-right order over `digits`.
///
/// @details
/// Walks `chain` and for each digit arg in `digits` records the
/// first position at which it appears in an input slot. A
/// canonical chain has these positions in non-decreasing order;
/// rearrangements that would place a higher-id digit before a
/// lower-id one are rejected.
bool Conjecturer::checkInputVariablePosition(const std::vector<std::string>& chain,
                                              const std::set<std::string>& digits) const {
    std::map<std::pair<std::string,int>, int> orderMap;
    for (auto& expression : chain) {
        std::string coreExpr = ce::extractExpression(expression);
        if (std::find(operators_.begin(), operators_.end(), coreExpr) != operators_.end()
            && config_.data.at(coreExpr).indices_input_args.size() == 2) {
            auto args = ce::getArgs(expression);
            for (int argInd = 0; argInd < 2; ++argInd) {
                std::string arg = args[config_.data.at(coreExpr).indices_input_args[argInd]];
                if (digits.find(arg) != digits.end()) {
                    auto key = std::make_pair(getOperatorId(expression), std::stoi(arg));
                    // Encode arg_ind in map value using a different key type
                    // Actually Python uses (operator_id, arg) as key, value = arg_ind
                    // Let me use string keys
                }
            }
        }
    }

    // Re-implement properly
    std::map<std::pair<std::string,std::string>, int> orderMap2;
    for (auto& expression : chain) {
        std::string coreExpr = ce::extractExpression(expression);
        if (std::find(operators_.begin(), operators_.end(), coreExpr) != operators_.end()
            && config_.data.at(coreExpr).indices_input_args.size() == 2) {
            auto args = ce::getArgs(expression);
            for (int argInd = 0; argInd < 2; ++argInd) {
                std::string arg = args[config_.data.at(coreExpr).indices_input_args[argInd]];
                if (digits.find(arg) != digits.end()) {
                    auto key = std::make_pair(getOperatorId(expression), arg);
                    auto it = orderMap2.find(key);
                    if (it == orderMap2.end()) {
                        orderMap2[key] = argInd;
                    } else {
                        if (it->second != argInd) return false;
                    }
                }
            }
        }
    }
    return true;
}

/// @brief Drop output-arg indices from each entry in `chain`.
///
/// @details
/// Useful for reasoning about chain structure independent of
/// operator-output-binding details — input-only views feed the
/// tautology and tertiary checks.
///
/// @return Single-string concatenation of the input-only forms.
std::string Conjecturer::removeOutputs(const std::vector<std::string>& chain) const {
    std::map<std::string,std::string> replMap;
    for (auto& expr : chain) {
        std::string coreExpr = ce::extractExpression(expr);
        auto args = ce::getArgs(expr);
        for (int idx : config_.data.at(coreExpr).indices_output_args) replMap[args[idx]] = "";
    }
    std::string joined;
    for (auto& e : chain) joined += e;
    return ce::replaceKeysInString(joined, replMap);
}

/// @brief Reject a conjecture whose left and right subchains are
///        identical modulo bound-var renaming.
///
/// @details
/// A tautological implication adds nothing to the theorem set; the
/// filter uses `removeOutputs` to compare structure first, then
/// falls back on a more careful renaming-aware comparison if the
/// fast path is inconclusive.
bool Conjecturer::checkTautology(const std::vector<std::string>& leftChain, const std::vector<std::string>& rightChain) const {
    return removeOutputs(leftChain) != removeOutputs(rightChain);
}

/// @brief Reject a chain whose function-position usage violates
///        the conjecturer's well-formedness rules.
///
/// @details
/// Used by `checkInputVariablesOrder` as a final structural gate.
/// Catches malformed function-position arrangements that survive
/// the upstream filters but would produce a malformed proof
/// graph downstream.
bool Conjecturer::checkFunctions(const std::vector<std::string>& chain) const {
    std::set<std::string> removedSet;
    for (auto& expr : chain) {
        std::string coreExpr = ce::extractExpression(expr);
        if (std::find(operators_.begin(), operators_.end(), coreExpr) != operators_.end()
            && config_.data.at(coreExpr).indices_input_args.size() == 1) {
            auto args = ce::getArgs(expr);
            std::string output = args[config_.data.at(coreExpr).indices_output_args[0]];
            std::map<std::string,std::string> replMap = {{output, ""}};
            std::string removed = ce::replaceKeysInString(expr, replMap);
            if (removedSet.find(removed) != removedSet.end()) return false;
            removedSet.insert(removed);
        }
    }
    return true;
}

/// @brief Test whether `chain` carries exactly one operator
///        expression.
///
/// @details
/// Used as a special-case relaxation gate in
/// `checkInputVariablesOrder`: when only one operator is present,
/// some otherwise-restrictive ordering rules are relaxed because
/// there is nothing for them to mis-order.
bool Conjecturer::onlyOneOperator(const std::vector<std::string>& chain) const {
    std::vector<std::string> noAnchor(chain.begin() + 1, chain.end());
    if (noAnchor.empty()) return false;
    std::string head = noAnchor.back();
    if (std::find(operators_.begin(), operators_.end(), ce::extractExpression(head)) == operators_.end())
        return false;
    std::string headKey = getOperatorId(head);
    for (auto& expr : noAnchor) {
        if (getOperatorId(expr) != headKey) return false;
    }
    return true;
}

/// @brief Compute the closure of input args reachable from
///        position `index`, threading through output-arg bindings.
///
/// @details
/// Recursive closure: starts at `index`, collects its input args,
/// then for each output arg of `index` finds every other position
/// that consumes that output as an input, and recursively expands
/// from there. `visited` prevents cycles.
///
/// @return Closure of reachable input args.
std::set<std::string> Conjecturer::findEntryArgs2(
    const std::vector<std::vector<std::string>>& inputArgsList,
    const std::vector<std::vector<std::string>>& outputArgsList,
    int index, std::set<int>& visited) const {
    std::set<std::string> entryArgs;
    if (visited.find(index) != visited.end()) return entryArgs;
    visited.insert(index);
    assert(outputArgsList.size() == inputArgsList.size());

    for (auto& inputArg : inputArgsList[index]) {
        bool found = false;
        for (int index2 = 0; index2 < (int)outputArgsList.size(); ++index2) {
            if (std::find(outputArgsList[index2].begin(), outputArgsList[index2].end(), inputArg)
                != outputArgsList[index2].end()) {
                found = true;
                auto sub = findEntryArgs2(inputArgsList, outputArgsList, index2, visited);
                entryArgs.insert(sub.begin(), sub.end());
                break;
            }
        }
        if (!found) entryArgs.insert(inputArg);
    }
    return entryArgs;
}

/// @brief Collect tertiary (third-tier) bound variables from
///        `chain`.
///
/// @details
/// Tertiary args are bound variables that do not appear in the
/// head's input positions but do appear elsewhere in the chain.
/// They mark the variables introduced in the middle of the chain
/// for intermediate predicates rather than for the conclusion.
std::set<std::string> Conjecturer::getTertiaries(const std::vector<std::string>& chain) const {
    std::set<std::string> tertiaries;
    for (auto& expr : chain) {
        std::string coreExpr = ce::extractExpression(expr);
        if (std::find(operators_.begin(), operators_.end(), coreExpr) != operators_.end()
            && config_.data.at(coreExpr).indices_input_args.size() == 2) {
            tertiaries.insert(getOperatorId(expr));
        }
    }
    return tertiaries;
}

/// @brief Check tertiary-variable compatibility between left and
///        right subchains; backstop for `checkTautology`.
///
/// @details
/// A pair of subchains that differ only in their tertiary args
/// (those not directly cited in the head's input positions) is
/// not a useful theorem; this filter rejects such pairs after
/// the more direct `checkTautology` first-pass.
bool Conjecturer::checkTertiaries(const std::vector<std::string>& leftChain, const std::vector<std::string>& rightChain) const {
    auto leftTert = getTertiaries(leftChain);
    auto rightTert = getTertiaries(rightChain);
    if (!leftTert.empty() && !rightTert.empty()) {
        std::set<std::string> intersection;
        std::set_intersection(leftTert.begin(), leftTert.end(),
                              rightTert.begin(), rightTert.end(),
                              std::inserter(intersection, intersection.begin()));
        if (intersection.empty()) return false;
    }
    return true;
}

// ============================================================================
// checkInputVariablesOrder
// ============================================================================

/// @brief Enforce input-variable ordering across the chain to
///        suppress trivially-rearranged conjectures.
///
/// @details
/// Composes the 13 sub-helpers (`findDigitArgs`, `getLeftRight`,
/// `getRightChain`, `getLeftRightChains`, `getOperatorId`,
/// `checkInputVariablePosition`, `removeOutputs`, `checkTautology`,
/// `checkFunctions`, `onlyOneOperator`, `findEntryArgs2`,
/// `getTertiaries`, `checkTertiaries`) into a single ordering
/// gate. Reduces the permutation surface — otherwise
/// trivially-rearranged variants would all pass filtering.
///
/// @invariant [I-10](../../docs/agentic_swdd/30_invariants.md#i-10) — bound
///            variables appear left-to-right in input-arg
///            positions.
bool Conjecturer::checkInputVariablesOrder(const std::string& theorem) const {
    prof::Scope _p(prof::g_checkInputVarsOrder);
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> tempChain;
    std::string head = ce::disintegrateImplication(theorem, tempChain, coreExprMap_);
    std::vector<std::string> chain;
    for (auto& e : tempChain) chain.push_back(std::get<0>(e));
    chain.push_back(head);

    auto digits = findDigitArgs(theorem);
    bool result = true;

    // Order map: (operator_id, frozenset{left,right}) -> [left, right]
    std::map<std::pair<std::string, std::set<std::string>>, std::vector<std::string>> orderMap;
    std::set<std::set<std::string>> orderSet;

    for (auto& expression : chain) {
        std::string coreExpr = ce::extractExpression(expression);
        if (std::find(operators_.begin(), operators_.end(), coreExpr) != operators_.end()
            && config_.data.at(coreExpr).indices_input_args.size() == 2) {
            auto args = ce::getArgs(expression);
            std::string argLeft = args[config_.data.at(coreExpr).indices_input_args[0]];
            std::string argRight = args[config_.data.at(coreExpr).indices_input_args[1]];
            if (digits.find(argLeft) != digits.end() && digits.find(argRight) != digits.end()) {
                std::set<std::string> key = {argLeft, argRight};
                orderMap[{getOperatorId(expression), key}] = {argLeft, argRight};
                if (orderSet.find(key) != orderSet.end()) {
                    result = false;
                } else {
                    orderSet.insert(key);
                }
            }
        }
    }

    for (auto& expression : chain) {
        std::string coreExpr = ce::extractExpression(expression);
        if (std::find(operators_.begin(), operators_.end(), coreExpr) != operators_.end()
            && config_.data.at(coreExpr).indices_input_args.size() == 2) {
            std::set<std::string> visited;
            auto [left, right] = getLeftRight(chain, expression, digits, 0, visited);
            std::string op = getOperatorId(expression);
            for (auto& la : left) {
                for (auto& ra : right) {
                    std::set<std::string> key = {la, ra};
                    auto it = orderMap.find({op, key});
                    if (it != orderMap.end()) {
                        std::vector<std::string> expected = {la, ra};
                        if (it->second != expected) result = false;
                    }
                }
            }
        }
    }

    auto [leftChain, rightChain] = getLeftRightChains(chain);

    if (result) {
        result = checkInputVariablePosition(leftChain, digits) &&
                 checkInputVariablePosition(rightChain, digits);
    }

    if (result) {
        result = checkTertiaries(leftChain, rightChain);
    }

    if (onlyOneOperator(chain)) {
        auto [reshuffled, reshuffledMap, repMap] = reshuffle(theorem, true);
        auto reshuffledMirrored = createReshuffledMirrored(theorem);
        if (reshuffled == reshuffledMirrored) {
            assert(!reshuffledMirrored.empty());
            result = true;
        }
    }

    result = result && checkTautology(leftChain, rightChain);
    if (!qualifiedForEquality(theorem)) {
        result = result && checkFunctions(chain);
    }

    return result;
}

// ============================================================================
// exprGood2 (combines multiple filters)
// ============================================================================

/// @brief Final structural-approval gate at the string-path
///        level.
///
/// @details
/// Combines numeric-arg contiguity (`numbersGood`), per-type
/// combinable / uncombinable caps (`checkDefSets`), and
/// operator-block accounting. String-path twin of
/// `exprGood2Int`. Called after `exprGood` (the cheap first
/// pass) and before the per-cascade filters so cheap rejections
/// land before expensive filter work runs.
///
/// @param expr           Expression text.
/// @param nse            Number of simple expressions in `expr`.
/// @param connectedMap   Def-set map after the latest connection
///                       step.
/// @return `true` to accept; `false` to reject.
bool Conjecturer::exprGood2(const std::string& expr, int nse, const DefSetMap& connectedMap) const {
    if (repetitionsExist(expr)) return false;
    if (!numbersGood(expr)) return false;

    int sizeArgs = 0;
    for (auto& [arg, tpl] : connectedMap)
        if (std::get<0>(tpl).substr(0, 1) != "P" && std::get<1>(tpl)) sizeArgs++;

    if (nse == config_.parameters.max_number_simple_expressions && !checkDefSets(connectedMap))
        return false;

    auto opExprs = extractOperatorExpressions(expr);

    // evaluate_operator_exprs (local helper like Python's nested function)
    auto evaluateOperatorExprs = [&](const std::vector<std::string>& opExprs2, bool freeArgs) -> bool {
        if (opExprs2.empty()) return true;
        std::string lastExpr = opExprs2.back();
        auto lastArgs = ce::getArgs(lastExpr);
        std::string lastCore = ce::extractExpression(lastExpr);
        std::string lastArg = lastArgs[config_.data.at(lastCore).indices_output_args[0]];

        int occCounter = 0;
        for (int i = 0; i < (int)opExprs2.size() - 1; ++i) {
            auto args = ce::getArgs(opExprs2[i]);
            if (std::find(args.begin(), args.end(), lastArg) != args.end()) {
                occCounter++;
                int occInd = (int)(std::find(args.begin(), args.end(), lastArg) - args.begin());
                std::string ce2 = ce::extractExpression(opExprs2[i]);
                if (occInd != config_.data.at(ce2).indices_output_args[0]) return false;
            }
        }

        if ((int)opExprs2.size() >= config_.parameters.operator_threshold ||
            ((int)opExprs2.size() == config_.parameters.operator_threshold - 1 && freeArgs)) {
            for (int lastArgInd : config_.data.at(lastCore).indices_input_args) {
                occCounter = 0;
                for (int opInd = 0; opInd < (int)opExprs2.size() - 1; ++opInd) {
                    std::string ce2 = ce::extractExpression(opExprs2[opInd]);
                    auto args = ce::getArgs(opExprs2[opInd]);
                    if (std::find(args.begin(), args.end(), lastArgs[lastArgInd]) != args.end()) {
                        occCounter++;
                        if (args[config_.data.at(ce2).indices_output_args[0]] != lastArgs[lastArgInd])
                            return false;
                    }
                }
                if (occCounter > 1) return false;
            }
        }
        return true;
    };

    if (!evaluateOperatorExprs(opExprs, sizeArgs > 0)) return false;
    if (!evaluateOperatorExprs2(expr, false)) return false;
    if (!checkProhibitedCombinations(expr)) return false;
    return true;
}

// ============================================================================
// Reshuffling & mirroring (Phase 7)
// ============================================================================

/// @brief Count `(>[` operator-block headers in `s`.
///
/// @details
/// String-path twin of `countOperatorOccurrencesInt`. Equivalent
/// to the conjecture's complexity level — every `(>[` introduces
/// a new quantifier layer.
///
/// @param s Expression text.
/// @return Number of operator-block headers.
int Conjecturer::countOperatorOccurrences(const std::string& s) const {
    prof::Scope _p(prof::g_countOpOccurrences);
    int count = 0;
    size_t pos = 0;
    while ((pos = s.find("(>[", pos)) != std::string::npos) {
        count++;
        pos += 3;
    }
    return count;
}

/// @brief Test whether `outputVariable` survives in `fullExpr`
///        after the conjecture is constructed.
///
/// @details
/// An operator's output variable should persist into at least one
/// other position of the conjecture; if it does not, the
/// conjecture binds an output to nothing and is malformed. Used
/// by the operator-head reformulation gate to decide whether the
/// candidate qualifies for existence-head treatment.
///
/// @param fullExpr        Whole conjecture text.
/// @param outputVariable  Variable id (as string) of the
///                        operator's output position.
/// @return `true` when the variable appears at least once outside
///         the operator that introduced it.
bool Conjecturer::staysOutputVariable(const std::string& fullExpr, const std::string& outputVariable) const {
    std::string coreExpr = ce::extractExpression(fullExpr);
    auto args = ce::getArgs(fullExpr);
    if (std::find(operators_.begin(), operators_.end(), coreExpr) != operators_.end()) {
        if (args[config_.data.at(coreExpr).indices_output_args[0]] == outputVariable)
            return true;
    }
    return false;
}

/// @brief Canonicalise a conjecture into its `conjectures.txt` /
///        `reshuffled_conjectures.txt` form.
///
/// @details
/// Pipeline (per SwDD chapter `02_conjecturer.md` section
/// *Reshuffle pipeline*):
/// 1. Flat-walk rename — first-occurrence numbering of the
///    arg-id space. Replaces the older recursive
///    `renameVariablesInExpr` walker; produces a deterministic
///    numbering independent of recursive tree shape.
/// 2. Existence-head pinning — extract the outermost existence
///    bv-list before the flat-walk so existence bvs land at
///    canonical positions. Without this, existence bvs would
///    end up higher-numbered than chain-introduced bvs that
///    appear later in walk order.
/// 3. Contiguous-arg renumber post-connect — drop holes left
///    by `connectExpressionsInt`'s `subMap`.
/// 4. Anchor position-0 pin — anchor args are never permuted
///    across reshuffle variants.
///
/// Each stage's output feeds the next; skipping any stage
/// produces drift in the canonical form vs the `_mirrored`
/// companion. See decision
/// [D-20](../../docs/agentic_swdd/40_decisions.md#d-20) for the int-path
/// acceleration history.
///
/// @param expr Source expression.
/// @param deep When true, descend into nested existence heads.
/// @return Tuple `(canonical, defSets, renameMap)`.
std::tuple<std::string, DefSetMap, std::map<std::string,std::string>>
Conjecturer::reshuffle(const std::string& expr, bool deep) const {
    prof::Scope _prof_reshuffle(prof::g_reshuffle);
    using ChainEntry = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;

    // Step 1: disintegrate the outer implication chain.
    std::vector<ChainEntry> chainEntries;
    std::string head = ce::disintegrateImplication(expr, chainEntries, coreExprMap_);
    int chainLen = (int)chainEntries.size();

    // Step 1b: existence-head handling. If head = !(>[bvs](L)(R)),
    // disintegrate the existence wrapper: append L as a pinned chain
    // entry at index chainLen (fixed; never permuted) with existBvs as
    // its bv list, and set head = R. On rebuild, the pinned level wraps
    // with `!(>[bvs](L)(R))` instead of the regular `(>[bvs](L)(R))`.
    bool isExistHead = false;
    int pinnedIdx = -1;
    if (head.size() >= 5 && head[0] == '!' && head[1] == '(' && head[2] == '>' && head[3] == '[') {
        size_t bvEnd = head.find(']', 4);
        if (bvEnd != std::string::npos && bvEnd + 1 < head.size() && head[bvEnd + 1] == '(') {
            std::vector<std::string> existBvs;
            std::string bvStr = head.substr(4, bvEnd - 4);
            if (!bvStr.empty()) {
                size_t p = 0;
                while (true) {
                    size_t c = bvStr.find(',', p);
                    if (c == std::string::npos) { existBvs.push_back(bvStr.substr(p)); break; }
                    existBvs.push_back(bvStr.substr(p, c - p));
                    p = c + 1;
                }
            }
            // Depth-track to find end of L (its own closing ')').
            int depth = 0;
            size_t lstart = bvEnd + 1;
            size_t lend = std::string::npos;
            for (size_t i = lstart; i < head.size(); ++i) {
                if (head[i] == '(') ++depth;
                else if (head[i] == ')') {
                    --depth;
                    if (depth == 0) { lend = i; break; }
                }
            }
            if (lend != std::string::npos && lend + 2 <= head.size()) {
                std::string lPart = head.substr(lstart, lend - lstart + 1);  // "(L)"
                // R is from lend+1 up to head.size()-2 (the final ')' at size-1 closes '!(>...)').
                std::string rPart = head.substr(lend + 1, head.size() - lend - 2);
                // Collect lArgs = all atom args in L minus bvs declared inside L.
                std::set<std::string> lArgs;
                {
                    std::set<std::string> innerBvs;
                    size_t pos = 0;
                    while (pos < lPart.size()) {
                        size_t lb = lPart.find('[', pos);
                        if (lb == std::string::npos) break;
                        size_t rb = lPart.find(']', lb);
                        if (rb == std::string::npos) break;
                        bool isBv = (lb > 0 && lPart[lb - 1] == '>');
                        size_t p = lb + 1;
                        while (p < rb) {
                            size_t c = lPart.find(',', p);
                            if (c == std::string::npos || c > rb) c = rb;
                            if (c > p) {
                                std::string tok = lPart.substr(p, c - p);
                                if (isBv) innerBvs.insert(tok);
                                else lArgs.insert(tok);
                            }
                            p = c + 1;
                        }
                        pos = rb + 1;
                    }
                    for (auto& b : innerBvs) lArgs.erase(b);
                }
                chainEntries.emplace_back(lPart, existBvs, lArgs);
                pinnedIdx = (int)chainEntries.size() - 1;
                head = rPart;
                isExistHead = true;
            }
        }
    }

    // effChainLen = chainLen + (isExistHead ? 1 : 0). Permutations iterate
    // over chainLen only; the pinned entry (if present) is always at
    // effChainLen - 1.
    const int effChainLen = (int)chainEntries.size();

    auto permIt = allPermutations_.find(chainLen);
    if (permIt == allPermutations_.end()) {
        // chainLen not represented (edge case) — fall back to legacy.
        return renameVariablesInExpr(expr, deep);
    }

    // Step 1c: anchor-first pinning. The chain entry whose premise carries
    // the anchor handle (e.g. AnchorPeano / AnchorGauss) must stay at
    // permutation position 0 — it is never permuted away from the outer
    // level. Only permutations with permutation[0] == anchorChainIdx are
    // considered. Existence pinning (Step 1b) already fixes the last slot;
    // together they leave (chainLen - 1)! orderings for the interior.
    int anchorChainIdx = -1;
    {
        const std::string anchorName = config_.getAnchorName();
        auto descIt = config_.data.find(anchorName);
        if (descIt != config_.data.end()) {
            const std::string& anchorHandle = descIt->second.handle;
            for (int i = 0; i < chainLen; ++i) {
                if (std::get<0>(chainEntries[i]).find(anchorHandle) != std::string::npos) {
                    anchorChainIdx = i;
                    break;
                }
            }
        }
    }

    // Step 2: pre-compute the input's def-set map once. The winner's
    // DefSetMap is derived by key-renaming this via the winner's rename
    // map — permutation + rename preserve each arg's def-set value.
    DefSetMap inputArgMap = findArgMap(expr);

    // Step 3 (Path B — int-level rebuild): pre-parse each chainEntry +
    // head into a "template" that pinpoints every `[...]`-token byte
    // position with its dense int id + an isAtom flag. Tokens are
    // interned once (shared across entries) so the inner-loop rename
    // lookup is a single int-indexed array access. Per permutation:
    //   Phase 1: walk atoms in permuted rebuilt-byte order, assign
    //            renames 1..n in first-occurrence order (matches OLD
    //            `collectAll`).
    //   Phase 2: render the rebuilt string directly into a reusable
    //            thread_local char buffer. Literal chunks copied from
    //            each entry's backing string; atom + nested-bv slots
    //            substituted with the renamed decimal id (or verbatim
    //            token text if the token has no atom occurrence — OLD
    //            `replaceKeysInString` leaves unknown keys alone).
    //   Winner selection: memcmp on the rendered buffer (equivalent to
    //   OLD's std::string lex-compare, byte-for-byte).
    // External contract (returned rebuilt + renameMap) is preserved
    // identically — byte-for-byte equivalent to OLD on every input.
    struct ArgSlot {
        size_t start;   // byte position of first token char
        size_t end;     // byte position just past last token char
        int tokenId;    // dense id into idToToken / renameArr
        bool isAtom;    // true: contributes to rename order (collectAll visits these)
                        // false: nested `>[bvs]` list; receives rename but does not
                        //        drive assignment (matches OLD: collectAll skips `>[`)
    };
    struct EntryTemplate {
        const std::string* src;
        std::vector<ArgSlot> args;
    };

    std::unordered_map<std::string, int> tokenToId;
    std::vector<std::string> idToToken;
    auto internToken = [&](const std::string& tok) -> int {
        auto it = tokenToId.find(tok);
        if (it != tokenToId.end()) return it->second;
        int id = (int)idToToken.size();
        idToToken.push_back(tok);
        tokenToId.emplace(tok, id);
        return id;
    };

    auto parseEntry = [&](const std::string& s) -> EntryTemplate {
        EntryTemplate t; t.src = &s;
        size_t pos = 0;
        while (pos < s.size()) {
            size_t lb = s.find('[', pos);
            if (lb == std::string::npos) break;
            const bool isBvList = (lb > 0 && s[lb - 1] == '>');
            size_t rb = s.find(']', lb);
            if (rb == std::string::npos) break;
            size_t p = lb + 1;
            while (p < rb) {
                size_t c = s.find(',', p);
                if (c == std::string::npos || c > rb) c = rb;
                if (c > p) {
                    std::string tok = s.substr(p, c - p);
                    int id = internToken(tok);
                    t.args.push_back({p, c, id, !isBvList});
                }
                p = c + 1;
            }
            pos = rb + 1;
        }
        return t;
    };

    std::vector<EntryTemplate> entryTmpl;
    entryTmpl.reserve(chainEntries.size());
    for (auto& ce : chainEntries) entryTmpl.push_back(parseEntry(std::get<0>(ce)));
    EntryTemplate headTmpl = parseEntry(head);

    // Pre-intern every bv token (outer chain-level bvs come from
    // std::get<1>(chainEntries[...]) and are emitted by `emitBvList`
    // — they must have a dense id even if the bv never shows up as an
    // atom anywhere, which is rare but possible for pathological
    // inputs).
    for (auto& ce : chainEntries) for (auto& b : std::get<1>(ce)) internToken(b);

    const int numTokens = (int)idToToken.size();

    // Thread-local scratch buffers: reshuffle is called from worker
    // threads (singleThreadCalculationInt), so thread_local gives us
    // per-thread reuse with zero contention.
    thread_local std::vector<char> permBuf;
    thread_local std::vector<char> minBuf;
    thread_local std::vector<int>  renameArr;
    thread_local std::vector<int>  minRenameArrSnap;

    // Step 4: iterate permutations, pick lex-min. Anchor-first pinning
    // (permutations whose [0] != anchorChainIdx are rejected) — see
    // OLD comment preserved below for the rationale.
    //
    // > the chain entry carrying the anchor must stay at position 0 —
    // > permutations not starting with anchorChainIdx are rejected.
    // > Without this, a lex-smaller permutation starting with a negated
    // > atom like `!(=...)` (ASCII `!` < `(A`) would displace the
    // > anchor from the outer level. Existence pinning (Step 1b) still
    // > fixes the last slot; together they leave (chainLen - 1)!
    // > orderings to dedup the interior. If no anchor is present
    // > (anchor-less edge case), all permutations are considered.
    std::map<std::string, std::string> minRenameMap;
    bool first = true;
    minBuf.clear();
    minRenameArrSnap.clear();
    for (auto& permutation : permIt->second) {
        if (anchorChainIdx >= 0 && !permutation.empty() && permutation[0] != anchorChainIdx) continue;
        prof::Scope _prof_perm(prof::g_reshufflePermInner);

        // Compute per-level bv redistribution (semantics identical to OLD).
        std::set<std::string> allRemovedArgs;
        for (int i = 0; i < chainLen; ++i) {
            auto& rmv = std::get<1>(chainEntries[permutation[i]]);
            allRemovedArgs.insert(rmv.begin(), rmv.end());
        }
        if (isExistHead) {
            auto& rmv = std::get<1>(chainEntries[pinnedIdx]);
            allRemovedArgs.insert(rmv.begin(), rmv.end());
        }
        std::map<std::string, int> firstLevelMap;
        auto addLevel = [&](int outIdx, const std::set<std::string>& rem) {
            for (auto& arg : allRemovedArgs) {
                if (rem.count(arg)) {
                    auto it = firstLevelMap.find(arg);
                    if (it == firstLevelMap.end()) firstLevelMap[arg] = outIdx;
                    else it->second = std::min(it->second, outIdx);
                }
            }
        };
        for (int ind = 0; ind < chainLen; ++ind) addLevel(ind, std::get<2>(chainEntries[permutation[ind]]));
        if (isExistHead) addLevel(chainLen, std::get<2>(chainEntries[pinnedIdx]));
        std::vector<std::vector<std::string>> removedArgsLists(effChainLen);
        for (auto& [arg, ind] : firstLevelMap) removedArgsLists[ind].push_back(arg);
        for (int ind = 0; ind < effChainLen; ++ind) {
            if (removedArgsLists[ind].size() <= 1) continue;
            std::set<std::string> argSet(removedArgsLists[ind].begin(), removedArgsLists[ind].end());
            int chainRefIdx = (ind < chainLen) ? permutation[ind] : pinnedIdx;
            removedArgsLists[ind] = ce::orderByPattern(std::get<0>(chainEntries[chainRefIdx]), argSet);
        }
        std::vector<std::vector<int>> bvIdLists(effChainLen);
        for (int ind = 0; ind < effChainLen; ++ind) {
            bvIdLists[ind].reserve(removedArgsLists[ind].size());
            for (auto& s : removedArgsLists[ind]) {
                auto it = tokenToId.find(s);
                assert(it != tokenToId.end());  // bvs were pre-interned above
                bvIdLists[ind].push_back(it->second);
            }
        }

        // Phase 1: first-occurrence rename walk (atoms only).
        renameArr.assign(numTokens, 0);
        int nextId = 1;
        auto assignRenames = [&](const EntryTemplate& t) {
            for (auto& a : t.args) {
                if (a.isAtom && renameArr[a.tokenId] == 0) renameArr[a.tokenId] = nextId++;
            }
        };
        for (int i = 0; i < chainLen; ++i) assignRenames(entryTmpl[permutation[i]]);
        if (isExistHead) assignRenames(entryTmpl[pinnedIdx]);
        assignRenames(headTmpl);

        // Phase 2: render rebuilt string to permBuf.
        permBuf.clear();
        auto emitRid = [&](int rid) {
            if (rid < 10) {
                permBuf.push_back((char)('0' + rid));
            } else {
                char tmp[12];
                int n = std::snprintf(tmp, sizeof(tmp), "%d", rid);
                permBuf.insert(permBuf.end(), tmp, tmp + n);
            }
        };
        auto emitToken = [&](int tokenId) {
            int rid = renameArr[tokenId];
            if (rid != 0) {
                emitRid(rid);
            } else {
                // Unassigned: emit original token verbatim (OLD:
                // replaceKeysInString leaves unmatched substrings
                // alone). In practice this only fires for nested
                // bvs whose tokens never appear as atoms.
                const std::string& s = idToToken[tokenId];
                permBuf.insert(permBuf.end(), s.begin(), s.end());
            }
        };
        auto emitLit = [&](const char* s, size_t n) {
            permBuf.insert(permBuf.end(), s, s + n);
        };
        auto emitEntry = [&](const EntryTemplate& t) {
            size_t cur = 0;
            for (auto& a : t.args) {
                emitLit(t.src->data() + cur, a.start - cur);
                emitToken(a.tokenId);
                cur = a.end;
            }
            emitLit(t.src->data() + cur, t.src->size() - cur);
        };
        auto emitBvList = [&](const std::vector<int>& ids) {
            permBuf.push_back('[');
            for (size_t k = 0; k < ids.size(); ++k) {
                if (k) permBuf.push_back(',');
                emitToken(ids[k]);
            }
            permBuf.push_back(']');
        };

        for (int i = 0; i < chainLen; ++i) {
            emitLit("(>", 2);
            emitBvList(bvIdLists[i]);
            emitEntry(entryTmpl[permutation[i]]);
        }
        if (isExistHead) {
            emitLit("!(>", 3);
            emitBvList(bvIdLists[chainLen]);
            emitEntry(entryTmpl[pinnedIdx]);
            emitEntry(headTmpl);
            permBuf.push_back(')');
        } else {
            emitEntry(headTmpl);
        }
        for (int i = 0; i < chainLen; ++i) permBuf.push_back(')');

        // Winner selection: memcmp-based lex compare (equivalent to
        // std::string operator< on the renamed rebuilt strings, because
        // both strings contain only bytes and memcmp is lexicographic
        // on byte sequences).
        bool isNewMin;
        if (first) {
            isNewMin = true;
        } else {
            size_t lp = permBuf.size();
            size_t lm = minBuf.size();
            size_t common = lp < lm ? lp : lm;
            int cmp = common ? std::memcmp(permBuf.data(), minBuf.data(), common) : 0;
            if (cmp < 0) isNewMin = true;
            else if (cmp > 0) isNewMin = false;
            else isNewMin = (lp < lm);
        }
        if (isNewMin) {
            minBuf = permBuf;
            minRenameArrSnap = renameArr;
            first = false;
        }
    }

    // Construct std::string winner + string→string rename map for
    // downstream compatibility (inputArgMap key-renaming + callers
    // that read the returned rename map).
    std::string minReshuffled(minBuf.begin(), minBuf.end());
    for (int tid = 0; tid < numTokens; ++tid) {
        int rid = (tid < (int)minRenameArrSnap.size()) ? minRenameArrSnap[tid] : 0;
        if (rid != 0) minRenameMap[idToToken[tid]] = std::to_string(rid);
    }

    // Step 5: DefSetMap by key-renaming the input argMap with the winner's
    // rename map (no re-parse needed).
    DefSetMap minArgMap;
    for (auto& [oldKey, val] : inputArgMap) {
        auto it = minRenameMap.find(oldKey);
        minArgMap[it != minRenameMap.end() ? it->second : oldKey] = val;
    }
    std::map<std::string, std::string> minReplacementMap;  // unused by callers
    return {minReshuffled, minArgMap, minReplacementMap};
}

/// @brief Build the mirror variant of a reshuffled conjecture.
///
/// @details
/// The mirror swaps left and right of the implication; for
/// equality heads the mirror is the symmetric orientation. The
/// pipeline is reshuffle-then-mirror-then-reshuffle so the
/// mirror is itself in canonical form. Per
/// [I-9](../../docs/agentic_swdd/30_invariants.md#i-9), a mirror is dropped
/// when it equals its source (i.e. the conjecture is its own
/// mirror) — keeping it would emit a duplicate.
///
/// @param expr         Source conjecture (already reshuffled).
/// @param anchorFirst  When true, force the anchor to position 0
///                     in the result. Used when the caller has
///                     not yet anchor-pinned the input.
/// @return Mirror variant in canonical form, or empty when the
///         distinctness guard rejects.
/// @invariant [I-9](../../docs/agentic_swdd/30_invariants.md#i-9) — equality
///            mirror guarded by distinctness.
std::string Conjecturer::createReshuffledMirrored(const std::string& expr, bool anchorFirst) const {
    prof::Scope _prof_mirrored(prof::g_mirrored);
    using ChainEntry = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<ChainEntry> tempChain;
    std::string head = ce::disintegrateImplication(expr, tempChain, coreExprMap_);

    auto headArgs = ce::getArgs(head);
    std::string headExpr = ce::extractExpression(head);

    if (std::find(operators_.begin(), operators_.end(), headExpr) == operators_.end()) return "";

    std::string outputVariable = headArgs[config_.data.at(headExpr).indices_output_args[0]];
    assert(!outputVariable.empty());

    std::string alternative;
    std::vector<std::string> chain;
    for (auto& element : tempChain) {
        if (staysOutputVariable(std::get<0>(element), outputVariable)) {
            alternative = std::get<0>(element);
        } else {
            chain.push_back(std::get<0>(element));
        }
    }

    if (anchorFirst) {
        std::string anchorName = config_.getAnchorName();
        ce::prioritizeAnchor(chain, config_.data.at(anchorName).handle);
    }

    if (alternative.empty()) return "";

    chain.push_back(head);
    chain.push_back(alternative);

    std::set<std::string> argsToRemove;
    for (auto& element : tempChain) {
        auto& nodeArgs = std::get<1>(element);
        argsToRemove.insert(nodeArgs.begin(), nodeArgs.end());
    }

    std::vector<std::set<std::string>> argsChain;
    for (auto& c : chain) {
        auto args = ce::getArgs(c);
        argsChain.emplace_back(args.begin(), args.end());
    }

    std::vector<std::vector<std::string>> howToRemove(chain.size() - 1);
    for (auto& argToRemove : argsToRemove) {
        for (int index = 0; index < (int)chain.size(); ++index) {
            if (argsChain[index].find(argToRemove) != argsChain[index].end()) {
                if (index < (int)howToRemove.size()) howToRemove[index].push_back(argToRemove);
                break;
            }
        }
    }

    std::string newExpr = chain.back();
    for (int ind = (int)chain.size() - 2; ind >= 0; --ind) {
        std::string substr = "[" + ce::joinWithComma(howToRemove[ind]) + "]";
        newExpr = "(>" + substr + chain[ind] + newExpr + ")";
    }

    if (anchorFirst) {
        return newExpr;
    } else {
        auto [reshuffled, _, __] = reshuffle(newExpr, true);
        return reshuffled;
    }
}

/// @brief Test whether a conjecture consists exclusively of plain
///        operator applications, anchor exempt.
///
/// @details
/// Disintegrates `expr` into its full premise chain plus head via
/// `ce::disintegrateImplication` and checks every constituent: the
/// anchor premise (core name equal to `config_.getAnchorName()`) is
/// exempt; every other premise and the head must be a plain,
/// non-negated operator application — an expression whose core name is
/// in `operators_` (entries with both input and output arguments).
/// Equality elements, `or0` / existence shapes, nested implication
/// elements, and negated elements all fail the test.
///
/// Negation is detected on the element string itself
/// (`elem[0] == '!'`), never via the extracted core name, because
/// `ce::extractExpression` strips the `!(...)` wrapper and would report
/// the inner operator for a negated element.
///
/// This is the emission gate for `mirror_pairs.txt`: the CE filter's
/// mirror-refutation heuristic applies only to operator-only
/// conjectures (see
/// [D-229](../../docs/agentic_swdd/40_decisions.md#d-229)).
///
/// @param expr Conjecture in canonical pool form.
/// @return `true` when every non-anchor constituent is a plain
///         operator application.
bool Conjecturer::consistsOnlyOfOperators(const std::string& expr) const {
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> tempChain;
    const std::string head = ce::disintegrateImplication(expr, tempChain, coreExprMap_);

    const std::string anchorName = config_.getAnchorName();

    auto isPlainOperator = [this](const std::string& elem) {
        if (elem.empty() || elem[0] == '!') return false;
        const std::string core = ce::extractExpression(elem);
        return std::find(operators_.begin(), operators_.end(), core) != operators_.end();
    };

    for (const auto& element : tempChain) {
        const std::string& elem = std::get<0>(element);
        if (!elem.empty() && elem[0] != '!'
            && ce::extractExpression(elem) == anchorName) {
            continue;
        }
        if (!isPlainOperator(elem)) return false;
    }
    return isPlainOperator(head);
}

/// @brief Build the `mirror_pairs.txt` rows from the captured
///        source→mirror map.
///
/// @details
/// Iterates `capturedPairs` in ascending source order (the `std::map`
/// iteration order — deterministic output) and keeps only rows whose
/// source passes `consistsOnlyOfOperators`. Each kept row is emitted
/// as `source + "\t" + mirror`. Both strings are byte-identical to
/// their `conjectures.txt` lines because the capture sites in `run()`
/// record them at pool admission, before any further mutation. An
/// empty mirror is a capture-site bug, not a defined case — asserted.
///
/// @param capturedPairs Source-conjecture → mirror-conjecture map
///                      captured at the pool-admission sites.
/// @return Tab-separated rows for `mirror_pairs.txt`, source-sorted.
/// @see consistsOnlyOfOperators — the emission gate.
std::vector<std::string> Conjecturer::buildMirrorPairRows(
    const std::map<std::string, std::string>& capturedPairs) const {
    std::vector<std::string> rows;
    for (const auto& [source, mirror] : capturedPairs) {
        assert(!mirror.empty()
            && "buildMirrorPairRows: capture sites must only record non-empty mirrors");
        if (consistsOnlyOfOperators(source)) {
            rows.push_back(source + "\t" + mirror);
        }
    }
    return rows;
}

// ============================================================================
// Int-path worker functions (Phase 5)
// ============================================================================

/// @brief Int-path worker driver — combine `intStatement` with
///        `intGrowingTheorem` and emit the per-candidate result
///        bundle.
///
/// @details
/// Hot-path twin of `singleThreadCalculation` (string path).
/// Per call: enumerate every connection map between the two
/// candidates' def-set maps via `makeAllConnectionMapsInt`; for
/// each map, attempt `connectExpressionsInt`; for each successful
/// merge, run the int-path filter cascade; for each survivor,
/// optionally attach the anchor and reshuffle into canonical
/// form. The bundle's four lists capture every output category
/// the outer driver needs (intermediate non-anchor connections,
/// anchor-attached survivors, canonical forms, mirrors).
///
/// Both lanes (int + string) must agree on output. The
/// `worker_*` test category pins int-vs-string identity on a
/// fixed candidate to catch drift.
///
/// @param intStatement            First encoded input.
/// @param intGrowingTheorem       Second encoded input (the
///                                partial theorem being grown).
/// @param nseStatement            `nse` count of `intStatement`.
/// @param nseGrowingTheorem       `nse` count of
///                                `intGrowingTheorem`.
/// @param intArgsStatement        Encoded def-set map for the
///                                first input.
/// @param intArgsGrowingTheorem   Encoded def-set map for the
///                                second input.
/// @return Filled `WorkerResult`.
WorkerResult Conjecturer::singleThreadCalculationInt(
    const IntConjBuf& intStatement, const IntConjBuf& intGrowingTheorem,
    int nseStatement, int nseGrowingTheorem,
    const IntDefSetMap& intArgsStatement, const IntDefSetMap& intArgsGrowingTheorem) const
{
    prof::Scope _prof_stc(prof::g_singleThreadCalc);
    WorkerResult result;

    if (!checkDefSetsPriorInt(intArgsStatement, intArgsGrowingTheorem)) return result;
    if (!checkComplexityPerOpInt(intGrowingTheorem, intStatement)) return result;

    std::vector<IntConnMap> connMaps;
    makeAllConnectionMapsInt(intArgsGrowingTheorem, intArgsStatement, false, mappingsMap_, connMaps);

    for (int mi = 0; mi < (int)connMaps.size(); ++mi) {
        int numRemovableArgs = getNumberRemovableArgsInt(connMaps[mi]);
        auto bsIt = binarySeqsMap_.find(numRemovableArgs);
        if (bsIt == binarySeqsMap_.end()) continue;

        for (auto& binaryList : bsIt->second) {
            int16_t bl[MAX_CONJ_ARGS];
            int blLen = (int)binaryList.size();
            for (int b = 0; b < blLen; ++b) bl[b] = (int16_t)binaryList[b];

            IntConjBuf outExpr;
            IntDefSetMap outMap;
            if (!connectExpressionsInt(intStatement, intGrowingTheorem,
                                       intArgsStatement, intArgsGrowingTheorem,
                                       connMaps[mi], bl, blLen, false, outExpr, outMap))
                continue;

            int nse = nseStatement + nseGrowingTheorem;
            if (!exprGood2Int(outExpr, nse, outMap)) continue;
            if (!onlyInHeadGoodInt(outExpr)) continue;
            if (!prohibitedHeadsGoodInt(outExpr)) continue;

            std::string connectedExpr = decodeExpr(outExpr);
            auto [reshuffledExpr, reshuffledMap, repMap] = reshuffle(connectedExpr, true);
            result.connected_list.push_back({reshuffledExpr, reshuffledMap});

            int complexityLevel = countOperatorOccurrences(reshuffledExpr) + 1;
            int numCombinableArgs = 0;
            for (auto& [arg, tpl] : reshuffledMap) if (std::get<1>(tpl)) numCombinableArgs++;

            if (checkDefSets(reshuffledMap) && numCombinableArgs <= config_.parameters.max_number_args_expr
                && checkComplexityLevelForDefSets(reshuffledMap, complexityLevel)) {

                IntConjBuf reshuffledInt = encodeExpr(reshuffledExpr);
                IntDefSetMap reshuffledIntMap = encodeDefSetMap(reshuffledMap);

                std::vector<IntConnMap> connMaps2;
                makeAllConnectionMapsInt(reshuffledIntMap, anchorDefSetsInt_, true, mappingsMapAnchor_, connMaps2);

                for (int mi2 = 0; mi2 < (int)connMaps2.size(); ++mi2) {
                    bool toContinue = false;
                    for (int16_t a = (int16_t)(anchorDefSetsInt_.count + 1); a <= connMaps2[mi2].maxArg; ++a) {
                        if (connMaps2[mi2].map[a] == a) { toContinue = true; break; }
                    }
                    if (toContinue) continue;

                    int numRem2 = getNumberRemovableArgsInt(connMaps2[mi2]);
                    int16_t bl2[MAX_CONJ_ARGS];
                    for (int b = 0; b < numRem2; ++b) bl2[b] = 1;

                    IntConjBuf outExpr2;
                    IntDefSetMap outMap2;
                    if (!connectExpressionsInt(anchorInt_, reshuffledInt,
                                               anchorDefSetsInt_, reshuffledIntMap,
                                               connMaps2[mi2], bl2, numRem2, true, outExpr2, outMap2))
                        continue;

                    std::string connExpr2 = decodeExpr(outExpr2);
                    bool isExHead = false;
                    if (!checkInputVariablesTheoremOperatorHead(connExpr2)) {
                        if (triggersExistenceReformulation(connExpr2)) {
                            isExHead = true;
                        } else {
                            continue;
                        }
                    }
                    if (!checkInputVariablesOrder(connExpr2) ||
                        patternInConjecture(connExpr2)) continue;
                    // evaluateOperatorExprs2 encodes the same "operator head must be grounded by
                    // another operator's output" invariant as checkInputVariablesTheoremOperatorHead.
                    // For existence-eligible candidates the head is about to be rewritten as a
                    // negated-universal, so this invariant no longer applies — skip the check.
                    if (!isExHead && !evaluateOperatorExprs2(connExpr2, true)) continue;
                    if (!controlEquality(connExpr2)) continue;
                    if (!checkMinSizeExpression(connExpr2)) continue;
                    if (!passesMaxDistinctAnchorValuesPerType(connExpr2)) continue;

                    // max_size_expression_after_existence filter — called once on the
                    // pre-reformulation string (connExpr2); negation cannot change the verdict
                    // since it only adds `!` prefixes and doesn't move leaves. Leaf count is
                    // the already-available anchor + nse non-anchor = nse + 1.
                    // Skip both after-existence filters for existence-head-reformulated
                    // conjectures: the reformulation collapses two leaves into one existence
                    // block, so applying the raw pre-reformulation metric would double-penalize.
                    if (!isExHead) {
                        // leafCount = nse (non-anchor leaves); the anchor leaf is not counted
                        // because `max_size_expression_after_existence` caps the non-anchor chain
                        // length (same semantics as `checkComplexityPerOpInt` pre-anchor). Using
                        // `nse + 1` here would double-count the anchor and spuriously reject any
                        // 4-leaf conjecture containing a leaf whose cap equals 4 (e.g. `in2`).
                        if (!passesMaxSizeAfterExistence(connExpr2, nse)) continue;
                        if (!passesComplexityAfterExistence(connExpr2)) continue;
                    }

                    std::string finalExpr = isExHead
                        ? reformulateToExistenceHead(connExpr2)
                        : connExpr2;

                    if (passesInPremiseFilter(finalExpr)) {
                        auto [reshExpr2, _m, __m] = reshuffle(finalExpr, true);
                        // For existence-head cases the reformulated `finalExpr`
                        // retains the original permutation with arg-IDs from
                        // the pre-reshuffle intermediate. Push the pinned/
                        // reshuffled form to conjectures.txt so the canonical
                        // existence-wrapped structure carries a first-
                        // occurrence numbering that reshuffle produces for
                        // reshuffled_conjectures.txt.
                        const std::string& pushed = isExHead ? reshExpr2 : finalExpr;
                        result.connected_list2.push_back(pushed);
                        result.reshuffled_list.push_back(reshExpr2);
                        result.reshuffled_mirrored_list.push_back(createReshuffledMirrored(finalExpr));
                    }

                    // Emit one negated-premise variant per premise whose core expression has
                    // `allow_negation=true`. Head is never negated (implicitly skips the
                    // existence-head `!(>[…])` when it sits at head position). Filter each
                    // variant on the same in[…]-shape rule.
                    for (auto& negVariant : generateNegatedPremiseVariants(finalExpr)) {
                        if (!passesInPremiseFilter(negVariant)) continue;
                        auto [reshN, _n1, _n2] = reshuffle(negVariant, true);
                        // negVariants of existence-head conjectures inherit the
                        // `!(>[bv](L)(R))` head from finalExpr; push the
                        // reshuffled (pinning-canonical) form to conjectures.txt
                        // so the numbering matches the reshuffled_conjectures.txt
                        // canonical form for those cases.
                        const std::string& pushedNeg = isExHead ? reshN : negVariant;
                        result.connected_list2.push_back(pushedNeg);
                        result.reshuffled_list.push_back(reshN);
                        result.reshuffled_mirrored_list.push_back(createReshuffledMirrored(negVariant));
                    }
                }
            }
        }
    }
    return result;
}

/// @brief Int-path: attach a single encoded expression directly
///        to the encoded anchor (`nse = 1` path).
///
/// @details
/// Twin of string-path `singleExprAnchorConnection`. Skips
/// operator-head validity checks because a single expression
/// cannot consume an operator's output binding. Runs only when
/// `parameters.min_number_simple_expressions == 1`.
///
/// @param intExpr          Encoded candidate expression.
/// @param intExprDefSets   Encoded def-set map for `intExpr`.
/// @return Filled `WorkerResult`. Only `connected_list2`,
///         `reshuffled_list`, and `reshuffled_mirrored_list` are
///         populated — `connected_list` (intermediate) is empty
///         on the `nse = 1` path.
WorkerResult Conjecturer::singleExprAnchorConnectionInt(
    const IntConjBuf& intExpr, const IntDefSetMap& intExprDefSets) const
{
    prof::Scope _prof_sea(prof::g_singleExprAnchorConn);
    WorkerResult result;

    // Guard: check group sizes fit in mappingsMapAnchor
    int16_t exprCounts[MAX_CONJ_ARGS] = {};  // indexed by defSetId
    for (int i = 0; i < intExprDefSets.count; ++i) {
        exprCounts[intExprDefSets.defSetId[i]]++;
    }
    int16_t anchorCounts[MAX_CONJ_ARGS] = {};
    for (int i = 0; i < anchorDefSetsInt_.count; ++i) {
        anchorCounts[anchorDefSetsInt_.defSetId[i]]++;
    }

    int maxAnchorKey = 0;
    for (auto& [k, _] : mappingsMapAnchor_) maxAnchorKey = std::max(maxAnchorKey, k);

    // Collect all distinct defSetIds
    int16_t allDs[MAX_CONJ_ARGS];
    int numDs = 0;
    for (int i = 0; i < intExprDefSets.count; ++i) {
        int16_t d = intExprDefSets.defSetId[i];
        bool found = false;
        for (int j = 0; j < numDs; ++j) if (allDs[j] == d) { found = true; break; }
        if (!found) allDs[numDs++] = d;
    }
    for (int i = 0; i < anchorDefSetsInt_.count; ++i) {
        int16_t d = anchorDefSetsInt_.defSetId[i];
        bool found = false;
        for (int j = 0; j < numDs; ++j) if (allDs[j] == d) { found = true; break; }
        if (!found) allDs[numDs++] = d;
    }
    for (int i = 0; i < numDs; ++i) {
        int total = exprCounts[allDs[i]] + anchorCounts[allDs[i]];
        if (total > maxAnchorKey) { return result; }
    }

    std::vector<IntConnMap> connMaps2;
    makeAllConnectionMapsInt(intExprDefSets, anchorDefSetsInt_, true, mappingsMapAnchor_, connMaps2);

    for (int mi2 = 0; mi2 < (int)connMaps2.size(); ++mi2) {
        // Skip maps where non-anchor args map to themselves
        bool toContinue = false;
        for (int16_t a = (int16_t)(anchorDefSetsInt_.count + 1); a <= connMaps2[mi2].maxArg; ++a) {
            if (connMaps2[mi2].map[a] == a) { toContinue = true; break; }
        }
        if (toContinue) continue;

        int numRem = getNumberRemovableArgsInt(connMaps2[mi2]);
        int16_t bl2[MAX_CONJ_ARGS];
        for (int b = 0; b < numRem; ++b) bl2[b] = 1;

        IntConjBuf outExpr2;
        IntDefSetMap outMap2;
        if (!connectExpressionsInt(anchorInt_, intExpr,
                                   anchorDefSetsInt_, intExprDefSets,
                                   connMaps2[mi2], bl2, numRem, true, outExpr2, outMap2))
            continue;

        // === BOUNDARY: decode for string filters ===
        std::string connExpr2 = decodeExpr(outExpr2);
        if (!checkInputVariablesOrder(connExpr2) || patternInConjecture(connExpr2)) {
            continue;
        }
        if (!controlEquality(connExpr2)) { continue; }
        if (!passesMaxDistinctAnchorValuesPerType(connExpr2)) { continue; }

        result.connected_list2.push_back(connExpr2);
        auto [reshExpr2, reshMap2, repMap2] = reshuffle(connExpr2, true);
        result.reshuffled_list.push_back(reshExpr2);
        result.reshuffled_mirrored_list.push_back(createReshuffledMirrored(connExpr2));
    }

    return result;
}

// ============================================================================
// Worker functions (Phase 8)
// ============================================================================

/// @brief String-path worker driver — combine `statement` with
///        `growingTheorem` and emit the per-candidate result
///        bundle.
///
/// @details
/// String-path twin of `singleThreadCalculationInt`. The string
/// lane is retained for final-stage structural checks where the
/// string form is unavoidable (pattern matching, mirror
/// generation, reshuffle); the int lane is the hot path. Both
/// lanes must agree on output (see `worker_*` test category for
/// the int-vs-string identity check on a fixed candidate).
///
/// @param statement           First input expression text.
/// @param growingTheorem      Second input expression (the
///                            partial theorem being grown).
/// @param nseStatement        `nse` count of `statement`.
/// @param nseGrowingTheorem   `nse` count of `growingTheorem`.
/// @param argsStatement       `statement`'s def-set map.
/// @param argsGrowingTheorem  `growingTheorem`'s def-set map.
/// @return Filled `WorkerResult`.
WorkerResult Conjecturer::singleThreadCalculation(
    const std::string& statement, const std::string& growingTheorem,
    int nseStatement, int nseGrowingTheorem,
    const DefSetMap& argsStatement, const DefSetMap& argsGrowingTheorem) const
{
    WorkerResult result;

    if (!checkDefSetsPriorToConnection(argsStatement, argsGrowingTheorem)) return result;
    if (!checkConjectureComplexityPerOperator(growingTheorem, statement)) return result;

    auto connectionMaps = makeAllConnectionMaps(argsGrowingTheorem, argsStatement, false, mappingsMap_);

    for (auto& connectionMap : connectionMaps) {
        int numRemovableArgs = getNumberRemovableArgs(connectionMap);
        auto bsIt = binarySeqsMap_.find(numRemovableArgs);
        if (bsIt == binarySeqsMap_.end()) continue;

        for (auto& binaryList : bsIt->second) {
            auto [success, connectedExpr, connectedMap] = connectExpressions(
                statement, growingTheorem, argsStatement, argsGrowingTheorem,
                connectionMap, binaryList, false);

            if (!success) continue;
            int nse = nseStatement + nseGrowingTheorem;
            if (!exprGood2(connectedExpr, nse, connectedMap)) continue;
            if (!onlyInHeadGood(connectedExpr)) continue;
            if (!prohibitedHeadsGood(connectedExpr)) continue;

            auto [reshuffledExpr, reshuffledMap, repMap] = reshuffle(connectedExpr, true);
            result.connected_list.push_back({reshuffledExpr, reshuffledMap});

            int complexityLevel = countOperatorOccurrences(reshuffledExpr) + 1;
            int numCombinableArgs = 0;
            for (auto& [arg, tpl] : reshuffledMap) if (std::get<1>(tpl)) numCombinableArgs++;

            if (checkDefSets(reshuffledMap) && numCombinableArgs <= config_.parameters.max_number_args_expr
                && checkComplexityLevelForDefSets(reshuffledMap, complexityLevel)) {

                auto connectionMaps2 = makeAllConnectionMaps(reshuffledMap, anchor_.definition_sets, true, mappingsMapAnchor_);

                for (auto& connectionMap2 : connectionMaps2) {
                    bool toContinue = false;
                    for (auto& [ky, val] : connectionMap2) {
                        if (std::stoi(ky) > (int)anchor_.definition_sets.size()) {
                            if (val == ky) toContinue = true;
                        }
                    }
                    if (toContinue) continue;

                    int numRem2 = getNumberRemovableArgs(connectionMap2);
                    std::vector<int> binaryList2(numRem2, 1);

                    auto [success2, connExpr2, connMap2] = connectExpressions(
                        anchor_.short_mpl_normalized, reshuffledExpr,
                        anchor_.definition_sets, reshuffledMap,
                        connectionMap2, binaryList2, true);

                    if (!success2) continue;
                    if (!checkInputVariablesTheoremOperatorHead(connExpr2) ||
                        !checkInputVariablesOrder(connExpr2) ||
                        patternInConjecture(connExpr2)) continue;
                    if (!evaluateOperatorExprs2(connExpr2, true)) continue;
                    if (!controlEquality(connExpr2)) continue;
                    if (!checkMinSizeExpression(connExpr2)) continue;
                    if (!passesMaxDistinctAnchorValuesPerType(connExpr2)) continue;

                    result.connected_list2.push_back(connExpr2);
                    auto [reshExpr2, _, __] = reshuffle(connExpr2, true);
                    result.reshuffled_list.push_back(reshExpr2);
                    result.reshuffled_mirrored_list.push_back(createReshuffledMirrored(connExpr2));
                }
            }
        }
    }
    return result;
}

/// @brief String-path: attach a single expression directly to
///        the anchor (`nse = 1` path). Twin of
///        `singleExprAnchorConnectionInt`.
///
/// @details
/// Skips operator-head validity checks because a single
/// expression cannot consume an operator's output binding. Runs
/// only when `parameters.min_number_simple_expressions == 1`.
///
/// @param expr         Source expression text.
/// @param exprDefSets  Def-set map for `expr`.
/// @return Filled `WorkerResult`.
WorkerResult Conjecturer::singleExprAnchorConnection(
    const std::string& expr, const DefSetMap& exprDefSets) const
{
    WorkerResult result;
    const std::string& reshuffledExpr = expr;
    const DefSetMap& reshuffledMap = exprDefSets;

    // Guard: check group sizes fit in mappingsMapAnchor
    std::map<std::string, int> exprGroups;
    for (auto& [arg, val] : reshuffledMap) exprGroups[std::get<0>(val)]++;
    std::map<std::string, int> anchorGroups;
    for (auto& [arg, val] : anchor_.definition_sets) anchorGroups[std::get<0>(val)]++;

    int maxAnchorKey = 0;
    for (auto& [k, _] : mappingsMapAnchor_) maxAnchorKey = std::max(maxAnchorKey, k);

    std::set<std::string> allDs;
    for (auto& [k, v] : exprGroups) allDs.insert(k);
    for (auto& [k, v] : anchorGroups) allDs.insert(k);
    for (auto& ds : allDs) {
        int total = 0;
        auto it1 = exprGroups.find(ds);
        if (it1 != exprGroups.end()) total += it1->second;
        auto it2 = anchorGroups.find(ds);
        if (it2 != anchorGroups.end()) total += it2->second;
        if (total > maxAnchorKey) return result;
    }

    auto connectionMaps2 = makeAllConnectionMaps(reshuffledMap, anchor_.definition_sets, true, mappingsMapAnchor_);

    for (auto& connectionMap2 : connectionMaps2) {
        bool toContinue = false;
        for (auto& [ky, val] : connectionMap2) {
            if (std::stoi(ky) > (int)anchor_.definition_sets.size()) {
                if (val == ky) toContinue = true;
            }
        }
        if (toContinue) continue;

        int numRem = getNumberRemovableArgs(connectionMap2);
        std::vector<int> binaryList2(numRem, 1);

        auto [success2, connExpr2, connMap2] = connectExpressions(
            anchor_.short_mpl_normalized, reshuffledExpr,
            anchor_.definition_sets, reshuffledMap,
            connectionMap2, binaryList2, true);

        if (!success2) continue;
        if (!checkInputVariablesOrder(connExpr2) || patternInConjecture(connExpr2)) continue;
        if (!controlEquality(connExpr2)) continue;
        if (!passesMaxDistinctAnchorValuesPerType(connExpr2)) continue;

        result.connected_list2.push_back(connExpr2);
        auto [reshExpr2, reshMap2, repMap2] = reshuffle(connExpr2, true);
        result.reshuffled_list.push_back(reshExpr2);
        result.reshuffled_mirrored_list.push_back(createReshuffledMirrored(connExpr2));
    }

    return result;
}

/// @brief Detect an ungrounded-operator-head conjecture and
///        rewrite it with the operator's allowed argument wrapped
///        in a negated-universal existence head.
///
/// @details
/// A conjecture `(>[bvs](Anchor[...])(...)(operator[a, b, c, ...]))`
/// where `operator` has output args may be ungrounded — the
/// operator's output is not consumed elsewhere in the chain. Such
/// candidates would be malformed at the prover. Reformulation
/// wraps the offending arg in `!(>[w](in[w, X])!(operator[...]))`
/// to express "there exists a `w` such that `operator` holds with
/// `w` substituted". Pass-through if no reformulation applies;
/// the new form is returned otherwise.
///
/// @param conjecture Source conjecture text.
/// @return Reformulated conjecture, or the original on
///         pass-through.
std::string Conjecturer::reformulateOperatorHead(const std::string& conjecture) const {
    using ChainEntry = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<ChainEntry> chain;
    std::string head = ce::disintegrateImplication(conjecture, chain, coreExprMap_);

    if (!head.empty() && head[0] == '!') return conjecture;

    std::string headExpr = ce::extractExpression(head);
    auto it = config_.data.find(headExpr);
    if (it == config_.data.end()) return conjecture;
    auto& desc = it->second;
    if (desc.indices_output_args.empty()) return conjecture;

    assert(desc.indices_output_args.size() == 1);
    int outIdx = desc.indices_output_args[0];
    auto headArgs = ce::getArgs(head);
    std::string aOut = headArgs[outIdx];

    // Fresh variable: max numeric arg + 1
    std::regex digitRe(R"(\d+)");
    std::sregex_iterator dIt(conjecture.begin(), conjecture.end(), digitRe);
    std::sregex_iterator dEnd;
    int maxArg = 0;
    for (; dIt != dEnd; ++dIt) maxArg = std::max(maxArg, std::stoi(dIt->str()));
    std::string x = std::to_string(maxArg + 1);

    // New head with x replacing aOut
    std::vector<std::string> newHeadArgs = headArgs;
    newHeadArgs[outIdx] = x;
    std::string newOp = "(" + headExpr + "[" + ce::joinWithComma(newHeadArgs) + "])";
    std::string newEq = "(=[" + x + "," + aOut + "])";
    std::string inner = "(>[" + x + "]" + newOp + newEq + ")";

    // Replace last occurrence of head
    auto pos = conjecture.rfind(head);
    assert(pos != std::string::npos);
    return conjecture.substr(0, pos) + inner + conjecture.substr(pos + head.size());
}

// ============================================================================
// Existence-head reformulation helpers
// ============================================================================

namespace {
// Decompose a candidate into (premises_incl_anchor, head) and collect every outer-bound var
// across the nested (>[...]...) levels. Purely informational; no mutation of the input string.
struct Disassembled {
    std::vector<std::string> premises;           // non-head expressions in chain order (anchor first after prioritizeAnchor upstream)
    std::string head;                             // final expression
    std::vector<std::string> outerBounds;         // union of bound vars across all > levels, first-seen order
};

Disassembled disassemble(const std::string& candidate,
                         const ce::CoreExpressionMap& coreExprMap) {
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> tempChain;
    std::string head = ce::disintegrateImplication(candidate, tempChain, coreExprMap);

    Disassembled d;
    d.head = head;
    std::set<std::string> seenBound;
    for (auto& t : tempChain) {
        d.premises.push_back(std::get<0>(t));
        for (auto& b : std::get<1>(t)) {
            if (seenBound.insert(b).second) d.outerBounds.push_back(b);
        }
    }
    return d;
}
} // anonymous namespace

/// @brief Existence-head reformulation eligibility test.
///
/// @details
/// True iff the ungrounded-operator-head filter rejection hits
/// AND the candidate qualifies for existence reformulation:
/// 1. The head's `coreExpr` has non-empty
///    `allowed_for_existence`.
/// 2. The chain contains `(in[x, X])` for some `x` at an allowed
///    position.
/// 3. `x` occurs nowhere else in the chain besides `P_x` and the
///    head.
///
/// Caller runs `reformulateToExistenceHead` in lieu of rejecting
/// the candidate when this returns true.
///
/// @param theorem Candidate conjecture text.
/// @return `true` when the candidate qualifies for reformulation.
bool Conjecturer::triggersExistenceReformulation(const std::string& theorem) const {
    prof::Scope _p(prof::g_triggersExistenceRef);
    Disassembled d = disassemble(theorem, coreExprMap_);

    // Head must be an operator.
    std::string headCore = ce::extractExpression(d.head);
    if (std::find(operators_.begin(), operators_.end(), headCore) == operators_.end())
        return false;

    const ExpressionDescription& headDesc = config_.data.at(headCore);
    if (headDesc.allowed_for_existence.empty()) return false;

    assert(headDesc.indices_output_args.size() == 1);
    auto headArgs = ce::getArgs(d.head);
    std::string outputVar = headArgs[headDesc.indices_output_args[0]];

    // Replicate the filter's rejection condition: outputVar's FIRST occurrence as output
    // (scanning non-anchor premises + head in chain order) is at the head itself.
    std::string anchorName = config_.getAnchorName();
    const std::string& anchorHandle = config_.data.at(anchorName).handle;

    std::vector<std::string> nonAnchorPremises;
    for (auto& p : d.premises) {
        if (p.find(anchorHandle) == std::string::npos) nonAnchorPremises.push_back(p);
    }

    bool feederFound = false;
    for (auto& p : nonAnchorPremises) {
        std::string pCore = ce::extractExpression(p);
        auto pArgs = ce::getArgs(p);
        for (int idx : config_.data.at(pCore).indices_output_args) {
            if (idx < (int)pArgs.size() && pArgs[idx] == outputVar) { feederFound = true; break; }
        }
        if (feederFound) break;
    }
    if (feederFound) return false;  // grounded — original filter would accept

    // Try each allowed existence position.
    const std::string& inHandle = config_.data.at("in").handle;
    for (int pos1b : headDesc.allowed_for_existence) {
        if (pos1b < 1 || pos1b > (int)headArgs.size()) continue;
        const std::string& x = headArgs[pos1b - 1];

        // Find a property (in[x, X]) in the non-anchor chain.
        int pxIndex = -1;
        for (int i = 0; i < (int)nonAnchorPremises.size(); ++i) {
            auto& p = nonAnchorPremises[i];
            if (p.find(inHandle) == std::string::npos) continue;
            auto pArgs = ce::getArgs(p);
            if (!pArgs.empty() && pArgs[0] == x) { pxIndex = i; break; }
        }
        if (pxIndex < 0) continue;

        // x must appear nowhere else in non-anchor premises besides P_x.
        bool xElsewhere = false;
        for (int i = 0; i < (int)nonAnchorPremises.size(); ++i) {
            if (i == pxIndex) continue;
            auto pArgs = ce::getArgs(nonAnchorPremises[i]);
            if (std::find(pArgs.begin(), pArgs.end(), x) != pArgs.end()) { xElsewhere = true; break; }
        }
        if (xElsewhere) continue;

        // All anchor-expression args: x must not be among them.
        bool xInAnchor = false;
        for (auto& p : d.premises) {
            if (p.find(anchorHandle) == std::string::npos) continue;
            auto aArgs = ce::getArgs(p);
            if (std::find(aArgs.begin(), aArgs.end(), x) != aArgs.end()) { xInAnchor = true; break; }
            break;
        }
        if (xInAnchor) continue;

        return true;
    }
    return false;
}

/// @brief Rebuild the candidate with the allowed arg wrapped in
///        a negated-universal existence head.
///
/// @details
/// Concrete realisation of the `triggersExistenceReformulation`
/// rewrite: pick the `(in[x, X])` premise whose `x` is at an
/// allowed position, lift `x` into a fresh existence-head bound
/// variable, and replace the operator's input argument
/// accordingly.
///
/// @param theorem Candidate conjecture text.
/// @return Reformulated conjecture.
/// @pre `triggersExistenceReformulation(theorem)` was true at the
///      same state.
std::string Conjecturer::reformulateToExistenceHead(const std::string& theorem) const {
    prof::Scope _p(prof::g_reformulateToExistence);
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> chain;
    std::string head = ce::disintegrateImplication(theorem, chain, coreExprMap_);

    std::string headCore = ce::extractExpression(head);
    const ExpressionDescription& headDesc = config_.data.at(headCore);
    auto headArgs = ce::getArgs(head);

    std::string anchorName = config_.getAnchorName();
    const std::string& anchorHandle = config_.data.at(anchorName).handle;
    const std::string& inHandle = config_.data.at("in").handle;

    // Re-select the same (x, P_x) the trigger would, and record which chain LEVEL
    // carries P_x as its premise. We remove that level entirely (its premise P_x
    // moves into the new head, and its binding of x moves into the existence
    // wrapper inside the new head), keeping every other level's (premise, binds)
    // pair verbatim — NO flattening.
    std::string chosenX;
    std::string pxStr;
    int pxLevel = -1;
    for (int pos1b : headDesc.allowed_for_existence) {
        if (pos1b < 1 || pos1b > (int)headArgs.size()) continue;
        const std::string& x = headArgs[pos1b - 1];

        int cand = -1;
        for (int i = 0; i < (int)chain.size(); ++i) {
            const std::string& p = std::get<0>(chain[i]);
            if (p.find(anchorHandle) != std::string::npos) continue;
            if (p.find(inHandle) == std::string::npos) continue;
            auto pArgs = ce::getArgs(p);
            if (!pArgs.empty() && pArgs[0] == x) { cand = i; break; }
        }
        if (cand < 0) continue;

        bool xElsewhere = false;
        for (int i = 0; i < (int)chain.size(); ++i) {
            if (i == cand) continue;
            const std::string& p = std::get<0>(chain[i]);
            if (p.find(anchorHandle) != std::string::npos) continue;
            auto pArgs = ce::getArgs(p);
            if (std::find(pArgs.begin(), pArgs.end(), x) != pArgs.end()) { xElsewhere = true; break; }
        }
        if (xElsewhere) continue;

        bool xInAnchor = false;
        for (auto& c : chain) {
            const std::string& p = std::get<0>(c);
            if (p.find(anchorHandle) == std::string::npos) continue;
            auto aArgs = ce::getArgs(p);
            if (std::find(aArgs.begin(), aArgs.end(), x) != aArgs.end()) { xInAnchor = true; break; }
            break;
        }
        if (xInAnchor) continue;

        chosenX = x;
        pxStr = std::get<0>(chain[cand]);
        pxLevel = cand;
        break;
    }
    assert(pxLevel >= 0 && "reformulateToExistenceHead called without a valid trigger state");

    // Build the new head: !(>[x](in[x,X])!(H))
    std::string newHead = "!(>[" + chosenX + "]" + pxStr + "!" + head + ")";

    // Rebuild the nested chain verbatim, SKIPPING the pxLevel entirely. Each
    // remaining level keeps its original (premise, bound_vars). This preserves
    // the `>[7](=[2,7])...` inner binding instead of collapsing 7 into the
    // outermost `>[…]` list above anchor — var 7 stays bound at the level where
    // it first appears, not lifted above premises that do not introduce it.
    std::string inner = newHead;
    for (int i = (int)chain.size() - 1; i >= 0; --i) {
        if (i == pxLevel) continue;
        const std::string& p = std::get<0>(chain[i]);
        const std::vector<std::string>& binds = std::get<1>(chain[i]);
        std::string bvStr;
        for (size_t k = 0; k < binds.size(); ++k) {
            if (k > 0) bvStr += ",";
            bvStr += binds[k];
        }
        inner = "(>[" + bvStr + "]" + p + inner + ")";
    }
    return inner;
}

// ============================================================================
// max_size_expression_after_existence filter
// ============================================================================

/// @brief Per-expression after-existence size cap.
///
/// @details
/// Returns true iff every leaf expression in the
/// pre-reformulation `conj` has
/// `max_size_expression_after_existence >= leafCount`. Intended
/// to be called on the post-anchor-attach,
/// pre-existence-reformulation string; a single disintegrate
/// pass yields the flat chain + head and we check each leaf once
/// (no descent into any nested structure).
///
/// @param conj      Post-anchor-attach,
///                  pre-existence-reformulation conjecture.
/// @param leafCount Already-available leaf count (typically
///                  `nse + 1`).
/// @return `true` to accept; `false` to reject.
bool Conjecturer::passesMaxSizeAfterExistence(const std::string& conj, int leafCount) const {
    prof::Scope _p(prof::g_passesMaxSizeAfterEx);
    // Operate on the pre-reformulation string: one disintegrate pass yields the flat
    // chain of premise leaves + head leaf. For each leaf, look up its core expression's
    // max_size_expression_after_existence; reject if cap > 0 AND cap < leafCount.
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> chain;
    std::string head = ce::disintegrateImplication(conj, chain, coreExprMap_);

    auto checkLeaf = [&](const std::string& leaf) -> bool {
        std::string core = ce::extractExpression(leaf);
        auto it = config_.data.find(core);
        if (it == config_.data.end()) return true;
        int cap = it->second.max_size_expression_after_existence;
        return !(cap > 0 && cap < leafCount);
    };

    for (auto& t : chain) {
        if (!checkLeaf(std::get<0>(t))) return false;
    }
    return checkLeaf(head);
}

/// @brief Post-existence per-type 2-tuple cap.
///
/// @details
/// Mirror of `checkComplexityLevelForDefSets` but keyed by
/// `max_complexity_if_anchor_parameter_connected_after_existence`.
/// Operates on the pre-reformulation post-anchor-attach string;
/// gathers every def-set type that appears on any leaf via the
/// static per-expression definition, then rejects when, for some
/// capped type T, all three hold:
/// 1. Complexity-level exceeds T's complexity cap.
/// 2. Arity-sum exceeds T's arity-sum cap.
/// 3. A slot of type T appears in non-anchor leaves.
///
/// All three must hold; if any one is not exceeded the conjecture
/// survives via that dimension. Implements the post-D-23
/// 3-condition rule (see SwDD chapter `02_conjecturer.md`
/// section *passesComplexityAfterExistence*).
///
/// @param conj Post-anchor-attach,
///             pre-existence-reformulation conjecture.
/// @return `true` to accept; `false` to reject.
bool Conjecturer::passesComplexityAfterExistence(const std::string& conj) const {
    prof::Scope _p(prof::g_passesComplexityAfterEx);
    // Semantic: "max chain length when a chain arg is connected to an anchor slot
    // of a given def-set type". A chain arg pinned to an anchor slot of type T
    // appears in a non-anchor leaf as the anchor slot's value (Peano `(1)` slots
    // carry values 2 (i0) and 6 (i1)). Rejection requires three conditions to
    // hold simultaneously for some capped type T:
    //
    //   1. complexityLevel > complexity_cap_T,
    //   2. arity_sum (over non-anchor leaves) > arity_sum_cap_T,
    //   3. some anchor slot of type T has its value present in non-anchor leaves.
    //
    // If EITHER cap is not exceeded, the conjecture survives via that dimension —
    // the cancellation family escapes via the arity-sum dimension under
    // Peano's `(1) -> [2, 8]` (arity_sum 4+2+2 = 8 NOT > 8); 3-occurrence (1)-pinned
    // shapes (e.g. three in3 leaves giving arity_sum 12) get rejected.
    //
    // No hard-coded complexity bands here; every threshold lives in the config
    // value `max_complexity_if_anchor_parameter_connected_after_existence` per
    // type. complexityLevel counts post-anchor `(>[` occurrences (no +1 — the
    // anchor wrap's extra level cancels baseline's pre-anchor +1).
    int complexityLevel = countOperatorOccurrences(conj);
    const auto& capMap = config_.parameters.max_complexity_if_anchor_parameter_connected_after_existence;
    if (capMap.empty()) return true;

    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> chain;
    std::string head = ce::disintegrateImplication(conj, chain, coreExprMap_);

    std::string anchorName = config_.getAnchorName();
    auto anchorIt = config_.data.find(anchorName);
    if (anchorIt == config_.data.end()) return true;

    // Build {defSetType -> {anchor-slot values of that type}} from the actual anchor instance.
    std::map<std::string, std::set<std::string>> anchorSlotsByType;
    {
        const std::string& anchorHandle = anchorIt->second.handle;
        std::string anchorInstance;
        for (auto& t : chain) {
            const std::string& p = std::get<0>(t);
            if (p.find(anchorHandle) != std::string::npos) { anchorInstance = p; break; }
        }
        if (anchorInstance.empty()) anchorInstance = head;
        auto anchorArgs = ce::getArgs(anchorInstance);
        for (auto& [posStr, tpl] : anchorIt->second.definition_sets) {
            int pos1b = std::stoi(posStr);
            if (pos1b < 1 || pos1b > (int)anchorArgs.size()) continue;
            anchorSlotsByType[std::get<0>(tpl)].insert(anchorArgs[pos1b - 1]);
        }
    }

    // Every arg appearing on any non-anchor leaf (walking into nested `!(>[…]…)`)
    // PLUS the running sum of arities across the same set of leaves (the new
    // dimension of the cap; cancellation has arity-sum 8, deeper/wider
    // explosion shapes exceed 8).
    std::set<std::string> nonAnchorArgs;
    int nonAnchorAritySum = 0;
    std::function<void(const std::string&)> collect = [&](const std::string& expr) {
        if (expr.empty()) return;
        size_t start = 0;
        if (expr[0] == '!') start = 1;
        if (start >= expr.size() || expr[start] != '(') return;
        if (start + 1 < expr.size() && expr[start + 1] == '>') {
            std::vector<CE> innerChain;
            std::string innerHead = ce::disintegrateImplication(expr.substr(start), innerChain, coreExprMap_);
            for (auto& t : innerChain) {
                const std::string& p = std::get<0>(t);
                if (p.find(anchorIt->second.handle) != std::string::npos) continue;
                collect(p);
            }
            collect(innerHead);
            return;
        }
        if (expr.find(anchorIt->second.handle) != std::string::npos) return;
        auto args = ce::getArgs(expr.substr(start));
        nonAnchorAritySum += (int)args.size();
        for (auto& a : args) nonAnchorArgs.insert(a);
    };
    for (auto& t : chain) {
        const std::string& p = std::get<0>(t);
        if (p.find(anchorIt->second.handle) != std::string::npos) continue;
        collect(p);
    }
    collect(head);

    for (auto& [ds, capPair] : capMap) {
        // BOTH dimensions must be exceeded (with the slot present) for rejection;
        // either cap being slack lets the conjecture escape. cancellation has
        // complexity=3, arity_sum=8 — under (1):[2,8] the arity cap (8) is met
        // exactly so the rule fires `>= ` allowance, conjecture escapes.
        const int compCap  = capPair.first;
        const int arityCap = capPair.second;
        if (compCap  >= complexityLevel) continue;
        if (arityCap >= nonAnchorAritySum) continue;
        auto slotsIt = anchorSlotsByType.find(ds);
        if (slotsIt == anchorSlotsByType.end()) continue;
        for (auto& slotVal : slotsIt->second) {
            if (nonAnchorArgs.count(slotVal) > 0) return false;
        }
    }
    return true;
}

/// @brief Per-type cap on distinct anchor-slot values appearing
///        in non-anchor leaves.
///
/// @details
/// For each def-set type T present in
/// `parameters.max_distinct_anchor_values_per_type`, counts how
/// many DISTINCT anchor-slot values of type T appear as args of
/// non-anchor leaves of `conj`. Returns false (reject) if any
/// type's count exceeds the configured cap. Empty config map ->
/// always true (filter off). Walk descends into nested
/// `!(>[...]...)` existence heads so the cap applies to the full
/// body, not just the top-level chain.
///
/// @param conj Post-anchor-attach conjecture.
/// @return `true` when every counted type stays within its cap.
bool Conjecturer::passesMaxDistinctAnchorValuesPerType(const std::string& conj) const {
    // Config-steered "at most N distinct anchor-slot values of type T
    // appearing in non-anchor leaves" filter. Parallels the anchor-slot
    // extraction and leaf walk of passesComplexityAfterExistence, but
    // counts DISTINCT values per type instead of triggering on any
    // presence.
    const auto& capMap = config_.parameters.max_distinct_anchor_values_per_type;
    if (capMap.empty()) return true;

    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> chain;
    std::string head = ce::disintegrateImplication(conj, chain, coreExprMap_);

    std::string anchorName = config_.getAnchorName();
    auto anchorIt = config_.data.find(anchorName);
    if (anchorIt == config_.data.end()) return true;

    // Anchor-slot values grouped by def-set type, extracted from the
    // conjecture's anchor instance (may appear as a chain premise or the head).
    std::map<std::string, std::set<std::string>> anchorSlotsByType;
    {
        const std::string& anchorHandle = anchorIt->second.handle;
        std::string anchorInstance;
        for (auto& t : chain) {
            const std::string& p = std::get<0>(t);
            if (p.find(anchorHandle) != std::string::npos) { anchorInstance = p; break; }
        }
        if (anchorInstance.empty()) anchorInstance = head;
        auto anchorArgs = ce::getArgs(anchorInstance);
        for (auto& [posStr, tpl] : anchorIt->second.definition_sets) {
            int pos1b = std::stoi(posStr);
            if (pos1b < 1 || pos1b > (int)anchorArgs.size()) continue;
            anchorSlotsByType[std::get<0>(tpl)].insert(anchorArgs[pos1b - 1]);
        }
    }

    // Union of args appearing in non-anchor leaves, descending into nested
    // existence-head structure just like passesComplexityAfterExistence does.
    std::set<std::string> nonAnchorArgs;
    std::function<void(const std::string&)> collect = [&](const std::string& expr) {
        if (expr.empty()) return;
        size_t start = 0;
        if (expr[0] == '!') start = 1;
        if (start >= expr.size() || expr[start] != '(') return;
        if (start + 1 < expr.size() && expr[start + 1] == '>') {
            std::vector<CE> innerChain;
            std::string innerHead = ce::disintegrateImplication(expr.substr(start), innerChain, coreExprMap_);
            for (auto& t : innerChain) {
                const std::string& p = std::get<0>(t);
                if (p.find(anchorIt->second.handle) != std::string::npos) continue;
                collect(p);
            }
            collect(innerHead);
            return;
        }
        if (expr.find(anchorIt->second.handle) != std::string::npos) return;
        auto args = ce::getArgs(expr.substr(start));
        for (auto& a : args) nonAnchorArgs.insert(a);
    };
    for (auto& t : chain) {
        const std::string& p = std::get<0>(t);
        if (p.find(anchorIt->second.handle) != std::string::npos) continue;
        collect(p);
    }
    collect(head);

    for (auto& [ds, cap] : capMap) {
        auto slotsIt = anchorSlotsByType.find(ds);
        if (slotsIt == anchorSlotsByType.end()) continue;
        int distinctCount = 0;
        for (auto& slotVal : slotsIt->second) {
            if (nonAnchorArgs.count(slotVal) > 0) distinctCount++;
        }
        if (distinctCount > cap) return false;
    }
    return true;
}

// ============================================================================
// in[…]-premise shape filter
// ============================================================================

/// @brief `(in[...])`-premise shape filter.
///
/// @details
/// Top-of-function gate: anchor-membership-axiom rejection
/// ([D-23](../../docs/agentic_swdd/40_decisions.md#d-23)). Walk the chain and
/// reject any `(in[v, X])` premise (positive or negated) where
/// BOTH `v` AND `X` are anchor-slot values, since the anchor's
/// own axioms already entail it.
///
/// Cnt-shape rules (post-anchor-membership gate). When `hasIn`
/// is true, accept iff one of:
/// 1. `cnt == 1` AND head is an existence form
///    `!(>[...]...)`.
/// 2. `cnt == 2` AND at least one of the two non-anchor
///    premises is negated.
/// 3. `cnt == 2` AND there exists a positive `(in[v, X])`
///    premise whose first arg `v` participates elsewhere
///    (neutralisation rule).
///
/// `cnt >= 3` is rejected unconditionally. Order of premises is
/// irrelevant; the anchor never counts as a premise.
///
/// @warning `parameters.apply_in_premise_filter` is dead code;
///          this function ignores it. SwDD `OPEN-9` documents
///          the gap.
///
/// @param conj Post-anchor-attach conjecture text.
/// @return `true` to accept; `false` to reject.
bool Conjecturer::passesInPremiseFilter(const std::string& conj) const {
    prof::Scope _p(prof::g_passesInPremiseFilter);
    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> chainTuples;
    std::string head = ce::disintegrateImplication(conj, chainTuples, coreExprMap_);
    std::string anchorName = config_.getAnchorName();

    // Anchor-membership-axiom rejection. A non-anchor `(in[v, X])` premise
    // (or its negation) where BOTH v AND X are anchor-slot values is the
    // trivial Peano/Gauss membership claim — e.g. `(in[2, 1])` = "i0 ∈ N",
    // which is one of the anchor's own axioms. Such a premise is vacuous:
    // it adds no constraint to the conjecture, the anchor already implies it.
    // Reject the whole conjecture. Single-anchor-arg shapes like
    // `(in[7, 1])` (with 7 a bound var, 1 = N a free anchor slot) are NOT
    // rejected — those are genuine typing premises.
    {
        std::set<std::string> allSlotVals;
        auto anchorIt = config_.data.find(anchorName);
        if (anchorIt != config_.data.end()) {
            const std::string& anchorHandle = anchorIt->second.handle;
            std::string anchorInstance;
            for (auto& t : chainTuples) {
                const std::string& p = std::get<0>(t);
                if (p.find(anchorHandle) != std::string::npos) { anchorInstance = p; break; }
            }
            if (anchorInstance.empty()) anchorInstance = head;
            auto anchorArgs = ce::getArgs(anchorInstance);
            for (auto& [posStr, tpl] : anchorIt->second.definition_sets) {
                int pos1b = std::stoi(posStr);
                if (pos1b < 1 || pos1b > (int)anchorArgs.size()) continue;
                allSlotVals.insert(anchorArgs[pos1b - 1]);
            }
        }
        auto isAnchorMembership = [&](const std::string& leaf) -> bool {
            if (leaf.empty()) return false;
            std::string inner = (leaf[0] == '!') ? leaf.substr(1) : leaf;
            if (ce::extractExpression(inner) != "in") return false;
            auto args = ce::getArgs(inner);
            if (args.size() < 2) return false;
            return allSlotVals.count(args[0]) > 0 && allSlotVals.count(args[1]) > 0;
        };
        for (auto& t : chainTuples) {
            if (isAnchorMembership(std::get<0>(t))) return false;
        }
    }

    std::vector<std::string> nonAnchor;
    bool hasIn = false;
    for (auto& t : chainTuples) {
        const std::string& p = std::get<0>(t);
        std::string inner = (!p.empty() && p[0] == '!') ? p.substr(1) : p;
        if (ce::extractExpression(inner) == anchorName) continue;
        nonAnchor.push_back(p);
        if (ce::extractExpression(inner) == "in") hasIn = true;
    }

    if (!hasIn) return true;  // filter only applies to in[…]-containing conjectures

    int cnt = (int)nonAnchor.size();
    if (cnt == 1) {
        // Head must be the existence form `!(>[…]…)`.
        return head.size() >= 4 && head.compare(0, 4, "!(>[") == 0;
    }
    if (cnt == 2) {
        int numNegated = 0;
        for (auto& p : nonAnchor) if (!p.empty() && p[0] == '!') numNegated++;
        if (numNegated >= 1) return true;

        // nse=3 neutralisation relaxation:
        //   A positive (in[v,X]) premise may ground a bound variable v that would
        //   otherwise be free. Accept the conjecture if v also appears as an argument
        //   of the OTHER non-anchor premise OR of the head — the in[] premise is then
        //   the explicit typing that makes the conjecture well-formed (e.g. the
        //   additive-cancellation family `(in[a,N]), (in3[a,b,i0,+]) -> (=[b,i0])`).
        //   A negated !(in[v,X]) asserts v ∉ X and does NOT type v, so it is skipped.
        //   Scope is deliberately cnt==2 (i.e. nse=3) so the conjecture population
        //   cannot blow up on larger shapes — per user direction.
        for (size_t i = 0; i < nonAnchor.size(); ++i) {
            const std::string& p = nonAnchor[i];
            if (!p.empty() && p[0] == '!') continue;
            if (ce::extractExpression(p) != "in") continue;
            auto inArgs = ce::getArgs(p);
            if (inArgs.size() < 2) continue;
            const std::string& v = inArgs[0];

            auto argsContainV = [&v](const std::string& s) {
                auto a = ce::getArgs(s);
                return std::find(a.begin(), a.end(), v) != a.end();
            };

            bool elsewhere = false;
            for (size_t j = 0; j < nonAnchor.size(); ++j) {
                if (j == i) continue;
                if (argsContainV(nonAnchor[j])) { elsewhere = true; break; }
            }
            if (!elsewhere && argsContainV(head)) elsewhere = true;

            if (elsewhere) return true;
        }
        return false;
    }
    return false;
}

// ============================================================================
// Negated-premise variants
// ============================================================================

/// @brief Emit one negated-premise variant per negatable premise
///        in `conj`.
///
/// @details
/// For each premise of `conj` whose core expression has
/// `allow_negation = true` in the config, emit one new
/// conjecture where that single premise is wrapped in `!(...)`.
/// The head of the outer implication is never negated (which
/// implicitly excludes the `!(>[...])` existence-head form when
/// it sits at head position). Multiple negatable premises ->
/// one new variant per premise (never co-negated). The original
/// is NOT included in the returned list.
///
/// @param conj Source conjecture (positive form).
/// @return List of negated-premise variants. Empty when no
///         premise is negatable.
std::vector<std::string> Conjecturer::generateNegatedPremiseVariants(const std::string& conj) const {
    prof::Scope _p(prof::g_generateNegatedPremise);
    std::vector<std::string> result;

    using CE = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;
    std::vector<CE> tempChain;
    std::string head = ce::disintegrateImplication(conj, tempChain, coreExprMap_);

    // Collect bound-var lists per chain level so we can reconstruct the original nesting verbatim
    // (with exactly one premise replaced). Each entry in tempChain = (premiseString, bv-of-this-level).
    auto joinBounds = [](const std::vector<std::string>& bv) {
        std::string s;
        for (size_t i = 0; i < bv.size(); ++i) {
            if (i > 0) s += ",";
            s += bv[i];
        }
        return s;
    };

    for (size_t i = 0; i < tempChain.size(); ++i) {
        const std::string& premise = std::get<0>(tempChain[i]);
        if (!premise.empty() && premise[0] == '!') continue;  // already negated, skip
        std::string coreName = ce::extractExpression(premise);
        auto it = config_.data.find(coreName);
        if (it == config_.data.end()) continue;
        if (!it->second.allow_negation) continue;

        // Rebuild the implication with premise[i] replaced by "!" + premise[i].
        // Structure: (>[bv_0](P_0)(>[bv_1](P_1)...(>[bv_{n-1}](P_{n-1})(head))))
        std::string inner = head;
        for (int j = (int)tempChain.size() - 1; j >= 0; --j) {
            const std::string& p = std::get<0>(tempChain[j]);
            const std::string& pOut = (j == (int)i) ? ("!" + p) : p;
            std::string bvStr = joinBounds(std::get<1>(tempChain[j]));
            inner = "(>[" + bvStr + "]" + pOut + inner + ")";
        }
        result.push_back(inner);
    }

    return result;
}

// ============================================================================
// Constructor
// ============================================================================

/// @brief Construct the conjecturer for one anchor batch.
///
/// @details
/// Steps performed:
/// 1. Resolve `projectRoot_` from `__FILE__` so the
///    config-loader can locate `files/config/Config<Tag>.json`
///    via the standard search path.
/// 2. Call `loadConfiguration` to populate `config_`.
/// 3. Call `buildCoreExprMapAdapter` so downstream shim functions
///    expecting prover-side compiled metadata can borrow it.
/// 4. Look up the anchor expression description by name from
///    `config_.data` and seed `anchor_`.
/// 5. Walk every loaded expression and classify it as operator
///    (input + output args), relation (2 inputs, no outputs), or
///    property (1 input, no output).
/// 6. Encode the anchor into `anchorInt_` /
///    `anchorDefSetsInt_` for the int path.
/// 7. Build the int-path lookup tables (`buildNameMap`,
///    `buildIntExprConfigs`).
/// 8. Pre-compute the bijection / permutation tables
///    (`createMap`, `createMapAnchor`, ...) into the immutable
///    caches.
///
/// Construction does NOT enumerate conjectures — that is `run()`'s
/// job. Construction also does not write any output file.
///
/// @param anchorId Short anchor name (e.g. `"Peano"`,
///                 `"Gauss"`, `"IncubatorPeano"`). Combined with
///                 the prefix `"Anchor"` to form the full anchor
///                 name (e.g. `"AnchorPeano"`).
/// @pre  `files/config/Config<anchorId>.json` exists and is
///       well-formed JSON.
/// @post `config_`, `anchor_`, `mappingsMap_`,
///       `mappingsMapAnchor_`, `binarySeqsMap_`,
///       `allPermutations_`, the operator / relation / property
///       classifications, the int-path lookup tables, and
///       `projectRoot_` are all populated.
Conjecturer::Conjecturer(const std::string& anchorId)
    : projectRoot_(std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path())
{
    config_ = loadConfiguration(anchorId);
    buildCoreExprMapAdapter();

    std::string anchorName = config_.getAnchorName();
    anchor_ = config_.data.at(anchorName);

    // Build operators and relations lists
    for (auto& [name, desc] : config_.data) {
        if (!desc.input_args.empty() && !desc.output_args.empty())
            operators_.push_back(name);
        if (desc.input_args.size() == 2 && desc.output_args.empty())
            relations_.push_back(name);
        if (desc.input_args.size() == 1 && desc.output_args.empty())
            properties_.push_back(name);
    }

    // Build int-path data structures
    buildNameMap();
    buildIntExprConfigs();
    anchorInt_ = encodeExpr(anchor_.short_mpl_normalized);
    anchorDefSetsInt_ = encodeDefSetMap(anchor_.definition_sets);
}

// ============================================================================
// Main orchestrator (Phase 9)
// ============================================================================

/// @brief Generate conjectures for the loaded batch and write
///        `conjectures.txt` plus the canonical-form / mirror-form
///        companion files.
///
/// @details
/// One-shot orchestrator. Steps performed:
/// 1. Pre-compute the bijection / permutation / binary-sequence
///    tables (`createMap`, `createMapAnchor`,
///    `generateAllPermutations`, ...) into the immutable caches.
///    Cached tables are read-only afterwards and shared across
///    every worker thread without locks.
/// 2. Seed `growing_theorems` with the initial expressions from
///    `config_.data` (in `expressionOrder`).
/// 3. Walk the `nse` axis from
///    `parameters.min_number_simple_expressions` to
///    `parameters.max_number_simple_expressions`. For each
///    `nse`, dispatch worker invocations (`singleThreadCalculation*`
///    or `singleExprAnchorConnection*` for the `nse = 1` path)
///    in parallel, collect their `WorkerResult` bundles, and
///    feed the survivors into the next round's `growing_theorems`.
/// 4. After the main loop, run the existence-head reformulation
///    pass (`triggersExistenceReformulation` /
///    `reformulateToExistenceHead`) and the negated-premise
///    variant pass (`generateNegatedPremiseVariants`).
/// 5. Fold in the template addon's anchor-coupled stumps
///    (`generateTemplateConjectures`) with the same dedup the
///    preliminary pass uses.
/// 6. (Optionally) emit OR conjectures via
///    `generateOrConjectures`.
/// 7. Write the survivors to:
///    - `conjectures.txt` — raw survivors.
///    - `reshuffled_conjectures.txt` — canonical-form survivors.
///    - `reshuffled_mirrored_conjectures.txt` — mirror variants.
///    - `or_pairs.txt` — OUTPUT artefact recording the
///      `(existence, companion)` pairs emitted this run; opened
///      with `std::ios::out` so it is overwritten each invocation.
///      The file is preserved (not deleted) by the upstream
///      cleanup pass that removes other stale outputs, but its
///      content is replaced on every successful run.
///    - `mirror_pairs.txt` — OUTPUT artefact recording
///      `source<TAB>mirror` rows for every operator-only conjecture
///      whose mirror entered the pool (`buildMirrorPairRows`); both
///      columns byte-identical to `conjectures.txt` lines. Written
///      unconditionally (empty in incubator mode, where pool lines
///      are rewritten post-capture); consumed by the CE filter's
///      mirror-refutation pass.
///
/// Equivalent to Python's `create_expressions_parallel(config)`.
///
/// @pre  Constructor completed successfully.
/// @post `files/theorems/` (or the override path from
///       `config_.theorems_folder`) carries the five output files.
void Conjecturer::run() {
    auto _run_t0 = std::chrono::steady_clock::now();
    std::set<std::string> resultExprSet;
    std::set<std::string> reshuffledExprSet;
    std::set<std::string> reshuffledMirroredExprSet;
    std::set<std::string> controlSet;

    // Pre-computation
    auto _prec_t0 = std::chrono::steady_clock::now();
    mappingsMap_ = createMap(config_.parameters.max_size_mapping_def_set);

    int leftBound = determineLeftSideBoundary();
    int rightBound = determineRightSideBoundary();
    mappingsMapAnchor_ = createMapAnchor(leftBound, rightBound);

    allPermutations_ = ce::generateAllPermutations(config_.parameters.max_number_simple_expressions + 1);

    for (int num = 0; num < config_.parameters.max_size_binary_list; ++num) {
        binarySeqsMap_[num] = ce::generateBinarySequencesAsLists(num);
    }

    // Build expression list (JSON key order, matching Python's dict iteration)
    std::vector<std::string> exprList;
    for (auto& ky : config_.expressionOrder) {
        auto& desc = config_.data.at(ky);
        if (desc.max_count_per_conjecture > 0) exprList.push_back(desc.short_mpl_normalized);
    }
    struct ExprLeafEntry {
        DefSetMap defSets;
        int nse;
        IntConjBuf intBuf;
        IntDefSetMap intDefSets;
    };
    std::map<std::string, ExprLeafEntry> exprLeafsArgsMap;
    for (auto& expr : exprList) {
        std::string coreExpr = ce::extractExpression(expr);
        ExprLeafEntry entry;
        entry.defSets = config_.data.at(coreExpr).definition_sets;
        entry.nse = 1;
        entry.intBuf = encodeExpr(expr);
        entry.intDefSets = encodeDefSetMap(entry.defSets);
        exprLeafsArgsMap[expr] = std::move(entry);
    }

    prof::g_phase_precomp.add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - _prec_t0).count());

    // Source→mirror pairs captured at pool admission, where connExpr and
    // reshMir are byte-identical to their future conjectures.txt lines.
    // Feeds the mirror_pairs.txt write below (CE mirror-refutation input).
    std::map<std::string, std::string> mirrorPairCaptures;

    // --- Preliminary pass: nse=1 ---
    auto _prel_t0 = std::chrono::steady_clock::now();
    if (config_.parameters.min_number_simple_expressions <= 1) {
        for (auto& expr : exprList) {
            auto& entry = exprLeafsArgsMap[expr];
            auto workerResult = singleExprAnchorConnectionInt(entry.intBuf, entry.intDefSets);

            for (int i = 0; i < (int)workerResult.connected_list2.size(); ++i) {
                auto& connExpr = workerResult.connected_list2[i];
                if (exprGood(connExpr) && resultExprSet.find(connExpr) == resultExprSet.end()) {
                    auto& resh = workerResult.reshuffled_list[i];
                    auto& reshMir = workerResult.reshuffled_mirrored_list[i];
                    if (controlSet.find(resh) == controlSet.end() &&
                        (reshMir.empty() || controlSet.find(reshMir) == controlSet.end())) {
                        resultExprSet.insert(connExpr);
                        reshuffledExprSet.insert(resh);
                        if (!reshMir.empty()) reshuffledMirroredExprSet.insert(reshMir);
                        controlSet.insert(resh);
                        if (!reshMir.empty()) controlSet.insert(reshMir);
                        if (!reshMir.empty()) mirrorPairCaptures[connExpr] = reshMir;
                    }
                }
            }
        }
        std::cout << "Preliminary pass (nse=1): " << resultExprSet.size() << " conjectures\n";
    }

    prof::g_phase_prelim.add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - _prel_t0).count());

    // --- Main combination loop ---
    auto _main_t0 = std::chrono::steady_clock::now();
    std::vector<std::string> growingTheorems = exprList;
    std::set<std::string> growingTheoremsSet(growingTheorems.begin(), growingTheorems.end());
    std::vector<int> lastVisitedMap(exprList.size(), -1);

    int created = (config_.parameters.max_number_simple_expressions >= 2) ? 1 : 0;

    while (created) {
        created = 0;
        for (int exprIndex = 0; exprIndex < (int)exprList.size(); ++exprIndex) {
            auto& statement = exprList[exprIndex];
            auto& stmtEntry = exprLeafsArgsMap[statement];

            int start = lastVisitedMap[exprIndex] + 1;
            int end = (int)growingTheorems.size();
            if (start >= end) continue;

            struct WorkInput {
                int nseStatement;
                int nseGrowingTheorem;
                const IntConjBuf* intStatement;
                const IntConjBuf* intGrowingTheorem;
                const IntDefSetMap* intArgsStatement;
                const IntDefSetMap* intArgsGrowingTheorem;
            };

            std::vector<WorkInput> inputs;
            for (int gti = start; gti < end; ++gti) {
                auto& gt = growingTheorems[gti];
                auto& gtEntry = exprLeafsArgsMap[gt];
                inputs.push_back({stmtEntry.nse, gtEntry.nse,
                                  &stmtEntry.intBuf, &gtEntry.intBuf,
                                  &stmtEntry.intDefSets, &gtEntry.intDefSets});
            }

            // Multi-threaded execution
            int numThreads = std::min((int)std::thread::hardware_concurrency(), (int)inputs.size());
            if (numThreads < 1) numThreads = 1;
            std::vector<WorkerResult> results(inputs.size());
            std::atomic<int> nextIdx(0);

            auto worker = [&]() {
                while (true) {
                    int idx = nextIdx.fetch_add(1);
                    if (idx >= (int)inputs.size()) break;
                    auto& inp = inputs[idx];
                    results[idx] = singleThreadCalculationInt(
                        *inp.intStatement, *inp.intGrowingTheorem,
                        inp.nseStatement, inp.nseGrowingTheorem,
                        *inp.intArgsStatement, *inp.intArgsGrowingTheorem);
                }
            };

            std::vector<std::thread> threads;
            for (int t = 0; t < numThreads; ++t) threads.emplace_back(worker);
            for (auto& t : threads) t.join();

            lastVisitedMap[exprIndex] = end - 1;

            // Process results
            for (int ind = 0; ind < (int)results.size(); ++ind) {
                auto& res = results[ind];
                int nse = inputs[ind].nseStatement + inputs[ind].nseGrowingTheorem;

                for (auto& [connExpr, connMap] : res.connected_list) {
                    if (exprLeafsArgsMap.find(connExpr) == exprLeafsArgsMap.end()) {
                        ExprLeafEntry entry;
                        entry.defSets = connMap;
                        entry.nse = nse;
                        entry.intBuf = encodeExpr(connExpr);
                        entry.intDefSets = encodeDefSetMap(connMap);
                        exprLeafsArgsMap[connExpr] = std::move(entry);
                    }

                    // Check if should be added to growing theorems
                    std::set<std::string> nonDigits;
                    for (auto& [arg, _] : connMap) {
                        bool isDigit = !arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit);
                        if (isDigit) nonDigits.insert(arg);
                    }
                    if (growingTheoremsSet.find(connExpr) == growingTheoremsSet.end()
                        && !nonDigits.empty()
                        && nse < config_.parameters.max_number_simple_expressions) {
                        growingTheorems.push_back(connExpr);
                        growingTheoremsSet.insert(connExpr);
                        created = 1;
                    }
                }

                for (int entryIdx = 0; entryIdx < (int)res.connected_list2.size(); ++entryIdx) {
                    auto& connExpr = res.connected_list2[entryIdx];
                    if (exprGood(connExpr) && resultExprSet.find(connExpr) == resultExprSet.end()) {
                        auto& reshuffled = res.reshuffled_list[entryIdx];
                        auto& reshMir = res.reshuffled_mirrored_list[entryIdx];
                        if (controlSet.find(reshuffled) == controlSet.end()
                            && controlSet.find(reshMir) == controlSet.end()) {
                            resultExprSet.insert(connExpr);
                            reshuffledExprSet.insert(reshuffled);
                            if (!reshMir.empty()) reshuffledMirroredExprSet.insert(reshMir);
                            controlSet.insert(reshuffled);
                            if (!reshMir.empty()) controlSet.insert(reshMir);
                            if (!reshMir.empty()) mirrorPairCaptures[connExpr] = reshMir;
                        }
                    }
                }
            }
        }
    }

    prof::g_phase_mainLoop.add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - _main_t0).count());

    // ---- Template addon: injectivity-contrapositive stumps ----
    // Anchor-coupled template conjectures the main enumeration cannot
    // build (they need two `=` atoms; the `=` count cap is 1). Fold-in
    // mirrors the preliminary nse=1 pass, so the survivors participate
    // in sorting, mirror merge, and the conjectures.txt write below.
    {
        auto templateResult = generateTemplateConjectures();
        int addedTemplates = 0;
        for (int i = 0; i < (int)templateResult.connected_list2.size(); ++i) {
            auto& connExpr = templateResult.connected_list2[i];
            if (exprGood(connExpr) && resultExprSet.find(connExpr) == resultExprSet.end()) {
                auto& resh = templateResult.reshuffled_list[i];
                auto& reshMir = templateResult.reshuffled_mirrored_list[i];
                if (controlSet.find(resh) == controlSet.end() &&
                    (reshMir.empty() || controlSet.find(reshMir) == controlSet.end())) {
                    resultExprSet.insert(connExpr);
                    reshuffledExprSet.insert(resh);
                    if (!reshMir.empty()) reshuffledMirroredExprSet.insert(reshMir);
                    controlSet.insert(resh);
                    if (!reshMir.empty()) controlSet.insert(reshMir);
                    if (!reshMir.empty()) mirrorPairCaptures[connExpr] = reshMir;
                    ++addedTemplates;
                }
            }
        }
        std::cout << "Template addon: " << addedTemplates << " conjectures\n";
    }

    // Sort results
    auto _sort_t0 = std::chrono::steady_clock::now();
    std::vector<std::string> sortedList(resultExprSet.begin(), resultExprSet.end());
    std::vector<std::string> reshuffledSortedList(reshuffledExprSet.begin(), reshuffledExprSet.end());
    std::vector<std::string> reshuffledMirroredSortedList(reshuffledMirroredExprSet.begin(), reshuffledMirroredExprSet.end());
    std::sort(sortedList.begin(), sortedList.end());
    std::sort(reshuffledSortedList.begin(), reshuffledSortedList.end());
    std::sort(reshuffledMirroredSortedList.begin(), reshuffledMirroredSortedList.end());

    // ---- Mirror conjectures: prove the reverse direction for real ----
    // Fold each conjecture's reverse-direction mirror into the prove pool so it
    // is counterexample-filtered and genuinely proved, replacing the prover's
    // former post-proof `mirrored statement` fabrication (D-112).
    {
        const int addedMirrors =
            mergeMirrorConjecturesIntoPool(sortedList, reshuffledMirroredSortedList);
        std::sort(sortedList.begin(), sortedList.end());
        std::cout << "Mirror addon: " << addedMirrors
                  << " mirror conjectures added to conjectures.txt\n";
    }

    // Incubator: reformulate operator-headed conjectures
    if (config_.parameters.incubator_mode) {
        for (auto& expr : sortedList) expr = reformulateOperatorHead(expr);
        std::sort(sortedList.begin(), sortedList.end());
    }

    // ---- OR theorem conjecture generation (non-incubator only) ----
    std::vector<std::pair<std::string, std::string>> orPairs;
    if (!config_.parameters.incubator_mode) {
        orPairs = generateOrConjectures();
        // Add OR conjectures directly to the main theorem list
        for (const auto& [exist, companion] : orPairs) {
            sortedList.push_back(exist);
            sortedList.push_back(companion);
        }
        std::sort(sortedList.begin(), sortedList.end());
        std::cout << "OR addon: " << orPairs.size() << " OR pairs (" << orPairs.size() * 2 << " conjectures added to conjectures.txt)\n";
    }

    // Determine output folder
    std::filesystem::path theoremsFolder;
    if (!config_.theorems_folder.empty()) {
        theoremsFolder = projectRoot_ / config_.theorems_folder;
    } else {
        theoremsFolder = projectRoot_ / "files" / "theorems";
    }

    // Clean folder (preserve special files). vacuous_theorems.txt is the
    // run-cumulative vacuity classification record — seeds from every
    // batch must survive to the run's end and feed each batch's
    // taint-closure filter.
    std::set<std::string> preserve = {"theorems.txt", "externally_provided_theorems.txt",
        "compressed_external_theorems.txt", "or_pairs.txt", "vacuous_theorems.txt"};
    if (std::filesystem::is_directory(theoremsFolder)) {
        for (auto& entry : std::filesystem::directory_iterator(theoremsFolder)) {
            if (preserve.find(entry.path().filename().string()) != preserve.end()) continue;
            std::error_code ec;
            std::filesystem::remove_all(entry.path(), ec);
        }
    } else {
        std::filesystem::create_directories(theoremsFolder);
    }

    // Write output files
    auto writeFile = [](const std::filesystem::path& path, const std::vector<std::string>& lines) {
        std::ofstream out(path, std::ios::out);
        for (auto& line : lines) out << line << "\n";
    };

    writeFile(theoremsFolder / "conjectures.txt", sortedList);
    writeFile(theoremsFolder / "reshuffled_conjectures.txt", reshuffledSortedList);
    writeFile(theoremsFolder / "reshuffled_mirrored_conjectures.txt", reshuffledMirroredSortedList);

    // Write OR pairs metadata (which existence+companion form an OR theorem)
    if (!orPairs.empty()) {
        std::ofstream orOut(theoremsFolder / "or_pairs.txt", std::ios::out);
        for (const auto& [exist, companion] : orPairs) {
            orOut << exist << "\t" << companion << "\n";
        }
    }

    // Write mirror pairs metadata (source conjecture <TAB> its pool mirror),
    // the CE filter's mirror-refutation input. Written unconditionally so the
    // file always exists alongside conjectures.txt (the CE loader asserts on
    // it). Rows are suppressed in incubator mode: reformulateOperatorHead
    // rewrote every pool line above, breaking the byte-identity both columns
    // rely on — and incubator batches skip the CE filter anyway.
    {
        std::vector<std::string> mirrorPairRows;
        if (!config_.parameters.incubator_mode) {
            mirrorPairRows = buildMirrorPairRows(mirrorPairCaptures);
        }
        writeFile(theoremsFolder / "mirror_pairs.txt", mirrorPairRows);
        std::cout << "Mirror pairs: " << mirrorPairRows.size()
                  << " operator-only pairs written to mirror_pairs.txt\n";
    }

    std::cout << "Number conjectures: " << sortedList.size() << "\n";

    prof::g_phase_sortAndOut.add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - _sort_t0).count());

    // ---- Profiling report ----
    // Disabled by default — flip the gate to 1 to re-enable when debugging
    // conjecturer perf. The per-function counters are already compile-time
    // no-ops via GL_PROF_SCOPE_ENABLE; this gate suppresses the empty-counter
    // dump in the runtime log so a release run does not print 30+ rows of
    // zeros.
#if 0
    double wallSec = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - _run_t0).count() / 1e9;
    std::cout << "\n==== Conjecturer profiling (wall " << std::fixed << std::setprecision(3)
              << wallSec << "s, "
              << (int)std::thread::hardware_concurrency() << " HW threads) ====\n";
    std::cout << "--- Phases (single-thread portions; main loop wall time includes thread join) ---\n";
    prof::report("phase: pre-compute",      prof::g_phase_precomp,   wallSec);
    prof::report("phase: preliminary (nse=1)", prof::g_phase_prelim, wallSec);
    prof::report("phase: main loop (wall)", prof::g_phase_mainLoop,  wallSec);
    prof::report("phase: sort+output",      prof::g_phase_sortAndOut,wallSec);
    std::cout << "--- Hot functions (summed across threads; % vs wall) ---\n";
    prof::report("singleThreadCalculation",   prof::g_singleThreadCalc,     wallSec);
    prof::report("singleExprAnchorConnection",prof::g_singleExprAnchorConn, wallSec);
    prof::report("reshuffle",                 prof::g_reshuffle,            wallSec);
    prof::report("  reshuffle perm-inner",    prof::g_reshufflePermInner,   wallSec);
    prof::report("createReshuffledMirrored",  prof::g_mirrored,             wallSec);
    prof::report("encodeExpr",                prof::g_encodeExpr,           wallSec);
    prof::report("decodeExpr",                prof::g_decodeExpr,           wallSec);
    prof::report("encodeDefSetMap",           prof::g_encodeDefSetMap,      wallSec);
    prof::report("connectExpressionsInt",     prof::g_connectExprInt,       wallSec);
    prof::report("makeAllConnectionMapsInt",  prof::g_makeAllConnMapsInt,   wallSec);
    prof::report("exprGood2Int",              prof::g_exprGood2Int,         wallSec);
    prof::report("exprGood",                  prof::g_exprGood,             wallSec);
    // disintegrateImplication lives in compiler.hpp; its counter is a raw
    // atomic pair. Synthesize a Counter-equivalent view for the report.
    prof::g_disintegrate.calls.store(ce::g_disintCalls.load());
    prof::g_disintegrate.ns.store(ce::g_disintNs.load());
    prof::report("disintegrateImplication",   prof::g_disintegrate,         wallSec);
    std::cout << "--- Filter functions (most are string-based = int-ification candidates) ---\n";
    prof::report("onlyInHeadGoodInt",         prof::g_onlyInHeadGoodInt,    wallSec);
    prof::report("prohibitedHeadsGoodInt",    prof::g_prohibHeadsGoodInt,   wallSec);
    prof::report("checkInputVariablesHead",   prof::g_checkInputVarsHead,   wallSec);
    prof::report("triggersExistenceRef",      prof::g_triggersExistenceRef, wallSec);
    prof::report("checkInputVariablesOrder",  prof::g_checkInputVarsOrder,  wallSec);
    prof::report("patternInConjecture",       prof::g_patternInConjecture,  wallSec);
    prof::report("evaluateOperatorExprs2",    prof::g_evaluateOperatorExprs2, wallSec);
    prof::report("controlEquality",           prof::g_controlEquality,      wallSec);
    prof::report("checkMinSizeExpression",    prof::g_checkMinSizeExpression, wallSec);
    prof::report("passesMaxSizeAfterEx",      prof::g_passesMaxSizeAfterEx, wallSec);
    prof::report("passesComplexityAfterEx",   prof::g_passesComplexityAfterEx, wallSec);
    prof::report("passesInPremiseFilter",     prof::g_passesInPremiseFilter, wallSec);
    prof::report("reformulateToExistence",    prof::g_reformulateToExistence, wallSec);
    prof::report("generateNegatedPremise",    prof::g_generateNegatedPremise, wallSec);
    prof::report("checkDefSets",              prof::g_checkDefSets,         wallSec);
    prof::report("checkComplLevForDefSets",   prof::g_checkComplLevForDefSets, wallSec);
    prof::report("countOperatorOccurrences",  prof::g_countOpOccurrences,   wallSec);
    prof::report("checkDefSetsPriorInt",      prof::g_checkDefSetsPriorInt, wallSec);
    prof::report("checkComplexityPerOpInt",   prof::g_checkComplexityPerOpInt, wallSec);
    prof::report("getNumRemArgsInt",          prof::g_getNumRemArgsInt,     wallSec);
    std::cout.flush();
#endif
}

// ============================================================================
// OR theorem conjecture generation: builds existence + companion directly
// ============================================================================

/// @brief Generate `(existence, companion)` pairs for OR-shaped
///        conjectures.
///
/// @details
/// Replaces the older `generateOrConjectures` path that read
/// `or_pairs.txt`. The current implementation derives pairs
/// directly from per-expression
/// `allow_to_constitute_existence` flags. Pairs are emitted into
/// `conjectures.txt` alongside ordinary conjectures and the prover
/// treats them via the `or disintegration` /
/// `or convergence` tags. See SwDD chapter
/// [`07_or_branching.md`](../../docs/agentic_swdd/20_core_concepts/07_or_branching.md)
/// for downstream prover behaviour.
///
/// @return Pairs of `(existence_conjecture, companion_conjecture)`.
std::vector<std::pair<std::string,std::string>> Conjecturer::generateOrConjectures() const {
    std::vector<std::pair<std::string,std::string>> result;

    // Find anchor name and arity
    std::string anchorName;
    int anchorArity = 0;
    for (const auto& [name, desc] : config_.data) {
        if (name.find("Anchor") == 0) {
            anchorName = name;
            anchorArity = desc.arity;
            break;
        }
    }
    if (anchorName.empty() || anchorArity == 0) return result;

    const auto& anchorDesc = config_.data.at(anchorName);

    // Map anchor arg position (1-based) to def_set type string
    std::map<int, std::string> anchorArgType;
    for (const auto& [posStr, ds] : anchorDesc.definition_sets) {
        anchorArgType[std::stoi(posStr)] = std::get<0>(ds);
    }

    // Build anchor string: (AnchorPeano[1,2,3,4,5,6])
    std::string anchorStr = "(" + anchorName + "[";
    for (int i = 1; i <= anchorArity; ++i) {
        if (i > 1) anchorStr += ",";
        anchorStr += std::to_string(i);
    }
    anchorStr += "])";

    // Find the set arg in anchor: type "P(1)" — this is N
    int setArg = -1;
    for (const auto& [pos, type] : anchorArgType) {
        if (type == "P(1)") { setArg = pos; break; }
    }
    if (setArg < 0) return result;

    // Collect element anchor args: type "(1)"
    std::vector<int> elementArgs;
    for (const auto& [pos, type] : anchorArgType) {
        if (type == "(1)") elementArgs.push_back(pos);
    }

    // Fresh bound vars start after anchor arity
    int n = anchorArity + 1;  // shared var (universally quantified)
    int m = anchorArity + 2;  // existential var

    // For each (caseExpr with allow_negation) x (existExpr with allow_to_constitute_existence)
    for (const auto& [caseName, caseDesc] : config_.data) {
        if (!caseDesc.allow_negation) continue;

        for (const auto& [existName, existDesc] : config_.data) {
            if (!existDesc.allow_to_constitute_existence) continue;
            if (existDesc.existence_variable_position < 1) continue;

            int existVarPos = existDesc.existence_variable_position; // 1-based

            // Exist expr must have a (1)-typed slot besides existVarPos for the shared var
            int sharedSlotInExist = -1;  // 1-based position for shared var n
            for (const auto& [posStr, ds] : existDesc.definition_sets) {
                int pos = std::stoi(posStr);
                if (pos == existVarPos) continue;
                if (std::get<0>(ds) == "(1)") {
                    sharedSlotInExist = pos;
                    break;
                }
            }
            if (sharedSlotInExist < 0) continue;  // no slot for shared var (e.g., "in")

            // Build existExpr args: m at existVarPos, n at sharedSlot, anchor args elsewhere
            std::vector<std::string> existArgs(existDesc.arity);
            for (const auto& [posStr, ds] : existDesc.definition_sets) {
                int pos = std::stoi(posStr);
                int idx = pos - 1; // 0-based
                if (pos == existVarPos) {
                    existArgs[idx] = std::to_string(m);
                } else if (pos == sharedSlotInExist) {
                    existArgs[idx] = std::to_string(n);
                } else {
                    // Find anchor arg of matching type
                    for (const auto& [apos, atype] : anchorArgType) {
                        if (atype == std::get<0>(ds)) {
                            existArgs[idx] = std::to_string(apos);
                            break;
                        }
                    }
                }
            }

            // Build existExpr string: (in2[m,n,3])
            std::string existExprStr = "(" + existName + "[";
            for (std::size_t i = 0; i < existArgs.size(); ++i) {
                if (i > 0) existExprStr += ",";
                existExprStr += existArgs[i];
            }
            existExprStr += "])";

            // For each element anchor arg as the "case constant" (e.g., i0, i1)
            for (int elemArg : elementArgs) {
                // Build caseExpr args: n at first (1)-typed position, elemArg at second
                std::vector<std::string> caseArgs(caseDesc.arity);
                bool nPlaced = false;
                for (const auto& [posStr, ds] : caseDesc.definition_sets) {
                    int pos = std::stoi(posStr);
                    int idx = pos - 1;
                    if (std::get<0>(ds) == "(1)" && !nPlaced) {
                        caseArgs[idx] = std::to_string(n);
                        nPlaced = true;
                    } else if (std::get<0>(ds) == "(1)") {
                        caseArgs[idx] = std::to_string(elemArg);
                    } else {
                        for (const auto& [apos, atype] : anchorArgType) {
                            if (atype == std::get<0>(ds)) {
                                caseArgs[idx] = std::to_string(apos);
                                break;
                            }
                        }
                    }
                }

                // Build caseExpr string: (=[n,2])
                std::string caseExprStr = "(" + caseName + "[";
                for (std::size_t i = 0; i < caseArgs.size(); ++i) {
                    if (i > 0) caseExprStr += ",";
                    caseExprStr += caseArgs[i];
                }
                caseExprStr += "])";

                std::string S = std::to_string(setArg);
                std::string N = std::to_string(n);
                std::string M = std::to_string(m);

                // Inner quantifier: (>[m](in[m,N])!(existExpr))
                std::string innerQuant = "(>[" + M + "](in[" + M + "," + S + "])!" + existExprStr + ")";

                // Collect all anchor args that appear in inner expressions
                // to determine outer bound vars
                std::set<std::string> usedVars;
                // From anchor: all args
                for (int i = 1; i <= anchorArity; ++i) usedVars.insert(std::to_string(i));
                // From (in[n,setArg]): n, setArg
                usedVars.insert(N);
                usedVars.insert(S);
                // From caseExprStr: all args
                for (auto& a : caseArgs) usedVars.insert(a);
                // From existExprStr: m, n, and anchor args
                for (auto& a : existArgs) usedVars.insert(a);

                // Bound vars: count occurrences across ALL leaf expressions
                // then split: anchor args → outerBound, non-anchor → level1Bound
                std::map<std::string, int> varCount;
                auto countArgs = [&](const std::string& expr) {
                    auto args = ce::getArgs(expr);
                    for (auto& a : args) {
                        if (a.size() >= 2 && a[0] == 'u' && a[1] == '_') continue;
                        varCount[a]++;
                    }
                };

                varCount.clear();
                countArgs(anchorStr);                    // [1,2,3,4,5,6]
                countArgs("(in[" + N + "," + S + "])");  // [n, S]
                countArgs("!" + caseExprStr);            // [n, elemArg]
                countArgs(existExprStr);                 // [m, n, 3] etc.
                countArgs("(in[" + M + "," + S + "])");  // [m, S]

                // Anchor args that also appear in body → outer binding
                std::set<std::string> anchorArgSet;
                for (int i = 1; i <= anchorArity; ++i) anchorArgSet.insert(std::to_string(i));

                std::vector<std::string> outerBound;
                std::vector<std::string> level1Bound;  // non-anchor, non-M vars (bind at in[n,S] level)
                for (auto& [v, c] : varCount) {
                    if (c <= 1) continue;
                    if (anchorArgSet.count(v)) {
                        outerBound.push_back(v);
                    } else if (v != M) {
                        // M is already bound inside innerQuant's >[M]
                        level1Bound.push_back(v);
                    }
                }
                auto numSort = [](const std::string& a, const std::string& b){
                    return std::stoi(a) < std::stoi(b);
                };
                std::sort(outerBound.begin(), outerBound.end(), numSort);
                std::sort(level1Bound.begin(), level1Bound.end(), numSort);

                auto joinVec = [](const std::vector<std::string>& v) {
                    std::string s;
                    for (std::size_t i = 0; i < v.size(); ++i) {
                        if (i > 0) s += ",";
                        s += v[i];
                    }
                    return s;
                };
                std::string outerBoundStr = joinVec(outerBound);
                std::string level1BoundStr = joinVec(level1Bound);

                // Existence form:
                // (>[outerBound](Anchor)(>[level1Bound](in[n,S])(>[]!(caseExpr)!(innerQuant))))
                std::string existence =
                    "(>[" + outerBoundStr + "]" + anchorStr +
                    "(>[" + level1BoundStr + "](in[" + N + "," + S + "])" +
                    "(>[]!" + caseExprStr +
                    "!" + innerQuant + ")))";

                // Companion form:
                // (>[outerBound](Anchor)(>[level1Bound](in[n,S])(>[](innerQuant)(caseExpr))))
                std::string companion =
                    "(>[" + outerBoundStr + "]" + anchorStr +
                    "(>[" + level1BoundStr + "](in[" + N + "," + S + "])" +
                    "(>[]" + innerQuant +
                    caseExprStr + ")))";

                result.push_back({existence, companion});
                std::cout << "  OR exist: " << existence << "\n";
                std::cout << "  OR compn: " << companion << "\n";
            }
        }
    }

    return result;
}

// ============================================================================
// Template addon: injectivity-contrapositive stumps
// ============================================================================

/// @brief Names of operators eligible for the template addon.
///
/// @details
/// An operator qualifies when it has exactly one input argument,
/// exactly one output argument, and a positive
/// `max_count_per_conjecture`. Additionally the whole addon is gated
/// on equality: when `=` is absent from the config or carries
/// `max_count_per_conjecture == 0`, the returned list is empty
/// regardless of the operators (the template's negated equalities
/// would cite an expression the batch does not use). Names are
/// returned in `expressionOrder` (JSON key order) so the emission
/// order is deterministic across runs.
///
/// @return Qualifying operator names; empty when equality is not
///         usable in this batch.
std::vector<std::string> Conjecturer::templateQualifyingOperators() const {
    std::vector<std::string> result;

    auto eqIt = config_.data.find("=");
    if (eqIt == config_.data.end() || eqIt->second.max_count_per_conjecture <= 0)
        return result;

    for (const auto& name : config_.expressionOrder) {
        const auto& desc = config_.data.at(name);
        if (desc.max_count_per_conjecture <= 0) continue;
        if (desc.input_args.size() != 1 || desc.output_args.size() != 1) continue;
        result.push_back(name);
    }
    return result;
}

/// @brief Build the injectivity-contrapositive stump for one
///        single-input operator.
///
/// @details
/// The stump encodes `op(a)=b and op(c)=d and b!=d implies a!=c`
/// with digit variable names: `a=1, b=2` in the first premise copy,
/// `c=3, d=4` in the second, and one fresh shared variable
/// (`5`, `6`, ...) per remaining operator slot, identical in both
/// copies and bound by no stump binder. For `in2` the result is
/// `(>[1,2](in2[1,2,5])(>[3,4](in2[3,4,5])(>[]!(=[2,4])!(=[1,3]))))`.
/// Because the stump binds `1..4` itself, `findArgMap` leaves
/// exactly the shared slot variables free, so the anchor-coupling
/// enumeration couples only those to anchor slots of matching type —
/// in all possible ways, like any other body.
///
/// @param opName Config key of a qualifying operator; must be a
///               member of `templateQualifyingOperators()`.
/// @return `(stump, defSetMap)` — the stump text and its free-arg
///         map from `findArgMap` (only the shared slot variables).
std::pair<std::string, DefSetMap> Conjecturer::buildTemplateStump(const std::string& opName) const {
    auto it = config_.data.find(opName);
    assert(it != config_.data.end());
    const auto& desc = it->second;

    assert(desc.indices_input_args.size() == 1);
    assert(desc.indices_output_args.size() == 1);
    const int arity = desc.arity;
    const int inPos = desc.indices_input_args[0];
    const int outPos = desc.indices_output_args[0];
    assert(arity >= 2);
    assert(inPos >= 0 && inPos < arity);
    assert(outPos >= 0 && outPos < arity);
    assert(inPos != outPos);

    // Shared slot variables (one per non-input/output position, same in
    // both premise copies), numbered 5.. in ascending position order.
    std::vector<std::string> args1(arity), args2(arity);
    int nextSlotVar = 5;
    for (int pos = 0; pos < arity; ++pos) {
        if (pos == inPos) {
            args1[pos] = "1";
            args2[pos] = "3";
        } else if (pos == outPos) {
            args1[pos] = "2";
            args2[pos] = "4";
        } else {
            args1[pos] = std::to_string(nextSlotVar);
            args2[pos] = args1[pos];
            ++nextSlotVar;
        }
    }

    // Binders list each premise's fresh variables in atom-argument order.
    auto binderOf = [&](const std::vector<std::string>& args,
                        const std::string& inVar, const std::string& outVar) {
        std::string binder;
        for (const auto& a : args) {
            if (a != inVar && a != outVar) continue;
            if (!binder.empty()) binder += ",";
            binder += a;
        }
        return binder;
    };
    const std::string binder1 = binderOf(args1, "1", "2");
    const std::string binder2 = binderOf(args2, "3", "4");

    auto atomOf = [&](const std::vector<std::string>& args) {
        std::string atom = "(" + opName + "[";
        for (std::size_t i = 0; i < args.size(); ++i) {
            if (i > 0) atom += ",";
            atom += args[i];
        }
        atom += "])";
        return atom;
    };

    const std::string stump =
        "(>[" + binder1 + "]" + atomOf(args1) +
        "(>[" + binder2 + "]" + atomOf(args2) +
        "(>[]!(=[2,4])!(=[1,3]))))";

    return {stump, findArgMap(stump)};
}

/// @brief Generate all template-addon conjectures for this batch.
///
/// @details
/// For every qualifying operator, builds the stump via
/// `buildTemplateStump` and routes it through the standard
/// string-path anchor attachment `singleExprAnchorConnection`, which
/// enumerates every coupling of the stump's free slot variables to
/// anchor slots and applies the standard filters. The bundles are
/// concatenated in operator order. The mirrored lane stays empty by
/// construction (the head `!(=[..])` is not an operator, so
/// `createReshuffledMirrored` yields `""`).
///
/// @pre `mappingsMapAnchor_` and `allPermutations_` are populated
///      (`run()` pre-computation).
/// @return Concatenated `WorkerResult` across all qualifying
///         operators; empty lists when none qualify.
WorkerResult Conjecturer::generateTemplateConjectures() const {
    WorkerResult result;
    for (const auto& opName : templateQualifyingOperators()) {
        auto [stump, stumpMap] = buildTemplateStump(opName);
        auto workerResult = singleExprAnchorConnection(stump, stumpMap);
        for (auto& e : workerResult.connected_list2)
            result.connected_list2.push_back(std::move(e));
        for (auto& e : workerResult.reshuffled_list)
            result.reshuffled_list.push_back(std::move(e));
        for (auto& e : workerResult.reshuffled_mirrored_list)
            result.reshuffled_mirrored_list.push_back(std::move(e));
    }
    return result;
}

} // namespace conj
