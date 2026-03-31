
/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschraenkt)

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
#include <algorithm>
#include <numeric>
#include <cassert>
#include <regex>
#include <sstream>
#include <iterator>

namespace conj {

using json = nlohmann::json;

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
// ConfigurationData::getAnchorName
// ============================================================================

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

std::string extractBetweenBrackets(const std::string& s, size_t startIndex) {
    auto start = s.find('[', startIndex);
    if (start == std::string::npos) return "";
    auto end = s.find(']', start);
    if (end == std::string::npos) return "";
    return s.substr(start + 1, end - start - 1);
}

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

std::string replaceIntegerInString(const std::string& bigString, const std::string& targetInt, const std::string& replacementInt) {
    std::string pattern = "\\b" + targetInt + "\\b";
    std::regex re(pattern);
    return std::regex_replace(bigString, re, replacementInt);
}

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

int getNumberRemovableArgs(const std::map<std::string,std::string>& mapping) {
    std::map<std::string,std::string> subMap;
    for (auto& [k, v] : mapping) {
        if (v != k) subMap[k] = v;
    }
    std::set<std::string> vals;
    for (auto& [k, v] : subMap) vals.insert(v);
    return (int)vals.size();
}

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

        if (p.contains("max_values_for_def_sets") && p["max_values_for_def_sets"].is_object())
            for (auto& [k, v] : p["max_values_for_def_sets"].items())
                cp.max_values_for_def_sets[k] = v.get<int>();
        if (p.contains("max_values_for_uncomb_def_sets") && p["max_values_for_uncomb_def_sets"].is_object())
            for (auto& [k, v] : p["max_values_for_uncomb_def_sets"].items())
                cp.max_values_for_uncomb_def_sets[k] = v.get<int>();
        if (p.contains("max_values_for_def_sets_prior_connection") && p["max_values_for_def_sets_prior_connection"].is_object())
            for (auto& [k, v] : p["max_values_for_def_sets_prior_connection"].items())
                cp.max_values_for_def_sets_prior_connection[k] = v.get<int>();
        if (p.contains("max_complexity_if_anchor_parameter_connected") && p["max_complexity_if_anchor_parameter_connected"].is_object())
            for (auto& [k, v] : p["max_complexity_if_anchor_parameter_connected"].items())
                cp.max_complexity_if_anchor_parameter_connected[k] = v.get<int>();

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
            (fullMplRaw.size() >= 4 && fullMplRaw.substr(fullMplRaw.size()-4) == ".txt"
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
        desc.max_size_expression = spec.value("max_size_expression", 0);
        desc.min_size_expression = spec.value("min_size_expression", 1);

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
        ic.maxSizeExpr = (int16_t)desc.max_size_expression;
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
    for (auto& [ds, val] : config_.parameters.max_complexity_if_anchor_parameter_connected) {
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

IntConjBuf Conjecturer::encodeExpr(const std::string& expr) const {
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

std::string Conjecturer::decodeExpr(const IntConjBuf& buf) const {
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

IntDefSetMap Conjecturer::encodeDefSetMap(const DefSetMap& dsm) const {
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

std::tuple<bool, std::string, DefSetMap>
Conjecturer::connectExpressions(const std::string& expr1, const std::string& expr2,
                                const DefSetMap& map1, const DefSetMap& map2,
                                const std::map<std::string,std::string>& subMap,
                                const std::vector<int>& binaryList, bool connectToAnchor) const {
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

    std::string connectedExpr = "(>[" + ce::joinWithComma(argsToRemove) + "]"
        + newExpr1 + newExpr2 + ")";

    if (newExpr1 == newExpr2) success = false;

    return {success, connectedExpr, connectedMap};
}

// ============================================================================
// Int-path: connectExpressions
// ============================================================================

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

int Conjecturer::getNumberRemovableArgsInt(const IntConnMap& connMap) const {
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

bool Conjecturer::connectExpressionsInt(
    const IntConjBuf& expr1, const IntConjBuf& expr2,
    const IntDefSetMap& map1, const IntDefSetMap& map2,
    const IntConnMap& subMap,
    const int16_t* binaryList, int binaryLen,
    bool connectToAnchor,
    IntConjBuf& outExpr, IntDefSetMap& outMap) const
{
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
                // Merge outer bound vars + expr1's first block's bound vars
                outExpr.data[outExpr.len++] = (int16_t)(numToRemove + bc);
                for (int i = 0; i < numToRemove; ++i)
                    outExpr.data[outExpr.len++] = argsToRemove[i];
                for (int i = 0; i < bc; ++i) {
                    int16_t bv = expr1.data[pos++];
                    int16_t mapped = (bv <= subMap.maxArg && subMap.map[bv] != 0) ? subMap.map[bv] : bv;
                    outExpr.data[outExpr.len++] = mapped;
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

    return true;
}

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

void Conjecturer::makeAllConnectionMapsInt(
    const IntDefSetMap& argsMap1, const IntDefSetMap& argsMap2,
    bool withAnchor, const MappingsMap& mappingsMap,
    std::vector<IntConnMap>& outMaps) const
{
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

bool Conjecturer::checkDefSetsPriorInt(const IntDefSetMap& argsStmt, const IntDefSetMap& argsGT) const {
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

int Conjecturer::countOperatorOccurrencesInt(const IntConjBuf& buf) const {
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

bool Conjecturer::onlyInHeadGoodInt(const IntConjBuf& buf) const {
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

bool Conjecturer::prohibitedHeadsGoodInt(const IntConjBuf& buf) const {
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

bool Conjecturer::checkComplexityPerOpInt(const IntConjBuf& growingTheorem, const IntConjBuf& statement) const {
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

    // Check each expression's max_size_expression against chainLen
    pos = 0;
    while (pos < growingTheorem.len) {
        int16_t bc = growingTheorem.data[pos++]; pos += bc;
        int16_t nid = growingTheorem.data[pos++];
        int16_t ar = growingTheorem.data[pos++]; pos += ar;
        if (nid > 0 && nid < (int16_t)intExprConfigs_.size()) {
            if (intExprConfigs_[nid].maxSizeExpr > 0 && intExprConfigs_[nid].maxSizeExpr < chainLen)
                return false;
        }
    }
    pos = 0;
    while (pos < statement.len) {
        int16_t bc = statement.data[pos++]; pos += bc;
        int16_t nid = statement.data[pos++];
        int16_t ar = statement.data[pos++]; pos += ar;
        if (nid > 0 && nid < (int16_t)intExprConfigs_.size()) {
            if (intExprConfigs_[nid].maxSizeExpr > 0 && intExprConfigs_[nid].maxSizeExpr < chainLen)
                return false;
        }
    }
    return true;
}

bool Conjecturer::exprGood2Int(const IntConjBuf& buf, int nse, const IntDefSetMap& connectedMap) const {
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

bool Conjecturer::exprGood(const std::string& expr) const {
    if (expr.substr(0, 3) == "(>[" && expr.substr(0, 4) != "(>[]") {
        if (!repetitionsExist(expr)) return true;
    }
    return false;
}

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

bool Conjecturer::checkDefSets(const DefSetMap& argMap) const {
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

bool Conjecturer::checkComplexityLevelForDefSets(const DefSetMap& argMap, int complexityLevel) const {
    std::set<std::string> defSets;
    for (auto& [arg, tpl] : argMap) defSets.insert(std::get<0>(tpl));
    for (auto& ds : defSets) {
        auto it = config_.parameters.max_complexity_if_anchor_parameter_connected.find(ds);
        if (it != config_.parameters.max_complexity_if_anchor_parameter_connected.end() && it->second < complexityLevel)
            return false;
    }
    return true;
}

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

bool Conjecturer::patternInConjecture(const std::string& conjecture) const {
    for (auto& pat : config_.patterns_to_exclude) {
        if (std::regex_search(conjecture, pat)) return true;
    }
    return false;
}

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
            if (it->second.max_size_expression < (int)chain.size()) return false;
        }
    }
    return true;
}

bool Conjecturer::checkMinSizeExpression(const std::string& conjecture) const {
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

bool Conjecturer::controlEquality(const std::string& conjecture) const {
    std::regex pat(R"(\(=\[\d+,\d+\]\))");
    std::smatch m;
    bool result = true;
    if (std::regex_search(conjecture, m, pat)) {
        auto args = ce::getArgs(m.str());
        if (std::stoi(args[0]) > std::stoi(args[1])) result = false;
    }
    return result && countArgumentsFilter(conjecture);
}

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

bool Conjecturer::evaluateOperatorExprs2(const std::string& expression, bool anchorAttached) const {
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
            assert(it0 != argMap.end() && it1 != argMap.end());
            bool cond = ((!it0->second.second.empty() && !it1->second.second.empty() && it0->second.first.empty() && it1->second.first.empty()) ||
                         (!it0->second.first.empty() && !it1->second.first.empty() && it0->second.second.empty() && it1->second.second.empty()));
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

bool Conjecturer::checkInputVariablesTheoremOperatorHead(const std::string& theorem) const {
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

std::string Conjecturer::getOperatorId(const std::string& expr) const {
    std::string coreExpr = ce::extractExpression(expr);
    if (std::find(operators_.begin(), operators_.end(), coreExpr) == operators_.end()) return "";
    std::map<std::string,std::string> replMap;
    auto args = ce::getArgs(expr);
    for (int idx : config_.data.at(coreExpr).indices_input_args) replMap[args[idx]] = "";
    for (int idx : config_.data.at(coreExpr).indices_output_args) replMap[args[idx]] = "";
    return ce::replaceKeysInString(expr, replMap);
}

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

bool Conjecturer::checkTautology(const std::vector<std::string>& leftChain, const std::vector<std::string>& rightChain) const {
    return removeOutputs(leftChain) != removeOutputs(rightChain);
}

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

bool Conjecturer::checkInputVariablesOrder(const std::string& theorem) const {
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

int Conjecturer::countOperatorOccurrences(const std::string& s) const {
    int count = 0;
    size_t pos = 0;
    while ((pos = s.find("(>[", pos)) != std::string::npos) {
        count++;
        pos += 3;
    }
    return count;
}

bool Conjecturer::staysOutputVariable(const std::string& fullExpr, const std::string& outputVariable) const {
    std::string coreExpr = ce::extractExpression(fullExpr);
    auto args = ce::getArgs(fullExpr);
    if (std::find(operators_.begin(), operators_.end(), coreExpr) != operators_.end()) {
        if (args[config_.data.at(coreExpr).indices_output_args[0]] == outputVariable)
            return true;
    }
    return false;
}

std::tuple<std::string, DefSetMap, std::map<std::string,std::string>>
Conjecturer::reshuffle(const std::string& expr, bool deep) const {
    using ChainEntry = std::tuple<std::string, std::vector<std::string>, std::set<std::string>>;

    auto [minReshuffled, minArgMap, minReplacementMap] = renameVariablesInExpr(expr, deep);

    std::vector<ChainEntry> chainEntries;
    std::string head = ce::disintegrateImplication(expr, chainEntries, coreExprMap_);

    int chainLen = (int)chainEntries.size();
    auto permIt = allPermutations_.find(chainLen);
    if (permIt == allPermutations_.end()) return {minReshuffled, minArgMap, minReplacementMap};

    for (auto& permutation : permIt->second) {
        // create_last_occurrence_map
        std::set<std::string> allRemovedArgs;
        for (int i = 0; i < chainLen; ++i) {
            auto& removedArgs = std::get<1>(chainEntries[permutation[i]]);
            allRemovedArgs.insert(removedArgs.begin(), removedArgs.end());
        }

        std::map<std::string, int> lastOccurrenceMap;
        for (int ind = 0; ind < chainLen; ++ind) {
            auto& remainingArgs = std::get<2>(chainEntries[permutation[ind]]);
            for (auto& arg : allRemovedArgs) {
                if (remainingArgs.find(arg) != remainingArgs.end()) {
                    auto it = lastOccurrenceMap.find(arg);
                    if (it == lastOccurrenceMap.end()) {
                        lastOccurrenceMap[arg] = ind;
                    } else {
                        it->second = std::min(it->second, ind);
                    }
                }
            }
        }

        // build_removed_args_lists
        std::vector<std::vector<std::string>> removedArgsLists(chainLen);
        for (auto& [arg, ind] : lastOccurrenceMap) {
            removedArgsLists[ind].push_back(arg);
        }

        // make_reshuffled_expression
        std::string reshuffled = head;
        for (int ind = chainLen - 1; ind >= 0; --ind) {
            std::string substr = "[" + ce::joinWithComma(removedArgsLists[ind]) + "]";
            reshuffled = "(>" + substr + std::get<0>(chainEntries[permutation[ind]]) + reshuffled + ")";
        }

        auto [renamed, argMap, repMap] = renameVariablesInExpr(reshuffled, deep);
        if (renamed < minReshuffled) {
            minReshuffled = renamed;
            minArgMap = argMap;
            minReplacementMap = repMap;
        }
    }

    return {minReshuffled, minArgMap, minReplacementMap};
}

std::string Conjecturer::createReshuffledMirrored(const std::string& expr, bool anchorFirst) const {
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

// ============================================================================
// Int-path worker functions (Phase 5)
// ============================================================================

WorkerResult Conjecturer::singleThreadCalculationInt(
    const IntConjBuf& intStatement, const IntConjBuf& intGrowingTheorem,
    int nseStatement, int nseGrowingTheorem,
    const IntDefSetMap& intArgsStatement, const IntDefSetMap& intArgsGrowingTheorem) const
{
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

            // === BOUNDARY: decode to string for reshuffle ===
            std::string connectedExpr = decodeExpr(outExpr);
            auto [reshuffledExpr, reshuffledMap, repMap] = reshuffle(connectedExpr, true);
            result.connected_list.push_back({reshuffledExpr, reshuffledMap});

            // String guards (must match exactly — int guards can diverge)
            int complexityLevel = countOperatorOccurrences(reshuffledExpr) + 1;
            int numCombinableArgs = 0;
            for (auto& [arg, tpl] : reshuffledMap) if (std::get<1>(tpl)) numCombinableArgs++;

            if (checkDefSets(reshuffledMap) && numCombinableArgs <= config_.parameters.max_number_args_expr
                && checkComplexityLevelForDefSets(reshuffledMap, complexityLevel)) {

                // Re-encode for int-path anchor connection
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
                    if (!checkInputVariablesTheoremOperatorHead(connExpr2) ||
                        !checkInputVariablesOrder(connExpr2) ||
                        patternInConjecture(connExpr2)) continue;
                    if (!evaluateOperatorExprs2(connExpr2, true)) continue;
                    if (!controlEquality(connExpr2)) continue;
                    if (!checkMinSizeExpression(connExpr2)) continue;

                    result.connected_list2.push_back(connExpr2);
                    auto [reshExpr2, _m, __m] = reshuffle(connExpr2, true);
                    result.reshuffled_list.push_back(reshExpr2);
                    result.reshuffled_mirrored_list.push_back(createReshuffledMirrored(connExpr2));
                }
            }
        }
    }
    return result;
}

WorkerResult Conjecturer::singleExprAnchorConnectionInt(
    const IntConjBuf& intExpr, const IntDefSetMap& intExprDefSets) const
{
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

        result.connected_list2.push_back(connExpr2);
        auto [reshExpr2, reshMap2, repMap2] = reshuffle(connExpr2, true);
        result.reshuffled_list.push_back(reshExpr2);
        result.reshuffled_mirrored_list.push_back(createReshuffledMirrored(connExpr2));
    }

    return result;
}

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
// Constructor
// ============================================================================

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

void Conjecturer::run() {
    std::set<std::string> resultExprSet;
    std::set<std::string> reshuffledExprSet;
    std::set<std::string> reshuffledMirroredExprSet;
    std::set<std::string> controlSet;

    // Pre-computation
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

    // --- Preliminary pass: nse=1 ---
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
                    }
                }
            }
        }
        std::cout << "Preliminary pass (nse=1): " << resultExprSet.size() << " conjectures\n";
    }

    // --- Main combination loop ---
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
                        }
                    }
                }
            }
        }
    }

    // Sort results
    std::vector<std::string> sortedList(resultExprSet.begin(), resultExprSet.end());
    std::vector<std::string> reshuffledSortedList(reshuffledExprSet.begin(), reshuffledExprSet.end());
    std::vector<std::string> reshuffledMirroredSortedList(reshuffledMirroredExprSet.begin(), reshuffledMirroredExprSet.end());
    std::sort(sortedList.begin(), sortedList.end());
    std::sort(reshuffledSortedList.begin(), reshuffledSortedList.end());
    std::sort(reshuffledMirroredSortedList.begin(), reshuffledMirroredSortedList.end());

    // Incubator: reformulate operator-headed conjectures
    if (config_.parameters.incubator_mode) {
        for (auto& expr : sortedList) expr = reformulateOperatorHead(expr);
        std::sort(sortedList.begin(), sortedList.end());
    }

    // Determine output folder
    std::filesystem::path theoremsFolder;
    if (!config_.theorems_folder.empty()) {
        theoremsFolder = projectRoot_ / config_.theorems_folder;
    } else {
        theoremsFolder = projectRoot_ / "files" / "theorems";
    }

    // Clean folder (preserve special files)
    std::set<std::string> preserve = {"proved_theorems.txt", "externally_provided_theorems.txt", "compressed_external_theorems.txt"};
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

    writeFile(theoremsFolder / "theorems.txt", sortedList);
    writeFile(theoremsFolder / "reshuffled_theorems.txt", reshuffledSortedList);
    writeFile(theoremsFolder / "reshuffled_mirrored_theorems.txt", reshuffledMirroredSortedList);

    std::cout << "Number conjectures: " << resultExprSet.size() << "\n";
}

// ============================================================================
// Debug
// ============================================================================

void Conjecturer::dumpDebug(const std::string& tag, const std::string& data) const {
#ifdef CONJ_DEBUG_DUMP
    auto path = projectRoot_ / ".debug" / ("conj_dump_" + tag + ".txt");
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out << data;
#endif
}

} // namespace conj
