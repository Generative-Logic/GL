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

#include <string>
#include <vector>
#include <map>
#include <set>
#include <chrono>
#include <regex>
#include <cassert>
#include <tuple>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <unordered_set>
#include <unordered_map>
#include <cstring>
#include <utility>
#include <variant>
#include <filesystem>
#include <fstream>
#include <json.hpp>
#include "parameters.hpp"
#include <iostream>

// ============================================================================
// namespace ce — formerly create_expressions_shim.hpp
// ============================================================================

namespace ce {

// Returns all unique permutations of the input, in lexicographic order.
std::map<int, std::vector<std::vector<int>>> generateAllPermutations(int n);

// Binary sequences of given length (as lists of 0/1).
std::vector<std::vector<int>> generateBinarySequencesAsLists(int n);

using Mapping = std::map<int,int>;

inline const std::filesystem::path DEFINITIONS_FOLDER =
    std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() / "files" / "definitions";

struct CoreExpressionConfig {
    int arity;
    std::variant<std::string, std::filesystem::path> definition; // inline pattern OR file path
    std::string signature;

    // Slot -> (pattern, mandatory)
    std::map<std::string, std::pair<std::string, bool>> definitionSets;

    // New fields to match Python's logic
    std::vector<std::string> inputArgs;
    std::vector<std::string> outputArgs;

    // Indices for logic replication (calculated from signature)
    std::vector<int> inputIndices;
    std::vector<int> outputIndices;

    CoreExpressionConfig()
        : arity(0),
        definition(std::string()),
        signature(),
        definitionSets(),
        inputArgs(),
        outputArgs(),
        inputIndices(),
        outputIndices() {
    }

    CoreExpressionConfig(int arity_,
        const std::variant<std::string, std::filesystem::path>& definition_,
        const std::string& signature_)
        : arity(arity_),
        definition(definition_),
        signature(signature_),
        definitionSets(),
        inputArgs(),
        outputArgs(),
        inputIndices(),
        outputIndices() {
    }
};


struct AnchorInfo {
    std::string exampleExpression;
    int arity;
    std::map<std::string, std::string> definitionSets;
    std::string name;

    AnchorInfo()
        : exampleExpression(),
        arity(0),
        definitionSets(),
        name() {
    }

    AnchorInfo(const std::string& exampleExpression_,
        int arity_,
        const std::map<std::string, std::string>& definitionSets_,
        const std::string& name_)
        : exampleExpression(exampleExpression_),
        arity(arity_),
        definitionSets(definitionSets_),
        name(name_) {
    }
};

inline std::string
findAnchorKey(const std::map<std::string, ce::CoreExpressionConfig>& coreExpressionMap)
{
    for (const auto& kv : coreExpressionMap) {
        const std::string& k = kv.first;
        if (k.rfind("Anchor", 0) == 0) {
            return k;
        }
    }
    return "";
}

inline std::string makeAnchorSignature(const std::string& name, int arity)
{
    if (arity < 0) {
        throw std::invalid_argument("arity must be non-negative");
    }

    std::string sig;
    sig.reserve(1 + name.size() + 1 + arity * 2 + 2);

    sig.push_back('(');
    sig += name;
    sig.push_back('[');

    for (int i = 1; i <= arity; ++i) {
        if (i > 1) sig.push_back(',');
        sig += std::to_string(i);
    }

    sig += "])";
    return sig;
}


inline AnchorInfo initAnchor(const std::map<std::string, ce::CoreExpressionConfig>& coreExpressionMap, const std::string& anchorID) {

    std::string key = "Anchor" + anchorID;

    if (coreExpressionMap.find(key) == coreExpressionMap.end()) {
        assert(false && "The requested Anchor ID was not found in the configuration.");
    }

    std::string exampleExpression;
    int arity = 0;
    std::map<std::string, std::string> definitionSetsStr;

    auto it = coreExpressionMap.find(key);
    if (it != coreExpressionMap.end()) {
        exampleExpression = makeAnchorSignature(key, it->second.arity);
        arity = it->second.arity;

        for (const auto& kv : it->second.definitionSets) {
            definitionSetsStr[kv.first] = kv.second.first;
        }
    }
    return AnchorInfo(exampleExpression, arity, definitionSetsStr, key);
}


struct TreeNode1 {
    std::string value;
    TreeNode1* left;
    TreeNode1* right;
    std::set<std::string> arguments;

    TreeNode1()
        : value(),
        left(NULL),
        right(NULL),
        arguments() {
    }

    TreeNode1(const std::string& value_, int numberLeafs_)
        : value(value_),
        left(NULL),
        right(NULL),
        arguments() {
    }
};


inline std::string readTreeFromFile(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::in | std::ios::binary);
    if (!in) {
        return std::string();
    }
    std::string content;
    in.seekg(0, std::ios::end);
    std::streampos len = in.tellg();
    in.seekg(0, std::ios::beg);
    content.resize(static_cast<std::size_t>(len));
    if (len > 0) {
        in.read(&content[0], len);
    }

    std::string stripped;
    stripped.reserve(content.size());
    for (std::size_t i = 0; i < content.size(); ++i) {
        char c = content[i];
        if (c != '\n' && c != ' ' && c != '\t' && c != '\r') {
            stripped.push_back(c);
        }
    }
    return stripped;
}

// Forward declaration required for modifyCoreExpressionMap logic
inline std::vector<std::string> getArgs(const std::string& expr);


inline std::map<std::string, CoreExpressionConfig>
modifyCoreExpressionMap(const std::filesystem::path& configPath)
{
    using json = nlohmann::json;

    auto strip_ws = [](const std::string& s) -> std::string {
        std::string out; out.reserve(s.size());
        for (char c : s) {
            if (c != '\n' && c != ' ' && c != '\t' && c != '\r') out.push_back(c);
        }
        return out;
        };

    std::ifstream in(configPath);
    if (!in) {
        return {};
    }
    json j;
    in >> j;

    std::map<std::string, CoreExpressionConfig> resolved;
    const std::filesystem::path cfgDir = configPath.parent_path();

    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string name = it.key();
        const json& spec = it.value();

        if (!spec.is_object()) continue;
        if (!spec.contains("arity")) continue;

        const int arity = spec.value("arity", 0);

        std::string definition_text;
        std::string full_mpl_raw = spec.value("full_mpl", std::string{});
        const bool looks_like_file =
            (!full_mpl_raw.empty()) &&
            (full_mpl_raw.size() >= 4 &&
                full_mpl_raw.rfind(".txt") == full_mpl_raw.size() - 4 ||
                full_mpl_raw.find('/') != std::string::npos ||
                full_mpl_raw.find('\\') != std::string::npos);

        if (looks_like_file) {
            std::filesystem::path p(full_mpl_raw);

            std::vector<std::filesystem::path> candidates;
            if (p.is_absolute()) {
                candidates.push_back(p);
            }
            else {
                candidates.push_back(DEFINITIONS_FOLDER / p.filename());
                candidates.push_back(cfgDir / p);
                candidates.push_back(cfgDir / p.filename());
            }

            bool loaded = false;
            std::error_code ec;
            for (const auto& cand : candidates) {
                if (std::filesystem::is_regular_file(cand, ec) && !ec) {
                    definition_text = readTreeFromFile(cand);
                    loaded = true;
                    break;
                }
            }
            if (!loaded) {
                definition_text = strip_ws(full_mpl_raw);
            }
        }
        else {
            definition_text = strip_ws(full_mpl_raw);
        }

        std::string signature;
        if (spec.contains("short_mpl") && spec["short_mpl"].is_string()) {
            signature = strip_ws(spec["short_mpl"].get<std::string>());
        }
        else {
            signature.reserve(name.size() + static_cast<std::size_t>(3 * std::max(arity, 1)));
            signature += "("; signature += name; signature += "[";
            if (arity > 0) {
                for (int i = 1; i <= arity; ++i) {
                    if (i > 1) signature += ",";
                    signature += std::to_string(i);
                }
            }
            signature += "])";
        }

        CoreExpressionConfig cfg(
            arity,
            std::variant<std::string, std::filesystem::path>(definition_text),
            signature
        );

        if (spec.contains("definition_sets") && spec["definition_sets"].is_object()) {
            const json& ds = spec["definition_sets"];
            for (auto sit = ds.begin(); sit != ds.end(); ++sit) {
                const std::string slot = sit.key();
                const json& node = sit.value();

                std::string pattern;
                bool mandatory = false;

                if (node.is_array()) {
                    if (!node.empty() && node[0].is_string()) {
                        pattern = strip_ws(node[0].get<std::string>());
                    }
                    if (node.size() >= 2) {
                        if (node[1].is_boolean()) {
                            mandatory = node[1].get<bool>();
                        }
                        else if (node[1].is_string()) {
                            const std::string s = node[1].get<std::string>();
                            mandatory = (s == "true" || s == "True" || s == "1");
                        }
                        else if (node[1].is_number_integer()) {
                            mandatory = (node[1].get<int>() != 0);
                        }
                    }
                }
                else if (node.is_object()) {
                    if (node.contains("pattern") && node["pattern"].is_string())
                        pattern = strip_ws(node["pattern"].get<std::string>());
                    if (node.contains("mandatory")) {
                        if (node["mandatory"].is_boolean())
                            mandatory = node["mandatory"].get<bool>();
                        else if (node["mandatory"].is_string()) {
                            const std::string s = node["mandatory"].get<std::string>();
                            mandatory = (s == "true" || s == "True" || s == "1");
                        }
                        else if (node["mandatory"].is_number_integer()) {
                            mandatory = (node["mandatory"].get<int>() != 0);
                        }
                    }
                }
                else if (node.is_string()) {
                    pattern = strip_ws(node.get<std::string>());
                }

                if (!pattern.empty()) {
                    cfg.definitionSets[slot] = std::make_pair(pattern, mandatory);
                }
            }
        }

        if (spec.contains("input_args") && spec["input_args"].is_array()) {
            cfg.inputArgs = spec["input_args"].get<std::vector<std::string>>();
        }
        if (spec.contains("output_args") && spec["output_args"].is_array()) {
            cfg.outputArgs = spec["output_args"].get<std::vector<std::string>>();
        }

        std::vector<std::string> signatureParams = getArgs(cfg.signature);

        for (const auto& argName : cfg.inputArgs) {
            auto it = std::find(signatureParams.begin(), signatureParams.end(), argName);
            if (it != signatureParams.end()) {
                cfg.inputIndices.push_back(static_cast<int>(std::distance(signatureParams.begin(), it)));
            }
        }

        for (const auto& argName : cfg.outputArgs) {
            auto it = std::find(signatureParams.begin(), signatureParams.end(), argName);
            if (it != signatureParams.end()) {
                cfg.outputIndices.push_back(static_cast<int>(std::distance(signatureParams.begin(), it)));
            }
        }

        resolved.emplace(name, std::move(cfg));
    }

    return resolved;
}


inline std::map<std::string, CoreExpressionConfig>
modifyCoreExpressionMap(std::string anchorID)
{
    const auto configPath =
        std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
        / "files" / "config" / ("Config" + anchorID + ".json");

    return modifyCoreExpressionMap(configPath);
}


inline std::vector<std::string> getArgs(const std::string& expr) {
    std::vector<std::string> out;

    std::size_t start = expr.find('[', 0);
    if (start == std::string::npos) {
        return out;
    }
    std::size_t end = expr.find(']', start);
    if (end == std::string::npos) {
        return out;
    }

    std::size_t begin = start + 1;
    std::size_t len = end > begin ? (end - begin) : 0;
    std::string subExpr = expr.substr(begin, len);

    if (subExpr.empty()) {
        return out;
    }

    std::size_t pos = 0;
    while (pos <= subExpr.size()) {
        std::size_t comma = subExpr.find(',', pos);
        if (comma == std::string::npos) {
            out.push_back(subExpr.substr(pos));
            break;
        }
        out.push_back(subExpr.substr(pos, comma - pos));
        pos = comma + 1;
    }
    return out;
}

inline std::string extractExpression(const std::string& s) {
    std::size_t index = s.find('[');
    if (index != std::string::npos) {
        if (!s.empty() && s[0] == '(') {
            return s.substr(1, index - 1);
        }
        else {
            return s.substr(0, index);
        }
    }
    return std::string();
}

inline std::string extractExpressionUniversal(const std::string& s) {
    std::size_t index = s.find('[');
    if (index != std::string::npos) {
        if (!s.empty() && s[0] == '(') {
            return s.substr(1, index - 1);
        }
        else if (s.size() >= 2 && s[0] == '!' && s[1] == '(') {
            return s.substr(2, index - 2);
        }
    }
    return std::string();
}

inline std::string extractExpressionFromNegation(const std::string& s) {
    std::size_t startIndex = s.find("!(");
    std::size_t endIndex = s.find('[');
    if (startIndex != std::string::npos && endIndex != std::string::npos && startIndex < endIndex) {
        return s.substr(startIndex + 2, endIndex - (startIndex + 2));
    }
    return std::string();
}

TreeNode1* parseExpr(const std::string& treeStrIn);

inline void nodeToStr(const TreeNode1* node, std::string& out) {
    if (node == NULL) {
        return;
    }

    const std::string& v = node->value;

    if (!v.empty() && v[0] == '>') {
        out += "(";
        out += v;
    }
    else if (v == "&") {
        out += "(&";
    }
    else if (v.size() >= 2 && v[0] == '!' && v[1] == '>') {
        out += "!(";
        out += v.substr(1);
    }
    else if (v == "!&") {
        out += "!(&";
    }
    else if (!v.empty() && v[0] == '!') {
        out += "!(";
        if (v.size() >= 3) {
            out += v.substr(2, v.size() - 3);
        }
    }
    else {
        out += "(";
        out += v;
    }

    if (node->left != NULL) {
        nodeToStr(node->left, out);
    }
    if (node->right != NULL) {
        nodeToStr(node->right, out);
    }
    out.push_back(')');
}

inline std::string treeToExpr(const TreeNode1* root) {
    std::string localExpr;
    nodeToStr(root, localExpr);
    return localExpr;
}

inline std::string joinWithComma(const std::vector<std::string>& v) {
    std::string out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) out.push_back(',');
        out += v[i];
    }
    return out;
}


struct KeyTrie {
    struct Node {
        std::map<char, int> next;
        int terminalIndex;
        Node() : next(), terminalIndex(-1) {}
    };
    std::vector<Node> nodes;
    std::vector<std::string> orderedKeys;

    KeyTrie() : nodes(1), orderedKeys() {}

    void build(const std::vector<std::string>& keys) {
        orderedKeys = keys;
        for (std::size_t k = 0; k < keys.size(); ++k) {
            const std::string& s = keys[k];
            int cur = 0;
            for (std::size_t i = 0; i < s.size(); ++i) {
                const char ch = s[i];
                std::map<char, int>::iterator it = nodes[cur].next.find(ch);
                if (it == nodes[cur].next.end()) {
                    nodes[cur].next.insert(std::make_pair(ch, static_cast<int>(nodes.size())));
                    nodes.push_back(Node());
                    cur = static_cast<int>(nodes.size() - 1);
                }
                else {
                    cur = it->second;
                }
            }
            if (nodes[cur].terminalIndex == -1) nodes[cur].terminalIndex = static_cast<int>(k);
        }
    }

    bool matchFirst(const std::string& text, std::size_t pos, int& keyIndex,
        std::size_t& matchLen) const {
        int cur = 0;
        keyIndex = -1;
        matchLen = 0;

        int bestIndex = -1;
        std::size_t bestLen = 0;

        for (std::size_t i = pos; i < text.size(); ++i) {
            std::map<char, int>::const_iterator it = nodes[cur].next.find(text[i]);
            if (it == nodes[cur].next.end()) break;
            cur = it->second;

            if (nodes[cur].terminalIndex != -1) {
                bestIndex = nodes[cur].terminalIndex;
                bestLen = i - pos + 1;
            }
        }

        if (bestIndex != -1) {
            keyIndex = bestIndex;
            matchLen = bestLen;
            return true;
        }
        return false;
    }

};

inline std::string makeCacheKey(const std::vector<std::string>& keys) {
    std::string key;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        key += keys[i];
        key.push_back('\x1F');
    }
    return key;
}

inline const KeyTrie& getCompiledMatcher(const std::vector<std::string>& sortedKeys) {
    thread_local std::unordered_map<std::string, KeyTrie> cache;
    const std::string cacheKey = makeCacheKey(sortedKeys);
    auto it = cache.find(cacheKey);
    if (it != cache.end()) return it->second;
    KeyTrie trie; trie.build(sortedKeys);
    return cache.emplace(cacheKey, std::move(trie)).first->second;
}


inline std::string replaceKeysInString(const std::string& bigString,
    const std::map<std::string, std::string>& replacementMap) {
    if (replacementMap.empty()) return bigString;

    std::vector<std::string> keys;
    keys.reserve(replacementMap.size());
    for (std::map<std::string, std::string>::const_iterator it = replacementMap.begin();
        it != replacementMap.end(); ++it) {
        keys.push_back(it->first);
    }
    std::sort(keys.begin(), keys.end());

    const KeyTrie& matcher = getCompiledMatcher(keys);

    std::string out;
    out.reserve(bigString.size());
    const std::size_t n = bigString.size();
    std::size_t i = 0;

    while (i < n) {
        const char prev = (i == 0) ? '\0' : bigString[i - 1];
        if (prev == '[' || prev == ',') {
            int keyIndex = -1;
            std::size_t mlen = 0;
            if (matcher.matchFirst(bigString, i, keyIndex, mlen)) {
                const std::size_t nextPos = i + mlen;
                if (nextPos < n) {
                    const char nextc = bigString[nextPos];
                    if (nextc == ']' || nextc == ',') {
                        const std::string& key = matcher.orderedKeys[static_cast<std::size_t>(keyIndex)];
                        std::map<std::string, std::string>::const_iterator rit = replacementMap.find(key);
                        const std::string& repl = (rit != replacementMap.end()) ? rit->second : key;
                        out += repl;
                        i = nextPos;
                        continue;
                    }
                }
            }
        }
        out.push_back(bigString[i]);
        ++i;
    }

    return out;
}

inline std::string cleanExpr(const std::string& expr) {
    std::string out;
    out.reserve(expr.size());
    std::size_t i = 0, n = expr.size();
    while (i < n) {
        std::size_t start = expr.find(">[", i);
        if (start == std::string::npos) {
            out.append(expr, i, n - i);
            break;
        }
        out.append(expr, i, start - i);
        std::size_t close = expr.find(']', start + 2);
        if (close == std::string::npos) {
            out.append(expr, start, n - start);
            break;
        }
        std::size_t innerOpen = expr.find('[', start + 2);
        if (innerOpen != std::string::npos && innerOpen < close) {
            out.append(expr, start, (start + 2) - start);
            i = start + 2;
            continue;
        }
        out += ">[]";
        i = close + 1;
    }
    return out;
}

inline std::vector<std::string> orderByPattern(const std::string& inputStr,
    const std::set<std::string>& argSet) {
    std::vector<std::string> result;
    if (argSet.empty()) return result;

    const std::string cleaned = cleanExpr(inputStr);

    std::map<char, std::vector<std::string> > buckets;
    for (std::set<std::string>::const_iterator it = argSet.begin(); it != argSet.end(); ++it) {
        if (!it->empty()) {
            buckets[(*it)[0]].push_back(*it);
        }
    }
    for (std::map<char, std::vector<std::string> >::iterator bit = buckets.begin(); bit != buckets.end(); ++bit) {
        std::vector<std::string>& v = bit->second;
        std::sort(v.begin(), v.end(), [](const std::string& a, const std::string& b) {
            if (a.size() != b.size()) return a.size() > b.size();
            return a < b;
            });
    }

    std::map<std::string, std::size_t> firstOccurrence;
    const std::size_t n = cleaned.size();
    for (std::size_t i = 0; i < n; ++i) {
        const char prev = (i == 0) ? '\0' : cleaned[i - 1];
        if (prev != '[' && prev != ',') continue;

        std::map<char, std::vector<std::string> >::const_iterator bit = buckets.find(cleaned[i]);
        if (bit == buckets.end()) continue;

        const std::vector<std::string>& candidates = bit->second;
        for (std::size_t k = 0; k < candidates.size(); ++k) {
            const std::string& key = candidates[k];
            const std::size_t len = key.size();
            if (i + len > n) continue;
            if (cleaned.compare(i, len, key) != 0) continue;
            const std::size_t nextPos = i + len;
            if (nextPos < n) {
                const char nextc = cleaned[nextPos];
                if (nextc == ']' || nextc == ',') {
                    if (firstOccurrence.find(key) == firstOccurrence.end()) {
                        firstOccurrence.insert(std::make_pair(key, i));
                        if (firstOccurrence.size() == argSet.size()) break;
                    }
                }
            }
        }
        if (firstOccurrence.size() == argSet.size()) break;
    }

    std::vector<std::pair<std::string, std::size_t> > found(firstOccurrence.begin(), firstOccurrence.end());
    std::sort(found.begin(), found.end(),
        [](const std::pair<std::string, std::size_t>& a, const std::pair<std::string, std::size_t>& b) {
            return a.second < b.second;
        });
    result.reserve(found.size());
    for (std::size_t i = 0; i < found.size(); ++i) {
        result.push_back(found[i].first);
    }
    return result;
}

inline void deleteTree(TreeNode1* n) {
    if (n == NULL) return;
    deleteTree(n->left);
    deleteTree(n->right);
    delete n;
}

inline std::string disintegrateImplication(
    const std::string& exprForDesintegration,
    std::vector< std::tuple<
    std::string,
    std::vector<std::string>,
    std::set<std::string>
    > >& chain,
    const std::map<std::string, CoreExpressionConfig>& coreExpressionMap) {

    TreeNode1* root = parseExpr(exprForDesintegration);

    std::string head;

    TreeNode1* node = root;
    while (true) {
        if (node != NULL) {
            if (!node->value.empty() && node->value[0] == '>') {
                const std::string leftExpr = treeToExpr(node->left);
                const std::vector<std::string> args = getArgs(node->value);

                std::set<std::string> leftArgs;
                if (node->left != NULL) {
                    leftArgs = node->left->arguments;
                }

                chain.push_back(std::make_tuple(leftExpr, args, leftArgs));
                node = node->right;
            }
            else {
                head = treeToExpr(node);
                break;
            }
        }
        else {
            break;
        }
    }

    deleteTree(root);
    return head;
}



inline void prioritizeAnchor(std::vector<std::string>& chain, const std::string& anchor) {
    for (std::size_t i = 0; i < chain.size(); ++i) {
        if (chain[i].find(anchor) != std::string::npos) {
            const std::string picked = chain[i];
            chain.erase(chain.begin() + static_cast<std::ptrdiff_t>(i));
            chain.insert(chain.begin(), picked);
            break;
        }
    }
}

inline bool staysOutputVariable(const std::string& fullExpr,
    const std::string& outputVariable,
    const std::map<std::string, CoreExpressionConfig>& coreExpressionMap) {

    const std::string coreExpr = extractExpression(fullExpr);
    auto it = coreExpressionMap.find(coreExpr);

    if (it == coreExpressionMap.end() || it->second.outputIndices.empty()) {
        return false;
    }

    const std::vector<std::string> args = getArgs(fullExpr);

    for (int outIdx : it->second.outputIndices) {
        if (outIdx >= 0 && outIdx < static_cast<int>(args.size())) {
            if (args[outIdx] == outputVariable) {
                return true;
            }
        }
    }

    return false;
}

inline std::string createReshuffledMirrored(const std::string& expr,
    const std::string& anchorName,
    bool anchorFirst,
    const std::map<std::string, CoreExpressionConfig>& coreExpressionMap) {

    std::vector< std::tuple< std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
    const std::string head = disintegrateImplication(expr, tempChain, coreExpressionMap);

    const std::vector<std::string> headArgs = getArgs(head);
    const std::string headExpr = extractExpression(head);

    std::string outputVariable;
    auto itHead = coreExpressionMap.find(headExpr);

    if (itHead != coreExpressionMap.end() && !itHead->second.outputIndices.empty()) {
        int primaryOutputIndex = itHead->second.outputIndices[0];
        assert(primaryOutputIndex >= 0 && primaryOutputIndex < static_cast<int>(headArgs.size()));
        outputVariable = headArgs[primaryOutputIndex];
    }

    if (outputVariable.empty()) {
        return std::string();
    }

    std::string alternative;
    std::vector<std::string> chain;
    for (std::size_t i = 0; i < tempChain.size(); ++i) {
        const std::string& leftExpr = std::get<0>(tempChain[i]);

        if (ce::staysOutputVariable(leftExpr, outputVariable, coreExpressionMap)) {
            alternative = leftExpr;
        }
        else {
            chain.push_back(leftExpr);
        }
    }

    if (anchorFirst) {
        prioritizeAnchor(chain, anchorName);
    }

    if (alternative.empty()) {
        return std::string();
    }

    chain.push_back(head);
    chain.push_back(alternative);

    std::set<std::string> argsToRemove;
    for (const auto& t : tempChain) {
        const std::vector<std::string>& nodeArgs = std::get<1>(t);
        argsToRemove.insert(nodeArgs.begin(), nodeArgs.end());
    }

    std::vector< std::set<std::string> > argsChain;
    argsChain.reserve(chain.size());
    for (const auto& c : chain) {
        const std::vector<std::string> a = getArgs(c);
        argsChain.emplace_back(a.begin(), a.end());
    }

    if (chain.empty()) return std::string();
    std::vector< std::vector<std::string> > howToRemove(chain.size() - 1);

    for (const std::string& argToRemove : argsToRemove) {
        for (std::size_t idx = 0; idx < chain.size(); ++idx) {
            if (argsChain[idx].find(argToRemove) != argsChain[idx].end()) {
                if (idx < howToRemove.size()) {
                    howToRemove[idx].push_back(argToRemove);
                }
                break;
            }
        }
    }

    std::string newExpr = chain.back();
    for (int ind = static_cast<int>(chain.size()) - 2; ind >= 0; --ind) {
        const std::vector<std::string>& v = howToRemove[ind];
        std::string joined;
        for (std::size_t j = 0; j < v.size(); ++j) {
            if (j > 0) joined.push_back(',');
            joined += v[j];
        }
        newExpr = "(>[" + joined + "]" + chain[ind] + newExpr + ")";
    }

    return newExpr;
}

inline std::string trimCopy(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    std::size_t j = s.size();
    while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1]))) --j;
    return s.substr(i, j - i);
}

inline std::set<std::string> extractDifference(const std::string& s) {
    std::set<std::string> firstSet;
    std::set<std::string> secondSet;

    {
        const std::regex re1(R"(>\[([^\]]*)\])");
        std::sregex_iterator it(s.begin(), s.end(), re1);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            const std::string inside = (*it)[1].str();
            std::size_t start = 0;
            while (start <= inside.size()) {
                std::size_t pos = inside.find(',', start);
                const std::string token = trimCopy(inside.substr(
                    start, (pos == std::string::npos ? inside.size() : pos) - start));
                if (!token.empty()) firstSet.insert(token);
                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        }
    }

    {
        const std::regex re2(R"(\[([^\]]*)\])");
        std::sregex_iterator it(s.begin(), s.end(), re2);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            const std::size_t lpos = static_cast<std::size_t>((*it).position());
            if (lpos > 0 && s[lpos - 1] == '>') {
                continue;
            }
            const std::string inside = (*it)[1].str();
            std::size_t start = 0;
            while (start <= inside.size()) {
                std::size_t pos = inside.find(',', start);
                const std::string token = trimCopy(inside.substr(
                    start, (pos == std::string::npos ? inside.size() : pos) - start));
                if (!token.empty()) secondSet.insert(token);
                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        }
    }

    std::set<std::string> diff;
    for (std::set<std::string>::const_iterator it = secondSet.begin();
        it != secondSet.end(); ++it) {
        if (firstSet.find(*it) == firstSet.end()) {
            diff.insert(*it);
        }
    }
    return diff;
}

inline bool expressionIsSimple(const std::string& expr) {
    if (expr.size() >= 2 && expr[0] == '(' && expr[1] == '>') {
        return false;
    }
    if (expr.size() >= 3 && expr[0] == '!' && expr[1] == '(' && expr[2] == '>') {
        return false;
    }
    if (expr.size() >= 2 && expr[0] == '(' && expr[1] == '&') {
        return false;
    }
    if (expr.size() >= 3 && expr[0] == '!' && expr[1] == '(' && expr[2] == '&') {
        return false;
    }
    return true;
}


inline std::pair<std::string, std::string>
extractKeyValue(const std::string& expr2,
    const std::map<std::string, ce::CoreExpressionConfig>& coreExpressionMap) {
    TreeNode1* root = parseExpr(expr2);

    std::string value;
    TreeNode1* node = root;
    while (node != NULL) {
        if (!node->value.empty() && node->value[0] == '>') {
            node = node->right;
        }
        else {
            value = treeToExpr(node);
            break;
        }
    }

    std::string key;
    if (value.empty()) {
        key = expr2;
    }
    else {
        const std::size_t pos = expr2.rfind(value);
        if (pos != std::string::npos) {
            key = expr2.substr(0, pos) + expr2.substr(pos + value.size());
        }
        else {
            key = expr2;
        }
    }

    deleteTree(root);
    return std::make_pair(key, value);
}


} // namespace ce

// ============================================================================
// argument_analyzer.hpp content (formerly a separate header)
// ============================================================================

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

// ce:: types provided by namespace ce above

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
