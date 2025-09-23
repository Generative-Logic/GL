
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
#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <variant>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <regex>

// This shim replaces small portions of create_expressions.py that analyze_expressions.py calls.
// Keep surface area minimal: we only expose helpers used in quick mode scaffold.

namespace ce {


    // Returns all unique permutations of the input, in lexicographic order.
    // For an empty input, returns one empty permutation: [[]].
    std::map<int, std::vector<std::vector<int>>> generateAllPermutations(int n);
    
// Binary sequences of given length (as lists). For scaffold, we return simple 0/1 combos.
std::vector<std::vector<int>> generateBinarySequencesAsLists(int n);


// A placeholder for "mappings" generation; in the scaffold, we return empty data.
using Mapping = std::map<int,int>;
std::vector<Mapping> generateAllMappings(int maxDef, int maxTarget);


inline const std::filesystem::path DEFINITIONS_FOLDER =
    std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() / "files" / "definitions";

struct CoreExpressionConfig {
    int arity;
    std::map<std::string, std::string> placeholders;
    std::string binder;
    std::variant<std::string, std::filesystem::path> definition; // inline pattern OR file path
    int levelFlag;
    std::string signature;
    bool specialFlag;

    // Default constructor
    CoreExpressionConfig()
        : arity(0),
        placeholders(),
        binder(),
        definition(std::string()),
        levelFlag(0),
        signature(),
        specialFlag(false) {
    }

    // Full constructor
    CoreExpressionConfig(int arity_,
        const std::map<std::string, std::string>& placeholders_,
        const std::string& binder_,
        const std::variant<std::string, std::filesystem::path>& definition_,
        int levelFlag_,
        const std::string& signature_,
        bool specialFlag_)
        : arity(arity_),
        placeholders(placeholders_),
        binder(binder_),
        definition(definition_),
        levelFlag(levelFlag_),
        signature(signature_),
        specialFlag(specialFlag_) {
    }
};

struct AnchorInfo {
    std::string exampleExpression;                               // "(NaturalNumbers[1,2,3,4,5,6])"
    int arity;                                                   // coreExpressionMap["NaturalNumbers"].arity
    std::map<std::string, std::string> definitionSets;          // NEW: definition sets
    std::string name;                                            // "NaturalNumbers"

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

inline AnchorInfo initAnchor(const std::map<std::string, ce::CoreExpressionConfig>& coreExpressionMap) {
    const std::string key = "NaturalNumbers";

    std::string exampleExpression = "(NaturalNumbers[1,2,3,4,5,6])";
    int arity = 0;
    std::map<std::string, std::string> definitionSets;

    std::map<std::string, ce::CoreExpressionConfig>::const_iterator it = coreExpressionMap.find(key);
    if (it != coreExpressionMap.end()) {
        arity = it->second.arity;
        // placeholders -> definitionSets (same type)
        definitionSets = it->second.placeholders; // copy
    }

    return AnchorInfo(exampleExpression, arity, definitionSets, key);
}




inline const std::map<std::string, CoreExpressionConfig> CORE_EXPRESSION_MAP = {
    { "inN",
      CoreExpressionConfig{
        1,
        { {"1","(1)"}, {"N","P(1)"} },
        "(x(1)P(1))",
        std::string("(in[1,N])"),
        1,
        "inN[1]",
        false
      }
    },
    { "in",
      CoreExpressionConfig{
        2,
        { {"1","(1)"}, {"2","P(1)"} },
        "(x(1)P(1))",
        std::string("(in[1,2])"),
        0,
        "in[1,2]",
        false
      }
    },
    { "=",
      CoreExpressionConfig{
        2,
        { {"1","(1)"}, {"2","(1)"} },
        "(x(1)(1))",
        std::string("(=[1,2])"),
        1,
        "=[1,2]",
        true
      }
    },
    { "fXY",
      CoreExpressionConfig{
        3,
        { {"1","P(x(1)(1))"}, {"2","P(1)"}, {"3","P(1)"} },
        "(xP(x(1)(1))(xP(1)P(1)))",
        DEFINITIONS_FOLDER / "fXY.txt",
        1,
        "fXY[f,X,Y]",
        false
      }
    },
    { "fXYZ",
      CoreExpressionConfig{
        4,
        { {"1","P(x(1)(x(1)(1)))"}, {"2","P(1)"}, {"3","P(1)"}, {"4","P(1)"} },
        "(xP(x(1)(x(1)(1)))(xP(1)(xP(1)P(1))))",
        DEFINITIONS_FOLDER / "fXYZ.txt",
        1,
        "fXY[f,X,Y,Z]",
        false
      }
    },
    { "in2",
      CoreExpressionConfig{
        3,
        { {"1","(1)"}, {"2","(1)"}, {"3","P(x(1)(1))"} },
        "(x(1)(x(1)P(x(1)(1))))",
        std::string("(in2[1,2,3])"),
        0,
        "in2[1,2,3]",
        false
      }
    },
    { "in3",
      CoreExpressionConfig{
        4,
        { {"1","(1)"}, {"2","(1)"}, {"3","(1)"}, {"4","P(x(1)(x(1)(1)))"} },
        "(x(1)(x(1)(x(1)P(x(1)(x(1)(1))))))",
        std::string("(in3[1,2,3,4])"),
        0,
        "in3[1,2,3,4]",
        true
      }
    },
    { "NaturalNumbers",
      CoreExpressionConfig{
        6,
        {
          {"1","P(1)"},
          {"2","(1)"},
          {"3","(1)"},
          {"4","P(x(1)(1))"},
          {"5","P(x(1)(x(1)(1)))"},
          {"6","P(x(1)(x(1)(1)))"}
        },
        "",
        DEFINITIONS_FOLDER / "NaturalNumbers.txt",
        1,
        "NaturalNumbers[N,i0,i1,s,+,*]",
        false
      }
    }
};

inline const std::set<std::string> operators{ "in2", "in3" };

struct TreeNode1 {
    std::string value;
    int numberLeafs;
    TreeNode1* left;
    TreeNode1* right;
    std::set<std::string> arguments;

    TreeNode1()
        : value(),
        numberLeafs(0),
        left(NULL),
        right(NULL),
        arguments() {
    }

    TreeNode1(const std::string& value_, int numberLeafs_)
        : value(value_),
        numberLeafs(numberLeafs_),
        left(NULL),
        right(NULL),
        arguments() {
    }
};



// inline const std::map<std::string, CoreExpressionConfig> CORE_EXPRESSION_MAP = { ... };
// inline const std::filesystem::path DEFINITIONS_FOLDER = std::filesystem::path(__FILE__).parent_path() / "definitions";

// ---- helper: read & strip whitespace like Python's read_tree_from_file + replace ----
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

    // Remove '\n', ' ', '\t', '\r' exactly like the Python replace calls
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

// ---- conversion of modify_core_expression_map() ----
// In Python it mutates the global dict in-place. Since CORE_EXPRESSION_MAP is const here,
// we return a resolved copy with file-based definitions replaced by their stripped contents.
inline std::map<std::string, CoreExpressionConfig> modifyCoreExpressionMap() {
    std::map<std::string, CoreExpressionConfig> resolved = CORE_EXPRESSION_MAP;

    for (std::map<std::string, CoreExpressionConfig>::iterator it = resolved.begin();
        it != resolved.end(); ++it) {

        CoreExpressionConfig& cfg = it->second;

        if (std::holds_alternative<std::filesystem::path>(cfg.definition)) {
            std::filesystem::path p = std::get<std::filesystem::path>(cfg.definition);

            std::error_code ec;
            bool isFile = std::filesystem::is_regular_file(p, ec);
            if (isFile && !ec) {
                std::string definition = readTreeFromFile(p);
                cfg.definition = definition; // replace path with stripped string content
            }
        }
    }
    return resolved;
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
    return std::string(); // Python returned ""
}

inline std::string extractExpressionFromNegation(const std::string& s) {
    std::size_t startIndex = s.find("!(");
    std::size_t endIndex = s.find('[');
    if (startIndex != std::string::npos && endIndex != std::string::npos && startIndex < endIndex) {
        return s.substr(startIndex + 2, endIndex - (startIndex + 2));
    }
    return std::string();
}

std::pair<TreeNode1*, int> parseExpr(
    const std::string& treeStrIn,
    const std::map<std::string, ce::CoreExpressionConfig>& coreExpressionMap);

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
        out += v.substr(1);                // strip leading '!'
    }
    else if (v == "!&") {
        out += "!(&";
    }
    else if (!v.empty() && v[0] == '!') {
        out += "!(";
        if (v.size() >= 3) {
            // node.value like "!(...)" -> take inside between "!(" and trailing ")"
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

        // Track the longest terminal we see while walking forward.
        int bestIndex = -1;
        std::size_t bestLen = 0;

        for (std::size_t i = pos; i < text.size(); ++i) {
            std::map<char, int>::const_iterator it = nodes[cur].next.find(text[i]);
            if (it == nodes[cur].next.end()) break;
            cur = it->second;

            if (nodes[cur].terminalIndex != -1) {
                bestIndex = nodes[cur].terminalIndex;
                bestLen = i - pos + 1;
                // Do NOT break: keep going to prefer the longest possible match.
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
    std::sort(keys.begin(), keys.end()); // stable, matches Python's sorted(keys)

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
    std::string,                    // leftExpr
    std::vector<std::string>,       // args of the current implication node
    std::set<std::string>           // node->left.arguments  (SET, like Python)
    > >& chain,
    const std::map<std::string, CoreExpressionConfig>& coreExpressionMap) {

    std::pair<TreeNode1*, int> pr = parseExpr(exprForDesintegration, coreExpressionMap);
    TreeNode1* root = pr.first;

    std::string head;

    TreeNode1* node = root;
    while (true) {
        if (node != NULL) {
            if (!node->value.empty() && node->value[0] == '>') {
                const std::string leftExpr = treeToExpr(node->left);
                const std::vector<std::string> args = getArgs(node->value);

                std::set<std::string> leftArgs;
                if (node->left != NULL) {
                    leftArgs = node->left->arguments; // assumes TreeNode1::arguments is std::set<std::string>
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

inline std::set<std::string>
findDigitArgs(const std::string& theorem,
    const AnchorInfo& anchor,
    const std::map<std::string, ce::CoreExpressionConfig>& coreExpressionMap) {
    // temp_chain: [(leftExpr, argsAtNode, leftArgsSet)]
    std::vector< std::tuple<
        std::string,                    // left expr
        std::vector<std::string>,       // args of current '>' node
        std::set<std::string>           // left node's arguments (set)
    > > tempChain;

    const std::string head =
        ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

    // Build chain: all left exprs + final head
    std::vector<std::string> chain;
    chain.reserve(tempChain.size() + 1);
    for (std::size_t i = 0; i < tempChain.size(); ++i) {
        chain.push_back(std::get<0>(tempChain[i]));
    }
    chain.push_back(head);

    // Collect all args from operator nodes only
    std::set<std::string> allArgs;
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const std::string coreExpression = ce::extractExpression(chain[i]);
        if (ce::operators.find(coreExpression) != ce::operators.end()) {
            const std::vector<std::string> args = ce::getArgs(chain[i]);
            for (std::size_t k = 0; k < args.size(); ++k) {
                allArgs.insert(args[k]);
            }
        }
    }

    // Remove args of any element that contains the anchor name
    for (std::size_t i = 0; i < chain.size(); ++i) {
        if (chain[i].find(anchor.name) != std::string::npos) {
            const std::vector<std::string> args = ce::getArgs(chain[i]);
            for (std::size_t k = 0; k < args.size(); ++k) {
                std::set<std::string>::iterator it = allArgs.find(args[k]);
                if (it != allArgs.end()) {
                    allArgs.erase(it);
                }
            }
        }
    }

    // For operator nodes, remove the last two args (outputs)
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const std::string coreExpression = ce::extractExpression(chain[i]);
        if (ce::operators.find(coreExpression) != ce::operators.end()) {
            const std::vector<std::string> args = ce::getArgs(chain[i]);
            if (args.size() >= 2U) {
                const std::size_t a = args.size() - 2U;
                const std::size_t b = args.size() - 1U;
                std::set<std::string>::iterator ita = allArgs.find(args[a]);
                if (ita != allArgs.end()) allArgs.erase(ita);
                std::set<std::string>::iterator itb = allArgs.find(args[b]);
                if (itb != allArgs.end()) allArgs.erase(itb);
            }
        }
    }

    return allArgs;
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
    const std::string& outputVariable) {
    const std::string coreExpr = extractExpression(fullExpr);
    const std::vector<std::string> args = getArgs(fullExpr);

    if (coreExpr == "in") {
        if (args.size() >= 1 && args[0] == outputVariable) return true;
    }
    if (coreExpr == "in2") {
        if (args.size() >= 2 && args[1] == outputVariable) return true;
    }
    if (coreExpr == "in3") {
        if (args.size() >= 3 && args[2] == outputVariable) return true;
    }
    return false;
}

inline std::string createReshuffledMirrored(const std::string& expr,
    const std::string& anchorName,
    bool anchorFirst,
    const std::map<std::string, CoreExpressionConfig>& coreExpressionMap) {
    // Build the implication chain
    std::vector< std::tuple< std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
    const std::string head = disintegrateImplication(expr, tempChain, coreExpressionMap);

    // Determine output variable from head
    const std::vector<std::string> headArgs = getArgs(head);
    const std::string headExpr = extractExpression(head);

    std::string outputVariable;
    if (headExpr == "in") {
        if (!headArgs.empty()) outputVariable = headArgs[0];
    }
    else if (headExpr == "in2") {
        if (headArgs.size() >= 2U) outputVariable = headArgs[1];
    }
    else if (headExpr == "in3") {
        if (headArgs.size() >= 3U) outputVariable = headArgs[2];
    }
    assert(!outputVariable.empty());

    // Split chain into "alternative" (the one that keeps output var) and the rest
    std::string alternative;
    std::vector<std::string> chain;
    for (std::size_t i = 0; i < tempChain.size(); ++i) {
        const std::string& leftExpr = std::get<0>(tempChain[i]);
        if (ce::staysOutputVariable(leftExpr, outputVariable)) {
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

    // Final build order: remaining left exprs ... then head, then alternative
    chain.push_back(head);
    chain.push_back(alternative);

    // Collect all args to remove (union over the per-node arg lists from tempChain)
    std::set<std::string> argsToRemove;
    for (std::size_t i = 0; i < tempChain.size(); ++i) {
        const std::vector<std::string>& nodeArgs = std::get<1>(tempChain[i]);
        for (std::size_t k = 0; k < nodeArgs.size(); ++k) {
            argsToRemove.insert(nodeArgs[k]);
        }
    }

    // For each element in chain, precompute its argument set
    std::vector< std::set<std::string> > argsChain;
    argsChain.reserve(chain.size());
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const std::vector<std::string> a = getArgs(chain[i]);
        std::set<std::string> as;
        for (std::size_t k = 0; k < a.size(); ++k) as.insert(a[k]);
        argsChain.push_back(as);
    }

    // For each arg_to_remove, record the first index in chain where it occurs
    if (chain.size() < 1U) {
        return std::string();
    }
    std::vector< std::vector<std::string> > howToRemove(chain.size() - 1U);
    for (std::set<std::string>::const_iterator it = argsToRemove.begin();
        it != argsToRemove.end(); ++it) {
        const std::string& argToRemove = *it;
        for (std::size_t idx = 0; idx < chain.size(); ++idx) {
            if (argsChain[idx].find(argToRemove) != argsChain[idx].end()) {
                if (idx < howToRemove.size()) {
                    howToRemove[idx].push_back(argToRemove);
                }
                break;
            }
        }
    }

    // Reconstruct expression by nesting from the end
    std::string newExpr = chain[chain.size() - 1U];
    for (std::ptrdiff_t ind = static_cast<std::ptrdiff_t>(chain.size()) - 2; ind >= 0; --ind) {
        // join howToRemove[ind] with commas inside [...]
        const std::vector<std::string>& v = howToRemove[static_cast<std::size_t>(ind)];
        std::string joined;
        for (std::size_t j = 0; j < v.size(); ++j) {
            if (j > 0) joined.push_back(',');
            joined += v[j];
        }
        const std::string substr = std::string("[") + joined + "]";
        newExpr = std::string("(>") + substr + chain[static_cast<std::size_t>(ind)] + newExpr + ")";
    }

    // Python had a naming glitch; the intent is to return the rebuilt expression.
    return newExpr;
}

inline std::string trimCopy(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    std::size_t j = s.size();
    while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1]))) --j;
    return s.substr(i, j - i);
}

// Extract tokens in >[ ... ] and in [ ... ] not preceded by '>'.
// Return the set difference: tokens in the second set but not in the first.
inline std::set<std::string> extractDifference(const std::string& s) {
    std::set<std::string> firstSet;
    std::set<std::string> secondSet;

    // 1) >[ ... ]
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

    // 2) [ ... ] not preceded by '>'  (std::regex has no lookbehind; filter manually)
    {
        const std::regex re2(R"(\[([^\]]*)\])");
        std::sregex_iterator it(s.begin(), s.end(), re2);
        std::sregex_iterator end;
        for (; it != end; ++it) {
            const std::size_t lpos = static_cast<std::size_t>((*it).position()); // position of '['
            if (lpos > 0 && s[lpos - 1] == '>') {
                continue; // skip those preceded by '>'
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

    // 3) difference: secondSet - firstSet
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
    return true;
}


inline std::pair<std::string, std::string>
extractKeyValue(const std::string& expr2,
    const std::map<std::string, ce::CoreExpressionConfig>& coreExpressionMap) {
    // Parse to tree
    std::pair<TreeNode1*, int> pr = parseExpr(expr2, coreExpressionMap);
    TreeNode1* root = pr.first;

    // Walk to the right until we leave '>' nodes; then stringify that subtree.
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

    // Compute key by removing the last occurrence of `value` from expr2.
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

    // If your parseExpr allocates nodes, free them here:
    // freeTree(root);

    return std::make_pair(key, value);
}




} // namespace ce
