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

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <atomic>
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
// namespace ce — *create_expressions* helpers (formerly
// create_expressions_shim.hpp). The compiler-side counterpart to the
// Python create_expressions module.
//
// Hosts:
//   - the language-level configuration types `CoreExpressionConfig` and
//     `AnchorInfo`,
//   - `loadCoreExpressionMap` / `modifyCoreExpressionMap` — JSON config
//     loaders that produce the per-anchor `coreExpressionMap` consumed by
//     `ExpressionAnalyzer`,
//   - the parser surface (`getArgs`, `extractExpression`, `extractExpressionUniversal`,
//     `extractExpressionFromNegation`, `cleanExpr`, `replaceKeysInString`,
//     `joinWithComma`, `orderByPattern`),
//   - the tree-shaped intermediate (`TreeNode1` + `nodeToStr` + `treeToExpr` +
//     `deleteTree`),
//   - the disintegration kernel (`disintegrateImplication`) and its companion
//     `createReshuffledMirrored`,
//   - utility analyzers (`KeyTrie`, `ArgumentAnalyzer`, `expressionIsSimple`,
//     `prioritizeAnchor`, `staysOutputVariable`, `extractDifference`).
//
// Almost everything is `inline` so the compiler can fold the small parser
// helpers into hot-path callers without out-of-line dispatch. The two
// non-inline declarations (`generateAllPermutations`, `generateBinarySequencesAsLists`)
// have bodies in compiler.cpp because they are larger and rarely inlined.
// ============================================================================

namespace ce {

/// @brief All unique permutations of the integers `0..n-1`, grouped by the
/// number of fixed points (zero-based).
///
/// @details
/// Bodies in compiler.cpp. The output is `std::map<int, std::vector<std::vector<int>>>`
/// where the integer key is the count of positions where `perm[i] == i`
/// (the number of fixed points) and the value is the list of all
/// permutations with that fixed-point count. Within each group permutations
/// are emitted in lexicographic order, so iteration is deterministic.
///
/// Used by the conjecturer when generating renamings of placeholder
/// variables and by the static request pipeline when enumerating
/// arg-permutations of a key.
///
/// @param n Length of the permutation. `n < 0` returns an empty map; `n == 0`
///          returns a map with the trivial empty permutation under key 0.
/// @return Map from fixed-point count to permutation list, lexicographically
///         ordered within each group.
/// @see [`generateBinarySequencesAsLists`](#generatebinarysequencesaslists) —
///      the binary-mask counterpart.
std::map<int, std::vector<std::vector<int>>> generateAllPermutations(int n);

/// @brief All `2^n` binary sequences of length `n`, encoded as
/// `std::vector<int>` of 0/1 bits, MSB first.
///
/// @details
/// Body in compiler.cpp. Pre-reserves `1 << n` rows; iterates `mask` from
/// 0 to `2^n - 1`; each row is the big-endian bit list of `mask`. Used as
/// a binary-mask generator inside `makeNormalizedKeysForAdmission` and the
/// static-pipeline subkey enumeration.
///
/// @param n Sequence length. `n < 0` returns an empty vector; `n == 0`
///          returns a vector with one empty inner vector (the unique
///          length-0 sequence).
/// @return Vector of `2^n` bit-pattern rows.
/// @pre  `n` should be non-negative and small; `n > 30` will overflow
///       memory long before the loop completes.
std::vector<std::vector<int>> generateBinarySequencesAsLists(int n);

using Mapping = std::map<int,int>;

inline const std::filesystem::path DEFINITIONS_FOLDER =
    std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path() / "files" / "definitions";

/// @brief Per-core-expression configuration record loaded from
/// `files/config/Config<Anchor>.json`.
///
/// @details
/// One `CoreExpressionConfig` describes a single named expression in the
/// MPL — its arity, signature, definition body (inline pattern or path
/// to a file in `files/definitions/`), and the typing-set patterns
/// associated with each definition slot.
///
/// Fields:
/// - `arity`           — declared arity. The signature must declare exactly
///   this many positional placeholders.
/// - `definition`      — `variant<string, filesystem::path>`. A string is
///   treated as an inline pattern; a path is a relative reference into
///   `files/definitions/` whose contents are read at load time.
/// - `signature`       — the canonical string form, e.g. `(in[1,2])`.
///   Each numeric placeholder is one positional argument.
/// - `definitionSets`  — slot name → (typing-set pattern, mandatory flag).
///   Used by `addStatement` and the typing-set checks.
/// - `inputArgs`       — positional names of the operator's input
///   arguments.
/// - `outputArgs`      — positional names of the operator's output
///   arguments. Most operators have arity 0 or 1 outputs; multi-output
///   operators are not supported by the static pipeline (the
///   `single-output-arg` assert in `makeNormalizedKeysForAdmission`
///   guards this).
/// - `inputIndices` / `outputIndices` — derived index arrays into the
///   signature placeholders. Computed once at load time so hot-path
///   consumers can avoid re-parsing.
///
/// @see [`AnchorInfo`](#anchorinfo) — the per-anchor view derived from
///      the `Anchor<ID>` entry of a `CoreExpressionConfig` map.
/// @see `loadCoreExpressionMap`, `modifyCoreExpressionMap` — JSON loaders.
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


/// @brief Per-anchor summary derived from a `CoreExpressionConfig` map.
///
/// @details
/// `initAnchor("Peano")` constructs an `AnchorInfo` for `AnchorPeano` by
/// looking up its `CoreExpressionConfig` and projecting:
/// - `name`             — the full anchor key, e.g. `"AnchorPeano"`.
/// - `arity`            — the declared arity of the anchor.
/// - `exampleExpression` — a synthesized example, produced by
///   `makeAnchorSignature(name, arity)`, e.g. `"(AnchorPeano[1,2,3,4])"`.
/// - `definitionSets`   — slot name → typing-set pattern (the mandatory
///   flag from `CoreExpressionConfig::definitionSets` is dropped here
///   because the per-anchor consumers all treat the typing as
///   mandatory).
///
/// `AnchorInfo` is the structure handed to the compiler-side init pass
/// in `ExpressionAnalyzer`. It is built once per batch and never mutated
/// afterwards.
///
/// @see `initAnchor`, `findAnchorKey`, `makeAnchorSignature`.
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

/// @brief Find the `Anchor<ID>` entry of a `coreExpressionMap`.
///
/// @details
/// Walks the map in `std::map` (sorted) order and returns the first key
/// whose prefix is `"Anchor"`. There is at most one anchor per
/// configuration; the assumption is enforced by the `Config<ID>.json`
/// schema (one `Anchor<ID>` block per file) but not asserted here.
///
/// @param coreExpressionMap Output of `loadCoreExpressionMap` /
///                          `modifyCoreExpressionMap`.
/// @return The full anchor key string (e.g. `"AnchorPeano"`), or empty
///         string if no key starts with `"Anchor"` (a missing anchor
///         tripping `initAnchor`'s assert).
/// @see `initAnchor` — primary consumer.
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

/// @brief Synthesize the canonical example expression for an anchor name.
///
/// @details
/// Produces `"(<name>[1,2,...,arity])"`. Used as the `exampleExpression`
/// slot of `AnchorInfo`. Throws `std::invalid_argument` if `arity < 0`.
///
/// @param name  Anchor name (e.g. `"AnchorPeano"`).
/// @param arity Declared arity. Must be non-negative.
/// @return Canonical signature string.
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


/// @brief Build an `AnchorInfo` for a named anchor inside a `coreExpressionMap`.
///
/// @details
/// Looks up `"Anchor" + anchorID` in `coreExpressionMap` and projects the
/// per-anchor view documented at `AnchorInfo`. Asserts via `assert(false ...)`
/// when the key is missing — a firing assert here means the JSON config
/// for the requested anchor is missing or has the wrong key shape.
///
/// @param coreExpressionMap Output of `loadCoreExpressionMap` /
///                          `modifyCoreExpressionMap`.
/// @param anchorID          Anchor short name without the `"Anchor"`
///                          prefix (e.g. `"Peano"`, `"Gauss"`).
/// @return Populated `AnchorInfo`. The `definitionSets` field carries
///         only the patterns; the mandatory flag is dropped.
/// @invariant [I-19](../../docs/30_invariants.md#i-19) — the missing-anchor
///            assert is intentional; a misnamed batch trips it loud.
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


/// @brief Binary-tree intermediate used by the parser to represent an MPL
/// expression as an explicit tree before flattening back to the canonical
/// string form.
///
/// @details
/// Each node carries a `value` (the expression name or operator), pointers
/// to `left` / `right` children (null for leaves), and a set of bound
/// argument names propagated upward. Used by `readTreeFromFile`,
/// `nodeToStr`, `treeToExpr`, `disintegrateImplication`, and the
/// `ArgumentAnalyzer` parser.
///
/// Lifetime is owned by the constructing function. The companion
/// `deleteTree(node)` function recursively frees a tree built with
/// raw `new`.
///
/// @see `readTreeFromFile`, `nodeToStr`, `treeToExpr`, `deleteTree`.
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


/// @brief Split the comma-separated argument list of a flat MPL expression.
///
/// @details
/// Operates on flat canonical MPL form: `(<name>[arg1,arg2,...,argN])` where
/// each `argI` is itself a *flat* identifier (no inner brackets). Implementation:
/// finds the **first** `[` and the **first** `]` after it, takes the substring
/// between them, and splits on every `,` it sees.
///
/// **Important.** This is *not* a bracket-balanced parse. If a nested expression
/// containing its own `[` / `]` / `,` appears as one of the args (e.g.
/// `(p[(q[a,b]),c])`), `getArgs` will mis-parse:
///
/// - `find(']', start)` returns the **inner** `]` of `(q[a,b])`, so the
///   captured substring is `(q[a,b` (truncated).
/// - The inner `,` between `a` and `b` is then treated as an outer-arg
///   separator, yielding `{"(q[a", "b"}`.
///
/// Callers therefore must only pass flat expressions. The static request
/// pipeline and most prover hot paths satisfy this naturally because they
/// pre-disintegrate compound forms into atomic predicates before reaching
/// `getArgs`. For the few callers that need a balanced parse, the canonical
/// path goes through `parseExpr` (a real recursive descent) instead.
///
/// @param expr Flat canonical MPL expression text.
/// @return Vector of argument strings in source order, possibly empty if
///         the expression has no `[...]` block.
/// @pre  `expr` is flat MPL (no nested `(`/`)`/`[`/`]` inside the args).
/// @warning Nested-expression input is silently mis-parsed; see details above.
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

/// @brief Extract the *core expression name* from a canonical MPL expression.
///
/// @details
/// Strips the leading `(` and reads the name up to the first `[`. For
/// `(=[a,b])` returns `"="`; for `(in2[x,y,z])` returns `"in2"`; for
/// `(AnchorPeano[N,...])` returns `"AnchorPeano"`. Companion functions
/// `extractExpressionUniversal` (handles both negated and non-negated
/// forms) and `extractExpressionFromNegation` (specifically for
/// `!(<...>)` shape) cover the negation cases.
///
/// @param s Canonical MPL expression text. Must NOT be a negated form;
///          for that use `extractExpressionUniversal` or
///          `extractExpressionFromNegation`.
/// @return Expression name (no enclosing parens or brackets).
inline std::string extractExpression(const std::string& s) {
    std::size_t index = s.find('[');
    if (index != std::string::npos) {
        if (!s.empty() && s[0] == '(') {
            return s.substr(1, index - 1);
        }
        else if (s.size() >= 2 && s[0] == '!' && s[1] == '(') {
            // Negated expression: !(name[args]) -> name
            return s.substr(2, index - 2);
        }
        else {
            return s.substr(0, index);
        }
    }
    return std::string();
}

/// @brief Extract the core expression name from a possibly-negated form.
///
/// @details
/// Handles both `(p[...])` and `!(p[...])` — chooses the right
/// extractor (`extractExpression` or `extractExpressionFromNegation`)
/// based on the leading character. Returns the name only; the negation
/// flag is implicit in the input string and not propagated.
///
/// @param s Canonical MPL expression text, possibly negated.
/// @return Core expression name (no negation prefix, no parens, no
///         brackets).
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

/// @brief Extract the core expression name from a `!(p[...])` form.
///
/// @details
/// Specialized for the negated shape: skips the leading `!(` and reads
/// the name up to the first `[`. For `!(=[a,b])` returns `"="`; for
/// `!(in[x,y])` returns `"in"`. Companion to `extractExpression`.
///
/// @param s Canonical negated MPL expression. Must start with `!(`.
/// @return Core expression name (no negation prefix, no parens, no
///         brackets).
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


/// @brief Trie data structure used by `replaceKeysInString` for greedy
/// longest-match key replacement.
///
/// @details
/// Nodes carry transition arrays + an optional terminal index that
/// identifies the matched key in the original key-list order. `build`
/// constructs the trie from a vector of keys; `matchFirst` returns the
/// longest matching key starting at `pos` in `text` (or `false` if
/// none).
///
/// Tries are cached process-wide by `getCompiledMatcher` keyed on
/// `makeCacheKey`'s joined sorted-key string, so repeated invocations
/// with the same key set don't re-pay the construction cost.
///
/// @see `replaceKeysInString` — primary consumer.
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


/// @brief Multi-key substring replacement using a precomputed `KeyTrie`.
///
/// @details
/// Replaces every occurrence of any key from the substitution map with
/// its mapped value, scanning left-to-right and using `KeyTrie::matchFirst`
/// for greedy longest-match dispatch. Cached `KeyTrie`s are kept in a
/// process-wide map keyed on the joined sorted-key string
/// (`makeCacheKey`), so repeated invocations with the same key set
/// re-use the same trie — important on the static-pipeline path where
/// per-LB caches must compile predictably.
///
/// @param bigString       Source text.
/// @param replacementMap  Map from search key to replacement.
/// @return New string with replacements applied.
/// @see [`KeyTrie`](#keytrie), `makeCacheKey`, `getCompiledMatcher`.
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

/// @brief Strip non-canonical artifacts (whitespace, surplus parens) from
/// an MPL expression text.
///
/// @details
/// Defensive normalizer for cases where an upstream produced a
/// non-canonical form. Canonical MPL never has whitespace; this function
/// removes any space / tab / newline that might have leaked in, plus any
/// extraneous outer wrapping. Used by the compressor and a few
/// external-theorem ingest paths.
///
/// @param expr Possibly-non-canonical expression text.
/// @return Canonical form.
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

// Temporary profiling globals (session_24042026 int-story campaign).
// Summed over all callers from compiler.hpp::disintegrateImplication.
// Toggle GL_DISINT_PROFILE to 1 to re-enable the atomic bookkeeping
// (has measurable 32-thread contention overhead — keep off in release).
#ifndef GL_DISINT_PROFILE
#define GL_DISINT_PROFILE 0
#endif
inline std::atomic<uint64_t> g_disintCalls{0};
inline std::atomic<uint64_t> g_disintNs{0};
inline std::atomic<uint64_t> g_disintCacheHits{0};

/// @brief Decompose an implication string into its premise / head /
/// remaining-args triple, applying the disintegration normalization.
///
/// @details
/// `disintegrateImplication` is the parser-side counterpart of the
/// prover's `disintegrateExpr2`: it walks an implication expression and
/// extracts the canonical chain of `(key, value, remaining_args)` triples
/// that the prover's hash engine expects. The result is cached in a
/// thread-local `DisintCache` keyed on the input string so repeat calls
/// with the same expression are O(1).
///
/// Three thread-safe atomic counters (`g_disintCalls`, `g_disintNs`,
/// `g_disintCacheHits`) are bumped per call for profiling. They are
/// inspected by some run-mode summaries but never affect behaviour.
///
/// @return The disintegrated head expression text. Side-effect: also
/// fills the per-thread `DisintCache::chain` with the full
/// `(key, value, remainingArgs)` chain so subsequent callers can read
/// it via the same accessor pattern.
/// @pre  Input expression follows MPL grammar; nested implications are
///       fully bracket-balanced.
/// @see `prover.hpp::disintegrateExpr2` — runtime counterpart.
inline std::string disintegrateImplication(
    const std::string& exprForDesintegration,
    std::vector< std::tuple<
    std::string,
    std::vector<std::string>,
    std::set<std::string>
    > >& chain,
    const std::map<std::string, CoreExpressionConfig>& coreExpressionMap) {

#if GL_DISINT_PROFILE
    auto _t0 = std::chrono::steady_clock::now();
#endif

    // Thread-local last-input cache: the same `expr` is typically
    // disintegrated ~10 times in a row by successive filters on one
    // candidate (checkInputVariablesTheoremOperatorHead →
    // checkInputVariablesOrder → evaluateOperatorExprs2 → triggersExistenceRef
    // → passesMaxSizeAfterExistence → passesComplexityAfterExistence →
    // reshuffle → createReshuffledMirrored → passesInPremiseFilter → ...).
    // A single-slot thread_local cache matches the access pattern: each
    // new candidate misses once, then all subsequent filter calls on the
    // same string hit.
    using ChainT = std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>>;
    struct DisintCache {
        std::string key;
        std::string head;
        ChainT chain;
        bool valid = false;
    };
    thread_local DisintCache _dc;
    if (_dc.valid && _dc.key == exprForDesintegration) {
        chain = _dc.chain;
    #if GL_DISINT_PROFILE
        auto _dt_hit = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - _t0).count();
        g_disintCalls.fetch_add(1, std::memory_order_relaxed);
        g_disintNs.fetch_add((uint64_t)_dt_hit, std::memory_order_relaxed);
        g_disintCacheHits.fetch_add(1, std::memory_order_relaxed);
    #endif
        return _dc.head;
    }

    // Iterative walker (session_24042026 int-story campaign). Replaces the
    // earlier `parseExpr -> TreeNode1 tree -> treeToExpr + node->arguments`
    // pipeline with a direct string walk that allocates nothing but the
    // output chain entries.
    //
    // Structure we're peeling: `(>[bvs1](p1)(>[bvs2](p2)...(head)))`.
    // Each layer of peel: strip outer `(>[bvs]` prefix and the matching
    // `)` suffix (by decrementing `end`), extract the first paren-balanced
    // `(premise)` substring, continue with the rest. When the current
    // range does not start with `(>[`, the range IS the head — return it.
    //
    // `leftArgs` (set of free variables in the premise subtree) is
    // reconstructed by walking `[...]` arg-lists inside the premise:
    // collect atoms (non-`>[`) minus all bvs declared in nested `>[bvs]`
    // inside that premise. This reproduces parseExpr's recursive
    // `node->arguments = union - this-level-bvs` for the standard MPL
    // shapes the conjecturer emits (atoms, `!atom`, `&`, `!&`, `>`,
    // `!>`). See G-34 for the cache-shape stability caveat.
    std::string head;
    {
        const std::string& s = exprForDesintegration;
        size_t cursor = 0;
        size_t end = s.size();
        while (cursor + 3 <= end
               && s[cursor] == '(' && s[cursor + 1] == '>' && s[cursor + 2] == '[') {
            // Parse this layer's bv list: `[id1,id2,...]`.
            const size_t bvStart = cursor + 3;
            const size_t bvClose = s.find(']', bvStart);
            if (bvClose == std::string::npos || bvClose >= end) break;
            std::vector<std::string> bvs;
            {
                size_t p = bvStart;
                while (p < bvClose) {
                    size_t c = s.find(',', p);
                    if (c == std::string::npos || c > bvClose) c = bvClose;
                    if (c > p) bvs.emplace_back(s.substr(p, c - p));
                    p = c + 1;
                }
            }

            // Locate the premise. Two shapes allowed:
            //   (prem)      — positive form, a paren-balanced subexpression.
            //   !(prem)     — negated form, `!` prefix + paren-balanced body.
            // Anything else at this position means the layer-peel can't
            // continue; the rest is the head.
            if (bvClose + 1 >= end) break;
            size_t premRealStart = bvClose + 1;
            size_t balanceFrom;
            if (s[premRealStart] == '(') {
                balanceFrom = premRealStart;
            } else if (s[premRealStart] == '!' && premRealStart + 1 < end && s[premRealStart + 1] == '(') {
                balanceFrom = premRealStart + 1;
            } else {
                break;
            }
            size_t premEnd = balanceFrom;
            int depth = 0;
            for (; premEnd < end; ++premEnd) {
                char ch = s[premEnd];
                if (ch == '(') ++depth;
                else if (ch == ')') { --depth; if (depth == 0) break; }
            }
            if (premEnd >= end) break;
            // premise = s[premRealStart .. premEnd] inclusive (includes `!` prefix if present).
            std::string leftExpr = s.substr(premRealStart, premEnd - premRealStart + 1);

            // leftArgs: atoms in the premise minus bvs declared in any
            // nested `>[...]` inside it.
            std::set<std::string> leftArgs;
            {
                std::set<std::string> nestedBvs;
                size_t pos = 0;
                while (pos < leftExpr.size()) {
                    size_t lb = leftExpr.find('[', pos);
                    if (lb == std::string::npos) break;
                    const bool isBv = (lb > 0 && leftExpr[lb - 1] == '>');
                    size_t rb = leftExpr.find(']', lb);
                    if (rb == std::string::npos) break;
                    size_t p = lb + 1;
                    while (p < rb) {
                        size_t c = leftExpr.find(',', p);
                        if (c == std::string::npos || c > rb) c = rb;
                        if (c > p) {
                            std::string tok = leftExpr.substr(p, c - p);
                            if (isBv) nestedBvs.insert(std::move(tok));
                            else leftArgs.insert(std::move(tok));
                        }
                        p = c + 1;
                    }
                    pos = rb + 1;
                }
                for (auto& b : nestedBvs) leftArgs.erase(b);
            }

            chain.emplace_back(std::move(leftExpr), std::move(bvs), std::move(leftArgs));

            // Advance past the premise; the matching `)` of this wrapper
            // sits at `end - 1`, so shrink the range by 1 to account for
            // it (we will peel it along with this layer).
            cursor = premEnd + 1;
            end = end - 1;
        }
        head = s.substr(cursor, end - cursor);
    }

    // Populate cache for next caller on this thread.
    _dc.key = exprForDesintegration;
    _dc.chain = chain;
    _dc.head = head;
    _dc.valid = true;

#if GL_DISINT_PROFILE
    auto _dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - _t0).count();
    g_disintCalls.fetch_add(1, std::memory_order_relaxed);
    g_disintNs.fetch_add((uint64_t)_dt, std::memory_order_relaxed);
#endif
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

/// @brief Produce the *mirrored* form of an external theorem expression,
/// keeping the anchor's role-arguments in place but permuting the
/// non-anchor premises.
///
/// @details
/// Used by the `--mirror-externals` entry-point in `main.cpp` and by the
/// external-theorem ingest path inside `run_modes::fullRun`. Given an
/// external theorem `expr`, the anchor name (`anchorName`), a flag
/// indicating whether the anchor is on the left or right side of the
/// implication, and the relevant `coreExpressionMap`, this function:
///
/// 1. Identifies the anchor's role arguments inside `expr`.
/// 2. Permutes the non-anchor premises while keeping the anchor's role
///    arguments fixed (the *mirror* operation; the canonical
///    `mirrored from` provenance tag).
/// 3. Returns the mirrored expression text, or empty if the input
///    cannot be mirrored (e.g. no anchor present or anchor in a
///    role-incompatible position).
///
/// The mirroring used here corresponds to the sound `reformulated from`
/// path used for real-math output. The incubator pipeline emits a
/// distinct `incubator back reformulation` tag for its unsound
/// counterpart; the two must never be conflated downstream.
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

/// @brief True iff the given MPL expression is *not* a top-level
/// compound (`(>...)` / `(&...)` / their negated counterparts).
///
/// @details
/// Used as a fast guard in places that need to choose between scalar
/// and structured handling. The implementation looks only at the
/// first few characters: it returns `false` when `expr` starts with
/// `(>`, `!(>`, `(&`, or `!(&`, and `true` otherwise. As a
/// consequence, negated atomic predicates such as `!(p[a])` are
/// reported as simple, even though they carry a leading `!`. Inner
/// parentheses elsewhere in the string are NOT inspected — callers
/// that need a deeper structural test should route through
/// `parseExpr` and walk the resulting tree.
///
/// @param expr Canonical MPL expression text.
/// @return False for top-level compound shapes; true for everything
///         else, including negated atomics.
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
