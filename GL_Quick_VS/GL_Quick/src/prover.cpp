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

#include "prover.hpp"
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
#include <cmath>

namespace gl {

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




/// @brief Construct the prover for a named anchor batch.
///
/// @details
/// One-shot constructor. Steps performed:
/// 1. Default-initialize the per-batch state slots (parameters, body LB,
///    permutation/mapping caches, dependency tracker, mailboxes,
///    contradiction table, etc.) via the member initialiser list.
/// 2. Locate `files/config/Config<anchorID>.json` via a search-path walk
///    (the default search list covers typical run environments).
/// 3. Load the JSON config into `coreExpressionMap` via
///    `ce::modifyCoreExpressionMap` and build `anchorInfo` via
///    `ce::initAnchor` (which asserts the anchor key exists).
/// 4. Populate the `operators` and `expressionsFromConfig` sets from
///    `coreExpressionMap` so hot-path classification queries are O(log N).
/// 5. Seed the per-batch counters (`implCounter`, `existenceCounter`,
///    `andCounter`, `orCounter`, `variableCounter`) from
///    `ProverParameters` defaults; these get bumped on every fresh
///    expression-name mint during the run.
/// 6. Build `allMappingsAna` and `allPermutationsAna` for the fixed
///    arity range so the static request pipeline does not have to
///    enumerate at runtime.
///
/// Construction does **not** run the pipeline — it only loads
/// configuration and prepares the per-batch state. The pipeline is
/// driven by `run_modes::fullRun(anchorID)`.
///
/// @param anchorID Short anchor name (e.g. `"Peano"`, `"Gauss"`,
///                 `"IncubatorPeano"`). The `"Anchor"` prefix is added
///                 internally by `ce::initAnchor`.
/// @pre  `files/config/Config<anchorID>.json` exists and is well-formed.
/// @post `coreExpressionMap`, `anchorInfo`, `operators`,
///       `expressionsFromConfig`, and the per-batch counters are
///       populated. The body LB tree is empty (no theorem loaded yet).
/// @invariant [I-1](../../docs/30_invariants.md#i-1) — every
///            theorem-load path must call
///            `precompileStructuralOperators` before disintegration.
///            That call is in the per-theorem load site, not the
///            constructor.
/// @see [`ce::modifyCoreExpressionMap`](compiler.hpp#modifycoreexpressionmap),
///      [`ce::initAnchor`](compiler.hpp#initanchor).
// Default constructor definition
ExpressionAnalyzer::ExpressionAnalyzer(std::string anchorID)
    :parameters(),
    body(),
    allMappingsAna(),
    allPermutationsAna(),
    globalDependencies(),
    inductionMemoryBlocks(),
    contradictionTable(),
    index(),
    boxes(),
    indexCE(),
    boxesCE(),
	expressionsFromConfig()
{
    std::string configFileName = "Config" + anchorID + ".json";

    // Adjust paths as needed for your specific execution environment
    std::vector<std::string> searchPaths = {
        "files/config/" + configFileName,
        "../../../files/config/" + configFileName,
        "../files/config/" + configFileName,
        "GL/files/config/" + configFileName
    };

    // ---------------------------------------------------------
    // LOAD JSON PARAMETERS (Using existing path logic)
    // ---------------------------------------------------------
    namespace fs = std::filesystem;

    // Path logic matches readSimpleFacts / saveProvedTheorems
    fs::path configPath = fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
        / "files" / "config" / ("Config" + anchorID + ".json");

    if (fs::exists(configPath)) {
        try {
            std::ifstream f(configPath);
            nlohmann::json j;
            f >> j;

            if (j.contains("prover_parameters")) {
                auto& pp = j["prover_parameters"];
                if (pp.contains("sizeAllBinariesAna")) parameters.sizeAllBinariesAna = pp["sizeAllBinariesAna"];
                if (pp.contains("maxIterationNumberProof")) parameters.maxIterationNumberProof = pp["maxIterationNumberProof"];
                if (pp.contains("numberIterationsConjectureFiltering")) parameters.numberIterationsConjectureFiltering = pp["numberIterationsConjectureFiltering"];
                if (pp.contains("maxSizeDefSetMapping")) parameters.maxSizeDefSetMapping = pp["maxSizeDefSetMapping"];
                if (pp.contains("maxSizeTargetSetMapping")) parameters.maxSizeTargetSetMapping = pp["maxSizeTargetSetMapping"];
                if (pp.contains("maxNumberSecondaryVariables")) parameters.maxNumberSecondaryVariables = pp["maxNumberSecondaryVariables"];
                if (pp.contains("sizeAllPermutationsAna")) parameters.sizeAllPermutationsAna = pp["sizeAllPermutationsAna"];
                if (pp.contains("minNumOperatorsKey")) parameters.minNumOperatorsKey = pp["minNumOperatorsKey"];
                if (pp.contains("minNumOperatorsKeyCE")) parameters.minNumOperatorsKeyCE = pp["minNumOperatorsKeyCE"];
                if (pp.contains("maxIterationNumberVariable")) parameters.maxIterationNumberVariable = pp["maxIterationNumberVariable"];
                if (pp.contains("standardMaxSecondaryNumber")) parameters.standardMaxSecondaryNumber = pp["standardMaxSecondaryNumber"];
                if (pp.contains("trackHistory")) parameters.trackHistory = pp["trackHistory"];
                if (pp.contains("standardMaxAdmissionDepth")) parameters.standardMaxAdmissionDepth = pp["standardMaxAdmissionDepth"];
                if (pp.contains("inductionMaxAdmissionDepth")) parameters.inductionMaxAdmissionDepth = pp["inductionMaxAdmissionDepth"];
                if (pp.contains("inductionMaxSecondaryNumber")) parameters.inductionMaxSecondaryNumber = pp["inductionMaxSecondaryNumber"];
                if (pp.contains("counterExampleBoundary")) parameters.counterExampleBoundary = pp["counterExampleBoundary"];
                if (pp.contains("minLenLongKey")) parameters.minLenLongKey = pp["minLenLongKey"];
                if (pp.contains("maxLenHypoKey")) parameters.maxLenHypoKey = pp["maxLenHypoKey"];
                if (pp.contains("debug")) parameters.debug = pp["debug"];
                if (pp.contains("compressor_mode")) parameters.compressor_mode = pp["compressor_mode"];
                if (pp.contains("ban_disintegration")) parameters.ban_disintegration = pp["ban_disintegration"];
                if (pp.contains("max_origin_per_expr")) parameters.max_origin_per_expr = pp["max_origin_per_expr"];
                if (pp.contains("compressor_max_origins_per_expr")) parameters.compressor_max_origins_per_expr = pp["compressor_max_origins_per_expr"];
                if (pp.contains("compressor_hash_bursts")) parameters.compressor_hash_bursts = pp["compressor_hash_bursts"];
                if (pp.contains("try_contradiction")) parameters.try_contradiction = pp["try_contradiction"];
                if (pp.contains("skip_ce_filter")) parameters.skip_ce_filter = pp["skip_ce_filter"];
                if (pp.contains("skip_eq_classes")) parameters.skip_eq_classes = pp["skip_eq_classes"];
                if (pp.contains("incubator_mode")) parameters.incubator_mode = pp["incubator_mode"];
                if (pp.contains("allow_multiplication")) parameters.allow_multiplication = pp["allow_multiplication"];
                if (pp.contains("max_partition_size")) parameters.max_partition_size = pp["max_partition_size"];
                if (pp.contains("max_or_depth")) parameters.max_or_depth = pp["max_or_depth"];
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load config from " << configPath << ": " << e.what() << std::endl;
        }
    }

    std::cout << "[INIT] generating all mappings (defSet=" << parameters.maxSizeDefSetMapping
              << ", targetSet=" << parameters.maxSizeTargetSetMapping << ")..." << std::endl;
    allMappingsAna = generateAllMappings(
        parameters.maxSizeDefSetMapping,
        parameters.maxSizeTargetSetMapping
    );

    std::cout << "[INIT] generating permutations (size=" << parameters.sizeAllPermutationsAna << ")..." << std::endl;
    allPermutationsAna = ce::generateAllPermutations(
        parameters.sizeAllPermutationsAna
    );

    implCounter = 0;
    existenceCounter = 0;
    andCounter = 0;
    orCounter = 0;
    variableCounter = 0;

    // Cross-batch persistent naming: load the per-batch GL_binary_<Tag>.json
    // (which Python pre-populated by copying GL_binary_shared.json before
    // invoking this executable). The loader populates compiledExpressions,
    // registers repetitionExclusionMap entries for spontaneous operators,
    // and bumps the four shared counters so newly-allocated names start
    // above the highest already in use. variableCounter remains zero.
    // No-op on a clean run when the file does not yet exist.
    {
        fs::path glBinDir = fs::path(__FILE__).parent_path()
                            .parent_path().parent_path().parent_path()
                          / "files" / "GL_binaries";
        fs::path glBinFile = glBinDir / ("GL_binary_" + anchorID + ".json");
        this->loadGlBinary(glBinFile);
    }

    Dependencies dependencyTable;
    globalDependencies = dependencyTable;

    std::cout << "[INIT] compiling core expression map..." << std::endl;
    coreExpressionMap = ce::modifyCoreExpressionMap(anchorID);
    std::cout << "[INIT] core expression map done, " << coreExpressionMap.size() << " entries." << std::endl;

    operators.clear();
    for (const auto& kv : coreExpressionMap) {
        // Python equivalent: if input_args and output_args
        if (!kv.second.inputArgs.empty() && !kv.second.outputArgs.empty()) {
            operators.insert(kv.first);
        }
    }



    this->anchorID_ = anchorID;

    // Check for explicit anchor_name override in config JSON
    std::string effectiveAnchorID = anchorID;
    if (fs::exists(configPath)) {
        try {
            std::ifstream f2(configPath);
            nlohmann::json j2;
            f2 >> j2;
            if (j2.contains("anchor_name")) {
                std::string anchorName = j2["anchor_name"].get<std::string>();
                // anchor_name is the full key (e.g. "AnchorIncubator"),
                // but initAnchor prepends "Anchor", so strip it
                if (anchorName.substr(0, 6) == "Anchor") {
                    effectiveAnchorID = anchorName.substr(6);
                }
            }
        } catch (...) {}
    }
    std::cout << "[INIT] initAnchor (effectiveID=" << effectiveAnchorID << ")..." << std::endl;
    anchorInfo = initAnchor(coreExpressionMap, effectiveAnchorID);
    std::cout << "[INIT] initAnchor done." << std::endl;

    permanentBodies.clear();
    permanentBodies.push_back(&body);

    logicalCores = std::max(1u, std::thread::hardware_concurrency());
    //logicalCores = 1;
    assert(logicalCores >= 1);

    compileCoreExpressionMap();

}

//#pragma optimize("", off)
/// @brief Canonicalize an MPL expression — strip whitespace and tighten
/// the outer wrapping.
///
/// @details
/// Canonical MPL never carries whitespace; if upstream produces an
/// expression with stray spaces or tabs, `smoothenExpr` removes them.
/// Also handles a few legacy wrapping cases (extra outer parens that
/// crept in from older code paths) by normalising to the canonical
/// `(<name>[args])` form. Idempotent: calling twice returns the same
/// string.
///
/// @param expr Possibly non-canonical MPL text.
/// @return Canonical form.
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
        ce::TreeNode1* root = ce::parseExpr(smoothenedExpr);

        std::string localExpr;
        bool foundAnd = false;
        Helper::nodeToStr(root, localExpr, foundAnd);

        Helper::deleteTree(root);

        smoothenedExpr = localExpr;
        found = foundAnd;
    }

    return smoothenedExpr;
}

/// @brief Recursive helper for `groomExpr` — parse one subexpression of a
/// tree-format MPL string, returning its flat-list form.
///
/// @details
/// Walks `s` starting at `index`, recognising the tree-format constructs
/// (`(>...)`, `(&...)`, `!(>...)`, `!(&...)`, atomic predicates,
/// negated atomic predicates) and emitting one entry per subexpression
/// into the returned flat vector. Updates `index` to the position
/// immediately after the consumed subexpression so the caller can
/// continue parsing.
///
/// @param s     Tree-format MPL text.
/// @param side  Side flag (0 = left, 1 = right) — used by the caller's
///              recursion to keep the parent's left/right ordering
///              correct.
/// @param index In/out: parse position. Advanced past the consumed
///              subexpression on success.
/// @return Flat-list form of the parsed subexpression.
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

/// @brief Convert a tree-format MPL expression into a flat
/// `std::vector<std::string>` representation suitable for the hash engine.
///
/// @details
/// Top-level driver around `groomSubexpr`. The flat form is what the
/// hash-engine inserts and look-ups operate on — every node of the tree
/// becomes one element of the returned vector. The flat-list form is
/// what `addToHashMemory`'s `key` parameter expects.
///
/// Round-trip: `groomExpr(treeStr) → vector → ce::treeToExpr(parsed) →
/// canonical string` should reproduce a smoothed form of `treeStr` for
/// any well-formed MPL input. The flat-list pass is canonical for
/// hash-engine purposes, not for human display.
///
/// @param treeStr Tree-format MPL expression text.
/// @return Flat-list form, one entry per syntactic node.
std::vector<std::string> ExpressionAnalyzer::groomExpr(const std::string& treeStr) {
    std::size_t index = 0;
    return groomSubexpr(treeStr, 1, index);
}

/// @brief Extract the args quantified out by the most recent `>[...]` of
/// an implication.
///
/// @details
/// Used during disintegration to identify which variables become bound
/// in the head and need fresh-name minting (via `renameLastRemoved`).
/// Walks the expression's outermost `>` operator and returns its
/// bound-variable list.
///
/// @param expr Canonical MPL implication text.
/// @return Vector of bound-variable names from the outermost `>[...]`.
///         Empty if the expression has no outer quantifier.
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

/// @brief Rename the args quantified out by the most recent `>[...]` to
/// fresh `it_<startInt>_lev_<lev>_<id>` names.
///
/// @details
/// Used together with `listLastRemovedArgs` during disintegration. Each
/// last-removed arg becomes a fresh, unique variable name; `startInt`
/// is the seed used for the iteration counter and is bumped per arg.
/// The new `startInt` is returned so the caller can continue minting
/// names without collision.
///
/// **Per [I-17](../../docs/30_invariants.md#i-17), the freshness check
/// inside `disintegrateExpr2`'s use of `savedStartInt` assumes
/// `startInt` is a single monotonic counter. A parallel counter for
/// these names would break the freshness check; do not introduce one.**
///
/// @param expr      Implication text whose outer `>[...]` args are to
///                  be renamed.
/// @param startInt  Seed for the iteration counter.
/// @return `(renamed expression, new startInt, helper text)`. The
///         third tuple slot carries auxiliary notes the caller may
///         want — typically a stringified summary of which args got
///         which new names.
/// @invariant [I-17](../../docs/30_invariants.md#i-17) — single
///            monotonic counter assumption.
std::tuple<std::string, int, std::string>
ExpressionAnalyzer::renameLastRemoved(const std::string& expr, int startInt) {
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


/// @brief Expand a compact MPL expression into its full operator-by-operator
/// form via the `coreExpressionMap`'s definitions.
///
/// @details
/// Compact MPL operators (e.g. `or0`, `existence2`, the per-anchor
/// structural operators) are syntactic sugar; their semantics are
/// declared in `Config<anchor>.json` as longer expansions. `expandExpr`
/// substitutes each compact operator with its full body, producing the
/// canonical un-sugared form that the verifier's chapter-walks expect.
///
/// Used by `precompileStructuralOperators` (per
/// [I-1](../../docs/30_invariants.md#i-1)) on every theorem-load path
/// before disintegration.
///
/// @param expr Possibly-compact MPL expression text.
/// @return Fully-expanded canonical form.
/// @invariant [I-1](../../docs/30_invariants.md#i-1) — expansion must
///            happen on every theorem-load path before disintegration.
std::string ExpressionAnalyzer::expandExpr(const std::string& expr) {
    // 1. Check if expression is simple (e.g., "Macro[arg1, arg2]")
    if (!ce::expressionIsSimple(expr)) {
        return expr;
    }

    // 2. Extract the Head (Macro Name)
    std::string head = ce::extractExpressionUniversal(expr);
    auto it = this->coreExpressionMap.find(head);
    if (it == this->coreExpressionMap.end()) {
        // Not a known macro, return original
        return expr;
    }
    const ce::CoreExpressionConfig& cfg = it->second;

    // 3. Retrieve the Definition Template
    // Assuming definition is stored as a string
    if (!std::holds_alternative<std::string>(cfg.definition)) {
        return expr;
    }
    std::string defString = std::get<std::string>(cfg.definition);

    // 4. Extract Arguments
    // replacingArgs: The values passed in the current expression
    std::vector<std::string> replacingArgs = ce::getArgs(expr);
    // argsToBeReplaced: The formal parameters defined in the macro signature
    std::vector<std::string> argsToBeReplaced = ce::getArgs(cfg.signature);

    // Basic arity check
    if (replacingArgs.size() != argsToBeReplaced.size()) {
        return expr;
    }

    // 5. Collision Avoidance (Disentanglement)
    // Goal: If an input argument (e.g., "x") is NOT a formal parameter but appears 
    // inside the definition (as a bound variable), we must rename the bound variable 
    // in the definition to avoid capturing the input "x".

    std::map<std::string, std::string> collisionRenamingMap;
    int counter = 1;

    // Check every input argument for potential collision
    for (const std::string& inputArg : replacingArgs) {

        // Check if this input arg is actually one of the formal parameters.
        // If it is, it will be replaced by the expansion map later, so it's safe.
        bool isFormalParam = false;
        for (const std::string& param : argsToBeReplaced) {
            if (param == inputArg) {
                isFormalParam = true;
                break;
            }
        }

        if (isFormalParam) {
            continue;
        }

        // If we are here, 'inputArg' is being passed in, but it is NOT a parameter slot.
        // We must ensure 'defString' does not contain 'inputArg' as a local variable.
        // We speculatively add it to the renaming map. 
        // ce::replaceKeysInString will only perform the replacement if the key exists.

        // Generate a unique name: inputArg + Counter
        std::string newName;
        while (true) {
            newName = inputArg + std::to_string(counter);

            // Ensure the new name doesn't exist in the definition OR the input args
            // (Simple string find is used here for safety; ideally token-based)
            if (defString.find(newName) == std::string::npos) {
                // Also ensure we haven't generated a name that conflicts with another input
                bool nameClash = false;
                for (const auto& existingArg : replacingArgs) {
                    if (existingArg == newName) nameClash = true;
                }
                if (!nameClash) break;
            }
            counter++;
        }

        collisionRenamingMap[inputArg] = newName;
    }

    // Apply collision renaming to the definition template BEFORE standard expansion
    if (!collisionRenamingMap.empty()) {
        defString = ce::replaceKeysInString(defString, collisionRenamingMap);
    }

    // 6. Standard Expansion
    // Map formal parameters to input arguments
    std::map<std::string, std::string> expansionMap;
    for (std::size_t i = 0; i < argsToBeReplaced.size(); ++i) {
        expansionMap[argsToBeReplaced[i]] = replacingArgs[i];
    }

    // 7. Final Substitution
    std::string expandedExpr = ce::replaceKeysInString(defString, expansionMap);

    return expandedExpr;
}




/// @brief Compute the *global key* for a memory block — the canonical
/// vector identifying its position in the LB tree.
///
/// @details
/// The global key is a sequence of `exprKey` values from the LB up to
/// the root sentinel, in deepest-first order. Used as a `std::map` key
/// in cross-LB tables (e.g. proof-stack caches) where a `Memory*`
/// pointer alone would not survive teardown / re-creation of the LB.
///
/// @param memoryBlock LB to compute the key for.
/// @return Vector of `exprKey` values from this LB up to the root,
///         deepest first. Returns an empty vector for the root sentinel
///         itself (its `exprKey` is empty by definition).
/// @see The full-LB-chain trace requirement for any debug trap that
///      identifies a target LB; this function returns exactly the
///      chain such traps must match.
std::vector<std::string> ExpressionAnalyzer::getGlobalKey(const Memory& memoryBlock) {
    std::vector<std::string> globalKey;
    const Memory* current = &memoryBlock;
    while (true) {
        if (!current->exprKey.empty()) {
            globalKey.push_back(current->exprKey);
        }
        if (current->parentMemory != NULL) {
            current = current->parentMemory;
        }
        else {
            break;
        }
    }
    std::reverse(globalKey.begin(), globalKey.end());
    return globalKey;
}

// ---------------------------------------------------------------------------
// Incubator: implication multiplication helpers
// ---------------------------------------------------------------------------

// Replace oldArg with newArg at argument boundaries only (won't match v1 inside v10,
// and won't match "7" inside "u_7" — underscore counts as a word character).
static std::string replaceArgInString(const std::string& str,
                                      const std::string& oldArg,
                                      const std::string& newArg) {
    auto isWordChar = [](char c) -> bool {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    };
    std::string result;
    result.reserve(str.size());
    std::size_t i = 0;
    while (i < str.size()) {
        bool atBoundary = (i == 0) || !isWordChar(str[i - 1]);
        if (atBoundary && str.compare(i, oldArg.size(), oldArg) == 0) {
            std::size_t after = i + oldArg.size();
            bool endBoundary = (after >= str.size()) || !isWordChar(str[after]);
            if (endBoundary) {
                result += newArg;
                i = after;
                continue;
            }
        }
        result += str[i];
        ++i;
    }
    return result;
}

// After variable replacement, >[...] lists may contain duplicates. Deduplicate them.
static std::string deduplicateBoundVars(const std::string& impl) {
    std::string result;
    result.reserve(impl.size());
    std::size_t i = 0;
    while (i < impl.size()) {
        // Look for "(>["
        if (i + 2 < impl.size() && impl[i] == '(' && impl[i + 1] == '>' && impl[i + 2] == '[') {
            result += "(>[";
            i += 3;
            std::size_t closeBracket = impl.find(']', i);
            if (closeBracket != std::string::npos) {
                std::string varList = impl.substr(i, closeBracket - i);
                // Parse comma-separated vars, deduplicate preserving order
                std::vector<std::string> vars;
                std::set<std::string> seen;
                std::size_t pos = 0;
                while (pos <= varList.size()) {
                    std::size_t comma = varList.find(',', pos);
                    std::string var;
                    if (comma == std::string::npos) {
                        var = varList.substr(pos);
                        pos = varList.size() + 1;
                    } else {
                        var = varList.substr(pos, comma - pos);
                        pos = comma + 1;
                    }
                    if (!var.empty() && seen.find(var) == seen.end()) {
                        // Strip u_-prefixed vars: they are fixed anchor args,
                        // must not appear as bound (wildcard) variables
                        if (var.size() >= 2 && var[0] == 'u' && var[1] == '_') continue;
                        seen.insert(var);
                        vars.push_back(var);
                    }
                }
                for (std::size_t j = 0; j < vars.size(); ++j) {
                    if (j > 0) result += ',';
                    result += vars[j];
                }
                i = closeBracket; // the ']' will be copied on next iteration
            }
        } else {
            result += impl[i];
            ++i;
        }
    }
    return result;
}

// Generate all set partitions (Bell number) of elements.
// Cap: if > 5 elements, return only original + all pairwise equalities.
using Partition = std::vector<std::vector<std::string>>;

static std::vector<Partition> generateSetPartitions(const std::vector<std::string>& elements, int cap = 5) {
    std::vector<Partition> result;
    if (elements.empty()) {
        result.push_back(Partition());
        return result;
    }

    // Beyond cap: pairwise + triples only.
    if (static_cast<int>(elements.size()) > cap) {
        // Original: all singletons
        Partition orig;
        for (auto& e : elements) orig.push_back({e});
        result.push_back(orig);
        // All pairwise equalities
        for (std::size_t i = 0; i < elements.size(); ++i) {
            for (std::size_t j = i + 1; j < elements.size(); ++j) {
                Partition p;
                for (std::size_t k = 0; k < elements.size(); ++k) {
                    if (k == j) continue;
                    if (k == i) {
                        std::vector<std::string> merged = {elements[i], elements[j]};
                        std::sort(merged.begin(), merged.end());
                        p.push_back(merged);
                    } else {
                        p.push_back({elements[k]});
                    }
                }
                result.push_back(p);
            }
        }
        // All triple equalities
        for (std::size_t i = 0; i < elements.size(); ++i) {
            for (std::size_t j = i + 1; j < elements.size(); ++j) {
                for (std::size_t l = j + 1; l < elements.size(); ++l) {
                    Partition p;
                    for (std::size_t k = 0; k < elements.size(); ++k) {
                        if (k == j || k == l) continue;
                        if (k == i) {
                            std::vector<std::string> merged = {elements[i], elements[j], elements[l]};
                            std::sort(merged.begin(), merged.end());
                            p.push_back(merged);
                        } else {
                            p.push_back({elements[k]});
                        }
                    }
                    result.push_back(p);
                }
            }
        }
        return result;
    }

    // Recursive Bell partition
    std::string first = elements[0];
    std::vector<std::string> rest(elements.begin() + 1, elements.end());
    auto subPartitions = generateSetPartitions(rest, cap);

    for (auto& sp : subPartitions) {
        // Option 1: first as new singleton class
        {
            Partition p = sp;
            p.push_back({first});
            result.push_back(p);
        }
        // Option 2: add first to each existing class
        for (std::size_t i = 0; i < sp.size(); ++i) {
            Partition p = sp;
            p[i].push_back(first);
            std::sort(p[i].begin(), p[i].end());
            result.push_back(p);
        }
    }
    return result;
}

// Numeric-aware comparison for variable names (handles "2" < "10" correctly).
static bool numericLess(const std::string& a, const std::string& b) {
    bool aDigit = !a.empty() && std::all_of(a.begin(), a.end(),
                      [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
    bool bDigit = !b.empty() && std::all_of(b.begin(), b.end(),
                      [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
    if (aDigit && bDigit) return std::stoi(a) < std::stoi(b);
    return a < b;
}

/// @brief Expand an implication into all variable-renamed copies needed
/// to cover every meaningful binding partition.
///
/// @details
/// Used by `addToHashMemory` (Path 1) to produce the per-copy
/// `(key, value)` pairs that get installed in the hash engine. For an
/// implication `(>[v0,v1](p[v0])(q[v0,v1]))`, multiplyImplication
/// generates the family of permutations / partition variants whose
/// renormalized keys differ. The first copy is the input as-is; later
/// copies carry origin records under the `"multiplied from"` tag so the
/// verifier can chain the multiplications back to the original.
///
/// **Per [I-24](../../docs/30_invariants.md#i-24)**, multiplyImplication
/// must NOT equate two distinct free `u_*` anchor parameters — they are
/// the pinned anchor slots and cannot be unified by partition renaming.
/// The function asserts on attempts to do so; a firing assert here means
/// the input implication had a malformed anchor binding.
///
/// @param implication Source implication text.
/// @return Vector of all multiplied copies. Always non-empty (at least
///         the input itself). Order is deterministic.
/// @invariant [I-24](../../docs/30_invariants.md#i-24) — no
///            free-anchor-param equation.
/// @see `addToHashMemory` — primary consumer (Path 1 head LMV inserts).
std::vector<std::string> ExpressionAnalyzer::multiplyImplication(
    const std::string& implication)
{
    // Gated by parameters.allow_multiplication (was parameters.incubator_mode
    // pre-2026-04-29). CE filter still multiplies regardless to keep the
    // filter's reachability fan-out unchanged.
    if (!parameters.allow_multiplication && !ceFilteringActive) return {implication};

    // 1. Parse implication to get chain + head
    std::vector< std::tuple<
        std::string,
        std::vector<std::string>,
        std::set<std::string>
    > > tempChain;
    std::string head = ce::disintegrateImplication(implication, tempChain, coreExpressionMap);

    // 2. Collect all bound vars from >[...] lists
    std::set<std::string> allBoundVars;
    for (auto& tpl : tempChain) {
        for (auto& arg : std::get<1>(tpl)) {
            if (!arg.empty()) allBoundVars.insert(arg);
        }
    }
    // 3. Collect all expressions (premises + head) for type lookup
    //    Skip Anchor expressions — their (1)-typed args inflate partitions needlessly.
    std::vector<std::string> allExprs;
    for (auto& tpl : tempChain) {
        const std::string& expr = std::get<0>(tpl);
        if (expr.find("(Anchor") == std::string::npos)
            allExprs.push_back(expr);
    }
    if (head.find("(Anchor") == std::string::npos)
        allExprs.push_back(head);

    // 4. Collect ALL unique variables (bound + u_) appearing in expressions
    std::set<std::string> allVars;
    allVars.insert(allBoundVars.begin(), allBoundVars.end());
    for (const std::string& expr : allExprs) {
        std::vector<std::string> args = ce::getArgs(expr);
        for (const std::string& arg : args) {
            allVars.insert(arg);
        }
    }

    // 5. Determine which vars are (1)-typed (both bound and u_)
    std::vector<std::string> oneTypedVars;
    for (const std::string& var : allVars) {
        bool isOneTyped = false;
        for (const std::string& expr : allExprs) {
            std::string coreName = ce::extractExpression(expr);
            if (coreName.empty()) continue;
            auto cfgIt = coreExpressionMap.find(coreName);
            if (cfgIt == coreExpressionMap.end()) continue;

            std::vector<std::string> args = ce::getArgs(expr);
            for (std::size_t i = 0; i < args.size(); ++i) {
                if (args[i] == var) {
                    std::string posKey = std::to_string(i + 1);
                    auto dsIt = cfgIt->second.definitionSets.find(posKey);
                    if (dsIt != cfgIt->second.definitionSets.end() &&
                        dsIt->second.first == "(1)") {
                        isOneTyped = true;
                        break;
                    }
                }
            }
            if (isOneTyped) break;
        }
        if (isOneTyped) oneTypedVars.push_back(var);
    }

    std::sort(oneTypedVars.begin(), oneTypedVars.end(), numericLess);

    if (oneTypedVars.size() <= 1) return {implication};

    // 6. Generate set partitions (equivalence relations on oneTypedVars).
    //    Each partition can compose multiple equalization groups simultaneously.
    auto partitions = generateSetPartitions(oneTypedVars, parameters.max_partition_size);

    // 7. For each partition, substitute and deduplicate.
    //    u_ variables always win as representative — non-u_ replaced by u_.
    std::set<std::string> seen;
    std::vector<std::string> result;

    for (auto& partition : partitions) {
        // Skip partitions where any equivalence class contains two different u_ vars.
        // Two distinct free anchor parameters must not be equated by partition-based
        // multiplication: doing so silently rewrites a free slot of the rule body
        // (chapter-1115 bug — `u_6` overwritten by `u_2` inside `existence2`).
        bool hasDoubleU = false;
        for (auto& eqClass : partition) {
            int uCount = 0;
            for (const std::string& v : eqClass) {
                if (v.size() >= 2 && v[0] == 'u' && v[1] == '_') uCount++;
            }
            if (uCount >= 2) { hasDoubleU = true; break; }
        }
        if (hasDoubleU) continue;

        std::string copy = implication;

        for (auto& eqClass : partition) {
            if (eqClass.size() <= 1) continue;

            // Pick representative: prefer u_ variable, else first element
            std::string rep = eqClass[0];
            for (const std::string& v : eqClass) {
                if (v.size() >= 2 && v[0] == 'u' && v[1] == '_') {
                    rep = v;
                    break;
                }
            }

            // Replace all non-representative vars with representative
            for (const std::string& v : eqClass) {
                if (v == rep) continue;
                copy = replaceArgInString(copy, v, rep);
            }
        }

        copy = deduplicateBoundVars(copy);

        // Skip copies whose head is a trivial equality (=[x,x]).
        {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string>> > copyChain;
            std::string copyHead = ce::disintegrateImplication(copy, copyChain, coreExpressionMap);
            if (ce::extractExpression(copyHead) == "=") {
                auto eqArgs = ce::getArgs(copyHead);
                if (eqArgs.size() == 2 && eqArgs[0] == eqArgs[1]) continue;
            }
        }

        if (seen.find(copy) == seen.end()) {
            seen.insert(copy);
            result.push_back(copy);
        }
    }

    return result;
}


// Turn optimizations OFF for just this section
//#pragma optimize("", off)

// addToHashMemory(), makeNormalizedKeysForAdmission(), lessByName(),
// lessByOriginal() — moved to memory.cpp.


// growBaseCandidates(), generateEncodedRequestsStatic(), and
// generateEncodedRequestsStaticPairs() — moved to memory.cpp (alongside
// the inline filterIntEncodedStatements helper from prover.hpp).

// generateEncodedRequestsStaticCE() and filterIntEncodedStatementsCE() —
// moved to filter.cpp.


// checkLocalEncodedMemoryStatic() — moved to memory.cpp.

// ========================================================================
// Static pipeline: IntEncodedExpr-based, zero-alloc request generation
// ========================================================================

// makeMandatoryEncodedStatementLists1Static() and …2Static() — moved to memory.cpp.

//#pragma optimize("", off)
/// @brief Drive the per-LB hypothesis-reaction step — install the
/// recursion hypothesis (or check-zero / check-induction-condition
/// goal) into the LB before its hash burst.
///
/// @details
/// Recursion / induction LBs carry a hypothesis that must be
/// available as a statement before the hash burst can fire. For
/// the successor case, the hypothesis is `(=[s(rec), zero]) → false`
/// (the inductive step's antecedent); for the zero case, the
/// hypothesis is the variable's typing predicate.
///
/// `reactToHypo` reads the LB's `recursionHypothesis` field
/// (populated at LB-creation in `prover.cpp`) and installs the
/// hypothesis statement via `addExprToMemoryBlock`. Without this
/// step, induction-block hash bursts would lack their entry-point
/// premise and never fire.
///
/// @param mb Owning induction / recursion LB.
/// @see `Memory::recursionHypothesis` (memory.hpp).
void ExpressionAnalyzer::reactToHypo(Memory& mb) {
    static const std::regex reInt(R"(^int_lev_\d+_\d+$)");
    const std::string hypoMarker = "_hypo_";

    // Iterate through all equivalence classes maps
    for (const auto& [validityName, classes] : mb.equivalenceClassesMap) {

        // 1. Find the LAST occurrence of "_hypo_"
        // The validity name format is now: ..._var0_x_var1_y_hypo_expr
        size_t posHypo = validityName.rfind(hypoMarker);

        if (posHypo == std::string::npos) {
            continue;
        }

        // Skip OR-tagged validity names to avoid interference with case analysis
        if (validityName.find("__or__") != std::string::npos) {
            continue;
        }

        // 2. Parse variables from validityName
        // Canonical grammar: parent + "_boundary_" + "_var0_x_var1_y_hypo_expr"
        // We look for "_boundary__var0_" to isolate the parent from the hypothesis payload.
        std::string boundaryMarker = "_boundary__var0_";
        size_t posBoundary = validityName.rfind(boundaryMarker, posHypo);

        if (posBoundary == std::string::npos) continue;

        // EXTRACT: The original validity name (before the "_boundary_" that begins the hypo scope).
        std::string parentValidity = validityName.substr(0, posBoundary);

        // posVar points at the "_var0_" sub-token — skip the "_boundary_" (10 chars).
        size_t posVar = posBoundary + 10;

        std::set<std::string> targetVars;
        int idx = 0;
        size_t currentPos = posVar;
        size_t suffixPos = posHypo; // Variables end where "_hypo_" begins

        while (true) {
            std::string currentMarker = "_var" + std::to_string(idx) + "_";

            // Verify the marker exists at the current position
            if (currentPos + currentMarker.size() > validityName.size() ||
                validityName.compare(currentPos, currentMarker.size(), currentMarker) != 0) {
                // Sequence broken or malformed
                break;
            }

            size_t startContent = currentPos + currentMarker.size();

            // Prepare to find the next marker
            std::string nextMarker = "_var" + std::to_string(idx + 1) + "_";
            // Search for next marker, bounded by suffixPos
            size_t nextPos = validityName.find(nextMarker, startContent);

            size_t endContent;
            bool isLast = false;

            if (nextPos == std::string::npos || nextPos >= suffixPos) {
                // If next marker not found or appears after/at the hypo marker, this is the last variable
                endContent = suffixPos;
                isLast = true;
            }
            else {
                endContent = nextPos;
            }

            // Extract variable name
            std::string extractedVar = validityName.substr(startContent, endContent - startContent);
            targetVars.insert(extractedVar);

            if (isLast) break;

            // Move to next iteration
            currentPos = nextPos;
            idx++;
        }

        if (targetVars.empty()) continue;

        // 3. Check Equivalence Classes
        for (const auto& eqClass : classes) {
            for (const std::string& var : eqClass.variables) {

                // Check if this variable is one of the target variables
                if (targetVars.find(var) == targetVars.end()) continue;

                // Check "other" variables constraint
                bool conditionMet = true;
                bool hasOthers = false;

                for (const std::string& other : eqClass.variables) {
                    if (other == var) continue;

                    hasOthers = true;
                    // Check if other variable matches "int_lev_d+_d+"
                    if (!std::regex_match(other, reInt)) {
                        conditionMet = false;
                        break;
                    }
                }

                // If yes, add (=[var, var_copy])
                if (conditionMet && hasOthers) {
                    std::string Y = var;
                    std::string Y_copy = Y + "_copy";
                    std::string equalityExpr = "(=[" + Y + "," + Y_copy + "])";

                    std::string finalExpr = removeUPrefixFromArguments(equalityExpr);

                    std::set<int> levels;
                    levels.insert(mb.level);

                    // Dead-end axiom — see "variable copy" tag
                    std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                    origin.first = "variable copy";

                    addExprToMemoryBlock(finalExpr,
                        mb,
                        -1, // iteration
                        0,  // status (Local statement)
                        levels,
                        origin,
                        -1, // coreId
                        -1, // auxyIndex
                        parentValidity, // UPDATED: Use the extracted original validity name
                        false);
                }
            }
        }
    }
}


// =================== Main driver ===================

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

/// @brief Run one elementary logical step on a single LB — the per-LB
/// hash-burst worker.
///
/// @details
/// Called from `proveKernel`'s parallel pool, once per `(LB, iteration)`
/// pair per worker thread. This function is the single most-cited
/// kernel of the prover; it generates the static request stream from
/// `body.intEncodedStatements`, dispatches each request through
/// `checkLocalEncodedMemoryStatic`, processes the resulting head
/// emissions, drains the per-LB mail, applies any equivalence-class
/// rewrites that came in this iteration, and wires up the
/// hashburst dump that all multi-day investigations have hinged on.
///
/// **🔴 SACRED INFRASTRUCTURE. 🔴** The hashburst dump inside this
/// function is OFF-LIMITS to autonomous edits — including:
/// - The per-section lambdas (`dumpHeader`, `dumpLBChain`,
///   `dumpEncodedStatements`, `dumpToBeProved`, `dumpExprOriginMap`,
///   `dumpHashOriginals`, `dumpAdmissionMap`, `dumpEncodedMapMarkers`,
///   `dumpMailIn`, `dumpCompiledExpressions`).
/// - The three call sites (ENTRY / EARLY-EXIT at `!isActive` return /
///   final EXIT).
/// - The section ordering inside each call site.
/// - The output file path `.debug/hashburst_trace.txt`.
/// - The `isXxxTargetLB` lambda's chain-match condition.
/// - Activate / deactivate state.
///
/// **Why the dump is sacred.** Multi-day investigations
/// ([D-39](../../docs/40_decisions.md#d-39) race, mail unification,
/// equality1 cycles, `&& false` regression, cap-30 walker, `_orint_`
/// mistag) all hinged on diffs of this dump's output. Stable structure
/// means the user can grep / diff / scroll the same section names
/// across sessions. Even small touches risk breaking that compounding
/// value. If the user wants the trap pointed at a new LB or turned
/// off, the user says so explicitly.
///
/// @return Reference to `body` for chaining (the caller passes a
///         reference and gets the same reference back).
/// @invariant [I-21](../../docs/30_invariants.md#i-21) — `internalMailIn`
///            cleared at TOP of hashburst after absorb, not END.
/// @invariant [I-26](../../docs/30_invariants.md#i-26) — mail-out
///            statements/implications MAIN-ONLY; exprOriginMap
///            ALL-SCOPES.
/// @invariant [I-28](../../docs/30_invariants.md#i-28) — cross-LB
///            writes during the parallel phase deferred to
///            post-`pool.join()` collectors.
/// @see The hashburst-dump immutability contract in the project's
///      ABSOLUTE-RULES section: read before any edit anywhere near
///      this function.
Memory& ExpressionAnalyzer::performElementaryLogicalStep(Memory& body,
    unsigned coreId,
    const ParentChildrenMap& index,
    PerCoreMailboxes& boxes) {

    if (!body.isActive) {
        return body;
    }

    // Hashburst trap helpers — split into per-section lambdas. Each call site
    // (ENTRY / EARLY-EXIT at !isActive return / final EXIT) composes the
    // subset of sections it needs, all writing to .debug/hashburst_trace.txt.
    //
    // 🔴 DO NOT TOUCH THIS DUMP. 🔴 CLAUDE.md Rule 14: the entire hashburst
    // dump infrastructure — lambdas, call sites, sections, file path, AND the
    // LB-targeting chain match below — requires explicit user approval to
    // change. No autonomous edits, including activate/deactivate or LB
    // retargeting. Mirrored in memory entry `feedback_dump_structure_immutable.md`.
    //
    // Current target: Gauss fold theorem recursion-block-#1 successor LB
    // body `(in2[rec0,9,3])` — induction-step LB created at
    // `tempMb->exprKey = tempExpr` inside `addRecursionBlocks` (the
    // `(in2[recArg, digitArg, sName])` shape with recursionCounter=0,
    // digitArg=9, sName=3 for the active Gauss fold theorem).
    // Chain (innermost → root):
    //   [0] (in2[rec0,9,3])             -- recursion-block-#1 successor LB
    //   [1] (in2[9,10,3])               -- in2 premise
    //   [2] (fold[1,3,4,8,2,9,12])      -- fold theorem head
    //   [3] (in3[9,10,11,5])            -- in3 wrapper
    //   [4] (AnchorGauss[1,2,3,4,5,6,7,8])
    //   [5] <root> (empty exprKey, parentMemory == nullptr)
    auto isChap193TargetLB = [&]() -> bool {
        const Memory* p1 = body.parentMemory;
        const Memory* p2 = p1 ? p1->parentMemory : nullptr;
        const Memory* p3 = p2 ? p2->parentMemory : nullptr;
        const Memory* p4 = p3 ? p3->parentMemory : nullptr;
        const Memory* p5 = p4 ? p4->parentMemory : nullptr;
        return body.exprKey == "(in2[rec0,9,3])"
            && p1 && p1->exprKey == "(in2[9,10,3])"
            && p2 && p2->exprKey == "(fold[1,3,4,8,2,9,12])"
            && p3 && p3->exprKey == "(in3[9,10,11,5])"
            && p4 && p4->exprKey == "(AnchorGauss[1,2,3,4,5,6,7,8])"
            && p5 && p5->exprKey.empty()
            && p5->parentMemory == nullptr;
    };
    auto dumpHeader = [&](std::ofstream& f, const std::string& label, int count) {
        f << "\n=== HASHBURST " << label << " #" << count
          << " | stmts=" << body.encodedStatements.size()
          << " | origins=" << body.exprOriginMap.size()
          << " | hashMem=" << body.overallHashMemory.originals.size()
          << " | toBeProved=" << body.toBeProved.size()
          << " ===\n";
    };
    auto dumpLBChain = [&](std::ofstream& f) {
        f << "-- LB chain (innermost -> root):\n";
        const Memory* cur = &body;
        int depth = 0;
        while (cur != nullptr) {
            f << "  [" << depth << "] " << cur->exprKey << "\n";
            cur = cur->parentMemory;
            ++depth;
        }
    };
    auto dumpEncodedStatements = [&](std::ofstream& f) {
        f << "-- encodedStatements:\n";
        for (size_t i = 0; i < body.encodedStatements.size(); ++i) {
            f << "  [" << i << "] " << body.encodedStatements[i].original
              << " | v=" << body.encodedStatements[i].validityName << "\n";
        }
    };
    auto dumpToBeProved = [&](std::ofstream& f) {
        f << "-- toBeProved (" << body.toBeProved.size() << "):\n";
        for (const auto& [key, val] : body.toBeProved) {
            const auto& auxies = std::get<0>(val);
            const auto& tags   = std::get<1>(val);
            f << "  " << key.original << " | v=" << key.validityName;
            f << " | auxies={";
            bool firstA = true;
            for (int a : auxies) { if (!firstA) f << ","; f << a; firstA = false; }
            f << "} | tags={";
            bool firstT = true;
            for (const auto& t : tags) { if (!firstT) f << ","; f << t; firstT = false; }
            f << "}\n";
        }
    };
    auto dumpExprOriginMap = [&](std::ofstream& f) {
        f << "-- exprOriginMap:\n";
        for (const auto& [key, origins] : body.exprOriginMap) {
            f << "  " << key.original << " | v=" << key.validityName << "\n";
            for (const auto& [tag, deps] : origins) {
                f << "    <- " << tag;
                for (const auto& d : deps) f << " | " << d.original << " (v=" << d.validityName << ")";
                f << "\n";
            }
        }
    };
    auto dumpHashOriginals = [&](std::ofstream& f) {
        f << "-- overallHashMemory.originals (" << body.overallHashMemory.originals.size() << "):\n";
        for (const auto& orig : body.overallHashMemory.originals) {
            f << " ";
            for (const auto& s : orig) f << " " << s;
            f << "\n";
        }
    };
    auto dumpAdmissionMap = [&](std::ofstream& f) {
        f << "-- overallHashMemory.admissionMap (" << body.overallHashMemory.admissionMap.size() << "):\n";
        for (const auto& [keyEv, values] : body.overallHashMemory.admissionMap) {
            f << "  markerExpr=" << keyEv.original << " | v=" << keyEv.validityName
              << " | entries=" << values.size() << "\n";
            int ei = 0;
            for (const auto& v : values) {
                f << "    [" << ei++ << "] key={";
                for (const auto& s : v.key) f << s << " ";
                f << "} remainingArgs={";
                for (const auto& r : v.remainingArgs) f << r << " ";
                f << "} maxDepth=" << v.standardMaxAdmissionDepth
                  << " maxSec=" << v.standardMaxSecondaryNumber
                  << " flag=" << v.flag << "\n";
            }
        }
    };
    auto dumpEncodedMapMarkers = [&](std::ofstream& f) {
        std::vector<const LocalMemoryValue*> markerLmvs;
        for (const auto& [intKey, lmvList] : body.overallHashMemory.encodedMap) {
            for (const auto& lmv : lmvList) {
                if (lmv.value.find("marker") != std::string::npos) {
                    markerLmvs.push_back(&lmv);
                }
            }
        }
        std::sort(markerLmvs.begin(), markerLmvs.end(),
            [](const LocalMemoryValue* a, const LocalMemoryValue* b) {
                if (a->value != b->value) return a->value < b->value;
                if (a->key != b->key) return a->key < b->key;
                return a->remainingArgs < b->remainingArgs;
            });
        f << "-- overallHashMemory.encodedMap marker entries (" << markerLmvs.size() << "):\n";
        for (const auto* lmv : markerLmvs) {
            f << "  markerExpr=" << lmv->value << "\n";
            f << "    key={";
            for (const auto& s : lmv->key) f << s << " ";
            f << "}\n";
            f << "    remainingArgs={";
            for (const auto& r : lmv->remainingArgs) f << r << " ";
            f << "}\n";
        }
    };
    auto dumpMailIn = [&](std::ofstream& f) {
        f << "-- mailIn.statements (" << body.mailIn.statements.size() << "):\n";
        for (const auto& st : body.mailIn.statements) {
            f << "  " << st.first.original
              << " (vName=" << st.first.validityName << ")\n";
        }
        f << "-- mailIn.implications (" << body.mailIn.implications.size() << "):\n";
        for (const auto& imp : body.mailIn.implications) {
            f << "  full: " << std::get<4>(imp) << "\n";
        }
        f << "-- mailIn.exprOriginMap (" << body.mailIn.exprOriginMap.size() << "):\n";
        for (const auto& [key, origins] : body.mailIn.exprOriginMap) {
            f << "  " << key.original << " | v=" << key.validityName << "\n";
            for (const auto& [tag, deps] : origins) {
                f << "    <- " << tag;
                for (const auto& d : deps) f << " | " << d.original << " (v=" << d.validityName << ")";
                f << "\n";
            }
        }
    };
    auto dumpCompiledExpressions = [&](std::ofstream& f) {
        f << "-- compiledExpressions (" << this->compiledExpressions.size() << "):\n";
        for (const auto& [name, le] : this->compiledExpressions) {
            f << "  " << name << " | sig=" << le.signature << "\n";
        }
    };

    // ENTRY trap — full dump (LB chain, encodedStatements, toBeProved,
    // exprOriginMap, hashOriginals, admissionMap, encodedMap markers, mailIn
    // sections, plus compiledExpressions one-shot on burst #1). Chain target
    // documented at the `isChap193TargetLB` lambda above.
    if (isChap193TargetLB()) {
        static std::mutex entryMtx;
        std::lock_guard<std::mutex> lock(entryMtx);
        static int entryCount = 0;
        ++entryCount;
        std::ofstream f(".debug/hashburst_trace.txt",
                        entryCount == 1 ? std::ios::trunc : std::ios::app);
        dumpHeader(f, "ENTRY", entryCount);
        dumpLBChain(f);
        dumpEncodedStatements(f);
        dumpToBeProved(f);
        dumpExprOriginMap(f);
        dumpHashOriginals(f);
        dumpAdmissionMap(f);
        dumpEncodedMapMarkers(f);
        dumpMailIn(f);
        if (entryCount == 1) dumpCompiledExpressions(f);
        f.flush();
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

        this->addToHashMemory(chain, head, remainingArgsKey, body, body.overallHashMemory, levels,
            orImpl, parameters.standardMaxAdmissionDepth, parameters.standardMaxSecondaryNumber, false,
            parameters.minNumOperatorsKey, "implication", true, orImpl, "main");
    }

    // expr_origin_map = mail_in merged into current via addOrigin (D-49).
    // Pre-D-49: raw map merge with body-wins-on-conflict. mailIn's full
    // origin vector landed verbatim; existing body entries overrode mail
    // entries on conflict. With cap=1 effectively enforced only by
    // addOrigin and bypassed here, the policy that ought to govern which
    // origin survives (foundation displaces convenience) was inert.
    // Routing through addOrigin lets the cap-full preference replacement
    // (anything > equality1/equality2) decide per-key which origin wins.
    {
        // Sort mail origins for deterministic encoding order
        for (auto& kv : body.mailIn.exprOriginMap) {
            std::sort(kv.second.begin(), kv.second.end());
        }
        const int maxOriginsForBulk = parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr
            : parameters.max_origin_per_expr;
        for (const auto& kv : body.mailIn.exprOriginMap) {
            for (const auto& origin : kv.second) {
                addOrigin(body.exprOriginMap, kv.first, origin, maxOriginsForBulk);
            }
        }
    }

    // Sync mail-arrived equality origins into the receiving LB's
    // EquivalenceClass.equalityOriginMap. The bulk-merge above covers
    // body.exprOriginMap, but the per-class equalityOriginMap of any class
    // already containing both args of (=[a,b]) is not updated by that merge.
    // Without this sync, mergeTwoEquivalenceClasses (prover.hpp) consults
    // class state when deciding whether to emit equality2 cross-pair origin
    // records and does not see mail-derived origins as already-known —
    // producing redundant transitive-closure records that can form cycles
    // when distinct bridge variables (commonArg) generate mutually-pointing
    // origins. The chapter-22 origin-chain-termination failures on theorem
    // 12 zero-case were the trigger for this sync.
    //
    // Equalities whose vars are not yet class-bound in any scope are NOT
    // synced here — they are picked up via prover.hpp's updateEquivalenceClasses
    // when their own absorption (line 1483-1502 below) fires; line 5886 of
    // prover.hpp seeds the class with the incoming equality's origin
    // parameter, which itself is read from body.mailIn.exprOriginMap at
    // line 1492-1498.
    if (parameters.trackHistory) {
        const int maxOrig = parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr
            : parameters.max_origin_per_expr;
        for (const auto& kv : body.mailIn.exprOriginMap) {
            const std::string& expr = kv.first.original;
            // 2-arg positive equality only. Negated-equality keys begin
            // with "!(=[" — they are skipped by the prefix gate.
            if (expr.size() < 3 || expr.compare(0, 3, "(=[") != 0) continue;
            const std::vector<std::string> args = ce::getArgs(expr);
            if (args.size() != 2) continue;
            const std::string& vName = kv.first.validityName;
            auto itClasses = body.equivalenceClassesMap.find(vName);
            if (itClasses == body.equivalenceClassesMap.end()) continue;
            for (auto& cls : itClasses->second) {
                if (cls.variables.find(args[0]) == cls.variables.end()) continue;
                if (cls.variables.find(args[1]) == cls.variables.end()) continue;
                for (const auto& origin : kv.second) {
                    addOrigin(cls.equalityOriginMap, kv.first, origin, maxOrig);
                }
            }
        }
    }

    // Arena for IntEncodedExpr copies (stable pointers through eval)
    TypedArena<IntEncodedExpr> exprArena(65536);

    // Request buffer
    std::vector<StaticRequest> reqBuf;
    reqBuf.resize(8192);
    int16_t totalReqs = 0;

    // --- Absorb internalMailIn (integration-side revival channel) ---
    // Runs in BOTH ceFilteringActive and normal mode. Each tuple carries
    // its own validity (may be non-main), so the absorb uses the tuple's
    // vName when calling addExprToMemoryBlock. status=1 routes the incoming
    // constituent through the full disintegration pipeline (unlike legacy
    // mailIn absorb which uses status=3 to skip disintegration) — this is
    // the whole point of integration revival: re-fire the pipeline so the
    // (now rewritten, or newly-admissible) constituent can be absorbed.
    //
    // Cleared immediately after absorb so subsequent inserts during this
    // hashburst body (via applyEquivalenceClassToRejectedMapIntegration or
    // revisitRejectedIntegration2) survive to next hashburst.
    {
        std::vector< std::pair<ExpressionWithValidity, std::set<int>> > sortedInternal(
            body.internalMailIn.statements.begin(), body.internalMailIn.statements.end());
        for (const auto& entry : sortedInternal) {
            const std::string& statement = entry.first.original;
            const std::string& vName     = entry.first.validityName;
            const std::set<int>& levels  = entry.second;

            EncodedExpression encStmt(statement, vName);
            if (body.statementLevelsMap.find(encStmt) == body.statementLevelsMap.end()) {
                std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                if (parameters.trackHistory) {
                    ExpressionWithValidity stVal(statement, vName);
                    auto oit = body.internalMailIn.exprOriginMap.find(stVal);
                    // D-45: every internalMailIn.statements entry must have a
                    // paired internalMailIn.exprOriginMap entry. The previous
                    // defensive empty-origin fallback masked a sender-side
                    // bug — replaced with a hard assert.
                    assert(oit != body.internalMailIn.exprOriginMap.end() && !oit->second.empty());
                    origin = oit->second.front();
                }
                addExprToMemoryBlock(statement, body, -1, /*status=*/1, levels,
                                     origin, coreId, -1, vName, false);
            }
        }
        body.internalMailIn.statements.clear();
        body.internalMailIn.exprOriginMap.clear();
    }

    // Sort mail statements for deterministic absorption.
    // Element type is pair<ExpressionWithValidity, levels>; the EWV's
    // validityName is "main" for routing-channel traffic by I-26 sender
    // contract — performElementaryLogicalStep asserts this on consumption
    // (D-53, 2026-05-07; renumbered from main's D-46 on merge into sandbox/incub_fix).
    std::vector< std::pair<ExpressionWithValidity, std::set<int> > > sortedStatements(
        body.mailIn.statements.begin(), body.mailIn.statements.end());

    if (ceFilteringActive) {
        // --- CE mode: absorb mail, then enumerate all statement combos ---
        for (std::vector< std::pair<ExpressionWithValidity, std::set<int> > >::const_iterator it = sortedStatements.begin();
            it != sortedStatements.end(); ++it) {
            const std::string& statement = it->first.original;
            const std::string& vName = it->first.validityName;
            // I-26: routing-channel (mailIn) is main-only by sender contract.
            // The EWV's validityName must be "main" here. Non-main scopes
            // reach Mail-shaped absorb only via internalMailIn (separate
            // top-of-burst block earlier in this function).
            assert(vName == "main");
            const std::set<int>& levels = it->second;
            EncodedExpression encStmt(statement, vName);
            ExpressionWithValidity stVal(statement, vName);
            if (body.statementLevelsMap.find(encStmt) == body.statementLevelsMap.end()) {
                std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                if (parameters.trackHistory) {
                    auto oit = body.mailIn.exprOriginMap.find(stVal);
                    if (oit != body.mailIn.exprOriginMap.end() && !oit->second.empty()) {
                        origin = oit->second.front();
                    }
                }
                addExprToMemoryBlock(statement, body, -1, 3, levels, origin, coreId, -1, vName, false);
            }
        }

        totalReqs = this->generateEncodedRequestsStaticCE(body, body.overallHashMemory,
            exprArena, reqBuf.data(), 8192);

    } else {
        // --- Normal mode: 5-batch mandatory pipeline ---

        // working memory (only from mail_in implications)
        Memory working;
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
            this->addToHashMemory(chain, head, remainingArgsKey, body, working.overallHashMemory, levels,
                orImpl, parameters.standardMaxAdmissionDepth, parameters.standardMaxSecondaryNumber, false,
                parameters.minNumOperatorsKey, "implication", true, orImpl, "main");
        }

        // Encode mail statements for static filter
        std::vector<IntEncodedExpr> mailIntEncoded;
        mailIntEncoded.reserve(sortedStatements.size());
        for (const auto& st : sortedStatements) {
            EncodedExpression tmp(st.first.original, st.first.validityName);
            mailIntEncoded.push_back(encodeExpression(tmp, body.nameMap));
        }

        int16_t mslBuf[4096];

        // --- Batch 1: working memory from mail_in implications ---
        if (!working.overallHashMemory.encodedMap.empty()) {
            int16_t nMsl1 = this->makeMandatoryEncodedStatementLists1Static(
                working.overallHashMemory,
                body.intLocalEncodedStatements.data(),
                static_cast<int16_t>(body.intLocalEncodedStatements.size()),
                mslBuf, 4096);
            if (nMsl1 > 0) {
                totalReqs += this->generateEncodedRequestsStatic(body, working.overallHashMemory,
                    mslBuf, nMsl1,
                    body.intLocalEncodedStatements.data(),
                    exprArena,
                    reqBuf.data() + totalReqs, static_cast<int16_t>(8192 - totalReqs));
            }
        }

        // absorb incoming statements
        for (std::vector< std::pair<ExpressionWithValidity, std::set<int> > >::const_iterator it = sortedStatements.begin();
            it != sortedStatements.end(); ++it) {
            const std::string& statement = it->first.original;
            const std::string& vName = it->first.validityName;
            // I-26: routing-channel (mailIn) is main-only by sender contract.
            // The EWV's validityName must be "main" here. Non-main scopes
            // reach Mail-shaped absorb only via internalMailIn (separate
            // top-of-burst block earlier in this function).
            assert(vName == "main");
            const std::set<int>& levels = it->second;
            EncodedExpression encStmt(statement, vName);
            ExpressionWithValidity stVal(statement, vName);
            if (body.statementLevelsMap.find(encStmt) == body.statementLevelsMap.end()) {
                std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                if (parameters.trackHistory) {
                    auto oit = body.mailIn.exprOriginMap.find(stVal);
                    // D-45: every mailIn.statements entry must have a paired
                    // mailIn.exprOriginMap entry. The previous defensive
                    // empty-origin fallback masked a sender-side bug —
                    // replaced with a hard assert.
                    assert(oit != body.mailIn.exprOriginMap.end() && !oit->second.empty());
                    origin = oit->second.front();
                }
                addExprToMemoryBlock(statement, body, -1, 3, levels, origin, coreId, -1, vName, false);
            }
        }

        // --- Batch 2: new local delta ---
        {
            int16_t nMsl2 = this->makeMandatoryEncodedStatementLists1Static(
                body.overallHashMemory,
                body.intLocalEncodedStatementsDelta.data(),
                static_cast<int16_t>(body.intLocalEncodedStatementsDelta.size()),
                mslBuf, 4096);

            if (nMsl2 > 0) {
                totalReqs += this->generateEncodedRequestsStatic(body, body.overallHashMemory,
                    mslBuf, nMsl2,
                    body.intLocalEncodedStatementsDelta.data(),
                    exprArena,
                    reqBuf.data() + totalReqs, static_cast<int16_t>(8192 - totalReqs));
            }
        }

        // --- Batch 3: local × mail pairs (fully static) ---
        {
            MandatoryPair pairsBuf[8192];
            int16_t nPairs = this->makeMandatoryEncodedStatementLists2Static(
                body, body.overallHashMemory,
                body.intLocalEncodedStatements.data(),
                static_cast<int16_t>(body.intLocalEncodedStatements.size()),
                mailIntEncoded.data(),
                static_cast<int16_t>(mailIntEncoded.size()),
                pairsBuf, 8192);
            if (nPairs > 0) {
                totalReqs += this->generateEncodedRequestsStaticPairs(body, body.overallHashMemory,
                    pairsBuf, nPairs,
                    body.intLocalEncodedStatements.data(),
                    mailIntEncoded.data(),
                    exprArena,
                    reqBuf.data() + totalReqs, static_cast<int16_t>(8192 - totalReqs));
            }
        }

        // --- Batch 4: localHashMemory — mail mandatory singles ---
        if (!body.localHashMemory.encodedMap.empty() && !mailIntEncoded.empty()) {
            int16_t nMsl4 = this->makeMandatoryEncodedStatementLists1Static(
                body.localHashMemory,
                mailIntEncoded.data(),
                static_cast<int16_t>(mailIntEncoded.size()),
                mslBuf, 4096);
            if (nMsl4 > 0) {
                totalReqs += this->generateEncodedRequestsStatic(body, body.localHashMemory,
                    mslBuf, nMsl4,
                    mailIntEncoded.data(),
                    exprArena,
                    reqBuf.data() + totalReqs, static_cast<int16_t>(8192 - totalReqs));
            }
        }

        // --- Batch 5: localHashMemoryDelta ---
        if (!body.localHashMemoryDelta.encodedMap.empty()) {
            int16_t nMsl5 = this->makeMandatoryEncodedStatementLists1Static(
                body.localHashMemoryDelta,
                body.intEncodedStatements.data(),
                static_cast<int16_t>(body.intEncodedStatements.size()),
                mslBuf, 4096);
            if (nMsl5 > 0) {
                totalReqs += this->generateEncodedRequestsStatic(body, body.localHashMemoryDelta,
                    mslBuf, nMsl5,
                    body.intEncodedStatements.data(),
                    exprArena,
                    reqBuf.data() + totalReqs, static_cast<int16_t>(8192 - totalReqs));
            }
        }
    } // end normal mode

    // clear mail_in and local delta
    body.mailIn.statements.clear();
    body.mailIn.implications.clear();
    body.mailIn.exprOriginMap.clear();

    if (body.contradictionIndex == -1) {
        body.localEncodedStatementsDelta.clear();
        body.intLocalEncodedStatementsDelta.clear();
    }

    body.localHashMemoryDelta = HashMemory();

    // Fixpoint eval loop — ONE loop, fully static requests
    {
        bool changed = true;
        while (changed) {
            changed = false;
            if (!body.isActive) {
                // EARLY-EXIT trap — !isActive fixpoint early return.
                // Same dump as final EXIT but at this earlier return point.
                if (isChap193TargetLB()) {
                    static std::mutex earlyExitMtx;
                    std::lock_guard<std::mutex> lock(earlyExitMtx);
                    static int earlyExitCount = 0;
                    ++earlyExitCount;
                    std::ofstream f(".debug/hashburst_trace.txt", std::ios::app);
                    dumpHeader(f, "EARLY-EXIT", earlyExitCount);
                    dumpLBChain(f);
                    dumpEncodedStatements(f);
                    dumpToBeProved(f);
                    dumpExprOriginMap(f);
                    dumpHashOriginals(f);
                    dumpAdmissionMap(f);
                    dumpEncodedMapMarkers(f);
                    dumpMailIn(f);
                    f.flush();
                }
                return body;
            }

            for (int16_t r = 0; r < totalReqs; ++r) {
                const StaticRequest& req = reqBuf[r];

                // Dependency check via intKnownStatements (O(1) per element)
                bool skip = false;
                for (int16_t s = 0; s < req.count; ++s) {
                    int32_t key = packStatementKey(req.intExprs[s]->originalId,
                                                   req.intExprs[s]->validityId);
                    if (body.intKnownStatements.find(key) == body.intKnownStatements.end()) {
                        skip = true; break;
                    }
                }
                if (skip) continue;

                const std::size_t prevSize = body.intKnownStatements.size();
                this->checkLocalEncodedMemoryStatic(req, body, coreId);
                if (body.intKnownStatements.size() > prevSize) {
                    changed = true;
                }
            }
        }
    }

    // Send to per-core mailboxes for children of 'body'
    this->sendMail(body, body.mailOut, coreId, index, boxes);
    body.mailOut.statements.clear();
    body.mailOut.implications.clear();
    body.mailOut.exprOriginMap.clear();

	reactToHypo(body);

    // EXIT trap — final return. Same dump composition as ENTRY (full).
    if (isChap193TargetLB()) {
        static std::mutex exitMtx;
        std::lock_guard<std::mutex> lock(exitMtx);
        static int exitCount = 0;
        ++exitCount;
        std::ofstream f(".debug/hashburst_trace.txt", std::ios::app);
        dumpHeader(f, "EXIT", exitCount);
        dumpLBChain(f);
        dumpEncodedStatements(f);
        dumpToBeProved(f);
        dumpExprOriginMap(f);
        dumpHashOriginals(f);
        dumpAdmissionMap(f);
        dumpEncodedMapMarkers(f);
        dumpMailIn(f);
        f.flush();
    }

    return body;
}

/// @brief Walk a theorem expression and return the set of args that must
/// stay literal (immutable) — typically anchor slots and external
/// arguments.
///
/// @details
/// Used by the disintegration code to know which args may NOT be
/// renamed during fresh-name minting. Anchor slots (the args of the
/// outermost `Anchor*` operator) are always immutable; external
/// arguments depend on the theorem's structure. The result feeds
/// into `multiplyImplication`'s I-24 free-anchor-param check.
///
/// @param theorem Canonical MPL theorem text.
/// @return Set of argument names that must not be renamed.
/// @see `multiplyImplication` — consumer of the result via I-24.
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
    std::set<std::string> digits = findDigitArgs(theorem, anchorInfo, coreExpressionMap);
    immutables.insert(digits.begin(), digits.end());
    immutables.erase(digit);

    // Propagate immutability forward along the chain
    bool found = true;
    while (found) {
        found = false;

        for (std::size_t i = 0; i < chain.size(); ++i) {
            const std::string& expression = chain[i];
            const std::string coreExpr = ce::extractExpression(expression);

            // Use the pre-populated 'operators' attribute for efficiency
            if (this->operators.find(coreExpr) != this->operators.end()) {

                auto it = this->coreExpressionMap.find(coreExpr);
                assert(it != this->coreExpressionMap.end());

                const ce::CoreExpressionConfig& cfg = it->second;
                const std::vector<std::string> args = ce::getArgs(expression);

                // 1. Check if all configured Inputs are already Immutable
                bool inputsAreImmutable = true;
                for (int idx : cfg.inputIndices) {
                    // BLUNT ASSERT: Fail if config index is out of bounds for this expression
                    assert(idx >= 0 && idx < static_cast<int>(args.size()));

                    if (immutables.find(args[idx]) == immutables.end()) {
                        inputsAreImmutable = false;
                        break;
                    }
                }

                // 2. If inputs are immutable, add configured Outputs to immutables
                if (inputsAreImmutable) {
                    for (int idx : cfg.outputIndices) {
                        // BLUNT ASSERT: Fail if config index is out of bounds
                        assert(idx >= 0 && idx < static_cast<int>(args.size()));

                        const std::string& outputArg = args[idx];
                        if (immutables.find(outputArg) == immutables.end()) {
                            immutables.insert(outputArg);
                            found = true;
                        }
                    }
                }
            }
        }
    }

    return immutables;
}

/// @brief Look up an LB by its global key (the deepest-first chain of
/// `exprKey` values from the LB up to root).
///
/// @details
/// Companion to `getGlobalKey`: round-trip
/// `accessMemory(getGlobalKey(mb), root)` returns a pointer equal to
/// `&mb` for any LB still alive. Walks the LB tree from `bodyOfProves1`
/// step by step, taking the `simpleMap[K]` edge that matches each chain
/// element. Returns `nullptr` if any step doesn't have a matching
/// child — typically because the target LB has been torn down between
/// iterations.
///
/// @param theoremKey     Global key of the target LB (deepest first).
/// @param bodyOfProves1  Root sentinel from which to walk. Typically
///                       `body` for the main LB tree or `ceBody` for
///                       the CE-filter tree.
/// @return Pointer to the matching LB, or `nullptr` on miss.
/// @see `getGlobalKey` — produces the matching key.
Memory* ExpressionAnalyzer::accessMemory(const std::vector<std::string>& theoremKey,
    Memory& bodyOfProves1) {
    Memory* memoryBlock = &bodyOfProves1;
    for (std::size_t i = 0; i < theoremKey.size(); ++i) {
        const std::string& key = theoremKey[i];
        std::map<std::string, Memory*>::iterator it = memoryBlock->simpleMap.find(key);
        if (it == memoryBlock->simpleMap.end() || it->second == NULL) {
            return NULL;
        }
        memoryBlock = it->second; // follow the pointer
    }
    return memoryBlock;
}

/// @brief Try to back-reformulate a theorem's head from its disintegrated
/// form into a higher-level operator head.
///
/// @details
/// One of the three back-paths gated by `!parameters.ban_disintegration`
/// per I-7. The other two are Pass B disintegration and
/// hypo-disintegration. Back-reformulation looks at the head of a theorem
/// and asks whether it could be re-expressed using a configured operator
/// from `coreExpressionMap` — converting, for instance, the disintegrated
/// `(p[a]) ∧ (q[a])` form back into a single-operator equivalent.
///
/// Returns `true` iff the back-reformulation succeeded (i.e. the theorem
/// has a viable higher-level head); `backReformulated` is populated with
/// the rewritten form. Returns `false` if no operator matches, in which
/// case `backReformulated` stays unchanged. The function additionally
/// gates on `parameters.incubator_mode` — non-incubator runs early-exit
/// false because the back-reformulation product is unsound for the
/// real-math output set; incubator artefacts are tagged separately as
/// `incubator back reformulation` and never promoted to the main
/// theorem list.
///
/// @param theorem           Source theorem text.
/// @param backReformulated  Reformulated form on success; untouched on
///                          failure.
/// @return True iff back-reformulation succeeded.
/// @invariant I-7 — guarded by `!parameters.ban_disintegration`.
bool ExpressionAnalyzer::tryBackReformulateOperatorHead(
    const std::string& theorem,
    std::string& backReformulated)
{
    if (!parameters.incubator_mode) return false;

    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

    // Must have exactly 2 chain elements (anchor + operator)
    if (tempChain.size() != 2) return false;

    // First chain element must be an Anchor
    {
        std::string firstName = ce::extractExpression(std::get<0>(tempChain[0]));
        if (firstName.find("Anchor") == std::string::npos) return false;
    }

    // Head must be equality or negated equality
    bool negated = false;
    std::string eqExpr = head;
    if (!head.empty() && head[0] == '!') {
        negated = true;
        eqExpr = head.substr(1);
    }

    std::string headExpr = ce::extractExpression(eqExpr);
    if (headExpr != "=") return false;

    std::vector<std::string> eqArgs = ce::getArgs(eqExpr);
    if (eqArgs.size() != 2) return false;

    // Last chain element must be an operator
    const std::string& lastPremise = std::get<0>(tempChain.back());
    std::string opName = ce::extractExpression(lastPremise);
    auto cfgIt = coreExpressionMap.find(opName);
    if (cfgIt == coreExpressionMap.end()) return false;
    if (cfgIt->second.outputIndices.empty()) return false;  // not an operator

    int outIdx = cfgIt->second.outputIndices[0];  // 0-based
    std::vector<std::string> opArgs = ce::getArgs(lastPremise);
    if (outIdx >= static_cast<int>(opArgs.size())) return false;

    std::string x = opArgs[outIdx];  // the fresh variable in operator

    // Verify: one of the equality args must be x
    std::string a_out;
    if (eqArgs[0] == x) {
        a_out = eqArgs[1];
    } else if (eqArgs[1] == x) {
        a_out = eqArgs[0];
    } else {
        return false;  // x doesn't appear in equality — not our pattern
    }

    // Back-reformulate: replace x with a_out in the operator, drop the equality + inner >
    opArgs[outIdx] = a_out;
    std::string cleanHead = "(" + opName + "[";
    for (size_t i = 0; i < opArgs.size(); ++i) {
        if (i > 0) cleanHead += ",";
        cleanHead += opArgs[i];
    }
    cleanHead += "])";

    // Negate the operator head if the equality was negated
    if (negated) {
        cleanHead = "!" + cleanHead;
    }

    // Rebuild chain without the last element (operator was a premise, now it's the head)
    std::vector<std::string> outerChain;
    for (size_t i = 0; i < tempChain.size() - 1; ++i) {
        outerChain.push_back(std::get<0>(tempChain[i]));
    }

    backReformulated = reconstructImplication(outerChain, cleanHead);
    return true;
}

// analyze_expressions.cpp
// Thread-safe: locks globalTheoremList only when mutating it.
/// @brief Drain the per-thread `updateGlobalDirectTuples` queue into
/// the global theorem list — direct theorem path.
///
/// @details
/// Post-`pool.join()` collector for theorems that were proved by a
/// direct path (rather than via the auxy-indexed indirect path
/// handled by `updateGlobal`). Same I-28 contract: workers never
/// touch the global state during the parallel phase, only enqueue;
/// this function is the single drain point.
///
/// Result: a fresh row in `globalTheoremList` and matching entries
/// in `fullTheoremList` and `cachedProofStacks` so the proof-graph
/// generator can pick the theorem up at export time.
///
/// @param theorem Theorem text being recorded.
/// @param coreId  Producing thread id.
/// @invariant [I-28](../../docs/30_invariants.md#i-28).
/// @see `updateGlobal` — auxy-indexed counterpart.
void ExpressionAnalyzer::updateGlobalDirect(const std::string& theorem, int coreId) {
    // 1) Disintegrate to chain + head
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    Mail mailOut;

    deactivateUnnecessary(theorem, this->body);  // deactivates all blocks tied to this theorem.

    std::string value = ce::disintegrateImplication(theorem, tempChain, this->coreExpressionMap);

    std::vector<std::string> ky;
    ky.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

    // Save original decomposition before sections 3/4 overwrite ky/value
    const std::vector<std::string> origKy = ky;
    const std::string origValue = value;

    // 2) Original implication broadcast deferred until after back-reformulation check (section 5)
    //    If back-reformulation succeeds, only the back-reformulated version is broadcast.
    bool willBackReformulate = false;

    std::vector<std::pair<std::string, std::string>> outTheorems;
    if (reformulateTheorem(theorem, outTheorems)) {
        for (const auto& pair : outTheorems) {
            const std::string& refTheoremRaw = pair.first;
            const std::string& refTheoremCompiled = pair.second;

            {
                std::lock_guard<std::mutex> lock(this->theoremListMutex);
                this->globalTheoremList.emplace_back(refTheoremCompiled, "reformulated statement", theorem, "-1");
            }
            std::cout << refTheoremCompiled << std::endl;

            if (parameters.trackHistory) {
                ExpressionWithValidity ev(refTheoremCompiled, "main");
                addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }

            tempChain.clear();
            std::string valueR = ce::disintegrateImplication(refTheoremCompiled, tempChain, this->coreExpressionMap);

            ky.clear();
            ky.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

            Memory* memoryBlockR = accessMemory(ky, this->body);
            if (memoryBlockR != NULL) {
                std::pair<std::string, std::vector<ExpressionWithValidity>> originR;
                originR.first = "implication";
                originR.second.push_back(ExpressionWithValidity(refTheoremCompiled, "main"));
                for (const std::string& keyElement : ky) {
                    originR.second.emplace_back(keyElement, "main");
                }

                std::set<int> levelsR;
                for (int i = 0; i <= static_cast<int>(ky.size()); ++i) levelsR.insert(i);

                addStatement(valueR, *memoryBlockR, false, levelsR, originR, "main");
            }

            mailOut.implications.insert(
                std::make_tuple(ky, valueR, std::set<std::string>(), std::set<int>(), refTheoremCompiled)
            );

        }

    }

    // 4) Try to create reshuffled/mirrored variant (anchor-first)
    const std::string reshuffledMirrored =
        ce::createReshuffledMirrored(theorem, this->anchorInfo.name, true, this->coreExpressionMap);  // :contentReference[oaicite:3]{index=3}

    if (!reshuffledMirrored.empty()) {
        {
            std::lock_guard<std::mutex> lock(this->theoremListMutex);
            this->globalTheoremList.emplace_back(reshuffledMirrored, "mirrored statement", theorem, "-1");
        }

        if (parameters.trackHistory) {
            ExpressionWithValidity ev(reshuffledMirrored, "main");
            addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        // rebuild chain for the mirrored expr
        tempChain.clear();
        value = ce::disintegrateImplication(reshuffledMirrored, tempChain, this->coreExpressionMap);

        ky.clear();
        ky.reserve(tempChain.size());
        for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

        // Queue the mirrored implication (no addStatement — let it fire naturally via hash burst)
        mailOut.implications.insert(
            std::make_tuple(ky, value, std::set<std::string>(), std::set<int>(), reshuffledMirrored)
        );
    }

    // 5) Incubator: back-reformulate operator-equality pattern
    // Must be before sendMail so the back-reformulated implication gets flushed.
    std::string backReformulated;
    willBackReformulate = tryBackReformulateOperatorHead(theorem, backReformulated);
    if (!willBackReformulate) {
        // No back-reformulation: broadcast and record original theorem
        mailOut.implications.insert(std::make_tuple(origKy, origValue, std::set<std::string>(), std::set<int>(), theorem));
        if (parameters.trackHistory) {
            ExpressionWithValidity ev(theorem, "main");
            addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }
        {
            std::lock_guard<std::mutex> lock(this->theoremListMutex);
            this->globalTheoremList.emplace_back(theorem, "direct", "-1", "-1");
        }
        std::cout << theorem << std::endl;
        checkOrCompletion(theorem, coreId);
    }
    if (willBackReformulate) {
        // Register original as "direct" so it gets a full proof chapter
        mailOut.implications.insert(std::make_tuple(origKy, origValue, std::set<std::string>(), std::set<int>(), theorem));
        if (parameters.trackHistory) {
            ExpressionWithValidity ev(theorem, "main");
            addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }
        {
            std::lock_guard<std::mutex> lock(this->theoremListMutex);
            this->globalTheoremList.emplace_back(theorem, "direct", "-1", "-1");
            this->globalTheoremList.emplace_back(
                backReformulated, "incubator back reformulation", theorem, "-1");
        }
        std::cout << theorem << std::endl;
        std::cout << backReformulated << std::endl;
        checkOrCompletion(theorem, coreId);

        if (parameters.trackHistory) {
            ExpressionWithValidity ev(backReformulated, "main");
            addOrigin(mailOut.exprOriginMap, ev,
                std::make_pair("theorem", std::vector<ExpressionWithValidity>()),
                (parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr));
        }

        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> brChain;
        std::string brHead = ce::disintegrateImplication(backReformulated, brChain, coreExpressionMap);

        std::vector<std::string> brKey;
        brKey.reserve(brChain.size());
        for (std::size_t i = 0; i < brChain.size(); ++i)
            brKey.push_back(std::get<0>(brChain[i]));

        // Queue the back-reformulated implication (no addStatement — let it fire naturally via hash burst)
        mailOut.implications.insert(
            std::make_tuple(brKey, brHead, std::set<std::string>(), std::set<int>(), backReformulated)
        );
    }

    // 6) Flush mail
    assert(coreId >= 0);
    sendMail(this->body, mailOut, coreId, index, boxes);
}

//#pragma optimize("", off)


// Helper to recursively deactivate a block and ONLY its induction-related descendants.
// Helper to recursively deactivate blocks starting from the root.
// Traverses the entire tree and deactivates any node where:
// 1. All its children are inactive.
// 2. It has no pending proofs (toBeProved is empty).
/// @brief Walk the entire LB tree marking inactive LBs as torn down.
///
/// @details
/// Called between outer iterations to reclaim memory for LBs whose
/// purpose is complete. Walks every LB depth-first; when the
/// deactivation predicate fires, sets `isActive = false` and frees
/// per-LB state (intEncodedStatements, hash memories,
/// equivalence-class state). Children of a deactivated LB are also
/// deactivated.
///
/// Cached proof stacks (`cachedProofStacks`) are populated in a
/// pre-teardown pass before this walk so theorems proven inside
/// torn-down LBs still have their proof stack at export time.
///
/// @see `deactivateUnnecessary` — narrower variant that targets a
///      specific theorem's LBs.
/// @see `cachedProofStacks` (member) — per-theorem stack cache.
void ExpressionAnalyzer::deactivateRecursively() {
    // Recursive lambda for post-order traversal
    std::function<void(Memory*)> traverse =
        [&](Memory* node) {
        if (!node) return;

        // 1. Recurse into children first (Post-order)
        for (auto& kv : node->simpleMap) {
            if (kv.second) {
                traverse(kv.second);
            }
        }

        // 2. Check if all children are deactivated
        bool allChildrenDeactivated = true;
        for (const auto& kv : node->simpleMap) {
            Memory* child = kv.second;
            if (child && child->isActive) {
                allChildrenDeactivated = false;
                break;
            }
        }

        // 3. Deactivate node if conditions are met:
        //    - All children are inactive
        //    - No pending proofs in the current node

        int mainValidityCount = 0;
        for (const auto& kv : node->toBeProved) {
            // kv.first is the EncodedExpression key
            if (kv.first.validityName == "main") {
                mainValidityCount++;
            }
        }

        if (allChildrenDeactivated && mainValidityCount == 0 && !node->primedForContradiction) {
            node->isActive = false;

        }
        };

    // Start traversal from the root of the ExpressionAnalyzer
    traverse(&this->body);
}

//#pragma optimize("", off)


/// @brief Deactivate the LBs that were created exclusively for proving
/// the given theorem and are no longer needed.
///
/// @details
/// Targeted variant of `deactivateRecursively`. Used at the end of
/// each theorem's main-batch proof to trim LBs that supported only
/// that theorem's hash bursts. The criterion is: an LB whose chain
/// of ancestors traces back to a unique `theorem`-creation site and
/// whose `localEncodedStatementsSet` is empty after the proof
/// completes.
///
/// Same teardown protocol as `deactivateRecursively` — proof stacks
/// are cached pre-teardown via `cachedProofStacks`.
///
/// @param theorem Theorem text identifying the per-theorem LBs.
/// @param root    Root sentinel from which to walk.
void ExpressionAnalyzer::deactivateUnnecessary(const std::string& theorem, Memory& root) {
    // 1) Collect original indices for this theorem
    std::vector<int> originals;
    originals.reserve(8);
    for (const auto& kv : this->globalDependencies.originalAuxyMap) {
        if (kv.second.expr == theorem) 
            originals.push_back(kv.first);
    }

    // 2) Disintegrate theorem to find its Memory path
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    ce::disintegrateImplication(theorem, tempChain, this->coreExpressionMap);
    if (tempChain.empty()) return;

    // Walk down the chain from root, collecting the blocks on the path
    Memory* node = &root;
    std::vector<Memory*> chainBlocks;
    chainBlocks.reserve(tempChain.size());

    for (const auto& t : tempChain) {
        const std::string& key = std::get<0>(t);
        auto it = node->simpleMap.find(key);
        if (it == node->simpleMap.end() || it->second == nullptr) {
            // Path not present (may have been pruned earlier) — nothing to do safely.
            return;
        }
        node = it->second;
        chainBlocks.push_back(node);
    }

    // Names depend only on the last node on the path
    const std::string zeroArgName = this->findZeroArgName(*node);
    const std::string sName = this->findSName(*node);
    if (zeroArgName.empty() || sName.empty()) {
        // Can't construct the expected keys — bail gracefully.
        return;
    }

    // 3) Deactivate per-original auxiliaries under the final node
    for (int originalIndex : originals) {
        auto itIV = this->globalDependencies.originalInductionVariableMap.find(originalIndex);
        if (itIV == this->globalDependencies.originalInductionVariableMap.end()) continue;

        const std::string& indVar = std::get<0>(itIV->second); // e.g. "i2"
        const std::string& recCounter = std::get<1>(itIV->second); // numeric string
        const std::string  recArg = std::string("rec") + recCounter;

        const std::string k_rec_in2 = "(in2[" + recArg + "," + indVar + "," + sName + "])";
        const std::string k_eq_side = "(=[s(" + recArg + ")," + zeroArgName + "])";

        if (auto it1 = node->simpleMap.find(k_rec_in2); it1 != node->simpleMap.end() && it1->second) {
            it1->second->isActive = false;

        }
        if (auto it2 = node->simpleMap.find(k_eq_side); it2 != node->simpleMap.end() && it2->second) {
            it2->second->isActive = false;

        }
        // No asserts: partial deactivation is still useful/safe.
    }

    // 4) Bubble up: if a block has no active children, deactivate it too
    for (auto rit = chainBlocks.rbegin(); rit != chainBlocks.rend(); ++rit) {
        Memory* block = *rit;
        bool anyActiveChild = false;
        for (const auto& kv : block->simpleMap) {
            Memory* child = kv.second;
            if (child && child->isActive) { anyActiveChild = true; break; }
        }
        if (!anyActiveChild && block->toBeProved.size() == 0)
        {
            block->isActive = false;
        }
    }

	deactivateRecursively();  // Further prune the tree globally
}



// analyze_expressions.cpp
// Thread-safe: locks globalDependencies briefly, and locks globalTheoremList only when pushing.
/// @brief Drain the per-thread `updateGlobalTuples` queue into the
/// global theorem list / dependency tracker.
///
/// @details
/// Post-`pool.join()` collector for the indirect (auxy-indexed) path.
/// During the parallel hash burst, workers cannot write to
/// `globalTheoremList` directly per
/// [I-28](../../docs/30_invariants.md#i-28); they enqueue tuples
/// `(auxyIndex, allLevelsInvolved, coreId)` instead. This function
/// is the single drain point that locks `theoremListMutex` /
/// `dependenciesMutex` once and folds every queued tuple into the
/// global side-tables.
///
/// @param auxyIndex          Auxiliary index identifying the theorem.
/// @param allLevelsInvolved  Flag: theorem spans every LB level.
/// @param coreId             Producing thread id (purely informational).
/// @invariant [I-28](../../docs/30_invariants.md#i-28) — drain runs
///            only after `pool.join()`.
/// @see `updateGlobalDirect` — direct (non-auxy) counterpart.
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
    Memory* memoryBlock = accessMemory(ky, this->body);
    if (memoryBlock != NULL) {
        // --- Induction typing gate ---
        // Before promoting an induction theorem, verify the induction variable is
        // constrained to N (anchor_args[0]). A sound induction requires
        // `(in[indVar,N])` derivable from the current chain; without it, the
        // theorem universally quantifies over everything rather than over N, and
        // the implicit-typing inference of the original induction-scheduling is
        // unsound. See docs/induction_typing_plan.md.
        if (!indVar.empty()) {
            const std::string anchorN = findNName(*memoryBlock);
            const std::string typingGoal =
                std::string("(in[") + indVar + "," + anchorN + "])";

            if (memoryBlock->wholeExpressions.find(EncodedExpression(typingGoal, "main"))
                == memoryBlock->wholeExpressions.end()) {
                // Typing not established — reject induction promotion silently.
                return;
            }
        }

        std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>>>::iterator it = memoryBlock->toBeProved.find(EncodedExpression(value, "main"));
        if (it != memoryBlock->toBeProved.end()) {
            // Python asserts this is an empty set
            assert(std::get<std::set<int>>(it->second).empty());
            memoryBlock->toBeProved.erase(it);

            // Build origin and levels
            std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
            origin.first = "implication";
            origin.second.push_back(ExpressionWithValidity(expr, "main"));
            for (const std::string& keyElement : ky) {
                // Construct the object in-place with the key and "main"
                origin.second.emplace_back(keyElement, "main");
            }

            std::set<int> levels;
            for (int i = 0; i <= static_cast<int>(ky.size()); ++i) levels.insert(i);

            // Add the head as a statement at this memory block
            addStatement(value, *memoryBlock, false, levels, origin, "main");

            // Queue the implication in global mail_out
            mailOut.implications.insert(std::make_tuple(ky, value, std::set<std::string>(), std::set<int>(), expr));

            if (parameters.trackHistory) {
                ExpressionWithValidity ev(expr, "main");
                addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }

            // ---- record in globalTheoremList (short critical section) ----
            {
                std::lock_guard<std::mutex> lock(this->theoremListMutex);
                this->globalTheoremList.emplace_back(expr, "induction", indVar, recCounter);
            }
            std::cout << expr << std::endl;
            checkOrCompletion(expr, coreId);

            std::vector<std::pair<std::string, std::string>> outTheorems;
            if (reformulateTheorem(expr, outTheorems)) {
                for (const auto& pair : outTheorems) {
                    const std::string& refTheoremRaw = pair.first;
                    const std::string& refTheoremCompiled = pair.second;

                    {
                        std::lock_guard<std::mutex> lock(this->theoremListMutex);
                        this->globalTheoremList.emplace_back(refTheoremCompiled, "reformulated statement", expr, "-1");
                    }
                    std::cout << refTheoremCompiled << std::endl;

                    if (parameters.trackHistory) {
                        ExpressionWithValidity ev(refTheoremCompiled, "main");
                        addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    }

                    tempChain.clear();
                    std::string valueR = ce::disintegrateImplication(refTheoremCompiled, tempChain, this->coreExpressionMap);

                    ky.clear();
                    ky.reserve(tempChain.size());
                    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

                    Memory* memoryBlockR = accessMemory(ky, this->body);
                    if (memoryBlockR != NULL) {
                        std::pair<std::string, std::vector<ExpressionWithValidity>> originR;
                        originR.first = "implication";
                        originR.second.push_back(ExpressionWithValidity(refTheoremCompiled, "main"));
                        for (const std::string& keyElement : ky) {
                            originR.second.emplace_back(keyElement, "main");
                        }

                        std::set<int> levelsR;
                        for (int i = 0; i <= static_cast<int>(ky.size()); ++i) levelsR.insert(i);

                        addStatement(valueR, *memoryBlockR, false, levelsR, originR, "main");
                    }

                    mailOut.implications.insert(
                        std::make_tuple(ky, valueR, std::set<std::string>(), std::set<int>(), refTheoremCompiled)
                    );
                }
            }


            // Try anchor-first reshuffled / mirrored variant
            const std::string reshuffledMirrored =
                ce::createReshuffledMirrored(expr, this->anchorInfo.name, true, this->coreExpressionMap);  // :contentReference[oaicite:1]{index=1}

            if (!reshuffledMirrored.empty()) {
                {
                    std::lock_guard<std::mutex> lock(this->theoremListMutex);
                    this->globalTheoremList.emplace_back(reshuffledMirrored, "mirrored statement", expr, "-1");
                }

                if (parameters.trackHistory) {
                    ExpressionWithValidity ev(reshuffledMirrored, "main");
                    addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                }

                // Disintegrate mirrored and try to add its head too
                tempChain.clear();
                std::string valueM =
                    ce::disintegrateImplication(reshuffledMirrored, tempChain, this->coreExpressionMap);

                ky.clear();
                ky.reserve(tempChain.size());
                for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

                // Queue the mirrored implication (no addStatement — let it fire naturally via hash burst)
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


/// @brief Update the admission map after a hash-engine deposit — record
/// the depth/secondary-iteration budgets the rule consumes when fired.
///
/// @details
/// Called from `addToHashMemory` (Path 1) when
/// `performAdmissionMapUpdate == true`. Walks the per-(expression,
/// validity) entry of `mb.overallHashMemory.admissionMap`, finds or
/// creates the matching `AdmissionMapValue`, and folds in the new
/// depth / secondary-number budgets. The result is the cap that
/// `isAdmitted` uses to gate firing.
///
/// Per [I-22](../../docs/30_invariants.md#i-22), revival from
/// `rejectedMapIntegration` does NOT clean the admission entry — the
/// two side-tables are intentionally not coupled, so this function
/// only adds; existing entries are preserved.
///
/// @param expr   Expression whose admission record is being updated.
/// @param mb     Owning LB.
/// @param ...    Routing parameters.
/// @invariant [I-22](../../docs/30_invariants.md#i-22) — additive only.
inline void ExpressionAnalyzer::updateAdmissionMap3(const std::string& expr,
    Memory& memoryBlock,
    int maxAdmissionDepth,
    int maxSecondaryNumber,
    bool partOfRecursion) {
    // 1) Compute digit arguments of the expression (uses analyzer's anchor)
    std::set<std::string> digitArgs = findDigitArgs(expr, this->anchorInfo, this->coreExpressionMap);

    // 2) Walk up the Memory chain to find the first block whose exprKey shares a digit arg
    Memory* tempMb = &memoryBlock;
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

                this->updateAdmissionMap(*tempMb,
                    key,
                    remainingArgs,
                    maxAdmissionDepth,
                    maxSecondaryNumber,
                    partOfRecursion,
                    "main");

                
            }
            break;
        }
        else {
            if (tempMb->parentMemory != NULL) {
                tempMb = tempMb->parentMemory;
            }
            else {
                break;
            }
        }
    }
}


/// @brief Negate an MPL expression — wrap with `!(...)` or peel off an
/// existing `!(...)`.
///
/// @details
/// Idempotent up to one application: `negate(negate(x)) == x` for
/// well-formed input. Used by `addStatement` and the contradiction
/// machinery to produce the negated form of a candidate without
/// re-parsing.
///
/// **Per [OPEN-MPL-1](../../docs/SwDD.md#remaining-open) (resolved
/// partial)**: double-negation cancellation happens at two sites; this
/// is one of them. There is no universal normaliser, so a raw `!!X`
/// in an external theorem would hash distinctly from `X` if it ever
/// reaches the hash engine without going through `negate`.
///
/// @param expr Canonical MPL text. May or may not start with `!(`.
/// @return Negated form.
std::string ExpressionAnalyzer::negate(std::string expr) {
    if (!expr.empty() && expr.front() == '!') return expr.substr(1);
    return "!" + expr;
}

//#pragma optimize("", off)

/// @brief Install an anchor expression and register its slot mapping in the
/// LB.
///
/// @details
/// An anchor expression like `(AnchorPeano[N, i, s, p, m, j])` declares
/// the per-batch typing of every anchor-slot variable. `handleAnchor`:
/// 1. Records the anchor's slot-to-variable map so subsequent
///    statements can reference slots by canonical name.
/// 2. Per [I-11](../../docs/30_invariants.md#i-11), asserts that no
///    anchor-slot name appears in any inner `>[...]` bound-variable
///    list — anchor slots are GLOBAL pinned variables and binding them
///    inside an inner quantifier is malformed.
/// 3. Emits the `x_<varname>` prefixed marker statements that
///    [OPEN-14 in `docs/SwDD.md`](../../docs/SwDD.md#open-questions)
///    documents.
///
/// @param expr           Anchor expression text.
/// @param memoryBlock    Owning LB.
/// @param involvedLevels LB levels.
/// @invariant [I-11](../../docs/30_invariants.md#i-11) — anchor slot
///            names not in inner `>[...]` lists.
void ExpressionAnalyzer::handleAnchor(const std::string& expr, Memory& memoryBlock, const std::set<int>& involvedLevels)
{   
    std::string anchorCheck = "(" + this->anchorInfo.name;

    if (expr.rfind(anchorCheck, 0) == 0)
    {
        if (memoryBlock.exprKey.rfind("(Anchor", 0) != 0)
        {
            return;
        }

        std::vector<std::string> args = ce::getArgs(expr);
        std::string coreExpr = ce::extractExpression(expr);

        std::map<std::string, std::string> replacementMap;
        for (const auto& [slot, pattern] : anchorInfo.definitionSets)
        {
            if (pattern == "(1)")
            {
                int index = std::stoi(slot) - 1;

                if (args[index].rfind("x", 0) == 0)
                {
					return;
                }

                replacementMap.insert(std::make_pair(args[index], "x" + args[index]));
            }
        }

        std::string replacedAnchor = ce::replaceKeysInString(expr, replacementMap);
		EncodedExpression encReplacedAnchor(replacedAnchor, "main");
		ExpressionWithValidity evReplacedAnchor(replacedAnchor, "main");

        memoryBlock.localEncodedStatements.push_back(EncodedExpression(replacedAnchor, "main"));
        memoryBlock.localEncodedStatementsSet.insert(EncodedExpression(replacedAnchor, "main"));
        memoryBlock.localEncodedStatementsDelta.push_back(EncodedExpression(replacedAnchor, "main"));
        memoryBlock.encodedStatements.push_back(EncodedExpression(replacedAnchor, "main"));
        { IntEncodedExpr ie = encodeExpression(memoryBlock.encodedStatements.back(), memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);
          memoryBlock.intLocalEncodedStatements.push_back(ie);
          memoryBlock.intLocalEncodedStatementsDelta.push_back(ie); }
        memoryBlock.statementLevelsMap[encReplacedAnchor] = involvedLevels;
        memoryBlock.intKnownStatements.insert(packStatementKey(
            memoryBlock.nameMap.encode(encReplacedAnchor.original),
            memoryBlock.nameMap.encode(encReplacedAnchor.validityName)));
        memoryBlock.wholeExpressions.insert(EncodedExpression(replacedAnchor, "main"));

        std::pair<std::string, std::vector<ExpressionWithValidity>> handledOrigin;
        handledOrigin.first = "anchor handling";
		handledOrigin.second.push_back(ExpressionWithValidity(expr, "main"));

        addOrigin(memoryBlock.exprOriginMap, evReplacedAnchor, std::make_pair("anchor handling", std::vector<ExpressionWithValidity>{ ExpressionWithValidity(expr, "main") }), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

        memoryBlock.mailOut.statements.insert(std::make_pair(ExpressionWithValidity(replacedAnchor, "main"), involvedLevels));

        addOrigin(memoryBlock.mailOut.exprOriginMap, evReplacedAnchor, handledOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
/// @brief Predicate: is the given expression allowed to be mailed to
/// child LBs from this one?
///
/// @details
/// Mail-out gating per [I-26](../../docs/30_invariants.md#i-26):
/// statements/implications mailed out must be MAIN-validity-only;
/// the exprOriginMap mirror is ALL-SCOPES. This function enforces
/// the validity-side gate by reading the expression's scope and the
/// LB's `canBeSentSet` / `canBeSentMarkerSet` filters. Returns
/// false if the expression carries a non-`"main"` scope on the
/// implication/statement channel.
///
/// @param expression  Canonical MPL expression text plus scope.
/// @param body        Owning LB whose `canBeSentSet` defines the
///                    mail-eligibility filters.
/// @return True iff the expression is mailable on the routing channel.
/// @invariant [I-26](../../docs/30_invariants.md#i-26).
bool ExpressionAnalyzer::allowedForMail(const std::string& expression, Memory& body) {
    static const std::regex reInt(R"(int_lev_\d+_\d+)");
    std::smatch match;

    if (!std::regex_search(expression, match, reInt)) {
        return true;
    }

    if (body.canBeSentSet.find(expression) != body.canBeSentSet.end()) {
        return true;
    }

    std::set<std::string> intVars;
    auto begin = std::sregex_iterator(expression.begin(), expression.end(), reInt);
    auto end = std::sregex_iterator();
    for (std::sregex_iterator i = begin; i != end; ++i) {
        intVars.insert(i->str());
    }

    if (intVars.size() != 1) {
        return false;
    }

    if (expression.rfind("(>", 0) == 0) {
        return false;
    }

    // Check if replaced with marker it is included in Memory::canBeSentMarkerSet
    std::string theVar = *intVars.begin();
    std::map<std::string, std::string> replacementMap;
    replacementMap[theVar] = "marker";

    std::string markerExpr = ce::replaceKeysInString(expression, replacementMap);

    if (body.canBeSentMarkerSet.find(markerExpr) != body.canBeSentMarkerSet.end()) {
        std::string core = ce::extractExpression(expression);
        auto itComp = compiledExpressions.find(core);
        assert(itComp != compiledExpressions.end());
        if (itComp->second.category != "atomic") {
            return true;
        }
    }

    return false;
}

std::set<std::string> ExpressionAnalyzer::extractRemainingArgs(const std::string& str) {
    std::set<std::string> remainingArgs;

    // 1. Check >[...] blocks (Strict Forbidden Zone for "u_")
    {
        const static std::regex re1(R"(>\[([^\]]*)\])");
        std::sregex_iterator it(str.begin(), str.end(), re1);
        std::sregex_iterator end;

        for (; it != end; ++it) {
            const std::string inside = (*it)[1].str();
            std::size_t start = 0;
            while (start <= inside.size()) {
                std::size_t pos = inside.find(',', start);
                std::string token = ce::trimCopy(inside.substr(
                    start, (pos == std::string::npos ? inside.size() : pos) - start));

                if (!token.empty()) {
                    // CRITICAL ASSERTION: No "u_" allowed here at all
                    bool hasUPrefix = startsWith(token, "u_", 2);
                    assert(!hasUPrefix && "Violation: Argument starting with 'u_' found inside >[...] block");
                }

                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        }
    }

    // 2. Collect "u_" args from [...] blocks (NOT preceded by '>')
    {
        const static std::regex re2(R"(\[([^\]]*)\])");
        std::sregex_iterator it(str.begin(), str.end(), re2);
        std::sregex_iterator end;

        for (; it != end; ++it) {
            const std::size_t lpos = static_cast<std::size_t>((*it).position());

            // Skip if preceded by '>' (handled above)
            if (lpos > 0 && str[lpos - 1] == '>') {
                continue;
            }

            const std::string inside = (*it)[1].str();
            std::size_t start = 0;
            while (start <= inside.size()) {
                std::size_t pos = inside.find(',', start);
                std::string token = ce::trimCopy(inside.substr(
                    start, (pos == std::string::npos ? inside.size() : pos) - start));

                // Collect only if it starts with "u_"
                if (startsWith(token, "u_", 2)) {

                    // Count occurrences of "u_" to ensure it only appears once (at the start)
                    int uCount = 0;
                    std::size_t p = 0;
                    while ((p = token.find("u_", p)) != std::string::npos) {
                        uCount++;
                        p += 2;
                    }
                    assert(uCount == 1 && "Argument starting with 'u_' must not contain additional 'u_' substrings");

                    // Insert without "u_" prefix
                    remainingArgs.insert(token.substr(2));
                }

                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        }
    }

    return remainingArgs;
}

/// @brief Produce reformulated variants of a theorem — alternative
/// equivalent expressions that may be easier for the prover to absorb.
///
/// @details
/// Walks the theorem looking for reformulation opportunities:
/// - swap commutative-operator argument order,
/// - apply known operator-equivalences from `coreExpressionMap`,
/// - extract sub-implications that are independently reformulable.
///
/// Each successful reformulation becomes a `(tag, expression)` pair
/// in `outTheorems`. The tag distinguishes the reformulation type
/// (`reformulated from`, `mirrored from`, `incubator back reformulation`,
/// etc.). The tag is load-bearing: `reformulated from` is the
/// real-math (sound) provenance, while `incubator back reformulation`
/// is an incubator-only artefact that is unsound and must NEVER be
/// folded into a `reformulated from`. Downstream consumers must
/// preserve this distinction.
///
/// @param theorem      Source theorem text.
/// @param outTheorems  Output vector. Each pair is
///                     `(reformulation tag, reformulated text)`.
/// @return True iff at least one reformulation was produced.
bool ExpressionAnalyzer::reformulateTheorem(const std::string& theorem, std::vector<std::pair<std::string, std::string>>& outTheorems) {
    if (parameters.ban_disintegration) return false;
    
    bool foundAny = false;

    // 1. Disintegrate the original theorem
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> originalChain;
    std::string head = ce::disintegrateImplication(theorem, originalChain, this->coreExpressionMap);

    // Disintegration safety check
    if (originalChain.empty()) {
        return false;
    }

    // 2. Identify indices of all Definitions in the chain
    std::vector<size_t> defIndices;
    for (size_t i = 0; i < originalChain.size(); ++i) {
        std::string core = ce::extractExpressionUniversal(std::get<0>(originalChain[i]));
        auto it = compiledExpressions.find(core);
        if (it != compiledExpressions.end() && !it->second.definedSet.empty()) {
            defIndices.push_back(i);
        }
    }

    if (defIndices.empty()) return false;

    // 3. Iterate through each definition to see if it qualifies for reformulation
    for (size_t targetDefIdx : defIndices) {

        // A. Construct Reshuffled Key: [Others..., TargetDef]
        // We preserve the relative order of the non-target premises
        std::vector<std::string> reshuffledKey;
        reshuffledKey.reserve(originalChain.size());

        for (size_t i = 0; i < originalChain.size(); ++i) {
            if (i != targetDefIdx) {
                reshuffledKey.push_back(std::get<0>(originalChain[i]));
            }
        }
        // Add the target definition at the VERY END
        reshuffledKey.push_back(std::get<0>(originalChain[targetDefIdx]));

        // B. Reconstruct the Full Theorem with the ORIGINAL Head
        // This is crucial: we need the system to calculate the binding scope 
        // for the new order of premises.
        std::string tempTheorem = reconstructImplication(reshuffledKey, head);

        // C. Disintegrate again to inspect the binding of the last link (our TargetDef)
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
        std::string tempHead = ce::disintegrateImplication(tempTheorem, tempChain, this->coreExpressionMap);

        // Safety checks
        if (tempChain.empty() || tempHead != head) continue;

        // The last link in tempChain corresponds to reshuffledKey.back() (our TargetDef)
        const auto& lastLink = tempChain.back();
        const std::string& lastDefExpr = std::get<0>(lastLink);
        const std::vector<std::string>& boundVars = std::get<1>(lastLink);

        // D. Verify Logic: Bound Variable == Set Argument
        std::string core = ce::extractExpressionUniversal(lastDefExpr);
        auto leIt = compiledExpressions.find(core);
        if (leIt == compiledExpressions.end()) continue;

        // Find the expected argument position from the signature
        std::vector<std::string> sigArgs = ce::getArgs(leIt->second.signature);
        auto itSig = std::find(sigArgs.begin(), sigArgs.end(), leIt->second.definedSet);
        if (itSig == sigArgs.end()) continue;

        size_t argIndex = std::distance(sigArgs.begin(), itSig);
        std::vector<std::string> actualArgs = ce::getArgs(lastDefExpr);
        if (argIndex >= actualArgs.size()) continue;

        std::string expectedArg = actualArgs[argIndex];

        // CHECK: Is this definition acting as the peeling layer?
        // The last implication must bind exactly one variable, and it must be the set argument.
        if (boundVars.size() == 1 && boundVars[0] == expectedArg) {

            // FOUND VALID REFORMULATION!

            // E. Apply Transformation
            // Structure: (Premises... -> !( > [vars] LastDef !Head ))

            // Prepare Negated Head
            std::string negatedHead;
            if (head.size() > 1 && head[0] == '!') {
                negatedHead = head.substr(1);
            }
            else {
                negatedHead = "!" + head;
            }

            // Build the Special Innermost Tail
            std::string varsStr = boundVars[0];

            // 1. Raw Version (Original Code)
            std::string rawInnerTail = "!(>[" + varsStr + "]" + lastDefExpr + negatedHead + ")";

            // 2. Compiled Version (New Logic)
            std::string compiledInnerTail = this->compileCoreExpressionMapCore(rawInnerTail, this->implCounter, this->existenceCounter, this->andCounter, this->orCounter, this->variableCounter);

            // Wrap the remaining premises (Standard Implication)
            std::string currentTailRaw = rawInnerTail;
            std::string currentTailCompiled = compiledInnerTail;

            for (int i = static_cast<int>(tempChain.size()) - 2; i >= 0; --i) {
                const auto& link = tempChain[i];
                const std::string& premise = std::get<0>(link);
                const std::vector<std::string>& vars = std::get<1>(link);

                std::string vStr;
                for (size_t v = 0; v < vars.size(); ++v) {
                    if (v > 0) vStr += ",";
                    vStr += vars[v];
                }

                std::string wrapper = "(>[" + vStr + "]" + premise;
                currentTailRaw = wrapper + currentTailRaw + ")";
                currentTailCompiled = wrapper + currentTailCompiled + ")";
            }

            outTheorems.push_back(std::make_pair(currentTailRaw, currentTailCompiled));
            foundAny = true;
        }
    }

    return foundAny;
}
// Turn optimizations OFF for just this section
//#pragma optimize("", off)

/// @brief Install an equality `(=[a,b])` into an LB.
///
/// @details
/// - Asserts `args[0] != args[1]` per I-9. A firing assert here means an
///   upstream produced a trivial reflexive equality, which the
///   conjecturer / prover should never emit (per I-8 trivial equality
///   is forbidden in heads).
/// - Searches `memoryBlock.equivalenceClassesMap` for an existing class
///   containing either `args[0]` or `args[1]`; if found, merges the
///   other arg into the class via `mergeTwoEquivalenceClasses` per I-33.
/// - Otherwise creates a fresh
///   `EquivalenceClass{ {args[0], args[1]}, ... }` and seeds
///   `equalityLevelsMap` / `equalityOriginMap`.
/// - Emits the canonical mirror `(=[args[1], args[0]])` as a separate
///   statement so the verifier sees both directions; the per-pair
///   `args[0] != args[1]` assert keeps this from looping.
///
/// @param expr         Equality expression text.
/// @param memoryBlock  Owning LB.
/// @param local        True if the deposit is local-only (no propagation
///                     to the overall hash memory).
/// @param levels       LB levels at which the equality is admissible.
/// @param ...          Routing parameters.
/// @invariant I-9 — mirror guard.
/// @invariant I-33 — class-merge cross-vN preconditions.
/// @see `addNegatedEquality` — negated counterpart.
void ExpressionAnalyzer::addEquality(const std::string& expr,
    Memory& memoryBlock,
    bool local,
    const std::set<int>& levels,
    const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
    const std::string& validityName,
    bool allowSymmetry)
{


    // Reconstruct these objects locally
    EncodedExpression encodedExpr(expr, validityName);
    ExpressionWithValidity exprWithValidity(expr, validityName);

    if (memoryBlock.wholeExpressions.find(encodedExpr) == memoryBlock.wholeExpressions.end())
    {
        memoryBlock.wholeExpressions.insert(EncodedExpression(expr, validityName));


        // 1. Register the original equality
        memoryBlock.statementLevelsMap[encodedExpr] = levels;
        memoryBlock.intKnownStatements.insert(packStatementKey(
            memoryBlock.nameMap.encode(encodedExpr.original),
            memoryBlock.nameMap.encode(encodedExpr.validityName)));

        if (parameters.trackHistory) {
            addOrigin(memoryBlock.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        memoryBlock.encodedStatements.push_back(encodedExpr);
        { IntEncodedExpr ie = encodeExpression(encodedExpr, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);

        if (local) {
            memoryBlock.localEncodedStatements.push_back(encodedExpr);
            memoryBlock.localEncodedStatementsSet.insert(encodedExpr);
            memoryBlock.localEncodedStatementsDelta.push_back(encodedExpr);
            memoryBlock.intLocalEncodedStatements.push_back(ie);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);

            if (validityName == "main") {
                memoryBlock.mailOut.statements.insert(std::make_pair(ExpressionWithValidity(expr, "main"), levels));
            }
            if (parameters.trackHistory) {
                addOrigin(memoryBlock.mailOut.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }
        }
        } // close IntEncodedExpr ie scope

        if (allowSymmetry) {
        // 2. Register the MIRRORED equality
        std::vector<std::string> args = ce::getArgs(expr);
        assert(args.size() == 2);

        if (args[0] != args[1]) {
        std::string mirrored = "(=[" + args[1] + "," + args[0] + "])";

        EncodedExpression encodedMirrored(mirrored, validityName);
        ExpressionWithValidity mirroredWithValidity(mirrored, validityName);

        assert(memoryBlock.wholeExpressions.find(encodedMirrored) == memoryBlock.wholeExpressions.end());
        memoryBlock.wholeExpressions.insert(EncodedExpression(mirrored, validityName));

        memoryBlock.statementLevelsMap[encodedMirrored] = levels;
        memoryBlock.intKnownStatements.insert(packStatementKey(
            memoryBlock.nameMap.encode(encodedMirrored.original),
            memoryBlock.nameMap.encode(encodedMirrored.validityName)));

        if (parameters.trackHistory) {
            std::pair<std::string, std::vector<ExpressionWithValidity>> mirroredOrigin;
            mirroredOrigin.first = "symmetry of equality";
            mirroredOrigin.second.push_back(exprWithValidity); // Derived from original

            addOrigin(memoryBlock.exprOriginMap, mirroredWithValidity, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap, mirroredWithValidity, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        memoryBlock.encodedStatements.push_back(encodedMirrored);
        { IntEncodedExpr ieM = encodeExpression(encodedMirrored, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ieM);

        if (local) {
            memoryBlock.localEncodedStatements.push_back(encodedMirrored);
            memoryBlock.localEncodedStatementsSet.insert(encodedMirrored);
            memoryBlock.localEncodedStatementsDelta.push_back(encodedMirrored);
            memoryBlock.intLocalEncodedStatements.push_back(ieM);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ieM);

            if (validityName == "main") {
                memoryBlock.mailOut.statements.insert(std::make_pair(ExpressionWithValidity(mirrored, "main"), levels));
            }
            // Origin for mirrored already handled in trackHistory block above
        }
        } // close IntEncodedExpr ieM scope
        } // end if args[0] != args[1]
        } // end if allowSymmetry
    }
}

//#pragma optimize("", off)

/// @brief Install a negated equality `!(=[a,b])` into an LB.
///
/// @details
/// Per I-12, `!(=[a,b])` expands **one-sidedly** via equivalence classes
/// — never as a symmetric cross-product. The expansion picks the
/// deeper-scoped variable as the substitution target so the resulting
/// negated expression sits at the union scope of the original and the
/// class's scope.
///
/// Concrete: if `b` is in a class `C = {b, c}` at scope `S`, then
/// `!(=[a, b])` produces `!(=[a, c])` at the deeper of (current scope,
/// S). It does NOT also produce `!(=[c, a])` — that would be the
/// symmetric counterpart, which the one-sided rule forbids.
///
/// @param expr         Negated equality expression text.
/// @param memoryBlock  Owning LB.
/// @param local        True if the deposit is local-only (no propagation
///                     to the overall hash memory).
/// @param levels       LB levels at which the negated equality is
///                     admissible.
/// @param ...          Routing parameters.
/// @invariant I-12 — one-sided expansion via equivalence classes; no
///            symmetric cross-product.
/// @see `addEquality` — positive counterpart.
/// @see `applyEquivalenceClassToNegatedEquality` — class-driven sibling
///      rewrite.
void ExpressionAnalyzer::addNegatedEquality(const std::string& expr,
    Memory& memoryBlock,
    bool local,
    const std::set<int>& levels,
    const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
    const std::string& validityName,
    bool allowSymmetry)
{
    // Mirror of addEquality for negated form !(=[a,b]).
    // Pair invariant: original and mirror are registered together or not at all.
    // Sole gateway — any external path that inserts a negated equality into
    // wholeExpressions without passing through here breaks the invariant and
    // will trip the mirror-absence assert below.

    EncodedExpression encodedExpr(expr, validityName);
    ExpressionWithValidity exprWithValidity(expr, validityName);

    if (memoryBlock.wholeExpressions.find(encodedExpr) == memoryBlock.wholeExpressions.end())
    {
        memoryBlock.wholeExpressions.insert(EncodedExpression(expr, validityName));

        // 1. Register the original negated equality
        memoryBlock.statementLevelsMap[encodedExpr] = levels;
        memoryBlock.intKnownStatements.insert(packStatementKey(
            memoryBlock.nameMap.encode(encodedExpr.original),
            memoryBlock.nameMap.encode(encodedExpr.validityName)));

        if (parameters.trackHistory) {
            addOrigin(memoryBlock.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        memoryBlock.encodedStatements.push_back(encodedExpr);
        { IntEncodedExpr ie = encodeExpression(encodedExpr, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);

        if (local) {
            memoryBlock.localEncodedStatements.push_back(encodedExpr);
            memoryBlock.localEncodedStatementsSet.insert(encodedExpr);
            memoryBlock.localEncodedStatementsDelta.push_back(encodedExpr);
            memoryBlock.intLocalEncodedStatements.push_back(ie);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);

            if (validityName == "main") {
                memoryBlock.mailOut.statements.insert(std::make_pair(ExpressionWithValidity(expr, "main"), levels));
            }
            if (parameters.trackHistory) {
                addOrigin(memoryBlock.mailOut.exprOriginMap, exprWithValidity, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }
        }
        } // close IntEncodedExpr ie scope

        if (allowSymmetry) {
        // 2. Register the MIRRORED negated equality
        std::vector<std::string> args = ce::getArgs(expr);
        assert(args.size() == 2);

        if (args[0] != args[1]) {
        std::string mirrored = "!(=[" + args[1] + "," + args[0] + "])";

        EncodedExpression encodedMirrored(mirrored, validityName);
        ExpressionWithValidity mirroredWithValidity(mirrored, validityName);

        assert(memoryBlock.wholeExpressions.find(encodedMirrored) == memoryBlock.wholeExpressions.end());
        memoryBlock.wholeExpressions.insert(EncodedExpression(mirrored, validityName));

        memoryBlock.statementLevelsMap[encodedMirrored] = levels;
        memoryBlock.intKnownStatements.insert(packStatementKey(
            memoryBlock.nameMap.encode(encodedMirrored.original),
            memoryBlock.nameMap.encode(encodedMirrored.validityName)));

        if (parameters.trackHistory) {
            std::pair<std::string, std::vector<ExpressionWithValidity>> mirroredOrigin;
            mirroredOrigin.first = "symmetry of inequality";
            mirroredOrigin.second.push_back(exprWithValidity); // Derived from original

            addOrigin(memoryBlock.exprOriginMap, mirroredWithValidity, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap, mirroredWithValidity, mirroredOrigin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        memoryBlock.encodedStatements.push_back(encodedMirrored);
        { IntEncodedExpr ieM = encodeExpression(encodedMirrored, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ieM);

        if (local) {
            memoryBlock.localEncodedStatements.push_back(encodedMirrored);
            memoryBlock.localEncodedStatementsSet.insert(encodedMirrored);
            memoryBlock.localEncodedStatementsDelta.push_back(encodedMirrored);
            memoryBlock.intLocalEncodedStatements.push_back(ieM);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ieM);

            if (validityName == "main") {
                memoryBlock.mailOut.statements.insert(std::make_pair(ExpressionWithValidity(mirrored, "main"), levels));
            }
            // Origin for mirrored already handled in trackHistory block above
        }
        } // close IntEncodedExpr ieM scope
        } // end if args[0] != args[1]
        } // end if allowSymmetry
    }
}

//#pragma optimize("", off)

void ExpressionAnalyzer::checkNecessityForEquality(const std::string& inputExprStr, Memory& mb, std::string validityName) {

    if (parameters.ban_disintegration) return;
    
    // Use special function to make all args "u_"
    std::string genericInputStr = prefixArgumentsWithU(inputExprStr);

    // Parse the input expression string
    std::string inputName = ce::extractExpression(genericInputStr);
    std::vector<std::string> inputArgs = ce::getArgs(genericInputStr);

    // Iterate through all original implications stored in localMemory
    for (const std::vector<std::string>& chain : mb.overallHashMemory.originals) {
        if (chain.empty()) continue;

        // "Head" is the last element of the implication chain
        std::string headStr = chain.back();
        std::string headName = ce::extractExpression(headStr);
        std::vector<std::string> headArgs = ce::getArgs(headStr);

        // Structural check: Name and Arity must match the input expression
        if (headName != inputName) continue;
        if (headArgs.size() != inputArgs.size()) continue;

        // Check if head is "Generic except for one arg X"
        // Condition: Exactly one argument in head does NOT start with "u_" (the constant).
        int constantIndex = -1;
        int constantCount = 0;

        for (size_t i = 0; i < headArgs.size(); ++i) {
            // Check if argument starts with "u_"
            bool isVar = (headArgs[i].size() >= 2 && headArgs[i][0] == 'u' && headArgs[i][1] == '_');

            if (!isVar) {
                constantCount++;
                constantIndex = static_cast<int>(i);
            }
        }

        // Requirement: Exactly one arg is not u_
        if (constantCount != 1) continue;

        // NEW: Check that definition set of this arg is (1)
        auto itCfg = this->coreExpressionMap.find(headName);
        if (itCfg == this->coreExpressionMap.end()) continue;
        const auto& cfg = itCfg->second;

        // definitionSets keys are 1-based strings ("1", "2", etc.)
        std::string argPos = std::to_string(constantIndex + 1);
        auto itDS = cfg.definitionSets.find(argPos);

        // FIX: Access .first because definitionSets values are pair<string, bool>
        if (itDS == cfg.definitionSets.end() || itDS->second.first != "(1)") {
            continue;
        }

        // Requirement: Check that head is equal input with exception of one arg
        // We compare all arguments except the constantIndex.
        bool match = true;
        for (size_t i = 0; i < headArgs.size(); ++i) {
            if (static_cast<int>(i) == constantIndex) continue;

            if (headArgs[i] != inputArgs[i]) {
                match = false;
                break;
            }
        }
        if (!match) continue;

        // "Replacement map is not identity. in one arg they r not equal. these two args are use for replacement"
        std::map<std::string, std::string> replacementMap;
        replacementMap[headArgs[constantIndex]] = inputArgs[constantIndex];

        bool conditionMet = false;
        std::string targetVar = inputArgs[constantIndex]; // The variable we are tracking

        // Iterate over premises (all elements except the last one)
        for (size_t k = 0; k < chain.size() - 1; ++k) {
            std::string premise = chain[k];

            // Apply replacement (Replace the constant from Head with the variable from Input)
            std::string instantiatedPremise = ce::replaceKeysInString(premise, replacementMap);

            std::string removed = removeUPrefixFromArguments(instantiatedPremise);
            std::vector<std::string> argsVec = ce::getArgs(removed);
            std::set<std::string> argsSet(argsVec.begin(), argsVec.end());

            prepareIntegration(removed, argsSet, mb, validityName);

            // NEW: Explicitly check inputArgs[constantIndex] occurs 2 times
            // and did NOT occur 2 times before replacement

            // 1. Count in instantiated premise
            std::vector<std::string> pArgs = ce::getArgs(instantiatedPremise);
            int countAfter = 0;
            for (const auto& arg : pArgs) {
                if (arg == targetVar) countAfter++;
            }

            // 2. Count in original premise
            std::vector<std::string> origArgs = ce::getArgs(premise);
            int countBefore = 0;
            for (const auto& arg : origArgs) {
                if (arg == targetVar) countBefore++;
            }

            // Condition: Occurs explicitly 2 times now, but count was different before
            if (countAfter >= 2 && countBefore == 1) {
                conditionMet = true;
                // Do NOT break — every premise of a matched chain must be
                // sent through prepareIntegration. The condition-detection
                // was historically a side quest of this function; installing
                // the full integration bookkeeping for every premise of
                // every head-matching implication is now a primary purpose
                // (needed by the OR-branching machinery, among others).
            }
        }

        if (conditionMet) {
            // NEW CHECK: Check if the duplicated var (targetVar) already has another var =[var,var2]
            // We check the equivalence classes for the current validity scope.
            bool hasExistingEquality = false;
            auto itMap = mb.equivalenceClassesMap.find(validityName);
            if (itMap != mb.equivalenceClassesMap.end()) {
                for (const auto& ec : itMap->second) {
                    if (ec.variables.count(targetVar)) {
                        // User Request: assert that an eq class which contains the var has more than 1 element
                        assert(ec.variables.size() > 1);

                        hasExistingEquality = true;
                        break;
                    }
                }
            }

            // If yes, do not add a new copy
            if (hasExistingEquality) {
                continue;
            }

            // "call addExprToMemoryBlock() for '(=[Y,Y_copy])'"

            // Assert that Y is available
            assert(inputArgs.size() > static_cast<size_t>(constantIndex));

            std::string Y = inputArgs[constantIndex]; // Extract the input arg (Y)
            std::string Y_copy = Y + "_copy";
            std::string equalityExpr = "(=[" + Y + "," + Y_copy + "])";

            // Try to inherit levels
            std::set<int> levels;

            // Create origin info (dead-end axiom — see "variable copy" tag)
            std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
            origin.first = "variable copy";

            std::string finalExpr = removeUPrefixFromArguments(equalityExpr);

            levels.insert(mb.level);

            // Add the new equality
            addExprToMemoryBlock(finalExpr,
                mb,
                -1, // iteration
                0,  // status (Local statement)
                levels,
                origin,
                -1, // coreId
                -1, // auxyIndex
                "main",
                false);
        }
    }
}

// Count disjuncts in an expanded OR expression like !(&!(&!(d1)!(d2))!(d3))
// Parses the nested right-associative !(&...) structure.
static int countOrDisjuncts(const std::string& expr) {
    if (expr.size() < 4 || expr.substr(0, 3) != "!(&") return 1;
    // Strip !(&...) → inner content (two children, each starting with !)
    std::string inner = expr.substr(3, expr.size() - 4);
    // Find end of first child by tracking paren depth
    int depth = 0;
    bool seenOpen = false;
    for (size_t i = 0; i < inner.size(); ++i) {
        if (inner[i] == '(' || inner[i] == '[') { depth++; seenOpen = true; }
        if (inner[i] == ')' || inner[i] == ']') depth--;
        if (seenOpen && depth == 0) {
            return countOrDisjuncts(inner.substr(0, i + 1)) + 1;
        }
    }
    return 1; // malformed fallback
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

/// @brief Inner kernel of `addExprToMemoryBlock` — performs the actual
/// deposit after the public entry has run dedupe and shape dispatch.
///
/// @details
/// Splits responsibility:
/// - Atomic statement: push into `encodedStatements` /
///   `intEncodedStatements` / `localEncodedStatementsSet` /
///   `intKnownStatements`. Recompute `Memory::deltaNumberStatements`.
/// - Origin record: append a `(tag, antecedents)` entry into
///   `mb.exprOriginMap` and (when applicable) the mail-out mirror,
///   capped per `addOrigin`'s preference policy.
/// - Mail-out: when the deposit should propagate to a child LB,
///   queue it into `mb.mailOut[coreId].statements` per
///   [I-26](../../docs/30_invariants.md#i-26) (main-only validity).
/// - Equivalence-class admit: trigger `applyEquivalenceClass` on any
///   class whose variables overlap with the new statement's args.
///
/// The split lets `addExprToMemoryBlock` short-circuit on
/// already-known statements without paying the full deposit cost.
///
/// @invariant [I-26](../../docs/30_invariants.md#i-26) — mail-out
///            statements/implications MAIN-ONLY; exprOriginMap
///            ALL-SCOPES.
void ExpressionAnalyzer::addExprToMemoryBlockKernel(const std::string& expr,
    Memory& memoryBlock,
    int status,
    const std::set<int>& involvedLevels,
    const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
    const std::string& validityName,
    int coreId,
    int iteration)
{
    bool isLocal = (status == 0 || status == 1);
    ExpressionWithValidity exprVal(expr, validityName);
    if (status != 2) {
        if (this->isEquality(expr)) {
            addEquality(expr, memoryBlock, isLocal, involvedLevels, origin, validityName, isLocal);
        }
        else if (this->isNegatedEquality(expr)) {
            addNegatedEquality(expr, memoryBlock, isLocal, involvedLevels, origin, validityName, isLocal);
        }

        memoryBlock.wholeExpressions.insert(EncodedExpression(expr, validityName));
    }

    

    // Directly use 'expr' here
    std::vector<ExpressionWithValidity> added = this->addStatement(expr,
        memoryBlock,
        isLocal,
        involvedLevels,
        origin,
        validityName);

    std::vector<ExpressionWithValidity> newStatements = added;

    std::vector<ExpressionWithValidity> sortedNew = newStatements;
    std::sort(sortedNew.begin(), sortedNew.end());
    for (std::size_t idx = 0; idx < sortedNew.size(); ++idx)
    {
        // Each entry carries its own deposit scope. addStatement now returns
        // ExpressionWithValidity pairs so cross-scope deposits (descendant or
        // ancestor direction) ride out through newStatements with the deposit's
        // actual scope. Use the pair's validity throughout the loop body — the
        // kernel's lookups, admission updates, toBeProved discharge, and
        // validity-name promotion all follow the deposit scope, not the
        // caller's `validityName` parameter.
        const std::string& addExpression = sortedNew[idx].original;
        const std::string& effectiveValidity = sortedNew[idx].validityName;
        updateAdmissionMapIntegration(addExpression, memoryBlock, effectiveValidity);
        updateAdmissionMapRecursion(addExpression, memoryBlock, effectiveValidity);

		EncodedExpression encAddExpression(addExpression, effectiveValidity);

        std::map<EncodedExpression, std::set<int> >::const_iterator sit =
            memoryBlock.statementLevelsMap.find(encAddExpression);
        assert(sit != memoryBlock.statementLevelsMap.end());
        const std::set<int>& addExpressionLevels = sit->second;

        bool allLevelsInvolved = (addExpressionLevels.size() == static_cast<std::size_t>(memoryBlock.level + 1));
        if ((addExpressionLevels.size() == static_cast<std::size_t>(memoryBlock.level)) && addExpressionLevels.find(0) == addExpressionLevels.end())
        {
			allLevelsInvolved = true;
        }

        if (memoryBlock.isPartOfRecursion)
        {
            std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>> >::iterator itProof =
                memoryBlock.toBeProved.find(EncodedExpression(addExpression, effectiveValidity));
            if (itProof != memoryBlock.toBeProved.end() && effectiveValidity == "main") {

                if (!parameters.compressor_mode) { // <--- COMPRESSOR BYPASS
                    std::set<int> auxiesCopy = std::get<std::set<int>>(itProof->second);
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
                    }
                }
            }
        }
        else {
            if (this->isProved(addExpression) && allLevelsInvolved && status != 0 && effectiveValidity == "main") {
                std::string fullTheorem;
                if (memoryBlock.primedForContradiction && !memoryBlock.contradictionTheorem.empty()) {
                    fullTheorem = memoryBlock.contradictionTheorem;
                } else {
                    std::vector<std::string> globalKey = this->getGlobalKey(memoryBlock);
                    fullTheorem = this->reconstructImplication(globalKey, addExpression);
                }

                std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>> >::iterator itTBP2 =
                    memoryBlock.toBeProved.find(EncodedExpression(addExpression, effectiveValidity));
                if (itTBP2 != memoryBlock.toBeProved.end()) {

                    if (!parameters.compressor_mode) { // <--- COMPRESSOR BYPASS
                        {
                            std::lock_guard<std::mutex> lock(this->updateGlobalDirectMutex);
                            this->updateGlobalDirectTuples.push_back(std::make_tuple(fullTheorem, coreId));
                        }
                        memoryBlock.toBeProved.erase(itTBP2);
                        int16_t addExprScopeId = memoryBlock.nameMap.encodePush(NameMap::MAIN_ID, addExpression);
                        std::string addExprRootedScope = memoryBlock.nameMap.decode(addExprScopeId);
                        this->cleanUpIntegrationPreparation(addExpression, addExprRootedScope, memoryBlock);
                    }
                }
            }
        }

        std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>> >::iterator itProof =
            memoryBlock.toBeProved.find(EncodedExpression(addExpression, effectiveValidity));

        if (effectiveValidity != "main" && status == 1 && (itProof != memoryBlock.toBeProved.end()))
        {
            int16_t valId = memoryBlock.nameMap.encode(effectiveValidity);
            std::string orExpr;
            std::string orBranchBody;
            OrScopeKind orKind = classifyOrScope(memoryBlock.nameMap, valId, orExpr, orBranchBody);

            if (orKind == OrScopeKind::Integration) {
                // OR integration branch proved: emit the wrapping OR expression at
                // the OR-branches' shared PARENT scope (the immediate ancestor of
                // the _boundary_orint_…_((…)) push), then wipe all sibling branches
                // (including this one) whose top-of-stack payload classifies as
                // orint_ with the same orSignature.
                //
                // Pre-D-30 the OR was unconditionally emitted at "main". That was
                // correct only when the OR-branches' parent IS main; for ORs nested
                // inside a hypothetical scope (e.g. interval-body's backward
                // direction at _boundary_(implication26[…])), the OR landed one
                // scope above the Step-8 toBeProved goal at the implication's
                // hypo scope and never closed it. See D-30.
                std::size_t lastBoundary =
                    effectiveValidity.rfind(NameMap::BOUNDARY_STR, std::string::npos,
                                       NameMap::BOUNDARY_LEN);
                const std::string orEmitScope =
                    (lastBoundary == std::string::npos)
                        ? std::string("main")
                        : effectiveValidity.substr(0, lastBoundary);

                std::pair<std::string, std::vector<ExpressionWithValidity>> orOrigin;
                orOrigin.first = "or branch proven";
                orOrigin.second.push_back(ExpressionWithValidity(addExpression, effectiveValidity));
                addExprToMemoryBlock(orExpr,
                    memoryBlock,
                    iteration,
                    1,
                    involvedLevels,
                    orOrigin,
                    coreId,
                    -1,
                    orEmitScope,
                    false);

                this->cleanUpOrIntegrationBranches(orExpr, memoryBlock);
            }
            else if (orKind == OrScopeKind::NotOrScope) {
                // Implication integration path: effectiveValidity here is a rooted scope
                // name like "main_boundary_(implSig)". The canonical theorem to
                // publish at main is the bare implication signature — recover it
                // from top-of-stack of the scope's NameMap entry.
                const auto& stack = memoryBlock.nameMap.stackOfValidity[valId];
                assert(!stack.empty() && "NotOrScope branch requires non-empty stack");
                // Copy: nested addExprToMemoryBlock calls may push_back on idToSub,
                // invalidating any reference into it.
                const std::string bareSig = memoryBlock.nameMap.idToSub[stack.back()];

                std::pair<std::string, std::vector<ExpressionWithValidity>> localOrigin;
                localOrigin.first = "validity name";
                localOrigin.second.push_back(ExpressionWithValidity(addExpression, effectiveValidity));
                addExprToMemoryBlock(bareSig,
                    memoryBlock,
                    iteration,
                    1,
                    involvedLevels,
                    localOrigin,
                    coreId,
                    -1,
                    "main",
                    false);

                this->cleanUpIntegrationPreparation(bareSig, effectiveValidity, memoryBlock);
                memoryBlock.validityNamesToFilter.insert(effectiveValidity);
                memoryBlock.intValidityNamesToFilter.insert(valId);
            }
            // Disintegration: emission handled by the ordis merge block
            // below (sibling to this _orint_/NotOrScope block).
        }

        // ordis merge: convergence bookkeeping + per-branch cleanup.
        // Sibling to the toBeProved-discharge logic above and to the
        // _orint_/NotOrScope block. Function body in prover.hpp.
        ordisMerge(addExpression, effectiveValidity, addExpressionLevels, memoryBlock);
    }
}



bool ExpressionAnalyzer::checkForEquivalence(const std::string& expr,
    const std::string& validityName,
    Memory& memoryBlock)
{
    // 1. Parse arguments
    std::vector<std::string> args = ce::getArgs(expr);

    // 2. Identify Equivalence Classes for each argument
    // possibilities[i] will contain all valid variable substitutions for args[i]
    std::vector<std::vector<std::string>> possibilities;
    possibilities.reserve(args.size());

    // Access the equivalence class list for this validity scope
    auto itClasses = memoryBlock.equivalenceClassesMap.find(validityName);
    assert(itClasses != memoryBlock.equivalenceClassesMap.end());

    const std::vector<EquivalenceClass>& eqClasses = itClasses->second;

    for (const std::string& arg : args) {
        std::vector<std::string> varsForArg;
        bool foundClass = false;

        // Check if this arg belongs to any known equivalence class
        for (const auto& eqClass : eqClasses) {
            if (eqClass.variables.find(arg) != eqClass.variables.end()) {
                // Found: add all variables in this class as possibilities
                varsForArg.assign(eqClass.variables.begin(), eqClass.variables.end());
                foundClass = true;
                break; // Argument is assumed to belong to only one class per validity
            }
        }

        // If not found, it is its own only possibility
        if (!foundClass) {
            varsForArg.push_back(arg);
        }
        possibilities.push_back(std::move(varsForArg));
    }

    // 4. Generate All Variants (Cartesian Product) and Check
    // Using an iterative index counter to handle variable number of arguments
    std::vector<size_t> indices(possibilities.size(), 0);

    while (true) {
        // A. Construct replacement map for current indices
        std::map<std::string, std::string> replacements;
        for (size_t i = 0; i < args.size(); ++i) {
            // Map the original argument to the currently selected substitution
            replacements[args[i]] = possibilities[i][indices[i]];
        }

        // B. Generate variant using string replacement
        std::string variant = ce::replaceKeysInString(expr, replacements);

        // C. Check if this variant exists in wholeExpressions
        if (memoryBlock.wholeExpressions.find(EncodedExpression(variant, validityName)) !=
            memoryBlock.wholeExpressions.end()) {
            return true;
        }

        // D. Advance indices (odometer style)
        int i = static_cast<int>(possibilities.size()) - 1;
        while (i >= 0) {
            indices[i]++;
            if (indices[i] < possibilities[i].size()) {
                break; // Successfully incremented this position
            }
            else {
                indices[i] = 0; // Reset this position and carry over
                i--;
            }
        }
        // If we wrapped around the first index, we are done
        if (i < 0) break;
    }

    return false;
}
// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void ExpressionAnalyzer::updateWeakVariables(Memory& mb, const std::string& validityName, const std::string& eqlty) {
    // 1. Identify variables involved in the equality
    std::vector<std::string> eqVars = ce::getArgs(eqlty);
    if (eqVars.empty()) return;

    // 2. Locate the specific Equivalence Class containing these variables
    // We only look within the provided validityName scope
    auto itClasses = mb.equivalenceClassesMap.find(validityName);
    assert(itClasses != mb.equivalenceClassesMap.end());

    const std::vector<EquivalenceClass>& classes = itClasses->second;
    const EquivalenceClass* targetClass = nullptr;

    for (const auto& ec : classes) {
        // If any variable from the equality is in this class, this is the target.
        for (const std::string& v : eqVars) {
            if (ec.variables.find(v) != ec.variables.end()) {
                targetClass = &ec;
                break;
            }
        }
        if (targetClass) break;
    }

    if (!targetClass) assert(false);

    // 3. Logic to identify Weak Variables (Renamed Candidates)
    // Matches logic: int_ > it_, lexicographical sort.
    static const std::regex PATTERN_INT(R"(int_lev_\d+_\d+)");
    static const std::regex PATTERN_IT(R"(it_\d+_lev_\d+_\d+)");

    std::vector<std::string> intVars;
    std::vector<std::string> itVars;

    for (const std::string& var : targetClass->variables) {
        if (std::regex_match(var, PATTERN_INT)) {
            intVars.push_back(var);
        }
        else if (std::regex_match(var, PATTERN_IT)) {
            itVars.push_back(var);
        }
    }

    // Determine Canonical Variable
    std::string canonicalVar;
    if (!intVars.empty()) {
        std::sort(intVars.begin(), intVars.end());
        canonicalVar = intVars.front();
    }
    else if (!itVars.empty()) {
        std::sort(itVars.begin(), itVars.end());
        canonicalVar = itVars.front();
    }
    else {
        return; // No variables match the patterns, so no "weak" variables to record.
    }

    // 4. Save Weak Variables to mb.weakVariables
    // Weak = Matches pattern AND is not canonical
    auto checkAndAdd = [&](const std::vector<std::string>& vars) {
        for (const auto& var : vars) {
            if (var != canonicalVar) {
                mb.weakVariables.insert(ExpressionWithValidity(var, validityName));
            }
        }
        };

    checkAndAdd(intVars);
    checkAndAdd(itVars);
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
/// @brief Public entry to install an expression into an LB — the main
/// kernel of the prover.
///
/// @details
/// Every expression that lands in any LB goes through this function:
/// fresh statements from the prover loop, mail-in deposits from the
/// hash burst, head emissions from `checkLocalEncodedMemoryStatic`,
/// and disintegration products from `disintegrateExpr2`. The function
/// dispatches by expression shape:
///
/// 1. **Atomic predicate** (e.g. `(p[a])`) — installs as a statement
///    via the kernel; updates `encodedStatements`,
///    `intEncodedStatements`, `localEncodedStatementsSet`, and
///    `intKnownStatements`.
/// 2. **Equality** (e.g. `(=[a,b])`) — routes through `addEquality`
///    so the equivalence-class machinery can absorb it
///    (with mirror per [I-9](../../docs/30_invariants.md#i-9)).
/// 3. **Negated equality** (e.g. `!(=[a,b])`) — routes through
///    `addNegatedEquality` for the one-sided expansion per
///    [I-12](../../docs/30_invariants.md#i-12).
/// 4. **Anchor** (e.g. `(AnchorPeano[...])`) — routes through
///    `handleAnchor` to register the anchor's slot mappings.
/// 5. **Implication** (e.g. `(>[v](p[v])(q[v]))`) — installed as a
///    hash rule via `addToHashMemory` (after disintegration when
///    appropriate).
///
/// **Per [I-27](../../docs/30_invariants.md#i-27)**, ancestor-scan
/// dedupe runs at this entry (Site F / Site H): if the expression
/// already exists in any ancestor scope, the deposit is skipped.
/// This is what keeps cross-LB writes during the parallel phase
/// from causing duplicate-deposit storms
/// ([I-28](../../docs/30_invariants.md#i-28)).
///
/// @param expr           Canonical MPL expression text.
/// @param memoryBlock    Target LB.
/// @param involvedLevels LB levels at which the expression is
///                       admissible.
/// @param ...            Many trailing parameters routing the deposit
///                       through specific paths (origin tag,
///                       admission-update flag, mail-out flag,
///                       sender LB, etc.).
/// @invariant [I-9](../../docs/30_invariants.md#i-9) — addEquality
///            mirror guard.
/// @invariant [I-12](../../docs/30_invariants.md#i-12) — !(=[a,b])
///            one-sided expansion.
/// @invariant [I-27](../../docs/30_invariants.md#i-27) — Site F / H
///            ancestor-scan dedupe at entry.
/// @invariant [I-28](../../docs/30_invariants.md#i-28) — cross-LB
///            writes deferred to post-`pool.join()` collectors.
/// @see `addExprToMemoryBlockKernel` — inner kernel.
/// @see `addToHashMemory` (memory.cpp) — implication branch.
/// @see `addEquality`, `addNegatedEquality`, `handleAnchor` —
///      shape-specific branches.
void ExpressionAnalyzer::addExprToMemoryBlock(const std::string& expr,
    Memory& memoryBlock,
    int iteration,
    int status,
    const std::set<int>& involvedLevels,
    const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
    int coreId,
    int auxyIndex,
    std::string validityName,
    bool doNotDisintegrate,
    bool allowOrDisintegration) {

    // D-32: OR-disintegration may fire only when general disintegration
    // is allowed. If doNotDisintegrate is true the disintegrateExpr2
    // call below is skipped anyway, so this assignment is defensive —
    // it documents the coupling and survives future refactors.
    if (doNotDisintegrate) {
        allowOrDisintegration = false;
    }

    // --- Static (int-based) early checks ---
    // Fast path: duplicate + validity filter with just nm.encode (no regex, no vector alloc)
    int16_t origId = memoryBlock.nameMap.encode(expr);
    int16_t valId = memoryBlock.nameMap.encode(validityName);

    // Site F — dup suppression via ancestor scan. A known statement at any
    // ancestor scope (including self) means the same fact already holds at a
    // strictly weaker set of assumptions, so the child-scope insertion is
    // redundant. ancestorsOf[valId] includes valId itself and every strict
    // prefix scope registered via encodePush. For a flat root (e.g. legacy
    // integration cleanSignature-as-validity) the list is just {valId},
    // degenerating to the old strict-equality check — safe under the
    // MAIN_ID transition guard.
    if (!parameters.compressor_mode) {
        for (int16_t anc : memoryBlock.nameMap.ancestorsOf[valId]) {
            if (memoryBlock.intKnownStatements.count(packStatementKey(origId, anc))) return;
        }
    }

    // Site H — ancestor-scan the int validity blacklist. A blacklist entry at
    // any ancestor scope filters every descendant scope (deeper scopes inherit
    // the filter because they carry strictly more assumptions).
    for (int16_t anc : memoryBlock.nameMap.ancestorsOf[valId]) {
        if (memoryBlock.intValidityNamesToFilter.count(anc)) return;
    }

    // Passed fast checks — now do full encoding for axed var check + downstream
    EncodedExpression encExpr(expr, validityName);
    IntEncodedExpr ie = encodeExpression(encExpr, memoryBlock.nameMap);

    for (int16_t a = 0; a < ie.arity; ++a) {
        if (memoryBlock.intAxedVariables.count(ie.argFullId[a])) return;
    }

    ExpressionWithValidity exprVal(expr, validityName);


    if (status == 4)
    {
        memoryBlock.localEncodedStatements.push_back(encExpr);
        memoryBlock.localEncodedStatementsSet.insert(encExpr);
		memoryBlock.localEncodedStatementsDelta.push_back(encExpr);
        memoryBlock.encodedStatements.push_back(encExpr);
        memoryBlock.intEncodedStatements.push_back(ie);
        memoryBlock.intLocalEncodedStatements.push_back(ie);
        memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
		memoryBlock.statementLevelsMap[encExpr] = involvedLevels;
        memoryBlock.intKnownStatements.insert(packStatementKey(ie.originalId, ie.validityId));
		memoryBlock.wholeExpressions.insert(encExpr);

        return;
    }

    if (memoryBlock.equivalenceClassesMap.find(validityName) == memoryBlock.equivalenceClassesMap.end())
    {
        memoryBlock.equivalenceClassesMap[validityName] = std::vector<EquivalenceClass>();
        assert(memoryBlock.eqClassSttmntIndexMapMap.find(validityName) == memoryBlock.eqClassSttmntIndexMapMap.end());
        memoryBlock.eqClassSttmntIndexMapMap[validityName] = std::map<std::set<std::string>, int>();
    }


    std::string negation = this->negate(expr);
    int16_t negOrigId = memoryBlock.nameMap.encode(negation);
    // Site G — contradiction check via ancestor scan. A negation proved at any
    // ancestor scope (including self) contradicts the incoming positive:
    // ancestors carry strictly fewer assumptions, so the negation there is at
    // least as strong. Strict descendants of ie.validityId are NOT counted —
    // a deeper-scope negation holds under extra assumptions and must not
    // block the weaker positive at the parent scope.
    bool contradictionFound = false;
    for (int16_t anc : memoryBlock.nameMap.ancestorsOf[ie.validityId]) {
        if (memoryBlock.intKnownStatements.count(packStatementKey(negOrigId, anc)) > 0) {
            contradictionFound = true;
            break;
        }
    }
    if (contradictionFound)
    {
        // Incubator contradiction: record provenance, emit theorem, deactivate
        if (memoryBlock.primedForContradiction) {
            // Record origin for the triggering expression before early return.
            // Without this, buildStack cannot trace it as a contradiction dependency.
            if (parameters.trackHistory) {
                int maxOrig0 = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
                addOrigin(memoryBlock.exprOriginMap, exprVal, origin, maxOrig0);
            }

            std::string theorem = memoryBlock.contradictionTheorem;
            memoryBlock.contradictionTheorem.clear();
            memoryBlock.isActive = false;
            if (!theorem.empty()) {
                // Record contradiction provenance BEFORE broadcast
                if (parameters.trackHistory) {
                    // Extract cleanOp from exprKey: "__contradiction__(op[...])"
                    std::string cleanOp;
                    const std::string prefix = "__contradiction__";
                    if (memoryBlock.exprKey.substr(0, prefix.size()) == prefix) {
                        cleanOp = memoryBlock.exprKey.substr(prefix.size());
                    }
                    std::string negCleanOp = "!" + cleanOp;

                    // The negated expression is proved by contradiction:
                    // expr and negate(expr) both exist in the LB
                    ExpressionWithValidity evResult(negCleanOp, "main");
                    std::vector<ExpressionWithValidity> deps;
                    deps.push_back(ExpressionWithValidity(expr, validityName));
                    deps.push_back(ExpressionWithValidity(negation, validityName));
                    deps.push_back(ExpressionWithValidity(cleanOp, "main"));
                    int maxOrig = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
                    // D-51: contradiction record stays IN the __contradiction__
                    // LB only — no upward propagation to ancestors, no broadcast
                    // via mailOut. The chapter walker (buildStack) finds the
                    // record by switching INTO this contradiction LB via the
                    // ancestor's simpleMap when the negated head has no
                    // direct origin in the LB before the head. The previous
                    // pendingAncestorOrigins drain (drained at proveKernel
                    // post-pool.join, see D-39) duplicated the recipe across
                    // every ancestor, which violates the parent→child mail
                    // direction and produces chapter-walker confusion in
                    // nested-contradiction cases (chapter-37-style missing
                    // task-formulation rows). The local write at memoryBlock
                    // is what the buildStack switch will read.
                    addOrigin(memoryBlock.exprOriginMap, evResult,
                        std::make_pair("contradiction", deps), maxOrig);
                }
                updateGlobalDirect(theorem, coreId);
            }
            return;
        }
        // CE filter contradiction
        if (memoryBlock.contradictionIndex >= 0)
        {
			this->contradictionTable[memoryBlock.contradictionIndex].successful = true;
            memoryBlock.isActive = false;

            return;
        }
        // Vacuous truth in induction steps: contradiction in premises means
        // the induction step is vacuously true. Only for partOfRecursion LBs.
        // Confined to contradicting pairs at validity "main": a hypo-scope
        // contradiction must not justify a main-scope conclusion (would put
        // hypo-scope antecedents under a non-hypo origin row, violating the
        // rule that the visualizer-relevant proof graph contains no hypo
        // expressions).
        else if (memoryBlock.isPartOfRecursion && validityName == "main")
        {
            // Level gate: at least one ingredient must carry mb.level in
            // statementLevelsMap, i.e. be rooted in this LB's own inferences.
            // Without this gate, a contradiction between two anchor-inherited
            // statements (both carrying only level 0) could close this LB as
            // vacuously true even though neither fact was produced here —
            // unsound, since the induction step head would be "derived" from
            // premises that don't depend on the LB's recursion assumption.
            {
                const auto itEx = memoryBlock.statementLevelsMap.find(
                    EncodedExpression(expr, validityName));
                const auto itNeg = memoryBlock.statementLevelsMap.find(
                    EncodedExpression(negation, validityName));
                const bool exHasMbLevel =
                    (itEx != memoryBlock.statementLevelsMap.end() &&
                     itEx->second.count(memoryBlock.level) > 0);
                const bool negHasMbLevel =
                    (itNeg != memoryBlock.statementLevelsMap.end() &&
                     itNeg->second.count(memoryBlock.level) > 0);
                if (!exHasMbLevel && !negHasMbLevel) {
                    // Neither ingredient is LB-level-rooted. Skip vacuous
                    // truth emission; the LB stays active.
                    return;
                }
            }

            // Record origin for the triggering expression before early return.
            // Without this, buildStack cannot trace it as a vacuous truth dependency.
            if (parameters.trackHistory) {
                int maxOrig0 = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
                addOrigin(memoryBlock.exprOriginMap, exprVal, origin, maxOrig0);
            }

            // Use the LB's concrete recursion hypothesis as the 3rd
            // ingredient. For tempMb (check_induction_condition) this is
            // `(in2[rec, var, s])` — the same form used inside derivation
            // chains. For tempMb2 (check_zero) this is `(=[var, i0])`
            // which also equals exprKey. Using exprKey directly for tempMb
            // would fail the verifier trace because exprKey uses the
            // rec<N>-specific form whereas derivations use the generic
            // rec form — they rename to different v-numbers post
            // process_proof_graphs.
            std::string recursionPremise = memoryBlock.recursionHypothesis;
            // Fallback for non-induction LBs that somehow reach this branch
            // (should not happen — guarded above by isPartOfRecursion).
            if (recursionPremise.empty()) recursionPremise = memoryBlock.exprKey;

            // Find the toBeProved head and mark it proved
            for (auto& [tbpKey, tbpVal] : memoryBlock.toBeProved) {
                if (tbpKey.validityName == "main") {
                    // Record proof graph origin — 3 ingredients:
                    //  [0] expr at validityName
                    //  [1] negation at validityName
                    //  [2] recursionPremise at "main" (so the verifier can
                    //      confirm at least one of [0]/[1] traces back to
                    //      this LB's recursion hypothesis via chapter-local
                    //      origin chain)
                    if (parameters.trackHistory) {
                        int maxOrig0 = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
                        ExpressionWithValidity evHead(tbpKey.original, "main");
                        std::vector<ExpressionWithValidity> deps;
                        deps.push_back(ExpressionWithValidity(expr, validityName));
                        deps.push_back(ExpressionWithValidity(negation, validityName));
                        deps.push_back(ExpressionWithValidity(recursionPremise, "main"));
                        addOrigin(memoryBlock.exprOriginMap, evHead,
                            std::make_pair("vacuous truth", deps), maxOrig0);
                        addOrigin(memoryBlock.mailOut.exprOriginMap, evHead,
                            std::make_pair("vacuous truth", deps), maxOrig0);
                    }

                    // Mark head as proved (status=1) to trigger updateGlobal chain
                    // tbpKey.original has full expression (e.g. "(in3[7,8,9,4])"),
                    // tbpKey.name is just the bare name (e.g. "in3") which would crash.
                    std::set<int> lv;
                    for (int l = 0; l <= memoryBlock.level; ++l) lv.insert(l);
                    std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                    origin.first = "vacuous truth";
                    origin.second.push_back(ExpressionWithValidity(expr, validityName));
                    origin.second.push_back(ExpressionWithValidity(negation, validityName));
                    origin.second.push_back(ExpressionWithValidity(recursionPremise, "main"));
                    this->addExprToMemoryBlock(tbpKey.original, memoryBlock, iteration, 1, lv, origin, coreId, -1, "main", false);
                    break;
                }
            }
            memoryBlock.isActive = false;
            return;
        }
    }

    if (memoryBlock.contradictionIndex >= 0)
    {
        return;
    }



    if (status == 2) {
        std::map<EncodedExpression, std::tuple<std::set<int>, std::set<std::string>> >::iterator itTBP = memoryBlock.toBeProved.find(encExpr);
        if (itTBP != memoryBlock.toBeProved.end()) {
            if (auxyIndex >= 0) std::get<std::set<int>>(itTBP->second).insert(auxyIndex);
        }
        else {
            if (auxyIndex >= 0) {
                std::tuple<std::set<int>, std::set<std::string>> s;
                std::get<std::set<int>>(s).insert(auxyIndex);

                memoryBlock.toBeProved.insert(std::make_pair(encExpr, s));
            }
            else 
            {
                memoryBlock.toBeProved.insert(std::make_pair(encExpr, std::tuple<std::set<int>, std::set<std::string>>()));
            }

			checkNecessityForEquality(expr, memoryBlock, validityName);
        }

        std::vector<std::string> args = ce::getArgs(expr);
		std::set<std::string> remainingArgs(args.begin(), args.end());


        prepareIntegration(expr, remainingArgs, memoryBlock, validityName);
        memoryBlock.canBeSentSet.insert(expr);

        const std::string coreExpr = ce::extractExpression(expr);
        auto it = this->coreExpressionMap.find(coreExpr);

        if (it != this->coreExpressionMap.end() && !it->second.inputIndices.empty()) {
            this->updateAdmissionMap3(expr,
                memoryBlock,
                parameters.inductionMaxAdmissionDepth,
                parameters.inductionMaxSecondaryNumber,
                true);
        }
        return;
    }
    else {
        if (parameters.trackHistory) {
            addOrigin(memoryBlock.exprOriginMap, exprVal, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap, exprVal, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        if (status == 3)
        {
            bool isSimple = ce::expressionIsSimple(expr);
            std::vector<std::string> inputStream;

            if (isSimple) {
                addExprToMemoryBlockKernel(expr, memoryBlock, status, involvedLevels, origin, validityName, coreId, iteration);
            }
        }

        if (status != 3) {
            std::set<ExpressionWithValidity> imps;
            std::set<ExpressionWithValidity> stmts;
            int newStart = memoryBlock.startInt;

            if (!doNotDisintegrate && !checkForEquivalence(expr, validityName, memoryBlock))
            {
                auto tpl = this->disintegrateExpr2(expr,
                        memoryBlock,
                        iteration,
                        status == 0,
                        validityName,
                        /*trackHistoryLocal=*/true,
                        allowOrDisintegration);
                imps = std::get<0>(tpl);
                stmts = std::get<1>(tpl);
                memoryBlock.startInt = std::get<2>(tpl);
            }
            else
            {
				stmts.insert(ExpressionWithValidity(expr, validityName));
            }

            for (const auto& impEv : imps)
            {
                const std::string& impStr = impEv.original;
                const std::string& impValidity = impEv.validityName;

                std::pair<std::string, std::string> kv = ce::extractKeyValue(impStr, coreExpressionMap);
                const std::string& ky = kv.first;
                const std::string& vlue = kv.second;

                std::set<std::string> remainingArgsKey = extractRemainingArgs(ky);

                std::vector< std::tuple<
                    std::string,                    // leftExpr
                    std::vector<std::string>,       // args of the current implication node
                    std::set<std::string>           // node->left.arguments  (SET, like Python)
                > >tempChain;
                std::string head = ce::disintegrateImplication(impStr, tempChain, coreExpressionMap);
                std::vector<std::string> chain;
                for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));


                this->addToHashMemory(chain, head, remainingArgsKey,
                    memoryBlock, memoryBlock.overallHashMemory,
                    involvedLevels, impStr,
                    parameters.standardMaxAdmissionDepth, parameters.standardMaxSecondaryNumber, false,
                    parameters.minNumOperatorsKey, "implication", true, impStr, impValidity);
                this->addToHashMemory(chain, head, remainingArgsKey,
                    memoryBlock, memoryBlock.localHashMemory,
                    involvedLevels, impStr,
                    parameters.standardMaxAdmissionDepth, parameters.standardMaxSecondaryNumber, false,
                    parameters.minNumOperatorsKey, "implication", true, impStr, impValidity);
                this->addToHashMemory(chain, head, remainingArgsKey,
                    memoryBlock, memoryBlock.localHashMemoryDelta,
                    involvedLevels, impStr,
                    parameters.standardMaxAdmissionDepth, parameters.standardMaxSecondaryNumber, false,
                    parameters.minNumOperatorsKey, "implication", true, impStr, impValidity);

                // Mail-out contract: implications/statements main-only.
                // Non-main rules stay local — receivers re-derive them
                // from the mailed v=main statements + their own
                // disintegration. mailOut.exprOriginMap (history)
                // continues to carry entries for all scopes per the
                // existing trackExpansionHistory invariant; only the
                // hashmem-rule-shaped channel is gated. Pre-fix this
                // gate was missing, and impl24-scope rules from the
                // SE2 LB shipped to the contradiction LB and got
                // installed at v=main with origin keyed at the
                // sender's deeper scope, breaking the visualizer
                // walk's exprOriginMap lookup.
                if (impValidity == "main" && allowedForMail(impStr, memoryBlock))
                {
                    auto& impSet = memoryBlock.mailOut.implications;

                    impSet.insert(std::make_tuple(chain,
                        head,
                        remainingArgsKey,
                        involvedLevels,
                        impStr));
                }
            }

            if (imps.size() > 0)
            {
                // Check necessity for equality for all pending proofs
                for (auto const& [key, val] : memoryBlock.toBeProved) {
                    this->checkNecessityForEquality(key.original, memoryBlock, key.validityName);
                }
            }

            // Process statements — each carries its own validity name
            // (normal stmts have parent validity, OR branches have branch validity)
            for (const auto& ev : stmts)
            {
                auto itOrigin = memoryBlock.exprOriginMap.find(ev);
                assert(itOrigin != memoryBlock.exprOriginMap.end() && !itOrigin->second.empty());

                addExprToMemoryBlockKernel(ev.original, memoryBlock, status, involvedLevels,
                    itOrigin->second.front(), ev.validityName, coreId, iteration);
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

            zeroArgName = findZeroArgNameFromAnchor(sub);
            sName = findSArgNameFromAnchor(sub);
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

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

// Member of gl::ExpressionAnalyzer
void ExpressionAnalyzer::addTheoremToMemory(const std::string& expr,
                                            Memory& memory,
                                            int iteration,
                                            bool proved,
                                            Dependencies& dependencyTable) {
    // D-45: `proved` parameter is dead. The single caller
    // (prover.cpp:7028 inside the conjecture-batch loop) always
    // passes false, and the body's proved=true branch was already
    // unreachable. Guard with assert so we notice if a future caller
    // re-introduces proved=true (the branch's body has been deleted
    // and would need to be reconstructed alongside a real origin row).
    assert(!proved);
    (void)proved;

    // 1) Disintegrate implication -> chain (left nodes) + head (rightmost)
    std::vector< std::tuple<
        std::string,                  // left expression
        std::vector<std::string>,     // node args
        std::set<std::string>         // left-node arguments
    > > tempChain;
    const std::string head = ce::disintegrateImplication(expr, tempChain, this->coreExpressionMap);

    std::vector<std::string> chain;
    chain.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

    // 2) Walk/build the memory path for the chain
    Memory* memoryBlock = &memory;
    for (std::size_t index = 0; index < chain.size(); ++index) {
        const std::string& element = chain[index];
        bool isNewBody = false;

        std::map<std::string, Memory*>::iterator it = memoryBlock->simpleMap.find(element);
        if (it != memoryBlock->simpleMap.end() && it->second != NULL) {
            memoryBlock = it->second;
        } else {
            Memory* child = new Memory();
            memoryBlock->simpleMap[element] = child;

            child->parentMemory = memoryBlock;
            child->level = static_cast<int>(index);
            child->exprKey = element;

            this->permanentBodies.push_back(child);

            memoryBlock = child;
            isNewBody = true;

            std::set<int> lv; lv.insert(child->level);
            std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
            origin.first = "task formulation";
            this->addExprToMemoryBlock(element, *child, iteration, 0, lv, origin, -1, -1, "main", false);
        }

        // 3) At the last node: set up recursion/auxiliaries.
        // (D-45: the historical proved=true branch was dead — single caller
        // always passes proved=false. Guarded by assert at function entry.)
        if (index == chain.size() - 1) {
            {
                // digits / immutables
                const std::set<std::string> digits = findDigitArgs(expr, this->anchorInfo, coreExpressionMap);
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
                        Memory* tempMb = new Memory();
                        memoryBlock->simpleMap[tempExpr] = tempMb;

                        tempMb->level = static_cast<int>(index + 1);
                        tempMb->parentMemory = memoryBlock;
                        tempMb->exprKey = tempExpr;
                        tempMb->recursionHypothesis = tempExpr2;
                        tempMb->isPartOfRecursion = true;



                        // history tag for auxy implication
						const ExpressionWithValidity auxyImplicationVal(auxyImplication, "main");
                        addOrigin(tempMb->exprOriginMap, auxyImplicationVal, std::make_pair("recursion", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                        addOrigin(tempMb->mailOut.exprOriginMap, auxyImplicationVal, std::make_pair("recursion", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

                        this->permanentBodies.push_back(tempMb);

                        std::set<int> lv;
                        lv.insert(tempMb->level);

                        this->addToHashMemory(auxyChain, auxyHead, remainingArgsKey,
                            *tempMb, tempMb->overallHashMemory, lv,
                            auxyImplication,
                            parameters.inductionMaxAdmissionDepth,
                            parameters.inductionMaxSecondaryNumber,
                            true,
                            parameters.minNumOperatorsKey,
                            "implication", true, auxyImplication, "main");


                        // add (in2[rec,...]) to temp block
                        {
                            std::set<int> lv; lv.insert(tempMb->level);
							std::pair<std::string, std::vector<ExpressionWithValidity>> or1 = std::make_pair("recursion", std::vector<ExpressionWithValidity>());
                            this->addExprToMemoryBlock(tempExpr2, *tempMb, iteration, 0, lv, or1, -1, -1, "main", false);
                        }

                        // queue head as "to be proved" with auxy index
                        this->addExprToMemoryBlock(head,
                                                   *tempMb,
                                                   iteration,
                                                   2,
                                                   std::set<int>(),
                                                   std::pair<std::string, std::vector<ExpressionWithValidity>>(),
							                       -1,     
                                                   dependencyTable.auxyIndex,
                                                   "main",
                                                   false);

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
                        Memory* tempMb2 = new Memory();
                        memoryBlock->simpleMap[tempExpr3] = tempMb2;

                        tempMb2->level = static_cast<int>(index + 1);
                        tempMb2->parentMemory = memoryBlock;
                        tempMb2->exprKey = tempExpr4;
                        tempMb2->recursionHypothesis = tempExpr4;
                        tempMb2->isPartOfRecursion = true;

                        // Register toBeProved (status=2) BEFORE adding expression (status=0).
                        // When head == tempExpr4 (e.g. companion theorem where head is (=[7,2])),
                        // the status=0 add puts it in intKnownStatements, which would cause
                        // the status=2 call to return early without creating the toBeProved entry.
                        this->addExprToMemoryBlock(head,
                                                   *tempMb2,
                                                   iteration,
                                                   2,
                                                   std::set<int>(),
                                                   std::pair<std::string, std::vector<ExpressionWithValidity>>(),
                                                   -1,
                                                   dependencyTable.auxyIndex,
                                                   "main",
                                                   false);

                        {
                            std::set<int> lv; lv.insert(tempMb2->level);
							std::pair<std::string, std::vector<ExpressionWithValidity>> or2 = std::make_pair("recursion", std::vector<ExpressionWithValidity>());
                            this->addExprToMemoryBlock(tempExpr4, *tempMb2, iteration, 0, lv, or2, -1, -1, "main", false);
                        }

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
                EncodedExpression headEnc(head, "main");
                if (memoryBlock->wholeExpressions.find(headEnc) != memoryBlock->wholeExpressions.end()) {
                    // Head already derived (e.g. from anchor disintegration) — register directly
                    std::string fullTheorem = reconstructImplication(chain, head);
                    {
                        std::lock_guard<std::mutex> lock(this->theoremListMutex);
                        this->globalTheoremList.emplace_back(fullTheorem, "direct", "-1", "-1");
                    }
                    std::cout << fullTheorem << std::endl;
                } else {
                    this->addExprToMemoryBlock(head, *memoryBlock, iteration, 2, std::set<int>(), std::pair<std::string, std::vector<ExpressionWithValidity>>(), -1, -1, "main", false);
                }
            }

            // --- Incubator: create contradiction LB ---
            if (parameters.try_contradiction && !proved) {
                // Trap: reformulated operator conjecture?
                // head = =[x,a], one non-anchor operator premise, equality concerns output var.
                // Create __contradiction__(op[a]) under the ANCHOR (predecessor), seeded with
                // clean operator expression. Any contradiction in that LB -> !(op[a]).
                bool reformulatedOperator = false;
                if (parameters.incubator_mode) {
                    std::string eqExpr = ce::extractExpression(head);
                    if (eqExpr == "=" && chain.size() >= 2) {
                        const std::string& lastPremise = chain.back();
                        std::string opName = ce::extractExpression(lastPremise);
                        auto cfgIt = coreExpressionMap.find(opName);
                        if (cfgIt != coreExpressionMap.end() && !cfgIt->second.outputIndices.empty()) {
                            int outIdx = cfgIt->second.outputIndices[0];
                            std::vector<std::string> opArgs = ce::getArgs(lastPremise);
                            std::vector<std::string> eqArgs = ce::getArgs(head);
                            if (eqArgs.size() == 2 && outIdx < static_cast<int>(opArgs.size())) {
                                std::string x = opArgs[outIdx];
                                std::string a_out;
                                if (eqArgs[0] == x) a_out = eqArgs[1];
                                else if (eqArgs[1] == x) a_out = eqArgs[0];

                                if (!a_out.empty()) {
                                    // Back-reformulate: clean operator with original output var
                                    opArgs[outIdx] = a_out;
                                    std::string cleanOp = "(" + opName + "[";
                                    for (size_t ci = 0; ci < opArgs.size(); ++ci) {
                                        if (ci > 0) cleanOp += ",";
                                        cleanOp += opArgs[ci];
                                    }
                                    cleanOp += "])";

                                    std::vector<std::string> outerChain(chain.begin(), chain.end() - 1);
                                    std::string negCleanOp = "!" + cleanOp;
                                    std::string contrTheorem = reconstructImplication(outerChain, negCleanOp);

                                    // Create __contradiction__(op[a]) under the predecessor (Anchor LB)
                                    Memory* predecessor = memoryBlock->parentMemory;
                                    const std::string contradictionKey = "__contradiction__" + cleanOp;
                                    if (predecessor != nullptr &&
                                        predecessor->simpleMap.find(contradictionKey) == predecessor->simpleMap.end()) {

                                        Memory* contradictionLB = new Memory();
                                        predecessor->simpleMap[contradictionKey] = contradictionLB;

                                        contradictionLB->parentMemory = predecessor;
                                        contradictionLB->level = static_cast<int>(index);
                                        contradictionLB->exprKey = contradictionKey;
                                        contradictionLB->primedForContradiction = true;
                                        contradictionLB->contradictionTheorem = contrTheorem;

                                        this->permanentBodies.push_back(contradictionLB);

                                        // Seed with clean operator expression as fuel
                                        std::set<int> lv; lv.insert(contradictionLB->level);
                                        std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                                        origin.first = "task formulation";
                                        this->addExprToMemoryBlock(cleanOp, *contradictionLB, iteration, 0, lv, origin, -1, -1, "main", false);

                                    }
                                    reformulatedOperator = true;
                                }
                            }
                        }
                    }
                }

                if (!reformulatedOperator) {
                    const std::string contradictionKey = "__contradiction__" + head;
                    if (memoryBlock->simpleMap.find(contradictionKey) == memoryBlock->simpleMap.end()) {
                        Memory* contradictionLB = new Memory();
                        memoryBlock->simpleMap[contradictionKey] = contradictionLB;

                        contradictionLB->parentMemory = memoryBlock;
                        contradictionLB->level = static_cast<int>(index + 1);
                        contradictionLB->exprKey = contradictionKey;
                        contradictionLB->primedForContradiction = true;
                        contradictionLB->contradictionTheorem = reconstructImplication(chain, negate(head));

                        this->permanentBodies.push_back(contradictionLB);

                        // Add the head expression as fuel (status 0)
                        std::set<int> lv; lv.insert(contradictionLB->level);
                        std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                        origin.first = "task formulation";
                        this->addExprToMemoryBlock(head, *contradictionLB, iteration, 0, lv, origin, -1, -1, "main", false);

                    }
                }
            }

        }
    }
}
// Turn optimizations OFF for just this section
//#pragma optimize("", off)



void ExpressionAnalyzer::revisitRejected2(const std::string& markedExpr,
    Memory& memoryBlock,
    std::string validityName)
{
    ExpressionWithValidity evKey(markedExpr, validityName);
    auto& rm = memoryBlock.overallHashMemory.rejectedMap;

    // Guard: prevent re-entrant processing of the same marker
    if (memoryBlock.overallHashMemory.revisitInProgress.count(evKey)) {
        return;
    }

    auto itRM = rm.find(evKey);
    if (itRM == rm.end()) {
        return;
    }

    memoryBlock.overallHashMemory.revisitInProgress.insert(evKey);

    // Snapshot the rejected set — the loop below may indirectly insert
    // new entries into rm[evKey] via addExprToMemoryBlock → updateRejectedMap.
    // We process only the entries that existed before this call.
    std::set<RejectedMapValue> snapshot = itRM->second;

    // Remove only the snapshot entries; preserve any newly added ones.
    for (const auto& val : snapshot) {
        itRM->second.erase(val);
    }
    if (itRM->second.empty()) {
        rm.erase(itRM);
    }

    // No-equality direct revival — the rejection record's cohort is mailed
    // verbatim to internalMailIn. The verifier-relevant `disintegration`
    // origin for each child was written at the original production site by
    // disintegrateExprCore2's trackExpansionHistory and is already present
    // in memoryBlock.exprOriginMap; the mail's `equality1` self-source
    // origin cannot displace it (addOrigin's cap-full preference protects
    // foundation origins from convenience tags — see D-49 / I-35).
    std::set<std::string> noEqualities;

    for (const auto& val : snapshot) {
        const std::string& rejectedStmt = val.renamedExpression;
        std::vector<std::string> markedArgs = ce::getArgs(markedExpr);
        std::vector<std::string> rejectedArgs = ce::getArgs(rejectedStmt);

        if (markedArgs.size() != rejectedArgs.size()) {
            continue;
        }

        int markerIndex = -1;
        for (size_t i = 0; i < markedArgs.size(); ++i) {
            if (markedArgs[i] == "marker") {
                markerIndex = static_cast<int>(i);
                break;
            }
        }
        assert(markerIndex != -1);

        bool argsIdentical = true;
        for (size_t i = 0; i < markedArgs.size(); ++i) {
            if (static_cast<int>(i) == markerIndex) continue;
            if (markedArgs[i] != rejectedArgs[i]) {
                argsIdentical = false;
                break;
            }
        }
        assert(argsIdentical && "Mismatch between marked and rejected args (other than marker)");
        if (!argsIdentical) continue;

        // statementLevelsMap consistency check intentionally absent here —
        // the compact form's level entry may legitimately be missing when
        // the rejection was committed before the kernel's stmts-loop wrote
        // the compound's entry. val.levels (captured at buffer time, may
        // be empty) is the authoritative deposit-time levels set.

        // Deposit the stored cohort onto internalMailIn for absorb at the
        // next hashburst. pre == post (no rewrite participated here);
        // noEqualities is empty. The mail's equality1 origin is a soft
        // placeholder that does NOT overwrite the foundation `disintegration`
        // origin already in memoryBlock.exprOriginMap.
        this->emitIntegrationRevivalToInternalMailIn(
            val.concreteConstituent, val.concreteConstituent,
            val.siblings,            val.siblings,
            val.levels,
            validityName,
            noEqualities,
            memoryBlock);
    }

    memoryBlock.overallHashMemory.revisitInProgress.erase(evKey);
    cleanAdmissionMap(markedExpr, validityName, memoryBlock);
}


// Integration-side counterpart to revisitRejected2. Walks
// rejectedMapIntegration[markedKey, validityName] and, for each stored
// entry, emits the constituent + siblings onto internalMailIn so the next
// hashburst re-runs the full disintegration pipeline at the constituent's
// original validity. Unlike revisitRejected2:
//   * no addExprToMemoryBlock call (mailIn-only revival, linear),
//   * no cleanAdmissionMap / admission-key erasure (user-specified —
//     integration revival keeps the admission rule for future hits),
//   * no revisitInProgress guard (no re-entry through addStatement).
// The rejectedMapIntegration entry itself IS erased — the rejection is
// resolved (the constituents have been handed to the revival channel).
void ExpressionAnalyzer::revisitRejectedIntegration2(const std::string& markedKey,
    Memory& memoryBlock,
    const std::string& validityName) {

    auto& rmi = memoryBlock.overallHashMemory.rejectedMapIntegration;
    auto it = rmi.find(ExpressionWithValidity(markedKey, validityName));
    if (it == rmi.end()) return;

    // Snapshot — we erase the entry after processing.
    std::set<RejectedMapIntegrationValue> snapshot = it->second;

    std::set<std::string> noEqualities;  // direct revival — no equi rewrite participated.

    for (const auto& val : snapshot) {
        std::set<int> lvls;
        EncodedExpression encCompound(val.compoundExpression, validityName);
        auto itLev = memoryBlock.statementLevelsMap.find(encCompound);
        if (itLev != memoryBlock.statementLevelsMap.end()) lvls = itLev->second;

        // Direct-revisit path: no rewrite happened, so pre == post. For
        // equality1 origin, origin.second[0] still needs to be a sensible
        // source — passing the constituent itself lets the verifier walk
        // it as both source and result (with an empty equalities list).
        this->emitIntegrationRevivalToInternalMailIn(val.concreteConstituent,
                                                     val.concreteConstituent,
                                                     val.siblings,
                                                     val.siblings,
                                                     lvls,
                                                     validityName,
                                                     noEqualities,
                                                     memoryBlock);
    }

    rmi.erase(it);
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


void ExpressionAnalyzer::updateAdmissionMapRecursion(const std::string& expression,
    Memory& mb,
    const std::string& validityName)
{
    // 1. Initial Guard Clauses: Check Operator
    std::string core = ce::extractExpression(expression);
    if (this->operators.find(core) == this->operators.end()) {
        return;
    }

    auto itComp = this->coreExpressionMap.find(core);
    assert(itComp != this->coreExpressionMap.end());

    const auto& cfg = itComp->second;
    assert(!cfg.outputIndices.empty());

    // 2. Identify Output Argument
    std::vector<std::string> args = ce::getArgs(expression);
    int outIdx = cfg.outputIndices[0];

    assert(outIdx >= 0 && outIdx < static_cast<int>(args.size()));

    std::string inputOutputArg = args[outIdx];

    // 3. Prepare Lookup
    std::map<std::string, std::string> markerMap;
    markerMap[inputOutputArg] = "marker";
    std::string markedExpr = ce::replaceKeysInString(expression, markerMap);

    if (!mb.overallHashMemory.admissionStatusMap[ExpressionWithValidity(markedExpr, validityName)])
    {
        return;
    }

    // 4. Lookup in Admission Map
    ExpressionWithValidity evMarked(markedExpr, validityName);
    auto itAdm = mb.overallHashMemory.admissionMap.find(evMarked);

    // Guard: Return if not found
    if (itAdm == mb.overallHashMemory.admissionMap.end()) {
        return;
    }

    // 5. Iterate over Admission Values (Snapshot)
    std::vector<AdmissionMapValue> values(itAdm->second.begin(), itAdm->second.end());

    for (const auto& val : values) {
        for (const std::string& keyElem : val.key) {

            // --- Flattened Logic Start ---

            std::string kCore = ce::extractExpression(keyElem);

            // Skip non-operators
            if (this->operators.find(kCore) == this->operators.end()) continue;

            auto itKComp = this->coreExpressionMap.find(kCore);
            assert(itKComp != this->coreExpressionMap.end());
            assert(!itKComp->second.outputIndices.empty());

            std::vector<std::string> kArgs = ce::getArgs(keyElem);
            int kOutIdx = itKComp->second.outputIndices[0];

            assert(kOutIdx >= 0 && kOutIdx < static_cast<int>(kArgs.size()));

            std::string kOutputArg = kArgs[kOutIdx];

            // Condition: Output arg must NOT be in remaining args
            if (val.remainingArgs.find(kOutputArg) != val.remainingArgs.end()) continue;

            // Check for Match
            std::map<std::string, std::string> kMarkerMap;
            kMarkerMap[kOutputArg] = "marker";
            std::string testExpr = ce::replaceKeysInString(keyElem, kMarkerMap);

            if (testExpr != markedExpr) continue;

            // --- Match Found ---

            mb.overallHashMemory.productsOfRecursion.insert(inputOutputArg);
            mb.overallHashMemory.productsOfRecursionIds.insert(mb.nameMap.encode(inputOutputArg));

            // 6. Construct New Admission Entry
            std::map<std::string, std::string> replacementMap;
            replacementMap[kOutputArg] = inputOutputArg;

            std::vector<std::string> newKey;
            newKey.reserve(val.key.size());
            for (const std::string& k : val.key) {
                newKey.push_back(ce::replaceKeysInString(k, replacementMap));
            }

            std::set<std::string> newRemainingArgs = val.remainingArgs;
            newRemainingArgs.insert(inputOutputArg);

            // 7. Validate New Key Candidates
            for (const std::string& newKeyElem : newKey) {
                std::string newCore = ce::extractExpression(newKeyElem);

                if (this->operators.find(newCore) == this->operators.end()) continue;

                auto itNewComp = this->coreExpressionMap.find(newCore);
                assert(itNewComp != this->coreExpressionMap.end());
                assert(!itNewComp->second.outputIndices.empty());

                std::vector<std::string> newKArgs = ce::getArgs(newKeyElem);
                int newOutIdx = itNewComp->second.outputIndices[0];

                assert(newOutIdx >= 0 && newOutIdx < static_cast<int>(newKArgs.size()));

                std::string newOutputArg = newKArgs[newOutIdx];

                // Condition: New output NOT in new remaining args
                if (newRemainingArgs.find(newOutputArg) != newRemainingArgs.end()) continue;

                // Condition: All other inputs MUST be in new remaining args
                bool allInputsPresent = true;
                for (int i = 0; i < static_cast<int>(newKArgs.size()); ++i) {
                    if (i == newOutIdx) continue;
                    if (newRemainingArgs.find(newKArgs[i]) == newRemainingArgs.end()) {
                        allInputsPresent = false;
                        break;
                    }
                }

                if (!allInputsPresent) continue;

                // 8. Insert New Rule
                std::map<std::string, std::string> newMarkerMap;
                newMarkerMap[newOutputArg] = "marker";
                std::string newMarkedExpr = ce::replaceKeysInString(newKeyElem, newMarkerMap);

                // COPY matched value 'val' to preserve:
                // - standardMaxAdmissionDepth
                // - standardMaxSecondaryNumber
                // - flag
                AdmissionMapValue newVal = val;

                // Overwrite the specific fields for the new entry
                newVal.key = newKey;
                newVal.remainingArgs = newRemainingArgs;

                mb.overallHashMemory.admissionMap[ExpressionWithValidity(newMarkedExpr, validityName)].insert(newVal);
                mb.overallHashMemory.admissionStatusMap[ExpressionWithValidity(newMarkedExpr, validityName)] = mb.overallHashMemory.admissionStatusMap[ExpressionWithValidity(markedExpr, validityName)];

                // Populate vars-in-keys cache for the algebra equi-class hook.
                for (const std::string& a : ce::getArgs(newMarkedExpr)) {
                    if (a != "marker") {
                        mb.overallHashMemory.varsInAdmissionMapKeys.insert(a);
                    }
                }

                this->revisitRejected2(newMarkedExpr, mb, validityName);
            }
        }
    }
}

//#pragma optimize("", off)






// loadFactsForCEFiltering(), addConjectureForCEFiltering(),
// releaseCEBatchMemory() — moved to filter.cpp.

void ExpressionAnalyzer::destroyParentChildrenMap(ParentChildrenMap& m) {
    for (ParentChildrenMap::iterator it = m.begin(); it != m.end(); ++it)
        std::vector<Memory*>().swap(it->second);
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






// filterConjecturesWithCE() — moved to filter.cpp.




using gl::ParentChildrenMap;
using gl::PerCoreMailboxes;

ParentChildrenMap ExpressionAnalyzer::buildParentChildrenMap(const std::vector<Memory*>& roots) const {
    // 1) Collect reachable nodes and direct children
    std::unordered_map<Memory*, std::vector<Memory*>> direct;
    std::unordered_set<Memory*> seen;
    std::vector<Memory*> stack;
    for (Memory* r : roots) if (r) stack.push_back(r);

    while (!stack.empty()) {
        Memory* cur = stack.back(); stack.pop_back();
        if (!cur || !seen.insert(cur).second) continue;

        std::vector<Memory*> kids;
        kids.reserve(cur->simpleMap.size());
        for (std::map<std::string, Memory*>::const_iterator it = cur->simpleMap.begin();
            it != cur->simpleMap.end(); ++it) {
            Memory* ch = it->second;
            if (!ch) continue;
            kids.push_back(ch);
            stack.push_back(ch);
        }
        direct[cur] = std::move(kids);
    }

    // 2) Transitive closure with memoization; cycle-safe
    std::unordered_map<Memory*, std::unordered_set<Memory*>> memo;
    std::unordered_set<Memory*> visiting;

    std::function<const std::unordered_set<Memory*>& (Memory*)> dfs =
        [&](Memory* node) -> const std::unordered_set<Memory*>&{
        std::unordered_map<Memory*, std::unordered_set<Memory*>>::iterator mit = memo.find(node);
        if (mit != memo.end()) return mit->second;

        if (!visiting.insert(node).second) {
            static const std::unordered_set<Memory*> empty;
            return empty; // break cycles defensively
        }

        std::unordered_set<Memory*> acc;
        std::vector<Memory*>& kids = direct[node]; // default-constructed empty if missing
        for (std::size_t i = 0; i < kids.size(); ++i) {
            Memory* ch = kids[i];
            if (!ch) continue;
            acc.insert(ch);
            const std::unordered_set<Memory*>& sub = dfs(ch);
            acc.insert(sub.begin(), sub.end());
        }

        visiting.erase(node);
        return memo.emplace(node, std::move(acc)).first->second;
        };

    // Sort direct keys by exprKey for deterministic DFS order
    std::vector<Memory*> directKeys;
    directKeys.reserve(direct.size());
    for (auto& kv : direct) directKeys.push_back(kv.first);
    std::sort(directKeys.begin(), directKeys.end(),
        [](const Memory* a, const Memory* b) { return a->exprKey < b->exprKey; });
    for (Memory* dk : directKeys) dfs(dk);

    // 3) Convert sets to sorted vectors for deterministic order
    ParentChildrenMap out;
    out.reserve(memo.size());
    for (auto& kv : memo) {
        std::vector<Memory*> v(kv.second.begin(), kv.second.end());
        std::sort(v.begin(), v.end(),
            [](const Memory* a, const Memory* b) { return a->exprKey < b->exprKey; });
        out.emplace(kv.first, std::move(v));
    }

    return out;
}


PerCoreMailboxes ExpressionAnalyzer::buildPerCoreMailboxes(const ParentChildrenMap& index) const {
    unsigned cores = logicalCores;

    // Recipients are all values in the index
    std::unordered_set<Memory*> recipients;
    for (const auto& kv : index) 
        for (Memory* ch : kv.second) 
            if (ch) recipients.insert(ch);

    PerCoreMailboxes boxes;
    for (Memory* b : recipients) 
        boxes[b] = std::vector<Mail>(cores);
    return boxes;
}

void ExpressionAnalyzer::smashMail(PerCoreMailboxes& boxes) const{
    // Sort by body exprKey for deterministic processing order
    std::vector<std::pair<Memory*, std::vector<Mail>*>> sortedBoxes;
    sortedBoxes.reserve(boxes.size());
    for (auto& kv : boxes) {
        if (kv.first) sortedBoxes.push_back({kv.first, &kv.second});
    }
    std::sort(sortedBoxes.begin(), sortedBoxes.end(),
        [](const auto& a, const auto& b) { return a.first->exprKey < b.first->exprKey; });

    for (auto& [body, slotsPtr] : sortedBoxes) {
        // Merge statements and implications (set insert — order-independent)
        for (Mail& m : *slotsPtr) {
            body->mailIn.statements.insert(m.statements.begin(), m.statements.end());
            body->mailIn.implications.insert(m.implications.begin(), m.implications.end());
            m.statements.clear();
            m.implications.clear();
        }

        // Merge exprOriginMap: accumulate origins (order-independent union)
        for (Mail& m : *slotsPtr) {
            for (auto& kv2 : m.exprOriginMap) {
                auto& existing = body->mailIn.exprOriginMap[kv2.first];
                for (auto& orig : kv2.second) {
                    if (std::find(existing.begin(), existing.end(), orig) == existing.end())
                        existing.push_back(std::move(orig));
                }
            }
            m.exprOriginMap.clear();
        }
    }
}

void ExpressionAnalyzer::activateZeroCondition(Memory& memoryBlock)
{
    std::vector<std::string> args0 = ce::getArgs(memoryBlock.exprKey);
    std::string zeroArgName = this->findZeroArgName(memoryBlock);
    std::string eqExpr = std::string("(=[s(") + args0[0] + ")," + zeroArgName + "])";
    if (memoryBlock.parentMemory != NULL) {
        std::map<std::string, Memory*>::iterator itChild =
            memoryBlock.parentMemory->simpleMap.find(eqExpr);
        if (itChild != memoryBlock.parentMemory->simpleMap.end() && itChild->second) {
            itChild->second->isActive = true;
        }
    }
}

/// @brief Drive one outer iteration over every LB in `bodies` — the
/// prover's main parallel hash-burst loop.
///
/// @details
/// Spawns a worker pool sized at `logicalCores`; each worker pulls
/// LBs off the queue and calls `performElementaryLogicalStep` on each.
/// Cross-LB state mutations are forbidden during the worker phase per
/// [I-28](../../docs/30_invariants.md#i-28); each worker writes only
/// to its own LB's local state and to the per-thread queues
/// (`updateGlobalTuples`, `updateGlobalDirectTuples`, the
/// per-core-slot mail outboxes).
///
/// After `pool.join()`, the post-parallel collector phase runs:
/// 1. `smashMail` — sorts per-recipient mail by `exprKey` and drains
///    each LB's `mailOut` into the receivers' `mailIn` slots, with
///    the per-item validity assert per
///    [I-26](../../docs/30_invariants.md#i-26).
/// 2. `updateGlobal` / `updateGlobalDirect` — drains the per-thread
///    tuple queues into the global theorem list / dependencies.
/// 3. Equivalence-class merger pass — folds any new equalities
///    discovered this iteration.
///
/// The loop continues until either:
/// - no LB produced any new statement this iteration
///   (`Memory::deltaNumberStatements == 0` everywhere), or
/// - the iteration cap from `parameters` is reached.
///
/// @param bodies        LB pointers to iterate over.
/// @param iteration     Outer iteration counter.
/// @param ...           Routing parameters.
/// @invariant [I-26](../../docs/30_invariants.md#i-26),
///            [I-28](../../docs/30_invariants.md#i-28).
/// @see `performElementaryLogicalStep` — the per-LB worker.
/// @see `smashMail` — post-parallel mail-drain step.
void ExpressionAnalyzer::proveKernel(const std::vector<Memory*>& bodies,
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

            Memory* b = bodies[i];
            if (!b || !b->isActive) continue;

            this->performElementaryLogicalStep(*b, cid, index, boxes);
        }
        };

    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned t = 0; t < workers; ++t) pool.emplace_back(worker, t);
    for (auto& th : pool) th.join();

    this->smashMail(boxes);  // activate recipients for next iteration

    // Sort thread-collected vectors for deterministic processing order
    std::sort(inductionMemoryBlocks.begin(), inductionMemoryBlocks.end(),
        [](const Memory* a, const Memory* b) { return a->exprKey < b->exprKey; });
    std::sort(updateGlobalTuples.begin(), updateGlobalTuples.end());
    std::sort(updateGlobalDirectTuples.begin(), updateGlobalDirectTuples.end());

    for (int blockIndex = 0; blockIndex < inductionMemoryBlocks.size(); blockIndex++)
    {
        Memory* memoryBlock = inductionMemoryBlocks[blockIndex];
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

    // (D-39 pendingAncestorOrigins drain retired by D-51 — contradiction record
    // now lives only in the __contradiction__ LB, no upward propagation. Drain
    // queue + sort + iteration removed in the post-D-52 cleanup commit.)

	inductionMemoryBlocks.clear();
	updateGlobalTuples.clear();
	updateGlobalDirectTuples.clear();
}

void ExpressionAnalyzer::removeExpressionFromMemoryBlock(const EncodedExpression& encExpr, Memory& mb, int state) {
    if (state == 0)
    {
        // Erase matching entries from int vectors in parallel with string vectors.
        // Find indices to remove, erase from back to front.
        auto eraseParallel = [](auto& strVec, auto& intVec, const EncodedExpression& target) {
            for (std::size_t i = strVec.size(); i-- > 0; ) {
                if (strVec[i] == target) {
                    strVec.erase(strVec.begin() + static_cast<std::ptrdiff_t>(i));
                    intVec.erase(intVec.begin() + static_cast<std::ptrdiff_t>(i));
                }
            }
        };
        eraseParallel(mb.localEncodedStatements, mb.intLocalEncodedStatements, encExpr);
        eraseParallel(mb.localEncodedStatementsDelta, mb.intLocalEncodedStatementsDelta, encExpr);
        eraseParallel(mb.encodedStatements, mb.intEncodedStatements, encExpr);
    }
    else if (state == 1) {
        // Remove from toBeProved
        assert(mb.toBeProved.count(encExpr) == 1);
        mb.toBeProved.erase(encExpr);
    }
}
// Turn optimizations OFF for just this section
//#pragma optimize("", off)


void ExpressionAnalyzer::cleanUpIntegrationPreparationCore(const Instruction& instructions,
    const std::string& rootedScope,
    Memory& mb)
{
    (void)instructions;
    std::string prefixVar0 = rootedScope + "_boundary__var0_";

    std::set<EncodedExpression> stmtsToRemove;
    for (const auto& stmt : mb.encodedStatements) {
        if (stmt.validityName == rootedScope
            || stmt.validityName.rfind(prefixVar0, 0) == 0) {
            stmtsToRemove.insert(stmt);
        }
    }
    for (const auto& stmt : stmtsToRemove) {
        removeExpressionFromMemoryBlock(stmt, mb, 0);
    }

    std::vector<EncodedExpression> tbpToRemove;
    for (const auto& kv : mb.toBeProved) {
        if (kv.first.validityName == rootedScope) {
            tbpToRemove.push_back(kv.first);
        }
    }
    for (const auto& stmt : tbpToRemove) {
        removeExpressionFromMemoryBlock(stmt, mb, 1);
    }
}


// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void ExpressionAnalyzer::cleanUpIntegrationPreparation(const std::string& expression,
    const std::string& rootedScope,
    Memory& memoryBlock)
{
    std::string replExpr = prefixArgumentsWithU(expression);
    Instruction instructions;
    prepareIntegrationCore(replExpr, instructions, memoryBlock, expression);

    cleanUpIntegrationPreparationCore(instructions, rootedScope, memoryBlock);
}

void ExpressionAnalyzer::cleanUpOrIntegrationBranches(const std::string& orExprSig,
    Memory& mb)
{
    const NameMap& nm = mb.nameMap;
    std::vector<int16_t> victims;
    std::string orExprOut;
    std::string bodyOut;

    // Scan every validity id (skip slot 0 = invalid, slot 1 = MAIN).
    for (int16_t id = 2; id < nm.nextId; ++id) {
        if (classifyOrScope(nm, id, orExprOut, bodyOut) == OrScopeKind::Integration
            && orExprOut == orExprSig) {
            victims.push_back(id);
        }
    }

    for (int16_t id : victims) {
        const std::string vname = nm.decode(id);

        // Remove statements recorded at this branch scope.
        std::vector<EncodedExpression> stmtsToRemove;
        for (const auto& stmt : mb.encodedStatements) {
            if (stmt.validityName == vname) stmtsToRemove.push_back(stmt);
        }
        for (const auto& s : stmtsToRemove) {
            removeExpressionFromMemoryBlock(s, mb, 0);
        }

        // Remove toBeProved entries at this branch scope.
        std::vector<EncodedExpression> tbpToRemove;
        for (const auto& kv : mb.toBeProved) {
            if (kv.first.validityName == vname) tbpToRemove.push_back(kv.first);
        }
        for (const auto& s : tbpToRemove) {
            removeExpressionFromMemoryBlock(s, mb, 1);
        }

        // Block future insertions at this branch scope.
        mb.validityNamesToFilter.insert(vname);
        mb.intValidityNamesToFilter.insert(id);
    }
}

std::string ExpressionAnalyzer::prefixArgumentsWithU(const std::string& expr) {
    std::vector<std::string> args = ce::getArgs(expr);
    std::map<std::string, std::string> replacementMap;

    for (const std::string& arg : args) {


        replacementMap[arg] = "u_" + arg;
    }

    return ce::replaceKeysInString(expr, replacementMap);
}

std::string ExpressionAnalyzer::removeUPrefixFromArguments(const std::string& expr) {
    std::vector<std::string> args = ce::getArgs(expr);
    std::map<std::string, std::string> replacementMap;

    for (const std::string& arg : args) {
        // Only remove "u_" if the argument actually starts with it
        if (startsWith(arg, "u_", 2)) {
            replacementMap[arg] = arg.substr(2);
        }
    }

    return ce::replaceKeysInString(expr, replacementMap);
}

std::string ExpressionAnalyzer::addMissingU(const std::string& expr) {
    std::vector<std::string> args = ce::getArgs(expr);
    std::map<std::string, std::string> replacementMap;

    for (const std::string& arg : args) {
        // Check if the argument does NOT start with "u_"
        if (!startsWith(arg, "u_", 2) && arg != "marker") {
            replacementMap[arg] = "u_" + arg;
        }
    }

    // Replace the arguments in the expression string
    return ce::replaceKeysInString(expr, replacementMap);
}

std::vector<std::string> ExpressionAnalyzer::listLastRemovedArgsLE(const LogicalEntity& le) {
    // Check that the category is explicitly "existence"
    assert(le.category == "existence" && "LogicalEntity category must be 'existence'");

    std::set<std::string> uniqueArgs;

    for (const std::string& element : le.elements) {
        std::vector<std::string> args = ce::getArgs(element);
        for (const std::string& arg : args) {
            // Check if arg does NOT start with "u_"
            if (!startsWith(arg, "u_", 2)) {
                uniqueArgs.insert(arg);
            }
        }
    }

    // Convert set back to vector (sorted by default due to std::set)
    return std::vector<std::string>(uniqueArgs.begin(), uniqueArgs.end());
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void ExpressionAnalyzer::makeAdmissionKeys(const std::vector<std::string>& implication,
    const std::string& expression,
    HashMemory& localMemory,
    std::string validityName,
    Memory& mb)
{
    // 1. Check expression arguments
    //    Must have at least one "repl_" and all others must be "u_", "repl_", or "marker"
    std::vector<std::string> exprArgs = ce::getArgs(expression);
    bool hasRepl = false;

    for (const std::string& arg : exprArgs) {
        if (startsWith(arg, "repl_", 5)) {
            hasRepl = true;
        }
        else if (startsWith(arg, "u_", 2)) {
            continue;
        }
        else if (arg == "marker") {
            continue;
        }
        else {
            // Argument is not allowed (must be u_, repl_, or marker)
            return;
        }
    }

    if (!hasRepl) {
        return;
    }

    localMemory.triggersForAdmissionSetIntegration.insert(ExpressionWithValidity(expression, validityName));

    // 2. Validate Head vs Expression
    if (implication.empty()) return;
    const std::string& head = implication.back();

    // A. Check Core Expression Match
    std::string headCore = ce::extractExpressionUniversal(head);
    std::string exprCore = ce::extractExpressionUniversal(expression);

    if (headCore != exprCore) {
        return;
    }

    // B. Check Negation Status Match
    //    Assuming canonical form where negation always puts '!' at the start
    bool headIsNegated = (!head.empty() && head[0] == '!');
    bool exprIsNegated = (!expression.empty() && expression[0] == '!');

    if (headIsNegated != exprIsNegated) {
        return;
    }

    // C. Check Arguments and 1:1 "u_" relation
    std::vector<std::string> headArgs = ce::getArgs(head);
    assert(headArgs.size() == exprArgs.size());

    std::map<std::string, std::string> replacementMap;

    for (size_t i = 0; i < headArgs.size(); ++i) {
        const std::string& hArg = headArgs[i];
        const std::string& eArg = exprArgs[i];

        bool hIsU = startsWith(hArg, "u_", 2);
        bool eIsU = startsWith(eArg, "u_", 2);

        // "Important: head and expression must have all 'u_' args in the same position."
        if (hIsU || eIsU) {
            if (hArg != eArg) {
                return; // Violation of 1:1 relation
            }
        }
        else {
            // Map non-u head arg to expression arg
            replacementMap[hArg] = eArg;
        }
    }

    // 3. Apply Mapping to Implication Body (Head excluded)
    for (size_t i = 0; i < implication.size() - 1; ++i) {
        std::string mappedElement = ce::replaceKeysInString(implication[i], replacementMap);

        // Analyze args of the mapped element
        std::vector<std::string> outArgs = ce::getArgs(mappedElement);
        int markerCount = 0;
        bool argsValid = true;

        for (const std::string& arg : outArgs) {
            if (arg == "marker") {
                markerCount++;
            }
            else if (startsWith(arg, "repl_", 5)) {
                continue;
            }
            else if (startsWith(arg, "u_", 2)) {
                continue;
            }
            else {
                argsValid = false;
                break;
            }
        }

        // Admission Condition: Exactly one "marker" and valid args
        if (argsValid && markerCount == 1) {
            const std::string admissionKey = removeUPrefixFromArguments(mappedElement);
            localMemory.admissionSetIntegration.insert(ExpressionWithValidity(admissionKey, validityName));

            // New admission-set-integration key — revive any prior
            // integration-side rejection keyed at this marker form.
            // `mb.overallHashMemory.rejectedMapIntegration` is the fixed
            // lookup target; firing here is safe regardless of whether
            // `localMemory` is overallHashMemory or a local/delta mirror
            // (the overall-memory insert happens first in the enclosing
            // addToHashMemory 3-way fan-out, so by any subsequent call the
            // admission key is visible in overall memory).
            this->revisitRejectedIntegration2(admissionKey, mb, validityName);
        }

        // OR integration trigger: if premise is a compound OR expression with
        // only repl_/u_/marker args (no bound pi_lev_ vars), trigger integration.
        // This enables the prover to construct OR expressions from their elements.
        if (argsValid) {
            std::string premiseCore = ce::extractExpressionUniversal(mappedElement);
            auto ceIt = compiledExpressions.find(premiseCore);
            if (ceIt != compiledExpressions.end() && ceIt->second.category == "or") {
                // TODO: call prepareIntegration for the OR expression (with u_ intact)
            }
        }
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)


std::string ExpressionAnalyzer::expandSignature(const LogicalEntity& le)
{
    const std::string& category = le.category;
    const std::string& signature = le.signature;
    const std::vector<std::string>& elements = le.elements;

    // 1. Assert that all signature arguments start with 'u_'
    std::vector<std::string> sigArgsVec = ce::getArgs(signature);
    std::set<std::string> sigArgs(sigArgsVec.begin(), sigArgsVec.end());

    for (const auto& arg : sigArgs) {
        bool startsWithU = (arg.size() >= 2 && arg[0] == 'u' && arg[1] == '_');
        assert(startsWithU && "Signature arguments must start with 'u_' as per assumption.");
    }

    std::string result;

    // --- CASE 1: AND (&) ---
    if (category == "and") {
        if (elements.empty()) result = signature;
        else if (elements.size() == 1) result = elements[0];
        else {
            // Build nested hierarchy: (& result next)
            std::string current = elements[0];
            for (size_t i = 1; i < elements.size(); ++i) {
                current = "(&" + current + elements[i] + ")";
            }
            result = current;
        }
    }

    // --- CASE 2: EXISTENCE ---
    else if (category == "existence") {
        // Expecting at least 2 elements: [0]=body/definition, [1]=head/claim
        if (elements.size() < 2) result = signature;
        else {
            std::string body = elements[0];
            std::string head = elements[1];

            // Handle Head Negation: "remove internal negation" -> effectively invert
            std::string negatedHead;
            if (!head.empty() && head[0] == '!') {
                negatedHead = head.substr(1); // Remove existing '!'
            }
            else {
                negatedHead = "!" + head;     // Add '!'
            }

            // Determine Bound Variables: Args in body that are NOT in signature
            std::vector<std::string> bodyArgs = ce::getArgs(body);
            std::set<std::string> boundVars;

            for (const auto& arg : bodyArgs) {
                if (sigArgs.find(arg) == sigArgs.end()) {
                    boundVars.insert(arg);
                }
            }

            // Construct: !(>[bound_vars] body negated_head)
            std::string varsStr;
            for (const auto& v : boundVars) {
                if (!varsStr.empty()) varsStr += ",";
                varsStr += v;
            }

            result = "!(>[" + varsStr + "]" + body + negatedHead + ")";
        }
    }

    // --- CASE 3: IMPLICATION ---
    else if (category == "implication") {
        if (elements.empty()) result = signature;
        else {
            std::vector<std::string> chain = elements;
            std::string head = chain.back();
            chain.pop_back();

            result = this->reconstructImplicationFullBind(chain, head);
        }
    }

    // --- CASE 4: OR ---
    else if (category == "or") {
        if (elements.empty()) result = signature;
        else if (elements.size() == 1) result = elements[0];
        else {
            // Rebuild nested !(&!(...) !(...)) from flat disjunct list
            // !(&!a !b) then !(&!(&!a !b) !c) etc.
            std::string current = "!(&!" + elements[0] + "!" + elements[1] + ")";
            for (size_t i = 2; i < elements.size(); ++i) {
                current = "!(&" + current + "!" + elements[i] + ")";
            }
            result = current;
        }
    }

    // --- ERROR ---
    else {
        assert(false && "Forbidden category encountered in expandSignature. Must be 'and', 'existence', 'implication', or 'or'.");
        return signature;
    }

    // --- FINAL RENAMING (Skipped for Implication) ---
    if (category == "implication") {
        return result;
    }

    // 1. Collect all tokens from the resulting string to identify what needs renaming
    std::set<std::string> tokens;
    ce::TreeNode1* root = ce::parseExpr(result);

    std::vector<ce::TreeNode1*> stack;
    if (root) {
        stack.push_back(root);
    }

    while (!stack.empty()) {
        ce::TreeNode1* curr = stack.back();
        stack.pop_back();

        // Extract arguments from the current node's value (e.g., ">[a,b]" or "Macro[a,b]")
        std::vector<std::string> args = ce::getArgs(curr->value);

        for (const std::string& t : args) {
            // Simple trim (if needed, though parseExpr strips most whitespace)
            size_t first = t.find_first_not_of(" \t\r\n");
            if (first != std::string::npos) {
                size_t last = t.find_last_not_of(" \t\r\n");
                tokens.insert(t.substr(first, (last - first + 1)));
            }
        }

        // Push children to the stack to continue traversal
        if (curr->left) stack.push_back(curr->left);
        if (curr->right) stack.push_back(curr->right);
    }

    // Clean up the allocated syntax tree
    ce::deleteTree(root);

    // 2. Build the Replacement Map based on rules
    std::map<std::string, std::string> renameMap;
    // Regex for it_... and int_... patterns
    const std::regex reIt(R"(^it_\d+_lev_\d+_\d+$)");
    const std::regex reInt(R"(^int_lev_\d+_\d+$)");

    for (const std::string& t : tokens) {
        // Rule A: All u_ args lose u_
        if (t.size() >= 2 && t[0] == 'u' && t[1] == '_') {
            renameMap[t] = t.substr(2);
        }
        // Rule B: All other args...
        else {
            // ...except it_... and int_...
            if (std::regex_match(t, reIt) || std::regex_match(t, reInt)) {
                // Keep as is
            }
            // ...gain c_ prefix
            else {
                renameMap[t] = "c_" + t;
            }
        }
    }

    // 3. Apply Renaming
    return ce::replaceKeysInString(result, renameMap);
}

// In GL_Quick_VS/GL_Quick/src/analyze_expressions.cpp

void ExpressionAnalyzer::disintegrateExprCore2(const std::string& expr,
    std::vector<LogicalEntity>& instructions,
    Memory& memoryBlock,
    int iteration,
    std::map<std::string, std::pair<std::set<ExpressionWithValidity>, std::set<std::string>>>& collected,
    std::map<std::string, std::vector<std::string>>& newVarMap,
    std::string validityName,
    std::vector<ExpressionWithValidity>& orBranchStatements,
    bool trackHistoryLocal,
    bool allowOrDisintegration)
{
    // Capture startInt at the start of core() as reference
    int referenceStartInt = memoryBlock.startInt;

    if (iteration == -1)
    {
        iteration = 0;
    }

    const std::string addedExpression = addMissingU(expr);

    // 1. Find the instruction corresponding to the current expression
    auto it = std::find_if(instructions.begin(), instructions.end(),
        [&](const LogicalEntity& le) { return le.signature == addedExpression; });

    // Determine the current statement signature for the map key
    std::string currentStatement;
    if (it == instructions.end()) {
        // ---- Negated existence expansion ----
        // !(existence2[args]) is not in instructions, but (existence2[args]) is.
        // existence category encodes !(>[bound](left)!(right)), so:
        //   !(existence2[args]) = !!(>[bound](left)!(right)) = (>[bound](left)!(right))
        // which is an implication: from left derive !(right).
        if (addedExpression.size() >= 2 && addedExpression[0] == '!' && addedExpression[1] == '(') {
            std::string innerExpr = addedExpression.substr(1);
            auto itInner = std::find_if(instructions.begin(), instructions.end(),
                [&](const LogicalEntity& le) { return le.signature == innerExpr; });

            if (itInner != instructions.end() && itInner->category == "existence"
                && itInner->elements.size() == 2) {
                // elements[0] = left  (e.g. (in[pi_lev_0_X,u_1]))
                // elements[1] = right (e.g. (in2[pi_lev_0_X,u_7,u_3]))
                // The pi_lev_ bound variable must be replaced with a fresh integer
                // so that renamingChain (which expects pure-integer changeables) works.
                std::string boundVar;
                for (const auto& elem : itInner->elements) {
                    for (const auto& arg : ce::getArgs(elem)) {
                        if (!startsWith(arg, "u_", 2)) { boundVar = arg; break; }
                    }
                    if (!boundVar.empty()) break;
                }
                std::string freshInt = std::to_string(memoryBlock.startInt++);
                std::map<std::string, std::string> bvMap;
                bvMap[boundVar] = freshInt;

                std::string left  = ce::replaceKeysInString(itInner->elements[0], bvMap);
                std::string right = ce::replaceKeysInString(itInner->elements[1], bvMap);

                // Two implications from !(>[bound](left)!(right)) = (>[bound](left)!(right)):
                //   1) left -> !(right)
                //   2) right -> !(left)
                currentStatement = removeUPrefixFromArguments(addedExpression);

                std::string impl1 = reconstructImplicationFullBind({ left }, "!" + right);
                std::string impl2 = reconstructImplicationFullBind({ right }, "!" + left);
                collected[currentStatement].first.insert(ExpressionWithValidity(impl1, validityName));
                collected[currentStatement].first.insert(ExpressionWithValidity(impl2, validityName));

                if (parameters.trackHistory && trackHistoryLocal) {
                    int maxOrig = parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr;
                    std::string srcExpr = removeUPrefixFromArguments(expr);

                    // Expansion origin: negated existence -> implications directly
                    // (same pattern as other expansions: source expression -> products)
                    std::string impl1Clean = removeUPrefixFromArguments(impl1);
                    std::string impl2Clean = removeUPrefixFromArguments(impl2);
                    std::pair<std::string, std::vector<ExpressionWithValidity>> expOrigin;
                    expOrigin.first = "expansion";
                    expOrigin.second.push_back(ExpressionWithValidity(srcExpr, validityName));
                    addOrigin(memoryBlock.exprOriginMap,
                        ExpressionWithValidity(impl1Clean, validityName), expOrigin, maxOrig);
                    addOrigin(memoryBlock.mailOut.exprOriginMap,
                        ExpressionWithValidity(impl1Clean, validityName), expOrigin, maxOrig);
                    addOrigin(memoryBlock.exprOriginMap,
                        ExpressionWithValidity(impl2Clean, validityName), expOrigin, maxOrig);
                    addOrigin(memoryBlock.mailOut.exprOriginMap,
                        ExpressionWithValidity(impl2Clean, validityName), expOrigin, maxOrig);
                }

                return;
            }
        }

        currentStatement = removeUPrefixFromArguments(addedExpression);
        collected[currentStatement]; // Ensure entry exists
        return;
    }

    const LogicalEntity& ent = *it;
    currentStatement = removeUPrefixFromArguments(ent.signature);
    collected[currentStatement]; // Ensure entry exists

    // --- LAMBDA: History Tracking ---
    // Handles "expansion" origin for the entity, and optionally "disintegration" origin for its children
    auto trackExpansionHistory = [&](const LogicalEntity& entityToTrack,
        bool trackChildren = false)
        {
            if (!parameters.trackHistory) return;
            // Hypothetical disintegration path: suppress all history writes.
            // disintegrateExprHypothetically uses disintegrateExpr2 only to
            // discover structure; its products are throw-away and must never
            // feed exprOriginMap / mailOut.exprOriginMap (buildStack followed
            // such an orphan origin and crashed).
            if (!trackHistoryLocal) return;

            // 1. Record Expansion Origin
            std::pair<std::string, std::vector<ExpressionWithValidity>> originExpansion;
            originExpansion.first = "expansion";
            originExpansion.second.push_back(ExpressionWithValidity(removeUPrefixFromArguments(expr), validityName));

            std::string expandedSignature = expandSignature(entityToTrack);
            ExpressionWithValidity expandVal(expandedSignature, validityName);

            // Using the addOrigin helper handles the std::vector insertion 
            // and respects the max_origins limit automatically.
            addOrigin(memoryBlock.exprOriginMap, expandVal, originExpansion, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            addOrigin(memoryBlock.mailOut.exprOriginMap, expandVal, originExpansion, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

            // 2. Record Disintegration Origin for Children (if requested)
            if (trackChildren) {
                std::pair<std::string, std::vector<ExpressionWithValidity>> originDisintegration;
                originDisintegration.first = "disintegration";
                originDisintegration.second.push_back(ExpressionWithValidity(expandedSignature, validityName));

                for (const auto& elem : entityToTrack.elements) {
                    ExpressionWithValidity elemVal(removeUPrefixFromArguments(elem), validityName);

                    // addOrigin handles the vector push_back and respects compressor_max_origins_per_expr
                    addOrigin(memoryBlock.exprOriginMap, elemVal, originDisintegration, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    addOrigin(memoryBlock.mailOut.exprOriginMap, elemVal, originDisintegration, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                }
            }
        };
    // --------------------------------

    if (ent.category == "implication") {
        assert(!ent.elements.empty() && "Implication category cannot be empty!");

        std::vector<std::string> key(ent.elements.begin(), ent.elements.end() - 1);
        std::string value = ent.elements.back();

        // Add implication specifically to THIS statement's entry (first set)
        collected[currentStatement].first.insert(
            ExpressionWithValidity(reconstructImplicationFullBind(key, value), validityName));

        // Use Lambda (Children not tracked for implication in original code)
        trackExpansionHistory(ent, false);
    }
    else if (ent.category == "and") {
        // Insert elements into the second set, removing the "u_" prefix first
        for (const auto& el : ent.elements) {
            collected[currentStatement].second.insert(removeUPrefixFromArguments(el));
        }

        // Recurse on all elements
        for (const auto& elem : ent.elements) {
            disintegrateExprCore2(elem, instructions, memoryBlock, iteration, collected, newVarMap, validityName, orBranchStatements, trackHistoryLocal, allowOrDisintegration);
        }

        // Use Lambda (Children tracked for AND)
        trackExpansionHistory(ent, true);
    }
    else if (ent.category == "existence")
    {
        std::vector<std::string> removedArgs = listLastRemovedArgsLE(ent);
        if (removedArgs.empty()) {
            for (const auto& elem : ent.elements) {
                disintegrateExprCore2(elem, instructions, memoryBlock, iteration, collected, newVarMap, validityName, orBranchStatements, trackHistoryLocal, allowOrDisintegration);
            }
            return;
        }

        assert(removedArgs.size() == 1);
        std::string boundVar = removedArgs[0];

        // Helper lambda to process a specific path (Unconditional Generation)
        auto processPath = [&](const std::string& newVar, std::vector<LogicalEntity> instructions) {

            // Create Renamed Elements
            std::map<std::string, std::string> replacementMap;
            replacementMap[boundVar] = newVar;

            std::vector<std::string> renamedElements;
            renamedElements.reserve(ent.elements.size());
            for (const auto& elem : ent.elements) {
                renamedElements.push_back(ce::replaceKeysInString(elem, replacementMap));
            }

            // CRUCIAL: Map the new variable to its immediate defining elements
            newVarMap[newVar] = renamedElements;

            std::map<std::string, std::string> replacementMap2;
            assert(!startsWith(boundVar, "u_", 2));
            replacementMap2[boundVar] = "u_" + newVar;

            // Apply replacement map on all signatures and elements of every instruction
            for (LogicalEntity& le : instructions) {
                le.signature = ce::replaceKeysInString(le.signature, replacementMap2);
                for (std::string& elem : le.elements) {
                    elem = ce::replaceKeysInString(elem, replacementMap2);
                }
            }

            // Recurse using the modified instructions
            for (const auto& rel : renamedElements) {
                disintegrateExprCore2(rel, instructions, memoryBlock, iteration, collected, newVarMap, validityName, orBranchStatements, trackHistoryLocal, allowOrDisintegration);
            }
            };

        // Check if at least one element is an operator expression
        bool hasOperator = false;
        for (const std::string& el : ent.elements) {
            std::string core = ce::extractExpression(el);
            if (this->operators.find(core) != this->operators.end()) {
                hasOperator = true;
                break;
            }
        }

        // 1. Generate "it_" path (Algebra) - Conditional
        if (hasOperator)
        {
            std::string newVar = "it_" + std::to_string(iteration) +
                "_lev_" + std::to_string(memoryBlock.level) +
                "_" + std::to_string(memoryBlock.startInt);

            memoryBlock.startInt++;

            if (parameters.trackHistory && trackHistoryLocal)
            {
                std::map<std::string, std::string> replacementMap3;
                assert(!startsWith(boundVar, "u_", 2));
                replacementMap3[boundVar] = newVar;

                LogicalEntity modifiedEnt = ent;
                for (auto& elem : modifiedEnt.elements) {
                    elem = ce::replaceKeysInString(elem, replacementMap3);
                }

                // Use Lambda (Children tracked for Algebra/Operator path)
                trackExpansionHistory(modifiedEnt, true);
            }

            processPath(newVar, instructions);
        }

        // 2. Unconditionally generate "int_" path (Integration)
        {
            std::string newVar = "int_lev_" + std::to_string(memoryBlock.level) +
                "_" + std::to_string(memoryBlock.startInt);


            memoryBlock.startInt++;

            if (parameters.trackHistory && trackHistoryLocal)
            {
                std::map<std::string, std::string> replacementMap3;
                assert(!startsWith(boundVar, "u_", 2));
                replacementMap3[boundVar] = newVar;

                LogicalEntity modifiedEnt = ent;
                for (auto& elem : modifiedEnt.elements) {
                    elem = ce::replaceKeysInString(elem, replacementMap3);
                }

                // Use Lambda (Children tracked for Integration path)
                trackExpansionHistory(modifiedEnt, true);
            }

            processPath(newVar, instructions);
        }
    }
    else if (ent.category == "or") {
        // OR disintegration: compiled validity name approach (LIFO stack).
        // Branch validity = parentValidity_(orSignature)_(disjunct)
        // Mutual exclusion implications go to parent validity (hash memory).
        // Branch disjuncts go to orBranchStatements as ExpressionWithValidity.

        // Check OR nesting depth — count existing "_(or" occurrences in validity name
        int currentOrDepth = 0;
        {
            size_t pos = 0;
            while ((pos = validityName.find("_(or", pos)) != std::string::npos) {
                ++currentOrDepth;
                pos += 4;
            }
        }

        if (currentOrDepth < parameters.max_or_depth) {
            // Compute the OR's expanded De-Morgan form once; used as the
            // disintegration origin for the K mutual-exclusion implications
            // below, mirroring the &/existence pattern in trackExpansionHistory.
            // See D-55.
            std::string expandedOrSignature = expandSignature(ent);
            std::pair<std::string, std::vector<ExpressionWithValidity>> originOrDisintegration;
            originOrDisintegration.first = "disintegration";
            originOrDisintegration.second.push_back(
                ExpressionWithValidity(expandedOrSignature, validityName));

            // Hoisted from the orAdmitted block: the K mutual-exclusion
            // implications below cite expandedOrSignature as their disintegration
            // origin, which requires the matching expansion-origin record
            // (expandedOrSignature -> compact OR name) to exist regardless of
            // whether per-branch case-split fires. Pre-fix, trackExpansionHistory
            // ran only when orAdmitted was true, leaving the K implications
            // referenceable only by an expansion row that did not exist.
            trackExpansionHistory(ent, false);

            // 1. Generate N implications: for each d_i, (!d_0 & ... & !d_{i-1} & !d_{i+1} & ... & !d_{N-1}) -> d_i
            //    These go to collected (parent validity) for normal hash memory insertion.
            //    Each is also stamped with `disintegration` origin pointing at the
            //    OR's expanded form so the proof graph can audit the K rules back
            //    to the originating OR (verifier check_disintegration's `or` branch
            //    rebuilds and verifies the exact shape).
            for (size_t i = 0; i < ent.elements.size(); ++i) {
                std::vector<std::string> premises;
                for (size_t j = 0; j < ent.elements.size(); ++j) {
                    if (j != i) premises.push_back("!" + ent.elements[j]);
                }
                std::string impStr = reconstructImplicationFullBind(premises, ent.elements[i]);
                collected[currentStatement].first.insert(ExpressionWithValidity(impStr, validityName));

                if (parameters.trackHistory && trackHistoryLocal) {
                    // KEY u_-stripped (matches the &/existence pattern in
                    // trackExpansionHistory's section 2 — chapter rows
                    // surface u_-stripped expressions; the u_-prefixed form
                    // lives only in hash memory for unification).
                    ExpressionWithValidity impEv(removeUPrefixFromArguments(impStr), validityName);
                    int maxOrigins = parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr;
                    addOrigin(memoryBlock.exprOriginMap, impEv, originOrDisintegration, maxOrigins);
                    addOrigin(memoryBlock.mailOut.exprOriginMap, impEv, originOrDisintegration, maxOrigins);
                }
            }

            // 2. Check OR admission. Two paths (D-32, supersedes D-31):
            //    (a) Sharper bypass — disintegrate when the caller asserts
            //        allowOrDisintegration. This flag is set by
            //        checkLocalEncodedMemoryStatic only when the firing
            //        implication is itself a "product of disintegration"
            //        (lmv.productOfDisintegration, stamped at install time
            //        in addToHashMemory: at least one premise has an arg
            //        starting with "u_"). Anchor-bound rules — whose chains
            //        carry only concrete integer args — never trigger the
            //        bypass, eliminating the runtime explosion observed
            //        under D-31's broad implication-scope bypass.
            //        Coupled with general disintegration in
            //        addExprToMemoryBlock: doNotDisintegrate forces
            //        allowOrDisintegration=false.
            //    (b) Legacy orAdmissionSet gate — fallback for ORs whose
            //        caller did not set the flag. NOTE: orAdmissionSet has
            //        no .insert() site anywhere in the current codebase,
            //        so the legacy gate is structurally dead; it remains
            //        in place as a no-op fallback / future-extension
            //        point. See D-31 for the diagnosis trail.
            bool orAdmitted = allowOrDisintegration;
            if (!orAdmitted) {
                orAdmitted = true;
                for (size_t i = 0; i < ent.elements.size(); ++i) {
                    std::string cleanExpr = removeUPrefixFromArguments(ent.elements[i]);
                    if (memoryBlock.orAdmissionSet.find(ExpressionWithValidity(cleanExpr, validityName))
                        == memoryBlock.orAdmissionSet.end()) {
                        orAdmitted = false;
                        break;
                    }
                }
            }

            if (orAdmitted) {
                // 3. Build compiled OR signature and register disjunct count
                std::string orSignature = removeUPrefixFromArguments(ent.signature);  // e.g. "(or3[1,2,3])"
                memoryBlock.orDisjunctCount[orSignature] = static_cast<int>(ent.elements.size());

                // (trackExpansionHistory hoisted above the orAdmitted gate —
                //  see the K-implications block earlier in the OR case.)

                // 4. Each disjunct becomes a statement with branch validity name
                for (size_t i = 0; i < ent.elements.size(); ++i) {
                    std::string cleanExpr = removeUPrefixFromArguments(ent.elements[i]);
                    std::string orPayload = std::string("ordis_") + orSignature + "_(" + cleanExpr + ")";
                    int16_t orParentId = memoryBlock.nameMap.encode(validityName);
                    int16_t orBranchId = memoryBlock.nameMap.encodePush(orParentId, orPayload);
                    std::string branchValidity = memoryBlock.nameMap.decode(orBranchId);

                    std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                    origin.first = "or disintegration";
                    origin.second.push_back(ExpressionWithValidity(orSignature, validityName));

                    ExpressionWithValidity branchEv(cleanExpr, branchValidity);
                    if (trackHistoryLocal) {
                        addOrigin(memoryBlock.exprOriginMap, branchEv, origin,
                            (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    }

                    orBranchStatements.push_back(branchEv);
                }
            }
            // else: OR not admitted — implications already emitted above, branches skipped
        }
        // else: max OR depth reached — expression already in collected, no branch processing
    }
    else {
        // Default fallback
        // currentStatement was already inserted at top
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

std::tuple<std::set<ExpressionWithValidity>, std::set<ExpressionWithValidity>, int>
ExpressionAnalyzer::disintegrateExpr2(const std::string& expr,
    Memory& memoryBlock,
    int iteration,
    bool forceDeep,
    std::string validityName,
    bool trackHistoryLocal,
    bool allowOrDisintegration)
{
    int savedStartInt = memoryBlock.startInt;

    std::map<std::string, std::pair<std::set<ExpressionWithValidity>, std::set<std::string>>> collected;
    std::map<std::string, std::vector<std::string>> newVarMap;
    std::vector<ExpressionWithValidity> orBranchStatements;
    Instruction instructions;

    std::string replExpr = prefixArgumentsWithU(expr);
    prepareIntegrationCore(replExpr, instructions, memoryBlock, expr);

    disintegrateExprCore2(replExpr,
        instructions.data,
        memoryBlock,
        iteration,
        collected,
        newVarMap,
        validityName,
        orBranchStatements,
        trackHistoryLocal,
        allowOrDisintegration);

    std::set<ExpressionWithValidity> finalImplications;
    std::set<std::string> finalStringStatements;

    std::function<void(const std::string&)> addToFinal;
    addToFinal = [&](const std::string& s) {
        if (finalStringStatements.count(s)) return;
        finalStringStatements.insert(s);

        // If key is missing, add it to collected
        if (collected.find(s) == collected.end()) {
            collected[s]; // Creates empty pair
        }

        // Now access the entry (guaranteed to exist)
        auto& pr = collected[s];
        finalImplications.insert(pr.first.begin(), pr.first.end());


        // Recurse for every child
        for (const auto& child : pr.second) {
            addToFinal(child);
        }
    };

    if (forceDeep) {
        static const std::regex reIt(R"(^it_\d+_lev_(\d+)_(\d+)$)");

        // 1. Filter and Collect Phase
        for (auto const& [stmt, pr] : collected) {
            bool allow = true;
            std::vector<std::string> args = ce::getArgs(stmt);

            for (const std::string& arg : args) {
                std::smatch match;
                // Rule: If a statement has a new "it_" variable with level == mb.level, do not allow it.
                if (std::regex_match(arg, match, reIt)) {
                    try {
                        int level = std::stoi(match[1].str());
                        int id = std::stoi(match[2].str());

                        if (id >= savedStartInt && level == memoryBlock.level) {
                            allow = false;
                            break;
                        }
                    }
                    catch (...) {}
                }
            }

            if (allow) {
                addToFinal(stmt);
            }
        }

        memoryBlock.canBeSentSet.insert(finalStringStatements.begin(), finalStringStatements.end());
        for (const auto& ev : finalImplications) memoryBlock.canBeSentSet.insert(ev.original);
        // Convert to ExpressionWithValidity (all with parent validity in forceDeep)
        std::set<ExpressionWithValidity> finalStatements;
        for (const auto& s : finalStringStatements)
            finalStatements.insert(ExpressionWithValidity(s, validityName));
        // Add OR branches (if any arose during deep disintegration)
        for (const auto& ev : orBranchStatements)
            finalStatements.insert(ev);
        return std::make_tuple(finalImplications, finalStatements, memoryBlock.startInt);
    }

    // --- FIX: Deferred Rejection Structures ---
    // Algebra-side pending rejection. Carries the same cohort information as
    // PendingRejectionIntegration so the revival path (revisitRejected2) can
    // mail-emit the constituent + siblings without re-running
    // prepareIntegrationCore to recover them. compactExpr + levels are
    // captured at buffer time from the disintegrated compound.
    struct PendingRejection {
        std::string removedU;                 // concreteConstituent (post u_-strip)
        std::string markedExpr;               // marker form (key into rejectedMap)
        std::vector<std::string> siblings;    // other body elements, concrete form
        std::string compactExpr;              // the compound being disintegrated
        std::set<int> levels;                 // levels of the compound at rejection
    };
    // Map var -> list of potential rejections
    std::map<std::string, std::vector<PendingRejection>> pendingRejections;

    // Integration-side counterpart: the rejected constituent plus the full
    // sibling list (incl. the (in[]) typing element) captured at rejection
    // time, so the internalMailIn revival can re-emit the full body together.
    struct PendingRejectionIntegration {
        std::string concreteConstituent;      // removedU
        std::string markedExpr;               // marker form (key into rejectedMapIntegration)
        std::vector<std::string> siblings;    // all OTHER body elements, concrete form
    };
    std::map<std::string, std::vector<PendingRejectionIntegration>> pendingRejectionsIntegration;

    // Track successfully admitted variables in this call
    std::set<std::string> admittedVars;

    static const std::regex reIt(R"(^it_\d+_lev_\d+_\d+$)");
    static const std::regex reInt(R"(^int_lev_\d+_\d+$)");
    std::string topLevelExprClean = removeUPrefixFromArguments(replExpr);

    size_t previousSize = 0;

    previousSize = finalStringStatements.size();

    // Pass A: Unconditional Statements
    for (auto const& [stmt, impls] : collected) {
        if (finalStringStatements.find(stmt) != finalStringStatements.end()) continue;

        std::vector<std::string> args = ce::getArgs(stmt);
        bool hasNewVars = false;
        for (const std::string& arg : args) {
            if (std::regex_match(arg, reIt) || std::regex_match(arg, reInt)) {
                size_t lastUnderscore = arg.rfind('_');
                size_t levPos = arg.find("_lev_"); // Find position of "_lev_" marker

                if (lastUnderscore != std::string::npos && levPos != std::string::npos) {
                    try {
                        // Extract ID (after last underscore)
                        int id = std::stoi(arg.substr(lastUnderscore + 1));

                        // Extract Level (between "_lev_" and last underscore)
                        // "_lev_" has length 5, so the number starts at levPos + 5
                        int varLevel = std::stoi(arg.substr(levPos + 5, lastUnderscore - (levPos + 5)));

                        // Compare both ID and Level
                        if (varLevel == memoryBlock.level && id >= savedStartInt) {
                            hasNewVars = true;
                            break;
                        }
                    }
                    catch (...) {}
                }
            }
        }

        if (!hasNewVars) {
			addToFinal(stmt);
        }
    }

    if (!parameters.compressor_mode && !parameters.ban_disintegration) {
        // Pass B: New Variable Admission
        // Gated by !parameters.ban_disintegration (collapsed from short-lived
        // allow_disintegration flag on 2026-04-29 — see D-28; original gate
        // was !parameters.incubator_mode pre-2026-04-29). See I-7.
        for (const auto& [var, expressions] : newVarMap)
        {
            bool isVarAdmitted = false;

            for (const auto& stmt : expressions) {
                std::string removedU = removeUPrefixFromArguments(stmt);
                std::string markedExpr = makeMarkedExpr(removedU, var);

                // 1. Check for Iteration Variable (it_...)
                if (std::regex_match(var, reIt)) {
                    std::string core = ce::extractExpression(removedU);
                    bool hasOperator = (this->operators.find(core) != this->operators.end());

                    if (hasOperator) {
                        if (isAdmitted(memoryBlock, removedU, var, markedExpr, validityName)) {
                            isVarAdmitted = true;
                            break;
                        }
                        else if (isAllowedAsOperatorInput(memoryBlock, removedU, var)) {
                            isVarAdmitted = true;
                            break;
                        }
                        else {
                            // Buffer rejection instead of calling updateRejectedMap immediately.
                            // Capture the cohort (other expressions sharing this it_ var)
                            // and the parent compound's levels so the revival path can
                            // mail-emit the full body without re-running
                            // prepareIntegrationCore. Symmetric to the int_ branch below.
                            std::vector<std::string> siblings;
                            siblings.reserve(expressions.size() > 0 ? expressions.size() - 1 : 0);
                            for (const auto& other : expressions) {
                                if (&other == &stmt) continue;
                                siblings.push_back(removeUPrefixFromArguments(other));
                            }
                            std::set<int> lvls;
                            {
                                EncodedExpression encCompound(topLevelExprClean, validityName);
                                auto itLev = memoryBlock.statementLevelsMap.find(encCompound);
                                if (itLev != memoryBlock.statementLevelsMap.end()) {
                                    lvls = itLev->second;
                                }
                            }
                            PendingRejection pr;
                            pr.removedU    = removedU;
                            pr.markedExpr  = markedExpr;
                            pr.siblings    = std::move(siblings);
                            pr.compactExpr = topLevelExprClean;
                            pr.levels      = std::move(lvls);
                            pendingRejections[var].push_back(std::move(pr));
                        }
                    }
                }
                // 2. Check for Integration Variable (int_...)
                else if (std::regex_match(var, reInt)) {
                    if (isAdmittedIntegration(memoryBlock, removedU, var, markedExpr, validityName)) {
                        isVarAdmitted = true;
                        break;
                    }
                    else if (memoryBlock.overallHashMemory.admissionSetIntegration.find(ExpressionWithValidity(markedExpr, validityName)) !=
                        memoryBlock.overallHashMemory.admissionSetIntegration.end()) {
                        cleanAdmissionMap(markedExpr, validityName, memoryBlock);
                        isVarAdmitted = true;
                        break;
                    }
                    else {
                        // Buffer rejection symmetric to the it_ branch above.
                        // Skip (in[...]) typing elements — per user spec, they
                        // never become a rejectedMapIntegration key (siblings
                        // still carry them for re-emission).
                        const std::string core = ce::extractExpression(removedU);
                        const bool isInTyping = (core == "in");
                        if (!isInTyping) {
                            std::vector<std::string> siblings;
                            siblings.reserve(expressions.size() > 0 ? expressions.size() - 1 : 0);
                            for (const auto& other : expressions) {
                                if (&other == &stmt) continue;
                                siblings.push_back(removeUPrefixFromArguments(other));
                            }
                            PendingRejectionIntegration pri;
                            pri.concreteConstituent = removedU;
                            pri.markedExpr          = markedExpr;
                            pri.siblings            = std::move(siblings);
                            pendingRejectionsIntegration[var].push_back(std::move(pri));
                        }
                    }
                }
            }

            if (isVarAdmitted) {
                admittedVars.insert(var); // Mark as locally admitted

                for (const auto& stmt : expressions) {
                    std::string cleanStmt = removeUPrefixFromArguments(stmt);
                    addToFinal(cleanStmt);
                }
            }
        }
    }

    // ---------------------------------------------------------
    // Cascade Admission for Dependent Variables (int_ and it_)
    // ---------------------------------------------------------

    std::string sLevel = std::to_string(memoryBlock.level);
    std::string intPrefix = "int_lev_" + sLevel + "_";
    std::string levSubstring = "_lev_" + sLevel + "_";

    while (true) {
        bool hasGrown = false;

        // Iterate through all potential statements in 'collected'
        for (const auto& stmt : finalStringStatements)
        {
            std::vector<std::string> args = ce::getArgs(stmt);

            bool hasAdmittedVar = false;
            std::vector<std::string> candidates;

            // First pass: Classify arguments in this statement
            for (const std::string& arg : args) {
                // Check if already admitted
                if (admittedVars.find(arg) != admittedVars.end()) {
                    hasAdmittedVar = true;
                }
                // Check if it is a potential candidate (Target Syntax + New + Not Admitted)
                else {
                    bool isTarget = false;
                    // Check A: int_lev_X_
                    if (arg.rfind(intPrefix, 0) == 0) {
                        isTarget = true;
                    }
                    // Check B: it_..._lev_X_
                    else if (arg.size() > 3 && arg[0] == 'i' && arg[1] == 't' && arg[2] == '_') {
                        if (arg.find(levSubstring) != std::string::npos) {
                            isTarget = true;
                        }
                    }

                    if (isTarget) {
                        // Must be in newVarMap (new) and not yet admitted
                        if (newVarMap.find(arg) != newVarMap.end()) {
                            candidates.push_back(arg);
                        }
                    }
                }
            }

            // Admission Rule: New vars are admitted ONLY if they share an expression with an admitted var
            if (hasAdmittedVar && !candidates.empty()) {
                for (const auto& cand : candidates) {
                    // Double-check to ensure we don't process the same candidate twice in one pass
                    if (admittedVars.find(cand) == admittedVars.end()) {

                        admittedVars.insert(cand);
                        hasGrown = true;

                        // Pull in all defining expressions for this newly admitted variable
                        // Note: We use operator[] safely here because we verified existence in newVarMap above
                        for (const auto& rawExpr : newVarMap.find(cand)->second) {
                            std::string cleanStmt = removeUPrefixFromArguments(rawExpr);
                            addToFinal(cleanStmt);
                        }
                    }
                }
            }
        }

        // Break if no new variables were admitted in this pass
        if (!hasGrown) {
            break;
        }
    }



    // --- FIX: Commit Rejections Only for Never-Admitted Variables ---
    for (const auto& [var, failures] : pendingRejections) {
        if (admittedVars.find(var) == admittedVars.end()) {

            // Variable was truly rejected (dependencies never resolved)
            for (const auto& fail : failures) {
                updateRejectedMap(fail.removedU,
                                  fail.markedExpr,
                                  fail.compactExpr,
                                  fail.removedU,         // concreteConstituent (= removedU)
                                  fail.siblings,
                                  fail.levels,
                                  memoryBlock.overallHashMemory,
                                  iteration,
                                  validityName);
            }
        }
    }

    // Integration-side commit — mirror of algebra block above. A pending
    // integration-rejection is committed to rejectedMapIntegration only if
    // the int_ var stayed un-admitted after cascade admission.
    for (const auto& [var, failures] : pendingRejectionsIntegration) {
        if (admittedVars.find(var) == admittedVars.end()) {
            for (const auto& fail : failures) {
                updateRejectedMapIntegration(fail.markedExpr,
                                             fail.concreteConstituent,
                                             fail.siblings,
                                             topLevelExprClean,
                                             memoryBlock.overallHashMemory,
                                             validityName);
            }
        }
    }

    // Convert to ExpressionWithValidity (normal stmts with parent validity)
    std::set<ExpressionWithValidity> finalStatements;
    for (const auto& s : finalStringStatements)
        finalStatements.insert(ExpressionWithValidity(s, validityName));
    // Add OR branches (with their own branch validity names)
    for (const auto& ev : orBranchStatements)
        finalStatements.insert(ev);

    return std::make_tuple(finalImplications, finalStatements, memoryBlock.startInt);
}






void ExpressionAnalyzer::overwriteOrigins(std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>& left,
    const std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>& right,
    int maxOrigins) {
    for (auto it = right.begin(); it != right.end(); ++it) {
        auto& leftVec = left[it->first];
        for (const auto& orig : it->second) {
            if (leftVec.size() < static_cast<size_t>(maxOrigins)) {
                if (std::find(leftVec.begin(), leftVec.end(), orig) == leftVec.end()) {
                    leftVec.push_back(orig);
                }
            }
        }
    }
}




// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void ExpressionAnalyzer::prove(int numberIterations,
    const std::vector<Memory*>& bodies,
    const ParentChildrenMap& index,
    PerCoreMailboxes& boxes) {

    for (int it = 0; it < numberIterations; ++it) {
        // Debug: count total expressions across all active bodies
        std::size_t totalExprs = 0;
        std::size_t activeBodies = 0;
        for (std::size_t bi = 0; bi < bodies.size(); ++bi) {
            if (bodies[bi] && bodies[bi]->isActive) {
                totalExprs += bodies[bi]->encodedStatements.size();
                ++activeBodies;
            }
        }
        std::cout << "Hash burst: " << it
                  << "  active_bodies=" << activeBodies
                  << "  total_exprs=" << totalExprs << std::endl;

        auto burstStart = std::chrono::high_resolution_clock::now();
        proveKernel(bodies, index, boxes);                     // runs, then smashMail(boxes)
        auto burstEnd = std::chrono::high_resolution_clock::now();
        double burstSec = std::chrono::duration<double>(burstEnd - burstStart).count();
        std::cout << "  dt=" << burstSec << "s" << std::endl;
    }
}

// readSimpleFacts() and saveFilteredConjectures() — moved to filter.cpp.

void ExpressionAnalyzer::saveProvedTheorems() {
    namespace fs = std::filesystem;

    // Resolve path: <repo>/files/theorems/proved_theorems.txt
    const auto theoremsDir =
        fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
        / "files" / "theorems";

    const auto filePath = theoremsDir / "proved_theorems.txt";

    // Create directory if it doesn't exist
    std::error_code ec;
    fs::create_directories(theoremsDir, ec);

    // Open in append mode
    std::ofstream ofs(filePath, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filePath << " for appending." << std::endl;
        return;
    }

    int count = 0; // Move counter outside to use it after unlocking

    // START CHANGE: Explicit scope block
    {
        std::lock_guard<std::mutex> lock(this->theoremListMutex);

        for (const auto& entry : this->globalTheoremList) {
            // entry is tuple<theorem, method, var, recCounter>
            const std::string& theorem = std::get<0>(entry);
            const std::string& method = std::get<1>(entry);

            // Filter out debug/internal entries if necessary, or save all.
            // "debug" method entries are usually from findEnds and shouldn't be saved as proved theorems.
            if (method != "debug") {
                ofs << theorem << "\n";
                count++;
            }
        }
    }
    // END CHANGE: lock is released here, before the function exits or prints to console

    std::cout << "Saved " << count << " new proved theorems to file." << std::endl;
}

void ExpressionAnalyzer::saveProvedTheoremsFiltered(const std::vector<std::string>& essentialTheorems,
                                                    const std::unordered_set<std::string>& externalTheorems) {
    namespace fs = std::filesystem;

    const auto theoremsDir =
        fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
        / "files" / "theorems";

    const auto filePath = theoremsDir / "proved_theorems.txt";

    std::error_code ec;
    fs::create_directories(theoremsDir, ec);

    // Build compact→expanded lookup from globalTheoremList.
    // globalTheoremList stores expanded forms; the compressor may have
    // reverse-compiled existence heads to compact form.
    // We compile each globalTheoremList entry the same way the compressor
    // constructor does, then map back.
    std::unordered_map<std::string, std::string> compactToExpanded;
    for (const auto& tpl : globalTheoremList) {
        const std::string& expanded = std::get<0>(tpl);
        compactToExpanded[expanded] = expanded;  // identity: already expanded

        // Also try reverse-compiling to build compact key
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
        std::string head = ce::disintegrateImplication(expanded, tempChain, coreExpressionMap);

        if (startsWith(head, "!(>", 3)) {
            std::string compiledHead = compileCoreExpressionMapCore(
                head, implCounter, existenceCounter, andCounter, orCounter, variableCounter);
            std::vector<std::string> chain;
            for (const auto& t : tempChain) chain.push_back(std::get<0>(t));
            std::string compact = reconstructImplication(chain, compiledHead);
            compactToExpanded[compact] = expanded;
        }
    }

    std::ofstream ofs(filePath, std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filePath << " for writing." << std::endl;
        return;
    }

    // Build set of mirrored-of-external theorems to also exclude
    std::unordered_set<std::string> mirroredOfExternal;
    if (!externalTheorems.empty()) {
        for (const auto& tpl : globalTheoremList) {
            const std::string& expr = std::get<0>(tpl);
            const std::string& method = std::get<1>(tpl);
            const std::string& ref = std::get<2>(tpl);
            if (method == "mirrored statement" && externalTheorems.count(ref)) {
                mirroredOfExternal.insert(expr);
            }
        }
    }

    // Second file: compiled forms (for proof graph pruning — keeps or0, existence2, etc.)
    const auto compiledPath = theoremsDir / "compiled_proved_theorems.txt";
    std::ofstream ofsCompiled(compiledPath, std::ios::trunc);

    int written = 0;
    for (const std::string& theorem : essentialTheorems) {
        // Skip surviving external theorems and their mirrors — they are not GL-proved
        if (externalTheorems.count(theorem) || mirroredOfExternal.count(theorem)) continue;

        auto it = compactToExpanded.find(theorem);
        std::string toWrite = (it != compactToExpanded.end()) ? it->second : theorem;

        // compiled_proved_theorems.txt: write as-is (keeps compiled heads)
        if (ofsCompiled.is_open()) ofsCompiled << toWrite << "\n";

        // proved_theorems.txt: expand runtime expressions (or0, etc.) to base form
        // so subsequent batches can parse them.
        toWrite = expandToBaseForm(toWrite);
        ofs << toWrite << "\n";
        ++written;
    }

    std::cout << "Rewrote proved_theorems.txt with " << written
        << " globally essential theorems (excluded "
        << (externalTheorems.size() + mirroredOfExternal.size())
        << " external/mirrored-of-external)." << std::endl;
}


// Rewrites `theorem` in place so that every structural !(&...) and
// !(>...) sub-expression is replaced by its compiled name
// (or<N> / existence<N>). Idempotent.
//
// Must run BEFORE any disintegrateImplication / addTheoremToMemory /
// broadcastTheorems call — otherwise raw !(&...) / !(>...) fragments end
// up as LB exprKeys or mail payloads and crash every downstream core that
// assumes ce::extractExpression(exprKey) maps into compiledExpressions.
//
// Substring-scan approach (no disintegrate+reconstruct): we never rebuild
// the outer (>...) chain, so bound-variable lists in the original theorem
// stay intact even when compileCoreExpressionMapCore renames the bound
// variable inside a !(>) via renameLastRemoved.
void ExpressionAnalyzer::precompileStructuralOperators(std::string& theorem) {
    auto compileAtMarker = [&](const char* marker /* "!(&" or "!(>" */) {
        while (true) {
            auto pos = theorem.find(marker);
            if (pos == std::string::npos) return;

            // Bracket-balanced scan starting at the '(' right after '!'.
            int depth = 0;
            std::size_t end = std::string::npos;
            for (std::size_t i = pos + 1; i < theorem.size(); ++i) {
                if (theorem[i] == '(') {
                    ++depth;
                } else if (theorem[i] == ')') {
                    --depth;
                    if (depth == 0) { end = i; break; }
                }
            }
            if (end == std::string::npos) return;  // malformed, bail

            std::string sub = theorem.substr(pos, end - pos + 1);
            std::string compiled = this->compileCoreExpressionMapCore(
                sub, this->implCounter, this->existenceCounter,
                this->andCounter, this->orCounter, this->variableCounter);

            if (compiled.empty() || compiled == sub) return;  // no progress
            theorem = theorem.substr(0, pos) + compiled + theorem.substr(end + 1);
        }
    };

    // Pass 1: compile all !(&...) — these become (or<N>[...]).
    compileAtMarker("!(&");
    // Pass 2: compile all !(>...) — these become (existence<N>[...]).
    compileAtMarker("!(>");
}


std::string ExpressionAnalyzer::headSwitchOne(const std::string& theorem) const {
    // Stateless head-switch (contrapositive) construction.
    // Returns the rebuilt implication string if the chain has a negated,
    // non-quantified premise; returns an empty string otherwise. The byte-
    // level construction is identical to the one inlined inside the
    // big-iteration loop in analyzeExpressions: same disintegrateImplication
    // call, same rightmost-negated-premise scan, same manual `(>[vars]…)`
    // concatenation.
    std::vector<std::tuple<std::string, std::vector<std::string>,
                           std::set<std::string>>> chain;
    std::string head = ce::disintegrateImplication(theorem, chain, coreExpressionMap);

    int negIdx = -1;
    for (int i = (int)chain.size() - 1; i >= 0; --i) {
        const std::string& prem = std::get<0>(chain[i]);
        if (!prem.empty() && prem[0] == '!') {
            if (std::get<1>(chain[i]).empty()) negIdx = i;
            break;
        }
    }
    if (negIdx < 0) return std::string();

    const std::string& negPrem = std::get<0>(chain[negIdx]);
    std::string newHead = negPrem.substr(1);
    std::string newPrem = "!" + head;

    std::string rebuilt = newHead;
    for (int i = (int)chain.size() - 1; i >= 0; --i) {
        const auto& link = chain[i];
        const std::string& premise = (i == negIdx) ? newPrem : std::get<0>(link);
        const std::vector<std::string>& vars = std::get<1>(link);

        std::string vStr;
        for (size_t v = 0; v < vars.size(); ++v) {
            if (v > 0) vStr += ",";
            vStr += vars[v];
        }
        rebuilt = "(>[" + vStr + "]" + premise + rebuilt + ")";
    }
    return rebuilt;
}


void ExpressionAnalyzer::broadcastTheorems(const std::vector<std::string>& provedTheorems,
                                           const std::string& originTag) {
    if (provedTheorems.empty()) return;

    Mail broadcastMail;
    for (const std::string& thOriginal : provedTheorems) {

        // 1. Disintegrate the theorem to inspect its head
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
        std::string head = ce::disintegrateImplication(thOriginal, tempChain, this->coreExpressionMap);

        std::string finalTheorem = thOriginal;
        std::string finalHead = head;
        std::vector<std::string> finalChain;
        finalChain.reserve(tempChain.size());
        for (const auto& t : tempChain) finalChain.push_back(std::get<0>(t));

        // 2. Check if the head is an existence statement (starts with "!(>")
        if (startsWith(head, "!(>", 3)) {
            // Compile the head
            std::string compiledHead = this->compileCoreExpressionMapCore(
                head,
                this->implCounter,
                this->existenceCounter,
                this->andCounter,
                this->orCounter,
                this->variableCounter
            );

            // Reconstruct the theorem with the original chain and the NEW compiled head
            finalTheorem = this->reconstructImplication(finalChain, compiledHead);

            // Re-disintegrate the NEW theorem to ensure the chain/head structure is perfectly consistent
            // (e.g. reconstructImplication might reorder premises, though typically it preserves them)
            tempChain.clear();
            finalChain.clear();
            finalHead = ce::disintegrateImplication(finalTheorem, tempChain, this->coreExpressionMap);
            for (const auto& t : tempChain) finalChain.push_back(std::get<0>(t));
        }

        // 3. Queue the (potentially modified) theorem for broadcast
        broadcastMail.implications.insert(std::make_tuple(
            finalChain,
            finalHead,
            std::set<std::string>(), // remainingArgsKey (empty for global facts)
            std::set<int>(),         // levels (empty for global facts)
            finalTheorem
        ));

        if (parameters.trackHistory) {
            ExpressionWithValidity ev(finalTheorem, "main");
            auto originBroadcast = std::make_pair(originTag, std::vector<ExpressionWithValidity>());

            addOrigin(broadcastMail.exprOriginMap, ev, originBroadcast, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }
    }

    // Distribute to all permanent bodies
    for (Memory* mb : permanentBodies) {
        if (mb == nullptr) continue;

        // Ensure mailbox has slots
        if (boxes[mb].empty()) {
            boxes[mb].resize(logicalCores);
        }

        // Push to slot 0 (will be picked up by the corresponding worker)
        Mail& dest = boxes[mb][0];
        dest.implications.insert(broadcastMail.implications.begin(), broadcastMail.implications.end());

        if (parameters.trackHistory) {
            dest.exprOriginMap.insert(broadcastMail.exprOriginMap.begin(), broadcastMail.exprOriginMap.end());
        }
    }

    std::cout << "Distributed knowledge to " << permanentBodies.size() << " memory blocks." << std::endl;
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void ExpressionAnalyzer::prefillIntegrationMapsRecursive(Memory* mb) {
    if (mb == nullptr) return;
    if (mb->isPartOfRecursion)
    {
        return;
    }

    // Collect expression keys from the current block and all its predecessors
    std::vector<std::string> exprKeys;
    Memory* current = mb;
    while (current != nullptr) {
        if (!current->exprKey.empty()) {
            exprKeys.push_back(current->exprKey);
        }
        current = current->parentMemory;
    }

    // Process all collected expressions for the current block *mb
    for (const std::string& expr : exprKeys) {
        std::string core = ce::extractExpression(expr);
        if (this->operators.find(core) != this->operators.end()) {
            auto it = this->coreExpressionMap.find(core);
            assert(it != this->coreExpressionMap.end());

            auto itComp = this->compiledExpressions.find(core);
            assert(itComp != this->compiledExpressions.end());

            if (itComp->second.category != "atomic") {
                const ce::CoreExpressionConfig& cfg = it->second;
                assert(!cfg.outputIndices.empty());

                std::vector<std::string> args = ce::getArgs(expr);
                // Assuming the operator has one output variable as implied by context
                int outIdx = cfg.outputIndices[0];

                assert(outIdx >= 0 && outIdx < static_cast<int>(args.size()));

                std::string outputArg = args[outIdx];

                std::map<std::string, std::string> replacementMap;
                replacementMap[outputArg] = "marker";
                std::string rplExpr2 = ce::replaceKeysInString(expr, replacementMap);

                std::vector<std::string> tempArgs = ce::getArgs(rplExpr2);
                std::set<std::string> argSet(tempArgs.begin(), tempArgs.end());

                // Per instructions: erase marker from argSet and call prepareIntegration
                assert(argSet.find("marker") != argSet.end());

                argSet.erase("marker");
                prepareIntegration(rplExpr2, argSet, *mb, "main");
            }
        }
    }

    // Recurse into children
    for (std::map<std::string, Memory*>::iterator it = mb->simpleMap.begin(); it != mb->simpleMap.end(); ++it) {
        prefillIntegrationMapsRecursive(it->second);
    }
}

//#pragma optimize("", off)

void ExpressionAnalyzer::disintegrateExprHypothetically(const std::string& expr, Memory& memoryBlock, std::string validityName) {
    if (parameters.ban_disintegration) return;

    // 1. Extract input variables
    std::vector<std::string> inputVars = ce::getArgs(expr);

	for (const auto& var : inputVars) {
        if (var == "marker") {
            return; // Invalid variable name
        }
    }

    // 2. Call disintegrateExpr2 FIRST
    // We only need the structural products — no history writes.
    //
    // Dual defense:
    //   (a) trackHistoryLocal=false suppresses all writes inside the
    //       trackExpansionHistory lambda, the negated-existence-expansion
    //       block, and the "or disintegration" addOrigin (primary fix).
    //   (b) We hand disintegrateExpr2 a SENTINEL validity so that *if*
    //       anything ever leaks into exprOriginMap / mailOut despite (a),
    //       the leaked entries are trivially identifiable and assert-caught
    //       in buildStack. The sentinel embeds the real validityName for
    //       traceability.
    //
    // IMPORTANT: the rest of this function must keep using the original
    // `validityName` (for statementLevelsMap / equivalenceClassesMap lookups
    // and for constructing `newValidityName`). The sentinel stays local.
    // Sentinel payload embeds the integration goal (expr) so any leaked
    // deposits at sentinel scope are self-identifying. The
    // "product_of_hypo_disintegration_of_integration_goal_" marker lets
    // cross-scope consumers (buildStack, Site A descendant iteration, proof-
    // graph rendering) exclude sentinel-scope orphans without pattern
    // guessing or substring heuristics on random tokens.
    int16_t parentValId = memoryBlock.nameMap.encode(validityName);
    int16_t sentinelValId = memoryBlock.nameMap.encodePush(
        parentValId,
        "product_of_hypo_disintegration_of_integration_goal_" + expr);
    std::string sentinelValidity = memoryBlock.nameMap.decode(sentinelValId);
    auto result = disintegrateExpr2(expr, memoryBlock, -1, true, sentinelValidity, /*trackHistoryLocal=*/false);

    // Only look at new statements (index 1)
    const std::set<ExpressionWithValidity>& newStatementsEv = std::get<1>(result);

    if (newStatementsEv.size() == 1)
    {
        return;
    }
    // Extract just the expression strings for downstream use
    std::set<std::string> newStatements;
    for (const auto& ev : newStatementsEv) newStatements.insert(ev.original);

    // 3. Search for defSets for each input variable in the new statements
    std::vector<std::string> targetVars; // Variables with defSet == "(1)"

    for (const std::string& var : inputVars) {
        bool foundDef = false;
        std::string foundDefSetVal;

        for (const std::string& stmt : newStatements) {
            std::string coreName = ce::extractExpression(stmt);
            auto itCfg = this->coreExpressionMap.find(coreName);

            // "search for at least one expression which has entry in coreExpressionMap"
            if (itCfg != this->coreExpressionMap.end()) {
                const auto& cfg = itCfg->second;
                std::vector<std::string> sArgs = ce::getArgs(stmt);

                // Check if 'var' is an argument in this statement
                for (size_t k = 0; k < sArgs.size(); ++k) {
                    if (sArgs[k] == var) {
                        // "there read defSet"
                        std::string argPos = std::to_string(k + 1);
                        auto itDS = cfg.definitionSets.find(argPos);

                        if (itDS != cfg.definitionSets.end()) {
                            foundDefSetVal = itDS->second.first;
                            foundDef = true;
                            goto def_found; // Break out of statement loop
                        }
                    }
                }
            }
        }

    def_found:
        // "assert u found def set for each var"
        assert(foundDef);

        // "if there is at least one eligible var proceed"
        // Eligible means defSet == "(1)" based on context.
        // Deduplicate: if expr has the same variable in multiple argument
        // positions (e.g. (f[v,v,w])), inputVars contains duplicates.
        // Each variable must appear in newValidityName only once.
        if (foundDefSetVal == "(1)") {
            if (std::find(targetVars.begin(), targetVars.end(), var) == targetVars.end()) {
                targetVars.push_back(var);
            }
        }
    }

    // "otherwise return"
    if (targetVars.empty()) {
        return;
    }

    // 4. Create a new validity name
    // Format: old + "_boundary_" + "_var0_x_var1_y_hypo_expr"
    // Example: main_boundary__var0_x_var1_y_hypo_(in[x,1])
    std::string hypoPayload;
    for (size_t i = 0; i < targetVars.size(); ++i) {
        hypoPayload += "_var" + std::to_string(i) + "_" + targetVars[i];
    }
    hypoPayload += "_hypo_" + expr;
    int16_t hypoParentId = memoryBlock.nameMap.encode(validityName);
    int16_t hypoNewValidityId = memoryBlock.nameMap.encodePush(hypoParentId, hypoPayload);
    std::string newValidityName = memoryBlock.nameMap.decode(hypoNewValidityId);

    // 5. Find levels
    std::set<int> levels;
    EncodedExpression originEnc(expr, validityName);
    auto itLev = memoryBlock.statementLevelsMap.find(originEnc);

    if (itLev != memoryBlock.statementLevelsMap.end()) {
        levels = itLev->second;
    }
    else {
        levels.insert(memoryBlock.level);
    }
    levels.insert(memoryBlock.level);

    // 6. Check Duplication and Add Equalities
    bool anyDuplicationFound = false;

    for (const std::string& var : targetVars) {
        bool foundDuplicationForVar = false;

        // Check duplication in newStatements
        for (const std::string& e : newStatements) {
            std::vector<std::string> eArgs = ce::getArgs(e);
            int count = 0;
            for (const std::string& arg : eArgs) {
                if (arg == var) count++;
            }
            if (count > 1) {
                foundDuplicationForVar = true;
                break;
            }
        }

        if (foundDuplicationForVar) {
            anyDuplicationFound = true;

            // Check if equality exists in validityName (not "main")
            bool hasExistingEquality = false;
            auto itMap = memoryBlock.equivalenceClassesMap.find(validityName);
            if (itMap != memoryBlock.equivalenceClassesMap.end()) {
                for (const auto& ec : itMap->second) {
                    if (ec.variables.count(var) && ec.variables.size() > 1) {
                        hasExistingEquality = true;
                        break;
                    }
                }
            }

            if (!hasExistingEquality) {
                std::string Y = var;
                std::string Y_copy = Y + "_copy";
                std::string equalityExpr = "(=[" + Y + "," + Y_copy + "])";
                std::string finalExpr = removeUPrefixFromArguments(equalityExpr);

                // Dead-end axiom — see "variable copy" tag
                std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                origin.first = "variable copy";

                addExprToMemoryBlock(finalExpr, memoryBlock, -1, 0, levels, origin, -1, -1, validityName, false);
            }
        }
    }

    // 7. Register NEW STATEMENTS with NEW validity name

    for (const std::string& stmt : newStatements) {
        EncodedExpression enc(stmt, newValidityName);

        if (memoryBlock.wholeExpressions.find(enc) != memoryBlock.wholeExpressions.end()) {
            continue; // Skip if already present
		}

        memoryBlock.encodedStatements.push_back(enc);
        memoryBlock.localEncodedStatements.push_back(enc);
        memoryBlock.localEncodedStatementsSet.insert(enc);
        memoryBlock.localEncodedStatementsDelta.push_back(enc);
        { IntEncodedExpr ie = encodeExpression(enc, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);
          memoryBlock.intLocalEncodedStatements.push_back(ie);
          memoryBlock.intLocalEncodedStatementsDelta.push_back(ie); }
        memoryBlock.statementLevelsMap[enc] = levels;
        memoryBlock.intKnownStatements.insert(packStatementKey(
            memoryBlock.nameMap.encode(enc.original),
            memoryBlock.nameMap.encode(enc.validityName)));
        memoryBlock.wholeExpressions.insert(enc);
    }
}

//#pragma optimize("", off)

void ExpressionAnalyzer::prehandleAnchor(Memory* mb) {
    if (mb == nullptr) return;
    if (mb->isPartOfRecursion)
    {
        return;
    }

    std::string anchorPrefix = "(" + this->anchorInfo.name;
    bool isAnchorLB = (mb->exprKey.rfind(anchorPrefix, 0) == 0);

    // Skip processing for the Anchor LB itself, but allow recursion
    if (!isAnchorLB) {

        // 1. Trace the hierarchy to find the specific Anchor Key
        std::string anchorExprKey;
        Memory* current = mb;
        std::set<std::string> traceVariables; // Added: Collect variables from trace

        while (current != nullptr) {
            if (!current->exprKey.empty()) {
                // Check if this ancestor is the Anchor LB
                if (current->exprKey.rfind(anchorPrefix, 0) == 0) {
                    anchorExprKey = current->exprKey;
                    break;
                }
                // Added: Collect variables from the current trace element's key
                std::vector<std::string> kArgs = ce::getArgs(current->exprKey);
                traceVariables.insert(kArgs.begin(), kArgs.end());
            }
            current = current->parentMemory;
        }

        // 2. If we found an anchor ancestor, proceed
        if (!anchorExprKey.empty()) {
            std::vector<std::string> args = ce::getArgs(anchorExprKey);
            std::map<std::string, std::string> replacementMap;

            // 3. Create Replacement Map based on definitionSets == "(1)"
            //    and save the new variables to axedVariables
            for (const auto& [slot, pattern] : this->anchorInfo.definitionSets) {
                if (pattern == "(1)") {
                    try {
                        int index = std::stoi(slot) - 1;
                        if (index >= 0 && index < static_cast<int>(args.size())) {
                            std::string originalVar = args[index];

                            // Added: Check if the variable exists in the trace
                            if (traceVariables.find(originalVar) != traceVariables.end()) {
                                // Only apply x-prefix if not already present
                                if (originalVar.rfind("x", 0) != 0) {
                                    std::string xVar = "x" + originalVar;
                                    replacementMap[originalVar] = xVar;

                                    // Save to the new attribute
                                    mb->axedVariables.insert(xVar);
                                    mb->intAxedVariables.insert(mb->nameMap.encode(xVar));
                                }
                            }
                        }
                    }
                    catch (...) {
                        // Ignore malformed slots
                    }
                }
            }

            // 4. Create and Add the Anchor Expression using the map
            if (!replacementMap.empty()) {
                std::string replacedAnchor = ce::replaceKeysInString(anchorExprKey, replacementMap);

                EncodedExpression enc(replacedAnchor, "main");

                // Add only if not already present
                if (mb->wholeExpressions.find(enc) == mb->wholeExpressions.end()) {

                    std::set<int> levels{ 0 };

                    mb->localEncodedStatements.push_back(enc);
                    mb->localEncodedStatementsSet.insert(enc);
                    mb->localEncodedStatementsDelta.push_back(enc);
                    mb->encodedStatements.push_back(enc);
                    { IntEncodedExpr ie = encodeExpression(enc, mb->nameMap);
                      mb->intEncodedStatements.push_back(ie);
                      mb->intLocalEncodedStatements.push_back(ie);
                      mb->intLocalEncodedStatementsDelta.push_back(ie); }
                    mb->statementLevelsMap[enc] = levels;
                    mb->intKnownStatements.insert(packStatementKey(
                        mb->nameMap.encode(enc.original),
                        mb->nameMap.encode(enc.validityName)));
                    mb->wholeExpressions.insert(enc);

                    std::pair<std::string, std::vector<ExpressionWithValidity>> origin;
                    origin.first = "anchor handling";
                    // Cite the original anchor expression as the source
                    origin.second.push_back(ExpressionWithValidity(anchorExprKey, "main"));

                    ExpressionWithValidity encVal(replacedAnchor, "main");

                    // addOrigin handles the vector push_back and respects compressor_max_origins_per_expr
                    addOrigin(mb->exprOriginMap, encVal, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    // Also push the history line into mailOut so it propagates
                    // to descendant LBs — recursion blocks (isPartOfRecursion=true)
                    // are skipped at the top of this function and never register
                    // the axed anchor as a statement themselves, but implication
                    // firings inside them can still cite the axed anchor as an
                    // ingredient (the form is generated by hash propagation from
                    // this LB). Without the mailed history line, buildStack would
                    // crash with "no origin found" when walking back through that
                    // ingredient. We do not register the axed anchor as a
                    // statement on recursion LBs (see top-of-function early
                    // return) — only the history line travels.
                    addOrigin(mb->mailOut.exprOriginMap, encVal, origin, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                }
            }
        }
    }

    // 5. Recurse into children
    for (auto it = mb->simpleMap.begin(); it != mb->simpleMap.end(); ++it) {
        prehandleAnchor(it->second);
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void ExpressionAnalyzer::analyzeExpressions(const std::vector<std::string>& theorems,
    const std::vector<std::string>& provedTheorems,
    const std::vector<std::string>& externalTheorems) {

    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "Counter example based filtering started." << std::endl;

    // 1. CE Filtering
    std::vector<std::string> filteredConjectures = theorems;

    if (!parameters.skip_ce_filter) {
        std::vector<std::vector<std::string>> simpleFactsLists = readSimpleFacts();
        std::cout << "CE filter: " << filteredConjectures.size() << " conjectures, "
                  << simpleFactsLists.size() << " fact files." << std::endl;
        for (const auto& simpleFacts : simpleFactsLists)
        {
            filteredConjectures = filterConjecturesWithCE(filteredConjectures, simpleFacts);
            std::cout << "CE filter: " << filteredConjectures.size() << " survived." << std::endl;
        }
        saveFilteredConjectures(filteredConjectures);
    }

    std::cout << "Counter example based filtering finished." << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    std::cout << "Runtime counter example based filtering: " << dt.count() << " seconds\n";

    // =========================================================================
    // Helper: compile existence heads and register conjectures into grid
    // =========================================================================
    // Maps raw (uncompiled) conjecture string → compiled form (for existence heads)
    std::unordered_map<std::string, std::string> rawToCompiled;

    auto compileAndRegister = [&](const std::vector<std::string>& conjectures) {
        for (std::size_t ti = 0; ti < conjectures.size(); ++ti) {
            const std::string raw = conjectures[ti];
            std::string conj = raw;

            // Precompile ALL structural operators (!(&..) / !(>..)) anywhere
            // in the theorem so addTheoremToMemory's disintegrateImplication
            // produces chain elements and a head that are all already compiled
            // names (or<N> / existence<N>). Substring-scan approach leaves the
            // outer (>...) chain untouched, so the bound-variable lists stay
            // self-consistent even when the inner existence encoding renames
            // its bound variables.
            this->precompileStructuralOperators(conj);

            if (conj != raw) rawToCompiled[raw] = conj;

            this->addTheoremToMemory(conj, this->body, 0, false, this->globalDependencies);
        }
    };

    // =========================================================================
    // Helper: tear down the prover grid (delete children, reset body, clear maps)
    // =========================================================================
    auto destroyGrid = [&]() {
        // 1) Delete all dynamically allocated Memory children
        std::unordered_set<Memory*> seen;
        seen.insert(&this->body);  // never delete root
        std::vector<Memory*> stack;
        for (auto& kv : this->body.simpleMap)
            if (kv.second) stack.push_back(kv.second);

        while (!stack.empty()) {
            Memory* node = stack.back(); stack.pop_back();
            if (!node || !seen.insert(node).second) continue;
            for (auto& kv : node->simpleMap)
                if (kv.second) stack.push_back(kv.second);
            delete node;
        }

        // 2) Reset root body (keep binary-level state: coreExpressionMap, anchorInfo, etc.)
        std::map<std::string, Memory*>().swap(this->body.simpleMap);
        {
            Memory empty;
            using std::swap;
            swap(this->body.startInt, empty.startInt);
            swap(this->body.startIntRepl, empty.startIntRepl);
            swap(this->body.startIntPi, empty.startIntPi);
            swap(this->body.toBeProved, empty.toBeProved);
            swap(this->body.encodedStatements, empty.encodedStatements);
            swap(this->body.statementLevelsMap, empty.statementLevelsMap);
            this->body.parentMemory = nullptr;
            this->body.level = -1;
            swap(this->body.overallHashMemory, empty.overallHashMemory);
            swap(this->body.localHashMemory, empty.localHashMemory);
            swap(this->body.localHashMemoryDelta, empty.localHashMemoryDelta);
            swap(this->body.equivalenceClassesMap, empty.equivalenceClassesMap);
            swap(this->body.localEncodedStatements, empty.localEncodedStatements);
            swap(this->body.localEncodedStatementsDelta, empty.localEncodedStatementsDelta);
            swap(this->body.integrationStartIntMap, empty.integrationStartIntMap);
            swap(this->body.mailIn, empty.mailIn);
            swap(this->body.mailOut, empty.mailOut);
            swap(this->body.wholeExpressions, empty.wholeExpressions);
            swap(this->body.eqClassSttmntIndexMapMap, empty.eqClassSttmntIndexMapMap);
            this->body.isActive = true;
            this->body.isPartOfRecursion = false;
            this->body.deltaNumberStatements = 0;
            swap(this->body.exprOriginMap, empty.exprOriginMap);
            this->body.recursionCounter = 0;
            this->body.contradictionIndex = -1;
            swap(this->body.integrationPrepared, empty.integrationPrepared);
            swap(this->body.integrationPreparedMarker, empty.integrationPreparedMarker);
            swap(this->body.validityNamesToFilter, empty.validityNamesToFilter);
            swap(this->body.canBeSentSet, empty.canBeSentSet);
            swap(this->body.canBeSentMarkerSet, empty.canBeSentMarkerSet);
            swap(this->body.weakVariables, empty.weakVariables);
            swap(this->body.axedVariables, empty.axedVariables);
            this->body.primedForContradiction = false;
            this->body.contradictionTheorem.clear();
            swap(this->body.orAdmissionSet, empty.orAdmissionSet);
            swap(this->body.orBookkeeping, empty.orBookkeeping);
            swap(this->body.orDisjunctCount, empty.orDisjunctCount);
            // int16_t mirrors
            this->body.nameMap = NameMap();
            this->body.keyArena = KeyArena();
            swap(this->body.intEncodedStatements, empty.intEncodedStatements);
            swap(this->body.intLocalEncodedStatements, empty.intLocalEncodedStatements);
            swap(this->body.intLocalEncodedStatementsDelta, empty.intLocalEncodedStatementsDelta);
            swap(this->body.intKnownStatements, empty.intKnownStatements);
            swap(this->body.intValidityNamesToFilter, empty.intValidityNamesToFilter);
            swap(this->body.intAxedVariables, empty.intAxedVariables);
        }

        // 3) Clear grid containers
        destroyParentChildrenMap(index);
        destroyMailboxes(boxes);
        permanentBodies.clear();
        permanentBodies.push_back(&this->body);
    };

    // =========================================================================
    // Helper: build grid from current permanentBodies
    // =========================================================================
    auto buildGrid = [&]() {
        this->prefillIntegrationMapsRecursive(&this->body);
        this->prehandleAnchor(&this->body);
        index = buildParentChildrenMap(permanentBodies);
        boxes = buildPerCoreMailboxes(index);
        this->smashMail(boxes);
    };

    // =========================================================================
    // 2. Initial grid setup
    // =========================================================================
    // Pre-emit head-switched mirrors. Folds the contrapositive of every
    // structurally-qualifying conjecture into the initial pool so the prover
    // gets the same proof opportunities in one pass that the legacy
    // multi-iteration path reached across multiple passes. headSwitchOne
    // calls disintegrateImplication which expects structural operators in
    // compiled form (or<N> / existence<N>), not raw `!(&...)` / `!(>...)`.
    // Conjectures coming out of the CE filter are still raw, so we
    // precompile a working copy per conjecture before the structural scan.
    // The original filteredConjectures stays raw — compileAndRegister
    // precompiles it again (its own per-conjecture pass at the top of the
    // lambda), so the input contract is preserved.
    {
        std::unordered_set<std::string> seen(filteredConjectures.begin(),
                                             filteredConjectures.end());
        std::vector<std::string> mirrors;
        for (const std::string& conj : filteredConjectures) {
            std::string compiled = conj;
            this->precompileStructuralOperators(compiled);
            std::string mirror = headSwitchOne(compiled);
            if (mirror.empty()) continue;
            if (seen.insert(mirror).second) mirrors.push_back(mirror);
        }
        if (!mirrors.empty()) {
            std::cout << "[head-switch pre-emit] adding "
                      << mirrors.size() << " head-switched mirrors to the "
                      << "conjecture pool (was " << filteredConjectures.size()
                      << ")" << std::endl;
            filteredConjectures.insert(filteredConjectures.end(),
                                       mirrors.begin(), mirrors.end());
        }
    }
    compileAndRegister(filteredConjectures);
    buildGrid();

    std::cout << "Prover started." << std::endl;
    t0 = std::chrono::high_resolution_clock::now();

    // =========================================================================
    // SINGLE PASS: prove → compress → head-switch (orPairsFromHeadSwitch only).
    //
    // The head-switched mirrors of every structurally-qualifying conjecture
    // were folded into the conjecture pool already (see the pre-emit pass at
    // the top of analyzeExpressions, just before compileAndRegister). So one
    // prove call is enough — there is nothing the legacy big-iteration second
    // pass would have reached that the first pass does not. The post-prove
    // head-switch block here exists solely to populate `orPairsFromHeadSwitch`
    // for run_modes.cpp's OR-theorem construction. See D-24.
    // =========================================================================

    // ----- PROVE (warm-up + main) -----
    {
        int preIterations = 2;
        if (parameters.maxIterationNumberProof < preIterations)
            preIterations = parameters.maxIterationNumberProof;

        if (preIterations > 0) {
            std::cout << "Phase 1: Running " << preIterations << " warm-up iterations..." << std::endl;
            this->prove(preIterations, permanentBodies, index, boxes);
        }

        int remainingIterations = parameters.maxIterationNumberProof - preIterations;

        // Precompile structural operators in proved theorems before broadcasting.
        // Proved theorems from previous batches may contain !(&...) (OR) or
        // !(>[...]) (existence) that need compilation BEFORE disintegration.
        std::vector<std::string> compiledProved = provedTheorems;
        for (auto& pt : compiledProved) {
            this->precompileStructuralOperators(pt);
        }

        if (!compiledProved.empty()) {
            std::cout << "Injecting " << compiledProved.size() << " proved theorems via broadcast..." << std::endl;
            this->broadcastTheorems(compiledProved);
            this->smashMail(boxes);
        }

        if (!externalTheorems.empty()) {
            // Same precompile treatment as provedTheorems — externals may
            // carry raw !(&...) / !(>...) that would otherwise reach LB
            // memory uncompiled.
            std::vector<std::string> compiledExternals = externalTheorems;
            for (auto& et : compiledExternals) {
                this->precompileStructuralOperators(et);
            }
            std::cout << "Injecting " << compiledExternals.size()
                      << " external theorems via broadcast..." << std::endl;
            this->broadcastTheorems(compiledExternals, "externally provided theorem");
            this->smashMail(boxes);
        }

        if (remainingIterations > 0) {
            std::cout << "Phase 2: Running remaining " << remainingIterations << " iterations..." << std::endl;
            this->prove(remainingIterations, permanentBodies, index, boxes);
        }
    }

    // Compression is invoked from run_modes.cpp post-analyzeExpressions,
    // gated by skipCompression (true under incubator_mode / ban_disintegration).
    // See D-25.

    // ----- HEAD SWITCH (orPairsFromHeadSwitch only) -----
    // The mirrors themselves are already in the conjecture pool (pre-emit
    // pass) and were proved in the single prove call above. The walk here
    // exists solely to populate `orPairsFromHeadSwitch` (class member) so
    // run_modes.cpp:267+ can construct OR theorems from (existence,
    // companion) pairs.
    {
        std::vector<std::tuple<std::string, std::string, std::string, std::string>> snapshot;
        {
            std::lock_guard<std::mutex> lock(theoremListMutex);
            snapshot = globalTheoremList;
        }

        for (const auto& tpl : snapshot) {
            const std::string& theorem = std::get<0>(tpl);
            std::string rebuilt = headSwitchOne(theorem);
            if (rebuilt.empty()) continue;

            std::cout << "Head switch: " << rebuilt << std::endl;
            orPairsFromHeadSwitch.push_back({theorem, rebuilt});
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    dt = std::chrono::duration<double>(t1 - t0);
    std::cout << "Prover finished." << std::endl;
    std::cout << "Runtime prover: " << dt.count() << " seconds" << std::endl;
}











// ============================================================================
// OR theorem construction
// ============================================================================

std::string ExpressionAnalyzer::constructOrTheorem(
    const std::string& existenceThm, const std::string& companionThm) {

    // Disintegrate both theorems
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain1, chain2;
    std::string head1 = ce::disintegrateImplication(existenceThm, chain1, coreExpressionMap);
    std::string head2 = ce::disintegrateImplication(companionThm, chain2, coreExpressionMap);

    // Existence theorem: chain has shared premises + negated premise (!a), head = !b (existence)
    // Companion theorem: chain has shared premises + !b as premise, head = a

    // Find the negated premise in existence theorem (!a)
    int negPremIdx = -1;
    for (std::size_t i = 1; i < chain1.size(); ++i) {
        const std::string& prem = std::get<0>(chain1[i]);
        if (!prem.empty() && prem[0] == '!') {
            negPremIdx = (int)i;
            break;
        }
    }
    if (negPremIdx < 0) return "";

    // !a is chain1[negPremIdx], head1 = b (the existence, also negated)
    // In !a -> b form: !a is the negated premise, b is the head
    // OR = !(&(!a)(!b)) — but !a is already negated, and the head (b) is also !(>[...])
    // So !b = remove the ! from b, giving (>[m](domain)(![body]))
    // And !a is already the premise

    // The OR form: shared_premises -> !(&(!a)(!b))
    // !a = chain1[negPremIdx] (already negated)
    // !b = strip ! from head1, which gives the universal form

    std::string negA = std::get<0>(chain1[negPremIdx]);  // e.g., !(=[n,i0])
    std::string negB;
    if (!head1.empty() && head1[0] == '!') {
        negB = head1.substr(1);  // strip outer !, e.g., (>[m](in[m,N])(![in2[m,n,s]]))
    } else {
        negB = "!" + head1;
    }

    // Build disjuncts (un-negated forms)
    std::string disjA = negA.substr(1);  // strip ! from !(=[7,2]) → (=[7,2])
    std::string disjB;
    if (!negB.empty() && negB[0] == '!') {
        disjB = negB.substr(1);  // strip ! from !(existence2[1,7,3]) → (existence2[1,7,3])
    } else {
        disjB = negB;
    }

    // Collect unique args from both disjuncts (ordered by first appearance)
    std::vector<std::string> orArgs;
    {
        std::set<std::string> seen;
        auto collectArgs = [&](const std::string& expr) {
            for (const auto& a : ce::getArgs(expr)) {
                if (seen.insert(a).second) orArgs.push_back(a);
            }
        };
        collectArgs(disjA);
        collectArgs(disjB);
    }

    // Register OR expression in compiledExpressions
    std::string orCoreName = "or" + std::to_string(orCounter);
    {
        // Check if already registered with same elements
        bool found = false;
        for (const auto& [name, le] : compiledExpressions) {
            if (le.category == "or" && le.elements.size() == 2) {
                // Match elements after substituting u_ args
                found = true;  // for now, assume unique
                break;
            }
        }
    }

    // Build signature: (or0[u_1,u_2,...])
    std::string sigArgs;
    std::map<std::string, std::string> argToU;
    for (std::size_t i = 0; i < orArgs.size(); ++i) {
        std::string u = "u_" + std::to_string(i + 1);
        argToU[orArgs[i]] = u;
        if (i > 0) sigArgs += ",";
        sigArgs += u;
    }
    std::string signature = "(" + orCoreName + "[" + sigArgs + "])";

    // Build elements with u_ substitution
    std::string elemA = ce::replaceKeysInString(disjA, argToU);
    std::string elemB = ce::replaceKeysInString(disjB, argToU);

    LogicalEntity orLe("or", {elemA, elemB}, signature,
                       static_cast<int>(orArgs.size()));
    compiledExpressions.insert(std::make_pair(orCoreName, orLe));

    // Also register in coreExpressionMap for expandSignature/disintegration
    ce::CoreExpressionConfig orCfg;
    orCfg.arity = static_cast<int>(orArgs.size());
    orCfg.signature = signature;
    coreExpressionMap.insert(std::make_pair(orCoreName, orCfg));
    orCounter++;

    // Build compiled OR head: (or0[actual_args])
    std::string compiledOrHead = "(" + orCoreName + "[";
    for (std::size_t i = 0; i < orArgs.size(); ++i) {
        if (i > 0) compiledOrHead += ",";
        compiledOrHead += orArgs[i];
    }
    compiledOrHead += "])";

    std::cout << "OR compiled: !(&" << negA << negB << ") -> " << compiledOrHead << std::endl;

    // Build the shared premises (everything except the negated premise)
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> sharedChain;
    for (std::size_t i = 0; i < chain1.size(); ++i) {
        if ((int)i == negPremIdx) continue;
        sharedChain.push_back(chain1[i]);
    }

    // Reassemble with compiled OR head
    std::string result = compiledOrHead;
    for (std::size_t i = sharedChain.size(); i-- > 0; ) {
        const auto& [expr, args, _] = sharedChain[i];
        std::string joined;
        for (std::size_t j = 0; j < args.size(); ++j) {
            if (j > 0) joined += ',';
            joined += args[j];
        }
        result = "(>[" + joined + "]" + expr + result + ")";
    }

    return result;
}

void ExpressionAnalyzer::checkOrCompletion(const std::string& provedTheorem, int coreId) {
    std::lock_guard<std::mutex> lock(orCandidatesMutex);
    for (auto& oc : orCandidates) {
        if (provedTheorem == oc.existenceTheorem) oc.existenceProved = true;
        if (provedTheorem == oc.companionTheorem) oc.companionProved = true;
    }
}

std::string ExpressionAnalyzer::expandToBaseForm(const std::string& expr) const {
    // Expand batch-specific compiled expressions (and0, or0, implication0, existence0, ...)
    // back to structural form so subsequent batches can parse them.
    // Config-level expressions (in, =, in2, in3, fold, etc.) stay untouched.

    auto isCompiled = [](const std::string& name) -> bool {
        static const std::vector<std::string> prefixes = {"and", "or", "implication", "existence"};
        for (const auto& p : prefixes) {
            if (name.size() > p.size() && name.compare(0, p.size(), p) == 0) {
                bool allDigits = true;
                for (size_t i = p.size(); i < name.size(); ++i) {
                    if (!std::isdigit(static_cast<unsigned char>(name[i]))) { allDigits = false; break; }
                }
                if (allDigits) return true;
            }
        }
        return false;
    };

    std::string result = expr;
    bool changed = true;
    int safety = 0;
    while (changed && safety++ < 20) {
        changed = false;

        for (const auto& [name, le] : compiledExpressions) {
            if (!isCompiled(name)) continue;  // config-level expression — don't expand

            // Find all occurrences of (name[ or !(name[
            std::string pat1 = "(" + name + "[";
            std::string pat2 = "!(" + name + "[";

            for (int pass = 0; pass < 2; ++pass) {
                // Check negated pattern FIRST so !(name[...]) is matched
                // before (name[...]) which would match inside at offset 1.
                const std::string& pat = (pass == 0) ? pat2 : pat1;
                bool negated = (pass == 0);

                std::size_t pos = 0;
                while ((pos = result.find(pat, pos)) != std::string::npos) {
                    // Find matching ]
                    std::size_t bracketStart = pos + pat.size() - 1; // position of [
                    std::size_t bracketEnd = result.find(']', bracketStart);
                    if (bracketEnd == std::string::npos) { pos++; continue; }

                    // Check for closing )
                    std::size_t closePos = bracketEnd + 1;
                    if (closePos >= result.size() || result[closePos] != ')') { pos++; continue; }

                    // Extract args
                    std::string argStr = result.substr(bracketStart + 1, bracketEnd - bracketStart - 1);
                    std::vector<std::string> actualArgs = ce::getArgs("(" + name + "[" + argStr + "])");

                    // Build substitution map: u_1 -> actualArgs[0], etc.
                    std::vector<std::string> sigArgs = ce::getArgs(le.signature);
                    if (sigArgs.size() != actualArgs.size()) { pos++; continue; }
                    std::map<std::string, std::string> subst;
                    for (std::size_t i = 0; i < sigArgs.size(); ++i) {
                        subst[sigArgs[i]] = actualArgs[i];
                    }

                    // Build expanded form based on category
                    std::string expanded;
                    if (le.category == "or" && le.elements.size() == 2) {
                        // Elements are positive disjuncts. OR = !(&!a!b).
                        auto negate = [](const std::string& s) -> std::string {
                            if (!s.empty() && s[0] == '!') return s.substr(1);
                            return "!" + s;
                        };
                        std::string a = ce::replaceKeysInString(le.elements[0], subst);
                        std::string b = ce::replaceKeysInString(le.elements[1], subst);
                        expanded = "!(&" + negate(a) + negate(b) + ")";
                    } else if (le.category == "and") {
                        expanded = "(&";
                        for (const auto& elem : le.elements) {
                            expanded += ce::replaceKeysInString(elem, subst);
                        }
                        expanded += ")";
                    } else if (le.category == "existence" && le.elements.size() == 2) {
                        // Find the highest integer var used in the whole expression
                        // so the new bound var doesn't collide
                        int maxVar = 0;
                        {
                            // Scan all actual args in substitution values
                            for (const auto& kv : subst) {
                                try { int v = std::stoi(kv.second); if (v > maxVar) maxVar = v; }
                                catch (...) {}
                            }
                            // Also scan the current result string for [digits] patterns
                            for (std::size_t si = 0; si < result.size(); ++si) {
                                if (result[si] == '[' || result[si] == ',') {
                                    std::size_t start = si + 1;
                                    std::size_t end = start;
                                    while (end < result.size() && std::isdigit(static_cast<unsigned char>(result[end]))) end++;
                                    if (end > start && end < result.size() && (result[end] == ',' || result[end] == ']')) {
                                        try { int v = std::stoi(result.substr(start, end - start)); if (v > maxVar) maxVar = v; }
                                        catch (...) {}
                                    }
                                }
                            }
                        }
                        std::string freshVar = std::to_string(maxVar + 1);

                        // Find the raw bound var (non-u_ arg not in subst)
                        std::string rawBoundVar;
                        for (const auto& elem : le.elements) {
                            for (const auto& a : ce::getArgs(elem)) {
                                if (subst.find(a) == subst.end()) { rawBoundVar = a; break; }
                            }
                            if (!rawBoundVar.empty()) break;
                        }

                        // Replace raw bound var with fresh var in left/right
                        std::map<std::string, std::string> boundSubst = subst;
                        boundSubst[rawBoundVar] = freshVar;
                        std::string left = ce::replaceKeysInString(le.elements[0], boundSubst);
                        std::string right = ce::replaceKeysInString(le.elements[1], boundSubst);
                        expanded = "!(>[" + freshVar + "]" + left + "!" + right + ")";
                    } else if (le.category == "implication") {
                        // (>[bound](premise)(head)) — elements has chain + head
                        // For now, reconstruct from elements
                        std::string body;
                        for (std::size_t i = le.elements.size(); i-- > 0; ) {
                            std::string elem = ce::replaceKeysInString(le.elements[i], subst);
                            if (body.empty()) {
                                body = elem;
                            } else {
                                body = "(>[]" + elem + body + ")";
                            }
                        }
                        expanded = body;
                    } else {
                        pos++;
                        continue;  // unknown category, skip
                    }

                    // Apply negation if we matched !(name[...])
                    if (negated) {
                        // !(or0[...]) with or0 expanding to !(&!a!b) -> !(!(&!a!b)) = (&!a!b)
                        // General: negate the expanded form
                        if (!expanded.empty() && expanded[0] == '!') {
                            expanded = expanded.substr(1);  // double negation cancels
                        } else {
                            expanded = "!" + expanded;
                        }
                    }

                    // Replace in result
                    std::size_t matchLen = closePos + 1 - pos;
                    result.replace(pos, matchLen, expanded);
                    changed = true;
                    // Don't advance pos — re-scan from same position for nested expansions
                }
            }
        }
    }
    return result;
}

} // namespace gl
