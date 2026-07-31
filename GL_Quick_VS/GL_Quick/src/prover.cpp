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
#include "infra/hashburst_dump.hpp"
#include "infra/rt_tracker.hpp"
#include "memory_infra/global_memory_manager.hpp"
#include "memory_infra/scratch_arena.hpp"
#include "memory_infra/lb_deload.hpp"
#include "memory_infra/deload_stats.hpp"
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
#include <chrono>
#include <cstdio>

namespace gl {

    //Declarations
    static std::map<int64_t, std::vector<std::vector<int>>> generateAllMappings(int n, int m);

    // Assumes n, m > 0.
    // Returns a map keyed by the packed pair (int64_t(i) << 32 | uint32_t(j))
    // with 1 <= i < n and 1 <= j < m to the list of all functions
    // [0..i-1] -> [0..j-1], each function represented as a list<int> of
    // length i whose entries are in [0, j). Construction-built config state,
    // never iterated in-tree (point find only), so no observable order exists.
    std::map<int64_t, std::vector<std::vector<int>>> generateAllMappings(int n, int m) {
        std::map<int64_t, std::vector<std::vector<int>>> functionMap;
        // Order-preserving only for non-negative halves (Rule-19 contract
        // documentation; i and j are positive loop indices here).
        constexpr auto packMappingKey = [](int i, int j) -> int64_t {
            assert(i >= 0 && j >= 0
                && "packMappingKey: packing requires non-negative halves");
            return packInt32Pair(i, j);
        };

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

                functionMap[packMappingKey(i, j)] = functions;
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
/// @invariant [I-1](../../docs/agentic_swdd/30_invariants.md#i-1) — every
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
                if (pp.contains("maxNumberSecondaryVariablesOrint")) parameters.maxNumberSecondaryVariablesOrint = pp["maxNumberSecondaryVariablesOrint"];
                if (pp.contains("sizeAllPermutationsAna")) parameters.sizeAllPermutationsAna = pp["sizeAllPermutationsAna"];
                if (pp.contains("minNumOperatorsKey")) parameters.minNumOperatorsKey = pp["minNumOperatorsKey"];
                if (pp.contains("minNumOperatorsKeyCE")) parameters.minNumOperatorsKeyCE = pp["minNumOperatorsKeyCE"];
                if (pp.contains("maxIterationNumberVariable")) parameters.maxIterationNumberVariable = pp["maxIterationNumberVariable"];
                if (pp.contains("standardMaxSecondaryNumber")) parameters.standardMaxSecondaryNumber = pp["standardMaxSecondaryNumber"];
                if (pp.contains("trackHistory")) parameters.trackHistory = pp["trackHistory"];
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
                if (pp.contains("try_contradiction_negated_head")) parameters.try_contradiction_negated_head = pp["try_contradiction_negated_head"];
                if (pp.contains("skip_ce_filter")) parameters.skip_ce_filter = pp["skip_ce_filter"];
                if (pp.contains("mirror_refutation")) parameters.mirror_refutation = pp["mirror_refutation"];
                if (pp.contains("skip_eq_classes")) parameters.skip_eq_classes = pp["skip_eq_classes"];
                if (pp.contains("incubator_mode")) parameters.incubator_mode = pp["incubator_mode"];
                if (pp.contains("axed_anchor_exception")) parameters.axed_anchor_exception = pp["axed_anchor_exception"];
                if (pp.contains("allow_multiplication")) parameters.allow_multiplication = pp["allow_multiplication"];
                if (pp.contains("max_partition_size")) parameters.max_partition_size = pp["max_partition_size"];
                if (pp.contains("max_or_depth")) parameters.max_or_depth = pp["max_or_depth"];
                if (pp.contains("disable_lb_split")) parameters.disable_lb_split = pp["disable_lb_split"];
                if (pp.contains("lb_split")) parameters.lb_split = pp["lb_split"];
                if (pp.contains("split_growth_factor")) parameters.split_growth_factor = pp["split_growth_factor"];
                if (pp.contains("max_number_splits")) parameters.max_number_splits = pp["max_number_splits"];
                if (pp.contains("fixed_number_splits")) parameters.fixed_number_splits = pp["fixed_number_splits"];
                if (pp.contains("split_fallback_ratio")) parameters.split_fallback_ratio = pp["split_fallback_ratio"];
                if (pp.contains("second_split_submatch_cap")) parameters.second_split_submatch_cap = pp["second_split_submatch_cap"];
                if (pp.contains("min_split_work")) parameters.min_split_work = pp["min_split_work"];
                if (pp.contains("enable_quiesce_skip")) parameters.enable_quiesce_skip = pp["enable_quiesce_skip"];
                if (pp.contains("enable_extent_deload")) parameters.enable_extent_deload = pp["enable_extent_deload"];
                // Memory-size knobs (static pool/block/page, persistent
                // pool/block, hot-arena) are intentionally NOT read from the
                // config: they are fixed in parameters.hpp and identical for
                // every batch (D-139).
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load config from " << configPath << ": " << e.what() << std::endl;
        }
    }

    // Statification sizing triple must satisfy the carving contract (pool a
    // whole multiple of block with block a power of two, and block a whole
    // multiple of page with page a power of two). The values come solely from
    // the parameters.hpp struct defaults — config cannot override them
    // (D-139) — and a bad triple stops the run
    // here, not at the first allocation (I-19).
    assert(isValidStaticMemoryConfig(parameters.static_pool_bytes,
                                     parameters.static_block_bytes)
        && isValidStaticPageConfig(parameters.static_block_bytes,
                                   parameters.static_page_bytes));
    initStaticMemory(StaticMemoryConfig{ parameters.static_pool_bytes,
                                         parameters.static_block_bytes,
                                         parameters.static_page_bytes });

    // The persistent (second) pool shares the page size and must satisfy the
    // same carving contract; its blocks are smaller (the per-LB intToBeProved
    // content is tiny). PoolKind::Persistent so exhaustion names its own knob.
    assert(isValidStaticMemoryConfig(parameters.static_persistent_pool_bytes,
                                     parameters.static_persistent_block_bytes)
        && isValidStaticPageConfig(parameters.static_persistent_block_bytes,
                                   parameters.static_page_bytes));
    initPersistentMemory(StaticMemoryConfig{
        parameters.static_persistent_pool_bytes,
        parameters.static_persistent_block_bytes,
        parameters.static_page_bytes,
        PoolKind::Persistent });

    // The mail (third) pool is stand-alone and never deloaded; it backs the
    // cross-LB pull-model mail log (MailLog on this analyzer). Same carving
    // contract; PoolKind::Mail so exhaustion names its own knob. The
    // ExpressionAnalyzer-owned mailArena draws its blocks from this pool.
    assert(isValidStaticMemoryConfig(parameters.static_mail_pool_bytes,
                                     parameters.static_mail_block_bytes)
        && isValidStaticPageConfig(parameters.static_mail_block_bytes,
                                   parameters.static_page_bytes));
    initMailMemory(StaticMemoryConfig{
        parameters.static_mail_pool_bytes,
        parameters.static_mail_block_bytes,
        parameters.static_page_bytes,
        PoolKind::Mail });

    // The LB-body (fourth) pool is stand-alone and never deloaded; it backs the
    // LB object store (LbStore on this analyzer), which places the Memory node
    // shells off the malloc heap. Same carving contract; PoolKind::Lb so
    // exhaustion names its own knob. The LbStore draws its blocks from this pool.
    assert(isValidStaticMemoryConfig(parameters.static_lb_pool_bytes,
                                     parameters.static_lb_block_bytes)
        && isValidStaticPageConfig(parameters.static_lb_block_bytes,
                                   parameters.static_page_bytes));
    initLbMemory(StaticMemoryConfig{
        parameters.static_lb_pool_bytes,
        parameters.static_lb_block_bytes,
        parameters.static_page_bytes,
        PoolKind::Lb });

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

    // Scratch string arenas: one per worker slot (coreId-indexed), PLUS one
    // reserved slot (index logicalCores) for single-threaded contexts that
    // reach the absorb door's prefixArgumentsWithU off the worker pool (CE-fact
    // load, grid build, compressor, drains; g_currentCoreId == -1) — hence
    // logicalCores + 1. Drawn from the pool on the cold grant path, each
    // released back at its task's exit (no retention). No reserve cap — the
    // pool's own exhaustion assert is the backstop.
    initScratchArenas(logicalCores + 1);

    // Request-generation scratch arenas: a second per-slot set, holding the
    // generators' DFS frontier (byte-bump popTo) and per-batch containers (page
    // tier), isolated from the string-scratch arena above. Released per task.
    initGenScratchArenas(logicalCores);

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
/// **Per [I-17](../../docs/agentic_swdd/30_invariants.md#i-17), the freshness check
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
/// @invariant [I-17](../../docs/agentic_swdd/30_invariants.md#i-17) — single
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
/// [I-1](../../docs/agentic_swdd/30_invariants.md#i-1)) on every theorem-load path
/// before disintegration.
///
/// @param expr Possibly-compact MPL expression text.
/// @return Fully-expanded canonical form.
/// @invariant [I-1](../../docs/agentic_swdd/30_invariants.md#i-1) — expansion must
///            happen on every theorem-load path before disintegration.
std::string ExpressionAnalyzer::expandExpr(const std::string& expr) {
    // 1. Check if expression is simple (e.g., "Macro[arg1, arg2]")
    if (!ce::expressionIsSimple(expr)) {
        return expr;
    }

    // 2. Extract the Head (Macro Name)
    const ce::CoreExpressionConfig* cfgPtr =
        coreConfig(extractExpressionUniversalSpan(StrSpan(expr)));
    if (cfgPtr == nullptr) {
        // Not a known macro, return original
        return expr;
    }
    const ce::CoreExpressionConfig& cfg = *cfgPtr;

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
        if (!current->exprKey().empty()) {
            globalKey.push_back(current->exprKey());
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

/// @brief Span-fill overload of `getGlobalKey` — the 0%-heap production
/// form. See the declaration in `prover.hpp` for the full contract.
int32_t ExpressionAnalyzer::getGlobalKey(const Memory& memoryBlock,
                                         StrSpan* out, int32_t cap) {
    int32_t n = 0;
    const Memory* current = &memoryBlock;
    while (true) {
        const StrSpan k = current->exprKeyView();
        if (k.len != 0) {
            assert(n < cap && "getGlobalKey: global key exceeds caller capacity");
            out[n++] = k;
        }
        if (current->parentMemory != NULL) {
            current = current->parentMemory;
        }
        else {
            break;
        }
    }
    std::reverse(out, out + n);
    return n;
}

// ---------------------------------------------------------------------------
// Incubator: implication multiplication helpers
// ---------------------------------------------------------------------------

// Generate all set partitions (Bell number) of elements as an arena CSR of
// element indices — heap-free (row 252). Cap: if > cap elements, only original
// + all pairwise + all triple equalities. Doxygen lives on the prover.hpp
// declaration.
//
// std::vector<std::string> overload: build a StrSpan run over the (caller-stable)
// strings on the arena byte-bump tier and forward to the StrSpan-run form (the
// real implementation). The spans alias `elements`, stable for the whole call;
// the byte-bump run coexists with the span form's page-tier ping-pong buffers on
// one arena (09b pitfall 2).
void ExpressionAnalyzer::generateSetPartitions(
    const std::vector<std::string>& elements, int cap, ScratchArena& arena,
    PagedVector<int32_t>& outMember, PagedVector<int32_t>& outClassStart,
    PagedVector<int32_t>& outPartStart)
{
    const int32_t n = static_cast<int32_t>(elements.size());
    const StrSpan* spans = nullptr;
    if (n > 0) {
        StrSpan* s = reinterpret_cast<StrSpan*>(arena.resolve(
            arena.alloc(n * static_cast<int32_t>(sizeof(StrSpan)),
                        static_cast<int32_t>(alignof(StrSpan)))));
        for (int32_t i = 0; i < n; ++i)
            s[i] = StrSpan(elements[static_cast<std::size_t>(i)]);
        spans = s;
    }
    generateSetPartitions(spans, n, cap, arena, outMember, outClassStart,
        outPartStart);
}

// StrSpan-run overload — the real implementation.
void ExpressionAnalyzer::generateSetPartitions(
    const StrSpan* elements, int32_t n, int cap, ScratchArena& arena,
    PagedVector<int32_t>& outMember, PagedVector<int32_t>& outClassStart,
    PagedVector<int32_t>& outPartStart)
{
    // Max class size in the recursive branch is n (<= cap) + the peeled index;
    // a loud assert on the stack sort buffer is the Rule-19 tripwire.
    constexpr int32_t kMaxClassBuf = 64;

    // Within-class byte-lex order on element indices == the heap std::sort on
    // the member strings (elements are distinct, so no ties).
    const auto idxLess = [&elements](int32_t a, int32_t b) {
        return compareSpans(elements[a], elements[b]) < 0;
    };

    // n == 0: one empty partition.
    if (n == 0) {
        outClassStart.push_back(0);
        outPartStart.push_back(0);
        outPartStart.push_back(0);   // partition 0 spans classes [0, 0)
        return;
    }

    // n > cap: flat — all-singletons, then every pairwise, then every triple.
    if (n > cap) {
        outClassStart.push_back(0);
        outPartStart.push_back(0);
        const auto closeClass = [&]() {
            outClassStart.push_back(outMember.size());
        };
        const auto closePartition = [&]() {
            outPartStart.push_back(outClassStart.size() - 1);
        };
        for (int32_t k = 0; k < n; ++k) { outMember.push_back(k); closeClass(); }
        closePartition();
        for (int32_t i = 0; i < n; ++i)
            for (int32_t j = i + 1; j < n; ++j) {
                for (int32_t k = 0; k < n; ++k) {
                    if (k == j) continue;
                    if (k == i) {
                        int32_t buf[2] = { i, j };
                        std::sort(buf, buf + 2, idxLess);
                        outMember.push_back(buf[0]);
                        outMember.push_back(buf[1]);
                    } else {
                        outMember.push_back(k);
                    }
                    closeClass();
                }
                closePartition();
            }
        for (int32_t i = 0; i < n; ++i)
            for (int32_t j = i + 1; j < n; ++j)
                for (int32_t l = j + 1; l < n; ++l) {
                    for (int32_t k = 0; k < n; ++k) {
                        if (k == j || k == l) continue;
                        if (k == i) {
                            int32_t buf[3] = { i, j, l };
                            std::sort(buf, buf + 3, idxLess);
                            outMember.push_back(buf[0]);
                            outMember.push_back(buf[1]);
                            outMember.push_back(buf[2]);
                        } else {
                            outMember.push_back(k);
                        }
                        closeClass();
                    }
                    closePartition();
                }
        return;
    }

    // 1 <= n <= cap: iterative Bell enumeration on two ping-pong CSR buffers.
    // The heap recursion peels elements[0] and recurses on the rest, so the
    // outermost peel is applied LAST; expanding indices n-1 .. 0 reproduces the
    // recursion's Option-1-then-Option-2 emission order byte-for-byte.
    DirtyState dA = DirtyState::Clean, dB = DirtyState::Clean;
    PagedVector<int32_t> memA(&arena, &dA), csA(&arena, &dA), psA(&arena, &dA);
    PagedVector<int32_t> memB(&arena, &dB), csB(&arena, &dB), psB(&arena, &dB);
    PagedVector<int32_t>* curMem = &memA; PagedVector<int32_t>* curCs = &csA;
    PagedVector<int32_t>* curPs = &psA;
    PagedVector<int32_t>* nxtMem = &memB; PagedVector<int32_t>* nxtCs = &csB;
    PagedVector<int32_t>* nxtPs = &psB;

    // current = [ [] ] : one empty partition.
    curCs->push_back(0);
    curPs->push_back(0);
    curPs->push_back(0);

    for (int32_t idx = n - 1; idx >= 0; --idx) {
        nxtMem->clear(); nxtCs->clear(); nxtPs->clear();
        nxtCs->push_back(0);
        nxtPs->push_back(0);
        const int32_t P = curPs->size() - 1;

        const auto copyClass = [&](int32_t c) {
            const int32_t mBegin = (*curCs)[c];
            const int32_t mEnd = (*curCs)[c + 1];
            for (int32_t m = mBegin; m < mEnd; ++m)
                nxtMem->push_back((*curMem)[m]);
            nxtCs->push_back(nxtMem->size());
        };

        for (int32_t p = 0; p < P; ++p) {
            const int32_t cBegin = (*curPs)[p];
            const int32_t cEnd = (*curPs)[p + 1];

            // Option 1: copy sp's classes, then a new singleton {idx}.
            for (int32_t c = cBegin; c < cEnd; ++c) copyClass(c);
            nxtMem->push_back(idx);
            nxtCs->push_back(nxtMem->size());
            nxtPs->push_back(nxtCs->size() - 1);

            // Option 2: for each class iSel, copy sp with idx added to iSel,
            // that class re-sorted by string.
            for (int32_t iSel = cBegin; iSel < cEnd; ++iSel) {
                for (int32_t c = cBegin; c < cEnd; ++c) {
                    if (c == iSel) {
                        int32_t buf[kMaxClassBuf];
                        int32_t bn = 0;
                        const int32_t mBegin = (*curCs)[c];
                        const int32_t mEnd = (*curCs)[c + 1];
                        for (int32_t m = mBegin; m < mEnd; ++m) {
                            assert(bn < kMaxClassBuf
                                && "generateSetPartitions: class exceeds "
                                   "kMaxClassBuf (Rule-19 tripwire)");
                            buf[bn++] = (*curMem)[m];
                        }
                        assert(bn < kMaxClassBuf);
                        buf[bn++] = idx;
                        std::sort(buf, buf + bn, idxLess);
                        for (int32_t b = 0; b < bn; ++b)
                            nxtMem->push_back(buf[b]);
                        nxtCs->push_back(nxtMem->size());
                    } else {
                        copyClass(c);
                    }
                }
                nxtPs->push_back(nxtCs->size() - 1);
            }
        }
        std::swap(curMem, nxtMem);
        std::swap(curCs, nxtCs);
        std::swap(curPs, nxtPs);
    }

    // Copy the final current CSR into the caller's output columns.
    for (int32_t i = 0; i < curMem->size(); ++i)
        outMember.push_back((*curMem)[i]);
    for (int32_t i = 0; i < curCs->size(); ++i)
        outClassStart.push_back((*curCs)[i]);
    for (int32_t i = 0; i < curPs->size(); ++i)
        outPartStart.push_back((*curPs)[i]);
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
/// **Per [I-24](../../docs/agentic_swdd/30_invariants.md#i-24)**, multiplyImplication
/// must NOT equate two distinct free `u_*` anchor parameters — they are
/// the pinned anchor slots and cannot be unified by partition renaming.
/// The function asserts on attempts to do so; a firing assert here means
/// the input implication had a malformed anchor binding.
///
/// @param implication Source implication text.
/// @return Vector of all multiplied copies. Always non-empty (at least
///         the input itself). Order is deterministic.
/// @invariant [I-24](../../docs/agentic_swdd/30_invariants.md#i-24) — no
///            free-anchor-param equation.
/// @see `addToHashMemory` — primary consumer (Path 1 head LMV inserts).
// RETAINED HEAP ORACLE for the span-form multiplyImplication (no in-tree caller
// after the span form landed; the Rule-18 twin multiply_implication_span_matches_heap
// calls it). Self-contained: the word-boundary replacement is a local lambda copy
// of the deleted file-static replaceArgInString.
std::vector<std::string> ExpressionAnalyzer::multiplyImplication(
    const std::string& implication)
{
    // Local word-boundary replacement — verbatim copy of the deleted file-static
    // replaceArgInString (won't match v1 inside v10, nor "7" inside "u_7").
    const auto replaceArgHeap = [](const std::string& str,
                                   const std::string& oldArg,
                                   const std::string& newArg) -> std::string {
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
                bool endBoundary =
                    (after >= str.size()) || !isWordChar(str[after]);
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
    };

    // Gated by parameters.allow_multiplication (was parameters.incubator_mode
    // pre-2026-04-29). CE filter still multiplies regardless to keep the
    // filter's reachability fan-out unchanged.
    if (!parameters.allow_multiplication && !ceFilteringActive) return {implication};

    // 1. Parse implication to chain + head (row 238 span twin). Collect all
    //    bound vars (get<1>) into allBoundVars (a set, order-independent) and
    //    all premises (get<0>, skip Anchor) into allExprs in triple order, then
    //    the head — byte-identical to the former two heap loops.
    std::set<std::string> allBoundVars;
    std::vector<std::string> allExprs;
    StrSpan headSpan;
    ce::disintegrateImplicationSpans(StrSpan(implication), headSpan,
        [&allBoundVars, &allExprs](StrSpan keySpan, const StrSpan* bvSpans, int32_t bvN) {
            for (int32_t k = 0; k < bvN; ++k) {
                if (!bvSpans[k].empty()) allBoundVars.insert(bvSpans[k].toStdString());
            }
            std::string expr = keySpan.toStdString();
            if (expr.find("(Anchor") == std::string::npos)
                allExprs.push_back(std::move(expr));
        });
    std::string head = headSpan.toStdString();
    if (head.find("(Anchor") == std::string::npos)
        allExprs.push_back(head);

    // 4. Collect ALL unique variables (bound + u_) appearing in expressions
    std::set<std::string> allVars;
    allVars.insert(allBoundVars.begin(), allBoundVars.end());
    for (const std::string& expr : allExprs) {
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(StrSpan(expr), args,
                                           ExecutionParameters::MAX_ARITY);
        for (int32_t a = 0; a < argsN; ++a) {
            allVars.insert(args[a].toStdString());
        }
    }

    // 5. Determine which vars are (1)-typed (both bound and u_)
    std::vector<std::string> oneTypedVars;
    for (const std::string& var : allVars) {
        bool isOneTyped = false;
        for (const std::string& expr : allExprs) {
            const ce::CoreExpressionConfig* cfg =
                coreConfig(extractExpressionSpan(StrSpan(expr)));
            if (cfg == nullptr) continue;

            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(StrSpan(expr), args,
                                               ExecutionParameters::MAX_ARITY);
            for (int32_t i = 0; i < argsN; ++i) {
                if (equalSpans(args[i], StrSpan(var))) {
                    std::string posKey = std::to_string(i + 1);
                    auto dsIt = cfg->definitionSets.find(posKey);
                    if (dsIt != cfg->definitionSets.end() &&
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

    // 6. Generate set partitions (equivalence relations on oneTypedVars) as an
    //    arena CSR of element indices — heap-free (row 252). Each partition can
    //    compose multiple equalization groups simultaneously.
    const unsigned gspSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gspArena = genScratchArenas().forSlot(gspSlot);
    DirtyState gspDirty = DirtyState::Clean;
    PagedVector<int32_t> partMember(&gspArena, &gspDirty);
    PagedVector<int32_t> partClassStart(&gspArena, &gspDirty);
    PagedVector<int32_t> partPartStart(&gspArena, &gspDirty);
    generateSetPartitions(oneTypedVars, parameters.max_partition_size, gspArena,
        partMember, partClassStart, partPartStart);

    // 7. For each partition, substitute and deduplicate.
    //    u_ variables always win as representative — non-u_ replaced by u_.
    std::set<std::string> seen;
    std::vector<std::string> result;

    const int32_t numParts = partPartStart.size() - 1;
    for (int32_t pIdx = 0; pIdx < numParts; ++pIdx) {
        const int32_t cBegin = partPartStart[pIdx];
        const int32_t cEnd = partPartStart[pIdx + 1];

        // Skip partitions where any equivalence class contains two different u_ vars.
        // Two distinct free anchor parameters must not be equated by partition-based
        // multiplication: doing so silently rewrites a free slot of the rule body
        // (chapter-1115 bug — `u_6` overwritten by `u_2` inside `existence2`).
        bool hasDoubleU = false;
        for (int32_t c = cBegin; c < cEnd && !hasDoubleU; ++c) {
            int uCount = 0;
            for (int32_t m = partClassStart[c]; m < partClassStart[c + 1]; ++m) {
                const std::string& v = oneTypedVars[static_cast<std::size_t>(partMember[m])];
                if (v.size() >= 2 && v[0] == 'u' && v[1] == '_') uCount++;
            }
            if (uCount >= 2) hasDoubleU = true;
        }
        if (hasDoubleU) continue;

        std::string copy = implication;

        for (int32_t c = cBegin; c < cEnd; ++c) {
            const int32_t mBegin = partClassStart[c];
            const int32_t mEnd = partClassStart[c + 1];
            if (mEnd - mBegin <= 1) continue;

            // Pick representative: prefer u_ variable, else first element.
            const std::string* rep =
                &oneTypedVars[static_cast<std::size_t>(partMember[mBegin])];
            for (int32_t m = mBegin; m < mEnd; ++m) {
                const std::string& v = oneTypedVars[static_cast<std::size_t>(partMember[m])];
                if (v.size() >= 2 && v[0] == 'u' && v[1] == '_') { rep = &v; break; }
            }

            // Replace all non-representative vars with the representative.
            for (int32_t m = mBegin; m < mEnd; ++m) {
                const std::string& v = oneTypedVars[static_cast<std::size_t>(partMember[m])];
                if (v == *rep) continue;
                copy = replaceArgHeap(copy, v, *rep);
            }
        }

        {
            // Boundary copy in this still-heap caller: build the deduplicated
            // implication on the string tier, materialize back to `copy` (reused
            // downstream by disintegrateImplicationSpans + the `seen`/`result`
            // std::string containers). The span never outlives dbvScope.
            const unsigned dbvSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : scratchArenas().slotCount() - 1;
            ScratchArena& dbvArena = scratchArenas().forSlot(dbvSlot);
            ScratchScope dbvScope(dbvArena);
            copy = deduplicateBoundVarsScratch(StrSpan(copy), dbvArena).toStdString();
        }

        // Skip copies whose head is a trivial equality (=[x,x]).
        {
            StrSpan copyHeadSpan;
            ce::disintegrateImplicationSpans(StrSpan(copy), copyHeadSpan,
                [](StrSpan, const StrSpan*, int32_t) {});   // head only
            std::string copyHead = copyHeadSpan.toStdString();
            if (equalSpans(extractExpressionSpan(StrSpan(copyHead)), StrSpan("=", 1))) {
                StrSpan eqArgs[ExecutionParameters::MAX_ARITY];
                const int32_t eqArgsN = getArgsSpans(StrSpan(copyHead), eqArgs,
                                                     ExecutionParameters::MAX_ARITY);
                if (eqArgsN == 2 && equalSpans(eqArgs[0], eqArgs[1])) continue;
            }
        }

        if (seen.find(copy) == seen.end()) {
            seen.insert(copy);
            result.push_back(copy);
        }
    }

    return result;
}

// Span form of multiplyImplication — the production path (the vector<string>
// form above is the retained heap oracle). Emits each surviving multiplied copy
// as a CopyRef into `out`; copy bytes ride outStrArena (string tier). Byte-for-
// byte identical emitted copy SEQUENCE to the heap oracle (Doxygen on the
// prover.hpp declaration).
void ExpressionAnalyzer::multiplyImplication(StrSpan implication,
    ScratchArena& outStrArena, PagedVector<CopyRef>& out)
{
    // Emit one copy = a verbatim slice of `s` onto outStrArena; record a CopyRef.
    const auto emitCopy = [&](StrSpan s) {
        assert(s.len > 0 && "multiplyImplication: empty copy");
        const ArenaOffset off = outStrArena.alloc(s.len, 1);
        std::memcpy(outStrArena.resolve(off), s.ptr,
                    static_cast<std::size_t>(s.len));
        out.push_back(CopyRef{ off, s.len });
    };

    if (!parameters.allow_multiplication && !ceFilteringActive) {
        emitCopy(implication);
        return;
    }

    // Per-slot scratch arenas: genArena hosts the page-tier dedup/index/CSR
    // containers + the byte-bump sortedVars run; outStrArena (string tier) holds
    // the emitted copies + the per-partition transient build (09b pitfall 2 /
    // I-124). A var/expr reference is an (offset,len) slice into the
    // caller-stable `implication` (row 62) — no bytes copied.
    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& genArena = genScratchArenas().forSlot(slot);
    DirtyState d = DirtyState::Clean;

    struct VarRef { int32_t off; int32_t len; };
    const auto refSpan = [&](VarRef r) -> StrSpan {
        return StrSpan(implication.ptr + r.off, r.len);
    };
    const auto makeRef = [&](StrSpan s) -> VarRef {
        return VarRef{ static_cast<int32_t>(s.ptr - implication.ptr), s.len };
    };

    // 1. Parse to chain + head; collect unique vars (allVars) via a dedup
    //    ColdHashSet (order-independent — the numericLessSpan sort below washes
    //    out the mint order), and every non-Anchor expr (triple keys + head).
    ColdHashSet<BytesKeyStore> allVarsSet(&genArena, &d);
    PagedVector<VarRef> allVars(&genArena, &d);
    PagedVector<VarRef> allExprs(&genArena, &d);
    const auto addVar = [&](StrSpan v) {
        if (v.empty()) return;
        const int32_t before = allVarsSet.count();
        allVarsSet.mint(v);
        if (allVarsSet.count() > before) allVars.push_back(makeRef(v));
    };
    StrSpan headSpan;
    ce::disintegrateImplicationSpans(implication, headSpan,
        [&](StrSpan keySpan, const StrSpan* bvSpans, int32_t bvN) {
            for (int32_t k = 0; k < bvN; ++k) addVar(bvSpans[k]);
            if (!containsSpan(keySpan, StrSpan("(Anchor", 7)))
                allExprs.push_back(makeRef(keySpan));
        });
    if (!containsSpan(headSpan, StrSpan("(Anchor", 7)))
        allExprs.push_back(makeRef(headSpan));

    for (int32_t e = 0; e < allExprs.size(); ++e) {
        const StrSpan exprSpan = refSpan(allExprs[e]);
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(exprSpan, args,
                                           ExecutionParameters::MAX_ARITY);
        for (int32_t a = 0; a < argsN; ++a) addVar(args[a]);
    }

    // 2. Which vars are (1)-typed (appear at a (1)-typed slot of some expr). The
    //    position key is the decimal of (i+1); definitionSets is a non-transparent
    //    std::map, so scan it comparing keys as spans (heap-free, byte-identical
    //    to the former std::to_string(i+1) + find).
    PagedVector<VarRef> oneTypedVars(&genArena, &d);
    for (int32_t vi = 0; vi < allVars.size(); ++vi) {
        const StrSpan var = refSpan(allVars[vi]);
        bool isOneTyped = false;
        for (int32_t e = 0; e < allExprs.size() && !isOneTyped; ++e) {
            const StrSpan exprSpan = refSpan(allExprs[e]);
            const ce::CoreExpressionConfig* cfg =
                coreConfig(extractExpressionSpan(exprSpan));
            if (cfg == nullptr) continue;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(exprSpan, args,
                                               ExecutionParameters::MAX_ARITY);
            for (int32_t i = 0; i < argsN; ++i) {
                if (!equalSpans(args[i], var)) continue;
                char posBuf[12];
                const int32_t posN = writeDecimalDigits(posBuf, i + 1);
                const StrSpan posSpan(posBuf, posN);
                for (const auto& kv : cfg->definitionSets) {
                    if (equalSpans(StrSpan(kv.first), posSpan)) {
                        if (kv.second.first == "(1)") isOneTyped = true;
                        break;
                    }
                }
                if (isOneTyped) break;
            }
        }
        if (isOneTyped) oneTypedVars.push_back(allVars[vi]);
    }

    const int32_t oneN = oneTypedVars.size();
    if (oneN <= 1) { emitCopy(implication); return; }

    // 3. Sort oneTypedVars by numericLessSpan (the numericLess twin) into a
    //    contiguous StrSpan run for the generateSetPartitions StrSpan overload.
    int32_t* order = reinterpret_cast<int32_t*>(genArena.resolve(genArena.alloc(
        oneN * static_cast<int32_t>(sizeof(int32_t)),
        static_cast<int32_t>(alignof(int32_t)))));
    for (int32_t i = 0; i < oneN; ++i) order[i] = i;
    std::sort(order, order + oneN, [&](int32_t a, int32_t b) {
        return numericLessSpan(refSpan(oneTypedVars[a]),
                               refSpan(oneTypedVars[b]));
    });
    StrSpan* sortedVars = reinterpret_cast<StrSpan*>(genArena.resolve(genArena.alloc(
        oneN * static_cast<int32_t>(sizeof(StrSpan)),
        static_cast<int32_t>(alignof(StrSpan)))));
    for (int32_t i = 0; i < oneN; ++i)
        sortedVars[i] = refSpan(oneTypedVars[order[i]]);

    // 4. Set partitions (Bell) of sortedVars as an arena CSR of element indices.
    PagedVector<int32_t> partMember(&genArena, &d);
    PagedVector<int32_t> partClassStart(&genArena, &d);
    PagedVector<int32_t> partPartStart(&genArena, &d);
    generateSetPartitions(sortedVars, oneN, parameters.max_partition_size,
        genArena, partMember, partClassStart, partPartStart);

    // 5. For each partition: substitute non-representatives, dedup bound vars,
    //    skip trivial-equality heads, content-dedup, emit survivors. Fresh copies
    //    ride outStrArena (string tier); the `seen` set rides genArena (page
    //    tier, a different arena — I-3-safe). Dropped partitions reclaim their
    //    per-partition scratch via popTo.
    ColdHashSet<BytesKeyStore> seen(&genArena, &d);
    const int32_t numParts = partPartStart.size() - 1;
    for (int32_t pIdx = 0; pIdx < numParts; ++pIdx) {
        const int32_t cBegin = partPartStart[pIdx];
        const int32_t cEnd = partPartStart[pIdx + 1];

        // Skip partitions equating two distinct free u_ anchor params (I-24).
        bool hasDoubleU = false;
        for (int32_t c = cBegin; c < cEnd && !hasDoubleU; ++c) {
            int uCount = 0;
            for (int32_t m = partClassStart[c]; m < partClassStart[c + 1]; ++m) {
                const StrSpan v = sortedVars[partMember[m]];
                if (v.len >= 2 && v.ptr[0] == 'u' && v.ptr[1] == '_') uCount++;
            }
            if (uCount >= 2) hasDoubleU = true;
        }
        if (hasDoubleU) continue;

        const ArenaOffset partMark = outStrArena.cursor();
        ScratchString cur = copyScratch(outStrArena, implication);

        for (int32_t c = cBegin; c < cEnd; ++c) {
            const int32_t mBegin = partClassStart[c];
            const int32_t mEnd = partClassStart[c + 1];
            if (mEnd - mBegin <= 1) continue;

            // Representative: prefer a u_ member, else the first element.
            StrSpan rep = sortedVars[partMember[mBegin]];
            for (int32_t m = mBegin; m < mEnd; ++m) {
                const StrSpan v = sortedVars[partMember[m]];
                if (v.len >= 2 && v.ptr[0] == 'u' && v.ptr[1] == '_') {
                    rep = v; break;
                }
            }
            for (int32_t m = mBegin; m < mEnd; ++m) {
                const StrSpan v = sortedVars[partMember[m]];
                if (equalSpans(v, rep)) continue;
                cur = replaceArgSpanScratch(StrSpan(cur), v, rep, outStrArena);
            }
        }

        cur = deduplicateBoundVarsScratch(StrSpan(cur), outStrArena);

        // Skip copies whose head is a trivial equality (=[x,x]).
        {
            StrSpan curHeadSpan;
            ce::disintegrateImplicationSpans(StrSpan(cur), curHeadSpan,
                [](StrSpan, const StrSpan*, int32_t) {});   // head only
            if (equalSpans(extractExpressionSpan(curHeadSpan), StrSpan("=", 1))) {
                StrSpan eqArgs[ExecutionParameters::MAX_ARITY];
                const int32_t eqArgsN = getArgsSpans(curHeadSpan, eqArgs,
                    ExecutionParameters::MAX_ARITY);
                if (eqArgsN == 2 && equalSpans(eqArgs[0], eqArgs[1])) {
                    outStrArena.popTo(partMark);
                    continue;
                }
            }
        }

        // Content-dedup via the seen ColdHashSet: fresh iff the count grew.
        const int32_t before = seen.count();
        seen.mint(StrSpan(cur));
        if (seen.count() > before) {
            emitCopy(StrSpan(cur));         // survivor persists on outStrArena
        } else {
            outStrArena.popTo(partMark);    // duplicate: reclaim partition scratch
        }
    }
}


// Turn optimizations OFF for just this section
//#pragma optimize("", off)

// The hash-engine bodies live in memory.cpp: addToHashMemory(),
// makeNormalizedKeysForAdmission(), lessByName(), lessByOriginal(),
// filterIntEncodedStatements(), the two obligatory-stump builders,
// generateEncodedRequestsStatic(), and checkLocalEncodedMemoryStatic().

// ========================================================================
// Static pipeline: IntEncodedExpr-based, zero-alloc request generation
// ========================================================================

// makeMandatoryEncodedStatementLists1Static() and …2Static() — moved to memory.cpp.

//#pragma optimize("", off)
/// @brief Install the dead-end variable-copy equalities for every
/// hypo-scoped validity — the once-per-burst reaction to hypothetical
/// disintegration.
///
/// @details
/// `disintegrateExprHypothetically` opens validity scopes named
/// `parent + "_boundary_" + "_var0_<x>[_var1_<y>...]" + "_hypo_" + expr`,
/// one per hypothetically disintegrated integration goal. Once per burst
/// (phase 3, the `REACT_TO_HYPO` scope — one worker owns the LB), this
/// function walks every equivalence-class scope of the LB and, on each
/// hypo-scoped row, checks every class member that is one of the payload
/// variables: when ALL of its class-mates are `int_lev_*` names and at
/// least one class-mate exists, the dead-end equality `(=[var,var_copy])`
/// is installed at the PARENT scope via `addExprToMemoryBlock` with the
/// `variableCopy` origin (the "variable copy" proof tag).
///
/// The interior is statified
/// (I-142; 09b/09c cookbooks):
/// - Row order: a validity-id array on the per-slot gen-scratch byte-bump
///   tier, sorted by `compareSpans(decodeView(a), decodeView(b))` — the
///   decoded-name lex order of the retired string-pair snapshot, never id
///   order (I-84). Tie-free (distinct vids decode to distinct names) and
///   mint-free for the whole sort window (I-3).
/// - Row names: the cheap `_hypo_` gate runs on the raw `decodeView`
///   span; a passing row is copied to a per-row `ScratchString` on the
///   string-scratch tier so the `parseHypoScopeVars` spans survive the
///   class loop's NameMap mints.
/// - Per-row class read: an arena MEMBER-RUN SNAPSHOT (`memberPool` /
///   `classStarts`, gen-scratch page tier, cleared per row) taken BEFORE
///   the class loop. The loop's own `addExprToMemoryBlock` can splice
///   this same map's shared blob pool through a DIFFERENT key (the
///   parent-scope equality), so live blob views would both see mid-row
///   mutations the retired heap decode never saw and dangle; the snapshot
///   reproduces the heap `decodeClassesById` isolation exactly, and is
///   content-complete because the loop reads only member ids. Bucket
///   content is read at row start — the decode-at-the-point-of-processing
///   semantics: an earlier row's mutations to a later row's bucket ARE
///   seen, exactly as before.
/// - Firing product: exact-length `ScratchString` build of
///   `(=[var,var_copy])` plus the byte-exact `removeUPrefixScratch` twin;
///   the add door receives byte-identical spans, levels, origin, and
///   flags in the identical call order.
///
/// The door levels ride a hoisted one-element stack run (`{ mb.level }`):
/// `addExprToMemoryBlock` takes an ascending-unique `(const int*, int32_t)`
/// run, caller-owned per the levels-run chain contract. `mb.level` is
/// written only at LB creation, so one run per call equals the retired
/// one set per firing.
///
/// @param mb The LB whose equivalence-class scopes are surveyed; the add
///           door mints into `mb.nameMap` (phase-3 single-owner access).
/// @see parseHypoScopeVars, HypoScopeParse — the name parse;
///      `disintegrateExprHypothetically` — the hypo-scope producer;
///      `EqClassNameCaches::kindOf` — the class-mate tier probe.
void ExpressionAnalyzer::reactToHypo(Memory& mb) {
    const StrSpan kHypo("_hypo_", 6);

    // Per-slot scratch arenas (the pendingWipeScopes drain idiom below):
    // gen-scratch hosts the vid array + snapshot pools + blob straddle
    // peeks; the string arena hosts the per-row name copy + the firing
    // product builds. Never mixed (I-124).
    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    ScratchArena& strArena = scratchArenas().forSlot(
        (g_currentCoreId >= 0) ? static_cast<unsigned>(g_currentCoreId)
                               : scratchArenas().slotCount() - 1);
    const ArenaOffset gMark = gArena.cursor();

    // Iterate all equivalence-class scopes via a decoded-name lex-sorted
    // vid snapshot — identical iteration order to the former string-keyed
    // map, never id order (I-84). The sort window is mint-free, so the
    // comparator's decodeView spans stay valid throughout (I-3).
    const int32_t n = mb.equivalenceClassesMap.count();
    if (n == 0) return;
    NameId* vids = reinterpret_cast<NameId*>(gArena.resolve(
        gArena.alloc(n * static_cast<int32_t>(sizeof(NameId)),
                     alignof(NameId))));
    for (int32_t kid = 1; kid <= n; ++kid)
        vids[kid - 1] = mb.equivalenceClassesMap.keyAt(kid);
    std::sort(vids, vids + n, [&](NameId a, NameId b) {
        return compareSpans(mb.nameMap.decodeView(a),
                            mb.nameMap.decodeView(b)) < 0;
    });

    // Member-run snapshot pools — declared once, cleared per row (the
    // cleanUpExpressions idiom). Page tier; freed by the destructors.
    DirtyState poolDirty = DirtyState::Clean;
    PagedVector<NameId> memberPool(&gArena, &poolDirty);
    PagedVector<int32_t> classStarts(&gArena, &poolDirty);

    // Hoisted door levels run — one per call; mb.level is written only at
    // LB creation, so every retired per-firing set held this same value.
    const int lvRun[1] = { mb.level };

    for (int32_t r = 0; r < n; ++r) {
        const NameId classesValId = vids[r];

        // Cheap pre-filter on the raw span (gate 1; pure read). The full
        // parse below re-runs it on the owned copy — byte-identical
        // content, so the double check is unobservable.
        const StrSpan rawName = mb.nameMap.decodeView(classesValId);
        if (rfindSpanBefore(rawName, kHypo, rawName.len) == -1) continue;

        // Own the row name on the string tier: the parse spans (parent +
        // vars) alias it and must survive the class loop's NameMap mints.
        ScratchScope rowScope(strArena);
        const ScratchString rowName =
            ScratchString::copyFrom(strArena, rawName.ptr, rawName.len);
        HypoScopeParse parse;
        if (!parseHypoScopeVars(StrSpan(rowName), parse)) continue;

        // Member-run snapshot for THIS row, read at row start (current
        // content — see the Doxygen). The class loop reads ONLY the pools;
        // its addExprToMemoryBlock may splice the class map's blob pool.
        memberPool.clear();
        classStarts.clear();
        const int32_t bucketId = mb.equivalenceClassesMap.lookup(classesValId);
        const int32_t classCount =
            (bucketId == 0) ? 0 : mb.equivalenceClassesMap.runLen(bucketId);
        for (int32_t j = 0; j < classCount; ++j) {
            int32_t blen = 0;
            const char* bp = mb.equivalenceClassesMap.peekRecordBytes(
                bucketId, j, blen, gArena);
            const EquivalenceClassView cls{ bp, blen };
            classStarts.push_back(static_cast<int32_t>(memberPool.size()));
            const int32_t memberCount = cls.memberCount();
            for (int32_t k = 0; k < memberCount; ++k)
                memberPool.push_back(cls.memberId(k));
        }

        // Check the equivalence classes (pool reads only).
        for (int32_t j = 0; j < classCount; ++j) {
            const int32_t mStart = classStarts[j];
            const int32_t mEnd = (j + 1 < classCount)
                ? classStarts[j + 1]
                : static_cast<int32_t>(memberPool.size());
            for (int32_t mi = mStart; mi < mEnd; ++mi) {
                const NameId varId = memberPool[mi];
                // decodeView is safe here: the span's last read (the
                // equality fill below) completes before the add door
                // mints, and kindOf writes only kindById_ — a different
                // container (I-3); the next iteration re-decodes fresh.
                const StrSpan var = mb.nameMap.decodeView(varId);

                // Membership in the payload variables — linear equalSpans
                // over the insertion-order parse array (set order was
                // unobservable: membership + empty only).
                bool isTarget = false;
                for (int32_t vi = 0; vi < parse.varCount; ++vi) {
                    if (equalSpans(parse.vars[vi], var)) {
                        isTarget = true;
                        break;
                    }
                }
                if (!isTarget) continue;

                // Check "other" variables constraint (id compares, as the
                // retired memberIds loop did).
                bool conditionMet = true;
                bool hasOthers = false;
                for (int32_t oi = mStart; oi < mEnd; ++oi) {
                    const NameId otherId = memberPool[oi];
                    if (otherId == varId) continue;

                    hasOthers = true;
                    // Other variable must be an int_lev_* name.
                    if (mb.eqClassNameCaches.kindOf(otherId, mb.nameMap)
                        != NameKind::IntLev) {
                        conditionMet = false;
                        break;
                    }
                }

                // If yes, add (=[var, var_copy])
                if (conditionMet && hasOthers) {
                    // Exact-length fill of "(=[" var "," var "_copy])" on
                    // the string tier (09c Section 4 doctrine).
                    const int32_t eqLen = 2 * var.len + 11;
                    char* buf = strArena.allocBytes(eqLen);
                    int32_t at = 0;
                    std::memcpy(buf + at, "(=[", 3); at += 3;
                    std::memcpy(buf + at, var.ptr,
                                static_cast<size_t>(var.len));
                    at += var.len;
                    buf[at++] = ',';
                    std::memcpy(buf + at, var.ptr,
                                static_cast<size_t>(var.len));
                    at += var.len;
                    std::memcpy(buf + at, "_copy", 5); at += 5;
                    buf[at++] = ']';
                    buf[at++] = ')';
                    assert(at == eqLen);
                    const ScratchString equalityExpr =
                        ScratchString::wrap(strArena, buf, eqLen);

                    const ScratchString finalExpr =
                        removeUPrefixScratch(strArena, StrSpan(equalityExpr));

                    // Dead-end axiom — see "variable copy" tag
                    const TransientOrigin origin{
                        true, OriginTag::variableCopy, nullptr, 0 };

                    addExprToMemoryBlock(StrSpan(finalExpr),
                        mb,
                        -1, // iteration
                        0,  // status (Local statement)
                        lvRun, 1,
                        origin,
                        -1, // coreId
                        -1, // auxyIndex
                        parse.parentValidity, // the extracted parent scope
                        false);
                }
            }
        }
    }
    gArena.popTo(gMark);   // vid array + straddle peeks; pools are page-tier
}


// =================== Main driver ===================

// Per-split-part submatch counter — the hashburst work metric. See prover.hpp.
thread_local int64_t ExpressionAnalyzer::g_growthMatchCount = 0;
// I-28 detect-and-defer trial: set true only inside a parallel phase-1/phase-3
// worker task (the runPhase wrappers); gates updateAdmissionMap3's ancestor write.
thread_local bool ExpressionAnalyzer::g_inParallelWorkerPhase = false;
// Absorb-door scratch accessor: the phase-1/phase-3 workers publish their
// coreId here on entry; prefixArgumentsWithU reads it to pick its per-slot
// scratch arena. -1 (the default, never overwritten on single-threaded setup
// threads) maps to the reserved scratch slot.
thread_local int ExpressionAnalyzer::g_currentCoreId = -1;

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

/// @see Declaration in `prover.hpp` for the full contract.
void ExpressionAnalyzer::performElem2(const Memory& body, unsigned coreId,
    int processID, int splitCount, int partCount,
    const SplitStumpRef& splitStump,
    SealedPageSet& sealedPages,
    std::atomic<bool>& burstShouldStop) {

    // The LB split's expression dimension. A stump-split sub-part carries a bucket
    // of stumps, and every request it generates contains one of them; an ordinary
    // whole-LB part carries none.
    assert((splitStump.count == 0) == (splitStump.stumps == nullptr)
        && "performElem2: stump bucket present iff non-empty");
    // A stump bucket only ever rides a MULTI-PART LB: the whole-LB expression split
    // runs partCount bucket parts (at splitCount == 1). The burst early-exit (I-76)
    // is disabled whenever partCount > 1, because a sibling bailing on another
    // part's stop is a scheduling race; a bucket part is exactly such a sibling, and
    // this assert is what makes the g_isMultiPart gate cover it.
    assert((splitStump.count == 0 || partCount > 1)
        && "performElem2: a stump bucket belongs to a MULTI-PART LB — the "
           "early-exit gate reads g_isMultiPart and would re-enable at 1");
    assert((splitStump.count == 0 || splitStump.total >= 1)
        && "performElem2: a stump sub-part has a place among its siblings");
    assert(partCount >= 1 && "performElem2: partCount is the LB's part count (>= 1)");

    // Per-call RT tracker home. performElem2 is the hashburst (request
    // generation + the inline fixpoint check) — where a runaway LB spends its
    // time — and its REQGEN_BATCH* / FIXPOINT_LOOP RT_SCOPE_HERE markers already
    // record into the thread-local tracker this declares. Stripped to nothing
    // when RT_MEASUREMENT == 0. RT requires disable_lb_split (one part per LB) so
    // this call measures the LB's whole hashburst on one thread, writing one
    // .rt/<chain>.log per LB with no cross-part races. See
    // D-110, I-59.
    RT_TRACKER_DECL(body);

    // Per-executor LB-split context (thread-local; read by partitionAccepts so
    // this executor generates only its id % splitCount == processID slice).
    g_splitProcessID = processID;
    g_splitCount = splitCount;
    // Per-burst multi-part signal for the early-exit gate (I-76), independent of
    // the rule dimension: a whole-LB expression split runs partCount bucket parts
    // at splitCount == 1, and each must run to completion (D-121).
    g_isMultiPart = (partCount > 1);

    // The request keys + the IntEncodedExpr copies now ride this slot's gen
    // scratch arena (genScratchArenas().forSlot(coreId)), released per worker
    // task at this function's single exit — so each executor starts with a
    // fresh arena and no per-entry reset is needed. Keys stay private to this
    // executor's slot (one executor per slot at a time), so parallel LB-split
    // parts never write a shared key arena (D-125).

    // This slot's scratch string arena is filled by the firing-check path
    // below (transient strings under per-call rewind windows; FiringRecord
    // fields stay owned copies / sealed pages, so nothing points into the
    // arena across the call) and RELEASED back to the pool at this executor's
    // exit (see the end of this function) — no blocks retained after the
    // worker finishes.

    // Submatch counter reset: zero the per-split-part tally at the start of this
    // part. The generator's grow-DFS and the merge's preEvaluateFromEncoded bump it
    // on every match this part owns; canAccept caps the burst on it, and the worker
    // reads it after this call to drive the split policy. See D-109.
    g_growthMatchCount = 0;

    // Streaming consumer: each generated request is checked inline (dependency
    // skip + checkLocalEncodedMemoryStatic) instead of being buffered and
    // checked afterwards. The external `burstShouldStop` flag — per-LB, shared
    // by all the LB's parts and living OUTSIDE the LB — lets a SINGLE-part LB end
    // its burst early without writing anything on the LB (I-66); a multi-part LB
    // runs every part to completion (I-76 / g_isMultiPart).
    //
    // No submatch cap: main-path bursts run to COMPLETION and a straggler is split
    // preemptively next iteration (the stats-driven trigger in proveKernel), never
    // truncated mid-burst. The CE filter was already uncapped (ceFilteringActive).
    BurstSink sink{ this, &body, coreId, &sealedPages, &burstShouldStop,
                    SealedRecordCursor<FiringRecord>(sealedPages) };

    // ===================================================================
    // Request generation. performElemPhase1's absorb filled
    // body.workingMemory / body.intExternalStatements from THIS burst's
    // freshly-arrived mailIn.statements (status=3 disintegration), so the
    // same-burst
    // request blocks below see all newly-arrived rules and facts.
    // ===================================================================

    if (ceFilteringActive) {
        RT_SCOPE_HERE("REQGEN_CE_MODE");
        // --- CE mode: no element is obligatory, so the stump is empty and every
        // base candidate is itself a request.
        this->generateEncodedRequestsStatic(body, body.overallHashMemory,
            /*stumpLen=*/0, /*stumps=*/nullptr, /*stumpCount=*/0,
            IntStmtView(), IntStmtView(), SplitStumpRef{}, coreId, sink);
    } else {
        // --- Normal mode: 5-batch obligatory-stump pipeline ---
        // Batch 1 reads the persistent `body.workingMemory` and the
        // mail-pair batches read `body.intExternalStatements`, both
        // filled by the absorb above.

        Stump stumpBuf[4096];

        // --- Batch 1: mail-recovered rules (body.workingMemory) ---
        // Filled by the absorb above (status=3 recovered implications). The
        // leading sink.canAccept() skips this batch's stump build + generate
        // once the LB hit the cap or was early-exited by another part.
        if (sink.canAccept() && !body.workingMemory.encodedMap.empty()) {
            RT_SCOPE_HERE("REQGEN_BATCH1_WORKING_MEMORY");
            NameId nMsl1 = this->makeMandatoryEncodedStatementLists1Static(
                body.workingMemory, body.nameMap,
                IntStmtView(body.intLocalEncodedStatements),
                stumpBuf, 4096);
            if (nMsl1 > 0) {
                this->generateEncodedRequestsStatic(body, body.workingMemory,
                    /*stumpLen=*/1, stumpBuf, nMsl1,
                    IntStmtView(body.intLocalEncodedStatements), IntStmtView(),
                    splitStump, coreId, sink);
            }
        }

        // --- Batch 2: new local delta ---
        if (sink.canAccept()) {
            RT_SCOPE_HERE("REQGEN_BATCH2_LOCAL_DELTA");
            NameId nMsl2 = this->makeMandatoryEncodedStatementLists1Static(
                body.overallHashMemory, body.nameMap,
                IntStmtView(body.intLocalEncodedStatementsDelta),
                stumpBuf, 4096);

            if (nMsl2 > 0) {
                this->generateEncodedRequestsStatic(body, body.overallHashMemory,
                    /*stumpLen=*/1, stumpBuf, nMsl2,
                    IntStmtView(body.intLocalEncodedStatementsDelta), IntStmtView(),
                    splitStump, coreId, sink);
            }
        }

        // --- Batch 3: local × mail pairs (two-element stumps) ---
        // Mail side = body.intExternalStatements (filled by the absorb above).
        if (sink.canAccept()) {
            RT_SCOPE_HERE("REQGEN_BATCH3_LOCAL_X_MAIL");
            Stump pairsBuf[8192];
            NameId nPairs = this->makeMandatoryEncodedStatementLists2Static(
                body, body.overallHashMemory,
                IntStmtView(body.intLocalEncodedStatements),
                IntStmtView(body.intExternalStatements),
                pairsBuf, 8192);
            if (nPairs > 0) {
                this->generateEncodedRequestsStatic(body, body.overallHashMemory,
                    /*stumpLen=*/2, pairsBuf, nPairs,
                    IntStmtView(body.intLocalEncodedStatements),
                    IntStmtView(body.intExternalStatements),
                    splitStump, coreId, sink);
            }
        }

        // --- Batch 4: localHashMemory — mail one-element stumps ---
        // Mail side = body.intExternalStatements (filled by the absorb above).
        if (sink.canAccept() && !body.localHashMemory.encodedMap.empty() && !body.intExternalStatements.empty()) {
            RT_SCOPE_HERE("REQGEN_BATCH4_LOCAL_X_MAIL_SINGLES");
            NameId nMsl4 = this->makeMandatoryEncodedStatementLists1Static(
                body.localHashMemory, body.nameMap,
                IntStmtView(body.intExternalStatements),
                stumpBuf, 4096);
            if (nMsl4 > 0) {
                this->generateEncodedRequestsStatic(body, body.localHashMemory,
                    /*stumpLen=*/1, stumpBuf, nMsl4,
                    IntStmtView(body.intExternalStatements), IntStmtView(),
                    splitStump, coreId, sink);
            }
        }

        // --- Batch 5: localHashMemoryDelta ---
        if (sink.canAccept() && !body.localHashMemoryDelta.encodedMap.empty()) {
            RT_SCOPE_HERE("REQGEN_BATCH5_LOCAL_HASH_DELTA");
            NameId nMsl5 = this->makeMandatoryEncodedStatementLists1Static(
                body.localHashMemoryDelta, body.nameMap,
                IntStmtView(body.intEncodedStatements),
                stumpBuf, 4096);
            if (nMsl5 > 0) {
                this->generateEncodedRequestsStatic(body, body.localHashMemoryDelta,
                    /*stumpLen=*/1, stumpBuf, nMsl5,
                    IntStmtView(body.intEncodedStatements), IntStmtView(),
                    splitStump, coreId, sink);
            }
        }
    } // end normal mode

    // Per-step delta clears are deliberately NOT done here. At splitCount>1 a
    // later executor's request generation still reads body.localHashMemoryDelta
    // / body.intLocalEncodedStatementsDelta for its own id % splitCount slice
    // (batches 2 and 5); clearing per-executor would wipe the input the next
    // executor needs. performElemPhase2 clears them ONCE, after every executor's
    // reqgen and before the merge (D-126). The clear is
    // still pre-merge, so admissionKeysAlgebra / deferredIntegrationPreps are
    // reset before applyFiringRecords repopulates them.

    // Checking is inline in BurstSink::consume; there is no separate fixpoint
    // loop. The hashburst captures firings onto the task's sealed-page record
    // chain (merged by the
    // caller), never into `intKnownStatements`, so the dependency skip-set is
    // burst-fixed and the single generate-and-check pass reaches the same fixed
    // point an iterated loop would (D-104). Per I-66 phase 2 never mutates the
    // LB's `isActive`; deactivation / discharge happen in phase 3's post-burst
    // absorb (`dischargeContradiction` / `dischargeToBeProved`). The streaming
    // early-exit only sets the external `burstShouldStop` flag, never the LB.


    // No return: the split-policy marker is this part's submatch count
    // (g_growthMatchCount), read by the worker right after this call and reduced to
    // the busiest part in the finalize sweep. The former emitted-request count
    // (sink.produced) no longer drives the cap or the split.

    // Release this slot's scratch string arena: return its blocks to the pool
    // and bump the generation so any ScratchString that escaped this executor
    // asserts at its access site (I-116). No retention — the next executor on
    // this slot re-acquires lazily. performElem2 is single-exit, so this runs
    // on every path.
    scratchArenas().forSlot(coreId).releaseAll();
    // The request-generation scratch arena rides the same per-task lifetime:
    // return its blocks to the pool on every exit path (the DFS stack frames and
    // the per-batch containers held this iteration).
    genScratchArenas().forSlot(coreId).releaseAll();
}

/// @see Declaration in `prover.hpp` for the full contract.
bool ExpressionAnalyzer::isStraggler(int64_t work, int64_t totalWork,
    int cores, int64_t minSplitWork) {
    assert(cores >= 1 && "isStraggler: cores is the machine core count (>= 1)");
    // Idle-core fair-share: an LB whose work alone exceeds the ideally-balanced
    // per-core load leaves cores idle, so splitting it across cores is justified.
    // The floor suppresses splitting a trivially cheap iteration (per-bucket work
    // below the fixed setup cost). Integer division -> deterministic verdict.
    const int64_t fairShare = totalWork / static_cast<int64_t>(cores);
    return work > fairShare && work >= minSplitWork;
}

/// @see Declaration in `prover.hpp` for the full contract.
void ExpressionAnalyzer::performElemPhase2(Memory& body,
    SealedPageSet* const* parts, int32_t partCount) {
    // Per-LB finalize of the FLAT executor pool: apply this LB's captured firing
    // records and drain the per-burst admission / integration staging. The
    // executors already ran in proveKernel's flat (LB, part) sweep, each read-only
    // on its LB (D-116) and appending its records to its own sealed page set;
    // `parts` hands THIS LB's sealed part sets in part order. This step is
    // single-LB and runs after the pool join, so it may freely mutate the LB. The
    // split decision (which LBs are stragglers next iteration) lives in proveKernel's
    // end-of-iteration stats pass via isStraggler, not here.

    // ENTRANCE isActive gate. performElemPhase1's absorb may have discharged the
    // LB (primed/CE contradiction, vacuous-truth induction step, toBeProved
    // match), flipping body.isActive=false; such an LB generated no executor
    // tasks, so partCount is 0 and the merge is skipped. The drains below
    // still run so any pre-deactivation staging is replayed.
    if (body.isActive) {
        // Per-step delta clears — ONCE per LB, before the merge (kept here, not
        // in performElem2, so a later part's request generation still sees the
        // deltas; D-126). Pre-merge, so admissionKeysAlgebra
        // / deferredIntegrationPreps are reset before applyFiringRecords fills
        // them.
        body.intLocalEncodedStatementsDelta.clear();
        body.localHashMemoryDelta.resetToFresh();
        body.admissionKeysAlgebra.clear();
        body.deferredIntegrationPreps.clear();

        // Merge the parts' captured deposits in canonical sorted order —
        // partition-independent (D-117): the same
        // firing SET sorts identically regardless of how it was split or in what
        // thread order the parts completed. This is why flat parallel execution
        // stays deterministic.
        this->applyFiringRecords(body, parts, partCount);
    }

    // Quiescence (D-194): admission-map churn this burst is a
    // mutation the phase-3 statement-count diff cannot see — a marker firing
    // registers an admission template without depositing a statement, yet it can
    // enable a fresh it_/int_ admission (and thus a firing) next burst. Flag it
    // so the LB stays awake. Read while the staging vectors still hold this
    // burst's records, before the drains below consume them.
    if (!body.admissionKeysAlgebra.empty() || !body.deferredIntegrationPreps.empty())
        body.mutatedThisBurst = true;

    // Drain the per-burst admission / integration records (replay in firing
    // order before phase 3's post-burst standardProcessing absorb). Run for
    // every LB, active or discharged in phase 1.
    this->drainAdmissionKeysAlgebra(body);
    this->drainDeferredIntegrationPreps(body);

    // The drained staging vectors carried sealed views into the tasks' page
    // sets; clear them NOW, before the post-join sweep frees the pages, so
    // no dangling view ever sits in a Memory container — and no staging can
    // leak across iterations into a later drain whose pages are long freed.
    // (The pre-merge clears above remain: they cover the inactive path's
    // bookkeeping and the D-126 delta contract.)
    body.admissionKeysAlgebra.clear();
    body.deferredIntegrationPreps.clear();

}

/// @see Declaration in `prover.hpp` for the full contract.
void ExpressionAnalyzer::performElemPhase1(Memory& body, unsigned coreId) {
    // Publish this worker's slot for the absorb door's per-slot scratch arena
    // (read deep in disintegration by prefixArgumentsWithU).
    g_currentCoreId = static_cast<int>(coreId);
    // Unified working-set handshake (the user's "a worker uploads itself"):
    // claim this LB and make it resident before ANYTHING reads it — the dump
    // trap below included. All three phases use the same handshake
    // (I-114). It reloads a cold LB, making room first
    // via the load<->evict exchange, and bounds its `Busy` wait with the
    // stuck-assert (I-113). The claim is released at the end
    // of the body so the steward may reclaim the LB once done. CE-filter
    // clones run with NO steward (always resident, never deloaded) and skip
    // the handshake entirely.
    if (steward)
        steward->claimAndLoadForWork(body, /*phase=*/1,
                                     lbdeload::kDeloadDirectory);

    // Quiescence SLEEP detector reset (D-194): the per-burst
    // non-statement mutation flag starts clean, and the statement-count baseline
    // is snapped BEFORE the pre-burst mail absorb below, so any mail-absorbed
    // deposit this burst counts as work (keeps the LB awake). Read back at
    // phase-3 exit to decide whether hasWork clears.
    body.mutatedThisBurst = false;
    body.encodedCountAtBurstStart = body.intEncodedStatementsCount();

    // ENTRY trap — delegate to hashburst_dump (relocated into phase 1 with the
    // performElem phase split; the dump sections / file path / target predicate
    // are unchanged, Rule 14). Mirrored in `feedback_dump_structure_immutable.md`.
    if (hashburst_dump::isTargetLB(body)) {
        hashburst_dump::dumpEntry(body, this->compiledExpressions);
    }

    // Per-step delta-class tracker: cleared at the start of every
    // elementary step (in performElemPhase1) so the set tracks ONLY classes
    // touched during THIS step's mail-absorb pass. Populated by
    // `updateEquivalenceClasses` immediately after a merged class is
    // committed to `equivalenceClassesMap`; consumed by
    // `applyEquiClasses` (between mail-absorb-clear and request generation).
    body.changedClassesThisStep.clear();

    { RT_SCOPE_HERE("PRE_FIXPOINT_MAIL_ABSORB");
    // ===================================================================
    // PRE-FIXPOINT MAIL ABSORB.
    // Absorbing mail before the fixpoint avoids one PK of latency between
    // a proven implication arriving at a receiver's mailIn and its rule
    // firing locally — the latency that pushes the Peano-incube
    // `__contradiction__(=[a,b])` LBs past MAX_NAME_IDS before they can
    // discharge.
    //
    // Mail arriving at this PK is absorbed FIRST, so the resulting
    // rules/facts are visible to THIS burst's request generation and
    // fixpoint. The persistent per-LB fields
    // workingMemory/externalStatements/intExternalStatements serve as
    // same-burst staging: cleared and refilled by this absorb, read by
    // the request-gen blocks below. D-76 deferred compact-form broadcast
    // and the Mail::implications deletion stay unchanged.
    // ===================================================================

    // Empty the per-burst external scratch, then refill from THIS cycle's
    // fresh mail below. These are NOT persistent across bursts (unlike the
    // local-premise containers); this absorb is their sole writer.
    body.workingMemory.resetToFresh();
    body.intExternalStatements.clear();

    // The sameIterationInternalMail absorb runs AFTER the hashburst loop
    // (the game resets after hashburst). The mailIn (parent-emitted)
    // absorb that follows STAYS in its pre-hashburst position.

    // Pre-hashburst absorb pipeline — a single `standardProcessing`
    // call routing three mail channels:
    //
    //   - externalMailIn = &body.mailIn — external arrivals from
    //     other LBs' mail, delivered by the MailLog phase-1 pull (status=3).
    //   - internalMailIn = body.nextIterationInternalMail —
    //     cross-iteration deferred emissions from the previous
    //     step's pre-burst discharge AND post-burst discharge
    //     AND the vacuous-truth path inside addExprToMemoryBlock
    //     AND the updateGlobalDirect siblings (status=1).
    //   - internalMailOut = body.sameIterationInternalMail —
    //     any discharge parent-scope emission during THIS absorb
    //     writes here; it then feeds the same step's hashburst
    //     rule firings (which also write to sameIter) and gets
    //     drained by the post-burst standardProcessing call.
    //     Zero-delay routing for pre-burst discharge emissions.
    //
    // The pre-burst call runs the full pipeline — mailIn-derived
    // statements get applied and discharged in the same step they
    // arrive, rather than waiting for the post-burst block's
    // first-pass.

    // New mail system (D-137): pull every un-ingested ancestor
    // batch into mailIn before this absorb drains it. The pull reads ancestors'
    // frozen logs and writes only this LB's own cursor row and its own mailIn —
    // parallel-safe, because commits happen only at the single-threaded
    // post-join seam. CE / compressor LBs are isolated/flat and never registered,
    // so they are gated out.
    int64_t trackedMailInBlocks = 0;
    if (!ceFilteringActive && !parameters.compressor_mode) {
        this->mailLog.pull(&body, body.mailIn);
        trackedMailInBlocks = body.mailIn.blocksHeld();
        const int64_t simultaneousMailInBlocks =
            routingMailInBlocksInFlight.fetch_add(
                trackedMailInBlocks, std::memory_order_relaxed)
            + trackedMailInBlocks;
        int64_t observedPeak =
            peakRoutingMailInBlocks.load(std::memory_order_relaxed);
        while (simultaneousMailInBlocks > observedPeak
            && !peakRoutingMailInBlocks.compare_exchange_weak(
                observedPeak, simultaneousMailInBlocks,
                std::memory_order_relaxed)) {
        }
    }

    // mailIn is HOT (I-101): the absorb reads its canonical
    // sorted snapshots directly — no transient heap Mail. Cleared here, after the
    // drain returns; nothing between standardProcessing's drain and its return
    // reads it.
    this->standardProcessing(body,
                             /*externalMailIn=*/&body.mailIn,
                             /*internalMailIn =*/body.nextIterationInternalMail,
                             /*internalMailOut=*/body.sameIterationInternalMail,
                             coreId);
    body.mailIn.clear();
    if (!ceFilteringActive && !parameters.compressor_mode) {
        const int64_t beforeRelease = routingMailInBlocksInFlight.fetch_sub(
            trackedMailInBlocks, std::memory_order_relaxed);
        assert(beforeRelease >= trackedMailInBlocks
            && "routing mailIn attribution counter underflow");
    }
    } // RT_SCOPE PRE_FIXPOINT_MAIL_ABSORB

    // Release the claim — phase 1 is done with this LB, so it is deloadable
    // again (the steward may now reclaim it). CE clones have no steward.
    if (steward) {
        assert(body.stewardClaim.load(std::memory_order_relaxed)
                   == static_cast<uint8_t>(Memory::StewardClaim::WorkerOwned)
               && "phase-1 release of an LB not held WorkerOwned");
        body.stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle),
            std::memory_order_release);
    }
}

/// @see Declaration in `prover.hpp` for the full contract.
void ExpressionAnalyzer::performElemPhase3(Memory& body, unsigned coreId) {
    // Publish this worker's slot for the absorb door's per-slot scratch arena
    // (read deep in disintegration by prefixArgumentsWithU).
    g_currentCoreId = static_cast<int>(coreId);
    // Unified working-set handshake (all phases equivalent): claim + load
    // this LB before phase 3 reads it; CE clones run with no steward.
    if (steward)
        steward->claimAndLoadForWork(body, /*phase=*/3,
                                     lbdeload::kDeloadDirectory);

    { RT_SCOPE_HERE("POST_FIXPOINT_MAIL_FLUSH");
    this->standardProcessing(body,
                             /*externalMailIn=*/nullptr,
                             /*internalMailIn =*/body.sameIterationInternalMail,
                             /*internalMailOut=*/body.nextIterationInternalMail,
                             coreId);

    // mailOut is no longer flushed here. It accumulates this step's fillMailOut
    // output (phase 1 + phase 3) and is drained once, single-threaded, by the
    // commit barrier in proveKernel (D-137). Clearing it here
    // would leave that commit nothing to ship.
    } // RT_SCOPE POST_FIXPOINT_MAIL_FLUSH

    { RT_SCOPE_HERE("REACT_TO_HYPO");
	reactToHypo(body);
    } // RT_SCOPE REACT_TO_HYPO

    { RT_SCOPE_HERE("END_OF_BURST_SANITIZE");
    // End-of-burst sanitization: walk `expandedImplications` and
    // `toBeProved` and rewrite entries whose `it_/int_` args are
    // now downprioritized under the active equi-classes. One pass
    // per burst — equi-class machinery has stabilized by this point.
    this->sanitizeHashMemory(body);
    this->sanitizeToBeProved(body);

    // Disproved-goal cleanup: probe the inbox against this LB's MAIN goals
    // and, on a hit, erase the goal plus its integration machinery — the
    // matched scope roots land on pendingWipeScopes so the radical wipe
    // below removes the nested state in this same burst.
    this->drainDisprovedGoals(body);

    // Dead _ordis_ branch retirement: a branch whose asserted disjunct is
    // refuted (staged by ordisMerge's probe) is wiped, its cohort's
    // bookkeeping shrinks, and convergence re-checks at the reduced count.
    // Runs after the disproof drain (wholesale-retired cohorts are gone
    // first) and before the wipe drain below so the branch wipes land in
    // this same burst.
    this->drainDeadOrBranches(body);

    // D-72: drain pending subtree wipes
    // queued during this burst by impl-closure call sites
    // (addExprToMemoryBlockKernel ~3872 / ~3950) and by
    // cleanUpOrIntegrationBranches. Wipe per closed scope runs the
    // full radical sweep over Memory + HashMemory; see the Doxygen
    // block above the declaration in memory.hpp for the structure
    // list and rationale. Deferring to here is what makes wiping
    // intStatementLevelsMap / equivalenceClassesMap safe: the kernel's
    // sortedNew loops have all completed for this burst, so no
    // in-flight iteration can be invalidated.
    if (!body.pendingWipeScopes.empty()) {
        // Quiescence (D-194): a subtree wipe eradicates state
        // (and can leave the net statement count unchanged if it also erased what
        // this burst added), so flag it as a mutation directly rather than relying
        // on the count diff.
        body.mutatedThisBurst = true;
        // Snapshot the cold set's ids then reset it, so wipeSubtree's own
        // inserts into other fields run against a fresh container.
        // Sort the IDS by their DECODED names (compareSpans == std::string
        // byte-lex): the ColdHashSet dedups vids and the interner is
        // injective, so the id sort is tie-free and reproduces the former
        // decoded-string std::sort drain order exactly (I-84). The id run
        // rides the per-slot gen-scratch byte-bump tier (I-124), reclaimed
        // by the popTo below; the comparator's decodeView spans are
        // consumed inside each pure-read comparison and nothing mints into
        // the NameMap before the wipes consume ids, not spans (I-3).
        const unsigned slot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : genScratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(slot);
        const ArenaOffset drainMark = gArena.cursor();
        const int32_t n = body.pendingWipeScopes.count();
        NameId* ids = reinterpret_cast<NameId*>(gArena.resolve(
            gArena.alloc(n * static_cast<int32_t>(sizeof(NameId)),
                         alignof(NameId))));
        for (int32_t i = 1; i <= n; ++i)
            ids[i - 1] = body.pendingWipeScopes.decode(i);
        body.pendingWipeScopes.resetToFresh();
        std::sort(ids, ids + n, [&](NameId a, NameId b) {
            return compareSpans(body.nameMap.decodeView(a),
                                body.nameMap.decodeView(b)) < 0;
        });
        for (int32_t k = 0; k < n; ++k)
            body.wipeSubtree(ids[k]);
        gArena.popTo(drainMark);
    }
    } // RT_SCOPE END_OF_BURST_SANITIZE

    // Quiescence SLEEP (D-194): the burst has finished
    // mutating. Clear hasWork unless this burst produced work — a statement
    // deposit / equi-class rewrite / subtree-wipe erase (count changed vs the
    // phase-1 baseline), a flagged non-statement mutation (admission churn or
    // wipe), or a pending emission on an INTERNAL mail channel (revival on
    // sameIter, a cross-iteration deferral on nextIter — both are inputs this
    // LB's own NEXT burst absorbs). A sound over-approximation: when any of
    // these holds the LB stays dirty and is swept again next iteration
    // (Rule 19 — a missed wake would be unsound, an extra sweep is merely
    // slow). Reads only this LB's own resident/never-deloaded state; no
    // residency input (I-106 / I-108).
    //
    // mailOut is deliberately NOT in this fold. It is OUTGOING-ONLY under the
    // pull model (I-64: fillMailOut and the deposit doors write it; the
    // single-threaded commit barrier is its sole consumer): no burst path
    // derives local state from it — the one historical burst-path read, the
    // retired transitive-ship dedup, is dead code — so pending mailOut content
    // can never make this LB's NEXT burst productive. And the barrier's commit
    // sweep iterates `bodies` regardless of sweep status, so that content is
    // committed whether or not this LB sweeps. Including it was never a valid
    // activity signal, and it false-fired the shadow check on the ROOT: the
    // post-join drains (the updateGlobal* theorem sends, the D-76 compaction
    // flush) append to the root's mailOut AFTER the barrier, so a genuinely
    // no-op root burst saw "mailOut non-empty" here and re-armed hasWork.
    const bool sleepMutated = body.mutatedThisBurst;
    const int32_t sleepCountNow = body.intEncodedStatementsCount();
    const bool sleepSameIterNonEmpty = !body.sameIterationInternalMail.empty();
    const bool sleepNextIterNonEmpty = !body.nextIterationInternalMail.empty();
    body.hasWork =
          sleepMutated
       || sleepCountNow != body.encodedCountAtBurstStart
       || sleepSameIterNonEmpty
       || sleepNextIterNonEmpty;

#ifdef QUIESCE_SHADOW_CHECK
    // Empirical soundness validator (off by default, in-tree). Under the flag the
    // skip filter does NOT exclude a would-be-skipped LB — it sweeps it anyway and
    // records the predicate's verdict in shadowWouldSkip. If that "quiescent"
    // burst actually produced work, hasWork is now set and this fires at the exact
    // LB: the predicate is unsound (a missed wake door or a missed mutation choke
    // point). Rule 19 in spirit — surface it loudly at its origin. The forensic
    // dump below prints exactly WHICH SLEEP condition re-triggered and the full
    // parentMemory chain (Rule 12 — full chain, never exprKey alone), so one run
    // identifies the miss precisely.
    if (body.shadowWouldSkip && body.hasWork) {
        std::ostringstream fx;
        fx << "[QUIESCE-SHADOW] predicate miss at burst="
           << ::gl::rt_tracker::g_currentHashburstIndex
           << " phase=3 (performElemPhase3 exit)\n";
        fx << "[QUIESCE-SHADOW] LB chain (innermost -> root):\n";
        for (const Memory* cur = &body; cur != nullptr;
             cur = cur->parentMemory) {
            fx << "[QUIESCE-SHADOW]   '" << cur->exprKey() << "'"
               << (cur->parentMemory == nullptr ? "  (root sentinel)" : "")
               << "\n";
        }
        fx << "[QUIESCE-SHADOW] SLEEP conditions re-triggered this burst:\n";
        fx << "[QUIESCE-SHADOW]   mutatedThisBurst        = "
           << (sleepMutated ? "TRUE" : "false") << "\n";
        fx << "[QUIESCE-SHADOW]   encodedStatements count = "
           << sleepCountNow << " now vs " << body.encodedCountAtBurstStart
           << " at phase-1 entry"
           << (sleepCountNow != body.encodedCountAtBurstStart
                   ? "  <-- CHANGED" : "  (unchanged)") << "\n";
        fx << "[QUIESCE-SHADOW]   sameIterationInternalMail non-empty = "
           << (sleepSameIterNonEmpty ? "TRUE" : "false") << "\n";
        fx << "[QUIESCE-SHADOW]   nextIterationInternalMail non-empty = "
           << (sleepNextIterNonEmpty ? "TRUE" : "false") << "\n";
        fx << "[QUIESCE-SHADOW] latch state: shadowWouldSkip=true (so at this "
              "iteration's active-build hasWork was false AND mailPeek was "
              "false); hasWork now=true\n";
        fx << "[QUIESCE-SHADOW] mailPeek(now, frozen logs) = "
           << (mailLog.mailPeek(&body) ? "TRUE" : "false")
           << "  (commits happen only at the barrier, so this equals the "
              "active-build value)\n";
        fx << "[QUIESCE-SHADOW] isActive=" << (body.isActive ? "true" : "false")
           << " dischargedForever=" << (body.dischargedForever ? "true" : "false")
           << " primedForContradiction="
           << (body.primedForContradiction ? "true" : "false")
           << " toBeProved count=" << body.intToBeProved.count() << "\n";
        std::cerr << fx.str() << std::flush;
        assert(!body.hasWork
            && "QUIESCE_SHADOW_CHECK: a would-be-skipped LB's burst mutated state "
               "— the quiescence predicate is unsound (missed wake or missed "
               "mutation choke point)");
    }
#endif

    // EXIT trap — delegate to hashburst_dump (relocated here from the inline
    // performElem tail with the phase split; dump unchanged, Rule 14).
    if (hashburst_dump::isTargetLB(body)) {
        hashburst_dump::dumpExit(body);
        // No abort — let the pipeline run to completion. The trap
        // continues firing for every EXIT of this LB; the trace
        // accumulates all of them.
    }

    // Release the claim — phase 3 is done with this LB; deloadable again
    // (the steward may now reclaim it). CE clones have no steward.
    if (steward) {
        assert(body.stewardClaim.load(std::memory_order_relaxed)
                   == static_cast<uint8_t>(Memory::StewardClaim::WorkerOwned)
               && "phase-3 release of an LB not held WorkerOwned");
        body.stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle),
            std::memory_order_release);
    }
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

    // Start with all digit args except the given 'digit'. findDigitArgs is now 0%
    // heap (sorted-unique StrSpan run); insert directly into the still-heap set.
    StrSpan digitBuf[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    const int32_t digitN = findDigitArgs(theorem, anchorInfo, digitBuf,
                                         ExecutionParameters::MAX_ADMISSION_REM_ARGS);
    for (int32_t i = 0; i < digitN; ++i)
        immutables.emplace(digitBuf[i].ptr, static_cast<std::size_t>(digitBuf[i].len));
    immutables.erase(digit);

    // Propagate immutability forward along the chain
    bool found = true;
    while (found) {
        found = false;

        for (std::size_t i = 0; i < chain.size(); ++i) {
            const std::string& expression = chain[i];
            const StrSpan coreSpan = extractExpressionSpan(StrSpan(expression));

            // Use the pre-populated 'operators' attribute for efficiency
            if (this->operators.find(std::string_view(coreSpan.ptr,
                    static_cast<std::size_t>(coreSpan.len))) != this->operators.end()) {

                const ce::CoreExpressionConfig* cfg = coreConfig(coreSpan);
                assert(cfg && "findImmutableArgs: operator core must be compiled");
                const std::vector<std::string> args = ce::getArgs(expression);

                // 1. Check if all configured Inputs are already Immutable
                bool inputsAreImmutable = true;
                for (int idx : cfg->inputIndices) {
                    // BLUNT ASSERT: Fail if config index is out of bounds for this expression
                    assert(idx >= 0 && idx < static_cast<int>(args.size()));

                    if (immutables.find(args[idx]) == immutables.end()) {
                        inputsAreImmutable = false;
                        break;
                    }
                }

                // 2. If inputs are immutable, add configured Outputs to immutables
                if (inputsAreImmutable) {
                    for (int idx : cfg->outputIndices) {
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
/// step by step, taking the `SimpleMapStore` edge that matches each chain
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
        Memory* child = simpleMapStore.findChild(memoryBlock, key);
        if (child == NULL) {
            return NULL;
        }
        memoryBlock = child; // follow the pointer
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
    const ce::CoreExpressionConfig* cfg = coreConfig(StrSpan(opName));
    if (cfg == nullptr) return false;
    if (cfg->outputIndices.empty()) return false;  // not an operator

    int outIdx = cfg->outputIndices[0];  // 0-based
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
/// @brief Record one proven direct theorem into the global theorem list —
/// direct theorem path.
///
/// @details
/// Post-`pool.join()` collector for theorems that were proved by a
/// direct path (rather than via the auxy-indexed indirect path
/// handled by `updateGlobal`). Same I-28 contract: workers never
/// touch the global state during the parallel phase, only seal the
/// theorem bytes onto `updateGlobalDirectPages`; `drainUpdateGlobalDirect`
/// is the single sorted replay point that calls this per record.
///
/// Result: a fresh row in `globalTheoremList` and matching entries
/// in `fullTheoremList` and `cachedProofStacks` so the proof-graph
/// generator can pick the theorem up at export time.
///
/// @param theorem Theorem text being recorded.
/// @param coreId  Producing thread id.
/// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28).
/// @see `updateGlobal` — auxy-indexed counterpart.
/// @brief Append a theorem tuple to `globalTheoremList` unless the theorem
///        string is already present — the sink dedup.
///
/// @details
/// The single door for every `globalTheoremList` insertion. Locks
/// `theoremListMutex`, probes the companion `globalTheoremStrings` set, and
/// appends only on first sight of the theorem string (first emission wins).
/// See the declaration in `prover.hpp` for the full contract.
///
/// @param theorem  The theorem string (the dedup key).
/// @param method   Emission-path label (`direct`, `induction`, ...).
/// @param aux2     Method-specific third column.
/// @param aux3     Method-specific fourth column.
/// @param producer The emitting grid LB (vacuous-reversion attribution), or
///                 `nullptr` for producer-less paths.
/// @return `true` when appended, `false` when the duplicate was dropped.
bool ExpressionAnalyzer::appendGlobalTheorem(const std::string& theorem,
    const std::string& method, const std::string& aux2,
    const std::string& aux3, const Memory* producer) {
    std::lock_guard<std::mutex> lock(this->theoremListMutex);
    if (!this->globalTheoremStrings.insert(theorem).second) return false;
    this->globalTheoremList.emplace_back(theorem, method, aux2, aux3);
    this->globalTheoremProducers.push_back(producer);
    return true;
}

/// @brief Retract every global-theorem row whose producer LB turned out
///        unintentionally contradictory.
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full contract.
/// Implementation notes: the sweep walks the list back to front so the
/// aligned erases keep indices valid; the `fullTheoremList` sweep matches on
/// the theorem string (the same key the list rows carry); the dedup-set
/// erase frees the string for a later legitimate re-prove.
///
/// @return Nothing.
/// @invariant `globalTheoremProducers` stays aligned with
///            `globalTheoremList` across the erases.
/// @see appendGlobalTheorem, hasContradictedAncestor.
void ExpressionAnalyzer::revertVacuousGlobalTheorems() {
    std::lock_guard<std::mutex> lock(this->theoremListMutex);
    assert(this->globalTheoremProducers.size()
               == this->globalTheoremList.size()
        && "producer attribution out of step with globalTheoremList");
    for (std::size_t i = this->globalTheoremList.size(); i-- > 0; ) {
        const Memory* producer = this->globalTheoremProducers[i];
        if (producer == nullptr || !hasContradictedAncestor(producer)) continue;
        const std::string theorem = std::get<0>(this->globalTheoremList[i]);
        std::cout << "[VACUOUS-REVERT] " << theorem << std::endl;
        this->retractedVacuousTheorems.push_back(theorem);
        this->globalTheoremStrings.erase(theorem);
        this->globalTheoremList.erase(this->globalTheoremList.begin()
            + static_cast<std::ptrdiff_t>(i));
        this->globalTheoremProducers.erase(this->globalTheoremProducers.begin()
            + static_cast<std::ptrdiff_t>(i));
        for (std::size_t j = this->fullTheoremList.size(); j-- > 0; ) {
            if (std::get<0>(this->fullTheoremList[j]) == theorem) {
                this->fullTheoremList.erase(this->fullTheoremList.begin()
                    + static_cast<std::ptrdiff_t>(j));
            }
        }
    }
}

/// @brief Post-prove vacuity certificate — same-chain contradictory heads
///        retire every theorem on that premise chain as a vacuous truth.
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full
/// contract. Implementation notes: theorems group by their SORTED premise
/// chain (`ce::disintegrateImplication` chain elements, unit-separator
/// joined — multiset identity); a group holding both some head `X` and
/// `negate(X)` is classified whole; erasures walk descending so the
/// aligned `globalTheoremProducers` indices stay valid.
///
/// @return The classified theorem strings, in pool order.
/// @invariant `globalTheoremProducers` stays aligned with
///            `globalTheoremList` across the erases.
/// @see revertVacuousGlobalTheorems.
std::vector<std::string> ExpressionAnalyzer::classifyVacuousPremisePairs() {
    std::lock_guard<std::mutex> lock(this->theoremListMutex);
    assert(this->globalTheoremProducers.size()
               == this->globalTheoremList.size()
        && "producer attribution out of step with globalTheoremList");
    struct ChainGroup {
        std::unordered_set<std::string> heads;
        std::vector<std::size_t> indices;
        std::vector<std::string> headByIndex;
    };
    std::map<std::string, ChainGroup> byChain;
    for (std::size_t i = 0; i < this->globalTheoremList.size(); ++i) {
        const std::string& thm = std::get<0>(this->globalTheoremList[i]);
        std::vector<std::tuple<std::string, std::vector<std::string>,
                               std::set<std::string>>> tempChain;
        const std::string head = ce::disintegrateImplication(
            thm, tempChain, this->coreExpressionMap);
        std::vector<std::string> ky;
        ky.reserve(tempChain.size());
        for (const auto& el : tempChain) ky.push_back(std::get<0>(el));
        std::sort(ky.begin(), ky.end());
        std::string key;
        for (const auto& el : ky) { key += el; key += '\x1f'; }
        ChainGroup& g = byChain[key];
        g.heads.insert(head);
        g.indices.push_back(i);
        g.headByIndex.push_back(head);
    }
    std::vector<std::size_t> vacIdx;
    for (const auto& kv : byChain) {
        const ChainGroup& g = kv.second;
        if (g.indices.size() < 2) continue;
        bool contradictory = false;
        for (const std::string& h : g.headByIndex) {
            if (g.heads.count(negate(h)) != 0) { contradictory = true; break; }
        }
        if (!contradictory) continue;
        vacIdx.insert(vacIdx.end(), g.indices.begin(), g.indices.end());
    }
    std::sort(vacIdx.begin(), vacIdx.end());
    std::vector<std::string> classified;
    classified.reserve(vacIdx.size());
    for (const std::size_t i : vacIdx)
        classified.push_back(std::get<0>(this->globalTheoremList[i]));
    for (std::size_t j = vacIdx.size(); j-- > 0; ) {
        const std::size_t i = vacIdx[j];
        const std::string& theorem = classified[j];
        std::cout << "[VACUOUS-PREMISE-PAIR] " << theorem << std::endl;
        this->globalTheoremStrings.erase(theorem);
        this->globalTheoremList.erase(this->globalTheoremList.begin()
            + static_cast<std::ptrdiff_t>(i));
        this->globalTheoremProducers.erase(this->globalTheoremProducers.begin()
            + static_cast<std::ptrdiff_t>(i));
        for (std::size_t f = this->fullTheoremList.size(); f-- > 0; ) {
            if (std::get<0>(this->fullTheoremList[f]) == theorem) {
                this->fullTheoremList.erase(this->fullTheoremList.begin()
                    + static_cast<std::ptrdiff_t>(f));
            }
        }
    }
    return classified;
}

void ExpressionAnalyzer::updateGlobalDirect(const std::string& theorem, int coreId,
    const Memory* producer) {
    // 1) Disintegrate to chain + head
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    Mail mailOut;

    deactivateUnnecessary(theorem, this->body);  // deactivates all blocks tied to this theorem.

    std::string value = ce::disintegrateImplication(theorem, tempChain, this->coreExpressionMap);

    std::vector<std::string> ky;
    ky.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

    // The opposite-polarity conjecture (same chain, negated head) is settled
    // by this theorem too: a verbatim contradiction LB's disproof settles the
    // ORIGINAL conjecture, and a proved head makes the opposite-head
    // conjecture false — either way its recursion auxiliaries are pointless.
    // Retire them through the same matching walk; a twin form that matches
    // no originalAuxyMap entry is a defined no-op.
    deactivateUnnecessary(reconstructImplication(ky, negate(value)), this->body);

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

            this->appendGlobalTheorem(refTheoremCompiled, "reformulated statement", theorem, "-1", producer);
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
                // L3 span-record: the "implication" record antecedents (the
                // broadcast theorem + its N decomposition-chain keys, all stable
                // std::string locals) fill a bounded stack OriginDep[] for the
                // span door; "main" is a static literal.
                OriginDep implDeps[64];
                int implDepN = 0;
                implDeps[implDepN++] = { StrSpan(refTheoremCompiled), StrSpan("main", 4) };
                for (const std::string& keyElement : ky) {
                    assert(implDepN < 64 && "broadcast origin chain exceeds 64");
                    implDeps[implDepN++] = { StrSpan(keyElement), StrSpan("main", 4) };
                }

                int lvRunR[65];
                assert(static_cast<int>(ky.size()) + 1 <= 65
                    && "broadcast level run exceeds lvRunR");
                for (int i = 0; i <= static_cast<int>(ky.size()); ++i) lvRunR[i] = i;
                const int32_t lvRunRN = static_cast<int32_t>(ky.size()) + 1;

                // Cross-LB deposit into the target block's
                // `nextIterationInternalMail`. memoryBlockR is a
                // different LB; its elementary step will be
                // run later by the parallel pool, and its pre-burst
                // standardProcessing call will drain
                // nextIterationInternalMail alongside mailIn.
                if (!memoryBlockR->dischargedForever) {
                    // SEAM DOOR (D-196): memoryBlockR is a
                    // DIFFERENT LB whose nextIterationInternalMail rides its
                    // DELOADABLE arena (I-102). The uniform handshake makes it
                    // resident, claim-correct (never Dumped-but-resident), and
                    // holds WorkerOwned across the write so the barrier seam
                    // window's executors cannot evict it mid-deposit; the
                    // release below makes it evictable again (the deposit
                    // lives in the arena and rides the raw image). A
                    // dischargedForever target is skipped -- it never drains
                    // its nextIter again (dead deposit), and the reload
                    // asserts on a discharged LB.
                    steward->claimAndLoadForWork(*memoryBlockR, /*phase=*/4,
                                                 lbdeload::kDeloadDirectory);
                    const int maxOrigins = parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr;
                    insertInternalStatement(memoryBlockR->nextIterationInternalMail,
                        memoryBlockR->nameMap, StrSpan(valueR), StrSpan("main", 4),
                        lvRunR, lvRunRN);
                    if (parameters.trackHistory) {
                        addInternalMailOrigin(memoryBlockR->nextIterationInternalMail,
                                      memoryBlockR->originInterner,
                                      StrSpan(valueR), StrSpan("main", 4),
                                      OriginTag::implication, implDeps, implDepN,
                                      maxOrigins);
                    }
                    // WAKE DOOR 1 (D-194): a cross-LB deposit
                    // into memoryBlockR's nextIterationInternalMail is new work it
                    // will drain next step — mark it dirty so the skip filter
                    // sweeps it even if it had converged.
                    memoryBlockR->hasWork = true;
                    memoryBlockR->stewardClaim.store(
                        static_cast<uint8_t>(Memory::StewardClaim::Idle),
                        std::memory_order_release);
                }
            }

            // ASIC 0.1 reshuffle: the legacy Mail::implications channel is
            // removed; this implication now travels SOLELY as the D-76
            // compact (implication<N>[...]) statement (deferred compile +
            // deposit, drained single-threaded post-pool.join — parallel
            // worker path, so never inline; I-28). Linked to the original
            // by a `compilation` origin (also the receiver's mandatory
            // paired-origin; I-44: exprOriginMap is documentation, not a
            // proof input).
            recordPendingCompaction(refTheoremCompiled, static_cast<int>(ky.size()), coreId);

        }

    }

    // 5) Incubator: back-reformulate operator-equality pattern
    // Must be before the mail merge so the back-reformulated implication ships.
    std::string backReformulated;
    willBackReformulate = tryBackReformulateOperatorHead(theorem, backReformulated);
    if (!willBackReformulate) {
        // No back-reformulation: record original theorem. ASIC 0.1
        // reshuffle: it travels SOLELY as the D-76 compact statement
        // (legacy Mail::implications removed); deferred compile+deposit.
        recordPendingCompaction(theorem, static_cast<int>(origKy.size()), coreId);
        if (parameters.trackHistory) {
            ExpressionWithValidity ev(theorem, "main");
            addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }
        this->appendGlobalTheorem(theorem, "direct", "-1", "-1", producer);
        std::cout << theorem << std::endl;
        checkOrCompletion(theorem, coreId);
    }
    if (willBackReformulate) {
        // Register original as "direct" so it gets a full proof chapter.
        // ASIC 0.1 reshuffle: it travels SOLELY as the D-76 compact
        // statement (legacy Mail::implications removed); deferred deposit.
        recordPendingCompaction(theorem, static_cast<int>(origKy.size()), coreId);
        if (parameters.trackHistory) {
            ExpressionWithValidity ev(theorem, "main");
            addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }
        this->appendGlobalTheorem(theorem, "direct", "-1", "-1", producer);
        this->appendGlobalTheorem(
            backReformulated, "incubator back reformulation", theorem, "-1",
            producer);
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

        // ASIC 0.1 reshuffle: back-reformulated implication travels SOLELY
        // as the D-76 compact statement (legacy Mail::implications
        // removed); deferred compile+deposit, single-threaded post-join.
        recordPendingCompaction(backReformulated, static_cast<int>(brKey.size()), coreId);
    }

    // 6) Flush mail. New mail system (D-137): this local mailOut
    // carries only "theorem" / "reformulated statement" origin lines
    // (documentation, I-44 -- the proof-affecting statements ride
    // nextIterationInternalMail). Merge them into the root's mailOut; the NEXT
    // commit barrier ships them into the root's log for descendants to pull,
    // preserving the old post-smashMail one-iteration-later delivery. Runs
    // single-threaded post-join. The root's mailOut is deloadable, so the seam
    // claims and reloads it across the deposit.
    assert(coreId >= 0);
    steward->claimAndLoadForWork(this->body, /*phase=*/4,
                                 lbdeload::kDeloadDirectory);
    mergeBatchIntoMailOut(mailOut, this->body);
    this->body.stewardClaim.store(
        static_cast<uint8_t>(Memory::StewardClaim::Idle),
        std::memory_order_release);
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
        simpleMapStore.forEachChild(node, [&](const gl::StrSpan&, Memory* child) {
            if (child) {
                traverse(child);
            }
        });

        // 2. Check if all children are deactivated
        bool allChildrenDeactivated = true;
        simpleMapStore.forEachChild(node, [&](const gl::StrSpan&, Memory* child) {
            if (child && child->isActive) {
                allChildrenDeactivated = false;
            }
        });

        // 3. Deactivate node if conditions are met:
        //    - All children are inactive
        //    - No pending proofs in the current node

        // The survey reads intToBeProved, which now lives in the PERSISTENT pool
        // and is always resident (I-108) — so it runs on
        // every ACTIVE node regardless of whether the node's MAIN arena is
        // deloaded. This is also what makes the quiescent-burst skip
        // (D-194) preserve the I-48 deactivation schedule with
        // zero extra work: this survey is a tree-wide, post-join walk from the
        // root, so a SKIPPED LB (unswept this iteration but still isActive) is
        // still surveyed here, reading only never-deloaded state (intToBeProved
        // persistent I-108; child isActive + simpleMap edges on the never-deloaded
        // LB slab, I-109 / I-110) — it forces no reload of a cold skipped LB.
        // Gating on isActive (not residency) is the determinism fix:
        // the old residency gate SKIPPED a deloaded-but-active node, and which
        // nodes are deloaded is timing-dependent, so the deactivation decision
        // (and thus the theorem set) was non-deterministic. An already-inactive
        // node needs no survey (deactivation is monotonic; isActive implies
        // !dischargedForever, so the persistent registry is still populated).
        if (node->isActive) {
            int mainValidityCount = 0;
            for (int32_t tbpId = 1; tbpId <= node->intToBeProved.count(); ++tbpId) {
                // Low 32 key bits carry the validity id; "main" is MAIN_ID (I-48).
                if (Codec<StatementKey>::decode(node->intToBeProved.keyAt(tbpId)).validity
                    == NameMap::MAIN_ID) {
                    mainValidityCount++;
                }
            }

            if (allChildrenDeactivated && mainValidityCount == 0
                && !node->primedForContradiction) {
                node->isActive = false;
            }
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
/// whose `intLocalEncodedStatementsSet` is empty after the proof
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
    const std::string head =
        ce::disintegrateImplication(theorem, tempChain, this->coreExpressionMap);
    if (tempChain.empty()) return;

    // Walk down the chain from root, collecting the blocks on the path
    Memory* node = &root;
    std::vector<Memory*> chainBlocks;
    chainBlocks.reserve(tempChain.size());

    for (const auto& t : tempChain) {
        const std::string& key = std::get<0>(t);
        Memory* nextNode = simpleMapStore.findChild(node, key);
        if (nextNode == nullptr) {
            // Path not present (may have been pruned earlier) — nothing to do safely.
            return;
        }
        node = nextNode;
        chainBlocks.push_back(node);
    }

    // Retire both contradiction twins of this settled head at the final
    // node. Emission-time counterpart of the D-224 goal-closure staging:
    // the staging path fires only when the chain LB's own goal closes,
    // which for a theorem proved by induction or in another LB happens one
    // or more bursts after emission (on the broadcast re-fire) — this flip
    // closes that lag at the emission drain, the same single-threaded seam
    // the staged drain uses. A missing twin is a defined case (the head
    // never took contradiction LBs, or they were retired already).
    for (const std::string& seed : { head, this->negate(head) }) {
        if (Memory* twin = simpleMapStore.findChild(node, "__contradiction__" + seed)) {
            twin->isActive = false;
        }
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

        if (Memory* c1 = simpleMapStore.findChild(node, k_rec_in2)) {
            c1->isActive = false;

        }
        if (Memory* c2 = simpleMapStore.findChild(node, k_eq_side)) {
            c2->isActive = false;

        }
        // No asserts: partial deactivation is still useful/safe.
    }

    // 4) Bubble up: if a block has no active children, deactivate it too
    for (auto rit = chainBlocks.rbegin(); rit != chainBlocks.rend(); ++rit) {
        Memory* block = *rit;
        bool anyActiveChild = false;
        simpleMapStore.forEachChild(block, [&](const gl::StrSpan&, Memory* child) {
            if (child && child->isActive) { anyActiveChild = true; }
        });
        // Disable the LB when no active children AND no MAIN-namescope
        // toBeProved entries remain. Entries at deeper scopes (hypothetical,
        // OR-branch, integration boundaries) do NOT keep the LB alive —
        // their proof obligations belong to the deeper context and the LB
        // itself has no work left at its own scope. If only deeper-scope
        // entries linger and no children are active, disable.
        // intToBeProved is in the PERSISTENT pool (always resident,
        // I-108), so the survey runs on every ACTIVE block
        // regardless of its MAIN arena's deload state. Gating on isActive (not
        // residency) is the determinism fix — the old residency gate skipped a
        // deloaded-but-active block, and which blocks are deloaded is
        // timing-dependent, so the deactivation decision was non-deterministic.
        if (!anyActiveChild && block->isActive)
        {
            // I-48 survey on the packed keys: the low 16 bits carry the
            // validity id, and "main" is the pre-registered MAIN_ID.
            std::size_t mainTbpCount = 0;
            for (int32_t tbpId = 1; tbpId <= block->intToBeProved.count(); ++tbpId) {
                if (Codec<StatementKey>::decode(block->intToBeProved.keyAt(tbpId)).validity
                    == NameMap::MAIN_ID) {
                    ++mainTbpCount; break;
                }
            }
            if (mainTbpCount == 0) {
                block->isActive = false;
            }
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
/// [I-28](../../docs/agentic_swdd/30_invariants.md#i-28); they enqueue tuples
/// `(auxyIndex, allLevelsInvolved, coreId)` instead. This function
/// is the single drain point that locks `theoremListMutex` /
/// `dependenciesMutex` once and folds every queued tuple into the
/// global side-tables.
///
/// @param auxyIndex          Auxiliary index identifying the theorem.
/// @param allLevelsInvolved  Flag: theorem spans every LB level.
/// @param coreId             Producing thread id (purely informational).
/// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — drain runs
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
        // Vacuous-premise emission gate (the induction twin of the
        // drainUpdateGlobalDirect gate): an induction theorem whose chain
        // LB sits at or below a contradicted premise set is vacuous — do
        // not promote it. Runs at the single-threaded post-join drain of
        // updateGlobalTuples, so the parent walk is race-free.
        if (hasContradictedAncestor(memoryBlock)) {
            return;
        }
        // --- Induction typing gate ---
        // Before promoting an induction theorem, verify the induction variable is
        // constrained to N (anchor_args[0]). A sound induction requires
        // `(in[indVar,N])` derivable from the current chain; without it, the
        // theorem universally quantifies over everything rather than over N, and
        // the implicit-typing inference of the original induction-scheduling is
        // unsound. See docs/agentic_swdd/induction_typing_plan.md.
        if (!indVar.empty()) {
            const std::string anchorN = findNName(*memoryBlock);
            const std::string typingGoal =
                std::string("(in[") + indVar + "," + anchorN + "])";

            const StatementFlags* typingRow = lookupStatementFlags(
                memoryBlock->intKnownStatements, memoryBlock->nameMap, typingGoal, "main");
            if (!(typingRow && typingRow->registered)) {
                // Typing not established — reject induction promotion silently.
                return;
            }
        }

        // Non-minting packed probe: a goal whose original was never
        // interned cannot be registered, so a lookup miss is a definitive
        // probe miss (the same negative branch the string find produced).
        const int32_t tbpValId =
            lookupToBeProved(memoryBlock->intToBeProved, memoryBlock->nameMap,
                             value, "main");
        if (tbpValId != 0) {
            // Python asserts this is an empty set
            assert(memoryBlock->intToBeProved.runLen(tbpValId) == 0);
            const NameId tbpOrigId = memoryBlock->nameMap.lookup(value);
            assert(tbpOrigId != 0);
            memoryBlock->intToBeProved.eraseSet(
                packStatementKey(tbpOrigId, NameMap::MAIN_ID));

            // Build origin and levels
            // L3 span-record: the "implication" record antecedents (proved head
            // + N decomposition-chain keys, stable std::string locals) fill a
            // bounded stack OriginDep[]; "main" is a static literal.
            OriginDep implDeps[64];
            int implDepN = 0;
            implDeps[implDepN++] = { StrSpan(expr), StrSpan("main", 4) };
            for (const std::string& keyElement : ky) {
                assert(implDepN < 64 && "broadcast origin chain exceeds 64");
                implDeps[implDepN++] = { StrSpan(keyElement), StrSpan("main", 4) };
            }

            int lvRun[65];
            assert(static_cast<int>(ky.size()) + 1 <= 65
                && "broadcast level run exceeds lvRun");
            for (int i = 0; i <= static_cast<int>(ky.size()); ++i) lvRun[i] = i;
            const int32_t lvRunN = static_cast<int32_t>(ky.size()) + 1;

            // Deposit the proved head into this LB's
            // `nextIterationInternalMail` for absorption at the start
            // of its next elementary step (alongside mailIn).
            if (!memoryBlock->dischargedForever) {
                // SEAM DOOR (D-196): this LB just ran its
                // phases, but the barrier seam window's executors may have
                // evicted it since its release — the uniform handshake reloads
                // it claim-correctly and holds WorkerOwned across the write;
                // the release makes it evictable again. A dischargedForever LB
                // has no next step to drain the deposit (dead), so it is
                // skipped (I-102).
                steward->claimAndLoadForWork(*memoryBlock, /*phase=*/4,
                                             lbdeload::kDeloadDirectory);
                const int maxOrigins = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr;
                insertInternalStatement(memoryBlock->nextIterationInternalMail,
                    memoryBlock->nameMap, StrSpan(value), StrSpan("main", 4),
                    lvRun, lvRunN);
                if (parameters.trackHistory) {
                    addInternalMailOrigin(memoryBlock->nextIterationInternalMail,
                                  memoryBlock->originInterner,
                                  StrSpan(value), StrSpan("main", 4),
                                  OriginTag::implication, implDeps, implDepN,
                                  maxOrigins);
                }
                // WAKE DOOR 2 (D-194): updateGlobal deposits
                // the proved head into this LB's nextIterationInternalMail (and
                // erased a main goal above) — new work next step; keep it swept.
                memoryBlock->hasWork = true;
                memoryBlock->stewardClaim.store(
                    static_cast<uint8_t>(Memory::StewardClaim::Idle),
                    std::memory_order_release);
            }

            // ASIC 0.1 reshuffle: the legacy Mail::implications channel is
            // removed; this implication travels SOLELY as the D-76 compact
            // (implication<N>[...]) statement, linked to the original by a
            // `compilation` origin (also the receiver's mandatory
            // paired-origin; I-44: exprOriginMap is documentation, not a
            // proof input). Deferred compile+deposit (parallel worker path
            // -> I-28); drained single-threaded post-pool.join.
            recordPendingCompaction(expr, static_cast<int>(ky.size()), coreId);

            if (parameters.trackHistory) {
                ExpressionWithValidity ev(expr, "main");
                addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }

            // ---- record in globalTheoremList (short critical section) ----
            //
            // FullBind-rebuild expr before registering. The runtime side keeps
            // the as-scheduled `expr` (used downstream for compaction, origin,
            // reformulation); the REGISTRY entry must be FullBind so
            // the verifier's origin meta-check (which walks the chapter's
            // binary-canonical citation, itself emitted in FullBind by
            // broadcastTheorems / the deferred-compaction drain) finds a
            // left-column hit.
            const std::string exprFullBind =
                this->reconstructImplicationFullBind(ky, value);
            this->appendGlobalTheorem(exprFullBind, "induction", indVar,
                recCounter, memoryBlock);
            std::cout << exprFullBind << std::endl;
            // checkOrCompletion below intentionally keeps `expr` (the
            // partial-bind, as-scheduled form) because or_pairs.txt is loaded
            // verbatim from the conjecturer's output and stores theorems in
            // that pre-FullBind shape. Comparing against `exprFullBind` would
            // miss every OR-pair match.
            checkOrCompletion(expr, coreId);

            std::vector<std::pair<std::string, std::string>> outTheorems;
            if (reformulateTheorem(expr, outTheorems)) {
                for (const auto& pair : outTheorems) {
                    const std::string& refTheoremRaw = pair.first;
                    const std::string& refTheoremCompiled = pair.second;

                    this->appendGlobalTheorem(refTheoremCompiled,
                        "reformulated statement", expr, "-1", memoryBlock);
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
                        // L3 span-record: the "implication" record antecedents
                        // (broadcast theorem + N decomposition-chain keys, stable
                        // std::string locals) fill a bounded stack OriginDep[].
                        OriginDep implDeps[64];
                        int implDepN = 0;
                        implDeps[implDepN++] = { StrSpan(refTheoremCompiled), StrSpan("main", 4) };
                        for (const std::string& keyElement : ky) {
                            assert(implDepN < 64 && "broadcast origin chain exceeds 64");
                            implDeps[implDepN++] = { StrSpan(keyElement), StrSpan("main", 4) };
                        }

                        int lvRunR[65];
                        assert(static_cast<int>(ky.size()) + 1 <= 65
                            && "broadcast level run exceeds lvRunR");
                        for (int i = 0; i <= static_cast<int>(ky.size()); ++i) lvRunR[i] = i;
                        const int32_t lvRunRN = static_cast<int32_t>(ky.size()) + 1;

                        // Cross-LB deposit into the target block's
                // `nextIterationInternalMail` for next-iter absorb.
                if (!memoryBlockR->dischargedForever) {
                    // SEAM DOOR (D-196): the uniform
                    // handshake reloads the recipient claim-correctly and
                    // holds WorkerOwned across the write (the barrier seam
                    // window's executors cannot evict it mid-deposit); the
                    // release makes it evictable again. A dischargedForever
                    // target is skipped -- it never drains its nextIter again
                    // (dead deposit; I-102).
                    steward->claimAndLoadForWork(*memoryBlockR, /*phase=*/4,
                                                 lbdeload::kDeloadDirectory);
                    const int maxOrigins = parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr;
                    insertInternalStatement(memoryBlockR->nextIterationInternalMail,
                        memoryBlockR->nameMap, StrSpan(valueR), StrSpan("main", 4),
                        lvRunR, lvRunRN);
                    if (parameters.trackHistory) {
                        addInternalMailOrigin(memoryBlockR->nextIterationInternalMail,
                                      memoryBlockR->originInterner,
                                      StrSpan(valueR), StrSpan("main", 4),
                                      OriginTag::implication, implDeps, implDepN,
                                      maxOrigins);
                    }
                    // WAKE DOOR 1 (D-194): cross-LB deposit
                    // into memoryBlockR's nextIterationInternalMail — new work
                    // next step; keep it swept.
                    memoryBlockR->hasWork = true;
                    memoryBlockR->stewardClaim.store(
                        static_cast<uint8_t>(Memory::StewardClaim::Idle),
                        std::memory_order_release);
                }
                    }

                    // ASIC 0.1 reshuffle: reformulated implication travels
                    // SOLELY as the D-76 compact statement (legacy
                    // Mail::implications removed); deferred compile+deposit.
                    recordPendingCompaction(refTheoremCompiled, static_cast<int>(ky.size()), coreId);
                }
            }


            // Flush mail. New mail system (D-137): the local
            // mailOut carries only origin lines (documentation, I-44); merge them
            // into the root's mailOut, shipped by the NEXT commit barrier
            // (one-iteration delay, matching the old post-smashMail timing).
            // Single-threaded post-join; claim/reload the root because mailOut
            // rides its deloadable arena.
            assert(coreId >= 0);
            steward->claimAndLoadForWork(this->body, /*phase=*/4,
                                         lbdeload::kDeloadDirectory);
            mergeBatchIntoMailOut(mailOut, this->body);
            this->body.stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        }
    }
}


/// @brief Seed the recursion admission budget for a freshly-registered
///        operator GOAL at the scope that owns the goal's numeral.
///
/// @details
/// Sole caller: `addExprToMemoryBlock`'s `status == 2` (toBeProved) branch,
/// when the goal's head operator has input args. The goal `(op[…,X])` is an
/// operator output to be PROVED; the admission budget seeded here is what lets
/// the hash engine unfold that operator backward while chasing the goal, capped
/// by `maxAdmissionDepth` / `maxSecondaryNumber`. The back-climb finds the first
/// ancestor LB whose `exprKey` shares a digit argument with the goal — the
/// recursion/fold block that introduced that numeral — and seeds the admission
/// there (delegating to `updateAdmissionMap`). A self-match seeds the calling
/// LB. This is goal-side backward-admission setup, NOT result aggregation.
///
/// I-28 detect-and-defer (trial): when the climb lands on a STRICT ANCESTOR and
/// the caller is a parallel phase-1/phase-3 worker (`g_inParallelWorkerPhase`),
/// writing the ancestor would be a cross-LB write that races sibling recursion
/// children sharing one fold ancestor. In that case the resolved
/// `updateAdmissionMap` call is sealed onto the shared `deferredAncestorPages`
/// `SealedPageSet` record chain and replayed single-threaded by
/// `drainDeferredAncestorAdmissions` post-join. A
/// self-write (always the worker's own LB) and any write outside the parallel
/// sweep (grid build, the drain itself) stay inline.
///
/// @param expr               Operator goal whose admission budget is seeded.
/// @param memoryBlock        LB the goal was registered in (climb start).
/// @param maxAdmissionDepth  Recursion admission depth cap.
/// @param maxSecondaryNumber Recursion secondary-iteration cap.
/// @param partOfRecursion    Marks the seeded entry as recursion-owned.
/// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — a
///            cross-LB ancestor seed never lands inline in a parallel worker.
/// @see `drainDeferredAncestorAdmissions`, `g_inParallelWorkerPhase`,
///      `updateAdmissionMap`.
inline void ExpressionAnalyzer::updateAdmissionMap3(StrSpan expr,
    Memory& memoryBlock,
    int maxAdmissionDepth,
    int maxSecondaryNumber,
    bool partOfRecursion) {
    // 1) Compute digit arguments of the expression (uses analyzer's anchor).
    // findDigitArgs is now 0% heap (sorted-unique StrSpan run == the former set
    // order); materialize the std::set in this still-heap caller. The compareSpans
    // order is preserved, which the downstream remaining-args run relies on -- that
    // run is deload-observable through the deferred-admission replay.
    StrSpan digitBuf[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    const int32_t digitN = findDigitArgs(expr, this->anchorInfo, digitBuf,
                                         ExecutionParameters::MAX_ADMISSION_REM_ARGS);
    // digitBuf is already sorted-unique (== the former std::set<std::string>
    // order), so it IS the digit-arg set -- no std::set materialized.

    // 2) Walk up the Memory chain to find the first block whose exprKey shares a digit arg
    Memory* tempMb = &memoryBlock;
    while (true) {
        // Args of this block's exprKey (zero-copy slices; no std::set -- the
        // intersection below is a pure membership test, order not observable).
        StrSpan keyArgs[ExecutionParameters::MAX_ARITY];
        const int32_t keyArgsN = getArgsSpans(tempMb->exprKeyView(), keyArgs,
                                              ExecutionParameters::MAX_ARITY);

        // Check intersection with the digit args (nested equalSpans scan; the
        // bool result is independent of iteration order).
        bool intersects = false;
        for (int32_t i = 0; i < digitN && !intersects; ++i) {
            for (int32_t j = 0; j < keyArgsN; ++j) {
                if (equalSpans(digitBuf[i], keyArgs[j])) { intersects = true; break; }
            }
        }

        if (intersects) {
            // Remaining args = digitArgs ∪ { last argument of expr }
            StrSpan exprArgs[ExecutionParameters::MAX_ARITY];
            const int32_t exprArgsN = getArgsSpans(StrSpan(expr), exprArgs,
                                                   ExecutionParameters::MAX_ARITY);
            if (exprArgsN > 0) {
                // remainingArgs = digitArgs U { last arg }, sorted-unique. Seed
                // with digitBuf (already sorted-unique == the former std::set
                // order), then sorted-insert-with-dedup the last arg. This run is
                // DELOAD-OBSERVABLE via the deferred-admission replay, so it MUST
                // reproduce the former std::set ascending order byte-for-byte
                // (compareSpans == std::string::compare). Every span aliases
                // `expr` (caller-stable) -- no interner mint touches them (I-3).
                StrSpan remRun[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                assert(digitN <= ExecutionParameters::MAX_ADMISSION_REM_ARGS
                    && "updateAdmissionMap3: digit-arg count exceeds cap");
                for (int32_t i = 0; i < digitN; ++i) remRun[i] = digitBuf[i];
                int32_t remRunN = digitN;
                {
                    const StrSpan last = exprArgs[exprArgsN - 1];
                    int32_t pos = 0;
                    bool dup = false;
                    while (pos < remRunN) {
                        const int c = compareSpans(remRun[pos], last);
                        if (c == 0) { dup = true; break; }
                        if (c > 0) break;
                        ++pos;
                    }
                    if (!dup) {
                        assert(remRunN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                            && "updateAdmissionMap3: remaining-arg count exceeds cap");
                        for (int32_t k = remRunN; k > pos; --k) remRun[k] = remRun[k - 1];
                        remRun[pos] = last;
                        ++remRunN;
                    }
                }

                // I-28 detect-and-defer: a STRICT-ANCESTOR seed (the climb left
                // the calling LB) reached from inside a parallel worker is a
                // cross-LB write — seal it onto the shared page set for the
                // single-threaded post-join drain. A self-seed (tempMb ==
                // &memoryBlock, the worker's own LB) and any seed outside the
                // parallel sweep (grid build, the drain's own re-entry) apply
                // inline. The mutex serializes ALL page-set writes (alloc /
                // appendRecord are not internally thread-safe).
                if (tempMb != &memoryBlock && g_inParallelWorkerPhase) {
                    std::lock_guard<std::mutex> lock(this->deferredAncestorAdmissionsMutex);
                    assert(this->deferredAncestorPages
                        && "updateAdmissionMap3: deferredAncestorPages not emplaced by proveKernel");
                    SealedPageSet& ps = *this->deferredAncestorPages;
                    // key = [expr] : one SealedString + a 1-element SealedSpan.
                    SealedString keyElems[1] = {
                        SealedString::copyFrom(ps, expr.ptr, expr.len) };
                    SealedSpan<SealedString> keySpan =
                        SealedSpan<SealedString>::copyFrom(ps, keyElems, 1);
                    // remainingArgs : seal each remRun element, then the run
                    // (byte-identical to the former sorted-unique set order).
                    SealedString remElems[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                    for (int32_t i = 0; i < remRunN; ++i)
                        remElems[i] = SealedString::copyFrom(ps, remRun[i].ptr, remRun[i].len);
                    SealedSpan<SealedString> remSpan =
                        SealedSpan<SealedString>::copyFrom(ps, remElems, remRunN);
                    ps.appendRecord(DeferredAncestorAdmission{ tempMb, keySpan,
                        remSpan, maxAdmissionDepth, maxSecondaryNumber,
                        partOfRecursion });
                }
                else {
                    // Inline: key = [expr], remainingArgs = the sorted-unique
                    // remRun. Both outlive the call (expr caller-stable; remRun a
                    // stack copy of spans over stable input).
                    StrSpan keyRun[1] = { StrSpan(expr) };
                    this->updateAdmissionMap(*tempMb,
                        keyRun, 1,
                        remRun, remRunN,
                        maxAdmissionDepth,
                        maxSecondaryNumber,
                        partOfRecursion,
                        StrSpan("main", 4));
                    // WAKE DOOR 3 (inline twin, D-194): an
                    // inline (single-threaded) admission write into tempMb — its
                    // own LB when self-seeding (harmless), or an ancestor on a
                    // non-parallel re-entry climb — is rule state that can fire
                    // next burst; keep the target swept.
                    tempMb->hasWork = true;
                }
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


/// @see Declaration in `prover.hpp` for the full contract.
int ExpressionAnalyzer::sealedStringRunLexCompare(
    const SealedSpan<SealedString>& a, const SealedSpan<SealedString>& b) {
    const int32_t na = a.size();
    const int32_t nb = b.size();
    const int32_t common = na < nb ? na : nb;
    for (int32_t k = 0; k < common; ++k) {
        const int c = compareSpans(StrSpan(a[k]), StrSpan(b[k]));
        if (c != 0) return c;
    }
    return na < nb ? -1 : (na > nb ? 1 : 0);
}

/// @see Declaration in `prover.hpp` for the full contract.
int ExpressionAnalyzer::deferredAncestorAdmissionLess(
    const DeferredAncestorAdmission& a, const DeferredAncestorAdmission& b) const {
    const int c0 = compareSpans(a.ancestor->exprKeyView(), b.ancestor->exprKeyView());
    if (c0 != 0) return c0;
    const int c1 = sealedStringRunLexCompare(a.key, b.key);
    if (c1 != 0) return c1;
    const int c2 = sealedStringRunLexCompare(a.remainingArgs, b.remainingArgs);
    if (c2 != 0) return c2;
    if (a.partOfRecursion != b.partOfRecursion)
        return a.partOfRecursion ? 1 : -1;   // false < true
    if (a.maxAdmissionDepth != b.maxAdmissionDepth)
        return a.maxAdmissionDepth < b.maxAdmissionDepth ? -1 : 1;
    if (a.maxSecondaryNumber != b.maxSecondaryNumber)
        return a.maxSecondaryNumber < b.maxSecondaryNumber ? -1 : 1;
    return 0;
}

/// @see Declaration in `prover.hpp` for the full contract.
void ExpressionAnalyzer::drainDeferredAncestorAdmissions() {
    // proveKernel emplaces the page set before phase-1 and this is its sole,
    // unconditionally-reached post-join call site — it is always engaged here.
    assert(deferredAncestorPages
        && "drainDeferredAncestorAdmissions: proveKernel must emplace the page set");
    SealedPageSet& ps = *deferredAncestorPages;
    if (ps.recordCount() == 0) { deferredAncestorPages.reset(); return; }

    // The drain runs single-threaded post-join (g_currentCoreId == -1 -> the
    // reserved last gen slot). Gather record pointers, then an index sort on the
    // deterministic 6-field chain (I-107). The refs / idx ride the gen-scratch
    // byte-bump + page tiers; the sealed strings ride deferredAncestorPages
    // (never crossed, I-124).
    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();
    DirtyState d = DirtyState::Clean;
    PagedVector<const DeferredAncestorAdmission*> refs(&gArena, &d);
    ps.forEachRecord<DeferredAncestorAdmission>(
        [&](const DeferredAncestorAdmission& r) { refs.push_back(&r); });

    const int32_t n = static_cast<int32_t>(refs.size());
    int32_t* idx = reinterpret_cast<int32_t*>(
        gArena.resolve(gArena.alloc(n * sizeof(int32_t), alignof(int32_t))));
    for (int32_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx, idx + n, [&](int32_t a, int32_t b) {
        return deferredAncestorAdmissionLess(*refs[static_cast<size_t>(a)],
                                             *refs[static_cast<size_t>(b)]) < 0;
    });

    for (int32_t oi = 0; oi < n; ++oi) {
        const DeferredAncestorAdmission& r = *refs[static_cast<size_t>(idx[oi])];
        // A permanently-deactivated ancestor has no next step to consume the
        // seed and must not be resurrected (I-102 / I-112) — skip, mirroring the
        // dischargedForever guard on the proven-head deposit drain.
        if (r.ancestor->dischargedForever) continue;
        // SEAM DOOR (D-196): the pager (a phase window or
        // the barrier seam window) may have evicted the ancestor — the
        // uniform handshake reloads it claim-correctly and holds WorkerOwned
        // across the admission write below; released after.
        steward->claimAndLoadForWork(*r.ancestor, /*phase=*/4,
                                     lbdeload::kDeloadDirectory);
        // SealedSpan<SealedString> -> StrSpan[] stack runs for the C2 signature.
        // The sealed bytes stay valid until freePages() below (updateAdmissionMap
        // mints the ANCESTOR's interners, never this page set; I-3).
        StrSpan keyRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        const int32_t kN = r.key.size();
        assert(kN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
        for (int32_t i = 0; i < kN; ++i) keyRun[i] = StrSpan(r.key[i]);
        StrSpan remRun[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
        const int32_t rN = r.remainingArgs.size();
        assert(rN <= ExecutionParameters::MAX_ADMISSION_REM_ARGS);
        for (int32_t i = 0; i < rN; ++i) remRun[i] = StrSpan(r.remainingArgs[i]);
        this->updateAdmissionMap(*r.ancestor, keyRun, kN, remRun, rN,
            r.maxAdmissionDepth, r.maxSecondaryNumber, r.partOfRecursion,
            StrSpan("main", 4));
        // WAKE DOOR 3 (D-194): this replays a cross-LB
        // admission seed into the ancestor's admission map (rule state, not a
        // mailbox) — it can enable a fresh firing next burst, so keep the
        // ancestor swept.
        r.ancestor->hasWork = true;
        r.ancestor->stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle),
            std::memory_order_release);
    }
    gArena.popTo(mark);
    ps.seal();
    ps.freePages();
    deferredAncestorPages.reset();
}

/// @see Declaration in `prover.hpp` for the full contract.
int ExpressionAnalyzer::updateGlobalDirectLess(const UpdateGlobalDirectRec& a,
                                               const UpdateGlobalDirectRec& b) {
    const int c = compareSpans(StrSpan(a.theorem), StrSpan(b.theorem));
    if (c != 0) return c;
    if (a.coreId != b.coreId) return a.coreId < b.coreId ? -1 : 1;
    return 0;
}

/// @see Declaration in `prover.hpp` for the full contract.
void ExpressionAnalyzer::drainUpdateGlobalDirect() {
    // proveKernel emplaces the page set before phase-1 and this is its sole,
    // unconditionally-reached post-join call site — always engaged here.
    assert(updateGlobalDirectPages
        && "drainUpdateGlobalDirect: proveKernel must emplace the page set");
    SealedPageSet& ps = *updateGlobalDirectPages;
    if (ps.recordCount() == 0) { updateGlobalDirectPages.reset(); return; }

    // Single-threaded post-join (g_currentCoreId == -1 -> reserved last gen
    // slot). Gather record pointers, index-sort on the (theorem bytes, coreId)
    // total order — the former std::sort over std::tuple<std::string,int>. The
    // refs / idx ride the gen-scratch tiers; the sealed strings ride
    // updateGlobalDirectPages (never crossed, I-124).
    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();
    DirtyState d = DirtyState::Clean;
    PagedVector<const UpdateGlobalDirectRec*> refs(&gArena, &d);
    ps.forEachRecord<UpdateGlobalDirectRec>(
        [&](const UpdateGlobalDirectRec& r) { refs.push_back(&r); });

    const int32_t n = static_cast<int32_t>(refs.size());
    int32_t* idx = reinterpret_cast<int32_t*>(
        gArena.resolve(gArena.alloc(n * sizeof(int32_t), alignof(int32_t))));
    for (int32_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx, idx + n, [&](int32_t a, int32_t b) {
        return updateGlobalDirectLess(*refs[static_cast<size_t>(a)],
                                      *refs[static_cast<size_t>(b)]) < 0;
    });

    for (int32_t oi = 0; oi < n; ++oi) {
        const UpdateGlobalDirectRec& r = *refs[static_cast<size_t>(idx[oi])];
        // Vacuous-premise emission gate: a theorem sealed by an LB whose
        // premise set turned out inconsistent (self-or-ancestor
        // mainContradiction) is vacuous — suppress it. The flags were set
        // in this iteration's worker phases, so the single-threaded drain
        // sees them; the parent walk is race-free here (the sanctioned
        // I-28 seam).
        assert(r.producer
            && "drainUpdateGlobalDirect: seal sites must carry the producer LB");
        if (hasContradictedAncestor(r.producer)) continue;
        // Materialize the std::string at the still-heap updateGlobalDirect edge
        // (the sink is not a statification target; the sealed bytes crossing the
        // join is the win). The sealed strings are live until freePages() below.
        updateGlobalDirect(StrSpan(r.theorem).toStdString(), r.coreId,
                           r.producer);

        // Disproof deposit: a primed __contradiction__ discharge emits a
        // theorem whose head is negate(seed) (I-165), so hand the seed to the
        // parent's disproved-goal inbox. The parent's own end-of-burst
        // drainDisprovedGoals probes it against the parent's MAIN goals —
        // a hit is a disproof (the verbatim twin fired), a miss is the
        // complement twin proving the goal (normal closure handles it) or an
        // already-closed goal; both are defined outcomes, so the deposit is
        // unconditional for primed producers. Single-threaded post-join seam;
        // the inbox rides the never-deloaded persistent pool (the I-108
        // pattern), so no claim is needed while the parent is cold, and the
        // hasWork wake satisfies I-153.
        if (r.producer->primedForContradiction) {
            Memory* parent = r.producer->parentMemory;
            assert(parent
                && "primed contradiction LB must have a parent LB");
            const StrSpan key = r.producer->exprKeyView();
            static const char CONTRA_PFX[] = "__contradiction__";
            const int32_t pfxLen =
                static_cast<int32_t>(sizeof(CONTRA_PFX) - 1);
            assert(key.len > pfxLen
                && std::memcmp(key.ptr, CONTRA_PFX,
                               static_cast<size_t>(pfxLen)) == 0
                && "primed LB exprKey must carry the __contradiction__ prefix");
            parent->pendingDisprovedGoals.mint(
                StrSpan(key.ptr + pfxLen, key.len - pfxLen));
            parent->hasWork = true;
        }
    }
    gArena.popTo(mark);
    ps.seal();
    ps.freePages();
    updateGlobalDirectPages.reset();
}

/// @brief End-of-burst drain of the disproved-goal inbox — erase a disproved
///        MAIN goal and every piece of integration machinery its preparation
///        spawned.
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full contract
/// (deposit protocol, the six cleanup steps, determinism, and the flagged
/// verbatim-match limitation). Implementation notes:
///
/// - Heap-free (Rule 28): seed snapshots ride the string-scratch tier, the
///   root-name / dead-compact scratch rides the gen-scratch tiers, all
///   reclaimed at exit.
/// - No NameMap mint occurs anywhere in the body — every probe is a
///   non-minting `lookup`, so `decodeView` / `decodeSubView` spans stay valid
///   across the whole drain (I-3 different-container rule; the erases mutate
///   other containers only).
/// - Scope wipes are DELEGATED: matched roots go onto `pendingWipeScopes`
///   (drained immediately after this function in `performElemPhase3`), so the
///   radical sweep itself stays single-sourced in `Memory::wipeSubtree`.
///
/// @param body The LB whose inbox is drained (resident — its own burst).
/// @invariant Seeds process in byte-lex order; every erase predicate is a
///            pure function of container content (I-84 determinism).
/// @see Memory::pendingDisprovedGoals, Memory::wipeSubtree,
///      ExpressionAnalyzer::splitSubproofPayload,
///      ExpressionAnalyzer::templateMatchesGoalWithMarkers.
void ExpressionAnalyzer::drainDisprovedGoals(Memory& body) {
    const int32_t seedN = body.pendingDisprovedGoals.count();
    if (seedN == 0) return;
    // Quiescence (D-194): the cleanup eradicates state; flag the mutation
    // directly rather than relying on a count diff.
    body.mutatedThisBurst = true;

    const unsigned gSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    const ArenaOffset gMark = gArena.cursor();
    const unsigned sSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(sSlot);
    ScratchScope sScope(sArena);

    // Snapshot the seed bytes onto the string tier (the inbox resets below so
    // wipe-time inserts elsewhere run against a fresh container), then sort
    // byte-lex for a deterministic processing order (I-84). The spans alias
    // sArena bytes under the function-level sScope — nothing rewinds sArena
    // below them (09c pitfall 4 discipline).
    StrSpan* seeds = reinterpret_cast<StrSpan*>(gArena.resolve(
        gArena.alloc(seedN * static_cast<int32_t>(sizeof(StrSpan)),
                     static_cast<int32_t>(alignof(StrSpan)))));
    for (int32_t i = 1; i <= seedN; ++i) {
        const StrSpan v = body.pendingDisprovedGoals.view(i);
        char* buf = sArena.allocBytes(v.len);
        std::memcpy(buf, v.ptr, static_cast<size_t>(v.len));
        seeds[i - 1] = StrSpan(buf, v.len);
    }
    body.pendingDisprovedGoals.resetToFresh();
    std::sort(seeds, seeds + seedN, [](const StrSpan& a, const StrSpan& b) {
        return compareSpans(a, b) < 0;
    });

    for (int32_t si = 0; si < seedN; ++si) {
        const StrSpan seed = seeds[si];
        const NameId goalId = body.nameMap.lookup(seed);
        if (goalId == 0) continue;   // never a goal here — complement twin
        const int64_t goalPk = packStatementKey(goalId, NameMap::MAIN_ID);
        if (body.intToBeProved.lookup(goalPk) == 0) continue;  // closed / twin

        // ---- 1. the goal row ----
        body.intToBeProved.eraseSet(goalPk);

        // The negated form of the seed — the disproof PRODUCT, which the
        // MAIN-keyed origin sweep below must never erase.
        StrSpan negSeed;
        if (seed.len > 0 && seed.ptr[0] == '!') {
            negSeed = StrSpan(seed.ptr + 1, seed.len - 1);
        } else {
            char* nb = sArena.allocBytes(seed.len + 1);
            nb[0] = '!';
            std::memcpy(nb + 1, seed.ptr, static_cast<size_t>(seed.len));
            negSeed = StrSpan(nb, seed.len + 1);
        }

        // ---- 2. MAIN-level scope roots that embed the goal ----
        // Subtree membership for the erases below is the TEXT gate
        // (equal-or-boundary-prefix against the root names): the origin /
        // cohort key spaces are originInterner / lbStateInterner, never the
        // NameMap, so a NameMap-id bitmap cannot serve them (I-91, I-93).
        //
        // A subproof root whose PRODUCT already closed (the bare compact /
        // OR signature is a REGISTERED statement at main) is SKIPPED
        // entirely: its scope was wiped by the success path at closure, the
        // product is a legitimate proven fact other proofs may cite, and its
        // preserved history rows are what the chapter walker reads — only
        // never-closed machinery is dead.
        DirtyState rootsDirty = DirtyState::Clean;
        PagedVector<StrSpan> rootNames(&gArena, &rootsDirty);
        static const char SENT_PFX[] =
            "product_of_hypo_disintegration_of_integration_goal_";
        const int32_t sentLen = static_cast<int32_t>(sizeof(SENT_PFX) - 1);
        static const char HYPO_TOK[] = "_hypo_";
        const int32_t hypoLen = static_cast<int32_t>(sizeof(HYPO_TOK) - 1);
        static const char ORINT_PFX[] = "orint_";
        const int32_t orintLen = static_cast<int32_t>(sizeof(ORINT_PFX) - 1);
        for (NameId id = 2; id <= body.nameMap.nameCount(); ++id) {
            if (body.nameMap.stackEmpty(id)) continue;
            if (body.nameMap.parentOf(id) != NameMap::MAIN_ID) continue;
            const StrSpan payload =
                body.nameMap.decodeSubView(body.nameMap.stackBack(id));
            bool matched = false;
            StrSpan gOut, bOut;
            if (splitSubproofPayload(payload, gOut, bOut)) {
                matched = equalSpans(gOut, seed);
                if (matched) {
                    // Derive the subproof's product: the bare compact
                    // (Case A) or the OR signature (Case OR branch).
                    StrSpan product;
                    if (bOut.len > 0 && bOut.ptr[0] == '(') {
                        product = bOut;
                    } else if (bOut.len > orintLen
                               && std::memcmp(bOut.ptr, ORINT_PFX,
                                              static_cast<size_t>(orintLen))
                                      == 0) {
                        int32_t sep = -1;
                        for (int32_t p = orintLen; p + 1 < bOut.len; ++p) {
                            if (bOut.ptr[p] == '_'
                                && bOut.ptr[p + 1] == '(') { sep = p; break; }
                        }
                        if (sep > orintLen)
                            product = StrSpan(bOut.ptr + orintLen,
                                              sep - orintLen);
                    }
                    if (product.len > 0) {
                        const StatementFlags* row = lookupStatementFlags(
                            body.intKnownStatements, body.nameMap, product,
                            StrSpan("main", 4));
                        if (row && row->registered) continue;  // closed — keep
                    }
                }
            } else if (payload.len == sentLen + seed.len
                       && std::memcmp(payload.ptr, SENT_PFX,
                                      static_cast<size_t>(sentLen)) == 0
                       && equalSpans(StrSpan(payload.ptr + sentLen, seed.len),
                                     seed)) {
                matched = true;   // the hypo sentinel of this goal
            } else if (payload.len > hypoLen + seed.len
                       && payload.ptr[0] == '_'
                       && equalSpans(StrSpan(payload.ptr + payload.len
                                                 - seed.len, seed.len),
                                     seed)
                       && std::memcmp(payload.ptr + payload.len - seed.len
                                          - hypoLen,
                                      HYPO_TOK,
                                      static_cast<size_t>(hypoLen)) == 0) {
                matched = true;   // the `_var…_hypo_<goal>` working scope
            }
            if (!matched) continue;
            body.pendingWipeScopes.mint(id);
            body.intValidityNamesToFilter.mint(id);
            rootNames.push_back(body.nameMap.decodeView(id));
        }

        // ---- 3. goal-template gates at MAIN ----
        const auto tmplMatches = [&](int32_t tid) -> bool {
            const StrSpan t = body.templateInterner.decodeView(tid);
            if (templateMatchesGoalWithMarkers(t, seed)) return true;
            StrSpan tg, tb;
            return splitSubproofPayload(t, tg, tb) && equalSpans(tg, seed);
        };
        body.integrationPrepared.eraseIf([&](int64_t k) {
            const StatementKey sk = Codec<StatementKey>::decode(k);
            return sk.validity == NameMap::MAIN_ID && tmplMatches(sk.orig);
        });
        body.integrationPreparedMarker.eraseIf([&](int64_t k) {
            const StatementKey sk = Codec<StatementKey>::decode(k);
            return sk.validity == NameMap::MAIN_ID && tmplMatches(sk.orig);
        });
        body.integrationStartIntMap.eraseIf([&](NameId tid) {
            return tmplMatches(tid);
        });

        // Origin keys pack BOTH halves in the originInterner space — NEVER
        // the NameMap (I-91) — so subtree membership must gate on the DECODED
        // validity name against the wiped root names (the expandedImplications
        // text-gate idiom), not on the NameMap bitmap.
        const StrSpan boundaryTok(
            NameMap::BOUNDARY_STR,
            static_cast<int32_t>(NameMap::BOUNDARY_LEN));
        const auto underWipedRoot = [&](StrSpan vName) -> bool {
            for (int32_t ri = 0; ri < rootNames.size(); ++ri) {
                const StrSpan rn = rootNames[ri];
                if (equalSpans(vName, rn)) return true;
                if (vName.len > rn.len + boundaryTok.len
                    && equalSpans(StrSpan(vName.ptr, rn.len), rn)
                    && equalSpans(StrSpan(vName.ptr + rn.len,
                                          boundaryTok.len),
                                  boundaryTok))
                    return true;
            }
            return false;
        };

        // ---- 4. exprOriginMap: wiped-subtree rows + MAIN goal rows ----
        // (The maintainer-approved deviation from wipeSubtree's I-44
        // preserve rule — disproof cleanup only; the success path keeps its
        // history untouched.)
        // MAIN-keyed rows erase on GOAL-TEXT containment, with the products
        // excepted: the negated fact `negate(goal)` and the anchor-rooted
        // theorem rows stay. Rows citing a dead subproof's bare compact but
        // NOT the goal stay too — the same compact text can belong to a
        // CLOSED subproof of another goal (per-goal duplication), whose main
        // statement and history are load-bearing for the chapter walker.
        body.exprOriginMap.eraseBlobIf([&](int64_t k) {
            StrSpan e, v;
            decodeOriginKeyView(k, body.originInterner, e, v);
            if (underWipedRoot(v)) return true;
            if (equalSpans(v, StrSpan("main", 4))
                && containsSpan(e, seed)
                && !equalSpans(e, negSeed)
                && !containsSpan(e, StrSpan("(Anchor", 7)))
                return true;
            return false;
        });

        // ---- 5. internal-mail origins at wiped scopes ----
        // (The statements are filtered by wipeSubtree itself right after this
        // drain; without the origins sweep the orphaned history runs would
        // outlive their statements in the discharge-surviving container.
        // Same originInterner key space as exprOriginMap — text gate.)
        body.sameIterationInternalMail.filterOrigins([&](int64_t k) {
            StrSpan e, v;
            decodeOriginKeyView(k, body.originInterner, e, v);
            return underWipedRoot(v);
        });

        // ---- 6. OR cohort bookkeeping under wiped scopes ----
        // Cohort identity is (parentValidity, orSignature) in the
        // lbStateInterner space (I-167) — the parent arrives as TEXT, so the
        // dead test is the equal-or-boundary-prefix gate against each wiped
        // root name (the wipeSubtree expandedImplications idiom).
        const auto cohortDead = [&](int32_t cohortId) -> bool {
            const LbStatePairKey pv =
                decodeOrCohortIds(body.lbStateInterner, cohortId);
            return underWipedRoot(body.lbStateInterner.decodeView(pv.high));
        };
        body.orDisjunctCount.eraseIf([&](int32_t cid) {
            return cohortDead(cid);
        });
        body.orBookkeeping.eraseSetIf([&](const LbStatePairKey& k) {
            return cohortDead(static_cast<int32_t>(k.low));
        });
    }
    gArena.popTo(gMark);
}


/// @brief End-of-burst drain of the dead OR-branch inbox — retire refuted
///        `_ordis_` branches, shrink their cohorts, re-check convergence.
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full contract
/// (staging protocol, the four retirement steps, the defined skips, and the
/// history-preservation rule). Implementation notes:
///
/// - Heap-free (Rule 28): the vid snapshot, the per-cohort ledgers, and the
///   `orBookkeeping` row snapshot ride the gen-scratch tier; the rebuilt
///   branch-validity strings ride the string-scratch tier; all reclaimed at
///   exit.
/// - The only NameMap writes happen inside the re-fired `ordisMerge` (both
///   encodes are lookup hits — the branch scope and the statement were
///   interned by their original deposits), so no sub-table mint invalidates
///   the held `decodeSubView` spans (I-3); `lbStateInterner.decodeView`
///   spans survive that interner's own mints (paged appends never relocate).
/// - The `orBookkeeping` surgery is erase-plus-reinsert through the two
///   existing doors (`eraseSetIf`, `insertSorted` with `DecodedIdLess`), so
///   the rebuilt runs keep the canonical decoded-lex order.
/// - Scope wipes are DELEGATED to `pendingWipeScopes` (drained immediately
///   after this function in `performElemPhase3`).
///
/// @param body The LB whose inbox is drained (resident — its own burst).
/// @invariant Staged vids process in decoded-name byte-lex order; every
///            erase predicate and rebuilt run is a pure function of
///            container content (I-84 determinism).
/// @see Memory::pendingDeadOrBranches, ordisMerge, mintOrCohortId,
///      decodeOrCohortIds.
void ExpressionAnalyzer::drainDeadOrBranches(Memory& body) {
    const int32_t seedN = body.pendingDeadOrBranches.count();
    if (seedN == 0) return;
    // Quiescence (D-194): the retirement eradicates state; flag the mutation
    // directly rather than relying on a count diff.
    body.mutatedThisBurst = true;

    const unsigned gSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    const ArenaOffset gMark = gArena.cursor();
    const unsigned sSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(sSlot);
    ScratchScope sScope(sArena);

    // Snapshot the staged vids, reset the inbox (a re-fired ordisMerge below
    // runs its probe against a fresh container), sort by decoded scope name
    // (the pendingWipeScopes drain idiom: the pod set dedups and the NameMap
    // is injective, so the sort is tie-free — I-84).
    NameId* vids = reinterpret_cast<NameId*>(gArena.resolve(
        gArena.alloc(seedN * static_cast<int32_t>(sizeof(NameId)),
                     alignof(NameId))));
    for (int32_t i = 1; i <= seedN; ++i)
        vids[i - 1] = body.pendingDeadOrBranches.decode(i);
    body.pendingDeadOrBranches.resetToFresh();
    std::sort(vids, vids + seedN, [&](NameId a, NameId b) {
        return compareSpans(body.nameMap.decodeView(a),
                            body.nameMap.decodeView(b)) < 0;
    });

    // ---- 1. per-seed: wipe queue + cohort/disjunct ledger ----
    DirtyState ledgerDirty = DirtyState::Clean;
    PagedVector<int32_t> deadCohortIds(&gArena, &ledgerDirty);   // per seed
    PagedVector<int32_t> deadDjIds(&gArena, &ledgerDirty);       // per seed
    for (int32_t si = 0; si < seedN; ++si) {
        const NameId vid = vids[si];
        StrSpan orSigV, branchBodyV;
        const OrScopeKind kind = classifyOrScopeView(body.nameMap, vid,
                                                     orSigV, branchBodyV);
        assert(kind == OrScopeKind::Disintegration
            && "pendingDeadOrBranches vid must be an _ordis_ branch scope");
        (void)kind;
        const StrSpan vName = body.nameMap.decodeView(vid);
        assert(!body.nameMap.stackEmpty(vid));
        const int32_t payloadLen = body.nameMap.decodeSubView(
            body.nameMap.stackBack(vid)).len;
        const StrSpan parentV(vName.ptr,
            vName.len - static_cast<int32_t>(NameMap::BOUNDARY_LEN)
                - payloadLen);
        const int32_t cohortId = mintOrCohortId(body.lbStateInterner,
            body.lbStateInterner.encode(parentV),
            body.lbStateInterner.encode(orSigV));
        if (body.orDisjunctCount.lookup(cohortId) == 0) continue;
        // Defined skip above: the cohort was retired wholesale by this same
        // seam's drainDisprovedGoals — its parent scope lies in a
        // disproof-wiped subtree, which covers this branch scope too.
        body.pendingWipeScopes.mint(vid);
        body.intValidityNamesToFilter.mint(vid);
        const int32_t deadDjId = body.lbStateInterner.encode(branchBodyV);
        deadCohortIds.push_back(cohortId);
        deadDjIds.push_back(deadDjId);

        // Record the retired disjunct with the SHALLOWEST scope where its
        // negation is `known` — the reductio ingredient a later
        // reduced-cohort `or convergence` row cites for this branch. The
        // walk re-runs the ordisMerge probe (the record is immortal, I-58,
        // so the staged branch is guaranteed a hit); preferring the
        // shallowest hit cites the fact where it genuinely lives (typically
        // main) rather than an inherited rewrite inside the dead branch.
        {
            assert(branchBodyV.len >= 2);
            const StrSpan deadAsm(branchBodyV.ptr + 1, branchBodyV.len - 2);
            NameId negId = 0;
            if (deadAsm.len > 0 && deadAsm.ptr[0] == '!') {
                negId = body.nameMap.lookup(
                    StrSpan(deadAsm.ptr + 1, deadAsm.len - 1));
            } else {
                char* nb = sArena.allocBytes(deadAsm.len + 1);
                nb[0] = '!';
                std::memcpy(nb + 1, deadAsm.ptr,
                            static_cast<std::size_t>(deadAsm.len));
                negId = body.nameMap.lookup(StrSpan(nb, deadAsm.len + 1));
            }
            assert(negId != 0
                && "staged dead branch lost its refutation interning");
            NameId refutScope = 0;
            for (NameId a = vid; a != 0; a = body.nameMap.parentOf(a)) {
                const StatementFlags* row = body.intKnownStatements.find(
                    StatementKey{ negId, a });
                if (row && row->known) refutScope = a;
                if (a == NameMap::MAIN_ID) break;
            }
            assert(refutScope != 0
                && "staged dead branch lost its refutation record");
            body.orRetiredDisjuncts.insertSorted(cohortId,
                packInt32Pair(deadDjId, refutScope));
        }
    }
    const int32_t deadN = deadCohortIds.size();
    if (deadN == 0) { gArena.popTo(gMark); return; }

    const auto isDeadCohort = [&](int32_t cid) -> bool {
        for (int32_t i = 0; i < deadN; ++i)
            if (deadCohortIds[i] == cid) return true;
        return false;
    };
    const auto isDeadDj = [&](int32_t cid, int32_t dj) -> bool {
        for (int32_t i = 0; i < deadN; ++i)
            if (deadCohortIds[i] == cid && deadDjIds[i] == dj) return true;
        return false;
    };

    // ---- 2. shrink each cohort's structural count ----
    // Zero survivors retires the cohort wholesale (count row erased); the
    // joint contradiction the fully-refuted disjunction implies is owned by
    // the K rules + contradiction machinery, never by branch retirement.
    for (int32_t i = 0; i < deadN; ++i) {
        const int32_t cid = deadCohortIds[i];
        bool firstOfCohort = true;
        for (int32_t j = 0; j < i; ++j)
            if (deadCohortIds[j] == cid) { firstOfCohort = false; break; }
        if (!firstOfCohort) continue;
        int32_t cohortDead = 0;
        for (int32_t j = 0; j < deadN; ++j)
            if (deadCohortIds[j] == cid) ++cohortDead;
        const int32_t cntRow = body.orDisjunctCount.lookup(cid);
        assert(cntRow != 0);
        const int oldCount = body.orDisjunctCount.valueAt(cntRow);
        const int newCount = oldCount - cohortDead;
        assert(newCount >= 0
            && "dead-branch retirement decremented an _ordis_ cohort below zero");
        if (newCount == 0) {
            // Every disjunct refuted: the disjunction's scope holds a joint
            // contradiction. The cohort retires wholesale — the count row
            // leaves the container here, and every bookkeeping run of the
            // cohort empties in the surgery below (all entries are dead).
            // The contradiction itself surfaces through the pre-existing
            // machinery: the parent-scope K mutual-exclusion rules hold all
            // the refutations, fire the last disjunct as a fact, and the
            // resulting statement/negation pair drives the primed discharge
            // (a reductio LB) or the vacuous-premise flag + global-theorem
            // reversion (a normal LB at main).
            body.orDisjunctCount.erase(cid);
        } else {
            body.orDisjunctCount.setValueAt(cntRow, newCount);
        }
    }

    // ---- 3. orBookkeeping surgery: drop dead disjunct entries ----
    // Snapshot every row of a dead cohort (key + surviving run), erase them
    // all, reinsert the rows that keep at least one survivor. Required for
    // soundness: a stale dead entry inflates the run length against the
    // reduced count, so a later test could fire a convergence no surviving
    // branch ever derived.
    PagedVector<LbStatePairKey> rowKeys(&gArena, &ledgerDirty);
    PagedVector<int32_t> rowStarts(&gArena, &ledgerDirty);
    PagedVector<int32_t> rowSurvivors(&gArena, &ledgerDirty);
    const int32_t obN = body.orBookkeeping.count();
    for (int32_t id = 1; id <= obN; ++id) {
        const LbStatePairKey key = Codec<LbStatePairKey>::decode(
            body.orBookkeeping.keyAt(id));
        const int32_t cid = static_cast<int32_t>(key.low);
        if (!isDeadCohort(cid)) continue;
        rowKeys.push_back(key);
        rowStarts.push_back(rowSurvivors.size());
        const int32_t rl = body.orBookkeeping.runLen(id);
        for (int32_t j = 0; j < rl; ++j) {
            const int32_t dj = body.orBookkeeping.valueAt(id, j);
            if (!isDeadDj(cid, dj)) rowSurvivors.push_back(dj);
        }
    }
    rowStarts.push_back(rowSurvivors.size());
    body.orBookkeeping.eraseSetIf([&](const LbStatePairKey& k) {
        return isDeadCohort(static_cast<int32_t>(k.low));
    });
    for (int32_t r = 0; r < rowKeys.size(); ++r) {
        for (int32_t j = rowStarts[r]; j < rowStarts[r + 1]; ++j) {
            body.orBookkeeping.insertSorted(rowKeys[r], rowSurvivors[j],
                DecodedIdLess{ &body.lbStateInterner });
        }
    }

    // ---- 4. convergence re-check at the reduced count ----
    // A surviving row whose run already reaches the new count re-fires
    // through ordisMerge — the canonical promotion path. The deposit spans
    // (expression + rebuilt branch validity) alias lbStateInterner cold bytes
    // and the string-scratch arena respectively, both stable across the
    // re-fire's mints. A missing levels row is the already-converged defined
    // skip (the expression was promoted and collapsed from its branches by an
    // earlier full-cohort convergence).
    const StrSpan BOUNDARY_ORDIS("_boundary_ordis_", 16);
    for (int32_t r = 0; r < rowKeys.size(); ++r) {
        const int32_t survN = rowStarts[r + 1] - rowStarts[r];
        if (survN == 0) continue;
        const int32_t cid = static_cast<int32_t>(rowKeys[r].low);
        const int* cnt = body.orDisjunctCount.find(cid);
        assert(cnt != nullptr);
        if (survN < *cnt) continue;
        const StrSpan exprV = body.lbStateInterner.decodeView(
            static_cast<int32_t>(rowKeys[r].high));
        const LbStatePairKey cohort =
            decodeOrCohortIds(body.lbStateInterner, cid);
        const StrSpan parentV = body.lbStateInterner.decodeView(cohort.high);
        const StrSpan sigV = body.lbStateInterner.decodeView(cohort.low);
        const StrSpan djV = body.lbStateInterner.decodeView(
            rowSurvivors[rowStarts[r]]);
        const int32_t bvLen =
            parentV.len + BOUNDARY_ORDIS.len + sigV.len + 1 + djV.len;
        char* bvBuf = sArena.allocBytes(bvLen);
        {
            int32_t at = 0;
            std::memcpy(bvBuf + at, parentV.ptr,
                static_cast<std::size_t>(parentV.len));
            at += parentV.len;
            std::memcpy(bvBuf + at, BOUNDARY_ORDIS.ptr,
                static_cast<std::size_t>(BOUNDARY_ORDIS.len));
            at += BOUNDARY_ORDIS.len;
            std::memcpy(bvBuf + at, sigV.ptr,
                static_cast<std::size_t>(sigV.len));
            at += sigV.len;
            bvBuf[at++] = '_';
            std::memcpy(bvBuf + at, djV.ptr,
                static_cast<std::size_t>(djV.len));
            at += djV.len;
            assert(at == bvLen);
        }
        const StrSpan branchV(bvBuf, bvLen);
        const int32_t lvlsId = lookupStatementLevels(
            body.intStatementLevelsMap, body.nameMap, exprV, branchV);
        if (lvlsId == 0) continue;
        int lvRun[256];
        const int32_t lvN = coldIntRunAt(
            body.intStatementLevelsMap, lvlsId, lvRun, 256);
        ordisMerge(exprV, branchV, lvRun, lvN, body);
    }
    gArena.popTo(gMark);
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
/// **Per [OPEN-MPL-1](../../docs/agentic_swdd/SwDD.md#remaining-open) (resolved
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


// Turn optimizations OFF for just this section
//#pragma optimize("", off)
/// @brief Predicate: is the given expression allowed to be mailed to
/// child LBs from this one?
///
/// @details
/// Mail-out gating per [I-26](../../docs/agentic_swdd/30_invariants.md#i-26):
/// statements/implications mailed out must be MAIN-validity-only;
/// the exprOriginMap mirror is ALL-SCOPES. This function enforces
/// the validity-side gate by reading the expression's scope and the
/// LB's `canBeSentIds` / `canBeSentMarkerIds` filters. Returns
/// false if the expression carries a non-`"main"` scope on the
/// implication/statement channel.
///
/// @param expression  Canonical MPL expression text (span over a
///                    caller-stable buffer; the interior is mint-free so the
///                    span cannot dangle).
/// @param body        Owning LB whose `canBeSentIds` defines the
///                    mail-eligibility filters.
/// @return True iff the expression is mailable on the routing channel.
/// @invariant [I-26](../../docs/agentic_swdd/30_invariants.md#i-26).
bool ExpressionAnalyzer::allowedForMail(StrSpan expression, const Memory& body) {
    // Single pre-pass: existence + single-distinct collection of the
    // int_lev_\d+_\d+ tokens. This folds the former separate regex existence
    // gate and the regex-iterator std::set collection into one
    // scan — verdict-identical because both are pure reads with zero mints and
    // zero side effects between them (the memo probe below inserted nothing).
    // theVar is the sole distinct token's span (into `expression`) on verdict 1.
    StrSpan theVar;
    const int intLevVerdict = scanSingleDistinctIntLev(expression, theVar);
    if (intLevVerdict == 0) {
        return true;
    }

    // Non-minting probe: a never-interned expression cannot carry a
    // memo entry (inserts encode at single-threaded sites).
    {
        const NameId exprId = body.nameMap.lookup(expression);
        if (exprId != 0 && body.canBeSentIds.contains(exprId)) {
            return true;
        }
    }

    if (intLevVerdict != 1) {
        return false;
    }

    if (expression.len >= 2 && expression.ptr[0] == '(' && expression.ptr[1] == '>') {
        return false;
    }

    // Check if replaced with marker it is included in Memory::canBeSentMarkerIds.
    // Build the marker form on the per-slot string-scratch arena (fresh bytes,
    // byte-exact twin of ce::replaceKeysInString), held as a NAMED ScratchString
    // so its span stays live to the lookup. This path is mint-free (I-3), so
    // theVar / expression spans cannot dangle.
    const unsigned strSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(strSlot);
    ScratchScope scope(strArena);
    const StrReplacement pair[1] = { { theVar, StrSpan("marker", 6) } };
    const ScratchString markerExpr = replaceKeysScratch(strArena, expression, pair, 1);

    const NameId markerId = body.nameMap.lookup(StrSpan(markerExpr));
    if (markerId != 0 && body.canBeSentMarkerIds.contains(markerId)) {
        const LogicalEntity* e = compiledEntity(extractExpressionSpan(expression));
        assert(e && "allowedForMail: core expression must be compiled");
        if (!equalSpans(StrSpan(e->category), StrSpan("atomic", 6))) {
            return true;
        }
    }

    return false;
}

/// @brief Collect the `u_`-prefixed "remaining" argument names of an
///        expression, with the `u_` prefix stripped, as zero-copy slices.
///
/// @details Scans every `[...]` argument block NOT immediately preceded by `>`
///          (a `>[...]` block is the forbidden zone for `u_` and is only
///          asserted clean). Each comma-separated, trimmed token beginning with
///          `u_` yields one output slice `token[2:]` — a `StrSpan` into the
///          caller's `str` bytes, so the slices live exactly as long as `str`.
///          Duplicates are emitted in scan order; the caller dedups (typically
///          by inserting into a `std::set<std::string>`), which is why this
///          routine owns no container and is 0% heap. The lexical scan is the
///          byte-exact twin of the former `std::sregex_iterator` over
///          `\[([^\]]*)\]` (see `extract_remaining_args_matches_regex_oracle`),
///          and the forbidden-zone / additional-`u_` invariants are asserted
///          exactly as before.
///
/// @param str Expression to scan (borrowed; slices point into it).
/// @param out Caller-provided array receiving the `u_`-stripped slices.
/// @param cap Capacity of `out`; overflow asserts (Rule 19 tripwire).
/// @return Number of slices written to `out[0..return)` (duplicates possible).
///
/// @invariant No token inside a `>[...]` block starts with `u_`; a `u_` token
///            contains no further `u_` substring — both asserted.
/// @see The caller `addExprToMemoryBlock` builds the deduped `std::set` for
///      `addToHashMemory`.
int32_t ExpressionAnalyzer::extractRemainingArgs(StrSpan str, StrSpan* out, int32_t cap) {
    int32_t count = 0;
    const StrSpan s = str;
    const int32_t n = s.len;

    // Byte-exact trim twin of ce::trimCopy: the IDENTICAL
    // std::isspace(static_cast<unsigned char>(c)) predicate on both ends, so it
    // matches trimCopy byte-for-byte under any locale (MPL is space-free — this
    // trims nothing in practice — but byte-fidelity keeps the twin honest).
    const auto trimSpan = [](StrSpan t) -> StrSpan {
        int32_t i = 0;
        while (i < t.len && std::isspace(static_cast<unsigned char>(t.ptr[i]))) ++i;
        int32_t j = t.len;
        while (j > i && std::isspace(static_cast<unsigned char>(t.ptr[j - 1]))) --j;
        return StrSpan(t.ptr + i, j - i);
    };

    // 1. Check >[...] blocks (Strict Forbidden Zone for "u_"). Lexical twin of
    //    the unanchored iterator over >\[([^\]]*)\]: an attempt needs "> [" at
    //    pos, the group runs to the FIRST ']' at/after pos+2, resume after ']';
    //    a failed start advances one byte; no ']' at/after an attempt means no
    //    match here NOR at any later attempt (\] can never match), so terminate.
    {
        int32_t pos = 0;
        while (pos + 1 < n) {
            if (s.ptr[pos] != '>' || s.ptr[pos + 1] != '[') { ++pos; continue; }
            int32_t close = pos + 2;
            while (close < n && s.ptr[close] != ']') ++close;
            if (close >= n) break;   // no ']' here nor at any later attempt
            // group = [pos+2, close): comma-split, trim, assert no u_.
            int32_t start = pos + 2;
            for (int32_t p = pos + 2; p <= close; ++p) {
                if (p == close || s.ptr[p] == ',') {
                    const StrSpan token = trimSpan(StrSpan(s.ptr + start, p - start));
                    if (token.len > 0) {
                        const bool hasUPrefix = token.len >= 2
                            && token.ptr[0] == 'u' && token.ptr[1] == '_';
                        assert(!hasUPrefix && "Violation: Argument starting with 'u_' found inside >[...] block");
                        (void)hasUPrefix;
                    }
                    start = p + 1;
                }
            }
            pos = close + 1;
        }
    }

    // 2. Collect "u_" args from [...] blocks (NOT preceded by '>'). Lexical twin
    //    of the unanchored iterator over \[([^\]]*)\]: an attempt needs '[' at
    //    pos, the group runs to the first ']', resume after ']' ALWAYS, but the
    //    group is SKIPPED (not the consume) when the '[' is preceded by '>' —
    //    the old loop's `continue`; a naive skip-and-rescan of the interior
    //    would diverge (the "[a>[b]c]" trap).
    {
        int32_t pos = 0;
        while (pos < n) {
            if (s.ptr[pos] != '[') { ++pos; continue; }
            int32_t close = pos + 1;
            while (close < n && s.ptr[close] != ']') ++close;
            if (close >= n) break;   // no ']' here nor at any later attempt
            const bool skip = (pos > 0 && s.ptr[pos - 1] == '>');
            if (!skip) {
                int32_t start = pos + 1;
                for (int32_t p = pos + 1; p <= close; ++p) {
                    if (p == close || s.ptr[p] == ',') {
                        const StrSpan token = trimSpan(StrSpan(s.ptr + start, p - start));
                        if (token.len >= 2 && token.ptr[0] == 'u' && token.ptr[1] == '_') {
                            // uCount = non-overlapping "u_" occurrences, step +2
                            // (replicating the old `p = find("u_", p); p += 2`).
                            int uCount = 0;
                            int32_t q = 0;
                            while (q + 1 < token.len) {
                                if (token.ptr[q] == 'u' && token.ptr[q + 1] == '_') {
                                    ++uCount; q += 2;
                                } else {
                                    ++q;
                                }
                            }
                            assert(uCount == 1 && "Argument starting with 'u_' must not contain additional 'u_' substrings");
                            (void)uCount;
                            // Emit the "u_"-stripped slice (a view into `str`;
                            // token == "u_" emits the empty slice, the substr(2)
                            // twin). The caller dedups/sorts (e.g. into a set).
                            assert(count < cap && "extractRemainingArgs: out capacity exceeded");
                            out[count++] = StrSpan(token.ptr + 2, token.len - 2);
                        }
                        start = p + 1;
                    }
                }
            }
            pos = close + 1;
        }
    }

    return count;
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
        const LogicalEntity* le = compiledEntity(
            extractExpressionUniversalSpan(StrSpan(std::get<0>(originalChain[i]))));
        if (le && !le->definedSet.empty()) {
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

        // D. Verify Logic: Bound Variable == Set Argument
        const LogicalEntity* le = compiledEntity(
            extractExpressionUniversalSpan(StrSpan(lastDefExpr)));
        if (le == nullptr) continue;

        // Find the expected argument position from the signature
        std::vector<std::string> sigArgs = ce::getArgs(le->signature);
        auto itSig = std::find(sigArgs.begin(), sigArgs.end(), le->definedSet);
        if (itSig == sigArgs.end()) continue;

        size_t argIndex = std::distance(sigArgs.begin(), itSig);
        std::vector<std::string> actualArgs = ce::getArgs(lastDefExpr);
        if (argIndex >= actualArgs.size()) continue;

        std::string expectedArg = actualArgs[argIndex];

        // CHECK: Is this definition acting as the peeling layer?
        //
        // Pre-unification this was `boundVars.size() == 1 && boundVars[0] ==
        // expectedArg`, reading the reconstructed last-link >[...]. Under the
        // unified binder rule (D-75) every
        // non-u_ variable is bound, so that cardinality no longer encodes the
        // peeling-layer condition. Re-express the SAME predicate directly from
        // originalChain — exactly the set the old sparse reconstructImplication
        // placed at the targetDef (last) link: a non-u_ arg of the targetDef
        // that occurs >=2x across (all premises + head) and appears in NO
        // other premise (so its first occurrence in the [others..., targetDef,
        // head] order was the targetDef link). This keeps the reformulation
        // set byte-identical; only the emitted theorem's outer >[...] widen
        // (DoD-covered). The inner existence binder stays exactly the single
        // peeled set-argument (see below).
        std::map<std::string, int> reformOcc;
        auto reformCount = [&](const std::string& e) {
            for (const std::string& a : ce::getArgs(e)) {
                if (a.size() >= 2 && a[0] == 'u' && a[1] == '_') continue;
                reformOcc[a]++;
            }
        };
        for (size_t ci = 0; ci < originalChain.size(); ++ci)
            reformCount(std::get<0>(originalChain[ci]));
        reformCount(head);

        std::set<std::string> reformOtherArgs;
        for (size_t ci = 0; ci < originalChain.size(); ++ci) {
            if (ci == targetDefIdx) continue;
            for (const std::string& a : ce::getArgs(std::get<0>(originalChain[ci])))
                reformOtherArgs.insert(a);
        }

        std::set<std::string> lastLinkBound;
        for (const std::string& a : actualArgs) {
            if (a.size() >= 2 && a[0] == 'u' && a[1] == '_') continue;
            if (reformOcc[a] >= 2 && reformOtherArgs.find(a) == reformOtherArgs.end())
                lastLinkBound.insert(a);
        }

        if (lastLinkBound.size() == 1 && *lastLinkBound.begin() == expectedArg) {

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

            // Build the Special Innermost Tail.
            // The gate guarantees the peeling set is exactly {expectedArg}, so
            // the negated-existence binds precisely the Definition's
            // set-argument — byte-identical to the pre-unification
            // `boundVars[0]`. This inner existential binder is NOT subject to
            // the unified "bind all non-u_" rule (that rule governs the outer
            // universal-implication wrappers, which DO widen via tempChain);
            // the existential's variable is fixed by the reformulation itself.
            std::string varsStr = expectedArg;

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
/// @param levels       Ascending-unique level run — LB levels at which the
///                     equality is admissible (caller-owned stack/arena
///                     memory, never a cold-column pointer).
/// @param levelCount   The level count (`>= 0`; 0 with null = empty).
/// @param ...          Routing parameters.
/// @invariant I-9 — mirror guard.
/// @invariant I-33 — class-merge cross-vN preconditions.
/// @see `addNegatedEquality` — negated counterpart.
void ExpressionAnalyzer::addEquality(StrSpan expr,
    Memory& memoryBlock,
    bool local,
    const int* levels, int32_t levelCount,
    const TransientOrigin& origin,
    StrSpan validityName)
{


    const StatementFlags* eqRow = lookupStatementFlags(
        memoryBlock.intKnownStatements, memoryBlock.nameMap, expr, validityName);
    if (!(eqRow && eqRow->registered))
    {
        // 1. Register the original equality
        const int64_t pkEq = packStatementKey(
            memoryBlock.nameMap.encode(expr),
            memoryBlock.nameMap.encode(validityName));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkEq, levels, levels + levelCount);
        upsertStatementKey(memoryBlock.intKnownStatements, pkEq,
            local, /*registered=*/true, /*known=*/true);

        if (parameters.trackHistory && origin.present) {
            assert(origin.tag != OriginTag::COUNT);
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, expr, StrSpan(validityName), origin.tag, origin.deps, origin.depN, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            // mailOut.exprOriginMap addOrigin is done in `fillMailOut`.
        }

        { IntEncodedExpr ie = encodeExpression(expr, StrSpan(validityName), memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);

        if (local) {
            memoryBlock.intLocalEncodedStatementsSet.mint(pkEq);
            memoryBlock.intLocalEncodedStatements.push_back(ie);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
            // mailOut.statements + mailOut.exprOriginMap are written in
            // `fillMailOut`.
        }
        } // close IntEncodedExpr ie scope

        // 2. Register the MIRRORED equality
        StrSpan argSpans[ExecutionParameters::MAX_ARITY];
        const int32_t argN = getArgsSpans(expr, argSpans,
                                          ExecutionParameters::MAX_ARITY);
        assert(argN == 2);

        if (!equalSpans(argSpans[0], argSpans[1])) {
        // Boundary string: the mirror is registered (encodeExpression /
        // intKnownStatements) and origin-tracked. Built "(=[a1,a0])" on the
        // per-slot string-scratch arena; every consumer (lookupStatementFlags /
        // nameMap.encode / encodeExpression / addOriginEncoded) mints into its
        // own cold pool, not this arena, and argSpans alias the caller-stable
        // expr, so the mirror span stays live for the whole block (I-3).
        const unsigned mSlot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : scratchArenas().slotCount() - 1;
        ScratchArena& mArena = scratchArenas().forSlot(mSlot);
        ScratchScope mScope(mArena);
        const int32_t mirrorLen = 3 + argSpans[1].len + 1 + argSpans[0].len + 2;
        char* mBuf = mArena.allocBytes(mirrorLen);
        int32_t mAt = 0;
        mBuf[mAt++] = '('; mBuf[mAt++] = '='; mBuf[mAt++] = '[';
        std::memcpy(mBuf + mAt, argSpans[1].ptr,
                    static_cast<std::size_t>(argSpans[1].len));
        mAt += argSpans[1].len;
        mBuf[mAt++] = ',';
        std::memcpy(mBuf + mAt, argSpans[0].ptr,
                    static_cast<std::size_t>(argSpans[0].len));
        mAt += argSpans[0].len;
        mBuf[mAt++] = ']'; mBuf[mAt++] = ')';
        assert(mAt == mirrorLen);
        const StrSpan mirrored(mBuf, mirrorLen);
        const StrSpan validitySpan(validityName);

        const StatementFlags* mirrorRow = lookupStatementFlags(
            memoryBlock.intKnownStatements, memoryBlock.nameMap, mirrored, validitySpan);
        assert(!(mirrorRow && mirrorRow->registered));
        (void)mirrorRow;

        const int64_t pkEqM = packStatementKey(
            memoryBlock.nameMap.encode(mirrored),
            memoryBlock.nameMap.encode(validitySpan));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkEqM, levels, levels + levelCount);
        upsertStatementKey(memoryBlock.intKnownStatements, pkEqM,
            local, /*registered=*/true, /*known=*/true);

        if (parameters.trackHistory) {
            // symmetry-of-equality mirror origin: one antecedent = the
            // original equality (expr, validityName), spanned in place.
            const OriginDep mirroredDeps[1] = {
                { expr, StrSpan(validityName) } };
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, mirrored, validitySpan, OriginTag::symmetryOfEquality, mirroredDeps, 1, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            // mailOut.exprOriginMap addOrigin for the mirror is done in
            // `fillMailOut`.
        }

        { IntEncodedExpr ieM = encodeExpression(mirrored, validitySpan, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ieM);

        if (local) {
            memoryBlock.intLocalEncodedStatementsSet.mint(pkEqM);
            memoryBlock.intLocalEncodedStatements.push_back(ieM);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ieM);

            // mailOut.statements for the mirror are written in
            // `fillMailOut`. The mirror's intLocalEncodedStatementsDelta
            // push above ensures fillMailOut picks it up.
        }
        } // close IntEncodedExpr ieM scope
        } // end if args[0] != args[1]
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
/// @param levels       Ascending-unique level run — LB levels at which the
///                     negated equality is admissible (caller-owned).
/// @param levelCount   The level count (`>= 0`; 0 with null = empty).
/// @param ...          Routing parameters.
/// @invariant I-12 — one-sided expansion via equivalence classes; no
///            symmetric cross-product.
/// @see `addEquality` — positive counterpart.
/// @see `applyEquivalenceClassToNegatedEquality` — class-driven sibling
///      rewrite.
void ExpressionAnalyzer::addNegatedEquality(StrSpan expr,
    Memory& memoryBlock,
    bool local,
    const int* levels, int32_t levelCount,
    const TransientOrigin& origin,
    StrSpan validityName)
{
    // Mirror of addEquality for negated form !(=[a,b]).
    // Pair invariant: original and mirror are registered together or not at all.
    // Sole gateway — any external path that registers a negated equality
    // without passing through here breaks the invariant and will trip the
    // mirror-absence assert below.

    const StatementFlags* negRow = lookupStatementFlags(
        memoryBlock.intKnownStatements, memoryBlock.nameMap, expr, validityName);
    if (!(negRow && negRow->registered))
    {
        // 1. Register the original negated equality
        const int64_t pkNeg = packStatementKey(
            memoryBlock.nameMap.encode(expr),
            memoryBlock.nameMap.encode(validityName));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkNeg, levels, levels + levelCount);
        upsertStatementKey(memoryBlock.intKnownStatements, pkNeg,
            local, /*registered=*/true, /*known=*/true);

        if (parameters.trackHistory && origin.present) {
            assert(origin.tag != OriginTag::COUNT);
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, expr, StrSpan(validityName), origin.tag, origin.deps, origin.depN, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            // mailOut.exprOriginMap addOrigin is done in `fillMailOut`.
        }

        { IntEncodedExpr ie = encodeExpression(expr, StrSpan(validityName), memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);

        if (local) {
            memoryBlock.intLocalEncodedStatementsSet.mint(pkNeg);
            memoryBlock.intLocalEncodedStatements.push_back(ie);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
            // mailOut.statements + mailOut.exprOriginMap are written in
            // `fillMailOut`.
        }
        } // close IntEncodedExpr ie scope

        // 2. Register the MIRRORED negated equality
        StrSpan argSpans[ExecutionParameters::MAX_ARITY];
        const int32_t argN = getArgsSpans(expr, argSpans,
                                          ExecutionParameters::MAX_ARITY);
        assert(argN == 2);

        if (!equalSpans(argSpans[0], argSpans[1])) {
        // Boundary string: the mirror is registered (encodeExpression /
        // intKnownStatements) and origin-tracked. Built "!(=[a1,a0])" on the
        // per-slot string-scratch arena; every consumer (lookupStatementFlags /
        // nameMap.encode / encodeExpression / addOriginEncoded) mints into its
        // own cold pool, not this arena, and argSpans alias the caller-stable
        // expr, so the mirror span stays live for the whole block (I-3).
        const unsigned mSlot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : scratchArenas().slotCount() - 1;
        ScratchArena& mArena = scratchArenas().forSlot(mSlot);
        ScratchScope mScope(mArena);
        const int32_t mirrorLen = 4 + argSpans[1].len + 1 + argSpans[0].len + 2;
        char* mBuf = mArena.allocBytes(mirrorLen);
        int32_t mAt = 0;
        mBuf[mAt++] = '!'; mBuf[mAt++] = '('; mBuf[mAt++] = '='; mBuf[mAt++] = '[';
        std::memcpy(mBuf + mAt, argSpans[1].ptr,
                    static_cast<std::size_t>(argSpans[1].len));
        mAt += argSpans[1].len;
        mBuf[mAt++] = ',';
        std::memcpy(mBuf + mAt, argSpans[0].ptr,
                    static_cast<std::size_t>(argSpans[0].len));
        mAt += argSpans[0].len;
        mBuf[mAt++] = ']'; mBuf[mAt++] = ')';
        assert(mAt == mirrorLen);
        const StrSpan mirrored(mBuf, mirrorLen);
        const StrSpan validitySpan(validityName);

        const StatementFlags* negMirrorRow = lookupStatementFlags(
            memoryBlock.intKnownStatements, memoryBlock.nameMap, mirrored, validitySpan);
        assert(!(negMirrorRow && negMirrorRow->registered));
        (void)negMirrorRow;

        const int64_t pkNegM = packStatementKey(
            memoryBlock.nameMap.encode(mirrored),
            memoryBlock.nameMap.encode(validitySpan));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkNegM, levels, levels + levelCount);
        upsertStatementKey(memoryBlock.intKnownStatements, pkNegM,
            local, /*registered=*/true, /*known=*/true);

        if (parameters.trackHistory) {
            // symmetry-of-inequality mirror origin: one antecedent = the
            // original inequality (expr, validityName), spanned in place.
            const OriginDep mirroredDeps[1] = {
                { expr, StrSpan(validityName) } };
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, mirrored, validitySpan, OriginTag::symmetryOfInequality, mirroredDeps, 1, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            // mailOut.exprOriginMap addOrigin for the mirror is done in
            // `fillMailOut`.
        }

        { IntEncodedExpr ieM = encodeExpression(mirrored, validitySpan, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ieM);

        if (local) {
            memoryBlock.intLocalEncodedStatementsSet.mint(pkNegM);
            memoryBlock.intLocalEncodedStatements.push_back(ieM);
            memoryBlock.intLocalEncodedStatementsDelta.push_back(ieM);

            // mailOut.statements for the mirror are written in
            // `fillMailOut`. The mirror's intLocalEncodedStatementsDelta
            // push above ensures fillMailOut picks it up.
        }
        } // close IntEncodedExpr ieM scope
        } // end if args[0] != args[1]
    }
}

//#pragma optimize("", off)

void ExpressionAnalyzer::checkNecessityForEquality(StrSpan inputExprStr, Memory& mb, StrSpan validityName) {

    if (parameters.ban_disintegration) return;

    // Use special function to make all args "u_" (D7). ONE ScratchScope spans the
    // whole function; genericInput's name/args are read throughout — held named, no
    // rewind inside the chain loop (all fresh bytes byte-bump forward, reclaimed at
    // exit; prepareIntegration's own nested scopes rewind only their allocations).
    const unsigned neSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(neSlot);
    ScratchScope sc(sArena);
    const ScratchString genericInput = prefixArgumentsWithUScratch(sArena, StrSpan(inputExprStr));

    // Parse the input expression string (D8).
    const StrSpan inputName = extractExpressionSpan(StrSpan(genericInput));
    StrSpan inputArgs[ExecutionParameters::MAX_ARITY];
    const int32_t inputArgN = getArgsSpans(StrSpan(genericInput), inputArgs,
                                           ExecutionParameters::MAX_ARITY);

    // Iterate all original implications (D9): the shared decoded-lex index over the
    // originals rule registry (byte-bump int32[] on genScratchArenas) reproduces the
    // former originalChains snapshot + std::sort order; the loop writes
    // (prepareIntegration per matched premise), so order is observable (I-84). Each
    // chain's rule ids are read zero-copy (IdVecKeyView) into a stack buffer, then
    // copyFrom'd onto sArena (0% heap; the ids extracted BEFORE any originals mint).
    const int32_t chainCount = mb.overallHashMemory.originals.count();
    const unsigned ocSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& ocArena = genScratchArenas().forSlot(ocSlot);
    ScratchScope ocScope(ocArena);
    int32_t* chainIdx = (chainCount == 0) ? nullptr
        : reinterpret_cast<int32_t*>(ocArena.resolve(ocArena.alloc(
              chainCount * static_cast<int32_t>(sizeof(int32_t)),
              static_cast<int32_t>(alignof(int32_t)))));
    const int32_t chainN = sortOriginalChainIndex(mb, chainIdx, chainCount);
    for (int32_t cix = 0; cix < chainN; ++cix) {
        // Per-cix scope (mirrors the former chain.clear()): chainSpans + the
        // premise-loop temporaries are freed each iteration; genericInput /
        // inputArgs (below the mark) survive.
        ScratchScope cixScope(sArena);
        // Read the chain key's rule ids ZERO-COPY into a stack buffer BEFORE any
        // mint (prepareIntegration -> addToHashMemory mints `originals`, the SAME
        // set, so a held keyAt span would dangle -- I-3).
        int32_t chainIds[64];
        int32_t count;
        {
            const StrSpan kb = mb.overallHashMemory.originals.inner().keyAt(chainIdx[cix]);
            const IdVecKeyView kv = viewIdVecKey(kb);
            count = kv.count();
            assert(count <= 64 && "checkNecessityForEquality: chain exceeds 64");
            for (int32_t i = 0; i < count; ++i) chainIds[i] = kv.idAt(i);
        }
        if (count == 0) continue;
        // Copy each chain element onto sArena BEFORE the premise loop (which mints
        // ruleInterner via prepareIntegration -> addToHashMemory, so a raw
        // decodeView would dangle -- I-3). Held under cixScope.
        StrSpan chainSpans[64];
        for (int32_t i = 0; i < count; ++i) {
            const StrSpan dv = mb.ruleInterner.decodeView(chainIds[i]);
            chainSpans[i] = StrSpan(ScratchString::copyFrom(sArena, dv.ptr, dv.len));
        }

        // "Head" is the last element of the implication chain (D10).
        const StrSpan headStr = chainSpans[count - 1];
        const StrSpan headName = extractExpressionSpan(headStr);
        StrSpan headArgs[ExecutionParameters::MAX_ARITY];
        const int32_t headArgN = getArgsSpans(headStr, headArgs,
                                              ExecutionParameters::MAX_ARITY);

        // Structural check: Name and Arity must match the input expression
        if (!equalSpans(headName, inputName)) continue;
        if (headArgN != inputArgN) continue;

        // Check if head is "Generic except for one arg X"
        // Condition: Exactly one argument in head does NOT start with "u_" (the constant).
        int constantIndex = -1;
        int constantCount = 0;

        for (int32_t i = 0; i < headArgN; ++i) {
            // Check if argument starts with "u_"
            const bool isVar = (headArgs[i].len >= 2 && headArgs[i].ptr[0] == 'u' && headArgs[i].ptr[1] == '_');

            if (!isVar) {
                constantCount++;
                constantIndex = static_cast<int>(i);
            }
        }

        // Requirement: Exactly one arg is not u_
        if (constantCount != 1) continue;

        // NEW: Check that definition set of this arg is (1)
        const ce::CoreExpressionConfig* cfg = coreConfig(headName);
        if (cfg == nullptr) continue;

        // definitionSets keys are 1-based; findDefinitionSet builds the decimal
        // key on the stack and linear-scans -> nullptr == find()==end(), the
        // value .first compared span-wise (0% heap).
        const std::pair<std::string, bool>* ds =
            findDefinitionSet(cfg, constantIndex + 1);
        if (ds == nullptr || !equalSpans(StrSpan(ds->first), StrSpan("(1)", 3))) {
            continue;
        }

        // Requirement: Check that head is equal input with exception of one arg
        // We compare all arguments except the constantIndex.
        bool match = true;
        for (int32_t i = 0; i < headArgN; ++i) {
            if (static_cast<int>(i) == constantIndex) continue;

            if (!equalSpans(headArgs[i], inputArgs[i])) {
                match = false;
                break;
            }
        }
        if (!match) continue;

        // "Replacement map is not identity. in one arg they r not equal. these two args are use for replacement"
        const StrReplacement rp[1] = { { headArgs[constantIndex], inputArgs[constantIndex] } };

        bool conditionMet = false;
        const StrSpan targetVar = inputArgs[constantIndex]; // The variable we are tracking

        // Iterate over premises (all elements except the last one)
        for (int32_t k = 0; k + 1 < count; ++k) {
            const StrSpan premise = chainSpans[k];   // stable copy on sArena (cixScope)

            // Apply replacement (Replace the constant from Head with the variable from Input)
            const ScratchString instantiatedPremise =
                replaceKeysScratch(sArena, premise, rp, 1);

            const ScratchString removed =
                removeUPrefixScratch(sArena, StrSpan(instantiatedPremise));
            // argsSet: sorted-unique StrSpan run over `removed` (the prepareIntegration
            // span core's @pre); spans into `removed` (on sArena, held).
            StrSpan removedArgs[ExecutionParameters::MAX_ARITY];
            const int32_t removedArgN = getArgsSpans(StrSpan(removed), removedArgs,
                                                     ExecutionParameters::MAX_ARITY);
            StrSpan argsSet[ExecutionParameters::MAX_ARITY];
            int32_t argsSetN = 0;
            for (int32_t a = 0; a < removedArgN; ++a) {
                int32_t lo = 0, hi = argsSetN; bool dup = false;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    const int c = compareSpans(argsSet[mid], removedArgs[a]);
                    if (c == 0) { dup = true; break; }
                    if (c < 0) lo = mid + 1; else hi = mid;
                }
                if (dup) continue;
                for (int32_t z = argsSetN; z > lo; --z) argsSet[z] = argsSet[z - 1];
                argsSet[lo] = removedArgs[a]; ++argsSetN;
            }

            prepareIntegration(StrSpan(removed), argsSet, argsSetN, mb, StrSpan(validityName), inputExprStr);

            // NEW: Explicitly check inputArgs[constantIndex] occurs 2 times
            // and did NOT occur 2 times before replacement

            // 1. Count in instantiated premise
            StrSpan pArgs[ExecutionParameters::MAX_ARITY];
            const int32_t pArgN = getArgsSpans(StrSpan(instantiatedPremise), pArgs,
                                               ExecutionParameters::MAX_ARITY);
            int countAfter = 0;
            for (int32_t a = 0; a < pArgN; ++a) {
                if (equalSpans(pArgs[a], targetVar)) countAfter++;
            }

            // 2. Count in original premise
            StrSpan origArgs[ExecutionParameters::MAX_ARITY];
            const int32_t origArgN = getArgsSpans(premise, origArgs,
                                                  ExecutionParameters::MAX_ARITY);
            int countBefore = 0;
            for (int32_t a = 0; a < origArgN; ++a) {
                if (equalSpans(origArgs[a], targetVar)) countBefore++;
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
            {
                const NameId targetVarId = mb.nameMap.lookup(targetVar);
                // Zero-copy view walk over the scope's class run — the former
                // decodeClassesAt heap snapshot. Pure read (lookup non-minting,
                // classHasMember / memberCount), the whole scan consumed before
                // any mint, so the peeked views stay valid. Bucket resolution
                // mirrors decodeClassesAt: name -> id -> bucket, {} at either miss.
                const NameId gateValId = mb.nameMap.lookup(validityName);
                const int32_t bucketId = (gateValId == 0)
                    ? 0 : mb.equivalenceClassesMap.lookup(gateValId);
                if (bucketId != 0) {
                    const unsigned slot = (g_currentCoreId >= 0)
                        ? static_cast<unsigned>(g_currentCoreId)
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& gArena = genScratchArenas().forSlot(slot);
                    const int32_t cc = mb.equivalenceClassesMap.runLen(bucketId);
                    for (int32_t ci = 0; ci < cc; ++ci) {
                        ScratchScope viewScope(gArena);
                        int32_t blen = 0;
                        const char* bp = mb.equivalenceClassesMap.peekRecordBytes(
                            bucketId, ci, blen, gArena);
                        const EquivalenceClassView view{ bp, blen };
                        if (targetVarId != 0 && classHasMember(view, targetVarId)) {
                            // User Request: assert that an eq class which contains the var has more than 1 element
                            assert(view.memberCount() > 1);
                            hasExistingEquality = true;
                            break;
                        }
                    }
                }
            }

            // If yes, do not add a new copy
            if (hasExistingEquality) {
                continue;
            }

            // "call addExprToMemoryBlock() for '(=[Y,Y_copy])'"

            // Assert that Y is available
            assert(inputArgN > constantIndex);

            // Extract the input arg (Y) and build "(=[Y,Y_copy])" on sArena (D14,
            // the reactToHypo precedent): "(=[" + Y + "," + Y + "_copy])".
            const StrSpan Y = inputArgs[constantIndex];
            const int32_t elen = 11 + 2 * Y.len;  // 3 + Y + 1 + Y + 7 ("_copy])")
            char* eb = sArena.allocBytes(elen);
            int32_t eat = 0;
            eb[eat++] = '('; eb[eat++] = '='; eb[eat++] = '[';
            if (Y.len > 0) std::memcpy(eb + eat, Y.ptr, static_cast<size_t>(Y.len));
            eat += Y.len;
            eb[eat++] = ',';
            if (Y.len > 0) std::memcpy(eb + eat, Y.ptr, static_cast<size_t>(Y.len));
            eat += Y.len;
            const char eSfx[] = "_copy])";
            std::memcpy(eb + eat, eSfx, 7); eat += 7;
            assert(eat == elen);
            const ScratchString equalityExpr = ScratchString::wrap(sArena, eb, elen);

            // Create origin info (dead-end axiom — see "variable copy" tag)
            const TransientOrigin origin{ true, OriginTag::variableCopy, nullptr, 0 };

            const ScratchString finalExpr =
                removeUPrefixScratch(sArena, StrSpan(equalityExpr));

            // Door levels: the LB's own level as a one-element stack run.
            const int lvRun[1] = { mb.level };

            // Add the new equality
            addExprToMemoryBlock(StrSpan(finalExpr),
                mb,
                -1, // iteration
                0,  // status (Local statement)
                lvRun, 1,
                origin,
                -1, // coreId
                -1, // auxyIndex
                StrSpan("main", 4),
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



bool ExpressionAnalyzer::checkForEquivalence(const std::string& expr,
    const std::string& validityName,
    Memory& memoryBlock)
{
    return checkForEquivalence(StrSpan(expr), StrSpan(validityName), memoryBlock);
}

bool ExpressionAnalyzer::checkForEquivalence(StrSpan expr,
    StrSpan validityName,
    Memory& memoryBlock)
{
    // 1. Parse arguments (D1: spans over the stable `expr`).
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int32_t argN = getArgsSpans(StrSpan(expr), args, ExecutionParameters::MAX_ARITY);

    // Access the equivalence class list for this validity scope
    const NameId eqClassesValId = memoryBlock.nameMap.lookup(validityName);
    assert(eqClassesValId != 0
        && memoryBlock.equivalenceClassesMap.lookup(eqClassesValId) != 0
        && "disintegrateEquivalenceClass: expected a bucket at this validity");
    // Member-run snapshot (page tier) taken ONCE before the args loop — the
    // former decodeClassesById heap vector, reduced to member ids (the loop
    // reads only membership + member names). The reused snapshot decouples from
    // the Cartesian expansion's later mints EXACTLY as the heap snapshot did
    // (I-142); classStarts[ci] is the start of class ci's
    // member run in memberPool.
    const unsigned eqSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& eqArena = genScratchArenas().forSlot(eqSlot);
    DirtyState snapDirty = DirtyState::Clean;
    PagedVector<NameId> memberPool(&eqArena, &snapDirty);
    PagedVector<int32_t> classStarts(&eqArena, &snapDirty);
    {
        const int32_t bucketId =
            memoryBlock.equivalenceClassesMap.lookup(eqClassesValId);
        const int32_t cc = (bucketId == 0)
            ? 0 : memoryBlock.equivalenceClassesMap.runLen(bucketId);
        for (int32_t ci = 0; ci < cc; ++ci) {
            ScratchScope peekScope(eqArena);
            int32_t blen = 0;
            const char* bp = memoryBlock.equivalenceClassesMap.peekRecordBytes(
                bucketId, ci, blen, eqArena);
            const EquivalenceClassView v{ bp, blen };
            classStarts.push_back(static_cast<int32_t>(memberPool.size()));
            for (int32_t k = 0; k < v.memberCount(); ++k)
                memberPool.push_back(v.memberId(k));
        }
    }

    // 2. Per-arg possibility RANGE over memberPool (D2): either a class member
    // run [possStart, possStart+possLen) in decoded-lex storage order, or a "self"
    // singleton (the arg span itself). The class-membership scan is id-based
    // (unchanged); the chosen member decodes at the D5 replacement build.
    int32_t possStart[ExecutionParameters::MAX_ARITY];
    int32_t possLen[ExecutionParameters::MAX_ARITY];
    bool    possSelf[ExecutionParameters::MAX_ARITY];
    StrSpan possSelfSpan[ExecutionParameters::MAX_ARITY];
    for (int32_t ai = 0; ai < argN; ++ai) {
        const StrSpan arg = args[ai];
        // Non-minting probe: a never-interned arg is in no class.
        const NameId argId = memoryBlock.nameMap.lookup(arg);
        const int32_t classCount = classStarts.size();
        bool foundClass = false;
        for (int32_t ci = 0; ci < classCount && !foundClass; ++ci) {
            const int32_t start = classStarts[ci];
            const int32_t end = (ci + 1 < classCount)
                ? classStarts[ci + 1] : static_cast<int32_t>(memberPool.size());
            bool contains = false;
            if (argId != 0) {
                for (int32_t k = start; k < end; ++k) {
                    if (memberPool[k] == argId) { contains = true; break; }
                }
            }
            if (contains) {
                possStart[ai] = start; possLen[ai] = end - start; possSelf[ai] = false;
                foundClass = true; // one class per validity
            }
        }
        // If not found, it is its own only possibility.
        if (!foundClass) {
            possSelf[ai] = true; possSelfSpan[ai] = arg; possStart[ai] = 0; possLen[ai] = 1;
        }
    }

    // 4. Generate All Variants (Cartesian Product) and Check.
    int32_t indices[ExecutionParameters::MAX_ARITY];
    for (int32_t i = 0; i < argN; ++i) indices[i] = 0;

    const unsigned vSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& vArena = scratchArenas().forSlot(vSlot);

    while (true) {
        // A/B. Per-tuple StrReplacement run + variant (D5). The chosen member id
        // decodes zero-copy via decodeView; this is SAFE because the ONLY interner
        // touch inside the loop is the non-minting lookupStatementFlags probe, so
        // no NameMap mint intervenes before the span's last use (I-3, 09c §1).
        ScratchScope vScope(vArena);
        StrReplacement rep[ExecutionParameters::MAX_ARITY];
        for (int32_t i = 0; i < argN; ++i) {
            rep[i].key = args[i];
            rep[i].value = possSelf[i]
                ? possSelfSpan[i]
                : memoryBlock.nameMap.decodeView(memberPool[possStart[i] + indices[i]]);
        }
        const ScratchString variant = replaceKeysScratch(vArena, StrSpan(expr), rep, argN);

        // C. Check if this variant is a REGISTERED statement flagged FULLY
        // DISINTEGRATED. A variant that is merely present (mail-origin, or
        // a local twin whose witness was rejected) was NOT fully disintegrated,
        // so it must NOT suppress this disintegration — gate on the
        // fullyDisintegrated flag, not local-ness and not mere presence.
        const StatementFlags* cfeVarRow = lookupStatementFlags(
            memoryBlock.intKnownStatements, memoryBlock.nameMap, StrSpan(variant), StrSpan(validityName));
        if (cfeVarRow && cfeVarRow->registered && cfeVarRow->fullyDisintegrated) {
            return true;
        }

        // D. Advance indices (odometer style).
        int32_t i = argN - 1;
        while (i >= 0) {
            indices[i]++;
            if (indices[i] < possLen[i]) {
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

void ExpressionAnalyzer::updateWeakVariables(Memory& mb, StrSpan validityName, StrSpan eqlty) {
    // 1. Identify variables involved in the equality (arity-bounded spans).
    StrSpan eqVarSpans[ExecutionParameters::MAX_ARITY];
    const int32_t eqVarN = getArgsSpans(eqlty, eqVarSpans,
                                        ExecutionParameters::MAX_ARITY);
    if (eqVarN == 0) return;
    // Non-minting id probes: a variable never interned cannot be a class
    // member (members are interned by rebuildIntMirror).
    NameId eqVarIds[ExecutionParameters::MAX_ARITY];
    for (int32_t i = 0; i < eqVarN; ++i)
        eqVarIds[i] = mb.nameMap.lookup(eqVarSpans[i]);

    // 2-4. Locate the target class at this validity via a zero-copy view walk
    // (no heap EquivalenceClass decode), then record its weak members inline.
    // Steps 3-4 mint only NameMap / intWeakVariables / the name-kind cache,
    // never equivalenceClassesMap, so the peeked blob stays valid across them.
    // The class blob is peeked onto the per-slot gen-scratch byte-bump tier via
    // the ScratchArena& peekBlobContiguous overload (contiguous -> pool pointer,
    // straddle -> fresh arena copy), each iteration reclaimed by its own
    // ScratchScope. This completes the equi-class page-straddle-buffer sweep
    // (the cleanUpExpressions / updateEquivalenceClasses /
    // applyEquivalenceClassToNegatedEquality siblings already retired their
    // std::vector<char> buffers via this overload).
    const NameId weakValId = mb.nameMap.lookup(validityName);
    const int32_t bucketId = mb.equivalenceClassesMap.lookup(weakValId);
    assert(weakValId != 0 && bucketId != 0
        && "updateWeakVariables: expected a bucket at this validity");
    const std::size_t classCount = static_cast<std::size_t>(
        mb.equivalenceClassesMap.runLen(bucketId));
    const unsigned wvSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& wvArena = genScratchArenas().forSlot(wvSlot);

    bool found = false;
    for (std::size_t ci = 0; ci < classCount && !found; ++ci) {
        ScratchScope clsScope(wvArena);
        int32_t clsLen = 0;
        const char* clsPtr = mb.equivalenceClassesMap.inner().peekBlobContiguous(
            bucketId, static_cast<int32_t>(ci), clsLen, wvArena);
        const EquivalenceClassView ec{ clsPtr, clsLen };

        // First class containing any equality variable is the target — same
        // selection as the former decoded-vector scan (blob-run order).
        bool isTarget = false;
        for (int32_t i = 0; i < eqVarN; ++i) {
            if (eqVarIds[i] != 0 && classHasMember(ec, eqVarIds[i])) {
                isTarget = true;
                break;
            }
        }
        if (!isTarget) continue;
        found = true;

        // 3. Canonical variable: first int_lev_* member in decoded-lex order,
        // fallback first it_*_lev_* (firstSpecialMemberId, templated on the view).
        const NameId canonicalId = firstSpecialMemberId(ec, mb);
        if (canonicalId == 0) return;  // no special-tier members -> nothing weak

        // 4. Weak = special tier (int_/it_) AND not canonical. Dual-write via the
        // packed twin (D-134). Write site, minting allowed.
        const NameId weakValidityId = mb.nameMap.encode(validityName);
        for (int32_t k = 0; k < ec.memberCount(); ++k) {
            const NameId id = ec.memberId(k);
            if (id == canonicalId) continue;
            const NameKind kind = mb.eqClassNameCaches.kindOf(id, mb.nameMap);
            if (kind == NameKind::Normal) continue;
            mb.intWeakVariables.mint(packStatementKey(id, weakValidityId));
        }
    }
    if (!found) assert(false);
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
///    via the kernel; updates `intEncodedStatements`,
///    `intEncodedStatements`, `intLocalEncodedStatementsSet`, and
///    `intKnownStatements`.
/// 2. **Equality** (e.g. `(=[a,b])`) — routes through `addEquality`
///    so the equivalence-class machinery can absorb it
///    (with mirror per [I-9](../../docs/agentic_swdd/30_invariants.md#i-9)).
/// 3. **Negated equality** (e.g. `!(=[a,b])`) — routes through
///    `addNegatedEquality` for the one-sided expansion per
///    [I-12](../../docs/agentic_swdd/30_invariants.md#i-12).
/// 4. **Implication** (e.g. `(>[v](p[v])(q[v]))`) — installed as a
///    hash rule via `addToHashMemory` (after disintegration when
///    appropriate).
///
/// Anchor slot typing is NOT dispatched here: the x-prefixed anchor
/// variants are installed grid-wide by the `prehandleAnchor` pre-pass
/// before the iteration loop runs.
///
/// The `status` mode selector controls the deposit path; the full
/// table lives in [`docs/agentic_swdd/10_pipeline/04_prover.md`](../../docs/agentic_swdd/10_pipeline/04_prover.md)
/// (`addExprToMemoryBlock — status parameter reference`). Statuses
/// 0/1 disintegrate via `disintegrateExpr2`. **status 3** (external-mail
/// absorb, ASIC 0.1 reshuffle) is admitted to `disintegrateExpr2` ONLY
/// when the expression is a compact implication `(implication<N>...)`
/// (gate at the call site, OUTSIDE disintegrateExpr2); a plain mail
/// statement is absorbed as a statement, never disintegrated. Recovered
/// implications install through the usual persistent 3-way
/// `addToHashMemory` path.
///
/// **Per [I-27](../../docs/agentic_swdd/30_invariants.md#i-27)**, ancestor-scan
/// dedupe runs at this entry (Site F / Site H): if the expression
/// already exists in any ancestor scope, the deposit is skipped.
/// This is what keeps cross-LB writes during the parallel phase
/// from causing duplicate-deposit storms
/// ([I-28](../../docs/agentic_swdd/30_invariants.md#i-28)).
///
/// @param expr           Canonical MPL expression text.
/// @param memoryBlock    Target LB.
/// @param involvedLevels Ascending-unique level run — LB levels at which
///                       the expression is admissible (caller-owned per
///                       the levels-run chain contract).
/// @param involvedLevelCount The level count (`>= 0`).
/// @param ...            Many trailing parameters routing the deposit
///                       through specific paths (origin tag,
///                       admission-update flag, mail-out flag,
///                       sender LB, etc.).
/// @invariant [I-9](../../docs/agentic_swdd/30_invariants.md#i-9) — addEquality
///            mirror guard.
/// @invariant [I-12](../../docs/agentic_swdd/30_invariants.md#i-12) — !(=[a,b])
///            one-sided expansion.
/// @invariant [I-27](../../docs/agentic_swdd/30_invariants.md#i-27) — Site F / H
///            ancestor-scan dedupe at entry.
/// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — cross-LB
///            writes deferred to post-`pool.join()` collectors.
/// @see `addExprToMemoryBlockKernel` — inner kernel.
/// @see `addToHashMemory` (memory.cpp) — implication branch.
/// @see `addEquality`, `addNegatedEquality` — shape-specific branches.
void ExpressionAnalyzer::addExprToMemoryBlock(StrSpan expr,
    Memory& memoryBlock,
    int iteration,
    int status,
    const int* involvedLevels, int32_t involvedLevelCount,
    const TransientOrigin& origin,
    int coreId,
    int auxyIndex,
    StrSpan validityName,
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
    NameId origId = memoryBlock.nameMap.encode(expr);
    NameId valId = memoryBlock.nameMap.encode(validityName);

    // Site F — dup suppression via ancestor scan. A known statement at any
    // ancestor scope (including self) means the same fact already holds at a
    // strictly weaker set of assumptions, so the child-scope insertion is
    // redundant. ancestorsOf[valId] includes valId itself and every strict
    // prefix scope registered via encodePush. For a flat root (e.g. legacy
    // integration cleanSignature-as-validity) the list is just {valId},
    // degenerating to the old strict-equality check — safe under the
    // MAIN_ID transition guard.
    if (!parameters.compressor_mode) {
        for (int32_t ancK = 0, ancN = memoryBlock.nameMap.ancLen(valId);
             ancK < ancN; ++ancK) {
            const NameId anc = memoryBlock.nameMap.ancAt(valId, ancK);
            const StatementFlags* kf = memoryBlock.intKnownStatements.find(StatementKey{ origId, anc });
            if (kf != nullptr && kf->known) return;
        }
    }

    // Site H — ancestor-scan the int validity blacklist. A blacklist entry at
    // any ancestor scope filters every descendant scope (deeper scopes inherit
    // the filter because they carry strictly more assumptions).
    for (int32_t ancK = 0, ancN = memoryBlock.nameMap.ancLen(valId);
         ancK < ancN; ++ancK) {
        const NameId anc = memoryBlock.nameMap.ancAt(valId, ancK);
        if (memoryBlock.intValidityNamesToFilter.contains(anc)) return;
    }

    // Passed fast checks — now do full encoding for axed var check + downstream.
    // Span twin: parse + encode in one pass off the stable expr/validityName
    // buffers, no intermediate EncodedExpression heap.
    IntEncodedExpr ie =
        encodeExpression(expr, validityName, memoryBlock.nameMap);

    // Axed-variable containment: a deposit carrying an x-copy name in any
    // argument slot is dropped — the copies exist to serve as anchor-premise
    // completions, never to breed derived x-facts. Flag-gated exception
    // (D-234, parameters.axed_anchor_exception,
    // default off): a POSITIVE anchor-category statement passes — an
    // anchor-bridge rule firing on the x-copied batch anchor derives an
    // external anchor's x-form (e.g. (AnchorPeano[1,x2,3,4,5,x6]) from
    // (AnchorIncubator3[1,x2,3,4,5,x6,x7,8,9])), and that form must be a
    // live statement so external-anchor-premise rules can bind element
    // arguments at anchor slot values. Off by default because a live
    // x-anchor opens the whole external anchor's rule universe at
    // x-arguments and explodes the batch runtime. Negated anchor forms stay
    // filtered even when enabled: a derived negated x-anchor at main would
    // pair against the positive form as a spurious premise-inconsistency
    // (D-236).
    if (!(parameters.axed_anchor_exception && ie.isAnchor && !ie.negation)) {
        for (NameId a = 0; a < ie.arity; ++a) {
            if (memoryBlock.intAxedVariables.contains(ie.argFullId[a])) return;
        }
    }


    if (status == 4)
    {
        memoryBlock.intLocalEncodedStatementsSet.mint(
            packStatementKey(ie.originalId, ie.validityId));
        memoryBlock.intEncodedStatements.push_back(ie);
        memoryBlock.intLocalEncodedStatements.push_back(ie);
        memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
        memoryBlock.intStatementLevelsMap.assignSetRange(
            packStatementKey(ie.originalId, ie.validityId),
            involvedLevels, involvedLevels + involvedLevelCount);
        upsertStatementKey(memoryBlock.intKnownStatements,
            packStatementKey(ie.originalId, ie.validityId),
            /*local=*/true, /*registered=*/true, /*known=*/true);

        return;
    }

    // Past the span-native fast paths (Site F/H dedupe, axed-var scan, status-4).
    // The deeper deposit paths are span-native now: disintegrateExpr2 /
    // checkForEquivalence / checkNecessityForEquality / prepareIntegration /
    // updateAdmissionMap3 all take StrSpan, and the DisintProducts channels are
    // consumed directly -- no expr / validity std::string is materialized here.

    const NameId classValidityId = memoryBlock.nameMap.encode(validityName); // write site
    if (memoryBlock.equivalenceClassesMap.lookup(classValidityId) == 0)
    {
        // Open an empty bucket for this validity (the cold-map form of the
        // former `map[id] = {}`) so a never-filled scope still appears as a
        // 0-class row in the dump, exactly as the std::map seeding did.
        memoryBlock.assignClassesById(classValidityId, {});
        // eqClassSttmntIndexMapMap is flattened to a byte-key map
        // (bytes(validityId ++ memberIds) -> waterline); there is no
        // per-validity inner map to pre-create — entries install on first
        // upsert (the prior empty-map seeding + its assert had no flat analog).
    }


    // CE-filter contradiction detection is no longer gated here. A CE LB's
    // fired heads now enter `intEncodedStatements` like any derived statement (the
    // path below), so both the post-burst `dischargeContradiction` sweep and the
    // mid-burst read-only `burstDeactivates` predicate (prover.hpp) see them and
    // refute the conjecture — handled uniformly with the incubator and
    // vacuous-truth reactions.



    if (status == 2) {
        // Invariant: a toBeProved entry whose body args contain
        // it_/int_ names is class-rewritable by sanitizeToBeProved,
        // and class-rewritable goals never participate in induction
        // discharge. Therefore they must NOT carry an auxy index.
        // Auxies are reserved for induction-discharge bookkeeping
        // and the class-rewrite path bypasses that bookkeeping by
        // construction. Mixing the two means a sanitize pass would
        // rewrite a TBP body while a parallel auxy expects the
        // pre-rename form, breaking the induction discharge contract.
        {
            bool hasItInt = false;
            StrSpan assertArgs[ExecutionParameters::MAX_ARITY];
            const int assertArgN = getArgsSpans(expr, assertArgs, ExecutionParameters::MAX_ARITY);
            for (int ai = 0; ai < assertArgN; ++ai) {
                const StrSpan& a = assertArgs[ai];
                if (isIntLevShape(a) || isItLevShape(a)) {
                    hasItInt = true;
                    break;
                }
            }
            assert((!hasItInt || auxyIndex < 0) &&
                   "toBeProved entry with it_/int_ args must have empty auxies "
                   "(auxies are for induction discharge; class-rewrite path is "
                   "disjoint from induction).");
        }

        // Packed goal key — origId/valId were encoded at function entry, so
        // the goal write mints nothing new.
        const int64_t pkTBP = packStatementKey(origId, valId);

        const int32_t tbpExistingId = memoryBlock.intToBeProved.lookup(pkTBP);
        if (tbpExistingId != 0) {
            if (auxyIndex >= 0) {
                memoryBlock.intToBeProved.insertSorted(pkTBP, auxyIndex);
            }
        }
        else {
            if (auxyIndex >= 0)
                memoryBlock.intToBeProved.insertSorted(pkTBP, auxyIndex);
            else
                memoryBlock.intToBeProved.assignSet(pkTBP, nullptr, 0);

			checkNecessityForEquality(expr, memoryBlock, validityName);
        }

        StrSpan argSpans[ExecutionParameters::MAX_ARITY];
        const int argSpanN = getArgsSpans(expr, argSpans, ExecutionParameters::MAX_ARITY);
        // Sorted-unique stack run over argSpans (ascending == the former std::set
        // order, the prepareIntegration span core's precondition), 0% heap.
        StrSpan unchRun[ExecutionParameters::MAX_ARITY];
        int32_t unchN = 0;
        for (int ai = 0; ai < argSpanN; ++ai) {
            const StrSpan v = argSpans[ai];
            int32_t lo = 0, hi = unchN; bool dup = false;
            while (lo < hi) {
                const int32_t mid = (lo + hi) / 2;
                const int c = compareSpans(unchRun[mid], v);
                if (c == 0) { dup = true; break; }
                if (c < 0) lo = mid + 1; else hi = mid;
            }
            if (dup) continue;
            assert(unchN < ExecutionParameters::MAX_ARITY
                && "prepareIntegration: unchangeable args exceed MAX_ARITY");
            for (int32_t z = unchN; z > lo; --z) unchRun[z] = unchRun[z - 1];
            unchRun[lo] = v; ++unchN;
        }
        prepareIntegration(expr, unchRun, unchN, memoryBlock, validityName, expr);
        // `allowedForMail` probes the memo only for `int_lev_*`-carrying
        // expressions (its lexical scanSingleDistinctIntLev gate short-circuits
        // everything else), so any other entry could never be read.
        if (containsSpan(expr, StrSpan("int_lev_", 8))) {
            memoryBlock.canBeSentIds.mint(memoryBlock.nameMap.encode(expr));
        }

        const ce::CoreExpressionConfig* cfg = coreConfig(extractExpressionSpan(expr));

        if (cfg != nullptr && !cfg->inputIndices.empty()) {
            this->updateAdmissionMap3(expr,
                memoryBlock,
                parameters.inductionMaxAdmissionDepth,
                parameters.inductionMaxSecondaryNumber,
                true);
        }
        return;
    }
    else {
        if (parameters.trackHistory && origin.present) {
            assert(origin.tag != OriginTag::COUNT);
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, expr, validityName, origin.tag, origin.deps, origin.depN, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        // Statuses 0/1/3 reach here (status 2 returned above, status 4
        // returned earlier). Status 0/1 (local derivations) always
        // disintegrate. status 3 (external-mail / non-local absorb) is
        // admitted to disintegrateExpr2 ONLY for the two RULE-CARRIER
        // shapes: a compact implication "(implication<N>...)" and a
        // negated compact existence "!(existence<N>...)" (a universal in
        // De-Morgan clothing — disintegrating it installs its universal
        // rule and mints NO witnesses; witness minting belongs to the
        // positive existence opening). A plain mail statement/fact must
        // NOT be disintegrated under status=3: disintegrating it mints
        // fresh it_/int_ names, which (a) overflows the static
        // MAX_NAME_IDS cap (prover.hpp assert) and (b) drives the
        // non-convergence runaway. Without the negated-existence
        // admission, a descendant LB receives a universal premise (e.g.
        // an induction step sub-LB inheriting "7 has no predecessor")
        // as a bare statement whose rule can then never meet the
        // descendant-local facts it must refute — the ancestor's rule
        // copy cannot see child-local typings, mail flows only
        // parent->children (I-57). The guard is OUTSIDE
        // disintegrateExpr2 (at this call site), not at the
        // in-disintegrate variable-admission point.
        {
            // disintegrateExpr2's bool output, consumed after the stmts loop to
            // flag expr's intKnownStatements entry for cFE.
            bool fullDisintegrationHappened = false;

            // Compact implication = literal "(implication" followed by a
            // digit (the compileImplicationToCompact form "(implication<N>[...]").
            const bool isCompactImplication =
                expr.len > 12 && startsWithSpan(expr, "(implication", 12)
                && expr.ptr[12] >= '0' && expr.ptr[12] <= '9';

            // Negated compact existence = literal "!(existence" followed by
            // a digit — the second rule-carrier shape admitted at status 3.
            const bool isNegatedExistence =
                expr.len > 11 && startsWithSpan(expr, "!(existence", 11)
                && expr.ptr[11] >= '0' && expr.ptr[11] <= '9';

            // Suppress disintegration when an equivalence-class variant of
            // this expression is already a registered statement.
            const bool cfeEq = (!doNotDisintegrate)
                ? checkForEquivalence(expr, validityName, memoryBlock) : false;
            const bool willDisintegrate =
                !doNotDisintegrate && !cfeEq
                && (status != 3 || isCompactImplication || isNegatedExistence);
            // Hoist the two return channels ABOVE the if/else so both branches
            // fill ONE `out` (per-slot arena, freed per task); consumed directly
            // below via forEachSorted (no std::set<EWV> edge-materialization).
            const unsigned deSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : genScratchArenas().slotCount() - 1;
            DisintProducts out(&genScratchArenas().forSlot(deSlot));
            if (willDisintegrate)
            {
                // Rule-19 invariant: a non-rule-carrier must only reach
                // disintegrateExpr2 under a local status (0/1). status=3 is
                // admissible here ONLY for the two rule-carrier shapes —
                // compact implication or negated compact existence (the
                // guard enforces it); anything else reaching here with
                // status=3 fires this assert and pins the forbidden path.
                assert(isCompactImplication || isNegatedExistence
                    || status == 0 || status == 1);
                fullDisintegrationHappened = this->disintegrateExpr2(expr,
                        memoryBlock,
                        iteration,
                        status == 0,
                        validityName,
                        out,
                        /*trackHistoryLocal=*/true,
                        allowOrDisintegration);
            }
            else
            {
                out.statements.append(expr, validityName);
            }

            out.implications.forEachSorted([&](StrSpan impStr, StrSpan impValidity)
            {
                // Row 248: the key-only span twin of the retired extractKeyValue
                // (its value field is dead here). kyS is a NAMED ScratchString on
                // the string-tier arena, held live under kvScope across the
                // extractRemainingArgs read below (09c pitfall 4). impStr aliases
                // out's channel interner (stable across the loop; the callee mints
                // are into DIFFERENT interners, I-3).
                const unsigned kvSlot = (g_currentCoreId >= 0)
                    ? static_cast<unsigned>(g_currentCoreId)
                    : scratchArenas().slotCount() - 1;
                ScratchArena& kvArena = scratchArenas().forSlot(kvSlot);
                ScratchScope kvScope(kvArena);
                const ScratchString kyS =
                    ce::extractKeyValueKeyScratch(impStr, kvArena);

                // extractRemainingArgs is 0% heap (span output); build the
                // addToHashMemory sink's sorted-unique run (remKeyRun) directly.
                // The slices point into `kyS`, which lives under kvScope.
                StrSpan remainingArgsSpans[ExecutionParameters::MAX_KEY_SLOTS];
                const int32_t remainingArgsN = extractRemainingArgs(
                    StrSpan(kyS), remainingArgsSpans, ExecutionParameters::MAX_KEY_SLOTS);
                StrSpan remKeyRun[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                int32_t remKeyRunN = 0;
                for (int32_t i = 0; i < remainingArgsN; ++i) {
                    const StrSpan v = remainingArgsSpans[i];
                    int32_t lo = 0, hi = remKeyRunN; bool dup = false;
                    while (lo < hi) {
                        const int32_t mid = (lo + hi) / 2;
                        const int c = compareSpans(remKeyRun[mid], v);
                        if (c == 0) { dup = true; break; }
                        if (c < 0) lo = mid + 1; else hi = mid;
                    }
                    if (dup) continue;
                    assert(remKeyRunN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                        && "addToHashMemory rem run exceeds cap");
                    for (int32_t z = remKeyRunN; z > lo; --z) remKeyRun[z] = remKeyRun[z - 1];
                    remKeyRun[lo] = v; ++remKeyRunN;
                }

                // Row 238: span twin — chainRun from each triple's key (get<0>);
                // head = headSpan. Spans slice impStr (the channel span, stable).
                StrSpan chainRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                int32_t chainRunN = 0;
                StrSpan headSpan;
                ce::disintegrateImplicationSpans(impStr, headSpan,
                    [&chainRun, &chainRunN](StrSpan keySpan, const StrSpan*, int32_t) {
                        assert(chainRunN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                            && "addToHashMemory chain run exceeds cap");
                        chainRun[chainRunN++] = keySpan;
                    });


                // Distribute the recovered implication by status (ASIC 0.1
                // reshuffle). overallHashMemory (full visibility) gets it
                // in every case. status=3 (external-mail absorb) is NOT a
                // local derivation: its rule must NOT enter localHashMemory
                // / localHashMemoryDelta (those are local-impl-only — B5's
                // delta batch would otherwise treat an external rule as a
                // local delta); instead it is staged into the per-burst
                // workingMemory, which the NEXT burst's Batch 1 reads.
                // status 0/1 (local derivations) keep the local 3-way
                // fan-out and never touch workingMemory.
                // Span-run edges: chainRun / remKeyRun (built above), headSpan,
                // and the impStr / impValidity channel spans, all stable across
                // the four installs (addToHashMemory mints DIFFERENT interners).
                this->addToHashMemory(chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN,
                    memoryBlock, memoryBlock.overallHashMemory,
                    involvedLevels, involvedLevelCount, impStr,
                    parameters.maxIterationNumberVariable, parameters.standardMaxSecondaryNumber, false,
                    parameters.minNumOperatorsKey, StrSpan("implication", 11), true, impStr, impValidity);
                if (status == 3)
                {
                    this->addToHashMemory(chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN,
                        memoryBlock, memoryBlock.workingMemory,
                        involvedLevels, involvedLevelCount, impStr,
                        parameters.maxIterationNumberVariable, parameters.standardMaxSecondaryNumber, false,
                        parameters.minNumOperatorsKey, StrSpan("implication", 11), true, impStr, impValidity);
                }
                else
                {
                    this->addToHashMemory(chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN,
                        memoryBlock, memoryBlock.localHashMemory,
                        involvedLevels, involvedLevelCount, impStr,
                        parameters.maxIterationNumberVariable, parameters.standardMaxSecondaryNumber, false,
                        parameters.minNumOperatorsKey, StrSpan("implication", 11), true, impStr, impValidity);
                    this->addToHashMemory(chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN,
                        memoryBlock, memoryBlock.localHashMemoryDelta,
                        involvedLevels, involvedLevelCount, impStr,
                        parameters.maxIterationNumberVariable, parameters.standardMaxSecondaryNumber, false,
                        parameters.minNumOperatorsKey, StrSpan("implication", 11), true, impStr, impValidity);
                }

                // Track this implication for the once-per-burst
                // `sanitizeHashMemory` walk that runs at the END of
                // the elementary step (`performElemPhase3`). When a later equi-
                // class downprioritizes an `it_/int_` arg of this
                // implication, the sanitizer mails the canonical-form
                // rewrite to `sameIterationInternalMail` and eradicates this
                // entry's traces from the LB.
                memoryBlock.expandedImplications.mint(LbStatePairKey{
                    memoryBlock.lbStateInterner.encode(impStr),
                    memoryBlock.lbStateInterner.encode(impValidity) });
                // (The cross-LB broadcast of this index entry was removed: the
                // mail expandedImplications column was never serialized/delivered
                // — Codec<Mail> carries only statements + exprOriginMap — so the
                // receiver-side merge was always a no-op. Each LB populates its own
                // expandedImplications index locally, here.)

                // Mail-out contract: cross-LB rule propagation is
                // MAIN-ONLY. Non-main rules stay local — receivers
                // re-derive them from the mailed v=main statements + their
                // own disintegration. mailOut.exprOriginMap (history)
                // continues to carry entries for all scopes per the
                // existing trackExpansionHistory invariant; only the
                // rule-propagation channel is gated. (Pre-fix this gate
                // was missing, and impl24-scope rules from the SE2 LB
                // shipped to the contradiction LB and got installed at
                // v=main with origin keyed at the sender's deeper scope,
                // breaking the visualizer walk's exprOriginMap lookup.)
                //
                // The disintegration-recovered implication is NOT
                // re-broadcast or re-compacted here, and does NOT go
                // through recordPendingCompaction. A locally-derived
                // implication already carries its compact/compiled form in
                // `expressions`, and a mail-recovered one's compact form is
                // the very statement being disintegrated — so no compilation
                // is needed and none is done. recordPendingCompaction here
                // would produce an N-LB-fan-out re-broadcast runaway.
                // Receivers still install the recovered rule locally
                // (overallHashMemory + workingMemory above); the Mail-out
                // MAIN-ONLY contract (I-26) above is unchanged.
            });

            if (out.implications.count() > 0)
            {
                // Check necessity for equality for all pending proofs.
                // Snapshot the packed goal keys sorted by decoded
                // (original, validity) -- byte-identical processing order to the
                // former decodeToBeProvedSorted walk (I-87). The keys are
                // NameMap-id pairs and NameMap only appends, so the snapshot
                // survives checkNecessityForEquality's mints AND any goal-registry
                // insert/erase during the walk; each row's two strings are decoded
                // FRESH right before use (after any prior mint), so no decodeView
                // span crosses a mint (I-3, 09c pitfall 5). tbpKeys rides the
                // slot's gen-scratch byte-bump tier, allocated BEFORE any callee
                // ScratchScope mark, so every inner rewind pops back above it.
                const unsigned tbpSlot = (g_currentCoreId >= 0)
                    ? static_cast<unsigned>(g_currentCoreId)
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& tbpArena = genScratchArenas().forSlot(tbpSlot);
                ScratchScope tbpScope(tbpArena);
                int32_t tbpN = 0;
                int64_t* tbpKeys = sortToBeProvedKeys(memoryBlock.intToBeProved,
                    memoryBlock.nameMap, tbpArena, tbpN);
                const unsigned goalStrSlot = (g_currentCoreId >= 0)
                    ? static_cast<unsigned>(g_currentCoreId)
                    : scratchArenas().slotCount() - 1;
                ScratchArena& goalStrArena = scratchArenas().forSlot(goalStrSlot);
                for (int32_t i = 0; i < tbpN; ++i) {
                    const int64_t k = tbpKeys[i];
                    const NameId oid = Codec<StatementKey>::decode(k).orig;
                    const NameId vid = Codec<StatementKey>::decode(k).validity;
                    // copyFrom per row onto the string tier: checkNecessityFor-
                    // Equality mints NameMap, so a raw decodeView would dangle
                    // (I-3, 09c pitfall 5).
                    ScratchScope goalRowScope(goalStrArena);
                    const StrSpan os = memoryBlock.nameMap.decodeView(oid);
                    const ScratchString original =
                        ScratchString::copyFrom(goalStrArena, os.ptr, os.len);
                    const StrSpan vs = memoryBlock.nameMap.decodeView(vid);
                    const ScratchString validity =
                        ScratchString::copyFrom(goalStrArena, vs.ptr, vs.len);
                    this->checkNecessityForEquality(StrSpan(original), memoryBlock, StrSpan(validity));
                }
            }

            // Process statements — each carries its own validity name (normal
            // stmts have parent validity, OR branches have branch validity).
            out.statements.forEachSorted([&](StrSpan evOrig, StrSpan evValidity)
            {
                int64_t pkEv = 0;
                const bool evHasKey = lookupOriginKey(memoryBlock.originInterner,
                    evOrig, evValidity, pkEv);
                assert(evHasKey);
                (void)evHasKey;
                const int32_t oidOrigin = memoryBlock.exprOriginMap.lookup(pkEv);
                assert(oidOrigin != 0 && memoryBlock.exprOriginMap.runLen(oidOrigin) > 0);
                // frontId: peek exprOriginMap record 0 zero-copy (IdOriginBlobView)
                // instead of an owning recordAt decode. The peek is valid while the
                // run is resident + unmutated; the deps are copied out BEFORE
                // addStatement (the next mint), so no straddle-dangle (I-3).
                const unsigned foSlot = (g_currentCoreId >= 0)
                    ? static_cast<unsigned>(g_currentCoreId)
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& foArena = genScratchArenas().forSlot(foSlot);
                ScratchScope foScope(foArena);
                int32_t foBlen = 0;
                const char* foBytes = memoryBlock.exprOriginMap.peekRecordBytes(
                    oidOrigin, 0, foBlen, foArena);
                const IdOriginBlobView foView = viewIdOriginBlob(foBytes, foBlen);
                // Copy each dep's (expression, validity) onto the string tier
                // (MANDATORY: addStatement mints originInterner, the SAME interner
                // the deps decode from, I-3). An equality1 tail can run up to the
                // 64-dep chain cap.
                const unsigned foStrSlot = (g_currentCoreId >= 0)
                    ? static_cast<unsigned>(g_currentCoreId)
                    : scratchArenas().slotCount() - 1;
                ScratchArena& foStrArena = scratchArenas().forSlot(foStrSlot);
                ScratchScope foStrScope(foStrArena);
                OriginDep frontDeps[64];
                int frontDepN = 0;
                for (int32_t d = 0; d < foView.depN; ++d) {
                    assert(frontDepN < 64 && "front origin chain exceeds 64");
                    StrSpan eSpan, vSpan;
                    decodeOriginKeyView(idOriginDepAt(foView, d),
                        memoryBlock.originInterner, eSpan, vSpan);
                    const ScratchString eC = ScratchString::copyFrom(foStrArena, eSpan.ptr, eSpan.len);
                    const ScratchString vC = ScratchString::copyFrom(foStrArena, vSpan.ptr, vSpan.len);
                    frontDeps[frontDepN++] = { StrSpan(eC), StrSpan(vC) };
                }
                const TransientOrigin frontTO{
                    true, static_cast<OriginTag>(foView.tag), frontDeps, frontDepN };

                // `addStatement` is the single statement-add door.
                // It dispatches internally to `addEquality` /
                // `addNegatedEquality` for equality-shaped inputs and
                // sets the expression's `registered` membership bit.
                {
                    const bool isLocal = (status == 0 || status == 1);

                    // Id-form out-param on the per-slot gen-scratch arena —
                    // addStatement is void now (PagedVector is non-movable, cannot
                    // be returned). Page-tier buffer; coexists with addStatement's
                    // own gen-scratch usage on this slot (each tracks its own pages).
                    const unsigned addedSlot = (g_currentCoreId >= 0)
                        ? static_cast<unsigned>(g_currentCoreId)
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& addedArena = genScratchArenas().forSlot(addedSlot);
                    DirtyState addedDirty = DirtyState::Clean;
                    PagedVector<IntEncodedExpr> added(&addedArena, &addedDirty);
                    this->addStatement(evOrig, memoryBlock, isLocal,
                        involvedLevels, involvedLevelCount, frontTO,
                        evValidity, added);

                    // `added` is not read before here, so index-sort it into the
                    // exact former std::sort(std::vector<EWV>) order
                    // (sortStatementRows) and decode each row at the consumption
                    // edge. Each row's (original, validity) is copied into a per-row
                    // string-scratch ScratchString (byte-identical to nameMap.decode,
                    // on the string tier) rather than a heap std::string; the string
                    // tier is disjoint from the NameMap / interner cold tables the
                    // consumers mint into, so the StrSpan handed to each consumer
                    // survives those mints where a raw decodeView span would dangle
                    // (I-3 / I-138). The order index
                    // rides a ScratchScope on addedArena's byte-bump tier; a per-row
                    // ScratchScope on the string arena reclaims the two copies.
                    ScratchScope addedSortScope(addedArena);
                    const int32_t* order =
                        sortStatementRows(added, memoryBlock.nameMap, addedArena);
                    const unsigned rowStrSlot = (g_currentCoreId >= 0)
                        ? static_cast<unsigned>(g_currentCoreId)
                        : scratchArenas().slotCount() - 1;
                    ScratchArena& rowStrArena = scratchArenas().forSlot(rowStrSlot);
                    for (int32_t idx = 0; idx < added.size(); ++idx) {
                        const IntEncodedExpr& addedRow = added[order[idx]];
                        ScratchScope rowStrScope(rowStrArena);
                        const StrSpan addOrigView =
                            memoryBlock.nameMap.decodeView(addedRow.originalId);
                        const ScratchString addExpression = ScratchString::copyFrom(
                            rowStrArena, addOrigView.ptr, addOrigView.len);
                        const StrSpan addValidView =
                            memoryBlock.nameMap.decodeView(addedRow.validityId);
                        const ScratchString effectiveValidity = ScratchString::copyFrom(
                            rowStrArena, addValidView.ptr, addValidView.len);
                        updateAdmissionMapIntegration(StrSpan(addExpression), memoryBlock,
                                                      StrSpan(effectiveValidity));
                        updateAdmissionMapRecursion(StrSpan(addExpression), memoryBlock,
                                                    StrSpan(effectiveValidity));

                        const int32_t addLvlsId = lookupStatementLevels(
                            memoryBlock.intStatementLevelsMap, memoryBlock.nameMap,
                            StrSpan(addExpression), StrSpan(effectiveValidity));
                        assert(addLvlsId
                               && "addStatement post-loop intStatementLevelsMap invariant violated");
                        int lvRun[256];
                        const int32_t lvN = coldIntRunAt(
                            memoryBlock.intStatementLevelsMap, addLvlsId,
                            lvRun, 256);

                        ordisMerge(StrSpan(addExpression), StrSpan(effectiveValidity),
                                   lvRun, lvN, memoryBlock);
                    }
                }

                // The disintegrateExpr2 `stmts` output is NOT staged into
                // intExternalStatements, which holds the raw mailIn
                // expressions 1:1 (filled in the absorb drain loop), not
                // the recovered/decomposed forms. The recovered stmts are
                // still installed into the LB by the addStatement loop
                // above.
            });

            // expr entered disintegrateExpr2 and came back fully disintegrated:
            // flag its own registry entries so cFE suppresses future equivalence-
            // class variants of expr (a rejecting / partial twin stays unflagged
            // and never suppresses — the Gauss-fold mirror-bug fix). The stmts
            // loop above registered expr's entry when it was returned among stmts;
            // if it was not self-returned, insert the entry.
            if (fullDisintegrationHappened) {
                const bool isLocalExpr = (status == 0 || status == 1);
                const int64_t intKeyFull = packStatementKey(
                    memoryBlock.nameMap.encode(expr),
                    memoryBlock.nameMap.encode(validityName));
                upsertStatementKey(memoryBlock.intKnownStatements, intKeyFull,
                    isLocalExpr, /*registered=*/true, /*known=*/true,
                    /*fullyDisintegrated=*/true);
            }

        }
    }
}

std::vector<std::string> ExpressionAnalyzer::extractSubstringsForAuxy(const std::string& expr2) {
    // Lexical twin of the unanchored iterator over !?\(([^>(\[]+\[[^\]]*\])\):
    // at each '(' the [^>(\[]+ run is maximal and excludes '>','(','[', so its
    // terminator is the first such byte and the pattern then REQUIRES '[' — if
    // the terminator is '>' or '(' the attempt fails with no shorter-run retry
    // possible ('[' cannot occur inside the run); [^\]]* terminates at the
    // first ']'. One deterministic scan per attempt decides exactly what the
    // engine decides. Leftmost / resume-after-match / failed-start-advances-one;
    // a later match may START inside a failed attempt's window ("((a[b]))"
    // matches at offset 1), which the ++pos rule preserves. A '!' immediately
    // preceding the matched '(' belongs to the subexpression — a NEGATED leaf
    // premise (e.g. the OR-companion theorem's !(existence3[...]) universal)
    // must keep its negation, or the induction-hypothesis rule built from the
    // chain silently inverts that premise into an unsound positive form. The
    // whole match (WITH the leading '!' when present and the surrounding
    // parens) is pushed as a std::string — the createAuxyImplication
    // tuple-return boundary heap. THE RULE: in the burst kernel the only
    // permitted heap is the Rule-14 hashburst_trace.txt dump; this
    // tuple-return boundary heap is an OPEN violation to statify, not a
    // sanctioned island.
    const StrSpan s(expr2);
    const int32_t n = s.len;
    std::vector<std::string> out;
    int32_t pos = 0;
    while (pos < n) {
        if (s.ptr[pos] != '(') { ++pos; continue; }
        int32_t i = pos + 1;
        while (i < n && s.ptr[i] != '>' && s.ptr[i] != '(' && s.ptr[i] != '[') ++i;
        if (i - (pos + 1) < 1) { ++pos; continue; }   // [^>(\[]+ needs >= 1
        if (i >= n || s.ptr[i] != '[') { ++pos; continue; }
        int32_t j = i + 1;
        while (j < n && s.ptr[j] != ']') ++j;
        if (j >= n) { ++pos; continue; }              // no ']'
        if (j + 1 >= n || s.ptr[j + 1] != ')') { ++pos; continue; }
        const int32_t start = (pos > 0 && s.ptr[pos - 1] == '!') ? pos - 1 : pos;
        out.push_back(std::string(s.ptr + start,
            static_cast<std::size_t>(j + 1 - start + 1)));  // s[start .. j+1] inclusive
        pos = j + 2;
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
    // untouchables as a sorted-unique StrSpan array (spans into digitArgs /
    // recArg / subexprs, all alive to function end); membership via
    // binary_search(compareSpans); replMap2 reuses the same run. The tuple
    // returns (chain / remainingArgsKey / subexprs) are still heap-allocated
    // (I-138) — an OPEN violation to statify, not a sanctioned "by decision"
    // carve-out. THE RULE: in the burst kernel the only permitted heap is the
    // Rule-14 hashburst_trace.txt dump.
    constexpr int kUntouchCap = 4 * ExecutionParameters::MAX_ARITY;
    StrSpan untouchables[kUntouchCap];
    int32_t untouchN = 0;
    for (const std::string& d : digitArgs) {
        assert(untouchN < kUntouchCap);
        untouchables[untouchN++] = StrSpan(d);
    }
    assert(untouchN < kUntouchCap);
    untouchables[untouchN++] = StrSpan(recArg);

    std::vector<std::string> chain;
    std::string zeroArgName;
    std::string sName;

    // replace arg -> recArg in the whole expression (value is an input slice,
    // so replaceKeysToString needs no arena)
    StrReplacement replPair[1];
    replPair[0].key = StrSpan(arg);
    replPair[0].value = StrSpan(recArg);
    const std::string replExpr = replaceKeysToString(StrSpan(expr), replPair, 1);

    // collect non-top-level subexpressions "(X[...])" with X != '>'
    const std::vector<std::string> subexprs = extractSubstringsForAuxy(replExpr);
    for (std::size_t i = 0; i < subexprs.size(); ++i) {
        const std::string& sub = subexprs[i];
        if (sub.find(anchorName) != std::string::npos) {
            StrSpan anchorArgs[ExecutionParameters::MAX_ARITY];
            const int32_t anchorN = getArgsSpans(
                StrSpan(sub), anchorArgs, ExecutionParameters::MAX_ARITY);
            // add all anchor args to untouchables (spans into the stable sub)
            for (int32_t k = 0; k < anchorN; ++k) {
                assert(untouchN < kUntouchCap);
                untouchables[untouchN++] = anchorArgs[k];
            }

            zeroArgName = findZeroArgNameFromAnchor(sub);
            sName = findSArgNameFromAnchor(sub);
        }
        else {
            chain.push_back(sub);
        }
    }

    // Sort + dedup so binary_search membership and the replMap2 walk reproduce
    // the former std::set (ascending, unique) byte-for-byte.
    std::sort(untouchables, untouchables + untouchN,
        [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) < 0; });
    {
        int32_t w = 0;
        for (int32_t r = 0; r < untouchN; ++r) {
            if (w == 0 || !equalSpans(untouchables[w - 1], untouchables[r])) {
                untouchables[w++] = untouchables[r];
            }
        }
        untouchN = w;
    }

    // remaining_args_key: args from all but the last element of chain that intersect untouchables
    std::set<std::string> remainingArgsKey;
    if (chain.size() >= 1U) {
        for (std::size_t elementIndex = 0; elementIndex + 1 < chain.size(); ++elementIndex) {
            StrSpan elementArgs[ExecutionParameters::MAX_ARITY];
            const int32_t elementN = getArgsSpans(
                StrSpan(chain[elementIndex]), elementArgs,
                ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < elementN; ++a) {
                if (std::binary_search(untouchables, untouchables + untouchN,
                        elementArgs[a],
                        [](const StrSpan& x, const StrSpan& y) {
                            return compareSpans(x, y) < 0;
                        })) {
                    remainingArgsKey.insert(elementArgs[a].toStdString());
                }
            }
        }
    }

    // prefix all untouchables with "u_" inside chain
    if (untouchN > 0) {
        const unsigned uSlot = (g_currentCoreId >= 0)
            ? static_cast<unsigned>(g_currentCoreId)
            : scratchArenas().slotCount() - 1;
        ScratchArena& uArena = scratchArenas().forSlot(uSlot);
        ScratchScope uScope(uArena);

        // Build the "u_"+key run ONCE (fresh bytes on the string-scratch arena
        // under a ScratchScope spanning the whole chain loop), then reuse it
        // across every chain element via replaceKeysScratch.
        StrReplacement replPairs2[kUntouchCap];
        for (int32_t i = 0; i < untouchN; ++i) {
            char* v = uArena.allocBytes(2 + untouchables[i].len);
            v[0] = 'u';
            v[1] = '_';
            if (untouchables[i].len > 0) {
                std::memcpy(v + 2, untouchables[i].ptr,
                    static_cast<size_t>(untouchables[i].len));
            }
            replPairs2[i].key = untouchables[i];
            replPairs2[i].value = StrSpan(v, 2 + untouchables[i].len);
        }
        for (std::size_t i = 0; i < chain.size(); ++i) {
            chain[i] = replaceKeysScratch(
                uArena, StrSpan(chain[i]), replPairs2, untouchN).toStdString();
        }
    }

    // head is last element; chain are the preceding elements
    assert(!chain.empty());
    const std::string head = chain.back();
    chain.pop_back();

    const std::string implication = reconstructImplication(chain, head);

    return std::make_tuple(implication, chain, head, remainingArgsKey, zeroArgName, sName);
}

/// @brief Detect the antisymmetry conjecture shape and name the
///        variable to duplicate — the fourth variable-copy trigger.
///
/// @details
/// An antisymmetry-shaped conjecture carries two positive premises of
/// the same operator whose argument lists are equal except exactly two
/// positions holding the same two arguments swapped
/// (`P[..,x,..,y,..]` / `P[..,y,..,x,..]`), with an equality head over
/// exactly that argument pair. Proving such a head forces downstream
/// rule instances to bind two template-distinct slots to one name — the
/// collision pattern the exact normalized-key request match cannot
/// assemble. The cure is a dead-end variable-copy axiom `(=[x,x_copy])`
/// at the innermost premise LB: the equivalence class then generates
/// all-distinct statement variants and the standard machinery (demand
/// markers, witness revival, rule firings) closes the proof.
///
/// This function is the pure detection half: it scans every ordered
/// pair of chain elements for the shape and, on the first match,
/// reports the swapped argument that appears first left-to-right in the
/// earlier premise — the variable the caller duplicates. Both verdicts
/// are defined results of the detection contract (most conjectures
/// simply do not have the shape).
///
/// Guards: both premises positive (no `!` prefix); identical operator
/// name and arity; exactly two mismatching argument positions, mutually
/// swapped, holding plain atoms (no parentheses — the shallow
/// `ce::getArgs` parse is exact only for flat argument lists); the head
/// a positive `(=[u,v])` over distinct arguments whose set equals the
/// swapped pair. The swap condition itself excludes equal atoms at the
/// mismatch positions (equal atoms cannot mismatch).
///
/// @param chain      The disintegrated premise chain of the conjecture
///                   (raw element strings, e.g. `"(preorder[1,4,7,8])"`).
/// @param head       The conjecture head (e.g. `"(=[7,8])"`).
/// @param copyVarOut On detection, the variable to duplicate (the
///                   swapped argument at the earlier mismatch position
///                   of the earlier element); untouched otherwise.
/// @return True iff the antisymmetry shape was detected.
/// @see ExpressionAnalyzer::addTheoremToMemory — the deposit site;
///      ExpressionAnalyzer::checkNecessityForEquality and
///      ExpressionAnalyzer::reactToHypo — sibling variable-copy
///      emission sites sharing the `variableCopy` origin contract.
bool detectAntisymmetryCopyVar(const std::vector<std::string>& chain,
                               const std::string& head,
                               std::string& copyVarOut) {
    if (head.empty() || head[0] == '!') return false;
    if (ce::extractExpression(head) != "=") return false;
    const std::vector<std::string> headArgs = ce::getArgs(head);
    if (headArgs.size() != 2 || headArgs[0] == headArgs[1]) return false;

    for (std::size_t i = 0; i < chain.size(); ++i) {
        const std::string& p1 = chain[i];
        if (p1.empty() || p1[0] == '!') continue;
        for (std::size_t j = i + 1; j < chain.size(); ++j) {
            const std::string& p2 = chain[j];
            if (p2.empty() || p2[0] == '!') continue;
            if (ce::extractExpression(p1) != ce::extractExpression(p2)) continue;
            const std::vector<std::string> a1 = ce::getArgs(p1);
            const std::vector<std::string> a2 = ce::getArgs(p2);
            if (a1.size() != a2.size() || a1.empty()) continue;

            int p = -1, q = -1;
            bool tooMany = false;
            for (std::size_t k = 0; k < a1.size(); ++k) {
                if (a1[k] == a2[k]) continue;
                if (p < 0) p = static_cast<int>(k);
                else if (q < 0) q = static_cast<int>(k);
                else { tooMany = true; break; }
            }
            if (tooMany || q < 0) continue;

            const std::string& x = a1[static_cast<std::size_t>(p)];
            const std::string& y = a1[static_cast<std::size_t>(q)];
            if (a2[static_cast<std::size_t>(p)] != y
             || a2[static_cast<std::size_t>(q)] != x) continue;
            if (x.find('(') != std::string::npos
             || y.find('(') != std::string::npos) continue;

            const bool headMatches =
                (headArgs[0] == x && headArgs[1] == y) ||
                (headArgs[0] == y && headArgs[1] == x);
            if (!headMatches) continue;

            copyVarOut = x;
            return true;
        }
    }
    return false;
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

        Memory* existingChild = simpleMapStore.findChild(memoryBlock, element);
        if (existingChild != NULL) {
            memoryBlock = existingChild;
        } else {
            Memory* child = lbStore.create<Memory>();
            simpleMapStore.linkChild(memoryBlock, element, child);

            child->parentMemory = memoryBlock;
            child->level = static_cast<int>(index);
            child->setExprKey(element);

            this->permanentBodies.push_back(child);

            memoryBlock = child;
            isNewBody = true;

            const int lvRun[1] = { child->level };
            const TransientOrigin origin{ true, OriginTag::taskFormulation, nullptr, 0 };
            this->addExprToMemoryBlock(element, *child, iteration, 0, lvRun, 1, origin, -1, -1, StrSpan("main", 4), false);

            // LB-creation paired mailOut write — documented exception to the
            // "fillMailOut sole writer" invariant. fillMailOut runs only
            // during prover steps (step 1's pre-burst standardProcessing);
            // buildGrid()'s startup commit needs mailOut already populated so
            // the LB's task-formulation entry lands in its log and reaches every
            // descendant on that descendant's first phase-1 pull. Without this
            // pair, a descendant's exprOriginMap would miss the dep history at
            // its step 1.
            if (parameters.trackHistory) {
                const int maxOriginsLocal = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr;
                ExpressionWithValidity ev(element, "main");
                child->insertMailOutStatement(StrSpan(ev.original),
                    StrSpan(ev.validityName), lvRun, 1);
                child->addMailOutOrigin(StrSpan(ev.original),
                    StrSpan(ev.validityName), origin.tag, origin.deps,
                    origin.depN, maxOriginsLocal);
            }
        }

        // 3) At the last node: set up recursion/auxiliaries.
        // (D-45: the historical proved=true branch was dead — single caller
        // always passes proved=false. Guarded by assert at function entry.)
        if (index == chain.size() - 1) {
            {
                // digits / immutables. findDigitArgs is now 0% heap (sorted-unique
                // StrSpan run); materialize the std::set in this still-heap caller.
                StrSpan digitBuf[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                const int32_t digitN = findDigitArgs(expr, this->anchorInfo, digitBuf,
                                                     ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                std::set<std::string> digits;
                for (int32_t i = 0; i < digitN; ++i)
                    digits.emplace(digitBuf[i].ptr, static_cast<std::size_t>(digitBuf[i].len));
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
                    if (simpleMapStore.findChild(memoryBlock, tempExpr) == nullptr) {
                        Memory* tempMb = lbStore.create<Memory>();
                        simpleMapStore.linkChild(memoryBlock, tempExpr, tempMb);

                        tempMb->level = static_cast<int>(index + 1);
                        tempMb->parentMemory = memoryBlock;
                        tempMb->setExprKey(tempExpr);
                        tempMb->recursionHypothesisId =
                            tempMb->nameMap.encode(tempExpr2);
                        tempMb->isPartOfRecursion = true;



                        // history tag for auxy implication
                        // L3 span-record door. "recursion" has empty antecedents
                        // (P1); the KEY `auxyImplication` is a stable std::string
                        // local, "main" a static literal.
                        addOriginEncoded(tempMb->exprOriginMap, tempMb->originInterner, StrSpan(auxyImplication), StrSpan("main", 4), OriginTag::recursion, nullptr, 0, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));

                        this->permanentBodies.push_back(tempMb);

                        const int lvRun[1] = { tempMb->level };

                        StrSpan auxyChainRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                        int32_t auxyChainRunN = 0;
                        for (const std::string& s : auxyChain) {
                            assert(auxyChainRunN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                                && "addToHashMemory auxyChain run exceeds cap");
                            auxyChainRun[auxyChainRunN++] = StrSpan(s);
                        }
                        StrSpan auxyRemRun[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                        int32_t auxyRemRunN = 0;
                        for (const std::string& s : remainingArgsKey) {
                            assert(auxyRemRunN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                                && "addToHashMemory auxyRem run exceeds cap");
                            auxyRemRun[auxyRemRunN++] = StrSpan(s);
                        }
                        this->addToHashMemory(auxyChainRun, auxyChainRunN, StrSpan(auxyHead),
                            auxyRemRun, auxyRemRunN,
                            *tempMb, tempMb->overallHashMemory, lvRun, 1,
                            StrSpan(auxyImplication),
                            parameters.inductionMaxAdmissionDepth,
                            parameters.inductionMaxSecondaryNumber,
                            true,
                            parameters.minNumOperatorsKey,
                            StrSpan("implication", 11), true, StrSpan(auxyImplication), StrSpan("main", 4));


                        // add (in2[rec,...]) to temp block
                        {
                            const int lvRun2[1] = { tempMb->level };
							const TransientOrigin or1{ true, OriginTag::recursion, nullptr, 0 };
                            this->addExprToMemoryBlock(tempExpr2, *tempMb, iteration, 0, lvRun2, 1, or1, -1, -1, StrSpan("main", 4), false);
                        }

                        // queue head as "to be proved" with auxy index
                        this->addExprToMemoryBlock(head,
                                                   *tempMb,
                                                   iteration,
                                                   2,
                                                   nullptr, 0,
                                                   TransientOrigin{},
							                       -1,     
                                                   dependencyTable.auxyIndex,
                                                   StrSpan("main", 4),
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

                    if (simpleMapStore.findChild(memoryBlock, tempExpr3) == nullptr) {
                        Memory* tempMb2 = lbStore.create<Memory>();
                        simpleMapStore.linkChild(memoryBlock, tempExpr3, tempMb2);

                        tempMb2->level = static_cast<int>(index + 1);
                        tempMb2->parentMemory = memoryBlock;
                        // _induction_<recN>_ marker: recursion-block-#2
                        // instances across DIFFERENT induction triads share
                        // the same substituted equality (distinct routing
                        // keys (=[s(recN),zero]), identical exprKey after
                        // substitution — Rule 12), making coexisting LBs
                        // chain-identical; the full chain is the LB identity
                        // and the deload file identity. The suffix carries
                        // the per-triad recArg so every instance is disjoint.
                        // Parsers are unaffected (getArgs reads only the
                        // first [...] window); the routing key
                        // (tempExpr3) and the recursion hypothesis keep the
                        // unmarked form.
                        tempMb2->setExprKey(
                            tempExpr4 + "_induction_" + recArg + "_");
                        tempMb2->recursionHypothesisId =
                            tempMb2->nameMap.encode(tempExpr4);
                        tempMb2->isPartOfRecursion = true;

                        // Register toBeProved (status=2) BEFORE adding expression (status=0).
                        // When head == tempExpr4 (e.g. companion theorem where head is (=[7,2])),
                        // the status=0 add puts it in intKnownStatements, which would cause
                        // the status=2 call to return early without creating the toBeProved entry.
                        this->addExprToMemoryBlock(head,
                                                   *tempMb2,
                                                   iteration,
                                                   2,
                                                   nullptr, 0,
                                                   TransientOrigin{},
                                                   -1,
                                                   dependencyTable.auxyIndex,
                                                   StrSpan("main", 4),
                                                   false);

                        {
                            const int lvRun4[1] = { tempMb2->level };
							const TransientOrigin or2{ true, OriginTag::recursion, nullptr, 0 };
                            this->addExprToMemoryBlock(tempExpr4, *tempMb2, iteration, 0, lvRun4, 1, or2, -1, -1, StrSpan("main", 4), false);
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
                const StatementFlags* headRow = lookupStatementFlags(
                    memoryBlock->intKnownStatements, memoryBlock->nameMap, head, "main");
                if (headRow && headRow->registered) {
                    // Head already derived (e.g. from anchor disintegration) — register directly
                    std::string fullTheorem = reconstructImplication(chain, head);
                    this->appendGlobalTheorem(fullTheorem, "direct", "-1",
                        "-1", memoryBlock);
                    std::cout << fullTheorem << std::endl;
                } else {
                    this->addExprToMemoryBlock(head, *memoryBlock, iteration, 2, nullptr, 0, TransientOrigin{}, -1, -1, StrSpan("main", 4), false);
                }
            }

            // Antisymmetry-shaped conjecture: deposit the dead-end
            // variable-copy axiom (=[X,X_copy]) at this innermost LB so
            // the equivalence class can generate the all-distinct
            // statement variants the exact request match needs
            // (D-235). Family sibling rows
            // sharing this LB re-run the deposit; the door dedups.
            {
                std::string copyVar;
                if (detectAntisymmetryCopyVar(chain, head, copyVar)) {
                    const std::string copyEquality =
                        "(=[" + copyVar + "," + copyVar + "_copy])";
                    const TransientOrigin copyOrigin{
                        true, OriginTag::variableCopy, nullptr, 0 };
                    const int lvRunCopy[1] = { memoryBlock->level };
                    this->addExprToMemoryBlock(StrSpan(copyEquality),
                        *memoryBlock, iteration, 0, lvRunCopy, 1,
                        copyOrigin, -1, -1, StrSpan("main", 4), false);

                    // LB-creation paired mailOut write — the same timing
                    // exception as the premise deposit above: buildGrid's
                    // startup commit needs mailOut populated so
                    // descendants (the contradiction LBs created below)
                    // receive the axiom and its history line on their
                    // first phase-1 pull.
                    if (parameters.trackHistory) {
                        const int maxOriginsLocal = parameters.compressor_mode
                            ? parameters.compressor_max_origins_per_expr
                            : parameters.max_origin_per_expr;
                        ExpressionWithValidity ev(copyEquality, "main");
                        memoryBlock->insertMailOutStatement(StrSpan(ev.original),
                            StrSpan(ev.validityName), lvRunCopy, 1);
                        memoryBlock->addMailOutOrigin(StrSpan(ev.original),
                            StrSpan(ev.validityName), copyOrigin.tag,
                            copyOrigin.deps, copyOrigin.depN, maxOriginsLocal);
                    }
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
                        const ce::CoreExpressionConfig* cfg = coreConfig(StrSpan(opName));
                        if (cfg != nullptr && !cfg->outputIndices.empty()) {
                            int outIdx = cfg->outputIndices[0];
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
                                        simpleMapStore.findChild(predecessor, contradictionKey) == nullptr) {

                                        Memory* contradictionLB = lbStore.create<Memory>();
                                        simpleMapStore.linkChild(predecessor, contradictionKey, contradictionLB);

                                        contradictionLB->parentMemory = predecessor;
                                        contradictionLB->level = static_cast<int>(index);
                                        contradictionLB->setExprKey(contradictionKey);
                                        contradictionLB->primedForContradiction = true;
                                        contradictionLB->contradictionTheoremId =
                                            contradictionLB->nameMap.encode(contrTheorem);

                                        this->permanentBodies.push_back(contradictionLB);

                                        // Seed with clean operator expression as fuel
                                        const int lvRun[1] = { contradictionLB->level };
                                        const TransientOrigin origin{ true, OriginTag::taskFormulation, nullptr, 0 };
                                        this->addExprToMemoryBlock(cleanOp, *contradictionLB, iteration, 0, lvRun, 1, origin, -1, -1, StrSpan("main", 4), false);

                                        // LB-creation paired mailOut write — see comment at the
                                        // chain-walk LB-creation site (~line 4618) for rationale.
                                        // Documented exception to fillMailOut sole-writer invariant.
                                        if (parameters.trackHistory) {
                                            const int maxOriginsLocal = parameters.compressor_mode
                                                ? parameters.compressor_max_origins_per_expr
                                                : parameters.max_origin_per_expr;
                                            ExpressionWithValidity ev(cleanOp, "main");
                                            contradictionLB->insertMailOutStatement(
                                                StrSpan(ev.original),
                                                StrSpan(ev.validityName), lvRun, 1);
                                            contradictionLB->addMailOutOrigin(
                                                StrSpan(ev.original),
                                                StrSpan(ev.validityName), origin.tag,
                                                origin.deps, origin.depN,
                                                maxOriginsLocal);
                                        }
                                    }
                                    reformulatedOperator = true;
                                }
                            }
                        }
                    }
                }

                if (!reformulatedOperator) {
                    const std::string contradictionKey = "__contradiction__" + head;
                    if (simpleMapStore.findChild(memoryBlock, contradictionKey) == nullptr) {
                        Memory* contradictionLB = lbStore.create<Memory>();
                        simpleMapStore.linkChild(memoryBlock, contradictionKey, contradictionLB);

                        contradictionLB->parentMemory = memoryBlock;
                        contradictionLB->level = static_cast<int>(index + 1);
                        contradictionLB->setExprKey(contradictionKey);
                        contradictionLB->primedForContradiction = true;
                        contradictionLB->contradictionTheoremId =
                            contradictionLB->nameMap.encode(
                                reconstructImplication(chain, negate(head)));

                        this->permanentBodies.push_back(contradictionLB);

                        // Add the head expression as fuel (status 0)
                        const int lvRun[1] = { contradictionLB->level };
                        const TransientOrigin origin{ true, OriginTag::taskFormulation, nullptr, 0 };
                        this->addExprToMemoryBlock(head, *contradictionLB, iteration, 0, lvRun, 1, origin, -1, -1, StrSpan("main", 4), false);

                        // LB-creation paired mailOut write — see comment at the
                        // chain-walk LB-creation site (~line 4618) for rationale.
                        // Documented exception to fillMailOut sole-writer invariant.
                        if (parameters.trackHistory) {
                            const int maxOriginsLocal = parameters.compressor_mode
                                ? parameters.compressor_max_origins_per_expr
                                : parameters.max_origin_per_expr;
                            ExpressionWithValidity ev(head, "main");
                            contradictionLB->insertMailOutStatement(
                                StrSpan(ev.original), StrSpan(ev.validityName),
                                lvRun, 1);
                            contradictionLB->addMailOutOrigin(
                                StrSpan(ev.original), StrSpan(ev.validityName),
                                origin.tag, origin.deps, origin.depN,
                                maxOriginsLocal);
                        }

                    }
                }
            }

            // --- Incubator: complementary contradiction LB (reductio) ---
            // Assumes the NEGATION of the head; a contradiction then proves
            // the conjecture itself. Dual of the try_contradiction LB above,
            // which assumes the head verbatim and can only ever disprove it.
            if (parameters.try_contradiction_negated_head && !proved) {
                const std::string negHead = negate(head);
                const std::string contradictionKey = "__contradiction__" + negHead;
                if (simpleMapStore.findChild(memoryBlock, contradictionKey) == nullptr) {
                    Memory* contradictionLB = lbStore.create<Memory>();
                    simpleMapStore.linkChild(memoryBlock, contradictionKey, contradictionLB);

                    contradictionLB->parentMemory = memoryBlock;
                    contradictionLB->level = static_cast<int>(index + 1);
                    contradictionLB->setExprKey(contradictionKey);
                    contradictionLB->primedForContradiction = true;
                    contradictionLB->contradictionTheoremId =
                        contradictionLB->nameMap.encode(
                            reconstructImplication(chain, head));

                    this->permanentBodies.push_back(contradictionLB);

                    // Add the negated head as fuel (status 0)
                    const int lvRun[1] = { contradictionLB->level };
                    const TransientOrigin origin{ true, OriginTag::taskFormulation, nullptr, 0 };
                    this->addExprToMemoryBlock(negHead, *contradictionLB, iteration, 0, lvRun, 1, origin, -1, -1, StrSpan("main", 4), false);

                    // LB-creation paired mailOut write — see comment at the
                    // chain-walk LB-creation site for rationale. Documented
                    // exception to fillMailOut sole-writer invariant.
                    if (parameters.trackHistory) {
                        const int maxOriginsLocal = parameters.compressor_mode
                            ? parameters.compressor_max_origins_per_expr
                            : parameters.max_origin_per_expr;
                        ExpressionWithValidity ev(negHead, "main");
                        contradictionLB->insertMailOutStatement(
                            StrSpan(ev.original), StrSpan(ev.validityName),
                            lvRun, 1);
                        contradictionLB->addMailOutOrigin(
                            StrSpan(ev.original), StrSpan(ev.validityName),
                            origin.tag, origin.deps, origin.depN,
                            maxOriginsLocal);
                    }
                }
            }

        }
    }
}
// Turn optimizations OFF for just this section
//#pragma optimize("", off)



void ExpressionAnalyzer::revisitRejected2(StrSpan markedExpr,
    Memory& memoryBlock,
    StrSpan validityName)
{
    auto& rm = memoryBlock.overallHashMemory.rejectedMap;


    // Guard: prevent re-entrant processing of the same marker. Non-minting
    // probe — a never-interned template is never in progress.
    int64_t revisitPk = 0;
    const bool revisitKeyKnown = lookupTemplateKey(memoryBlock.templateInterner,
        memoryBlock.nameMap, markedExpr, validityName, revisitPk);
    if (revisitKeyKnown
        && memoryBlock.overallHashMemory.revisitInProgress.contains(revisitPk)) {
        return;
    }

    // Packed probe via the guard's non-minting lookup above — a
    // never-interned template has no rejection cohort.
    if (!revisitKeyKnown) {
        return;
    }
    const int32_t rmId = rm.lookup(revisitPk);
    if (rmId == 0) {
        return;
    }

    // [user-directed budget guard] Revival must respect the secondary-variable
    // budget; without a cnt cap it re-admits sums with unbounded secondary it_
    // vars (revival otherwise bypasses isAdmitted's gate entirely). markedExpr is
    // the marker form (the witness slot is "marker", NOT counted), so
    // cnt(markedExpr) < maxNumberSecondaryVariables(=2) is equivalent to the
    // revived constituent's cnt <= 2 — symmetric with the isAdmitted gate.
    // l+p' has markedExpr cnt=1 (just p'), so it still revives; a marked cnt>=2
    // (constituent cnt>=3) is no longer reanimated.
    if (countPatternOccurrences(markedExpr, memoryBlock.overallHashMemory, memoryBlock.nameMap)
        >= parameters.maxNumberSecondaryVariables) {
        return;
    }

    // [user-directed fix] it_depth control for revival. Revival otherwise
    // bypasses the iteration-depth bound entirely (the comment on the count
    // guard above notes the same for secondary count), re-admitting sums
    // whose it_ depth exceeds what the disintegration gate allows.
    // Parameterised via parameters.maxIterationNumberVariable (not
    // hardcoded). Strict >, aligned with the request-building gate in
    // filterIntEncodedStatements: generations up to and including the cap
    // revive. (Unlike the count guard above, the iteration quantity has no
    // marked-vs-constituent offset, so the comparison matches the request
    // side verbatim.)
    if (extractMaxIterationNumber(markedExpr) > parameters.maxIterationNumberVariable) {
        return;
    }

    memoryBlock.overallHashMemory.revisitInProgress.mint(
        mintTemplateKey(memoryBlock.templateInterner, memoryBlock.nameMap,
                        markedExpr, validityName));

    // Snapshot the rejected cohort as VERBATIM blob copies on the gen
    // scratch arena — copy BEFORE erase: eraseBlobIf restructures the blob
    // pool, so peeked pointers into `rm` dangle past it. The historical
    // operation order (snapshot -> erase -> process) is preserved exactly,
    // so the loop below still processes only the entries that existed
    // before this call (entries it adds indirectly via addExprToMemoryBlock
    // -> updateRejectedMap re-mint the key afresh). The cold run is stored
    // canonical under DecodedRejectedValueLess (the insertRejectedValue RMW
    // contract), so walking the copies in run order IS walking the former
    // RejectedValueSet snapshot — the set's dedup is a no-op on a
    // canonical run.
    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();

    struct BlobRec { ArenaOffset off; int32_t len; };
    DirtyState recsDirty = DirtyState::Clean;
    PagedVector<BlobRec> recs(&gArena, &recsDirty);
    const int32_t runLen = rm.runLen(rmId);
    for (int32_t j = 0; j < runLen; ++j) {
        int32_t bl = 0;
        const char* bp = rm.peekRecordBytes(rmId, j, bl, gArena);
        const ArenaOffset off = gArena.alloc(bl, 1);
        std::memcpy(gArena.resolve(off), bp, static_cast<std::size_t>(bl));
        recs.push_back(BlobRec{ off, bl });
    }

    // Remove the snapshot entries. At this single-threaded point the cold run
    // holds exactly the snapshot (nothing was added since the snapshot above),
    // so erasing them empties the run and the key is dropped — the heap path's
    // "preserve newly-added" branch was unreachable here too. Entries the
    // emission loop below adds via updateRejectedMap re-mint the key afresh.
    rm.eraseBlobIf([revisitPk](int64_t k) { return k == revisitPk; });

    // No-equality direct revival — the rejection record's cohort is mailed
    // verbatim to sameIterationInternalMail (the span/run door takes an empty
    // equalities array). The verifier-relevant `disintegration` origin for
    // each child was written at the original production site by
    // disintegrateExprCore2's trackExpansionHistory and is already present
    // in memoryBlock.exprOriginMap; the mail's `equality1` self-source
    // origin cannot displace it (addOrigin's cap-full preference protects
    // foundation origins from convenience tags — see D-49 / I-35).
    for (int32_t r = 0; r < recs.size(); ++r) {
        const BlobRec rec = recs[r];
        const RejectedValueBlobView view(gArena.resolve(rec.off), rec.len);

        // Zero-copy per-record span: the loop's only mutating sink is the
        // revival emission door, which mints nameMap + originInterner —
        // never valueInterner — and the walked blob bytes are gen-arena
        // copies, so valueInterner decodeView spans stay valid across the
        // whole loop (I-3 different-interner rule).
        const StrSpan rejectedStmt =
            memoryBlock.valueInterner.decodeView(view.renamedExpressionId());

        StrSpan markedArgs[ExecutionParameters::MAX_ARITY];
        const int markedN = getArgsSpans(markedExpr, markedArgs,
            ExecutionParameters::MAX_ARITY);
        StrSpan rejectedArgs[ExecutionParameters::MAX_ARITY];
        const int rejectedN = getArgsSpans(rejectedStmt, rejectedArgs,
            ExecutionParameters::MAX_ARITY);

        if (markedN != rejectedN) {
            continue;
        }

        int markerIndex = -1;
        for (int i = 0; i < markedN; ++i) {
            if (equalSpans(markedArgs[i], StrSpan("marker", 6))) {
                markerIndex = i;
                break;
            }
        }
        assert(markerIndex != -1);

        bool argsIdentical = true;
        for (int i = 0; i < markedN; ++i) {
            if (i == markerIndex) continue;
            if (!equalSpans(markedArgs[i], rejectedArgs[i])) {
                argsIdentical = false;
                break;
            }
        }
        assert(argsIdentical && "Mismatch between marked and rejected args (other than marker)");
        if (!argsIdentical) continue;

        // [user-directed fix] control the PREPARED WITNESS that jumps into
        // existence on revival. It sits at the marker slot of the rejected
        // statement (rejectedArgs[markerIndex]); markedExpr only carries the
        // "marker" placeholder, so the markedExpr-level guard above cannot see
        // it. Skip revival when the witness's iteration depth exceeds
        // parameters.maxIterationNumberVariable — strict >, aligned with the
        // markedExpr-depth guard and the request-building gate: generations
        // up to and including the cap revive.
        if (extractMaxIterationNumber(rejectedArgs[markerIndex]) > parameters.maxIterationNumberVariable) {
            continue;
        }

        // intStatementLevelsMap consistency check intentionally absent here —
        // the compact form's level entry may legitimately be missing when
        // the rejection was committed before the kernel's stmts-loop wrote
        // the compound's entry. The record's levels (captured at buffer time,
        // may be empty) are the authoritative deposit-time levels set.

        // Deposit the stored cohort onto sameIterationInternalMail for absorb
        // at the next hashburst. pre == post (no rewrite participated here);
        // the equalities array is empty. The 64 cap mirrors the door's own
        // OriginDep[64] bound; a firing assert is a gift (Rule 19).
        const StrSpan valConcrete =
            memoryBlock.valueInterner.decodeView(view.concreteConstituentId());
        const int32_t sibN = view.siblingCount();
        assert(sibN <= 64 && "revival sibling cohort exceeds 64");
        StrSpan sibBuf[64];
        for (int32_t i = 0; i < sibN; ++i) {
            sibBuf[i] = memoryBlock.valueInterner.decodeView(view.siblingId(i));
        }
        const int32_t levelCount = view.levelCount();
        int* levelRun = nullptr;
        if (levelCount > 0) {
            levelRun = reinterpret_cast<int*>(gArena.resolve(gArena.alloc(
                levelCount * static_cast<int32_t>(sizeof(int)), alignof(int))));
            view.copyLevels(levelRun);
        }
        this->emitIntegrationRevivalToInternalMailIn(
            valConcrete, valConcrete,
            sibBuf, sibN, sibBuf, sibN,
            levelRun, levelCount,
            validityName,
            nullptr, 0,
            memoryBlock);
    }

    gArena.popTo(mark);

    {
        int64_t donePk = 0;
        const bool ok = lookupTemplateKey(memoryBlock.templateInterner,
            memoryBlock.nameMap, markedExpr, validityName, donePk);
        assert(ok); // inserted above — both halves are interned
        memoryBlock.overallHashMemory.revisitInProgress.erase(donePk);
    }

    cleanAdmissionMap(markedExpr, validityName, memoryBlock);
}


// Integration-side counterpart to revisitRejected2. Walks
// rejectedMapIntegration[markedKey, validityName] and, for each stored
// entry, emits the constituent + siblings onto sameIterationInternalMail so the next
// hashburst re-runs the full disintegration pipeline at the constituent's
// original validity. Unlike revisitRejected2:
//   * no addExprToMemoryBlock call (mailIn-only revival, linear),
//   * no cleanAdmissionMap / admission-key erasure (user-specified —
//     integration revival keeps the admission rule for future hits),
//   * no revisitInProgress guard (no re-entry through addStatement).
// The rejectedMapIntegration entry itself IS erased — the rejection is
// resolved (the constituents have been handed to the revival channel).
void ExpressionAnalyzer::revisitRejectedIntegration2(StrSpan markedKey,
    Memory& memoryBlock,
    StrSpan validityName) {

    auto& rmi = memoryBlock.overallHashMemory.rejectedMapIntegration;
    // Non-minting probe — a never-interned template has no rejection cohort.
    int64_t markedPk = 0;
    if (!lookupTemplateKey(memoryBlock.templateInterner, memoryBlock.nameMap,
                           markedKey, validityName, markedPk)) {
        return;
    }
    // Non-minting presence probe — 0 means no rejection cohort.
    if (rmi.lookup(markedPk) == 0) return;

    // Snapshot the rejected-integration cohort as VERBATIM blob copies + views
    // on the gen-scratch arena — copy BEFORE erase: eraseBlobIf restructures the
    // blob pool so peeked pointers into `rmi` dangle past it. The cold run is
    // stored canonical under DecodedRejectedIntegrationValueLess (the
    // insertRejectedIntegrationValue RMW contract), so walking the views in run
    // order IS walking the former RejectedIntegrationValueSet snapshot — the
    // set's dedup is a no-op on a canonical run. The revisitRejected2 shape:
    // snapshot -> erase -> process -> popTo. The emission loop below never reads
    // rmi (it deposits onto sameIterationInternalMail), so erasing before the
    // loop is behaviour-identical to the former erase-after order.
    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();
    const RejectedIntegrationRunSnapshot snap =
        snapshotRejectedIntegrationRun(rmi, markedPk, gArena);

    rmi.eraseBlobIf([markedPk](int64_t k) { return k == markedPk; });

    for (int32_t vi = 0; vi < snap.count; ++vi) {
        const RejectedIntegrationValueBlobView& val = snap.views[vi];
        // Zero-copy per-record spans: the loop's only mutating sink is the
        // revival emission door, which mints nameMap + originInterner —
        // never valueInterner — and the walked blob bytes are gen-arena
        // copies, so valueInterner decodeView spans stay valid across the
        // whole loop (I-3 different-interner rule; the revisitRejected2
        // precedent).
        const StrSpan valConcrete =
            memoryBlock.valueInterner.decodeView(val.concreteConstituentId());
        const int32_t sibN = val.siblingCount();
        assert(sibN <= 64 && "integration revival sibling cohort exceeds 64");
        StrSpan sibBuf[64];
        for (int32_t i = 0; i < sibN; ++i)
            sibBuf[i] = memoryBlock.valueInterner.decodeView(val.siblingId(i));

        // Caller-owned stack copy of the compound's level run — never a
        // pointer into the cold value column (I-136).
        const StrSpan valCompound =
            memoryBlock.valueInterner.decodeView(val.compoundExpressionId());
        int lvRun[256];
        int32_t lvN = 0;
        const int32_t compoundLvlsId = lookupStatementLevels(
            memoryBlock.intStatementLevelsMap, memoryBlock.nameMap,
            valCompound, validityName);
        if (compoundLvlsId)
            lvN = coldIntRunAt(memoryBlock.intStatementLevelsMap,
                               compoundLvlsId, lvRun, 256);

        // Direct-revisit path: no rewrite happened, so pre == post. For
        // equality1 origin, origin.second[0] still needs to be a sensible
        // source — passing the constituent itself lets the verifier walk
        // it as both source and result (with an empty equalities list).
        this->emitIntegrationRevivalToInternalMailIn(
            valConcrete, valConcrete,
            sibBuf, sibN, sibBuf, sibN,
            lvRun, lvN,
            validityName,
            nullptr, 0,
            memoryBlock);
    }

    gArena.popTo(mark);
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


void ExpressionAnalyzer::updateAdmissionMapRecursion(StrSpan expression,
    Memory& mb,
    StrSpan validityName)
{
    // 1. Initial Guard Clauses: Check Operator. The core name is a StrSpan slice
    // of `expression` (extractExpressionSpan is the byte-exact twin of
    // ce::extractExpression) — it keys `operators`
    // (std::set<std::string, std::less<>>, transparent) and the
    // compiled-definition layer through the coreConfig reader fence, so no
    // std::string is materialized on this path at all
    // (I-137).
    const StrSpan coreSpan = extractExpressionSpan(expression);
    if (this->operators.find(std::string_view(coreSpan.ptr,
            static_cast<std::size_t>(coreSpan.len))) == this->operators.end()) {
        return;
    }

    const ce::CoreExpressionConfig* cfg = coreConfig(coreSpan);
    assert(cfg && "updateAdmissionMapRecursion: operator core must be compiled");
    assert(!cfg->outputIndices.empty());

    // 2. Identify Output Argument. The statement arg-parse rides the span twin
    // (getArgsSpans == ce::getArgs, byte-identical); the one output arg is a
    // StrSpan slice of `expression` (a caller-stable buffer disjoint from the
    // value / template / NameMap interners minted below, so it survives every
    // mint — I-3). It feeds the replacementMap substitution + newRemainingArgs
    // membership/encode as spans, no heap std::string (row 41/50,
    // I-138 L9 Item 2).
    StrSpan argSpans[ExecutionParameters::MAX_ARITY];
    const int32_t argN = getArgsSpans(expression, argSpans,
                                      ExecutionParameters::MAX_ARITY);
    int outIdx = cfg->outputIndices[0];

    assert(outIdx >= 0 && outIdx < argN);

    const StrSpan inputOutputArg = argSpans[outIdx];

    // 3. Prepare Lookup. The marker-form template key rides the per-slot
    // BYTE-scratch arena — replaceKeysScratch is the byte-exact twin of
    // ce::replaceKeysInString (greedy-longest, token-boundary), so markedExpr's
    // bytes reproduce the former heap std::string. The ScratchScope spans to
    // function end because the convergence compare (testExpr vs markedExpr) reads
    // it deep in the value loop; the arena is disjoint from the template / NameMap
    // / value interners minted below, and nested callee scratch scopes rewind LIFO
    // above it, so markedExpr stays valid to its last use.
    const unsigned strSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(strSlot);
    ScratchScope markedScope(strArena);
    StrReplacement markerPair[1];
    markerPair[0].key = inputOutputArg;
    markerPair[0].value = StrSpan("marker", 6);
    const ScratchString markedExpr =
        replaceKeysScratch(strArena, expression, markerPair, 1);

    // The historical EWV map created a default-false status entry on this probe
    // and the dump's status section counts it — the cold map reproduces that
    // byte-for-byte: a miss inserts the default-false entry, then returns
    // (D-172).
    const int64_t markedPk = mintTemplateKey(mb.templateInterner, mb.nameMap,
                                             StrSpan(markedExpr), validityName);
    {
        const uint8_t* stProbe =
            mb.overallHashMemory.admissionStatusMap.find(markedPk);
        if (stProbe == nullptr) {
            mb.overallHashMemory.admissionStatusMap.upsert(
                markedPk, static_cast<uint8_t>(0));
            return;
        }
        if (*stProbe == 0) {
            return;
        }
    }

    // 4. Lookup in Admission Map
    const int32_t admPropId = mb.overallHashMemory.admissionMap.lookup(markedPk);

    // Guard: Return if not found
    if (admPropId == 0) {
        return;
    }

    // 5. Iterate over Admission Values (Snapshot). The cold run is kept sorted
    // by DecodedAdmissionValueLess, so the snapshot order matches the former
    // heap set. Copy the run's blobs verbatim onto gen-scratch BEFORE the loop:
    // insertAdmissionIdsBlob below mutates admissionMap mid-loop, so a live peek
    // would dangle; the blob copies survive it (snapshotAdmissionRun, 09b
    // pitfall 6, I-99). admissionRecordsAt stays the retained oracle.
    const unsigned admSnapSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& admSnapArena = genScratchArenas().forSlot(admSnapSlot);
    const ArenaOffset admSnapMark = admSnapArena.cursor();
    const AdmissionRunSnapshot admPropSnap = snapshotAdmissionRun(
        mb.overallHashMemory.admissionMap, markedPk, admSnapArena);

    // The per-value and per-match gen-tier index runs share admSnapArena's
    // per-slot genScratchArena instance: they sit ABOVE the snapshot on the
    // byte-bump tier, so the LIFO scopes reclaim them without disturbing it.
    ScratchArena& genArena = admSnapArena;

    for (int32_t admValIx = 0; admValIx < admPropSnap.count; ++admValIx) {
        const AdmissionValueBlobView& val = admPropSnap.views[admValIx];
        // The value loop MINTS valueInterner (encode + insertAdmissionIdsBlob
        // grow its cold pages), so the value's decoded key elements AND its
        // remaining-arg members are COPIED ONCE onto the string arena and held
        // for the whole match walk — a raw decodeView span would dangle at the
        // first later mint from an EARLIER matching keyElem's insert (I-3; the
        // S9 keyVec/remVec precedent). The span index arrays ride the gen tier
        // ABOVE the admission snapshot; valStrScope/valGenScope reclaim them per
        // value.
        ScratchScope valStrScope(strArena);
        ScratchScope valGenScope(genArena);

        const int32_t valKeyN = val.keyCount();
        StrSpan* valKeySpans = (valKeyN == 0) ? nullptr
            : reinterpret_cast<StrSpan*>(genArena.resolve(genArena.alloc(
                  valKeyN * static_cast<int32_t>(sizeof(StrSpan)),
                  static_cast<int32_t>(alignof(StrSpan)))));
        for (int32_t e = 0; e < valKeyN; ++e) {
            const StrSpan dv = mb.valueInterner.decodeView(val.keyId(e));
            valKeySpans[e] = StrSpan(ScratchString::copyFrom(strArena, dv.ptr, dv.len));
        }

        // valRemaining: the former std::set<std::string> becomes a sorted-unique
        // StrSpan membership run (members copyFrom on the string arena, index on
        // the gen tier). std::set == sorted-unique, so sort + adjacent-dedup by
        // compareSpans reproduces the set content AND order EXACTLY; the two
        // membership probes become std::binary_search(compareSpans).
        const int32_t valRemRawN = val.remCount();
        StrSpan* valRemaining = (valRemRawN == 0) ? nullptr
            : reinterpret_cast<StrSpan*>(genArena.resolve(genArena.alloc(
                  valRemRawN * static_cast<int32_t>(sizeof(StrSpan)),
                  static_cast<int32_t>(alignof(StrSpan)))));
        for (int32_t r = 0; r < valRemRawN; ++r) {
            const StrSpan dv = mb.valueInterner.decodeView(val.remId(r));
            valRemaining[r] = StrSpan(ScratchString::copyFrom(strArena, dv.ptr, dv.len));
        }
        std::sort(valRemaining, valRemaining + valRemRawN,
            [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) < 0; });
        int32_t valRemN = 0;
        for (int32_t r = 0; r < valRemRawN; ++r) {
            if (valRemN == 0
                || !equalSpans(valRemaining[valRemN - 1], valRemaining[r]))
                valRemaining[valRemN++] = valRemaining[r];
        }

        for (int32_t keIx = 0; keIx < valKeyN; ++keIx) {
            const StrSpan keyElem = valKeySpans[keIx];

            // --- Flattened Logic Start ---

            const StrSpan kCoreSpan = extractExpressionSpan(keyElem);

            // Skip non-operators
            if (this->operators.find(std::string_view(kCoreSpan.ptr,
                    static_cast<std::size_t>(kCoreSpan.len))) == this->operators.end()) continue;

            const ce::CoreExpressionConfig* kCfg = coreConfig(kCoreSpan);
            assert(kCfg && "updateAdmissionMapRecursion: key operator core must be compiled");
            assert(!kCfg->outputIndices.empty());

            StrSpan kArgs[ExecutionParameters::MAX_ARITY];
            const int32_t kArgsN = getArgsSpans(keyElem, kArgs,
                                                ExecutionParameters::MAX_ARITY);
            int kOutIdx = kCfg->outputIndices[0];

            assert(kOutIdx >= 0 && kOutIdx < kArgsN);

            // kOutputArg is a slice of keyElem (a stable string-arena copy).
            const StrSpan kOutputArg = kArgs[kOutIdx];

            // Condition: Output arg must NOT be in remaining args (binary_search
            // over the sorted-unique run == the former std::set membership).
            if (std::binary_search(valRemaining, valRemaining + valRemN, kOutputArg,
                    [](const StrSpan& a, const StrSpan& b) {
                        return compareSpans(a, b) < 0;
                    })) continue;

            // Per-keyElem string window: testExpr and (on a match) the newKey
            // rewrites + newMarkedExpr live here, freed at keyElem end.
            ScratchScope keScope(strArena);

            // Check for Match. Single-key marker map -> StrReplacement[1];
            // replaceKeysScratch is the byte-exact twin of ce::replaceKeysInString
            // (greedy-longest, token-boundary), so testExpr is byte-identical.
            StrReplacement kMarkerPair[1];
            kMarkerPair[0].key = kOutputArg;
            kMarkerPair[0].value = StrSpan("marker", 6);
            const ScratchString testExpr =
                replaceKeysScratch(strArena, keyElem, kMarkerPair, 1);

            if (!equalSpans(StrSpan(testExpr), StrSpan(markedExpr))) continue;

            // --- Match Found ---

            mb.overallHashMemory.productsOfRecursionIds.mint(mb.nameMap.encode(inputOutputArg));

            // 6. Construct New Admission Entry.
            // replacementMap {kOutputArg -> inputOutputArg} as a stack
            // StrReplacement[1]; replaceKeysScratch is the byte-exact twin of
            // ce::replaceKeysInString (single key, greedy-longest, token-boundary).
            // newKey is a StrSpan run over per-element rewrites on the string
            // arena; each element is valueInterner.encode'd at the insert below.
            StrReplacement replPair[1];
            replPair[0].key = kOutputArg;
            replPair[0].value = inputOutputArg;

            // Match-level gen-tier window: the newKey / newRemaining span arrays.
            ScratchScope matchGenScope(genArena);
            StrSpan* newKeySpans = (valKeyN == 0) ? nullptr
                : reinterpret_cast<StrSpan*>(genArena.resolve(genArena.alloc(
                      valKeyN * static_cast<int32_t>(sizeof(StrSpan)),
                      static_cast<int32_t>(alignof(StrSpan)))));
            for (int32_t k = 0; k < valKeyN; ++k)
                newKeySpans[k] = StrSpan(
                    replaceKeysScratch(strArena, valKeySpans[k], replPair, 1));
            const int32_t newKeyN = valKeyN;

            // newRemainingArgs = distinct(valRemaining ∪ {inputOutputArg}) as a
            // sorted StrSpan run on the byte-bump gen-scratch tier (the
            // createAuxyImplication untouchables recipe, row 41). Spans alias the
            // valRemaining string-arena copies + inputOutputArg (a slice of the
            // caller-stable `expression`), both alive to the match's end and
            // disjoint from the value interner minted below (I-3). Membership rides
            // std::binary_search(compareSpans); the sorted-encode is a direct
            // in-order valueInterner.encode loop at the insert site below,
            // byte-identical to the former std::set encodeValueSetSorted (the
            // run is sorted-unique).
            const int32_t newRemCap = valRemN + 1;
            StrSpan* newRemaining = reinterpret_cast<StrSpan*>(genArena.resolve(
                genArena.alloc(
                    newRemCap * static_cast<int32_t>(sizeof(StrSpan)),
                    static_cast<int32_t>(alignof(StrSpan)))));
            int32_t newRemainingN = 0;
            for (int32_t r = 0; r < valRemN; ++r)
                newRemaining[newRemainingN++] = valRemaining[r];
            newRemaining[newRemainingN++] = inputOutputArg;
            std::sort(newRemaining, newRemaining + newRemainingN,
                [](const StrSpan& a, const StrSpan& b) {
                    return compareSpans(a, b) < 0;
                });
            {
                int32_t w = 0;
                for (int32_t r = 0; r < newRemainingN; ++r) {
                    if (w == 0
                        || !equalSpans(newRemaining[w - 1], newRemaining[r])) {
                        newRemaining[w++] = newRemaining[r];
                    }
                }
                newRemainingN = w;
            }

            // 7. Validate New Key Candidates
            for (int32_t nkIx = 0; nkIx < newKeyN; ++nkIx) {
                const StrSpan newKeyElem = newKeySpans[nkIx];
                const StrSpan newCoreSpan = extractExpressionSpan(newKeyElem);

                if (this->operators.find(std::string_view(newCoreSpan.ptr,
                        static_cast<std::size_t>(newCoreSpan.len))) == this->operators.end()) continue;

                const ce::CoreExpressionConfig* newCfg = coreConfig(newCoreSpan);
                assert(newCfg && "updateAdmissionMapRecursion: new key operator core must be compiled");
                assert(!newCfg->outputIndices.empty());

                StrSpan newKArgs[ExecutionParameters::MAX_ARITY];
                const int32_t newKArgsN = getArgsSpans(newKeyElem, newKArgs,
                                                       ExecutionParameters::MAX_ARITY);
                int newOutIdx = newCfg->outputIndices[0];

                assert(newOutIdx >= 0 && newOutIdx < newKArgsN);

                const StrSpan newOutputArg = newKArgs[newOutIdx];

                // Condition: New output NOT in new remaining args (binary_search
                // over the sorted StrSpan run == the former std::set membership).
                if (std::binary_search(newRemaining, newRemaining + newRemainingN,
                        newOutputArg,
                        [](const StrSpan& a, const StrSpan& b) {
                            return compareSpans(a, b) < 0;
                        })) continue;

                // Condition: All other inputs MUST be in new remaining args.
                bool allInputsPresent = true;
                for (int i = 0; i < newKArgsN; ++i) {
                    if (i == newOutIdx) continue;
                    if (!std::binary_search(newRemaining,
                            newRemaining + newRemainingN, newKArgs[i],
                            [](const StrSpan& a, const StrSpan& b) {
                                return compareSpans(a, b) < 0;
                            })) {
                        allInputsPresent = false;
                        break;
                    }
                }

                if (!allInputsPresent) continue;

                // 8. Insert New Rule. Single-key marker map -> StrReplacement[1]
                // (replaceKeysScratch byte-exact twin of ce::replaceKeysInString).
                StrReplacement newMarkerPair[1];
                newMarkerPair[0].key = newOutputArg;
                newMarkerPair[0].value = StrSpan("marker", 6);
                const ScratchString newMarkedExpr =
                    replaceKeysScratch(strArena, newKeyElem, newMarkerPair, 1);

                // depth/sec/flag come from the matched snapshot record 'val'
                // (read-side, stays). The two SEQUENCED mints stay ORDERED:
                // encodeValueVector(newKey) FIRST, then the sorted-unique
                // newRemaining — a direct in-order valueInterner.encode loop is
                // encodeValueSpanSetSorted's byte-identical twin (no scratch
                // dedup set), and no owning value object is built.
                const ArenaOffset bMark = genArena.cursor();
                const int32_t bKeyCount = newKeyN;
                int32_t* bKeyIds = (bKeyCount > 0)
                    ? reinterpret_cast<int32_t*>(genArena.resolve(
                          genArena.alloc(bKeyCount * sizeof(int32_t), alignof(int32_t))))
                    : nullptr;
                for (int32_t j = 0; j < bKeyCount; ++j)
                    bKeyIds[j] = mb.valueInterner.encode(newKeySpans[j]);
                int32_t* bRemIds = (newRemainingN > 0)
                    ? reinterpret_cast<int32_t*>(genArena.resolve(
                          genArena.alloc(newRemainingN * sizeof(int32_t), alignof(int32_t))))
                    : nullptr;
                for (int32_t j = 0; j < newRemainingN; ++j)
                    bRemIds[j] = mb.valueInterner.encode(newRemaining[j]);

                // consumed-skip (mirror drainAdmissionKeysAlgebra): never re-add a
                // consumed key to admissionMap (would violate isAdmitted's
                // admissionMap/consumedAdmissionKeys mutual-exclusion assert).
                const int64_t newMarkedPk = mintTemplateKey(mb.templateInterner,
                    mb.nameMap, StrSpan(newMarkedExpr), validityName);
                if (mb.overallHashMemory.consumedAdmissionKeys.contains(newMarkedPk)) {
                    genArena.popTo(bMark);
                    continue;
                }

                insertAdmissionIdsBlob(mb.overallHashMemory.admissionMap,
                                       newMarkedPk, val.depth(),
                                       val.sec(), val.flagByte() != 0,
                                       bKeyIds, bKeyCount, bRemIds, newRemainingN,
                                       mb.valueInterner, genArena);
                genArena.popTo(bMark);
                const uint8_t srcStatusVal =
                    mb.overallHashMemory.admissionStatusMap.findOr(
                        markedPk, static_cast<uint8_t>(0));
                mb.overallHashMemory.admissionStatusMap.upsert(
                    newMarkedPk, srcStatusVal);

                // Populate vars-in-keys cache for the algebra equi-class hook.
                StrSpan nmArgs[ExecutionParameters::MAX_ARITY];
                const int32_t nmArgsN = getArgsSpans(StrSpan(newMarkedExpr), nmArgs,
                                                     ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < nmArgsN; ++a) {
                    if (!equalSpans(nmArgs[a], StrSpan("marker", 6))) {
                        mb.overallHashMemory.varsInAdmissionMapKeys.mint(mb.templateInterner.encode(nmArgs[a]));
                    }
                }

                // revisitRejected2 now takes StrSpan validityName; pass the span
                // this scope already holds (no per-match std::string copy).
                this->revisitRejected2(StrSpan(newMarkedExpr), mb, validityName);
            }
        }
    }
    admSnapArena.popTo(admSnapMark);
}

//#pragma optimize("", off)


// loadFactsForCEFiltering(), addConjectureForCEFiltering(),
// releaseCEBatchMemory() — moved to filter.cpp.


// filterConjecturesWithCE() — moved to filter.cpp.






void ExpressionAnalyzer::activateZeroCondition(Memory& memoryBlock)
{
    std::vector<std::string> args0 = ce::getArgs(memoryBlock.exprKey());
    std::string zeroArgName = this->findZeroArgName(memoryBlock);
    std::string eqExpr = std::string("(=[s(") + args0[0] + ")," + zeroArgName + "])";
    if (memoryBlock.parentMemory != NULL) {
        Memory* zeroChild = simpleMapStore.findChild(memoryBlock.parentMemory, eqExpr);
        if (zeroChild) {
            // The wake target is a block parked at birth that never
            // entered an active snapshot — waking a DISCHARGED block
            // (one that ran and deactivated) would resurrect frozen
            // content (I-112).
            assert(!zeroChild->dischargedForever
                && "activateZeroCondition on a discharged LB - "
                   "deactivation is permanent");
            zeroChild->isActive = true;
            // WAKE DOOR 6 / case F (D-194): this births an
            // induction zero block into the active set bypassing every mailbox,
            // so it must be marked dirty to be swept its first burst.
            zeroChild->hasWork = true;
            // First active burst processes a full backlog with no prior submatch
            // history — split it preemptively this iteration (D-201).
            zeroChild->justActivated = true;
        }
    }
}

/// @brief Drive one outer iteration over every LB in `bodies` — the
/// prover's main parallel hash-burst loop.
///
/// @details
/// Spawns a worker pool sized at `logicalCores`; the cycle runs as three
/// barriered phase sweeps (`performElemPhase1` -> join -> phase 2 -> join ->
/// `performElemPhase3` -> join), each a work-stealing sweep over the active LBs.
/// Phase 2 is the **flat (LB, part) executor pool** (one work-stealing pool over
/// every active LB's parts — a straggler contributes `logicalCores` expression
/// buckets, everything else one part) followed by a per-LB finalize
/// (`performElemPhase2`), in a <= 2-round loop: round 1 runs the producers +
/// unsplit bursts, round 2 runs the stragglers' buckets. The split is decided by
/// the end-of-iteration `isStraggler` stats pass (D-201).
/// Cross-LB state mutations are forbidden during the worker phase per
/// [I-28](../../docs/agentic_swdd/30_invariants.md#i-28); each worker writes only
/// to its own LB's local state and to the per-thread collectors
/// (`updateGlobalTuples`, the `updateGlobalDirectPages` sealed set, the
/// per-core-slot mail outboxes).
///
/// After `pool.join()`, the post-parallel collector phase runs:
/// 1. The `MailLog` commit barrier — appends each LB's `mailOut` as one
///    batch (statements + exprOriginMap) to its log, which descendants pull
///    at the next iteration's phase 1; the per-item validity assert per
///    [I-26](../../docs/agentic_swdd/30_invariants.md#i-26) runs at the
///    receiver-side absorb.
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
/// @invariant [I-26](../../docs/agentic_swdd/30_invariants.md#i-26),
///            [I-28](../../docs/agentic_swdd/30_invariants.md#i-28).
/// @see `performElemPhase1` / `performElemPhase2` / `performElemPhase3` — the per-LB phase workers.
/// @see `MailLog` — the per-LB pull logs that carry cross-LB mail (commit barrier here).
void ExpressionAnalyzer::proveKernel(const std::vector<Memory*>& bodies) {

    if (bodies.empty()) return;

    const unsigned workers = logicalCores;

    // Active set at this cycle's start. The barriered phases run phase 1 / 2 / 3
    // over exactly the LBs a per-LB loop would have stepped
    // (active when pulled): intra-cycle deactivation only flips active -> inactive
    // (a child's reactivation is in the post-join drain, I-28), and an LB never
    // reads another's state intra-cycle, so collecting the set once here matches
    // the old `if(!isActive) continue` admission. See D-114.
    std::vector<Memory*> active;
    active.reserve(bodies.size());
    // Quiescent-burst skip (D-194): sweep an active LB iff it
    // may produce work this burst. `skipEnabled` gates the whole filter off during
    // warm-up (belt), when the config disables it, and in compressor mode (LBs
    // unregistered in the mail log, so mailPeek would be invalid — gated out). An
    // eligible LB is skipped only when it has NO producer-side work (hasWork clear)
    // AND NO un-ingested ancestor mail (mailPeek false). mailPeek is polled fresh
    // each iteration, so a cross-LB commit at iteration N wakes the recipient at
    // the N+1 active-build — the identical iteration the reference would pull, so
    // no mail latency is added (I-55). The predicate reads ONLY never-deloaded
    // logical state (hasWork on the LB slab, mailPeek on the mail pool); it never
    // reads resident() / blocksInUse() — the I-106 / I-108 / D-149 determinism
    // doctrine. A skipped LB stays isActive and keeps its claim/pager state
    // untouched: it is unswept this round, not deactivated, so the pager's sweep
    // window never includes it and it evicts and stays evicted at 4 GiB.
    const bool skipEnabled = parameters.enable_quiesce_skip
        && !parameters.compressor_mode && !warmUpPhase;
    int skipped = 0;
    for (Memory* b : bodies) {
        if (!(b && b->isActive)) continue;
#ifdef QUIESCE_SHADOW_CHECK
        // Shadow: record the predicate's verdict but NEVER actually skip — the LB
        // is swept and performElemPhase3 asserts its "quiescent" burst was a no-op.
        b->shadowWouldSkip = skipEnabled && !b->hasWork && !mailLog.mailPeek(b);
        active.push_back(b);
#else
        if (skipEnabled && !b->hasWork && !mailLog.mailPeek(b)) { ++skipped; continue; }
        active.push_back(b);
#endif
    }
    lastSweptCount = static_cast<int>(active.size());
    lastSkippedCount = skipped;

    // Once-per-batch LB-size telemetry (this batch's first kernel barrier,
    // single-threaded): the active-LB count and the resident LBs' blocksHeld
    // distribution — raw inputs for later tuning constants (Rule 16 / I-44,
    // never a prover input). Gathered only on the first barrier via the guard.
    if (deloadStats().lbHistogramPending()) {
        std::vector<int64_t> blocksHeld;
        blocksHeld.reserve(active.size());
        for (Memory* b : active)
            if (b->lbMemory.manager.resident())
                blocksHeld.push_back(b->lbMemory.manager.blocksHeld());
        deloadStats().reportLbHistogram(static_cast<int>(active.size()),
                                        blocksHeld);
    }

    // Active-LB eviction is now CONTINUOUS — the steward drains the working
    // set in every phase window (the working-set pager,
    // D-161), not via a barrier-armed plan. No
    // kernel-entry arming step.

    // Fresh per-iteration staging page set for deferred cross-LB admission seeds
    // (I-28): emplace a Filling set BEFORE phase-1 (single-threaded), sealed +
    // freed + reset by the post-join drain below. Non-movable SealedPageSet in a
    // std::optional stays at a stable address so its SealedString views hold.
    deferredAncestorPages.emplace();
    deferredAncestorPages->bind(&staticMemory());

    // Fresh per-iteration staging page set for proven direct theorems (Part B of
    // the row-338 statification): dischargeToBeProved / dischargeContradiction
    // seal each theorem's bytes here so they cross the parallel -> join boundary
    // verbatim (the LB deloads after its burst); drained by
    // drainUpdateGlobalDirect below. Non-movable in a std::optional -> stable
    // address for its SealedString views.
    updateGlobalDirectPages.emplace();
    updateGlobalDirectPages->bind(&staticMemory());

    // Barriered phase model: phase 1 for ALL LBs, join, phase 2 for ALL, join,
    // phase 3 for ALL, join. Each phase is its own work-stealing sweep over the
    // active set; the join between phases is the barrier. This is byte-identical
    // to the per-LB performElem loop (phase 1/2/3 are pure-extraction helpers and
    // no cross-LB observation happens intra-cycle — all cross-LB mail is delayed
    // to the next cycle via the MailLog commit barrier + phase-1 pull). Phase 2 is the
    // hashburst — the phase where a heavy LB's split gives it N executor threads.
    // `runPhase` is a local generic-lambda helper (Rule 18 exempt).
    auto runPhase = [&active, workers](std::atomic<std::size_t>& next,
                                       auto&& body_fn) {
        auto worker = [&active, &next, workers, &body_fn](unsigned coreId) {
            const unsigned cid = workers ? (coreId % workers) : 0U;
            for (;;) {
                std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= active.size()) break;
                body_fn(*active[i], cid);
            }
            };
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (unsigned t = 0; t < workers; ++t) pool.emplace_back(worker, t);
        for (auto& th : pool) th.join();
        };

    // Phase 1 opens a working-set window over its dispatch cursor
    // (D-161, I-114): the steward
    // prefetches the upcoming LBs and, above the watermarks, drains the
    // deloadable rest. Every LB's handshake (claimAndLoadForWork) makes it
    // resident before its body and releases the claim after, so the steward
    // can reclaim it once done.
    {
        std::atomic<std::size_t> phase1Cursor{ 0 };
        steward->beginPhaseWindow(/*phase=*/1, &phase1Cursor, &active, workers,
                                  lbdeload::kDeloadDirectory);
        runPhase(phase1Cursor, [this](Memory& b, unsigned cid) {
            g_inParallelWorkerPhase = true;
            this->performElemPhase1(b, cid);
            g_inParallelWorkerPhase = false;
        });
        steward->endPhaseWindow();
    }

    // Phase 2 opens TWO working-set windows PER PASS
    // (D-196): an EXECUTOR window over the real dispatch
    // atomic (`next` over the flat `execOrder` task list — duplicates for
    // split parts are fine: the enqueue dedup and the claim CAS collapse
    // them, and split LBs are ineligible victims anyway), then, after the
    // executor join, a FINALIZE window over `nextLi` / `toRun`. This retires
    // the frozen `phase2Cursor` (a completion cursor stored only by the
    // finalize, which sat at 0 through the whole hashburst — the steward was
    // blind in the heaviest phase and every phase-2 reload ran inline on
    // workers). Redo passes open their own windows over their own vectors,
    // so indices are always in the right space. The unified handshake uses
    // the single stewardClaim word — the separate burst-claim word and the
    // mid-burst throttle gate are retired (the pager subsumes them).

    // ---- Phase 2: FLAT executor pool + per-LB finalize, EXPRESSION-BUCKET split ----
    // FLAT parallelism, never nested: one work-stealing pool over the flat list of
    // (LB, part) executor tasks across the LBs of this pass. No LB spawns
    // sub-threads, so the cores are never oversubscribed by nesting. With the
    // read-only fixpoint (D-116), distinct tasks on one LB only READ it (the shared
    // NameMap is read via lookup) and each writes its OWN sealed page set's record
    // chain, so same-LB tasks never race; tasks on different LBs are independent.
    //
    // The split is decided BEFORE the iteration (the end-of-iteration stats pass
    // below, D-201): a straggler (numberOfParts > 1) carries
    // logicalCores expression buckets. No cap, no escalation, no discarded burst.
    // The pass loop is at most TWO rounds (I-75):
    //
    //   round 1: a straggler dispatches one produceOnly PRODUCER (produceExpressionStumps,
    //            whole-LB, fires nothing); every other LB dispatches one unsplit burst
    //   round 2: the producer's regular and terminal pre-stumps are dealt into
    //            buckets, one bucket part each
    //
    // Requeued work runs in the NEXT round, never appended to the running one: the
    // working-set pager has registered this round's task list and dispatch cursor, so
    // the task vector cannot grow under it. An LB is finalised ONCE, in the round after
    // which it has no tasks left. The merge is partition-independent (applyFiringRecords
    // sorts, D-117 / I-77), so which parts came from which round cannot matter.
    //
    // The rule dimension (partitionAccepts / g_splitCount) is OFF on the main path
    // (splitCount stays 1). disable_lb_split and the incubator run UNSPLIT; the CE
    // filter is unaffected (its own splitCount=1 loop in filter.cpp).
    struct ExecTask {
        Memory* lb;
        std::size_t li;          // index into `active`
        int processID;           // 0 on the main path (rule dimension off)
        int splitCount;          // 1 on the main path (partitionAccepts accepts all rules)
        int partCount;           // concurrent parts of this LB this burst (= bucket count);
                                 // sets g_isMultiPart in performElem2 (early-exit gate)
        bool produceOnly;        // a round-1 stump PRODUCER (runs produceExpressionStumps,
                                 // not performElem2); its buckets requeue for round 2
        SplitStumpRef stump;     // empty for an unsplit part; set for a bucket part
        std::atomic<bool>* stop;
        std::atomic<int>* partsLeft;
    };
    const bool mainPath = parameters.lb_split && !parameters.disable_lb_split;
    // A straggler splits into logicalCores expression buckets; that is the only
    // split dimension now, so the whole-machine core count is the per-LB part
    // ceiling (Rule 19 -- a machine past the named constant stops HERE).
    assert(static_cast<int>(logicalCores) <= kMaxSplitParts
        && "logicalCores exceeds kMaxSplitParts - raise the named constant "
           "deliberately for a machine with more cores than the ceiling");

    if (!active.empty()) {
    const std::size_t M = active.size();
    // Per-LB phase-2 early-exit flags, EXTERNAL to the LB so the hashburst stays
    // strictly read-only on it (I-66). One flag per LB, shared by all its parts;
    // sized once so the &stopFlags[li] handed to tasks stay stable across passes
    // (the vector is never resized; std::atomic is not movable).
    std::vector<std::atomic<bool>> stopFlags(M);
    // Per-LB remaining-parts counter (working-set pager): each executor part
    // decrements it after sealing, and the part that drops it to zero releases the
    // LB's claim to Idle -- so an executor-done LB becomes deloadable immediately,
    // bounding the phase-2 resident set to the working set (I-114). Re-armed per pass.
    std::vector<std::atomic<int>> partsRemaining(M);

    // Sealed record sets accumulate ACROSS this iteration's rounds: a non-straggler's
    // round-1 burst is kept while a straggler's buckets run in round 2. Deque:
    // SealedPageSet is non-movable, so the views' owner pointers stay stable (D-164).
    std::deque<SealedPageSet> pageStore;
    std::vector<std::vector<SealedPageSet*>> keptParts(M);
    // Per-LB total work this iteration = SUM of its parts' submatch counts
    // (split-invariant, D-117) -- the straggler classifier's input (the end-of-
    // iteration stats pass below). lbMaxSub is the busiest single part, kept only
    // for the split-ineffective self-control report.
    std::vector<int64_t> lbTotalSub(M, 0);
    std::vector<int64_t> lbMaxSub(M, 0);
    // Diagnostics for the split-ineffective report: how many stump work items the
    // producer returned and how many buckets it dealt them into (1 = not really split).
    std::vector<int32_t> lbStumps(M, 0);
    std::vector<int32_t> lbBuckets(M, 1);
    std::vector<char> lbFinalized(M, 0);
    // The stumps a producer task returns, copied off its sealed pages so those pages
    // go back to the pool at once. The bucket tasks point into these runs, so the
    // storage must outlive the pass: deque, never reallocated.
    std::deque<std::vector<ExpressionStump>> stumpStore;

    // Task build. A straggler (numberOfParts > 1, set last iteration by the stats
    // pass below) dispatches ONE producer task that enumerates the LB's expression
    // stumps; its buckets run in round 2. Every other LB dispatches ONE unsplit part.
    // The rule dimension is off on the main path: splitCount stays 1 so
    // partitionAccepts accepts every rule.
    std::vector<ExecTask> tasks;
    for (std::size_t li = 0; li < M; ++li) {
        Memory* b = active[li];
        if (!b->isActive) continue;  // discharged in phase 1 -> no executor tasks
        // Split preemptively if the submatch stat flagged it last iteration
        // (numberOfParts > 1) OR it just activated this iteration (justActivated,
        // one-shot: no prior burst for the stat to see). Consume the flag here.
        const bool straggler = mainPath && (b->numberOfParts > 1 || b->justActivated);
        b->justActivated = false;
        tasks.push_back(ExecTask{ b, li, /*processID=*/0, /*splitCount=*/1,
            /*partCount=*/1, /*produceOnly=*/straggler, SplitStumpRef{},
            &stopFlags[li], &partsRemaining[li] });
    }

    int passNo = 0;
    for (;;) {
        ++passNo;
        // Round 1: producers + unsplit bursts. Round 2: the producers' buckets.
        assert(passNo <= 2
            && "phase-2 pass loop exceeded two rounds - a bucket part requeued");

        for (std::size_t li = 0; li < M; ++li) {
            stopFlags[li].store(false, std::memory_order_relaxed);
            partsRemaining[li].store(0, std::memory_order_relaxed);
        }
        for (const ExecTask& t : tasks)
            partsRemaining[t.li].fetch_add(1, std::memory_order_relaxed);

        // This pass's page sets append to the store; `base` is where they start.
        const std::size_t base = pageStore.size();
        for (std::size_t i = 0; i < tasks.size(); ++i) pageStore.emplace_back();
        // A second set per task, for the stumps a round-1 producer returns: the
        // record chain is single-type per set (I-135) and the firing records already
        // own the first.
        std::deque<SealedPageSet> stumpPages(tasks.size());
        std::vector<char> stumpBound(tasks.size(), 0);
        std::vector<int32_t> taskStumps(tasks.size(), 0);
        // Per-(LB, part) submatch tally: each worker reads g_growthMatchCount right
        // after its performElem2 returns (the matches owned by this part, per
        // partitionAccepts). See D-109.
        std::vector<int64_t> taskSubMatches(tasks.size(), 0);
        // Flat executor order for this pass's pager window: one entry per task.
        std::vector<Memory*> execOrder;
        execOrder.reserve(tasks.size());
        for (const ExecTask& t : tasks) execOrder.push_back(t.lb);

        if (!tasks.empty()) {
            std::atomic<std::size_t> next{ 0 };
            // EXECUTOR window: the pool's real dispatch atomic is the pager cursor,
            // so the steward prefetches upcoming tasks' LBs and evicts behind the
            // executor sweep in real time.
            steward->beginPhaseWindow(/*phase=*/2, &next, &execOrder, workers,
                                      lbdeload::kDeloadDirectory);
            auto worker = [this, &tasks, &pageStore, base, &stumpPages, &stumpBound,
                           &taskStumps, &taskSubMatches, &next, workers,
                           mainPath](unsigned coreId) {
                const unsigned cid = workers ? (coreId % workers) : 0U;
                // The phase-2 executor pool runs in parallel like phases 1/3, so
                // publish this worker's slot. Every g_currentCoreId-resolved per-slot
                // arena consumer reached from performElem2 must pick THIS worker's
                // slot, not the single reserved slot every unpublished worker would
                // share (that slot is single-threaded-only).
                g_currentCoreId = static_cast<int>(cid);
                for (;;) {
                    std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= tasks.size()) break;
                    const ExecTask& t = tasks[i];
                    // Unified working-set handshake: claim this LB so the steward
                    // will not deload it under us, reloading if it is cold. A split
                    // sibling that already owns it returns immediately (resident).
                    steward->claimAndLoadForWork(*t.lb, /*phase=*/2,
                                                 lbdeload::kDeloadDirectory);
                    // The task's exclusive write window on its page set opens here
                    // and closes at the seal below -- the records' strings then cross
                    // the pool join read-only.
                    SealedPageSet& ps = pageStore[base + i];
                    ps.bind(&staticMemory());
                    if (t.produceOnly) {
                        // Round-1 stump PRODUCER: enumerate the whole LB's
                        // expression stumps at g_splitCount == 1 (so the filter
                        // accepts every rule), retaining terminal pre-stumps for
                        // recordable nodes replaced by a deeper level. It fires
                        // nothing and deposits nothing (ps stays empty); the
                        // classify deals all work items into buckets that run in
                        // round 2. Grow MORE stumps than buckets (a small multiple
                        // of logicalCores) so the round-robin deal evens out the
                        // buckets' grow-tree sizes.
                        g_splitProcessID = 0;
                        g_splitCount = 1;
                        g_isMultiPart = false;
                        ps.seal();
                        stumpPages[i].bind(&staticMemory());
                        stumpBound[i] = 1;
                        const int32_t stumpTarget = static_cast<int32_t>(
                            logicalCores) * kStumpsPerBucketTarget;
                        taskStumps[i] = this->produceExpressionStumps(
                            *t.lb, cid, stumpTarget, stumpPages[i]);
                        stumpPages[i].seal();
                        taskSubMatches[i] = 0;  // the producer does no firing work
                        // The producer's level column held this slot's gen arena;
                        // the stumps were copied onto the sealed pages, so nothing
                        // points into it — hand the blocks back now.
                        genScratchArenas().forSlot(cid).releaseAll();
                    } else {
                        // A burst part: an unsplit LB, or one expression bucket of a
                        // straggler. Runs to completion (no cap). partCount > 1 for a
                        // bucket keeps the early-exit off (I-76 / g_isMultiPart).
                        this->performElem2(*t.lb, cid, t.processID, t.splitCount,
                                           t.partCount, t.stump, ps, *t.stop);
                        ps.seal();
                        taskSubMatches[i] = g_growthMatchCount;
                    }
                    // Last part of this LB to finish releases its claim to Idle (all
                    // parts have sealed -> no part still reads it), so the steward may
                    // now deload it. The finalize re-claims + reloads it.
                    if (t.partsLeft->fetch_sub(1, std::memory_order_acq_rel) == 1) {
                        assert(t.lb->stewardClaim.load(std::memory_order_relaxed)
                                   == static_cast<uint8_t>(
                                          Memory::StewardClaim::WorkerOwned)
                               && "phase-2 executor release of an LB not held "
                                  "WorkerOwned");
                        t.lb->stewardClaim.store(
                            static_cast<uint8_t>(Memory::StewardClaim::Idle),
                            std::memory_order_release);
                    }
                }
                };
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (unsigned t = 0; t < workers; ++t) pool.emplace_back(worker, t);
            for (auto& th : pool) th.join();
            // Close the executor window at its join (its cursor and order die with
            // this block; queued stale tasks drop on the generation).
            steward->endPhaseWindow();
        }

        // ---- Classify every part, single-threaded: deal a producer's stumps into
        // buckets (requeue for round 2), or keep a burst part ----
        std::vector<ExecTask> nextTasks;
        for (std::size_t i = 0; i < tasks.size(); ++i) {
            const ExecTask& t = tasks[i];
            const std::size_t li = t.li;
            SealedPageSet& ps = pageStore[base + i];

            // A round-1 PRODUCER. It fired nothing (ps is empty). Deal its stumps
            // into logicalCores expression buckets and requeue one bucket part each.
            if (t.produceOnly) {
                ps.freePages();
                if (taskStumps[i] == 0) {
                    // No statement survives the filter -> the LB's real burst would
                    // generate nothing either. Run it once, unsplit, this iteration
                    // (guarantees the burst happens; its work then feeds the stats).
                    nextTasks.push_back(ExecTask{ t.lb, li, /*processID=*/0,
                        /*splitCount=*/1, /*partCount=*/1, /*produceOnly=*/false,
                        SplitStumpRef{}, &stopFlags[li], &partsRemaining[li] });
                    continue;
                }
                stumpStore.emplace_back();
                std::vector<ExpressionStump>& run = stumpStore.back();
                run.reserve(static_cast<std::size_t>(taskStumps[i]));
                stumpPages[i].forEachRecord<ExpressionStump>(
                    [&run](const ExpressionStump& s) { run.push_back(s); });
                assert(static_cast<int32_t>(run.size()) == taskStumps[i]
                    && "stump record chain disagrees with the producer's count");

                // Deal the regular and terminal pre-stumps into BUCKETS, one part
                // per bucket. Round-robin, so
                // a bucket draws stumps from across the name-sorted list rather than
                // one contiguous slice; the stumps are permuted into bucket order so
                // each bucket is a contiguous run. Grown to a multiple of buckets
                // (kStumpsPerBucketTarget), so the deal evens out the buckets'
                // grow-tree sizes; fewer stumps than buckets means the LB could not
                // fan wider. The generator builds the filter once per bucket and the
                // emitter's seen-set collapses a request two of a bucket's stumps
                // both reach (I-158).
                const int32_t nStumps = static_cast<int32_t>(run.size());
                lbStumps[li] = nStumps;
                const NameId buckets = static_cast<NameId>(
                    std::min<int32_t>(nStumps, static_cast<int32_t>(logicalCores)));
                lbBuckets[li] = buckets;
                std::vector<ExpressionStump> dealt;
                dealt.reserve(run.size());
                std::vector<int32_t> bucketStart(
                    static_cast<std::size_t>(buckets) + 1, 0);
                for (NameId k = 0; k < buckets; ++k) {
                    bucketStart[static_cast<std::size_t>(k)] =
                        static_cast<int32_t>(dealt.size());
                    for (int32_t j = k; j < nStumps; j += buckets)
                        dealt.push_back(run[static_cast<std::size_t>(j)]);
                }
                bucketStart[static_cast<std::size_t>(buckets)] =
                    static_cast<int32_t>(dealt.size());
                run.swap(dealt);
                for (NameId k = 0; k < buckets; ++k) {
                    const int32_t lo = bucketStart[static_cast<std::size_t>(k)];
                    const int32_t hi = bucketStart[static_cast<std::size_t>(k) + 1];
                    nextTasks.push_back(ExecTask{ t.lb, li, /*processID=*/0,
                        /*splitCount=*/1, /*partCount=*/buckets, /*produceOnly=*/false,
                        SplitStumpRef{ run.data() + lo,
                                       static_cast<NameId>(hi - lo), k, buckets },
                        &stopFlags[li], &partsRemaining[li] });
                }
                continue;
            }

            // A burst part (an unsplit LB or one expression bucket): keep it and add
            // its submatch count to the LB's split-invariant TOTAL work (the straggler
            // classifier's input); lbMaxSub tracks the busiest single part for the
            // split-ineffective report.
            keptParts[li].push_back(&ps);
            lbTotalSub[li] += taskSubMatches[i];
            if (taskSubMatches[i] > lbMaxSub[li]) lbMaxSub[li] = taskSubMatches[i];
        }
        for (std::size_t i = 0; i < tasks.size(); ++i)
            if (stumpBound[i]) stumpPages[i].freePages();

        // ---- Finalize every LB that has no work left ----
        std::vector<char> hasNext(M, 0);
        for (const ExecTask& t : nextTasks) hasNext[t.li] = 1;
        std::vector<Memory*> toFinalize;
        std::vector<std::size_t> finalizeLi;
        for (std::size_t li = 0; li < M; ++li) {
            if (hasNext[li] || lbFinalized[li]) continue;
            lbFinalized[li] = 1;
            toFinalize.push_back(active[li]);
            finalizeLi.push_back(li);
        }
        if (!toFinalize.empty()) {
            std::atomic<std::size_t> nextLi{ 0 };
            // FINALIZE window: nextLi is a real fetch_add dispatch cursor over
            // toFinalize -- the pager tracks the finalize sweep like phases 1/3.
            steward->beginPhaseWindow(/*phase=*/2, &nextLi, &toFinalize, workers,
                                      lbdeload::kDeloadDirectory);
            auto finalizeWorker = [this, &toFinalize, &finalizeLi, &keptParts,
                                   &nextLi](unsigned tIdx) {
                // The finalize pool runs in parallel like phases 1/3, so publish this
                // worker's slot. The per-slot scratch / gen arenas reached deep in the
                // drains must pick THIS worker's arena, not the single reserved slot
                // every worker would otherwise share.
                g_currentCoreId = static_cast<int>(tIdx);
                for (;;) {
                    std::size_t j = nextLi.fetch_add(1, std::memory_order_relaxed);
                    if (j >= toFinalize.size()) break;
                    Memory* b = toFinalize[j];
                    const std::size_t li = finalizeLi[j];
                    // Make the LB resident before finalize writes it: an active LB was
                    // released to Idle by its last executor part; an inactive one (no
                    // executor task) may be cold. The handshake reloads it.
                    steward->claimAndLoadForWork(
                        *b, /*phase=*/2, lbdeload::kDeloadDirectory);
                    // The split decision is no longer here (no cap, no escalation):
                    // it is the single-threaded end-of-iteration stats pass below.
                    // Hand the LB's sealed part sets, in part order, to the finalize
                    // (applyFiringRecords reads each chain in append order, then
                    // sorts - the merge is partition- and pass-independent). Empty for
                    // a phase-1-discharged LB: nothing deposits.
                    const int count = static_cast<int>(keptParts[li].size());
                    assert(count <= kMaxSplitParts
                        && "LB part count exceeds kMaxSplitParts - raise the named "
                           "constant deliberately, never cap the split silently");
                    this->performElemPhase2(*b, keptParts[li].data(), count);
                    // Phase 2 is done with this LB: release the claim so the steward
                    // may reclaim it.
                    assert(b->stewardClaim.load(std::memory_order_relaxed)
                               == static_cast<uint8_t>(
                                      Memory::StewardClaim::WorkerOwned)
                           && "phase-2 finalize release of an LB not held WorkerOwned");
                    b->stewardClaim.store(
                        static_cast<uint8_t>(Memory::StewardClaim::Idle),
                        std::memory_order_release);
                }
                };
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (unsigned t = 0; t < workers; ++t) pool.emplace_back(finalizeWorker, t);
            for (auto& th : pool) th.join();
            steward->endPhaseWindow();
            // Records consumed -- free the finalised LBs' pages in one sweep. After
            // this point any straggler view asserts at its access site.
            for (std::size_t j = 0; j < finalizeLi.size(); ++j)
                for (SealedPageSet* set : keptParts[finalizeLi[j]]) set->freePages();
        }

        if (nextTasks.empty()) break;
        tasks = std::move(nextTasks);
    }

    // ---- End-of-iteration straggler classification (the split TRIGGER) ----
    // Set each LB's split for the NEXT iteration from this iteration's completed,
    // deterministic work totals. work(L) = sum of L's parts' submatch counts
    // (split-invariant, D-117). A straggler is an LB whose work exceeds the ideally-
    // balanced per-core load T / logicalCores (so it alone leaves cores idle) AND
    // clears the setup break-even min_split_work; it runs as logicalCores expression
    // buckets next iteration, every other LB unsplit. Integer arithmetic only, over a
    // fixed-order single-threaded sweep -> the split set is a deterministic function
    // of proof state (two runs stay byte-identical). Recomputed every iteration, so an
    // LB whose work falls back below the bar returns to unsplit.
    if (mainPath) {
        int64_t T = 0;
        for (std::size_t li = 0; li < M; ++li) T += lbTotalSub[li];
        const int64_t fairShare = T / static_cast<int64_t>(logicalCores);
        for (std::size_t li = 0; li < M; ++li) {
            const bool straggler = ExpressionAnalyzer::isStraggler(
                lbTotalSub[li], T, static_cast<int>(logicalCores),
                static_cast<int64_t>(parameters.min_split_work));
            const bool wasUnsplit = active[li]->numberOfParts == 1;
            active[li]->numberOfParts =
                straggler ? static_cast<int>(logicalCores) : 1;
            // Split-onset report: printed once per unsplit -> split transition so
            // the run log shows which LBs the trigger promoted. Pure observation ->
            // determinism intact.
            if (straggler && wasUnsplit)
                std::cout << "[SPLIT] straggler: work=" << lbTotalSub[li]
                          << " T=" << T
                          << " parts=" << active[li]->numberOfParts
                          << " lb=" << active[li]->exprKey() << "\n";
            // Self-control report: a split LB whose busiest bucket is still a global
            // outlier has an irreducibly-serial core (a "runs and runs" induction LB)
            // that bucketing cannot subdivide. Pure observation -> determinism intact.
            if (straggler && lbMaxSub[li] > fairShare)
                std::cout << "[SPLIT] ineffective: work=" << lbTotalSub[li]
                          << " maxPart=" << lbMaxSub[li]
                          << " buckets=" << lbBuckets[li]
                          << " stumps=" << lbStumps[li]
                          << " fairShare=" << fairShare
                          << " lb=" << active[li]->exprKey() << "\n";
        }
    }
    }  // active non-empty

    // Phase 3 opens the same working-set window (all phases equivalent).
    {
        std::atomic<std::size_t> phase3Cursor{ 0 };
        steward->beginPhaseWindow(/*phase=*/3, &phase3Cursor, &active, workers,
                                  lbdeload::kDeloadDirectory);
        runPhase(phase3Cursor, [this](Memory& b, unsigned cid) {
            g_inParallelWorkerPhase = true;
            this->performElemPhase3(b, cid);
            g_inParallelWorkerPhase = false;
        });
        steward->endPhaseWindow();
    }

    int64_t deloadableMailOutBytes = 0;
    for (const Memory* lb : bodies) {
        if (lb != nullptr) deloadableMailOutBytes += lb->mailOutLiveBytes;
    }
    peakDeloadableMailOutBytes =
        std::max(peakDeloadableMailOutBytes, deloadableMailOutBytes);

    // A grid with no initially dormant LB has no future catch-up reader. Every
    // LB that can consume the currently retained window ran phase 1 before this
    // post-phase-3 join, so release the delivered blob/ref pages and their
    // global-id dictionary before the next commit opens a fresh window. Every
    // active LB cleared mailIn after its phase-1 absorb; mailOut still carries
    // sender-local ids, so no live global mail id crosses this seam. The
    // grid-build policy is fixed for the whole execution batch: any initially
    // dormant LB keeps full history and the matching global-id space even if it
    // later activates (I-162).
    if (!parameters.compressor_mode && rollingMailHistoryEnabled) {
        for (const Memory* lb : bodies) {
            assert(lb == nullptr || lb->mailIn.empty()
                && "mail interner retirement requires every mailIn to be empty");
        }
        this->mailLog.retireDeliveredBatches();
        resetMailInterner();
    }

    // New mail system (D-137): the commit barrier. Each LB's
    // mailOut -- this step's fillMailOut output, plus (for the root) the
    // post-join updateGlobal*/D-76 drains from the PREVIOUS iteration -- becomes
    // one appended batch in its log; mailOut is then cleared. Single-threaded
    // (post pool.join), so the logs are frozen for the next iteration's parallel
    // phase-1 pulls. Replaces the per-step smashMail.
    //
    // Iterates `bodies`, not just the active set: the post-join updateGlobal* /
    // D-76 drains below append to the ROOT's mailOut after this barrier, and the
    // root can itself deactivate (the deactivation predicate fires once all its
    // children are done), so an active-only barrier could strand the root's
    // proven-rule mail. The empty-mailOut skip keeps inactive / produced-nothing
    // LBs from accruing empty batches. Anything written to an LB's mailOut AFTER
    // this point rides the NEXT iteration's commit -- a one-iteration delay,
    // matching the old post-smashMail sendMail timing. The compressor's LBs are
    // flat and never registered, so it is gated out.
    if (!parameters.compressor_mode) {
        // BARRIER SEAM WINDOW A (D-196): the commit sweep
        // is a linear walk over `bodies`, so it gets a dispatch-style pager
        // window like a phase — the planner prefetches ahead of the commit
        // cursor and drains behind it, so the sweep's reloads no longer
        // accumulate monotonically with the planner parked (the 4 GiB
        // between-iterations wall). Each committed LB goes through the
        // UNIFORM DOOR (claimAndLoadForWork -> commit -> release Idle): the
        // claim hold keeps the executors off the LB while commit serializes
        // its private deloadable mailOut ids, and the claim word ends
        // consistent — a seam reload can never leave Dumped-but-resident.
        //
        // Discharge distinction: an LB that merely DEACTIVATED this iteration
        // (isActive false; dischargedForever not yet set — the discharge
        // block runs later in this same barrier) is legal to door-reload for
        // its final commit; a PREVIOUSLY-discharged LB is structurally
        // skipped by the empty-mailOut check — its last commit (this sweep,
        // the barrier before its discharge) cleared mailOut, and nothing
        // appends to a non-root mailOut afterwards (fillMailOut needs a phase
        // run; the post-join drains merge into the ROOT's mailOut only). The
        // one theoretical exception is the ROOT itself deactivating with the
        // drains appending after this sweep — that path lands on
        // ensureLoaded's discharged-assert exactly as before (a real bug we
        // want loud, Rule 19), and is unreachable in practice: the prove loop
        // ends when the root deactivates.
        std::atomic<std::size_t> commitCursor{ 0 };
        steward->beginPhaseWindow(/*phase=*/4, &commitCursor, &bodies, workers,
                                  lbdeload::kDeloadDirectory);
        for (std::size_t bi = 0; bi < bodies.size(); ++bi) {
            commitCursor.store(bi, std::memory_order_relaxed);
            Memory* lb = bodies[bi];
            if (!lb) continue;
            if (!lb->mailOutPending) continue;
            steward->claimAndLoadForWork(*lb, /*phase=*/4,
                                         lbdeload::kDeloadDirectory);
            assert(!lb->mailOut.empty()
                && "mailOutPending set for an empty outgoing mailbox");
            this->mailLog.commit(lb, lb->mailOut);
            lb->clearMailOut();
            lb->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        }
        steward->endPhaseWindow();
    }

    // Sort thread-collected vectors for deterministic processing order
    std::sort(inductionMemoryBlocks.begin(), inductionMemoryBlocks.end(),
        [](const Memory* a, const Memory* b) { return a->exprKey() < b->exprKey(); });
    std::sort(updateGlobalTuples.begin(), updateGlobalTuples.end());

    // BARRIER SEAM WINDOW B (D-196): the post-join drains
    // below (deferred ancestor admissions, updateGlobal, updateGlobalDirect,
    // the compaction flush) deposit into RANDOM-ACCESS recipient LBs (per-
    // theorem accessMemory tree lookups) — there is no sweep order to
    // register, so the window's cursor is PINNED AT 0 over `active`. That
    // pin is the honest Belady origin between iterations: the next use of
    // active[i] IS position i (the next kernel re-sweeps from 0), so the
    // planner keeps the next iteration's head resident (pre-warming phase 1)
    // and evicts everything else farthest-first WHILE the drains churn —
    // bounding the residency that previously grew monotonically with no
    // window open. Each drain reload goes through the uniform door
    // (claimAndLoadForWork -> write -> release Idle): recipients are
    // WorkerOwned across their deposit (never evicted mid-write), evictable
    // the moment they release (the deposit lives in the arena and rides the
    // raw image — recipient churn under pressure is correct behavior).
    {
        std::atomic<std::size_t> drainCursor{ 0 };
        steward->beginPhaseWindow(/*phase=*/4, &drainCursor, &active, workers,
                                  lbdeload::kDeloadDirectory);

    for (int blockIndex = 0; blockIndex < inductionMemoryBlocks.size(); blockIndex++)
    {
        Memory* memoryBlock = inductionMemoryBlocks[blockIndex];
        this->activateZeroCondition(*memoryBlock);
    }

    // I-28 detect-and-defer (trial): replay the cross-LB ancestor admission
    // seeds the parallel phase-1/phase-3 workers deferred, single-threaded here.
    this->drainDeferredAncestorAdmissions();

    for (const std::tuple<int, bool, int>& t : updateGlobalTuples)
    {
        updateGlobal(std::get<0>(t), std::get<1>(t), std::get<2>(t));
    }

    // Replay the proven direct theorems staged on updateGlobalDirectPages, sorted
    // (the former std::sort over updateGlobalDirectTuples + the replay loop).
    this->drainUpdateGlobalDirect();
    // Retract already-submitted rows whose producer turned out
    // unintentionally contradictory this iteration (the vacuous-premise
    // flag): the emission gates cover new submissions, this sweep covers
    // prior-iteration ones.
    this->revertVacuousGlobalTheorems();
    // Twin deactivation rides directly behind the theorem drain so every
    // firing/closure of this iteration has staged its twins first.
    this->drainPendingTwinDeactivations();
    // Vacuous-premise subtree deactivation: every LB whose main turned
    // contradictory this iteration takes its whole subtree out of the
    // search — nothing sound and non-vacuous can be proved below an
    // inconsistent premise set.
    this->drainSubtreeDeactivations();

    // Deferred compaction drain (D-76,
    // Option A). Single-threaded, post-pool.join(): sorting makes the
    // implication<N> allocation a deterministic function of the broadcast
    // set (by string), so compileImplicationToCompact's global implCounter /
    // compiledExpressions / repetitionExclusionMap mutation is injective and
    // reproducible instead of the racy parallel allocation that violated
    // I-28 and broke determinism. Each compact form is then merged into the
    // root's mailOut, shipped by the MailLog commit barrier -- a two-iteration
    // delay vs the producing burst, matching the old post-smashMail timing.
    std::sort(pendingCompactionQueue.begin(), pendingCompactionQueue.end());
    {
        std::map<int, Mail> compactionMailByCore;
        for (const std::tuple<std::string, int, int>& e : pendingCompactionQueue)
        {
            const std::string& original = std::get<0>(e);
            const int cId = std::get<2>(e);
            const std::string compactImpl = compileImplicationToCompact(original);
            // Level set MUST be empty. The implication rule always
            // deposits std::set<int>() for its levels.
            // The receiver-side addToHashMemory call propagates these levels
            // into the rule's intStatementLevelsMap entry, and when the rule
            // fires, the derived statement's levels are computed as the union
            // of the rule levels and the matching premises' levels. The
            // empty set keeps the derived levels equal to the premise-side
            // union only, which is what the allLevelsInvolved discharge gate
            // in prover.hpp::dischargeToBeProved expects
            // (size == memoryBlock.level + 1). Filling compactLevels with
            // {0, 1, ..., kySize} would inject an extra level into every
            // derived statement that fired against a mail-arrived rule,
            // making size > level + 1 so the discharge gate returns false
            // and the theorem is never promoted to globalTheoremList.
            std::set<int> compactLevels;
            ExpressionWithValidity compactEv(compactImpl, "main");
            Mail& m = compactionMailByCore[cId];
            m.statements.insert(std::make_pair(compactEv, compactLevels));
            if (parameters.trackHistory) {
                // Cite the binary's CANONICAL reconstruction in the
                // `compilation` row's rest[0], not the input `original`.
                // compileImplicationToCompact dedups alpha-equivalent
                // inputs (two distinct theorem reformulations whose only
                // difference is the order of two co-bound variables in
                // a `>[w_i,w_j]` group, e.g. `[w2,w4,w5][w3,w4,w6]
                // [w1,w5,w6]` vs `[w2,w5,w4][w3,w5,w6][w1,w4,w6]`) to
                // the SAME implication<N> name. The binary stores only
                // the first-seen body; any later input whose binder
                // ordering differs from the binary registered form will
                // be alpha-equivalent but NOT structurally identical.
                //
                // The verifier's `check_compilation` reconstructs the
                // implication from the binary's `elements` via
                // `_build_implication_from_elements` (the Python mirror
                // of reconstructImplicationFullBind) and compares with
                // rest[0] modulo `_normalize_with_unchangeables`. The
                // normalizer rewrites BOTH binder content and arg
                // content to a fresh v<k> scheme, so any alpha-equivalent
                // form WHOSE BINDER LIST MATCHES THE BODY'S FIRST-
                // APPEARANCE ORDER will pass. Citing `original` as-is
                // can produce a body whose binder list `>[w_i,w_j]` is
                // inconsistent with the first-appearance order of w_i
                // and w_j in the body (the inner premise reads
                // `in3[w2,w_j,w_i,+]` so w_j appears before w_i but the
                // binder names them w_i first) — the resulting
                // normalized strings differ in `>[v_k,v_l]` vs
                // `>[v_l,v_k]` and the structural check fails.
                //
                // The canonical reconstruction below is the same form
                // the verifier rebuilds: take the binary's stored
                // elements (placeholder-numeric form `1..K`),
                // reconstruct via reconstructImplicationFullBind with
                // its own first-appearance binder rule. The resulting
                // numeric body is structurally identical to what
                // check_compilation's _try_expand builds, so the
                // normalize-then-compare matches by construction.
                // Citing the input `original` instead would risk a
                // binder-order misalignment that the verifier rejects.
                const std::string compactCore = extractExpressionUniversalSpan(StrSpan(compactImpl)).toStdString();
                auto cit = this->compiledExpressions.find(compactCore);
                assert(cit != this->compiledExpressions.end()
                    && "deferred-compaction drain: compact form must resolve to a compiledExpressions entry");
                const std::vector<std::string>& elems = cit->second.elements;
                assert(!elems.empty()
                    && "deferred-compaction drain: implication entry must have at least one element (head)");
                std::vector<std::string> canonicalKey(elems.begin(), elems.end() - 1);
                const std::string& canonicalHead = elems.back();
                const std::string canonicalOriginal = this->reconstructImplicationFullBind(canonicalKey, canonicalHead);
                addOrigin(m.exprOriginMap, compactEv,
                    std::make_pair("compilation", std::vector<ExpressionWithValidity>{ ExpressionWithValidity(canonicalOriginal, "main") }),
                    (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }
        }
        if (!compactionMailByCore.empty()) {
            steward->claimAndLoadForWork(this->body, /*phase=*/4,
                                         lbdeload::kDeloadDirectory);
        }
        for (std::map<int, Mail>::iterator mit = compactionMailByCore.begin(); mit != compactionMailByCore.end(); ++mit) {
            // New mail system (D-137): the D-76 compact
            // (implication<N>[...]) forms target the root's descendants, so merge
            // every per-core batch into the root's mailOut. The NEXT commit
            // barrier ships them into the root's log; descendants pull and recover
            // the rule via status=3 disintegration. This preserves the old
            // post-smashMail TWO-iteration compact-implication latency -- the
            // drain runs after the barrier, so its mail rides the next barrier and
            // is pulled the cycle after, exactly the old timing (G-49:
            // contradiction LBs are sensitive to this delay; it is unchanged). The
            // per-core key no longer routes anything; the assert stays as a
            // tripwire for a corrupt core id (Rule 19).
            assert(mit->first >= -1);
            mergeBatchIntoMailOut(mit->second, this->body);
        }
        if (!compactionMailByCore.empty()) {
            this->body.stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        }
    }
    pendingCompactionQueue.clear();

        // Close the barrier seam window: the drains are done; the discharge
        // barrier block below quiesces the steward before reading any state.
        steward->endPhaseWindow();
    }

    deloadableMailOutBytes = 0;
    for (const Memory* lb : bodies) {
        if (lb != nullptr) deloadableMailOutBytes += lb->mailOutLiveBytes;
    }
    peakDeloadableMailOutBytes =
        std::max(peakDeloadableMailOutBytes, deloadableMailOutBytes);

    // (D-39 pendingAncestorOrigins drain retired by D-51 — contradiction record
    // now lives only in the __contradiction__ LB, no upward propagation. Drain
    // queue + sort + iteration removed in the post-D-52 cleanup commit.)

	inductionMemoryBlocks.clear();
	updateGlobalTuples.clear();
	// updateGlobalDirectPages is reset by drainUpdateGlobalDirect above.

    // --- Statification: LAZY deload — dump-on-release only ---
    // (D-155) Still the LAST act of the
    // kernel, after EVERY post-join drain (activateZeroCondition,
    // updateGlobal*, the compaction flush) — no same-iteration consumer
    // can touch a cold LB; the next touch is the next iteration's
    // phase-1 ensureLoaded or a post-prove reader's explicit
    // ensureLoaded. The CE filter never enters proveKernel, so its
    // clones never deload.
    //
    // POLICY — pressure-lazy discharge (D-156),
    // decided HERE, executed by the steward (D-159):
    // do nothing until the pool forces it. An LB that left the active
    // set this iteration is flagged dischargedForever (deactivation is
    // permanent — no path flips isActive back) and joins
    // pendingDischarge in deterministic `active` order; NO file is
    // written, its blocks sit inside the pool budget, its content is
    // frozen. The barrier protocol (I-106): disarm
    // the previous iteration's trigger, QUIESCE the steward (every
    // previously released drain completes — quiesced block counts are
    // logical again), fold what it drained, enqueue this iteration's
    // discharges, then decide on quiesced counts:
    // - above the kReleaseHighWaterNum/Den hard bound: synchronous
    //   relief right here — pendings first (cheapest: frozen content,
    //   full canonical image, blocks never needed again), then, if
    //   still above, the parallel active-LB eviction sweep (full dump,
    //   phase-1 reload at next touch);
    // - above the steward::kWake watermark: hand the pending list to
    //   the steward and wake it — the drain runs in the background
    //   while the next iteration computes (discharged LBs are disjoint
    //   from everything the phases touch);
    // - below the wake mark: install the list and arm the one-shot
    //   grant trigger at the blocks-to-cross distance — a mid-iteration
    //   crossing wakes the steward, which drains the WHOLE installed
    //   list (whole barrier-fixed units, never a partial drain).
    // A run that never crosses the wake mark writes ZERO deload files.
    // The manifest is rewritten only when something was dumped. Every
    // dump is the skip/tail/full decision (a reloaded-then-unchanged LB
    // skips straight to block release). Whether the trigger fires in a
    // given iteration is run-invariant (monotone grant ledger), so the
    // dumped file SET stays deterministic; only the dumping thread and
    // its timing vary (unobservable). The pool stays the hard bound
    // (exhaustion still asserts).
    {
        assert(steward && "proveKernel barrier without a steward scope");
        staticMemory().disarmGrantTrigger();
        steward->quiesce();
        bool dumpedSomething = false;
        // Fold + clear the list the steward drained since the last
        // barrier (its content was handed over by a previous install).
        // Per-LB discharge fold: a discharged LB is dumped exactly once and
        // never reloaded, so non-residency IS its done-marker. Drop the
        // entries already deloaded (by the steward's background drain or a
        // synchronous pass) and keep the still-resident, not-yet-drained ones
        // for re-install. Removing per-LB rather than clearing the whole list
        // on a single drained flag is what stops an already-deloaded LB from
        // being re-installed and deloaded a second time under heavy pressure.
        {
            const std::size_t beforeFold = pendingDischarge.size();
            pendingDischarge.erase(
                std::remove_if(pendingDischarge.begin(),
                               pendingDischarge.end(),
                               [](const Memory* b) {
                                   return !b->lbMemory.manager.resident();
                               }),
                pendingDischarge.end());
            if (pendingDischarge.size() != beforeFold) dumpedSomething = true;
        }
        // No eviction plan to fold — active-LB eviction is continuous (the
        // working-set pager), not barrier-armed. Only inactive discharges.
        for (Memory* b : active) {
            // Every phase releases its claim when done, so no active LB may
            // still be WorkerOwned here — a missed release would make the next
            // iteration's handshake skip its load (Rule 19). Idle (resident or
            // cold) and Dumped (cold) are both valid carry-over states, so no
            // reset is needed.
            assert(b->stewardClaim.load(std::memory_order_relaxed)
                       != static_cast<uint8_t>(
                              Memory::StewardClaim::WorkerOwned)
                   && "active LB still WorkerOwned at the barrier");
            if (!b->isActive) {
                assert(!b->dischargedForever
                    && "discharged LB re-entered an active snapshot");
                // SEAM DOOR (D-196): the pager may have
                // deloaded this LB while it was active+Idle in a phase or the
                // barrier seam window; reload before discharge
                // (dischargeStatementContent reads the registry and asserts
                // residency). The uniform handshake keeps the claim word
                // consistent (never Dumped-but-resident); it runs here
                // post-quiesce with no window open, so its valve is a defined
                // no-op and nothing can race the claim. The discharge
                // protocol (D-157): capture the exact
                // gate records, empty the dischargeable containers, reshuffle —
                // every block returns NOW, zero I/O. The pressure-lazy pending
                // dump later writes the near-empty image.
                steward->claimAndLoadForWork(*b, /*phase=*/4,
                                             lbdeload::kDeloadDirectory);
                b->stewardClaim.store(
                    static_cast<uint8_t>(Memory::StewardClaim::Idle),
                    std::memory_order_release);
                b->dischargeStatementContent(dischargeScratch);
                // Stamp the ordinal single-threaded now, before any later
                // (steward) dump names the file.
                b->ensureDeloadOrdinal();
                pendingDischarge.push_back(b);
            }
        }
        // Hand the discharged list to the steward for the background
        // near-empty dump: wake it now if over the wake mark, else arm the
        // grant trigger to wake it on a mid-iteration crossing
        // (I-106). Active-LB pressure is handled
        // continuously by the working-set pager (the phase windows), so the
        // barrier no longer mass-deloads active LBs; the pool stays the hard
        // bound and genuine exhaustion asserts in a phase, never here.
        if (!pendingDischarge.empty()) {
            const int64_t totalBlocks = staticMemory().totalBlocks();
            const int64_t usedNow = staticMemory().blocksInUse();
            steward->installDischargeWork(pendingDischarge,
                                          lbdeload::kDeloadDirectory);
            if (usedNow * steward::kWakeDen
                    >= totalBlocks * steward::kWakeNum) {
                steward->wake();
            }
            else {
                const int64_t crossAt =
                    (totalBlocks * steward::kWakeNum + steward::kWakeDen - 1)
                    / steward::kWakeDen;
                const int64_t threshold = crossAt - usedNow;
                assert(threshold >= 1);
                MemorySteward* s = steward.get();
                staticMemory().armGrantTrigger(threshold,
                                               [s]() { s->wake(); });
            }
        }
        staticMemory().resetGrantLedger();
        if (dumpedSomething)
            lbdeload::rewriteRegistry(staticMemory().deloadRegistry(),
                                      staticMemory().extentSlabRegistry(),
                                      lbdeload::kDeloadDirectory);
        // BARRIER HEAD PREFETCH (D-196): warm the head of
        // the next iteration's sweep — one-shot HIGH-lane loads for the
        // Dumped, still-active LBs among the first window-width slots, so
        // phase 1 does not cold-start every iteration. Runs AFTER the
        // discharge decision (inactive heads are skipped) and after the
        // quiesce above, so no queued load can ever observe a
        // dischargedForever LB. Content-invisible cross-boundary I/O.
        // NOT harmless when unbounded: these loads execute with no window
        // open — no planner eviction can relieve them and the executor load
        // path has no valve — so prefetchHead caps its issues at the
        // prefetch budget (pending loads always fit inside the reserve's
        // free blocks), and after the LAST iteration the prove-scope guard
        // discards whatever is still queued (there is no next iteration to
        // warm). The unbudgeted, undiscarded form was the 4 GiB teardown
        // wall.
        steward->prefetchHead(
            &active,
            static_cast<std::size_t>(workers)
                + steward::kLookaheadWorkerMultiple
                      * static_cast<std::size_t>(workers),
            lbdeload::kDeloadDirectory);
    }
}

void ExpressionAnalyzer::removeExpressionFromMemoryBlock(StrSpan original, StrSpan validityName, Memory& mb, int state) {
    if (state == 0)
    {
        // Match on the int rows' (originalId, validityId) — equivalent to the
        // full-struct compare because stored rows are canonical-pipeline
        // encodings whose every field derives from that pair. lookup is
        // non-minting: a never-interned target has no stored row, so the
        // sweep finds nothing — the same no-op the string scan produced.
        const NameId origId = mb.nameMap.lookup(original);
        const NameId valId = mb.nameMap.lookup(validityName);
        if (origId == 0 || valId == 0) return;
        // Erase back to front to keep indices valid.
        auto eraseIntRows = [origId, valId](PagedVector<IntEncodedExpr>& intVec) {
            for (int32_t i = intVec.size(); i-- > 0; ) {
                if (intVec[i].originalId == origId
                    && intVec[i].validityId == valId) {
                    intVec.erase(i);
                }
            }
        };
        eraseIntRows(mb.intLocalEncodedStatements);
        eraseIntRows(mb.intLocalEncodedStatementsDelta);
        eraseIntRows(mb.intEncodedStatements);
    }
    else if (state == 1) {
        // Remove from the goal registry. The caller guarantees the goal is
        // registered, so both names were interned by its insert site — a
        // lookup miss here is a contract violation, not a probe miss.
        const NameId tbpOrigId = mb.nameMap.lookup(original);
        const NameId tbpValId = mb.nameMap.lookup(validityName);
        assert(tbpOrigId != 0 && tbpValId != 0
               && "removeExpressionFromMemoryBlock: goal names not interned");
        const int64_t pkTBP = packStatementKey(tbpOrigId, tbpValId);
        assert(mb.intToBeProved.lookup(pkTBP) != 0);
        mb.intToBeProved.eraseSet(pkTBP);
    }
}

void ExpressionAnalyzer::removeExpressionFromMemoryBlock(const EncodedExpression& encExpr, Memory& mb, int state) {
    // Owning-string overload: forward the two names as spans. NameMap::lookup
    // interns nothing, so the id verdict — and the erase / discharge it drives —
    // is identical to reading encExpr.original / encExpr.validityName directly.
    removeExpressionFromMemoryBlock(StrSpan(encExpr.original), StrSpan(encExpr.validityName), mb, state);
}
// Turn optimizations OFF for just this section
//#pragma optimize("", off)


/// @brief Queue every sibling branch of one proved OR-integration cohort for
///        radical subtree cleanup.
///
/// @details
/// Selects only validity scopes that classify as OR integration, carry the
/// exact outer signature @p orExprSig, and have @p parentValidityId as their
/// direct NameMap parent. The victims are synchronously filtered for the rest
/// of the current burst and queued for the end-of-burst `Memory::wipeSubtree`
/// drain. Because parent validity participates in selection, equal OR
/// signatures at another stack position remain live.
///
/// The scan stores ids on the request-generation scratch page tier and mints
/// no NameMap entries, preserving the caller-owned signature span and every
/// classify-view slice throughout both loops.
///
/// @param orExprSig Outer compiled OR signature shared by the cohort.
/// @param parentValidityId Exact direct parent of the proving goal scope.
/// @param mb Owning logic-block memory.
/// @return Nothing.
/// @invariant Every queued victim is a direct child of
///            @p parentValidityId and belongs to @p orExprSig.
/// @see Memory::wipeSubtree, ExpressionAnalyzer::classifyOrScopeView.
void ExpressionAnalyzer::cleanUpOrIntegrationBranches(
    StrSpan orExprSig,
    NameId parentValidityId,
    Memory& mb)
{
    const NameMap& nm = mb.nameMap;
    assert(parentValidityId != 0
           && parentValidityId <= nm.nameCount()
           && "cleanUpOrIntegrationBranches: invalid parent validity id");

    // The victim validity ids ride a PagedVector on the per-slot gen-scratch
    // PAGE tier (I-138), freed by ~PagedVector
    // at return.
    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    DirtyState victimDirty = DirtyState::Clean;
    PagedVector<NameId> victimIds(&gArena, &victimDirty);

    StrSpan orExprOutV;
    StrSpan bodyOutV;

    // Scan every validity id (skip slot 0 = invalid, slot 1 = MAIN). Store
    // only the id; the victim's decoded scope name is re-derived at the use
    // site below via nm.decodeView + a non-minting lookup that round-trips to
    // the same id. No NameMap mint occurs anywhere in this function —
    // classifyOrScopeView takes a const NameMap& and the loop-2 mints target
    // pendingWipeScopes / intValidityNamesToFilter, not the NameMap — which
    // is doubly load-bearing now: it keeps the parameter span AND the
    // classify twin's sub-table out-spans valid across both loops (I-3;
    // orExprOutV is consumed by equalSpans in the SAME iteration, and
    // bodyOutV — like the discarded bodyOut before it — is never read; on
    // a non-Integration return the outs stay untouched, so a stale span
    // from a previous iteration exists but is never read, identical to the
    // string form's stale content).
    const int32_t nameHighWater = nm.nameCount();
    for (NameId id = 2; id <= nameHighWater; ++id) {
        if (classifyOrScopeView(nm, id, orExprOutV, bodyOutV)
                == OrScopeKind::Integration
            && nm.parentOf(id) == parentValidityId
            && equalSpans(orExprOutV, orExprSig)) {
            victimIds.push_back(id);
        }
    }

    for (int32_t i = 0; i < victimIds.size(); ++i) {
        const NameId id = victimIds[i];

        // D-72: defer the radical sweep to
        // end-of-burst drainage by performElemPhase3 —
        // immediate wipe would invalidate the kernel's in-flight
        // sortedNew loop iteration (the convergence path runs inside
        // that loop).
        {
            // Re-derive the decoded scope name from the stored id; no
            // NameMap mint occurs before this lookup, so the span is valid.
            const StrSpan vname = nm.decodeView(id);
            const NameId wipeVid = mb.nameMap.lookup(vname);
            assert(wipeVid != 0);
            mb.pendingWipeScopes.mint(wipeVid);
        }

        // D-73 (sibling-site parity):
        // parity with the same closure-time logic at the NotOrScope
        // branch of addExprToMemoryBlock.
        // Without these synchronous inserts the sortedNew loop
        // continues to admit rule-fires whose consensus scope is a
        // descendant of the victim _orint_ branch for the remainder of
        // the current burst: the ancestor-walk gate at memory.cpp
        // (checkLocalEncodedMemoryStatic) sees an empty
        // intValidityNamesToFilter for every ancestor of the victim id
        // until end-of-burst wipeSubtree's belt-and-suspenders step 11
        // populates the filters. The intervening rule-fires burn
        // sortedNew budget, mint counters, and routing-side hash work
        // inside the about-to-be-wiped subtree, and prevent the
        // post-convergence closure of the enclosing implication.
        // Filter from the moment of OR convergence; the wipe still
        // drains at end-of-burst for radical descendant cleanup.
        mb.intValidityNamesToFilter.mint(id);
    }
}

/// @brief Prefix every argument of a flat canonical MPL expression with `u_`.
///
/// @details
/// Statified twin of the former `ce::getArgs` + `std::map<std::string,
/// std::string>` version: the args are sliced as stack `StrSpan`s
/// (`getArgsSpans`) and the substitution runs through `replaceKeysToString`,
/// the byte-exact twin of `ce::replaceKeysInString` (token-boundary,
/// greedy-longest, source-scanned — so duplicate arg keys are harmless, exactly
/// as the deduping map was). Unlike `removeUPrefixFromArguments`, the
/// replacement value `"u_"+arg` is NEW bytes (a prefix prepended, not a slice
/// of `expr`), so the values are built on the per-slot scratch arena for the
/// duration of the call (rewound at return by the `ScratchScope`). The worker
/// slot comes from `g_currentCoreId` (published by the phase-1/3 workers);
/// single-threaded setup contexts read -1 and use the reserved scratch slot.
/// The returned `std::string` is the escaping result the callers consume.
///
/// @param expr Flat canonical MPL expression text.
/// @return `expr` with each argument `u_`-prefixed; an arg-less `expr` passes
///         through unchanged.
/// @see `removeUPrefixFromArguments`, `replaceKeysToString`, `getArgsSpans`,
///      `g_currentCoreId`.
std::string ExpressionAnalyzer::prefixArgumentsWithU(const std::string& expr) {
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);
    if (argN == 0) return expr;   // no args: plain pass-through (empty-map path)

    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;   // reserved slot for single-threaded
    ScratchArena& arena = scratchArenas().forSlot(slot);
    ScratchScope scope(arena);

    StrReplacement pairs[ExecutionParameters::MAX_ARITY];
    for (int ai = 0; ai < argN; ++ai) {
        const StrSpan& arg = args[ai];
        // value = "u_" + arg : NEW bytes, bump-allocated on the scratch arena.
        char* v = arena.allocBytes(2 + arg.len);
        v[0] = 'u';
        v[1] = '_';
        if (arg.len > 0) std::memcpy(v + 2, arg.ptr, static_cast<size_t>(arg.len));
        pairs[ai].key = arg;
        pairs[ai].value = StrSpan(v, 2 + arg.len);
    }
    return replaceKeysToString(expr, pairs, argN);
}

/// @brief Scratch twin of @ref prefixArgumentsWithU — `u_`-prefix EVERY argument
///        of @p expr onto the scratch tier, with no heap `std::string`.
///
/// @details
/// Prefixes ALL arguments (contrast @ref prefixNonMarkerArgumentsWithUScratch,
/// which SKIPS the literal `"marker"`). Each argument yields one substitution
/// pair whose value `"u_"+arg` is FRESH bytes bump-allocated on @p arena (the
/// `prefixArgumentsWithU` idiom); the substitution runs through
/// `replaceKeysScratch` (the byte-exact greedy-longest twin of
/// `ce::replaceKeysInString`). A no-argument input contributes zero pairs and
/// degrades to a plain pass-through copy — byte-identical to the oracle's `return
/// expr`. An already-`u_` argument is prefixed AGAIN (`u_u_arg`), exactly as the
/// string oracle does — NOT special-cased. The result rides @p arena (the caller
/// owns the `ScratchScope`) and survives a later interner mint.
///
/// @param arena Scratch (string-tier) arena receiving the fresh value bytes and
///              the substituted result.
/// @param expr  Expression whose arguments are prefixed, as a span over stable input.
/// @return The substituted text as a `ScratchString` on @p arena.
/// @invariant Byte-identical to `prefixArgumentsWithU`; mints into no interner.
/// @see prefixArgumentsWithU (the retained heap oracle),
///      prefixNonMarkerArgumentsWithUScratch (the marker-skipping sibling),
///      replaceKeysScratch, checkNecessityForEquality (the consumer).
ScratchString ExpressionAnalyzer::prefixArgumentsWithUScratch(
    ScratchArena& arena, StrSpan expr) {
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);

    StrReplacement pairs[ExecutionParameters::MAX_ARITY];
    for (int ai = 0; ai < argN; ++ai) {
        const StrSpan& arg = args[ai];
        // value = "u_" + arg : NEW bytes, bump-allocated on the scratch arena.
        char* v = arena.allocBytes(2 + arg.len);
        v[0] = 'u';
        v[1] = '_';
        if (arg.len > 0) std::memcpy(v + 2, arg.ptr, static_cast<size_t>(arg.len));
        pairs[ai].key = arg;
        pairs[ai].value = StrSpan(v, 2 + arg.len);
    }
    return replaceKeysScratch(arena, expr, pairs, argN);
}

/// @brief `u_`-prefix every NON-`"marker"` argument of `expr` onto the scratch
///        tier — the faithful twin of `isAdmittedIntegration`'s inline
///        `{arg != "marker" -> u_+arg}` rename.
///
/// @details
/// Unlike `prefixArgumentsWithU`, which prefixes ALL arguments, this helper
/// skips any argument equal to the literal `"marker"`, leaving it verbatim.
/// Each non-marker argument yields one substitution pair whose value
/// `"u_"+arg` is FRESH bytes bump-allocated on `arena`; the substitution runs
/// through `replaceKeysScratch` (the byte-exact greedy-longest twin of
/// `ce::replaceKeysInString`). A no-argument or all-`"marker"` input
/// contributes zero pairs and degrades to a plain pass-through copy. The
/// result rides `arena` (the caller owns the `ScratchScope`) and survives a
/// later interner mint (the interner writes its own cold pool, not this arena).
///
/// @param arena Scratch (string-tier) arena receiving the fresh value bytes
///              and the substituted result.
/// @param expr  Expression whose arguments are prefixed, as a span over stable
///              input.
/// @return The substituted text as a `ScratchString` on `arena`.
/// @see prefixArgumentsWithU (contrast — prefixes ALL args),
///      replaceKeysScratch, isAdmittedIntegration.
ScratchString ExpressionAnalyzer::prefixNonMarkerArgumentsWithUScratch(
    ScratchArena& arena, StrSpan expr) {
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);

    StrReplacement pairs[ExecutionParameters::MAX_ARITY];
    int pairN = 0;
    for (int ai = 0; ai < argN; ++ai) {
        const StrSpan& arg = args[ai];
        if (equalSpans(arg, StrSpan("marker", 6))) continue;   // marker stays verbatim
        // value = "u_" + arg : NEW bytes, bump-allocated on the scratch arena.
        char* v = arena.allocBytes(2 + arg.len);
        v[0] = 'u';
        v[1] = '_';
        if (arg.len > 0) std::memcpy(v + 2, arg.ptr, static_cast<size_t>(arg.len));
        pairs[pairN].key = arg;
        pairs[pairN].value = StrSpan(v, 2 + arg.len);
        ++pairN;
    }
    return replaceKeysScratch(arena, expr, pairs, pairN);
}

std::string ExpressionAnalyzer::removeUPrefixFromArguments(const std::string& expr) {
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);
    StrReplacement pairs[ExecutionParameters::MAX_ARITY];
    int pairN = 0;
    for (int ai = 0; ai < argN; ++ai) {
        const StrSpan& arg = args[ai];
        // Only strip "u_" when the argument actually starts with it. The value is
        // the sub-span arg[2:], a slice of expr -- no new bytes, so no arena.
        if (arg.len >= 2 && arg.ptr[0] == 'u' && arg.ptr[1] == '_') {
            pairs[pairN].key = arg;
            pairs[pairN].value = StrSpan(arg.ptr + 2, arg.len - 2);
            ++pairN;
        }
    }
    return replaceKeysToString(expr, pairs, pairN);
}

/// @brief Compile a mail-bound implication into its compact named form
/// `(implication<N>[args])`.
///
/// @details
/// Wrapper invoked at every site where a proved or internal implication is
/// broadcast on the mail "implications" channel. It normalises the
/// implication's free arguments and delegates to the existing
/// `compileCoreExpressionMapCore` implication branch (the `(>` branch —
/// the only branch reached for a top-level non-negated implication),
/// returning the compact instance.
///
/// Free arguments arrive in `u_`-prefixed template form for internal
/// implications; a fully-bound proved theorem carries no `u_` at all
/// (every variable is bound — see D-75). `stripUPrefixAST` removes a
/// leading `u_` from every argument token at every nesting level, yielding
/// a uniform bare-argument `(>[]()())` string before compilation. A
/// fully-bound theorem then has zero free arguments and compiles to
/// `(implication<N>[])`; an implication with genuinely-free arguments
/// compiles to `(implication<N>[args])`.
///
/// Preparatory for ASIC 0.1: the returned compact form is additionally
/// deposited as a mail expression; it has no effect on which theorems
/// prove.
///
/// No defensive fallback (CLAUDE.md Rule 19): an implication that reached
/// the mail channel and fails to compile to a simple
/// `(implication<N>[...])` is a bug and asserts at its origin — it never
/// returns an empty / sentinel string the caller would have to
/// test-and-skip.
///
/// @param implication Canonical MPL implication string — the original
///        expanded `(>[]()())` as it enters the mail channel (the 5th
///        element of a `mailOut.implications` tuple).
/// @return The compact `(implication<N>[args])` form;
///         `(implication<N>[])` for a fully-bound theorem.
/// @invariant The result is non-empty, satisfies
///            `ce::expressionIsSimple`, and its core name begins with
///            `"implication"`.
/// @see stripUPrefixAST, compileCoreExpressionMapCore, excludeRepetitions
std::string ExpressionAnalyzer::compileImplicationToCompact(const std::string& implication) {
    // Normalise free args to bare form: u_-prefixed template params (internal
    // implications) and unprefixed bound vars (proved theorems) both reduce
    // to the same shape, so compileCoreExpressionMapCore's (>) branch sees a
    // uniform input. stripUPrefixAST walks the full parse tree (every nesting
    // level), unlike removeUPrefixFromArguments which only sees the first [...].
    const std::string stripped = stripUPrefixAST(implication);

    // The (>) branch of compileCoreExpressionMapCore allocates / reuses the
    // implication<N> name and returns the compact instance. Counters are the
    // canonical this-> members so allocation stays continuous and (because
    // this runs only on the single-threaded post-join broadcast drain)
    // deterministic.
    const std::string compact = compileCoreExpressionMapCore(
        stripped, this->implCounter, this->existenceCounter,
        this->andCounter, this->orCounter, this->variableCounter);

    // CLAUDE.md Rule 19 — no defensive fallback. An implication that entered
    // mail MUST compile to a simple (implication<N>[...]); a miss is a bug we
    // trap here, with the stack intact, not a value the caller skips.
    assert(!compact.empty()
        && "compileImplicationToCompact: empty compile result for a mailed implication");
    assert(ce::expressionIsSimple(compact)
        && "compileImplicationToCompact: result is not a simple compact form");
    assert(startsWith(extractExpressionUniversalSpan(StrSpan(compact)).toStdString(), "implication", 11)
        && "compileImplicationToCompact: result core name is not implication<N>");

    return compact;
}

/// @brief Record a mail-broadcast implication for deferred, single-threaded
/// compaction (D-76, Option A).
///
/// @details
/// The eight `updateGlobalDirect` / `updateGlobal` broadcast sites call this
/// instead of compiling inline. Compiling inline is unsound: the
/// contradiction path reaches `updateGlobalDirect` via the parallel
/// `addExprToMemoryBlock(... coreId ...)` worker call, and
/// `compileImplicationToCompact` mutates process-global `implCounter` /
/// `compiledExpressions` / `repetitionExclusionMap` with no lock — a data
/// race that made the compaction non-injective and the run
/// non-deterministic (violates I-28). This call only appends to
/// `pendingCompactionQueue` under `pendingCompactionMutex` (cheap,
/// parallel-safe — no global compile state touched). A single-threaded pass
/// after `pool.join()` sorts and drains the queue, performing the actual
/// compile + mail deposit deterministically.
///
/// @param original The original expanded implication string as it enters the
///        mail channel at the call site.
/// @param kySize `ky.size()` at the call site; the deferred pass rebuilds
///        `compactLevels` as `0 .. kySize`.
/// @param coreId The broadcasting core id (no longer routes anything; the
///        deferred pass merges the compact deposit into the root's mailOut).
/// @return void.
/// @invariant Touches only `pendingCompactionQueue` (under
///            `pendingCompactionMutex`); never the global name-allocation
///            state.
/// @see compileImplicationToCompact, excludeRepetitions
void ExpressionAnalyzer::recordPendingCompaction(const std::string& original, int kySize, int coreId) {
    std::lock_guard<std::mutex> lock(this->pendingCompactionMutex);
    this->pendingCompactionQueue.push_back(std::make_tuple(original, kySize, coreId));
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

/// @brief Collect the distinct non-`u_` argument names of an existence
///        `LogicalEntity` (the "last removed" bound-var candidates) into a
///        caller span buffer, sorted-unique — 0% heap.
///
/// @details
/// The span twin of the former `std::set<std::string>` collector: over every
/// `le.elements[i]`, `getArgsSpans` slices the flat `[...]` args and each arg
/// that does NOT begin `u_` is gathered as a `StrSpan` INTO `le.elements`
/// (input lifetime). The gather buffer is then `compareSpans`-sorted and
/// adjacent-deduplicated, reproducing the former `std::set<std::string>`
/// iteration order byte-for-byte (`compareSpans` == `std::string::compare`),
/// and the distinct run is copied into @p out. No arena is reached — every
/// output span aliases @p le, which must outlive the caller's use.
///
/// @param le  Existence `LogicalEntity` (asserted `category == "existence"`).
/// @param out Caller buffer receiving the distinct sorted argument spans.
/// @param cap Capacity of @p out (loud Rule-19 assert on overflow).
/// @return The number of distinct non-`u_` arguments written to @p out.
/// @invariant Allocates nothing; output spans alias @p le.elements.
/// @see getArgsSpans, compareSpans, equalSpans.
int32_t ExpressionAnalyzer::listLastRemovedArgsLE(const LogicalEntity& le,
                                                  StrSpan* out, int32_t cap) {
    // LE forwarder onto the span-run body (the additive twin) — build a stack
    // StrSpan run over le.elements (caller-stable), byte-identical.
    StrSpan elemRun[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS + 1];
    const int32_t elemFwdN = static_cast<int32_t>(le.elements.size());
    assert(elemFwdN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS + 1
        && "listLastRemovedArgsLE forwarder: element count exceeds cap");
    for (int32_t i = 0; i < elemFwdN; ++i) elemRun[i] = StrSpan(le.elements[i]);
    return listLastRemovedArgsLE(StrSpan(le.category), elemRun, elemFwdN, out, cap);
}

int32_t ExpressionAnalyzer::listLastRemovedArgsLE(StrSpan category,
    const StrSpan* elements, int32_t elemN, StrSpan* out, int32_t cap) {
    // Check that the category is explicitly "existence"
    assert(equalSpans(category, StrSpan("existence", 9))
        && "LogicalEntity category must be 'existence'");

    // Gather every non-'u_' arg span (slices of the elements, input lifetime).
    StrSpan buf[ExecutionParameters::MAX_KEY_SLOTS];
    int32_t bufN = 0;
    for (int32_t e = 0; e < elemN; ++e) {
        const StrSpan element = elements[e];
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(element, args,
                                           ExecutionParameters::MAX_ARITY);
        for (int32_t a = 0; a < argsN; ++a) {
            // Check if arg does NOT start with "u_"
            if (!(args[a].len >= 2 && args[a].ptr[0] == 'u' && args[a].ptr[1] == '_')) {
                assert(bufN < ExecutionParameters::MAX_KEY_SLOTS
                    && "listLastRemovedArgsLE: gathered arg count exceeds cap");
                buf[bufN++] = args[a];
            }
        }
    }

    // Sort + adjacent-dedup -> the std::set<std::string> order, byte-identically.
    std::sort(buf, buf + bufN, [](const StrSpan& a, const StrSpan& b) {
        return compareSpans(a, b) < 0;
    });
    int32_t outN = 0;
    for (int32_t i = 0; i < bufN; ++i) {
        if (i == 0 || !equalSpans(buf[i], buf[i - 1])) {
            assert(outN < cap
                && "listLastRemovedArgsLE: distinct arg count exceeds out cap");
            out[outN++] = buf[i];
        }
    }
    return outN;
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void ExpressionAnalyzer::makeAdmissionKeys(const std::vector<std::string>& implication,
    const std::string& expression,
    HashMemory& localMemory,
    const std::string& validityName,
    Memory& mb)
{
    // Vector forwarder onto the span-run body (the additive twin) — build a
    // stack StrSpan run over the caller-stable vector strings, byte-identical.
    StrSpan implRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
    const int32_t implFwdN = static_cast<int32_t>(implication.size());
    assert(implFwdN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1
        && "makeAdmissionKeys forwarder: implication length exceeds cap");
    for (int32_t i = 0; i < implFwdN; ++i) implRun[i] = StrSpan(implication[i]);
    makeAdmissionKeys(implRun, implFwdN, StrSpan(expression), localMemory,
                      StrSpan(validityName), mb);
}

void ExpressionAnalyzer::makeAdmissionKeys(const StrSpan* impl, int32_t implN,
    StrSpan expr,
    HashMemory& localMemory,
    StrSpan validityName,
    Memory& mb)
{
    // 1. Check expression arguments. Args are zero-copy slices of `expr`
    //    (a caller-stable buffer) — no owning vector.
    //    Must have at least one "repl_" and all others must be "u_", "repl_", or "marker".
    StrSpan exprArgs[ExecutionParameters::MAX_ARITY];
    const int32_t exprArgsN = getArgsSpans(expr, exprArgs,
                                           ExecutionParameters::MAX_ARITY);
    bool hasRepl = false;

    for (int32_t i = 0; i < exprArgsN; ++i) {
        const StrSpan arg = exprArgs[i];
        if (startsWithSpan(arg, "repl_", 5)) {
            hasRepl = true;
        }
        else if (startsWithSpan(arg, "u_", 2)) {
            continue;
        }
        else if (equalSpans(arg, StrSpan("marker", 6))) {
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

    localMemory.triggersForAdmissionSetIntegration.mint(
        mintTemplateKey(mb.templateInterner, mb.nameMap, expr, validityName));

    // 2. Validate Head vs Expression
    if (implN == 0) return;
    const StrSpan head = impl[implN - 1];

    // A. Check Core Expression Match — compare the two extract spans directly.
    if (!equalSpans(extractExpressionUniversalSpan(head),
                    extractExpressionUniversalSpan(expr))) {
        return;
    }

    // B. Check Negation Status Match
    //    Assuming canonical form where negation always puts '!' at the start
    bool headIsNegated = (head.len > 0 && head.ptr[0] == '!');
    bool exprIsNegated = (expr.len > 0 && expr.ptr[0] == '!');

    if (headIsNegated != exprIsNegated) {
        return;
    }

    // C. Check Arguments and 1:1 "u_" relation. Head args are slices of `head`
    //    (a caller-stable buffer).
    StrSpan headArgs[ExecutionParameters::MAX_ARITY];
    const int32_t headArgsN = getArgsSpans(head, headArgs,
                                           ExecutionParameters::MAX_ARITY);
    assert(headArgsN == exprArgsN);

    // The former std::map<std::string,std::string> rename map becomes a
    // StrReplacement stack run built directly in head-arg order, keys/values
    // spans into head/expression. On a REPEATED head arg the map's
    // last-write-wins is reproduced by overwriting the matching pair's value.
    // The pair ORDER is not observable — replaceKeysScratch is greedy-longest
    // token-boundary, order-independent — so this first-seen order is
    // byte-identical to the former sorted-map iteration.
    StrReplacement replPairs[ExecutionParameters::MAX_KEY_SLOTS];
    int32_t replPairsN = 0;
    for (int32_t i = 0; i < headArgsN; ++i) {
        const StrSpan hArg = headArgs[i];
        const StrSpan eArg = exprArgs[i];

        const bool hIsU = startsWithSpan(hArg, "u_", 2);
        const bool eIsU = startsWithSpan(eArg, "u_", 2);

        // "Important: head and expression must have all 'u_' args in the same position."
        if (hIsU || eIsU) {
            if (!equalSpans(hArg, eArg)) {
                return; // Violation of 1:1 relation
            }
        }
        else {
            // Map non-u head arg to expression arg, last-write-wins on a repeat.
            int32_t slot = -1;
            for (int32_t p = 0; p < replPairsN; ++p)
                if (equalSpans(replPairs[p].key, hArg)) { slot = p; break; }
            if (slot >= 0) {
                replPairs[slot].value = eArg;
            }
            else {
                assert(replPairsN < ExecutionParameters::MAX_KEY_SLOTS
                    && "makeAdmissionKeys: rename pair count exceeds cap");
                replPairs[replPairsN].key = hArg;
                replPairs[replPairsN].value = eArg;
                ++replPairsN;
            }
        }
    }

    // 3. Apply Mapping to Implication Body (Head excluded). mappedElement and
    //    admissionKey are fresh bytes on the per-slot string-scratch arena
    //    (replaceKeysScratch is the byte-exact twin of ce::replaceKeysInString),
    //    framed per element by elemScope. The interner mints below touch
    //    templateInterner / NameMap / originInterner — never this arena — so the
    //    held spans stay valid across the door calls (I-3).
    const unsigned akSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(akSlot);
    for (int32_t i = 0; i + 1 < implN; ++i) {
        ScratchScope elemScope(strArena);
        const ScratchString mappedElement =
            replaceKeysScratch(strArena, impl[i], replPairs, replPairsN);

        // Analyze args of the mapped element — slices of mappedElement.
        StrSpan outArgs[ExecutionParameters::MAX_ARITY];
        const int32_t outArgsN = getArgsSpans(StrSpan(mappedElement), outArgs,
                                              ExecutionParameters::MAX_ARITY);
        int markerCount = 0;
        bool argsValid = true;

        for (int32_t a = 0; a < outArgsN; ++a) {
            const StrSpan arg = outArgs[a];
            if (equalSpans(arg, StrSpan("marker", 6))) {
                markerCount++;
            }
            else if (startsWithSpan(arg, "repl_", 5)) {
                continue;
            }
            else if (startsWithSpan(arg, "u_", 2)) {
                continue;
            }
            else {
                argsValid = false;
                break;
            }
        }

        // Admission Condition: Exactly one "marker" and valid args
        if (argsValid && markerCount == 1) {
            const ScratchString admissionKey = removeUPrefixScratch(
                strArena, StrSpan(mappedElement));
            localMemory.admissionSetIntegration.mint(
                mintTemplateKey(mb.templateInterner, mb.nameMap,
                                StrSpan(admissionKey), validityName));

            // New admission-set-integration key — revive any prior
            // integration-side rejection keyed at this marker form.
            // `mb.overallHashMemory.rejectedMapIntegration` is the fixed
            // lookup target; firing here is safe regardless of whether
            // `localMemory` is overallHashMemory or a local/delta mirror
            // (the overall-memory insert happens first in the enclosing
            // addToHashMemory 3-way fan-out, so by any subsequent call the
            // admission key is visible in overall memory).
            this->revisitRejectedIntegration2(StrSpan(admissionKey), mb, validityName);
        }

        // OR integration trigger: if premise is a compound OR expression with
        // only repl_/u_/marker args (no bound pi_lev_ vars), trigger integration.
        // This enables the prover to construct OR expressions from their elements.
        if (argsValid) {
            const LogicalEntity* le = compiledEntity(
                extractExpressionUniversalSpan(StrSpan(mappedElement)));
            if (le && equalSpans(StrSpan(le->category), StrSpan("or", 2))) {
                // TODO: call prepareIntegration for the OR expression (with u_ intact)
            }
        }
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)


/// @brief Expand a `LogicalEntity`'s signature into its full expression form
///        (the AND / existence / implication / OR De-Morgan build followed by
///        the `u_`/`it_`/`int_`/`c_` variable renaming) onto a caller arena — 0%
///        heap.
///
/// @details
/// The arena-builder twin of the former heap `expandSignature`. Every string is
/// built with an EXPLICIT length on @p out's byte-bump (string) tier
/// (`assert(at == n)` per the fresh-string doctrine), and the former
/// `std::set<std::string>` / `std::vector<std::string>` scratch became fixed
/// stack `StrSpan` arrays sorted by `compareSpans` (byte-identical to the
/// `std::set` iteration order). There is NO local `ScratchScope`: the returned
/// `ScratchString`, the intermediate build bytes, and the `c_` rename values all
/// ride @p out, held live until the CALLER's scope reclaims it (09c pitfall 4).
///
/// Per category: AND / OR build the left-nested `(& …)` / `!(& …)` forms
/// incrementally; EXISTENCE builds `!(>[boundVars]body negatedHead)` where the
/// bound vars are the body args NOT in the signature (sorted-unique); IMPLICATION
/// returns `reconstructImplicationFullBindScratch(out, …)` DIRECTLY (no renaming,
/// matching the heap's early return). The final renaming collects tokens via
/// `collectExprTokens` (every variable token is inside a `[...]`) and runs
/// `replaceKeysScratch` — the byte-exact arena twin of the former
/// `replaceKeysToString`.
///
/// @param le  The `LogicalEntity` to expand (category one of and/existence/
///            implication/or; signature args asserted `u_`-prefixed).
/// @param out Caller string-tier scratch arena receiving the result AND all
///            intermediate build bytes; must outlive the caller's use of the
///            returned span.
/// @return The expanded, renamed signature as a `ScratchString` on @p out.
/// @invariant Allocates on the heap NOTHING; builds only on @p out; mints no
///            interner (the callers' downstream doors mint, never @p out — I-3).
/// @see reconstructImplicationFullBindScratch, replaceKeysScratch, collectExprTokens.
ScratchString ExpressionAnalyzer::expandSignature(const LogicalEntity& le, ScratchArena& out)
{
    // LE forwarder onto the span-run body (the additive twin) — build a stack
    // StrSpan run over le.elements (caller-stable), byte-identical.
    StrSpan elemRun[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS + 1];
    const int32_t elemFwdN = static_cast<int32_t>(le.elements.size());
    assert(elemFwdN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS + 1
        && "expandSignature forwarder: element count exceeds cap");
    for (int32_t i = 0; i < elemFwdN; ++i) elemRun[i] = StrSpan(le.elements[i]);
    return expandSignature(StrSpan(le.category), StrSpan(le.signature),
                           elemRun, elemFwdN, out);
}

ScratchString ExpressionAnalyzer::expandSignature(StrSpan category,
    StrSpan signature, const StrSpan* elements, int32_t elemN, ScratchArena& out)
{
    // 1. Assert that all signature arguments start with 'u_'.
    StrSpan sigArgsVec[ExecutionParameters::MAX_ARITY];
    const int32_t sigArgsN = getArgsSpans(signature, sigArgsVec,
                                          ExecutionParameters::MAX_ARITY);
    for (int32_t i = 0; i < sigArgsN; ++i) {
        assert(sigArgsVec[i].len >= 2 && sigArgsVec[i].ptr[0] == 'u'
            && sigArgsVec[i].ptr[1] == '_'
            && "Signature arguments must start with 'u_' as per assumption.");
    }

    StrSpan result;

    // --- CASE 1: AND (&) ---
    if (equalSpans(category, StrSpan("and", 3))) {
        if (elemN == 0) result = signature;
        else if (elemN == 1) result = elements[0];
        else {
            // Build the left-nested "(& current elem)" hierarchy on `out`.
            StrSpan current = elements[0];
            for (int32_t i = 1; i < elemN; ++i) {
                const StrSpan elem = elements[i];
                const int32_t n = 2 + current.len + elem.len + 1;  // "(&" ... ")"
                char* buf = out.allocBytes(n);
                int32_t at = 0;
                buf[at++] = '('; buf[at++] = '&';
                std::memcpy(buf + at, current.ptr, static_cast<size_t>(current.len)); at += current.len;
                std::memcpy(buf + at, elem.ptr, static_cast<size_t>(elem.len)); at += elem.len;
                buf[at++] = ')';
                assert(at == n);
                current = StrSpan(buf, n);
            }
            result = current;
        }
    }

    // --- CASE 2: EXISTENCE ---
    else if (equalSpans(category, StrSpan("existence", 9))) {
        // Expecting at least 2 elements: [0]=body/definition, [1]=head/claim
        if (elemN < 2) result = signature;
        else {
            const StrSpan body = elements[0];
            const StrSpan head = elements[1];

            // Handle Head Negation: strip a leading '!' or prepend one.
            StrSpan negatedHead;
            if (head.len > 0 && head.ptr[0] == '!') {
                negatedHead = StrSpan(head.ptr + 1, head.len - 1);
            }
            else {
                const int32_t hn = 1 + head.len;
                char* hb = out.allocBytes(hn);
                hb[0] = '!';
                std::memcpy(hb + 1, head.ptr, static_cast<size_t>(head.len));
                negatedHead = StrSpan(hb, hn);
            }

            // Bound variables: body args NOT in the signature, sorted-unique
            // (byte-identical to the former std::set<std::string> order).
            StrSpan bodyArgs[ExecutionParameters::MAX_ARITY];
            const int32_t bodyArgsN = getArgsSpans(body, bodyArgs,
                                                   ExecutionParameters::MAX_ARITY);
            StrSpan bvBuf[ExecutionParameters::MAX_ARITY];
            int32_t bvN = 0;
            for (int32_t i = 0; i < bodyArgsN; ++i) {
                bool inSig = false;
                for (int32_t j = 0; j < sigArgsN; ++j)
                    if (equalSpans(bodyArgs[i], sigArgsVec[j])) { inSig = true; break; }
                if (!inSig) {
                    assert(bvN < ExecutionParameters::MAX_ARITY
                        && "expandSignature: bound-var count exceeds MAX_ARITY");
                    bvBuf[bvN++] = bodyArgs[i];
                }
            }
            std::sort(bvBuf, bvBuf + bvN, [](const StrSpan& a, const StrSpan& b) {
                return compareSpans(a, b) < 0;
            });
            StrSpan boundVars[ExecutionParameters::MAX_ARITY];
            int32_t boundN = 0;
            for (int32_t i = 0; i < bvN; ++i)
                if (i == 0 || !equalSpans(bvBuf[i], bvBuf[i - 1]))
                    boundVars[boundN++] = bvBuf[i];

            // result = "!(>[" + join(boundVars, ",") + "]" + body + negatedHead + ")"
            int32_t varsLen = 0;
            for (int32_t i = 0; i < boundN; ++i) varsLen += boundVars[i].len;
            if (boundN > 0) varsLen += boundN - 1;  // separating commas
            const int32_t n = 4 + varsLen + 1 + body.len + negatedHead.len + 1;
            char* buf = out.allocBytes(n);
            int32_t at = 0;
            buf[at++] = '!'; buf[at++] = '('; buf[at++] = '>'; buf[at++] = '[';
            for (int32_t i = 0; i < boundN; ++i) {
                if (i > 0) buf[at++] = ',';
                std::memcpy(buf + at, boundVars[i].ptr, static_cast<size_t>(boundVars[i].len));
                at += boundVars[i].len;
            }
            buf[at++] = ']';
            std::memcpy(buf + at, body.ptr, static_cast<size_t>(body.len)); at += body.len;
            std::memcpy(buf + at, negatedHead.ptr, static_cast<size_t>(negatedHead.len)); at += negatedHead.len;
            buf[at++] = ')';
            assert(at == n);
            result = StrSpan(buf, n);
        }
    }

    // --- CASE 3: IMPLICATION (returns early, before the renaming) ---
    else if (equalSpans(category, StrSpan("implication", 11))) {
        if (elemN == 0) return copyScratch(out, signature);
        // chain = elements minus the last (head); no owning std::vector.
        const StrSpan head = elements[elemN - 1];
        const int32_t chainN = elemN - 1;
        assert(chainN <= 64 && "expandSignature: implication chain exceeds 64");
        StrSpan chainSpans[64];
        for (int32_t i = 0; i < chainN; ++i) chainSpans[i] = elements[i];
        return reconstructImplicationFullBindScratch(out, chainSpans, chainN, head);
    }

    // --- CASE 4: OR ---
    else if (equalSpans(category, StrSpan("or", 2))) {
        if (elemN == 0) result = signature;
        else if (elemN == 1) result = elements[0];
        else {
            // Rebuild nested !(&!(...) !(...)) from the flat disjunct list on `out`.
            const StrSpan e0 = elements[0];
            const StrSpan e1 = elements[1];
            StrSpan current;
            {
                const int32_t n = 4 + e0.len + 1 + e1.len + 1;  // "!(&!" e0 "!" e1 ")"
                char* buf = out.allocBytes(n);
                int32_t at = 0;
                buf[at++] = '!'; buf[at++] = '('; buf[at++] = '&'; buf[at++] = '!';
                std::memcpy(buf + at, e0.ptr, static_cast<size_t>(e0.len)); at += e0.len;
                buf[at++] = '!';
                std::memcpy(buf + at, e1.ptr, static_cast<size_t>(e1.len)); at += e1.len;
                buf[at++] = ')';
                assert(at == n);
                current = StrSpan(buf, n);
            }
            for (int32_t i = 2; i < elemN; ++i) {
                const StrSpan elem = elements[i];
                const int32_t n = 3 + current.len + 1 + elem.len + 1;  // "!(&" current "!" elem ")"
                char* buf = out.allocBytes(n);
                int32_t at = 0;
                buf[at++] = '!'; buf[at++] = '('; buf[at++] = '&';
                std::memcpy(buf + at, current.ptr, static_cast<size_t>(current.len)); at += current.len;
                buf[at++] = '!';
                std::memcpy(buf + at, elem.ptr, static_cast<size_t>(elem.len)); at += elem.len;
                buf[at++] = ')';
                assert(at == n);
                current = StrSpan(buf, n);
            }
            result = current;
        }
    }

    // --- ERROR ---
    else {
        assert(false && "Forbidden category encountered in expandSignature. Must be 'and', 'existence', 'implication', or 'or'.");
        return copyScratch(out, signature);
    }

    // --- FINAL RENAMING (Implication handled by its early return above) ---
    // Every variable token in `result` is inside some "[...]", so collectExprTokens
    // (a flat bracket scan) yields the same token set as the former ce::parseExpr
    // tree walk. Rules: a u_ arg loses the u_ (a slice); it_/int_ args are kept;
    // every other arg gains a c_ prefix (fresh bytes on `out`, held live -- NO local
    // ScratchScope, the caller owns `out`). replaceKeysScratch is the byte-exact
    // arena twin of the former replaceKeysToString.
    StrReplacement pairs[512];
    int32_t pairN = 0;
    int esLevel = 0, esId = 0;
    collectExprTokens(result, [&](const StrSpan& t) {
        if (t.len >= 2 && t.ptr[0] == 'u' && t.ptr[1] == '_') {
            assert(pairN < 512 && "expandSignature rename pairs exceed 512");
            pairs[pairN++] = StrReplacement{ t, StrSpan(t.ptr + 2, t.len - 2) };
        }
        else if (matchItLevId(t, esLevel, esId) || matchIntLevId(t, esLevel, esId)) {
            // it_ / int_ tokens are kept as-is (no replacement).
        }
        else {
            assert(pairN < 512 && "expandSignature rename pairs exceed 512");
            char* cval = out.allocBytes(2 + t.len);
            cval[0] = 'c'; cval[1] = '_';
            std::memcpy(cval + 2, t.ptr, static_cast<size_t>(t.len));
            pairs[pairN++] = StrReplacement{ t, StrSpan(cval, 2 + t.len) };
        }
    });

    return replaceKeysScratch(out, result, pairs, pairN);
}

// In GL_Quick_VS/GL_Quick/src/analyze_expressions.cpp

/// @brief Flatten one compiled OR entity into its ordered atomic leaves in the
///        current working instruction.
///
/// @details
/// The instantiated `WorkInstruction` already contains the outer OR and every
/// recursively prepared child entity. This function walks that read-only graph
/// without re-parsing or re-substituting strings: a child whose exact signature
/// resolves to another `or` entity is expanded immediately in the same call;
/// any non-OR child is appended as one leaf. Element order is preserved across
/// both left- and right-nested shapes. Duplicate entity records must be
/// structurally identical, recursion-path cycles assert, and capacity overflow
/// asserts before a write. The local recursive lambda is stack-only and the
/// function mints and allocates nothing.
///
/// @param instructions The instantiated, read-only instruction graph.
/// @param rootEntity The entity index of the outer OR.
/// @param out Caller array receiving ordered leaf spans.
/// @param cap Capacity of @p out.
/// @return Number of leaves written; always at least two.
/// @invariant Every returned span aliases `instructions`; the outer entity and
///            every recursively entered entity are category `or` with at least
///            two elements.
/// @see ExpressionAnalyzer::disintegrateExprCore2.
int32_t ExpressionAnalyzer::flattenOrLeaves(
    const WorkInstruction& instructions,
    int32_t rootEntity,
    StrSpan* out,
    int32_t cap) const
{
    assert(rootEntity >= 0 && rootEntity < instructions.entityCount());
    assert(out != nullptr);
    assert(cap >= 2);
    assert(cap <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
    assert(equalSpans(instructions.category(rootEntity), StrSpan("or", 2)));

    StrSpan recursionPath[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    int32_t recursionDepth = 0;
    int32_t outCount = 0;

    const auto walk = [&](int32_t entityIndex, const auto& self) -> void {
        assert(entityIndex >= 0 && entityIndex < instructions.entityCount());
        assert(equalSpans(instructions.category(entityIndex), StrSpan("or", 2)));
        const StrSpan signature = instructions.signature(entityIndex);
        for (int32_t i = 0; i < recursionDepth; ++i) {
            assert(!equalSpans(recursionPath[i], signature)
                && "flattenOrLeaves: cyclic OR instruction");
        }
        assert(recursionDepth < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
            && "flattenOrLeaves: OR depth exceeds cap");
        recursionPath[recursionDepth++] = signature;

        const int32_t elementCount = instructions.elemCount(entityIndex);
        assert(elementCount >= 2
            && "flattenOrLeaves: OR entity has fewer than two elements");
        for (int32_t i = 0; i < elementCount; ++i) {
            const StrSpan element = instructions.elemAt(entityIndex, i);
            int32_t childEntity = -1;
            for (int32_t candidate = 0;
                 candidate < instructions.entityCount(); ++candidate) {
                if (!equalSpans(instructions.signature(candidate), element)) continue;
                if (childEntity < 0) {
                    childEntity = candidate;
                    continue;
                }

                assert(equalSpans(instructions.category(childEntity),
                                  instructions.category(candidate))
                    && instructions.elemCount(childEntity)
                        == instructions.elemCount(candidate)
                    && "flattenOrLeaves: duplicate signature differs");
                for (int32_t j = 0; j < instructions.elemCount(childEntity); ++j) {
                    assert(equalSpans(instructions.elemAt(childEntity, j),
                                      instructions.elemAt(candidate, j))
                        && "flattenOrLeaves: duplicate signature differs");
                }
            }

            if (childEntity >= 0
                && equalSpans(instructions.category(childEntity), StrSpan("or", 2))) {
                self(childEntity, self);
            }
            else {
                assert(outCount < cap
                    && "flattenOrLeaves: leaf count exceeds cap");
                out[outCount++] = element;
            }
        }

        --recursionDepth;
    };

    walk(rootEntity, walk);
    assert(recursionDepth == 0);
    assert(outCount >= 2);
    return outCount;
}

void ExpressionAnalyzer::disintegrateExprCore2(StrSpan expr,
    const WorkInstruction& instructions,
    Memory& memoryBlock,
    int iteration,
    CollectedArena& collected,
    NewVarStore& newVarMap,
    StrSpan validityName,
    CollectedArena& orBranchStatements,
    bool trackHistoryLocal,
    bool allowOrDisintegration)
{
    // Capture startInt at the start of core() as reference
    int referenceStartInt = memoryBlock.startInt;

    if (iteration == -1)
    {
        iteration = 0;
    }

    // Per-slot string scratch for this level's leaf calculation strings; a
    // ScratchScope frees them when this invocation returns. collected /
    // orBranchStatements (the separate genScratchArenas page tier) hold interned
    // copies, so nothing this level builds needs to outlive it. Recursion nests
    // scopes by stack discipline -- an inner level allocates above this mark and
    // rewinds to its own, never touching these spans.
    const unsigned coreSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(coreSlot);
    ScratchScope coreScope(sArena);
    // "!" + s : a one-byte prefix (new bytes), built on the scratch arena.
    const auto prefixBang = [&](const StrSpan& s) -> StrSpan {
        char* b = sArena.allocBytes(1 + s.len);
        b[0] = '!';
        std::memcpy(b + 1, s.ptr, static_cast<size_t>(s.len));
        return StrSpan(b, 1 + s.len);
    };

    const ScratchString addedExpression = addMissingUScratch(sArena, StrSpan(expr));
    const StrSpan addedSpan(addedExpression);

    // 1. Find the instruction corresponding to the current expression — an index
    // scan over the WorkInstruction's span accessors (hit < 0 == the former
    // instructions.end() sentinel).
    int32_t hit = -1;
    for (int32_t i = 0; i < instructions.entityCount(); ++i) {
        if (equalSpans(instructions.signature(i), addedSpan)) { hit = i; break; }
    }

    // Determine the current statement signature for the map key
    ScratchString currentStatement;
    if (hit < 0) {
        // ---- Negated existence expansion ----
        // !(existence2[args]) is not in instructions, but (existence2[args]) is.
        // existence category encodes !(>[bound](left)!(right)), so:
        //   !(existence2[args]) = !!(>[bound](left)!(right)) = (>[bound](left)!(right))
        // which is an implication: from left derive !(right).
        if (addedSpan.len >= 2 && addedSpan.ptr[0] == '!' && addedSpan.ptr[1] == '(') {
            const StrSpan innerExpr(addedSpan.ptr + 1, addedSpan.len - 1);
            int32_t innerHit = -1;
            for (int32_t ii = 0; ii < instructions.entityCount(); ++ii) {
                if (equalSpans(instructions.signature(ii), innerExpr)) { innerHit = ii; break; }
            }
            if (innerHit >= 0
                && equalSpans(instructions.category(innerHit), StrSpan("existence", 9))
                && instructions.elemCount(innerHit) == 2) {
                // elements[0] = left  (e.g. (in[pi_lev_0_X,u_1]))
                // elements[1] = right (e.g. (in2[pi_lev_0_X,u_7,u_3]))
                // The pi_lev_ bound variable must be replaced with a fresh integer
                // so that renamingChain (which expects pure-integer changeables) works.
                StrSpan boundVar;
                for (int32_t ie = 0; ie < instructions.elemCount(innerHit); ++ie) {
                    const StrSpan elem = instructions.elemAt(innerHit, ie);
                    StrSpan eArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t eN = getArgsSpans(elem, eArgs, ExecutionParameters::MAX_ARITY);
                    for (int32_t k = 0; k < eN; ++k) {
                        if (!(eArgs[k].len >= 2 && eArgs[k].ptr[0] == 'u' && eArgs[k].ptr[1] == '_')) {
                            boundVar = eArgs[k]; break;
                        }
                    }
                    if (!boundVar.empty()) break;
                }
                char freshIntBuf[16];
                const int freshIntLen = std::snprintf(freshIntBuf, sizeof(freshIntBuf), "%d",
                              memoryBlock.startInt++);
                const StrReplacement bvPair{ boundVar,
                                             StrSpan(freshIntBuf, freshIntLen) };

                const ScratchString left  = replaceKeysScratch(sArena, instructions.elemAt(innerHit, 0), &bvPair, 1);
                const ScratchString right = replaceKeysScratch(sArena, instructions.elemAt(innerHit, 1), &bvPair, 1);
                const StrSpan leftSpan(left), rightSpan(right);

                // Two implications from !(>[bound](left)!(right)) = (>[bound](left)!(right)):
                //   1) left -> !(right)
                //   2) right -> !(left)
                currentStatement = removeUPrefixScratch(sArena, addedSpan);

                const ScratchString impl1 = reconstructImplicationFullBindScratch(sArena, &leftSpan, 1, prefixBang(rightSpan));
                const ScratchString impl2 = reconstructImplicationFullBindScratch(sArena, &rightSpan, 1, prefixBang(leftSpan));
                collected.insertImpl(StrSpan(currentStatement), StrSpan(impl1), StrSpan(validityName));
                collected.insertImpl(StrSpan(currentStatement), StrSpan(impl2), StrSpan(validityName));

                if (parameters.trackHistory && trackHistoryLocal) {
                    int maxOrig = parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr;
                    // Expansion origin: negated existence -> implications directly
                    // (same pattern as other expansions: source expression -> products)
                    // L3 span-record door. The "expansion" antecedent and the
                    // two impl KEYs are u_-stripped onto sArena (removeUPrefix-
                    // Scratch, the ScratchString twin), held as named locals so
                    // the spans stay live across both door calls; the door mints
                    // into originInterner, never sArena, so no span dangles (I-3).
                    const ScratchString expClean = removeUPrefixScratch(sArena, StrSpan(expr));
                    const ScratchString impl1Clean = removeUPrefixScratch(sArena, StrSpan(impl1));
                    const ScratchString impl2Clean = removeUPrefixScratch(sArena, StrSpan(impl2));
                    const OriginDep expDeps[1] = { { StrSpan(expClean), StrSpan(validityName) } };
                    addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner,
                        StrSpan(impl1Clean), StrSpan(validityName), OriginTag::expansion, expDeps, 1, maxOrig);
                    addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner,
                        StrSpan(impl2Clean), StrSpan(validityName), OriginTag::expansion, expDeps, 1, maxOrig);
                    // mailOut.exprOriginMap writes retired (single-writer
                    // policy; fillMailOut handles the outbound copy from
                    // the delta).
                }

                return;
            }
        }

        currentStatement = removeUPrefixScratch(sArena, addedSpan);
        collected.ensureKey(StrSpan(currentStatement)); // Ensure entry exists
        return;
    }

    // Read the hit entity's fields via the WorkInstruction span accessors (no
    // LogicalEntity materialized); the spans alias the read-only `instructions`
    // interner, stable for the whole call.
    const StrSpan entCategory = instructions.category(hit);
    const StrSpan entSignature = instructions.signature(hit);
    StrSpan entElems[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    const int32_t entElemN = instructions.elemCount(hit);
    assert(entElemN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
        && "disintegrateExprCore2: element count exceeds cap");
    for (int32_t j = 0; j < entElemN; ++j) entElems[j] = instructions.elemAt(hit, j);
    currentStatement = removeUPrefixScratch(sArena, entSignature);
    collected.ensureKey(StrSpan(currentStatement)); // Ensure entry exists

    // --- LAMBDA: History Tracking ---
    // Handles "expansion" origin for the entity, and optionally "disintegration"
    // origin for its children — span form (category / signature / elements run),
    // 0% heap via the expandSignature span overload.
    auto trackExpansionHistory = [&](StrSpan thCategory, StrSpan thSignature,
        const StrSpan* thElems, int32_t thElemN, bool trackChildren)
        {
            if (!parameters.trackHistory) return;
            // Hypothetical disintegration path: suppress all history writes.
            // disintegrateExprHypothetically uses disintegrateExpr2 only to
            // discover structure; its products are throw-away and must never
            // feed exprOriginMap / mailOut.exprOriginMap (buildStack followed
            // such an orphan origin and crashed).
            if (!trackHistoryLocal) return;

            // 1. Record Expansion Origin.
            // L3 span-record doors. The "expansion" antecedent is u_-stripped
            // onto sArena (removeUPrefixScratch, the ScratchString twin of the
            // heap removeUPrefixFromArguments); expandedSignature is a stable
            // local. The doors mint into originInterner, never sArena (I-3).
            const int expCap = (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr);
            const ScratchString exprClean = removeUPrefixScratch(sArena, StrSpan(expr));
            const OriginDep expDeps[1] = { { StrSpan(exprClean), StrSpan(validityName) } };

            const ScratchString expandedSignature = expandSignature(
                thCategory, thSignature, thElems, thElemN, sArena);

            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, StrSpan(expandedSignature), StrSpan(validityName), OriginTag::expansion, expDeps, 1, expCap);
            // Paired `mailOut.exprOriginMap` write. The expansion conjunction
            // is NEVER added to `intEncodedStatements` /
            // `intLocalEncodedStatementsDelta`, so `fillMailOut`'s delta loop
            // cannot ship it. Without this direct write, the receiver's
            // `buildStack` (chapter export) walks from a child's
            // `disintegration | <conjunction>` origin to the conjunction,
            // looks it up in `exprOriginMap`, finds nothing, and asserts.
            memoryBlock.addMailOutOrigin(StrSpan(expandedSignature),
                StrSpan(validityName), OriginTag::expansion, expDeps, 1, expCap);

            // 2. Record Disintegration Origin for Children (if requested)
            if (trackChildren) {
                const OriginDep disDeps[1] = { { StrSpan(expandedSignature), StrSpan(validityName) } };
                for (int32_t te = 0; te < thElemN; ++te) {
                    const StrSpan elem = thElems[te];
                    const ScratchString elemClean = removeUPrefixScratch(sArena, elem);

                    addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, StrSpan(elemClean), StrSpan(validityName), OriginTag::disintegration, disDeps, 1, expCap);
                    // Paired `mailOut.exprOriginMap` write. Even though each
                    // `elem` eventually enters `intLocalEncodedStatementsDelta`
                    // via recursive `disintegrateExprCore2`, some intermediate
                    // disintegration products do not directly hit the delta in
                    // this step (they are processed later in the recursion), so
                    // a delta-only path misses them.
                    memoryBlock.addMailOutOrigin(StrSpan(elemClean),
                        StrSpan(validityName), OriginTag::disintegration,
                        disDeps, 1, expCap);
                }
            }
        };
    // --------------------------------

    if (equalSpans(entCategory, StrSpan("implication", 11))) {
        assert(entElemN > 0 && "Implication category cannot be empty!");

        const int32_t keyCount = entElemN - 1;
        StrSpan keySpans[64];
        assert(keyCount <= 64 && "implication premise chain exceeds 64");
        for (int32_t ki = 0; ki < keyCount; ++ki) keySpans[ki] = entElems[ki];

        // Add implication specifically to THIS statement's entry (first set)
        collected.insertImpl(StrSpan(currentStatement),
            StrSpan(reconstructImplicationFullBindScratch(sArena, keySpans,
                keyCount, entElems[entElemN - 1])),
            StrSpan(validityName));

        // D-237: a disjunction-shaped premise element also
        // yields its per-leaf INTRO implications — `(>[bound](D_k)(orPremise))`,
        // one per flattened leaf — so a single true disjunct can state the
        // disjunction as a fact and fire this implication's rule. The K
        // mutual-exclusion rules (D-211) are the elimination half of an OR's
        // flat consumption; these are the introduction half. The leaf and the
        // or head are fed in RAW element form — never u_-stripped — so u_
        // arguments stay free and only the changeable arguments bind
        // (reconstructImplicationFullBindScratch drops u_ args from the
        // binder), preserving each argument's u_ status exactly as it appears
        // in the or premise.
        for (int32_t ki = 0; ki < keyCount; ++ki) {
            const StrSpan orPremise = entElems[ki];
            // The instruction graph carries no entities for an implication's
            // elements (prepareIntegrationCore recurses only into
            // existence/and/or), so the or premise resolves through the
            // compiled-definition map: a defined nullptr / non-or miss means
            // an atomic or non-or premise, not a failure.
            if (orPremise.len < 1 || orPremise.ptr[0] != '(') continue;
            const LogicalEntity* orLe =
                compiledEntity(extractExpressionUniversalSpan(orPremise));
            if (orLe == nullptr || orLe->category != "or") continue;

            // Flatten the or premise's contiguous nested-or leaves with the
            // premise's argument instantiation — the compiled-map twin of
            // flattenOrLeaves. Each level substitutes its signature args by
            // the instance args (fresh bytes on the string tier, cookbook
            // §4), recursing into or-category elements only.
            ScratchString leafHold[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            StrSpan introLeaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            int32_t introLeafN = 0;
            const auto flattenCompiledOr = [&](const LogicalEntity& node,
                                               StrSpan instance,
                                               const auto& self) -> void {
                StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
                const int32_t sigN = getArgsSpans(StrSpan(node.signature),
                    sigArgs, ExecutionParameters::MAX_ARITY);
                StrSpan instArgs[ExecutionParameters::MAX_ARITY];
                const int32_t instN = getArgsSpans(instance, instArgs,
                    ExecutionParameters::MAX_ARITY);
                assert(sigN == instN
                    && "or-intro: instance arity differs from compiled or");
                StrReplacement pairs[ExecutionParameters::MAX_ARITY];
                int32_t pairN = 0;
                for (int32_t a = 0; a < sigN; ++a) {
                    pairs[pairN].key = sigArgs[a];
                    pairs[pairN].value = instArgs[a];
                    ++pairN;
                }
                for (const std::string& rawElem : node.elements) {
                    const ScratchString subst = replaceKeysScratch(
                        sArena, StrSpan(rawElem), pairs, pairN);
                    const StrSpan substSpan(subst);
                    const LogicalEntity* childLe =
                        (substSpan.len >= 1 && substSpan.ptr[0] == '(')
                        ? compiledEntity(
                              extractExpressionUniversalSpan(substSpan))
                        : nullptr;
                    if (childLe != nullptr && childLe->category == "or") {
                        self(*childLe, substSpan, self);
                    } else {
                        assert(introLeafN
                                < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                            && "or-intro: leaf count exceeds cap");
                        leafHold[introLeafN] = subst;
                        introLeaves[introLeafN] = StrSpan(leafHold[introLeafN]);
                        ++introLeafN;
                    }
                }
            };
            flattenCompiledOr(*orLe, orPremise, flattenCompiledOr);
            assert(introLeafN >= 2
                && "or-intro: compiled or has fewer than two leaves");

            // History dep, loop-invariant: the intro rule unfolds from the
            // compact implication's compiled definition (its or premise's
            // structure), so the dep is the compact statement itself — the
            // same chapter-capable antecedent trackExpansionHistory cites for
            // the main rule's expansion record.
            const ScratchString introDepClean =
                removeUPrefixScratch(sArena, StrSpan(expr));
            const OriginDep introDeps[1] = {
                { StrSpan(introDepClean), StrSpan(validityName) } };
            const int introCap = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr;

            for (int32_t li = 0; li < introLeafN; ++li) {
                // I-4 tripwire: every changeable (non-u_) argument of the or
                // head must occur in the leaf premise, else the single-premise
                // reconstruction would leave it unbound. The current or
                // operator family repeats the element variable in every
                // disjunct; a shape that does not is a new design to raise,
                // never a silent emission.
                {
                    StrSpan headArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t headArgN = getArgsSpans(orPremise, headArgs,
                        ExecutionParameters::MAX_ARITY);
                    StrSpan leafArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t leafArgN = getArgsSpans(introLeaves[li],
                        leafArgs, ExecutionParameters::MAX_ARITY);
                    for (int32_t ha = 0; ha < headArgN; ++ha) {
                        if (headArgs[ha].len >= 2 && headArgs[ha].ptr[0] == 'u'
                            && headArgs[ha].ptr[1] == '_') {
                            continue;
                        }
                        bool inLeaf = false;
                        for (int32_t la = 0; la < leafArgN && !inLeaf; ++la) {
                            inLeaf = equalSpans(leafArgs[la], headArgs[ha]);
                        }
                        assert(inLeaf
                            && "or-intro: head variable absent from leaf premise");
                        (void)inLeaf;
                    }
                }

                const ScratchString introStr =
                    reconstructImplicationFullBindScratch(
                        sArena, &introLeaves[li], 1, orPremise);
                collected.insertImpl(StrSpan(currentStatement),
                    StrSpan(introStr), StrSpan(validityName));

                if (parameters.trackHistory && trackHistoryLocal) {
                    // KEY u_-stripped like every chapter-facing rule form; the
                    // u_-prefixed form lives only in hash memory.
                    const ScratchString introClean =
                        removeUPrefixScratch(sArena, StrSpan(introStr));
                    addOriginEncoded(memoryBlock.exprOriginMap,
                        memoryBlock.originInterner, StrSpan(introClean),
                        StrSpan(validityName), OriginTag::expansion,
                        introDeps, 1, introCap);
                }
            }
        }

        // Use Lambda (Children not tracked for implication in original code)
        trackExpansionHistory(entCategory, entSignature, entElems, entElemN, false);
    }
    else if (equalSpans(entCategory, StrSpan("and", 3))) {
        // Insert elements into the second set, removing the "u_" prefix first
        for (int32_t i = 0; i < entElemN; ++i) {
            collected.insertChild(StrSpan(currentStatement),
                StrSpan(removeUPrefixScratch(sArena, entElems[i])));
        }

        // Recurse on all elements
        for (int32_t i = 0; i < entElemN; ++i) {
            disintegrateExprCore2(entElems[i], instructions, memoryBlock, iteration, collected, newVarMap, validityName, orBranchStatements, trackHistoryLocal, allowOrDisintegration);
        }

        // Use Lambda (Children tracked for AND)
        trackExpansionHistory(entCategory, entSignature, entElems, entElemN, true);
    }
    else if (equalSpans(entCategory, StrSpan("existence", 9)))
    {
        StrSpan removedArgs[ExecutionParameters::MAX_KEY_SLOTS];
        const int32_t removedN = listLastRemovedArgsLE(entCategory, entElems,
            entElemN, removedArgs, ExecutionParameters::MAX_KEY_SLOTS);
        if (removedN == 0) {
            for (int32_t i = 0; i < entElemN; ++i) {
                disintegrateExprCore2(entElems[i], instructions, memoryBlock, iteration, collected, newVarMap, validityName, orBranchStatements, trackHistoryLocal, allowOrDisintegration);
            }
            return;
        }

        assert(removedN == 1);

        // There is no status=3 existence-disintegration ban HERE.
        // status=3 admission is governed solely OUTSIDE
        // disintegrateExpr2 — only compact "(implication<N>...)" forms
        // are admitted (see the addExprToMemoryBlock call-site guard).

        // boundVar aliases removedArgs[0] -> entElems -> the read-only
        // `instructions` interner, stable for the whole call (0% heap).
        const StrSpan boundVar = removedArgs[0];

        // Helper lambda to process a specific path (Unconditional Generation).
        // `base` is the outer WorkInstruction, read-only; each call rebuilds a
        // FRESH renamed WorkInstruction from the UNMUTATED base, reproducing the
        // two independent by-value copies the heap version made.
        auto processPath = [&](StrSpan newVar, const WorkInstruction& base) {
            // ppScope holds renamedElemSpans for the whole path (freed per call,
            // mirroring the former heap renamedElements RAII). The nested
            // rnStrScope below marks ABOVE these, so they survive the `renamed`
            // build's rewind (09c pitfall 4).
            ScratchScope ppScope(sArena);

            // Renamed elements (the recursion targets) as a stack StrSpan run on
            // the string tier — each is a replaceKeysScratch (byte-exact twin of
            // replaceKeysToString).
            const StrReplacement rnPair{ boundVar, newVar };
            StrSpan renamedElemSpans[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            assert(entElemN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "disintegrateExprCore2 processPath: element count exceeds cap");
            for (int32_t i = 0; i < entElemN; ++i)
                renamedElemSpans[i] = StrSpan(replaceKeysScratch(sArena, entElems[i], &rnPair, 1));

            // CRUCIAL: Map the new variable to its immediate defining elements.
            // newVar is brand-new (fresh startInt), so its elements append
            // consecutively (appendToTail's current-last-or-new contract).
            for (int32_t i = 0; i < entElemN; ++i)
                newVarMap.addElement(newVar, renamedElemSpans[i]);

            assert(!startsWithSpan(boundVar, "u_", 2));
            char uNewVar[256];
            assert(newVar.len + 3 <= static_cast<int32_t>(sizeof(uNewVar)));
            uNewVar[0] = 'u'; uNewVar[1] = '_';
            if (newVar.len > 0) std::memcpy(uNewVar + 2, newVar.ptr, static_cast<size_t>(newVar.len));
            const int32_t uNewVarLen = 2 + newVar.len;
            const StrReplacement u2Pair{ boundVar, StrSpan(uNewVar, uNewVarLen) };

            // Rebuild a FRESH renamed WorkInstruction (the working form is
            // read-only after commit, never mutated in place — I-133). The
            // boundVar -> u_newVar rename runs through replaceKeysScratch (the
            // byte-exact twin of the heap replaceKeysToString) on the string
            // tier; every field interns into the fresh instruction's OWN
            // throwaway arena ColdHashSet (NOT NameMap/valueInterner/etc — I-84).
            // `renamed` rides the genScratchArenas page tier, so it survives the
            // recursion and the string-tier rewind below.
            const unsigned rnStrSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : scratchArenas().slotCount() - 1;
            const unsigned rnGenSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : genScratchArenas().slotCount() - 1;
            ScratchArena& rnStrArena = scratchArenas().forSlot(rnStrSlot);
            ScratchScope rnStrScope(rnStrArena);
            WorkInstruction renamed(&genScratchArenas().forSlot(rnGenSlot));
            for (int32_t i = 0; i < base.entityCount(); ++i) {
                const int32_t start = renamed.elemMark();
                for (int32_t j = 0; j < base.elemCount(i); ++j)
                    renamed.addElement(StrSpan(
                        replaceKeysScratch(rnStrArena, base.elemAt(i, j), &u2Pair, 1)));
                renamed.commitEntity(base.category(i),
                    StrSpan(replaceKeysScratch(rnStrArena, base.signature(i), &u2Pair, 1)),
                    base.definedSet(i), base.arity(i), start);
            }
            renamed.setMarkedGoal(base.markedGoal());

            // Recurse using the renamed instruction.
            for (int32_t i = 0; i < entElemN; ++i) {
                disintegrateExprCore2(renamedElemSpans[i], renamed, memoryBlock, iteration, collected, newVarMap, validityName, orBranchStatements, trackHistoryLocal, allowOrDisintegration);
            }
            };

        // Check if at least one element is an operator expression (transparent
        // operators set -> zero-allocation string_view probe).
        bool hasOperator = false;
        for (int32_t i = 0; i < entElemN; ++i) {
            const StrSpan core = extractExpressionSpan(entElems[i]);
            if (this->operators.find(std::string_view(core.ptr, core.len))
                != this->operators.end()) {
                hasOperator = true;
                break;
            }
        }

        // 1. Generate "it_" path (Algebra) - Conditional
        if (hasOperator)
        {
            // newVar = "it_<iteration>_lev_<level>_<startInt>" -> a ScratchString
            // on the string tier (snprintf then copyFrom; %d == std::to_string for
            // the non-negative counters). startInt captured BEFORE the ++.
            char itBuf[96];
            const int itLen = std::snprintf(itBuf, sizeof(itBuf),
                "it_%d_lev_%d_%d", iteration, memoryBlock.level, memoryBlock.startInt);
            assert(itLen > 0 && itLen < static_cast<int>(sizeof(itBuf)));
            const ScratchString newVar = ScratchString::copyFrom(sArena, itBuf, itLen);

            memoryBlock.startInt++;

            if (parameters.trackHistory && trackHistoryLocal)
            {
                ScratchScope mhScope(sArena);
                assert(!startsWithSpan(boundVar, "u_", 2));
                const StrReplacement rm3Pair{ boundVar, StrSpan(newVar) };
                // No entity copy: the renamed elements are a stack StrSpan run
                // (signature unchanged in the rename, only the elements differ).
                StrSpan modElemSpans[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                for (int32_t i = 0; i < entElemN; ++i)
                    modElemSpans[i] = StrSpan(replaceKeysScratch(sArena, entElems[i], &rm3Pair, 1));

                // Use Lambda (Children tracked for Algebra/Operator path)
                trackExpansionHistory(entCategory, entSignature, modElemSpans, entElemN, true);
            }

            processPath(StrSpan(newVar), instructions);
        }

        // 2. Unconditionally generate "int_" path (Integration)
        {
            char intBuf[96];
            const int intLen = std::snprintf(intBuf, sizeof(intBuf),
                "int_lev_%d_%d", memoryBlock.level, memoryBlock.startInt);
            assert(intLen > 0 && intLen < static_cast<int>(sizeof(intBuf)));
            const ScratchString newVar = ScratchString::copyFrom(sArena, intBuf, intLen);

            memoryBlock.startInt++;

            if (parameters.trackHistory && trackHistoryLocal)
            {
                ScratchScope mhScope(sArena);
                assert(!startsWithSpan(boundVar, "u_", 2));
                const StrReplacement rm3Pair{ boundVar, StrSpan(newVar) };
                StrSpan modElemSpans[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                for (int32_t i = 0; i < entElemN; ++i)
                    modElemSpans[i] = StrSpan(replaceKeysScratch(sArena, entElems[i], &rm3Pair, 1));

                // Use Lambda (Children tracked for Integration path)
                trackExpansionHistory(entCategory, entSignature, modElemSpans, entElemN, true);
            }

            processPath(StrSpan(newVar), instructions);
        }
    }
    else if (equalSpans(entCategory, StrSpan("or", 2))) {
        // OR disintegration: flatten every contiguous nested OR now, in this
        // call, then treat the ordered atomic leaves as one cohort. The outer
        // signature remains the cohort/provenance identity; no intermediate
        // OR-valued branch is created for a later hashburst.
        StrSpan orLeaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        const int32_t orLeafN = flattenOrLeaves(
            instructions, hit, orLeaves,
            ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);

        // Check OR nesting depth — count existing "_(or" occurrences in validity name
        int currentOrDepth = 0;
        {
            int32_t pos = 0;
            while ((pos = findSpanFrom(validityName, StrSpan("_(or", 4), pos)) >= 0) {
                ++currentOrDepth;
                pos += 4;
            }
        }

        // The expansion record and the K mutual-exclusion implications are
        // emitted at EVERY OR depth: they are flat hash rules — the
        // disjunctive-syllogism consumption of the OR — and create no
        // scopes. Only the per-branch case-split below is gated on
        // max_or_depth (D-211).
        //
        // Compute the OR's expanded De-Morgan form once; used as the
        // disintegration origin for the K mutual-exclusion implications
        // below, mirroring the &/existence pattern in trackExpansionHistory.
        // See D-55.
        const ScratchString expandedOrSignature = expandSignature(
            entCategory, entSignature, orLeaves, orLeafN, sArena);
        // L3 span-record door. expandedOrSignature is a stable local; the
        // per-branch KEY is u_-stripped onto sArena in the loop. The
        // antecedent is loop-invariant, so the OriginDep is built once.
        const OriginDep orDisDeps[1] = {
            { StrSpan(expandedOrSignature), StrSpan(validityName) } };

        // The K mutual-exclusion implications cite expandedOrSignature as
        // their disintegration origin, which requires the matching
        // expansion-origin record (expandedOrSignature -> compact OR name)
        // to exist regardless of whether the per-branch case-split fires.
        trackExpansionHistory(entCategory, entSignature, orLeaves, orLeafN, false);

        // 1. Generate N implications: for each d_i, (!d_0 & ... & !d_{i-1} & !d_{i+1} & ... & !d_{N-1}) -> d_i
        //    These go to collected (parent validity) for normal hash memory insertion.
        //    Each is also stamped with `disintegration` origin pointing at the
        //    OR's expanded form so the proof graph can audit the K rules back
        //    to the originating OR (verifier check_disintegration's `or` branch
        //    rebuilds and verifies the exact shape).
        for (int32_t i = 0; i < orLeafN; ++i) {
            StrSpan premiseSpans[64];
            std::size_t premiseCount = 0;
            for (int32_t j = 0; j < orLeafN; ++j) {
                if (j != i) {
                    assert(premiseCount < 64 && "or-branch premise chain exceeds 64");
                    premiseSpans[premiseCount++] = prefixBang(orLeaves[j]);
                }
            }
            const ScratchString impStr = reconstructImplicationFullBindScratch(
                sArena, premiseSpans, static_cast<int>(premiseCount), orLeaves[i]);
            collected.insertImpl(StrSpan(currentStatement), StrSpan(impStr), StrSpan(validityName));

            if (parameters.trackHistory && trackHistoryLocal) {
                // KEY u_-stripped (matches the &/existence pattern in
                // trackExpansionHistory's section 2 — chapter rows
                // surface u_-stripped expressions; the u_-prefixed form
                // lives only in hash memory for unification).
                const ScratchString impClean = removeUPrefixScratch(sArena, StrSpan(impStr));
                int maxOrigins = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr;
                addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, StrSpan(impClean), StrSpan(validityName), OriginTag::disintegration, orDisDeps, 1, maxOrigins);
            }
        }

        // The per-branch case-split stays depth-gated: nested _ordis_
        // scopes are the scope explosion max_or_depth exists to prevent.
        if (currentOrDepth < parameters.max_or_depth) {
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
            //    (b) The legacy orAdmissionSet fallback is gone with its
            //        container (D-135): it had no
            //        insert site anywhere (D-31), so the always-empty set
            //        made the fallback loop constant-false — the flag IS
            //        the whole gate.
            const bool orAdmitted = allowOrDisintegration;


            if (orAdmitted) {
                // 3. Build compiled OR signature and register the disjunct
                //    count for this exact (parent validity, OR signature)
                //    cohort. Equal signatures under different parents are
                //    independent case splits.
                const ScratchString orSignature = removeUPrefixScratch(sArena, entSignature);  // e.g. "(or3[1,2,3])"
                {
                    // Set-or-insert (the operator[] = overwrite the cold
                    // set-once insert forbids): a re-registered cohort keeps
                    // its structural disjunct count.
                    const int32_t orParentId =
                        memoryBlock.lbStateInterner.encode(validityName);
                    const int32_t orSigId =
                        memoryBlock.lbStateInterner.encode(StrSpan(orSignature));
                    const int32_t orCohortId = mintOrCohortId(
                        memoryBlock.lbStateInterner, orParentId, orSigId);
                    const int orCnt = orLeafN;
                    const int32_t orSigRow =
                        memoryBlock.orDisjunctCount.lookup(orCohortId);
                    if (orSigRow != 0)
                        memoryBlock.orDisjunctCount.setValueAt(orSigRow, orCnt);
                    else
                        memoryBlock.orDisjunctCount.insert(orCohortId, orCnt);
                }

                // (trackExpansionHistory hoisted above the orAdmitted gate —
                //  see the K-implications block earlier in the OR case.)

                // 4. Each disjunct becomes a statement with branch validity name
                for (int32_t i = 0; i < orLeafN; ++i) {
                    // Per-disjunct string-tier scope: cleanExpr / orPayload /
                    // branchValidity are freed each iteration (mirroring the former
                    // heap std::strings). orSignature is built above the loop, so
                    // it survives every iteration's rewind.
                    ScratchScope oiScope(sArena);
                    const ScratchString cleanExpr = removeUPrefixScratch(sArena, orLeaves[i]);
                    // orPayload = "ordis_" + orSignature + "_(" + cleanExpr + ")"
                    // built explicit-length on the string tier.
                    const StrSpan orSigSpan(orSignature), cleanSpan(cleanExpr);
                    const int32_t opLen = 6 + orSigSpan.len + 2 + cleanSpan.len + 1;
                    char* opBuf = sArena.allocBytes(opLen);
                    int32_t opAt = 0;
                    std::memcpy(opBuf + opAt, "ordis_", 6); opAt += 6;
                    if (orSigSpan.len > 0) { std::memcpy(opBuf + opAt, orSigSpan.ptr, static_cast<size_t>(orSigSpan.len)); opAt += orSigSpan.len; }
                    std::memcpy(opBuf + opAt, "_(", 2); opAt += 2;
                    if (cleanSpan.len > 0) { std::memcpy(opBuf + opAt, cleanSpan.ptr, static_cast<size_t>(cleanSpan.len)); opAt += cleanSpan.len; }
                    opBuf[opAt++] = ')';
                    assert(opAt == opLen);
                    const StrSpan orPayload(opBuf, opLen);

                    NameId orParentId = memoryBlock.nameMap.encode(validityName);
                    NameId orBranchId = memoryBlock.nameMap.encodePush(orParentId, orPayload);
                    // copyFrom: the NEXT iteration's encode/encodePush mints NameMap,
                    // so a raw decodeView would dangle (I-3); the copy rides sArena.
                    const ScratchString branchValidity = ScratchString::copyFrom(sArena,
                        memoryBlock.nameMap.decodeView(orBranchId).ptr,
                        memoryBlock.nameMap.decodeView(orBranchId).len);

                    // L3 span-record door. orSignature / cleanExpr / branchValidity
                    // are stable locals -> span directly. The KEY spans also feed
                    // orBranchStatements.
                    const OriginDep orDisintDeps[1] = {
                        { StrSpan(orSignature), StrSpan(validityName) } };
                    if (trackHistoryLocal) {
                        addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, StrSpan(cleanExpr), StrSpan(branchValidity), OriginTag::orDisintegration, orDisintDeps, 1,
                            (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    }

                    orBranchStatements.insertImpl(StrSpan(), StrSpan(cleanExpr),
                                                  StrSpan(branchValidity));
                }
            }
            // else: OR not admitted — implications already emitted above, branches skipped
        }
        // else: max OR depth reached — K mutual-exclusion implications emitted above, no branch opening
    }
    else {
        // Default fallback
        // currentStatement was already inserted at top
    }
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

bool
ExpressionAnalyzer::disintegrateExpr2(StrSpan expr,
    Memory& memoryBlock,
    int iteration,
    bool forceDeep,
    StrSpan validityName,
    DisintProducts& out,
    bool trackHistoryLocal,
    bool allowOrDisintegration)
{
    int savedStartInt = memoryBlock.startInt;

    // collected / orBranchStatements are page-tier (allocPage) containers, so
    // they ride genScratchArenas -- NOT scratchArenas, whose allocBytes string
    // fill (prefixArgumentsWithU, below) would clobber a container page sharing
    // the slot when disintegration re-enters via the hypothetical / integration
    // paths (allocBytes targets pageHighWater()-1 and its ScratchScope rewind
    // poisons it). Same container-vs-string split as the request generators.
    const unsigned collSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    CollectedArena collected(&genScratchArenas().forSlot(collSlot));
    NewVarStore newVarMap(&genScratchArenas().forSlot(collSlot));
    CollectedArena orBranchStatements(&genScratchArenas().forSlot(collSlot));
    WorkInstruction instructions(&genScratchArenas().forSlot(collSlot));

    // String-tier scratch for the statified twins (prefixArgumentsWithUScratch /
    // removeUPrefixScratch / makeMarkedExprScratch). Deliberately the string
    // arena, NOT the genScratchArenas the page-tier containers above use. One
    // function-level scope holds the long-lived replExpr / topLevelExprClean
    // ScratchStrings; every per-block / per-element scope below nests ABOVE it
    // (byte-bump LIFO), so they survive the nested rewinds and are freed at
    // function return (I-124).
    ScratchArena& strArena = scratchArenas().forSlot(collSlot);
    ScratchScope fnScope(strArena);

    // replExpr = prefixArgumentsWithUScratch(expr) held as a ScratchString on the
    // string tier (0% heap); consumed by prepareIntegrationCore / core2 / the
    // topLevelExprClean strip below.
    const ScratchString replExpr = prefixArgumentsWithUScratch(strArena, StrSpan(expr));
    prepareIntegrationCore(StrSpan(replExpr), instructions, memoryBlock, expr);

    // The disintegration core now reads the arena WorkInstruction directly —
    // the former heap std::vector<LogicalEntity> bridge is gone.
    disintegrateExprCore2(StrSpan(replExpr),
        instructions,
        memoryBlock,
        iteration,
        collected,
        newVarMap,
        validityName,
        orBranchStatements,
        trackHistoryLocal,
        allowOrDisintegration);

    // finalStringStatements: membership + dedup ColdHashSet on the same per-slot
    // genScratchArenas page tier as collected (the admittedVars pattern). Iteration
    // is id / first-seen order EXCEPT the canBeSentIds mint in forceDeep, which sorts
    // a byte-bump snapshot to reproduce the former std::set<string> lex order (the
    // NameMap mint order feeds the dump, I-105). The cascade closure and the
    // set<EWV> conversions are order-independent (the set re-sorts).
    ScratchArena& fssArena = genScratchArenas().forSlot(collSlot);
    DirtyState fssDirty = DirtyState::Clean;
    ColdHashSet<BytesKeyStore> finalStringStatements(&fssArena, &fssDirty);

    // Recursive collector. A self-passing generic lambda rather than a
    // std::function so there is no type-erasure heap: the closure captures by
    // reference and recurses through its own `self` parameter.
    const auto addToFinal = [&](const StrSpan& s, const auto& self) -> void {
        const int32_t beforeN = finalStringStatements.count();
        finalStringStatements.mint(s);
        if (finalStringStatements.count() == beforeN) return;  // already present

        collected.forImpls(s, [&](const StrSpan& original, const StrSpan& validity) {
            out.implications.append(original, validity);
        });

        // Recurse for every child
        collected.forChildren(s, [&](const StrSpan& child) {
            self(child, self);
        });
    };

    if (forceDeep) {
        // 1. Filter and Collect Phase
        collected.forEachKey([&](const StrSpan& stmt) {
            bool allow = true;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN =
                getArgsSpans(StrSpan(stmt), args, ExecutionParameters::MAX_ARITY);

            for (int32_t ai = 0; ai < argN; ++ai) {
                // Rule: a statement carrying a new "it_" variable at level ==
                // mb.level is not allowed.
                int level = 0, id = 0;
                if (matchItLevId(args[ai], level, id)) {
                    if (id >= savedStartInt && level == memoryBlock.level) {
                        allow = false;
                        break;
                    }
                }
            }

            if (allow) {
                addToFinal(stmt, addToFinal);
            }
        });

        // Memo only the `int_lev_*` carriers — the only entries
        // `allowedForMail` can ever probe.
        // canBeSentIds must see the int_lev_ carriers in the former std::set<string>
        // lex order (the NameMap mint order feeds the dump, I-105). Collect the
        // carrier ids, sort by decoded bytes on the gen arena's byte-bump tier (no
        // heap; independent of the page tier holding finalStringStatements), mint.
        if (finalStringStatements.count() > 0) {
            ScratchScope sortScope(fssArena);
            const int32_t fssN = finalStringStatements.count();
            int32_t* ids = reinterpret_cast<int32_t*>(
                fssArena.resolve(fssArena.alloc(fssN * 4, 4)));
            int32_t carrierN = 0;
            for (int32_t id = 1; id <= fssN; ++id) {
                if (containsSpan(finalStringStatements.keyAt(id), StrSpan("int_lev_", 8)))
                    ids[carrierN++] = id;
            }
            std::sort(ids, ids + carrierN, [&](int32_t a, int32_t b) {
                return compareSpans(finalStringStatements.keyAt(a),
                                    finalStringStatements.keyAt(b)) < 0;
            });
            for (int32_t i = 0; i < carrierN; ++i)
                memoryBlock.canBeSentIds.mint(
                    memoryBlock.nameMap.encode(finalStringStatements.keyAt(ids[i])));
        }
        // canBeSentIds from the int_lev_ implications, in the former
        // std::set<EWV> order == out.implications.forEachSorted (EWV::operator<).
        // out.implications is already filled by addToFinal above.
        out.implications.forEachSorted([&](const StrSpan& original, const StrSpan& /*v*/) {
            if (containsSpan(original, StrSpan("int_lev_", 8))) {
                memoryBlock.canBeSentIds.mint(memoryBlock.nameMap.encode(original));
            }
        });
        // forceDeep statements: append finalStringStatements (all at parent
        // validity) + the OR branches straight to the channel; forEachSorted at
        // the caller reproduces the former std::set<EWV> order byte-for-byte.
        for (int32_t id = 1; id <= finalStringStatements.count(); ++id)
            out.statements.append(finalStringStatements.keyAt(id), validityName);
        orBranchStatements.forImpls(StrSpan(), [&](const StrSpan& o, const StrSpan& v) {
            out.statements.append(o, v);
        });
        // forceDeep has no admission pass: an expression with witnesses (any
        // existence) is not fully disintegrated here; an atomic (no witnesses) is.
        return newVarMap.empty();
    }

    // Rejection staging: the algebra (it_) and integration (int_) buffers are
    // RejectionStores on the per-slot genScratchArenas page tier. Records append in
    // Pass-B order, which IS the former std::map lex-var-then-insertion drain order,
    // so each drain walks the list in order. The revival path (revisitRejected2) reads
    // the buffered constituent + siblings without re-running prepareIntegrationCore;
    // compactExpr + levels are captured at buffer time (algebra side only -- the
    // integration side passes an empty compactExpr and no levels).
    RejectionStore pendingRejections(&genScratchArenas().forSlot(collSlot));
    RejectionStore pendingRejectionsIntegration(&genScratchArenas().forSlot(collSlot));

    // Track successfully admitted variables in this call (membership only --
    // mint/contains, never iterated). Transient ColdHashSet on the per-slot
    // genScratchArenas (page-tier container) arena -- NOT the scratchArenas
    // string arena the door helpers fill with allocBytes, whose scratch-fill +
    // ScratchScope rewind would poison a page-tier container sharing the slot.
    // Freed with the function frame -- no malloc heap.
    const unsigned admittedSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& admittedArena = genScratchArenas().forSlot(admittedSlot);
    DirtyState admittedDirty = DirtyState::Clean;
    ColdHashSet<BytesKeyStore> admittedVars(&admittedArena, &admittedDirty);

    // topLevelExprClean = removeUPrefix(replExpr) held as a ScratchString on the
    // string tier under fnScope (used in Pass B + the integration rejection
    // commit; the intervening per-block scopes nest above it).
    const ScratchString topLevelExprClean = removeUPrefixScratch(strArena, StrSpan(replExpr));

    // Pass A: Unconditional Statements
    collected.forEachKey([&](const StrSpan& stmt) {
        if (finalStringStatements.contains(stmt)) return;

        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argN =
            getArgsSpans(StrSpan(stmt), args, ExecutionParameters::MAX_ARITY);
        bool hasNewVars = false;
        for (int32_t ai = 0; ai < argN; ++ai) {
            // A new it_/int_ variable at this LB's level (id at or past the
            // saved start) keeps the statement out of the unconditional set.
            int varLevel = 0, id = 0;
            if (matchItLevId(args[ai], varLevel, id)
                || matchIntLevId(args[ai], varLevel, id)) {
                if (varLevel == memoryBlock.level && id >= savedStartInt) {
                    hasNewVars = true;
                    break;
                }
            }
        }

        if (!hasNewVars) {
			addToFinal(stmt, addToFinal);
        }
    });

    if (!parameters.compressor_mode && !parameters.ban_disintegration) {
        // Pass B: New Variable Admission
        // Gated by !parameters.ban_disintegration (collapsed from short-lived
        // allow_disintegration flag on 2026-04-29 — see D-28; original gate
        // was !parameters.incubator_mode pre-2026-04-29). See I-7.
        newVarMap.forEachVarSorted([&](int32_t varId) {
            // var is a zero-copy span into newVarMap (read-only in Pass B, so no
            // mint into its interner intervenes -- decodeView-safe, I-3).
            const StrSpan var = newVarMap.varAt(varId);
            bool isVarAdmitted = false;
            int classLevel = 0, classId = 0;  // discarded out-params for the it_/int_ name classifier
            const int32_t elemN = newVarMap.elemCount(varId);

            for (int32_t j = 0; j < elemN; ++j) {
                // ru / marked held as ScratchStrings for the whole j-iteration on
                // the string tier; jsc frees them at each iteration (former heap
                // removedU / markedExpr). isAdmitted / isAdmittedIntegration /
                // the rejected doors take spans now.
                ScratchScope jsc(strArena);
                const ScratchString ru =
                    removeUPrefixScratch(strArena, newVarMap.elemAt(varId, j));
                const ScratchString marked =
                    makeMarkedExprScratch(strArena, StrSpan(ru), var);

                // 1. Check for Iteration Variable (it_...)
                if (matchItLevId(var, classLevel, classId)) {
                    const StrSpan core = extractExpressionSpan(StrSpan(ru));
                    bool hasOperator = (this->operators.find(
                        std::string_view(core.ptr, core.len)) != this->operators.end());

                    if (hasOperator) {
                        if (isAdmitted(memoryBlock, StrSpan(ru), var, StrSpan(marked), validityName)) {
                            isVarAdmitted = true;
                            break;
                        }
                        else if (isAllowedAsOperatorInput(memoryBlock, StrSpan(ru), var)) {
                            isVarAdmitted = true;
                            break;
                        }
                        else {
                            // Buffer rejection instead of calling updateRejectedMap immediately.
                            // Capture the cohort (other expressions sharing this it_ var)
                            // and the parent compound's levels so the revival path can
                            // mail-emit the full body without re-running
                            // prepareIntegrationCore. Symmetric to the int_ branch below.
                            ScratchArena& rArena = scratchArenas().forSlot(collSlot);
                            ScratchScope rsc(rArena);
                            StrSpan sibSpans[256];
                            int32_t sibN = 0;
                            for (int32_t k = 0; k < elemN; ++k) {
                                if (k == j) continue;
                                assert(sibN < 256 && "rejection siblings exceed 256");
                                sibSpans[sibN++] = StrSpan(removeUPrefixScratch(
                                    rArena, newVarMap.elemAt(varId, k)));
                            }
                            int32_t levArr[256];
                            int32_t levN = 0;
                            {
                                const int32_t compoundLvlsId2 = lookupStatementLevels(
                                    memoryBlock.intStatementLevelsMap, memoryBlock.nameMap,
                                    StrSpan(topLevelExprClean), validityName);
                                if (compoundLvlsId2) {
                                    levN = coldIntRunAt(
                                        memoryBlock.intStatementLevelsMap,
                                        compoundLvlsId2, levArr, 256);
                                }
                            }
                            pendingRejections.addRejection(var, StrSpan(ru),
                                StrSpan(marked), StrSpan(topLevelExprClean),
                                sibSpans, sibN, levArr, levN);
                        }
                    }
                }
                // 2. Check for Integration Variable (int_...)
                else if (matchIntLevId(var, classLevel, classId)) {
                    if (isAdmittedIntegration(memoryBlock, StrSpan(ru), var, StrSpan(marked), validityName)) {
                        isVarAdmitted = true;
                        break;
                    }
                    else if ([&]() {
                        // Non-minting probe — a never-interned template was
                        // never registered in the admission set.
                        int64_t setPk = 0;
                        return lookupTemplateKey(memoryBlock.templateInterner,
                                   memoryBlock.nameMap, StrSpan(marked), validityName, setPk)
                            && memoryBlock.overallHashMemory.admissionSetIntegration.contains(setPk);
                    }()) {
                        cleanAdmissionMap(StrSpan(marked), validityName, memoryBlock);
                        isVarAdmitted = true;
                        break;
                    }
                    else {
                        // Buffer rejection symmetric to the it_ branch above.
                        // Skip (in[...]) typing elements — per user spec, they
                        // never become a rejectedMapIntegration key (siblings
                        // still carry them for re-emission).
                        const StrSpan core = extractExpressionSpan(StrSpan(ru));
                        const bool isInTyping = equalSpans(core, StrSpan("in", 2));
                        if (!isInTyping) {
                            ScratchArena& rArena = scratchArenas().forSlot(collSlot);
                            ScratchScope rsc(rArena);
                            StrSpan sibSpans[256];
                            int32_t sibN = 0;
                            for (int32_t k = 0; k < elemN; ++k) {
                                if (k == j) continue;
                                assert(sibN < 256 && "rejection siblings exceed 256");
                                sibSpans[sibN++] = StrSpan(removeUPrefixScratch(
                                    rArena, newVarMap.elemAt(varId, k)));
                            }
                            pendingRejectionsIntegration.addRejection(var,
                                StrSpan(ru), StrSpan(marked), StrSpan(),
                                sibSpans, sibN, nullptr, 0);
                        }
                    }
                }
            }

            if (isVarAdmitted) {
                admittedVars.mint(var); // Mark as locally admitted

                for (int32_t j = 0; j < elemN; ++j) {
                    ScratchScope sc(strArena);
                    const ScratchString cleanStmt =
                        removeUPrefixScratch(strArena, newVarMap.elemAt(varId, j));
                    addToFinal(StrSpan(cleanStmt), addToFinal);
                }
            }
        });
    }

    // ---------------------------------------------------------
    // Cascade Admission for Dependent Variables (int_ and it_)
    // ---------------------------------------------------------

    // Stack-buffer prefixes (no per-call heap): byte-identical to the former
    // "int_lev_<level>_" / "_lev_<level>_" std::strings (level is int, so %d
    // matches std::to_string); used directly in the std::string rfind/find
    // probes below.
    char intPrefix[40];
    char levSubstring[40];
    const int intPrefixLen = std::snprintf(intPrefix, sizeof(intPrefix), "int_lev_%d_", memoryBlock.level);
    const int levSubLen = std::snprintf(levSubstring, sizeof(levSubstring), "_lev_%d_", memoryBlock.level);

    while (true) {
        bool hasGrown = false;

        // Iterate through all potential statements in 'collected'. id / first-seen
        // order is fine here: the cascade is a closure (the while-loop converges to
        // the same admitted set regardless of visitation order), and addToFinal's
        // inserts beyond passCount are picked up on the next pass. newVarMap is still
        // a std::map this commit, so its lookups materialize the arg span (commit 4
        // removes that round-trip).
        const int32_t passCount = finalStringStatements.count();
        for (int32_t fid = 1; fid <= passCount; ++fid)
        {
            const StrSpan stmt = finalStringStatements.keyAt(fid);
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(stmt, args, ExecutionParameters::MAX_ARITY);

            bool hasAdmittedVar = false;
            StrSpan candidates[ExecutionParameters::MAX_ARITY];
            int32_t candN = 0;

            // First pass: Classify arguments in this statement
            for (int32_t ai = 0; ai < argN; ++ai) {
                const StrSpan arg = args[ai];
                // Check if already admitted
                if (admittedVars.contains(arg)) {
                    hasAdmittedVar = true;
                }
                // Check if it is a potential candidate (Target Syntax + New + Not Admitted)
                else {
                    bool isTarget = false;
                    // Check A: int_lev_X_
                    if (arg.len >= intPrefixLen
                        && std::memcmp(arg.ptr, intPrefix, static_cast<size_t>(intPrefixLen)) == 0) {
                        isTarget = true;
                    }
                    // Check B: it_..._lev_X_
                    else if (arg.len > 3 && arg.ptr[0] == 'i' && arg.ptr[1] == 't' && arg.ptr[2] == '_') {
                        if (containsSpan(arg, StrSpan(levSubstring, levSubLen))) {
                            isTarget = true;
                        }
                    }

                    if (isTarget) {
                        // Must be in newVarMap (new) and not yet admitted
                        if (newVarMap.hasVar(arg)) {
                            candidates[candN++] = arg;
                        }
                    }
                }
            }

            // Admission Rule: New vars are admitted ONLY if they share an expression with an admitted var
            if (hasAdmittedVar && candN > 0) {
                for (int32_t ci = 0; ci < candN; ++ci) {
                    const StrSpan cand = candidates[ci];
                    // Double-check to ensure we don't process the same candidate twice in one pass
                    if (!admittedVars.contains(cand)) {

                        admittedVars.mint(cand);
                        hasGrown = true;

                        // Pull in all defining expressions for this newly admitted
                        // variable (verified present via hasVar above).
                        const int32_t cvId = newVarMap.lookupVar(cand);
                        const int32_t cvN = newVarMap.elemCount(cvId);
                        for (int32_t j = 0; j < cvN; ++j) {
                            ScratchScope sc(strArena);
                            const ScratchString cleanStmt =
                                removeUPrefixScratch(strArena, newVarMap.elemAt(cvId, j));
                            addToFinal(StrSpan(cleanStmt), addToFinal);
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
    // Records are in lex-var-then-insertion order (Pass-B order); walk in order and
    // skip records whose var was admitted (in Pass B or the cascade).
    for (int32_t r = 0; r < pendingRejections.count(); ++r) {
        const RejRec& fail = pendingRejections.recAt(r);
        if (admittedVars.contains(pendingRejections.decode(fail.varId))) continue;
        // Variable was truly rejected (dependencies never resolved). decode
        // returns spans into the RejectionStore's own interner (page tier, not
        // mutated during the drain); updateRejectedMap mints valueInterner /
        // templateInterner / nameMap (different), so the spans stay valid (I-3).
        const StrSpan removedUF = pendingRejections.decode(fail.s1Id);
        const StrSpan markedExprF = pendingRejections.decode(fail.s2Id);
        const StrSpan compactExprF = pendingRejections.decode(fail.s3Id);
        StrSpan siblingsF[256];
        assert(fail.sibCount <= 256 && "pendingRejections siblings exceed cap");
        for (int32_t k = 0; k < fail.sibCount; ++k)
            siblingsF[k] = pendingRejections.sibAt(fail, k);
        int lvRunF[256];
        assert(fail.levCount <= 256
            && "pendingRejections level run exceeds lvRunF");
        for (int32_t k = 0; k < fail.levCount; ++k)
            lvRunF[k] = pendingRejections.levAt(fail, k);
        updateRejectedMap(removedUF, markedExprF, compactExprF,
                          removedUF,         // concreteConstituent (= removedU)
                          siblingsF, fail.sibCount, lvRunF, fail.levCount,
                          memoryBlock.overallHashMemory, memoryBlock, iteration, validityName);
        // Park-demand rendezvous (D-243): the
        // admission-marker side fires revisitRejected2 exactly once, at the
        // moment the marker key is written. A cohort parked AFTER that
        // one-shot probe would sleep forever — witness-generation pacing
        // defers firing-fed parkings by a mail hop, so the demand can
        // arrive first. Probe the admission map for this parked key's
        // demand; if it is already live, fire the revival now — whichever
        // side arrives second wakes the other. A consumed demand needs no
        // wake: its constituent was admitted through another route.
        int64_t parkedPk = 0;
        if (lookupTemplateKey(memoryBlock.templateInterner, memoryBlock.nameMap,
                              markedExprF, validityName, parkedPk)
            && memoryBlock.overallHashMemory.admissionMap.lookup(parkedPk) != 0) {
            revisitRejected2(markedExprF, memoryBlock, validityName);
        }
    }

    // Integration-side commit — mirror of algebra block above. A pending
    // integration-rejection is committed to rejectedMapIntegration only if
    // the int_ var stayed un-admitted after cascade admission.
    for (int32_t r = 0; r < pendingRejectionsIntegration.count(); ++r) {
        const RejRec& fail = pendingRejectionsIntegration.recAt(r);
        if (admittedVars.contains(pendingRejectionsIntegration.decode(fail.varId))) continue;
        const StrSpan markedExprF = pendingRejectionsIntegration.decode(fail.s2Id);
        const StrSpan concreteF = pendingRejectionsIntegration.decode(fail.s1Id);
        StrSpan siblingsF[256];
        assert(fail.sibCount <= 256 && "pendingRejectionsIntegration siblings exceed cap");
        for (int32_t k = 0; k < fail.sibCount; ++k)
            siblingsF[k] = pendingRejectionsIntegration.sibAt(fail, k);
        updateRejectedMapIntegration(markedExprF, concreteF, siblingsF, fail.sibCount,
                                     StrSpan(topLevelExprClean),
                                     memoryBlock.overallHashMemory, memoryBlock, validityName);
    }

    // Append the normal statements (all at parent validity) + the OR branches
    // (with their own branch validity names) straight to the channel; forEachSorted
    // at the caller reproduces the former std::set<EWV> order byte-for-byte.
    // out.implications is already filled by addToFinal.
    for (int32_t id = 1; id <= finalStringStatements.count(); ++id)
        out.statements.append(finalStringStatements.keyAt(id), validityName);
    orBranchStatements.forImpls(StrSpan(), [&](const StrSpan& o, const StrSpan& v) {
        out.statements.append(o, v);
    });

    // fullDisintegrationHappened: true iff every existence inside the compound
    // got at least one ADMITTED witness. Group the witness vars in newVarMap by
    // their marker-body signature: makeMarkedExpr replaces the witness name with
    // "marker", so the it_ and int_ spawned from one existence share a signature
    // and fall in the same group; a group is covered when any of its vars is in
    // admittedVars. No existence (newVarMap empty) => no groups => vacuously true
    // (e.g. a plain in3 has nothing to witness). Erring to false is always safe
    // (cFE just disintegrates); only a wrong true re-introduces the mirror bug.
    bool fullDisintegrationHappened = true;
    {
        // existenceGroups -> two ColdHashSets on the per-slot genScratchArenas:
        // allSigs (every existence's marker signature) and coveredSigs (those with an
        // admitted witness). makeMarkedExpr maps a witness name to "marker", so all
        // witnesses of one existence share one signature; full disintegration means
        // every signature is covered. coveredSigs is a subset of allSigs by
        // construction, so that is exactly coveredSigs.count() == allSigs.count().
        // Counts only, so the witness iteration order is unobservable.
        ScratchArena& egArena = genScratchArenas().forSlot(collSlot);
        DirtyState egDirty = DirtyState::Clean;
        ColdHashSet<BytesKeyStore> allSigs(&egArena, &egDirty);
        ColdHashSet<BytesKeyStore> coveredSigs(&egArena, &egDirty);
        ScratchArena& egStr = scratchArenas().forSlot(collSlot);

        for (int32_t wid = 1; wid <= newVarMap.varCount(); ++wid) {
            const StrSpan witnessVar = newVarMap.varAt(wid);
            const int32_t wN = newVarMap.elemCount(wid);
            ScratchScope sc(egStr);
            // Build this witness's marker set, sort + dedup, join into a canonical
            // signature key (separator 0x01, never present in MPL text).
            StrSpan markers[256];
            int32_t markerN = 0;
            for (int32_t j = 0; j < wN; ++j) {
                assert(markerN < 256 && "existence body exceeds 256 markers");
                const ScratchString clean = removeUPrefixScratch(egStr, newVarMap.elemAt(wid, j));
                const ScratchString marked = makeMarkedExprScratch(egStr, StrSpan(clean), witnessVar);
                markers[markerN++] = StrSpan(marked);
            }
            std::sort(markers, markers + markerN,
                [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) < 0; });
            markerN = static_cast<int32_t>(std::unique(markers, markers + markerN,
                [](const StrSpan& a, const StrSpan& b) { return compareSpans(a, b) == 0; }) - markers);
            int32_t sigLen = 0;
            for (int32_t i = 0; i < markerN; ++i) { if (i) ++sigLen; sigLen += markers[i].len; }
            char* sigBuf = egStr.allocBytes(sigLen);
            int32_t at = 0;
            for (int32_t i = 0; i < markerN; ++i) {
                if (i) sigBuf[at++] = '\x01';
                std::memcpy(sigBuf + at, markers[i].ptr, static_cast<size_t>(markers[i].len));
                at += markers[i].len;
            }
            const StrSpan sigKey(sigBuf, sigLen);
            allSigs.mint(sigKey);
            if (admittedVars.contains(witnessVar)) coveredSigs.mint(sigKey);
        }
        fullDisintegrationHappened = (allSigs.count() == coveredSigs.count());
    }

    // out.implications / out.statements are filled in place above (addToFinal +
    // the direct statement/OR-branch appends); nothing to copy here.
    return fullDisintegrationHappened;
}









// Turn optimizations OFF for just this section
//#pragma optimize("", off)

void ExpressionAnalyzer::prove(int numberIterations,
    const std::vector<Memory*>& bodies) {

    // Steward scope (D-159): the background memory
    // custodian lives exactly as long as this iteration loop. Started
    // parked; every kernel barrier quiesces it before reading state;
    // the guard quiesces + stops + destroys it before prove() returns,
    // so destroyGrid, the CE filter, and every post-prove reader see a
    // steward-free world.
    assert(!steward && "nested prove() steward scopes");
    // Plans never cross prove scopes: a plan computed at a previous
    // scope's final barrier was never armed and is dropped here (the
    // kernel re-plans from live state at its next pressured barrier).
    stewardEvictionPlan.clear();
    steward = std::make_unique<MemorySteward>();
    // The worker count sizes the I/O executor pool (steward::ioThreadCountFor
    // — 4 executors at 32 workers).
    steward->start(logicalCores);
    // FORENSIC EXHAUSTION TRAP: at a main-pool wall, grantLocked prints the
    // pool counters and this census (the grid + steward view) to stderr
    // BEFORE the exhaustion assert aborts — resolving policy-missed-victims
    // versus pinned-working-set at the point of death. The census runs under
    // the pool mutex; printExhaustionCensus is manager-mutex-free by
    // construction (its contract). Cleared first in the scope guard so no
    // post-prove grant ever runs a reporter over a stale grid reference.
    staticMemory().setExhaustionReporter(
        [&bodies, s = steward.get()](int64_t inUse, int64_t total) {
            printExhaustionCensus(bodies, s, inUse, total);
        });
    struct StewardScope {
        std::unique_ptr<MemorySteward>& s;
        ~StewardScope() {
            // TEARDOWN LOAD-SHEDDING: after the last iteration, the final
            // barrier's head-prefetch loads (and any straggler window tasks)
            // are pure waste — there is no next iteration to warm, no window
            // is open (the planner cannot evict to make room), and the
            // executor load path has no valve. Discard everything queued but
            // unstarted (defined load-shedding, the ring-drop doctrine);
            // in-flight tasks finish and the quiesce below waits for them.
            // The unbudgeted, undiscarded form of these loads was the 4 GiB
            // teardown wall (D-196).
            s->discardQueuedIoTasks();
            // The final barrier may have re-armed the grant trigger; the
            // chapter export's post-prove reloads acquire blocks and would
            // fire a trigger pointing at this stopped steward (the
            // wake-without-work assert). No armed trigger may outlive its
            // steward — post-prove block traffic is not a steward signal
            // (D-158).
            staticMemory().disarmGrantTrigger();
            s->quiesce();
            s->stop();
            // The exhaustion reporter clears LAST — after every teardown
            // action that can still grant (the in-flight executor loads the
            // quiesce just waited out), so a teardown exhaustion prints a
            // full census. Lifetime: the reporter captures prove()'s
            // `bodies` reference (alive for the whole prove scope, including
            // this guard) and the steward pointer (alive until the reset
            // below) — both outlive every possible invocation.
            staticMemory().setExhaustionReporter(nullptr);
            s.reset();
        }
    } stewardScope{ steward };

    for (int it = 0; it < numberIterations; ++it) {
        // Debug: count total expressions across all active bodies
        std::size_t totalExprs = 0;
        std::size_t activeBodies = 0;
        for (std::size_t bi = 0; bi < bodies.size(); ++bi) {
            if (bodies[bi] && bodies[bi]->isActive) {
                // Deload-tolerant count: most LBs are cold between
                // iterations and must not reload just to be counted.
                totalExprs += static_cast<std::size_t>(
                    bodies[bi]->intEncodedStatementsCount());
                ++activeBodies;
            }
        }
        std::cout << "Hash burst: " << it
                  << "  active_bodies=" << activeBodies
                  << "  total_exprs=" << totalExprs << std::endl;
        std::cout.flush();

        // Publish the burst index so every RTTracker constructed by the
        // upcoming proveKernel captures it in its `.rt/<chain>.log`
        // header. Written here on the main thread between bursts; read
        // by worker threads inside the phase helpers. No
        // concurrent write while workers run, so plain int is safe.
        ::gl::rt_tracker::g_currentHashburstIndex = it;

        auto burstStart = std::chrono::high_resolution_clock::now();
        proveKernel(bodies);
        auto burstEnd = std::chrono::high_resolution_clock::now();
        double burstSec = std::chrono::duration<double>(burstEnd - burstStart).count();
        // active_bodies (above) is the PRE-skip count; swept is the post-skip
        // sweep size and skipped the quiescent LBs excluded this iteration
        // (D-194). Telemetry only (Rule 16 / I-44).
        std::cout << "  dt=" << burstSec << "s"
                  << "  swept=" << lastSweptCount
                  << "  skipped=" << lastSkippedCount << std::endl;
        std::cout.flush();

        // Global deactivation sweep once per iteration, post-drain. The
        // sweep used to run only from deactivateUnnecessary's tail — i.e.
        // only when a theorem was emitted — so LBs whose last main-scope
        // goal closed after the batch's final emission stayed active to
        // batch end. Same I-48 predicate as before, evaluated every
        // iteration on this single-threaded seam; reads only never-deloaded
        // state (intToBeProved persistent I-108, isActive + tree edges on
        // the LB slab I-109/I-110).
        deactivateRecursively();
    }
}

// readSimpleFacts() and saveFilteredConjectures() — moved to filter.cpp.

void ExpressionAnalyzer::saveProvedTheorems() {
    namespace fs = std::filesystem;

    // Resolve path: <repo>/files/theorems/theorems.txt
    const auto theoremsDir =
        fs::path(__FILE__).parent_path().parent_path().parent_path().parent_path()
        / "files" / "theorems";

    const auto filePath = theoremsDir / "theorems.txt";

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

    const auto filePath = theoremsDir / "theorems.txt";

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

    // Second file: compiled forms (for proof graph pruning — keeps or0, existence2, etc.)
    const auto compiledPath = theoremsDir / "compiled_theorems.txt";
    std::ofstream ofsCompiled(compiledPath, std::ios::trunc);

    int written = 0;
    for (const std::string& theorem : essentialTheorems) {
        // Skip surviving external theorems — they are imported, not GL-proved here
        if (externalTheorems.count(theorem)) continue;

        auto it = compactToExpanded.find(theorem);
        std::string toWrite = (it != compactToExpanded.end()) ? it->second : theorem;

        // compiled_theorems.txt: write as-is (keeps compiled heads)
        if (ofsCompiled.is_open()) ofsCompiled << toWrite << "\n";

        // theorems.txt: expand runtime expressions (or0, etc.) to base form
        // so subsequent batches can parse them.
        toWrite = expandToBaseForm(toWrite);
        ofs << toWrite << "\n";
        ++written;
    }

    std::cout << "Rewrote theorems.txt with " << written
        << " globally essential theorems (excluded "
        << externalTheorems.size()
        << " external)." << std::endl;
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
    // Negate the old head; an already-negated head loses its `!` instead
    // (double-negation cancellation — GL has no universal `!!` normalizer,
    // and a raw `!!(...)` premise breaks every downstream compiled-core
    // consumer).
    std::string newPrem = (!head.empty() && head[0] == '!')
        ? head.substr(1)
        : "!" + head;

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

        // 2. Rebuild the theorem via reconstructImplicationFullBind so its
        //    bound-variable lists are ALWAYS in first-occurrence order (the
        //    single binder rule, I-4). If the head is an uncompiled existence
        //    ("!(>"), compile it to existence<N> first; otherwise use the head
        //    as-is. Previously the reconstruction was nested inside the
        //    existence check, so a non-existence-head external was broadcast as
        //    the raw thOriginal with its binders untouched -- a non-occurrence-
        //    order binder then leaked into the `compilation` row's rest[0] and
        //    the verifier's check_compilation rejected it.
        std::string compiledHead = head;
        if (startsWith(head, "!(>", 3)) {
            compiledHead = this->compileCoreExpressionMapCore(
                head,
                this->implCounter,
                this->existenceCounter,
                this->andCounter,
                this->orCounter,
                this->variableCounter
            );
        }

        finalTheorem = this->reconstructImplication(finalChain, compiledHead);

        // Re-disintegrate so finalChain / finalHead reflect the rebuilt theorem.
        tempChain.clear();
        finalChain.clear();
        finalHead = ce::disintegrateImplication(finalTheorem, tempChain, this->coreExpressionMap);
        for (const auto& t : tempChain) finalChain.push_back(std::get<0>(t));

        // 3. Queue the (potentially modified) theorem for broadcast.
        // ASIC 0.1 reshuffle: the legacy Mail::implications tuple is
        // removed; the external travels SOLELY as the inline compact
        // statement deposited just below (broadcastTheorems is
        // single-threaded at LOAD, so inline compile is race-free and
        // makes the external available from burst 1 — the deferred queue
        // would drain too late).

        if (parameters.trackHistory) {
            ExpressionWithValidity ev(finalTheorem, "main");
            auto originBroadcast = std::make_pair(originTag, std::vector<ExpressionWithValidity>());
            addOrigin(broadcastMail.exprOriginMap, ev, originBroadcast, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        // ASIC 0.1 reshuffle: externals must ALSO travel as the D-76
        // compact (implication<N>[...]) statement, because the legacy
        // Mail::implications channel is being retired and the receiver
        // now recovers rules via status=3 disintegration of the compact
        // form. broadcastTheorems runs single-threaded at LOAD time
        // (pre-proveKernel), so the post-pool.join() deferred-compaction
        // drain (recordPendingCompaction) would deliver these a full
        // prove-entry too late — the externals must be installed by
        // burst 1. Compile INLINE here instead: single-threaded at this
        // site, so there is no I-28 race on the global implCounter /
        // compiledExpressions / repetitionExclusionMap state that forced
        // the parallel broadcast sites to defer. Shape mirrors the
        // deferred drain (compact statement at "main" + EMPTY compactLevels
        // + paired `compilation` origin).
        {
            const std::string compactImpl = compileImplicationToCompact(finalTheorem);
            // compactLevels MUST stay EMPTY here, as at the
            // deferred-compaction drain in proveKernel. The implication
            // rule always deposits std::set<int>(); via addToHashMemory the
            // deposited level run becomes the installed rule's levels, so a
            // non-empty set injects an extra level into every statement derived
            // from that rule, making the derived level-set size exceed
            // memoryBlock.level + 1. The allLevelsInvolved discharge gate in
            // prover.hpp::dischargeToBeProved then refuses to promote the derived head to
            // globalTheoremList: the fact enters intEncodedStatements but is never
            // recorded as proved or broadcast.
            std::set<int> compactLevels;
            ExpressionWithValidity compactEv(compactImpl, "main");
            broadcastMail.statements.insert(std::make_pair(compactEv, compactLevels));
            if (parameters.trackHistory) {
                addOrigin(broadcastMail.exprOriginMap, compactEv,
                    std::make_pair("compilation", std::vector<ExpressionWithValidity>{ ExpressionWithValidity(finalTheorem, "main") }),
                    (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }
        }
    }

    // New mail system (D-137): store the broadcast ONCE in the
    // root's log -- every descendant pulls it on the normal ancestor walk -- and
    // self-inject the root, which has no ancestor to pull from. Replaces the old
    // per-LB boxes broadcast. Single-threaded at LOAD, before any pull, so the
    // batch is in place for burst 1. broadcastMail carries only statements +
    // exprOriginMap, exactly what mergeBatchInto and commit keep.
    mergeBatchIntoMailIn(broadcastMail, this->body.mailIn);
    this->mailLog.commit(&this->body, std::move(broadcastMail));
    // WAKE DOOR 5 (D-194): the seed batch self-injects the
    // root's mailIn directly, so wake the root here; every descendant is woken
    // by mailPeek on the next active-build (it sees the root's bumped commit
    // count). A no-op on burst 1 (root born dirty), load-bearing for the
    // between-warm-up-and-main broadcast where the root may have gone quiescent.
    this->body.hasWork = true;

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
        if (!current->exprKey().empty()) {
            exprKeys.push_back(current->exprKey());
        }
        current = current->parentMemory;
    }

    // Process all collected expressions for the current block *mb
    for (const std::string& expr : exprKeys) {
        const StrSpan coreSpan = extractExpressionSpan(StrSpan(expr));
        if (this->operators.find(std::string_view(coreSpan.ptr,
                static_cast<std::size_t>(coreSpan.len))) != this->operators.end()) {
            const ce::CoreExpressionConfig* cfg = coreConfig(coreSpan);
            assert(cfg && "prefillIntegrationMapsRecursive: operator core must be in coreExpressionMap");

            const LogicalEntity* le = compiledEntity(coreSpan);
            assert(le && "prefillIntegrationMapsRecursive: operator core must be compiled");

            if (!equalSpans(StrSpan(le->category), StrSpan("atomic", 6))) {
                assert(!cfg->outputIndices.empty());

                std::vector<std::string> args = ce::getArgs(expr);
                // Assuming the operator has one output variable as implied by context
                int outIdx = cfg->outputIndices[0];

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
                StrSpan unchRun[ExecutionParameters::MAX_ARITY];
                int32_t unchN = 0;
                for (const std::string& a : argSet) {
                    assert(unchN < ExecutionParameters::MAX_ARITY
                        && "prepareIntegration: unchangeable args exceed MAX_ARITY");
                    unchRun[unchN++] = StrSpan(a);
                }
                prepareIntegration(StrSpan(rplExpr2), unchRun, unchN, *mb, StrSpan("main", 4), StrSpan());
            }
        }
    }

    // Recurse into children
    simpleMapStore.forEachChild(mb, [&](const gl::StrSpan&, Memory* child) {
        prefillIntegrationMapsRecursive(child);
    });
}

//#pragma optimize("", off)

void ExpressionAnalyzer::disintegrateExprHypothetically(StrSpan expr, Memory& memoryBlock, StrSpan validityName) {
    if (parameters.ban_disintegration) return;

    // Function-level string-tier scope holding the sentinel / hypo payloads and
    // the copyFrom'd sentinelValidity / newValidityName (0% heap); the per-tv
    // (=[Y,Y_copy]) build nests above it.
    const unsigned hypoStrSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& hypoStrArena = scratchArenas().forSlot(hypoStrSlot);
    ScratchScope hypoStrScope(hypoStrArena);

    // 1. Extract input variables (D16: spans over the stable `expr`).
    StrSpan inputVars[ExecutionParameters::MAX_ARITY];
    const int32_t inputVarN = getArgsSpans(expr, inputVars,
                                           ExecutionParameters::MAX_ARITY);

    for (int32_t i = 0; i < inputVarN; ++i) {
        if (equalSpans(inputVars[i], StrSpan("marker", 6))) {
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
    // `validityName` (for intStatementLevelsMap / equivalenceClassesMap lookups
    // and for constructing `newValidityName`). The sentinel stays local.
    // Sentinel payload embeds the integration goal (expr) so any leaked
    // deposits at sentinel scope are self-identifying. The
    // "product_of_hypo_disintegration_of_integration_goal_" marker lets
    // cross-scope consumers (buildStack, Site A descendant iteration, proof-
    // graph rendering) exclude sentinel-scope orphans without pattern
    // guessing or substring heuristics on random tokens.
    NameId parentValId = memoryBlock.nameMap.encode(validityName);
    // Sentinel payload = "product_of_hypo_disintegration_of_integration_goal_"
    // + expr, built explicit-length on the string tier (freed after encodePush);
    // encodePush takes a span.
    NameId sentinelValId;
    {
        ScratchScope spScope(hypoStrArena);
        static const char SENT_PREFIX[] =
            "product_of_hypo_disintegration_of_integration_goal_";
        const int32_t spLen = static_cast<int32_t>(sizeof(SENT_PREFIX) - 1);
        const int32_t sLen = spLen + expr.len;
        char* sb = hypoStrArena.allocBytes(sLen);
        std::memcpy(sb, SENT_PREFIX, static_cast<size_t>(spLen));
        if (expr.len > 0) std::memcpy(sb + spLen, expr.ptr, static_cast<size_t>(expr.len));
        sentinelValId = memoryBlock.nameMap.encodePush(parentValId, StrSpan(sb, sLen));
    }
    // copyFrom: disintegrateExpr2 below mints NameMap heavily, so a raw decodeView
    // would dangle (I-3); the copy rides the string tier (a different arena).
    const ScratchString sentinelValidity = ScratchString::copyFrom(hypoStrArena,
        memoryBlock.nameMap.decodeView(sentinelValId).ptr,
        memoryBlock.nameMap.decodeView(sentinelValId).len);
    const unsigned deSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    DisintProducts out(&genScratchArenas().forSlot(deSlot));
    disintegrateExpr2(StrSpan(expr), memoryBlock, -1, true, StrSpan(sentinelValidity), out,
        /*trackHistoryLocal=*/false);

    // Only look at new statements (the out.statements channel). forEachSorted walks
    // in EWV::operator< order, byte-identical to the former std::set order; count()
    // is the deduped entry count == the former newStatementsEv.size().
    if (out.statements.count() == 1)
    {
        return;
    }
    // D19(b) user zero-heap order: iterate the disintegrateExpr2 EWV set directly
    // rather than materializing a std::set<std::string> of the originals. Byte
    // contract: ExpressionWithValidity::operator< is (original, validityName)
    // lexicographic (mail_types.hpp), so reading `.original` in set order yields
    // the former std::set<std::string> order EXACTLY, with equal originals
    // ADJACENT. Step 3's first-match goto = lex-smallest matching original in both
    // forms; step 6 is order-immaterial (count>1 break); step 7 keys on the
    // CONSTANT newValidityName, so the first visit of each distinct original
    // registers (identical push_back + mint) and each adjacent duplicate hits the
    // now-registered probe and continues — the same registration sequence as the
    // deduped set. The transient EncodedExpression stays deleted (D25 doors).

    // 3. Search for defSets for each input variable in the new statements (D20).
    // targetVars is a first-seen-order StrSpan run over `expr` (spans into
    // inputVars); the defSet value stays a std::string (a compiled-def config value).
    StrSpan targetVars[ExecutionParameters::MAX_ARITY];
    int32_t targetVarN = 0;

    for (int32_t vi = 0; vi < inputVarN; ++vi) {
        const StrSpan var = inputVars[vi];
        bool foundDef = false;
        // foundDefSetVal is a span into the never-deloaded config (coreConfig,
        // I-137), stable across the walk; a first-match guard reproduces the
        // former `goto def_found` (the loop is a pure search, so full iteration
        // with a first-match guard is byte-identical).
        StrSpan foundDefSetVal;

        out.statements.forEachSorted([&](StrSpan stmt, StrSpan /*v*/) {
            if (foundDef) return;
            const ce::CoreExpressionConfig* cfg =
                coreConfig(extractExpressionSpan(stmt));

            // "search for at least one expression which has entry in coreExpressionMap"
            if (cfg != nullptr) {
                StrSpan sArgs[ExecutionParameters::MAX_ARITY];
                const int32_t sArgN = getArgsSpans(stmt, sArgs,
                                                   ExecutionParameters::MAX_ARITY);

                // Check if 'var' is an argument in this statement
                for (int32_t k = 0; k < sArgN; ++k) {
                    if (equalSpans(sArgs[k], var)) {
                        // "there read defSet"
                        const std::pair<std::string, bool>* ds =
                            findDefinitionSet(cfg, k + 1);
                        if (ds != nullptr) {
                            foundDefSetVal = StrSpan(ds->first);
                            foundDef = true;
                            return; // first match (== the former goto def_found)
                        }
                    }
                }
            }
        });

        // "assert u found def set for each var"
        assert(foundDef);

        // "if there is at least one eligible var proceed"
        // Eligible means defSet == "(1)" based on context.
        // Deduplicate: if expr has the same variable in multiple argument
        // positions (e.g. (f[v,v,w])), inputVars contains duplicates.
        // Each variable must appear in newValidityName only once.
        if (equalSpans(foundDefSetVal, StrSpan("(1)", 3))) {
            bool present = false;
            for (int32_t t = 0; t < targetVarN; ++t)
                if (equalSpans(targetVars[t], var)) { present = true; break; }
            if (!present) {
                targetVars[targetVarN++] = var;
            }
        }
    }

    // "otherwise return"
    if (targetVarN == 0) {
        return;
    }

    // 4. Create a new validity name
    // Format: old + "_var0_x_var1_y_hypo_expr"
    // Example: main__var0_x_var1_y_hypo_(in[x,1])
    // hypoPayload = concat over targetVars of ("_var<i>_" + targetVars[i]) then
    // "_hypo_" + expr, built explicit-length on the string tier (%d ==
    // std::to_string for non-negative i, freed after encodePush); encodePush
    // takes a span.
    NameId hypoParentId = memoryBlock.nameMap.encode(validityName);
    NameId hypoNewValidityId;
    {
        ScratchScope hpScope(hypoStrArena);
        int32_t hpLen = 0;
        char di[16];
        for (int32_t i = 0; i < targetVarN; ++i) {
            const int dn = std::snprintf(di, sizeof(di), "%d", i);
            hpLen += 4 + dn + 1 + targetVars[i].len;   // "_var" + <i> + "_" + var
        }
        static const char HYPO_MID[] = "_hypo_";
        const int32_t hmLen = static_cast<int32_t>(sizeof(HYPO_MID) - 1);
        hpLen += hmLen + expr.len;
        char* hb = hypoStrArena.allocBytes(hpLen);
        int32_t hat = 0;
        for (int32_t i = 0; i < targetVarN; ++i) {
            std::memcpy(hb + hat, "_var", 4); hat += 4;
            const int dn = std::snprintf(di, sizeof(di), "%d", i);
            std::memcpy(hb + hat, di, static_cast<size_t>(dn)); hat += dn;
            hb[hat++] = '_';
            if (targetVars[i].len > 0) {
                std::memcpy(hb + hat, targetVars[i].ptr, static_cast<size_t>(targetVars[i].len));
                hat += targetVars[i].len;
            }
        }
        std::memcpy(hb + hat, HYPO_MID, static_cast<size_t>(hmLen)); hat += hmLen;
        if (expr.len > 0) { std::memcpy(hb + hat, expr.ptr, static_cast<size_t>(expr.len)); hat += expr.len; }
        assert(hat == hpLen);
        hypoNewValidityId = memoryBlock.nameMap.encodePush(hypoParentId, StrSpan(hb, hpLen));
    }
    // copyFrom: step 6's addExprToMemoryBlock + step 7's encode/nameMap.encode
    // mint NameMap, so a raw decodeView would dangle (I-3); the copy rides the
    // string tier (a different arena) and survives every mint.
    const ScratchString newValidityName = ScratchString::copyFrom(hypoStrArena,
        memoryBlock.nameMap.decodeView(hypoNewValidityId).ptr,
        memoryBlock.nameMap.decodeView(hypoNewValidityId).len);

    // 5. Find levels — caller-owned stack run; the sorted-unique union
    // (origin levels + the LB's own level) via insertLevelSorted covers
    // both the miss-branch singleton and the unconditional insert.
    int lvRun[256];
    int32_t lvN = 0;
    const int32_t originLvlsId = lookupStatementLevels(
        memoryBlock.intStatementLevelsMap, memoryBlock.nameMap, expr, validityName);

    if (originLvlsId) {
        lvN = coldIntRunAt(memoryBlock.intStatementLevelsMap, originLvlsId,
                           lvRun, 256);
    }
    lvN = insertLevelSorted(lvRun, lvN, memoryBlock.level, 256);

    // 6. Check Duplication and Add Equalities
    bool anyDuplicationFound = false;

    for (int32_t tv = 0; tv < targetVarN; ++tv) {
        const StrSpan var = targetVars[tv];
        bool foundDuplicationForVar = false;

        // Check duplication in newStatements (D22) — first-match guard (the loop
        // is order-immaterial: it only sets a bool on any count>1).
        out.statements.forEachSorted([&](StrSpan e, StrSpan /*v*/) {
            if (foundDuplicationForVar) return;
            StrSpan eArgs[ExecutionParameters::MAX_ARITY];
            const int32_t eArgN = getArgsSpans(e, eArgs,
                                               ExecutionParameters::MAX_ARITY);
            int count = 0;
            for (int32_t a = 0; a < eArgN; ++a) {
                if (equalSpans(eArgs[a], var)) count++;
            }
            if (count > 1) foundDuplicationForVar = true;
        });

        if (foundDuplicationForVar) {
            anyDuplicationFound = true;

            // Check if equality exists in validityName (not "main")
            bool hasExistingEquality = false;
            {
                const NameId varId = memoryBlock.nameMap.lookup(var);
                // Zero-copy view walk over the scope's class run — the former
                // decodeClassesAt heap snapshot. The scan is a pure read consumed
                // BEFORE the addExprToMemoryBlock mint below, so the peeked views
                // stay valid. Bucket resolution mirrors decodeClassesAt.
                const NameId dupValId = memoryBlock.nameMap.lookup(validityName);
                const int32_t bucketId = (dupValId == 0)
                    ? 0 : memoryBlock.equivalenceClassesMap.lookup(dupValId);
                if (bucketId != 0) {
                    const unsigned slot = (g_currentCoreId >= 0)
                        ? static_cast<unsigned>(g_currentCoreId)
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& gArena = genScratchArenas().forSlot(slot);
                    const int32_t cc = memoryBlock.equivalenceClassesMap.runLen(bucketId);
                    for (int32_t ci = 0; ci < cc; ++ci) {
                        ScratchScope viewScope(gArena);
                        int32_t blen = 0;
                        const char* bp = memoryBlock.equivalenceClassesMap.peekRecordBytes(
                            bucketId, ci, blen, gArena);
                        const EquivalenceClassView view{ bp, blen };
                        if (varId != 0 && view.memberCount() > 1
                            && classHasMember(view, varId)) {
                            hasExistingEquality = true;
                            break;
                        }
                    }
                }
            }

            if (!hasExistingEquality) {
                // Build "(=[Y,Y_copy])" on the string arena then u_-strip (D23, the
                // reactToHypo precedent): "(=[" + Y + "," + Y + "_copy])".
                const unsigned hSlot = (g_currentCoreId >= 0)
                    ? static_cast<unsigned>(g_currentCoreId)
                    : scratchArenas().slotCount() - 1;
                ScratchArena& hArena = scratchArenas().forSlot(hSlot);
                ScratchScope hScope(hArena);
                const StrSpan Y = var;
                const int32_t elen = 11 + 2 * Y.len;  // 3 + Y + 1 + Y + 7 ("_copy])")
                char* eb = hArena.allocBytes(elen);
                int32_t eat = 0;
                eb[eat++] = '('; eb[eat++] = '='; eb[eat++] = '[';
                if (Y.len > 0) std::memcpy(eb + eat, Y.ptr, static_cast<size_t>(Y.len));
                eat += Y.len;
                eb[eat++] = ',';
                if (Y.len > 0) std::memcpy(eb + eat, Y.ptr, static_cast<size_t>(Y.len));
                eat += Y.len;
                const char eSfx[] = "_copy])";
                std::memcpy(eb + eat, eSfx, 7); eat += 7;
                assert(eat == elen);
                const ScratchString equalityExpr = ScratchString::wrap(hArena, eb, elen);
                const ScratchString finalExpr =
                    removeUPrefixScratch(hArena, StrSpan(equalityExpr));

                // Dead-end axiom — see "variable copy" tag
                const TransientOrigin origin{ true, OriginTag::variableCopy, nullptr, 0 };

                addExprToMemoryBlock(StrSpan(finalExpr), memoryBlock, -1, 0, lvRun, lvN, origin, -1, -1, validityName, false);
            }
        }
    }

    // 7. Register NEW STATEMENTS with NEW validity name

    // D25: the transient EncodedExpression is deleted — the statement bytes are a
    // StrSpan over `stmt`, the validity a StrSpan over the D21 owned decode copy
    // `newValidityName` (stable across the loop's mints; it rides its own
    // std::string, not the NameMap pool). Span doors: lookupStatementFlags(StrSpan,
    // StrSpan), encodeExpression(StrSpan, StrSpan, NameMap&), nameMap.encode(StrSpan)
    // — same bytes -> same ids, same mint order (find-or-mint on identical bytes).
    out.statements.forEachSorted([&](StrSpan stmtSpan, StrSpan /*v*/) {
        const StrSpan newVldSpan(newValidityName);

        const StatementFlags* stmtRow = lookupStatementFlags(
            memoryBlock.intKnownStatements, memoryBlock.nameMap,
            stmtSpan, newVldSpan);
        if (stmtRow && stmtRow->registered) {
            return; // Skip if already present
        }

        { IntEncodedExpr ie = encodeExpression(stmtSpan, newVldSpan, memoryBlock.nameMap);
          memoryBlock.intEncodedStatements.push_back(ie);
          memoryBlock.intLocalEncodedStatements.push_back(ie);
          memoryBlock.intLocalEncodedStatementsDelta.push_back(ie);
          memoryBlock.intLocalEncodedStatementsSet.mint(
              packStatementKey(ie.originalId, ie.validityId));
          memoryBlock.intStatementLevelsMap.assignSetRange(
              packStatementKey(ie.originalId, ie.validityId),
              lvRun, lvRun + lvN); }
        upsertStatementKey(memoryBlock.intKnownStatements, packStatementKey(
            memoryBlock.nameMap.encode(stmtSpan),
            memoryBlock.nameMap.encode(newVldSpan)),
            /*local=*/true, /*registered=*/true, /*known=*/true);
    });
}

//#pragma optimize("", off)

void ExpressionAnalyzer::prehandleAnchor(Memory* mb) {
    if (mb == nullptr) return;
    if (mb->isPartOfRecursion)
    {
        return;
    }

    std::string anchorPrefix = "(" + this->anchorInfo.name;
    bool isAnchorLB = (mb->exprKey().rfind(anchorPrefix, 0) == 0);

    // Skip processing for the Anchor LB itself, but allow recursion
    if (!isAnchorLB) {

        // 1. Trace the hierarchy to find the specific Anchor Key
        std::string anchorExprKey;
        Memory* current = mb;
        std::set<std::string> traceVariables; // Added: Collect variables from trace

        while (current != nullptr) {
            if (!current->exprKey().empty()) {
                // Check if this ancestor is the Anchor LB
                if (current->exprKey().rfind(anchorPrefix, 0) == 0) {
                    anchorExprKey = current->exprKey();
                    break;
                }
                // Added: Collect variables from the current trace element's key
                std::vector<std::string> kArgs = ce::getArgs(current->exprKey());
                traceVariables.insert(kArgs.begin(), kArgs.end());
            }
            current = current->parentMemory;
        }

        // 2. If we found an anchor ancestor, proceed
        if (!anchorExprKey.empty()) {
            std::vector<std::string> args = ce::getArgs(anchorExprKey);
            std::map<std::string, std::string> replacementMap;

            // 3. Create Replacement Map based on definitionSets == "(1)"
            //    and save the new variables to intAxedVariables
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

                                    mb->intAxedVariables.mint(mb->nameMap.encode(xVar));
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
                const StatementFlags* anchorRow = lookupStatementFlags(
                    mb->intKnownStatements, mb->nameMap,
                    enc.original, enc.validityName);
                if (!(anchorRow && anchorRow->registered)) {

                    const int lv0[1] = { 0 };

                    { IntEncodedExpr ie = encodeExpression(enc, mb->nameMap);
                      mb->intEncodedStatements.push_back(ie);
                      mb->intLocalEncodedStatements.push_back(ie);
                      mb->intLocalEncodedStatementsDelta.push_back(ie);
                      mb->intLocalEncodedStatementsSet.mint(
                          packStatementKey(ie.originalId, ie.validityId));
                      mb->intStatementLevelsMap.assignSetRange(
                          packStatementKey(ie.originalId, ie.validityId),
                          lv0, lv0 + 1); }
                    upsertStatementKey(mb->intKnownStatements, packStatementKey(
                        mb->nameMap.encode(enc.original),
                        mb->nameMap.encode(enc.validityName)),
                        /*local=*/true, /*registered=*/true, /*known=*/true);

                    // L3 span-record door. The "anchor handling" antecedent
                    // (the original anchor expression `anchorExprKey`) and the
                    // KEY (`replacedAnchor`) are in-hand std::string locals;
                    // "main" is a static literal.
                    const StrSpan pMain("main", 4);
                    const OriginDep pDeps[1] = { { StrSpan(anchorExprKey), pMain } };
                    addOriginEncoded(mb->exprOriginMap, mb->originInterner, StrSpan(replacedAnchor), pMain, OriginTag::anchorHandling, pDeps, 1, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
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
                }
            }
        }
    }

    // 5. Recurse into children
    simpleMapStore.forEachChild(mb, [&](const gl::StrSpan&, Memory* child) {
        prehandleAnchor(child);
    });
}

std::vector<std::string> ExpressionAnalyzer::runCeFilterOnly(
    const std::vector<std::string>& theorems) {

    auto t0 = std::chrono::high_resolution_clock::now();
    std::cout << "Counter example based filtering started." << std::endl;

    std::vector<std::string> filteredConjectures = theorems;

    if (!parameters.skip_ce_filter) {
        if (parameters.mirror_refutation) {
            const auto mirrorPairsPath =
                std::filesystem::path(__FILE__).parent_path().parent_path()
                    .parent_path().parent_path() / "files" / "theorems"
                    / "mirror_pairs.txt";
            mirrorPartnerMap = loadMirrorPairs(mirrorPairsPath);
            std::cout << "CE filter: " << mirrorPartnerMap.size()
                      << " mirror-partner keys loaded." << std::endl;
        }
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

    return filteredConjectures;
}

/// @brief Tear down the LB grid at batch end — delete every child Memory and
///        reset the root's grid links. See the declaration for the full
///        contract.
///
/// @details
/// The four `HashMemory` instances per LB are arena-backed (every `encodedMap`
/// on the cold `lbMemory.manager`) yet declared before that arena, so their
/// `~PagedVector` would otherwise `freePage` on a dead arena. Empty each LB's
/// `encodedMap` while the arena is alive — children before `delete`, the root in
/// place — then drop the grid links. A deload-released LB is already empty, so
/// the release is residency-gated.
void ExpressionAnalyzer::destroyGrid() {
    // 1) Delete every child Memory, emptying its arena-backed encodedMaps
    //    first so ~Memory does not freePage on an already-released arena.
    std::unordered_set<Memory*> seen;
    seen.insert(&this->body);  // never delete the root
    std::vector<Memory*> stack;
    simpleMapStore.forEachChild(&this->body, [&](const gl::StrSpan&, Memory* child) {
        if (child) stack.push_back(child);
    });
    while (!stack.empty()) {
        Memory* node = stack.back(); stack.pop_back();
        if (!node || !seen.insert(node).second) continue;
        simpleMapStore.forEachChild(node, [&](const gl::StrSpan&, Memory* child) {
            if (child) stack.push_back(child);
        });
        if (node->lbMemory.manager.resident()) {
            node->overallHashMemory.releaseAllCold();
            node->localHashMemory.releaseAllCold();
            node->localHashMemoryDelta.releaseAllCold();
            node->workingMemory.releaseAllCold();
        }
        lbStore.destroy(node);
    }
    // 2) Empty the root's arena-backed cold HashMemory containers in place (its
    //    arena destructs before them at end of process — same hazard).
    if (this->body.lbMemory.manager.resident()) {
        this->body.overallHashMemory.releaseAllCold();
        this->body.localHashMemory.releaseAllCold();
        this->body.localHashMemoryDelta.releaseAllCold();
        this->body.workingMemory.releaseAllCold();
    }
    // 3) Drop child links and reset the grid LB list to the root.
    simpleMapStore.clear();
    permanentBodies.clear();
    permanentBodies.push_back(&this->body);
    // New mail system (D-137): drop the per-LB pull logs and
    // cursors. The Memory* keys were just freed above, so clearing here keeps a
    // stale pointer-keyed entry from aliasing a reused address next batch.
    this->mailLog.clear();
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
void ExpressionAnalyzer::analyzeExpressions(const std::vector<std::string>& theorems,
    const std::vector<std::string>& provedTheorems,
    const std::vector<std::string>& externalTheorems) {

    // Statification: one gl_quick invocation = one batch — empty the
    // deload directory at batch START only (end-of-run files stay on
    // disk for post-run inspection). The accumulated manifest resets
    // with it.
    // Close any prior batch's extent file BEFORE the purge deletes it (an open
    // handle blocks the delete on Windows), then purge + reopen a fresh one.
    staticMemory().closeExtentFile();
    lbdeload::purgeDeloadDirectory(lbdeload::kDeloadDirectory);
    staticMemory().resetDeloadRegistry();
    if (parameters.enable_extent_deload) {
        // Preallocate 1.5x the pool: live extent bytes are the Dumped (non-
        // resident) LBs' slabs, so the extent legitimately exceeds pool size
        // (disk > RAM is the point of deload). Growth covers under-estimates.
        const int64_t initialBytes =
            staticMemory().totalBlocks() * staticMemory().blockBytes() * 3 / 2;
        staticMemory().openExtentFile(
            std::filesystem::path(lbdeload::kDeloadDirectory) / "extent.bin",
            initialBytes);
    }
    pendingDischarge.clear();
    stewardEvictionPlan.clear();

    // 1. CE Filtering — extracted to `runCeFilterOnly` so the `--ce-only`
    // fast-iteration entry point can reuse the exact same code path. Returns
    // the surviving conjectures (== `theorems` when `skip_ce_filter` is true).
    std::vector<std::string> filteredConjectures = runCeFilterOnly(theorems);

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

    // Grid teardown is the member `ExpressionAnalyzer::destroyGrid()`, called
    // after the proof-graph export in `run_modes::fullRun` (the grid must
    // survive `analyzeExpressions` for the export).

    // =========================================================================
    // Helper: build grid from current permanentBodies
    // =========================================================================
    auto buildGrid = [&]() {
        this->prefillIntegrationMapsRecursive(&this->body);
        this->prehandleAnchor(&this->body);

        // Process-documentation telemetry for the mail-history experiment.
        // Count after every LB has been created and anchor prehandling has run,
        // at the exact program point where the retention policy is fixed.
        this->dormantLogicBlocksAtGridBuild = 0;
        for (const Memory* lb : permanentBodies) {
            assert(lb != nullptr && "buildGrid: permanentBodies contains null");
            if (!lb->isActive) ++this->dormantLogicBlocksAtGridBuild;
        }
        this->rollingMailHistoryEnabled =
            this->dormantLogicBlocksAtGridBuild == 0;

        // New mail system (D-137): register every LB so the
        // parallel phase-1 pull only advances pre-existing cursor cells, never
        // inserting or rehashing. Every LB the prover will ever run -- the root
        // &body included (it is pushed into permanentBodies at construction) --
        // is in permanentBodies and no LB is born after this point, so this
        // single pass covers every ancestor any pull will read.
        for (Memory* lb : permanentBodies) {
            if (!lb) continue;
            std::vector<const Memory*> ancestors;
            for (const Memory* p = lb->parentMemory; p != nullptr; p = p->parentMemory) {
                ancestors.push_back(p);
            }
            this->mailLog.registerLb(lb, ancestors);
        }

        // New mail system (D-137): commit every LB's startup
        // mailOut (the LB-creation paired writes at theorem-load +
        // prehandleAnchor) into its own log, so each descendant pulls its
        // ancestors' startup batches at step-1 phase-1. Replaces the old
        // sendMail-into-boxes + smashMail dispatch. Single-threaded at grid
        // build; every LB was registered just above, so commit never inserts.
        for (Memory* lb : permanentBodies) {
            if (!lb) continue;
            this->mailLog.commit(lb, lb->mailOut);
            lb->clearMailOut();
        }
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
    auto t0 = std::chrono::high_resolution_clock::now();

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
            // Quiescence: warm-up bursts never skip (D-194) —
            // they are productive priming and the belt keeps every active LB swept.
            this->warmUpPhase = true;
            this->prove(preIterations, permanentBodies);
            this->warmUpPhase = false;
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
        }

        if (remainingIterations > 0) {
            std::cout << "Phase 2: Running remaining " << remainingIterations << " iterations..." << std::endl;
            this->prove(remainingIterations, permanentBodies);
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
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration<double>(t1 - t0);
    std::cout << "Prover finished." << std::endl;
    std::cout << "Runtime prover: " << dt.count() << " seconds" << std::endl;
    // Statification sizing telemetry: drives the static_pool_bytes /
    // static_block_bytes config audit (see G-52).
    std::cout << "[STATIC-MEMORY] peak blocks in use: "
              << staticMemory().peakBlocksInUse() << " of "
              << staticMemory().totalBlocks() << " ("
              << staticMemory().blockBytes() << " bytes per block)"
              << std::endl;
    // Extent-file occupancy (D-195): live = Σ Dumped
    // raw-LB image bytes, allocated = Σ slab class sizes (alloc/live =
    // internal slack), file = preallocation high-water (file/alloc =
    // free-list + preallocation overhang) — the class-ladder / preallocation
    // tuning inputs. Zero when the extent path is off.
    if (staticMemory().useExtent()) {
        constexpr double kGiBd = 1024.0 * 1024.0 * 1024.0;
        std::cout << "[STATIC-MEMORY] extent file: live "
                  << (static_cast<double>(staticMemory().extentLiveBytes())
                      / kGiBd)
                  << " GiB | allocated "
                  << (static_cast<double>(staticMemory().extentAllocatedBytes())
                      / kGiBd)
                  << " GiB | file "
                  << (static_cast<double>(staticMemory().extentFileBytes())
                      / kGiBd)
                  << " GiB" << std::endl;
    }
    // End-of-batch deload telemetry summary, then reset the process-wide
    // aggregate for the next batch (single-threaded here — the steward is
    // stopped once prove() returns). Pure telemetry (Rule 16 / I-44).
    deloadStats().reportSummary();
    deloadStats().reset();
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
