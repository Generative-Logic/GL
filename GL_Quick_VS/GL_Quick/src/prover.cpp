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
#include "infra/diagnostics_log.hpp"

#include "parameters.hpp"
#include "infra/hashburst_dump.hpp"
#include "infra/rt_tracker.hpp"
#include "infra/mem_tracker.hpp"
#include "memory_infra/global_memory_manager.hpp"
#include "memory_infra/scratch_arena.hpp"
#include "memory_infra/lb_deload.hpp"
#include "memory_infra/deload_stats.hpp"
#include "gpu/phase2_projection.hpp"
#ifdef GL_CUDA
#include "gpu/phase2_cuda.hpp"
#include "gpu/phase2_sealing.hpp"
#endif
#include <iostream>
#include <fstream>
#include <array>
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
/// @param selectedPhase2Backend Explicit Phase 2 execution backend retained
///                              for every proof iteration in this batch.
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
ExpressionAnalyzer::ExpressionAnalyzer(
    std::string anchorID,
    std::optional<Phase2Backend> selectedPhase2Backend)
    :parameters(),
    phase2Backend(Phase2Backend::cpu),
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
                if (pp.contains("use_gpu")) parameters.use_gpu = pp["use_gpu"];
                if (pp.contains("allow_ssd_deload")) parameters.allow_ssd_deload = pp["allow_ssd_deload"];
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

    phase2Backend = selectedPhase2Backend.value_or(
        parameters.use_gpu ? Phase2Backend::cuda : Phase2Backend::cpu);
#ifndef GL_CUDA
    // A build without CUDA support (a Makefile build with USE_CUDA=0) has no
    // CUDA route at all: selecting one is refused at startup, never worked
    // around by the processor route (I-211).
    assert(phase2Backend == Phase2Backend::cpu
        && "the CUDA Phase 2 route was selected, but this build has no CUDA "
           "support (build the Visual Studio project, or make USE_CUDA=1)");
#endif
    // The single backend-ownership boundary derives the SSD policy: CUDA and
    // SSD deload are mutually exclusive, so a CUDA batch runs the
    // resident-only steward; a processor batch keeps its configured
    // permission (default true) and pages through the working-set steward.
    parameters.allow_ssd_deload = parameters.allow_ssd_deload
        && phase2Backend == Phase2Backend::cpu;
    std::cout << "[EXECUTION-POLICY] phase2_backend="
              << (phase2Backend == Phase2Backend::cuda ? "cuda" : "cpu")
              << " config_use_gpu=" << (parameters.use_gpu ? "true" : "false")
              << " allow_ssd_deload="
              << (parameters.allow_ssd_deload ? "true" : "false")
              << (selectedPhase2Backend.has_value()
                    ? " source=command-line-override" : " source=batch-config")
              << std::endl;
    // Batch header in the diagnostics log: every telemetry line that follows
    // belongs to this analyzer's batch until the next header.
    diagnosticsLog() << "=== batch " << anchorID << " phase2_backend="
                     << (phase2Backend == Phase2Backend::cuda ? "cuda" : "cpu")
                     << " ===" << std::endl;

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
    // Same shape as the string registry — logicalCores worker slots PLUS the
    // reserved single-threaded slot (index logicalCores) that every
    // g_currentCoreId == -1 caller resolves to: the seam drains, the CE-fact
    // load, the compressor, and any steward / io thread that reaches
    // gen-scratch (a canonical reload's reverse-index rebuild). Without it the
    // "reserved" fallback was worker logicalCores-1's own arena and its
    // rule-index staging pool — a shared arena between a worker and whichever
    // non-worker thread hit the fallback while that worker ran.
    initGenScratchArenas(logicalCores + 1);

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
// filterIntEncodedStatements(), generateEncodedRequestsStatic(), and
// checkLocalEncodedMemoryStatic().

// ========================================================================
// Static pipeline: IntEncodedExpr-based, zero-alloc request generation
// ========================================================================

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
thread_local uint64_t ExpressionAnalyzer::g_gpuGrowAttemptsByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local uint64_t ExpressionAnalyzer::g_gpuGrowFrontierByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local uint64_t ExpressionAnalyzer::g_gpuGrowSubkeysByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local uint64_t ExpressionAnalyzer::g_gpuGrowRequestsByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local uint64_t ExpressionAnalyzer::g_gpuProducerAttemptsByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local uint64_t ExpressionAnalyzer::g_gpuProducerSurvivorsByDepth[
    ExecutionParameters::MAX_EXPRESSIONS + 1]{};
thread_local ExpressionAnalyzer::GpuEvaluationUsage
    ExpressionAnalyzer::g_gpuEvaluationUsage{};
// I-28 detect-and-defer trial: set true only inside a parallel phase-1/phase-3
// worker task (the runPhase wrappers); gates updateAdmissionMap3's ancestor write.
thread_local bool ExpressionAnalyzer::g_inParallelWorkerPhase = false;
// Absorb-door scratch accessor: the phase-1/phase-3 workers publish their
// coreId here on entry; prefixArgumentsWithU reads it to pick its per-slot
// scratch arena. -1 (the default, never overwritten on single-threaded setup
// threads) maps to the reserved scratch slot.
thread_local int ExpressionAnalyzer::g_currentCoreId = -1;
#if RT_MEASUREMENT
// [RT phase measurement] C15 submatch-attribution state (see prover.hpp).
thread_local bool ExpressionAnalyzer::g_crtArmed = false;
thread_local NameId ExpressionAnalyzer::g_crtPreorderId = 0;
thread_local NameId ExpressionAnalyzer::g_crtFiveId = 0;
std::atomic<int64_t> ExpressionAnalyzer::g_crtAttempts{ 0 };
std::atomic<int64_t> ExpressionAnalyzer::g_crtAttemptsUnlinked{ 0 };
std::atomic<int64_t> ExpressionAnalyzer::g_crtAccepted{ 0 };
std::atomic<int64_t> ExpressionAnalyzer::g_crtAcceptedUnlinked{ 0 };
#endif

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

/// @see Declaration in `prover.hpp` for the full contract.
void ExpressionAnalyzer::performElem2(const Memory& body, unsigned coreId,
    int partCount,
    const SplitStumpRef& splitStump,
    SealedPageSet& sealedPages,
    std::atomic<int64_t>& doomLine) {

    // The LB split's expression dimension. A stump-split sub-part carries a bucket
    // of stumps, and every request it generates contains one of them; an ordinary
    // whole-LB part carries none.
    assert((splitStump.count == 0) == (splitStump.stumps == nullptr)
        && "performElem2: stump bucket present iff non-empty");
    // A stump bucket only ever rides a MULTI-PART LB: the whole-LB expression split
    // runs partCount bucket parts, and the doom line's winner
    // selection relies on each bucket part carrying its bucket ordinal among
    // partCount siblings.
    assert((splitStump.count == 0 || partCount > 1)
        && "performElem2: a stump bucket belongs to a MULTI-PART LB");
    // The burst reads the rule indexes: every staged install write was
    // flushed at its window's close (D-333), so this
    // slot's stagings are empty.
    assert(ruleStagings().slotIsEmpty(coreId)
        && "performElem2: staged rule-index writes reached the burst - a flush seam was missed");
    assert((splitStump.count == 0 || splitStump.total >= 1)
        && "performElem2: a stump sub-part has a place among its siblings");
    assert(partCount >= 1 && "performElem2: partCount is the LB's part count (>= 1)");
    // The doom line identifies its winner by the part's invocation ordinal; an
    // unsplit part is ordinal 0 by construction (default SplitStumpRef).
    assert((splitStump.count > 0 || splitStump.ordinal == 0)
        && "performElem2: an unsplit part carries ordinal 0");

    // Per-call RT tracker home. performElem2 is the hashburst (request
    // generation + the inline fixpoint check) — where a runaway LB spends its
    // time — and its REQGEN_BATCH* / FIXPOINT_LOOP RT_SCOPE_HERE markers already
    // record into the thread-local tracker this declares. Stripped to nothing
    // when RT_MEASUREMENT == 0. RT requires disable_lb_split (one part per LB) so
    // this call measures the LB's whole hashburst on one thread, writing one
    // .rt/<chain>.log per LB with no cross-part races. See
    // D-110, I-59.
    RT_TRACKER_DECL(body);

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
    // part. The generator's grow-DFS bumps it on every match this part owns;
    // canAccept caps the burst on it, and the worker
    // reads it after this call to drive the split policy. See D-109.
    g_growthMatchCount = 0;

#if RT_MEASUREMENT
    // [RT phase measurement] arm the C15 attribution for the dump-target
    // LB's parts only; reassigned on every entry so no stale thread state.
    g_crtArmed = gl::hashburst_dump::isTargetLB(body);
    if (g_crtArmed) {
        g_crtPreorderId = body.nameMap.lookup(std::string("preorder"));
        g_crtFiveId = body.nameMap.lookup(std::string("5"));
        if (g_crtPreorderId == 0 || g_crtFiveId == 0) g_crtArmed = false;
    }
#endif

    // Streaming consumer: each generated request is checked inline (dependency
    // skip + checkLocalEncodedMemoryStatic) instead of being buffered and
    // checked afterwards. The external `doomLine` atomic — per-LB, shared by
    // all the LB's parts and living OUTSIDE the LB — lets any part end its
    // burst early without writing anything on the LB (I-66): a doom trigger
    // publishes its deterministic stream position, every part stops once its
    // own counter passes that position, and the finalize merges the winning
    // part's chain alone.
    //
    // No submatch cap: a burst without a doom trigger runs to COMPLETION and a
    // straggler is split preemptively next iteration (the stats-driven trigger
    // in proveKernel), never truncated mid-burst. The CE filter was already
    // uncapped (ceFilteringActive).
    BurstSink sink{ this, &body, coreId, &sealedPages, &doomLine,
                    static_cast<int32_t>(splitStump.ordinal),
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
        // --- CE mode: no ingredient is obligatory, so the term list is empty and
        // every whole key is a request.
        //
        // The counter-example filter runs one unsplit LB per conjecture, which is
        // what licenses its submatch tally going unread (the cap is bypassed in
        // BurstSink::canAccept, and an unsplit LB has no split policy). It is the
        // only caller for which that holds, so the statement lives here and not in
        // the generator, which cannot tell this call from a termless prover batch.
        assert(partCount == 1
            && "performElem2: the counter-example filter runs one unsplit LB per "
               "conjecture");
        this->generateEncodedRequestsStatic(body, body.overallHashMemory,
            /*terms=*/nullptr, /*termCount=*/0,
            SplitStumpRef{}, coreId, sink);
    } else {
        // --- Normal mode: four request batches, one per rule registry ---
        // Each batch names, as mandatory-containment terms, the ingredient a
        // request must be new because of; batch 5 names none, because there the
        // rule itself is what is new. The registries differ, which is why the
        // batches cannot share one call: the registry drives the key length, the
        // target owner map and every owner probe. Batch 1 reads the persistent
        // `body.workingMemory` and the mail batches read
        // `body.intExternalStatements`, all filled by the absorb above.

        // --- Batch 1: mail-recovered rules (body.workingMemory) ---
        // Filled by the absorb above (status=3 recovered implications). What is
        // new here is the RULE, so a request has to carry it against a fact this
        // LB already holds: one term naming the local statements. The leading
        // sink.canAccept() skips the batch once the LB hit the cap or was
        // early-exited by another part; the emptiness test skips a search whose
        // only term no candidate could satisfy.
        if (sink.canAccept() && !body.workingMemory.encodedMap.empty()
            && !body.intLocalEncodedStatements.empty()) {
            RT_SCOPE_HERE("REQGEN_BATCH1_WORKING_MEMORY");
            MandatoryTerm terms[1];
            terms[0].views[0] = IntStmtView(body.intLocalEncodedStatements);
            terms[0].viewCount = 1;

            this->generateEncodedRequestsStatic(body, body.workingMemory,
                terms, /*termCount=*/1,
                splitStump, coreId, sink);
        }

        // --- Batch 2+3: everything new this burst, in ONE search ---
        // Both former batches read the same rule registry (overallHashMemory) and
        // the same statement universe (intEncodedStatements); they differed only
        // in what they obliged a request to contain. The mandatory-containment
        // control states that difference directly, so one enumeration covers both
        // and reaches each request exactly once — no pairing merge, and no
        // cross-batch duplicate for a request that carries both a fresh local
        // statement and a fresh arrival.
        //
        // Term 1: a statement derived here this burst.
        // Term 2: a mail arrival AND a local statement — a purely-external
        //         combination already fired at the ancestor that mailed it (I-57),
        //         and batch 4 below covers arrivals against this LB's own rules.
        if (sink.canAccept()) {
            RT_SCOPE_HERE("REQGEN_BATCH23_NEW_THIS_BURST");
            // The mail side is staged by the same absorb that offers each arrival
            // to the statement registry, so a staged arrival that clears the
            // iteration cap is registered here and the one grow universe below
            // reaches it. A miss would silently drop every request that arrival
            // could serve, so it is a hard stop, not a widened universe.
            for (int32_t k = 0; k < body.intExternalStatements.size(); ++k) {
                const IntEncodedExpr& ext = body.intExternalStatements[k];
                if (ext.maxIteration > parameters.maxIterationNumberVariable)
                    continue;
                assert(body.intKnownStatements.find(
                           StatementKey{ ext.originalId, ext.validityId }) != nullptr
                    && "performElem2: a staged mail arrival must still be "
                       "registered at this LB - the absorb stages only what the "
                       "registry accepted, and every registry erase purges the "
                       "matching staged row");
            }

            MandatoryTerm terms[2];
            terms[0].views[0] = IntStmtView(body.intLocalEncodedStatementsDelta);
            terms[0].viewCount = 1;
            terms[1].views[0] = IntStmtView(body.intExternalStatements);
            terms[1].views[1] = IntStmtView(body.intLocalEncodedStatements);
            terms[1].viewCount = 2;

            this->generateEncodedRequestsStatic(body, body.overallHashMemory,
                terms, /*termCount=*/2,
                splitStump, coreId, sink);
        }

        // --- Batch 4: localHashMemory x fresh mail ---
        // Every rule this LB already owns has to be tested against the arrivals
        // this burst brought in, so the mandatory ingredient is the mail side:
        // one term naming body.intExternalStatements (filled by the absorb above).
        if (sink.canAccept() && !body.localHashMemory.encodedMap.empty()
            && !body.intExternalStatements.empty()) {
            RT_SCOPE_HERE("REQGEN_BATCH4_LOCAL_X_MAIL_SINGLES");
            MandatoryTerm terms[1];
            terms[0].views[0] = IntStmtView(body.intExternalStatements);
            terms[0].viewCount = 1;

            this->generateEncodedRequestsStatic(body, body.localHashMemory,
                terms, /*termCount=*/1,
                splitStump, coreId, sink);
        }

        // --- Batch 5: localHashMemoryDelta ---
        // The rule is what is new here, and a rule that lands this burst must
        // meet every visible statement, not only the ones that arrived with it.
        // So there is no mandatory ingredient at all: the term list is EMPTY and
        // every whole key of this registry is a request. (Naming the whole
        // statement universe as a term would say the same thing and then pay a
        // per-candidate mask test that can never fail.)
        if (sink.canAccept() && !body.localHashMemoryDelta.encodedMap.empty()) {
            RT_SCOPE_HERE("REQGEN_BATCH5_LOCAL_HASH_DELTA");
            this->generateEncodedRequestsStatic(body, body.localHashMemoryDelta,
                /*terms=*/nullptr, /*termCount=*/0,
                splitStump, coreId, sink);
        }
    } // end normal mode

    // Per-step delta clears are deliberately NOT done here. A later part of a
    // multi-part LB still reads body.localHashMemoryDelta /
    // body.intLocalEncodedStatementsDelta; clearing per-part would wipe the
    // input the next part needs. performElemPhase2 clears them ONCE, after every part's
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
    // early-exit only lowers the external `doomLine` atomic, never the LB.


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
    // the per-batch containers held this iteration). The slot's rule-index
    // stagings ride it too and must be empty here (every window flushed).
    assert(ruleStagings().slotIsEmpty(coreId)
        && "performElem2 exit: staged rule-index writes outlive their window");
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
    SealedPageSet* const* parts, int32_t partCount,
    int64_t doomLine,
    bool firingRecordsCanonical) {
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
        body.admissionKeysOrdis2.clear();

        // Merge the parts' captured deposits in canonical sorted order —
        // partition-independent (D-117): the same
        // firing SET sorts identically regardless of how it was split or in what
        // thread order the parts completed. This is why flat parallel execution
        // stays deterministic.
        //
        // Doom line set: a doom trigger fired this burst, so the merge takes
        // ONLY the winning part's chain (the lexicographic-minimum
        // (position, ordinal) trigger — a pure function of the parts'
        // deterministic streams). The other parts' chains are discarded unread:
        // the trigger guarantees phase 3 discharges the LB, so sibling records
        // are wasted-once-doomed and dropping them keeps the merged set
        // deterministic (their tails end wherever the line caught them). An
        // unsplit LB is its own winner, so its early-exit merge is unchanged.
        if (firingRecordsCanonical) {
            assert(partCount == 1
                && "canonical GPU output is one compacted chain per LB");
            assert(doomLine == kNoDoomLine
                && "canonical GPU output already applied doom selection");
            this->applyFiringRecords(body, parts, partCount, true);
        } else if (doomLine != kNoDoomLine) {
            assert(partCount > 0
                && "performElemPhase2: a doom trigger implies a fired part");
            const int32_t w = doomLineOrdinal(doomLine);
            assert(w < partCount
                && "performElemPhase2: doom-line winner outside this LB's parts");
            SealedPageSet* const winner[1] = { parts[w] };
            this->applyFiringRecords(body, winner, 1, false);
        } else {
            this->applyFiringRecords(body, parts, partCount, false);
        }
    }

    // Quiescence (D-194): admission-map churn this burst is a
    // mutation the phase-3 statement-count diff cannot see — a marker firing
    // registers an admission template without depositing a statement, yet it can
    // enable a fresh it_/int_ admission (and thus a firing) next burst. Flag it
    // so the LB stays awake. Read while the staging vectors still hold this
    // burst's records, before the drains below consume them.
    if (!body.admissionKeysAlgebra.empty() || !body.deferredIntegrationPreps.empty()
        || !body.admissionKeysOrdis2.empty())
        body.mutatedThisBurst = true;

    // Drain the per-burst admission / integration / demand records (replay
    // in firing order before phase 3's post-burst standardProcessing
    // absorb). Run for every LB, active or discharged in phase 1.
    this->drainAdmissionKeysAlgebra(body);
    this->drainDeferredIntegrationPreps(body);
    this->drainAdmissionKeysOrdis2(body);

    // The drained staging vectors carried sealed views into the tasks' page
    // sets; clear them NOW, before the post-join sweep frees the pages, so
    // no dangling view ever sits in a Memory container — and no staging can
    // leak across iterations into a later drain whose pages are long freed.
    // (The pre-merge clears above remain: they cover the inactive path's
    // bookkeeping and the D-126 delta contract.)
    body.admissionKeysAlgebra.clear();
    body.deferredIntegrationPreps.clear();
    body.admissionKeysOrdis2.clear();

}

/// @see Declaration in `prover.hpp` for the full contract.
void ExpressionAnalyzer::performElemPhase1(Memory& body, unsigned coreId) {
    // Publish this worker's slot for the absorb door's per-slot scratch arena
    // (read deep in disintegration by prefixArgumentsWithU).
    g_currentCoreId = static_cast<int>(coreId);
    // Per-call phase-1 tracker: feeds the cross-burst aggregate so the
    // pre-burst absorb (standardProcessing and the deposit-door tree under
    // it) is attributed down to its atomic sections. Per-LB `.rt` files stay
    // trigger-gated; the aggregate records every call.
    RT_TRACKER_DECL(body);
#if PHASE13_DEEP_TIMING
    const auto recordPhase1Detail =
        [this, coreId](Phase13TimingSlot slot,
                       std::chrono::steady_clock::time_point started) {
            // No rows bound = the defined no-measurement state (unit tests /
            // CE clones run phase helpers outside proveKernel's binding) —
            // same contract as standardProcessing's recordPhase13Detail.
            if (phase13TimingRows == nullptr) return;
            assert(coreId < phase13TimingWorkers
                && "Phase 1 timing row is not bound to this worker");
            phase13TimingRows[
                static_cast<std::size_t>(coreId) * kPhase13TimingSlotCount
                + static_cast<std::size_t>(slot)] +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count();
        };
    const auto claimLoadStarted = std::chrono::steady_clock::now();
#endif
    // Unified working-set handshake (the user's "a worker uploads itself"):
    // claim this LB and make it resident before ANYTHING reads it — the dump
    // trap below included. All three phases use the same handshake
    // (I-114). It reloads a cold LB, making room first
    // via the load<->evict exchange, and bounds its `Busy` wait with the
    // stuck-assert (I-113). The claim is released at the end
    // of the body so the steward may reclaim the LB once done. CE-filter
    // clones run with NO steward (always resident, never deloaded) and skip
    // the handshake entirely.
    {
        RT_SCOPE("PH1_CLAIM_LOAD");
        if (steward)
            steward->claimAndLoadForWork(body, /*phase=*/1,
                                         lbdeload::kDeloadDirectory);
    }
#if PHASE13_DEEP_TIMING
    recordPhase1Detail(Phase13TimingSlot::claimLoad, claimLoadStarted);
    const auto burstSetupStarted = std::chrono::steady_clock::now();
#endif

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
#if RT_MEASUREMENT
        // [RT phase measurement] fresh per-burst attribution tallies;
        // burst separator in the hit log (truncate on the first burst of a run).
        g_crtAttempts.store(0, std::memory_order_relaxed);
        g_crtAttemptsUnlinked.store(0, std::memory_order_relaxed);
        g_crtAccepted.store(0, std::memory_order_relaxed);
        g_crtAcceptedUnlinked.store(0, std::memory_order_relaxed);
        {
            static std::atomic<int> crtBurstNo{ 0 };
            const int b = crtBurstNo.fetch_add(1, std::memory_order_relaxed);
            std::ofstream hf(".debug/c15_successful_submatches.txt",
                             b == 0 ? std::ios::trunc : std::ios::app);
            hf << "== target-LB burst " << (b + 1) << " ==\n";
        }
#endif
    }

    // The per-step delta-class tracker `changedClassesThisStep` is
    // deliberately NOT cleared here: its sole clear is the tail of
    // `standardProcessing` (Step 7), after every consumer has read it. A
    // delta minted OUTSIDE any `standardProcessing` call — LB seeding
    // (`addTheoremToMemory` copy axioms / equality premises), post-join
    // barrier deposits — must survive to this step's `applyEquiClasses`,
    // which back-applies it to every pre-existing statement (Pass 1). An
    // entry clear here discards exactly those deltas unconsumed (the
    // waterline seeded at class creation then hides the pre-existing
    // statements from Pass 2 forever).

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
#if PHASE13_DEEP_TIMING
    recordPhase1Detail(Phase13TimingSlot::burstSetup, burstSetupStarted);
    const auto routingMailPullStarted = std::chrono::steady_clock::now();
#endif

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
        RT_SCOPE("PH1_MAIL_PULL");
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
#if PHASE13_DEEP_TIMING
    recordPhase1Detail(Phase13TimingSlot::routingMailPull,
                       routingMailPullStarted);
#endif

    // mailIn is HOT (I-101): the absorb reads its canonical
    // sorted snapshots directly — no transient heap Mail. Cleared here, after the
    // drain returns; nothing between standardProcessing's drain and its return
    // reads it.
    this->standardProcessing(body,
                             /*externalMailIn=*/&body.mailIn,
                             /*internalMailIn =*/body.nextIterationInternalMail,
                             /*internalMailOut=*/body.sameIterationInternalMail,
                             coreId);
#if PHASE13_DEEP_TIMING
    const auto routingMailCleanupStarted =
        std::chrono::steady_clock::now();
#endif
    body.mailIn.clear();
    if (!ceFilteringActive && !parameters.compressor_mode) {
        const int64_t beforeRelease = routingMailInBlocksInFlight.fetch_sub(
            trackedMailInBlocks, std::memory_order_relaxed);
        assert(beforeRelease >= trackedMailInBlocks
            && "routing mailIn attribution counter underflow");
    }
#if PHASE13_DEEP_TIMING
    recordPhase1Detail(Phase13TimingSlot::routingMailCleanup,
                       routingMailCleanupStarted);
#endif
    } // RT_SCOPE PRE_FIXPOINT_MAIL_ABSORB

    // Release the claim — phase 1 is done with this LB, so it is deloadable
    // again (the steward may now reclaim it). CE clones have no steward.
#if PHASE13_DEEP_TIMING
    const auto releaseClaimStarted = std::chrono::steady_clock::now();
#endif
    if (steward) {
        assert(body.stewardClaim.load(std::memory_order_relaxed)
                   == static_cast<uint8_t>(Memory::StewardClaim::WorkerOwned)
               && "phase-1 release of an LB not held WorkerOwned");
        body.stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle),
            std::memory_order_release);
    }
#if PHASE13_DEEP_TIMING
    recordPhase1Detail(Phase13TimingSlot::releaseClaim,
                       releaseClaimStarted);
#endif
}

/// @see Declaration in `prover.hpp` for the full contract.
void ExpressionAnalyzer::performElemPhase3(Memory& body, unsigned coreId) {
    // Publish this worker's slot for the absorb door's per-slot scratch arena
    // (read deep in disintegration by prefixArgumentsWithU).
    g_currentCoreId = static_cast<int>(coreId);
#if PHASE13_DEEP_TIMING
    const auto recordPhase3Detail =
        [this, coreId](Phase13TimingSlot slot,
                       std::chrono::steady_clock::time_point started) {
            // No rows bound = the defined no-measurement state — same
            // contract as standardProcessing's recordPhase13Detail.
            if (phase13TimingRows == nullptr) return;
            assert(coreId < phase13TimingWorkers
                && "Phase 3 timing row is not bound to this worker");
            phase13TimingRows[
                static_cast<std::size_t>(coreId) * kPhase13TimingSlotCount
                + static_cast<std::size_t>(slot)] +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started).count();
        };
    const auto claimLoadStarted = std::chrono::steady_clock::now();
#endif
    // Per-call phase-3 tracker: feeds the cross-burst aggregate so the
    // post-burst absorb, the discharge machinery, and the end-of-burst
    // sanitize drains are attributed down to their atomic sections.
    RT_TRACKER_DECL(body);
    // Unified working-set handshake (all phases equivalent): claim + load
    // this LB before phase 3 reads it; CE clones run with no steward.
    {
        RT_SCOPE("PH3_CLAIM_LOAD");
        if (steward)
            steward->claimAndLoadForWork(body, /*phase=*/3,
                                         lbdeload::kDeloadDirectory);
    }
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::claimLoad, claimLoadStarted);
#endif

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

#if PHASE13_DEEP_TIMING
    const auto reactToHypothesisStarted = std::chrono::steady_clock::now();
#endif
    { RT_SCOPE_HERE("REACT_TO_HYPO");
	reactToHypo(body);
    } // RT_SCOPE REACT_TO_HYPO
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::reactToHypothesis,
                       reactToHypothesisStarted);
#endif

    { RT_SCOPE_HERE("END_OF_BURST_SANITIZE");
    // End-of-burst sanitization: walk `toBeProved` and rewrite goals
    // whose `it_/int_` args are now downprioritized under the active
    // equi-classes. One pass per burst — equi-class machinery has
    // stabilized by this point. (The hash-memory twin is gone: the rules
    // stay canonical through the install gate and the applyEquiClasses
    // compact hook, D-312.)
#if PHASE13_DEEP_TIMING
    const auto sanitizeToBeProvedStarted = std::chrono::steady_clock::now();
#endif
    {
        RT_SCOPE("SANITIZE_TOBEPROVED");
        this->sanitizeToBeProved(body);
    }
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::sanitizeToBeProved,
                       sanitizeToBeProvedStarted);
    const auto drainDisprovedGoalsStarted = std::chrono::steady_clock::now();
#endif

    // Disproved-goal cleanup: probe the inbox against this LB's MAIN goals
    // and, on a hit, erase the goal plus its integration machinery — the
    // matched scope roots land on pendingWipeScopes so the radical wipe
    // below removes the nested state in this same burst.
    {
        RT_SCOPE("DRAIN_DISPROVED_GOALS");
        this->drainDisprovedGoals(body);
    }
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::drainDisprovedGoals,
                       drainDisprovedGoalsStarted);
    const auto drainDeadOrBranchesStarted = std::chrono::steady_clock::now();
#endif

    // Dead _ordis_ branch retirement: a branch whose asserted disjunct is
    // refuted (staged by ordisMerge's probe) is wiped, its cohort's
    // bookkeeping shrinks, and convergence re-checks at the reduced count.
    // Runs after the disproof drain (wholesale-retired cohorts are gone
    // first) and before the wipe drain below so the branch wipes land in
    // this same burst.
    {
        RT_SCOPE("DRAIN_DEAD_OR_BRANCHES");
        this->drainDeadOrBranches(body);
    }
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::drainDeadOrBranches,
                       drainDeadOrBranchesStarted);
    const auto freezeResolvedOrBranchesStarted =
        std::chrono::steady_clock::now();
#endif

    // Duplicate-or-cohort retirement: when a class merge made two OPEN
    // cohorts at one parent equi variants of each other, keep the most
    // advanced (branch-statement census, lex tie-break) and retire the
    // rest — live branches to pendingWipeScopes, scheduling rows erased,
    // frozen branches and history kept. Runs before the freeze sweep so
    // retired branches are never evaluated, and before the wipe drain so
    // the subtree wipes land in this same burst.
    {
        RT_SCOPE("RETIRE_DUPLICATE_OR_COHORTS");
        this->retireDuplicateOrCohorts(body);
    }

    // Or-branch freeze: every live _ordis_ branch whose chain goals are all
    // known at the branch or above is frozen — its subtree leaves the request
    // universe from the next burst on, nothing is wiped. Runs after the
    // retirement drain (this burst's dead branches are filtered first) and
    // before the release drain.
    {
        RT_SCOPE("FREEZE_RESOLVED_OR_BRANCHES");
        this->freezeResolvedOrBranches(body);
    }
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::freezeResolvedOrBranches,
                       freezeResolvedOrBranchesStarted);
    const auto drainPendingOrReleasesStarted =
        std::chrono::steady_clock::now();
#endif

    // Sequenced or-disintegration: release the next pending branch of every
    // cohort whose live branch resolved this burst (a toBeProved goal
    // reached under the branch, or the branch retired refuted by the drain
    // above — the retirement staging must land first so a refuted live
    // branch releases its successor in this same burst).
    {
        RT_SCOPE("DRAIN_PENDING_OR_RELEASES");
        this->drainPendingOrReleases(body);
    }
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::drainPendingOrReleases,
                       drainPendingOrReleasesStarted);
    const auto wipeSubtreesStarted = std::chrono::steady_clock::now();
#endif

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
        RT_SCOPE("WIPE_SUBTREES");
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
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::wipeSubtrees,
                       wipeSubtreesStarted);
    const auto sweepAncestorKnownRowsStarted =
        std::chrono::steady_clock::now();
#endif

    // Ancestor-known sweep (I-187): drop
    // statement-LIST rows at non-main scopes whose text a strict ancestor
    // knows — the branch-first residue the deposit-time gates cannot see
    // (NameMap has no child index). Runs LAST: after the sanitize twins
    // (rewrites stabilized) and after the dead-branch and wipe drains
    // (retired scopes are gone, so the sweep never touches a scope a wipe
    // just erased). Registry rows survive as the dedup tombstones the
    // gates and re-check paths read.
    {
        RT_SCOPE("SWEEP_ANCESTOR_KNOWN_ROWS");
        this->sweepAncestorKnownRows(body);
    }
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::sweepAncestorKnownRows,
                       sweepAncestorKnownRowsStarted);
#endif
    } // RT_SCOPE END_OF_BURST_SANITIZE

#if PHASE13_DEEP_TIMING
    const auto quiescenceAndDumpsStarted =
        std::chrono::steady_clock::now();
#endif

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

#if MEM_MEASUREMENT
    // End-of-burst memory poll. This is the last point at which the burst's LB
    // is still claimed and resident, and a deloaded container reads empty behind
    // a residency assert (I-111), so an idle LB is unpollable by construction.
    // Writes only this worker's own accumulator row; folded at the barrier.
    gl::mem_tracker::addLbSample(
        body.lbMemory,
        static_cast<unsigned>(coreId));
#endif

    // EXIT trap — delegate to hashburst_dump (relocated here from the inline
    // performElem tail with the phase split; dump unchanged, Rule 14).
    if (hashburst_dump::isTargetLB(body)) {
        hashburst_dump::dumpExit(body);
        // No abort — let the pipeline run to completion. The trap
        // continues firing for every EXIT of this LB; the trace
        // accumulates all of them.
#if RT_MEASUREMENT
        // Per-burst submatch attribution for the hashburst dump's target LB.
        // File-bound like the phase report beside it, never stdout.
        std::filesystem::create_directories(".rt");
        std::ofstream c15Log(".rt/burst_phases.log", std::ios::app);
        c15Log << "[C15-SUB] accepted="
                  << g_crtAccepted.load(std::memory_order_relaxed)
                  << " acceptedUnlinkedDiv="
                  << g_crtAcceptedUnlinked.load(std::memory_order_relaxed)
                  << " attempts="
                  << g_crtAttempts.load(std::memory_order_relaxed)
                  << " attemptsUnlinkedDiv="
                  << g_crtAttemptsUnlinked.load(std::memory_order_relaxed)
                  << "\n";
#endif
    }

    // Release the claim — phase 3 is done with this LB; deloadable again
    // (the steward may now reclaim it). CE clones have no steward.
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::quiescenceAndDumps,
                       quiescenceAndDumpsStarted);
    const auto releaseClaimStarted = std::chrono::steady_clock::now();
#endif
    if (steward) {
        assert(body.stewardClaim.load(std::memory_order_relaxed)
                   == static_cast<uint8_t>(Memory::StewardClaim::WorkerOwned)
               && "phase-3 release of an LB not held WorkerOwned");
        body.stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle),
            std::memory_order_release);
    }
#if PHASE13_DEEP_TIMING
    recordPhase3Detail(Phase13TimingSlot::releaseClaim,
                       releaseClaimStarted);
#endif
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
    if (!this->globalTheoremStrings.insert(theorem).second) {
        // Method-aware upgrade: a first-class registration REPLACES an
        // existing proved-not-broadcast row in place. The tier records a
        // closure that never circulated, so a level-complete (or
        // constructed) derivation of the same theorem supersedes it; every
        // other duplicate is dropped (first emission wins), and a tier
        // arrival never demotes an existing row of any method.
        if (method == "proved not broadcast") return false;
        for (std::size_t i = 0; i < this->globalTheoremList.size(); ++i) {
            auto& tpl = this->globalTheoremList[i];
            if (std::get<0>(tpl) != theorem) continue;
            if (std::get<1>(tpl) != "proved not broadcast") return false;
            std::get<1>(tpl) = method;
            std::get<2>(tpl) = aux2;
            std::get<3>(tpl) = aux3;
            this->globalTheoremProducers[i] = producer;
            // The in-run or-construction scan marks a tier row scanned when
            // it skips it; the upgraded row is first-class and must be
            // scannable as new, so the settled mark leaves with the tier
            // method.
            this->orInRunScannedRows.erase(theorem);
            // The tier registration mirrored its chapter row into
            // fullTheoremList; the upgraded row re-mirrors through the
            // first-class path, so the tier mirror leaves.
            for (std::size_t f = this->fullTheoremList.size(); f-- > 0; ) {
                if (std::get<0>(this->fullTheoremList[f]) == theorem
                    && std::get<1>(this->fullTheoremList[f])
                           == "proved not broadcast") {
                    this->fullTheoremList.erase(this->fullTheoremList.begin()
                        + static_cast<std::ptrdiff_t>(f));
                }
            }
            return true;
        }
        assert(false
            && "appendGlobalTheorem: globalTheoremStrings out of step with "
               "globalTheoremList");
        return false;
    }
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

/// @brief Provenance completion for a tier row's LB chain.
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full
/// contract. Implementation notes: the decoded scan collects the pending
/// deposits first and mutates the map only afterwards (no writes while
/// walking the blob); the residency door is the claim handshake when the
/// steward is live and `ensureLoadedForRead` in the steward-less
/// unit-test context — two defined contexts, matching the existing
/// dual-context sites.
///
/// @return Nothing.
void ExpressionAnalyzer::repairTierCitationOrigins(Memory* producer) {
    assert(producer && "repairTierCitationOrigins: tier registration without a producer LB");
    const int maxOrigins = parameters.compressor_mode
        ? parameters.compressor_max_origins_per_expr
        : parameters.max_origin_per_expr;
    for (Memory* mb = producer; mb != nullptr; mb = mb->parentMemory) {
        if (steward) {
            steward->claimAndLoadForWork(*mb, /*phase=*/4,
                                         lbdeload::kDeloadDirectory);
        } else {
            mb->ensureLoadedForRead(lbdeload::kDeloadDirectory);
        }

        std::set<std::string> pendingDeposits;
        {
            const auto rows = decodeOriginMapSorted(mb->exprOriginMap,
                                                    mb->originInterner);
            for (const auto& row : rows) {
                for (const auto& line : row.second) {
                    for (const ExpressionWithValidity& dep : line.second) {
                        if (dep.validityName != "main") continue;
                        if (!startsWith(dep.original, "(>[", 3)) continue;
                        if (this->globalTheoremStrings.count(dep.original) == 0) continue;
                        int64_t pk = 0;
                        if (lookupOriginKey(mb->originInterner, dep.original,
                                            "main", pk)) {
                            const int32_t oid = mb->exprOriginMap.lookup(pk);
                            if (oid != 0 && mb->exprOriginMap.runLen(oid) > 0) {
                                continue;   // already documented here
                            }
                        }
                        pendingDeposits.insert(dep.original);
                    }
                }
            }
        }
        for (const std::string& thm : pendingDeposits) {
            addOriginEncoded(mb->exprOriginMap, mb->originInterner,
                ExpressionWithValidity(thm, "main"),
                std::make_pair(std::string("theorem"),
                               std::vector<ExpressionWithValidity>()),
                maxOrigins);
        }

        if (steward) {
            mb->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        }
    }
}

void ExpressionAnalyzer::updateGlobalDirect(const std::string& theorem, int coreId,
    const Memory* producer, bool registerGlobally) {
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

    // Level-refused registration (D-278) — proved-but-not-broadcast tier:
    // the goal lifecycle above ran level-free; every CIRCULATION surface
    // below — reformulation, broadcast, compaction staging, mail merge —
    // stays refused when the sealed verdict says the proof did not consume
    // every premise level (the theorem's stronger form is the fact that
    // should circulate). But the closure IS a sound proof, so the theorem
    // is RECORDED: a global-list row with method "proved not broadcast"
    // and its own chapter. It never travels as a rule and never feeds the
    // or construction; its one downstream consumer is the pre-split merge,
    // which may cite it as a guard variant. Excluded from theorems.txt.
    if (!registerGlobally) {
        // Compressor-mode re-derivation audits FIRST-CLASS registration
        // only — its theorem lists must stay byte-identical to the run it
        // audits — so the tier registration is a defined no-op there,
        // exactly like both or-construction seams.
        if (!parameters.compressor_mode) {
            if (this->appendGlobalTheorem(theorem, "proved not broadcast", "-1", "-1", producer)) {
                this->fullTheoremList.emplace_back(theorem, "proved not broadcast", "-1", "-1");
                std::cout << "Proved, not broadcast (level-refused registration): "
                          << theorem << std::endl;
                // The tier chapter walks this LB chain's real derivation;
                // complete any theorem citation whose broadcast history
                // line will never land (the LB dies in this window).
                repairTierCitationOrigins(const_cast<Memory*>(producer));
            }
        }
        return;
    }

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

            // No settled-head deposit at the reformulated chain LB: the
            // compact broadcast below reaches it through root mail, the rule
            // installs, and the head re-derives from the LB's own seed
            // premises with residence-honest levels {0..chain-1}. A direct
            // deposit would have to fabricate a level run — the retired
            // {0..ky.size()} stamp poisoned the receiving LB's level
            // arithmetic (one past its level universe) and broke the
            // allLevelsInvolved verdict of every statement derived from it
            // (D-324).

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
            if (typingRow == nullptr) {
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

            // No settled-head deposit here: the compact broadcast below
            // reaches this LB through root mail, the rule installs, and the
            // head re-derives from the LB's own seed premises with
            // residence-honest levels {0..chain-1}. A direct deposit would
            // have to fabricate a level run — the retired {0..ky.size()}
            // stamp poisoned this LB's level arithmetic (one past its level
            // universe) and broke the allLevelsInvolved verdict of every
            // statement derived from it
            // (D-324).
            // WAKE DOOR 2 (D-194): a main goal was erased above — state the
            // deactivation survey must see next step; keep the LB swept (a
            // dischargedForever LB has no next step, I-102). hasWork is
            // never-deloaded logical state (I-153), so no claim handshake
            // is needed for this flag-only write.
            if (!memoryBlock->dischargedForever) {
                memoryBlock->hasWork = true;
            }

            // FullBind-rebuild expr for every OUTWARD channel. The
            // as-scheduled `expr` is the conjecturer's pre-FullBind shape —
            // its outer binder omits anchor slots unused beyond the anchor
            // premise, leaving them as bare free variables. That form must
            // never circulate: compacting it lifts the unbound anchor slots
            // into template parameters, producing the forbidden
            // (Anchor...[...,u_N]) rule elements (an anchor expression never
            // carries u_ arguments) whose canonical `compilation` citation
            // has no origin row for the chapter walker. Compaction, the
            // `theorem` origin row, and the registry all take the FullBind
            // form; `expr` survives only for or_pairs matching and as
            // reformulateTheorem input (its outputs rebind themselves).
            const std::string exprFullBind =
                this->reconstructImplicationFullBind(ky, value);

            // ASIC 0.1 reshuffle: the legacy Mail::implications channel is
            // removed; this implication travels SOLELY as the D-76 compact
            // (implication<N>[...]) statement, linked to the original by a
            // `compilation` origin (also the receiver's mandatory
            // paired-origin; I-44: exprOriginMap is documentation, not a
            // proof input). Deferred compile+deposit (parallel worker path
            // -> I-28); drained single-threaded post-pool.join.
            recordPendingCompaction(exprFullBind, static_cast<int>(ky.size()), coreId);

            if (parameters.trackHistory) {
                ExpressionWithValidity ev(exprFullBind, "main");
                addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }

            // ---- record in globalTheoremList (short critical section) ----
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

                    // No settled-head deposit at the reformulated chain LB:
                    // the compact broadcast below reaches it through root
                    // mail, the rule installs, and the head re-derives from
                    // the LB's own seed premises with residence-honest levels
                    // {0..chain-1}. A direct deposit would have to fabricate
                    // a level run — the retired {0..ky.size()} stamp poisoned
                    // the receiving LB's level arithmetic (one past its level
                    // universe) and broke the allLevelsInvolved verdict of
                    // every statement derived from it
                    // (D-324).

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
int ExpressionAnalyzer::compareProducerChains(const Memory* a, const Memory* b) {
    if (a == b) return 0;
    while (true) {
        if (a == nullptr) return (b == nullptr) ? 0 : -1;
        if (b == nullptr) return 1;
        if (a == b) {
            // Chains converged on a shared ancestor while every deeper key
            // compared equal — two distinct LBs with a byte-equal full chain
            // would break the full-chain identity every chain-matched
            // consumer (deload naming, debug traps) relies on.
            assert(false && "compareProducerChains: distinct LBs share a "
                            "byte-equal full parentMemory chain");
            return 0;
        }
        const int c = compareSpans(a->exprKeyView(), b->exprKeyView());
        if (c != 0) return c;
        a = a->parentMemory;
        b = b->parentMemory;
    }
}

/// @see Declaration in `prover.hpp` for the full contract.
int ExpressionAnalyzer::updateGlobalDirectLess(const UpdateGlobalDirectRec& a,
                                               const UpdateGlobalDirectRec& b) {
    const int c = compareSpans(StrSpan(a.theorem), StrSpan(b.theorem));
    if (c != 0) return c;
    // Level-verdict tiebreak, TRUE FIRST: when one theorem seals from both a
    // level-complete and a level-poor route in the same iteration, the sink's
    // string dedup makes the FIRST-drained record's method the registration —
    // so the level-complete record must drain first, else the theorem lands
    // in the proved-not-broadcast tier although a full-level proof closed.
    if (a.allLevelsInvolved != b.allLevelsInvolved)
        return a.allLevelsInvolved ? -1 : 1;
    // Producer-chain tiebreak: the deterministic LB identity (leaf-to-root
    // exprKey bytes). coreId must not participate anywhere in this order —
    // it is the phase-dispatch worker slot, a scheduling race outcome, and
    // for a level-poor pair the first record's producer chain becomes the
    // tier chapter's derivation.
    return compareProducerChains(a.producer, b.producer);
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
    // slot). Gather record pointers, index-sort on the (theorem bytes, level
    // verdict true-first, producer chain) total order — every key a pure
    // function of proof state, never the dispatch coreId. The refs / idx ride
    // the gen-scratch tiers; the sealed strings ride updateGlobalDirectPages
    // (never crossed, I-124).
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
        // The sealed level verdict rides along: a false verdict runs only the
        // lifecycle section inside plus the proved-not-broadcast tier
        // registration (D-278).
        updateGlobalDirect(StrSpan(r.theorem).toStdString(), r.coreId,
                           r.producer, r.allLevelsInvolved);

        // Settled-goal deposit: a primed __contradiction__ discharge emits a
        // theorem whose head is negate(seed) (I-165), so hand the seed to the
        // parent's settled-goal inbox. The parent's own end-of-burst
        // drainDisprovedGoals probes BOTH directions against the parent's
        // MAIN goals — seed == goal is a disproof (the verbatim twin fired;
        // goal + machinery wiped), negate(seed) == goal is a proof (the
        // complement twin fired; goal closed with success semantics,
        // D-279); a double miss is an
        // already-closed goal — a defined outcome, so the deposit is
        // unconditional for primed producers. It runs EVEN when the record's
        // level verdict refused registration above: goal settlement is
        // level-free by contract (D-278). Single-threaded post-join seam;
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

/// @brief End-of-burst drain of the settled-goal inbox — close the MAIN goal
///        a contradiction twin settled: a disproved goal loses its whole
///        integration machinery, a proved goal closes with the ordinary
///        success-path semantics.
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full contract
/// (deposit protocol, the two settlement directions, the six disproof cleanup
/// steps, determinism, and the flagged verbatim-match limitation).
/// Implementation notes:
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

    // Shared goal-scope enumeration for both settlement directions below:
    // visit every MAIN-parented scope whose payload embeds `goal` — the
    // `<goal>_subproof_<bare>` shape (split into gOut/bOut), the hypo
    // sentinel, or the `_var…_hypo_<goal>` working scope — and hand each hit
    // to onMatch(id, isSubproof, bOut). Non-minting throughout (I-3).
    const auto forEachGoalScopeRoot = [&](const StrSpan goal, auto&& onMatch) {
        static const char SENT_PFX[] =
            "product_of_hypo_disintegration_of_integration_goal_";
        const int32_t sentLen = static_cast<int32_t>(sizeof(SENT_PFX) - 1);
        static const char HYPO_TOK[] = "_hypo_";
        const int32_t hypoLen = static_cast<int32_t>(sizeof(HYPO_TOK) - 1);
        for (NameId id = 2; id <= body.nameMap.nameCount(); ++id) {
            if (body.nameMap.stackEmpty(id)) continue;
            if (body.nameMap.parentOf(id) != NameMap::MAIN_ID) continue;
            const StrSpan payload =
                body.nameMap.decodeSubView(body.nameMap.stackBack(id));
            StrSpan gOut, bOut;
            if (splitSubproofPayload(payload, gOut, bOut)) {
                if (equalSpans(gOut, goal))
                    onMatch(id, /*isSubproof=*/true, bOut);
            } else if (payload.len == sentLen + goal.len
                       && std::memcmp(payload.ptr, SENT_PFX,
                                      static_cast<size_t>(sentLen)) == 0
                       && equalSpans(StrSpan(payload.ptr + sentLen, goal.len),
                                     goal)) {
                onMatch(id, /*isSubproof=*/false, StrSpan());
            } else if (payload.len > hypoLen + goal.len
                       && payload.ptr[0] == '_'
                       && equalSpans(StrSpan(payload.ptr + payload.len
                                                 - goal.len, goal.len),
                                     goal)
                       && std::memcmp(payload.ptr + payload.len - goal.len
                                          - hypoLen,
                                      HYPO_TOK,
                                      static_cast<size_t>(hypoLen)) == 0) {
                onMatch(id, /*isSubproof=*/false, StrSpan());
            }
        }
    };

    for (int32_t si = 0; si < seedN; ++si) {
        const StrSpan seed = seeds[si];

        // The seed's negation (double-negation cancelling). Both directions
        // read it: for a PROOF settlement it IS the parent goal the twin
        // proved (I-165: theorem head == negate(seed)); for a DISPROOF hit it
        // is the disproof PRODUCT, which the MAIN-keyed origin sweep below
        // must never erase.
        StrSpan negSeed;
        if (seed.len > 0 && seed.ptr[0] == '!') {
            negSeed = StrSpan(seed.ptr + 1, seed.len - 1);
        } else {
            char* nb = sArena.allocBytes(seed.len + 1);
            nb[0] = '!';
            std::memcpy(nb + 1, seed.ptr, static_cast<size_t>(seed.len));
            negSeed = StrSpan(nb, seed.len + 1);
        }

        // ---- PROOF direction (D-279) ----
        // The twin emitted the theorem headed negate(seed); when that head is
        // a registered MAIN goal here, the goal SUCCEEDED — close it with the
        // ordinary success-path semantics: erase the row, stage the goal's
        // scope wipes (wipeSubtree preserve rules — exprOriginMap stays,
        // Rule 16 / I-44), and nothing else. No origin erase, no gate/cohort
        // erase, no closed-subproof exception: those are disproof-only
        // scrubbing of a dead goal's machinery; a proved goal's history and
        // products stay like any proved statement's. Twin retirement (and,
        // when the sealed level verdict allowed it, registration) already
        // happened at the deposit seam (drainUpdateGlobalDirect ->
        // deactivateUnnecessary); a level-refused settlement still closes
        // here (D-278).
        {
            const NameId provedGoalId = body.nameMap.lookup(negSeed);
            if (provedGoalId != 0) {
                const int64_t provedPk =
                    packStatementKey(provedGoalId, NameMap::MAIN_ID);
                if (body.intToBeProved.lookup(provedPk) != 0) {
                    body.intToBeProved.eraseSet(provedPk);
                    forEachGoalScopeRoot(negSeed,
                        [&](NameId id, bool, StrSpan) {
                            body.pendingWipeScopes.mint(id);
                            body.intValidityNamesToFilter.mint(id);
                        });
                }
            }
        }

        // ---- DISPROOF direction ----
        const NameId goalId = body.nameMap.lookup(seed);
        if (goalId == 0) continue;   // seed never a goal here
        const int64_t goalPk = packStatementKey(goalId, NameMap::MAIN_ID);
        if (body.intToBeProved.lookup(goalPk) == 0) continue;  // already closed

        // ---- 1. the goal row ----
        body.intToBeProved.eraseSet(goalPk);

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
        static const char ORINT_PFX[] = "orint_";
        const int32_t orintLen = static_cast<int32_t>(sizeof(ORINT_PFX) - 1);
        forEachGoalScopeRoot(seed,
            [&](NameId id, bool isSubproof, StrSpan bOut) {
                if (isSubproof) {
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
                        if (row != nullptr) return;  // closed — keep
                    }
                }
                body.pendingWipeScopes.mint(id);
                body.intValidityNamesToFilter.mint(id);
                rootNames.push_back(body.nameMap.decodeView(id));
            });

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
        body.orPendingBranches.eraseSetIf([&](int32_t k) {
            return cohortDead(k);
        });
        body.orPendingLevels.eraseSetIf([&](int32_t k) {
            return cohortDead(k);
        });
        // Processed-or ledger rows at wiped scopes leave with the cohort
        // rows: a disproof-cleaned scope can be re-derived and must
        // re-process its ors from scratch (keys are NameMap vids —
        // decode, then the wiped-root text gate like the origin sweeps
        // above).
        body.processedOrLedger.eraseSetIf([&](NameId vid) {
            return underWipedRoot(body.nameMap.decodeView(vid));
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

        // Sequenced release: a retired live branch RESOLVES its cohort — if
        // branches are still pending, stage the cohort for the release drain
        // that runs right after this one (drainPendingOrReleases).
        if (body.orPendingBranches.lookup(cohortId) != 0)
            body.pendingOrReleases.mint(cohortId);

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
                if (row != nullptr) refutScope = a;
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
            // Wholesale retirement also drops the cohort's pending queue —
            // nothing may release into a fully-refuted cohort. (A sequenced
            // cohort reaches zero survivors only after every branch was
            // released, so this is normally a no-op; the guard covers the
            // disproof-drain overlap.)
            body.orPendingBranches.eraseSetIf(
                [&](int32_t k) { return k == cid; });
            body.orPendingLevels.eraseSetIf(
                [&](int32_t k) { return k == cid; });
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


/// @brief Duplicate-or-cohort retirement — see the declaration's Doxygen
///        block in `prover.hpp` for the full contract.
///
/// @details
/// Implementation notes: the ledger snapshot is taken up front because
/// the per-loser `clearProcessedOr` restructures the container
/// mid-drain; grouping is an O(n²) canonical-span compare per scope
/// (n = the recorded ors of one scope — unbounded, so the per-scope
/// working arrays ride the gen-scratch byte tier sized by the actual
/// count); the branch-statement census is one pass over
/// `intEncodedStatements` per duplicate group with a per-open-member
/// prefix probe. Nothing in the drain mints into the NameMap or the
/// `lbStateInterner`, so every held `decodeView` span stays valid to
/// last use (I-3); the wipe itself is delegated to this same burst's
/// `pendingWipeScopes` drain.
///
/// @param body The LB whose processed-or ledger is swept.
void ExpressionAnalyzer::retireDuplicateOrCohorts(Memory& body)
{
    const int32_t scopeN = body.processedOrLedger.count();
    if (scopeN == 0) return;

    const unsigned gSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    const ArenaOffset gMark = gArena.cursor();
    const unsigned sSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(sSlot);

    // Ledger snapshot (gen-scratch page tier): scopes with fewer than two
    // recorded ors can hold no duplicate and are skipped at the source.
    struct LedgerScopeRow { NameId vid; int32_t start; int32_t n; };
    DirtyState snapDirty = DirtyState::Clean;
    PagedVector<int64_t> entryPool(&gArena, &snapDirty);
    PagedVector<LedgerScopeRow> scopeRows(&gArena, &snapDirty);
    for (int32_t id = 1; id <= scopeN; ++id) {
        const int32_t n = body.processedOrLedger.runLen(id);
        if (n < 2) continue;
        const LedgerScopeRow row{ body.processedOrLedger.decodeKey(id),
                                  static_cast<int32_t>(entryPool.size()), n };
        for (int32_t j = 0; j < n; ++j)
            entryPool.push_back(body.processedOrLedger.valueAt(id, j));
        scopeRows.push_back(row);
    }

    const StrSpan BOUNDARY_ORDIS("_boundary_ordis_", 16);

    for (int32_t si = 0; si < static_cast<int32_t>(scopeRows.size()); ++si) {
        const LedgerScopeRow& sr = scopeRows[static_cast<std::size_t>(si)];
        const StrSpan vName = body.nameMap.decodeView(sr.vid);
        // Non-minting cohort recovery: an uninterned parent means no
        // cohort ever opened at this scope — nothing to retire.
        const int32_t parentId = body.lbStateInterner.lookup(vName);
        if (parentId == 0) continue;

        // Per-scope working arrays on the gen-scratch byte tier, sized by
        // the actual recorded-or count — a scope's ledger run is unbounded
        // (an FTA shortcut main records hundreds), so no stack cap. The
        // nested clearProcessedOr allocations below are LIFO above this
        // mark; the per-scope popTo reclaims everything.
        ScratchScope scopeStrScope(sArena);
        const ArenaOffset scopeMark = gArena.cursor();
        const auto allocRun = [&](int32_t bytes, int32_t align) -> char* {
            return gArena.resolve(gArena.alloc(bytes, align));
        };
        StrSpan* text = reinterpret_cast<StrSpan*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(StrSpan)),
            static_cast<int32_t>(alignof(StrSpan))));
        NameId* canonId = reinterpret_cast<NameId*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(NameId)),
            static_cast<int32_t>(alignof(NameId))));
        NameId* entryId = reinterpret_cast<NameId*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(NameId)),
            static_cast<int32_t>(alignof(NameId))));
        int32_t* cohortIdOf = reinterpret_cast<int32_t*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t))));
        int32_t* groupOf = reinterpret_cast<int32_t*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t))));
        int32_t* score = reinterpret_cast<int32_t*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t))));
        for (int32_t j = 0; j < sr.n; ++j) {
            const int64_t e = entryPool[static_cast<std::size_t>(sr.start + j)];
            // The ledger's canonical half is current (re-keyed at every
            // class change at this scope, I-219), so the grouping below is
            // an id comparison — no canonicalization at the drain.
            entryId[j] = ledgerOriginalId(e);
            canonId[j] = ledgerCanonicalId(e);
            text[j] = body.nameMap.decodeView(entryId[j]);
            cohortIdOf[j] = 0;
            const int32_t cid = lookupOrCohortId(body.lbStateInterner,
                parentId, body.lbStateInterner.lookup(text[j]));
            if (cid != 0 && body.orDisjunctCount.lookup(cid) != 0) {
                cohortIdOf[j] = cid;   // OPEN cohort
            }
            groupOf[j] = j;
            for (int32_t k = 0; k < j; ++k) {
                if (canonId[k] == canonId[j]) { groupOf[j] = groupOf[k]; break; }
            }
        }

        for (int32_t g = 0; g < sr.n; ++g) {
            if (groupOf[g] != g) continue;   // one drain per group leader
            int32_t openN = 0;
            for (int32_t j = 0; j < sr.n; ++j)
                if (groupOf[j] == g && cohortIdOf[j] != 0) ++openN;
            if (openN < 2) continue;         // no duplicated OPEN case split

            // Census: statements already registered under each open
            // member's `_ordis_` branch scopes (prefix
            // `<parent>_boundary_ordis_<sig>_(`, which every branch scope
            // and branch descendant extends).
            for (int32_t j = 0; j < sr.n; ++j) score[j] = 0;
            for (int32_t r = 0; r < body.intEncodedStatements.size(); ++r) {
                const StrSpan rowV = body.nameMap.decodeView(
                    body.intEncodedStatements[r].validityId);
                if (rowV.len <= vName.len + BOUNDARY_ORDIS.len) continue;
                if (!equalSpans(StrSpan(rowV.ptr, vName.len), vName)) continue;
                if (!equalSpans(StrSpan(rowV.ptr + vName.len,
                                        BOUNDARY_ORDIS.len),
                                BOUNDARY_ORDIS)) continue;
                const StrSpan tail(rowV.ptr + vName.len + BOUNDARY_ORDIS.len,
                                   rowV.len - vName.len - BOUNDARY_ORDIS.len);
                for (int32_t j = 0; j < sr.n; ++j) {
                    if (groupOf[j] != g || cohortIdOf[j] == 0) continue;
                    const StrSpan sig = text[j];
                    if (tail.len > sig.len + 1
                        && equalSpans(StrSpan(tail.ptr, sig.len), sig)
                        && tail.ptr[sig.len] == '_') {
                        ++score[j];
                    }
                }
            }

            // Keeper: most branch statements; ties to the byte-lex
            // smaller signature (deterministic).
            int32_t keep = -1;
            for (int32_t j = 0; j < sr.n; ++j) {
                if (groupOf[j] != g || cohortIdOf[j] == 0) continue;
                if (keep < 0 || score[j] > score[keep]
                    || (score[j] == score[keep]
                        && compareSpans(text[j], text[keep]) < 0)) {
                    keep = j;
                }
            }
            assert(keep >= 0);

            for (int32_t j = 0; j < sr.n; ++j) {
                if (groupOf[j] != g || cohortIdOf[j] == 0 || j == keep)
                    continue;
                const int32_t loserCid = cohortIdOf[j];
                // Live branches of the loser leave the live registry NOW
                // (the freeze sweep runs after this drain) and their
                // subtrees ride this burst's wipe drain. Frozen branches,
                // orBookkeeping runs and every history line stay.
                const int32_t liveN = body.orLiveBranches.count();
                for (int32_t li = 1; li <= liveN; ++li) {
                    const NameId bvid = body.orLiveBranches.decode(li);
                    StrSpan orSigV, branchBodyV;
                    if (classifyOrScopeView(body.nameMap, bvid,
                                            orSigV, branchBodyV)
                            != OrScopeKind::Disintegration) continue;
                    if (body.nameMap.parentOf(bvid) != sr.vid) continue;
                    if (!equalSpans(orSigV, text[j])) continue;
                    body.pendingWipeScopes.mint(bvid);
                }
                body.orLiveBranches.eraseIf([&](const NameId& v) {
                    return body.pendingWipeScopes.lookup(v) != 0;
                });
                // Scheduling rows leave; a stale pendingOrReleases
                // staging is the release drain's defined skip on the
                // missing count row.
                body.orDisjunctCount.eraseIf(
                    [loserCid](int32_t cid) { return cid == loserCid; });
                body.orPendingBranches.eraseSet(loserCid);
                body.orPendingLevels.eraseSet(loserCid);
                body.orStarterPick.erase(loserCid);
                clearProcessedOr(body, entryId[j], sr.vid);
                body.mutatedThisBurst = true;
            }
        }
        gArena.popTo(scopeMark);
    }
    gArena.popTo(gMark);
}


/// @brief End-of-burst release drain of the sequenced or-disintegration
///        cohorts — mint the next pending branch of every resolved cohort.
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full contract
/// (staging sources, the defined skips, the ranking, the release channel).
/// Implementation notes:
///
/// - Heap-free (Rule 28): the cohort snapshot rides the gen-scratch tier,
///   the branch-validity string the string tier, both reclaimed at exit;
///   the disjunct / clean-form arrays are bounded stack runs
///   (`MAX_INSTRUCTION_ELEMENTS`).
/// - Span lifetimes: every decoded span in the loop aliases
///   `lbStateInterner` cold pages, and nothing in the loop mints into that
///   interner (the seed door mints NameMap, the origin door mints
///   `originInterner`, the queue rebuild inserts KNOWN ids), so the spans
///   stay valid to last use (I-3).
/// - The released seed rides `sameIterationInternalMail` exactly like an
///   `ordisMerge` promotion; the branch scope name itself is minted by the
///   door's `NameMap::encode`, which re-derives boundary parentage
///   (I-139), so no explicit `encodePush` is needed here.
///
/// @param memoryBlock The LB whose release inbox is drained.
/// @invariant Staged cohorts process in decoded (parent, signature) order;
///            the released disjunct is a pure function of the remaining
///            pending set plus the LB's anchor arguments (I-84).
/// @see Memory::pendingOrReleases, Memory::orPendingBranches,
///      Memory::orPendingLevels, pickTopOrDisjunct, collectAnchorArgs.
void ExpressionAnalyzer::freezeResolvedOrBranches(Memory& body) {
    const int32_t liveN = body.orLiveBranches.count();
    if (liveN == 0) return;
    const int32_t goalN = body.intToBeProved.count();
    for (int32_t li = 1; li <= liveN; ++li) {
        const NameId vid = body.orLiveBranches.decode(li);
        // A retired or wipe-closed branch never receives a deposit again:
        // dropped by the compaction below, never evaluated.
        if (body.intValidityNamesToFilter.lookup(vid) != 0) continue;
        assert(body.frozenOrBranches.lookup(vid) == 0
            && "a frozen branch must not be live");
        StrSpan orSigV, branchBodyV;
        const OrScopeKind kind = classifyOrScopeView(body.nameMap, vid,
                                                     orSigV, branchBodyV);
        assert(kind == OrScopeKind::Disintegration
            && "orLiveBranches vid must be an _ordis_ branch scope");
        (void)kind;
        const NameId chainStart = body.nameMap.parentOf(vid);
        assert(chainStart != 0
            && "an _ordis_ branch scope has a cohort parent");
        bool resolved = true;
        for (int32_t gi = 1; gi <= goalN && resolved; ++gi) {
            const StatementKey goal = body.intToBeProved.decodeKey(gi);
            // Chain goal: its scope is the cohort parent or an ancestor of it.
            if (!body.nameMap.ancContains(chainStart, goal.validity)) continue;
            if (!ancestorKnown(body, goal.orig, vid, /*includeSelf=*/true))
                resolved = false;
        }
        if (resolved) body.frozenOrBranches.mint(vid);
    }
    body.orLiveBranches.eraseIf([&](const NameId& v) {
        return body.frozenOrBranches.lookup(v) != 0
            || body.intValidityNamesToFilter.lookup(v) != 0;
    });
}

void ExpressionAnalyzer::drainPendingOrReleases(Memory& body) {
    const int32_t stagedN = body.pendingOrReleases.count();
    if (stagedN == 0) return;
    // No goals, no or branches (I-206): a goal-less LB
    // releases nothing — the staging drops (a goal-less registry never regains
    // a goal, so the drop is final); its live branches freeze vacuously.
    if (body.intToBeProved.empty()) {
        body.pendingOrReleases.resetToFresh();
        return;
    }

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

    // Snapshot + reset (the pendingDeadOrBranches idiom), then sort by the
    // decoded cohort identity — parent bytes, then signature bytes. The pod
    // set dedups and the cohort id is injective over the (parent, orSig)
    // pair, so the sort is tie-free (I-84).
    int32_t* cids = reinterpret_cast<int32_t*>(gArena.resolve(
        gArena.alloc(stagedN * static_cast<int32_t>(sizeof(int32_t)),
                     alignof(int32_t))));
    for (int32_t i = 1; i <= stagedN; ++i)
        cids[i - 1] = body.pendingOrReleases.decode(i);
    body.pendingOrReleases.resetToFresh();
    std::sort(cids, cids + stagedN, [&](int32_t a, int32_t b) {
        const LbStatePairKey ca = decodeOrCohortIds(body.lbStateInterner, a);
        const LbStatePairKey cb = decodeOrCohortIds(body.lbStateInterner, b);
        const int c = compareSpans(
            body.lbStateInterner.decodeView(static_cast<int32_t>(ca.high)),
            body.lbStateInterner.decodeView(static_cast<int32_t>(cb.high)));
        if (c != 0) return c < 0;
        return compareSpans(
            body.lbStateInterner.decodeView(static_cast<int32_t>(ca.low)),
            body.lbStateInterner.decodeView(static_cast<int32_t>(cb.low)))
            < 0;
    });

    for (int32_t ci = 0; ci < stagedN; ++ci) {
        const int32_t cid = cids[ci];
        // Defined skips: the cohort retired wholesale (count row gone — a
        // stale staging), or nothing is pending.
        if (body.orDisjunctCount.lookup(cid) == 0) continue;
        const int32_t pRow = body.orPendingBranches.lookup(cid);
        if (pRow == 0) continue;
        const LbStatePairKey cohort =
            decodeOrCohortIds(body.lbStateInterner, cid);
        const StrSpan parentV = body.lbStateInterner.decodeView(
            static_cast<int32_t>(cohort.high));
        const StrSpan sigV = body.lbStateInterner.decodeView(
            static_cast<int32_t>(cohort.low));
        // A filtered parent scope belongs to a retired region — the seed
        // deposit would be refused there; leave the queue untouched.
        const NameId parentVid = body.nameMap.lookup(parentV);
        assert(parentVid != 0
            && "ordis cohort parent scope must be interned");
        if (body.intValidityNamesToFilter.lookup(parentVid) != 0) continue;

        // Rank the pending disjuncts over their clean forms and pick the
        // release. The wrapped payload bodies come off the queue run in
        // decoded-lex storage order; ids are captured before the rebuild.
        const int32_t pendN = body.orPendingBranches.runLen(pRow);
        assert(pendN >= 1
            && pendN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
            && "ordis pending run exceeds the flattened-leaf cap");
        int32_t djIds[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        StrSpan cleanSpans[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        for (int32_t j = 0; j < pendN; ++j) {
            djIds[j] = body.orPendingBranches.valueAt(pRow, j);
            const StrSpan w = body.lbStateInterner.decodeView(djIds[j]);
            assert(w.len >= 2 && w.ptr[0] == '('
                && w.ptr[w.len - 1] == ')'
                && "pending ordis disjunct must be a wrapped (disjunct)");
            cleanSpans[j] = StrSpan(w.ptr + 1, w.len - 2);
        }
        StrSpan anchorArgs[ExecutionParameters::MAX_ARITY];
        const int32_t anchorArgN = collectAnchorArgs(
            body, anchorArgs, ExecutionParameters::MAX_ARITY);
        int32_t top =
            pickTopOrDisjunct(cleanSpans, pendN, anchorArgs, anchorArgN);
        // Route-(a) starter override, ONE-SHOT: a cohort opened by the
        // demand probe releases its ADMITTED disjunct first — the demand
        // names which branch carries the relevance. The row is erased on
        // consumption (later releases rank as usual); a recorded starter no
        // longer pending (already released or retired) falls back to the
        // ranking — a defined skip, not a failure.
        {
            const int32_t* starterPayload = body.orStarterPick.find(cid);
            if (starterPayload != nullptr) {
                const int32_t starterId = *starterPayload;
                body.orStarterPick.erase(cid);
                for (int32_t j = 0; j < pendN; ++j) {
                    if (djIds[j] == starterId) {
                        top = j;
                        break;
                    }
                }
            }
        }
        const StrSpan wrappedTop = body.lbStateInterner.decodeView(djIds[top]);

        // Branch validity "<parent>_boundary_ordis_<orSig>_(<disjunct>)" on
        // the string tier, explicit-length.
        ScratchScope relScope(sArena);
        const StrSpan BOUNDARY_ORDIS("_boundary_ordis_", 16);
        const int32_t bvLen = parentV.len + BOUNDARY_ORDIS.len + sigV.len
            + 1 + wrappedTop.len;
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
            std::memcpy(bvBuf + at, wrappedTop.ptr,
                static_cast<std::size_t>(wrappedTop.len));
            at += wrappedTop.len;
            assert(at == bvLen);
        }
        const StrSpan branchV(bvBuf, bvLen);

        // The cohort's seed level run (stored at cohort mint time).
        int lvRun[256];
        int32_t lvN = 0;
        const int32_t lRow = body.orPendingLevels.lookup(cid);
        if (lRow != 0) {
            const int32_t ln = body.orPendingLevels.runLen(lRow);
            assert(ln <= 256 && "ordis seed level run exceeds 256");
            for (int32_t j = 0; j < ln; ++j)
                lvRun[j] = body.orPendingLevels.valueAt(lRow, j);
            lvN = ln;
        }

        // Quiescence (D-194): the release creates work for the next absorb.
        body.mutatedThisBurst = true;
        insertInternalStatement(body.sameIterationInternalMail, body.nameMap,
            cleanSpans[top], branchV, lvRun, lvN);
        if (parameters.trackHistory) {
            const OriginDep relDeps[1] = { { sigV, parentV } };
            addInternalMailOrigin(body.sameIterationInternalMail,
                body.originInterner, cleanSpans[top], branchV,
                OriginTag::orDisintegration, relDeps, 1,
                (parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr));
        }

        // The released disjunct leaves the queue: rebuild the run without it
        // through the canonical doors. An emptied queue drops the cohort's
        // level row too.
        body.orPendingBranches.eraseSetIf(
            [&](int32_t k) { return k == cid; });
        for (int32_t j = 0; j < pendN; ++j) {
            if (j == top) continue;
            body.orPendingBranches.insertSorted(cid, djIds[j],
                DecodedIdLess{ &body.lbStateInterner });
        }
        if (pendN == 1) {
            body.orPendingLevels.eraseSetIf(
                [&](int32_t k) { return k == cid; });
        }
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
/// An expression whose `int_lev` witness tokens all carry a level strictly
/// below the LB's own `level` passes unconditionally (parent-level pass):
/// those names arrived by ancestor mail and are already known at every
/// descendant. Own-level tokens qualify only through the memo filters.
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

    // Parent-level pass: an expression whose int_lev tokens ALL carry a level
    // strictly below this LB's own level is mailable unconditionally — such
    // names can only have arrived by ancestor mail (mail flows down only,
    // I-57), and every ancestor deposit is pulled by ALL of this LB's
    // descendants directly, so the names are known below by construction.
    // Implication-shaped text is still refused (rules travel only as
    // compacts, I-54). Own-level tokens fall through to the memo gates.
    if (!(expression.len >= 2 && expression.ptr[0] == '(' && expression.ptr[1] == '>')
        && allIntLevLevelsBelow(expression, body.level)) {
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
    // Callers arrive through the addStatement door, which pins the level-run
    // invariant: non-empty, and either exactly {-1} or all-values-≥0.
    assert(levelCount > 0
        && ((levelCount == 1 && levels[0] == -1) || levels[0] >= 0)
        && "addEquality: invalid statement level run");

    const StatementFlags* eqRow = lookupStatementFlags(
        memoryBlock.intKnownStatements, memoryBlock.nameMap, expr, validityName);
    if (eqRow == nullptr)
    {
        // 1. Register the original equality
        const int64_t pkEq = packStatementKey(
            memoryBlock.nameMap.encode(expr),
            memoryBlock.nameMap.encode(validityName));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkEq, levels, levels + levelCount);
        upsertStatementKey(memoryBlock.intKnownStatements, pkEq, local);

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
        assert(mirrorRow == nullptr);
        (void)mirrorRow;

        const int64_t pkEqM = packStatementKey(
            memoryBlock.nameMap.encode(mirrored),
            memoryBlock.nameMap.encode(validitySpan));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkEqM, levels, levels + levelCount);
        upsertStatementKey(memoryBlock.intKnownStatements, pkEqM, local);

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
    // Callers arrive through the addStatement door, which pins the level-run
    // invariant: non-empty, and either exactly {-1} or all-values-≥0.
    assert(levelCount > 0
        && ((levelCount == 1 && levels[0] == -1) || levels[0] >= 0)
        && "addNegatedEquality: invalid statement level run");

    const StatementFlags* negRow = lookupStatementFlags(
        memoryBlock.intKnownStatements, memoryBlock.nameMap, expr, validityName);
    if (negRow == nullptr)
    {
        // 1. Register the original negated equality
        const int64_t pkNeg = packStatementKey(
            memoryBlock.nameMap.encode(expr),
            memoryBlock.nameMap.encode(validityName));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkNeg, levels, levels + levelCount);
        upsertStatementKey(memoryBlock.intKnownStatements, pkNeg, local);

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
        assert(negMirrorRow == nullptr);
        (void)negMirrorRow;

        const int64_t pkNegM = packStatementKey(
            memoryBlock.nameMap.encode(mirrored),
            memoryBlock.nameMap.encode(validitySpan));
        memoryBlock.intStatementLevelsMap.assignSetRange(
            pkNegM, levels, levels + levelCount);
        upsertStatementKey(memoryBlock.intKnownStatements, pkNegM, local);

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

    // Atomic RT attribution: self-time = the whole-registry chain scan
    // (decode + head match); the mint-heavy premise handling below opens
    // its own nested rows.
    RT_SCOPE_HERE("CNFE");

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
    int32_t chainN = 0;
    {
        RT_SCOPE_HERE("CNFE_SORT_CHAINS");
        chainN = sortOriginalChainIndex(mb, chainIdx, chainCount);
    }
    RT_NOTE_ITERATIONS_HERE(chainN);
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

        // Head-first gate. Every check up to the match verdict is a pure read
        // (no mint), so the head -- the last element of the implication chain
        // (D10) -- is inspected through a raw decodeView span and the chain is
        // copied onto sArena only once it has matched (below). Almost every
        // chain leaves at the name compare; for it nothing beyond the head is
        // decoded and nothing is copied.
        const StrSpan headView = mb.ruleInterner.decodeView(chainIds[count - 1]);
        const StrSpan headName = extractExpressionSpan(headView);

        // Structural check: Name and Arity must match the input expression
        if (!equalSpans(headName, inputName)) continue;
        StrSpan headArgs[ExecutionParameters::MAX_ARITY];
        const int32_t headArgN = getArgsSpans(headView, headArgs,
                                              ExecutionParameters::MAX_ARITY);
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

        // Matched: copy each chain element onto sArena BEFORE the premise loop
        // (which mints ruleInterner via prepareIntegration -> addToHashMemory,
        // so the raw decodeView spans above would dangle -- I-3). Held under
        // cixScope. The head's argument spans are re-derived from the stable
        // copy because headArgs[constantIndex] feeds the replacement pair used
        // across those mints.
        StrSpan chainSpans[64];
        for (int32_t i = 0; i < count; ++i) {
            const StrSpan dv = mb.ruleInterner.decodeView(chainIds[i]);
            chainSpans[i] = StrSpan(ScratchString::copyFrom(sArena, dv.ptr, dv.len));
        }
        const StrSpan headStr = chainSpans[count - 1];
        const int32_t headArgNCopy = getArgsSpans(headStr, headArgs,
                                                  ExecutionParameters::MAX_ARITY);
        assert(headArgNCopy == headArgN
               && "checkNecessityForEquality: head copy parses differently");
        (void)headArgNCopy;

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

            {
                RT_SCOPE_HERE("CNFE_PREPARE_INTEGRATION");
                prepareIntegration(StrSpan(removed), argsSet, argsSetN, mb, StrSpan(validityName), inputExprStr);
            }

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
        if (cfeVarRow && cfeVarRow->fullyDisintegrated) {
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
    bool allowOrDisintegration,
    IntEncodedExpr* registeredOut) {

    // Two-route or-cohort opening (admission-based ordis; restores the D-32
    // gate READ the sequenced branch had overridden): route (b) is the
    // threaded product-of-disintegration signal (the FiringRecord /
    // disintegration-signal plumbing fed from D-32's
    // productOfDisintegration stamp) — such heads open unconditionally, as
    // in rungs 1+2; route (a) is the demand probe at the or arm, enabled by
    // allowOrProbe for every real deposit through this door. The
    // rule-intrinsic doNotDisintegrate (D-241 intro-fired suppression,
    // integration-justified rules) and the D-29 firing-context clauses block
    // both routes. The hypothetical-disintegration path calls
    // disintegrateExpr2 directly (bypassing this door) with both defaults
    // false, so no cohort ever mints — and no park ever writes — at a
    // sentinel scope.
    allowOrDisintegration = allowOrDisintegration && !doNotDisintegrate;
    const bool allowOrProbe = !doNotDisintegrate;

    // Atomic RT attribution: the door's own row carries the gate / glue
    // self-time; every heavyweight sub-call below opens its own nested row.
    RT_SCOPE_HERE("DOOR_ADD_EXPR");

    // --- Static (int-based) early checks ---
    // Fast path: duplicate + validity filter with just nm.encode (no regex, no vector alloc)
    NameId origId = 0;
    NameId valId = 0;
    {
        RT_SCOPE_HERE("DOOR_ENTRY_GATES");
        origId = memoryBlock.nameMap.encode(expr);
        valId = memoryBlock.nameMap.encode(validityName);

        // Site F — dup suppression via ancestor scan. A known statement at any
        // ancestor scope (including self) means the same fact already holds at a
        // strictly weaker set of assumptions, so the child-scope insertion is
        // redundant. ancestorsOf[valId] includes valId itself and every strict
        // prefix scope registered via encodePush. For a flat root (e.g. legacy
        // integration cleanSignature-as-validity) the list is just {valId},
        // degenerating to the old strict-equality check — safe under the
        // MAIN_ID transition guard. The scan is the shared `ancestorKnown`
        // predicate (one definition of the contract across every door); a
        // refusal still carries the or-branch resolution signal (I-174) — the
        // branch derived the expression even though the deposit is redundant.
        if (!parameters.compressor_mode
            && ancestorKnown(memoryBlock, origId, valId, /*includeSelf=*/true)) {
            stageOrReleaseForRefusedDeposit(memoryBlock, origId, valId);
            return;
        }

        // Site H — ancestor-scan the int validity blacklist. A blacklist entry at
        // any ancestor scope filters every descendant scope (deeper scopes inherit
        // the filter because they carry strictly more assumptions).
        for (int32_t ancK = 0, ancN = memoryBlock.nameMap.ancLen(valId);
             ancK < ancN; ++ancK) {
            const NameId anc = memoryBlock.nameMap.ancAt(valId, ancK);
            if (memoryBlock.intValidityNamesToFilter.contains(anc)) return;
        }
    }

    // The canonical door: only the class-canonical form of a deposit is
    // admitted. For every status except a goal (2) and every shape except
    // the two equality shapes (goal closure is literal and neither shape is
    // ever multiplied) and anchors (a scope identity, I-53), the text is
    // rewritten under the equivalence classes at its OWN scope
    // (`canonicalFormAtScope`: a fresh scope's bucket is empty, so seeds
    // keep the spelling their scope name / identity was minted from). A
    // changed deposit: (1) keeps the RAW text's producer history line
    // (Rule 16 — the canonical form's bridge below terminates on it) and,
    // for a local status, ships that line to mailOut so a receiver's
    // chapter walk terminates on it too (the D-286 ship — the raw form
    // never becomes a delta row, so fillMailOut would never carry it);
    // (2) ends the deposit when the canonical form is already known at
    // this scope or a strict ancestor — the same Site-F refusal as for the
    // raw text, releasing the or-branch resolution signal for BOTH ids
    // (I-174); (3) otherwise proceeds with the canonical text, its origin
    // replaced by ONE `equality1` bridge (deps[0] = raw form at this scope,
    // then each applied `(=[member,canonical])` cited where its row lives)
    // when the canonical form has no history row yet (the I-34 gate), and
    // its level run = the deposit's levels ∪ the applied pairs' class
    // levels. `applyEquiClasses` still multiplies the registered canonical
    // form into its orbit, so a literal non-equality goal closes from the
    // product in the same standardProcessing call. The canonical text and
    // every span the rest of the door reads from it ride the string-tier
    // arena under this function-level scope; every nested scope the
    // deposit paths open rewinds above it (LIFO).
    static constexpr int32_t kDoorLevelCap = 256;
    assert(involvedLevelCount <= kDoorLevelCap
        && "addExprToMemoryBlock: level run exceeds kDoorLevelCap");
    int doorLevels[kDoorLevelCap];
    int32_t doorLevelN = involvedLevelCount;
    for (int32_t li = 0; li < involvedLevelCount; ++li) doorLevels[li] = involvedLevels[li];
    OriginDep bridgeDeps[1 + ExecutionParameters::MAX_ARITY];
    TransientOrigin doorOrigin = origin;
    // A deposit the door REWRITES under an equality of this LB's own is this
    // LB's class product — an equality1 rewrite performed here — so it flips
    // to a LOCAL derivation whatever the arrival status: the canonical form
    // registers local (visible to the main-goal discharge's delta and to
    // fillMailOut) and the raw form's producer line ships to mailOut with it,
    // exactly the history an applyEquivalenceClass product leaves behind. A
    // rewrite licensed only by mailed equalities stays the sender's
    // knowledge and keeps the arrival's locality. A mailed fact that
    // canonicalizes onto a main goal's literal text under a local hypothesis
    // would otherwise sit in the registry as the sender's non-local row,
    // never discharge the goal, and Site-F-refuse the local derivation of
    // the same text.
    bool rewriteIsLocal = false;
    const unsigned doorStrSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& doorStrArena = scratchArenas().forSlot(doorStrSlot);
    ScratchScope doorStrScope(doorStrArena);
    if (status != 2 && !parameters.skip_eq_classes
        && !isEquality(expr) && !isNegatedEquality(expr)) {
        RT_SCOPE_HERE("DOOR_CANONICALIZE");
        const CanonicalForm cf = canonicalFormAtScope(
            expr, validityName, memoryBlock, doorStrArena,
            doorLevels, doorLevelN, kDoorLevelCap);
        if (cf.changed) {
            for (int32_t k = 0; k < cf.eqN; ++k) {
                if (cf.eqLocal[k]) rewriteIsLocal = true;
            }
            // Known canonical form -> the deposit ends here: no row, no
            // history line for the new raw spelling (nothing will cite
            // it), both ids carry the or-branch resolution signal.
            const NameId canonId = memoryBlock.nameMap.lookup(cf.text);
            if (!parameters.compressor_mode && canonId != 0
                && ancestorKnown(memoryBlock, canonId, valId, /*includeSelf=*/true)) {
                stageOrReleaseForRefusedDeposit(memoryBlock, origId, valId);
                stageOrReleaseForRefusedDeposit(memoryBlock, canonId, valId);
                return;
            }
            const int originCap = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr;
            if (parameters.trackHistory) {
                // A rewritten deposit without a producer line would leave
                // the bridge citing a form no chapter can resolve
                // (buildStack: no origin found) — surface the site here.
                assert(origin.present
                    && "addExprToMemoryBlock: canonical door rewrote a deposit that carries no origin line");
                assert(origin.tag != OriginTag::COUNT);
                addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner,
                    expr, validityName, origin.tag, origin.deps, origin.depN, originCap);
                // A canonical form that registers local mails; the raw form
                // never becomes a delta row, so its line ships from here (the
                // D-286 ship) whenever the registered row will be local.
                if (status == 0 || status == 1 || rewriteIsLocal) {
                    copyOriginRowsToMailOut(memoryBlock, expr, validityName, originCap);
                }
                if (!originRowExists(memoryBlock, cf.text, validityName)) {
                    const int bridgeN = 1 + cf.eqN;
                    assert(bridgeN <= 1 + ExecutionParameters::MAX_ARITY
                        && "addExprToMemoryBlock: bridge dep run exceeds cap");
                    bridgeDeps[0] = OriginDep{ expr, validityName };
                    for (int32_t bk = 0; bk < cf.eqN; ++bk) {
                        // The cite scope may be a NameMap decode; the mints
                        // below (canonical text, its argument ids) precede
                        // the bridge write, so copy it onto the string tier
                        // (I-3).
                        const StrSpan cite = findEqualityCiteScope(
                            memoryBlock, cf.eqJust[bk], validityName);
                        const ScratchString citeHold =
                            ScratchString::copyFrom(doorStrArena, cite.ptr, cite.len);
                        bridgeDeps[1 + bk] = OriginDep{ cf.eqJust[bk], StrSpan(citeHold) };
                    }
                    doorOrigin = TransientOrigin{
                        true, OriginTag::equality1, bridgeDeps, bridgeN };
                } else {
                    // Canonical target already documented — no second line
                    // (the I-34 gate); the door registers with no origin write.
                    doorOrigin = TransientOrigin{};
                }
            }
            expr = cf.text;
            origId = memoryBlock.nameMap.encode(expr);
        }
    }
    involvedLevels = doorLevels;
    involvedLevelCount = doorLevelN;

    // Passed fast checks — now do full encoding for axed var check + downstream.
    // Span twin: parse + encode in one pass off the stable expr/validityName
    // buffers, no intermediate EncodedExpression heap.
    IntEncodedExpr ie =
        encodeExpression(expr, validityName, memoryBlock.nameMap);
    if (registeredOut != nullptr) *registeredOut = ie;

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
        // A levels row is never empty and never mixes the non-derived tier
        // {-1} with real levels (the addStatement door rule; status 4
        // bypasses that door, so the invariant is pinned here too).
        assert(involvedLevelCount > 0
            && ((involvedLevelCount == 1 && involvedLevels[0] == -1)
                || involvedLevels[0] >= 0)
            && "status-4 fact load: invalid statement level run");
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
            /*local=*/true);

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

            {
                RT_SCOPE_HERE("DOOR_GOAL_CHECK_NECESSITY");
                checkNecessityForEquality(expr, memoryBlock, validityName);
            }
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
        {
            RT_SCOPE_HERE("DOOR_PREPARE_INTEGRATION");
            prepareIntegration(expr, unchRun, unchN, memoryBlock, validityName, expr);
        }
        // `allowedForMail` probes the memo only for `int_lev_*`-carrying
        // expressions (its lexical scanSingleDistinctIntLev gate short-circuits
        // everything else), so any other entry could never be read.
        if (containsSpan(expr, StrSpan("int_lev_", 8))) {
            memoryBlock.canBeSentIds.mint(memoryBlock.nameMap.encode(expr));
        }

        const ce::CoreExpressionConfig* cfg = coreConfig(extractExpressionSpan(expr));

        if (cfg != nullptr && !cfg->inputIndices.empty()) {
            RT_SCOPE_HERE("DOOR_UPDATE_ADMISSION3");
            this->updateAdmissionMap3(expr,
                memoryBlock,
                parameters.inductionMaxAdmissionDepth,
                parameters.inductionMaxSecondaryNumber,
                true);
        }
        return;
    }
    else {
        if (parameters.trackHistory && doorOrigin.present) {
            RT_SCOPE_HERE("DOOR_ORIGIN_WRITE");
            assert(doorOrigin.tag != OriginTag::COUNT);
            addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner, expr, validityName, doorOrigin.tag, doorOrigin.deps, doorOrigin.depN, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        // Statuses 0/1/3/5 reach here (status 2 returned above, status 4
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
        // non-convergence runaway. Status 5 (flag-5 relay arrival,
        // D-284) IS admitted with witness minting: the
        // sender's disintegration parked this compact's products for lack
        // of admission demand, and this receiver may hold the demand. The
        // mint volume stays bounded — only parked compacts are relayed,
        // products remain demand-gated (Pass B parks undemanded ones
        // here exactly like local derivations), and a status-5
        // disintegration never re-stages a relay (no cascade). Without the negated-existence
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
            const bool isCompactImplication = isCompactImplicationSpan(expr);

            // Negated compact existence = literal "!(existence" followed by
            // a digit — the second rule-carrier shape admitted at status 3.
            const bool isNegatedExistence =
                expr.len > 11 && startsWithSpan(expr, "!(existence", 11)
                && expr.ptr[11] >= '0' && expr.ptr[11] <= '9';

            // Or-uniqueness gate: an or<N> deposit whose equi class already
            // has a fully-processed representative at this scope stays a
            // PASSIVE statement — registered below and multiplied by
            // applyEquiClasses like any statement, but never disintegrated
            // (no K/subset-exclusion compacts, no cohort, no ordis park),
            // on every entry path (local and the flag-5 relay alike).
            // Ordered before checkForEquivalence so a suppressed or skips
            // the Cartesian probe.
            const bool orCompactDeposit = isOrCompactSpan(expr);
            bool orEquiDuplicate = false;
            if (orCompactDeposit && !doNotDisintegrate) {
                RT_SCOPE_HERE("DOOR_OR_EQUI_GATE");
                orEquiDuplicate =
                    orEquiRepresentativeRecorded(memoryBlock, expr, validityName);
            }
            // Suppress disintegration when an equivalence-class variant of
            // this expression is already a registered statement.
            bool cfeEq = false;
            if (!doNotDisintegrate && !orEquiDuplicate) {
                RT_SCOPE_HERE("DOOR_CHECK_EQUIVALENCE");
                cfeEq = checkForEquivalence(expr, validityName, memoryBlock);
            }
            const bool willDisintegrate =
                !doNotDisintegrate && !orEquiDuplicate && !cfeEq
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
                // disintegrateExpr2 under a local status (0/1) or the
                // flag-5 relay status (5). status=3 is admissible here ONLY
                // for the two rule-carrier shapes — compact implication or
                // negated compact existence (the guard enforces it);
                // anything else reaching here with status=3 fires this
                // assert and pins the forbidden path.
                assert(isCompactImplication || isNegatedExistence
                    || status == 0 || status == 1 || status == 5);
                {
                    RT_SCOPE_HERE("DOOR_DISINTEGRATE");
                    fullDisintegrationHappened = this->disintegrateExpr2(expr,
                            memoryBlock,
                            iteration,
                            status == 0,
                            validityName,
                            out,
                            /*trackHistoryLocal=*/true,
                            allowOrDisintegration,
                            involvedLevels, involvedLevelCount,
                            allowOrProbe);
                }
                // First representative: record the or's full processing so
                // later equi variants at this scope stay passive (the
                // or-uniqueness gate above).
                if (orCompactDeposit) {
                    recordProcessedOr(memoryBlock, origId, valId);
                }
                // Uniform admission — no not-self-returned exemption: a fully
                // disintegrated deposit (e.g. a broadcast theorem compact
                // decomposed into its rule) is OFFERED to the one admission
                // door like any statement. The channel dedups on the
                // composite key, so a self-returned expr makes this a no-op;
                // a not-self-returned expr joins the stmts loop below and is
                // admitted (row + levels) or refused by the door's gates —
                // never row-stamped outside admission. A PARTIALLY
                // disintegrated deposit is deliberately NOT offered: it must
                // stay re-processable (a row would Site-F-block the retry of
                // its unwitnessed existences).
                if (fullDisintegrationHappened) {
                    out.statements.append(expr, validityName);
                }
                // Flag-5 relay staging (D-284): the highest
                // uncovered existence groups' compacts await a flag-5 mail
                // deposit so a demand-holding descendant can disintegrate
                // them. LOCAL deposits only — a mailed carrier (status 3/5)
                // never re-stages (delivery already reached all descendants);
                // main scope only (I-26). The NameMap mint is single-threaded
                // per LB (worker claim / post-join drain).
                if ((status == 0 || status == 1)
                    && equalSpans(validityName, StrSpan("main", 4))) {
                    out.relayCompacts.forEachSorted(
                        [&](StrSpan rc, StrSpan rcValidity) {
                            (void)rcValidity;
                            memoryBlock.stagePendingRelay(
                                memoryBlock.nameMap.encode(rc), iteration,
                                involvedLevels, involvedLevelCount);
                        });
                }
            }
            else
            {
                out.statements.append(expr, validityName);
            }

            {
            RT_SCOPE_HERE("DOOR_IMPLICATIONS_WALK");
            out.implications.forEachSortedWithSource([&](StrSpan impStr, StrSpan impValidity,
                                                         StrSpan carrier)
            {
                RT_SCOPE_HERE("DOOR_INSTALL_IMPLICATION");
                // The per-implication install — the canonical gate, the
                // status-selected instances, the D-274 row, both indexes —
                // lives in installExpandedImplication. The mail-out contract
                // stays MAIN-ONLY (I-26): a disintegration-recovered
                // implication is neither re-broadcast nor re-compacted here
                // (a locally derived one carries its compact form in
                // `expressions`, a mail-recovered one IS the statement being
                // disintegrated; recordPendingCompaction here would fan out).
                this->installExpandedImplication(memoryBlock, impStr, impValidity, carrier,
                    status, iteration, involvedLevels, involvedLevelCount,
                    expr, validityName);
            });
            } // RT_SCOPE DOOR_IMPLICATIONS_WALK

            if (out.implications.count() > 0)
            {
                RT_SCOPE_HERE("DOOR_GOALS_CHECK_NECESSITY");
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
            RT_SCOPE_HERE("DOOR_STATEMENTS_WALK");
            out.statements.forEachSorted([&](StrSpan evOrig, StrSpan evValidity)
            {
                RT_SCOPE_HERE("DOOR_STATEMENT_ROW");
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
                    // Locality: local statuses register everything local.
                    // A flag-5 relay arrival (status 5) splits per statement
                    // (D-284): the CARRIER itself registers
                    // non-local — only local delta rows reach fillMailOut, so
                    // it is structurally never re-forwarded (delivery already
                    // reached every descendant) — while its disintegration
                    // products are THIS LB's own derivations (witnesses
                    // minted here) and register local, mailable and visible
                    // to the local-delta request batches.
                    const bool isCarrier = (status == 5)
                        && equalSpans(evOrig, expr)
                        && equalSpans(evValidity, validityName);
                    const bool isLocal = (status == 0 || status == 1)
                        || (status == 5 && !isCarrier)
                        || rewriteIsLocal;

                    // The single-token carriers among a status-5
                    // disintegration's products memoize as sendable
                    // (canBeSentIds) exactly like the force-deep local
                    // path's products — "products mail normally". Sorted
                    // walk order (forEachSorted) keeps the mint order
                    // deterministic (I-105). The carrier is excluded.
                    if (status == 5 && !isCarrier
                        && containsSpan(evOrig, StrSpan("int_lev_", 8))) {
                        memoryBlock.canBeSentIds.mint(
                            memoryBlock.nameMap.encode(evOrig));
                    }

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
                    {
                        RT_SCOPE_HERE("DOOR_ADD_STATEMENT");
                        this->addStatement(evOrig, memoryBlock, isLocal,
                            involvedLevels, involvedLevelCount, frontTO,
                            evValidity, added);
                    }

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
                        {
                            RT_SCOPE_HERE("DOOR_ADMISSION_INTEGRATION");
                            updateAdmissionMapIntegration(StrSpan(addExpression), memoryBlock,
                                                          StrSpan(effectiveValidity));
                        }
                        {
                            RT_SCOPE_HERE("DOOR_ADMISSION_RECURSION");
                            updateAdmissionMapRecursion(StrSpan(addExpression), memoryBlock,
                                                        StrSpan(effectiveValidity));
                        }

                        const int32_t addLvlsId = lookupStatementLevels(
                            memoryBlock.intStatementLevelsMap, memoryBlock.nameMap,
                            StrSpan(addExpression), StrSpan(effectiveValidity));
                        assert(addLvlsId
                               && memoryBlock.intStatementLevelsMap.runLen(addLvlsId) > 0
                               && "addStatement post-loop intStatementLevelsMap invariant violated");
                        int lvRun[256];
                        const int32_t lvN = coldIntRunAt(
                            memoryBlock.intStatementLevelsMap, addLvlsId,
                            lvRun, 256);

                        {
                            RT_SCOPE_HERE("DOOR_ORDIS_MERGE");
                            ordisMerge(StrSpan(addExpression), StrSpan(effectiveValidity),
                                       lvRun, lvN, memoryBlock);
                        }
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
            // OR the `fullyDisintegrated` bit onto expr's registry row so cFE
            // suppresses future equivalence-class variants of expr (a
            // rejecting / partial twin stays unflagged and never suppresses —
            // the Gauss-fold mirror-bug fix). Bookkeeping on top of admission,
            // nothing more: the row exists iff the admission door created one
            // (expr rode the stmts loop above via the uniform-admission
            // offer); the marker never creates a row and never grants `known`
            // — a statement the door refused stays exactly as the door left
            // it.
            if (fullDisintegrationHappened) {
                const NameId markOrigId = memoryBlock.nameMap.encode(expr);
                const NameId markValId = memoryBlock.nameMap.encode(validityName);
                if (memoryBlock.intKnownStatements.find(
                        StatementKey{ markOrigId, markValId }) != nullptr) {
                    upsertStatementKey(memoryBlock.intKnownStatements,
                        packStatementKey(markOrigId, markValId),
                        /*local=*/false, /*fullyDisintegrated=*/true);
                }
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

/// @brief Detect the output-collision conjecture shape and name the
///        variable to duplicate — the fifth variable-copy trigger.
///
/// @details
/// A conjecture states an equality between two compound terms by fusing
/// their result binders into one name: two premises of the same operator
/// carry the same name in the operator's output slot while their input
/// slots differ (`(in3[a,c,r,·])` / `(in3[b,c,r,·])` encodes
/// `a·c = b·c`). A pool rule written with two distinct result variables
/// for such a premise pair can then never match — request matching only
/// renames, it never binds two rule variables to one statement name — so
/// the rule's firings, demand markers, and integration seeds all starve.
/// The cure is the same dead-end variable-copy axiom `(=[r,r_copy])` the
/// antisymmetry trigger deposits: the equivalence class generates the
/// all-distinct statement variants and the standard machinery closes the
/// proof.
///
/// This function is the pure detection half: it scans every ordered pair
/// of chain elements for the shape and, on the first match, reports the
/// shared output name and the chain index of the SECOND colliding
/// premise — the LB the caller deposits at. The first colliding premise
/// is typically an anchor-adjacent chain prefix shared by foreign
/// conjectures, and mail flows to every descendant, so a deposit there
/// leaks the copy into subtrees that never asked for it; every genuine
/// consumer (deeper premise LBs, the contradiction twins under the
/// innermost premise, recursion auxiliaries) sits at or below the second
/// colliding premise. Both verdicts are defined results of the detection
/// contract.
///
/// Guards: both premises positive (no `!` prefix); identical operator
/// name and arity, with a compiled core config carrying exactly one
/// output slot; the output argument equal in both and a plain atom (no
/// parentheses — the shallow `ce::getArgs` parse is exact only for flat
/// argument lists); every argument that is neither an input slot nor the
/// output slot byte-equal between the two premises; at least one input
/// slot differing; no third positive premise matching the same
/// (operator, non-input/output arguments, output name) signature — the
/// gate is exactly-two by design, extendable later; and the shared name
/// is never an ANCHOR argument (the I-24 analogue: an anchor-slot name
/// is a theory constant — e.g. zero in `s(a)=0 ∧ s(b)=0 → a=b` — and
/// duplicating a constant variant-fans every statement carrying it and
/// breaches the one-changeable-arg integration contract). Theorem chains
/// never carry `u_`-prefixed arguments; the shared name is asserted plain.
///
/// @param chain         The disintegrated premise chain of the
///                      conjecture (raw element strings).
/// @param coreMap       Compiled operator configurations — supplies each
///                      operator's input/output slot classification.
/// @param copyVarOut    On detection, the shared output name to
///                      duplicate; untouched otherwise.
/// @param firstIndexOut On detection, the chain index of the second
///                      colliding premise — the deposit LB; untouched
///                      otherwise.
/// @return True iff the output-collision shape was detected.
/// @see gl::detectAntisymmetryCopyVar — the sibling trigger whose
///      deposit contract (name shape, origin tag, mailOut pairing) this
///      trigger shares; ExpressionAnalyzer::addTheoremToMemory — the
///      deposit site.
bool detectOutputCollisionCopyVar(const std::vector<std::string>& chain,
                                  const ce::CoreExpressionMap& coreMap,
                                  std::string& copyVarOut,
                                  std::size_t& firstIndexOut) {
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const std::string& p1 = chain[i];
        if (p1.empty() || p1[0] == '!') continue;
        const std::string op = ce::extractExpression(p1);
        const ce::CoreExpressionMap::const_iterator cfgIt = coreMap.find(op);
        if (cfgIt == coreMap.end()) continue;
        const ce::CoreExpressionConfig& cfg = cfgIt->second;
        if (cfg.outputIndices.size() != 1) continue;
        const int outIdx = cfg.outputIndices[0];
        const std::vector<std::string> a1 = ce::getArgs(p1);
        if (outIdx < 0 || outIdx >= static_cast<int>(a1.size())) continue;

        const auto isInputSlot = [&cfg](std::size_t k) -> bool {
            for (const int idx : cfg.inputIndices) {
                if (idx == static_cast<int>(k)) return true;
            }
            return false;
        };

        for (std::size_t j = i + 1; j < chain.size(); ++j) {
            const std::string& p2 = chain[j];
            if (p2.empty() || p2[0] == '!') continue;
            if (ce::extractExpression(p2) != op) continue;
            const std::vector<std::string> a2 = ce::getArgs(p2);
            if (a2.size() != a1.size()) continue;

            const std::string& r = a1[static_cast<std::size_t>(outIdx)];
            if (a2[static_cast<std::size_t>(outIdx)] != r) continue;
            if (r.find('(') != std::string::npos) continue;
            assert(r.rfind("u_", 0) != 0
                && "output-collision copy: theorem chains never carry u_ arguments");

            // ANCHOR GUARD (the I-24 analogue): never copy an anchor-slot
            // name. A shared output that is an anchor argument is a THEORY
            // CONSTANT (e.g. zero in `s(a)=0 ∧ s(b)=0 → a=b`), not a
            // theorem-bound result variable; duplicating a constant
            // variant-fans every statement carrying it across the grid and
            // breaches the one-changeable-arg integration contract.
            {
                bool rIsAnchorArg = false;
                for (std::size_t k2 = 0; k2 < chain.size() && !rIsAnchorArg;
                     ++k2) {
                    const std::string& el = chain[k2];
                    if (el.rfind("(Anchor", 0) != 0) continue;
                    const std::vector<std::string> aArgs = ce::getArgs(el);
                    for (const std::string& a : aArgs) {
                        if (a == r) { rIsAnchorArg = true; break; }
                    }
                }
                if (rIsAnchorArg) continue;
            }

            bool nonIoEqual = true;
            bool inputDiffers = false;
            for (std::size_t k = 0; k < a1.size(); ++k) {
                if (static_cast<int>(k) == outIdx) continue;
                if (isInputSlot(k)) {
                    if (a1[k] != a2[k]) inputDiffers = true;
                } else if (a1[k] != a2[k]) {
                    nonIoEqual = false;
                    break;
                }
            }
            if (!nonIoEqual || !inputDiffers) continue;

            // Exactly-two gate: a third positive premise matching the same
            // (operator, non-input/output args, output name) signature
            // suppresses the fire — narrow by design, extendable later.
            bool third = false;
            for (std::size_t k2 = 0; k2 < chain.size() && !third; ++k2) {
                if (k2 == i || k2 == j) continue;
                const std::string& p3 = chain[k2];
                if (p3.empty() || p3[0] == '!') continue;
                if (ce::extractExpression(p3) != op) continue;
                const std::vector<std::string> a3 = ce::getArgs(p3);
                if (a3.size() != a1.size()) continue;
                if (a3[static_cast<std::size_t>(outIdx)] != r) continue;
                bool nio = true;
                for (std::size_t k = 0; k < a1.size(); ++k) {
                    if (static_cast<int>(k) == outIdx || isInputSlot(k)) continue;
                    if (a1[k] != a3[k]) { nio = false; break; }
                }
                if (nio) third = true;
            }
            if (third) continue;

            copyVarOut = r;
            // Deposit at the SECOND colliding premise (maintainer decision
            // 2026-08-10): the first colliding premise is typically an
            // anchor-adjacent chain prefix SHARED by foreign conjectures,
            // and a deposit there mails the copy into their subtrees
            // (full-run Peano breach). Every consumer of the copy — the
            // deeper premise LBs, the contradiction twins under the
            // innermost premise, recursion auxiliaries — sits at or below
            // the second colliding premise, so mail still reaches them all.
            firstIndexOut = j;
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

    // Output-collision conjecture: two same-operator premises fuse their
    // result binders into one shared output name (the MPL encoding of an
    // equality between compound terms). Detect before the walk; the
    // deposit lands during the walk at the SECOND colliding premise LB —
    // mail flows only ancestor-to-descendant, and every consumer of the
    // copy (deeper premise LBs, contradiction twins under the innermost
    // premise, recursion auxiliaries) sits at or below it, while foreign
    // conjectures sharing only the first colliding premise as a chain
    // prefix never see it.
    std::string collisionCopyVar;
    std::size_t collisionDepositIndex = 0;
    const bool haveOutputCollision = detectOutputCollisionCopyVar(
        chain, this->coreExpressionMap, collisionCopyVar, collisionDepositIndex);

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

        // Output-collision copy axiom (=[r,r_copy]) at the second
        // colliding premise LB — same dead-end contract as the
        // antisymmetry deposit below (variableCopy origin, zero deps);
        // family sibling rows sharing this LB re-run the deposit and the
        // door dedups.
        if (haveOutputCollision && index == collisionDepositIndex) {
            const std::string copyEquality =
                "(=[" + collisionCopyVar + "," + collisionCopyVar + "_copy])";
            const TransientOrigin copyOrigin{
                true, OriginTag::variableCopy, nullptr, 0 };
            const int lvRunCopy[1] = { memoryBlock->level };
            this->addExprToMemoryBlock(StrSpan(copyEquality),
                *memoryBlock, iteration, 0, lvRunCopy, 1,
                copyOrigin, -1, -1, StrSpan("main", 4), false);

            // LB-creation paired mailOut write — the same timing exception
            // as the premise deposit above: buildGrid's startup commit
            // needs mailOut populated so every descendant receives the
            // axiom and its history line on its first phase-1 pull.
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
                if (headRow != nullptr) {
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
        // the record's levels are the authoritative deposit-time set: the
        // park captured frame ∪ registry
        // (D-327), so the run is empty only when
        // the parking compound itself was non-derived ({-1} tier); the
        // addStatement door re-stamps {-1} for that case.

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
}

void ExpressionAnalyzer::resetParkedOrStatementRegistries(Memory& mb,
    StrSpan original,
    StrSpan validityName)
{
    // encode (not lookup) — mirrors resetResentExpressionRegistries: the
    // statement and scope are both interned already (the statement was
    // deposited and the cohort parked there), so these are id fetches.
    const NameId origId = mb.nameMap.encode(original);
    const NameId valId = mb.nameMap.encode(validityName);
    mb.intStatementLevelsMap.eraseSet(packStatementKey(origId, valId));
    mb.intKnownStatements.erase(StatementKey{ origId, valId });
    // Stored rows are canonical-pipeline encodings, so the id pair is the
    // full identity; erase back to front to keep indices valid. The local
    // vectors are included — the parked statement is a LOCAL statement (the
    // non-local resetResentExpressionRegistries contract excludes them).
    for (int32_t i = mb.intEncodedStatements.size(); i-- > 0; ) {
        if (mb.intEncodedStatements[i].originalId == origId
            && mb.intEncodedStatements[i].validityId == valId) {
            mb.intEncodedStatements.erase(i);
        }
    }
    for (int32_t i = mb.intLocalEncodedStatements.size(); i-- > 0; ) {
        if (mb.intLocalEncodedStatements[i].originalId == origId
            && mb.intLocalEncodedStatements[i].validityId == valId) {
            mb.intLocalEncodedStatements.erase(i);
        }
    }
    for (int32_t i = mb.intLocalEncodedStatementsDelta.size(); i-- > 0; ) {
        if (mb.intLocalEncodedStatementsDelta[i].originalId == origId
            && mb.intLocalEncodedStatementsDelta[i].validityId == valId) {
            mb.intLocalEncodedStatementsDelta.erase(i);
        }
    }
    // The external staging is a view of the registry for this burst's
    // request generation: a row whose registry entry is gone must leave
    // it too, or request generation keeps a mandatory ingredient that
    // can no longer be a premise.
    for (int32_t i = mb.intExternalStatements.size(); i-- > 0; ) {
        if (mb.intExternalStatements[i].originalId == origId
            && mb.intExternalStatements[i].validityId == valId) {
            mb.intExternalStatements.erase(i);
        }
    }
    // The revived statement must re-enter the door as the representative:
    // drop its processed-or row (idempotent — a second park key of the
    // same or finds it already cleared).
    clearProcessedOr(mb, origId, valId);
}

/// @brief The or-uniqueness gate's probe — see the declaration's Doxygen
///        block in `prover.hpp` for the full contract.
///
/// @details
/// Implementation notes: the operator prefix (the bytes up to and
/// including the first `[`) must match before any canonicalization is
/// attempted — two different `or<N>` operators are never equi variants
/// of each other; the recorded text's canonicalization rides a per-call
/// string-tier scratch scope with a throwaway level run (the gate must
/// not touch the deposit's own level run) and mints nothing, so every
/// `decodeView` span held across the loop stays valid (I-3).
///
/// @param mb           Owning LB.
/// @param expr         The deposit's (door-canonical) or-compact text.
/// @param validityName The deposit's scope.
/// @return `true` when a recorded equi variant exists at the scope.
NameId ExpressionAnalyzer::canonicalOrIdAtScope(Memory& mb, NameId origId,
    StrSpan validityName)
{
    assert(origId != 0
        && "canonicalOrIdAtScope: the original id must be minted");
    const StrSpan text = mb.nameMap.decodeView(origId);
    assert(isOrCompactSpan(text)
        && "canonicalOrIdAtScope: ledger rows hold or compacts only");
    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(slot);
    ScratchScope canonScope(strArena);
    int lvBuf[256];
    int32_t lvN = 0;
    const CanonicalForm cf = canonicalFormAtScope(
        text, validityName, mb, strArena, lvBuf, lvN, 256);
    if (!cf.changed) return origId;
    // The one mint of the ledger path: `text` (a decodeView span) is dead
    // from here on; `cf.text` rides the scratch scope and dies at return.
    return mb.nameMap.encode(cf.text);
}

bool ExpressionAnalyzer::orEquiRepresentativeRecorded(Memory& mb,
    StrSpan expr,
    StrSpan validityName)
{
    assert(isOrCompactSpan(expr)
        && "orEquiRepresentativeRecorded: caller must gate on the or-compact shape");

    const NameId vid = mb.nameMap.lookup(validityName);
    if (vid == 0) return false;   // fresh scope: nothing recorded there yet
    const int32_t row = mb.processedOrLedger.lookup(vid);
    if (row == 0) return false;
    // Never interned here -> no recorded original and no re-keyed canonical
    // text can equal it (both halves are minted ids of this NameMap).
    const NameId exprId = mb.nameMap.lookup(expr);
    if (exprId == 0) return false;
    const int32_t n = mb.processedOrLedger.runLen(row);
    for (int32_t j = 0; j < n; ++j) {
        const int64_t v = mb.processedOrLedger.valueAt(row, j);
        if (ledgerCanonicalId(v) == exprId || ledgerOriginalId(v) == exprId)
            return true;
    }
    return false;
}

void ExpressionAnalyzer::recordProcessedOr(Memory& mb, NameId origId,
    NameId valId)
{
    assert(origId != 0 && valId != 0
        && "recordProcessedOr: both ids must be minted by the door path");
    // The scope-name view is read by the canonicalization only, before the
    // possible mint inside — no span outlives the mint (I-3).
    const NameId canonId =
        canonicalOrIdAtScope(mb, origId, mb.nameMap.decodeView(valId));
    mb.processedOrLedger.insertSorted(valId, packInt32Pair(canonId, origId),
        [](int64_t a, int64_t b) { return a < b; });
}

void ExpressionAnalyzer::clearProcessedOr(Memory& mb, NameId origId,
    NameId valId)
{
    assert(origId != 0 && valId != 0
        && "clearProcessedOr: both ids must be minted");
    const int32_t row = mb.processedOrLedger.lookup(valId);
    if (row == 0) return;   // defined already-cleared state
    const int32_t n = mb.processedOrLedger.runLen(row);

    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();
    int64_t* keep = reinterpret_cast<int64_t*>(gArena.resolve(gArena.alloc(
        (n > 0 ? n : 1) * static_cast<int32_t>(sizeof(int64_t)),
        static_cast<int32_t>(alignof(int64_t)))));
    int32_t keepN = 0;
    for (int32_t j = 0; j < n; ++j) {
        const int64_t v = mb.processedOrLedger.valueAt(row, j);
        if (ledgerOriginalId(v) != origId) keep[keepN++] = v;
    }
    if (keepN != n) {
        mb.processedOrLedger.assignSet(valId, keep, keepN);
    }
    gArena.popTo(mark);
}

void ExpressionAnalyzer::recanonicalizeProcessedOrLedger(Memory& mb,
    NameId valId)
{
    assert(valId != 0
        && "recanonicalizeProcessedOrLedger: the scope id must be minted");
    const int32_t row = mb.processedOrLedger.lookup(valId);
    if (row == 0) return;   // no or processed at this scope: defined no-op
    const int32_t n = mb.processedOrLedger.runLen(row);
    if (n == 0) return;     // every row cleared: defined no-op

    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();
    int64_t* next = reinterpret_cast<int64_t*>(gArena.resolve(gArena.alloc(
        n * static_cast<int32_t>(sizeof(int64_t)),
        static_cast<int32_t>(alignof(int64_t)))));
    const unsigned sSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(sSlot);
    ScratchScope nameScope(strArena);
    // Arena-held copy of the scope name: the per-row mint below may
    // invalidate a bare NameMap view (I-3).
    const StrSpan vView = mb.nameMap.decodeView(valId);
    const ScratchString vHold =
        ScratchString::copyFrom(strArena, vView.ptr, vView.len);
    const StrSpan vName(vHold);
    bool changed = false;
    for (int32_t j = 0; j < n; ++j) {
        const int64_t v = mb.processedOrLedger.valueAt(row, j);
        const NameId origId = ledgerOriginalId(v);
        const NameId canonId = canonicalOrIdAtScope(mb, origId, vName);
        next[j] = packInt32Pair(canonId, origId);
        if (canonId != ledgerCanonicalId(v)) changed = true;
    }
    if (changed) {
        std::sort(next, next + n);
        mb.processedOrLedger.assignSet(valId, next, n);
    }
    gArena.popTo(mark);
}

void ExpressionAnalyzer::revisitRejectedOrdis(StrSpan markedExpr,
    Memory& memoryBlock,
    StrSpan validityName)
{
    auto& rmo = memoryBlock.overallHashMemory.rejectedMapOrdis;

    // Non-minting probe — a never-interned template was never parked; a
    // key without a parked run is a defined miss.
    int64_t revisitPk = 0;
    if (!lookupTemplateKey(memoryBlock.templateInterner,
            memoryBlock.nameMap, markedExpr, validityName, revisitPk)) {
        return;
    }
    if (rmo.lookup(revisitPk) == 0) {
        return;
    }

    // DEDICATED re-entrancy guard (never shared with revisitInProgress — a
    // concurrent general revisit of the same packed key must not swallow an
    // ordis wake).
    if (memoryBlock.overallHashMemory.ordisRevisitInProgress.contains(revisitPk)) {
        return;
    }
    memoryBlock.overallHashMemory.ordisRevisitInProgress.mint(revisitPk);

    // Snapshot the parked cohort as VERBATIM blob copies on the gen scratch
    // arena — copy BEFORE erase (the eraseBlobIf restructures the blob
    // pool; the revisitRejected2 discipline). Erasing the whole key first
    // also makes a legitimate re-park by the re-consumption below start
    // from a clean slate.
    const unsigned ordisSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(ordisSlot);
    const ArenaOffset mark = gArena.cursor();
    const RejectedOrdisRunSnapshot snap =
        snapshotRejectedOrdisRun(rmo, revisitPk, gArena);
    rmo.eraseBlobIf([revisitPk](int64_t k) { return k == revisitPk; });

    for (int32_t r = 0; r < snap.count; ++r) {
        const RejectedOrdisValueBlobView& v = snap.views[r];
        // decodeView is I-3-safe across the loop: the sinks below mint
        // nameMap + originInterner, never valueInterner, and the walked
        // blob bytes are gen-arena copies.
        const StrSpan parkedStmt =
            memoryBlock.valueInterner.decodeView(v.orStatementId());
        const int32_t lvN = v.levelCount();
        int lvRun[256];
        assert(lvN <= 256 && "parked ordis level run exceeds lvRun");
        for (int32_t j = 0; j < lvN; ++j) lvRun[j] = v.levelAt(j);

        // The un-know (maintainer-approved): the parked statement is a
        // known local statement, so without this reset the Site F ancestor
        // dedup would drop the re-deposit before the or consumption
        // re-runs.
        resetParkedOrStatementRegistries(memoryBlock, parkedStmt,
                                         validityName);

        // Re-deposit; the absorb re-runs the full or consumption (fresh
        // probe, standing starter tie rule, K rules deduped, fresh-cohort
        // guard). pre == post self-source equality1 mirrors the
        // revisitRejected2 revival door; the statement's foundation origin
        // already sits in exprOriginMap and D-49's cap-full preference
        // protects it.
        memoryBlock.mutatedThisBurst = true;
        insertInternalStatement(memoryBlock.sameIterationInternalMail,
            memoryBlock.nameMap, parkedStmt, validityName, lvRun, lvN);
        if (parameters.trackHistory) {
            OriginDep dep[1] = { { parkedStmt, validityName } };
            addInternalMailOrigin(memoryBlock.sameIterationInternalMail,
                memoryBlock.originInterner, parkedStmt, validityName,
                OriginTag::equality1, dep, 1,
                (parameters.compressor_mode
                     ? parameters.compressor_max_origins_per_expr
                     : parameters.max_origin_per_expr));
        }
    }

    gArena.popTo(mark);

    memoryBlock.overallHashMemory.ordisRevisitInProgress.erase(revisitPk);
}

void ExpressionAnalyzer::revisitRejectedOrdis2(StrSpan groundText,
    Memory& memoryBlock,
    StrSpan validityName)
{
    auto& rmo2 = memoryBlock.overallHashMemory.rejectedMapOrdis2;

    // Non-minting probe — a never-interned ground text was never filed; a
    // key without a parked run is a defined miss (a demand with no supply
    // waits in admissionMapOrdis2 for a later deposit's route-(c) probe).
    int64_t revisitPk = 0;
    if (!lookupTemplateKey(memoryBlock.templateInterner,
            memoryBlock.nameMap, groundText, validityName, revisitPk)) {
        return;
    }
    if (rmo2.lookup(revisitPk) == 0) {
        return;
    }

    // DEDICATED re-entrancy guard (never shared with ordisRevisitInProgress
    // — a concurrent old-map wake of the same packed key must not swallow
    // an ordis2 wake).
    if (memoryBlock.overallHashMemory.ordis2RevisitInProgress.contains(revisitPk)) {
        return;
    }
    memoryBlock.overallHashMemory.ordis2RevisitInProgress.mint(revisitPk);

    // Snapshot the filed cohort as VERBATIM blob copies on the gen scratch
    // arena — copy BEFORE erase (the eraseBlobIf restructures the blob
    // pool; the revisitRejectedOrdis discipline). Erasing the whole key
    // first also makes a legitimate re-park by the re-consumption below
    // start from a clean slate; the or's entries under OTHER disjunct keys
    // (and in rejectedMapOrdis) stay — a stale later wake re-deposits an
    // already-open or, which the cohort bootstrap guard turns into a no-op,
    // and scope wipe clears the rest at discharge.
    const unsigned ordisSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(ordisSlot);
    const ArenaOffset mark = gArena.cursor();
    const RejectedOrdisRunSnapshot snap =
        snapshotRejectedOrdisRun(rmo2, revisitPk, gArena);
    rmo2.eraseBlobIf([revisitPk](int64_t k) { return k == revisitPk; });

    for (int32_t r = 0; r < snap.count; ++r) {
        const RejectedOrdisValueBlobView& v = snap.views[r];
        // decodeView is I-3-safe across the loop: the sinks below mint
        // nameMap + originInterner, never valueInterner, and the walked
        // blob bytes are gen-arena copies.
        const StrSpan parkedStmt =
            memoryBlock.valueInterner.decodeView(v.orStatementId());
        const int32_t lvN = v.levelCount();
        int lvRun[256];
        assert(lvN <= 256 && "filed ordis2 level run exceeds lvRun");
        for (int32_t j = 0; j < lvN; ++j) lvRun[j] = v.levelAt(j);

        // The un-know (the I-178 exception, same as the old-map wake): the
        // parked statement is a known local statement, so without this
        // reset the Site F ancestor dedup would drop the re-deposit before
        // the or consumption re-runs.
        resetParkedOrStatementRegistries(memoryBlock, parkedStmt,
                                         validityName);

        // Re-deposit; the absorb re-runs the full or consumption, whose
        // route-(c) probe now reads the freshly-drained demand entry and
        // opens (consuming it). pre == post self-source equality1 mirrors
        // the revisitRejectedOrdis revival door.
        memoryBlock.mutatedThisBurst = true;
        insertInternalStatement(memoryBlock.sameIterationInternalMail,
            memoryBlock.nameMap, parkedStmt, validityName, lvRun, lvN);
        if (parameters.trackHistory) {
            OriginDep dep[1] = { { parkedStmt, validityName } };
            addInternalMailOrigin(memoryBlock.sameIterationInternalMail,
                memoryBlock.originInterner, parkedStmt, validityName,
                OriginTag::equality1, dep, 1,
                (parameters.compressor_mode
                     ? parameters.compressor_max_origins_per_expr
                     : parameters.max_origin_per_expr));
        }
    }

    gArena.popTo(mark);

    memoryBlock.overallHashMemory.ordis2RevisitInProgress.erase(revisitPk);
    // Demand consumption happens only at the route-(c) open (D-267),
    // never here.
}


// Integration-side counterpart to revisitRejected2. Walks
// rejectedMapIntegration[markedKey, validityName] and, for each stored
// entry, emits the constituent + siblings onto sameIterationInternalMail so the next
// hashburst re-runs the full disintegration pipeline at the constituent's
// original validity. Unlike revisitRejected2:
//   * no addExprToMemoryBlock call (mailIn-only revival, linear),
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
        // Ordis-only values never seed recursion propagation — they are
        // cohort-opening demand evidence, invisible to general admission;
        // the derived keys this walk inserts stay untagged by construction.
        if (val.ordisByte() != 0) {
            continue;
        }
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
                        })) {
                    continue;
                }

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

                if (!allInputsPresent) {
                    continue;
                }

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

                const int64_t newMarkedPk = mintTemplateKey(mb.templateInterner,
                    mb.nameMap, StrSpan(newMarkedExpr), validityName);
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
                // Ordis revival at the recursion-propagation key gain — a
                // parked cohort matching the derived key wakes by mail.
                this->revisitRejectedOrdis(StrSpan(newMarkedExpr), mb, validityName);
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

#if MEM_MEASUREMENT
    // One iteration, one memory sample. Zeroing here (single-threaded, before
    // any worker spawns) makes the fold below count each LB active in this
    // iteration exactly once.
    gl::mem_tracker::resetIteration();
#endif

    // Reduced-or closure for every or already in the registry (GL-binary
    // load, external-theorem precompile, prior seams) BEFORE this
    // iteration's bursts can flat-consume an or statement. Single-threaded
    // here — workers spawn later (I-137); idempotent, so the per-iteration
    // re-run costs one deduping registry scan.
    preMintReducedOrs();

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
        && !parameters.compressor_mode && !ceFilteringActive && !warmUpPhase;
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

#if RT_MEASUREMENT
    // [RT phase measurement - file-bound, see .rt/burst_phases.log]
    // Per-LB wall-time ledgers for this iteration, indexed like `active`:
    // phase-1 sweep, phase-2 executor compute, phase-2 finalize, phase-3 sweep,
    // plus a copy of the split-invariant submatch work for the report.
    std::vector<int64_t> trapPh1Ns(active.size(), 0);
    std::vector<int64_t> trapPh2Ns(active.size(), 0);
    std::vector<int64_t> trapFinNs(active.size(), 0);
    std::vector<int64_t> trapPh3Ns(active.size(), 0);
    std::vector<int64_t> trapWork(active.size(), 0);
#endif

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
                                       auto&& body_fn
#if RT_MEASUREMENT
                                       , int64_t* perLbNs = nullptr
#endif
                                       ) {
        auto worker = [&active, &next, workers, &body_fn
#if RT_MEASUREMENT
                       , perLbNs
#endif
                       ](unsigned coreId) {
            const unsigned cid = workers ? (coreId % workers) : 0U;
            for (;;) {
                std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= active.size()) break;
#if RT_MEASUREMENT
                // [RT phase measurement] per-LB wall time; each index i
                // is dispatched to exactly one worker, so the write is race-free.
                const auto trapT0 = std::chrono::steady_clock::now();
#endif
                body_fn(*active[i], cid);
#if RT_MEASUREMENT
                if (perLbNs != nullptr)
                    perLbNs[i] += std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - trapT0).count();
#endif
            }
            };
        std::vector<std::thread> pool;
        pool.reserve(workers);
        for (unsigned t = 0; t < workers; ++t) pool.emplace_back(worker, t);
        for (auto& th : pool) th.join();
        };

#if PHASE13_DEEP_TIMING
    std::vector<int64_t> phase1DetailRows(
        static_cast<std::size_t>(workers) * kPhase13TimingSlotCount, 0);
    std::vector<int64_t> phase3DetailRows(
        static_cast<std::size_t>(workers) * kPhase13TimingSlotCount, 0);
    static constexpr const char* phase13TimingLabels[] = {
        "claim_load",
        "burst_setup",
        "routing_mail_pull",
        "external_origin_absorb",
        "external_statement_absorb",
        "internal_origin_absorb",
        "internal_statement_absorb",
        "internal_mail_clear",
        "apply_equivalence_classes",
        "enrich_products_of_recursion",
        "cleanup_expressions",
        "discharge_contradiction",
        "discharge_to_be_proved",
        "discharge_contradiction_scopes",
        "fill_mail_out",
        "changed_classes_clear",
        "routing_mail_cleanup",
        "react_to_hypothesis",
        "sanitize_to_be_proved",
        "drain_disproved_goals",
        "drain_dead_or_branches",
        "freeze_resolved_or_branches",
        "drain_pending_or_releases",
        "wipe_subtrees",
        "sweep_ancestor_known_rows",
        "quiescence_and_dumps",
        "release_claim"
    };
    static_assert(sizeof(phase13TimingLabels) / sizeof(phase13TimingLabels[0])
                      == kPhase13TimingSlotCount,
                  "Phase 1/3 timing labels must cover every slot");
    const auto printPhase13Detail =
        [workers](int phase, double barrierSeconds,
                  const std::vector<int64_t>& rows) {
            int64_t categorizedNs = 0;
            diagnosticsLog() << "[PHASE13-DETAIL] phase=" << phase
                      << " barrier_seconds=" << barrierSeconds;
            for (std::size_t slot = 0; slot < kPhase13TimingSlotCount; ++slot) {
                int64_t slotNs = 0;
                for (unsigned worker = 0; worker < workers; ++worker) {
                    slotNs += rows[static_cast<std::size_t>(worker)
                                       * kPhase13TimingSlotCount
                                   + slot];
                }
                categorizedNs += slotNs;
                diagnosticsLog() << ' ' << phase13TimingLabels[slot] << "_worker_seconds="
                          << static_cast<double>(slotNs) / 1e9;
            }
            diagnosticsLog() << " categorized_worker_seconds="
                      << static_cast<double>(categorizedNs) / 1e9
                      << std::endl;
        };
#endif

    // Phase 1 opens a working-set window over its dispatch cursor
    // (D-161, I-114): the steward
    // prefetches the upcoming LBs and, above the watermarks, drains the
    // deloadable rest. Every LB's handshake (claimAndLoadForWork) makes it
    // resident before its body and releases the claim after, so the steward
    // can reclaim it once done.
    const auto phase1Started = std::chrono::steady_clock::now();
    {
        std::atomic<std::size_t> phase1Cursor{ 0 };
#if PHASE13_DEEP_TIMING
        phase13TimingRows = phase1DetailRows.data();
        phase13TimingWorkers = workers;
#endif
        steward->beginPhaseWindow(/*phase=*/1, &phase1Cursor, &active, workers,
                                  lbdeload::kDeloadDirectory);
        runPhase(phase1Cursor, [this](Memory& b, unsigned cid) {
            g_inParallelWorkerPhase = true;
            this->performElemPhase1(b, cid);
            g_inParallelWorkerPhase = false;
        }
#if RT_MEASUREMENT
        , trapPh1Ns.data()
#endif
        );
        steward->endPhaseWindow();
#if PHASE13_DEEP_TIMING
        phase13TimingRows = nullptr;
        phase13TimingWorkers = 0;
#endif
    }
    const double phase1IterationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase1Started).count();

    // The always-on Phase 2 measurement starts after the phase-1 barrier and
    // stops before phase 3. It includes scheduler setup, processor or CUDA
    // execution, projection/transfers, sealing, finalization, and split stats.
    const auto phase2Started = std::chrono::steady_clock::now();

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
    // disable_lb_split and the incubator run UNSPLIT; the CE filter is unaffected
    // (its own single-part loop in filter.cpp).
    struct ExecTask {
        Memory* lb;
        std::size_t li;          // index into `active`
        int partCount;           // concurrent parts of this LB this burst (= bucket count)
        bool produceOnly;        // a round-1 stump PRODUCER (runs produceExpressionStumps,
                                 // not performElem2); its buckets requeue for round 2
        SplitStumpRef stump;     // empty for an unsplit part; set for a bucket part
        std::atomic<int64_t>* doomLine;  // the LB's shared packed (position, ordinal) stop line
        std::atomic<int>* partsLeft;
    };
    const bool mainPath = parameters.lb_split && !parameters.disable_lb_split
        && !ceFilteringActive;
    // A straggler splits into logicalCores expression buckets; that is the only
    // split dimension now, so the whole-machine core count is the per-LB part
    // ceiling (Rule 19 -- a machine past the named constant stops HERE).
    assert(static_cast<int>(logicalCores) <= kMaxSplitParts
        && "logicalCores exceeds kMaxSplitParts - raise the named constant "
           "deliberately for a machine with more cores than the ceiling");

    if (!active.empty()) {
    const std::size_t M = active.size();
    // Per-LB phase-2 early-exit doom lines, EXTERNAL to the LB so the hashburst
    // stays strictly read-only on it (I-66). One packed (position, ordinal)
    // atomic per LB, shared by all its parts and lowered only by CAS-min; sized
    // once so the &doomLines[li] handed to tasks stay stable across passes
    // (the vector is never resized; std::atomic is not movable).
    std::vector<std::atomic<int64_t>> doomLines(M);
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
    // Doom-line latch: 1 when any of this LB's parts hit a burst early-exit
    // (a burstDeactivates trigger) in ANY pass this iteration. An early-exited
    // part stops at a timing-dependent overshoot past the published line in
    // its own stream (I-191 protects the merged CONTENT, not the tallies), so
    // a doomed LB's submatch tally is not a deterministic work figure — the
    // straggler statistic below excludes it. Whether a doom trigger fires at
    // all IS deterministic (the winning line's position is always reached by
    // its part), so the exclusion keeps the split set a pure function of
    // proof state (I-160). Latched after each pass's join because the
    // per-pass doomLines reset would erase a pass-1 doom.
    std::vector<char> lbDoomed(M, 0);
    // Diagnostics for the split-ineffective report: how many stump work items the
    // producer returned and how many buckets it dealt them into (1 = not really split).
    std::vector<int32_t> lbStumps(M, 0);
    std::vector<int32_t> lbBuckets(M, 1);
    std::vector<char> lbFinalized(M, 0);
    std::vector<char> lbFiringRecordsCanonical(M, 0);
    // Device-projection capacity census. The first executor task that claims an
    // LB records its post-phase-1 read image once; split bucket siblings and a
    // second pass see the same read-only LB and are collapsed by projectionCounted.
    // Atomics make the observation race-free; addition and maximum are independent
    // of which sibling wins the exchange. This telemetry never gates proof flow.
    std::vector<std::atomic<uint8_t>> projectionCounted(M);
    for (std::atomic<uint8_t>& counted : projectionCounted)
        counted.store(0, std::memory_order_relaxed);
    std::atomic<uint64_t> projectionLogicalBlocks{ 0 };
    std::atomic<uint64_t> projectionStatements{ 0 };
    std::atomic<uint64_t> projectionMaximumStatements{ 0 };
    std::atomic<uint64_t> projectionNameRecords{ 0 };
    std::atomic<uint64_t> projectionMaximumNameRecords{ 0 };
    std::atomic<uint64_t> projectionNameBytes{ 0 };
    std::atomic<uint64_t> projectionMaximumNameBytes{ 0 };
    constexpr std::size_t kProjectionSemanticMetricCount = 20;
    std::array<std::atomic<uint64_t>,
               kProjectionSemanticMetricCount> projectionSemanticTotals;
    std::array<std::atomic<uint64_t>,
               kProjectionSemanticMetricCount> projectionSemanticMaximums;
    for (std::size_t metric = 0;
         metric < kProjectionSemanticMetricCount; ++metric) {
        projectionSemanticTotals[metric].store(0, std::memory_order_relaxed);
        projectionSemanticMaximums[metric].store(0, std::memory_order_relaxed);
    }
    constexpr std::size_t kTaskProjectionMetricCount = 4;
    std::array<std::atomic<uint64_t>,
               kTaskProjectionMetricCount> taskProjectionTotals;
    std::array<std::atomic<uint64_t>,
               kTaskProjectionMetricCount> taskProjectionMaximums;
    for (std::size_t metric = 0;
         metric < kTaskProjectionMetricCount; ++metric) {
        taskProjectionTotals[metric].store(0, std::memory_order_relaxed);
        taskProjectionMaximums[metric].store(0, std::memory_order_relaxed);
    }
    gpuRequestFilterCalls.store(0, std::memory_order_relaxed);
    gpuProducerFilterCalls.store(0, std::memory_order_relaxed);
    gpuFilterInputStatements.store(0, std::memory_order_relaxed);
    gpuFilterOutputStatements.store(0, std::memory_order_relaxed);
    gpuFilterMaximumInputStatements.store(0, std::memory_order_relaxed);
    gpuFilterMaximumOutputStatements.store(0, std::memory_order_relaxed);
    constexpr std::size_t kGpuGrowDepthCount =
        ExecutionParameters::MAX_EXPRESSIONS + 1;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuGrowAttemptsByDepth;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuGrowFrontierByDepth;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuGrowSubkeysByDepth;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuGrowRequestsByDepth;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuProducerAttemptsByDepth;
    std::array<std::atomic<uint64_t>, kGpuGrowDepthCount>
        gpuProducerSurvivorsByDepth;
    for (std::size_t depth = 0; depth < kGpuGrowDepthCount; ++depth) {
        gpuGrowAttemptsByDepth[depth].store(0, std::memory_order_relaxed);
        gpuGrowFrontierByDepth[depth].store(0, std::memory_order_relaxed);
        gpuGrowSubkeysByDepth[depth].store(0, std::memory_order_relaxed);
        gpuGrowRequestsByDepth[depth].store(0, std::memory_order_relaxed);
        gpuProducerAttemptsByDepth[depth].store(0, std::memory_order_relaxed);
        gpuProducerSurvivorsByDepth[depth].store(0, std::memory_order_relaxed);
    }
    constexpr std::size_t kGpuEvaluationTotalCount = 15;
    constexpr std::size_t kGpuEvaluationMaximumCount = 3;
    std::array<std::atomic<uint64_t>, kGpuEvaluationTotalCount>
        gpuEvaluationTotals;
    std::array<std::atomic<uint64_t>, kGpuEvaluationMaximumCount>
        gpuEvaluationMaximums;
    for (std::atomic<uint64_t>& total : gpuEvaluationTotals)
        total.store(0, std::memory_order_relaxed);
    for (std::atomic<uint64_t>& maximum : gpuEvaluationMaximums)
        maximum.store(0, std::memory_order_relaxed);
    // The stumps a producer task returns, copied off its sealed pages so those pages
    // go back to the pool at once. The bucket tasks point into these runs, so the
    // storage must outlive the pass: deque, never reallocated.
    std::deque<std::vector<ExpressionStump>> stumpStore;

    // Task build. A straggler (numberOfParts > 1, set last iteration by the stats
    // pass below) dispatches ONE producer task that enumerates the LB's expression
    // stumps; its buckets run in round 2. Every other LB dispatches ONE unsplit part.
    std::vector<ExecTask> tasks;
    for (std::size_t li = 0; li < M; ++li) {
        Memory* b = active[li];
        if (!b->isActive) continue;  // discharged in phase 1 -> no executor tasks
        // Split preemptively if the submatch stat flagged it last iteration
        // (numberOfParts > 1) OR it just activated this iteration (justActivated,
        // one-shot: no prior burst for the stat to see). Consume the flag here.
        const bool straggler = mainPath && (b->numberOfParts > 1 || b->justActivated);
        b->justActivated = false;
        tasks.push_back(ExecTask{ b, li,
            /*partCount=*/1, /*produceOnly=*/straggler, SplitStumpRef{},
            &doomLines[li], &partsRemaining[li] });
    }

    int passNo = 0;
    for (;;) {
        ++passNo;
        // Round 1: producers + unsplit bursts. Round 2: the producers' buckets.
        assert(passNo <= 2
            && "phase-2 pass loop exceeded two rounds - a bucket part requeued");

        for (std::size_t li = 0; li < M; ++li) {
            doomLines[li].store(kNoDoomLine, std::memory_order_relaxed);
            partsRemaining[li].store(0, std::memory_order_relaxed);
        }
        for (const ExecTask& t : tasks) {
            if (phase2Backend == Phase2Backend::cpu || t.produceOnly)
                partsRemaining[t.li].fetch_add(1, std::memory_order_relaxed);
        }

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
        // after its performElem2 returns (the submatches this part visited). See
        // D-109.
        std::vector<int64_t> taskSubMatches(tasks.size(), 0);
        // CUDA sealing emits one already-canonical chain per projected LB. The
        // first task for that LB owns the page set; sibling tasks still retain
        // their independent split-work counts but do not own duplicate chains.
        std::vector<int32_t> gpuTaskIndices(tasks.size(), -1);
        std::vector<char> gpuOutputOwners(tasks.size(), 0);
#if RT_MEASUREMENT
        // [RT phase measurement] per-task compute wall time (the
        // produce/burst call only; claim/load IO excluded).
        std::vector<int64_t> taskNs(tasks.size(), 0);
#endif
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
                            &taskStumps, &taskSubMatches,
                            &projectionCounted, &projectionLogicalBlocks,
                            &projectionStatements, &projectionMaximumStatements,
                            &projectionNameRecords, &projectionMaximumNameRecords,
                            &projectionNameBytes, &projectionMaximumNameBytes,
                            &projectionSemanticTotals,
                            &projectionSemanticMaximums,
                            &taskProjectionTotals,
                            &taskProjectionMaximums,
                            &gpuGrowAttemptsByDepth,
                            &gpuGrowFrontierByDepth,
                            &gpuGrowSubkeysByDepth,
                            &gpuGrowRequestsByDepth,
                            &gpuProducerAttemptsByDepth,
                            &gpuProducerSurvivorsByDepth,
                            &gpuEvaluationTotals,
                            &gpuEvaluationMaximums,
#if RT_MEASUREMENT
                           &taskNs,
#endif
                           &next, workers,
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
                    if (this->phase2Backend == Phase2Backend::cuda
                        && !t.produceOnly) continue;
                    // Unified working-set handshake: claim this LB so the steward
                    // will not deload it under us, reloading if it is cold. A split
                    // sibling that already owns it returns immediately (resident).
                    steward->claimAndLoadForWork(*t.lb, /*phase=*/2,
                                                 lbdeload::kDeloadDirectory);
                    if (projectionCounted[t.li].exchange(
                            1, std::memory_order_relaxed) == 0) {
                        const gpu::Phase2ProjectionUsage projectionUsage =
                            gpu::measurePhase2ProjectionUsage(*t.lb, *this);
                        const uint64_t statementCount =
                            projectionUsage.statements;
                        const uint64_t nameRecordCount =
                            projectionUsage.nameRecords;
                        const uint64_t nameByteCount = projectionUsage.nameBytes;
                        projectionLogicalBlocks.fetch_add(
                            1, std::memory_order_relaxed);
                        projectionStatements.fetch_add(
                            statementCount, std::memory_order_relaxed);
                        projectionNameRecords.fetch_add(
                            nameRecordCount, std::memory_order_relaxed);
                        projectionNameBytes.fetch_add(
                            nameByteCount, std::memory_order_relaxed);
                        auto publishMaximum = [](std::atomic<uint64_t>& maximum,
                                                 uint64_t value) {
                            uint64_t observed = maximum.load(
                                std::memory_order_relaxed);
                            while (observed < value
                                && !maximum.compare_exchange_weak(
                                    observed, value,
                                    std::memory_order_relaxed,
                                    std::memory_order_relaxed)) {
                            }
                        };
                        publishMaximum(
                            projectionMaximumStatements, statementCount);
                        publishMaximum(
                            projectionMaximumNameRecords, nameRecordCount);
                        publishMaximum(
                            projectionMaximumNameBytes, nameByteCount);
                        const uint64_t semanticValues[
                            kProjectionSemanticMetricCount] = {
                            projectionUsage.ruleStringRecords,
                            projectionUsage.ruleStringBytes,
                            projectionUsage.nameSlots,
                            projectionUsage.byteMapViews,
                            projectionUsage.byteMapEntries,
                            projectionUsage.byteMapSlots,
                            projectionUsage.byteKeyBytes,
                            projectionUsage.blobRecords,
                            projectionUsage.blobBytes,
                            projectionUsage.reverseMapViews,
                            projectionUsage.reverseMapEntries,
                            projectionUsage.reverseMapSlots,
                            projectionUsage.reverseKeyBytes,
                            projectionUsage.reverseOwners,
                            projectionUsage.podMapViews,
                            projectionUsage.podMapEntries,
                            projectionUsage.podMapSlots,
                            projectionUsage.podRunValues,
                            projectionUsage.mandatoryStatementKeys,
                            projectionUsage.metadataBytes
                        };
                        for (std::size_t metric = 0;
                             metric < kProjectionSemanticMetricCount; ++metric) {
                            projectionSemanticTotals[metric].fetch_add(
                                semanticValues[metric],
                                std::memory_order_relaxed);
                            publishMaximum(
                                projectionSemanticMaximums[metric],
                                semanticValues[metric]);
                        }
                    }
                    // The task's exclusive write window on its page set opens here
                    // and closes at the seal below -- the records' strings then cross
                    // the pool join read-only.
                    SealedPageSet& ps = pageStore[base + i];
                    ps.bind(&staticMemory());
#if RT_MEASUREMENT
                    // [RT phase measurement] compute wall time of this
                    // task; index i belongs to exactly this worker (race-free).
                    const auto trapT0 = std::chrono::steady_clock::now();
#endif
                    for (std::size_t depth = 0;
                         depth < kGpuGrowDepthCount; ++depth) {
                        g_gpuGrowAttemptsByDepth[depth] = 0;
                        g_gpuGrowFrontierByDepth[depth] = 0;
                        g_gpuGrowSubkeysByDepth[depth] = 0;
                        g_gpuGrowRequestsByDepth[depth] = 0;
                        g_gpuProducerAttemptsByDepth[depth] = 0;
                        g_gpuProducerSurvivorsByDepth[depth] = 0;
                    }
                    g_gpuEvaluationUsage = GpuEvaluationUsage{};
                    if (t.produceOnly) {
                        // Round-1 stump PRODUCER: enumerate the whole LB's
                        // expression stumps, retaining terminal pre-stumps for
                        // recordable nodes replaced by a deeper level. It fires
                        // nothing and deposits nothing (ps stays empty); the
                        // classify deals all work items into buckets that run in
                        // round 2. Grow MORE stumps than buckets (a small multiple
                        // of logicalCores) so the round-robin deal evens out the
                        // buckets' grow-tree sizes.
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
                        assert(ruleStagings().slotIsEmpty(cid)
                            && "producer exit: staged rule-index writes outlive their window");
                        genScratchArenas().forSlot(cid).releaseAll();
                    } else {
                        // A burst part: an unsplit LB, or one expression bucket of a
                        // straggler. Runs to completion unless the LB's doom line
                        // stops it at a deterministic stream position.
                        const gpu::Phase2TaskProjectionUsage taskUsage =
                            gpu::measurePhase2TaskProjectionUsage(
                                *t.lb,
                                static_cast<uint32_t>(t.stump.count),
                                this->ceFilteringActive);
                        const uint64_t taskValues[kTaskProjectionMetricCount] = {
                            taskUsage.tasks, taskUsage.batches,
                            taskUsage.terms, taskUsage.stumps
                        };
                        for (std::size_t metric = 0;
                             metric < kTaskProjectionMetricCount; ++metric) {
                            taskProjectionTotals[metric].fetch_add(
                                taskValues[metric], std::memory_order_relaxed);
                            uint64_t observed = taskProjectionMaximums[metric].load(
                                std::memory_order_relaxed);
                            while (observed < taskValues[metric]
                                && !taskProjectionMaximums[metric].compare_exchange_weak(
                                    observed, taskValues[metric],
                                    std::memory_order_relaxed,
                                    std::memory_order_relaxed)) {
                            }
                        }
                        this->performElem2(*t.lb, cid, t.partCount, t.stump, ps,
                                           *t.doomLine);
                        ps.seal();
                        taskSubMatches[i] = g_growthMatchCount;
                    }
                    for (std::size_t depth = 0;
                         depth < kGpuGrowDepthCount; ++depth) {
                        gpuGrowAttemptsByDepth[depth].fetch_add(
                            g_gpuGrowAttemptsByDepth[depth],
                            std::memory_order_relaxed);
                        gpuGrowFrontierByDepth[depth].fetch_add(
                            g_gpuGrowFrontierByDepth[depth],
                            std::memory_order_relaxed);
                        gpuGrowSubkeysByDepth[depth].fetch_add(
                            g_gpuGrowSubkeysByDepth[depth],
                            std::memory_order_relaxed);
                        gpuGrowRequestsByDepth[depth].fetch_add(
                            g_gpuGrowRequestsByDepth[depth],
                            std::memory_order_relaxed);
                        gpuProducerAttemptsByDepth[depth].fetch_add(
                            g_gpuProducerAttemptsByDepth[depth],
                            std::memory_order_relaxed);
                        gpuProducerSurvivorsByDepth[depth].fetch_add(
                            g_gpuProducerSurvivorsByDepth[depth],
                            std::memory_order_relaxed);
                    }
                    const uint64_t evaluationTotals[
                        kGpuEvaluationTotalCount] = {
                        g_gpuEvaluationUsage.requests,
                        g_gpuEvaluationUsage.dependencyPassRequests,
                        g_gpuEvaluationUsage.reverseOwners,
                        g_gpuEvaluationUsage.candidateOwners,
                        g_gpuEvaluationUsage.encodedHits,
                        g_gpuEvaluationUsage.localValues,
                        g_gpuEvaluationUsage.headRecords,
                        g_gpuEvaluationUsage.markerRecords,
                        g_gpuEvaluationUsage.demandRecords,
                        g_gpuEvaluationUsage.generatedBytes,
                        g_gpuEvaluationUsage.levelValues,
                        g_gpuEvaluationUsage.originDependencies,
                        g_gpuEvaluationUsage.markerKeys,
                        g_gpuEvaluationUsage.markerRemainingArgs,
                        g_gpuEvaluationUsage.markerArgs
                    };
                    for (std::size_t metric = 0;
                         metric < kGpuEvaluationTotalCount; ++metric) {
                        gpuEvaluationTotals[metric].fetch_add(
                            evaluationTotals[metric],
                            std::memory_order_relaxed);
                    }
                    const uint64_t evaluationMaximums[
                        kGpuEvaluationMaximumCount] = {
                        g_gpuEvaluationUsage.maximumReverseOwnersPerRequest,
                        g_gpuEvaluationUsage.maximumCandidateOwnersPerRequest,
                        g_gpuEvaluationUsage.maximumLocalValuesPerHit
                    };
                    for (std::size_t metric = 0;
                         metric < kGpuEvaluationMaximumCount; ++metric) {
                        uint64_t observed = gpuEvaluationMaximums[metric].load(
                            std::memory_order_relaxed);
                        while (observed < evaluationMaximums[metric]
                            && !gpuEvaluationMaximums[metric].compare_exchange_weak(
                                observed, evaluationMaximums[metric],
                                std::memory_order_relaxed,
                                std::memory_order_relaxed)) {
                        }
                    }
#if RT_MEASUREMENT
                    // [RT phase measurement]
                    taskNs[i] = std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - trapT0).count();
#endif
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

#ifdef GL_CUDA
        if (phase2Backend == Phase2Backend::cuda) {
            // Named audited capacity profiles (gpu/phase2_projection.hpp,
            // gpu/phase2_cuda.hpp) replace direct numeric assignments here.
            // The FTA shortcut keeps its compact measured image; every other
            // anchor runs the broad full-run image. One process owns exactly
            // one profile because the device owners below are static.
            const bool mainRunCapacity = anchorInfo.name != "AnchorFTA";
            const gpu::Phase2ProjectionProfile projectionProfile =
                mainRunCapacity
                    ? gpu::Phase2ProjectionProfile::fullRun
                    : gpu::Phase2ProjectionProfile::ftaShortcut;
            const gpu::Phase2ProjectionCapacity projectionCapacity =
                gpu::phase2ProjectionCapacityFor(projectionProfile);
            gpu::Phase2ProjectionCapacity projectionShardCapacity =
                gpu::phase2ProjectionCapacityFor(
                    gpu::Phase2ProjectionProfile::ftaShortcut);
            if (mainRunCapacity) {
                // Twelve construction workers share a dynamically claimed
                // chunk. Each shard owns two average shares of every resident
                // column so one data-heavy logical block cannot overflow its
                // worker while the complete merged chunk still fits the device.
                const auto constructionShardCeiling = [](uint32_t original,
                    uint32_t complete) {
                    return std::max(original,
                        complete
                                / gpu::kPhase2ProjectionConstructionShardShareDivisor
                            + static_cast<uint32_t>(static_cast<bool>(complete
                                % gpu::kPhase2ProjectionConstructionShardShareDivisor)));
                };
                projectionShardCapacity.logicalBlocks =
                    constructionShardCeiling(
                        projectionShardCapacity.logicalBlocks,
                        projectionCapacity.logicalBlocks);
                projectionShardCapacity.statements = constructionShardCeiling(
                    projectionShardCapacity.statements,
                    projectionCapacity.statements);
                projectionShardCapacity.nameRecords = constructionShardCeiling(
                    projectionShardCapacity.nameRecords,
                    projectionCapacity.nameRecords);
                projectionShardCapacity.nameBytes = constructionShardCeiling(
                    projectionShardCapacity.nameBytes,
                    projectionCapacity.nameBytes);
                projectionShardCapacity.nameSlots = constructionShardCeiling(
                    projectionShardCapacity.nameSlots,
                    projectionCapacity.nameSlots);
                projectionShardCapacity.ruleStringRecords =
                    constructionShardCeiling(
                        projectionShardCapacity.ruleStringRecords,
                        projectionCapacity.ruleStringRecords);
                projectionShardCapacity.ruleStringBytes =
                    constructionShardCeiling(
                        projectionShardCapacity.ruleStringBytes,
                        projectionCapacity.ruleStringBytes);
                projectionShardCapacity.byteMapViews =
                    constructionShardCeiling(
                        projectionShardCapacity.byteMapViews,
                        projectionCapacity.byteMapViews);
                projectionShardCapacity.byteMapEntries =
                    constructionShardCeiling(
                        projectionShardCapacity.byteMapEntries,
                        projectionCapacity.byteMapEntries);
                projectionShardCapacity.byteMapSlots =
                    constructionShardCeiling(
                        projectionShardCapacity.byteMapSlots,
                        projectionCapacity.byteMapSlots);
                projectionShardCapacity.byteKeyBytes =
                    constructionShardCeiling(
                        projectionShardCapacity.byteKeyBytes,
                        projectionCapacity.byteKeyBytes);
                projectionShardCapacity.blobRecords = constructionShardCeiling(
                    projectionShardCapacity.blobRecords,
                    projectionCapacity.blobRecords);
                projectionShardCapacity.blobBytes = constructionShardCeiling(
                    projectionShardCapacity.blobBytes,
                    projectionCapacity.blobBytes);
                projectionShardCapacity.reverseMapViews =
                    constructionShardCeiling(
                        projectionShardCapacity.reverseMapViews,
                        projectionCapacity.reverseMapViews);
                projectionShardCapacity.reverseMapEntries =
                    constructionShardCeiling(
                        projectionShardCapacity.reverseMapEntries,
                        projectionCapacity.reverseMapEntries);
                projectionShardCapacity.reverseMapSlots =
                    constructionShardCeiling(
                        projectionShardCapacity.reverseMapSlots,
                        projectionCapacity.reverseMapSlots);
                projectionShardCapacity.reverseKeyBytes =
                    constructionShardCeiling(
                        projectionShardCapacity.reverseKeyBytes,
                        projectionCapacity.reverseKeyBytes);
                projectionShardCapacity.reverseOwners =
                    constructionShardCeiling(
                        projectionShardCapacity.reverseOwners,
                        projectionCapacity.reverseOwners);
                projectionShardCapacity.podMapViews = constructionShardCeiling(
                    projectionShardCapacity.podMapViews,
                    projectionCapacity.podMapViews);
                projectionShardCapacity.podMapEntries =
                    constructionShardCeiling(
                        projectionShardCapacity.podMapEntries,
                        projectionCapacity.podMapEntries);
                projectionShardCapacity.podMapSlots = constructionShardCeiling(
                    projectionShardCapacity.podMapSlots,
                    projectionCapacity.podMapSlots);
                projectionShardCapacity.podRunValues =
                    constructionShardCeiling(
                        projectionShardCapacity.podRunValues,
                        projectionCapacity.podRunValues);
                projectionShardCapacity.mandatoryStatementKeys =
                    constructionShardCeiling(
                        projectionShardCapacity.mandatoryStatementKeys,
                        projectionCapacity.mandatoryStatementKeys);
                projectionShardCapacity.metadataBytes =
                    constructionShardCeiling(
                        projectionShardCapacity.metadataBytes,
                        projectionCapacity.metadataBytes);
            }

            const gpu::Phase2TaskProjectionCapacity taskCapacity =
                gpu::kPhase2TaskProjectionCapacity;
            const gpu::Phase2FilterScheduleCapacity filterCapacity =
                gpu::kPhase2FilterScheduleCapacity;
            // A single split Gauss task family owns more than one million live
            // breadth-frontier nodes. Growth and ordering therefore share one
            // named accepted-event ceiling for the indivisible family.
            const gpu::Phase2GrowthCapacity growthCapacity = mainRunCapacity
                ? gpu::kFullRunPhase2GrowthCapacity
                : gpu::kFtaShortcutPhase2GrowthCapacity;
            const gpu::Phase2OrderingCapacity orderingCapacity = mainRunCapacity
                ? gpu::kFullRunPhase2OrderingCapacity
                : gpu::kFtaShortcutPhase2OrderingCapacity;
            // Full-run incubator bursts materialize marker work at request scale;
            // the shortcut profile retains its independently audited columns.
            const gpu::Phase2EvaluationCapacity evaluationCapacity =
                mainRunCapacity
                    ? gpu::kFullRunPhase2EvaluationCapacity
                    : gpu::kFtaShortcutPhase2EvaluationCapacity;

            // High-volume gate counters remain isolated from proof flow and are
            // disabled after fixing the production capacities from the census.
            constexpr bool kCollectPhase2GrowthCensus = false;

            // These owners initialize only on the first selected CUDA pass and
            // retain every host and device allocation until process exit.
            static const gpu::Phase2ProjectionProfile fixedProjectionProfile =
                projectionProfile;
            assert(fixedProjectionProfile == projectionProfile
                && "CUDA Phase 2 static owners require one capacity profile "
                   "per process");
            constexpr std::size_t kGpuProjectionShardCount =
                gpu::kPhase2ProjectionConstructionShardCount;
            static gpu::Phase2ProjectionArena hostProjection(
                projectionCapacity);
            static std::array<gpu::Phase2ProjectionArena,
                kGpuProjectionShardCount> projectionShards{
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity),
                    gpu::Phase2ProjectionArena(projectionShardCapacity) };
            static gpu::CudaPhase2ProjectionBuffer deviceProjection(
                projectionCapacity);
            static gpu::Phase2TaskProjectionArena hostTasks(taskCapacity);
            static gpu::CudaPhase2TaskBuffer deviceTasks(taskCapacity);
            static gpu::Phase2FilterScheduleArena filterSchedule(
                filterCapacity);
            static gpu::CudaPhase2FilterSortBuffer deviceFilter(
                filterCapacity);
            static gpu::Phase2GrowthScheduleArena growthSchedule(
                growthCapacity.calls);
            static gpu::CudaPhase2GrowthBuffer deviceGrowth(growthCapacity);
            static gpu::CudaPhase2OrderingBuffer deviceOrdering(
                orderingCapacity);
            static gpu::CudaPhase2EvaluationBuffer deviceEvaluation(
                evaluationCapacity);
            static gpu::Phase2SealingArena sealing(evaluationCapacity);

            static const bool fixedDeviceOwnershipReported = [&]() {
                const uint64_t projectionBytes =
                    deviceProjection.fixedAllocationBytes();
                const uint64_t taskBytes = deviceTasks.fixedAllocationBytes();
                const uint64_t filterBytes = deviceFilter.fixedAllocationBytes();
                const uint64_t growthBytes = deviceGrowth.fixedAllocationBytes();
                const uint64_t orderingBytes =
                    deviceOrdering.fixedAllocationBytes();
                const uint64_t evaluationBytes =
                    deviceEvaluation.fixedAllocationBytes();
                const uint64_t totalBytes = projectionBytes + taskBytes
                    + filterBytes + growthBytes + orderingBytes
                    + evaluationBytes;
                diagnosticsLog() << "[GPU-PHASE2-MEMORY] projection_bytes="
                    << projectionBytes
                    << " task_bytes=" << taskBytes
                    << " filter_bytes=" << filterBytes
                    << " growth_bytes=" << growthBytes
                    << " ordering_bytes=" << orderingBytes
                    << " evaluation_bytes=" << evaluationBytes
                    << " total_fixed_device_bytes=" << totalBytes << std::endl;
                return true;
            }();
            static_cast<void>(fixedDeviceOwnershipReported);

            // A full run can expose more state than shortcut's one projection.
            // Keep every split LB's task family together and greedily fill a
            // deterministic chunk against every fixed projection/task column.
            // No projected state survives a chunk boundary.
            const auto addProjectionUsage = [](
                gpu::Phase2ProjectionUsage& total,
                const gpu::Phase2ProjectionUsage& add) {
                total.logicalBlocks += add.logicalBlocks;
                total.statements += add.statements;
                total.nameRecords += add.nameRecords;
                total.nameBytes += add.nameBytes;
                total.nameSlots += add.nameSlots;
                total.ruleStringRecords += add.ruleStringRecords;
                total.ruleStringBytes += add.ruleStringBytes;
                total.byteMapViews += add.byteMapViews;
                total.byteMapEntries += add.byteMapEntries;
                total.byteMapSlots += add.byteMapSlots;
                total.byteKeyBytes += add.byteKeyBytes;
                total.blobRecords += add.blobRecords;
                total.blobBytes += add.blobBytes;
                total.reverseMapViews += add.reverseMapViews;
                total.reverseMapEntries += add.reverseMapEntries;
                total.reverseMapSlots += add.reverseMapSlots;
                total.reverseKeyBytes += add.reverseKeyBytes;
                total.reverseOwners += add.reverseOwners;
                total.podMapViews += add.podMapViews;
                total.podMapEntries += add.podMapEntries;
                total.podMapSlots += add.podMapSlots;
                total.podRunValues += add.podRunValues;
                total.mandatoryStatementKeys += add.mandatoryStatementKeys;
                total.metadataBytes += add.metadataBytes;
            };
            const auto projectionFits = [&projectionCapacity](
                const gpu::Phase2ProjectionUsage& total,
                const gpu::Phase2ProjectionUsage& add) {
                return total.logicalBlocks + add.logicalBlocks
                           <= projectionCapacity.logicalBlocks
                    && total.statements + add.statements
                           <= projectionCapacity.statements
                    && total.nameRecords + add.nameRecords
                           <= projectionCapacity.nameRecords
                    && total.nameBytes + add.nameBytes
                           <= projectionCapacity.nameBytes
                    && total.nameSlots + add.nameSlots
                           <= projectionCapacity.nameSlots
                    && total.ruleStringRecords + add.ruleStringRecords
                           <= projectionCapacity.ruleStringRecords
                    && total.ruleStringBytes + add.ruleStringBytes
                           <= projectionCapacity.ruleStringBytes
                    && total.byteMapViews + add.byteMapViews
                           <= projectionCapacity.byteMapViews
                    && total.byteMapEntries + add.byteMapEntries
                           <= projectionCapacity.byteMapEntries
                    && total.byteMapSlots + add.byteMapSlots
                           <= projectionCapacity.byteMapSlots
                    && total.byteKeyBytes + add.byteKeyBytes
                           <= projectionCapacity.byteKeyBytes
                    && total.blobRecords + add.blobRecords
                           <= projectionCapacity.blobRecords
                    && total.blobBytes + add.blobBytes
                           <= projectionCapacity.blobBytes
                    && total.reverseMapViews + add.reverseMapViews
                           <= projectionCapacity.reverseMapViews
                    && total.reverseMapEntries + add.reverseMapEntries
                           <= projectionCapacity.reverseMapEntries
                    && total.reverseMapSlots + add.reverseMapSlots
                           <= projectionCapacity.reverseMapSlots
                    && total.reverseKeyBytes + add.reverseKeyBytes
                           <= projectionCapacity.reverseKeyBytes
                    && total.reverseOwners + add.reverseOwners
                           <= projectionCapacity.reverseOwners
                    && total.podMapViews + add.podMapViews
                           <= projectionCapacity.podMapViews
                    && total.podMapEntries + add.podMapEntries
                           <= projectionCapacity.podMapEntries
                    && total.podMapSlots + add.podMapSlots
                           <= projectionCapacity.podMapSlots
                    && total.podRunValues + add.podRunValues
                           <= projectionCapacity.podRunValues
                    && total.mandatoryStatementKeys
                           + add.mandatoryStatementKeys
                           <= projectionCapacity.mandatoryStatementKeys
                    && total.metadataBytes + add.metadataBytes
                           <= projectionCapacity.metadataBytes;
            };
            const auto addTaskUsage = [](
                gpu::Phase2TaskProjectionUsage& total,
                const gpu::Phase2TaskProjectionUsage& add) {
                total.tasks += add.tasks;
                total.batches += add.batches;
                total.terms += add.terms;
                total.stumps += add.stumps;
            };
            const auto taskFits = [&taskCapacity](
                const gpu::Phase2TaskProjectionUsage& total,
                const gpu::Phase2TaskProjectionUsage& add) {
                return total.tasks + add.tasks <= taskCapacity.tasks
                    && total.batches + add.batches <= taskCapacity.batches
                    && total.terms + add.terms <= taskCapacity.terms
                    && total.stumps + add.stumps <= taskCapacity.stumps;
            };
            const auto hashMemoryBit = [](gpu::DeviceHashMemoryKind memory) {
                return 1u << static_cast<uint32_t>(memory);
            };
            const auto selectedHashMemoriesFor = [this, &hashMemoryBit](
                const Memory& body) {
                if (ceFilteringActive) {
                    return hashMemoryBit(
                        gpu::DeviceHashMemoryKind::overall);
                }
                uint32_t selected = 0;
                if (!body.workingMemory.encodedMap.empty()
                    && !body.intLocalEncodedStatements.empty()) {
                    selected |= hashMemoryBit(
                        gpu::DeviceHashMemoryKind::working);
                }
                if (!body.intLocalEncodedStatementsDelta.empty()
                    || (!body.intExternalStatements.empty()
                        && !body.intLocalEncodedStatements.empty())) {
                    selected |= hashMemoryBit(
                        gpu::DeviceHashMemoryKind::overall);
                }
                if (!body.localHashMemory.encodedMap.empty()
                    && !body.intExternalStatements.empty()) {
                    selected |= hashMemoryBit(
                        gpu::DeviceHashMemoryKind::local);
                }
                if (!body.localHashMemoryDelta.encodedMap.empty()) {
                    selected |= hashMemoryBit(
                        gpu::DeviceHashMemoryKind::localDelta);
                }
                return selected;
            };
            std::vector<std::vector<std::size_t>> tasksByLogicalBlock(M);
            std::vector<gpu::Phase2TaskProjectionUsage> cudaTaskUsages(
                tasks.size());
            std::vector<uint32_t> selectedHashMemoriesByLi(M, 0);
            for (std::size_t taskPosition = 0;
                 taskPosition < tasks.size(); ++taskPosition) {
                const ExecTask& task = tasks[taskPosition];
                if (task.produceOnly) continue;
                cudaTaskUsages[taskPosition] =
                    gpu::measurePhase2TaskProjectionUsage(
                        *task.lb,
                        static_cast<uint32_t>(task.stump.count),
                        ceFilteringActive);
                if (cudaTaskUsages[taskPosition].tasks != 0) {
                    const uint32_t selected =
                        selectedHashMemoriesFor(*task.lb);
                    assert(selected != 0);
                    if (selectedHashMemoriesByLi[task.li] == 0)
                        selectedHashMemoriesByLi[task.li] = selected;
                    else
                        assert(selectedHashMemoriesByLi[task.li] == selected);
                    tasksByLogicalBlock[task.li].push_back(taskPosition);
                }
            }
            std::vector<std::vector<std::size_t>> cudaTaskChunks;
            std::vector<std::size_t> cudaTaskChunk;
            cudaTaskChunk.reserve(taskCapacity.tasks);
            gpu::Phase2ProjectionUsage chunkProjectionUsage{};
            gpu::Phase2TaskProjectionUsage chunkTaskUsage{};
            gpu::Phase2ProjectionUsage passProjectionUsage{};
            gpu::Phase2TaskProjectionUsage passTaskUsage{};
            for (std::size_t taskPosition = 0;
                 taskPosition < tasks.size(); ++taskPosition) {
                const ExecTask& task = tasks[taskPosition];
                if (task.produceOnly) continue;
                std::vector<std::size_t>& family =
                    tasksByLogicalBlock[task.li];
                if (family.empty()) continue;
                assert(task.lb->lbMemory.manager.resident()
                    && "CUDA projection packing requires resident state");
                const uint32_t selectedHashMemories =
                    selectedHashMemoriesByLi[task.li];
                assert(selectedHashMemories != 0);
                const gpu::Phase2ProjectionUsage familyProjectionUsage =
                    gpu::measurePhase2ProjectionUsage(
                        *task.lb, *this, selectedHashMemories);
                gpu::Phase2TaskProjectionUsage familyTaskUsage{};
                for (const std::size_t familyTaskPosition : family) {
                    addTaskUsage(familyTaskUsage,
                        cudaTaskUsages[familyTaskPosition]);
                }
                if (!cudaTaskChunk.empty()
                    && (!projectionFits(
                            chunkProjectionUsage, familyProjectionUsage)
                        || !taskFits(chunkTaskUsage, familyTaskUsage))) {
                    cudaTaskChunks.push_back(std::move(cudaTaskChunk));
                    cudaTaskChunk.clear();
                    cudaTaskChunk.reserve(taskCapacity.tasks);
                    chunkProjectionUsage = gpu::Phase2ProjectionUsage{};
                    chunkTaskUsage = gpu::Phase2TaskProjectionUsage{};
                }
                assert(projectionFits(
                           chunkProjectionUsage, familyProjectionUsage)
                    && "one logical block's CUDA projection exceeds capacity");
                assert(taskFits(chunkTaskUsage, familyTaskUsage)
                    && "one logical block's CUDA task family exceeds capacity");
                cudaTaskChunk.insert(
                    cudaTaskChunk.end(), family.begin(), family.end());
                addProjectionUsage(
                    chunkProjectionUsage, familyProjectionUsage);
                addTaskUsage(chunkTaskUsage, familyTaskUsage);
                addProjectionUsage(
                    passProjectionUsage, familyProjectionUsage);
                addTaskUsage(passTaskUsage, familyTaskUsage);
                family.clear();
            }
            if (!cudaTaskChunk.empty())
                cudaTaskChunks.push_back(std::move(cudaTaskChunk));

            diagnosticsLog() << "[GPU-PACKING] pass=" << passNo
                      << " chunks=" << cudaTaskChunks.size()
                      << " logical_blocks="
                      << passProjectionUsage.logicalBlocks
                      << " statements=" << passProjectionUsage.statements
                      << " name_records=" << passProjectionUsage.nameRecords
                      << " name_bytes=" << passProjectionUsage.nameBytes
                      << " name_slots=" << passProjectionUsage.nameSlots
                      << " rule_string_records="
                      << passProjectionUsage.ruleStringRecords
                      << " rule_string_bytes="
                      << passProjectionUsage.ruleStringBytes
                      << " byte_map_views="
                      << passProjectionUsage.byteMapViews
                      << " byte_map_entries="
                      << passProjectionUsage.byteMapEntries
                      << " byte_map_slots="
                      << passProjectionUsage.byteMapSlots
                      << " byte_key_bytes="
                      << passProjectionUsage.byteKeyBytes
                      << " blob_records="
                      << passProjectionUsage.blobRecords
                      << " blob_bytes=" << passProjectionUsage.blobBytes
                      << " reverse_map_views="
                      << passProjectionUsage.reverseMapViews
                      << " reverse_map_entries="
                      << passProjectionUsage.reverseMapEntries
                      << " reverse_map_slots="
                      << passProjectionUsage.reverseMapSlots
                      << " reverse_key_bytes="
                      << passProjectionUsage.reverseKeyBytes
                      << " reverse_owners="
                      << passProjectionUsage.reverseOwners
                      << " pod_map_views="
                      << passProjectionUsage.podMapViews
                      << " pod_map_entries="
                      << passProjectionUsage.podMapEntries
                      << " pod_map_slots="
                      << passProjectionUsage.podMapSlots
                      << " pod_run_values="
                      << passProjectionUsage.podRunValues
                      << " mandatory_statement_keys="
                      << passProjectionUsage.mandatoryStatementKeys
                      << " metadata_bytes="
                      << passProjectionUsage.metadataBytes
                      << " tasks=" << passTaskUsage.tasks
                      << " batches=" << passTaskUsage.batches
                      << " terms=" << passTaskUsage.terms
                      << " stumps=" << passTaskUsage.stumps
                      << std::endl;

            for (const std::vector<std::size_t>& cudaTaskPositions
                 : cudaTaskChunks) {
            const auto gpuPassStarted = std::chrono::steady_clock::now();
            hostProjection.clear();
            for (gpu::Phase2ProjectionArena& shard : projectionShards)
                shard.clear();
            hostTasks.clear();
            filterSchedule.clear();
            growthSchedule.clear();

            std::vector<char> projectionSeen(M, 0);
            std::vector<std::size_t> projectionLis;
            std::vector<Memory*> projectionOrder;
            std::vector<uint32_t> projectionSelectedHashMemories;
            std::array<std::size_t, gpu::kMaxProjectedBlocksPerChunk>
                outputTaskByBlock{};
            for (const std::size_t taskPosition : cudaTaskPositions) {
                const ExecTask& task = tasks[taskPosition];
                if (projectionSeen[task.li] != 0) continue;
                projectionSeen[task.li] = 1;
                assert(projectionLis.size()
                    < projectionCapacity.logicalBlocks);
                projectionLis.push_back(task.li);
                projectionOrder.push_back(task.lb);
                assert(selectedHashMemoriesByLi[task.li] != 0);
                projectionSelectedHashMemories.push_back(
                    selectedHashMemoriesByLi[task.li]);
                outputTaskByBlock[projectionLis.size() - 1] = taskPosition;
                gpuOutputOwners[taskPosition] = 1;
            }

            if (!projectionOrder.empty()) {
                std::vector<int32_t> projectedBlockByLi(M, -1);
                std::atomic<std::size_t> projectionCursor{ 0 };
                steward->beginPhaseWindow(
                    /*phase=*/2, &projectionCursor, &projectionOrder,
                    static_cast<unsigned>(kGpuProjectionShardCount),
                    lbdeload::kDeloadDirectory);
                std::array<std::array<uint32_t,
                        gpu::kMaxProjectedBlocksPerChunk>,
                    kGpuProjectionShardCount> shardBlockIndices{};
                std::array<uint32_t, kGpuProjectionShardCount>
                    shardBlockCounts{};
                const auto projectionWorker = [this, &projectionCursor,
                    &projectionOrder, &projectionSelectedHashMemories,
                    &shardBlockIndices,
                    &shardBlockCounts](std::size_t shardIndex) {
                    gpu::Phase2ProjectionArena& shard =
                        projectionShards[shardIndex];
                    for (;;) {
                        const std::size_t blockIndex = projectionCursor.fetch_add(
                            1, std::memory_order_relaxed);
                        if (blockIndex >= projectionOrder.size()) break;
                        Memory* logicalBlock = projectionOrder[blockIndex];
                        this->steward->claimAndLoadForWork(
                            *logicalBlock, /*phase=*/2,
                            lbdeload::kDeloadDirectory);
                        const uint32_t localBlockIndex =
                            static_cast<uint32_t>(shard.logicalBlocks.size());
                        assert(localBlockIndex < shardBlockIndices[shardIndex].size());
                        const gpu::DeviceLogicalBlockProjection projected =
                            shard.appendLogicalBlock(
                                *logicalBlock, *this,
                                projectionSelectedHashMemories[blockIndex]);
                        assert(projected.statementCount
                            == static_cast<uint32_t>(
                                logicalBlock->intEncodedStatements.size()));
                        shardBlockIndices[shardIndex][localBlockIndex] =
                            static_cast<uint32_t>(blockIndex);
                        assert(logicalBlock->stewardClaim.load(
                                   std::memory_order_relaxed)
                               == static_cast<uint8_t>(
                                      Memory::StewardClaim::WorkerOwned));
                        logicalBlock->stewardClaim.store(
                            static_cast<uint8_t>(Memory::StewardClaim::Idle),
                            std::memory_order_release);
                    }
                    shardBlockCounts[shardIndex] =
                        static_cast<uint32_t>(shard.logicalBlocks.size());
                };
                std::array<std::thread, kGpuProjectionShardCount>
                    projectionThreads;
                for (std::size_t shardIndex = 0;
                     shardIndex < kGpuProjectionShardCount; ++shardIndex) {
                    projectionThreads[shardIndex] = std::thread(
                        projectionWorker, shardIndex);
                }
                for (std::thread& thread : projectionThreads) thread.join();
                steward->endPhaseWindow();

                std::array<gpu::DeviceLogicalBlockProjection,
                    gpu::kMaxProjectedBlocksPerChunk>
                    canonicalDescriptors{};
                std::array<uint8_t, gpu::kMaxProjectedBlocksPerChunk>
                    canonicalDescriptorSeen{};
                std::size_t mergedBlockCount = 0;
                for (std::size_t shardIndex = 0;
                     shardIndex < kGpuProjectionShardCount; ++shardIndex) {
                    const uint32_t destinationBlockOffset =
                        hostProjection.appendShard(projectionShards[shardIndex]);
                    assert(destinationBlockOffset == mergedBlockCount);
                    for (uint32_t localBlockIndex = 0;
                         localBlockIndex < shardBlockCounts[shardIndex];
                         ++localBlockIndex) {
                        const uint32_t blockIndex =
                            shardBlockIndices[shardIndex][localBlockIndex];
                        assert(blockIndex < projectionOrder.size());
                        assert(canonicalDescriptorSeen[blockIndex] == 0);
                        canonicalDescriptorSeen[blockIndex] = 1;
                        canonicalDescriptors[blockIndex] =
                            hostProjection.logicalBlocks[
                                destinationBlockOffset + localBlockIndex];
                    }
                    mergedBlockCount += shardBlockCounts[shardIndex];
                }
                assert(mergedBlockCount == projectionOrder.size());
                assert(hostProjection.logicalBlocks.size()
                    == projectionOrder.size());
                for (std::size_t blockIndex = 0;
                     blockIndex < projectionOrder.size(); ++blockIndex) {
                    assert(canonicalDescriptorSeen[blockIndex] == 1);
                    hostProjection.logicalBlocks[blockIndex] =
                        canonicalDescriptors[blockIndex];
                    assert(projectedBlockByLi[projectionLis[blockIndex]] < 0);
                    projectedBlockByLi[projectionLis[blockIndex]] =
                        static_cast<int32_t>(blockIndex);
                }
                const auto gpuProjectionFinished =
                    std::chrono::steady_clock::now();
                deviceProjection.beginPhase2Upload(hostProjection);

                std::vector<Memory*> scheduleOrder;
                scheduleOrder.reserve(cudaTaskPositions.size());
                for (const std::size_t taskPosition : cudaTaskPositions)
                    scheduleOrder.push_back(tasks[taskPosition].lb);
                std::atomic<std::size_t> scheduleCursor{ 0 };
                steward->beginPhaseWindow(
                    /*phase=*/2, &scheduleCursor, &scheduleOrder, 1,
                    lbdeload::kDeloadDirectory);
                std::size_t scheduleIndex = 0;
                for (const std::size_t taskPosition : cudaTaskPositions) {
                    const ExecTask& task = tasks[taskPosition];
                    scheduleCursor.store(
                        scheduleIndex, std::memory_order_relaxed);
                    steward->claimAndLoadForWork(
                        *task.lb, /*phase=*/2,
                        lbdeload::kDeloadDirectory);
                    const int32_t blockIndexSigned =
                        projectedBlockByLi[task.li];
                    assert(blockIndexSigned >= 0);
                    const uint32_t blockIndex =
                        static_cast<uint32_t>(blockIndexSigned);

                    // Match the processor's hard staged-arrival invariant before
                    // the device sees the immutable projection.
                    for (int32_t index = 0;
                         index < task.lb->intExternalStatements.size(); ++index) {
                        const IntEncodedExpr& external =
                            task.lb->intExternalStatements[index];
                        if (external.maxIteration
                            > parameters.maxIterationNumberVariable) continue;
                        assert(task.lb->intKnownStatements.find(
                                   StatementKey{
                                       external.originalId,
                                       external.validityId }) != nullptr
                               && "a staged mail arrival must remain registered "
                                  "before CUDA Phase 2 projection");
                    }

                    std::array<gpu::Phase2RequestBatchInput, 4> batches{};
                    uint32_t batchCount = 0;
                    if (ceFilteringActive) {
                        assert(task.partCount == 1 && task.stump.count == 0);
                        batches[0].kind =
                            gpu::DeviceRequestBatchKind::counterExample;
                        batches[0].memory =
                            gpu::DeviceHashMemoryKind::overall;
                        batchCount = 1;
                    } else {
                        if (!task.lb->workingMemory.encodedMap.empty()
                            && !task.lb->intLocalEncodedStatements.empty()) {
                            gpu::Phase2RequestBatchInput& input =
                                batches[batchCount++];
                            input.kind =
                                gpu::DeviceRequestBatchKind::workingRules;
                            input.memory = gpu::DeviceHashMemoryKind::working;
                            input.terms[0].views[0] =
                                gpu::DeviceMandatoryViewKind::local;
                            input.terms[0].viewCount = 1;
                            input.termCount = 1;
                        }

                        const bool hasNewThisBurst =
                            !task.lb->intLocalEncodedStatementsDelta.empty()
                            || (!task.lb->intExternalStatements.empty()
                                && !task.lb->intLocalEncodedStatements.empty());
                        if (hasNewThisBurst) {
                            gpu::Phase2RequestBatchInput& input =
                                batches[batchCount++];
                            input.kind =
                                gpu::DeviceRequestBatchKind::newThisBurst;
                            input.memory = gpu::DeviceHashMemoryKind::overall;
                            input.terms[0].views[0] =
                                gpu::DeviceMandatoryViewKind::localDelta;
                            input.terms[0].viewCount = 1;
                            input.terms[1].views[0] =
                                gpu::DeviceMandatoryViewKind::external;
                            input.terms[1].views[1] =
                                gpu::DeviceMandatoryViewKind::local;
                            input.terms[1].viewCount = 2;
                            input.termCount = 2;
                        }

                        if (!task.lb->localHashMemory.encodedMap.empty()
                            && !task.lb->intExternalStatements.empty()) {
                            gpu::Phase2RequestBatchInput& input =
                                batches[batchCount++];
                            input.kind = gpu::DeviceRequestBatchKind::
                                localRulesWithMail;
                            input.memory = gpu::DeviceHashMemoryKind::local;
                            input.terms[0].views[0] =
                                gpu::DeviceMandatoryViewKind::external;
                            input.terms[0].viewCount = 1;
                            input.termCount = 1;
                        }

                        if (!task.lb->localHashMemoryDelta.encodedMap.empty()) {
                            gpu::Phase2RequestBatchInput& input =
                                batches[batchCount++];
                            input.kind =
                                gpu::DeviceRequestBatchKind::localDeltaRules;
                            input.memory = gpu::DeviceHashMemoryKind::localDelta;
                        }
                    }
                    assert(batchCount > 0 && batchCount <= batches.size());
                    const uint32_t selectedHashMemories =
                        selectedHashMemoriesByLi[task.li];
                    assert(selectedHashMemories != 0);
                    for (uint32_t localBatch = 0;
                         localBatch < batchCount; ++localBatch) {
                        assert((selectedHashMemories
                            & hashMemoryBit(batches[localBatch].memory)) != 0);
                    }

                    const uint32_t gpuTaskIndex =
                        static_cast<uint32_t>(hostTasks.tasks.size());
                    const uint32_t batchOffset =
                        static_cast<uint32_t>(hostTasks.batches.size());
                    hostTasks.appendTask(
                        blockIndex, batches.data(), batchCount,
                        task.stump.stumps,
                        static_cast<uint32_t>(task.stump.count),
                        task.stump.ordinal, task.stump.total,
                        parameters.maxIterationNumberVariable,
                        ceFilteringActive ? 1u : 0u);
                    gpuTaskIndices[taskPosition] =
                        static_cast<int32_t>(gpuTaskIndex);

                    const uint32_t statementCount = hostProjection.
                        logicalBlocks[blockIndex].statementCount;
                    for (uint32_t localBatch = 0;
                         localBatch < batchCount; ++localBatch) {
                        const uint32_t filterCallIndex =
                            static_cast<uint32_t>(filterSchedule.calls.size());
                        filterSchedule.appendCall(
                            blockIndex, batches[localBatch].memory,
                            parameters.maxIterationNumberVariable,
                            /*alsoAcceptFullKeys=*/1, statementCount);
                        growthSchedule.appendCall(
                            gpuTaskIndex, batchOffset + localBatch,
                            filterCallIndex);
                    }
                    assert(task.lb->stewardClaim.load(
                               std::memory_order_relaxed)
                           == static_cast<uint8_t>(
                                  Memory::StewardClaim::WorkerOwned));
                    task.lb->stewardClaim.store(
                        static_cast<uint8_t>(Memory::StewardClaim::Idle),
                        std::memory_order_release);
                    ++scheduleIndex;
                    scheduleCursor.store(
                        scheduleIndex, std::memory_order_relaxed);
                }
                assert(scheduleIndex == scheduleOrder.size());
                steward->endPhaseWindow();

                const auto gpuDeviceRouteStarted =
                    std::chrono::steady_clock::now();
                const double gpuPreparationSeconds =
                    std::chrono::duration<double>(
                        gpuDeviceRouteStarted - gpuPassStarted).count();
                const double gpuProjectionSeconds =
                    std::chrono::duration<double>(
                        gpuProjectionFinished - gpuPassStarted).count();
                const double gpuScheduleSeconds =
                    std::chrono::duration<double>(
                        gpuDeviceRouteStarted - gpuProjectionFinished).count();
                const gpu::Phase2FilterClassCensus filterClassCensus =
                    filterSchedule.measureClassReuse();

                // Process-owned CUDA events measure the task upload, the deferred
                // projection-upload join, and every semantic kernel through
                // doom-prefix selection. The projection upload starts before host
                // scheduling so complete Phase 2 timing captures their overlap.
                static gpu::CudaPhase2DeviceTimer gpuDeviceTimer;
                gpuDeviceTimer.start();

                deviceTasks.upload(hostTasks);
                deviceProjection.finishPhase2Upload();
                const uint32_t retainedRows = deviceFilter.filterAndSort(
                    deviceProjection, filterSchedule);
                const gpu::Phase2GrowthResult growthResult =
                    deviceGrowth.runRequestGrowth(
                        deviceProjection, deviceTasks, deviceFilter,
                        growthSchedule,
                        gpu::DevicePhase2GrowthParameters{
                            parameters.maxLenHypoKey,
                            parameters.maxNumberSecondaryVariables,
                            parameters.maxNumberSecondaryVariablesOrint,
                            gpu::kDeviceGrowthProductionCooperativeSpan,
                            kCollectPhase2GrowthCensus ? 2u : 0u });
                const gpu::Phase2OrderingResult orderingResult =
                    deviceOrdering.orderAndDeduplicate(
                        deviceProjection, deviceTasks, deviceGrowth);
                const gpu::Phase2EvaluationResult evaluationResult =
                    deviceEvaluation.expandEvaluationWork(
                        deviceProjection, deviceTasks, deviceGrowth,
                        deviceOrdering);
                const gpu::Phase2FiringExpressionResult firingResult =
                    deviceEvaluation.materializeFiringExpressions(
                        deviceProjection, deviceTasks, deviceGrowth,
                        deviceOrdering);
                assert(deviceEvaluation.orderFiringRecords(deviceProjection)
                    == firingResult.firingRecordCount);
                const uint32_t retainedFiringCount =
                    deviceEvaluation.selectDoomPrefixes(deviceProjection);

                const double gpuDeviceSeconds = gpuDeviceTimer.stopSeconds();
                const auto gpuFinalizeStarted =
                    std::chrono::steady_clock::now();
                const double gpuDeviceRouteSeconds =
                    std::chrono::duration<double>(
                        gpuFinalizeStarted - gpuDeviceRouteStarted).count();

                std::array<uint32_t, gpu::kMaxPhase2TasksPerChunk>
                    taskSubkeyCounts{};
                assert(deviceGrowth.downloadTaskSubkeyCounts(
                           taskSubkeyCounts.data(),
                           static_cast<uint32_t>(taskSubkeyCounts.size()))
                       == hostTasks.tasks.size());
                for (const std::size_t taskPosition : cudaTaskPositions) {
                    assert(gpuTaskIndices[taskPosition] >= 0);
                    taskSubMatches[taskPosition] = taskSubkeyCounts[
                        static_cast<uint32_t>(gpuTaskIndices[taskPosition])];
                }

                std::array<int64_t, gpu::kMaxProjectedBlocksPerChunk>
                    deviceDoomLines{};
                assert(deviceEvaluation.downloadDoomLines(
                           deviceDoomLines.data(),
                           static_cast<uint32_t>(deviceDoomLines.size()))
                       == projectionOrder.size());
                std::array<SealedPageSet*, gpu::kMaxProjectedBlocksPerChunk>
                    outputs{};
                for (std::size_t blockIndex = 0;
                     blockIndex < projectionOrder.size(); ++blockIndex) {
                    const std::size_t taskPosition =
                        outputTaskByBlock[blockIndex];
                    SealedPageSet& pages = pageStore[base + taskPosition];
                    pages.bind(&staticMemory());
                    outputs[blockIndex] = &pages;
                    const std::size_t li = projectionLis[blockIndex];
                    doomLines[li].store(
                        deviceDoomLines[blockIndex],
                        std::memory_order_relaxed);
                    lbFiringRecordsCanonical[li] = 1;
                }
                assert(sealing.downloadAndSeal(
                           deviceEvaluation, hostProjection, firingResult,
                           retainedFiringCount, outputs.data(),
                           static_cast<uint32_t>(projectionOrder.size()))
                       == retainedFiringCount);

                const auto gpuPassFinished = std::chrono::steady_clock::now();
                const double gpuFinalizeSeconds = std::chrono::duration<double>(
                    gpuPassFinished - gpuFinalizeStarted).count();
                const double gpuPassSeconds = std::chrono::duration<double>(
                    gpuPassFinished - gpuPassStarted).count();
                const gpu::Phase2ProjectionTiming& projectionTiming =
                    hostProjection.timing;
                assert(projectionTiming.logicalBlocks
                    == projectionOrder.size());
                constexpr double kNanosecondsPerSecond = 1000000000.0;
                const double projectionWorkerCpuSeconds =
                    static_cast<double>(
                        projectionTiming.preflightNanoseconds
                        + projectionTiming.statementNanoseconds
                        + projectionTiming.nameNanoseconds
                        + projectionTiming.ruleStringNanoseconds
                        + projectionTiming.byteMapNanoseconds
                         + projectionTiming.reverseMapNanoseconds
                         + projectionTiming.podMapNanoseconds
                         + projectionTiming.finalNanoseconds)
                    / kNanosecondsPerSecond;
                const double projectionMergeSeconds = static_cast<double>(
                    projectionTiming.mergeNanoseconds) / kNanosecondsPerSecond;
                diagnosticsLog() << "[GPU-PHASE2] pass=" << passNo
                          << " logical_blocks=" << projectionOrder.size()
                          << " tasks=" << hostTasks.tasks.size()
                          << " batches=" << hostTasks.batches.size()
                          << " filter_calls="
                          << filterClassCensus.callCount
                          << " filter_classes="
                          << filterClassCensus.uniqueClassCount
                          << " filter_duplicate_calls="
                          << filterClassCensus.duplicateCallCount
                          << " filter_class_rows="
                          << filterClassCensus.uniqueExaminedRows
                          << " filter_duplicate_rows="
                          << filterClassCensus.duplicateExaminedRows
                          << " filter_max_class_multiplicity="
                          << filterClassCensus.maximumClassMultiplicity
                          << " retained_rows=" << retainedRows
                          << " maximum_frontier="
                          << growthResult.maximumFrontierCount
                          << " accepted_events="
                          << growthResult.acceptedEventCount
                          << " unique_requests="
                          << orderingResult.uniqueRequestCount
                          << " dependency_pass_requests="
                          << evaluationResult.dependencyPassCount
                          << " reverse_owners="
                          << evaluationResult.reverseOwnerCount
                          << " candidate_owners="
                          << evaluationResult.candidateOwnerCount
                          << " encoded_hits="
                          << evaluationResult.encodedHitCount
                          << " local_values="
                          << evaluationResult.localValueCount
                          << " firing_records="
                          << firingResult.firingRecordCount
                          << " retained_firing_records="
                          << retainedFiringCount
                          << " generated_bytes="
                          << firingResult.generatedByteCount
                          << " level_values="
                          << firingResult.levelValueCount
                          << " provenance_dependencies="
                          << firingResult.originDependencyCount
                          << " cooperative_nodes_peak="
                          << growthResult.maximumCooperativeNodeCount
                          << " growth_prefix_payload_peak="
                          << growthResult.maximumPrefixPayloadValues
                          << " growth_prefix_variables_peak="
                          << growthResult.maximumPrefixVariableValues
                          << " growth_prefix_secondary_peak="
                          << growthResult.maximumPrefixSecondaryValues
                          << " marker_keys=" << firingResult.markerKeyCount
                          << " marker_remaining_args="
                          << firingResult.markerRemainingArgCount
                          << " marker_args=" << firingResult.markerArgCount;
                for (uint32_t bucket = 0;
                     bucket < gpu::kDeviceGrowthSpanBucketCount; ++bucket) {
                    diagnosticsLog() << " growth_span_nodes_b" << bucket << "="
                              << growthResult.spanNodeCounts[bucket]
                              << " growth_span_candidates_b" << bucket << "="
                              << growthResult.spanCandidateCounts[bucket];
                }
                for (uint32_t depth = 1;
                     depth < gpu::kDeviceGrowthCensusDepthCount; ++depth) {
                    const gpu::Phase2GrowthGateCensus& census =
                        growthResult.gateDepthCounts[depth];
                    diagnosticsLog() << " growth_d" << depth << "_attempts="
                              << census.candidateAttempts
                              << " growth_d" << depth << "_mandatory="
                              << census.mandatoryReachable
                              << " growth_d" << depth << "_validity="
                              << census.validityComparable
                              << " growth_d" << depth << "_hypothesis="
                              << census.hypothesisCompatible
                              << " growth_d" << depth << "_secondary="
                              << census.secondaryCompatible
                              << " growth_d" << depth << "_key_length="
                              << census.keyLengthAllowed
                              << " growth_d" << depth << "_subkey_present="
                              << census.subkeyPresent
                              << " growth_d" << depth << "_owner="
                              << census.ownerSatisfied
                              << " growth_d" << depth << "_whole="
                              << census.wholeKeyPresent
                              << " growth_d" << depth << "_terms="
                              << census.termsSatisfied
                              << " growth_d" << depth << "_events="
                              << census.acceptedEvents
                              << " growth_d" << depth << "_children="
                              << census.children;
                }
                diagnosticsLog() << " prepare_seconds=" << gpuPreparationSeconds
                          << " projection_seconds=" << gpuProjectionSeconds
                          << " projection_worker_cpu_seconds="
                          << projectionWorkerCpuSeconds
                          << " projection_merge_seconds="
                          << projectionMergeSeconds
                          << " projection_preflight_seconds="
                          << static_cast<double>(
                                 projectionTiming.preflightNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_statement_seconds="
                          << static_cast<double>(
                                 projectionTiming.statementNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_name_seconds="
                          << static_cast<double>(
                                 projectionTiming.nameNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_name_record_seconds="
                          << static_cast<double>(
                                 projectionTiming.nameRecordNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_name_sort_seconds="
                          << static_cast<double>(
                                 projectionTiming.nameSortNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_name_rank_seconds="
                          << static_cast<double>(
                                 projectionTiming.nameRankNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_name_slot_seconds="
                          << static_cast<double>(
                                 projectionTiming.nameSlotNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_rule_seconds="
                          << static_cast<double>(
                                 projectionTiming.ruleStringNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_byte_map_seconds="
                          << static_cast<double>(
                                 projectionTiming.byteMapNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_reverse_map_seconds="
                          << static_cast<double>(
                                 projectionTiming.reverseMapNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_pod_map_seconds="
                          << static_cast<double>(
                                 projectionTiming.podMapNanoseconds)
                                 / kNanosecondsPerSecond
                          << " projection_final_seconds="
                          << static_cast<double>(
                                 projectionTiming.finalNanoseconds)
                                 / kNanosecondsPerSecond
                          << " schedule_seconds=" << gpuScheduleSeconds
                          << " device_seconds=" << gpuDeviceSeconds
                          << " device_route_seconds=" << gpuDeviceRouteSeconds
                          << " finalize_seconds=" << gpuFinalizeSeconds
                          << " route_seconds=" << gpuPassSeconds << std::endl;
            }
        }
        }
#else
        assert(phase2Backend == Phase2Backend::cpu
            && "the CUDA Phase 2 backend requires a CUDA build "
               "(the Visual Studio project, or make USE_CUDA=1)");
#endif

        // Latch this pass's doom lines (post-join relaxed loads see the final
        // CAS-min values) before the next pass's reset can erase them.
        for (std::size_t li = 0; li < M; ++li)
            if (doomLines[li].load(std::memory_order_relaxed) != kNoDoomLine)
                lbDoomed[li] = 1;

        // ---- Classify every part, single-threaded: deal a producer's stumps into
        // buckets (requeue for round 2), or keep a burst part ----
        std::vector<ExecTask> nextTasks;
        for (std::size_t i = 0; i < tasks.size(); ++i) {
            const ExecTask& t = tasks[i];
            const std::size_t li = t.li;
#if RT_MEASUREMENT
            // [RT phase measurement] producer and burst parts both count.
            trapPh2Ns[li] += taskNs[i];
#endif
            SealedPageSet& ps = pageStore[base + i];

            // A round-1 PRODUCER. It fired nothing (ps is empty). Deal its stumps
            // into logicalCores expression buckets and requeue one bucket part each.
            if (t.produceOnly) {
                ps.freePages();
                if (taskStumps[i] == 0) {
                    // No statement survives the filter -> the LB's real burst would
                    // generate nothing either. Run it once, unsplit, this iteration
                    // (guarantees the burst happens; its work then feeds the stats).
                    nextTasks.push_back(ExecTask{ t.lb, li,
                        /*partCount=*/1, /*produceOnly=*/false,
                        SplitStumpRef{}, &doomLines[li], &partsRemaining[li] });
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
                    nextTasks.push_back(ExecTask{ t.lb, li,
                        /*partCount=*/buckets, /*produceOnly=*/false,
                        SplitStumpRef{ run.data() + lo,
                                       static_cast<NameId>(hi - lo), k, buckets },
                        &doomLines[li], &partsRemaining[li] });
                }
                continue;
            }

            // A burst part (an unsplit LB or one expression bucket): keep it and add
            // its submatch count to the LB's split-invariant TOTAL work (the straggler
            // classifier's input); lbMaxSub tracks the busiest single part for the
            // split-ineffective report.
            if (phase2Backend == Phase2Backend::cpu
                || gpuOutputOwners[i] != 0) {
                keptParts[li].push_back(&ps);
            }
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
                                   &doomLines, &lbFiringRecordsCanonical,
                                   &nextLi
#if RT_MEASUREMENT
                                   , &trapFinNs
#endif
                                   ](unsigned tIdx) {
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
                    // a phase-1-discharged LB: nothing deposits. The LB's final doom
                    // line rides along: when a doom trigger fired, the finalize
                    // merges only the winning part's chain. The pool has joined, so
                    // this relaxed load sees the final CAS-min value.
                    const int count = static_cast<int>(keptParts[li].size());
                    assert(count <= kMaxSplitParts
                        && "LB part count exceeds kMaxSplitParts - raise the named "
                           "constant deliberately, never cap the split silently");
#if RT_MEASUREMENT
                    // [RT phase measurement] finalize wall time; each li
                    // is finalized by exactly one worker (race-free).
                    const auto trapT0 = std::chrono::steady_clock::now();
#endif
                    const bool firingRecordsCanonical =
                        lbFiringRecordsCanonical[li] != 0;
                    this->performElemPhase2(
                        *b, keptParts[li].data(), count,
                        firingRecordsCanonical
                            ? kNoDoomLine
                            : doomLines[li].load(std::memory_order_relaxed),
                        firingRecordsCanonical);
#if RT_MEASUREMENT
                    trapFinNs[li] += std::chrono::duration_cast<
                        std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - trapT0).count();
#endif
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

    // GPU capacity projections go to the common diagnostics log so the
    // run log stays readable; one block per burst per batch.
    std::ostream& gpuCapacityLog = diagnosticsLog();
    gpuCapacityLog << "[GPU-CAPACITY] logical_blocks="
              << projectionLogicalBlocks.load(std::memory_order_relaxed)
              << " statements="
              << projectionStatements.load(std::memory_order_relaxed)
              << " maximum_statements_per_lb="
              << projectionMaximumStatements.load(std::memory_order_relaxed)
              << " name_records="
              << projectionNameRecords.load(std::memory_order_relaxed)
              << " maximum_name_records_per_lb="
              << projectionMaximumNameRecords.load(std::memory_order_relaxed)
              << " name_bytes="
              << projectionNameBytes.load(std::memory_order_relaxed)
              << " maximum_name_bytes_per_lb="
              << projectionMaximumNameBytes.load(std::memory_order_relaxed)
              << std::endl;
    const auto semanticTotal = [&](std::size_t metric) {
        return projectionSemanticTotals[metric].load(
            std::memory_order_relaxed);
    };
    const auto semanticMaximum = [&](std::size_t metric) {
        return projectionSemanticMaximums[metric].load(
            std::memory_order_relaxed);
    };
    gpuCapacityLog << "[GPU-SEMANTIC-CAPACITY] rule_string_records="
              << semanticTotal(0)
              << " maximum_rule_string_records_per_lb=" << semanticMaximum(0)
              << " rule_string_bytes=" << semanticTotal(1)
              << " maximum_rule_string_bytes_per_lb=" << semanticMaximum(1)
              << " name_slots=" << semanticTotal(2)
              << " maximum_name_slots_per_lb=" << semanticMaximum(2)
              << " byte_map_views=" << semanticTotal(3)
              << " byte_map_entries=" << semanticTotal(4)
              << " maximum_byte_map_entries_per_lb=" << semanticMaximum(4)
              << " byte_map_slots=" << semanticTotal(5)
              << " maximum_byte_map_slots_per_lb=" << semanticMaximum(5)
              << " byte_key_bytes=" << semanticTotal(6)
              << " maximum_byte_key_bytes_per_lb=" << semanticMaximum(6)
              << " blob_records=" << semanticTotal(7)
              << " maximum_blob_records_per_lb=" << semanticMaximum(7)
              << " blob_bytes=" << semanticTotal(8)
              << " maximum_blob_bytes_per_lb=" << semanticMaximum(8)
              << " reverse_map_views=" << semanticTotal(9)
              << " reverse_map_entries_upper=" << semanticTotal(10)
              << " maximum_reverse_map_entries_upper_per_lb="
              << semanticMaximum(10)
              << " reverse_map_slots_upper=" << semanticTotal(11)
              << " maximum_reverse_map_slots_upper_per_lb="
              << semanticMaximum(11)
              << " reverse_key_bytes_upper=" << semanticTotal(12)
              << " maximum_reverse_key_bytes_upper_per_lb="
              << semanticMaximum(12)
              << " reverse_owners=" << semanticTotal(13)
              << " maximum_reverse_owners_per_lb=" << semanticMaximum(13)
              << " pod_map_views=" << semanticTotal(14)
              << " pod_map_entries=" << semanticTotal(15)
              << " maximum_pod_map_entries_per_lb=" << semanticMaximum(15)
              << " pod_map_slots=" << semanticTotal(16)
              << " maximum_pod_map_slots_per_lb=" << semanticMaximum(16)
              << " pod_run_values=" << semanticTotal(17)
              << " maximum_pod_run_values_per_lb=" << semanticMaximum(17)
              << " mandatory_statement_keys=" << semanticTotal(18)
              << " maximum_mandatory_statement_keys_per_lb="
              << semanticMaximum(18)
              << " metadata_bytes=" << semanticTotal(19)
              << " maximum_metadata_bytes_per_lb=" << semanticMaximum(19)
              << std::endl;
    const auto taskTotal = [&](std::size_t metric) {
        return taskProjectionTotals[metric].load(std::memory_order_relaxed);
    };
    const auto taskMaximum = [&](std::size_t metric) {
        return taskProjectionMaximums[metric].load(std::memory_order_relaxed);
    };
    gpuCapacityLog << "[GPU-TASK-CAPACITY] tasks=" << taskTotal(0)
              << " maximum_tasks_per_part=" << taskMaximum(0)
              << " batches=" << taskTotal(1)
              << " maximum_batches_per_part=" << taskMaximum(1)
              << " terms=" << taskTotal(2)
              << " maximum_terms_per_part=" << taskMaximum(2)
              << " stumps=" << taskTotal(3)
              << " maximum_stumps_per_part=" << taskMaximum(3)
              << std::endl;
    gpuCapacityLog << "[GPU-FILTER-CAPACITY] request_calls="
              << gpuRequestFilterCalls.load(std::memory_order_relaxed)
              << " producer_calls="
              << gpuProducerFilterCalls.load(std::memory_order_relaxed)
              << " input_statements="
              << gpuFilterInputStatements.load(std::memory_order_relaxed)
              << " maximum_input_statements_per_call="
              << gpuFilterMaximumInputStatements.load(std::memory_order_relaxed)
              << " output_statements="
              << gpuFilterOutputStatements.load(std::memory_order_relaxed)
              << " maximum_output_statements_per_call="
              << gpuFilterMaximumOutputStatements.load(std::memory_order_relaxed)
              << std::endl;
    gpuCapacityLog << "[GPU-GROW-CAPACITY]";
    for (std::size_t depth = 0; depth < kGpuGrowDepthCount; ++depth) {
        gpuCapacityLog << " attempts_d" << depth << "="
                  << gpuGrowAttemptsByDepth[depth].load(std::memory_order_relaxed)
                  << " frontier_d" << depth << "="
                  << gpuGrowFrontierByDepth[depth].load(std::memory_order_relaxed)
                  << " subkeys_d" << depth << "="
                  << gpuGrowSubkeysByDepth[depth].load(std::memory_order_relaxed)
                  << " requests_d" << depth << "="
                  << gpuGrowRequestsByDepth[depth].load(std::memory_order_relaxed)
                  << " producer_attempts_d" << depth << "="
                  << gpuProducerAttemptsByDepth[depth].load(
                         std::memory_order_relaxed)
                  << " producer_survivors_d" << depth << "="
                  << gpuProducerSurvivorsByDepth[depth].load(
                         std::memory_order_relaxed);
    }
    gpuCapacityLog << std::endl;
    const auto evaluationTotal = [&](std::size_t metric) {
        return gpuEvaluationTotals[metric].load(std::memory_order_relaxed);
    };
    const auto evaluationMaximum = [&](std::size_t metric) {
        return gpuEvaluationMaximums[metric].load(std::memory_order_relaxed);
    };
    gpuCapacityLog << "[GPU-EVAL-CAPACITY] requests=" << evaluationTotal(0)
              << " dependency_pass_requests=" << evaluationTotal(1)
              << " reverse_owners=" << evaluationTotal(2)
              << " maximum_reverse_owners_per_request="
              << evaluationMaximum(0)
              << " candidate_owners=" << evaluationTotal(3)
              << " maximum_candidate_owners_per_request="
              << evaluationMaximum(1)
              << " encoded_hits=" << evaluationTotal(4)
              << " local_values=" << evaluationTotal(5)
              << " maximum_local_values_per_hit=" << evaluationMaximum(2)
              << " head_records=" << evaluationTotal(6)
              << " marker_records=" << evaluationTotal(7)
              << " demand_records=" << evaluationTotal(8)
              << " generated_bytes=" << evaluationTotal(9)
              << " level_values=" << evaluationTotal(10)
              << " origin_dependencies=" << evaluationTotal(11)
              << " marker_keys=" << evaluationTotal(12)
              << " marker_remaining_args=" << evaluationTotal(13)
              << " marker_args=" << evaluationTotal(14)
              << std::endl;

    // ---- End-of-iteration straggler classification (the split TRIGGER) ----
    // Set each LB's split for the NEXT iteration from this iteration's completed,
    // deterministic work totals. work(L) = sum of L's parts' submatch counts
    // (split-invariant, D-117). A straggler is an LB whose work exceeds the ideally-
    // balanced per-core load T / logicalCores (so it alone leaves cores idle) AND
    // clears the setup break-even min_split_work; it runs as logicalCores expression
    // buckets next iteration, every other LB unsplit. Integer arithmetic only, over a
    // fixed-order single-threaded sweep -> the split set is a deterministic function
    // of proof state (two runs stay byte-identical). Recomputed every iteration, so an
    // LB whose work falls back below the bar returns to unsplit. A doomed LB
    // (burst early-exit) contributes NOTHING: its tally was cut at a
    // timing-dependent overshoot, so folding it into T (or classifying it)
    // would make the statistic — and any borderline straggler verdict —
    // a race outcome.
    if (mainPath) {
        int64_t T = 0;
        for (std::size_t li = 0; li < M; ++li)
            if (!lbDoomed[li]) T += lbTotalSub[li];
        const int64_t fairShare = T / static_cast<int64_t>(logicalCores);
        // Split reports go to the common diagnostics log so the run log
        // stays readable; one line per onset / self-control hit.
        std::ostream& splitLog = diagnosticsLog();
        for (std::size_t li = 0; li < M; ++li) {
            // An early-exited LB is discharging out of the grid: its cut
            // tally is not comparable work and it never bursts again — the
            // unsplit default is its final split state.
            if (lbDoomed[li]) {
                active[li]->numberOfParts = 1;
                continue;
            }
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
                splitLog << "[SPLIT] straggler: work=" << lbTotalSub[li]
                         << " T=" << T
                         << " parts=" << active[li]->numberOfParts
                         << " lb=" << active[li]->exprKey() << "\n";
            // Self-control report: a split LB whose busiest bucket is still a global
            // outlier has an irreducibly-serial core (a "runs and runs" induction LB)
            // that bucketing cannot subdivide. Pure observation -> determinism intact.
            if (straggler && lbMaxSub[li] > fairShare)
                splitLog << "[SPLIT] ineffective: work=" << lbTotalSub[li]
                         << " maxPart=" << lbMaxSub[li]
                         << " buckets=" << lbBuckets[li]
                         << " stumps=" << lbStumps[li]
                         << " fairShare=" << fairShare
                         << " lb=" << active[li]->exprKey() << "\n";
        }
    }
#if RT_MEASUREMENT
    // [RT phase measurement] carry the split-invariant work totals out
    // to the post-phase-3 report (lbTotalSub is scoped to this block).
    for (std::size_t li = 0; li < M; ++li) trapWork[li] = lbTotalSub[li];
#endif
    }  // active non-empty

    const double phase2IterationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase2Started).count();
    phase2CumulativeSeconds += phase2IterationSeconds;
    ++phase2MeasuredIterations;
    diagnosticsLog() << "[PHASE2-TIMING] backend="
              << (phase2Backend == Phase2Backend::cuda ? "cuda" : "cpu")
              << " iteration_seconds=" << phase2IterationSeconds
              << " cumulative_seconds=" << phase2CumulativeSeconds
              << " iterations=" << phase2MeasuredIterations
              << " active_logical_blocks=" << active.size() << std::endl;
#if RT_MEASUREMENT
    gl::rt_tracker::addRtPhaseWallSeconds(2, phase2IterationSeconds);
#endif

    // Phase 3 opens the same working-set window (all phases equivalent).
    const auto phase3Started = std::chrono::steady_clock::now();
    {
        std::atomic<std::size_t> phase3Cursor{ 0 };
#if PHASE13_DEEP_TIMING
        phase13TimingRows = phase3DetailRows.data();
        phase13TimingWorkers = workers;
#endif
        steward->beginPhaseWindow(/*phase=*/3, &phase3Cursor, &active, workers,
                                  lbdeload::kDeloadDirectory);
        runPhase(phase3Cursor, [this](Memory& b, unsigned cid) {
            g_inParallelWorkerPhase = true;
            this->performElemPhase3(b, cid);
            g_inParallelWorkerPhase = false;
        }
#if RT_MEASUREMENT
        , trapPh3Ns.data()
#endif
        );
        steward->endPhaseWindow();
#if PHASE13_DEEP_TIMING
        phase13TimingRows = nullptr;
        phase13TimingWorkers = 0;
#endif
    }
    const double phase3IterationSeconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - phase3Started).count();
    diagnosticsLog() << "[PHASE13-TIMING] phase1_seconds="
              << phase1IterationSeconds
              << " phase3_seconds=" << phase3IterationSeconds
              << " active_logical_blocks=" << active.size() << std::endl;
#if RT_MEASUREMENT
    // Register this iteration's phase wall-clock with the RT aggregate so
    // its header can state attributed worker-seconds against the single
    // timeline (effective parallelism per phase).
    gl::rt_tracker::addRtPhaseWallSeconds(1, phase1IterationSeconds);
    gl::rt_tracker::addRtPhaseWallSeconds(3, phase3IterationSeconds);
#endif
#if PHASE13_DEEP_TIMING
    printPhase13Detail(1, phase1IterationSeconds, phase1DetailRows);
    printPhase13Detail(3, phase3IterationSeconds, phase3DetailRows);
#endif

#if MEM_MEASUREMENT
    // Fold this iteration's per-worker rows into one coherent instant. Placed
    // immediately after the phase-3 window closes and before the steward's
    // discharge / deload work below, so the pool footprints it reads are the
    // iteration's high-water rather than the post-eviction remainder.
    gl::mem_tracker::commitIterationSample();
#endif

#if RT_MEASUREMENT
    // [RT phase measurement - file-bound, see .rt/burst_phases.log]
    // Per-iteration per-LB wall-time report: per-phase totals, then the top LBs
    // by summed wall time with their FULL parentMemory chain (Rule 12 - the
    // [SPLIT] lines print only the ambiguous leaf exprKey). Observation only.
    //
    // Written to `.rt/burst_phases.log`, never stdout: this is measurement the
    // acceleration campaigns read, and the main run log stays free of it. The
    // per-batch / per-section split lives beside it in `.rt/_aggregate_<tag>.log`.
    if (mainPath && !active.empty()) {
        auto trapSec = [](int64_t ns) {
            return static_cast<double>(ns) / 1e9; };
        std::filesystem::create_directories(".rt");
        std::ofstream phaseLog(".rt/burst_phases.log", std::ios::app);
        int64_t s1 = 0, s2 = 0, sf = 0, s3 = 0;
        for (std::size_t li = 0; li < active.size(); ++li) {
            s1 += trapPh1Ns[li]; s2 += trapPh2Ns[li];
            sf += trapFinNs[li]; s3 += trapPh3Ns[li];
        }
        phaseLog << "[BURST-PH] ph1=" << trapSec(s1) << "s ph2=" << trapSec(s2)
                  << "s fin=" << trapSec(sf) << "s ph3=" << trapSec(s3) << "s\n";
        std::vector<std::size_t> trapOrder(active.size());
        for (std::size_t li = 0; li < trapOrder.size(); ++li) trapOrder[li] = li;
        std::sort(trapOrder.begin(), trapOrder.end(),
            [&](std::size_t a, std::size_t b) {
                const int64_t ta = trapPh1Ns[a] + trapPh2Ns[a]
                                 + trapFinNs[a] + trapPh3Ns[a];
                const int64_t tb = trapPh1Ns[b] + trapPh2Ns[b]
                                 + trapFinNs[b] + trapPh3Ns[b];
                return ta > tb; });
        const std::size_t trapTopK =
            std::min<std::size_t>(8, trapOrder.size());
        for (std::size_t k = 0; k < trapTopK; ++k) {
            const std::size_t li = trapOrder[k];
            const int64_t tot = trapPh1Ns[li] + trapPh2Ns[li]
                              + trapFinNs[li] + trapPh3Ns[li];
            if (tot < 500000000LL) break;  // report only LBs >= 0.5 s
            std::string chain;
            for (const Memory* p = active[li]; p != nullptr;
                 p = p->parentMemory) {
                if (!chain.empty()) chain += " <- ";
                const std::string key = p->exprKey();
                chain += key.empty() ? std::string("(root)") : key;
            }
            phaseLog << "[RT-LB] tot=" << trapSec(tot)
                      << "s ph1=" << trapSec(trapPh1Ns[li])
                      << "s ph2=" << trapSec(trapPh2Ns[li])
                      << "s fin=" << trapSec(trapFinNs[li])
                      << "s ph3=" << trapSec(trapPh3Ns[li])
                      << "s work=" << trapWork[li]
                      << " chain=" << chain << "\n";
        }
    }
#endif

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
    if (!parameters.compressor_mode && !ceFilteringActive
        && rollingMailHistoryEnabled) {
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
    if (!parameters.compressor_mode && !ceFilteringActive) {
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

    // In-run OR-theorem construction (A16 Phase 2,
    // D-266): fold this iteration's new proved
    // rows into or theorems and queue their implication-compact broadcast
    // on pendingCompactionQueue so the drain below ships them with this
    // iteration's batch. Placed after the vacuous retraction sweep so a
    // row retracted this iteration is never scanned as an or source.
    this->constructOrTheoremsInRun();

    // Pre-split merge directly after the or seam: an or theorem minted
    // this window can license a merge in the same window, and the merged
    // theorem's broadcast rides the compaction drain just below.
    this->constructOrEliminationInRun();

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
        // Incubator shape routing (D-332):
        // the root's log is read by the anchor LB alone, so a nested rule
        // (anchorOnlyRule false) is ALSO merged into the anchor LB's mailOut --
        // the anchor's log reaches every LB below it, the root's log reaches
        // the anchor LB itself.
        std::map<int, Mail> nestedMailByCore;
        Memory* const nestedStore = (parameters.incubator_mode && !pendingCompactionQueue.empty())
            ? this->incubatorAnchorLb() : nullptr;
        for (const std::tuple<std::string, int, int>& e : pendingCompactionQueue)
        {
            const std::string& original = std::get<0>(e);
            const int cId = std::get<2>(e);
            const bool nested = nestedStore != nullptr && !this->anchorOnlyRule(original);
            const std::string compactImpl = compileImplicationToCompact(original);
            // Level set MUST be empty. The implication rule always
            // deposits std::set<int>() for its levels.
            // The receiver-side addToHashMemory call propagates these levels
            // into the rule's intStatementLevelsMap entry, and when the rule
            // fires, the derived statement's levels are computed as the union
            // of the rule levels and the matching premises' levels. The
            // empty set keeps the derived levels equal to the premise-side
            // union only, which is what the allLevelsInvolved registration
            // verdict computed in prover.hpp::dischargeToBeProved expects
            // (size == memoryBlock.level + 1). Filling compactLevels with
            // {0, 1, ..., kySize} would inject an extra level into every
            // derived statement that fired against a mail-arrived rule,
            // making size > level + 1 so the sealed verdict turns false and
            // the drain refuses appendGlobalTheorem — the goal still closes
            // (closure is level-free), but the theorem is silently lost.
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
            if (nested) {
                Mail& nm = nestedMailByCore[cId];
                nm.statements.insert(std::make_pair(compactEv, compactLevels));
                if (parameters.trackHistory) {
                    // The same rows as the root batch (keyed by the compact).
                    const auto oit = m.exprOriginMap.find(compactEv);
                    assert(oit != m.exprOriginMap.end()
                        && "deferred-compaction drain: the compact's compilation row must exist");
                    for (const OriginLine& line : oit->second) {
                        addOrigin(nm.exprOriginMap, compactEv, line,
                            (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    }
                }
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
        // Shape-routed nested compacts: the anchor LB's mailOut too (its log
        // reaches every LB below it), under the anchor's claim.
        if (!nestedMailByCore.empty()) {
            assert(nestedStore != nullptr);
            steward->claimAndLoadForWork(*nestedStore, /*phase=*/4,
                                         lbdeload::kDeloadDirectory);
            for (std::map<int, Mail>::iterator mit = nestedMailByCore.begin(); mit != nestedMailByCore.end(); ++mit) {
                mergeBatchIntoMailOut(mit->second, *nestedStore);
            }
            nestedStore->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        }
    }
    pendingCompactionQueue.clear();

    // The compaction drain registers fresh compacts; close the reduced-or
    // closure again so any or it minted is covered before the next
    // iteration's bursts (idempotent; D-268).
    preMintReducedOrs();

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
                if (parameters.allow_ssd_deload) {
                    // Stamp the ordinal single-threaded now, before any later
                    // steward dump names the file.
                    b->ensureDeloadOrdinal();
                    pendingDischarge.push_back(b);
                }
            }
        }
        // Hand the discharged list to the steward for the background
        // near-empty dump: wake it now if over the wake mark, else arm the
        // grant trigger to wake it on a mid-iteration crossing
        // (I-106). Active-LB pressure is handled
        // continuously by the working-set pager (the phase windows), so the
        // barrier no longer mass-deloads active LBs; the pool stays the hard
        // bound and genuine exhaustion asserts in a phase, never here.
        assert(parameters.allow_ssd_deload || pendingDischarge.empty());
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
        // intEncodedStatements erasures shift the positions the persistent
        // per-class statement-index waterlines (eqClassSttmntIndexMapMap)
        // point into — record each erased position and repair the
        // waterlines, or equivalence-class application silently skips the
        // rows that slid under a stale waterline. The exact-key admission
        // gates keep (originalId, validityId) unique in the list, so the
        // capture buffer is a loud Rule-19 tripwire, not a real capacity.
        int32_t erasedPos[8];
        int32_t erasedN = 0;
        for (int32_t i = mb.intEncodedStatements.size(); i-- > 0; ) {
            if (mb.intEncodedStatements[i].originalId == origId
                && mb.intEncodedStatements[i].validityId == valId) {
                assert(erasedN < 8
                    && "removeExpressionFromMemoryBlock: duplicate (text, scope) rows exceed the erased-position capture");
                erasedPos[erasedN++] = i;
                mb.intEncodedStatements.erase(i);
            }
        }
        if (erasedN > 0) {
            // Collected back to front; the repair takes an ascending run.
            std::sort(erasedPos, erasedPos + erasedN);
            repairEqClassWaterlines(mb, erasedPos, erasedN);
        }
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

/// @brief End-of-burst ancestor-known sweep: drop statement-LIST rows at
///        non-main scopes whose text a strict ancestor knows
///        (I-187).
///
/// @details
/// Detection first, mutation second: one read-only pass over
/// `intEncodedStatements` collects the packed `(originalId, validityId)`
/// keys of every non-main row whose text `ancestorKnown` reports at a
/// strict ancestor; the hits then drop in decoded-lex
/// `(original, validity)` order (I-84) through
/// `removeExpressionFromMemoryBlock(state=0)` — the statement lists only,
/// registry rows surviving as tombstones. The decodeView spans handed to
/// the removal door stay valid because the door only does non-minting
/// lookups and `PagedVector` erases — no NameMap mint (I-3). Scratch
/// (hit list on the page tier, sort index on the byte-bump tier) rides
/// the per-slot gen-scratch arena, reclaimed before return. Full
/// contract at the declaration.
///
/// @param body Owning LB; statement lists and waterlines mutated in place
///             through the removal door.
/// @invariant Registry rows are never erased here (I-58 / I-85).
/// @see sweepAncestorKnownRows (decl) — placement and tombstone rationale.
void ExpressionAnalyzer::sweepAncestorKnownRows(Memory& body) {
    if (parameters.compressor_mode) return;
    const unsigned slot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset sweepMark = gArena.cursor();
    {
        DirtyState hitsDirty = DirtyState::Clean;
        PagedVector<int64_t> hits(&gArena, &hitsDirty);
        for (int32_t i = 0; i < body.intEncodedStatements.size(); ++i) {
            const NameId origId = body.intEncodedStatements[i].originalId;
            const NameId valId = body.intEncodedStatements[i].validityId;
            if (valId == NameMap::MAIN_ID) continue;
            if (ancestorKnown(body, origId, valId, /*includeSelf=*/false)) {
                hits.push_back(packStatementKey(origId, valId));
            }
        }
        const int32_t hitN = hits.size();
        if (hitN > 0) {
            int32_t* idx = reinterpret_cast<int32_t*>(gArena.resolve(
                gArena.alloc(hitN * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t k = 0; k < hitN; ++k) idx[k] = k;
            std::sort(idx, idx + hitN, [&](int32_t a, int32_t b) {
                const StatementKey ka = Codec<StatementKey>::decode(hits[a]);
                const StatementKey kb = Codec<StatementKey>::decode(hits[b]);
                const int c = compareSpans(body.nameMap.decodeView(ka.orig),
                                           body.nameMap.decodeView(kb.orig));
                if (c != 0) return c < 0;
                return compareSpans(body.nameMap.decodeView(ka.validity),
                                    body.nameMap.decodeView(kb.validity)) < 0;
            });
            for (int32_t k = 0; k < hitN; ++k) {
                const StatementKey key =
                    Codec<StatementKey>::decode(hits[idx[k]]);
                removeExpressionFromMemoryBlock(
                    body.nameMap.decodeView(key.orig),
                    body.nameMap.decodeView(key.validity),
                    body, /*state=*/0);
            }
            // Quiescence (D-194): removals can cancel a genuine addition in
            // the count diff — flag the mutation directly.
            body.mutatedThisBurst = true;
        }
    }
    gArena.popTo(sweepMark);
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
/// @param kySize `ky.size()` at the call site (0 for premise-chain-less
///        callers); participates only in the drain's deterministic sort —
///        the deposited level set is always empty (see the drain's
///        level-set comment).
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
            // Elements carry each disjunct's TRUE polarity, so a disjunct's
            // conjunct form is its negation WITH double-negation cancellation:
            // a negated element contributes its bare positive core, a positive
            // element gains the '!' prefix. Byte-identical to the former
            // blind-prefix builder for all-positive elements.
            const auto negLen = [](const StrSpan& e) -> int32_t {
                return (e.len > 0 && e.ptr[0] == '!') ? e.len - 1 : e.len + 1;
            };
            const auto putNeg = [](char* buf, int32_t at, const StrSpan& e) -> int32_t {
                if (e.len > 0 && e.ptr[0] == '!') {
                    std::memcpy(buf + at, e.ptr + 1, static_cast<size_t>(e.len - 1));
                    return at + e.len - 1;
                }
                buf[at++] = '!';
                std::memcpy(buf + at, e.ptr, static_cast<size_t>(e.len));
                return at + e.len;
            };
            const StrSpan e0 = elements[0];
            const StrSpan e1 = elements[1];
            StrSpan current;
            {
                const int32_t n = 3 + negLen(e0) + negLen(e1) + 1;  // "!(&" neg(e0) neg(e1) ")"
                char* buf = out.allocBytes(n);
                int32_t at = 0;
                buf[at++] = '!'; buf[at++] = '('; buf[at++] = '&';
                at = putNeg(buf, at, e0);
                at = putNeg(buf, at, e1);
                buf[at++] = ')';
                assert(at == n);
                current = StrSpan(buf, n);
            }
            for (int32_t i = 2; i < elemN; ++i) {
                // The or-so-far is a DISJUNCT of the next level, so it
                // enters the AND negated like any other disjunct — via
                // putNeg's double-negation cancellation its `!(&…)` form
                // contributes the bare positive `(&…)`:
                // `!(&(&!D_1!D_2)!D_3)` = (D_1 ∨ D_2) ∨ D_3. Inserting
                // `current` un-negated read as ¬(D_1 ∨ D_2) ∨ D_3 — the
                // mirrored polarity defect flagged in D-260, live once
                // ≥3-element ors are consumed in-run.
                const StrSpan elem = elements[i];
                const int32_t n = 3 + negLen(current) + negLen(elem) + 1;  // "!(&" neg(current) neg(elem) ")"
                char* buf = out.allocBytes(n);
                int32_t at = 0;
                buf[at++] = '!'; buf[at++] = '('; buf[at++] = '&';
                at = putNeg(buf, at, current);
                at = putNeg(buf, at, elem);
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

/// @brief Consume one ordered true-polarity disjunct cohort — the or
///        machinery's whole downstream, shared by the compiled-or arm and
///        the negated-AND De-Morgan door.
///
/// @details
/// See the declaration for the full contract. The body is the former or-arm
/// downstream of `disintegrateExprCore2`, extracted verbatim; the expansion
/// history is inlined (the arm used the function-local
/// `trackExpansionHistory` lambda) so the door — which runs before the
/// lambda's definition point — shares it.
///
/// @param expr              Deposited expression (history antecedent).
/// @param entSignature      u_-form cohort signature.
/// @param orLeaves          Ordered disjunct spans at TRUE polarity.
/// @param orLeafN           Leaf count.
/// @param currentStatement  u_-stripped `collected` key.
/// @param memoryBlock       Owning LB.
/// @param collected         Product sink.
/// @param validityName      Deposit scope.
/// @param trackHistoryLocal History suppression flag.
/// @param allowOrDisintegration Route (b) signal.
/// @param orSeedLevels      Seed level run.
/// @param orSeedLevelCount  Seed level count.
/// @param allowOrProbe      Real-deposit flag.
/// @param sArena            Caller's per-slot string scratch.
/// @invariant LB-local writes only (I-28); registry reads via
///            `compiledEntity` only.
/// @see disintegrateExprCore2, flattenOrLeaves.
void ExpressionAnalyzer::consumeOrLeavesCohort(StrSpan expr,
    StrSpan entSignature,
    const StrSpan* orLeaves, int32_t orLeafN,
    StrSpan currentStatement,
    Memory& memoryBlock,
    CollectedArena& collected,
    StrSpan validityName,
    bool trackHistoryLocal,
    bool allowOrDisintegration,
    const int* orSeedLevels, int32_t orSeedLevelCount,
    bool allowOrProbe,
    ScratchArena& sArena,
    WorkInstruction& instructions,
    int iteration,
    NewVarStore& newVarMap,
    StrSpan parentWitness,
    WitnessMetaStore* witnessMeta)
{
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
    const StrSpan entCategory("or", 2);
    const ScratchString expandedOrSignature = expandSignature(
        entCategory, entSignature, orLeaves, orLeafN, sArena);
    // L3 span-record door. expandedOrSignature is a stable local; the
    // per-branch KEY is u_-stripped onto sArena in the loop. The
    // antecedent is loop-invariant, so the OriginDep is built once.
    const OriginDep orDisDeps[1] = {
        { StrSpan(expandedOrSignature), StrSpan(validityName) } };

    // The K mutual-exclusion implications cite expandedOrSignature as
    // their disintegration origin, which requires the matching
    // expansion-origin record (expandedOrSignature -> deposited
    // expression) to exist regardless of whether the per-branch
    // case-split fires. Inlined trackExpansionHistory section 1: the
    // paired mailOut write ships the record the statement delta cannot
    // (the expansion conjunction never enters the delta).
    if (parameters.trackHistory && trackHistoryLocal) {
        const int expCap = (parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr
            : parameters.max_origin_per_expr);
        const ScratchString exprClean = removeUPrefixScratch(sArena, StrSpan(expr));
        const OriginDep expDeps[1] = { { StrSpan(exprClean), StrSpan(validityName) } };
        addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner,
            StrSpan(expandedOrSignature), StrSpan(validityName),
            OriginTag::expansion, expDeps, 1, expCap);
        memoryBlock.addMailOutOrigin(StrSpan(expandedOrSignature),
            StrSpan(validityName), OriginTag::expansion, expDeps, 1, expCap);
    }

    // 1. The or implications — the K mutual-exclusion rules
    //    (!d_0 & ... & !d_{i-1} & !d_{i+1} & ... & !d_{N-1}) -> d_i and, for a
    //    registry or with k >= 3 leaves, the subset-exclusion rules (I-184,
    //    D-269) — come from the or entity's compiled `implications` list
    //    (D-309): one compact instance
    //    `(implication<N>[args])` per rule, u_p -> instance argument p
    //    positionally (repeated arguments are legal). Each instance is a
    //    product STATEMENT of this cohort exactly like an `and` element: a
    //    child of the or's key, registered at the parent scope, and
    //    disintegrated through the implication branch of
    //    `disintegrateExprCore2`, which expands it into the same full-bind
    //    rule the on-the-spot construction produced (the compact's elements
    //    through reconstructImplicationFullBindScratch) and writes the rule's
    //    `expansion` origin citing the compact. The compact itself takes the
    //    `disintegration` origin the rule carried (the verifier's
    //    check_disintegration or branch expands it back). A De-Morgan door
    //    cohort (`!(op[args])`, no or entity) keeps the on-the-spot K-rule
    //    construction in the else branch.
    const LogicalEntity* orLe = nullptr;
    if (entSignature.len > 0 && entSignature.ptr[0] == '(') {
        const LogicalEntity* cand =
            compiledEntity(extractExpressionUniversalSpan(entSignature));
        if (cand != nullptr && cand->category == "or") orLe = cand;
    }
    if (orLe != nullptr) {
        assert(static_cast<int32_t>(orLe->implications.size())
                == expectedOrImplicationCount(orLeafN)
            && "consumeOrLeavesCohort: or-implication list incomplete (I-208)");
        StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
        const int32_t sigN = getArgsSpans(StrSpan(orLe->signature), sigArgs,
            ExecutionParameters::MAX_ARITY);
        StrSpan instArgs[ExecutionParameters::MAX_ARITY];
        const int32_t instN = getArgsSpans(entSignature, instArgs,
            ExecutionParameters::MAX_ARITY);
        assert(sigN == instN
            && "consumeOrLeavesCohort: instance arity differs from the compiled or");
        StrReplacement pairs[ExecutionParameters::MAX_ARITY];
        for (int32_t p = 0; p < sigN; ++p) {
            pairs[p].key = sigArgs[p];
            pairs[p].value = instArgs[p];
        }
        const int maxOrigins = parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr
            : parameters.max_origin_per_expr;
        // The instruction's marked goal is re-set by every
        // prepareIntegrationCore call; copy it onto the arena first so the
        // span never aliases the interner it is minted back into (I-3).
        const StrSpan goalView = instructions.markedGoal();
        const ScratchString goalCopy =
            ScratchString::copyFrom(sArena, goalView.ptr, goalView.len);
        for (const std::string& tmpl : orLe->implications) {
            // Per-instance scope: the instance bytes live through the
            // recursion (the WorkInstruction copies what it keeps).
            ScratchScope instScope(sArena);
            const ScratchString inst =
                replaceKeysScratch(sArena, StrSpan(tmpl), pairs, sigN);
            const ScratchString instClean =
                removeUPrefixScratch(sArena, StrSpan(inst));
            collected.insertChild(StrSpan(currentStatement), StrSpan(instClean));
            if (parameters.trackHistory && trackHistoryLocal) {
                // The compact takes the disintegration origin the rule
                // carried (KEY u_-stripped: chapter rows surface u_-stripped
                // expressions).
                addOriginEncoded(memoryBlock.exprOriginMap,
                    memoryBlock.originInterner, StrSpan(instClean),
                    StrSpan(validityName), OriginTag::disintegration,
                    orDisDeps, 1, maxOrigins);
                // Paired `mailOut.exprOriginMap` write (D-274): the compact
                // registers as a statement and rides the delta, and a
                // receiver resolving its row needs the antecedent chain
                // shipped alongside (the or's expansion record is pair-mailed
                // above).
                memoryBlock.addMailOutOrigin(StrSpan(instClean),
                    StrSpan(validityName), OriginTag::disintegration,
                    orDisDeps, 1, maxOrigins);
            }
            prepareIntegrationCore(StrSpan(inst), instructions, memoryBlock,
                StrSpan(goalCopy));
            disintegrateExprCore2(StrSpan(inst), instructions, memoryBlock,
                iteration, collected, newVarMap, validityName,
                trackHistoryLocal, allowOrDisintegration,
                orSeedLevels, orSeedLevelCount, allowOrProbe,
                parentWitness, witnessMeta);
        }
    }
    else {
        // De-Morgan door cohort: the K rules are built on the spot — the
        // negated compound resolves to no or entity, so there is no compiled
        // list (K rules only; the subset-exclusion family is registry-or
        // only). Each is stamped with the `disintegration` origin pointing
        // at the OR's expanded form so the proof graph can audit the K rules
        // back to the originating OR (the verifier's De-Morgan branch
        // rebuilds and verifies the exact shape).
        for (int32_t i = 0; i < orLeafN; ++i) {
            StrSpan premiseSpans[64];
            std::size_t premiseCount = 0;
            for (int32_t j = 0; j < orLeafN; ++j) {
                if (j != i) {
                    assert(premiseCount < 64 && "or-branch premise chain exceeds 64");
                    // negateScratch, not prefixBang: a disjunct carries its
                    // true polarity, so a negated leaf's exclusion premise is
                    // its bare positive core, never a double negation.
                    premiseSpans[premiseCount++] = negateScratch(sArena, orLeaves[j]);
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
                // Paired `mailOut.exprOriginMap` write
                // (D-274). The K mutual-exclusion
                // implications are flat hash rules — never statements, never in
                // the delta — so without this mirror a descendant that receives
                // one of their fired heads by mail holds a citing row it cannot
                // resolve (`buildStack` asserts). The antecedent
                // `expandedOrSignature` record is itself pair-mailed above, so
                // the receiver's chain closes.
                memoryBlock.addMailOutOrigin(StrSpan(impClean),
                    StrSpan(validityName), OriginTag::disintegration,
                    orDisDeps, 1, maxOrigins);
            }
        }
    }

    // The per-branch case-split stays depth-gated: nested _ordis_
    // scopes are the scope explosion max_or_depth exists to prevent.
    // A depth-gated or neither opens nor parks (the single-layer
    // contract): the K rules above are its whole consumption.
    if (currentOrDepth < parameters.max_or_depth) {
        // 2. TWO-ROUTE cohort opening (admission-based ordis; restores
        //    the D-32 distinction the sequenced branch had overridden).
        //    Route (b): the firing rule is a product of disintegration
        //    (allowOrDisintegration — the threaded D-32 signal); such
        //    heads open unconditionally, as in rungs 1+2. Route (a):
        //    demand-driven — some disjunct's product template is
        //    already an algebra admissionMap key (tagged or untagged;
        //    the probe consults the admission maps ONLY, never the I-6
        //    shape rule, and never consumes). Neither route: the
        //    cohort PARKS in rejectedMapOrdis under its operator-based
        //    product templates and revives by mail at admission key
        //    gain. Probe and park run only for real deposits
        //    (allowOrProbe); the hypothetical path neither opens nor
        //    parks. The explosion protection is BOTH the demand filter
        //    and the sequenced one-branch-at-a-time release below.

        // Clean leaf forms, computed once for probe / park / bootstrap
        // (byte-identical to the former per-leaf strip in the mint
        // block; multiple spans under the enclosing scope).
        ScratchString cleanHold[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        StrSpan cleanLeaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        for (int32_t i = 0; i < orLeafN; ++i) {
            cleanHold[i] = removeUPrefixScratch(sArena, orLeaves[i]);
            cleanLeaves[i] = StrSpan(cleanHold[i]);
        }

        // Product templates of leaf `li`, fed to `sink(StrSpan)` inside
        // one scratch window: an existence disjunct yields its compiled
        // definition body's witness facts with the witness slot in
        // marker form (body instantiated in u_-form so the witness is
        // the only non-u_ token — the listLastRemovedArgsLE
        // discriminator, mirroring the real existence consumption); an
        // operator-application disjunct yields itself; equalities (no
        // writer can produce a matching demand), negated disjuncts
        // (deferred by design), and non-operator cores yield nothing.
        const auto forEachProductTemplate = [&](int32_t li, auto&& sink) {
            const StrSpan cleanLeaf = cleanLeaves[li];
            if (cleanLeaf.len > 0 && cleanLeaf.ptr[0] == '!') return;
            if (isEquality(cleanLeaf)) return;
            const StrSpan core = extractExpressionSpan(cleanLeaf);
            const LogicalEntity* le = compiledEntity(core);
            if (le != nullptr && le->category == "existence") {
                ScratchScope tmplScope(sArena);
                StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
                const int32_t sigN = getArgsSpans(StrSpan(le->signature),
                    sigArgs, ExecutionParameters::MAX_ARITY);
                StrSpan instArgs[ExecutionParameters::MAX_ARITY];
                const int32_t instN = getArgsSpans(orLeaves[li], instArgs,
                    ExecutionParameters::MAX_ARITY);
                assert(sigN == instN
                    && "ordis product template: existence instance arity "
                       "differs from the compiled definition");
                StrReplacement pairs[ExecutionParameters::MAX_ARITY];
                for (int32_t a = 0; a < sigN; ++a) {
                    pairs[a].key = sigArgs[a];
                    pairs[a].value = instArgs[a];
                }
                ScratchString bodyHold[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                StrSpan bodyElems[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                int32_t bodyN = 0;
                for (const std::string& rawElem : le->elements) {
                    assert(bodyN < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                        && "ordis product template: body element count "
                           "exceeds cap");
                    bodyHold[bodyN] = replaceKeysScratch(sArena,
                        StrSpan(rawElem), pairs, sigN);
                    bodyElems[bodyN] = StrSpan(bodyHold[bodyN]);
                    ++bodyN;
                }
                StrSpan witnesses[ExecutionParameters::MAX_ARITY];
                const int32_t wN = listLastRemovedArgsLE(
                    StrSpan("existence", 9), bodyElems, bodyN,
                    witnesses, ExecutionParameters::MAX_ARITY);
                assert(wN == 1
                    && "ordis product template: existence definition must "
                       "bind exactly one witness");
                for (int32_t b = 0; b < bodyN; ++b) {
                    const StrSpan bCore = extractExpressionSpan(bodyElems[b]);
                    if (this->operators.find(std::string_view(bCore.ptr,
                            static_cast<std::size_t>(bCore.len)))
                        == this->operators.end()) {
                        continue;
                    }
                    StrSpan bArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t bArgsN = getArgsSpans(bodyElems[b],
                        bArgs, ExecutionParameters::MAX_ARITY);
                    bool hasWitness = false;
                    for (int32_t a = 0; a < bArgsN; ++a) {
                        if (equalSpans(bArgs[a], witnesses[0])) {
                            hasWitness = true;
                            break;
                        }
                    }
                    if (!hasWitness) continue;
                    const ScratchString markedU = makeMarkedExprScratch(
                        sArena, bodyElems[b], witnesses[0]);
                    const ScratchString tmpl =
                        removeUPrefixScratch(sArena, StrSpan(markedU));
                    sink(StrSpan(tmpl));
                }
            } else if (this->operators.find(std::string_view(core.ptr,
                           static_cast<std::size_t>(core.len)))
                       != this->operators.end()) {
                sink(cleanLeaf);
            }
        };

        const bool routeB = allowOrDisintegration;
        bool probeHit = false;
        bool leafAdmitted[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
        if (!routeB && allowOrProbe) {
            for (int32_t i = 0; i < orLeafN; ++i) {
                forEachProductTemplate(i, [&](StrSpan tmpl) {
                    if (leafAdmitted[i]) return;
                    // isAdmitted's prologue, read-only: non-minting
                    // template probe + key presence. No updateAdmissionMap,
                    // no status write.
                    int64_t probePk = 0;
                    if (lookupTemplateKey(memoryBlock.templateInterner,
                            memoryBlock.nameMap, tmpl, validityName, probePk)
                        && memoryBlock.overallHashMemory.admissionMap.lookup(
                               probePk) != 0) {
                        leafAdmitted[i] = true;
                        probeHit = true;
                    }
                });
            }
        }
        // Route (c) — ordis2 demand (D-267):
        // per-leaf GROUND probe of admissionMapOrdis2 ONLY (never the
        // algebra map — the compact-only branch-seed property depends on
        // this path writing/reading no admission entry). A demand key's
        // template half is a ground premise text, so the clean leaf is the
        // exact probe (ground-to-ground byte equality through the template
        // interner). Non-minting. Placed BEFORE the park in this chain, so
        // the demand-first arrival order opens without ever parking.
        bool demandHit = false;
        bool leafDemand[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
        if (!routeB && allowOrProbe
            && !memoryBlock.overallHashMemory.admissionMapOrdis2.empty()) {
            for (int32_t i = 0; i < orLeafN; ++i) {
                int64_t demandPk = 0;
                if (lookupTemplateKey(memoryBlock.templateInterner,
                        memoryBlock.nameMap, cleanLeaves[i], validityName,
                        demandPk)
                    && memoryBlock.overallHashMemory.admissionMapOrdis2.lookup(
                           demandPk) != 0) {
                    leafDemand[i] = true;
                    demandHit = true;
                }
            }
        }
        // No goals, no or branches (I-206): an LB whose
        // goal registry is empty — any scope; evaluated at every open attempt,
        // the registry empties mid-run — neither opens nor parks a cohort. The
        // flat 1a rules above landed regardless.
        const bool lbHasGoals = !memoryBlock.intToBeProved.empty();
        const bool orAdmitted = lbHasGoals && (routeB || probeHit || demandHit);

        if (orAdmitted) {
            // 3. Cohort identity + bootstrap guard. Equal signatures
            //    under different parents are independent case splits
            //    (I-167). An existing count row means this exact cohort
            //    is already scheduled (its branches live, pending,
            //    retired, or converged) — a re-deposit of the same or
            //    statement must NOT re-mint branch scopes or reset the
            //    release sequence; the K mutual-exclusion implications
            //    above were still (re-)emitted.
            const ScratchString orSignature = removeUPrefixScratch(sArena, entSignature);  // e.g. "(or3[1,2,3])"
            const int32_t orParentId =
                memoryBlock.lbStateInterner.encode(validityName);
            const int32_t orSigId =
                memoryBlock.lbStateInterner.encode(StrSpan(orSignature));
            const int32_t orCohortId = mintOrCohortId(
                memoryBlock.lbStateInterner, orParentId, orSigId);
            if (memoryBlock.orDisjunctCount.lookup(orCohortId) == 0) {
                // Fresh cohort: register the FULL structural leaf count
                // (convergence semantics unchanged — dead-branch
                // retirement shrinks it in place, D-242), queue EVERY
                // leaf, and stage the cohort for release. The bootstrap
                // mints NO branch and sends NO mail: this code runs
                // inside standardProcessing's internal-mail drain, and
                // the drain's step-3 clear would wipe a seed inserted
                // into the channel being drained. The end-of-burst
                // drainPendingOrReleases — which runs AFTER that clear —
                // performs the first release exactly like every later
                // one (top-ranked disjunct, seed + origin on
                // sameIterationInternalMail, absorbed next step).
                memoryBlock.orDisjunctCount.insert(orCohortId, orLeafN);

                // Route (a) / route (c) starter: the demand names WHICH
                // branch carries the relevance — the matched disjunct
                // opens first (ties among several matched fall to the
                // standing ranking). Demand WINS over admission when both
                // hit (maintainer decision): the demand-matched leaf is
                // the one a starved rule is waiting to consume. Recorded
                // as a one-shot side row consumed by
                // drainPendingOrReleases; route (b) writes no row and
                // starts at the top of the ranking.
                int32_t starterLeafIx = -1;
                if (!routeB) {
                    const bool* pick = demandHit ? leafDemand : leafAdmitted;
                    StrSpan admitted[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                    int32_t admittedLeaf[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                    int32_t admittedN = 0;
                    for (int32_t i = 0; i < orLeafN; ++i) {
                        if (pick[i]) {
                            admitted[admittedN] = cleanLeaves[i];
                            admittedLeaf[admittedN] = i;
                            ++admittedN;
                        }
                    }
                    assert(admittedN > 0
                        && "route (a)/(c) cohort opened without a matched "
                           "disjunct");
                    StrSpan anchorArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t anchorArgN = collectAnchorArgs(
                        memoryBlock, anchorArgs,
                        ExecutionParameters::MAX_ARITY);
                    starterLeafIx = admittedLeaf[pickTopOrDisjunct(
                        admitted, admittedN, anchorArgs, anchorArgN)];
                }

                // "Opening consumes both" (D-267):
                // each demand-matched leaf's entry leaves admissionMapOrdis2
                // with this open. Runs on the single-threaded absorb seam;
                // re-sweeps stay idempotent because the bootstrap guard
                // above already gates re-deposits. The park side's entries
                // were erased by the wake (or never existed on the
                // demand-first order) — the pair self-liquidates.
                if (demandHit) {
                    for (int32_t i = 0; i < orLeafN; ++i) {
                        if (!leafDemand[i]) continue;
                        int64_t consumePk = 0;
                        const bool present = lookupTemplateKey(
                            memoryBlock.templateInterner, memoryBlock.nameMap,
                            cleanLeaves[i], validityName, consumePk);
                        assert(present
                            && "demand-matched leaf lost its template key");
                        (void)present;
                        memoryBlock.overallHashMemory.admissionMapOrdis2
                            .eraseBlobIf([consumePk](int64_t k) {
                                return k == consumePk;
                            });
                    }
                }

                // Queue each leaf in the wrapped payload-body form (the
                // orBookkeeping id convention). The clean strings ride
                // the enclosing scope's byte-bump tier (fresh-string
                // recipe: multiple spans under one scope).
                for (int32_t i = 0; i < orLeafN; ++i) {
                    const StrSpan cleanSpan = cleanLeaves[i];
                    const int32_t wLen = cleanSpan.len + 2;
                    char* wBuf = sArena.allocBytes(wLen);
                    wBuf[0] = '(';
                    if (cleanSpan.len > 0)
                        std::memcpy(wBuf + 1, cleanSpan.ptr,
                            static_cast<size_t>(cleanSpan.len));
                    wBuf[wLen - 1] = ')';
                    const int32_t wrappedId = memoryBlock.lbStateInterner
                        .encode(StrSpan(wBuf, wLen));
                    memoryBlock.orPendingBranches.insertSorted(orCohortId,
                        wrappedId,
                        DecodedIdLess{ &memoryBlock.lbStateInterner });
                    if (i == starterLeafIx) {
                        memoryBlock.orStarterPick.upsert(orCohortId,
                                                         wrappedId);
                    }
                }
                for (int32_t l = 0; l < orSeedLevelCount; ++l) {
                    memoryBlock.orPendingLevels.insertSorted(orCohortId,
                        orSeedLevels[l],
                        [](int32_t a, int32_t b) { return a < b; });
                }
                memoryBlock.pendingOrReleases.mint(orCohortId);
            }
        } else if (lbHasGoals && allowOrProbe) {
            // 4. PARK: no route admitted — file the cohort under each
            //    operator-based disjunct product template at this
            //    validity. Value = the clean or statement + the seed
            //    level run (the complete reopening context; the key
            //    carries the validity). The RMW dedups re-parks. No
            //    park-time rendezvous is needed: the probe above just
            //    missed synchronously, and every later admission key
            //    gain passes a revisitRejectedOrdis seam.
            const ScratchString orSignature =
                removeUPrefixScratch(sArena, entSignature);
            const int32_t parkStmtId =
                memoryBlock.valueInterner.encode(StrSpan(orSignature));
            const unsigned parkSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : genScratchArenas().slotCount() - 1;
            ScratchArena& parkArena = genScratchArenas().forSlot(parkSlot);
            for (int32_t i = 0; i < orLeafN; ++i) {
                forEachProductTemplate(i, [&](StrSpan tmpl) {
                    const int64_t parkPk = mintTemplateKey(
                        memoryBlock.templateInterner, memoryBlock.nameMap,
                        tmpl, validityName);
                    insertRejectedOrdisIdsBlob(
                        memoryBlock.overallHashMemory.rejectedMapOrdis,
                        parkPk, parkStmtId,
                        orSeedLevels, orSeedLevelCount,
                        memoryBlock.valueInterner, parkArena);
                });
            }
            // DUAL FILING (D-267): the same
            // cohort additionally files in rejectedMapOrdis2 under each
            // ELIGIBLE disjunct's clean GROUND text (polarity verbatim,
            // I-175 — a negated compound files WITH its '!', because
            // A15-family demands are minted negated) at this validity —
            // the demand map's key language, so the drain's wake is a
            // plain key rendezvous. The old filing above is untouched
            // (I-178); only drainAdmissionKeysOrdis2 wakes this index.
            for (int32_t i = 0; i < orLeafN; ++i) {
                if (!ordis2KeyEligible(cleanLeaves[i])) continue;
                const int64_t park2Pk = mintTemplateKey(
                    memoryBlock.templateInterner, memoryBlock.nameMap,
                    cleanLeaves[i], validityName);
                insertRejectedOrdisIdsBlob(
                    memoryBlock.overallHashMemory.rejectedMapOrdis2,
                    park2Pk, parkStmtId,
                    orSeedLevels, orSeedLevelCount,
                    memoryBlock.valueInterner, parkArena);
            }
        }
        // else: hypothetical-disintegration path — neither open nor park.
    }
    // else: max OR depth reached — K mutual-exclusion implications emitted above, no branch opening
}

void ExpressionAnalyzer::disintegrateExprCore2(StrSpan expr,
    WorkInstruction& instructions,
    Memory& memoryBlock,
    int iteration,
    CollectedArena& collected,
    NewVarStore& newVarMap,
    StrSpan validityName,
    bool trackHistoryLocal,
    bool allowOrDisintegration,
    const int* orSeedLevels,
    int32_t orSeedLevelCount,
    bool allowOrProbe,
    StrSpan parentWitness,
    WitnessMetaStore* witnessMeta)
{
    // Capture startInt at the start of core() as reference
    int referenceStartInt = memoryBlock.startInt;

    if (iteration == -1)
    {
        iteration = 0;
    }

    // Per-slot string scratch for this level's leaf calculation strings; a
    // ScratchScope frees them when this invocation returns. collected (the
    // separate genScratchArenas page tier) holds interned
    // copies, so nothing this level builds needs to outlive it. Recursion nests
    // scopes by stack discipline -- an inner level allocates above this mark and
    // rewinds to its own, never touching these spans.
    const unsigned coreSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(coreSlot);
    ScratchScope coreScope(sArena);

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
        // existence category encodes !(>[bound](left)!(right)), so
        //   !(existence<N>[args]) = (>[bound](left)!(right)):
        // from left derive !right, and contrapositively from right derive
        // !left. The inner existence is resolved through the REGISTRY —
        // never through the prepared instruction: prepareIntegrationCore
        // commits no entity for a negated compound input (its polarity
        // guard), so the instruction is empty here.
        if (addedSpan.len >= 2 && addedSpan.ptr[0] == '!' && addedSpan.ptr[1] == '(') {
            const StrSpan innerExpr(addedSpan.ptr + 1, addedSpan.len - 1);
            const LogicalEntity* exLe =
                compiledEntity(extractExpressionUniversalSpan(innerExpr));
            if (exLe != nullptr && exLe->category == "existence"
                && exLe->elements.size() == 2) {
                // The two rules — left -> !right and right -> !left — come
                // from the existence entity's compiled `implications` list
                // (D-310): one compact instance
                // `(implication<N>[args])` per rule, u_p -> the inner
                // instance's argument p positionally. Each instance is a
                // product STATEMENT of the negated existence exactly like an
                // `and` element: a child of its key, registered at this
                // scope, and disintegrated through the implication branch of
                // this function, which expands it into the very rule the
                // on-the-spot construction produced (the compact's elements
                // through reconstructImplicationFullBindScratch — the bound
                // variable is the registry's placeholder, as for every mailed
                // compact) and writes the rule's `expansion` origin citing
                // the compact. The compact itself takes the `expansion`
                // origin the rule carried, citing the negated existence.
                assert(static_cast<int32_t>(exLe->implications.size())
                        == expectedExistenceImplicationCount(
                               static_cast<int32_t>(exLe->elements.size()))
                    && "negated-existence expansion: existence-implication list incomplete (I-209)");
                StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
                const int32_t sigN = getArgsSpans(StrSpan(exLe->signature), sigArgs,
                    ExecutionParameters::MAX_ARITY);
                StrSpan instArgs[ExecutionParameters::MAX_ARITY];
                const int32_t instN = getArgsSpans(innerExpr, instArgs,
                    ExecutionParameters::MAX_ARITY);
                assert(sigN == instN
                    && "negated-existence expansion: instance arity differs from the compiled existence");
                StrReplacement pairs[ExecutionParameters::MAX_ARITY];
                for (int32_t p = 0; p < sigN; ++p) {
                    pairs[p].key = sigArgs[p];
                    pairs[p].value = instArgs[p];
                }
                currentStatement = removeUPrefixScratch(sArena, addedSpan);
                const int maxOrig = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr;
                // Expansion antecedent: the negated existence, u_-stripped
                // (chapter rows surface u_-stripped expressions), a named
                // local so the span stays live across every door call; the
                // doors mint into originInterner, never sArena (I-3).
                const ScratchString expClean = removeUPrefixScratch(sArena, StrSpan(expr));
                const OriginDep expDeps[1] = { { StrSpan(expClean), StrSpan(validityName) } };
                // The instruction's marked goal is re-set by every
                // prepareIntegrationCore call; copy it onto the arena first so
                // the span never aliases the interner it is minted back into (I-3).
                const StrSpan goalView = instructions.markedGoal();
                const ScratchString goalCopy =
                    ScratchString::copyFrom(sArena, goalView.ptr, goalView.len);
                for (const std::string& tmpl : exLe->implications) {
                    // Per-instance scope: the instance bytes live through the
                    // recursion (the WorkInstruction copies what it keeps).
                    ScratchScope instScope(sArena);
                    const ScratchString inst =
                        replaceKeysScratch(sArena, StrSpan(tmpl), pairs, sigN);
                    const ScratchString instClean =
                        removeUPrefixScratch(sArena, StrSpan(inst));
                    collected.insertChild(StrSpan(currentStatement), StrSpan(instClean));
                    if (parameters.trackHistory && trackHistoryLocal) {
                        addOriginEncoded(memoryBlock.exprOriginMap,
                            memoryBlock.originInterner, StrSpan(instClean),
                            StrSpan(validityName), OriginTag::expansion,
                            expDeps, 1, maxOrig);
                        // Paired `mailOut.exprOriginMap` write (D-274): the
                        // compact registers as a statement and rides the
                        // delta; a receiver resolving its row needs the
                        // antecedent chain shipped alongside.
                        memoryBlock.addMailOutOrigin(StrSpan(instClean),
                            StrSpan(validityName), OriginTag::expansion,
                            expDeps, 1, maxOrig);
                    }
                    prepareIntegrationCore(StrSpan(inst), instructions, memoryBlock,
                        StrSpan(goalCopy));
                    disintegrateExprCore2(StrSpan(inst), instructions, memoryBlock,
                        iteration, collected, newVarMap, validityName,
                        trackHistoryLocal, allowOrDisintegration,
                        orSeedLevels, orSeedLevelCount, allowOrProbe,
                        parentWitness, witnessMeta);
                }

                return;
            }
        }

        // ---- Negated-AND De-Morgan expansion (or-twin) ----
        // !(op[args]) whose compiled definition body is an AND is a
        // disjunction in De-Morgan clothing: !(& C1 .. Cn) = !C1 v .. v !Cn,
        // with every disjunct at its TRUE polarity — a negated conjunct's
        // disjunct is its bare positive core (negateScratch cancellation,
        // I-175). The ordered leaves run the same consumption as a compiled
        // or fact — K mutual-exclusion rules at this scope plus the
        // two-route cohort machinery — with the negated compound itself as
        // the cohort signature: no orN operator exists or is minted, the
        // registry is only read (parallel-phase safe). The flat negated
        // statement still registers through the ensureKey below, exactly as
        // before; a leaf that is itself an or application re-disintegrates
        // as an ordinary or fact when a branch asserts it.
        if (addedSpan.len >= 2 && addedSpan.ptr[0] == '!' && addedSpan.ptr[1] == '(') {
            const StrSpan negCore = extractExpressionUniversalSpan(addedSpan);
            const LogicalEntity* negLe = compiledEntity(negCore);
            if (negLe != nullptr && negLe->category == "and") {
                const StrSpan innerInst(addedSpan.ptr + 1, addedSpan.len - 1);
                StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
                const int32_t sigN = getArgsSpans(StrSpan(negLe->signature),
                    sigArgs, ExecutionParameters::MAX_ARITY);
                StrSpan instArgs[ExecutionParameters::MAX_ARITY];
                const int32_t instN = getArgsSpans(innerInst, instArgs,
                    ExecutionParameters::MAX_ARITY);
                assert(sigN == instN
                    && "negated-AND door: instance arity differs from the "
                       "compiled definition");
                StrReplacement pairs[ExecutionParameters::MAX_ARITY];
                for (int32_t a = 0; a < sigN; ++a) {
                    pairs[a].key = sigArgs[a];
                    pairs[a].value = instArgs[a];
                }
                ScratchString leafHold[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                StrSpan orLeaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                int32_t orLeafN = 0;
                for (const std::string& rawElem : negLe->elements) {
                    assert(orLeafN < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                        && "negated-AND door: element count exceeds cap");
                    leafHold[orLeafN] = replaceKeysScratch(sArena,
                        StrSpan(rawElem), pairs, sigN);
                    orLeaves[orLeafN] =
                        negateScratch(sArena, StrSpan(leafHold[orLeafN]));
                    ++orLeafN;
                }
                currentStatement = removeUPrefixScratch(sArena, addedSpan);
                collected.ensureKey(StrSpan(currentStatement));
                consumeOrLeavesCohort(expr, addedSpan, orLeaves, orLeafN,
                    StrSpan(currentStatement), memoryBlock, collected,
                    validityName, trackHistoryLocal, allowOrDisintegration,
                    orSeedLevels, orSeedLevelCount, allowOrProbe, sArena,
                    instructions, iteration, newVarMap, parentWitness,
                    witnessMeta);
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
            disintegrateExprCore2(entElems[i], instructions, memoryBlock, iteration, collected, newVarMap, validityName, trackHistoryLocal, allowOrDisintegration, orSeedLevels, orSeedLevelCount, allowOrProbe, parentWitness, witnessMeta);
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
                disintegrateExprCore2(entElems[i], instructions, memoryBlock, iteration, collected, newVarMap, validityName, trackHistoryLocal, allowOrDisintegration, orSeedLevels, orSeedLevelCount, allowOrProbe, parentWitness, witnessMeta);
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

            // Recurse using the renamed instruction. A nested existence
            // reached through this path records THIS witness as its enclosing
            // witness (the relay-selection parent chain).
            for (int32_t i = 0; i < entElemN; ++i) {
                disintegrateExprCore2(renamedElemSpans[i], renamed, memoryBlock, iteration, collected, newVarMap, validityName, trackHistoryLocal, allowOrDisintegration, orSeedLevels, orSeedLevelCount, allowOrProbe, newVar, witnessMeta);
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

            // Relay-selection metadata: this witness's spawning compact is the
            // frame's instantiated existence (currentStatement); the enclosing
            // witness is the recursion's parent. Appended BEFORE processPath so
            // record order == newVarMap key order (the consumer asserts it).
            if (witnessMeta != nullptr)
                witnessMeta->append(StrSpan(newVar), StrSpan(currentStatement),
                                    parentWitness);

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

            // Relay-selection metadata — the int_ twin of the it_ append above
            // (one existence's two witnesses share instance and parent).
            if (witnessMeta != nullptr)
                witnessMeta->append(StrSpan(newVar), StrSpan(currentStatement),
                                    parentWitness);

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
        // OR-valued branch is created for a later hashburst. The whole
        // downstream — expansion history, K mutual-exclusion rules, depth
        // gate, two-route cohort machinery — lives in consumeOrLeavesCohort,
        // shared with the negated-AND De-Morgan door.
        StrSpan orLeaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        const int32_t orLeafN = flattenOrLeaves(
            instructions, hit, orLeaves,
            ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
        consumeOrLeavesCohort(expr, entSignature, orLeaves, orLeafN,
            StrSpan(currentStatement), memoryBlock, collected, validityName,
            trackHistoryLocal, allowOrDisintegration,
            orSeedLevels, orSeedLevelCount, allowOrProbe, sArena,
            instructions, iteration, newVarMap, parentWitness, witnessMeta);
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
    bool allowOrDisintegration,
    const int* orSeedLevels,
    int32_t orSeedLevelCount,
    bool allowOrProbe)
{
    int savedStartInt = memoryBlock.startInt;

    // collected is a page-tier (allocPage) container, so
    // it rides genScratchArenas -- NOT scratchArenas, whose allocBytes string
    // fill (prefixArgumentsWithU, below) would clobber a container page sharing
    // the slot when disintegration re-enters via the hypothetical / integration
    // paths (allocBytes targets pageHighWater()-1 and its ScratchScope rewind
    // poisons it). Same container-vs-string split as the request generators.
    const unsigned collSlot = (g_currentCoreId >= 0)
        ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    CollectedArena collected(&genScratchArenas().forSlot(collSlot));
    NewVarStore newVarMap(&genScratchArenas().forSlot(collSlot));
    WitnessMetaStore witnessMeta(&genScratchArenas().forSlot(collSlot));
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
    {
        RT_SCOPE_HERE("DISINT_PREPARE_CORE");
        prepareIntegrationCore(StrSpan(replExpr), instructions, memoryBlock, expr);
    }

    // The disintegration core now reads the arena WorkInstruction directly —
    // the former heap std::vector<LogicalEntity> bridge is gone.
    {
        RT_SCOPE_HERE("DISINT_CORE");
        disintegrateExprCore2(StrSpan(replExpr),
            instructions,
            memoryBlock,
            iteration,
            collected,
            newVarMap,
            validityName,
            trackHistoryLocal,
            allowOrDisintegration,
            orSeedLevels,
            orSeedLevelCount,
            allowOrProbe,
            /*parentWitness=*/StrSpan(),
            &witnessMeta);
    }

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
            // `s` is the statement whose disintegration produced the rule —
            // the carrier the install door records.
            out.implications.append(original, validity, s);
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
        // validity) straight to the channel; forEachSorted at
        // the caller reproduces the former std::set<EWV> order byte-for-byte.
        // (Sequenced or-disintegration: branch seeds ride
        // sameIterationInternalMail from the OR case itself, not this
        // channel — the absorb runs them through the full kernel pipeline.)
        for (int32_t id = 1; id <= finalStringStatements.count(); ++id)
            out.statements.append(finalStringStatements.keyAt(id), validityName);
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
    RT_SCOPE_HERE("DISINT_PASSES_AB");
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
                        // Ancestor-inclusive probe (D-288):
                        // a fired input-slot demand key lands at
                        // deeperOf(subkey constituents), which may be a strict
                        // ancestor of this deposit's scope (ordis branch).
                        if (isAdmittedIncludingAncestors(memoryBlock, StrSpan(ru), var, StrSpan(marked), validityName)) {
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
                            // Level capture = frame ∪ registry
                            // (D-327). The
                            // compound's registry levels row is written by the
                            // kernel's stmts-loop AFTER disintegrateExpr2
                            // returns, so on a fresh local deposit the registry
                            // lookup alone misses by construction and the
                            // record would park level-empty — the revival
                            // would then stamp the witness facts {-1}, hiding
                            // the compound's real levels from every downstream
                            // level union. The deposit's own levels are in
                            // hand as orSeedLevels; a re-arrival's registry
                            // row may carry levels the frame does not, so
                            // merge both (ascending runs, non-negative tier
                            // only — an all-non-derived park stays empty and
                            // the addStatement door re-stamps {-1} at
                            // revival).
                            int32_t levArr[256];
                            int32_t levN = 0;
                            {
                                int32_t regArr[256];
                                int32_t regN = 0;
                                const int32_t compoundLvlsId2 = lookupStatementLevels(
                                    memoryBlock.intStatementLevelsMap, memoryBlock.nameMap,
                                    StrSpan(topLevelExprClean), validityName);
                                if (compoundLvlsId2) {
                                    regN = coldIntRunAt(
                                        memoryBlock.intStatementLevelsMap,
                                        compoundLvlsId2, regArr, 256);
                                }
                                int32_t ai = 0, bi = 0;
                                while (ai < orSeedLevelCount && orSeedLevels[ai] < 0) ++ai;
                                while (bi < regN && regArr[bi] < 0) ++bi;
                                while (ai < orSeedLevelCount && bi < regN) {
                                    assert(levN < 256 && "park level union exceeds levArr");
                                    if (orSeedLevels[ai] < regArr[bi]) {
                                        levArr[levN++] = orSeedLevels[ai++];
                                    } else if (regArr[bi] < orSeedLevels[ai]) {
                                        levArr[levN++] = regArr[bi++];
                                    } else {
                                        levArr[levN++] = orSeedLevels[ai++];
                                        ++bi;
                                    }
                                }
                                while (ai < orSeedLevelCount) {
                                    assert(levN < 256 && "park level union exceeds levArr");
                                    levArr[levN++] = orSeedLevels[ai++];
                                }
                                while (bi < regN) {
                                    assert(levN < 256 && "park level union exceeds levArr");
                                    levArr[levN++] = regArr[bi++];
                                }
                            }
                            pendingRejections.addRejection(var, StrSpan(ru),
                                StrSpan(marked), StrSpan(topLevelExprClean),
                                sibSpans, sibN, levArr, levN);
                        }
                    }
                }
                // 2. Check for Integration Variable (int_...)
                else if (matchIntLevId(var, classLevel, classId)) {                    if (isAdmittedIntegration(memoryBlock, StrSpan(ru), var, StrSpan(marked), validityName)) {
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
        // wake: its constituent was admitted through another route. The
        // probe is VALUE-level (admissionRunHasRegularValue): an ordis-only
        // key is cohort-opening demand evidence and must not wake the
        // general rejectedMap.
        int64_t parkedPk = 0;
        if (lookupTemplateKey(memoryBlock.templateInterner, memoryBlock.nameMap,
                              markedExprF, validityName, parkedPk)) {
            const unsigned rdvSlot = (g_currentCoreId >= 0)
                ? static_cast<unsigned>(g_currentCoreId)
                : genScratchArenas().slotCount() - 1;
            if (admissionRunHasRegularValue(
                    memoryBlock.overallHashMemory.admissionMap, parkedPk,
                    genScratchArenas().forSlot(rdvSlot))) {
                revisitRejected2(markedExprF, memoryBlock, validityName);
            }
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

    // Append the normal statements (all at parent validity) straight to the
    // channel; forEachSorted
    // at the caller reproduces the former std::set<EWV> order byte-for-byte.
    // out.implications is already filled by addToFinal.
    // (Sequenced or-disintegration: branch seeds ride
    // sameIterationInternalMail from the OR case itself, not this channel —
    // the absorb runs them through the full kernel pipeline.)
    for (int32_t id = 1; id <= finalStringStatements.count(); ++id)
        out.statements.append(finalStringStatements.keyAt(id), validityName);

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

        // Per-witness signature ids for the relay selection below (mint order,
        // parallel to witnessMeta's records).
        const int32_t wTotal = newVarMap.varCount();
        int32_t* sigIdOf = (wTotal > 0)
            ? reinterpret_cast<int32_t*>(egArena.resolve(
                  egArena.alloc(wTotal * 4, 4)))
            : nullptr;

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
            sigIdOf[wid - 1] = allSigs.mint(sigKey);
            if (admittedVars.contains(witnessVar)) coveredSigs.mint(sigKey);
        }
        fullDisintegrationHappened = (allSigs.count() == coveredSigs.count());

        // Relay selection (D-284): mark the HIGHEST
        // uncovered existence groups — an uncovered group whose enclosing
        // group is covered (or that is the top compound) relays its spawning
        // compact; deeper uncovered groups ride inside it. The caller stages
        // the compacts for flag-5 mail (local statuses only). witnessMeta is
        // record-parallel to newVarMap's mint order (asserted).
        if (!fullDisintegrationHappened && wTotal > 0) {
            assert(witnessMeta.count() == wTotal
                && "relay selection: witnessMeta out of step with newVarMap");
            int32_t* parentIdx = reinterpret_cast<int32_t*>(egArena.resolve(
                egArena.alloc(wTotal * 4, 4)));
            for (int32_t i = 0; i < wTotal; ++i) {
                assert(equalSpans(witnessMeta.witnessAt(i),
                                  newVarMap.varAt(i + 1))
                    && "relay selection: witnessMeta order mismatch");
                const StrSpan parent = witnessMeta.parentAt(i);
                if (parent.empty()) {
                    parentIdx[i] = -1;
                } else {
                    const int32_t pid = newVarMap.lookupVar(parent);
                    assert(pid != 0
                        && "relay selection: enclosing witness not in newVarMap");
                    parentIdx[i] = pid - 1;
                }
            }
            bool* relayMask = reinterpret_cast<bool*>(egArena.resolve(
                egArena.alloc(wTotal, 1)));
            const int32_t selected = selectRelayWitnesses(sigIdOf, parentIdx,
                wTotal,
                [&](int32_t sid) {
                    return coveredSigs.lookup(allSigs.keyAt(sid)) != 0;
                },
                relayMask);
            if (selected > 0) {
                // Dedup by spawning-compact instance (the it_/int_ pair of one
                // existence selects together); the channel dedups on the
                // composite key, so repeated appends are no-ops.
                for (int32_t i = 0; i < wTotal; ++i) {
                    if (relayMask[i])
                        out.relayCompacts.append(witnessMeta.instanceAt(i),
                                                 validityName);
                }
            }
        }
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
    steward = std::make_unique<MemorySteward>(parameters.allow_ssd_deload);
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
        // Mirror the burst boundary into the diagnostics log so its telemetry
        // lines stay attributable to their hashburst.
        diagnosticsLog() << "--- hashburst " << it
                         << " active_bodies=" << activeBodies
                         << " total_exprs=" << totalExprs
                         << " ---" << std::endl;

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


/// @brief Alpha-variant test for a head-switched mirror — see the
///        declaration's Doxygen block in `prover.hpp` for the full
///        contract.
///
/// @details
/// Implementation notes: the candidate bijection π comes from a
/// simultaneous token walk over the two heads (token = `[A-Za-z0-9_]+`
/// run; anything else is delimiter and must match byte-for-byte); the
/// global verification applies π to each source chain link with the
/// token-boundary `ce::replaceKeysInString` and compares the link
/// multisets via one sort. Load-time / setup code — not on the statified
/// burst paths, so heap containers are in contract here.
///
/// @param original The source conjecture (compiled form).
/// @param mirror   Its `headSwitchOne` output (compiled form).
/// @return True iff the mirror is an alpha-variant of the source.
bool ExpressionAnalyzer::mirrorIsAlphaVariant(const std::string& original,
                                              const std::string& mirror) const {
    // 1. Disintegrate both sides.
    std::vector<std::tuple<std::string, std::vector<std::string>,
                           std::set<std::string>>> chainO, chainM;
    const std::string headO =
        ce::disintegrateImplication(original, chainO, coreExpressionMap);
    const std::string headM =
        ce::disintegrateImplication(mirror, chainM, coreExpressionMap);
    if (chainO.size() != chainM.size()) return false;

    // Bound-variable universe of the source (every binder-list entry).
    std::set<std::string> bound;
    for (const auto& link : chainO)
        for (const std::string& v : std::get<1>(link)) bound.insert(v);

    // 2. Candidate bijection from the token-aligned head pairing.
    const auto isTokByte = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9') || c == '_';
    };
    std::map<std::string, std::string> pi;     // source var -> mirror var
    std::map<std::string, std::string> piInv;
    {
        std::size_t i = 0, j = 0;
        while (i < headO.size() || j < headM.size()) {
            const bool ta = i < headO.size() && isTokByte(headO[i]);
            const bool tb = j < headM.size() && isTokByte(headM[j]);
            if (ta != tb) return false;
            if (!ta) {
                if (i >= headO.size() || j >= headM.size()
                    || headO[i] != headM[j]) return false;
                ++i; ++j;
                continue;
            }
            const std::size_t i0 = i, j0 = j;
            while (i < headO.size() && isTokByte(headO[i])) ++i;
            while (j < headM.size() && isTokByte(headM[j])) ++j;
            const std::string tokO = headO.substr(i0, i - i0);
            const std::string tokM = headM.substr(j0, j - j0);
            const bool boundO = bound.count(tokO) != 0;
            const bool boundM = bound.count(tokM) != 0;
            if (tokO != tokM && (!boundO || !boundM)) return false;
            if (boundO != boundM) return false;
            if (boundO) {
                const auto it = pi.find(tokO);
                if (it != pi.end()) {
                    if (it->second != tokM) return false;
                } else {
                    if (piInv.count(tokM)) return false;
                    pi[tokO] = tokM;
                    piInv[tokM] = tokO;
                }
            }
        }
    }
    // Complete π to a permutation: untouched bound variables map to
    // themselves (must not collide with an existing image).
    for (const std::string& v : bound) {
        if (pi.count(v)) continue;
        if (piInv.count(v)) return false;
        pi[v] = v;
        piInv[v] = v;
    }

    // 3. Global verification: π-image of the source links == mirror links
    //    as a multiset (premise + ordered bound-var list per link).
    const char SEP = '\x01';
    std::vector<std::string> imgO, keysM;
    imgO.reserve(chainO.size());
    keysM.reserve(chainM.size());
    for (const auto& link : chainO) {
        std::string key = ce::replaceKeysInString(std::get<0>(link), pi);
        for (const std::string& v : std::get<1>(link)) {
            key += SEP;
            const auto it = pi.find(v);
            key += (it != pi.end()) ? it->second : v;
        }
        imgO.push_back(std::move(key));
    }
    for (const auto& link : chainM) {
        std::string key = std::get<0>(link);
        for (const std::string& v : std::get<1>(link)) {
            key += SEP;
            key += v;
        }
        keysM.push_back(std::move(key));
    }
    std::sort(imgO.begin(), imgO.end());
    std::sort(keysM.begin(), keysM.end());
    return imgO == keysM;
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


/// @brief Compile a batch of established theorems to compact rule carriers and
///        store the batch ONCE in `store`'s mail log (declaration Doxygen in
///        `prover.hpp` carries the full contract).
///
/// @details
/// Disintegrate + rebuild each theorem in first-occurrence binder order (I-4),
/// compile to its compact, deposit as a main-scope statement with an empty level
/// set (I-51) plus the `originTag` / `compilation` origin rows; commit the batch
/// into `store`'s log, self-inject `store.mailIn`, raise `store.hasWork`.
/// Single-threaded, before any pull.
///
/// @param provedTheorems The theorems to seed (structural operators precompiled).
/// @param originTag Origin tag of the seed rows.
/// @param store The LB whose log stores the batch and whose inbox is self-injected.
/// @param nestedStore When non-null, the store of every theorem `anchorOnlyRule`
///        rejects (the anchor LB in incubator mode); null = no split.
/// @see collectMailAncestors, incubatorAnchorLb, anchorOnlyRule.
void ExpressionAnalyzer::broadcastTheorems(const std::vector<std::string>& provedTheorems,
                                           const std::string& originTag,
                                           Memory& store,
                                           Memory* nestedStore) {
    if (provedTheorems.empty()) return;

    // One batch per store; the compile order stays the list order (the
    // implication<N> numbering is observable), only the destination differs.
    Mail storeBatch;
    Mail nestedBatch;
    for (const std::string& thOriginal : provedTheorems) {
        const bool toNested = nestedStore != nullptr
            && !this->anchorOnlyRule(thOriginal);
        Mail& broadcastMail = toNested ? nestedBatch : storeBatch;

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
            // memoryBlock.level + 1. The allLevelsInvolved registration verdict
            // sealed in prover.hpp::dischargeToBeProved then turns false and the
            // drain refuses appendGlobalTheorem: the goal still closes (closure
            // is level-free), but the derived head is never recorded as proved
            // or broadcast.
            std::set<int> compactLevels;
            ExpressionWithValidity compactEv(compactImpl, "main");
            broadcastMail.statements.insert(std::make_pair(compactEv, compactLevels));
            if (parameters.trackHistory) {
                // I-52: cite the binary's CANONICAL reconstruction, not the
                // input finalTheorem. compileImplicationToCompact registers a
                // normalized premise order (and dedups alpha-equivalent bodies
                // to the first-seen entry), so finalTheorem's premise order
                // can differ from the registered elements even for a single
                // source; the verifier's check_compilation rebuilds from the
                // binary, so the citation must be the same reconstruction.
                // Mirror of the deferred-compaction drain in proveKernel.
                const std::string compactCore = extractExpressionUniversalSpan(StrSpan(compactImpl)).toStdString();
                auto cit = this->compiledExpressions.find(compactCore);
                assert(cit != this->compiledExpressions.end()
                    && "broadcastTheorems: compact form must resolve to a compiledExpressions entry");
                const std::vector<std::string>& elems = cit->second.elements;
                assert(!elems.empty()
                    && "broadcastTheorems: implication entry must have at least one element (head)");
                std::vector<std::string> canonicalKey(elems.begin(), elems.end() - 1);
                const std::string& canonicalHead = elems.back();
                const std::string canonicalOriginal = this->reconstructImplicationFullBind(canonicalKey, canonicalHead);
                addOrigin(broadcastMail.exprOriginMap, compactEv,
                    std::make_pair("compilation", std::vector<ExpressionWithValidity>{ ExpressionWithValidity(canonicalOriginal, "main") }),
                    (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }
        }
    }

    // New mail system (D-137): store the broadcast ONCE in `store`'s log --
    // every LB that lists `store` among its mail ancestors pulls it on the
    // normal ancestor walk -- and self-inject `store`, which never pulls its
    // own log. `store` is the root sentinel (reaching every LB, or only the
    // anchor LB in incubator mode -- collectMailAncestors) or, for an
    // incubator batch's externals, the anchor LB (reaching the anchor LB and
    // every LB below it). Single-threaded at LOAD, before any pull, so the
    // batch is in place for burst 1. broadcastMail carries only statements +
    // exprOriginMap, exactly what mergeBatchInto and commit keep.
    mergeBatchIntoMailIn(storeBatch, store.mailIn);
    this->mailLog.commit(&store, std::move(storeBatch));
    // WAKE DOOR 5 (D-194): the seed batch self-injects `store`'s mailIn
    // directly, so wake it here; every reader of its log is woken by mailPeek
    // on the next active-build (it sees the bumped commit count). A no-op on
    // burst 1 (LBs born dirty), load-bearing for the between-warm-up-and-main
    // broadcast where `store` may have gone quiescent.
    store.hasWork = true;
    // Shape-routed nested rules: the anchor LB's log (every LB below it) plus
    // the anchor LB's own inbox.
    if (nestedStore != nullptr && !nestedBatch.statements.empty()) {
        mergeBatchIntoMailIn(nestedBatch, nestedStore->mailIn);
        this->mailLog.commit(nestedStore, std::move(nestedBatch));
        nestedStore->hasWork = true;
    }

    std::cout << "Distributed knowledge to " << permanentBodies.size() << " memory blocks." << std::endl;
}

/// @brief Collect the mail ancestors an LB registers with `MailLog`, nearest
///        first (declaration Doxygen in `prover.hpp` carries the full contract).
///
/// @details
/// The whole `parentMemory` chain outside incubator mode; in incubator mode the
/// walk stops at the anchor LB, so an LB strictly below it never lists the root
/// sentinel and the root's log reaches only the anchor LB
/// (D-332).
///
/// @param lb  The LB being registered.
/// @param out Receives the ancestor list, nearest first (cleared first).
void ExpressionAnalyzer::collectMailAncestors(const Memory* lb,
                                              std::vector<const Memory*>& out) const {
    assert(lb != nullptr && "collectMailAncestors: null LB");
    out.clear();
    bool stoppedAtAnchor = false;
    for (const Memory* p = lb->parentMemory; p != nullptr; p = p->parentMemory) {
        out.push_back(p);
        if (parameters.incubator_mode && this->isAnchorLb(*p)) {
            stoppedAtAnchor = true;
            break;
        }
    }
    // Incubator contract: every LB strictly below the anchor LB has the anchor
    // LB on its chain (every incubator conjecture starts with the anchor
    // premise). The root sentinel and the anchor LB itself walk to the root.
    assert((!parameters.incubator_mode || stoppedAtAnchor
            || lb->parentMemory == nullptr || this->isAnchorLb(*lb))
        && "collectMailAncestors: incubator LB without an anchor LB on its chain");
}

/// @brief The anchor LB of an incubator grid -- the root sentinel's one anchor
///        child (declaration Doxygen in `prover.hpp` carries the full contract).
///
/// @details
/// Walks the root's children (`simpleMapStore.forEachChild`) and asserts exactly
/// one child carries the anchor prefix.
///
/// @return The anchor LB (never null).
Memory* ExpressionAnalyzer::incubatorAnchorLb() const {
    assert(parameters.incubator_mode
        && "incubatorAnchorLb: only an incubator grid has a single anchor LB");
    Memory* anchorLb = nullptr;
    int anchorChildren = 0;
    this->simpleMapStore.forEachChild(&this->body,
        [&](const StrSpan& /*key*/, Memory* child) {
            if (this->isAnchorLb(*child)) {
                anchorLb = child;
                ++anchorChildren;
            }
        });
    assert(anchorChildren == 1
        && "incubatorAnchorLb: the root sentinel must have exactly one anchor child");
    assert(anchorLb != nullptr);
    return anchorLb;
}

/// @brief True iff an incubator-derived theorem's rule can only fire at the
///        anchor LB (declaration Doxygen in `prover.hpp` carries the contract).
///
/// @details
/// Single premise → true. Two premises → true iff the second premise is an
/// operator application (not an equality, negation or implication) and the
/// head is `(=[a,b])` with one side a non-anchor argument of that operator and
/// the other side a `(1)`-typed anchor slot value. Otherwise false.
///
/// @param theorem The theorem text.
/// @return True iff the rule is anchor-only.
bool ExpressionAnalyzer::anchorOnlyRule(const std::string& theorem) const {
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain;
    const std::string head = ce::disintegrateImplication(theorem, chain, this->coreExpressionMap);
    assert(!chain.empty() && "anchorOnlyRule: a theorem without premises");
    // The first premise is an anchor: this batch's, or an earlier incubator
    // batch's for a loaded theorem (the cross-anchor bridge conjecture
    // `current anchor -> previous anchor` derives that premise at the anchor
    // LB). The earlier anchor need not be a core expression of this batch.
    const std::string& anchorPremise = std::get<0>(chain[0]);
    assert(anchorPremise.rfind("(Anchor", 0) == 0
        && "anchorOnlyRule: an incubator-derived theorem starts with an anchor premise");
    if (chain.size() == 1) return true;
    if (chain.size() != 2) return false;

    const std::string& premise2 = std::get<0>(chain[1]);
    const bool operatorPremise = premise2.size() > 2 && premise2[0] == '('
        && premise2[1] != '=' && premise2[1] != '>';
    if (!operatorPremise) return false;
    if (head.rfind("(=[", 0) != 0) return false;
    const std::vector<std::string> headArgs = ce::getArgs(head);
    if (headArgs.size() != 2) return false;

    // The anchor digits are the anchor's slot values that an equality can
    // mention: every slot of an incubator anchor other than the carrier set
    // and the function symbols, which are never equated. Reading the slot
    // values off the premise (not the definition sets) keeps the rule valid
    // for an earlier batch's anchor that this batch's configuration does not
    // define.
    const std::vector<std::string> anchorArgs = ce::getArgs(anchorPremise);
    const std::set<std::string> anchorArgSet(anchorArgs.begin(), anchorArgs.end());
    const std::set<std::string>& digits = anchorArgSet;
    std::set<std::string> operatorOutputs;
    for (const std::string& a : ce::getArgs(premise2)) {
        if (anchorArgSet.count(a) == 0) operatorOutputs.insert(a);
    }
    const auto connects = [&](const std::string& x, const std::string& d) {
        return operatorOutputs.count(x) != 0 && digits.count(d) != 0;
    };
    return connects(headArgs[0], headArgs[1]) || connects(headArgs[1], headArgs[0]);
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
    // both the miss-branch singleton and the unconditional insert. The
    // filtered read keeps the {-1} non-derived tier out of the union.
    int lvRun[256];
    int32_t lvN = 0;
    const int32_t originLvlsId = lookupStatementLevels(
        memoryBlock.intStatementLevelsMap, memoryBlock.nameMap, expr, validityName);

    if (originLvlsId) {
        lvN = coldIntRunNonNegAt(memoryBlock.intStatementLevelsMap, originLvlsId,
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

    // 7. Deposit NEW STATEMENTS with NEW validity name on internal mail

    // The constituents ride sameIterationInternalMail — the absorb runs each
    // through the full kernel pipeline (addStatement shape dispatch with the
    // equality / negated-equality gateways' mirror pair registration, Site F
    // ancestor dedup, equivalence classes), the same channel the or-branch
    // seeds ([I-176](../../../docs/agentic_swdd/30_invariants.md#i-176)) and
    // the ordis revival use. No direct registration here: a direct upsert
    // cannot maintain the gateways' mirror pair invariant. stmtSpan aliases
    // out.statements' storage and newVldSpan the owned decode copy
    // `newValidityName`; the deposit's two flat encodes write only the names
    // table, so neither span moves (I-3).
    out.statements.forEachSorted([&](StrSpan stmtSpan, StrSpan /*v*/) {
        const StrSpan newVldSpan(newValidityName);
        memoryBlock.mutatedThisBurst = true;
        insertInternalStatement(memoryBlock.sameIterationInternalMail,
            memoryBlock.nameMap, stmtSpan, newVldSpan, lvRun, lvN);
        if (parameters.trackHistory) {
            // Terminal `hypothesis` line — one tag, zero dependencies, at the
            // hypo validity ONLY. Hypothesis constituents steer proof
            // direction but are not part of any proof: no walk continues
            // through this line, and it must never be written at the parent
            // or main scope (the 2026-04-10 orphan-origin buildStack abort).
            addInternalMailOrigin(memoryBlock.sameIterationInternalMail,
                memoryBlock.originInterner, stmtSpan, newVldSpan,
                OriginTag::hypothesis, nullptr, 0,
                (parameters.compressor_mode
                     ? parameters.compressor_max_origins_per_expr
                     : parameters.max_origin_per_expr));
        }
    });
}

//#pragma optimize("", off)

/// @brief Grid-wide axed-anchor pre-pass: walk the LB tree once; outside
///        recursion subtrees every LB under the batch anchor mints its
///        axed x-names into `intAxedVariables` AND registers its
///        x-substituted anchor expression as a `{-1}`-tier statement with
///        an `anchor handling` history line; recursion subtrees take one
///        of two containment modes (set-only, or the anchor-numeral
///        exception).
///
/// @details
/// For each LB whose ancestor chain reaches the batch anchor LB, the pass
/// collects every argument cited by the chain's exprKeys (the trace) and
/// x-copies exactly the anchor's `(1)`-typed slots whose value appears in
/// that trace: slot value `6` becomes `x6`, minted into the LB's
/// `intAxedVariables`. The x-copy is the anchor-premise completion for
/// rules whose element arguments collide with anchor slot values (the
/// element-vs-anchor-slot collision, I-36 family); the minted axed set
/// makes the containment checks — the `addExprToMemoryBlock` prologue,
/// the `addStatement` door, and the orbit-commit refusal — drop every
/// DERIVED x-citing deposit, so the copies never breed x-facts. The
/// statement write is direct — deliberately bypassing those doors.
///
/// Recursion subtrees (D-272,
/// I-188): a block-#1 root whose goal (the
/// theorem head, queued verbatim at creation) cites a `(1)`-typed anchor
/// slot value puts its whole subtree in the ANCHOR-NUMERAL EXCEPTION mode
/// — untouched by this pass, so the ancestor's axed-anchor statement
/// arriving by mail registers at the inert door and the historical
/// premise-completion machinery runs there (anchor-numeral theorems such
/// as unit-product need it). Every other recursion subtree — block-#2
/// `_induction_` side-chains always included — is SET-ONLY: axed names
/// minted, no statement, the mailed ancestor form refused at the armed
/// door (a descendant's trace-accumulated axed set is a superset of its
/// ancestors'), keeping those induction sub-blocks entirely x-free.
///
/// @param mb Subtree root to process; the sole production caller passes
///           the grid root once per grid build (from `buildGrid`).
/// @param axedMode Threaded subtree mode: 0 outside recursion subtrees,
///                 1 set-only containment, 2 anchor-numeral exception
///                 (see the mode comment in the body); callers pass 0.
/// @invariant Registration is level-tier `{-1}` (definitional, not
///            derived) and dedup-guarded by the `intKnownStatements`
///            lookup, so re-walks never double-register.
/// @see `addExprToMemoryBlock` — the axed-containment door; SwDD
///      `20_core_concepts/06_anchors_and_scopes.md`.
void ExpressionAnalyzer::prehandleAnchor(Memory* mb, int axedMode) {
    if (mb == nullptr) return;

    // Recursion-subtree mode, threaded down the walk (only subtree roots
    // carry `isPartOfRecursion`):
    //   0 — outside recursion subtrees: mint + register (full treatment).
    //   1 — set-only containment: mint the axed names so the door refuses
    //       every x-citing deposit (the mailed ancestor-form axed anchor
    //       included); no statement. The subtree stays entirely x-free.
    //   2 — anchor-numeral exception: the subtree is untouched — no mint,
    //       no registration — so the ancestor's axed-anchor statement
    //       arriving by mail registers at the inert door and the
    //       historical premise-completion machinery runs inside this
    //       grid's induction blocks. Chosen at a block-#1 recursion root
    //       whose goal (the theorem head verbatim) cites a `(1)`-typed
    //       anchor slot value; block-#2 roots (`_induction_` equality
    //       side-chains) always take mode 1 — their synthetic
    //       `(=[digitArg,zero])` goal cites the zero numeral by
    //       construction, not because the theorem is about numerals.
    int mode = axedMode;
    const bool isSubtreeRoot = (axedMode == 0) && mb->isPartOfRecursion;
    if (isSubtreeRoot) mode = 1;

    const bool isAnchorLB = this->isAnchorLb(*mb);

    // Skip the body work for the Anchor LB itself; still recurse below.
    // Mode-2 descendants skip it too — their whole subtree is untouched.
    if (!isAnchorLB && mode != 2) {

        // 1. Trace the hierarchy to find the specific Anchor Key
        std::string anchorExprKey;
        Memory* current = mb;
        std::set<std::string> traceVariables; // Collect variables from trace

        while (current != nullptr) {
            if (!current->exprKey().empty()) {
                // Check if this ancestor is the Anchor LB
                if (this->isAnchorLb(*current)) {
                    anchorExprKey = current->exprKey();
                    break;
                }
                // Collect variables from the current trace element's key
                std::vector<std::string> kArgs = ce::getArgs(current->exprKey());
                traceVariables.insert(kArgs.begin(), kArgs.end());
            }
            current = current->parentMemory;
        }

        // 2. If we found an anchor ancestor, proceed
        if (!anchorExprKey.empty()) {
            std::vector<std::string> args = ce::getArgs(anchorExprKey);
            std::map<std::string, std::string> replacementMap;

            // 3. Collect the (1)-typed slot VALUES and build the
            //    replacement map (trace-guarded). Minting is deferred until
            //    the subtree mode is final: a mode-2 root must stay
            //    entirely untouched.
            std::set<std::string> slotValues;
            for (const auto& [slot, pattern] : this->anchorInfo.definitionSets) {
                if (pattern == "(1)") {
                    try {
                        int index = std::stoi(slot) - 1;
                        if (index >= 0 && index < static_cast<int>(args.size())) {
                            std::string originalVar = args[index];
                            slotValues.insert(originalVar);

                            // Check if the variable exists in the trace
                            if (traceVariables.find(originalVar) != traceVariables.end()) {
                                // Only apply x-prefix if not already present
                                if (originalVar.rfind("x", 0) != 0) {
                                    replacementMap[originalVar] = "x" + originalVar;
                                }
                            }
                        }
                    }
                    catch (...) {
                        // Ignore malformed slots
                    }
                }
            }

            // 3b. Anchor-numeral exception: at a block-#1 recursion root,
            //     scan the LB's own goals (the theorem head, queued
            //     verbatim at recursion-block creation) for a (1)-typed
            //     slot value cited as a whole token. A hit upgrades the
            //     subtree to mode 2 (untouched).
            if (isSubtreeRoot
                && mb->exprKey().find("_induction_") == std::string::npos
                && !slotValues.empty()) {
                const auto citesSlotValue = [&slotValues](const std::string& text) {
                    const auto isTok = [](char c) {
                        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z')
                            || (c >= 'A' && c <= 'Z') || c == '_';
                    };
                    size_t i = 0;
                    while (i < text.size()) {
                        if (!isTok(text[i])) { ++i; continue; }
                        size_t j = i;
                        while (j < text.size() && isTok(text[j])) ++j;
                        if (slotValues.count(text.substr(i, j - i)) != 0)
                            return true;
                        i = j;
                    }
                    return false;
                };
                for (const DecodedToBeProvedRow& row :
                     decodeToBeProvedSorted(mb->intToBeProved, mb->nameMap)) {
                    if (citesSlotValue(row.original)) { mode = 2; break; }
                }
            }

            // 3c. Mint the axed names (modes 0 and 1 — the armed door).
            if (mode != 2) {
                for (const auto& [originalVar, xVar] : replacementMap) {
                    mb->intAxedVariables.mint(mb->nameMap.encode(xVar));
                }
            }

            // 4. Create and Add the Anchor Expression using the map —
            //    mode 0 only (D-272): mode-1
            //    subtrees mint the axed set above but never hold the
            //    axed-anchor statement, so no x-citing firing can exist
            //    there; mode-2 subtrees receive the ancestor's form by
            //    mail instead.
            if (!replacementMap.empty() && mode == 0) {
                std::string replacedAnchor = ce::replaceKeysInString(anchorExprKey, replacementMap);

                EncodedExpression enc(replacedAnchor, "main");

                // Add only if not already present
                const StatementFlags* anchorRow = lookupStatementFlags(
                    mb->intKnownStatements, mb->nameMap,
                    enc.original, enc.validityName);
                if (anchorRow == nullptr) {

                    // Anchors are definitional, not derived — the {-1}
                    // non-derived tier, transparent to level accounting.
                    const int lv0[1] = { -1 };

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
                        /*local=*/true);

                    // L3 span-record door. The "anchor handling" antecedent
                    // (the original anchor expression `anchorExprKey`) and the
                    // KEY (`replacedAnchor`) are in-hand std::string locals;
                    // "main" is a static literal.
                    const StrSpan pMain("main", 4);
                    const OriginDep pDeps[1] = { { StrSpan(anchorExprKey), pMain } };
                    addOriginEncoded(mb->exprOriginMap, mb->originInterner, StrSpan(replacedAnchor), pMain, OriginTag::anchorHandling, pDeps, 1, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    // The history line is written locally on the same LB
                    // that registers the statement, so buildStack resolves
                    // any firing that cites the axed anchor as an
                    // antecedent. The statement also rides the Delta into
                    // mailOut, but every descendant's axed set is a superset
                    // of this LB's (the trace accumulates down the chain), so
                    // the mailed copy is refused at descendant doors rather
                    // than re-registered — recursion subtrees in particular
                    // never hold it (D-272).
                }
            }
        }
    }

    // 5. Recurse into children, threading the subtree mode
    simpleMapStore.forEachChild(mb, [&](const gl::StrSpan&, Memory* child) {
        prehandleAnchor(child, mode);
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
    if (parameters.allow_ssd_deload && parameters.enable_extent_deload) {
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
        // The ancestor list is the LB's whole chain outside incubator mode
        // and stops at the anchor LB inside it (collectMailAncestors,
        // D-332).
        std::vector<const Memory*> ancestors;
        for (Memory* lb : permanentBodies) {
            if (!lb) continue;
            this->collectMailAncestors(lb, ancestors);
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
            // Normalize-recognize-skip (maintainer-designed): a mirror that
            // is the SAME statement as its source up to bound-variable
            // renaming (the totality shape — symmetric binder prefix with
            // identical guards) would prove the identical theorem twice on
            // its own conjecture grid. Recognize the alpha-variant and skip
            // scheduling it; genuinely different mirrors (the Peano or0
            // parents, predicate-swapping shapes) keep being scheduled so
            // the pair or-construction stays intact.
            if (mirrorIsAlphaVariant(compiled, mirror)) {
                diagnosticsLog() << "[head-switch pre-emit] mirror is an "
                             "alpha-variant of its source - skipped: "
                          << conj << std::endl;
                continue;
            }
            if (seen.insert(mirror).second) mirrors.push_back(mirror);
        }
        if (!mirrors.empty()) {
            diagnosticsLog() << "[head-switch pre-emit] adding "
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
            // Theorems proved by earlier batches: stored in the root's log
            // (every LB outside incubator mode; only the anchor LB inside it --
            // D-332). In incubator mode a
            // nested rule (anchorOnlyRule false) goes to the anchor LB's log
            // instead, so the premise LBs below the anchor receive it.
            this->broadcastTheorems(compiledProved, "broadcast", this->body,
                parameters.incubator_mode ? this->incubatorAnchorLb() : nullptr);
        }

        if (!externalTheorems.empty()) {
            // Same precompile treatment as provedTheorems — externals may
            // carry raw !(&...) / !(>...) that would otherwise reach LB
            // memory uncompiled.
            std::vector<std::string> compiledExternals = externalTheorems;
            for (auto& et : compiledExternals) {
                this->precompileStructuralOperators(et);
            }
            // Keep the compiled twins for the post-run registry export —
            // base-form externals are registry-independent on the wire, but
            // chapter citations carry these compiled forms.
            this->precompiledExternalsExport = compiledExternals;
            std::cout << "Injecting " << compiledExternals.size()
                      << " external theorems via broadcast..." << std::endl;
            // Externals are multi-premise pool lemmas that fire in the premise
            // LBs below the anchor, so they must reach EVERY LB: in incubator
            // mode the root's log has one reader (the anchor LB), hence the
            // batch is stored in the anchor LB's log, which the anchor LB
            // (self-injected) and every LB below it read.
            Memory& externalsStore = parameters.incubator_mode
                ? *this->incubatorAnchorLb()
                : this->body;
            this->broadcastTheorems(compiledExternals, "externally provided theorem",
                                    externalsStore, nullptr);
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
            // Zero-circulation contract: proved-not-broadcast rows never
            // seed the export seam's or construction either.
            if (std::get<1>(tpl) == "proved not broadcast") continue;
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

    // Existence theorem: chain has shared premises + one or more binder-free
    // negated premises (!d_1, !d_2, ...), head = the last disjunct.

    // Every binder-free negated premise is a disjunct of the OR:
    // classically  !d_1 → (!d_2 → h)  ⟺  d_1 ∨ d_2 ∨ h, so the whole
    // hypothesis chain folds into one n-ary OR head. A negated premise
    // that binds variables cannot fold — its binder would orphan (the
    // or head's arguments must be bound by the surviving premise links)
    // — so it stays a shared premise.
    std::vector<int> foldedIdxs;
    for (std::size_t i = 1; i < chain1.size(); ++i) {
        const std::string& prem = std::get<0>(chain1[i]);
        if (!prem.empty() && prem[0] == '!' && std::get<1>(chain1[i]).empty()) {
            foldedIdxs.push_back((int)i);
        }
    }
    if (foldedIdxs.empty()) return "";

    // Build disjuncts in their TRUE polarity, chain order, head LAST.
    // A premise disjunct is the un-negated form of its negated premise;
    // the head enters VERBATIM — a negated head stays a negated disjunct.
    // Registered or-elements carry each disjunct's real sign; every
    // expansion negates elements with double-negation cancellation.
    // Storing only the positive core would flip a negated disjunct's
    // sign on round-trip and register a FALSE or theorem
    // ((a=b) OR (c=d) instead of (a=b) OR !(c=d)).
    std::vector<std::string> disjuncts;
    disjuncts.reserve(foldedIdxs.size() + 1);
    for (int idx : foldedIdxs) {
        disjuncts.push_back(std::get<0>(chain1[idx]).substr(1));  // strip !
    }
    disjuncts.push_back(head1);

    // Collect unique args across all disjuncts (ordered by first appearance)
    std::vector<std::string> orArgs;
    {
        std::set<std::string> seen;
        for (const std::string& d : disjuncts) {
            for (const auto& a : ce::getArgs(d)) {
                if (seen.insert(a).second) orArgs.push_back(a);
            }
        }
    }

    // Build the u_-canonical elements FIRST — the ordered element list is
    // the or-operator's registry identity.
    std::map<std::string, std::string> argToU;
    for (std::size_t i = 0; i < orArgs.size(); ++i) {
        argToU[orArgs[i]] = "u_" + std::to_string(i + 1);
    }
    std::vector<std::string> elems;
    elems.reserve(disjuncts.size());
    for (const std::string& d : disjuncts) {
        elems.push_back(ce::replaceKeysInString(d, argToU));
    }

    // Registry dedup-or-mint (I-23) through the one shared site. Exact
    // element equality implies the identical u_-position mapping, so the
    // instance args below line up with a reused signature.
    const std::string orCoreName =
        findOrMintOrOperator(elems, static_cast<int>(orArgs.size()));

    // Build compiled OR head: (or0[actual_args])
    std::string compiledOrHead = "(" + orCoreName + "[";
    for (std::size_t i = 0; i < orArgs.size(); ++i) {
        if (i > 0) compiledOrHead += ",";
        compiledOrHead += orArgs[i];
    }
    compiledOrHead += "])";

    {
        // Report the flat De Morgan base form: each disjunct negated with
        // double-negation cancellation.
        std::string negatedAnd = "!(&";
        for (const std::string& d : disjuncts) {
            negatedAnd += (!d.empty() && d[0] == '!') ? d.substr(1) : "!" + d;
        }
        negatedAnd += ")";
        std::cout << "OR compiled: " << negatedAnd << " -> " << compiledOrHead << std::endl;
    }

    // Build the shared premises (everything except the folded negated premises)
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> sharedChain;
    for (std::size_t i = 0; i < chain1.size(); ++i) {
        if (std::binary_search(foldedIdxs.begin(), foldedIdxs.end(), (int)i)) continue;
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

// Doxygen at the declaration (prover.hpp).
std::string ExpressionAnalyzer::findOrMintOrOperator(
    const std::vector<std::string>& elems, int arity, bool* mintedOut) {

    if (mintedOut != nullptr) *mintedOut = false;

    // Registry dedup (I-23): reuse an existing or-category entry whose
    // ordered element list matches exactly — the same (elements, category)
    // identity the binary load path keys repetitionExclusionMap on.
    // Minting unconditionally would give the same OR structure a fresh
    // name per constructing batch, and the verifier's registry comparison
    // then fails on the operator name. compiledExpressions is name-ordered
    // — the scan is deterministic.
    for (const auto& [name, le] : compiledExpressions) {
        if (le.category == "or" && le.elements == elems) {
            return name;
        }
    }

    const std::string orCoreName = "or" + std::to_string(orCounter);
    std::string sigArgs;
    for (int i = 0; i < arity; ++i) {
        if (i > 0) sigArgs += ",";
        sigArgs += "u_" + std::to_string(i + 1);
    }
    const std::string signature = "(" + orCoreName + "[" + sigArgs + "])";
    LogicalEntity orLe("or", elems, signature, arity);
    compiledExpressions.insert(std::make_pair(orCoreName, orLe));

    // Also register in coreExpressionMap for expandSignature/disintegration.
    ce::CoreExpressionConfig orCfg;
    orCfg.arity = arity;
    orCfg.signature = signature;
    coreExpressionMap.insert(std::make_pair(orCoreName, orCfg));
    orCounter++;

    // Or mint hook: the K-rule compacts ride the or's entry
    // (D-309); the subset-exclusion family is appended
    // at the preMintReducedOrs seam once the reduced closure exists.
    compileOrKRules(orCoreName);

    if (mintedOut != nullptr) *mintedOut = true;
    return orCoreName;
}

// Doxygen at the declaration (prover.hpp).
void ExpressionAnalyzer::flattenRegistryOrLeaves(
    const LogicalEntity& orLe, std::vector<std::string>& outLeaves) const {

    assert(orLe.category == "or"
        && "flattenRegistryOrLeaves: entity must be or-category");

    int depth = 0;
    const auto walk = [&](const LogicalEntity& node, const std::string& instance,
                          const auto& self) -> void {
        assert(depth < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
            && "flattenRegistryOrLeaves: OR depth exceeds cap");
        ++depth;
        const std::vector<std::string> sigArgs = ce::getArgs(node.signature);
        const std::vector<std::string> instArgs = ce::getArgs(instance);
        assert(sigArgs.size() == instArgs.size()
            && "flattenRegistryOrLeaves: instance arity differs from compiled or");
        std::map<std::string, std::string> subst;
        for (std::size_t a = 0; a < sigArgs.size(); ++a) {
            subst[sigArgs[a]] = instArgs[a];
        }
        for (const std::string& rawElem : node.elements) {
            const std::string elem = ce::replaceKeysInString(rawElem, subst);
            const LogicalEntity* childLe =
                (!elem.empty() && elem[0] == '(')
                ? compiledEntity(extractExpressionUniversalSpan(StrSpan(elem)))
                : nullptr;
            if (childLe != nullptr && childLe->category == "or") {
                self(*childLe, elem, self);
            } else {
                outLeaves.push_back(elem);
            }
        }
        --depth;
    };
    walk(orLe, orLe.signature, walk);
    assert(outLeaves.size() >= 2
        && "flattenRegistryOrLeaves: or entity flattens to fewer than two leaves");
}

// Doxygen at the declaration (prover.hpp).
int ExpressionAnalyzer::renumberULeaves(std::vector<std::string>& leaves,
                                        std::vector<std::string>* orderOut) {
    std::vector<std::string> order;
    std::set<std::string> seen;
    for (const std::string& d : leaves) {
        for (const std::string& a : ce::getArgs(d)) {
            assert(a.rfind("u_", 0) == 0
                && "renumberULeaves: registry leaf argument is not a u_ token");
            if (seen.insert(a).second) order.push_back(a);
        }
    }
    if (orderOut != nullptr) *orderOut = order;
    std::map<std::string, std::string> ren;
    for (std::size_t i = 0; i < order.size(); ++i) {
        ren[order[i]] = "u_" + std::to_string(i + 1);
    }
    for (std::string& d : leaves) {
        d = ce::replaceKeysInString(d, ren);
    }
    return static_cast<int>(order.size());
}

// Doxygen at the declaration (prover.hpp).
void ExpressionAnalyzer::preMintReducedOrs() {
    std::vector<std::string> work;
    std::set<std::string> queued;
    for (const auto& [name, le] : compiledExpressions) {
        if (le.category == "or" && queued.insert(name).second) {
            work.push_back(name);
        }
    }
    for (std::size_t w = 0; w < work.size(); ++w) {
        const LogicalEntity* le = compiledEntity(StrSpan(work[w]));
        assert(le != nullptr && le->category == "or"
            && "preMintReducedOrs: queued name must resolve to an or entity");
        std::vector<std::string> leaves;
        flattenRegistryOrLeaves(*le, leaves);
        const int k = static_cast<int>(leaves.size());
        assert(k <= ExecutionParameters::kMaxReducedOrLeaves
            && "preMintReducedOrs: or leaf count exceeds kMaxReducedOrLeaves");
        if (k < 3) continue;
        for (int i = 0; i < k; ++i) {
            std::vector<std::string> reduced;
            reduced.reserve(static_cast<std::size_t>(k) - 1);
            for (int j = 0; j < k; ++j) {
                if (j != i) reduced.push_back(leaves[j]);
            }
            const int arity = renumberULeaves(reduced);
            bool minted = false;
            const std::string rname =
                findOrMintOrOperator(reduced, arity, &minted);
            if (minted && queued.insert(rname).second) {
                work.push_back(rname);
            }
        }
    }

    // Or-implication completion (D-309). Every or now
    // has its reduced closure, so the subset-exclusion compacts can resolve
    // their heads. Name-ordered snapshot (deterministic; the compiles insert
    // implication entries into the same map). An or loaded from a binary
    // written before the field existed compiles its K-rules here; any
    // other or with an empty list is a missed mint hook.
    std::vector<std::string> orNames;
    for (const auto& [name, le] : compiledExpressions) {
        if (le.category == "or") orNames.push_back(name);
    }
    for (const std::string& name : orNames) {
        if (compiledEntity(StrSpan(name))->implications.empty()) {
            assert(orsAwaitingImplications.count(name) == 1
                && "preMintReducedOrs: an or minted without its K-rule compacts (missed mint hook)");
            compileOrKRules(name);
        }
        const LogicalEntity* le = compiledEntity(StrSpan(name));
        std::vector<std::string> leaves;
        flattenRegistryOrLeaves(*le, leaves);
        const int32_t k = static_cast<int32_t>(leaves.size());
        if (k >= 3 && static_cast<int32_t>(le->implications.size()) == k) {
            compileOrSubsetExclusions(name);
        }
        assert(static_cast<int32_t>(compiledEntity(StrSpan(name))->implications.size())
                == expectedOrImplicationCount(k)
            && "preMintReducedOrs: or-implication list incomplete after the closure seam");
    }
    orsAwaitingImplications.clear();

    // Existence-implication completion (D-310):
    // an existence loaded from a binary written before the field existed
    // compiles its two compacts here; any other two-element existence with
    // an empty list is a missed mint hook. Name-ordered snapshot (the
    // compiles insert implication entries into the same map).
    std::vector<std::string> exNames;
    for (const auto& [name, le] : compiledExpressions) {
        if (le.category == "existence") exNames.push_back(name);
    }
    for (const std::string& name : exNames) {
        const int32_t elemCount = static_cast<int32_t>(
            compiledEntity(StrSpan(name))->elements.size());
        if (compiledEntity(StrSpan(name))->implications.empty()
            && expectedExistenceImplicationCount(elemCount) > 0) {
            assert(existencesAwaitingImplications.count(name) == 1
                && "preMintReducedOrs: an existence minted without its implication compacts (missed mint hook)");
            compileExistenceImplications(name);
        }
        assert(static_cast<int32_t>(compiledEntity(StrSpan(name))->implications.size())
                == expectedExistenceImplicationCount(elemCount)
            && "preMintReducedOrs: existence-implication list incomplete after the seam");
    }
    existencesAwaitingImplications.clear();
}

/// @brief Compile an or's K-rule compacts into its `implications` list.
/// @details Doxygen contract at the declaration (prover.hpp).
/// @param orName Registry name of an or-category entity.
/// @see `compileOrSubsetExclusions`, `preMintReducedOrs`.
void ExpressionAnalyzer::compileOrKRules(const std::string& orName) {
    const auto it = compiledExpressions.find(orName);
    assert(it != compiledExpressions.end() && it->second.category == "or"
        && "compileOrKRules: name must resolve to an or entity");
    // Already carrying its list (registry reuse / reload re-registration):
    // a defined state, not a failure.
    if (!it->second.implications.empty()) return;

    std::vector<std::string> leaves;
    flattenRegistryOrLeaves(it->second, leaves);
    const std::size_t k = leaves.size();
    const std::size_t arity = ce::getArgs(it->second.signature).size();

    std::vector<std::string> compacts;
    compacts.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        std::vector<std::string> premises;
        premises.reserve(k - 1);
        for (std::size_t j = 0; j < k; ++j) {
            // negate, not a blind "!" prefix: a disjunct carries its true
            // polarity, so a negated leaf's exclusion premise is its bare
            // positive core (I-175).
            if (j != i) premises.push_back(negate(leaves[j]));
        }
        const std::string rule = reconstructImplicationFullBind(premises, leaves[i]);
        const std::string compact = prefixArgumentsWithU(compileImplicationToCompact(rule));
        // Every leaf occurs in every K-rule, so the compact spans the or's
        // full token set — its argument list is a permutation of the or's
        // signature tokens.
        assert(ce::getArgs(compact).size() == arity
            && "compileOrKRules: K-rule compact does not span the or's tokens");
        compacts.push_back(compact);
    }
    // Re-find: the compiles above insert implication entries into the same
    // map (node-stable, but the re-fetch is the honest form).
    compiledExpressions.find(orName)->second.implications = std::move(compacts);
}

/// @brief Append an or's subset-exclusion compacts to its `implications`
///        list, after the K-rules.
/// @details Doxygen contract at the declaration (prover.hpp).
/// @param orName Registry name of an or-category entity with 3 ≤ k leaves.
/// @see `compileOrKRules`, `preMintReducedOrs`.
void ExpressionAnalyzer::compileOrSubsetExclusions(const std::string& orName) {
    const auto it = compiledExpressions.find(orName);
    assert(it != compiledExpressions.end() && it->second.category == "or"
        && "compileOrSubsetExclusions: name must resolve to an or entity");
    std::vector<std::string> leaves;
    flattenRegistryOrLeaves(it->second, leaves);
    const int32_t k = static_cast<int32_t>(leaves.size());
    assert(k >= 3 && k <= ExecutionParameters::kMaxReducedOrLeaves
        && "compileOrSubsetExclusions: leaf count outside [3, kMaxReducedOrLeaves]");
    assert(static_cast<int32_t>(it->second.implications.size()) == k
        && "compileOrSubsetExclusions: precondition — the list holds exactly the K-rules");
    const std::size_t arity = ce::getArgs(it->second.signature).size();

    std::vector<std::string> compacts;
    // j = 1..k-2 excluded disjuncts; size-j index subsets in lexicographic
    // order — the disintegrator's enumeration (consumeOrLeavesCohort 1b).
    for (int32_t exclN = 1; exclN <= k - 2; ++exclN) {
        int32_t sel[ExecutionParameters::kMaxReducedOrLeaves];
        for (int32_t s = 0; s < exclN; ++s) sel[s] = s;
        for (;;) {
            bool exclMask[ExecutionParameters::kMaxReducedOrLeaves] = {};
            for (int32_t s = 0; s < exclN; ++s) exclMask[sel[s]] = true;

            // Survivors in parent order; u_-renumbered to the reduced or's
            // registry identity, keeping the parent tokens' first-appearance
            // order — the head's argument order (token u_<p> is parent
            // signature position p).
            std::vector<std::string> reduced;
            for (int32_t j = 0; j < k; ++j) {
                if (!exclMask[j]) reduced.push_back(leaves[j]);
            }
            std::vector<std::string> tokOrder;
            renumberULeaves(reduced, &tokOrder);
            std::vector<StrSpan> canon;
            canon.reserve(reduced.size());
            for (const std::string& r : reduced) canon.emplace_back(r);
            const std::string* redName =
                compiledOrByElements(canon.data(), static_cast<int32_t>(canon.size()));
            assert(redName != nullptr
                && "compileOrSubsetExclusions: reduced or missing (I-185 violated)");

            std::string head = "(" + *redName + "[";
            for (std::size_t s = 0; s < tokOrder.size(); ++s) {
                if (s > 0) head += ',';
                head += tokOrder[s];
            }
            head += "])";

            std::vector<std::string> premises;
            premises.reserve(static_cast<std::size_t>(exclN));
            for (int32_t s = 0; s < exclN; ++s) premises.push_back(negate(leaves[sel[s]]));

            const std::string rule = reconstructImplicationFullBind(premises, head);
            const std::string compact = prefixArgumentsWithU(compileImplicationToCompact(rule));
            // Excluded leaves + surviving head together cover every leaf, so
            // the compact spans the or's full token set.
            assert(ce::getArgs(compact).size() == arity
                && "compileOrSubsetExclusions: compact does not span the or's tokens");
            compacts.push_back(compact);

            int32_t pos = exclN - 1;
            while (pos >= 0 && sel[pos] == k - exclN + pos) --pos;
            if (pos < 0) break;
            ++sel[pos];
            for (int32_t s = pos + 1; s < exclN; ++s) sel[s] = sel[s - 1] + 1;
        }
    }
    std::vector<std::string>& list = compiledExpressions.find(orName)->second.implications;
    list.insert(list.end(), compacts.begin(), compacts.end());
}

/// @brief Complete or-implication list length for @p k leaves.
/// @details Doxygen contract at the declaration (prover.hpp).
/// @param k The or's flattened leaf count.
/// @return k for k < 3; 2^k − 2 for k ≥ 3.
int32_t ExpressionAnalyzer::expectedOrImplicationCount(int32_t k) {
    assert(k >= 2 && k <= ExecutionParameters::kMaxReducedOrLeaves
        && "expectedOrImplicationCount: leaf count outside [2, kMaxReducedOrLeaves]");
    if (k < 3) return k;
    return (int32_t{ 1 } << k) - 2;
}

/// @brief Compile an existence's two implication compacts into its
///        `implications` list.
/// @details Doxygen contract at the declaration (prover.hpp).
/// @param exName Registry name of an existence-category entity.
/// @see `preMintReducedOrs`, `disintegrateExprCore2`.
void ExpressionAnalyzer::compileExistenceImplications(const std::string& exName) {
    const auto it = compiledExpressions.find(exName);
    assert(it != compiledExpressions.end() && it->second.category == "existence"
        && "compileExistenceImplications: name must resolve to an existence entity");
    // Already carrying its list (registry reuse / reload re-registration):
    // a defined state, not a failure.
    if (!it->second.implications.empty()) return;
    // Copies: the compiles below insert implication entries into the same
    // map (node-stable, but the copy is the honest form).
    const std::vector<std::string> elements = it->second.elements;
    const std::size_t arity = ce::getArgs(it->second.signature).size();
    // A non-[left, right] shape (test-only) carries no compacts — a defined
    // state, the formula's zero.
    if (expectedExistenceImplicationCount(static_cast<int32_t>(elements.size())) == 0) {
        return;
    }

    std::vector<std::string> compacts;
    compacts.reserve(2);
    for (std::size_t i = 0; i < 2; ++i) {
        // [0]: left -> !right; [1]: right -> !left. negate, not a blind "!"
        // prefix: a negated element's negation is its bare positive core
        // (I-175).
        const std::vector<std::string> premises = { elements[i] };
        const std::string head = negate(elements[1 - i]);
        const std::string rule = reconstructImplicationFullBind(premises, head);
        // The registry stores the bound variable as `1` — the very token
        // compileImplicationToCompact's u_ strip produces from `u_1` — so
        // it is renamed to the compiler's placeholder first (the same step
        // the `!(>` compile branch takes).
        const std::tuple<std::string, int, std::string> renamed =
            renameLastRemoved(rule, this->variableCounter);
        this->variableCounter++;
        const std::string compact =
            prefixArgumentsWithU(compileImplicationToCompact(std::get<0>(renamed)));
        // Premise and head together carry every existence token (the bound
        // variable is the only non-u_ argument), so the compact spans the
        // existence's full arity.
        assert(ce::getArgs(compact).size() == arity
            && "compileExistenceImplications: compact does not span the existence's tokens");
        compacts.push_back(compact);
    }
    // Re-find: the compiles above insert implication entries into the same
    // map (node-stable, but the re-fetch is the honest form).
    compiledExpressions.find(exName)->second.implications = std::move(compacts);
}

/// @brief Complete existence-implication list length for @p elemCount
///        registry elements.
/// @details Doxygen contract at the declaration (prover.hpp).
/// @param elemCount The existence entity's element count.
/// @return 2 for the `[left, right]` shape; 0 otherwise.
int32_t ExpressionAnalyzer::expectedExistenceImplicationCount(int32_t elemCount) {
    assert(elemCount >= 0
        && "expectedExistenceImplicationCount: negative element count");
    return elemCount == 2 ? 2 : 0;
}

/// @brief Construct every OR theorem licensed by the head-switch pairs
/// against the CURRENT `globalTheoremList`, register each in the global
/// registries, and report the parents it subsumes.
///
/// @details
/// Walks `orPairsFromHeadSwitch` — one `(theorem, companion)` pair per
/// proved theorem whose straightened shape carries a binder-free negated
/// premise (`headSwitchOne`). A SINGLE proved direction licenses the
/// disjunction: classically `!d_1 → (!d_2 → h)` is equivalent to
/// `d_1 ∨ d_2 ∨ h`, and every other direction is derivable from it, so
/// the companion's own proof is never required — it is recorded only as
/// the or-theorem row's second parent reference.
///
/// A pair whose source theorem is no longer in `globalTheoremList` is
/// skipped — compression pruning and vacuity retraction remove rows
/// between the pair walk and this call, and both removals are defined
/// pipeline states, not failures. Mirror pairs `(x, y)` and `(y, x)`
/// describe the same disjunction; the canonical (min, max) disjunct-set
/// dedup constructs it once.
///
/// @param consumedParents Out-parameter accumulating the proved parent
///        theorems subsumed by a constructed OR; the caller drops them
///        from the saved theorem files.
/// @return The constructed OR theorems in compiled form, in pair order.
/// @invariant Single-threaded seam — must run after `prove()` has joined.
/// @see `constructOrTheorem` — per-pair OR builder.
/// @see `headSwitchOne` — pair producer.
std::vector<std::string> ExpressionAnalyzer::constructOrTheoremsFromPairs(
    std::set<std::string>& consumedParents) {

    std::vector<std::string> orTheorems;

    std::unordered_set<std::string> provedSet;
    for (const auto& t : this->globalTheoremList)
        provedSet.insert(std::get<0>(t));

    std::set<std::pair<std::string, std::string>> walkEmitted;

    for (const auto& pr : this->orPairsFromHeadSwitch) {
        const std::string& exist = pr.first;
        const std::string& comp = pr.second;

        // Source theorem pruned by compression or retracted as vacuous —
        // a defined pipeline state; the pair no longer licenses an OR.
        if (!provedSet.count(exist)) continue;

        // (exist, comp) and (comp, exist) describe the same OR (mirror
        // reformulations of one another). Canonicalize by lexicographically
        // sorting the pair; a pair this walk already emitted is skipped
        // outright (its first sighting consumed both parents).
        const std::pair<std::string, std::string> pairKey(
            std::min(exist, comp), std::max(exist, comp));
        if (!walkEmitted.insert(pairKey).second) {
            std::cout << "OR variant skipped (duplicate disjunct-set already constructed)"
                      << std::endl;
            continue;
        }

        // Cross-seam idempotence: an or built in-run is recovered from the
        // shared ledger — reconstructing from THIS pair's side could mint
        // the mirrored operator. A ledger miss constructs and registers as
        // before. Either way the subsumption bookkeeping and the return
        // vector fire, so the export caller still writes the or row and
        // drops its parents from the saved files.
        std::string orThm;
        const auto built = orBuiltByPair.find(pairKey);
        if (built != orBuiltByPair.end()) {
            orThm = built->second;
            std::cout << "OR theorem already constructed in-run: " << orThm << std::endl;
        } else {
            orThm = constructOrTheorem(exist, comp);
            assert(!orThm.empty()
                && "constructOrTheoremsFromPairs: pair source must carry a negated premise");
            orBuiltByPair.emplace(pairKey, orThm);
            if (appendGlobalTheorem(orThm, "or theorem", exist, comp)) {
                this->fullTheoremList.emplace_back(orThm, "or theorem", exist, comp);
                std::cout << "OR theorem constructed: " << orThm << std::endl;
            }
        }
        orTheorems.push_back(orThm);

        consumedParents.insert(exist);
        std::cout << "  parent removed (subsumed by OR): " << exist << std::endl;
        if (provedSet.count(comp)) {
            consumedParents.insert(comp);
            std::cout << "  parent removed (subsumed by OR): " << comp << std::endl;
        }
    }

    return orTheorems;
}

/// @brief In-run OR-theorem construction and broadcast (A16 Phase 2).
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full
/// contract. Implementation notes: the unscanned-row snapshot is taken
/// under `theoremListMutex` and construction runs outside it —
/// `appendGlobalTheorem` takes the same lock again, and
/// `constructOrTheorem` touches only the compile registries, which no
/// worker reads while the phase-4 barrier holds. Scan order is
/// `globalTheoremList` append order, so the `or<N>` mint sequence is a
/// deterministic function of the proof history.
///
/// @return Nothing.
/// @invariant Single-threaded phase-4 barrier seam only.
/// @see `constructOrTheoremsFromPairs` — the post-run export seam.
void ExpressionAnalyzer::constructOrTheoremsInRun() {
    // Compressor re-derivation is a redundancy probe whose theorem list
    // must stay byte-identical to the run it audits; its ors are built by
    // the export seam on the survivor list. Defined mode branch.
    if (parameters.compressor_mode) return;

    std::vector<std::string> newRows;
    {
        std::lock_guard<std::mutex> lock(theoremListMutex);
        for (const auto& tpl : globalTheoremList) {
            const std::string& thm = std::get<0>(tpl);
            // Proved-not-broadcast rows never feed the or construction —
            // the tier's contract is zero circulation; mark scanned so the
            // row is settled, then skip.
            if (std::get<1>(tpl) == "proved not broadcast") {
                orInRunScannedRows.insert(thm);
                continue;
            }
            if (orInRunScannedRows.insert(thm).second) newRows.push_back(thm);
        }
    }

    for (const std::string& thm : newRows) {
        const std::string companion = headSwitchOne(thm);
        if (companion.empty()) continue;

        // Canonical pair gate: a mirror row scanned later describes the
        // SAME disjunction with mirrored element order — constructing it
        // would double-mint or<N> (mirrored orders are distinct registry
        // identities). The shared ledger makes the first-scanned side the
        // one that builds.
        const std::pair<std::string, std::string> pairKey(
            std::min(thm, companion), std::max(thm, companion));
        if (orBuiltByPair.count(pairKey)) continue;

        const std::string orThm = constructOrTheorem(thm, companion);
        assert(!orThm.empty()
            && "constructOrTheoremsInRun: head-switched row must carry a negated premise");
        orBuiltByPair.emplace(pairKey, orThm);

        // String dedup: two DIFFERENT pairs can still fold to one or
        // theorem; the second registers nothing and broadcasts nothing.
        if (!appendGlobalTheorem(orThm, "or theorem", thm, companion)) continue;
        this->fullTheoremList.emplace_back(orThm, "or theorem", thm, companion);
        std::cout << "OR theorem constructed (in-run): " << orThm << std::endl;

        // Broadcast like any proved implication: the deferred-compaction
        // drain that follows this seam compiles the or theorem to its
        // implication compact and merges it into the root's mailOut (empty
        // level set; status-3 rule door at every receiver). A premise-free
        // or theorem has no implication-compact shape — registry+list only.
        if (startsWith(orThm, "(>[", 3)) {
            recordPendingCompaction(orThm, /*kySize=*/0, /*coreId=*/-1);
        }
    }

    // Close the reduced-or single-elimination closure over everything now
    // registered (including the ors this seam just minted) so the
    // single-exclusion emission's registry lookup can hard-assert on a
    // miss (D-268). Idempotent.
    preMintReducedOrs();
}

/// @brief Token-aligned variable unification (or-elimination comparator).
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full
/// contract. Implementation notes: tokens are maximal `[A-Za-z0-9_]+`
/// runs; every byte between tokens must match exactly, which keeps
/// polarity literal (I-175 — a `!` on one side only fails the walk).
/// A purely numeric token with value >= 9 is a bound non-anchor
/// variable on BOTH sides or the walk fails; it binds through the
/// caller-shared bijection. All other tokens compare byte-equal.
///
/// @return True iff the walk completes with a consistent bijection.
bool ExpressionAnalyzer::alignOrEliminationExprs(const std::string& a,
                                                 const std::string& b,
                                                 std::map<std::string, std::string>& forward,
                                                 std::map<std::string, std::string>& reverse) {
    const auto isTokenChar = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9') || c == '_';
    };
    const auto isBindableVar = [](const std::string& tok) {
        for (char c : tok) {
            if (c < '0' || c > '9') return false;
        }
        return !tok.empty() && (tok.size() > 1 || tok[0] > '8');
    };

    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.size() && j < b.size()) {
        if (!isTokenChar(a[i]) || !isTokenChar(b[j])) {
            if (a[i] != b[j]) return false;
            ++i;
            ++j;
            continue;
        }
        std::size_t ie = i;
        std::size_t je = j;
        while (ie < a.size() && isTokenChar(a[ie])) ++ie;
        while (je < b.size() && isTokenChar(b[je])) ++je;
        const std::string ta = a.substr(i, ie - i);
        const std::string tb = b.substr(j, je - j);

        const bool bindA = isBindableVar(ta);
        const bool bindB = isBindableVar(tb);
        if (bindA != bindB) return false;
        if (bindA) {
            const auto fIt = forward.find(ta);
            if (fIt != forward.end()) {
                if (fIt->second != tb) return false;
            } else {
                if (reverse.count(tb)) return false;
                forward.emplace(ta, tb);
                reverse.emplace(tb, ta);
            }
        } else if (ta != tb) {
            return false;
        }
        i = ie;
        j = je;
    }
    return i == a.size() && j == b.size();
}

/// @brief License probe for the pre-split merge.
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full
/// contract. Implementation notes: the row snapshot is taken under
/// `theoremListMutex` and the scan runs outside it (the seam is
/// single-threaded; the lock only orders against the append door's own
/// locking discipline). Scan order is `globalTheoremList` append order,
/// so the citation choice is a deterministic function of the proof
/// history. The side-premise cover additionally requires every bound
/// non-anchor variable of an or-theorem premise to be in the leaf
/// bijection's domain — an or theorem whose premise mentions a variable
/// its disjuncts do not bind is not fully instantiated by the guards
/// and cannot license the merge.
///
/// @return The licensing or-theorem row, or "" (defined no-license).
std::string ExpressionAnalyzer::findOrEliminationLicense(const std::string& guardA,
                                                         const std::string& guardB,
                                                         const std::vector<std::string>& commonChain) {
    std::vector<std::string> rows;
    {
        std::lock_guard<std::mutex> lock(theoremListMutex);
        rows.reserve(globalTheoremList.size());
        for (const auto& tpl : globalTheoremList) rows.push_back(std::get<0>(tpl));
    }

    const std::set<std::string> commonSet(commonChain.begin(), commonChain.end());

    const auto isTokenChar = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9') || c == '_';
    };
    const auto occursAsToken = [&isTokenChar](const std::string& hay, const std::string& tok) {
        std::size_t pos = 0;
        while ((pos = hay.find(tok, pos)) != std::string::npos) {
            const std::size_t end = pos + tok.size();
            const bool leftOk = (pos == 0) || !isTokenChar(hay[pos - 1]);
            const bool rightOk = (end >= hay.size()) || !isTokenChar(hay[end]);
            if (leftOk && rightOk) return true;
            ++pos;
        }
        return false;
    };
    const auto eachVarMapped = [&isTokenChar](const std::string& expr,
                                              const std::map<std::string, std::string>& forward) {
        std::size_t i = 0;
        while (i < expr.size()) {
            if (!isTokenChar(expr[i])) { ++i; continue; }
            std::size_t ie = i;
            while (ie < expr.size() && isTokenChar(expr[ie])) ++ie;
            const std::string tok = expr.substr(i, ie - i);
            bool numeric = true;
            for (char c : tok) {
                if (c < '0' || c > '9') { numeric = false; break; }
            }
            if (numeric && (tok.size() > 1 || tok[0] > '8') && !forward.count(tok)) {
                return false;
            }
            i = ie;
        }
        return true;
    };

    for (const std::string& row : rows) {
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain;
        const std::string head = ce::disintegrateImplication(row, chain, coreExpressionMap);

        const LogicalEntity* le = compiledEntity(extractExpressionUniversalSpan(StrSpan(head)));
        if (le == nullptr || le->category != "or") continue;

        std::vector<std::string> uLeaves;
        flattenRegistryOrLeaves(*le, uLeaves);
        if (uLeaves.size() != 2) continue;   // k=2 in the first build

        // Instantiate the registry-canonical leaves with the head's args.
        const std::vector<std::string> sigArgs = ce::getArgs(le->signature);
        const std::vector<std::string> headArgs = ce::getArgs(head);
        if (sigArgs.size() != headArgs.size()) continue;
        std::map<std::string, std::string> subst;
        for (std::size_t k = 0; k < sigArgs.size(); ++k) subst[sigArgs[k]] = headArgs[k];
        const std::string leaf0 = ce::replaceKeysInString(uLeaves[0], subst);
        const std::string leaf1 = ce::replaceKeysInString(uLeaves[1], subst);

        for (int ordering = 0; ordering < 2; ++ordering) {
            const std::string& g0 = (ordering == 0) ? guardA : guardB;
            const std::string& g1 = (ordering == 0) ? guardB : guardA;
            std::map<std::string, std::string> forward;
            std::map<std::string, std::string> reverse;
            if (!alignOrEliminationExprs(leaf0, g0, forward, reverse)) continue;
            if (!alignOrEliminationExprs(leaf1, g1, forward, reverse)) continue;

            bool covered = true;
            for (const auto& link : chain) {
                const std::string& prem = std::get<0>(link);
                if (!eachVarMapped(prem, forward)) { covered = false; break; }
                const std::string img = ce::replaceKeysInString(prem, forward);
                if (commonSet.count(img)) continue;
                if (startsWith(img, "(in[", 4)) {
                    const std::vector<std::string> inArgs = ce::getArgs(img);
                    if (!inArgs.empty()) {
                        bool occurs = false;
                        for (const std::string& c : commonChain) {
                            if (occursAsToken(c, inArgs[0])) { occurs = true; break; }
                        }
                        if (occurs) continue;
                    }
                }
                covered = false;
                break;
            }
            if (covered) return row;
        }
    }
    return std::string();
}

/// @brief In-run pre-split merge (or elimination).
///
/// @details
/// See the declaration's Doxygen block in `prover.hpp` for the full
/// contract. Implementation notes: the unscanned-row snapshot is taken
/// under `theoremListMutex` and everything else runs outside it (the
/// append door re-takes the lock). Rows appended by this drain are not
/// scanned in the same drain (the or seam's snapshot idiom), so a
/// merged theorem can become a variant of a later merge one iteration
/// later — deterministic chaining. Pairs whose licensing or theorem
/// has not been minted yet park on `orElimPendingPairs` and are
/// re-probed every iteration in insertion order.
///
/// @return Nothing.
/// @invariant Single-threaded phase-4 barrier seam only; no registry
///            mints, no LB interaction.
void ExpressionAnalyzer::constructOrEliminationInRun() {
    // Same defined no-op as the or seam: the compressor's re-derivation
    // must stay byte-identical to the run it audits.
    if (parameters.compressor_mode) return;

    std::vector<std::string> newRows;
    {
        std::lock_guard<std::mutex> lock(theoremListMutex);
        for (const auto& tpl : globalTheoremList) {
            const std::string& thm = std::get<0>(tpl);
            if (orElimInRunScannedRows.insert(thm).second) newRows.push_back(thm);
        }
    }

    // File new rows into the guard index; a second guard under one common
    // key forms a candidate pair, parked for the license probe below.
    for (const std::string& thm : newRows) {
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain;
        const std::string head = ce::disintegrateImplication(thm, chain, coreExpressionMap);
        if (chain.size() < 2) continue;                    // a guard needs a shared premise before it
        if (!std::get<1>(chain.back()).empty()) continue;  // pre-split guards are binder-free

        const std::string& guard = std::get<0>(chain.back());
        std::string key;
        for (std::size_t k = 0; k + 1 < chain.size(); ++k) {
            key += std::get<0>(chain[k]);
            key += '\x01';
        }
        key += '\x02';
        key += head;

        std::vector<std::pair<std::string, std::string>>& entries = orElimGuardIndex[key];
        for (const std::pair<std::string, std::string>& other : entries) {
            if (other.second == guard) continue;
            orElimPendingPairs.push_back({other.first, other.second, thm, guard});
        }
        entries.emplace_back(thm, guard);
    }

    // License probe over the parked pairs, insertion order. A pair leaves
    // only by merging (or the append door's string dedup); a licenseless
    // pair parks for a later iteration's or mints.
    std::vector<std::array<std::string, 4>> stillPending;
    stillPending.reserve(orElimPendingPairs.size());
    for (const std::array<std::string, 4>& pending : orElimPendingPairs) {
        const std::string& variantA = pending[0];
        const std::string& guardA = pending[1];
        const std::string& variantB = pending[2];
        const std::string& guardB = pending[3];

        const std::pair<std::string, std::string> pairKey(
            std::min(variantA, variantB), std::max(variantA, variantB));
        if (orElimBuiltByPair.count(pairKey)) continue;

        // Rebuild the shared context from variant A — byte-identical on
        // variant B by the guard-index key construction.
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain;
        const std::string head = ce::disintegrateImplication(variantA, chain, coreExpressionMap);
        assert(chain.size() >= 2
            && "constructOrEliminationInRun: parked variant lost its guard chain");
        std::vector<std::string> commonChain;
        commonChain.reserve(chain.size() - 1);
        for (std::size_t k = 0; k + 1 < chain.size(); ++k) {
            commonChain.push_back(std::get<0>(chain[k]));
        }

        const std::string license = findOrEliminationLicense(guardA, guardB, commonChain);
        if (license.empty()) {
            stillPending.push_back(pending);
            continue;
        }

        const std::string merged = reconstructImplicationFullBind(commonChain, head);
        orElimBuiltByPair.emplace(pairKey, merged);

        // String dedup door: two different pre-split pairs can fold to one
        // merged theorem; the second registers and broadcasts nothing.
        if (!appendGlobalTheorem(merged, "or elimination", variantA, variantB)) continue;
        this->fullTheoremList.emplace_back(merged, "or elimination", variantA, variantB);
        orElimCitedOrByMerged.emplace(merged, license);
        std::cout << "OR elimination merged (in-run): " << merged << std::endl;

        // Broadcast like any proved implication via the deferred-compaction
        // drain that follows this seam (empty level set at the drain; a
        // premise-free result registers without broadcast).
        if (startsWith(merged, "(>[", 3)) {
            recordPendingCompaction(merged, /*kySize=*/0, /*coreId=*/-1);
        }
    }
    orElimPendingPairs.swap(stillPending);
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
                    if (le.category == "or" && le.elements.size() >= 2) {
                        // Elements carry each disjunct's true polarity;
                        // negate() cancels a double negation, so a negated
                        // disjunct contributes its bare positive core. The
                        // inner AND is flat n-ary — the parser accepts any
                        // conjunct count (anchor definition bodies parse
                        // the same shape).
                        auto negate = [](const std::string& s) -> std::string {
                            if (!s.empty() && s[0] == '!') return s.substr(1);
                            return "!" + s;
                        };
                        expanded = "!(&";
                        for (const auto& elem : le.elements) {
                            expanded += negate(ce::replaceKeysInString(elem, subst));
                        }
                        expanded += ")";
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
                        // Negate the head element WITH double-negation
                        // cancellation — a negated element contributes its
                        // bare positive core; a blind '!' prefix would emit
                        // a malformed !!(...) row that crashes any later
                        // load of the base form.
                        const std::string negRight =
                            (!right.empty() && right[0] == '!')
                                ? right.substr(1) : "!" + right;
                        expanded = "!(>[" + freshVar + "]" + left + negRight + ")";
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
