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
#include "prover_initialization.hpp"
#include "../project_paths.hpp"
#include "../infra/peak_heap.hpp"
#include "../infra/phase2_trace.hpp"
#include "../infra/diagnostics_log.hpp"

#include "../parameters.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../infra/rt_tracker.hpp"
#include "../infra/memory_census.hpp"
#include "../memory_infra/global_memory_manager.hpp"
#include "../memory_infra/scratch_arena.hpp"
#include "../memory_infra/lb_deload.hpp"
#include "../memory_infra/deload_stats.hpp"
#include "../gpu/phase2_projection.hpp"
#ifdef GL_CUDA
#include "../gpu/phase2_cuda.hpp"
#include "../gpu/phase2_sealing.hpp"
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
#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif


namespace gl {
/// @brief Bind the inputs used to initialize one prover batch.
/// @details Stores explicit references to the owners; construction does not run initialization.
/// @param parameters Existing batch state or operation used during initialization.
/// @param phase2Backend Existing batch state or operation used during initialization.
/// @param body Existing batch state or operation used during initialization.
/// @param allMappingsAna Existing batch state or operation used during initialization.
/// @param allPermutationsAna Existing batch state or operation used during initialization.
/// @param globalDependencies Existing batch state or operation used during initialization.
/// @param definitions Existing batch state or operation used during initialization.
/// @param anchorInfo Existing batch state or operation used during initialization.
/// @param anchorID_ Existing batch state or operation used during initialization.
/// @param permanentBodies Existing batch state or operation used during initialization.
/// @param logicalCores Existing batch state or operation used during initialization.
/// @param structuralCompiler Existing batch state or operation used during initialization.
/// @param supportingRuleCompiler Existing batch state or operation used during initialization.
/// @param compilationPersistence Existing batch state or operation used during initialization.
/// @param compilationReuse Reuse component whose state a precompiled configuration replaces.
/// @return Constructed initialization interface.
/// @brief Release the initialization interface.
/// @details On the CUDA route the owned `gpu::CudaContextWarmup` joins its
/// thread here; this is the translation unit where that type is complete.
/// @return No value.
ProverInitialization::~ProverInitialization() = default;
ProverInitialization::ProverInitialization(ProverParameters& parameters, Phase2Backend& phase2Backend, Memory& body, std::map<int64_t, std::vector<std::vector<int> > >& allMappingsAna, std::map<int, std::vector<std::vector<int> > >& allPermutationsAna, Dependencies& globalDependencies, CompiledDefinitionRegistry& definitions, ce::AnchorInfo& anchorInfo, std::string& anchorID_, std::vector<Memory*>& permanentBodies, unsigned& logicalCores, StructuralCompiler& structuralCompiler, SupportingRuleCompiler& supportingRuleCompiler, CompilationPersistence& compilationPersistence, CompilationReuse& compilationReuse) : parameters(parameters), phase2Backend(phase2Backend), body(body), allMappingsAna(allMappingsAna), allPermutationsAna(allPermutationsAna), globalDependencies(globalDependencies), definitions(definitions), anchorInfo(anchorInfo), anchorID_(anchorID_), permanentBodies(permanentBodies), logicalCores(logicalCores), structuralCompiler(structuralCompiler), supportingRuleCompiler(supportingRuleCompiler), compilationPersistence(compilationPersistence), compilationReuse(compilationReuse) {}
/// @brief Initialize the configuration, compiler and memory services for one batch.
/// @details Executes the existing configuration loads, backend selection, pool
/// setup, mapping construction, registry load and rule preparation in their
/// order. The caller owns every referenced store.
///
/// With @p precompiled, the definition compile (`compileCoreExpressionMap`,
/// `compileContrapositionFamilies`) is replaced by installing the recorded
/// registry and reuse state; every other step runs unchanged. The record must
/// come from a prover of the same configuration constructed without a
/// compiled-registry input, so the installed state is exactly what this
/// prover's own compile would produce.
/// @param anchorID Requested anchor configuration.
/// @param selectedPhase2Backend Optional caller-selected phase-two backend.
/// @param glBinaryFile Optional compiled-registry input.
/// @param precompiled Optional recorded compile of the same configuration;
///        exclusive with @p glBinaryFile (asserted).
/// @return No value.
/// @invariant Runs at the prover construction boundary, before any proof work.
void ProverInitialization::initialize(const std::string& anchorID, const std::optional<Phase2Backend>& selectedPhase2Backend, const std::optional<std::filesystem::path>& glBinaryFile, const PrecompiledDefinitions* precompiled) {
    assert(!(precompiled != nullptr && glBinaryFile.has_value())
        && "a precompiled configuration and a compiled-registry input are exclusive");
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
    fs::path configPath = gl::projectRoot()
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
                if (pp.contains("maxIntegrationDepth")) {
                    assert(pp["maxIntegrationDepth"].is_number_integer());
                    const int64_t depth = pp["maxIntegrationDepth"].get<int64_t>();
                    assert(depth >= 0 && depth < INT_MAX && "invalid maxIntegrationDepth");
                    parameters.maxIntegrationDepth = static_cast<int>(depth);
                }
                if (pp.contains("standardMaxSecondaryNumber")) parameters.standardMaxSecondaryNumber = pp["standardMaxSecondaryNumber"];
                if (pp.contains("maxNumberFilteredStatements")) {
                    assert(pp["maxNumberFilteredStatements"].is_number_integer()
                        && "maxNumberFilteredStatements must be an integer");
                    const int64_t filteredLimit = pp["maxNumberFilteredStatements"].get<int64_t>();
                    assert(filteredLimit > 0
                        && filteredLimit <= ExecutionParameters::MAX_FILTERED_POSITION_CODE
                        && "maxNumberFilteredStatements exceeds the filtered-position representation");
                    parameters.maxNumberFilteredStatements = static_cast<int>(filteredLimit);
                }
                if (pp.contains("trackHistory")) parameters.trackHistory = pp["trackHistory"];
                if (pp.contains("weak_induction")) parameters.weak_induction = pp["weak_induction"];
                if (pp.contains("strong_induction")) parameters.strong_induction = pp["strong_induction"];
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

    assert(parameters.maxNumberFilteredStatements > 0
        && parameters.maxNumberFilteredStatements <= ExecutionParameters::MAX_FILTERED_POSITION_CODE);
    phase2Backend = selectedPhase2Backend.value_or(
        parameters.use_gpu ? Phase2Backend::cuda : Phase2Backend::cpu);
#ifndef GL_CUDA
    // A build without CUDA support (a Makefile build with USE_CUDA=0) has no
    // CUDA route at all: selecting one is refused at startup, never worked
    // around by the processor route (I-211).
    assert(phase2Backend == Phase2Backend::cpu
        && "the CUDA Phase 2 route was selected, but this build has no CUDA "
           "support (build the Visual Studio project, or make USE_CUDA=1)");
#else
    // Create the CUDA context on a background thread while the rest of the
    // batch sets up (D-554); the processor route never
    // touches CUDA.
    if (phase2Backend == Phase2Backend::cuda) {
        assert(!cudaContextWarmup && "the CUDA context warm-up starts once per batch");
        cudaContextWarmup = std::make_unique<gpu::CudaContextWarmup>();
    }
#endif
    // Both Phase 2 backends keep the configured SSD permission (default true)
    // and page through the working-set steward: the CUDA route claims every
    // logic block it reads inside a steward window and keeps a window open
    // while the device output grows the pool (D-409).
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
    // Memory census: the fixed per-object sizes behind the pool figures (a
    // Memory shell occupies one LbStore slot; slots never straddle a block).
    {
        const std::size_t slotStride =
            (sizeof(Memory) + alignof(Memory) - 1) / alignof(Memory) * alignof(Memory);
        diagnosticsLog() << "[lb-geometry] memory_object_bytes=" << sizeof(Memory)
                         << " lb_memory_bytes=" << sizeof(LbMemory)
                         << " lb_arena_bytes=" << sizeof(LbArena)
                         << " int_encoded_expr_bytes=" << sizeof(IntEncodedExpr)
                         << " lb_store_slot_stride_bytes=" << slotStride
                         << " lb_store_slots_per_block="
                         << static_cast<std::size_t>(parameters.static_lb_block_bytes) / slotStride
                         << std::endl;
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
    // Prover-owned mailArena draws its blocks from this pool.
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


    // Cross-batch persistent naming: load the per-batch GL_binary_<Tag>.json
    // the caller hands in (the batch driver passes the file Python
    // pre-populated by copying GL_binary_shared.json before invoking this
    // executable). The loader populates compiledExpressions, restores
    // generated and public logical-predicate reuse entries, and bumps the
    // four shared counters so newly-allocated names start above the highest
    // already in use. variableCounter remains zero. No-op on a clean run
    // when the file does not yet exist. Without a file — every unit test —
    // nothing is read: the registry is compiled from the config alone, so
    // no run output reaches a test.
    if (glBinaryFile) {
        compilationPersistence.loadGlBinary(*glBinaryFile);
    }

    Dependencies dependencyTable;
    globalDependencies = dependencyTable;

    std::cout << "[INIT] compiling core expression map..." << std::endl;
    definitions.configure(ce::modifyCoreExpressionMap(anchorID));
    std::cout << "[INIT] core expression map done, " << this->definitions.configurations().size() << " entries." << std::endl;

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
    anchorInfo = initAnchor(this->definitions.configurations(), effectiveAnchorID);
    std::cout << "[INIT] initAnchor done." << std::endl;

    permanentBodies.clear();
    permanentBodies.push_back(&body);

    logicalCores = std::max(1u, std::thread::hardware_concurrency());
    //logicalCores = 1;
    assert(logicalCores >= 1);

    // Scratch string arenas: one per worker slot (coreId-indexed), PLUS one
    // reserved slot (index logicalCores) for single-threaded contexts that
    // reach the absorb door's prefixArgumentsWithU off the worker pool (CE-fact
    // load, grid build, compressor, drains; gl::WorkerContext::coreId() == -1) — hence
    // logicalCores + 1. Drawn from the pool on the cold grant path, each
    // released back at its task's exit (no retention). No reserve cap — the
    // pool's own exhaustion assert is the backstop.
    initScratchArenas(logicalCores + 1);

    // Request-generation scratch arenas: a second per-slot set, holding the
    // generators' DFS frontier (byte-bump popTo) and per-batch containers (page
    // tier), isolated from the string-scratch arena above. Released per task.
    // Same shape as the string registry — logicalCores worker slots PLUS the
    // reserved single-threaded slot (index logicalCores) that every
    // gl::WorkerContext::coreId() == -1 caller resolves to: the seam drains, the CE-fact
    // load, the compressor, and any steward / io thread that reaches
    // gen-scratch (a canonical reload's reverse-index rebuild). Without it the
    // "reserved" fallback was worker logicalCores-1's own arena and its
    // rule-index staging pool — a shared arena between a worker and whichever
    // non-worker thread hit the fallback while that worker ran.
    initGenScratchArenas(logicalCores + 1);

    // Compaction scratch: one permutation scratch per worker slot on the
    // never-deloaded LB-body pool, for the worker's end-of-phase-1 compaction
    // of its own claimed logic block (compactIfFragmented).
    initCompactionScratchArenas(logicalCores + 1);

    // Batch byte sharing: the process-wide dictionary on the LB-body pool and
    // one nomination table per worker slot (same slot shape as above).
    initBatchByteSharing();

    if (precompiled != nullptr) {
        // The recorded compile of this configuration: the registry and the
        // reuse state are the complete compile output (the compilers hold
        // only references to these two stores).
        definitions = precompiled->registry;
        compilationReuse.restoreState(precompiled->reuse);
    } else {
        structuralCompiler.compileCoreExpressionMap();
        supportingRuleCompiler.compileContrapositionFamilies();
        supportingRuleCompiler.compileDisintegrationPackages();
    }
}
/// @brief Enumerate the finite argument mappings used by the request pipeline.
/// @details Preserves packed-key order and mixed-radix enumeration of every function for the configured arity ranges.
/// @param n Exclusive domain-size limit.
/// @param m Exclusive target-size limit.
/// @return Existing mapping table indexed by packed domain and target sizes.
std::map<int64_t, std::vector<std::vector<int>>> ProverInitialization::generateAllMappings(int n, int m) {
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
}
