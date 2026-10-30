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

// Counterexample filtering uses the coordinator phase adapters and the
// CounterexampleFilter operation interface.

#include "project_paths.hpp"
#include "filter.hpp"
#include "prover.hpp"
#include "parameters.hpp"
#include "infra/rt_tracker.hpp"
#include "memory_infra/arena_stack.hpp"
#include "memory_infra/scratch_arena.hpp"
#include "memory_infra/str_ops.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace gl {











    // ------------------------------------------------------------------
    // CE-filter top-level orchestrator (moved from prover.cpp in commit E).
    // ------------------------------------------------------------------

    std::vector<std::string> Prover::filterConjecturesWithCE(
        const std::vector<std::string>& conjectures,
        const std::vector<std::string>& simpleFacts)
    {
        ceFilteringActive = true;

        // One contradictionTable slot per conjecture. Each worker writes only its
        // own (disjoint) index, so the post-pool survivor reads are race-free.
        contradictionTable.clear();
        contradictionTable.reserve(conjectures.size());
        for (int i = 0; i < static_cast<int>(conjectures.size()); ++i)
            contradictionTable.push_back(ContradictionItem(conjectures[i], false));

        // Load the fact base ONCE into a single template LB (status-4 statements;
        // empty hashmap — see Memory::cloneFactsTemplate). Each conjecture runs on
        // its own throwaway clone of this template.
        counterexampleFilter.loadFactsForCEFiltering(simpleFacts, /*batchSize=*/1);
        const Memory* templateLB = ceSimpleMapStore.findChild(&ceBody, std::to_string(0));

        // Pre-intern every CE LB's exprKey ("0".."N-1") into the process-wide
        // skeletonInterner() SINGLE-THREADED, before the worker pool spawns. The
        // skeletonInterner is "single-threaded write side only" (I-83): the
        // workers below each call lb->setExprKey(std::to_string(i)), and a
        // concurrent mint() from the parallel pool races on its cold key store +
        // hash index, tripping the "minted key not findable at its id" desync
        // assert (I-82). After this loop every key already exists, so the
        // workers' setExprKey calls take the lookup-only (pure read) path, which
        // is safe to run concurrently on the now-unchanging table.
        for (std::size_t i = 0; i < conjectures.size(); ++i)
            (void)skeletonInterner().intern(std::to_string(i));

        const unsigned workers = std::max(1u, logicalCores);
        // The CE filter runs exactly one hashburst per conjecture.
        assert(parameters.numberIterationsConjectureFiltering == 1
            && "CE filter does one hashburst per conjecture");

        const auto ceStart = std::chrono::high_resolution_clock::now();
        const double phase2Before = phase2CumulativeSeconds;
        if (phase2Backend == Phase2Backend::cuda) {
            // Independent clones share setup and use all construction workers.
            // The automatic planner packs their immutable images into bounded
            // device chunks and continues when processing storage fills.
            constexpr std::size_t kCeCudaBatchSize = 64;
            for (std::size_t begin = 0; begin < conjectures.size();
                 begin += kCeCudaBatchSize) {
                const std::size_t count = std::min(
                    kCeCudaBatchSize, conjectures.size() - begin);
                std::vector<Memory*> batch(count, nullptr);
                std::atomic<std::size_t> nextClone{ 0 };
                auto cloneWorker = [this, &conjectures, templateLB, begin, count,
                                    &batch, &nextClone](unsigned coreId) {
                    // addConjectureForCEFiltering reaches per-slot scratch before
                    // phase 1 publishes a slot, so clone construction owns the
                    // worker's scratch slot explicitly.
                    gl::WorkerContext::setCoreId(static_cast<int>(coreId));
                    for (;;) {
                        const std::size_t local = nextClone.fetch_add(
                            1, std::memory_order_relaxed);
                        if (local >= count) break;
                        const std::size_t conjectureIndex = begin + local;
                        Memory* lb = templateLB->cloneFactsTemplate(lbStore);
                        lb->parentMemory = &ceBody;
                        lb->setExprKey(std::to_string(conjectureIndex));
                        counterexampleFilter.addConjectureForCEFiltering(
                            conjectures[conjectureIndex], lb,
                            static_cast<int>(conjectureIndex));
                        batch[local] = lb;
                    }
                };
                std::vector<std::thread> pool;
                const unsigned cloneWorkers = static_cast<unsigned>(
                    std::min<std::size_t>(workers, count));
                pool.reserve(cloneWorkers);
                for (unsigned t = 0; t < cloneWorkers; ++t)
                    pool.emplace_back(cloneWorker, t);
                for (std::thread& thread : pool) thread.join();
                for (Memory* lb : batch)
                    assert(lb != nullptr && "CUDA CE batch clone was not built");

                // prove() supplies the resident-only steward and sends all CE
                // Phase 2 tasks through the selected CUDA backend in one sweep.
                prove(/*numberIterations=*/1, batch);
                for (Memory* lb : batch) lbStore.destroy(lb);
            }
        }
        else {
            // CPU ASIC path: independent single-owner clones run through the
            // established direct one-conjecture Phase 1/2/3 path.
            std::atomic<std::size_t> next{ 0 };
            auto worker = [this, &conjectures, templateLB, &next](unsigned coreId) {
                gl::WorkerContext::setCoreId(static_cast<int>(coreId));
                for (;;) {
                    const std::size_t i = next.fetch_add(
                        1, std::memory_order_relaxed);
                    if (i >= conjectures.size()) break;

                    Memory* lb = templateLB->cloneFactsTemplate(lbStore);
                    lb->parentMemory = &ceBody;
                    lb->setExprKey(std::to_string(i));
                    counterexampleFilter.addConjectureForCEFiltering(
                        conjectures[i], lb, static_cast<int>(i));

                    this->performElemPhase1(*lb, coreId);
                    SealedPageSet sealedPages;
                    sealedPages.bind(&staticMemory());
                    std::atomic<int64_t> doomLine{ kNoDoomLine };
                    phase2Executor.performElem2(
                        *lb, coreId, /*partCount=*/1, SplitStumpRef{}, sealedPages,
                        doomLine);
                    sealedPages.seal();
                    SealedPageSet* onePart[1] = { &sealedPages };
                    phase2Executor.performElemPhase2(*lb, onePart, 1,
                        doomLine.load(std::memory_order_relaxed));
                    this->performElemPhase3(*lb, coreId);
                    sealedPages.freePages();

                    lbStore.destroy(lb);
                }
            };
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (unsigned t = 0; t < workers; ++t)
                pool.emplace_back(worker, t);
            for (std::thread& thread : pool) thread.join();
        }
        const auto ceElapsed =
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - ceStart).count();
        std::cout << "CE filter: " << conjectures.size() << " conjectures, "
                  << workers << " workers, backend="
                  << (phase2Backend == Phase2Backend::cuda ? "cuda" : "cpu")
                  << ", phase2=" << (phase2CumulativeSeconds - phase2Before)
                  << "s, overall=" << ceElapsed << "s" << std::endl;
        std::cout.flush();

        // Extra explicit cleanup “as if between batches” (no-op if already clean)
        counterexampleFilter.releaseCEBatchMemory();

        // Mirror refutation: a CE-refuted operator-only conjecture dooms its
        // pool mirror too (D-229). Runs
        // single-threaded between the pool join and survivor collection so
        // the flip is a pure function of this pass's CE verdicts.
        if (parameters.mirror_refutation) {
            const int flipped = applyMirrorRefutations(
                contradictionTable, conjectures, mirrorPartnerMap);
            std::cout << "CE filter: mirror refutation flipped " << flipped
                      << " conjectures." << std::endl;
        }

        // Keep non-contradictory conjectures
        std::vector<std::string> filtered;
        filtered.reserve(conjectures.size());
        for (int i = 0; i < static_cast<int>(conjectures.size()); ++i)
            if (!contradictionTable[i].successful) filtered.push_back(conjectures[i]);

        ceFilteringActive = false;
        return filtered;
    }

    /// @brief Load `mirror_pairs.txt` into a symmetric partner map.
    ///
    /// @details
    /// Reads the conjecturer-written pairs artifact (one
    /// `source<TAB>mirror` row per operator-only conjecture whose mirror
    /// entered the pool; both columns byte-identical to `conjectures.txt`
    /// lines) and inserts BOTH directions — `source → mirror` and
    /// `mirror → source` — with per-key duplicate suppression, so the CE
    /// filter's flip pass can follow the relation from whichever member a
    /// counterexample refutes. Rows are read in file order (source-sorted by
    /// the writer), keeping each key's partner vector deterministic.
    /// Trailing `\r` is stripped; empty lines are skipped (an empty file is
    /// a defined result: no pairs, no flips).
    ///
    /// The file's existence is part of the pipeline contract: the
    /// conjecturer writes it unconditionally whenever `conjectures.txt` is
    /// written, so absence means the pipeline was invoked out of order —
    /// asserted, not tolerated.
    ///
    /// @param path Full path to `files/theorems/mirror_pairs.txt`.
    /// @return Symmetric partner map; empty when the file has no rows.
    /// @see [D-229](../../docs/agentic_swdd/40_decisions.md#d-229)
    MirrorPartnerMap loadMirrorPairs(const std::filesystem::path& path) {
        assert(std::filesystem::exists(path)
            && "loadMirrorPairs: mirror_pairs.txt missing - the conjecturer "
               "writes it whenever conjectures.txt is written; run "
               "--conjecture <Tag> first");

        MirrorPartnerMap partners;
        auto addDirected = [&partners](const std::string& from, const std::string& to) {
            auto& vec = partners[from];
            if (std::find(vec.begin(), vec.end(), to) == vec.end()) {
                vec.push_back(to);
            }
        };

        std::ifstream in(path);
        assert(in && "loadMirrorPairs: mirror_pairs.txt exists but cannot be opened");
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            const std::size_t tabPos = line.find('\t');
            assert(tabPos != std::string::npos
                && "loadMirrorPairs: malformed row - every non-empty line is "
                   "source<TAB>mirror by the writer contract");
            std::string source = line.substr(0, tabPos);
            std::string mirror = line.substr(tabPos + 1);
            assert(!source.empty() && !mirror.empty()
                && "loadMirrorPairs: empty column - violates the writer contract");
            addDirected(source, mirror);
            addDirected(mirror, source);
        }
        return partners;
    }

    /// @brief Flip the mirror partners of CE-refuted conjectures to refuted.
    ///
    /// @details
    /// The mirror-refutation heuristic's flip pass. Snapshots which slots the
    /// CE bursts refuted (`table[i].successful` seeds), then sweeps them in
    /// ascending index order: for every seeded conjecture, each partner from
    /// `partners` that is present in this batch has its slot's `successful`
    /// set `true`. Only CE-confirmed refutations seed — a flipped slot never
    /// seeds further flips (no cascading) — so the outcome is a pure function
    /// of the CE verdicts plus the pairs file, independent of sweep order. A
    /// partner absent from `conjectures` is a defined case (refuted or
    /// flipped in an earlier fact-file pass, or deduplicated at generation),
    /// not a failure.
    ///
    /// Refutation-side only: this marks conjectures as dropped pre-prover; no
    /// mirror ever re-enters as a proof step
    /// ([I-81](../../docs/agentic_swdd/30_invariants.md#i-81) / D-112).
    ///
    /// @param table       The CE contradiction table, one slot per
    ///                    conjecture; flipped in place.
    /// @param conjectures The batch's conjecture strings, index-aligned with
    ///                    `table`.
    /// @param partners    Symmetric partner map from `loadMirrorPairs`.
    /// @return Number of slots flipped by this pass.
    /// @see loadMirrorPairs — builds `partners`.
    int applyMirrorRefutations(std::vector<ContradictionItem>& table,
        const std::vector<std::string>& conjectures,
        const MirrorPartnerMap& partners) {
        assert(table.size() == conjectures.size()
            && "applyMirrorRefutations: table and conjecture list must be "
               "index-aligned");
        if (partners.empty()) return 0;

        std::unordered_map<std::string, int> slotOf;
        slotOf.reserve(conjectures.size());
        for (int i = 0; i < static_cast<int>(conjectures.size()); ++i) {
            slotOf.emplace(conjectures[i], i);
        }

        std::vector<char> seed(table.size(), 0);
        for (std::size_t i = 0; i < table.size(); ++i) {
            seed[i] = table[i].successful ? 1 : 0;
        }

        int flipped = 0;
        for (int i = 0; i < static_cast<int>(table.size()); ++i) {
            if (!seed[i]) continue;
            const auto it = partners.find(conjectures[i]);
            if (it == partners.end()) continue;
            for (const std::string& partner : it->second) {
                const auto slotIt = slotOf.find(partner);
                if (slotIt == slotOf.end()) continue;
                ContradictionItem& row = table[slotIt->second];
                if (!row.successful) {
                    row.successful = true;
                    ++flipped;
                }
            }
        }
        return flipped;
    }

} // namespace gl
