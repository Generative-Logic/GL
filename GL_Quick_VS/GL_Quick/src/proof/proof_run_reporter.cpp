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
#include "proof_run_reporter.hpp"
#include "../memory_infra/deload_stats.hpp"
#include <iostream>
namespace gl {
/// @brief Bind the proof-run reporting inputs.
/// @details Borrows read-only views of the original block list and theorem records.
/// @param permanentBodies Address-stable blocks of this batch.
/// @param globalTheoremList Published theorem records.
/// @return Constructed reporting interface.
ProofRunReporter::ProofRunReporter(const std::vector<Memory*>& permanentBodies, const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& globalTheoremList) : permanentBodies(permanentBodies), globalTheoremList(globalTheoremList) {}
/// @brief Print the existing completion and memory observations.
/// @details Samples completion time, checks block/witness accounting, writes the established output lines and resets the deload telemetry at the original run-end boundary.
/// @param initialBodyCount Block count recorded before proof execution.
/// @param initialWitnessCounterSum Witness counter sum recorded before execution.
/// @param t0 Existing proof-run start timestamp.
/// @return No value.
void ProofRunReporter::reportCompletion(size_t initialBodyCount, int64_t initialWitnessCounterSum, std::chrono::high_resolution_clock::time_point t0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto dt = std::chrono::duration<double>(t1 - t0);
    std::cout << "Prover finished." << std::endl;
    std::cout << "Runtime prover: " << dt.count() << " seconds" << std::endl;
    assert(permanentBodies.size() == initialBodyCount);
    int64_t finalWitnessCounterSum = 0;
    int64_t sourceBlockPeakSum = 0;
    for (const Memory* memory : permanentBodies) {
        finalWitnessCounterSum += memory->startInt;
        sourceBlockPeakSum += memory->parkedSourcePeakBytes;
    }
    assert(finalWitnessCounterSum >= initialWitnessCounterSum);
    std::cout << "[WITNESS-STORAGE] permanent_mints_during_prover="
              << finalWitnessCounterSum - initialWitnessCounterSum
              << " source_container_block_peak_sum_bytes=" << sourceBlockPeakSum
              << std::endl;
    std::cout << "Number proven theorems: " << globalTheoremList.size() << std::endl;
    // Statification sizing telemetry: drives the static_pool_bytes /
    // static_block_bytes config audit (see G-52).
    std::cout << "[STATIC-MEMORY] peak blocks in use: "
              << staticMemory().peakBlocksInUse() << " of "
              << staticMemory().totalBlocks() << " ("
              << staticMemory().blockBytes() << " bytes per block)"
              << std::endl;
    if (gpuHostMemory().initialized()) {
        auto& pool = gpuHostMemory();
        const int64_t block = pool.blockBytes();
        std::cout << "[GPU-HOST-MEMORY] reserved_bytes=" << pool.totalBlocks() * block
            << " committed_bytes=" << pool.gpuHostCommittedBytes()
            << " live_bytes=" << pool.blocksInUse() * block
            << " peak_live_bytes=" << pool.attribution().peakBlocks * block
            << " pinned_bytes=" << pool.gpuHostPinnedBytes()
            << " peak_pinned_bytes=" << pool.gpuHostPeakPinnedBytes() << std::endl;
    }
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
}
