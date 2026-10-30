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

#include "../memory_infra/global_memory_manager.hpp"

#include <cstdint>
#include <ostream>

namespace gl {

    /// @brief Whole-process memory census of one batch — the process,
    ///        heap, device and CUDA host-arena side of the memory playbook.
    ///
    /// @details
    /// The four static pools report their own peaks and, since the pool
    /// attribution, which consumer filled them (`GlobalMemoryManager::
    /// attribution`). Everything outside the pools was unmeasured: the malloc
    /// heap (among others the CUDA route's fixed-capacity host projection
    /// arenas, which are `std::vector` columns), the page-locked upload staging,
    /// the CUDA driver's own host memory, and the device reservation itself.
    /// This namespace closes that gap with three always-on, cheap instruments:
    ///
    /// - **Process samples per prover step.** `markPhase` (called at the
    ///   prover's single-threaded seams) reads the operating system's working
    ///   set and private bytes, and — when the build links mimalloc — the
    ///   heap's committed bytes, and folds them into the maxima of the step
    ///   that is ending; `sampleNow` folds an extra sample into the current step
    ///   at a known intra-step high point (the CUDA projection). It also keeps
    ///   the main pool's attribution step in lock-step (`setAttributionPhase`).
    /// - **CUDA host-arena bytes per chunk.** `noteGpuChunk` keeps the per-field
    ///   maxima of the chunk's host projection, projection shards, task,
    ///   schedule and sealing arenas, and pinned staging.
    /// - **Device totals.** `noteGpuDevice` records the card's total, the free
    ///   bytes before the fixed reservation, and the reservation.
    ///
    /// `writeReport` prints the census once per batch to the diagnostics log
    /// and as markdown tables to the run's memory log, beside the existing pool
    /// tables.
    ///
    /// @invariant Nothing here is a proof input (Rule 16): no prover, deload or
    ///            steward decision reads a census value.
    /// @invariant `markPhase` / `sampleNow` / `noteGpuChunk` take one internal
    ///            mutex and fixed-size state only — no heap on the sampling
    ///            path.
    /// @see `docs/agentic_swdd/_meta/memory_measurement.md`,
    ///      `GlobalMemoryManager::attribution`.
    namespace memory_census {

        /// @brief One reading of the process's memory counters.
        ///
        /// @details
        /// `workingSetBytes` is the resident set (RAM pages mapped now).
        /// `privateBytes` is the commit charge of private memory (Windows
        /// `PrivateUsage`; Linux `VmData`). `heapCommittedBytes` is mimalloc's
        /// committed-bytes counter, or -1 when the build does not link a
        /// mimalloc that exposes its statistics. `heapLiveBytes` is the live
        /// C++ heap (`heap_profile::liveBytes`), or -1 when the build keeps the
        /// default allocation operators.
        struct ProcessSample {
            int64_t workingSetBytes = 0;
            int64_t privateBytes = 0;
            int64_t heapCommittedBytes = -1;
            int64_t heapLiveBytes = -1;
        };

        /// @brief Read the process's current memory counters.
        ///
        /// @details
        /// One operating-system query (plus one mimalloc statistics copy when
        /// available). Asserts that the query succeeds (Rule 19).
        ///
        /// @return The current reading.
        ProcessSample readProcess();

        /// @brief Lifetime peaks the operating system and the allocator keep
        ///        for the whole process.
        ///
        /// @details
        /// `peakWorkingSetBytes` (Windows `PeakWorkingSetSize`, Linux `VmHWM`),
        /// `peakPrivateBytes` (Windows `PeakPagefileUsage`, Linux `VmPeak`), and
        /// mimalloc's committed peak (-1 when unavailable). These are true
        /// peaks, unlike the per-step maxima, which only see the sampled
        /// instants.
        struct ProcessPeaks {
            int64_t peakWorkingSetBytes = 0;
            int64_t peakPrivateBytes = 0;
            int64_t heapCommittedPeakBytes = -1;
            int64_t pageFaultCount = -1;
        };

        /// @brief Read the lifetime process peaks.
        ///
        /// @return The peaks; asserts the operating-system query succeeds.
        ProcessPeaks readProcessPeaks();

        /// @brief Current committed backing across the four static pools.
        ///
        /// @details Decommitted returned blocks are excluded. Subtracting this
        /// from working set remains an estimate of residency outside the pools:
        /// committed pages may be untouched or trimmed by the operating system.
        ///
        /// @return Current committed bytes of initialized pools.
        int64_t poolTouchedBytes();

        /// @brief Close the current prover step with a process sample and make
        ///        `next` current, for the census and the main pool alike.
        ///
        /// @details
        /// Reads the process counters, folds them into the maxima of the step
        /// that is ending, then switches both the census's current step and
        /// the main pool's attribution step (`staticMemory().
        /// setAttributionPhase`) to `next`. Called at the prover's
        /// single-threaded seams. Thread-safe (one mutex).
        ///
        /// @param next The step now starting; not `PoolPhase::Count`.
        void markPhase(PoolPhase next);

        /// @brief Fold one process sample into the current step's maxima
        ///        without switching the step.
        ///
        /// @details
        /// Used at a known intra-step high point — after the CUDA route
        /// merged its host projection shards and before it releases them.
        /// Thread-safe (one mutex).
        void sampleNow();

        /// @brief Host bytes one CUDA chunk's arenas reserve.
        ///
        /// @details
        /// `projection` is the largest concurrent construction-window and
        /// retained-image column footprint. `shards` counts separately owned
        /// shard columns (zero for slices). `tasks` / `filterSchedule` /
        /// `growthSchedule` / `sealing` count the other fixed-capacity host
        /// arenas, and `pinned` counts page-locked upload staging.
        struct GpuChunkHostBytes {
            uint64_t projection = 0;
            uint64_t shards = 0;
            uint64_t tasks = 0;
            uint64_t filterSchedule = 0;
            uint64_t growthSchedule = 0;
            uint64_t sealing = 0;
            uint64_t pinned = 0;
        };

        /// @brief Fold one CUDA chunk's host-arena bytes into the batch
        ///        maxima.
        ///
        /// @details
        /// Keeps each field's maximum over chunks, the maximum of the per-chunk
        /// accounting sum, and the chunk count. The sum is an upper bound when
        /// construction and sealing peak at different times; process samples
        /// measure actual concurrent residency. Thread-safe (one mutex).
        ///
        /// @param bytes The chunk's host-arena bytes.
        void noteGpuChunk(const GpuChunkHostBytes& bytes);

        /// @brief Record the device's memory totals once.
        ///
        /// @details
        /// Called when the CUDA route builds its fixed device reservation. A
        /// repeated call with the same values is accepted (the reservation is
        /// process-wide); different values assert.
        ///
        /// @param totalBytes           Device total.
        /// @param freeBeforePoolBytes  Device free before the reservation.
        /// @param budgetBytes          The fixed reservation: the working budget
        ///                             plus the dedicated frontier-store blocks.
        void noteGpuDevice(uint64_t totalBytes, uint64_t freeBeforePoolBytes,
                           uint64_t budgetBytes);

        /// @brief Per-step maxima of the process samples.
        ///
        /// @details
        /// Indexed by `PoolPhase`. `samples[s]` counts the readings folded into
        /// step `s`; the other arrays hold the maximum over those readings of
        /// the working set, the private bytes, the working set minus
        /// `poolTouchedBytes()` at that instant (the outside-the-pools
        /// estimate), and the heap's committed bytes (-1 when unavailable).
        struct PhaseMaxima {
            int64_t samples[kPoolPhaseCount] = {};
            int64_t workingSet[kPoolPhaseCount] = {};
            int64_t privateBytes[kPoolPhaseCount] = {};
            int64_t nonPoolWorkingSet[kPoolPhaseCount] = {};
            int64_t heapCommitted[kPoolPhaseCount] = {};
            int64_t heapLive[kPoolPhaseCount] = {};
        };

        /// @brief Snapshot the per-step maxima.
        ///
        /// @return A copy taken under the census mutex.
        PhaseMaxima phaseMaxima();

        /// @brief Batch maxima of the CUDA host arenas and the device totals.
        ///
        /// @details
        /// `maxOfEach` holds each field's own maximum over chunks (reached in
        /// possibly different chunks); `maxChunkTotal` is the largest
        /// single-chunk sum. `deviceSeen` is false on a processor-route batch.
        struct GpuMaxima {
            int64_t chunks = 0;
            GpuChunkHostBytes maxOfEach{};
            uint64_t maxChunkTotal = 0;
            bool deviceSeen = false;
            uint64_t deviceTotalBytes = 0;
            uint64_t deviceFreeBeforePoolBytes = 0;
            uint64_t deviceBudgetBytes = 0;
        };

        /// @brief Snapshot the CUDA maxima and device totals.
        ///
        /// @return A copy taken under the census mutex.
        GpuMaxima gpuMaxima();

        /// @brief Print the batch census: diagnostics lines and markdown
        ///        tables.
        ///
        /// @details
        /// Writes, to `diagnostics`, one `[pool-purpose]` line per pool (the
        /// purpose split at the pool's peak and each purpose's own peak), one
        /// `[pool-phase]` line per main-pool step that ran, one
        /// `[process-memory]` line (lifetime process and heap peaks, pool
        /// reservation and touched bytes), one `[process-phase]` line per
        /// sampled step, and one `[gpu-memory]` line when the CUDA route ran.
        /// Writes the same content as markdown tables to `markdown` (the run's
        /// `.debug/memory_<descriptor>.log`). Called once per batch from the
        /// run-mode teardown. Asserts the streams stay good (Rule 19).
        ///
        /// @param diagnostics The diagnostics log stream.
        /// @param markdown    The run's memory log stream.
        void writeReport(std::ostream& diagnostics, std::ostream& markdown);

        /// @brief Discard all census state (tests only).
        ///
        /// @details
        /// Restores the process-start state — current step `Outside`, zero
        /// maxima, no device record — so a unit test can run an independent
        /// scenario. Not called by the pipeline. Does not touch the pools.
        void resetForTest();

    } // namespace memory_census

} // namespace gl
