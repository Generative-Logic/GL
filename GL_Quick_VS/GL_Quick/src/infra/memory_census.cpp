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

#include "memory_census.hpp"
#include "heap_profile.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <iomanip>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#ifdef __APPLE__
#include <mach/mach.h>
#endif

#ifdef USE_MIMALLOC
#include <mimalloc.h>
#if MI_MALLOC_VERSION >= 220
#include <mimalloc-stats.h>
#define GL_CENSUS_HEAP_STATS 1
#endif
#endif

namespace gl {
    namespace memory_census {

        namespace {

            /// @brief The census's whole mutable state — one static instance.
            ///
            /// @details
            /// Fixed-size arrays only, so the sampling path never touches the
            /// heap. Guarded by `mutex`.
            struct CensusState {
                std::mutex mutex;
                PoolPhase current = PoolPhase::Outside;
                PhaseMaxima phases{};
                GpuMaxima gpu{};
            };

            /// @brief The process-wide census state.
            ///
            /// @return The singleton (function-local static).
            CensusState& state() {
                static CensusState instance;
                return instance;
            }

            /// @brief Read mimalloc's committed-bytes counter.
            ///
            /// @param peak `true` for the lifetime peak, `false` for the
            ///             current value.
            /// @return The counter, or -1 when the build has no mimalloc
            ///         statistics.
            int64_t heapCommitted(bool peak) {
#ifdef GL_CENSUS_HEAP_STATS
#ifdef mi_stats_t_decl
                // mimalloc 3.x (Linux distributions): the record carries its
                // own size and statistics version, and the read reports
                // whether it accepted them.
                mi_stats_t_decl(stats);
                const bool read = mi_stats_get(&stats);
                assert(read && "mi_stats_get refused the statistics record");
#else
                // mimalloc 2.2 (the Windows vcpkg build): the size is an argument.
                mi_stats_t stats{};
                mi_stats_get(sizeof(stats), &stats);
#endif
                return peak ? stats.committed.peak : stats.committed.current;
#else
                static_cast<void>(peak);
                return -1;
#endif
            }

#if !defined(_WIN32) && !defined(__APPLE__)
            /// @brief Read one `kB` field of `/proc/self/status` in bytes.
            ///
            /// @param field Field name with its colon, e.g. `"VmRSS:"`.
            /// @return The field value in bytes; asserts it exists.
            int64_t procStatusBytes(const char* field) {
                FILE* f = std::fopen("/proc/self/status", "r");
                assert(f != nullptr && "memory census: /proc/self/status must open");
                char line[256];
                const std::size_t n = std::strlen(field);
                int64_t kib = -1;
                while (std::fgets(line, sizeof(line), f) != nullptr) {
                    if (std::strncmp(line, field, n) == 0) {
                        long long v = 0;
                        const int got = std::sscanf(line + n, "%lld", &v);
                        assert(got == 1);
                        kib = static_cast<int64_t>(v);
                        break;
                    }
                }
                std::fclose(f);
                assert(kib >= 0 && "memory census: status field missing");
                return kib * 1024;
            }
#endif

            /// @brief Sum `totalBlocks() * blockBytes()` over the initialized
            ///        pools — the fixed program-start reservation.
            ///
            /// @return Reserved pool bytes.
            int64_t poolReservationBytes() {
                int64_t total = 0;
                for (GlobalMemoryManager* m : { &staticMemory(), &persistentMemory(),
                                                &mailMemory(), &lbMemory() })
                    if (m->initialized())
                        total += m->totalBlocks() * m->blockBytes();
                return total;
            }

            /// @brief Fold one reading into step `s`'s maxima. Caller holds
            ///        the census mutex.
            ///
            /// @param st     The census state.
            /// @param s      Step index.
            /// @param sample The reading.
            /// @param touched `poolTouchedBytes()` at the reading.
            void foldLocked(CensusState& st, int s, const ProcessSample& sample,
                            int64_t touched) {
                PhaseMaxima& p = st.phases;
                ++p.samples[s];
                p.workingSet[s] = std::max(p.workingSet[s], sample.workingSetBytes);
                p.privateBytes[s] = std::max(p.privateBytes[s], sample.privateBytes);
                p.nonPoolWorkingSet[s] = std::max(p.nonPoolWorkingSet[s],
                    sample.workingSetBytes - touched);
                p.heapCommitted[s] = (p.samples[s] == 1)
                    ? sample.heapCommittedBytes
                    : std::max(p.heapCommitted[s], sample.heapCommittedBytes);
                p.heapLive[s] = (p.samples[s] == 1)
                    ? sample.heapLiveBytes
                    : std::max(p.heapLive[s], sample.heapLiveBytes);
                if (sample.heapLiveBytes >= 0)
                    heap_profile::noteCensusSample(sample.heapLiveBytes,
                        s == static_cast<int>(PoolPhase::Barrier));
            }

            /// @brief Bytes to MiB for the markdown tables.
            ///
            /// @param bytes A byte count.
            /// @return The MiB value.
            double mib(int64_t bytes) {
                return static_cast<double>(bytes) / (1024.0 * 1024.0);
            }

            /// @brief Write a heap-committed figure in MiB, or `n/a` when the
            ///        build has no allocator statistics (-1).
            ///
            /// @param out   Destination stream.
            /// @param bytes The figure, -1 meaning unavailable.
            void writeHeapMib(std::ostream& out, int64_t bytes) {
                if (bytes < 0) out << "n/a";
                else out << std::fixed << std::setprecision(1) << mib(bytes);
            }

        } // namespace

        /// @brief Read the process's current memory counters.
        ///
        /// @return The current reading; asserts the query succeeds.
        ProcessSample readProcess() {
            ProcessSample s;
#ifdef _WIN32
            PROCESS_MEMORY_COUNTERS_EX pmc{};
            pmc.cb = sizeof(pmc);
            const BOOL ok = GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));
            assert(ok && "memory census: GetProcessMemoryInfo failed");
            s.workingSetBytes = static_cast<int64_t>(pmc.WorkingSetSize);
            s.privateBytes = static_cast<int64_t>(pmc.PrivateUsage);
#elif defined(__APPLE__)
            // macOS has no /proc: the task's virtual-memory record holds the
            // resident size and the physical footprint (private memory).
            task_vm_info_data_t info{};
            mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
            const kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&info), &count);
            assert(status == KERN_SUCCESS && "memory census: task_info(TASK_VM_INFO) failed");
            s.workingSetBytes = static_cast<int64_t>(info.resident_size);
            s.privateBytes = static_cast<int64_t>(info.phys_footprint);
#else
            s.workingSetBytes = procStatusBytes("VmRSS:");
            s.privateBytes = procStatusBytes("VmData:");
#endif
            s.heapCommittedBytes = heapCommitted(false);
            s.heapLiveBytes = heap_profile::countsNewDelete() ? heap_profile::liveBytes() : -1;
            return s;
        }

        /// @brief Read the lifetime process peaks.
        ///
        /// @return The peaks; asserts the query succeeds.
        ProcessPeaks readProcessPeaks() {
            ProcessPeaks p;
#ifdef _WIN32
            PROCESS_MEMORY_COUNTERS_EX pmc{};
            pmc.cb = sizeof(pmc);
            const BOOL ok = GetProcessMemoryInfo(GetCurrentProcess(),
                reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc));
            assert(ok && "memory census: GetProcessMemoryInfo failed");
            p.peakWorkingSetBytes = static_cast<int64_t>(pmc.PeakWorkingSetSize);
            p.peakPrivateBytes = static_cast<int64_t>(pmc.PeakPagefileUsage);
#elif defined(__APPLE__)
            task_vm_info_data_t info{};
            mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
            const kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&info), &count);
            assert(status == KERN_SUCCESS && "memory census: task_info(TASK_VM_INFO) failed");
            p.peakWorkingSetBytes = static_cast<int64_t>(info.resident_size_peak);
            p.peakPrivateBytes = static_cast<int64_t>(info.ledger_phys_footprint_peak);
#else
            p.peakWorkingSetBytes = procStatusBytes("VmHWM:");
            p.peakPrivateBytes = procStatusBytes("VmPeak:");
#endif
            p.heapCommittedPeakBytes = heapCommitted(true);
            return p;
        }

        /// @brief Sum of every initialized pool's carve high-water in bytes.
        ///
        /// @return Touched pool bytes.
        int64_t poolTouchedBytes() {
            int64_t total = 0;
            for (GlobalMemoryManager* m : { &staticMemory(), &persistentMemory(),
                                            &mailMemory(), &lbMemory() })
                if (m->initialized())
                    total += m->peakBlocksInUse() * m->blockBytes();
            return total;
        }

        /// @brief Close the current step with a sample and make `next`
        ///        current for the census and the main pool.
        ///
        /// @param next The step now starting; not `PoolPhase::Count`.
        void markPhase(PoolPhase next) {
            assert(next != PoolPhase::Count);
            const ProcessSample sample = readProcess();
            const int64_t touched = poolTouchedBytes();
            CensusState& st = state();
            {
                std::lock_guard<std::mutex> lock(st.mutex);
                foldLocked(st, static_cast<int>(st.current), sample, touched);
                st.current = next;
            }
            if (staticMemory().initialized())
                staticMemory().setAttributionPhase(next);
        }

        /// @brief Fold one sample into the current step without switching.
        void sampleNow() {
            const ProcessSample sample = readProcess();
            const int64_t touched = poolTouchedBytes();
            CensusState& st = state();
            std::lock_guard<std::mutex> lock(st.mutex);
            foldLocked(st, static_cast<int>(st.current), sample, touched);
        }

        /// @brief Fold one CUDA chunk's host-arena bytes into the maxima.
        ///
        /// @param bytes The chunk's host-arena bytes.
        void noteGpuChunk(const GpuChunkHostBytes& bytes) {
            CensusState& st = state();
            std::lock_guard<std::mutex> lock(st.mutex);
            GpuMaxima& g = st.gpu;
            ++g.chunks;
            GpuChunkHostBytes& m = g.maxOfEach;
            m.projection = std::max(m.projection, bytes.projection);
            m.shards = std::max(m.shards, bytes.shards);
            m.tasks = std::max(m.tasks, bytes.tasks);
            m.filterSchedule = std::max(m.filterSchedule, bytes.filterSchedule);
            m.growthSchedule = std::max(m.growthSchedule, bytes.growthSchedule);
            m.sealing = std::max(m.sealing, bytes.sealing);
            m.pinned = std::max(m.pinned, bytes.pinned);
            const uint64_t total = bytes.projection + bytes.shards + bytes.tasks
                + bytes.filterSchedule + bytes.growthSchedule + bytes.sealing
                + bytes.pinned;
            g.maxChunkTotal = std::max(g.maxChunkTotal, total);
        }

        /// @brief Record the device totals once (repeats must match).
        ///
        /// @param totalBytes          Device total.
        /// @param freeBeforePoolBytes Device free before the reservation.
        /// @param budgetBytes         The fixed reservation.
        void noteGpuDevice(uint64_t totalBytes, uint64_t freeBeforePoolBytes,
                           uint64_t budgetBytes) {
            CensusState& st = state();
            std::lock_guard<std::mutex> lock(st.mutex);
            GpuMaxima& g = st.gpu;
            if (g.deviceSeen) {
                assert(g.deviceTotalBytes == totalBytes
                    && g.deviceFreeBeforePoolBytes == freeBeforePoolBytes
                    && g.deviceBudgetBytes == budgetBytes
                    && "memory census: the device reservation changed");
                return;
            }
            g.deviceSeen = true;
            g.deviceTotalBytes = totalBytes;
            g.deviceFreeBeforePoolBytes = freeBeforePoolBytes;
            g.deviceBudgetBytes = budgetBytes;
        }

        /// @brief Snapshot the per-step maxima.
        ///
        /// @return A copy.
        PhaseMaxima phaseMaxima() {
            CensusState& st = state();
            std::lock_guard<std::mutex> lock(st.mutex);
            return st.phases;
        }

        /// @brief Snapshot the CUDA maxima and device totals.
        ///
        /// @return A copy.
        GpuMaxima gpuMaxima() {
            CensusState& st = state();
            std::lock_guard<std::mutex> lock(st.mutex);
            return st.gpu;
        }

        /// @brief Print the batch census to the diagnostics log and the
        ///        memory log.
        ///
        /// @param diagnostics The diagnostics log stream.
        /// @param markdown    The run's memory log stream.
        void writeReport(std::ostream& diagnostics, std::ostream& markdown) {
            struct NamedPool { const char* name; GlobalMemoryManager* pool; };
            const NamedPool pools[] = {
                { "main", &staticMemory() }, { "persistent", &persistentMemory() },
                { "mail", &mailMemory() }, { "lb", &lbMemory() } };

            markdown << "\n### Pool attribution by purpose (MiB)\n\n"
                     << "`At peak` is the split at the pool's lifetime high-water "
                        "instant (sums to the peak); `own peak` is each purpose's "
                        "own high-water (never summed).\n\n"
                     << "| Pool | Purpose | At peak | Own peak |\n"
                     << "|---|---|---:|---:|\n";
            for (const NamedPool& np : pools) {
                if (!np.pool->initialized()) continue;
                const PoolAttribution a = np.pool->attribution();
                const int64_t bb = np.pool->blockBytes();
                diagnostics << "[pool-purpose] pool=" << np.name
                            << " peak_blocks=" << a.peakBlocks
                            << " block_bytes=" << bb;
                for (int k = 0; k < kPoolPurposeCount; ++k)
                    diagnostics << " at_peak_" << poolPurposeName(static_cast<PoolPurpose>(k))
                                << "=" << a.atPeak[k];
                for (int k = 0; k < kPoolPurposeCount; ++k)
                    diagnostics << " own_peak_" << poolPurposeName(static_cast<PoolPurpose>(k))
                                << "=" << a.purposePeak[k];
                diagnostics << std::endl;
                for (int k = 0; k < kPoolPurposeCount; ++k) {
                    if (a.atPeak[k] == 0 && a.purposePeak[k] == 0) continue;
                    markdown << "| " << np.name << " | "
                             << poolPurposeName(static_cast<PoolPurpose>(k))
                             << " | " << std::fixed << std::setprecision(1)
                             << mib(a.atPeak[k] * bb) << " | "
                             << mib(a.purposePeak[k] * bb) << " |\n";
                }
            }

            if (staticMemory().initialized()) {
                const PoolAttribution a = staticMemory().attribution();
                const int64_t bb = staticMemory().blockBytes();
                markdown << "\n### Main-pool peak per prover step (MiB)\n\n"
                         << "Highest level reached while the step was current, "
                            "and its purpose split at that instant.\n\n"
                         << "| Step | Peak";
                for (int k = 0; k < kPoolPurposeCount; ++k)
                    markdown << " | " << poolPurposeName(static_cast<PoolPurpose>(k));
                markdown << " |\n|---|---:";
                for (int k = 0; k < kPoolPurposeCount; ++k) markdown << "|---:";
                markdown << "|\n";
                for (int s = 0; s < kPoolPhaseCount; ++s) {
                    if (a.phasePeak[s] == 0) continue;
                    const char* name = poolPhaseName(static_cast<PoolPhase>(s));
                    diagnostics << "[pool-phase] pool=main phase=" << name
                                << " peak_blocks=" << a.phasePeak[s];
                    markdown << "| " << name << " | " << std::fixed
                             << std::setprecision(1) << mib(a.phasePeak[s] * bb);
                    for (int k = 0; k < kPoolPurposeCount; ++k) {
                        diagnostics << " " << poolPurposeName(static_cast<PoolPurpose>(k))
                                    << "=" << a.phaseAtPeak[s][k];
                        markdown << " | " << mib(a.phaseAtPeak[s][k] * bb);
                    }
                    diagnostics << std::endl;
                    markdown << " |\n";
                }
            }

            const ProcessPeaks peaks = readProcessPeaks();
            const int64_t reservation = poolReservationBytes();
            const int64_t touched = poolTouchedBytes();
            diagnostics << "[process-memory] os_peak_working_set_bytes="
                        << peaks.peakWorkingSetBytes
                        << " os_peak_private_bytes=" << peaks.peakPrivateBytes
                        << " heap_committed_peak_bytes=" << peaks.heapCommittedPeakBytes
                        << " pool_reservation_bytes=" << reservation
                        << " pool_touched_bytes=" << touched << std::endl;
            markdown << "\n### Process memory (MiB)\n\n"
                     << "| Measure | MiB | Meaning |\n|---|---:|---|\n"
                     << std::fixed << std::setprecision(1)
                     << "| Peak working set | " << mib(peaks.peakWorkingSetBytes)
                     << " | Operating-system lifetime peak of resident memory |\n"
                     << "| Peak private bytes | " << mib(peaks.peakPrivateBytes)
                     << " | Lifetime peak commit charge (includes the pools' committed blocks) |\n"
                     << "| Heap committed peak | ";
            writeHeapMib(markdown, peaks.heapCommittedPeakBytes);
            markdown << " | mimalloc committed peak (heap only; the pools reserve their own address space) |\n"
                     << "| Pool reservations | " << mib(reservation)
                     << " | Four static pools' address space, reserved at program start |\n"
                     << "| Pool bytes committed | " << mib(touched)
                     << " | Sum of the pools' carve high-waters; a block is committed when first carved |\n";

            const PhaseMaxima pm = phaseMaxima();
            markdown << "\n### Process memory per prover step (maximum over the "
                        "samples at the step's seams, MiB)\n\n"
                     << "| Step | Samples | Working set | Private | Outside pools (estimate) | Heap committed | Heap live (C++) |\n"
                     << "|---|---:|---:|---:|---:|---:|---:|\n";
            for (int s = 0; s < kPoolPhaseCount; ++s) {
                if (pm.samples[s] == 0) continue;
                const char* name = poolPhaseName(static_cast<PoolPhase>(s));
                diagnostics << "[process-phase] phase=" << name
                            << " samples=" << pm.samples[s]
                            << " max_working_set_bytes=" << pm.workingSet[s]
                            << " max_private_bytes=" << pm.privateBytes[s]
                            << " max_non_pool_working_set_bytes=" << pm.nonPoolWorkingSet[s]
                            << " max_heap_committed_bytes=" << pm.heapCommitted[s]
                            << " max_heap_live_bytes=" << pm.heapLive[s]
                            << std::endl;
                markdown << "| " << name << " | " << pm.samples[s] << " | "
                         << std::fixed << std::setprecision(1)
                         << mib(pm.workingSet[s]) << " | " << mib(pm.privateBytes[s])
                         << " | " << mib(pm.nonPoolWorkingSet[s]) << " | ";
                writeHeapMib(markdown, pm.heapCommitted[s]);
                markdown << " | ";
                writeHeapMib(markdown, pm.heapLive[s]);
                markdown << " |\n";
            }
            heap_profile::writeReport(diagnostics, markdown);

            const GpuMaxima g = gpuMaxima();
            if (g.deviceSeen || g.chunks > 0) {
                const GpuChunkHostBytes& m = g.maxOfEach;
                diagnostics << "[gpu-memory] device_total_bytes=" << g.deviceTotalBytes
                            << " device_free_before_pool_bytes=" << g.deviceFreeBeforePoolBytes
                            << " device_budget_bytes=" << g.deviceBudgetBytes
                            << " chunks=" << g.chunks
                            << " max_chunk_host_total_bytes=" << g.maxChunkTotal
                            << " max_projection_bytes=" << m.projection
                            << " max_shards_bytes=" << m.shards
                            << " max_tasks_bytes=" << m.tasks
                            << " max_filter_schedule_bytes=" << m.filterSchedule
                            << " max_growth_schedule_bytes=" << m.growthSchedule
                            << " max_sealing_bytes=" << m.sealing
                            << " max_pinned_bytes=" << m.pinned << std::endl;
                markdown << "\n### CUDA route: device and host arenas (MiB)\n\n"
                         << "| Measure | MiB |\n|---|---:|\n" << std::fixed
                         << std::setprecision(1)
                         << "| Device total | " << mib(static_cast<int64_t>(g.deviceTotalBytes)) << " |\n"
                         << "| Device used before the reservation (context, other users) | "
                         << mib(static_cast<int64_t>(g.deviceTotalBytes - g.deviceFreeBeforePoolBytes)) << " |\n"
                         << "| Fixed Phase 2 reservation | " << mib(static_cast<int64_t>(g.deviceBudgetBytes)) << " |\n"
                         << "| Chunks | " << g.chunks << " |\n"
                         << "| Largest chunk host total | " << mib(static_cast<int64_t>(g.maxChunkTotal)) << " |\n"
                         << "| Largest host projection | " << mib(static_cast<int64_t>(m.projection)) << " |\n"
                         << "| Largest projection shards' own column storage (shards are slices of the projection) | "
                         << mib(static_cast<int64_t>(m.shards)) << " |\n"
                         << "| Largest task arena | " << mib(static_cast<int64_t>(m.tasks)) << " |\n"
                         << "| Largest filter schedule | " << mib(static_cast<int64_t>(m.filterSchedule)) << " |\n"
                         << "| Largest growth schedule | " << mib(static_cast<int64_t>(m.growthSchedule)) << " |\n"
                         << "| Largest sealing arena | " << mib(static_cast<int64_t>(m.sealing)) << " |\n"
                         << "| Largest pinned staging | " << mib(static_cast<int64_t>(m.pinned)) << " |\n";
            }
            assert(diagnostics.good() && markdown.good());
        }

        /// @brief Discard all census state (tests only).
        void resetForTest() {
            CensusState& st = state();
            std::lock_guard<std::mutex> lock(st.mutex);
            st.current = PoolPhase::Outside;
            st.phases = PhaseMaxima{};
            st.gpu = GpuMaxima{};
        }

    } // namespace memory_census
} // namespace gl
