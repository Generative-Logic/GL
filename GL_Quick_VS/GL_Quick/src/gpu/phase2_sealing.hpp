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

#include "phase2_cuda.hpp"
#include "phase2_allocation.hpp"
#include "phase2_projection.hpp"
#include "../memory.hpp"

#include <cstdint>
#include <vector>

namespace gl::gpu {

    /// @brief A CUDA discharge observed in the current, still-retained growth level.
    inline constexpr int64_t phase2PendingBlockStop = INT64_MAX - 1;

    /// @brief Ownership policy for one completed output download.
    ///
    /// @details
    /// Streaming downloads use raw device order and leave page sets filling.
    /// An optional receiver assembles fragments instead of appending records.
    /// A repeated marker header reuses already-sealed scalar and invariant data;
    /// only that marker's next key fragment is copied on subsequent downloads.
    struct Phase2SealingOptions {
        bool canonicalOrder{ true };
        bool finishOutputs{ true };
        void (*receive)(void*, uint32_t, const FiringRecord&, const DevicePhase2FiringRecord&){ nullptr };
        void* context{ nullptr };
        const FiringRecord* repeatedMarkerHeader{ nullptr };
        // One state per block: INT64_MAX active, pending until level end, zero stopped.
        int64_t* blockStops{ nullptr };
    };

    /// @brief Borrowed device owners retained until a complete request stream ends.
    struct Phase2StreamBuffers {
        const CudaPhase2ProjectionBuffer& projection;
        const CudaPhase2TaskBuffer& tasks;
        const CudaPhase2FilterSortBuffer& filter;
        CudaPhase2GrowthBuffer& growth;
        CudaPhase2OrderingBuffer& ordering;
        CudaPhase2EvaluationBuffer& evaluation;
    };

    /// @brief Logical totals and measured storage pressure across all drains.
    struct Phase2StreamExecutionResult {
        Phase2GrowthStreamResult growth{};
        Phase2WorkFeedback feedback{};
        uint64_t evaluatedRequests{0};
        uint64_t sealedRecords{0};
        uint64_t firingRecords{0};
        uint64_t outputDownloads{0};
        uint64_t outputTransferBytes{0};
        double generationAndDrainSeconds{0};
        double sealingSeconds{0};
        uint64_t outputColumns[7]{};
    };

    /// @brief Processor staging for device-decided Phase 2 output, sized to
    ///        the output the chunk actually produces.
    ///
    /// @details
    /// Construction records the evaluator's immutable capacity and allocates
    /// nothing. `downloadAndSeal` grows each download column and the per-record
    /// sealing scratch to the used counts of its window (the largest window so
    /// far; never beyond the capacity) and reuses them for later windows, so a
    /// chunk holds only the host memory its own output needs. It decodes
    /// projected identifiers and copies already-decided record content into
    /// ordinary `FiringRecord` page chains. It performs no rule evaluation,
    /// semantic gate, canonical sort, or proof-state mutation.
    class Phase2SealingArena {
    public:
        /// @brief Record the evaluator capacity; allocates nothing.
        ///
        /// @details
        /// The download columns, the canonical index column and the
        /// string/dependency scratch start empty and grow in `downloadAndSeal`
        /// to the used counts of the windows it seals.
        ///
        /// @param fixedCapacity Immutable evaluator element and byte ceilings.
        /// @return An empty reusable sealing arena.
        /// @invariant No column grows beyond the evaluator capacity.
        explicit Phase2SealingArena(Phase2EvaluationCapacity fixedCapacity);

        Phase2SealingArena(const Phase2SealingArena&) = delete;
        Phase2SealingArena& operator=(const Phase2SealingArena&) = delete;
        Phase2SealingArena(Phase2SealingArena&&) = delete;
        Phase2SealingArena& operator=(Phase2SealingArena&&) = delete;

        /// @brief Download retained GPU records into their owning block pages.
        ///
        /// @details
        /// Downloads the exact used evaluator prefixes and, when requested, the
        /// canonical index permutation. Each retained header is translated to an
        /// ordinary `FiringRecord` by byte-copying generated slices, decoding
        /// local NameMap and rule-interner identifiers from the same immutable
        /// projection, and preserving all device verdict flags. Output page sets
        /// receive records in the selected order. Streaming callers leave those
        /// sets filling and may assemble key fragments through a receiver.
        /// No request, substitution, scope, admission, mail, doom, or provenance
        /// decision is recomputed on the processor.
        ///
        /// @param evaluation Device evaluator containing the completed sweep.
        /// @param projection Host twin of the projection uploaded for that sweep.
        /// @param used Exact materialization counts returned by the evaluator.
        /// @param retainedFiringCount Number of records retained in this window.
        /// @param outputs One bound, filling `SealedPageSet` per logical block.
        /// @param outputCount Number of output sets; exactly the projected block
        ///                    count.
        /// @param options Ordering, page-set completion and fragment ownership.
        /// @return Number of `FiringRecord`s appended across all output sets.
        /// @invariant The evaluator and projection describe the same sweep and
        ///            canonical indices, when selected, are grouped by block.
        uint32_t downloadAndSeal(
            const CudaPhase2EvaluationBuffer& evaluation,
            const Phase2ProjectionArena& projection,
            const Phase2FiringExpressionResult& used,
            uint32_t retainedFiringCount,
            SealedPageSet* const* outputs,
            uint32_t outputCount,
            Phase2SealingOptions options = {});

        /// @brief Execute a complete GPU request stream through bounded downloads.
        ///
        /// @details
        /// Freezes generation at each request drain, evaluates every request
        /// through bounded owner/value windows, and copies
        /// complete results to host-owned pages. Marker keys assemble before a
        /// record is published. No inference or substitution runs on the host.
        /// A CUDA discharge retains the complete current growth level and stops
        /// subsequent levels for that block. Capacity drains never commit a stop.
        /// Output pages seal only after generation and evaluation are exhausted.
        ///
        /// @param buffers Device owners holding the immutable input and scratch.
        /// @param projection Host twin of the uploaded projection.
        /// @param tasks Host twin of uploaded original task identities.
        /// @param schedule Original call schedule used by generation.
        /// @param parameters Existing GPU growth policy.
        /// @param plan Allocated processing capacities for occupancy measurement.
        /// @param outputs Bound filling pages for each projected logical block.
        /// @param outputCount Number of projected blocks and output page sets.
        /// @param taskSubkeyCounts Complete cumulative subkey counts per task.
        /// @param taskCapacity Number of supplied task counter slots.
        /// @param doomLines Optional per-block stop states; returns zero or INT64_MAX.
        /// @return Logical totals, transfer bytes and completed storage pressure.
        /// @invariant Scratch fits one CPU-supported output fragment; snapshots
        ///            stay immutable and no statement is deposited during drains.
        Phase2StreamExecutionResult runWindows(Phase2StreamBuffers& buffers,
            const Phase2ProjectionArena& projection,
            const Phase2TaskProjectionArena& tasks,
            const Phase2GrowthScheduleArena& schedule,
            const DevicePhase2GrowthParameters& parameters,
            const Phase2BufferPlan& plan, SealedPageSet* const* outputs,
            uint32_t outputCount, uint64_t* taskSubkeyCounts, uint32_t taskCapacity, int64_t* doomLines = nullptr);

        /// @brief Heap bytes the sealing arena's download and scratch columns
        ///        hold — the largest window sealed so far in this chunk.
        ///
        /// @details
        /// Sums `vectorCapacityBytes` over every evaluator output column and
        /// the reusable sealing scratch; zero before the first window.
        /// Observability only (Rule 16).
        ///
        /// @return Reserved heap bytes.
        uint64_t hostBytes() const {
            return vectorCapacityBytes(firingRecords_, generatedBytes_,
                levelValues_, originDependencies_, markerKeys_,
                markerRemainingArgs_, markerArgs_, firingOrder_,
                sealedStringsScratch_, sealedDependenciesScratch_);
        }

    private:
        Phase2EvaluationCapacity capacity_{};
        std::vector<DevicePhase2FiringRecord> firingRecords_;
        std::vector<char> generatedBytes_;
        std::vector<int32_t> levelValues_;
        std::vector<DeviceEvaluationDependency> originDependencies_;
        std::vector<DeviceEvaluationByteSlice> markerKeys_;
        std::vector<int32_t> markerRemainingArgs_;
        std::vector<DeviceEvaluationByteSlice> markerArgs_;
        std::vector<uint32_t> firingOrder_;
        std::vector<SealedString> sealedStringsScratch_;
        std::vector<SealedExpressionWithValidity> sealedDependenciesScratch_;
    };

} // namespace gl::gpu
