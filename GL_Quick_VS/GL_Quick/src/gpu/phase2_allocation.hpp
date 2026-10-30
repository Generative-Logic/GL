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

namespace gl::gpu {

    /// @brief Independent processing demands used to apportion the GPU budget.
    enum class Phase2WorkDimension : uint32_t {
        events, requests, frontier, prefixPayload, prefixVariables, prefixSecondary,
        candidates, evaluationWork, firings, bytes, levels, dependencies,
        markerKeys, markerRemaining, markerArguments, frontierStore, count
    };
    inline constexpr uint32_t phase2WorkDimensionCount =
        static_cast<uint32_t>(Phase2WorkDimension::count);

    /// @brief Recent logical demand, independent of physical buffer capacities.
    struct Phase2WorkDemand { uint64_t items[phase2WorkDimensionCount]{}; };

    /// @brief Completed-batch occupancy and full-buffer pressure for redistribution.
    struct Phase2WorkFeedback {
        uint64_t occupied[phase2WorkDimensionCount]{};
        uint64_t drains[phase2WorkDimensionCount]{};
    };

    /// @brief Fitting immutable input and schedule shapes for one proposed chunk.
    struct Phase2BufferPlanInput {
        Phase2ProjectionCapacity projection{};
        Phase2TaskProjectionCapacity tasks{};
        Phase2FilterScheduleCapacity filter{};
    };

    /// @brief Convert a measured immutable input census into allocation capacities.
    ///
    /// @details
    /// Every one of the 24 projection columns and four task columns receives
    /// its measured count, with one reserved element for empty arrays. Repeated
    /// filter scans have a wide logical count and share physical retained spans.
    /// No measurement is silently narrowed or omitted.
    ///
    /// @param usage Measured projection column counts for the complete chunk.
    /// @param tasks Measured task, batch, term and stump counts.
    /// @param examinedRows Complete repeated-call filter scan volume.
    /// @param retainedRows Maximum retained rows across distinct filter classes.
    /// @param maximumRows Largest individual filter statement slice.
    /// @return Positive capacities for every immutable input allocation.
    /// @invariant Physical column counts fit signed 32-bit device indexing.
    Phase2BufferPlanInput makePhase2BufferPlanInput(const Phase2ProjectionUsage& usage,
        const Phase2TaskProjectionUsage& tasks, uint64_t examinedRows,
        uint64_t retainedRows, uint64_t maximumRows);

    /// @brief Defined scheduling outcomes before any GPU buffer is allocated.
    enum class Phase2BufferPlanState { fits, needsSmallerChunk, needsOutputSegmentation };

    /// @brief Processing capacities and complete block costs for one chunk.
    ///
    /// @details
    /// Allocation order is projection, task, filter, growth, ordering and evaluation. A non-fitting plan names a scheduling action and
    /// must never be used to construct buffers. A single LB rejected by the
    /// minimum-input check is the caller's retained oversized-LB assertion.
    struct Phase2BufferPlan {
        Phase2BufferPlanState state{Phase2BufferPlanState::needsSmallerChunk};
        Phase2GrowthCapacity growth{};
        Phase2OrderingCapacity ordering{};
        Phase2EvaluationCapacity evaluation{};
        Phase2DeviceAllocationSize allocations[6]{};
        uint64_t blocks[6]{};
        uint64_t totalBlocks{0};
        /// Whole blocks of the growth frontier store drawn from the dedicated
        /// frontier-store reservation instead of the working budget.
        uint64_t frontierStoreDedicatedBlocks{0};
    };

    /// @brief Seed processing proportions from recorded complete-pipeline demand.
    ///
    /// @details
    /// Counts seed relative demand; device row sizes and actual library scratch
    /// determine their costs. Candidate-window demand follows frontier demand,
    /// and the shared evaluation-work window follows observed owner expansion.
    ///
    /// @return Nonzero initial demand for every processing dimension.
    Phase2WorkDemand initialPhase2WorkDemand();

    /// @brief Bound one CPU-supported firing with one marker-key fragment.
    ///
    /// @details
    /// CPU scratch strings and sealed spans fit one host arena block. A device
    /// fragment reserves a head plus one key, a complete level run, bounded head
    /// dependencies, remaining-argument identifiers and head argument slices.
    /// Larger collections are handled by value/key windows, not larger spans.
    ///
    /// @param hostBlockBytes Existing CPU arena block size.
    /// @return Minimum output columns covering a CPU-supported single fragment.
    Phase2OutputUsage minimumPhase2OutputUsage(uint32_t hostBlockBytes);

    /// @brief Update allocation demand at a completed batch boundary.
    ///
    /// @details
    /// A decaying peak retains experience while current occupancy and repeated
    /// drains increase the corresponding demand. This changes storage planning
    /// only; no proof decision reads these values.
    ///
    /// @param demand In/out previous processing demand.
    /// @param feedback Completed-batch occupancy and drain counts.
    /// @return Nothing.
    /// @invariant Called only after the measured batch has completed.
    void observePhase2WorkDemand(Phase2WorkDemand& demand, const Phase2WorkFeedback& feedback);

    /// @brief Fit immutable input and weighted processing storage into whole blocks.
    ///
    /// @details
    /// Preserves minimum progress scratch, scales measured demand with exact
    /// allocation queries and lends leftover blocks to a flexible processing
    /// dimension. Simultaneous output floors reserve a pending fragment's needs.
    /// Insufficient input space asks for a smaller LB chunk; insufficient output
    /// floors ask for further output segmentation without consuming work.
    /// With dedicated frontier-store blocks the store takes the largest power
    /// of two up to `phase2FrontierStoreMaximumRows` whose two buffers fit those
    /// blocks, whatever the measured level demand, and the blocks are not
    /// counted against `availableBlocks`, so the store never shrinks the other
    /// columns.
    ///
    /// @param input Immutable input and schedule column capacities.
    /// @param availableBlocks Whole 2 MiB working blocks available for this chunk.
    /// @param demand Nonzero recent processing demand.
    /// @param minimumOutput Optional simultaneous output requirements for lending.
    /// @param minimumOnly Measure minimum progress storage for chunk packing.
    /// @param frontierStoreBlocks Dedicated frontier-store blocks beyond
    ///        `availableBlocks` (`Phase2GpuBudget::frontierStoreBlocks`); zero
    ///        plans the store inside the working blocks.
    /// @return Fitting capacities or an explicit continuation action.
    Phase2BufferPlan planPhase2Buffers(const Phase2BufferPlanInput& input,
        uint64_t availableBlocks, const Phase2WorkDemand& demand,
        Phase2OutputUsage minimumOutput = {}, bool minimumOnly = false,
        uint64_t frontierStoreBlocks = 0);
}
