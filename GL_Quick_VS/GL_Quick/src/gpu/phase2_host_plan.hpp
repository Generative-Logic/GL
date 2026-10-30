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

#include "phase2_allocation.hpp"
#include "phase2_sealing.hpp"
#include "request_frontier_store.hpp"
#include <optional>

namespace gl::gpu {

    /// @brief Add one immutable projection census to another.
    /// @details Checks each wide sum without narrowing device counts.
    /// @param total Destination census.
    /// @param added Source census.
    /// @return Nothing.
    inline void addPhase2HostUsage(Phase2ProjectionUsage& total,
        const Phase2ProjectionUsage& added) {
        constexpr uint64_t Phase2ProjectionUsage::* fields[] = {
            &Phase2ProjectionUsage::logicalBlocks,
            &Phase2ProjectionUsage::statements,
            &Phase2ProjectionUsage::nameRecords,
            &Phase2ProjectionUsage::nameBytes,
            &Phase2ProjectionUsage::nameSlots,
            &Phase2ProjectionUsage::ruleStringRecords,
            &Phase2ProjectionUsage::ruleStringBytes,
            &Phase2ProjectionUsage::byteMapViews,
            &Phase2ProjectionUsage::byteMapEntries,
            &Phase2ProjectionUsage::byteMapSlots,
            &Phase2ProjectionUsage::byteKeyBytes,
            &Phase2ProjectionUsage::blobRecords,
            &Phase2ProjectionUsage::blobBytes,
            &Phase2ProjectionUsage::reverseMapViews,
            &Phase2ProjectionUsage::reverseMapEntries,
            &Phase2ProjectionUsage::reverseMapSlots,
            &Phase2ProjectionUsage::reverseKeyBytes,
            &Phase2ProjectionUsage::reverseOwners,
            &Phase2ProjectionUsage::podMapViews,
            &Phase2ProjectionUsage::podMapEntries,
            &Phase2ProjectionUsage::podMapSlots,
            &Phase2ProjectionUsage::podRunValues,
            &Phase2ProjectionUsage::mandatoryStatementKeys,
            &Phase2ProjectionUsage::metadataBytes
        };
        for (const auto field : fields) {
            assert(total.*field <= UINT64_MAX - added.*field);
            total.*field += added.*field;
        }
    }

    /// @brief Bound the pages of one append-only paged column.
    /// @details Includes element packing and all three page-directory levels.
    /// The fixed GPU-host page geometry is part of this storage contract.
    /// @param count Maximum retained elements.
    /// @param elementBytes Positive element size no larger than one page.
    /// @return Conservative complete page count, zero for an empty column.
    inline uint64_t phase2HostColumnPages(uint64_t count, uint64_t elementBytes) {
        constexpr uint64_t page = 2048, fanout = page / sizeof(int32_t);
        assert(elementBytes > 0 && elementBytes <= page);
        if (count == 0) return 0;
        uint64_t perPage = 1;
        while (perPage <= page / elementBytes / 2) perPage *= 2;
        const uint64_t data = (count - 1) / perPage + 1;
        assert(data <= fanout * fanout * fanout);
        return data + (data - 1) / fanout + 1
            + (data - 1) / (fanout * fanout) + 1 + 1;
    }

    /// @brief Bound a paged arena's physical blocks and pointer directories.
    /// @details Counts page blocks, full byte-bump blocks and worst-case spilled
    /// pointer directories without taking credit for their inline capacity.
    /// @param pages Maximum simultaneously retained pages.
    /// @param bumpBlocks Exclusive full-block byte allocations.
    /// @return Whole-block physical bytes including arena metadata.
    inline uint64_t phase2HostArenaBytes(uint64_t pages, uint64_t bumpBlocks = 0) {
        constexpr uint64_t block = 262144, perBlock = block / 2048;
        const auto blocks = [](uint64_t count, uint64_t capacity) {
            return count == 0 ? 0 : (count - 1) / capacity + 1;
        };
        const uint64_t data = blocks(pages, perBlock);
        return (data + bumpBlocks + blocks(pages, block / sizeof(char*))
            + blocks(data, block / sizeof(char*))
            + blocks(bumpBlocks, block / sizeof(char*))) * block;
    }

    /// @brief Bound the two temporary dictionaries of one projection builder.
    /// @details Shared keys and one prefix table coexist. Counts include key
    /// locations, page-contained bytes, map values, prefix heads/transitions and
    /// both generations of each hash index. Key packing uses at most twice its
    /// logical bytes plus one page. Each arena is rounded independently.
    /// @param usage Frozen complete logical-block projection census.
    /// @return Conservative physical bytes for the builder's paged scratch.
    inline uint64_t phase2HostDictionaryBytes(const Phase2ProjectionUsage& usage) {
        assert(usage.byteMapEntries <= INT32_MAX && usage.byteKeyBytes <= INT32_MAX);
        const uint64_t entries = usage.byteMapEntries;
        uint64_t slots = 64;
        while (slots < 2 * entries) slots *= 2;
        const auto pages = phase2HostColumnPages;
        const uint64_t keyPages = pages(2 * usage.byteKeyBytes + 2048, 1)
            + pages(entries, sizeof(ColdStringLocation));
        const uint64_t indexPages = 2 * pages(slots, sizeof(int32_t));
        const uint64_t shared = keyPages + indexPages + pages(entries, sizeof(uint32_t));
        constexpr uint64_t transitionBytes = 3 * sizeof(int32_t)
            + ExecutionParameters::MAX_ARITY * sizeof(int16_t);
        const uint64_t prefix = keyPages + 2 * indexPages
            + pages(entries, sizeof(int32_t))
            + pages(entries, transitionBytes)
            + pages(entries, sizeof(uint64_t)) + pages(entries, sizeof(int32_t));
        return phase2HostArenaBytes(shared) + phase2HostArenaBytes(prefix);
    }

    /// @brief Measure the exact fixed chunk-directory column layout.
    /// @details The executor binds its columns in this same order.
    /// @param memories Number of source logical-block slots in the pass.
    /// @param logicalBlocks Number of projected blocks in this chunk.
    /// @param tasks Number of projected task records in this chunk.
    /// @return Aligned payload layout before block rounding.
    inline GpuHostLayout phase2HostDirectoryLayout(uint64_t memories,
        uint64_t logicalBlocks, uint64_t tasks) {
        GpuHostLayout layout;
        layout.add<char>(memories);
        layout.add<std::size_t>(logicalBlocks);
        layout.add<Memory*>(logicalBlocks);
        layout.add<uint32_t>(logicalBlocks);
        layout.add<std::size_t>(logicalBlocks);
        layout.add<int32_t>(memories);
        layout.add<std::size_t>(logicalBlocks);
        layout.add<DeviceLogicalBlockProjection>(logicalBlocks);
        layout.add<uint8_t>(logicalBlocks);
        layout.add<Memory*>(tasks);
        layout.add<uint64_t>(tasks);
        layout.add<SealedPageSet*>(logicalBlocks);
        layout.add<int64_t>(logicalBlocks);
        return layout;
    }

    /// @brief Bound request-growth resident support at the execution seam.
    /// @details Includes both cache tails, prefix measurements, five registered
    /// transfer blocks, optional disk transfer blocks and retained-call flags.
    /// Resident-only frontiers may use remaining pool capacity beyond this
    /// provision; exhaustion retains the existing assertion and never enables disk.
    /// @param capacity Planned growth columns.
    /// @param diskAllowed Existing request-frontier spill permission.
    /// @return Whole-block resident support provision including metadata.
    inline uint64_t phase2HostGrowthBytes(const Phase2GrowthCapacity& capacity,
        bool diskAllowed) {
        const uint64_t pages = 2 * phase2HostColumnPages(phase2FrontierHostCacheRows,
            sizeof(DeviceGrowthNode)) + phase2HostColumnPages(
                std::min(phase2FrontierExecutionRows, capacity.frontierRecords), sizeof(uint64_t));
        assert(capacity.calls <= 262144);
        return phase2HostArenaBytes(pages, 6 + (diskAllowed ? 2 : 0));
    }

    /// @brief Simultaneous host storage before and after construction completes.
    struct Phase2HostFootprint {
        uint64_t fixedBytes = 0;
        uint64_t constructionOneBytes = 0, constructionTwoBytes = 0;
        uint64_t executionBytes = 0;
        uint64_t oneWindowPeakBytes = 0, twoWindowPeakBytes = 0;
        uint32_t windows = 0;
    };

    /// @brief Choose overlapping or serial upload within one hard host limit.
    /// @details Zero windows is the defined request to try fewer whole logical
    /// blocks. Fixed storage and each stage's peak share the same reservation.
    /// @param footprint Measured simultaneous storage by lifetime.
    /// @param availableBytes Free capacity usable as one contiguous region.
    /// @return Complete footprint with two, one, or zero upload windows.
    inline Phase2HostFootprint selectPhase2HostWindows(Phase2HostFootprint footprint,
        uint64_t availableBytes) {
        const auto sum = [](uint64_t a, uint64_t b) {
            assert(a <= UINT64_MAX - b); return a + b;
        };
        footprint.oneWindowPeakBytes = sum(footprint.fixedBytes,
            std::max(footprint.constructionOneBytes, footprint.executionBytes));
        footprint.twoWindowPeakBytes = sum(footprint.fixedBytes,
            std::max(footprint.constructionTwoBytes, footprint.executionBytes));
        footprint.windows = footprint.twoWindowPeakBytes <= availableBytes ? 2
            : footprint.oneWindowPeakBytes <= availableBytes ? 1 : 0;
        return footprint;
    }

    /// @brief Measured column storage one projection construction window targets.
    /// @details The executor (`Phase2Executor::execute`) closes a window before
    /// the logical block that would take it past this bound, and the planner
    /// (`planPhase2HostStorage`) reproduces exactly that partition, so both read
    /// this one value. A logical block larger than the bound owns its exact
    /// window. Two windows are alive while one uploads and the next is built;
    /// 32 MiB keeps the projection step's GPU-host high-water small
    /// (D-491, amending D-447's 128 MiB).
    constexpr uint64_t kPhase2HostWindowTargetBytes = 32ull * 1024 * 1024;

    /// @brief Measure all GPU-owned chunk storage before acquiring its leases.
    /// @details Reproduces the executor's whole-block windows of
    /// `kPhase2HostWindowTargetBytes` and twelve round-robin shards. Two-window
    /// bounds include the preceding upload and
    /// first-fit gaps left by differently sized alternating windows;
    /// downloads and request backing instead coexist after construction ends.
    /// Already live pool metadata and pass directories are excluded here and
    /// subtracted by the caller. Contiguous free capacity also bounds fragmentation.
    /// @param input Fixed projection and schedule capacities.
    /// @param plan Fitting device plan whose output capacities must be downloaded.
    /// @param usages Consecutive whole-block censuses in canonical chunk order.
    /// @param memories Source logical-block slot count for chunk directories.
    /// @param availableBytes Free contiguous capacity after pass-level owners.
    /// @param diskAllowed Existing request-frontier spill permission.
    /// @return Physical footprint and selected upload window count.
    inline Phase2HostFootprint planPhase2HostStorage(const Phase2BufferPlanInput& input,
        const Phase2BufferPlan& plan, GpuHostSpan<const Phase2ProjectionUsage> usages,
        uint64_t memories, uint64_t availableBytes, bool diskAllowed) {
        assert(plan.state == Phase2BufferPlanState::fits && !usages.empty());
        constexpr uint64_t windowTarget = kPhase2HostWindowTargetBytes;
        constexpr std::size_t shards = kPhase2ProjectionConstructionShardCount;
        const auto rounded = [](const GpuHostLayout& layout) { return layout.roundedBytes(262144); };
        Phase2HostFootprint result;
        result.fixedBytes = rounded(phase2HostDirectoryLayout(memories,
            input.projection.logicalBlocks, input.tasks.tasks))
            + rounded(Phase2ProjectionArena::hostLayout(input.projection,
                input.projection, Phase2ProjectionArena::Role::Sealing))
            + rounded(Phase2TaskProjectionArena::hostLayout(input.tasks))
            + rounded(Phase2FilterScheduleArena::hostLayout(input.filter))
            + rounded(Phase2GrowthScheduleArena::hostLayout(plan.growth.calls));
        GpuHostLayout objects, threads;
        objects.add<std::optional<Phase2ProjectionArena>>(shards);
        threads.add<std::thread>(shards);
        result.fixedBytes += rounded(objects) + rounded(threads);
        GpuHostLayout windows;
        windows.add<std::optional<Phase2ProjectionArena>>(2);
        result.fixedBytes += rounded(windows);
        result.executionBytes = rounded(Phase2SealingArena::hostLayout(plan.evaluation))
            + phase2HostGrowthBytes(plan.growth, diskAllowed);
        uint64_t previousBacking = 0, previousBegin = 0;
        std::size_t begin = 0;
        while (begin < usages.size()) {
            Phase2ProjectionUsage usage{}, perShard[shards]{};
            uint64_t dictionary[shards]{};
            std::size_t end = begin;
            while (end < usages.size()) {
                auto candidate = usage;
                addPhase2HostUsage(candidate, usages[end]);
                const auto capacity = makePhase2BufferPlanInput(candidate, {}, 0, 0, 0).projection;
                if (end > begin && phase2ProjectionHostBytes(capacity) > windowTarget) break;
                usage = candidate;
                const std::size_t shard = (end - begin) % shards;
                addPhase2HostUsage(perShard[shard], usages[end]);
                dictionary[shard] = std::max(dictionary[shard], phase2HostDictionaryBytes(usages[end]));
                ++end;
            }
            Phase2ProjectionCapacity storage{};
            uint64_t scratch = 0;
            for (std::size_t shard = 0; shard < shards; ++shard) {
                const auto capacity = makePhase2BufferPlanInput(perShard[shard], {}, 0, 0, 0).projection;
                storage = addProjectionCapacity(storage, capacity);
                scratch += rounded(Phase2ProjectionArena::hostLayout(capacity,
                    capacity, Phase2ProjectionArena::Role::Slice)) + dictionary[shard];
            }
            const auto capacity = makePhase2BufferPlanInput(usage, {}, 0, 0, 0).projection;
            const uint64_t backing = rounded(Phase2ProjectionArena::hostLayout(capacity,
                storage, Phase2ProjectionArena::Role::Backing));
            result.constructionOneBytes = std::max(result.constructionOneBytes, backing + scratch);
            // The older slot is free before allocation. First fit places this
            // window in the prefix hole only when the complete range fits;
            // otherwise it follows the still-live upload. Scratch may use the
            // holes, but reserving it after both windows bounds every placement.
            const uint64_t nextBegin = backing <= previousBegin ? 0 : previousBegin + previousBacking;
            const uint64_t occupiedEnd = std::max(nextBegin + backing, previousBegin + previousBacking);
            result.constructionTwoBytes = std::max(result.constructionTwoBytes, occupiedEnd + scratch);
            previousBegin = nextBegin;
            previousBacking = backing;
            begin = end;
        }
        return selectPhase2HostWindows(result, availableBytes);
    }
}
