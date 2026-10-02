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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>

namespace gl::gpu {

    /// @brief Physical allocation granularity shared by all streaming columns.
    inline constexpr uint64_t kPhase2GpuBlockBytes = 2ull * 1024 * 1024;

    /// @brief Default ceiling of the GPU working budget.
    ///
    /// @details
    /// Every chunk's plan reserves the whole budget and sizes its growth,
    /// ordering and evaluation scratch to fill it, so an uncapped budget takes
    /// nearly all free VRAM (6.5 GiB on an 8 GiB card) and the matching host
    /// commit, whatever the chunk needs. Measured on the full `--GPU` run
    /// (D-412): at 4 GiB every batch keeps its chunk and drain
    /// counts and its prover time, and each batch process commits 2.5 GiB less;
    /// 3 GiB still passes but multiplies the evaluation download windows.
    /// `GL_GPU_BUDGET_MIB` overrides it.
    inline constexpr uint64_t kPhase2DefaultGpuBudgetBytes = 4ull * 1024 * 1024 * 1024;

    /// @brief Available GPU budget after the runtime reserve and block rounding.
    struct Phase2GpuBudget {
        uint64_t reserveBytes{ 0 };
        uint64_t bytes{ 0 };
        uint32_t blocks{ 0 };
    };

    /// @brief Compute a card-independent GPU working budget.
    ///
    /// @details
    /// Keeps the larger of 256 MiB or five percent of total device memory free.
    /// The working storage after that reserve is capped by `limitBytes`, or by
    /// `kPhase2DefaultGpuBudgetBytes` when no limit is given; neither ever
    /// permits consuming the reserve. Rounding releases the incomplete block.
    ///
    /// @param freeBytes Free memory measured after CUDA initialization.
    /// @param totalBytes Total device memory from the same measurement.
    /// @param limitBytes Optional ceiling (`GL_GPU_BUDGET_MIB`), zero for the
    ///                   default ceiling.
    /// @return Whole-block working budget and the separate runtime reserve.
    /// @invariant The working allocation plus reserve fits measured free memory.
    inline Phase2GpuBudget makePhase2GpuBudget(uint64_t freeBytes,
        uint64_t totalBytes, uint64_t limitBytes = 0) {
        assert(totalBytes > 0 && freeBytes <= totalBytes);
        Phase2GpuBudget result{};
        result.reserveBytes = std::max(256ull * 1024 * 1024,
            static_cast<unsigned long long>(totalBytes / 20));
        assert(freeBytes > result.reserveBytes
            && "GPU memory cannot accommodate the runtime reserve");
        uint64_t available = freeBytes - result.reserveBytes;
        available = std::min(available,
            limitBytes != 0 ? limitBytes : kPhase2DefaultGpuBudgetBytes);
        const uint64_t blocks = available / kPhase2GpuBlockBytes;
        assert(blocks > 0 && blocks <= std::numeric_limits<uint32_t>::max());
        result.blocks = static_cast<uint32_t>(blocks);
        result.bytes = blocks * kPhase2GpuBlockBytes;
        return result;
    }

    /// @brief One column's minimum progress unit, demand weight and useful ceiling.
    ///
    /// @details
    /// All fields count physical blocks. Zero maximum means an inactive column;
    /// an active column has a positive weight and minimum. Input projection
    /// minima cover the complete selected LB image. Processing minima cover one
    /// bounded step, while weights reflect measured occupied byte demand.
    struct Phase2GpuBlockDemand {
        uint32_t minimum{ 0 };
        uint32_t weight{ 0 };
        uint32_t maximum{ 0 };
    };

    /// @brief Divide one GPU budget between competing bounded columns.
    ///
    /// @details
    /// Assigns every minimum first, then lends spare blocks to the column with
    /// the smallest next-block allocation divided by its demand weight. Exact
    /// integer cross multiplication and column-order ties make the plan stable.
    /// Maximum ceilings prevent reserving storage a bounded input cannot use.
    /// No allocation, pointer movement or proof decision occurs in this planner.
    ///
    /// @param budgetBlocks Total blocks available at a quiescent boundary.
    /// @param demands Column requirements in stable descriptor order.
    /// @param count Number of descriptors and output entries.
    /// @param allocations Caller-owned output block counts.
    /// @return Total blocks assigned; spare blocks remain in the common owner.
    /// @invariant Every output fits its bounds and their sum fits the budget.
    inline uint32_t distributePhase2GpuBlocks(uint32_t budgetBlocks,
        const Phase2GpuBlockDemand* demands, uint32_t count,
        uint32_t* allocations) {
        assert(budgetBlocks > 0 && count > 0);
        assert(demands != nullptr && allocations != nullptr);
        uint64_t used = 0;
        for (uint32_t index = 0; index < count; ++index) {
            const Phase2GpuBlockDemand& demand = demands[index];
            assert(demand.minimum <= demand.maximum);
            assert(demand.maximum == 0
                ? demand.minimum == 0 && demand.weight == 0
                : demand.minimum > 0 && demand.weight > 0);
            allocations[index] = demand.minimum;
            used += demand.minimum;
        }
        assert(used <= budgetBlocks
            && "one LB input and minimum execution scratch must fit GPU memory");
        while (used < budgetBlocks) {
            uint32_t best = count;
            for (uint32_t index = 0; index < count; ++index) {
                if (allocations[index] == demands[index].maximum) continue;
                if (best == count
                    || (static_cast<uint64_t>(allocations[index]) + 1)
                           * demands[best].weight
                        < (static_cast<uint64_t>(allocations[best]) + 1)
                           * demands[index].weight) best = index;
            }
            if (best == count) break;
            ++allocations[best];
            ++used;
        }
        return static_cast<uint32_t>(used);
    }

    /// @brief One capacity-bounded portion of an immutable work stream.
    struct Phase2WorkWindow {
        uint64_t begin{ 0 };
        uint32_t count{ 0 };
    };

    /// @brief Select the next portion without consuming its continuation.
    ///
    /// @details
    /// A zero count is the defined full-buffer or completed-stream state. The
    /// caller advances its cursor only after output ownership is secured, so
    /// a transfer or a downstream pause cannot lose the current portion.
    ///
    /// @param cursor Number of items whose processing is already committed.
    /// @param total Number of items in this immutable stream.
    /// @param available Free entries in the destination processing buffer.
    /// @return Consecutive half-open work window starting at cursor.
    /// @invariant Selection never advances the cursor or crosses either bound.
    inline Phase2WorkWindow nextPhase2WorkWindow(uint64_t cursor,
        uint64_t total, uint32_t available) {
        assert(cursor <= total);
        return { cursor, static_cast<uint32_t>(std::min(
            total - cursor, static_cast<uint64_t>(available))) };
    }

    /// @brief Independently bounded columns of one provenance-bearing output.
    enum class Phase2OutputColumn : uint32_t {
        records, bytes, levels, dependencies, markerKeys, markerRemaining,
        markerArguments, count
    };

    /// @brief Wide counts or capacities for every output column.
    struct Phase2OutputUsage {
        uint64_t columns[static_cast<uint32_t>(Phase2OutputColumn::count)]{};
    };

    /// @brief Select the longest consecutive output prefix fitting every column.
    ///
    /// @details
    /// The input contains device-measured output bounds. Selection performs
    /// storage planning only and does not evaluate or reject proof work. A zero
    /// prefix means the current output must drain or its next record must use
    /// segmented emission. The caller leaves that record's cursor unchanged.
    ///
    /// @param demands Per-record bounds in the unchanged work-window order.
    /// @param count Number of supplied records; zero is an empty window.
    /// @param capacity Allocated capacities for all output columns.
    /// @param occupied Counts already retained in the current output buffer;
    ///                 increased only by the selected prefix's requirements.
    /// @return Number of consecutive records whose reservations all fit.
    /// @invariant No selected record crosses any column's capacity.
    inline uint32_t boundedPhase2OutputPrefix(const Phase2OutputUsage* demands,
        uint32_t count, const Phase2OutputUsage& capacity,
        Phase2OutputUsage& occupied) {
        assert(count == 0 || demands != nullptr);
        constexpr uint32_t columns = static_cast<uint32_t>(Phase2OutputColumn::count);
        for (uint32_t column = 0; column < columns; ++column)
            assert(occupied.columns[column] <= capacity.columns[column]);
        uint32_t selected = 0;
        while (selected < count) {
            bool fits = true;
            for (uint32_t column = 0; column < columns; ++column) {
                if (demands[selected].columns[column]
                    > capacity.columns[column] - occupied.columns[column]) {
                    fits = false;
                    break;
                }
            }
            if (!fits) break;
            for (uint32_t column = 0; column < columns; ++column)
                occupied.columns[column] += demands[selected].columns[column];
            ++selected;
        }
        return selected;
    }

}
