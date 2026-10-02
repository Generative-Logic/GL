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

#include "test_harness.hpp"
#include "../gpu/phase2_streaming.hpp"
#include "../memory_infra/paged_sorted_ordinals.hpp"

#include <numeric>
#include <vector>

TEST(phase2_streaming, automatic_budget_covers_device_sizes_and_competing_usage) {
    constexpr uint64_t gib = 1024ull * 1024 * 1024;
    for (uint64_t total : {4 * gib, 8 * gib, 24 * gib, 80 * gib}) {
        for (uint64_t free : {total, total * 3 / 4, total / 2}) {
            const auto budget = gl::gpu::makePhase2GpuBudget(free, total);
            ASSERT_TRUE(budget.bytes + budget.reserveBytes <= free);
            ASSERT_TRUE(budget.bytes <= gl::gpu::kPhase2DefaultGpuBudgetBytes);
            // Below the default ceiling the budget takes every whole block
            // after the reserve; above it, exactly the ceiling.
            if (free - budget.reserveBytes < gl::gpu::kPhase2DefaultGpuBudgetBytes)
                ASSERT_TRUE(free - budget.bytes - budget.reserveBytes
                    < gl::gpu::kPhase2GpuBlockBytes);
            else
                ASSERT_EQ(budget.bytes, gl::gpu::kPhase2DefaultGpuBudgetBytes);
            ASSERT_EQ(budget.bytes, budget.blocks * gl::gpu::kPhase2GpuBlockBytes);
            ASSERT_TRUE(budget.reserveBytes >= 256ull * 1024 * 1024);
            ASSERT_TRUE(budget.reserveBytes >= total / 20);
        }
    }
}

TEST(phase2_streaming, test_budget_never_borrows_the_runtime_reserve) {
    constexpr uint64_t gib = 1024ull * 1024 * 1024;
    const auto small = gl::gpu::makePhase2GpuBudget(7 * gib, 8 * gib, gib + 1);
    ASSERT_EQ(small.bytes, gib);
    // An explicit limit above the default ceiling is honoured up to the free
    // memory after the reserve; the automatic budget stops at the ceiling.
    const auto excessive = gl::gpu::makePhase2GpuBudget(7 * gib, 8 * gib, 8 * gib);
    const auto automatic = gl::gpu::makePhase2GpuBudget(7 * gib, 8 * gib);
    ASSERT_EQ(automatic.bytes, gl::gpu::kPhase2DefaultGpuBudgetBytes);
    ASSERT_TRUE(excessive.bytes > automatic.bytes);
    ASSERT_TRUE(excessive.bytes + excessive.reserveBytes <= 7 * gib);
    ASSERT_TRUE(7 * gib - excessive.bytes - excessive.reserveBytes
        < gl::gpu::kPhase2GpuBlockBytes);
}

TEST(phase2_streaming, allocation_lends_blocks_to_a_large_input_and_caps_inactive_columns) {
    const gl::gpu::Phase2GpuBlockDemand demands[] = {
        {13, 1, 13}, {1, 4, 100}, {1, 2, 100}, {0, 0, 0}};
    uint32_t allocation[4]{};
    ASSERT_EQ(gl::gpu::distributePhase2GpuBlocks(25, demands, 4, allocation), 25u);
    ASSERT_EQ(allocation[0], 13u);
    ASSERT_EQ(allocation[1], 8u);
    ASSERT_EQ(allocation[2], 4u);
    ASSERT_EQ(allocation[3], 0u);
    ASSERT_EQ(gl::gpu::distributePhase2GpuBlocks(15, demands, 4, allocation), 15u);
    ASSERT_EQ(allocation[1], 1u);
    ASSERT_EQ(allocation[2], 1u);
}

TEST(phase2_streaming, allocation_releases_unusable_surplus_and_uses_stable_ties) {
    const gl::gpu::Phase2GpuBlockDemand demands[] = {{1, 1, 3}, {1, 1, 3}};
    uint32_t allocation[2]{};
    ASSERT_EQ(gl::gpu::distributePhase2GpuBlocks(3, demands, 2, allocation), 3u);
    ASSERT_EQ(allocation[0], 2u);
    ASSERT_EQ(allocation[1], 1u);
    ASSERT_EQ(gl::gpu::distributePhase2GpuBlocks(10, demands, 2, allocation), 6u);
    ASSERT_EQ(allocation[0], 3u);
    ASSERT_EQ(allocation[1], 3u);
}

TEST(phase2_streaming, windows_pause_without_consuming_and_cross_32_bit_totals) {
    const uint64_t start = uint64_t{1} << 32;
    const auto paused = gl::gpu::nextPhase2WorkWindow(start, start + 19, 0);
    ASSERT_EQ(paused.begin, start);
    ASSERT_EQ(paused.count, 0u);
    uint64_t cursor = start;
    for (uint32_t expected : {7u, 7u, 5u}) {
        const auto window = gl::gpu::nextPhase2WorkWindow(cursor, start + 19, 7);
        ASSERT_EQ(window.begin, cursor);
        ASSERT_EQ(window.count, expected);
        const auto same = gl::gpu::nextPhase2WorkWindow(cursor, start + 19, 7);
        ASSERT_EQ(same.begin, window.begin);
        ASSERT_EQ(same.count, window.count);
        cursor += window.count;
    }
    ASSERT_EQ(cursor, start + 19);
    ASSERT_EQ(gl::gpu::nextPhase2WorkWindow(cursor, cursor, 7).count, 0u);
}

TEST(phase2_streaming, paged_sort_matches_oracle_across_many_runs_and_replays) {
    gl::LbArena arena{ &gl::staticMemory() };
    std::vector<int32_t> values(1301);
    for (int32_t index = 0; index < static_cast<int32_t>(values.size()); ++index)
        values[index] = (index * 73 + 19) % 101;
    const auto less = [&values](int32_t left, int32_t right) {
        return values[left] != values[right]
            ? values[left] < values[right] : left < right;
    };
    std::vector<int32_t> expected(values.size());
    std::iota(expected.begin(), expected.end(), 0);
    std::sort(expected.begin(), expected.end(), less);
    for (int32_t runSlots : {1, 3, 17, 512, 2048}) {
        gl::PagedSortedOrdinals index(arena,
            static_cast<int32_t>(values.size()), less, runSlots);
        for (int32_t ordinal : expected) ASSERT_EQ(index.next(), ordinal);
        index.reset();
        for (int32_t ordinal : expected) ASSERT_EQ(index.next(), ordinal);
    }
    gl::PagedSortedOrdinals empty(arena, 0, less, 1);
    empty.reset();
}

TEST(phase2_streaming, output_prefix_reserves_every_column_without_consuming_blocked_work) {
    constexpr uint32_t columns = static_cast<uint32_t>(gl::gpu::Phase2OutputColumn::count);
    for (uint32_t limiting = 0; limiting < columns; ++limiting) {
        gl::gpu::Phase2OutputUsage capacity{}, occupied{}, demands[3]{};
        for (uint32_t column = 0; column < columns; ++column) {
            capacity.columns[column] = 20;
            occupied.columns[column] = 1;
            for (auto& demand : demands) demand.columns[column] = 2;
        }
        capacity.columns[limiting] = 5;
        ASSERT_EQ(gl::gpu::boundedPhase2OutputPrefix(demands, 3, capacity, occupied), 2u);
        ASSERT_EQ(occupied.columns[limiting], 5ull);
        const auto before = occupied;
        ASSERT_EQ(gl::gpu::boundedPhase2OutputPrefix(demands + 2, 1, capacity, occupied), 0u);
        for (uint32_t column = 0; column < columns; ++column)
            ASSERT_EQ(occupied.columns[column], before.columns[column]);
        occupied = {};
        ASSERT_EQ(gl::gpu::boundedPhase2OutputPrefix(demands + 2, 1, capacity, occupied), 1u);
    }
}

TEST(phase2_streaming, output_prefix_handles_empty_and_individually_segmented_records) {
    gl::gpu::Phase2OutputUsage capacity{}, occupied{}, demand{};
    ASSERT_EQ(gl::gpu::boundedPhase2OutputPrefix(nullptr, 0, capacity, occupied), 0u);
    const auto bytes = static_cast<uint32_t>(gl::gpu::Phase2OutputColumn::bytes);
    capacity.columns[bytes] = uint64_t{1} << 32;
    demand.columns[bytes] = capacity.columns[bytes] + 1;
    ASSERT_EQ(gl::gpu::boundedPhase2OutputPrefix(&demand, 1, capacity, occupied), 0u);
    ASSERT_EQ(occupied.columns[bytes], 0ull);
    demand.columns[bytes] = capacity.columns[bytes];
    ASSERT_EQ(gl::gpu::boundedPhase2OutputPrefix(&demand, 1, capacity, occupied), 1u);
    ASSERT_EQ(occupied.columns[bytes], uint64_t{1} << 32);
}
