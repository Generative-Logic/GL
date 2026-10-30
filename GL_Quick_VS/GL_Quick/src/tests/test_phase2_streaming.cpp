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

#include "request_frontier_fixture.hpp"
#include <cstring>
#include <cstdlib>
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

TEST(phase2_streaming, budget_grants_dedicated_frontier_store_only_from_spare_memory) {
    constexpr uint64_t mib = 1024ull * 1024;
    constexpr uint64_t gib = 1024 * mib;
    const uint64_t want = 896 * mib;
    // Plenty of spare memory: the whole dedicated store beside a full working budget.
    const auto roomy = gl::gpu::makePhase2GpuBudget(7 * gib, 8 * gib, 0, want);
    ASSERT_EQ(roomy.bytes, gl::gpu::kPhase2DefaultGpuBudgetBytes);
    ASSERT_EQ(roomy.frontierStoreBlocks, static_cast<uint32_t>(want / gl::gpu::kPhase2GpuBlockBytes));
    ASSERT_EQ(roomy.totalBlocks(), roomy.blocks + roomy.frontierStoreBlocks);
    // Part of it: only the whole blocks left after the working budget and reserve.
    const uint64_t tightFree = 4 * gib + 512 * mib + 300 * mib;
    const auto tight = gl::gpu::makePhase2GpuBudget(tightFree, 8 * gib, 0, want);
    ASSERT_EQ(tight.bytes, gl::gpu::kPhase2DefaultGpuBudgetBytes);
    ASSERT_TRUE(tight.frontierStoreBlocks > 0);
    ASSERT_TRUE(tight.frontierStoreBlocks < roomy.frontierStoreBlocks);
    ASSERT_TRUE(tightFree - tight.reserveBytes - tight.bytes
        - static_cast<uint64_t>(tight.frontierStoreBlocks) * gl::gpu::kPhase2GpuBlockBytes
        < gl::gpu::kPhase2GpuBlockBytes);
    // None when the working budget already takes everything after the reserve,
    // and none when nothing is requested.
    const auto full = gl::gpu::makePhase2GpuBudget(2 * gib, 8 * gib, 0, want);
    ASSERT_EQ(full.frontierStoreBlocks, 0u);
    const auto unrequested = gl::gpu::makePhase2GpuBudget(7 * gib, 8 * gib);
    ASSERT_EQ(unrequested.frontierStoreBlocks, 0u);
    ASSERT_EQ(unrequested.blocks, roomy.blocks);
    for (const auto& budget : {roomy, tight, full}) {
        ASSERT_TRUE(static_cast<uint64_t>(budget.totalBlocks()) * gl::gpu::kPhase2GpuBlockBytes
            + budget.reserveBytes <= 7 * gib);
    }
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


TEST(request_frontier, exact_round_trip_pruning_and_reset_across_storage_boundaries) {
    gl::tests::RequestFrontierTestFiles files("request_frontier_store_tests");
    for (bool disk : {false, true}) {
        for (int32_t budget : {1, 3, 33, 129}) {
            gl::LbArena arena(&gl::staticMemory());
            gl::gpu::RequestFrontierStore frontier(arena, disk ? &files.files[0] : nullptr, budget);
            gl::gpu::DeviceGrowthNode scratch[17]{};
            ASSERT_TRUE(frontier.empty());
            frontier.appendRun(nullptr, 0);
            frontier.copyRunTo(0, 0, nullptr);
            const uint8_t all[3] = {1, 1, 1};
            ASSERT_EQ(frontier.pruneByCall(all, 3, scratch, 17), 0ull);
            for (int cycle = 0; cycle < 4; ++cycle) {
                std::vector<gl::gpu::DeviceGrowthNode> expected;
                for (int i = 0; i < 521; ++i) {
                    gl::gpu::DeviceGrowthNode row{};
                    row.callIndex = i % 3;
                    row.runOrdinal = cycle * 1000 + i;
                    row.filteredPositions[0] = i * 7;
                    row.validityId = i % 11;
                    expected.push_back(row);
                }
                // Cross several cache boundaries in a single append.
                frontier.appendRun(expected.data(), static_cast<int32_t>(expected.size()));
                ASSERT_EQ(frontier.size(), static_cast<uint64_t>(expected.size()));
                for (int32_t begin = 0; begin < static_cast<int32_t>(expected.size()); begin += 17) {
                    const int32_t count = std::min(17, static_cast<int32_t>(expected.size()) - begin);
                    frontier.copyRunTo(begin, count, scratch);
                    ASSERT_EQ(std::memcmp(scratch, expected.data() + begin, count * sizeof(scratch[0])), 0);
                }
                const uint8_t keep[3] = {
                    static_cast<uint8_t>(cycle != 2),
                    static_cast<uint8_t>(cycle == 0),
                    static_cast<uint8_t>(cycle != 2)};
                std::vector<gl::gpu::DeviceGrowthNode> kept;
                for (const auto& row : expected) if (keep[row.callIndex] != 0) kept.push_back(row);
                ASSERT_EQ(frontier.pruneByCall(keep, 3, scratch, 17),
                    static_cast<uint64_t>(expected.size() - kept.size()));
                ASSERT_EQ(frontier.size(), static_cast<uint64_t>(kept.size()));
                for (std::size_t i = 0; i < kept.size(); ++i) {
                    frontier.copyRunTo(i, 1, scratch);
                    ASSERT_EQ(std::memcmp(scratch, &kept[i], sizeof(scratch[0])), 0);
                }
                // Appending after disk compaction keeps the compacted prefix.
                frontier.appendRun(expected.data(), 2);
                frontier.copyRunTo(kept.size(), 2, scratch);
                ASSERT_EQ(std::memcmp(scratch, expected.data(), 2 * sizeof(scratch[0])), 0);
                frontier.resetKeepingPages();
                ASSERT_TRUE(frontier.empty());
            }
            const auto& stats = frontier.stats();
            if (disk) {
                ASSERT_TRUE(stats.writes > 0 && stats.reads > 0);
                ASSERT_TRUE(stats.writtenBytes > 0 && stats.readBytes > 0);
                ASSERT_TRUE(stats.peakResidentPayloadBytes <= static_cast<uint64_t>(budget) * sizeof(scratch[0]));
                ASSERT_TRUE(stats.peakResidentStorageBytes >= stats.peakResidentPayloadBytes);
            } else {
                ASSERT_EQ(stats.writes, 0ull);
                ASSERT_EQ(stats.reads, 0ull);
                ASSERT_TRUE(stats.peakResidentPayloadBytes > static_cast<uint64_t>(budget) * sizeof(scratch[0]));
            }
        }
    }
}

TEST(request_frontier, independent_files_and_no_spill_below_limit) {
    gl::tests::RequestFrontierTestFiles files("request_frontier_independent_tests");
    gl::LbArena arena(&gl::staticMemory());
    gl::gpu::RequestFrontierStore first(arena, &files.files[0], 3);
    gl::gpu::RequestFrontierStore second(arena, &files.files[1], 3);
    gl::gpu::DeviceGrowthNode rows[4]{}, read[4]{};
    for (uint32_t i = 0; i < 4; ++i) rows[i].runOrdinal = i;
    first.appendRun(rows, 3);
    ASSERT_EQ(first.stats().writes, 0ull);
    first.appendRun(rows + 3, 1);
    for (auto& row : rows) row.runOrdinal += 10;
    second.appendRun(rows, 4);
    first.copyRunTo(0, 4, read);
    for (uint32_t i = 0; i < 4; ++i) ASSERT_EQ(read[i].runOrdinal, i);
    second.copyRunTo(0, 4, read);
    for (uint32_t i = 0; i < 4; ++i) ASSERT_EQ(read[i].runOrdinal, i + 10);
    first.resetKeepingPages();
    second.copyRunTo(0, 4, read);
    ASSERT_EQ(read[3].runOrdinal, 13u);
}

TEST(request_frontier, invalid_operation_probe) {
    std::string mode;
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    ASSERT_EQ(_dupenv_s(&value, &length, "GL_TEST_REQUEST_FRONTIER_INVALID"), 0);
    if (value != nullptr) { mode = value; std::free(value); }
#else
    if (const char* value = std::getenv("GL_TEST_REQUEST_FRONTIER_INVALID")) mode = value;
#endif
    if (mode.empty()) return;
    gl::tests::RequestFrontierTestFiles files("request_frontier_invalid_tests");
    gl::LbArena arena(&gl::staticMemory());
    gl::gpu::DeviceGrowthNode rows[4]{};
    if (mode == "budget") {
        gl::gpu::RequestFrontierStore invalid(arena, &files.files[0], 0);
    } else {
        gl::gpu::RequestFrontierStore frontier(arena, &files.files[0], 1);
        frontier.appendRun(rows, 4);
        if (mode == "range") frontier.copyRunTo(4, 1, rows);
        else if (mode == "closed") {
            files.files[0].close();
            frontier.copyRunTo(0, 1, rows);
        } else if (mode == "short_read") {
            files.files[0].preallocate(0);
            frontier.copyRunTo(0, 1, rows);
        } else if (mode == "write") {
            files.files[0].close();
            frontier.appendRun(rows, 1);
        } else if (mode == "call") {
            const uint8_t keep[] = {2};
            frontier.pruneByCall(keep, 1, rows, 4);
        } else ASSERT_TRUE(false);
    }
    ASSERT_TRUE(false);
}
