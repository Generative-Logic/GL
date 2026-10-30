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

/// @file
/// @brief Unit tests for the operating-system address-space primitives behind
///        the static pools, and for the pools' commit-on-carve.
///
/// @details
/// Covers `gl::reserveAddressSpace`, `gl::commitAddressSpace` and
/// `gl::releaseAddressSpace` directly (alignment, zero-filled commits, page
/// widening, content kept across a re-commit), and the `GlobalMemoryManager`
/// contract that a block commits before use and returned blocks can be trimmed
/// without changing live content or recycle order. On Windows the reserved /
/// committed / freed page states are read back with `VirtualQuery`; elsewhere
/// the tests check accessibility and content only.

#include "test_harness.hpp"

#include "../memory_infra/address_space.hpp"
#include "../memory_infra/global_memory_manager.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace {
    // The Windows state (MEM_RESERVE / MEM_COMMIT / MEM_FREE) of the page
    // holding `address`.
    DWORD pageState(const void* address) {
        MEMORY_BASIC_INFORMATION info{};
        const SIZE_T got = VirtualQuery(address, &info, sizeof(info));
        assert(got == sizeof(info));
        return info.State;
    }
}
#endif

TEST(address_space, reserve_is_aligned_uncommitted_and_released) {
    constexpr int64_t bytes = 1 << 20;
    char* const base = gl::reserveAddressSpace(bytes);
    ASSERT_TRUE(base != nullptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(base) % 4096, uintptr_t(0));
#ifdef _WIN32
    ASSERT_EQ(pageState(base), DWORD(MEM_RESERVE));
    ASSERT_EQ(pageState(base + bytes - 1), DWORD(MEM_RESERVE));
#endif
    gl::releaseAddressSpace(base, bytes);
#ifdef _WIN32
    ASSERT_EQ(pageState(base), DWORD(MEM_FREE));
#endif
}

TEST(address_space, commit_reads_zero_and_leaves_neighbours_reserved) {
    constexpr int64_t bytes = 1 << 20;
    constexpr int64_t block = 1 << 18;
    char* const base = gl::reserveAddressSpace(bytes);
    char* const second = base + block;
    gl::commitAddressSpace(second, block);
    for (int64_t i = 0; i < block; i += 4096) ASSERT_EQ(second[i], char(0));
    ASSERT_EQ(second[block - 1], char(0));
    std::memset(second, 0x5A, static_cast<std::size_t>(block));
    ASSERT_EQ(second[0], char(0x5A));
    ASSERT_EQ(second[block - 1], char(0x5A));
#ifdef _WIN32
    ASSERT_EQ(pageState(second), DWORD(MEM_COMMIT));
    ASSERT_EQ(pageState(second + block - 1), DWORD(MEM_COMMIT));
    ASSERT_EQ(pageState(base), DWORD(MEM_RESERVE));
    ASSERT_EQ(pageState(second + block), DWORD(MEM_RESERVE));
#endif
    gl::releaseAddressSpace(base, bytes);
}

TEST(address_space, commit_widens_to_whole_pages_and_recommit_keeps_content) {
    constexpr int64_t bytes = 1 << 16;
    char* const base = gl::reserveAddressSpace(bytes);
    // A sub-page range commits the whole page holding it.
    gl::commitAddressSpace(base + 100, 50);
    base[0] = 'a';
    base[4095] = 'z';
    // Committing a committed page again leaves its content unchanged.
    gl::commitAddressSpace(base, 4096);
    ASSERT_EQ(base[0], 'a');
    ASSERT_EQ(base[4095], 'z');
#ifdef _WIN32
    ASSERT_EQ(pageState(base), DWORD(MEM_COMMIT));
    ASSERT_EQ(pageState(base + 4096), DWORD(MEM_RESERVE));
#endif
    gl::releaseAddressSpace(base, bytes);
}

TEST(static_memory, pool_commits_each_block_when_first_carved) {
    // 1 MiB pool / 256 KiB blocks. Carving runs upward from the base, so the
    // first grant is the base block and the next fresh carve its neighbour.
    constexpr int32_t block = 1 << 18;
    gl::GlobalMemoryManager m;
    m.init(gl::StaticMemoryConfig{ 1 << 20, block, 1 << 11 });
    char* const first = m.acquireBlock();
#ifdef _WIN32
    ASSERT_EQ(pageState(first), DWORD(MEM_COMMIT));
    ASSERT_EQ(pageState(first + block - 1), DWORD(MEM_COMMIT));
    ASSERT_EQ(pageState(first + block), DWORD(MEM_RESERVE));
#endif
    std::memset(first, 0x3C, static_cast<std::size_t>(block));
    m.releaseBlock(first);
    char* const recycled = m.acquireBlock();
    ASSERT_EQ(recycled, first);                    // recycled before a fresh carve
    ASSERT_EQ(recycled[block - 1], char(0x3C));    // stayed committed, content kept
    ASSERT_EQ(m.peakBlocksInUse(), int64_t(1));    // commit = carve high-water
    char* const second = m.acquireBlock();
    ASSERT_EQ(second, first + block);
    ASSERT_EQ(second[0], char(0));
#ifdef _WIN32
    ASSERT_EQ(pageState(second), DWORD(MEM_COMMIT));
    ASSERT_EQ(pageState(second + block), DWORD(MEM_RESERVE));
#endif
    ASSERT_EQ(m.peakBlocksInUse(), int64_t(2));
    m.releaseBlock(recycled);
    m.releaseBlock(second);
}

TEST(address_space, decommit_recommit_clears_only_the_selected_range) {
    constexpr int64_t block = 1 << 18;
    char* const base = gl::reserveAddressSpace(3 * block);
    gl::commitAddressSpace(base, 3 * block);
    std::memset(base, 0x5A, static_cast<std::size_t>(3 * block));
    gl::decommitAddressSpace(base + block, block);
    ASSERT_EQ(base[block - 1], char(0x5A));
    ASSERT_EQ(base[2 * block], char(0x5A));
#ifdef _WIN32
    ASSERT_EQ(pageState(base + block), DWORD(MEM_RESERVE));
    ASSERT_EQ(pageState(base), DWORD(MEM_COMMIT));
    ASSERT_EQ(pageState(base + 2 * block), DWORD(MEM_COMMIT));
#endif
    gl::commitAddressSpace(base + block, block);
    for (int64_t i = block; i < 2 * block; ++i) ASSERT_EQ(base[i], char(0));
    gl::releaseAddressSpace(base, 3 * block);
}

TEST(static_memory, trim_preserves_warm_reserve_live_content_and_recycle_order) {
    constexpr int32_t block = 1 << 18;
    gl::GlobalMemoryManager m;
    ASSERT_EQ(m.commitment().currentBytes, int64_t(0));
    m.init({8LL * block, block, 2048});
    char* rows[6];
    m.acquireBlocks(6, rows);
    for (int i = 0; i < 6; ++i) std::memset(rows[i], i + 1, block);
    // Keep row 2 live; release the rest in a non-address order.
    char* returned[] = {rows[4], rows[0], rows[3], rows[1], rows[5]};
    m.releaseBlocks(returned, 5);
    const auto grants = m.grantsSinceBarrier();
    ASSERT_EQ(m.trimRecycledBlocks(2LL * block), int64_t(3LL * block));
    ASSERT_EQ(m.blocksInUse(), int64_t(1));
    ASSERT_EQ(m.grantsSinceBarrier(), grants);
    ASSERT_EQ(m.commitment().currentBytes, int64_t(3LL * block));
    ASSERT_EQ(m.commitment().peakBytes, int64_t(6LL * block));
    ASSERT_EQ(m.trimRecycledBlocks(2LL * block), int64_t(0));
    ASSERT_EQ(rows[2][block - 1], char(3));
#ifdef _WIN32
    ASSERT_EQ(pageState(rows[2]), DWORD(MEM_COMMIT));
    ASSERT_EQ(pageState(rows[4]), DWORD(MEM_COMMIT));
    ASSERT_EQ(pageState(rows[0]), DWORD(MEM_COMMIT));
    ASSERT_EQ(pageState(rows[3]), DWORD(MEM_RESERVE));
#endif
    char* reused[5];
    m.acquireBlocks(5, reused);
    for (int i = 0; i < 5; ++i) ASSERT_EQ(reused[i], returned[i]);
    ASSERT_EQ(reused[0][block - 1], char(5));
    ASSERT_EQ(reused[1][block - 1], char(1));
    for (int i = 2; i < 5; ++i) ASSERT_EQ(reused[i][block - 1], char(0));
    ASSERT_EQ(m.commitment().recommittedBytes, int64_t(3LL * block));
    ASSERT_EQ(m.commitment().decommittedBytes, int64_t(3LL * block));
    ASSERT_EQ(m.commitment().currentBytes, int64_t(6LL * block));
    m.releaseBlocks(reused, 5); m.releaseBlock(rows[2]);
    ASSERT_EQ(m.trimRecycledBlocks(0), int64_t(6LL * block));
    ASSERT_EQ(m.commitment().currentBytes, int64_t(0));
    ASSERT_EQ(m.peakBlocksInUse(), int64_t(6));
}

TEST(static_memory, trim_zero_empty_and_last_carved_recommit) {
    constexpr int32_t block = 1 << 18;
    gl::GlobalMemoryManager m; m.init({2LL * block, block, 2048});
    ASSERT_EQ(m.trimRecycledBlocks(0), int64_t(0));
    char* const first = m.acquireBlock(); first[0] = 'x';
    ASSERT_EQ(m.trimRecycledBlocks(0), int64_t(0));
    ASSERT_EQ(first[0], 'x');
    m.releaseBlock(first);
    ASSERT_EQ(m.trimRecycledBlocks(block), int64_t(0));
    ASSERT_EQ(m.trimRecycledBlocks(0), int64_t(block));
    char* const same = m.acquireBlock();
    ASSERT_EQ(same, first); ASSERT_EQ(same[0], char(0));
    ASSERT_EQ(m.commitment().recommittedBytes, int64_t(block));
    ASSERT_EQ(m.commitment().trimCalls, int64_t(4));
    m.releaseBlock(same);
}

TEST(address_space, invalid_decommit_assertion_probe) {
    const auto mode = gl::tests::testEnvironment("GL_TEST_DECOMMIT_ASSERT");
    if (!mode) return;
    constexpr int64_t bytes = 1 << 18;
    char* const base = gl::reserveAddressSpace(bytes);
    gl::commitAddressSpace(base, bytes);
    if (*mode == "address") gl::decommitAddressSpace(base + 1, bytes - 1);
    else if (*mode == "size") gl::decommitAddressSpace(base, bytes - 1);
    else if (*mode == "reserve") {
        gl::GlobalMemoryManager m; m.init({2 * bytes, static_cast<int32_t>(bytes), 2048});
        m.trimRecycledBlocks(1);
    } else ASSERT_TRUE(false);
    gl::releaseAddressSpace(base, bytes);
}


TEST(static_memory, concurrent_trim_never_discards_a_live_block) {
    constexpr int32_t block = 1 << 18;
    gl::GlobalMemoryManager m; m.init({16LL * block, block, 2048});
    std::atomic<bool> start{false};
    std::thread workers[4];
    for (int worker = 0; worker < 4; ++worker)
        workers[worker] = std::thread([&, worker] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int i = 0; i < 128; ++i) {
                char* const value = m.acquireBlock();
                const char expected = static_cast<char>(worker * 32 + i % 32);
                value[0] = expected; value[block - 1] = expected;
                std::this_thread::yield();
                ASSERT_EQ(value[0], expected); ASSERT_EQ(value[block - 1], expected);
                m.releaseBlock(value);
            }
        });
    start.store(true, std::memory_order_release);
    for (int i = 0; i < 256; ++i) { m.trimRecycledBlocks(0); std::this_thread::yield(); }
    for (auto& worker : workers) worker.join();
    ASSERT_EQ(m.blocksInUse(), int64_t(0));
    m.trimRecycledBlocks(0);
    ASSERT_EQ(m.commitment().currentBytes, int64_t(0));
}
