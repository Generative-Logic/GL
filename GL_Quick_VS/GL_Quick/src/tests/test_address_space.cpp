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
/// contract that a block is committed when first carved and stays committed
/// while recycled (D-418). On Windows the reserved /
/// committed / freed page states are read back with `VirtualQuery`; elsewhere
/// the tests check accessibility and content only.

#include "test_harness.hpp"

#include "../memory_infra/address_space.hpp"
#include "../memory_infra/global_memory_manager.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>

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
