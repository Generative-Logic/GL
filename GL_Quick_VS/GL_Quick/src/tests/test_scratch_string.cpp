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
/// @brief Unit tests for the ScratchString view type and the ScratchScope RAII
///        rewind guard (string statification, scratch / cold half).
///
/// @details
/// Covers `ScratchString::copyFrom` / `wrap` construction, accessor semantics,
/// the empty-string contract (no arena coupling, always valid), the
/// `toStdString` boundary escape, and `ScratchScope`'s mark-at-construction /
/// rewind-at-destruction behavior including the keep-below-the-mark
/// pattern. The escape ASSERTS (generation mismatch, rewound-past-birth)
/// abort by design and are exercised by the poison checks on the raw spans
/// instead — the harness has no death tests.

#include "test_harness.hpp"

#include "../memory_infra/scratch_string.hpp"

#include <cstring>
#include <string>

namespace {
    // 1 MiB pool / 256 KiB block.
    const gl::StaticMemoryConfig kHotStrTestCfg{ 1 << 20, 1 << 18 };
}

TEST(hot_string, copy_from_and_accessors) {
    gl::GlobalMemoryManager m;
    m.init(kHotStrTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    const char* src = "(in[(s[a]),N])";
    const int32_t len = static_cast<int32_t>(std::strlen(src));
    const gl::ScratchString s = gl::ScratchString::copyFrom(a, src, len);
    ASSERT_FALSE(s.empty());
    ASSERT_EQ(s.size(), len);
    ASSERT_EQ(std::memcmp(s.data(), src, static_cast<size_t>(len)), 0);
    ASSERT_EQ(s[0], '(');
    ASSERT_EQ(s[len - 1], ')');
    // The copy is independent of the source.
    ASSERT_NE(s.data(), src);
}

TEST(hot_string, empty_string_has_no_arena_coupling) {
    const gl::ScratchString def;
    ASSERT_TRUE(def.empty());
    ASSERT_EQ(def.size(), 0);
    ASSERT_EQ(def.toStdString(), std::string());

    gl::GlobalMemoryManager m;
    m.init(kHotStrTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    const gl::ScratchString viaCopy = gl::ScratchString::copyFrom(a, "x", 0);
    ASSERT_TRUE(viaCopy.empty());
    // No allocation happened for the empty result.
    ASSERT_EQ(a.usedBytes(), static_cast<int64_t>(0));
    // Empty views stay valid across resets — no liveness coupling.
    a.allocBytes(8);
    a.reset();
    ASSERT_TRUE(viaCopy.empty());
    ASSERT_EQ(viaCopy.toStdString(), std::string());
}

TEST(hot_string, wrap_after_fill_then_wrap_pattern) {
    gl::GlobalMemoryManager m;
    m.init(kHotStrTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    // The string-operation pattern: compute output length, allocate once,
    // fill, wrap.
    const std::string left = "(=[";
    const std::string right = "a,b])";
    const int32_t outLen = static_cast<int32_t>(left.size() + right.size());
    char* buf = a.allocBytes(outLen);
    std::memcpy(buf, left.data(), left.size());
    std::memcpy(buf + left.size(), right.data(), right.size());
    const gl::ScratchString s = gl::ScratchString::wrap(a, buf, outLen);
    ASSERT_EQ(s.size(), outLen);
    ASSERT_EQ(s.toStdString(), std::string("(=[a,b])"));
    ASSERT_EQ(s.data(), buf);
}

TEST(hot_string, to_std_string_round_trip) {
    gl::GlobalMemoryManager m;
    m.init(kHotStrTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    const std::string original =
        "(>[i0,s,+,i1](AnchorPeano[N,i0,s,+,*,i1]))";
    const gl::ScratchString s = gl::ScratchString::copyFrom(
        a, original.data(), static_cast<int32_t>(original.size()));
    ASSERT_EQ(s.toStdString(), original);
}

TEST(hot_string, hot_scope_rewinds_and_poisons_inner_allocations) {
    gl::GlobalMemoryManager m;
    m.init(kHotStrTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    const char* raw = nullptr;
    const int64_t before = a.usedBytes();
    {
        gl::ScratchScope scope(a);
        const gl::ScratchString inner =
            gl::ScratchString::copyFrom(a, "scratch-content", 15);
        raw = inner.data();
        ASSERT_EQ(a.usedBytes(), before + 15);
    }
    // The window closed: cursor restored, span poisoned.
    ASSERT_EQ(a.usedBytes(), before);
    ASSERT_EQ(raw[0], static_cast<char>(gl::kArenaPoisonByte));
    ASSERT_EQ(raw[14], static_cast<char>(gl::kArenaPoisonByte));
}

TEST(hot_string, keep_below_the_mark_survives_the_scope) {
    gl::GlobalMemoryManager m;
    m.init(kHotStrTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    // Executor-lifetime string born BEFORE the per-call window.
    const gl::ScratchString keep = gl::ScratchString::copyFrom(a, "main", 4);
    {
        gl::ScratchScope scope(a);
        gl::ScratchString::copyFrom(a, "transient", 9);
    }
    // The kept view is below the mark — alive and intact after the window.
    ASSERT_EQ(keep.size(), 4);
    ASSERT_EQ(keep.toStdString(), std::string("main"));
    // The next window reuses the freed space without touching the keep.
    {
        gl::ScratchScope scope(a);
        const gl::ScratchString next = gl::ScratchString::copyFrom(a, "again!", 6);
        ASSERT_EQ(next.toStdString(), std::string("again!"));
        ASSERT_EQ(keep.toStdString(), std::string("main"));
    }
}

TEST(hot_string, nested_scopes_close_in_stack_order) {
    gl::GlobalMemoryManager m;
    m.init(kHotStrTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    const int64_t base = a.usedBytes();
    {
        gl::ScratchScope outer(a);
        const gl::ScratchString o = gl::ScratchString::copyFrom(a, "outer", 5);
        const int64_t afterOuterAlloc = a.usedBytes();
        {
            gl::ScratchScope inner(a);
            gl::ScratchString::copyFrom(a, "inner-scratch", 13);
        }
        // Inner closed: back to the outer's position; outer's view alive.
        ASSERT_EQ(a.usedBytes(), afterOuterAlloc);
        ASSERT_EQ(o.toStdString(), std::string("outer"));
    }
    ASSERT_EQ(a.usedBytes(), base);
}
