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
/// @brief Unit tests for the sealed-page handoff (string statification —
///        staging-record strings).
///
/// @details
/// Covers the `SealedPageSet` three-state lifecycle (Filling → Sealed →
/// Freed), exact-length bump allocation across blocks, the cold-side
/// accounting of its block traffic, poison-on-free, the never-used
/// lifecycle, and `SealedString` construction, accessors, empty-string
/// contract, and read validity across the seal (the pool-join crossing the
/// views must survive). Also covers `SealedSpan<T>` — the typed-array
/// analogue — its memcpy round-trip, the align-up of a wide-element run over
/// byte-granular page storage, range-for, and the empty-span contract.
/// The record-chain surface (`appendRecord` / `recordCount` /
/// `forEachRecord` / `SealedRecordCursor`) is covered by the
/// `record_chain_*` / `record_cursor_*` / `record_payload_*` tests:
/// append==iteration order, string-alloc interleaving, the incremental
/// cursor drain across appends and the seal, the drop-unread `freePages`
/// reclaim, and payload address stability.

#include "test_harness.hpp"

#include "../memory_infra/sealed_pages.hpp"
#include "../memory_infra/str_ops.hpp"

#include <cstdint>
#include <cstring>
#include <string>

namespace {
    // 1 MiB pool / 256 KiB block.
    const gl::StaticMemoryConfig kSealedTestCfg{ 1 << 20, 1 << 18 };
}

TEST(sealed_pages, lifecycle_fill_seal_free) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    ASSERT_TRUE(set.filling());
    char* p = set.alloc(16);
    std::memset(p, 0x66, 16);
    // One carved block; the small page table stays inline (no pool block).
    ASSERT_EQ(set.blocksHeld(), 1);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(1));
    ASSERT_EQ(set.bytesAllocated(), static_cast<int64_t>(16));
    set.seal();
    ASSERT_TRUE(set.sealed());
    // Sealed content is readable (the consumer-side window).
    ASSERT_EQ(p[0], static_cast<char>(0x66));
    set.freePages();
    ASSERT_TRUE(set.freed());
    ASSERT_EQ(set.blocksHeld(), 0);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
    // Poison landed before the release.
    ASSERT_EQ(p[0], static_cast<char>(gl::kArenaPoisonByte));
    ASSERT_EQ(p[15], static_cast<char>(gl::kArenaPoisonByte));
}

TEST(sealed_pages, never_used_set_destructs_cleanly) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    {
        gl::SealedPageSet set;
        set.bind(&m);
        // No alloc, no seal — an empty task output is a defined result.
    }
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
}

TEST(sealed_pages, allocations_span_pages_under_demand) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    const int32_t pb = m.pageBytes();
    char* a = set.alloc(pb);            // fills page 0 exactly
    char* b = set.alloc(64);            // opens a fresh page
    // Two pages carved from one block — no second block under this demand.
    ASSERT_EQ(set.blocksHeld(), 1);
    const std::ptrdiff_t gap = a < b ? b - a : a - b;
    ASSERT_GE(gap, static_cast<std::ptrdiff_t>(pb));
    ASSERT_EQ(set.bytesAllocated(), static_cast<int64_t>(pb) + 64);
    set.seal();
    set.freePages();
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
}

TEST(sealed_pages, sealed_string_copy_and_accessors) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    const std::string src = "(in3[i0,i1,v1,+])";
    const gl::SealedString s = gl::SealedString::copyFrom(
        set, src.data(), static_cast<int32_t>(src.size()));
    ASSERT_FALSE(s.empty());
    ASSERT_EQ(s.size(), static_cast<int32_t>(src.size()));
    ASSERT_EQ(s[0], '(');
    ASSERT_EQ(std::memcmp(s.data(), src.data(), src.size()), 0);
    // Views stay valid across the seal — the pool-join crossing.
    set.seal();
    ASSERT_EQ(s.toStdString(), src);
    // The StrSpan conversion is the zero-copy twin of toStdString().
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(s), gl::StrSpan(src)));
    set.freePages();
}

TEST(sealed_pages, sealed_string_empty_contract) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    const gl::SealedString def;
    ASSERT_TRUE(def.empty());
    ASSERT_EQ(def.size(), 0);
    const gl::SealedString viaCopy =
        gl::SealedString::copyFrom(set, "x", 0);
    ASSERT_TRUE(viaCopy.empty());
    ASSERT_EQ(set.bytesAllocated(), static_cast<int64_t>(0));
    ASSERT_EQ(set.blocksHeld(), 0);
    // Empty views have no owner coupling: valid even after free.
    set.seal();
    set.freePages();
    ASSERT_EQ(def.toStdString(), std::string());
    ASSERT_EQ(viaCopy.toStdString(), std::string());
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(def), gl::StrSpan()));
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(viaCopy), gl::StrSpan()));
}

TEST(sealed_pages, many_strings_pack_tightly) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    gl::SealedString views[8];
    std::string sources[8];
    for (int i = 0; i < 8; ++i) {
        sources[i] = "(=[v" + std::to_string(i) + ",zero])";
        views[i] = gl::SealedString::copyFrom(
            set, sources[i].data(),
            static_cast<int32_t>(sources[i].size()));
    }
    set.seal();
    for (int i = 0; i < 8; ++i) {
        ASSERT_EQ(views[i].toStdString(), sources[i]);
    }
    // All in one block — exact-length packing.
    ASSERT_EQ(set.blocksHeld(), 1);
    set.freePages();
}

TEST(sealed_pages, sealed_span_int_round_trip) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    const int src[5] = { 3, 1, 4, 1, 5 };
    const gl::SealedSpan<int> span =
        gl::SealedSpan<int>::copyFrom(set, src, 5);
    ASSERT_FALSE(span.empty());
    ASSERT_EQ(span.size(), 5);
    ASSERT_EQ(span[0], 3);
    ASSERT_EQ(span[4], 5);
    // The run base is aligned for int access despite byte-granular storage.
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(span.data())
                  % alignof(int),
              static_cast<std::uintptr_t>(0));
    set.seal();
    int sum = 0;
    for (int v : span) sum += v;     // range-for over the sealed run
    ASSERT_EQ(sum, 14);
    set.freePages();
}

TEST(sealed_pages, sealed_span_struct_run_aligned) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    // A 3-byte string first nudges the byte cursor off any wide alignment;
    // copyFrom must still hand back storage aligned for SealedString.
    gl::SealedString::copyFrom(set, "abc", 3);
    const std::string a = "(in2[i0,v1,s])";
    const std::string b = "(=[v0,zero])";
    gl::SealedString strs[2];
    strs[0] = gl::SealedString::copyFrom(
        set, a.data(), static_cast<int32_t>(a.size()));
    strs[1] = gl::SealedString::copyFrom(
        set, b.data(), static_cast<int32_t>(b.size()));
    const gl::SealedSpan<gl::SealedString> span =
        gl::SealedSpan<gl::SealedString>::copyFrom(set, strs, 2);
    ASSERT_EQ(span.size(), 2);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(span.data())
                  % alignof(gl::SealedString),
              static_cast<std::uintptr_t>(0));
    set.seal();
    ASSERT_EQ(span[0].toStdString(), a);
    ASSERT_EQ(span[1].toStdString(), b);
    set.freePages();
}

namespace {
    /// Small POD record for the record-chain tests.
    struct ChainP {
        int32_t a;
        int64_t b;
    };
}

TEST(sealed_pages, record_chain_append_foreach_order_and_count) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    {
        gl::SealedPageSet set;
        set.bind(&m);
        ASSERT_EQ(set.recordCount(), 0);
        const int32_t kN = 300;
        for (int32_t i = 0; i < kN; ++i) {
            const ChainP p{ i, static_cast<int64_t>(i) * 7 - 3 };
            const ChainP& stored = set.appendRecord(p);
            ASSERT_EQ(stored.a, i);
            ASSERT_EQ(stored.b, static_cast<int64_t>(i) * 7 - 3);
        }
        ASSERT_EQ(set.recordCount(), kN);
        int32_t seen = 0;
        set.forEachRecord<ChainP>([&](const ChainP& r) {
            ASSERT_EQ(r.a, seen);
            ASSERT_EQ(r.b, static_cast<int64_t>(seen) * 7 - 3);
            ++seen;
        });
        ASSERT_EQ(seen, kN);
        // Sealed-legal read: the walk yields the same append sequence.
        set.seal();
        ASSERT_EQ(set.recordCount(), kN);
        seen = 0;
        set.forEachRecord<ChainP>([&](const ChainP& r) {
            ASSERT_EQ(r.a, seen);
            ++seen;
        });
        ASSERT_EQ(seen, kN);
        set.freePages();
    }
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
}

TEST(sealed_pages, record_chain_interleaved_with_string_allocs) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    gl::SealedString views[6];
    std::string sources[6];
    for (int i = 0; i < 6; ++i) {
        sources[i] = "(=[v" + std::to_string(i) + ",zero])";
        // Alternate: one record, then one string, on the SAME set.
        set.appendRecord(ChainP{ i, i * 11LL });
        views[i] = gl::SealedString::copyFrom(
            set, sources[i].data(),
            static_cast<int32_t>(sources[i].size()));
    }
    ASSERT_EQ(set.recordCount(), 6);
    int i = 0;
    set.forEachRecord<ChainP>([&](const ChainP& r) {
        ASSERT_EQ(r.a, i);
        ASSERT_EQ(r.b, i * 11LL);
        ++i;
    });
    ASSERT_EQ(i, 6);
    // The interleaved strings read back correctly.
    for (int k = 0; k < 6; ++k) {
        ASSERT_EQ(views[k].toStdString(), sources[k]);
    }
    set.seal();
    set.freePages();
}

TEST(sealed_pages, record_cursor_incremental_drain) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    gl::SealedRecordCursor<ChainP> cursor(set);
    // Empty chain: caught up immediately (a defined result).
    ASSERT_TRUE(cursor.next() == nullptr);
    for (int32_t i = 0; i < 3; ++i)
        set.appendRecord(ChainP{ i, i * 2LL });
    for (int32_t i = 0; i < 3; ++i) {
        const ChainP* r = cursor.next();
        ASSERT_TRUE(r != nullptr);
        ASSERT_EQ(r->a, i);
    }
    ASSERT_TRUE(cursor.next() == nullptr);   // caught up
    // Appending resumes the SAME cursor at exactly the new records.
    for (int32_t i = 3; i < 5; ++i)
        set.appendRecord(ChainP{ i, i * 2LL });
    for (int32_t i = 3; i < 5; ++i) {
        const ChainP* r = cursor.next();
        ASSERT_TRUE(r != nullptr);
        ASSERT_EQ(r->a, i);
        ASSERT_EQ(r->b, i * 2LL);
    }
    ASSERT_TRUE(cursor.next() == nullptr);
    set.seal();
    ASSERT_TRUE(cursor.next() == nullptr);   // still caught up post-seal
    // A fresh cursor on the sealed set sees all 5 from the head.
    gl::SealedRecordCursor<ChainP> fresh(set);
    for (int32_t i = 0; i < 5; ++i) {
        const ChainP* r = fresh.next();
        ASSERT_TRUE(r != nullptr);
        ASSERT_EQ(r->a, i);
    }
    ASSERT_TRUE(fresh.next() == nullptr);
    set.freePages();
    // Assert-path coverage (append after seal, cursor/forEachRecord after
    // free, record-type stride mismatch) is not harness-testable — the
    // asserts abort — so those contracts are pinned by the asserts
    // themselves, per the house convention.
}

TEST(sealed_pages, record_chain_dies_with_free_pages) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    {
        gl::SealedPageSet set;
        set.bind(&m);
        for (int32_t i = 0; i < 40; ++i)
            set.appendRecord(ChainP{ i, -i * 1LL });
        const std::string s = "(in2[i0,v1,s])";
        gl::SealedString::copyFrom(
            set, s.data(), static_cast<int32_t>(s.size()));
        set.seal();
        // The redo-discard shape: records are NEVER read; freePages is the
        // uniform reclaim, and the destructor's blocksHeld()==0 assert IS
        // the check that the chain needed no separate cleanup.
        set.freePages();
        ASSERT_TRUE(set.freed());
        ASSERT_EQ(set.blocksHeld(), 0);
    }
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
}

TEST(sealed_pages, record_payload_address_stable) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    // Capture the stored-payload addresses of the first appends...
    const ChainP* captured[8];
    for (int32_t i = 0; i < 8; ++i)
        captured[i] = &set.appendRecord(ChainP{ i, i * 1000LL });
    // ...then grow the chain well past several pages.
    for (int32_t i = 8; i < 500; ++i)
        set.appendRecord(ChainP{ i, i * 1000LL });
    ASSERT_EQ(set.recordCount(), 500);
    // The captured pointers still read the original record fields
    // (the pointer-permutation-sort prerequisite). Compare the fields, not
    // the raw bytes: ChainP has alignment padding, and a memcmp over
    // uninitialized padding is non-deterministic (it is unspecified whether
    // the temporary's padding is zeroed).
    for (int32_t i = 0; i < 8; ++i) {
        ASSERT_EQ(captured[i]->a, i);
        ASSERT_EQ(captured[i]->b, i * 1000LL);
    }
    set.seal();
    for (int32_t i = 0; i < 8; ++i) {
        ASSERT_EQ(captured[i]->a, i);
        ASSERT_EQ(captured[i]->b, i * 1000LL);
    }
    set.freePages();
}

TEST(sealed_pages, sealed_span_empty_contract) {
    gl::GlobalMemoryManager m;
    m.init(kSealedTestCfg);
    gl::SealedPageSet set;
    set.bind(&m);
    const gl::SealedSpan<int> def;
    ASSERT_TRUE(def.empty());
    ASSERT_EQ(def.size(), 0);
    ASSERT_TRUE(def.begin() == def.end());   // empty range, no iterations
    const int dummy = 7;
    const gl::SealedSpan<int> viaCopy =
        gl::SealedSpan<int>::copyFrom(set, &dummy, 0);
    ASSERT_TRUE(viaCopy.empty());
    // The empty span never touches the set.
    ASSERT_EQ(set.bytesAllocated(), static_cast<int64_t>(0));
    ASSERT_EQ(set.blocksHeld(), 0);
    set.seal();
    set.freePages();
}
