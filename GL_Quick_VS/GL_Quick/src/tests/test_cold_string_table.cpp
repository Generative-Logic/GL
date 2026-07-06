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
/// @brief Unit tests for `ColdStringTable` (string statification — the cold
///        per-interner storage, arena-backed).
///
/// @details
/// Covers find-or-mint id stability, non-minting lookup, view/decode round
/// trips, the no-straddle within-page interning (page tails padded so every
/// string reads back as one contiguous `view`) exercised by page-crossing and
/// full-page strings, find-index growth across many mints, the canonical
/// lengths+bytes dump and its `bulkLoad` inverse with exact location
/// reproduction, the cross-arena `copyFrom` clone path, the `resetToFresh`
/// lifecycle, and dirty-flag discipline (content mutations escalate; lookups
/// and index rebuilds never do).

#include "test_harness.hpp"

#include "../memory_infra/cold_string_table.hpp"

#include <string>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block.
    const gl::StaticMemoryConfig kColdTestCfg{ 1 << 20, 1 << 18 };

    // 4 MiB pool / 256 KiB block / 256-byte page: dirCap = 64, so the locations
    // column (8-byte) two-levels past 2049 keys and the byte pool past 16 KiB —
    // the two-level page-directory spill, inside a ColdStringTable.
    const gl::StaticMemoryConfig kStrSpillCfg{ 1 << 22, 1 << 18, 256 };

    std::string numbered(int i) {
        return "(in2[v" + std::to_string(i) + ",s,N])";
    }
}

TEST(cold_string_table, intern_lookup_view_round_trip) {
    gl::GlobalMemoryManager g;
    g.init(kColdTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdStringTable t(&lb, &dirty);
    ASSERT_TRUE(t.empty());

    const std::string a = "(=[a,b])";
    const std::string b = "(in[(s[a]),N])";
    const int32_t idA = t.intern(gl::StrSpan(a));
    const int32_t idB = t.intern(gl::StrSpan(b));
    ASSERT_EQ(idA, 1);
    ASSERT_EQ(idB, 2);
    ASSERT_EQ(t.count(), 2);

    // Find-or-mint is idempotent.
    ASSERT_EQ(t.intern(gl::StrSpan(a)), idA);
    ASSERT_EQ(t.count(), 2);

    // Non-minting probe: hit and miss.
    ASSERT_EQ(t.lookup(gl::StrSpan(b)), idB);
    const std::string absent = "(never[seen])";
    ASSERT_EQ(t.lookup(gl::StrSpan(absent)), 0);

    // Views and boundary decode.
    ASSERT_EQ(t.view(idA).toStdString(), a);
    ASSERT_EQ(t.decodeString(idB), b);
    // Content mutation escalated the shared dirty flag.
    ASSERT_TRUE(dirty != gl::DirtyState::Clean);

    // The empty string is a regular value (LogicalEntity fields are often
    // empty): find-or-mint id, hit on lookup, empty view, "" decode.
    const std::string empty;
    ASSERT_EQ(t.lookup(gl::StrSpan(empty)), 0);
    const int32_t idEmpty = t.intern(gl::StrSpan(empty));
    ASSERT_EQ(idEmpty, 3);
    ASSERT_EQ(t.intern(gl::StrSpan(empty)), idEmpty);
    ASSERT_EQ(t.lookup(gl::StrSpan(empty)), idEmpty);
    ASSERT_TRUE(t.view(idEmpty).empty());
    ASSERT_EQ(t.decodeString(idEmpty), empty);
}

TEST(cold_string_table, lookup_never_escalates_dirty) {
    gl::GlobalMemoryManager g;
    g.init(kColdTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdStringTable t(&lb, &dirty);
    const std::string a = "(=[a,b])";
    t.intern(gl::StrSpan(a));
    dirty = gl::DirtyState::Clean;      // the dump/reload reset points
    ASSERT_EQ(t.lookup(gl::StrSpan(a)), 1);
    t.view(1);
    t.rebuildFindIndex();               // derived maintenance
    ASSERT_TRUE(dirty == gl::DirtyState::Clean);
}

TEST(cold_string_table, many_strings_cross_pages) {
    gl::GlobalMemoryManager g;
    g.init(kColdTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdStringTable t(&lb, &dirty);
    // Every cold string fits one page (no straddle): a near-page string fills
    // most of a page, so the next one cannot share its remainder and is padded
    // onto a fresh page. view() therefore always returns a CONTIGUOUS span.
    const int32_t pageBytes = g.pageBytes();
    const std::string nearA(static_cast<size_t>(pageBytes - 8), 'x');
    const int32_t idA = t.intern(gl::StrSpan(nearA));
    ASSERT_EQ(t.view(idA).toStdString(), nearA);       // contiguous, single page
    ASSERT_EQ(t.intern(gl::StrSpan(nearA)), idA);      // dedup (hashCold==hashSpan)
    ASSERT_EQ(t.lookup(gl::StrSpan(nearA)), idA);
    // A second near-page string of equal length, different content -> distinct
    // id, padded onto a fresh page, still contiguous.
    const std::string nearB(static_cast<size_t>(pageBytes - 8), 'z');
    const int32_t idB = t.intern(gl::StrSpan(nearB));
    ASSERT_NE(idB, idA);
    ASSERT_EQ(t.view(idB).toStdString(), nearB);
    ASSERT_EQ(t.decodeString(idB), nearB);
    // The exactly-one-page boundary case still fits (n == elemsPerPage).
    const std::string full(static_cast<size_t>(pageBytes), 'w');
    const int32_t idFull = t.intern(gl::StrSpan(full));
    ASSERT_EQ(t.view(idFull).toStdString(), full);

    // Many strings: index growth + id stability under rehash, every view
    // contiguous across the padded pool.
    std::vector<int32_t> ids;
    for (int i = 0; i < 300; ++i) {
        ids.push_back(t.intern(gl::StrSpan(numbered(i))));
    }
    for (int i = 0; i < 300; ++i) {
        ASSERT_EQ(t.lookup(gl::StrSpan(numbered(i))),
                  ids[static_cast<size_t>(i)]);
        ASSERT_EQ(t.view(ids[static_cast<size_t>(i)]).toStdString(),
                  numbered(i));
    }
}

TEST(cold_string_table, canonical_dump_and_bulk_load_reproduce_locations) {
    gl::GlobalMemoryManager g;
    g.init(kColdTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdStringTable t(&lb, &dirty);
    // Mix of sizes, including a multi-KB string and a near-page string that
    // forces page-tail padding (no straddle), plus an empty row.
    std::vector<std::string> originals;
    originals.push_back("(=[a,b])");
    originals.push_back(std::string(5000, 'y'));
    originals.push_back(std::string(
        static_cast<size_t>(g.pageBytes() - 4), 'q'));  // near page -> pads next
    originals.push_back(std::string());   // empty row in the image
    originals.push_back("(in3[i0,i1,v1,+])");
    for (int i = 0; i < 40; ++i) originals.push_back(numbered(i));
    for (const std::string& s : originals) t.intern(gl::StrSpan(s));

    // Canonical content image: lengths + concatenated bytes, id order.
    std::vector<char> lengthBytes;
    std::vector<char> contentBytes;
    t.appendLengthBytes(lengthBytes, 0);
    t.appendContentBytes(contentBytes, 0);
    ASSERT_EQ(static_cast<int64_t>(lengthBytes.size()),
              static_cast<int64_t>(t.count()) * 4);
    ASSERT_EQ(static_cast<int64_t>(contentBytes.size()),
              t.contentBytesFrom(0));

    // Reload into a fresh table on a fresh arena: identical ids, views,
    // lookups — and identical content (the same logical strings).
    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    gl::ColdStringTable t2(&lb2, &dirty2);
    t2.bulkLoad(reinterpret_cast<const int32_t*>(lengthBytes.data()),
                contentBytes.data(),
                static_cast<int64_t>(contentBytes.size()),
                t.count());
    ASSERT_EQ(t2.count(), t.count());
    for (int32_t id = 1; id <= t.count(); ++id) {
        // Every string is single-page, so view() (contiguous) reproduces too;
        // decodeString is the general check.
        ASSERT_EQ(t.decodeString(id), t2.decodeString(id));
    }
    for (const std::string& s : originals) {
        ASSERT_EQ(t2.lookup(gl::StrSpan(s)), t.lookup(gl::StrSpan(s)));
    }
    // A second dump of the reloaded table is byte-identical — the image is a
    // pure function of logical content.
    std::vector<char> lengthBytes2;
    std::vector<char> contentBytes2;
    t2.appendLengthBytes(lengthBytes2, 0);
    t2.appendContentBytes(contentBytes2, 0);
    ASSERT_TRUE(lengthBytes2 == lengthBytes);
    ASSERT_TRUE(contentBytes2 == contentBytes);
}

TEST(cold_string_table, copy_from_reproduces_ids) {
    gl::GlobalMemoryManager g;
    g.init(kColdTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdStringTable t(&lb, &dirty);
    for (int i = 0; i < 25; ++i) t.intern(gl::StrSpan(numbered(i)));

    gl::LbArena lbClone(&g);
    gl::DirtyState dirtyClone = gl::DirtyState::Clean;
    gl::ColdStringTable clone(&lbClone, &dirtyClone);
    clone.copyFrom(t);
    ASSERT_EQ(clone.count(), t.count());
    for (int i = 0; i < 25; ++i) {
        const std::string s = numbered(i);
        ASSERT_EQ(clone.lookup(gl::StrSpan(s)), t.lookup(gl::StrSpan(s)));
        ASSERT_EQ(clone.decodeString(clone.lookup(gl::StrSpan(s))), s);
    }
    // The clone mints independently afterwards.
    const std::string fresh = "(fresh[x])";
    const int32_t id = clone.intern(gl::StrSpan(fresh));
    ASSERT_EQ(id, clone.count());
    ASSERT_EQ(t.lookup(gl::StrSpan(fresh)), 0);
}

TEST(cold_string_table, reset_to_fresh_clears_and_reuses) {
    gl::GlobalMemoryManager g;
    g.init(kColdTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdStringTable t(&lb, &dirty);
    t.intern(gl::StrSpan(std::string(2000, 'a')));
    t.intern(gl::StrSpan(std::string(2000, 'b')));
    ASSERT_EQ(t.count(), 2);

    t.resetToFresh();
    ASSERT_EQ(t.count(), 0);
    ASSERT_TRUE(t.empty());
    // Every id is invalid; the table re-mints from 1. The emptied string
    // bytes become arena holes the copying compaction reclaims.
    const std::string again = "(=[a,b])";
    ASSERT_EQ(t.lookup(gl::StrSpan(again)), 0);
    ASSERT_EQ(t.intern(gl::StrSpan(again)), 1);
    ASSERT_EQ(t.decodeString(1), again);
}

TEST(cold_string_table, two_level_spill) {
    // Intern enough distinct short strings at a 256-byte page that the table's
    // backing PagedVector columns (locations + byte pool) AND its PagedHashIndex
    // bucket array go two-level. The table stays correct across the spill.
    gl::GlobalMemoryManager g;
    g.init(kStrSpillCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdStringTable t(&lb, &dirty);
    const int32_t N = 3000;                        // > 2049 locations -> two-level
    std::vector<std::string> strs;
    for (int32_t i = 0; i < N; ++i) strs.push_back(numbered(i));   // distinct, short
    for (int32_t i = 0; i < N; ++i)
        ASSERT_EQ(t.intern(gl::StrSpan(strs[static_cast<size_t>(i)])), i + 1);
    ASSERT_EQ(t.count(), N);
    ASSERT_TRUE(g.twoLevelPromotions() > 0);        // a column / the index spilled
    for (int32_t i = 0; i < N; ++i) {
        ASSERT_EQ(t.lookup(gl::StrSpan(strs[static_cast<size_t>(i)])), i + 1);
        ASSERT_EQ(t.decodeString(i + 1), strs[static_cast<size_t>(i)]);
        ASSERT_EQ(t.view(i + 1).toStdString(), strs[static_cast<size_t>(i)]);
    }
    ASSERT_EQ(t.lookup(gl::StrSpan(std::string("(absent[zzz])"))), 0);

    // Deload byte-identity across the spill: dump lengths + content, reload on a
    // fresh arena, re-dump -> byte-identical (the canonical-bytes guarantee).
    std::vector<char> lengthBytes, contentBytes;
    t.appendLengthBytes(lengthBytes, 0);
    t.appendContentBytes(contentBytes, 0);
    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    gl::ColdStringTable u(&lb2, &dirty2);
    u.bulkLoad(reinterpret_cast<const int32_t*>(lengthBytes.data()),
               contentBytes.data(),
               static_cast<int64_t>(contentBytes.size()), t.count());
    ASSERT_EQ(u.count(), t.count());
    for (int32_t i = 0; i < N; ++i)
        ASSERT_EQ(u.lookup(gl::StrSpan(strs[static_cast<size_t>(i)])), i + 1);
    std::vector<char> lengthBytes2, contentBytes2;
    u.appendLengthBytes(lengthBytes2, 0);
    u.appendContentBytes(contentBytes2, 0);
    ASSERT_TRUE(lengthBytes2 == lengthBytes);
    ASSERT_TRUE(contentBytes2 == contentBytes);
}
